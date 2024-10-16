import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l3955_395572

theorem polynomial_identity_sum (b₁ b₂ b₃ b₄ c₁ c₂ c₃ c₄ : ℝ) :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃) * (x^2 + b₄*x + c₄)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ + b₄*c₄ = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l3955_395572


namespace NUMINAMATH_CALUDE_parallelogram_angle_difference_l3955_395524

theorem parallelogram_angle_difference (a b : ℝ) : 
  a = 65 → -- smaller angle is 65 degrees
  a + b = 180 → -- adjacent angles in a parallelogram are supplementary
  b - a = 50 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_angle_difference_l3955_395524


namespace NUMINAMATH_CALUDE_investment_period_ratio_l3955_395548

/-- Represents the ratio of investments and investment periods for two business partners -/
structure BusinessPartnership where
  investment_ratio : ℚ
  period_ratio : ℚ
  b_profit : ℕ
  total_profit : ℕ

/-- Theorem stating that given the conditions of the business partnership, 
    the ratio of investment periods is 2:1 -/
theorem investment_period_ratio 
  (p : BusinessPartnership) 
  (h1 : p.investment_ratio = 3)
  (h2 : p.b_profit = 4000)
  (h3 : p.total_profit = 28000) :
  p.period_ratio = 2 := by
  sorry

#check investment_period_ratio

end NUMINAMATH_CALUDE_investment_period_ratio_l3955_395548


namespace NUMINAMATH_CALUDE_max_teams_in_tournament_l3955_395570

/-- The number of players in each team -/
def players_per_team : ℕ := 3

/-- The maximum number of games that can be played in the tournament -/
def max_games : ℕ := 200

/-- The number of games played between two teams -/
def games_between_teams : ℕ := players_per_team * players_per_team

/-- The function to calculate the total number of games for a given number of teams -/
def total_games (n : ℕ) : ℕ := games_between_teams * (n * (n - 1) / 2)

/-- The theorem stating the maximum number of teams that can participate -/
theorem max_teams_in_tournament : 
  ∃ (n : ℕ), n > 0 ∧ total_games n ≤ max_games ∧ ∀ m : ℕ, m > n → total_games m > max_games :=
by sorry

end NUMINAMATH_CALUDE_max_teams_in_tournament_l3955_395570


namespace NUMINAMATH_CALUDE_smartphone_demand_l3955_395527

theorem smartphone_demand (d p : ℝ) (k : ℝ) :
  (d * p = k) →  -- Demand is inversely proportional to price
  (30 * 600 = k) →  -- 30 customers purchase at $600
  (20 * 900 = k) →  -- 20 customers purchase at $900
  True :=
by
  sorry

end NUMINAMATH_CALUDE_smartphone_demand_l3955_395527


namespace NUMINAMATH_CALUDE_range_of_k_line_equation_when_OB_2OA_l3955_395510

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 6)^2 + y^2 = 20

-- Define the line l
def line_l (k x y : ℝ) : Prop := y = k * x

-- Define the condition that line l intersects circle C at two distinct points
def intersects_at_two_points (k : ℝ) : Prop := 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the condition OB = 2OA
def OB_equals_2OA (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
    x₂ = 2 * x₁ ∧ y₂ = 2 * y₁

-- Theorem for the range of k
theorem range_of_k (k : ℝ) : intersects_at_two_points k → -Real.sqrt 5 / 2 < k ∧ k < Real.sqrt 5 / 2 := 
  sorry

-- Theorem for the equation of line l when OB = 2OA
theorem line_equation_when_OB_2OA (k : ℝ) : OB_equals_2OA k → k = 1 ∨ k = -1 :=
  sorry

end NUMINAMATH_CALUDE_range_of_k_line_equation_when_OB_2OA_l3955_395510


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_l3955_395533

theorem smallest_n_for_candy (n : ℕ) : 
  (∀ m : ℕ, m > 0 → (25 * m) % 10 = 0 ∧ (25 * m) % 18 = 0 ∧ (25 * m) % 20 = 0 → m ≥ n) →
  (25 * n) % 10 = 0 ∧ (25 * n) % 18 = 0 ∧ (25 * n) % 20 = 0 →
  n = 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_l3955_395533


namespace NUMINAMATH_CALUDE_enrollment_ways_count_l3955_395514

/-- The number of elective courses -/
def num_courses : ℕ := 4

/-- The number of students -/
def num_students : ℕ := 3

/-- The number of courses each student must choose -/
def courses_per_student : ℕ := 2

/-- The number of different ways each course can have students enrolled -/
def num_enrollment_ways : ℕ := 114

theorem enrollment_ways_count :
  (num_courses = 4) →
  (num_students = 3) →
  (courses_per_student = 2) →
  (num_enrollment_ways = 114) := by
  sorry

end NUMINAMATH_CALUDE_enrollment_ways_count_l3955_395514


namespace NUMINAMATH_CALUDE_odd_function_properties_and_inequality_l3955_395558

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x - a * 2^(-x)

theorem odd_function_properties_and_inequality (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) →
  (a = 1 ∧ ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (∀ t : ℝ, (∀ x : ℝ, f a (x - t) + f a (x^2 - t^2) ≥ 0) → t = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_properties_and_inequality_l3955_395558


namespace NUMINAMATH_CALUDE_spears_from_log_l3955_395513

/-- The number of spears Marcy can make from a sapling -/
def spears_from_sapling : ℕ := 3

/-- The total number of spears Marcy can make from 6 saplings and a log -/
def total_spears : ℕ := 27

/-- The number of saplings used -/
def num_saplings : ℕ := 6

/-- Theorem: Marcy can make 9 spears from a single log -/
theorem spears_from_log : 
  ∃ (L : ℕ), L = total_spears - (num_saplings * spears_from_sapling) ∧ L = 9 :=
by sorry

end NUMINAMATH_CALUDE_spears_from_log_l3955_395513


namespace NUMINAMATH_CALUDE_no_real_roots_l3955_395537

theorem no_real_roots : ∀ x : ℝ, 4 * x^2 - 5 * x + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3955_395537


namespace NUMINAMATH_CALUDE_square_circle_relation_l3955_395545

theorem square_circle_relation (s : ℝ) (h : s > 0) :
  (4 * s = π * (s / Real.sqrt 2)^2) → s = 8 / π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_relation_l3955_395545


namespace NUMINAMATH_CALUDE_solution_correctness_l3955_395541

/-- The set of solutions for x^2 - y^2 = 105 where x and y are natural numbers -/
def SolutionsA : Set (ℕ × ℕ) :=
  {(53, 52), (19, 16), (13, 8), (11, 4)}

/-- The set of solutions for 2x^2 + 5xy - 12y^2 = 28 where x and y are natural numbers -/
def SolutionsB : Set (ℕ × ℕ) :=
  {(8, 5)}

theorem solution_correctness :
  (∀ (x y : ℕ), x^2 - y^2 = 105 ↔ (x, y) ∈ SolutionsA) ∧
  (∀ (x y : ℕ), 2*x^2 + 5*x*y - 12*y^2 = 28 ↔ (x, y) ∈ SolutionsB) := by
  sorry

end NUMINAMATH_CALUDE_solution_correctness_l3955_395541


namespace NUMINAMATH_CALUDE_circle_radius_squared_l3955_395552

theorem circle_radius_squared (r : ℝ) 
  (AB CD : ℝ) (angle_APD : ℝ) (BP : ℝ) : 
  AB = 10 → 
  CD = 7 → 
  angle_APD = 60 * π / 180 → 
  BP = 8 → 
  r^2 = 73 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_squared_l3955_395552


namespace NUMINAMATH_CALUDE_functional_equation_characterization_l3955_395518

theorem functional_equation_characterization
  (a : ℤ) (ha : a ≠ 0)
  (f g : ℚ → ℚ)
  (h : ∀ x y : ℚ, f (x + g y) = g x + f y + a * y) :
  (∃ n : ℤ, n ≠ 0 ∧ n ≠ 1 ∧ a = n^2 - n) ∧
  (∃ n : ℤ, ∃ v : ℚ, (n ≠ 0 ∧ n ≠ 1) ∧
    ((∀ x : ℚ, f x = n * x + v ∧ g x = n * x) ∨
     (∀ x : ℚ, f x = (1 - n) * x + v ∧ g x = (1 - n) * x))) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_characterization_l3955_395518


namespace NUMINAMATH_CALUDE_lines_intersect_implies_planes_intersect_l3955_395544

-- Define the space
variable (S : Type*) [NormedAddCommGroup S] [InnerProductSpace ℝ S] [CompleteSpace S]

-- Define lines and planes
def Line (S : Type*) [NormedAddCommGroup S] := Set S
def Plane (S : Type*) [NormedAddCommGroup S] := Set S

-- Define the subset relation
def IsSubset {S : Type*} (A B : Set S) := A ⊆ B

-- Define intersection for lines and planes
def Intersect {S : Type*} (A B : Set S) := ∃ x, x ∈ A ∧ x ∈ B

-- Theorem statement
theorem lines_intersect_implies_planes_intersect
  (m n : Line S) (α β : Plane S)
  (hm : m ≠ n) (hα : α ≠ β)
  (hmα : IsSubset m α) (hnβ : IsSubset n β)
  (hmn : Intersect m n) :
  Intersect α β :=
sorry

end NUMINAMATH_CALUDE_lines_intersect_implies_planes_intersect_l3955_395544


namespace NUMINAMATH_CALUDE_mat_weaving_in_12_days_l3955_395590

/-- Represents a group of mat weavers -/
structure WeaverGroup where
  weavers : ℕ
  mats : ℕ
  days : ℕ

/-- Calculates the number of mats a group can weave in a given number of days -/
def mats_in_days (group : WeaverGroup) (target_days : ℕ) : ℕ :=
  (group.mats * target_days) / group.days

/-- Group A of mat weavers -/
def group_A : WeaverGroup :=
  { weavers := 4, mats := 4, days := 4 }

/-- Group B of mat weavers -/
def group_B : WeaverGroup :=
  { weavers := 6, mats := 9, days := 3 }

/-- Group C of mat weavers -/
def group_C : WeaverGroup :=
  { weavers := 8, mats := 16, days := 4 }

theorem mat_weaving_in_12_days :
  mats_in_days group_A 12 = 12 ∧
  mats_in_days group_B 12 = 36 ∧
  mats_in_days group_C 12 = 48 := by
  sorry

end NUMINAMATH_CALUDE_mat_weaving_in_12_days_l3955_395590


namespace NUMINAMATH_CALUDE_prob_no_adjacent_fir_value_l3955_395597

/-- The number of pine trees -/
def num_pine : ℕ := 5

/-- The number of cedar trees -/
def num_cedar : ℕ := 6

/-- The number of fir trees -/
def num_fir : ℕ := 7

/-- The total number of trees -/
def total_trees : ℕ := num_pine + num_cedar + num_fir

/-- The probability that no two fir trees are adjacent when planted in a random order -/
def prob_no_adjacent_fir : ℚ :=
  (Nat.choose (num_pine + num_cedar + 1) num_fir) / (Nat.choose total_trees num_fir)

theorem prob_no_adjacent_fir_value : prob_no_adjacent_fir = 1 / 40 := by
  sorry

end NUMINAMATH_CALUDE_prob_no_adjacent_fir_value_l3955_395597


namespace NUMINAMATH_CALUDE_green_ball_probability_l3955_395550

-- Define the total number of balls
def total_balls : ℕ := 20

-- Define the number of red balls
def red_balls : ℕ := 5

-- Define the number of yellow balls
def yellow_balls : ℕ := 5

-- Define the number of green balls
def green_balls : ℕ := 10

-- Define the probability of drawing a green ball given it's not red
def prob_green_given_not_red : ℚ := green_balls / (total_balls - red_balls)

-- Theorem statement
theorem green_ball_probability :
  prob_green_given_not_red = 2/3 :=
sorry

end NUMINAMATH_CALUDE_green_ball_probability_l3955_395550


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3955_395542

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 20 → p.Prime → ¬(p ∣ n)

theorem smallest_composite_no_small_factors : 
  (is_composite 667 ∧ has_no_small_prime_factors 667) ∧ 
  (∀ m : ℕ, m < 667 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3955_395542


namespace NUMINAMATH_CALUDE_thirtieth_term_value_l3955_395551

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 30th term of the specific arithmetic sequence -/
def thirtieth_term : ℝ := arithmetic_sequence 3 4 30

theorem thirtieth_term_value : thirtieth_term = 119 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_value_l3955_395551


namespace NUMINAMATH_CALUDE_dark_tiles_fraction_is_three_fourths_l3955_395560

/-- Represents a square tiling pattern -/
structure TilingPattern where
  size : ℕ
  dark_tiles_per_corner : ℕ
  is_symmetrical : Bool

/-- Calculates the fraction of dark tiles in a tiling pattern -/
def fraction_of_dark_tiles (pattern : TilingPattern) : ℚ :=
  if pattern.is_symmetrical
  then (4 * pattern.dark_tiles_per_corner : ℚ) / (pattern.size * pattern.size : ℚ)
  else 0

/-- Theorem stating that a 4x4 symmetrical pattern with 3 dark tiles per corner 
    has 3/4 of its tiles dark -/
theorem dark_tiles_fraction_is_three_fourths 
  (pattern : TilingPattern) 
  (h1 : pattern.size = 4) 
  (h2 : pattern.dark_tiles_per_corner = 3) 
  (h3 : pattern.is_symmetrical = true) : 
  fraction_of_dark_tiles pattern = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_dark_tiles_fraction_is_three_fourths_l3955_395560


namespace NUMINAMATH_CALUDE_negation_equivalence_l3955_395595

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) ↔ (∃ x : ℝ, x^2 + 2*x + 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3955_395595


namespace NUMINAMATH_CALUDE_rented_cars_at_3600_optimal_rent_max_monthly_revenue_l3955_395592

/-- Represents the rental company's car fleet and pricing model. -/
structure RentalCompany where
  totalCars : ℕ := 100
  initialRent : ℕ := 3000
  rentIncrease : ℕ := 50
  maintenanceCostRented : ℕ := 150
  maintenanceCostUnrented : ℕ := 50

/-- Calculates the number of rented cars given a specific rent. -/
def rentedCars (company : RentalCompany) (rent : ℕ) : ℕ :=
  company.totalCars - (rent - company.initialRent) / company.rentIncrease

/-- Calculates the monthly revenue for the rental company. -/
def monthlyRevenue (company : RentalCompany) (rent : ℕ) : ℕ :=
  let rented := rentedCars company rent
  rented * (rent - company.maintenanceCostRented) -
    (company.totalCars - rented) * company.maintenanceCostUnrented

/-- Theorem stating the correct number of rented cars at 3600 yuan rent. -/
theorem rented_cars_at_3600 (company : RentalCompany) :
    rentedCars company 3600 = 88 := by sorry

/-- Theorem stating the optimal rent that maximizes revenue. -/
theorem optimal_rent (company : RentalCompany) :
    ∃ (optimalRent : ℕ), optimalRent = 4050 ∧
    ∀ (rent : ℕ), monthlyRevenue company rent ≤ monthlyRevenue company optimalRent := by sorry

/-- Theorem stating the maximum monthly revenue. -/
theorem max_monthly_revenue (company : RentalCompany) :
    ∃ (maxRevenue : ℕ), maxRevenue = 307050 ∧
    ∀ (rent : ℕ), monthlyRevenue company rent ≤ maxRevenue := by sorry

end NUMINAMATH_CALUDE_rented_cars_at_3600_optimal_rent_max_monthly_revenue_l3955_395592


namespace NUMINAMATH_CALUDE_total_chips_is_90_l3955_395531

/-- The total number of chips Viviana and Susana have together -/
def total_chips (viviana_vanilla : ℕ) (susana_chocolate : ℕ) : ℕ :=
  let viviana_chocolate := susana_chocolate + 5
  let susana_vanilla := (3 * viviana_vanilla) / 4
  viviana_vanilla + viviana_chocolate + susana_vanilla + susana_chocolate

/-- Theorem stating that the total number of chips is 90 -/
theorem total_chips_is_90 :
  total_chips 20 25 = 90 := by
  sorry

#eval total_chips 20 25

end NUMINAMATH_CALUDE_total_chips_is_90_l3955_395531


namespace NUMINAMATH_CALUDE_place_balls_count_l3955_395507

/-- The number of ways to place six numbered balls into six numbered boxes --/
def place_balls : ℕ :=
  let n : ℕ := 6  -- number of balls and boxes
  let k : ℕ := 2  -- number of balls placed in boxes with the same number
  let choose_two : ℕ := n.choose k
  let derangement_four : ℕ := 8  -- number of valid derangements for remaining 4 balls
  choose_two * derangement_four

/-- Theorem stating that the number of ways to place the balls is 120 --/
theorem place_balls_count : place_balls = 120 := by
  sorry

end NUMINAMATH_CALUDE_place_balls_count_l3955_395507


namespace NUMINAMATH_CALUDE_rainy_days_last_week_l3955_395596

theorem rainy_days_last_week (n : ℤ) : 
  (∃ (R NR : ℕ), 
    R + NR = 7 ∧ 
    n * R + 3 * NR = 20 ∧ 
    3 * NR = n * R + 10) →
  (∃ (R : ℕ), R = 2) :=
sorry

end NUMINAMATH_CALUDE_rainy_days_last_week_l3955_395596


namespace NUMINAMATH_CALUDE_abs_sum_lt_abs_diff_when_product_negative_l3955_395538

theorem abs_sum_lt_abs_diff_when_product_negative (a b : ℝ) : 
  a * b < 0 → |a + b| < |a - b| := by
sorry

end NUMINAMATH_CALUDE_abs_sum_lt_abs_diff_when_product_negative_l3955_395538


namespace NUMINAMATH_CALUDE_prob_first_odd_given_two_odd_one_even_l3955_395562

/-- Represents the outcome of picking a ball -/
inductive BallType
| Odd
| Even

/-- Represents the result of picking 3 balls -/
structure ThreePickResult :=
  (first second third : BallType)

def is_valid_pick (result : ThreePickResult) : Prop :=
  (result.first = BallType.Odd ∧ result.second = BallType.Odd ∧ result.third = BallType.Even) ∨
  (result.first = BallType.Odd ∧ result.second = BallType.Even ∧ result.third = BallType.Odd) ∨
  (result.first = BallType.Even ∧ result.second = BallType.Odd ∧ result.third = BallType.Odd)

def total_balls : ℕ := 100
def odd_balls : ℕ := 50
def even_balls : ℕ := 50

theorem prob_first_odd_given_two_odd_one_even :
  ∀ (sample_space : Set ThreePickResult) (prob : Set ThreePickResult → ℝ),
  (∀ result ∈ sample_space, is_valid_pick result) →
  (∀ A ⊆ sample_space, 0 ≤ prob A ∧ prob A ≤ 1) →
  prob sample_space = 1 →
  prob {result ∈ sample_space | result.first = BallType.Odd} = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_first_odd_given_two_odd_one_even_l3955_395562


namespace NUMINAMATH_CALUDE_expression_evaluation_l3955_395529

theorem expression_evaluation (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  ((((x+2)^2 * (x^2-2*x+4)^2) / (x^3+8)^2)^2) * ((((x-2)^2 * (x^2+2*x+4)^2) / (x^3-8)^2)^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3955_395529


namespace NUMINAMATH_CALUDE_ron_ticket_sales_l3955_395508

/-- Proves that Ron sold 12 student tickets given the problem conditions -/
theorem ron_ticket_sales
  (student_price : ℝ)
  (adult_price : ℝ)
  (total_tickets : ℕ)
  (total_income : ℝ)
  (h1 : student_price = 2)
  (h2 : adult_price = 4.5)
  (h3 : total_tickets = 20)
  (h4 : total_income = 60)
  : ∃ (student_tickets : ℕ) (adult_tickets : ℕ),
    student_tickets + adult_tickets = total_tickets ∧
    student_price * student_tickets + adult_price * adult_tickets = total_income ∧
    student_tickets = 12 :=
by sorry

end NUMINAMATH_CALUDE_ron_ticket_sales_l3955_395508


namespace NUMINAMATH_CALUDE_total_days_on_orbius5_l3955_395567

/-- Definition of the Orbius-5 calendar system -/
structure Orbius5Calendar where
  daysPerYear : Nat := 250
  regularSeasonDays : Nat := 49
  leapSeasonDays : Nat := 51
  regularSeasonsPerYear : Nat := 2
  leapSeasonsPerYear : Nat := 3
  cycleYears : Nat := 10

/-- Definition of the astronaut's visits -/
structure AstronautVisits where
  firstVisitRegularSeasons : Nat := 1
  secondVisitRegularSeasons : Nat := 2
  secondVisitLeapSeasons : Nat := 3
  thirdVisitYears : Nat := 3
  fourthVisitCycles : Nat := 1

/-- Function to calculate total days spent on Orbius-5 -/
def totalDaysOnOrbius5 (calendar : Orbius5Calendar) (visits : AstronautVisits) : Nat :=
  sorry

/-- Theorem stating the total days spent on Orbius-5 -/
theorem total_days_on_orbius5 (calendar : Orbius5Calendar) (visits : AstronautVisits) :
  totalDaysOnOrbius5 calendar visits = 3578 := by
  sorry

end NUMINAMATH_CALUDE_total_days_on_orbius5_l3955_395567


namespace NUMINAMATH_CALUDE_infinitely_many_special_numbers_l3955_395559

/-- A natural number n such that n^2 + 1 has no divisors of the form k^2 + 1 except 1 and itself. -/
def SpecialNumber (n : ℕ) : Prop :=
  ∀ k : ℕ, k^2 + 1 ∣ n^2 + 1 → k^2 + 1 = 1 ∨ k^2 + 1 = n^2 + 1

/-- The set of SpecialNumbers is infinite. -/
theorem infinitely_many_special_numbers : Set.Infinite {n : ℕ | SpecialNumber n} := by
  sorry


end NUMINAMATH_CALUDE_infinitely_many_special_numbers_l3955_395559


namespace NUMINAMATH_CALUDE_scientific_notation_of_104000_l3955_395535

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation := sorry

/-- The given number from the problem -/
def givenNumber : ℝ := 104000

theorem scientific_notation_of_104000 :
  toScientificNotation givenNumber = ScientificNotation.mk 1.04 5 (by norm_num) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_104000_l3955_395535


namespace NUMINAMATH_CALUDE_runners_meet_again_l3955_395564

def track_length : ℝ := 600

def runner_speeds : List ℝ := [3.6, 4.2, 5.4, 6.0]

def meeting_time : ℝ := 1000

theorem runners_meet_again :
  ∀ (speed : ℝ), speed ∈ runner_speeds →
  ∃ (n : ℕ), speed * meeting_time = n * track_length :=
by sorry

end NUMINAMATH_CALUDE_runners_meet_again_l3955_395564


namespace NUMINAMATH_CALUDE_jesse_banana_sharing_l3955_395511

theorem jesse_banana_sharing (total_bananas : ℕ) (bananas_per_friend : ℕ) (h1 : total_bananas = 21) (h2 : bananas_per_friend = 7) :
  total_bananas / bananas_per_friend = 3 :=
by sorry

end NUMINAMATH_CALUDE_jesse_banana_sharing_l3955_395511


namespace NUMINAMATH_CALUDE_base_five_to_decimal_l3955_395579

/-- Converts a list of digits in a given base to its decimal representation. -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ (digits.length - 1 - i)) 0

/-- The decimal representation of 3412 in base 5 is 482. -/
theorem base_five_to_decimal : to_decimal [3, 4, 1, 2] 5 = 482 := by sorry

end NUMINAMATH_CALUDE_base_five_to_decimal_l3955_395579


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l3955_395503

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu

/-- Calculates the hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sheila's work schedule --/
def sheila_schedule : WorkSchedule :=
  { hours_mon_wed_fri := 8
  , hours_tue_thu := 6
  , weekly_earnings := 252 }

/-- Theorem stating that Sheila's hourly wage is $7 --/
theorem sheila_hourly_wage : hourly_wage sheila_schedule = 7 := by
  sorry


end NUMINAMATH_CALUDE_sheila_hourly_wage_l3955_395503


namespace NUMINAMATH_CALUDE_inequalities_theorem_l3955_395581

theorem inequalities_theorem (a b c d : ℝ) 
  (h1 : a > 0) (h2 : 0 > b) (h3 : b > -a) (h4 : c < d) (h5 : d < 0) : 
  (a / b + b / c < 0) ∧ (a - c > b - d) ∧ (a * (d - c) > b * (d - c)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l3955_395581


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l3955_395505

/-- Converts a number from base 5 to decimal --/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The number given in the problem in base 5 --/
def problemNumber : List Nat := [3, 3, 0, 4, 2, 0, 3, 1, 2]

/-- The decimal representation of the problem number --/
def decimalNumber : Nat := base5ToDecimal problemNumber

/-- Checks if a number is prime --/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- Theorem stating the largest prime divisor of the problem number --/
theorem largest_prime_divisor :
  ∃ (p : Nat), p = 11019 ∧ 
    isPrime p ∧ 
    (decimalNumber % p = 0) ∧
    (∀ q : Nat, isPrime q → decimalNumber % q = 0 → q ≤ p) :=
by sorry


end NUMINAMATH_CALUDE_largest_prime_divisor_l3955_395505


namespace NUMINAMATH_CALUDE_physics_marks_l3955_395584

theorem physics_marks (P C M : ℝ) 
  (avg_total : (P + C + M) / 3 = 55)
  (avg_pm : (P + M) / 2 = 90)
  (avg_pc : (P + C) / 2 = 70) :
  P = 155 := by
sorry

end NUMINAMATH_CALUDE_physics_marks_l3955_395584


namespace NUMINAMATH_CALUDE_angle_with_parallel_sides_l3955_395517

-- Define the concept of parallel angles
def parallel_angles (A B : Real) : Prop := sorry

-- Theorem statement
theorem angle_with_parallel_sides (A B : Real) :
  parallel_angles A B → A = 45 → (B = 45 ∨ B = 135) := by
  sorry

end NUMINAMATH_CALUDE_angle_with_parallel_sides_l3955_395517


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3955_395593

theorem least_subtraction_for_divisibility :
  ∃ (n : ℕ), n ≤ 7 ∧
  12 ∣ (652543 - n) ∧
  ∀ (m : ℕ), m < n → ¬(12 ∣ (652543 - m)) :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3955_395593


namespace NUMINAMATH_CALUDE_hanks_fruit_purchase_l3955_395582

/-- The total amount spent on fruits at Clark's Food Store -/
def total_spent (apple_price pear_price orange_price grape_price : ℚ)
                (apple_quantity pear_quantity orange_quantity grape_quantity : ℚ)
                (apple_discount pear_discount orange_discount grape_discount : ℚ) : ℚ :=
  (apple_price * apple_quantity * (1 - apple_discount)) +
  (pear_price * pear_quantity * (1 - pear_discount)) +
  (orange_price * orange_quantity * (1 - orange_discount)) +
  (grape_price * grape_quantity * (1 - grape_discount))

theorem hanks_fruit_purchase : 
  total_spent 40 50 30 60 14 18 10 8 0.1 0.05 0.15 0 = 2094 := by
  sorry

end NUMINAMATH_CALUDE_hanks_fruit_purchase_l3955_395582


namespace NUMINAMATH_CALUDE_teacher_number_game_l3955_395549

theorem teacher_number_game (x : ℤ) : 
  let ben_result := ((x + 3) * 2) - 2
  let sue_result := ((ben_result + 1) * 2) + 4
  sue_result = 2 * x + 30 := by
sorry

end NUMINAMATH_CALUDE_teacher_number_game_l3955_395549


namespace NUMINAMATH_CALUDE_arcsin_symmetry_l3955_395585

theorem arcsin_symmetry (x : ℝ) (h : x ∈ Set.Icc (-1) 1) :
  Real.arcsin (-x) = -Real.arcsin x := by
  sorry

end NUMINAMATH_CALUDE_arcsin_symmetry_l3955_395585


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l3955_395555

/-- Given a hyperbola defined by the equation x²/16 - y²/25 = 1, 
    the slopes of its asymptotes are ±5/4. -/
theorem hyperbola_asymptote_slopes :
  ∀ (x y : ℝ), x^2/16 - y^2/25 = 1 →
  ∃ (m : ℝ), m = 5/4 ∧ (y = m*x ∨ y = -m*x) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l3955_395555


namespace NUMINAMATH_CALUDE_correct_sampling_classification_l3955_395583

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a survey method with its characteristics --/
structure SurveyMethod where
  selectionProcess : String
  sampleSize : Nat

/-- Classifies a survey method into a sampling method --/
def classifySamplingMethod (method : SurveyMethod) : SamplingMethod :=
  sorry

/-- The total number of students in the survey --/
def totalStudents : Nat := 200

/-- The first survey method used --/
def method1 : SurveyMethod :=
  { selectionProcess := "Random selection by student council"
    sampleSize := 20 }

/-- The second survey method used --/
def method2 : SurveyMethod :=
  { selectionProcess := "Selecting students with numbers ending in 2"
    sampleSize := totalStudents / 10 }

/-- Theorem stating the correct classification of the two survey methods --/
theorem correct_sampling_classification :
  classifySamplingMethod method1 = SamplingMethod.SimpleRandom ∧
  classifySamplingMethod method2 = SamplingMethod.Systematic :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_classification_l3955_395583


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3955_395565

theorem log_equality_implies_golden_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.log a / Real.log 4 = Real.log b / Real.log 6 ∧ 
       Real.log a / Real.log 4 = Real.log (a + b) / Real.log 9) : 
  b / a = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3955_395565


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l3955_395532

theorem initial_mean_calculation (n : ℕ) (initial_mean corrected_mean : ℚ) : 
  n = 50 → 
  corrected_mean = 36.02 → 
  n * initial_mean + 1 = n * corrected_mean → 
  initial_mean = 36 := by
sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l3955_395532


namespace NUMINAMATH_CALUDE_work_completion_time_l3955_395569

theorem work_completion_time (work_rate_individual : ℝ) (total_work : ℝ) : 
  work_rate_individual > 0 → total_work > 0 →
  (total_work / work_rate_individual = 50) →
  (total_work / (2 * work_rate_individual) = 25) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3955_395569


namespace NUMINAMATH_CALUDE_craig_walking_distance_l3955_395528

/-- The distance Craig rode on the bus in miles -/
def bus_distance : ℝ := 3.83

/-- The difference between the bus distance and walking distance in miles -/
def distance_difference : ℝ := 3.67

/-- The distance Craig walked in miles -/
def walking_distance : ℝ := bus_distance - distance_difference

theorem craig_walking_distance : walking_distance = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_craig_walking_distance_l3955_395528


namespace NUMINAMATH_CALUDE_fraction_cube_theorem_l3955_395515

theorem fraction_cube_theorem :
  (2 : ℚ) / 5 ^ 3 = 8 / 125 :=
by sorry

end NUMINAMATH_CALUDE_fraction_cube_theorem_l3955_395515


namespace NUMINAMATH_CALUDE_tan_x_plus_pi_fourth_l3955_395504

theorem tan_x_plus_pi_fourth (x : ℝ) (h : Real.tan x = 2) : 
  Real.tan (x + π / 4) = -3 := by sorry

end NUMINAMATH_CALUDE_tan_x_plus_pi_fourth_l3955_395504


namespace NUMINAMATH_CALUDE_total_fruits_count_l3955_395521

-- Define the given conditions
def gerald_apple_bags : ℕ := 5
def gerald_apples_per_bag : ℕ := 30
def gerald_orange_bags : ℕ := 4
def gerald_oranges_per_bag : ℕ := 25

def pam_apple_bags : ℕ := 6
def pam_orange_bags : ℕ := 4

def sue_apple_bags : ℕ := 2 * gerald_apple_bags
def sue_orange_bags : ℕ := gerald_orange_bags / 2

def pam_apples_per_bag : ℕ := 3 * gerald_apples_per_bag
def pam_oranges_per_bag : ℕ := 2 * gerald_oranges_per_bag

def sue_apples_per_bag : ℕ := gerald_apples_per_bag - 10
def sue_oranges_per_bag : ℕ := gerald_oranges_per_bag + 5

-- Theorem statement
theorem total_fruits_count : 
  (gerald_apple_bags * gerald_apples_per_bag + 
   gerald_orange_bags * gerald_oranges_per_bag +
   pam_apple_bags * pam_apples_per_bag + 
   pam_orange_bags * pam_oranges_per_bag +
   sue_apple_bags * sue_apples_per_bag + 
   sue_orange_bags * sue_oranges_per_bag) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_count_l3955_395521


namespace NUMINAMATH_CALUDE_exam_score_problem_l3955_395588

theorem exam_score_problem (total_questions : ℕ) 
  (correct_score wrong_score total_score : ℤ) : 
  total_questions = 80 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 130 →
  ∃ (correct_answers wrong_answers : ℕ),
    correct_answers + wrong_answers = total_questions ∧
    correct_score * correct_answers + wrong_score * wrong_answers = total_score ∧
    correct_answers = 42 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l3955_395588


namespace NUMINAMATH_CALUDE_trigonometric_calculations_l3955_395502

theorem trigonometric_calculations :
  (2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1 / 2) ∧
  ((-1)^2023 + 2 * Real.sin (45 * π / 180) - Real.cos (30 * π / 180) + Real.sin (60 * π / 180) + Real.tan (60 * π / 180)^2 = 2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_calculations_l3955_395502


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3955_395594

theorem circle_diameter_from_area :
  ∀ (r d : ℝ),
  r > 0 →
  d = 2 * r →
  π * r^2 = 225 * π →
  d = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3955_395594


namespace NUMINAMATH_CALUDE_simple_interest_double_l3955_395568

/-- The factor by which a sum of money increases under simple interest -/
def simple_interest_factor (rate : ℝ) (time : ℝ) : ℝ :=
  1 + rate * time

theorem simple_interest_double :
  simple_interest_factor 0.1 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_double_l3955_395568


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3955_395580

theorem quadratic_form_sum (x : ℝ) : ∃ (a b c : ℝ), 
  (15 * x^2 + 90 * x + 405 = a * (x + b)^2 + c) ∧ (a + b + c = 288) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3955_395580


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_2pi_3_l3955_395522

theorem cos_2alpha_minus_2pi_3 (α : Real) (h : Real.sin (π/6 + α) = 3/5) : 
  Real.cos (2*α - 2*π/3) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_2pi_3_l3955_395522


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l3955_395536

theorem framed_painting_ratio : 
  ∀ (x : ℝ),
  x > 0 →
  (30 + 2*x) * (20 + 4*x) = 1500 →
  (min (30 + 2*x) (20 + 4*x)) / (max (30 + 2*x) (20 + 4*x)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l3955_395536


namespace NUMINAMATH_CALUDE_min_value_theorem_l3955_395540

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → 3*x + 2*y + y/x ≥ 11 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3955_395540


namespace NUMINAMATH_CALUDE_sandy_average_price_per_book_l3955_395530

/-- Represents a bookshop visit with the number of books bought and the total price paid -/
structure BookshopVisit where
  books : ℕ
  price : ℚ

/-- Calculates the average price per book given a list of bookshop visits -/
def averagePricePerBook (visits : List BookshopVisit) : ℚ :=
  (visits.map (λ v => v.price)).sum / (visits.map (λ v => v.books)).sum

/-- The theorem statement for Sandy's bookshop visits -/
theorem sandy_average_price_per_book :
  let visits : List BookshopVisit := [
    { books := 65, price := 1080 },
    { books := 55, price := 840 },
    { books := 45, price := 765 },
    { books := 35, price := 630 }
  ]
  averagePricePerBook visits = 16575 / 1000 := by
  sorry


end NUMINAMATH_CALUDE_sandy_average_price_per_book_l3955_395530


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_five_l3955_395586

theorem arithmetic_square_root_of_five :
  ∃ x : ℝ, x > 0 ∧ x^2 = 5 ∧ ∀ y : ℝ, y^2 = 5 → y = x ∨ y = -x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_five_l3955_395586


namespace NUMINAMATH_CALUDE_remainder_theorem_l3955_395546

theorem remainder_theorem (P K Q R K' Q' S' T : ℕ) 
  (h1 : P = K * Q + R)
  (h2 : Q = K' * Q' + S')
  (h3 : R * Q' = T)
  (h4 : Q' ≠ 0) :
  P % (K * K') = K * S' + T / Q' :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3955_395546


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_and_discriminant_l3955_395575

theorem quadratic_equation_roots_and_discriminant :
  let a : ℝ := 1
  let b : ℝ := 5
  let c : ℝ := 0
  let f : ℝ → ℝ := λ x => x^2 + b*x + c
  let discriminant := b^2 - 4*a*c
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0) ∧ 
              (x₁ = 0 ∧ x₂ = -5) ∧
              discriminant = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_and_discriminant_l3955_395575


namespace NUMINAMATH_CALUDE_percentage_of_eight_l3955_395563

theorem percentage_of_eight : ∃ p : ℝ, (p / 100) * 8 = 0.06 ∧ p = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_eight_l3955_395563


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3955_395525

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  -- a, b, c are sides opposite to angles A, B, C respectively
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- Given conditions
  ((2 * Real.cos A - 1) * Real.sin B + 2 * Real.cos A = 1) ∧
  (5 * b^2 = a^2 + 2 * c^2) →
  -- Conclusions
  (A = π / 3) ∧
  (Real.sin B / Real.sin C = 3 / 4) := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3955_395525


namespace NUMINAMATH_CALUDE_cube_sqrt_16_equals_8_times_8_l3955_395539

theorem cube_sqrt_16_equals_8_times_8 : 
  (8 : ℝ) * 8 = (Real.sqrt 16)^3 := by sorry

end NUMINAMATH_CALUDE_cube_sqrt_16_equals_8_times_8_l3955_395539


namespace NUMINAMATH_CALUDE_suit_price_calculation_l3955_395512

theorem suit_price_calculation (original_price : ℝ) : 
  original_price * 1.25 * 0.75 = 187.5 → original_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_suit_price_calculation_l3955_395512


namespace NUMINAMATH_CALUDE_trailing_zeros_340_factorial_l3955_395526

-- Define a function to count trailing zeros in a factorial
def trailingZerosInFactorial (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

-- Theorem statement
theorem trailing_zeros_340_factorial :
  trailingZerosInFactorial 340 = 83 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_340_factorial_l3955_395526


namespace NUMINAMATH_CALUDE_power_function_through_point_l3955_395553

/-- A power function that passes through the point (9, 3) -/
def f (x : ℝ) : ℝ := x^(1/2)

/-- Theorem stating that f(9) = 3 -/
theorem power_function_through_point : f 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3955_395553


namespace NUMINAMATH_CALUDE_jebb_take_home_pay_l3955_395578

/-- Calculates the take-home pay after tax deduction -/
def takeHomePay (totalPay : ℝ) (taxRate : ℝ) : ℝ :=
  totalPay * (1 - taxRate)

/-- Proves that Jebb's take-home pay is $585 -/
theorem jebb_take_home_pay :
  let totalPay : ℝ := 650
  let taxRate : ℝ := 0.1
  takeHomePay totalPay taxRate = 585 := by
  sorry

end NUMINAMATH_CALUDE_jebb_take_home_pay_l3955_395578


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3955_395500

theorem quadratic_root_problem (m : ℝ) : 
  (3^2 - m * 3 + 3 = 0) → 
  (∃ (x : ℝ), x ≠ 3 ∧ x^2 - m * x + 3 = 0 ∧ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3955_395500


namespace NUMINAMATH_CALUDE_find_x_l3955_395587

-- Define the variables
variable (a b x : ℝ)
variable (r : ℝ)

-- State the theorem
theorem find_x (h1 : b ≠ 0) (h2 : r = (3 * a) ^ (2 * b)) (h3 : r = a ^ b * x ^ (2 * b)) : x = 3 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_find_x_l3955_395587


namespace NUMINAMATH_CALUDE_correct_total_items_l3955_395561

/-- Represents the requirements for a packed lunch --/
structure LunchRequirements where
  sandwiches_per_student : ℕ
  bread_slices_per_sandwich : ℕ
  chips_per_student : ℕ
  apples_per_student : ℕ
  granola_bars_per_student : ℕ

/-- Represents the number of students in each group --/
structure StudentGroups where
  group_a : ℕ
  group_b : ℕ
  group_c : ℕ

/-- Calculates the total number of items needed for packed lunches --/
def calculate_total_items (req : LunchRequirements) (groups : StudentGroups) :
  (ℕ × ℕ × ℕ × ℕ) :=
  let total_students := groups.group_a + groups.group_b + groups.group_c
  let total_bread_slices := total_students * req.sandwiches_per_student * req.bread_slices_per_sandwich
  let total_chips := total_students * req.chips_per_student
  let total_apples := total_students * req.apples_per_student
  let total_granola_bars := total_students * req.granola_bars_per_student
  (total_bread_slices, total_chips, total_apples, total_granola_bars)

/-- Theorem stating the correct calculation of total items needed --/
theorem correct_total_items :
  let req : LunchRequirements := {
    sandwiches_per_student := 2,
    bread_slices_per_sandwich := 4,
    chips_per_student := 1,
    apples_per_student := 3,
    granola_bars_per_student := 1
  }
  let groups : StudentGroups := {
    group_a := 10,
    group_b := 15,
    group_c := 20
  }
  calculate_total_items req groups = (360, 45, 135, 45) :=
by
  sorry


end NUMINAMATH_CALUDE_correct_total_items_l3955_395561


namespace NUMINAMATH_CALUDE_swimmers_pass_count_l3955_395509

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the swimming problem setup --/
structure SwimmingProblem where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  totalTime : ℝ

/-- Calculates the number of times the swimmers pass each other --/
def countPasses (problem : SwimmingProblem) : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem swimmers_pass_count (problem : SwimmingProblem) 
  (h1 : problem.poolLength = 120)
  (h2 : problem.swimmer1.speed = 4)
  (h3 : problem.swimmer2.speed = 3)
  (h4 : problem.swimmer1.startPosition = 0)
  (h5 : problem.swimmer2.startPosition = 120)
  (h6 : problem.totalTime = 15 * 60) : 
  countPasses problem = 29 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_pass_count_l3955_395509


namespace NUMINAMATH_CALUDE_min_gennadies_for_festival_l3955_395556

/-- Represents the number of people with a specific name -/
structure NameCount where
  alexanders : Nat
  borises : Nat
  vasilies : Nat

/-- Calculates the minimum number of Gennadies required -/
def minGennadies (counts : NameCount) : Nat :=
  counts.borises - 1 - (counts.alexanders + counts.vasilies)

/-- Theorem stating the minimum number of Gennadies required for the given counts -/
theorem min_gennadies_for_festival (counts : NameCount) 
  (h_alex : counts.alexanders = 45)
  (h_boris : counts.borises = 122)
  (h_vasily : counts.vasilies = 27) :
  minGennadies counts = 49 := by
  sorry

#eval minGennadies { alexanders := 45, borises := 122, vasilies := 27 }

end NUMINAMATH_CALUDE_min_gennadies_for_festival_l3955_395556


namespace NUMINAMATH_CALUDE_quadratic_expression_minimum_l3955_395598

theorem quadratic_expression_minimum (x y : ℝ) : 
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5 ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_minimum_l3955_395598


namespace NUMINAMATH_CALUDE_als_original_portion_l3955_395523

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1000 →
  a - 100 + 2*b + 2*c = 1500 →
  a = 400 :=
by sorry

end NUMINAMATH_CALUDE_als_original_portion_l3955_395523


namespace NUMINAMATH_CALUDE_min_value_of_sum_fractions_l3955_395543

theorem min_value_of_sum_fractions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) / c + (a + c) / b + (b + c) / a ≥ 6 ∧
  ((a + b) / c + (a + c) / b + (b + c) / a = 6 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_fractions_l3955_395543


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_inequality_l3955_395591

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x := by sorry

theorem negation_of_exponential_inequality :
  (¬ ∃ x : ℝ, Real.exp x > x) ↔ (∀ x : ℝ, Real.exp x ≤ x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_inequality_l3955_395591


namespace NUMINAMATH_CALUDE_parabola_points_theorem_l3955_395519

/-- Parabola passing through given points -/
def parabola (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + x + c

theorem parabola_points_theorem :
  ∃ (a c m n : ℝ),
    (parabola a c 0 = -2) ∧
    (parabola a c 1 = 1) ∧
    (parabola a c 2 = m) ∧
    (parabola a c n = -2) ∧
    (a = 2) ∧
    (c = -2) ∧
    (m = 8) ∧
    (n = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_theorem_l3955_395519


namespace NUMINAMATH_CALUDE_correct_probability_l3955_395547

/-- The number of options for the first three digits -/
def first_three_options : ℕ := 3

/-- The number of permutations of the last four digits (0, 1, 6, 6) -/
def last_four_permutations : ℕ := 12

/-- The total number of possible phone numbers -/
def total_possible_numbers : ℕ := first_three_options * last_four_permutations

/-- The probability of dialing the correct number -/
def probability_correct : ℚ := 1 / total_possible_numbers

theorem correct_probability : probability_correct = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_probability_l3955_395547


namespace NUMINAMATH_CALUDE_jenna_smoothies_l3955_395520

/-- Given that Jenna can make 15 smoothies from 3 strawberries, 
    prove that she can make 90 smoothies from 18 strawberries. -/
theorem jenna_smoothies (smoothies_per_three : ℕ) (strawberries : ℕ) 
  (h1 : smoothies_per_three = 15) 
  (h2 : strawberries = 18) : 
  (smoothies_per_three * strawberries) / 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_jenna_smoothies_l3955_395520


namespace NUMINAMATH_CALUDE_quadratic_decreasing_implies_a_bound_l3955_395599

/-- Given a quadratic function f(x) = x^2 + 4ax + 2 that is decreasing on (-∞, 6),
    prove that a ≤ -3. -/
theorem quadratic_decreasing_implies_a_bound 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^2 + 4*a*x + 2)
  (h2 : ∀ x y, x < y → y < 6 → f y < f x) :
  a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_implies_a_bound_l3955_395599


namespace NUMINAMATH_CALUDE_perpendicular_vectors_condition_l3955_395589

/-- Given two vectors m and n in ℝ², if m is perpendicular to n,
    then the second component of n is -2 times the first component of m. -/
theorem perpendicular_vectors_condition (m n : ℝ × ℝ) :
  m = (1, 2) →
  n.1 = a →
  n.2 = -1 →
  m.1 * n.1 + m.2 * n.2 = 0 →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_condition_l3955_395589


namespace NUMINAMATH_CALUDE_curve_self_intersection_l3955_395516

-- Define the curve
def x (t : ℝ) : ℝ := t^3 - t - 2
def y (t : ℝ) : ℝ := t^3 - t^2 - 9*t + 5

-- Define the self-intersection point
def intersection_point : ℝ × ℝ := (22, -4)

-- Theorem statement
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ 
    x a = x b ∧ 
    y a = y b ∧ 
    (x a, y a) = intersection_point :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l3955_395516


namespace NUMINAMATH_CALUDE_tank_volume_in_cubic_yards_l3955_395574

/-- Conversion factor from cubic feet to cubic yards -/
def cubicFeetToCubicYards : ℚ := 1 / 27

/-- Volume of the tank in cubic feet -/
def tankVolumeCubicFeet : ℚ := 216

/-- Theorem: The volume of the tank in cubic yards is 8 -/
theorem tank_volume_in_cubic_yards :
  tankVolumeCubicFeet * cubicFeetToCubicYards = 8 := by
  sorry

end NUMINAMATH_CALUDE_tank_volume_in_cubic_yards_l3955_395574


namespace NUMINAMATH_CALUDE_range_of_a_l3955_395534

-- Define the propositions p and q
def p (m a : ℝ) : Prop := m^2 - 7*a*m + 12*a^2 < 0 ∧ a > 0

def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧ 
  ∃ (c : ℝ), c > 0 ∧ x^2 / (m - 1) + y^2 / (2 - m - c) = 1

-- Define the theorem
theorem range_of_a : 
  (∀ m a : ℝ, ¬(q m) → ¬(p m a)) ∧ 
  (∃ m a : ℝ, ¬(q m) ∧ p m a) → 
  {a : ℝ | 1/3 ≤ a ∧ a ≤ 3/8} = {a : ℝ | ∃ m : ℝ, p m a} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3955_395534


namespace NUMINAMATH_CALUDE_no_hexagon_with_special_point_l3955_395554

-- Define a hexagon as a set of 6 points in 2D space
def Hexagon := Fin 6 → ℝ × ℝ

-- Define convexity for a hexagon
def is_convex (h : Hexagon) : Prop := sorry

-- Define a point being inside a hexagon
def is_inside (p : ℝ × ℝ) (h : Hexagon) : Prop := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- The main theorem
theorem no_hexagon_with_special_point :
  ¬ ∃ (h : Hexagon) (m : ℝ × ℝ),
    is_convex h ∧
    is_inside m h ∧
    (∀ i j : Fin 6, i ≠ j → distance (h i) (h j) > 1) ∧
    (∀ i : Fin 6, distance m (h i) < 1) :=
by sorry

end NUMINAMATH_CALUDE_no_hexagon_with_special_point_l3955_395554


namespace NUMINAMATH_CALUDE_discounted_price_approx_l3955_395573

/-- The original price of the shirt in rupees -/
def original_price : ℝ := 746.67

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.25

/-- The discounted price of the shirt -/
def discounted_price : ℝ := original_price * (1 - discount_rate)

/-- Theorem stating that the discounted price is approximately 560 rupees -/
theorem discounted_price_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |discounted_price - 560| < ε :=
sorry

end NUMINAMATH_CALUDE_discounted_price_approx_l3955_395573


namespace NUMINAMATH_CALUDE_female_attendees_on_time_l3955_395501

/-- Proves that the fraction of female attendees who arrived on time is 0.9 -/
theorem female_attendees_on_time 
  (total_attendees : ℝ) 
  (male_ratio : ℝ) 
  (male_on_time_ratio : ℝ) 
  (not_on_time_ratio : ℝ) 
  (h1 : male_ratio = 3/5) 
  (h2 : male_on_time_ratio = 7/8) 
  (h3 : not_on_time_ratio = 0.115) : 
  let female_ratio := 1 - male_ratio
  let female_on_time_ratio := 
    (1 - not_on_time_ratio - male_ratio * male_on_time_ratio) / female_ratio
  female_on_time_ratio = 0.9 := by
sorry

end NUMINAMATH_CALUDE_female_attendees_on_time_l3955_395501


namespace NUMINAMATH_CALUDE_more_women_than_men_l3955_395506

theorem more_women_than_men (total : ℕ) (ratio : ℚ) : 
  total = 18 → ratio = 7/11 → ∃ (men women : ℕ), 
    men + women = total ∧ 
    (men : ℚ) / (women : ℚ) = ratio ∧ 
    women - men = 4 :=
sorry

end NUMINAMATH_CALUDE_more_women_than_men_l3955_395506


namespace NUMINAMATH_CALUDE_six_ronna_grams_scientific_notation_l3955_395566

/-- Represents the number of zeros after a number for the 'ronna' prefix --/
def ronna_zeros : ℕ := 27

/-- Converts a number with the 'ronna' prefix to its scientific notation --/
def ronna_to_scientific (n : ℝ) : ℝ := n * (10 ^ ronna_zeros)

/-- Theorem stating that 6 ronna grams is equal to 6 × 10^27 grams --/
theorem six_ronna_grams_scientific_notation :
  ronna_to_scientific 6 = 6 * (10 ^ 27) := by sorry

end NUMINAMATH_CALUDE_six_ronna_grams_scientific_notation_l3955_395566


namespace NUMINAMATH_CALUDE_adjacent_sum_theorem_l3955_395571

/-- Represents a 3x3 table with numbers from 1 to 9 -/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Checks if a table contains each number from 1 to 9 exactly once -/
def is_valid (t : Table) : Prop :=
  ∀ n : Fin 9, ∃! (i j : Fin 3), t i j = n

/-- Checks if the table has 1, 2, 3, and 4 in the correct positions -/
def correct_positions (t : Table) : Prop :=
  t 0 0 = 0 ∧ t 2 0 = 1 ∧ t 0 2 = 2 ∧ t 2 2 = 3

/-- Returns the sum of adjacent numbers to the given position -/
def adjacent_sum (t : Table) (i j : Fin 3) : ℕ :=
  (if i > 0 then (t (i-1) j).val + 1 else 0) +
  (if i < 2 then (t (i+1) j).val + 1 else 0) +
  (if j > 0 then (t i (j-1)).val + 1 else 0) +
  (if j < 2 then (t i (j+1)).val + 1 else 0)

/-- The main theorem -/
theorem adjacent_sum_theorem (t : Table) :
  is_valid t →
  correct_positions t →
  (∃ i j : Fin 3, t i j = 4 ∧ adjacent_sum t i j = 9) →
  (∃ i j : Fin 3, t i j = 5 ∧ adjacent_sum t i j = 29) :=
by sorry

end NUMINAMATH_CALUDE_adjacent_sum_theorem_l3955_395571


namespace NUMINAMATH_CALUDE_remainder_problem_l3955_395576

theorem remainder_problem (n : ℤ) (h : n % 18 = 10) : (2 * n) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3955_395576


namespace NUMINAMATH_CALUDE_field_ratio_proof_l3955_395557

theorem field_ratio_proof (length width : ℝ) : 
  length = 24 → 
  width = 13.5 → 
  (2 * width) / length = 9 / 8 := by
sorry

end NUMINAMATH_CALUDE_field_ratio_proof_l3955_395557


namespace NUMINAMATH_CALUDE_det_M_eq_26_l3955_395577

/-- The determinant of a 2x2 matrix -/
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- The specific 2x2 matrix we're interested in -/
def M : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3; -2, 4]

/-- Theorem stating that the determinant of M is 26 -/
theorem det_M_eq_26 : det2x2 (M 0 0) (M 0 1) (M 1 0) (M 1 1) = 26 := by
  sorry

end NUMINAMATH_CALUDE_det_M_eq_26_l3955_395577
