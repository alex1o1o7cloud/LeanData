import Mathlib

namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_225_with_ones_and_zeros_l3905_390544

def is_composed_of_ones_and_zeros (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1 ∨ d = 0

def smallest_divisible_by_225_with_ones_and_zeros : ℕ := 11111111100

theorem smallest_number_divisible_by_225_with_ones_and_zeros :
  (smallest_divisible_by_225_with_ones_and_zeros % 225 = 0) ∧
  is_composed_of_ones_and_zeros smallest_divisible_by_225_with_ones_and_zeros ∧
  ∀ n : ℕ, n < smallest_divisible_by_225_with_ones_and_zeros →
    ¬(n % 225 = 0 ∧ is_composed_of_ones_and_zeros n) :=
by sorry

#eval smallest_divisible_by_225_with_ones_and_zeros

end NUMINAMATH_CALUDE_smallest_number_divisible_by_225_with_ones_and_zeros_l3905_390544


namespace NUMINAMATH_CALUDE_vidyas_age_l3905_390539

theorem vidyas_age (vidya_age : ℕ) (mother_age : ℕ) : 
  mother_age = 3 * vidya_age + 5 →
  mother_age = 44 →
  vidya_age = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_vidyas_age_l3905_390539


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l3905_390510

/-- A continuous monotonous function satisfying the given inequality -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ 
  Monotone f ∧
  f 0 = 1 ∧
  ∀ x y : ℝ, f (x + y) ≥ f x * f y - f (x * y) + 1

/-- The main theorem stating that any function satisfying the conditions must be f(x) = x + 1 -/
theorem unique_satisfying_function (f : ℝ → ℝ) (hf : SatisfyingFunction f) : 
  ∀ x : ℝ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l3905_390510


namespace NUMINAMATH_CALUDE_equation_solution_l3905_390574

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (x + 2 = 1 / (x - 2)) ↔ (x = Real.sqrt 5 ∨ x = -Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3905_390574


namespace NUMINAMATH_CALUDE_dollar_sum_squared_zero_l3905_390559

/-- Definition of the $ operation for real numbers -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem: For real numbers x and y, (x + y)^2 $ (y + x)^2 = 0 -/
theorem dollar_sum_squared_zero (x y : ℝ) : dollar ((x + y)^2) ((y + x)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dollar_sum_squared_zero_l3905_390559


namespace NUMINAMATH_CALUDE_pizza_cost_distribution_l3905_390569

theorem pizza_cost_distribution (total_cost : ℚ) (num_students : ℕ) 
  (price1 price2 : ℚ) (h1 : total_cost = 26) (h2 : num_students = 7) 
  (h3 : price1 = 371/100) (h4 : price2 = 372/100) : 
  ∃ (x y : ℕ), x + y = num_students ∧ 
  x * price1 + y * price2 = total_cost ∧ 
  y = 3 := by sorry

end NUMINAMATH_CALUDE_pizza_cost_distribution_l3905_390569


namespace NUMINAMATH_CALUDE_sin_two_a_value_l3905_390527

theorem sin_two_a_value (a : ℝ) (h : Real.sin a - Real.cos a = 4/3) : 
  Real.sin (2 * a) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_a_value_l3905_390527


namespace NUMINAMATH_CALUDE_kathryn_salary_l3905_390593

/-- Calculates Kathryn's monthly salary given her expenses and remaining money --/
def monthly_salary (rent : ℕ) (remaining : ℕ) : ℕ :=
  let food_travel := 2 * rent
  let total_expenses := rent + food_travel
  let shared_rent := rent / 2
  let adjusted_expenses := total_expenses - (rent - shared_rent)
  adjusted_expenses + remaining

/-- Proves that Kathryn's monthly salary is $5000 given the problem conditions --/
theorem kathryn_salary :
  let rent : ℕ := 1200
  let remaining : ℕ := 2000
  monthly_salary rent remaining = 5000 := by
  sorry

#eval monthly_salary 1200 2000

end NUMINAMATH_CALUDE_kathryn_salary_l3905_390593


namespace NUMINAMATH_CALUDE_video_game_marathon_points_l3905_390576

theorem video_game_marathon_points : 
  ∀ (jack_points alex_bella_points : ℕ),
    jack_points = 8972 →
    alex_bella_points = 21955 →
    jack_points + alex_bella_points = 30927 := by
  sorry

end NUMINAMATH_CALUDE_video_game_marathon_points_l3905_390576


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l3905_390541

/-- The volume of a cube inscribed in a specific pyramid -/
theorem inscribed_cube_volume (base_side : ℝ) (h : base_side = 2) :
  let pyramid_height := 2 * Real.sqrt 3 / 3
  let cube_side := 2 * Real.sqrt 3 / 9
  let cube_volume := cube_side ^ 3
  cube_volume = 8 * Real.sqrt 3 / 243 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l3905_390541


namespace NUMINAMATH_CALUDE_digital_earth_not_equal_gis_l3905_390556

-- Define the concept of Digital Earth
def DigitalEarth : Type := Unit

-- Define Geographic Information Technology
def GeographicInformationTechnology : Type := Unit

-- Define other related technologies
def RemoteSensing : Type := Unit
def GPS : Type := Unit
def VirtualTechnology : Type := Unit
def NetworkTechnology : Type := Unit

-- Define the correct properties of Digital Earth
axiom digital_earth_properties : 
  DigitalEarth → 
  (GeographicInformationTechnology × VirtualTechnology × NetworkTechnology)

-- Define the incorrect statement
def incorrect_statement : Prop :=
  DigitalEarth = GeographicInformationTechnology

-- Theorem to prove
theorem digital_earth_not_equal_gis : ¬incorrect_statement :=
sorry

end NUMINAMATH_CALUDE_digital_earth_not_equal_gis_l3905_390556


namespace NUMINAMATH_CALUDE_tan_pi_plus_alpha_eq_two_implies_fraction_eq_three_l3905_390503

theorem tan_pi_plus_alpha_eq_two_implies_fraction_eq_three (α : Real) 
  (h : Real.tan (π + α) = 2) : 
  (Real.sin (α - π) + Real.cos (π - α)) / (Real.sin (π + α) - Real.cos (π - α)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_plus_alpha_eq_two_implies_fraction_eq_three_l3905_390503


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3905_390524

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 22*a - 8 = 0 →
  b^3 - 15*b^2 + 22*b - 8 = 0 →
  c^3 - 15*c^2 + 22*c - 8 = 0 →
  (a / ((1/a) + b*c)) + (b / ((1/b) + c*a)) + (c / ((1/c) + a*b)) = 181/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3905_390524


namespace NUMINAMATH_CALUDE_run_difference_is_240_l3905_390562

/-- The width of the street in feet -/
def street_width : ℝ := 30

/-- The side length of the square block in feet -/
def block_side : ℝ := 500

/-- The perimeter of Sarah's run (inner side of the block) -/
def sarah_perimeter : ℝ := 4 * block_side

/-- The perimeter of Sam's run (outer side of the block) -/
def sam_perimeter : ℝ := 4 * (block_side + 2 * street_width)

/-- The difference in distance run by Sam and Sarah -/
def run_difference : ℝ := sam_perimeter - sarah_perimeter

theorem run_difference_is_240 : run_difference = 240 := by
  sorry

end NUMINAMATH_CALUDE_run_difference_is_240_l3905_390562


namespace NUMINAMATH_CALUDE_mean_ice_cream_sales_l3905_390554

def ice_cream_sales : List ℕ := [100, 92, 109, 96, 103, 96, 105]

theorem mean_ice_cream_sales :
  (ice_cream_sales.sum : ℚ) / ice_cream_sales.length = 100.14 := by
  sorry

end NUMINAMATH_CALUDE_mean_ice_cream_sales_l3905_390554


namespace NUMINAMATH_CALUDE_gcd_of_35_91_840_l3905_390523

theorem gcd_of_35_91_840 : Nat.gcd 35 (Nat.gcd 91 840) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_35_91_840_l3905_390523


namespace NUMINAMATH_CALUDE_parabola_properties_l3905_390565

-- Define the parabola function
def f (a x : ℝ) : ℝ := x^2 + (a + 2) * x - 2 * a + 1

-- State the theorem
theorem parabola_properties (a : ℝ) :
  -- 1. The parabola always passes through the point (2, 9)
  (f a 2 = 9) ∧
  -- 2. The vertex of the parabola lies on the curve y = -x^2 + 4x + 5
  (∃ x y : ℝ, y = f a x ∧ y = -x^2 + 4*x + 5 ∧ 
    ∀ t : ℝ, f a t ≥ f a x) ∧
  -- 3. When the quadratic equation has two distinct real roots,
  --    the range of the larger root is (-1, 2) ∪ (5, +∞)
  (∀ x : ℝ, (f a x = 0 ∧ 
    (∃ y : ℝ, y ≠ x ∧ f a y = 0)) →
    ((x > -1 ∧ x < 2) ∨ x > 5)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3905_390565


namespace NUMINAMATH_CALUDE_wednesday_temperature_l3905_390545

/-- Given the high temperatures for three consecutive days (Monday, Tuesday, Wednesday),
    prove that Wednesday's temperature is 12°C. -/
theorem wednesday_temperature
  (monday tuesday wednesday : ℝ)
  (h1 : tuesday = monday + 4)
  (h2 : wednesday = monday - 6)
  (h3 : tuesday = 22) :
  wednesday = 12 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_temperature_l3905_390545


namespace NUMINAMATH_CALUDE_cube_difference_divisibility_l3905_390558

theorem cube_difference_divisibility (a b : ℤ) :
  24 ∣ ((2 * a + 1)^3 - (2 * b + 1)^3) ↔ 3 ∣ (a - b) := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_divisibility_l3905_390558


namespace NUMINAMATH_CALUDE_cos_18_minus_cos_54_l3905_390515

theorem cos_18_minus_cos_54 :
  Real.cos (18 * π / 180) - Real.cos (54 * π / 180) =
  -16 * (Real.cos (9 * π / 180))^4 + 24 * (Real.cos (9 * π / 180))^2 - 4 := by
sorry

end NUMINAMATH_CALUDE_cos_18_minus_cos_54_l3905_390515


namespace NUMINAMATH_CALUDE_correct_ages_l3905_390579

/-- Represents the ages of a family -/
structure FamilyAges where
  kareem : ℕ
  son : ℕ
  daughter : ℕ
  wife : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.kareem = 3 * ages.son ∧
  ages.daughter = ages.son / 2 ∧
  ages.kareem + 10 + ages.son + 10 + ages.daughter + 10 = 120 ∧
  ages.wife = ages.kareem - 8

/-- Theorem stating that the given ages satisfy the problem conditions -/
theorem correct_ages : 
  let ages : FamilyAges := ⟨60, 20, 10, 52⟩
  satisfiesConditions ages :=
by sorry

end NUMINAMATH_CALUDE_correct_ages_l3905_390579


namespace NUMINAMATH_CALUDE_workers_paid_four_fifties_is_31_l3905_390529

/-- Represents the payment structure for workers -/
structure PaymentStructure where
  total_workers : Nat
  payment_per_worker : Nat
  hundred_bills : Nat
  fifty_bills : Nat
  workers_paid_two_hundreds : Nat

/-- Calculates the number of workers paid with four $50 bills -/
def workers_paid_four_fifties (p : PaymentStructure) : Nat :=
  let remaining_hundreds := p.hundred_bills - 2 * p.workers_paid_two_hundreds
  let workers_paid_mixed := remaining_hundreds
  let fifties_for_mixed := 2 * workers_paid_mixed
  let remaining_fifties := p.fifty_bills - fifties_for_mixed
  remaining_fifties / 4

/-- Theorem stating that given the specific payment structure, 31 workers are paid with four $50 bills -/
theorem workers_paid_four_fifties_is_31 :
  let p : PaymentStructure := {
    total_workers := 108,
    payment_per_worker := 200,
    hundred_bills := 122,
    fifty_bills := 188,
    workers_paid_two_hundreds := 45
  }
  workers_paid_four_fifties p = 31 := by
  sorry

end NUMINAMATH_CALUDE_workers_paid_four_fifties_is_31_l3905_390529


namespace NUMINAMATH_CALUDE_max_min_triangle_area_l3905_390550

/-- A point on the 10x10 grid -/
structure GridPoint where
  x : Fin 11
  y : Fin 11

/-- The configuration of three pieces on the grid -/
structure Configuration where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The area of a triangle formed by three points -/
def triangleArea (p1 p2 p3 : GridPoint) : ℚ :=
  sorry

/-- Check if two grid points are adjacent -/
def isAdjacent (p1 p2 : GridPoint) : Prop :=
  sorry

/-- A valid move between two configurations -/
def validMove (c1 c2 : Configuration) : Prop :=
  sorry

/-- A sequence of configurations representing a valid solution -/
def ValidSolution : Type :=
  sorry

/-- The minimum triangle area over all configurations in a solution -/
def minTriangleArea (sol : ValidSolution) : ℚ :=
  sorry

theorem max_min_triangle_area :
  (∃ (sol : ValidSolution), minTriangleArea sol = 5/2) ∧
  (∀ (sol : ValidSolution), minTriangleArea sol ≤ 5/2) := by
  sorry

end NUMINAMATH_CALUDE_max_min_triangle_area_l3905_390550


namespace NUMINAMATH_CALUDE_dihedral_angle_range_l3905_390542

/-- The dihedral angle between two adjacent lateral faces of a regular n-sided pyramid -/
def dihedral_angle (n : ℕ) (h : ℝ) : ℝ :=
  sorry

/-- The internal angle of a regular n-sided polygon -/
def internal_angle (n : ℕ) : ℝ :=
  sorry

theorem dihedral_angle_range (n : ℕ) (h : ℝ) :
  0 < dihedral_angle n h ∧ dihedral_angle n h < π :=
by sorry

end NUMINAMATH_CALUDE_dihedral_angle_range_l3905_390542


namespace NUMINAMATH_CALUDE_prob_B_wins_at_least_one_l3905_390517

/-- The probability of player A winning against player B in a single match. -/
def prob_A_win : ℝ := 0.5

/-- The probability of player B winning against player A in a single match. -/
def prob_B_win : ℝ := 0.3

/-- The probability of a tie between players A and B in a single match. -/
def prob_tie : ℝ := 0.2

/-- The number of matches played between A and B. -/
def num_matches : ℕ := 2

/-- Theorem: The probability of B winning at least one match against A in two independent matches. -/
theorem prob_B_wins_at_least_one (h1 : prob_A_win + prob_B_win + prob_tie = 1) :
  1 - (1 - prob_B_win) ^ num_matches = 0.51 := by
  sorry

end NUMINAMATH_CALUDE_prob_B_wins_at_least_one_l3905_390517


namespace NUMINAMATH_CALUDE_jellybean_problem_l3905_390507

theorem jellybean_problem (initial_count : ℕ) : 
  (initial_count : ℝ) * (3/4)^3 = 27 → initial_count = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l3905_390507


namespace NUMINAMATH_CALUDE_book_price_calculation_l3905_390513

/-- Represents the price of a single book -/
def book_price : ℝ := 20

/-- Represents the number of books bought per month -/
def books_per_month : ℕ := 3

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents the total sale price of all books at the end of the year -/
def total_sale_price : ℝ := 500

/-- Represents the total loss incurred -/
def total_loss : ℝ := 220

theorem book_price_calculation : 
  book_price * (books_per_month * months_in_year) - total_sale_price = total_loss :=
sorry

end NUMINAMATH_CALUDE_book_price_calculation_l3905_390513


namespace NUMINAMATH_CALUDE_eraser_cost_tyler_eraser_cost_l3905_390583

/-- Calculates the cost of each eraser given Tyler's shopping scenario -/
theorem eraser_cost (initial_amount : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) 
  (eraser_count : ℕ) (remaining_amount : ℕ) : ℕ :=
  by
  sorry

/-- Proves that each eraser costs $4 in Tyler's specific scenario -/
theorem tyler_eraser_cost : eraser_cost 100 8 5 10 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_eraser_cost_tyler_eraser_cost_l3905_390583


namespace NUMINAMATH_CALUDE_valid_triangulations_l3905_390514

/-- A triangulation of a triangle is a division of the triangle into n smaller triangles
    such that no three vertices are collinear and each vertex belongs to the same number of segments -/
structure Triangulation :=
  (n : ℕ)  -- number of smaller triangles
  (no_collinear : Bool)  -- no three vertices are collinear
  (equal_vertex_degree : Bool)  -- each vertex belongs to the same number of segments

/-- The set of valid n values for triangulations -/
def ValidTriangulations : Set ℕ := {1, 3, 7, 19}

/-- Theorem stating that the only valid triangulations are those with n in ValidTriangulations -/
theorem valid_triangulations (t : Triangulation) :
  t.no_collinear ∧ t.equal_vertex_degree → t.n ∈ ValidTriangulations := by
  sorry

end NUMINAMATH_CALUDE_valid_triangulations_l3905_390514


namespace NUMINAMATH_CALUDE_magnitude_of_z_l3905_390509

theorem magnitude_of_z (i : ℂ) (h : i^2 = -1) : Complex.abs ((1 + i) / i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l3905_390509


namespace NUMINAMATH_CALUDE_unique_good_days_count_l3905_390571

/-- Represents the change factor for an ingot on a given day type -/
structure IngotFactor where
  good : ℝ
  bad : ℝ

/-- Calculates the final value of an ingot after a week -/
def finalValue (factor : IngotFactor) (goodDays : ℕ) : ℝ :=
  factor.good ^ goodDays * factor.bad ^ (7 - goodDays)

/-- The problem statement -/
theorem unique_good_days_count :
  ∃! goodDays : ℕ,
    goodDays ≤ 7 ∧
    let goldFactor : IngotFactor := { good := 1.3, bad := 0.7 }
    let silverFactor : IngotFactor := { good := 1.2, bad := 0.8 }
    (finalValue goldFactor goodDays < 1 ∧ finalValue silverFactor goodDays > 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_good_days_count_l3905_390571


namespace NUMINAMATH_CALUDE_bike_ride_time_l3905_390528

/-- Given a consistent bike riding speed where 1 mile takes 4 minutes,
    prove that the time required to ride 4.5 miles is 18 minutes. -/
theorem bike_ride_time (speed : ℝ) (distance_to_park : ℝ) : 
  speed = 1 / 4 →  -- Speed in miles per minute
  distance_to_park = 4.5 → -- Distance to park in miles
  distance_to_park / speed = 18 := by
  sorry

end NUMINAMATH_CALUDE_bike_ride_time_l3905_390528


namespace NUMINAMATH_CALUDE_sin_difference_inequality_l3905_390536

theorem sin_difference_inequality (a b : ℝ) :
  ((0 ≤ a ∧ a < b ∧ b ≤ π / 2) ∨ (π ≤ a ∧ a < b ∧ b ≤ 3 * π / 2)) →
  a - Real.sin a < b - Real.sin b :=
by sorry

end NUMINAMATH_CALUDE_sin_difference_inequality_l3905_390536


namespace NUMINAMATH_CALUDE_club_truncator_season_probability_l3905_390578

/-- Represents the possible outcomes of a soccer match -/
inductive MatchResult
| Win
| Lose
| Tie

/-- Represents the season results for Club Truncator -/
structure SeasonResult :=
  (wins : ℕ)
  (losses : ℕ)
  (ties : ℕ)

/-- The number of teams in the league -/
def numTeams : ℕ := 8

/-- The number of matches Club Truncator plays -/
def numMatches : ℕ := 7

/-- The probability of winning a single match -/
def winProb : ℚ := 2/5

/-- The probability of losing a single match -/
def loseProb : ℚ := 1/5

/-- The probability of tying a single match -/
def tieProb : ℚ := 2/5

/-- Checks if a season result has more wins than losses -/
def moreWinsThanLosses (result : SeasonResult) : Prop :=
  result.wins > result.losses

/-- The probability of Club Truncator finishing with more wins than losses -/
def probMoreWinsThanLosses : ℚ := 897/2187

theorem club_truncator_season_probability :
  probMoreWinsThanLosses = 897/2187 := by sorry

end NUMINAMATH_CALUDE_club_truncator_season_probability_l3905_390578


namespace NUMINAMATH_CALUDE_book_pages_proof_l3905_390518

/-- Proves that a book has 500 pages given specific writing and damage conditions -/
theorem book_pages_proof (total_pages : ℕ) : 
  (150 : ℕ) < total_pages →
  (0.8 * 0.7 * (total_pages - 150 : ℕ) : ℝ) = 196 →
  total_pages = 500 := by
sorry

end NUMINAMATH_CALUDE_book_pages_proof_l3905_390518


namespace NUMINAMATH_CALUDE_linear_equation_condition_l3905_390568

theorem linear_equation_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ k l m : ℝ, (a^2 - 4) * x^2 + (2 - 3*a) * x + (a + 1) * y + 3*a = k * x + l * y + m) → 
  (a = 2 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l3905_390568


namespace NUMINAMATH_CALUDE_equation_solution_l3905_390540

theorem equation_solution : ∃ x : ℝ, (6*x + 7)^2 * (3*x + 4) * (x + 1) = 6 :=
  have h1 : (6 * (-2/3) + 7)^2 * (3 * (-2/3) + 4) * (-2/3 + 1) = 6 := by sorry
  have h2 : (6 * (-5/3) + 7)^2 * (3 * (-5/3) + 4) * (-5/3 + 1) = 6 := by sorry
  ⟨-2/3, h1⟩

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3905_390540


namespace NUMINAMATH_CALUDE_black_dogs_count_l3905_390584

def total_dogs : Nat := 45
def brown_dogs : Nat := 20
def white_dogs : Nat := 10

theorem black_dogs_count : total_dogs - (brown_dogs + white_dogs) = 15 := by
  sorry

end NUMINAMATH_CALUDE_black_dogs_count_l3905_390584


namespace NUMINAMATH_CALUDE_min_sum_squares_l3905_390551

theorem min_sum_squares (a b c d e f g h : Int) : 
  a ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  b ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  c ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  d ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  e ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  f ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  g ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  h ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h →
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 98 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3905_390551


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3905_390599

def M : Set ℝ := {x | Real.sqrt x < 2}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem intersection_of_M_and_N :
  M ∩ N = {x | 1/3 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3905_390599


namespace NUMINAMATH_CALUDE_josh_work_hours_l3905_390572

/-- Proves that Josh works 8 hours a day given the problem conditions -/
theorem josh_work_hours :
  ∀ (h : ℝ),
  (20 * h * 9 + (20 * h - 40) * 4.5 = 1980) →
  h = 8 :=
by sorry

end NUMINAMATH_CALUDE_josh_work_hours_l3905_390572


namespace NUMINAMATH_CALUDE_largest_valid_number_l3905_390587

def is_valid_number (n : ℕ) : Prop :=
  (Nat.digits 10 n).length = 85 ∧
  (Nat.digits 10 n).sum = (Nat.digits 10 n).prod

def target_number : ℕ := 8322 * 10^81 + (10^81 - 1)

theorem largest_valid_number :
  is_valid_number target_number ∧
  ∀ m : ℕ, is_valid_number m → m ≤ target_number := by
  sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3905_390587


namespace NUMINAMATH_CALUDE_best_approximation_l3905_390532

-- Define the function f(x) = x^2 - 3x - 4.6
def f (x : ℝ) : ℝ := x^2 - 3*x - 4.6

-- Define the table of values
def table : List (ℝ × ℝ) := [
  (-1.13, 4.67),
  (-1.12, 4.61),
  (-1.11, 4.56),
  (-1.10, 4.51),
  (-1.09, 4.46),
  (-1.08, 4.41),
  (-1.07, 4.35)
]

-- Define the given options
def options : List ℝ := [-1.073, -1.089, -1.117, -1.123]

-- Theorem statement
theorem best_approximation :
  ∃ (x : ℝ), x ∈ options ∧
  ∀ (y : ℝ), y ∈ options → |f x| ≤ |f y| ∧
  x = -1.117 := by
  sorry

end NUMINAMATH_CALUDE_best_approximation_l3905_390532


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3905_390592

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ), 
  a = 30 → b = 40 → c^2 = a^2 + b^2 → c = 50 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3905_390592


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3905_390555

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 - x + 1 < 0) ↔ ∃ x : ℝ, x^2 - x + 1 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3905_390555


namespace NUMINAMATH_CALUDE_hilt_garden_border_rocks_l3905_390548

/-- The number of rocks Mrs. Hilt needs to complete her garden border -/
def total_rocks_needed (rocks_on_hand : ℕ) (additional_rocks_needed : ℕ) : ℕ :=
  rocks_on_hand + additional_rocks_needed

/-- Theorem: Mrs. Hilt needs 125 rocks in total to complete her garden border -/
theorem hilt_garden_border_rocks : 
  total_rocks_needed 64 61 = 125 := by
  sorry

end NUMINAMATH_CALUDE_hilt_garden_border_rocks_l3905_390548


namespace NUMINAMATH_CALUDE_road_trip_distance_l3905_390530

/-- Road trip problem -/
theorem road_trip_distance (total_time hours_driving friend_distance jenna_speed friend_speed : ℝ)
  (h1 : total_time = 10)
  (h2 : hours_driving = total_time - 1)
  (h3 : friend_distance = 100)
  (h4 : jenna_speed = 50)
  (h5 : friend_speed = 20) :
  jenna_speed * (hours_driving - friend_distance / friend_speed) = 200 :=
by sorry

end NUMINAMATH_CALUDE_road_trip_distance_l3905_390530


namespace NUMINAMATH_CALUDE_courtyard_width_main_theorem_l3905_390506

/-- Proves that the width of a rectangular courtyard is 16 meters -/
theorem courtyard_width : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (length width brick_length brick_width : ℝ) =>
    length = 30 ∧
    brick_length = 0.2 ∧
    brick_width = 0.1 ∧
    (length * width) / (brick_length * brick_width) = 24000 →
    width = 16

/-- Main theorem proof -/
theorem main_theorem : courtyard_width 30 16 0.2 0.1 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_width_main_theorem_l3905_390506


namespace NUMINAMATH_CALUDE_fraction_of_one_third_is_one_eighth_l3905_390561

theorem fraction_of_one_third_is_one_eighth (a b c d : ℚ) : 
  a = 1/3 → b = 1/8 → (b/a = c/d) → (c = 3 ∧ d = 8) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_one_third_is_one_eighth_l3905_390561


namespace NUMINAMATH_CALUDE_product_of_five_terms_l3905_390519

/-- A line passing through the origin with normal vector (3,1) -/
def line_l (x y : ℝ) : Prop := 3 * x + y = 0

/-- Sequence a_n where (a_{n+1}, a_n) lies on the line for all positive integers n -/
def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, line_l (a (n + 1)) (a n)

theorem product_of_five_terms (a : ℕ → ℝ) :
  sequence_property a → a 2 = 6 → a 1 * a 2 * a 3 * a 4 * a 5 = -32 := by
  sorry

end NUMINAMATH_CALUDE_product_of_five_terms_l3905_390519


namespace NUMINAMATH_CALUDE_problem_statement_l3905_390525

theorem problem_statement (p q r s : ℝ) (ω : ℂ) 
  (hp : p ≠ -1) (hq : q ≠ -1) (hr : r ≠ -1) (hs : s ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : 1/(p + ω) + 1/(q + ω) + 1/(r + ω) + 1/(s + ω) = 3/ω^2) :
  1/(p + 1) + 1/(q + 1) + 1/(r + 1) + 1/(s + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3905_390525


namespace NUMINAMATH_CALUDE_express_u_in_terms_of_f_and_g_l3905_390502

/-- Given functions u, f, and g satisfying certain conditions, 
    prove that u can be expressed in terms of f and g. -/
theorem express_u_in_terms_of_f_and_g 
  (u f g : ℝ → ℝ)
  (h1 : ∀ x, u (x + 1) + u (x - 1) = 2 * f x)
  (h2 : ∀ x, u (x + 4) + u (x - 4) = 2 * g x) :
  ∀ x, u x = g (x + 4) - f (x + 7) + f (x + 5) - f (x + 3) + f (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_express_u_in_terms_of_f_and_g_l3905_390502


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3905_390588

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

theorem vector_magnitude_problem (m : ℝ) :
  (‖a + b m‖ = ‖a - b m‖) → ‖a + 2 • (b m)‖ = 5 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l3905_390588


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3905_390508

theorem at_least_one_not_less_than_two 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_three : a + b + c = 3) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3905_390508


namespace NUMINAMATH_CALUDE_king_midas_gold_l3905_390589

theorem king_midas_gold (x : ℝ) (h : x > 1) : 
  let initial_gold := 1
  let spent_fraction := 1 / x
  let remaining_gold := initial_gold - spent_fraction * initial_gold
  let needed_fraction := (initial_gold - remaining_gold) / remaining_gold
  needed_fraction = 1 / (x - 1) := by
sorry

end NUMINAMATH_CALUDE_king_midas_gold_l3905_390589


namespace NUMINAMATH_CALUDE_frances_towel_weight_frances_towel_weight_is_240_ounces_l3905_390582

/-- Calculates the weight of Frances's towels in ounces given the conditions of the beach towel problem -/
theorem frances_towel_weight (mary_towel_count : ℕ) (total_weight_pounds : ℕ) : ℕ :=
  let frances_towel_count := mary_towel_count / 4
  let total_weight_ounces := total_weight_pounds * 16
  let mary_towel_weight_ounces := (total_weight_ounces / mary_towel_count) * mary_towel_count
  let frances_towel_weight_ounces := total_weight_ounces - mary_towel_weight_ounces
  frances_towel_weight_ounces

/-- Proves that Frances's towels weigh 240 ounces given the conditions of the beach towel problem -/
theorem frances_towel_weight_is_240_ounces : frances_towel_weight 24 60 = 240 := by
  sorry

end NUMINAMATH_CALUDE_frances_towel_weight_frances_towel_weight_is_240_ounces_l3905_390582


namespace NUMINAMATH_CALUDE_tan_315_degrees_l3905_390505

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l3905_390505


namespace NUMINAMATH_CALUDE_boys_in_class_l3905_390535

/-- Proves that in a class with a 3:4 ratio of girls to boys and 35 total students, the number of boys is 20. -/
theorem boys_in_class (total_students : ℕ) (girls_to_boys_ratio : ℚ) : total_students = 35 → girls_to_boys_ratio = 3 / 4 → ∃ (boys : ℕ), boys = 20 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l3905_390535


namespace NUMINAMATH_CALUDE_remaining_plums_l3905_390564

def gyuris_plums (initial : ℝ) (given_to_sungmin : ℝ) (given_to_dongju : ℝ) : ℝ :=
  initial - given_to_sungmin - given_to_dongju

theorem remaining_plums :
  gyuris_plums 1.6 0.8 0.3 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_remaining_plums_l3905_390564


namespace NUMINAMATH_CALUDE_quartic_sum_l3905_390533

theorem quartic_sum (f : ℝ → ℝ) :
  (∃ (a b c d : ℝ), ∀ x, f x = a*x^4 + b*x^3 + c*x^2 + d*x + (f 0)) →
  (f 1 = 10) →
  (f 2 = 20) →
  (f 3 = 30) →
  (f 10 + f (-6) = 8104) :=
by sorry

end NUMINAMATH_CALUDE_quartic_sum_l3905_390533


namespace NUMINAMATH_CALUDE_lawrence_county_kids_l3905_390501

theorem lawrence_county_kids (home_percentage : Real) (kids_at_home : ℕ) : 
  home_percentage = 0.607 →
  kids_at_home = 907611 →
  ∃ total_kids : ℕ, total_kids = (kids_at_home : Real) / home_percentage := by
    sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_l3905_390501


namespace NUMINAMATH_CALUDE_max_annual_profit_l3905_390552

/-- Represents the annual production quantity -/
def x : Type := { n : ℕ // n > 0 }

/-- Calculates the annual sales revenue in million yuan -/
def salesRevenue (x : x) : ℝ :=
  if x.val ≤ 20 then 33 * x.val - x.val^2 else 260

/-- Calculates the total annual investment in million yuan -/
def totalInvestment (x : x) : ℝ := 1 + 0.01 * x.val

/-- Calculates the annual profit in million yuan -/
def annualProfit (x : x) : ℝ := salesRevenue x - totalInvestment x

/-- Theorem stating the maximum annual profit and the production quantity that achieves it -/
theorem max_annual_profit :
  ∃ (x_max : x), 
    (∀ (x : x), annualProfit x ≤ annualProfit x_max) ∧
    (x_max.val = 16) ∧
    (annualProfit x_max = 156) := by sorry

end NUMINAMATH_CALUDE_max_annual_profit_l3905_390552


namespace NUMINAMATH_CALUDE_new_barbell_cost_l3905_390563

theorem new_barbell_cost (old_cost : ℝ) (percentage_increase : ℝ) : 
  old_cost = 250 → percentage_increase = 0.3 → 
  old_cost + old_cost * percentage_increase = 325 := by
  sorry

end NUMINAMATH_CALUDE_new_barbell_cost_l3905_390563


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3905_390520

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem: If a - b > 0 and ab < 0, then the point P(a,b) lies in the fourth quadrant -/
theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a - b > 0) (h2 : a * b < 0) :
  fourth_quadrant (Point.mk a b) := by
  sorry


end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3905_390520


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3905_390567

theorem sufficient_not_necessary (a b : ℝ) (h : a ≠ b) :
  (a > abs b → a^3 + b^3 > a^2*b + a*b^2) ∧
  ¬(a^3 + b^3 > a^2*b + a*b^2 → a > abs b) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3905_390567


namespace NUMINAMATH_CALUDE_solve_magazine_problem_l3905_390547

def magazine_problem (cost_price selling_price gain : ℚ) : Prop :=
  ∃ (num_magazines : ℕ), 
    (selling_price - cost_price) * num_magazines = gain ∧
    num_magazines > 0

theorem solve_magazine_problem : 
  magazine_problem 3 3.5 5 → ∃ (num_magazines : ℕ), num_magazines = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_magazine_problem_l3905_390547


namespace NUMINAMATH_CALUDE_earnings_difference_theorem_l3905_390526

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  inv_ratio_a : ℕ
  inv_ratio_b : ℕ
  inv_ratio_c : ℕ
  ret_ratio_a : ℕ
  ret_ratio_b : ℕ
  ret_ratio_c : ℕ

/-- Calculates the earnings difference between investors b and a -/
def earnings_difference (data : InvestmentData) (total_earnings : ℕ) : ℕ :=
  let total_ratio := data.inv_ratio_a * data.ret_ratio_a + 
                     data.inv_ratio_b * data.ret_ratio_b + 
                     data.inv_ratio_c * data.ret_ratio_c
  let unit_earning := total_earnings / total_ratio
  (data.inv_ratio_b * data.ret_ratio_b - data.inv_ratio_a * data.ret_ratio_a) * unit_earning

/-- Theorem: Given the investment ratios 3:4:5, return ratios 6:5:4, and total earnings 10150,
    the earnings difference between b and a is 350 -/
theorem earnings_difference_theorem : 
  let data : InvestmentData := {
    inv_ratio_a := 3, inv_ratio_b := 4, inv_ratio_c := 5,
    ret_ratio_a := 6, ret_ratio_b := 5, ret_ratio_c := 4
  }
  earnings_difference data 10150 = 350 := by
  sorry


end NUMINAMATH_CALUDE_earnings_difference_theorem_l3905_390526


namespace NUMINAMATH_CALUDE_parallel_lines_k_equals_two_l3905_390504

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The slope-intercept form of the line 2x - y + 2 = 0 -/
def line1_slope_intercept (x y : ℝ) : Prop := y = 2 * x + 2

/-- The slope-intercept form of the line y = kx + 1 -/
def line2_slope_intercept (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

theorem parallel_lines_k_equals_two :
  (∀ x y : ℝ, 2 * x - y + 2 = 0 ↔ y = k * x + 1) → k = 2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_equals_two_l3905_390504


namespace NUMINAMATH_CALUDE_power_sum_equality_l3905_390580

theorem power_sum_equality : (-1 : ℤ) ^ (6^2) + (1 : ℤ) ^ (3^3) = 2 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3905_390580


namespace NUMINAMATH_CALUDE_complex_fraction_problem_l3905_390500

theorem complex_fraction_problem (x y : ℂ) (k : ℝ) 
  (h : (x + k * y) / (x - k * y) + (x - k * y) / (x + k * y) = 1) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 41 / 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_problem_l3905_390500


namespace NUMINAMATH_CALUDE_inequality_proof_l3905_390596

noncomputable def a : ℝ := 1 + Real.tan (-0.2)
noncomputable def b : ℝ := Real.log (0.8 * Real.exp 1)
noncomputable def c : ℝ := 1 / Real.exp 0.2

theorem inequality_proof : c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3905_390596


namespace NUMINAMATH_CALUDE_negative_division_equality_l3905_390516

theorem negative_division_equality : (-81) / (-9) = 9 := by
  sorry

end NUMINAMATH_CALUDE_negative_division_equality_l3905_390516


namespace NUMINAMATH_CALUDE_equation_solutions_count_l3905_390573

theorem equation_solutions_count : 
  let count := Finset.filter (fun k => 
    k % 2 = 1 ∧ 
    (Finset.filter (fun p : ℕ × ℕ => 
      let (m, n) := p
      2^(4*m^2) + 2^(m^2 - n^2 + 4) = 2^(k + 4) + 2^(3*m^2 + n^2 + k) ∧ 
      m > 0 ∧ n > 0
    ) (Finset.product (Finset.range 101) (Finset.range 101))).card = 2
  ) (Finset.range 101)
  count.card = 18 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l3905_390573


namespace NUMINAMATH_CALUDE_greatest_five_digit_integer_l3905_390566

def reverse_digits (n : Nat) : Nat :=
  -- Implementation of reverse_digits function
  sorry

def is_five_digit (n : Nat) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

theorem greatest_five_digit_integer (p : Nat) : 
  is_five_digit p ∧ 
  is_five_digit (reverse_digits p) ∧
  p % 63 = 0 ∧
  (reverse_digits p) % 63 = 0 ∧
  p % 11 = 0 →
  p ≤ 99729 :=
sorry

end NUMINAMATH_CALUDE_greatest_five_digit_integer_l3905_390566


namespace NUMINAMATH_CALUDE_allocation_five_to_three_l3905_390531

/-- The number of ways to allocate n identical objects to k distinct groups,
    with each group receiving at least one object -/
def allocations (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 6 ways to allocate 5 identical objects to 3 distinct groups,
    with each group receiving at least one object -/
theorem allocation_five_to_three :
  allocations 5 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_allocation_five_to_three_l3905_390531


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3905_390534

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 2 + 3 * Complex.I → z = 3 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3905_390534


namespace NUMINAMATH_CALUDE_sqrt_18_greater_than_pi_l3905_390585

theorem sqrt_18_greater_than_pi : Real.sqrt 18 > Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_greater_than_pi_l3905_390585


namespace NUMINAMATH_CALUDE_safety_gear_to_test_tube_ratio_l3905_390590

def total_budget : ℚ := 325
def flask_cost : ℚ := 150
def remaining_budget : ℚ := 25

def test_tube_cost : ℚ := (2/3) * flask_cost

def total_spent : ℚ := total_budget - remaining_budget

def safety_gear_cost : ℚ := total_spent - flask_cost - test_tube_cost

theorem safety_gear_to_test_tube_ratio :
  safety_gear_cost / test_tube_cost = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_safety_gear_to_test_tube_ratio_l3905_390590


namespace NUMINAMATH_CALUDE_djibo_age_proof_l3905_390595

/-- Djibo's current age -/
def djibo_age : ℕ := 17

/-- Djibo's sister's current age -/
def sister_age : ℕ := 28

/-- Sum of Djibo's and his sister's ages 5 years ago -/
def sum_ages_5_years_ago : ℕ := 35

theorem djibo_age_proof :
  djibo_age = 17 ∧
  sister_age = 28 ∧
  (djibo_age - 5) + (sister_age - 5) = sum_ages_5_years_ago :=
by sorry

end NUMINAMATH_CALUDE_djibo_age_proof_l3905_390595


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3905_390586

theorem inequality_system_solution_set :
  let S := {x : ℝ | 3 * x - 1 ≥ x + 1 ∧ x + 4 > 4 * x - 2}
  S = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3905_390586


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3905_390597

theorem polynomial_expansion (x : ℝ) :
  (3*x^2 + 4*x + 8)*(x - 2) - (x - 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x - 2)*(x + 6) =
  6*x^3 - 4*x^2 - 26*x + 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3905_390597


namespace NUMINAMATH_CALUDE_common_tangent_l3905_390537

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C2 (x y : ℝ) : Prop := (x-3)^2 + (y-4)^2 = 16

-- Define the line x = -1
def tangent_line (x : ℝ) : Prop := x = -1

-- Theorem statement
theorem common_tangent :
  (∀ x y : ℝ, C1 x y → tangent_line x → (x^2 + y^2 = 1 ∧ x = -1)) ∧
  (∀ x y : ℝ, C2 x y → tangent_line x → ((x-3)^2 + (y-4)^2 = 16 ∧ x = -1)) :=
sorry

end NUMINAMATH_CALUDE_common_tangent_l3905_390537


namespace NUMINAMATH_CALUDE_correct_divisor_l3905_390594

theorem correct_divisor (incorrect_divisor : ℕ) (incorrect_quotient : ℕ) (correct_quotient : ℕ) :
  incorrect_divisor = 63 →
  incorrect_quotient = 24 →
  correct_quotient = 42 →
  (incorrect_divisor * incorrect_quotient) / correct_quotient = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_l3905_390594


namespace NUMINAMATH_CALUDE_expression_value_l3905_390577

theorem expression_value (a b c : ℤ) (ha : a = 8) (hb : b = 10) (hc : c = 3) :
  (2 * a - (b - 2 * c)) - ((2 * a - b) - 2 * c) + 3 * (a - c) = 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3905_390577


namespace NUMINAMATH_CALUDE_f_min_at_300_l3905_390522

/-- The quadratic expression we're minimizing -/
def f (x : ℝ) : ℝ := x^2 - 600*x + 369

/-- The theorem stating that f(x) takes its minimum value when x = 300 -/
theorem f_min_at_300 : 
  ∀ x : ℝ, f x ≥ f 300 := by sorry

end NUMINAMATH_CALUDE_f_min_at_300_l3905_390522


namespace NUMINAMATH_CALUDE_problem_1_l3905_390560

theorem problem_1 : 23 * (-5) - (-3) / (3/108) = -7 := by sorry

end NUMINAMATH_CALUDE_problem_1_l3905_390560


namespace NUMINAMATH_CALUDE_unique_triple_product_sum_l3905_390521

theorem unique_triple_product_sum : 
  ∃! (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    a * b = c ∧ b * c = a ∧ c * a = b ∧ a + b + c = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_product_sum_l3905_390521


namespace NUMINAMATH_CALUDE_expand_product_l3905_390511

theorem expand_product (y : ℝ) : 3 * (y - 4) * (y + 9) = 3 * y^2 + 15 * y - 108 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3905_390511


namespace NUMINAMATH_CALUDE_m_range_l3905_390553

theorem m_range : 
  let m : ℝ := (-Real.sqrt 3 / 3) * (-2 * Real.sqrt 21)
  5 < m ∧ m < 6 := by
sorry

end NUMINAMATH_CALUDE_m_range_l3905_390553


namespace NUMINAMATH_CALUDE_charge_with_interest_after_one_year_l3905_390575

/-- Calculates the amount owed after one year given an initial charge and simple annual interest rate -/
def amount_owed_after_one_year (initial_charge : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_charge * (1 + interest_rate)

/-- Theorem stating that a $35 charge with 7% simple annual interest results in $37.45 owed after one year -/
theorem charge_with_interest_after_one_year :
  let initial_charge : ℝ := 35
  let interest_rate : ℝ := 0.07
  amount_owed_after_one_year initial_charge interest_rate = 37.45 := by
  sorry

#eval amount_owed_after_one_year 35 0.07

end NUMINAMATH_CALUDE_charge_with_interest_after_one_year_l3905_390575


namespace NUMINAMATH_CALUDE_gcd_84_210_l3905_390598

theorem gcd_84_210 : Nat.gcd 84 210 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_84_210_l3905_390598


namespace NUMINAMATH_CALUDE_total_apples_is_340_l3905_390581

/-- The number of apples Kylie picked -/
def kylie_apples : ℕ := 66

/-- The number of apples Kayla picked -/
def kayla_apples : ℕ := 274

/-- The relationship between Kayla's and Kylie's apples -/
axiom kayla_kylie_relation : kayla_apples = 4 * kylie_apples + 10

/-- The total number of apples picked by Kylie and Kayla -/
def total_apples : ℕ := kylie_apples + kayla_apples

/-- Theorem: The total number of apples picked by Kylie and Kayla is 340 -/
theorem total_apples_is_340 : total_apples = 340 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_340_l3905_390581


namespace NUMINAMATH_CALUDE_carolyn_shared_marbles_l3905_390543

/-- The number of marbles Carolyn shared with Diana -/
def marbles_shared (initial_marbles final_marbles : ℕ) : ℕ :=
  initial_marbles - final_marbles

/-- Theorem stating that Carolyn shared 42 marbles with Diana -/
theorem carolyn_shared_marbles :
  let initial_marbles : ℕ := 47
  let final_marbles : ℕ := 5
  marbles_shared initial_marbles final_marbles = 42 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_shared_marbles_l3905_390543


namespace NUMINAMATH_CALUDE_count_solutions_eq_4n_l3905_390591

/-- The number of integer solutions (x, y) for |x| + |y| = n -/
def count_solutions (n : ℕ) : ℕ :=
  4 * n

/-- Theorem: For any positive integer n, the number of integer solutions (x, y) 
    satisfying |x| + |y| = n is equal to 4n -/
theorem count_solutions_eq_4n (n : ℕ) (hn : n > 0) : 
  count_solutions n = 4 * n := by sorry

end NUMINAMATH_CALUDE_count_solutions_eq_4n_l3905_390591


namespace NUMINAMATH_CALUDE_inequality_proof_l3905_390546

theorem inequality_proof (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c > 0) :
  a + b ≤ 2 * c ∧ 2 * c ≤ 3 * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3905_390546


namespace NUMINAMATH_CALUDE_remainder_theorem_l3905_390557

/-- The dividend polynomial -/
def P (x : ℝ) : ℝ := x^100 - 2*x^51 + 1

/-- The divisor polynomial -/
def D (x : ℝ) : ℝ := x^2 - 1

/-- The proposed remainder -/
def R (x : ℝ) : ℝ := -2*x + 2

/-- Theorem stating that R is the remainder of P divided by D -/
theorem remainder_theorem : 
  ∃ Q : ℝ → ℝ, ∀ x : ℝ, P x = D x * Q x + R x :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3905_390557


namespace NUMINAMATH_CALUDE_dog_bones_found_l3905_390512

/-- Given a dog initially has 15 bones and ends up with 23 bones, 
    prove that the number of bones found is 23 - 15. -/
theorem dog_bones_found (initial_bones final_bones : ℕ) 
  (h1 : initial_bones = 15) 
  (h2 : final_bones = 23) : 
  final_bones - initial_bones = 23 - 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_bones_found_l3905_390512


namespace NUMINAMATH_CALUDE_remainder_problem_l3905_390570

theorem remainder_problem (y : ℤ) : 
  y % 23 = 19 → y % 276 = 180 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l3905_390570


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l3905_390549

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem ninth_term_of_arithmetic_sequence :
  ∀ (a₁ d : ℚ),
    a₁ = 2/3 →
    arithmetic_sequence a₁ d 17 = 3/2 →
    arithmetic_sequence a₁ d 9 = 13/12 :=
by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l3905_390549


namespace NUMINAMATH_CALUDE_prime_sum_squares_l3905_390538

theorem prime_sum_squares (p q : ℕ) : 
  Prime p → Prime q → 
  ∃ (x y : ℕ), x^2 = p + q ∧ y^2 = p + 7*q → 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l3905_390538
