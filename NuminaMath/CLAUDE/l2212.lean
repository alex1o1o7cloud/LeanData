import Mathlib

namespace intersection_of_logarithmic_functions_l2212_221253

theorem intersection_of_logarithmic_functions :
  ∃! x : ℝ, x > 0 ∧ 2 * Real.log x = Real.log (3 * x) :=
sorry

end intersection_of_logarithmic_functions_l2212_221253


namespace solve_equation_l2212_221235

theorem solve_equation (r : ℚ) : 3 * (r - 7) = 4 * (2 - 2 * r) + 4 → r = 3 := by
  sorry

end solve_equation_l2212_221235


namespace dog_food_cost_l2212_221241

-- Define the given constants
def puppy_cost : ℚ := 10
def days : ℕ := 21
def food_per_day : ℚ := 1/3
def food_per_bag : ℚ := 7/2
def total_cost : ℚ := 14

-- Define the theorem
theorem dog_food_cost :
  let total_food := days * food_per_day
  let bags_needed := total_food / food_per_bag
  let food_cost := total_cost - puppy_cost
  food_cost / bags_needed = 2 := by
sorry

end dog_food_cost_l2212_221241


namespace second_stop_off_is_two_l2212_221224

/-- Represents the number of passengers on the trolley at various stages --/
structure TrolleyPassengers where
  initial : Nat
  second_stop_off : Nat
  second_stop_on : Nat
  third_stop_off : Nat
  third_stop_on : Nat
  final : Nat

/-- The trolley problem with given conditions --/
def trolleyProblem : TrolleyPassengers where
  initial := 10
  second_stop_off := 2  -- This is what we want to prove
  second_stop_on := 20  -- Twice the initial number
  third_stop_off := 18
  third_stop_on := 2
  final := 12

/-- Theorem stating that the number of people who got off at the second stop is 2 --/
theorem second_stop_off_is_two (t : TrolleyPassengers) : 
  t.initial = 10 ∧ 
  t.second_stop_on = 2 * t.initial ∧ 
  t.third_stop_off = 18 ∧ 
  t.third_stop_on = 2 ∧ 
  t.final = 12 →
  t.second_stop_off = 2 := by
  sorry

#check second_stop_off_is_two trolleyProblem

end second_stop_off_is_two_l2212_221224


namespace inequality_proof_l2212_221217

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / (x^2 + y^2)) + (1 / x^2) + (1 / y^2) ≥ 10 / ((x + y)^2) := by
  sorry

end inequality_proof_l2212_221217


namespace units_digit_difference_l2212_221220

/-- Returns the units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Predicate for a natural number being even -/
def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

theorem units_digit_difference (p : ℕ) 
  (h1 : p > 0)
  (h2 : isEven p)
  (h3 : unitsDigit p > 0)
  (h4 : unitsDigit (p + 1) = 7) :
  unitsDigit (p^3) - unitsDigit (p^2) = 0 := by
sorry

end units_digit_difference_l2212_221220


namespace largest_non_60multiple_composite_sum_l2212_221295

/-- A positive integer is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- A function that represents the sum of a positive integral multiple of 60 and a positive composite integer -/
def SumOf60MultipleAndComposite (k m : ℕ) : ℕ := 60 * (k + 1) + m

theorem largest_non_60multiple_composite_sum :
  ∀ n : ℕ, n > 5 →
    ∃ k m : ℕ, IsComposite m ∧ n = SumOf60MultipleAndComposite k m :=
by sorry

end largest_non_60multiple_composite_sum_l2212_221295


namespace kellys_snacks_l2212_221245

theorem kellys_snacks (peanuts raisins : ℝ) (h1 : peanuts = 0.1) (h2 : raisins = 0.4) :
  peanuts + raisins = 0.5 := by
  sorry

end kellys_snacks_l2212_221245


namespace exponent_division_l2212_221254

theorem exponent_division (a : ℝ) : a ^ 3 / a = a ^ 2 := by
  sorry

end exponent_division_l2212_221254


namespace proportional_function_and_value_l2212_221265

/-- Given that y+3 is directly proportional to x and y=7 when x=2, prove:
    1. The function expression for y in terms of x
    2. The value of y when x = -1/2 -/
theorem proportional_function_and_value (y : ℝ → ℝ) (k : ℝ) 
    (h1 : ∀ x, y x + 3 = k * x)  -- y+3 is directly proportional to x
    (h2 : y 2 = 7)  -- when x=2, y=7
    : (∀ x, y x = 5*x - 3) ∧ (y (-1/2) = -11/2) := by
  sorry

end proportional_function_and_value_l2212_221265


namespace not_p_necessary_not_sufficient_for_not_q_l2212_221286

-- Define the conditions p and q
def p (a : ℝ) : Prop := a < 0
def q (a : ℝ) : Prop := a^2 > a

-- Theorem statement
theorem not_p_necessary_not_sufficient_for_not_q :
  (∀ a, ¬(q a) → ¬(p a)) ∧ 
  (∃ a, ¬(p a) ∧ q a) :=
by sorry

end not_p_necessary_not_sufficient_for_not_q_l2212_221286


namespace cos_alpha_plus_pi_implies_sin_alpha_plus_three_halves_pi_l2212_221282

theorem cos_alpha_plus_pi_implies_sin_alpha_plus_three_halves_pi 
  (α : Real) 
  (h : Real.cos (α + Real.pi) = -2/3) : 
  Real.sin (α + 3/2 * Real.pi) = -2/3 := by
  sorry

end cos_alpha_plus_pi_implies_sin_alpha_plus_three_halves_pi_l2212_221282


namespace movie_length_after_cut_l2212_221256

/-- The final length of a movie after cutting a scene -/
def final_movie_length (original_length scene_cut : ℕ) : ℕ :=
  original_length - scene_cut

/-- Theorem: The final length of the movie is 52 minutes -/
theorem movie_length_after_cut :
  final_movie_length 60 8 = 52 := by
  sorry

end movie_length_after_cut_l2212_221256


namespace jim_journey_l2212_221218

theorem jim_journey (total_journey : ℕ) (remaining_miles : ℕ) 
  (h1 : total_journey = 1200)
  (h2 : remaining_miles = 558) :
  total_journey - remaining_miles = 642 := by
sorry

end jim_journey_l2212_221218


namespace number_equality_l2212_221211

theorem number_equality : ∃ x : ℝ, x / 0.144 = 14.4 / 0.0144 ∧ x = 144 := by
  sorry

end number_equality_l2212_221211


namespace proportion_solution_l2212_221236

theorem proportion_solution :
  ∀ x : ℝ, (0.75 / x = 5 / 6) → x = 0.9 := by
  sorry

end proportion_solution_l2212_221236


namespace sequence_not_ap_gp_l2212_221268

theorem sequence_not_ap_gp (a b c d n : ℝ) : 
  a < b ∧ b < c ∧ a > 1 ∧ 
  b = a + d ∧ c = a + 2*d ∧ 
  n > 1 →
  ¬(∃r : ℝ, (Real.log n / Real.log b - Real.log n / Real.log a = r) ∧ 
             (Real.log n / Real.log c - Real.log n / Real.log b = r)) ∧
  ¬(∃r : ℝ, (Real.log n / Real.log b) / (Real.log n / Real.log a) = r ∧ 
             (Real.log n / Real.log c) / (Real.log n / Real.log b) = r) :=
by sorry

end sequence_not_ap_gp_l2212_221268


namespace fraction_simplification_l2212_221239

theorem fraction_simplification (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) :
  1 / x - 1 / (x - 1) = -1 / (x * (x - 1)) := by
  sorry

end fraction_simplification_l2212_221239


namespace least_number_of_marbles_eight_forty_satisfies_least_number_is_eight_forty_l2212_221200

theorem least_number_of_marbles (n : ℕ) : n > 0 ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n → n ≥ 840 := by
  sorry

theorem eight_forty_satisfies (n : ℕ) : 
  3 ∣ 840 ∧ 4 ∣ 840 ∧ 5 ∣ 840 ∧ 7 ∣ 840 ∧ 8 ∣ 840 := by
  sorry

theorem least_number_is_eight_forty : 
  ∃ (n : ℕ), n > 0 ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n ∧
  ∀ (m : ℕ), (m > 0 ∧ 3 ∣ m ∧ 4 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 8 ∣ m) → n ≤ m := by
  sorry

end least_number_of_marbles_eight_forty_satisfies_least_number_is_eight_forty_l2212_221200


namespace distinct_solutions_condition_l2212_221274

theorem distinct_solutions_condition (a x y : ℝ) : 
  x ≠ y → x = a - y^2 → y = a - x^2 → a > 3/4 := by
  sorry

end distinct_solutions_condition_l2212_221274


namespace target_score_proof_l2212_221294

theorem target_score_proof (a b c : ℕ) 
  (h1 : 2 * b + c = 29) 
  (h2 : 2 * a + c = 43) : 
  a + b + c = 36 := by
sorry

end target_score_proof_l2212_221294


namespace triangle_problem_l2212_221251

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition: 2c - a = 2b*cos(A) -/
def satisfiesCondition (t : Triangle) : Prop :=
  2 * t.c - t.a = 2 * t.b * Real.cos t.A

/-- Theorem stating the two parts of the problem -/
theorem triangle_problem (t : Triangle) 
  (h : satisfiesCondition t) : 
  t.B = π / 3 ∧ 
  (t.a = 2 ∧ t.b = Real.sqrt 7 → t.c = 3) :=
sorry

end triangle_problem_l2212_221251


namespace perpendicular_lines_a_equals_one_l2212_221206

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) : ℝ × ℝ → Prop := λ p => a * p.1 + p.2 + 2 = 0
def l₂ (a : ℝ) : ℝ × ℝ → Prop := λ p => p.1 + (a - 2) * p.2 + 1 = 0

-- Define perpendicularity of lines
def perpendicular (f g : ℝ × ℝ → Prop) : Prop :=
  ∃ (m₁ m₂ : ℝ), (∀ p, f p ↔ p.2 = m₁ * p.1 + 0) ∧
                 (∀ p, g p ↔ p.2 = m₂ * p.1 + 0) ∧
                 m₁ * m₂ = -1

-- State the theorem
theorem perpendicular_lines_a_equals_one :
  perpendicular (l₁ a) (l₂ a) → a = 1 := by
  sorry

end perpendicular_lines_a_equals_one_l2212_221206


namespace initial_cars_correct_l2212_221249

/-- Represents the car dealership scenario -/
structure CarDealership where
  initialCars : ℕ
  initialSilverPercent : ℚ
  newShipment : ℕ
  newNonSilverPercent : ℚ
  finalSilverPercent : ℚ

/-- The car dealership scenario with given conditions -/
def scenario : CarDealership :=
  { initialCars := 360,  -- This is what we want to prove
    initialSilverPercent := 15 / 100,
    newShipment := 80,
    newNonSilverPercent := 30 / 100,
    finalSilverPercent := 25 / 100 }

/-- Theorem stating that the initial number of cars is correct given the conditions -/
theorem initial_cars_correct (d : CarDealership) : 
  d.initialCars = scenario.initialCars →
  d.initialSilverPercent = scenario.initialSilverPercent →
  d.newShipment = scenario.newShipment →
  d.newNonSilverPercent = scenario.newNonSilverPercent →
  d.finalSilverPercent = scenario.finalSilverPercent →
  d.finalSilverPercent * (d.initialCars + d.newShipment) = 
    d.initialSilverPercent * d.initialCars + (1 - d.newNonSilverPercent) * d.newShipment :=
by sorry

#check initial_cars_correct

end initial_cars_correct_l2212_221249


namespace largest_reciprocal_l2212_221202

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/8 → b = 3/4 → c = 1/2 → d = 10 → e = -2 →
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end largest_reciprocal_l2212_221202


namespace cake_muffin_buyers_l2212_221271

theorem cake_muffin_buyers (cake_buyers : ℕ) (muffin_buyers : ℕ) (both_buyers : ℕ) 
  (prob_neither : ℚ) (h1 : cake_buyers = 50) (h2 : muffin_buyers = 40) 
  (h3 : both_buyers = 19) (h4 : prob_neither = 29/100) : 
  ∃ total_buyers : ℕ, 
    total_buyers = 100 ∧ 
    (cake_buyers + muffin_buyers - both_buyers : ℚ) + prob_neither * total_buyers = total_buyers :=
by sorry

end cake_muffin_buyers_l2212_221271


namespace jack_final_plate_count_l2212_221247

/-- Represents the number of plates Jack has of each type and the total number of plates --/
structure PlateCount where
  flower : ℕ
  checked : ℕ
  polkaDot : ℕ
  total : ℕ

/-- Calculates the final number of plates Jack has --/
def finalPlateCount (initial : PlateCount) : PlateCount :=
  let newPolkaDot := 2 * initial.checked
  let newFlower := initial.flower - 1
  { flower := newFlower
  , checked := initial.checked
  , polkaDot := newPolkaDot
  , total := newFlower + initial.checked + newPolkaDot
  }

/-- Theorem stating that Jack ends up with 27 plates --/
theorem jack_final_plate_count :
  let initial := { flower := 4, checked := 8, polkaDot := 0, total := 12 : PlateCount }
  (finalPlateCount initial).total = 27 := by
  sorry

end jack_final_plate_count_l2212_221247


namespace number_of_boys_l2212_221269

theorem number_of_boys (total_kids : ℕ) (girls : ℕ) (boys : ℕ) :
  total_kids = 9 →
  girls = 3 →
  total_kids = girls + boys →
  boys = 6 := by
sorry

end number_of_boys_l2212_221269


namespace license_plate_difference_l2212_221281

def florida_combinations : ℕ := 26^4 * 10^3
def georgia_combinations : ℕ := 26^3 * 10^3

theorem license_plate_difference : 
  florida_combinations - georgia_combinations = 439400000 := by
  sorry

end license_plate_difference_l2212_221281


namespace lending_period_is_one_year_l2212_221277

/-- Proves that the lending period is 1 year given the problem conditions --/
theorem lending_period_is_one_year 
  (principal : ℝ)
  (borrowing_rate : ℝ)
  (lending_rate : ℝ)
  (annual_gain : ℝ)
  (h1 : principal = 5000)
  (h2 : borrowing_rate = 0.04)
  (h3 : lending_rate = 0.05)
  (h4 : annual_gain = 50)
  : ∃ t : ℝ, t = 1 ∧ principal * lending_rate * t - principal * borrowing_rate * t = annual_gain :=
sorry

end lending_period_is_one_year_l2212_221277


namespace expected_small_supermarkets_l2212_221246

/-- Represents the types of supermarkets --/
inductive SupermarketType
| Small
| Medium
| Large

/-- Represents the count of each type of supermarket --/
structure SupermarketCounts where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Represents the sample size for each type of supermarket --/
structure SampleSizes where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the expected number of small supermarkets in a subsample --/
def expectedSmallInSubsample (counts : SupermarketCounts) (sample : SampleSizes) (subsampleSize : ℕ) : ℚ :=
  (sample.small : ℚ) / ((sample.small + sample.medium + sample.large) : ℚ) * subsampleSize

/-- Theorem stating the expected number of small supermarkets in the subsample --/
theorem expected_small_supermarkets 
  (counts : SupermarketCounts)
  (sample : SampleSizes)
  (h1 : counts.small = 72 ∧ counts.medium = 24 ∧ counts.large = 12)
  (h2 : sample.small + sample.medium + sample.large = 9)
  (h3 : sample.small = 6 ∧ sample.medium = 2 ∧ sample.large = 1)
  : expectedSmallInSubsample counts sample 3 = 2 := by
  sorry

end expected_small_supermarkets_l2212_221246


namespace video_library_disk_space_l2212_221259

/-- Calculates the average disk space per hour of video in a library, rounded to the nearest integer -/
def averageDiskSpacePerHour (totalDays : ℕ) (totalSpace : ℕ) : ℕ :=
  let totalHours : ℕ := totalDays * 24
  let exactAverage : ℚ := totalSpace / totalHours
  (exactAverage + 1/2).floor.toNat

/-- Theorem stating that for a 15-day video library occupying 24000 MB, 
    the average disk space per hour rounded to the nearest integer is 67 MB -/
theorem video_library_disk_space :
  averageDiskSpacePerHour 15 24000 = 67 := by
  sorry

end video_library_disk_space_l2212_221259


namespace trick_deck_cost_l2212_221250

/-- The cost of a trick deck satisfies the given conditions -/
theorem trick_deck_cost : ∃ (cost : ℕ), 
  6 * cost + 2 * cost = 64 ∧ cost = 8 := by
  sorry

end trick_deck_cost_l2212_221250


namespace three_common_points_l2212_221232

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Function to check if a point satisfies the first equation -/
def satisfiesEq1 (p : Point) : Prop :=
  (2 * p.x - 3 * p.y + 6) * (5 * p.x + 2 * p.y - 10) = 0

/-- Function to check if a point satisfies the second equation -/
def satisfiesEq2 (p : Point) : Prop :=
  (p.x - 2 * p.y + 1) * (3 * p.x - 4 * p.y + 8) = 0

/-- The main theorem stating that there are exactly 3 common points -/
theorem three_common_points :
  ∃ (p1 p2 p3 : Point),
    satisfiesEq1 p1 ∧ satisfiesEq2 p1 ∧
    satisfiesEq1 p2 ∧ satisfiesEq2 p2 ∧
    satisfiesEq1 p3 ∧ satisfiesEq2 p3 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    ∀ (p : Point), satisfiesEq1 p ∧ satisfiesEq2 p → p = p1 ∨ p = p2 ∨ p = p3 :=
sorry


end three_common_points_l2212_221232


namespace larger_number_problem_l2212_221288

theorem larger_number_problem (x y : ℝ) : 3 * y = 4 * x → y - x = 8 → y = 32 := by
  sorry

end larger_number_problem_l2212_221288


namespace volume_ratio_of_rotated_triangle_l2212_221298

/-- Given a right-angled triangle with perpendicular sides of lengths a and b,
    the ratio of the volume of the solid formed by rotating around side a
    to the volume of the solid formed by rotating around side b is b : a. -/
theorem volume_ratio_of_rotated_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (1 / 3 * π * b^2 * a) / (1 / 3 * π * a^2 * b) = b / a :=
sorry

end volume_ratio_of_rotated_triangle_l2212_221298


namespace system_of_equations_l2212_221272

theorem system_of_equations (x y c d : ℝ) : 
  x ≠ 0 →
  y ≠ 0 →
  d ≠ 0 →
  8 * x - 5 * y = c →
  10 * y - 12 * x = d →
  c / d = -2 / 3 := by
sorry

end system_of_equations_l2212_221272


namespace solve_for_C_l2212_221257

theorem solve_for_C : ∃ C : ℝ, (2 * C - 3 = 11) ∧ (C = 7) := by sorry

end solve_for_C_l2212_221257


namespace simplify_expression_l2212_221221

theorem simplify_expression (x : ℝ) (h : -1 < x ∧ x < 3) :
  Real.sqrt ((x - 3)^2) + |x + 1| = 4 := by
  sorry

end simplify_expression_l2212_221221


namespace magnitude_of_8_minus_15i_l2212_221231

theorem magnitude_of_8_minus_15i : Complex.abs (8 - 15*I) = 17 := by
  sorry

end magnitude_of_8_minus_15i_l2212_221231


namespace vertical_shift_theorem_l2212_221222

/-- The original line function -/
def original_line (x : ℝ) : ℝ := 2 * x

/-- The vertical shift amount -/
def shift : ℝ := 5

/-- The resulting line after vertical shift -/
def shifted_line (x : ℝ) : ℝ := original_line x + shift

theorem vertical_shift_theorem :
  ∀ x : ℝ, shifted_line x = 2 * x + 5 := by sorry

end vertical_shift_theorem_l2212_221222


namespace average_of_six_numbers_l2212_221238

theorem average_of_six_numbers (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 6.2)
  (h2 : (c + d) / 2 = 6.1)
  (h3 : (e + f) / 2 = 6.9) :
  (a + b + c + d + e + f) / 6 = 6.4 := by
  sorry

end average_of_six_numbers_l2212_221238


namespace gcd_1729_1768_l2212_221262

theorem gcd_1729_1768 : Nat.gcd 1729 1768 = 13 := by
  sorry

end gcd_1729_1768_l2212_221262


namespace problem_statement_l2212_221234

theorem problem_statement : 
  (∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≥ 0) ∧ ¬(∀ a b : ℝ, a < b → 1/a > 1/b) := by
  sorry

end problem_statement_l2212_221234


namespace negative_x_squared_times_x_cubed_l2212_221283

theorem negative_x_squared_times_x_cubed (x : ℝ) : (-x^2) * x^3 = -x^5 := by
  sorry

end negative_x_squared_times_x_cubed_l2212_221283


namespace complete_square_quadratic_l2212_221261

theorem complete_square_quadratic (x : ℝ) : 
  (∃ c d : ℝ, x^2 + 6*x - 5 = 0 ↔ (x + c)^2 = d) → 
  (∃ c : ℝ, (x + c)^2 = 14) := by
sorry

end complete_square_quadratic_l2212_221261


namespace min_perimeter_rectangle_sphere_area_l2212_221285

/-- Given a rectangle ABCD with area 8, when its perimeter is minimized
    and triangle ACD is folded along diagonal AC to form a pyramid D-ABC,
    the surface area of the circumscribed sphere of this pyramid is 16π. -/
theorem min_perimeter_rectangle_sphere_area :
  ∀ (x y : ℝ),
  x > 0 → y > 0 →
  x * y = 8 →
  (∀ a b : ℝ, a > 0 → b > 0 → a * b = 8 → 2*(x + y) ≤ 2*(a + b)) →
  16 * Real.pi = 4 * Real.pi * (2 : ℝ)^2 := by
sorry

end min_perimeter_rectangle_sphere_area_l2212_221285


namespace convex_polygon_30_sides_diagonals_l2212_221290

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end convex_polygon_30_sides_diagonals_l2212_221290


namespace jacket_sale_profit_l2212_221225

/-- Calculates the merchant's gross profit for a jacket sale --/
theorem jacket_sale_profit (purchase_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  purchase_price = 42 ∧ 
  markup_percentage = 0.3 ∧ 
  discount_percentage = 0.2 → 
  let selling_price := purchase_price / (1 - markup_percentage)
  let discounted_price := selling_price * (1 - discount_percentage)
  discounted_price - purchase_price = 6 := by
  sorry

end jacket_sale_profit_l2212_221225


namespace special_number_exists_l2212_221264

/-- A function that removes the trailing zero from a binary representation -/
def removeTrailingZero (n : ℕ) : ℕ := sorry

/-- A function that converts a natural number to its ternary representation -/
def toTernary (n : ℕ) : ℕ := sorry

/-- The theorem stating the existence of the special number -/
theorem special_number_exists : ∃ N : ℕ, 
  N % 2 = 0 ∧ 
  removeTrailingZero N = toTernary (N / 3) := by
  sorry

end special_number_exists_l2212_221264


namespace right_triangle_acute_angles_l2212_221266

theorem right_triangle_acute_angles (a b : ℝ) : 
  a > 0 → b > 0 → -- Angles are positive
  a + b + 90 = 180 → -- Sum of angles in a triangle
  a / b = 7 / 2 → -- Ratio of acute angles
  (a = 70 ∧ b = 20) ∨ (a = 20 ∧ b = 70) := by sorry

end right_triangle_acute_angles_l2212_221266


namespace problem_statement_l2212_221212

theorem problem_statement (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_squares_nonzero : a^2 + b^2 + c^2 ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a^2 + b^2 + c^2)) = -7 := by
sorry

end problem_statement_l2212_221212


namespace solve_quadratic_equation_solve_linear_equation_l2212_221296

-- Equation 1
theorem solve_quadratic_equation (x : ℝ) :
  3 * x^2 + 6 * x - 4 = 0 ↔ x = (-3 + Real.sqrt 21) / 3 ∨ x = (-3 - Real.sqrt 21) / 3 := by
  sorry

-- Equation 2
theorem solve_linear_equation (x : ℝ) :
  3 * x * (2 * x + 1) = 4 * x + 2 ↔ x = -1/2 ∨ x = 2/3 := by
  sorry

end solve_quadratic_equation_solve_linear_equation_l2212_221296


namespace quadratic_equation_sum_l2212_221284

theorem quadratic_equation_sum (r s : ℝ) : 
  (∀ x, 9 * x^2 - 36 * x - 81 = 0 ↔ (x + r)^2 = s) →
  r + s = 11 := by
sorry

end quadratic_equation_sum_l2212_221284


namespace leanna_leftover_money_l2212_221273

/-- Represents the amount of money Leanna has left after purchasing one CD and two cassettes --/
def money_left_over (total_money : ℕ) (cd_price : ℕ) : ℕ :=
  let cassette_price := total_money - 2 * cd_price
  total_money - (cd_price + 2 * cassette_price)

/-- Theorem stating that Leanna will have $5 left over if she chooses to buy one CD and two cassettes --/
theorem leanna_leftover_money : 
  money_left_over 37 14 = 5 := by
  sorry

end leanna_leftover_money_l2212_221273


namespace triangle_inequalities_and_side_relationships_l2212_221230

theorem triangle_inequalities_and_side_relationships (a b c : ℝ) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (∃ (x y z : ℝ), x^2 = a ∧ y^2 = b ∧ z^2 = c ∧ x + y > z ∧ y + z > x ∧ z + x > y) ∧
  (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ a + b + c) ∧
  (a + b + c ≤ 2 * Real.sqrt (a * b) + 2 * Real.sqrt (b * c) + 2 * Real.sqrt (c * a)) :=
by sorry

end triangle_inequalities_and_side_relationships_l2212_221230


namespace house_payment_theorem_l2212_221215

/-- Calculates the amount still owed on a house given the initial price,
    down payment percentage, and percentage paid by parents. -/
def amount_owed (price : ℝ) (down_payment_percent : ℝ) (parents_payment_percent : ℝ) : ℝ :=
  let remaining_after_down := price * (1 - down_payment_percent)
  remaining_after_down * (1 - parents_payment_percent)

/-- Theorem stating that for a $100,000 house with 20% down payment
    and 30% of the remaining balance paid by parents, $56,000 is still owed. -/
theorem house_payment_theorem :
  amount_owed 100000 0.2 0.3 = 56000 := by
  sorry

end house_payment_theorem_l2212_221215


namespace water_percentage_in_dried_grapes_l2212_221201

/-- 
Given:
- Fresh grapes contain 90% water by weight
- 25 kg of fresh grapes yield 3.125 kg of dried grapes

Prove that the percentage of water in dried grapes is 20%
-/
theorem water_percentage_in_dried_grapes :
  let fresh_grape_weight : ℝ := 25
  let dried_grape_weight : ℝ := 3.125
  let fresh_water_percentage : ℝ := 90
  let dried_water_percentage : ℝ := (dried_grape_weight - (fresh_grape_weight * (1 - fresh_water_percentage / 100))) / dried_grape_weight * 100
  dried_water_percentage = 20 := by sorry

end water_percentage_in_dried_grapes_l2212_221201


namespace octagon_diagonals_l2212_221289

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l2212_221289


namespace lucky_larry_calculation_l2212_221293

theorem lucky_larry_calculation (a b c d e : ℚ) : 
  a = 16 ∧ b = 2 ∧ c = 3 ∧ d = 12 → 
  (a / (b / (c * (d / e))) = a / b / c * d / e) → 
  e = 9 := by
sorry

end lucky_larry_calculation_l2212_221293


namespace nancy_limes_l2212_221267

def fred_limes : ℕ := 36
def alyssa_limes : ℕ := 32
def total_limes : ℕ := 103

theorem nancy_limes : total_limes - (fred_limes + alyssa_limes) = 35 := by
  sorry

end nancy_limes_l2212_221267


namespace favorite_subject_problem_l2212_221205

theorem favorite_subject_problem (total_students : ℕ) 
  (math_fraction : ℚ) (english_fraction : ℚ) (science_fraction : ℚ)
  (h_total : total_students = 30)
  (h_math : math_fraction = 1 / 5)
  (h_english : english_fraction = 1 / 3)
  (h_science : science_fraction = 1 / 7) :
  total_students - 
  (↑total_students * math_fraction).floor - 
  (↑total_students * english_fraction).floor - 
  ((↑total_students - (↑total_students * math_fraction).floor - (↑total_students * english_fraction).floor) * science_fraction).floor = 12 := by
sorry

end favorite_subject_problem_l2212_221205


namespace inequality_problem_l2212_221242

/-- Given an inequality with parameter a, prove that a = 8 and find the solution set -/
theorem inequality_problem (a : ℝ) : 
  (∀ x : ℝ, |x^2 - 4*x + a| + |x - 3| ≤ 5 → x ≤ 3) ∧ 
  (∃ x : ℝ, x = 3 ∧ |x^2 - 4*x + a| + |x - 3| = 5) →
  a = 8 ∧ ∀ x : ℝ, (|x^2 - 4*x + a| + |x - 3| ≤ 5 ↔ 2 ≤ x ∧ x ≤ 3) :=
by sorry


end inequality_problem_l2212_221242


namespace matt_cookies_left_l2212_221255

/-- Represents the cookie-making scenario -/
structure CookieScenario where
  flour_per_batch : ℕ        -- pounds of flour per batch
  cookies_per_batch : ℕ      -- number of cookies per batch
  flour_bags : ℕ             -- number of flour bags used
  flour_per_bag : ℕ          -- pounds of flour per bag
  cookies_eaten : ℕ          -- number of cookies eaten

/-- Calculates the number of cookies left after baking and eating -/
def cookies_left (scenario : CookieScenario) : ℕ :=
  let total_flour := scenario.flour_bags * scenario.flour_per_bag
  let total_batches := total_flour / scenario.flour_per_batch
  let total_cookies := total_batches * scenario.cookies_per_batch
  total_cookies - scenario.cookies_eaten

/-- Theorem stating the number of cookies left in Matt's scenario -/
theorem matt_cookies_left :
  let matt_scenario : CookieScenario := {
    flour_per_batch := 2,
    cookies_per_batch := 12,
    flour_bags := 4,
    flour_per_bag := 5,
    cookies_eaten := 15
  }
  cookies_left matt_scenario = 105 := by
  sorry


end matt_cookies_left_l2212_221255


namespace lassi_production_l2212_221248

theorem lassi_production (mangoes : ℕ) (lassis : ℕ) : 
  (3 * lassis = 13 * mangoes) → (15 * lassis = 65 * mangoes) :=
by sorry

end lassi_production_l2212_221248


namespace unique_number_with_appended_digits_sum_l2212_221243

theorem unique_number_with_appended_digits_sum (A : ℕ) : 
  (∃ B : ℕ, B ≤ 999 ∧ 1000 * A + B = A * (A + 1) / 2) ↔ A = 1999 :=
sorry

end unique_number_with_appended_digits_sum_l2212_221243


namespace continued_fraction_result_l2212_221258

-- Define the continued fraction representation of x
noncomputable def x : ℝ := 2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / 2)))

-- State the theorem
theorem continued_fraction_result :
  1 / ((x + 1) * (x - 3)) = (3 + Real.sqrt 3) / (-6) :=
sorry

end continued_fraction_result_l2212_221258


namespace car_production_proof_l2212_221219

/-- The total number of cars produced over two days, given production on each day --/
def total_cars (day1 : ℕ) (day2 : ℕ) : ℕ := day1 + day2

/-- The number of cars produced on the second day is twice that of the first day --/
def double_production (day1 : ℕ) : ℕ := 2 * day1

theorem car_production_proof (day1 : ℕ) (h1 : day1 = 60) :
  total_cars day1 (double_production day1) = 180 := by
  sorry


end car_production_proof_l2212_221219


namespace leak_empty_time_proof_l2212_221244

/-- The time (in hours) it takes to fill the tank without a leak -/
def fill_time_without_leak : ℝ := 3

/-- The time (in hours) it takes to fill the tank with a leak -/
def fill_time_with_leak : ℝ := 4

/-- The capacity of the tank -/
def tank_capacity : ℝ := 1

/-- The time (in hours) it takes for the leak to empty the tank -/
def leak_empty_time : ℝ := 12

theorem leak_empty_time_proof :
  let fill_rate := tank_capacity / fill_time_without_leak
  let combined_rate := tank_capacity / fill_time_with_leak
  let leak_rate := fill_rate - combined_rate
  leak_empty_time = tank_capacity / leak_rate :=
by sorry

end leak_empty_time_proof_l2212_221244


namespace complex_fraction_simplification_l2212_221292

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 + 4 * i) / (1 + 2 * i) = 11 / 5 - 2 / 5 * i :=
by
  sorry

end complex_fraction_simplification_l2212_221292


namespace sequence_gcd_property_l2212_221279

theorem sequence_gcd_property (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ i : ℕ, a i = i :=
sorry

end sequence_gcd_property_l2212_221279


namespace imaginary_number_real_part_l2212_221214

theorem imaginary_number_real_part (a : ℝ) : 
  let z : ℂ := a + (Complex.I / (1 - Complex.I))
  (∃ (b : ℝ), z = Complex.I * b) → a = (1/2 : ℝ) := by
sorry

end imaginary_number_real_part_l2212_221214


namespace smallest_digit_divisible_by_9_l2212_221204

/-- The number formed by inserting a digit d between 586 and 17 -/
def number (d : Nat) : Nat := 586000 + d * 1000 + 17

/-- Predicate to check if a number is divisible by 9 -/
def divisible_by_9 (n : Nat) : Prop := n % 9 = 0

theorem smallest_digit_divisible_by_9 :
  ∃ (d : Nat), d < 10 ∧ divisible_by_9 (number d) ∧
  ∀ (d' : Nat), d' < d → ¬(divisible_by_9 (number d')) :=
sorry

end smallest_digit_divisible_by_9_l2212_221204


namespace pure_imaginary_complex_number_l2212_221278

/-- Given a complex number z defined in terms of a real number m, 
    prove that when z is a pure imaginary number, m = -3 -/
theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (m^2 + m - 6)/m + (m^2 - 2*m)*I
  (z.re = 0 ∧ z.im ≠ 0) → m = -3 :=
by sorry

end pure_imaginary_complex_number_l2212_221278


namespace hexagon_sixth_angle_l2212_221291

/-- The sum of angles in a hexagon -/
def hexagon_angle_sum : ℝ := 720

/-- The given angles in the hexagon -/
def given_angles : List ℝ := [150, 110, 120, 130, 100]

/-- Theorem: In a hexagon where five angles are 150°, 110°, 120°, 130°, and 100°, 
    the measure of the sixth angle is 110°. -/
theorem hexagon_sixth_angle : 
  hexagon_angle_sum - (given_angles.sum) = 110 := by sorry

end hexagon_sixth_angle_l2212_221291


namespace cube_face_perimeter_l2212_221237

/-- The sum of the lengths of sides of one face of a cube with side length 9 cm is 36 cm -/
theorem cube_face_perimeter (cube_side_length : ℝ) (h : cube_side_length = 9) : 
  4 * cube_side_length = 36 := by
  sorry

end cube_face_perimeter_l2212_221237


namespace cubic_root_nature_l2212_221229

theorem cubic_root_nature :
  ∃ (p n1 n2 : ℝ), p > 0 ∧ n1 < 0 ∧ n2 < 0 ∧
  p^3 + 3*p^2 - 4*p - 12 = 0 ∧
  n1^3 + 3*n1^2 - 4*n1 - 12 = 0 ∧
  n2^3 + 3*n2^2 - 4*n2 - 12 = 0 ∧
  ∀ x : ℝ, x^3 + 3*x^2 - 4*x - 12 = 0 → x = p ∨ x = n1 ∨ x = n2 :=
by sorry

end cubic_root_nature_l2212_221229


namespace complex_square_simplification_l2212_221260

theorem complex_square_simplification :
  (4 - 3 * Complex.I) ^ 2 = 7 - 24 * Complex.I := by
  sorry

end complex_square_simplification_l2212_221260


namespace school_play_girls_l2212_221207

/-- The number of girls in the school play -/
def num_girls : ℕ := 6

/-- The number of boys in the school play -/
def num_boys : ℕ := 8

/-- The total number of parents attending the premiere -/
def total_parents : ℕ := 28

/-- The number of parents per child -/
def parents_per_child : ℕ := 2

theorem school_play_girls :
  num_girls = 6 ∧
  num_boys * parents_per_child + num_girls * parents_per_child = total_parents :=
sorry

end school_play_girls_l2212_221207


namespace min_value_theorem_l2212_221216

/-- A circle C with equation x^2 + y^2 - 4x - 2y + 1 = 0 -/
def CircleC (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- A line l with equation ax + by - 2 = 0 -/
def LineL (a b x y : ℝ) : Prop :=
  a*x + b*y - 2 = 0

/-- The theorem stating the minimum value of 1/a + 2/b -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_symmetry : ∃ (x y : ℝ), CircleC x y ∧ LineL a b x y) :
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 →
    (∃ (x y : ℝ), CircleC x y ∧ LineL a' b' x y) →
    1/a + 2/b ≤ 1/a' + 2/b') →
  1/a + 2/b = 4 :=
sorry

end min_value_theorem_l2212_221216


namespace find_numbers_l2212_221263

theorem find_numbers (A B C : ℕ) 
  (h1 : Nat.gcd A B = 2)
  (h2 : Nat.lcm A B = 60)
  (h3 : Nat.gcd A C = 3)
  (h4 : Nat.lcm A C = 42) :
  A = 6 ∧ B = 20 ∧ C = 21 := by
  sorry

end find_numbers_l2212_221263


namespace fruit_condition_percentage_l2212_221276

theorem fruit_condition_percentage 
  (total_oranges : ℕ) 
  (total_bananas : ℕ) 
  (rotten_oranges_percent : ℚ) 
  (rotten_bananas_percent : ℚ) 
  (h1 : total_oranges = 600) 
  (h2 : total_bananas = 400) 
  (h3 : rotten_oranges_percent = 15 / 100) 
  (h4 : rotten_bananas_percent = 8 / 100) : 
  (1 - (rotten_oranges_percent * total_oranges + rotten_bananas_percent * total_bananas) / 
   (total_oranges + total_bananas)) * 100 = 878 / 10 :=
by sorry

end fruit_condition_percentage_l2212_221276


namespace x_intercept_of_parallel_lines_l2212_221270

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 m2 : ℚ) : Prop := m1 = m2

/-- Line l1 with slope m1 and y-intercept b1 -/
def line1 (x y : ℚ) (m1 b1 : ℚ) : Prop := y = m1 * x + b1

/-- Line l2 with slope m2 and y-intercept b2 -/
def line2 (x y : ℚ) (m2 b2 : ℚ) : Prop := y = m2 * x + b2

/-- The x-intercept of a line with slope m and y-intercept b -/
def x_intercept (m b : ℚ) : ℚ := -b / m

theorem x_intercept_of_parallel_lines 
  (a : ℚ) 
  (h_parallel : parallel (-(a+2)/3) (-(a-1)/2)) : 
  x_intercept (-(a+2)/3) (5/3) = 5/9 := by sorry

end x_intercept_of_parallel_lines_l2212_221270


namespace eustace_age_in_three_years_l2212_221226

/-- Proves that Eustace will be 39 years old in 3 years, given the conditions -/
theorem eustace_age_in_three_years
  (eustace_age : ℕ)
  (milford_age : ℕ)
  (h1 : eustace_age = 2 * milford_age)
  (h2 : milford_age + 3 = 21) :
  eustace_age + 3 = 39 := by
  sorry

end eustace_age_in_three_years_l2212_221226


namespace andrew_work_hours_l2212_221227

/-- The number of days Andrew worked on his Science report -/
def days_worked : ℝ := 3

/-- The number of hours Andrew worked each day -/
def hours_per_day : ℝ := 2.5

/-- The total number of hours Andrew worked on his Science report -/
def total_hours : ℝ := days_worked * hours_per_day

theorem andrew_work_hours : total_hours = 7.5 := by
  sorry

end andrew_work_hours_l2212_221227


namespace quadratic_function_properties_l2212_221275

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define g(x) = f(x) - x^2
def g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x - x^2

-- Theorem statement
theorem quadratic_function_properties
  (a b c : ℝ)
  (origin : f a b c 0 = 0)
  (symmetry : ∀ x, f a b c (1 - x) = f a b c (1 + x))
  (odd_g : ∀ x, g a b c x = -g a b c (-x))
  : f a b c = fun x ↦ x^2 - 2*x :=
by sorry

end quadratic_function_properties_l2212_221275


namespace board_numbers_theorem_l2212_221240

def pairwise_sums : List ℕ := [5, 8, 9, 13, 14, 14, 15, 17, 18, 23]

def is_valid_set (s : List ℕ) : Prop :=
  s.length = 5 ∧
  (List.map (λ (x, y) => x + y) (s.product s)).filter (λ x => x ∉ s) = pairwise_sums

theorem board_numbers_theorem :
  ∃ (s : List ℕ), is_valid_set s ∧ s.prod = 4752 := by sorry

end board_numbers_theorem_l2212_221240


namespace max_true_statements_l2212_221223

theorem max_true_statements (c d : ℝ) : 
  let statements := [
    (1 / c > 1 / d),
    (c^2 < d^2),
    (c > d),
    (c > 0),
    (d > 0)
  ]
  ∃ (true_statements : List Bool), 
    true_statements.length ≤ 4 ∧ 
    (∀ i, i < statements.length → 
      (true_statements.get! i = true ↔ statements.get! i)) :=
by sorry

end max_true_statements_l2212_221223


namespace smallest_n_is_101_l2212_221203

/-- Represents a square in the n × n table -/
structure Square where
  row : Nat
  col : Nat

/-- Represents a rectangle in the table -/
structure Rectangle where
  topLeft : Square
  bottomRight : Square

/-- Represents the n × n table -/
structure Table (n : Nat) where
  blueSquares : Finset Square
  rectangles : Finset Rectangle
  uniquePartition : Prop
  oneBluePerRectangle : Prop

/-- The main theorem -/
theorem smallest_n_is_101 :
  ∀ n : Nat,
  ∃ (t : Table n),
  (t.blueSquares.card = 101) →
  t.uniquePartition →
  t.oneBluePerRectangle →
  n ≥ 101 ∧ ∃ (t' : Table 101), 
    t'.blueSquares.card = 101 ∧
    t'.uniquePartition ∧
    t'.oneBluePerRectangle :=
sorry

end smallest_n_is_101_l2212_221203


namespace min_value_xy_over_x2_plus_2y2_l2212_221287

theorem min_value_xy_over_x2_plus_2y2 (x y : ℝ) 
  (hx : 0.4 ≤ x ∧ x ≤ 0.6) (hy : 0.3 ≤ y ∧ y ≤ 0.5) :
  (∃ m : ℝ, m = (x * y) / (x^2 + 2 * y^2) ∧ 
    (∀ x' y' : ℝ, 0.4 ≤ x' ∧ x' ≤ 0.6 → 0.3 ≤ y' ∧ y' ≤ 0.5 → 
      m ≤ (x' * y') / (x'^2 + 2 * y'^2)) ∧
    m = 1/3) := by
  sorry

end min_value_xy_over_x2_plus_2y2_l2212_221287


namespace pyramid_volume_from_rectangle_l2212_221209

/-- The volume of a pyramid formed from a rectangle with specific dimensions -/
theorem pyramid_volume_from_rectangle (AB BC : ℝ) (h : AB = 15 * Real.sqrt 2 ∧ BC = 17 * Real.sqrt 2) :
  let P : ℝ × ℝ × ℝ := (15 * Real.sqrt 2 / 2, 17 * Real.sqrt 2 / 2, Real.sqrt 257)
  let base_area : ℝ := (1 / 2) * AB * BC
  let volume : ℝ := (1 / 3) * base_area * P.2.2
  volume = 85 * Real.sqrt 257 := by
  sorry

end pyramid_volume_from_rectangle_l2212_221209


namespace right_pyramid_base_side_length_l2212_221210

/-- Given a right pyramid with a square base, if the area of one lateral face is 120 square meters
    and the slant height is 40 meters, then the length of the side of its base is 6 meters. -/
theorem right_pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  (2 * area_lateral_face) / slant_height = 6 :=
by sorry

end right_pyramid_base_side_length_l2212_221210


namespace infinite_possibilities_for_A_squared_l2212_221208

/-- Given a 3x3 matrix A with real entries such that A^4 = 0, 
    there are infinitely many possible matrices that A^2 can be. -/
theorem infinite_possibilities_for_A_squared 
  (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A ^ 4 = 0) : 
  ∃ S : Set (Matrix (Fin 3) (Fin 3) ℝ), 
    (∀ B ∈ S, ∃ A : Matrix (Fin 3) (Fin 3) ℝ, A ^ 4 = 0 ∧ A ^ 2 = B) ∧ 
    Set.Infinite S :=
by sorry

end infinite_possibilities_for_A_squared_l2212_221208


namespace present_age_of_b_l2212_221213

theorem present_age_of_b (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10))  -- In 10 years, A will be twice as old as B was 10 years ago
  (h2 : a = b + 9)              -- A is now 9 years older than B
  : b = 39 := by               -- The present age of B is 39 years
  sorry

end present_age_of_b_l2212_221213


namespace cookie_comparison_l2212_221233

theorem cookie_comparison (a b c : ℕ) (ha : a = 7) (hb : b = 8) (hc : c = 5) :
  (1 : ℚ) / c > (1 : ℚ) / a ∧ (1 : ℚ) / c > (1 : ℚ) / b :=
sorry

end cookie_comparison_l2212_221233


namespace combined_salary_proof_l2212_221297

/-- The combined salary of two people A and B -/
def combinedSalary (salaryA salaryB : ℝ) : ℝ := salaryA + salaryB

/-- The savings of a person given their salary and spending percentage -/
def savings (salary spendingPercentage : ℝ) : ℝ := salary * (1 - spendingPercentage)

theorem combined_salary_proof (salaryA salaryB : ℝ) 
  (hSpendA : savings salaryA 0.8 = savings salaryB 0.85)
  (hSalaryB : salaryB = 8000) :
  combinedSalary salaryA salaryB = 14000 := by
  sorry

end combined_salary_proof_l2212_221297


namespace product_of_base9_digits_9876_l2212_221299

/-- Converts a base 10 number to base 9 --/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers --/
def productOfList (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 9 representation of 9876₁₀ is 192 --/
theorem product_of_base9_digits_9876 :
  productOfList (toBase9 9876) = 192 := by
  sorry

end product_of_base9_digits_9876_l2212_221299


namespace jamie_yellow_balls_l2212_221228

/-- Proves that Jamie bought 32 yellow balls given the initial conditions and final total -/
theorem jamie_yellow_balls :
  let initial_red : ℕ := 16
  let initial_blue : ℕ := 2 * initial_red
  let lost_red : ℕ := 6
  let final_total : ℕ := 74
  let remaining_red : ℕ := initial_red - lost_red
  let yellow_balls : ℕ := final_total - (remaining_red + initial_blue)
  yellow_balls = 32 := by
  sorry

end jamie_yellow_balls_l2212_221228


namespace line_parallel_perpendicular_implies_planes_perpendicular_l2212_221252

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → plane_perpendicular α β :=
sorry

end line_parallel_perpendicular_implies_planes_perpendicular_l2212_221252


namespace total_subjects_l2212_221280

/-- Given the number of subjects taken by Monica, prove the total number of subjects taken by all four students. -/
theorem total_subjects (monica : ℕ) (h1 : monica = 10) : ∃ (marius millie michael : ℕ),
  marius = monica + 4 ∧
  millie = marius + 3 ∧
  michael = 2 * millie ∧
  monica + marius + millie + michael = 75 :=
by sorry

end total_subjects_l2212_221280
