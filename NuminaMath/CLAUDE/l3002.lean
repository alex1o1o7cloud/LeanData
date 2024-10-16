import Mathlib

namespace NUMINAMATH_CALUDE_complex_root_magnitude_l3002_300202

theorem complex_root_magnitude (z : ℂ) (h : z^2 - z + 1 = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_magnitude_l3002_300202


namespace NUMINAMATH_CALUDE_num_pyramids_from_rectangular_solid_l3002_300210

/-- A rectangular solid (cuboid) --/
structure RectangularSolid where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 2 × Fin 8)
  faces : Finset (Fin 4 × Fin 8)

/-- A pyramid (tetrahedron) formed by vertices of a rectangular solid --/
structure Pyramid where
  vertices : Finset (Fin 4)

/-- The set of all possible pyramids that can be formed from a rectangular solid --/
def allPyramids (rs : RectangularSolid) : Finset Pyramid :=
  sorry

/-- The main theorem stating that the number of different pyramids is 106 --/
theorem num_pyramids_from_rectangular_solid (rs : RectangularSolid) :
  (allPyramids rs).card = 106 := by
  sorry

end NUMINAMATH_CALUDE_num_pyramids_from_rectangular_solid_l3002_300210


namespace NUMINAMATH_CALUDE_paul_reading_theorem_l3002_300238

/-- Calculates the total number of books read given a weekly reading rate and number of weeks -/
def total_books_read (books_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  books_per_week * num_weeks

/-- Proves that reading 4 books per week for 5 weeks results in 20 books read -/
theorem paul_reading_theorem : 
  total_books_read 4 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_paul_reading_theorem_l3002_300238


namespace NUMINAMATH_CALUDE_dairy_water_mixture_l3002_300281

theorem dairy_water_mixture (pure_dairy : ℝ) (profit_percentage : ℝ) 
  (h1 : pure_dairy > 0)
  (h2 : profit_percentage = 25) : 
  let total_mixture := pure_dairy * (1 + profit_percentage / 100)
  let water_added := total_mixture - pure_dairy
  (water_added / total_mixture) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dairy_water_mixture_l3002_300281


namespace NUMINAMATH_CALUDE_arrangements_with_A_not_first_is_48_l3002_300259

/-- The number of ways to arrange 3 people from 5, including A and B, with A not at the head -/
def arrangements_with_A_not_first (total_people : ℕ) (selected_people : ℕ) : ℕ :=
  (total_people * (total_people - 1) * (total_people - 2)) -
  ((total_people - 1) * (total_people - 2))

/-- Theorem stating that the number of arrangements with A not at the head is 48 -/
theorem arrangements_with_A_not_first_is_48 :
  arrangements_with_A_not_first 5 3 = 48 := by
  sorry

#eval arrangements_with_A_not_first 5 3

end NUMINAMATH_CALUDE_arrangements_with_A_not_first_is_48_l3002_300259


namespace NUMINAMATH_CALUDE_remainder_negation_l3002_300261

theorem remainder_negation (a : ℤ) : 
  (a % 1999 = 1) → ((-a) % 1999 = 1998) := by
  sorry

end NUMINAMATH_CALUDE_remainder_negation_l3002_300261


namespace NUMINAMATH_CALUDE_leftover_eggs_l3002_300295

theorem leftover_eggs (abigail_eggs beatrice_eggs carson_eggs : ℕ) 
  (h1 : abigail_eggs = 37)
  (h2 : beatrice_eggs = 49)
  (h3 : carson_eggs = 14) :
  (abigail_eggs + beatrice_eggs + carson_eggs) % 12 = 4 := by
sorry

end NUMINAMATH_CALUDE_leftover_eggs_l3002_300295


namespace NUMINAMATH_CALUDE_range_of_a_l3002_300268

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) ↔ -1 < a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3002_300268


namespace NUMINAMATH_CALUDE_parabola_circle_separation_l3002_300237

/-- The range of 'a' for a parabola y^2 = 4ax with directrix separate from the circle x^2 + y^2 - 2y = 0 -/
theorem parabola_circle_separation (a : ℝ) : 
  (∀ x y : ℝ, y^2 = 4*a*x → x^2 + y^2 - 2*y ≠ 0) →
  (∀ x y : ℝ, x = a → x^2 + y^2 - 2*y ≠ 0) →
  a > 1 ∨ a < -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_separation_l3002_300237


namespace NUMINAMATH_CALUDE_average_height_is_141_l3002_300248

def student_heights : List ℝ := [145, 142, 138, 136, 143, 146, 138, 144, 137, 141]

theorem average_height_is_141 :
  (student_heights.sum / student_heights.length : ℝ) = 141 := by
  sorry

end NUMINAMATH_CALUDE_average_height_is_141_l3002_300248


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_13_l3002_300279

theorem smallest_n_divisible_by_13 : 
  ∃ (n : ℕ), (13 ∣ (5^n + n^5)) ∧ (∀ m : ℕ, m < n → ¬(13 ∣ (5^m + m^5))) ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_13_l3002_300279


namespace NUMINAMATH_CALUDE_solution_is_one_l3002_300221

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  (7 / (x^2 + x)) - (3 / (x - x^2)) = 1 + ((7 - x^2) / (x^2 - 1))

/-- Theorem stating that x = 1 is the solution to the equation -/
theorem solution_is_one : equation 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_one_l3002_300221


namespace NUMINAMATH_CALUDE_unique_division_representation_l3002_300209

theorem unique_division_representation (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (a : ℚ) / b = b + a / 10 ↔ a = 5 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_division_representation_l3002_300209


namespace NUMINAMATH_CALUDE_weight_difference_l3002_300276

theorem weight_difference (w_a w_b w_c w_d w_e : ℝ) : 
  (w_a + w_b + w_c) / 3 = 84 →
  (w_a + w_b + w_c + w_d) / 4 = 80 →
  (w_b + w_c + w_d + w_e) / 4 = 79 →
  w_a = 80 →
  w_e > w_d →
  w_e - w_d = 8 := by
sorry


end NUMINAMATH_CALUDE_weight_difference_l3002_300276


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l3002_300224

/-- The shortest distance between a point on the parabola y = -x^2 + 5x + 7 
    and a point on the line y = 2x - 3 is 31√5/20 -/
theorem shortest_distance_parabola_to_line :
  let parabola := fun x : ℝ => -x^2 + 5*x + 7
  let line := fun x : ℝ => 2*x - 3
  ∃ (d : ℝ), d = (31 * Real.sqrt 5) / 20 ∧
    ∀ (x₁ x₂ : ℝ), 
      d ≤ Real.sqrt ((x₁ - x₂)^2 + (parabola x₁ - line x₂)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l3002_300224


namespace NUMINAMATH_CALUDE_jennifer_spending_l3002_300236

theorem jennifer_spending (initial_amount : ℚ) : 
  (initial_amount - (initial_amount / 5 + initial_amount / 6 + initial_amount / 2) = 16) → 
  initial_amount = 120 := by
sorry

end NUMINAMATH_CALUDE_jennifer_spending_l3002_300236


namespace NUMINAMATH_CALUDE_smallest_area_right_triangle_l3002_300265

theorem smallest_area_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area1 := a * b / 2
  let c := Real.sqrt (a^2 + b^2)
  let area2 := a * Real.sqrt (b^2 - a^2) / 2
  area2 < area1 := by sorry

end NUMINAMATH_CALUDE_smallest_area_right_triangle_l3002_300265


namespace NUMINAMATH_CALUDE_jason_money_last_week_l3002_300287

/-- Given information about Fred and Jason's money before and after washing cars,
    prove how much money Jason had last week. -/
theorem jason_money_last_week
  (fred_money_last_week : ℕ)
  (fred_money_now : ℕ)
  (jason_money_now : ℕ)
  (fred_earned : ℕ)
  (h1 : fred_money_last_week = 19)
  (h2 : fred_money_now = 40)
  (h3 : jason_money_now = 69)
  (h4 : fred_earned = 21)
  (h5 : fred_money_now = fred_money_last_week + fred_earned) :
  jason_money_now - fred_earned = 48 :=
by sorry

end NUMINAMATH_CALUDE_jason_money_last_week_l3002_300287


namespace NUMINAMATH_CALUDE_runner_average_speed_l3002_300289

/-- Calculates the average speed during the second part of a run given the following conditions:
  * The runner runs at 18 mph for 3 hours
  * The runner then runs for an additional 5 hours
  * The total distance covered is 124 miles in 8 hours
-/
theorem runner_average_speed 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (second_time : ℝ) 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (h1 : initial_speed = 18)
  (h2 : initial_time = 3)
  (h3 : second_time = 5)
  (h4 : total_time = 8)
  (h5 : total_distance = 124) :
  (total_distance - initial_speed * initial_time) / second_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_runner_average_speed_l3002_300289


namespace NUMINAMATH_CALUDE_alien_rock_count_l3002_300242

/-- Converts a three-digit number in base 7 to base 10 --/
def base7ToBase10 (hundreds tens units : ℕ) : ℕ :=
  hundreds * 7^2 + tens * 7^1 + units * 7^0

/-- The number of rocks seen by the alien --/
def alienRocks : ℕ := base7ToBase10 3 5 1

theorem alien_rock_count : alienRocks = 183 := by sorry

end NUMINAMATH_CALUDE_alien_rock_count_l3002_300242


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3002_300277

theorem polynomial_factorization (a : ℝ) : a^2 - 5*a - 6 = (a - 6) * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3002_300277


namespace NUMINAMATH_CALUDE_range_of_a_l3002_300292

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (a * x^2 - x + 1/4 * a)

def q (a : ℝ) : Prop := ∀ x > 0, 3^x - 9^x < a

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3002_300292


namespace NUMINAMATH_CALUDE_unique_intersection_l3002_300206

/-- The first function f(x) = x^2 - 7x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 7*x + 3

/-- The second function g(x) = -3x^2 + 5x - 6 -/
def g (x : ℝ) : ℝ := -3*x^2 + 5*x - 6

/-- The theorem stating that f and g intersect at exactly one point (3/2, -21/4) -/
theorem unique_intersection :
  ∃! p : ℝ × ℝ, 
    p.1 = 3/2 ∧ 
    p.2 = -21/4 ∧ 
    f p.1 = g p.1 ∧
    ∀ x : ℝ, f x = g x → x = p.1 := by
  sorry

#check unique_intersection

end NUMINAMATH_CALUDE_unique_intersection_l3002_300206


namespace NUMINAMATH_CALUDE_triangle_inequality_l3002_300207

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (a + b + c)^2 < 4 * (a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3002_300207


namespace NUMINAMATH_CALUDE_no_regular_lattice_polygon_except_square_l3002_300214

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A regular n-gon with vertices at lattice points -/
structure RegularLatticePolygon where
  n : ℕ
  vertices : Fin n → LatticePoint

/-- Predicate to check if a set of points forms a regular n-gon -/
def IsRegularPolygon (poly : RegularLatticePolygon) : Prop :=
  ∀ i j : Fin poly.n,
    (poly.vertices i).x ^ 2 + (poly.vertices i).y ^ 2 =
    (poly.vertices j).x ^ 2 + (poly.vertices j).y ^ 2

/-- Main theorem: No regular n-gon with vertices at lattice points exists for n ≠ 4 -/
theorem no_regular_lattice_polygon_except_square :
  ∀ n : ℕ, n ≠ 4 → ¬∃ (poly : RegularLatticePolygon), poly.n = n ∧ IsRegularPolygon poly :=
sorry

end NUMINAMATH_CALUDE_no_regular_lattice_polygon_except_square_l3002_300214


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3002_300267

/-- A sequence is geometric if the ratio between any two consecutive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : IsGeometric a) 
    (h_sum : a 4 + a 6 = 10) : 
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 100 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3002_300267


namespace NUMINAMATH_CALUDE_walking_time_equals_time_saved_l3002_300272

/-- Represents the scenario of a man walking and his wife driving to meet him -/
structure CommuteScenario where
  usual_drive_time : ℝ
  actual_drive_time : ℝ
  time_saved : ℝ
  walking_time : ℝ

/-- Theorem stating that the walking time equals the time saved -/
theorem walking_time_equals_time_saved (scenario : CommuteScenario) 
  (h1 : scenario.usual_drive_time > 0)
  (h2 : scenario.actual_drive_time > 0)
  (h3 : scenario.time_saved > 0)
  (h4 : scenario.walking_time > 0)
  (h5 : scenario.usual_drive_time = scenario.actual_drive_time + scenario.time_saved)
  (h6 : scenario.walking_time = scenario.time_saved) : 
  scenario.walking_time = scenario.time_saved :=
by sorry

end NUMINAMATH_CALUDE_walking_time_equals_time_saved_l3002_300272


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l3002_300299

def circle_equation (x y : ℝ) : Prop :=
  x^2 + 2*x - 4*y - 7 = -y^2 + 8*x

def center_and_radius_sum (c d s : ℝ) : ℝ :=
  c + d + s

theorem circle_center_radius_sum :
  ∃ (c d s : ℝ),
    (∀ (x y : ℝ), circle_equation x y ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    center_and_radius_sum c d s = 5 + 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l3002_300299


namespace NUMINAMATH_CALUDE_min_sum_fraction_l3002_300226

theorem min_sum_fraction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ 3 / Real.rpow 162 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_fraction_l3002_300226


namespace NUMINAMATH_CALUDE_sin_2x_derivative_l3002_300217

open Real

theorem sin_2x_derivative (x : ℝ) : 
  deriv (λ x => sin (2 * x)) x = 2 * cos (2 * x) := by sorry

end NUMINAMATH_CALUDE_sin_2x_derivative_l3002_300217


namespace NUMINAMATH_CALUDE_quadratic_solution_average_l3002_300258

theorem quadratic_solution_average (c : ℝ) :
  c < 3 →  -- Condition for real and distinct solutions
  let equation := fun x : ℝ => 3 * x^2 - 6 * x + c
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ = 0 ∧ equation x₂ = 0 ∧ (x₁ + x₂) / 2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_solution_average_l3002_300258


namespace NUMINAMATH_CALUDE_number_of_divisors_3003_l3002_300240

theorem number_of_divisors_3003 : Finset.card (Nat.divisors 3003) = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_3003_l3002_300240


namespace NUMINAMATH_CALUDE_problem_solution_l3002_300213

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a + 1)*x + (a^2 - 5) = 0}

theorem problem_solution :
  (∀ a : ℝ, A ∩ B a = {2} → a = -1 ∨ a = -3) ∧
  (∀ a : ℝ, A ∪ B a = A → a ≤ -3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3002_300213


namespace NUMINAMATH_CALUDE_no_real_solutions_for_equation_l3002_300243

theorem no_real_solutions_for_equation : 
  ¬ ∃ x : ℝ, (x + 4)^2 = 3*(x - 2) := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_equation_l3002_300243


namespace NUMINAMATH_CALUDE_polly_tweets_l3002_300227

/-- Represents the tweet rate (tweets per minute) for each of Polly's activities -/
structure TweetRate where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Represents the duration (in minutes) of each of Polly's activities -/
structure ActivityDuration where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Calculates the total number of tweets given the tweet rates and activity durations -/
def totalTweets (rate : TweetRate) (duration : ActivityDuration) : ℕ :=
  rate.happy * duration.happy + rate.hungry * duration.hungry + rate.mirror * duration.mirror

/-- Theorem stating that given Polly's specific tweet rates and activity durations, 
    the total number of tweets is 1340 -/
theorem polly_tweets : 
  ∀ (rate : TweetRate) (duration : ActivityDuration),
  rate.happy = 18 ∧ rate.hungry = 4 ∧ rate.mirror = 45 ∧
  duration.happy = 20 ∧ duration.hungry = 20 ∧ duration.mirror = 20 →
  totalTweets rate duration = 1340 := by
sorry

end NUMINAMATH_CALUDE_polly_tweets_l3002_300227


namespace NUMINAMATH_CALUDE_valid_assignment_probability_l3002_300282

/-- A regular dodecahedron with 12 numbered faces -/
structure NumberedDodecahedron :=
  (assignment : Fin 12 → Fin 12)
  (injective : Function.Injective assignment)

/-- Two numbers are consecutive if they differ by 1 or are 1 and 12 -/
def consecutive (a b : Fin 12) : Prop :=
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 0 ∧ b = 11) ∨ (a = 11 ∧ b = 0)

/-- The set of all possible numbered dodecahedrons -/
def allAssignments : Finset NumberedDodecahedron := sorry

/-- The set of valid assignments where no consecutive numbers are on adjacent faces -/
def validAssignments : Finset NumberedDodecahedron := sorry

/-- The probability of a valid assignment -/
def validProbability : ℚ := (validAssignments.card : ℚ) / (allAssignments.card : ℚ)

/-- The main theorem stating that the probability is 1/100 -/
theorem valid_assignment_probability :
  validProbability = 1 / 100 := by sorry

end NUMINAMATH_CALUDE_valid_assignment_probability_l3002_300282


namespace NUMINAMATH_CALUDE_food_cost_theorem_l3002_300274

def sandwich_cost : ℝ := 4

def juice_cost (sandwich_cost : ℝ) : ℝ := 2 * sandwich_cost

def milk_cost (sandwich_cost juice_cost : ℝ) : ℝ :=
  0.75 * (sandwich_cost + juice_cost)

def total_cost (sandwich_cost juice_cost milk_cost : ℝ) : ℝ :=
  sandwich_cost + juice_cost + milk_cost

theorem food_cost_theorem :
  total_cost sandwich_cost (juice_cost sandwich_cost) (milk_cost sandwich_cost (juice_cost sandwich_cost)) = 21 := by
  sorry

end NUMINAMATH_CALUDE_food_cost_theorem_l3002_300274


namespace NUMINAMATH_CALUDE_number_satisfying_proportion_l3002_300275

theorem number_satisfying_proportion : 
  let x : ℚ := 3
  (x + 1) / (x + 5) = (x + 5) / (x + 13) := by
sorry

end NUMINAMATH_CALUDE_number_satisfying_proportion_l3002_300275


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_20112021_base5_l3002_300235

/-- Converts a base 5 number represented as a string to a natural number -/
def base5ToNat (s : String) : ℕ := sorry

/-- Checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- Finds the largest prime divisor of a natural number -/
def largestPrimeDivisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_20112021_base5 :
  largestPrimeDivisor (base5ToNat "20112021") = 419 := by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_20112021_base5_l3002_300235


namespace NUMINAMATH_CALUDE_system_solution_l3002_300203

theorem system_solution : 
  ∃! (s : Set (ℝ × ℝ)), s = {(12, 10), (-10, -12)} ∧ 
    ∀ (x y : ℝ), (x, y) ∈ s ↔ 
      ((3/2 : ℝ)^(x-y) - (2/3 : ℝ)^(x-y) = 65/36 ∧
       x*y - x + y = 118) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3002_300203


namespace NUMINAMATH_CALUDE_division_problem_l3002_300246

theorem division_problem (total : ℕ) (a b c : ℕ) : 
  total = 770 →
  a = b + 40 →
  c = a + 30 →
  total = a + b + c →
  b = 220 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3002_300246


namespace NUMINAMATH_CALUDE_ink_needed_per_whiteboard_l3002_300220

-- Define the given conditions
def num_classes : ℕ := 5
def whiteboards_per_class : ℕ := 2
def ink_cost_per_ml : ℚ := 50 / 100  -- 50 cents = 0.5 dollars
def total_daily_cost : ℚ := 100

-- Define the function to calculate ink needed per whiteboard
def ink_per_whiteboard : ℚ :=
  let total_whiteboards : ℕ := num_classes * whiteboards_per_class
  let total_ink_ml : ℚ := total_daily_cost / ink_cost_per_ml
  total_ink_ml / total_whiteboards

-- Theorem to prove
theorem ink_needed_per_whiteboard : ink_per_whiteboard = 20 := by
  sorry

end NUMINAMATH_CALUDE_ink_needed_per_whiteboard_l3002_300220


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3002_300278

theorem complex_equation_solution (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  z = Complex.I + 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3002_300278


namespace NUMINAMATH_CALUDE_fraction_simplification_l3002_300262

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) : (2 * a) / (2 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3002_300262


namespace NUMINAMATH_CALUDE_triangle_theorem_l3002_300285

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition from the problem -/
def satisfies_condition (t : Triangle) : Prop :=
  t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h : satisfies_condition t) : 
  t.A = π / 3 ∧ 
  (t.a = Real.sqrt 3 → 
    ∀ (area : ℝ), area ≤ 3 * Real.sqrt 3 / 4 → 
      ∃ (t' : Triangle), satisfies_condition t' ∧ t'.a = Real.sqrt 3 ∧ 
        area = 1 / 2 * t'.b * t'.c * Real.sin t'.A) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3002_300285


namespace NUMINAMATH_CALUDE_equalize_expenses_l3002_300291

/-- The amount LeRoy paid initially -/
def leroy_paid : ℝ := 240

/-- The amount Bernardo paid initially -/
def bernardo_paid : ℝ := 360

/-- The total discount received -/
def discount : ℝ := 60

/-- The amount LeRoy should pay Bernardo to equalize expenses -/
def payment_to_equalize : ℝ := 30

theorem equalize_expenses : 
  let total_cost := leroy_paid + bernardo_paid - discount
  let each_share := total_cost / 2
  payment_to_equalize = each_share - leroy_paid :=
by sorry

end NUMINAMATH_CALUDE_equalize_expenses_l3002_300291


namespace NUMINAMATH_CALUDE_problem_solution_l3002_300247

-- Define the variables and functions
def f (x : ℝ) := 2 * x + 1
def g (x : ℝ) := x^2 + 2 * x

-- State the theorem
theorem problem_solution :
  ∃ (a b n : ℝ),
    f 2 = 5 ∧
    g 2 = a ∧
    f n = b ∧
    g n = -1 ∧
    a = 8 ∧
    b = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3002_300247


namespace NUMINAMATH_CALUDE_expression_simplification_l3002_300216

theorem expression_simplification :
  (3 * Real.sqrt 12) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 6) = Real.sqrt 3 + 2 * Real.sqrt 2 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3002_300216


namespace NUMINAMATH_CALUDE_aaron_position_100_l3002_300251

/-- Represents a position on a 2D plane -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a direction -/
inductive Direction
  | East
  | North
  | West
  | South

/-- Defines Aaron's movement rules -/
def nextPosition (current : Position) (dir : Direction) (visited : List Position) : Position × Direction :=
  sorry

/-- Calculates Aaron's position after n moves -/
def aaronPosition (n : Nat) : Position :=
  sorry

/-- Theorem stating Aaron's position after 100 moves -/
theorem aaron_position_100 : aaronPosition 100 = Position.mk 22 (-6) := by
  sorry

end NUMINAMATH_CALUDE_aaron_position_100_l3002_300251


namespace NUMINAMATH_CALUDE_polygon_20_vertices_has_170_diagonals_l3002_300294

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 20 vertices has 170 diagonals -/
theorem polygon_20_vertices_has_170_diagonals :
  num_diagonals 20 = 170 := by
  sorry

end NUMINAMATH_CALUDE_polygon_20_vertices_has_170_diagonals_l3002_300294


namespace NUMINAMATH_CALUDE_delta_phi_solution_l3002_300286

def δ (x : ℝ) : ℝ := 4 * x + 5

def φ (x : ℝ) : ℝ := 5 * x + 4

theorem delta_phi_solution :
  ∃ x : ℝ, δ (φ x) = 4 ∧ x = -17/20 := by
  sorry

end NUMINAMATH_CALUDE_delta_phi_solution_l3002_300286


namespace NUMINAMATH_CALUDE_fraction_equality_l3002_300233

theorem fraction_equality (p r s u : ℚ) 
  (h1 : p / r = 8)
  (h2 : s / r = 5)
  (h3 : s / u = 1 / 3) :
  u / p = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3002_300233


namespace NUMINAMATH_CALUDE_modulus_of_z_l3002_300253

-- Define the complex number z
def z : ℂ := 3 - 4 * Complex.I

-- Theorem statement
theorem modulus_of_z : Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3002_300253


namespace NUMINAMATH_CALUDE_sum_of_abs_and_square_zero_l3002_300270

theorem sum_of_abs_and_square_zero (x y : ℝ) :
  |x + 3| + (2 * y - 5)^2 = 0 → x + 2 * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abs_and_square_zero_l3002_300270


namespace NUMINAMATH_CALUDE_cube_volume_from_edge_sum_l3002_300204

/-- Given a cube where the sum of the lengths of all edges is 96 cm, 
    prove that its volume is 512 cubic centimeters. -/
theorem cube_volume_from_edge_sum (edge_sum : ℝ) (volume : ℝ) : 
  edge_sum = 96 → volume = (edge_sum / 12)^3 → volume = 512 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_edge_sum_l3002_300204


namespace NUMINAMATH_CALUDE_milk_tea_sales_ratio_l3002_300252

theorem milk_tea_sales_ratio (total_sales : ℕ) (okinawa_ratio : ℚ) (chocolate_sales : ℕ) : 
  total_sales = 50 →
  okinawa_ratio = 3 / 10 →
  chocolate_sales = 15 →
  (total_sales - (okinawa_ratio * total_sales).num - chocolate_sales) * 5 = total_sales * 2 := by
  sorry

end NUMINAMATH_CALUDE_milk_tea_sales_ratio_l3002_300252


namespace NUMINAMATH_CALUDE_bikers_meeting_time_l3002_300215

def biker1_time : ℕ := 12
def biker2_time : ℕ := 18
def biker3_time : ℕ := 24

theorem bikers_meeting_time :
  Nat.lcm (Nat.lcm biker1_time biker2_time) biker3_time = 72 := by
  sorry

end NUMINAMATH_CALUDE_bikers_meeting_time_l3002_300215


namespace NUMINAMATH_CALUDE_hyperbola_and_condition_implies_m_range_l3002_300218

/-- Represents a hyperbola equation -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 3) + y^2 / (m - 4) = 1

/-- Condition for all real x -/
def condition_for_all_x (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m*x + m + 3 ≥ 0

/-- The range of m -/
def m_range (m : ℝ) : Prop :=
  -2 ≤ m ∧ m < 4

theorem hyperbola_and_condition_implies_m_range :
  ∀ m : ℝ, is_hyperbola m ∧ condition_for_all_x m → m_range m :=
sorry

end NUMINAMATH_CALUDE_hyperbola_and_condition_implies_m_range_l3002_300218


namespace NUMINAMATH_CALUDE_derivative_zero_not_sufficient_nor_necessary_l3002_300280

-- Define a real-valued function
variable (f : ℝ → ℝ)
-- Define a real number x
variable (x : ℝ)

-- Define what it means for a function to have an extremum at a point
def has_extremum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- Define the statement to be proved
theorem derivative_zero_not_sufficient_nor_necessary :
  ¬(∀ f x, (deriv f x = 0 → has_extremum f x) ∧ (has_extremum f x → deriv f x = 0)) :=
sorry

end NUMINAMATH_CALUDE_derivative_zero_not_sufficient_nor_necessary_l3002_300280


namespace NUMINAMATH_CALUDE_afternoon_campers_calculation_l3002_300254

-- Define the number of campers who went rowing in the morning
def morning_campers : ℝ := 15.5

-- Define the total number of campers who went rowing that day
def total_campers : ℝ := 32.75

-- Define the number of campers who went rowing in the afternoon
def afternoon_campers : ℝ := total_campers - morning_campers

-- Theorem to prove
theorem afternoon_campers_calculation :
  afternoon_campers = 17.25 := by sorry

end NUMINAMATH_CALUDE_afternoon_campers_calculation_l3002_300254


namespace NUMINAMATH_CALUDE_inequality_solution_inequality_proof_l3002_300230

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part (I)
theorem inequality_solution (x : ℝ) :
  f (x - 1) + f (x + 3) ≥ 6 ↔ x ≤ -3 ∨ x ≥ 3 :=
sorry

-- Theorem for part (II)
theorem inequality_proof (a b : ℝ) (h1 : |a| < 1) (h2 : |b| < 1) (h3 : a ≠ 0) :
  f (a * b) > |a| * f (b / a) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_inequality_proof_l3002_300230


namespace NUMINAMATH_CALUDE_no_binomial_arithmetic_progression_l3002_300232

theorem no_binomial_arithmetic_progression :
  ∀ (n k : ℕ+), k ≤ n →
    ¬∃ (d : ℚ), 
      (Nat.choose n (k + 1) : ℚ) - (Nat.choose n k : ℚ) = d ∧
      (Nat.choose n (k + 2) : ℚ) - (Nat.choose n (k + 1) : ℚ) = d ∧
      (Nat.choose n (k + 3) : ℚ) - (Nat.choose n (k + 2) : ℚ) = d :=
by sorry

end NUMINAMATH_CALUDE_no_binomial_arithmetic_progression_l3002_300232


namespace NUMINAMATH_CALUDE_sum_interior_angles_is_3240_l3002_300284

/-- A regular polygon Q where each interior angle is 9 times its corresponding exterior angle -/
structure RegularPolygon where
  n : ℕ  -- number of sides
  interior_angle : ℝ  -- measure of each interior angle
  exterior_angle : ℝ  -- measure of each exterior angle
  is_regular : interior_angle = 9 * exterior_angle
  sum_exterior : n * exterior_angle = 360

/-- The sum of interior angles of a RegularPolygon -/
def sum_interior_angles (Q : RegularPolygon) : ℝ :=
  Q.n * Q.interior_angle

/-- Theorem: The sum of interior angles of a RegularPolygon is 3240° -/
theorem sum_interior_angles_is_3240 (Q : RegularPolygon) :
  sum_interior_angles Q = 3240 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_is_3240_l3002_300284


namespace NUMINAMATH_CALUDE_pencils_count_l3002_300263

/-- The number of pencils originally in the jar -/
def original_pencils : ℕ := 87

/-- The number of pencils removed from the jar -/
def removed_pencils : ℕ := 4

/-- The number of pencils left in the jar after removal -/
def remaining_pencils : ℕ := 83

/-- Theorem stating that the original number of pencils equals the sum of removed and remaining pencils -/
theorem pencils_count : original_pencils = removed_pencils + remaining_pencils := by
  sorry

end NUMINAMATH_CALUDE_pencils_count_l3002_300263


namespace NUMINAMATH_CALUDE_train_length_l3002_300222

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : speed = 52 → time = 18 → ∃ length : ℝ, abs (length - 259.92) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3002_300222


namespace NUMINAMATH_CALUDE_sine_equality_proof_l3002_300211

theorem sine_equality_proof (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * π / 180) = Real.sin (720 * π / 180) → n = 0 :=
by sorry

end NUMINAMATH_CALUDE_sine_equality_proof_l3002_300211


namespace NUMINAMATH_CALUDE_binomial_15_5_l3002_300219

theorem binomial_15_5 : Nat.choose 15 5 = 3003 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_5_l3002_300219


namespace NUMINAMATH_CALUDE_seventh_term_of_sequence_l3002_300200

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

theorem seventh_term_of_sequence (a₁ q : ℝ) (h₁ : a₁ = 3) (h₂ : q = Real.sqrt 2) :
  geometric_sequence a₁ q 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_sequence_l3002_300200


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l3002_300244

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Ioo (-1) 1

-- State the theorem
theorem domain_of_composite_function :
  {x : ℝ | ∃ y ∈ domain_f, y = 2*x + 1} = Set.Ioo (-1) 0 := by sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l3002_300244


namespace NUMINAMATH_CALUDE_max_discarded_grapes_l3002_300208

theorem max_discarded_grapes (n : ℕ) : ∃ (q : ℕ), n = 7 * q + 6 ∧ 
  ∀ (r : ℕ), r < 7 → n ≠ 7 * (q + 1) + r :=
by sorry

end NUMINAMATH_CALUDE_max_discarded_grapes_l3002_300208


namespace NUMINAMATH_CALUDE_quadruple_primes_l3002_300249

theorem quadruple_primes (p q r : Nat) (n : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ n > 0 ∧ p^2 = q^2 + r^n →
  ((p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4)) :=
by sorry

end NUMINAMATH_CALUDE_quadruple_primes_l3002_300249


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l3002_300266

theorem simplify_and_ratio : ∃ (a b : ℤ), 
  (∀ k, (6 * k + 12) / 6 = a * k + b) ∧ 
  (a : ℚ) / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l3002_300266


namespace NUMINAMATH_CALUDE_expand_product_l3002_300223

theorem expand_product (x : ℝ) : (x + 4) * (x - 7) = x^2 - 3*x - 28 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3002_300223


namespace NUMINAMATH_CALUDE_highest_score_is_174_l3002_300257

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  totalInnings : ℕ
  overallAverage : ℚ
  scoreDifference : ℕ
  averageExcludingExtremes : ℚ

/-- Calculates the highest score of a batsman given their statistics -/
def highestScore (stats : BatsmanStats) : ℚ :=
  let totalRuns := stats.overallAverage * stats.totalInnings
  let runsExcludingExtremes := stats.averageExcludingExtremes * (stats.totalInnings - 2)
  let sumExtremes := totalRuns - runsExcludingExtremes
  (sumExtremes + stats.scoreDifference) / 2

/-- Theorem stating that the highest score is 174 for the given statistics -/
theorem highest_score_is_174 (stats : BatsmanStats)
  (h1 : stats.totalInnings = 46)
  (h2 : stats.overallAverage = 60)
  (h3 : stats.scoreDifference = 140)
  (h4 : stats.averageExcludingExtremes = 58) :
  highestScore stats = 174 := by
  sorry

#eval highestScore {
  totalInnings := 46,
  overallAverage := 60,
  scoreDifference := 140,
  averageExcludingExtremes := 58
}

end NUMINAMATH_CALUDE_highest_score_is_174_l3002_300257


namespace NUMINAMATH_CALUDE_sum_of_remaining_digits_l3002_300297

theorem sum_of_remaining_digits 
  (total_count : Nat) 
  (known_count : Nat) 
  (total_average : ℚ) 
  (known_average : ℚ) 
  (h1 : total_count = 20) 
  (h2 : known_count = 14) 
  (h3 : total_average = 500) 
  (h4 : known_average = 390) :
  (total_count : ℚ) * total_average - (known_count : ℚ) * known_average = 4540 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_remaining_digits_l3002_300297


namespace NUMINAMATH_CALUDE_equal_share_of_tea_l3002_300245

-- Define the total number of cups of tea
def total_cups : ℕ := 10

-- Define the number of people sharing the tea
def num_people : ℕ := 5

-- Define the number of cups each person receives
def cups_per_person : ℚ := total_cups / num_people

-- Theorem to prove
theorem equal_share_of_tea :
  cups_per_person = 2 := by sorry

end NUMINAMATH_CALUDE_equal_share_of_tea_l3002_300245


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l3002_300205

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- State the theorem
theorem set_operations_and_subset :
  (A ∩ B = {x | 3 ≤ x ∧ x < 6}) ∧
  (A ∪ B = {x | 2 < x ∧ x < 9}) ∧
  ({a : ℝ | C a ⊆ B} = {a | 2 ≤ a ∧ a ≤ 8}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l3002_300205


namespace NUMINAMATH_CALUDE_paper_sheets_count_l3002_300231

/-- Represents the dimensions of a rectangle in centimeters -/
structure Dimensions where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℝ := d.width * d.height

/-- Converts meters to centimeters -/
def meters_to_cm (m : ℝ) : ℝ := m * 100

theorem paper_sheets_count :
  let plank : Dimensions := ⟨meters_to_cm 6, meters_to_cm 4⟩
  let paper : Dimensions := ⟨60, 20⟩
  (area plank) / (area paper) = 200 := by sorry

end NUMINAMATH_CALUDE_paper_sheets_count_l3002_300231


namespace NUMINAMATH_CALUDE_problem_solution_l3002_300269

def f (a x : ℝ) : ℝ := |x - 1| + |x + a^2|

theorem problem_solution :
  (∀ x : ℝ, f (Real.sqrt 2) x ≥ 6 ↔ x ≤ -7/2 ∨ x ≥ 5/2) ∧
  (∃ x₀ : ℝ, f a x₀ < 4*a ↔ 2 - Real.sqrt 3 < a ∧ a < 2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3002_300269


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l3002_300234

/-- Proves that the given polar equation is equivalent to the given rectangular equation. -/
theorem polar_to_rectangular_equivalence :
  ∀ (r φ x y : ℝ),
  (r = 2 / (4 - Real.sin φ)) ↔ 
  (x^2 / (2/Real.sqrt 15)^2 + (y - 2/15)^2 / (8/15)^2 = 1 ∧
   x = r * Real.cos φ ∧
   y = r * Real.sin φ) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l3002_300234


namespace NUMINAMATH_CALUDE_solve_sqrt_equation_l3002_300260

theorem solve_sqrt_equation (x : ℝ) (h : x > 0) :
  Real.sqrt ((3 / x) + 3) = 2 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_sqrt_equation_l3002_300260


namespace NUMINAMATH_CALUDE_pies_sold_in_week_l3002_300296

/-- The number of pies sold daily -/
def daily_sales : ℕ := 8

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pies sold in a week -/
def weekly_sales : ℕ := daily_sales * days_in_week

theorem pies_sold_in_week : weekly_sales = 56 := by
  sorry

end NUMINAMATH_CALUDE_pies_sold_in_week_l3002_300296


namespace NUMINAMATH_CALUDE_p_arithmetic_fibonacci_subsequence_l3002_300229

/-- Definition of a p-arithmetic Fibonacci sequence -/
def pArithmeticFibonacci (p : ℕ) (v : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, v (n + 2) = v (n + 1) + v n

/-- Theorem: The terms of a p-arithmetic Fibonacci sequence whose indices are divisible by p
    form another arithmetic Fibonacci sequence -/
theorem p_arithmetic_fibonacci_subsequence (p : ℕ) (v : ℕ → ℕ) 
    (h : pArithmeticFibonacci p v) :
  ∀ n : ℕ, n ≥ 1 → v ((n - 1) * p) + v (n * p) = v ((n + 1) * p) :=
by sorry

end NUMINAMATH_CALUDE_p_arithmetic_fibonacci_subsequence_l3002_300229


namespace NUMINAMATH_CALUDE_company_j_payroll_company_j_payroll_correct_l3002_300250

/-- Calculates the total monthly payroll for factory workers given the conditions of Company J. -/
theorem company_j_payroll (factory_workers : ℕ) (office_workers : ℕ) 
  (office_payroll : ℕ) (salary_difference : ℕ) : ℕ :=
  let factory_workers := 15
  let office_workers := 30
  let office_payroll := 75000
  let salary_difference := 500
  30000

theorem company_j_payroll_correct : 
  company_j_payroll 15 30 75000 500 = 30000 := by sorry

end NUMINAMATH_CALUDE_company_j_payroll_company_j_payroll_correct_l3002_300250


namespace NUMINAMATH_CALUDE_find_divisor_l3002_300298

theorem find_divisor (dividend quotient : ℕ) (h1 : dividend = 62976) (h2 : quotient = 123) :
  dividend / quotient = 512 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3002_300298


namespace NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l3002_300212

theorem sin_negative_thirty_degrees :
  let θ : Real := 30 * Real.pi / 180
  (∀ x, Real.sin (-x) = -Real.sin x) →  -- sine is an odd function
  Real.sin θ = 1/2 →                    -- sin 30° = 1/2
  Real.sin (-θ) = -1/2 := by
    sorry

end NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l3002_300212


namespace NUMINAMATH_CALUDE_ellipse_tangent_collinearity_and_min_area_l3002_300264

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the right focus F
def F : ℝ × ℝ := (1, 0)

-- Define the point P on the line x = 4
def P : ℝ → ℝ × ℝ := λ t => (4, t)

-- Define the tangent points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the area of triangle PAB
def area_PAB (t : ℝ) : ℝ := sorry

theorem ellipse_tangent_collinearity_and_min_area :
  -- Part 1: A, F, and B are collinear
  ∃ k : ℝ, (1 - k) * A.1 + k * B.1 = F.1 ∧ (1 - k) * A.2 + k * B.2 = F.2 ∧
  -- Part 2: The minimum area of triangle PAB is 9/2
  ∃ t : ℝ, area_PAB t = 9/2 ∧ ∀ s : ℝ, area_PAB s ≥ area_PAB t := by
sorry

end NUMINAMATH_CALUDE_ellipse_tangent_collinearity_and_min_area_l3002_300264


namespace NUMINAMATH_CALUDE_dvd_rental_count_l3002_300293

theorem dvd_rental_count (total_spent : ℝ) (cost_per_dvd : ℝ) (h1 : total_spent = 4.8) (h2 : cost_per_dvd = 1.2) :
  total_spent / cost_per_dvd = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_dvd_rental_count_l3002_300293


namespace NUMINAMATH_CALUDE_horner_method_f_2_l3002_300256

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem horner_method_f_2 :
  f 2 = horner_eval [4, 0, -3, 2, 5, 1] 2 ∧ horner_eval [4, 0, -3, 2, 5, 1] 2 = 123 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_2_l3002_300256


namespace NUMINAMATH_CALUDE_floor_square_minus_floor_product_l3002_300271

theorem floor_square_minus_floor_product (x : ℝ) : x = 12.7 → ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 17 := by
  sorry

end NUMINAMATH_CALUDE_floor_square_minus_floor_product_l3002_300271


namespace NUMINAMATH_CALUDE_drama_club_neither_math_nor_physics_l3002_300241

theorem drama_club_neither_math_nor_physics 
  (total : ℕ) 
  (math : ℕ) 
  (physics : ℕ) 
  (both : ℕ) 
  (h1 : total = 60) 
  (h2 : math = 36) 
  (h3 : physics = 27) 
  (h4 : both = 20) : 
  total - (math + physics - both) = 17 := by
  sorry

end NUMINAMATH_CALUDE_drama_club_neither_math_nor_physics_l3002_300241


namespace NUMINAMATH_CALUDE_twentieth_century_power_diff_l3002_300288

def is_20th_century (year : ℕ) : Prop := 1900 ≤ year ∧ year ≤ 1999

def is_power_diff (year : ℕ) : Prop :=
  ∃ (n k : ℕ), year = 2^n - 2^k

theorem twentieth_century_power_diff :
  {year : ℕ | is_20th_century year ∧ is_power_diff year} = {1984, 1920} := by
  sorry

end NUMINAMATH_CALUDE_twentieth_century_power_diff_l3002_300288


namespace NUMINAMATH_CALUDE_smallest_block_volume_l3002_300239

theorem smallest_block_volume (a b c : ℕ) : 
  (a - 1) * (b - 1) * (c - 1) = 240 → 
  a * b * c ≥ 385 ∧ 
  ∃ (a₀ b₀ c₀ : ℕ), (a₀ - 1) * (b₀ - 1) * (c₀ - 1) = 240 ∧ a₀ * b₀ * c₀ = 385 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l3002_300239


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3002_300255

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 12) : 
  x^2 + 2*x*y + 3*y^2 ≤ 18 + 12*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3002_300255


namespace NUMINAMATH_CALUDE_simplify_fraction_l3002_300290

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3002_300290


namespace NUMINAMATH_CALUDE_min_sum_of_squares_min_sum_of_squares_value_l3002_300228

theorem min_sum_of_squares (a b c d : ℤ) : 
  a^2 ≠ b^2 → a^2 ≠ c^2 → a^2 ≠ d^2 → b^2 ≠ c^2 → b^2 ≠ d^2 → c^2 ≠ d^2 →
  (a*b + c*d)^2 + (a*d - b*c)^2 = 2004 →
  ∀ (w x y z : ℤ), w^2 ≠ x^2 → w^2 ≠ y^2 → w^2 ≠ z^2 → x^2 ≠ y^2 → x^2 ≠ z^2 → y^2 ≠ z^2 →
  (w*x + y*z)^2 + (w*z - x*y)^2 = 2004 →
  a^2 + b^2 + c^2 + d^2 ≤ w^2 + x^2 + y^2 + z^2 :=
by sorry

theorem min_sum_of_squares_value (a b c d : ℤ) : 
  a^2 ≠ b^2 → a^2 ≠ c^2 → a^2 ≠ d^2 → b^2 ≠ c^2 → b^2 ≠ d^2 → c^2 ≠ d^2 →
  (a*b + c*d)^2 + (a*d - b*c)^2 = 2004 →
  ∃ (x y : ℤ), x^2 + y^2 = 2004 ∧ a^2 + b^2 + c^2 + d^2 = 2 * (x + y) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_min_sum_of_squares_value_l3002_300228


namespace NUMINAMATH_CALUDE_circle_c_properties_l3002_300225

-- Define the circle C
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line l
structure LineL where
  b : ℝ

-- Define point N
def pointN : ℝ × ℝ := (0, 3)

-- Define the theorem
theorem circle_c_properties (c : CircleC) (l : LineL) :
  -- Condition 1: Circle C's center is on the line x - 2y = 0
  c.center.1 = 2 * c.center.2 →
  -- Condition 2: Circle C is tangent to the positive half of the y-axis
  c.center.2 > 0 →
  -- Condition 3: The chord obtained by intersecting the x-axis is 2√3 long
  2 * Real.sqrt 3 = 2 * Real.sqrt (c.radius^2 - c.center.2^2) →
  -- Condition 4: Line l intersects circle C at two points
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2 ∧
    A.2 = -2 * A.1 + l.b ∧ B.2 = -2 * B.1 + l.b →
  -- Condition 5: The circle with AB as its diameter passes through the origin
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2 ∧
    A.2 = -2 * A.1 + l.b ∧ B.2 = -2 * B.1 + l.b ∧
    A.1 * B.1 + A.2 * B.2 = 0 →
  -- Condition 6-9 are implicitly included in the structure of CircleC
  -- Prove:
  -- 1. The standard equation of circle C is (x - 2)² + (y - 1)² = 4
  ((c.center = (2, 1) ∧ c.radius = 2) ∨
  -- 2. The value of b in the equation y = -2x + b is (5 ± √15) / 2
   (l.b = (5 + Real.sqrt 15) / 2 ∨ l.b = (5 - Real.sqrt 15) / 2)) ∧
  -- 3. The y-coordinate of the center of circle C is in the range (0, 2]
   (0 < c.center.2 ∧ c.center.2 ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_c_properties_l3002_300225


namespace NUMINAMATH_CALUDE_three_lines_intersection_l3002_300283

/-- A line in the plane represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of intersection points between three lines -/
def num_intersections (l1 l2 l3 : Line) : ℕ :=
  sorry

theorem three_lines_intersection :
  let l1 : Line := { a := -4, b := 6, c := 2 }
  let l2 : Line := { a := 1, b := 2, c := 2 }
  let l3 : Line := { a := -4, b := 6, c := 3 }
  num_intersections l1 l2 l3 = 2 :=
by sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l3002_300283


namespace NUMINAMATH_CALUDE_range_of_m_l3002_300273

-- Define the propositions p and q
def p (x : ℝ) : Prop := x + 2 ≥ 0 ∧ x - 10 ≤ 0

def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)

def not_q (x m : ℝ) : Prop := ¬(q x m)

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (m : ℝ) : Prop :=
  (∀ x, not_q x m → not_p x) ∧ 
  (∃ x, not_p x ∧ ¬(not_q x m))

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, necessary_not_sufficient m ↔ m ≥ 9 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l3002_300273


namespace NUMINAMATH_CALUDE_brown_eyes_fraction_l3002_300201

theorem brown_eyes_fraction (total_students : ℕ) 
  (brown_eyes_black_hair : ℕ) 
  (h1 : total_students = 18) 
  (h2 : brown_eyes_black_hair = 6) 
  (h3 : brown_eyes_black_hair * 2 = brown_eyes_black_hair + brown_eyes_black_hair) :
  (brown_eyes_black_hair * 2 : ℚ) / total_students = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_brown_eyes_fraction_l3002_300201
