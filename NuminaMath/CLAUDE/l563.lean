import Mathlib

namespace NUMINAMATH_CALUDE_no_valid_positive_x_for_equal_volume_increase_l563_56397

theorem no_valid_positive_x_for_equal_volume_increase (x : ℝ) : 
  x > 0 → 
  π * (5 + x)^2 * 10 - π * 5^2 * 10 ≠ π * 5^2 * (10 + x) - π * 5^2 * 10 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_positive_x_for_equal_volume_increase_l563_56397


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l563_56396

/-- Given a geometric sequence {a_n} where a_1 = 2 and a_1 + a_3 + a_5 = 14,
    prove that 1/a_1 + 1/a_3 + 1/a_5 = 7/8 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : a 1 = 2) 
    (h2 : a 1 + a 3 + a 5 = 14) 
    (h3 : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = a n * q) :
    1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l563_56396


namespace NUMINAMATH_CALUDE_continued_fraction_sum_l563_56307

theorem continued_fraction_sum (v w x y z : ℕ+) : 
  (v : ℚ) + 1 / ((w : ℚ) + 1 / ((x : ℚ) + 1 / ((y : ℚ) + 1 / (z : ℚ)))) = 222 / 155 →
  10^4 * v.val + 10^3 * w.val + 10^2 * x.val + 10 * y.val + z.val = 12354 := by
sorry

end NUMINAMATH_CALUDE_continued_fraction_sum_l563_56307


namespace NUMINAMATH_CALUDE_one_man_work_time_l563_56319

-- Define the work as a unit
def total_work : ℝ := 1

-- Define the time taken by the group
def group_time : ℝ := 6

-- Define the number of men and women in the group
def num_men : ℝ := 10
def num_women : ℝ := 15

-- Define the time taken by one woman
def woman_time : ℝ := 225

-- Define the time taken by one man (to be proved)
def man_time : ℝ := 100

-- Theorem statement
theorem one_man_work_time :
  (num_men / man_time + num_women / woman_time) * group_time = total_work →
  1 / man_time = 1 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_one_man_work_time_l563_56319


namespace NUMINAMATH_CALUDE_grants_test_score_l563_56300

theorem grants_test_score (hunter_score john_score grant_score : ℕ) :
  hunter_score = 45 →
  john_score = 2 * hunter_score →
  grant_score = john_score + 10 →
  grant_score = 100 := by
sorry

end NUMINAMATH_CALUDE_grants_test_score_l563_56300


namespace NUMINAMATH_CALUDE_problem_solution_l563_56368

theorem problem_solution (a b : ℚ) 
  (h1 : 5 + a = 7 - b) 
  (h2 : 3 + b = 8 + a) : 
  4 - a = 11/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l563_56368


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l563_56369

theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 - (2*k - 1)*x₁ + k^2 - k = 0 ∧
  x₂^2 - (2*k - 1)*x₂ + k^2 - k = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l563_56369


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l563_56341

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = -103 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l563_56341


namespace NUMINAMATH_CALUDE_common_y_intercept_l563_56344

theorem common_y_intercept (l₁ l₂ l₃ : ℝ → ℝ) (b : ℝ) :
  (∀ x, l₁ x = 1/2 * x + b) →
  (∀ x, l₂ x = 1/3 * x + b) →
  (∀ x, l₃ x = 1/4 * x + b) →
  ((-2*b) + (-3*b) + (-4*b) = 36) →
  b = -4 := by sorry

end NUMINAMATH_CALUDE_common_y_intercept_l563_56344


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l563_56395

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the concept of symmetry with respect to the origin
def symmetricToOrigin (p q : Point2D) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

-- Define the fourth quadrant
def inFourthQuadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

-- Theorem statement
theorem point_in_fourth_quadrant (a : ℝ) (P P_1 : Point2D) :
  a < 0 →
  P = Point2D.mk (-a^2 - 1) (-a + 3) →
  symmetricToOrigin P P_1 →
  inFourthQuadrant P_1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l563_56395


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l563_56317

theorem sqrt_sum_simplification :
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 = (a * Real.sqrt 3 + b * Real.sqrt 11) / c) ∧
    (∀ (a' b' c' : ℕ), 
      (a' > 0 ∧ b' > 0 ∧ c' > 0) →
      (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 = (a' * Real.sqrt 3 + b' * Real.sqrt 11) / c') →
      c ≤ c') ∧
    a + b + c = 113 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l563_56317


namespace NUMINAMATH_CALUDE_ferry_tourists_count_l563_56332

/-- Calculates the total number of tourists transported by a ferry -/
def total_tourists (trips : ℕ) (initial_tourists : ℕ) (decrease : ℕ) : ℕ :=
  trips * (2 * initial_tourists - (trips - 1) * decrease) / 2

/-- Proves that the total number of tourists transported is 798 -/
theorem ferry_tourists_count :
  total_tourists 7 120 2 = 798 := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourists_count_l563_56332


namespace NUMINAMATH_CALUDE_fraction_classification_l563_56381

-- Define a fraction as a pair of integers (numerator, denominator)
def Fraction := ℤ × ℤ

-- Define proper fractions
def ProperFraction (f : Fraction) : Prop := f.1.natAbs < f.2.natAbs ∧ f.2 ≠ 0

-- Define improper fractions
def ImproperFraction (f : Fraction) : Prop := f.1.natAbs ≥ f.2.natAbs ∧ f.2 ≠ 0

-- Theorem stating that all fractions are either proper or improper
theorem fraction_classification (f : Fraction) : f.2 ≠ 0 → ProperFraction f ∨ ImproperFraction f :=
sorry

end NUMINAMATH_CALUDE_fraction_classification_l563_56381


namespace NUMINAMATH_CALUDE_subtraction_problem_l563_56388

theorem subtraction_problem : 
  (845.59 : ℝ) - 249.27 = 596.32 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l563_56388


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l563_56322

theorem average_of_six_numbers (sequence : Fin 6 → ℝ) 
  (h1 : (sequence 0 + sequence 1 + sequence 2 + sequence 3) / 4 = 25)
  (h2 : (sequence 3 + sequence 4 + sequence 5) / 3 = 35)
  (h3 : sequence 3 = 25) :
  (sequence 0 + sequence 1 + sequence 2 + sequence 3 + sequence 4 + sequence 5) / 6 = 30 := by
sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l563_56322


namespace NUMINAMATH_CALUDE_beach_shells_problem_l563_56360

theorem beach_shells_problem (jillian_shells savannah_shells clayton_shells : ℕ) 
  (friend_count friend_received : ℕ) :
  jillian_shells = 29 →
  clayton_shells = 8 →
  friend_count = 2 →
  friend_received = 27 →
  jillian_shells + savannah_shells + clayton_shells = friend_count * friend_received →
  savannah_shells = 17 := by
  sorry

end NUMINAMATH_CALUDE_beach_shells_problem_l563_56360


namespace NUMINAMATH_CALUDE_pencil_distribution_l563_56382

/-- Given an initial number of pencils, number of containers, and additional pencils,
    calculate the number of pencils that can be evenly distributed per container. -/
def evenlyDistributedPencils (initialPencils : ℕ) (containers : ℕ) (additionalPencils : ℕ) : ℕ :=
  (initialPencils + additionalPencils) / containers

theorem pencil_distribution (initialPencils : ℕ) (containers : ℕ) (additionalPencils : ℕ) 
    (h1 : initialPencils = 150)
    (h2 : containers = 5)
    (h3 : additionalPencils = 30) :
  evenlyDistributedPencils initialPencils containers additionalPencils = 36 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l563_56382


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l563_56305

/-- Given an infinite geometric series with common ratio 1/4 and sum 16,
    the second term of the sequence is 3. -/
theorem second_term_of_geometric_series (a : ℝ) :
  (∑' n, a * (1/4)^n = 16) → a * (1/4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l563_56305


namespace NUMINAMATH_CALUDE_inequality_solution_and_absolute_value_bound_l563_56386

-- Define the solution set
def solution_set : Set ℝ := Set.Icc (-1) 2

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := |2 * x - a| ≤ 3

-- Theorem statement
theorem inequality_solution_and_absolute_value_bound (a : ℝ) :
  (∀ x, x ∈ solution_set ↔ inequality a x) →
  (a = 1 ∧ ∀ x m, |x - m| < a → |x| < |m| + 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_and_absolute_value_bound_l563_56386


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l563_56376

theorem simplify_complex_fraction :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7 + Real.sqrt 9) =
  -(2 * Real.sqrt 6 - 2 * Real.sqrt 2 + 2 * Real.sqrt 14) / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l563_56376


namespace NUMINAMATH_CALUDE_total_weight_is_130_l563_56331

-- Define the weights as real numbers
variable (M D C : ℝ)

-- State the conditions
variable (h1 : D + C = 60)
variable (h2 : C = (1/5) * M)
variable (h3 : D = 46)

-- Theorem to prove
theorem total_weight_is_130 : M + D + C = 130 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_130_l563_56331


namespace NUMINAMATH_CALUDE_sum_to_zero_l563_56391

/-- Given an initial sum of 2b - 1, where one addend is increased by 3b - 8 and another is decreased by -b - 7,
    prove that subtracting 6b - 2 from the third addend makes the total sum zero. -/
theorem sum_to_zero (b : ℝ) : 
  let initial_sum := 2*b - 1
  let increase := 3*b - 8
  let decrease := -b - 7
  let subtraction := 6*b - 2
  initial_sum + increase - decrease - subtraction = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_to_zero_l563_56391


namespace NUMINAMATH_CALUDE_park_fencing_cost_l563_56370

/-- Proves that the cost of fencing a rectangular park with given dimensions and fencing cost is 175 rupees -/
theorem park_fencing_cost (length width area perimeter_cost : ℝ) : 
  length / width = 3 / 2 →
  length * width = 3750 →
  perimeter_cost = 0.7 →
  (2 * length + 2 * width) * perimeter_cost = 175 := by
  sorry

#check park_fencing_cost

end NUMINAMATH_CALUDE_park_fencing_cost_l563_56370


namespace NUMINAMATH_CALUDE_eight_members_left_for_treasurer_l563_56385

/-- Represents a club with members and officer positions -/
structure Club where
  totalMembers : ℕ
  presidentChosen : Bool
  secretaryChosen : Bool

/-- Function to calculate remaining members for treasurer position -/
def remainingMembersForTreasurer (club : Club) : ℕ :=
  club.totalMembers - (if club.presidentChosen then 1 else 0) - (if club.secretaryChosen then 1 else 0)

/-- Theorem stating that in a club of 10 members, after choosing president and secretary,
    there are 8 members left for treasurer position -/
theorem eight_members_left_for_treasurer (club : Club) 
    (h1 : club.totalMembers = 10)
    (h2 : club.presidentChosen = true)
    (h3 : club.secretaryChosen = true) :
  remainingMembersForTreasurer club = 8 := by
  sorry

#eval remainingMembersForTreasurer { totalMembers := 10, presidentChosen := true, secretaryChosen := true }

end NUMINAMATH_CALUDE_eight_members_left_for_treasurer_l563_56385


namespace NUMINAMATH_CALUDE_probability_two_even_balls_l563_56390

theorem probability_two_even_balls (n : ℕ) (h1 : n = 16) :
  let total_balls := n
  let even_balls := n / 2
  let prob_first_even := even_balls / total_balls
  let prob_second_even := (even_balls - 1) / (total_balls - 1)
  prob_first_even * prob_second_even = 7 / 30 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_two_even_balls_l563_56390


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l563_56379

/-- Given a circle x^2 + y^2 = 16 and a line y = x + b, if there are at least three points
    on the circle with a distance of 1 from the line, then -3√2 ≤ b ≤ 3√2 -/
theorem circle_line_distance_range (b : ℝ) :
  (∃ (p q r : ℝ × ℝ),
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (p.1^2 + p.2^2 = 16) ∧ (q.1^2 + q.2^2 = 16) ∧ (r.1^2 + r.2^2 = 16) ∧
    (abs (p.1 - p.2 + b) / Real.sqrt 2 = 1) ∧
    (abs (q.1 - q.2 + b) / Real.sqrt 2 = 1) ∧
    (abs (r.1 - r.2 + b) / Real.sqrt 2 = 1)) →
  -3 * Real.sqrt 2 ≤ b ∧ b ≤ 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_circle_line_distance_range_l563_56379


namespace NUMINAMATH_CALUDE_bruce_payment_l563_56343

/-- Calculates the total amount Bruce paid for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1110 for his purchase -/
theorem bruce_payment : total_amount 8 70 10 55 = 1110 := by
  sorry

end NUMINAMATH_CALUDE_bruce_payment_l563_56343


namespace NUMINAMATH_CALUDE_janette_beef_jerky_left_l563_56350

/-- Calculates the number of beef jerky pieces Janette has left after her camping trip and giving half to her brother. -/
def beef_jerky_left (days : ℕ) (initial_pieces : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) : ℕ :=
  let daily_consumption := breakfast + lunch + dinner
  let total_consumption := daily_consumption * days
  let remaining_after_trip := initial_pieces - total_consumption
  remaining_after_trip / 2

/-- Theorem stating that Janette will have 10 pieces of beef jerky left. -/
theorem janette_beef_jerky_left :
  beef_jerky_left 5 40 1 1 2 = 10 := by
  sorry

#eval beef_jerky_left 5 40 1 1 2

end NUMINAMATH_CALUDE_janette_beef_jerky_left_l563_56350


namespace NUMINAMATH_CALUDE_hiking_team_gloves_l563_56372

/-- The minimum number of gloves needed for a hiking team -/
def minimum_gloves (total_participants small_members medium_members large_members num_activities : ℕ) : ℕ :=
  (small_members + medium_members + large_members) * num_activities

/-- Theorem: The hiking team needs 225 gloves -/
theorem hiking_team_gloves :
  let total_participants := 75
  let small_members := 20
  let medium_members := 38
  let large_members := 17
  let num_activities := 3
  minimum_gloves total_participants small_members medium_members large_members num_activities = 225 := by
  sorry


end NUMINAMATH_CALUDE_hiking_team_gloves_l563_56372


namespace NUMINAMATH_CALUDE_dads_strawberries_weight_l563_56357

/-- The weight of Marco's dad's strawberries -/
def dads_strawberries (total_weight marco_weight : ℕ) : ℕ :=
  total_weight - marco_weight

/-- Theorem stating that Marco's dad's strawberries weigh 9 pounds -/
theorem dads_strawberries_weight :
  dads_strawberries 23 14 = 9 := by
  sorry

end NUMINAMATH_CALUDE_dads_strawberries_weight_l563_56357


namespace NUMINAMATH_CALUDE_unique_solution_when_p_equals_two_l563_56325

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^(1/3) + (2 - x)^(1/3)

-- State the theorem
theorem unique_solution_when_p_equals_two :
  ∃! p : ℝ, ∃! x : ℝ, f x = p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_when_p_equals_two_l563_56325


namespace NUMINAMATH_CALUDE_fibonacci_unique_triple_l563_56365

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def satisfies_conditions (a b c : ℕ) : Prop :=
  b < a ∧ c < a ∧ ∀ n, (fibonacci n - n * b * c^n) % a = 0

theorem fibonacci_unique_triple :
  ∃! (triple : ℕ × ℕ × ℕ), 
    let (a, b, c) := triple
    satisfies_conditions a b c ∧ a = 5 ∧ b = 2 ∧ c = 3 :=
sorry

end NUMINAMATH_CALUDE_fibonacci_unique_triple_l563_56365


namespace NUMINAMATH_CALUDE_all_fruits_fall_on_day_14_l563_56348

/-- The number of fruits on the tree initially -/
def initial_fruits : ℕ := 60

/-- The number of fruits that fall on day n according to the original pattern -/
def fruits_falling (n : ℕ) : ℕ := n

/-- The sum of fruits that have fallen up to day n according to the original pattern -/
def sum_fallen (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of fruits remaining on the tree after n days of the original pattern -/
def fruits_remaining (n : ℕ) : ℕ := max 0 (initial_fruits - sum_fallen n)

/-- The day when the original pattern stops -/
def pattern_stop_day : ℕ := 10

/-- The number of days needed to finish the remaining fruits after the original pattern stops -/
def additional_days : ℕ := fruits_remaining pattern_stop_day

/-- The total number of days needed for all fruits to fall -/
def total_days : ℕ := pattern_stop_day + additional_days - 1

theorem all_fruits_fall_on_day_14 : total_days = 14 := by
  sorry

end NUMINAMATH_CALUDE_all_fruits_fall_on_day_14_l563_56348


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l563_56342

/-- The eccentricity of a hyperbola with equation x²/2 - y² = 1 is √6/2 -/
theorem hyperbola_eccentricity : 
  let hyperbola := {(x, y) : ℝ × ℝ | x^2/2 - y^2 = 1}
  ∃ e : ℝ, e = (Real.sqrt 6) / 2 ∧ 
    ∀ (a b c : ℝ), 
      (a^2 = 2 ∧ b^2 = 1 ∧ c^2 = a^2 + b^2) → 
      e = c / a :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l563_56342


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_right_angle_l563_56363

theorem triangle_angle_ratio_right_angle (α β γ : ℝ) (h_sum : α + β + γ = π) 
  (h_ratio : α = 3 * γ ∧ β = 2 * γ) : α = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_right_angle_l563_56363


namespace NUMINAMATH_CALUDE_power_of_power_three_l563_56392

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l563_56392


namespace NUMINAMATH_CALUDE_system_solution_l563_56349

theorem system_solution (x y z : ℝ) 
  (eq1 : x^2 + 27 = -8*y + 10*z)
  (eq2 : y^2 + 196 = 18*z + 13*x)
  (eq3 : z^2 + 119 = -3*x + 30*y) :
  x + 3*y + 5*z = 127.5 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l563_56349


namespace NUMINAMATH_CALUDE_probability_less_than_4_l563_56330

/-- A square in the 2D plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point in the square satisfies a condition -/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The specific square with vertices (0,0), (0,3), (3,3), and (3,0) -/
def specificSquare : Square :=
  { bottomLeft := (0, 0), sideLength := 3 }

/-- The condition x + y < 4 -/
def conditionLessThan4 (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 4

theorem probability_less_than_4 :
  probability specificSquare conditionLessThan4 = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_4_l563_56330


namespace NUMINAMATH_CALUDE_exponent_multiplication_l563_56338

theorem exponent_multiplication (a : ℝ) : a^2 * a^6 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l563_56338


namespace NUMINAMATH_CALUDE_benny_total_spend_l563_56378

def total_cost (soft_drink_cost : ℕ) (candy_bar_cost : ℕ) (num_candy_bars : ℕ) : ℕ :=
  soft_drink_cost + candy_bar_cost * num_candy_bars

theorem benny_total_spend :
  let soft_drink_cost : ℕ := 2
  let candy_bar_cost : ℕ := 5
  let num_candy_bars : ℕ := 5
  total_cost soft_drink_cost candy_bar_cost num_candy_bars = 27 := by
sorry

end NUMINAMATH_CALUDE_benny_total_spend_l563_56378


namespace NUMINAMATH_CALUDE_fraction_of_x_l563_56303

theorem fraction_of_x (w x y f : ℝ) : 
  2/w + f*x = 2/y → 
  w*x = y → 
  (w + x)/2 = 0.5 → 
  f = 2/x - 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_x_l563_56303


namespace NUMINAMATH_CALUDE_apples_left_l563_56310

/-- Proves that given the conditions in the problem, the number of boxes of apples left is 3 -/
theorem apples_left (saturday_boxes : ℕ) (sunday_boxes : ℕ) (apples_per_box : ℕ) (apples_sold : ℕ) :
  saturday_boxes = 50 →
  sunday_boxes = 25 →
  apples_per_box = 10 →
  apples_sold = 720 →
  (saturday_boxes + sunday_boxes) * apples_per_box - apples_sold = 3 * apples_per_box :=
by sorry

end NUMINAMATH_CALUDE_apples_left_l563_56310


namespace NUMINAMATH_CALUDE_line_slope_l563_56339

theorem line_slope (x y : ℝ) : 
  (x / 4 + y / 3 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l563_56339


namespace NUMINAMATH_CALUDE_expression_evaluation_l563_56366

theorem expression_evaluation (a b c : ℚ) (ha : a = 12) (hb : b = 15) (hc : c = 19) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b) + a) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b) + 1) = a + b + c - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l563_56366


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l563_56334

theorem max_sum_on_circle : ∀ x y : ℤ, x^2 + y^2 = 20 → x + y ≤ 6 := by sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l563_56334


namespace NUMINAMATH_CALUDE_max_a_for_nonpositive_f_existence_of_m_for_a_eq_1_max_a_equals_one_l563_56327

theorem max_a_for_nonpositive_f (a : ℝ) : 
  (∃ m : ℝ, m > 0 ∧ m^3 - a*m^2 + (a^2 - 2)*m + 1 ≤ 0) → a ≤ 1 :=
by sorry

theorem existence_of_m_for_a_eq_1 : 
  ∃ m : ℝ, m > 0 ∧ m^3 - m^2 + (1^2 - 2)*m + 1 ≤ 0 :=
by sorry

theorem max_a_equals_one : 
  (∃ a : ℝ, (∃ m : ℝ, m > 0 ∧ m^3 - a*m^2 + (a^2 - 2)*m + 1 ≤ 0) ∧ 
    ∀ b : ℝ, (∃ n : ℝ, n > 0 ∧ n^3 - b*n^2 + (b^2 - 2)*n + 1 ≤ 0) → b ≤ a) ∧
  (∃ m : ℝ, m > 0 ∧ m^3 - 1*m^2 + (1^2 - 2)*m + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_nonpositive_f_existence_of_m_for_a_eq_1_max_a_equals_one_l563_56327


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l563_56361

theorem inverse_variation_problem (y x : ℝ) (k : ℝ) :
  (∀ x y, y * x^2 = k) →  -- y varies inversely as x^2
  (1 * 4^2 = k) →         -- when x = 4, y = 1
  (0.25 * x^2 = k) →      -- condition for y = 0.25
  x = 8 :=                -- prove x = 8
by
  sorry

#check inverse_variation_problem

end NUMINAMATH_CALUDE_inverse_variation_problem_l563_56361


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l563_56323

theorem quadratic_roots_problem (m : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x : ℝ, x^2 - (2*m - 1)*x + m^2 = 0) →
  (x₁^2 - (2*m - 1)*x₁ + m^2 = 0) →
  (x₂^2 - (2*m - 1)*x₂ + m^2 = 0) →
  (x₁ ≠ x₂) →
  ((x₁ + 1) * (x₂ + 1) = 3) →
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l563_56323


namespace NUMINAMATH_CALUDE_disprove_statement_l563_56314

theorem disprove_statement : ∃ (a b c : ℤ), a > b ∧ b > c ∧ ¬(a + b > c) :=
  sorry

end NUMINAMATH_CALUDE_disprove_statement_l563_56314


namespace NUMINAMATH_CALUDE_original_slices_count_l563_56399

/-- The number of slices in the original loaf of bread -/
def S : ℕ := 27

/-- The number of slices Andy ate -/
def slices_andy_ate : ℕ := 6

/-- The number of slices Emma used for toast -/
def slices_for_toast : ℕ := 20

/-- The number of slices left after making toast -/
def slices_left : ℕ := 1

/-- Theorem stating that the original number of slices equals the sum of slices eaten,
    used for toast, and left over -/
theorem original_slices_count : S = slices_andy_ate + slices_for_toast + slices_left :=
by sorry

end NUMINAMATH_CALUDE_original_slices_count_l563_56399


namespace NUMINAMATH_CALUDE_point_on_line_between_l563_56362

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Check if a point is between two other points -/
def between (p q r : Point) : Prop :=
  collinear p q r ∧
  min p.x r.x ≤ q.x ∧ q.x ≤ max p.x r.x ∧
  min p.y r.y ≤ q.y ∧ q.y ≤ max p.y r.y

theorem point_on_line_between (p₁ p₂ q : Point) 
  (h₁ : p₁ = ⟨8, 16⟩) 
  (h₂ : p₂ = ⟨2, 6⟩)
  (h₃ : q = ⟨5, 11⟩) : 
  between p₁ q p₂ := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_between_l563_56362


namespace NUMINAMATH_CALUDE_rectangle_circle_ratio_l563_56315

theorem rectangle_circle_ratio (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ)
  (h1 : square_area = 1225)
  (h2 : rectangle_area = 140)
  (h3 : rectangle_breadth = 10) :
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_ratio_l563_56315


namespace NUMINAMATH_CALUDE_angle_measure_l563_56318

theorem angle_measure (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l563_56318


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_transform_l563_56367

-- Define the polynomial
def p (x : ℝ) : ℝ := 15 * x^3 - 35 * x^2 + 20 * x - 2

-- Theorem statement
theorem root_sum_reciprocal_transform (a b c : ℝ) :
  p a = 0 → p b = 0 → p c = 0 →  -- a, b, c are roots of p
  a ≠ b → b ≠ c → a ≠ c →        -- roots are distinct
  0 < a → a < 1 →                -- 0 < a < 1
  0 < b → b < 1 →                -- 0 < b < 1
  0 < c → c < 1 →                -- 0 < c < 1
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_transform_l563_56367


namespace NUMINAMATH_CALUDE_max_chosen_squares_29x29_l563_56356

/-- The maximum number of squares that can be chosen on an n×n chessboard 
    such that for every selected square, there exists at most one square 
    with both row and column numbers greater than or equal to the selected 
    square's row and column numbers. -/
def max_chosen_squares (n : ℕ) : ℕ :=
  if n % 2 = 0 then 3 * (n / 2) else 3 * (n / 2) + 1

/-- Theorem stating that the maximum number of chosen squares for a 29×29 chessboard is 43. -/
theorem max_chosen_squares_29x29 : max_chosen_squares 29 = 43 := by
  sorry

end NUMINAMATH_CALUDE_max_chosen_squares_29x29_l563_56356


namespace NUMINAMATH_CALUDE_robin_chocolate_chip_cookies_l563_56387

/-- Given information about Robin's cookies --/
structure CookieInfo where
  cookies_per_bag : ℕ
  oatmeal_cookies : ℕ
  baggies : ℕ

/-- Calculate the number of chocolate chip cookies --/
def chocolate_chip_cookies (info : CookieInfo) : ℕ :=
  info.baggies * info.cookies_per_bag - info.oatmeal_cookies

/-- Theorem: Robin has 23 chocolate chip cookies --/
theorem robin_chocolate_chip_cookies :
  let info : CookieInfo := {
    cookies_per_bag := 6,
    oatmeal_cookies := 25,
    baggies := 8
  }
  chocolate_chip_cookies info = 23 := by
  sorry

end NUMINAMATH_CALUDE_robin_chocolate_chip_cookies_l563_56387


namespace NUMINAMATH_CALUDE_dandelion_counts_l563_56393

/-- Represents the state of dandelions in the meadow on a given day -/
structure DandelionState where
  yellow : ℕ
  white : ℕ

/-- The lifecycle of dandelions -/
def dandelionLifecycle : Prop :=
  ∀ d : DandelionState, d.yellow = d.white

/-- Yesterday's dandelion state -/
def yesterday : DandelionState :=
  { yellow := 20, white := 14 }

/-- Today's dandelion state -/
def today : DandelionState :=
  { yellow := 15, white := 11 }

/-- Theorem: Given the dandelion lifecycle and the counts for yesterday and today,
    the number of yellow dandelions the day before yesterday was 25, and
    the number of white dandelions tomorrow will be 9. -/
theorem dandelion_counts
  (h : dandelionLifecycle)
  (hy : yesterday.yellow = 20 ∧ yesterday.white = 14)
  (ht : today.yellow = 15 ∧ today.white = 11) :
  (yesterday.white + today.white = 25) ∧
  (yesterday.yellow - today.white = 9) :=
by sorry

end NUMINAMATH_CALUDE_dandelion_counts_l563_56393


namespace NUMINAMATH_CALUDE_real_sqrt_reciprocal_range_l563_56328

theorem real_sqrt_reciprocal_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (5 - x)) ↔ x < 5 := by sorry

end NUMINAMATH_CALUDE_real_sqrt_reciprocal_range_l563_56328


namespace NUMINAMATH_CALUDE_mother_age_four_times_yujeong_age_l563_56364

/-- Represents the age difference between the current year and the year in question -/
def yearDifference : ℕ := 2

/-- Yujeong's current age -/
def yujeongCurrentAge : ℕ := 12

/-- Yujeong's mother's current age -/
def motherCurrentAge : ℕ := 42

/-- Theorem stating that 2 years ago, Yujeong's mother's age was 4 times Yujeong's age -/
theorem mother_age_four_times_yujeong_age :
  (motherCurrentAge - yearDifference) = 4 * (yujeongCurrentAge - yearDifference) := by
  sorry

end NUMINAMATH_CALUDE_mother_age_four_times_yujeong_age_l563_56364


namespace NUMINAMATH_CALUDE_diagonals_not_parallel_32gon_l563_56333

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals parallel to sides in a regular polygon with n sides -/
def num_parallel_diagonals (n : ℕ) : ℕ := (n / 2) * ((n - 4) / 2)

/-- The number of diagonals not parallel to any side in a regular 32-sided polygon -/
theorem diagonals_not_parallel_32gon : 
  num_diagonals 32 - num_parallel_diagonals 32 = 240 := by
  sorry


end NUMINAMATH_CALUDE_diagonals_not_parallel_32gon_l563_56333


namespace NUMINAMATH_CALUDE_second_player_can_prevent_win_l563_56306

/-- Represents a position on the infinite grid -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Represents a move in the game -/
inductive Move
  | X (pos : Position)
  | O (pos : Position)

/-- Represents the game state -/
def GameState := List Move

/-- A strategy for the second player -/
def Strategy := GameState → Position

/-- Checks if a list of positions contains 11 consecutive X's -/
def hasElevenConsecutiveXs (positions : List Position) : Prop :=
  sorry

/-- Checks if a game state has a winning condition for the first player -/
def isWinningState (state : GameState) : Prop :=
  sorry

/-- The main theorem stating that the second player can prevent the first player from winning -/
theorem second_player_can_prevent_win :
  ∃ (strategy : Strategy),
    ∀ (game : GameState),
      ¬(isWinningState game) :=
sorry

end NUMINAMATH_CALUDE_second_player_can_prevent_win_l563_56306


namespace NUMINAMATH_CALUDE_money_division_l563_56373

theorem money_division (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a = 80) (h2 : a = (2/3) * (b + c)) (h3 : b = (6/9) * (a + c)) : 
  a + b + c = 200 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l563_56373


namespace NUMINAMATH_CALUDE_largest_lower_bound_area_l563_56309

/-- A point in the Euclidean plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The convex hull of a set of points -/
def convex_hull (points : Set Point) : Set Point := sorry

/-- The area of a set of points -/
def area (s : Set Point) : ℝ := sorry

/-- A convex set of points -/
def is_convex (s : Set Point) : Prop := sorry

theorem largest_lower_bound_area (points : Set Point) :
  ∀ (s : Set Point), (is_convex s ∧ points ⊆ s) →
    area (convex_hull points) ≤ area s :=
by sorry

end NUMINAMATH_CALUDE_largest_lower_bound_area_l563_56309


namespace NUMINAMATH_CALUDE_season_games_count_l563_56394

/-- The number of games in a football season -/
def season_games : ℕ := 16

/-- Archie's record for touchdown passes in a season -/
def archie_record : ℕ := 89

/-- Richard's average touchdowns per game -/
def richard_avg : ℕ := 6

/-- Required average touchdowns in final two games to beat the record -/
def final_avg : ℕ := 3

/-- Number of final games -/
def final_games : ℕ := 2

theorem season_games_count :
  ∃ (x : ℕ), 
    x + final_games = season_games ∧
    richard_avg * x + final_avg * final_games > archie_record :=
by sorry

end NUMINAMATH_CALUDE_season_games_count_l563_56394


namespace NUMINAMATH_CALUDE_dad_steps_count_l563_56351

theorem dad_steps_count (dad_masha_ratio : ℕ → ℕ → Prop)
                        (masha_yasha_ratio : ℕ → ℕ → Prop)
                        (masha_yasha_total : ℕ) :
  dad_masha_ratio 3 5 →
  masha_yasha_ratio 3 5 →
  masha_yasha_total = 400 →
  ∃ (dad_steps : ℕ), dad_steps = 90 := by
  sorry

end NUMINAMATH_CALUDE_dad_steps_count_l563_56351


namespace NUMINAMATH_CALUDE_cos_identity_l563_56358

theorem cos_identity : 
  (2 * (Real.cos (15 * π / 180))^2 - Real.cos (30 * π / 180) = 1) :=
by
  have h : Real.cos (30 * π / 180) = 2 * (Real.cos (15 * π / 180))^2 - 1 := by sorry
  sorry

end NUMINAMATH_CALUDE_cos_identity_l563_56358


namespace NUMINAMATH_CALUDE_trolley_passengers_third_stop_l563_56353

/-- Proves the number of people who got off on the third stop of a trolley ride --/
theorem trolley_passengers_third_stop 
  (initial_passengers : ℕ) 
  (second_stop_off : ℕ) 
  (second_stop_on_multiplier : ℕ) 
  (third_stop_on : ℕ) 
  (final_passengers : ℕ) 
  (h1 : initial_passengers = 10)
  (h2 : second_stop_off = 3)
  (h3 : second_stop_on_multiplier = 2)
  (h4 : third_stop_on = 2)
  (h5 : final_passengers = 12) :
  initial_passengers - second_stop_off + second_stop_on_multiplier * initial_passengers - 
  (initial_passengers - second_stop_off + second_stop_on_multiplier * initial_passengers + third_stop_on - final_passengers) = 17 := by
  sorry


end NUMINAMATH_CALUDE_trolley_passengers_third_stop_l563_56353


namespace NUMINAMATH_CALUDE_equation_solution_l563_56316

theorem equation_solution : ∃ x : ℝ, 4 * x - 2 = 2 * (x + 2) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l563_56316


namespace NUMINAMATH_CALUDE_biker_distance_difference_l563_56311

/-- The difference in distance traveled between two bikers with different speeds over a fixed time -/
theorem biker_distance_difference (alberto_speed bjorn_speed : ℝ) (race_duration : ℝ) 
  (h1 : alberto_speed = 18)
  (h2 : bjorn_speed = 15)
  (h3 : race_duration = 6) :
  alberto_speed * race_duration - bjorn_speed * race_duration = 18 :=
by sorry

end NUMINAMATH_CALUDE_biker_distance_difference_l563_56311


namespace NUMINAMATH_CALUDE_equation_solution_l563_56354

theorem equation_solution : ∃ y : ℚ, (8 + 3.2 * y = 0.8 * y + 40) ∧ (y = 40 / 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l563_56354


namespace NUMINAMATH_CALUDE_system_solution_l563_56383

-- Define the system of equations
def equation1 (x y p : ℝ) : Prop := (x - p)^2 = 16 * (y - 3 + p)
def equation2 (x y : ℝ) : Prop := y^2 + ((x - 3) / (|x| - 3))^2 = 1

-- Define the solution set
def is_solution (x y p : ℝ) : Prop :=
  equation1 x y p ∧ equation2 x y

-- Define the valid range for p
def valid_p (p : ℝ) : Prop :=
  (p > 3 ∧ p ≤ 4) ∨ (p > 12 ∧ p ≠ 19) ∨ (p > 19)

-- Theorem statement
theorem system_solution :
  ∀ p : ℝ, valid_p p →
    ∃ x y : ℝ, is_solution x y p ∧
      x = p + 4 * Real.sqrt (p - 3) ∧
      y = 0 :=
sorry

end NUMINAMATH_CALUDE_system_solution_l563_56383


namespace NUMINAMATH_CALUDE_intersection_theorem_l563_56384

def M : Set ℝ := {x | x^2 - 4 > 0}

def N : Set ℝ := {x | (1 - x) / (x - 3) > 0}

theorem intersection_theorem : N ∩ (Set.univ \ M) = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l563_56384


namespace NUMINAMATH_CALUDE_equation_solution_l563_56337

theorem equation_solution :
  ∃ x : ℚ, (1 / (x + 2) + 3 * x / (x + 2) + 4 / (x + 2) = 1) ∧ (x = -3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l563_56337


namespace NUMINAMATH_CALUDE_relationship_abc_l563_56329

theorem relationship_abc :
  ∀ (a b c : ℝ), a = 2 → b = 3 → c = 4 → c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l563_56329


namespace NUMINAMATH_CALUDE_happy_boys_count_l563_56302

theorem happy_boys_count (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neutral_children : ℕ) (total_boys : ℕ) (total_girls : ℕ) (sad_girls : ℕ) 
  (neutral_boys : ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  total_boys = 16 →
  total_girls = 44 →
  sad_girls = 4 →
  neutral_boys = 4 →
  ∃ (happy_boys : ℕ), happy_boys > 0 →
  happy_boys = 6 :=
by sorry

end NUMINAMATH_CALUDE_happy_boys_count_l563_56302


namespace NUMINAMATH_CALUDE_least_months_to_double_l563_56321

-- Define the initial borrowed amount
def initial_amount : ℝ := 1500

-- Define the monthly interest rate
def monthly_rate : ℝ := 0.03

-- Function to calculate the amount owed after t months
def amount_owed (t : ℕ) : ℝ :=
  initial_amount * (1 + monthly_rate) ^ t

-- Theorem statement
theorem least_months_to_double : ∀ n : ℕ, n < 25 → amount_owed n ≤ 2 * initial_amount ∧
  amount_owed 25 > 2 * initial_amount :=
sorry

end NUMINAMATH_CALUDE_least_months_to_double_l563_56321


namespace NUMINAMATH_CALUDE_closest_whole_number_to_ratio_l563_56380

theorem closest_whole_number_to_ratio : 
  let ratio := (10^4000 + 3*10^4002) / (2*10^4001 + 4*10^4001)
  ∃ (n : ℕ), n = 5 ∧ ∀ (m : ℕ), m ≠ n → |ratio - (n : ℝ)| < |ratio - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_closest_whole_number_to_ratio_l563_56380


namespace NUMINAMATH_CALUDE_relationship_p_q_l563_56320

theorem relationship_p_q (k : ℝ) : ∃ (p₀ q₀ p₁ : ℝ),
  (p₀ * q₀^2 = k) ∧ 
  (p₀ = 16) ∧ 
  (q₀ = 4) ∧ 
  (p₁ * 8^2 = k) → 
  p₁ = 4 := by
sorry

end NUMINAMATH_CALUDE_relationship_p_q_l563_56320


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l563_56389

theorem perpendicular_lines_a_values (a : ℝ) : 
  (∃ (x y : ℝ), ax + 2*y + 6 = 0 ∧ x + a*(a+1)*y + (a^2 - 1) = 0) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    ax₁ + 2*y₁ + 6 = 0 ∧ 
    x₂ + a*(a+1)*y₂ + (a^2 - 1) = 0 →
    (x₂ - x₁) * (ax₁ + 2*y₁) + (y₂ - y₁) * (2*x₁ - 2*a*y₁) = 0) →
  a = 0 ∨ a = -3/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l563_56389


namespace NUMINAMATH_CALUDE_count_ordered_pairs_l563_56346

theorem count_ordered_pairs : ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 128) (Finset.product (Finset.range 129) (Finset.range 129))).card ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_l563_56346


namespace NUMINAMATH_CALUDE_quadratic_sum_l563_56398

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, 5 * x^2 - 30 * x - 45 = a * (x + b)^2 + c) → 
  a + b + c = -88 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l563_56398


namespace NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l563_56324

theorem quadratic_equation_with_given_roots (x y : ℝ) 
  (h : x^2 - 6*x + 9 = -|y - 1|) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
    (∀ z : ℝ, a*z^2 + b*z + c = 0 ↔ z = x ∨ z = y) ∧
    a = 1 ∧ b = -4 ∧ c = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l563_56324


namespace NUMINAMATH_CALUDE_andy_incorrect_answers_l563_56375

/-- Represents the number of incorrect answers for each person -/
structure TestResults where
  andy : ℕ
  beth : ℕ
  charlie : ℕ
  daniel : ℕ

/-- The theorem stating that Andy gets 14 questions wrong given the conditions -/
theorem andy_incorrect_answers (results : TestResults) : results.andy = 14 :=
  by
  have h1 : results.andy + results.beth = results.charlie + results.daniel :=
    sorry
  have h2 : results.andy + results.daniel = results.beth + results.charlie + 6 :=
    sorry
  have h3 : results.charlie = 8 :=
    sorry
  sorry

end NUMINAMATH_CALUDE_andy_incorrect_answers_l563_56375


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l563_56304

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  first : ℚ
  diff : ℚ

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (a : ArithmeticSequence) (n : ℕ) : ℚ :=
  n / 2 * (2 * a.first + (n - 1) * a.diff)

/-- The nth term of an arithmetic sequence -/
def nth_term (a : ArithmeticSequence) (n : ℕ) : ℚ :=
  a.first + (n - 1) * a.diff

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ, sum_n_terms a n / sum_n_terms b n = n / (n + 1)) →
  nth_term a 4 / nth_term b 4 = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l563_56304


namespace NUMINAMATH_CALUDE_tournament_has_24_players_l563_56359

/-- Represents a tournament with the given conditions --/
structure Tournament where
  n : ℕ  -- Total number of players
  pointsAgainstLowest12 : ℕ → ℚ  -- Points each player earned against the lowest 12
  totalPoints : ℕ → ℚ  -- Total points of each player

/-- The conditions of the tournament --/
def tournamentConditions (t : Tournament) : Prop :=
  -- Each player plays against every other player
  ∀ i, t.totalPoints i ≤ (t.n - 1 : ℚ)
  -- Half of each player's points are from the lowest 12
  ∧ ∀ i, 2 * t.pointsAgainstLowest12 i = t.totalPoints i
  -- There are exactly 12 lowest-scoring players
  ∧ ∃ lowest12 : Finset ℕ, lowest12.card = 12 
    ∧ ∀ i ∈ lowest12, ∀ j ∉ lowest12, t.totalPoints i ≤ t.totalPoints j

/-- The theorem stating that the tournament has 24 players --/
theorem tournament_has_24_players (t : Tournament) 
  (h : tournamentConditions t) : t.n = 24 :=
sorry

end NUMINAMATH_CALUDE_tournament_has_24_players_l563_56359


namespace NUMINAMATH_CALUDE_length_of_diagonal_l563_56335

/-- Given a quadrilateral ABCD with specific side lengths, prove the length of AC -/
theorem length_of_diagonal (AB DC AD : ℝ) (h1 : AB = 15) (h2 : DC = 24) (h3 : AD = 7) :
  ∃ AC : ℝ, abs (AC - Real.sqrt (417 + 112 * Real.sqrt 6)) < 0.05 :=
sorry

end NUMINAMATH_CALUDE_length_of_diagonal_l563_56335


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l563_56355

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3*a*x + 9 < 0) ↔ (a < -2 ∨ a > 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l563_56355


namespace NUMINAMATH_CALUDE_delivery_tip_cost_is_eight_l563_56301

/-- Calculates the delivery and tip cost given grocery order details --/
def delivery_and_tip_cost (original_order : ℝ) 
                          (tomatoes_old : ℝ) (tomatoes_new : ℝ)
                          (lettuce_old : ℝ) (lettuce_new : ℝ)
                          (celery_old : ℝ) (celery_new : ℝ)
                          (total_bill : ℝ) : ℝ :=
  let price_increase := (tomatoes_new - tomatoes_old) + 
                        (lettuce_new - lettuce_old) + 
                        (celery_new - celery_old)
  let new_grocery_cost := original_order + price_increase
  total_bill - new_grocery_cost

/-- Theorem stating that the delivery and tip cost is $8.00 --/
theorem delivery_tip_cost_is_eight :
  delivery_and_tip_cost 25 0.99 2.20 1.00 1.75 1.96 2.00 35 = 8 :=
by sorry


end NUMINAMATH_CALUDE_delivery_tip_cost_is_eight_l563_56301


namespace NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l563_56326

theorem rectangle_triangle_equal_area (b h : ℝ) : 
  b > 0 → 
  h > 0 → 
  h ≤ 2 → 
  b * h = (1/2) * b * (1 - h/2) → 
  h = 2/5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l563_56326


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l563_56308

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a partition of a rectangle into four smaller rectangles -/
structure RectanglePartition where
  total : Rectangle
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ

/-- The theorem statement -/
theorem fourth_rectangle_area 
  (partition : RectanglePartition)
  (h1 : partition.total.length = 20)
  (h2 : partition.total.width = 12)
  (h3 : partition.area1 = 24)
  (h4 : partition.area2 = 48)
  (h5 : partition.area3 = 36) :
  partition.total.length * partition.total.width - (partition.area1 + partition.area2 + partition.area3) = 112 :=
by sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_l563_56308


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l563_56345

-- Define the triangle PQR
structure Triangle :=
  (P Q R : Point)
  (altitude : ℝ)
  (base : ℝ)

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : Point)
  (width : ℝ)
  (height : ℝ)

-- Define the problem
def inscribed_rectangle_problem (triangle : Triangle) (rect : Rectangle) : Prop :=
  -- Rectangle ABCD is inscribed in triangle PQR
  -- Side AD of the rectangle is on side PR of the triangle
  -- Triangle's altitude from vertex Q to side PR is 8 inches
  triangle.altitude = 8 ∧
  -- PR = 12 inches
  triangle.base = 12 ∧
  -- Length of AB is equal to a third the length of AD
  rect.width = rect.height / 3 ∧
  -- The area of the rectangle is 64/3 square inches
  rect.width * rect.height = 64 / 3

-- Theorem statement
theorem inscribed_rectangle_area 
  (triangle : Triangle) (rect : Rectangle) :
  inscribed_rectangle_problem triangle rect → 
  rect.width * rect.height = 64 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l563_56345


namespace NUMINAMATH_CALUDE_double_discount_reduction_l563_56312

theorem double_discount_reduction (original_price : ℝ) (discount : ℝ) : 
  discount = 0.4 → 
  (1 - (1 - discount) * (1 - discount)) * 100 = 64 := by
sorry

end NUMINAMATH_CALUDE_double_discount_reduction_l563_56312


namespace NUMINAMATH_CALUDE_fabian_sugar_packs_l563_56374

/-- The number of packs of sugar Fabian wants to buy -/
def sugar_packs : ℕ := 3

/-- The price of apples in dollars per kilogram -/
def apple_price : ℚ := 2

/-- The price of walnuts in dollars per kilogram -/
def walnut_price : ℚ := 6

/-- The price of sugar in dollars per pack -/
def sugar_price : ℚ := apple_price - 1

/-- The amount of apples Fabian wants to buy in kilograms -/
def apple_amount : ℚ := 5

/-- The amount of walnuts Fabian wants to buy in kilograms -/
def walnut_amount : ℚ := 1/2

/-- The total amount Fabian needs to pay in dollars -/
def total_cost : ℚ := 16

theorem fabian_sugar_packs : 
  sugar_packs = (total_cost - apple_price * apple_amount - walnut_price * walnut_amount) / sugar_price := by
  sorry

end NUMINAMATH_CALUDE_fabian_sugar_packs_l563_56374


namespace NUMINAMATH_CALUDE_river_width_l563_56347

/-- Two ferries traveling between opposite banks of a river -/
structure FerrySystem where
  /-- Width of the river -/
  width : ℝ
  /-- Distance from one bank where ferries first meet -/
  first_meeting : ℝ
  /-- Distance from the other bank where ferries second meet -/
  second_meeting : ℝ

/-- Theorem stating the width of the river given the meeting points -/
theorem river_width (fs : FerrySystem) 
    (h1 : fs.first_meeting = 700)
    (h2 : fs.second_meeting = 400) : 
    fs.width = 1700 := by
  sorry

#check river_width

end NUMINAMATH_CALUDE_river_width_l563_56347


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l563_56352

def tripling_time : ℕ := 30  -- seconds
def total_time : ℕ := 300    -- seconds (5 minutes)
def final_count : ℕ := 1239220
def halfway_time : ℕ := 150  -- seconds (2.5 minutes)

def tripling_events (t : ℕ) : ℕ := t / tripling_time

theorem initial_bacteria_count :
  ∃ (n : ℕ),
    n * (3 ^ (tripling_events total_time)) / 2 = final_count ∧
    (n * (3 ^ (tripling_events halfway_time))) / 2 * (3 ^ (tripling_events halfway_time)) = final_count ∧
    n = 42 :=
by sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l563_56352


namespace NUMINAMATH_CALUDE_point_satisfies_conditions_l563_56377

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of the line y = -2x + 3 -/
def on_line (p : Point) : Prop :=
  p.y = -2 * p.x + 3

/-- The condition for a point to be in the first quadrant -/
def in_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The point (1, 1) -/
def point : Point :=
  { x := 1, y := 1 }

theorem point_satisfies_conditions :
  in_first_quadrant point ∧ on_line point :=
by sorry

end NUMINAMATH_CALUDE_point_satisfies_conditions_l563_56377


namespace NUMINAMATH_CALUDE_probability_greater_than_two_l563_56371

/-- A standard die has 6 sides -/
def die_sides : ℕ := 6

/-- The number of outcomes greater than 2 -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a number greater than 2 on a standard six-sided die -/
theorem probability_greater_than_two : 
  (favorable_outcomes : ℚ) / die_sides = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_greater_than_two_l563_56371


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l563_56336

/-- A function f defined piecewise on the real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then 2^x + 1 else -x^2 + a*x + 1

/-- The theorem stating the range of 'a' for which f is increasing on ℝ. -/
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l563_56336


namespace NUMINAMATH_CALUDE_number_divisibility_l563_56313

theorem number_divisibility (N : ℕ) : 
  N % 5 = 0 ∧ N % 4 = 2 → N / 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_divisibility_l563_56313


namespace NUMINAMATH_CALUDE_egg_count_and_weight_l563_56340

/-- Conversion factor from ounces to grams -/
def ouncesToGrams : ℝ := 28.3495

/-- Initial number of eggs -/
def initialEggs : ℕ := 47

/-- Number of whole eggs added -/
def addedEggs : ℕ := 5

/-- Total weight of eggs in ounces -/
def totalWeightOunces : ℝ := 143.5

theorem egg_count_and_weight :
  (initialEggs + addedEggs = 52) ∧
  (abs (totalWeightOunces * ouncesToGrams - 4067.86) < 0.01) :=
by sorry

end NUMINAMATH_CALUDE_egg_count_and_weight_l563_56340
