import Mathlib

namespace work_completion_time_l3304_330494

/-- The number of days A takes to complete the work alone -/
def a_days : ℝ := 10

/-- The number of days B takes to complete the work alone -/
def b_days : ℝ := 20

/-- The number of days A leaves before the work is completed -/
def a_leave_before : ℝ := 5

/-- The total number of days to complete the work -/
def total_days : ℝ := 10

/-- Theorem stating that given the conditions, the work is completed in 10 days -/
theorem work_completion_time :
  (1 / a_days + 1 / b_days) * (total_days - a_leave_before) + (1 / b_days) * a_leave_before = 1 :=
sorry

end work_completion_time_l3304_330494


namespace tennis_ball_distribution_l3304_330433

theorem tennis_ball_distribution (initial_balls : ℕ) (containers : ℕ) : 
  initial_balls = 100 → 
  containers = 5 → 
  (initial_balls / 2) / containers = 10 := by
  sorry

end tennis_ball_distribution_l3304_330433


namespace age_puzzle_l3304_330463

theorem age_puzzle (A : ℕ) (N : ℚ) (h1 : A = 24) (h2 : (A + 3) * N - (A - 3) * N = A) : N = 4 := by
  sorry

end age_puzzle_l3304_330463


namespace zachary_did_more_pushups_l3304_330472

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 51

/-- The number of push-ups David did -/
def david_pushups : ℕ := 44

/-- The difference in push-ups between Zachary and David -/
def pushup_difference : ℕ := zachary_pushups - david_pushups

theorem zachary_did_more_pushups : pushup_difference = 7 := by
  sorry

end zachary_did_more_pushups_l3304_330472


namespace three_digit_integers_with_remainders_l3304_330419

theorem three_digit_integers_with_remainders : 
  let n : ℕ → Prop := λ x => 
    100 ≤ x ∧ x < 1000 ∧ 
    x % 7 = 3 ∧ 
    x % 8 = 4 ∧ 
    x % 10 = 6
  (∃! (l : List ℕ), l.length = 4 ∧ ∀ x ∈ l, n x) := by
  sorry

end three_digit_integers_with_remainders_l3304_330419


namespace parallel_vectors_m_value_l3304_330427

/-- Given two 2D vectors a and b, where a is parallel to b, prove that m = -1 --/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a = (1, 2) →
  b = (2, 3 - m) →
  (∃ (k : ℝ), a = k • b) →
  m = -1 := by
sorry

end parallel_vectors_m_value_l3304_330427


namespace remainder_sum_powers_mod_seven_l3304_330428

theorem remainder_sum_powers_mod_seven :
  (9^7 + 8^8 + 7^9) % 7 = 3 := by
  sorry

end remainder_sum_powers_mod_seven_l3304_330428


namespace eagles_score_is_24_l3304_330415

/-- The combined score of both teams -/
def total_score : ℕ := 56

/-- The margin by which the Falcons won -/
def winning_margin : ℕ := 8

/-- The score of the Eagles -/
def eagles_score : ℕ := total_score / 2 - winning_margin / 2

theorem eagles_score_is_24 : eagles_score = 24 := by
  sorry

end eagles_score_is_24_l3304_330415


namespace inequality_solution_set_l3304_330493

-- Define the inequality
def inequality (k x : ℝ) : Prop :=
  k * (x^2 + 6*x - k) * (x^2 + x - 12) > 0

-- Define the solution set
def solution_set (k : ℝ) : Set ℝ :=
  {x | inequality k x}

-- Theorem statement
theorem inequality_solution_set (k : ℝ) :
  solution_set k = Set.Ioo (-4 : ℝ) 3 ↔ k ∈ Set.Iic (-9 : ℝ) :=
sorry

end inequality_solution_set_l3304_330493


namespace sum_of_repeating_decimals_l3304_330452

def repeating_decimal_3 : ℚ := 1/3
def repeating_decimal_6 : ℚ := 2/3

theorem sum_of_repeating_decimals : 
  repeating_decimal_3 + repeating_decimal_6 = 1 := by sorry

end sum_of_repeating_decimals_l3304_330452


namespace product_equivalence_l3304_330466

theorem product_equivalence : 
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := by
  sorry

end product_equivalence_l3304_330466


namespace equality_addition_l3304_330435

theorem equality_addition (a b : ℝ) : a = b → a + 3 = 3 + b := by
  sorry

end equality_addition_l3304_330435


namespace factory_weekly_production_l3304_330477

/-- Represents a toy factory with its production characteristics -/
structure ToyFactory where
  daysPerWeek : ℕ
  dailyProduction : ℕ
  constDailyProduction : Bool

/-- Calculates the weekly production of toys for a given factory -/
def weeklyProduction (factory : ToyFactory) : ℕ :=
  factory.daysPerWeek * factory.dailyProduction

/-- Theorem: The weekly production of the given factory is 6500 toys -/
theorem factory_weekly_production :
  ∀ (factory : ToyFactory),
    factory.daysPerWeek = 5 →
    factory.dailyProduction = 1300 →
    factory.constDailyProduction = true →
    weeklyProduction factory = 6500 := by
  sorry

end factory_weekly_production_l3304_330477


namespace profit_maximum_l3304_330432

/-- The bank's profit function -/
def profit_function (k : ℝ) (x : ℝ) : ℝ := 0.045 * k * x^2 - k * x^3

/-- Theorem stating that the profit function reaches its maximum at x = 0.03 -/
theorem profit_maximum (k : ℝ) (h : k > 0) :
  ∃ (max : ℝ), ∀ (x : ℝ), x > 0 → profit_function k x ≤ profit_function k 0.03 :=
sorry

end profit_maximum_l3304_330432


namespace t_range_theorem_l3304_330449

theorem t_range_theorem (t x y z : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
  3 * x^2 + 3 * z * x + z^2 = 1 →
  3 * y^2 + 3 * y * z + z^2 = 4 →
  x^2 - x * y + y^2 = t →
  (3 - Real.sqrt 5) / 2 ≤ t ∧ t ≤ 1 := by
sorry

end t_range_theorem_l3304_330449


namespace special_sequence_remainder_l3304_330474

def sequence_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n < 69 → 3 * a n = a (n - 1) + a (n + 1)

theorem special_sequence_remainder :
  ∀ a : ℕ → ℤ,
  sequence_condition a →
  a 0 = 0 →
  a 1 = 1 →
  a 2 = 3 →
  a 3 = 8 →
  a 4 = 21 →
  ∃ k : ℤ, a 69 = 6 * k + 4 :=
by sorry

end special_sequence_remainder_l3304_330474


namespace point_M_coordinates_l3304_330469

-- Define the points A, B, C, and M
def A : ℝ × ℝ := (2, -4)
def B : ℝ × ℝ := (-1, 3)
def C : ℝ × ℝ := (3, 4)
def M : ℝ × ℝ := (-11, -15)

-- Define vectors
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

-- State the theorem
theorem point_M_coordinates :
  vec C M = (2 : ℝ) • (vec C A) + (3 : ℝ) • (vec C B) → M = (-11, -15) := by
  sorry


end point_M_coordinates_l3304_330469


namespace complex_magnitude_l3304_330480

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l3304_330480


namespace greatest_of_three_consecutive_integers_sum_21_l3304_330490

theorem greatest_of_three_consecutive_integers_sum_21 :
  ∀ x y z : ℤ, 
    (y = x + 1) → 
    (z = y + 1) → 
    (x + y + z = 21) → 
    (max x (max y z) = 8) :=
by
  sorry

end greatest_of_three_consecutive_integers_sum_21_l3304_330490


namespace factorization_equality_l3304_330443

theorem factorization_equality (a b c x y : ℝ) :
  -a * (x - y) - b * (y - x) + c * (x - y) = (y - x) * (a - b - c) := by
  sorry

end factorization_equality_l3304_330443


namespace number_with_one_third_equal_to_twelve_l3304_330410

theorem number_with_one_third_equal_to_twelve (x : ℝ) : (1 / 3 : ℝ) * x = 12 → x = 36 := by
  sorry

end number_with_one_third_equal_to_twelve_l3304_330410


namespace initial_mean_calculation_l3304_330439

theorem initial_mean_calculation (n : ℕ) (initial_wrong : ℝ) (corrected : ℝ) (new_mean : ℝ) :
  n = 50 →
  initial_wrong = 23 →
  corrected = 48 →
  new_mean = 30.5 →
  (n : ℝ) * new_mean = (n : ℝ) * (n * new_mean - (corrected - initial_wrong)) / n :=
by sorry

end initial_mean_calculation_l3304_330439


namespace city_g_highest_growth_l3304_330438

structure City where
  name : String
  pop1990 : ℕ
  pop2000 : ℕ

def cities : List City := [
  ⟨"F", 50, 60⟩,
  ⟨"G", 60, 90⟩,
  ⟨"H", 70, 80⟩,
  ⟨"I", 100, 110⟩,
  ⟨"J", 150, 180⟩
]

def growthRate (c : City) : ℚ :=
  (c.pop2000 : ℚ) / (c.pop1990 : ℚ)

def adjustedGrowthRate (c : City) : ℚ :=
  if c.name = "H" then
    growthRate c * (11 / 10)
  else
    growthRate c

theorem city_g_highest_growth :
  ∀ c ∈ cities, c.name ≠ "G" →
    adjustedGrowthRate (cities[1]) ≥ adjustedGrowthRate c := by
  sorry

end city_g_highest_growth_l3304_330438


namespace root_in_interval_l3304_330406

-- Define the function f(x) = x³ - x - 3
def f (x : ℝ) : ℝ := x^3 - x - 3

-- State the theorem
theorem root_in_interval :
  Continuous f ∧ f 1 < 0 ∧ 0 < f 2 →
  ∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ f x = 0 :=
by sorry

end root_in_interval_l3304_330406


namespace max_colored_pages_l3304_330411

/-- The cost in cents to print a colored page -/
def cost_per_page : ℕ := 4

/-- The budget in dollars -/
def budget : ℕ := 30

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The maximum number of colored pages that can be printed -/
def max_pages : ℕ := (budget * cents_per_dollar) / cost_per_page

theorem max_colored_pages : max_pages = 750 := by
  sorry

end max_colored_pages_l3304_330411


namespace grayson_speed_l3304_330412

/-- Grayson's motorboat trip -/
structure GraysonTrip where
  speed1 : ℝ  -- Speed during the first hour
  time1 : ℝ   -- Time of the first part (1 hour)
  speed2 : ℝ  -- Speed during the second part (20 mph)
  time2 : ℝ   -- Time of the second part (0.5 hours)

/-- Rudy's rowboat trip -/
structure RudyTrip where
  speed : ℝ   -- Rudy's speed (10 mph)
  time : ℝ    -- Rudy's travel time (3 hours)

/-- The main theorem -/
theorem grayson_speed (g : GraysonTrip) (r : RudyTrip) 
  (h1 : g.time1 = 1)
  (h2 : g.time2 = 0.5)
  (h3 : g.speed2 = 20)
  (h4 : r.speed = 10)
  (h5 : r.time = 3)
  (h6 : g.speed1 * g.time1 + g.speed2 * g.time2 = r.speed * r.time + 5) :
  g.speed1 = 25 := by
  sorry

end grayson_speed_l3304_330412


namespace birthday_gifts_l3304_330416

theorem birthday_gifts (gifts_12th : ℕ) (fewer_gifts : ℕ) : 
  gifts_12th = 20 → fewer_gifts = 8 → 
  gifts_12th + (gifts_12th - fewer_gifts) = 32 := by
  sorry

end birthday_gifts_l3304_330416


namespace triangle_circumscribed_circle_diameter_l3304_330473

theorem triangle_circumscribed_circle_diameter 
  (a : ℝ) (A : ℝ) (D : ℝ) :
  a = 10 ∧ A = π/4 ∧ D = a / Real.sin A → D = 10 * Real.sqrt 2 :=
by sorry

end triangle_circumscribed_circle_diameter_l3304_330473


namespace river_width_is_8km_l3304_330420

/-- Represents the boat's journey across the river -/
structure RiverCrossing where
  boat_speed : ℝ
  current_speed : ℝ
  crossing_time : ℝ

/-- Calculates the width of the river based on the given conditions -/
def river_width (rc : RiverCrossing) : ℝ :=
  rc.boat_speed * rc.crossing_time

/-- Theorem stating that the width of the river is 8 km under the given conditions -/
theorem river_width_is_8km (rc : RiverCrossing) 
  (h1 : rc.boat_speed = 4)
  (h2 : rc.current_speed = 3)
  (h3 : rc.crossing_time = 2) : 
  river_width rc = 8 := by
  sorry

end river_width_is_8km_l3304_330420


namespace complex_equation_solution_l3304_330425

theorem complex_equation_solution (x y : ℝ) : 
  (Complex.mk (2 * x - 1) 1 = Complex.mk y (y - 2)) → x = 2 ∧ y = 3 := by
  sorry

end complex_equation_solution_l3304_330425


namespace system_solutions_l3304_330487

/-- The first equation of the system -/
def equation1 (x y z : ℝ) : Prop :=
  5 * x^2 + 3 * y^2 + 3 * x * y + 2 * x * z - y * z - 10 * y + 5 = 0

/-- The second equation of the system -/
def equation2 (x y z : ℝ) : Prop :=
  49 * x^2 + 65 * y^2 + 49 * z^2 - 14 * x * y - 98 * x * z + 14 * y * z - 182 * x - 102 * y + 182 * z + 233 = 0

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  equation1 x y z ∧ equation2 x y z

/-- The theorem stating that the given points are the only solutions to the system -/
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = 2/7 ∧ y = 1 ∧ z = -12/7) := by
  sorry

end system_solutions_l3304_330487


namespace additional_flies_needed_l3304_330475

/-- Represents the number of flies eaten by the frog each day of the week -/
def flies_eaten_per_day : List Nat := [3, 2, 4, 5, 1, 2, 3]

/-- Calculates the total number of flies eaten in a week -/
def total_flies_needed : Nat := flies_eaten_per_day.sum

/-- Number of flies Betty caught in the morning -/
def morning_catch : Nat := 5

/-- Number of flies Betty caught in the afternoon -/
def afternoon_catch : Nat := 6

/-- Number of flies that escaped -/
def escaped_flies : Nat := 1

/-- Calculates the total number of flies Betty successfully caught -/
def total_flies_caught : Nat := morning_catch + afternoon_catch - escaped_flies

/-- Theorem stating the number of additional flies Betty needs -/
theorem additional_flies_needed : 
  total_flies_needed - total_flies_caught = 10 := by sorry

end additional_flies_needed_l3304_330475


namespace planes_with_three_common_points_l3304_330404

-- Define a plane in 3D space
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define a point in 3D space
def Point : Type := ℝ × ℝ × ℝ

-- Define what it means for a point to be on a plane
def on_plane (p : Point) (plane : Plane) : Prop :=
  let (x, y, z) := p
  plane x y z

-- Define what it means for two planes to intersect
def intersect (p1 p2 : Plane) : Prop :=
  ∃ (p : Point), on_plane p p1 ∧ on_plane p p2

-- Define what it means for two planes to coincide
def coincide (p1 p2 : Plane) : Prop :=
  ∀ (p : Point), on_plane p p1 ↔ on_plane p p2

-- Theorem statement
theorem planes_with_three_common_points 
  (p1 p2 : Plane) (a b c : Point)
  (h1 : on_plane a p1 ∧ on_plane a p2)
  (h2 : on_plane b p1 ∧ on_plane b p2)
  (h3 : on_plane c p1 ∧ on_plane c p2) :
  intersect p1 p2 ∨ coincide p1 p2 :=
sorry

end planes_with_three_common_points_l3304_330404


namespace unique_solution_condition_l3304_330459

theorem unique_solution_condition (s : ℝ) : 
  (∃! x : ℝ, (s * x - 3) / (x + 1) = x) ↔ (s = 1 + 2 * Real.sqrt 3 ∨ s = 1 - 2 * Real.sqrt 3) :=
by sorry

end unique_solution_condition_l3304_330459


namespace two_true_propositions_l3304_330448

theorem two_true_propositions :
  let prop1 := ∀ a : ℝ, a > -1 → a > -2
  let prop2 := ∀ a : ℝ, a > -2 → a > -1
  let prop3 := ∀ a : ℝ, a ≤ -1 → a ≤ -2
  let prop4 := ∀ a : ℝ, a ≤ -2 → a ≤ -1
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) :=
by
  sorry

end two_true_propositions_l3304_330448


namespace solution_set_for_a_equals_two_range_for_empty_solution_set_l3304_330429

def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |x - a|

theorem solution_set_for_a_equals_two :
  {x : ℝ | f 2 x > 2} = {x : ℝ | x > 3/2} := by sorry

theorem range_for_empty_solution_set :
  {a : ℝ | a > 0 ∧ ∀ x, f a x < 2*a} = {a : ℝ | a > 1} := by sorry

end solution_set_for_a_equals_two_range_for_empty_solution_set_l3304_330429


namespace points_on_line_equidistant_l3304_330401

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the line 4x + 7y = 28 -/
def onLine (p : Point) : Prop :=
  4 * p.x + 7 * p.y = 28

/-- Defines the condition of being equidistant from coordinate axes -/
def equidistant (p : Point) : Prop :=
  |p.x| = |p.y|

/-- Defines the condition of being in quadrant I -/
def inQuadrantI (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Defines the condition of being in quadrant II -/
def inQuadrantII (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Defines the condition of being in quadrant III -/
def inQuadrantIII (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Defines the condition of being in quadrant IV -/
def inQuadrantIV (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

theorem points_on_line_equidistant :
  ∀ p : Point, onLine p ∧ equidistant p →
    (inQuadrantI p ∨ inQuadrantII p) ∧
    ¬(inQuadrantIII p ∨ inQuadrantIV p) :=
by sorry

end points_on_line_equidistant_l3304_330401


namespace fraction_sum_equality_l3304_330464

theorem fraction_sum_equality : (20 : ℚ) / 24 + (20 : ℚ) / 25 = 49 / 30 := by
  sorry

end fraction_sum_equality_l3304_330464


namespace third_grade_students_l3304_330434

theorem third_grade_students (total_students : ℕ) (sample_size : ℕ) (first_grade_sample : ℕ) (second_grade_sample : ℕ) :
  total_students = 2000 →
  sample_size = 100 →
  first_grade_sample = 30 →
  second_grade_sample = 30 →
  (total_students * (sample_size - first_grade_sample - second_grade_sample)) / sample_size = 800 := by
  sorry

end third_grade_students_l3304_330434


namespace fraction_multiplication_division_main_proof_l3304_330446

theorem fraction_multiplication_division (a b c d e f : ℚ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 → e ≠ 0 → f ≠ 0 →
  (a / b * c / d) / (e / f) = (a * c * f) / (b * d * e) :=
by sorry

theorem main_proof : (3 / 4 * 5 / 6) / (7 / 8) = 5 / 7 :=
by sorry

end fraction_multiplication_division_main_proof_l3304_330446


namespace triangle_abc_properties_l3304_330470

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where √3b = 2c sin B, c = √7, and a + b = 5, prove that:
    1. The angle C is equal to π/3
    2. The area of triangle ABC is (3√3)/2 -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π/2 →
  0 < B → B < π/2 →
  0 < C → C < π/2 →
  Real.sqrt 3 * b = 2 * c * Real.sin B →
  c = Real.sqrt 7 →
  a + b = 5 →
  C = π/3 ∧ (1/2 * a * b * Real.sin C = (3 * Real.sqrt 3) / 2) := by
  sorry

end triangle_abc_properties_l3304_330470


namespace arithmetic_sequence_sum_property_arithmetic_sequence_sum_l3304_330460

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of terms with indices that add up to the same value is constant -/
theorem arithmetic_sequence_sum_property (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  ∀ i j k l : ℕ, i + l = j + k → a i + a l = a j + a k :=
sorry

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h_sum : a 3 + a 7 = 37) : a 2 + a 4 + a 6 + a 8 = 74 :=
sorry

end arithmetic_sequence_sum_property_arithmetic_sequence_sum_l3304_330460


namespace x_minus_y_value_l3304_330497

theorem x_minus_y_value (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 17) 
  (eq2 : x + 3 * y = 1) : 
  x - y = 69 / 13 := by
sorry

end x_minus_y_value_l3304_330497


namespace sum_integers_negative20_to_10_l3304_330496

def sum_integers (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_negative20_to_10 :
  sum_integers (-20) 10 = -155 := by
  sorry

end sum_integers_negative20_to_10_l3304_330496


namespace robot_gather_time_l3304_330442

/-- The time (in minutes) it takes a robot to create a battery -/
def create_time : ℕ := 9

/-- The number of robots working simultaneously -/
def num_robots : ℕ := 10

/-- The number of batteries manufactured in 5 hours -/
def batteries_produced : ℕ := 200

/-- The time (in hours) taken to manufacture the batteries -/
def production_time : ℕ := 5

/-- The time (in minutes) it takes a robot to gather materials for a battery -/
def gather_time : ℕ := 6

theorem robot_gather_time :
  gather_time = 6 ∧
  create_time = 9 ∧
  num_robots = 10 ∧
  batteries_produced = 200 ∧
  production_time = 5 →
  num_robots * batteries_produced * (gather_time + create_time) = production_time * 60 :=
by sorry

end robot_gather_time_l3304_330442


namespace sum_of_digits_3_plus_4_pow_17_l3304_330479

/-- The sum of the tens digit and the ones digit of (3+4)^17 in integer form is 7 -/
theorem sum_of_digits_3_plus_4_pow_17 : 
  (((3 + 4)^17 / 10) % 10 + (3 + 4)^17 % 10) = 7 := by sorry

end sum_of_digits_3_plus_4_pow_17_l3304_330479


namespace total_cakes_served_l3304_330458

/-- The number of cakes served on Sunday -/
def sunday_cakes : ℕ := 3

/-- The number of cakes served during lunch on Monday -/
def monday_lunch_cakes : ℕ := 5

/-- The number of cakes served during dinner on Monday -/
def monday_dinner_cakes : ℕ := 6

/-- The number of cakes thrown away on Tuesday -/
def tuesday_thrown_cakes : ℕ := 4

/-- The total number of cakes served on Monday -/
def monday_total_cakes : ℕ := monday_lunch_cakes + monday_dinner_cakes

/-- The number of cakes initially prepared for Tuesday (before throwing away) -/
def tuesday_initial_cakes : ℕ := 2 * monday_total_cakes

/-- The total number of cakes served on Tuesday after throwing away some -/
def tuesday_final_cakes : ℕ := tuesday_initial_cakes - tuesday_thrown_cakes

/-- Theorem stating that the total number of cakes served over three days is 32 -/
theorem total_cakes_served : sunday_cakes + monday_total_cakes + tuesday_final_cakes = 32 := by
  sorry

end total_cakes_served_l3304_330458


namespace largest_multiple_of_seven_below_negative_85_l3304_330440

theorem largest_multiple_of_seven_below_negative_85 :
  ∀ n : ℤ, n % 7 = 0 ∧ n < -85 → n ≤ -91 :=
by
  sorry

end largest_multiple_of_seven_below_negative_85_l3304_330440


namespace right_triangle_angle_b_l3304_330445

/-- Given a right triangle ABC with ∠A = 70°, prove that ∠B = 20° -/
theorem right_triangle_angle_b (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  C = 90 →           -- One angle is 90° (right angle)
  A = 70 →           -- Given ∠A = 70°
  B = 20 :=          -- To prove: ∠B = 20°
by sorry

end right_triangle_angle_b_l3304_330445


namespace line_intersects_ellipse_once_l3304_330436

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define a point P on the ellipse
def point_on_ellipse (x₀ y₀ : ℝ) : Prop := ellipse x₀ y₀

-- Define the line l
def line (x₀ y₀ x y : ℝ) : Prop := 3 * x₀ * x + 4 * y₀ * y - 12 = 0

-- Theorem statement
theorem line_intersects_ellipse_once (x₀ y₀ : ℝ) 
  (h_point : point_on_ellipse x₀ y₀) :
  ∃! (x y : ℝ), ellipse x y ∧ line x₀ y₀ x y :=
sorry

end line_intersects_ellipse_once_l3304_330436


namespace min_value_H_negative_reals_l3304_330492

-- Define the concept of an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the function H
def H (a b : ℝ) (f g : ℝ → ℝ) (x : ℝ) : ℝ := a * f x + b * g x + 1

-- State the theorem
theorem min_value_H_negative_reals 
  (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : OddFunction f) (hg : OddFunction g)
  (hmax : ∃ M, M = 5 ∧ ∀ x > 0, H a b f g x ≤ M) :
  ∃ m, m = -3 ∧ ∀ x < 0, H a b f g x ≥ m :=
sorry

end min_value_H_negative_reals_l3304_330492


namespace particle_diameter_scientific_notation_l3304_330422

/-- Converts a decimal number to scientific notation -/
def to_scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem particle_diameter_scientific_notation :
  to_scientific_notation 0.00000021 = (2.1, -7) :=
sorry

end particle_diameter_scientific_notation_l3304_330422


namespace unique_solution_floor_equation_l3304_330453

theorem unique_solution_floor_equation :
  ∃! (x : ℝ), x > 0 ∧ x * ↑(⌊x⌋) = 72 ∧ x = 9 := by
  sorry

end unique_solution_floor_equation_l3304_330453


namespace area_triangle_PTU_l3304_330402

/-- Regular octagon with side length 3 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Triangle formed by vertices P, T, and U in the regular octagon -/
def triangle_PTU (octagon : RegularOctagon) : Set (Fin 3 → ℝ × ℝ) := sorry

/-- Area of a triangle -/
def triangle_area (t : Set (Fin 3 → ℝ × ℝ)) : ℝ := sorry

/-- Main theorem: Area of triangle PTU in a regular octagon with side length 3 -/
theorem area_triangle_PTU (octagon : RegularOctagon) :
  triangle_area (triangle_PTU octagon) = (9 * Real.sqrt 2 + 9) / 2 := by sorry

end area_triangle_PTU_l3304_330402


namespace system_solution_l3304_330499

theorem system_solution : 
  ∃ (x y z : ℝ), 
    (x + y + z = 15 ∧ 
     x^2 + y^2 + z^2 = 81 ∧ 
     x*y + x*z = 3*y*z) ∧
    ((x = 6 ∧ y = 3 ∧ z = 6) ∨ 
     (x = 6 ∧ y = 6 ∧ z = 3)) := by
  sorry

#check system_solution

end system_solution_l3304_330499


namespace olympic_high_school_contest_l3304_330441

theorem olympic_high_school_contest (f s : ℕ) : 
  f > 0 → s > 0 → (2 * f) / 5 = (4 * s) / 5 → f = 2 * s := by
  sorry

#check olympic_high_school_contest

end olympic_high_school_contest_l3304_330441


namespace oplus_calculation_l3304_330455

def oplus (x y : ℚ) : ℚ := 1 / (x - y) + y

theorem oplus_calculation :
  (oplus 2 (-3) = -2 - 4/5) ∧
  (oplus (oplus (-4) (-1)) (-5) = -4 - 8/11) := by
  sorry

end oplus_calculation_l3304_330455


namespace pants_price_problem_l3304_330495

theorem pants_price_problem (total_cost shirt_price pants_price shoes_price : ℚ) : 
  total_cost = 340 →
  shirt_price = (3/4) * pants_price →
  shoes_price = pants_price + 10 →
  total_cost = shirt_price + pants_price + shoes_price →
  pants_price = 120 := by
sorry

end pants_price_problem_l3304_330495


namespace clouddale_rainfall_2008_l3304_330483

def average_monthly_rainfall_2007 : ℝ := 45.2
def rainfall_increase_2008 : ℝ := 3.5
def months_in_year : ℕ := 12

theorem clouddale_rainfall_2008 :
  let average_monthly_rainfall_2008 := average_monthly_rainfall_2007 + rainfall_increase_2008
  let total_rainfall_2008 := average_monthly_rainfall_2008 * months_in_year
  total_rainfall_2008 = 584.4 := by
sorry

end clouddale_rainfall_2008_l3304_330483


namespace only_vehicle_green_light_is_random_l3304_330484

-- Define the type for events
inductive Event
  | TriangleInequality
  | SunRise
  | VehicleGreenLight
  | NegativeAbsoluteValue

-- Define a predicate for random events
def isRandomEvent : Event → Prop :=
  fun e => match e with
    | Event.TriangleInequality => false
    | Event.SunRise => false
    | Event.VehicleGreenLight => true
    | Event.NegativeAbsoluteValue => false

-- Theorem statement
theorem only_vehicle_green_light_is_random :
  ∀ e : Event, isRandomEvent e ↔ e = Event.VehicleGreenLight :=
by sorry

end only_vehicle_green_light_is_random_l3304_330484


namespace celebrity_baby_photo_match_probability_l3304_330471

/-- The number of celebrities and baby photos -/
def n : ℕ := 4

/-- The probability of correctly matching all celebrities with their baby photos -/
def correct_match_probability : ℚ := 1 / (n.factorial : ℚ)

/-- Theorem stating that the probability of correctly matching all celebrities
    with their baby photos when guessing at random is 1/24 -/
theorem celebrity_baby_photo_match_probability :
  correct_match_probability = 1 / 24 := by
  sorry

end celebrity_baby_photo_match_probability_l3304_330471


namespace max_value_theorem_l3304_330418

/-- The maximum value of ab/(a+b) + ac/(a+c) + bc/(b+c) given the conditions -/
theorem max_value_theorem (a b c : ℝ) 
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
  (h_sum : a + b + c = 3)
  (h_product : a * b * c = 1) :
  (a * b / (a + b) + a * c / (a + c) + b * c / (b + c)) ≤ 3 / 2 ∧ 
  ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧ a * b * c = 1 ∧
    a * b / (a + b) + a * c / (a + c) + b * c / (b + c) = 3 / 2 :=
by sorry

end max_value_theorem_l3304_330418


namespace x_one_value_l3304_330408

theorem x_one_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h_eq : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + (x₃ - x₄)^2 + x₄^2 = 1/5) :
  x₁ = 4/5 := by
  sorry

end x_one_value_l3304_330408


namespace tangent_surface_area_l3304_330421

/-- Given a sphere of radius R and a point S at distance 2R from the center,
    the surface area formed by tangent lines from S to the sphere is 3πR^2/2 -/
theorem tangent_surface_area (R : ℝ) (h : R > 0) :
  let sphere_radius := R
  let point_distance := 2 * R
  let surface_area := (3 / 2) * π * R^2
  surface_area = (3 / 2) * π * sphere_radius^2 :=
by sorry

end tangent_surface_area_l3304_330421


namespace ratio_equality_l3304_330403

theorem ratio_equality (x y a b : ℝ) 
  (h1 : x / y = 3)
  (h2 : (2 * a - x) / (3 * b - y) = 3) :
  a / b = 9 / 2 := by
sorry

end ratio_equality_l3304_330403


namespace negation_equivalence_l3304_330424

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2*x > 2) ↔ (∀ x : ℝ, x^2 - 2*x ≤ 2) := by
  sorry

end negation_equivalence_l3304_330424


namespace not_square_or_cube_l3304_330488

theorem not_square_or_cube (n : ℕ+) :
  ¬ ∃ (k m : ℤ), (n * (n + 1) * (n + 2) * (n + 3) : ℤ) = k^2 ∨
                 (n * (n + 1) * (n + 2) * (n + 3) : ℤ) = m^3 :=
by sorry

end not_square_or_cube_l3304_330488


namespace jindra_dice_count_l3304_330450

/-- Represents the number of dice in half a layer -/
def half_layer : ℕ := 18

/-- Represents the number of complete layers -/
def complete_layers : ℕ := 6

/-- Theorem stating the total number of dice Jindra had yesterday -/
theorem jindra_dice_count : 
  (2 * half_layer * complete_layers) + half_layer = 234 := by
  sorry

end jindra_dice_count_l3304_330450


namespace sphere_volume_radius_3_l3304_330498

/-- The volume of a sphere with radius 3 cm is 36π cm³. -/
theorem sphere_volume_radius_3 :
  let r : ℝ := 3
  let volume := (4 / 3) * Real.pi * r ^ 3
  volume = 36 * Real.pi :=
by sorry

end sphere_volume_radius_3_l3304_330498


namespace both_pipes_open_time_l3304_330456

/-- The time it takes for pipe p to fill the cistern alone -/
def p_time : ℚ := 12

/-- The time it takes for pipe q to fill the cistern alone -/
def q_time : ℚ := 15

/-- The additional time it takes for pipe q to fill the cistern after pipe p is turned off -/
def additional_time : ℚ := 6

/-- The theorem stating that the time both pipes are open together is 4 minutes -/
theorem both_pipes_open_time : 
  ∃ (t : ℚ), 
    t * (1 / p_time + 1 / q_time) + additional_time * (1 / q_time) = 1 ∧ 
    t = 4 := by
  sorry

end both_pipes_open_time_l3304_330456


namespace max_value_inequality_l3304_330426

theorem max_value_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 6 + 8 * y * z ≤ Real.sqrt 22 ∧
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^2 + y^2 + z^2 = 1 ∧
    2 * x * y * Real.sqrt 6 + 8 * y * z = Real.sqrt 22 := by
  sorry

end max_value_inequality_l3304_330426


namespace asymptote_sum_l3304_330430

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x^3 + A*x^2 + B*x + C = (x + 1)*(x - 3)*(x - 4)) → 
  A + B + C = 11 := by
  sorry

end asymptote_sum_l3304_330430


namespace m_range_l3304_330405

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the parabola
def on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 2 * P.1

-- Define the distance ratio condition
def satisfies_distance_ratio (P : ℝ × ℝ) (m : ℝ) : Prop :=
  (P.1 + 1)^2 + P.2^2 = m^2 * ((P.1 - 1)^2 + P.2^2)

-- Main theorem
theorem m_range (P : ℝ × ℝ) (m : ℝ) 
  (h1 : on_parabola P) 
  (h2 : satisfies_distance_ratio P m) : 
  1 ≤ m ∧ m ≤ Real.sqrt 3 :=
sorry

end m_range_l3304_330405


namespace regular_16gon_symmetry_sum_l3304_330468

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add necessary fields here

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := sorry

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def smallestRotationAngle (p : RegularPolygon n) : ℝ := sorry

theorem regular_16gon_symmetry_sum :
  ∀ (p : RegularPolygon 16),
    (linesOfSymmetry p : ℝ) + smallestRotationAngle p = 38.5 := by
  sorry

end regular_16gon_symmetry_sum_l3304_330468


namespace system_no_solution_implies_m_equals_two_l3304_330481

/-- Represents a 2x3 augmented matrix -/
structure AugmentedMatrix (α : Type*) :=
  (a11 a12 a13 a21 a22 a23 : α)

/-- Checks if the given augmented matrix represents a system with no real solution -/
def has_no_real_solution (A : AugmentedMatrix ℝ) : Prop :=
  ∀ x y : ℝ, A.a11 * x + A.a12 * y ≠ A.a13 ∨ A.a21 * x + A.a22 * y ≠ A.a23

theorem system_no_solution_implies_m_equals_two :
  ∀ m : ℝ, 
    let A : AugmentedMatrix ℝ := ⟨m, 4, m + 2, 1, m, m⟩
    has_no_real_solution A → m = 2 :=
by
  sorry

end system_no_solution_implies_m_equals_two_l3304_330481


namespace polynomial_rewrite_l3304_330437

variable (x y : ℝ)

def original_polynomial := x^3 - 3*x^2*y + 3*x*y^2 - y^3

theorem polynomial_rewrite :
  ((x^3 - y^3) - (3*x^2*y - 3*x*y^2) = original_polynomial x y) ∧
  ((x^3 + 3*x*y^2) - (3*x^2*y + y^3) = original_polynomial x y) ∧
  ((3*x*y^2 - 3*x^2*y) - (y^3 - x^3) = original_polynomial x y) ∧
  ¬((x^3 - 3*x^2*y) - (3*x*y^2 + y^3) = original_polynomial x y) :=
by sorry

end polynomial_rewrite_l3304_330437


namespace water_needed_in_quarts_l3304_330400

/-- Represents the ratio of water to lemon juice in the lemonade mixture -/
def water_to_lemon_ratio : ℚ := 4

/-- Represents the total number of parts in the mixture -/
def total_parts : ℚ := water_to_lemon_ratio + 1

/-- Represents the total volume of the mixture in gallons -/
def total_volume : ℚ := 1

/-- Represents the number of quarts in a gallon -/
def quarts_per_gallon : ℚ := 4

/-- Theorem stating the amount of water needed in quarts -/
theorem water_needed_in_quarts : 
  (water_to_lemon_ratio / total_parts) * total_volume * quarts_per_gallon = 16/5 := by
  sorry

end water_needed_in_quarts_l3304_330400


namespace problem_solution_l3304_330491

def f (m : ℝ) (x : ℝ) : ℝ := |x + 1| + |m - x|

theorem problem_solution :
  (∀ x : ℝ, f 3 x ≥ 6 ↔ (x ≤ -2 ∨ x ≥ 4)) ∧
  (∀ m : ℝ, (∀ x : ℝ, f m x ≥ 8) ↔ (m ≤ -9 ∨ m ≥ 7)) :=
sorry

end problem_solution_l3304_330491


namespace range_of_fraction_l3304_330447

theorem range_of_fraction (x y : ℝ) (h : x^2 + y^2 + 2*x = 0) :
  -1 ≤ (y - x) / (x - 1) ∧ (y - x) / (x - 1) ≤ 1/3 :=
sorry

end range_of_fraction_l3304_330447


namespace equation_solution_range_l3304_330417

theorem equation_solution_range (m : ℝ) :
  (∃ x : ℝ, 1 - 2 * Real.sin x ^ 2 + 2 * Real.cos x - m = 0) ↔ -3/2 ≤ m ∧ m ≤ 3 := by
sorry

end equation_solution_range_l3304_330417


namespace triangle_area_triangle_area_is_32_l3304_330409

/-- The area of the right triangle formed by the lines y = x, x = -8, and the x-axis -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    let line1 : ℝ → ℝ → Prop := fun x y => y = x
    let line2 : ℝ → Prop := fun x => x = -8
    let x_axis : ℝ → Prop := fun y => y = 0
    let intersection_point : ℝ × ℝ := (-8, -8)
    let base : ℝ := 8
    let height : ℝ := 8
    (∀ x y, line1 x y → line2 x → (x, y) = intersection_point) ∧
    (∀ x, line2 x → x_axis 0) ∧
    (area = (1/2) * base * height) →
    area = 32

theorem triangle_area_is_32 : triangle_area 32 := by sorry

end triangle_area_triangle_area_is_32_l3304_330409


namespace tangent_ellipse_hyperbola_l3304_330454

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 4

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y - 2)^2 = 4

-- Define the tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse x y ∧ hyperbola x y m ∧
  ∀ (x' y' : ℝ), ellipse x' y' ∧ hyperbola x' y' m → (x = x' ∧ y = y')

-- Theorem statement
theorem tangent_ellipse_hyperbola :
  ∀ m : ℝ, are_tangent m → m = 1/3 := by sorry

end tangent_ellipse_hyperbola_l3304_330454


namespace horner_method_eval_l3304_330423

def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

theorem horner_method_eval :
  f (-4) = 220 := by
  sorry

end horner_method_eval_l3304_330423


namespace cos_2x_plus_2y_l3304_330461

theorem cos_2x_plus_2y (x y : ℝ) (h : Real.cos x * Real.cos y - Real.sin x * Real.sin y = 1/4) :
  Real.cos (2*x + 2*y) = -7/8 := by
  sorry

end cos_2x_plus_2y_l3304_330461


namespace number_problem_l3304_330478

theorem number_problem (x : ℝ) : 0.3 * x = 0.6 * 150 + 120 → x = 700 := by
  sorry

end number_problem_l3304_330478


namespace min_value_expression_l3304_330482

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5/3) :
  4 / (a + 2*b) + 9 / (2*a + b) ≥ 5 :=
sorry

end min_value_expression_l3304_330482


namespace cosine_sum_equality_l3304_330451

theorem cosine_sum_equality : 
  Real.cos (15 * π / 180) * Real.cos (30 * π / 180) + 
  Real.cos (105 * π / 180) * Real.sin (30 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end cosine_sum_equality_l3304_330451


namespace symmetry_proof_l3304_330465

/-- Given two lines in the xy-plane, this function returns true if they are symmetric with respect to the line y = x -/
def are_symmetric_lines (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, line1 x y ↔ line2 y x

/-- The original line: 2x + 3y + 6 = 0 -/
def original_line (x y : ℝ) : Prop :=
  2 * x + 3 * y + 6 = 0

/-- The symmetric line to be proved: 3x + 2y + 6 = 0 -/
def symmetric_line (x y : ℝ) : Prop :=
  3 * x + 2 * y + 6 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line with respect to y = x -/
theorem symmetry_proof : are_symmetric_lines original_line symmetric_line :=
sorry

end symmetry_proof_l3304_330465


namespace olympic_mascot_pricing_and_purchasing_l3304_330413

theorem olympic_mascot_pricing_and_purchasing
  (small_price large_price : ℝ)
  (h1 : large_price - 2 * small_price = 20)
  (h2 : 3 * small_price + 2 * large_price = 390)
  (budget : ℝ) (total_sets : ℕ)
  (h3 : budget = 1500)
  (h4 : total_sets = 20) :
  small_price = 50 ∧ 
  large_price = 120 ∧ 
  (∃ m : ℕ, m ≤ total_sets ∧ 
    m * large_price + (total_sets - m) * small_price ≤ budget ∧
    ∀ n : ℕ, n > m → n * large_price + (total_sets - n) * small_price > budget) ∧
  (7 : ℕ) * large_price + (total_sets - 7) * small_price ≤ budget :=
by sorry

end olympic_mascot_pricing_and_purchasing_l3304_330413


namespace arithmetic_sequence_square_root_l3304_330486

theorem arithmetic_sequence_square_root (x : ℝ) :
  x > 0 →
  (∃ d : ℝ, 2^2 + d = x^2 ∧ x^2 + d = 5^2) →
  x = Real.sqrt 14.5 :=
by sorry

end arithmetic_sequence_square_root_l3304_330486


namespace subtracted_value_l3304_330467

def original_number : ℝ := 54

theorem subtracted_value (x : ℝ) :
  ((original_number - x) / 7 = 7) ∧
  ((original_number - 34) / 10 = 2) →
  x = 5 := by
  sorry

end subtracted_value_l3304_330467


namespace x_needs_18_days_l3304_330485

/-- The time needed for x to finish the remaining work after y leaves -/
def remaining_time_for_x (x_time y_time y_worked : ℚ) : ℚ :=
  (1 - y_worked / y_time) * x_time

/-- Proof that x needs 18 days to finish the remaining work -/
theorem x_needs_18_days (x_time y_time y_worked : ℚ) 
  (hx : x_time = 36)
  (hy : y_time = 24)
  (hw : y_worked = 12) :
  remaining_time_for_x x_time y_time y_worked = 18 := by
  sorry

#eval remaining_time_for_x 36 24 12

end x_needs_18_days_l3304_330485


namespace square_area_comparison_l3304_330431

theorem square_area_comparison (a b : ℝ) (h : b = 4 * a) :
  b ^ 2 = 16 * a ^ 2 := by sorry

end square_area_comparison_l3304_330431


namespace train_length_l3304_330407

/-- The length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) :
  speed = 36 * (1000 / 3600) →
  time = 25.997920166386688 →
  bridge_length = 150 →
  speed * time - bridge_length = 109.97920166386688 := by
  sorry

end train_length_l3304_330407


namespace log_product_theorem_l3304_330457

theorem log_product_theorem (c d : ℕ+) : 
  (d.val - c.val = 435) → 
  (Real.log d.val / Real.log c.val = 2) → 
  (c.val + d.val = 930) := by
sorry

end log_product_theorem_l3304_330457


namespace geometric_sequence_fifth_term_l3304_330476

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_sum : a 1 + 2 * a 2 = 4)
  (h_product : (a 4) ^ 2 = 4 * a 3 * a 7) :
  a 5 = 1/8 := by
sorry

end geometric_sequence_fifth_term_l3304_330476


namespace equilateral_triangle_side_length_l3304_330444

/-- 
Given an equilateral triangle with a perimeter of 63 cm, 
prove that the length of one side is 21 cm.
-/
theorem equilateral_triangle_side_length 
  (perimeter : ℝ) 
  (is_equilateral : Bool) :
  perimeter = 63 ∧ is_equilateral = true → 
  ∃ (side_length : ℝ), side_length = 21 ∧ perimeter = 3 * side_length :=
by
  sorry

end equilateral_triangle_side_length_l3304_330444


namespace no_adjacent_birch_probability_l3304_330414

def num_maple : ℕ := 4
def num_oak : ℕ := 5
def num_birch : ℕ := 6
def num_pine : ℕ := 2
def total_trees : ℕ := num_maple + num_oak + num_birch + num_pine

def probability_no_adjacent_birch : ℚ := 21 / 283

theorem no_adjacent_birch_probability :
  let total_arrangements := (total_trees.choose num_birch : ℚ)
  let valid_arrangements := ((total_trees - num_birch + 1).choose num_birch : ℚ)
  valid_arrangements / total_arrangements = probability_no_adjacent_birch :=
by sorry

end no_adjacent_birch_probability_l3304_330414


namespace point_on_line_l3304_330489

theorem point_on_line (m n : ℝ) : 
  (m = n / 6 - 2 / 5) ∧ (m + p = (n + 18) / 6 - 2 / 5) → p = 3 :=
by sorry

end point_on_line_l3304_330489


namespace candy_ratio_problem_l3304_330462

/-- Proof of candy ratio problem -/
theorem candy_ratio_problem (chocolate_bars : ℕ) (mm_multiplier : ℕ) (candies_per_basket : ℕ) (num_baskets : ℕ)
  (h1 : chocolate_bars = 5)
  (h2 : mm_multiplier = 7)
  (h3 : candies_per_basket = 10)
  (h4 : num_baskets = 25) :
  (num_baskets * candies_per_basket - chocolate_bars - mm_multiplier * chocolate_bars) / (mm_multiplier * chocolate_bars) = 6 := by
  sorry

end candy_ratio_problem_l3304_330462
