import Mathlib

namespace janet_walk_results_l3770_377005

/-- Represents Janet's walk in the city --/
structure JanetWalk where
  blocks_north : ℕ
  blocks_west : ℕ
  blocks_south : ℕ
  walking_speed : ℕ

/-- Calculates the time Janet needs to get home --/
def time_to_home (walk : JanetWalk) : ℚ :=
  (walk.blocks_west : ℚ) / walk.walking_speed

/-- Calculates the ratio of blocks walked east to south --/
def east_south_ratio (walk : JanetWalk) : ℚ × ℚ :=
  (walk.blocks_west, walk.blocks_south)

/-- Theorem stating the results of Janet's walk --/
theorem janet_walk_results (walk : JanetWalk) 
  (h1 : walk.blocks_north = 3)
  (h2 : walk.blocks_west = 7 * walk.blocks_north)
  (h3 : walk.blocks_south = 8)
  (h4 : walk.walking_speed = 2) :
  time_to_home walk = 21/2 ∧ east_south_ratio walk = (21, 8) := by
  sorry

#eval time_to_home { blocks_north := 3, blocks_west := 21, blocks_south := 8, walking_speed := 2 }
#eval east_south_ratio { blocks_north := 3, blocks_west := 21, blocks_south := 8, walking_speed := 2 }

end janet_walk_results_l3770_377005


namespace grocery_receipt_total_cost_l3770_377087

/-- The total cost of three items after applying a tax -/
def totalCostAfterTax (sponge shampoo soap taxRate : ℚ) : ℚ :=
  let preTaxTotal := sponge + shampoo + soap
  let taxAmount := preTaxTotal * taxRate
  preTaxTotal + taxAmount

/-- Theorem stating that the total cost after tax for the given items is $15.75 -/
theorem grocery_receipt_total_cost :
  totalCostAfterTax (420/100) (760/100) (320/100) (5/100) = 1575/100 := by
  sorry

end grocery_receipt_total_cost_l3770_377087


namespace pencil_distribution_ways_l3770_377013

/-- The number of ways to distribute pencils among friends -/
def distribute_pencils (total_pencils : ℕ) (num_friends : ℕ) (min_pencils : ℕ) : ℕ :=
  Nat.choose (total_pencils - num_friends * min_pencils + num_friends - 1) (num_friends - 1)

/-- Theorem: There are 6 ways to distribute 8 pencils among 3 friends with at least 2 pencils each -/
theorem pencil_distribution_ways : distribute_pencils 8 3 2 = 6 := by
  sorry

end pencil_distribution_ways_l3770_377013


namespace maria_score_l3770_377031

/-- Represents a math contest scoring system -/
structure ScoringSystem where
  correct_points : ℝ
  incorrect_penalty : ℝ

/-- Represents a contestant's performance in the math contest -/
structure ContestPerformance where
  total_questions : ℕ
  correct_answers : ℕ
  incorrect_answers : ℕ
  unanswered_questions : ℕ

/-- Calculates the total score for a contestant given their performance and the scoring system -/
def calculate_score (performance : ContestPerformance) (system : ScoringSystem) : ℝ :=
  (performance.correct_answers : ℝ) * system.correct_points -
  (performance.incorrect_answers : ℝ) * system.incorrect_penalty

/-- Theorem stating that Maria's score in the contest is 12.5 -/
theorem maria_score :
  let system : ScoringSystem := { correct_points := 1, incorrect_penalty := 0.25 }
  let performance : ContestPerformance := {
    total_questions := 30,
    correct_answers := 15,
    incorrect_answers := 10,
    unanswered_questions := 5
  }
  calculate_score performance system = 12.5 := by
  sorry

end maria_score_l3770_377031


namespace sixteen_horses_walking_legs_l3770_377034

/-- Given a number of horses and an equal number of men, with half riding and half walking,
    calculate the number of legs walking on the ground. -/
def legs_walking (num_horses : ℕ) : ℕ :=
  let num_men := num_horses
  let num_walking_men := num_men / 2
  let men_legs := num_walking_men * 2
  let horse_legs := num_horses * 4
  men_legs + horse_legs

/-- Theorem stating that with 16 horses and men, half riding and half walking,
    there are 80 legs walking on the ground. -/
theorem sixteen_horses_walking_legs :
  legs_walking 16 = 80 := by
  sorry

end sixteen_horses_walking_legs_l3770_377034


namespace angle4_measure_l3770_377030

-- Define the triangle and its angles
structure Triangle :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)
  (angle4 : ℝ)
  (angle5 : ℝ)
  (angle6 : ℝ)

-- Define the theorem
theorem angle4_measure (t : Triangle) 
  (h1 : t.angle1 = 76)
  (h2 : t.angle2 = 27)
  (h3 : t.angle3 = 17)
  (h4 : t.angle1 + t.angle2 + t.angle3 + t.angle5 + t.angle6 = 180) -- Sum of angles in the large triangle
  (h5 : t.angle4 + t.angle5 + t.angle6 = 180) -- Sum of angles in the small triangle
  : t.angle4 = 120 := by
  sorry

end angle4_measure_l3770_377030


namespace consecutive_negative_integers_sum_l3770_377088

theorem consecutive_negative_integers_sum (x : ℤ) : 
  x < 0 ∧ x * (x + 1) = 3080 → x + (x + 1) = -111 := by
  sorry

end consecutive_negative_integers_sum_l3770_377088


namespace lucky_larry_coincidence_l3770_377025

theorem lucky_larry_coincidence :
  let a : ℚ := 2
  let b : ℚ := 3
  let c : ℚ := 4
  let d : ℚ := 5
  let f : ℚ := 4/5
  (a - b - c + d * f) = (a - (b - (c - (d * f)))) := by
  sorry

end lucky_larry_coincidence_l3770_377025


namespace mary_earnings_proof_l3770_377042

/-- Calculates Mary's weekly earnings after deductions --/
def maryWeeklyEarnings (
  maxHours : Nat)
  (regularRate : ℚ)
  (overtimeRateIncrease : ℚ)
  (additionalRateIncrease : ℚ)
  (regularHours : Nat)
  (overtimeHours : Nat)
  (taxRate1 : ℚ)
  (taxRate2 : ℚ)
  (taxRate3 : ℚ)
  (taxThreshold1 : ℚ)
  (taxThreshold2 : ℚ)
  (insuranceFee : ℚ)
  (weekendBonus : ℚ)
  (weekendShiftHours : Nat) : ℚ :=
  sorry

theorem mary_earnings_proof :
  maryWeeklyEarnings 70 10 0.3 0.6 40 20 0.15 0.1 0.25 400 600 50 75 8 = 691.25 := by
  sorry

end mary_earnings_proof_l3770_377042


namespace prob_three_red_prob_same_color_prob_not_same_color_l3770_377070

-- Define the probability of drawing a red ball
def prob_red : ℚ := 1 / 2

-- Define the probability of drawing a yellow ball
def prob_yellow : ℚ := 1 - prob_red

-- Define the number of draws
def num_draws : ℕ := 3

-- Theorem for the probability of drawing three red balls
theorem prob_three_red :
  prob_red ^ num_draws = 1 / 8 := by sorry

-- Theorem for the probability of drawing three balls of the same color
theorem prob_same_color :
  prob_red ^ num_draws + prob_yellow ^ num_draws = 1 / 4 := by sorry

-- Theorem for the probability of not drawing all three balls of the same color
theorem prob_not_same_color :
  1 - (prob_red ^ num_draws + prob_yellow ^ num_draws) = 3 / 4 := by sorry

end prob_three_red_prob_same_color_prob_not_same_color_l3770_377070


namespace unique_solution_exists_l3770_377054

theorem unique_solution_exists : ∃! x : ℕ, 
  x < 5311735 ∧
  x % 5 = 0 ∧
  x % 715 = 10 ∧
  x % 247 = 140 ∧
  x % 391 = 245 ∧
  x % 187 = 109 ∧
  x = 10020 := by
sorry

end unique_solution_exists_l3770_377054


namespace sqrt_meaningful_iff_leq_eight_l3770_377002

theorem sqrt_meaningful_iff_leq_eight (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 8 - x) ↔ x ≤ 8 := by
  sorry

end sqrt_meaningful_iff_leq_eight_l3770_377002


namespace number_problem_l3770_377093

theorem number_problem (x : ℚ) : x + (-5/12) - (-5/2) = 1/3 → x = -7/4 := by
  sorry

end number_problem_l3770_377093


namespace system_solution_l3770_377046

theorem system_solution (a b : ℤ) : 
  (∃ x y : ℤ, a * x + 5 * y = 15 ∧ 4 * x - b * y = -2) →
  (4 * (-3) - b * (-1) = -2) →
  (a * 5 + 5 * 4 = 15) →
  a^2023 + (-1/10 * b : ℚ)^2023 = -2 := by
  sorry

end system_solution_l3770_377046


namespace function_order_l3770_377071

/-- A quadratic function f(x) = x^2 + bx + c that satisfies f(x-1) = f(3-x) for all x ∈ ℝ -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The symmetry condition of the function -/
axiom symmetry (b c : ℝ) : ∀ x, f b c (x - 1) = f b c (3 - x)

/-- Theorem stating the order of f(0), f(-2), and f(5) -/
theorem function_order (b c : ℝ) : f b c 0 < f b c (-2) ∧ f b c (-2) < f b c 5 := by
  sorry

end function_order_l3770_377071


namespace simplify_trig_expression_l3770_377082

theorem simplify_trig_expression (x : ℝ) : 
  (3 + 3 * Real.sin x - 3 * Real.cos x) / (3 + 3 * Real.sin x + 3 * Real.cos x) = Real.tan (x / 2) := by
  sorry

end simplify_trig_expression_l3770_377082


namespace hyperbola_eccentricity_range_l3770_377090

-- Define the eccentricities
variable (e₁ e₂ : ℝ)

-- Define the parameters of the hyperbola
variable (a b : ℝ)

-- Define the coordinates of the intersection point M
variable (x y : ℝ)

-- Define the coordinates of the foci
variable (c : ℝ)

-- Theorem statement
theorem hyperbola_eccentricity_range 
  (h1 : a > b) 
  (h2 : b > 0)
  (h3 : x^2 / a^2 - y^2 / b^2 = 1)  -- Hyperbola equation
  (h4 : x > 0 ∧ y > 0)  -- M is in the first quadrant
  (h5 : (x + c) * (x - c) + y^2 = 0)  -- F₁M · F₂M = 0
  (h6 : 3/4 ≤ e₁ ∧ e₁ ≤ 3*Real.sqrt 10/10)  -- Range of e₁
  (h7 : 1/e₁^2 + 1/e₂^2 = 1)  -- Relationship between e₁ and e₂
  : 3*Real.sqrt 2/4 ≤ e₂ ∧ e₂ < Real.sqrt 2 := by
  sorry

end hyperbola_eccentricity_range_l3770_377090


namespace parallel_line_equation_l3770_377077

/-- A line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- The polar axis -/
def polarAxis : Set (ℝ × ℝ) :=
  {p | p.2 = 0}

/-- A line is parallel to the polar axis -/
def isParallelToPolarAxis (l : PolarLine) : Prop :=
  ∃ c : ℝ, ∀ ρ θ : ℝ, l.equation ρ θ ↔ ρ * Real.sin θ = c

/-- The theorem stating that a line parallel to the polar axis has the equation ρ sin θ = c -/
theorem parallel_line_equation (l : PolarLine) :
  isParallelToPolarAxis l ↔
  ∃ c : ℝ, ∀ ρ θ : ℝ, l.equation ρ θ ↔ ρ * Real.sin θ = c :=
sorry

end parallel_line_equation_l3770_377077


namespace initial_mean_calculation_l3770_377016

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 23 ∧ 
  correct_value = 43 ∧ 
  corrected_mean = 36.5 →
  (n : ℝ) * ((n * corrected_mean - (correct_value - wrong_value)) / n) = 36.1 * n :=
by sorry

end initial_mean_calculation_l3770_377016


namespace roots_of_polynomials_l3770_377021

theorem roots_of_polynomials (α : ℂ) : 
  α^2 + α - 1 = 0 → α^3 - 2*α + 1 = 0 := by
  sorry

end roots_of_polynomials_l3770_377021


namespace measure_string_l3770_377003

theorem measure_string (string_length : ℚ) (h : string_length = 2/3) :
  string_length - (1/4 * string_length) = 1/2 := by
  sorry

end measure_string_l3770_377003


namespace yellow_marbles_count_l3770_377079

theorem yellow_marbles_count (total : ℕ) (white yellow green red : ℕ) : 
  total = 50 →
  white = total / 2 →
  green = yellow / 2 →
  red = 7 →
  total = white + yellow + green + red →
  yellow = 12 :=
by
  sorry

end yellow_marbles_count_l3770_377079


namespace min_value_quadratic_l3770_377026

/-- Given a quadratic function f(x) = ax² + bx where a > 0 and b > 0,
    and the slope of the tangent line at x = 1 is 2,
    prove that the minimum value of (8a + b) / (ab) is 9 -/
theorem min_value_quadratic (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_tangent : 2*a + b = 2) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (8*x + y) / (x*y) ≥ 9) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (8*x + y) / (x*y) = 9) :=
sorry

end min_value_quadratic_l3770_377026


namespace ruby_height_l3770_377075

/-- Given the heights of various people, prove Ruby's height -/
theorem ruby_height
  (janet_height : ℕ)
  (charlene_height : ℕ)
  (pablo_height : ℕ)
  (ruby_height : ℕ)
  (h1 : janet_height = 62)
  (h2 : charlene_height = 2 * janet_height)
  (h3 : pablo_height = charlene_height + 70)
  (h4 : ruby_height = pablo_height - 2)
  : ruby_height = 192 := by
  sorry


end ruby_height_l3770_377075


namespace existence_of_nth_root_l3770_377098

theorem existence_of_nth_root (n b : ℕ) (h_n : n > 1) (h_b : b > 1)
  (h : ∀ k : ℕ, k > 1 → ∃ a_k : ℤ, (k : ℤ) ∣ (b - a_k ^ n)) :
  ∃ A : ℤ, (A : ℤ) ^ n = b :=
sorry

end existence_of_nth_root_l3770_377098


namespace modular_inverse_of_5_mod_23_l3770_377011

theorem modular_inverse_of_5_mod_23 :
  ∃ x : ℕ, x ≤ 22 ∧ (5 * x) % 23 = 1 :=
by
  use 14
  sorry

end modular_inverse_of_5_mod_23_l3770_377011


namespace quiz_contest_orderings_l3770_377029

theorem quiz_contest_orderings (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end quiz_contest_orderings_l3770_377029


namespace girl_walking_distance_l3770_377024

/-- The distance traveled by a girl walking at a constant speed for a given time. -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: A girl walking at 5 kmph for 6 hours travels 30 kilometers. -/
theorem girl_walking_distance :
  let speed : ℝ := 5
  let time : ℝ := 6
  distance_traveled speed time = 30 := by
  sorry

end girl_walking_distance_l3770_377024


namespace student_average_grade_previous_year_l3770_377010

/-- Represents the average grade of a student for a given year -/
structure YearlyAverage where
  courses : ℕ
  average : ℝ

/-- Calculates the total points for a year -/
def totalPoints (ya : YearlyAverage) : ℝ := ya.courses * ya.average

theorem student_average_grade_previous_year 
  (last_year : YearlyAverage)
  (prev_year : YearlyAverage)
  (h1 : last_year.courses = 6)
  (h2 : last_year.average = 100)
  (h3 : prev_year.courses = 5)
  (h4 : (totalPoints last_year + totalPoints prev_year) / (last_year.courses + prev_year.courses) = 81) :
  prev_year.average = 58.2 := by
  sorry


end student_average_grade_previous_year_l3770_377010


namespace game_download_time_l3770_377081

theorem game_download_time (total_size : ℕ) (downloaded : ℕ) (speed : ℕ) : 
  total_size = 880 → downloaded = 310 → speed = 3 → 
  (total_size - downloaded) / speed = 190 := by
  sorry

end game_download_time_l3770_377081


namespace triangle_properties_l3770_377059

open Real

structure Triangle (A B C : ℝ) where
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

theorem triangle_properties (A B C : ℝ) (h : Triangle A B C) 
  (h1 : A + B = 3 * C) (h2 : 2 * sin (A - C) = sin B) (h3 : ∃ (AB : ℝ), AB = 5) :
  sin A = (3 * sqrt 10) / 10 ∧ 
  ∃ (height : ℝ), height = 6 ∧ 
    height * 5 / 2 = (sqrt 10 * 3 * sqrt 5 * sqrt 2) / 2 := by
  sorry


end triangle_properties_l3770_377059


namespace point_on_line_l3770_377017

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let A : Point := ⟨0, 3⟩
  let B : Point := ⟨-8, 0⟩
  let C : Point := ⟨16/3, 5⟩
  collinear A B C := by
  sorry


end point_on_line_l3770_377017


namespace bus_ride_cost_l3770_377053

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 3.75

/-- The cost of a train ride from town P to town Q -/
def train_cost : ℝ := bus_cost + 2.35

/-- The theorem stating the cost of a bus ride from town P to town Q -/
theorem bus_ride_cost : bus_cost = 3.75 := by sorry

/-- The condition that a train ride costs $2.35 more than a bus ride -/
axiom train_cost_difference : train_cost = bus_cost + 2.35

/-- The condition that the combined cost of one train ride and one bus ride is $9.85 -/
axiom combined_cost : train_cost + bus_cost = 9.85

end bus_ride_cost_l3770_377053


namespace work_completion_time_l3770_377043

/-- The number of days A takes to complete the work alone -/
def a_days : ℝ := 12

/-- The number of days B takes to complete the work alone -/
def b_days : ℝ := 27.99999999999998

/-- The number of days A worked alone before B joined -/
def a_solo_days : ℝ := 2

/-- The total number of days it takes to complete the work when A and B work together -/
def total_days : ℝ := 9

theorem work_completion_time :
  let a_rate : ℝ := 1 / a_days
  let b_rate : ℝ := 1 / b_days
  let combined_rate : ℝ := a_rate + b_rate
  let work_done_by_a_solo : ℝ := a_rate * a_solo_days
  let remaining_work : ℝ := 1 - work_done_by_a_solo
  remaining_work / combined_rate + a_solo_days = total_days := by
  sorry

end work_completion_time_l3770_377043


namespace tamika_always_wins_l3770_377064

theorem tamika_always_wins : ∀ a b : ℕ, 
  a ∈ ({11, 12, 13} : Set ℕ) → 
  b ∈ ({11, 12, 13} : Set ℕ) → 
  a ≠ b → 
  a * b > (2 + 3 + 4) := by
sorry

end tamika_always_wins_l3770_377064


namespace subset_condition_l3770_377097

theorem subset_condition (a b : ℝ) : 
  let A : Set ℝ := {x | x^2 - 1 = 0}
  let B : Set ℝ := {y | y^2 - 2*a*y + b = 0}
  (B ⊆ A) ∧ (B ≠ ∅) → 
  ((a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) := by
sorry

end subset_condition_l3770_377097


namespace percentage_calculation_l3770_377092

theorem percentage_calculation (P : ℝ) : 
  P * (0.3 * (0.5 * 4000)) = 90 → P = 0.15 := by
  sorry

end percentage_calculation_l3770_377092


namespace area_difference_circle_square_l3770_377067

/-- The difference between the area of a circle with diameter 8 inches and 
    the area of a square with diagonal 8 inches is approximately 18.3 square inches. -/
theorem area_difference_circle_square : 
  let circle_diameter : ℝ := 8
  let square_diagonal : ℝ := 8
  let circle_area : ℝ := π * (circle_diameter / 2)^2
  let square_area : ℝ := (square_diagonal^2) / 2
  let area_difference : ℝ := circle_area - square_area
  ∃ ε > 0, abs (area_difference - 18.3) < ε ∧ ε < 0.1 :=
by sorry

end area_difference_circle_square_l3770_377067


namespace cube_color_probability_l3770_377015

/-- Represents the three possible colors for a cube face -/
inductive Color
  | Red
  | Blue
  | Yellow

/-- Represents a cube with colored faces -/
structure Cube where
  faces : Fin 6 → Color

/-- The probability of each color -/
def colorProb : Color → ℚ
  | _ => 1/3

/-- Checks if a cube configuration satisfies the condition -/
def satisfiesCondition (c : Cube) : Bool :=
  sorry -- Implementation details omitted

/-- Calculates the probability of a cube satisfying the condition -/
noncomputable def probabilityOfSatisfyingCondition : ℚ :=
  sorry -- Implementation details omitted

/-- The main theorem to prove -/
theorem cube_color_probability :
  probabilityOfSatisfyingCondition = 73/243 :=
sorry

end cube_color_probability_l3770_377015


namespace potato_peeling_result_l3770_377068

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ := 60
  homer_rate : ℕ := 4
  christen_rate : ℕ := 6
  homer_solo_time : ℕ := 5

/-- Calculates the number of potatoes Christen peeled and the total time taken -/
def peel_potatoes (scenario : PotatoPeeling) : ℕ × ℕ := by
  sorry

/-- Theorem stating the correct result of the potato peeling scenario -/
theorem potato_peeling_result (scenario : PotatoPeeling) :
  peel_potatoes scenario = (24, 9) := by
  sorry

end potato_peeling_result_l3770_377068


namespace parallel_vectors_x_value_l3770_377095

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = 4 := by
  sorry

end parallel_vectors_x_value_l3770_377095


namespace shopkeeper_profit_percentage_l3770_377018

theorem shopkeeper_profit_percentage 
  (theft_percentage : ℝ) 
  (loss_percentage : ℝ) 
  (profit_percentage : ℝ) : 
  theft_percentage = 60 → 
  loss_percentage = 56 → 
  (1 - theft_percentage / 100) * (1 + profit_percentage / 100) = 1 - loss_percentage / 100 → 
  profit_percentage = 10 := by sorry

end shopkeeper_profit_percentage_l3770_377018


namespace prob_both_white_is_zero_l3770_377033

/-- Two boxes containing marbles -/
structure TwoBoxes where
  box1 : Finset ℕ
  box2 : Finset ℕ
  total_marbles : box1.card + box2.card = 36
  box1_black : ∀ m ∈ box1, m = 0  -- 0 represents black marbles
  prob_both_black : (box1.card : ℚ) / 36 * (box2.filter (λ m => m = 0)).card / box2.card = 18 / 25

/-- The probability of drawing two white marbles -/
def prob_both_white (boxes : TwoBoxes) : ℚ :=
  (boxes.box1.filter (λ m => m ≠ 0)).card / boxes.box1.card *
  (boxes.box2.filter (λ m => m ≠ 0)).card / boxes.box2.card

theorem prob_both_white_is_zero (boxes : TwoBoxes) : prob_both_white boxes = 0 := by
  sorry

end prob_both_white_is_zero_l3770_377033


namespace smallest_positive_period_dependence_l3770_377063

noncomputable def f (a b x : ℝ) : ℝ := a * (Real.cos x)^2 + b * Real.sin x + Real.tan x

theorem smallest_positive_period_dependence (a b : ℝ) :
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x : ℝ), f a b (x + p) = f a b x) ∧
  (∀ (q : ℝ), 0 < q ∧ q < p → ∃ (x : ℝ), f a b (x + q) ≠ f a b x) ∧
  (∀ (a' : ℝ), ∃ (p' : ℝ), p' > 0 ∧ 
    (∀ (x : ℝ), f a' b (x + p') = f a' b x) ∧
    (∀ (q : ℝ), 0 < q ∧ q < p' → ∃ (x : ℝ), f a' b (x + q) ≠ f a' b x) ∧
    p' = p) ∧
  (∃ (b' : ℝ), b' ≠ b → 
    ∀ (p' : ℝ), (∀ (x : ℝ), f a b' (x + p') = f a b' x) →
    (∀ (q : ℝ), 0 < q ∧ q < p' → ∃ (x : ℝ), f a b' (x + q) ≠ f a b' x) →
    p' ≠ p) :=
by sorry


end smallest_positive_period_dependence_l3770_377063


namespace cos_18_degrees_l3770_377086

theorem cos_18_degrees : Real.cos (18 * π / 180) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end cos_18_degrees_l3770_377086


namespace carolyns_project_time_l3770_377089

/-- Represents the embroidering project with given parameters -/
structure EmbroideringProject where
  stitches_per_minute : ℕ
  flower_stitches : ℕ
  unicorn_stitches : ℕ
  godzilla_stitches : ℕ
  num_flowers : ℕ
  num_unicorns : ℕ
  num_godzillas : ℕ
  embroidering_time_before_break : ℕ
  break_duration : ℕ

/-- Calculates the total time needed for the embroidering project -/
def total_time (project : EmbroideringProject) : ℕ :=
  sorry

/-- Theorem stating that the total time for Carolyn's project is 1265 minutes -/
theorem carolyns_project_time :
  let project : EmbroideringProject := {
    stitches_per_minute := 4,
    flower_stitches := 60,
    unicorn_stitches := 180,
    godzilla_stitches := 800,
    num_flowers := 50,
    num_unicorns := 3,
    num_godzillas := 1,
    embroidering_time_before_break := 30,
    break_duration := 5
  }
  total_time project = 1265 := by sorry

end carolyns_project_time_l3770_377089


namespace blueberry_pies_l3770_377004

theorem blueberry_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) :
  total_pies = 36 →
  apple_ratio = 3 →
  blueberry_ratio = 4 →
  cherry_ratio = 5 →
  blueberry_ratio * (total_pies / (apple_ratio + blueberry_ratio + cherry_ratio)) = 12 :=
by sorry

end blueberry_pies_l3770_377004


namespace ned_bomb_diffusal_l3770_377014

/-- Represents the problem of Ned racing to deactivate a time bomb -/
def BombDefusalProblem (total_flights : ℕ) (time_per_flight : ℕ) (bomb_timer : ℕ) (time_spent : ℕ) : Prop :=
  let flights_gone := time_spent / time_per_flight
  let flights_left := total_flights - flights_gone
  let time_left := bomb_timer - (flights_left * time_per_flight)
  time_left = 17

/-- Theorem stating that Ned will have 17 seconds to diffuse the bomb -/
theorem ned_bomb_diffusal :
  BombDefusalProblem 20 11 72 165 :=
sorry

end ned_bomb_diffusal_l3770_377014


namespace cone_base_area_l3770_377080

/-- The area of the base of a cone with slant height 10 and lateral surface that unfolds into a semicircle -/
theorem cone_base_area (l : ℝ) (r : ℝ) : 
  l = 10 →                       -- Slant height is 10
  l = 2 * r →                    -- Lateral surface unfolds into a semicircle
  π * r^2 = 25 * π :=            -- Area of the base is 25π
by sorry

end cone_base_area_l3770_377080


namespace max_sum_given_constraints_l3770_377061

theorem max_sum_given_constraints (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) :
  x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end max_sum_given_constraints_l3770_377061


namespace fraction_multiplication_l3770_377051

theorem fraction_multiplication : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5060 = 759 := by
  sorry

end fraction_multiplication_l3770_377051


namespace polynomial_remainder_l3770_377078

theorem polynomial_remainder (y : ℝ) : 
  ∃ (Q : ℝ → ℝ), y^50 = (y^2 - 5*y + 6) * Q y + ((3^50 - 2^50)*y + (2^50 - 2*3^50 + 2*2^50)) := by
  sorry

end polynomial_remainder_l3770_377078


namespace fiftieth_term_l3770_377049

/-- Sequence defined as a_n = (n + 4) * x^(n-1) for n ≥ 1 -/
def a (n : ℕ) (x : ℝ) : ℝ := (n + 4) * x^(n - 1)

/-- The 50th term of the sequence is 54x^49 -/
theorem fiftieth_term (x : ℝ) : a 50 x = 54 * x^49 := by
  sorry

end fiftieth_term_l3770_377049


namespace cube_volume_from_face_perimeter_l3770_377052

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 32) :
  let side_length := face_perimeter / 4
  let volume := side_length ^ 3
  volume = 512 :=
by sorry

end cube_volume_from_face_perimeter_l3770_377052


namespace minimum_employees_l3770_377094

theorem minimum_employees (work_days : ℕ) (rest_days : ℕ) (daily_requirement : ℕ) : 
  work_days = 5 →
  rest_days = 2 →
  daily_requirement = 32 →
  ∃ min_employees : ℕ,
    min_employees = (daily_requirement * 7 + work_days - 1) / work_days ∧
    min_employees * work_days ≥ daily_requirement * 7 ∧
    ∀ n : ℕ, n < min_employees → n * work_days < daily_requirement * 7 :=
by
  sorry

#eval (32 * 7 + 5 - 1) / 5  -- Should output 45

end minimum_employees_l3770_377094


namespace project_hours_ratio_l3770_377058

/-- Represents the hours charged by Kate -/
def kate_hours : ℕ := sorry

/-- Represents the hours charged by Pat -/
def pat_hours : ℕ := 2 * kate_hours

/-- Represents the hours charged by Mark -/
def mark_hours : ℕ := kate_hours + 110

/-- The total hours charged by all three -/
def total_hours : ℕ := 198

theorem project_hours_ratio :
  pat_hours + kate_hours + mark_hours = total_hours ∧
  pat_hours.gcd mark_hours = pat_hours ∧
  (pat_hours / pat_hours.gcd mark_hours) = 1 ∧
  (mark_hours / pat_hours.gcd mark_hours) = 3 :=
sorry

end project_hours_ratio_l3770_377058


namespace roots_equation_l3770_377039

theorem roots_equation (c d r s : ℝ) : 
  (c^2 - 7*c + 12 = 0) →
  (d^2 - 7*d + 12 = 0) →
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) →
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) →
  s = 169/12 := by
sorry

end roots_equation_l3770_377039


namespace cubic_polynomial_root_l3770_377099

theorem cubic_polynomial_root (Q : ℝ → ℝ) : 
  (∀ x, Q x = x^3 - 6*x^2 + 12*x - 11) →
  (∃ a b c : ℤ, ∀ x, Q x = x^3 + a*x^2 + b*x + c) →
  Q (Real.rpow 3 (1/3) + 2) = 0 :=
by sorry

end cubic_polynomial_root_l3770_377099


namespace integer_triple_solution_l3770_377060

theorem integer_triple_solution (a b c : ℤ) 
  (eq1 : a + b * c = 2017) 
  (eq2 : b + c * a = 8) : 
  c ∈ ({-6, 0, 2, 8} : Set ℤ) := by
  sorry

end integer_triple_solution_l3770_377060


namespace horse_race_probability_l3770_377085

theorem horse_race_probability (X Y Z : ℝ) 
  (no_draw : X + Y + Z = 1)
  (prob_X : X = 1/4)
  (prob_Y : Y = 3/5) : 
  Z = 3/20 := by
  sorry

end horse_race_probability_l3770_377085


namespace min_value_shifted_quadratic_l3770_377035

/-- Given a quadratic function f(x) = x^2 + 4x + 7 - a with minimum value 2,
    prove that g(x) = f(x - 2015) also has minimum value 2 -/
theorem min_value_shifted_quadratic (a : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 + 4*x + 7 - a) ∧ 
   (∃ m, m = 2 ∧ ∀ x, f x ≥ m)) →
  (∃ (g : ℝ → ℝ), (∀ x, g x = (x - 2015)^2 + 4*(x - 2015) + 7 - a) ∧ 
   (∃ m, m = 2 ∧ ∀ x, g x ≥ m)) :=
by sorry

end min_value_shifted_quadratic_l3770_377035


namespace mark_to_jenna_ratio_l3770_377084

/-- The number of math problems in the homework -/
def total_problems : ℕ := 20

/-- The number of problems Martha finished -/
def martha_problems : ℕ := 2

/-- The number of problems Jenna finished -/
def jenna_problems : ℕ := 4 * martha_problems - 2

/-- The number of problems Angela finished -/
def angela_problems : ℕ := 9

/-- The number of problems Mark finished -/
def mark_problems : ℕ := total_problems - (martha_problems + jenna_problems + angela_problems)

/-- Theorem stating the ratio of problems Mark finished to problems Jenna finished -/
theorem mark_to_jenna_ratio : 
  (mark_problems : ℚ) / jenna_problems = 1 / 2 := by sorry

end mark_to_jenna_ratio_l3770_377084


namespace stock_price_calculation_l3770_377020

/-- Proves that given an income of 15000 from an 80% stock and an investment of 37500,
    the price of the stock is 50% of its face value. -/
theorem stock_price_calculation 
  (income : ℝ) 
  (investment : ℝ) 
  (yield : ℝ) 
  (h1 : income = 15000) 
  (h2 : investment = 37500) 
  (h3 : yield = 80) : 
  (income * 100 / (investment * yield)) = 0.5 := by
sorry

end stock_price_calculation_l3770_377020


namespace base4_calculation_l3770_377012

/-- Convert a number from base 4 to base 10 -/
def base4_to_base10 (n : ℕ) : ℕ := sorry

/-- Convert a number from base 10 to base 4 -/
def base10_to_base4 (n : ℕ) : ℕ := sorry

/-- Multiplication in base 4 -/
def mul_base4 (a b : ℕ) : ℕ := 
  base10_to_base4 (base4_to_base10 a * base4_to_base10 b)

/-- Division in base 4 -/
def div_base4 (a b : ℕ) : ℕ := 
  base10_to_base4 (base4_to_base10 a / base4_to_base10 b)

theorem base4_calculation : 
  div_base4 (mul_base4 231 24) 3 = 2310 := by sorry

end base4_calculation_l3770_377012


namespace sin_6phi_l3770_377019

theorem sin_6phi (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * Real.sqrt 8) / 5) :
  Real.sin (6 * φ) = -198 * Real.sqrt 2 / 15625 := by
  sorry

end sin_6phi_l3770_377019


namespace smallest_n_for_non_simplest_fraction_l3770_377065

theorem smallest_n_for_non_simplest_fraction : ∃ (d : ℕ), d > 1 ∧ d ∣ (17 + 2) ∧ d ∣ (3 * 17^2 + 7) ∧
  ∀ (n : ℕ), n > 0 ∧ n < 17 → ∀ (k : ℕ), k > 1 → ¬(k ∣ (n + 2) ∧ k ∣ (3 * n^2 + 7)) :=
by sorry

#check smallest_n_for_non_simplest_fraction

end smallest_n_for_non_simplest_fraction_l3770_377065


namespace book_arrangement_problem_l3770_377027

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (math_copies : ℕ) (novel_copies : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial math_copies * Nat.factorial novel_copies)

/-- Theorem stating the number of arrangements for the given problem -/
theorem book_arrangement_problem :
  arrange_books 7 3 2 = 420 := by sorry

end book_arrangement_problem_l3770_377027


namespace equation_one_integral_root_l3770_377055

theorem equation_one_integral_root :
  ∃! x : ℤ, x - 9 / (x - 5 : ℚ) = 7 - 9 / (x - 5 : ℚ) := by
  sorry

end equation_one_integral_root_l3770_377055


namespace functions_same_domain_range_not_necessarily_equal_l3770_377057

theorem functions_same_domain_range_not_necessarily_equal :
  ∃ (A B : Type) (f g : A → B), (∀ x : A, ∃ y : B, f x = y ∧ g x = y) ∧ f ≠ g :=
sorry

end functions_same_domain_range_not_necessarily_equal_l3770_377057


namespace expression_evaluation_l3770_377040

theorem expression_evaluation : 
  (150^2 - 12^2) / (90^2 - 21^2) * ((90 + 21) * (90 - 21)) / ((150 + 12) * (150 - 12)) = 2 := by
  sorry

end expression_evaluation_l3770_377040


namespace class_average_age_problem_l3770_377072

theorem class_average_age_problem (original_students : ℕ) (new_students : ℕ) (new_average_age : ℕ) (average_decrease : ℕ) :
  original_students = 18 →
  new_students = 18 →
  new_average_age = 32 →
  average_decrease = 4 →
  ∃ (original_average : ℕ),
    (original_students * original_average + new_students * new_average_age) / (original_students + new_students) = original_average - average_decrease ∧
    original_average = 40 := by
  sorry

end class_average_age_problem_l3770_377072


namespace exclusive_proposition_range_l3770_377007

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ (h k r : ℝ), ∀ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

def q (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 - 4 * x + m > 0

-- Define the range of m
def m_range (m : ℝ) : Prop := m < 1/2 ∨ m > 2

-- Theorem statement
theorem exclusive_proposition_range :
  ∀ m : ℝ, (p m ∧ ¬q m) ∨ (¬p m ∧ q m) → m_range m :=
by sorry

end exclusive_proposition_range_l3770_377007


namespace max_value_constraint_max_value_attained_unique_max_value_l3770_377044

theorem max_value_constraint (x y z : ℝ) :
  x^2 + 2*x + (1/5)*y^2 + 7*z^2 = 6 →
  7*x + 10*y + z ≤ 55 :=
by sorry

theorem max_value_attained :
  ∃ x y z : ℝ, x^2 + 2*x + (1/5)*y^2 + 7*z^2 = 6 ∧ 7*x + 10*y + z = 55 :=
by sorry

theorem unique_max_value (x y z : ℝ) :
  x^2 + 2*x + (1/5)*y^2 + 7*z^2 = 6 ∧ 7*x + 10*y + z = 55 →
  x = -13/62 ∧ y = 175/31 ∧ z = 1/62 :=
by sorry

end max_value_constraint_max_value_attained_unique_max_value_l3770_377044


namespace lilith_cap_collection_years_l3770_377073

/-- Represents the cap collection problem for Lilith --/
def cap_collection_problem (years : ℕ) : Prop :=
  let first_year_caps := 3 * 12
  let subsequent_year_caps := 5 * 12
  let christmas_caps := 40
  let lost_caps := 15
  let total_caps := 401
  
  first_year_caps +
  (years - 1) * subsequent_year_caps +
  years * christmas_caps -
  years * lost_caps = total_caps

/-- Theorem stating that Lilith has been collecting caps for 5 years --/
theorem lilith_cap_collection_years : 
  ∃ (years : ℕ), years > 0 ∧ cap_collection_problem years ∧ years = 5 := by
  sorry

end lilith_cap_collection_years_l3770_377073


namespace problem_solution_l3770_377045

theorem problem_solution (x y z : ℚ) 
  (hx : x = 1/3) (hy : y = 1/2) (hz : z = 5/8) :
  x * y * (1 - z) = 1/16 := by
sorry

end problem_solution_l3770_377045


namespace sum_of_squares_l3770_377069

theorem sum_of_squares (x y z : ℤ) : 
  x + y + 57 = 0 → y - z + 17 = 0 → x - z + 44 = 0 → x^2 + y^2 + z^2 = 1993 := by
  sorry

end sum_of_squares_l3770_377069


namespace pole_length_l3770_377001

/-- The length of a pole that fits diagonally in a rectangular opening -/
theorem pole_length (w h : ℝ) (hw : w > 0) (hh : h > 0) : 
  (w + 4)^2 + (h + 2)^2 = 100 → w^2 + h^2 = 100 :=
by sorry

end pole_length_l3770_377001


namespace shrimp_trap_problem_l3770_377050

theorem shrimp_trap_problem (victor_shrimp : ℕ) (austin_shrimp : ℕ) :
  victor_shrimp = 26 →
  (victor_shrimp + austin_shrimp + (victor_shrimp + austin_shrimp) / 2) * 7 / 11 = 42 →
  austin_shrimp + 8 = victor_shrimp :=
by sorry

end shrimp_trap_problem_l3770_377050


namespace smallest_twin_prime_pair_mean_l3770_377076

/-- Twin prime pair -/
def is_twin_prime_pair (p q : Nat) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ q = p + 2

/-- Smallest twin prime pair -/
def smallest_twin_prime_pair (p q : Nat) : Prop :=
  is_twin_prime_pair p q ∧ ∀ (r s : Nat), is_twin_prime_pair r s → p ≤ r

/-- Arithmetic mean of two numbers -/
def arithmetic_mean (a b : Nat) : Rat :=
  (a + b : Rat) / 2

theorem smallest_twin_prime_pair_mean :
  ∃ (p q : Nat), smallest_twin_prime_pair p q ∧ arithmetic_mean p q = 4 :=
sorry

end smallest_twin_prime_pair_mean_l3770_377076


namespace circle_area_from_circumference_l3770_377037

-- Define the circumference of the circle
def circumference : ℝ := 36

-- Theorem stating that the area of a circle with circumference 36 cm is 324/π cm²
theorem circle_area_from_circumference :
  (π * (circumference / (2 * π))^2) = 324 / π := by sorry

end circle_area_from_circumference_l3770_377037


namespace union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l3770_377074

variable {U : Type} -- Universe set
variable (A B C : Set U) -- Sets A, B, C in the universe U

-- Commutativity
theorem union_comm : A ∪ B = B ∪ A := by sorry
theorem inter_comm : A ∩ B = B ∩ A := by sorry

-- Associativity
theorem union_assoc : A ∪ (B ∪ C) = (A ∪ B) ∪ C := by sorry
theorem inter_assoc : A ∩ (B ∩ C) = (A ∩ B) ∩ C := by sorry

-- Distributivity
theorem inter_union_distrib : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := by sorry
theorem union_inter_distrib : A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) := by sorry

-- Idempotence
theorem union_idem : A ∪ A = A := by sorry
theorem inter_idem : A ∩ A = A := by sorry

-- De Morgan's Laws
theorem de_morgan_union : (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ := by sorry
theorem de_morgan_inter : (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ := by sorry

end union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l3770_377074


namespace acid_concentration_percentage_l3770_377047

/-- 
Given a solution with 1.6 litres of pure acid in 8 litres of total volume,
prove that the percentage concentration of the acid is 20%.
-/
theorem acid_concentration_percentage (pure_acid : ℝ) (total_volume : ℝ) :
  pure_acid = 1.6 →
  total_volume = 8 →
  (pure_acid / total_volume) * 100 = 20 := by
  sorry

end acid_concentration_percentage_l3770_377047


namespace cubic_sum_over_product_l3770_377022

theorem cubic_sum_over_product (a b c : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 30) (h5 : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) :
  (a^3 + b^3 + c^3) / (a*b*c) = 33 := by sorry

end cubic_sum_over_product_l3770_377022


namespace existence_of_special_number_l3770_377062

theorem existence_of_special_number : 
  ∃ (n : ℕ) (N : ℕ), n > 2 ∧ 
  N = 2 * 10^(n+1) - 9 ∧ 
  N % 1991 = 0 := by
sorry

end existence_of_special_number_l3770_377062


namespace smallest_number_l3770_377009

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their respective bases
def num1 : List Nat := [5, 8]  -- 85 in base 9
def num2 : List Nat := [0, 1, 2]  -- 210 in base 6
def num3 : List Nat := [0, 0, 0, 1]  -- 1000 in base 4
def num4 : List Nat := [1, 1, 1, 1, 1, 1]  -- 111111 in base 2

-- Theorem statement
theorem smallest_number :
  to_base_10 num4 2 < to_base_10 num1 9 ∧
  to_base_10 num4 2 < to_base_10 num2 6 ∧
  to_base_10 num4 2 < to_base_10 num3 4 :=
by sorry

end smallest_number_l3770_377009


namespace integral_cos_sin_l3770_377066

theorem integral_cos_sin : ∫ x in (0)..(π/2), (1 + Real.cos x) / (1 + Real.sin x + Real.cos x) = Real.log 2 + π/2 := by
  sorry

end integral_cos_sin_l3770_377066


namespace equation_solution_difference_l3770_377038

theorem equation_solution_difference : ∃ (r s : ℝ),
  (∀ x, (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3 ↔ (x = r ∨ x = s)) ∧
  r > s ∧
  r - s = 2 :=
by sorry

end equation_solution_difference_l3770_377038


namespace expression_value_l3770_377091

theorem expression_value : 1 - (-2) - 3 - (-4) - 5 - (-6) - 7 - (-8) = 6 := by
  sorry

end expression_value_l3770_377091


namespace negative_one_to_2002_is_smallest_positive_integer_l3770_377096

theorem negative_one_to_2002_is_smallest_positive_integer :
  (-1 : ℤ) ^ 2002 = 1 ∧ ∀ n : ℤ, n > 0 → n ≥ 1 :=
by sorry

end negative_one_to_2002_is_smallest_positive_integer_l3770_377096


namespace divisible_by_120_l3770_377048

theorem divisible_by_120 (n : ℤ) : ∃ k : ℤ, n^5 - 5*n^3 + 4*n = 120*k := by
  sorry

end divisible_by_120_l3770_377048


namespace max_episodes_l3770_377000

/-- Represents a character in the TV show -/
structure Character where
  id : Nat

/-- Represents the state of knowledge for each character -/
structure KnowledgeState where
  knows_mystery : Set Character
  knows_others_know : Set (Character × Character)
  knows_others_dont_know : Set (Character × Character)

/-- Represents an episode of the TV show -/
inductive Episode
  | LearnMystery (c : Character)
  | LearnSomeoneKnows (c1 c2 : Character)
  | LearnSomeoneDoesntKnow (c1 c2 : Character)

/-- The number of characters in the TV show -/
def num_characters : Nat := 20

/-- Theorem: The maximum number of unique episodes is 780 -/
theorem max_episodes :
  ∃ (episodes : List Episode),
    episodes.length = 780 ∧
    episodes.Nodup ∧
    (∀ e : Episode, e ∈ episodes) ∧
    (∀ c : List Character, c.length = num_characters →
      ∃ (initial_state : KnowledgeState),
        ∃ (final_state : KnowledgeState),
          episodes.foldl
            (fun state episode =>
              match episode with
              | Episode.LearnMystery c =>
                { state with knows_mystery := state.knows_mystery ∪ {c} }
              | Episode.LearnSomeoneKnows c1 c2 =>
                { state with knows_others_know := state.knows_others_know ∪ {(c1, c2)} }
              | Episode.LearnSomeoneDoesntKnow c1 c2 =>
                { state with knows_others_dont_know := state.knows_others_dont_know ∪ {(c1, c2)} })
            initial_state
          = final_state) :=
  sorry

end max_episodes_l3770_377000


namespace fixed_point_on_line_l3770_377008

/-- The line equation parameterized by k -/
def line_equation (x y k : ℝ) : Prop :=
  (1 + 4*k)*x - (2 - 3*k)*y + (2 - 3*k) = 0

/-- The fixed point that the line passes through -/
def fixed_point : ℝ × ℝ := (0, 1)

/-- Theorem stating that the fixed point is the unique point that satisfies the line equation for all k -/
theorem fixed_point_on_line :
  ∀ (k : ℝ), line_equation (fixed_point.1) (fixed_point.2) k ∧
  ∀ (x y : ℝ), (∀ (k : ℝ), line_equation x y k) → (x, y) = fixed_point :=
by sorry

end fixed_point_on_line_l3770_377008


namespace polynomial_value_for_special_x_l3770_377056

theorem polynomial_value_for_special_x :
  let x : ℝ := 1 / (2 - Real.sqrt 3)
  x^6 - 2 * Real.sqrt 3 * x^5 - x^4 + x^3 - 4 * x^2 + 2 * x - Real.sqrt 3 = 4 := by
  sorry

end polynomial_value_for_special_x_l3770_377056


namespace book_reading_time_l3770_377006

theorem book_reading_time (total_pages : ℕ) (planned_days : ℕ) (extra_pages : ℕ) 
    (h1 : total_pages = 960)
    (h2 : planned_days = 20)
    (h3 : extra_pages = 12) :
  (total_pages : ℚ) / ((total_pages / planned_days + extra_pages) : ℚ) = 16 := by
  sorry

end book_reading_time_l3770_377006


namespace vasilya_wins_l3770_377083

/-- Represents a stick with a given length -/
structure Stick where
  length : ℝ
  length_pos : length > 0

/-- Represents a game state with a list of sticks -/
structure GameState where
  sticks : List Stick

/-- Represents a player's strategy for breaking sticks -/
def Strategy := GameState → Nat → Stick

/-- Defines the initial game state with a single 10 cm stick -/
def initialState : GameState :=
  { sticks := [{ length := 10, length_pos := by norm_num }] }

/-- Defines the game play for 18 breaks with alternating players -/
def playGame (petyaStrategy vasilyaStrategy : Strategy) : GameState :=
  sorry -- Implementation of game play

/-- Theorem stating that Vasilya can always ensure at least one stick is not shorter than 1 cm -/
theorem vasilya_wins (petyaStrategy : Strategy) : 
  ∃ (vasilyaStrategy : Strategy), ∃ (s : Stick), s ∈ (playGame petyaStrategy vasilyaStrategy).sticks ∧ s.length ≥ 1 := by
  sorry


end vasilya_wins_l3770_377083


namespace total_fans_count_l3770_377023

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- Calculates the total number of fans -/
def total_fans (fans : FanCounts) : ℕ :=
  fans.yankees + fans.mets + fans.red_sox

/-- Theorem: Given the ratios and number of Mets fans, prove the total number of fans is 360 -/
theorem total_fans_count (fans : FanCounts) 
  (yankees_mets_ratio : fans.yankees = 3 * fans.mets / 2)
  (mets_redsox_ratio : fans.red_sox = 5 * fans.mets / 4)
  (mets_count : fans.mets = 96) :
  total_fans fans = 360 := by
  sorry

#eval total_fans { yankees := 144, mets := 96, red_sox := 120 }

end total_fans_count_l3770_377023


namespace three_leaf_clover_count_l3770_377028

theorem three_leaf_clover_count :
  ∀ (total_leaves : ℕ) (three_leaf_count : ℕ),
    total_leaves = 1000 →
    3 * three_leaf_count + 4 = total_leaves →
    three_leaf_count = 332 := by
  sorry

end three_leaf_clover_count_l3770_377028


namespace least_k_value_l3770_377032

/-- The number of factors in the original equation -/
def n : ℕ := 2016

/-- The total number of factors on both sides of the equation -/
def total_factors : ℕ := 2 * n

/-- A function representing the left-hand side of the equation after erasing factors -/
def left_side (k : ℕ) (x : ℝ) : ℝ := sorry

/-- A function representing the right-hand side of the equation after erasing factors -/
def right_side (k : ℕ) (x : ℝ) : ℝ := sorry

/-- Predicate to check if the equation has no real solutions after erasing k factors -/
def no_real_solutions (k : ℕ) : Prop :=
  ∀ x : ℝ, left_side k x ≠ right_side k x

/-- Predicate to check if at least one factor remains on each side after erasing k factors -/
def factors_remain (k : ℕ) : Prop :=
  k < total_factors

/-- The main theorem stating that 2016 is the least value of k satisfying the conditions -/
theorem least_k_value :
  (∀ k < n, ¬(no_real_solutions k ∧ factors_remain k)) ∧
  (no_real_solutions n ∧ factors_remain n) :=
sorry

end least_k_value_l3770_377032


namespace geometric_sequence_first_term_l3770_377036

/-- A geometric sequence with third term 3 and fifth term 27 has first term 1/3 -/
theorem geometric_sequence_first_term (a : ℝ) (r : ℝ) : 
  a * r^2 = 3 → a * r^4 = 27 → a = 1/3 := by
  sorry

end geometric_sequence_first_term_l3770_377036


namespace rational_closure_l3770_377041

theorem rational_closure (x y : ℚ) (h : y ≠ 0) :
  (∃ a b : ℤ, (x + y = a / b ∧ b ≠ 0)) ∧
  (∃ c d : ℤ, (x - y = c / d ∧ d ≠ 0)) ∧
  (∃ e f : ℤ, (x * y = e / f ∧ f ≠ 0)) ∧
  (∃ g h : ℤ, (x / y = g / h ∧ h ≠ 0)) :=
by sorry

end rational_closure_l3770_377041
