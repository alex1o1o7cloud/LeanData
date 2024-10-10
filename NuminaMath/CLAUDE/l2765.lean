import Mathlib

namespace lolita_milk_consumption_l2765_276583

/-- Lolita's weekly milk consumption --/
theorem lolita_milk_consumption :
  let weekday_consumption : ℕ := 3
  let saturday_consumption : ℕ := 2 * weekday_consumption
  let sunday_consumption : ℕ := 3 * weekday_consumption
  let weekdays : ℕ := 5
  weekdays * weekday_consumption + saturday_consumption + sunday_consumption = 30 := by
  sorry

end lolita_milk_consumption_l2765_276583


namespace tangent_line_y_intercept_l2765_276517

/-- Given a real number a, f(x) = ax - ln x, and l is the tangent line to f at (1, f(1)),
    prove that the y-intercept of l is 1. -/
theorem tangent_line_y_intercept (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x - Real.log x
  let f' : ℝ → ℝ := λ x => a - 1 / x
  let slope : ℝ := f' 1
  let point : ℝ × ℝ := (1, f 1)
  let l : ℝ → ℝ := λ x => slope * (x - point.1) + point.2
  l 0 = 1 := by sorry

end tangent_line_y_intercept_l2765_276517


namespace tank_fill_time_l2765_276586

def fill_time_A : ℝ := 60
def fill_time_B : ℝ := 40

theorem tank_fill_time :
  let total_time : ℝ := 30
  let first_half_time : ℝ := total_time / 2
  let second_half_time : ℝ := total_time / 2
  let fill_rate_A : ℝ := 1 / fill_time_A
  let fill_rate_B : ℝ := 1 / fill_time_B
  (fill_rate_B * first_half_time) + ((fill_rate_A + fill_rate_B) * second_half_time) = 1 :=
by sorry

end tank_fill_time_l2765_276586


namespace joan_balloons_l2765_276572

/-- The number of blue balloons Joan has now -/
def total_balloons (initial : ℕ) (gained : ℕ) : ℕ :=
  initial + gained

/-- Proof that Joan has 11 blue balloons now -/
theorem joan_balloons : total_balloons 9 2 = 11 := by
  sorry

end joan_balloons_l2765_276572


namespace wall_painting_contribution_l2765_276504

/-- Calculates the individual contribution for a wall painting project --/
theorem wall_painting_contribution
  (total_area : ℝ)
  (coverage_per_gallon : ℝ)
  (cost_per_gallon : ℝ)
  (num_coats : ℕ)
  (h_total_area : total_area = 1600)
  (h_coverage : coverage_per_gallon = 400)
  (h_cost : cost_per_gallon = 45)
  (h_coats : num_coats = 2) :
  (total_area / coverage_per_gallon * cost_per_gallon * num_coats) / 2 = 180 := by
  sorry

#check wall_painting_contribution

end wall_painting_contribution_l2765_276504


namespace translation_of_points_l2765_276573

/-- Given two points A and B in ℝ², if A is translated to A₁, 
    then B translated by the same vector results in B₁ -/
theorem translation_of_points (A B A₁ B₁ : ℝ × ℝ) : 
  A = (-1, 0) → 
  B = (1, 2) → 
  A₁ = (2, -1) → 
  B₁.1 = B.1 + (A₁.1 - A.1) ∧ B₁.2 = B.2 + (A₁.2 - A.2) → 
  B₁ = (4, 1) := by
  sorry

end translation_of_points_l2765_276573


namespace equation_solution_l2765_276568

theorem equation_solution : ∃ x : ℚ, 5*x + 9*x = 360 - 7*(x + 4) ∧ x = 332/21 := by
  sorry

end equation_solution_l2765_276568


namespace equation_solution_l2765_276558

theorem equation_solution : 
  ∃ (x : ℚ), x ≠ 1 ∧ x ≠ (1/2 : ℚ) ∧ (x / (x - 1) = 3 / (2*x - 2) - 2) ∧ x = (7/6 : ℚ) := by
  sorry

end equation_solution_l2765_276558


namespace smallest_m_is_correct_l2765_276591

/-- The smallest positive value of m for which the equation 15x^2 - mx + 630 = 0 has integral solutions -/
def smallest_m : ℕ := 195

/-- Predicate to check if a quadratic equation ax^2 + bx + c = 0 has integral solutions -/
def has_integral_solutions (a b c : ℤ) : Prop :=
  ∃ x : ℤ, a * x^2 + b * x + c = 0

theorem smallest_m_is_correct :
  (∀ m : ℕ, m < smallest_m → ¬(has_integral_solutions 15 (-m) 630)) ∧
  (has_integral_solutions 15 (-smallest_m) 630) :=
sorry

end smallest_m_is_correct_l2765_276591


namespace star_composition_l2765_276552

-- Define the star operation
def star (x y : ℝ) : ℝ := x^3 - x*y

-- Theorem statement
theorem star_composition (j : ℝ) : star j (star j j) = 2*j^3 - j^4 := by
  sorry

end star_composition_l2765_276552


namespace distance_to_point_l2765_276565

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 2*y + 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := sorry

-- Define the distance function between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem distance_to_point : distance circle_center (10, 5) = 4 * Real.sqrt 5 := by
  sorry

end distance_to_point_l2765_276565


namespace fruit_basket_count_l2765_276522

/-- The number of possible fruit baskets with at least one piece of fruit -/
def num_fruit_baskets (num_apples : Nat) (num_oranges : Nat) : Nat :=
  (num_apples + 1) * (num_oranges + 1) - 1

/-- Theorem stating the number of fruit baskets with 7 apples and 12 oranges -/
theorem fruit_basket_count :
  num_fruit_baskets 7 12 = 103 := by
  sorry

end fruit_basket_count_l2765_276522


namespace descent_problem_l2765_276592

/-- A function that calculates the final elevation after descending --/
def final_elevation (initial_elevation rate_of_descent duration : ℝ) : ℝ :=
  initial_elevation - rate_of_descent * duration

/-- Theorem stating that descending from 400 feet at 10 feet per minute for 5 minutes results in an elevation of 350 feet --/
theorem descent_problem :
  final_elevation 400 10 5 = 350 := by
  sorry

end descent_problem_l2765_276592


namespace angle_AOF_is_118_l2765_276581

/-- Given a configuration of angles where:
    ∠AOB = ∠BOC
    ∠COD = ∠DOE = ∠EOF
    ∠AOD = 82°
    ∠BOE = 68°
    Prove that ∠AOF = 118° -/
theorem angle_AOF_is_118 (AOB BOC COD DOE EOF AOD BOE : ℝ) : 
  AOB = BOC ∧ 
  COD = DOE ∧ DOE = EOF ∧
  AOD = 82 ∧
  BOE = 68 →
  AOB + BOC + COD + DOE + EOF = 118 := by
  sorry

end angle_AOF_is_118_l2765_276581


namespace yellow_ball_probability_l2765_276563

-- Define the number of balls of each color
def red_balls : ℕ := 2
def yellow_balls : ℕ := 5
def blue_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + yellow_balls + blue_balls

-- Define the probability of choosing a yellow ball
def prob_yellow : ℚ := yellow_balls / total_balls

-- Theorem statement
theorem yellow_ball_probability : prob_yellow = 5 / 11 := by
  sorry

end yellow_ball_probability_l2765_276563


namespace line_through_intersection_l2765_276539

/-- The line l: ax - y + b = 0 passes through the intersection point of 
    lines l₁: 2x - 2y - 3 = 0 and l₂: 3x - 5y + 1 = 0 
    if and only if 17a + 4b = 11 -/
theorem line_through_intersection (a b : ℝ) : 
  (∃ x y : ℝ, 2*x - 2*y - 3 = 0 ∧ 3*x - 5*y + 1 = 0 ∧ a*x - y + b = 0) ↔ 
  17*a + 4*b = 11 := by
sorry

end line_through_intersection_l2765_276539


namespace y_squared_equals_zx_sufficient_not_necessary_l2765_276580

-- Define a function to check if three numbers form an arithmetic sequence
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

-- Define the theorem
theorem y_squared_equals_zx_sufficient_not_necessary 
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (is_arithmetic_sequence (Real.log x) (Real.log y) (Real.log z) → y^2 = z*x) ∧
  ¬(y^2 = z*x → is_arithmetic_sequence (Real.log x) (Real.log y) (Real.log z)) :=
by sorry

end y_squared_equals_zx_sufficient_not_necessary_l2765_276580


namespace work_days_calculation_l2765_276571

theorem work_days_calculation (days_a days_b : ℕ) (wage_c : ℕ) (total_earning : ℕ) :
  days_a = 6 →
  days_b = 9 →
  wage_c = 95 →
  total_earning = 1406 →
  ∃ (days_c : ℕ),
    (3 * wage_c * days_a + 4 * wage_c * days_b + 5 * wage_c * days_c = 5 * total_earning) ∧
    days_c = 4 :=
by sorry

end work_days_calculation_l2765_276571


namespace recipe_total_cups_l2765_276505

/-- Calculates the total cups of ingredients in a recipe given the ratio and amount of sugar -/
def total_cups (butter_ratio : ℚ) (flour_ratio : ℚ) (sugar_ratio : ℚ) (sugar_cups : ℚ) : ℚ :=
  let total_ratio := butter_ratio + flour_ratio + sugar_ratio
  let part_size := sugar_cups / sugar_ratio
  (total_ratio * part_size)

/-- Proves that for a recipe with butter:flour:sugar ratio of 1:5:3 and 6 cups of sugar, 
    the total amount of ingredients is 18 cups -/
theorem recipe_total_cups : 
  total_cups 1 5 3 6 = 18 := by
  sorry

end recipe_total_cups_l2765_276505


namespace solution_set_part_i_solution_set_part_ii_l2765_276550

-- Define the function f(x) = |2x-a| + 5x
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + 5*x

-- Part I: Solution set when a = 3
theorem solution_set_part_i :
  ∀ x : ℝ, f 3 x ≥ 5*x + 1 ↔ x ≤ 1 ∨ x ≥ 2 := by sorry

-- Part II: Value of a for given solution set
theorem solution_set_part_ii :
  (∀ x : ℝ, f 3 x ≤ 0 ↔ x ≤ -1) := by sorry

end solution_set_part_i_solution_set_part_ii_l2765_276550


namespace ant_collision_theorem_l2765_276582

/-- Represents the possible numbers of ants on the track -/
def PossibleAntCounts : Set ℕ := {10, 11, 14, 25}

/-- Represents a configuration of ants on the track -/
structure AntConfiguration where
  clockwise : ℕ
  counterclockwise : ℕ

/-- Checks if a given configuration is valid -/
def isValidConfiguration (config : AntConfiguration) : Prop :=
  config.clockwise * config.counterclockwise = 24

theorem ant_collision_theorem
  (track_length : ℕ)
  (ant_speed : ℕ)
  (collision_pairs : ℕ)
  (h1 : track_length = 60)
  (h2 : ant_speed = 1)
  (h3 : collision_pairs = 48) :
  ∀ (total_ants : ℕ),
    (∃ (config : AntConfiguration),
      config.clockwise + config.counterclockwise = total_ants ∧
      isValidConfiguration config) →
    total_ants ∈ PossibleAntCounts :=
sorry

end ant_collision_theorem_l2765_276582


namespace card_number_solution_l2765_276527

theorem card_number_solution : ∃ (L O M N S V : ℕ), 
  (L < 10) ∧ (O < 10) ∧ (M < 10) ∧ (N < 10) ∧ (S < 10) ∧ (V < 10) ∧
  (L ≠ O) ∧ (L ≠ M) ∧ (L ≠ N) ∧ (L ≠ S) ∧ (L ≠ V) ∧
  (O ≠ M) ∧ (O ≠ N) ∧ (O ≠ S) ∧ (O ≠ V) ∧
  (M ≠ N) ∧ (M ≠ S) ∧ (M ≠ V) ∧
  (N ≠ S) ∧ (N ≠ V) ∧
  (S ≠ V) ∧
  (0 < O) ∧ (O < M) ∧ (O < S) ∧
  (L + O * S + O * M + N * M * S + O * M = 10 * M * S + V * M * S) :=
by sorry


end card_number_solution_l2765_276527


namespace janes_breakfast_l2765_276537

theorem janes_breakfast (b m : ℕ) : 
  b + m = 7 →
  (90 * b + 40 * m) % 100 = 0 →
  b = 4 :=
by sorry

end janes_breakfast_l2765_276537


namespace perfect_square_condition_l2765_276524

theorem perfect_square_condition (A B : ℤ) : 
  (800 < A ∧ A < 1300) → 
  B > 1 → 
  A = B^4 → 
  (∃ n : ℤ, A = n^2) ↔ (B = 5 ∨ B = 6) := by
sorry

end perfect_square_condition_l2765_276524


namespace chord_length_theorem_l2765_276528

/-- Represents a circle with a given radius and center point -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/-- Checks if a smaller circle is internally tangent to a larger circle -/
def is_internally_tangent (small large : Circle) : Prop :=
  (small.center.1 - large.center.1)^2 + (small.center.2 - large.center.2)^2 = (large.radius - small.radius)^2

/-- Represents the common external tangent chord length -/
def common_external_tangent_chord_length_squared (c1 c2 c3 : Circle) : ℝ := 72

theorem chord_length_theorem (c1 c2 c3 : Circle)
  (h1 : c1.radius = 3)
  (h2 : c2.radius = 6)
  (h3 : c3.radius = 9)
  (h4 : are_externally_tangent c1 c2)
  (h5 : is_internally_tangent c1 c3)
  (h6 : is_internally_tangent c2 c3) :
  common_external_tangent_chord_length_squared c1 c2 c3 = 72 := by
  sorry

end chord_length_theorem_l2765_276528


namespace necessary_but_not_sufficient_l2765_276514

/-- The equation represents a circle if and only if this condition holds -/
def is_circle (a : ℝ) : Prop := 4 + 4 - 4*a > 0

/-- The condition we're examining -/
def condition (a : ℝ) : Prop := a ≤ 2

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, is_circle a → condition a) ∧
  ¬(∀ a : ℝ, condition a → is_circle a) :=
sorry

end necessary_but_not_sufficient_l2765_276514


namespace circle_tangency_radius_sum_l2765_276566

/-- A circle with center D(r, r) is tangent to the positive x and y-axes
    and externally tangent to a circle centered at (5,0) with radius 1.
    The sum of all possible radii of the circle with center D is 12. -/
theorem circle_tangency_radius_sum : 
  ∀ r : ℝ, 
    (r > 0) →
    ((r - 5)^2 + r^2 = (r + 1)^2) →
    (∃ s : ℝ, (s > 0) ∧ ((s - 5)^2 + s^2 = (s + 1)^2) ∧ (r + s = 12)) :=
by sorry

end circle_tangency_radius_sum_l2765_276566


namespace B_power_2023_l2765_276576

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, -1]]

theorem B_power_2023 :
  B ^ 2023 = ![![ 0,  1,  0],
               ![-1,  0,  0],
               ![ 0,  0, -1]] := by sorry

end B_power_2023_l2765_276576


namespace room_size_l2765_276577

/-- Given two square carpets in a square room, prove the room's side length is 19 meters. -/
theorem room_size (small_carpet big_carpet room : ℝ) : 
  small_carpet > 0 ∧ 
  big_carpet = 2 * small_carpet ∧
  (room - small_carpet - big_carpet)^2 = 4 ∧
  (room - big_carpet) * (room - small_carpet) = 14 →
  room = 19 := by
  sorry

end room_size_l2765_276577


namespace pure_imaginary_product_l2765_276567

theorem pure_imaginary_product (a : ℝ) : 
  (Complex.I * (6 - a) : ℂ) = (3 - Complex.I) * (a + 2 * Complex.I) → a = -2/3 := by
  sorry

end pure_imaginary_product_l2765_276567


namespace seven_factorial_divisors_l2765_276588

/-- The number of positive divisors of n! -/
def num_divisors_factorial (n : ℕ) : ℕ := sorry

/-- 7! has 60 positive divisors -/
theorem seven_factorial_divisors : num_divisors_factorial 7 = 60 := by sorry

end seven_factorial_divisors_l2765_276588


namespace inequality_solution_set_l2765_276531

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, m * (x - 1) > x^2 - x ↔ 1 < x ∧ x < 2) → m = 2 := by
  sorry

end inequality_solution_set_l2765_276531


namespace solution_pairs_l2765_276547

theorem solution_pairs (a b : ℝ) :
  2 * (a^2 + 1) * (b^2 + 1) = (a + 1) * (b + 1) * (a * b + 1) →
  ((a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1)) := by
sorry

end solution_pairs_l2765_276547


namespace art_arrangement_probability_l2765_276564

/-- The probability of arranging n items with k specific items consecutive -/
def consecutive_probability (n : ℕ) (k : ℕ) : ℚ :=
  if k ≤ n ∧ k > 0 then
    (Nat.factorial (n - k + 1) * Nat.factorial k) / Nat.factorial n
  else
    0

/-- Theorem: The probability of arranging 12 pieces of art with 4 specific pieces consecutive is 1/55 -/
theorem art_arrangement_probability :
  consecutive_probability 12 4 = 1 / 55 := by
  sorry

#eval consecutive_probability 12 4

end art_arrangement_probability_l2765_276564


namespace multiple_between_factorials_l2765_276597

theorem multiple_between_factorials (n : ℕ) (h : n ≥ 4) :
  ∃ k : ℕ, n.factorial < k * n^3 ∧ k * n^3 < (n + 1).factorial := by
  sorry

end multiple_between_factorials_l2765_276597


namespace systematic_sampling_interval_example_l2765_276538

/-- Calculates the sampling interval for systematic sampling -/
def systematicSamplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The systematic sampling interval for a population of 2000 and sample size of 50 is 40 -/
theorem systematic_sampling_interval_example :
  systematicSamplingInterval 2000 50 = 40 := by
  sorry

#eval systematicSamplingInterval 2000 50

end systematic_sampling_interval_example_l2765_276538


namespace number_categorization_l2765_276530

def S : Set ℝ := {-2.5, 0, 8, -2, Real.pi/2, 0.7, -2/3, -1.12112112, 3/4}

theorem number_categorization :
  (∃ P I R : Set ℝ,
    P = {x ∈ S | x > 0} ∧
    I = {x ∈ S | ∃ n : ℤ, x = n} ∧
    R = {x ∈ S | ¬∃ q : ℚ, x = q} ∧
    P = {8, Real.pi/2, 0.7, 3/4} ∧
    I = {0, 8, -2} ∧
    R = {Real.pi/2, -1.12112112}) :=
by sorry

end number_categorization_l2765_276530


namespace sister_ages_l2765_276554

theorem sister_ages (x y : ℕ) (h1 : x - y = 4) (h2 : x^3 - y^3 = 988) : x = 11 ∧ y = 7 := by
  sorry

end sister_ages_l2765_276554


namespace problem_solution_l2765_276521

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem problem_solution :
  (B (1/5) ⊂ A) ∧
  ({a : ℝ | A ∩ B a = B a} = {0, 1/3, 1/5}) := by
  sorry

end problem_solution_l2765_276521


namespace longitude_latitude_unique_identification_l2765_276511

/-- A point on the Earth's surface --/
structure EarthPoint where
  longitude : Real
  latitude : Real

/-- Function to determine if a description can uniquely identify a point --/
def canUniquelyIdentify (description : EarthPoint → Prop) : Prop :=
  ∀ (p1 p2 : EarthPoint), description p1 → description p2 → p1 = p2

/-- Theorem stating that longitude and latitude can uniquely identify a point --/
theorem longitude_latitude_unique_identification :
  canUniquelyIdentify (λ p : EarthPoint => p.longitude = 118 ∧ p.latitude = 40) :=
sorry

end longitude_latitude_unique_identification_l2765_276511


namespace coin_flip_problem_l2765_276520

theorem coin_flip_problem (n : ℕ) : 
  (1 + n : ℚ) / 2^n = 3/16 ↔ n = 5 := by sorry

end coin_flip_problem_l2765_276520


namespace defective_engine_fraction_l2765_276562

theorem defective_engine_fraction :
  let total_batches : ℕ := 5
  let engines_per_batch : ℕ := 80
  let non_defective_engines : ℕ := 300
  let total_engines : ℕ := total_batches * engines_per_batch
  let defective_engines : ℕ := total_engines - non_defective_engines
  (defective_engines : ℚ) / total_engines = 1 / 4 := by
  sorry

end defective_engine_fraction_l2765_276562


namespace evaluate_eight_to_nine_thirds_l2765_276561

theorem evaluate_eight_to_nine_thirds : 8^(9/3) = 512 := by
  sorry

end evaluate_eight_to_nine_thirds_l2765_276561


namespace rate_percent_proof_l2765_276595

/-- Simple interest formula -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem rate_percent_proof (principal interest time : ℚ) 
  (h1 : principal = 800)
  (h2 : interest = 192)
  (h3 : time = 4)
  (h4 : simple_interest principal (6 : ℚ) time = interest) : 
  (6 : ℚ) = (interest * 100) / (principal * time) := by
  sorry

end rate_percent_proof_l2765_276595


namespace cargo_loaded_in_bahamas_l2765_276544

/-- The amount of cargo loaded in the Bahamas is equal to the difference between the final amount of cargo and the initial amount of cargo. -/
theorem cargo_loaded_in_bahamas (initial_cargo final_cargo : ℕ) 
  (h1 : initial_cargo = 5973)
  (h2 : final_cargo = 14696) :
  final_cargo - initial_cargo = 8723 := by
  sorry

end cargo_loaded_in_bahamas_l2765_276544


namespace lakers_win_in_seven_l2765_276560

/-- The probability of the Celtics winning a single game -/
def p_celtics : ℚ := 2/3

/-- The probability of the Lakers winning a single game -/
def p_lakers : ℚ := 1 - p_celtics

/-- The number of ways to choose 3 games out of 6 -/
def ways_to_choose_3_of_6 : ℕ := 20

/-- The probability that the Lakers win the NBA finals in exactly 7 games -/
theorem lakers_win_in_seven (p_celtics : ℚ) (p_lakers : ℚ) (ways_to_choose_3_of_6 : ℕ) :
  p_celtics = 2/3 →
  p_lakers = 1 - p_celtics →
  ways_to_choose_3_of_6 = 20 →
  (ways_to_choose_3_of_6 : ℚ) * p_lakers^3 * p_celtics^3 * p_lakers = 160/2187 :=
by sorry

end lakers_win_in_seven_l2765_276560


namespace triangle_special_angles_l2765_276546

theorem triangle_special_angles (A B C : ℝ) (a b c : ℝ) :
  A > 0 ∧ B > 0 ∧ C > 0 ∧  -- Angles are positive
  A + B + C = Real.pi ∧    -- Sum of angles in a triangle
  C = 2 * A ∧              -- Angle C is twice angle A
  b = 2 * a ∧              -- Side b is twice side a
  a * Real.sin B = b * Real.sin A ∧  -- Law of sines
  a * Real.sin C = c * Real.sin A ∧  -- Law of sines
  a^2 + b^2 = c^2          -- Pythagorean theorem
  →
  A = Real.pi / 6 ∧ B = Real.pi / 2 ∧ C = Real.pi / 3 :=
by sorry

end triangle_special_angles_l2765_276546


namespace tangent_two_identities_l2765_276535

open Real

theorem tangent_two_identities (α : ℝ) (h : tan α = 2) :
  (2 * sin α + 2 * cos α) / (sin α - cos α) = 8 ∧
  (cos (π - α) * cos (π / 2 + α) * sin (α - 3 * π / 2)) /
  (sin (3 * π + α) * sin (α - π) * cos (π + α)) = -1/2 := by
  sorry

end tangent_two_identities_l2765_276535


namespace exponent_for_28_decimal_places_l2765_276541

def base : ℝ := 10^4 * 3.456789

theorem exponent_for_28_decimal_places :
  ∀ n : ℕ, (∃ m : ℕ, base^n * 10^28 = m ∧ m < base^n * 10^29) → n = 14 := by
  sorry

end exponent_for_28_decimal_places_l2765_276541


namespace triangle_proof_l2765_276555

theorem triangle_proof (A B C : ℝ) (a b c : ℝ) :
  a = Real.sqrt 7 →
  b = 2 →
  a * Real.sin B - Real.sqrt 3 * b * Real.cos A = 0 →
  (A = π / 3 ∧ 
   (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2) :=
by sorry

end triangle_proof_l2765_276555


namespace polynomial_simplification_l2765_276525

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 3 * x^2 + 6 * x - 8) = x^3 + x^2 + 3 * x + 3 := by
  sorry

end polynomial_simplification_l2765_276525


namespace not_monotone_decreasing_if_f2_gt_f1_l2765_276585

theorem not_monotone_decreasing_if_f2_gt_f1 
  (f : ℝ → ℝ) (h : f 2 > f 1) : 
  ¬(∀ x y : ℝ, x ≤ y → f x ≥ f y) := by
  sorry

end not_monotone_decreasing_if_f2_gt_f1_l2765_276585


namespace total_attendees_l2765_276507

/-- The admission fee for children in dollars -/
def child_fee : ℚ := 1.5

/-- The admission fee for adults in dollars -/
def adult_fee : ℚ := 4

/-- The total amount collected in dollars -/
def total_collected : ℚ := 5050

/-- The number of children who attended -/
def num_children : ℕ := 700

/-- The number of adults who attended -/
def num_adults : ℕ := 1500

/-- Theorem: The total number of people who entered the fair is 2200 -/
theorem total_attendees : num_children + num_adults = 2200 := by
  sorry

end total_attendees_l2765_276507


namespace vector_addition_l2765_276579

theorem vector_addition :
  let v1 : Fin 2 → ℝ := ![5, -9]
  let v2 : Fin 2 → ℝ := ![-8, 14]
  v1 + v2 = ![(-3), 5] := by
  sorry

end vector_addition_l2765_276579


namespace numbers_less_than_reciprocals_l2765_276529

theorem numbers_less_than_reciprocals :
  let numbers : List ℚ := [-1/2, -3, 1/4, 4, 1/3]
  ∀ x ∈ numbers, (x < 1 / x) ↔ (x = -3 ∨ x = 1/4 ∨ x = 1/3) :=
by sorry

end numbers_less_than_reciprocals_l2765_276529


namespace fraction_ordering_l2765_276596

theorem fraction_ordering : 
  let a : ℚ := 6 / 29
  let b : ℚ := 8 / 31
  let c : ℚ := 10 / 39
  a < c ∧ c < b :=
by sorry

end fraction_ordering_l2765_276596


namespace tower_combinations_l2765_276502

theorem tower_combinations (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) : 
  (Nat.choose n k) = 10 := by
  sorry

end tower_combinations_l2765_276502


namespace smallest_number_divisible_l2765_276549

theorem smallest_number_divisible (n : ℕ) : 
  (∃ (k : ℕ), n - k = 44 ∧ 
   9 ∣ (n - k) ∧ 
   6 ∣ (n - k) ∧ 
   12 ∣ (n - k) ∧ 
   18 ∣ (n - k)) →
  (∀ (m : ℕ), m < n → 
    ¬(∃ (k : ℕ), m - k = 44 ∧ 
      9 ∣ (m - k) ∧ 
      6 ∣ (m - k) ∧ 
      12 ∣ (m - k) ∧ 
      18 ∣ (m - k))) →
  n = 80 :=
by sorry

end smallest_number_divisible_l2765_276549


namespace circle_center_l2765_276510

def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

theorem circle_center : 
  ∃ (h k : ℝ), (∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 4) ∧ h = 2 ∧ k = 0 := by
  sorry

end circle_center_l2765_276510


namespace parallel_lines_minimum_value_l2765_276590

theorem parallel_lines_minimum_value (m n : ℕ+) 
  (h_parallel : (2 : ℝ) / (n - 1 : ℝ) = (m : ℝ) / (n : ℝ)) : 
  (∀ k l : ℕ+, (2 : ℝ) / (l - 1 : ℝ) = (k : ℝ) / (l : ℝ) → 2 * m + n ≤ 2 * k + l) ∧ 
  (∃ k l : ℕ+, (2 : ℝ) / (l - 1 : ℝ) = (k : ℝ) / (l : ℝ) ∧ 2 * k + l = 9) :=
sorry

end parallel_lines_minimum_value_l2765_276590


namespace point_translation_and_line_l2765_276587

/-- Given a point (5,3) translated 4 units left and 1 unit down,
    if the resulting point lies on y = kx - 2, then k = 4 -/
theorem point_translation_and_line (k : ℝ) : 
  let original_point : ℝ × ℝ := (5, 3)
  let translated_point : ℝ × ℝ := (original_point.1 - 4, original_point.2 - 1)
  (translated_point.2 = k * translated_point.1 - 2) → k = 4 := by
sorry

end point_translation_and_line_l2765_276587


namespace snake_diet_l2765_276512

/-- The number of birds each snake eats per day in a forest ecosystem -/
def birds_per_snake (beetles_per_bird : ℕ) (snakes_per_jaguar : ℕ) (num_jaguars : ℕ) (total_beetles : ℕ) : ℕ :=
  (total_beetles / beetles_per_bird) / (num_jaguars * snakes_per_jaguar)

/-- Theorem stating that each snake eats 3 birds per day in the given ecosystem -/
theorem snake_diet :
  birds_per_snake 12 5 6 1080 = 3 := by
  sorry

#eval birds_per_snake 12 5 6 1080

end snake_diet_l2765_276512


namespace circle_tangency_sum_of_radii_l2765_276570

/-- A circle with center C(r, r) is tangent to the positive x and y-axes
    and externally tangent to a circle centered at (5,0) with radius 2.
    The sum of all possible values of r is 14. -/
theorem circle_tangency_sum_of_radii :
  ∀ r : ℝ,
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  (∃ s : ℝ, (s > 0) ∧ ((s - 5)^2 + s^2 = (s + 2)^2) ∧ (r + s = 14)) :=
by sorry

end circle_tangency_sum_of_radii_l2765_276570


namespace max_area_rectangular_frame_l2765_276518

/-- Represents the maximum area of a rectangular frame given budget constraints. -/
theorem max_area_rectangular_frame :
  ∃ (L W : ℕ),
    (3 * L + 5 * W ≤ 100) ∧
    (∀ (L' W' : ℕ), (3 * L' + 5 * W' ≤ 100) → L * W ≥ L' * W') ∧
    L * W = 40 := by
  sorry

end max_area_rectangular_frame_l2765_276518


namespace travis_cereal_cost_l2765_276534

/-- The amount Travis spends on cereal in a year -/
def cereal_cost (boxes_per_week : ℕ) (cost_per_box : ℚ) (weeks_per_year : ℕ) : ℚ :=
  (boxes_per_week * weeks_per_year : ℚ) * cost_per_box

/-- Theorem: Travis spends $312.00 on cereal in a year -/
theorem travis_cereal_cost :
  cereal_cost 2 3 52 = 312 := by
  sorry

#eval cereal_cost 2 3 52

end travis_cereal_cost_l2765_276534


namespace consecutive_squares_equality_l2765_276589

theorem consecutive_squares_equality :
  ∃ (a b c d : ℝ), (b = a + 1 ∧ c = b + 1 ∧ d = c + 1) ∧ (a^2 + b^2 = c^2 + d^2) := by
  sorry

end consecutive_squares_equality_l2765_276589


namespace quadratic_equation_solution_l2765_276506

theorem quadratic_equation_solution (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, k * x^2 + x - 3 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  (x₁ + x₂)^2 + x₁ * x₂ = 4 →
  k = 1/4 := by
  sorry

end quadratic_equation_solution_l2765_276506


namespace number_relationship_l2765_276513

theorem number_relationship : 4^(3/10) < 8^(1/4) ∧ 8^(1/4) < 3^(3/4) := by
  sorry

end number_relationship_l2765_276513


namespace race_length_is_1000_l2765_276598

/-- The length of a race, given the positions of two runners at the end. -/
def race_length (jack_position : ℕ) (distance_apart : ℕ) : ℕ :=
  jack_position + distance_apart

/-- Theorem stating that the race length is 1000 meters given the conditions -/
theorem race_length_is_1000 :
  race_length 152 848 = 1000 := by
  sorry

end race_length_is_1000_l2765_276598


namespace circle_center_x_coordinate_range_l2765_276594

/-- The problem statement as a theorem in Lean 4 -/
theorem circle_center_x_coordinate_range :
  ∀ (O A C M : ℝ × ℝ) (l : ℝ → ℝ) (a : ℝ),
    O = (0, 0) →
    A = (0, 3) →
    (∀ x, l x = x + 1) →
    C.2 = l C.1 →
    C.1 = a →
    ∃ r : ℝ, r = 1 ∧ ∀ p : ℝ × ℝ, (p.1 - C.1)^2 + (p.2 - C.2)^2 = r^2 →
      ∃ M : ℝ × ℝ, (M.1 - C.1)^2 + (M.2 - C.2)^2 = r^2 ∧
        (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4 * ((M.1 - O.1)^2 + (M.2 - O.2)^2) →
          -1 - Real.sqrt 7 / 2 ≤ a ∧ a ≤ -1 + Real.sqrt 7 / 2 :=
by sorry

end circle_center_x_coordinate_range_l2765_276594


namespace arithmetic_sequence_problem_l2765_276593

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_eq : a 2 + a 8 = 15 - a 5) : 
  a 5 = 5 := by
  sorry

end arithmetic_sequence_problem_l2765_276593


namespace blue_faces_cube_l2765_276584

theorem blue_faces_cube (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

end blue_faces_cube_l2765_276584


namespace tetrahedron_volume_ratio_l2765_276551

-- Define the point type
variable {Point : Type*}

-- Define the distance function
variable (dist : Point → Point → ℝ)

-- Define the volume function for tetrahedrons
variable (volume : Point → Point → Point → Point → ℝ)

-- Theorem statement
theorem tetrahedron_volume_ratio
  (A B C D B' C' D' : Point) :
  volume A B C D / volume A B' C' D' =
  (dist A B * dist A C * dist A D) / (dist A B' * dist A C' * dist A D') :=
sorry

end tetrahedron_volume_ratio_l2765_276551


namespace geometric_sequence_a4_value_l2765_276516

/-- A geometric sequence of real numbers. -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) * a (n - 1) = a n ^ 2

/-- Given a geometric sequence satisfying certain conditions, a_4 equals 8. -/
theorem geometric_sequence_a4_value (a : ℕ → ℝ) 
    (h_geom : geometric_sequence a)
    (h_sum : a 2 + a 6 = 34)
    (h_prod : a 3 * a 5 = 64) : 
  a 4 = 8 := by
  sorry

end geometric_sequence_a4_value_l2765_276516


namespace work_multiple_l2765_276501

/-- Given that P persons can complete a work W in 12 days, 
    and mP persons can complete half of the work (W/2) in 3 days,
    prove that the multiple m is 2. -/
theorem work_multiple (P : ℕ) (W : ℝ) (m : ℝ) 
  (h1 : P > 0) (h2 : W > 0) (h3 : m > 0)
  (complete_full : P * 12 * (W / (P * 12)) = W)
  (complete_half : m * P * 3 * (W / (2 * m * P * 3)) = W / 2) : 
  m = 2 := by
  sorry

end work_multiple_l2765_276501


namespace inspector_ratio_l2765_276503

/-- Represents the daily production of a workshop relative to the first workshop -/
structure WorkshopProduction where
  relative_production : ℚ

/-- Represents an inspector group -/
structure InspectorGroup where
  num_inspectors : ℕ

/-- Represents the factory setup and inspection process -/
structure Factory where
  workshops : Fin 6 → WorkshopProduction
  initial_products : ℚ
  inspector_speed : ℚ
  group_a : InspectorGroup
  group_b : InspectorGroup

/-- The theorem stating the ratio of inspectors in group A to group B -/
theorem inspector_ratio (f : Factory) : 
  f.workshops 0 = ⟨1⟩ ∧ 
  f.workshops 1 = ⟨1⟩ ∧ 
  f.workshops 2 = ⟨1⟩ ∧ 
  f.workshops 3 = ⟨1⟩ ∧ 
  f.workshops 4 = ⟨3/4⟩ ∧ 
  f.workshops 5 = ⟨8/3⟩ ∧
  (6 * (f.workshops 0).relative_production + 
   6 * (f.workshops 1).relative_production + 
   6 * (f.workshops 2).relative_production + 
   3 * f.initial_products = 6 * f.inspector_speed * f.group_a.num_inspectors) ∧
  (2 * (f.workshops 3).relative_production + 
   2 * (f.workshops 4).relative_production + 
   2 * f.initial_products = 2 * f.inspector_speed * f.group_b.num_inspectors) ∧
  (6 * (f.workshops 5).relative_production + 
   f.initial_products = 4 * f.inspector_speed * f.group_b.num_inspectors) →
  f.group_a.num_inspectors * 19 = f.group_b.num_inspectors * 18 :=
by sorry

end inspector_ratio_l2765_276503


namespace min_distance_circle_to_line_l2765_276575

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 2

-- Define point M
def point_M : ℝ × ℝ := (-2, 1)

-- Define trajectory E
def trajectory_E (x y : ℝ) : Prop := 4*x + 2*y - 3 = 0

-- Theorem statement
theorem min_distance_circle_to_line :
  ∃ (min_dist : ℝ),
    min_dist = (11 * Real.sqrt 5) / 10 - Real.sqrt 2 ∧
    ∀ (a b : ℝ × ℝ),
      circle_C a.1 a.2 →
      trajectory_E b.1 b.2 →
      Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≥ min_dist :=
sorry

end min_distance_circle_to_line_l2765_276575


namespace sunset_time_calculation_l2765_276578

/-- Represents time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  valid : minutes < 60

/-- Adds a duration to a time, wrapping around 24 hours if necessary -/
def addDuration (t : Time24) (d : Duration) : Time24 := sorry

/-- Converts 24-hour time to 12-hour time string (AM/PM) -/
def to12Hour (t : Time24) : String := sorry

theorem sunset_time_calculation 
  (sunrise : Time24) 
  (daylight : Duration) 
  (h_sunrise : sunrise.hours = 7 ∧ sunrise.minutes = 30)
  (h_daylight : daylight.hours = 11 ∧ daylight.minutes = 10) :
  to12Hour (addDuration sunrise daylight) = "6:40 PM" := by sorry

end sunset_time_calculation_l2765_276578


namespace last_four_digits_of_5_to_2011_last_four_digits_of_5_to_7_last_four_digits_of_5_to_2011_is_8125_l2765_276553

/-- The last four digits of 5^n -/
def lastFourDigits (n : ℕ) : ℕ := 5^n % 10000

/-- The cycle length of the last four digits of powers of 5 -/
def cycleLength : ℕ := 4

theorem last_four_digits_of_5_to_2011 :
  lastFourDigits 2011 = lastFourDigits 7 :=
by sorry

theorem last_four_digits_of_5_to_7 :
  lastFourDigits 7 = 8125 :=
by sorry

theorem last_four_digits_of_5_to_2011_is_8125 :
  lastFourDigits 2011 = 8125 :=
by sorry

end last_four_digits_of_5_to_2011_last_four_digits_of_5_to_7_last_four_digits_of_5_to_2011_is_8125_l2765_276553


namespace polynomial_evaluation_l2765_276515

theorem polynomial_evaluation : 
  103^5 - 5 * 103^4 + 10 * 103^3 - 10 * 103^2 + 5 * 103 - 1 = 11036846832 := by
  sorry

end polynomial_evaluation_l2765_276515


namespace binomial_coefficient_22_5_l2765_276519

theorem binomial_coefficient_22_5 
  (h1 : Nat.choose 20 3 = 1140)
  (h2 : Nat.choose 20 4 = 4845)
  (h3 : Nat.choose 20 5 = 15504) : 
  Nat.choose 22 5 = 26334 := by
  sorry

end binomial_coefficient_22_5_l2765_276519


namespace line_through_point_l2765_276557

/-- Given a line equation bx - (b+2)y = b-3 that passes through the point (3, -5), prove that b = -13/7 --/
theorem line_through_point (b : ℚ) : 
  (b * 3 - (b + 2) * (-5) = b - 3) → b = -13/7 := by
  sorry

end line_through_point_l2765_276557


namespace triangles_in_hexagon_count_l2765_276556

/-- The number of vertices in a hexagon -/
def hexagon_vertices : ℕ := 6

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of different triangles that can be formed using the vertices of a hexagon -/
def triangles_in_hexagon : ℕ := Nat.choose hexagon_vertices triangle_vertices

theorem triangles_in_hexagon_count :
  triangles_in_hexagon = 20 := by sorry

end triangles_in_hexagon_count_l2765_276556


namespace digit_sum_problem_l2765_276533

theorem digit_sum_problem (p q r : ℕ) : 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  0 < p ∧ p < 10 ∧ 0 < q ∧ q < 10 ∧ 0 < r ∧ r < 10 ∧
  (10 * p + q) * (10 * p + r) = 221 →
  p + q + r = 11 := by
sorry

end digit_sum_problem_l2765_276533


namespace partnership_profit_l2765_276526

/-- Calculates the total profit of a partnership business given the investments and one partner's share of the profit -/
theorem partnership_profit 
  (investment_A investment_B investment_C : ℕ) 
  (profit_share_A : ℕ) 
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : profit_share_A = 4260) :
  (investment_A + investment_B + investment_C) * profit_share_A / investment_A = 14200 := by
  sorry

#check partnership_profit

end partnership_profit_l2765_276526


namespace side_margin_width_l2765_276548

/-- Given a sheet of paper with dimensions and margin constraints, prove the side margin width. -/
theorem side_margin_width (sheet_width sheet_length top_bottom_margin : ℝ)
  (typing_area_percentage : ℝ) (h1 : sheet_width = 20)
  (h2 : sheet_length = 30) (h3 : top_bottom_margin = 3)
  (h4 : typing_area_percentage = 0.64) :
  ∃ (side_margin : ℝ),
    side_margin = 2 ∧
    (sheet_width - 2 * side_margin) * (sheet_length - 2 * top_bottom_margin) =
      typing_area_percentage * sheet_width * sheet_length :=
by sorry

end side_margin_width_l2765_276548


namespace circle_center_correct_l2765_276574

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 --/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle --/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Function to find the center of a circle given its equation --/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 4 8 4 (-24) 96
  let center := findCircleCenter eq
  center.x = -1 ∧ center.y = 3 := by sorry

end circle_center_correct_l2765_276574


namespace apple_purchase_problem_l2765_276532

theorem apple_purchase_problem (x : ℕ) : 
  (12 : ℚ) / x - (12 : ℚ) / (x + 2) = 1 / 12 → x + 2 = 18 := by
  sorry

end apple_purchase_problem_l2765_276532


namespace quadratic_distinct_roots_range_l2765_276523

/-- The range of m for which the quadratic equation (m-1)x^2 + 2x + 1 = 0 has two distinct real roots -/
theorem quadratic_distinct_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (m - 1) * x^2 + 2 * x + 1 = 0 ∧ (m - 1) * y^2 + 2 * y + 1 = 0) ↔ 
  (m < 2 ∧ m ≠ 1) :=
by sorry

end quadratic_distinct_roots_range_l2765_276523


namespace car_selection_problem_l2765_276542

theorem car_selection_problem (num_cars : ℕ) (selections_per_car : ℕ) (cars_per_client : ℕ)
  (h_num_cars : num_cars = 15)
  (h_selections_per_car : selections_per_car = 3)
  (h_cars_per_client : cars_per_client = 3) :
  (num_cars * selections_per_car) / cars_per_client = 15 := by
  sorry

end car_selection_problem_l2765_276542


namespace correct_operation_l2765_276545

theorem correct_operation (a : ℝ) : 2 * a^3 - a^3 = a^3 := by
  sorry

end correct_operation_l2765_276545


namespace expression_evaluation_l2765_276569

theorem expression_evaluation (b : ℚ) (h : b = 4/3) :
  (7*b^2 - 15*b + 5) * (3*b - 4) = 0 := by
  sorry

end expression_evaluation_l2765_276569


namespace tangent_line_to_circle_l2765_276543

theorem tangent_line_to_circle (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 3 → x^2 + y^2 = 1 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      x'^2 + y'^2 = 1 → (x' - x)^2 + (y' - y)^2 < δ^2 → 
        (y' - (k * x' + 3))^2 > ε^2 * ((x' - x)^2 + (y' - y)^2)) →
  k = 2 * Real.sqrt 2 ∨ k = -2 * Real.sqrt 2 := by
sorry

end tangent_line_to_circle_l2765_276543


namespace count_integer_segments_specific_triangle_l2765_276540

/-- Represents a right triangle ABC with integer leg lengths -/
structure RightTriangle where
  ab : ℕ  -- Length of leg AB
  bc : ℕ  -- Length of leg BC

/-- Calculates the number of distinct integer lengths of line segments 
    that can be drawn from vertex B to a point on hypotenuse AC -/
def count_integer_segments (t : RightTriangle) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem count_integer_segments_specific_triangle : 
  let t : RightTriangle := { ab := 20, bc := 21 }
  count_integer_segments t = 12 := by
  sorry

end count_integer_segments_specific_triangle_l2765_276540


namespace marcus_baseball_cards_l2765_276509

/-- Given that Carter has 152 baseball cards and Marcus has 58 more than Carter,
    prove that Marcus has 210 baseball cards. -/
theorem marcus_baseball_cards :
  let carter_cards : ℕ := 152
  let difference : ℕ := 58
  let marcus_cards : ℕ := carter_cards + difference
  marcus_cards = 210 := by sorry

end marcus_baseball_cards_l2765_276509


namespace divisibility_of_Z_l2765_276559

/-- Represents a 7-digit positive integer in the form abcabca -/
def Z (a b c : ℕ) : ℕ :=
  1000000 * a + 100000 * b + 10000 * c + 1000 * a + 100 * b + 10 * c + a

/-- Theorem stating that 1001 divides Z for any valid a, b, c -/
theorem divisibility_of_Z (a b c : ℕ) (ha : 0 < a) (ha' : a < 10) (hb : b < 10) (hc : c < 10) :
  1001 ∣ Z a b c := by
  sorry

end divisibility_of_Z_l2765_276559


namespace max_consecutive_expressible_l2765_276599

/-- A function that represents the expression x^3 + 2y^2 --/
def f (x y : ℤ) : ℤ := x^3 + 2*y^2

/-- The property of being expressible in the form x^3 + 2y^2 --/
def expressible (n : ℤ) : Prop := ∃ x y : ℤ, f x y = n

/-- A sequence of consecutive integers starting from a given integer --/
def consecutive_seq (start : ℤ) (length : ℕ) : Set ℤ :=
  {n : ℤ | start ≤ n ∧ n < start + length}

/-- The main theorem stating the maximal length of consecutive expressible integers --/
theorem max_consecutive_expressible :
  (∃ start : ℤ, ∀ n ∈ consecutive_seq start 5, expressible n) ∧
  (∀ start : ℤ, ∀ length : ℕ, length > 5 →
    ∃ n ∈ consecutive_seq start length, ¬expressible n) :=
sorry

end max_consecutive_expressible_l2765_276599


namespace swimmer_speed_l2765_276536

/-- The speed of a swimmer in still water, given downstream and upstream swim data -/
theorem swimmer_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 30) (h2 : upstream_distance = 20) 
  (h3 : time = 5) : ∃ (v_man v_stream : ℝ),
  downstream_distance / time = v_man + v_stream ∧
  upstream_distance / time = v_man - v_stream ∧
  v_man = 5 := by
  sorry

end swimmer_speed_l2765_276536


namespace arithmetic_sequence_properties_l2765_276500

/-- An arithmetic sequence with common difference d and special properties. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  t : ℚ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d
  h3 : a 1 + t^2 = a 2 + t^3
  h4 : a 2 + t^3 = a 3 + t

/-- The theorem stating the properties of the arithmetic sequence. -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.t = -1/2 ∧ seq.d = 3/8 ∧
  (∃ (m p r : ℕ), m < p ∧ p < r ∧
    seq.a m - 2*seq.t^m = seq.a p - 2*seq.t^p ∧
    seq.a p - 2*seq.t^p = seq.a r - 2*seq.t^r ∧
    seq.a r - 2*seq.t^r = 0 ∧
    m = 1 ∧ p = 3 ∧ r = 4) ∧
  (∀ n : ℕ, seq.a n = 3/8 * n - 11/8) :=
sorry

end arithmetic_sequence_properties_l2765_276500


namespace exam_mean_score_l2765_276508

/-- Given an exam score distribution where 60 is 2 standard deviations below the mean
    and 100 is 3 standard deviations above the mean, the mean score is 76. -/
theorem exam_mean_score (mean std_dev : ℝ)
  (h1 : mean - 2 * std_dev = 60)
  (h2 : mean + 3 * std_dev = 100) :
  mean = 76 := by
  sorry

end exam_mean_score_l2765_276508
