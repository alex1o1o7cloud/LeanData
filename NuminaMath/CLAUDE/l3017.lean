import Mathlib

namespace emerson_rowing_trip_l3017_301705

theorem emerson_rowing_trip (total_distance initial_distance second_part_distance : ℕ) 
  (h1 : total_distance = 39)
  (h2 : initial_distance = 6)
  (h3 : second_part_distance = 15) :
  total_distance - (initial_distance + second_part_distance) = 18 :=
by
  sorry

end emerson_rowing_trip_l3017_301705


namespace square_area_error_l3017_301745

theorem square_area_error (S : ℝ) (h : S > 0) : 
  let measured_side := S * (1 + 0.06)
  let actual_area := S^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 12.36 := by
sorry

end square_area_error_l3017_301745


namespace find_n_l3017_301783

theorem find_n : ∃ n : ℕ, 2^3 * 8 = 4^n ∧ n = 3 := by
  sorry

end find_n_l3017_301783


namespace regular_18gon_symmetry_sum_l3017_301780

/-- A regular polygon with n sides -/
structure RegularPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon) : ℕ := p.n

/-- The smallest positive angle of rotational symmetry for a regular polygon (in degrees) -/
def rotationalSymmetryAngle (p : RegularPolygon) : ℚ := 360 / p.n

/-- The theorem to be proved -/
theorem regular_18gon_symmetry_sum :
  let p : RegularPolygon := ⟨18, by norm_num⟩
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by
  sorry

end regular_18gon_symmetry_sum_l3017_301780


namespace problem_solution_l3017_301739

-- Define the function f
def f (a b x : ℝ) : ℝ := |x - a| - |x + b|

-- Define the function g
def g (a b x : ℝ) : ℝ := -x^2 - a*x - b

theorem problem_solution (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmax : ∀ x, f a b x ≤ 3) 
  (hmax_achieved : ∃ x, f a b x = 3) 
  (hg_less_f : ∀ x ≥ a, g a b x < f a b x) :
  (a + b = 3) ∧ (1/2 < a ∧ a < 3) := by
  sorry

end problem_solution_l3017_301739


namespace trailing_zeros_100_factorial_l3017_301772

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailing_zeros_100_factorial : trailingZeros 100 = 24 := by
  sorry

end trailing_zeros_100_factorial_l3017_301772


namespace intersection_complement_when_m_3_sufficient_necessary_condition_l3017_301796

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

theorem intersection_complement_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | 0 ≤ x ∧ x < 2} := by sorry

theorem sufficient_necessary_condition (m : ℝ) :
  (∀ x, x ∈ B m ↔ x ∈ A) ↔ 1 ≤ m ∧ m ≤ 4 := by sorry

end intersection_complement_when_m_3_sufficient_necessary_condition_l3017_301796


namespace polygon_sides_count_l3017_301785

theorem polygon_sides_count : ∀ n : ℕ,
  (n ≥ 3) →
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 5 :=
by
  sorry

#check polygon_sides_count

end polygon_sides_count_l3017_301785


namespace f_greater_g_when_x_greater_two_sum_greater_four_when_f_equal_l3017_301722

noncomputable def f (x : ℝ) : ℝ := (x - 1) / Real.exp (x - 1)

noncomputable def g (x : ℝ) : ℝ := f (4 - x)

theorem f_greater_g_when_x_greater_two :
  ∀ x : ℝ, x > 2 → f x > g x :=
sorry

theorem sum_greater_four_when_f_equal :
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ = f x₂ → x₁ + x₂ > 4 :=
sorry

end f_greater_g_when_x_greater_two_sum_greater_four_when_f_equal_l3017_301722


namespace largest_positive_integer_theorem_l3017_301768

/-- Binary operation @ defined as n @ n = n - (n * 5) -/
def binary_op (n : ℤ) : ℤ := n - (n * 5)

/-- Proposition: 1 is the largest positive integer n such that n @ n < 10 -/
theorem largest_positive_integer_theorem :
  ∀ n : ℕ, n > 1 → binary_op n ≥ 10 ∧ binary_op 1 < 10 := by
  sorry

end largest_positive_integer_theorem_l3017_301768


namespace tangent_line_intersection_l3017_301778

/-- Theorem: Tangent line intersection for two circles
    Given two circles:
    - Circle 1 with radius 3 and center (0, 0)
    - Circle 2 with radius 5 and center (12, 0)
    The x-coordinate of the point where a line tangent to both circles
    intersects the x-axis (to the right of the origin) is 9/2.
-/
theorem tangent_line_intersection (x : ℚ) : 
  (∃ y : ℚ, (x^2 + y^2 = 3^2 ∧ ((x - 12)^2 + y^2 = 5^2))) → x = 9/2 := by
  sorry

#check tangent_line_intersection

end tangent_line_intersection_l3017_301778


namespace ellipse_triangle_perimeter_l3017_301784

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 25 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  let c := 4
  F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  ellipse x y

-- Theorem statement
theorem ellipse_triangle_perimeter
  (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ)
  (h_ellipse : point_on_ellipse P)
  (h_foci : foci F₁ F₂) :
  let perimeter := dist P F₁ + dist P F₂ + dist F₁ F₂
  perimeter = 18 :=
sorry

end ellipse_triangle_perimeter_l3017_301784


namespace work_completion_l3017_301736

/-- The number of men in the first group -/
def first_group : ℕ := 18

/-- The number of days for the first group to complete the work -/
def first_days : ℕ := 30

/-- The number of days for the second group to complete the work -/
def second_days : ℕ := 36

/-- The number of men in the second group -/
def second_group : ℕ := (first_group * first_days) / second_days

theorem work_completion :
  second_group = 15 := by sorry

end work_completion_l3017_301736


namespace impossible_to_use_all_parts_l3017_301743

theorem impossible_to_use_all_parts (p q r : ℕ) : 
  ¬∃ (x y z : ℕ), (2 * x + 2 * z = 2 * p + 2 * r + 2) ∧ 
                   (2 * x + y = 2 * p + q + 1) ∧ 
                   (y + z = q + r) :=
sorry

end impossible_to_use_all_parts_l3017_301743


namespace tv_price_change_l3017_301756

theorem tv_price_change (P : ℝ) : 
  P > 0 → (P * 0.9) * 1.3 = P * 1.17 := by
sorry

end tv_price_change_l3017_301756


namespace absolute_value_equality_l3017_301704

theorem absolute_value_equality (y : ℝ) : |y + 2| = |y - 3| → y = 1/2 := by
  sorry

end absolute_value_equality_l3017_301704


namespace inequality_proof_l3017_301735

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a^3 / (b^2 - 1)) + (b^3 / (c^2 - 1)) + (c^3 / (a^2 - 1)) ≥ (9 * Real.sqrt 3) / 2 := by
  sorry

end inequality_proof_l3017_301735


namespace initial_fuel_calculation_l3017_301747

/-- Calculates the initial amount of fuel in a car's tank given its fuel consumption rate,
    journey distance, and remaining fuel after the journey. -/
theorem initial_fuel_calculation (consumption_rate : ℝ) (journey_distance : ℝ) (fuel_left : ℝ) :
  consumption_rate = 12 →
  journey_distance = 275 →
  fuel_left = 14 →
  (consumption_rate / 100) * journey_distance + fuel_left = 47 := by
  sorry

#check initial_fuel_calculation

end initial_fuel_calculation_l3017_301747


namespace hike_remaining_distance_l3017_301763

/-- Calculates the remaining distance of a hike given the total distance and distance already hiked. -/
def remaining_distance (total : ℕ) (hiked : ℕ) : ℕ :=
  total - hiked

/-- Proves that for a 36-mile hike with 9 miles already hiked, 27 miles remain. -/
theorem hike_remaining_distance :
  remaining_distance 36 9 = 27 := by
  sorry

end hike_remaining_distance_l3017_301763


namespace combined_salaries_l3017_301751

theorem combined_salaries (salary_A : ℕ) (num_people : ℕ) (avg_salary : ℕ) : 
  salary_A = 8000 → 
  num_people = 5 → 
  avg_salary = 9000 → 
  (avg_salary * num_people - salary_A = 37000) := by
  sorry

end combined_salaries_l3017_301751


namespace equation_solution_l3017_301782

theorem equation_solution : ∃ (x : ℝ), 
  x > 0 ∧ 
  (1/4) * (5*x^2 - 4) = (x^2 - 40*x - 5) * (x^2 + 20*x + 2) ∧
  x = 20 + 10 * Real.sqrt 41 := by
  sorry

end equation_solution_l3017_301782


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l3017_301715

-- Define the first four prime numbers
def first_four_primes : List ℕ := [2, 3, 5, 7]

-- Define a function to calculate the reciprocal of a natural number
def reciprocal (n : ℕ) : ℚ := 1 / n

-- Define the arithmetic mean of a list of rational numbers
def arithmetic_mean (list : List ℚ) : ℚ := (list.sum) / list.length

-- Theorem statement
theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean (first_four_primes.map reciprocal) = 247 / 840 := by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l3017_301715


namespace floor_abs_plus_const_l3017_301764

theorem floor_abs_plus_const : 
  ⌊|(-47.3 : ℝ)| + 0.7⌋ = 48 := by
  sorry

end floor_abs_plus_const_l3017_301764


namespace solution_range_l3017_301728

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x - 1 ≥ a^2 ∧ x - 4 < 2*a) → 
  a ∈ Set.Ioo (-1 : ℝ) 3 :=
by sorry

end solution_range_l3017_301728


namespace rectangle_area_diagonal_relation_l3017_301714

theorem rectangle_area_diagonal_relation (l w d : ℝ) (h1 : l / w = 4 / 3) (h2 : l^2 + w^2 = d^2) :
  ∃ k : ℝ, l * w = k * d^2 ∧ k = 12 / 25 := by
  sorry

end rectangle_area_diagonal_relation_l3017_301714


namespace num_quadrilaterals_equals_choose_12_4_l3017_301765

/-- The number of ways to choose 4 items from 12 items -/
def choose_12_4 : ℕ := 495

/-- The number of distinct points on the circle -/
def num_points : ℕ := 12

/-- The number of vertices in a quadrilateral -/
def vertices_per_quadrilateral : ℕ := 4

/-- Theorem: The number of different convex quadrilaterals formed by selecting 4 vertices 
    from 12 distinct points on the circumference of a circle is equal to choose_12_4 -/
theorem num_quadrilaterals_equals_choose_12_4 : 
  choose_12_4 = Nat.choose num_points vertices_per_quadrilateral := by
  sorry

#eval choose_12_4  -- This should output 495
#eval Nat.choose num_points vertices_per_quadrilateral  -- This should also output 495

end num_quadrilaterals_equals_choose_12_4_l3017_301765


namespace number_of_parents_at_park_parents_at_park_l3017_301753

/-- Given a group of people at a park, prove the number of parents. -/
theorem number_of_parents_at_park (num_girls : ℕ) (num_boys : ℕ) (num_groups : ℕ) (group_size : ℕ) : ℕ :=
  let total_people := num_groups * group_size
  let total_children := num_girls + num_boys
  total_people - total_children

/-- Prove that there are 50 parents at the park given the specified conditions. -/
theorem parents_at_park : number_of_parents_at_park 14 11 3 25 = 50 := by
  sorry

end number_of_parents_at_park_parents_at_park_l3017_301753


namespace margaret_mean_score_l3017_301794

def scores : List ℕ := [85, 88, 90, 92, 94, 96, 100]

def cyprian_score_count : ℕ := 4
def margaret_score_count : ℕ := 3
def cyprian_mean : ℚ := 92

theorem margaret_mean_score (h1 : scores.length = cyprian_score_count + margaret_score_count)
  (h2 : cyprian_mean = (scores.sum - (scores.sum - cyprian_mean * cyprian_score_count)) / cyprian_score_count) :
  (scores.sum - cyprian_mean * cyprian_score_count) / margaret_score_count = 92.33 := by
  sorry

end margaret_mean_score_l3017_301794


namespace smallest_part_of_proportional_division_l3017_301721

theorem smallest_part_of_proportional_division (total : ℝ) (a b c d : ℝ) 
  (h_total : total = 80)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_prop : b = 3 * a ∧ c = 5 * a ∧ d = 7 * a)
  (h_sum : a + b + c + d = total) :
  a = 5 := by
sorry

end smallest_part_of_proportional_division_l3017_301721


namespace snow_probability_l3017_301700

theorem snow_probability (p1 p2 p3 : ℚ) (n1 n2 n3 : ℕ) : 
  p1 = 1/3 →
  p2 = 1/4 →
  p3 = 1/2 →
  n1 = 3 →
  n2 = 4 →
  n3 = 3 →
  1 - (1 - p1)^n1 * (1 - p2)^n2 * (1 - p3)^n3 = 2277/2304 := by
sorry

end snow_probability_l3017_301700


namespace gravel_path_cost_l3017_301720

/-- Calculates the cost of gravelling a path inside a rectangular plot. -/
theorem gravel_path_cost
  (plot_length : ℝ)
  (plot_width : ℝ)
  (path_width : ℝ)
  (cost_per_sqm : ℝ)
  (h1 : plot_length = 110)
  (h2 : plot_width = 65)
  (h3 : path_width = 2.5)
  (h4 : cost_per_sqm = 0.70) :
  let total_area := plot_length * plot_width
  let inner_length := plot_length - 2 * path_width
  let inner_width := plot_width - 2 * path_width
  let inner_area := inner_length * inner_width
  let path_area := total_area - inner_area
  path_area * cost_per_sqm = 595 :=
by sorry

end gravel_path_cost_l3017_301720


namespace sum_of_coordinates_A_l3017_301733

/-- Given points A, B, and C in a 2D plane satisfying specific conditions, 
    prove that the sum of the coordinates of A is 24. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (dist A C / dist A B = 1/3) →
  (dist B C / dist A B = 1/3) →
  B = (2, 6) →
  C = (4, 12) →
  A.1 + A.2 = 24 := by
  sorry

#check sum_of_coordinates_A

end sum_of_coordinates_A_l3017_301733


namespace slipper_price_calculation_l3017_301726

/-- Given a pair of slippers with original price P, prove that with a 10% discount,
    $5.50 embroidery cost per shoe, $10.00 shipping, and $66.00 total cost,
    the original price P must be $50.00. -/
theorem slipper_price_calculation (P : ℝ) : 
  (0.90 * P + 2 * 5.50 + 10.00 = 66.00) → P = 50.00 := by
  sorry

end slipper_price_calculation_l3017_301726


namespace club_membership_theorem_l3017_301752

theorem club_membership_theorem :
  ∃ n : ℕ, n ≥ 300 ∧ n % 8 = 0 ∧ n % 9 = 0 ∧ n % 11 = 0 ∧
  ∀ m : ℕ, m ≥ 300 ∧ m % 8 = 0 ∧ m % 9 = 0 ∧ m % 11 = 0 → m ≥ n :=
by
  use 792
  sorry

end club_membership_theorem_l3017_301752


namespace permutation_combination_equality_l3017_301774

theorem permutation_combination_equality (n : ℕ) : 
  (n * (n - 1) = (n + 1) * n / 2) → n! = 6 := by
  sorry

end permutation_combination_equality_l3017_301774


namespace derivative_of_f_l3017_301709

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2

-- State the theorem
theorem derivative_of_f :
  deriv f = fun x ↦ 2 * x + 2 := by sorry

end derivative_of_f_l3017_301709


namespace parallel_vectors_m_value_l3017_301755

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, -6)
  let b : ℝ × ℝ := (-1, m)
  are_parallel a b → m = 3 := by
  sorry

end parallel_vectors_m_value_l3017_301755


namespace equation_solution_l3017_301757

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => (5*x^2 + 70*x + 2) / (3*x + 28) - (4*x + 2)
  let sol1 : ℝ := (-48 + 28*Real.sqrt 22) / 14
  let sol2 : ℝ := (-48 - 28*Real.sqrt 22) / 14
  f sol1 = 0 ∧ f sol2 = 0 := by
  sorry

end equation_solution_l3017_301757


namespace exactly_one_and_two_white_mutually_exclusive_not_contradictory_l3017_301767

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The set of all possible outcomes when drawing two balls from the bag -/
def allOutcomes : Finset DrawOutcome := sorry

/-- The event of drawing exactly one white ball -/
def exactlyOneWhite (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.White ∧ outcome.second = BallColor.Red) ∨
  (outcome.first = BallColor.Red ∧ outcome.second = BallColor.White)

/-- The event of drawing exactly two white balls -/
def exactlyTwoWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∧ outcome.second = BallColor.White

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (event1 event2 : DrawOutcome → Prop) : Prop :=
  ∀ outcome, ¬(event1 outcome ∧ event2 outcome)

/-- Two events are contradictory if one of them must occur -/
def contradictory (event1 event2 : DrawOutcome → Prop) : Prop :=
  ∀ outcome, event1 outcome ∨ event2 outcome

theorem exactly_one_and_two_white_mutually_exclusive_not_contradictory :
  mutuallyExclusive exactlyOneWhite exactlyTwoWhite ∧
  ¬contradictory exactlyOneWhite exactlyTwoWhite :=
sorry

end exactly_one_and_two_white_mutually_exclusive_not_contradictory_l3017_301767


namespace masha_ate_ten_pies_l3017_301719

/-- Represents the eating rates of Masha and the bear -/
structure EatingRates where
  masha : ℝ
  bear : ℝ
  bear_faster : bear = 3 * masha

/-- Represents the distribution of food between Masha and the bear -/
structure FoodDistribution where
  total_pies : ℕ
  total_pies_positive : total_pies > 0
  masha_pies : ℕ
  bear_pies : ℕ
  pies_sum : masha_pies + bear_pies = total_pies
  equal_raspberries : ℝ  -- Represents the fact that they ate equal raspberries

/-- Theorem stating that Masha ate 10 pies given the problem conditions -/
theorem masha_ate_ten_pies (rates : EatingRates) (food : FoodDistribution) 
  (h_total_pies : food.total_pies = 40) :
  food.masha_pies = 10 := by
  sorry


end masha_ate_ten_pies_l3017_301719


namespace paula_tickets_l3017_301773

/-- The number of times Paula wants to ride the go-karts -/
def go_kart_rides : ℕ := 1

/-- The number of times Paula wants to ride the bumper cars -/
def bumper_car_rides : ℕ := 4

/-- The number of tickets required for one go-kart ride -/
def go_kart_tickets : ℕ := 4

/-- The number of tickets required for one bumper car ride -/
def bumper_car_tickets : ℕ := 5

/-- The total number of tickets Paula needs -/
def total_tickets : ℕ := go_kart_rides * go_kart_tickets + bumper_car_rides * bumper_car_tickets

theorem paula_tickets : total_tickets = 24 := by
  sorry

end paula_tickets_l3017_301773


namespace third_player_wins_probability_l3017_301775

/-- Represents a game where players take turns tossing a fair six-sided die. -/
structure DieTossingGame where
  num_players : ℕ
  target_player : ℕ
  prob_six : ℚ

/-- The probability that the target player is the first to toss a six. -/
noncomputable def probability_target_wins (game : DieTossingGame) : ℚ :=
  sorry

/-- Theorem stating the probability of the third player being the first to toss a six
    in a four-player game. -/
theorem third_player_wins_probability :
  let game := DieTossingGame.mk 4 3 (1/6)
  probability_target_wins game = 125/671 := by
  sorry

end third_player_wins_probability_l3017_301775


namespace game_size_proof_l3017_301760

/-- Given a game download scenario where:
  * 310 MB has already been downloaded
  * The remaining download speed is 3 MB/minute
  * It takes 190 more minutes to finish the download
  Prove that the total size of the game is 880 MB -/
theorem game_size_proof (already_downloaded : ℕ) (download_speed : ℕ) (remaining_time : ℕ) :
  already_downloaded = 310 →
  download_speed = 3 →
  remaining_time = 190 →
  already_downloaded + download_speed * remaining_time = 880 :=
by sorry

end game_size_proof_l3017_301760


namespace cover_rectangles_l3017_301769

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a circle with a center point and radius -/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- Returns the number of circles needed to cover a rectangle -/
def circles_to_cover (r : Rectangle) (circle_radius : ℝ) : ℕ :=
  sorry

theorem cover_rectangles :
  let r1 := Rectangle.mk 6 3
  let r2 := Rectangle.mk 5 3
  let circle_radius := Real.sqrt 2
  (circles_to_cover r1 circle_radius = 6) ∧
  (circles_to_cover r2 circle_radius = 5) := by
  sorry

end cover_rectangles_l3017_301769


namespace ellipse_m_values_l3017_301703

def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / 12 + y^2 / m = 1

def eccentricity (e : ℝ) : Prop :=
  e = 1/2

theorem ellipse_m_values (m : ℝ) :
  (∃ x y, ellipse_equation x y m) ∧ (∃ e, eccentricity e) →
  m = 9 ∨ m = 16 := by sorry

end ellipse_m_values_l3017_301703


namespace rectangle_area_ratio_l3017_301786

theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 3 / 4) (h2 : b / d = 3 / 4) :
  (a * b) / (c * d) = 9 / 16 := by
  sorry

end rectangle_area_ratio_l3017_301786


namespace congruence_and_range_implies_value_l3017_301792

theorem congruence_and_range_implies_value :
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -1234 [ZMOD 8] → n = 6 := by
  sorry

end congruence_and_range_implies_value_l3017_301792


namespace five_player_four_stage_tournament_outcomes_l3017_301788

/-- Represents a tournament with a fixed number of players and stages. -/
structure Tournament :=
  (num_players : ℕ)
  (num_stages : ℕ)

/-- Calculates the number of possible outcomes in a tournament. -/
def tournament_outcomes (t : Tournament) : ℕ :=
  2^t.num_stages

/-- Theorem stating that a tournament with 5 players and 4 stages has 16 possible outcomes. -/
theorem five_player_four_stage_tournament_outcomes :
  ∀ t : Tournament, t.num_players = 5 → t.num_stages = 4 →
  tournament_outcomes t = 16 :=
by sorry

end five_player_four_stage_tournament_outcomes_l3017_301788


namespace intersection_when_a_zero_union_equals_A_l3017_301741

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 5*x - 6 < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a - 1 ≤ x ∧ x < a + 5}

-- Theorem 1: A ∩ B when a = 0
theorem intersection_when_a_zero : A ∩ B 0 = {x : ℝ | -1 < x ∧ x < 5} := by sorry

-- Theorem 2: Range of values for a when A ∪ B = A
theorem union_equals_A (a : ℝ) : A ∪ B a = A ↔ a ∈ Set.Ioo 0 1 ∪ Set.Ici 6 := by sorry

end intersection_when_a_zero_union_equals_A_l3017_301741


namespace intersection_area_is_525_l3017_301761

/-- A cube with edge length 30 units -/
def Cube : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 30}

/-- Point A of the cube -/
def A : Fin 3 → ℝ := λ _ ↦ 0

/-- Point B of the cube -/
def B : Fin 3 → ℝ := λ i ↦ if i = 0 then 30 else 0

/-- Point C of the cube -/
def C : Fin 3 → ℝ := λ i ↦ if i = 2 then 30 else B i

/-- Point D of the cube -/
def D : Fin 3 → ℝ := λ _ ↦ 30

/-- Point P on edge AB -/
def P : Fin 3 → ℝ := λ i ↦ if i = 0 then 10 else 0

/-- Point Q on edge BC -/
def Q : Fin 3 → ℝ := λ i ↦ if i = 0 then 30 else if i = 2 then 20 else 0

/-- Point R on edge CD -/
def R : Fin 3 → ℝ := λ i ↦ if i = 1 then 15 else 30

/-- The plane PQR -/
def PlanePQR : Set (Fin 3 → ℝ) :=
  {x | 3 * x 0 + 2 * x 1 - 3 * x 2 = 30}

/-- The intersection of the cube and the plane PQR -/
def Intersection : Set (Fin 3 → ℝ) :=
  Cube ∩ PlanePQR

/-- The area of the intersection -/
noncomputable def IntersectionArea : ℝ := sorry

theorem intersection_area_is_525 :
  IntersectionArea = 525 := by sorry

end intersection_area_is_525_l3017_301761


namespace greatest_power_of_three_l3017_301790

def p : ℕ := (List.range 34).foldl (· * ·) 1

theorem greatest_power_of_three (k : ℕ) : k ≤ 15 ↔ (3^k : ℕ) ∣ p := by sorry

end greatest_power_of_three_l3017_301790


namespace inequality_proof_l3017_301776

theorem inequality_proof (a : ℝ) : (a^2 + a + 2) / Real.sqrt (a^2 + a + 1) ≥ 2 := by
  sorry

end inequality_proof_l3017_301776


namespace rectangular_plot_area_breadth_ratio_l3017_301791

theorem rectangular_plot_area_breadth_ratio :
  let breadth : ℕ := 13
  let length : ℕ := breadth + 10
  let area : ℕ := length * breadth
  area / breadth = 23 := by
sorry

end rectangular_plot_area_breadth_ratio_l3017_301791


namespace nested_fraction_evaluation_l3017_301754

theorem nested_fraction_evaluation :
  1 + 1 / (2 + 1 / (3 + 1 / (3 + 3))) = 63 / 44 := by
  sorry

end nested_fraction_evaluation_l3017_301754


namespace beth_initial_coins_l3017_301744

theorem beth_initial_coins (initial_coins : ℕ) : 
  (initial_coins + 35) / 2 = 80 → initial_coins = 125 := by
  sorry

end beth_initial_coins_l3017_301744


namespace square_area_from_diagonal_l3017_301749

theorem square_area_from_diagonal (d : ℝ) (h : d = 16 * Real.sqrt 2) : 
  (d / Real.sqrt 2) ^ 2 = 256 := by
  sorry

end square_area_from_diagonal_l3017_301749


namespace not_divisible_by_seven_l3017_301702

theorem not_divisible_by_seven (k : ℕ) : ¬(7 ∣ (2^(2*k - 1) + 2^k + 1)) := by
  sorry

end not_divisible_by_seven_l3017_301702


namespace t_shirt_cost_l3017_301781

/-- The cost of one T-shirt -/
def T : ℝ := sorry

/-- The cost of one pair of pants -/
def pants_cost : ℝ := 80

/-- The cost of one pair of shoes -/
def shoes_cost : ℝ := 150

/-- The discount rate applied to all items -/
def discount_rate : ℝ := 0.9

/-- The total cost Eugene pays after discount -/
def total_cost : ℝ := 558

theorem t_shirt_cost : T = 20 := by
  have h1 : total_cost = discount_rate * (4 * T + 3 * pants_cost + 2 * shoes_cost) := by sorry
  sorry

end t_shirt_cost_l3017_301781


namespace trevors_future_age_l3017_301750

/-- Proves Trevor's age when his older brother is three times Trevor's current age -/
theorem trevors_future_age (t b : ℕ) (h1 : t = 11) (h2 : b = 20) :
  ∃ x : ℕ, b + (x - t) = 3 * t ∧ x = 24 := by
  sorry

end trevors_future_age_l3017_301750


namespace coin_flip_probability_l3017_301758

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the set of 5 coins -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (halfDollar : CoinOutcome)

/-- The total number of possible outcomes when flipping 5 coins -/
def totalOutcomes : Nat := 32

/-- Predicate for successful outcomes (penny, dime, and half-dollar are heads) -/
def isSuccessfulOutcome (cs : CoinSet) : Prop :=
  cs.penny = CoinOutcome.Heads ∧ cs.dime = CoinOutcome.Heads ∧ cs.halfDollar = CoinOutcome.Heads

/-- The number of successful outcomes -/
def successfulOutcomes : Nat := 4

/-- The probability of getting heads on penny, dime, and half-dollar -/
def probability : Rat := 1 / 8

theorem coin_flip_probability :
  (successfulOutcomes : Rat) / totalOutcomes = probability :=
sorry

end coin_flip_probability_l3017_301758


namespace initial_population_is_4144_l3017_301734

/-- Represents the population changes in a village --/
def village_population (initial : ℕ) : ℕ :=
  let after_bombardment := initial * 90 / 100
  let after_departure := after_bombardment * 85 / 100
  let after_refugees := after_departure + 50
  let after_births := after_refugees * 105 / 100
  let after_employment := after_births * 92 / 100
  after_employment + 100

/-- Theorem stating that the initial population of 4144 results in a final population of 3213 --/
theorem initial_population_is_4144 : village_population 4144 = 3213 := by
  sorry

end initial_population_is_4144_l3017_301734


namespace root_twice_other_iff_a_equals_four_l3017_301710

theorem root_twice_other_iff_a_equals_four (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 - (2*a + 1)*x + a^2 + 2 = 0 ∧ 
    y^2 - (2*a + 1)*y + a^2 + 2 = 0 ∧ 
    y = 2*x) ↔ 
  a = 4 := by
sorry

end root_twice_other_iff_a_equals_four_l3017_301710


namespace intersection_of_M_and_N_l3017_301716

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end intersection_of_M_and_N_l3017_301716


namespace minimize_distance_to_point_l3017_301779

/-- Given points P(-2, -2) and R(2, m), prove that the value of m that minimizes 
    the distance PR is -2. -/
theorem minimize_distance_to_point (m : ℝ) : 
  let P : ℝ × ℝ := (-2, -2)
  let R : ℝ × ℝ := (2, m)
  let distance := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  (∀ k : ℝ, distance ≤ Real.sqrt ((P.1 - 2)^2 + (P.2 - k)^2)) → m = -2 :=
by sorry

end minimize_distance_to_point_l3017_301779


namespace fiona_finished_tenth_l3017_301701

/-- Represents a racer in the competition -/
inductive Racer
| Alice
| Ben
| Carlos
| Diana
| Emma
| Fiona

/-- The type of finishing positions -/
def Position := Fin 15

/-- The finishing order of the race -/
def FinishingOrder := Racer → Position

/-- Defines the relative positions of racers -/
def PlacesAhead (fo : FinishingOrder) (r1 r2 : Racer) (n : ℕ) : Prop :=
  (fo r1).val + n = (fo r2).val

/-- Defines the absolute position of a racer -/
def FinishedIn (fo : FinishingOrder) (r : Racer) (p : Position) : Prop :=
  fo r = p

theorem fiona_finished_tenth (fo : FinishingOrder) :
  PlacesAhead fo Racer.Emma Racer.Diana 4 →
  PlacesAhead fo Racer.Carlos Racer.Alice 2 →
  PlacesAhead fo Racer.Diana Racer.Ben 3 →
  PlacesAhead fo Racer.Carlos Racer.Fiona 3 →
  PlacesAhead fo Racer.Emma Racer.Fiona 2 →
  FinishedIn fo Racer.Ben ⟨7, by norm_num⟩ →
  FinishedIn fo Racer.Fiona ⟨10, by norm_num⟩ := by
  sorry

end fiona_finished_tenth_l3017_301701


namespace expression_evaluation_l3017_301759

theorem expression_evaluation :
  let a : ℤ := (-2)^2
  5 * a^2 - (a^2 - (2*a - 5*a^2) - 2*(a^2 - 3*a)) = 32 := by sorry

end expression_evaluation_l3017_301759


namespace triangle_properties_l3017_301738

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  (t.a + t.c) / (t.a + t.b) = (t.b - t.a) / t.c ∧
  t.b = Real.sqrt 14 ∧
  Real.sin t.A = 2 * Real.sin t.C

-- State the theorem
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.B = 2 * Real.pi / 3 ∧ min t.a (min t.b t.c) = Real.sqrt 2 := by
  sorry

end triangle_properties_l3017_301738


namespace sum_of_cubes_and_reciprocals_l3017_301799

/-- Given real numbers x and y satisfying x + y = 6 and x * y = 5,
    prove that x + (x^3 / y^2) + (y^3 / x^2) + y = 137.04 -/
theorem sum_of_cubes_and_reciprocals (x y : ℝ) 
  (h1 : x + y = 6) (h2 : x * y = 5) : 
  x + (x^3 / y^2) + (y^3 / x^2) + y = 137.04 := by
  sorry

end sum_of_cubes_and_reciprocals_l3017_301799


namespace f_properties_l3017_301797

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + (1 - x) / (1 + x)

theorem f_properties (a : ℝ) (h_a : a > 0) :
  -- 1. If f'(1) = 0, then a = 1
  (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 → x = 1 → a = 1) ∧
  -- 2. For a ≥ 2, f'(x) > 0 for all x > 0
  (a ≥ 2 → ∀ x : ℝ, x > 0 → (deriv (f a)) x > 0) ∧
  -- 3. For 0 < a < 2, f'(x) < 0 for 0 < x < sqrt((2-a)/a) and f'(x) > 0 for x > sqrt((2-a)/a)
  (0 < a ∧ a < 2 → 
    (∀ x : ℝ, 0 < x ∧ x < Real.sqrt ((2 - a) / a) → (deriv (f a)) x < 0) ∧
    (∀ x : ℝ, x > Real.sqrt ((2 - a) / a) → (deriv (f a)) x > 0)) ∧
  -- 4. The minimum value of f(x) is 1 if and only if a ≥ 2
  (∃ x : ℝ, x ≥ 0 ∧ ∀ y : ℝ, y ≥ 0 → f a x ≤ f a y ∧ f a x = 1) ↔ a ≥ 2 :=
sorry

end f_properties_l3017_301797


namespace even_function_property_l3017_301712

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : is_even (λ x ↦ f (x + 2))) 
  (h3 : f 1 = π / 3) : 
  f 3 + f (-3) = 2 * π / 3 := by
  sorry

end even_function_property_l3017_301712


namespace exists_n_sigma_gt_3n_forall_k_exists_n_sigma_gt_kn_l3017_301727

-- Define the sum of divisors function
def sigma (n : ℕ) : ℕ := sorry

-- Theorem 1: There exists a positive integer n such that σ(n) > 3n
theorem exists_n_sigma_gt_3n : ∃ n : ℕ, n > 0 ∧ sigma n > 3 * n := by sorry

-- Theorem 2: For any real number k > 1, there exists a positive integer n such that σ(n) > kn
theorem forall_k_exists_n_sigma_gt_kn : ∀ k : ℝ, k > 1 → ∃ n : ℕ, n > 0 ∧ (sigma n : ℝ) > k * n := by sorry

end exists_n_sigma_gt_3n_forall_k_exists_n_sigma_gt_kn_l3017_301727


namespace log_one_half_of_one_eighth_l3017_301725

theorem log_one_half_of_one_eighth (a : ℝ) : a = Real.log 0.125 / Real.log (1/2) → a = 3 := by
  sorry

end log_one_half_of_one_eighth_l3017_301725


namespace john_weekly_earnings_l3017_301737

/-- Calculates John's weekly earnings from crab fishing -/
def weekly_earnings (small_baskets medium_baskets large_baskets jumbo_baskets : ℕ)
  (small_per_basket medium_per_basket large_per_basket jumbo_per_basket : ℕ)
  (small_price medium_price large_price jumbo_price : ℕ) : ℕ :=
  (small_baskets * small_per_basket * small_price) +
  (medium_baskets * medium_per_basket * medium_price) +
  (large_baskets * large_per_basket * large_price) +
  (jumbo_baskets * jumbo_per_basket * jumbo_price)

theorem john_weekly_earnings :
  weekly_earnings 3 2 4 1 4 3 5 2 3 4 5 7 = 174 := by
  sorry

end john_weekly_earnings_l3017_301737


namespace tangent_line_max_difference_l3017_301740

theorem tangent_line_max_difference (m n : ℝ) :
  ((m + 1)^2 + (n + 1)^2 = 4) →  -- Condition for tangent line
  (∀ x y : ℝ, (m + 1) * x + (n + 1) * y = 2 → x^2 + y^2 ≤ 1) →  -- Line touches or is outside the circle
  (∃ x y : ℝ, (m + 1) * x + (n + 1) * y = 2 ∧ x^2 + y^2 = 1) →  -- Line touches the circle at least at one point
  (m - n ≤ 2 * Real.sqrt 2) ∧ (∃ m₀ n₀ : ℝ, m₀ - n₀ = 2 * Real.sqrt 2 ∧ 
    ((m₀ + 1)^2 + (n₀ + 1)^2 = 4) ∧
    (∀ x y : ℝ, (m₀ + 1) * x + (n₀ + 1) * y = 2 → x^2 + y^2 ≤ 1) ∧
    (∃ x y : ℝ, (m₀ + 1) * x + (n₀ + 1) * y = 2 ∧ x^2 + y^2 = 1)) :=
by sorry


end tangent_line_max_difference_l3017_301740


namespace tangent_length_correct_l3017_301711

/-- Two circles S₁ and S₂ touching at point A with radii R and r respectively (R > r).
    B is a point on S₁ such that AB = a. -/
structure TangentCircles where
  R : ℝ
  r : ℝ
  a : ℝ
  h₁ : R > r
  h₂ : R > 0
  h₃ : r > 0
  h₄ : a > 0

/-- The length of the tangent from B to S₂ -/
noncomputable def tangentLength (c : TangentCircles) (external : Bool) : ℝ :=
  if external then
    c.a * Real.sqrt ((c.R + c.r) / c.R)
  else
    c.a * Real.sqrt ((c.R - c.r) / c.R)

theorem tangent_length_correct (c : TangentCircles) :
  (∀ external, tangentLength c external = 
    if external then c.a * Real.sqrt ((c.R + c.r) / c.R)
    else c.a * Real.sqrt ((c.R - c.r) / c.R)) :=
by sorry

end tangent_length_correct_l3017_301711


namespace smallest_ccd_value_l3017_301798

theorem smallest_ccd_value (C D : ℕ) : 
  (1 ≤ C ∧ C ≤ 9) →
  (1 ≤ D ∧ D ≤ 9) →
  C ≠ D →
  (10 * C + D : ℕ) < 100 →
  (100 * C + 10 * C + D : ℕ) < 1000 →
  (10 * C + D : ℕ) = (100 * C + 10 * C + D : ℕ) / 7 →
  (∀ (C' D' : ℕ), 
    (1 ≤ C' ∧ C' ≤ 9) →
    (1 ≤ D' ∧ D' ≤ 9) →
    C' ≠ D' →
    (10 * C' + D' : ℕ) < 100 →
    (100 * C' + 10 * C' + D' : ℕ) < 1000 →
    (10 * C' + D' : ℕ) = (100 * C' + 10 * C' + D' : ℕ) / 7 →
    (100 * C + 10 * C + D : ℕ) ≤ (100 * C' + 10 * C' + D' : ℕ)) →
  (100 * C + 10 * C + D : ℕ) = 115 :=
by sorry

end smallest_ccd_value_l3017_301798


namespace loss_percentage_calculation_l3017_301708

def cost_price : ℝ := 120
def selling_price : ℝ := 102
def gain_price : ℝ := 144
def gain_percentage : ℝ := 20

theorem loss_percentage_calculation :
  (cost_price - selling_price) / cost_price * 100 = 15 := by
  sorry

end loss_percentage_calculation_l3017_301708


namespace part1_part2_l3017_301717

-- Define the function y
def y (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x + a - 2

-- Part 1
theorem part1 : ∀ a : ℝ, (∀ x : ℝ, y a x ≥ -2) ↔ a ∈ Set.Ici (1/3) :=
sorry

-- Part 2
def solution_set (a : ℝ) : Set ℝ :=
  if a > 0 then { x | -1/a < x ∧ x < 1 }
  else if a = 0 then { x | x < 1 }
  else if -1 < a ∧ a < 0 then { x | x < 1 ∨ x > -1/a }
  else if a = -1 then { x | x ≠ 1 }
  else { x | x < -1/a ∨ x > 1 }

theorem part2 : ∀ a : ℝ, ∀ x : ℝ, x ∈ solution_set a ↔ a * x^2 + (1 - a) * x - 1 < 0 :=
sorry

end part1_part2_l3017_301717


namespace inequality_solution_range_l3017_301706

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) → (a > 3 ∨ a < 1) := by
sorry

end inequality_solution_range_l3017_301706


namespace sum_of_roots_quadratic_l3017_301789

theorem sum_of_roots_quadratic (x : ℝ) : 
  x^2 - 6*x + 8 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = 6 ∧ x^2 - 6*x + 8 = (x - r₁) * (x - r₂) :=
by sorry

end sum_of_roots_quadratic_l3017_301789


namespace projectile_height_time_l3017_301746

theorem projectile_height_time : ∃ t : ℝ, t > 0 ∧ -5*t^2 + 25*t = 30 ∧ ∀ s : ℝ, s > 0 ∧ -5*s^2 + 25*s = 30 → t ≤ s := by
  sorry

end projectile_height_time_l3017_301746


namespace questionnaires_15_16_l3017_301770

/-- Represents the number of questionnaires collected for each age group -/
structure QuestionnaireData where
  age_8_10 : ℕ
  age_11_12 : ℕ
  age_13_14 : ℕ
  age_15_16 : ℕ

/-- Represents the sampling data -/
structure SamplingData where
  total_sample : ℕ
  sample_11_12 : ℕ

/-- Theorem stating the number of questionnaires drawn from the 15-16 years old group -/
theorem questionnaires_15_16 (data : QuestionnaireData) (sampling : SamplingData) :
  data.age_8_10 = 120 →
  data.age_11_12 = 180 →
  data.age_13_14 = 240 →
  sampling.total_sample = 300 →
  sampling.sample_11_12 = 60 →
  (data.age_8_10 + data.age_11_12 + data.age_13_14 + data.age_15_16) * sampling.sample_11_12 = 
    sampling.total_sample * data.age_11_12 →
  (sampling.total_sample * data.age_15_16) / (data.age_8_10 + data.age_11_12 + data.age_13_14 + data.age_15_16) = 120 :=
by sorry

end questionnaires_15_16_l3017_301770


namespace black_area_after_three_cycles_l3017_301793

/-- Represents the fraction of black area remaining after a number of cycles. -/
def blackAreaFraction (cycles : ℕ) : ℚ :=
  (2 / 3) ^ cycles

/-- The number of cycles in the problem. -/
def numCycles : ℕ := 3

/-- Theorem stating that after three cycles, 8/27 of the original area remains black. -/
theorem black_area_after_three_cycles :
  blackAreaFraction numCycles = 8 / 27 := by
  sorry

end black_area_after_three_cycles_l3017_301793


namespace smallest_integer_satisfying_inequality_l3017_301732

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 12 → x ≥ 7 ∧ 7 < 3*7 - 12 :=
by sorry

end smallest_integer_satisfying_inequality_l3017_301732


namespace sum_of_odd_powers_l3017_301771

theorem sum_of_odd_powers (x y z : ℝ) (n : ℕ) (h1 : x + y + z = 1) 
  (h2 : Real.arctan x + Real.arctan y + Real.arctan z = π / 4) : 
  x^(2*n + 1) + y^(2*n + 1) + z^(2*n + 1) = 1 := by
  sorry

end sum_of_odd_powers_l3017_301771


namespace complex_combination_equality_l3017_301729

/-- Given complex numbers Q, E, D, and F, prove that their combination equals 1 + 117i -/
theorem complex_combination_equality (Q E D F : ℂ) 
  (hQ : Q = 7 + 3*I) 
  (hE : E = 2*I) 
  (hD : D = 7 - 3*I) 
  (hF : F = 1 + I) : 
  (Q * E * D) + F = 1 + 117*I := by
  sorry

end complex_combination_equality_l3017_301729


namespace constant_x_coordinate_l3017_301718

/-- Definition of the ellipse C -/
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Right focus F -/
def F : ℝ × ℝ := (1, 0)

/-- Left vertex A -/
def A : ℝ × ℝ := (-2, 0)

/-- Right vertex B -/
def B : ℝ × ℝ := (2, 0)

/-- Line l passing through F, not coincident with x-axis -/
def l (k : ℝ) (x y : ℝ) : Prop := y = k*(x - F.1) ∧ k ≠ 0

/-- Intersection points M and N of line l with ellipse C -/
def intersectionPoints (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | C p.1 p.2 ∧ l k p.1 p.2}

/-- Line AM -/
def lineAM (M : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - A.2) * (M.1 - A.1) = (x - A.1) * (M.2 - A.2)

/-- Line BN -/
def lineBN (N : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - B.2) * (N.1 - B.1) = (x - B.1) * (N.2 - B.2)

/-- Theorem: x-coordinate of intersection point T is constant -/
theorem constant_x_coordinate (k : ℝ) (M N : ℝ × ℝ) (h1 : M ∈ intersectionPoints k) (h2 : N ∈ intersectionPoints k) (h3 : M ≠ N) :
  ∃ (T : ℝ × ℝ), lineAM M T.1 T.2 ∧ lineBN N T.1 T.2 ∧ T.1 = 4 := by sorry

end constant_x_coordinate_l3017_301718


namespace cab_driver_average_income_l3017_301795

def income : List ℝ := [45, 50, 60, 65, 70]

theorem cab_driver_average_income :
  (income.sum / income.length : ℝ) = 58 := by
  sorry

end cab_driver_average_income_l3017_301795


namespace min_sum_of_four_primes_l3017_301707

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem min_sum_of_four_primes :
  ∀ a b c d s : ℕ,
  is_prime a → is_prime b → is_prime c → is_prime d → is_prime s →
  s = a + b + c + d →
  s ≥ 11 :=
sorry

end min_sum_of_four_primes_l3017_301707


namespace z_in_third_quadrant_l3017_301766

/-- The complex number z -/
def z : ℂ := (-8 + Complex.I) * Complex.I

/-- A complex number is in the third quadrant if its real part is negative and its imaginary part is negative -/
def is_in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

/-- Theorem: z is located in the third quadrant of the complex plane -/
theorem z_in_third_quadrant : is_in_third_quadrant z := by
  sorry

end z_in_third_quadrant_l3017_301766


namespace gcd_a_b_is_one_or_three_l3017_301748

def a (n : ℤ) : ℤ := n^5 + 6*n^3 + 8*n
def b (n : ℤ) : ℤ := n^4 + 4*n^2 + 3

theorem gcd_a_b_is_one_or_three (n : ℤ) : Nat.gcd (Int.natAbs (a n)) (Int.natAbs (b n)) = 1 ∨ Nat.gcd (Int.natAbs (a n)) (Int.natAbs (b n)) = 3 := by
  sorry

end gcd_a_b_is_one_or_three_l3017_301748


namespace polynomial_coefficient_product_l3017_301730

/-- Given a polynomial x^4 - (a-2)x^3 + 5x^2 + (b+3)x - 1 where the coefficients of x^3 and x are zero, prove that ab = -6 -/
theorem polynomial_coefficient_product (a b : ℝ) : 
  (a - 2 = 0) → (b + 3 = 0) → a * b = -6 := by
  sorry

end polynomial_coefficient_product_l3017_301730


namespace sum_reciprocals_bound_l3017_301787

theorem sum_reciprocals_bound (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  1 / (1 + a) + 1 / (1 + b) + 1 / (1 + c) + 1 / (1 + d) > 1 := by
sorry

end sum_reciprocals_bound_l3017_301787


namespace fraction_addition_l3017_301713

theorem fraction_addition (c : ℝ) : (5 + 5 * c) / 7 + 3 = (26 + 5 * c) / 7 := by
  sorry

end fraction_addition_l3017_301713


namespace coefficient_x_squared_expansion_l3017_301723

theorem coefficient_x_squared_expansion : 
  let p : Polynomial ℤ := (X + 1)^5 * (X - 2)
  p.coeff 2 = -15 := by sorry

end coefficient_x_squared_expansion_l3017_301723


namespace min_value_problem_l3017_301777

theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 9*a*b + 9*b^2 + 3*c^2 ≥ 60 ∧
  (a^2 + 9*a*b + 9*b^2 + 3*c^2 = 60 ↔ a = 6 ∧ b = 2 ∧ c = 3) :=
sorry

end min_value_problem_l3017_301777


namespace fraction_less_than_one_l3017_301742

theorem fraction_less_than_one (a b : ℝ) (h1 : a > b) (h2 : b > 0) : b / a < 1 := by
  sorry

end fraction_less_than_one_l3017_301742


namespace smallest_multiple_with_factors_l3017_301762

theorem smallest_multiple_with_factors : 
  ∀ n : ℕ+, 
    (936 * n : ℕ) % 2^5 = 0 ∧ 
    (936 * n : ℕ) % 3^3 = 0 ∧ 
    (936 * n : ℕ) % 11^2 = 0 → 
    n ≥ 4356 :=
by sorry

end smallest_multiple_with_factors_l3017_301762


namespace convex_curve_properties_l3017_301731

/-- Represents a convex curve in a 2D plane -/
structure ConvexCurve where
  -- Add necessary fields and properties for a convex curve
  -- This is a simplified representation

/-- Defines the reflection of a curve about a point -/
def reflect (K : ConvexCurve) (O : Point) : ConvexCurve :=
  sorry

/-- Defines the arithmetic mean of two curves -/
def arithmeticMean (K1 K2 : ConvexCurve) : ConvexCurve :=
  sorry

/-- Checks if a curve has a center of symmetry -/
def hasCenterOfSymmetry (K : ConvexCurve) : Prop :=
  sorry

/-- Calculates the diameter of a curve -/
def diameter (K : ConvexCurve) : ℝ :=
  sorry

/-- Calculates the width of a curve -/
def width (K : ConvexCurve) : ℝ :=
  sorry

/-- Calculates the length of a curve -/
def length (K : ConvexCurve) : ℝ :=
  sorry

/-- Calculates the area enclosed by a curve -/
def area (K : ConvexCurve) : ℝ :=
  sorry

theorem convex_curve_properties (K : ConvexCurve) (O : Point) :
  let K' := reflect K O
  let K_star := arithmeticMean K K'
  (hasCenterOfSymmetry K_star) ∧
  (diameter K_star = diameter K) ∧
  (width K_star = width K) ∧
  (length K_star = length K) ∧
  (area K_star ≥ area K) :=
by
  sorry

end convex_curve_properties_l3017_301731


namespace b_10_value_l3017_301724

theorem b_10_value (a b : ℕ → ℝ) 
  (h1 : ∀ n, (a n) * (a (n + 1)) = 2^n)
  (h2 : ∀ n, (a n) + (a (n + 1)) = b n)
  (h3 : a 1 = 1) :
  b 10 = 64 := by
sorry

end b_10_value_l3017_301724
