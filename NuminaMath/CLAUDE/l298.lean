import Mathlib

namespace NUMINAMATH_CALUDE_percent_of_a_is_4b_l298_29894

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.2 * b) :
  (4 * b) / a * 100 = 333.33 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_is_4b_l298_29894


namespace NUMINAMATH_CALUDE_cost_of_ingredients_l298_29892

-- Define the given values
def popcorn_sales_per_day : ℕ := 50
def cotton_candy_multiplier : ℕ := 3
def activity_duration : ℕ := 5
def rent : ℕ := 30
def final_earnings : ℕ := 895

-- Define the theorem
theorem cost_of_ingredients :
  let total_daily_sales := popcorn_sales_per_day + cotton_candy_multiplier * popcorn_sales_per_day
  let total_sales := total_daily_sales * activity_duration
  let earnings_after_rent := total_sales - rent
  earnings_after_rent - final_earnings = 75 := by
sorry

end NUMINAMATH_CALUDE_cost_of_ingredients_l298_29892


namespace NUMINAMATH_CALUDE_midsphere_radius_is_geometric_mean_l298_29857

/-- A regular tetrahedron with its associated spheres -/
structure RegularTetrahedron where
  /-- The radius of the insphere (inscribed sphere) -/
  r_in : ℝ
  /-- The radius of the circumsphere (circumscribed sphere) -/
  r_out : ℝ
  /-- The radius of the midsphere (edge-touching sphere) -/
  r_mid : ℝ
  /-- The radii are positive -/
  h_positive : r_in > 0 ∧ r_out > 0 ∧ r_mid > 0

/-- The radius of the midsphere is the geometric mean of the radii of the insphere and circumsphere -/
theorem midsphere_radius_is_geometric_mean (t : RegularTetrahedron) :
  t.r_mid ^ 2 = t.r_in * t.r_out := by
  sorry

end NUMINAMATH_CALUDE_midsphere_radius_is_geometric_mean_l298_29857


namespace NUMINAMATH_CALUDE_may_red_yarns_l298_29852

/-- The number of scarves May can knit using one yarn -/
def scarves_per_yarn : ℕ := 3

/-- The number of blue yarns May bought -/
def blue_yarns : ℕ := 6

/-- The number of yellow yarns May bought -/
def yellow_yarns : ℕ := 4

/-- The total number of scarves May will be able to make -/
def total_scarves : ℕ := 36

/-- The number of red yarns May bought -/
def red_yarns : ℕ := 2

theorem may_red_yarns : 
  scarves_per_yarn * (blue_yarns + yellow_yarns + red_yarns) = total_scarves := by
  sorry

end NUMINAMATH_CALUDE_may_red_yarns_l298_29852


namespace NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l298_29845

/-- Represents the possible positions on the circle -/
inductive Position : Type
| one : Position
| two : Position
| three : Position
| four : Position
| five : Position
| six : Position
| seven : Position

/-- Determines if a position is odd-numbered -/
def is_odd (p : Position) : Bool :=
  match p with
  | Position.one => true
  | Position.two => false
  | Position.three => true
  | Position.four => false
  | Position.five => true
  | Position.six => false
  | Position.seven => true

/-- Represents a single jump of the bug -/
def jump (p : Position) : Position :=
  match p with
  | Position.one => Position.three
  | Position.two => Position.five
  | Position.three => Position.five
  | Position.four => Position.seven
  | Position.five => Position.seven
  | Position.six => Position.two
  | Position.seven => Position.two

/-- Represents multiple jumps of the bug -/
def multi_jump (p : Position) (n : Nat) : Position :=
  match n with
  | 0 => p
  | n + 1 => jump (multi_jump p n)

/-- The main theorem to prove -/
theorem bug_position_after_2023_jumps :
  multi_jump Position.seven 2023 = Position.two := by
  sorry


end NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l298_29845


namespace NUMINAMATH_CALUDE_euro_problem_l298_29805

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- Theorem statement
theorem euro_problem (a : ℝ) :
  euro a (euro 4 5) = 640 → a = 8 := by
  sorry

end NUMINAMATH_CALUDE_euro_problem_l298_29805


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_l298_29841

theorem necessary_and_sufficient (p q : Prop) : 
  (p → q) → (q → p) → (p ↔ q) := by
  sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_l298_29841


namespace NUMINAMATH_CALUDE_total_steps_rachel_l298_29813

theorem total_steps_rachel (steps_up steps_down : ℕ) 
  (h1 : steps_up = 567) 
  (h2 : steps_down = 325) : 
  steps_up + steps_down = 892 := by
sorry

end NUMINAMATH_CALUDE_total_steps_rachel_l298_29813


namespace NUMINAMATH_CALUDE_mixture_ratio_change_l298_29853

def initial_ratio : ℚ := 3 / 2
def initial_total : ℚ := 20
def added_water : ℚ := 10

def milk : ℚ := initial_total * (initial_ratio / (1 + initial_ratio))
def water : ℚ := initial_total * (1 / (1 + initial_ratio))

def new_water : ℚ := water + added_water
def new_ratio : ℚ := milk / new_water

theorem mixture_ratio_change :
  new_ratio = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_mixture_ratio_change_l298_29853


namespace NUMINAMATH_CALUDE_min_clerks_problem_l298_29861

theorem min_clerks_problem : ∃ n : ℕ, n > 0 ∧ (Nat.choose n 4 = 3 * Nat.choose n 3) ∧ ∀ m : ℕ, m > 0 ∧ m < n → Nat.choose m 4 ≠ 3 * Nat.choose m 3 := by
  sorry

end NUMINAMATH_CALUDE_min_clerks_problem_l298_29861


namespace NUMINAMATH_CALUDE_drinks_preparation_l298_29880

/-- Given a number of pitchers and the capacity of each pitcher in glasses,
    calculate the total number of glasses that can be filled. -/
def total_glasses (num_pitchers : ℕ) (glasses_per_pitcher : ℕ) : ℕ :=
  num_pitchers * glasses_per_pitcher

/-- Theorem stating that 9 pitchers, each filling 6 glasses, results in 54 glasses total. -/
theorem drinks_preparation :
  total_glasses 9 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_drinks_preparation_l298_29880


namespace NUMINAMATH_CALUDE_subset_condition_l298_29839

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | -m < x ∧ x < m}

theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l298_29839


namespace NUMINAMATH_CALUDE_work_completion_days_l298_29802

/-- Calculates the initial number of days planned to complete a work given the total number of men,
    number of absent men, and the number of days taken by the remaining men. -/
def initialDays (totalMen : ℕ) (absentMen : ℕ) (daysWithAbsent : ℕ) : ℕ :=
  (totalMen - absentMen) * daysWithAbsent / totalMen

/-- Proves that given 20 men where 10 become absent and the remaining 10 complete the work in 40 days,
    the original plan was to complete the work in 20 days. -/
theorem work_completion_days :
  initialDays 20 10 40 = 20 := by
  sorry

#eval initialDays 20 10 40

end NUMINAMATH_CALUDE_work_completion_days_l298_29802


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_l298_29832

theorem gcf_lcm_sum (A B : ℕ) : 
  (A = Nat.gcd 9 (Nat.gcd 15 27)) →
  (B = Nat.lcm 9 (Nat.lcm 15 27)) →
  A + B = 138 := by
sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_l298_29832


namespace NUMINAMATH_CALUDE_star_three_neg_two_l298_29818

/-- Definition of the ☆ operation for rational numbers -/
def star (a b : ℚ) : ℚ := b^3 - abs (b - a)

/-- Theorem stating that 3☆(-2) = -13 -/
theorem star_three_neg_two : star 3 (-2) = -13 := by sorry

end NUMINAMATH_CALUDE_star_three_neg_two_l298_29818


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l298_29865

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (F P Q : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  F = (c, 0) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x < -a ∨ x > a)) →
  (∀ (x y : ℝ), x^2 + y^2 = b^2 / 4 → ((P.1 - x) * (F.1 - P.1) + (P.2 - y) * (F.2 - P.2) = 0)) →
  Q.1^2 + Q.2^2 = b^2 / 4 →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = ((P.1 - F.1)^2 + (P.2 - F.2)^2) / 4 →
  c^2 / a^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l298_29865


namespace NUMINAMATH_CALUDE_car_count_l298_29868

/-- The total number of cars in a rectangular arrangement -/
def total_cars (front_to_back : ℕ) (left_to_right : ℕ) : ℕ :=
  front_to_back * left_to_right

/-- Theorem stating the total number of cars given the position of red cars -/
theorem car_count (red_from_front red_from_left red_from_back red_from_right : ℕ) 
    (h1 : red_from_front + red_from_back = 25)
    (h2 : red_from_left + red_from_right = 35) :
    total_cars (red_from_front + red_from_back - 1) (red_from_left + red_from_right - 1) = 816 := by
  sorry

#eval total_cars 24 34  -- Should output 816

end NUMINAMATH_CALUDE_car_count_l298_29868


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l298_29873

/-- Given a circle C with equation x^2 + y^2 + y = 0, its center is (0, -1/2) and its radius is 1/2 -/
theorem circle_center_and_radius (x y : ℝ) :
  x^2 + y^2 + y = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (0, -1/2) ∧
    radius = 1/2 ∧
    ∀ (point : ℝ × ℝ), point.1^2 + point.2^2 + point.2 = 0 ↔
      (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l298_29873


namespace NUMINAMATH_CALUDE_segment_translation_l298_29859

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the translation function
def translate_left (p : Point) (units : ℝ) : Point :=
  (p.1 - units, p.2)

-- Define the problem statement
theorem segment_translation :
  let A : Point := (-1, 4)
  let B : Point := (-4, 1)
  let A₁ : Point := translate_left A 4
  let B₁ : Point := translate_left B 4
  A₁ = (-5, 4) ∧ B₁ = (-8, 1) := by sorry

end NUMINAMATH_CALUDE_segment_translation_l298_29859


namespace NUMINAMATH_CALUDE_first_phase_revenue_calculation_l298_29896

/-- Represents a two-phase sales scenario -/
structure SalesScenario where
  total_purchase : ℝ
  first_markup : ℝ
  second_markup : ℝ
  total_revenue_increase : ℝ

/-- Calculates the revenue from the first phase of sales -/
def first_phase_revenue (s : SalesScenario) : ℝ :=
  sorry

/-- Theorem stating the first phase revenue for the given scenario -/
theorem first_phase_revenue_calculation (s : SalesScenario) 
  (h1 : s.total_purchase = 180000)
  (h2 : s.first_markup = 0.25)
  (h3 : s.second_markup = 0.16)
  (h4 : s.total_revenue_increase = 0.20) :
  first_phase_revenue s = 100000 :=
sorry

end NUMINAMATH_CALUDE_first_phase_revenue_calculation_l298_29896


namespace NUMINAMATH_CALUDE_train_length_l298_29804

/-- The length of a train given its speed, bridge length, and time to pass the bridge -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (passing_time : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  bridge_length = 140 →
  passing_time = 40 →
  (train_speed * passing_time) - bridge_length = 360 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l298_29804


namespace NUMINAMATH_CALUDE_hawks_score_l298_29816

theorem hawks_score (total_points : ℕ) (first_day_margin : ℕ) (second_day_margin : ℕ)
  (h_total : total_points = 130)
  (h_first_margin : first_day_margin = 10)
  (h_second_margin : second_day_margin = 20)
  (h_equal_total : ∃ (eagles_total hawks_total : ℕ),
    eagles_total + hawks_total = total_points ∧ eagles_total = hawks_total) :
  ∃ (hawks_score : ℕ), hawks_score = 65 ∧ hawks_score * 2 = total_points :=
sorry

end NUMINAMATH_CALUDE_hawks_score_l298_29816


namespace NUMINAMATH_CALUDE_boys_girls_percentage_difference_l298_29872

theorem boys_girls_percentage_difference : ¬ (∀ (girls boys : ℝ), 
  boys = girls * (1 + 0.25) → girls = boys * (1 - 0.25)) := by
  sorry

end NUMINAMATH_CALUDE_boys_girls_percentage_difference_l298_29872


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l298_29884

theorem partial_fraction_decomposition_product (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 3 →
    (x^2 - 13) / ((x - 2) * (x + 2) * (x - 3)) =
    A / (x - 2) + B / (x + 2) + C / (x - 3)) →
  A * B * C = 81 / 100 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l298_29884


namespace NUMINAMATH_CALUDE_unique_m_for_inequality_l298_29881

/-- The approximate value of log_10(2) -/
def log10_2 : ℝ := 0.3010

/-- The theorem stating that 155 is the unique positive integer m satisfying the inequality -/
theorem unique_m_for_inequality : ∃! (m : ℕ), m > 0 ∧ (10 : ℝ)^(m - 1) < 2^512 ∧ 2^512 < 10^m :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_m_for_inequality_l298_29881


namespace NUMINAMATH_CALUDE_computer_price_ratio_l298_29824

theorem computer_price_ratio (x : ℝ) (h : 1.3 * x = 351) :
  (x + 1.3 * x) / x = 2.3 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_ratio_l298_29824


namespace NUMINAMATH_CALUDE_same_row_exists_l298_29850

/-- Represents a seating arrangement for a class session -/
def SeatingArrangement := Fin 50 → Fin 7

theorem same_row_exists (morning afternoon : SeatingArrangement) : 
  ∃ (s1 s2 : Fin 50), s1 ≠ s2 ∧ morning s1 = morning s2 ∧ afternoon s1 = afternoon s2 := by
  sorry

end NUMINAMATH_CALUDE_same_row_exists_l298_29850


namespace NUMINAMATH_CALUDE_alex_painting_time_l298_29823

/-- Given Jose's painting rate and the combined painting rate of Jose and Alex,
    calculate Alex's individual painting rate. -/
theorem alex_painting_time (jose_time : ℝ) (combined_time : ℝ) (alex_time : ℝ) : 
  jose_time = 7 → combined_time = 7 / 3 → alex_time = 7 / 2 := by
  sorry

#check alex_painting_time

end NUMINAMATH_CALUDE_alex_painting_time_l298_29823


namespace NUMINAMATH_CALUDE_race_result_l298_29863

-- Define the participants
inductive Participant
| Hare
| Fox
| Moose

-- Define the possible positions
inductive Position
| First
| Second

-- Define the statements made by the squirrels
def squirrel1_statement (winner : Participant) (second : Participant) : Prop :=
  winner = Participant.Hare ∧ second = Participant.Fox

def squirrel2_statement (winner : Participant) (second : Participant) : Prop :=
  winner = Participant.Moose ∧ second = Participant.Hare

-- Define the owl's statement
def owl_statement (s1 : Prop) (s2 : Prop) : Prop :=
  (s1 ∧ ¬s2) ∨ (¬s1 ∧ s2)

-- The main theorem
theorem race_result :
  ∃ (winner second : Participant),
    owl_statement (squirrel1_statement winner second) (squirrel2_statement winner second) →
    winner = Participant.Moose ∧ second = Participant.Fox :=
by sorry

end NUMINAMATH_CALUDE_race_result_l298_29863


namespace NUMINAMATH_CALUDE_max_districts_in_park_l298_29833

theorem max_districts_in_park (park_side : ℝ) (district_length : ℝ) (district_width : ℝ)
  (h_park_side : park_side = 14)
  (h_district_length : district_length = 8)
  (h_district_width : district_width = 2) :
  ⌊(park_side^2) / (district_length * district_width)⌋ = 12 := by
sorry

end NUMINAMATH_CALUDE_max_districts_in_park_l298_29833


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l298_29877

/-- Represents the distribution of painted balls among different colors. -/
structure BallDistribution where
  totalBalls : ℕ
  numColors : ℕ
  equalColorCount : ℕ
  doubleColorCount : ℕ
  ballsPerEqualColor : ℕ
  ballsPerDoubleColor : ℕ

/-- Theorem stating the correct distribution of balls among colors. -/
theorem ball_distribution_theorem (d : BallDistribution) : 
  d.totalBalls = 600 ∧ 
  d.numColors = 15 ∧ 
  d.equalColorCount = 10 ∧ 
  d.doubleColorCount = 5 ∧ 
  d.ballsPerDoubleColor = 2 * d.ballsPerEqualColor →
  d.ballsPerEqualColor = 30 ∧ 
  d.ballsPerDoubleColor = 60 ∧
  d.totalBalls = d.equalColorCount * d.ballsPerEqualColor + d.doubleColorCount * d.ballsPerDoubleColor :=
by sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l298_29877


namespace NUMINAMATH_CALUDE_fraction_comparison_l298_29854

theorem fraction_comparison (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 3 ∧ x ≠ 5/3 → 
  (8*x - 3 > 5 - 3*x ↔ (8/11 < x ∧ x < 5/3) ∨ (5/3 < x ∧ x ≤ 3)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l298_29854


namespace NUMINAMATH_CALUDE_unique_solution_for_prime_power_equation_l298_29826

theorem unique_solution_for_prime_power_equation :
  ∀ m p x : ℕ,
    Prime p →
    2^m * p^2 + 27 = x^3 →
    m = 1 ∧ p = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_prime_power_equation_l298_29826


namespace NUMINAMATH_CALUDE_sum_and_product_problem_l298_29886

theorem sum_and_product_problem (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 ∧ x^2 + y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_problem_l298_29886


namespace NUMINAMATH_CALUDE_division_problem_l298_29875

theorem division_problem (L S q : ℕ) : 
  L - S = 1000 → 
  L = 1100 → 
  L = S * q + 10 → 
  q = 10 := by sorry

end NUMINAMATH_CALUDE_division_problem_l298_29875


namespace NUMINAMATH_CALUDE_jeremy_cannot_be_sure_l298_29838

theorem jeremy_cannot_be_sure (n : ℕ) : ∃ (remaining_permutations : ℝ), 
  remaining_permutations > 1 ∧ 
  remaining_permutations = (2^n).factorial / 2^(n * 2^(n-1)) := by
  sorry

#check jeremy_cannot_be_sure

end NUMINAMATH_CALUDE_jeremy_cannot_be_sure_l298_29838


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l298_29849

theorem power_fraction_simplification : (8^15) / (16^7) = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l298_29849


namespace NUMINAMATH_CALUDE_motel_rent_reduction_l298_29843

/-- Represents a motel with rooms rented at two different prices -/
structure Motel :=
  (price1 : ℕ)
  (price2 : ℕ)
  (total_rent : ℕ)
  (room_change : ℕ)

/-- The percentage reduction in total rent when changing room prices -/
def rent_reduction_percentage (m : Motel) : ℚ :=
  ((m.price2 - m.price1) * m.room_change : ℚ) / m.total_rent * 100

/-- Theorem stating that for a motel with specific conditions, 
    changing 10 rooms from $60 to $40 results in a 10% rent reduction -/
theorem motel_rent_reduction 
  (m : Motel) 
  (h1 : m.price1 = 40)
  (h2 : m.price2 = 60)
  (h3 : m.total_rent = 2000)
  (h4 : m.room_change = 10) :
  rent_reduction_percentage m = 10 := by
  sorry

#eval rent_reduction_percentage ⟨40, 60, 2000, 10⟩

end NUMINAMATH_CALUDE_motel_rent_reduction_l298_29843


namespace NUMINAMATH_CALUDE_arcsin_zero_l298_29810

theorem arcsin_zero : Real.arcsin 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_zero_l298_29810


namespace NUMINAMATH_CALUDE_inequality_proof_l298_29851

theorem inequality_proof (x : ℝ) (h1 : 3/2 ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l298_29851


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l298_29898

/-- Given two vectors OA and OB in R², prove that if they are perpendicular
    and OA = (-1, 2) and OB = (3, m), then m = 3/2. -/
theorem perpendicular_vectors_m_value (OA OB : ℝ × ℝ) (m : ℝ) :
  OA = (-1, 2) →
  OB = (3, m) →
  OA.1 * OB.1 + OA.2 * OB.2 = 0 →
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l298_29898


namespace NUMINAMATH_CALUDE_max_food_per_guest_l298_29879

theorem max_food_per_guest (total_food : ℝ) (min_guests : ℕ) 
  (h1 : total_food = 325) 
  (h2 : min_guests = 163) : 
  ∃ (max_food : ℝ), max_food ≤ 2 ∧ max_food > total_food / min_guests :=
by
  sorry

end NUMINAMATH_CALUDE_max_food_per_guest_l298_29879


namespace NUMINAMATH_CALUDE_tammy_climbing_speed_l298_29822

/-- Tammy's mountain climbing problem -/
theorem tammy_climbing_speed 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (h1 : total_time = 14) 
  (h2 : total_distance = 52) 
  (h3 : speed_difference = 0.5) 
  (h4 : time_difference = 2) : 
  ∃ (v : ℝ), v > 0 ∧ 
    v * ((total_time + time_difference) / 2) + 
    (v + speed_difference) * ((total_time - time_difference) / 2) = total_distance ∧
    v + speed_difference = 4 := by
  sorry


end NUMINAMATH_CALUDE_tammy_climbing_speed_l298_29822


namespace NUMINAMATH_CALUDE_min_value_of_expression_l298_29848

theorem min_value_of_expression (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0) 
  (h_zeros : x₁^2 - 4*a*x₁ + a^2 = 0 ∧ x₂^2 - 4*a*x₂ + a^2 = 0) :
  x₁ + x₂ + a / (x₁ * x₂) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l298_29848


namespace NUMINAMATH_CALUDE_monogram_cost_per_backpack_l298_29840

/-- Proves the cost of monogramming each backpack --/
theorem monogram_cost_per_backpack 
  (num_backpacks : ℕ)
  (original_price : ℚ)
  (discount_percent : ℚ)
  (total_cost : ℚ)
  (h1 : num_backpacks = 5)
  (h2 : original_price = 20)
  (h3 : discount_percent = 20 / 100)
  (h4 : total_cost = 140) :
  (total_cost - num_backpacks * (original_price * (1 - discount_percent))) / num_backpacks = 12 := by
  sorry

#check monogram_cost_per_backpack

end NUMINAMATH_CALUDE_monogram_cost_per_backpack_l298_29840


namespace NUMINAMATH_CALUDE_set_equality_l298_29870

def set_a : Set ℕ := {x : ℕ | 2 * x + 3 ≥ 3 * x}
def set_b : Set ℕ := {0, 1, 2, 3}

theorem set_equality : set_a = set_b := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l298_29870


namespace NUMINAMATH_CALUDE_inequality_proof_l298_29846

theorem inequality_proof (a b c d : ℝ) 
  (h1 : b < 0) (h2 : 0 < a) (h3 : d < c) (h4 : c < 0) : 
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l298_29846


namespace NUMINAMATH_CALUDE_equivalence_of_propositions_l298_29831

theorem equivalence_of_propositions (a b c : ℝ) :
  (a < b → a + c < b + c) ∧
  (a + c < b + c → a < b) ∧
  (a ≥ b → a + c ≥ b + c) ∧
  (a + c ≥ b + c → a ≥ b) := by
  sorry

end NUMINAMATH_CALUDE_equivalence_of_propositions_l298_29831


namespace NUMINAMATH_CALUDE_accessories_cost_l298_29807

theorem accessories_cost (computer_cost : ℝ) (playstation_worth : ℝ) (pocket_payment : ℝ)
  (h1 : computer_cost = 700)
  (h2 : playstation_worth = 400)
  (h3 : pocket_payment = 580) :
  let playstation_sold := playstation_worth * 0.8
  let total_available := playstation_sold + pocket_payment
  let accessories_cost := total_available - computer_cost
  accessories_cost = 200 := by sorry

end NUMINAMATH_CALUDE_accessories_cost_l298_29807


namespace NUMINAMATH_CALUDE_twentieth_is_thursday_l298_29809

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Definition of a month with the given condition -/
structure Month where
  dates : List Date
  threeSundaysOnEvenDates : ∃ (d1 d2 d3 : Date),
    d1 ∈ dates ∧ d2 ∈ dates ∧ d3 ∈ dates ∧
    d1.dayOfWeek = DayOfWeek.Sunday ∧ d2.dayOfWeek = DayOfWeek.Sunday ∧ d3.dayOfWeek = DayOfWeek.Sunday ∧
    d1.day % 2 = 0 ∧ d2.day % 2 = 0 ∧ d3.day % 2 = 0

/-- Theorem stating that the 20th is a Thursday in a month with three Sundays on even dates -/
theorem twentieth_is_thursday (m : Month) : 
  ∃ (d : Date), d ∈ m.dates ∧ d.day = 20 ∧ d.dayOfWeek = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_twentieth_is_thursday_l298_29809


namespace NUMINAMATH_CALUDE_right_triangle_area_perimeter_l298_29808

theorem right_triangle_area_perimeter 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 13) 
  (h_leg : a = 5) : 
  (1/2 * a * b = 30) ∧ (a + b + c = 30) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_perimeter_l298_29808


namespace NUMINAMATH_CALUDE_probability_above_parabola_l298_29806

def is_single_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def above_parabola (a b : ℕ) : Prop := ∀ x : ℚ, b > a * x^2 + b * x

def count_valid_pairs : ℕ := 69

def total_pairs : ℕ := 81

theorem probability_above_parabola :
  (count_valid_pairs : ℚ) / (total_pairs : ℚ) = 23 / 27 := by sorry

end NUMINAMATH_CALUDE_probability_above_parabola_l298_29806


namespace NUMINAMATH_CALUDE_solution_characterization_l298_29842

/-- The set of solutions to the equation x y z + x y + y z + z x + x + y + z = 1977 --/
def SolutionSet : Set (ℕ × ℕ × ℕ) :=
  {(1, 22, 42), (1, 42, 22), (22, 1, 42), (22, 42, 1), (42, 1, 22), (42, 22, 1)}

/-- The equation x y z + x y + y z + z x + x + y + z = 1977 --/
def SatisfiesEquation (x y z : ℕ) : Prop :=
  x * y * z + x * y + y * z + z * x + x + y + z = 1977

theorem solution_characterization :
  ∀ x y z : ℕ, (x > 0 ∧ y > 0 ∧ z > 0) →
    (SatisfiesEquation x y z ↔ (x, y, z) ∈ SolutionSet) := by
  sorry

end NUMINAMATH_CALUDE_solution_characterization_l298_29842


namespace NUMINAMATH_CALUDE_probability_sum_three_l298_29844

/-- Represents the color of a ball --/
inductive BallColor
  | Red
  | Yellow
  | Blue

/-- Represents the score of a ball --/
def score (color : BallColor) : ℕ :=
  match color with
  | BallColor.Red => 1
  | BallColor.Yellow => 2
  | BallColor.Blue => 3

/-- The total number of balls in the bag --/
def totalBalls : ℕ := 6

/-- The number of possible outcomes when drawing two balls with replacement --/
def totalOutcomes : ℕ := totalBalls * totalBalls

/-- The number of favorable outcomes (sum of scores is 3) --/
def favorableOutcomes : ℕ := 12

/-- Theorem stating that the probability of drawing two balls with a sum of scores equal to 3 is 1/3 --/
theorem probability_sum_three (h : favorableOutcomes = 12) :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_probability_sum_three_l298_29844


namespace NUMINAMATH_CALUDE_probability_non_yellow_jelly_bean_l298_29835

/-- The probability of selecting a non-yellow jelly bean from a bag -/
theorem probability_non_yellow_jelly_bean 
  (red : ℕ) (green : ℕ) (yellow : ℕ) (blue : ℕ)
  (h_red : red = 4)
  (h_green : green = 5)
  (h_yellow : yellow = 9)
  (h_blue : blue = 10) :
  (red + green + blue : ℚ) / (red + green + yellow + blue) = 19 / 28 := by
sorry

end NUMINAMATH_CALUDE_probability_non_yellow_jelly_bean_l298_29835


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l298_29817

theorem arithmetic_calculations :
  (5 + (-6) + 3 - (-4) = 6) ∧
  (-1^2024 - (2 - (-2)^3) / (-2/5) * 5/2 = 123/2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l298_29817


namespace NUMINAMATH_CALUDE_cosine_rationality_l298_29866

theorem cosine_rationality (x : ℝ) 
  (h1 : ∃ q : ℚ, (Real.sin (64 * x) + Real.sin (65 * x)) = ↑q)
  (h2 : ∃ r : ℚ, (Real.cos (64 * x) + Real.cos (65 * x)) = ↑r) :
  ∃ (a b : ℚ), (Real.cos (64 * x) = ↑a ∧ Real.cos (65 * x) = ↑b) :=
sorry

end NUMINAMATH_CALUDE_cosine_rationality_l298_29866


namespace NUMINAMATH_CALUDE_complex_product_real_imag_parts_l298_29829

theorem complex_product_real_imag_parts : 
  let Z : ℂ := (1 + Complex.I) * (2 - Complex.I)
  let m : ℝ := Z.re
  let n : ℝ := Z.im
  m * n = 3 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_imag_parts_l298_29829


namespace NUMINAMATH_CALUDE_expression_sign_negative_l298_29891

theorem expression_sign_negative :
  0 < 1 ∧ 1 < Real.pi / 2 →
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → Real.sin x < Real.sin y) →
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → Real.cos y < Real.cos x) →
  (Real.cos (Real.cos 1) - Real.cos 1) * (Real.sin (Real.sin 1) - Real.sin 1) < 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_sign_negative_l298_29891


namespace NUMINAMATH_CALUDE_power_sum_equality_l298_29821

theorem power_sum_equality : (-1)^53 + 2^(5^3 - 2^3 + 3^2) = 2^126 - 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l298_29821


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l298_29882

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two :
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l298_29882


namespace NUMINAMATH_CALUDE_lcm_of_12_25_45_60_l298_29862

theorem lcm_of_12_25_45_60 : Nat.lcm 12 (Nat.lcm 25 (Nat.lcm 45 60)) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_25_45_60_l298_29862


namespace NUMINAMATH_CALUDE_employee_count_l298_29825

/-- The number of employees in an organization (excluding the manager) -/
def num_employees : ℕ := sorry

/-- The average monthly salary of employees (excluding manager) in Rs. -/
def avg_salary : ℕ := 2000

/-- The increase in average salary when manager's salary is added, in Rs. -/
def salary_increase : ℕ := 200

/-- The manager's monthly salary in Rs. -/
def manager_salary : ℕ := 5800

theorem employee_count :
  (num_employees * avg_salary + manager_salary) / (num_employees + 1) = avg_salary + salary_increase ∧
  num_employees = 18 := by sorry

end NUMINAMATH_CALUDE_employee_count_l298_29825


namespace NUMINAMATH_CALUDE_vacation_payment_difference_is_zero_l298_29869

/-- Represents the vacation expenses and payments of four friends -/
structure VacationExpenses where
  alice_paid : ℝ
  bob_paid : ℝ
  charlie_paid : ℝ
  donna_paid : ℝ
  alice_to_charlie : ℝ
  bob_to_donna : ℝ

/-- Theorem stating that the difference between Alice's payment to Charlie
    and Bob's payment to Donna is zero, given the vacation expenses -/
theorem vacation_payment_difference_is_zero
  (expenses : VacationExpenses)
  (h1 : expenses.alice_paid = 90)
  (h2 : expenses.bob_paid = 150)
  (h3 : expenses.charlie_paid = 120)
  (h4 : expenses.donna_paid = 240)
  (h5 : expenses.alice_paid + expenses.bob_paid + expenses.charlie_paid + expenses.donna_paid = 600)
  (h6 : (expenses.alice_paid + expenses.bob_paid + expenses.charlie_paid + expenses.donna_paid) / 4 = 150)
  (h7 : expenses.alice_to_charlie = 150 - expenses.alice_paid)
  (h8 : expenses.bob_to_donna = 150 - expenses.bob_paid)
  : expenses.alice_to_charlie - expenses.bob_to_donna = 0 := by
  sorry

#check vacation_payment_difference_is_zero

end NUMINAMATH_CALUDE_vacation_payment_difference_is_zero_l298_29869


namespace NUMINAMATH_CALUDE_division_of_powers_l298_29895

theorem division_of_powers (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a * b^2) / (a * b) = b :=
by sorry

end NUMINAMATH_CALUDE_division_of_powers_l298_29895


namespace NUMINAMATH_CALUDE_max_value_constraint_l298_29889

theorem max_value_constraint (p q r : ℝ) (h : 9 * p^2 + 4 * q^2 + 25 * r^2 = 4) :
  5 * p + 3 * q + 10 * r ≤ 10 * Real.sqrt 13 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l298_29889


namespace NUMINAMATH_CALUDE_coin_problem_l298_29855

theorem coin_problem (x : ℕ) : 
  (x + (x + 3) + (20 - 2*x) = 23) →  -- Total coins
  (5*x + 10*(x + 3) + 25*(20 - 2*x) = 320) →  -- Total value
  (20 - 2*x) - x = 2  -- Difference between 25-cent and 5-cent coins
  := by sorry

end NUMINAMATH_CALUDE_coin_problem_l298_29855


namespace NUMINAMATH_CALUDE_tens_digit_of_five_pow_2023_l298_29815

theorem tens_digit_of_five_pow_2023 : (5^2023 / 10) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_five_pow_2023_l298_29815


namespace NUMINAMATH_CALUDE_trains_crossing_time_l298_29874

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 120 →
  train_speed_kmh = 54 →
  (2 * train_length) / (2 * (train_speed_kmh * 1000 / 3600)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_trains_crossing_time_l298_29874


namespace NUMINAMATH_CALUDE_chemical_reaction_results_l298_29827

/-- Represents the chemical reaction between CaCO3 and HCl -/
structure ChemicalReaction where
  temperature : ℝ
  pressure : ℝ
  hcl_moles : ℝ
  cacl2_moles : ℝ
  co2_moles : ℝ
  h2o_moles : ℝ
  std_enthalpy_change : ℝ

/-- Calculates the amount of CaCO3 required and the change in enthalpy -/
def calculate_reaction_results (reaction : ChemicalReaction) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct results of the chemical reaction -/
theorem chemical_reaction_results :
  let reaction := ChemicalReaction.mk 25 1 4 2 2 2 (-178)
  let (caco3_grams, enthalpy_change) := calculate_reaction_results reaction
  caco3_grams = 200.18 ∧ enthalpy_change = -356 := by
  sorry

end NUMINAMATH_CALUDE_chemical_reaction_results_l298_29827


namespace NUMINAMATH_CALUDE_table_price_is_84_l298_29830

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The condition that the price of 2 chairs and 1 table is 60% of the price of 1 chair and 2 tables -/
def price_ratio_condition : Prop :=
  2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

/-- The condition that the price of 1 table and 1 chair is $96 -/
def total_price_condition : Prop :=
  chair_price + table_price = 96

theorem table_price_is_84 
  (h1 : price_ratio_condition) 
  (h2 : total_price_condition) : 
  table_price = 84 := by sorry

end NUMINAMATH_CALUDE_table_price_is_84_l298_29830


namespace NUMINAMATH_CALUDE_ellipse_sine_intersections_l298_29897

/-- An ellipse with center (h, k) and semi-major and semi-minor axes a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of an ellipse -/
def ellipse_eq (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.b^2 = 1

/-- The graph of y = sin x -/
def sine_graph (x y : ℝ) : Prop :=
  y = Real.sin x

/-- A point (x, y) is an intersection point if it satisfies both equations -/
def is_intersection_point (e : Ellipse) (x y : ℝ) : Prop :=
  ellipse_eq e x y ∧ sine_graph x y

/-- The theorem stating that there exists an ellipse with more than 8 intersection points -/
theorem ellipse_sine_intersections :
  ∃ (e : Ellipse), ∃ (points : Finset (ℝ × ℝ)),
    (∀ (p : ℝ × ℝ), p ∈ points → is_intersection_point e p.1 p.2) ∧
    points.card > 8 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_sine_intersections_l298_29897


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l298_29887

/-- Proves that given a journey of 540 miles, where the last 120 miles are traveled at 40 mph,
    and the average speed for the entire journey is 54 mph, the speed for the first 420 miles
    must be 60 mph. -/
theorem journey_speed_calculation (v : ℝ) : 
  v > 0 →                           -- Assume positive speed
  540 / (420 / v + 120 / 40) = 54 → -- Average speed equation
  v = 60 :=                         -- Conclusion: speed for first part is 60 mph
by sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l298_29887


namespace NUMINAMATH_CALUDE_consecutive_non_prime_powers_l298_29860

theorem consecutive_non_prime_powers (r : ℕ) (hr : r > 0) :
  ∃ x : ℕ, ∀ i ∈ Finset.range r, ¬ ∃ (p : ℕ) (n : ℕ), Prime p ∧ x + i + 1 = p ^ n :=
sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_powers_l298_29860


namespace NUMINAMATH_CALUDE_lansing_elementary_students_l298_29883

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := 247

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := num_schools * students_per_school

/-- Theorem stating the total number of elementary students in Lansing -/
theorem lansing_elementary_students : total_students = 6175 := by
  sorry

end NUMINAMATH_CALUDE_lansing_elementary_students_l298_29883


namespace NUMINAMATH_CALUDE_job_duration_l298_29856

theorem job_duration (daily_wage : ℕ) (daily_fine : ℕ) (total_earnings : ℕ) (absent_days : ℕ) :
  daily_wage = 10 →
  daily_fine = 2 →
  total_earnings = 216 →
  absent_days = 7 →
  ∃ (work_days : ℕ), work_days * daily_wage - absent_days * daily_fine = total_earnings ∧ work_days = 23 :=
by sorry

end NUMINAMATH_CALUDE_job_duration_l298_29856


namespace NUMINAMATH_CALUDE_sum_of_prime_divisors_2018_l298_29819

theorem sum_of_prime_divisors_2018 : ∃ p q : Nat, 
  p.Prime ∧ q.Prime ∧ 
  p ≠ q ∧
  p * q = 2018 ∧
  (∀ r : Nat, r.Prime → r ∣ 2018 → r = p ∨ r = q) ∧
  p + q = 1011 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_divisors_2018_l298_29819


namespace NUMINAMATH_CALUDE_moon_radius_scientific_notation_l298_29885

/-- The radius of the moon in meters -/
def moon_radius : ℝ := 1738000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Theorem stating that the moon's radius is equal to its scientific notation representation -/
theorem moon_radius_scientific_notation :
  ∃ (sn : ScientificNotation), moon_radius = sn.coefficient * (10 : ℝ) ^ sn.exponent :=
sorry

end NUMINAMATH_CALUDE_moon_radius_scientific_notation_l298_29885


namespace NUMINAMATH_CALUDE_cricket_team_captain_age_l298_29811

theorem cricket_team_captain_age
  (team_size : ℕ)
  (captain_age : ℕ)
  (wicket_keeper_age : ℕ)
  (team_average_age : ℕ)
  (h1 : team_size = 11)
  (h2 : wicket_keeper_age = captain_age + 3)
  (h3 : (team_size - 2) * (team_average_age - 1) = team_size * team_average_age - captain_age - wicket_keeper_age)
  (h4 : team_average_age = 23) :
  captain_age = 26 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_captain_age_l298_29811


namespace NUMINAMATH_CALUDE_only_valid_quadruples_l298_29847

/-- A quadruple of non-negative integers satisfying the given conditions -/
structure ValidQuadruple where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  eq : a * b = 2 * (1 + c * d)
  triangle : (a - c) + (b - d) > c + d ∧ 
             (a - c) + (c + d) > b - d ∧ 
             (b - d) + (c + d) > a - c

/-- The theorem stating that only two specific quadruples satisfy the conditions -/
theorem only_valid_quadruples : 
  ∀ q : ValidQuadruple, (q.a = 1 ∧ q.b = 2 ∧ q.c = 0 ∧ q.d = 1) ∨ 
                        (q.a = 2 ∧ q.b = 1 ∧ q.c = 1 ∧ q.d = 0) := by
  sorry

end NUMINAMATH_CALUDE_only_valid_quadruples_l298_29847


namespace NUMINAMATH_CALUDE_courtyard_paving_l298_29828

/-- The length of the courtyard in meters -/
def courtyard_length : ℝ := 25

/-- The width of the courtyard in meters -/
def courtyard_width : ℝ := 20

/-- The length of a brick in meters -/
def brick_length : ℝ := 0.15

/-- The width of a brick in meters -/
def brick_width : ℝ := 0.08

/-- The total number of bricks required to cover the courtyard -/
def total_bricks : ℕ := 41667

theorem courtyard_paving :
  ⌈(courtyard_length * courtyard_width) / (brick_length * brick_width)⌉ = total_bricks := by
  sorry

end NUMINAMATH_CALUDE_courtyard_paving_l298_29828


namespace NUMINAMATH_CALUDE_solve_motel_problem_l298_29834

def motel_problem (higher_rate : ℕ) : Prop :=
  ∃ (num_higher_rate : ℕ) (num_lower_rate : ℕ),
    -- There are two types of room rates: $40 and a higher amount
    higher_rate > 40 ∧
    -- The actual total rent charged was $1000
    num_higher_rate * higher_rate + num_lower_rate * 40 = 1000 ∧
    -- If 10 rooms at the higher rate were rented for $40 instead, the total rent would be reduced by 20%
    (num_higher_rate - 10) * higher_rate + (num_lower_rate + 10) * 40 = 800

theorem solve_motel_problem : 
  ∃ (higher_rate : ℕ), motel_problem higher_rate ∧ higher_rate = 60 :=
sorry

end NUMINAMATH_CALUDE_solve_motel_problem_l298_29834


namespace NUMINAMATH_CALUDE_fraction_simplification_l298_29803

theorem fraction_simplification (a b : ℝ) (h : a ≠ b ∧ a ≠ -b) :
  (5 * a + 3 * b) / (a^2 - b^2) - (2 * a) / (a^2 - b^2) = 3 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l298_29803


namespace NUMINAMATH_CALUDE_carly_shipping_cost_l298_29893

/-- Calculates the shipping cost given a flat fee, per-pound cost, and weight -/
def shipping_cost (flat_fee : ℝ) (per_pound_cost : ℝ) (weight : ℝ) : ℝ :=
  flat_fee + per_pound_cost * weight

/-- Theorem: The shipping cost for Carly's package is $9.00 -/
theorem carly_shipping_cost :
  shipping_cost 5 0.8 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_carly_shipping_cost_l298_29893


namespace NUMINAMATH_CALUDE_new_tires_cost_l298_29836

def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def total_spent : ℝ := 387.85

theorem new_tires_cost (new_tires_cost : ℝ) : 
  new_tires_cost = total_spent - (speakers_cost + cd_player_cost) :=
by sorry

end NUMINAMATH_CALUDE_new_tires_cost_l298_29836


namespace NUMINAMATH_CALUDE_parking_fines_count_l298_29837

/-- Represents the number of citations issued for each category -/
structure Citations where
  littering : ℕ
  offLeash : ℕ
  parking : ℕ

/-- Theorem stating that given the conditions, the number of parking fines is 16 -/
theorem parking_fines_count (c : Citations) : 
  c.littering = 4 ∧ 
  c.littering = c.offLeash ∧ 
  c.littering + c.offLeash + c.parking = 24 → 
  c.parking = 16 := by
sorry

end NUMINAMATH_CALUDE_parking_fines_count_l298_29837


namespace NUMINAMATH_CALUDE_octagon_diagonal_intersection_probability_l298_29814

/-- The number of vertices in a regular octagon -/
def octagon_vertices : ℕ := 8

/-- The number of diagonals in a regular octagon -/
def octagon_diagonals : ℕ := octagon_vertices * (octagon_vertices - 3) / 2

/-- The number of ways to select two distinct diagonals from a regular octagon -/
def ways_to_select_two_diagonals : ℕ := 
  Nat.choose octagon_diagonals 2

/-- The number of ways to select four vertices from a regular octagon -/
def ways_to_select_four_vertices : ℕ := 
  Nat.choose octagon_vertices 4

/-- The probability that two randomly selected distinct diagonals 
    in a regular octagon intersect at a point strictly within the octagon -/
theorem octagon_diagonal_intersection_probability : 
  (ways_to_select_four_vertices : ℚ) / ways_to_select_two_diagonals = 7 / 19 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonal_intersection_probability_l298_29814


namespace NUMINAMATH_CALUDE_ratio_problem_l298_29800

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) :
  x / y = 11 / 6 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l298_29800


namespace NUMINAMATH_CALUDE_sons_age_l298_29876

/-- Given a man and his son, where the man is 28 years older than his son,
    and in two years the man's age will be twice the age of his son,
    prove that the present age of the son is 26 years. -/
theorem sons_age (son_age man_age : ℕ) : 
  man_age = son_age + 28 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 26 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l298_29876


namespace NUMINAMATH_CALUDE_circle_radius_in_square_configuration_l298_29867

/-- A configuration of five congruent circles packed inside a unit square,
    where one circle is centered at the center of the square and
    the other four are tangent to the central circle and two adjacent sides of the square. -/
structure CircleConfiguration where
  radius : ℝ
  is_unit_square : ℝ
  circle_count : ℕ
  central_circle_exists : Bool
  external_circles_tangent : Bool

/-- The radius of each circle in the described configuration is √2 / (4 + 2√2) -/
theorem circle_radius_in_square_configuration (config : CircleConfiguration) 
  (h1 : config.is_unit_square = 1)
  (h2 : config.circle_count = 5)
  (h3 : config.central_circle_exists = true)
  (h4 : config.external_circles_tangent = true) :
  config.radius = Real.sqrt 2 / (4 + 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_in_square_configuration_l298_29867


namespace NUMINAMATH_CALUDE_projection_theorem_l298_29871

def vector1 : ℝ × ℝ := (-4, 2)
def vector2 : ℝ × ℝ := (3, 5)

theorem projection_theorem (v : ℝ × ℝ) :
  ∃ (p : ℝ × ℝ), 
    (∃ (k1 : ℝ), p = Prod.mk (k1 * v.1) (k1 * v.2) ∧ 
      (p.1 - vector1.1) * v.1 + (p.2 - vector1.2) * v.2 = 0) ∧
    (∃ (k2 : ℝ), p = Prod.mk (k2 * v.1) (k2 * v.2) ∧ 
      (p.1 - vector2.1) * v.1 + (p.2 - vector2.2) * v.2 = 0) →
    p = (-39/29, 91/29) := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l298_29871


namespace NUMINAMATH_CALUDE_project_completion_days_l298_29812

/-- Calculates the number of days required to complete a project given normal work hours, 
    extra work hours, and total project hours. -/
theorem project_completion_days 
  (normal_hours : ℕ) 
  (extra_hours : ℕ) 
  (total_project_hours : ℕ) 
  (h1 : normal_hours = 10)
  (h2 : extra_hours = 5)
  (h3 : total_project_hours = 1500) : 
  total_project_hours / (normal_hours + extra_hours) = 100 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_days_l298_29812


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_quarter_range_of_a_when_q_sufficient_not_necessary_l298_29858

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

def q (x : ℝ) : Prop := ∃ m : ℝ, 1 < m ∧ m < 2 ∧ x = (1/2)^(m-1)

-- Part I
theorem range_of_x_when_a_is_quarter :
  ∀ x : ℝ, (p x (1/4) ∧ q x) ↔ (1/2 < x ∧ x < 3/4) :=
sorry

-- Part II
theorem range_of_a_when_q_sufficient_not_necessary :
  (∀ x a : ℝ, q x → p x a) ∧ 
  (∃ x a : ℝ, p x a ∧ ¬(q x)) ↔ 
  (∀ a : ℝ, (1/3 ≤ a ∧ a ≤ 1/2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_quarter_range_of_a_when_q_sufficient_not_necessary_l298_29858


namespace NUMINAMATH_CALUDE_extended_line_segment_l298_29890

/-- Given a line segment AB extended to points P and Q, prove the expressions for P and Q -/
theorem extended_line_segment (A B P Q : ℝ × ℝ) : 
  (∃ (k₁ k₂ : ℝ), k₁ > 0 ∧ k₂ > 0 ∧ 
    7 * (P.1 - B.1) = 2 * (P.1 - A.1) ∧
    7 * (P.2 - B.2) = 2 * (P.2 - A.2) ∧
    5 * (Q.1 - B.1) = (Q.1 - A.1) ∧
    5 * (Q.2 - B.2) = (Q.2 - A.2)) →
  (P = (-2/5 : ℝ) • A + (7/5 : ℝ) • B ∧
   Q = (-1/4 : ℝ) • A + (5/4 : ℝ) • B) := by
sorry

end NUMINAMATH_CALUDE_extended_line_segment_l298_29890


namespace NUMINAMATH_CALUDE_scaling_transformation_result_l298_29820

/-- A scaling transformation in a 2D plane -/
structure ScalingTransformation where
  x_scale : ℝ
  y_scale : ℝ

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Apply a scaling transformation to a point -/
def applyTransformation (t : ScalingTransformation) (p : Point) : Point :=
  { x := t.x_scale * p.x,
    y := t.y_scale * p.y }

theorem scaling_transformation_result :
  let A : Point := { x := 1/3, y := -2 }
  let φ : ScalingTransformation := { x_scale := 3, y_scale := 1/2 }
  let A' : Point := applyTransformation φ A
  A'.x = 1 ∧ A'.y = -1 := by sorry

end NUMINAMATH_CALUDE_scaling_transformation_result_l298_29820


namespace NUMINAMATH_CALUDE_thirteen_rectangles_l298_29801

/-- A rectangle with integer side lengths. -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Checks if a rectangle meets the given criteria. -/
def meetsConditions (rect : Rectangle) : Prop :=
  rect.width > 0 ∧ rect.height > 0 ∧
  2 * (rect.width + rect.height) = 80 ∧
  ∃ k : ℕ, rect.width = 3 * k

/-- Two rectangles are considered congruent if they have the same dimensions (ignoring orientation). -/
def areCongruent (rect1 rect2 : Rectangle) : Prop :=
  (rect1.width = rect2.width ∧ rect1.height = rect2.height) ∨
  (rect1.width = rect2.height ∧ rect1.height = rect2.width)

/-- The main theorem stating that there are exactly 13 non-congruent rectangles meeting the conditions. -/
theorem thirteen_rectangles :
  ∃ (rectangles : Finset Rectangle),
    rectangles.card = 13 ∧
    (∀ rect ∈ rectangles, meetsConditions rect) ∧
    (∀ rect, meetsConditions rect → ∃ unique_rect ∈ rectangles, areCongruent rect unique_rect) :=
  sorry

end NUMINAMATH_CALUDE_thirteen_rectangles_l298_29801


namespace NUMINAMATH_CALUDE_quadratic_transform_l298_29878

theorem quadratic_transform (p q r : ℝ) :
  (∃ m l : ℝ, ∀ x : ℝ, px^2 + qx + r = 5*(x - 3)^2 + 15 ∧ 2*px^2 + 2*qx + 2*r = m*(x - 3)^2 + l) →
  (∃ m l : ℝ, ∀ x : ℝ, 2*px^2 + 2*qx + 2*r = m*(x - 3)^2 + l) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transform_l298_29878


namespace NUMINAMATH_CALUDE_kia_vehicles_count_l298_29888

/-- The number of Kia vehicles on the lot -/
def num_kia (total vehicles : ℕ) (num_dodge num_hyundai : ℕ) : ℕ :=
  total - num_dodge - num_hyundai

/-- Theorem stating the number of Kia vehicles on the lot -/
theorem kia_vehicles_count :
  let total := 400
  let num_dodge := total / 2
  let num_hyundai := num_dodge / 2
  num_kia total num_dodge num_hyundai = 100 := by
sorry

end NUMINAMATH_CALUDE_kia_vehicles_count_l298_29888


namespace NUMINAMATH_CALUDE_david_pushups_count_l298_29864

/-- The number of push-ups done by Zachary -/
def zachary_pushups : ℕ := 19

/-- The number of push-ups done by David -/
def david_pushups : ℕ := 3 * zachary_pushups

theorem david_pushups_count : david_pushups = 57 := by
  sorry

end NUMINAMATH_CALUDE_david_pushups_count_l298_29864


namespace NUMINAMATH_CALUDE_square_root_equation_l298_29899

theorem square_root_equation (x : ℝ) :
  Real.sqrt (3 * x + 4) = 12 → x = 140 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l298_29899
