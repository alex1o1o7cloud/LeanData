import Mathlib

namespace davids_math_marks_l44_4449

/-- Calculates the marks in an unknown subject given the marks in other subjects and the average --/
def calculate_unknown_marks (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + physics + chemistry + biology)

theorem davids_math_marks :
  let english := 81
  let physics := 82
  let chemistry := 67
  let biology := 85
  let average := 76
  calculate_unknown_marks english physics chemistry biology average = 65 := by
  sorry

#eval calculate_unknown_marks 81 82 67 85 76

end davids_math_marks_l44_4449


namespace binomial_27_6_l44_4443

theorem binomial_27_6 (h1 : Nat.choose 26 4 = 14950)
                      (h2 : Nat.choose 26 5 = 65780)
                      (h3 : Nat.choose 26 6 = 230230) :
  Nat.choose 27 6 = 296010 := by
  sorry

end binomial_27_6_l44_4443


namespace restaurant_bill_division_l44_4469

/-- Given a group of friends dividing a restaurant bill evenly, this theorem proves
    the number of friends in the group based on the total bill and individual payment. -/
theorem restaurant_bill_division (total_bill : ℕ) (individual_payment : ℕ) 
    (h1 : total_bill = 135)
    (h2 : individual_payment = 45) :
    total_bill / individual_payment = 3 := by
  sorry

end restaurant_bill_division_l44_4469


namespace total_cost_is_48_l44_4420

/-- The cost of a pencil case in yuan -/
def pencil_case_cost : ℕ := 8

/-- The cost of a backpack in yuan -/
def backpack_cost : ℕ := 5 * pencil_case_cost

/-- The total cost of a backpack and a pencil case in yuan -/
def total_cost : ℕ := backpack_cost + pencil_case_cost

/-- Theorem stating that the total cost of a backpack and a pencil case is 48 yuan -/
theorem total_cost_is_48 : total_cost = 48 := by
  sorry

end total_cost_is_48_l44_4420


namespace selling_price_for_target_profit_impossibility_of_daily_profit_maximum_profit_l44_4453

-- Define the variables and constants
variable (x : ℝ)  -- price increase
def original_price : ℝ := 40
def cost_price : ℝ := 30
def initial_sales : ℝ := 600
def sales_decrease_rate : ℝ := 10

-- Define the profit function
def profit (x : ℝ) : ℝ :=
  (original_price + x - cost_price) * (initial_sales - sales_decrease_rate * x)

-- Theorem 1: Selling price for 10,000 yuan monthly profit
theorem selling_price_for_target_profit :
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ profit x₁ = 10000 ∧ profit x₂ = 10000 ∧
  (x₁ + original_price = 80 ∨ x₁ + original_price = 50) ∧
  (x₂ + original_price = 80 ∨ x₂ + original_price = 50) :=
sorry

-- Theorem 2: Impossibility of 15,000 yuan daily profit
theorem impossibility_of_daily_profit :
  ¬∃ x, profit x = 15000 * 30 :=
sorry

-- Theorem 3: Price and value for maximum profit
theorem maximum_profit :
  ∃ x_max, ∀ x, profit x ≤ profit x_max ∧
  x_max + original_price = 65 ∧ profit x_max = 12250 :=
sorry

end selling_price_for_target_profit_impossibility_of_daily_profit_maximum_profit_l44_4453


namespace complement_intersection_theorem_l44_4477

-- Define the universal set U
def U : Finset Nat := {2, 3, 4, 5, 6}

-- Define set A
def A : Finset Nat := {2, 5, 6}

-- Define set B
def B : Finset Nat := {3, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ B) ∩ A = {2, 6} := by sorry

end complement_intersection_theorem_l44_4477


namespace chocolates_per_box_l44_4451

-- Define the problem parameters
def total_boxes : ℕ := 20
def total_chocolates : ℕ := 500

-- Theorem statement
theorem chocolates_per_box :
  total_chocolates / total_boxes = 25 :=
by
  sorry -- Proof omitted

end chocolates_per_box_l44_4451


namespace x_y_z_order_l44_4405

-- Define the constants
noncomputable def x : ℝ := Real.exp (3⁻¹ * Real.log 3)
noncomputable def y : ℝ := Real.exp (6⁻¹ * Real.log 7)
noncomputable def z : ℝ := 7 ^ (1/7 : ℝ)

-- State the theorem
theorem x_y_z_order : z < y ∧ y < x := by
  sorry

end x_y_z_order_l44_4405


namespace tom_bike_miles_per_day_l44_4470

theorem tom_bike_miles_per_day 
  (total_miles : ℕ) 
  (days_in_year : ℕ) 
  (first_period_days : ℕ) 
  (miles_per_day_first_period : ℕ) 
  (h1 : total_miles = 11860)
  (h2 : days_in_year = 365)
  (h3 : first_period_days = 183)
  (h4 : miles_per_day_first_period = 30) :
  (total_miles - miles_per_day_first_period * first_period_days) / (days_in_year - first_period_days) = 35 :=
by sorry

end tom_bike_miles_per_day_l44_4470


namespace min_soldiers_to_add_l44_4430

theorem min_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) :
  (∃ (M : ℕ), (N + M) % 7 = 0 ∧ (N + M) % 12 = 0) ∧
  (∀ (K : ℕ), K < 82 → ¬((N + K) % 7 = 0 ∧ (N + K) % 12 = 0)) ∧
  ((N + 82) % 7 = 0 ∧ (N + 82) % 12 = 0) :=
by sorry

end min_soldiers_to_add_l44_4430


namespace engineer_teams_count_l44_4438

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of ways to form a team of engineers with given constraints -/
def engineerTeams : ℕ :=
  let totalEngineers := 15
  let phdEngineers := 5
  let msEngineers := 6
  let bsEngineers := 4
  let teamSize := 5
  let minPhd := 2
  let minMs := 2
  let minBs := 1
  (choose phdEngineers minPhd) * (choose msEngineers minMs) * (choose bsEngineers minBs)

theorem engineer_teams_count :
  engineerTeams = 600 := by sorry

end engineer_teams_count_l44_4438


namespace triangle_angle_measure_l44_4447

theorem triangle_angle_measure (y : ℝ) : 
  y > 0 ∧ 
  y < 180 ∧ 
  3*y > 0 ∧ 
  3*y < 180 ∧
  y + 3*y + 40 = 180 → 
  y = 35 := by
sorry

end triangle_angle_measure_l44_4447


namespace janes_minnows_l44_4413

theorem janes_minnows (prize_minnows : ℕ) (total_players : ℕ) (win_percentage : ℚ) (leftover_minnows : ℕ) 
  (h1 : prize_minnows = 3)
  (h2 : total_players = 800)
  (h3 : win_percentage = 15 / 100)
  (h4 : leftover_minnows = 240) :
  prize_minnows * (win_percentage * total_players).floor + leftover_minnows = 600 := by
  sorry

end janes_minnows_l44_4413


namespace joe_not_eating_pizza_probability_l44_4424

theorem joe_not_eating_pizza_probability 
  (p_eat : ℚ) 
  (h_eat : p_eat = 5/8) : 
  1 - p_eat = 3/8 := by
  sorry

end joe_not_eating_pizza_probability_l44_4424


namespace correct_answer_l44_4499

theorem correct_answer (x : ℚ) (h : 2 * x = 80) : x / 3 = 40 / 3 := by
  sorry

end correct_answer_l44_4499


namespace system_solution_l44_4417

theorem system_solution (a : ℝ) (h : a ≠ 0) :
  ∃! (x : ℝ), 3 * x + 2 * x = 15 * a ∧ (1 / a) * x + x = 9 → x = 6 ∧ a = 2 := by
  sorry

end system_solution_l44_4417


namespace typist_salary_l44_4487

theorem typist_salary (x : ℝ) : 
  (x * 1.1 * 0.95 = 1045) → x = 1000 := by sorry

end typist_salary_l44_4487


namespace tracksuit_discount_problem_l44_4403

/-- Given a tracksuit with an original price, if it is discounted by 20% and the discount amount is 30 yuan, then the actual amount spent is 120 yuan. -/
theorem tracksuit_discount_problem (original_price : ℝ) : 
  (original_price - original_price * 0.8 = 30) → 
  (original_price * 0.8 = 120) := by
sorry

end tracksuit_discount_problem_l44_4403


namespace divisibility_of_ones_l44_4486

theorem divisibility_of_ones (p : ℕ) (h_prime : Nat.Prime p) (h_ge_7 : p ≥ 7) :
  ∃ k : ℤ, (10^(p-1) - 1) / 9 = k * p := by
  sorry

end divisibility_of_ones_l44_4486


namespace max_set_size_with_prime_triple_sums_l44_4457

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if the sum of any three elements in a list is prime -/
def allTripleSumsPrime (l : List ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ l → b ∈ l → c ∈ l → a ≠ b → b ≠ c → a ≠ c → isPrime (a + b + c)

/-- The main theorem -/
theorem max_set_size_with_prime_triple_sums :
  ∀ (l : List ℕ), (∀ x ∈ l, x > 0) → l.Nodup → allTripleSumsPrime l → l.length ≤ 4 :=
by sorry

end max_set_size_with_prime_triple_sums_l44_4457


namespace circumcircle_equation_l44_4498

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-1, 5)
def B : ℝ × ℝ := (5, 5)
def C : ℝ × ℝ := (6, -2)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 20 = 0

-- Theorem statement
theorem circumcircle_equation :
  (circle_equation A.1 A.2) ∧
  (circle_equation B.1 B.2) ∧
  (circle_equation C.1 C.2) ∧
  (∀ (x y : ℝ), circle_equation x y → 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2) :=
by sorry

end circumcircle_equation_l44_4498


namespace remaining_water_is_one_cup_l44_4461

/-- Represents Harry's hike and water consumption --/
structure HikeData where
  total_distance : ℝ
  initial_water : ℝ
  duration : ℝ
  leak_rate : ℝ
  last_mile_consumption : ℝ
  first_miles_rate : ℝ

/-- Calculates the remaining water after the hike --/
def remaining_water (data : HikeData) : ℝ :=
  data.initial_water
  - (data.first_miles_rate * (data.total_distance - 1))
  - data.last_mile_consumption
  - (data.leak_rate * data.duration)

/-- Theorem stating that the remaining water is 1 cup --/
theorem remaining_water_is_one_cup (data : HikeData)
  (h1 : data.total_distance = 7)
  (h2 : data.initial_water = 9)
  (h3 : data.duration = 2)
  (h4 : data.leak_rate = 1)
  (h5 : data.last_mile_consumption = 2)
  (h6 : data.first_miles_rate = 0.6666666666666666)
  : remaining_water data = 1 := by
  sorry

end remaining_water_is_one_cup_l44_4461


namespace sum_x_y_z_l44_4432

theorem sum_x_y_z (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 2 * y) :
  x + y + z = 10 * x := by
  sorry

end sum_x_y_z_l44_4432


namespace cubic_function_tangent_and_minimum_l44_4425

/-- A cubic function with parameters m and n -/
def f (m n : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + n*x + 1

/-- The derivative of f -/
def f' (m n : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + n

theorem cubic_function_tangent_and_minimum (m n : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ f m n x = 1 ∧ f' m n x = 0) →
  (∃ x : ℝ, ∀ y : ℝ, f m n y ≥ f m n x ∧ f m n x = -31) →
  m = 12 ∧ n = 36 := by
  sorry

end cubic_function_tangent_and_minimum_l44_4425


namespace perfect_square_condition_l44_4427

theorem perfect_square_condition (W L : ℤ) : 
  (1000 < W) → (W < 2000) → (L > 1) → (W = 2 * L^3) → 
  (∃ m : ℤ, W = m^2) → (L = 8) :=
sorry

end perfect_square_condition_l44_4427


namespace one_third_of_recipe_flour_l44_4475

-- Define the original amount of flour in the recipe
def original_flour : ℚ := 16/3

-- Define the fraction of the recipe we're making
def recipe_fraction : ℚ := 1/3

-- Define the result we want to prove
def result : ℚ := 16/9

-- Theorem statement
theorem one_third_of_recipe_flour :
  recipe_fraction * original_flour = result := by sorry

end one_third_of_recipe_flour_l44_4475


namespace knights_arrangement_exists_l44_4493

/-- Represents a knight in King Arthur's court -/
structure Knight where
  id : ℕ

/-- Represents the relationship between knights -/
inductive Relationship
  | Friend
  | Enemy

/-- Represents the seating arrangement of knights around a round table -/
def Arrangement := List Knight

/-- Function to determine if two knights are enemies -/
def areEnemies (k1 k2 : Knight) : Prop := sorry

/-- Function to count the number of enemies a knight has -/
def enemyCount (k : Knight) (knights : List Knight) : ℕ := sorry

/-- Function to check if an arrangement is valid (no adjacent enemies) -/
def isValidArrangement (arr : Arrangement) : Prop := sorry

/-- Main theorem: There exists a valid arrangement of knights -/
theorem knights_arrangement_exists (n : ℕ) (knights : List Knight) :
  knights.length = 2 * n →
  (∀ k ∈ knights, enemyCount k knights ≤ n - 1) →
  ∃ arr : Arrangement, arr.length = 2 * n ∧ isValidArrangement arr :=
sorry

end knights_arrangement_exists_l44_4493


namespace sqrt_m_minus_n_l44_4423

theorem sqrt_m_minus_n (m n : ℝ) 
  (h1 : Real.sqrt (m - 3) = 3) 
  (h2 : Real.sqrt (n + 1) = 2) : 
  Real.sqrt (m - n) = 3 := by
sorry

end sqrt_m_minus_n_l44_4423


namespace line_parallel_to_parallel_planes_l44_4448

-- Define the concept of a plane
structure Plane :=
  (p : Set (ℝ × ℝ × ℝ))

-- Define the concept of a line
structure Line :=
  (l : Set (ℝ × ℝ × ℝ))

-- Define parallel relationship between planes
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Define parallel relationship between a line and a plane
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define when a line is within a plane
def line_within_plane (l : Line) (p : Plane) : Prop := sorry

-- Theorem statement
theorem line_parallel_to_parallel_planes 
  (p1 p2 : Plane) (l : Line) 
  (h1 : parallel_planes p1 p2) 
  (h2 : parallel_line_plane l p1) :
  parallel_line_plane l p2 ∨ line_within_plane l p2 := 
sorry

end line_parallel_to_parallel_planes_l44_4448


namespace exam_average_l44_4435

theorem exam_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) (h₁ : n₁ = 15) (h₂ : n₂ = 10)
  (h₃ : avg₁ = 75/100) (h₄ : avg_total = 83/100) (h₅ : n₁ + n₂ = 25) :
  let avg₂ := (((n₁ + n₂ : ℚ) * avg_total) - (n₁ * avg₁)) / n₂
  avg₂ = 95/100 := by
sorry

end exam_average_l44_4435


namespace tan_240_degrees_l44_4442

theorem tan_240_degrees : Real.tan (240 * π / 180) = Real.sqrt 3 := by
  sorry

end tan_240_degrees_l44_4442


namespace parabola_y_intercepts_l44_4428

/-- The number of y-intercepts of the parabola x = 3y^2 - 4y + 5 -/
def num_y_intercepts : ℕ := 0

/-- The parabola equation: x = 3y^2 - 4y + 5 -/
def parabola_equation (y : ℝ) : ℝ := 3 * y^2 - 4 * y + 5

theorem parabola_y_intercepts :
  (∀ y : ℝ, parabola_equation y ≠ 0) ∧ num_y_intercepts = 0 := by sorry

end parabola_y_intercepts_l44_4428


namespace pie_division_l44_4401

theorem pie_division (total_pie : ℚ) (num_people : ℕ) (individual_share : ℚ) : 
  total_pie = 5/8 →
  num_people = 4 →
  individual_share = total_pie / num_people →
  individual_share = 5/32 := by
sorry

end pie_division_l44_4401


namespace range_of_a_l44_4445

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 / Real.exp x - 2 * x

theorem range_of_a (a : ℝ) (h : f (a - 3) + f (2 * a^2) ≤ 0) :
  -3/2 ≤ a ∧ a ≤ 1 := by sorry

end range_of_a_l44_4445


namespace polynomial_coefficient_bound_l44_4482

theorem polynomial_coefficient_bound (a b c d : ℝ) : 
  (∀ x : ℝ, |x| < 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) → 
  |a| + |b| + |c| + |d| ≤ 7 := by
sorry

end polynomial_coefficient_bound_l44_4482


namespace collinear_probability_l44_4400

/-- The number of dots in one side of the square grid -/
def grid_size : ℕ := 5

/-- The total number of dots in the grid -/
def total_dots : ℕ := grid_size * grid_size

/-- The number of dots to be chosen -/
def chosen_dots : ℕ := 4

/-- The number of ways to choose 4 dots from the grid -/
def total_choices : ℕ := Nat.choose total_dots chosen_dots

/-- The number of collinear sets of 4 dots in the grid -/
def collinear_sets : ℕ := 28

/-- The probability of choosing 4 collinear dots from a 5x5 grid -/
theorem collinear_probability : 
  (collinear_sets : ℚ) / total_choices = 14 / 6325 := by sorry

end collinear_probability_l44_4400


namespace ages_sum_l44_4409

theorem ages_sum (a b c : ℕ) : 
  a = 20 + 2 * (b + c) →
  a^2 = 1980 + 3 * (b + c)^2 →
  a + b + c = 68 := by
sorry

end ages_sum_l44_4409


namespace range_of_a_l44_4450

/-- Custom binary operation on real numbers -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of 'a' given the inequality condition -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end range_of_a_l44_4450


namespace jennas_number_l44_4419

theorem jennas_number (x : ℝ) : 3 * ((3 * x + 20) - 5) = 225 → x = 20 := by
  sorry

end jennas_number_l44_4419


namespace isosceles_triangle_l44_4485

/-- If in a triangle with sides a and b, and their opposite angles α and β, 
    the equation a / cos(α) = b / cos(β) holds, then a = b. -/
theorem isosceles_triangle (a b α β : Real) : 
  0 < a ∧ 0 < b ∧ 0 < α ∧ α < π ∧ 0 < β ∧ β < π →  -- Ensuring valid triangle
  a / Real.cos α = b / Real.cos β →
  a = b :=
by sorry

end isosceles_triangle_l44_4485


namespace problem_statement_l44_4491

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x
noncomputable def g (x : ℝ) : ℝ := Real.log x + x + 1

theorem problem_statement :
  (∀ x : ℝ, f x > 0) ∧
  (∃ x₀ : ℝ, x₀ > 0 ∧ g x₀ = 0) ∧
  (¬(∀ x : ℝ, f x > 0) ↔ (∃ x₀ : ℝ, f x₀ ≤ 0)) ∧
  (¬(∃ x₀ : ℝ, x₀ > 0 ∧ g x₀ = 0) ↔ (∀ x : ℝ, x > 0 → g x ≠ 0)) :=
by sorry

end problem_statement_l44_4491


namespace remainder_problem_l44_4465

theorem remainder_problem (N : ℤ) (h : N % 221 = 43) : N % 17 = 9 := by
  sorry

end remainder_problem_l44_4465


namespace sector_central_angle_l44_4478

theorem sector_central_angle (arc_length radius : ℝ) (h1 : arc_length = 2 * Real.pi) (h2 : radius = 2) :
  arc_length / radius = Real.pi := by
  sorry

end sector_central_angle_l44_4478


namespace power_boat_travel_time_l44_4436

/-- Represents the scenario of a power boat and raft traveling on a river -/
structure RiverTravel where
  boatSpeed : ℝ  -- Speed of the power boat relative to the river
  riverSpeed : ℝ  -- Speed of the river current
  totalTime : ℝ  -- Total time until the boat meets the raft after returning
  travelTime : ℝ  -- Time taken by the boat to travel from A to B

/-- The conditions of the river travel scenario -/
def riverTravelConditions (rt : RiverTravel) : Prop :=
  rt.riverSpeed = rt.boatSpeed / 2 ∧
  rt.totalTime = 12 ∧
  (rt.boatSpeed + rt.riverSpeed) * rt.travelTime + 
    (rt.boatSpeed - rt.riverSpeed) * (rt.totalTime - rt.travelTime) = 
    rt.riverSpeed * rt.totalTime

/-- The theorem stating that under the given conditions, 
    the travel time from A to B is 6 hours -/
theorem power_boat_travel_time 
  (rt : RiverTravel) 
  (h : riverTravelConditions rt) : 
  rt.travelTime = 6 := by
  sorry

end power_boat_travel_time_l44_4436


namespace number_division_problem_l44_4411

theorem number_division_problem (N : ℝ) (x : ℝ) : 
  ((N - 34) / 10 = 2) → ((N - 5) / x = 7) → x = 7 := by
  sorry

end number_division_problem_l44_4411


namespace first_method_is_simple_random_second_method_is_systematic_l44_4464

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a student in the survey --/
structure Student where
  id : Nat
  deriving Repr

/-- Represents the survey setup --/
structure Survey where
  totalStudents : Nat
  selectedStudents : Nat
  selectionCriteria : Student → Bool

/-- Determines the sampling method based on the survey setup --/
def determineSamplingMethod (s : Survey) : SamplingMethod :=
  sorry

/-- The first survey method --/
def firstSurvey : Survey :=
  { totalStudents := 200
  , selectedStudents := 20
  , selectionCriteria := λ _ => true }

/-- The second survey method --/
def secondSurvey : Survey :=
  { totalStudents := 200
  , selectedStudents := 20
  , selectionCriteria := λ student => student.id % 10 = 2 }

/-- Theorem stating that the first method is simple random sampling --/
theorem first_method_is_simple_random :
  determineSamplingMethod firstSurvey = SamplingMethod.SimpleRandom :=
  sorry

/-- Theorem stating that the second method is systematic sampling --/
theorem second_method_is_systematic :
  determineSamplingMethod secondSurvey = SamplingMethod.Systematic :=
  sorry

end first_method_is_simple_random_second_method_is_systematic_l44_4464


namespace inequality_proof_equality_conditions_l44_4406

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  x * (1 - 2*x) * (1 - 3*x) + y * (1 - 2*y) * (1 - 3*y) + z * (1 - 2*z) * (1 - 3*z) ≥ 0 :=
sorry

theorem equality_conditions (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  x * (1 - 2*x) * (1 - 3*x) + y * (1 - 2*y) * (1 - 3*y) + z * (1 - 2*z) * (1 - 3*z) = 0 ↔
  ((x = 0 ∧ y = 1/2 ∧ z = 1/2) ∨
   (y = 0 ∧ z = 1/2 ∧ x = 1/2) ∨
   (z = 0 ∧ x = 1/2 ∧ y = 1/2) ∨
   (x = 1/3 ∧ y = 1/3 ∧ z = 1/3)) :=
sorry

end inequality_proof_equality_conditions_l44_4406


namespace sequence_prime_value_l44_4496

theorem sequence_prime_value (p : ℕ) (a : ℕ → ℤ) : 
  Prime p →
  a 0 = 0 →
  a 1 = 1 →
  (∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - p * a n) →
  (∃ m : ℕ, a m = -1) →
  p = 5 := by
sorry

end sequence_prime_value_l44_4496


namespace negation_of_proposition_l44_4422

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, ∃ n₀ : ℤ, n₀ ≤ x^2) ↔ (∃ x₀ : ℝ, ∀ n : ℤ, n > x₀^2) := by
  sorry

end negation_of_proposition_l44_4422


namespace unique_solution_system_l44_4446

theorem unique_solution_system (x : ℝ) : 
  (3 * x^2 + 8 * x - 3 = 0 ∧ 3 * x^4 + 2 * x^3 - 10 * x^2 + 30 * x - 9 = 0) ↔ x = -3 :=
by sorry

end unique_solution_system_l44_4446


namespace intersection_line_intersection_distance_l44_4454

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

-- Define the line
def line (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
  circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
  A ≠ B

-- Theorem for the line equation
theorem intersection_line (A B : ℝ × ℝ) (h : intersection_points A B) :
  line A.1 A.2 ∧ line B.1 B.2 :=
sorry

-- Theorem for the distance between intersection points
theorem intersection_distance (A B : ℝ × ℝ) (h : intersection_points A B) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5 * Real.sqrt 2 :=
sorry

end intersection_line_intersection_distance_l44_4454


namespace minimum_score_needed_l44_4480

def current_scores : List ℕ := [90, 80, 70, 60, 85]
def num_current_tests : ℕ := current_scores.length
def sum_current_scores : ℕ := current_scores.sum
def current_average : ℚ := sum_current_scores / num_current_tests
def target_increase : ℚ := 3
def num_total_tests : ℕ := num_current_tests + 1

theorem minimum_score_needed (min_score : ℕ) : 
  (sum_current_scores + min_score) / num_total_tests ≥ current_average + target_increase ∧
  ∀ (score : ℕ), score < min_score → 
    (sum_current_scores + score) / num_total_tests < current_average + target_increase →
  min_score = 95 := by
  sorry

end minimum_score_needed_l44_4480


namespace sum_of_y_values_is_0_04_l44_4429

-- Define the function g
def g (x : ℝ) : ℝ := (5*x)^2 - 5*x + 2

-- State the theorem
theorem sum_of_y_values_is_0_04 :
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ g y₁ = 12 ∧ g y₂ = 12 ∧ y₁ + y₂ = 0.04 ∧
  ∀ y : ℝ, g y = 12 → y = y₁ ∨ y = y₂ := by
  sorry

end sum_of_y_values_is_0_04_l44_4429


namespace largest_cube_surface_area_l44_4444

-- Define the dimensions of the cuboid
def cuboid_width : ℝ := 12
def cuboid_length : ℝ := 16
def cuboid_height : ℝ := 14

-- Define the function to calculate the surface area of a cube
def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length^2

-- Theorem statement
theorem largest_cube_surface_area :
  let max_side_length := min cuboid_width (min cuboid_length cuboid_height)
  cube_surface_area max_side_length = 864 := by
sorry

end largest_cube_surface_area_l44_4444


namespace function_increasing_range_l44_4489

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then Real.log x / Real.log a else a * x - 2

theorem function_increasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ 
  1 < a ∧ a ≤ 2 :=
sorry

end function_increasing_range_l44_4489


namespace apps_deleted_proof_l44_4431

/-- The number of apps Dave had at the start -/
def initial_apps : ℕ := 23

/-- The number of apps Dave had after deleting some -/
def remaining_apps : ℕ := 5

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := initial_apps - remaining_apps

theorem apps_deleted_proof : deleted_apps = 18 := by
  sorry

end apps_deleted_proof_l44_4431


namespace circular_seating_nine_seven_l44_4421

/-- The number of ways to choose 7 people from 9 and seat them around a circular table -/
def circular_seating_arrangements (total_people : ℕ) (seats : ℕ) : ℕ :=
  (total_people.choose (total_people - seats)) * (seats - 1).factorial

theorem circular_seating_nine_seven :
  circular_seating_arrangements 9 7 = 25920 := by
  sorry

end circular_seating_nine_seven_l44_4421


namespace arithmetic_sequence_terms_count_l44_4441

theorem arithmetic_sequence_terms_count (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 15 → aₙ = 99 → d = 4 → a₁ + (n - 1) * d = aₙ → n = 22 := by
  sorry

end arithmetic_sequence_terms_count_l44_4441


namespace decimal_arithmetic_l44_4473

theorem decimal_arithmetic : 3.456 - 1.78 + 0.032 = 1.678 := by
  sorry

end decimal_arithmetic_l44_4473


namespace unique_modular_solution_l44_4479

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -1453 [ZMOD 10] := by
  sorry

end unique_modular_solution_l44_4479


namespace iron_conducts_electricity_is_deductive_reasoning_l44_4484

-- Define the set of all objects
variable (Object : Type)

-- Define the property of being a metal
variable (is_metal : Object → Prop)

-- Define the property of conducting electricity
variable (conducts_electricity : Object → Prop)

-- Define iron as an object
variable (iron : Object)

-- Theorem statement
theorem iron_conducts_electricity_is_deductive_reasoning
  (h1 : ∀ x, is_metal x → conducts_electricity x)  -- All metals conduct electricity
  (h2 : is_metal iron)                             -- Iron is a metal
  : conducts_electricity iron                      -- Therefore, iron conducts electricity
  := by sorry

-- The fact that this can be proved using only the given premises
-- demonstrates that this is deductive reasoning

end iron_conducts_electricity_is_deductive_reasoning_l44_4484


namespace cos_thirty_degrees_l44_4497

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end cos_thirty_degrees_l44_4497


namespace M_intersect_N_eq_interval_l44_4408

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2}
def N : Set ℝ := {x | ∃ y, x^2 + y^2 = 2}

-- Define the interval [0, √2]
def interval : Set ℝ := {x | 0 ≤ x ∧ x ≤ Real.sqrt 2}

-- State the theorem
theorem M_intersect_N_eq_interval : M ∩ N = interval := by sorry

end M_intersect_N_eq_interval_l44_4408


namespace empty_solution_set_non_empty_solution_set_l44_4437

-- Define the inequality
def inequality (x a : ℝ) : Prop := |x - 4| + |3 - x| < a

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | inequality x a}

-- Theorem for the empty solution set case
theorem empty_solution_set (a : ℝ) :
  solution_set a = ∅ ↔ a ≤ 1 :=
sorry

-- Theorem for the non-empty solution set case
theorem non_empty_solution_set (a : ℝ) :
  solution_set a ≠ ∅ ↔ a > 1 :=
sorry

end empty_solution_set_non_empty_solution_set_l44_4437


namespace doughnut_cost_theorem_l44_4488

def total_cost (chocolate_count : ℕ) (glazed_count : ℕ) (maple_count : ℕ) (strawberry_count : ℕ)
               (chocolate_price : ℚ) (glazed_price : ℚ) (maple_price : ℚ) (strawberry_price : ℚ)
               (chocolate_discount : ℚ) (maple_discount : ℚ) (free_glazed : ℕ) : ℚ :=
  let chocolate_cost := (chocolate_count : ℚ) * chocolate_price * (1 - chocolate_discount)
  let glazed_cost := (glazed_count : ℚ) * glazed_price
  let maple_cost := (maple_count : ℚ) * maple_price * (1 - maple_discount)
  let strawberry_cost := (strawberry_count : ℚ) * strawberry_price
  let free_glazed_savings := (free_glazed : ℚ) * glazed_price
  chocolate_cost + glazed_cost + maple_cost + strawberry_cost - free_glazed_savings

theorem doughnut_cost_theorem :
  total_cost 10 8 5 2 2 1 (3/2) (5/2) (15/100) (1/10) 1 = 143/4 := by
  sorry

end doughnut_cost_theorem_l44_4488


namespace sixth_term_value_l44_4476

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sixth_term_value (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : arithmetic_sequence a)
  (h3 : ∀ n : ℕ, a (n + 1) - a n = 2) :
  a 6 = 11 := by
  sorry

end sixth_term_value_l44_4476


namespace bella_score_l44_4455

theorem bella_score (n : ℕ) (avg_without : ℚ) (avg_with : ℚ) (bella_score : ℚ) : 
  n = 18 →
  avg_without = 75 →
  avg_with = 76 →
  bella_score = (n * avg_with) - ((n - 1) * avg_without) →
  bella_score = 93 := by
  sorry

end bella_score_l44_4455


namespace parallel_vectors_angle_l44_4468

theorem parallel_vectors_angle (θ : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (h_parallel : ∃ (k : Real), k ≠ 0 ∧ (1 - Real.sin θ, 1) = k • (1/2, 1 + Real.sin θ)) :
  θ = π / 4 := by
  sorry

end parallel_vectors_angle_l44_4468


namespace inequality_solution_l44_4426

theorem inequality_solution (x y : ℝ) : 
  Real.sqrt 3 * Real.tan x - (Real.sin y) ^ (1/4) - 
  Real.sqrt ((3 / (Real.cos x)^2) + Real.sqrt (Real.sin y) - 6) ≥ Real.sqrt 3 ↔ 
  ∃ (n k : ℤ), x = π/4 + n*π ∧ y = k*π :=
by sorry

end inequality_solution_l44_4426


namespace inequality_proof_l44_4458

theorem inequality_proof (a b : ℝ) (h : (6*a + 9*b)/(a + b) < (4*a - b)/(a - b)) :
  abs b < abs a ∧ abs a < 2 * abs b :=
by sorry

end inequality_proof_l44_4458


namespace fraction_evaluation_l44_4483

theorem fraction_evaluation : (5 * 7) / 10 = 3.5 := by sorry

end fraction_evaluation_l44_4483


namespace train_length_l44_4415

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 126 → time_s = 16 → speed_kmh * (5/18) * time_s = 560 := by
  sorry

#check train_length

end train_length_l44_4415


namespace plane_distance_proof_l44_4416

-- Define the plane's speed in still air
def plane_speed : ℝ := 262.5

-- Define the time taken with tail wind
def time_with_wind : ℝ := 3

-- Define the time taken against wind
def time_against_wind : ℝ := 4

-- Define the wind speed (to be solved)
def wind_speed : ℝ := 37.5

-- Define the distance (to be proved)
def distance : ℝ := 900

-- Theorem statement
theorem plane_distance_proof :
  distance = (plane_speed + wind_speed) * time_with_wind ∧
  distance = (plane_speed - wind_speed) * time_against_wind :=
by sorry

end plane_distance_proof_l44_4416


namespace complex_to_polar_l44_4466

open Complex

theorem complex_to_polar (θ : ℝ) : 
  (1 + Real.sin θ + I * Real.cos θ) / (1 + Real.sin θ - I * Real.cos θ) = 
  Complex.exp (I * (π / 2 - θ)) :=
sorry

end complex_to_polar_l44_4466


namespace system_solution_l44_4433

theorem system_solution (x y : ℝ) : 
  (x + y = 5 ∧ 2 * x - 3 * y = 20) ↔ (x = 7 ∧ y = -2) := by
  sorry

end system_solution_l44_4433


namespace volume_to_surface_area_ratio_l44_4460

/-- Represents the shape created by arranging 8 unit cubes -/
structure CubeShape where
  center_cube : Unit
  surrounding_cubes : Fin 6 → Unit
  top_cube : Unit

/-- Calculates the volume of the CubeShape -/
def volume (shape : CubeShape) : ℕ := 8

/-- Calculates the surface area of the CubeShape -/
def surface_area (shape : CubeShape) : ℕ := 28

/-- Theorem stating that the ratio of volume to surface area is 2/7 -/
theorem volume_to_surface_area_ratio (shape : CubeShape) :
  (volume shape : ℚ) / (surface_area shape : ℚ) = 2 / 7 := by
  sorry

end volume_to_surface_area_ratio_l44_4460


namespace chess_class_percentage_l44_4495

/-- Proves that 20% of students attend chess class given the conditions of the problem -/
theorem chess_class_percentage (total_students : ℕ) (swimming_students : ℕ) 
  (h1 : total_students = 1000)
  (h2 : swimming_students = 20)
  (h3 : ∀ (chess_percentage : ℚ), 
    chess_percentage * total_students * (1/10) = swimming_students) :
  ∃ (chess_percentage : ℚ), chess_percentage = 1/5 := by
  sorry

end chess_class_percentage_l44_4495


namespace wills_game_cost_l44_4412

/-- Proves that the cost of Will's new game is $47 --/
theorem wills_game_cost (initial_money : ℕ) (num_toys : ℕ) (toy_price : ℕ) (game_cost : ℕ) : 
  initial_money = 83 →
  num_toys = 9 →
  toy_price = 4 →
  initial_money = game_cost + (num_toys * toy_price) →
  game_cost = 47 := by
sorry

end wills_game_cost_l44_4412


namespace winter_clothing_boxes_l44_4472

/-- Given that each box contains 10 pieces of clothing and the total number of pieces is 60,
    prove that the number of boxes is 6. -/
theorem winter_clothing_boxes (pieces_per_box : ℕ) (total_pieces : ℕ) (num_boxes : ℕ) :
  pieces_per_box = 10 →
  total_pieces = 60 →
  num_boxes * pieces_per_box = total_pieces →
  num_boxes = 6 :=
by
  sorry

end winter_clothing_boxes_l44_4472


namespace exactly_one_true_proposition_l44_4462

open Real

theorem exactly_one_true_proposition :
  let prop1 := ∀ x : ℝ, x^4 > x^2
  let prop2 := ∀ p q : Prop, (¬(p ∧ q)) → (¬p ∧ ¬q)
  let prop3 := (¬∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0)
  (¬prop1 ∧ ¬prop2 ∧ prop3) :=
by sorry

#check exactly_one_true_proposition

end exactly_one_true_proposition_l44_4462


namespace segment_equality_l44_4404

/-- Given points A, B, C, D, E, F on a line with certain distance relationships,
    prove that CD = AB = EF. -/
theorem segment_equality 
  (A B C D E F : ℝ) -- Points on a line represented as real numbers
  (h1 : A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F) -- Points are ordered on the line
  (h2 : C - A = E - C) -- AC = CE
  (h3 : D - B = F - D) -- BD = DF
  (h4 : D - A = F - C) -- AD = CF
  : D - C = B - A ∧ D - C = F - E := by
  sorry

end segment_equality_l44_4404


namespace impossible_to_achieve_goal_state_l44_4407

/-- Represents a jar with a certain volume of tea and amount of sugar -/
structure Jar where
  volume : ℕ
  sugar : ℕ

/-- Represents the state of the system with three jars -/
structure SystemState where
  jar1 : Jar
  jar2 : Jar
  jar3 : Jar

/-- Represents a transfer of tea between jars -/
inductive Transfer where
  | from1to2 : Transfer
  | from1to3 : Transfer
  | from2to1 : Transfer
  | from2to3 : Transfer
  | from3to1 : Transfer
  | from3to2 : Transfer

def initial_state : SystemState :=
  { jar1 := { volume := 0, sugar := 0 },
    jar2 := { volume := 700, sugar := 50 },
    jar3 := { volume := 800, sugar := 60 } }

def transfer_amount : ℕ := 100

def is_valid_state (s : SystemState) : Prop :=
  s.jar1.volume + s.jar2.volume + s.jar3.volume = 1500 ∧
  s.jar1.volume % transfer_amount = 0 ∧
  s.jar2.volume % transfer_amount = 0 ∧
  s.jar3.volume % transfer_amount = 0

def apply_transfer (s : SystemState) (t : Transfer) : SystemState :=
  sorry

def is_goal_state (s : SystemState) : Prop :=
  s.jar1.volume = 0 ∧ s.jar2.sugar = s.jar3.sugar

theorem impossible_to_achieve_goal_state :
  ∀ (transfers : List Transfer),
    let final_state := transfers.foldl apply_transfer initial_state
    is_valid_state final_state → ¬is_goal_state final_state :=
  sorry

end impossible_to_achieve_goal_state_l44_4407


namespace sequence_problem_l44_4434

/-- Given a sequence {aₙ}, prove that a₁₉ = 1/16 under specific conditions -/
theorem sequence_problem (a : ℕ → ℚ) : 
  (a 4 = 1) → 
  (a 6 = 1/3) → 
  (∃ d : ℚ, ∀ n : ℕ, 1/(a (n+1)) - 1/(a n) = d) → 
  (a 19 = 1/16) := by
sorry

end sequence_problem_l44_4434


namespace equality_implies_a_equals_two_l44_4418

theorem equality_implies_a_equals_two (a : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x + a = (x + 1)*(x + 2)) → a = 2 := by
sorry

end equality_implies_a_equals_two_l44_4418


namespace wrapping_paper_area_theorem_l44_4452

/-- Represents a box with a square base -/
structure Box where
  side : ℝ
  height : ℝ

/-- Represents the wrapping paper -/
structure WrappingPaper where
  side : ℝ

/-- Calculates the area of the wrapping paper needed to wrap the box -/
def wrappingPaperArea (b : Box) (w : WrappingPaper) : ℝ :=
  w.side * w.side

/-- Theorem stating the area of wrapping paper needed -/
theorem wrapping_paper_area_theorem (s : ℝ) (h : s > 0) :
  let b : Box := { side := 2 * s, height := 3 * s }
  let w : WrappingPaper := { side := 4 * s }
  wrappingPaperArea b w = 24 * s^2 := by
  sorry

#check wrapping_paper_area_theorem

end wrapping_paper_area_theorem_l44_4452


namespace plant_order_after_365_days_l44_4456

-- Define the plants
inductive Plant
| Cactus
| Dieffenbachia
| Orchid

-- Define the order of plants
def PlantOrder := List Plant

-- Define the initial order
def initialOrder : PlantOrder := [Plant.Cactus, Plant.Dieffenbachia, Plant.Orchid]

-- Define Luna's swap operation (left and center)
def lunaSwap (order : PlantOrder) : PlantOrder :=
  match order with
  | [a, b, c] => [b, a, c]
  | _ => order

-- Define Sam's swap operation (right and center)
def samSwap (order : PlantOrder) : PlantOrder :=
  match order with
  | [a, b, c] => [a, c, b]
  | _ => order

-- Define a single day's operation (Luna's swap followed by Sam's swap)
def dailyOperation (order : PlantOrder) : PlantOrder :=
  samSwap (lunaSwap order)

-- Define the operation for multiple days
def multiDayOperation (order : PlantOrder) (days : Nat) : PlantOrder :=
  match days with
  | 0 => order
  | n + 1 => multiDayOperation (dailyOperation order) n

-- Theorem to prove
theorem plant_order_after_365_days :
  multiDayOperation initialOrder 365 = [Plant.Orchid, Plant.Cactus, Plant.Dieffenbachia] :=
sorry

end plant_order_after_365_days_l44_4456


namespace sphere_radius_when_area_equals_volume_l44_4474

theorem sphere_radius_when_area_equals_volume :
  ∀ R : ℝ,
  R > 0 →
  (4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3) →
  R = 3 := by
sorry

end sphere_radius_when_area_equals_volume_l44_4474


namespace spinner_sector_areas_l44_4410

/-- Represents a circular spinner with WIN and BONUS sectors -/
structure Spinner where
  radius : ℝ
  win_prob : ℝ
  bonus_prob : ℝ

/-- Calculates the area of a sector given its probability and the total area -/
def sector_area (prob : ℝ) (total_area : ℝ) : ℝ := prob * total_area

/-- Theorem stating the areas of WIN and BONUS sectors for a specific spinner -/
theorem spinner_sector_areas (s : Spinner) 
  (h_radius : s.radius = 15)
  (h_win_prob : s.win_prob = 1/3)
  (h_bonus_prob : s.bonus_prob = 1/4) :
  let total_area := π * s.radius^2
  sector_area s.win_prob total_area = 75 * π ∧ 
  sector_area s.bonus_prob total_area = 56.25 * π := by
  sorry


end spinner_sector_areas_l44_4410


namespace five_digit_divisibility_count_l44_4490

/-- The count of 5-digit numbers with a specific divisibility property -/
theorem five_digit_divisibility_count : 
  (Finset.filter 
    (fun n : ℕ => 
      10000 ≤ n ∧ n ≤ 99999 ∧ 
      (n / 50 + n % 50) % 7 = 0)
    (Finset.range 100000)).card = 14400 := by
  sorry

end five_digit_divisibility_count_l44_4490


namespace binomial_15_choose_3_l44_4463

theorem binomial_15_choose_3 : Nat.choose 15 3 = 455 := by
  sorry

end binomial_15_choose_3_l44_4463


namespace unique_integer_satisfying_equation_l44_4471

theorem unique_integer_satisfying_equation : 
  ∃! (n : ℕ), n > 0 ∧ (n + 1500) / 90 = ⌊Real.sqrt n⌋ ∧ n = 4530 := by
  sorry

end unique_integer_satisfying_equation_l44_4471


namespace billys_coin_piles_l44_4481

theorem billys_coin_piles (x : ℕ) : 
  (x + 3) * 4 = 20 → x = 2 := by
  sorry

end billys_coin_piles_l44_4481


namespace bag_original_price_l44_4402

/-- Given a bag sold for $120 after a 20% discount, prove its original price was $150 -/
theorem bag_original_price (discounted_price : ℝ) (discount_rate : ℝ) : 
  discounted_price = 120 → 
  discount_rate = 0.2 → 
  discounted_price = (1 - discount_rate) * 150 := by
sorry

end bag_original_price_l44_4402


namespace tyrones_dimes_l44_4494

/-- Given Tyrone's coin collection and total money, prove the number of dimes he has. -/
theorem tyrones_dimes (value_without_dimes : ℚ) (total_value : ℚ) (dime_value : ℚ) :
  value_without_dimes = 11 →
  total_value = 13 →
  dime_value = 1 / 10 →
  (total_value - value_without_dimes) / dime_value = 20 :=
by sorry

end tyrones_dimes_l44_4494


namespace gcd_problem_l44_4492

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = (2 * k + 1) * 8723) :
  Int.gcd (8 * b^2 + 55 * b + 144) (4 * b + 15) = 3 := by
  sorry

end gcd_problem_l44_4492


namespace triangle_problem_l44_4414

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a / b = 7 / 5 →
  b / c = 5 / 3 →
  (1 / 2) * b * c * Real.sin A = 45 * Real.sqrt 3 →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →
  (2 * R * Real.sin A = a) →
  Real.cos A = -1 / 2 ∧ R = 14 := by
  sorry

end triangle_problem_l44_4414


namespace multiplication_as_difference_of_squares_l44_4439

theorem multiplication_as_difference_of_squares :
  ∀ a b : ℚ,
  (19 + 2/3) * (20 + 1/3) = (a - b) * (a + b) →
  a = 20 ∧ b = 1/3 := by
sorry

end multiplication_as_difference_of_squares_l44_4439


namespace quadratic_equation_solution_l44_4440

theorem quadratic_equation_solution :
  ∃ (a b : ℝ),
    (∀ x : ℝ, x^2 - 4*x + 9 = 25 ↔ (x = a ∨ x = b)) ∧
    a ≥ b ∧
    3*a + 2*b = 10 + 2*Real.sqrt 5 :=
by sorry

end quadratic_equation_solution_l44_4440


namespace simplify_expression_l44_4467

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : a^3 + b^3 = a + b) (h2 : a^2 + b^2 = 3*a + b) :
  a/b + b/a + 1/(a*b) = (9*a + 3*b + 3)/(3*a + b - 1) := by
  sorry

end simplify_expression_l44_4467


namespace eat_cereal_together_l44_4459

/-- The time needed for two people to eat a certain amount of cereal together -/
def time_to_eat_together (fat_rate : ℚ) (thin_rate : ℚ) (total_amount : ℚ) : ℚ :=
  total_amount / (fat_rate + thin_rate)

/-- Theorem stating the time needed for Mr. Fat and Mr. Thin to eat 5 pounds of cereal together -/
theorem eat_cereal_together :
  let fat_rate : ℚ := 1 / 12
  let thin_rate : ℚ := 1 / 40
  let total_amount : ℚ := 5
  time_to_eat_together fat_rate thin_rate total_amount = 600 / 13 := by sorry

end eat_cereal_together_l44_4459
