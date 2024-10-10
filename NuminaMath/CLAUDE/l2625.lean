import Mathlib

namespace total_toy_count_l2625_262586

/-- The number of toys each person has -/
structure ToyCount where
  jaxon : ℕ
  gabriel : ℕ
  jerry : ℕ

/-- The conditions of the problem -/
def problem_conditions (tc : ToyCount) : Prop :=
  tc.jaxon = 15 ∧
  tc.gabriel = 2 * tc.jaxon ∧
  tc.jerry = tc.gabriel + 8

/-- The theorem to prove -/
theorem total_toy_count (tc : ToyCount) 
  (h : problem_conditions tc) : tc.jaxon + tc.gabriel + tc.jerry = 83 := by
  sorry


end total_toy_count_l2625_262586


namespace math_majors_consecutive_probability_l2625_262576

def total_people : ℕ := 10
def math_majors : ℕ := 5
def physics_majors : ℕ := 3
def chemistry_majors : ℕ := 2

theorem math_majors_consecutive_probability :
  let total_arrangements := Nat.choose total_people math_majors
  let consecutive_arrangements := total_people
  (consecutive_arrangements : ℚ) / total_arrangements = 5 / 126 := by
  sorry

end math_majors_consecutive_probability_l2625_262576


namespace optimization_problem_l2625_262567

theorem optimization_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 1 ∧ 2 * x₀ * y₀ = 1/4) ∧
  (∀ (x₁ y₁ : ℝ), x₁ > 0 → y₁ > 0 → 2 * x₁ + y₁ = 1 → 2 * x₁ * y₁ ≤ 1/4) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 1 ∧ 4 * x₀^2 + y₀^2 = 1/2) ∧
  (∀ (x₁ y₁ : ℝ), x₁ > 0 → y₁ > 0 → 2 * x₁ + y₁ = 1 → 4 * x₁^2 + y₁^2 ≥ 1/2) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 1 ∧ 1/x₀ + 1/y₀ = 3 + 2 * Real.sqrt 2) ∧
  (∀ (x₁ y₁ : ℝ), x₁ > 0 → y₁ > 0 → 2 * x₁ + y₁ = 1 → 1/x₁ + 1/y₁ ≥ 3 + 2 * Real.sqrt 2) :=
by sorry

end optimization_problem_l2625_262567


namespace product_sum_fractions_l2625_262512

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end product_sum_fractions_l2625_262512


namespace simplify_equation_l2625_262509

theorem simplify_equation : ∀ x : ℝ, 
  3 * x + 4.8 * x - 10 * x = 11 * (1 / 5) ↔ -2.2 * x = 2.2 := by
  sorry

end simplify_equation_l2625_262509


namespace international_data_daily_cost_l2625_262526

def regular_plan_cost : ℚ := 175
def total_charges : ℚ := 210
def stay_duration : ℕ := 10

theorem international_data_daily_cost : 
  (total_charges - regular_plan_cost) / stay_duration = (35 : ℚ) / 10 := by
  sorry

end international_data_daily_cost_l2625_262526


namespace distance_downstream_20min_l2625_262516

/-- Calculates the distance traveled downstream by a boat -/
def distance_traveled (boat_speed wind_speed current_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + current_speed + 0.1 * wind_speed) * time

/-- Theorem: Distance traveled downstream in 20 minutes -/
theorem distance_downstream_20min (c w : ℝ) :
  distance_traveled 26 w c (1/3) = 26/3 + c/3 + 0.1*w/3 := by
  sorry

end distance_downstream_20min_l2625_262516


namespace intramural_teams_l2625_262520

theorem intramural_teams (num_boys : ℕ) (num_girls : ℕ) (max_teams : ℕ) :
  num_boys = 32 →
  max_teams = 8 →
  (∃ (boys_per_team : ℕ), num_boys = max_teams * boys_per_team) →
  (∃ (girls_per_team : ℕ), num_girls = max_teams * girls_per_team) →
  ∃ (k : ℕ), num_girls = 8 * k :=
by sorry

end intramural_teams_l2625_262520


namespace blanket_collection_l2625_262515

/-- Proves the number of blankets collected on the last day of a three-day collection drive -/
theorem blanket_collection (team_size : ℕ) (first_day_per_person : ℕ) (total_blankets : ℕ) : 
  team_size = 15 → 
  first_day_per_person = 2 → 
  total_blankets = 142 → 
  total_blankets - (team_size * first_day_per_person + 3 * (team_size * first_day_per_person)) = 22 := by
  sorry

#check blanket_collection

end blanket_collection_l2625_262515


namespace average_of_three_numbers_l2625_262559

theorem average_of_three_numbers (x : ℝ) : 
  (2 + 76 + x) / 3 = 5 → x = -63 := by
  sorry

end average_of_three_numbers_l2625_262559


namespace square_of_sum_l2625_262534

theorem square_of_sum (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 := by
  sorry

end square_of_sum_l2625_262534


namespace cube_sum_from_sum_and_product_l2625_262581

theorem cube_sum_from_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x * y = 15) : x^3 + y^3 = 550 := by
  sorry

end cube_sum_from_sum_and_product_l2625_262581


namespace broker_commission_slump_l2625_262539

theorem broker_commission_slump (X : ℝ) (h : X > 0) :
  let Y : ℝ := (4 / 5) * X
  let income_unchanged := 0.04 * X = 0.05 * Y
  let slump_percentage := (1 - Y / X) * 100
  income_unchanged → slump_percentage = 20 := by
sorry

end broker_commission_slump_l2625_262539


namespace walk_distance_proof_l2625_262599

def walk_duration : ℝ := 5
def min_speed : ℝ := 3
def max_speed : ℝ := 4

def possible_distance (d : ℝ) : Prop :=
  ∃ (speed : ℝ), min_speed ≤ speed ∧ speed ≤ max_speed ∧ d = speed * walk_duration

theorem walk_distance_proof :
  possible_distance 19 ∧
  ¬ possible_distance 12 ∧
  ¬ possible_distance 14 ∧
  ¬ possible_distance 24 ∧
  ¬ possible_distance 35 :=
by sorry

end walk_distance_proof_l2625_262599


namespace b_share_proof_l2625_262552

/-- The number of days B takes to complete the work alone -/
def b_days : ℕ := 10

/-- The number of days A takes to complete the work alone -/
def a_days : ℕ := 15

/-- The total wages for the work in Rupees -/
def total_wages : ℕ := 5000

/-- The share of wages B should receive when working together with A -/
def b_share : ℕ := 3000

/-- Theorem stating that B's share of the wages when working with A is 3000 Rupees -/
theorem b_share_proof : 
  b_share = (b_days * total_wages) / (a_days + b_days) := by sorry

end b_share_proof_l2625_262552


namespace cyclist_speed_problem_l2625_262583

theorem cyclist_speed_problem (x y : ℝ) :
  y = x + 5 ∧  -- Y's speed is 5 mph faster than X's
  100 / y + 1/6 + 20 / y = 80 / x + 1/4 ∧  -- Time equality equation
  x > 0 ∧ y > 0  -- Positive speeds
  → x = 10 :=
by sorry

end cyclist_speed_problem_l2625_262583


namespace lianliang_run_distance_l2625_262561

/-- The length of the playground in meters -/
def playground_length : ℕ := 110

/-- The difference between the length and width of the playground in meters -/
def length_width_difference : ℕ := 15

/-- The width of the playground in meters -/
def playground_width : ℕ := playground_length - length_width_difference

/-- The perimeter of the playground in meters -/
def playground_perimeter : ℕ := (playground_length + playground_width) * 2

theorem lianliang_run_distance : playground_perimeter = 230 := by
  sorry

end lianliang_run_distance_l2625_262561


namespace unique_number_satisfying_conditions_l2625_262591

def digit_product (n : ℕ) : ℕ := 
  if n = 0 then 1 else (n % 10) * digit_product (n / 10)

def digit_sum (n : ℕ) : ℕ := 
  if n = 0 then 0 else (n % 10) + digit_sum (n / 10)

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

theorem unique_number_satisfying_conditions : 
  ∃! x : ℕ, digit_product x = 44 * x - 86868 ∧ is_cube (digit_sum x) ∧ x = 1989 :=
sorry

end unique_number_satisfying_conditions_l2625_262591


namespace remarkable_number_l2625_262533

theorem remarkable_number : ∃ (x : ℝ), 
  x > 0 ∧ 
  (x - ⌊x⌋) * ⌊x⌋ = (x - ⌊x⌋)^2 ∧ 
  x = (1 + Real.sqrt 5) / 2 := by
  sorry

end remarkable_number_l2625_262533


namespace problem_1_problem_2_l2625_262510

-- Problem 1
theorem problem_1 (z : ℂ) (h : z = (Complex.I - 1) / Real.sqrt 2) :
  z^20 + z^10 + 1 = -Complex.I := by sorry

-- Problem 2
theorem problem_2 (z : ℂ) (h : Complex.abs (z - (3 + 4*Complex.I)) = 1) :
  4 ≤ Complex.abs z ∧ Complex.abs z ≤ 6 := by sorry

end problem_1_problem_2_l2625_262510


namespace new_person_weight_is_105_l2625_262570

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (leaving_weight : ℝ) (average_increase : ℝ) : ℝ :=
  leaving_weight + initial_count * average_increase

/-- Theorem stating that under the given conditions, the weight of the new person is 105 kg -/
theorem new_person_weight_is_105 :
  weight_of_new_person 8 85 2.5 = 105 := by
  sorry

#eval weight_of_new_person 8 85 2.5

end new_person_weight_is_105_l2625_262570


namespace decreasing_quadratic_l2625_262549

theorem decreasing_quadratic (m : ℝ) : 
  (∀ x₁ x₂ : ℤ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → 
    x₁^2 + (m-1)*x₁ + 1 > x₂^2 + (m-1)*x₂ + 1) ↔ 
  m ≤ -8 := by
  sorry

end decreasing_quadratic_l2625_262549


namespace quadrilateral_k_value_l2625_262558

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A quadrilateral formed by two lines and the positive semi-axes -/
structure Quadrilateral where
  l₁ : Line
  l₂ : Line

/-- Predicate to check if a quadrilateral has a circumscribed circle -/
def has_circumscribed_circle (q : Quadrilateral) : Prop :=
  sorry

/-- The quadrilateral formed by the given lines and axes -/
def quad (k : ℝ) : Quadrilateral :=
  { l₁ := { a := 1, b := 3, c := -7 },
    l₂ := { a := k, b := 1, c := -2 } }

theorem quadrilateral_k_value :
  ∀ k : ℝ, has_circumscribed_circle (quad k) → k = -3 := by
  sorry

end quadrilateral_k_value_l2625_262558


namespace painted_cube_problem_l2625_262517

theorem painted_cube_problem (n : ℕ) (h1 : n > 2) :
  (2 * (n - 2)^2 = 2 * (n - 2) * n^2) → n = 3 := by
  sorry

end painted_cube_problem_l2625_262517


namespace rose_needs_more_l2625_262522

def paintbrush_cost : ℚ := 2.40
def paints_cost : ℚ := 9.20
def easel_cost : ℚ := 6.50
def rose_has : ℚ := 7.10

theorem rose_needs_more : 
  paintbrush_cost + paints_cost + easel_cost - rose_has = 11 :=
by sorry

end rose_needs_more_l2625_262522


namespace solution_set_equivalence_l2625_262555

theorem solution_set_equivalence (x : ℝ) :
  (|(8 - x) / 4| < 3) ↔ (4 < x ∧ x < 20) := by
  sorry

end solution_set_equivalence_l2625_262555


namespace not_or_false_implies_and_or_l2625_262551

theorem not_or_false_implies_and_or (p q : Prop) :
  ¬(¬p ∨ ¬q) → (p ∧ q) ∧ (p ∨ q) := by
  sorry

end not_or_false_implies_and_or_l2625_262551


namespace inverse_function_value_l2625_262513

/-- Given that f is the inverse function of g(x) = ax, and f(4) = 2, prove that a = 2 -/
theorem inverse_function_value (a : ℝ) (f g : ℝ → ℝ) :
  (∀ x, g x = a * x) →  -- g is defined as g(x) = ax
  (∀ x, f (g x) = x) →  -- f is the inverse function of g
  (∀ x, g (f x) = x) →  -- f is the inverse function of g (reverse composition)
  f 4 = 2 →             -- f(4) = 2
  a = 2 := by
  sorry

end inverse_function_value_l2625_262513


namespace f_neg_three_l2625_262572

def f (x : ℝ) : ℝ := x^2 + x

theorem f_neg_three : f (-3) = 6 := by sorry

end f_neg_three_l2625_262572


namespace min_sum_at_24_l2625_262584

/-- Arithmetic sequence with general term a_n = 2n - 49 -/
def a (n : ℕ) : ℤ := 2 * n - 49

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := n * (a 1 + a n) / 2

theorem min_sum_at_24 :
  ∀ k : ℕ, k ≠ 0 → S 24 ≤ S k :=
sorry

end min_sum_at_24_l2625_262584


namespace number_puzzle_solution_l2625_262544

theorem number_puzzle_solution :
  ∃ x : ℚ, x^2 + 100 = (x - 12)^2 ∧ x = 11/6 := by
  sorry

end number_puzzle_solution_l2625_262544


namespace flour_per_cake_l2625_262580

/-- The amount of flour needed for each cake given the initial conditions -/
theorem flour_per_cake 
  (traci_flour : ℕ) 
  (harris_flour : ℕ) 
  (traci_cakes : ℕ) 
  (harris_cakes : ℕ) 
  (h1 : traci_flour = 500)
  (h2 : harris_flour = 400)
  (h3 : traci_cakes = 9)
  (h4 : harris_cakes = 9) :
  (traci_flour + harris_flour) / (traci_cakes + harris_cakes) = 50 := by
  sorry

end flour_per_cake_l2625_262580


namespace book_sale_loss_l2625_262548

/-- Given that the cost price of 15 books equals the selling price of 20 books,
    prove that there is a 25% loss. -/
theorem book_sale_loss (C S : ℝ) (h : 15 * C = 20 * S) : 
  (C - S) / C = 1 / 4 := by
  sorry

end book_sale_loss_l2625_262548


namespace gas_refill_amount_l2625_262502

/-- Calculates the amount of gas needed to refill a car's tank --/
theorem gas_refill_amount (initial_gas tank_capacity store_trip doctor_trip : ℝ) 
  (h1 : initial_gas = 10)
  (h2 : tank_capacity = 12)
  (h3 : store_trip = 6)
  (h4 : doctor_trip = 2) :
  tank_capacity - (initial_gas - store_trip - doctor_trip) = 10 := by
  sorry

end gas_refill_amount_l2625_262502


namespace heathers_oranges_l2625_262568

/-- The total number of oranges Heather has after receiving more from Russell -/
def total_oranges (initial : Float) (received : Float) : Float :=
  initial + received

/-- Theorem stating that Heather's total oranges is the sum of her initial oranges and those received from Russell -/
theorem heathers_oranges (initial : Float) (received : Float) :
  total_oranges initial received = initial + received := by
  sorry

end heathers_oranges_l2625_262568


namespace equation_solutions_l2625_262547

theorem equation_solutions :
  (∃! x : ℝ, (2 / x = 3 / (x + 2)) ∧ x = 4) ∧
  (¬ ∃ x : ℝ, 1 / (x - 2) = (1 - x) / (2 - x) - 3) :=
by sorry

end equation_solutions_l2625_262547


namespace min_milk_candies_l2625_262514

/-- Represents the number of chocolate candies -/
def chocolate : ℕ := sorry

/-- Represents the number of watermelon candies -/
def watermelon : ℕ := sorry

/-- Represents the number of milk candies -/
def milk : ℕ := sorry

/-- The number of watermelon candies is at most 3 times the number of chocolate candies -/
axiom watermelon_condition : watermelon ≤ 3 * chocolate

/-- The number of milk candies is at least 4 times the number of chocolate candies -/
axiom milk_condition : milk ≥ 4 * chocolate

/-- The total number of chocolate and watermelon candies is no less than 2020 -/
axiom total_condition : chocolate + watermelon ≥ 2020

/-- The minimum number of milk candies required is 2020 -/
theorem min_milk_candies : milk ≥ 2020 := by sorry

end min_milk_candies_l2625_262514


namespace problem_1_problem_2_problem_3_problem_4_l2625_262550

-- Problem 1
theorem problem_1 : 12 - (-1) + (-7) = 6 := by sorry

-- Problem 2
theorem problem_2 : -3.5 * (-3/4) / (7/8) = 3 := by sorry

-- Problem 3
theorem problem_3 : (1/3 - 1/6 - 1/12) * (-12) = -1 := by sorry

-- Problem 4
theorem problem_4 : (-2)^4 / (-4) * (-1/2)^2 - 1^2 = -2 := by sorry

end problem_1_problem_2_problem_3_problem_4_l2625_262550


namespace computer_table_cost_price_l2625_262540

/-- A furniture shop owner charges 10% more than the cost price. If a customer paid Rs. 8800 for a computer table, then the cost price of the computer table was Rs. 8000. -/
theorem computer_table_cost_price (selling_price : ℝ) (markup_percentage : ℝ) 
  (h1 : selling_price = 8800)
  (h2 : markup_percentage = 0.10) : 
  ∃ (cost_price : ℝ), cost_price = 8000 ∧ selling_price = cost_price * (1 + markup_percentage) := by
  sorry

end computer_table_cost_price_l2625_262540


namespace parallel_line_angle_theorem_l2625_262518

-- Define the structure for our geometric configuration
structure ParallelLineConfig where
  -- Angle QTV
  angle_QTV : ℝ
  -- Angle SUV
  angle_SUV : ℝ
  -- Angle TVU
  angle_TVU : ℝ
  -- Assumption that PQ and RS are parallel
  parallel_PQ_RS : True
  -- Assumptions about the given angles
  h_QTV : angle_QTV = 30
  h_SUV : angle_SUV = 40

-- Theorem statement
theorem parallel_line_angle_theorem (config : ParallelLineConfig) :
  config.angle_TVU = 70 := by
  sorry

end parallel_line_angle_theorem_l2625_262518


namespace Q_value_at_negative_one_l2625_262562

/-- The cubic polynomial P(x) -/
def P (x : ℝ) : ℝ := x^3 + 8*x^2 - x + 3

/-- The roots of P(x) -/
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

/-- Q is a monic polynomial with roots ab - c^2, ac - b^2, bc - a^2 -/
def Q (x : ℝ) : ℝ := x^3 + 67*x^2 + 67*x + 1537

theorem Q_value_at_negative_one :
  P a = 0 → P b = 0 → P c = 0 → Q (-1) = 1536 := by
  sorry

end Q_value_at_negative_one_l2625_262562


namespace johnnys_age_l2625_262528

theorem johnnys_age : ∃ (age : ℕ), 
  (age + 2 = 2 * (age - 3)) ∧ age = 8 := by sorry

end johnnys_age_l2625_262528


namespace cube_properties_l2625_262565

/-- Given a cube with volume 343 cubic centimeters, prove its surface area and internal space diagonal --/
theorem cube_properties (V : ℝ) (h : V = 343) : 
  ∃ (s : ℝ), s > 0 ∧ s^3 = V ∧ 6 * s^2 = 294 ∧ s * Real.sqrt 3 = 7 * Real.sqrt 3 :=
by sorry

end cube_properties_l2625_262565


namespace complex_root_of_unity_product_l2625_262597

theorem complex_root_of_unity_product (s : ℂ) (h1 : s^6 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) = 6 := by
  sorry

end complex_root_of_unity_product_l2625_262597


namespace fgh_supermarkets_count_l2625_262563

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 37

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 14

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 60

theorem fgh_supermarkets_count :
  us_supermarkets = 37 ∧
  us_supermarkets + canada_supermarkets = total_supermarkets ∧
  us_supermarkets = canada_supermarkets + 14 :=
by sorry

end fgh_supermarkets_count_l2625_262563


namespace movie_date_communication_l2625_262541

theorem movie_date_communication (p : ℝ) (h1 : p = 0.9) :
  p * p + (1 - p) * (1 - p) = 0.82 := by
  sorry

end movie_date_communication_l2625_262541


namespace max_circumference_circle_in_parabola_l2625_262589

/-- A circle located inside the parabola x^2 = 4y and passing through its vertex -/
structure CircleInParabola where
  center : ℝ × ℝ
  radius : ℝ
  inside_parabola : ∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 ≤ radius^2 → x^2 ≤ 4*y
  passes_through_vertex : (0 - center.1)^2 + (0 - center.2)^2 = radius^2

/-- The maximum circumference of a circle located inside the parabola x^2 = 4y 
    and passing through its vertex is 4π -/
theorem max_circumference_circle_in_parabola :
  ∃ (C : CircleInParabola), ∀ (D : CircleInParabola), 2 * Real.pi * C.radius ≥ 2 * Real.pi * D.radius ∧
  2 * Real.pi * C.radius = 4 * Real.pi :=
sorry

end max_circumference_circle_in_parabola_l2625_262589


namespace folded_paper_height_approx_l2625_262507

/-- The height of a folded sheet of paper -/
def folded_paper_height (initial_thickness : ℝ) (num_folds : ℕ) : ℝ :=
  initial_thickness * (2 ^ num_folds)

/-- Approximation of 2^10 -/
def approx_2_10 : ℝ := 1000

theorem folded_paper_height_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |folded_paper_height 0.1 20 - 100| < ε :=
sorry

end folded_paper_height_approx_l2625_262507


namespace sum_31_22_base4_l2625_262503

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number in base 10 -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_31_22_base4 :
  toBase4 (31 + 22) = [3, 1, 1] :=
sorry

end sum_31_22_base4_l2625_262503


namespace problem_1_problem_2_l2625_262519

-- Problem 1
theorem problem_1 : 2 * Real.sqrt 3 * 315 * 612 = 36600 := by sorry

-- Problem 2
theorem problem_2 : 2 * (Real.log 10 / Real.log 5) + (Real.log 0.25 / Real.log 5) = 2 := by sorry

end problem_1_problem_2_l2625_262519


namespace smallest_multiple_with_digit_sum_l2625_262574

def N : ℕ := 5 * 10^223 - 10^220 - 10^49 - 1

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem smallest_multiple_with_digit_sum :
  (N % 2009 = 0) ∧
  (sum_of_digits N = 2009) ∧
  (∀ m : ℕ, m < N → (m % 2009 = 0 ∧ sum_of_digits m = 2009) → False) :=
by sorry

end smallest_multiple_with_digit_sum_l2625_262574


namespace felicity_lollipop_collection_l2625_262588

/-- The number of sticks needed to finish the fort -/
def total_sticks : ℕ := 400

/-- The number of times Felicity's family goes to the store per week -/
def store_visits_per_week : ℕ := 3

/-- The percentage of completion of the fort -/
def fort_completion_percentage : ℚ := 60 / 100

/-- The number of weeks Felicity has been collecting lollipops -/
def collection_weeks : ℕ := 80

theorem felicity_lollipop_collection :
  collection_weeks = (fort_completion_percentage * total_sticks) / store_visits_per_week := by
  sorry

end felicity_lollipop_collection_l2625_262588


namespace pyramid_top_value_l2625_262571

/-- Represents a three-level pyramid of numbers -/
structure NumberPyramid where
  bottomLeft : ℕ
  bottomRight : ℕ
  middleLeft : ℕ
  middleRight : ℕ
  top : ℕ

/-- Checks if a NumberPyramid is valid according to the sum rule -/
def isValidPyramid (p : NumberPyramid) : Prop :=
  p.middleLeft = p.bottomLeft ∧
  p.middleRight = p.bottomRight ∧
  p.top = p.middleLeft + p.middleRight

theorem pyramid_top_value (p : NumberPyramid) 
  (h1 : p.bottomLeft = 35)
  (h2 : p.bottomRight = 47)
  (h3 : isValidPyramid p) : 
  p.top = 82 := by
  sorry

#check pyramid_top_value

end pyramid_top_value_l2625_262571


namespace playground_children_count_l2625_262564

/-- Calculate the final number of children on the playground --/
theorem playground_children_count 
  (initial_girls : ℕ) 
  (initial_boys : ℕ) 
  (additional_girls : ℕ) 
  (additional_boys : ℕ) 
  (children_leaving : ℕ) 
  (h1 : initial_girls = 28)
  (h2 : initial_boys = 35)
  (h3 : additional_girls = 5)
  (h4 : additional_boys = 7)
  (h5 : children_leaving = 15) : 
  (initial_girls + initial_boys + additional_girls + additional_boys) - children_leaving = 60 := by
  sorry

end playground_children_count_l2625_262564


namespace tournament_theorem_l2625_262531

/-- Represents a team in the tournament -/
inductive Team : Type
| A
| B
| C

/-- Represents the state of a player (active or eliminated) -/
inductive PlayerState : Type
| Active
| Eliminated

/-- Represents the state of the tournament -/
structure TournamentState :=
  (team_players : Team → Fin 9 → PlayerState)
  (matches_played : ℕ)
  (champion_wins : ℕ)

/-- The rules of the tournament -/
def tournament_rules (initial_state : TournamentState) : Prop :=
  ∀ (t : Team), ∃ (i : Fin 9), initial_state.team_players t i = PlayerState.Active

/-- The condition for a team to be eliminated -/
def team_eliminated (state : TournamentState) (t : Team) : Prop :=
  ∀ (i : Fin 9), state.team_players t i = PlayerState.Eliminated

/-- The condition for the tournament to end -/
def tournament_ended (state : TournamentState) : Prop :=
  ∃ (t1 t2 : Team), t1 ≠ t2 ∧ team_eliminated state t1 ∧ team_eliminated state t2

/-- The main theorem to prove -/
theorem tournament_theorem 
  (initial_state : TournamentState) 
  (h_rules : tournament_rules initial_state) :
  (∃ (final_state : TournamentState), 
    tournament_ended final_state ∧ 
    final_state.champion_wins ≥ 9) ∧
  (∀ (final_state : TournamentState),
    tournament_ended final_state → 
    final_state.champion_wins = 11 → 
    final_state.matches_played ≥ 24) :=
sorry

end tournament_theorem_l2625_262531


namespace function_value_at_two_l2625_262542

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x * g y - g (x * y) = x^2 + y^2

/-- The main theorem stating that g(2) = 5 for any function satisfying the functional equation -/
theorem function_value_at_two (g : ℝ → ℝ) (h : FunctionalEquation g) : g 2 = 5 := by
  sorry

end function_value_at_two_l2625_262542


namespace rectangle_area_l2625_262595

theorem rectangle_area (a b : ℝ) (h1 : a + b = 8) (h2 : 2*a^2 + 2*b^2 = 68) :
  a * b = 15 := by sorry

end rectangle_area_l2625_262595


namespace allan_bought_three_balloons_l2625_262573

/-- The number of balloons Allan bought at the park -/
def balloons_bought_by_allan : ℕ := 3

/-- Allan's initial number of balloons -/
def allan_initial_balloons : ℕ := 2

/-- Jake's initial number of balloons -/
def jake_initial_balloons : ℕ := 6

theorem allan_bought_three_balloons :
  balloons_bought_by_allan = 3 ∧
  allan_initial_balloons = 2 ∧
  jake_initial_balloons = 6 ∧
  jake_initial_balloons = (allan_initial_balloons + balloons_bought_by_allan + 1) :=
by sorry

end allan_bought_three_balloons_l2625_262573


namespace intersection_of_A_and_B_l2625_262527

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

def B : Set ℤ := {x | ∃ k : ℕ, k < 3 ∧ x = 2 * k + 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 3, 5} := by sorry

end intersection_of_A_and_B_l2625_262527


namespace soccer_balls_added_l2625_262590

theorem soccer_balls_added (initial_balls final_balls : ℕ) (h1 : initial_balls = 6) (h2 : final_balls = 24) :
  final_balls - initial_balls = 18 := by
  sorry

end soccer_balls_added_l2625_262590


namespace second_number_calculation_l2625_262593

theorem second_number_calculation (A B : ℕ) (h1 : A - B = 88) (h2 : A = 110) : B = 22 := by
  sorry

end second_number_calculation_l2625_262593


namespace sum_of_ages_in_ten_years_l2625_262504

/-- Theorem: Sum of ages in 10 years -/
theorem sum_of_ages_in_ten_years (my_current_age brother_current_age : ℕ) : 
  my_current_age = 20 →
  my_current_age + 10 = 2 * (brother_current_age + 10) →
  (my_current_age + 10) + (brother_current_age + 10) = 45 :=
by sorry

end sum_of_ages_in_ten_years_l2625_262504


namespace four_wheelers_count_l2625_262598

/-- Given a parking lot with only 2-wheelers and 4-wheelers, and a total of 58 wheels,
    prove that the number of 4-wheelers can be expressed in terms of the number of 2-wheelers. -/
theorem four_wheelers_count (x y : ℕ) (h1 : 2 * x + 4 * y = 58) :
  y = (29 - x) / 2 := by
  sorry

end four_wheelers_count_l2625_262598


namespace binomial_coefficient_two_l2625_262524

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : 
  Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l2625_262524


namespace min_value_and_angle_l2625_262525

theorem min_value_and_angle (A : Real) : 
  0 ≤ A ∧ A ≤ 2 * Real.pi →
  (∀ θ : Real, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
    2 * Real.sin (A / 2) + Real.sin A ≤ 2 * Real.sin (θ / 2) + Real.sin θ) →
  2 * Real.sin (A / 2) + Real.sin A = -4 ∧ A = Real.pi :=
by sorry

end min_value_and_angle_l2625_262525


namespace triangle_side_length_l2625_262536

theorem triangle_side_length 
  (a b c : ℝ) 
  (B : ℝ) 
  (h1 : b = 3) 
  (h2 : c = Real.sqrt 6) 
  (h3 : B = π / 3) 
  (h4 : b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B) : 
  a = (Real.sqrt 6 + 3 * Real.sqrt 2) / 2 := by
sorry

end triangle_side_length_l2625_262536


namespace isosceles_trapezoid_area_l2625_262560

-- Define the vertices of the trapezoid
def v1 : ℝ × ℝ := (0, 0)
def v2 : ℝ × ℝ := (8, 0)
def v3 : ℝ × ℝ := (6, 10)
def v4 : ℝ × ℝ := (2, 10)

-- Define the trapezoid
def isosceles_trapezoid (v1 v2 v3 v4 : ℝ × ℝ) : Prop :=
  -- Add conditions for isosceles trapezoid here
  True

-- Calculate the area of the trapezoid
def trapezoid_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  -- Add area calculation here
  0

-- Theorem statement
theorem isosceles_trapezoid_area :
  isosceles_trapezoid v1 v2 v3 v4 →
  trapezoid_area v1 v2 v3 v4 = 60 := by
  sorry

end isosceles_trapezoid_area_l2625_262560


namespace division_remainder_problem_l2625_262594

theorem division_remainder_problem (R Q D : ℕ) : 
  D = 3 * Q →
  D = 3 * R + 3 →
  251 = D * Q + R →
  R = 8 := by sorry

end division_remainder_problem_l2625_262594


namespace quadratic_derivative_bound_l2625_262535

-- Define a quadratic function
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

-- State the theorem
theorem quadratic_derivative_bound :
  ∃ A : ℝ,
    (∀ a b c : ℝ,
      (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |QuadraticFunction a b c x| ≤ 1) →
      |b| ≤ A) ∧
    (∃ a b c : ℝ,
      (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |QuadraticFunction a b c x| ≤ 1) ∧
      |b| = A) ∧
    (∀ A' : ℝ,
      (∀ a b c : ℝ,
        (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |QuadraticFunction a b c x| ≤ 1) →
        |b| ≤ A') →
      A ≤ A') :=
by sorry

end quadratic_derivative_bound_l2625_262535


namespace x_range_theorem_l2625_262554

-- Define the points and line
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)
def l (a : ℝ) : ℝ → ℝ := fun x => a - x

-- Define the condition for points relative to the line
def point_line_condition (a : ℝ) : Prop :=
  (l a O.1 = O.2 ∨ l a A.1 = A.2) ∨ (l a O.1 - O.2) * (l a A.1 - A.2) < 0

-- Define the function h
def h (a : ℝ) : ℝ := a^2 + 2*a + 3

-- State the theorem
theorem x_range_theorem :
  ∀ x : ℝ, (∀ a : ℝ, point_line_condition a → x^2 + 4*x - 2 ≤ h a) ↔ -5 ≤ x ∧ x ≤ 1 :=
sorry

end x_range_theorem_l2625_262554


namespace flower_bunch_problem_l2625_262508

/-- The number of flowers in each bunch initially -/
def initial_flowers_per_bunch : ℕ := sorry

/-- The number of bunches initially -/
def initial_bunches : ℕ := 8

/-- The number of flowers per bunch in the alternative scenario -/
def alternative_flowers_per_bunch : ℕ := 12

/-- The number of bunches in the alternative scenario -/
def alternative_bunches : ℕ := 6

theorem flower_bunch_problem :
  initial_flowers_per_bunch * initial_bunches = alternative_flowers_per_bunch * alternative_bunches ∧
  initial_flowers_per_bunch = 9 := by sorry

end flower_bunch_problem_l2625_262508


namespace factor_of_polynomial_l2625_262521

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), 4 * x^3 + 6 * x^2 + 11 * x - 6 = (x - 1/2) * q x := by
  sorry

end factor_of_polynomial_l2625_262521


namespace equal_roots_quadratic_l2625_262592

theorem equal_roots_quadratic (a : ℕ) : 
  (∀ x : ℝ, x^2 - a*x + (a + 3) = 0 → (∃! y : ℝ, y^2 - a*y + (a + 3) = 0)) → 
  a = 6 :=
sorry

end equal_roots_quadratic_l2625_262592


namespace quadratic_two_distinct_roots_l2625_262532

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k = 0 → ∃ y : ℝ, y ≠ x ∧ y^2 - 2*y + k = 0) ↔ k < 1 :=
by sorry

end quadratic_two_distinct_roots_l2625_262532


namespace cone_volume_in_cylinder_with_spheres_l2625_262500

/-- The volume of a cone inscribed in a cylinder with specific properties --/
theorem cone_volume_in_cylinder_with_spheres (r h : ℝ) : 
  r = 1 → 
  h = 12 / (3 + 2 * Real.sqrt 3) →
  ∃ (cone_volume : ℝ), 
    cone_volume = (2/3) * Real.pi ∧
    cone_volume = (1/3) * Real.pi * r^2 * 1 ∧
    ∃ (sphere_radius : ℝ),
      sphere_radius = 2 * Real.sqrt 3 - 3 ∧
      sphere_radius > 0 ∧
      sphere_radius < r ∧
      h = 2 * sphere_radius :=
by sorry

end cone_volume_in_cylinder_with_spheres_l2625_262500


namespace uncool_parents_count_l2625_262505

theorem uncool_parents_count (total_students cool_dads cool_moms both_cool : ℕ) 
  (h1 : total_students = 35)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 20)
  (h4 : both_cool = 11) :
  total_students - (cool_dads + cool_moms - both_cool) = 8 := by
sorry

end uncool_parents_count_l2625_262505


namespace problem_solution_l2625_262501

theorem problem_solution (m n c : Int) (hm : m = -4) (hn : n = -5) (hc : c = -7) :
  m - n - c = 8 := by
  sorry

end problem_solution_l2625_262501


namespace total_situps_l2625_262523

-- Define the performance profiles for each participant
def adam_performance (round : ℕ) : ℕ :=
  match round with
  | 1 => 40
  | 2 => 35
  | 3 => 30
  | 4 => 20
  | _ => 0

def barney_performance (round : ℕ) : ℕ :=
  max (45 - 3 * (round - 1)) 0

def carrie_performance (round : ℕ) : ℕ :=
  match round with
  | 1 | 2 => 90
  | 3 | 4 => 80
  | 5 => 70
  | _ => 0

def jerrie_performance (round : ℕ) : ℕ :=
  match round with
  | 1 | 2 => 95
  | 3 | 4 => 101
  | 5 => 94
  | 6 => 87
  | 7 => 80
  | _ => 0

-- Define the number of rounds for each participant
def adam_rounds : ℕ := 4
def barney_rounds : ℕ := 6
def carrie_rounds : ℕ := 5
def jerrie_rounds : ℕ := 7

-- Define the total sit-ups for each participant
def adam_total : ℕ := (List.range adam_rounds).map adam_performance |>.sum
def barney_total : ℕ := (List.range barney_rounds).map barney_performance |>.sum
def carrie_total : ℕ := (List.range carrie_rounds).map carrie_performance |>.sum
def jerrie_total : ℕ := (List.range jerrie_rounds).map jerrie_performance |>.sum

-- Theorem statement
theorem total_situps :
  adam_total + barney_total + carrie_total + jerrie_total = 1353 := by
  sorry

end total_situps_l2625_262523


namespace quadratic_properties_l2625_262582

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - 2*a^2

-- Theorem statement
theorem quadratic_properties (a : ℝ) (h : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ f a x₁ = 0 ∧ f a x₂ = 0) ∧
  (f a 0 = -2 → 
    (∃ x y : ℝ, (x = 1/2 ∨ x = -1/2) ∧ y = -9/4 ∧ 
    ∀ t : ℝ, f a t ≥ f a x)) :=
by sorry

end quadratic_properties_l2625_262582


namespace expand_and_simplify_l2625_262538

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 8) = x^2 + 5*x - 24 := by
  sorry

end expand_and_simplify_l2625_262538


namespace equilateral_triangle_circle_areas_l2625_262529

theorem equilateral_triangle_circle_areas (s : ℝ) (h : s = 12) :
  let r := s / 2
  let sector_area := (π * r^2) / 3
  let triangle_area := (s^2 * Real.sqrt 3) / 4
  let shaded_area := 2 * (sector_area - triangle_area)
  ∃ (a b c : ℝ), shaded_area = a * π - b * Real.sqrt c ∧ a + b + c = 33 :=
sorry

end equilateral_triangle_circle_areas_l2625_262529


namespace diophantine_equation_solution_l2625_262530

theorem diophantine_equation_solution (x y z : ℤ) :
  2 * x^2 + 3 * y^2 = z^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end diophantine_equation_solution_l2625_262530


namespace loot_box_loss_l2625_262557

theorem loot_box_loss (cost_per_box : ℝ) (avg_value_per_box : ℝ) (total_spent : ℝ)
  (h1 : cost_per_box = 5)
  (h2 : avg_value_per_box = 3.5)
  (h3 : total_spent = 40) :
  (total_spent / cost_per_box) * (cost_per_box - avg_value_per_box) = 12 :=
by sorry

end loot_box_loss_l2625_262557


namespace select_and_assign_volunteers_eq_30_l2625_262545

/-- The number of ways to select and assign volunteers for a two-day event -/
def select_and_assign_volunteers : ℕ :=
  let total_volunteers : ℕ := 5
  let selected_volunteers : ℕ := 4
  let days : ℕ := 2
  let volunteers_per_day : ℕ := 2
  (total_volunteers.choose selected_volunteers) *
  ((selected_volunteers.choose volunteers_per_day) * (days.factorial))

/-- Theorem stating that the number of ways to select and assign volunteers is 30 -/
theorem select_and_assign_volunteers_eq_30 :
  select_and_assign_volunteers = 30 := by
  sorry

end select_and_assign_volunteers_eq_30_l2625_262545


namespace complement_intersection_problem_l2625_262579

theorem complement_intersection_problem (U A B : Set ℕ) : 
  U = {1, 2, 3, 4, 5} →
  A = {1, 2, 3} →
  B = {3, 4, 5} →
  (U \ A) ∩ B = {4, 5} := by
  sorry

end complement_intersection_problem_l2625_262579


namespace circle_area_theorem_l2625_262585

/-- Given a circle with radius R and four smaller circles with radius R/2 constructed
    as described in the problem, this theorem states that the sum of the areas of the
    overlapping parts of the smaller circles equals the area of the original circle
    minus the areas of the non-overlapping parts of the smaller circles. -/
theorem circle_area_theorem (R : ℝ) (h : R > 0) :
  let big_circle_area := π * R^2
  let small_circle_area := π * (R/2)^2
  let segment_area := (π/4 - 1/2) * R^2
  let overlap_area := 2 * segment_area
  overlap_area = big_circle_area - 4 * (small_circle_area - segment_area) := by
  sorry


end circle_area_theorem_l2625_262585


namespace min_value_problem_l2625_262511

theorem min_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_constraint : x + Real.sqrt 3 * y + z = 6) :
  ∃ (min_val : ℝ), min_val = 37/4 ∧ 
  ∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 → 
  x' + Real.sqrt 3 * y' + z' = 6 → 
  x'^3 + y'^2 + 3*z' ≥ min_val :=
sorry

end min_value_problem_l2625_262511


namespace certain_number_problem_l2625_262566

theorem certain_number_problem (N : ℚ) :
  (5 / 6 : ℚ) * N - (5 / 16 : ℚ) * N = 150 → N = 288 := by
  sorry

end certain_number_problem_l2625_262566


namespace sum_of_squares_square_of_sum_sum_of_three_squares_sum_of_fourth_powers_l2625_262546

-- Part 1
theorem sum_of_squares (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) :
  a^2 + b^2 = 5 := by sorry

-- Part 2
theorem square_of_sum (a b c : ℝ) :
  (a + b + c)^2 = a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c := by sorry

-- Part 3
theorem sum_of_three_squares (a b c : ℝ) (h1 : a + b + c = 6) (h2 : a*b + b*c + a*c = 11) :
  a^2 + b^2 + c^2 = 14 := by sorry

-- Part 4
theorem sum_of_fourth_powers (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) :
  a^4 + b^4 + c^4 = 18 := by sorry

end sum_of_squares_square_of_sum_sum_of_three_squares_sum_of_fourth_powers_l2625_262546


namespace clothing_discount_problem_l2625_262537

theorem clothing_discount_problem (discount_rate : ℝ) (savings : ℝ) 
  (h1 : discount_rate = 0.2)
  (h2 : savings = 10)
  (h3 : ∀ x, (1 - discount_rate) * (x + savings) = x) :
  ∃ x, x = 40 ∧ (1 - discount_rate) * (x + savings) = x := by
sorry

end clothing_discount_problem_l2625_262537


namespace prob_five_diamond_three_l2625_262596

-- Define a standard deck of cards
def standard_deck : Nat := 52

-- Define the number of 5s in a deck
def num_fives : Nat := 4

-- Define the number of diamonds in a deck
def num_diamonds : Nat := 13

-- Define the number of 3s in a deck
def num_threes : Nat := 4

-- Define our specific event
def event_probability : ℚ :=
  (num_fives : ℚ) / standard_deck *
  (num_diamonds : ℚ) / (standard_deck - 1) *
  (num_threes : ℚ) / (standard_deck - 2)

-- Theorem statement
theorem prob_five_diamond_three :
  event_probability = 1 / 663 := by
  sorry

end prob_five_diamond_three_l2625_262596


namespace rectangle_measurement_error_l2625_262575

/-- Represents the measurement error in a rectangle's dimensions and area -/
structure RectangleMeasurement where
  length_excess : ℝ  -- Percentage excess in length measurement
  width_deficit : ℝ  -- Percentage deficit in width measurement
  area_error : ℝ     -- Percentage error in calculated area

/-- Theorem stating the relationship between measurement errors in a rectangle -/
theorem rectangle_measurement_error 
  (r : RectangleMeasurement) 
  (h1 : r.length_excess = 8)
  (h2 : r.area_error = 2.6) :
  r.width_deficit = 5 :=
sorry

end rectangle_measurement_error_l2625_262575


namespace fraction_equality_l2625_262543

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hxy : x ≠ y) :
  (x * y) / (x^2 - x * y) = y / (x - y) := by
  sorry

end fraction_equality_l2625_262543


namespace curve_is_line_l2625_262577

/-- The equation (x^2 + y^2 - 2)√(x-3) = 0 represents a line -/
theorem curve_is_line : 
  ∃ (a b : ℝ), ∀ (x y : ℝ), (x^2 + y^2 - 2) * Real.sqrt (x - 3) = 0 → y = a * x + b :=
sorry

end curve_is_line_l2625_262577


namespace probability_of_correct_answer_l2625_262569

theorem probability_of_correct_answer (options : Nat) (correct_options : Nat) : 
  options = 4 → correct_options = 1 → (correct_options : ℚ) / options = 1/4 := by
  sorry

end probability_of_correct_answer_l2625_262569


namespace sqrt_ratio_equals_sqrt_five_l2625_262553

theorem sqrt_ratio_equals_sqrt_five : 
  Real.sqrt (3^2 + 4^2) / Real.sqrt (4 + 1) = Real.sqrt 5 := by sorry

end sqrt_ratio_equals_sqrt_five_l2625_262553


namespace joe_hvac_cost_l2625_262578

/-- The total cost of an HVAC system with given parameters. -/
def hvac_total_cost (num_zones : ℕ) (vents_per_zone : ℕ) (cost_per_vent : ℕ) : ℕ :=
  num_zones * vents_per_zone * cost_per_vent

/-- Theorem stating that the total cost of Joe's HVAC system is $20,000. -/
theorem joe_hvac_cost :
  hvac_total_cost 2 5 2000 = 20000 := by
  sorry

end joe_hvac_cost_l2625_262578


namespace burger_cost_l2625_262506

/-- Represents the cost of items in cents -/
structure Cost where
  burger : ℕ
  soda : ℕ
  fry : ℕ

/-- Alice's purchase -/
def alice_purchase (c : Cost) : ℕ :=
  4 * c.burger + 2 * c.soda + 3 * c.fry

/-- Bill's purchase -/
def bill_purchase (c : Cost) : ℕ :=
  3 * c.burger + c.soda + 2 * c.fry

theorem burger_cost :
  ∃ (c : Cost), alice_purchase c = 480 ∧ bill_purchase c = 360 ∧ c.burger = 80 :=
by sorry

end burger_cost_l2625_262506


namespace madeline_score_is_28_l2625_262556

/-- Represents the score and mistakes in a Geometry exam -/
structure GeometryExam where
  totalQuestions : ℕ
  scorePerQuestion : ℕ
  madelineMistakes : ℕ
  leoMistakes : ℕ
  brentMistakes : ℕ
  brentScore : ℕ

/-- Calculates Madeline's score in the Geometry exam -/
def madelineScore (exam : GeometryExam) : ℕ :=
  exam.totalQuestions * exam.scorePerQuestion - exam.madelineMistakes * exam.scorePerQuestion

/-- Theorem: Given the conditions, Madeline's score in the Geometry exam is 28 -/
theorem madeline_score_is_28 (exam : GeometryExam)
  (h1 : exam.madelineMistakes = 2)
  (h2 : exam.leoMistakes = 2 * exam.madelineMistakes)
  (h3 : exam.brentMistakes = exam.leoMistakes + 1)
  (h4 : exam.brentScore = 25)
  (h5 : exam.totalQuestions = exam.brentScore + exam.brentMistakes)
  (h6 : exam.scorePerQuestion = 1) :
  madelineScore exam = 28 := by
  sorry


end madeline_score_is_28_l2625_262556


namespace lottery_probability_l2625_262587

theorem lottery_probability (max_number : ℕ) 
  (prob_1_to_15 : ℚ) (prob_1_or_larger : ℚ) :
  max_number ≥ 15 →
  prob_1_to_15 = 1 / 3 →
  prob_1_or_larger = 2 / 3 →
  (∀ n : ℕ, n ≤ 15 → n ≥ 1) →
  (∀ n : ℕ, n ≤ max_number → n ≥ 1) →
  (probability_less_equal_15 : ℚ) →
  probability_less_equal_15 = prob_1_or_larger :=
by sorry

end lottery_probability_l2625_262587
