import Mathlib

namespace sixteen_bananas_equal_nineteen_grapes_l1163_116302

/-- The cost relationship between bananas, oranges, and grapes -/
structure FruitCosts where
  banana_orange_ratio : ℚ  -- 4 bananas = 3 oranges
  orange_grape_ratio : ℚ   -- 5 oranges = 8 grapes

/-- Calculate the number of grapes equivalent in cost to a given number of bananas -/
def grapes_for_bananas (costs : FruitCosts) (num_bananas : ℕ) : ℕ :=
  let oranges : ℚ := (num_bananas : ℚ) * costs.banana_orange_ratio
  let grapes : ℚ := oranges * costs.orange_grape_ratio
  grapes.ceil.toNat

/-- Theorem stating that 16 bananas cost as much as 19 grapes -/
theorem sixteen_bananas_equal_nineteen_grapes (costs : FruitCosts) 
    (h1 : costs.banana_orange_ratio = 3/4)
    (h2 : costs.orange_grape_ratio = 8/5) : 
  grapes_for_bananas costs 16 = 19 := by
  sorry

#eval grapes_for_bananas ⟨3/4, 8/5⟩ 16

end sixteen_bananas_equal_nineteen_grapes_l1163_116302


namespace housewife_money_l1163_116330

theorem housewife_money (initial_money : ℚ) : 
  (1 - 2/3) * initial_money = 50 → initial_money = 150 := by
  sorry

end housewife_money_l1163_116330


namespace max_B_at_125_l1163_116385

def B (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.3 ^ k)

theorem max_B_at_125 :
  ∀ k : ℕ, k ≤ 500 → B 125 ≥ B k :=
by sorry

end max_B_at_125_l1163_116385


namespace exists_positive_m_for_field_l1163_116357

/-- The dimensions of a rectangular field -/
def field_length (m : ℝ) : ℝ := 4*m + 6

/-- The width of a rectangular field -/
def field_width (m : ℝ) : ℝ := 2*m - 5

/-- The area of the rectangular field -/
def field_area : ℝ := 159

/-- Theorem stating that there exists a positive real number m that satisfies the field dimensions and area -/
theorem exists_positive_m_for_field : ∃ m : ℝ, m > 0 ∧ field_length m * field_width m = field_area := by
  sorry

end exists_positive_m_for_field_l1163_116357


namespace complex_magnitude_l1163_116387

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end complex_magnitude_l1163_116387


namespace decreasing_interval_of_even_function_l1163_116311

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + m * x + 4

theorem decreasing_interval_of_even_function (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  {x : ℝ | ∀ y, x ≤ y → f m x ≥ f m y} = Set.Ici (0 : ℝ) :=
sorry

end decreasing_interval_of_even_function_l1163_116311


namespace partial_fraction_decomposition_l1163_116329

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), C = 32/9 ∧ D = 13/9 ∧
  ∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 →
    (5*x - 3) / (x^2 - 5*x - 14) = C / (x - 7) + D / (x + 2) :=
by sorry

end partial_fraction_decomposition_l1163_116329


namespace quadratic_root_sum_power_l1163_116303

/-- Given a quadratic equation x^2 + mx + 3 = 0 with roots 1 and n, prove (m + n)^2023 = -1 -/
theorem quadratic_root_sum_power (m n : ℝ) : 
  (1 : ℝ) ^ 2 + m * 1 + 3 = 0 → 
  n ^ 2 + m * n + 3 = 0 → 
  (m + n) ^ 2023 = -1 := by
  sorry

end quadratic_root_sum_power_l1163_116303


namespace sine_graph_transformation_l1163_116328

theorem sine_graph_transformation (x : ℝ) :
  let f (x : ℝ) := Real.sin (x + π / 6)
  let g (x : ℝ) := f (x + π / 4)
  let h (x : ℝ) := g (x / 2)
  h x = Real.sin (x / 2 + 5 * π / 12) := by sorry

end sine_graph_transformation_l1163_116328


namespace intersection_point_unique_l1163_116326

/-- The line equation (x+3)/2 = (y-1)/3 = (z-1)/5 -/
def line_eq (x y z : ℝ) : Prop :=
  (x + 3) / 2 = (y - 1) / 3 ∧ (y - 1) / 3 = (z - 1) / 5

/-- The plane equation 2x + 3y + 7z - 52 = 0 -/
def plane_eq (x y z : ℝ) : Prop :=
  2 * x + 3 * y + 7 * z - 52 = 0

/-- The intersection point (-1, 4, 6) -/
def intersection_point : ℝ × ℝ × ℝ := (-1, 4, 6)

theorem intersection_point_unique :
  ∀ x y z : ℝ, line_eq x y z ∧ plane_eq x y z ↔ (x, y, z) = intersection_point :=
by sorry

end intersection_point_unique_l1163_116326


namespace inequality_proof_l1163_116315

theorem inequality_proof (x a : ℝ) (f : ℝ → ℝ) 
  (h1 : f = λ x => x^2 - x + 1) 
  (h2 : |x - a| < 1) : 
  |f x - f a| < 2 * (|a| + 1) := by
  sorry

end inequality_proof_l1163_116315


namespace binomial_sum_l1163_116313

theorem binomial_sum : Nat.choose 12 4 + Nat.choose 10 3 = 615 := by
  sorry

end binomial_sum_l1163_116313


namespace school_contribution_l1163_116333

def book_cost : ℕ := 12
def num_students : ℕ := 30
def sally_paid : ℕ := 40

theorem school_contribution : 
  ∃ (school_amount : ℕ), 
    school_amount = book_cost * num_students - sally_paid ∧ 
    school_amount = 320 := by
  sorry

end school_contribution_l1163_116333


namespace fishbowl_count_l1163_116389

theorem fishbowl_count (fish_per_bowl : ℕ) (total_fish : ℕ) (h1 : fish_per_bowl = 23) (h2 : total_fish = 6003) :
  total_fish / fish_per_bowl = 261 := by
  sorry

end fishbowl_count_l1163_116389


namespace rectangle_circle_area_ratio_l1163_116335

theorem rectangle_circle_area_ratio (w l r : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 2 * Real.pi * r) :
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
sorry

end rectangle_circle_area_ratio_l1163_116335


namespace max_profit_transport_plan_l1163_116345

/-- Represents the transportation problem for fruits A, B, and C. -/
structure FruitTransport where
  total_trucks : ℕ
  total_tons : ℕ
  tons_per_truck_A : ℕ
  tons_per_truck_B : ℕ
  tons_per_truck_C : ℕ
  profit_per_ton_A : ℕ
  profit_per_ton_B : ℕ
  profit_per_ton_C : ℕ
  min_trucks_per_fruit : ℕ

/-- Calculates the profit for a given transportation plan. -/
def calculate_profit (ft : FruitTransport) (x y : ℕ) : ℕ :=
  ft.profit_per_ton_A * ft.tons_per_truck_A * x +
  ft.profit_per_ton_B * ft.tons_per_truck_B * y +
  ft.profit_per_ton_C * ft.tons_per_truck_C * (ft.total_trucks - x - y)

/-- States that the given transportation plan maximizes profit. -/
theorem max_profit_transport_plan (ft : FruitTransport)
  (h_total_trucks : ft.total_trucks = 20)
  (h_total_tons : ft.total_tons = 100)
  (h_tons_A : ft.tons_per_truck_A = 6)
  (h_tons_B : ft.tons_per_truck_B = 5)
  (h_tons_C : ft.tons_per_truck_C = 4)
  (h_profit_A : ft.profit_per_ton_A = 500)
  (h_profit_B : ft.profit_per_ton_B = 600)
  (h_profit_C : ft.profit_per_ton_C = 400)
  (h_min_trucks : ft.min_trucks_per_fruit = 2) :
  ∃ (x y : ℕ),
    x = 2 ∧
    y = 16 ∧
    ft.total_trucks - x - y = 2 ∧
    calculate_profit ft x y = 57200 ∧
    ∀ (x' y' : ℕ),
      x' ≥ ft.min_trucks_per_fruit →
      y' ≥ ft.min_trucks_per_fruit →
      ft.total_trucks - x' - y' ≥ ft.min_trucks_per_fruit →
      calculate_profit ft x' y' ≤ calculate_profit ft x y :=
by
  sorry

end max_profit_transport_plan_l1163_116345


namespace inverse_proportion_ordering_l1163_116368

/-- Proves that for points on an inverse proportion function, 
    if x₁ < 0 < x₂, then y₁ < y₂ -/
theorem inverse_proportion_ordering (x₁ x₂ y₁ y₂ : ℝ) : 
  x₁ < 0 → 0 < x₂ → y₁ = 6 / x₁ → y₂ = 6 / x₂ → y₁ < y₂ := by
  sorry

end inverse_proportion_ordering_l1163_116368


namespace brick_weight_l1163_116364

theorem brick_weight :
  ∀ x : ℝ, x = 2 + x / 2 → x = 4 :=
by
  sorry

end brick_weight_l1163_116364


namespace store_visits_per_week_l1163_116384

/-- The number of store visits per week given the fort's completion status and collection period -/
theorem store_visits_per_week 
  (total_sticks : ℕ)
  (completion_percentage : ℚ)
  (collection_weeks : ℕ)
  (h1 : total_sticks = 400)
  (h2 : completion_percentage = 3/5)
  (h3 : collection_weeks = 80) :
  (completion_percentage * total_sticks) / collection_weeks = 3 := by
  sorry

end store_visits_per_week_l1163_116384


namespace remainder_11_power_603_mod_500_l1163_116346

theorem remainder_11_power_603_mod_500 : 11^603 % 500 = 331 := by
  sorry

end remainder_11_power_603_mod_500_l1163_116346


namespace decimal_division_l1163_116358

theorem decimal_division (x y : ℚ) (hx : x = 0.25) (hy : y = 0.005) : x / y = 50 := by
  sorry

end decimal_division_l1163_116358


namespace integral_reciprocal_e_l1163_116316

open Real MeasureTheory

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem integral_reciprocal_e : ∫ x in Set.Icc (1/Real.exp 1) (Real.exp 1), f x = 2 := by
  sorry

end integral_reciprocal_e_l1163_116316


namespace arithmetic_mean_of_fractions_l1163_116325

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((3 / 8 : ℚ) + (5 / 9 : ℚ)) = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l1163_116325


namespace a_less_than_b_l1163_116373

theorem a_less_than_b (x a b : ℝ) (h1 : x > 0) (h2 : a * b ≠ 0) (h3 : a * x < b * x + 1) : a < b := by
  sorry

end a_less_than_b_l1163_116373


namespace local_extremum_and_minimum_l1163_116365

-- Define the function f
def f (a b x : ℝ) : ℝ := a^2 * x^3 + 3 * a * x^2 - b * x - 1

-- State the theorem
theorem local_extremum_and_minimum (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≥ f a b 1) ∧
  (f a b 1 = 0) ∧
  (∀ x ≥ 0, f a b x ≥ -1) →
  a = -1/2 ∧ b = -9/4 ∧ ∀ x ≥ 0, f a b x ≥ -1 :=
by sorry

end local_extremum_and_minimum_l1163_116365


namespace number_problem_l1163_116369

theorem number_problem (x : ℝ) : 0.65 * x = 0.8 * x - 21 → x = 140 := by
  sorry

end number_problem_l1163_116369


namespace asterisk_replacement_l1163_116304

theorem asterisk_replacement : ∃ x : ℝ, (x / 18) * (x / 72) = 1 ∧ x = 36 := by
  sorry

end asterisk_replacement_l1163_116304


namespace sqrt_sum_equals_eight_sqrt_two_l1163_116394

theorem sqrt_sum_equals_eight_sqrt_two : 
  Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) = 8 * Real.sqrt 2 := by
  sorry

end sqrt_sum_equals_eight_sqrt_two_l1163_116394


namespace curve_range_l1163_116308

/-- The curve y^2 - xy + 2x + k = 0 passes through the point (a, -a) -/
def passes_through (k a : ℝ) : Prop :=
  (-a)^2 - a * (-a) + 2 * a + k = 0

/-- The range of k values for which the curve passes through (a, -a) for some real a -/
def k_range (k : ℝ) : Prop :=
  ∃ a : ℝ, passes_through k a

theorem curve_range :
  ∀ k : ℝ, k_range k → k ≤ 1/2 := by sorry

end curve_range_l1163_116308


namespace difference_of_squares_625_375_l1163_116349

theorem difference_of_squares_625_375 : 625^2 - 375^2 = 250000 := by
  sorry

end difference_of_squares_625_375_l1163_116349


namespace mapping_not_necessarily_injective_l1163_116372

-- Define sets A and B
variable (A B : Type)

-- Define a mapping from A to B
variable (f : A → B)

-- Theorem stating that it's possible for two different elements in A to have the same image in B
theorem mapping_not_necessarily_injective :
  ∃ (x y : A), x ≠ y ∧ f x = f y :=
sorry

end mapping_not_necessarily_injective_l1163_116372


namespace group_trip_cost_l1163_116377

/-- The total cost for a group trip, given the number of people and cost per person. -/
def total_cost (num_people : ℕ) (cost_per_person : ℕ) : ℕ :=
  num_people * cost_per_person

/-- Proof that the total cost for 15 people at $900 each is $13,500. -/
theorem group_trip_cost :
  total_cost 15 900 = 13500 := by
  sorry

end group_trip_cost_l1163_116377


namespace average_distance_and_monthly_expense_l1163_116362

/-- Represents the daily distance traveled relative to the standard distance -/
def daily_distances : List ℤ := [-8, -11, -14, 0, -16, 41, 8]

/-- The standard distance in kilometers -/
def standard_distance : ℕ := 50

/-- Gasoline consumption in liters per 100 km -/
def gasoline_consumption : ℚ := 6 / 100

/-- Gasoline price in yuan per liter -/
def gasoline_price : ℚ := 77 / 10

/-- Number of days in a month -/
def days_in_month : ℕ := 30

theorem average_distance_and_monthly_expense :
  let avg_distance := standard_distance + (daily_distances.sum / daily_distances.length : ℚ)
  let monthly_expense := (days_in_month : ℚ) * avg_distance * gasoline_consumption * gasoline_price
  avg_distance = standard_distance ∧ monthly_expense = 693 := by
  sorry

end average_distance_and_monthly_expense_l1163_116362


namespace arithmetic_sequence_first_term_l1163_116376

/-- Sum of first n terms of an arithmetic sequence -/
def T (b : ℚ) (n : ℕ) : ℚ := n * (2 * b + (n - 1) * 5) / 2

/-- The problem statement -/
theorem arithmetic_sequence_first_term (b : ℚ) :
  (∃ k : ℚ, ∀ n : ℕ, n > 0 → T b (4 * n) / T b n = k) →
  b = 5 / 2 := by
  sorry

end arithmetic_sequence_first_term_l1163_116376


namespace roulette_probability_l1163_116356

/-- Represents a roulette wheel with sections A, B, and C. -/
structure RouletteWheel where
  probA : ℚ
  probB : ℚ
  probC : ℚ

/-- The sum of probabilities for all sections in a roulette wheel is 1. -/
def validWheel (wheel : RouletteWheel) : Prop :=
  wheel.probA + wheel.probB + wheel.probC = 1

/-- Theorem: Given a valid roulette wheel with probA = 1/4 and probB = 1/2, probC must be 1/4. -/
theorem roulette_probability (wheel : RouletteWheel) 
  (h_valid : validWheel wheel) 
  (h_probA : wheel.probA = 1/4) 
  (h_probB : wheel.probB = 1/2) : 
  wheel.probC = 1/4 := by
  sorry

end roulette_probability_l1163_116356


namespace lateral_angle_cosine_l1163_116388

/-- A regular triangular pyramid with an inscribed sphere -/
structure RegularPyramid where
  -- The ratio of the intersection point on an edge
  intersectionRatio : ℝ
  -- Assumption that the pyramid is regular and has an inscribed sphere
  regular : Bool
  hasInscribedSphere : Bool

/-- The angle between a lateral face and the base plane of the pyramid -/
def lateralAngle (p : RegularPyramid) : ℝ := sorry

/-- Main theorem: The cosine of the lateral angle is 7/10 -/
theorem lateral_angle_cosine (p : RegularPyramid) 
  (h1 : p.intersectionRatio = 1.55)
  (h2 : p.regular = true)
  (h3 : p.hasInscribedSphere = true) : 
  Real.cos (lateralAngle p) = 7/10 := by sorry

end lateral_angle_cosine_l1163_116388


namespace hyperbola_properties_l1163_116392

/-- Hyperbola C with equation x^2 - 4y^2 = 1 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 - 4*p.2^2 = 1}

/-- The asymptotes of hyperbola C -/
def asymptotes : Set (ℝ × ℝ) := {p | p.1 + 2*p.2 = 0 ∨ p.1 - 2*p.2 = 0}

/-- The imaginary axis length of hyperbola C -/
def imaginary_axis_length : ℝ := 1

/-- Theorem: The asymptotes and imaginary axis length of hyperbola C -/
theorem hyperbola_properties :
  (∀ p ∈ C, p ∈ asymptotes ↔ p.1^2 = 4*p.2^2) ∧
  imaginary_axis_length = 1 := by
  sorry

end hyperbola_properties_l1163_116392


namespace sector_angle_when_length_equals_area_l1163_116332

/-- Theorem: For a circular sector with arc length and area both equal to 6,
    the central angle in radians is 3. -/
theorem sector_angle_when_length_equals_area (r : ℝ) (θ : ℝ) : 
  r * θ = 6 → -- arc length = r * θ = 6
  (1/2) * r^2 * θ = 6 → -- area = (1/2) * r^2 * θ = 6
  θ = 3 := by sorry

end sector_angle_when_length_equals_area_l1163_116332


namespace total_rain_time_l1163_116341

def rain_duration_day1 : ℕ := 10

def rain_duration_day2 (d1 : ℕ) : ℕ := d1 + 2

def rain_duration_day3 (d2 : ℕ) : ℕ := 2 * d2

def total_rain_duration (d1 d2 d3 : ℕ) : ℕ := d1 + d2 + d3

theorem total_rain_time :
  total_rain_duration rain_duration_day1 
    (rain_duration_day2 rain_duration_day1) 
    (rain_duration_day3 (rain_duration_day2 rain_duration_day1)) = 46 := by
  sorry

end total_rain_time_l1163_116341


namespace parallelogram_opposite_sides_l1163_116378

/-- A parallelogram in a 2D Cartesian plane -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def diagonalIntersection (p : Parallelogram) : ℝ × ℝ := (0, 1)

def lineAB : LineEquation := { a := 1, b := -2, c := -2 }

theorem parallelogram_opposite_sides (p : Parallelogram) 
  (h1 : diagonalIntersection p = (0, 1))
  (h2 : lineAB = { a := 1, b := -2, c := -2 }) :
  ∃ (lineCD : LineEquation), lineCD = { a := 1, b := -2, c := 6 } := by
  sorry

end parallelogram_opposite_sides_l1163_116378


namespace solution_range_l1163_116327

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem solution_range (a b c : ℝ) :
  (f a b c 3 = 0.5) →
  (f a b c 4 = -0.5) →
  (f a b c 5 = -1) →
  ∃ x : ℝ, (ax^2 + b*x + c = 0) ∧ (3 < x) ∧ (x < 4) :=
by sorry

end solution_range_l1163_116327


namespace sufficient_condition_for_inequality_l1163_116398

theorem sufficient_condition_for_inequality (m : ℝ) (h1 : m ≠ 0) :
  (m > 2 → m + 4 / m > 4) ∧ ¬(m + 4 / m > 4 → m > 2) := by
  sorry

end sufficient_condition_for_inequality_l1163_116398


namespace sum_of_alternate_angles_less_than_450_l1163_116353

-- Define a heptagon
structure Heptagon where
  vertices : Fin 7 → ℝ × ℝ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of a heptagon being inscribed in a circle
def is_inscribed (h : Heptagon) (c : Circle) : Prop :=
  ∀ i : Fin 7, dist c.center (h.vertices i) = c.radius

-- Define the property of a point being inside a polygon
def is_inside (p : ℝ × ℝ) (h : Heptagon) : Prop :=
  sorry -- Definition of a point being inside a polygon

-- Define the angle at a vertex of the heptagon
def angle_at_vertex (h : Heptagon) (i : Fin 7) : ℝ :=
  sorry -- Definition of angle at a vertex

-- Theorem statement
theorem sum_of_alternate_angles_less_than_450 (h : Heptagon) (c : Circle) :
  is_inscribed h c → is_inside c.center h →
  angle_at_vertex h 0 + angle_at_vertex h 2 + angle_at_vertex h 4 < 450 :=
sorry

end sum_of_alternate_angles_less_than_450_l1163_116353


namespace work_completion_time_l1163_116379

-- Define the work completion time for Person A
def person_a_time : ℝ := 24

-- Define the combined work completion time for Person A and Person B
def combined_time : ℝ := 15

-- Define the work completion time for Person B
def person_b_time : ℝ := 40

-- Theorem statement
theorem work_completion_time :
  (1 / person_a_time + 1 / person_b_time = 1 / combined_time) := by
  sorry

end work_completion_time_l1163_116379


namespace fraction_equality_l1163_116380

theorem fraction_equality : (10^9 + 10^6) / (3 * 10^4) = 100100 / 3 := by
  sorry

end fraction_equality_l1163_116380


namespace mary_balloon_count_l1163_116395

/-- The number of black balloons Nancy has -/
def nancy_balloons : ℕ := 7

/-- The factor by which Mary's balloons exceed Nancy's -/
def mary_factor : ℕ := 4

/-- The number of black balloons Mary has -/
def mary_balloons : ℕ := nancy_balloons * mary_factor

theorem mary_balloon_count : mary_balloons = 28 := by
  sorry

end mary_balloon_count_l1163_116395


namespace stratified_sampling_problem_l1163_116310

theorem stratified_sampling_problem (high_school_students : ℕ) (middle_school_students : ℕ)
  (middle_school_sample : ℕ) (total_sample : ℕ) :
  high_school_students = 3500 →
  middle_school_students = 1500 →
  middle_school_sample = 30 →
  (middle_school_students : ℚ) / (high_school_students + middle_school_students : ℚ) * middle_school_sample = total_sample →
  total_sample = 100 := by
sorry


end stratified_sampling_problem_l1163_116310


namespace wendy_ribbon_left_l1163_116360

/-- The amount of ribbon Wendy has left after using some for wrapping presents -/
def ribbon_left (initial : ℕ) (used : ℕ) : ℕ :=
  initial - used

/-- Theorem: Given Wendy bought 84 inches of ribbon and used 46 inches, 
    the amount of ribbon left is 38 inches -/
theorem wendy_ribbon_left : 
  ribbon_left 84 46 = 38 := by
  sorry

end wendy_ribbon_left_l1163_116360


namespace cube_gt_iff_gt_l1163_116336

theorem cube_gt_iff_gt (a b : ℝ) : a^3 > b^3 ↔ a > b := by sorry

end cube_gt_iff_gt_l1163_116336


namespace retailer_markup_percentage_l1163_116318

/-- Proves that a retailer who marks up goods by x%, offers a 15% discount, 
    and makes 27.5% profit, must have marked up the goods by 50% --/
theorem retailer_markup_percentage 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (actual_profit_percentage : ℝ)
  (h1 : discount_percentage = 15)
  (h2 : actual_profit_percentage = 27.5)
  (h3 : cost_price > 0)
  (h4 : markup_percentage > 0)
  : 
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  selling_price = cost_price * (1 + actual_profit_percentage / 100) →
  markup_percentage = 50 := by
  sorry

end retailer_markup_percentage_l1163_116318


namespace fourth_person_height_l1163_116321

/-- Heights of four people in increasing order -/
def Heights := Fin 4 → ℝ

/-- The common difference between the heights of the first three people -/
def common_difference (h : Heights) : ℝ := h 1 - h 0

theorem fourth_person_height (h : Heights) 
  (increasing : ∀ i j, i < j → h i < h j)
  (common_diff : h 2 - h 1 = h 1 - h 0)
  (last_diff : h 3 - h 2 = 6)
  (avg_height : (h 0 + h 1 + h 2 + h 3) / 4 = 77) :
  h 3 = h 0 + 2 * (common_difference h) + 6 := by
  sorry

end fourth_person_height_l1163_116321


namespace initial_distance_proof_l1163_116366

/-- The initial distance between two cars on a main road --/
def initial_distance : ℝ := 165

/-- The total distance traveled by the first car --/
def car1_distance : ℝ := 65

/-- The distance traveled by the second car --/
def car2_distance : ℝ := 62

/-- The final distance between the two cars --/
def final_distance : ℝ := 38

/-- Theorem stating that the initial distance is correct given the problem conditions --/
theorem initial_distance_proof :
  initial_distance = car1_distance + car2_distance + final_distance :=
by sorry

end initial_distance_proof_l1163_116366


namespace fixed_point_exponential_l1163_116331

/-- The function f(x) = a^(x-1) + 4 always passes through the point (1, 5) for any a > 0 and a ≠ 1 -/
theorem fixed_point_exponential (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 4
  f 1 = 5 := by sorry

end fixed_point_exponential_l1163_116331


namespace evenPerfectSquareFactorsCount_l1163_116386

/-- The number of even perfect square factors of 2^6 * 7^12 * 3^2 -/
def evenPerfectSquareFactors : ℕ :=
  let n : ℕ := 2^6 * 7^12 * 3^2
  -- Count of valid combinations for exponents a, b, c
  let aCount : ℕ := 3  -- a can be 2, 4, or 6
  let bCount : ℕ := 7  -- b can be 0, 2, 4, 6, 8, 10, 12
  let cCount : ℕ := 2  -- c can be 0 or 2
  aCount * bCount * cCount

/-- Theorem: The number of even perfect square factors of 2^6 * 7^12 * 3^2 is 42 -/
theorem evenPerfectSquareFactorsCount : evenPerfectSquareFactors = 42 := by
  sorry

end evenPerfectSquareFactorsCount_l1163_116386


namespace multiply_add_theorem_l1163_116370

theorem multiply_add_theorem : 15 * 30 + 45 * 15 + 90 = 1215 := by
  sorry

end multiply_add_theorem_l1163_116370


namespace arithmetic_geometric_inequality_l1163_116323

/-- Given two sequences (aₖ) and (bₖ) satisfying certain conditions, 
    prove that aₖ > bₖ for all k between 2 and n-1 inclusive. -/
theorem arithmetic_geometric_inequality (n : ℕ) (a b : ℕ → ℝ) 
  (h_n : n ≥ 3)
  (h_a_arith : ∀ k l : ℕ, k < l → l ≤ n → a l - a k = (l - k) * (a 2 - a 1))
  (h_b_geom : ∀ k l : ℕ, k < l → l ≤ n → b l / b k = (b 2 / b 1) ^ (l - k))
  (h_a_pos : ∀ k : ℕ, k ≤ n → 0 < a k)
  (h_b_pos : ∀ k : ℕ, k ≤ n → 0 < b k)
  (h_a_inc : ∀ k : ℕ, k < n → a k < a (k + 1))
  (h_b_inc : ∀ k : ℕ, k < n → b k < b (k + 1))
  (h_eq_first : a 1 = b 1)
  (h_eq_last : a n = b n) :
  ∀ k : ℕ, 2 ≤ k → k < n → a k > b k :=
sorry

end arithmetic_geometric_inequality_l1163_116323


namespace find_C_value_l1163_116324

theorem find_C_value (D : ℝ) (h1 : 4 * C - 2 * D - 3 = 26) (h2 : D = 3) : C = 8.75 := by
  sorry

end find_C_value_l1163_116324


namespace equation_solution_l1163_116320

theorem equation_solution : 
  ∃ x₁ x₂ : ℚ, x₁ = 8/3 ∧ x₂ = 2 ∧ 
  (∀ x : ℚ, x^2 - 6*x + 9 = (5 - 2*x)^2 ↔ x = x₁ ∨ x = x₂) :=
sorry

end equation_solution_l1163_116320


namespace probability_no_consecutive_ones_l1163_116361

/-- Sequence without consecutive ones -/
def SeqWithoutConsecutiveOnes (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | (n+2) => SeqWithoutConsecutiveOnes (n+1) + SeqWithoutConsecutiveOnes n

/-- Total number of possible sequences -/
def TotalSequences (n : ℕ) : ℕ := 2^n

theorem probability_no_consecutive_ones :
  (SeqWithoutConsecutiveOnes 12 : ℚ) / (TotalSequences 12) = 377 / 4096 := by
  sorry

#eval SeqWithoutConsecutiveOnes 12
#eval TotalSequences 12

end probability_no_consecutive_ones_l1163_116361


namespace right_angle_constraint_l1163_116375

/-- Given two points A and B on the x-axis, and a point P on a line,
    prove that if ∠APB is a right angle, then the distance between A and B
    is at least 10 units. -/
theorem right_angle_constraint (m : ℝ) (h_m : m > 0) :
  (∃ (x y : ℝ), 3 * x + 4 * y + 25 = 0 ∧
    ((x + m) * (x - m) + y * y = 0)) →
  m ≥ 5 :=
by sorry

end right_angle_constraint_l1163_116375


namespace race_heartbeats_l1163_116322

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Proves that the total number of heartbeats during a 30-mile race is 21600 -/
theorem race_heartbeats :
  let heart_rate : ℕ := 120  -- beats per minute
  let pace : ℕ := 6          -- minutes per mile
  let distance : ℕ := 30     -- miles
  total_heartbeats heart_rate pace distance = 21600 := by
sorry

#eval total_heartbeats 120 6 30

end race_heartbeats_l1163_116322


namespace masha_can_pay_with_five_ruble_coins_l1163_116300

theorem masha_can_pay_with_five_ruble_coins 
  (p c n : ℕ+) 
  (h : 2 * p.val + c.val + 7 * n.val = 100) : 
  5 ∣ (p.val + 3 * c.val + n.val) := by
  sorry

end masha_can_pay_with_five_ruble_coins_l1163_116300


namespace diamond_three_eight_l1163_116348

-- Define the operation ⋄
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem diamond_three_eight : diamond 3 8 = 60 := by
  sorry

end diamond_three_eight_l1163_116348


namespace sum_parity_eq_parity_of_M_l1163_116390

/-- Represents the parity of a number -/
inductive Parity
  | Even
  | Odd

/-- The sum of N even numbers and M odd numbers -/
def sum_parity (N M : ℕ) : Parity :=
  match M % 2 with
  | 0 => Parity.Even
  | _ => Parity.Odd

/-- The parity of a natural number -/
def parity (n : ℕ) : Parity :=
  match n % 2 with
  | 0 => Parity.Even
  | _ => Parity.Odd

/-- Theorem: The parity of the sum of N even numbers and M odd numbers
    is equal to the parity of M -/
theorem sum_parity_eq_parity_of_M (N M : ℕ) :
  sum_parity N M = parity M := by sorry

end sum_parity_eq_parity_of_M_l1163_116390


namespace greatest_four_digit_divisible_by_3_5_6_l1163_116359

theorem greatest_four_digit_divisible_by_3_5_6 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ 3 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n → n ≤ 9990 :=
by
  sorry

end greatest_four_digit_divisible_by_3_5_6_l1163_116359


namespace sum_of_fractions_zero_l1163_116383

theorem sum_of_fractions_zero (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h_sum : a + b + c = d) : 
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 0 := by
sorry

end sum_of_fractions_zero_l1163_116383


namespace greatest_y_value_l1163_116340

theorem greatest_y_value (y : ℝ) : 3 * y^2 + 5 * y + 3 = 3 → y ≤ 0 := by
  sorry

end greatest_y_value_l1163_116340


namespace x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l1163_116350

theorem x_gt_one_sufficient_not_necessary_for_abs_x_gt_one :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧ 
  ¬(∀ x : ℝ, |x| > 1 → x > 1) :=
by sorry

end x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l1163_116350


namespace collinearity_proof_collinear_vectors_l1163_116338

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable section

def are_collinear (a b c : V) : Prop := ∃ (t : ℝ), c - a = t • (b - a)

theorem collinearity_proof 
  (e₁ e₂ A B C D : V) 
  (h₁ : e₁ ≠ 0) 
  (h₂ : e₂ ≠ 0) 
  (h₃ : ¬ ∃ (t : ℝ), e₁ = t • e₂) 
  (h₄ : B - A = e₁ + e₂) 
  (h₅ : C - B = 2 • e₁ + 8 • e₂) 
  (h₆ : D - C = 3 • (e₁ - e₂)) : 
  are_collinear A B D :=
sorry

theorem collinear_vectors 
  (e₁ e₂ : V) 
  (h₁ : e₁ ≠ 0) 
  (h₂ : e₂ ≠ 0) 
  (h₃ : ¬ ∃ (t : ℝ), e₁ = t • e₂) :
  ∀ (k : ℝ), (∃ (t : ℝ), k • e₁ + e₂ = t • (e₁ + k • e₂)) ↔ (k = 1 ∨ k = -1) :=
sorry

end

end collinearity_proof_collinear_vectors_l1163_116338


namespace common_factor_of_polynomial_l1163_116317

/-- The common factor of the polynomial 3ma^2 - 6mab is 3ma -/
theorem common_factor_of_polynomial (m a b : ℤ) :
  ∃ (k₁ k₂ : ℤ), 3 * m * a^2 - 6 * m * a * b = 3 * m * a * (k₁ * a + k₂ * b) :=
sorry

end common_factor_of_polynomial_l1163_116317


namespace washing_machine_capacity_l1163_116367

/-- Given a total amount of clothes and a number of washing machines, 
    calculate the amount of clothes one washing machine can wash per day. -/
def clothes_per_machine (total_clothes : ℕ) (num_machines : ℕ) : ℕ :=
  total_clothes / num_machines

/-- Theorem stating that for 200 pounds of clothes and 8 machines, 
    each machine can wash 25 pounds per day. -/
theorem washing_machine_capacity : clothes_per_machine 200 8 = 25 := by
  sorry

end washing_machine_capacity_l1163_116367


namespace dishonest_dealer_profit_l1163_116382

/-- The profit percentage of a dishonest dealer who uses 800 grams instead of 1000 grams per kg -/
theorem dishonest_dealer_profit (actual_weight : ℕ) (claimed_weight : ℕ) : 
  actual_weight = 800 ∧ claimed_weight = 1000 → 
  (claimed_weight - actual_weight : ℚ) / claimed_weight * 100 = 20 := by
  sorry

#check dishonest_dealer_profit

end dishonest_dealer_profit_l1163_116382


namespace alice_profit_l1163_116355

/-- Calculates the profit from selling friendship bracelets -/
def calculate_profit (total_bracelets : ℕ) (material_cost : ℚ) (given_away : ℕ) (price_per_bracelet : ℚ) : ℚ :=
  let bracelets_sold := total_bracelets - given_away
  let revenue := (bracelets_sold : ℚ) * price_per_bracelet
  revenue - material_cost

/-- Theorem: Alice's profit from selling friendship bracelets is $8.00 -/
theorem alice_profit :
  calculate_profit 52 3 8 (1/4) = 8 := by
  sorry

end alice_profit_l1163_116355


namespace seventh_grade_percentage_l1163_116307

theorem seventh_grade_percentage 
  (seventh_graders : ℕ) 
  (sixth_graders : ℕ) 
  (sixth_grade_percentage : ℚ) :
  seventh_graders = 64 →
  sixth_graders = 76 →
  sixth_grade_percentage = 38/100 →
  (↑seventh_graders : ℚ) / (↑sixth_graders / sixth_grade_percentage) = 32/100 :=
by sorry

end seventh_grade_percentage_l1163_116307


namespace fourteen_stones_per_bracelet_l1163_116347

/-- Given a total number of stones and a number of bracelets, 
    calculate the number of stones per bracelet. -/
def stones_per_bracelet (total_stones : ℕ) (num_bracelets : ℕ) : ℕ :=
  total_stones / num_bracelets

/-- Theorem: Given 140 stones and 10 bracelets, 
    prove that there are 14 stones per bracelet. -/
theorem fourteen_stones_per_bracelet :
  stones_per_bracelet 140 10 = 14 := by
  sorry

end fourteen_stones_per_bracelet_l1163_116347


namespace greatest_integer_problem_l1163_116301

theorem greatest_integer_problem : 
  ∃ (n : ℕ), n < 150 ∧ 
  (∃ (k l : ℕ), n = 9 * k - 2 ∧ n = 6 * l - 4) ∧
  (∀ (m : ℕ), m < 150 → 
    (∃ (k' l' : ℕ), m = 9 * k' - 2 ∧ m = 6 * l' - 4) → 
    m ≤ n) ∧
  n = 146 :=
sorry

end greatest_integer_problem_l1163_116301


namespace root_sum_eighth_power_l1163_116396

theorem root_sum_eighth_power (r s : ℝ) : 
  (r^2 - 2*r*Real.sqrt 6 + 3 = 0) →
  (s^2 - 2*s*Real.sqrt 6 + 3 = 0) →
  r^8 + s^8 = 93474 := by
sorry

end root_sum_eighth_power_l1163_116396


namespace two_color_draw_count_l1163_116374

def total_balls : ℕ := 6
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def blue_balls : ℕ := 1
def draw_count : ℕ := 3

def ways_two_colors : ℕ := 13

theorem two_color_draw_count :
  ways_two_colors = (total_balls.choose draw_count) - 
    (red_balls * white_balls * blue_balls) - 
    (if white_balls ≥ draw_count then 1 else 0) :=
by sorry

end two_color_draw_count_l1163_116374


namespace negation_of_universal_proposition_l1163_116397

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x - 3 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x - 3 < 0) := by
  sorry

end negation_of_universal_proposition_l1163_116397


namespace cab_driver_average_income_l1163_116309

theorem cab_driver_average_income 
  (incomes : List ℝ) 
  (h_incomes : incomes = [400, 250, 650, 400, 500]) 
  (h_days : incomes.length = 5) : 
  (incomes.sum / incomes.length : ℝ) = 440 := by
sorry

end cab_driver_average_income_l1163_116309


namespace jake_bitcoin_proportion_l1163_116334

/-- The proportion of bitcoins Jake gave to his brother -/
def proportion_to_brother : ℚ := 1/2

/-- Jake's initial fortune in bitcoins -/
def initial_fortune : ℕ := 80

/-- First donation amount in bitcoins -/
def first_donation : ℕ := 20

/-- Second donation amount in bitcoins -/
def second_donation : ℕ := 10

/-- Jake's final amount of bitcoins -/
def final_amount : ℕ := 80

theorem jake_bitcoin_proportion :
  let remaining_after_first_donation := initial_fortune - first_donation
  let remaining_after_giving_to_brother := remaining_after_first_donation * (1 - proportion_to_brother)
  let amount_after_tripling := remaining_after_giving_to_brother * 3
  amount_after_tripling - second_donation = final_amount :=
by sorry

end jake_bitcoin_proportion_l1163_116334


namespace shortest_segment_right_triangle_l1163_116337

theorem shortest_segment_right_triangle (a b c : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) : 
  ∃ (t : ℝ), t = 2 * Real.sqrt 3 ∧ 
  ∀ (x y : ℝ), x * y = (a * b) / 2 → 
  t ≤ Real.sqrt (x^2 + y^2 - 2 * x * y * (b / c)) := by
  sorry

end shortest_segment_right_triangle_l1163_116337


namespace yulgi_allowance_l1163_116363

theorem yulgi_allowance (Y G : ℕ) 
  (sum : Y + G = 6000)
  (sum_minus_diff : Y + G - (Y - G) = 4800)
  (Y_greater : Y > G) : Y = 3600 := by
sorry

end yulgi_allowance_l1163_116363


namespace draw_with_replacement_l1163_116393

-- Define the number of balls in the bin
def num_balls : ℕ := 15

-- Define the number of draws
def num_draws : ℕ := 4

-- Define the function to calculate the number of ways to draw balls
def ways_to_draw (n : ℕ) (k : ℕ) : ℕ := n ^ k

-- Theorem statement
theorem draw_with_replacement :
  ways_to_draw num_balls num_draws = 50625 := by
  sorry

end draw_with_replacement_l1163_116393


namespace product_of_integers_l1163_116391

theorem product_of_integers (w x y z : ℤ) : 
  0 < w → w < x → x < y → y < z → w + z = 5 → w * x * y * z = 36 := by
  sorry

end product_of_integers_l1163_116391


namespace modular_inverse_of_5_mod_37_l1163_116319

theorem modular_inverse_of_5_mod_37 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 36 ∧ (5 * x) % 37 = 1 := by
  sorry

end modular_inverse_of_5_mod_37_l1163_116319


namespace petya_final_amount_l1163_116314

/-- Represents the juice distribution problem between Petya and Masha -/
structure JuiceDistribution where
  total : ℝ
  petya_initial : ℝ
  masha_initial : ℝ
  transferred : ℝ
  h_total : total = 10
  h_initial_sum : petya_initial + masha_initial = total
  h_after_transfer : petya_initial + transferred = 3 * (masha_initial - transferred)
  h_masha_reduction : masha_initial - transferred = (1/3) * masha_initial

/-- Theorem stating that Petya's final amount of juice is 7.5 liters -/
theorem petya_final_amount (jd : JuiceDistribution) : 
  jd.petya_initial + jd.transferred = 7.5 := by
  sorry


end petya_final_amount_l1163_116314


namespace no_adjacent_standing_probability_l1163_116343

/-- Represents a person's standing state -/
inductive State
  | Standing
  | Seated

/-- Represents the circular arrangement of people -/
def Arrangement := Vector State 10

/-- Checks if two adjacent people are standing -/
def hasAdjacentStanding (arr : Arrangement) : Bool :=
  sorry

/-- Checks if an arrangement is valid according to the problem rules -/
def isValidArrangement (arr : Arrangement) : Bool :=
  sorry

/-- The total number of possible arrangements -/
def totalArrangements : Nat :=
  2^8

/-- The number of valid arrangements where no two adjacent people stand -/
def validArrangements : Nat :=
  sorry

theorem no_adjacent_standing_probability :
  (validArrangements : ℚ) / totalArrangements = 1 / 64 :=
sorry

end no_adjacent_standing_probability_l1163_116343


namespace robs_double_cards_fraction_l1163_116352

theorem robs_double_cards_fraction (total_cards : ℕ) (jess_doubles : ℕ) (jess_ratio : ℕ) :
  total_cards = 24 →
  jess_doubles = 40 →
  jess_ratio = 5 →
  (jess_doubles / jess_ratio : ℚ) / total_cards = 1 / 3 := by
  sorry

end robs_double_cards_fraction_l1163_116352


namespace range_of_2cos_squared_l1163_116399

theorem range_of_2cos_squared (x : ℝ) : 0 ≤ 2 * (Real.cos x)^2 ∧ 2 * (Real.cos x)^2 ≤ 2 := by
  sorry

end range_of_2cos_squared_l1163_116399


namespace investment_result_unique_initial_investment_l1163_116381

/-- Represents the growth of an investment over time with compound interest and additional investments. -/
def investment_growth (initial_investment : ℝ) : ℝ :=
  let after_compound := initial_investment * (1 + 0.20)^3
  let after_triple := after_compound * 3
  after_triple * (1 + 0.15)

/-- Theorem stating that an initial investment of $10,000 results in $59,616 after the given growth pattern. -/
theorem investment_result : investment_growth 10000 = 59616 := by
  sorry

/-- Theorem proving the uniqueness of the initial investment that results in $59,616. -/
theorem unique_initial_investment (x : ℝ) :
  investment_growth x = 59616 → x = 10000 := by
  sorry

end investment_result_unique_initial_investment_l1163_116381


namespace tangent_perpendicular_points_l1163_116312

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_perpendicular_points :
  ∀ x y : ℝ, f x = y →
    (3 * x^2 + 1 = 4 ∨ 3 * x^2 + 1 = -1/4) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) :=
by sorry

end tangent_perpendicular_points_l1163_116312


namespace sufficient_not_necessary_condition_l1163_116354

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y, x > y ∧ y > 0 → x / y > 1) ∧
  (∃ x y, x / y > 1 ∧ ¬(x > y ∧ y > 0)) :=
sorry

end sufficient_not_necessary_condition_l1163_116354


namespace regular_triangular_pyramid_volume_l1163_116342

/-- The volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume 
  (a b γ : ℝ) 
  (h_a : a > 0) 
  (h_b : b > 0) 
  (h_γ : 0 < γ ∧ γ < π) : 
  ∃ V : ℝ, V = (1/3) * (a^2 * Real.sqrt 3 / 4) * 
    Real.sqrt (b^2 - (a * Real.sqrt 3 / (2 * Real.cos (γ/2)))^2) := by
  sorry

end regular_triangular_pyramid_volume_l1163_116342


namespace beth_crayon_packs_l1163_116371

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 10

/-- The number of extra crayons not in packs -/
def extra_crayons : ℕ := 6

/-- The total number of crayons Beth has -/
def total_crayons : ℕ := 46

/-- The number of packs of crayons Beth has -/
def num_packs : ℕ := (total_crayons - extra_crayons) / crayons_per_pack

theorem beth_crayon_packs :
  num_packs = 4 :=
sorry

end beth_crayon_packs_l1163_116371


namespace grand_forest_trail_length_l1163_116306

/-- Represents the length of Jamie's hike on the Grand Forest Trail -/
def GrandForestTrail : Type :=
  { hike : Vector ℝ 5 // 
    hike.get 0 + hike.get 1 + hike.get 2 = 42 ∧
    (hike.get 1 + hike.get 2) / 2 = 15 ∧
    hike.get 3 + hike.get 4 = 40 ∧
    hike.get 0 + hike.get 3 = 36 }

/-- The total length of the Grand Forest Trail is 82 miles -/
theorem grand_forest_trail_length (hike : GrandForestTrail) :
  hike.val.get 0 + hike.val.get 1 + hike.val.get 2 + hike.val.get 3 + hike.val.get 4 = 82 :=
by sorry

end grand_forest_trail_length_l1163_116306


namespace principal_is_900_l1163_116339

/-- Proves that given the conditions of the problem, the principal must be $900 -/
theorem principal_is_900 (P R : ℝ) : 
  (P * (R + 3) * 3) / 100 = (P * R * 3) / 100 + 81 → P = 900 := by
  sorry

end principal_is_900_l1163_116339


namespace saucer_surface_area_l1163_116344

/-- The surface area of a saucer with given dimensions -/
theorem saucer_surface_area (radius : ℝ) (rim_thickness : ℝ) (cap_height : ℝ) 
  (h1 : radius = 3)
  (h2 : rim_thickness = 1)
  (h3 : cap_height = 1.5) :
  2 * Real.pi * radius * cap_height + Real.pi * (radius^2 - (radius - rim_thickness)^2) = 14 * Real.pi :=
by sorry

end saucer_surface_area_l1163_116344


namespace christophers_speed_l1163_116305

/-- Given a distance of 5 miles and a time of 1.25 hours, the speed is 4 miles per hour -/
theorem christophers_speed (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 5 → time = 1.25 → speed = distance / time → speed = 4 := by
  sorry

end christophers_speed_l1163_116305


namespace sum_f_negative_l1163_116351

-- Define the function f
variable (f : ℝ → ℝ)

-- State the properties of f
axiom f_symmetry (x : ℝ) : f (4 - x) = -f x
axiom f_monotone_increasing (x y : ℝ) : x > 2 → y > x → f y > f x

-- Define the theorem
theorem sum_f_negative (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ < 4) 
  (h2 : (x₁ - 2) * (x₂ - 2) < 0) : 
  f x₁ + f x₂ < 0 :=
sorry

end sum_f_negative_l1163_116351
