import Mathlib

namespace NUMINAMATH_CALUDE_harry_travel_time_l2798_279838

/-- Calculates the total travel time for Harry's journey --/
def total_travel_time (initial_bus_time remaining_bus_time : ℕ) : ℕ :=
  let bus_time := initial_bus_time + remaining_bus_time
  let walk_time := bus_time / 2
  bus_time + walk_time

/-- Proves that Harry's total travel time is 60 minutes --/
theorem harry_travel_time :
  total_travel_time 15 25 = 60 := by
  sorry

end NUMINAMATH_CALUDE_harry_travel_time_l2798_279838


namespace NUMINAMATH_CALUDE_M_subset_N_l2798_279815

-- Define the set M
def M : Set ℝ := {x | ∃ k : ℤ, x = (k / 2 : ℝ) * 180 + 45}

-- Define the set N
def N : Set ℝ := {x | ∃ k : ℤ, x = (k / 4 : ℝ) * 180 + 45}

-- Theorem statement
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l2798_279815


namespace NUMINAMATH_CALUDE_car_distance_theorem_l2798_279802

/-- The distance covered by a car given initial time and adjusted speed -/
theorem car_distance_theorem (initial_time : ℝ) (adjusted_speed : ℝ) : 
  initial_time = 6 →
  adjusted_speed = 60 →
  (initial_time * 3 / 2) * adjusted_speed = 540 := by
sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l2798_279802


namespace NUMINAMATH_CALUDE_gcd_of_12_and_20_l2798_279874

theorem gcd_of_12_and_20 : Nat.gcd 12 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_12_and_20_l2798_279874


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2798_279829

/-- Given a point M and two lines, this theorem proves that the second line passes through M and is perpendicular to the first line. -/
theorem perpendicular_line_through_point (x₀ y₀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  (a₁ * x₀ + b₁ * y₀ + c₁ ≠ 0) →  -- M is not on the first line
  (a₁ * b₂ = -a₂ * b₁) →          -- Lines are perpendicular
  (a₂ * x₀ + b₂ * y₀ + c₂ = 0) →  -- Second line passes through M
  ∃ (k : ℝ), k ≠ 0 ∧ a₂ = k * 4 ∧ b₂ = k * 3 ∧ c₂ = k * (-13) ∧
             a₁ = k * 3 ∧ b₁ = k * (-4) ∧ c₁ = k * 6 ∧
             x₀ = 4 ∧ y₀ = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2798_279829


namespace NUMINAMATH_CALUDE_remainder_equality_l2798_279847

theorem remainder_equality (P P' D : ℕ) (hP : P > P') : 
  let R := P % D
  let R' := P' % D
  let r := (P * P') % D
  let r' := (R * R') % D
  r = r' := by
sorry

end NUMINAMATH_CALUDE_remainder_equality_l2798_279847


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2798_279877

theorem triangle_angle_measure (P Q R : ℝ) (h1 : R = 3 * Q) (h2 : Q = 30) :
  P + Q + R = 180 → P = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2798_279877


namespace NUMINAMATH_CALUDE_sqrt_13_parts_sum_l2798_279893

theorem sqrt_13_parts_sum (a b : ℝ) : 
  (3 : ℝ) < Real.sqrt 13 ∧ Real.sqrt 13 < 4 →
  a = ⌊Real.sqrt 13⌋ →
  b = Real.sqrt 13 - ⌊Real.sqrt 13⌋ →
  a^2 + b - Real.sqrt 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_13_parts_sum_l2798_279893


namespace NUMINAMATH_CALUDE_school_colors_percentage_l2798_279819

theorem school_colors_percentage (N : ℝ) (h_pos : N > 0) : 
  let girls := 0.45 * N
  let boys := N - girls
  let girls_in_colors := 0.60 * girls
  let boys_in_colors := 0.80 * boys
  let total_in_colors := girls_in_colors + boys_in_colors
  (total_in_colors / N) = 0.71 := by
  sorry

end NUMINAMATH_CALUDE_school_colors_percentage_l2798_279819


namespace NUMINAMATH_CALUDE_expression_simplification_l2798_279890

theorem expression_simplification 
  (a b c x : ℝ) 
  (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : 
  (x - a)^3 + a*x / ((a - b)*(a - c)) + 
  (x - b)^3 + b*x / ((b - a)*(b - c)) + 
  (x - c)^3 + c*x / ((c - a)*(c - b)) = 
  a + b + c + 3*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2798_279890


namespace NUMINAMATH_CALUDE_team_games_count_l2798_279899

/-- The number of games the team plays -/
def total_games : ℕ := 14

/-- The number of shots John gets per foul -/
def shots_per_foul : ℕ := 2

/-- The number of times John gets fouled per game -/
def fouls_per_game : ℕ := 5

/-- The percentage of games John plays -/
def games_played_percentage : ℚ := 4/5

/-- The total number of free throws John gets -/
def total_free_throws : ℕ := 112

theorem team_games_count :
  total_games = 14 ∧
  shots_per_foul = 2 ∧
  fouls_per_game = 5 ∧
  games_played_percentage = 4/5 ∧
  total_free_throws = 112 →
  (total_games : ℚ) * games_played_percentage * (shots_per_foul * fouls_per_game) = total_free_throws := by
  sorry

end NUMINAMATH_CALUDE_team_games_count_l2798_279899


namespace NUMINAMATH_CALUDE_expression_simplification_l2798_279892

theorem expression_simplification (x : ℤ) 
  (h1 : -1 ≤ x) (h2 : x < 2) (h3 : x ≠ 1) : 
  ((x + 1) / (x^2 - 1) + x / (x - 1)) / ((x + 1) / (x^2 - 2*x + 1)) = x - 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2798_279892


namespace NUMINAMATH_CALUDE_max_value_of_f_l2798_279885

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) * (x - a)

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x - 4

theorem max_value_of_f (a : ℝ) :
  (f' a (-1) = 0) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 4, f a x = 42) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a x ≤ 42) :=
by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_max_value_of_f_l2798_279885


namespace NUMINAMATH_CALUDE_minimum_value_problem_l2798_279886

theorem minimum_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) ≥ -2040200 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (a + 1/b) * (a + 1/b - 2020) + (b + 1/a) * (b + 1/a - 2020) = -2040200 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_problem_l2798_279886


namespace NUMINAMATH_CALUDE_m_minus_n_values_l2798_279844

theorem m_minus_n_values (m n : ℤ) 
  (hm : |m| = 14)
  (hn : |n| = 23)
  (hmn_pos : m + n > 0) :
  m - n = -9 ∨ m - n = -37 := by
sorry

end NUMINAMATH_CALUDE_m_minus_n_values_l2798_279844


namespace NUMINAMATH_CALUDE_max_popsicles_with_10_dollars_l2798_279882

/-- Represents the available popsicle purchase options -/
structure PopsicleOption where
  quantity : ℕ
  price : ℕ

/-- Finds the maximum number of popsicles that can be purchased with a given budget -/
def maxPopsicles (options : List PopsicleOption) (budget : ℕ) : ℕ :=
  sorry

/-- The main theorem proving that 23 is the maximum number of popsicles that can be purchased -/
theorem max_popsicles_with_10_dollars :
  let options : List PopsicleOption := [
    ⟨1, 1⟩,  -- Single popsicle
    ⟨3, 2⟩,  -- 3-popsicle box
    ⟨5, 3⟩,  -- 5-popsicle box
    ⟨10, 4⟩  -- 10-popsicle box
  ]
  let budget := 10
  maxPopsicles options budget = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_popsicles_with_10_dollars_l2798_279882


namespace NUMINAMATH_CALUDE_janice_stairs_walked_l2798_279837

/-- The number of flights of stairs to Janice's office -/
def flights_to_office : ℕ := 3

/-- The number of times Janice goes up the stairs in a day -/
def times_up : ℕ := 5

/-- The number of times Janice goes down the stairs in a day -/
def times_down : ℕ := 3

/-- The total number of flights Janice walks in a day -/
def total_flights : ℕ := flights_to_office * times_up + flights_to_office * times_down

theorem janice_stairs_walked : total_flights = 24 := by
  sorry

end NUMINAMATH_CALUDE_janice_stairs_walked_l2798_279837


namespace NUMINAMATH_CALUDE_smallest_natural_solution_l2798_279801

theorem smallest_natural_solution (n : ℕ) : 
  (2023 / 2022 : ℝ) ^ (36 * (1 - (2/3 : ℝ)^(n+1)) / (1 - 2/3)) > (2023 / 2022 : ℝ) ^ 96 ↔ n ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_natural_solution_l2798_279801


namespace NUMINAMATH_CALUDE_inequality_proof_l2798_279824

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1/3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2798_279824


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l2798_279804

theorem systematic_sampling_interval 
  (population : ℕ) 
  (sample_size : ℕ) 
  (h1 : population = 800) 
  (h2 : sample_size = 40) 
  (h3 : population > 0) 
  (h4 : sample_size > 0) :
  population / sample_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l2798_279804


namespace NUMINAMATH_CALUDE_l_shape_area_l2798_279891

/-- The area of an L shape formed by removing a smaller rectangle from a larger rectangle -/
theorem l_shape_area (big_length big_width small_length small_width : ℕ) : 
  big_length = 8 →
  big_width = 5 →
  small_length = big_length - 2 →
  small_width = big_width - 2 →
  big_length * big_width - small_length * small_width = 22 :=
by sorry

end NUMINAMATH_CALUDE_l_shape_area_l2798_279891


namespace NUMINAMATH_CALUDE_geometric_sequence_ak_l2798_279825

/-- Given a geometric sequence {a_n} with sum S_n = k * 2^n - 3, prove a_k = 12 -/
theorem geometric_sequence_ak (k : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = k * 2^n - 3) →
  (∀ n, S (n + 1) - S n = a (n + 1)) →
  (∀ n, a (n + 1) = 2 * a n) →
  a k = 12 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ak_l2798_279825


namespace NUMINAMATH_CALUDE_sean_whistles_l2798_279868

/-- Given that Sean has 32 more whistles than Charles and Charles has 13 whistles,
    prove that Sean has 45 whistles. -/
theorem sean_whistles (charles_whistles : ℕ) (sean_extra_whistles : ℕ) 
  (h1 : charles_whistles = 13)
  (h2 : sean_extra_whistles = 32) :
  charles_whistles + sean_extra_whistles = 45 := by
  sorry

end NUMINAMATH_CALUDE_sean_whistles_l2798_279868


namespace NUMINAMATH_CALUDE_movie_of_the_year_threshold_l2798_279860

def total_members : ℕ := 775
def threshold : ℚ := 1/4

theorem movie_of_the_year_threshold : 
  ∀ n : ℕ, (n : ℚ) ≥ threshold * total_members ∧ 
  ∀ m : ℕ, m < n → (m : ℚ) < threshold * total_members → n = 194 := by
  sorry

end NUMINAMATH_CALUDE_movie_of_the_year_threshold_l2798_279860


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2798_279821

theorem tangent_line_to_circle (m : ℝ) : 
  (∃ x y : ℝ, y = m * x ∧ x^2 + y^2 - 4*x + 2 = 0 ∧
   ∀ x' y' : ℝ, y' = m * x' → x'^2 + y'^2 - 4*x' + 2 ≥ 0) →
  m = 1 ∨ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2798_279821


namespace NUMINAMATH_CALUDE_hotel_stay_duration_l2798_279858

theorem hotel_stay_duration (cost_per_night_per_person : ℕ) (num_people : ℕ) (total_cost : ℕ) : 
  cost_per_night_per_person = 40 →
  num_people = 3 →
  total_cost = 360 →
  total_cost = cost_per_night_per_person * num_people * 3 :=
by
  sorry

end NUMINAMATH_CALUDE_hotel_stay_duration_l2798_279858


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_radius_from_inscribed_rectangle_l2798_279866

/-- The area of a semicircle with an inscribed 1×3 rectangle -/
theorem semicircle_area_with_inscribed_rectangle (r : ℝ) : 
  (r^2 = 5/4) → -- The radius squared equals 5/4
  (π * r^2 / 2 = 5*π/4) := -- The area of the semicircle equals 5π/4
by sorry

/-- The relationship between the radius and the inscribed rectangle -/
theorem radius_from_inscribed_rectangle (r : ℝ) :
  (r^2 = 5/4) ↔ -- The radius squared equals 5/4
  (∃ (w h : ℝ), w = 1 ∧ h = 3 ∧ w^2 + (h/2)^2 = r^2) := -- There exists a 1×3 rectangle inscribed
by sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_radius_from_inscribed_rectangle_l2798_279866


namespace NUMINAMATH_CALUDE_max_hot_dogs_is_3250_l2798_279813

/-- Represents the available pack sizes and their prices --/
structure PackInfo where
  size : Nat
  price : Rat

/-- The maximum number of hot dogs that can be purchased with the given budget --/
def maxHotDogs (packs : List PackInfo) (budget : Rat) : Nat :=
  sorry

/-- The available pack sizes and prices --/
def availablePacks : List PackInfo := [
  ⟨8, 155/100⟩,
  ⟨20, 305/100⟩,
  ⟨250, 2295/100⟩
]

/-- The budget in dollars --/
def totalBudget : Rat := 300

/-- Theorem stating that the maximum number of hot dogs that can be purchased is 3250 --/
theorem max_hot_dogs_is_3250 :
  maxHotDogs availablePacks totalBudget = 3250 := by sorry

end NUMINAMATH_CALUDE_max_hot_dogs_is_3250_l2798_279813


namespace NUMINAMATH_CALUDE_employee_hire_year_l2798_279897

/-- Rule of 70 retirement provision -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year an employee was hired -/
def hire_year : ℕ := 1970

/-- The age at which the employee was hired -/
def hire_age : ℕ := 32

/-- The year the employee becomes eligible to retire -/
def retirement_year : ℕ := 2008

theorem employee_hire_year :
  rule_of_70 (hire_age + (retirement_year - hire_year)) (retirement_year - hire_year) ∧
  ∀ y, y > hire_year →
    ¬rule_of_70 (hire_age + (retirement_year - y)) (retirement_year - y) :=
sorry

end NUMINAMATH_CALUDE_employee_hire_year_l2798_279897


namespace NUMINAMATH_CALUDE_c_decreases_as_r_increases_l2798_279852

theorem c_decreases_as_r_increases (e n r : ℝ) (h_e : e > 0) (h_n : n > 0) (h_r : r > 0) :
  ∀ (R₁ R₂ : ℝ), R₁ > 0 → R₂ > 0 → R₂ > R₁ →
  (e * n) / (R₁ + n * r) > (e * n) / (R₂ + n * r) := by
sorry

end NUMINAMATH_CALUDE_c_decreases_as_r_increases_l2798_279852


namespace NUMINAMATH_CALUDE_john_total_paint_l2798_279814

/-- The number of primary colors John has -/
def num_colors : ℕ := 3

/-- The amount of paint John has for each color (in liters) -/
def paint_per_color : ℕ := 5

/-- The total amount of paint John has (in liters) -/
def total_paint : ℕ := num_colors * paint_per_color

theorem john_total_paint : total_paint = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_total_paint_l2798_279814


namespace NUMINAMATH_CALUDE_system_solution_l2798_279876

theorem system_solution :
  ∃ (x y₁ y₂ : ℝ),
    (x / 5 + 3 = 4) ∧
    (x^2 - 4*x*y₁ + 3*y₁^2 = 36) ∧
    (x^2 - 4*x*y₂ + 3*y₂^2 = 36) ∧
    (x = 5) ∧
    (y₁ = 10/3 + Real.sqrt 133 / 3) ∧
    (y₂ = 10/3 - Real.sqrt 133 / 3) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l2798_279876


namespace NUMINAMATH_CALUDE_mushroom_picking_theorem_l2798_279812

/-- Calculates the total number of mushrooms picked over a three-day trip --/
def total_mushrooms (day1_revenue : ℕ) (day2_picked : ℕ) (price_per_mushroom : ℕ) : ℕ :=
  let day1_picked := day1_revenue / price_per_mushroom
  let day3_picked := 2 * day2_picked
  day1_picked + day2_picked + day3_picked

/-- The total number of mushrooms picked over three days is 65 --/
theorem mushroom_picking_theorem :
  total_mushrooms 58 12 2 = 65 := by
  sorry

#eval total_mushrooms 58 12 2

end NUMINAMATH_CALUDE_mushroom_picking_theorem_l2798_279812


namespace NUMINAMATH_CALUDE_arithmetic_progression_polynomial_j_eq_neg_40_l2798_279839

/-- A polynomial of degree 4 with four distinct real zeros in arithmetic progression -/
structure ArithmeticProgressionPolynomial where
  j : ℝ
  k : ℝ
  roots : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → roots i ≠ roots j
  arithmetic_progression : ∃ (b d : ℝ), ∀ i, roots i = b + i * d

/-- The coefficient j of an ArithmeticProgressionPolynomial is -40 -/
theorem arithmetic_progression_polynomial_j_eq_neg_40 (p : ArithmeticProgressionPolynomial) :
  p.j = -40 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_polynomial_j_eq_neg_40_l2798_279839


namespace NUMINAMATH_CALUDE_opposite_reciprocal_sum_l2798_279849

theorem opposite_reciprocal_sum (m n c d : ℝ) : 
  m = -n → c * d = 1 → m + n + 3 * c * d - 10 = -7 := by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_sum_l2798_279849


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2798_279831

theorem functional_equation_solution (f : ℝ → ℝ) (a : ℝ) 
  (h : ∀ x y : ℝ, f x * f y - a * f (x * y) = x + y) : 
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2798_279831


namespace NUMINAMATH_CALUDE_marbles_remainder_l2798_279810

theorem marbles_remainder (r p g : ℕ) 
  (hr : r % 7 = 5) 
  (hp : p % 7 = 4) 
  (hg : g % 7 = 2) : 
  (r + p + g) % 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_marbles_remainder_l2798_279810


namespace NUMINAMATH_CALUDE_min_value_of_squared_ratios_l2798_279808

theorem min_value_of_squared_ratios (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b)^2 + (b / c)^2 + (c / a)^2 ≥ 3 ∧
  ((a / b)^2 + (b / c)^2 + (c / a)^2 = 3 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_squared_ratios_l2798_279808


namespace NUMINAMATH_CALUDE_distance_when_parallel_max_distance_l2798_279854

/-- A parabola with vertex at the origin -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2}

/-- Two points on the parabola -/
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- Assumption that P and Q are on the parabola -/
axiom h_P_on_parabola : P ∈ Parabola
axiom h_Q_on_parabola : Q ∈ Parabola

/-- Assumption that OP is perpendicular to OQ -/
axiom h_perpendicular : (P.1 * Q.1 + P.2 * Q.2) = 0

/-- Distance from a point to a line -/
def distanceToLine (point : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ := sorry

/-- The line PQ -/
def LinePQ : Set (ℝ × ℝ) := sorry

/-- Statement: When PQ is parallel to x-axis, distance from O to PQ is 1 -/
theorem distance_when_parallel : 
  P.2 = Q.2 → distanceToLine O LinePQ = 1 := sorry

/-- Statement: The maximum distance from O to PQ is 1 -/
theorem max_distance : 
  ∀ P Q : ℝ × ℝ, P ∈ Parabola → Q ∈ Parabola → 
  (P.1 * Q.1 + P.2 * Q.2) = 0 → 
  distanceToLine O LinePQ ≤ 1 := sorry

end NUMINAMATH_CALUDE_distance_when_parallel_max_distance_l2798_279854


namespace NUMINAMATH_CALUDE_division_simplification_l2798_279800

theorem division_simplification (x y : ℝ) : 6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l2798_279800


namespace NUMINAMATH_CALUDE_cubic_diff_linear_diff_mod_six_l2798_279884

theorem cubic_diff_linear_diff_mod_six (x y : ℤ) : 
  (x^3 - y^3) % 6 = (x - y) % 6 := by sorry

end NUMINAMATH_CALUDE_cubic_diff_linear_diff_mod_six_l2798_279884


namespace NUMINAMATH_CALUDE_at_op_difference_l2798_279880

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 2 * x

-- State the theorem
theorem at_op_difference : (at_op 5 3) - (at_op 3 5) = -4 := by
  sorry

end NUMINAMATH_CALUDE_at_op_difference_l2798_279880


namespace NUMINAMATH_CALUDE_vector_magnitude_l2798_279889

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![x, 2]
def b : Fin 2 → ℝ := ![2, 1]
def c (x : ℝ) : Fin 2 → ℝ := ![3, x]

-- Define the parallel condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ i, v i = k * w i

-- State the theorem
theorem vector_magnitude (x : ℝ) :
  parallel (a x) b →
  ‖b + c x‖ = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2798_279889


namespace NUMINAMATH_CALUDE_monochromatic_triangle_in_K17_l2798_279896

/-- A coloring of the edges of a complete graph -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A triangle in a graph is a set of three distinct vertices -/
def Triangle (n : ℕ) := { t : Fin n × Fin n × Fin n // t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2 }

/-- A triangle is monochromatic if all its edges have the same color -/
def IsMonochromatic (n : ℕ) (c : Coloring n) (t : Triangle n) : Prop :=
  c t.val.1 t.val.2.1 = c t.val.1 t.val.2.2 ∧ 
  c t.val.1 t.val.2.1 = c t.val.2.1 t.val.2.2

/-- The main theorem: any 3-coloring of K_17 contains a monochromatic triangle -/
theorem monochromatic_triangle_in_K17 :
  ∀ (c : Coloring 17), ∃ (t : Triangle 17), IsMonochromatic 17 c t :=
sorry


end NUMINAMATH_CALUDE_monochromatic_triangle_in_K17_l2798_279896


namespace NUMINAMATH_CALUDE_abs_sum_range_l2798_279871

theorem abs_sum_range : 
  (∀ x : ℝ, |x + 2| + |x + 3| ≥ 1) ∧ 
  (∃ y : ℝ, ∀ ε > 0, ∃ x : ℝ, |x + 2| + |x + 3| < y + ε) ∧ 
  y = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_range_l2798_279871


namespace NUMINAMATH_CALUDE_max_a_2016_gt_44_l2798_279870

/-- Definition of the sequence a_{n,k} -/
def a (n k : ℕ) : ℝ :=
  sorry

/-- The maximum value of a_{n,k} for a given n -/
def m (n : ℕ) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem max_a_2016_gt_44 
  (h1 : ∀ k, 1 ≤ k ∧ k ≤ 2016 → 0 < a 0 k)
  (h2 : ∀ n k, n ≥ 0 ∧ 1 ≤ k ∧ k < 2016 → a (n+1) k = a n k + 1 / (2 * a n (k+1)))
  (h3 : ∀ n, n ≥ 0 → a (n+1) 2016 = a n 2016 + 1 / (2 * a n 1)) :
  m 2016 > 44 :=
sorry

end NUMINAMATH_CALUDE_max_a_2016_gt_44_l2798_279870


namespace NUMINAMATH_CALUDE_x_value_l2798_279842

theorem x_value (x y : ℚ) (h1 : x / y = 12 / 3) (h2 : y = 27) : x = 108 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2798_279842


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2798_279835

/-- Given a rhombus with diagonals of 10 inches and 24 inches, its perimeter is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2798_279835


namespace NUMINAMATH_CALUDE_unique_solution_proof_l2798_279898

/-- The value of q for which the quadratic equation qx^2 - 16x + 8 = 0 has exactly one solution -/
def unique_solution_q : ℝ := 8

/-- The quadratic equation qx^2 - 16x + 8 = 0 -/
def quadratic_equation (q x : ℝ) : ℝ := q * x^2 - 16 * x + 8

theorem unique_solution_proof :
  ∀ q : ℝ, q ≠ 0 →
  (∃! x : ℝ, quadratic_equation q x = 0) ↔ q = unique_solution_q :=
sorry

end NUMINAMATH_CALUDE_unique_solution_proof_l2798_279898


namespace NUMINAMATH_CALUDE_sqrt_four_fourth_powers_sum_l2798_279869

theorem sqrt_four_fourth_powers_sum (h : ℝ) : 
  h = Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) → h = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_fourth_powers_sum_l2798_279869


namespace NUMINAMATH_CALUDE_portrait_ratio_l2798_279853

/-- Prove that the ratio of students who had portraits taken before lunch
    to the total number of students is 1:3 -/
theorem portrait_ratio :
  ∀ (before_lunch after_lunch not_taken : ℕ),
  before_lunch + after_lunch + not_taken = 24 →
  after_lunch = 10 →
  not_taken = 6 →
  (before_lunch : ℚ) / 24 = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_portrait_ratio_l2798_279853


namespace NUMINAMATH_CALUDE_petyas_journey_fraction_l2798_279862

/-- The fraction of the journey Petya completed before remembering his pen -/
def journey_fraction (total_time walking_time early_arrival late_arrival : ℚ) : ℚ :=
  walking_time / total_time

theorem petyas_journey_fraction :
  let total_time : ℚ := 20
  let early_arrival : ℚ := 3
  let late_arrival : ℚ := 7
  ∃ (walking_time : ℚ),
    journey_fraction total_time walking_time early_arrival late_arrival = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_petyas_journey_fraction_l2798_279862


namespace NUMINAMATH_CALUDE_circle_radius_is_six_l2798_279843

theorem circle_radius_is_six (r : ℝ) : r > 0 → 3 * (2 * Real.pi * r) = Real.pi * r^2 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_six_l2798_279843


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2798_279864

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 2) :
  let set := [1 - 1 / n, 1 - 2 / n] ++ List.replicate (n - 2) 1
  List.sum set / n = 1 - 3 / n^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2798_279864


namespace NUMINAMATH_CALUDE_polygon_sides_l2798_279807

theorem polygon_sides (d : ℕ) (v : ℕ) : d = 77 ∧ v = 1 → ∃ n : ℕ, n * (n - 3) / 2 = d ∧ n + v = 15 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2798_279807


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l2798_279809

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  let angle := Real.pi / 3  -- 60 degrees in radians
  a = (2, 0) →
  ‖a + 2 • b‖ = 2 * Real.sqrt 3 →
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt ((a.1^2 + a.2^2) * (b.1^2 + b.2^2))) = Real.cos angle →
  ‖b‖ = 1 :=
by sorry

#check vector_magnitude_problem

end NUMINAMATH_CALUDE_vector_magnitude_problem_l2798_279809


namespace NUMINAMATH_CALUDE_books_per_bookshelf_l2798_279879

theorem books_per_bookshelf 
  (total_books : ℕ) 
  (num_bookshelves : ℕ) 
  (h1 : total_books = 38) 
  (h2 : num_bookshelves = 19) 
  (h3 : num_bookshelves > 0) :
  total_books / num_bookshelves = 2 := by
  sorry

end NUMINAMATH_CALUDE_books_per_bookshelf_l2798_279879


namespace NUMINAMATH_CALUDE_tom_flashlight_batteries_l2798_279830

/-- The number of batteries Tom used on his flashlights -/
def batteries_on_flashlights : ℕ := 28

/-- The number of batteries Tom used in his toys -/
def batteries_in_toys : ℕ := 15

/-- The number of batteries Tom used in his controllers -/
def batteries_in_controllers : ℕ := 2

/-- The difference between the number of batteries on flashlights and in toys -/
def battery_difference : ℕ := 13

theorem tom_flashlight_batteries :
  batteries_on_flashlights = batteries_in_toys + battery_difference := by
  sorry

end NUMINAMATH_CALUDE_tom_flashlight_batteries_l2798_279830


namespace NUMINAMATH_CALUDE_milk_mixture_water_content_l2798_279806

theorem milk_mixture_water_content 
  (initial_water_percentage : ℝ)
  (initial_milk_volume : ℝ)
  (pure_milk_volume : ℝ)
  (h1 : initial_water_percentage = 5)
  (h2 : initial_milk_volume = 10)
  (h3 : pure_milk_volume = 15) :
  let total_water := initial_water_percentage / 100 * initial_milk_volume
  let total_volume := initial_milk_volume + pure_milk_volume
  let final_water_percentage := total_water / total_volume * 100
  final_water_percentage = 2 := by
sorry

end NUMINAMATH_CALUDE_milk_mixture_water_content_l2798_279806


namespace NUMINAMATH_CALUDE_solve_marbles_problem_l2798_279811

def marbles_problem (initial : ℕ) (gifted : ℕ) (final : ℕ) : Prop :=
  ∃ (lost : ℕ), initial - lost - gifted = final

theorem solve_marbles_problem :
  marbles_problem 85 25 43 → (∃ (lost : ℕ), lost = 17) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_marbles_problem_l2798_279811


namespace NUMINAMATH_CALUDE_albert_took_five_candies_l2798_279867

/-- The number of candies Albert took away -/
def candies_taken (initial final : ℕ) : ℕ := initial - final

/-- Proof that Albert took 5 candies -/
theorem albert_took_five_candies :
  candies_taken 76 71 = 5 := by
  sorry

end NUMINAMATH_CALUDE_albert_took_five_candies_l2798_279867


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l2798_279881

/-- Definition of a circle with center (h, k) and radius r -/
def Circle (h k r : ℝ) := {(x, y) : ℝ × ℝ | (x - h)^2 + (y - k)^2 = r^2}

/-- The intersection line of two circles -/
def IntersectionLine (c1 c2 : ℝ × ℝ × ℝ) : ℝ × ℝ → Prop :=
  let (h1, k1, r1) := c1
  let (h2, k2, r2) := c2
  λ (x, y) => x + y = -59/34

theorem intersection_line_of_circles :
  let c1 : ℝ × ℝ × ℝ := (-12, -6, 15)
  let c2 : ℝ × ℝ × ℝ := (4, 11, 9)
  ∀ (p : ℝ × ℝ), p ∈ Circle c1.1 c1.2.1 c1.2.2 ∩ Circle c2.1 c2.2.1 c2.2.2 →
    IntersectionLine c1 c2 p :=
by
  sorry

#check intersection_line_of_circles

end NUMINAMATH_CALUDE_intersection_line_of_circles_l2798_279881


namespace NUMINAMATH_CALUDE_angle_value_l2798_279836

theorem angle_value (PQR PQS QRS : ℝ) (x : ℝ) : 
  PQR = 120 → PQS = 2*x → QRS = x → PQR = PQS + QRS → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_l2798_279836


namespace NUMINAMATH_CALUDE_root_sum_absolute_value_l2798_279848

theorem root_sum_absolute_value (m : ℤ) (a b c : ℤ) : 
  (∃ (m : ℤ), ∀ (x : ℤ), x^3 - 2023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  |a| + |b| + |c| = 102 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_absolute_value_l2798_279848


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l2798_279845

-- Define the set of real numbers where the expression is meaningful
def meaningfulSet : Set ℝ :=
  {x : ℝ | 3 - x ≥ 0 ∧ x + 1 ≠ 0}

-- Theorem statement
theorem meaningful_expression_range :
  meaningfulSet = {x : ℝ | x ≤ 3 ∧ x ≠ -1} := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l2798_279845


namespace NUMINAMATH_CALUDE_permutation_difference_l2798_279851

def permutation (n : ℕ) (r : ℕ) : ℕ :=
  (n - r + 1).factorial / (n - r).factorial

theorem permutation_difference : permutation 8 4 - 2 * permutation 8 2 = 1568 := by
  sorry

end NUMINAMATH_CALUDE_permutation_difference_l2798_279851


namespace NUMINAMATH_CALUDE_equation_equality_l2798_279822

theorem equation_equality (a b : ℝ) : 1 - a^2 + 2*a*b - b^2 = 1 - (a^2 - 2*a*b + b^2) := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l2798_279822


namespace NUMINAMATH_CALUDE_remaining_pennies_l2798_279873

def initial_pennies : ℕ := 989
def spent_pennies : ℕ := 728

theorem remaining_pennies :
  initial_pennies - spent_pennies = 261 :=
by sorry

end NUMINAMATH_CALUDE_remaining_pennies_l2798_279873


namespace NUMINAMATH_CALUDE_shaded_portion_is_four_ninths_l2798_279820

-- Define the square ABCD
def square_side : ℝ := 6

-- Define the shaded areas
def shaded_area_1 : ℝ := 2 * 2
def shaded_area_2 : ℝ := 4 * 4 - 2 * 2
def shaded_area_3 : ℝ := 6 * 6

-- Total square area
def total_area : ℝ := square_side * square_side

-- Total shaded area
def total_shaded_area : ℝ := shaded_area_1 + shaded_area_2

-- Theorem to prove
theorem shaded_portion_is_four_ninths :
  total_shaded_area / total_area = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_shaded_portion_is_four_ninths_l2798_279820


namespace NUMINAMATH_CALUDE_cosine_amplitude_l2798_279850

theorem cosine_amplitude (c d : ℝ) (hc : c < 0) (hd : d > 0) 
  (hmax : ∀ x, c * Real.cos (d * x) ≤ 3) 
  (hmin : ∀ x, -3 ≤ c * Real.cos (d * x)) 
  (hmax_achieved : ∃ x, c * Real.cos (d * x) = 3) 
  (hmin_achieved : ∃ x, c * Real.cos (d * x) = -3) : 
  c = -3 := by
sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l2798_279850


namespace NUMINAMATH_CALUDE_stream_speed_l2798_279832

/-- Given a boat that travels downstream and upstream, calculate the speed of the stream -/
theorem stream_speed (downstream_distance upstream_distance : ℝ) 
                     (downstream_time upstream_time : ℝ) 
                     (h1 : downstream_distance = 72)
                     (h2 : upstream_distance = 30)
                     (h3 : downstream_time = 3)
                     (h4 : upstream_time = 3) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧
    stream_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2798_279832


namespace NUMINAMATH_CALUDE_solution_range_solution_range_converse_l2798_279865

/-- The system of equations has two distinct solutions -/
def has_two_distinct_solutions (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
  y₁ = Real.sqrt (-x₁^2 - 2*x₁) ∧ x₁ + y₁ - m = 0 ∧
  y₂ = Real.sqrt (-x₂^2 - 2*x₂) ∧ x₂ + y₂ - m = 0

/-- The main theorem -/
theorem solution_range (m : ℝ) : 
  has_two_distinct_solutions m → m ∈ Set.Icc 0 (-1 + Real.sqrt 2) :=
by
  sorry

/-- The converse of the main theorem -/
theorem solution_range_converse (m : ℝ) : 
  m ∈ Set.Ioo 0 (-1 + Real.sqrt 2) → has_two_distinct_solutions m :=
by
  sorry

end NUMINAMATH_CALUDE_solution_range_solution_range_converse_l2798_279865


namespace NUMINAMATH_CALUDE_wright_brothers_first_flight_l2798_279887

/-- Represents the different groups of brothers mentioned in the problem -/
inductive Brothers
  | Bell
  | Hale
  | Wright
  | Leon

/-- Represents an aircraft -/
structure Aircraft where
  name : String

/-- Represents a flight achievement -/
structure FlightAchievement where
  date : String
  aircraft : Aircraft
  achievers : Brothers

/-- The first powered human flight -/
def first_powered_flight : FlightAchievement :=
  { date := "December 1903"
  , aircraft := { name := "Flyer 1" }
  , achievers := Brothers.Wright }

/-- Theorem stating that the Wright Brothers achieved the first powered human flight -/
theorem wright_brothers_first_flight :
  first_powered_flight.achievers = Brothers.Wright :=
by sorry

end NUMINAMATH_CALUDE_wright_brothers_first_flight_l2798_279887


namespace NUMINAMATH_CALUDE_abes_age_l2798_279816

theorem abes_age :
  ∀ (present_age : ℕ), 
    (present_age + (present_age - 7) = 37) → 
    present_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_abes_age_l2798_279816


namespace NUMINAMATH_CALUDE_cost_of_thousand_gum_in_dollars_l2798_279823

/-- The cost of a single piece of gum in cents -/
def cost_of_one_gum : ℕ := 1

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of gum pieces we're calculating the cost for -/
def num_gum_pieces : ℕ := 1000

/-- Theorem: The cost of 1000 pieces of gum in dollars is 10.00 -/
theorem cost_of_thousand_gum_in_dollars : 
  (num_gum_pieces * cost_of_one_gum : ℚ) / cents_per_dollar = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_thousand_gum_in_dollars_l2798_279823


namespace NUMINAMATH_CALUDE_meeting_distance_meeting_distance_is_correct_l2798_279894

/-- The distance between two people moving towards each other -/
theorem meeting_distance (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  2 * (a + b)
where
  /-- Person A's speed in km/h -/
  speed_a : ℝ := a
  /-- Person B's speed in km/h -/
  speed_b : ℝ := b
  /-- Time taken for them to meet in hours -/
  meeting_time : ℝ := 2
  /-- The two people start from different locations -/
  different_start_locations : Prop := True
  /-- The two people start at the same time -/
  same_start_time : Prop := True
  /-- The two people move towards each other -/
  move_towards_each_other : Prop := True

theorem meeting_distance_is_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  meeting_distance a b ha hb = 2 * (a + b) := by sorry

end NUMINAMATH_CALUDE_meeting_distance_meeting_distance_is_correct_l2798_279894


namespace NUMINAMATH_CALUDE_exam_problem_solution_l2798_279841

theorem exam_problem_solution (pA pB pC : ℝ) 
  (hA : pA = 1/3) 
  (hB : pB = 1/4) 
  (hC : pC = 1/5) 
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - pA) * (1 - pB) * (1 - pC) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_exam_problem_solution_l2798_279841


namespace NUMINAMATH_CALUDE_imaginary_part_sum_of_complex_fractions_l2798_279878

theorem imaginary_part_sum_of_complex_fractions :
  Complex.im (1 / Complex.ofReal (-2) + Complex.I + 1 / (Complex.ofReal 1 - 2 * Complex.I)) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_sum_of_complex_fractions_l2798_279878


namespace NUMINAMATH_CALUDE_stamp_collection_fraction_l2798_279833

/-- Given the stamp collection scenario, prove that KJ has half the stamps of AJ -/
theorem stamp_collection_fraction :
  ∀ (cj kj aj : ℕ) (f : ℚ),
  -- CJ has 5 more than twice the number of stamps that KJ has
  cj = 2 * kj + 5 →
  -- KJ has a certain fraction of the number of stamps AJ has
  kj = f * aj →
  -- The three boys have 930 stamps in total
  cj + kj + aj = 930 →
  -- AJ has 370 stamps
  aj = 370 →
  -- The fraction of stamps KJ has compared to AJ is 1/2
  f = 1/2 := by
sorry


end NUMINAMATH_CALUDE_stamp_collection_fraction_l2798_279833


namespace NUMINAMATH_CALUDE_addition_problem_solution_l2798_279818

/-- Represents a digit in the addition problem -/
structure Digit :=
  (value : Nat)
  (property : value < 10)

/-- Represents the addition problem -/
structure AdditionProblem :=
  (E : Digit)
  (I : Digit)
  (G : Digit)
  (H : Digit)
  (T : Digit)
  (F : Digit)
  (V : Digit)
  (R : Digit)
  (N : Digit)
  (all_different : ∀ d1 d2 : Digit, d1.value = d2.value → d1 = d2)
  (E_is_nine : E.value = 9)
  (G_is_odd : G.value % 2 = 1)
  (equation_holds : 
    10000 * E.value + 1000 * I.value + 100 * G.value + 10 * H.value + T.value +
    10000 * F.value + 1000 * I.value + 100 * V.value + 10 * E.value =
    10000000 * T.value + 1000000 * H.value + 100000 * I.value + 10000 * R.value +
    1000 * T.value + 100 * E.value + 10 * E.value + N.value)

theorem addition_problem_solution (problem : AdditionProblem) : problem.I.value = 4 := by
  sorry

end NUMINAMATH_CALUDE_addition_problem_solution_l2798_279818


namespace NUMINAMATH_CALUDE_min_value_M_l2798_279888

theorem min_value_M (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + b^2 + c^2 = 1) :
  (4 * Real.sqrt 3) / 3 ≤ max (a + 1/b) (max (b + 1/c) (c + 1/a)) ∧
  ∃ a b c, 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 + c^2 = 1 ∧
    (4 * Real.sqrt 3) / 3 = max (a + 1/b) (max (b + 1/c) (c + 1/a)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_M_l2798_279888


namespace NUMINAMATH_CALUDE_clock_hand_positions_l2798_279875

/-- Represents the number of minutes after 12:00 when the clock hands overlap -/
def overlap_time : ℚ := 720 / 11

/-- Represents the number of times the clock hands overlap in 12 hours -/
def overlap_count : ℕ := 11

/-- Represents the number of times the clock hands form right angles in 12 hours -/
def right_angle_count : ℕ := 22

/-- Represents the number of times the clock hands form straight angles in 12 hours -/
def straight_angle_count : ℕ := 11

/-- Proves that the clock hands overlap, form right angles, and straight angles
    the specified number of times in a 12-hour period -/
theorem clock_hand_positions :
  overlap_count = 11 ∧
  right_angle_count = 22 ∧
  straight_angle_count = 11 :=
by sorry

end NUMINAMATH_CALUDE_clock_hand_positions_l2798_279875


namespace NUMINAMATH_CALUDE_four_digit_sum_l2798_279863

theorem four_digit_sum (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  (6 * (a + b + c + d) * 1111 = 73326) →
  ({a, b, c, d} : Set ℕ) = {1, 2, 3, 5} :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_l2798_279863


namespace NUMINAMATH_CALUDE_arithmetic_sequence_bounds_l2798_279857

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A six-term arithmetic sequence containing 4 and 20 (in that order) -/
structure ArithSeqWithFourAndTwenty where
  a : ℕ → ℝ
  is_arithmetic : IsArithmeticSequence a
  has_four_and_twenty : ∃ i j : ℕ, i < j ∧ j < 6 ∧ a i = 4 ∧ a j = 20

/-- The theorem stating the largest and smallest possible values of z-r -/
theorem arithmetic_sequence_bounds (seq : ArithSeqWithFourAndTwenty) :
  (∃ zr : ℝ, zr = seq.a 5 - seq.a 0 ∧ zr ≤ 80 ∧ 
  ∀ zr' : ℝ, zr' = seq.a 5 - seq.a 0 → zr' ≤ zr) ∧
  (∃ zr : ℝ, zr = seq.a 5 - seq.a 0 ∧ zr ≥ 16 ∧ 
  ∀ zr' : ℝ, zr' = seq.a 5 - seq.a 0 → zr' ≥ zr) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_bounds_l2798_279857


namespace NUMINAMATH_CALUDE_paulines_garden_rows_l2798_279856

/-- Represents Pauline's garden --/
structure Garden where
  tomatoes : ℕ
  cucumbers : ℕ
  potatoes : ℕ
  extra_capacity : ℕ
  spaces_per_row : ℕ

/-- Calculates the number of rows in the garden --/
def number_of_rows (g : Garden) : ℕ :=
  (g.tomatoes + g.cucumbers + g.potatoes + g.extra_capacity) / g.spaces_per_row

/-- Theorem: The number of rows in Pauline's garden is 10 --/
theorem paulines_garden_rows :
  let g : Garden := {
    tomatoes := 3 * 5,
    cucumbers := 5 * 4,
    potatoes := 30,
    extra_capacity := 85,
    spaces_per_row := 15
  }
  number_of_rows g = 10 := by
  sorry

end NUMINAMATH_CALUDE_paulines_garden_rows_l2798_279856


namespace NUMINAMATH_CALUDE_prob_not_adjacent_ten_chairs_l2798_279817

-- Define the number of chairs
def n : ℕ := 10

-- Define the probability function
def prob_not_adjacent (n : ℕ) : ℚ :=
  1 - (n - 1 : ℚ) / (n.choose 2 : ℚ)

-- Theorem statement
theorem prob_not_adjacent_ten_chairs :
  prob_not_adjacent n = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_adjacent_ten_chairs_l2798_279817


namespace NUMINAMATH_CALUDE_sugar_water_concentration_increases_l2798_279872

theorem sugar_water_concentration_increases 
  (a b m : ℝ) 
  (h1 : b > a) 
  (h2 : a > 0) 
  (h3 : m > 0) : 
  a / b < (a + m) / (b + m) := by
sorry

end NUMINAMATH_CALUDE_sugar_water_concentration_increases_l2798_279872


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l2798_279861

theorem unique_triplet_solution : 
  ∃! (x y z : ℕ+), 
    (x^y.val + y.val^x.val = z.val^y.val) ∧ 
    (x^y.val + 2012 = y.val^(z.val + 1)) ∧
    x = 6 ∧ y = 2 ∧ z = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l2798_279861


namespace NUMINAMATH_CALUDE_daniels_purchase_worth_l2798_279827

/-- The total worth of Daniel's purchases -/
def total_worth (taxable_purchase : ℝ) (tax_free_items : ℝ) : ℝ :=
  taxable_purchase + tax_free_items

/-- The amount of sales tax paid on taxable purchases -/
def sales_tax (taxable_purchase : ℝ) (tax_rate : ℝ) : ℝ :=
  taxable_purchase * tax_rate

theorem daniels_purchase_worth :
  ∃ (taxable_purchase : ℝ),
    sales_tax taxable_purchase 0.05 = 0.30 ∧
    total_worth taxable_purchase 18.7 = 24.7 := by
  sorry

end NUMINAMATH_CALUDE_daniels_purchase_worth_l2798_279827


namespace NUMINAMATH_CALUDE_broken_line_circle_cover_l2798_279846

/-- A closed broken line on a plane -/
structure ClosedBrokenLine :=
  (points : Set (ℝ × ℝ))
  (is_closed : sorry)
  (length : ℝ)

/-- Theorem: Any closed broken line of length 1 on a plane can be covered by a circle of radius 1/4 -/
theorem broken_line_circle_cover (L : ClosedBrokenLine) (h : L.length = 1) :
  ∃ (center : ℝ × ℝ), ∀ p ∈ L.points, dist p center ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_broken_line_circle_cover_l2798_279846


namespace NUMINAMATH_CALUDE_train_distance_problem_l2798_279840

/-- The distance between two points A and B, given train speeds and time difference --/
theorem train_distance_problem (v_ab v_ba : ℝ) (time_diff : ℝ) : 
  v_ab = 160 → v_ba = 120 → time_diff = 1 → 
  ∃ D : ℝ, D / v_ba = D / v_ab + time_diff ∧ D = 480 := by
  sorry

#check train_distance_problem

end NUMINAMATH_CALUDE_train_distance_problem_l2798_279840


namespace NUMINAMATH_CALUDE_factorial_division_l2798_279883

theorem factorial_division : Nat.factorial 5 / Nat.factorial (5 - 3) = 60 := by sorry

end NUMINAMATH_CALUDE_factorial_division_l2798_279883


namespace NUMINAMATH_CALUDE_subset_union_equality_l2798_279803

theorem subset_union_equality (n : ℕ+) (A : Fin (n + 1) → Set (Fin n)) 
  (h : ∀ i, (A i).Nonempty) :
  ∃ (I J : Set (Fin (n + 1))), I.Nonempty ∧ J.Nonempty ∧ I ∩ J = ∅ ∧
    (⋃ i ∈ I, A i) = (⋃ j ∈ J, A j) := by
  sorry

end NUMINAMATH_CALUDE_subset_union_equality_l2798_279803


namespace NUMINAMATH_CALUDE_letters_in_mailboxes_l2798_279834

theorem letters_in_mailboxes :
  (number_of_ways : ℕ) →
  (number_of_letters : ℕ) →
  (number_of_mailboxes : ℕ) →
  (number_of_letters = 4) →
  (number_of_mailboxes = 3) →
  (number_of_ways = number_of_mailboxes ^ number_of_letters) :=
by sorry

end NUMINAMATH_CALUDE_letters_in_mailboxes_l2798_279834


namespace NUMINAMATH_CALUDE_g_sum_property_l2798_279859

-- Define the function g
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + 5

-- State the theorem
theorem g_sum_property (a b c : ℝ) : g a b c 10 = 3 → g a b c 10 + g a b c (-10) = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_property_l2798_279859


namespace NUMINAMATH_CALUDE_smallest_pencil_count_l2798_279895

theorem smallest_pencil_count (p : ℕ) : 
  (p > 0) →
  (p % 6 = 5) → 
  (p % 7 = 3) → 
  (p % 8 = 7) → 
  (∀ q : ℕ, q > 0 → q % 6 = 5 → q % 7 = 3 → q % 8 = 7 → p ≤ q) →
  p = 35 := by
sorry

end NUMINAMATH_CALUDE_smallest_pencil_count_l2798_279895


namespace NUMINAMATH_CALUDE_ladder_problem_l2798_279826

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l2798_279826


namespace NUMINAMATH_CALUDE_jump_rope_median_and_mode_l2798_279828

def jump_rope_scores : List ℕ := [129, 130, 130, 130, 132, 132, 135, 135, 137, 137]

def median (scores : List ℕ) : ℚ := sorry

def mode (scores : List ℕ) : ℕ := sorry

theorem jump_rope_median_and_mode :
  median jump_rope_scores = 132 ∧ mode jump_rope_scores = 130 := by sorry

end NUMINAMATH_CALUDE_jump_rope_median_and_mode_l2798_279828


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2798_279805

theorem no_integer_solutions : ¬ ∃ (m n : ℤ), m + 2*n = 2*m*n - 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2798_279805


namespace NUMINAMATH_CALUDE_convex_pentagon_integer_point_l2798_279855

-- Define a point in 2D space
structure Point where
  x : ℤ
  y : ℤ

-- Define a pentagon as a list of 5 points
def Pentagon := List Point

-- Define a predicate to check if a pentagon is convex
def isConvex (p : Pentagon) : Prop := sorry

-- Define a predicate to check if a point is inside or on the boundary of a pentagon
def isInsideOrOnBoundary (point : Point) (p : Pentagon) : Prop := sorry

-- The main theorem
theorem convex_pentagon_integer_point 
  (p : Pentagon) 
  (h1 : p.length = 5) 
  (h2 : isConvex p) : 
  ∃ (point : Point), isInsideOrOnBoundary point p := by
  sorry

end NUMINAMATH_CALUDE_convex_pentagon_integer_point_l2798_279855
