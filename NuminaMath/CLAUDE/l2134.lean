import Mathlib

namespace NUMINAMATH_CALUDE_line_does_not_intersect_curve_l2134_213455

/-- The function representing the curve y = (|x|-1)/(|x-1|) -/
noncomputable def f (x : ℝ) : ℝ := (abs x - 1) / (abs (x - 1))

/-- The theorem stating the condition for non-intersection -/
theorem line_does_not_intersect_curve (m : ℝ) :
  (∀ x : ℝ, m * x ≠ f x) ↔ (-1 ≤ m ∧ m < -3 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_line_does_not_intersect_curve_l2134_213455


namespace NUMINAMATH_CALUDE_specific_right_triangle_l2134_213420

/-- A right triangle with specific side lengths -/
structure RightTriangle where
  -- The length of the hypotenuse
  ab : ℝ
  -- The length of one of the other sides
  ac : ℝ
  -- The length of the remaining side
  bc : ℝ
  -- Constraint that this is a right triangle (Pythagorean theorem)
  pythagorean : ab ^ 2 = ac ^ 2 + bc ^ 2

/-- Theorem: In a right triangle with hypotenuse 5 and one side 4, the other side is 3 -/
theorem specific_right_triangle :
  ∃ (t : RightTriangle), t.ab = 5 ∧ t.ac = 4 ∧ t.bc = 3 := by
  sorry


end NUMINAMATH_CALUDE_specific_right_triangle_l2134_213420


namespace NUMINAMATH_CALUDE_canadian_scientist_ratio_l2134_213448

/-- Proves that the ratio of Canadian scientists to total scientists is 1:5 -/
theorem canadian_scientist_ratio (total : ℕ) (usa : ℕ) : 
  total = 70 → 
  usa = 21 → 
  (total - (total / 2) - usa) / total = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_canadian_scientist_ratio_l2134_213448


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l2134_213494

theorem quadratic_root_implies_m (x m : ℝ) : 
  x = -1 → x^2 + m*x = 3 → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l2134_213494


namespace NUMINAMATH_CALUDE_problem_1_l2134_213446

theorem problem_1 : (1.5 - 0.6) * (3 - 1.8) = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2134_213446


namespace NUMINAMATH_CALUDE_neighborhood_cable_cost_l2134_213457

/-- Calculates the total cost of cable for a neighborhood given the street layout and cable requirements. -/
theorem neighborhood_cable_cost
  (east_west_streets : ℕ)
  (east_west_length : ℝ)
  (north_south_streets : ℕ)
  (north_south_length : ℝ)
  (cable_per_street_mile : ℝ)
  (cable_cost_per_mile : ℝ)
  (h1 : east_west_streets = 18)
  (h2 : east_west_length = 2)
  (h3 : north_south_streets = 10)
  (h4 : north_south_length = 4)
  (h5 : cable_per_street_mile = 5)
  (h6 : cable_cost_per_mile = 2000) :
  (east_west_streets * east_west_length + north_south_streets * north_south_length) *
  cable_per_street_mile * cable_cost_per_mile = 760000 := by
  sorry


end NUMINAMATH_CALUDE_neighborhood_cable_cost_l2134_213457


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2134_213431

def unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  unit_vector a → unit_vector b → (a.1 * b.1 + a.2 * b.2 = -1/2) →
  (a.1 - 3*b.1)^2 + (a.2 - 3*b.2)^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2134_213431


namespace NUMINAMATH_CALUDE_equation_solution_l2134_213439

theorem equation_solution (x : ℤ) : 9*x + 2 ≡ 7 [ZMOD 15] ↔ x ≡ 10 [ZMOD 15] := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2134_213439


namespace NUMINAMATH_CALUDE_angle_slope_relationship_l2134_213454

theorem angle_slope_relationship (α k : ℝ) :
  (k = Real.tan α) →
  (α < π / 3 → k < Real.sqrt 3) ∧
  ¬(k < Real.sqrt 3 → α < π / 3) :=
sorry

end NUMINAMATH_CALUDE_angle_slope_relationship_l2134_213454


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2134_213487

/-- Two lines are parallel if they do not intersect -/
def Parallel (l1 l2 : Set Point) : Prop := l1 ∩ l2 = ∅

/-- A point lies on a line if it is a member of the line's point set -/
def PointOnLine (p : Point) (l : Set Point) : Prop := p ∈ l

theorem parallel_line_through_point 
  (l l₁ : Set Point) (M : Point) 
  (h_parallel : Parallel l l₁)
  (h_M_not_on_l : ¬ PointOnLine M l)
  (h_M_not_on_l₁ : ¬ PointOnLine M l₁) :
  ∃ l₂ : Set Point, Parallel l₂ l ∧ Parallel l₂ l₁ ∧ PointOnLine M l₂ :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2134_213487


namespace NUMINAMATH_CALUDE_interest_rate_proof_l2134_213498

/-- Proves that the interest rate is 5% given the specified loan conditions -/
theorem interest_rate_proof (principal : ℝ) (time : ℝ) (interest : ℝ) :
  principal = 3000 →
  time = 5 →
  interest = principal - 2250 →
  (interest * 100) / (principal * time) = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l2134_213498


namespace NUMINAMATH_CALUDE_exists_bijection_Z_to_H_l2134_213412

-- Define the set ℍ
def ℍ : Set ℚ :=
  { x | ∀ S : Set ℚ, 
    (1/2 ∈ S) → 
    (∀ y ∈ S, 1/(1+y) ∈ S ∧ y/(1+y) ∈ S) → 
    x ∈ S }

-- State the theorem
theorem exists_bijection_Z_to_H : ∃ f : ℤ → ℍ, Function.Bijective f := by
  sorry

end NUMINAMATH_CALUDE_exists_bijection_Z_to_H_l2134_213412


namespace NUMINAMATH_CALUDE_erased_number_problem_l2134_213496

theorem erased_number_problem (n : Nat) (x : Nat) : 
  n = 69 → 
  x ≤ n →
  x ≥ 1 →
  (((n * (n + 1)) / 2 - x) : ℚ) / (n - 1 : ℚ) = 35 + (7 : ℚ) / 17 →
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_erased_number_problem_l2134_213496


namespace NUMINAMATH_CALUDE_average_weight_increase_l2134_213410

theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total_weight := 7 * initial_average
  let new_total_weight := initial_total_weight - 75 + 99.5
  let new_average := new_total_weight / 7
  new_average - initial_average = 3.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2134_213410


namespace NUMINAMATH_CALUDE_randy_initial_money_l2134_213486

/-- Calculates the initial amount of money Randy had in his piggy bank. -/
def initial_money (cost_per_trip : ℕ) (trips_per_month : ℕ) (months : ℕ) (money_left : ℕ) : ℕ :=
  cost_per_trip * trips_per_month * months + money_left

/-- Proves that Randy started with $200 given the problem conditions. -/
theorem randy_initial_money :
  initial_money 2 4 12 104 = 200 := by
  sorry

end NUMINAMATH_CALUDE_randy_initial_money_l2134_213486


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2134_213479

/-- If ax^2 + 28x + 9 is the square of a binomial, then a = 196/9 -/
theorem quadratic_is_square_of_binomial (a : ℚ) : 
  (∃ p q : ℚ, ∀ x : ℚ, a * x^2 + 28 * x + 9 = (p * x + q)^2) → 
  a = 196 / 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2134_213479


namespace NUMINAMATH_CALUDE_find_m_l2134_213480

def U : Set Nat := {1, 2, 3, 4}

def A (m : ℤ) : Set Nat := {x ∈ U | x^2 - 5*x + m = 0}

theorem find_m : ∃ m : ℤ, (U \ A m) = {1, 4} ∧ m = 6 := by sorry

end NUMINAMATH_CALUDE_find_m_l2134_213480


namespace NUMINAMATH_CALUDE_set_intersection_equality_l2134_213438

def M : Set ℤ := {1, 2, 3}
def N : Set ℤ := {x : ℤ | 1 < x ∧ x < 4}

theorem set_intersection_equality : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l2134_213438


namespace NUMINAMATH_CALUDE_tangency_points_x_coordinates_l2134_213404

/-- Given a curve y = x^m and a point A(1,0), prove the x-coordinates of the first two tangency points -/
theorem tangency_points_x_coordinates (m : ℕ) (hm : m > 1) :
  let curve (x : ℝ) := x^m
  let tangent_line (a : ℝ) (x : ℝ) := m * a^(m-1) * (x - a) + a^m
  let a₁ := (tangent_line ⁻¹) 0 1  -- x-coordinate where tangent line passes through (1,0)
  let a₂ := (tangent_line ⁻¹) 0 a₁ -- x-coordinate where tangent line passes through (a₁,0)
  a₁ = m / (m - 1) ∧ a₂ = (m / (m - 1))^2 := by
sorry


end NUMINAMATH_CALUDE_tangency_points_x_coordinates_l2134_213404


namespace NUMINAMATH_CALUDE_exists_z_satisfying_equation_l2134_213445

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (3 * x)^3 + 3 * x + 5

-- State the theorem
theorem exists_z_satisfying_equation :
  ∃ z : ℝ, f (3 * z) = 3 ∧ z = -2 / 729 := by
  sorry

end NUMINAMATH_CALUDE_exists_z_satisfying_equation_l2134_213445


namespace NUMINAMATH_CALUDE_train_speed_l2134_213450

/-- Given a train that travels 80 km in 40 minutes, prove its speed is 120 kmph -/
theorem train_speed (distance : ℝ) (time_minutes : ℝ) (speed : ℝ) : 
  distance = 80 ∧ time_minutes = 40 → speed = distance / (time_minutes / 60) → speed = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2134_213450


namespace NUMINAMATH_CALUDE_odometer_sum_of_squares_l2134_213413

/-- Represents the odometer reading as a three-digit number -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds + tens + ones ≤ 9

/-- Represents the car's journey -/
structure CarJourney where
  duration : Nat
  avg_speed : Nat
  start_reading : OdometerReading
  end_reading : OdometerReading
  journey_valid : 
    duration = 8 ∧ 
    avg_speed = 65 ∧
    end_reading.hundreds = start_reading.ones ∧
    end_reading.tens = start_reading.tens ∧
    end_reading.ones = start_reading.hundreds

theorem odometer_sum_of_squares (journey : CarJourney) : 
  journey.start_reading.hundreds^2 + 
  journey.start_reading.tens^2 + 
  journey.start_reading.ones^2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_odometer_sum_of_squares_l2134_213413


namespace NUMINAMATH_CALUDE_product_of_invertible_labels_l2134_213444

def is_invertible (f : ℕ → Bool) := f 2 = false ∧ f 3 = true ∧ f 4 = true ∧ f 5 = true

theorem product_of_invertible_labels (f : ℕ → Bool) (h : is_invertible f) :
  (List.filter (λ i => f i) [2, 3, 4, 5]).prod = 60 :=
by sorry

end NUMINAMATH_CALUDE_product_of_invertible_labels_l2134_213444


namespace NUMINAMATH_CALUDE_select_team_count_l2134_213463

/-- The number of players in the basketball team -/
def total_players : ℕ := 12

/-- The number of players to be selected for the team -/
def team_size : ℕ := 5

/-- The number of twins in the team -/
def num_twins : ℕ := 2

/-- Calculates the binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of ways to select the team with the given conditions -/
def select_team : ℕ := binomial total_players team_size - binomial (total_players - num_twins) (team_size - num_twins)

theorem select_team_count : select_team = 672 := by
  sorry

end NUMINAMATH_CALUDE_select_team_count_l2134_213463


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2134_213467

theorem simplify_trig_expression (θ : Real) (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = 2 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2134_213467


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l2134_213414

theorem dogwood_tree_count (current_trees planted_trees : ℕ) : 
  current_trees = 34 → planted_trees = 49 → current_trees + planted_trees = 83 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l2134_213414


namespace NUMINAMATH_CALUDE_condition_relationship_l2134_213465

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a > 2 ∧ b > 2 → a + b > 4) ∧
  (∃ a b, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l2134_213465


namespace NUMINAMATH_CALUDE_sum_of_three_times_m_and_half_n_square_diff_minus_square_sum_l2134_213435

-- Part 1
theorem sum_of_three_times_m_and_half_n (m n : ℝ) :
  3 * m + (1/2) * n = 3 * m + (1/2) * n := by sorry

-- Part 2
theorem square_diff_minus_square_sum (a b : ℝ) :
  (a - b)^2 - (a + b)^2 = (a - b)^2 - (a + b)^2 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_times_m_and_half_n_square_diff_minus_square_sum_l2134_213435


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2134_213461

/-- The number of ways to arrange books on a shelf --/
def arrange_books (math_books : ℕ) (english_books : ℕ) : ℕ :=
  Nat.factorial 3 * Nat.factorial math_books * Nat.factorial english_books

/-- Theorem stating the number of ways to arrange 4 math books, 7 English books, and 1 journal --/
theorem book_arrangement_count :
  arrange_books 4 7 = 725760 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2134_213461


namespace NUMINAMATH_CALUDE_max_t_value_l2134_213436

theorem max_t_value (f : ℝ → ℝ) (a : ℝ) (t : ℝ) : 
  (∀ x : ℝ, f x = (x + 1)^2) →
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ t → f (x + a) ≤ 2*x - 4) →
  t ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_t_value_l2134_213436


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l2134_213405

theorem quadratic_two_real_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 4*x + 2 = 0 ∧ a * y^2 - 4*y + 2 = 0) ↔ 
  (a ≤ 2 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l2134_213405


namespace NUMINAMATH_CALUDE_chinese_english_time_difference_l2134_213499

/-- The number of hours Ryan spends daily learning English -/
def english_hours : ℕ := 6

/-- The number of hours Ryan spends daily learning Chinese -/
def chinese_hours : ℕ := 7

/-- Theorem: The difference between the time spent on learning Chinese and English is 1 hour -/
theorem chinese_english_time_difference :
  chinese_hours - english_hours = 1 := by
  sorry

end NUMINAMATH_CALUDE_chinese_english_time_difference_l2134_213499


namespace NUMINAMATH_CALUDE_det_A_eq_one_l2134_213493

/-- The matrix A_n as defined in the problem -/
def A (n : ℕ+) : Matrix (Fin n) (Fin n) ℚ :=
  λ i j => (i.val + j.val - 2).choose (j.val - 1)

/-- The theorem stating that the determinant of A_n is 1 for all positive integers n -/
theorem det_A_eq_one (n : ℕ+) : Matrix.det (A n) = 1 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_one_l2134_213493


namespace NUMINAMATH_CALUDE_nth_equation_proof_l2134_213434

theorem nth_equation_proof (n : ℕ) : 2 * n * (2 * n + 2) + 1 = (2 * n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_proof_l2134_213434


namespace NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l2134_213482

-- Define the equations
def equation1 (x : ℝ) : Prop := (3 / (x^2 - 9)) + (x / (x - 3)) = 1
def equation2 (x : ℝ) : Prop := 2 - (1 / (2 - x)) = (3 - x) / (x - 2)

-- Theorem for equation 1
theorem equation1_solution : 
  ∃! x : ℝ, equation1 x ∧ x ≠ 3 ∧ x ≠ -3 := by sorry

-- Theorem for equation 2
theorem equation2_no_solution : 
  ∀ x : ℝ, ¬(equation2 x ∧ x ≠ 2) := by sorry

end NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l2134_213482


namespace NUMINAMATH_CALUDE_seating_problem_l2134_213472

/-- The number of ways to seat people on a bench with given constraints -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) (min_gap : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of seating arrangements for the given problem -/
theorem seating_problem : seating_arrangements 9 3 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_seating_problem_l2134_213472


namespace NUMINAMATH_CALUDE_bernoulli_zero_success_l2134_213409

/-- The number of trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The probability of failure in each trial -/
def q : ℚ := 1 - p

/-- The number of successes we're interested in -/
def k : ℕ := 0

/-- Theorem: The probability of 0 successes in 7 Bernoulli trials 
    with success probability 2/7 is (5/7)^7 -/
theorem bernoulli_zero_success : 
  (n.choose k) * p^k * q^(n-k) = (5/7)^7 := by sorry

end NUMINAMATH_CALUDE_bernoulli_zero_success_l2134_213409


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2134_213451

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- State the theorem
theorem line_perpendicular_to_plane 
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_parallel : parallel m n)
  (h_perpendicular : perpendicular m β) :
  perpendicular n β := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2134_213451


namespace NUMINAMATH_CALUDE_cube_multiplication_division_equality_l2134_213443

theorem cube_multiplication_division_equality : (12 ^ 3 * 6 ^ 3) / 432 = 864 := by
  sorry

end NUMINAMATH_CALUDE_cube_multiplication_division_equality_l2134_213443


namespace NUMINAMATH_CALUDE_percentage_calculation_l2134_213475

theorem percentage_calculation (x : ℝ) (h : x ≠ 0) : (x + 0.5 * x) / (0.75 * x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2134_213475


namespace NUMINAMATH_CALUDE_vitamin_d_pack_size_l2134_213423

/-- The number of Vitamin A supplements in each pack -/
def vitamin_a_pack_size : ℕ := 7

/-- The smallest number of each type of vitamin sold -/
def smallest_quantity_sold : ℕ := 119

/-- Theorem stating that the number of Vitamin D supplements in each pack is 17 -/
theorem vitamin_d_pack_size :
  ∃ (n m x : ℕ),
    n * vitamin_a_pack_size = m * x ∧
    n * vitamin_a_pack_size = smallest_quantity_sold ∧
    x > 1 ∧
    x < vitamin_a_pack_size ∧
    x = 17 := by
  sorry

end NUMINAMATH_CALUDE_vitamin_d_pack_size_l2134_213423


namespace NUMINAMATH_CALUDE_harvest_duration_l2134_213426

def harvest_problem (weekly_earning : ℕ) (total_earning : ℕ) : Prop :=
  weekly_earning * 89 = total_earning

theorem harvest_duration : harvest_problem 2 178 := by
  sorry

end NUMINAMATH_CALUDE_harvest_duration_l2134_213426


namespace NUMINAMATH_CALUDE_line_equation_l2134_213460

/-- A line passing through (2,3) with opposite-sign intercepts -/
structure LineWithOppositeIntercepts where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (2,3)
  passes_through : 3 = m * 2 + b
  -- The line has opposite-sign intercepts
  opposite_intercepts : (b ≠ 0 ∧ (-b/m) * b < 0) ∨ (b = 0 ∧ m ≠ 0)

/-- The equation of the line is either 3x - 2y = 0 or x - y + 1 = 0 -/
theorem line_equation (l : LineWithOppositeIntercepts) :
  (l.m = 3/2 ∧ l.b = 0) ∨ (l.m = 1 ∧ l.b = -1) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l2134_213460


namespace NUMINAMATH_CALUDE_hyperbola_C_equation_l2134_213462

/-- A hyperbola passing through a point and sharing asymptotes with another hyperbola -/
def hyperbola_C (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), (a > 0 ∧ b > 0) ∧ 
  (x^2 / a^2 - y^2 / b^2 = 1) ∧
  (3^2 / a^2 - 2 / b^2 = 1) ∧
  (a^2 / b^2 = 3)

/-- Theorem stating the standard equation of hyperbola C -/
theorem hyperbola_C_equation :
  ∀ x y : ℝ, hyperbola_C x y → (x^2 / 3 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_C_equation_l2134_213462


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l2134_213419

theorem square_rectangle_area_relation :
  ∀ x : ℝ,
  let square_side : ℝ := x - 3
  let rect_length : ℝ := x - 4
  let rect_width : ℝ := x + 5
  let square_area : ℝ := square_side ^ 2
  let rect_area : ℝ := rect_length * rect_width
  (rect_area = 3 * square_area) →
  (∃ y : ℝ, y ≠ x ∧ 
    let square_side' : ℝ := y - 3
    let rect_length' : ℝ := y - 4
    let rect_width' : ℝ := y + 5
    let square_area' : ℝ := square_side' ^ 2
    let rect_area' : ℝ := rect_length' * rect_width'
    (rect_area' = 3 * square_area')) →
  x + y = 7 :=
by sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l2134_213419


namespace NUMINAMATH_CALUDE_base_2_representation_of_123_l2134_213456

theorem base_2_representation_of_123 : 
  ∃ (a b c d e f g : ℕ), 
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1) ∧
    123 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_123_l2134_213456


namespace NUMINAMATH_CALUDE_dog_treats_duration_l2134_213478

theorem dog_treats_duration (treats_per_day : ℕ) (cost_per_treat : ℚ) (total_spent : ℚ) : 
  treats_per_day = 2 → cost_per_treat = 1/10 → total_spent = 6 → 
  (total_spent / cost_per_treat) / treats_per_day = 30 := by
  sorry

end NUMINAMATH_CALUDE_dog_treats_duration_l2134_213478


namespace NUMINAMATH_CALUDE_intersection_nonempty_l2134_213449

def M (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 - 1 = k * (p.1 + 1)}

def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.2 = 0}

theorem intersection_nonempty (k : ℝ) : ∃ p : ℝ × ℝ, p ∈ M k ∩ N := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_l2134_213449


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2134_213418

theorem binomial_expansion_coefficient (p : ℝ) : 
  (∃ k : ℕ, Nat.choose 5 k * p^k = 80 ∧ 2*k = 6) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2134_213418


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l2134_213452

/-- Calculates the sampling interval for systematic sampling -/
def samplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The sampling interval for a population of 800 and sample size of 40 is 20 -/
theorem systematic_sampling_interval :
  samplingInterval 800 40 = 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l2134_213452


namespace NUMINAMATH_CALUDE_negation_of_nonnegative_squares_l2134_213473

theorem negation_of_nonnegative_squares :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_nonnegative_squares_l2134_213473


namespace NUMINAMATH_CALUDE_negative_324_same_terminal_side_as_36_l2134_213441

/-- Two angles have the same terminal side if their difference is a multiple of 360 degrees -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β - α = k * 360

/-- The main theorem: -324° has the same terminal side as 36° -/
theorem negative_324_same_terminal_side_as_36 :
  same_terminal_side 36 (-324) := by
  sorry

end NUMINAMATH_CALUDE_negative_324_same_terminal_side_as_36_l2134_213441


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2134_213490

theorem quadratic_inequality (x : ℝ) : x ^ 2 - 4 * x - 21 ≤ 0 ↔ x ∈ Set.Icc (-3) 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2134_213490


namespace NUMINAMATH_CALUDE_product_of_sums_l2134_213411

theorem product_of_sums : 
  (8 - Real.sqrt 500 + 8 + Real.sqrt 500) * (12 - Real.sqrt 72 + 12 + Real.sqrt 72) = 384 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_l2134_213411


namespace NUMINAMATH_CALUDE_vector_proof_l2134_213425

theorem vector_proof (a b : ℝ × ℝ) : 
  b = (1, -2) → 
  (a.1 * b.1 + a.2 * b.2 = -Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) → 
  Real.sqrt (a.1^2 + a.2^2) = 3 * Real.sqrt 5 → 
  a = (-3, 6) := by sorry

end NUMINAMATH_CALUDE_vector_proof_l2134_213425


namespace NUMINAMATH_CALUDE_distance_between_blue_lights_l2134_213484

/-- Represents the sequence of lights -/
inductive Light
| Blue
| Yellow

/-- The pattern of lights -/
def light_pattern : List Light := [Light.Blue, Light.Blue, Light.Yellow, Light.Yellow, Light.Yellow]

/-- The distance between each light in inches -/
def light_distance : ℕ := 8

/-- Calculates the position of the nth blue light -/
def blue_light_position (n : ℕ) : ℕ :=
  sorry

/-- Calculates the distance between two positions in feet -/
def distance_in_feet (pos1 pos2 : ℕ) : ℚ :=
  sorry

theorem distance_between_blue_lights :
  distance_in_feet (blue_light_position 4) (blue_light_position 26) = 100/3 :=
sorry

end NUMINAMATH_CALUDE_distance_between_blue_lights_l2134_213484


namespace NUMINAMATH_CALUDE_percentage_fraction_difference_l2134_213474

theorem percentage_fraction_difference : 
  (65 / 100 * 40) - (4 / 5 * 25) = 6 := by sorry

end NUMINAMATH_CALUDE_percentage_fraction_difference_l2134_213474


namespace NUMINAMATH_CALUDE_class_funds_calculation_l2134_213407

/-- Proves that the class funds amount to $14 given the problem conditions -/
theorem class_funds_calculation (total_contribution student_count student_contribution : ℕ) 
  (h1 : total_contribution = 90)
  (h2 : student_count = 19)
  (h3 : student_contribution = 4) :
  total_contribution - (student_count * student_contribution) = 14 := by
  sorry

#check class_funds_calculation

end NUMINAMATH_CALUDE_class_funds_calculation_l2134_213407


namespace NUMINAMATH_CALUDE_congruence_system_solution_l2134_213432

theorem congruence_system_solution :
  ∃ x : ℤ, (x ≡ 1 [ZMOD 6] ∧ x ≡ 9 [ZMOD 14] ∧ x ≡ 7 [ZMOD 15]) ↔ x ≡ 37 [ZMOD 210] :=
by sorry

end NUMINAMATH_CALUDE_congruence_system_solution_l2134_213432


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2134_213428

/-- An even function on ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem sufficient_not_necessary
  (f : ℝ → ℝ) (hf : EvenFunction f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ - f x₂ = 0) ∧
  (∃ x₁ x₂ : ℝ, f x₁ - f x₂ = 0 ∧ x₁ + x₂ ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2134_213428


namespace NUMINAMATH_CALUDE_tv_show_episodes_l2134_213417

/-- Given a TV show with the following properties:
  * It ran for 10 seasons
  * The first half of seasons had 20 episodes per season
  * There were 225 total episodes
  This theorem proves that the number of episodes per season in the second half was 25. -/
theorem tv_show_episodes (total_seasons : ℕ) (first_half_episodes : ℕ) (total_episodes : ℕ) :
  total_seasons = 10 →
  first_half_episodes = 20 →
  total_episodes = 225 →
  (total_episodes - (total_seasons / 2 * first_half_episodes)) / (total_seasons / 2) = 25 :=
by sorry

end NUMINAMATH_CALUDE_tv_show_episodes_l2134_213417


namespace NUMINAMATH_CALUDE_simplify_expression_l2134_213406

theorem simplify_expression (z : ℝ) : z - 3 + 4*z + 5 - 6*z + 7 - 8*z + 9 = -9*z + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2134_213406


namespace NUMINAMATH_CALUDE_price_increase_l2134_213416

theorem price_increase (P : ℝ) (x : ℝ) (h1 : P > 0) :
  1.25 * P * (1 + x / 100) = 1.625 * P → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_l2134_213416


namespace NUMINAMATH_CALUDE_remaining_pool_area_l2134_213424

/-- The area of the remaining pool space given a circular pool with diameter 13 meters
    and a rectangular obstacle with dimensions 2.5 meters by 4 meters. -/
theorem remaining_pool_area :
  let pool_diameter : ℝ := 13
  let obstacle_length : ℝ := 2.5
  let obstacle_width : ℝ := 4
  let pool_area := π * (pool_diameter / 2) ^ 2
  let obstacle_area := obstacle_length * obstacle_width
  pool_area - obstacle_area = 132.7325 * π - 10 := by sorry

end NUMINAMATH_CALUDE_remaining_pool_area_l2134_213424


namespace NUMINAMATH_CALUDE_rotation_volumes_equal_l2134_213492

/-- The volume obtained by rotating a region about the y-axis -/
noncomputable def rotationVolume (region : Set (ℝ × ℝ)) : ℝ := sorry

/-- The region enclosed by x^2 = 4y, x^2 = -4y, x = 4, and x = -4 -/
def region1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 = 4*p.2 ∨ p.1^2 = -4*p.2) ∧ (p.1 = 4 ∨ p.1 = -4)}

/-- The region defined by x^2 + y^2 ≤ 16, x^2 + (y-2)^2 ≥ 4, and x^2 + (y+2)^2 ≥ 4 -/
def region2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 16 ∧ p.1^2 + (p.2-2)^2 ≥ 4 ∧ p.1^2 + (p.2+2)^2 ≥ 4}

/-- The theorem stating that the volumes of rotation are equal -/
theorem rotation_volumes_equal : rotationVolume region1 = rotationVolume region2 := by
  sorry

end NUMINAMATH_CALUDE_rotation_volumes_equal_l2134_213492


namespace NUMINAMATH_CALUDE_quadratic_point_relationship_l2134_213469

/-- A quadratic function of the form y = -(x-1)² + k -/
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := -(x - 1)^2 + k

theorem quadratic_point_relationship (k : ℝ) (y₁ y₂ y₃ : ℝ) :
  quadratic_function k (-1) = y₁ →
  quadratic_function k 2 = y₂ →
  quadratic_function k 4 = y₃ →
  y₃ < y₁ ∧ y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_relationship_l2134_213469


namespace NUMINAMATH_CALUDE_product_remainder_l2134_213489

theorem product_remainder (a b c : ℕ) (ha : a = 2456) (hb : b = 8743) (hc : c = 92431) :
  (a * b * c) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l2134_213489


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l2134_213485

theorem subset_implies_m_equals_one (m : ℝ) :
  let A : Set ℝ := {-1, 2, 2*m - 1}
  let B : Set ℝ := {2, m^2}
  B ⊆ A → m = 1 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l2134_213485


namespace NUMINAMATH_CALUDE_caramel_distribution_solution_l2134_213491

def caramel_distribution (a b c d : ℕ) : Prop :=
  a + b + c + d = 26 ∧
  ∃ (x y : ℕ),
    a = x + y ∧
    b = 2 * x ∧
    c = x + y ∧
    d = x + (2 * y + x) ∧
    x > 0 ∧ y > 0

theorem caramel_distribution_solution :
  caramel_distribution 5 6 5 10 :=
sorry

end NUMINAMATH_CALUDE_caramel_distribution_solution_l2134_213491


namespace NUMINAMATH_CALUDE_trapezoid_bases_solutions_l2134_213402

theorem trapezoid_bases_solutions :
  let valid_pair : ℕ × ℕ → Prop := fun (b₁, b₂) =>
    b₁ + b₂ = 60 ∧ 
    b₁ % 9 = 0 ∧ 
    b₂ % 9 = 0 ∧ 
    b₁ > 0 ∧ 
    b₂ > 0 ∧ 
    (60 : ℝ) * (b₁ + b₂) / 2 = 1800
  ∃! (solutions : List (ℕ × ℕ)),
    solutions.length = 3 ∧ 
    ∀ pair, pair ∈ solutions ↔ valid_pair pair :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_bases_solutions_l2134_213402


namespace NUMINAMATH_CALUDE_complex_point_in_fourth_quadrant_l2134_213488

theorem complex_point_in_fourth_quadrant (a b : ℝ) :
  (a^2 - 4*a + 5 > 0) ∧ (-b^2 + 2*b - 6 < 0) :=
by
  sorry

#check complex_point_in_fourth_quadrant

end NUMINAMATH_CALUDE_complex_point_in_fourth_quadrant_l2134_213488


namespace NUMINAMATH_CALUDE_john_notebooks_l2134_213459

/-- Calculates the maximum number of notebooks that can be purchased with a given amount of money, considering a bulk discount. -/
def max_notebooks (total_cents : ℕ) (notebook_price : ℕ) (discount : ℕ) (bulk_size : ℕ) : ℕ :=
  let discounted_price := notebook_price - discount
  let bulk_set_price := discounted_price * bulk_size
  let bulk_sets := total_cents / bulk_set_price
  let remaining_cents := total_cents % bulk_set_price
  let additional_notebooks := remaining_cents / notebook_price
  bulk_sets * bulk_size + additional_notebooks

/-- Proves that given 2545 cents, with notebooks costing 235 cents each and a 15 cent discount
    per notebook when bought in sets of 5, the maximum number of notebooks that can be purchased is 11. -/
theorem john_notebooks : max_notebooks 2545 235 15 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_john_notebooks_l2134_213459


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2134_213495

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2134_213495


namespace NUMINAMATH_CALUDE_sum_of_digits_for_four_elevenths_l2134_213470

theorem sum_of_digits_for_four_elevenths : ∃ (x y : ℕ), 
  (x < 10 ∧ y < 10) ∧ 
  (4 : ℚ) / 11 = (x * 10 + y : ℚ) / 99 ∧
  x + y = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_for_four_elevenths_l2134_213470


namespace NUMINAMATH_CALUDE_candy_jar_problem_l2134_213476

/-- Represents the number of candies in a jar -/
structure JarContents where
  red : ℕ
  yellow : ℕ

/-- The problem statement -/
theorem candy_jar_problem :
  ∀ (jar1 jar2 : JarContents),
    -- Both jars have the same total number of candies
    (jar1.red + jar1.yellow = jar2.red + jar2.yellow) →
    -- Jar 1 has a red to yellow ratio of 7:3
    (7 * jar1.yellow = 3 * jar1.red) →
    -- Jar 2 has a red to yellow ratio of 5:4
    (5 * jar2.yellow = 4 * jar2.red) →
    -- The total number of yellow candies is 108
    (jar1.yellow + jar2.yellow = 108) →
    -- The difference in red candies between Jar 1 and Jar 2 is 21
    (jar1.red - jar2.red = 21) :=
by sorry

end NUMINAMATH_CALUDE_candy_jar_problem_l2134_213476


namespace NUMINAMATH_CALUDE_stock_price_increase_l2134_213483

theorem stock_price_increase (opening_price closing_price : ℝ) 
  (percent_increase : ℝ) : 
  opening_price = 6 → 
  percent_increase = 33.33 → 
  closing_price = opening_price * (1 + percent_increase / 100) → 
  closing_price = 8 := by
sorry

end NUMINAMATH_CALUDE_stock_price_increase_l2134_213483


namespace NUMINAMATH_CALUDE_factorization_equality_l2134_213415

theorem factorization_equality (m n : ℝ) : m^2*n + 2*m*n^2 + n^3 = n*(m+n)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2134_213415


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2134_213442

theorem absolute_value_equality (x : ℝ) (y : ℝ) :
  y > 0 →
  |3 * x - 2 * Real.log y| = 3 * x + 2 * Real.log y →
  x = 0 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2134_213442


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l2134_213437

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -2 then x + 2
  else if x < 3 then 2^x
  else Real.log x

theorem f_composition_negative_two : f (f (-2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l2134_213437


namespace NUMINAMATH_CALUDE_matching_probability_is_one_third_l2134_213422

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans a person has -/
def JellyBeans.total (jb : JellyBeans) : ℕ := jb.blue + jb.green + jb.yellow

/-- Abe's jelly beans -/
def abe : JellyBeans := { blue := 2, green := 2, yellow := 0 }

/-- Bob's jelly beans -/
def bob : JellyBeans := { blue := 3, green := 1, yellow := 2 }

/-- The probability of two people showing matching color jelly beans -/
def matchingProbability (person1 person2 : JellyBeans) : ℚ :=
  let totalProb : ℚ := 
    (person1.blue * person2.blue + person1.green * person2.green) / 
    (person1.total * person2.total)
  totalProb

theorem matching_probability_is_one_third : 
  matchingProbability abe bob = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_matching_probability_is_one_third_l2134_213422


namespace NUMINAMATH_CALUDE_intersection_M_N_l2134_213430

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_M_N : M ∩ N = {5, 7, 9} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2134_213430


namespace NUMINAMATH_CALUDE_some_athletes_not_honor_society_l2134_213453

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Athlete : U → Prop)
variable (Disciplined : U → Prop)
variable (HonorSocietyMember : U → Prop)

-- Define the given conditions
variable (some_athletes_not_disciplined : ∃ x, Athlete x ∧ ¬Disciplined x)
variable (all_honor_society_disciplined : ∀ x, HonorSocietyMember x → Disciplined x)

-- State the theorem
theorem some_athletes_not_honor_society :
  ∃ x, Athlete x ∧ ¬HonorSocietyMember x :=
sorry

end NUMINAMATH_CALUDE_some_athletes_not_honor_society_l2134_213453


namespace NUMINAMATH_CALUDE_proposition_and_variants_l2134_213464

theorem proposition_and_variants (x y : ℝ) :
  -- Original proposition
  (x^2 + y^2 = 0 → x * y = 0) ∧
  -- Converse (false)
  ¬(x * y = 0 → x^2 + y^2 = 0) ∧
  -- Inverse (false)
  ¬(x^2 + y^2 ≠ 0 → x * y ≠ 0) ∧
  -- Contrapositive (true)
  (x * y ≠ 0 → x^2 + y^2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_proposition_and_variants_l2134_213464


namespace NUMINAMATH_CALUDE_intersection_equal_B_l2134_213433

def A : Set ℝ := {x | x^2 - 4*x - 21 = 0}

def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem intersection_equal_B (m : ℝ) : 
  (A ∩ B m) = B m ↔ m = 0 ∨ m = -1/7 ∨ m = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equal_B_l2134_213433


namespace NUMINAMATH_CALUDE_inequalities_propositions_l2134_213497

theorem inequalities_propositions :
  (∀ a b : ℝ, a * b > 0 → a > b → 1 / a < 1 / b) ∧
  (∀ a b : ℝ, a > abs b → a^2 > b^2) ∧
  (∃ a b c d : ℝ, a > b ∧ a > d ∧ a - c ≤ b - d) ∧
  (∃ a b m : ℝ, a < b ∧ m > 0 ∧ a / b ≥ (a + m) / (b + m)) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_propositions_l2134_213497


namespace NUMINAMATH_CALUDE_candy_per_box_l2134_213400

/-- Given that Billy bought 7 boxes of candy and had a total of 21 pieces,
    prove that each box contained 3 pieces of candy. -/
theorem candy_per_box (num_boxes : ℕ) (total_pieces : ℕ) (h1 : num_boxes = 7) (h2 : total_pieces = 21) :
  total_pieces / num_boxes = 3 := by
sorry

end NUMINAMATH_CALUDE_candy_per_box_l2134_213400


namespace NUMINAMATH_CALUDE_end_of_week_stock_l2134_213401

def pencils_per_day : ℕ := 100
def working_days_per_week : ℕ := 5
def initial_stock : ℕ := 80
def pencils_sold : ℕ := 350

theorem end_of_week_stock : 
  pencils_per_day * working_days_per_week + initial_stock - pencils_sold = 230 := by
  sorry

end NUMINAMATH_CALUDE_end_of_week_stock_l2134_213401


namespace NUMINAMATH_CALUDE_bowl_capacity_l2134_213477

/-- Given a bowl filled with oil and vinegar, prove its capacity. -/
theorem bowl_capacity (oil_density vinegar_density : ℝ)
                      (oil_fraction vinegar_fraction : ℝ)
                      (total_weight : ℝ) :
  oil_density = 5 →
  vinegar_density = 4 →
  oil_fraction = 2/3 →
  vinegar_fraction = 1/3 →
  total_weight = 700 →
  oil_fraction * oil_density + vinegar_fraction * vinegar_density = total_weight / 150 :=
by sorry

end NUMINAMATH_CALUDE_bowl_capacity_l2134_213477


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_280_450_l2134_213429

theorem lcm_gcf_ratio_280_450 : Nat.lcm 280 450 / Nat.gcd 280 450 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_280_450_l2134_213429


namespace NUMINAMATH_CALUDE_xy_greater_than_xz_l2134_213481

theorem xy_greater_than_xz (x y z : ℝ) 
  (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) : x * y > x * z := by
  sorry

end NUMINAMATH_CALUDE_xy_greater_than_xz_l2134_213481


namespace NUMINAMATH_CALUDE_correct_mark_is_63_l2134_213440

/-- Proves that the correct mark is 63 given the conditions of the problem -/
theorem correct_mark_is_63 (n : ℕ) (wrong_mark : ℕ) (avg_increase : ℚ) : 
  n = 40 → 
  wrong_mark = 83 → 
  avg_increase = 1/2 → 
  (wrong_mark - (n * avg_increase : ℚ).floor : ℤ) = 63 := by
  sorry

end NUMINAMATH_CALUDE_correct_mark_is_63_l2134_213440


namespace NUMINAMATH_CALUDE_power_sum_divisibility_l2134_213427

theorem power_sum_divisibility (k : ℕ) :
  7 ∣ (2^k + 3^k) ↔ k % 6 = 3 := by sorry

end NUMINAMATH_CALUDE_power_sum_divisibility_l2134_213427


namespace NUMINAMATH_CALUDE_smallest_angle_theorem_l2134_213408

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.sin (3 * x) * Real.sin (4 * x) = Real.cos (3 * x) * Real.cos (4 * x)

-- Define the theorem
theorem smallest_angle_theorem :
  ∃ (x : ℝ), x > 0 ∧ x < π ∧ equation x ∧
  (∀ (y : ℝ), y > 0 ∧ y < x → ¬equation y) ∧
  x = 90 * (π / 180) / 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_theorem_l2134_213408


namespace NUMINAMATH_CALUDE_contrapositive_quadratic_roots_l2134_213458

theorem contrapositive_quadratic_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, a * x^2 - b * x + c = 0 → x > 0) → a * c > 0
  ↔
  a * c ≤ 0 → ∃ x : ℝ, a * x^2 - b * x + c = 0 ∧ x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_quadratic_roots_l2134_213458


namespace NUMINAMATH_CALUDE_ellipse_equation_l2134_213466

theorem ellipse_equation (e : ℝ) (h_e : e = (2/5) * Real.sqrt 5) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    e = Real.sqrt (a^2 - b^2) / a ∧
    1^2 / b^2 + 0^2 / a^2 = 1 ∧
    (∀ x y : ℝ, x^2 / b^2 + y^2 / a^2 = 1 ↔ x^2 + (1/5) * y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2134_213466


namespace NUMINAMATH_CALUDE_vowels_on_board_l2134_213403

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 2

/-- The total number of vowels written on the board -/
def total_vowels : ℕ := num_vowels * times_written

theorem vowels_on_board : total_vowels = 10 := by
  sorry

end NUMINAMATH_CALUDE_vowels_on_board_l2134_213403


namespace NUMINAMATH_CALUDE_john_bought_three_tshirts_l2134_213421

/-- The number of t-shirts John bought -/
def num_tshirts : ℕ := 3

/-- The cost of each t-shirt in dollars -/
def tshirt_cost : ℕ := 20

/-- The amount spent on pants in dollars -/
def pants_cost : ℕ := 50

/-- The total amount spent in dollars -/
def total_spent : ℕ := 110

theorem john_bought_three_tshirts :
  num_tshirts * tshirt_cost + pants_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_john_bought_three_tshirts_l2134_213421


namespace NUMINAMATH_CALUDE_potato_division_l2134_213471

theorem potato_division (total_potatoes : ℕ) (num_people : ℕ) (potatoes_per_person : ℕ) :
  total_potatoes = 24 →
  num_people = 3 →
  total_potatoes = num_people * potatoes_per_person →
  potatoes_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_potato_division_l2134_213471


namespace NUMINAMATH_CALUDE_vehicle_speeds_and_distance_l2134_213468

theorem vehicle_speeds_and_distance (total_distance : ℝ) 
  (speed_ratio : ℝ) (time_delay : ℝ) :
  total_distance = 90 →
  speed_ratio = 1.5 →
  time_delay = 1/3 →
  ∃ (speed_slow speed_fast distance_traveled : ℝ),
    speed_slow = 90 ∧
    speed_fast = 135 ∧
    distance_traveled = 30 ∧
    speed_fast = speed_ratio * speed_slow ∧
    total_distance / speed_slow - total_distance / speed_fast = time_delay ∧
    distance_traveled = speed_slow * time_delay :=
by sorry

end NUMINAMATH_CALUDE_vehicle_speeds_and_distance_l2134_213468


namespace NUMINAMATH_CALUDE_intersection_A_B_l2134_213447

-- Define set A
def A : Set ℝ := {x | x^2 - 1 ≥ 0}

-- Define set B
def B : Set ℝ := {x | 1 ≤ x ∧ x < 3}

-- Theorem statement
theorem intersection_A_B : 
  ∀ x : ℝ, x ∈ A ∩ B ↔ 1 ≤ x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2134_213447
