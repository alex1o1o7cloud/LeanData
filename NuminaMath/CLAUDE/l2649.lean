import Mathlib

namespace NUMINAMATH_CALUDE_opposite_is_negation_l2649_264990

-- Define the concept of opposite number
def opposite (a : ℝ) : ℝ := -a

-- Theorem stating that the opposite of a is -a
theorem opposite_is_negation (a : ℝ) : opposite a = -a := by
  sorry

end NUMINAMATH_CALUDE_opposite_is_negation_l2649_264990


namespace NUMINAMATH_CALUDE_bus_ride_cost_l2649_264926

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 1.50

/-- The cost of a train ride from town P to town Q -/
def train_cost : ℝ := bus_cost + 6.85

/-- The theorem stating the cost of a bus ride from town P to town Q -/
theorem bus_ride_cost : bus_cost = 1.50 := by
  have h1 : train_cost = bus_cost + 6.85 := by rfl
  have h2 : bus_cost + train_cost = 9.85 := by sorry
  sorry

end NUMINAMATH_CALUDE_bus_ride_cost_l2649_264926


namespace NUMINAMATH_CALUDE_probability_multiple_of_100_is_zero_l2649_264960

def is_single_digit_multiple_of_5 (n : ℕ) : Prop :=
  n > 0 ∧ n < 10 ∧ n % 5 = 0

def is_prime_less_than_50 (p : ℕ) : Prop :=
  Nat.Prime p ∧ p < 50

def is_multiple_of_100 (n : ℕ) : Prop :=
  n % 100 = 0

theorem probability_multiple_of_100_is_zero :
  ∀ (n p : ℕ), is_single_digit_multiple_of_5 n → is_prime_less_than_50 p →
  ¬(is_multiple_of_100 (n * p)) :=
sorry

end NUMINAMATH_CALUDE_probability_multiple_of_100_is_zero_l2649_264960


namespace NUMINAMATH_CALUDE_calculation_proof_l2649_264924

theorem calculation_proof : 
  Real.sqrt 12 - abs (-1) + (1/2)⁻¹ + (2023 + Real.pi)^0 = 2 * Real.sqrt 3 + 2 := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l2649_264924


namespace NUMINAMATH_CALUDE_frog_reach_edge_prob_l2649_264935

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Defines the 4x4 grid with wraparound edges -/
def Grid := Set Position

/-- Checks if a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Represents a single hop direction -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Applies a hop in the given direction, considering wraparound -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨p.x, (p.y + 1) % 4⟩
  | Direction.Down => ⟨p.x, (p.y - 1 + 4) % 4⟩
  | Direction.Left => ⟨(p.x - 1 + 4) % 4, p.y⟩
  | Direction.Right => ⟨(p.x + 1) % 4, p.y⟩

/-- Defines the probability of reaching an edge within n hops -/
def probReachEdge (start : Position) (n : Nat) : ℝ :=
  sorry

theorem frog_reach_edge_prob :
  probReachEdge ⟨3, 3⟩ 5 = 1 := by sorry

end NUMINAMATH_CALUDE_frog_reach_edge_prob_l2649_264935


namespace NUMINAMATH_CALUDE_smallest_batch_size_l2649_264920

theorem smallest_batch_size (N : ℕ) (h1 : N > 70) (h2 : (21 * N) % 70 = 0) :
  N ≥ 80 ∧ ∀ m : ℕ, m > 70 ∧ (21 * m) % 70 = 0 → m ≥ N := by
  sorry

end NUMINAMATH_CALUDE_smallest_batch_size_l2649_264920


namespace NUMINAMATH_CALUDE_work_of_two_springs_in_series_l2649_264972

/-- The work required to stretch a system of two springs in series -/
theorem work_of_two_springs_in_series 
  (k₁ k₂ : Real) 
  (x : Real) 
  (h₁ : k₁ = 3000) -- 3 kN/m = 3000 N/m
  (h₂ : k₂ = 6000) -- 6 kN/m = 6000 N/m
  (h₃ : x = 0.05)  -- 5 cm = 0.05 m
  : (1/2) * (1 / (1/k₁ + 1/k₂)) * x^2 = 2.5 := by
  sorry

#check work_of_two_springs_in_series

end NUMINAMATH_CALUDE_work_of_two_springs_in_series_l2649_264972


namespace NUMINAMATH_CALUDE_jerry_action_figures_l2649_264944

/-- Given an initial count of action figures, a number removed, and a final count,
    this function calculates how many action figures were added. -/
def actionFiguresAdded (initial final removed : ℕ) : ℕ :=
  final + removed - initial

/-- Theorem stating that given the specific conditions in the problem,
    the number of action figures added must be 11. -/
theorem jerry_action_figures :
  actionFiguresAdded 7 8 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_jerry_action_figures_l2649_264944


namespace NUMINAMATH_CALUDE_tylers_age_l2649_264969

theorem tylers_age (T B S : ℕ) : 
  T = B - 3 → 
  T + B + S = 25 → 
  S = B + 1 → 
  T = 6 := by
sorry

end NUMINAMATH_CALUDE_tylers_age_l2649_264969


namespace NUMINAMATH_CALUDE_min_value_inequality_l2649_264908

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1/a + 1/b + 2 * Real.sqrt (a * b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2649_264908


namespace NUMINAMATH_CALUDE_smallest_number_proof_l2649_264952

def smallest_number : ℕ := 3153

theorem smallest_number_proof :
  (∀ n : ℕ, n < smallest_number →
    ¬(((n + 3) % 18 = 0) ∧ ((n + 3) % 25 = 0) ∧ ((n + 3) % 21 = 0))) ∧
  ((smallest_number + 3) % 18 = 0) ∧
  ((smallest_number + 3) % 25 = 0) ∧
  ((smallest_number + 3) % 21 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l2649_264952


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l2649_264904

theorem root_exists_in_interval : ∃ x : ℝ, 3/2 < x ∧ x < 2 ∧ 2^x = x^2 + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l2649_264904


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2649_264980

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 2*x - 3 < 0} = {x : ℝ | -3 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2649_264980


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_six_l2649_264976

theorem at_least_one_not_less_than_six (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ¬(a + 4 / b < 6 ∧ b + 9 / c < 6 ∧ c + 16 / a < 6) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_six_l2649_264976


namespace NUMINAMATH_CALUDE_mean_proportional_of_segments_l2649_264997

theorem mean_proportional_of_segments (a b : ℝ) (ha : a = 2) (hb : b = 6) :
  ∃ c : ℝ, c ^ 2 = a * b ∧ c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_of_segments_l2649_264997


namespace NUMINAMATH_CALUDE_table_capacity_l2649_264987

theorem table_capacity (invited : ℕ) (no_show : ℕ) (tables : ℕ) : 
  invited = 45 → no_show = 35 → tables = 5 → (invited - no_show) / tables = 2 := by
  sorry

end NUMINAMATH_CALUDE_table_capacity_l2649_264987


namespace NUMINAMATH_CALUDE_car_travel_distance_l2649_264941

theorem car_travel_distance (speed1 speed2 total_distance average_speed : ℝ) 
  (h1 : speed1 = 75)
  (h2 : speed2 = 80)
  (h3 : total_distance = 320)
  (h4 : average_speed = 77.4193548387097)
  (h5 : total_distance = 2 * (total_distance / 2)) : 
  total_distance / 2 = 160 := by
  sorry

#check car_travel_distance

end NUMINAMATH_CALUDE_car_travel_distance_l2649_264941


namespace NUMINAMATH_CALUDE_count_sets_satisfying_union_l2649_264910

theorem count_sets_satisfying_union (A B : Set ℕ) : 
  A = {1, 2} → 
  (A ∪ B = {1, 2, 3, 4, 5}) → 
  (∃! (count : ℕ), ∃ (S : Finset (Set ℕ)), 
    (Finset.card S = count) ∧ 
    (∀ C ∈ S, A ∪ C = {1, 2, 3, 4, 5}) ∧
    (∀ D, A ∪ D = {1, 2, 3, 4, 5} → D ∈ S) ∧
    count = 4) :=
by sorry

end NUMINAMATH_CALUDE_count_sets_satisfying_union_l2649_264910


namespace NUMINAMATH_CALUDE_some_number_value_l2649_264919

theorem some_number_value (x : ℝ) : (85 + 32 / x) * x = 9637 → x = 113 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2649_264919


namespace NUMINAMATH_CALUDE_mean_median_difference_l2649_264912

/-- Represents the frequency histogram data for student absences -/
structure AbsenceData where
  zero_days : Nat
  one_day : Nat
  two_days : Nat
  three_days : Nat
  four_days : Nat
  total_students : Nat
  sum_condition : zero_days + one_day + two_days + three_days + four_days = total_students

/-- Calculates the mean number of days absent -/
def calculate_mean (data : AbsenceData) : Rat :=
  (0 * data.zero_days + 1 * data.one_day + 2 * data.two_days + 3 * data.three_days + 4 * data.four_days) / data.total_students

/-- Calculates the median number of days absent -/
def calculate_median (data : AbsenceData) : Nat :=
  if data.zero_days + data.one_day < data.total_students / 2 then 2 else 1

/-- Theorem stating the difference between mean and median -/
theorem mean_median_difference (data : AbsenceData) 
  (h : data.total_students = 20 ∧ 
       data.zero_days = 4 ∧ 
       data.one_day = 2 ∧ 
       data.two_days = 5 ∧ 
       data.three_days = 6 ∧ 
       data.four_days = 3) : 
  calculate_mean data - calculate_median data = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l2649_264912


namespace NUMINAMATH_CALUDE_problem_solution_l2649_264989

theorem problem_solution (x y : ℝ) (h : 2 * x = Real.log (x + y - 1) + Real.log (x - y - 1) + 4) :
  2015 * x^2 + 2016 * y^3 = 8060 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2649_264989


namespace NUMINAMATH_CALUDE_statement_B_is_algorithm_l2649_264999

-- Define what constitutes an algorithm
def is_algorithm (statement : String) : Prop :=
  ∃ (steps : List String), steps.length > 0 ∧ steps.all (λ step => step ≠ "")

-- Define the given statements
def statement_A : String := "At home, it is generally the mother who cooks."
def statement_B : String := "Cooking rice involves the steps of washing the pot, rinsing the rice, adding water, and heating."
def statement_C : String := "Cooking outdoors is called a picnic."
def statement_D : String := "Rice is necessary for cooking."

-- Theorem to prove
theorem statement_B_is_algorithm :
  is_algorithm statement_B ∧
  ¬is_algorithm statement_A ∧
  ¬is_algorithm statement_C ∧
  ¬is_algorithm statement_D :=
sorry

end NUMINAMATH_CALUDE_statement_B_is_algorithm_l2649_264999


namespace NUMINAMATH_CALUDE_total_pieces_eq_59_l2649_264909

/-- The number of pieces of clothing in the first load -/
def first_load : ℕ := 32

/-- The number of equal loads for the remaining clothing -/
def num_equal_loads : ℕ := 9

/-- The number of pieces of clothing in each of the equal loads -/
def pieces_per_equal_load : ℕ := 3

/-- The total number of pieces of clothing Will had to wash -/
def total_pieces : ℕ := first_load + num_equal_loads * pieces_per_equal_load

theorem total_pieces_eq_59 : total_pieces = 59 := by sorry

end NUMINAMATH_CALUDE_total_pieces_eq_59_l2649_264909


namespace NUMINAMATH_CALUDE_max_value_on_triangle_vertices_l2649_264981

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a triangle in 2D space
structure Triangle where
  P : Point2D
  Q : Point2D
  R : Point2D

-- Define a linear function f(x, y) = ax + by + c
def linearFunction (a b c : ℝ) (p : Point2D) : ℝ :=
  a * p.x + b * p.y + c

-- Define a predicate to check if a point is in or on a triangle
def isInOrOnTriangle (t : Triangle) (p : Point2D) : Prop :=
  sorry -- The actual implementation is not needed for the theorem statement

-- Theorem statement
theorem max_value_on_triangle_vertices 
  (t : Triangle) (a b c : ℝ) (p : Point2D) 
  (h : isInOrOnTriangle t p) : 
  linearFunction a b c p ≤ max 
    (linearFunction a b c t.P) 
    (max (linearFunction a b c t.Q) (linearFunction a b c t.R)) := by
  sorry


end NUMINAMATH_CALUDE_max_value_on_triangle_vertices_l2649_264981


namespace NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_focus_l2649_264966

/-- Given a hyperbola with equation x²/4 - y²/5 = 1, prove that the standard equation
    of a parabola with its focus at the left focus of the hyperbola is y² = -12x. -/
theorem parabola_equation_from_hyperbola_focus (x y : ℝ) :
  (x^2 / 4 - y^2 / 5 = 1) →
  ∃ (x₀ y₀ : ℝ), (x₀ = -3 ∧ y₀ = 0) ∧
    (∀ (x' y' : ℝ), y'^2 = -12 * x' ↔ 
      ((x' - x₀)^2 + (y' - y₀)^2 = (x' - (x₀ + 3/4))^2 + y'^2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_focus_l2649_264966


namespace NUMINAMATH_CALUDE_worm_pages_in_four_volumes_l2649_264968

/-- Represents a collection of book volumes -/
structure BookCollection where
  num_volumes : ℕ
  pages_per_volume : ℕ

/-- Calculates the number of pages a worm burrows through in a book collection -/
def worm_burrowed_pages (books : BookCollection) : ℕ :=
  (books.num_volumes - 2) * books.pages_per_volume

/-- Theorem stating the number of pages a worm burrows through in a specific book collection -/
theorem worm_pages_in_four_volumes :
  let books : BookCollection := ⟨4, 200⟩
  worm_burrowed_pages books = 400 := by sorry

end NUMINAMATH_CALUDE_worm_pages_in_four_volumes_l2649_264968


namespace NUMINAMATH_CALUDE_total_turtles_l2649_264917

theorem total_turtles (kristen_turtles : ℕ) (kris_turtles : ℕ) (trey_turtles : ℕ) :
  kristen_turtles = 12 →
  kris_turtles = kristen_turtles / 4 →
  trey_turtles = 5 * kris_turtles →
  kristen_turtles + kris_turtles + trey_turtles = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_turtles_l2649_264917


namespace NUMINAMATH_CALUDE_susan_spending_l2649_264916

def carnival_spending (initial_amount food_cost : ℝ) : ℝ :=
  let ride_cost := 2 * food_cost
  let game_cost := 0.5 * food_cost
  let total_spent := food_cost + ride_cost + game_cost
  initial_amount - total_spent

theorem susan_spending :
  carnival_spending 80 15 = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_susan_spending_l2649_264916


namespace NUMINAMATH_CALUDE_negation_of_positive_quadratic_l2649_264977

theorem negation_of_positive_quadratic (x : ℝ) :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_positive_quadratic_l2649_264977


namespace NUMINAMATH_CALUDE_long_division_unique_solution_l2649_264959

theorem long_division_unique_solution :
  ∃! (dividend divisor quotient : ℕ),
    dividend ≥ 100000 ∧ dividend < 1000000 ∧
    divisor ≥ 100 ∧ divisor < 1000 ∧
    quotient ≥ 100 ∧ quotient < 1000 ∧
    quotient % 10 = 8 ∧
    (divisor * (quotient / 100)) % 10 = 5 ∧
    dividend = divisor * quotient :=
by sorry

end NUMINAMATH_CALUDE_long_division_unique_solution_l2649_264959


namespace NUMINAMATH_CALUDE_farmer_apples_l2649_264964

/-- The number of apples remaining after giving some away -/
def applesRemaining (initial : ℕ) (givenAway : ℕ) : ℕ := initial - givenAway

/-- Theorem: A farmer with 127 apples who gives away 88 apples has 39 apples remaining -/
theorem farmer_apples : applesRemaining 127 88 = 39 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l2649_264964


namespace NUMINAMATH_CALUDE_multiplication_mistake_difference_l2649_264991

theorem multiplication_mistake_difference : 
  let correct_number : ℕ := 139
  let correct_multiplier : ℕ := 43
  let mistaken_multiplier : ℕ := 34
  (correct_number * correct_multiplier) - (correct_number * mistaken_multiplier) = 1251 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_difference_l2649_264991


namespace NUMINAMATH_CALUDE_original_number_l2649_264946

theorem original_number (x : ℝ) : (1.15 * (1.10 * x) = 632.5) → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l2649_264946


namespace NUMINAMATH_CALUDE_two_fifths_of_seven_point_five_l2649_264913

theorem two_fifths_of_seven_point_five : (2 / 5 : ℚ) * (15 / 2 : ℚ) = (3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_of_seven_point_five_l2649_264913


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l2649_264933

theorem gcd_digits_bound (a b : ℕ) (ha : 1000000 ≤ a ∧ a < 10000000) (hb : 1000000 ≤ b ∧ b < 10000000)
  (hlcm : Nat.lcm a b < 100000000000) : Nat.gcd a b < 1000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l2649_264933


namespace NUMINAMATH_CALUDE_snow_volume_calculation_l2649_264953

/-- Calculates the total volume of snow on a sidewalk with two layers -/
theorem snow_volume_calculation 
  (length : ℝ) 
  (width : ℝ) 
  (depth1 : ℝ) 
  (depth2 : ℝ) 
  (h1 : length = 25) 
  (h2 : width = 3) 
  (h3 : depth1 = 1/3) 
  (h4 : depth2 = 1/4) : 
  length * width * depth1 + length * width * depth2 = 43.75 := by
  sorry

end NUMINAMATH_CALUDE_snow_volume_calculation_l2649_264953


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2649_264956

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) := by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, 2 * x + 1 ≤ 0) ↔ (∀ x : ℝ, 2 * x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2649_264956


namespace NUMINAMATH_CALUDE_right_triangle_area_l2649_264928

/-- The area of a right triangle with hypotenuse 5 and one leg 3 is 6 -/
theorem right_triangle_area : ∀ (a b c : ℝ), 
  a = 3 → 
  c = 5 → 
  a^2 + b^2 = c^2 → 
  (1/2) * a * b = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2649_264928


namespace NUMINAMATH_CALUDE_min_knights_in_tournament_l2649_264903

def knight_tournament (total_knights : ℕ) : Prop :=
  ∃ (lancelot_not_dueled : ℕ),
    lancelot_not_dueled = total_knights / 4 ∧
    ∃ (tristan_dueled : ℕ),
      tristan_dueled = (total_knights - lancelot_not_dueled - 1) / 7 ∧
      (total_knights - lancelot_not_dueled - 1) % 7 = 0

theorem min_knights_in_tournament :
  ∀ n : ℕ, knight_tournament n → n ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_min_knights_in_tournament_l2649_264903


namespace NUMINAMATH_CALUDE_special_nine_digit_numbers_exist_l2649_264982

/-- Represents a nine-digit number in the specified format -/
structure NineDigitNumber where
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ
  b₁ : ℕ
  b₂ : ℕ
  b₃ : ℕ
  h₁ : a₁ ≠ 0
  h₂ : b₁ * 100 + b₂ * 10 + b₃ = 2 * (a₁ * 100 + a₂ * 10 + a₃)

/-- The value of the nine-digit number -/
def NineDigitNumber.value (n : NineDigitNumber) : ℕ :=
  n.a₁ * 100000000 + n.a₂ * 10000000 + n.a₃ * 1000000 +
  n.b₁ * 100000 + n.b₂ * 10000 + n.b₃ * 1000 +
  n.a₁ * 100 + n.a₂ * 10 + n.a₃

/-- Theorem stating the existence of the special nine-digit numbers -/
theorem special_nine_digit_numbers_exist : ∃ (n : NineDigitNumber),
  (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧
    n.value = (p₁ * p₂ * p₃ * p₄ * p₅)^2) ∧
  (n.value = 100200100 ∨ n.value = 225450225) :=
sorry

end NUMINAMATH_CALUDE_special_nine_digit_numbers_exist_l2649_264982


namespace NUMINAMATH_CALUDE_period_3_odd_function_inequality_l2649_264914

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem period_3_odd_function_inequality (f : ℝ → ℝ) (a : ℝ) 
    (h_periodic : is_periodic f 3)
    (h_odd : is_odd f)
    (h_f1 : f 1 < 1)
    (h_f2 : f 2 = (2*a - 1)/(a + 1)) :
    a < -1 ∨ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_period_3_odd_function_inequality_l2649_264914


namespace NUMINAMATH_CALUDE_orange_ratio_problem_l2649_264915

theorem orange_ratio_problem (michaela_oranges : ℕ) (total_oranges : ℕ) (remaining_oranges : ℕ) :
  michaela_oranges = 20 →
  total_oranges = 90 →
  remaining_oranges = 30 →
  (total_oranges - remaining_oranges - michaela_oranges) / michaela_oranges = 2 :=
by sorry

end NUMINAMATH_CALUDE_orange_ratio_problem_l2649_264915


namespace NUMINAMATH_CALUDE_factorial_sum_equals_natural_sum_squared_l2649_264900

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (k : ℕ) : ℕ := (List.range k).map factorial |>.sum

def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

theorem factorial_sum_equals_natural_sum_squared :
  ∀ k n : ℕ, sum_of_factorials k = (sum_of_naturals n)^2 ↔ (k = 1 ∧ n = 1) ∨ (k = 3 ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_natural_sum_squared_l2649_264900


namespace NUMINAMATH_CALUDE_complex_magnitude_calculation_l2649_264918

theorem complex_magnitude_calculation (ω : ℂ) (h : ω = 7 + 3*I) :
  Complex.abs (ω^2 + 5*ω + 50) = Real.sqrt 18874 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_calculation_l2649_264918


namespace NUMINAMATH_CALUDE_correct_yellow_balls_drawn_l2649_264993

/-- Calculates the number of yellow balls to be drawn in a stratified sampling -/
def yellowBallsToDraw (totalBalls : ℕ) (yellowBalls : ℕ) (sampleSize : ℕ) : ℕ :=
  (yellowBalls * sampleSize) / totalBalls

theorem correct_yellow_balls_drawn (totalBalls : ℕ) (yellowBalls : ℕ) (sampleSize : ℕ) 
    (h1 : totalBalls = 800) 
    (h2 : yellowBalls = 40) 
    (h3 : sampleSize = 60) : 
  yellowBallsToDraw totalBalls yellowBalls sampleSize = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_yellow_balls_drawn_l2649_264993


namespace NUMINAMATH_CALUDE_total_tax_collection_l2649_264906

/-- Represents the farm tax collection in a village -/
structure FarmTaxCollection where
  totalTax : ℝ
  farmerTax : ℝ
  farmerLandRatio : ℝ

/-- Theorem: Given a farmer's tax payment and land ratio, prove the total tax collected -/
theorem total_tax_collection (ftc : FarmTaxCollection) 
  (h1 : ftc.farmerTax = 480)
  (h2 : ftc.farmerLandRatio = 0.3125)
  : ftc.totalTax = 1536 := by
  sorry

#check total_tax_collection

end NUMINAMATH_CALUDE_total_tax_collection_l2649_264906


namespace NUMINAMATH_CALUDE_smallest_k_l2649_264925

theorem smallest_k (a b c k : ℤ) : 
  (a + 2 = b - 2) → 
  (a + 2 = (c : ℚ) / 2) → 
  (a + b + c = 2001 * k) → 
  (∀ m : ℤ, m > 0 → m < k → ¬(∃ a' b' c' : ℤ, 
    (a' + 2 = b' - 2) ∧ 
    (a' + 2 = (c' : ℚ) / 2) ∧ 
    (a' + b' + c' = 2001 * m))) → 
  k = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_l2649_264925


namespace NUMINAMATH_CALUDE_ellipse_condition_l2649_264998

/-- The equation of the graph is x^2 + 9y^2 - 6x + 27y = k -/
def graph_equation (x y k : ℝ) : Prop :=
  x^2 + 9*y^2 - 6*x + 27*y = k

/-- A non-degenerate ellipse has a positive right-hand side when in standard form -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  k > -29.25

theorem ellipse_condition (k : ℝ) :
  (∀ x y, graph_equation x y k ↔ is_non_degenerate_ellipse k) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2649_264998


namespace NUMINAMATH_CALUDE_negation_equivalence_l2649_264979

theorem negation_equivalence :
  (¬ ∀ (a b : ℝ), a + b = 0 → a^2 + b^2 = 0) ↔
  (∃ (a b : ℝ), a + b = 0 ∧ a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2649_264979


namespace NUMINAMATH_CALUDE_no_eighteen_consecutive_good_numbers_l2649_264934

/-- A natural number is good if it has exactly two prime divisors. -/
def IsGood (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ (∀ r : ℕ, Prime r → r ∣ n → r = p ∨ r = q)

/-- Theorem: It is impossible for 18 consecutive natural numbers to all be good. -/
theorem no_eighteen_consecutive_good_numbers :
  ¬∃ start : ℕ, ∀ i : ℕ, i < 18 → IsGood (start + i) := by
  sorry

end NUMINAMATH_CALUDE_no_eighteen_consecutive_good_numbers_l2649_264934


namespace NUMINAMATH_CALUDE_board_number_equation_l2649_264957

theorem board_number_equation (n : ℤ) : 7 * n + 3 = (3 * n + 7) + 84 ↔ n = 22 := by
  sorry

end NUMINAMATH_CALUDE_board_number_equation_l2649_264957


namespace NUMINAMATH_CALUDE_oprah_car_collection_reduction_l2649_264973

/-- The number of years required to reduce a car collection -/
def years_to_reduce (initial_cars : ℕ) (target_cars : ℕ) (cars_per_year : ℕ) : ℕ :=
  (initial_cars - target_cars) / cars_per_year

/-- Theorem: It takes 60 years to reduce Oprah's car collection from 3500 to 500 cars -/
theorem oprah_car_collection_reduction :
  years_to_reduce 3500 500 50 = 60 := by
  sorry

end NUMINAMATH_CALUDE_oprah_car_collection_reduction_l2649_264973


namespace NUMINAMATH_CALUDE_gabriel_pages_read_l2649_264967

theorem gabriel_pages_read (beatrix_pages cristobal_pages gabriel_pages : ℕ) : 
  beatrix_pages = 704 →
  cristobal_pages = 3 * beatrix_pages + 15 →
  gabriel_pages = 3 * (cristobal_pages + beatrix_pages) →
  gabriel_pages = 8493 := by
  sorry

end NUMINAMATH_CALUDE_gabriel_pages_read_l2649_264967


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l2649_264992

/-- A quadrilateral inscribed in a circle with given properties -/
structure InscribedQuadrilateral where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The lengths of the four sides of the quadrilateral -/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- The theorem stating the properties of the specific inscribed quadrilateral -/
theorem inscribed_quadrilateral_fourth_side
  (q : InscribedQuadrilateral)
  (h_radius : q.radius = 100 * Real.sqrt 6)
  (h_side1 : q.side1 = 100)
  (h_side2 : q.side2 = 200)
  (h_side3 : q.side3 = 200) :
  q.side4 = 100 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l2649_264992


namespace NUMINAMATH_CALUDE_puppy_weight_l2649_264974

theorem puppy_weight (puppy smaller_kitten larger_kitten : ℝ)
  (total_weight : puppy + smaller_kitten + larger_kitten = 30)
  (weight_comparison1 : puppy + larger_kitten = 3 * smaller_kitten)
  (weight_comparison2 : puppy + smaller_kitten = larger_kitten) :
  puppy = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_puppy_weight_l2649_264974


namespace NUMINAMATH_CALUDE_travel_time_calculation_l2649_264907

/-- Given a person travels 2 miles in 8 minutes, prove they will travel 5 miles in 20 minutes at the same rate. -/
theorem travel_time_calculation (distance_1 : ℝ) (time_1 : ℝ) (distance_2 : ℝ) 
  (h1 : distance_1 = 2) 
  (h2 : time_1 = 8) 
  (h3 : distance_2 = 5) :
  (distance_2 / (distance_1 / time_1)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l2649_264907


namespace NUMINAMATH_CALUDE_dinner_bill_split_l2649_264961

theorem dinner_bill_split (total_bill : ℝ) (num_friends : ℕ) 
  (h_total_bill : total_bill = 150)
  (h_num_friends : num_friends = 6) :
  let silas_payment := total_bill / 2
  let remaining_amount := total_bill - silas_payment
  let tip := total_bill * 0.1
  let total_to_split := remaining_amount + tip
  let num_remaining_friends := num_friends - 1
  total_to_split / num_remaining_friends = 18 := by
sorry

end NUMINAMATH_CALUDE_dinner_bill_split_l2649_264961


namespace NUMINAMATH_CALUDE_fifth_root_of_x_fourth_root_of_x_l2649_264975

theorem fifth_root_of_x_fourth_root_of_x (x : ℝ) (hx : x > 0) :
  (x * x^(1/4))^(1/5) = x^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_x_fourth_root_of_x_l2649_264975


namespace NUMINAMATH_CALUDE_integer_equation_solution_l2649_264947

theorem integer_equation_solution : 
  ∀ m n : ℕ+, m^2 + 2*n^2 = 3*(m + 2*n) ↔ (m = 3 ∧ n = 3) ∨ (m = 4 ∧ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_integer_equation_solution_l2649_264947


namespace NUMINAMATH_CALUDE_specific_cistern_wet_surface_area_l2649_264930

/-- Calculates the total wet surface area of a rectangular cistern. -/
def cistern_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem stating the total wet surface area of a specific cistern. -/
theorem specific_cistern_wet_surface_area :
  cistern_wet_surface_area 10 6 1.35 = 103.2 := by
  sorry

end NUMINAMATH_CALUDE_specific_cistern_wet_surface_area_l2649_264930


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l2649_264943

theorem range_of_a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 4) :
  ∃ (x : ℝ), -3 < x ∧ x < 6 ∧ ∃ (a' b' : ℝ), 1 < a' ∧ a' < 4 ∧ -2 < b' ∧ b' < 4 ∧ x = a' - b' :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l2649_264943


namespace NUMINAMATH_CALUDE_points_on_line_l2649_264901

/-- Given points M(a, 1/b) and N(b, 1/c) on the line x + y = 1,
    prove that points P(c, 1/a) and Q(1/c, b) are also on the same line. -/
theorem points_on_line (a b c : ℝ) (ha : a + 1/b = 1) (hb : b + 1/c = 1) :
  c + 1/a = 1 ∧ 1/c + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_l2649_264901


namespace NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l2649_264922

theorem smallest_third_term_geometric_progression (d : ℝ) :
  (5 : ℝ) * (33 + 2 * d) = (8 + d) ^ 2 →
  ∃ (x : ℝ), x = 5 + 2 * d + 28 ∧
    ∀ (y : ℝ), (5 : ℝ) * (33 + 2 * y) = (8 + y) ^ 2 →
      5 + 2 * y + 28 ≥ x ∧
      x ≥ -21 :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l2649_264922


namespace NUMINAMATH_CALUDE_absolute_value_equals_negative_l2649_264923

theorem absolute_value_equals_negative (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_negative_l2649_264923


namespace NUMINAMATH_CALUDE_simplify_expression_l2649_264983

theorem simplify_expression : (576 : ℝ) ^ (1/4) * (216 : ℝ) ^ (1/2) = 72 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2649_264983


namespace NUMINAMATH_CALUDE_farm_area_theorem_l2649_264902

/-- Represents a rectangular farm with fencing on one long side, one short side, and the diagonal -/
structure RectangularFarm where
  short_side : ℝ
  long_side : ℝ
  diagonal : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Calculates the area of a rectangular farm -/
def farm_area (farm : RectangularFarm) : ℝ :=
  farm.short_side * farm.long_side

/-- Theorem: If a rectangular farm has a short side of 30 meters, and the cost of fencing
    one long side, one short side, and the diagonal at Rs. 15 per meter totals Rs. 1800,
    then the area of the farm is 1200 square meters -/
theorem farm_area_theorem (farm : RectangularFarm)
    (h1 : farm.short_side = 30)
    (h2 : farm.fencing_cost_per_meter = 15)
    (h3 : farm.total_fencing_cost = 1800)
    (h4 : farm.long_side + farm.short_side + farm.diagonal = farm.total_fencing_cost / farm.fencing_cost_per_meter)
    (h5 : farm.diagonal^2 = farm.long_side^2 + farm.short_side^2) :
    farm_area farm = 1200 := by
  sorry


end NUMINAMATH_CALUDE_farm_area_theorem_l2649_264902


namespace NUMINAMATH_CALUDE_equal_distances_l2649_264970

/-- Represents a right triangle with squares on its sides -/
structure RightTriangleWithSquares where
  -- The lengths of the sides of the right triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The acute angle α
  α : ℝ
  -- Conditions
  right_triangle : c^2 = a^2 + b^2
  acute_angle : 0 < α ∧ α < π / 2
  angle_sum : α + (π / 2 - α) = π / 2

/-- The theorem stating that the distances O₁O₂ and CO₃ are equal -/
theorem equal_distances (t : RightTriangleWithSquares) : 
  (t.a + t.b) / Real.sqrt 2 = t.c / Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_equal_distances_l2649_264970


namespace NUMINAMATH_CALUDE_repetend_of_five_seventeenths_l2649_264921

/-- The decimal representation of 5/17 has a repetend of 294117647058823529 -/
theorem repetend_of_five_seventeenths :
  ∃ (n : ℕ), (5 : ℚ) / 17 = (n : ℚ) / 999999999999999999 ∧
  n = 294117647058823529 := by
  sorry

end NUMINAMATH_CALUDE_repetend_of_five_seventeenths_l2649_264921


namespace NUMINAMATH_CALUDE_no_natural_solution_l2649_264962

theorem no_natural_solution :
  ¬∃ (x y : ℕ), x^2 + y^2 + 1 = 6*x*y := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l2649_264962


namespace NUMINAMATH_CALUDE_complex_polynomial_root_implies_abs_c_165_l2649_264931

def complex_polynomial (a b c : ℤ) : ℂ → ℂ := fun z ↦ a * z^4 + b * z^3 + c * z^2 + b * z + a

theorem complex_polynomial_root_implies_abs_c_165 (a b c : ℤ) :
  complex_polynomial a b c (3 + I) = 0 →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 165 := by sorry

end NUMINAMATH_CALUDE_complex_polynomial_root_implies_abs_c_165_l2649_264931


namespace NUMINAMATH_CALUDE_min_sum_pqr_l2649_264948

/-- Given five positive integers with pairwise GCDs as specified, 
    the minimum sum of p, q, and r is 9 -/
theorem min_sum_pqr (a b c d e : ℕ+) 
  (h : ∃ (p q r : ℕ+), Set.toFinset {Nat.gcd a.val b.val, Nat.gcd a.val c.val, 
    Nat.gcd a.val d.val, Nat.gcd a.val e.val, Nat.gcd b.val c.val, 
    Nat.gcd b.val d.val, Nat.gcd b.val e.val, Nat.gcd c.val d.val, 
    Nat.gcd c.val e.val, Nat.gcd d.val e.val} = 
    Set.toFinset {2, 3, 4, 5, 6, 7, 8, p.val, q.val, r.val}) : 
  (∃ (p q r : ℕ+), Set.toFinset {Nat.gcd a.val b.val, Nat.gcd a.val c.val, 
    Nat.gcd a.val d.val, Nat.gcd a.val e.val, Nat.gcd b.val c.val, 
    Nat.gcd b.val d.val, Nat.gcd b.val e.val, Nat.gcd c.val d.val, 
    Nat.gcd c.val e.val, Nat.gcd d.val e.val} = 
    Set.toFinset {2, 3, 4, 5, 6, 7, 8, p.val, q.val, r.val} ∧ 
    p.val + q.val + r.val = 9 ∧ 
    ∀ (p' q' r' : ℕ+), Set.toFinset {Nat.gcd a.val b.val, Nat.gcd a.val c.val, 
      Nat.gcd a.val d.val, Nat.gcd a.val e.val, Nat.gcd b.val c.val, 
      Nat.gcd b.val d.val, Nat.gcd b.val e.val, Nat.gcd c.val d.val, 
      Nat.gcd c.val e.val, Nat.gcd d.val e.val} = 
      Set.toFinset {2, 3, 4, 5, 6, 7, 8, p'.val, q'.val, r'.val} → 
      p'.val + q'.val + r'.val ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_pqr_l2649_264948


namespace NUMINAMATH_CALUDE_football_game_ratio_l2649_264986

theorem football_game_ratio : 
  -- Given conditions
  let total_start : ℕ := 600
  let girls_start : ℕ := 240
  let remaining : ℕ := 480
  let girls_left : ℕ := girls_start / 8

  -- Derived values
  let boys_start : ℕ := total_start - girls_start
  let total_left : ℕ := total_start - remaining
  let boys_left : ℕ := total_left - girls_left

  -- Theorem statement
  boys_left * 4 = boys_start :=
by sorry

end NUMINAMATH_CALUDE_football_game_ratio_l2649_264986


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2649_264950

theorem inscribed_cube_surface_area (outer_cube_surface_area : ℝ) :
  outer_cube_surface_area = 54 →
  ∃ (inner_cube_surface_area : ℝ),
    inner_cube_surface_area = 18 ∧
    (∃ (outer_cube_side : ℝ) (sphere_diameter : ℝ) (inner_cube_side : ℝ),
      outer_cube_side^3 = outer_cube_surface_area / 6 ∧
      sphere_diameter = outer_cube_side ∧
      inner_cube_side^2 * 3 = sphere_diameter^2 ∧
      inner_cube_surface_area = 6 * inner_cube_side^2) :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2649_264950


namespace NUMINAMATH_CALUDE_divisibility_condition_l2649_264911

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔ 
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2649_264911


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2649_264949

theorem unique_solution_condition (k : ℤ) : 
  (∃! x : ℝ, (3 * x + 5) * (x - 4) = -40 + k * x) ↔ (k = 8 ∨ k = -22) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2649_264949


namespace NUMINAMATH_CALUDE_raft_sticks_difference_l2649_264996

theorem raft_sticks_difference (simon_sticks : ℕ) (total_sticks : ℕ) : 
  simon_sticks = 36 →
  total_sticks = 129 →
  let gerry_sticks := (2 * simon_sticks) / 3
  let simon_and_gerry_sticks := simon_sticks + gerry_sticks
  let micky_sticks := total_sticks - simon_and_gerry_sticks
  micky_sticks - simon_and_gerry_sticks = 9 := by
  sorry

end NUMINAMATH_CALUDE_raft_sticks_difference_l2649_264996


namespace NUMINAMATH_CALUDE_wheel_marking_theorem_l2649_264978

theorem wheel_marking_theorem :
  ∃ (R : ℝ), R > 0 ∧
    ∀ (θ : ℝ), 0 ≤ θ ∧ θ < 360 →
      ∃ (n : ℕ), ∃ (k : ℤ),
        n / (2 * π * R) = θ / 360 + k ∧
        0 ≤ n / (2 * π * R) - k ∧
        n / (2 * π * R) - k < 1 / 360 :=
by sorry

end NUMINAMATH_CALUDE_wheel_marking_theorem_l2649_264978


namespace NUMINAMATH_CALUDE_part_one_part_two_l2649_264988

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

-- Part 1
theorem part_one (m : ℝ) (h : m > 0) :
  (∀ x, p x → q m x) → m ∈ Set.Ici 4 := by sorry

-- Part 2
theorem part_two (x : ℝ) :
  (p x ∨ q 5 x) ∧ ¬(p x ∧ q 5 x) →
  x ∈ Set.Ioc (-3) (-2) ∪ Set.Ioc 6 7 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2649_264988


namespace NUMINAMATH_CALUDE_inequality_solution_l2649_264929

theorem inequality_solution (x : ℝ) : 
  (-1 < (x^2 - 10*x + 9) / (x^2 - 4*x + 8) ∧ (x^2 - 10*x + 9) / (x^2 - 4*x + 8) < 1) ↔ x > 1/6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2649_264929


namespace NUMINAMATH_CALUDE_election_votes_l2649_264932

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (62 : ℚ) / 100 * total_votes - (38 : ℚ) / 100 * total_votes = 408) :
  (62 : ℚ) / 100 * total_votes = 1054 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_l2649_264932


namespace NUMINAMATH_CALUDE_problem_solution_l2649_264955

theorem problem_solution : (3.242 * 14) / 100 = 0.45388 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2649_264955


namespace NUMINAMATH_CALUDE_inequality_proof_l2649_264995

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (3 : ℝ) / 2 < 1 / (a^3 + 1) + 1 / (b^3 + 1) ∧ 1 / (a^3 + 1) + 1 / (b^3 + 1) ≤ 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2649_264995


namespace NUMINAMATH_CALUDE_leftover_money_l2649_264954

/-- Calculates the leftover money after reading books and buying candy -/
theorem leftover_money
  (payment_rate : ℚ)
  (pages_per_book : ℕ)
  (books_read : ℕ)
  (candy_cost : ℚ)
  (h1 : payment_rate = 1 / 100)  -- $0.01 per page
  (h2 : pages_per_book = 150)
  (h3 : books_read = 12)
  (h4 : candy_cost = 15) :
  payment_rate * (pages_per_book * books_read : ℚ) - candy_cost = 3 := by
sorry

end NUMINAMATH_CALUDE_leftover_money_l2649_264954


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2649_264938

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2, 4}

-- Define set N
def N : Set Nat := {2, 4, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl M ∩ Set.compl N : Set Nat) = {3} :=
by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2649_264938


namespace NUMINAMATH_CALUDE_our_system_is_linear_l2649_264971

/-- Represents a linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  -- ax + by = c

/-- Represents a system of two equations -/
structure EquationSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- Checks if an equation is linear -/
def isLinear (eq : LinearEquation) : Prop :=
  eq.a ≠ 0 ∨ eq.b ≠ 0

/-- Checks if a system consists of two linear equations -/
def isSystemOfTwoLinearEquations (sys : EquationSystem) : Prop :=
  isLinear sys.eq1 ∧ isLinear sys.eq2

/-- The specific system we want to prove is a system of two linear equations -/
def ourSystem : EquationSystem :=
  { eq1 := { a := 1, b := 1, c := 5 }  -- x + y = 5
    eq2 := { a := 0, b := 1, c := 2 }  -- y = 2
  }

/-- Theorem stating that our system is a system of two linear equations -/
theorem our_system_is_linear : isSystemOfTwoLinearEquations ourSystem := by
  sorry


end NUMINAMATH_CALUDE_our_system_is_linear_l2649_264971


namespace NUMINAMATH_CALUDE_log_18_15_l2649_264927

-- Define the logarithm base 10 (lg) function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the variables a and b
variable (a b : ℝ)

-- State the theorem
theorem log_18_15 (h1 : lg 2 = a) (h2 : lg 3 = b) :
  (Real.log 15) / (Real.log 18) = (b - a + 1) / (a + 2 * b) := by
  sorry

end NUMINAMATH_CALUDE_log_18_15_l2649_264927


namespace NUMINAMATH_CALUDE_propositions_B_and_C_l2649_264945

theorem propositions_B_and_C :
  (∀ x : ℚ, ∃ y : ℚ, y = (1/3) * x^2 + (1/2) * x + 1) ∧
  (∃ x y : ℤ, 3 * x - 2 * y = 10) := by
  sorry

end NUMINAMATH_CALUDE_propositions_B_and_C_l2649_264945


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l2649_264939

theorem abs_inequality_equivalence (x : ℝ) : 
  |5 - 2*x| < 3 ↔ 1 < x ∧ x < 4 :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l2649_264939


namespace NUMINAMATH_CALUDE_octagon_has_eight_sides_l2649_264936

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Theorem stating that an octagon has 8 sides -/
theorem octagon_has_eight_sides : octagon_sides = 8 := by
  sorry

end NUMINAMATH_CALUDE_octagon_has_eight_sides_l2649_264936


namespace NUMINAMATH_CALUDE_subtracted_number_l2649_264958

theorem subtracted_number (x y : ℤ) (h1 : x = 40) (h2 : 6 * x - y = 102) : y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l2649_264958


namespace NUMINAMATH_CALUDE_mixture_volume_proportion_l2649_264994

/-- Given two solutions P and Q, where P is 80% carbonated water and Q is 55% carbonated water,
    if a mixture of P and Q contains 67.5% carbonated water, then the volume of P in the mixture
    is 50% of the total volume. -/
theorem mixture_volume_proportion (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  0.80 * x + 0.55 * y = 0.675 * (x + y) →
  x / (x + y) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_mixture_volume_proportion_l2649_264994


namespace NUMINAMATH_CALUDE_sum_nonzero_digits_base8_999_l2649_264951

/-- Converts a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the non-zero elements of a list -/
def sumNonZero (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The sum of non-zero digits in the base 8 representation of 999 is 19 -/
theorem sum_nonzero_digits_base8_999 : sumNonZero (toBase8 999) = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_nonzero_digits_base8_999_l2649_264951


namespace NUMINAMATH_CALUDE_specific_grid_square_count_l2649_264937

/-- Represents a square grid with some incomplete squares at the edges -/
structure SquareGrid :=
  (width : ℕ)
  (height : ℕ)
  (hasIncompleteEdges : Bool)

/-- Counts the number of squares of a given size in the grid -/
def countSquares (grid : SquareGrid) (size : ℕ) : ℕ :=
  sorry

/-- Counts the total number of squares in the grid -/
def totalSquares (grid : SquareGrid) : ℕ :=
  (countSquares grid 1) + (countSquares grid 2) + (countSquares grid 3)

/-- The main theorem stating that the total number of squares in the specific grid is 38 -/
theorem specific_grid_square_count :
  ∃ (grid : SquareGrid), grid.width = 5 ∧ grid.height = 5 ∧ grid.hasIncompleteEdges = true ∧ totalSquares grid = 38 :=
  sorry

end NUMINAMATH_CALUDE_specific_grid_square_count_l2649_264937


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2649_264965

theorem unique_solution_quadratic_system :
  ∃! y : ℚ, (9 * y^2 + 8 * y - 3 = 0) ∧ (27 * y^2 + 35 * y - 12 = 0) ∧ (y = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2649_264965


namespace NUMINAMATH_CALUDE_cubic_sum_values_l2649_264940

def M (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![a, b, c],
    ![b, c, a],
    ![c, a, b]]

theorem cubic_sum_values (a b c : ℂ) :
  M a b c ^ 2 = 1 →
  a * b * c = -1 →
  (a^3 + b^3 + c^3 = -2) ∨ (a^3 + b^3 + c^3 = -4) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_values_l2649_264940


namespace NUMINAMATH_CALUDE_unique_a_value_l2649_264905

def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + 4 = 0}

theorem unique_a_value : ∃! a : ℝ, (B a).Nonempty ∧ B a ⊆ A ∧ a = 4 := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l2649_264905


namespace NUMINAMATH_CALUDE_triangle_inradius_l2649_264985

/-- Given a triangle with perimeter 20 cm and area 25 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : p = 20) 
  (h_area : A = 25) 
  (h_inradius : A = r * p / 2) : 
  r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l2649_264985


namespace NUMINAMATH_CALUDE_twelfth_even_multiple_of_5_l2649_264942

/-- The nth positive integer that is both even and a multiple of 5 -/
def evenMultipleOf5 (n : ℕ) : ℕ := 10 * n

/-- Proof that the 12th positive integer that is both even and a multiple of 5 is 120 -/
theorem twelfth_even_multiple_of_5 : evenMultipleOf5 12 = 120 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_even_multiple_of_5_l2649_264942


namespace NUMINAMATH_CALUDE_chip_cost_theorem_l2649_264984

theorem chip_cost_theorem (calories_per_chip : ℕ) (chips_per_bag : ℕ) (cost_per_bag : ℕ) (target_calories : ℕ) : 
  calories_per_chip = 10 →
  chips_per_bag = 24 →
  cost_per_bag = 2 →
  target_calories = 480 →
  (target_calories / (calories_per_chip * chips_per_bag)) * cost_per_bag = 4 := by
  sorry

end NUMINAMATH_CALUDE_chip_cost_theorem_l2649_264984


namespace NUMINAMATH_CALUDE_trailing_zeros_count_product_trailing_zeros_l2649_264963

def product : ℕ := 25^7 * 8^3

theorem trailing_zeros_count (n : ℕ) : ℕ :=
  sorry

theorem product_trailing_zeros : trailing_zeros_count product = 9 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_count_product_trailing_zeros_l2649_264963
