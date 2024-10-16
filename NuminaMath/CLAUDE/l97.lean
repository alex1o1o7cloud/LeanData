import Mathlib

namespace NUMINAMATH_CALUDE_divisor_condition_l97_9782

theorem divisor_condition (k : ℕ+) :
  (∃ (n : ℕ+), (8 * k * n - 1) ∣ (4 * k^2 - 1)^2) ↔ Even k :=
sorry

end NUMINAMATH_CALUDE_divisor_condition_l97_9782


namespace NUMINAMATH_CALUDE_weight_difference_theorem_l97_9753

-- Define the given conditions
def joe_weight : ℝ := 44
def initial_average : ℝ := 30
def new_average : ℝ := 31
def final_average : ℝ := 30

-- Define the number of students in the initial group
def initial_students : ℕ := 13

-- Define the theorem
theorem weight_difference_theorem :
  let total_weight_with_joe := initial_average * initial_students + joe_weight
  let remaining_students := initial_students + 1 - 2
  let final_total_weight := final_average * remaining_students
  let leaving_students_total_weight := total_weight_with_joe - final_total_weight
  let leaving_students_average_weight := leaving_students_total_weight / 2
  leaving_students_average_weight - joe_weight = -7 := by sorry

end NUMINAMATH_CALUDE_weight_difference_theorem_l97_9753


namespace NUMINAMATH_CALUDE_average_team_goals_l97_9770

-- Define the average goals per game for each player
def carter_goals : ℚ := 4
def shelby_goals : ℚ := carter_goals / 2
def judah_goals : ℚ := 2 * shelby_goals - 3
def morgan_goals : ℚ := judah_goals + 1
def alex_goals : ℚ := carter_goals / 2 - 2
def taylor_goals : ℚ := 1 / 3

-- Define the total goals per game for the team
def team_goals : ℚ := carter_goals + shelby_goals + judah_goals + morgan_goals + alex_goals + taylor_goals

-- Theorem statement
theorem average_team_goals : team_goals = 28 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_team_goals_l97_9770


namespace NUMINAMATH_CALUDE_firm_partners_count_l97_9792

theorem firm_partners_count (partners associates : ℕ) : 
  partners / associates = 2 / 63 →
  partners / (associates + 35) = 1 / 34 →
  partners = 14 :=
by sorry

end NUMINAMATH_CALUDE_firm_partners_count_l97_9792


namespace NUMINAMATH_CALUDE_rectangle_area_relation_l97_9791

/-- 
For a rectangle with area 12 and sides of length x and y,
the function relationship between y and x is y = 12/x.
-/
theorem rectangle_area_relation (x y : ℝ) (h : x * y = 12) : 
  y = 12 / x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_relation_l97_9791


namespace NUMINAMATH_CALUDE_vector_triangle_inequality_l97_9794

variable {V : Type*} [NormedAddCommGroup V]

theorem vector_triangle_inequality (a b : V) : ‖a + b‖ ≤ ‖a‖ + ‖b‖ := by
  sorry

end NUMINAMATH_CALUDE_vector_triangle_inequality_l97_9794


namespace NUMINAMATH_CALUDE_equation_solutions_l97_9713

theorem equation_solutions : 
  ∃! (s : Set ℝ), (∀ x ∈ s, |x - 2| = |x - 5| + |x - 8|) ∧ s = {5, 11} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l97_9713


namespace NUMINAMATH_CALUDE_pencil_arrangement_theorem_l97_9705

def yellow_pencils : ℕ := 6
def red_pencils : ℕ := 3
def blue_pencils : ℕ := 4

def total_pencils : ℕ := yellow_pencils + red_pencils + blue_pencils

def total_arrangements : ℕ := Nat.factorial total_pencils / (Nat.factorial yellow_pencils * Nat.factorial red_pencils * Nat.factorial blue_pencils)

def arrangements_with_adjacent_blue : ℕ := Nat.factorial (total_pencils - blue_pencils + 1) / (Nat.factorial yellow_pencils * Nat.factorial red_pencils)

theorem pencil_arrangement_theorem :
  total_arrangements - arrangements_with_adjacent_blue = 274400 := by
  sorry

end NUMINAMATH_CALUDE_pencil_arrangement_theorem_l97_9705


namespace NUMINAMATH_CALUDE_not_p_neither_sufficient_nor_necessary_for_not_q_l97_9789

theorem not_p_neither_sufficient_nor_necessary_for_not_q : ∃ (a : ℝ),
  (¬(a > 0) ∧ ¬(a^2 > a)) ∨ (¬(a > 0) ∧ (a^2 > a)) ∨ ((a > 0) ∧ ¬(a^2 > a)) :=
by sorry

end NUMINAMATH_CALUDE_not_p_neither_sufficient_nor_necessary_for_not_q_l97_9789


namespace NUMINAMATH_CALUDE_max_value_2x_plus_y_l97_9711

theorem max_value_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x*y = 5) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ ∀ (z : ℝ), 2 * x + y ≤ z → z ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_2x_plus_y_l97_9711


namespace NUMINAMATH_CALUDE_disjoint_subsets_prime_products_l97_9725

/-- A function that constructs 100 disjoint subsets of positive integers -/
def construct_subsets : Fin 100 → Set ℕ := sorry

/-- Predicate to check if a number is a product of m distinct primes from a set -/
def is_product_of_m_primes (n : ℕ) (m : ℕ) (S : Set ℕ) : Prop := sorry

/-- Main theorem statement -/
theorem disjoint_subsets_prime_products :
  ∃ (A : Fin 100 → Set ℕ), 
    (∀ i j, i ≠ j → Disjoint (A i) (A j)) ∧
    (∀ (S : Set ℕ) (hS : Set.Infinite S) (h_prime : ∀ p ∈ S, Nat.Prime p),
      ∃ (m : ℕ) (a : Fin 100 → ℕ), 
        ∀ i, a i ∈ A i ∧ is_product_of_m_primes (a i) m S) :=
sorry

end NUMINAMATH_CALUDE_disjoint_subsets_prime_products_l97_9725


namespace NUMINAMATH_CALUDE_range_of_t_range_of_t_lower_bound_l97_9734

def A : Set ℝ := {x | -3 < x ∧ x < 7}
def B (t : ℝ) : Set ℝ := {x | t + 1 < x ∧ x < 2*t - 1}

theorem range_of_t (t : ℝ) : B t ⊆ A → t ≤ 4 :=
  sorry

theorem range_of_t_lower_bound : ∀ ε > 0, ∃ t : ℝ, t ≤ 4 - ε ∧ B t ⊆ A :=
  sorry

end NUMINAMATH_CALUDE_range_of_t_range_of_t_lower_bound_l97_9734


namespace NUMINAMATH_CALUDE_modular_inverse_of_2_mod_191_l97_9775

theorem modular_inverse_of_2_mod_191 : ∃ x : ℕ, x < 191 ∧ (2 * x) % 191 = 1 :=
  ⟨96, by
    constructor
    · simp
    · norm_num
  ⟩

#eval (2 * 96) % 191  -- This should output 1

end NUMINAMATH_CALUDE_modular_inverse_of_2_mod_191_l97_9775


namespace NUMINAMATH_CALUDE_ellipse_x_intercept_l97_9778

-- Define the ellipse
def ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ := (0, 3)
  let F₂ := (4, 0)
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 8

-- Theorem statement
theorem ellipse_x_intercept :
  ellipse (0, 0) →
  ∃ x : ℝ, x ≠ 0 ∧ ellipse (x, 0) ∧ x = 45/8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_x_intercept_l97_9778


namespace NUMINAMATH_CALUDE_base8_arithmetic_result_l97_9728

/-- Convert a base 8 number to base 10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Convert a base 10 number to base 8 --/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Perform base 8 arithmetic: multiply by 2 and subtract --/
def base8Arithmetic (a b : ℕ) : ℕ :=
  base10ToBase8 ((base8ToBase10 a) * 2 - (base8ToBase10 b))

theorem base8_arithmetic_result :
  base8Arithmetic 45 76 = 14 := by sorry

end NUMINAMATH_CALUDE_base8_arithmetic_result_l97_9728


namespace NUMINAMATH_CALUDE_trajectory_and_fixed_points_l97_9744

-- Define the points and lines
def F : ℝ × ℝ := (1, 0)
def H : ℝ × ℝ := (1, 2)
def l : Set (ℝ × ℝ) := {p | p.1 = -1}

-- Define the trajectory C
def C : Set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}

-- Define the function for the circle with MN as diameter
def circle_MN (m : ℝ) : Set (ℝ × ℝ) := 
  {p | p.1^2 + 2*p.1 - 3 + p.2^2 + (4/m)*p.2 = 0}

-- Theorem statement
theorem trajectory_and_fixed_points :
  -- Part 1: Trajectory C
  (∀ Q : ℝ × ℝ, (∃ P : ℝ × ℝ, P ∈ l ∧ 
    (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = (Q.1 - F.1)^2 + (Q.2 - F.2)^2) 
    → Q ∈ C) ∧
  -- Part 2: Fixed points on the circle
  (∀ m : ℝ, m ≠ 0 → (-3, 0) ∈ circle_MN m ∧ (1, 0) ∈ circle_MN m) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_fixed_points_l97_9744


namespace NUMINAMATH_CALUDE_tenth_square_area_l97_9796

/-- The area of the nth square in a sequence where each square is formed by connecting
    the midpoints of the previous square's sides, and the first square has a side length of 2. -/
def square_area (n : ℕ) : ℚ :=
  2 * (1 / 2) ^ (n - 1)

/-- Theorem stating that the area of the 10th square in the sequence is 1/256. -/
theorem tenth_square_area :
  square_area 10 = 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_tenth_square_area_l97_9796


namespace NUMINAMATH_CALUDE_f_inequality_l97_9761

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State that f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Define the condition that f(x) > f'(x) for all x
axiom f_greater_than_f' : ∀ x, f x > f' x

-- State the theorem to be proved
theorem f_inequality : 3 * f (Real.log 2) > 2 * f (Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l97_9761


namespace NUMINAMATH_CALUDE_hyperbola_condition_l97_9720

/-- The equation (x^2)/(k-2) + (y^2)/(5-k) = 1 represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  k < 2 ∨ k > 5

/-- The general form of the equation -/
def equation (x y k : ℝ) : Prop :=
  x^2 / (k - 2) + y^2 / (5 - k) = 1

theorem hyperbola_condition (k : ℝ) :
  (∀ x y, equation x y k → is_hyperbola k) ∧
  (is_hyperbola k → ∃ x y, equation x y k) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l97_9720


namespace NUMINAMATH_CALUDE_trapezoid_area_property_l97_9736

/-- Represents the area of a trapezoid with bases and altitude in arithmetic progression -/
def trapezoid_area (a : ℝ) : ℝ := a ^ 2

/-- The area of a trapezoid with bases and altitude in arithmetic progression
    can be any non-negative real number -/
theorem trapezoid_area_property :
  ∀ (J : ℝ), J ≥ 0 → ∃ (a : ℝ), trapezoid_area a = J :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_property_l97_9736


namespace NUMINAMATH_CALUDE_sin_180_degrees_l97_9733

theorem sin_180_degrees : Real.sin (π) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_180_degrees_l97_9733


namespace NUMINAMATH_CALUDE_age_of_replaced_man_l97_9701

/-- Given a group of 8 men where two are replaced by two women, prove the age of one of the replaced men. -/
theorem age_of_replaced_man
  (n : ℕ) -- Total number of people
  (m : ℕ) -- Number of men initially
  (w : ℕ) -- Number of women replacing men
  (A : ℝ) -- Initial average age of men
  (increase : ℝ) -- Increase in average age after replacement
  (known_man_age : ℕ) -- Age of one of the replaced men
  (women_avg_age : ℝ) -- Average age of the women
  (h1 : n = 8)
  (h2 : m = 8)
  (h3 : w = 2)
  (h4 : increase = 2)
  (h5 : known_man_age = 10)
  (h6 : women_avg_age = 23)
  : ∃ (other_man_age : ℕ), other_man_age = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_age_of_replaced_man_l97_9701


namespace NUMINAMATH_CALUDE_tetrahedron_with_two_square_intersections_l97_9706

/-- A tetrahedron in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- The intersection of a tetrahedron and a plane -/
def intersection (t : Tetrahedron) (p : Plane) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Predicate to check if a set of points forms a square -/
def is_square (s : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- The side length of a square -/
def side_length (s : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem stating the existence of a tetrahedron with the desired properties -/
theorem tetrahedron_with_two_square_intersections :
  ∃ (t : Tetrahedron) (p1 p2 : Plane),
    p1 ≠ p2 ∧
    is_square (intersection t p1) ∧
    is_square (intersection t p2) ∧
    side_length (intersection t p1) ≤ 1 ∧
    side_length (intersection t p2) ≥ 100 :=
  sorry

end NUMINAMATH_CALUDE_tetrahedron_with_two_square_intersections_l97_9706


namespace NUMINAMATH_CALUDE_mitch_weekly_earnings_is_118_80_l97_9788

/-- Mitch's weekly earnings after expenses and taxes -/
def mitchWeeklyEarnings : ℝ :=
  let monToWedEarnings := 3 * 5 * 3
  let thuFriEarnings := 2 * 6 * 4
  let satEarnings := 4 * 6
  let sunEarnings := 5 * 8
  let totalEarnings := monToWedEarnings + thuFriEarnings + satEarnings + sunEarnings
  let afterExpenses := totalEarnings - 25
  let taxAmount := afterExpenses * 0.1
  afterExpenses - taxAmount

/-- Theorem stating that Mitch's weekly earnings after expenses and taxes is $118.80 -/
theorem mitch_weekly_earnings_is_118_80 :
  mitchWeeklyEarnings = 118.80 := by sorry

end NUMINAMATH_CALUDE_mitch_weekly_earnings_is_118_80_l97_9788


namespace NUMINAMATH_CALUDE_quadrilateral_iff_interior_exterior_sum_equal_l97_9777

/-- A polygon has 4 sides if and only if the sum of its interior angles is equal to the sum of its exterior angles. -/
theorem quadrilateral_iff_interior_exterior_sum_equal (n : ℕ) : n = 4 ↔ (n - 2) * 180 = 360 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_iff_interior_exterior_sum_equal_l97_9777


namespace NUMINAMATH_CALUDE_line_equivalence_slope_and_intercept_l97_9731

/-- The vector representation of the line -/
def line_vector (x y : ℝ) : ℝ := 2 * (x - 3) + (-1) * (y - (-4))

/-- The slope-intercept form of the line -/
def line_slope_intercept (x y : ℝ) : Prop := y = 2 * x - 10

theorem line_equivalence :
  ∀ x y : ℝ, line_vector x y = 0 ↔ line_slope_intercept x y :=
sorry

theorem slope_and_intercept :
  ∃ m b : ℝ, (∀ x y : ℝ, line_vector x y = 0 → y = m * x + b) ∧ m = 2 ∧ b = -10 :=
sorry

end NUMINAMATH_CALUDE_line_equivalence_slope_and_intercept_l97_9731


namespace NUMINAMATH_CALUDE_luke_stickers_l97_9757

/-- Calculates the number of stickers Luke has left after a series of events. -/
def stickers_left (initial : ℕ) (bought : ℕ) (birthday : ℕ) (given : ℕ) (used : ℕ) : ℕ :=
  initial + bought + birthday - given - used

/-- Proves that Luke has 39 stickers left after the given events. -/
theorem luke_stickers : stickers_left 20 12 20 5 8 = 39 := by
  sorry

end NUMINAMATH_CALUDE_luke_stickers_l97_9757


namespace NUMINAMATH_CALUDE_sqrt_division_equality_l97_9743

theorem sqrt_division_equality : Real.sqrt 3 / Real.sqrt (1/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_division_equality_l97_9743


namespace NUMINAMATH_CALUDE_fewer_toys_by_machine_a_l97_9799

/-- The number of toys machine A makes per minute -/
def machine_a_rate : ℕ := 8

/-- The number of toys machine B makes per minute -/
def machine_b_rate : ℕ := 10

/-- The number of toys machine B made -/
def machine_b_toys : ℕ := 100

/-- The time both machines operated, in minutes -/
def operation_time : ℕ := machine_b_toys / machine_b_rate

/-- The number of toys machine A made -/
def machine_a_toys : ℕ := machine_a_rate * operation_time

theorem fewer_toys_by_machine_a : machine_b_toys - machine_a_toys = 20 := by
  sorry

end NUMINAMATH_CALUDE_fewer_toys_by_machine_a_l97_9799


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l97_9798

theorem min_sum_absolute_values (x : ℝ) :
  ∃ (min : ℝ), min = 4 ∧ 
  (∀ y : ℝ, |y + 3| + |y + 6| + |y + 7| ≥ min) ∧
  (|x + 3| + |x + 6| + |x + 7| = min) :=
sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l97_9798


namespace NUMINAMATH_CALUDE_range_of_a_l97_9730

-- Define the conditions
def p (x : ℝ) : Prop := (x + 1)^2 > 4
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a :
  (∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(q x a) ∧ p x)) →
  (∀ a : ℝ, a ≥ 1 ↔ (∀ x : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(q x a) ∧ p x))) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l97_9730


namespace NUMINAMATH_CALUDE_pablo_works_seven_hours_l97_9774

/-- Represents the puzzle-solving scenario for Pablo --/
structure PuzzleScenario where
  pieces_per_hour : ℕ
  small_puzzles : ℕ
  small_puzzle_pieces : ℕ
  large_puzzles : ℕ
  large_puzzle_pieces : ℕ
  days_to_complete : ℕ

/-- Calculates the hours Pablo works on puzzles each day --/
def hours_per_day (scenario : PuzzleScenario) : ℚ :=
  let total_pieces := scenario.small_puzzles * scenario.small_puzzle_pieces +
                      scenario.large_puzzles * scenario.large_puzzle_pieces
  let total_hours := total_pieces / scenario.pieces_per_hour
  total_hours / scenario.days_to_complete

/-- Theorem stating that Pablo works 7 hours per day on puzzles --/
theorem pablo_works_seven_hours (scenario : PuzzleScenario) 
  (h1 : scenario.pieces_per_hour = 100)
  (h2 : scenario.small_puzzles = 8)
  (h3 : scenario.small_puzzle_pieces = 300)
  (h4 : scenario.large_puzzles = 5)
  (h5 : scenario.large_puzzle_pieces = 500)
  (h6 : scenario.days_to_complete = 7) :
  hours_per_day scenario = 7 := by
  sorry

end NUMINAMATH_CALUDE_pablo_works_seven_hours_l97_9774


namespace NUMINAMATH_CALUDE_systematic_sampling_probabilities_l97_9747

/-- Systematic sampling probabilities -/
theorem systematic_sampling_probabilities
  (population : ℕ)
  (sample_size : ℕ)
  (removed : ℕ)
  (h_pop : population = 1005)
  (h_sample : sample_size = 50)
  (h_removed : removed = 5) :
  (removed : ℚ) / population = 5 / 1005 ∧
  (sample_size : ℚ) / population = 50 / 1005 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_probabilities_l97_9747


namespace NUMINAMATH_CALUDE_hania_age_in_5_years_l97_9738

-- Define the current year as a reference point
def current_year : ℕ := 2023

-- Define Samir's age in 5 years
def samir_age_in_5_years : ℕ := 20

-- Define the relationship between Samir's current age and Hania's age 10 years ago
axiom samir_hania_age_relation : 
  ∃ (samir_current_age hania_age_10_years_ago : ℕ),
    samir_current_age = samir_age_in_5_years - 5 ∧
    samir_current_age = hania_age_10_years_ago / 2

-- Theorem to prove
theorem hania_age_in_5_years : 
  ∃ (hania_current_age : ℕ),
    hania_current_age + 5 = 45 :=
sorry

end NUMINAMATH_CALUDE_hania_age_in_5_years_l97_9738


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l97_9772

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sequence_term (n : ℕ) : ℕ := factorial n + n

def sum_sequence (n : ℕ) : ℕ := 
  List.sum (List.map sequence_term (List.range n))

theorem units_digit_of_sum : 
  (sum_sequence 10) % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l97_9772


namespace NUMINAMATH_CALUDE_inequality_solution_set_l97_9727

def solution_set : Set ℝ := {x | x ≤ -3 ∨ x ≥ 0}

theorem inequality_solution_set :
  ∀ x : ℝ, x * (x + 3) ≥ 0 ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l97_9727


namespace NUMINAMATH_CALUDE_multiply_and_add_l97_9781

theorem multiply_and_add : 42 * 52 + 48 * 42 = 4200 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_l97_9781


namespace NUMINAMATH_CALUDE_auction_price_problem_l97_9718

theorem auction_price_problem (tv_initial_cost : ℝ) (tv_price_increase_ratio : ℝ) 
  (phone_price_increase_ratio : ℝ) (total_received : ℝ) :
  tv_initial_cost = 500 →
  tv_price_increase_ratio = 2 / 5 →
  phone_price_increase_ratio = 0.4 →
  total_received = 1260 →
  ∃ (phone_initial_price : ℝ),
    phone_initial_price = 400 ∧
    total_received = tv_initial_cost * (1 + tv_price_increase_ratio) + 
                     phone_initial_price * (1 + phone_price_increase_ratio) :=
by sorry

end NUMINAMATH_CALUDE_auction_price_problem_l97_9718


namespace NUMINAMATH_CALUDE_diameter_endpoint2_coordinates_l97_9771

def circle_center : ℝ × ℝ := (1, 2)
def diameter_endpoint1 : ℝ × ℝ := (4, 6)

theorem diameter_endpoint2_coordinates :
  let midpoint := circle_center
  let endpoint1 := diameter_endpoint1
  let endpoint2 := (2 * midpoint.1 - endpoint1.1, 2 * midpoint.2 - endpoint1.2)
  endpoint2 = (-2, -2) :=
sorry

end NUMINAMATH_CALUDE_diameter_endpoint2_coordinates_l97_9771


namespace NUMINAMATH_CALUDE_gasoline_price_decrease_l97_9764

theorem gasoline_price_decrease (a : ℝ) : 
  (∃ (initial_price final_price : ℝ), 
    initial_price = 8.1 ∧ 
    final_price = 7.8 ∧ 
    initial_price * (1 - a/100)^2 = final_price) → 
  8.1 * (1 - a/100)^2 = 7.8 :=
by sorry

end NUMINAMATH_CALUDE_gasoline_price_decrease_l97_9764


namespace NUMINAMATH_CALUDE_two_numbers_problem_l97_9760

theorem two_numbers_problem :
  ∃! (a b : ℕ), a > b ∧ (a / b : ℚ) * 6 = 10 ∧ (a - b : ℤ) + 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l97_9760


namespace NUMINAMATH_CALUDE_min_value_fraction_l97_9748

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 5) :
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 5 → 1/x + 4/(y+1) ≤ 1/a + 4/(b+1)) ∧
  (1/x + 4/(y+1) = 3/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l97_9748


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l97_9709

/-- Triangle ABC with specific properties -/
structure TriangleABC where
  -- Sides opposite to angles A, B, C
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles A, B, C
  A : ℝ
  B : ℝ
  C : ℝ
  -- Given conditions
  side_angle_relation : (2 * a - b) * Real.cos C = c * Real.cos B
  c_value : c = 2
  area : (1/2) * a * b * Real.sin C = Real.sqrt 3
  -- Triangle properties
  angle_sum : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

/-- Main theorem about the properties of Triangle ABC -/
theorem triangle_abc_properties (t : TriangleABC) : 
  t.C = Real.pi / 3 ∧ t.a + t.b + t.c = 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l97_9709


namespace NUMINAMATH_CALUDE_f_5_eq_neg_f_3_l97_9758

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def periodic_negative (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

def quadratic_on_interval (f : ℝ → ℝ) : Prop := ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_5_eq_neg_f_3 (f : ℝ → ℝ) 
  (h1 : is_odd f) 
  (h2 : periodic_negative f) 
  (h3 : quadratic_on_interval f) : 
  f 5 = -f 3 := by
  sorry

end NUMINAMATH_CALUDE_f_5_eq_neg_f_3_l97_9758


namespace NUMINAMATH_CALUDE_apples_left_l97_9785

def initial_apples : ℕ := 150
def sold_percentage : ℚ := 30 / 100
def given_percentage : ℚ := 20 / 100
def donated_apples : ℕ := 2

theorem apples_left : 
  let remaining_after_sale := initial_apples - (↑initial_apples * sold_percentage).floor
  let remaining_after_given := remaining_after_sale - (↑remaining_after_sale * given_percentage).floor
  remaining_after_given - donated_apples = 82 := by sorry

end NUMINAMATH_CALUDE_apples_left_l97_9785


namespace NUMINAMATH_CALUDE_longest_watching_time_l97_9759

structure Show where
  episodes : ℕ
  minutesPerEpisode : ℕ
  speed : ℚ

def watchingTimePerDay (s : Show) (days : ℕ) : ℚ :=
  (s.episodes * s.minutesPerEpisode : ℚ) / (s.speed * (days * 60))

theorem longest_watching_time (showA showB showC : Show) (days : ℕ) :
  showA.episodes = 20 ∧ 
  showA.minutesPerEpisode = 30 ∧ 
  showA.speed = 1.2 ∧
  showB.episodes = 25 ∧ 
  showB.minutesPerEpisode = 45 ∧ 
  showB.speed = 1 ∧
  showC.episodes = 30 ∧ 
  showC.minutesPerEpisode = 40 ∧ 
  showC.speed = 0.9 ∧
  days = 5 →
  watchingTimePerDay showC days > watchingTimePerDay showA days ∧
  watchingTimePerDay showC days > watchingTimePerDay showB days :=
sorry

end NUMINAMATH_CALUDE_longest_watching_time_l97_9759


namespace NUMINAMATH_CALUDE_sixty_marbles_l97_9717

/-- The number of marbles in a bag with specific color distributions --/
def total_marbles : ℕ → Prop :=
  fun n =>
    ∃ (yellow green red blue : ℕ),
      yellow = 20 ∧
      green = yellow / 2 ∧
      red = blue ∧
      blue = n / 4 ∧
      n = yellow + green + red + blue

/-- The theorem stating that there are 60 marbles in the bag --/
theorem sixty_marbles : total_marbles 60 := by
  sorry

#check sixty_marbles

end NUMINAMATH_CALUDE_sixty_marbles_l97_9717


namespace NUMINAMATH_CALUDE_unique_factorization_l97_9741

theorem unique_factorization (E F G H : ℕ+) : 
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H →
  E * F = 120 →
  G * H = 120 →
  E - F = G + H - 2 →
  E = 30 := by
sorry

end NUMINAMATH_CALUDE_unique_factorization_l97_9741


namespace NUMINAMATH_CALUDE_tree_shadow_length_l97_9726

/-- Given a tree and a flag pole, proves the length of the tree's shadow. -/
theorem tree_shadow_length 
  (tree_height : ℝ) 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (h1 : tree_height = 12)
  (h2 : flagpole_height = 150)
  (h3 : flagpole_shadow = 100) : 
  (tree_height * flagpole_shadow) / flagpole_height = 8 :=
by sorry

end NUMINAMATH_CALUDE_tree_shadow_length_l97_9726


namespace NUMINAMATH_CALUDE_triangle_properties_l97_9740

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.a = 3)
  (h2 : abc.c = 2)
  (h3 : Real.sin abc.A = Real.cos (π/2 - abc.B)) : 
  Real.cos abc.C = 7/9 ∧ 
  (1/2 * abc.a * abc.b * Real.sin abc.C) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l97_9740


namespace NUMINAMATH_CALUDE_cubic_function_property_l97_9762

/-- Given a cubic function f(x) = ax³ - bx + 1 where a and b are real numbers,
    prove that if f(-2) = -1, then f(2) = 3. -/
theorem cubic_function_property (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^3 - b * x + 1)
    (h2 : f (-2) = -1) : 
  f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l97_9762


namespace NUMINAMATH_CALUDE_part_one_part_two_l97_9787

-- Define the logarithmic function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem for part 1
theorem part_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 8 = 3) :
  a = 2 := by sorry

-- Theorem for part 2
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 → {x : ℝ | f a x ≤ f a (2 - 3*x)} = {x : ℝ | 0 < x ∧ x ≤ 1/2}) ∧
  (0 < a ∧ a < 1 → {x : ℝ | f a x ≤ f a (2 - 3*x)} = {x : ℝ | 1/2 ≤ x ∧ x < 2/3}) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l97_9787


namespace NUMINAMATH_CALUDE_triangle_side_length_l97_9714

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  Real.cos A = 3/5 ∧
  Real.sin B = Real.sqrt 5 / 5 ∧
  a = 2 →
  c = 11 * Real.sqrt 5 / 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l97_9714


namespace NUMINAMATH_CALUDE_smallest_multiple_l97_9704

theorem smallest_multiple (y : ℕ) : y = 32 ↔ 
  (y > 0 ∧ 
   900 * y % 1152 = 0 ∧ 
   ∀ z : ℕ, z > 0 → z < y → 900 * z % 1152 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l97_9704


namespace NUMINAMATH_CALUDE_second_player_wins_alice_wins_l97_9702

/-- Represents the frequency of each letter in the string -/
def LetterFrequency := Char → Nat

/-- The game state -/
structure GameState where
  frequencies : LetterFrequency
  playerTurn : Bool -- true for first player, false for second player

/-- Checks if all frequencies are even -/
def allEven (freq : LetterFrequency) : Prop :=
  ∀ c, Even (freq c)

/-- Represents a valid move in the game -/
inductive Move where
  | erase (c : Char) (n : Nat)

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.erase c n =>
      { frequencies := λ x => if x = c then state.frequencies x - n else state.frequencies x,
        playerTurn := ¬state.playerTurn }

/-- Checks if the game is over (all frequencies are zero) -/
def isGameOver (state : GameState) : Prop :=
  ∀ c, state.frequencies c = 0

/-- The winning strategy for the second player -/
def secondPlayerStrategy (state : GameState) : Move :=
  sorry -- Implementation not required for the statement

/-- The main theorem stating that the second player can always win -/
theorem second_player_wins (initialState : GameState) :
  ¬initialState.playerTurn →
  ∃ (strategy : GameState → Move),
    ∀ (moves : List Move),
      let finalState := (moves.foldl applyMove initialState)
      (isGameOver finalState ∧ ¬finalState.playerTurn) ∨
      (¬isGameOver finalState ∧ allEven finalState.frequencies) :=
  sorry

/-- The specific game instance from the problem -/
def initialGameState : GameState :=
  { frequencies := λ c =>
      if c = 'А' then 3
      else if c = 'О' then 3
      else if c = 'Д' then 2
      else if c = 'Я' then 2
      else if c ∈ ['Г', 'Р', 'С', 'К', 'У', 'Т', 'Н', 'Л', 'И', 'М', 'П'] then 1
      else 0,
    playerTurn := false }

/-- Theorem specific to the given problem instance -/
theorem alice_wins : 
  ∃ (strategy : GameState → Move),
    ∀ (moves : List Move),
      let finalState := (moves.foldl applyMove initialGameState)
      (isGameOver finalState ∧ ¬finalState.playerTurn) ∨
      (¬isGameOver finalState ∧ allEven finalState.frequencies) :=
  sorry

end NUMINAMATH_CALUDE_second_player_wins_alice_wins_l97_9702


namespace NUMINAMATH_CALUDE_triangle_area_l97_9779

/-- Given a triangle ABC with |AB| = 2, |AC| = 3, and AB · AC = -3, 
    prove that the area of triangle ABC is (3√3)/2 -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1^2 + AB.2^2 = 4) →
  (AC.1^2 + AC.2^2 = 9) →
  (AB.1 * AC.1 + AB.2 * AC.2 = -3) →
  (1/2 * Real.sqrt ((AB.1 * AC.2 - AB.2 * AC.1)^2) = (3 * Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l97_9779


namespace NUMINAMATH_CALUDE_triangle_properties_l97_9739

noncomputable section

/-- Triangle ABC with internal angles A, B, C opposite sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C ∧
  t.a * t.c * Real.cos t.B = -3

/-- The area of the triangle -/
def TriangleArea (t : Triangle) : ℝ :=
  (3 * Real.sqrt 3) / 2

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  TriangleArea t = (3 * Real.sqrt 3) / 2 ∧
  t.b ≥ Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l97_9739


namespace NUMINAMATH_CALUDE_pyramid_volume_l97_9765

theorem pyramid_volume (base_side : ℝ) (height : ℝ) (volume : ℝ) : 
  base_side = 1/3 → height = 1 → volume = (1/3) * (base_side^2) * height → volume = 1/27 := by
  sorry

#check pyramid_volume

end NUMINAMATH_CALUDE_pyramid_volume_l97_9765


namespace NUMINAMATH_CALUDE_extremum_values_l97_9755

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_values (a b : ℝ) :
  f_deriv a b 1 = 0 ∧ f a b 1 = 10 → a = 4 ∧ b = -11 := by
  sorry

end NUMINAMATH_CALUDE_extremum_values_l97_9755


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l97_9752

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sqrt (3 + 4 * Real.sqrt 6 - (16 * Real.sqrt 3 - 8 * Real.sqrt 2) * Real.sin x) = 4 * Real.sin x - Real.sqrt 3) ↔ 
  ∃ k : ℤ, x = (-1)^k * (π / 4) + 2 * k * π :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l97_9752


namespace NUMINAMATH_CALUDE_smallest_cube_sum_solution_l97_9773

/-- The smallest positive integer solution for the equation u³ + v³ + w³ = x³ -/
theorem smallest_cube_sum_solution :
  let P : ℕ → ℕ → ℕ → ℕ → Prop :=
    fun u v w x => u^3 + v^3 + w^3 = x^3 ∧ 
                   u < v ∧ v < w ∧ w < x ∧
                   v = u + 1 ∧ w = v + 1 ∧ x = w + 1
  ∀ u v w x, P u v w x → x ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_sum_solution_l97_9773


namespace NUMINAMATH_CALUDE_total_distance_four_runners_l97_9769

/-- The total distance run by four runners, where one runner ran 51 miles
    and the other three ran the same distance of 48 miles each, is 195 miles. -/
theorem total_distance_four_runners :
  ∀ (katarina tomas tyler harriet : ℕ),
    katarina = 51 →
    tomas = 48 →
    tyler = 48 →
    harriet = 48 →
    katarina + tomas + tyler + harriet = 195 :=
by
  sorry

end NUMINAMATH_CALUDE_total_distance_four_runners_l97_9769


namespace NUMINAMATH_CALUDE_collinear_vectors_l97_9723

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (e₁ e₂ : V)
variable (A B C : V)
variable (k : ℝ)

theorem collinear_vectors (h1 : e₁ ≠ 0 ∧ e₂ ≠ 0 ∧ ¬ ∃ (r : ℝ), e₁ = r • e₂)
  (h2 : B - A = 2 • e₁ + k • e₂)
  (h3 : C - B = e₁ - 3 • e₂)
  (h4 : ∃ (t : ℝ), C - A = t • (B - A)) :
  k = -6 := by sorry

end NUMINAMATH_CALUDE_collinear_vectors_l97_9723


namespace NUMINAMATH_CALUDE_rectangle_in_circle_area_l97_9700

theorem rectangle_in_circle_area (r : ℝ) (w h : ℝ) :
  r = 5 ∧ w = 6 ∧ h = 2 →
  w * h ≤ π * r^2 →
  w * h = 12 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_in_circle_area_l97_9700


namespace NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l97_9749

/-- The number of distinct arrangements of n beads on a necklace,
    considering rotational and reflectional symmetry -/
def necklace_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of distinct arrangements
    of 8 beads on a necklace is 2520 -/
theorem eight_bead_necklace_arrangements :
  necklace_arrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l97_9749


namespace NUMINAMATH_CALUDE_smallest_valid_number_l97_9729

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 10^11 ∧ n < 10^12) ∧  -- 12-digit number
  (n % 36 = 0) ∧             -- divisible by 36
  (∀ d : ℕ, d < 10 → ∃ k : ℕ, (n / 10^k) % 10 = d)  -- contains each digit 0-9

theorem smallest_valid_number :
  (is_valid_number 100023457896) ∧
  (∀ m : ℕ, m < 100023457896 → ¬(is_valid_number m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l97_9729


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l97_9763

def f (x : ℝ) := -x^2 + 2*x + 8

theorem f_increasing_on_interval :
  ∀ x y, x < y ∧ y ≤ 1 → f x < f y :=
by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l97_9763


namespace NUMINAMATH_CALUDE_number_problem_l97_9767

theorem number_problem (n : ℝ) : (40 / 100) * (3 / 5) * n = 36 → n = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l97_9767


namespace NUMINAMATH_CALUDE_angle4_is_70_l97_9750

-- Define the angles
variable (angle1 angle2 angle3 angle4 angle5 : ℝ)

-- Define the conditions
axiom angle1_plus_angle2 : angle1 + angle2 = 180
axiom angle4_eq_angle5 : angle4 = angle5
axiom triangle_sum : angle1 + angle3 + angle5 = 180
axiom angle1_value : angle1 = 50
axiom angle3_value : angle3 = 60

-- Theorem to prove
theorem angle4_is_70 : angle4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_angle4_is_70_l97_9750


namespace NUMINAMATH_CALUDE_rectangles_in_7x4_grid_l97_9707

/-- The number of rectangles in a grid -/
def num_rectangles (columns rows : ℕ) : ℕ :=
  (columns + 1).choose 2 * (rows + 1).choose 2

/-- Theorem: In a 7x4 grid, the number of rectangles is 280 -/
theorem rectangles_in_7x4_grid :
  num_rectangles 7 4 = 280 := by sorry

end NUMINAMATH_CALUDE_rectangles_in_7x4_grid_l97_9707


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l97_9732

theorem min_value_of_expression (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 5 * Real.sqrt 6 / 3 :=
by sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) = 5 * Real.sqrt 6 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l97_9732


namespace NUMINAMATH_CALUDE_triangle_value_l97_9722

def base_7_to_10 (a b : ℕ) : ℕ := a * 7 + b

def base_9_to_10 (a b : ℕ) : ℕ := a * 9 + b

theorem triangle_value :
  ∃! t : ℕ, t < 10 ∧ base_7_to_10 5 t = base_9_to_10 t 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_value_l97_9722


namespace NUMINAMATH_CALUDE_storage_area_wheels_l97_9721

theorem storage_area_wheels (bicycles tricycles unicycles cars : ℕ)
  (h_bicycles : bicycles = 24)
  (h_tricycles : tricycles = 14)
  (h_unicycles : unicycles = 10)
  (h_cars : cars = 18) :
  let total_wheels := bicycles * 2 + tricycles * 3 + unicycles * 1 + cars * 4
  let unicycle_wheels := unicycles * 1
  let ratio_numerator := unicycle_wheels
  let ratio_denominator := total_wheels
  (total_wheels = 172) ∧ 
  (ratio_numerator = 5 ∧ ratio_denominator = 86) :=
by sorry

end NUMINAMATH_CALUDE_storage_area_wheels_l97_9721


namespace NUMINAMATH_CALUDE_lotus_growth_model_l97_9708

def y (x : ℕ) : ℚ := (32 / 3) * (3 / 2) ^ x

theorem lotus_growth_model :
  (y 2 = 24) ∧ 
  (y 3 = 36) ∧ 
  (∀ n : ℕ, y n ≤ 10 * y 0 → n ≤ 5) ∧
  (y 6 > 10 * y 0) := by
  sorry

end NUMINAMATH_CALUDE_lotus_growth_model_l97_9708


namespace NUMINAMATH_CALUDE_min_value_S_l97_9735

theorem min_value_S (a b c : ℤ) (h1 : a + b + c = 2) 
  (h2 : (2*a + b*c)*(2*b + c*a)*(2*c + a*b) > 200) : 
  ∃ (m : ℤ), m = 256 ∧ 
  ∀ (x y z : ℤ), x + y + z = 2 → 
  (2*x + y*z)*(2*y + z*x)*(2*z + x*y) > 200 → 
  (2*x + y*z)*(2*y + z*x)*(2*z + x*y) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_S_l97_9735


namespace NUMINAMATH_CALUDE_selection_theorem_l97_9710

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of girls -/
def num_girls : ℕ := 5

/-- The total number of boys -/
def num_boys : ℕ := 7

/-- The number of representatives to be selected -/
def num_representatives : ℕ := 5

theorem selection_theorem :
  /- At least one girl is selected -/
  (choose (num_girls + num_boys) num_representatives - choose num_boys num_representatives = 771) ∧
  /- Boy A and Girl B are selected -/
  (choose (num_girls + num_boys - 2) (num_representatives - 2) = 120) ∧
  /- At least one of Boy A or Girl B is selected -/
  (choose (num_girls + num_boys) num_representatives - choose (num_girls + num_boys - 2) num_representatives = 540) :=
by sorry

end NUMINAMATH_CALUDE_selection_theorem_l97_9710


namespace NUMINAMATH_CALUDE_jelly_servings_count_jelly_servings_mixed_number_l97_9712

-- Define the total amount of jelly in tablespoons
def total_jelly : ℚ := 113 / 3

-- Define the serving size in tablespoons
def serving_size : ℚ := 3 / 2

-- Define the number of servings
def num_servings : ℚ := total_jelly / serving_size

-- Theorem to prove
theorem jelly_servings_count :
  num_servings = 226 / 9 := by sorry

-- Proof that the result is equivalent to 25 1/9
theorem jelly_servings_mixed_number :
  ∃ (n : ℕ) (m : ℚ), n = 25 ∧ m = 1 / 9 ∧ num_servings = n + m := by sorry

end NUMINAMATH_CALUDE_jelly_servings_count_jelly_servings_mixed_number_l97_9712


namespace NUMINAMATH_CALUDE_tan_half_product_squared_l97_9783

theorem tan_half_product_squared (a b : Real) 
  (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) : 
  (Real.tan (a / 2) * Real.tan (b / 2))^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_product_squared_l97_9783


namespace NUMINAMATH_CALUDE_guessing_game_solution_l97_9793

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem guessing_game_solution :
  ∃! n : ℕ,
    1 ≤ n ∧ n ≤ 99 ∧
    is_perfect_square n ∧
    ¬(n < 5) ∧
    (n < 7 ∨ n < 10 ∨ n ≥ 100) ∧
    n = 9 :=
by sorry

end NUMINAMATH_CALUDE_guessing_game_solution_l97_9793


namespace NUMINAMATH_CALUDE_solve_linear_equation_l97_9797

theorem solve_linear_equation (x : ℝ) : 2*x + 3*x + 4*x = 12 + 9 + 6 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l97_9797


namespace NUMINAMATH_CALUDE_nancy_mowing_time_l97_9756

/-- The time it takes Nancy to mow the yard alone -/
def nancy_time : ℝ := 3

/-- The time it takes Peter to mow the yard alone -/
def peter_time : ℝ := 4

/-- The time it takes Nancy and Peter to mow the yard together -/
def combined_time : ℝ := 1.71428571429

theorem nancy_mowing_time :
  1 / nancy_time + 1 / peter_time = 1 / combined_time := by
  sorry

end NUMINAMATH_CALUDE_nancy_mowing_time_l97_9756


namespace NUMINAMATH_CALUDE_min_value_on_line_equality_condition_l97_9703

theorem min_value_on_line (a b : ℝ) : 
  a > 0 → b > 0 → a + 2*b - 1 = 0 → 4/(a + b) + 1/b ≥ 9 :=
by sorry

theorem equality_condition (a b : ℝ) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 2*b - 1 = 0 ∧ 4/(a + b) + 1/b = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_line_equality_condition_l97_9703


namespace NUMINAMATH_CALUDE_sequence_sum_divisible_by_five_l97_9751

/-- Represents a four-digit integer -/
structure FourDigitInt where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : thousands ≥ 1 ∧ thousands ≤ 9 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Represents a sequence of four FourDigitInts with the given property -/
structure SpecialSequence where
  term1 : FourDigitInt
  term2 : FourDigitInt
  term3 : FourDigitInt
  term4 : FourDigitInt
  property : term2.hundreds = term1.tens ∧ term2.tens = term1.units ∧
             term3.hundreds = term2.tens ∧ term3.tens = term2.units ∧
             term4.hundreds = term3.tens ∧ term4.tens = term3.units ∧
             term1.hundreds = term4.tens ∧ term1.tens = term4.units

/-- Calculates the sum of all terms in the sequence -/
def sequenceSum (seq : SpecialSequence) : Nat :=
  let toNum (t : FourDigitInt) := t.thousands * 1000 + t.hundreds * 100 + t.tens * 10 + t.units
  toNum seq.term1 + toNum seq.term2 + toNum seq.term3 + toNum seq.term4

theorem sequence_sum_divisible_by_five (seq : SpecialSequence) :
  ∃ k : Nat, sequenceSum seq = 5 * k := by
  sorry


end NUMINAMATH_CALUDE_sequence_sum_divisible_by_five_l97_9751


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_l97_9780

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x ≤ 5}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

-- Theorem for (∁ₐA) ∩ B
theorem intersection_complement_A_B : (Aᶜ) ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_l97_9780


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l97_9715

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  totalStudents : Nat
  groupSize : Nat
  numGroups : Nat
  selectedNumber : Nat
  selectedGroup : Nat

/-- Calculates the number of the student selected from a target group. -/
def getSelectedNumber (s : SystematicSampling) (targetGroup : Nat) : Nat :=
  s.selectedNumber + (targetGroup - s.selectedGroup) * s.groupSize

/-- Theorem stating the result of the systematic sampling problem. -/
theorem systematic_sampling_result (s : SystematicSampling) 
  (h1 : s.totalStudents = 50)
  (h2 : s.groupSize = 5)
  (h3 : s.numGroups = 10)
  (h4 : s.selectedNumber = 12)
  (h5 : s.selectedGroup = 3) :
  getSelectedNumber s 8 = 37 := by
  sorry

#check systematic_sampling_result

end NUMINAMATH_CALUDE_systematic_sampling_result_l97_9715


namespace NUMINAMATH_CALUDE_initial_cats_l97_9776

theorem initial_cats (initial_cats final_cats bought_cats : ℕ) : 
  final_cats = initial_cats + bought_cats ∧ 
  final_cats = 54 ∧ 
  bought_cats = 43 →
  initial_cats = 11 := by sorry

end NUMINAMATH_CALUDE_initial_cats_l97_9776


namespace NUMINAMATH_CALUDE_unique_prime_seventh_power_l97_9737

theorem unique_prime_seventh_power (p : ℕ) : 
  Prime p ∧ ∃ q, Prime q ∧ p + 25 = q^7 ↔ p = 103 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_seventh_power_l97_9737


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l97_9754

theorem complex_modulus_problem (z : ℂ) : z = (2 - I) / (1 + I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l97_9754


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l97_9719

theorem weight_loss_challenge (W : ℝ) (x : ℝ) (h : x > 0) :
  W * (1 - x / 100 + 2 / 100) = W * (100 - 12.28) / 100 →
  x = 14.28 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l97_9719


namespace NUMINAMATH_CALUDE_perpendicular_condition_acute_angle_condition_l97_9795

/-- Given vectors in R² -/
def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

/-- Dot product of two 2D vectors -/
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (u v : Fin 2 → ℝ) : Prop := dot_product u v = 0

/-- The angle between two vectors is acute if their dot product is positive -/
def acute_angle (u v : Fin 2 → ℝ) : Prop := dot_product u v > 0

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (u v : Fin 2 → ℝ) : Prop := ∃ (k : ℝ), ∀ (i : Fin 2), u i = k * v i

theorem perpendicular_condition (x : ℝ) : 
  perpendicular (λ i => a i + 2 * b x i) (λ i => 2 * a i - b x i) ↔ x = -2 ∨ x = 7/2 := by
  sorry

theorem acute_angle_condition (x : ℝ) :
  acute_angle a (b x) ∧ ¬ parallel a (b x) ↔ x > -2 ∧ x ≠ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_condition_acute_angle_condition_l97_9795


namespace NUMINAMATH_CALUDE_number_of_cows_farm_cows_l97_9768

/-- The number of cows in a farm given their husk consumption -/
theorem number_of_cows (total_bags : ℕ) (total_days : ℕ) (cow_days : ℕ) : ℕ :=
  total_bags * cow_days / total_days

/-- Proof that there are 26 cows in the farm -/
theorem farm_cows : number_of_cows 26 26 26 = 26 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cows_farm_cows_l97_9768


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l97_9724

/-- The complex number z is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

/-- Given that (2-ai)/(1+i) is pure imaginary and a is real, prove that a = 2 -/
theorem complex_pure_imaginary_condition (a : ℝ) 
  (h : isPureImaginary ((2 - a * Complex.I) / (1 + Complex.I))) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l97_9724


namespace NUMINAMATH_CALUDE_square_diff_divided_by_three_l97_9786

theorem square_diff_divided_by_three : (123^2 - 120^2) / 3 = 243 := by sorry

end NUMINAMATH_CALUDE_square_diff_divided_by_three_l97_9786


namespace NUMINAMATH_CALUDE_cory_candy_money_needed_l97_9716

/-- The amount of money Cory needs to buy two packs of candies -/
theorem cory_candy_money_needed 
  (cory_initial_money : ℝ) 
  (candy_pack_cost : ℝ) 
  (number_of_packs : ℕ) 
  (h1 : cory_initial_money = 20) 
  (h2 : candy_pack_cost = 49) 
  (h3 : number_of_packs = 2) : 
  candy_pack_cost * number_of_packs - cory_initial_money = 78 := by
  sorry

#check cory_candy_money_needed

end NUMINAMATH_CALUDE_cory_candy_money_needed_l97_9716


namespace NUMINAMATH_CALUDE_wire_cutting_l97_9784

def wire_lengths : List Nat := [1008, 1260, 882, 1134]

theorem wire_cutting (segment_length : Nat) (total_segments : Nat) : 
  (∀ l ∈ wire_lengths, l % segment_length = 0) ∧
  (∀ d : Nat, (∀ l ∈ wire_lengths, l % d = 0) → d ≤ segment_length) ∧
  segment_length = 126 ∧
  total_segments = (List.sum (List.map (· / segment_length) wire_lengths)) ∧
  total_segments = 34 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l97_9784


namespace NUMINAMATH_CALUDE_minimize_y_l97_9745

/-- The function y in terms of x, a, b, and c -/
def y (x a b c : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + c * x

/-- The theorem stating that (a + b - c/2) / 2 minimizes y -/
theorem minimize_y (a b c : ℝ) :
  ∃ (x : ℝ), ∀ (z : ℝ), y z a b c ≥ y ((a + b - c/2) / 2) a b c :=
sorry

end NUMINAMATH_CALUDE_minimize_y_l97_9745


namespace NUMINAMATH_CALUDE_train_speed_l97_9746

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 2500) (h2 : time = 100) :
  length / time = 25 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l97_9746


namespace NUMINAMATH_CALUDE_game_results_l97_9766

/-- A game between two players A and B with specific winning conditions -/
structure Game where
  pA : ℝ  -- Probability of A winning a single game
  pB : ℝ  -- Probability of B winning a single game
  hpA : pA = 2/3
  hpB : pB = 1/3
  hprob : pA + pB = 1

/-- The number of games played when the match is decided -/
def num_games (g : Game) : ℕ → ℝ
  | 2 => g.pA^2 + g.pB^2
  | 3 => g.pB * g.pA^2 + g.pA * g.pB^2
  | 4 => g.pA * g.pB * g.pA^2 + g.pB * g.pA * g.pB^2
  | 5 => g.pB * g.pA * g.pB * g.pA + g.pA * g.pB * g.pA * g.pB
  | _ => 0

/-- The probability that B wins exactly one game and A wins the match -/
def prob_B_wins_one (g : Game) : ℝ :=
  g.pB * g.pA^2 + g.pA * g.pB * g.pA^2

/-- The expected number of games played -/
def expected_games (g : Game) : ℝ :=
  2 * (num_games g 2) + 3 * (num_games g 3) + 4 * (num_games g 4) + 5 * (num_games g 5)

theorem game_results (g : Game) :
  prob_B_wins_one g = 20/81 ∧
  num_games g 2 = 5/9 ∧
  num_games g 3 = 2/9 ∧
  num_games g 4 = 10/81 ∧
  num_games g 5 = 8/81 ∧
  expected_games g = 224/81 := by
  sorry

end NUMINAMATH_CALUDE_game_results_l97_9766


namespace NUMINAMATH_CALUDE_mistaken_addition_problem_l97_9742

/-- Given a two-digit number and conditions from the problem, prove it equals 49. -/
theorem mistaken_addition_problem (A B : ℕ) : 
  B = 9 →
  A * 10 + 6 + 253 = 299 →
  A * 10 + B = 49 :=
by sorry

end NUMINAMATH_CALUDE_mistaken_addition_problem_l97_9742


namespace NUMINAMATH_CALUDE_equation_solutions_l97_9790

theorem equation_solutions (x : ℝ) : 
  (x^3 + 2*x)^(1/5) = (x^5 - 2*x)^(1/3) ↔ x = 0 ∨ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l97_9790
