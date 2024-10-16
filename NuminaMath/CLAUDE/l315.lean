import Mathlib

namespace NUMINAMATH_CALUDE_correct_years_passed_l315_31523

def initial_ages : List Nat := [19, 34, 37, 42, 48]

def new_stem_leaf_plot : List (Nat × List Nat) := 
  [(1, []), (2, [5, 5]), (3, []), (4, [0, 3, 8]), (5, [4])]

def years_passed (initial : List Nat) (new_plot : List (Nat × List Nat)) : Nat :=
  sorry

theorem correct_years_passed :
  years_passed initial_ages new_stem_leaf_plot = 6 := by sorry

end NUMINAMATH_CALUDE_correct_years_passed_l315_31523


namespace NUMINAMATH_CALUDE_third_grade_trees_l315_31502

theorem third_grade_trees (second_grade_trees : ℕ) (third_grade_trees : ℕ) : 
  second_grade_trees = 15 →
  third_grade_trees < 3 * second_grade_trees →
  third_grade_trees = 42 →
  true :=
by sorry

end NUMINAMATH_CALUDE_third_grade_trees_l315_31502


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_diagonal_squared_l315_31525

/-- An isosceles trapezoid with bases a and b, lateral side c, and diagonal d -/
structure IsoscelesTrapezoid (a b c d : ℝ) : Prop where
  bases_positive : 0 < a ∧ 0 < b
  lateral_positive : 0 < c
  diagonal_positive : 0 < d
  is_isosceles : true  -- This is a placeholder for the isosceles property

/-- The diagonal of an isosceles trapezoid satisfies d^2 = ab + c^2 -/
theorem isosceles_trapezoid_diagonal_squared 
  (a b c d : ℝ) (trap : IsoscelesTrapezoid a b c d) : 
  d^2 = a * b + c^2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_diagonal_squared_l315_31525


namespace NUMINAMATH_CALUDE_f_properties_l315_31572

noncomputable section

variables (a b : ℝ)

def f (x : ℝ) := Real.log ((a^x) - (b^x))

theorem f_properties (h1 : a > 1) (h2 : b > 0) (h3 : b < 1) :
  -- 1. Domain of f is (0, +∞)
  (∀ x > 0, (a^x) - (b^x) > 0) ∧
  -- 2. f is strictly increasing on its domain
  (∀ x y, 0 < x ∧ x < y → f a b x < f a b y) ∧
  -- 3. f(x) > 0 for all x > 1 iff a - b ≥ 1
  (∀ x > 1, f a b x > 0) ↔ a - b ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_f_properties_l315_31572


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l315_31596

/-- Given a boat that travels 13 km/hr along a stream and 5 km/hr against the same stream,
    its speed in still water is 9 km/hr. -/
theorem boat_speed_in_still_water
  (speed_along_stream : ℝ)
  (speed_against_stream : ℝ)
  (h_along : speed_along_stream = 13)
  (h_against : speed_against_stream = 5) :
  (speed_along_stream + speed_against_stream) / 2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l315_31596


namespace NUMINAMATH_CALUDE_marta_book_count_l315_31589

/-- The number of books on Marta's shelf after all changes -/
def final_book_count (initial_books added_books removed_books birthday_multiplier : ℕ) : ℕ :=
  initial_books + added_books - removed_books + birthday_multiplier * initial_books

/-- Theorem stating the final number of books on Marta's shelf -/
theorem marta_book_count : final_book_count 38 10 5 3 = 157 := by
  sorry

#eval final_book_count 38 10 5 3

end NUMINAMATH_CALUDE_marta_book_count_l315_31589


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l315_31526

/-- Represents the total number of handshakes -/
def total_handshakes : ℕ := 281

/-- Calculates the number of handshakes between gymnasts given the total number of gymnasts -/
def gymnast_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the proposition that the coach's handshakes are minimized -/
def coach_handshakes_minimized (k : ℕ) : Prop :=
  ∃ (n : ℕ), 
    gymnast_handshakes n + k = total_handshakes ∧
    ∀ (m : ℕ), m > n → gymnast_handshakes m > total_handshakes

/-- The main theorem stating that the minimum number of coach's handshakes is 5 -/
theorem min_coach_handshakes : 
  ∃ (k : ℕ), k = 5 ∧ coach_handshakes_minimized k :=
sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l315_31526


namespace NUMINAMATH_CALUDE_probability_six_heads_ten_coins_l315_31533

def num_coins : ℕ := 10
def num_heads : ℕ := 6

theorem probability_six_heads_ten_coins :
  (Nat.choose num_coins num_heads : ℚ) / (2 ^ num_coins) = 210 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_six_heads_ten_coins_l315_31533


namespace NUMINAMATH_CALUDE_arithmetic_sequence_average_l315_31536

theorem arithmetic_sequence_average (a₁ aₙ d : ℚ) (n : ℕ) (h₁ : a₁ = 15) (h₂ : aₙ = 35) (h₃ : d = 1/4) 
  (h₄ : aₙ = a₁ + (n - 1) * d) : 
  (n * (a₁ + aₙ)) / (2 * n) = 25 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_average_l315_31536


namespace NUMINAMATH_CALUDE_probability_of_double_l315_31584

/-- A domino is a pair of integers -/
def Domino := ℕ × ℕ

/-- The set of all possible dominos with integers from 0 to 6 -/
def DominoSet : Set Domino :=
  {d | d.1 ≤ 6 ∧ d.2 ≤ 6 ∧ d.1 ≤ d.2}

/-- A double is a domino where both numbers are the same -/
def isDouble (d : Domino) : Prop := d.1 = d.2

/-- The number of elements in the DominoSet -/
def totalDominos : ℕ := Finset.card (Finset.filter (fun d => d.1 ≤ 6 ∧ d.2 ≤ 6 ∧ d.1 ≤ d.2) (Finset.product (Finset.range 7) (Finset.range 7)))

/-- The number of doubles in the DominoSet -/
def totalDoubles : ℕ := Finset.card (Finset.range 7)

theorem probability_of_double : (totalDoubles : ℚ) / totalDominos = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_double_l315_31584


namespace NUMINAMATH_CALUDE_circle_symmetry_l315_31563

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

/-- Definition of the symmetry line -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- Definition of symmetry between points with respect to the line -/
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = y₁ + 1 ∧ y₂ = x₁ - 1

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

/-- Theorem stating that C₂ is symmetric to C₁ with respect to the given line -/
theorem circle_symmetry :
  ∀ x y : ℝ, C₂ x y ↔ ∃ x₁ y₁ : ℝ, C₁ x₁ y₁ ∧ symmetric_points x₁ y₁ x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l315_31563


namespace NUMINAMATH_CALUDE_sum_first_ten_natural_numbers_l315_31509

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of the first 10 natural numbers is 55 -/
theorem sum_first_ten_natural_numbers : triangular_number 10 = 55 := by
  sorry

#eval triangular_number 10  -- This should output 55

end NUMINAMATH_CALUDE_sum_first_ten_natural_numbers_l315_31509


namespace NUMINAMATH_CALUDE_train_passing_time_l315_31516

/-- The time it takes for a train to pass a man running in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length > 0 →
  train_speed > man_speed →
  train_speed > 0 →
  man_speed ≥ 0 →
  ∃ (t : ℝ), t > 0 ∧ t < 22 ∧ t * (train_speed - man_speed) * (1000 / 3600) = train_length :=
by
  sorry

/-- Example with given values -/
example : 
  ∃ (t : ℝ), t > 0 ∧ t < 22 ∧ t * (68 - 8) * (1000 / 3600) = 350 :=
by
  apply train_passing_time 350 68 8
  · linarith
  · linarith
  · linarith
  · linarith

end NUMINAMATH_CALUDE_train_passing_time_l315_31516


namespace NUMINAMATH_CALUDE_platform_length_l315_31555

/-- Given a train of length 300 meters that takes 30 seconds to cross a platform
    and 18 seconds to cross a signal pole, the length of the platform is 200 meters. -/
theorem platform_length (train_length : ℝ) (platform_cross_time : ℝ) (pole_cross_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_cross_time = 30)
  (h3 : pole_cross_time = 18) :
  let platform_length := (train_length * platform_cross_time / pole_cross_time) - train_length
  platform_length = 200 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l315_31555


namespace NUMINAMATH_CALUDE_triangle_properties_l315_31535

/-- Given a triangle ABC with specific properties, prove its angle B and area. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  b * Real.cos C = (2 * a - c) * Real.cos B →
  b = Real.sqrt 7 →
  a + c = 4 →
  B = π / 3 ∧ 
  (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l315_31535


namespace NUMINAMATH_CALUDE_least_area_rectangle_l315_31598

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Theorem: The least possible area of a rectangle with perimeter 200 and area divisible by 10 is 900 -/
theorem least_area_rectangle :
  ∃ (r : Rectangle),
    perimeter r = 200 ∧
    area r % 10 = 0 ∧
    area r = 900 ∧
    ∀ (s : Rectangle),
      perimeter s = 200 →
      area s % 10 = 0 →
      area r ≤ area s :=
sorry

end NUMINAMATH_CALUDE_least_area_rectangle_l315_31598


namespace NUMINAMATH_CALUDE_class_president_election_l315_31549

theorem class_president_election (total_votes : ℕ) 
  (emily_votes : ℕ) (fiona_votes : ℕ) : 
  emily_votes = total_votes / 4 →
  fiona_votes = total_votes / 3 →
  emily_votes + fiona_votes = 77 →
  total_votes = 132 := by
sorry

end NUMINAMATH_CALUDE_class_president_election_l315_31549


namespace NUMINAMATH_CALUDE_prob_three_in_same_group_l315_31506

-- Define the total number of students
def total_students : ℕ := 800

-- Define the number of lunch groups
def num_groups : ℕ := 4

-- Define a function to calculate the size of each group
def group_size (total : ℕ) (groups : ℕ) : ℕ := total / groups

-- Define a function to calculate the probability of one student being in a specific group
def prob_in_group (groups : ℕ) : ℚ := 1 / groups

-- Theorem: The probability of three specific students being in the same group is 1/16
theorem prob_three_in_same_group :
  (prob_in_group num_groups) * (prob_in_group num_groups) = 1 / 16 := by
  sorry

-- This represents the probability of the second and third student being in the same group as the first

end NUMINAMATH_CALUDE_prob_three_in_same_group_l315_31506


namespace NUMINAMATH_CALUDE_age_difference_l315_31550

/-- Represents the ages of a mother and daughter pair -/
structure AgesPair where
  mother : ℕ
  daughter : ℕ

/-- Checks if the digits of the daughter's age are the reverse of the mother's age digits -/
def AgesPair.isReverse (ages : AgesPair) : Prop :=
  ages.daughter = ages.mother % 10 * 10 + ages.mother / 10

/-- Checks if in 13 years, the mother will be twice as old as the daughter -/
def AgesPair.futureCondition (ages : AgesPair) : Prop :=
  ages.mother + 13 = 2 * (ages.daughter + 13)

/-- The main theorem stating the age difference -/
theorem age_difference (ages : AgesPair) 
  (h1 : ages.isReverse)
  (h2 : ages.futureCondition) :
  ages.mother - ages.daughter = 40 := by
  sorry

/-- Example usage of the theorem -/
example : ∃ (ages : AgesPair), ages.isReverse ∧ ages.futureCondition ∧ ages.mother - ages.daughter = 40 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l315_31550


namespace NUMINAMATH_CALUDE_hyperbola_sum_l315_31568

/-- Given a hyperbola with center (1, -1), one focus at (1, 5), one vertex at (1, 2),
    and equation ((y-k)^2 / a^2) - ((x-h)^2 / b^2) = 1,
    prove that h + k + a + b = 3√3 + 3 -/
theorem hyperbola_sum (h k a b : ℝ) : 
  h = 1 ∧ k = -1 ∧  -- center at (1, -1)
  ∃ (x y : ℝ), x = 1 ∧ y = 5 ∧  -- one focus at (1, 5)
    (y - k)^2 = (x - h)^2 + a^2 ∧  -- relationship between focus, center, and a
  ∃ (x y : ℝ), x = 1 ∧ y = 2 ∧  -- one vertex at (1, 2)
    (y - k)^2 = a^2 ∧  -- relationship between vertex, center, and a
  ∀ (x y : ℝ), ((y - k)^2 / a^2) - ((x - h)^2 / b^2) = 1  -- equation of hyperbola
  →
  h + k + a + b = 3 * Real.sqrt 3 + 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l315_31568


namespace NUMINAMATH_CALUDE_xyz_sum_l315_31529

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = y * z + x)
  (h2 : y * z + x = x * z + y)
  (h3 : x * y + z = 47) : 
  x + y + z = 48 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l315_31529


namespace NUMINAMATH_CALUDE_construction_labor_cost_l315_31579

def worker_salary : ℕ := 100
def electrician_salary : ℕ := 2 * worker_salary
def plumber_salary : ℕ := (5 * worker_salary) / 2
def architect_salary : ℕ := (7 * worker_salary) / 2

def project_cost : ℕ := 2 * worker_salary + electrician_salary + plumber_salary + architect_salary

def total_cost : ℕ := 3 * project_cost

theorem construction_labor_cost : total_cost = 3000 := by
  sorry

end NUMINAMATH_CALUDE_construction_labor_cost_l315_31579


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l315_31560

-- Define the function f(x) = x³ - x² - x
def f (x : ℝ) := x^3 - x^2 - x

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1/3 : ℝ) 1, 
    ∀ y ∈ Set.Ioo (-1/3 : ℝ) 1, 
      x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l315_31560


namespace NUMINAMATH_CALUDE_student_pairs_count_l315_31553

def number_of_students : ℕ := 15

def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem student_pairs_count :
  choose number_of_students 2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_student_pairs_count_l315_31553


namespace NUMINAMATH_CALUDE_smallest_norm_u_l315_31524

theorem smallest_norm_u (u : ℝ × ℝ) (h : ‖u + (5, 2)‖ = 10) :
  ∃ (v : ℝ × ℝ), ‖v‖ = 10 - Real.sqrt 29 ∧ ∀ w : ℝ × ℝ, ‖w + (5, 2)‖ = 10 → ‖v‖ ≤ ‖w‖ := by
  sorry

end NUMINAMATH_CALUDE_smallest_norm_u_l315_31524


namespace NUMINAMATH_CALUDE_jose_play_time_l315_31512

/-- Calculates the total hours played given the minutes spent on football and basketball -/
def total_hours_played (football_minutes : ℕ) (basketball_minutes : ℕ) : ℚ :=
  (football_minutes + basketball_minutes : ℚ) / 60

/-- Proves that given Jose played football for 30 minutes and basketball for 60 minutes, 
    the total time he played is equal to 1.5 hours -/
theorem jose_play_time : total_hours_played 30 60 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_jose_play_time_l315_31512


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l315_31511

theorem min_value_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 4) : 
  (1 / x + 4 / y + 9 / z) ≥ 9 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 4 ∧ 1 / x + 4 / y + 9 / z = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l315_31511


namespace NUMINAMATH_CALUDE_college_sports_participation_l315_31542

/-- The total number of students who play at least one sport (cricket or basketball) -/
def total_students (cricket_players basketball_players both_players : ℕ) : ℕ :=
  cricket_players + basketball_players - both_players

/-- Theorem stating the total number of students playing at least one sport -/
theorem college_sports_participation : 
  total_students 500 600 220 = 880 := by
  sorry

end NUMINAMATH_CALUDE_college_sports_participation_l315_31542


namespace NUMINAMATH_CALUDE_sarahs_wallet_l315_31583

theorem sarahs_wallet (total_amount : ℕ) (total_bills : ℕ) (two_dollar_bills : ℕ) : 
  total_amount = 160 →
  total_bills = 120 →
  two_dollar_bills + (total_bills - two_dollar_bills) = total_bills →
  2 * two_dollar_bills + (total_bills - two_dollar_bills) = total_amount →
  two_dollar_bills = 40 := by
sorry

end NUMINAMATH_CALUDE_sarahs_wallet_l315_31583


namespace NUMINAMATH_CALUDE_ellipse_line_theorem_l315_31571

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  right_vertex : ℝ × ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: For an ellipse with given properties, if a line intersects a chord at its midpoint A(1, 1/2), then the line has equation x + 2y - 2 = 0 -/
theorem ellipse_line_theorem (e : Ellipse) (l : Line) :
  e.center = (0, 0) ∧
  e.left_focus = (-Real.sqrt 3, 0) ∧
  e.right_vertex = (2, 0) ∧
  (∃ (B C : ℝ × ℝ), B ≠ C ∧ (1, 1/2) = ((B.1 + C.1)/2, (B.2 + C.2)/2) ∧
    (∀ (x y : ℝ), x^2/4 + y^2 = 1 → l.a * x + l.b * y + l.c = 0 → (x, y) = B ∨ (x, y) = C)) →
  l.a = 1 ∧ l.b = 2 ∧ l.c = -2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_theorem_l315_31571


namespace NUMINAMATH_CALUDE_right_triangle_altitude_new_triangle_l315_31507

theorem right_triangle_altitude_new_triangle (a b c h : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : h > 0)
  (h5 : a^2 + b^2 = c^2)  -- Pythagorean theorem
  (h6 : a * b = c * h)    -- Area relationship
  : (c + h)^2 = (a + b)^2 + h^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_new_triangle_l315_31507


namespace NUMINAMATH_CALUDE_inverse_proportionality_l315_31565

theorem inverse_proportionality (X Y K : ℝ) (h1 : XY = K - 1) (h2 : K > 1) :
  ∃ c : ℝ, ∀ x y : ℝ, (x ≠ 0 ∧ y ≠ 0) → (X = x ∧ Y = y) → x * y = c :=
sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l315_31565


namespace NUMINAMATH_CALUDE_principal_mistake_l315_31587

theorem principal_mistake : ¬∃ (x y : ℕ), 2 * x = 2 * y + 11 := by
  sorry

end NUMINAMATH_CALUDE_principal_mistake_l315_31587


namespace NUMINAMATH_CALUDE_subcommittee_count_l315_31537

def total_members : ℕ := 12
def num_teachers : ℕ := 5
def subcommittee_size : ℕ := 5

def valid_subcommittees : ℕ := Nat.choose total_members subcommittee_size - Nat.choose (total_members - num_teachers) subcommittee_size

theorem subcommittee_count :
  valid_subcommittees = 771 :=
sorry

end NUMINAMATH_CALUDE_subcommittee_count_l315_31537


namespace NUMINAMATH_CALUDE_binomial_max_and_expectation_l315_31532

/-- The probability mass function for a binomial distribution with 20 trials and 2 successes -/
def f (p : ℝ) : ℝ := 190 * p^2 * (1 - p)^18

/-- The value of p that maximizes f(p) -/
def p₀ : ℝ := 0.1

/-- The number of items in a box -/
def box_size : ℕ := 200

/-- The number of items initially inspected -/
def initial_inspection : ℕ := 20

/-- The cost of inspecting one item -/
def inspection_cost : ℝ := 2

/-- The compensation fee for one defective item -/
def compensation_fee : ℝ := 25

/-- The expected number of defective items in the remaining items after initial inspection -/
def expected_defective : ℝ := 18

theorem binomial_max_and_expectation :
  (∀ p, p > 0 ∧ p < 1 → f p ≤ f p₀) ∧
  expected_defective = (box_size - initial_inspection : ℝ) * p₀ := by sorry

end NUMINAMATH_CALUDE_binomial_max_and_expectation_l315_31532


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l315_31541

theorem right_triangle_perimeter (area : ℝ) (leg : ℝ) (h_area : area = 180) (h_leg : leg = 30) :
  ∃ (perimeter : ℝ), perimeter = 42 + 2 * Real.sqrt 261 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l315_31541


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extremum_l315_31505

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem f_monotonicity_and_extremum :
  (∀ x, x > 0 → f x ≤ f (1 : ℝ)) ∧
  (∀ x y, 0 < x ∧ x < 1 ∧ 1 < y → f x > f 1 ∧ f y > f 1) ∧
  f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extremum_l315_31505


namespace NUMINAMATH_CALUDE_complex_square_on_negative_imaginary_axis_l315_31593

/-- A complex number z lies on the negative half of the imaginary axis if its real part is 0 and its imaginary part is negative -/
def lies_on_negative_imaginary_axis (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im < 0

theorem complex_square_on_negative_imaginary_axis (a : ℝ) :
  lies_on_negative_imaginary_axis ((a + Complex.I) ^ 2) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_on_negative_imaginary_axis_l315_31593


namespace NUMINAMATH_CALUDE_equation_solution_l315_31591

theorem equation_solution : ∃! x : ℚ, x - 3/4 = 5/12 - 1/3 ∧ x = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l315_31591


namespace NUMINAMATH_CALUDE_smallest_n_for_divisibility_l315_31531

/-- Given a positive odd number m, find the smallest natural number n 
    such that 2^1989 divides m^n - 1 -/
theorem smallest_n_for_divisibility (m : ℕ) (h_m_pos : 0 < m) (h_m_odd : Odd m) :
  ∃ (k : ℕ), ∃ (n : ℕ),
    (∀ (i : ℕ), i ≤ k → m % (2^i) = 1) ∧
    (m % (2^(k+1)) ≠ 1) ∧
    (n = 2^(1989 - k)) ∧
    (2^1989 ∣ m^n - 1) ∧
    (∀ (j : ℕ), j < n → ¬(2^1989 ∣ m^j - 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisibility_l315_31531


namespace NUMINAMATH_CALUDE_solution_set_of_f_x_plus_one_gt_zero_l315_31582

def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + (1 - x)) = f x

def monotone_decreasing_from_one (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f y < f x

theorem solution_set_of_f_x_plus_one_gt_zero
  (f : ℝ → ℝ)
  (h_sym : symmetric_about_one f)
  (h_mono : monotone_decreasing_from_one f)
  (h_f_zero : f 0 = 0) :
  {x : ℝ | f (x + 1) > 0} = Set.Ioo (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_f_x_plus_one_gt_zero_l315_31582


namespace NUMINAMATH_CALUDE_two_digit_primes_with_prime_digits_l315_31569

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def digits_are_prime (n : ℕ) : Prop :=
  is_prime (n / 10) ∧ is_prime (n % 10)

theorem two_digit_primes_with_prime_digits :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, is_two_digit n ∧ is_prime n ∧ digits_are_prime n) ∧
    (∀ n, is_two_digit n → is_prime n → digits_are_prime n → n ∈ s) ∧
    s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_two_digit_primes_with_prime_digits_l315_31569


namespace NUMINAMATH_CALUDE_range_of_m_l315_31557

theorem range_of_m (m : ℝ) : 
  let p := (1^2 + 1^2 - 2*m*1 + 2*m*1 + 2*m^2 - 4 < 0)
  let q := ∀ (x y : ℝ), m*x - y + 1 + 2*m = 0 → (x > 0 → y ≥ 0)
  (p ∨ q) ∧ ¬(p ∧ q) → ((-1 < m ∧ m < 0) ∨ m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l315_31557


namespace NUMINAMATH_CALUDE_max_b_value_l315_31538

-- Define a lattice point
def is_lattice_point (x y : ℤ) : Prop := true

-- Define the line equation
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 4

-- Define the condition for the line not passing through lattice points
def no_lattice_intersection (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 150 → is_lattice_point x y →
    line_equation m x ≠ y

-- State the theorem
theorem max_b_value :
  ∃ b : ℚ, b = 50/147 ∧
  (∀ m : ℚ, 1/3 < m ∧ m < b → no_lattice_intersection m) ∧
  (∀ b' : ℚ, b < b' →
    ∃ m : ℚ, 1/3 < m ∧ m < b' ∧ ¬(no_lattice_intersection m)) :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l315_31538


namespace NUMINAMATH_CALUDE_fourth_person_height_l315_31543

/-- Heights of four people in increasing order -/
def Heights := Fin 4 → ℕ

/-- The condition that heights are in increasing order -/
def increasing_heights (h : Heights) : Prop :=
  ∀ i j, i < j → h i < h j

/-- The condition for the differences between heights -/
def height_differences (h : Heights) : Prop :=
  h 1 - h 0 = 2 ∧ h 2 - h 1 = 2 ∧ h 3 - h 2 = 6

/-- The condition for the average height -/
def average_height (h : Heights) : Prop :=
  (h 0 + h 1 + h 2 + h 3) / 4 = 79

theorem fourth_person_height (h : Heights) 
  (inc : increasing_heights h) 
  (diff : height_differences h) 
  (avg : average_height h) : 
  h 3 = 85 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l315_31543


namespace NUMINAMATH_CALUDE_factorization_of_x4_plus_256_l315_31554

theorem factorization_of_x4_plus_256 (x : ℝ) : 
  x^4 + 256 = (x^2 - 8*x + 32) * (x^2 + 8*x + 32) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x4_plus_256_l315_31554


namespace NUMINAMATH_CALUDE_max_angle_at_C_l315_31547

/-- The line c given by the equation y = x + 1 -/
def line_c : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}

/-- Point A with coordinates (1, 0) -/
def point_A : ℝ × ℝ := (1, 0)

/-- Point B with coordinates (3, 0) -/
def point_B : ℝ × ℝ := (3, 0)

/-- Point C with coordinates (1, 2) -/
def point_C : ℝ × ℝ := (1, 2)

/-- The angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating that C maximizes the angle ACB -/
theorem max_angle_at_C :
  point_C ∈ line_c ∧
  ∀ p ∈ line_c, angle point_A p point_B ≤ angle point_A point_C point_B :=
by sorry

end NUMINAMATH_CALUDE_max_angle_at_C_l315_31547


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l315_31577

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (prod_sum_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c + 2*(a + b + c) = 672 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l315_31577


namespace NUMINAMATH_CALUDE_oblique_drawing_properties_l315_31573

-- Define the intuitive drawing using the oblique method
structure ObliqueDrawing where
  x_scale : ℝ
  y_scale : ℝ
  angle : ℝ

-- Define the properties of the oblique drawing
def is_valid_oblique_drawing (d : ObliqueDrawing) : Prop :=
  d.x_scale = 1 ∧ d.y_scale = 1/2 ∧ (d.angle = 135 ∨ d.angle = 45)

-- Theorem stating the properties of oblique drawing
theorem oblique_drawing_properties (d : ObliqueDrawing) 
  (h : is_valid_oblique_drawing d) : 
  d.x_scale = 1 ∧ 
  d.y_scale = 1/2 ∧ 
  (d.angle = 135 ∨ d.angle = 45) ∧ 
  ∃ (d' : ObliqueDrawing), is_valid_oblique_drawing d' ∧ d' ≠ d :=
sorry


end NUMINAMATH_CALUDE_oblique_drawing_properties_l315_31573


namespace NUMINAMATH_CALUDE_hall_dimensions_l315_31559

/-- Given a rectangular hall with width half of its length and area 800 sq. m,
    prove that the difference between length and width is 20 meters. -/
theorem hall_dimensions (length width : ℝ) : 
  width = length / 2 →
  length * width = 800 →
  length - width = 20 :=
by sorry

end NUMINAMATH_CALUDE_hall_dimensions_l315_31559


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l315_31567

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, x1 = 4 + 3 * Real.sqrt 2 ∧ x2 = 4 - 3 * Real.sqrt 2 ∧ 
    x1^2 - 8*x1 - 2 = 0 ∧ x2^2 - 8*x2 - 2 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = 3/2 ∧ x2 = -1 ∧ 
    2*x1^2 - x1 - 3 = 0 ∧ 2*x2^2 - x2 - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l315_31567


namespace NUMINAMATH_CALUDE_existence_of_h₁_h₂_l315_31518

theorem existence_of_h₁_h₂ :
  ∃ (h₁ h₂ : ℝ → ℝ),
    ∀ (g₁ g₂ : ℝ → ℝ) (x : ℝ),
      (∀ s, 1 ≤ g₁ s) →
      (∀ s, 1 ≤ g₂ s) →
      (∃ M, ∀ s, g₁ s ≤ M) →
      (∃ N, ∀ s, g₂ s ≤ N) →
      (⨆ s, (g₁ s) ^ x * g₂ s) = ⨆ t, x * h₁ t + h₂ t :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_h₁_h₂_l315_31518


namespace NUMINAMATH_CALUDE_amount_difference_l315_31548

def distribute_amount (total : ℝ) (p q r s t : ℝ) : Prop :=
  total = 25000 ∧
  p = 2 * q ∧
  s = 4 * r ∧
  q = r ∧
  p + q + r = (5/9) * total ∧
  s / (s + t) = 2/3 ∧
  s - p = 6944.4444

theorem amount_difference :
  ∀ (total p q r s t : ℝ),
  distribute_amount total p q r s t →
  s - p = 6944.4444 :=
by sorry

end NUMINAMATH_CALUDE_amount_difference_l315_31548


namespace NUMINAMATH_CALUDE_car_speed_problem_l315_31586

theorem car_speed_problem (v : ℝ) (h1 : v > 0) : 
  (60 / v - 60 / (v + 20) = 0.5) → v = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l315_31586


namespace NUMINAMATH_CALUDE_spending_on_games_l315_31501

theorem spending_on_games (total : ℚ) (movies burgers ice_cream music games : ℚ) : 
  total = 40 ∧ 
  movies = 1/4 ∧ 
  burgers = 1/8 ∧ 
  ice_cream = 1/5 ∧ 
  music = 1/4 ∧ 
  games = 3/20 ∧ 
  movies + burgers + ice_cream + music + games = 1 →
  total * games = 7 := by
sorry

end NUMINAMATH_CALUDE_spending_on_games_l315_31501


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_eq_three_fourths_l315_31562

/-- The equation (x - 3) / (ax - 2) = x has exactly one solution if and only if a = 3/4 -/
theorem unique_solution_iff_a_eq_three_fourths (a : ℝ) : 
  (∃! x : ℝ, (x - 3) / (a * x - 2) = x) ↔ a = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_eq_three_fourths_l315_31562


namespace NUMINAMATH_CALUDE_triangle_side_length_l315_31500

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (2 * x) + 2, Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem triangle_side_length 
  (A B C : ℝ) 
  (h1 : f A = 4) 
  (h2 : 0 < A ∧ A < π) 
  (h3 : Real.sin A * 1 / 2 = Real.sqrt 3 / 2) :
  ∃ (a : ℝ), a^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l315_31500


namespace NUMINAMATH_CALUDE_culprit_left_in_carriage_l315_31530

theorem culprit_left_in_carriage :
  -- Condition 1
  (∀ P W : Prop, P ∨ W) →
  -- Condition 2
  (∀ A P : Prop, A → P) →
  -- Condition 3
  (∀ A K : Prop, (¬A ∧ ¬K) ∨ (A ∧ K)) →
  -- Condition 4
  (∀ K : Prop, K) →
  -- Conclusion
  (∀ P : Prop, P) :=
by sorry

end NUMINAMATH_CALUDE_culprit_left_in_carriage_l315_31530


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l315_31561

theorem complex_power_magnitude : Complex.abs ((1 + Complex.I) ^ 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l315_31561


namespace NUMINAMATH_CALUDE_zoo_animals_l315_31519

theorem zoo_animals (M B L : ℕ) : 
  (26 ≤ M + B + L) ∧ (M + B + L ≤ 32) ∧
  (M + L > B) ∧
  (B + L = 2 * M) ∧
  (M + B = 3 * L + 3) ∧
  (2 * B = L) →
  B = 13 := by
sorry

end NUMINAMATH_CALUDE_zoo_animals_l315_31519


namespace NUMINAMATH_CALUDE_percentage_relation_l315_31540

theorem percentage_relation (x y c : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x = 2.5 * y) (h2 : 2 * y = (c / 100) * x) : c = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l315_31540


namespace NUMINAMATH_CALUDE_person_speed_in_mph_l315_31570

/-- Prove that a person crossing a 2500-meter street in 8 minutes has a speed of approximately 11.65 miles per hour. -/
theorem person_speed_in_mph : ∃ (speed : ℝ), abs (speed - 11.65) < 0.01 :=
  let street_length : ℝ := 2500 -- meters
  let crossing_time : ℝ := 8 -- minutes
  let meters_per_mile : ℝ := 1609.34
  let minutes_per_hour : ℝ := 60
  let distance_miles : ℝ := street_length / meters_per_mile
  let time_hours : ℝ := crossing_time / minutes_per_hour
  let speed : ℝ := distance_miles / time_hours
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_person_speed_in_mph_l315_31570


namespace NUMINAMATH_CALUDE_expression_evaluation_l315_31528

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 - z^2 + 2*x*y + 2*y*z = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l315_31528


namespace NUMINAMATH_CALUDE_number_divided_by_seven_l315_31585

theorem number_divided_by_seven (x : ℝ) : x / 7 = 5 / 14 → x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_seven_l315_31585


namespace NUMINAMATH_CALUDE_half_times_x_times_three_fourths_l315_31508

theorem half_times_x_times_three_fourths (x : ℚ) : x = 5/6 → (1/2 : ℚ) * x * (3/4 : ℚ) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_half_times_x_times_three_fourths_l315_31508


namespace NUMINAMATH_CALUDE_triangle_reconstruction_unique_l315_31514

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle on a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an acute triangle -/
structure AcuteTriangle where
  A : Point
  B : Point
  C : Point

/-- Represents the given information for triangle reconstruction -/
structure ReconstructionData where
  circumcircle : Circle
  C₀ : Point  -- Intersection of angle bisector from C with circumcircle
  A₁ : Point  -- Intersection of altitude from A with circumcircle
  B₁ : Point  -- Intersection of altitude from B with circumcircle

/-- Function to reconstruct the triangle from the given data -/
def reconstructTriangle (data : ReconstructionData) : AcuteTriangle :=
  sorry

/-- Theorem stating that the triangle can be uniquely reconstructed -/
theorem triangle_reconstruction_unique (data : ReconstructionData) :
  ∃! (triangle : AcuteTriangle),
    (Circle.center data.circumcircle).x ^ 2 + (Circle.center data.circumcircle).y ^ 2 = 
      data.circumcircle.radius ^ 2 ∧
    (data.C₀.x - triangle.C.x) ^ 2 + (data.C₀.y - triangle.C.y) ^ 2 = 
      data.circumcircle.radius ^ 2 ∧
    (data.A₁.x - triangle.A.x) ^ 2 + (data.A₁.y - triangle.A.y) ^ 2 = 
      data.circumcircle.radius ^ 2 ∧
    (data.B₁.x - triangle.B.x) ^ 2 + (data.B₁.y - triangle.B.y) ^ 2 = 
      data.circumcircle.radius ^ 2 :=
  sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_unique_l315_31514


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l315_31504

def is_valid_external_diagonals (d1 d2 d3 : ℝ) : Prop :=
  d1 > 0 ∧ d2 > 0 ∧ d3 > 0 ∧
  d1^2 + d2^2 > d3^2 ∧
  d1^2 + d3^2 > d2^2 ∧
  d2^2 + d3^2 > d1^2

theorem invalid_external_diagonals :
  ¬ (is_valid_external_diagonals 5 6 8) :=
by sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l315_31504


namespace NUMINAMATH_CALUDE_alice_chicken_amount_l315_31564

/-- Represents the grocery items in Alice's cart -/
structure GroceryCart where
  lettuce : ℕ
  cherryTomatoes : ℕ
  sweetPotatoes : ℕ
  broccoli : ℕ
  brusselSprouts : ℕ

/-- Calculates the total cost of items in the cart excluding chicken -/
def cartCost (cart : GroceryCart) : ℚ :=
  3 + 2.5 + (0.75 * cart.sweetPotatoes) + (2 * cart.broccoli) + 2.5

/-- Theorem: Alice has 1.5 pounds of chicken in her cart -/
theorem alice_chicken_amount (cart : GroceryCart) 
  (h1 : cart.lettuce = 1)
  (h2 : cart.cherryTomatoes = 1)
  (h3 : cart.sweetPotatoes = 4)
  (h4 : cart.broccoli = 2)
  (h5 : cart.brusselSprouts = 1)
  (h6 : 35 - (cartCost cart) - 11 = 6 * chicken_amount) :
  chicken_amount = 1.5 := by
  sorry

#check alice_chicken_amount

end NUMINAMATH_CALUDE_alice_chicken_amount_l315_31564


namespace NUMINAMATH_CALUDE_keith_bought_four_digimon_packs_l315_31521

/-- The number of Digimon card packs Keith bought -/
def num_digimon_packs : ℕ := 4

/-- The cost of each Digimon card pack in dollars -/
def digimon_pack_cost : ℝ := 4.45

/-- The cost of a deck of baseball cards in dollars -/
def baseball_deck_cost : ℝ := 6.06

/-- The total amount spent in dollars -/
def total_spent : ℝ := 23.86

/-- Theorem stating that Keith bought 4 packs of Digimon cards -/
theorem keith_bought_four_digimon_packs :
  (num_digimon_packs : ℝ) * digimon_pack_cost + baseball_deck_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_keith_bought_four_digimon_packs_l315_31521


namespace NUMINAMATH_CALUDE_partnership_investment_ratio_l315_31510

/-- Partnership investment problem -/
theorem partnership_investment_ratio :
  ∀ (x : ℝ) (m : ℝ),
  x > 0 →  -- A's investment is positive
  (12 * x) / (12 * x + 12 * x + 4 * m * x) = 1/3 →  -- A's share proportion
  m = 3 :=  -- Ratio of C's investment to A's
by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_ratio_l315_31510


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l315_31581

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 1) :
  Real.sqrt (((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2)) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l315_31581


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l315_31590

theorem shoe_price_calculation (num_shoes : ℕ) (num_shirts : ℕ) (shirt_price : ℚ) (total_earnings_per_person : ℚ) :
  num_shoes = 6 →
  num_shirts = 18 →
  shirt_price = 2 →
  total_earnings_per_person = 27 →
  ∃ (shoe_price : ℚ), 
    (num_shoes * shoe_price + num_shirts * shirt_price) / 2 = total_earnings_per_person ∧
    shoe_price = 3 :=
by sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l315_31590


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l315_31515

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid where
  base_side : ℝ
  lateral_face_equilateral : Bool

/-- A cube inscribed in a pyramid -/
structure InscribedCube where
  pyramid : Pyramid
  covers_base : Bool
  touches_summit : Bool

/-- The volume of the inscribed cube -/
def cube_volume (cube : InscribedCube) : ℝ := sorry

theorem inscribed_cube_volume 
  (cube : InscribedCube) 
  (h1 : cube.pyramid.base_side = 2) 
  (h2 : cube.pyramid.lateral_face_equilateral = true) 
  (h3 : cube.covers_base = true) 
  (h4 : cube.touches_summit = true) : 
  cube_volume cube = 8 := by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l315_31515


namespace NUMINAMATH_CALUDE_unique_prime_p_l315_31552

theorem unique_prime_p (p : ℕ) (hp : Nat.Prime p) (hp2 : Nat.Prime (5 * p^2 - 2)) : p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_p_l315_31552


namespace NUMINAMATH_CALUDE_polynomial_divisibility_implies_root_l315_31545

theorem polynomial_divisibility_implies_root (r : ℝ) : 
  (∃ (p : ℝ → ℝ), (∀ x, 9 * x^3 - 6 * x^2 - 48 * x + 54 = (x - r)^2 * p x)) → 
  r = 4/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_implies_root_l315_31545


namespace NUMINAMATH_CALUDE_pages_copied_l315_31592

def cost_per_page : ℚ := 3 / 100
def budget : ℚ := 15

theorem pages_copied (cost_per_page : ℚ) (budget : ℚ) :
  cost_per_page = 3 / 100 ∧ budget = 15 →
  budget / cost_per_page = 500 := by
sorry

end NUMINAMATH_CALUDE_pages_copied_l315_31592


namespace NUMINAMATH_CALUDE_jimmy_snow_shoveling_charge_l315_31551

/-- The amount Jimmy charges per driveway for snow shoveling -/
def jimmy_charge_per_driveway : ℝ := 1.50

theorem jimmy_snow_shoveling_charge :
  let candy_bar_price : ℝ := 0.75
  let candy_bar_count : ℕ := 2
  let lollipop_price : ℝ := 0.25
  let lollipop_count : ℕ := 4
  let driveways_shoveled : ℕ := 10
  let candy_store_spend : ℝ := candy_bar_price * candy_bar_count + lollipop_price * lollipop_count
  let snow_shoveling_earnings : ℝ := candy_store_spend * 6
  jimmy_charge_per_driveway = snow_shoveling_earnings / driveways_shoveled :=
by
  sorry

#check jimmy_snow_shoveling_charge

end NUMINAMATH_CALUDE_jimmy_snow_shoveling_charge_l315_31551


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l315_31566

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n, S n = n * (a 0 + a (n - 1)) / 2

/-- Theorem stating that if S_2 = 3 and S_4 = 15, then S_6 = 63 for an arithmetic sequence -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) 
    (h1 : seq.S 2 = 3) (h2 : seq.S 4 = 15) : seq.S 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l315_31566


namespace NUMINAMATH_CALUDE_dice_cube_volume_l315_31574

/-- The volume of a cube formed by stacking dice --/
theorem dice_cube_volume 
  (num_dice : ℕ) 
  (die_edge : ℝ) 
  (h1 : num_dice = 125) 
  (h2 : die_edge = 2) 
  (h3 : ∃ n : ℕ, n ^ 3 = num_dice) : 
  (die_edge * (num_dice : ℝ) ^ (1/3 : ℝ)) ^ 3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_dice_cube_volume_l315_31574


namespace NUMINAMATH_CALUDE_expansion_activity_optimal_time_l315_31534

theorem expansion_activity_optimal_time :
  ∀ (x y : ℕ),
    x + y = 15 →
    x = 2 * y - 3 →
    ∀ (m : ℕ),
      m ≤ 10 →
      10 - m > (m / 2) →
      6 * m + 8 * (10 - m) ≥ 68 :=
by
  sorry

end NUMINAMATH_CALUDE_expansion_activity_optimal_time_l315_31534


namespace NUMINAMATH_CALUDE_right_triangle_area_l315_31520

theorem right_triangle_area (a b c : ℝ) (h1 : a = 12) (h2 : c = 15) (h3 : a^2 + b^2 = c^2) : 
  (a * b) / 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l315_31520


namespace NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l315_31575

theorem parallelogram_altitude_base_ratio 
  (area : ℝ) (base : ℝ) (altitude : ℝ) 
  (h_area : area = 288) 
  (h_base : base = 12) 
  (h_area_formula : area = base * altitude) : 
  altitude / base = 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l315_31575


namespace NUMINAMATH_CALUDE_wall_width_is_three_l315_31578

/-- Proves that a rectangular wall with given proportions and volume has a width of 3 meters -/
theorem wall_width_is_three (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  l = 7 * h →
  volume = w * h * l →
  volume = 6804 →
  w = 3 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_is_three_l315_31578


namespace NUMINAMATH_CALUDE_function_inequality_implies_k_range_l315_31594

open Real

theorem function_inequality_implies_k_range (k : ℝ) : k > 0 → 
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → 
    (exp 2 * x₁ / exp x₁) / k ≤ (exp 2 * x₂^2 + 1) / (x₂ * (k + 1))) → 
  k ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_k_range_l315_31594


namespace NUMINAMATH_CALUDE_sheets_used_for_printing_james_sheets_used_l315_31556

/-- Calculate the number of sheets of paper used for printing books -/
theorem sheets_used_for_printing (num_books : ℕ) (pages_per_book : ℕ) 
  (pages_per_side : ℕ) (is_double_sided : Bool) : ℕ :=
  let total_pages := num_books * pages_per_book
  let pages_per_sheet := pages_per_side * (if is_double_sided then 2 else 1)
  total_pages / pages_per_sheet

/-- Prove that James uses 150 sheets of paper for printing his books -/
theorem james_sheets_used :
  sheets_used_for_printing 2 600 4 true = 150 := by
  sorry

end NUMINAMATH_CALUDE_sheets_used_for_printing_james_sheets_used_l315_31556


namespace NUMINAMATH_CALUDE_cleaning_payment_l315_31522

theorem cleaning_payment (rate : ℚ) (rooms : ℚ) : 
  rate = 12 / 3 → rooms = 9 / 4 → rate * rooms = 9 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_payment_l315_31522


namespace NUMINAMATH_CALUDE_square_sum_equals_sixteen_l315_31595

theorem square_sum_equals_sixteen (x : ℝ) : 
  (x - 1)^2 + 2*(x - 1)*(5 - x) + (5 - x)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_sixteen_l315_31595


namespace NUMINAMATH_CALUDE_triangle_shift_area_ratio_l315_31517

theorem triangle_shift_area_ratio (L α : ℝ) (h1 : 0 < α) (h2 : α < L) :
  let x := α / L
  (x * (2 * L^2 / 2) = (L^2 / 2 - (L - α)^2 / 2)) → x = (3 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_shift_area_ratio_l315_31517


namespace NUMINAMATH_CALUDE_arithmetic_progression_squares_l315_31599

theorem arithmetic_progression_squares (x : ℝ) : 
  ((x^2 - 2*x - 1)^2 + (x^2 + 2*x - 1)^2) / 2 = (x^2 + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_squares_l315_31599


namespace NUMINAMATH_CALUDE_triangle_radii_inequality_l315_31580

/-- Given a triangle ABC with circumradius R, inradius r, distance from circumcenter to centroid e,
    and distance from incenter to centroid f, prove that R² - e² ≥ 4(r² - f²),
    with equality if and only if the triangle is equilateral. -/
theorem triangle_radii_inequality (R r e f : ℝ) (hR : R > 0) (hr : r > 0) (he : e ≥ 0) (hf : f ≥ 0) :
  R^2 - e^2 ≥ 4*(r^2 - f^2) ∧
  (R^2 - e^2 = 4*(r^2 - f^2) ↔ ∃ (s : ℝ), R = s ∧ r = s/3 ∧ e = s/3 ∧ f = s/6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_radii_inequality_l315_31580


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l315_31576

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ
  black : ℕ
  yellow : ℕ
  blue : ℕ

/-- The probability of drawing a yellow marble as the second marble -/
def second_yellow_probability (bagA bagB bagC : Bag) : ℚ :=
  let total_A := bagA.white + bagA.black
  let total_B := bagB.yellow + bagB.blue
  let total_C := bagC.yellow + bagC.blue
  let prob_white_A : ℚ := bagA.white / total_A
  let prob_black_A : ℚ := bagA.black / total_A
  let prob_yellow_B : ℚ := bagB.yellow / total_B
  let prob_yellow_C : ℚ := bagC.yellow / total_C
  prob_white_A * prob_yellow_B + prob_black_A * prob_yellow_C

/-- The main theorem stating the probability of drawing a yellow marble as the second marble -/
theorem yellow_marble_probability :
  let bagA : Bag := ⟨3, 4, 0, 0⟩
  let bagB : Bag := ⟨0, 0, 6, 4⟩
  let bagC : Bag := ⟨0, 0, 2, 5⟩
  second_yellow_probability bagA bagB bagC = 103 / 245 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marble_probability_l315_31576


namespace NUMINAMATH_CALUDE_min_value_xy_l315_31558

theorem min_value_xy (x y : ℕ+) (h1 : x^2 + y^2 - 2017*(x:ℕ)*(y:ℕ) > 0) 
  (h2 : ∃ (z : ℕ), z^2 ≠ x^2 + y^2 - 2017*(x:ℕ)*(y:ℕ)) : 
  x^2 + y^2 - 2017*(x:ℕ)*(y:ℕ) ≥ 2019 := by
sorry

end NUMINAMATH_CALUDE_min_value_xy_l315_31558


namespace NUMINAMATH_CALUDE_max_triangles_correct_l315_31539

/-- The maximum number of triangles formed by drawing non-intersecting diagonals in a convex n-gon -/
def max_triangles (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 * n - 4 else 2 * n - 5

theorem max_triangles_correct (n : ℕ) (h : n ≥ 3) :
  max_triangles n = 
    (if n % 2 = 0 then 2 * n - 4 else 2 * n - 5) ∧
  ∀ k : ℕ, k ≤ max_triangles n :=
by sorry

end NUMINAMATH_CALUDE_max_triangles_correct_l315_31539


namespace NUMINAMATH_CALUDE_find_other_number_l315_31527

theorem find_other_number (x y : ℤ) : 
  ((x = 19 ∨ y = 19) ∧ 3 * x + 4 * y = 103) → 
  (x = 9 ∨ y = 9) := by
sorry

end NUMINAMATH_CALUDE_find_other_number_l315_31527


namespace NUMINAMATH_CALUDE_shelf_problem_solution_l315_31597

/-- Represents the thickness of a book type relative to an algebra book -/
structure BookThickness where
  thickness : ℚ
  pos : thickness > 0

/-- Represents the configuration of books on a shelf -/
structure ShelfConfiguration where
  algebra : ℕ
  geometry : ℕ
  history : ℕ

/-- Represents the problem setup -/
structure ShelfProblem where
  config1 : ShelfConfiguration
  config2 : ShelfConfiguration
  config3 : ShelfConfiguration
  geometry_thickness : BookThickness
  history_thickness : BookThickness
  distinct : config1.algebra ≠ config1.geometry ∧ config1.algebra ≠ config1.history ∧
             config1.geometry ≠ config1.history ∧ config2.algebra ≠ config2.geometry ∧
             config2.algebra ≠ config2.history ∧ config2.geometry ≠ config2.history ∧
             config3.algebra ≠ config1.algebra ∧ config3.algebra ≠ config1.geometry ∧
             config3.algebra ≠ config1.history ∧ config3.algebra ≠ config2.algebra ∧
             config3.algebra ≠ config2.geometry ∧ config3.algebra ≠ config2.history
  geometry_thicker : geometry_thickness.thickness > 1
  history_eq_geometry : history_thickness = geometry_thickness

theorem shelf_problem_solution (p : ShelfProblem) :
  p.config3.algebra = (p.config1.algebra * p.config2.geometry + p.config1.algebra * p.config2.history -
                       p.config2.algebra * p.config1.geometry - p.config2.algebra * p.config1.history) /
                      (p.config2.geometry + p.config2.history - p.config1.geometry - p.config1.history) :=
by sorry

end NUMINAMATH_CALUDE_shelf_problem_solution_l315_31597


namespace NUMINAMATH_CALUDE_all_propositions_true_l315_31546

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (lineparallel : Line → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (equidistant : Plane → Plane → Point → Prop)
variable (noncollinear : Point → Point → Point → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Define the theorem
theorem all_propositions_true :
  (∀ (n : Line) (α β : Plane), perpendicular n α → perpendicular n β → parallel α β) ∧
  (∀ (α β : Plane) (p q r : Point), noncollinear p q r → equidistant α β p → equidistant α β q → equidistant α β r → parallel α β) ∧
  (∀ (m n : Line) (α β : Plane), skew m n → contains α n → lineparallel n β → contains β m → lineparallel m α → parallel α β) :=
sorry

end NUMINAMATH_CALUDE_all_propositions_true_l315_31546


namespace NUMINAMATH_CALUDE_inequality_solution_set_l315_31503

theorem inequality_solution_set (x : ℝ) :
  x ≠ -7 →
  ((x^2 - 49) / (x + 7) < 0) ↔ (x < -7 ∨ (-7 < x ∧ x < 7)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l315_31503


namespace NUMINAMATH_CALUDE_smallest_number_l315_31513

theorem smallest_number (a b c d e : ℚ) : 
  a = 0.803 → b = 0.8003 → c = 0.8 → d = 0.8039 → e = 0.809 →
  c ≤ a ∧ c ≤ b ∧ c ≤ d ∧ c ≤ e := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l315_31513


namespace NUMINAMATH_CALUDE_triangle_properties_l315_31544

open Real

theorem triangle_properties (A B C a b c : Real) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π) (h5 : cos (2 * A) - 3 * cos (B + C) = 1) (h6 : a > 0 ∧ b > 0 ∧ c > 0) :
  -- Part 1
  A = π / 3 ∧
  -- Part 2
  (∃ S : Real, S = 5 * Real.sqrt 3 ∧ b = 5 → sin B * sin C = 5 / 7) ∧
  -- Part 3
  (a = 1 → ∃ l : Real, l = a + b + c ∧ 2 < l ∧ l ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l315_31544


namespace NUMINAMATH_CALUDE_pencil_grouping_l315_31588

theorem pencil_grouping (total_pencils : ℕ) (num_groups : ℕ) (pencils_per_group : ℕ) :
  total_pencils = 25 →
  num_groups = 5 →
  total_pencils = num_groups * pencils_per_group →
  pencils_per_group = 5 :=
by sorry

end NUMINAMATH_CALUDE_pencil_grouping_l315_31588
