import Mathlib

namespace sum_of_squares_of_roots_l948_94833

theorem sum_of_squares_of_roots (a b : ℝ) 
  (ha : a^2 - 6*a + 4 = 0) 
  (hb : b^2 - 6*b + 4 = 0) 
  (hab : a ≠ b) : 
  a^2 + b^2 = 28 := by
sorry

end sum_of_squares_of_roots_l948_94833


namespace complex_equality_l948_94853

/-- Given a complex number z = 1-ni, prove that m+ni = 2-i -/
theorem complex_equality (m n : ℝ) (z : ℂ) (h : z = 1 - n * Complex.I) :
  m + n * Complex.I = 2 - Complex.I := by sorry

end complex_equality_l948_94853


namespace midpoint_calculation_l948_94867

/-- Given two points A and B in a 2D plane, proves that 3x - 5y = -18,
    where (x, y) is the midpoint of AB. -/
theorem midpoint_calculation (A B : ℝ × ℝ) (h1 : A = (-8, 15)) (h2 : B = (16, -3)) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  3 * C.1 - 5 * C.2 = -18 := by
sorry

end midpoint_calculation_l948_94867


namespace max_reflections_before_target_angle_max_reflections_is_optimal_l948_94868

/-- The angle between the two reflecting lines in degrees -/
def angle_between_lines : ℝ := 5

/-- The target angle of incidence in degrees -/
def target_angle : ℝ := 85

/-- The maximum number of reflections -/
def max_reflections : ℕ := 17

theorem max_reflections_before_target_angle :
  ∀ n : ℕ, n * angle_between_lines ≤ target_angle ↔ n ≤ max_reflections :=
by sorry

theorem max_reflections_is_optimal :
  (max_reflections + 1) * angle_between_lines > target_angle :=
by sorry

end max_reflections_before_target_angle_max_reflections_is_optimal_l948_94868


namespace range_of_m_l948_94884

-- Define the conditions
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, p x → q x m) ∧  -- p is sufficient for q
  (∃ x, q x m ∧ ¬p x) ∧ -- p is not necessary for q
  (m > 0) →             -- given condition
  m ≥ 9 :=               -- conclusion
by sorry

end range_of_m_l948_94884


namespace hyperbola_equation_l948_94864

def is_hyperbola (a b : ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ t, (x t)^2 / a^2 - (y t)^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

def has_focus_at (x y : ℝ → ℝ) (fx fy : ℝ) : Prop :=
  ∃ t, x t = fx ∧ y t = fy

def has_asymptotes (x y : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ t, y t = m * x t ∨ y t = -m * x t

theorem hyperbola_equation (a b : ℝ) (x y : ℝ → ℝ) :
  is_hyperbola a b x y →
  has_focus_at x y 5 0 →
  has_asymptotes x y (3/4) →
  a^2 = 16 ∧ b^2 = 9 := by sorry

end hyperbola_equation_l948_94864


namespace decomposable_exponential_linear_cos_decomposable_l948_94804

-- Define a decomposable function
def Decomposable (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

-- Theorem for part 2
theorem decomposable_exponential_linear (b : ℝ) :
  Decomposable (λ x => 2*x + b + 2^x) → b > -2 :=
by sorry

-- Theorem for part 3
theorem cos_decomposable :
  Decomposable cos :=
by sorry

end decomposable_exponential_linear_cos_decomposable_l948_94804


namespace program_output_correct_l948_94818

def program_execution (initial_A initial_B : ℤ) : ℤ × ℤ × ℤ :=
  let A₁ := if initial_A < 0 then -initial_A else initial_A
  let B₁ := initial_B ^ 2
  let A₂ := A₁ + B₁
  let C  := A₂ - 2 * B₁
  let A₃ := A₂ / C
  let B₂ := B₁ * C + 1
  (A₃, B₂, C)

theorem program_output_correct :
  program_execution (-6) 2 = (5, 9, 2) := by
  sorry

end program_output_correct_l948_94818


namespace exponential_decreasing_range_l948_94876

/-- Given a monotonically decreasing exponential function f(x) = a^x on ℝ,
    prove that when f(x+1) ≥ 1, the range of x is (-∞, -1]. -/
theorem exponential_decreasing_range (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = a^x) :
  (∀ x y, x < y → f x > f y) →
  {x : ℝ | f (x + 1) ≥ 1} = Set.Iic (-1) := by
sorry

end exponential_decreasing_range_l948_94876


namespace xander_miles_more_l948_94896

/-- The problem statement and conditions --/
theorem xander_miles_more (t s : ℝ) 
  (h1 : t > 0) 
  (h2 : s > 0) 
  (h3 : s * t + 100 = (s + 10) * (t + 1.5)) : 
  (s + 15) * (t + 3) - s * t = 215 := by
  sorry

end xander_miles_more_l948_94896


namespace gas_cost_proof_l948_94866

/-- The original total cost of gas for a group of friends -/
def original_cost : ℝ := 200

/-- The number of friends initially -/
def initial_friends : ℕ := 5

/-- The number of additional friends who joined -/
def additional_friends : ℕ := 3

/-- The decrease in cost per person for the original friends -/
def cost_decrease : ℝ := 15

theorem gas_cost_proof :
  let total_friends := initial_friends + additional_friends
  let initial_cost_per_person := original_cost / initial_friends
  let final_cost_per_person := original_cost / total_friends
  initial_cost_per_person - final_cost_per_person = cost_decrease :=
by sorry

end gas_cost_proof_l948_94866


namespace tom_new_books_l948_94857

/-- Calculates the number of new books Tom bought given his initial, sold, and final book counts. -/
def new_books (initial : ℕ) (sold : ℕ) (final : ℕ) : ℕ :=
  final - (initial - sold)

/-- Proves that Tom bought 38 new books given the problem conditions. -/
theorem tom_new_books : new_books 5 4 39 = 38 := by
  sorry

end tom_new_books_l948_94857


namespace max_x_minus_y_l948_94881

theorem max_x_minus_y (x y : Real) (h1 : 0 < y) (h2 : y ≤ x) (h3 : x < π/2) (h4 : Real.tan x = 3 * Real.tan y) :
  ∃ (max_val : Real), max_val = π/6 ∧ x - y ≤ max_val ∧ ∃ (x' y' : Real), 0 < y' ∧ y' ≤ x' ∧ x' < π/2 ∧ Real.tan x' = 3 * Real.tan y' ∧ x' - y' = max_val :=
sorry

end max_x_minus_y_l948_94881


namespace diophantine_equation_solution_l948_94845

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+,
    (x * y * z + 2 * x + 3 * y + 6 * z = x * y + 2 * x * z + 3 * y * z) →
    (x = 4 ∧ y = 3 ∧ z = 1) :=
by sorry

end diophantine_equation_solution_l948_94845


namespace x_congruence_l948_94891

theorem x_congruence (x : ℤ) 
  (h1 : (2 + x) % 4 = 3 % 4)
  (h2 : (4 + x) % 16 = 8 % 16)
  (h3 : (6 + x) % 36 = 7 % 36) :
  x % 48 = 1 % 48 := by
sorry

end x_congruence_l948_94891


namespace p_true_q_false_l948_94831

theorem p_true_q_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : p ∧ ¬q :=
by
  sorry

end p_true_q_false_l948_94831


namespace charles_total_money_l948_94893

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The number of pennies Charles found on his way to school -/
def pennies_found : ℕ := 6

/-- The number of nickels Charles found on his way to school -/
def nickels_found : ℕ := 8

/-- The number of dimes Charles found on his way to school -/
def dimes_found : ℕ := 6

/-- The number of quarters Charles found on his way to school -/
def quarters_found : ℕ := 5

/-- The number of nickels Charles had at home -/
def nickels_at_home : ℕ := 3

/-- The number of dimes Charles had at home -/
def dimes_at_home : ℕ := 12

/-- The number of quarters Charles had at home -/
def quarters_at_home : ℕ := 7

/-- The number of half-dollars Charles had at home -/
def half_dollars_at_home : ℕ := 2

/-- The total amount of money Charles has -/
def total_money : ℚ :=
  penny_value * pennies_found +
  nickel_value * (nickels_found + nickels_at_home) +
  dime_value * (dimes_found + dimes_at_home) +
  quarter_value * (quarters_found + quarters_at_home) +
  half_dollar_value * half_dollars_at_home

theorem charles_total_money :
  total_money = 6.41 := by sorry

end charles_total_money_l948_94893


namespace pure_imaginary_equation_l948_94849

theorem pure_imaginary_equation (z : ℂ) (b : ℝ) : 
  (∃ (a : ℝ), z = a * Complex.I) → 
  (2 - Complex.I) * z = 4 - b * Complex.I → 
  b = -8 := by
sorry

end pure_imaginary_equation_l948_94849


namespace count_divisors_eq_twelve_l948_94851

/-- The number of natural numbers m such that 2023 ≡ 23 (mod m) -/
def count_divisors : ℕ :=
  (Finset.filter (fun m => m > 23 ∧ 2023 % m = 23) (Finset.range 2024)).card

theorem count_divisors_eq_twelve : count_divisors = 12 := by
  sorry

end count_divisors_eq_twelve_l948_94851


namespace cone_volume_divided_by_pi_l948_94843

/-- The volume of a cone formed from a 270-degree sector of a circle with radius 20, when divided by π, is equal to 1125√7. -/
theorem cone_volume_divided_by_pi (r l : Real) (h : Real) : 
  r = 15 → l = 20 → h = 5 * Real.sqrt 7 → (1/3 * π * r^2 * h) / π = 1125 * Real.sqrt 7 := by
  sorry

end cone_volume_divided_by_pi_l948_94843


namespace license_plate_theorem_l948_94890

def alphabet_size : ℕ := 25  -- Excluding 'A'
def letter_positions : ℕ := 4
def digit_positions : ℕ := 2
def total_digits : ℕ := 10

-- Define the function to calculate the number of license plate combinations
def license_plate_combinations : ℕ :=
  (alphabet_size.choose 2) *  -- Choose 2 letters from 25
  (letter_positions.choose 2) *  -- Choose 2 positions for one letter
  (total_digits) *  -- Choose first digit
  (total_digits - 1)  -- Choose second digit

-- Theorem statement
theorem license_plate_theorem :
  license_plate_combinations = 162000 := by
  sorry

end license_plate_theorem_l948_94890


namespace theater_seats_tom_wants_500_seats_l948_94808

/-- Calculates the number of seats in Tom's theater based on given conditions --/
theorem theater_seats (cost_per_sqft : ℝ) (sqft_per_seat : ℝ) (partner_share : ℝ) (tom_spend : ℝ) : ℝ :=
  let cost_per_seat := cost_per_sqft * sqft_per_seat
  let total_cost := tom_spend / (1 - partner_share)
  total_cost / (3 * cost_per_seat)

/-- Proves that Tom wants 500 seats in his theater --/
theorem tom_wants_500_seats :
  theater_seats 5 12 0.4 54000 = 500 := by
  sorry

end theater_seats_tom_wants_500_seats_l948_94808


namespace workshop_workers_l948_94889

/-- The total number of workers in a workshop with specific salary conditions -/
theorem workshop_workers (average_salary : ℕ) (technician_count : ℕ) (technician_salary : ℕ) (other_salary : ℕ) :
  average_salary = 8000 →
  technician_count = 7 →
  technician_salary = 20000 →
  other_salary = 6000 →
  ∃ (total_workers : ℕ),
    total_workers = technician_count + (technician_count * technician_salary + (total_workers - technician_count) * other_salary) / average_salary ∧
    total_workers = 49 :=
by sorry

end workshop_workers_l948_94889


namespace sleep_variance_proof_l948_94886

def sleep_data : List ℝ := [6, 6, 7, 6, 7, 8, 9]

theorem sleep_variance_proof :
  let n : ℕ := sleep_data.length
  let mean : ℝ := (sleep_data.sum) / n
  let variance : ℝ := (sleep_data.map (λ x => (x - mean)^2)).sum / n
  mean = 7 → variance = 8/7 := by
  sorry

end sleep_variance_proof_l948_94886


namespace jebbs_take_home_pay_l948_94855

/-- Calculates the take-home pay after tax -/
def takeHomePay (originalPay : ℝ) (taxRate : ℝ) : ℝ :=
  originalPay * (1 - taxRate)

/-- Proves that Jebb's take-home pay is $585 -/
theorem jebbs_take_home_pay :
  takeHomePay 650 0.1 = 585 := by
  sorry

end jebbs_take_home_pay_l948_94855


namespace new_average_weight_l948_94806

theorem new_average_weight (initial_count : ℕ) (initial_avg : ℚ) (new_weight : ℚ) :
  initial_count = 19 →
  initial_avg = 15 →
  new_weight = 13 →
  let total_weight := initial_count * initial_avg + new_weight
  let new_count := initial_count + 1
  new_count * (total_weight / new_count) = 298 :=
by
  sorry

end new_average_weight_l948_94806


namespace money_problem_l948_94869

theorem money_problem (a b : ℚ) : 
  a = 80/7 ∧ b = 40/7 →
  7*a + b < 100 ∧ 4*a - b = 40 ∧ b = (1/2) * a := by
  sorry

end money_problem_l948_94869


namespace intersection_of_solutions_range_of_a_l948_94800

-- Define the conditions
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) : Prop := -x^2 + 5*x - 6 ≥ 0

-- Part 1: Intersection of solutions when a = 1
theorem intersection_of_solutions :
  {x : ℝ | p x 1 ∧ q x} = {x : ℝ | 2 ≤ x ∧ x < 3} :=
sorry

-- Part 2: Range of a for which ¬p ↔ ¬q
theorem range_of_a :
  {a : ℝ | a > 0 ∧ ∀ x, ¬(p x a) ↔ ¬(q x)} = {a : ℝ | 1 < a ∧ a < 2} :=
sorry

end intersection_of_solutions_range_of_a_l948_94800


namespace fraction_simplification_l948_94814

theorem fraction_simplification (x : ℝ) (hx : x ≠ 0) :
  (42 * x^3) / (63 * x^5) = 2 / (3 * x^2) := by
  sorry

end fraction_simplification_l948_94814


namespace natural_number_pairs_l948_94802

theorem natural_number_pairs (a b : ℕ) :
  (∃ k : ℕ, b - 1 = k * (a + 1)) →
  (∃ m : ℕ, a^2 + a + 2 = m * b) →
  ∃ k : ℕ, a = 2 * k ∧ b = 2 * k^2 + 2 * k + 1 :=
by sorry

end natural_number_pairs_l948_94802


namespace quadratic_equation_integer_solutions_l948_94832

theorem quadratic_equation_integer_solutions (k : ℤ) : 
  (∃ x : ℤ, x > 0 ∧ (k^2 - 1) * x^2 - 6 * (3 * k - 1) * x + 72 = 0) ↔ 
  k = 1 ∨ k = 2 ∨ k = 3 := by
  sorry

end quadratic_equation_integer_solutions_l948_94832


namespace store_coloring_books_l948_94820

/-- The number of coloring books sold during the sale -/
def books_sold : ℕ := 39

/-- The number of shelves used after the sale -/
def shelves : ℕ := 9

/-- The number of books on each shelf after the sale -/
def books_per_shelf : ℕ := 9

/-- The initial number of coloring books in stock -/
def initial_stock : ℕ := books_sold + shelves * books_per_shelf

theorem store_coloring_books : initial_stock = 120 := by
  sorry

end store_coloring_books_l948_94820


namespace seventh_triangular_number_l948_94854

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The seventh triangular number is 28 -/
theorem seventh_triangular_number : triangular_number 7 = 28 := by sorry

end seventh_triangular_number_l948_94854


namespace quadratic_factorization_l948_94828

theorem quadratic_factorization (b : ℤ) : 
  (∃ (c d e f : ℤ), 15 * x^2 + b * x + 45 = (c * x + d) * (e * x + f)) → 
  ∃ (k : ℤ), b = 2 * k :=
sorry

end quadratic_factorization_l948_94828


namespace sin_odd_function_phi_l948_94822

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem sin_odd_function_phi (φ : ℝ) :
  is_odd_function (λ x => Real.sin (x + φ)) → φ = π :=
sorry

end sin_odd_function_phi_l948_94822


namespace quadratic_two_roots_l948_94863

theorem quadratic_two_roots (m : ℝ) (h : m < (1 : ℝ) / 4) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ + m = 0 ∧ x₂^2 - x₂ + m = 0 := by
  sorry

end quadratic_two_roots_l948_94863


namespace anne_travel_distance_l948_94879

/-- The distance traveled given time and speed -/
def distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Theorem: Anne's travel distance -/
theorem anne_travel_distance :
  let time : ℝ := 3
  let speed : ℝ := 2
  distance time speed = 6 := by sorry

end anne_travel_distance_l948_94879


namespace sqrt_three_expression_l948_94801

theorem sqrt_three_expression : 
  (Real.sqrt 3 + 2)^2023 * (Real.sqrt 3 - 2)^2024 = -Real.sqrt 3 + 2 := by
  sorry

end sqrt_three_expression_l948_94801


namespace triangle_side_length_l948_94899

theorem triangle_side_length (A B C : ℝ) (AC AB BC : ℝ) (angle_A : ℝ) :
  AC = Real.sqrt 2 →
  AB = 2 →
  (Real.sqrt 3 * Real.sin angle_A + Real.cos angle_A) / (Real.sqrt 3 * Real.cos angle_A - Real.sin angle_A) = Real.tan (5 * Real.pi / 12) →
  BC = Real.sqrt 2 :=
by sorry

end triangle_side_length_l948_94899


namespace remainder_512_210_mod_13_l948_94848

theorem remainder_512_210_mod_13 : 512^210 % 13 = 12 := by
  sorry

end remainder_512_210_mod_13_l948_94848


namespace polyhedron_space_diagonals_l948_94811

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem: A convex polyhedron Q with 30 vertices, 72 edges, 44 faces 
    (of which 32 are triangular and 12 are quadrilateral) has 339 space diagonals -/
theorem polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 32,
    quadrilateral_faces := 12
  }
  space_diagonals Q = 339 := by
  sorry


end polyhedron_space_diagonals_l948_94811


namespace cades_remaining_marbles_l948_94809

/-- Represents the number of marbles Cade has left after giving some away. -/
def marblesLeft (initial : ℕ) (givenAway : ℕ) : ℕ :=
  initial - givenAway

/-- Theorem stating that Cade's remaining marbles is the difference between his initial marbles and those given away. -/
theorem cades_remaining_marbles (initial : ℕ) (givenAway : ℕ) 
  (h : givenAway ≤ initial) : 
  marblesLeft initial givenAway = initial - givenAway :=
by
  sorry

#eval marblesLeft 87 8  -- Should output 79

end cades_remaining_marbles_l948_94809


namespace diamond_equation_solution_l948_94835

/-- Definition of the binary operation ◇ -/
noncomputable def diamond (k : ℝ) (a b : ℝ) : ℝ :=
  k / b

/-- Theorem stating the solution to the equation -/
theorem diamond_equation_solution (k : ℝ) (h1 : k = 2) :
  ∃ x : ℝ, diamond k 2023 (diamond k 7 x) = 150 ∧ x = 150 / 2023 := by
  sorry

/-- Properties of the binary operation ◇ -/
axiom diamond_assoc (k a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond k a (diamond k b c) = k * (diamond k a b) * c

axiom diamond_self (k a : ℝ) (ha : a ≠ 0) :
  diamond k a a = k

end diamond_equation_solution_l948_94835


namespace total_plums_eq_27_l948_94847

/-- The number of plums Alyssa picked -/
def alyssas_plums : ℕ := 17

/-- The number of plums Jason picked -/
def jasons_plums : ℕ := 10

/-- The total number of plums picked -/
def total_plums : ℕ := alyssas_plums + jasons_plums

theorem total_plums_eq_27 : total_plums = 27 := by
  sorry

end total_plums_eq_27_l948_94847


namespace carlos_has_largest_result_l948_94859

def starting_number : ℕ := 12

def alice_result : ℕ := ((starting_number - 2) * 3) + 3

def ben_result : ℕ := ((starting_number * 3) - 2) + 3

def carlos_result : ℕ := (starting_number - 2 + 3) * 3

theorem carlos_has_largest_result :
  carlos_result > alice_result ∧ carlos_result > ben_result :=
sorry

end carlos_has_largest_result_l948_94859


namespace parabola_vertex_l948_94852

/-- The vertex of the parabola y = 3(x+1)^2 + 4 is (-1, 4) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 3 * (x + 1)^2 + 4 → (∃ h k : ℝ, h = -1 ∧ k = 4 ∧ ∀ x y : ℝ, y = 3 * (x - h)^2 + k) :=
by sorry

end parabola_vertex_l948_94852


namespace even_odd_property_l948_94883

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

theorem even_odd_property (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_odd : is_odd_function (fun x ↦ f (x - 1)))
  (h_f2 : f 2 = 3) :
  f 5 + f 6 = 3 := by
sorry

end even_odd_property_l948_94883


namespace smallest_positive_integer_with_remainders_l948_94816

theorem smallest_positive_integer_with_remainders : ∃! x : ℕ+, 
  (x : ℤ) % 4 = 1 ∧ 
  (x : ℤ) % 5 = 2 ∧ 
  (x : ℤ) % 6 = 3 ∧ 
  ∀ y : ℕ+, 
    (y : ℤ) % 4 = 1 → 
    (y : ℤ) % 5 = 2 → 
    (y : ℤ) % 6 = 3 → 
    x ≤ y :=
by
  use 57
  sorry

end smallest_positive_integer_with_remainders_l948_94816


namespace skew_lines_sufficient_not_necessary_l948_94844

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line in 3D space
  -- This is a placeholder structure

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l₁ l₂ : Line3D) : Prop :=
  sorry

/-- Two lines do not intersect if they have no point in common -/
def do_not_intersect (l₁ l₂ : Line3D) : Prop :=
  sorry

theorem skew_lines_sufficient_not_necessary :
  (∀ l₁ l₂ : Line3D, are_skew l₁ l₂ → do_not_intersect l₁ l₂) ∧
  (∃ l₁ l₂ : Line3D, do_not_intersect l₁ l₂ ∧ ¬are_skew l₁ l₂) :=
sorry

end skew_lines_sufficient_not_necessary_l948_94844


namespace messages_sent_l948_94838

theorem messages_sent (lucia_day1 : ℕ) : 
  lucia_day1 > 20 →
  let alina_day1 := lucia_day1 - 20
  let lucia_day2 := lucia_day1 / 3
  let alina_day2 := 2 * alina_day1
  let lucia_day3 := lucia_day1
  let alina_day3 := alina_day1
  lucia_day1 + alina_day1 + lucia_day2 + alina_day2 + lucia_day3 + alina_day3 = 680 →
  lucia_day1 = 120 := by
sorry

end messages_sent_l948_94838


namespace g_of_3_equals_79_l948_94829

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 + 3 * x - 2

-- State the theorem
theorem g_of_3_equals_79 : g 3 = 79 := by
  sorry

end g_of_3_equals_79_l948_94829


namespace sam_initial_nickels_l948_94850

/-- Given information about Sam's nickels --/
structure SamNickels where
  initial : ℕ  -- Initial number of nickels
  given : ℕ    -- Number of nickels given by dad
  final : ℕ    -- Final number of nickels

/-- Theorem stating the initial number of nickels Sam had --/
theorem sam_initial_nickels (s : SamNickels) (h : s.final = s.initial + s.given) 
  (h_final : s.final = 63) (h_given : s.given = 39) : s.initial = 24 := by
  sorry

#check sam_initial_nickels

end sam_initial_nickels_l948_94850


namespace div_problem_l948_94870

theorem div_problem (a b c : ℚ) (h1 : a / b = 5) (h2 : b / c = 2/5) : c / a = 1/2 := by
  sorry

end div_problem_l948_94870


namespace world_cup_merchandise_problem_l948_94882

def total_items : ℕ := 90
def ornament_cost : ℕ := 40
def pendant_cost : ℕ := 25
def total_cost : ℕ := 2850
def ornament_price : ℕ := 50
def pendant_price : ℕ := 30
def min_profit : ℕ := 725

theorem world_cup_merchandise_problem :
  ∃ (ornaments pendants : ℕ),
    ornaments + pendants = total_items ∧
    ornament_cost * ornaments + pendant_cost * pendants = total_cost ∧
    ornaments = 40 ∧
    pendants = 50 ∧
    (∀ m : ℕ,
      m ≤ total_items ∧
      (ornament_price - ornament_cost) * (total_items - m) + (pendant_price - pendant_cost) * m ≥ min_profit
      → m ≤ 35) :=
by sorry

end world_cup_merchandise_problem_l948_94882


namespace rectangle_area_diagonal_l948_94830

theorem rectangle_area_diagonal (length width diagonal k : ℝ) : 
  length > 0 →
  width > 0 →
  diagonal > 0 →
  length / width = 5 / 2 →
  diagonal^2 = length^2 + width^2 →
  k = 10 / 29 →
  length * width = k * diagonal^2 := by
sorry

end rectangle_area_diagonal_l948_94830


namespace complex_equation_solution_l948_94858

theorem complex_equation_solution (x y z : ℂ) (h_real : x.im = 0)
  (h_sum : x + y + z = 5)
  (h_prod_sum : x * y + y * z + z * x = 5)
  (h_prod : x * y * z = 5) :
  x = 1 + (4 : ℂ) ^ (1/3 : ℂ) :=
sorry

end complex_equation_solution_l948_94858


namespace romance_movie_tickets_l948_94827

theorem romance_movie_tickets (horror_tickets romance_tickets : ℕ) : 
  horror_tickets = 3 * romance_tickets + 18 →
  horror_tickets = 93 →
  romance_tickets = 25 := by
sorry

end romance_movie_tickets_l948_94827


namespace max_value_theorem_l948_94842

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  (a^2 * b^2) / (a + b) + (a^2 * c^2) / (a + c) + (b^2 * c^2) / (b + c) ≤ 1/6 := by
  sorry

end max_value_theorem_l948_94842


namespace cubic_equation_roots_l948_94803

theorem cubic_equation_roots :
  let a : ℝ := 5
  let b : ℝ := (5 + Real.sqrt 61) / 2
  let f (x : ℝ) := x^3 - 5*x^2 - 9*x + 45
  (f a = 0 ∧ f b = 0 ∧ f (-b) = 0) ∧
  (∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ = -r₂) :=
by sorry

end cubic_equation_roots_l948_94803


namespace total_interest_calculation_l948_94887

/-- Given a principal amount and an interest rate, calculates the total interest after 10 years
    when the principal is trebled after 5 years and the initial 10-year simple interest is 1200. -/
theorem total_interest_calculation (P R : ℝ) : 
  (P * R * 10) / 100 = 1200 → 
  (P * R * 5) / 100 + (3 * P * R * 5) / 100 = 3000 := by
sorry

end total_interest_calculation_l948_94887


namespace tangent_line_at_x_1_l948_94874

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 4*x + 5

theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (3*x - y + 5 = 0) :=
by sorry

end tangent_line_at_x_1_l948_94874


namespace sine_function_properties_l948_94840

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem sine_function_properties :
  ∃ (A ω φ : ℝ),
    (f A ω φ 0 = 0) ∧
    (f A ω φ (π/2) = 2) ∧
    (f A ω φ π = 0) ∧
    (f A ω φ (3*π/2) = -2) ∧
    (f A ω φ (2*π) = 0) ∧
    (5*π/3 + π/3 = 2*π) →
    (A = 2) ∧
    (ω = 1/2) ∧
    (φ = 2*π/3) ∧
    (∀ x : ℝ, f A ω φ (x - π/3) = f A ω φ (-x - π/3)) :=
by sorry

end sine_function_properties_l948_94840


namespace fruit_amount_proof_l948_94819

/-- The cost of blueberries in dollars per carton -/
def blueberry_cost : ℚ := 5

/-- The weight of blueberries in ounces per carton -/
def blueberry_weight : ℚ := 6

/-- The cost of raspberries in dollars per carton -/
def raspberry_cost : ℚ := 3

/-- The weight of raspberries in ounces per carton -/
def raspberry_weight : ℚ := 8

/-- The number of batches of muffins -/
def num_batches : ℕ := 4

/-- The total savings in dollars by using raspberries instead of blueberries -/
def total_savings : ℚ := 22

/-- The amount of fruit in ounces required for each batch of muffins -/
def fruit_per_batch : ℚ := 12

theorem fruit_amount_proof :
  (total_savings / (num_batches : ℚ)) / 
  ((blueberry_cost / blueberry_weight) - (raspberry_cost / raspberry_weight)) = fruit_per_batch :=
sorry

end fruit_amount_proof_l948_94819


namespace train_distance_difference_l948_94807

/-- Proves the difference in distance traveled by two trains meeting each other --/
theorem train_distance_difference (v1 v2 total_distance : ℝ) 
  (h1 : v1 = 16)
  (h2 : v2 = 21)
  (h3 : total_distance = 444)
  (h4 : v1 > 0)
  (h5 : v2 > 0) :
  let t := total_distance / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  d2 - d1 = 60 := by sorry

end train_distance_difference_l948_94807


namespace simplify_and_sum_fraction_l948_94861

theorem simplify_and_sum_fraction : ∃ (a b : ℕ), 
  (a : ℚ) / b = 63 / 126 ∧ 
  (∀ (c d : ℕ), (c : ℚ) / d = 63 / 126 → a ≤ c ∧ b ≤ d) ∧ 
  a + b = 9 := by
sorry

end simplify_and_sum_fraction_l948_94861


namespace impossible_time_reduction_l948_94839

/-- Proves that it's impossible to reduce the time per kilometer by 1 minute when starting from a speed of 60 km/h. -/
theorem impossible_time_reduction (initial_speed : ℝ) (time_reduction : ℝ) : 
  initial_speed = 60 → time_reduction = 1 → ¬ (∃ (new_speed : ℝ), new_speed > 0 ∧ (1 / new_speed) * 60 = (1 / initial_speed) * 60 - time_reduction) :=
by
  sorry


end impossible_time_reduction_l948_94839


namespace circle_radius_l948_94815

theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 + 36 = 6*x + 24*y) → 
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ r = Real.sqrt 117 := by
  sorry

end circle_radius_l948_94815


namespace supermarket_spending_l948_94860

theorem supermarket_spending (total_spent : ℚ) 
  (h1 : total_spent = 150)
  (h2 : ∃ (fruits_veg meat bakery candy : ℚ),
    fruits_veg = 1/2 * total_spent ∧
    meat = 1/3 * total_spent ∧
    candy = 10 ∧
    fruits_veg + meat + bakery + candy = total_spent) :
  ∃ (bakery : ℚ), bakery = 1/10 * total_spent := by
  sorry

end supermarket_spending_l948_94860


namespace problem1_l948_94897

theorem problem1 : Real.sqrt 4 - (1/2)⁻¹ + (2 - 1/7)^0 = 1 := by
  sorry

end problem1_l948_94897


namespace function_with_two_zeros_m_range_l948_94813

theorem function_with_two_zeros_m_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  m < -2 ∨ m > 2 :=
sorry

end function_with_two_zeros_m_range_l948_94813


namespace f_range_l948_94836

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x ^ 3 + 2 * Real.sin x ^ 2 - 4 * Real.sin x + 3 * Real.cos x + 3 * Real.cos x ^ 2 - 2) / (Real.sin x - 1)

theorem f_range :
  Set.range (fun (x : ℝ) => f x) = Set.Icc 1 (1 + 3 * Real.sqrt 2) :=
sorry

end f_range_l948_94836


namespace stability_comparison_A_more_stable_than_B_l948_94865

/-- Represents a set of data with its variance -/
structure DataSet where
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines stability comparison between two DataSets -/
def more_stable (a b : DataSet) : Prop :=
  a.variance < b.variance

theorem stability_comparison (a b : DataSet) :
  a.variance < b.variance → more_stable a b :=
by
  sorry

/-- Example datasets A and B -/
def A : DataSet := ⟨0.03, by norm_num⟩
def B : DataSet := ⟨0.13, by norm_num⟩

/-- Theorem stating that A is more stable than B -/
theorem A_more_stable_than_B : more_stable A B :=
by
  sorry

end stability_comparison_A_more_stable_than_B_l948_94865


namespace equal_charge_at_60_minutes_l948_94805

/-- United Telephone's base rate in dollars -/
def united_base : ℝ := 9

/-- United Telephone's per-minute charge in dollars -/
def united_per_minute : ℝ := 0.25

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℝ := 12

/-- Atlantic Call's per-minute charge in dollars -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which both companies charge the same amount -/
def equal_charge_minutes : ℝ := 60

theorem equal_charge_at_60_minutes :
  united_base + united_per_minute * equal_charge_minutes =
  atlantic_base + atlantic_per_minute * equal_charge_minutes :=
by sorry

end equal_charge_at_60_minutes_l948_94805


namespace house_sale_gain_l948_94856

/-- Calculates the net gain from selling a house at a profit and buying it back at a loss -/
def netGainFromHouseSale (initialValue : ℝ) (profitPercent : ℝ) (lossPercent : ℝ) : ℝ :=
  let sellingPrice := initialValue * (1 + profitPercent)
  let buybackPrice := sellingPrice * (1 - lossPercent)
  sellingPrice - buybackPrice

/-- Theorem stating that selling a $200,000 house at 15% profit and buying it back at 5% loss results in $11,500 gain -/
theorem house_sale_gain :
  netGainFromHouseSale 200000 0.15 0.05 = 11500 := by
  sorry

end house_sale_gain_l948_94856


namespace raft_capacity_l948_94872

theorem raft_capacity (total_capacity : ℕ) (reduction_with_jackets : ℕ) (people_needing_jackets : ℕ) : 
  total_capacity = 21 → 
  reduction_with_jackets = 7 → 
  people_needing_jackets = 8 → 
  (total_capacity - (reduction_with_jackets * people_needing_jackets / (total_capacity - reduction_with_jackets))) = 17 := by
sorry

end raft_capacity_l948_94872


namespace square_circumcircle_integer_points_l948_94880

/-- The circumcircle of a square with side length 1978 contains no integer points other than the vertices of the square. -/
theorem square_circumcircle_integer_points :
  ∀ x y : ℤ,
  (x - 989)^2 + (y - 989)^2 = 2 * 989^2 →
  (x = 0 ∧ y = 0) ∨ (x = 0 ∧ y = 1978) ∨ (x = 1978 ∧ y = 0) ∨ (x = 1978 ∧ y = 1978) :=
by sorry


end square_circumcircle_integer_points_l948_94880


namespace least_exponent_sum_for_800_l948_94834

def isPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def isDistinctPowerSum (n : ℕ) (powers : List ℕ) : Prop :=
  (powers.length ≥ 2) ∧
  (powers.sum = n) ∧
  (∀ p ∈ powers, isPowerOfTwo p) ∧
  powers.Nodup

theorem least_exponent_sum_for_800 :
  ∃ (powers : List ℕ),
    isDistinctPowerSum 800 powers ∧
    (∀ (other_powers : List ℕ),
      isDistinctPowerSum 800 other_powers →
      (powers.map (fun p => (Nat.log p 2))).sum ≤ (other_powers.map (fun p => (Nat.log p 2))).sum) ∧
    (powers.map (fun p => (Nat.log p 2))).sum = 22 :=
sorry

end least_exponent_sum_for_800_l948_94834


namespace solution_difference_l948_94810

/-- The quadratic equation from the problem -/
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - 3*x + 9 = x + 41

/-- The two solutions of the quadratic equation -/
def solutions : Set ℝ :=
  {x : ℝ | quadratic_equation x}

/-- Theorem stating that the positive difference between the two solutions is 12 -/
theorem solution_difference : 
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 12 :=
sorry

end solution_difference_l948_94810


namespace sheridan_fish_count_l948_94892

/-- Calculates the remaining number of fish after giving some away -/
def remaining_fish (initial : Real) (given_away : Real) : Real :=
  initial - given_away

/-- Theorem: Mrs. Sheridan has 25.0 fish after giving away 22.0 from her initial 47.0 fish -/
theorem sheridan_fish_count : remaining_fish 47.0 22.0 = 25.0 := by
  sorry

end sheridan_fish_count_l948_94892


namespace fabric_C_required_is_120_l948_94812

/-- Calculates the amount of fabric C required for pants production every week -/
def fabric_C_required (
  kingsley_pants_per_day : ℕ)
  (kingsley_work_days : ℕ)
  (fabric_C_per_pants : ℕ) : ℕ :=
  kingsley_pants_per_day * kingsley_work_days * fabric_C_per_pants

/-- Proves that the amount of fabric C required for pants production every week is 120 yards -/
theorem fabric_C_required_is_120 :
  fabric_C_required 4 6 5 = 120 := by
  sorry

end fabric_C_required_is_120_l948_94812


namespace concentric_circles_track_width_l948_94862

def track_width (r1 r2 r3 : ℝ) : Prop :=
  2 * Real.pi * r2 - 2 * Real.pi * r1 = 20 * Real.pi ∧
  2 * Real.pi * r3 - 2 * Real.pi * r2 = 30 * Real.pi ∧
  r3 - r1 = 25

theorem concentric_circles_track_width :
  ∀ r1 r2 r3 : ℝ, track_width r1 r2 r3 :=
by
  sorry

end concentric_circles_track_width_l948_94862


namespace perfect_square_condition_l948_94841

theorem perfect_square_condition (n : ℕ) : 
  (∃ m : ℕ, n^2 + 3*n = m^2) ↔ n = 1 := by sorry

end perfect_square_condition_l948_94841


namespace trapezoid_shorter_base_l948_94846

/-- A trapezoid with a line joining the midpoints of its diagonals. -/
structure Trapezoid where
  /-- The length of the longer base of the trapezoid. -/
  longer_base : ℝ
  /-- The length of the shorter base of the trapezoid. -/
  shorter_base : ℝ
  /-- The length of the line joining the midpoints of the diagonals. -/
  midline_length : ℝ
  /-- The midline length is half the difference of the bases. -/
  midline_property : midline_length = (longer_base - shorter_base) / 2

/-- 
Given a trapezoid where the line joining the midpoints of the diagonals has length 5
and the longer base is 105, the shorter base has length 95.
-/
theorem trapezoid_shorter_base (t : Trapezoid) 
    (h1 : t.longer_base = 105)
    (h2 : t.midline_length = 5) : 
    t.shorter_base = 95 := by
  sorry


end trapezoid_shorter_base_l948_94846


namespace cubic_root_relation_l948_94885

theorem cubic_root_relation (m n p x₃ : ℝ) : 
  (∃ (z : ℂ), z^3 + (m/3)*z^2 + (n/3)*z + (p/3) = 0 ∧ 
               (z = 4 + 3*Complex.I ∨ z = 4 - 3*Complex.I ∨ z = x₃)) →
  x₃ > 0 →
  p = -75 * x₃ := by
sorry

end cubic_root_relation_l948_94885


namespace min_value_x_plus_2y_l948_94826

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (2 * x + y) + 1 / (y + 1) = 1) :
  x + 2 * y ≥ 1 / 2 + Real.sqrt 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    1 / (2 * x₀ + y₀) + 1 / (y₀ + 1) = 1 ∧
    x₀ + 2 * y₀ = 1 / 2 + Real.sqrt 3 :=
by sorry

end min_value_x_plus_2y_l948_94826


namespace tyler_erasers_count_l948_94888

def tyler_problem (initial_money : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) 
  (eraser_price : ℕ) (remaining_money : ℕ) : ℕ := 
  let money_after_scissors := initial_money - scissors_count * scissors_price
  let money_spent_on_erasers := money_after_scissors - remaining_money
  money_spent_on_erasers / eraser_price

theorem tyler_erasers_count : 
  tyler_problem 100 8 5 4 20 = 10 := by
  sorry

end tyler_erasers_count_l948_94888


namespace tangent_line_equation_l948_94871

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the point of tangency
def P : ℝ × ℝ := (1, 3)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), (
    -- The line y = mx + b passes through P
    m * P.1 + b = P.2 ∧
    -- The slope m is equal to f'(1)
    m = (6 : ℝ) * P.1 - 1 ∧
    -- The resulting equation is 2x - y + 1 = 0
    m = 2 ∧ b = 1 ∧
    ∀ x y, y = m * x + b ↔ 2 * x - y + 1 = 0
  ) := by sorry


end tangent_line_equation_l948_94871


namespace root_power_floor_l948_94821

theorem root_power_floor (a : ℝ) : 
  a^5 - a^3 + a - 2 = 0 → ⌊a^6⌋ = 3 := by
sorry

end root_power_floor_l948_94821


namespace smallest_solution_of_equation_l948_94873

theorem smallest_solution_of_equation (x : ℝ) : 
  (x = (7 - Real.sqrt 33) / 2) ↔ 
  (x < (7 + Real.sqrt 33) / 2 ∧ 1 / (x - 1) + 1 / (x - 5) = 4 / (x - 2)) :=
by sorry

end smallest_solution_of_equation_l948_94873


namespace compute_expression_l948_94877

theorem compute_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end compute_expression_l948_94877


namespace inscribed_triangles_inequality_l948_94823

/-- Two equilateral triangles inscribed in a circle -/
structure InscribedTriangles where
  r : ℝ
  S : ℝ

/-- Theorem: For two equilateral triangles inscribed in a circle with radius r,
    where S is the area of their common part, 2S ≥ √3 r² holds. -/
theorem inscribed_triangles_inequality (t : InscribedTriangles) : 2 * t.S ≥ Real.sqrt 3 * t.r^2 := by
  sorry

end inscribed_triangles_inequality_l948_94823


namespace inequality_implies_zero_for_nonpositive_l948_94895

/-- A function satisfying the given inequality for all real x and y -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) ≤ y * f x + f (f x)

/-- The main theorem: if f satisfies the inequality, then f(x) = 0 for all x ≤ 0 -/
theorem inequality_implies_zero_for_nonpositive
  (f : ℝ → ℝ) (h : SatisfiesInequality f) :
  ∀ x : ℝ, x ≤ 0 → f x = 0 := by
  sorry

end inequality_implies_zero_for_nonpositive_l948_94895


namespace arithmetic_sequence_50th_term_l948_94824

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_50th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_first : a 1 = 7)
  (h_fifteenth : a 15 = 41) :
  a 50 = 126 := by
  sorry

end arithmetic_sequence_50th_term_l948_94824


namespace lakers_win_probability_l948_94817

/-- The probability of the Lakers winning a single game -/
def p_lakers : ℚ := 2/3

/-- The probability of the Celtics winning a single game -/
def p_celtics : ℚ := 1/3

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The minimum number of games in the series -/
def min_games : ℕ := 5

/-- The probability of the Lakers winning the NBA Finals given that the series lasts at least 5 games -/
theorem lakers_win_probability : 
  (Finset.sum (Finset.range 3) (λ i => 
    (Nat.choose (games_to_win + i) i) * 
    (p_lakers ^ games_to_win) * 
    (p_celtics ^ i))) = 1040/729 := by sorry

end lakers_win_probability_l948_94817


namespace no_solution_to_inequality_system_l948_94878

theorem no_solution_to_inequality_system :
  ¬ ∃ x : ℝ, (2 * x + 3 ≥ x + 11) ∧ ((2 * x + 5) / 3 - 1 < 2 - x) := by
  sorry

end no_solution_to_inequality_system_l948_94878


namespace focus_of_specific_ellipse_l948_94894

/-- An ellipse with given major and minor axis endpoints -/
structure Ellipse where
  major_axis_start : ℝ × ℝ
  major_axis_end : ℝ × ℝ
  minor_axis_start : ℝ × ℝ
  minor_axis_end : ℝ × ℝ

/-- The focus of an ellipse with the greater x-coordinate -/
def focus_with_greater_x (e : Ellipse) : ℝ × ℝ :=
  sorry

/-- Theorem stating that for the given ellipse, the focus with greater x-coordinate is at (3, -2) -/
theorem focus_of_specific_ellipse :
  let e : Ellipse := {
    major_axis_start := (0, -2),
    major_axis_end := (6, -2),
    minor_axis_start := (3, 1),
    minor_axis_end := (3, -5)
  }
  focus_with_greater_x e = (3, -2) := by
  sorry

end focus_of_specific_ellipse_l948_94894


namespace sandwich_contest_difference_l948_94825

theorem sandwich_contest_difference : (5 : ℚ) / 6 - (2 : ℚ) / 3 = (1 : ℚ) / 6 := by sorry

end sandwich_contest_difference_l948_94825


namespace intersection_points_count_l948_94837

/-- The number of intersection points between y = |3x + 4| and y = -|4x + 3| -/
theorem intersection_points_count : ∃! p : ℝ × ℝ, 
  (p.2 = |3 * p.1 + 4|) ∧ (p.2 = -|4 * p.1 + 3|) := by
  sorry

end intersection_points_count_l948_94837


namespace smallest_k_for_same_remainder_l948_94898

theorem smallest_k_for_same_remainder : ∃ (k : ℕ), k > 0 ∧
  (∀ (n : ℕ), n > 0 → n < k → ¬((201 + n) % 24 = (9 + n) % 24)) ∧
  ((201 + k) % 24 = (9 + k) % 24) ∧
  (201 % 24 = 9 % 24) :=
by sorry

end smallest_k_for_same_remainder_l948_94898


namespace ellipse_equation_l948_94875

/-- Represents an ellipse with foci on coordinate axes and midpoint at origin -/
structure Ellipse where
  focal_distance : ℝ
  sum_distances : ℝ

/-- The equation of the ellipse when foci are on the x-axis -/
def ellipse_equation_x (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / ((e.sum_distances / 2)^2) + y^2 / ((e.sum_distances / 2)^2 - (e.focal_distance / 2)^2) = 1

/-- The equation of the ellipse when foci are on the y-axis -/
def ellipse_equation_y (e : Ellipse) (x y : ℝ) : Prop :=
  y^2 / ((e.sum_distances / 2)^2) + x^2 / ((e.sum_distances / 2)^2 - (e.focal_distance / 2)^2) = 1

/-- Theorem stating the equation of the ellipse given the conditions -/
theorem ellipse_equation (e : Ellipse) (h1 : e.focal_distance = 8) (h2 : e.sum_distances = 12) :
  ∀ x y : ℝ, ellipse_equation_x e x y ∨ ellipse_equation_y e x y :=
sorry

end ellipse_equation_l948_94875
