import Mathlib

namespace NUMINAMATH_CALUDE_unique_rectangle_existence_l1845_184557

theorem unique_rectangle_existence (a b : ℝ) (h : 0 < a ∧ a < b) :
  ∃! (x y : ℝ), 0 < x ∧ x < y ∧ 2 * (x + y) = a + b ∧ x * y = (a * b) / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_rectangle_existence_l1845_184557


namespace NUMINAMATH_CALUDE_equation_condition_l1845_184579

theorem equation_condition (a b c : ℤ) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 2 → (a > b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_equation_condition_l1845_184579


namespace NUMINAMATH_CALUDE_break_room_vacant_seats_l1845_184597

theorem break_room_vacant_seats :
  let total_tables : ℕ := 5
  let seats_per_table : ℕ := 8
  let occupied_tables : ℕ := 2
  let people_per_occupied_table : ℕ := 3
  let unusable_tables : ℕ := 1

  let usable_tables : ℕ := total_tables - unusable_tables
  let total_seats : ℕ := usable_tables * seats_per_table
  let occupied_seats : ℕ := occupied_tables * people_per_occupied_table

  total_seats - occupied_seats = 26 :=
by sorry

end NUMINAMATH_CALUDE_break_room_vacant_seats_l1845_184597


namespace NUMINAMATH_CALUDE_unique_solution_system_l1845_184574

theorem unique_solution_system (x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  (1 / x + 1 / y + 1 / z = 3) →
  (1 / (x * y) + 1 / (y * z) + 1 / (z * x) = 3) →
  (1 / (x * y * z) = 1) →
  (x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1845_184574


namespace NUMINAMATH_CALUDE_bill_caroline_age_ratio_l1845_184560

/-- Given the ages of Bill and Caroline, prove their age ratio -/
theorem bill_caroline_age_ratio :
  ∀ (bill_age caroline_age : ℕ),
  bill_age = 17 →
  bill_age + caroline_age = 26 →
  ∃ (n : ℕ), bill_age = n * caroline_age - 1 →
  (bill_age : ℚ) / caroline_age = 17 / 9 := by
  sorry

end NUMINAMATH_CALUDE_bill_caroline_age_ratio_l1845_184560


namespace NUMINAMATH_CALUDE_epidemic_supplies_theorem_l1845_184545

-- Define the prices of type A and B supplies
def price_A : ℕ := 16
def price_B : ℕ := 4

-- Define the conditions
axiom condition1 : 60 * price_A + 45 * price_B = 1140
axiom condition2 : 45 * price_A + 30 * price_B = 840

-- Define the total units and budget
def total_units : ℕ := 600
def total_budget : ℕ := 8000

-- Define the function to calculate the maximum number of type A units
def max_type_A : ℕ :=
  (total_budget - price_B * total_units) / (price_A - price_B)

-- Theorem to prove
theorem epidemic_supplies_theorem :
  price_A = 16 ∧ price_B = 4 ∧ max_type_A = 466 :=
sorry

end NUMINAMATH_CALUDE_epidemic_supplies_theorem_l1845_184545


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l1845_184524

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l1845_184524


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1845_184501

theorem complex_modulus_problem (z : ℂ) : z = (3 - Complex.I) / (1 + 2 * Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1845_184501


namespace NUMINAMATH_CALUDE_mathematicians_ages_l1845_184576

/-- Represents a mathematician --/
inductive Mathematician
| A
| B
| C

/-- Calculates the age of mathematician A or C given the base and smallest number --/
def calculate_age_A_C (base : ℕ) (smallest : ℕ) : ℕ :=
  smallest * base + (smallest + 2)

/-- Calculates the age of mathematician B given the base and smallest number --/
def calculate_age_B (base : ℕ) (smallest : ℕ) : ℕ :=
  smallest * base + (smallest + 1)

/-- Checks if the calculated age matches the product of the two largest numbers --/
def check_age_A_C (age : ℕ) (smallest : ℕ) : Prop :=
  age = (smallest + 4) * (smallest + 6)

/-- Checks if the calculated age matches the product of the next two consecutive numbers --/
def check_age_B (age : ℕ) (smallest : ℕ) : Prop :=
  age = (smallest + 2) * (smallest + 3)

theorem mathematicians_ages :
  ∃ (age_A age_B age_C : ℕ) (base_A base_B : ℕ) (smallest_A smallest_B smallest_C : ℕ),
    calculate_age_A_C base_A smallest_A = age_A ∧
    calculate_age_B base_B smallest_B = age_B ∧
    calculate_age_A_C base_A smallest_C = age_C ∧
    check_age_A_C age_A smallest_A ∧
    check_age_B age_B smallest_B ∧
    check_age_A_C age_C smallest_C ∧
    age_C < age_A ∧
    age_C < age_B ∧
    age_A = 48 ∧
    age_B = 56 ∧
    age_C = 35 ∧
    base_B = 10 :=
  by sorry

/-- Identifies the absent-minded mathematician --/
def absent_minded : Mathematician := Mathematician.B

end NUMINAMATH_CALUDE_mathematicians_ages_l1845_184576


namespace NUMINAMATH_CALUDE_sum_of_real_cube_roots_of_64_l1845_184512

theorem sum_of_real_cube_roots_of_64 :
  ∃ (x : ℝ), x^3 = 64 ∧ (∀ y : ℝ, y^3 = 64 → y = x) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_real_cube_roots_of_64_l1845_184512


namespace NUMINAMATH_CALUDE_brad_balloons_l1845_184595

theorem brad_balloons (total : ℕ) (red : ℕ) (green : ℕ) (blue : ℕ) 
  (h1 : total = 37)
  (h2 : red = 14)
  (h3 : green = 10)
  (h4 : total = red + green + blue) :
  blue = 13 := by
  sorry

end NUMINAMATH_CALUDE_brad_balloons_l1845_184595


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_perp_line_l1845_184589

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the "contained in" relation for a line in a plane
variable (contained_in : Line → Plane → Prop)

theorem line_perp_plane_implies_perp_line 
  (l m : Line) (α : Plane) :
  perpendicular_line_plane l α → contained_in m α → perpendicular_lines l m :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_perp_line_l1845_184589


namespace NUMINAMATH_CALUDE_quadratic_complex_roots_l1845_184539

theorem quadratic_complex_roots (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ a b : ℝ, b ≠ 0 ∧ (∀ x : ℂ, x^2 + p*x + q = 0 → x = Complex.mk a b ∨ x = Complex.mk a (-b))) →
  (∀ x : ℂ, x^2 + p*x + q = 0 → x.re = 1/2) →
  p = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_complex_roots_l1845_184539


namespace NUMINAMATH_CALUDE_village_population_l1845_184554

theorem village_population (population_percentage : ℝ) (partial_population : ℕ) (total_population : ℕ) :
  population_percentage = 80 →
  partial_population = 64000 →
  (population_percentage / 100) * total_population = partial_population →
  total_population = 80000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1845_184554


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l1845_184593

theorem solution_set_implies_a_value 
  (h : ∀ x : ℝ, -1 < x ∧ x < 2 ↔ -1/2 * x^2 + a * x > -1) : 
  a = 1/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l1845_184593


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1845_184559

theorem quadratic_inequality_solution (a c : ℝ) :
  (∀ x : ℝ, (a * x^2 + 2 * x + c < 0) ↔ (x < -1 ∨ x > 2)) →
  a + c = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1845_184559


namespace NUMINAMATH_CALUDE_exam_results_l1845_184568

theorem exam_results (failed_hindi : ℝ) (failed_english : ℝ) (failed_both : ℝ)
  (h1 : failed_hindi = 34)
  (h2 : failed_english = 44)
  (h3 : failed_both = 22) :
  100 - (failed_hindi + failed_english - failed_both) = 44 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l1845_184568


namespace NUMINAMATH_CALUDE_wall_ratio_l1845_184515

/-- Given a wall with specific dimensions, prove the ratio of its length to height --/
theorem wall_ratio (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  volume = w * h * l →
  w = 6.999999999999999 →
  volume = 86436 →
  l / h = 7 := by
sorry

end NUMINAMATH_CALUDE_wall_ratio_l1845_184515


namespace NUMINAMATH_CALUDE_degree_of_5m2n3_l1845_184543

/-- The degree of a monomial is the sum of the exponents of its variables. -/
def degree_of_monomial (m : ℕ) (n : ℕ) : ℕ := m + n

/-- The monomial 5m^2n^3 has degree 5. -/
theorem degree_of_5m2n3 : degree_of_monomial 2 3 = 5 := by sorry

end NUMINAMATH_CALUDE_degree_of_5m2n3_l1845_184543


namespace NUMINAMATH_CALUDE_age_of_b_l1845_184507

/-- Given three people a, b, and c, with their ages represented as natural numbers. -/
def problem (a b c : ℕ) : Prop :=
  -- The average age of a, b, and c is 26 years
  (a + b + c) / 3 = 26 ∧
  -- The average age of a and c is 29 years
  (a + c) / 2 = 29 →
  -- The age of b is 20 years
  b = 20

/-- Theorem stating that under the given conditions, the age of b must be 20 years -/
theorem age_of_b (a b c : ℕ) : problem a b c := by
  sorry

end NUMINAMATH_CALUDE_age_of_b_l1845_184507


namespace NUMINAMATH_CALUDE_equal_money_after_transfer_l1845_184506

/-- Given that Ann has $777 and Bill has $1,111, prove that if Bill gives $167 to Ann, 
    they will have equal amounts of money. -/
theorem equal_money_after_transfer (ann_initial : ℕ) (bill_initial : ℕ) (transfer : ℕ) : 
  ann_initial = 777 →
  bill_initial = 1111 →
  transfer = 167 →
  ann_initial + transfer = bill_initial - transfer :=
by
  sorry

#check equal_money_after_transfer

end NUMINAMATH_CALUDE_equal_money_after_transfer_l1845_184506


namespace NUMINAMATH_CALUDE_mary_initial_marbles_l1845_184577

/-- The number of yellow marbles Mary gave to Joan -/
def marbles_given : ℕ := 3

/-- The number of yellow marbles Mary has left -/
def marbles_left : ℕ := 6

/-- The initial number of yellow marbles Mary had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem mary_initial_marbles : initial_marbles = 9 := by
  sorry

end NUMINAMATH_CALUDE_mary_initial_marbles_l1845_184577


namespace NUMINAMATH_CALUDE_solution_set_f_geq_6_min_value_f_min_value_a_plus_2b_l1845_184550

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 3|

-- Theorem for the solution set of f(x) ≥ 6
theorem solution_set_f_geq_6 :
  {x : ℝ | f x ≥ 6} = {x : ℝ | x ≥ 1 ∨ x ≤ -2} :=
sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (m : ℝ), m = 4 ∧ ∀ x, f x ≥ m :=
sorry

-- Theorem for the minimum value of a + 2b
theorem min_value_a_plus_2b :
  ∀ (a b : ℝ), a > 0 → b > 0 → 2*a*b + a + 2*b = 4 →
  a + 2*b ≥ 2*Real.sqrt 5 - 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_6_min_value_f_min_value_a_plus_2b_l1845_184550


namespace NUMINAMATH_CALUDE_smallest_n_satisfies_conditions_count_non_seven_divisors_l1845_184587

def is_perfect_square (x : ℕ) : Prop := ∃ m : ℕ, x = m^2

def is_perfect_cube (x : ℕ) : Prop := ∃ m : ℕ, x = m^3

def is_perfect_seventh (x : ℕ) : Prop := ∃ m : ℕ, x = m^7

def smallest_n : ℕ := 2^6 * 3^10 * 7^14

theorem smallest_n_satisfies_conditions :
  is_perfect_square (smallest_n / 2) ∧
  is_perfect_cube (smallest_n / 3) ∧
  is_perfect_seventh (smallest_n / 7) := by sorry

theorem count_non_seven_divisors :
  (Finset.filter (fun d => ¬(d % 7 = 0)) (Nat.divisors smallest_n)).card = 77 := by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfies_conditions_count_non_seven_divisors_l1845_184587


namespace NUMINAMATH_CALUDE_bridge_length_l1845_184534

/-- The length of a bridge given train parameters --/
theorem bridge_length
  (train_length : ℝ)
  (crossing_time : ℝ)
  (train_speed_kmph : ℝ)
  (h1 : train_length = 100)
  (h2 : crossing_time = 26.997840172786177)
  (h3 : train_speed_kmph = 36) :
  ∃ (bridge_length : ℝ),
    bridge_length = 169.97840172786177 :=
by
  sorry

#check bridge_length

end NUMINAMATH_CALUDE_bridge_length_l1845_184534


namespace NUMINAMATH_CALUDE_third_generation_tail_length_l1845_184540

/-- The tail length increase factor between generations -/
def increase_factor : ℝ := 1.25

/-- The initial tail length of the first generation in centimeters -/
def initial_length : ℝ := 16

/-- The tail length of a given generation -/
def tail_length (generation : ℕ) : ℝ :=
  initial_length * (increase_factor ^ generation)

/-- Theorem: The tail length of the third generation is 25 cm -/
theorem third_generation_tail_length :
  tail_length 2 = 25 := by sorry

end NUMINAMATH_CALUDE_third_generation_tail_length_l1845_184540


namespace NUMINAMATH_CALUDE_square_difference_l1845_184530

theorem square_difference (n : ℝ) : 
  let m : ℝ := 4 * n + 3
  m^2 - 8 * m * n + 16 * n^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1845_184530


namespace NUMINAMATH_CALUDE_distance_inequality_l1845_184519

/-- Given five points A, B, C, D, E on a plane, 
    the sum of distances AB + CD + DE + EC 
    is less than or equal to 
    the sum of distances AC + AD + AE + BC + BD + BE -/
theorem distance_inequality (A B C D E : EuclideanSpace ℝ (Fin 2)) :
  dist A B + dist C D + dist D E + dist E C ≤ 
  dist A C + dist A D + dist A E + dist B C + dist B D + dist B E := by
  sorry

end NUMINAMATH_CALUDE_distance_inequality_l1845_184519


namespace NUMINAMATH_CALUDE_coefficient_x3_in_product_l1845_184544

theorem coefficient_x3_in_product : 
  let p1 : Polynomial ℤ := 2 * X^4 + 3 * X^3 - 4 * X^2 + 2
  let p2 : Polynomial ℤ := X^3 - 8 * X + 3
  (p1 * p2).coeff 3 = 41 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3_in_product_l1845_184544


namespace NUMINAMATH_CALUDE_hockey_league_games_l1845_184551

/-- The number of games played in a hockey league --/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) / 2 * games_per_pair

/-- Theorem: In a league with 12 teams, where each team plays 4 games against every other team, 
    the total number of games played is 264. --/
theorem hockey_league_games :
  total_games 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l1845_184551


namespace NUMINAMATH_CALUDE_chewing_gums_count_l1845_184580

/-- Given the total number of treats, chocolate bars, and candies, prove the number of chewing gums. -/
theorem chewing_gums_count 
  (total_treats : ℕ) 
  (chocolate_bars : ℕ) 
  (candies : ℕ) 
  (h1 : total_treats = 155) 
  (h2 : chocolate_bars = 55) 
  (h3 : candies = 40) : 
  total_treats - (chocolate_bars + candies) = 60 := by
  sorry

#check chewing_gums_count

end NUMINAMATH_CALUDE_chewing_gums_count_l1845_184580


namespace NUMINAMATH_CALUDE_calculation_proofs_l1845_184517

theorem calculation_proofs :
  (1 - 2^2 / (1/5) * 5 - (-10)^2 - |(-3)| = -123) ∧
  ((-1)^2023 + (-5) * ((-2)^3 + 2) - (-4)^2 / (-1/2) = 61) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l1845_184517


namespace NUMINAMATH_CALUDE_route_distance_l1845_184590

theorem route_distance (time_Q : ℝ) (time_Y : ℝ) (speed_ratio : ℝ) :
  time_Q = 2 →
  time_Y = 4/3 →
  speed_ratio = 3/2 →
  ∃ (distance : ℝ) (speed_Q : ℝ),
    distance = speed_Q * time_Q ∧
    distance = (speed_ratio * speed_Q) * time_Y ∧
    distance = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_route_distance_l1845_184590


namespace NUMINAMATH_CALUDE_negative_x_squared_to_fourth_negative_x_squared_y_cubed_l1845_184505

-- Problem 1
theorem negative_x_squared_to_fourth (x : ℝ) : (-x^2)^4 = x^8 := by sorry

-- Problem 2
theorem negative_x_squared_y_cubed (x y : ℝ) : (-x^2*y)^3 = -x^6*y^3 := by sorry

end NUMINAMATH_CALUDE_negative_x_squared_to_fourth_negative_x_squared_y_cubed_l1845_184505


namespace NUMINAMATH_CALUDE_condition_a_sufficient_not_necessary_l1845_184570

theorem condition_a_sufficient_not_necessary :
  (∀ a b c d : ℝ, a > b ∧ c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) :=
by sorry

end NUMINAMATH_CALUDE_condition_a_sufficient_not_necessary_l1845_184570


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1845_184564

/-
  Define the hyperbola equation
-/
def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-
  Define the asymptote equation
-/
def has_asymptote (x y : ℝ) : Prop :=
  y = 2 * x

/-
  Define the parabola equation
-/
def is_parabola (x y : ℝ) : Prop :=
  y^2 = 20 * x

/-
  State the theorem
-/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x y : ℝ, is_hyperbola a b x y ∧ has_asymptote x y) →
  (∃ x y : ℝ, is_parabola x y ∧ 
    ((x - 5)^2 + y^2 = a^2 + b^2 ∨ (x + 5)^2 + y^2 = a^2 + b^2)) →
  a^2 = 5 ∧ b^2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1845_184564


namespace NUMINAMATH_CALUDE_no_intersection_l1845_184541

/-- The function representing y = |3x + 6| -/
def f (x : ℝ) : ℝ := |3 * x + 6|

/-- The function representing y = -|4x - 3| -/
def g (x : ℝ) : ℝ := -|4 * x - 3|

/-- Theorem stating that there are no intersection points between f and g -/
theorem no_intersection :
  ¬∃ (x : ℝ), f x = g x :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l1845_184541


namespace NUMINAMATH_CALUDE_chocolate_cookie_percentage_l1845_184525

/-- Calculates the percentage of chocolate in cookies given the initial ingredients and leftover chocolate. -/
theorem chocolate_cookie_percentage
  (dough : ℝ)
  (initial_chocolate : ℝ)
  (leftover_chocolate : ℝ)
  (h_dough : dough = 36)
  (h_initial : initial_chocolate = 13)
  (h_leftover : leftover_chocolate = 4) :
  (initial_chocolate - leftover_chocolate) / (dough + initial_chocolate - leftover_chocolate) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cookie_percentage_l1845_184525


namespace NUMINAMATH_CALUDE_borrowed_amount_with_interest_l1845_184509

/-- Calculates the total amount to be returned given a borrowed amount and an interest rate. -/
def totalAmount (borrowed : ℝ) (interestRate : ℝ) : ℝ :=
  borrowed * (1 + interestRate)

/-- Proves that given a borrowed amount of $100 and an agreed increase of 10%, 
    the total amount to be returned is $110. -/
theorem borrowed_amount_with_interest : 
  totalAmount 100 0.1 = 110 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_amount_with_interest_l1845_184509


namespace NUMINAMATH_CALUDE_solve_equation_l1845_184511

theorem solve_equation (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1845_184511


namespace NUMINAMATH_CALUDE_desired_interest_percentage_l1845_184561

theorem desired_interest_percentage 
  (face_value : ℝ) 
  (dividend_rate : ℝ) 
  (market_value : ℝ) 
  (h1 : face_value = 56) 
  (h2 : dividend_rate = 0.09) 
  (h3 : market_value = 42) : 
  (dividend_rate * face_value) / market_value = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_desired_interest_percentage_l1845_184561


namespace NUMINAMATH_CALUDE_geometric_sequence_cos_ratio_l1845_184529

open Real

/-- Given an arithmetic sequence {a_n} with first term a₁ and common difference d,
    where 0 < d < 2π, if {cos a_n} forms a geometric sequence,
    then the common ratio of {cos a_n} is -1. -/
theorem geometric_sequence_cos_ratio
  (a₁ : ℝ) (d : ℝ) (h_d : 0 < d ∧ d < 2 * π)
  (h_geom : ∀ n : ℕ, n ≥ 1 → cos (a₁ + n * d) / cos (a₁ + (n - 1) * d) =
                           cos (a₁ + d) / cos a₁) :
  ∀ n : ℕ, n ≥ 1 → cos (a₁ + (n + 1) * d) / cos (a₁ + n * d) = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_cos_ratio_l1845_184529


namespace NUMINAMATH_CALUDE_opposite_face_of_A_is_F_l1845_184582

-- Define the set of labels
inductive Label
| A | B | C | D | E | F

-- Define the structure of a cube face
structure CubeFace where
  label : Label

-- Define the structure of a cube
structure Cube where
  faces : List CubeFace
  adjacent : Label → List Label

-- Define the property of being opposite faces
def isOpposite (cube : Cube) (l1 l2 : Label) : Prop :=
  l1 ∉ cube.adjacent l2 ∧ l2 ∉ cube.adjacent l1

-- Theorem statement
theorem opposite_face_of_A_is_F (cube : Cube) 
  (h1 : cube.faces.length = 6)
  (h2 : ∀ l : Label, l ∈ (cube.faces.map CubeFace.label))
  (h3 : cube.adjacent Label.A = [Label.B, Label.C, Label.D, Label.E]) :
  isOpposite cube Label.A Label.F :=
sorry

end NUMINAMATH_CALUDE_opposite_face_of_A_is_F_l1845_184582


namespace NUMINAMATH_CALUDE_largest_valid_n_l1845_184503

/-- A coloring of integers from 1 to 14 using two colors -/
def Coloring := Fin 14 → Bool

/-- Check if a coloring satisfies the condition for a given k -/
def valid_for_k (c : Coloring) (k : Nat) : Prop :=
  ∃ (i j i' j' : Fin 14),
    i < j ∧ j - i = k ∧ c i = c j ∧
    i' < j' ∧ j' - i' = k ∧ c i' ≠ c j'

/-- A coloring is valid up to n if it satisfies the condition for all k from 1 to n -/
def valid_coloring (c : Coloring) (n : Nat) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → valid_for_k c k

/-- The main theorem: 11 is the largest n for which a valid coloring exists -/
theorem largest_valid_n :
  (∃ c : Coloring, valid_coloring c 11) ∧
  (∀ c : Coloring, ¬valid_coloring c 12) := by
  sorry

end NUMINAMATH_CALUDE_largest_valid_n_l1845_184503


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1845_184546

theorem complex_magnitude_problem (z : ℂ) : z = (2 - Complex.I) / (1 + 2 * Complex.I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1845_184546


namespace NUMINAMATH_CALUDE_curve_transformation_l1845_184513

/-- Given a curve C in a plane rectangular coordinate system, 
    prove that its equation is 50x^2 + 72y^2 = 1 after an expansion transformation. -/
theorem curve_transformation (x y x' y' : ℝ) : 
  (x' = 5*x ∧ y' = 3*y) →  -- Transformation equations
  (2*x'^2 + 8*y'^2 = 1) →  -- Equation of transformed curve
  (50*x^2 + 72*y^2 = 1)    -- Equation of original curve C
  := by sorry

end NUMINAMATH_CALUDE_curve_transformation_l1845_184513


namespace NUMINAMATH_CALUDE_existence_of_solutions_l1845_184572

theorem existence_of_solutions (k : ℕ) (a : ℕ) (n : Fin k → ℕ) 
  (h1 : ∀ i, a > 0 ∧ n i > 0)
  (h2 : ∀ i j, i ≠ j → Nat.gcd (n i) (n j) = 1)
  (h3 : ∀ i, a ^ (n i) % (n i) = 1)
  (h4 : ∀ i, ¬(n i ∣ a - 1)) :
  ∃ (S : Finset ℕ), S.card ≥ 2^(k+1) - 2 ∧ 
    (∀ x ∈ S, x > 1 ∧ a^x % x = 1) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_solutions_l1845_184572


namespace NUMINAMATH_CALUDE_wheel_radius_l1845_184553

/-- The radius of a wheel given its circumference and number of revolutions --/
theorem wheel_radius (distance : ℝ) (revolutions : ℕ) (h : distance = 760.57 ∧ revolutions = 500) :
  ∃ (radius : ℝ), abs (radius - 0.242) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_wheel_radius_l1845_184553


namespace NUMINAMATH_CALUDE_molecular_weight_K2Cr2O7_is_296_l1845_184575

/-- The molecular weight of K2Cr2O7 in g/mole -/
def molecular_weight_K2Cr2O7 : ℝ := 296

/-- The number of moles given in the problem -/
def given_moles : ℝ := 4

/-- The total weight of the given moles in grams -/
def total_weight : ℝ := 1184

/-- Theorem stating that the molecular weight of K2Cr2O7 is 296 g/mole -/
theorem molecular_weight_K2Cr2O7_is_296 :
  molecular_weight_K2Cr2O7 = total_weight / given_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_K2Cr2O7_is_296_l1845_184575


namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l1845_184537

theorem sum_and_ratio_to_difference (x y : ℝ) : 
  x + y = 520 → x / y = 0.75 → y - x = 74 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l1845_184537


namespace NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l1845_184520

theorem arithmetic_sequence_remainder (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 3 →
  d = 8 →
  aₙ = 347 →
  aₙ = a₁ + (n - 1) * d →
  (n * (a₁ + aₙ) / 2) % 8 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l1845_184520


namespace NUMINAMATH_CALUDE_equation_one_solutions_l1845_184552

theorem equation_one_solutions (x : ℝ) :
  x - 2 = 4 * (x - 2)^2 ↔ x = 2 ∨ x = 9/4 := by
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l1845_184552


namespace NUMINAMATH_CALUDE_totalSavingsIs4440_l1845_184584

-- Define the employees and their properties
structure Employee where
  name : String
  hourlyRate : ℚ
  hoursPerDay : ℚ
  savingRate : ℚ

-- Define the constants
def daysPerWeek : ℚ := 5
def numWeeks : ℚ := 4

-- Define the list of employees
def employees : List Employee := [
  ⟨"Robby", 10, 10, 2/5⟩,
  ⟨"Jaylen", 10, 8, 3/5⟩,
  ⟨"Miranda", 10, 10, 1/2⟩,
  ⟨"Alex", 12, 6, 1/3⟩,
  ⟨"Beth", 15, 4, 1/4⟩,
  ⟨"Chris", 20, 3, 3/4⟩
]

-- Calculate weekly savings for an employee
def weeklySavings (e : Employee) : ℚ :=
  e.hourlyRate * e.hoursPerDay * daysPerWeek * e.savingRate

-- Calculate total savings for all employees over the given number of weeks
def totalSavings : ℚ :=
  (employees.map weeklySavings).sum * numWeeks

-- Theorem statement
theorem totalSavingsIs4440 : totalSavings = 4440 := by
  sorry

end NUMINAMATH_CALUDE_totalSavingsIs4440_l1845_184584


namespace NUMINAMATH_CALUDE_union_when_a_is_3_union_equals_B_iff_l1845_184569

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a^2 - 2}
def B : Set ℝ := Set.Ioo 1 5

-- Statement 1: When a = 3, A ∪ B = (1, 7]
theorem union_when_a_is_3 : 
  A 3 ∪ B = Set.Ioc 1 7 := by sorry

-- Statement 2: A ∪ B = B if and only if a ∈ (2, √7)
theorem union_equals_B_iff (a : ℝ) : 
  A a ∪ B = B ↔ a ∈ Set.Ioo 2 (Real.sqrt 7) := by sorry

end NUMINAMATH_CALUDE_union_when_a_is_3_union_equals_B_iff_l1845_184569


namespace NUMINAMATH_CALUDE_product_of_special_ratio_numbers_l1845_184538

theorem product_of_special_ratio_numbers (x y : ℝ) 
  (h : ∃ (k : ℝ), k > 0 ∧ x - y = k ∧ x + y = 2*k ∧ x^2 * y^2 = 18*k) : 
  x * y = 16 := by
sorry

end NUMINAMATH_CALUDE_product_of_special_ratio_numbers_l1845_184538


namespace NUMINAMATH_CALUDE_two_same_color_points_at_unit_distance_l1845_184556

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to color points
def colorPoint : Point → Color := sorry

-- Define a function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Theorem statement
theorem two_same_color_points_at_unit_distance :
  ∃ (p1 p2 : Point), colorPoint p1 = colorPoint p2 ∧ distance p1 p2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_two_same_color_points_at_unit_distance_l1845_184556


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l1845_184521

/-- A trapezoid with given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ

/-- The property that the midpoint segment length is half the difference of base lengths -/
def midpoint_property (t : Trapezoid) : Prop :=
  t.midpoint_segment = (t.longer_base - t.shorter_base) / 2

theorem trapezoid_shorter_base (t : Trapezoid) 
  (h1 : t.longer_base = 103)
  (h2 : t.midpoint_segment = 5)
  (h3 : midpoint_property t) :
  t.shorter_base = 93 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l1845_184521


namespace NUMINAMATH_CALUDE_Z_in_first_quadrant_l1845_184527

/-- A complex number is in the first quadrant if its real and imaginary parts are both positive. -/
def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

/-- The sum of two complex numbers (5+4i) and (-1+2i) -/
def Z : ℂ := Complex.mk 5 4 + Complex.mk (-1) 2

/-- Theorem: Z is located in the first quadrant of the complex plane -/
theorem Z_in_first_quadrant : is_in_first_quadrant Z := by
  sorry

end NUMINAMATH_CALUDE_Z_in_first_quadrant_l1845_184527


namespace NUMINAMATH_CALUDE_ngo_employees_proof_l1845_184585

/-- The number of literate employees in an NGO -/
def num_literate_employees : ℕ := 10

theorem ngo_employees_proof :
  let total_employees := num_literate_employees + 20
  let illiterate_wage_decrease := 300
  let total_wage_decrease := total_employees * 10
  illiterate_wage_decrease = total_wage_decrease →
  num_literate_employees = 10 := by
sorry

end NUMINAMATH_CALUDE_ngo_employees_proof_l1845_184585


namespace NUMINAMATH_CALUDE_root_product_simplification_l1845_184567

theorem root_product_simplification (a : ℝ) (ha : 0 < a) :
  (a ^ (1 / Real.sqrt a)) * (a ^ (1 / 3)) = a ^ (5 / 6) :=
by sorry

end NUMINAMATH_CALUDE_root_product_simplification_l1845_184567


namespace NUMINAMATH_CALUDE_ninth_grade_science_only_l1845_184596

/-- Represents the set of all ninth-grade students -/
def NinthGrade : Finset Nat := sorry

/-- Represents the set of students in the science class -/
def ScienceClass : Finset Nat := sorry

/-- Represents the set of students in the history class -/
def HistoryClass : Finset Nat := sorry

theorem ninth_grade_science_only :
  (NinthGrade.card = 120) →
  (ScienceClass.card = 85) →
  (HistoryClass.card = 75) →
  (NinthGrade = ScienceClass ∪ HistoryClass) →
  ((ScienceClass \ HistoryClass).card = 45) := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_science_only_l1845_184596


namespace NUMINAMATH_CALUDE_fano_plane_properties_l1845_184581

/-- A point in the Fano plane. -/
inductive Point
| P1 | P2 | P3 | P4 | P5 | P6 | P7

/-- A line in the Fano plane. -/
inductive Line
| L1 | L2 | L3 | L4 | L5 | L6 | L7

/-- The incidence relation between points and lines in the Fano plane. -/
def incidence : Point → Line → Prop
| Point.P1, Line.L1 => True
| Point.P1, Line.L2 => True
| Point.P1, Line.L3 => True
| Point.P2, Line.L1 => True
| Point.P2, Line.L4 => True
| Point.P2, Line.L5 => True
| Point.P3, Line.L1 => True
| Point.P3, Line.L6 => True
| Point.P3, Line.L7 => True
| Point.P4, Line.L2 => True
| Point.P4, Line.L4 => True
| Point.P4, Line.L6 => True
| Point.P5, Line.L2 => True
| Point.P5, Line.L5 => True
| Point.P5, Line.L7 => True
| Point.P6, Line.L3 => True
| Point.P6, Line.L4 => True
| Point.P6, Line.L7 => True
| Point.P7, Line.L3 => True
| Point.P7, Line.L5 => True
| Point.P7, Line.L6 => True
| _, _ => False

/-- The theorem stating that the Fano plane satisfies the required properties. -/
theorem fano_plane_properties :
  (∀ l : Line, ∃! (p1 p2 p3 : Point), p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    incidence p1 l ∧ incidence p2 l ∧ incidence p3 l ∧
    (∀ p : Point, incidence p l → p = p1 ∨ p = p2 ∨ p = p3)) ∧
  (∀ p : Point, ∃! (l1 l2 l3 : Line), l1 ≠ l2 ∧ l1 ≠ l3 ∧ l2 ≠ l3 ∧
    incidence p l1 ∧ incidence p l2 ∧ incidence p l3 ∧
    (∀ l : Line, incidence p l → l = l1 ∨ l = l2 ∨ l = l3)) :=
by sorry

end NUMINAMATH_CALUDE_fano_plane_properties_l1845_184581


namespace NUMINAMATH_CALUDE_nested_radical_twenty_l1845_184549

theorem nested_radical_twenty (x : ℝ) (h : x > 0) (eq : x = Real.sqrt (20 + x)) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_twenty_l1845_184549


namespace NUMINAMATH_CALUDE_hayleys_friends_l1845_184518

def total_stickers : ℕ := 72
def stickers_per_friend : ℕ := 8

theorem hayleys_friends :
  total_stickers / stickers_per_friend = 9 :=
by sorry

end NUMINAMATH_CALUDE_hayleys_friends_l1845_184518


namespace NUMINAMATH_CALUDE_remainder_theorem_l1845_184522

theorem remainder_theorem (P D E Q R M S C : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = E * M + S)
  (h3 : R < D)
  (h4 : S < E) :
  ∃ K, P = K * (D * E) + (S * D + R + C) ∧ S * D + R + C < D * E :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1845_184522


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_negative_three_l1845_184583

theorem no_linear_term_implies_m_equals_negative_three (m : ℝ) :
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x + 3) = a * x^2 + b) →
  m = -3 :=
sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_negative_three_l1845_184583


namespace NUMINAMATH_CALUDE_triangle_division_possible_l1845_184542

/-- Represents a part of the triangle -/
structure TrianglePart where
  numbers : List Nat
  area : Nat

/-- Represents the entire triangle -/
structure Triangle where
  parts : List TrianglePart

/-- The sum of numbers in a triangle part -/
def sumPart (part : TrianglePart) : Nat :=
  part.numbers.sum

/-- The total sum of all numbers in the triangle -/
def totalSum (triangle : Triangle) : Nat :=
  triangle.parts.map sumPart |>.sum

/-- Check if all parts have equal sums -/
def equalSums (triangle : Triangle) : Prop :=
  ∀ i j, i < triangle.parts.length → j < triangle.parts.length → 
    sumPart (triangle.parts.get ⟨i, by sorry⟩) = sumPart (triangle.parts.get ⟨j, by sorry⟩)

/-- Check if all parts have different areas -/
def differentAreas (triangle : Triangle) : Prop :=
  ∀ i j, i < triangle.parts.length → j < triangle.parts.length → i ≠ j → 
    (triangle.parts.get ⟨i, by sorry⟩).area ≠ (triangle.parts.get ⟨j, by sorry⟩).area

/-- The main theorem -/
theorem triangle_division_possible : 
  ∃ (t : Triangle), totalSum t = 63 ∧ t.parts.length = 3 ∧ equalSums t ∧ differentAreas t :=
sorry

end NUMINAMATH_CALUDE_triangle_division_possible_l1845_184542


namespace NUMINAMATH_CALUDE_toy_sale_analysis_l1845_184598

-- Define the cost price
def cost_price : ℝ := 20

-- Define the maximum profit percentage
def max_profit_percentage : ℝ := 0.3

-- Define the linear relationship between weekly sales volume and selling price
def sales_volume (x : ℝ) : ℝ := -10 * x + 300

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_volume x

-- Theorem statement
theorem toy_sale_analysis :
  -- Part 1: Verify the linear relationship
  (sales_volume 22 = 80 ∧ sales_volume 24 = 60) ∧
  -- Part 2: Verify the selling price for 210 yuan profit
  (∃ x : ℝ, x ≤ cost_price * (1 + max_profit_percentage) ∧ profit x = 210 ∧ x = 23) ∧
  -- Part 3: Verify the maximum profit
  (∃ x : ℝ, x = 25 ∧ profit x = 250 ∧ ∀ y : ℝ, profit y ≤ profit x) := by
  sorry


end NUMINAMATH_CALUDE_toy_sale_analysis_l1845_184598


namespace NUMINAMATH_CALUDE_remainder_problem_l1845_184586

theorem remainder_problem : (123456789012 : ℕ) % 252 = 228 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1845_184586


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1845_184510

theorem solution_set_of_inequality (x : ℝ) : 
  (x - 1) / (x + 3) < 0 ↔ -3 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1845_184510


namespace NUMINAMATH_CALUDE_area_between_graphs_l1845_184599

-- Define the functions
def f (x : ℝ) : ℝ := |2 * x| - 3
def g (x : ℝ) : ℝ := |x|

-- Define the area enclosed by the graphs
def enclosed_area : ℝ := 9

-- Theorem statement
theorem area_between_graphs :
  (∃ (a b : ℝ), a < b ∧
    (∀ x ∈ Set.Icc a b, f x ≠ g x) ∧
    (∀ x ∈ Set.Ioi b, f x = g x) ∧
    (∀ x ∈ Set.Iio a, f x = g x)) →
  (∫ (x : ℝ) in Set.Icc (-3) 3, |f x - g x|) = enclosed_area :=
sorry

end NUMINAMATH_CALUDE_area_between_graphs_l1845_184599


namespace NUMINAMATH_CALUDE_expression_simplification_l1845_184565

theorem expression_simplification (x y : ℝ) (h : y ≠ 0) :
  ((x + 3*y)^2 - (x + y)*(x - y)) / (2*y) = 3*x + 5*y := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1845_184565


namespace NUMINAMATH_CALUDE_system_solutions_l1845_184588

def equation1 (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

def equation2 (x y : ℝ) : ℝ := 
  (|x - 1| + |y - 1|) * (|x - 2| + |y - 2|) * (|x - 3| + |y - 4|)

theorem system_solutions :
  ∀ (x y : ℝ), 
    (equation1 x = 0 ∧ equation2 x y = 0) ↔ 
    ((x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 3 ∧ y = 4)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l1845_184588


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l1845_184528

def M : Set ℝ := {x | |x| < 3}
def N : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

theorem set_intersection_theorem : M ∩ N = {x | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l1845_184528


namespace NUMINAMATH_CALUDE_simplify_exponent_division_l1845_184504

theorem simplify_exponent_division (x : ℝ) (h : x ≠ 0) : x^6 / x^2 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponent_division_l1845_184504


namespace NUMINAMATH_CALUDE_water_bottle_capacity_l1845_184558

/-- The capacity of a water bottle in milliliters -/
def bottle_capacity : ℕ := 12800

/-- The volume of the smaller cup in milliliters -/
def small_cup : ℕ := 250

/-- The volume of the larger cup in milliliters -/
def large_cup : ℕ := 600

/-- The number of times water is scooped with the smaller cup -/
def small_cup_scoops : ℕ := 20

/-- The number of times water is scooped with the larger cup -/
def large_cup_scoops : ℕ := 13

/-- Conversion factor from milliliters to liters -/
def ml_to_l : ℚ := 1 / 1000

theorem water_bottle_capacity :
  (bottle_capacity : ℚ) * ml_to_l = 12.8 ∧
  bottle_capacity = small_cup * small_cup_scoops + large_cup * large_cup_scoops :=
sorry

end NUMINAMATH_CALUDE_water_bottle_capacity_l1845_184558


namespace NUMINAMATH_CALUDE_smallest_upper_bound_for_sum_of_square_roots_l1845_184523

theorem smallest_upper_bound_for_sum_of_square_roots :
  ∃ (M : ℝ), (∀ (x y z w : ℝ), x > 0 → y > 0 → z > 0 → w > 0 →
    Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
    Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) ≤ M) ∧
  (M = 4 / Real.sqrt 3) ∧
  (∀ (M' : ℝ), M' < M →
    ∃ (x y z w : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
      Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
      Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) > M') :=
by sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_for_sum_of_square_roots_l1845_184523


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l1845_184548

/-- The total capacity of a water tank in liters. -/
def tank_capacity : ℝ := 120

/-- The difference in water volume between 70% full and 40% full, in liters. -/
def volume_difference : ℝ := 36

/-- Theorem stating the total capacity of the tank given the volume difference between two fill levels. -/
theorem tank_capacity_proof :
  tank_capacity * (0.7 - 0.4) = volume_difference :=
sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l1845_184548


namespace NUMINAMATH_CALUDE_seongjun_has_500_ttakji_l1845_184566

/-- The number of ttakji Seongjun has -/
def seongjun_ttakji : ℕ := sorry

/-- The number of ttakji Seunga has -/
def seunga_ttakji : ℕ := 100

/-- The relationship between Seongjun's and Seunga's ttakji -/
axiom ttakji_relationship : (3 / 4 : ℚ) * seongjun_ttakji - 25 = 7 * (seunga_ttakji - 50)

theorem seongjun_has_500_ttakji : seongjun_ttakji = 500 := by sorry

end NUMINAMATH_CALUDE_seongjun_has_500_ttakji_l1845_184566


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1755_l1845_184555

theorem largest_prime_factor_of_1755 : ∃ p : Nat, Nat.Prime p ∧ p ∣ 1755 ∧ ∀ q : Nat, Nat.Prime q → q ∣ 1755 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1755_l1845_184555


namespace NUMINAMATH_CALUDE_one_is_not_prime_and_not_composite_l1845_184516

-- Define the properties of natural numbers based on their divisors
def has_only_one_divisor (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

-- Theorem to prove
theorem one_is_not_prime_and_not_composite : 
  ¬(is_prime 1 ∧ ¬is_composite 1) :=
sorry

end NUMINAMATH_CALUDE_one_is_not_prime_and_not_composite_l1845_184516


namespace NUMINAMATH_CALUDE_a_investment_is_400_l1845_184571

/-- Represents the investment scenario described in the problem -/
structure InvestmentScenario where
  a_investment : ℝ
  b_investment : ℝ
  total_profit : ℝ
  a_profit : ℝ
  b_investment_time : ℝ
  total_time : ℝ

/-- Theorem stating that given the conditions, A's investment was $400 -/
theorem a_investment_is_400 (scenario : InvestmentScenario) 
  (h1 : scenario.b_investment = 200)
  (h2 : scenario.total_profit = 100)
  (h3 : scenario.a_profit = 80)
  (h4 : scenario.b_investment_time = 6)
  (h5 : scenario.total_time = 12)
  (h6 : scenario.a_investment * scenario.total_time / 
        (scenario.b_investment * scenario.b_investment_time) = 
        scenario.a_profit / (scenario.total_profit - scenario.a_profit)) :
  scenario.a_investment = 400 := by
  sorry


end NUMINAMATH_CALUDE_a_investment_is_400_l1845_184571


namespace NUMINAMATH_CALUDE_water_depth_l1845_184532

theorem water_depth (ron_height dean_height water_depth : ℕ) : 
  ron_height = 13 →
  dean_height = ron_height + 4 →
  water_depth = 15 * dean_height →
  water_depth = 255 := by
sorry

end NUMINAMATH_CALUDE_water_depth_l1845_184532


namespace NUMINAMATH_CALUDE_pyramid_vertex_on_face_plane_l1845_184594

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parallelepiped in 3D space -/
structure Parallelepiped where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Represents a triangular pyramid in 3D space -/
structure TriangularPyramid where
  v1 : Point3D
  v2 : Point3D
  v3 : Point3D
  v4 : Point3D

/-- Checks if a point lies on a plane defined by three other points -/
def pointLiesOnPlane (p : Point3D) (p1 p2 p3 : Point3D) : Prop := sorry

/-- Main theorem: Each vertex of one pyramid lies on a face plane of the other pyramid -/
theorem pyramid_vertex_on_face_plane (p : Parallelepiped) : 
  let pyramid1 := TriangularPyramid.mk p.A p.B p.D p.D₁
  let pyramid2 := TriangularPyramid.mk p.A₁ p.B₁ p.C₁ p.C
  (pointLiesOnPlane pyramid1.v1 pyramid2.v1 pyramid2.v3 pyramid2.v4 ∧
   pointLiesOnPlane pyramid1.v2 pyramid2.v2 pyramid2.v3 pyramid2.v4 ∧
   pointLiesOnPlane pyramid1.v3 pyramid2.v1 pyramid2.v2 pyramid2.v4 ∧
   pointLiesOnPlane pyramid1.v4 pyramid2.v1 pyramid2.v2 pyramid2.v3) ∧
  (pointLiesOnPlane pyramid2.v1 pyramid1.v1 pyramid1.v3 pyramid1.v4 ∧
   pointLiesOnPlane pyramid2.v2 pyramid1.v2 pyramid1.v3 pyramid1.v4 ∧
   pointLiesOnPlane pyramid2.v3 pyramid1.v1 pyramid1.v2 pyramid1.v4 ∧
   pointLiesOnPlane pyramid2.v4 pyramid1.v1 pyramid1.v2 pyramid1.v3) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_vertex_on_face_plane_l1845_184594


namespace NUMINAMATH_CALUDE_midpoint_sum_invariant_l1845_184591

/-- Represents a polygon with n vertices in the Cartesian plane -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Creates a new polygon from the midpoints of the sides of the given polygon -/
def midpointPolygon {n : ℕ} (p : Polygon n) : Polygon n := sorry

/-- Sums the x-coordinates of a polygon's vertices -/
def sumXCoordinates {n : ℕ} (p : Polygon n) : ℝ := sorry

theorem midpoint_sum_invariant (p₁ : Polygon 200) 
  (h : sumXCoordinates p₁ = 4018) :
  let p₂ := midpointPolygon p₁
  let p₃ := midpointPolygon p₂
  let p₄ := midpointPolygon p₃
  sumXCoordinates p₄ = 4018 := by sorry

end NUMINAMATH_CALUDE_midpoint_sum_invariant_l1845_184591


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l1845_184578

theorem rectangle_dimension_change (L B : ℝ) (x : ℝ) (h_positive : L > 0 ∧ B > 0) :
  (1.20 * L) * (B * (1 - x / 100)) = 1.04 * (L * B) → x = 40 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l1845_184578


namespace NUMINAMATH_CALUDE_valid_numbers_count_l1845_184500

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    b = 2 * c ∧
    b = (a + c) / 2

theorem valid_numbers_count :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, is_valid_number n) ∧
    s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_count_l1845_184500


namespace NUMINAMATH_CALUDE_right_triangle_sides_l1845_184535

theorem right_triangle_sides (a d : ℝ) (k : ℕ) (h1 : a > 0) (h2 : d > 0) (h3 : k > 0) :
  a = 3 ∧ d = 1 ∧ k = 2 →
  (a + k * d) ^ 2 = a ^ 2 + (a + d) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l1845_184535


namespace NUMINAMATH_CALUDE_optimal_price_for_profit_l1845_184563

-- Define the linear relationship between price and sales volume
def sales_volume (x : ℝ) : ℝ := -10 * x + 400

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 10) * sales_volume x

-- State the theorem
theorem optimal_price_for_profit :
  ∃ x : ℝ, 
    x > 0 ∧ 
    profit x = 2160 ∧ 
    ∀ y : ℝ, y > 0 ∧ profit y = 2160 → sales_volume x ≤ sales_volume y := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_for_profit_l1845_184563


namespace NUMINAMATH_CALUDE_first_day_price_is_four_l1845_184573

/-- Represents the sales data for a pen store over three days -/
structure PenSales where
  price1 : ℝ  -- Price per pen on the first day
  quantity1 : ℝ  -- Number of pens sold on the first day

/-- The revenue is the same for all three days given the pricing and quantity changes -/
def sameRevenue (s : PenSales) : Prop :=
  s.price1 * s.quantity1 = (s.price1 - 1) * (s.quantity1 + 100) ∧
  s.price1 * s.quantity1 = (s.price1 + 2) * (s.quantity1 - 100)

/-- The price on the first day is 4 yuan -/
theorem first_day_price_is_four (s : PenSales) (h : sameRevenue s) : s.price1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_day_price_is_four_l1845_184573


namespace NUMINAMATH_CALUDE_student_ratio_proof_l1845_184533

theorem student_ratio_proof (m n : ℕ) (a b : ℝ) (α β : ℝ) 
  (h1 : α = 3 / 4)
  (h2 : β = 19 / 20)
  (h3 : a = α * b)
  (h4 : a = β * (a * m + b * n) / (m + n)) :
  m / n = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_student_ratio_proof_l1845_184533


namespace NUMINAMATH_CALUDE_fish_to_rice_value_l1845_184526

/-- Represents the exchange rate between fish and bread -/
def fish_to_bread : ℚ := 3 / 5

/-- Represents the exchange rate between bread and rice -/
def bread_to_rice : ℚ := 5 / 2

/-- Theorem stating that one fish is worth 3/2 bags of rice -/
theorem fish_to_rice_value : 
  (fish_to_bread * bread_to_rice)⁻¹ = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_fish_to_rice_value_l1845_184526


namespace NUMINAMATH_CALUDE_john_writing_speed_l1845_184547

/-- The number of books John writes -/
def num_books : ℕ := 3

/-- The number of pages in each book -/
def pages_per_book : ℕ := 400

/-- The number of days it takes John to write the books -/
def total_days : ℕ := 60

/-- The number of pages John writes per day -/
def pages_per_day : ℕ := (num_books * pages_per_book) / total_days

theorem john_writing_speed : pages_per_day = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_writing_speed_l1845_184547


namespace NUMINAMATH_CALUDE_crow_eating_time_l1845_184514

/-- Given a constant eating rate where 1/5 of the nuts are eaten in 8 hours,
    prove that it takes 10 hours to eat 1/4 of the nuts. -/
theorem crow_eating_time (eating_rate : ℝ → ℝ) (h1 : eating_rate (8 : ℝ) = 1/5) 
    (h2 : ∀ t1 t2 : ℝ, eating_rate (t1 + t2) = eating_rate t1 + eating_rate t2) : 
    ∃ t : ℝ, eating_rate t = 1/4 ∧ t = 10 := by
  sorry


end NUMINAMATH_CALUDE_crow_eating_time_l1845_184514


namespace NUMINAMATH_CALUDE_convex_polygon_with_arithmetic_angles_l1845_184531

/-- A convex polygon with interior angles forming an arithmetic sequence,
    where the smallest angle is 100° and the largest angle is 140°, has exactly 6 sides. -/
theorem convex_polygon_with_arithmetic_angles (n : ℕ) : 
  n ≥ 3 → -- convex polygon has at least 3 sides
  (∃ (a d : ℝ), 
    a = 100 ∧ -- smallest angle
    a + (n - 1) * d = 140 ∧ -- largest angle
    ∀ i : ℕ, i < n → a + i * d ≥ 0 ∧ a + i * d ≤ 180) → -- all angles are between 0° and 180°
  (n : ℝ) * (100 + 140) / 2 = 180 * (n - 2) →
  n = 6 :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_with_arithmetic_angles_l1845_184531


namespace NUMINAMATH_CALUDE_symmetry_condition_range_on_interval_range_positive_l1845_184502

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2 - a) * x - 2 * a

-- Theorem for symmetry condition
theorem symmetry_condition (a : ℝ) :
  (∀ x : ℝ, f a (1 + x) = f a (1 - x)) → a = 4 :=
sorry

-- Theorem for range on [0,4] when a = 4
theorem range_on_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → -9 ≤ f 4 x ∧ f 4 x ≤ -5) ∧
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 4 ∧ f 4 x = -9 ∧ f 4 y = -5) :=
sorry

-- Theorem for the range of x when f(x) > 0
theorem range_positive (a : ℝ) (x : ℝ) :
  (a = -2 → (f a x > 0 ↔ x ≠ -2)) ∧
  (a > -2 → (f a x > 0 ↔ x < -2 ∨ x > a)) ∧
  (a < -2 → (f a x > 0 ↔ -2 < x ∧ x < a)) :=
sorry

end NUMINAMATH_CALUDE_symmetry_condition_range_on_interval_range_positive_l1845_184502


namespace NUMINAMATH_CALUDE_factor_expression_l1845_184536

theorem factor_expression (x : ℝ) : x * (x + 3) + (x + 3) = (x + 1) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1845_184536


namespace NUMINAMATH_CALUDE_function_is_identity_l1845_184562

/-- A function satisfying the given functional equation for all positive real numbers -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (z + 1) * f (x + y) = f (x * f z + y) + f (y * f z + x)

/-- The main theorem: if f satisfies the equation, then f(x) = x for all positive real numbers -/
theorem function_is_identity {f : ℝ → ℝ} (hf : SatisfiesEquation f) :
    ∀ x : ℝ, x > 0 → f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_is_identity_l1845_184562


namespace NUMINAMATH_CALUDE_probability_opposite_corner_is_one_third_l1845_184592

/-- Represents a cube with its properties -/
structure Cube where
  vertices : Fin 8
  faces : Fin 6

/-- Represents the ant's position on the cube -/
inductive Position
  | Corner : Fin 8 → Position

/-- Represents a single move of the ant -/
def Move : Type := Position → Position

/-- The probability of the ant ending at the diagonally opposite corner after two moves -/
def probability_opposite_corner (c : Cube) : ℚ :=
  1/3

/-- Theorem stating that the probability of ending at the diagonally opposite corner is 1/3 -/
theorem probability_opposite_corner_is_one_third (c : Cube) :
  probability_opposite_corner c = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_opposite_corner_is_one_third_l1845_184592


namespace NUMINAMATH_CALUDE_scientific_notation_32000000_l1845_184508

theorem scientific_notation_32000000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 32000000 = a * (10 : ℝ) ^ n ∧ a = 3.2 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_32000000_l1845_184508
