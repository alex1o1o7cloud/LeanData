import Mathlib

namespace NUMINAMATH_CALUDE_reading_book_cost_is_12_l1653_165316

/-- The cost of a reading book given the total amount available and number of students -/
def reading_book_cost (total_amount : ℕ) (num_students : ℕ) : ℚ :=
  (total_amount : ℚ) / (num_students : ℚ)

/-- Theorem: The cost of each reading book is $12 -/
theorem reading_book_cost_is_12 :
  reading_book_cost 360 30 = 12 := by
  sorry

end NUMINAMATH_CALUDE_reading_book_cost_is_12_l1653_165316


namespace NUMINAMATH_CALUDE_no_double_application_function_l1653_165364

theorem no_double_application_function :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 2019 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l1653_165364


namespace NUMINAMATH_CALUDE_cube_configurations_l1653_165332

/-- The group of rotations around axes through midpoints of opposite edges of a 2×2×2 cube -/
def EdgeRotationGroup : Type := Unit

/-- The order of the EdgeRotationGroup -/
def EdgeRotationGroupOrder : ℕ := 7

/-- The number of fixed configurations under the identity rotation -/
def FixedConfigurationsIdentity : ℕ := 56

/-- The number of fixed configurations under each edge rotation -/
def FixedConfigurationsEdge : ℕ := 0

/-- The number of edge rotations -/
def NumEdgeRotations : ℕ := 6

/-- Burnside's Lemma applied to the cube configuration problem -/
theorem cube_configurations (g : EdgeRotationGroup) :
  (FixedConfigurationsIdentity + NumEdgeRotations * FixedConfigurationsEdge) / EdgeRotationGroupOrder = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_configurations_l1653_165332


namespace NUMINAMATH_CALUDE_neil_cookies_l1653_165328

theorem neil_cookies (total : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  remaining = 12 ∧ given_away = (2 : ℕ) * total / 5 ∧ remaining = (3 : ℕ) * total / 5 → total = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_neil_cookies_l1653_165328


namespace NUMINAMATH_CALUDE_min_sum_squares_l1653_165306

def f (x : ℝ) := |x + 1| - |x - 4|

def m₀ : ℝ := 5

theorem min_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 3 * a + 4 * b + 5 * c = m₀) :
  a^2 + b^2 + c^2 ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1653_165306


namespace NUMINAMATH_CALUDE_sum_of_roots_is_18_l1653_165329

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the symmetry property of g
def is_symmetric_about_3 (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

-- Define the property of having exactly six distinct real roots
def has_six_distinct_roots (g : ℝ → ℝ) : Prop :=
  ∃ (a b c d e f : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                        d ≠ e ∧ d ≠ f ∧
                        e ≠ f) ∧
  (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0 ∧ g e = 0 ∧ g f = 0) ∧
  (∀ x : ℝ, g x = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f))

-- Theorem statement
theorem sum_of_roots_is_18 (g : ℝ → ℝ) 
  (h1 : is_symmetric_about_3 g) 
  (h2 : has_six_distinct_roots g) :
  ∃ (a b c d e f : ℝ), 
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0 ∧ g e = 0 ∧ g f = 0) ∧
    (a + b + c + d + e + f = 18) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_18_l1653_165329


namespace NUMINAMATH_CALUDE_hexagon_area_error_l1653_165372

/-- If there's an 8% error in excess while measuring the sides of a hexagon, 
    the percentage of error in the estimated area is 16.64%. -/
theorem hexagon_area_error (s : ℝ) (h : s > 0) : 
  let true_area := (3 * Real.sqrt 3 / 2) * s^2
  let measured_side := 1.08 * s
  let estimated_area := (3 * Real.sqrt 3 / 2) * measured_side^2
  (estimated_area - true_area) / true_area * 100 = 16.64 := by
sorry

end NUMINAMATH_CALUDE_hexagon_area_error_l1653_165372


namespace NUMINAMATH_CALUDE_inequality_proof_l1653_165351

theorem inequality_proof (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  x^3 + x*y^2 + 2*x*y ≤ 2*x^2*y + x^2 + x + y :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1653_165351


namespace NUMINAMATH_CALUDE_haley_seeds_l1653_165330

/-- The number of seeds Haley planted in the big garden -/
def big_garden_seeds : ℕ := 35

/-- The number of small gardens Haley had -/
def small_gardens : ℕ := 7

/-- The number of seeds Haley planted in each small garden -/
def seeds_per_small_garden : ℕ := 3

/-- The total number of seeds Haley started with -/
def total_seeds : ℕ := big_garden_seeds + small_gardens * seeds_per_small_garden

theorem haley_seeds : total_seeds = 56 := by
  sorry

end NUMINAMATH_CALUDE_haley_seeds_l1653_165330


namespace NUMINAMATH_CALUDE_dan_found_two_dimes_l1653_165310

/-- The number of dimes Dan found -/
def dimes_found (barry_dimes dan_initial_dimes dan_final_dimes : ℕ) : ℕ :=
  dan_final_dimes - dan_initial_dimes

theorem dan_found_two_dimes :
  ∀ (barry_dimes dan_initial_dimes dan_final_dimes : ℕ),
    barry_dimes = 100 →
    dan_initial_dimes = barry_dimes / 2 →
    dan_final_dimes = 52 →
    dimes_found barry_dimes dan_initial_dimes dan_final_dimes = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_dan_found_two_dimes_l1653_165310


namespace NUMINAMATH_CALUDE_perfect_squares_condition_l1653_165377

theorem perfect_squares_condition (k : ℤ) : 
  (∃ n : ℤ, k + 1 = n^2) ∧ (∃ m : ℤ, 16*k + 1 = m^2) ↔ k = 0 ∨ k = 3 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_condition_l1653_165377


namespace NUMINAMATH_CALUDE_max_value_fraction_l1653_165356

theorem max_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -3 ∧ 1 ≤ y' ∧ y' ≤ 3 → (x' + 2*y') / x' ≤ (x + 2*y) / x) →
  (x + 2*y) / x = -1/5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1653_165356


namespace NUMINAMATH_CALUDE_number_line_movement_1_number_line_movement_2_number_line_movement_general_absolute_value_equality_l1653_165301

-- Problem 1
theorem number_line_movement_1 (A B : ℝ) :
  A = -2 → B = A + 5 → B = 3 ∧ |B - A| = 5 := by sorry

-- Problem 2
theorem number_line_movement_2 (A B : ℝ) :
  A = 5 → B = A - 4 + 7 → B = 8 ∧ |B - A| = 3 := by sorry

-- Problem 3
theorem number_line_movement_general (a b c A B : ℝ) :
  A = a → B = A + b - c → B = a + b - c ∧ |B - A| = |b - c| := by sorry

-- Problem 4
theorem absolute_value_equality (x : ℝ) :
  |x + 1| = |x - 2| ↔ x = 1/2 := by sorry

end NUMINAMATH_CALUDE_number_line_movement_1_number_line_movement_2_number_line_movement_general_absolute_value_equality_l1653_165301


namespace NUMINAMATH_CALUDE_problem_statement_l1653_165391

theorem problem_statement (a b : ℝ) (h : a + b = 1) :
  (a^3 + b^3 ≥ 1/4) ∧
  (∃ x : ℝ, |x - a| + |x - b| ≤ 5 → 0 ≤ 2*a + 3*b ∧ 2*a + 3*b ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1653_165391


namespace NUMINAMATH_CALUDE_burger_distribution_theorem_l1653_165334

/-- Represents the burger distribution problem --/
def burger_distribution (total_burgers : ℕ) (num_friends : ℕ) (slices_per_burger : ℕ) 
  (slices_friend3 : ℕ) (slices_friend4 : ℕ) (slices_era : ℕ) : Prop :=
  let total_slices := total_burgers * slices_per_burger
  let slices_for_friends12 := total_slices - (slices_friend3 + slices_friend4 + slices_era)
  slices_for_friends12 = 3

/-- Theorem stating that under the given conditions, the first and second friends get 3 slices combined --/
theorem burger_distribution_theorem : 
  burger_distribution 5 4 2 3 3 1 := by sorry

end NUMINAMATH_CALUDE_burger_distribution_theorem_l1653_165334


namespace NUMINAMATH_CALUDE_chess_tournament_principled_trios_l1653_165384

/-- Represents the number of chess players in the tournament -/
def n : ℕ := 2017

/-- Defines a principled trio of chess players -/
def PrincipledTrio (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ A ≤ n ∧ B ≤ n ∧ C ≤ n

/-- Calculates the maximum number of principled trios for an odd number of players -/
def max_principled_trios_odd (k : ℕ) : ℕ := (k^3 - k) / 24

/-- The maximum number of principled trios in the tournament -/
def max_principled_trios : ℕ := max_principled_trios_odd n

theorem chess_tournament_principled_trios :
  max_principled_trios = 341606288 :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_principled_trios_l1653_165384


namespace NUMINAMATH_CALUDE_quadratic_trinomial_prime_square_solution_l1653_165359

/-- A quadratic trinomial function -/
def f (x : ℤ) : ℤ := 2 * x^2 - x - 36

/-- Predicate to check if a number is prime -/
def is_prime (p : ℕ) : Prop := Nat.Prime p

/-- The main theorem statement -/
theorem quadratic_trinomial_prime_square_solution :
  ∃! x : ℤ, ∃ p : ℕ, is_prime p ∧ f x = p^2 ∧ x = 13 := by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_prime_square_solution_l1653_165359


namespace NUMINAMATH_CALUDE_angle_BDC_value_l1653_165392

-- Define the angles in degrees
def angle_ABD : ℝ := 118
def angle_BCD : ℝ := 82

-- Define the theorem
theorem angle_BDC_value :
  ∀ (angle_BDC : ℝ),
  -- ABC is a straight line (implied by the exterior angle theorem)
  angle_ABD = angle_BCD + angle_BDC →
  angle_BDC = 36 := by
sorry

end NUMINAMATH_CALUDE_angle_BDC_value_l1653_165392


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1653_165381

theorem cubic_equation_roots (m : ℝ) : 
  (m = 3 ∨ m = -2) → 
  ∃ (z₁ z₂ z₃ : ℝ), 
    z₁ = -1 ∧ z₂ = -3 ∧ z₃ = 4 ∧
    ∀ (z : ℝ), z^3 - (m^2 - m + 7) * z - (3 * m^2 - 3 * m - 6) = 0 ↔ 
      (z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1653_165381


namespace NUMINAMATH_CALUDE_even_factors_count_l1653_165385

def n : ℕ := 2^3 * 3^2 * 7^3 * 5^1

/-- The number of even natural-number factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem even_factors_count : num_even_factors n = 72 := by sorry

end NUMINAMATH_CALUDE_even_factors_count_l1653_165385


namespace NUMINAMATH_CALUDE_initial_balloons_l1653_165336

theorem initial_balloons (x : ℕ) : 
  Odd x ∧ 
  (x / 3 : ℚ) + 10 = 45 → 
  x = 105 := by
sorry

end NUMINAMATH_CALUDE_initial_balloons_l1653_165336


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1653_165355

theorem quadratic_form_sum (b c : ℝ) : 
  (∀ x, x^2 - 12*x + 49 = (x + b)^2 + c) → b + c = 7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1653_165355


namespace NUMINAMATH_CALUDE_count_single_colored_face_for_given_cube_l1653_165313

/-- Represents a cube cut in half and then into smaller cubes --/
structure CutCube where
  half_size : ℕ -- The number of small cubes along one edge of a half
  total_small_cubes : ℕ -- Total number of small cubes in each half

/-- Calculates the number of small cubes with only one colored face --/
def count_single_colored_face (c : CutCube) : ℕ :=
  4 * (c.half_size - 2) * (c.half_size - 2) * 2

/-- The theorem to be proved --/
theorem count_single_colored_face_for_given_cube :
  ∃ (c : CutCube), c.half_size = 4 ∧ c.total_small_cubes = 64 ∧ count_single_colored_face c = 32 :=
sorry

end NUMINAMATH_CALUDE_count_single_colored_face_for_given_cube_l1653_165313


namespace NUMINAMATH_CALUDE_fraction_simplification_l1653_165357

theorem fraction_simplification (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 4) * (x - 2)) := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1653_165357


namespace NUMINAMATH_CALUDE_jason_music_store_spending_l1653_165305

/-- The price of the flute Jason bought -/
def flute_price : ℚ := 142.46

/-- The price of the music stand Jason bought -/
def stand_price : ℚ := 8.89

/-- The price of the song book Jason bought -/
def book_price : ℚ := 7

/-- The total amount Jason spent at the music store -/
def total_spent : ℚ := flute_price + stand_price + book_price

/-- Theorem stating that the total amount Jason spent is $158.35 -/
theorem jason_music_store_spending :
  total_spent = 158.35 := by sorry

end NUMINAMATH_CALUDE_jason_music_store_spending_l1653_165305


namespace NUMINAMATH_CALUDE_units_digit_problem_l1653_165387

theorem units_digit_problem : (7 * 13 * 1957 - 7^4) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1653_165387


namespace NUMINAMATH_CALUDE_trapezoid_height_l1653_165395

/-- Represents a trapezoid with height x, bases 3x and 5x, and area 40 -/
structure Trapezoid where
  x : ℝ
  base1 : ℝ := 3 * x
  base2 : ℝ := 5 * x
  area : ℝ := 40

/-- The height of a trapezoid with the given properties is √10 -/
theorem trapezoid_height (t : Trapezoid) : t.x = Real.sqrt 10 := by
  sorry

#check trapezoid_height

end NUMINAMATH_CALUDE_trapezoid_height_l1653_165395


namespace NUMINAMATH_CALUDE_tan_alpha_equals_two_implies_expression_equals_three_l1653_165304

theorem tan_alpha_equals_two_implies_expression_equals_three (α : Real) 
  (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (2 * Real.sin α - 3 * Real.cos α) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_two_implies_expression_equals_three_l1653_165304


namespace NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_l1653_165360

/-- A decagon is a polygon with 10 vertices -/
def Decagon : Type := Unit

/-- The number of vertices in a decagon -/
def num_vertices (d : Decagon) : ℕ := 10

/-- The number of pairs of adjacent vertices in a decagon -/
def num_adjacent_pairs (d : Decagon) : ℕ := 20

/-- The total number of ways to choose 2 distinct vertices from a decagon -/
def total_vertex_pairs (d : Decagon) : ℕ := (num_vertices d).choose 2

/-- The probability of choosing two adjacent vertices in a decagon -/
def prob_adjacent_vertices (d : Decagon) : ℚ :=
  (num_adjacent_pairs d : ℚ) / (total_vertex_pairs d : ℚ)

theorem prob_adjacent_vertices_decagon :
  ∀ d : Decagon, prob_adjacent_vertices d = 4/9 := by sorry

end NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_l1653_165360


namespace NUMINAMATH_CALUDE_leonards_age_l1653_165398

theorem leonards_age (nina jerome leonard : ℕ) 
  (h1 : leonard = nina - 4)
  (h2 : nina = jerome / 2)
  (h3 : nina + jerome + leonard = 36) : 
  leonard = 6 := by
sorry

end NUMINAMATH_CALUDE_leonards_age_l1653_165398


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1653_165339

/-- 
A point P with coordinates (3+a, a-5) lies on the x-axis in a Cartesian coordinate system.
Prove that a = 5.
-/
theorem point_on_x_axis (a : ℝ) : (a - 5 = 0) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1653_165339


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1653_165378

theorem diophantine_equation_solution :
  ∀ x y z : ℕ,
    x ≤ y →
    x^2 + y^2 = 3 * 2016^z + 77 →
    ((x = 4 ∧ y = 8 ∧ z = 0) ∨
     (x = 14 ∧ y = 49 ∧ z = 1) ∨
     (x = 35 ∧ y = 70 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1653_165378


namespace NUMINAMATH_CALUDE_committee_probability_l1653_165337

def totalStudents : ℕ := 18
def numBoys : ℕ := 8
def numGirls : ℕ := 10
def committeeSize : ℕ := 4

theorem committee_probability : 
  let totalCommittees := Nat.choose totalStudents committeeSize
  let allBoysCommittees := Nat.choose numBoys committeeSize
  let allGirlsCommittees := Nat.choose numGirls committeeSize
  let probabilityAtLeastOneBoyOneGirl := 1 - (allBoysCommittees + allGirlsCommittees : ℚ) / totalCommittees
  probabilityAtLeastOneBoyOneGirl = 139 / 153 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l1653_165337


namespace NUMINAMATH_CALUDE_friend_walking_problem_l1653_165331

/-- Two friends walking on a trail problem -/
theorem friend_walking_problem (trail_length : ℝ) (meeting_distance : ℝ) 
  (h1 : trail_length = 43)
  (h2 : meeting_distance = 23)
  (h3 : meeting_distance < trail_length) :
  let rate_ratio := meeting_distance / (trail_length - meeting_distance)
  (rate_ratio - 1) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_friend_walking_problem_l1653_165331


namespace NUMINAMATH_CALUDE_combined_cost_theorem_l1653_165300

def wallet_cost : ℕ := 22
def purse_cost : ℕ := 4 * wallet_cost - 3

theorem combined_cost_theorem : wallet_cost + purse_cost = 107 := by
  sorry

end NUMINAMATH_CALUDE_combined_cost_theorem_l1653_165300


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1653_165369

-- Define the two fixed circles
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define the trajectory hyperbola
def trajectory_hyperbola (x y : ℝ) : Prop := 4*(x + 2)^2 - y^2 = 1

-- Define the concept of a moving circle being externally tangent to two fixed circles
def externally_tangent (x y : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (∃ (x_m y_m : ℝ), circle_M x_m y_m ∧ (x - x_m)^2 + (y - y_m)^2 = (r + 1)^2) ∧
  (∃ (x_n y_n : ℝ), circle_N x_n y_n ∧ (x - x_n)^2 + (y - y_n)^2 = (r + 2)^2)

-- The main theorem
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), externally_tangent x y → trajectory_hyperbola x y ∧ x < -2 :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1653_165369


namespace NUMINAMATH_CALUDE_mike_car_payment_l1653_165312

def car_price : ℝ := 35000
def loan_amount : ℝ := 20000
def interest_rate : ℝ := 0.15
def loan_period : ℝ := 1

def total_amount_to_pay : ℝ := car_price + loan_amount * interest_rate * loan_period

theorem mike_car_payment : total_amount_to_pay = 38000 :=
by sorry

end NUMINAMATH_CALUDE_mike_car_payment_l1653_165312


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1653_165343

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (n - 2) * 180 = 3 * 360 → n = 8 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1653_165343


namespace NUMINAMATH_CALUDE_book_cost_problem_l1653_165315

/-- Given two books with a total cost of 300 Rs, if one is sold at a 15% loss
    and the other at a 19% gain, and both are sold at the same price,
    then the cost of the book sold at a loss is 175 Rs. -/
theorem book_cost_problem (C₁ C₂ SP : ℝ) : 
  C₁ + C₂ = 300 →
  SP = 0.85 * C₁ →
  SP = 1.19 * C₂ →
  C₁ = 175 := by
sorry

end NUMINAMATH_CALUDE_book_cost_problem_l1653_165315


namespace NUMINAMATH_CALUDE_S_5_equals_31_l1653_165362

-- Define the sequence and its partial sum
def S (n : ℕ) : ℕ := 2^n - 1

-- State the theorem
theorem S_5_equals_31 : S 5 = 31 := by
  sorry

end NUMINAMATH_CALUDE_S_5_equals_31_l1653_165362


namespace NUMINAMATH_CALUDE_problem_statement_l1653_165386

theorem problem_statement (P Q : Prop) (h_P : P ↔ (2 + 2 = 5)) (h_Q : Q ↔ (3 > 2)) :
  (P ∨ Q) ∧ ¬(¬Q) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1653_165386


namespace NUMINAMATH_CALUDE_min_people_for_valid_seating_l1653_165396

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone. -/
def valid_seating (table : CircularTable) : Prop :=
  ∀ i : ℕ, i < table.total_chairs → ∃ j : ℕ, j < table.total_chairs ∧ 
    (((i + 1) % table.total_chairs = j) ∨ ((i + table.total_chairs - 1) % table.total_chairs = j))

/-- The main theorem stating the minimum number of people required for a valid seating arrangement. -/
theorem min_people_for_valid_seating :
  ∃ (table : CircularTable), table.total_chairs = 100 ∧ 
    valid_seating table ∧ table.seated_people = 25 ∧
    (∀ (smaller_table : CircularTable), smaller_table.total_chairs = 100 → 
      valid_seating smaller_table → smaller_table.seated_people ≥ 25) :=
sorry

end NUMINAMATH_CALUDE_min_people_for_valid_seating_l1653_165396


namespace NUMINAMATH_CALUDE_f_properties_l1653_165335

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) + (2 - a) * Real.exp x - a * x + a * Real.exp 1 / 2

theorem f_properties (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ∨
  (a > 0 ∧ ∃ x_min, ∀ x, f a x ≥ f a x_min) ∧
  (∀ x, f a x ≥ 0 ↔ a ∈ Set.Icc 0 (2 * Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1653_165335


namespace NUMINAMATH_CALUDE_power_of_product_l1653_165380

theorem power_of_product (a b : ℝ) : (-2 * a^2 * b^3)^3 = -8 * a^6 * b^9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1653_165380


namespace NUMINAMATH_CALUDE_larger_number_proof_l1653_165361

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1390)
  (h2 : L = 6 * S + 15) : 
  L = 1665 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1653_165361


namespace NUMINAMATH_CALUDE_tamara_brownie_earnings_l1653_165341

/-- Calculates the earnings from selling brownies --/
def brownie_earnings (pans : ℕ) (pieces_per_pan : ℕ) (price_per_piece : ℕ) : ℕ :=
  pans * pieces_per_pan * price_per_piece

/-- Proves that Tamara's earnings from brownies equal $32 --/
theorem tamara_brownie_earnings :
  brownie_earnings 2 8 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_tamara_brownie_earnings_l1653_165341


namespace NUMINAMATH_CALUDE_area_greater_than_four_thirds_e_cubed_greater_than_twenty_l1653_165363

-- Define the area function S(t)
noncomputable def S (t : ℝ) : ℝ :=
  ∫ x in (0)..(1/t), (Real.exp (t^2 * x))

-- State the theorem
theorem area_greater_than_four_thirds :
  ∀ t > 0, S t > 4/3 :=
by
  sorry

-- Additional fact that can be used in the proof
theorem e_cubed_greater_than_twenty : Real.exp 3 > 20 :=
by
  sorry

end NUMINAMATH_CALUDE_area_greater_than_four_thirds_e_cubed_greater_than_twenty_l1653_165363


namespace NUMINAMATH_CALUDE_divide_money_l1653_165324

theorem divide_money (total : ℝ) (a b c : ℝ) : 
  total = 364 →
  a = (1/2) * b →
  b = (1/2) * c →
  a + b + c = total →
  c = 208 := by
sorry

end NUMINAMATH_CALUDE_divide_money_l1653_165324


namespace NUMINAMATH_CALUDE_union_A_complement_B_when_m_neg_two_A_implies_B_iff_A_not_B_iff_A_subset_complement_B_iff_l1653_165382

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 < 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - m - 1) * (x - m - 7) > 0}

-- Theorem 1
theorem union_A_complement_B_when_m_neg_two :
  A ∪ (Set.univ \ B (-2)) = {x : ℝ | -2 < x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem A_implies_B_iff :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ A → x ∈ B m) ↔ m ∈ Set.Iic (-9) ∪ Set.Ici 1 := by sorry

-- Theorem 3
theorem A_not_B_iff :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ A → x ∉ B m) ↔ m ∈ Set.Icc (-5) (-3) := by sorry

-- Theorem 4
theorem A_subset_complement_B_iff :
  ∀ m : ℝ, A ⊆ (Set.univ \ B m) ↔ m ∈ Set.Ioo (-5) (-3) := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_when_m_neg_two_A_implies_B_iff_A_not_B_iff_A_subset_complement_B_iff_l1653_165382


namespace NUMINAMATH_CALUDE_average_of_multiples_10_to_400_l1653_165340

def multiples_of_10 (n : ℕ) : List ℕ :=
  (List.range ((400 - 10) / 10 + 1)).map (λ i => 10 * (i + 1))

theorem average_of_multiples_10_to_400 :
  (List.sum (multiples_of_10 400)) / (List.length (multiples_of_10 400)) = 205 := by
  sorry

end NUMINAMATH_CALUDE_average_of_multiples_10_to_400_l1653_165340


namespace NUMINAMATH_CALUDE_total_area_of_fields_l1653_165311

/-- The total area of three fields with given dimensions -/
theorem total_area_of_fields (d₁ : ℝ) (l₂ w₂ : ℝ) (b₃ h₃ : ℝ) : 
  d₁ = 12 → l₂ = 15 → w₂ = 8 → b₃ = 18 → h₃ = 10 → 
  (d₁^2 / 2) + (l₂ * w₂) + (b₃ * h₃) = 372 := by
  sorry

end NUMINAMATH_CALUDE_total_area_of_fields_l1653_165311


namespace NUMINAMATH_CALUDE_typing_orders_count_l1653_165349

/-- The number of letters to be typed -/
def n : ℕ := 9

/-- The index of the letter that has already been typed -/
def typed_letter : ℕ := 8

/-- The set of possible remaining letters after the typed_letter has been removed -/
def remaining_letters : Finset ℕ := Finset.filter (λ x => x ≠ typed_letter ∧ x ≤ n) (Finset.range (n + 1))

/-- The number of possible typing orders for the remaining letters -/
def num_typing_orders : ℕ :=
  (Finset.range 8).sum (λ k => Nat.choose 7 k * (k + 2))

theorem typing_orders_count :
  num_typing_orders = 704 :=
sorry

end NUMINAMATH_CALUDE_typing_orders_count_l1653_165349


namespace NUMINAMATH_CALUDE_complex_cube_root_l1653_165354

theorem complex_cube_root (x y d : ℤ) (z : ℂ) : 
  x > 0 → y > 0 → z = x + y * Complex.I → z^3 = -54 + d * Complex.I → z = 3 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l1653_165354


namespace NUMINAMATH_CALUDE_max_value_of_f_l1653_165373

/-- The function f(x) = x^3 - 3ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + 2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem max_value_of_f (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f a x ≥ f a 2) →
  (∃ x : ℝ, f a x = 18 ∧ ∀ y : ℝ, f a y ≤ f a x) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1653_165373


namespace NUMINAMATH_CALUDE_probability_three_heads_five_coins_l1653_165371

theorem probability_three_heads_five_coins :
  let n : ℕ := 5  -- number of coins
  let k : ℕ := 3  -- number of heads we want
  let p : ℚ := 1/2  -- probability of getting heads on a single coin toss
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k) = 5/16 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_heads_five_coins_l1653_165371


namespace NUMINAMATH_CALUDE_sum_m_n_eq_67_l1653_165318

-- Define the point R
def R : ℝ × ℝ := (8, 6)

-- Define the lines
def line1 (x y : ℝ) : Prop := 8 * y = 15 * x
def line2 (x y : ℝ) : Prop := 10 * y = 3 * x

-- Define points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define the conditions
axiom P_on_line1 : line1 P.1 P.2
axiom Q_on_line2 : line2 Q.1 Q.2
axiom R_is_midpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the length of PQ
def PQ_length : ℝ := sorry

-- Define m and n as positive integers
def m : ℕ+ := sorry
def n : ℕ+ := sorry

-- PQ length is equal to m/n
axiom PQ_length_eq_m_div_n : PQ_length = m.val / n.val

-- m and n are coprime
axiom m_n_coprime : Nat.Coprime m.val n.val

-- Theorem to prove
theorem sum_m_n_eq_67 : m.val + n.val = 67 := sorry

end NUMINAMATH_CALUDE_sum_m_n_eq_67_l1653_165318


namespace NUMINAMATH_CALUDE_ef_fraction_of_gh_l1653_165307

/-- Given a line segment GH with points E and F on it, prove that EF is 5/11 of GH -/
theorem ef_fraction_of_gh (G E F H : ℝ) : 
  (E ≤ F) → -- E is before or at F on the line
  (F ≤ H) → -- F is before or at H on the line
  (G ≤ E) → -- G is before or at E on the line
  (G - E = 5 * (H - E)) → -- GE = 5 * EH
  (G - F = 10 * (H - F)) → -- GF = 10 * FH
  F - E = 5 / 11 * (H - G) := by
sorry

end NUMINAMATH_CALUDE_ef_fraction_of_gh_l1653_165307


namespace NUMINAMATH_CALUDE_angle_A_measure_perimeter_range_l1653_165323

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  t.a / (Real.sqrt 3 * Real.cos t.A) = t.c / Real.sin t.C

-- Theorem for angle A
theorem angle_A_measure (t : Triangle) (h : given_condition t) : t.A = π / 3 :=
sorry

-- Theorem for perimeter range
theorem perimeter_range (t : Triangle) (h1 : given_condition t) (h2 : t.a = 6) :
  12 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_angle_A_measure_perimeter_range_l1653_165323


namespace NUMINAMATH_CALUDE_tetrahedrons_count_l1653_165317

/-- The number of tetrahedrons formed by choosing 4 vertices from a triangular prism -/
def tetrahedrons_from_prism : ℕ :=
  Nat.choose 6 4 - 3

/-- Theorem stating that the number of tetrahedrons is 12 -/
theorem tetrahedrons_count : tetrahedrons_from_prism = 12 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedrons_count_l1653_165317


namespace NUMINAMATH_CALUDE_rhombus_properties_l1653_165308

-- Define a rhombus
structure Rhombus (V : Type*) [NormedAddCommGroup V] :=
  (A B C D : V)
  (is_rhombus : True)  -- This is a placeholder for the rhombus property

-- Define the theorem
theorem rhombus_properties {V : Type*} [NormedAddCommGroup V] (r : Rhombus V) :
  (‖r.A - r.B‖ = ‖r.B - r.C‖) ∧ 
  (‖r.A - r.B - (r.C - r.D)‖ = ‖r.A - r.D + (r.B - r.C)‖) ∧
  (‖r.A - r.C‖^2 + ‖r.B - r.D‖^2 = 4 * ‖r.A - r.B‖^2) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_properties_l1653_165308


namespace NUMINAMATH_CALUDE_cylinder_radius_calculation_l1653_165365

/-- Regular prism with a cylinder -/
structure PrismWithCylinder where
  -- Base side length of the prism
  base_side : ℝ
  -- Lateral edge length of the prism
  lateral_edge : ℝ
  -- Distance between cylinder axis and line AB₁
  axis_distance : ℝ
  -- Radius of the cylinder
  cylinder_radius : ℝ

/-- Theorem stating the radius of the cylinder given the prism dimensions -/
theorem cylinder_radius_calculation (p : PrismWithCylinder) 
  (h1 : p.base_side = 1)
  (h2 : p.lateral_edge = 1 / Real.sqrt 3)
  (h3 : p.axis_distance = 1 / 4) :
  p.cylinder_radius = Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_calculation_l1653_165365


namespace NUMINAMATH_CALUDE_amazon_tide_problem_l1653_165353

theorem amazon_tide_problem (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (2 * x + φ)) →
  (abs φ < π / 2) →
  (∀ x, f (x - π / 3) = -f (-x - π / 3)) →
  (φ = -π / 3) ∧
  (∀ x, f (5 * π / 12 + x) = f (5 * π / 12 - x)) ∧
  (∀ x ∈ Set.Icc (-π / 3) (-π / 6), ∀ y ∈ Set.Icc (-π / 3) (-π / 6), x < y → f x > f y) ∧
  (∃ x ∈ Set.Ioo 0 (π / 2), (deriv f) x = 0) :=
by sorry

end NUMINAMATH_CALUDE_amazon_tide_problem_l1653_165353


namespace NUMINAMATH_CALUDE_complex_fourth_power_problem_l1653_165366

theorem complex_fourth_power_problem : ∃ (d : ℤ), (1 + 3*I : ℂ)^4 = 82 + d*I := by sorry

end NUMINAMATH_CALUDE_complex_fourth_power_problem_l1653_165366


namespace NUMINAMATH_CALUDE_equation_solution_l1653_165326

theorem equation_solution : 
  ∃! x : ℝ, |Real.sqrt ((x - 2)^2) - 1| = x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1653_165326


namespace NUMINAMATH_CALUDE_problem_solution_l1653_165327

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 5

-- Theorem statement
theorem problem_solution :
  -- Condition 1: y-1 is directly proportional to x+2
  (∃ k : ℝ, ∀ x y : ℝ, y = f x → y - 1 = k * (x + 2)) ∧
  -- Condition 2: When x=1, y=7
  (f 1 = 7) ∧
  -- Solution 1: The function f satisfies the conditions
  (∀ x : ℝ, f x = 2 * x + 5) ∧
  -- Solution 2: The point (-7/2, -2) lies on the graph of f
  (f (-7/2) = -2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1653_165327


namespace NUMINAMATH_CALUDE_four_students_three_communities_l1653_165319

/-- The number of ways to distribute n students among k communities,
    where each student goes to exactly one community and each community
    receives at least one student. -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 4 students among 3 communities
    results in 36 different arrangements. -/
theorem four_students_three_communities :
  distribute_students 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_four_students_three_communities_l1653_165319


namespace NUMINAMATH_CALUDE_probability_blue_after_red_l1653_165383

/-- Probability of picking a blue marble after removing a red one --/
theorem probability_blue_after_red (total : ℕ) (yellow : ℕ) (green : ℕ) (red : ℕ) (blue : ℕ) :
  total = 120 →
  yellow = 30 →
  green = yellow / 3 →
  red = 2 * green →
  blue = total - yellow - green - red →
  (blue : ℚ) / (total - 1 : ℚ) = 60 / 119 := by
  sorry

end NUMINAMATH_CALUDE_probability_blue_after_red_l1653_165383


namespace NUMINAMATH_CALUDE_f_min_value_a_range_l1653_165390

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for the minimum value of f(x)
theorem f_min_value : ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x, f x = m) ∧ m = 5 := by sorry

-- Theorem for the range of a
theorem a_range (x : ℝ) (hx : x ∈ Set.Icc (-3) 2) :
  (∀ a, f x ≥ |x + a|) ↔ a ∈ Set.Icc (-2) 3 := by sorry

end NUMINAMATH_CALUDE_f_min_value_a_range_l1653_165390


namespace NUMINAMATH_CALUDE_min_tank_cost_l1653_165399

/-- Represents the cost function for a rectangular water tank. -/
def tank_cost (x y : ℝ) : ℝ :=
  120 * (x * y) + 100 * (2 * 3 * x + 2 * 3 * y)

/-- Theorem stating the minimum cost for the water tank construction. -/
theorem min_tank_cost :
  let volume : ℝ := 300
  let depth : ℝ := 3
  let bottom_cost : ℝ := 120
  let wall_cost : ℝ := 100
  ∀ x y : ℝ,
    x > 0 → y > 0 →
    x * y * depth = volume →
    tank_cost x y ≥ 24000 ∧
    (x = 10 ∧ y = 10 → tank_cost x y = 24000) :=
by sorry

end NUMINAMATH_CALUDE_min_tank_cost_l1653_165399


namespace NUMINAMATH_CALUDE_expression_evaluation_l1653_165348

theorem expression_evaluation : -1^5 + (-3)^0 - (Real.sqrt 2)^2 + 4 * |-(1/4)| = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1653_165348


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l1653_165320

/-- The perimeter of a semicircle with radius 6.83 cm is approximately 35.12 cm. -/
theorem semicircle_perimeter_approx : 
  let r : Real := 6.83
  let perimeter : Real := π * r + 2 * r
  ∃ ε > 0, abs (perimeter - 35.12) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l1653_165320


namespace NUMINAMATH_CALUDE_imaginary_unit_power_sum_l1653_165368

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_unit_power_sum : i^2 + i^4 = 0 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_sum_l1653_165368


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1653_165379

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1653_165379


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_and_b_l1653_165350

/-- A function f(x) = ax³ + bx has an extreme value of -2 at x = 1 -/
def has_extreme_value (a b : ℝ) : Prop :=
  let f := fun x : ℝ => a * x^3 + b * x
  f 1 = -2 ∧ (deriv f) 1 = 0

/-- Theorem: If f(x) = ax³ + bx has an extreme value of -2 at x = 1, then a = 1 and b = -3 -/
theorem extreme_value_implies_a_and_b :
  ∀ a b : ℝ, has_extreme_value a b → a = 1 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_and_b_l1653_165350


namespace NUMINAMATH_CALUDE_garrison_size_l1653_165352

/-- Given a garrison with provisions and reinforcements, calculate the initial number of men. -/
theorem garrison_size (initial_days : ℕ) (reinforcement_arrival : ℕ) (remaining_days : ℕ) (reinforcement_size : ℕ) : 
  initial_days = 54 →
  reinforcement_arrival = 15 →
  remaining_days = 20 →
  reinforcement_size = 1900 →
  ∃ (initial_men : ℕ), 
    initial_men * (initial_days - reinforcement_arrival) = 
    (initial_men + reinforcement_size) * remaining_days ∧
    initial_men = 2000 :=
by sorry

end NUMINAMATH_CALUDE_garrison_size_l1653_165352


namespace NUMINAMATH_CALUDE_house_rent_fraction_l1653_165375

def salary : ℚ := 140000

def food_fraction : ℚ := 1/5
def clothes_fraction : ℚ := 3/5
def remaining_amount : ℚ := 14000

theorem house_rent_fraction :
  ∃ (house_rent_fraction : ℚ),
    house_rent_fraction * salary + food_fraction * salary + clothes_fraction * salary + remaining_amount = salary ∧
    house_rent_fraction = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_house_rent_fraction_l1653_165375


namespace NUMINAMATH_CALUDE_problem_solution_l1653_165322

theorem problem_solution : (88 * 707 - 38 * 707) / 1414 = 25 := by
  have h : 1414 = 707 * 2 := by sorry
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1653_165322


namespace NUMINAMATH_CALUDE_inequality_proof_l1653_165376

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1653_165376


namespace NUMINAMATH_CALUDE_robin_total_pieces_l1653_165388

/-- The number of pieces in a package of Type A gum -/
def type_a_gum_pieces : ℕ := 4

/-- The number of pieces in a package of Type B gum -/
def type_b_gum_pieces : ℕ := 8

/-- The number of pieces in a package of Type C gum -/
def type_c_gum_pieces : ℕ := 12

/-- The number of pieces in a package of Type X candy -/
def type_x_candy_pieces : ℕ := 6

/-- The number of pieces in a package of Type Y candy -/
def type_y_candy_pieces : ℕ := 10

/-- The number of packages of Type A gum Robin has -/
def robin_type_a_gum_packages : ℕ := 10

/-- The number of packages of Type B gum Robin has -/
def robin_type_b_gum_packages : ℕ := 5

/-- The number of packages of Type C gum Robin has -/
def robin_type_c_gum_packages : ℕ := 13

/-- The number of packages of Type X candy Robin has -/
def robin_type_x_candy_packages : ℕ := 8

/-- The number of packages of Type Y candy Robin has -/
def robin_type_y_candy_packages : ℕ := 6

/-- The total number of gum packages Robin has -/
def robin_total_gum_packages : ℕ := 28

/-- The total number of candy packages Robin has -/
def robin_total_candy_packages : ℕ := 14

theorem robin_total_pieces : 
  robin_type_a_gum_packages * type_a_gum_pieces +
  robin_type_b_gum_packages * type_b_gum_pieces +
  robin_type_c_gum_packages * type_c_gum_pieces +
  robin_type_x_candy_packages * type_x_candy_pieces +
  robin_type_y_candy_packages * type_y_candy_pieces = 344 :=
by sorry

end NUMINAMATH_CALUDE_robin_total_pieces_l1653_165388


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_15_with_sqrt_between_28_and_28_5_l1653_165321

theorem unique_integer_divisible_by_15_with_sqrt_between_28_and_28_5 :
  ∃! n : ℕ+, (15 ∣ n) ∧ (28 < (n : ℝ).sqrt) ∧ ((n : ℝ).sqrt < 28.5) := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_15_with_sqrt_between_28_and_28_5_l1653_165321


namespace NUMINAMATH_CALUDE_factorization_equality_l1653_165394

theorem factorization_equality (a x y : ℝ) : a^2 * (x - y) + 4 * (y - x) = (x - y) * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1653_165394


namespace NUMINAMATH_CALUDE_self_inverse_solutions_l1653_165346

def is_self_inverse (a d : ℝ) : Prop :=
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, 4; -9, d]
  M * M = 1

theorem self_inverse_solutions :
  ∃! n : ℕ, ∃ S : Finset (ℝ × ℝ),
    S.card = n ∧
    (∀ p : ℝ × ℝ, p ∈ S ↔ is_self_inverse p.1 p.2) :=
by sorry

end NUMINAMATH_CALUDE_self_inverse_solutions_l1653_165346


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l1653_165397

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x - 4 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by sorry

-- Theorem for the union of complement of A and B
theorem union_complement_A_B : (Aᶜ) ∪ B = {x : ℝ | 0 < x} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l1653_165397


namespace NUMINAMATH_CALUDE_larger_cross_section_distance_l1653_165347

/-- Right octagonal pyramid with two parallel cross sections -/
structure OctagonalPyramid where
  /-- Ratio of areas of two cross sections -/
  area_ratio : ℝ
  /-- Distance between the two cross sections -/
  cross_section_distance : ℝ

/-- Theorem about the distance of the larger cross section from the apex -/
theorem larger_cross_section_distance (pyramid : OctagonalPyramid) 
  (h_ratio : pyramid.area_ratio = 4 / 9)
  (h_distance : pyramid.cross_section_distance = 10) :
  ∃ (apex_distance : ℝ), apex_distance = 30 := by
  sorry

end NUMINAMATH_CALUDE_larger_cross_section_distance_l1653_165347


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_l1653_165389

theorem smallest_five_digit_divisible : ∃ n : ℕ, 
  (10000 ≤ n ∧ n < 100000) ∧ 
  (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ 2 ∣ m ∧ 3 ∣ m ∧ 8 ∣ m ∧ 9 ∣ m → n ≤ m) ∧
  2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n ∧
  n = 10008 := by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_l1653_165389


namespace NUMINAMATH_CALUDE_reciprocal_sum_equals_five_l1653_165374

theorem reciprocal_sum_equals_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x + y = 5 * x * y) (h2 : x = 2 * y) : 
  1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equals_five_l1653_165374


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l1653_165345

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l1653_165345


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l1653_165338

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel 
  (l : Line) (m : Line) (α : Plane) (β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β) :
  (parallel α β → perpendicular_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l1653_165338


namespace NUMINAMATH_CALUDE_janas_height_l1653_165309

/-- Given the heights of several people and their relationships, prove Jana's height. -/
theorem janas_height
  (kelly_jess : ℝ) -- Height difference between Kelly and Jess
  (jana_kelly : ℝ) -- Height difference between Jana and Kelly
  (jess_height : ℝ) -- Jess's height
  (jess_alex : ℝ) -- Height difference between Jess and Alex
  (alex_sam : ℝ) -- Height difference between Alex and Sam
  (h1 : jana_kelly = 5.5)
  (h2 : kelly_jess = -3.75)
  (h3 : jess_height = 72)
  (h4 : jess_alex = -1.25)
  (h5 : alex_sam = 0.5)
  : jess_height - kelly_jess + jana_kelly = 73.75 := by
  sorry

#check janas_height

end NUMINAMATH_CALUDE_janas_height_l1653_165309


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l1653_165358

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 28 < 0 → n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l1653_165358


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1653_165325

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2 - x) ↔ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1653_165325


namespace NUMINAMATH_CALUDE_normal_vector_perpendicular_cosine_angle_between_lines_distance_point_to_line_l1653_165314

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line2D where
  A : ℝ
  B : ℝ
  C : ℝ
  nonzero : A ≠ 0 ∨ B ≠ 0

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The vector perpendicular to a line is its normal vector -/
theorem normal_vector_perpendicular (l : Line2D) :
  let dir_vec := (-l.B, l.A)
  let normal_vec := (l.A, l.B)
  (dir_vec.1 * normal_vec.1 + dir_vec.2 * normal_vec.2 = 0) :=
sorry

/-- The cosine of the angle between two intersecting lines -/
theorem cosine_angle_between_lines (l₁ l₂ : Line2D) :
  let cos_theta := |(l₁.A * l₂.A + l₁.B * l₂.B) / (Real.sqrt (l₁.A^2 + l₁.B^2) * Real.sqrt (l₂.A^2 + l₂.B^2))|
  (0 ≤ cos_theta ∧ cos_theta ≤ 1) :=
sorry

/-- The distance from a point to a line -/
theorem distance_point_to_line (p : Point2D) (l : Line2D) :
  let d := |l.A * p.x + l.B * p.y + l.C| / Real.sqrt (l.A^2 + l.B^2)
  (d ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_normal_vector_perpendicular_cosine_angle_between_lines_distance_point_to_line_l1653_165314


namespace NUMINAMATH_CALUDE_ellipse_major_minor_distance_l1653_165367

/-- An ellipse with equation 4(x+2)^2 + 16y^2 = 64 -/
structure Ellipse where
  eq : ∀ x y : ℝ, 4 * (x + 2)^2 + 16 * y^2 = 64

/-- Point C is an endpoint of the major axis -/
def C (e : Ellipse) : ℝ × ℝ := sorry

/-- Point D is an endpoint of the minor axis -/
def D (e : Ellipse) : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_major_minor_distance (e : Ellipse) : 
  distance (C e) (D e) = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_minor_distance_l1653_165367


namespace NUMINAMATH_CALUDE_school_play_ticket_value_l1653_165393

/-- Calculates the total value of tickets sold for a school play --/
def total_ticket_value (student_price : ℕ) (adult_price : ℕ) (child_price : ℕ) (senior_price : ℕ)
                       (student_count : ℕ) (adult_count : ℕ) (child_count : ℕ) (senior_count : ℕ) : ℕ :=
  student_price * student_count + adult_price * adult_count + child_price * child_count + senior_price * senior_count

theorem school_play_ticket_value :
  total_ticket_value 6 8 4 7 20 12 15 10 = 346 := by
  sorry

end NUMINAMATH_CALUDE_school_play_ticket_value_l1653_165393


namespace NUMINAMATH_CALUDE_parabola_shift_left_one_l1653_165302

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * shift + p.b,
    c := p.a * shift^2 - p.b * shift + p.c }

theorem parabola_shift_left_one :
  let original := Parabola.mk 1 0 2
  let shifted := shift_parabola original 1
  shifted = Parabola.mk 1 2 3 := by
  sorry

#check parabola_shift_left_one

end NUMINAMATH_CALUDE_parabola_shift_left_one_l1653_165302


namespace NUMINAMATH_CALUDE_subtract_negative_l1653_165370

theorem subtract_negative : -2 - (-3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l1653_165370


namespace NUMINAMATH_CALUDE_problem_solution_l1653_165344

def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {-a, a^2 + 3}

theorem problem_solution (a : ℝ) : A ∪ B a = {1, 2, 4} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1653_165344


namespace NUMINAMATH_CALUDE_outbound_speed_calculation_l1653_165303

theorem outbound_speed_calculation (distance : ℝ) (return_speed : ℝ) (total_time : ℝ) :
  distance = 19.999999999999996 →
  return_speed = 4 →
  total_time = 5.8 →
  ∃ outbound_speed : ℝ, 
    outbound_speed = 25 ∧
    distance / outbound_speed + distance / return_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_outbound_speed_calculation_l1653_165303


namespace NUMINAMATH_CALUDE_sequence_term_l1653_165333

theorem sequence_term (n : ℕ) (a : ℕ → ℝ) : 
  (∀ k, a k = Real.sqrt (2 * k - 1)) → 
  a 23 = 3 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_term_l1653_165333


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l1653_165342

theorem volleyball_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) 
  (quadruplets_in_lineup : ℕ) :
  total_players = 17 →
  quadruplets = 4 →
  starters = 6 →
  quadruplets_in_lineup = 2 →
  (Nat.choose quadruplets quadruplets_in_lineup) * 
  (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 4290 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l1653_165342
