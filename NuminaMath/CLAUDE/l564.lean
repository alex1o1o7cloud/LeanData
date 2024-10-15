import Mathlib

namespace NUMINAMATH_CALUDE_two_red_two_blue_probability_l564_56413

def total_marbles : ℕ := 15 + 9

def red_marbles : ℕ := 15

def blue_marbles : ℕ := 9

def marbles_selected : ℕ := 4

theorem two_red_two_blue_probability :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2) / Nat.choose total_marbles marbles_selected = 108 / 361 :=
by sorry

end NUMINAMATH_CALUDE_two_red_two_blue_probability_l564_56413


namespace NUMINAMATH_CALUDE_tetrahedron_division_l564_56482

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ

/-- A plane passing through one edge and the midpoint of the opposite edge of a tetrahedron -/
structure DividingPlane where
  tetrahedron : RegularTetrahedron

/-- The parts into which a tetrahedron is divided by the planes -/
structure TetrahedronPart where
  tetrahedron : RegularTetrahedron
  planes : Finset DividingPlane

/-- The theorem stating the division of a regular tetrahedron by six specific planes -/
theorem tetrahedron_division (t : RegularTetrahedron) 
  (h_volume : t.volume = 1) 
  (planes : Finset DividingPlane) 
  (h_planes : planes.card = 6) 
  (h_plane_position : ∀ p ∈ planes, p.tetrahedron = t) : 
  ∃ (parts : Finset TetrahedronPart), 
    (parts.card = 24) ∧ 
    (∀ part ∈ parts, part.tetrahedron = t ∧ part.planes = planes) ∧
    (∀ part ∈ parts, ∃ v : ℝ, v = 1 / 24) :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_division_l564_56482


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l564_56441

theorem consecutive_integers_average (x : ℝ) : 
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 = 25 →
  ((x - 9) + (x + 1 - 8) + (x + 2 - 7) + (x + 3 - 6) + (x + 4 - 5) + (x + 5 - 4) + (x + 6 - 3) + (x + 7 - 2) + (x + 8 - 1) + (x + 9)) / 10 = 20.5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l564_56441


namespace NUMINAMATH_CALUDE_total_animal_eyes_l564_56458

theorem total_animal_eyes (num_snakes num_alligators : ℕ) 
  (snake_eyes alligator_eyes : ℕ) : ℕ :=
  by
    -- Define the number of snakes and alligators
    have h1 : num_snakes = 18 := by sorry
    have h2 : num_alligators = 10 := by sorry
    
    -- Define the number of eyes for each snake and alligator
    have h3 : snake_eyes = 2 := by sorry
    have h4 : alligator_eyes = 2 := by sorry
    
    -- Calculate total number of eyes
    have h5 : num_snakes * snake_eyes + num_alligators * alligator_eyes = 56 := by sorry
    
    exact 56

#check total_animal_eyes

end NUMINAMATH_CALUDE_total_animal_eyes_l564_56458


namespace NUMINAMATH_CALUDE_line_slope_l564_56417

theorem line_slope (x y : ℝ) :
  x + Real.sqrt 3 * y + 2 = 0 →
  (y - (-2 / Real.sqrt 3)) / (x - 0) = - Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l564_56417


namespace NUMINAMATH_CALUDE_curve_not_parabola_l564_56468

/-- The equation of the curve -/
def curve_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + (m + 1) * y^2 = m * (m + 1)

/-- Definition of a parabola in general form -/
def is_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧
    ∀ x y, f x y ↔ a * x^2 + b * x * y + c * y^2 + d * x + e * y = 0

/-- Theorem: The curve cannot be a parabola -/
theorem curve_not_parabola :
  ∀ m : ℝ, ¬(is_parabola (curve_equation m)) :=
sorry

end NUMINAMATH_CALUDE_curve_not_parabola_l564_56468


namespace NUMINAMATH_CALUDE_polynomial_factorization_l564_56431

theorem polynomial_factorization (a b : ℝ) :
  2 * a^3 - 3 * a^2 * b - 3 * a * b^2 + 2 * b^3 = (a + b) * (a - 2*b) * (2*a - b) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l564_56431


namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l564_56478

def total_spent : ℚ := 33.56
def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14

theorem jacket_cost_calculation : 
  total_spent - shorts_cost - shirt_cost = 7.43 := by sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l564_56478


namespace NUMINAMATH_CALUDE_total_cups_sold_l564_56481

def plastic_cups : ℕ := 284
def ceramic_cups : ℕ := 284

theorem total_cups_sold : plastic_cups + ceramic_cups = 568 := by
  sorry

end NUMINAMATH_CALUDE_total_cups_sold_l564_56481


namespace NUMINAMATH_CALUDE_ratio_sum_difference_l564_56438

theorem ratio_sum_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / y = (x + y) / (x - y) → x / y = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_difference_l564_56438


namespace NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_for_nonempty_solution_l564_56472

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 3| + |2*x - 1|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -7/4 ≤ x ∧ x ≤ 3/4} := by sorry

-- Theorem for the range of m when the solution set of f(x) < |m-1| is non-empty
theorem range_of_m_for_nonempty_solution (m : ℝ) :
  (∃ x : ℝ, f x < |m - 1|) → (m > 5 ∨ m < -3) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_for_nonempty_solution_l564_56472


namespace NUMINAMATH_CALUDE_quadratic_equation_at_most_one_solution_l564_56442

theorem quadratic_equation_at_most_one_solution (a : ℝ) : 
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a ≥ 9/8 ∨ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_at_most_one_solution_l564_56442


namespace NUMINAMATH_CALUDE_triangle_inequality_l564_56493

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b + b * c + c * a ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l564_56493


namespace NUMINAMATH_CALUDE_equation_solution_l564_56415

theorem equation_solution : ∃ x : ℚ, (5 * x + 9 * x = 420 - 10 * (x - 4)) ∧ x = 115 / 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l564_56415


namespace NUMINAMATH_CALUDE_missing_number_solution_l564_56477

theorem missing_number_solution : 
  ∃ x : ℝ, 0.72 * 0.43 + 0.12 * x = 0.3504 ∧ x = 0.34 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_solution_l564_56477


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l564_56423

theorem quadratic_one_solution (m : ℚ) : 
  (∃! y, 3 * y^2 - 7 * y + m = 0) ↔ m = 49/12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l564_56423


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l564_56484

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem arithmetic_progression_x_value :
  ∀ x : ℝ, is_arithmetic_progression (x - 3) (x + 2) (2*x - 1) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l564_56484


namespace NUMINAMATH_CALUDE_interest_rate_problem_l564_56454

/-- 
Given a principal amount P and an interest rate R,
if increasing the rate by 1% for 3 years results in Rs. 63 more interest,
then P = Rs. 2100.
-/
theorem interest_rate_problem (P R : ℚ) : 
  (P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 63) → P = 2100 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l564_56454


namespace NUMINAMATH_CALUDE_total_volume_of_cubes_total_volume_is_114_l564_56400

theorem total_volume_of_cubes : ℕ → ℕ → ℕ → ℕ → ℕ
  | carl_cube_count, carl_cube_side, kate_cube_count, kate_cube_side =>
    let carl_cube_volume := carl_cube_side ^ 3
    let kate_cube_volume := kate_cube_side ^ 3
    carl_cube_count * carl_cube_volume + kate_cube_count * kate_cube_volume

theorem total_volume_is_114 : 
  total_volume_of_cubes 4 3 6 1 = 114 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_cubes_total_volume_is_114_l564_56400


namespace NUMINAMATH_CALUDE_polynomial_roots_magnitude_l564_56416

theorem polynomial_roots_magnitude (c : ℂ) : 
  let p : ℂ → ℂ := λ x => (x^2 - 2*x + 2) * (x^2 - c*x + 4) * (x^2 - 4*x + 8)
  (∃ (s : Finset ℂ), s.card = 4 ∧ (∀ z ∈ s, p z = 0) ∧ (∀ z, p z = 0 → z ∈ s)) →
  Complex.abs c = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_magnitude_l564_56416


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l564_56448

theorem triangle_angle_inequality (A B C α : Real) : 
  A + B + C = π →
  A > 0 → B > 0 → C > 0 →
  α = min (2 * A - B) (min (3 * B - 2 * C) (π / 2 - A)) →
  α ≤ 2 * π / 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l564_56448


namespace NUMINAMATH_CALUDE_b_visited_zhougong_l564_56455

-- Define the celebrities
inductive Celebrity
| A
| B
| C

-- Define the places
inductive Place
| ZhougongTemple
| FamenTemple
| Wuzhangyuan

-- Define a function to represent whether a celebrity visited a place
def visited : Celebrity → Place → Prop := sorry

-- A visited more places than B
axiom a_visited_more : ∃ (p : Place), visited Celebrity.A p ∧ ¬visited Celebrity.B p

-- A did not visit Famen Temple
axiom a_not_famen : ¬visited Celebrity.A Place.FamenTemple

-- B did not visit Wuzhangyuan
axiom b_not_wuzhangyuan : ¬visited Celebrity.B Place.Wuzhangyuan

-- The three celebrities visited the same place
axiom same_place : ∃ (p : Place), visited Celebrity.A p ∧ visited Celebrity.B p ∧ visited Celebrity.C p

-- Theorem to prove
theorem b_visited_zhougong : visited Celebrity.B Place.ZhougongTemple := by sorry

end NUMINAMATH_CALUDE_b_visited_zhougong_l564_56455


namespace NUMINAMATH_CALUDE_proposition_p_false_l564_56452

theorem proposition_p_false : 
  ¬(∀ x : ℝ, x > 0 → x^2 - 3*x + 12 < 0) ∧ 
  (∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 12 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_proposition_p_false_l564_56452


namespace NUMINAMATH_CALUDE_n_value_for_specific_x_and_y_l564_56421

theorem n_value_for_specific_x_and_y :
  let x : ℕ := 3
  let y : ℕ := 1
  let n : ℤ := x - 3 * y^(x - y) + 1
  n = 1 := by sorry

end NUMINAMATH_CALUDE_n_value_for_specific_x_and_y_l564_56421


namespace NUMINAMATH_CALUDE_equation_solution_l564_56461

theorem equation_solution : ∃ x : ℚ, (1 / 3 - 1 / 4 : ℚ) = 1 / (2 * x) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l564_56461


namespace NUMINAMATH_CALUDE_circle_and_line_equations_l564_56488

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧
                 (2 - a)^2 + (4 - b)^2 = r^2 ∧
                 (1 - a)^2 + (3 - b)^2 = r^2 ∧
                 a - b + 1 = 0

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * x + 1

-- Define the dot product of OM and ON
def dot_product_OM_ON (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

theorem circle_and_line_equations :
  ∀ (k : ℝ),
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_l x₁ y₁ k ∧ line_l x₂ y₂ k ∧
    dot_product_OM_ON x₁ y₁ x₂ y₂ = 12) →
  (∀ (x y : ℝ), circle_C x y ↔ (x - 2)^2 + (y - 3)^2 = 1) ∧
  k = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_equations_l564_56488


namespace NUMINAMATH_CALUDE_brother_travel_distance_l564_56449

theorem brother_travel_distance (total_time : ℝ) (speed_diff : ℝ) (distance_diff : ℝ) :
  total_time = 120 ∧ speed_diff = 4 ∧ distance_diff = 40 →
  ∃ (x y : ℝ),
    x = 20 ∧ y = 60 ∧
    total_time / x - total_time / y = speed_diff ∧
    y - x = distance_diff :=
by sorry

end NUMINAMATH_CALUDE_brother_travel_distance_l564_56449


namespace NUMINAMATH_CALUDE_field_width_l564_56492

/-- Given a rectangular field with length 75 meters, where running around it 3 times
    covers a distance of 540 meters, prove that the width of the field is 15 meters. -/
theorem field_width (length : ℝ) (width : ℝ) (perimeter : ℝ) :
  length = 75 →
  3 * perimeter = 540 →
  perimeter = 2 * (length + width) →
  width = 15 := by
sorry

end NUMINAMATH_CALUDE_field_width_l564_56492


namespace NUMINAMATH_CALUDE_level_passing_game_l564_56497

/-- A fair six-sided die -/
def Die := Finset.range 6

/-- The number of times the die is rolled at level n -/
def rolls (n : ℕ) : ℕ := n

/-- The condition for passing a level -/
def pass_level (n : ℕ) (sum : ℕ) : Prop := sum > 2^n

/-- The maximum number of levels that can be passed -/
def max_levels : ℕ := 4

/-- The probability of passing the first three levels consecutively -/
def prob_pass_three : ℚ := 100 / 243

theorem level_passing_game :
  (∀ n : ℕ, n > max_levels → ¬∃ sum : ℕ, sum ≤ 6 * rolls n ∧ pass_level n sum) ∧
  (∃ sum : ℕ, sum ≤ 6 * rolls max_levels ∧ pass_level max_levels sum) ∧
  prob_pass_three = (2/3) * (5/6) * (20/27) :=
sorry

end NUMINAMATH_CALUDE_level_passing_game_l564_56497


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l564_56426

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l564_56426


namespace NUMINAMATH_CALUDE_tv_show_payment_ratio_l564_56494

/-- The ratio of payments to major and minor characters in a TV show -/
theorem tv_show_payment_ratio :
  let num_main_characters : ℕ := 5
  let num_minor_characters : ℕ := 4
  let minor_character_payment : ℕ := 15000
  let total_payment : ℕ := 285000
  let minor_characters_total : ℕ := num_minor_characters * minor_character_payment
  let major_characters_total : ℕ := total_payment - minor_characters_total
  (major_characters_total : ℚ) / minor_characters_total = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_payment_ratio_l564_56494


namespace NUMINAMATH_CALUDE_smallest_prime_with_42_divisors_l564_56465

-- Define a function to count the number of divisors
def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

-- Define the function F(p) = p^3 + 2p^2 + p
def F (p : ℕ) : ℕ := p^3 + 2*p^2 + p

-- Main theorem
theorem smallest_prime_with_42_divisors :
  ∃ (p : ℕ), Nat.Prime p ∧ 
             count_divisors (F p) = 42 ∧ 
             (∀ q < p, Nat.Prime q → count_divisors (F q) ≠ 42) ∧
             p = 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_with_42_divisors_l564_56465


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l564_56474

-- Define the proposition
def P (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0

-- Define the sufficient condition
def sufficient_condition (a : ℝ) : Prop := a ≥ 5

-- Theorem statement
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, sufficient_condition a → P a) ∧
  ¬(∀ a : ℝ, P a → sufficient_condition a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l564_56474


namespace NUMINAMATH_CALUDE_cuboid_colored_cubes_theorem_l564_56425

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of cubes colored on only one side when a cuboid is cut into unit cubes -/
def cubesColoredOnOneSide (c : Cuboid) : ℕ :=
  2 * ((c.width - 2) * (c.length - 2) + (c.width - 2) * (c.height - 2) + (c.length - 2) * (c.height - 2))

theorem cuboid_colored_cubes_theorem (c : Cuboid) 
    (h_width : c.width = 5)
    (h_length : c.length = 4)
    (h_height : c.height = 3) :
  cubesColoredOnOneSide c = 22 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_colored_cubes_theorem_l564_56425


namespace NUMINAMATH_CALUDE_average_salary_calculation_l564_56470

theorem average_salary_calculation (officer_salary : ℕ) (non_officer_salary : ℕ)
  (num_officers : ℕ) (num_non_officers : ℕ) :
  officer_salary = 430 →
  non_officer_salary = 110 →
  num_officers = 15 →
  num_non_officers = 465 →
  (officer_salary * num_officers + non_officer_salary * num_non_officers) / (num_officers + num_non_officers) = 120 := by
  sorry

#eval (430 * 15 + 110 * 465) / (15 + 465)

end NUMINAMATH_CALUDE_average_salary_calculation_l564_56470


namespace NUMINAMATH_CALUDE_parallel_vectors_m_l564_56432

def vector_a : Fin 3 → ℝ := ![2, 4, 3]
def vector_b (m : ℝ) : Fin 3 → ℝ := ![4, 8, m]

def parallel (u v : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ i, u i = k * v i

theorem parallel_vectors_m (m : ℝ) :
  parallel vector_a (vector_b m) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_l564_56432


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l564_56447

/-- Given a circle intersected by three equally spaced parallel lines creating
    chords of lengths 40, 40, and 36, the distance between adjacent lines is √38. -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (∃ (chord1 chord2 chord3 : ℝ), 
    chord1 = 40 ∧ 
    chord2 = 40 ∧ 
    chord3 = 36 ∧ 
    chord1^2 = 4 * (r^2 - (d/2)^2) ∧ 
    chord2^2 = 4 * (r^2 - (3*d/2)^2) ∧ 
    chord3^2 = 4 * (r^2 - d^2)) → 
  d = Real.sqrt 38 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l564_56447


namespace NUMINAMATH_CALUDE_berry_tuesday_temperature_l564_56408

def berry_temperatures : List Float := [99.1, 98.2, 99.3, 99.8, 99, 98.9]
def average_temperature : Float := 99
def days_in_week : Nat := 7

theorem berry_tuesday_temperature :
  let total_sum : Float := average_temperature * days_in_week.toFloat
  let known_sum : Float := berry_temperatures.sum
  let tuesday_temp : Float := total_sum - known_sum
  tuesday_temp = 98.7 := by sorry

end NUMINAMATH_CALUDE_berry_tuesday_temperature_l564_56408


namespace NUMINAMATH_CALUDE_sector_area_one_radian_unit_radius_l564_56450

/-- The area of a circular sector with central angle 1 radian and radius 1 unit is 1/2 square units. -/
theorem sector_area_one_radian_unit_radius : 
  let θ : Real := 1  -- Central angle in radians
  let r : Real := 1  -- Radius
  let sector_area := (1/2) * r * r * θ
  sector_area = 1/2 := by sorry

end NUMINAMATH_CALUDE_sector_area_one_radian_unit_radius_l564_56450


namespace NUMINAMATH_CALUDE_power_sum_l564_56462

theorem power_sum (a m n : ℝ) (h1 : a^m = 4) (h2 : a^n = 8) : a^(m+n) = 32 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l564_56462


namespace NUMINAMATH_CALUDE_intersection_empty_set_l564_56479

theorem intersection_empty_set (A : Set α) : ¬(¬(A ∩ ∅ = ∅)) := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_set_l564_56479


namespace NUMINAMATH_CALUDE_apple_pie_problem_l564_56428

/-- The number of apples needed per pie -/
def apples_per_pie (total_pies : ℕ) (apples_from_garden : ℕ) (apples_to_buy : ℕ) : ℕ :=
  (apples_from_garden + apples_to_buy) / total_pies

theorem apple_pie_problem :
  apples_per_pie 10 50 30 = 8 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_problem_l564_56428


namespace NUMINAMATH_CALUDE_cranberry_harvest_percentage_l564_56443

/-- Given the initial number of cranberries, the number eaten by elk, and the number left,
    prove that the percentage of cranberries harvested by humans is 40%. -/
theorem cranberry_harvest_percentage
  (total : ℕ)
  (eaten_by_elk : ℕ)
  (left : ℕ)
  (h1 : total = 60000)
  (h2 : eaten_by_elk = 20000)
  (h3 : left = 16000) :
  (total - eaten_by_elk - left) / total * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cranberry_harvest_percentage_l564_56443


namespace NUMINAMATH_CALUDE_infinite_series_solution_l564_56440

theorem infinite_series_solution : ∃! x : ℝ, x = (1 : ℝ) / (1 + x) ∧ |x| < 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_solution_l564_56440


namespace NUMINAMATH_CALUDE_train_passing_time_l564_56490

/-- The time taken for a train to pass a telegraph post -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 90 →
  train_speed_kmh = 36 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l564_56490


namespace NUMINAMATH_CALUDE_sum_m_n_equals_five_l564_56418

theorem sum_m_n_equals_five (m n : ℕ) (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : a = n * b) (h4 : a + b = m * (a - b)) : 
  m + n = 5 :=
sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_five_l564_56418


namespace NUMINAMATH_CALUDE_quarters_count_l564_56434

/-- Calculates the number of quarters in a jar given the following conditions:
  * The jar contains 123 pennies, 85 nickels, 35 dimes, and an unknown number of quarters.
  * The total cost of ice cream for 5 family members is $15.
  * After spending on ice cream, 48 cents remain. -/
def quarters_in_jar (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (ice_cream_cost : ℚ) (remaining_cents : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of quarters in the jar is 26. -/
theorem quarters_count : quarters_in_jar 123 85 35 15 48 = 26 := by
  sorry

end NUMINAMATH_CALUDE_quarters_count_l564_56434


namespace NUMINAMATH_CALUDE_curve_self_intersection_l564_56453

/-- A point on the curve defined by t --/
def curve_point (t : ℝ) : ℝ × ℝ :=
  (t^2 - 4, t^3 - 6*t + 7)

/-- The curve crosses itself if there exist two distinct real numbers that map to the same point --/
def self_intersection (p : ℝ × ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ curve_point a = p ∧ curve_point b = p

theorem curve_self_intersection :
  self_intersection (2, 7) := by
  sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l564_56453


namespace NUMINAMATH_CALUDE_gumball_range_l564_56435

theorem gumball_range (x : ℤ) : 
  let carolyn := 17
  let lew := 12
  let total := carolyn + lew + x
  let avg := total / 3
  (19 ≤ avg ∧ avg ≤ 25) →
  (max x - min x = 18) :=
by sorry

end NUMINAMATH_CALUDE_gumball_range_l564_56435


namespace NUMINAMATH_CALUDE_max_value_abc_l564_56405

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_abc : a + b + c = 3) :
  a^2 * b^3 * c^4 ≤ 2048/19683 ∧ ∃ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 ∧ a^2 * b^3 * c^4 = 2048/19683 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l564_56405


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l564_56410

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geo : geometric_sequence a)
  (h_fourth : a 4 = (a 2) ^ 2)
  (h_sum : a 2 + a 4 = 5 / 16) :
  a 5 = 1 / 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l564_56410


namespace NUMINAMATH_CALUDE_second_discarded_number_l564_56429

theorem second_discarded_number 
  (n₁ : ℕ) (a₁ : ℚ) (n₂ : ℕ) (a₂ : ℚ) (x₁ : ℚ) :
  n₁ = 50 →
  a₁ = 38 →
  n₂ = 48 →
  a₂ = 37.5 →
  x₁ = 45 →
  ∃ x₂ : ℚ, x₂ = 55 ∧ n₁ * a₁ = n₂ * a₂ + x₁ + x₂ :=
by sorry

end NUMINAMATH_CALUDE_second_discarded_number_l564_56429


namespace NUMINAMATH_CALUDE_p_twelve_equals_neg_five_l564_56444

/-- A quadratic function with specific properties -/
def p (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

/-- Theorem stating that p(12) = -5 given certain conditions -/
theorem p_twelve_equals_neg_five 
  (d e f : ℝ) 
  (h1 : ∀ x, p d e f (3.5 + x) = p d e f (3.5 - x)) -- axis of symmetry at x = 3.5
  (h2 : p d e f (-5) = -5) -- p(-5) = -5
  : p d e f 12 = -5 := by
  sorry

end NUMINAMATH_CALUDE_p_twelve_equals_neg_five_l564_56444


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l564_56498

theorem perfect_square_quadratic (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - (m+1)*x + 1 = k^2) → (m = 1 ∨ m = -3) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l564_56498


namespace NUMINAMATH_CALUDE_oreo_count_l564_56433

/-- The number of Oreos James has -/
def james_oreos : ℕ := 43

/-- The number of Oreos Jordan has -/
def jordan_oreos : ℕ := (james_oreos - 7) / 4

/-- The total number of Oreos between James and Jordan -/
def total_oreos : ℕ := james_oreos + jordan_oreos

theorem oreo_count : total_oreos = 52 := by
  sorry

end NUMINAMATH_CALUDE_oreo_count_l564_56433


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_A_l564_56487

open Set

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 0 < x}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem for the union of complement of B and A
theorem union_complement_B_A : (𝒰 \ B) ∪ A = {x | x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_A_l564_56487


namespace NUMINAMATH_CALUDE_total_apples_l564_56406

/-- Given 37 baskets with 17 apples each, prove that the total number of apples is 629. -/
theorem total_apples (num_baskets : ℕ) (apples_per_basket : ℕ) (h1 : num_baskets = 37) (h2 : apples_per_basket = 17) :
  num_baskets * apples_per_basket = 629 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_l564_56406


namespace NUMINAMATH_CALUDE_sally_quarters_l564_56499

theorem sally_quarters (x : ℕ) : 
  (x + 418 = 1178) → (x = 760) := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_l564_56499


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_solve_equation_l564_56439

-- Problem 1
theorem simplify_and_evaluate : 
  let f (x : ℝ) := (x^2 - 6*x + 9) / (x^2 - 1) / ((x^2 - 3*x) / (x + 1))
  f (-3) = -1/2 := by sorry

-- Problem 2
theorem solve_equation :
  ∃ (x : ℝ), x / (x + 1) = 2*x / (3*x + 3) - 1 ∧ x = -3/4 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_solve_equation_l564_56439


namespace NUMINAMATH_CALUDE_polynomial_equality_l564_56424

theorem polynomial_equality (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, a + a₁ * (x - 1) + a₂ * (x - 1)^2 + a₃ * (x - 1)^3 = x^3) →
  (a = 1 ∧ a₂ = 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l564_56424


namespace NUMINAMATH_CALUDE_carlys_dogs_l564_56469

theorem carlys_dogs (total_nails : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) :
  total_nails = 164 →
  three_legged_dogs = 3 →
  nails_per_paw = 4 →
  ∃ (four_legged_dogs : ℕ),
    four_legged_dogs * 4 * nails_per_paw + three_legged_dogs * 3 * nails_per_paw = total_nails ∧
    four_legged_dogs + three_legged_dogs = 11 :=
by sorry

end NUMINAMATH_CALUDE_carlys_dogs_l564_56469


namespace NUMINAMATH_CALUDE_cylinder_volume_l564_56483

theorem cylinder_volume (r : ℝ) (h : ℝ) : 
  r > 0 → h > 0 → h = 2 * r → 2 * π * r * h = π → π * r^2 * h = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l564_56483


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l564_56402

theorem cow_chicken_problem (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 2 * (cows + chickens) + 12) → cows = 6 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l564_56402


namespace NUMINAMATH_CALUDE_internet_bill_is_100_l564_56475

/-- Represents the financial transactions and balances in Liza's checking account --/
structure AccountState where
  initialBalance : ℕ
  rentPayment : ℕ
  paycheckDeposit : ℕ
  electricityBill : ℕ
  phoneBill : ℕ
  finalBalance : ℕ

/-- Calculates the internet bill given the account state --/
def calculateInternetBill (state : AccountState) : ℕ :=
  state.initialBalance + state.paycheckDeposit - state.rentPayment - state.electricityBill - state.phoneBill - state.finalBalance

/-- Theorem stating that the internet bill is $100 given the specified account state --/
theorem internet_bill_is_100 (state : AccountState) 
  (h1 : state.initialBalance = 800)
  (h2 : state.rentPayment = 450)
  (h3 : state.paycheckDeposit = 1500)
  (h4 : state.electricityBill = 117)
  (h5 : state.phoneBill = 70)
  (h6 : state.finalBalance = 1563) :
  calculateInternetBill state = 100 := by
  sorry

end NUMINAMATH_CALUDE_internet_bill_is_100_l564_56475


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_one_l564_56437

-- Define the curve function
def f (x : ℝ) : ℝ := x^2 + 3*x - 1

-- Define the derivative of the curve function
def f' (x : ℝ) : ℝ := 2*x + 3

theorem tangent_slope_at_point_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  f' x₀ = 5 ∧ f x₀ = y₀ ∧ y₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_one_l564_56437


namespace NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l564_56403

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + a = 0) :
  a + b ≥ 16 + (4/3) * Real.rpow 3 (1/3) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l564_56403


namespace NUMINAMATH_CALUDE_square_side_length_l564_56456

theorem square_side_length (perimeter : ℝ) (h : perimeter = 100) : 
  ∃ (side_length : ℝ), side_length = 25 ∧ 4 * side_length = perimeter := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l564_56456


namespace NUMINAMATH_CALUDE_animal_path_distance_l564_56411

theorem animal_path_distance : 
  let outer_radius : ℝ := 25
  let middle_radius : ℝ := 15
  let inner_radius : ℝ := 5
  let outer_arc : ℝ := (1/4) * 2 * Real.pi * outer_radius
  let middle_to_outer : ℝ := outer_radius - middle_radius
  let middle_arc : ℝ := (1/4) * 2 * Real.pi * middle_radius
  let to_center_and_back : ℝ := 2 * middle_radius
  let middle_to_inner : ℝ := middle_radius - inner_radius
  outer_arc + middle_to_outer + middle_arc + to_center_and_back + middle_arc + middle_to_inner = 27.5 * Real.pi + 50 := by
  sorry

end NUMINAMATH_CALUDE_animal_path_distance_l564_56411


namespace NUMINAMATH_CALUDE_sqrt_floor_equality_l564_56407

theorem sqrt_floor_equality (n : ℕ) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_sqrt_floor_equality_l564_56407


namespace NUMINAMATH_CALUDE_goldfish_preference_l564_56485

theorem goldfish_preference (total_students : ℕ) (johnson_fraction : ℚ) (henderson_fraction : ℚ) (total_preference : ℕ) :
  total_students = 30 →
  johnson_fraction = 1/6 →
  henderson_fraction = 1/5 →
  total_preference = 31 →
  ∃ feldstein_fraction : ℚ,
    feldstein_fraction = 2/3 ∧
    total_preference = johnson_fraction * total_students + henderson_fraction * total_students + feldstein_fraction * total_students :=
by
  sorry

end NUMINAMATH_CALUDE_goldfish_preference_l564_56485


namespace NUMINAMATH_CALUDE_problem_solution_l564_56476

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = 0) : 
  y = 21/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l564_56476


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l564_56489

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_is_zero :
  i^12345 + i^12346 + i^12347 + i^12348 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l564_56489


namespace NUMINAMATH_CALUDE_notebooks_needed_correct_l564_56496

/-- The number of notebooks needed to achieve a profit of $40 -/
def notebooks_needed : ℕ := 96

/-- The cost of 4 notebooks in dollars -/
def cost_of_four : ℚ := 15

/-- The selling price of 6 notebooks in dollars -/
def sell_price_of_six : ℚ := 25

/-- The desired profit in dollars -/
def desired_profit : ℚ := 40

/-- Theorem stating that the number of notebooks needed to achieve the desired profit is correct -/
theorem notebooks_needed_correct : 
  (notebooks_needed : ℚ) * (sell_price_of_six / 6 - cost_of_four / 4) ≥ desired_profit ∧
  ((notebooks_needed - 1) : ℚ) * (sell_price_of_six / 6 - cost_of_four / 4) < desired_profit :=
by sorry

end NUMINAMATH_CALUDE_notebooks_needed_correct_l564_56496


namespace NUMINAMATH_CALUDE_daily_wage_c_value_l564_56495

/-- The daily wage of worker c given the conditions of the problem -/
def daily_wage_c (days_a days_b days_c : ℕ) 
                 (ratio_a ratio_b ratio_c : ℕ) 
                 (total_earning : ℚ) : ℚ :=
  let wage_a := total_earning * 3 / (ratio_a * days_a + ratio_b * days_b + ratio_c * days_c)
  wage_a * ratio_c / ratio_a

/-- Theorem stating that the daily wage of c is $66.67 given the problem conditions -/
theorem daily_wage_c_value : 
  daily_wage_c 6 9 4 3 4 5 1480 = 200/3 := by sorry

end NUMINAMATH_CALUDE_daily_wage_c_value_l564_56495


namespace NUMINAMATH_CALUDE_game_value_proof_l564_56404

def super_nintendo_value : ℝ := 150
def store_credit_percentage : ℝ := 0.8
def tom_payment : ℝ := 80
def tom_change : ℝ := 10
def nes_sale_price : ℝ := 160

theorem game_value_proof :
  let credit := super_nintendo_value * store_credit_percentage
  let tom_actual_payment := tom_payment - tom_change
  let credit_used := nes_sale_price - tom_actual_payment
  credit - credit_used = 30 := by sorry

end NUMINAMATH_CALUDE_game_value_proof_l564_56404


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l564_56446

/-- Given planar vectors a and b satisfying the conditions,
    prove that the cosine of the angle between them is 1/2 -/
theorem cosine_of_angle_between_vectors
  (a b : ℝ × ℝ)  -- Planar vectors represented as pairs of real numbers
  (h1 : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 5)  -- a · (a + b) = 5
  (h2 : a.1^2 + a.2^2 = 4)  -- |a| = 2
  (h3 : b.1^2 + b.2^2 = 1)  -- |b| = 1
  : (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l564_56446


namespace NUMINAMATH_CALUDE_saturday_hourly_rate_l564_56436

/-- Calculates the hourly rate for Saturday work given the following conditions:
  * After-school hourly rate is $4.00
  * Total weekly hours worked is 18
  * Total weekly earnings is $88.00
  * Saturday hours worked is 8.0
-/
theorem saturday_hourly_rate
  (after_school_rate : ℝ)
  (total_hours : ℝ)
  (total_earnings : ℝ)
  (saturday_hours : ℝ)
  (h1 : after_school_rate = 4)
  (h2 : total_hours = 18)
  (h3 : total_earnings = 88)
  (h4 : saturday_hours = 8) :
  (total_earnings - after_school_rate * (total_hours - saturday_hours)) / saturday_hours = 6 :=
by sorry

end NUMINAMATH_CALUDE_saturday_hourly_rate_l564_56436


namespace NUMINAMATH_CALUDE_donut_selection_count_l564_56414

/-- The number of types of donuts available -/
def num_donut_types : ℕ := 3

/-- The number of donuts Pat wants to buy -/
def num_donuts_to_buy : ℕ := 4

/-- The number of ways to select donuts -/
def num_selections : ℕ := (num_donuts_to_buy + num_donut_types - 1).choose (num_donut_types - 1)

theorem donut_selection_count : num_selections = 15 := by
  sorry

end NUMINAMATH_CALUDE_donut_selection_count_l564_56414


namespace NUMINAMATH_CALUDE_coefficient_expansion_l564_56486

theorem coefficient_expansion (m : ℝ) : 
  (∃ c : ℝ, c = -160 ∧ c = 20 * m^3) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_expansion_l564_56486


namespace NUMINAMATH_CALUDE_exponent_problem_l564_56445

theorem exponent_problem (x m n : ℝ) (hm : x^m = 5) (hn : x^n = -2) : x^(m+2*n) = 20 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l564_56445


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l564_56420

theorem absolute_value_inequality (x : ℝ) :
  (3 ≤ |x + 3| ∧ |x + 3| ≤ 7) ↔ ((-10 ≤ x ∧ x ≤ -6) ∨ (0 ≤ x ∧ x ≤ 4)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l564_56420


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l564_56471

def is_valid_rectangle (l w : ℕ) : Prop :=
  2 * l + 2 * w = 56 ∧ (l ≥ w + 5 ∨ w ≥ l + 5)

def rectangle_area (l w : ℕ) : ℕ := l * w

theorem rectangle_area_difference : 
  ∃ (l₁ w₁ l₂ w₂ : ℕ),
    is_valid_rectangle l₁ w₁ ∧
    is_valid_rectangle l₂ w₂ ∧
    ∀ (l w : ℕ),
      is_valid_rectangle l w →
      rectangle_area l w ≤ rectangle_area l₁ w₁ ∧
      rectangle_area l w ≥ rectangle_area l₂ w₂ ∧
      rectangle_area l₁ w₁ - rectangle_area l₂ w₂ = 5 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l564_56471


namespace NUMINAMATH_CALUDE_b2f_to_decimal_l564_56401

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9
| A | B | C | D | E | F

/-- Converts a hexadecimal digit to its decimal value --/
def hexToDecimal (d : HexDigit) : ℕ :=
  match d with
  | HexDigit.D0 => 0
  | HexDigit.D1 => 1
  | HexDigit.D2 => 2
  | HexDigit.D3 => 3
  | HexDigit.D4 => 4
  | HexDigit.D5 => 5
  | HexDigit.D6 => 6
  | HexDigit.D7 => 7
  | HexDigit.D8 => 8
  | HexDigit.D9 => 9
  | HexDigit.A => 10
  | HexDigit.B => 11
  | HexDigit.C => 12
  | HexDigit.D => 13
  | HexDigit.E => 14
  | HexDigit.F => 15

/-- Converts a list of hexadecimal digits to its decimal value --/
def hexListToDecimal (l : List HexDigit) : ℕ :=
  l.enum.foldr (fun (i, d) acc => acc + (hexToDecimal d) * (16 ^ i)) 0

theorem b2f_to_decimal :
  hexListToDecimal [HexDigit.B, HexDigit.D2, HexDigit.F] = 2863 := by
  sorry

end NUMINAMATH_CALUDE_b2f_to_decimal_l564_56401


namespace NUMINAMATH_CALUDE_range_of_g_l564_56491

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → 29 ≤ g x ∧ g x ≤ 93 :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l564_56491


namespace NUMINAMATH_CALUDE_emily_cell_phone_cost_l564_56464

/-- Calculates the total cost of a cell phone plan based on given parameters. -/
def calculate_total_cost (base_cost : ℕ) (included_hours : ℕ) (extra_hour_cost : ℕ)
  (base_message_cost : ℕ) (base_message_limit : ℕ) (hours_used : ℕ) (messages_sent : ℕ) : ℕ :=
  let extra_hours := max (hours_used - included_hours) 0
  let extra_hour_charge := extra_hours * extra_hour_cost
  let base_message_charge := min messages_sent base_message_limit * base_message_cost
  let extra_messages := max (messages_sent - base_message_limit) 0
  let extra_message_charge := extra_messages * (2 * base_message_cost)
  base_cost + extra_hour_charge + base_message_charge + extra_message_charge

/-- Emily's cell phone plan cost theorem -/
theorem emily_cell_phone_cost :
  calculate_total_cost 30 50 15 10 150 52 200 = 8500 :=
by sorry

end NUMINAMATH_CALUDE_emily_cell_phone_cost_l564_56464


namespace NUMINAMATH_CALUDE_shooting_scenarios_correct_l564_56419

/-- Represents a shooting scenario with a total number of shots and hits -/
structure ShootingScenario where
  totalShots : Nat
  totalHits : Nat

/-- Calculates the number of possible situations for Scenario 1 -/
def scenario1Situations (s : ShootingScenario) : Nat :=
  if s.totalShots = 10 ∧ s.totalHits = 7 then
    12
  else
    0

/-- Calculates the number of possible situations for Scenario 2 -/
def scenario2Situations (s : ShootingScenario) : Nat :=
  if s.totalShots = 10 then
    144
  else
    0

/-- Calculates the number of possible situations for Scenario 3 -/
def scenario3Situations (s : ShootingScenario) : Nat :=
  if s.totalShots = 10 ∧ s.totalHits = 6 then
    50
  else
    0

theorem shooting_scenarios_correct :
  ∀ s : ShootingScenario,
    (scenario1Situations s = 12 ∨ scenario1Situations s = 0) ∧
    (scenario2Situations s = 144 ∨ scenario2Situations s = 0) ∧
    (scenario3Situations s = 50 ∨ scenario3Situations s = 0) :=
by sorry

end NUMINAMATH_CALUDE_shooting_scenarios_correct_l564_56419


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l564_56422

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 8 * x * y) : 1 / x + 1 / y = 8 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l564_56422


namespace NUMINAMATH_CALUDE_notebook_pencil_cost_l564_56451

/-- Given the prices of notebooks and pencils in two scenarios, prove the cost of one notebook and one pencil. -/
theorem notebook_pencil_cost
  (scenario1 : 6 * notebook_price + 4 * pencil_price = 9.2)
  (scenario2 : 3 * notebook_price + pencil_price = 3.8)
  : notebook_price + pencil_price = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_notebook_pencil_cost_l564_56451


namespace NUMINAMATH_CALUDE_equation_holds_iff_specific_values_l564_56430

theorem equation_holds_iff_specific_values :
  ∀ (a b p q : ℝ),
  (∀ x : ℝ, (2*x - 1)^20 - (a*x + b)^20 = (x^2 + p*x + q)^10) ↔
  ((b = (1/2) * (2^20 - 1)^(1/20) ∧ a = -(2^20 - 1)^(1/20)) ∨
   (b = -(1/2) * (2^20 - 1)^(1/20) ∧ a = (2^20 - 1)^(1/20))) ∧
  p = -1 ∧ q = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_equation_holds_iff_specific_values_l564_56430


namespace NUMINAMATH_CALUDE_train_distance_problem_l564_56466

theorem train_distance_problem (speed1 speed2 distance_diff : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 25)
  (h3 : distance_diff = 55)
  (h4 : speed1 > 0)
  (h5 : speed2 > 0) :
  ∃ (time distance1 distance2 : ℝ),
    time > 0 ∧
    distance1 = speed1 * time ∧
    distance2 = speed2 * time ∧
    distance2 = distance1 + distance_diff ∧
    distance1 + distance2 = 495 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_problem_l564_56466


namespace NUMINAMATH_CALUDE_sheridan_cats_bought_l564_56409

/-- The number of cats Mrs. Sheridan bought -/
def cats_bought (initial final : ℝ) : ℝ := final - initial

/-- Theorem stating that Mrs. Sheridan bought 43 cats -/
theorem sheridan_cats_bought :
  cats_bought 11.0 54 = 43 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_cats_bought_l564_56409


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_means_l564_56460

theorem arithmetic_and_geometric_means : 
  (let a := (5 + 17) / 2
   a = 11) ∧
  (let b := Real.sqrt (4 * 9)
   b = 6 ∨ b = -6) := by sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_means_l564_56460


namespace NUMINAMATH_CALUDE_moving_points_theorem_l564_56463

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- The main theorem -/
theorem moving_points_theorem (ABC : Triangle) (P Q : Point) (t : ℝ) :
  (ABC.B.x - ABC.A.x)^2 + (ABC.B.y - ABC.A.y)^2 = 36 →  -- AB = 6 cm
  (ABC.C.x - ABC.B.x)^2 + (ABC.C.y - ABC.B.y)^2 = 64 →  -- BC = 8 cm
  (ABC.C.x - ABC.B.x) * (ABC.B.y - ABC.A.y) = (ABC.C.y - ABC.B.y) * (ABC.B.x - ABC.A.x) →  -- ABC is right-angled at B
  P.x = ABC.A.x + t →  -- P moves from A towards B
  P.y = ABC.A.y →
  Q.x = ABC.B.x + 2 * t →  -- Q moves from B towards C
  Q.y = ABC.B.y →
  triangleArea P ABC.B Q = 5 →  -- Area of PBQ is 5 cm²
  t = 1  -- Time P moves is 1 second
  := by sorry

end NUMINAMATH_CALUDE_moving_points_theorem_l564_56463


namespace NUMINAMATH_CALUDE_sqrt_sum_quotient_l564_56459

theorem sqrt_sum_quotient : (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 75 = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_quotient_l564_56459


namespace NUMINAMATH_CALUDE_charlie_cleaning_time_l564_56467

theorem charlie_cleaning_time (alice_time bob_time charlie_time : ℚ) : 
  alice_time = 30 →
  bob_time = (3 / 4) * alice_time →
  charlie_time = (1 / 3) * bob_time →
  charlie_time = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_charlie_cleaning_time_l564_56467


namespace NUMINAMATH_CALUDE_original_price_calculation_l564_56473

theorem original_price_calculation (final_price : ℝ) : 
  final_price = 1120 → 
  ∃ (original_price : ℝ), 
    original_price * (1 - 0.3) * (1 - 0.2) = final_price ∧ 
    original_price = 2000 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l564_56473


namespace NUMINAMATH_CALUDE_cosine_equality_problem_l564_56480

theorem cosine_equality_problem :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (1018 * π / 180) ∧ n = 62 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_problem_l564_56480


namespace NUMINAMATH_CALUDE_f_properties_l564_56457

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_properties (a : ℝ) :
  (∀ x, a ≤ 0 → deriv (f a) x ≤ 0) ∧
  (a > 0 → ∀ x, x < Real.log (1/a) → deriv (f a) x < 0) ∧
  (a > 0 → ∀ x, x > Real.log (1/a) → deriv (f a) x > 0) ∧
  (a > 0 → ∀ x, f a x > 2 * Real.log a + 3/2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l564_56457


namespace NUMINAMATH_CALUDE_h_piecewise_l564_56427

/-- Piecewise function g(x) -/
noncomputable def g (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then 3 - x
  else if 0 ≤ x ∧ x ≤ 2 then Real.sqrt (9 - (x - 1.5)^2) - 3
  else if 2 ≤ x ∧ x ≤ 4 then 3 * (x - 2)
  else 0

/-- Function h(x) = g(x) + g(-x) -/
noncomputable def h (x : ℝ) : ℝ := g x + g (-x)

theorem h_piecewise :
  ∀ x : ℝ,
    ((-4 ≤ x ∧ x < -3) → h x = -3 * (x + 2)) ∧
    ((-3 ≤ x ∧ x < 0) → h x = 6) ∧
    ((0 ≤ x ∧ x < 2) → h x = 2 * Real.sqrt (9 - (x - 1.5)^2) - 6) ∧
    ((2 ≤ x ∧ x ≤ 4) → h x = 3 * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_h_piecewise_l564_56427


namespace NUMINAMATH_CALUDE_least_marbles_divisible_l564_56412

theorem least_marbles_divisible (n : ℕ) : n > 0 ∧ 
  (∀ k ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ), n % k = 0) →
  n ≥ 420 :=
by sorry

end NUMINAMATH_CALUDE_least_marbles_divisible_l564_56412
