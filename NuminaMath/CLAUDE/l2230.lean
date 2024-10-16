import Mathlib

namespace NUMINAMATH_CALUDE_chicken_admission_combinations_l2230_223037

theorem chicken_admission_combinations : Nat.choose 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_chicken_admission_combinations_l2230_223037


namespace NUMINAMATH_CALUDE_sample_size_is_six_l2230_223040

/-- Represents the total number of teachers -/
def total_teachers : ℕ := 18 + 12 + 6

/-- Represents the sample size -/
def n : ℕ := 6

/-- Checks if a number divides the total number of teachers -/
def divides_total (k : ℕ) : Prop := k ∣ total_teachers

/-- Checks if stratified sampling works for a given sample size -/
def stratified_sampling_works (k : ℕ) : Prop :=
  k ∣ 18 ∧ k ∣ 12 ∧ k ∣ 6

/-- Checks if systematic sampling works for a given sample size -/
def systematic_sampling_works (k : ℕ) : Prop :=
  divides_total k

/-- Checks if increasing the sample size by 1 requires excluding 1 person for systematic sampling -/
def exclusion_condition (k : ℕ) : Prop :=
  ¬(divides_total (k + 1)) ∧ ((k + 1) ∣ (total_teachers - 1))

theorem sample_size_is_six :
  stratified_sampling_works n ∧
  systematic_sampling_works n ∧
  exclusion_condition n ∧
  ∀ m : ℕ, m ≠ n →
    ¬(stratified_sampling_works m ∧
      systematic_sampling_works m ∧
      exclusion_condition m) :=
sorry

end NUMINAMATH_CALUDE_sample_size_is_six_l2230_223040


namespace NUMINAMATH_CALUDE_satisfactory_grades_fraction_l2230_223097

/-- Represents the grades in a class -/
structure ClassGrades where
  total_students : Nat
  grade_a : Nat
  grade_b : Nat
  grade_c : Nat
  grade_d : Nat
  grade_f : Nat

/-- Calculates the fraction of satisfactory grades -/
def satisfactory_fraction (grades : ClassGrades) : Rat :=
  (grades.grade_a + grades.grade_b + grades.grade_c : Rat) / grades.total_students

/-- The main theorem about the fraction of satisfactory grades -/
theorem satisfactory_grades_fraction :
  let grades : ClassGrades := {
    total_students := 30,
    grade_a := 8,
    grade_b := 7,
    grade_c := 6,
    grade_d := 5,
    grade_f := 4
  }
  satisfactory_fraction grades = 7 / 10 := by sorry

end NUMINAMATH_CALUDE_satisfactory_grades_fraction_l2230_223097


namespace NUMINAMATH_CALUDE_fourth_root_of_506250000_l2230_223058

theorem fourth_root_of_506250000 : (506250000 : ℝ) ^ (1/4 : ℝ) = 150 := by sorry

end NUMINAMATH_CALUDE_fourth_root_of_506250000_l2230_223058


namespace NUMINAMATH_CALUDE_rival_awards_l2230_223045

/-- Given Scott won 4 awards, Jessie won 3 times as many awards as Scott,
    and the rival won twice as many awards as Jessie,
    prove that the rival won 24 awards. -/
theorem rival_awards (scott_awards : ℕ) (jessie_awards : ℕ) (rival_awards : ℕ)
    (h1 : scott_awards = 4)
    (h2 : jessie_awards = 3 * scott_awards)
    (h3 : rival_awards = 2 * jessie_awards) :
  rival_awards = 24 := by
  sorry

end NUMINAMATH_CALUDE_rival_awards_l2230_223045


namespace NUMINAMATH_CALUDE_area_between_tangent_circles_l2230_223099

/-- The area of the region between two tangent circles -/
theorem area_between_tangent_circles (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₂ = 3 * r₁) :
  π * r₂^2 - π * r₁^2 = 32 * π * r₁^2 :=
sorry

end NUMINAMATH_CALUDE_area_between_tangent_circles_l2230_223099


namespace NUMINAMATH_CALUDE_square_root_sum_equals_ten_l2230_223025

theorem square_root_sum_equals_ten : 
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_ten_l2230_223025


namespace NUMINAMATH_CALUDE_unique_sequence_sum_property_l2230_223030

-- Define the sequence type
def UniqueIntegerSequence := ℕ+ → ℕ+

-- Define the property that every positive integer occurs exactly once
def IsUniqueSequence (a : UniqueIntegerSequence) : Prop :=
  ∀ n : ℕ+, ∃! k : ℕ+, a k = n

-- State the theorem
theorem unique_sequence_sum_property (a : UniqueIntegerSequence) 
    (h : IsUniqueSequence a) : 
    ∃ ℓ m : ℕ+, 1 < ℓ ∧ ℓ < m ∧ a 1 + a m = 2 * a ℓ := by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_sum_property_l2230_223030


namespace NUMINAMATH_CALUDE_calculate_expression_l2230_223019

theorem calculate_expression : 2023^0 + (-1/3) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2230_223019


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2230_223000

theorem negation_of_proposition (p : (ℝ × ℝ) → Prop) : 
  (¬ ∀ (x y : ℝ), x^2 + y^2 ≥ 0) ↔ (∃ (x y : ℝ), x^2 + y^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2230_223000


namespace NUMINAMATH_CALUDE_dress_discount_percentage_l2230_223096

theorem dress_discount_percentage (d : ℝ) (x : ℝ) 
  (h1 : d > 0) 
  (h2 : 0.6 * d = d * (1 - x / 100) * 0.8) : 
  x = 25 := by sorry

end NUMINAMATH_CALUDE_dress_discount_percentage_l2230_223096


namespace NUMINAMATH_CALUDE_complex_number_property_l2230_223011

theorem complex_number_property (a b : ℝ) (h1 : b ≠ 0) : 
  let z : ℂ := Complex.mk a b
  (z^2 - 4*b*z).im = 0 → a = 2*b := by
  sorry

end NUMINAMATH_CALUDE_complex_number_property_l2230_223011


namespace NUMINAMATH_CALUDE_max_M_value_l2230_223015

theorem max_M_value (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (eq1 : x - 2*y = z - 2*u) (eq2 : 2*y*z = u*x) (hzy : z ≥ y) :
  ∃ (M : ℝ), M > 0 ∧ M ≤ z/y ∧ ∀ (N : ℝ), (N > 0 ∧ N ≤ z/y) → N ≤ 6 + 4*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_M_value_l2230_223015


namespace NUMINAMATH_CALUDE_swimmers_meet_problem_l2230_223046

/-- Represents the number of times two swimmers meet in a pool -/
def swimmers_meet (pool_length : ℝ) (speed_a speed_b : ℝ) (time : ℝ) : ℕ :=
  sorry

theorem swimmers_meet_problem :
  swimmers_meet 90 3 2 (12 * 60) = 20 := by sorry

end NUMINAMATH_CALUDE_swimmers_meet_problem_l2230_223046


namespace NUMINAMATH_CALUDE_function_inequality_l2230_223006

open Real

/-- A function satisfying the given conditions -/
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧
  (∀ x, (x - 1) * (deriv f x - f x) > 0) ∧
  (∀ x, f (2 - x) = f x * Real.exp (2 - 2*x))

/-- The main theorem -/
theorem function_inequality (f : ℝ → ℝ) (h : satisfies_conditions f) :
  f 3 < Real.exp 3 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2230_223006


namespace NUMINAMATH_CALUDE_quadratic_one_solution_negative_k_value_l2230_223039

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x, 9 * x^2 + k * x + 36 = 0) ↔ k = 36 ∨ k = -36 :=
by sorry

theorem negative_k_value (k : ℝ) : 
  (∃! x, 9 * x^2 + k * x + 36 = 0) → k = -36 ∨ k = 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_negative_k_value_l2230_223039


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2230_223047

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x + 3 < 3*x + 1 ∧ x > m + 1) ↔ x > 1) → 
  m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2230_223047


namespace NUMINAMATH_CALUDE_cricket_average_increase_l2230_223092

def increase_average (current_innings : ℕ) (current_average : ℚ) (next_innings_runs : ℕ) : ℚ :=
  let total_runs := current_innings * current_average
  let new_total_runs := total_runs + next_innings_runs
  let new_average := new_total_runs / (current_innings + 1)
  new_average - current_average

theorem cricket_average_increase :
  increase_average 12 48 178 = 10 := by sorry

end NUMINAMATH_CALUDE_cricket_average_increase_l2230_223092


namespace NUMINAMATH_CALUDE_most_precise_announcement_l2230_223068

def K_approx : ℝ := 5.72788
def error_margin : ℝ := 0.00625

def is_valid_announcement (x : ℝ) : Prop :=
  ∀ y : ℝ, |y - K_approx| ≤ error_margin → |x - y| < 0.05

theorem most_precise_announcement :
  is_valid_announcement 5.7 ∧
  ∀ z : ℝ, is_valid_announcement z → |z - 5.7| < 0.05 :=
sorry

end NUMINAMATH_CALUDE_most_precise_announcement_l2230_223068


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_doubled_l2230_223057

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of sides in a dodecagon -/
def dodecagon_sides : ℕ := 12

/-- The theorem stating that the number of diagonals in a dodecagon, when doubled, is 108 -/
theorem dodecagon_diagonals_doubled :
  2 * (num_diagonals dodecagon_sides) = 108 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_doubled_l2230_223057


namespace NUMINAMATH_CALUDE_average_non_defective_cookies_l2230_223095

def cookie_counts : List Nat := [9, 11, 13, 16, 17, 18, 21, 22]

theorem average_non_defective_cookies :
  (cookie_counts.sum : ℚ) / cookie_counts.length = 127 / 8 := by
  sorry

end NUMINAMATH_CALUDE_average_non_defective_cookies_l2230_223095


namespace NUMINAMATH_CALUDE_x_percent_of_x_squared_is_nine_l2230_223049

theorem x_percent_of_x_squared_is_nine (x : ℝ) (h1 : x > 0) (h2 : x / 100 * x^2 = 9) : x = 10 * Real.rpow 3 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_x_percent_of_x_squared_is_nine_l2230_223049


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2230_223093

/-- 
Given a rectangle with perimeter 60 units and one side at least half the length of the other,
the maximum possible area is 200 square units.
-/
theorem rectangle_max_area : 
  ∀ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧                 -- sides are positive
  2 * (a + b) = 60 ∧              -- perimeter is 60
  a ≥ (1/2) * b ∧ b ≥ (1/2) * a → -- one side is at least half the other
  a * b ≤ 200 :=                  -- area is at most 200
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2230_223093


namespace NUMINAMATH_CALUDE_log_inequality_condition_l2230_223053

theorem log_inequality_condition (a b : ℝ) : 
  (∀ a b, Real.log a > Real.log b → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(Real.log a > Real.log b)) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_condition_l2230_223053


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2230_223061

theorem quadratic_two_distinct_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 - 3*x₁ + 1 = 0) ∧ (x₂^2 - 3*x₂ + 1 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2230_223061


namespace NUMINAMATH_CALUDE_inequality_count_l2230_223005

theorem inequality_count (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : x^2 < a^2) (hyb : |y| < |b|) : 
  ∃! (n : ℕ), n = 2 ∧ 
  (n = (if x^2 + y^2 < a^2 + b^2 then 1 else 0) +
       (if x^2 - y^2 < a^2 - b^2 then 1 else 0) +
       (if x^2 * y^2 < a^2 * b^2 then 1 else 0) +
       (if x^2 / y^2 < a^2 / b^2 then 1 else 0)) :=
sorry

end NUMINAMATH_CALUDE_inequality_count_l2230_223005


namespace NUMINAMATH_CALUDE_solve_for_x_l2230_223076

theorem solve_for_x (x y z : ℝ) 
  (eq1 : x + y = 75)
  (eq2 : (x + y) + y + z = 130)
  (eq3 : z = y + 10) :
  x = 52.5 := by sorry

end NUMINAMATH_CALUDE_solve_for_x_l2230_223076


namespace NUMINAMATH_CALUDE_range_of_a_l2230_223083

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0 ∧ x^2 - 2*(a + 7)*x + 5 ≤ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a ↔ -4 ≤ a ∧ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2230_223083


namespace NUMINAMATH_CALUDE_max_distance_between_sine_cosine_curves_l2230_223017

theorem max_distance_between_sine_cosine_curves : ∃ (C : ℝ),
  (∀ (m : ℝ), |2 * Real.sin m - 2 * Real.sqrt 3 * Real.cos m| ≤ C) ∧
  (∃ (m : ℝ), |2 * Real.sin m - 2 * Real.sqrt 3 * Real.cos m| = C) ∧
  C = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_between_sine_cosine_curves_l2230_223017


namespace NUMINAMATH_CALUDE_prime_pythagorean_inequality_l2230_223052

theorem prime_pythagorean_inequality (p m n : ℕ) 
  (hp : Prime p) 
  (hm : m > 0) 
  (hn : n > 0) 
  (heq : p^2 + m^2 = n^2) : 
  m > p := by
sorry

end NUMINAMATH_CALUDE_prime_pythagorean_inequality_l2230_223052


namespace NUMINAMATH_CALUDE_line_charts_show_trend_bar_charts_dont_l2230_223072

-- Define the types of charts
inductive Chart
| BarChart
| LineChart

-- Define the capabilities of charts
def can_show_amount (c : Chart) : Prop :=
  match c with
  | Chart.BarChart => true
  | Chart.LineChart => true

def can_reflect_changes (c : Chart) : Prop :=
  match c with
  | Chart.BarChart => false
  | Chart.LineChart => true

-- Define what it means to show a trend
def can_show_trend (c : Chart) : Prop :=
  can_show_amount c ∧ can_reflect_changes c

-- Theorem statement
theorem line_charts_show_trend_bar_charts_dont :
  can_show_trend Chart.LineChart ∧ ¬can_show_trend Chart.BarChart :=
sorry

end NUMINAMATH_CALUDE_line_charts_show_trend_bar_charts_dont_l2230_223072


namespace NUMINAMATH_CALUDE_village_population_equality_l2230_223055

/-- The rate of population increase for Village Y -/
def rate_Y : ℕ := sorry

/-- The initial population of Village X -/
def pop_X : ℕ := 76000

/-- The initial population of Village Y -/
def pop_Y : ℕ := 42000

/-- The rate of population decrease for Village X -/
def rate_X : ℕ := 1200

/-- The number of years after which the populations will be equal -/
def years : ℕ := 17

theorem village_population_equality :
  pop_X - (rate_X * years) = pop_Y + (rate_Y * years) ∧ rate_Y = 800 := by sorry

end NUMINAMATH_CALUDE_village_population_equality_l2230_223055


namespace NUMINAMATH_CALUDE_daisy_spending_difference_l2230_223051

def breakfast_muffin1_price : ℚ := 2
def breakfast_muffin2_price : ℚ := 3
def breakfast_coffee1_price : ℚ := 4
def breakfast_coffee2_discount : ℚ := 0.5
def lunch_soup_price : ℚ := 3.75
def lunch_salad_price : ℚ := 5.75
def lunch_lemonade_price : ℚ := 1
def lunch_service_charge_percent : ℚ := 10

def breakfast_total : ℚ := breakfast_muffin1_price + breakfast_muffin2_price + breakfast_coffee1_price + (breakfast_coffee1_price - breakfast_coffee2_discount)

def lunch_subtotal : ℚ := lunch_soup_price + lunch_salad_price + lunch_lemonade_price

def lunch_total : ℚ := lunch_subtotal + (lunch_subtotal * lunch_service_charge_percent / 100)

theorem daisy_spending_difference : lunch_total - breakfast_total = -0.95 := by
  sorry

end NUMINAMATH_CALUDE_daisy_spending_difference_l2230_223051


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2230_223022

/-- The quadratic equation x^2 - 2x - 6 = 0 has two distinct real roots -/
theorem quadratic_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*x₁ - 6 = 0) ∧ 
  (x₂^2 - 2*x₂ - 6 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2230_223022


namespace NUMINAMATH_CALUDE_piano_cost_solution_l2230_223020

def piano_cost_problem (total_lessons : ℕ) (lesson_cost : ℚ) (discount_percent : ℚ) (total_cost : ℚ) : Prop :=
  let original_lesson_cost := total_lessons * lesson_cost
  let discount_amount := discount_percent * original_lesson_cost
  let discounted_lesson_cost := original_lesson_cost - discount_amount
  let piano_cost := total_cost - discounted_lesson_cost
  piano_cost = 500

theorem piano_cost_solution :
  piano_cost_problem 20 40 0.25 1100 := by
  sorry

end NUMINAMATH_CALUDE_piano_cost_solution_l2230_223020


namespace NUMINAMATH_CALUDE_expression_evaluation_l2230_223059

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2230_223059


namespace NUMINAMATH_CALUDE_bobby_total_pieces_l2230_223018

def total_pieces_eaten (initial_candy : ℕ) (initial_chocolate : ℕ) (initial_licorice : ℕ) 
                       (additional_candy : ℕ) (additional_chocolate : ℕ) : ℕ :=
  (initial_candy + additional_candy) + (initial_chocolate + additional_chocolate) + initial_licorice

theorem bobby_total_pieces : 
  total_pieces_eaten 33 14 7 4 5 = 63 := by
  sorry

end NUMINAMATH_CALUDE_bobby_total_pieces_l2230_223018


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l2230_223091

/-- Represents a modified cube with smaller cubes removed from its corners -/
structure ModifiedCube where
  originalSideLength : ℕ
  removedCubeSideLength : ℕ

/-- Calculates the number of edges in a modified cube -/
def edgeCount (cube : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that a cube of side length 4 with unit cubes removed from corners has 48 edges -/
theorem modified_cube_edge_count :
  let cube : ModifiedCube := ⟨4, 1⟩
  edgeCount cube = 48 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_edge_count_l2230_223091


namespace NUMINAMATH_CALUDE_min_flips_theorem_l2230_223085

/-- Represents the color of a hat -/
inductive HatColor
| Blue
| Red

/-- Represents a gnome with a hat -/
structure Gnome where
  hat : HatColor

/-- Represents the state of all gnomes -/
def GnomeState := Fin 1000 → Gnome

/-- Counts the number of hat flips needed to reach a given state -/
def countFlips (initial final : GnomeState) : ℕ := sorry

/-- Checks if a given state allows all gnomes to make correct statements -/
def isValidState (state : GnomeState) : Prop := sorry

/-- The main theorem stating the minimum number of flips required -/
theorem min_flips_theorem (initial : GnomeState) :
  ∃ (final : GnomeState),
    isValidState final ∧
    countFlips initial final = 998 ∧
    ∀ (other : GnomeState),
      isValidState other →
      countFlips initial other ≥ 998 := by
  sorry

end NUMINAMATH_CALUDE_min_flips_theorem_l2230_223085


namespace NUMINAMATH_CALUDE_smallest_bob_number_l2230_223003

def alice_number : ℕ := 45

def is_valid_bob_number (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (p ∣ alice_number → p^2 ∣ n) ∧ (p ∣ n → p ∣ alice_number)

theorem smallest_bob_number :
  ∃ (bob_number : ℕ), is_valid_bob_number bob_number ∧
    ∀ (m : ℕ), is_valid_bob_number m → bob_number ≤ m ∧ bob_number = 2025 :=
sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l2230_223003


namespace NUMINAMATH_CALUDE_excircle_incircle_relation_l2230_223064

/-- Given a triangle ABC with inscribed circle radius r, excircle radii r_a, r_b, r_c,
    and semiperimeter p, prove that (r_a * r_b * r_c) / r = p^2 -/
theorem excircle_incircle_relation (r r_a r_b r_c p : ℝ) : r > 0 → r_a > 0 → r_b > 0 → r_c > 0 → p > 0 →
  (r_a * r_b * r_c) / r = p^2 := by sorry

end NUMINAMATH_CALUDE_excircle_incircle_relation_l2230_223064


namespace NUMINAMATH_CALUDE_alice_bake_time_proof_l2230_223087

/-- The time it takes Alice to bake a pie -/
def alice_bake_time : ℝ := 5

/-- The time it takes Bob to bake a pie -/
def bob_bake_time : ℝ := 6

/-- The total time given in the problem -/
def total_time : ℝ := 60

/-- The number of additional pies Alice can bake compared to Bob in the given time -/
def additional_pies : ℕ := 2

theorem alice_bake_time_proof :
  alice_bake_time = 5 ∧
  (total_time / bob_bake_time + additional_pies) * alice_bake_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_alice_bake_time_proof_l2230_223087


namespace NUMINAMATH_CALUDE_expression_has_four_terms_l2230_223089

/-- The expression with the asterisk replaced by a monomial -/
def expression (x : ℝ) : ℝ := (x^4 - 3)^2 + (x^3 + 3*x)^2

/-- The result after expanding and combining like terms -/
def expanded_result (x : ℝ) : ℝ := x^8 + x^6 + 9*x^2 + 9

/-- Theorem stating that the expanded result has exactly four terms -/
theorem expression_has_four_terms :
  ∃ (a b c d : ℝ → ℝ),
    (∀ x, expanded_result x = a x + b x + c x + d x) ∧
    (∀ x, a x ≠ 0 ∧ b x ≠ 0 ∧ c x ≠ 0 ∧ d x ≠ 0) ∧
    (∀ x, expression x = expanded_result x) :=
sorry

end NUMINAMATH_CALUDE_expression_has_four_terms_l2230_223089


namespace NUMINAMATH_CALUDE_school_population_l2230_223042

theorem school_population (b g t : ℕ) : 
  b = 4 * g → 
  g = 10 * t → 
  b + g + t = (51 * b) / 40 := by
sorry

end NUMINAMATH_CALUDE_school_population_l2230_223042


namespace NUMINAMATH_CALUDE_smallest_n_for_real_power_l2230_223077

def complex_i : ℂ := Complex.I

def is_real (z : ℂ) : Prop := z.im = 0

theorem smallest_n_for_real_power :
  ∃ (n : ℕ), n > 0 ∧ is_real ((1 + complex_i) ^ n) ∧
  ∀ (m : ℕ), 0 < m → m < n → ¬ is_real ((1 + complex_i) ^ m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_real_power_l2230_223077


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l2230_223041

/-- An arithmetic sequence with first term 3 and common difference 5 -/
def arithmeticSequence (n : ℕ) : ℕ := 3 + (n - 1) * 5

/-- The 150th term of the arithmetic sequence is 748 -/
theorem arithmetic_sequence_150th_term : arithmeticSequence 150 = 748 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l2230_223041


namespace NUMINAMATH_CALUDE_minimum_employment_age_is_25_l2230_223029

/-- The minimum age required to be employed at the company -/
def minimum_employment_age : ℕ := 25

/-- Jane's current age -/
def jane_current_age : ℕ := 28

/-- Years until Dara reaches minimum employment age -/
def years_until_dara_reaches_minimum_age : ℕ := 14

/-- Years until Dara is half Jane's age -/
def years_until_dara_half_jane_age : ℕ := 6

theorem minimum_employment_age_is_25 :
  minimum_employment_age = 25 :=
by sorry

end NUMINAMATH_CALUDE_minimum_employment_age_is_25_l2230_223029


namespace NUMINAMATH_CALUDE_spatial_relationships_l2230_223035

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularPL : Line → Plane → Prop)
variable (parallelPL : Line → Plane → Prop)
variable (perpendicularP : Plane → Plane → Prop)
variable (parallelP : Plane → Plane → Prop)

-- Define the theorem
theorem spatial_relationships 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (∀ (m n : Line) (α β : Plane), 
    perpendicularPL m α → perpendicularPL n β → perpendicular m n → perpendicularP α β) ∧
  (∀ (m n : Line) (α β : Plane), 
    perpendicularPL m α → parallelPL n β → parallelP α β → perpendicular m n) := by
  sorry

end NUMINAMATH_CALUDE_spatial_relationships_l2230_223035


namespace NUMINAMATH_CALUDE_last_four_digits_5_pow_2011_l2230_223073

def last_four_digits (n : ℕ) : ℕ := n % 10000

theorem last_four_digits_5_pow_2011 :
  last_four_digits (5^2011) = 8125 :=
by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_5_pow_2011_l2230_223073


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l2230_223043

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + Nat.factorial 6 = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l2230_223043


namespace NUMINAMATH_CALUDE_tan_theta_half_l2230_223033

theorem tan_theta_half (θ : Real) (h : (1 + Real.cos (2 * θ)) / Real.sin (2 * θ) = 2) : 
  Real.tan θ = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_half_l2230_223033


namespace NUMINAMATH_CALUDE_rational_power_equality_l2230_223007

theorem rational_power_equality (x y : ℚ) (n : ℕ) (h_odd : Odd n) (h_pos : 0 < n)
  (h_eq : x^n - 2*x = y^n - 2*y) : x = y := by
  sorry

end NUMINAMATH_CALUDE_rational_power_equality_l2230_223007


namespace NUMINAMATH_CALUDE_max_y_coordinate_value_l2230_223054

noncomputable def max_y_coordinate (θ : ℝ) : ℝ :=
  let r := Real.sin (3 * θ)
  r * Real.sin θ

theorem max_y_coordinate_value :
  ∃ (θ : ℝ), ∀ (φ : ℝ), max_y_coordinate θ ≥ max_y_coordinate φ ∧
  max_y_coordinate θ = 3 * (3 / 16) ^ (1 / 3) - 4 * 3 ^ (4 / 3) / 16 ^ (4 / 3) :=
sorry

end NUMINAMATH_CALUDE_max_y_coordinate_value_l2230_223054


namespace NUMINAMATH_CALUDE_simplify_expression_l2230_223050

theorem simplify_expression (x : ℝ) : (3 * x + 20) + (50 * x + 25) = 53 * x + 45 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2230_223050


namespace NUMINAMATH_CALUDE_train_crossing_time_l2230_223060

/-- Proves that a train with given length and speed takes the calculated time to cross a post -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 120 → 
  train_speed_kmh = 72 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2230_223060


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2230_223023

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) :
  ∃ k : ℤ, (n - 1) + n + (n + 1) + (n + 2) = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2230_223023


namespace NUMINAMATH_CALUDE_tangent_line_max_a_l2230_223032

/-- Given a real number a, if there exists a common tangent line to the curves y = x^2 and y = a ln x for x > 0, then a ≤ 2e -/
theorem tangent_line_max_a (a : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ 
    (∃ (m : ℝ), (2 * x = a / x) ∧ 
      (x^2 = a * Real.log x + m))) → 
  a ≤ 2 * Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_max_a_l2230_223032


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_simplification_l2230_223021

/-- Given distinct real numbers a, b, c, and d, the sum of four rational expressions
    simplifies to a linear polynomial. -/
theorem sum_of_fourth_powers_simplification 
  (a b c d : ℝ) (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) 
  (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  let f : ℝ → ℝ := λ x => 
    ((x + a)^4) / ((a - b)*(a - c)*(a - d)) + 
    ((x + b)^4) / ((b - a)*(b - c)*(b - d)) + 
    ((x + c)^4) / ((c - a)*(c - b)*(c - d)) + 
    ((x + d)^4) / ((d - a)*(d - b)*(d - c))
  ∀ x, f x = a + b + c + d + 4*x := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_simplification_l2230_223021


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2230_223036

-- Define the quadratic function
def f (x : ℝ) : ℝ := x * (1 - 3 * x)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x > 0} = Set.Ioo 0 (1/3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2230_223036


namespace NUMINAMATH_CALUDE_derivative_of_y_l2230_223027

open Real

noncomputable def y (x : ℝ) : ℝ := cos (2*x - 1) + 1 / (x^2)

theorem derivative_of_y :
  deriv y = λ x => -2 * sin (2*x - 1) - 2 / (x^3) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l2230_223027


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2230_223082

theorem polynomial_simplification (s : ℝ) : 
  (2 * s^2 + 5 * s - 3) - (s^2 + 9 * s - 6) = s^2 - 4 * s + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2230_223082


namespace NUMINAMATH_CALUDE_expansion_has_four_nonzero_terms_l2230_223065

/-- The polynomial expression to be expanded -/
def polynomial_expression (x : ℝ) : ℝ :=
  (2*x + 5) * (3*x^2 + x + 6) - 4*(x^3 + 3*x^2 - 4*x + 1)

/-- The expanded form of the polynomial expression -/
def expanded_polynomial (x : ℝ) : ℝ :=
  2*x^3 + 5*x^2 + 33*x + 26

/-- Theorem stating that the expansion has exactly 4 nonzero terms -/
theorem expansion_has_four_nonzero_terms :
  ∃ (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0),
  ∀ x, polynomial_expression x = a*x^3 + b*x^2 + c*x + d :=
sorry

end NUMINAMATH_CALUDE_expansion_has_four_nonzero_terms_l2230_223065


namespace NUMINAMATH_CALUDE_prize_distribution_and_cost_l2230_223075

/-- Represents the prize distribution and cost calculation for a school event --/
theorem prize_distribution_and_cost 
  (x : ℕ) -- number of first prize items
  (h1 : x + (3*x - 2) + (52 - 4*x) = 50) -- total prizes constraint
  (h2 : x > 0) -- ensure positive number of first prize items
  (h3 : 3*x - 2 ≥ 0) -- ensure non-negative number of second prize items
  : 
  (20*x + 14*(3*x - 2) + 8*(52 - 4*x) = 30*x + 388) ∧ 
  (3*x - 2 = 22 → 20*x + 14*(3*x - 2) + 8*(52 - 4*x) = 628)
  := by sorry


end NUMINAMATH_CALUDE_prize_distribution_and_cost_l2230_223075


namespace NUMINAMATH_CALUDE_cage_cost_calculation_l2230_223002

/-- The cost of a cage, given the amount paid and change received. -/
def cage_cost (paid : ℚ) (change : ℚ) : ℚ := paid - change

/-- Theorem stating that the cage costs $19.74 given the conditions -/
theorem cage_cost_calculation (paid : ℚ) (change : ℚ) 
  (h_paid : paid = 20) 
  (h_change : change = 0.26) : 
  cage_cost paid change = 19.74 := by
  sorry

end NUMINAMATH_CALUDE_cage_cost_calculation_l2230_223002


namespace NUMINAMATH_CALUDE_library_to_post_office_l2230_223012

def total_distance : ℝ := 0.8
def house_to_library : ℝ := 0.3
def post_office_to_house : ℝ := 0.4

theorem library_to_post_office :
  total_distance - house_to_library - post_office_to_house = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_library_to_post_office_l2230_223012


namespace NUMINAMATH_CALUDE_equation_solution_l2230_223086

theorem equation_solution : ∃! x : ℝ, (x - 6) / (x + 4) = (x + 3) / (x - 5) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2230_223086


namespace NUMINAMATH_CALUDE_pet_shop_dogs_l2230_223084

theorem pet_shop_dogs (total_dogs_bunnies : ℕ) (ratio_dogs : ℕ) (ratio_cats : ℕ) (ratio_bunnies : ℕ) 
  (h1 : total_dogs_bunnies = 330)
  (h2 : ratio_dogs = 7)
  (h3 : ratio_cats = 7)
  (h4 : ratio_bunnies = 8) :
  (ratio_dogs * total_dogs_bunnies) / (ratio_dogs + ratio_bunnies) = 154 := by
  sorry

#check pet_shop_dogs

end NUMINAMATH_CALUDE_pet_shop_dogs_l2230_223084


namespace NUMINAMATH_CALUDE_tangent_and_unique_zero_l2230_223078

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (a * Real.log x) / x

def g (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := f a (f a x) - t

theorem tangent_and_unique_zero (a : ℝ) (h1 : a > 0) :
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ = f a x₀ ∧ x₀ - 2 * y₀ = 0 ∧ 
    (∀ x : ℝ, x > 0 → x - 2 * f a x ≥ 0) ∧
    (∀ x : ℝ, x > 0 → x - 2 * f a x = 0 → x = x₀)) →
  (∃! t : ℝ, ∃! x : ℝ, x > 0 ∧ g a t x = 0) →
  (∀ t : ℝ, (∃! x : ℝ, x > 0 ∧ g a t x = 0) → t = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_unique_zero_l2230_223078


namespace NUMINAMATH_CALUDE_max_min_values_on_interval_l2230_223014

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^4 + 4 * x^3 + 34

-- Define the interval
def interval : Set ℝ := { x | -2 ≤ x ∧ x ≤ 1 }

-- State the theorem
theorem max_min_values_on_interval :
  (∃ x ∈ interval, f x = 50 ∧ ∀ y ∈ interval, f y ≤ 50) ∧
  (∃ x ∈ interval, f x = 33 ∧ ∀ y ∈ interval, f y ≥ 33) := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_on_interval_l2230_223014


namespace NUMINAMATH_CALUDE_count_mgons_with_two_acute_angles_correct_l2230_223066

/-- Given integers m and n where 4 < m < n, and a regular (2n+1)-gon with vertices set P,
    this function computes the number of convex m-gons with vertices in P
    that have exactly two acute internal angles. -/
def count_mgons_with_two_acute_angles (m n : ℕ) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

/-- Theorem stating that the count_mgons_with_two_acute_angles function
    correctly computes the number of m-gons with two acute angles in a (2n+1)-gon. -/
theorem count_mgons_with_two_acute_angles_correct (m n : ℕ) 
    (h1 : 4 < m) (h2 : m < n) : 
  count_mgons_with_two_acute_angles m n = 
    (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1)) := by
  sorry

#check count_mgons_with_two_acute_angles_correct

end NUMINAMATH_CALUDE_count_mgons_with_two_acute_angles_correct_l2230_223066


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2230_223063

theorem cube_root_simplification : 
  (25^3 + 30^3 + 35^3 : ℝ)^(1/3) = 5 * 684^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2230_223063


namespace NUMINAMATH_CALUDE_f_at_negative_four_l2230_223031

/-- The polynomial f(x) = 12 + 35x − 8x^2 + 79x^3 + 6x^4 + 5x^5 + 3x^6 -/
def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

/-- Theorem: The value of f(-4) is 3392 -/
theorem f_at_negative_four : f (-4) = 3392 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_four_l2230_223031


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l2230_223081

theorem remainder_of_large_number (N : ℕ) (h : N = 109876543210) :
  N % 180 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l2230_223081


namespace NUMINAMATH_CALUDE_unit_digit_of_12_pow_100_l2230_223071

-- Define the function to get the unit digit of a natural number
def unitDigit (n : ℕ) : ℕ := n % 10

-- Define the theorem
theorem unit_digit_of_12_pow_100 : unitDigit (12^100) = 6 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_12_pow_100_l2230_223071


namespace NUMINAMATH_CALUDE_cu2co3_2_weight_calculation_l2230_223098

-- Define the chemical equation coefficients
def cu_no3_2_coeff : ℚ := 2
def na2co3_coeff : ℚ := 3
def cu2co3_2_coeff : ℚ := 1

-- Define the available moles of reactants
def cu_no3_2_moles : ℚ := 1.85
def na2co3_moles : ℚ := 3.21

-- Define the molar mass of Cu2(CO3)2
def cu2co3_2_molar_mass : ℚ := 247.12

-- Define the function to calculate the limiting reactant
def limiting_reactant (cu_no3_2 : ℚ) (na2co3 : ℚ) : ℚ :=
  min (cu_no3_2 / cu_no3_2_coeff) (na2co3 / na2co3_coeff)

-- Define the function to calculate the moles of Cu2(CO3)2 produced
def cu2co3_2_produced (limiting : ℚ) : ℚ :=
  limiting * (cu2co3_2_coeff / cu_no3_2_coeff)

-- Define the function to calculate the weight of Cu2(CO3)2 produced
def cu2co3_2_weight (moles : ℚ) : ℚ :=
  moles * cu2co3_2_molar_mass

-- Theorem statement
theorem cu2co3_2_weight_calculation :
  cu2co3_2_weight (cu2co3_2_produced (limiting_reactant cu_no3_2_moles na2co3_moles)) = 228.586 := by
  sorry

end NUMINAMATH_CALUDE_cu2co3_2_weight_calculation_l2230_223098


namespace NUMINAMATH_CALUDE_car_speed_ratio_l2230_223004

theorem car_speed_ratio :
  ∀ (v₁ v₂ : ℝ), v₁ > 0 → v₂ > 0 →
  (3 * v₂ / v₁ - 3 * v₁ / v₂ = 1.1) →
  v₂ / v₁ = 6 / 5 := by
sorry

end NUMINAMATH_CALUDE_car_speed_ratio_l2230_223004


namespace NUMINAMATH_CALUDE_modular_congruence_problem_l2230_223001

theorem modular_congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 50238 ≡ n [ZMOD 23] ∧ n = 19 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_problem_l2230_223001


namespace NUMINAMATH_CALUDE_store_comparison_and_best_plan_l2230_223074

/- Define the prices and quantities -/
def racket_price : ℝ := 50
def ball_price : ℝ := 20
def racket_quantity : ℕ := 10
def ball_quantity : ℕ := 40

/- Define the cost functions for each store -/
def cost_store_a (x : ℝ) : ℝ := 20 * x + 300
def cost_store_b (x : ℝ) : ℝ := 16 * x + 400

/- Define the most cost-effective plan -/
def cost_effective_plan : ℝ := racket_price * racket_quantity + ball_price * (ball_quantity - racket_quantity) * 0.8

/- Theorem statement -/
theorem store_comparison_and_best_plan :
  (cost_store_b ball_quantity < cost_store_a ball_quantity) ∧
  (cost_effective_plan = 980) := by
  sorry


end NUMINAMATH_CALUDE_store_comparison_and_best_plan_l2230_223074


namespace NUMINAMATH_CALUDE_power_set_of_A_l2230_223062

def A : Set ℕ := {1, 2}

def B : Set (Set ℕ) := {x | x ⊆ A}

theorem power_set_of_A : B = {∅, {1}, {2}, {1, 2}} := by
  sorry

end NUMINAMATH_CALUDE_power_set_of_A_l2230_223062


namespace NUMINAMATH_CALUDE_alternating_color_probability_value_l2230_223094

/-- Represents the number of balls of each color in the box -/
def num_balls : ℕ := 5

/-- Represents the total number of balls in the box -/
def total_balls : ℕ := 3 * num_balls

/-- Calculates the number of ways to arrange the balls -/
def total_arrangements : ℕ := Nat.choose total_balls num_balls * Nat.choose (2 * num_balls) num_balls

/-- Calculates the number of successful sequences (alternating colors) -/
def successful_sequences : ℕ := 2 * (3 ^ (num_balls - 1))

/-- The probability of drawing balls with alternating colors -/
def alternating_color_probability : ℚ := successful_sequences / total_arrangements

theorem alternating_color_probability_value : alternating_color_probability = 162 / 1001 := by
  sorry

end NUMINAMATH_CALUDE_alternating_color_probability_value_l2230_223094


namespace NUMINAMATH_CALUDE_sally_orange_balloons_l2230_223009

/-- The number of orange balloons Sally has after losing some -/
def remaining_orange_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that Sally has 7 orange balloons after losing 2 -/
theorem sally_orange_balloons :
  remaining_orange_balloons 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sally_orange_balloons_l2230_223009


namespace NUMINAMATH_CALUDE_sum_range_l2230_223056

theorem sum_range : 
  let sum := (25/8 : ℚ) + (31/7 : ℚ) + (128/21 : ℚ)
  (27/2 : ℚ) < sum ∧ sum < 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_range_l2230_223056


namespace NUMINAMATH_CALUDE_top_card_is_eleven_l2230_223088

/-- Represents a card in the array -/
structure Card where
  row : Fin 3
  col : Fin 6
  number : Fin 18

/-- Represents the initial 3x6 array of cards -/
def initial_array : Array (Array Card) := sorry

/-- Folds the left third over the middle third -/
def fold_left_third (arr : Array (Array Card)) : Array (Array Card) := sorry

/-- Folds the right third over the overlapped left and middle thirds -/
def fold_right_third (arr : Array (Array Card)) : Array (Array Card) := sorry

/-- Folds the bottom half over the top half -/
def fold_bottom_half (arr : Array (Array Card)) : Array (Array Card) := sorry

/-- Performs all folding operations -/
def perform_all_folds (arr : Array (Array Card)) : Array (Array Card) :=
  arr |> fold_left_third |> fold_right_third |> fold_bottom_half

/-- The top card after all folds -/
def top_card (arr : Array (Array Card)) : Card := sorry

theorem top_card_is_eleven :
  (top_card (perform_all_folds initial_array)).number = 11 := by
  sorry

end NUMINAMATH_CALUDE_top_card_is_eleven_l2230_223088


namespace NUMINAMATH_CALUDE_grocer_sales_problem_l2230_223080

theorem grocer_sales_problem (sales1 sales3 sales4 sales5 : ℕ) 
  (h1 : sales1 = 5420)
  (h3 : sales3 = 6200)
  (h4 : sales4 = 6350)
  (h5 : sales5 = 6500)
  (target_average : ℕ) 
  (h_target : target_average = 6000) :
  ∃ sales2 : ℕ, 
    sales2 = 5530 ∧ 
    (sales1 + sales2 + sales3 + sales4 + sales5) / 5 = target_average :=
by
  sorry

end NUMINAMATH_CALUDE_grocer_sales_problem_l2230_223080


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l2230_223034

theorem quadratic_function_inequality (a b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + a*x + b
  |f 1| + 2 * |f 2| + |f 3| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l2230_223034


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_sqrt_differences_l2230_223028

theorem max_value_of_sum_of_sqrt_differences (x y z : ℝ) 
  (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) (hz : z ∈ Set.Icc 0 1) :
  Real.sqrt (|x - y|) + Real.sqrt (|y - z|) + Real.sqrt (|z - x|) ≤ Real.sqrt 2 + 1 ∧
  ∃ x y z, x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ z ∈ Set.Icc 0 1 ∧
    Real.sqrt (|x - y|) + Real.sqrt (|y - z|) + Real.sqrt (|z - x|) = Real.sqrt 2 + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_sqrt_differences_l2230_223028


namespace NUMINAMATH_CALUDE_five_or_king_probability_l2230_223090

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the ranks in a standard deck -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents the suits in a standard deck -/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- A card in the deck -/
structure Card :=
  (rank : Rank)
  (suit : Suit)

/-- The probability of drawing a specific card from the deck -/
def draw_probability (d : Deck) (c : Card) : ℚ :=
  1 / 52

/-- The probability of drawing a card with a specific rank -/
def draw_rank_probability (d : Deck) (r : Rank) : ℚ :=
  4 / 52

/-- Theorem: The probability of drawing either a 5 or a King from a standard 52-card deck is 2/13 -/
theorem five_or_king_probability (d : Deck) : 
  draw_rank_probability d Rank.Five + draw_rank_probability d Rank.King = 2 / 13 := by
  sorry


end NUMINAMATH_CALUDE_five_or_king_probability_l2230_223090


namespace NUMINAMATH_CALUDE_shopkeeper_pricing_l2230_223013

theorem shopkeeper_pricing (CP : ℝ) 
  (h1 : 0.65 * CP = 416) : 1.25 * CP = 800 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_pricing_l2230_223013


namespace NUMINAMATH_CALUDE_classmates_lateness_l2230_223070

theorem classmates_lateness 
  (charlize_lateness : ℕ) 
  (total_lateness : ℕ) 
  (num_classmates : ℕ) 
  (h1 : charlize_lateness = 20)
  (h2 : total_lateness = 140)
  (h3 : num_classmates = 4) :
  (total_lateness - charlize_lateness) / num_classmates = 30 :=
by sorry

end NUMINAMATH_CALUDE_classmates_lateness_l2230_223070


namespace NUMINAMATH_CALUDE_product_evaluation_l2230_223048

theorem product_evaluation : (2.5 : ℝ) * (50.5 + 0.15) = 126.625 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2230_223048


namespace NUMINAMATH_CALUDE_inequalities_hold_l2230_223026

theorem inequalities_hold (a b c x y z : ℝ) 
  (hx : x ≤ a) (hy : y ≤ b) (hz : z ≤ c) : 
  (x * y + y * z + z * x ≤ a * b + b * c + c * a) ∧ 
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧ 
  (x * y * z ≤ a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l2230_223026


namespace NUMINAMATH_CALUDE_max_log_sum_l2230_223010

theorem max_log_sum (x y : ℝ) (h1 : x + 4 * y = 40) (h2 : x > 0) (h3 : y > 0) :
  ∃ (m : ℝ), ∀ (x' y' : ℝ), x' + 4 * y' = 40 → x' > 0 → y' > 0 → 
    Real.log x' + Real.log y' ≤ m ∧ 
    ∃ (x₀ y₀ : ℝ), x₀ + 4 * y₀ = 40 ∧ x₀ > 0 ∧ y₀ > 0 ∧ Real.log x₀ + Real.log y₀ = m ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_log_sum_l2230_223010


namespace NUMINAMATH_CALUDE_right_triangle_trig_identity_l2230_223069

/-- Given a right triangle PQR with hypotenuse PQ = 15 and PR = 9, prove that sin Q = 4/5 and the trigonometric identity sin² Q + cos² Q = 1 holds. -/
theorem right_triangle_trig_identity (PQ PR : ℝ) (hPQ : PQ = 15) (hPR : PR = 9) :
  let sinQ := Real.sqrt (PQ^2 - PR^2) / PQ
  let cosQ := PR / PQ
  sinQ = 4/5 ∧ sinQ^2 + cosQ^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_identity_l2230_223069


namespace NUMINAMATH_CALUDE_weight_loss_calculation_l2230_223016

/-- Represents the weight loss calculation problem --/
theorem weight_loss_calculation 
  (current_weight : ℕ) 
  (previous_weight : ℕ) 
  (h1 : current_weight = 27) 
  (h2 : previous_weight = 128) :
  previous_weight - current_weight = 101 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_calculation_l2230_223016


namespace NUMINAMATH_CALUDE_pencils_bought_l2230_223024

theorem pencils_bought (glue_cost pencil_cost total_paid change : ℕ) 
  (h1 : glue_cost = 270)
  (h2 : pencil_cost = 210)
  (h3 : total_paid = 1000)
  (h4 : change = 100) :
  ∃ (num_pencils : ℕ), 
    glue_cost + num_pencils * pencil_cost = total_paid - change ∧ 
    num_pencils = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_bought_l2230_223024


namespace NUMINAMATH_CALUDE_complex_number_location_l2230_223067

theorem complex_number_location (z : ℂ) : 
  z = (1/2 : ℝ) * Complex.abs z + Complex.I ^ 2015 → 
  0 < z.re ∧ z.im < 0 := by
sorry

end NUMINAMATH_CALUDE_complex_number_location_l2230_223067


namespace NUMINAMATH_CALUDE_irene_weekly_income_l2230_223044

/-- Calculates the total weekly income after taxes and deductions for an employee with given conditions --/
def total_weekly_income (base_salary : ℕ) (base_hours : ℕ) (overtime_rate1 : ℕ) (overtime_rate2 : ℕ) (overtime_rate3 : ℕ) (tax_rate : ℚ) (insurance_premium : ℕ) (hours_worked : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the total weekly income is $645 --/
theorem irene_weekly_income :
  let base_salary := 500
  let base_hours := 40
  let overtime_rate1 := 20
  let overtime_rate2 := 30
  let overtime_rate3 := 40
  let tax_rate := 15 / 100
  let insurance_premium := 50
  let hours_worked := 50
  total_weekly_income base_salary base_hours overtime_rate1 overtime_rate2 overtime_rate3 tax_rate insurance_premium hours_worked = 645 :=
by
  sorry

end NUMINAMATH_CALUDE_irene_weekly_income_l2230_223044


namespace NUMINAMATH_CALUDE_product_digits_sum_l2230_223079

def A : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def B : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

def C : ℕ := (A * B) / 10000 % 10
def D : ℕ := A * B % 10

theorem product_digits_sum : C + D = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_digits_sum_l2230_223079


namespace NUMINAMATH_CALUDE_monotone_cubic_implies_m_bound_l2230_223038

/-- A function f: ℝ → ℝ is monotonically increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The cubic function f(x) = x³ + 2x² + mx - 5 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x - 5

theorem monotone_cubic_implies_m_bound :
  ∀ m : ℝ, MonotonicallyIncreasing (f m) → m ≥ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_monotone_cubic_implies_m_bound_l2230_223038


namespace NUMINAMATH_CALUDE_roots_sum_powers_l2230_223008

theorem roots_sum_powers (a b : ℝ) : 
  a^2 - 4*a + 5 = 0 → b^2 - 4*b + 5 = 0 → a^3 + a^4*b^2 + a^2*b^4 + b^3 = 154 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l2230_223008
