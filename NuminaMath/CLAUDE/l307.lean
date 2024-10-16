import Mathlib

namespace NUMINAMATH_CALUDE_digit_cancellation_fractions_l307_30715

theorem digit_cancellation_fractions :
  let valid_fractions : List (ℕ × ℕ) := [(26, 65), (16, 64), (19, 95), (49, 98)]
  ∀ a b c : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 →
    (10 * a + b) * c = (10 * b + c) * a →
    a < c →
    (10 * a + b, 10 * b + c) ∈ valid_fractions :=
by sorry

end NUMINAMATH_CALUDE_digit_cancellation_fractions_l307_30715


namespace NUMINAMATH_CALUDE_unequal_outcome_probability_l307_30736

def num_grandchildren : ℕ := 10

theorem unequal_outcome_probability :
  let total_outcomes := 2^num_grandchildren
  let equal_outcomes := Nat.choose num_grandchildren (num_grandchildren / 2)
  (total_outcomes - equal_outcomes) / total_outcomes = 193 / 256 := by
  sorry

end NUMINAMATH_CALUDE_unequal_outcome_probability_l307_30736


namespace NUMINAMATH_CALUDE_infinite_solutions_c_equals_six_l307_30712

theorem infinite_solutions_c_equals_six :
  ∃! c : ℝ, ∀ y : ℝ, 2 * (4 + c * y) = 12 * y + 8 :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_c_equals_six_l307_30712


namespace NUMINAMATH_CALUDE_product_correction_l307_30786

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The problem statement -/
theorem product_correction (a b : ℕ) : 
  10 ≤ a ∧ a < 100 →  -- a is a two-digit number
  a > 0 →  -- a is positive
  b > 0 →  -- b is positive
  reverse_digits a * b = 284 →
  a * b = 68 := by
sorry

end NUMINAMATH_CALUDE_product_correction_l307_30786


namespace NUMINAMATH_CALUDE_average_of_x_and_y_l307_30763

theorem average_of_x_and_y (x y : ℝ) : 
  (2 + 6 + 10 + x + y) / 5 = 18 → (x + y) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_average_of_x_and_y_l307_30763


namespace NUMINAMATH_CALUDE_son_age_problem_l307_30728

theorem son_age_problem (father_age : ℕ) (son_age : ℕ) : 
  father_age = 40 ∧ 
  father_age = 4 * son_age ∧ 
  father_age + 20 = 2 * (son_age + 20) → 
  son_age = 10 :=
by sorry

end NUMINAMATH_CALUDE_son_age_problem_l307_30728


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l307_30730

theorem triangle_ratio_theorem (A B C : Real) (hTriangle : A + B + C = PI) 
  (hCondition : 3 * Real.sin B * Real.cos C = Real.sin C * (1 - 3 * Real.cos B)) : 
  Real.sin C / Real.sin A = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l307_30730


namespace NUMINAMATH_CALUDE_intersection_M_N_l307_30703

def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | 2 * x - 1 > 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l307_30703


namespace NUMINAMATH_CALUDE_mowing_time_calculation_l307_30776

/-- Calculates the time required to mow a rectangular lawn -/
def mowing_time (length width swath overlap speed : ℚ) : ℚ :=
  let effective_swath := (swath - overlap) / 12
  let strips := width / effective_swath
  let total_distance := strips * length
  total_distance / speed

/-- Theorem stating the time required to mow the lawn under given conditions -/
theorem mowing_time_calculation :
  mowing_time 100 180 (30/12) (6/12) 4000 = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_mowing_time_calculation_l307_30776


namespace NUMINAMATH_CALUDE_horse_speed_problem_l307_30767

/-- A problem from "Nine Chapters on the Mathematical Art" about horse speeds and travel times. -/
theorem horse_speed_problem (x : ℝ) (h_x : x > 3) : 
  let distance : ℝ := 900
  let slow_horse_time : ℝ := x + 1
  let fast_horse_time : ℝ := x - 3
  let slow_horse_speed : ℝ := distance / slow_horse_time
  let fast_horse_speed : ℝ := distance / fast_horse_time
  2 * slow_horse_speed = fast_horse_speed :=
by sorry

end NUMINAMATH_CALUDE_horse_speed_problem_l307_30767


namespace NUMINAMATH_CALUDE_S_min_at_24_l307_30748

/-- The sequence term a_n as a function of n -/
def a (n : ℕ) : ℤ := 2 * n - 49

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n * (2 * a 1 + (n - 1) * 2) / 2

/-- Theorem stating that S reaches its minimum value when n = 24 -/
theorem S_min_at_24 : ∀ k : ℕ, S 24 ≤ S k :=
sorry

end NUMINAMATH_CALUDE_S_min_at_24_l307_30748


namespace NUMINAMATH_CALUDE_min_positive_translation_for_symmetry_l307_30783

open Real

theorem min_positive_translation_for_symmetry (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = sin (2 * x + π / 4)) →
  (∀ x, f (x - φ) = f (-x)) →
  (φ > 0) →
  (∀ ψ, ψ > 0 → ψ < φ → ¬(∀ x, f (x - ψ) = f (-x))) →
  φ = 3 * π / 8 := by
sorry

end NUMINAMATH_CALUDE_min_positive_translation_for_symmetry_l307_30783


namespace NUMINAMATH_CALUDE_train_passing_time_l307_30706

/-- Calculates the time for two trains to clear each other --/
theorem train_passing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 160)
  (h2 : length2 = 280)
  (h3 : speed1 = 42)
  (h4 : speed2 = 30) : 
  (length1 + length2) / ((speed1 + speed2) * (1000 / 3600)) = 22 := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l307_30706


namespace NUMINAMATH_CALUDE_trailing_zeros_2006_factorial_trailing_zeros_2006_factorial_is_500_l307_30701

theorem trailing_zeros_2006_factorial : Nat → Nat
| n => (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

theorem trailing_zeros_2006_factorial_is_500 :
  trailing_zeros_2006_factorial 2006 = 500 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_2006_factorial_trailing_zeros_2006_factorial_is_500_l307_30701


namespace NUMINAMATH_CALUDE_team_handedness_ratio_l307_30787

/-- A ball team with right-handed and left-handed players -/
structure BallTeam where
  right_handed : ℕ
  left_handed : ℕ

/-- Represents the attendance at practice -/
structure PracticeAttendance (team : BallTeam) where
  present_right : ℕ
  present_left : ℕ
  absent_right : ℕ
  absent_left : ℕ
  total_present : present_right + present_left = team.right_handed + team.left_handed - (absent_right + absent_left)
  all_accounted : present_right + absent_right = team.right_handed
  all_accounted_left : present_left + absent_left = team.left_handed

/-- The theorem representing the problem -/
theorem team_handedness_ratio (team : BallTeam) (attendance : PracticeAttendance team) :
  (2 : ℚ) / 3 * (team.right_handed + team.left_handed) = attendance.absent_right + attendance.absent_left →
  (2 : ℚ) / 3 * (attendance.present_right + attendance.present_left) = attendance.present_left →
  (attendance.absent_right : ℚ) / attendance.absent_left = 14 / 10 →
  (team.right_handed : ℚ) / team.left_handed = 14 / 10 := by
  sorry

end NUMINAMATH_CALUDE_team_handedness_ratio_l307_30787


namespace NUMINAMATH_CALUDE_equation_with_integer_roots_l307_30719

theorem equation_with_integer_roots :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (x y : ℤ), x ≠ y ∧
  (1 : ℚ) / (x + a) + (1 : ℚ) / (x + b) = (1 : ℚ) / c ∧
  (1 : ℚ) / (y + a) + (1 : ℚ) / (y + b) = (1 : ℚ) / c :=
by sorry

end NUMINAMATH_CALUDE_equation_with_integer_roots_l307_30719


namespace NUMINAMATH_CALUDE_regular_polygon_160_degrees_has_18_sides_l307_30743

/-- A regular polygon with interior angles measuring 160° has 18 sides. -/
theorem regular_polygon_160_degrees_has_18_sides :
  ∀ n : ℕ,
  n ≥ 3 →
  (180 * (n - 2) : ℝ) / n = 160 →
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_160_degrees_has_18_sides_l307_30743


namespace NUMINAMATH_CALUDE_cassies_nail_trimming_l307_30729

/-- The number of nails/claws Cassie needs to cut -/
def total_nails_to_cut (num_dogs : ℕ) (num_parrots : ℕ) (nails_per_dog_foot : ℕ) 
  (feet_per_dog : ℕ) (claws_per_parrot_leg : ℕ) (legs_per_parrot : ℕ) 
  (extra_claw : ℕ) : ℕ :=
  (num_dogs * nails_per_dog_foot * feet_per_dog) + 
  (num_parrots * claws_per_parrot_leg * legs_per_parrot) + 
  extra_claw

/-- Theorem stating the total number of nails/claws Cassie needs to cut -/
theorem cassies_nail_trimming :
  total_nails_to_cut 4 8 4 4 3 2 1 = 113 := by
  sorry

end NUMINAMATH_CALUDE_cassies_nail_trimming_l307_30729


namespace NUMINAMATH_CALUDE_triangle_vector_property_l307_30740

theorem triangle_vector_property (A B C : ℝ) (hAcute : 0 < A ∧ A < π/2) 
    (hBcute : 0 < B ∧ B < π/2) (hCcute : 0 < C ∧ C < π/2) 
    (hSum : A + B + C = π) :
  let a : ℝ × ℝ := (Real.sin C + Real.cos C, 2 - 2 * Real.sin C)
  let b : ℝ × ℝ := (1 + Real.sin C, Real.sin C - Real.cos C)
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  2 * Real.sin A ^ 2 + Real.cos B = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_vector_property_l307_30740


namespace NUMINAMATH_CALUDE_consecutive_years_product_l307_30760

theorem consecutive_years_product : (2014 - 2013) * (2013 - 2012) = 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_years_product_l307_30760


namespace NUMINAMATH_CALUDE_shadow_length_l307_30745

/-- Given two similar right triangles, if one has height 2 and base 4,
    and the other has height 2.5, then the base of the second triangle is 5. -/
theorem shadow_length (h1 h2 b1 b2 : ℝ) : 
  h1 = 2 → h2 = 2.5 → b1 = 4 → h1 / b1 = h2 / b2 → b2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_shadow_length_l307_30745


namespace NUMINAMATH_CALUDE_cosine_amplitude_l307_30731

/-- Given a cosine function y = a cos(bx + c) + d where a, b, c, d are positive constants,
    if the graph oscillates between 5 and 1, then a = 2. -/
theorem cosine_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_osc : ∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) :
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l307_30731


namespace NUMINAMATH_CALUDE_four_point_ratio_l307_30790

/-- Given four distinct points on a plane with segment lengths a, a, a, 2a, 2a, and b,
    prove that the ratio of b to a is 2√2 -/
theorem four_point_ratio (a b : ℝ) (h : a > 0) :
  ∃ (A B C D : ℝ × ℝ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    ({dist A B, dist A C, dist A D, dist B C, dist B D, dist C D} : Finset ℝ) =
      {a, a, a, 2*a, 2*a, b} →
    b / a = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_four_point_ratio_l307_30790


namespace NUMINAMATH_CALUDE_additional_interest_percentage_l307_30754

theorem additional_interest_percentage
  (initial_deposit : ℝ)
  (amount_after_3_years : ℝ)
  (target_amount : ℝ)
  (time_period : ℝ)
  (h1 : initial_deposit = 8000)
  (h2 : amount_after_3_years = 11200)
  (h3 : target_amount = 11680)
  (h4 : time_period = 3) :
  let original_interest := amount_after_3_years - initial_deposit
  let target_interest := target_amount - initial_deposit
  let additional_interest := target_interest - original_interest
  let additional_rate := (additional_interest * 100) / (initial_deposit * time_period)
  additional_rate = 2 := by
sorry

end NUMINAMATH_CALUDE_additional_interest_percentage_l307_30754


namespace NUMINAMATH_CALUDE_expense_increase_percentage_l307_30797

theorem expense_increase_percentage (salary : ℝ) (initial_savings_rate : ℝ) (new_savings : ℝ) :
  salary = 5500 →
  initial_savings_rate = 0.2 →
  new_savings = 220 →
  let initial_savings := salary * initial_savings_rate
  let initial_expenses := salary - initial_savings
  let expense_increase := initial_savings - new_savings
  (expense_increase / initial_expenses) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expense_increase_percentage_l307_30797


namespace NUMINAMATH_CALUDE_inequality_proof_l307_30738

theorem inequality_proof (n : ℕ) (hn : n > 0) :
  (2 * n^2 + 3 * n + 1)^n ≥ 6^n * (n!)^2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l307_30738


namespace NUMINAMATH_CALUDE_symmetric_points_power_l307_30710

/-- 
Given two points M and N in the Cartesian coordinate system,
prove that if they are symmetric with respect to the y-axis,
then b^a = 16.
-/
theorem symmetric_points_power (a b : ℝ) : 
  (2 * a = 8) → (2 = a + b) → ((-2 : ℝ) ^ (4 : ℕ) = 16) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_power_l307_30710


namespace NUMINAMATH_CALUDE_fermat_numbers_coprime_l307_30711

theorem fermat_numbers_coprime (m n : ℕ) (h : m ≠ n) : 
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := by sorry

end NUMINAMATH_CALUDE_fermat_numbers_coprime_l307_30711


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l307_30792

/-- A positive integer n is a perfect square if there exists an integer k such that n = k^2 -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- A positive integer n is a perfect fourth power if there exists an integer k such that n = k^4 -/
def IsPerfectFourthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^4

/-- The main theorem stating that 54 is the smallest positive integer satisfying the conditions -/
theorem smallest_n_satisfying_conditions : 
  (∀ m : ℕ, m > 0 ∧ m < 54 → ¬(IsPerfectSquare (2 * m) ∧ IsPerfectFourthPower (3 * m))) ∧ 
  (IsPerfectSquare (2 * 54) ∧ IsPerfectFourthPower (3 * 54)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l307_30792


namespace NUMINAMATH_CALUDE_purse_cost_multiple_l307_30749

theorem purse_cost_multiple (wallet_cost purse_cost : ℚ) : 
  wallet_cost = 22 →
  wallet_cost + purse_cost = 107 →
  ∃ n : ℕ, n ≤ 4 ∧ purse_cost < n * wallet_cost :=
by sorry

end NUMINAMATH_CALUDE_purse_cost_multiple_l307_30749


namespace NUMINAMATH_CALUDE_a_6_value_l307_30744

/-- An arithmetic sequence where a_2 and a_10 are roots of 2x^2 - x - 7 = 0 -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  2 * (a 2)^2 - (a 2) - 7 = 0 ∧
  2 * (a 10)^2 - (a 10) - 7 = 0

theorem a_6_value (a : ℕ → ℚ) (h : ArithmeticSequence a) : a 6 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_a_6_value_l307_30744


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l307_30768

/-- The equation of a line perpendicular to x - y = 0 and passing through (1, 0) -/
theorem perpendicular_line_equation :
  let l₁ : Set (ℝ × ℝ) := {p | p.1 - p.2 = 0}  -- The line x - y = 0
  let p : ℝ × ℝ := (1, 0)  -- The point (1, 0)
  let l₂ : Set (ℝ × ℝ) := {q | q.1 + q.2 - 1 = 0}  -- The line we want to prove
  (∀ x y, (x, y) ∈ l₂ ↔ x + y - 1 = 0) ∧  -- l₂ is indeed x + y - 1 = 0
  p ∈ l₂ ∧  -- l₂ passes through (1, 0)
  (∀ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ l₁ ∧ (x₂, y₂) ∈ l₁ ∧ x₁ ≠ x₂ →
    (x₁ - x₂) * ((1 - x) / (0 - y)) = -1)  -- l₂ is perpendicular to l₁
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l307_30768


namespace NUMINAMATH_CALUDE_roof_ratio_l307_30705

theorem roof_ratio (length width : ℝ) : 
  length * width = 784 →
  length - width = 42 →
  length / width = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_roof_ratio_l307_30705


namespace NUMINAMATH_CALUDE_number_is_composite_l307_30714

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the number formed by the given sequence of digits -/
def formNumber (digits : List Digit) : ℕ :=
  sorry

/-- Checks if a natural number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- The main theorem to be proved -/
theorem number_is_composite (digits : List Digit) :
  isComposite (formNumber digits) :=
sorry

end NUMINAMATH_CALUDE_number_is_composite_l307_30714


namespace NUMINAMATH_CALUDE_expression_evaluation_l307_30734

theorem expression_evaluation : 2 + 3 * 4 - 5 + 6 / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l307_30734


namespace NUMINAMATH_CALUDE_ticket_cost_calculation_l307_30772

/-- The total cost of tickets for various events -/
def total_cost (movie_price : ℚ) (football_price : ℚ) (concert_price : ℚ) (theater_price : ℚ) : ℚ :=
  8 * movie_price + 5 * football_price + 3 * concert_price + 4 * theater_price

/-- The theorem stating the total cost of tickets -/
theorem ticket_cost_calculation : ∃ (movie_price football_price concert_price theater_price : ℚ),
  (8 * movie_price = 2 * football_price) ∧
  (movie_price = 30) ∧
  (concert_price = football_price - 10) ∧
  (theater_price = 40 * (1 - 0.1)) ∧
  (total_cost movie_price football_price concert_price theater_price = 1314) :=
by
  sorry


end NUMINAMATH_CALUDE_ticket_cost_calculation_l307_30772


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l307_30722

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

/-- The y-intercept of the parabola -/
def d : ℝ := parabola 0

/-- The x-intercepts of the parabola -/
noncomputable def e : ℝ := (9 + Real.sqrt 33) / 6
noncomputable def f : ℝ := (9 - Real.sqrt 33) / 6

/-- Theorem stating that the sum of the y-intercept and x-intercepts is 7 -/
theorem parabola_intercepts_sum : d + e + f = 7 := by sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l307_30722


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l307_30725

theorem sum_of_quadratic_solutions : 
  let f (x : ℝ) := x^2 - 6*x - 22 - (2*x + 18)
  let roots := {x : ℝ | f x = 0}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ roots ∧ x₂ ∈ roots ∧ x₁ + x₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l307_30725


namespace NUMINAMATH_CALUDE_tire_usage_proof_l307_30742

/-- Represents the number of miles each tire is used when a car with 6 tires
    travels 40,000 miles, with each tire being used equally. -/
def miles_per_tire : ℕ := 26667

/-- The total number of tires. -/
def total_tires : ℕ := 6

/-- The number of tires used on the road at any given time. -/
def road_tires : ℕ := 4

/-- The total distance traveled by the car in miles. -/
def total_distance : ℕ := 40000

theorem tire_usage_proof :
  miles_per_tire * total_tires = total_distance * road_tires :=
sorry

end NUMINAMATH_CALUDE_tire_usage_proof_l307_30742


namespace NUMINAMATH_CALUDE_line_intersects_at_least_one_l307_30766

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of being skew lines
variable (skew : Line → Line → Prop)

-- Define the property of intersection
variable (intersects : Line → Line → Prop)

-- Define the property of a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (plane_intersection : Plane → Plane → Line)

-- State the theorem
theorem line_intersects_at_least_one 
  (m n l : Line) (α β : Plane) :
  skew m n →
  ¬(intersects l m) →
  ¬(intersects l n) →
  in_plane n β →
  plane_intersection α β = l →
  (intersects l m) ∨ (intersects l n) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_at_least_one_l307_30766


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l307_30798

-- Define the sets M and N
def M : Set ℕ := {a : ℕ | a = 0 ∨ ∃ x, x = a}
def N : Set ℕ := {1, 2}

-- State the theorem
theorem union_of_M_and_N :
  (∃ a : ℕ, M = {a, 0}) →  -- M = {a, 0}
  N = {1, 2} →             -- N = {1, 2}
  M ∩ N = {1} →            -- M ∩ N = {1}
  M ∪ N = {0, 1, 2} :=     -- M ∪ N = {0, 1, 2}
by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l307_30798


namespace NUMINAMATH_CALUDE_no_pentagon_cross_section_l307_30718

/-- A cube in 3D space -/
structure Cube

/-- A plane in 3D space -/
structure Plane

/-- Possible shapes that can result from the intersection of a plane and a cube -/
inductive CrossSection
  | EquilateralTriangle
  | Square
  | RegularPentagon
  | RegularHexagon

/-- The intersection of a plane and a cube -/
def intersect (c : Cube) (p : Plane) : CrossSection := sorry

/-- Theorem stating that a regular pentagon cannot be a cross-section of a cube -/
theorem no_pentagon_cross_section (c : Cube) (p : Plane) :
  intersect c p ≠ CrossSection.RegularPentagon := by sorry

end NUMINAMATH_CALUDE_no_pentagon_cross_section_l307_30718


namespace NUMINAMATH_CALUDE_julia_jonny_stairs_fraction_l307_30726

theorem julia_jonny_stairs_fraction (jonny_stairs : ℕ) (total_stairs : ℕ) 
  (h1 : jonny_stairs = 1269)
  (h2 : total_stairs = 1685) :
  (total_stairs - jonny_stairs : ℚ) / jonny_stairs = 416 / 1269 := by
  sorry

end NUMINAMATH_CALUDE_julia_jonny_stairs_fraction_l307_30726


namespace NUMINAMATH_CALUDE_no_solution_exists_l307_30795

theorem no_solution_exists : ¬∃ (x y : ℝ), 9^(y+1) / (1 + 4 / x^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l307_30795


namespace NUMINAMATH_CALUDE_tangent_line_slope_l307_30752

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem statement
theorem tangent_line_slope (a : ℝ) :
  (∃ x₀ : ℝ, f x₀ = a * x₀ + 16 ∧ 
             ∀ x : ℝ, x ≠ x₀ → f x ≠ a * x + 16) →
  a = 9 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l307_30752


namespace NUMINAMATH_CALUDE_ellipse_left_vertex_l307_30779

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The theorem stating the properties of the ellipse and its left vertex -/
theorem ellipse_left_vertex 
  (E : Ellipse) 
  (C : Circle) 
  (h_focus : C.h = 3 ∧ C.k = 0) -- One focus is the center of the circle
  (h_circle_eq : ∀ x y, x^2 + y^2 - 6*x + 8 = 0 ↔ (x - C.h)^2 + (y - C.k)^2 = C.r^2) -- Circle equation
  (h_minor_axis : E.b = 4) -- Minor axis is 8 in length
  : ∃ x, x = -5 ∧ (x / E.a)^2 + 0^2 / E.b^2 = 1 -- Left vertex is at (-5, 0)
:= by sorry

end NUMINAMATH_CALUDE_ellipse_left_vertex_l307_30779


namespace NUMINAMATH_CALUDE_orange_slices_problem_l307_30713

/-- The number of additional slices needed to fill the last container -/
def additional_slices_needed (total_slices : ℕ) (container_capacity : ℕ) : ℕ :=
  container_capacity - (total_slices % container_capacity)

/-- Theorem stating that given 329 slices and a container capacity of 4,
    3 additional slices are needed to fill the last container -/
theorem orange_slices_problem :
  additional_slices_needed 329 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_slices_problem_l307_30713


namespace NUMINAMATH_CALUDE_proportion_problem_l307_30796

theorem proportion_problem (x y z v : ℤ) : 
  (x * v = y * z) →
  (x + v = y + z + 7) →
  (x^2 + v^2 = y^2 + z^2 + 21) →
  (x^4 + v^4 = y^4 + z^4 + 2625) →
  ((x = -3 ∧ v = 8 ∧ y = -6 ∧ z = 4) ∨ 
   (x = 8 ∧ v = -3 ∧ y = 4 ∧ z = -6)) := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l307_30796


namespace NUMINAMATH_CALUDE_flower_producing_plants_l307_30700

theorem flower_producing_plants 
  (daisy_seeds sunflower_seeds : ℕ)
  (daisy_germination_rate sunflower_germination_rate flower_production_rate : ℚ)
  (h1 : daisy_seeds = 25)
  (h2 : sunflower_seeds = 25)
  (h3 : daisy_germination_rate = 3/5)
  (h4 : sunflower_germination_rate = 4/5)
  (h5 : flower_production_rate = 4/5) :
  ⌊(daisy_germination_rate * daisy_seeds + sunflower_germination_rate * sunflower_seeds) * flower_production_rate⌋ = 28 :=
by sorry

end NUMINAMATH_CALUDE_flower_producing_plants_l307_30700


namespace NUMINAMATH_CALUDE_special_triangle_vertices_l307_30799

/-- Definition of a triangle with specific properties -/
structure SpecialTriangle where
  -- Vertices
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Centroid
  S : ℝ × ℝ
  -- Orthocenter
  M : ℝ × ℝ
  -- Properties
  centroid_prop : S = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  orthocenter_prop : (M.1 - A.1) * (C.1 - B.1) + (M.2 - A.2) * (C.2 - B.2) = 0

/-- Theorem about the specific triangle -/
theorem special_triangle_vertices :
  ∃ (t : SpecialTriangle),
    t.A = (7, 3) ∧
    t.S = (5, -5/3) ∧
    t.M = (3, -1) ∧
    (t.B = (1, -1) ∧ t.C = (7, -7) ∨ t.B = (7, -7) ∧ t.C = (1, -1)) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_vertices_l307_30799


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l307_30735

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l307_30735


namespace NUMINAMATH_CALUDE_potatoes_cooked_l307_30727

theorem potatoes_cooked (total : ℕ) (cooking_time : ℕ) (remaining_time : ℕ) : 
  total = 15 → 
  cooking_time = 8 → 
  remaining_time = 72 → 
  total - (remaining_time / cooking_time) = 6 := by
sorry

end NUMINAMATH_CALUDE_potatoes_cooked_l307_30727


namespace NUMINAMATH_CALUDE_value_of_expression_l307_30737

theorem value_of_expression (x : ℝ) (h : x = -2) : (3 * x + 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l307_30737


namespace NUMINAMATH_CALUDE_inequality_proof_l307_30758

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l307_30758


namespace NUMINAMATH_CALUDE_alice_pens_count_l307_30723

/-- Proves that Alice has 60 pens given the conditions of the problem -/
theorem alice_pens_count :
  ∀ (alice_pens clara_pens alice_age clara_age : ℕ),
    clara_pens = (2 * alice_pens) / 5 →
    alice_pens - clara_pens = clara_age - alice_age →
    alice_age = 20 →
    clara_age > alice_age →
    clara_age + 5 = 61 →
    alice_pens = 60 := by
  sorry

end NUMINAMATH_CALUDE_alice_pens_count_l307_30723


namespace NUMINAMATH_CALUDE_log_simplification_l307_30707

theorem log_simplification (a b c d x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hx : 0 < x) (hy : 0 < y) :
  Real.log (a^2 / b) + Real.log (b / c) + Real.log (c / d^2) - Real.log ((a^2 * y) / (d^2 * x)) = Real.log (x / y) := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l307_30707


namespace NUMINAMATH_CALUDE_polynomial_factorization_l307_30756

theorem polynomial_factorization (x : ℤ) : 
  x^15 + x^8 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^8 - x^7 + x^6 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l307_30756


namespace NUMINAMATH_CALUDE_slope_range_for_intersection_l307_30770

/-- The set of possible slopes for a line with y-intercept (0,3) that intersects the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (1/20) ∨ m ≥ Real.sqrt (1/20)}

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 + 25 * y^2 = 100

/-- The line equation with slope m and y-intercept 3 -/
def line_equation (m x : ℝ) : ℝ :=
  m * x + 3

theorem slope_range_for_intersection :
  ∀ m : ℝ, (∃ x : ℝ, ellipse_equation x (line_equation m x)) ↔ m ∈ possible_slopes :=
by sorry

end NUMINAMATH_CALUDE_slope_range_for_intersection_l307_30770


namespace NUMINAMATH_CALUDE_student_calculation_error_l307_30708

theorem student_calculation_error (x y : ℝ) : 
  (5/4 : ℝ) * x = (4/5 : ℝ) * x + 36 ∧ 
  (7/3 : ℝ) * y = (3/7 : ℝ) * y + 28 → 
  x = 80 ∧ y = 14.7 := by
sorry

end NUMINAMATH_CALUDE_student_calculation_error_l307_30708


namespace NUMINAMATH_CALUDE_log_product_simplification_l307_30720

theorem log_product_simplification : 
  Real.log 9 / Real.log 8 * (Real.log 32 / Real.log 27) = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_log_product_simplification_l307_30720


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l307_30793

theorem largest_stamps_per_page (book1 book2 book3 : Nat) 
  (h1 : book1 = 1050) 
  (h2 : book2 = 1260) 
  (h3 : book3 = 1470) : 
  Nat.gcd book1 (Nat.gcd book2 book3) = 210 := by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l307_30793


namespace NUMINAMATH_CALUDE_count_is_six_l307_30784

/-- A type representing the blocks of digits --/
inductive DigitBlock
  | two
  | fortyfive
  | sixtyeight

/-- The set of all possible permutations of the digit blocks --/
def permutations : List (List DigitBlock) :=
  [DigitBlock.two, DigitBlock.fortyfive, DigitBlock.sixtyeight].permutations

/-- The count of all possible 5-digit numbers formed by the digits 2, 45, 68 --/
def count_five_digit_numbers : Nat := permutations.length

/-- Theorem stating that the count of possible 5-digit numbers is 6 --/
theorem count_is_six : count_five_digit_numbers = 6 := by sorry

end NUMINAMATH_CALUDE_count_is_six_l307_30784


namespace NUMINAMATH_CALUDE_marble_problem_l307_30769

theorem marble_problem (g j : ℕ) 
  (hg : g % 8 = 5) 
  (hj : j % 8 = 6) : 
  (g + 5 + j) % 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l307_30769


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l307_30709

/-- Represents the dimensions of a framed painting -/
structure FramedPainting where
  painting_width : ℝ
  painting_height : ℝ
  side_frame_width : ℝ

/-- Calculates the dimensions of the framed painting -/
def framed_dimensions (fp : FramedPainting) : (ℝ × ℝ) :=
  (fp.painting_width + 2 * fp.side_frame_width,
   fp.painting_height + 6 * fp.side_frame_width)

/-- Calculates the area of the framed painting -/
def framed_area (fp : FramedPainting) : ℝ :=
  let (w, h) := framed_dimensions fp
  w * h

/-- Calculates the area of the painting -/
def painting_area (fp : FramedPainting) : ℝ :=
  fp.painting_width * fp.painting_height

/-- Theorem stating the ratio of smaller to larger dimension of the framed painting -/
theorem framed_painting_ratio (fp : FramedPainting) 
  (h1 : fp.painting_width = 20)
  (h2 : fp.painting_height = 30)
  (h3 : framed_area fp = 3 * painting_area fp) :
  let (w, h) := framed_dimensions fp
  w / h = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l307_30709


namespace NUMINAMATH_CALUDE_kelly_carrot_harvest_l307_30751

/-- Calculates the total weight of carrots in pounds given the number of carrots
    harvested from three beds and the number of carrots per pound. -/
def total_carrot_weight (bed1 bed2 bed3 carrots_per_pound : ℕ) : ℚ :=
  (bed1 + bed2 + bed3 : ℚ) / carrots_per_pound

/-- Proves that the total weight of carrots is 39 pounds given the specific
    harvest numbers and weight ratio. -/
theorem kelly_carrot_harvest :
  total_carrot_weight 55 101 78 6 = 39 := by
  sorry

end NUMINAMATH_CALUDE_kelly_carrot_harvest_l307_30751


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l307_30717

theorem defective_shipped_percentage 
  (defective_rate : Real) 
  (shipped_rate : Real) 
  (h1 : defective_rate = 0.08) 
  (h2 : shipped_rate = 0.04) : 
  defective_rate * shipped_rate = 0.0032 := by
  sorry

#check defective_shipped_percentage

end NUMINAMATH_CALUDE_defective_shipped_percentage_l307_30717


namespace NUMINAMATH_CALUDE_range_of_f_on_interval_l307_30778

noncomputable def f (k : ℝ) (c : ℝ) (x : ℝ) : ℝ := x^k + c

theorem range_of_f_on_interval (k : ℝ) (c : ℝ) (h : k > 0) :
  Set.range (fun x => f k c x) ∩ Set.Ici 1 = Set.Ici (1 + c) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_l307_30778


namespace NUMINAMATH_CALUDE_initial_packages_l307_30773

theorem initial_packages (cupcakes_per_package : ℕ) (eaten_cupcakes : ℕ) (remaining_cupcakes : ℕ) :
  cupcakes_per_package = 4 →
  eaten_cupcakes = 5 →
  remaining_cupcakes = 7 →
  (eaten_cupcakes + remaining_cupcakes) / cupcakes_per_package = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_packages_l307_30773


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_value_l307_30762

theorem mean_equality_implies_y_value : ∃ y : ℝ,
  (4 + 7 + 11 + 14) / 4 = (10 + y + 5) / 3 ∧ y = 12 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_value_l307_30762


namespace NUMINAMATH_CALUDE_find_m_value_l307_30721

theorem find_m_value (n : ℕ) (m : ℕ) (h1 : n = 9998) (h2 : 72517 * (n + 1) = m) : 
  m = 725092483 := by
  sorry

end NUMINAMATH_CALUDE_find_m_value_l307_30721


namespace NUMINAMATH_CALUDE_emily_productivity_l307_30759

/-- Emily's work productivity over two days -/
theorem emily_productivity (p h : ℕ) : 
  p = 3 * h →                           -- Condition: p = 3h
  (p - 3) * (h + 3) - p * h = 6 * h - 9 -- Prove: difference in pages is 6h - 9
  := by sorry

end NUMINAMATH_CALUDE_emily_productivity_l307_30759


namespace NUMINAMATH_CALUDE_farmer_profit_is_960_l307_30753

/-- Represents the farmer's pig business -/
structure PigBusiness where
  num_piglets : ℕ
  sale_price : ℕ
  min_growth_months : ℕ
  feed_cost_per_month : ℕ
  pigs_sold_12_months : ℕ
  pigs_sold_16_months : ℕ

/-- Calculates the total profit for the pig business -/
def calculate_profit (business : PigBusiness) : ℕ :=
  let revenue := business.sale_price * (business.pigs_sold_12_months + business.pigs_sold_16_months)
  let feed_cost_12_months := business.feed_cost_per_month * business.min_growth_months * business.pigs_sold_12_months
  let feed_cost_16_months := business.feed_cost_per_month * 16 * business.pigs_sold_16_months
  let total_feed_cost := feed_cost_12_months + feed_cost_16_months
  revenue - total_feed_cost

/-- The farmer's profit is $960 -/
theorem farmer_profit_is_960 (business : PigBusiness) 
    (h1 : business.num_piglets = 6)
    (h2 : business.sale_price = 300)
    (h3 : business.min_growth_months = 12)
    (h4 : business.feed_cost_per_month = 10)
    (h5 : business.pigs_sold_12_months = 3)
    (h6 : business.pigs_sold_16_months = 3) :
    calculate_profit business = 960 := by
  sorry

end NUMINAMATH_CALUDE_farmer_profit_is_960_l307_30753


namespace NUMINAMATH_CALUDE_digits_of_large_number_l307_30777

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- 8^22 * 5^19 expressed as a natural number -/
def large_number : ℕ := 8^22 * 5^19

theorem digits_of_large_number :
  num_digits large_number = 35 := by sorry

end NUMINAMATH_CALUDE_digits_of_large_number_l307_30777


namespace NUMINAMATH_CALUDE_geometric_sequence_iff_c_eq_neg_one_l307_30747

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) (c : ℝ) : ℝ := 2^n + c

/-- The n-th term of the sequence a_n -/
def a (n : ℕ) (c : ℝ) : ℝ := S n c - S (n-1) c

/-- Predicate to check if a sequence is geometric -/
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, n > 0 → a (n+1) = r * a n

theorem geometric_sequence_iff_c_eq_neg_one (c : ℝ) :
  is_geometric (a · c) ↔ c = -1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_iff_c_eq_neg_one_l307_30747


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l307_30781

def U : Finset Nat := {1,2,3,4,5,6}
def M : Finset Nat := {1,3,4}

theorem complement_of_M_in_U : 
  (U \ M) = {2,5,6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l307_30781


namespace NUMINAMATH_CALUDE_solve_for_y_l307_30794

theorem solve_for_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) (h4 : x / y = 81) : 
  y = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l307_30794


namespace NUMINAMATH_CALUDE_fraction_sum_squared_l307_30757

theorem fraction_sum_squared (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_squared_l307_30757


namespace NUMINAMATH_CALUDE_sin_18_cos_12_plus_cos_18_sin_12_l307_30785

theorem sin_18_cos_12_plus_cos_18_sin_12 :
  Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
  Real.cos (18 * π / 180) * Real.sin (12 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_cos_12_plus_cos_18_sin_12_l307_30785


namespace NUMINAMATH_CALUDE_equation_system_solution_l307_30775

theorem equation_system_solution : ∃ (x y z : ℝ),
  (2 * x - 3 * y - z = 0) ∧
  (x + 3 * y - 14 * z = 0) ∧
  (z = 2) ∧
  ((x^2 + 3*x*y) / (y^2 + z^2) = 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l307_30775


namespace NUMINAMATH_CALUDE_program_arrangements_l307_30732

/-- The number of solo segments in the program -/
def num_solo_segments : ℕ := 5

/-- The number of chorus segments in the program -/
def num_chorus_segments : ℕ := 3

/-- The number of spaces available for chorus segments after arranging solo segments -/
def num_spaces_for_chorus : ℕ := num_solo_segments + 1 - 1 -- +1 for spaces between solos, -1 for not placing first

/-- The number of different programs that can be arranged -/
def num_programs : ℕ := (Nat.factorial num_solo_segments) * (num_spaces_for_chorus.choose num_chorus_segments)

theorem program_arrangements :
  num_programs = 7200 :=
sorry

end NUMINAMATH_CALUDE_program_arrangements_l307_30732


namespace NUMINAMATH_CALUDE_females_dont_listen_l307_30761

/-- A structure representing the survey results -/
structure SurveyResults where
  total_listen : Nat
  males_listen : Nat
  total_dont_listen : Nat
  total_respondents : Nat
  males_listen_le_total_listen : males_listen ≤ total_listen
  total_respondents_eq : total_respondents = total_listen + total_dont_listen

/-- The theorem stating the number of females who don't listen to the radio station -/
theorem females_dont_listen (survey : SurveyResults)
  (h_total_listen : survey.total_listen = 200)
  (h_males_listen : survey.males_listen = 75)
  (h_total_dont_listen : survey.total_dont_listen = 180)
  (h_total_respondents : survey.total_respondents = 380) :
  survey.total_dont_listen - (survey.total_respondents - survey.total_listen) = 180 := by
  sorry


end NUMINAMATH_CALUDE_females_dont_listen_l307_30761


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l307_30764

theorem geometric_sequence_fourth_term 
  (a : ℝ) -- first term
  (h1 : a ≠ 0) -- ensure first term is non-zero for division
  (h2 : (3*a + 3) / a = (6*a + 6) / (3*a + 3)) -- condition for geometric sequence
  (h3 : 3*a + 3 = a * ((3*a + 3) / a)) -- second term definition
  (h4 : 6*a + 6 = (3*a + 3) * ((3*a + 3) / a)) -- third term definition
  : a * ((3*a + 3) / a)^3 = -24 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l307_30764


namespace NUMINAMATH_CALUDE_right_triangle_square_distance_l307_30702

/-- Given a right triangle with hypotenuse forming the side of a square outside the triangle,
    and the sum of the legs being d, the distance from the right angle vertex to the center
    of the square is (d * √2) / 2. -/
theorem right_triangle_square_distance (d : ℝ) (h : d > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b = d ∧
    a^2 + b^2 = c^2 ∧
    (d * Real.sqrt 2) / 2 = c * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_square_distance_l307_30702


namespace NUMINAMATH_CALUDE_debt_settlement_possible_l307_30741

theorem debt_settlement_possible (vasya_coin_value : ℕ) (petya_coin_value : ℕ) 
  (debt : ℕ) (h1 : vasya_coin_value = 49) (h2 : petya_coin_value = 99) (h3 : debt = 1) :
  ∃ (n m : ℕ), vasya_coin_value * n - petya_coin_value * m = debt :=
by sorry

end NUMINAMATH_CALUDE_debt_settlement_possible_l307_30741


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l307_30774

theorem modulus_of_complex_number (z : ℂ) : z = 2 / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l307_30774


namespace NUMINAMATH_CALUDE_smallest_d_for_g_range_three_l307_30716

/-- The function g(x) defined as x^2 + 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- Theorem stating that 7 is the smallest value of d for which 3 is in the range of g(x) -/
theorem smallest_d_for_g_range_three :
  (∃ (d : ℝ), (∃ (x : ℝ), g d x = 3) ∧ (∀ (d' : ℝ), d' < d → ¬∃ (x : ℝ), g d' x = 3)) ∧
  (∃ (x : ℝ), g 7 x = 3) :=
sorry

end NUMINAMATH_CALUDE_smallest_d_for_g_range_three_l307_30716


namespace NUMINAMATH_CALUDE_complement_cardinality_l307_30771

def U : Finset Nat := {1,2,3,4,5,6}
def M : Finset Nat := {2,3,5}
def N : Finset Nat := {4,5}

theorem complement_cardinality : Finset.card (U \ (M ∪ N)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_cardinality_l307_30771


namespace NUMINAMATH_CALUDE_circular_film_radius_l307_30733

/-- The radius of a circular film formed by pouring a liquid from a rectangular box into water -/
theorem circular_film_radius (box_length box_width box_height film_thickness : ℝ) 
  (h1 : box_length = 8)
  (h2 : box_width = 4)
  (h3 : box_height = 15)
  (h4 : film_thickness = 0.2)
  : ∃ (r : ℝ), r = Real.sqrt (2400 / Real.pi) ∧ 
    π * r^2 * film_thickness = box_length * box_width * box_height :=
by sorry

end NUMINAMATH_CALUDE_circular_film_radius_l307_30733


namespace NUMINAMATH_CALUDE_x_minus_y_positive_l307_30765

theorem x_minus_y_positive (x y a : ℝ) 
  (h1 : x + y > 0) 
  (h2 : a < 0) 
  (h3 : a * y > 0) : 
  x - y > 0 := by sorry

end NUMINAMATH_CALUDE_x_minus_y_positive_l307_30765


namespace NUMINAMATH_CALUDE_laundry_time_calculation_l307_30724

theorem laundry_time_calculation (loads : ℕ) (wash_time dry_time : ℕ) : 
  loads = 8 → 
  wash_time = 45 → 
  dry_time = 60 → 
  (loads * (wash_time + dry_time)) / 60 = 14 := by
sorry

end NUMINAMATH_CALUDE_laundry_time_calculation_l307_30724


namespace NUMINAMATH_CALUDE_union_M_N_is_half_open_interval_l307_30780

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2}
def N : Set ℝ := {y | ∃ x, y = 2^x ∧ x < 0}

-- State the theorem
theorem union_M_N_is_half_open_interval :
  M ∪ N = Set.Icc 0 1 \ {1} :=
sorry

end NUMINAMATH_CALUDE_union_M_N_is_half_open_interval_l307_30780


namespace NUMINAMATH_CALUDE_max_area_difference_l307_30704

/-- A rectangle with integer dimensions and perimeter 160 cm -/
structure Rectangle where
  length : ℕ
  width : ℕ
  perimeter_constraint : length + width = 80

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The theorem statement -/
theorem max_area_difference : 
  ∃ (r1 r2 : Rectangle), 
    (∀ r : Rectangle, area r ≤ area r1 ∧ area r ≥ area r2) ∧
    area r1 - area r2 = 1521 ∧
    r1.length = 40 ∧ r1.width = 40 ∧
    r2.length = 1 ∧ r2.width = 79 := by
  sorry


end NUMINAMATH_CALUDE_max_area_difference_l307_30704


namespace NUMINAMATH_CALUDE_team_selection_theorem_l307_30791

def internal_medicine_doctors : ℕ := 12
def surgeons : ℕ := 8
def team_size : ℕ := 5

def select_team_with_restrictions (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem team_selection_theorem :
  (select_team_with_restrictions (internal_medicine_doctors + surgeons - 2) (team_size - 1) = 3060) ∧
  (Nat.choose (internal_medicine_doctors + surgeons) team_size - 
   Nat.choose internal_medicine_doctors team_size - 
   Nat.choose surgeons team_size = 14656) :=
by sorry

end NUMINAMATH_CALUDE_team_selection_theorem_l307_30791


namespace NUMINAMATH_CALUDE_product_calculation_l307_30750

theorem product_calculation : 3.5 * 7.2 * (6.3 - 1.4) = 122.5 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l307_30750


namespace NUMINAMATH_CALUDE_min_value_of_f_range_of_x_when_f_leq_5_l307_30746

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 4| + |x - 1|

-- Theorem for the minimum value of f(x)
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 3 :=
sorry

-- Theorem for the range of x when f(x) ≤ 5
theorem range_of_x_when_f_leq_5 :
  ∀ x, f x ≤ 5 ↔ 0 ≤ x ∧ x ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_range_of_x_when_f_leq_5_l307_30746


namespace NUMINAMATH_CALUDE_oligarch_wealth_comparison_l307_30782

/-- Represents the wealth of an oligarch at a given time -/
structure OligarchWealth where
  amount : ℝ
  year : ℕ
  name : String

/-- Represents the national wealth of the country -/
def NationalWealth : Type := ℝ

/-- The problem statement -/
theorem oligarch_wealth_comparison 
  (maximilian_2011 maximilian_2012 alejandro_2011 alejandro_2012 : OligarchWealth)
  (national_wealth : NationalWealth) :
  (alejandro_2012.amount = 2 * maximilian_2011.amount) →
  (maximilian_2012.amount < alejandro_2011.amount) →
  (national_wealth = alejandro_2012.amount + maximilian_2012.amount - alejandro_2011.amount - maximilian_2011.amount) →
  (maximilian_2011.amount > national_wealth) := by
  sorry

end NUMINAMATH_CALUDE_oligarch_wealth_comparison_l307_30782


namespace NUMINAMATH_CALUDE_inverse_g_sum_l307_30755

-- Define the function g
def g (x : ℝ) : ℝ := x * |x|

-- State the theorem
theorem inverse_g_sum : ∃ (a b : ℝ), g a = 9 ∧ g b = -64 ∧ a + b = -5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_sum_l307_30755


namespace NUMINAMATH_CALUDE_track_circumference_l307_30739

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (v1 v2 : ℝ) (t : ℝ) (h1 : v1 = 20) (h2 : v2 = 13) (h3 : t = 33 / 60) :
  v1 * t + v2 * t = 18.15 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_l307_30739


namespace NUMINAMATH_CALUDE_min_value_F_l307_30788

/-- The function F(x, y) -/
def F (x y : ℝ) : ℝ := x^2 + 8*y + y^2 + 14*x - 6

/-- The constraint equation -/
def constraint (x y : ℝ) : Prop := x^2 + y^2 + 25 = 10*(x + y)

/-- Theorem stating that the minimum value of F(x, y) is 29 under the given constraint -/
theorem min_value_F :
  ∃ (m : ℝ), m = 29 ∧
  (∀ x y : ℝ, constraint x y → F x y ≥ m) ∧
  (∃ x y : ℝ, constraint x y ∧ F x y = m) :=
sorry

end NUMINAMATH_CALUDE_min_value_F_l307_30788


namespace NUMINAMATH_CALUDE_distance_to_focus_l307_30789

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def P (y : ℝ) : ℝ × ℝ := (4, y)

-- Theorem statement
theorem distance_to_focus (y : ℝ) (h : parabola 4 y) : 
  Real.sqrt ((P y).1 - focus.1)^2 + ((P y).2 - focus.2)^2 = 5 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l307_30789
