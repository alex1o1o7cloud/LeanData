import Mathlib

namespace NUMINAMATH_CALUDE_trick_sum_prediction_l2033_203352

theorem trick_sum_prediction (a b : ℕ) (ha : 10000 ≤ a ∧ a < 100000) : 
  a + b + (99999 - b) = 100000 + a - 1 := by
  sorry

end NUMINAMATH_CALUDE_trick_sum_prediction_l2033_203352


namespace NUMINAMATH_CALUDE_unique_solution_implies_n_equals_8_l2033_203310

-- Define the quadratic equation
def quadratic_equation (n : ℝ) (x : ℝ) : ℝ := 4 * x^2 + n * x + 4

-- Define the discriminant of the quadratic equation
def discriminant (n : ℝ) : ℝ := n^2 - 4 * 4 * 4

-- Theorem statement
theorem unique_solution_implies_n_equals_8 :
  ∃! x : ℝ, quadratic_equation 8 x = 0 ∧
  ∀ n : ℝ, (∃! x : ℝ, quadratic_equation n x = 0) → n = 8 ∨ n = -8 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_n_equals_8_l2033_203310


namespace NUMINAMATH_CALUDE_alpha_value_l2033_203378

/-- Given that α is inversely proportional to β and directly proportional to γ,
    prove that α = 2.5 when β = 30 and γ = 6, given that α = 5 when β = 15 and γ = 3 -/
theorem alpha_value (α β γ : ℝ) (h1 : ∃ k : ℝ, α * β = k)
    (h2 : ∃ j : ℝ, α * γ = j) (h3 : α = 5 ∧ β = 15 ∧ γ = 3) :
  α = 2.5 ∧ β = 30 ∧ γ = 6 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l2033_203378


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2033_203390

theorem quadratic_inequality_solution (x : ℝ) :
  (-x^2 + 5*x - 4 < 0) ↔ (1 < x ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2033_203390


namespace NUMINAMATH_CALUDE_compute_F_2_f_3_l2033_203306

-- Define function f
def f (a : ℝ) : ℝ := a^2 - 3*a + 2

-- Define function F
def F (a b : ℝ) : ℝ := b + a^3

-- Theorem to prove
theorem compute_F_2_f_3 : F 2 (f 3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_compute_F_2_f_3_l2033_203306


namespace NUMINAMATH_CALUDE_soda_difference_l2033_203381

theorem soda_difference (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 79) (h2 : diet_soda = 53) : 
  regular_soda - diet_soda = 26 := by
  sorry

end NUMINAMATH_CALUDE_soda_difference_l2033_203381


namespace NUMINAMATH_CALUDE_middle_number_theorem_l2033_203389

theorem middle_number_theorem (x y z : ℤ) 
  (h_order : x < y ∧ y < z)
  (h_sum1 : x + y = 10)
  (h_sum2 : x + z = 21)
  (h_sum3 : y + z = 25) : 
  y = 7 := by
sorry

end NUMINAMATH_CALUDE_middle_number_theorem_l2033_203389


namespace NUMINAMATH_CALUDE_max_time_sum_of_digits_l2033_203356

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Fin 24
  minutes : Fin 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits for a Time24 -/
def timeSumOfDigits (t : Time24) : ℕ :=
  sumOfDigits t.hours.val + sumOfDigits t.minutes.val

/-- The maximum sum of digits for any Time24 -/
def maxTimeSumOfDigits : ℕ := 24

theorem max_time_sum_of_digits :
  ∀ t : Time24, timeSumOfDigits t ≤ maxTimeSumOfDigits := by sorry

end NUMINAMATH_CALUDE_max_time_sum_of_digits_l2033_203356


namespace NUMINAMATH_CALUDE_range_x_when_m_is_one_range_m_for_not_p_necessary_but_not_sufficient_l2033_203375

-- Define propositions p and q
def p (x m : ℝ) : Prop := |2 * x - m| ≥ 1
def q (x : ℝ) : Prop := (1 - 3 * x) / (x + 2) > 0

-- Theorem for part 1
theorem range_x_when_m_is_one :
  {x : ℝ | p x 1 ∧ q x} = {x : ℝ | -2 < x ∧ x ≤ 0} := by sorry

-- Theorem for part 2
theorem range_m_for_not_p_necessary_but_not_sufficient :
  {m : ℝ | ∀ x, q x → ¬(p x m) ∧ ∃ y, ¬(p y m) ∧ ¬(q y)} = {m : ℝ | -3 ≤ m ∧ m ≤ -1/3} := by sorry

end NUMINAMATH_CALUDE_range_x_when_m_is_one_range_m_for_not_p_necessary_but_not_sufficient_l2033_203375


namespace NUMINAMATH_CALUDE_triangle_side_length_l2033_203371

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  a = 2 → B = π / 3 → c = 3 → b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2033_203371


namespace NUMINAMATH_CALUDE_investment_amount_correct_l2033_203326

/-- Calculates the investment amount in T-shirt printing equipment -/
def calculate_investment (cost_per_shirt : ℚ) (selling_price : ℚ) (break_even_point : ℕ) : ℚ :=
  selling_price * break_even_point - cost_per_shirt * break_even_point

/-- Proves that the investment amount is correct -/
theorem investment_amount_correct (cost_per_shirt : ℚ) (selling_price : ℚ) (break_even_point : ℕ) :
  calculate_investment cost_per_shirt selling_price break_even_point = 1411 :=
by
  have h1 : cost_per_shirt = 3 := by sorry
  have h2 : selling_price = 20 := by sorry
  have h3 : break_even_point = 83 := by sorry
  sorry

#eval calculate_investment 3 20 83

end NUMINAMATH_CALUDE_investment_amount_correct_l2033_203326


namespace NUMINAMATH_CALUDE_total_amount_paid_l2033_203324

def grape_quantity : ℕ := 3
def grape_rate : ℕ := 70
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 55

theorem total_amount_paid : 
  grape_quantity * grape_rate + mango_quantity * mango_rate = 705 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l2033_203324


namespace NUMINAMATH_CALUDE_problem_hall_tilings_l2033_203355

/-- Represents a tiling configuration for a rectangular hall. -/
structure HallTiling where
  width : ℕ
  length : ℕ
  black_tiles : ℕ
  white_tiles : ℕ

/-- Counts the number of valid tiling configurations. -/
def countValidTilings (h : HallTiling) : ℕ :=
  sorry

/-- The specific hall configuration from the problem. -/
def problemHall : HallTiling :=
  { width := 2
  , length := 13
  , black_tiles := 11
  , white_tiles := 15 }

/-- Theorem stating that the number of valid tilings for the problem hall is 486. -/
theorem problem_hall_tilings :
  countValidTilings problemHall = 486 :=
sorry

end NUMINAMATH_CALUDE_problem_hall_tilings_l2033_203355


namespace NUMINAMATH_CALUDE_problem_solution_l2033_203365

theorem problem_solution : 
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁^3 - 3*x₁*y₁^2 = 2010) ∧ (y₁^3 - 3*x₁^2*y₁ = 2006) ∧
    (x₂^3 - 3*x₂*y₂^2 = 2010) ∧ (y₂^3 - 3*x₂^2*y₂ = 2006) ∧
    (x₃^3 - 3*x₃*y₃^2 = 2010) ∧ (y₃^3 - 3*x₃^2*y₃ = 2006) ∧
    ((1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 996/1005) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2033_203365


namespace NUMINAMATH_CALUDE_flea_collar_count_l2033_203308

/-- Represents the number of dogs with flea collars in a kennel -/
def dogs_with_flea_collars (total : ℕ) (with_tags : ℕ) (with_both : ℕ) (with_neither : ℕ) : ℕ :=
  total - with_tags + with_both - with_neither

/-- Theorem stating that in a kennel of 80 dogs, where 45 dogs wear tags, 
    6 dogs wear both tags and flea collars, and 1 dog wears neither, 
    the number of dogs wearing flea collars is 40. -/
theorem flea_collar_count : 
  dogs_with_flea_collars 80 45 6 1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_flea_collar_count_l2033_203308


namespace NUMINAMATH_CALUDE_initial_students_per_class_l2033_203318

theorem initial_students_per_class 
  (initial_classes : ℕ) 
  (added_classes : ℕ) 
  (total_students : ℕ) 
  (h1 : initial_classes = 15)
  (h2 : added_classes = 5)
  (h3 : total_students = 400) :
  (total_students / (initial_classes + added_classes) : ℚ) = 20 := by
sorry

end NUMINAMATH_CALUDE_initial_students_per_class_l2033_203318


namespace NUMINAMATH_CALUDE_original_eq_hyperbola_and_ellipse_l2033_203348

-- Define the original equation
def original_equation (x y : ℝ) : Prop := y^4 - 4*x^4 = 2*y^2 - 1

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := y^2 - 2*x^2 = 1

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := y^2 + 2*x^2 = 1

-- Theorem stating that the original equation is equivalent to the union of a hyperbola and an ellipse
theorem original_eq_hyperbola_and_ellipse :
  ∀ x y : ℝ, original_equation x y ↔ (hyperbola_equation x y ∨ ellipse_equation x y) :=
by sorry

end NUMINAMATH_CALUDE_original_eq_hyperbola_and_ellipse_l2033_203348


namespace NUMINAMATH_CALUDE_white_balls_count_l2033_203316

theorem white_balls_count 
  (total_balls : ℕ) 
  (total_draws : ℕ) 
  (white_draws : ℕ) 
  (h1 : total_balls = 20) 
  (h2 : total_draws = 404) 
  (h3 : white_draws = 101) : 
  (total_balls : ℚ) * (white_draws : ℚ) / (total_draws : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l2033_203316


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2033_203329

theorem smallest_positive_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 4 = 3 ∧ 
  n % 6 = 5 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 4 = 3 ∧ m % 6 = 5 → n ≤ m) ∧
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2033_203329


namespace NUMINAMATH_CALUDE_expression_factorization_l2033_203357

theorem expression_factorization (x : ℝ) :
  (20 * x^3 + 45 * x^2 - 10) - (-5 * x^3 + 15 * x^2 - 5) = 5 * (5 * x^3 + 6 * x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2033_203357


namespace NUMINAMATH_CALUDE_teacher_distribution_l2033_203331

/-- The number of ways to distribute 4 teachers to 3 places -/
def distribute_teachers : ℕ := 36

/-- The number of ways to choose 2 teachers out of 4 -/
def choose_two_from_four : ℕ := Nat.choose 4 2

/-- The number of ways to arrange 3 groups into 3 places -/
def arrange_three_groups : ℕ := 6

theorem teacher_distribution :
  distribute_teachers = choose_two_from_four * arrange_three_groups :=
sorry

end NUMINAMATH_CALUDE_teacher_distribution_l2033_203331


namespace NUMINAMATH_CALUDE_two_colonies_limit_days_l2033_203340

/-- Represents the number of days it takes for a single bacteria colony to reach the habitat limit -/
def single_colony_limit_days : ℕ := 20

/-- Represents the growth rate of the bacteria colony (doubling every day) -/
def growth_rate : ℚ := 2

/-- Represents the fixed habitat limit -/
def habitat_limit : ℚ := growth_rate ^ single_colony_limit_days

/-- Theorem stating that two colonies reach the habitat limit in the same number of days as one colony -/
theorem two_colonies_limit_days (initial_colonies : ℕ) (h : initial_colonies = 2) :
  (initial_colonies * growth_rate ^ single_colony_limit_days = habitat_limit) :=
sorry

end NUMINAMATH_CALUDE_two_colonies_limit_days_l2033_203340


namespace NUMINAMATH_CALUDE_basketball_game_probability_basketball_game_probability_proof_l2033_203330

/-- The probability that at least 7 out of 8 people stay for an entire basketball game,
    given that 4 are certain to stay and 4 have a 1/3 probability of staying. -/
theorem basketball_game_probability : ℝ :=
  let total_people : ℕ := 8
  let certain_people : ℕ := 4
  let uncertain_people : ℕ := 4
  let stay_probability : ℝ := 1/3
  let at_least_stay : ℕ := 7

  1/9

/-- Proof of the basketball game probability theorem -/
theorem basketball_game_probability_proof :
  basketball_game_probability = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_probability_basketball_game_probability_proof_l2033_203330


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2033_203359

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  (1 / m + 1 / n) ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2033_203359


namespace NUMINAMATH_CALUDE_correct_addition_and_rounding_l2033_203361

-- Define the addition operation
def add (a b : ℕ) : ℕ := a + b

-- Define the rounding operation to the nearest ten
def roundToNearestTen (n : ℕ) : ℕ :=
  let lastDigit := n % 10
  if lastDigit < 5 then
    n - lastDigit
  else
    n + (10 - lastDigit)

-- Theorem statement
theorem correct_addition_and_rounding :
  roundToNearestTen (add 46 37) = 80 := by sorry

end NUMINAMATH_CALUDE_correct_addition_and_rounding_l2033_203361


namespace NUMINAMATH_CALUDE_power_of_81_l2033_203387

theorem power_of_81 : (81 : ℝ) ^ (5/2) = 59049 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_l2033_203387


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l2033_203320

/-- The height of the pole in meters -/
def pole_height : ℝ := 10

/-- The time taken to reach the top of the pole in minutes -/
def total_time : ℕ := 17

/-- The distance the monkey slips in alternate minutes -/
def slip_distance : ℝ := 1

/-- The distance the monkey ascends in the first minute -/
def ascend_distance : ℝ := 1.8

/-- The number of complete ascend-slip cycles -/
def num_cycles : ℕ := (total_time - 1) / 2

theorem monkey_climb_theorem :
  ascend_distance + num_cycles * (ascend_distance - slip_distance) + ascend_distance = pole_height :=
sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l2033_203320


namespace NUMINAMATH_CALUDE_maximum_value_inequality_l2033_203350

theorem maximum_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : (2 : ℝ) / 5 ≤ z) (h2 : z ≤ min x y) (h3 : x * z ≥ (4 : ℝ) / 15) (h4 : y * z ≥ (1 : ℝ) / 5) :
  (1 : ℝ) / x + 2 / y + 3 / z ≤ 13 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    (2 : ℝ) / 5 ≤ z₀ ∧ z₀ ≤ min x₀ y₀ ∧ x₀ * z₀ ≥ (4 : ℝ) / 15 ∧ y₀ * z₀ ≥ (1 : ℝ) / 5 ∧
    (1 : ℝ) / x₀ + 2 / y₀ + 3 / z₀ = 13 := by
  sorry

end NUMINAMATH_CALUDE_maximum_value_inequality_l2033_203350


namespace NUMINAMATH_CALUDE_harriet_speed_l2033_203360

/-- Harriet's round trip between A-ville and B-town -/
theorem harriet_speed (total_time : ℝ) (time_to_b : ℝ) (speed_from_b : ℝ) :
  total_time = 5 →
  time_to_b = 3 →
  speed_from_b = 150 →
  ∃ (distance : ℝ) (speed_to_b : ℝ),
    distance = speed_from_b * (total_time - time_to_b) ∧
    distance = speed_to_b * time_to_b ∧
    speed_to_b = 100 := by
  sorry


end NUMINAMATH_CALUDE_harriet_speed_l2033_203360


namespace NUMINAMATH_CALUDE_oil_press_statement_is_false_l2033_203398

-- Define the oil press output function
def oil_press_output (num_presses : ℕ) (output : ℕ) : Prop :=
  num_presses > 0 ∧ output > 0 ∧ (num_presses * (output / num_presses) = output)

-- State the theorem
theorem oil_press_statement_is_false :
  oil_press_output 5 260 →
  ¬ (oil_press_output 20 7200) :=
by
  sorry

end NUMINAMATH_CALUDE_oil_press_statement_is_false_l2033_203398


namespace NUMINAMATH_CALUDE_acute_triangle_inequality_l2033_203322

/-- Triangle with acute angle opposite to side c -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b
  acute_angle : c^2 < a^2 + b^2

theorem acute_triangle_inequality (t : AcuteTriangle) :
  (t.a^2 + t.b^2 + t.c^2) / (t.a^2 + t.b^2) > 1 :=
sorry

end NUMINAMATH_CALUDE_acute_triangle_inequality_l2033_203322


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2033_203347

/-- Given two vectors a and b in R², where a = (-2, 2) and b = (x, -3),
    if a is perpendicular to b, then x = -3. -/
theorem perpendicular_vectors_x_value :
  let a : Fin 2 → ℝ := ![-2, 2]
  let b : Fin 2 → ℝ := ![x, -3]
  (∀ (i : Fin 2), a i * b i = 0) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2033_203347


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2033_203304

/-- Given a geometric sequence {a_n} with a_1 = 1 and a_5 = 1/9, 
    prove that the product a_2 * a_3 * a_4 = 1/27 -/
theorem geometric_sequence_product (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →                                -- first term
  a 5 = 1 / 9 →                            -- fifth term
  a 2 * a 3 * a 4 = 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2033_203304


namespace NUMINAMATH_CALUDE_jim_reading_speed_increase_l2033_203368

-- Define Jim's reading parameters
def original_rate : ℝ := 40 -- pages per hour
def original_total : ℝ := 600 -- pages per week
def time_reduction : ℝ := 4 -- hours
def new_total : ℝ := 660 -- pages per week

-- Theorem statement
theorem jim_reading_speed_increase :
  let original_time := original_total / original_rate
  let new_time := original_time - time_reduction
  let new_rate := new_total / new_time
  new_rate / original_rate = 1.5
  := by sorry

end NUMINAMATH_CALUDE_jim_reading_speed_increase_l2033_203368


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2033_203354

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a₂ + a₁₀ = 16, then a₄ + a₈ = 16 -/
theorem arithmetic_sequence_sum_property 
  (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 2 + a 10 = 16) : 
  a 4 + a 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2033_203354


namespace NUMINAMATH_CALUDE_solution_correctness_l2033_203385

variables {α : Type*} [Field α]
variables (x y z x' y' z' x'' y'' z'' u' v' w' : α)

def system_solution (u v w : α) : Prop :=
  x * u + y * v + z * w = u' ∧
  x' * u + y' * v + z' * w = v' ∧
  x'' * u + y'' * v + z'' * w = w'

theorem solution_correctness :
  ∃ (u v w : α),
    system_solution x y z x' y' z' x'' y'' z'' u' v' w' u v w ∧
    u = u' * x + v' * x' + w' * x'' ∧
    v = u' * y + v' * y' + w' * y'' ∧
    w = u' * z + v' * z' + w' * z'' :=
  sorry

end NUMINAMATH_CALUDE_solution_correctness_l2033_203385


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_l2033_203335

theorem min_value_sqrt_sum (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 1)^2 + (x + 2)^2) ≥ Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_l2033_203335


namespace NUMINAMATH_CALUDE_school_election_votes_l2033_203327

theorem school_election_votes (bob_votes : ℕ) (total_votes : ℕ) 
  (h1 : bob_votes = 48)
  (h2 : bob_votes = (2 : ℕ) * total_votes / (5 : ℕ)) :
  total_votes = 120 := by
  sorry

end NUMINAMATH_CALUDE_school_election_votes_l2033_203327


namespace NUMINAMATH_CALUDE_smallest_non_nine_divisible_by_999_l2033_203364

/-- Checks if a natural number contains the digit 9 --/
def containsNine (n : ℕ) : Prop :=
  ∃ (k : ℕ), n / (10^k) % 10 = 9

/-- Checks if a natural number is divisible by 999 --/
def divisibleBy999 (n : ℕ) : Prop :=
  n % 999 = 0

theorem smallest_non_nine_divisible_by_999 :
  ∀ n : ℕ, n > 0 → divisibleBy999 n → ¬containsNine n → n ≥ 112 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_nine_divisible_by_999_l2033_203364


namespace NUMINAMATH_CALUDE_chopped_cube_height_l2033_203349

/-- Given a unit cube with a corner chopped off through the midpoints of the three adjacent edges,
    when the freshly-cut face is placed on a table, the height of the remaining solid is 29/32. -/
theorem chopped_cube_height : 
  let cube_edge : ℝ := 1
  let midpoint_factor : ℝ := 1/2
  let chopped_volume : ℝ := 3/32
  let remaining_volume : ℝ := 1 - chopped_volume
  let base_area : ℝ := cube_edge^2
  remaining_volume / base_area = 29/32 := by sorry

end NUMINAMATH_CALUDE_chopped_cube_height_l2033_203349


namespace NUMINAMATH_CALUDE_lilys_balance_proof_l2033_203358

/-- Calculates Lily's final account balance after a series of transactions --/
def lilys_final_balance (initial_amount shirt_cost book_price book_discount 
  savings_rate gift_percentage : ℚ) (num_books : ℕ) : ℚ :=
  let shoes_cost := 3 * shirt_cost
  let discounted_book_price := book_price * (1 - book_discount)
  let total_book_cost := (num_books : ℚ) * discounted_book_price
  let remaining_after_purchases := initial_amount - shirt_cost - shoes_cost - total_book_cost
  let savings := remaining_after_purchases / 2
  let savings_with_interest := savings * (1 + savings_rate)
  let gift_cost := savings_with_interest * gift_percentage
  savings_with_interest - gift_cost

theorem lilys_balance_proof :
  lilys_final_balance 55 7 8 0.2 0.2 0.25 4 = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_lilys_balance_proof_l2033_203358


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2033_203321

theorem polynomial_factorization (a b : ℤ) :
  (∀ x : ℝ, 24 * x^2 - 158 * x - 147 = (12 * x + a) * (2 * x + b)) →
  a + 2 * b = -35 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2033_203321


namespace NUMINAMATH_CALUDE_card_A_total_percent_decrease_l2033_203393

def card_A_initial_value : ℝ := 150
def card_A_decrease_year1 : ℝ := 0.20
def card_A_decrease_year2 : ℝ := 0.30
def card_A_decrease_year3 : ℝ := 0.15

def card_A_value_after_three_years : ℝ :=
  card_A_initial_value * (1 - card_A_decrease_year1) * (1 - card_A_decrease_year2) * (1 - card_A_decrease_year3)

theorem card_A_total_percent_decrease :
  (card_A_initial_value - card_A_value_after_three_years) / card_A_initial_value = 0.524 := by
  sorry

end NUMINAMATH_CALUDE_card_A_total_percent_decrease_l2033_203393


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l2033_203312

/-- Represents the duration of a traffic light cycle in seconds -/
def cycle_duration : ℕ := 95

/-- Represents the number of color changes in a cycle -/
def color_changes : ℕ := 3

/-- Represents the duration of each color change in seconds -/
def change_duration : ℕ := 5

/-- Represents the duration of the observation interval in seconds -/
def observation_interval : ℕ := 5

/-- The probability of observing a color change during a random observation interval -/
theorem traffic_light_change_probability :
  (color_changes * change_duration : ℚ) / cycle_duration = 3 / 19 := by sorry

end NUMINAMATH_CALUDE_traffic_light_change_probability_l2033_203312


namespace NUMINAMATH_CALUDE_gcd_of_156_and_195_l2033_203380

theorem gcd_of_156_and_195 : Nat.gcd 156 195 = 39 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_156_and_195_l2033_203380


namespace NUMINAMATH_CALUDE_current_speed_current_speed_is_3_l2033_203366

/-- The speed of the current in a river, given the man's rowing speed in still water,
    the distance covered downstream, and the time taken to cover that distance. -/
theorem current_speed (mans_speed : ℝ) (distance : ℝ) (time : ℝ) : ℝ :=
  let downstream_speed := distance / (time / 3600)
  downstream_speed - mans_speed

/-- Proof that the speed of the current is 3 kmph -/
theorem current_speed_is_3 :
  current_speed 15 0.06 11.999040076793857 = 3 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_current_speed_is_3_l2033_203366


namespace NUMINAMATH_CALUDE_power_of_one_fourth_l2033_203317

theorem power_of_one_fourth (n : ℤ) : 1024 * (1 / 4 : ℚ) ^ n = 64 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_one_fourth_l2033_203317


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_g_l2033_203303

def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 10| + |x - 1|

theorem sum_of_max_and_min_g : 
  ∃ (max min : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → g x ≤ max) ∧ 
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 9 ∧ g x = max) ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → min ≤ g x) ∧ 
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 9 ∧ g x = min) ∧
    max + min = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_g_l2033_203303


namespace NUMINAMATH_CALUDE_det_dilation_matrix_5_l2033_203374

/-- A dilation matrix with scale factor k -/
def dilationMatrix (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ => k)

/-- Theorem: The determinant of a 3x3 dilation matrix with scale factor 5 is 125 -/
theorem det_dilation_matrix_5 :
  Matrix.det (dilationMatrix 5) = 125 := by
  sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_5_l2033_203374


namespace NUMINAMATH_CALUDE_time_from_velocity_and_displacement_l2033_203370

/-- 
Given two equations relating velocity (V), displacement (S), time (t), 
acceleration (g), and initial velocity (V₀), prove that t can be expressed 
in terms of S, V, and V₀.
-/
theorem time_from_velocity_and_displacement 
  (V g t V₀ S : ℝ) 
  (hV : V = g * t + V₀) 
  (hS : S = (1/2) * g * t^2 + V₀ * t) : 
  t = 2 * S / (V + V₀) := by
sorry

end NUMINAMATH_CALUDE_time_from_velocity_and_displacement_l2033_203370


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2033_203373

theorem complex_fraction_equality : 
  (1 - Complex.I * Real.sqrt 3) / ((Real.sqrt 3 + Complex.I) ^ 2) = 
  -(1/4) - (Real.sqrt 3 / 4) * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2033_203373


namespace NUMINAMATH_CALUDE_min_square_area_l2033_203311

/-- A monic quartic polynomial with integer coefficients -/
structure MonicQuarticPolynomial where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- The roots of a polynomial form a square on the complex plane -/
def roots_form_square (poly : MonicQuarticPolynomial) : Prop :=
  sorry

/-- The area of the square formed by the roots of a polynomial -/
def square_area (poly : MonicQuarticPolynomial) : ℝ :=
  sorry

/-- The minimum possible area of a square formed by the roots of a monic quartic polynomial
    with integer coefficients is 2 -/
theorem min_square_area (poly : MonicQuarticPolynomial) 
  (h : roots_form_square poly) : 
  ∃ (min_area : ℝ), min_area = 2 ∧ ∀ (p : MonicQuarticPolynomial), 
  roots_form_square p → square_area p ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_min_square_area_l2033_203311


namespace NUMINAMATH_CALUDE_graduation_messages_l2033_203394

theorem graduation_messages (n : ℕ) (h : n = 45) : 
  (n * (n - 1)) / 2 = 990 := by
  sorry

end NUMINAMATH_CALUDE_graduation_messages_l2033_203394


namespace NUMINAMATH_CALUDE_total_stars_is_10_pow_22_l2033_203334

/-- The number of galaxies in the universe -/
def num_galaxies : ℕ := 10^11

/-- The number of stars in each galaxy -/
def stars_per_galaxy : ℕ := 10^11

/-- The total number of stars in the universe -/
def total_stars : ℕ := num_galaxies * stars_per_galaxy

/-- Theorem stating that the total number of stars is 10^22 -/
theorem total_stars_is_10_pow_22 : total_stars = 10^22 := by
  sorry

end NUMINAMATH_CALUDE_total_stars_is_10_pow_22_l2033_203334


namespace NUMINAMATH_CALUDE_kite_parabolas_theorem_l2033_203383

/-- Represents a parabola in the form y = ax² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Represents the parameters of our problem -/
structure KiteParameters where
  parabola1 : Parabola
  parabola2 : Parabola
  kite_area : ℝ

/-- The main theorem statement -/
theorem kite_parabolas_theorem (params : KiteParameters) : 
  params.parabola1.a + params.parabola2.a = 1.04 :=
by
  sorry

/-- The specific instance of our problem -/
def our_problem : KiteParameters :=
  { parabola1 := { a := 2, b := -3 }
  , parabola2 := { a := -1, b := 5 }
  , kite_area := 20
  }

#check kite_parabolas_theorem our_problem

end NUMINAMATH_CALUDE_kite_parabolas_theorem_l2033_203383


namespace NUMINAMATH_CALUDE_quadratic_solution_l2033_203315

theorem quadratic_solution (y : ℝ) : 
  y > 0 ∧ 6 * y^2 + 5 * y - 12 = 0 ↔ y = (-5 + Real.sqrt 313) / 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2033_203315


namespace NUMINAMATH_CALUDE_equation_solutions_l2033_203395

theorem equation_solutions :
  (∀ x : ℝ, x^2 + 6*x - 7 = 0 ↔ x = -7 ∨ x = 1) ∧
  (∀ x : ℝ, 4*x*(2*x+1) = 3*(2*x+1) ↔ x = -1/2 ∨ x = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2033_203395


namespace NUMINAMATH_CALUDE_luzhou_gdp_scientific_correct_l2033_203396

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_in_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The GDP value of Luzhou City in 2022 -/
def luzhou_gdp : ℕ := 260150000000

/-- The scientific notation representation of Luzhou's GDP -/
def luzhou_gdp_scientific : ScientificNotation :=
  { coefficient := 2.6015
    exponent := 11
    coeff_in_range := by sorry }

/-- Theorem stating that the scientific notation representation is correct -/
theorem luzhou_gdp_scientific_correct :
  (luzhou_gdp_scientific.coefficient * (10 : ℝ) ^ luzhou_gdp_scientific.exponent) = luzhou_gdp := by
  sorry

end NUMINAMATH_CALUDE_luzhou_gdp_scientific_correct_l2033_203396


namespace NUMINAMATH_CALUDE_bus_stop_time_l2033_203307

theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : 
  speed_without_stops = 50 →
  speed_with_stops = 35 →
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 18 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_time_l2033_203307


namespace NUMINAMATH_CALUDE_solution_equals_answer_l2033_203379

/-- A perfect square is an integer that is the square of another integer. -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

/-- The set of all integer pairs (a, b) satisfying the given conditions. -/
def solution_set : Set (ℤ × ℤ) :=
  {p : ℤ × ℤ | is_perfect_square (p.1^2 - 4*p.2) ∧ is_perfect_square (p.2^2 - 4*p.1)}

/-- The set described in the answer. -/
def answer_set : Set (ℤ × ℤ) :=
  {p : ℤ × ℤ | (∃ n : ℤ, p = (0, n^2) ∨ p = (n^2, 0)) ∨
               (p.1 > 0 ∧ p.2 = -1 - p.1) ∨
               (p.2 > 0 ∧ p.1 = -1 - p.2) ∨
               p = (4, 4) ∨ p = (5, 6) ∨ p = (6, 5)}

theorem solution_equals_answer : solution_set = answer_set :=
  sorry

end NUMINAMATH_CALUDE_solution_equals_answer_l2033_203379


namespace NUMINAMATH_CALUDE_min_product_with_98_zeros_l2033_203377

/-- The number of trailing zeros in a positive integer -/
def trailingZeros (n : ℕ+) : ℕ := sorry

/-- The concatenation of two positive integers -/
def concat (a b : ℕ+) : ℕ+ := sorry

/-- The statement of the problem -/
theorem min_product_with_98_zeros :
  ∃ (m n : ℕ+),
    (∀ (x y : ℕ+), trailingZeros (x^x.val * y^y.val) = 98 → m.val * n.val ≤ x.val * y.val) ∧
    trailingZeros (m^m.val * n^n.val) = 98 ∧
    trailingZeros (concat (concat m m) (concat n n)) = 98 ∧
    m.val * n.val = 7350 := by
  sorry

end NUMINAMATH_CALUDE_min_product_with_98_zeros_l2033_203377


namespace NUMINAMATH_CALUDE_janice_typing_problem_l2033_203397

theorem janice_typing_problem (typing_speed : ℕ) (initial_typing_time : ℕ) 
  (additional_typing_time : ℕ) (erased_sentences : ℕ) (final_typing_time : ℕ) 
  (total_sentences : ℕ) : 
  typing_speed = 6 →
  initial_typing_time = 20 →
  additional_typing_time = 15 →
  erased_sentences = 40 →
  final_typing_time = 18 →
  total_sentences = 536 →
  total_sentences - (typing_speed * (initial_typing_time + additional_typing_time + final_typing_time) - erased_sentences) = 258 := by
  sorry

end NUMINAMATH_CALUDE_janice_typing_problem_l2033_203397


namespace NUMINAMATH_CALUDE_alpha_value_l2033_203343

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) : 
  α = -16 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l2033_203343


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l2033_203392

theorem davids_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℚ) 
  (num_subjects : ℕ) 
  (h1 : english = 45) 
  (h2 : mathematics = 35) 
  (h3 : physics = 52) 
  (h4 : biology = 55) 
  (h5 : average = 46.8) 
  (h6 : num_subjects = 5) :
  ∃ (chemistry : ℕ), 
    (english + mathematics + physics + biology + chemistry : ℚ) / num_subjects = average ∧ 
    chemistry = 47 := by
  sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l2033_203392


namespace NUMINAMATH_CALUDE_complement_intersection_S_T_l2033_203353

def S : Finset Int := {-2, -1, 0, 1, 2}
def T : Finset Int := {-1, 0, 1}

theorem complement_intersection_S_T :
  (S \ (S ∩ T)) = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_S_T_l2033_203353


namespace NUMINAMATH_CALUDE_problem_pyramid_volume_l2033_203300

/-- Triangular pyramid with given side lengths -/
structure TriangularPyramid where
  base_side : ℝ
  pa : ℝ
  pb : ℝ
  pc : ℝ

/-- Volume of a triangular pyramid -/
noncomputable def volume (p : TriangularPyramid) : ℝ :=
  sorry

/-- The specific triangular pyramid from the problem -/
def problem_pyramid : TriangularPyramid :=
  { base_side := 3
  , pa := 3
  , pb := 4
  , pc := 5 }

/-- Theorem stating that the volume of the problem pyramid is √11 -/
theorem problem_pyramid_volume :
  volume problem_pyramid = Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_pyramid_volume_l2033_203300


namespace NUMINAMATH_CALUDE_factor_expression_l2033_203313

theorem factor_expression (b : ℝ) : 294 * b^3 + 63 * b^2 - 21 * b = 21 * b * (14 * b^2 + 3 * b - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2033_203313


namespace NUMINAMATH_CALUDE_crow_votes_l2033_203388

/-- Represents the number of votes for each participant -/
structure Votes where
  rooster : ℕ
  crow : ℕ
  cuckoo : ℕ

/-- Represents Woodpecker's counts -/
structure WoodpeckerCounts where
  total : ℕ
  roosterAndCrow : ℕ
  crowAndCuckoo : ℕ
  cuckooAndRooster : ℕ

/-- The maximum error in Woodpecker's counts -/
def maxError : ℕ := 13

/-- Check if a number is within the error range of another number -/
def withinErrorRange (actual : ℕ) (counted : ℕ) : Prop :=
  (actual ≤ counted + maxError) ∧ (counted ≤ actual + maxError)

/-- The theorem to be proved -/
theorem crow_votes (v : Votes) (w : WoodpeckerCounts) 
  (h1 : withinErrorRange (v.rooster + v.crow + v.cuckoo) w.total)
  (h2 : withinErrorRange (v.rooster + v.crow) w.roosterAndCrow)
  (h3 : withinErrorRange (v.crow + v.cuckoo) w.crowAndCuckoo)
  (h4 : withinErrorRange (v.cuckoo + v.rooster) w.cuckooAndRooster)
  (h5 : w.total = 59)
  (h6 : w.roosterAndCrow = 15)
  (h7 : w.crowAndCuckoo = 18)
  (h8 : w.cuckooAndRooster = 20) :
  v.crow = 13 := by
  sorry

end NUMINAMATH_CALUDE_crow_votes_l2033_203388


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l2033_203333

theorem unique_four_digit_number :
  ∃! N : ℕ,
    N ≡ N^2 [ZMOD 10000] ∧
    N ≡ 7 [ZMOD 16] ∧
    1000 ≤ N ∧ N < 10000 ∧
    N = 3751 := by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l2033_203333


namespace NUMINAMATH_CALUDE_f_max_min_implies_a_range_l2033_203372

/-- The function f(x) defined in terms of a real parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Theorem stating that if f(x) has both a maximum and a minimum, then a is in the specified range -/
theorem f_max_min_implies_a_range (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ x, f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_implies_a_range_l2033_203372


namespace NUMINAMATH_CALUDE_tangent_and_inequality_l2033_203338

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 - x

theorem tangent_and_inequality (m n : ℝ) :
  (∃ (x : ℝ), x = 1 ∧ f m x = n) →  -- Point N(1, n) on the curve
  (∃ (x : ℝ), x = 1 ∧ (deriv (f m)) x = 1) →  -- Tangent with slope 1 (tan(π/4)) at x = 1
  (m = 2/3 ∧ n = -1/3) ∧  -- Part 1 of the theorem
  (∃ (k : ℕ), k = 2008 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-1) 3 → f m x ≤ k - 1993 ∧
    ∀ (k' : ℕ), k' < k → ∃ (x : ℝ), x ∈ Set.Icc (-1) 3 ∧ f m x > k' - 1993) :=
by sorry

end NUMINAMATH_CALUDE_tangent_and_inequality_l2033_203338


namespace NUMINAMATH_CALUDE_f_properties_l2033_203362

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (2 * ω * x - Real.pi / 6) + 2 * (Real.cos (ω * x))^2 - 1

def is_interval_of_increase (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_range (f : ℝ → ℝ) (S : Set ℝ) (R : Set ℝ) : Prop :=
  ∀ y ∈ R, ∃ x ∈ S, f x = y

theorem f_properties (ω : ℝ) (h : ω > 0) :
  (∀ k : ℤ, is_interval_of_increase (f 1) (-Real.pi/3 + k*Real.pi) (Real.pi/6 + k*Real.pi)) ∧
  (ω = 8/3 → is_range (f ω) (Set.Icc 0 (Real.pi/8)) (Set.Icc (1/2) 1)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2033_203362


namespace NUMINAMATH_CALUDE_factorization_mn_minus_mn_cubed_l2033_203309

theorem factorization_mn_minus_mn_cubed (m n : ℝ) : 
  m * n - m * n^3 = m * n * (1 + n) * (1 - n) := by sorry

end NUMINAMATH_CALUDE_factorization_mn_minus_mn_cubed_l2033_203309


namespace NUMINAMATH_CALUDE_simplify_sqrt_18_l2033_203382

theorem simplify_sqrt_18 : Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_18_l2033_203382


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2033_203369

/-- The sum of the infinite series Σ(n=1 to ∞) [(3n - 2) / (n(n + 1)(n + 3))] is equal to 11/12 -/
theorem infinite_series_sum : 
  ∑' (n : ℕ), (3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2033_203369


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l2033_203319

theorem quadratic_equation_condition (m : ℝ) : 
  (|m - 2| = 2 ∧ m - 4 ≠ 0) ↔ m = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l2033_203319


namespace NUMINAMATH_CALUDE_lettuce_calories_l2033_203367

/-- Calculates the calories in lettuce given the conditions of Jackson's meal -/
theorem lettuce_calories (
  pizza_crust : ℝ)
  (pizza_cheese : ℝ)
  (salad_dressing : ℝ)
  (total_calories_consumed : ℝ)
  (h1 : pizza_crust = 600)
  (h2 : pizza_cheese = 400)
  (h3 : salad_dressing = 210)
  (h4 : total_calories_consumed = 330) :
  let pizza_pepperoni := pizza_crust / 3
  let total_pizza := pizza_crust + pizza_pepperoni + pizza_cheese
  let pizza_consumed := total_pizza / 5
  let salad_consumed := total_calories_consumed - pizza_consumed
  let total_salad := salad_consumed * 4
  let lettuce := (total_salad - salad_dressing) / 3
  lettuce = 50 := by sorry

end NUMINAMATH_CALUDE_lettuce_calories_l2033_203367


namespace NUMINAMATH_CALUDE_disk_contains_origin_l2033_203341

theorem disk_contains_origin (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ a b c : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₄ > 0) (h₃ : y₁ > 0) (h₄ : y₂ > 0)
  (h₅ : x₂ < 0) (h₆ : x₃ < 0) (h₇ : y₃ < 0) (h₈ : y₄ < 0)
  (h₉ : (x₁ - a)^2 + (y₁ - b)^2 ≤ c^2)
  (h₁₀ : (x₂ - a)^2 + (y₂ - b)^2 ≤ c^2)
  (h₁₁ : (x₃ - a)^2 + (y₃ - b)^2 ≤ c^2)
  (h₁₂ : (x₄ - a)^2 + (y₄ - b)^2 ≤ c^2) :
  a^2 + b^2 ≤ c^2 := by
sorry

end NUMINAMATH_CALUDE_disk_contains_origin_l2033_203341


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l2033_203376

-- Define the set of points satisfying the inequalities
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > -1/2 * p.1 ∧ p.2 > 3 * p.1 + 6}

-- Define the quadrants
def quadrantI : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}
def quadrantII : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}
def quadrantIII : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 < 0 ∧ p.2 < 0}
def quadrantIV : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 < 0}

-- Theorem statement
theorem points_in_quadrants_I_and_II : 
  S ⊆ quadrantI ∪ quadrantII ∧ 
  S ∩ quadrantIII = ∅ ∧ 
  S ∩ quadrantIV = ∅ := by
  sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l2033_203376


namespace NUMINAMATH_CALUDE_tan_alpha_4_implies_fraction_9_l2033_203301

theorem tan_alpha_4_implies_fraction_9 (α : Real) (h : Real.tan α = 4) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_4_implies_fraction_9_l2033_203301


namespace NUMINAMATH_CALUDE_find_divisor_l2033_203325

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 919 →
  quotient = 17 →
  remainder = 11 →
  dividend = divisor * quotient + remainder →
  divisor = 53 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2033_203325


namespace NUMINAMATH_CALUDE_range_of_a_l2033_203363

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, 0 < a * x^2 - x + 1/(16*a)

def q (a : ℝ) : Prop := ∀ x : ℝ, x > 0 → Real.sqrt (2*x + 1) < 1 + a*x

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → (1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2033_203363


namespace NUMINAMATH_CALUDE_every_real_has_cube_root_l2033_203386

theorem every_real_has_cube_root : 
  ∀ y : ℝ, ∃ x : ℝ, x^3 = y := by sorry

end NUMINAMATH_CALUDE_every_real_has_cube_root_l2033_203386


namespace NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l2033_203345

-- Define a type for points in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define a function to check if four points are non-coplanar
def nonCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define a function to count the number of planes determined by four points
def countPlanes (p1 p2 p3 p4 : Point3D) : ℕ := sorry

-- Theorem statement
theorem four_noncoplanar_points_determine_four_planes 
  (p1 p2 p3 p4 : Point3D) 
  (h : nonCoplanar p1 p2 p3 p4) : 
  countPlanes p1 p2 p3 p4 = 4 := by sorry

end NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l2033_203345


namespace NUMINAMATH_CALUDE_line_equations_proof_l2033_203332

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) is on a given line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Checks if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Checks if two lines are perpendicular -/
def Line.isPerpendicularTo (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem line_equations_proof :
  let l1 : Line := { a := 3, b := -2, c := 1 }
  let l2 : Line := { a := 3, b := -2, c := 5 }
  let l3 : Line := { a := 3, b := -2, c := -5 }
  let l4 : Line := { a := 2, b := 3, c := 1 }
  (l1.containsPoint 1 2 ∧ l1.isParallelTo l2) ∧
  (l3.containsPoint 1 (-1) ∧ l3.isPerpendicularTo l4) := by sorry

end NUMINAMATH_CALUDE_line_equations_proof_l2033_203332


namespace NUMINAMATH_CALUDE_sandra_brought_twenty_pairs_l2033_203337

/-- Calculates the number of sock pairs Sandra brought given the initial conditions --/
def sandras_socks (initial_pairs : ℕ) (final_pairs : ℕ) : ℕ :=
  let moms_pairs := 3 * initial_pairs + 8
  let s := (final_pairs - initial_pairs - moms_pairs) * 5 / 6
  s

theorem sandra_brought_twenty_pairs :
  sandras_socks 12 80 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sandra_brought_twenty_pairs_l2033_203337


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2033_203314

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (5678 * 10 + N) % 6 = 0 → N ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2033_203314


namespace NUMINAMATH_CALUDE_other_communities_count_l2033_203344

theorem other_communities_count (total_boys : ℕ) 
  (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total_boys = 850 →
  muslim_percent = 34/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  ⌊(1 - (muslim_percent + hindu_percent + sikh_percent)) * total_boys⌋ = 238 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l2033_203344


namespace NUMINAMATH_CALUDE_x_plus_one_is_square_l2033_203346

def x : ℕ := (1 + 2) * (1 + 2^2) * (1 + 2^4) * (1 + 2^8) * (1 + 2^16) * (1 + 2^32) * (1 + 2^64) * (1 + 2^128) * (1 + 2^256)

theorem x_plus_one_is_square (x : ℕ := x) : x + 1 = 2^512 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_one_is_square_l2033_203346


namespace NUMINAMATH_CALUDE_loot_box_average_loss_l2033_203342

/-- Represents the loot box problem with given parameters --/
structure LootBoxProblem where
  cost_per_box : ℝ
  standard_item_value : ℝ
  rare_item_value : ℝ
  rare_item_probability : ℝ
  total_spent : ℝ

/-- Calculates the average loss per loot box --/
def average_loss (p : LootBoxProblem) : ℝ :=
  let standard_prob := 1 - p.rare_item_probability
  let expected_value := standard_prob * p.standard_item_value + p.rare_item_probability * p.rare_item_value
  p.cost_per_box - expected_value

/-- Theorem stating the average loss per loot box --/
theorem loot_box_average_loss :
  let p : LootBoxProblem := {
    cost_per_box := 5,
    standard_item_value := 3.5,
    rare_item_value := 15,
    rare_item_probability := 0.1,
    total_spent := 40
  }
  average_loss p = 0.35 := by sorry

end NUMINAMATH_CALUDE_loot_box_average_loss_l2033_203342


namespace NUMINAMATH_CALUDE_nina_spiders_count_l2033_203302

/-- Proves that Nina has 3 spiders given the conditions of the problem -/
theorem nina_spiders_count :
  ∀ (spiders : ℕ),
  (∃ (total_eyes : ℕ),
    total_eyes = 124 ∧
    total_eyes = 8 * spiders + 2 * 50) →
  spiders = 3 := by
sorry

end NUMINAMATH_CALUDE_nina_spiders_count_l2033_203302


namespace NUMINAMATH_CALUDE_exist_four_digit_square_sum_l2033_203384

/-- A four-digit number that is equal to the square of the sum of its first two digits and last two digits. -/
def IsFourDigitSquareSum (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ 
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 0 ≤ b ∧ b < 100 ∧
    n = 100 * a + b ∧ n = (a + b)^2

/-- There exist at least three distinct four-digit numbers that are equal to the square of the sum of their first two digits and last two digits. -/
theorem exist_four_digit_square_sum : 
  ∃ (n₁ n₂ n₃ : ℕ), n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₂ ≠ n₃ ∧ 
    IsFourDigitSquareSum n₁ ∧ IsFourDigitSquareSum n₂ ∧ IsFourDigitSquareSum n₃ := by
  sorry

end NUMINAMATH_CALUDE_exist_four_digit_square_sum_l2033_203384


namespace NUMINAMATH_CALUDE_largest_c_value_l2033_203351

theorem largest_c_value (c : ℝ) (h : (3*c + 4)*(c - 2) = 9*c) : 
  c ≤ 4 ∧ ∃ (c : ℝ), (3*c + 4)*(c - 2) = 9*c ∧ c = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_c_value_l2033_203351


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l2033_203399

noncomputable def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 8| + 1

theorem sum_of_max_min_g : 
  ∃ (max_g min_g : ℝ), 
    (∀ x ∈ Set.Icc 3 7, g x ≤ max_g) ∧
    (∃ x ∈ Set.Icc 3 7, g x = max_g) ∧
    (∀ x ∈ Set.Icc 3 7, min_g ≤ g x) ∧
    (∃ x ∈ Set.Icc 3 7, g x = min_g) ∧
    max_g + min_g = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l2033_203399


namespace NUMINAMATH_CALUDE_star_seven_three_l2033_203323

/-- Custom binary operation ∗ -/
def star (a b : ℤ) : ℤ := 4*a + 5*b - a*b

/-- Theorem stating that 7 ∗ 3 = 22 -/
theorem star_seven_three : star 7 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_star_seven_three_l2033_203323


namespace NUMINAMATH_CALUDE_tangent_equation_solution_l2033_203339

open Real

theorem tangent_equation_solution (x : ℝ) :
  (5.44 * tan (5 * x) - 2 * tan (3 * x) = tan (3 * x)^2 * tan (5 * x)) →
  (cos (3 * x) ≠ 0) →
  (cos (5 * x) ≠ 0) →
  ∃ k : ℤ, x = π * k := by
sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l2033_203339


namespace NUMINAMATH_CALUDE_banknote_problem_l2033_203391

/-- Represents the number of banknotes of each denomination -/
structure Banknotes where
  ten : ℕ
  twenty : ℕ
  fifty : ℕ

/-- The problem constraints -/
def valid_banknotes (b : Banknotes) : Prop :=
  b.ten > 0 ∧ b.twenty > 0 ∧ b.fifty > 0 ∧
  b.ten + b.twenty + b.fifty = 24 ∧
  10 * b.ten + 20 * b.twenty + 50 * b.fifty = 1000

theorem banknote_problem :
  ∃ (b : Banknotes), valid_banknotes b ∧ b.twenty = 4 :=
sorry

end NUMINAMATH_CALUDE_banknote_problem_l2033_203391


namespace NUMINAMATH_CALUDE_special_polynomial_is_x_squared_plus_one_l2033_203336

/-- A polynomial satisfying specific conditions -/
def SpecialPolynomial (p : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, p (x * y) = p x * p y - p x - p y + 2) ∧
  p 3 = 10 ∧
  p 4 = 17

/-- The theorem stating that the special polynomial is x^2 + 1 -/
theorem special_polynomial_is_x_squared_plus_one 
  (p : ℝ → ℝ) (hp : SpecialPolynomial p) :
  ∀ x : ℝ, p x = x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_is_x_squared_plus_one_l2033_203336


namespace NUMINAMATH_CALUDE_initial_bees_in_hive_l2033_203328

theorem initial_bees_in_hive (additional_bees : ℕ) (total_bees : ℕ) (h1 : additional_bees = 9) (h2 : total_bees = 25) :
  total_bees - additional_bees = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_bees_in_hive_l2033_203328


namespace NUMINAMATH_CALUDE_parabola_vertex_l2033_203305

/-- The parabola defined by y = (x-2)^2 + 4 has vertex at (2,4) -/
theorem parabola_vertex (x y : ℝ) :
  y = (x - 2)^2 + 4 → (2, 4) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2033_203305
