import Mathlib

namespace NUMINAMATH_CALUDE_meet_once_l1147_114772

/-- Represents the movement of Michael and the garbage truck --/
structure Movement where
  michaelSpeed : ℝ
  truckSpeed : ℝ
  pailDistance : ℝ
  truckStopTime : ℝ

/-- Calculates the number of times Michael and the truck meet --/
def meetingCount (m : Movement) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem meet_once (m : Movement) 
  (h1 : m.michaelSpeed = 6)
  (h2 : m.truckSpeed = 10)
  (h3 : m.pailDistance = 200)
  (h4 : m.truckStopTime = 40)
  (h5 : m.pailDistance = m.truckSpeed * (m.pailDistance / m.michaelSpeed - m.truckStopTime)) :
  meetingCount m = 1 := by
  sorry

#check meet_once

end NUMINAMATH_CALUDE_meet_once_l1147_114772


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1147_114739

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = Complex.abs (4 + 3*I)) :
  z.im = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1147_114739


namespace NUMINAMATH_CALUDE_cube_painting_problem_l1147_114703

theorem cube_painting_problem (n : ℕ) : 
  n > 0 →  -- Ensure n is positive
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 3 / 4 → 
  n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_painting_problem_l1147_114703


namespace NUMINAMATH_CALUDE_f_upper_bound_l1147_114748

theorem f_upper_bound (x : ℝ) (hx : x > 1) : Real.log x + Real.sqrt x - 1 < (3/2) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_f_upper_bound_l1147_114748


namespace NUMINAMATH_CALUDE_algebraic_expression_equals_one_l1147_114717

theorem algebraic_expression_equals_one
  (m n : ℝ)
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (h_diff : m - n = 1/2) :
  (m^2 - n^2) / (2*m^2 + 2*m*n) / (m - (2*m*n - n^2) / m) = 1 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_equals_one_l1147_114717


namespace NUMINAMATH_CALUDE_gerald_added_six_crayons_l1147_114796

/-- The number of crayons Gerald added to the box -/
def crayons_added (initial final : ℕ) : ℕ := final - initial

theorem gerald_added_six_crayons :
  let initial := 7
  let final := 13
  crayons_added initial final = 6 := by sorry

end NUMINAMATH_CALUDE_gerald_added_six_crayons_l1147_114796


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1147_114751

/-- A geometric sequence with a_3 = 2 and a_6 = 16 has a common ratio of 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n * (a 6 / a 3)^(1/3)) →  -- Geometric sequence property
  a 3 = 2 →                                     -- Given condition
  a 6 = 16 →                                    -- Given condition
  ∃ q : ℝ, (∀ n, a (n + 1) = a n * q) ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1147_114751


namespace NUMINAMATH_CALUDE_positive_solution_sum_l1147_114750

theorem positive_solution_sum (a b : ℕ+) (x : ℝ) : 
  x^2 + 10*x = 93 →
  x > 0 →
  x = Real.sqrt a - b →
  a + b = 123 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_sum_l1147_114750


namespace NUMINAMATH_CALUDE_xiaoning_score_is_87_l1147_114794

/-- The maximum score for a student's semester physical education comprehensive score. -/
def max_score : ℝ := 100

/-- The weight of the midterm exam score in the comprehensive score calculation. -/
def midterm_weight : ℝ := 0.3

/-- The weight of the final exam score in the comprehensive score calculation. -/
def final_weight : ℝ := 0.7

/-- Xiaoning's midterm exam score as a percentage. -/
def xiaoning_midterm : ℝ := 80

/-- Xiaoning's final exam score as a percentage. -/
def xiaoning_final : ℝ := 90

/-- Calculates the comprehensive score based on midterm and final exam scores and their weights. -/
def comprehensive_score (midterm : ℝ) (final : ℝ) : ℝ :=
  midterm * midterm_weight + final * final_weight

/-- Theorem stating that Xiaoning's physical education comprehensive score is 87 points. -/
theorem xiaoning_score_is_87 :
  comprehensive_score xiaoning_midterm xiaoning_final = 87 := by
  sorry

end NUMINAMATH_CALUDE_xiaoning_score_is_87_l1147_114794


namespace NUMINAMATH_CALUDE_range_of_m_l1147_114744

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x^2 / (2*m) - y^2 / (m-1) = 1 ∧ m > 1/3

def q (m : ℝ) : Prop := ∃ x y : ℝ, y^2 / 5 - x^2 / m = 1 ∧ 0 < m ∧ m < 15

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → 1/3 ≤ m ∧ m < 15 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1147_114744


namespace NUMINAMATH_CALUDE_dimes_in_tip_jar_l1147_114786

def nickel_value : ℚ := 0.05
def dime_value : ℚ := 0.10
def half_dollar_value : ℚ := 0.50

def shining_nickels : ℕ := 3
def shining_dimes : ℕ := 13
def tip_jar_half_dollars : ℕ := 9

def total_amount : ℚ := 6.65

theorem dimes_in_tip_jar :
  ∃ (tip_jar_dimes : ℕ),
    (shining_nickels * nickel_value + shining_dimes * dime_value +
     tip_jar_dimes * dime_value + tip_jar_half_dollars * half_dollar_value = total_amount) ∧
    tip_jar_dimes = 7 :=
by sorry

end NUMINAMATH_CALUDE_dimes_in_tip_jar_l1147_114786


namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l1147_114771

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2) 
  (h2 : Complex.abs (w^2 + z^2) = 8) : 
  Complex.abs (w^3 + z^3) = 20 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l1147_114771


namespace NUMINAMATH_CALUDE_gcd_values_for_special_m_l1147_114755

theorem gcd_values_for_special_m (m n : ℕ) (h : m + 6 = 9 * m) : 
  Nat.gcd m n = 3 ∨ Nat.gcd m n = 6 := by
sorry

end NUMINAMATH_CALUDE_gcd_values_for_special_m_l1147_114755


namespace NUMINAMATH_CALUDE_square_diagonal_length_l1147_114799

theorem square_diagonal_length (side_length : ℝ) (h : side_length = 30 * Real.sqrt 3) :
  Real.sqrt (2 * side_length ^ 2) = 30 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_length_l1147_114799


namespace NUMINAMATH_CALUDE_min_value_expression_l1147_114743

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^2 + b^2 + 2*a*b + 1 / (a + b)^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1147_114743


namespace NUMINAMATH_CALUDE_quadratic_sum_of_b_and_c_l1147_114713

/-- For the quadratic x^2 - 20x + 49, when written as (x+b)^2+c, b+c equals -61 -/
theorem quadratic_sum_of_b_and_c : ∃ b c : ℝ, 
  (∀ x : ℝ, x^2 - 20*x + 49 = (x+b)^2 + c) ∧ b + c = -61 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_b_and_c_l1147_114713


namespace NUMINAMATH_CALUDE_complex_equation_proof_l1147_114760

theorem complex_equation_proof (a b : ℝ) : (a - 2 * I) / I = b + I → a - b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l1147_114760


namespace NUMINAMATH_CALUDE_simplify_nested_radicals_l1147_114773

theorem simplify_nested_radicals : 
  Real.sqrt (8 + 4 * Real.sqrt 3) + Real.sqrt (8 - 4 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_radicals_l1147_114773


namespace NUMINAMATH_CALUDE_no_real_solutions_l1147_114792

theorem no_real_solutions : ¬∃ x : ℝ, (2*x - 6)^2 + 4 = -(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1147_114792


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l1147_114736

/-- A geometric sequence with the given first four terms -/
def geometric_sequence (y : ℝ) : ℕ → ℝ
  | 0 => 3
  | 1 => 9 * y
  | 2 => 27 * y^2
  | 3 => 81 * y^3
  | n + 4 => geometric_sequence y n * 3 * y

/-- The fifth term of the geometric sequence is 243y^4 -/
theorem fifth_term_of_sequence (y : ℝ) :
  geometric_sequence y 4 = 243 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l1147_114736


namespace NUMINAMATH_CALUDE_tangent_function_l1147_114791

/-- Given a function f(x) = ax / (x^2 + b), prove that if f(1) = 2 and f'(1) = 0, 
    then f(x) = 4x / (x^2 + 1) -/
theorem tangent_function (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x / (x^2 + b)
  (f 1 = 2) → (deriv f 1 = 0) → ∀ x, f x = 4 * x / (x^2 + 1) := by
sorry

end NUMINAMATH_CALUDE_tangent_function_l1147_114791


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1147_114701

theorem absolute_value_inequality (a : ℝ) : |a| ≠ -|-a| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1147_114701


namespace NUMINAMATH_CALUDE_no_single_solution_quadratic_inequality_l1147_114725

theorem no_single_solution_quadratic_inequality :
  ¬ ∃ (b : ℝ), ∃! (x : ℝ), |x^2 + 3*b*x + 4*b| ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_no_single_solution_quadratic_inequality_l1147_114725


namespace NUMINAMATH_CALUDE_tub_fill_time_is_24_minutes_l1147_114737

/-- Represents the tub filling problem -/
structure TubFilling where
  capacity : ℕ             -- Tub capacity in liters
  flow_rate : ℕ            -- Tap flow rate in liters per minute
  leak_rate : ℕ            -- Leak rate in liters per minute
  cycle_time : ℕ           -- Time for one on-off cycle in minutes

/-- Calculates the time needed to fill the tub -/
def fill_time (tf : TubFilling) : ℕ :=
  let net_gain_per_cycle := tf.flow_rate - tf.leak_rate * tf.cycle_time
  (tf.capacity + net_gain_per_cycle - 1) / net_gain_per_cycle * tf.cycle_time

/-- Theorem stating that the time to fill the tub is 24 minutes -/
theorem tub_fill_time_is_24_minutes :
  let tf : TubFilling := {
    capacity := 120,
    flow_rate := 12,
    leak_rate := 1,
    cycle_time := 2
  }
  fill_time tf = 24 := by sorry

end NUMINAMATH_CALUDE_tub_fill_time_is_24_minutes_l1147_114737


namespace NUMINAMATH_CALUDE_no_two_digit_product_equals_concatenation_l1147_114738

theorem no_two_digit_product_equals_concatenation : ¬∃ (a b c d : ℕ), 
  (0 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (0 ≤ d ∧ d ≤ 9) ∧ 
  ((10 * a + b) * (10 * c + d) = 1000 * a + 100 * b + 10 * c + d) :=
by sorry

end NUMINAMATH_CALUDE_no_two_digit_product_equals_concatenation_l1147_114738


namespace NUMINAMATH_CALUDE_coin_draw_probability_l1147_114759

def pennies : ℕ := 3
def nickels : ℕ := 5
def dimes : ℕ := 8
def total_coins : ℕ := pennies + nickels + dimes
def drawn_coins : ℕ := 8
def min_value : ℕ := 75

def successful_outcomes : ℕ := 321
def total_outcomes : ℕ := 12870

theorem coin_draw_probability :
  let prob := successful_outcomes / total_outcomes
  (∀ (outcome : Fin total_outcomes), 
    (outcome.val < successful_outcomes → 
      (∃ (p n d : ℕ), p + n + d = drawn_coins ∧ 
        p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧
        p * 1 + n * 5 + d * 10 ≥ min_value))) ∧
  (∀ (p n d : ℕ), p + n + d = drawn_coins → 
    p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧
    p * 1 + n * 5 + d * 10 ≥ min_value →
    (∃ (outcome : Fin total_outcomes), outcome.val < successful_outcomes)) →
  prob = successful_outcomes / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_coin_draw_probability_l1147_114759


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1147_114741

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + 7

-- Define the condition for the inequality
def inequality_condition (b : ℝ) : Prop :=
  ∀ x : ℝ, f b x < 0 ↔ (x < -2 ∨ x > 3)

-- Theorem statement
theorem quadratic_inequality_solution :
  ∃ b : ℝ, inequality_condition b ∧ b = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1147_114741


namespace NUMINAMATH_CALUDE_vertical_tangents_condition_l1147_114781

/-- The function f(x) = x(a - 1/e^x) has two distinct points with vertical tangents
    if and only if a is in the open interval (0, 2/e) -/
theorem vertical_tangents_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, (x * (a - Real.exp (-x))) = 0 → x = x₁ ∨ x = x₂) ∧
    (∀ x : ℝ, (a - (1 + x) * Real.exp (-x)) = 0 → x = x₁ ∨ x = x₂)) ↔
  (0 < a ∧ a < 2 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_vertical_tangents_condition_l1147_114781


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1147_114757

theorem complex_modulus_problem (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1147_114757


namespace NUMINAMATH_CALUDE_worker_a_time_l1147_114707

theorem worker_a_time (worker_b_time worker_ab_time worker_a_time : ℝ) : 
  worker_b_time = 12 →
  worker_ab_time = 5.454545454545454 →
  worker_a_time = 10.153846153846153 →
  (1 / worker_a_time + 1 / worker_b_time) * worker_ab_time = 1 :=
by sorry

end NUMINAMATH_CALUDE_worker_a_time_l1147_114707


namespace NUMINAMATH_CALUDE_part_one_part_two_l1147_114783

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| - |x + 1|

-- Theorem for part I
theorem part_one : 
  ∀ x : ℝ, f (-1/2) x ≤ -1 ↔ x ≥ 1/4 := by sorry

-- Theorem for part II
theorem part_two :
  (∃ a : ℝ, ∀ x : ℝ, f a x ≤ 2*a) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≤ 2*a) → a ≥ 1/3) ∧
  (∀ x : ℝ, f (1/3) x ≤ 2*(1/3)) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1147_114783


namespace NUMINAMATH_CALUDE_fourth_member_income_l1147_114767

/-- Proves that in a family of 4 members with a given average income and known incomes of 3 members, the income of the fourth member is as calculated. -/
theorem fourth_member_income
  (family_size : ℕ)
  (average_income : ℕ)
  (income1 income2 income3 : ℕ)
  (h1 : family_size = 4)
  (h2 : average_income = 10000)
  (h3 : income1 = 15000)
  (h4 : income2 = 6000)
  (h5 : income3 = 11000) :
  (family_size * average_income) - (income1 + income2 + income3) = 8000 := by
  sorry

#eval (4 * 10000) - (15000 + 6000 + 11000)

end NUMINAMATH_CALUDE_fourth_member_income_l1147_114767


namespace NUMINAMATH_CALUDE_initial_puppies_count_l1147_114700

/-- The number of puppies Alyssa had initially -/
def initial_puppies : ℕ := 7

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 5

/-- The number of puppies Alyssa has left -/
def puppies_left : ℕ := 2

/-- Theorem stating that the initial number of puppies equals the sum of puppies given away and puppies left -/
theorem initial_puppies_count : initial_puppies = puppies_given_away + puppies_left := by
  sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l1147_114700


namespace NUMINAMATH_CALUDE_distance_from_bogula_to_bolifoyn_l1147_114798

/-- The distance from Bogula to Bolifoyn in miles -/
def total_distance : ℝ := 10

/-- The time in hours at which they approach Pigtown -/
def time_to_pigtown : ℝ := 1

/-- The additional distance traveled after Pigtown in miles -/
def additional_distance : ℝ := 5

/-- The total travel time in hours -/
def total_time : ℝ := 4

theorem distance_from_bogula_to_bolifoyn :
  ∃ (distance_to_pigtown : ℝ),
    /- After 20 minutes (1/3 hour), half of the remaining distance to Pigtown is covered -/
    (1/3 * (total_distance / time_to_pigtown)) = distance_to_pigtown / 2 ∧
    /- The distance covered is twice less than the remaining distance to Pigtown -/
    (1/3 * (total_distance / time_to_pigtown)) = distance_to_pigtown / 3 ∧
    /- They took another 5 miles after approaching Pigtown -/
    total_distance = distance_to_pigtown + additional_distance ∧
    /- The total travel time is 4 hours -/
    total_time * (total_distance / total_time) = total_distance := by
  sorry


end NUMINAMATH_CALUDE_distance_from_bogula_to_bolifoyn_l1147_114798


namespace NUMINAMATH_CALUDE_election_vote_ratio_l1147_114793

theorem election_vote_ratio (marcy_votes barry_votes joey_votes : ℕ) : 
  marcy_votes = 66 →
  marcy_votes = 3 * barry_votes →
  joey_votes = 8 →
  barry_votes ≠ 0 →
  barry_votes / (joey_votes + 3) = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_election_vote_ratio_l1147_114793


namespace NUMINAMATH_CALUDE_quadratic_binomial_square_l1147_114753

theorem quadratic_binomial_square (b r : ℚ) : 
  (∀ x, b * x^2 + 20 * x + 16 = (r * x - 4)^2) → b = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_binomial_square_l1147_114753


namespace NUMINAMATH_CALUDE_distribute_balls_count_l1147_114715

/-- The number of ways to put 4 different balls into 4 different boxes, leaving exactly two boxes empty -/
def ways_to_distribute_balls : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem stating that the number of ways to distribute the balls is 84 -/
theorem distribute_balls_count : ways_to_distribute_balls = 84 := by
  sorry

end NUMINAMATH_CALUDE_distribute_balls_count_l1147_114715


namespace NUMINAMATH_CALUDE_negative_four_squared_times_negative_one_power_2022_l1147_114789

theorem negative_four_squared_times_negative_one_power_2022 :
  -4^2 * (-1)^2022 = -16 := by
  sorry

end NUMINAMATH_CALUDE_negative_four_squared_times_negative_one_power_2022_l1147_114789


namespace NUMINAMATH_CALUDE_smallest_zero_difference_l1147_114711

def u (n : ℕ) : ℤ := n^3 - n

def finite_difference (f : ℕ → ℤ) (k : ℕ) : ℕ → ℤ :=
  match k with
  | 0 => f
  | k+1 => λ n => finite_difference f k (n+1) - finite_difference f k n

theorem smallest_zero_difference :
  (∃ k : ℕ, ∀ n : ℕ, finite_difference u k n = 0) ∧
  (∀ k : ℕ, k < 4 → ∃ n : ℕ, finite_difference u k n ≠ 0) ∧
  (∀ n : ℕ, finite_difference u 4 n = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_zero_difference_l1147_114711


namespace NUMINAMATH_CALUDE_translation_result_l1147_114765

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point left by a given amount -/
def translateLeft (p : Point) (d : ℝ) : Point :=
  { x := p.x - d, y := p.y }

/-- Translate a point up by a given amount -/
def translateUp (p : Point) (d : ℝ) : Point :=
  { x := p.x, y := p.y + d }

/-- The initial point A -/
def A : Point :=
  { x := 2, y := 3 }

/-- The final point after translation -/
def finalPoint : Point :=
  translateUp (translateLeft A 3) 2

theorem translation_result :
  finalPoint = { x := -1, y := 5 } := by sorry

end NUMINAMATH_CALUDE_translation_result_l1147_114765


namespace NUMINAMATH_CALUDE_car_speed_calculation_l1147_114762

/-- Proves that a car's speed is 104 miles per hour given specific conditions -/
theorem car_speed_calculation (fuel_efficiency : ℝ) (fuel_consumed : ℝ) (time : ℝ)
  (h1 : fuel_efficiency = 64) -- km per liter
  (h2 : fuel_consumed = 3.9) -- gallons
  (h3 : time = 5.7) -- hours
  (h4 : (1 : ℝ) / 3.8 = 1 / 3.8) -- 1 gallon = 3.8 liters
  (h5 : (1 : ℝ) / 1.6 = 1 / 1.6) -- 1 mile = 1.6 kilometers
  : (fuel_efficiency * fuel_consumed * 3.8) / (time * 1.6) = 104 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_calculation_l1147_114762


namespace NUMINAMATH_CALUDE_inequality_range_l1147_114756

theorem inequality_range (t : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → 
    (1/8 * (2*t - t^2) ≤ x^2 - 3*x + 2 ∧ x^2 - 3*x + 2 ≤ 3 - t^2)) ↔ 
  (t ∈ Set.Icc (-1) (1 - Real.sqrt 3)) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l1147_114756


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l1147_114732

theorem sphere_hemisphere_volume_ratio (p : ℝ) (hp : p > 0) :
  (4 / 3 * Real.pi * p^3) / (1 / 2 * 4 / 3 * Real.pi * (2*p)^3) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l1147_114732


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1147_114729

theorem fraction_meaningful (x : ℝ) : 
  (x - 2) / (2 * x - 3) ≠ 0 ↔ x ≠ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1147_114729


namespace NUMINAMATH_CALUDE_theorem_60752_infinite_primes_4k_plus_1_l1147_114777

-- Theorem from problem 60752
theorem theorem_60752 (N : ℕ) (a : ℕ) (h : N = a^2 + 1) :
  ∃ p : ℕ, Prime p ∧ p ∣ N ∧ ∃ k : ℕ, p = 4 * k + 1 := sorry

theorem infinite_primes_4k_plus_1 :
  ∀ n : ℕ, ∃ p : ℕ, p > n ∧ Prime p ∧ ∃ k : ℕ, p = 4 * k + 1 := by sorry

end NUMINAMATH_CALUDE_theorem_60752_infinite_primes_4k_plus_1_l1147_114777


namespace NUMINAMATH_CALUDE_number_divided_by_002_equals_50_l1147_114749

theorem number_divided_by_002_equals_50 :
  ∃ x : ℝ, x / 0.02 = 50 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_002_equals_50_l1147_114749


namespace NUMINAMATH_CALUDE_dragon_castle_theorem_l1147_114704

/-- Represents the configuration of a dragon tethered to a cylindrical castle -/
structure DragonCastle where
  castle_radius : ℝ
  chain_length : ℝ
  chain_height : ℝ
  dragon_distance : ℝ

/-- Calculates the length of the chain touching the castle -/
def chain_on_castle (dc : DragonCastle) : ℝ :=
  sorry

/-- Theorem stating the properties of the dragon-castle configuration -/
theorem dragon_castle_theorem (dc : DragonCastle) 
  (h1 : dc.castle_radius = 10)
  (h2 : dc.chain_length = 30)
  (h3 : dc.chain_height = 6)
  (h4 : dc.dragon_distance = 6) :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.Prime c ∧
    chain_on_castle dc = (a - Real.sqrt b) / c ∧
    a = 90 ∧ b = 1440 ∧ c = 3 ∧
    a + b + c = 1533 :=
by sorry

end NUMINAMATH_CALUDE_dragon_castle_theorem_l1147_114704


namespace NUMINAMATH_CALUDE_interval_contains_integer_l1147_114747

theorem interval_contains_integer (a : ℝ) : 
  (∃ n : ℤ, 3*a < n ∧ n < 5*a - 2) ↔ (a > 1.2 ∧ a < 4/3) ∨ a > 1.4 :=
sorry

end NUMINAMATH_CALUDE_interval_contains_integer_l1147_114747


namespace NUMINAMATH_CALUDE_factorization_of_16x_squared_minus_1_l1147_114726

theorem factorization_of_16x_squared_minus_1 (x : ℝ) : 16 * x^2 - 1 = (4*x + 1) * (4*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_16x_squared_minus_1_l1147_114726


namespace NUMINAMATH_CALUDE_present_age_ratio_l1147_114719

/-- Given A's present age and the future ratio of ages, prove the present ratio of ages -/
theorem present_age_ratio (a b : ℕ) (h1 : a = 15) (h2 : (a + 6) * 5 = (b + 6) * 7) :
  5 * b = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_present_age_ratio_l1147_114719


namespace NUMINAMATH_CALUDE_max_tetrahedron_volume_l1147_114734

noncomputable def square_pyramid_volume (base_side : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_side^2 * height

theorem max_tetrahedron_volume
  (base_side : ℝ)
  (m_distance : ℝ)
  (h : base_side = 6)
  (d : m_distance = 10) :
  ∃ (max_vol : ℝ),
    max_vol = 24 ∧
    ∀ (vol : ℝ),
      ∃ (height : ℝ),
        vol = square_pyramid_volume base_side height →
        vol ≤ max_vol :=
by sorry

end NUMINAMATH_CALUDE_max_tetrahedron_volume_l1147_114734


namespace NUMINAMATH_CALUDE_ray_point_distance_product_l1147_114724

/-- Given two points on a ray from the origin, prove that the product of their distances
    equals the sum of the products of their coordinates. -/
theorem ray_point_distance_product (x₁ y₁ z₁ x₂ y₂ z₂ r₁ r₂ : ℝ) 
  (h₁ : r₁ = Real.sqrt (x₁^2 + y₁^2 + z₁^2))
  (h₂ : r₂ = Real.sqrt (x₂^2 + y₂^2 + z₂^2))
  (h_collinear : ∃ (t : ℝ), t > 0 ∧ x₂ = t * x₁ ∧ y₂ = t * y₁ ∧ z₂ = t * z₁) :
  r₁ * r₂ = x₁ * x₂ + y₁ * y₂ + z₁ * z₂ := by
  sorry

end NUMINAMATH_CALUDE_ray_point_distance_product_l1147_114724


namespace NUMINAMATH_CALUDE_function_evaluation_l1147_114758

theorem function_evaluation (f : ℝ → ℝ) 
  (h : ∀ x, f (x - 1) = x^2 + 1) : 
  f (-1) = 1 := by
sorry

end NUMINAMATH_CALUDE_function_evaluation_l1147_114758


namespace NUMINAMATH_CALUDE_total_toys_l1147_114770

theorem total_toys (mike_toys : ℕ) (annie_toys : ℕ) (tom_toys : ℕ) 
  (h1 : mike_toys = 6)
  (h2 : annie_toys = 3 * mike_toys)
  (h3 : tom_toys = annie_toys + 2) :
  mike_toys + annie_toys + tom_toys = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_l1147_114770


namespace NUMINAMATH_CALUDE_chair_circumference_l1147_114710

def parallelogram_circumference (side1 : ℝ) (side2 : ℝ) : ℝ :=
  2 * (side1 + side2)

theorem chair_circumference :
  let side1 := 18
  let side2 := 12
  parallelogram_circumference side1 side2 = 60 := by
sorry

end NUMINAMATH_CALUDE_chair_circumference_l1147_114710


namespace NUMINAMATH_CALUDE_kevin_wins_l1147_114705

/-- Represents a player in the chess game -/
inductive Player : Type
| Peter : Player
| Emma : Player
| Kevin : Player

/-- Represents the game results for each player -/
structure GameResults :=
  (wins : Player → ℕ)
  (losses : Player → ℕ)

/-- The theorem to prove -/
theorem kevin_wins (results : GameResults) : 
  results.wins Player.Peter = 4 →
  results.losses Player.Peter = 2 →
  results.wins Player.Emma = 3 →
  results.losses Player.Emma = 3 →
  results.losses Player.Kevin = 3 →
  results.wins Player.Kevin = 1 := by
  sorry


end NUMINAMATH_CALUDE_kevin_wins_l1147_114705


namespace NUMINAMATH_CALUDE_rectangle_area_relationship_l1147_114720

/-- Theorem: For a rectangle with area 4 and side lengths x and y, y = 4/x where x > 0 -/
theorem rectangle_area_relationship (x y : ℝ) (h1 : x > 0) (h2 : x * y = 4) : y = 4 / x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_relationship_l1147_114720


namespace NUMINAMATH_CALUDE_race_table_distance_l1147_114706

/-- Given a race with 11 equally spaced tables over 2100 meters, 
    the distance between the first and third table is 420 meters. -/
theorem race_table_distance (total_distance : ℝ) (num_tables : ℕ) :
  total_distance = 2100 →
  num_tables = 11 →
  (2 * (total_distance / (num_tables - 1))) = 420 := by
  sorry

end NUMINAMATH_CALUDE_race_table_distance_l1147_114706


namespace NUMINAMATH_CALUDE_expression_value_l1147_114764

theorem expression_value : 
  (45 + (23 / 89) * Real.sin (π / 6)) * (4 * (3 ^ 2) - 7 * ((-2) ^ 3)) = 4186 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1147_114764


namespace NUMINAMATH_CALUDE_remaining_salary_l1147_114709

theorem remaining_salary (salary : ℝ) (food_fraction house_rent_fraction clothes_fraction : ℝ) 
  (h1 : salary = 140000)
  (h2 : food_fraction = 1/5)
  (h3 : house_rent_fraction = 1/10)
  (h4 : clothes_fraction = 3/5)
  (h5 : food_fraction + house_rent_fraction + clothes_fraction < 1) :
  salary * (1 - (food_fraction + house_rent_fraction + clothes_fraction)) = 14000 :=
by sorry

end NUMINAMATH_CALUDE_remaining_salary_l1147_114709


namespace NUMINAMATH_CALUDE_toy_cost_l1147_114712

/-- Given Roger's initial amount, the cost of a game, and the number of toys he can buy,
    prove that each toy costs $7. -/
theorem toy_cost (initial_amount : ℕ) (game_cost : ℕ) (num_toys : ℕ) :
  initial_amount = 68 →
  game_cost = 47 →
  num_toys = 3 →
  (initial_amount - game_cost) / num_toys = 7 := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_l1147_114712


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1147_114746

theorem product_of_three_numbers (a b c : ℕ) : 
  a * b * c = 224 →
  a < b →
  b < c →
  2 * a = c →
  ∃ (x y z : ℕ), x * y * z = 224 ∧ 2 * x = z ∧ x < y ∧ y < z :=
by sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1147_114746


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l1147_114714

/-- The value of p for which the left focus of the hyperbola x²/3 - y² = 1
    is on the directrix of the parabola y² = 2px -/
theorem hyperbola_parabola_intersection (p : ℝ) : p = 4 := by
  -- Define the hyperbola equation
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 3 - y^2 = 1
  -- Define the parabola equation
  let parabola := fun (x y : ℝ) ↦ y^2 = 2 * p * x
  -- Define the condition that the left focus of the hyperbola is on the directrix of the parabola
  let focus_on_directrix := ∃ (x y : ℝ), hyperbola x y ∧ parabola x y
  sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l1147_114714


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l1147_114735

/-- Represents a right triangle with acute angles in the ratio 5:4 -/
structure RightTriangle where
  /-- First acute angle in degrees -/
  angle1 : ℝ
  /-- Second acute angle in degrees -/
  angle2 : ℝ
  /-- The triangle is a right triangle -/
  is_right_triangle : angle1 + angle2 = 90
  /-- The ratio of acute angles is 5:4 -/
  angle_ratio : angle1 / angle2 = 5 / 4

/-- Theorem: In a right triangle where the ratio of acute angles is 5:4,
    the measures of these angles are 50° and 40° -/
theorem right_triangle_acute_angles (t : RightTriangle) : 
  t.angle1 = 50 ∧ t.angle2 = 40 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_acute_angles_l1147_114735


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1147_114721

theorem complex_number_in_first_quadrant :
  let z : ℂ := Complex.I / (1 + Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1147_114721


namespace NUMINAMATH_CALUDE_fish_in_large_aquarium_l1147_114752

def fish_redistribution (initial_fish : ℕ) (additional_fish : ℕ) (small_aquarium_capacity : ℕ) : ℕ :=
  let total_fish := initial_fish + additional_fish
  total_fish - small_aquarium_capacity

theorem fish_in_large_aquarium :
  fish_redistribution 125 250 150 = 225 :=
by sorry

end NUMINAMATH_CALUDE_fish_in_large_aquarium_l1147_114752


namespace NUMINAMATH_CALUDE_enthalpy_change_reaction_l1147_114702

/-- Standard enthalpy of formation for Na₂O (s) in kJ/mol -/
def ΔH_f_Na2O : ℝ := -416

/-- Standard enthalpy of formation for H₂O (l) in kJ/mol -/
def ΔH_f_H2O : ℝ := -286

/-- Standard enthalpy of formation for NaOH (s) in kJ/mol -/
def ΔH_f_NaOH : ℝ := -427.8

/-- Standard enthalpy change of the reaction Na₂O + H₂O → 2NaOH at 298 K -/
def ΔH_reaction : ℝ := 2 * ΔH_f_NaOH - (ΔH_f_Na2O + ΔH_f_H2O)

theorem enthalpy_change_reaction :
  ΔH_reaction = -153.6 := by sorry

end NUMINAMATH_CALUDE_enthalpy_change_reaction_l1147_114702


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1147_114742

theorem yellow_balls_count (red_balls : ℕ) (yellow_balls : ℕ) 
  (h1 : red_balls = 1) 
  (h2 : (red_balls : ℚ) / (red_balls + yellow_balls) = 1 / 4) : 
  yellow_balls = 3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1147_114742


namespace NUMINAMATH_CALUDE_congruence_problem_l1147_114787

theorem congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 6 ∧ n ≡ 123456 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1147_114787


namespace NUMINAMATH_CALUDE_supermarket_spending_difference_l1147_114795

def initialAmount : ℕ := 53
def atmAmount : ℕ := 91
def remainingAmount : ℕ := 14

theorem supermarket_spending_difference : 
  (initialAmount + atmAmount - remainingAmount) - atmAmount = 39 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_difference_l1147_114795


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1147_114797

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 3 * (b + c))
  (second_eq : b = 5 * c) :
  a * b * c = 176 := by sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1147_114797


namespace NUMINAMATH_CALUDE_knights_in_february_l1147_114769

/-- Represents a city with knights and liars -/
structure City where
  inhabitants : ℕ
  knights_february : ℕ
  claims_february : ℕ
  claims_30th : ℕ

/-- The proposition that a city satisfies the given conditions -/
def satisfies_conditions (c : City) : Prop :=
  c.inhabitants = 366 ∧
  c.claims_february = 100 ∧
  c.claims_30th = 60 ∧
  c.knights_february ≤ 29

/-- The theorem stating that if a city satisfies the conditions, 
    then exactly 29 knights were born in February -/
theorem knights_in_february (c : City) :
  satisfies_conditions c → c.knights_february = 29 := by
  sorry

end NUMINAMATH_CALUDE_knights_in_february_l1147_114769


namespace NUMINAMATH_CALUDE_product_sum_and_reciprocals_geq_nine_l1147_114785

theorem product_sum_and_reciprocals_geq_nine (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_and_reciprocals_geq_nine_l1147_114785


namespace NUMINAMATH_CALUDE_three_distinct_zeroes_l1147_114774

/-- The piecewise function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then |2^x - 1| else 3 / (x - 1)

/-- The theorem stating the condition for three distinct zeroes -/
theorem three_distinct_zeroes (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = a ∧ f y = a ∧ f z = a) ↔ 0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_three_distinct_zeroes_l1147_114774


namespace NUMINAMATH_CALUDE_a_range_l1147_114727

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem a_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 a, f x ≤ 3) ∧ 
  (∃ x ∈ Set.Icc 0 a, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 a, f x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc 0 a, f x = 2) →
  a ∈ Set.Icc 1 2 :=
by sorry


end NUMINAMATH_CALUDE_a_range_l1147_114727


namespace NUMINAMATH_CALUDE_no_solution_when_m_equals_negative_one_l1147_114779

theorem no_solution_when_m_equals_negative_one :
  ∀ x : ℝ, (3 - 2*x) / (x - 3) + (2 + (-1)*x) / (3 - x) ≠ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_when_m_equals_negative_one_l1147_114779


namespace NUMINAMATH_CALUDE_books_per_week_after_second_l1147_114728

theorem books_per_week_after_second (total_books : ℕ) (first_week : ℕ) (second_week : ℕ) (total_weeks : ℕ) :
  total_books = 54 →
  first_week = 6 →
  second_week = 3 →
  total_weeks = 7 →
  (total_books - (first_week + second_week)) / (total_weeks - 2) = 9 :=
by sorry

end NUMINAMATH_CALUDE_books_per_week_after_second_l1147_114728


namespace NUMINAMATH_CALUDE_work_completion_time_l1147_114763

theorem work_completion_time (b_alone : ℝ) (together_time : ℝ) (b_remaining : ℝ) 
  (h1 : b_alone = 28)
  (h2 : together_time = 3)
  (h3 : b_remaining = 21) :
  ∃ a_alone : ℝ, 
    a_alone = 21 ∧ 
    together_time * (1 / a_alone + 1 / b_alone) + b_remaining * (1 / b_alone) = 1 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1147_114763


namespace NUMINAMATH_CALUDE_inequality_proof_l1147_114716

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (hx : x = Real.sqrt (a^2 + b^2)) (hy : y = Real.sqrt (c^2 + d^2)) :
  x * y ≥ a * c + b * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1147_114716


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l1147_114745

/-- Represents a 4x4x4 cube composed of smaller cubes -/
structure Cube4x4x4 where
  total_cubes : Nat
  face_size : Nat
  shaded_per_face : Nat

/-- Calculates the number of uniquely shaded cubes in a 4x4x4 cube -/
def count_shaded_cubes (cube : Cube4x4x4) : Nat :=
  sorry

/-- Theorem stating that 24 cubes are shaded on at least one face -/
theorem shaded_cubes_count (cube : Cube4x4x4) 
  (h1 : cube.total_cubes = 64)
  (h2 : cube.face_size = 4)
  (h3 : cube.shaded_per_face = 8) : 
  count_shaded_cubes cube = 24 := by
  sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l1147_114745


namespace NUMINAMATH_CALUDE_triangle_properties_l1147_114790

/-- Properties of a triangle ABC -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = t.b * Real.cos t.C + Real.sqrt 3 * t.c * Real.sin t.B)
  (h2 : t.b = 2)
  (h3 : t.a = Real.sqrt 3 * t.c) : 
  t.B = π / 6 ∧ t.a * t.c * Real.sin t.B / 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1147_114790


namespace NUMINAMATH_CALUDE_ab_product_l1147_114775

theorem ab_product (a b : ℚ) (h1 : 10 * a = 20) (h2 : 6 * b = 20) : 120 * a * b = 800 := by
  sorry

end NUMINAMATH_CALUDE_ab_product_l1147_114775


namespace NUMINAMATH_CALUDE_melanie_dimes_l1147_114733

/-- The number of dimes Melanie has after receiving dimes from her parents -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Theorem: Melanie has 19 dimes after receiving dimes from her parents -/
theorem melanie_dimes : total_dimes 7 8 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_l1147_114733


namespace NUMINAMATH_CALUDE_find_t_l1147_114788

-- Define variables
variable (t : ℝ)

-- Define functions for hours worked and hourly rates
def my_hours : ℝ := t + 2
def my_rate : ℝ := 4*t - 4
def bob_hours : ℝ := 4*t - 7
def bob_rate : ℝ := t + 3

-- State the theorem
theorem find_t : 
  my_hours * my_rate = bob_hours * bob_rate + 3 → t = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_t_l1147_114788


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1147_114776

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum1 : a 1 + a 2 = 4)
  (h_sum2 : a 2 + a 3 = 8) :
  a 7 = 13 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1147_114776


namespace NUMINAMATH_CALUDE_ticket_cost_theorem_l1147_114761

def adult_price : ℝ := 11
def child_price : ℝ := 8
def senior_price : ℝ := 9

def husband_discount : ℝ := 0.25
def parents_discount : ℝ := 0.15
def nephew_discount : ℝ := 0.10
def sister_discount : ℝ := 0.30

def num_adults : ℕ := 5
def num_children : ℕ := 4
def num_seniors : ℕ := 3

def total_cost : ℝ :=
  (adult_price * (1 - husband_discount) + adult_price) +  -- Mrs. Lopez and husband
  (senior_price * 2 * (1 - parents_discount)) +           -- Parents
  (child_price * 3 + child_price + adult_price * (1 - nephew_discount)) + -- Children and nephews
  senior_price +                                          -- Aunt (buy-one-get-one-free)
  (adult_price * 2) +                                     -- Two friends
  (adult_price * (1 - sister_discount))                   -- Sister

theorem ticket_cost_theorem : total_cost = 115.15 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_theorem_l1147_114761


namespace NUMINAMATH_CALUDE_prime_even_intersection_l1147_114784

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def P : Set ℕ := {n : ℕ | isPrime n}
def Q : Set ℕ := {n : ℕ | isEven n}

theorem prime_even_intersection : P ∩ Q = {2} := by sorry

end NUMINAMATH_CALUDE_prime_even_intersection_l1147_114784


namespace NUMINAMATH_CALUDE_oleg_event_guests_l1147_114708

theorem oleg_event_guests (total_guests men : ℕ) (h1 : total_guests = 80) (h2 : men = 40) :
  let women := men / 2
  let adults := men + women
  let original_children := total_guests - adults
  original_children + 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_oleg_event_guests_l1147_114708


namespace NUMINAMATH_CALUDE_midpoint_tetrahedron_volume_ratio_l1147_114766

/-- A regular tetrahedron in 3D space -/
structure RegularTetrahedron where
  -- We don't need to specify the vertices, just that it's a regular tetrahedron
  is_regular : Bool

/-- The tetrahedron formed by connecting the midpoints of the edges of a regular tetrahedron -/
def midpoint_tetrahedron (t : RegularTetrahedron) : RegularTetrahedron :=
  { is_regular := true }  -- The midpoint tetrahedron is also regular

/-- The volume of a tetrahedron -/
def volume (t : RegularTetrahedron) : ℝ :=
  sorry  -- We don't need to define this explicitly for the theorem

/-- 
  The ratio of the volume of the midpoint tetrahedron to the volume of the original tetrahedron
  is 1/8
-/
theorem midpoint_tetrahedron_volume_ratio (t : RegularTetrahedron) :
  volume (midpoint_tetrahedron t) / volume t = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_tetrahedron_volume_ratio_l1147_114766


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1147_114730

theorem linear_equation_solution (x m : ℝ) : 
  4 * x + 2 * m = 14 → x = 2 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1147_114730


namespace NUMINAMATH_CALUDE_xy_leq_half_sum_squares_l1147_114768

theorem xy_leq_half_sum_squares (x y : ℝ) : x * y ≤ (x^2 + y^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_leq_half_sum_squares_l1147_114768


namespace NUMINAMATH_CALUDE_train_time_lost_l1147_114723

/-- The problem of calculating the time lost by a train at stations -/
theorem train_time_lost (car_speed : ℝ) (distance : ℝ) (train_speed_factor : ℝ) : 
  car_speed = 120 →
  distance = 75 →
  train_speed_factor = 1.5 →
  let train_speed := car_speed * train_speed_factor
  let car_time := distance / car_speed
  let train_time_no_stops := distance / train_speed
  let time_lost := car_time - train_time_no_stops
  time_lost * 60 = 12.5 := by sorry

end NUMINAMATH_CALUDE_train_time_lost_l1147_114723


namespace NUMINAMATH_CALUDE_h_inverse_correct_l1147_114780

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := 4 * x + 5
def g (x : ℝ) : ℝ := 3 * x^2 - 2
def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of h
noncomputable def h_inv (x : ℝ) : ℝ := Real.sqrt ((x + 3) / 12)

-- Theorem statement
theorem h_inverse_correct (x : ℝ) : h (h_inv x) = x ∧ h_inv (h x) = x :=
  sorry

end NUMINAMATH_CALUDE_h_inverse_correct_l1147_114780


namespace NUMINAMATH_CALUDE_score_difference_l1147_114754

/-- Represents the runs scored by each batsman -/
structure BatsmanScores where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- Theorem stating the difference between A's and E's scores -/
theorem score_difference (scores : BatsmanScores) : scores.a - scores.e = 8 :=
  by
  have h1 : scores.a + scores.b + scores.c + scores.d + scores.e = 180 :=
    sorry -- Average score is 36, so total is 5 * 36 = 180
  have h2 : scores.d = scores.e + 5 := sorry -- D scored 5 more than E
  have h3 : scores.e = 20 := sorry -- E scored 20 runs
  have h4 : scores.b = scores.d + scores.e := sorry -- B scored as many as D and E combined
  have h5 : scores.b + scores.c = 107 := sorry -- B and C scored 107 between them
  have h6 : scores.e < scores.a := sorry -- E scored fewer runs than A
  sorry -- Prove that scores.a - scores.e = 8

end NUMINAMATH_CALUDE_score_difference_l1147_114754


namespace NUMINAMATH_CALUDE_car_average_speed_prove_car_average_speed_l1147_114722

/-- The average speed of a car given specific speed conditions -/
theorem car_average_speed : ℝ → Prop :=
  fun v : ℝ =>
    let half_distance := 1 / 2
    let faster_speed := v + 30
    let slower_speed := 0.7 * v
    let total_time := half_distance / faster_speed + half_distance / slower_speed
    v = 1 / total_time → v = 40

/-- Proof of the car's average speed -/
theorem prove_car_average_speed : ∃ v : ℝ, car_average_speed v :=
  sorry

end NUMINAMATH_CALUDE_car_average_speed_prove_car_average_speed_l1147_114722


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1147_114782

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x) / Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1147_114782


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l1147_114778

/-- An arithmetic progression with its sum sequence -/
structure ArithmeticProgression where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_progression_problem (seq : ArithmeticProgression) 
    (h1 : seq.a 1 + (seq.a 2)^2 = -3)
    (h2 : seq.S 5 = 10) :
  seq.a 9 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_problem_l1147_114778


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1147_114740

theorem negation_of_universal_proposition :
  (¬∀ x : ℝ, x ≥ 1 → Real.log x > 0) ↔ (∃ x : ℝ, x ≥ 1 ∧ Real.log x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1147_114740


namespace NUMINAMATH_CALUDE_bob_water_percentage_approx_36_percent_l1147_114718

/-- Represents a farmer with their crop acreages -/
structure Farmer where
  corn : ℝ
  cotton : ℝ
  beans : ℝ

/-- Calculates the total water usage for a farmer given water rates -/
def waterUsage (f : Farmer) (cornRate : ℝ) (cottonRate : ℝ) : ℝ :=
  f.corn * cornRate + f.cotton * cottonRate + f.beans * (2 * cornRate)

/-- The main theorem to prove -/
theorem bob_water_percentage_approx_36_percent 
  (bob : Farmer) 
  (brenda : Farmer)
  (bernie : Farmer)
  (cornRate : ℝ)
  (cottonRate : ℝ)
  (h_bob : bob = { corn := 3, cotton := 9, beans := 12 })
  (h_brenda : brenda = { corn := 6, cotton := 7, beans := 14 })
  (h_bernie : bernie = { corn := 2, cotton := 12, beans := 0 })
  (h_cornRate : cornRate = 20)
  (h_cottonRate : cottonRate = 80) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |waterUsage bob cornRate cottonRate / 
   (waterUsage bob cornRate cottonRate + 
    waterUsage brenda cornRate cottonRate + 
    waterUsage bernie cornRate cottonRate) - 0.36| < ε := by
  sorry

end NUMINAMATH_CALUDE_bob_water_percentage_approx_36_percent_l1147_114718


namespace NUMINAMATH_CALUDE_one_dollar_bills_count_l1147_114731

/-- Represents the wallet contents -/
structure Wallet where
  ones : ℕ
  twos : ℕ
  fives : ℕ

/-- The wallet satisfies the given conditions -/
def satisfies_conditions (w : Wallet) : Prop :=
  w.ones + w.twos + w.fives = 55 ∧
  w.ones * 1 + w.twos * 2 + w.fives * 5 = 126

/-- The theorem stating the number of one-dollar bills -/
theorem one_dollar_bills_count :
  ∃ (w : Wallet), satisfies_conditions w ∧ w.ones = 18 := by
  sorry

end NUMINAMATH_CALUDE_one_dollar_bills_count_l1147_114731
