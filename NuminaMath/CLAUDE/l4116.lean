import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l4116_411641

theorem equation_solution :
  let f : ℝ → ℝ := λ x => 2 * (x - 2)^2 - (6 - 3*x)
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 1/2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4116_411641


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l4116_411688

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) : 
  (∃ r : ℝ, r ≠ 0 ∧ a = 20 * r ∧ 5/4 = a * r) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l4116_411688


namespace NUMINAMATH_CALUDE_fraction_scaling_l4116_411681

theorem fraction_scaling (x y : ℝ) : 
  (5*x - 5*(5*y)) / ((5*x)^2 + (5*y)^2) = (1/5) * ((x - 5*y) / (x^2 + y^2)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_scaling_l4116_411681


namespace NUMINAMATH_CALUDE_min_questions_required_l4116_411612

/-- Represents the color of a ball -/
inductive Color
| White
| Black

/-- Represents a box containing a ball -/
structure Box where
  ball : Color

/-- Represents the state of the boxes -/
structure BoxState where
  boxes : Vector Box 2004
  white_count : Nat
  white_count_even : Even white_count

/-- Represents a question about two boxes -/
structure Question where
  box1 : Fin 2004
  box2 : Fin 2004
  box1_ne_box2 : box1 ≠ box2

/-- The result of asking a question -/
def ask_question (state : BoxState) (q : Question) : Bool :=
  match state.boxes[q.box1].ball, state.boxes[q.box2].ball with
  | Color.White, _ => true
  | _, Color.White => true
  | _, _ => false

/-- A strategy for asking questions -/
def Strategy := Nat → Question

/-- Checks if a strategy is successful for a given state -/
def strategy_successful (state : BoxState) (strategy : Strategy) : Prop :=
  ∃ n : Nat, ∃ i j : Fin 2004,
    i ≠ j ∧
    state.boxes[i].ball = Color.White ∧
    state.boxes[j].ball = Color.White ∧
    (∀ k < n, ask_question state (strategy k) = true)

/-- The main theorem stating the minimum number of questions required -/
theorem min_questions_required :
  ∀ (strategy : Strategy),
  (∀ state : BoxState, strategy_successful state strategy) →
  (∃ n : Nat, ∀ k, strategy k = strategy n → k ≥ 4005) :=
sorry

end NUMINAMATH_CALUDE_min_questions_required_l4116_411612


namespace NUMINAMATH_CALUDE_eleven_items_division_l4116_411658

theorem eleven_items_division (n : ℕ) (h : n = 11) : 
  (Finset.sum (Finset.range 3) (λ k => Nat.choose n (k + 3))) = 957 := by
  sorry

end NUMINAMATH_CALUDE_eleven_items_division_l4116_411658


namespace NUMINAMATH_CALUDE_product_scaling_l4116_411659

theorem product_scaling (a b c : ℕ) (ha : a = 268) (hb : b = 74) (hc : c = 19832) 
  (h : a * b = c) : (2.68 : ℝ) * 0.74 = 1.9832 := by
  sorry

end NUMINAMATH_CALUDE_product_scaling_l4116_411659


namespace NUMINAMATH_CALUDE_planes_through_skew_line_l4116_411601

/-- A structure representing a 3D space with lines and planes -/
structure Space3D where
  Line : Type
  Plane : Type
  in_plane : Line → Plane → Prop
  parallel : Plane → Line → Prop
  perpendicular : Plane → Plane → Prop
  skew : Line → Line → Prop

/-- The theorem statement -/
theorem planes_through_skew_line (S : Space3D) 
  (l m : S.Line) (α : S.Plane) 
  (h1 : S.skew l m) 
  (h2 : S.in_plane l α) : 
  (∃ (P : S.Plane), S.parallel P l ∧ ∃ (x : S.Line), S.in_plane x P ∧ x = m) ∧ 
  (∃ (Q : S.Plane), S.perpendicular Q α ∧ ∃ (y : S.Line), S.in_plane y Q ∧ y = m) := by
  sorry

end NUMINAMATH_CALUDE_planes_through_skew_line_l4116_411601


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_l4116_411602

theorem count_four_digit_numbers :
  let first_four_digit : Nat := 1000
  let last_four_digit : Nat := 9999
  (last_four_digit - first_four_digit + 1 : Nat) = 9000 := by
sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_l4116_411602


namespace NUMINAMATH_CALUDE_tim_stored_26_bales_l4116_411600

/-- The number of bales Tim stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Proof that Tim stored 26 bales in the barn -/
theorem tim_stored_26_bales (initial_bales final_bales : ℕ)
  (h1 : initial_bales = 28)
  (h2 : final_bales = 54) :
  bales_stored initial_bales final_bales = 26 := by
  sorry

end NUMINAMATH_CALUDE_tim_stored_26_bales_l4116_411600


namespace NUMINAMATH_CALUDE_f_of_two_equals_five_l4116_411674

/-- Given a function f(x) = x^2 + 2x - 3, prove that f(2) = 5 -/
theorem f_of_two_equals_five (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x - 3) : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_five_l4116_411674


namespace NUMINAMATH_CALUDE_range_of_f_l4116_411622

def f (x : ℝ) : ℝ := (x - 2)^2 - 1

theorem range_of_f :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3,
  ∃ y ∈ Set.Icc (-1 : ℝ) 8,
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-1 : ℝ) 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l4116_411622


namespace NUMINAMATH_CALUDE_phi_11_0_decomposition_l4116_411683

/-- The Φ₁₁⁰ series -/
def phi_11_0 : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 1
| 3 => 2
| 4 => 3
| 5 => 5
| 6 => 8
| 7 => 2
| 8 => 10
| 9 => 1
| n + 10 => phi_11_0 n

/-- The decomposed series -/
def c (n : ℕ) : ℚ := 3 * 8^n + 8 * 4^n

/-- Predicate to check if a sequence is an 11-arithmetic Fibonacci series -/
def is_11_arithmetic_fibonacci (f : ℕ → ℚ) : Prop :=
  ∀ n, f (n + 11) = f (n + 10) + f (n + 9)

/-- Predicate to check if a sequence is a geometric progression -/
def is_geometric_progression (f : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n, f (n + 1) = r * f n

theorem phi_11_0_decomposition :
  (∃ f g : ℕ → ℚ, 
    (∀ n, c n = f n + g n) ∧
    is_11_arithmetic_fibonacci f ∧
    is_11_arithmetic_fibonacci g ∧
    is_geometric_progression f ∧
    is_geometric_progression g) ∧
  (∀ n, (phi_11_0 n : ℚ) = c n) :=
sorry

end NUMINAMATH_CALUDE_phi_11_0_decomposition_l4116_411683


namespace NUMINAMATH_CALUDE_freight_train_speed_proof_l4116_411678

-- Define the total distance between points A and B
def total_distance : ℝ := 460

-- Define the time it takes for the trains to meet
def meeting_time : ℝ := 2

-- Define the speed of the passenger train
def passenger_train_speed : ℝ := 120

-- Define the speed of the freight train (to be proven)
def freight_train_speed : ℝ := 110

-- Theorem statement
theorem freight_train_speed_proof :
  total_distance = (passenger_train_speed + freight_train_speed) * meeting_time :=
by sorry

end NUMINAMATH_CALUDE_freight_train_speed_proof_l4116_411678


namespace NUMINAMATH_CALUDE_arrangement_exists_for_P_23_l4116_411695

/-- Fibonacci-like sequence defined by F_0 = 0, F_1 = 1, F_i = 3F_{i-1} - F_{i-2} for i ≥ 2 -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of an arrangement satisfying the given conditions for P = 23 -/
theorem arrangement_exists_for_P_23 : F 12 % 23 = 0 := by sorry

end NUMINAMATH_CALUDE_arrangement_exists_for_P_23_l4116_411695


namespace NUMINAMATH_CALUDE_borrowing_lending_period_l4116_411650

theorem borrowing_lending_period (principal : ℝ) (borrowing_rate : ℝ) (lending_rate : ℝ) (gain_per_year : ℝ) :
  principal = 9000 ∧ 
  borrowing_rate = 0.04 ∧ 
  lending_rate = 0.06 ∧ 
  gain_per_year = 180 → 
  (gain_per_year / (principal * (lending_rate - borrowing_rate))) = 1 := by
sorry

end NUMINAMATH_CALUDE_borrowing_lending_period_l4116_411650


namespace NUMINAMATH_CALUDE_infinitely_many_sqrt_eight_eight_eight_l4116_411662

theorem infinitely_many_sqrt_eight_eight_eight (k : ℕ) : 
  (9 * k - 1 + 0.888 : ℝ) < Real.sqrt (81 * k^2 - 2 * k) ∧ 
  Real.sqrt (81 * k^2 - 2 * k) < (9 * k - 1 + 0.889 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_sqrt_eight_eight_eight_l4116_411662


namespace NUMINAMATH_CALUDE_martin_trip_distance_is_1185_l4116_411651

/-- Calculates the total distance traveled during Martin's business trip --/
def martin_trip_distance : ℝ :=
  let segment1 := 70 * 3
  let segment2 := 80 * 4
  let segment3 := 65 * 3
  let segment4 := 50 * 2
  let segment5 := 90 * 4
  segment1 + segment2 + segment3 + segment4 + segment5

/-- Theorem stating that Martin's total trip distance is 1185 km --/
theorem martin_trip_distance_is_1185 :
  martin_trip_distance = 1185 := by sorry

end NUMINAMATH_CALUDE_martin_trip_distance_is_1185_l4116_411651


namespace NUMINAMATH_CALUDE_alicia_satisfaction_l4116_411619

/-- Represents the satisfaction equation for Alicia's activities --/
def satisfaction (reading : ℝ) (painting : ℝ) : ℝ := reading * painting

/-- Represents the constraint that t should be positive and less than 4 --/
def valid_t (t : ℝ) : Prop := 0 < t ∧ t < 4

theorem alicia_satisfaction (t : ℝ) : 
  valid_t t →
  satisfaction (12 - t) t = satisfaction (2*t + 2) (4 - t) →
  t = 2 :=
by sorry

end NUMINAMATH_CALUDE_alicia_satisfaction_l4116_411619


namespace NUMINAMATH_CALUDE_solve_sandwich_cost_l4116_411630

def sandwich_cost_problem (total_cost soda_cost : ℚ) : Prop :=
  let num_sandwiches : ℕ := 2
  let num_sodas : ℕ := 4
  let sandwich_cost : ℚ := (total_cost - num_sodas * soda_cost) / num_sandwiches
  total_cost = 838/100 ∧ soda_cost = 87/100 → sandwich_cost = 245/100

theorem solve_sandwich_cost : 
  sandwich_cost_problem (838/100) (87/100) := by
  sorry

end NUMINAMATH_CALUDE_solve_sandwich_cost_l4116_411630


namespace NUMINAMATH_CALUDE_multiplication_value_l4116_411682

theorem multiplication_value : 
  let original_number : ℝ := 6.5
  let divisor : ℝ := 6
  let result : ℝ := 13
  let multiplication_factor : ℝ := 12
  (original_number / divisor) * multiplication_factor = result := by
sorry

end NUMINAMATH_CALUDE_multiplication_value_l4116_411682


namespace NUMINAMATH_CALUDE_value_of_z_l4116_411675

theorem value_of_z (z : ℝ) : 
  (Real.sqrt 1.1) / (Real.sqrt 0.81) + (Real.sqrt z) / (Real.sqrt 0.49) = 2.879628878919216 → 
  z = 1.44 := by
sorry

end NUMINAMATH_CALUDE_value_of_z_l4116_411675


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l4116_411607

/-- Given a rectangular grid and two unshaded shapes within it, calculate the area of the shaded region. -/
theorem shaded_area_calculation (grid_width grid_height : ℝ)
  (triangle_base triangle_height : ℝ)
  (trapezoid_height trapezoid_top_base trapezoid_bottom_base : ℝ) :
  grid_width = 10 ∧ 
  grid_height = 5 ∧ 
  triangle_base = 3 ∧ 
  triangle_height = 2 ∧ 
  trapezoid_height = 3 ∧ 
  trapezoid_top_base = 3 ∧ 
  trapezoid_bottom_base = 6 →
  grid_width * grid_height - 
  (1/2 * triangle_base * triangle_height) - 
  (1/2 * (trapezoid_top_base + trapezoid_bottom_base) * trapezoid_height) = 33.5 := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_calculation_l4116_411607


namespace NUMINAMATH_CALUDE_exactly_one_from_each_class_passing_at_least_one_student_passing_l4116_411691

-- Define the probability of a student passing
def p_pass : ℝ := 0.6

-- Define the number of students from each class
def n_students : ℕ := 2

-- Define the probability of exactly one student from a class passing
def p_one_pass : ℝ := n_students * p_pass * (1 - p_pass)

-- Theorem for the first question
theorem exactly_one_from_each_class_passing : 
  p_one_pass * p_one_pass = 0.2304 := by sorry

-- Theorem for the second question
theorem at_least_one_student_passing : 
  1 - (1 - p_pass)^(2 * n_students) = 0.9744 := by sorry

end NUMINAMATH_CALUDE_exactly_one_from_each_class_passing_at_least_one_student_passing_l4116_411691


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4116_411603

/-- Given a line 2ax - by + 2 = 0 passing through the center of the circle (x + 1)^2 + (y - 2)^2 = 4,
    where a > 0 and b > 0, the minimum value of 1/a + 1/b is 4 -/
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : 2 * a * (-1) - b * 2 + 2 = 0) : 
  (∀ x y, (x + 1)^2 + (y - 2)^2 = 4 → 2 * a * x - b * y + 2 = 0) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 2 * a' * (-1) - b' * 2 + 2 = 0 → 1 / a' + 1 / b' ≥ 1 / a + 1 / b) →
  1 / a + 1 / b = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4116_411603


namespace NUMINAMATH_CALUDE_root_in_interval_l4116_411644

theorem root_in_interval : ∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2^x = 2 - x := by sorry

end NUMINAMATH_CALUDE_root_in_interval_l4116_411644


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l4116_411677

theorem quadratic_is_square_of_binomial (r : ℝ) (hr : r ≠ 0) :
  ∃ (p q : ℝ), ∀ x, r^2 * x^2 - 20 * x + 100 / r^2 = (p * x + q)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l4116_411677


namespace NUMINAMATH_CALUDE_complex_modulus_l4116_411697

theorem complex_modulus (z : ℂ) (h : z * Complex.I = -2 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l4116_411697


namespace NUMINAMATH_CALUDE_largest_factorial_divisor_l4116_411655

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem largest_factorial_divisor :
  ∀ m : ℕ, m > 98 → ¬(factorial m ∣ factorial 100 + factorial 99 + factorial 98) ∧
  (factorial 98 ∣ factorial 100 + factorial 99 + factorial 98) := by
  sorry

end NUMINAMATH_CALUDE_largest_factorial_divisor_l4116_411655


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l4116_411620

-- Define the lines
def line1 (x y : ℝ) : Prop := x + y + 1 = 0
def line2 (x y : ℝ) : Prop := 2*x - y + 8 = 0
def line3 (a x y : ℝ) : Prop := a*x + 3*y - 5 = 0

-- Define the set of possible a values
def possible_a_values : Set ℝ := {-3, -6}

-- State the theorem
theorem intersection_points_theorem (a : ℝ) :
  (∀ x y z : ℝ, (line1 x y ∧ line2 x y ∧ line3 a x y) → 
    (∃ w v : ℝ, (line1 w v ∧ line2 w v ∧ line3 a w v) ∧ (x ≠ w ∨ y ≠ v)) → 
    (∀ u t : ℝ, (line1 u t ∧ line2 u t ∧ line3 a u t) → 
      ((x = u ∧ y = t) ∨ (w = u ∧ v = t)))) →
  a ∈ possible_a_values :=
sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l4116_411620


namespace NUMINAMATH_CALUDE_initial_train_distance_l4116_411669

/-- Calculates the initial distance between two trains given their lengths, speeds, and time to meet. -/
theorem initial_train_distance
  (length1 : ℝ)
  (length2 : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (time : ℝ)
  (h1 : length1 = 100)
  (h2 : length2 = 200)
  (h3 : speed1 = 54)
  (h4 : speed2 = 72)
  (h5 : time = 1.999840012798976) :
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  let distance_covered := relative_speed * time * 3600
  distance_covered - (length1 + length2) = 251680.84161264498 :=
by sorry

end NUMINAMATH_CALUDE_initial_train_distance_l4116_411669


namespace NUMINAMATH_CALUDE_allowance_calculation_l4116_411680

/-- Represents John's weekly allowance in dollars -/
def weekly_allowance : ℝ := 2.10

/-- The fraction of allowance spent at the arcade -/
def arcade_fraction : ℚ := 3/5

/-- The fraction of remaining allowance spent at the toy store -/
def toy_store_fraction : ℚ := 2/7

/-- The fraction of remaining allowance spent at the bookstore -/
def bookstore_fraction : ℚ := 1/3

/-- The amount spent at the candy store in dollars -/
def candy_store_amount : ℝ := 0.40

/-- Theorem stating that given the spending pattern, the initial allowance was $2.10 -/
theorem allowance_calculation (A : ℝ) :
  A * (1 - arcade_fraction) * (1 - toy_store_fraction) * (1 - bookstore_fraction) = candy_store_amount →
  A = weekly_allowance := by
  sorry

#check allowance_calculation

end NUMINAMATH_CALUDE_allowance_calculation_l4116_411680


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_true_l4116_411618

theorem quadratic_inequality_always_true :
  ∀ x : ℝ, 3 * x^2 + 9 * x ≥ -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_true_l4116_411618


namespace NUMINAMATH_CALUDE_amy_earnings_l4116_411654

theorem amy_earnings (hourly_wage : ℝ) (hours_worked : ℝ) (tips : ℝ) :
  hourly_wage = 2 → hours_worked = 7 → tips = 9 →
  hourly_wage * hours_worked + tips = 23 := by
  sorry

end NUMINAMATH_CALUDE_amy_earnings_l4116_411654


namespace NUMINAMATH_CALUDE_some_number_value_l4116_411684

theorem some_number_value (x y z w N : ℝ) 
  (h1 : 4 * x * z + y * w = N)
  (h2 : x * w + y * z = 6)
  (h3 : (2 * x + y) * (2 * z + w) = 15) :
  N = 3 := by sorry

end NUMINAMATH_CALUDE_some_number_value_l4116_411684


namespace NUMINAMATH_CALUDE_gcd_of_sums_of_squares_l4116_411627

theorem gcd_of_sums_of_squares : Nat.gcd (130^2 + 240^2 + 350^2) (131^2 + 241^2 + 351^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_sums_of_squares_l4116_411627


namespace NUMINAMATH_CALUDE_roster_adjustment_count_l4116_411672

/-- Represents the number of class officers -/
def num_officers : ℕ := 5

/-- Represents the number of days in the duty roster -/
def num_days : ℕ := 5

/-- The number of ways to arrange the original Monday and Friday officers -/
def arrange_mon_fri : ℕ := 6

/-- The number of ways to choose an officer for each of Tuesday, Wednesday, and Thursday -/
def arrange_tue_thu : ℕ := 2

/-- The number of ways to arrange the remaining two officers for each of Tuesday, Wednesday, and Thursday -/
def arrange_remaining : ℕ := 2

/-- Theorem stating the total number of ways to adjust the roster -/
theorem roster_adjustment_count :
  (arrange_mon_fri * arrange_tue_thu * arrange_remaining) = 24 :=
sorry

end NUMINAMATH_CALUDE_roster_adjustment_count_l4116_411672


namespace NUMINAMATH_CALUDE_least_number_with_remainder_four_ninety_four_satisfies_conditions_ninety_four_is_least_number_l4116_411670

theorem least_number_with_remainder_four (n : ℕ) : 
  (n % 5 = 4 ∧ n % 6 = 4 ∧ n % 9 = 4 ∧ n % 18 = 4) → n ≥ 94 :=
by sorry

theorem ninety_four_satisfies_conditions : 
  94 % 5 = 4 ∧ 94 % 6 = 4 ∧ 94 % 9 = 4 ∧ 94 % 18 = 4 :=
by sorry

theorem ninety_four_is_least_number : 
  ∀ n : ℕ, (n % 5 = 4 ∧ n % 6 = 4 ∧ n % 9 = 4 ∧ n % 18 = 4) → n ≥ 94 :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_four_ninety_four_satisfies_conditions_ninety_four_is_least_number_l4116_411670


namespace NUMINAMATH_CALUDE_inequality_proof_l4116_411685

theorem inequality_proof (x y : ℝ) : x + y + Real.sqrt (x * y) ≤ 3 * (x + y - Real.sqrt (x * y)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4116_411685


namespace NUMINAMATH_CALUDE_reduced_rate_fraction_l4116_411690

/-- Represents the fraction of a day with reduced rates -/
def weekdayReducedRateFraction : ℚ := 12 / 24

/-- Represents the fraction of a day with reduced rates on weekends -/
def weekendReducedRateFraction : ℚ := 1

/-- Represents the number of weekdays in a week -/
def weekdaysPerWeek : ℕ := 5

/-- Represents the number of weekend days in a week -/
def weekendDaysPerWeek : ℕ := 2

/-- Represents the total number of days in a week -/
def daysPerWeek : ℕ := 7

/-- Theorem stating that the fraction of a week with reduced rates is 9/14 -/
theorem reduced_rate_fraction :
  (weekdayReducedRateFraction * weekdaysPerWeek + weekendReducedRateFraction * weekendDaysPerWeek) / daysPerWeek = 9 / 14 := by
  sorry


end NUMINAMATH_CALUDE_reduced_rate_fraction_l4116_411690


namespace NUMINAMATH_CALUDE_range_of_m_range_of_x_l4116_411617

-- Define the function f
def f (m x : ℝ) : ℝ := m * x^2 - m * x - 6 + m

-- Part 1
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x < 0) → m < 6/7 :=
sorry

-- Part 2
theorem range_of_x (x : ℝ) :
  (∀ m ∈ Set.Icc (-2) 2, f m x < 0) → -1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_x_l4116_411617


namespace NUMINAMATH_CALUDE_wire_cutting_l4116_411613

theorem wire_cutting (total_length : ℝ) (difference : ℝ) (longer_part : ℝ) : 
  total_length = 180 ∧ difference = 32 → longer_part = 106 :=
by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l4116_411613


namespace NUMINAMATH_CALUDE_sequence_sum_l4116_411640

-- Define the sequence type
def Sequence := Fin 10 → ℝ

-- Define the property of consecutive terms summing to 20
def ConsecutiveSum (s : Sequence) : Prop :=
  ∀ i : Fin 8, s i + s (i + 1) + s (i + 2) = 20

-- Define the theorem
theorem sequence_sum (s : Sequence) 
  (h1 : ConsecutiveSum s) 
  (h2 : s 4 = 8) : 
  s 0 + s 9 = 8 := by
  sorry


end NUMINAMATH_CALUDE_sequence_sum_l4116_411640


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l4116_411694

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The given condition for the sequence -/
def sequence_condition (a : ℕ → ℤ) : Prop :=
  a 2 - a 3 - a 7 - a 11 - a 13 + a 16 = 8

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : sequence_condition a) :
  a 9 = -4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l4116_411694


namespace NUMINAMATH_CALUDE_abs_sum_inequality_iff_l4116_411637

theorem abs_sum_inequality_iff (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 1| > a) ↔ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_iff_l4116_411637


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l4116_411610

theorem absolute_value_equation_solution :
  let f : ℝ → ℝ := λ x => |2*x + 4|
  let g : ℝ → ℝ := λ x => 1 - 3*x + x^2
  let solution1 : ℝ := (5 + Real.sqrt 37) / 2
  let solution2 : ℝ := (5 - Real.sqrt 37) / 2
  (∀ x : ℝ, f x = g x ↔ x = solution1 ∨ x = solution2) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l4116_411610


namespace NUMINAMATH_CALUDE_staircase_perimeter_l4116_411699

/-- Given a rectangle with a staircase-shaped region removed, 
    if the remaining area is 104 square feet, 
    then the perimeter of the remaining region is 52.4 feet. -/
theorem staircase_perimeter (width height : ℝ) (area remaining_area : ℝ) : 
  width = 10 →
  area = width * height →
  remaining_area = area - 40 →
  remaining_area = 104 →
  width + height + 3 + 5 + 20 = 52.4 :=
by sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l4116_411699


namespace NUMINAMATH_CALUDE_sum_of_81_and_15_l4116_411624

theorem sum_of_81_and_15 : 81 + 15 = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_81_and_15_l4116_411624


namespace NUMINAMATH_CALUDE_original_equals_scientific_l4116_411605

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 1570000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 1.57
    exponent := 9
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l4116_411605


namespace NUMINAMATH_CALUDE_refrigerator_price_l4116_411666

theorem refrigerator_price (refrigerator washing_machine : ℕ) 
  (h1 : washing_machine = refrigerator - 1490)
  (h2 : refrigerator + washing_machine = 7060) : 
  refrigerator = 4275 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_price_l4116_411666


namespace NUMINAMATH_CALUDE_expression_evaluation_l4116_411614

theorem expression_evaluation :
  let a : ℚ := -1
  let b : ℚ := 1/7
  (3*a^3 - 2*a*b + b^2) - 2*(-a^3 - a*b + 4*b^2) = -5 - 1/7 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4116_411614


namespace NUMINAMATH_CALUDE_binomial_not_divisible_by_prime_l4116_411647

theorem binomial_not_divisible_by_prime (p : ℕ) (n : ℕ) : 
  Prime p → 
  (∀ m : ℕ, m ≤ n → ¬(p ∣ Nat.choose n m)) ↔ 
  ∃ k s : ℕ, n = s * p^k - 1 ∧ 1 ≤ s ∧ s ≤ p :=
by sorry

end NUMINAMATH_CALUDE_binomial_not_divisible_by_prime_l4116_411647


namespace NUMINAMATH_CALUDE_max_baggies_count_l4116_411652

def cookies_per_bag : ℕ := 3
def chocolate_chip_cookies : ℕ := 2
def oatmeal_cookies : ℕ := 16

theorem max_baggies_count : 
  (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_bag = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_baggies_count_l4116_411652


namespace NUMINAMATH_CALUDE_a_is_most_suitable_l4116_411626

-- Define the participants
inductive Participant
  | A
  | B
  | C
  | D

-- Define the variance for each participant
def variance (p : Participant) : ℝ :=
  match p with
  | Participant.A => 0.15
  | Participant.B => 0.2
  | Participant.C => 0.4
  | Participant.D => 0.35

-- Define the function to find the most suitable participant
def most_suitable : Participant :=
  Participant.A

-- Theorem to prove A is the most suitable
theorem a_is_most_suitable :
  ∀ p : Participant, variance most_suitable ≤ variance p :=
by sorry

end NUMINAMATH_CALUDE_a_is_most_suitable_l4116_411626


namespace NUMINAMATH_CALUDE_equation_equivalent_to_circles_l4116_411663

def equation (x y : ℝ) : Prop :=
  x^4 - 16*x^2 + 2*x^2*y^2 - 16*y^2 + y^4 = 4*x^3 + 4*x*y^2 - 64*x

def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 16

def circle2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

theorem equation_equivalent_to_circles :
  ∀ x y : ℝ, equation x y ↔ (circle1 x y ∨ circle2 x y) :=
sorry

end NUMINAMATH_CALUDE_equation_equivalent_to_circles_l4116_411663


namespace NUMINAMATH_CALUDE_passing_game_properties_l4116_411693

/-- Represents a fair six-sided die --/
def Die := Fin 6

/-- The sum of n rolls of a fair six-sided die --/
def rollSum (n : ℕ) : Finset ℕ := sorry

/-- The condition for passing level n --/
def passCondition (n : ℕ) : Prop := ∀ x ∈ rollSum n, x > 2^n

/-- The probability of passing level n --/
def passProbability (n : ℕ) : ℚ := sorry

theorem passing_game_properties :
  (∀ n : ℕ, n ≥ 5 → ¬ passCondition n) ∧
  (passProbability 2 > passProbability 1) := by sorry

end NUMINAMATH_CALUDE_passing_game_properties_l4116_411693


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l4116_411636

/-- Given two circles with centers at (0, 0) and (17, 0) and radii 3 and 8 respectively,
    the x-coordinate of the point where a line tangent to both circles intersects the x-axis
    (to the right of the origin) is equal to 51/11. -/
theorem tangent_line_intersection (x : ℝ) : x > 0 →
  (x^2 = 3^2 + x^2) ∧ ((17 - x)^2 = 8^2 + x^2) → x = 51 / 11 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l4116_411636


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_210_462_l4116_411621

theorem lcm_gcf_ratio_210_462 : 
  (Nat.lcm 210 462) / (Nat.gcd 210 462) = 55 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_210_462_l4116_411621


namespace NUMINAMATH_CALUDE_original_avg_age_l4116_411615

def original_class_size : ℕ := 8
def new_students_size : ℕ := 8
def new_students_avg_age : ℕ := 32
def avg_age_decrease : ℕ := 4

theorem original_avg_age (original_avg : ℕ) :
  (original_avg * original_class_size + new_students_avg_age * new_students_size) / 
  (original_class_size + new_students_size) = original_avg - avg_age_decrease →
  original_avg = 40 :=
by sorry

end NUMINAMATH_CALUDE_original_avg_age_l4116_411615


namespace NUMINAMATH_CALUDE_grandfather_grandson_ages_l4116_411608

theorem grandfather_grandson_ages :
  ∀ (x y a b : ℕ),
    x > 70 →
    x - a = 10 * (y - a) →
    x + b = 8 * (y + b) →
    x = 71 ∧ y = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_grandfather_grandson_ages_l4116_411608


namespace NUMINAMATH_CALUDE_tourist_contact_probability_l4116_411665

/-- The probability that at least one tourist from the first group can contact at least one tourist from the second group -/
def contact_probability (p : ℝ) : ℝ := 1 - (1 - p)^42

/-- Theorem stating the probability of contact between two groups of tourists -/
theorem tourist_contact_probability (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  contact_probability p = 
  1 - (1 - p)^(6 * 7) :=
by sorry

end NUMINAMATH_CALUDE_tourist_contact_probability_l4116_411665


namespace NUMINAMATH_CALUDE_equation_implies_conditions_l4116_411638

theorem equation_implies_conditions (x y z w : ℝ) 
  (h : (2*x + y) / (y + z) = (z + w) / (w + 2*x)) :
  x = z/2 ∨ 2*x + y + z + w = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_implies_conditions_l4116_411638


namespace NUMINAMATH_CALUDE_fourth_year_area_l4116_411676

def initial_area : ℝ := 10000
def annual_increase : ℝ := 0.2

def area_after_n_years (n : ℕ) : ℝ :=
  initial_area * (1 + annual_increase) ^ n

theorem fourth_year_area :
  area_after_n_years 3 = 17280 :=
by sorry

end NUMINAMATH_CALUDE_fourth_year_area_l4116_411676


namespace NUMINAMATH_CALUDE_mary_has_fifty_cards_l4116_411673

/-- The number of Pokemon cards Mary has after receiving new cards from Sam -/
def marys_final_cards (initial_cards torn_cards new_cards : ℕ) : ℕ :=
  initial_cards - torn_cards + new_cards

/-- Theorem stating that Mary has 50 Pokemon cards after the given scenario -/
theorem mary_has_fifty_cards :
  marys_final_cards 33 6 23 = 50 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_fifty_cards_l4116_411673


namespace NUMINAMATH_CALUDE_trig_expression_value_l4116_411633

theorem trig_expression_value (α : Real) 
  (h1 : π/2 < α) 
  (h2 : α < π) 
  (h3 : Real.sin α + Real.cos α = 1/5) : 
  2 / (Real.cos α - Real.sin α) = -10/7 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l4116_411633


namespace NUMINAMATH_CALUDE_three_fish_thrown_back_l4116_411643

/-- Represents the number of fish caught by each family member and the total number of filets --/
structure FishingTrip where
  ben : Nat
  judy : Nat
  billy : Nat
  jim : Nat
  susie : Nat
  total_filets : Nat

/-- Calculates the number of fish thrown back given a fishing trip --/
def fish_thrown_back (trip : FishingTrip) : Nat :=
  let total_caught := trip.ben + trip.judy + trip.billy + trip.jim + trip.susie
  let kept := trip.total_filets / 2
  total_caught - kept

/-- Theorem stating that for the given fishing trip, 3 fish were thrown back --/
theorem three_fish_thrown_back : 
  let trip := FishingTrip.mk 4 1 3 2 5 24
  fish_thrown_back trip = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_fish_thrown_back_l4116_411643


namespace NUMINAMATH_CALUDE_bird_count_l4116_411692

theorem bird_count : 
  let cardinals : ℕ := 3
  let robins : ℕ := 4 * cardinals
  let blue_jays : ℕ := 2 * cardinals
  let sparrows : ℕ := 3 * cardinals + 1
  cardinals + robins + blue_jays + sparrows = 31 := by
  sorry

end NUMINAMATH_CALUDE_bird_count_l4116_411692


namespace NUMINAMATH_CALUDE_point_translation_l4116_411628

def initial_point : ℝ × ℝ := (-2, 3)

def translate_down (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - units)

def translate_right (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1 + units, p.2)

theorem point_translation :
  (translate_right (translate_down initial_point 3) 1) = (-1, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_translation_l4116_411628


namespace NUMINAMATH_CALUDE_binomial_ratio_sum_l4116_411623

theorem binomial_ratio_sum (n k : ℕ+) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k+1) : ℚ) = 2/3 ∧ 
  (Nat.choose n (k+1) : ℚ) / (Nat.choose n (k+2) : ℚ) = 3/4 ∧
  (∀ m l : ℕ+, (Nat.choose m l : ℚ) / (Nat.choose m (l+1) : ℚ) = 2/3 ∧ 
               (Nat.choose m (l+1) : ℚ) / (Nat.choose m (l+2) : ℚ) = 3/4 → 
               m = n ∧ l = k) →
  n + k = 47 := by
sorry

end NUMINAMATH_CALUDE_binomial_ratio_sum_l4116_411623


namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l4116_411629

/-- The x-coordinate of a point on the parabola y^2 = 6x that is twice as far from the focus as from the y-axis -/
theorem parabola_point_x_coordinate 
  (x y : ℝ) 
  (h1 : y^2 = 6*x) -- Point is on the parabola y^2 = 6x
  (h2 : (x - 3/2)^2 + y^2 = 4 * x^2) -- Distance to focus is twice distance to y-axis
  : x = 3/2 := by sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l4116_411629


namespace NUMINAMATH_CALUDE_rectangle_length_decrease_l4116_411625

theorem rectangle_length_decrease (b : ℝ) (x : ℝ) : 
  2 * b = 33.333333333333336 →
  (2 * b - x) * (b + 4) = 2 * b^2 + 75 →
  x = 2.833333333333336 := by sorry

end NUMINAMATH_CALUDE_rectangle_length_decrease_l4116_411625


namespace NUMINAMATH_CALUDE_smallest_common_factor_l4116_411631

theorem smallest_common_factor (n : ℕ) : 
  (∀ k < 165, k > 0 → Nat.gcd (11*k - 8) (5*k + 9) = 1) ∧ 
  Nat.gcd (11*165 - 8) (5*165 + 9) > 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l4116_411631


namespace NUMINAMATH_CALUDE_common_factor_of_polynomial_l4116_411648

/-- The common factor of a polynomial 4x(m-n) + 2y(m-n)^2 is 2(m-n) -/
theorem common_factor_of_polynomial (x y m n : ℤ) :
  ∃ (k : ℤ), (4*x*(m-n) + 2*y*(m-n)^2) = 2*(m-n) * k :=
by sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomial_l4116_411648


namespace NUMINAMATH_CALUDE_fraction_equality_l4116_411657

theorem fraction_equality (x y : ℝ) 
  (h : (1/3)^2 + (1/4)^2 / ((1/5)^2 + (1/6)^2) = 37*x / (73*y)) : 
  Real.sqrt x / Real.sqrt y = 75 * Real.sqrt 73 / (6 * Real.sqrt 61 * Real.sqrt 37) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4116_411657


namespace NUMINAMATH_CALUDE_units_digit_of_sum_factorials_1000_l4116_411609

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_of_sum_factorials_1000 :
  sum_of_factorials 1000 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_factorials_1000_l4116_411609


namespace NUMINAMATH_CALUDE_incorrect_analysis_is_B_l4116_411632

/-- Represents the content of the novel -/
def novel_content : String := "..."

/-- Represents the four analysis options -/
inductive Analysis
| A : Analysis
| B : Analysis
| C : Analysis
| D : Analysis

/-- Checks if an analysis is correct based on the novel content -/
def is_correct_analysis (a : Analysis) (content : String) : Prop :=
  match a with
  | Analysis.A => True  -- Assumed correct for this problem
  | Analysis.B => False -- Known to be incorrect
  | Analysis.C => True  -- Assumed correct for this problem
  | Analysis.D => True  -- Assumed correct for this problem

/-- The main theorem stating that option B is the incorrect analysis -/
theorem incorrect_analysis_is_B (content : String) :
  ∃ (a : Analysis), ¬(is_correct_analysis a content) ∧ a = Analysis.B :=
by
  sorry


end NUMINAMATH_CALUDE_incorrect_analysis_is_B_l4116_411632


namespace NUMINAMATH_CALUDE_fifth_month_sale_is_13562_l4116_411635

/-- The sale amount in the fifth month given the conditions of the problem -/
def fifth_month_sale (first_month : ℕ) (second_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (sixth_month : ℕ) (average : ℕ) : ℕ :=
  average * 6 - (first_month + second_month + third_month + fourth_month + sixth_month)

/-- Theorem stating that the fifth month sale is 13562 given the problem conditions -/
theorem fifth_month_sale_is_13562 :
  fifth_month_sale 6435 6927 6855 7230 5591 6600 = 13562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_is_13562_l4116_411635


namespace NUMINAMATH_CALUDE_max_cashback_selection_l4116_411649

structure Category where
  name : String
  cashback : Float
  expenses : Float

def calculate_cashback (c : Category) : Float :=
  c.cashback * c.expenses / 100

def total_cashback (categories : List Category) : Float :=
  categories.map calculate_cashback |> List.sum

theorem max_cashback_selection (categories : List Category) :
  let transport := { name := "Transport", cashback := 5, expenses := 2000 }
  let groceries := { name := "Groceries", cashback := 3, expenses := 5000 }
  let clothing := { name := "Clothing", cashback := 4, expenses := 3000 }
  let entertainment := { name := "Entertainment", cashback := 5, expenses := 3000 }
  let sports := { name := "Sports", cashback := 6, expenses := 1500 }
  let all_categories := [transport, groceries, clothing, entertainment, sports]
  let best_selection := [groceries, entertainment, clothing]
  categories = all_categories →
  (∀ selection : List Category,
    selection.length ≤ 3 →
    selection ⊆ categories →
    total_cashback selection ≤ total_cashback best_selection) :=
by sorry

end NUMINAMATH_CALUDE_max_cashback_selection_l4116_411649


namespace NUMINAMATH_CALUDE_certain_number_proof_l4116_411646

theorem certain_number_proof (x : ℝ) (h : 5 * x - 28 = 232) : x = 52 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l4116_411646


namespace NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l4116_411689

theorem quadratic_discriminant_nonnegative (x : ℤ) :
  x^2 * (49 - 40*x^2) ≥ 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l4116_411689


namespace NUMINAMATH_CALUDE_remainder_3_pow_123_plus_4_mod_8_l4116_411645

theorem remainder_3_pow_123_plus_4_mod_8 : 3^123 + 4 ≡ 7 [MOD 8] := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_123_plus_4_mod_8_l4116_411645


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l4116_411671

def is_prime (n : ℕ) : Prop := sorry

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ k, k ≥ start ∧ k < start + count → ¬ is_prime k

theorem smallest_prime_after_seven_nonprimes :
  ∃ n : ℕ, 
    (consecutive_nonprimes n 7) ∧ 
    (is_prime (n + 7)) ∧
    (∀ m : ℕ, m < n → ¬(consecutive_nonprimes m 7 ∧ is_prime (m + 7))) ∧
    (n + 7 = 97) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l4116_411671


namespace NUMINAMATH_CALUDE_triangle_x_theorem_l4116_411616

/-- A function that checks if three side lengths can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of positive integer values of x for which a triangle with sides 5, 12, and x^2 exists -/
def triangle_x_values : Set ℕ+ :=
  {x : ℕ+ | is_triangle 5 12 (x.val ^ 2)}

theorem triangle_x_theorem : triangle_x_values = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_triangle_x_theorem_l4116_411616


namespace NUMINAMATH_CALUDE_tax_percentage_proof_l4116_411642

def tax_problem (net_income : ℝ) (gross_income : ℝ) (untaxed_amount : ℝ) : Prop :=
  let taxable_income := gross_income - untaxed_amount
  let tax_rate := (gross_income - net_income) / taxable_income
  tax_rate = 0.10

theorem tax_percentage_proof :
  tax_problem 12000 13000 3000 := by
  sorry

end NUMINAMATH_CALUDE_tax_percentage_proof_l4116_411642


namespace NUMINAMATH_CALUDE_units_digit_difference_l4116_411661

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_difference (p : ℕ) 
  (h1 : p % 2 = 0) 
  (h2 : units_digit p > 0) 
  (h3 : units_digit (p + 2) = 8) : 
  units_digit (p^3) - units_digit (p^2) = 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_difference_l4116_411661


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_polynomial_l4116_411668

/-- Given a cubic polynomial P(x) = ax³ + bx² + cx + d satisfying P(x² + x) ≥ P(x + 1) for all real x,
    prove that the sum of its roots is -b/a. -/
theorem sum_of_roots_cubic_polynomial 
  (a b c d : ℝ) 
  (ha : a ≠ 0) 
  (h_inequality : ∀ x : ℝ, a*(x^2 + x)^3 + b*(x^2 + x)^2 + c*(x^2 + x) + d ≥ 
                           a*(x + 1)^3 + b*(x + 1)^2 + c*(x + 1) + d) : 
  (sum_of_roots : ℝ) → sum_of_roots = -b/a :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_polynomial_l4116_411668


namespace NUMINAMATH_CALUDE_train_seats_theorem_l4116_411664

/-- The total number of seats on the train -/
def total_seats : ℕ := 180

/-- The number of seats in Standard Class -/
def standard_seats : ℕ := 36

/-- The fraction of total seats in Comfort Class -/
def comfort_fraction : ℚ := 1/5

/-- The fraction of total seats in Premium Class -/
def premium_fraction : ℚ := 3/5

/-- Theorem stating that the total number of seats is 180 -/
theorem train_seats_theorem :
  (standard_seats : ℚ) + comfort_fraction * total_seats + premium_fraction * total_seats = total_seats := by
  sorry

end NUMINAMATH_CALUDE_train_seats_theorem_l4116_411664


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l4116_411696

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l4116_411696


namespace NUMINAMATH_CALUDE_opposite_to_A_is_E_l4116_411667

/-- Represents the labels on the cube faces -/
inductive Label
| A | B | C | D | E | F

/-- Represents a cube formed from six squares -/
structure Cube where
  faces : Fin 6 → Label

/-- Defines the property of being opposite faces on the cube -/
def are_opposite (c : Cube) (l1 l2 : Label) : Prop := sorry

/-- Defines the property of being adjacent faces on the cube -/
def are_adjacent (c : Cube) (l1 l2 : Label) : Prop := sorry

/-- Theorem: In a cube formed by folding six sequentially labeled squares A, B, C, D, E, F, 
    the face opposite to A is labeled E -/
theorem opposite_to_A_is_E (c : Cube) : 
  (c.faces 0 = Label.A) → 
  (c.faces 1 = Label.B) → 
  (c.faces 2 = Label.C) → 
  (c.faces 3 = Label.D) → 
  (c.faces 4 = Label.E) → 
  (c.faces 5 = Label.F) → 
  are_adjacent c Label.A Label.B →
  are_adjacent c Label.A Label.F →
  are_opposite c Label.A Label.E := by sorry

end NUMINAMATH_CALUDE_opposite_to_A_is_E_l4116_411667


namespace NUMINAMATH_CALUDE_gunther_typing_capacity_l4116_411653

/-- Given Gunther's typing speed and work day length, prove the number of words he can type in a day --/
theorem gunther_typing_capacity (words_per_set : ℕ) (minutes_per_set : ℕ) (minutes_per_day : ℕ) 
  (h1 : words_per_set = 160)
  (h2 : minutes_per_set = 3)
  (h3 : minutes_per_day = 480) :
  (minutes_per_day / minutes_per_set) * words_per_set = 25600 := by
  sorry

#eval (480 / 3) * 160

end NUMINAMATH_CALUDE_gunther_typing_capacity_l4116_411653


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l4116_411634

theorem trigonometric_simplification (A : ℝ) (h : 0 < A ∧ A < π / 2) :
  (2 + 2 * (Real.cos A / Real.sin A) - 3 * (1 / Real.sin A)) *
  (3 + 2 * (Real.sin A / Real.cos A) + 1 / Real.cos A) = 11 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l4116_411634


namespace NUMINAMATH_CALUDE_quadratic_one_solution_positive_n_for_one_solution_l4116_411611

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + n * x + 25 = 0) ↔ n = 20 ∨ n = -20 := by
  sorry

theorem positive_n_for_one_solution (n : ℝ) :
  n > 0 ∧ (∃! x : ℝ, 4 * x^2 + n * x + 25 = 0) → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_positive_n_for_one_solution_l4116_411611


namespace NUMINAMATH_CALUDE_target_equals_fraction_l4116_411656

/-- The decimal representation of a rational number -/
def decimal_rep (q : ℚ) : ℕ → ℕ := sorry

/-- A function that checks if a decimal representation is repeating -/
def is_repeating (d : ℕ → ℕ) : Prop := sorry

/-- The rational number represented by 0.2̄34 -/
def target : ℚ := sorry

theorem target_equals_fraction : 
  (is_repeating (decimal_rep target)) → 
  (∀ a b : ℤ, (a / b : ℚ) = target → ∃ k : ℤ, k * 116 = a ∧ k * 495 = b) →
  target = 116 / 495 := by sorry

end NUMINAMATH_CALUDE_target_equals_fraction_l4116_411656


namespace NUMINAMATH_CALUDE_box_volume_conversion_l4116_411686

/-- Proves that a box with a volume of 216 cubic feet has a volume of 8 cubic yards. -/
theorem box_volume_conversion (box_volume_cubic_feet : ℝ) 
  (h1 : box_volume_cubic_feet = 216) : 
  box_volume_cubic_feet / 27 = 8 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_conversion_l4116_411686


namespace NUMINAMATH_CALUDE_compressor_stations_theorem_l4116_411687

/-- Represents the configuration of three compressor stations -/
structure CompressorStations where
  x : ℝ  -- Distance between first and second stations
  y : ℝ  -- Distance between second and third stations
  z : ℝ  -- Distance between first and third stations
  a : ℝ  -- Additional parameter

/-- Conditions for the compressor stations configuration -/
def valid_configuration (c : CompressorStations) : Prop :=
  c.x + c.y = 3 * c.z ∧
  c.z + c.y = c.x + c.a ∧
  c.x + c.z = 60 ∧
  c.x > 0 ∧ c.y > 0 ∧ c.z > 0

/-- Theorem stating the valid range for parameter a and specific values when a = 42 -/
theorem compressor_stations_theorem :
  ∀ c : CompressorStations,
    valid_configuration c →
    (0 < c.a ∧ c.a < 60) ∧
    (c.a = 42 → c.x = 33 ∧ c.y = 48 ∧ c.z = 27) :=
by sorry

end NUMINAMATH_CALUDE_compressor_stations_theorem_l4116_411687


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l4116_411660

/-- The number of distinct red balls in the bag -/
def num_red_balls : ℕ := 4

/-- The number of distinct white balls in the bag -/
def num_white_balls : ℕ := 6

/-- The score for drawing a red ball -/
def red_score : ℕ := 2

/-- The score for drawing a white ball -/
def white_score : ℕ := 1

/-- The number of ways to draw 4 balls such that the number of red balls is not less than the number of white balls -/
def ways_to_draw_4_balls : ℕ := 115

/-- The number of ways to draw 5 balls such that the total score is at least 7 points -/
def ways_to_draw_5_balls_score_7_plus : ℕ := 186

/-- The number of ways to arrange 5 drawn balls (with a score of 8 points) such that only two red balls are adjacent -/
def ways_to_arrange_5_balls_score_8 : ℕ := 4320

theorem ball_drawing_theorem : 
  ways_to_draw_4_balls = 115 ∧ 
  ways_to_draw_5_balls_score_7_plus = 186 ∧ 
  ways_to_arrange_5_balls_score_8 = 4320 := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l4116_411660


namespace NUMINAMATH_CALUDE_ninth_term_is_17_l4116_411679

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum_property : a 3 + a 4 = 12
  diff_property : ∀ n, a (n + 1) - a n = d
  d_value : d = 2

/-- The 9th term of the arithmetic sequence is 17 -/
theorem ninth_term_is_17 (seq : ArithmeticSequence) : seq.a 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_17_l4116_411679


namespace NUMINAMATH_CALUDE_flower_bed_total_l4116_411698

theorem flower_bed_total (tulips carnations : ℕ) : 
  tulips = 3 → carnations = 4 → tulips + carnations = 7 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_total_l4116_411698


namespace NUMINAMATH_CALUDE_highest_affordable_price_is_8_l4116_411604

/-- The highest whole-dollar price per shirt Alec can afford -/
def highest_affordable_price (total_budget : ℕ) (num_shirts : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ) : ℕ :=
  sorry

/-- The proposition to be proved -/
theorem highest_affordable_price_is_8 :
  highest_affordable_price 180 20 5 (8/100) = 8 := by
  sorry

end NUMINAMATH_CALUDE_highest_affordable_price_is_8_l4116_411604


namespace NUMINAMATH_CALUDE_tony_rollercoasters_l4116_411639

/-- The number of rollercoasters Tony went on -/
def num_rollercoasters : ℕ := 5

/-- The speeds of the rollercoasters Tony went on -/
def rollercoaster_speeds : List ℝ := [50, 62, 73, 70, 40]

/-- The average speed of all rollercoasters Tony went on -/
def average_speed : ℝ := 59

/-- Theorem stating that the number of rollercoasters Tony went on is correct -/
theorem tony_rollercoasters :
  num_rollercoasters = rollercoaster_speeds.length ∧
  (rollercoaster_speeds.sum / num_rollercoasters : ℝ) = average_speed := by
  sorry

end NUMINAMATH_CALUDE_tony_rollercoasters_l4116_411639


namespace NUMINAMATH_CALUDE_number_division_problem_l4116_411606

theorem number_division_problem : ∃ x : ℝ, (x / 5 = 70 + x / 6) ∧ x = 2100 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l4116_411606
