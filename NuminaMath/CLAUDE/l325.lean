import Mathlib

namespace NUMINAMATH_CALUDE_theater_attendance_l325_32507

theorem theater_attendance (adult_price child_price total_people total_revenue : ℕ)
  (h1 : adult_price = 8)
  (h2 : child_price = 1)
  (h3 : total_people = 22)
  (h4 : total_revenue = 50) :
  ∃ (adults children : ℕ),
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_revenue ∧
    children = 18 := by
  sorry

end NUMINAMATH_CALUDE_theater_attendance_l325_32507


namespace NUMINAMATH_CALUDE_ball_probability_l325_32537

theorem ball_probability (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h1 : total = 60) 
  (h2 : red = 6) 
  (h3 : purple = 9) : 
  (total - (red + purple)) / total = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l325_32537


namespace NUMINAMATH_CALUDE_nicholas_crackers_l325_32524

theorem nicholas_crackers (marcus_crackers : ℕ) (mona_crackers : ℕ) (nicholas_crackers : ℕ)
  (h1 : marcus_crackers = 27)
  (h2 : marcus_crackers = 3 * mona_crackers)
  (h3 : nicholas_crackers = mona_crackers + 6) :
  nicholas_crackers = 15 := by
sorry

end NUMINAMATH_CALUDE_nicholas_crackers_l325_32524


namespace NUMINAMATH_CALUDE_division_equation_may_not_hold_l325_32580

theorem division_equation_may_not_hold (a b c : ℝ) : 
  a = b → ¬∀ c, a / c = b / c :=
by
  sorry

end NUMINAMATH_CALUDE_division_equation_may_not_hold_l325_32580


namespace NUMINAMATH_CALUDE_pyramid_volume_l325_32529

theorem pyramid_volume (base_side : ℝ) (height : ℝ) (volume : ℝ) :
  base_side = 1 / 2 →
  height = 1 →
  volume = (1 / 3) * (base_side ^ 2) * height →
  volume = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l325_32529


namespace NUMINAMATH_CALUDE_parabola_b_value_l325_32504

/-- Given a parabola y = x^2 + ax + b passing through (2, 5) and (-2, -11), prove b = -7 -/
theorem parabola_b_value (a b : ℝ) : 
  (5 = 2^2 + 2*a + b) ∧ (-11 = (-2)^2 + (-2)*a + b) → b = -7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_b_value_l325_32504


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l325_32549

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (a b c d e f : ℝ) : Prop :=
  (a / b = d / e) ∧ (b ≠ 0) ∧ (e ≠ 0)

theorem parallel_lines_condition (a : ℝ) :
  (a = 1 → are_parallel a 2 (-1) 1 (a + 1) (-4)) ∧
  (∃ b : ℝ, b ≠ 1 ∧ are_parallel b 2 (-1) 1 (b + 1) (-4)) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l325_32549


namespace NUMINAMATH_CALUDE_monotone_increasing_intervals_l325_32591

/-- The function f(x) = 2x^3 - 3x^2 - 36x + 16 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 36 * x + 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x - 36

theorem monotone_increasing_intervals :
  MonotoneOn f (Set.Ici (-2) ∩ Set.Iic (-2)) ∧
  MonotoneOn f (Set.Ici 3) :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_intervals_l325_32591


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l325_32530

theorem pentagon_area_sum (u v : ℤ) : 
  0 < v → v < u → (u^2 + 3*u*v = 451) → u + v = 21 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l325_32530


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficient_l325_32542

/-- Given a polynomial Q(x) = x^3 + 3x^2 + dx + 15 where (x - 3) is a factor,
    prove that the coefficient d equals -23. -/
theorem polynomial_factor_coefficient (d : ℝ) : 
  (∀ x, x^3 + 3*x^2 + d*x + 15 = (x - 3) * (x^2 + (3 + 3)*x + (d + 9 + 3*3))) → 
  d = -23 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficient_l325_32542


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l325_32543

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 2 < b ∧ b < 4) :
  -4 < a - b ∧ a - b < -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l325_32543


namespace NUMINAMATH_CALUDE_inequality_solution_l325_32574

theorem inequality_solution (m : ℝ) : 
  (∃ (a : ℝ), a = 5 ∧ 
   ∃ (x : ℝ), |x - 1| - |x + m| ≥ a ∧ 
   ∀ (b : ℝ), (∃ (y : ℝ), |y - 1| - |y + m| ≥ b) → b ≤ a) → 
  m = 4 ∨ m = -6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l325_32574


namespace NUMINAMATH_CALUDE_train_length_calculation_l325_32528

/-- Proves that a train with given speed, crossing a bridge of known length in a specific time, has a particular length. -/
theorem train_length_calculation (train_speed : Real) (bridge_length : Real) (crossing_time : Real) (train_length : Real) : 
  train_speed = 36 → -- speed in km/hr
  bridge_length = 132 → -- bridge length in meters
  crossing_time = 24.198064154867613 → -- time to cross the bridge in seconds
  train_length = 109.98064154867613 → -- train length in meters
  (train_speed * 1000 / 3600) * crossing_time = bridge_length + train_length := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l325_32528


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l325_32526

theorem sum_of_squares_zero_implies_sum (a b c : ℝ) :
  (a - 5)^2 + (b - 6)^2 + (c - 7)^2 = 0 → a + b + c = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l325_32526


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_fifth_term_l325_32520

/-- 
Theorem: There exists a natural number n such that the binomial coefficient 
of the 5th term in the expansion of (x - 2/x)^n is the largest, and n = 7 
is one such value.
-/
theorem largest_binomial_coefficient_fifth_term : 
  ∃ n : ℕ, (
    -- The binomial coefficient of the 5th term is the largest
    ∀ k : ℕ, k ≤ n → (n.choose 4) ≥ (n.choose k)
  ) ∧ 
  -- n = 7 is a valid solution
  (7 : ℕ) ∈ { m : ℕ | ∀ k : ℕ, k ≤ m → (m.choose 4) ≥ (m.choose k) } :=
by sorry


end NUMINAMATH_CALUDE_largest_binomial_coefficient_fifth_term_l325_32520


namespace NUMINAMATH_CALUDE_average_speed_calculation_l325_32546

theorem average_speed_calculation (total_distance : ℝ) (first_half_distance : ℝ) (second_half_distance : ℝ) 
  (first_half_speed : ℝ) (second_half_speed : ℝ) 
  (h1 : total_distance = 60)
  (h2 : first_half_distance = 30)
  (h3 : second_half_distance = 30)
  (h4 : first_half_speed = 48)
  (h5 : second_half_speed = 24)
  (h6 : total_distance = first_half_distance + second_half_distance) :
  (total_distance / (first_half_distance / first_half_speed + second_half_distance / second_half_speed)) = 32 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l325_32546


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l325_32538

def initial_amount : ℚ := 120

def sandwich_fraction : ℚ := 1/5
def museum_fraction : ℚ := 1/6
def book_fraction : ℚ := 1/2

def remaining_amount : ℚ := 
  initial_amount - (initial_amount * sandwich_fraction + 
                    initial_amount * museum_fraction + 
                    initial_amount * book_fraction)

theorem jennifer_remaining_money : remaining_amount = 16 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l325_32538


namespace NUMINAMATH_CALUDE_flash_ace_chase_l325_32541

/-- The problem of Flash catching Ace -/
theorem flash_ace_chase (x y : ℝ) (hx : x > 1) : 
  let ace_speed := 1  -- We can set Ace's speed to 1 without loss of generality
  let flash_east_speed := x * ace_speed
  let flash_west_speed := (x + 1) * ace_speed
  let east_headstart := 2 * y
  let west_headstart := y
  let east_distance := (flash_east_speed * east_headstart) / (flash_east_speed - ace_speed)
  let west_distance := (flash_west_speed * west_headstart) / (flash_west_speed - ace_speed)
  east_distance + west_distance = (2 * x * y) / (x - 1) + ((x + 1) * y) / x :=
by sorry

end NUMINAMATH_CALUDE_flash_ace_chase_l325_32541


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_mod_7_l325_32584

def factorial (n : ℕ) : ℕ := sorry

def sum_factorials (n : ℕ) : ℕ := sorry

theorem factorial_sum_remainder_mod_7 : sum_factorials 10 % 7 = 5 := by sorry

end NUMINAMATH_CALUDE_factorial_sum_remainder_mod_7_l325_32584


namespace NUMINAMATH_CALUDE_unique_snuggly_number_l325_32521

def is_snuggly (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + b ∧ a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 2 * a + b^2

theorem unique_snuggly_number : ∃! n : ℕ, is_snuggly n :=
  sorry

end NUMINAMATH_CALUDE_unique_snuggly_number_l325_32521


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l325_32512

theorem sqrt_equation_solution (y : ℝ) :
  y > 2 → (Real.sqrt (7 * y) / Real.sqrt (4 * (y - 2)) = 3) → y = 72 / 29 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l325_32512


namespace NUMINAMATH_CALUDE_equation_solutions_l325_32539

theorem equation_solutions : 
  {x : ℝ | (47 - 2*x)^(1/4) + (35 + 2*x)^(1/4) = 4} = {23, -17} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l325_32539


namespace NUMINAMATH_CALUDE_derivative_implies_limit_l325_32599

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define x₀ as a real number
variable (x₀ : ℝ)

-- State the theorem
theorem derivative_implies_limit 
  (h₁ : HasDerivAt f (-2) x₀) :
  ∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ → 
    |((f (x₀ - 1/2 * h) - f x₀) / h) - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_derivative_implies_limit_l325_32599


namespace NUMINAMATH_CALUDE_original_equals_scientific_l325_32585

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number we want to represent in scientific notation -/
def original_number : ℕ := 650000

/-- The scientific notation representation of the original number -/
def scientific_repr : ScientificNotation := {
  coefficient := 6.5,
  exponent := 5,
  is_valid := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (original_number : ℝ) = scientific_repr.coefficient * (10 : ℝ) ^ scientific_repr.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l325_32585


namespace NUMINAMATH_CALUDE_jonathan_typing_rate_l325_32515

/-- The typing rates of Susan, Jack, and the combined rate with Jonathan -/
def typing_problem (susan_rate jack_rate combined_rate jonathan_rate : ℚ) : Prop :=
  susan_rate = 1 / 30 ∧ 
  jack_rate = 1 / 24 ∧ 
  combined_rate = 1 / 10 ∧
  combined_rate = susan_rate + jack_rate + jonathan_rate

/-- Theorem stating that under the given conditions, Jonathan's typing rate is 1/40 -/
theorem jonathan_typing_rate : 
  ∀ susan_rate jack_rate combined_rate jonathan_rate : ℚ,
  typing_problem susan_rate jack_rate combined_rate jonathan_rate →
  jonathan_rate = 1 / 40 :=
by
  sorry

end NUMINAMATH_CALUDE_jonathan_typing_rate_l325_32515


namespace NUMINAMATH_CALUDE_expression_value_l325_32518

theorem expression_value (a b c d x : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c ≠ 0 ∧ d ≠ 0)
  (h3 : c * d = 1) 
  (h4 : |x| = Real.sqrt 7) : 
  (x^2 + (a + b + c * d) * x + Real.sqrt (a + b) + (c * d) ^ (1/3 : ℝ) = 8 + Real.sqrt 7) ∨
  (x^2 + (a + b + c * d) * x + Real.sqrt (a + b) + (c * d) ^ (1/3 : ℝ) = 8 - Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l325_32518


namespace NUMINAMATH_CALUDE_negation_equivalence_l325_32525

theorem negation_equivalence :
  (¬ ∃ x : ℝ, Real.exp x > x) ↔ (∀ x : ℝ, Real.exp x ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l325_32525


namespace NUMINAMATH_CALUDE_john_remaining_money_l325_32535

theorem john_remaining_money (initial_amount : ℕ) (spent_amount : ℕ) : 
  initial_amount = 1600 →
  initial_amount - spent_amount = spent_amount - 600 →
  initial_amount - spent_amount = 500 :=
by sorry

end NUMINAMATH_CALUDE_john_remaining_money_l325_32535


namespace NUMINAMATH_CALUDE_triangle_side_length_l325_32598

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 2√3, b = 2, and the area S_ABC = √3, then c = 2 or c = 2√7. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S_ABC : ℝ) :
  a = 2 * Real.sqrt 3 →
  b = 2 →
  S_ABC = Real.sqrt 3 →
  (c = 2 ∨ c = 2 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l325_32598


namespace NUMINAMATH_CALUDE_find_other_number_l325_32536

theorem find_other_number (A B : ℕ+) (hA : A = 24) (hHCF : Nat.gcd A B = 13) (hLCM : Nat.lcm A B = 312) : B = 169 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l325_32536


namespace NUMINAMATH_CALUDE_y_derivative_l325_32548

noncomputable def y (x : ℝ) : ℝ := 3 * Real.arcsin (3 / (4 * x + 1)) + 2 * Real.sqrt (4 * x^2 + 2 * x - 2)

theorem y_derivative (x : ℝ) (h : 4 * x + 1 > 0) :
  deriv y x = (7 * (4 * x + 1)) / (2 * Real.sqrt (4 * x^2 + 2 * x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_y_derivative_l325_32548


namespace NUMINAMATH_CALUDE_abs_x_minus_four_plus_x_l325_32565

theorem abs_x_minus_four_plus_x (x : ℝ) (h : |x - 3| + x - 3 = 0) : |x - 4| + x = 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_four_plus_x_l325_32565


namespace NUMINAMATH_CALUDE_solve_apples_problem_l325_32559

def apples_problem (initial_apples : ℕ) (given_to_father : ℕ) (apples_per_person : ℕ) : Prop :=
  let remaining_apples := initial_apples - given_to_father
  let friends := (remaining_apples - apples_per_person) / apples_per_person
  friends = 4

theorem solve_apples_problem :
  apples_problem 55 10 9 :=
by sorry

end NUMINAMATH_CALUDE_solve_apples_problem_l325_32559


namespace NUMINAMATH_CALUDE_solution_set_equals_interval_l325_32568

-- Define the solution set of |x-3| < 5
def solution_set : Set ℝ := {x : ℝ | |x - 3| < 5}

-- State the theorem
theorem solution_set_equals_interval : solution_set = Set.Ioo (-2) 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equals_interval_l325_32568


namespace NUMINAMATH_CALUDE_sumata_vacation_miles_l325_32509

/-- The total miles driven on a vacation -/
def total_miles_driven (days : ℝ) (miles_per_day : ℝ) : ℝ :=
  days * miles_per_day

/-- Proof that the Sumata family drove 1250 miles on their vacation -/
theorem sumata_vacation_miles : 
  total_miles_driven 5.0 250 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_sumata_vacation_miles_l325_32509


namespace NUMINAMATH_CALUDE_largest_solution_and_fraction_l325_32533

theorem largest_solution_and_fraction (x : ℝ) :
  (7 * x) / 4 + 2 = 8 / x →
  ∃ (a b c d : ℤ),
    x = (a + b * Real.sqrt c) / d ∧
    a = -4 ∧ b = 8 ∧ c = 15 ∧ d = 7 ∧
    x ≤ (-4 + 8 * Real.sqrt 15) / 7 ∧
    (a * c * d : ℚ) / b = -105/2 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_and_fraction_l325_32533


namespace NUMINAMATH_CALUDE_toucan_count_l325_32517

/-- The total number of toucans after joining all limbs -/
def total_toucans (initial_first initial_second initial_third joining_first joining_second joining_third : ℝ) : ℝ :=
  (initial_first + joining_first) + (initial_second + joining_second) + (initial_third + joining_third)

/-- Theorem stating the total number of toucans after joining -/
theorem toucan_count : 
  total_toucans 3.5 4.25 2.75 1.5 0.6 1.2 = 13.8 := by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l325_32517


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l325_32505

/-- Proves that the interest rate is 6% given the conditions of the problem -/
theorem interest_rate_calculation (principal : ℝ) (time : ℝ) (total_interest : ℝ) 
  (h1 : principal = 1000)
  (h2 : time = 8)
  (h3 : total_interest = 480)
  (h4 : total_interest = principal * (rate / 100) * time) :
  rate = 6 := by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_interest_rate_calculation_l325_32505


namespace NUMINAMATH_CALUDE_chocolate_distribution_l325_32592

theorem chocolate_distribution (num_boxes : ℕ) (total_pieces : ℕ) (h1 : num_boxes = 6) (h2 : total_pieces = 3000) :
  total_pieces / num_boxes = 500 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l325_32592


namespace NUMINAMATH_CALUDE_supercomputer_additions_in_half_hour_l325_32566

/-- Proves that a supercomputer performing 20,000 additions per second can complete 36,000,000 additions in half an hour. -/
theorem supercomputer_additions_in_half_hour :
  let additions_per_second : ℕ := 20000
  let seconds_in_half_hour : ℕ := 1800
  additions_per_second * seconds_in_half_hour = 36000000 :=
by sorry

end NUMINAMATH_CALUDE_supercomputer_additions_in_half_hour_l325_32566


namespace NUMINAMATH_CALUDE_complement_of_N_wrt_M_l325_32594

-- Define the universal set M
def M : Set Nat := {1, 2, 3, 4, 5}

-- Define the set N
def N : Set Nat := {2, 4}

-- Define the complement of N with respect to M
def complementM (N : Set Nat) : Set Nat := M \ N

-- Theorem statement
theorem complement_of_N_wrt_M :
  complementM N = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_wrt_M_l325_32594


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l325_32502

theorem abs_sum_inequality (x y z : ℝ) :
  |x| + |y| + |z| ≤ |x+y-z| + |y+z-x| + |z+x-y| := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l325_32502


namespace NUMINAMATH_CALUDE_max_ab_value_l325_32501

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * (x - 1)

theorem max_ab_value (a : ℝ) (h : a > 0) :
  (∃ b : ℝ, ∀ x : ℝ, f a x ≥ b) →
  (∃ M : ℝ, M = (exp 3) / 2 ∧ ∀ b : ℝ, (∀ x : ℝ, f a x ≥ b) → a * b ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l325_32501


namespace NUMINAMATH_CALUDE_F_bounded_and_amplitude_l325_32544

def F (a x : ℝ) : ℝ := x * |x - 2*a| + 3

theorem F_bounded_and_amplitude (a : ℝ) (h : a ≤ 1/2) :
  ∃ (m M : ℝ), (∀ x ∈ Set.Icc 1 2, m ≤ F a x ∧ F a x ≤ M) ∧
  (M - m = 3 - 2*a) := by sorry

end NUMINAMATH_CALUDE_F_bounded_and_amplitude_l325_32544


namespace NUMINAMATH_CALUDE_third_defendant_guilty_l325_32593

-- Define the set of defendants
inductive Defendant : Type
  | A
  | B
  | C

-- Define the accusation function
def accuses : Defendant → Defendant → Prop := sorry

-- Define the truth-telling property
def tells_truth (d : Defendant) : Prop := sorry

-- Define the guilt property
def is_guilty (d : Defendant) : Prop := sorry

-- Define the condition that each defendant accuses one of the other two
axiom each_accuses_one : ∀ d₁ d₂ d₃ : Defendant, d₁ ≠ d₂ → d₁ ≠ d₃ → d₂ ≠ d₃ → 
  (accuses d₁ d₂ ∨ accuses d₁ d₃) ∧ (accuses d₂ d₁ ∨ accuses d₂ d₃) ∧ (accuses d₃ d₁ ∨ accuses d₃ d₂)

-- Define the condition that the first defendant (A) is the only one telling the truth
axiom A_tells_truth : tells_truth Defendant.A ∧ ¬tells_truth Defendant.B ∧ ¬tells_truth Defendant.C

-- Define the condition that if accusations were changed, B would be the only one telling the truth
axiom if_changed_B_tells_truth : 
  ∀ d₁ d₂ d₃ : Defendant, d₁ ≠ d₂ → d₁ ≠ d₃ → d₂ ≠ d₃ → 
  (accuses d₁ d₂ → accuses d₁ d₃) → (accuses d₂ d₁ → accuses d₂ d₃) → (accuses d₃ d₁ → accuses d₃ d₂) →
  tells_truth Defendant.B ∧ ¬tells_truth Defendant.A ∧ ¬tells_truth Defendant.C

-- Theorem: Given the conditions, the third defendant (C) is guilty
theorem third_defendant_guilty : is_guilty Defendant.C := by
  sorry

end NUMINAMATH_CALUDE_third_defendant_guilty_l325_32593


namespace NUMINAMATH_CALUDE_max_perimeter_special_triangle_l325_32506

/-- Represents a triangle with integer side lengths -/
structure IntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- A triangle with one side four times another and the third side 20 -/
def SpecialTriangle (x : ℕ) : IntegerTriangle where
  a := x
  b := 4 * x
  c := 20
  triangle_inequality := sorry

/-- The perimeter of a triangle -/
def perimeter (t : IntegerTriangle) : ℕ := t.a + t.b + t.c

/-- Theorem stating the maximum perimeter of the special triangle -/
theorem max_perimeter_special_triangle :
  ∃ (t : IntegerTriangle), (∃ x, t = SpecialTriangle x) ∧
    (∀ (t' : IntegerTriangle), (∃ x, t' = SpecialTriangle x) → perimeter t' ≤ perimeter t) ∧
    perimeter t = 50 := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_special_triangle_l325_32506


namespace NUMINAMATH_CALUDE_parabola_c_value_l325_32597

/-- Represents a quadratic function of the form f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines a parabola with given properties -/
def parabola : QuadraticFunction :=
  { a := sorry
    b := sorry
    c := sorry }

theorem parabola_c_value :
  (parabola.a * 3^2 + parabola.b * 3 + parabola.c = -5) ∧
  (parabola.a * 5^2 + parabola.b * 5 + parabola.c = -3) →
  parabola.c = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l325_32597


namespace NUMINAMATH_CALUDE_max_sum_of_cubes_l325_32554

theorem max_sum_of_cubes (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) : 
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_cubes_l325_32554


namespace NUMINAMATH_CALUDE_downstream_distance_l325_32567

/-- Proves that the distance traveled downstream is 420 km given the conditions -/
theorem downstream_distance
  (downstream_time : ℝ)
  (upstream_speed : ℝ)
  (total_speed : ℝ)
  (h1 : downstream_time = 20)
  (h2 : upstream_speed = 12)
  (h3 : total_speed = 21) :
  downstream_time * total_speed = 420 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_l325_32567


namespace NUMINAMATH_CALUDE_extreme_value_derivative_l325_32589

/-- A function has an extreme value at a point -/
def has_extreme_value (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

/-- The relationship between extreme values and derivative -/
theorem extreme_value_derivative (f : ℝ → ℝ) (x : ℝ) 
  (hf : Differentiable ℝ f) :
  (has_extreme_value f x → deriv f x = 0) ∧
  ∃ g : ℝ → ℝ, Differentiable ℝ g ∧ deriv g 0 = 0 ∧ ¬ has_extreme_value g 0 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_derivative_l325_32589


namespace NUMINAMATH_CALUDE_tetrahedron_unique_large_angle_sum_l325_32545

/-- A tetrahedron is a structure with four vertices and six edges. -/
structure Tetrahedron :=
  (A B C D : Point)

/-- The plane angle between two edges at a vertex of a tetrahedron. -/
def planeAngle (t : Tetrahedron) (v1 v2 v3 : Point) : ℝ := sorry

/-- The property that the sum of any two plane angles at a vertex is greater than 180°. -/
def hasLargeAngleSum (t : Tetrahedron) (v : Point) : Prop :=
  ∀ (v1 v2 v3 : Point), v1 ≠ v2 → v1 ≠ v3 → v2 ≠ v3 →
    planeAngle t v v1 v2 + planeAngle t v v1 v3 > 180

/-- Theorem: No more than one vertex of a tetrahedron can have the large angle sum property. -/
theorem tetrahedron_unique_large_angle_sum (t : Tetrahedron) :
  ¬∃ (v1 v2 : Point), v1 ≠ v2 ∧ hasLargeAngleSum t v1 ∧ hasLargeAngleSum t v2 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_unique_large_angle_sum_l325_32545


namespace NUMINAMATH_CALUDE_city_C_highest_growth_l325_32573

structure City where
  name : String
  pop1970 : ℕ
  pop1980 : ℕ

def cities : List City := [
  { name := "A", pop1970 := 40, pop1980 := 50 },
  { name := "B", pop1970 := 50, pop1980 := 70 },
  { name := "C", pop1970 := 70, pop1980 := 100 },
  { name := "D", pop1970 := 100, pop1980 := 130 },
  { name := "E", pop1970 := 120, pop1980 := 160 }
]

def growthRatio (city : City) : ℚ :=
  city.pop1980 / city.pop1970

theorem city_C_highest_growth :
  ∃ c ∈ cities, c.name = "C" ∧
  ∀ other ∈ cities, growthRatio c ≥ growthRatio other :=
by sorry

end NUMINAMATH_CALUDE_city_C_highest_growth_l325_32573


namespace NUMINAMATH_CALUDE_two_digit_product_4536_l325_32551

theorem two_digit_product_4536 (a b : ℕ) 
  (h1 : 10 ≤ a ∧ a < 100) 
  (h2 : 10 ≤ b ∧ b < 100) 
  (h3 : a * b = 4536) 
  (h4 : a ≤ b) : 
  a = 21 := by
sorry

end NUMINAMATH_CALUDE_two_digit_product_4536_l325_32551


namespace NUMINAMATH_CALUDE_min_chord_length_l325_32508

/-- The minimum chord length of a circle intersected by a line passing through a fixed point -/
theorem min_chord_length (O : ℝ × ℝ) (r : ℝ) (P : ℝ × ℝ) : 
  O = (2, 3) → r = 3 → P = (1, 1) → 
  let d := Real.sqrt ((O.1 - P.1)^2 + (O.2 - P.2)^2)
  ∃ (A B : ℝ × ℝ), (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2 ∧ 
                   (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2 ∧
                   (∀ (X Y : ℝ × ℝ), 
                     (X.1 - O.1)^2 + (X.2 - O.2)^2 = r^2 → 
                     (Y.1 - O.1)^2 + (Y.2 - O.2)^2 = r^2 →
                     Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) ≥ 
                     Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) ∧
                   Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 :=
by sorry


end NUMINAMATH_CALUDE_min_chord_length_l325_32508


namespace NUMINAMATH_CALUDE_negation_equivalence_l325_32564

theorem negation_equivalence : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 1 > 0)) ↔ (∀ x : ℝ, x > 0 → x^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l325_32564


namespace NUMINAMATH_CALUDE_intersection_A_B_l325_32556

def A : Set ℝ := {-2, -1, 1, 2, 4}

def B : Set ℝ := {x : ℝ | (x + 2) * (x - 3) < 0}

theorem intersection_A_B : A ∩ B = {-1, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l325_32556


namespace NUMINAMATH_CALUDE_triangle_tangent_inequality_l325_32561

/-- Given a triangle ABC with sides a, b, c and tangential segments x, y, z
    from vertices A, B, C to the incircle respectively, if a ≥ b ≥ c,
    then az + by + cx ≥ (a² + b² + c²)/2 ≥ ax + by + cz. -/
theorem triangle_tangent_inequality (a b c x y z : ℝ) 
    (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a = y + z) (h4 : b = x + z) (h5 : c = x + y) :
    a * z + b * y + c * x ≥ (a^2 + b^2 + c^2) / 2 ∧ 
    (a^2 + b^2 + c^2) / 2 ≥ a * x + b * y + c * z := by
  sorry


end NUMINAMATH_CALUDE_triangle_tangent_inequality_l325_32561


namespace NUMINAMATH_CALUDE_sandbox_volume_l325_32576

def sandbox_length : ℝ := 312
def sandbox_width : ℝ := 146
def sandbox_depth : ℝ := 56

theorem sandbox_volume :
  sandbox_length * sandbox_width * sandbox_depth = 2555520 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_volume_l325_32576


namespace NUMINAMATH_CALUDE_counterexample_exists_l325_32571

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem counterexample_exists : ∃ n : ℕ, 
  (sum_of_digits n % 27 = 0) ∧ 
  (n % 27 ≠ 0) ∧ 
  (n = 9918) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l325_32571


namespace NUMINAMATH_CALUDE_power_function_decreasing_first_quadrant_l325_32534

/-- A power function with negative exponent is decreasing in the first quadrant -/
theorem power_function_decreasing_first_quadrant (n : ℝ) (h : n < 0) :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂^n < x₁^n :=
by sorry


end NUMINAMATH_CALUDE_power_function_decreasing_first_quadrant_l325_32534


namespace NUMINAMATH_CALUDE_seven_minus_a_greater_than_b_l325_32570

theorem seven_minus_a_greater_than_b (a b : ℝ) (h : b < a ∧ a < 0) : 7 - a > b := by
  sorry

end NUMINAMATH_CALUDE_seven_minus_a_greater_than_b_l325_32570


namespace NUMINAMATH_CALUDE_charity_fundraising_l325_32569

theorem charity_fundraising 
  (total_amount : ℕ) 
  (num_friends : ℕ) 
  (min_amount : ℕ) 
  (h1 : total_amount = 3000)
  (h2 : num_friends = 10)
  (h3 : min_amount = 300) :
  (total_amount / num_friends = min_amount) ∧ 
  (∀ (amount : ℕ), amount ≥ min_amount → amount * num_friends = total_amount → amount = min_amount) :=
by sorry

end NUMINAMATH_CALUDE_charity_fundraising_l325_32569


namespace NUMINAMATH_CALUDE_prime_sum_100_l325_32511

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the sum of a list of natural numbers -/
def listSum (l : List ℕ) : ℕ := sorry

theorem prime_sum_100 :
  ∃ (l : List ℕ), 
    (∀ x ∈ l, isPrime x) ∧ 
    (listSum l = 100) ∧ 
    (l.length = 9) ∧
    (∀ (m : List ℕ), (∀ y ∈ m, isPrime y) → (listSum m = 100) → m.length ≥ 9) :=
sorry

end NUMINAMATH_CALUDE_prime_sum_100_l325_32511


namespace NUMINAMATH_CALUDE_sequence_properties_l325_32572

/-- Given a sequence {a_n} with the sum formula S_n = 2n^2 - 26n -/
def S (n : ℕ) : ℤ := 2 * n^2 - 26 * n

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℤ := 4 * n - 28

theorem sequence_properties :
  (∀ n : ℕ, a n = S (n + 1) - S n) ∧
  (∀ n : ℕ, a (n + 1) - a n = 4) ∧
  (∃ n : ℕ, n = 6 ∨ n = 7) ∧ (∀ m : ℕ, S m ≥ S 6 ∧ S m ≥ S 7) := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l325_32572


namespace NUMINAMATH_CALUDE_hotel_rooms_rented_l325_32522

theorem hotel_rooms_rented (total_rooms : ℝ) (h1 : total_rooms > 0) : 
  let air_conditioned := (3/5) * total_rooms
  let rented_air_conditioned := (2/3) * air_conditioned
  let not_rented := total_rooms - (rented_air_conditioned + (1/5) * air_conditioned)
  (total_rooms - not_rented) / total_rooms = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_rented_l325_32522


namespace NUMINAMATH_CALUDE_area_of_triangle_pqr_l325_32563

-- Define the square pyramid
structure SquarePyramid where
  base_side : ℝ
  altitude : ℝ

-- Define points P, Q, R
structure PyramidPoints where
  p_ratio : ℝ
  q_ratio : ℝ
  r_ratio : ℝ

-- Define the theorem
theorem area_of_triangle_pqr 
  (pyramid : SquarePyramid) 
  (points : PyramidPoints) 
  (h1 : pyramid.base_side = 4) 
  (h2 : pyramid.altitude = 8) 
  (h3 : points.p_ratio = 1/4) 
  (h4 : points.q_ratio = 1/4) 
  (h5 : points.r_ratio = 3/4) : 
  ∃ (area : ℝ), area = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_pqr_l325_32563


namespace NUMINAMATH_CALUDE_tan_theta_plus_pi_third_l325_32586

theorem tan_theta_plus_pi_third (θ : Real) (h1 : 0 ≤ θ) (h2 : θ < 2 * Real.pi) :
  (Real.sin (3 * Real.pi / 4) : Real) / Real.cos (3 * Real.pi / 4) = Real.tan θ →
  Real.tan (θ + Real.pi / 3) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_plus_pi_third_l325_32586


namespace NUMINAMATH_CALUDE_all_propositions_false_l325_32527

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relations
def parallel (x y : Line ⊕ Plane) : Prop := sorry
def perpendicular (x y : Line ⊕ Plane) : Prop := sorry
def contains (p : Plane) (l : Line) : Prop := sorry
def intersects (p q : Plane) (l : Line) : Prop := sorry

-- Define the lines and planes
def m : Line := sorry
def n : Line := sorry
def a : Plane := sorry
def b : Plane := sorry

-- Define the propositions
def proposition1 : Prop :=
  ∀ (m n : Line) (a b : Plane),
    parallel (Sum.inl m) (Sum.inr a) →
    parallel (Sum.inl n) (Sum.inr b) →
    parallel (Sum.inr a) (Sum.inr b) →
    parallel (Sum.inl m) (Sum.inl n)

def proposition2 : Prop :=
  ∀ (m n : Line) (a b : Plane),
    parallel (Sum.inl m) (Sum.inl n) →
    contains a m →
    perpendicular (Sum.inl n) (Sum.inr b) →
    perpendicular (Sum.inr a) (Sum.inr b)

def proposition3 : Prop :=
  ∀ (m n : Line) (a b : Plane),
    intersects a b m →
    parallel (Sum.inl m) (Sum.inl n) →
    (parallel (Sum.inl n) (Sum.inr a) ∧ parallel (Sum.inl n) (Sum.inr b))

def proposition4 : Prop :=
  ∀ (m n : Line) (a b : Plane),
    perpendicular (Sum.inl m) (Sum.inl n) →
    intersects a b m →
    (perpendicular (Sum.inl n) (Sum.inr a) ∨ perpendicular (Sum.inl n) (Sum.inr b))

-- Theorem stating that all propositions are false
theorem all_propositions_false :
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 := by
  sorry

end NUMINAMATH_CALUDE_all_propositions_false_l325_32527


namespace NUMINAMATH_CALUDE_polynomial_simplification_l325_32519

theorem polynomial_simplification (p : ℝ) :
  (5 * p^4 - 4 * p^3 + 3 * p + 2) + (-3 * p^4 + 2 * p^3 - 7 * p^2 + 8) =
  2 * p^4 - 2 * p^3 - 7 * p^2 + 3 * p + 10 := by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l325_32519


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_equals_one_fourth_more_than_fifty_l325_32514

theorem thirty_percent_less_than_ninety_equals_one_fourth_more_than_fifty : 
  (90 : ℝ) * (1 - 0.3) = 50 * (1 + 0.25) := by sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_equals_one_fourth_more_than_fifty_l325_32514


namespace NUMINAMATH_CALUDE_power_sum_difference_l325_32582

theorem power_sum_difference : 3^(1+2+3+4) - (3^1 + 3^2 + 3^3 + 3^4) = 58929 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l325_32582


namespace NUMINAMATH_CALUDE_probability_at_least_one_white_l325_32577

/-- The probability of drawing at least one white ball from a box -/
theorem probability_at_least_one_white (total : ℕ) (white : ℕ) (red : ℕ) (draw : ℕ) :
  total = white + red →
  white = 8 →
  red = 2 →
  draw = 2 →
  (Nat.choose white 1 * Nat.choose red 1 + Nat.choose white 2 * Nat.choose red 0) / Nat.choose total draw = 44 / 45 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_white_l325_32577


namespace NUMINAMATH_CALUDE_louis_lemon_heads_l325_32579

/-- The number of Lemon Heads in each package -/
def lemon_heads_per_package : ℕ := 6

/-- The number of whole boxes Louis finished -/
def boxes_finished : ℕ := 9

/-- The total number of Lemon Heads Louis ate -/
def total_lemon_heads : ℕ := lemon_heads_per_package * boxes_finished

theorem louis_lemon_heads :
  total_lemon_heads = 54 :=
by sorry

end NUMINAMATH_CALUDE_louis_lemon_heads_l325_32579


namespace NUMINAMATH_CALUDE_makenna_larger_garden_l325_32523

-- Define the dimensions of Karl's garden
def karl_length : ℕ := 30
def karl_width : ℕ := 50

-- Define the dimensions of Makenna's garden
def makenna_length : ℕ := 35
def makenna_width : ℕ := 45

-- Define the area Karl allocates for trees
def karl_tree_area : ℕ := 300

-- Calculate the areas of both gardens
def karl_total_area : ℕ := karl_length * karl_width
def makenna_total_area : ℕ := makenna_length * makenna_width

-- Calculate Karl's vegetable area
def karl_veg_area : ℕ := karl_total_area - karl_tree_area

-- Define the difference between vegetable areas
def veg_area_difference : ℕ := makenna_total_area - karl_veg_area

-- Theorem statement
theorem makenna_larger_garden : veg_area_difference = 375 := by
  sorry

end NUMINAMATH_CALUDE_makenna_larger_garden_l325_32523


namespace NUMINAMATH_CALUDE_first_year_after_2000_with_digit_sum_15_l325_32540

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isAfter2000 (year : ℕ) : Prop :=
  year > 2000

theorem first_year_after_2000_with_digit_sum_15 :
  ∀ year : ℕ, isAfter2000 year → sumOfDigits year = 15 → year ≥ 2049 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2000_with_digit_sum_15_l325_32540


namespace NUMINAMATH_CALUDE_minimum_rent_is_36800_l325_32583

/-- Represents the minimum rent problem for a travel agency --/
def MinimumRentProblem (total_passengers : ℕ) (capacity_A capacity_B : ℕ) (rent_A rent_B : ℕ) (max_buses : ℕ) (max_B_diff : ℕ) : Prop :=
  ∃ (num_A num_B : ℕ),
    -- Total passengers condition
    num_A * capacity_A + num_B * capacity_B ≥ total_passengers ∧
    -- Maximum number of buses condition
    num_A + num_B ≤ max_buses ∧
    -- Condition on the difference between B and A buses
    num_B ≤ num_A + max_B_diff ∧
    -- Minimum rent calculation
    ∀ (other_A other_B : ℕ),
      other_A * capacity_A + other_B * capacity_B ≥ total_passengers →
      other_A + other_B ≤ max_buses →
      other_B ≤ other_A + max_B_diff →
      num_A * rent_A + num_B * rent_B ≤ other_A * rent_A + other_B * rent_B

/-- The minimum rent for the given problem is 36800 yuan --/
theorem minimum_rent_is_36800 :
  MinimumRentProblem 900 36 60 1600 2400 21 7 →
  ∃ (num_A num_B : ℕ), num_A * 1600 + num_B * 2400 = 36800 :=
sorry

end NUMINAMATH_CALUDE_minimum_rent_is_36800_l325_32583


namespace NUMINAMATH_CALUDE_cape_may_august_sharks_l325_32552

/-- The number of sharks in Daytona Beach in July -/
def daytona_july : ℕ := 23

/-- The number of sharks in Cape May in July -/
def cape_may_july : ℕ := 2 * daytona_july

/-- The number of sharks in Daytona Beach in August -/
def daytona_august : ℕ := daytona_july

/-- The number of sharks in Cape May in August -/
def cape_may_august : ℕ := 5 + 3 * daytona_august

theorem cape_may_august_sharks : cape_may_august = 74 := by
  sorry

end NUMINAMATH_CALUDE_cape_may_august_sharks_l325_32552


namespace NUMINAMATH_CALUDE_race_heartbeats_l325_32532

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

theorem race_heartbeats :
  total_heartbeats 160 6 30 = 28800 := by
  sorry

end NUMINAMATH_CALUDE_race_heartbeats_l325_32532


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l325_32557

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ + a₈ = 6, prove that 3a₂ + a₁₆ = 12 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arithmetic : arithmetic_sequence a) 
    (h_sum : a 3 + a 8 = 6) : 
  3 * a 2 + a 16 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l325_32557


namespace NUMINAMATH_CALUDE_quadratic_two_roots_condition_l325_32510

theorem quadratic_two_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + k = 0 ∧ y^2 - 6*y + k = 0) ↔ k < 9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_condition_l325_32510


namespace NUMINAMATH_CALUDE_matts_work_schedule_l325_32588

/-- Matt's work schedule problem -/
theorem matts_work_schedule (monday_minutes : ℕ) (wednesday_minutes : ℕ) : 
  monday_minutes = 450 →
  wednesday_minutes = 300 →
  wednesday_minutes - (monday_minutes / 2) = 75 := by
  sorry

end NUMINAMATH_CALUDE_matts_work_schedule_l325_32588


namespace NUMINAMATH_CALUDE_product_of_roots_plus_one_l325_32587

theorem product_of_roots_plus_one (a b c : ℂ) : 
  (x^3 - 15*x^2 + 22*x - 8 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (1 + a) * (1 + b) * (1 + c) = 46 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_one_l325_32587


namespace NUMINAMATH_CALUDE_fruit_prices_l325_32581

theorem fruit_prices (mango_price banana_price : ℝ) : 
  (3 * mango_price + 2 * banana_price = 40) →
  (2 * mango_price + 3 * banana_price = 35) →
  (mango_price = 10 ∧ banana_price = 5) := by
sorry

end NUMINAMATH_CALUDE_fruit_prices_l325_32581


namespace NUMINAMATH_CALUDE_third_term_of_specific_sequence_l325_32516

/-- Represents a geometric sequence of positive integers -/
structure GeometricSequence where
  first_term : ℕ
  common_ratio : ℕ
  first_term_pos : 0 < first_term

/-- The nth term of a geometric sequence -/
def nth_term (seq : GeometricSequence) (n : ℕ) : ℕ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

theorem third_term_of_specific_sequence :
  ∀ (seq : GeometricSequence),
    seq.first_term = 5 →
    nth_term seq 4 = 320 →
    nth_term seq 3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_specific_sequence_l325_32516


namespace NUMINAMATH_CALUDE_proposition_a_necessary_not_sufficient_for_b_l325_32553

theorem proposition_a_necessary_not_sufficient_for_b (h : ℝ) (h_pos : h > 0) :
  (∀ a b : ℝ, (|a - 1| < h ∧ |b - 1| < h) → |a - b| < 2 * h) ∧
  (∃ a b : ℝ, |a - b| < 2 * h ∧ ¬(|a - 1| < h ∧ |b - 1| < h)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_a_necessary_not_sufficient_for_b_l325_32553


namespace NUMINAMATH_CALUDE_theo_has_winning_strategy_l325_32590

/-- Game state representing the current turn number and cumulative score -/
structure GameState where
  turn : ℕ
  score : ℕ

/-- Player type -/
inductive Player
| Anatole
| Theo

/-- Game move representing the chosen number and resulting turn score -/
structure GameMove where
  number : ℕ
  turn_score : ℕ

/-- Represents a strategy for a player -/
def Strategy := GameState → GameMove

/-- Checks if a move is valid according to game rules -/
def is_valid_move (p : ℕ) (prev_move : Option GameMove) (current_move : GameMove) : Prop :=
  match prev_move with
  | none => current_move.number > 0
  | some prev => current_move.number > prev.number

/-- Checks if a player wins with a given move -/
def is_winning_move (p : ℕ) (state : GameState) (move : GameMove) : Prop :=
  (p ∣ (move.turn_score * (state.score + state.turn * move.turn_score)))

/-- Theorem stating that Theo has a winning strategy -/
theorem theo_has_winning_strategy (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2) :
  ∃ (theo_strategy : Strategy),
    ∀ (anatole_strategy : Strategy),
      ∃ (final_state : GameState),
        final_state.turn < p - 1 ∧
        is_winning_move p final_state (theo_strategy final_state) :=
  sorry

end NUMINAMATH_CALUDE_theo_has_winning_strategy_l325_32590


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l325_32578

def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ n = a + b

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l325_32578


namespace NUMINAMATH_CALUDE_berry_swap_difference_l325_32531

/-- The number of blueberries in each blue box -/
def blueberries_per_box : ℕ := 20

/-- The increase in total berries when swapping one blue box for one red box -/
def berry_increase : ℕ := 10

/-- The number of strawberries in each red box -/
def strawberries_per_box : ℕ := blueberries_per_box + berry_increase

/-- The change in the difference between total strawberries and total blueberries -/
def difference_change : ℕ := strawberries_per_box + blueberries_per_box

theorem berry_swap_difference :
  difference_change = 50 :=
sorry

end NUMINAMATH_CALUDE_berry_swap_difference_l325_32531


namespace NUMINAMATH_CALUDE_placemat_length_l325_32558

theorem placemat_length (r : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) : 
  r = 5 → n = 8 → w = 1 → 
  y = 2 * r * Real.sin ((π / n) / 2) →
  y = 10 * Real.sin (5 * π / 16) :=
by sorry

end NUMINAMATH_CALUDE_placemat_length_l325_32558


namespace NUMINAMATH_CALUDE_pet_shop_animals_l325_32547

theorem pet_shop_animals (kittens hamsters birds : ℕ) 
  (h1 : kittens = 32) 
  (h2 : hamsters = 15) 
  (h3 : birds = 30) : 
  kittens + hamsters + birds = 77 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_animals_l325_32547


namespace NUMINAMATH_CALUDE_book_pages_calculation_l325_32513

theorem book_pages_calculation (pages_remaining : ℕ) (percentage_read : ℚ) : 
  pages_remaining = 320 ∧ percentage_read = 1/5 → 
  pages_remaining / (1 - percentage_read) = 400 := by
sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l325_32513


namespace NUMINAMATH_CALUDE_inverse_g_sum_l325_32560

-- Define the function g
def g (x : ℝ) : ℝ := x * |x| + 3 * x

-- State the theorem
theorem inverse_g_sum : 
  ∃ (a b : ℝ), g a = 9 ∧ g b = -121 ∧ a + b = (3 * Real.sqrt 5 - 23) / 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_g_sum_l325_32560


namespace NUMINAMATH_CALUDE_subtraction_with_division_l325_32550

theorem subtraction_with_division : 3034 - (1002 / 200.4) = 3029 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_with_division_l325_32550


namespace NUMINAMATH_CALUDE_total_movies_is_74_l325_32503

/-- Represents the number of movies watched by each person -/
structure MovieCounts where
  dalton : ℕ
  hunter : ℕ
  alex : ℕ
  bella : ℕ
  chris : ℕ

/-- Represents the number of movies watched by different groups -/
structure SharedMovies where
  all_five : ℕ
  dalton_hunter_alex : ℕ
  bella_chris : ℕ
  dalton_bella : ℕ
  alex_chris : ℕ

/-- Calculates the total number of different movies watched -/
def total_different_movies (individual : MovieCounts) (shared : SharedMovies) : ℕ :=
  individual.dalton + individual.hunter + individual.alex + individual.bella + individual.chris -
  (4 * shared.all_five + 2 * shared.dalton_hunter_alex + shared.bella_chris + shared.dalton_bella + shared.alex_chris)

/-- Theorem stating that the total number of different movies watched is 74 -/
theorem total_movies_is_74 (individual : MovieCounts) (shared : SharedMovies)
    (h1 : individual = ⟨20, 26, 35, 29, 16⟩)
    (h2 : shared = ⟨5, 4, 3, 2, 4⟩) :
    total_different_movies individual shared = 74 := by
  sorry

end NUMINAMATH_CALUDE_total_movies_is_74_l325_32503


namespace NUMINAMATH_CALUDE_interns_escape_probability_l325_32555

/-- A permutation on n elements -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The probability that a random permutation on n elements has no cycle longer than k -/
noncomputable def prob_no_long_cycle (n k : ℕ) : ℝ := sorry

/-- The number of interns/drawers -/
def num_interns : ℕ := 44

/-- The maximum allowed cycle length for survival -/
def max_cycle_length : ℕ := 21

/-- The minimum required survival probability -/
def min_survival_prob : ℝ := 0.30

theorem interns_escape_probability :
  prob_no_long_cycle num_interns max_cycle_length > min_survival_prob := by sorry

end NUMINAMATH_CALUDE_interns_escape_probability_l325_32555


namespace NUMINAMATH_CALUDE_problem_solution_l325_32595

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 2)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 15)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 130)
  (eq4 : 16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 550) :
  25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ = 1492 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l325_32595


namespace NUMINAMATH_CALUDE_equation_solution_l325_32500

theorem equation_solution (x : ℝ) (h : x ≠ 0) : 4 / x^2 = x / 16 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l325_32500


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l325_32575

/-- Given a point P and a line L, this theorem proves that a specific equation
    represents a line passing through P and parallel to L. -/
theorem parallel_line_through_point (x y : ℝ) :
  let P : ℝ × ℝ := (2, Real.sqrt 3)
  let L : ℝ → ℝ → ℝ := fun x y => Real.sqrt 3 * x - y + 2
  let parallel_line : ℝ → ℝ → ℝ := fun x y => Real.sqrt 3 * x - y - Real.sqrt 3
  (parallel_line P.1 P.2 = 0) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ ∀ x y, parallel_line x y = k * L x y) :=
by sorry


end NUMINAMATH_CALUDE_parallel_line_through_point_l325_32575


namespace NUMINAMATH_CALUDE_circles_tangent_internally_l325_32596

/-- Two circles are tangent internally if the distance between their centers
    is equal to the difference of their radii --/
def are_tangent_internally (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2) = r1 - r2

/-- Given two circles with specified centers and radii, prove they are tangent internally --/
theorem circles_tangent_internally :
  let c1 : ℝ × ℝ := (0, 8)
  let c2 : ℝ × ℝ := (-6, 0)
  let r1 : ℝ := 12
  let r2 : ℝ := 2
  are_tangent_internally c1 c2 r1 r2 := by
  sorry


end NUMINAMATH_CALUDE_circles_tangent_internally_l325_32596


namespace NUMINAMATH_CALUDE_expand_polynomial_l325_32562

theorem expand_polynomial (x : ℝ) : (x + 3) * (2*x^2 - x + 4) = 2*x^3 + 5*x^2 + x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l325_32562
