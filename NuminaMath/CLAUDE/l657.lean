import Mathlib

namespace NUMINAMATH_CALUDE_last_four_digits_are_user_number_l657_65723

/-- Represents a mobile phone number -/
structure MobilePhoneNumber where
  digits : Fin 11 → Nat
  network_id : Fin 3 → Nat
  area_code : Fin 3 → Nat
  user_number : Fin 4 → Nat

/-- The structure of a mobile phone number -/
def mobile_number_structure (m : MobilePhoneNumber) : Prop :=
  (∀ i : Fin 3, m.network_id i = m.digits i) ∧
  (∀ i : Fin 3, m.area_code i = m.digits (i + 3)) ∧
  (∀ i : Fin 4, m.user_number i = m.digits (i + 7))

/-- Theorem stating that the last 4 digits of a mobile phone number represent the user number -/
theorem last_four_digits_are_user_number (m : MobilePhoneNumber) 
  (h : mobile_number_structure m) : 
  ∀ i : Fin 4, m.user_number i = m.digits (i + 7) := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_are_user_number_l657_65723


namespace NUMINAMATH_CALUDE_unclaimed_fraction_is_correct_l657_65767

/-- Represents a participant in the chocolate distribution --/
inductive Participant
  | Dave
  | Emma
  | Frank
  | George

/-- The ratio of chocolate distribution for each participant --/
def distribution_ratio (p : Participant) : Rat :=
  match p with
  | Participant.Dave => 4/10
  | Participant.Emma => 3/10
  | Participant.Frank => 2/10
  | Participant.George => 1/10

/-- The order in which participants claim their share --/
def claim_order : List Participant :=
  [Participant.Dave, Participant.Emma, Participant.Frank, Participant.George]

/-- Calculate the fraction of chocolates claimed by a participant --/
def claimed_fraction (p : Participant) (remaining : Rat) : Rat :=
  (distribution_ratio p) * remaining

/-- Calculate the fraction of chocolates that remains unclaimed --/
def unclaimed_fraction : Rat :=
  let initial_remaining : Rat := 1
  let final_remaining := claim_order.foldl
    (fun remaining p => remaining - claimed_fraction p remaining)
    initial_remaining
  final_remaining

/-- Theorem: The fraction of chocolates that remains unclaimed is 37.8/125 --/
theorem unclaimed_fraction_is_correct :
  unclaimed_fraction = 378/1250 := by
  sorry


end NUMINAMATH_CALUDE_unclaimed_fraction_is_correct_l657_65767


namespace NUMINAMATH_CALUDE_cube_inequality_l657_65753

theorem cube_inequality (a b : ℝ) (ha : a > 0) (hb : b < 0) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l657_65753


namespace NUMINAMATH_CALUDE_min_socks_for_twelve_pairs_l657_65786

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (black : ℕ)
  (yellow : ℕ)

/-- Represents the problem setup -/
def initialDrawer : SockDrawer :=
  { red := 120
  , green := 90
  , blue := 70
  , black := 50
  , yellow := 30 }

/-- The number of pairs we want to guarantee -/
def requiredPairs : ℕ := 12

/-- Function to calculate the minimum number of socks needed to guarantee the required pairs -/
def minSocksForPairs (drawer : SockDrawer) (pairs : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 28 socks are needed to guarantee 12 pairs -/
theorem min_socks_for_twelve_pairs :
  minSocksForPairs initialDrawer requiredPairs = 28 :=
sorry

end NUMINAMATH_CALUDE_min_socks_for_twelve_pairs_l657_65786


namespace NUMINAMATH_CALUDE_inequality_solution_l657_65744

def k : ℝ := 0.5

def inequality (θ x : ℝ) : Prop :=
  x^2 * Real.sin θ - k*x*(1 - x) + (1 - x)^2 * Real.cos θ ≥ 0

def solution_set : Set ℝ :=
  {θ | 0 ≤ θ ∧ θ ≤ 2*Real.pi ∧ ∀ x, 0 ≤ x ∧ x ≤ 1 → inequality θ x}

theorem inequality_solution :
  solution_set = {θ | (0 ≤ θ ∧ θ ≤ Real.pi/12) ∨ (23*Real.pi/12 ≤ θ ∧ θ ≤ 2*Real.pi)} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l657_65744


namespace NUMINAMATH_CALUDE_rachels_pizza_consumption_l657_65765

theorem rachels_pizza_consumption 
  (total_pizza : ℕ) 
  (bellas_pizza : ℕ) 
  (h1 : total_pizza = 952) 
  (h2 : bellas_pizza = 354) : 
  total_pizza - bellas_pizza = 598 := by
sorry

end NUMINAMATH_CALUDE_rachels_pizza_consumption_l657_65765


namespace NUMINAMATH_CALUDE_impossible_circle_assignment_l657_65799

-- Define the type for circles
def Circle := Fin 6

-- Define the connection relation between circles
def connected : Circle → Circle → Prop := sorry

-- Define the divisibility relation
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

-- Main theorem
theorem impossible_circle_assignment :
  ¬ ∃ (f : Circle → ℕ),
    (∀ i j : Circle, connected i j → (divides (f i) (f j) ∨ divides (f j) (f i))) ∧
    (∀ i j : Circle, ¬ connected i j → ¬ divides (f i) (f j) ∧ ¬ divides (f j) (f i)) :=
by sorry


end NUMINAMATH_CALUDE_impossible_circle_assignment_l657_65799


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l657_65754

theorem angle_in_second_quadrant (θ : Real) : 
  (Real.tan θ * Real.sin θ < 0) → 
  (Real.tan θ * Real.cos θ > 0) → 
  (0 < θ) ∧ (θ < Real.pi) := by
sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l657_65754


namespace NUMINAMATH_CALUDE_pinky_pies_l657_65770

theorem pinky_pies (helen_pies total_pies : ℕ) 
  (helen_made : helen_pies = 56)
  (total : total_pies = 203) :
  total_pies - helen_pies = 147 := by
  sorry

end NUMINAMATH_CALUDE_pinky_pies_l657_65770


namespace NUMINAMATH_CALUDE_scientists_communication_l657_65736

/-- A coloring of edges in a complete graph -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A triangle in a graph -/
def Triangle (n : ℕ) := { t : Fin n × Fin n × Fin n // t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2 }

/-- A monochromatic triangle under a given coloring -/
def MonochromaticTriangle (n : ℕ) (c : Coloring n) (t : Triangle n) :=
  c t.val.1 t.val.2.1 = c t.val.1 t.val.2.2 ∧
  c t.val.1 t.val.2.1 = c t.val.2.1 t.val.2.2

theorem scientists_communication :
  ∀ c : Coloring 17, ∃ t : Triangle 17, MonochromaticTriangle 17 c t :=
sorry

end NUMINAMATH_CALUDE_scientists_communication_l657_65736


namespace NUMINAMATH_CALUDE_division_problem_l657_65735

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 265 →
  divisor = 22 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 12 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l657_65735


namespace NUMINAMATH_CALUDE_customers_left_l657_65785

theorem customers_left (initial : ℕ) (first_leave_percent : ℚ) (second_leave_percent : ℚ) : 
  initial = 36 → 
  first_leave_percent = 1/2 → 
  second_leave_percent = 3/10 → 
  ⌊(initial - ⌊initial * first_leave_percent⌋) - ⌊(initial - ⌊initial * first_leave_percent⌋) * second_leave_percent⌋⌋ = 13 := by
  sorry

end NUMINAMATH_CALUDE_customers_left_l657_65785


namespace NUMINAMATH_CALUDE_monic_polynomial_theorem_l657_65787

def is_monic_degree_7 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d e f g : ℝ, ∀ x, p x = x^7 + a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + f*x + g

def satisfies_conditions (p : ℝ → ℝ) : Prop :=
  p 1 = 1 ∧ p 2 = 2 ∧ p 3 = 3 ∧ p 4 = 4 ∧ p 5 = 5 ∧ p 6 = 6 ∧ p 7 = 7

theorem monic_polynomial_theorem (p : ℝ → ℝ) 
  (h1 : is_monic_degree_7 p) 
  (h2 : satisfies_conditions p) : 
  p 8 = 5048 := by
  sorry

end NUMINAMATH_CALUDE_monic_polynomial_theorem_l657_65787


namespace NUMINAMATH_CALUDE_decreasing_linear_function_l657_65757

theorem decreasing_linear_function (x1 x2 : ℝ) (h : x2 > x1) : -6 * x2 < -6 * x1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_l657_65757


namespace NUMINAMATH_CALUDE_max_successful_throws_l657_65792

/-- Represents the number of free throws attempted by Andrew -/
def andrew_throws : ℕ → ℕ := λ a => a

/-- Represents the number of free throws attempted by Beatrice -/
def beatrice_throws : ℕ → ℕ := λ b => b

/-- Represents the total number of free throws -/
def total_throws : ℕ := 105

/-- Represents the success rate of Andrew's free throws -/
def andrew_success_rate : ℚ := 1/3

/-- Represents the success rate of Beatrice's free throws -/
def beatrice_success_rate : ℚ := 3/5

/-- Calculates the total number of successful free throws -/
def total_successful_throws (a b : ℕ) : ℚ :=
  andrew_success_rate * a + beatrice_success_rate * b

/-- Theorem stating the maximum number of successful free throws -/
theorem max_successful_throws :
  ∃ (a b : ℕ), 
    a > 0 ∧ 
    b > 0 ∧ 
    a + b = total_throws ∧
    ∀ (x y : ℕ), 
      x > 0 → 
      y > 0 → 
      x + y = total_throws → 
      total_successful_throws a b ≥ total_successful_throws x y ∧
      total_successful_throws a b = 59 :=
sorry

end NUMINAMATH_CALUDE_max_successful_throws_l657_65792


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l657_65709

-- Define sets M and N
def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Theorem stating that "a ∈ M" is a necessary but not sufficient condition for "a ∈ N"
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by
  sorry

end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l657_65709


namespace NUMINAMATH_CALUDE_tangent_product_equals_two_l657_65773

theorem tangent_product_equals_two (x y : Real) 
  (h1 : x = 21 * π / 180) 
  (h2 : y = 24 * π / 180) 
  (h3 : Real.tan (π / 4) = 1) 
  (h4 : π / 4 = x + y) : 
  (1 + Real.tan x) * (1 + Real.tan y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_equals_two_l657_65773


namespace NUMINAMATH_CALUDE_half_percent_to_decimal_l657_65749

theorem half_percent_to_decimal : (1 / 2 : ℚ) / 100 = (0.005 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_half_percent_to_decimal_l657_65749


namespace NUMINAMATH_CALUDE_no_four_identical_digits_in_powers_of_two_l657_65731

theorem no_four_identical_digits_in_powers_of_two :
  ∀ n : ℕ, ¬ ∃ a : ℕ, a < 10 ∧ (2^n : ℕ) % 10000 = a * 1111 :=
sorry

end NUMINAMATH_CALUDE_no_four_identical_digits_in_powers_of_two_l657_65731


namespace NUMINAMATH_CALUDE_arithmetic_sequence_quadratic_roots_l657_65750

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

-- Define the quadratic equation
def quadratic_equation (a : ℕ → ℝ) (k : ℕ) (x : ℝ) :=
  a k * x^2 + 2 * a (k + 1) * x + a (k + 2) = 0

-- Main theorem
theorem arithmetic_sequence_quadratic_roots
  (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0)
  (h_a : ∀ n, a n ≠ 0)
  (h_arith : arithmetic_sequence a d) :
  (∀ k, ∃ x, quadratic_equation a k x ∧ x = -1) ∧
  (∃ f : ℕ → ℝ, ∀ n, f (n + 1) - f n = -1/2 ∧
    ∃ x, quadratic_equation a n x ∧ f n = 1 / (x + 1)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_quadratic_roots_l657_65750


namespace NUMINAMATH_CALUDE_paint_wall_theorem_l657_65728

/-- The number of people needed to paint a wall in a given time, assuming a constant rate of painting. -/
def people_needed (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ) : ℕ :=
  (initial_people * initial_time) / new_time

/-- The additional number of people needed to paint a wall in a shorter time. -/
def additional_people_needed (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ) : ℕ :=
  (people_needed initial_people initial_time new_time) - initial_people

theorem paint_wall_theorem (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ) 
  (h1 : initial_people = 8) 
  (h2 : initial_time = 3) 
  (h3 : new_time = 2) :
  additional_people_needed initial_people initial_time new_time = 4 := by
  sorry

#check paint_wall_theorem

end NUMINAMATH_CALUDE_paint_wall_theorem_l657_65728


namespace NUMINAMATH_CALUDE_michael_work_days_l657_65704

-- Define the work rates for Michael, Adam, and Lisa
def M : ℚ := 1 / 40
def A : ℚ := 1 / 60
def L : ℚ := 1 / 60

-- Define the total work as 1 unit
def total_work : ℚ := 1

-- Theorem stating the conditions and the result to be proved
theorem michael_work_days :
  -- Condition 1: Michael, Adam, and Lisa can do the work together in 15 days
  M + A + L = 1 / 15 →
  -- Condition 2: After 10 days of working together, 2/3 of the work is completed
  (M + A + L) * 10 = 2 / 3 →
  -- Condition 3: Adam and Lisa complete the remaining 1/3 of the work in 8 days
  (A + L) * 8 = 1 / 3 →
  -- Conclusion: Michael takes 40 days to complete the work separately
  total_work / M = 40 :=
by sorry


end NUMINAMATH_CALUDE_michael_work_days_l657_65704


namespace NUMINAMATH_CALUDE_fraction_simplification_l657_65715

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l657_65715


namespace NUMINAMATH_CALUDE_distance_between_cities_l657_65733

/-- The distance between two cities given the speeds of two cars and their arrival time difference -/
theorem distance_between_cities (v_slow v_fast : ℝ) (time_diff : ℝ) : 
  v_slow = 72 →
  v_fast = 78 →
  time_diff = 1/3 →
  v_slow * (v_fast * time_diff / (v_fast - v_slow) + time_diff) = 312 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l657_65733


namespace NUMINAMATH_CALUDE_race_finish_times_l657_65712

/-- Represents the time difference at the finish line between two runners -/
def time_difference (distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  distance * (speed2 - speed1)

theorem race_finish_times (malcolm_speed joshua_speed alice_speed : ℝ) 
  (h1 : malcolm_speed = 5)
  (h2 : joshua_speed = 7)
  (h3 : alice_speed = 6)
  (race_distance : ℝ)
  (h4 : race_distance = 12) :
  time_difference race_distance malcolm_speed joshua_speed = 24 ∧
  time_difference race_distance malcolm_speed alice_speed = 12 :=
by sorry

end NUMINAMATH_CALUDE_race_finish_times_l657_65712


namespace NUMINAMATH_CALUDE_sum_of_cubes_l657_65739

theorem sum_of_cubes (a b c : ℝ) 
  (sum_condition : a + b + c = 3)
  (sum_of_products : a * b + a * c + b * c = 5)
  (product : a * b * c = -6) :
  a^3 + b^3 + c^3 = -36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l657_65739


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_equation_l657_65758

theorem min_sum_of_reciprocal_equation : 
  ∃ (x y z : ℕ+), 
    (1 : ℝ) / x + 4 / y + 9 / z = 1 ∧ 
    x + y + z = 36 ∧ 
    ∀ (a b c : ℕ+), (1 : ℝ) / a + 4 / b + 9 / c = 1 → a + b + c ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_equation_l657_65758


namespace NUMINAMATH_CALUDE_derivative_of_f_l657_65702

-- Define the function f(x) = (2 + x³)²
def f (x : ℝ) : ℝ := (2 + x^3)^2

-- State the theorem that the derivative of f(x) is 2(2 + x³) · 3x
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2 * (2 + x^3) * 3 * x := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l657_65702


namespace NUMINAMATH_CALUDE_divisible_by_nine_l657_65724

theorem divisible_by_nine (A : ℕ) : A < 10 → (83 * 1000 + A * 100 + 5) % 9 = 0 ↔ A = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l657_65724


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l657_65708

theorem range_of_a_minus_b (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 2) :
  ∀ x, -3 < x ∧ x < 0 ↔ ∃ a b, -1 < a ∧ a < b ∧ b < 2 ∧ x = a - b :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l657_65708


namespace NUMINAMATH_CALUDE_cindy_pen_addition_l657_65771

theorem cindy_pen_addition (initial_pens : ℕ) (mike_pens : ℕ) (sharon_pens : ℕ) (final_pens : ℕ)
  (h1 : initial_pens = 20)
  (h2 : mike_pens = 22)
  (h3 : sharon_pens = 19)
  (h4 : final_pens = 65) :
  final_pens - (initial_pens + mike_pens - sharon_pens) = 42 :=
by sorry

end NUMINAMATH_CALUDE_cindy_pen_addition_l657_65771


namespace NUMINAMATH_CALUDE_cos_alpha_value_l657_65722

theorem cos_alpha_value (α : Real) (h1 : α ∈ Set.Icc 0 (Real.pi / 2)) 
  (h2 : Real.sin (α - Real.pi / 6) = 3 / 5) : 
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l657_65722


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l657_65726

-- Define sets A and B
def A : Set ℝ := {x | x ≤ 7}
def B : Set ℝ := {x | x > 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x | 2 < x ∧ x ≤ 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l657_65726


namespace NUMINAMATH_CALUDE_berkeley_b_count_l657_65745

def abraham_total : ℕ := 20
def abraham_b : ℕ := 12
def berkeley_total : ℕ := 30

theorem berkeley_b_count : ℕ := by
  -- Define berkeley_b as the number of students in Mrs. Berkeley's class who received a 'B'
  -- Prove that berkeley_b = 18 given the conditions
  sorry

#check berkeley_b_count

end NUMINAMATH_CALUDE_berkeley_b_count_l657_65745


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l657_65790

/-- The ratio of the volume of a cone to the volume of a cylinder with specified dimensions -/
theorem cone_cylinder_volume_ratio :
  let cone_height : ℝ := 10
  let cylinder_height : ℝ := 30
  let radius : ℝ := 5
  let cone_volume := (1/3) * π * radius^2 * cone_height
  let cylinder_volume := π * radius^2 * cylinder_height
  cone_volume / cylinder_volume = 2/9 := by
sorry

end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l657_65790


namespace NUMINAMATH_CALUDE_complement_of_A_l657_65775

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem complement_of_A (x : ℝ) : x ∈ (U \ A) ↔ x < 1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l657_65775


namespace NUMINAMATH_CALUDE_log_expression_equals_three_halves_l657_65706

theorem log_expression_equals_three_halves :
  (Real.log (Real.sqrt 27) + Real.log 8 - 3 * Real.log (Real.sqrt 10)) / Real.log 1.2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_three_halves_l657_65706


namespace NUMINAMATH_CALUDE_line_points_relation_l657_65756

/-- Given a line x = 5y + 5 passing through points (a, n) and (a + 2, n + 0.4),
    prove that a = 5n + 5 -/
theorem line_points_relation (a n : ℝ) : 
  (a = 5 * n + 5 ∧ (a + 2) = 5 * (n + 0.4) + 5) → a = 5 * n + 5 := by
  sorry

end NUMINAMATH_CALUDE_line_points_relation_l657_65756


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_specific_values_l657_65784

theorem sqrt_equality_implies_specific_values (a b : ℕ) :
  0 < a → 0 < b → a < b →
  Real.sqrt (2 + Real.sqrt (45 + 20 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b →
  a = 2 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_specific_values_l657_65784


namespace NUMINAMATH_CALUDE_circle_area_tripled_l657_65769

theorem circle_area_tripled (r m : ℝ) (h : r > 0) (h' : m > 0) : 
  π * (r + m)^2 = 3 * (π * r^2) → r = (m * (1 + Real.sqrt 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l657_65769


namespace NUMINAMATH_CALUDE_wire_service_reporters_l657_65743

theorem wire_service_reporters (total : ℝ) 
  (country_x country_y country_z : ℝ)
  (xy_overlap yz_overlap xz_overlap xyz_overlap : ℝ)
  (finance environment social : ℝ)
  (h_total : total > 0)
  (h_x : country_x = 0.3 * total)
  (h_y : country_y = 0.2 * total)
  (h_z : country_z = 0.15 * total)
  (h_xy : xy_overlap = 0.05 * total)
  (h_yz : yz_overlap = 0.03 * total)
  (h_xz : xz_overlap = 0.02 * total)
  (h_xyz : xyz_overlap = 0.01 * total)
  (h_finance : finance = 0.1 * total)
  (h_environment : environment = 0.07 * total)
  (h_social : social = 0.05 * total) :
  (total - (country_x + country_y + country_z - xy_overlap - yz_overlap - xz_overlap + xyz_overlap) - 
   (finance + environment + social)) / total = 0.27 := by
sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l657_65743


namespace NUMINAMATH_CALUDE_count_positive_integer_solutions_l657_65780

/-- The number of positive integer solutions for the equation x + y + z + t = 15 -/
theorem count_positive_integer_solutions : 
  (Finset.filter (fun (x : ℕ × ℕ × ℕ × ℕ) => x.1 + x.2.1 + x.2.2.1 + x.2.2.2 = 15) 
    (Finset.product (Finset.range 15) (Finset.product (Finset.range 15) 
      (Finset.product (Finset.range 15) (Finset.range 15))))).card = 364 := by
  sorry

#check count_positive_integer_solutions

end NUMINAMATH_CALUDE_count_positive_integer_solutions_l657_65780


namespace NUMINAMATH_CALUDE_debby_jogging_distance_l657_65794

/-- Represents the number of kilometers Debby jogged on each day -/
structure JoggingDistance where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ

/-- Theorem stating that given the conditions, Debby jogged 5 kilometers on Tuesday -/
theorem debby_jogging_distance (d : JoggingDistance) 
  (h1 : d.monday = 2)
  (h2 : d.wednesday = 9)
  (h3 : d.monday + d.tuesday + d.wednesday = 16) :
  d.tuesday = 5 := by
  sorry

end NUMINAMATH_CALUDE_debby_jogging_distance_l657_65794


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l657_65755

/-- A function that checks if a natural number is a palindrome in a given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- A function that converts a natural number from one base to another. -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- A function that returns the number of digits of a natural number in a given base. -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∀ n : ℕ,
  (isPalindrome n 2 ∧ numDigits n 2 = 5) →
  (∃ b : ℕ, b > 2 ∧ isPalindrome (baseConvert n 2 b) b ∧ numDigits (baseConvert n 2 b) b = 3) →
  n ≥ 17 := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l657_65755


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l657_65727

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) 
  (first_part_time : ℝ) (first_part_speed : ℝ) :
  total_distance = 24 →
  total_time = 8 →
  first_part_time = 4 →
  first_part_speed = 4 →
  (total_distance - first_part_time * first_part_speed) / (total_time - first_part_time) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l657_65727


namespace NUMINAMATH_CALUDE_tan_half_angle_fourth_quadrant_l657_65791

theorem tan_half_angle_fourth_quadrant (α : Real) :
  (α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) →  -- α is in the fourth quadrant
  (Real.sin α + Real.cos α = 1 / 5) →               -- given condition
  Real.tan (α / 2) = -1 / 3 := by                   -- conclusion to prove
sorry


end NUMINAMATH_CALUDE_tan_half_angle_fourth_quadrant_l657_65791


namespace NUMINAMATH_CALUDE_expression_simplification_l657_65763

/-- Proves that the given expression simplifies to √3/3 when x = √3 + 1 -/
theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (((1 / (x - 2) + 1) / ((x^2 - 2*x + 1) / (x - 2))) : ℝ) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l657_65763


namespace NUMINAMATH_CALUDE_negation_of_proposition_l657_65793

theorem negation_of_proposition (p : Prop) : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 2 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 3*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l657_65793


namespace NUMINAMATH_CALUDE_area_of_five_arranged_triangles_l657_65716

/-- The area covered by five equilateral triangles arranged in a specific way -/
theorem area_of_five_arranged_triangles : 
  let side_length : ℝ := 2 * Real.sqrt 3
  let single_triangle_area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  let number_of_triangles : ℕ := 5
  let effective_triangles : ℝ := 4
  effective_triangles * single_triangle_area = 12 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_area_of_five_arranged_triangles_l657_65716


namespace NUMINAMATH_CALUDE_divisor_problem_l657_65742

theorem divisor_problem (n d k m : ℤ) : 
  n = k * d + 4 → 
  n + 15 = 5 * m + 4 → 
  d = 5 := by sorry

end NUMINAMATH_CALUDE_divisor_problem_l657_65742


namespace NUMINAMATH_CALUDE_rectangle_reconfiguration_l657_65705

/-- Given a 10 × 15 rectangle divided into two congruent polygons and reassembled into a new rectangle with length twice its width, the length of one side of the smaller rectangle formed by one of the polygons is 5√3. -/
theorem rectangle_reconfiguration (original_length original_width : ℝ)
  (new_length new_width z : ℝ) :
  original_length = 10 →
  original_width = 15 →
  original_length * original_width = new_length * new_width →
  new_length = 2 * new_width →
  z = new_length / 2 →
  z = 5 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_reconfiguration_l657_65705


namespace NUMINAMATH_CALUDE_cost_of_parts_l657_65796

-- Define the given values
def patent_cost : ℕ := 4500
def selling_price : ℕ := 180
def break_even_quantity : ℕ := 45

-- Define the theorem
theorem cost_of_parts :
  ∃ (parts_cost : ℕ),
    parts_cost + patent_cost = break_even_quantity * selling_price :=
by sorry

end NUMINAMATH_CALUDE_cost_of_parts_l657_65796


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l657_65788

/-- Given a circular seating arrangement where the 7th person is directly opposite the 27th person,
    prove that the total number of people in the circle is 40. -/
theorem circular_seating_arrangement (n : ℕ) : n = 40 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_arrangement_l657_65788


namespace NUMINAMATH_CALUDE_largest_B_for_divisibility_by_4_l657_65789

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def is_single_digit (n : ℕ) : Prop := n ≤ 9

def seven_digit_number (B X : ℕ) : ℕ := 4000000 + 100000 * B + 6000 + 792 * 10 + X

theorem largest_B_for_divisibility_by_4 :
  ∃ (B : ℕ), is_single_digit B ∧
  (∃ (X : ℕ), is_single_digit X ∧ is_divisible_by_4 (seven_digit_number B X)) ∧
  ∀ (B' : ℕ), is_single_digit B' →
    (∃ (X : ℕ), is_single_digit X ∧ is_divisible_by_4 (seven_digit_number B' X)) →
    B' ≤ B :=
by sorry

end NUMINAMATH_CALUDE_largest_B_for_divisibility_by_4_l657_65789


namespace NUMINAMATH_CALUDE_white_washing_cost_calculation_l657_65772

/-- Calculate the cost of white washing a room's walls --/
def white_washing_cost (room_length room_width room_height : ℝ)
                       (door_height door_width : ℝ)
                       (window_height window_width : ℝ)
                       (num_windows : ℕ)
                       (cost_per_sqft : ℝ) : ℝ :=
  let wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_height * door_width
  let window_area := num_windows * (window_height * window_width)
  let area_to_wash := wall_area - (door_area + window_area)
  area_to_wash * cost_per_sqft

/-- Theorem stating the cost of white washing the room --/
theorem white_washing_cost_calculation :
  white_washing_cost 25 15 12 6 3 4 3 3 4 = 3624 := by
  sorry

end NUMINAMATH_CALUDE_white_washing_cost_calculation_l657_65772


namespace NUMINAMATH_CALUDE_clock_hands_angle_at_1_10_clock_hands_angle_at_1_10_is_25_l657_65703

/-- The angle between clock hands at 1:10 -/
theorem clock_hands_angle_at_1_10 : ℝ := by
  -- Define constants
  let total_hours : ℕ := 12
  let total_degrees : ℝ := 360
  let minutes_passed : ℕ := 10

  -- Define speeds (degrees per minute)
  let hour_hand_speed : ℝ := total_degrees / (total_hours * 60)
  let minute_hand_speed : ℝ := total_degrees / 60

  -- Define initial positions at 1:00
  let initial_hour_hand_position : ℝ := 30
  let initial_minute_hand_position : ℝ := 0

  -- Calculate final positions at 1:10
  let final_hour_hand_position : ℝ := initial_hour_hand_position + hour_hand_speed * minutes_passed
  let final_minute_hand_position : ℝ := initial_minute_hand_position + minute_hand_speed * minutes_passed

  -- Calculate the angle between hands
  let angle_between_hands : ℝ := final_minute_hand_position - final_hour_hand_position

  -- Prove that the angle is 25°
  sorry

/-- The theorem states that the angle between the hour and minute hands at 1:10 is 25° -/
theorem clock_hands_angle_at_1_10_is_25 : clock_hands_angle_at_1_10 = 25 := by
  sorry

end NUMINAMATH_CALUDE_clock_hands_angle_at_1_10_clock_hands_angle_at_1_10_is_25_l657_65703


namespace NUMINAMATH_CALUDE_ratio_problem_l657_65719

theorem ratio_problem (second_part : ℝ) (percent : ℝ) (first_part : ℝ) : 
  second_part = 5 →
  percent = 120 →
  first_part / second_part = percent / 100 →
  first_part = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l657_65719


namespace NUMINAMATH_CALUDE_elevator_exit_theorem_l657_65713

/-- The number of ways passengers can exit an elevator -/
def elevator_exit_ways (num_passengers : ℕ) (total_floors : ℕ) (start_floor : ℕ) : ℕ :=
  (total_floors - start_floor + 1) ^ num_passengers

/-- Theorem: 6 passengers exiting an elevator in a 12-story building starting from the 3rd floor -/
theorem elevator_exit_theorem :
  elevator_exit_ways 6 12 3 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_elevator_exit_theorem_l657_65713


namespace NUMINAMATH_CALUDE_sequence_sum_l657_65781

def geometric_sequence (a : ℕ → ℚ) (r : ℚ) :=
  ∀ n, a (n + 1) = r * a n

theorem sequence_sum (a : ℕ → ℚ) (r : ℚ) :
  geometric_sequence a r →
  a 0 = 16384 →
  a 5 = 16 →
  r = 1/4 →
  a 3 + a 4 = 320 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l657_65781


namespace NUMINAMATH_CALUDE_cubic_function_max_value_l657_65761

/-- Given a cubic function f with a known minimum value on an interval,
    prove that its maximum value on the same interval is 43. -/
theorem cubic_function_max_value (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^3 - 6 * x^2 + a
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f x ≤ f y) →
  (∃ x ∈ Set.Icc (-2) 2, f x = 3) →
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f y ≤ f x ∧ f x = 43) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_max_value_l657_65761


namespace NUMINAMATH_CALUDE_equal_coefficients_implies_n_seven_l657_65725

theorem equal_coefficients_implies_n_seven (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
sorry

end NUMINAMATH_CALUDE_equal_coefficients_implies_n_seven_l657_65725


namespace NUMINAMATH_CALUDE_range_of_s_l657_65777

/-- A decreasing function with central symmetry property -/
def DecreasingSymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x > f y) ∧
  (∀ x, f (x - 1) = -f (2 - x))

/-- The main theorem -/
theorem range_of_s (f : ℝ → ℝ) (h : DecreasingSymmetricFunction f) :
  ∀ s : ℝ, f (s^2 - 2*s) + f (2 - s) ≤ 0 → s ≤ 1 ∨ s ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_s_l657_65777


namespace NUMINAMATH_CALUDE_triangle_existence_l657_65766

theorem triangle_existence (x : ℤ) : 
  (5 + x > 0) ∧ (2*x + 1 > 0) ∧ (3*x > 0) ∧
  (5 + x + 2*x + 1 > 3*x) ∧ (5 + x + 3*x > 2*x + 1) ∧ (2*x + 1 + 3*x > 5 + x) ↔ 
  x ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_existence_l657_65766


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l657_65741

theorem quadratic_inequality_condition (x : ℝ) : 
  2 * x^2 - 5 * x - 3 ≥ 0 ↔ x ≤ -1/2 ∨ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l657_65741


namespace NUMINAMATH_CALUDE_smallest_multiple_exceeding_100_l657_65721

theorem smallest_multiple_exceeding_100 : ∃ (n : ℕ), 
  n > 0 ∧ 
  n % 45 = 0 ∧ 
  (n - 100) % 7 = 0 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m % 45 = 0 ∧ (m - 100) % 7 = 0 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_exceeding_100_l657_65721


namespace NUMINAMATH_CALUDE_matrix_N_property_l657_65732

theorem matrix_N_property :
  ∃ (N : Matrix (Fin 3) (Fin 3) ℝ),
    (∀ (u : Fin 3 → ℝ), N.mulVec u = (3 : ℝ) • u) ∧
    N = !![3, 0, 0; 0, 3, 0; 0, 0, 3] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_property_l657_65732


namespace NUMINAMATH_CALUDE_henry_actual_earnings_l657_65734

/-- Represents Henry's summer job earnings --/
def HenryEarnings : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun (lawn_rate pile_rate driveway_rate lawns_mowed piles_raked driveways_shoveled : ℕ) =>
    lawn_rate * lawns_mowed + pile_rate * piles_raked + driveway_rate * driveways_shoveled

/-- Theorem stating Henry's actual earnings --/
theorem henry_actual_earnings :
  HenryEarnings 5 10 15 5 3 2 = 85 := by
  sorry

end NUMINAMATH_CALUDE_henry_actual_earnings_l657_65734


namespace NUMINAMATH_CALUDE_yarn_length_difference_l657_65748

theorem yarn_length_difference (green_length red_length : ℝ) : 
  green_length = 156 →
  red_length > 3 * green_length →
  green_length + red_length = 632 →
  red_length - 3 * green_length = 8 := by
sorry

end NUMINAMATH_CALUDE_yarn_length_difference_l657_65748


namespace NUMINAMATH_CALUDE_x29x_divisible_by_18_l657_65782

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def four_digit_number (x : ℕ) : ℕ := 1000 * x + 290 + x

theorem x29x_divisible_by_18 :
  ∃! x : ℕ, is_single_digit x ∧ (four_digit_number x) % 18 = 0 ∧ x = 2 :=
sorry

end NUMINAMATH_CALUDE_x29x_divisible_by_18_l657_65782


namespace NUMINAMATH_CALUDE_solution_set_implies_a_equals_two_existence_implies_m_greater_than_five_l657_65779

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem 1
theorem solution_set_implies_a_equals_two :
  (∀ x, f 2 x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) →
  (∃ a, ∀ x, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) →
  (∀ a, (∀ x, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2) :=
sorry

-- Theorem 2
theorem existence_implies_m_greater_than_five :
  (∃ x m, f 2 x + f 2 (x + 5) < m) →
  (∀ m, (∃ x, f 2 x + f 2 (x + 5) < m) → m > 5) :=
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_equals_two_existence_implies_m_greater_than_five_l657_65779


namespace NUMINAMATH_CALUDE_point_on_circle_l657_65717

/-- The coordinates of a point on the unit circle after moving counterclockwise from (1, 0) by an arc length of 2π/3 -/
theorem point_on_circle (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  P = (1, 0) → 
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = (2 * Real.pi / 3)^2 →
  Q.1^2 + Q.2^2 = 1 →
  Q = (-1/2, Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_point_on_circle_l657_65717


namespace NUMINAMATH_CALUDE_james_club_night_cost_l657_65762

def club_night_cost (entry_fee : ℚ) (friend_rounds : ℕ) (friends : ℕ) 
  (self_drinks : ℕ) (cocktail_price : ℚ) (non_alcoholic_price : ℚ) 
  (cocktail_discount : ℚ) (cocktails_bought : ℕ) (burger_price : ℚ) 
  (fries_price : ℚ) (food_tip_rate : ℚ) (drink_tip_rate : ℚ) : ℚ :=
  let total_drinks := friend_rounds * friends + self_drinks
  let non_alcoholic_drinks := total_drinks - cocktails_bought
  let cocktail_cost := cocktails_bought * cocktail_price
  let discounted_cocktail_cost := 
    if cocktails_bought ≥ 3 then cocktail_cost * (1 - cocktail_discount) else cocktail_cost
  let non_alcoholic_cost := non_alcoholic_drinks * non_alcoholic_price
  let food_cost := burger_price + fries_price
  let food_tip := food_cost * food_tip_rate
  let drink_tip := (cocktail_cost + non_alcoholic_cost) * drink_tip_rate
  entry_fee + discounted_cocktail_cost + non_alcoholic_cost + food_cost + food_tip + drink_tip

theorem james_club_night_cost :
  club_night_cost 30 3 10 8 10 5 0.2 7 20 8 0.2 0.15 = 308.35 := by
  sorry

end NUMINAMATH_CALUDE_james_club_night_cost_l657_65762


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_equals_sqrt29_over_4_l657_65701

theorem sqrt_sum_fractions_equals_sqrt29_over_4 :
  Real.sqrt (9 / 36 + 25 / 16) = Real.sqrt 29 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_equals_sqrt29_over_4_l657_65701


namespace NUMINAMATH_CALUDE_parabola_min_area_sum_l657_65714

/-- A parabola in the Cartesian plane -/
structure Parabola where
  eqn : ℝ → ℝ → Prop

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- A point lies on a parabola -/
def lies_on (p : Parabola) (point : ℝ × ℝ) : Prop := sorry

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given three points -/
def triangle_area (p q r : ℝ × ℝ) : ℝ := sorry

theorem parabola_min_area_sum (p : Parabola) (A B : ℝ × ℝ) :
  p.eqn = fun x y ↦ y^2 = 2*x →
  lies_on p A →
  lies_on p B →
  dot_product A B = -1 →
  let F := focus p
  let O := (0, 0)
  ∃ (min : ℝ), min = Real.sqrt 2 / 2 ∧
    ∀ (X Y : ℝ × ℝ), lies_on p X → lies_on p Y → dot_product X Y = -1 →
      triangle_area O F X + triangle_area O F Y ≥ min :=
sorry

end NUMINAMATH_CALUDE_parabola_min_area_sum_l657_65714


namespace NUMINAMATH_CALUDE_complex_ratio_pure_imaginary_l657_65752

theorem complex_ratio_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 3*I
  let z₂ : ℂ := 3 + 4*I
  (∃ (b : ℝ), z₁ / z₂ = b*I) → a = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_ratio_pure_imaginary_l657_65752


namespace NUMINAMATH_CALUDE_merry_go_round_duration_frank_duration_l657_65710

theorem merry_go_round_duration (dave_duration : ℕ) (chuck_factor : ℕ) (erica_percent : ℕ) (frank_percent : ℕ) : ℕ :=
  let chuck_duration := dave_duration * chuck_factor
  let erica_duration := chuck_duration + chuck_duration * erica_percent / 100
  let frank_duration := erica_duration + erica_duration * frank_percent / 100
  frank_duration

/-- Frank's duration on the merry-go-round -/
theorem frank_duration : merry_go_round_duration 10 5 30 20 = 78 := by
  sorry

end NUMINAMATH_CALUDE_merry_go_round_duration_frank_duration_l657_65710


namespace NUMINAMATH_CALUDE_face_mask_profit_l657_65737

/-- Calculates the total profit from selling face masks --/
def calculate_profit (num_boxes : ℕ) (price_per_mask : ℚ) (masks_per_box : ℕ) (total_cost : ℚ) : ℚ :=
  num_boxes * price_per_mask * masks_per_box - total_cost

/-- Proves that the total profit is $15 given the specified conditions --/
theorem face_mask_profit :
  let num_boxes : ℕ := 3
  let price_per_mask : ℚ := 1/2
  let masks_per_box : ℕ := 20
  let total_cost : ℚ := 15
  calculate_profit num_boxes price_per_mask masks_per_box total_cost = 15 := by
  sorry

#eval calculate_profit 3 (1/2) 20 15

end NUMINAMATH_CALUDE_face_mask_profit_l657_65737


namespace NUMINAMATH_CALUDE_rice_yield_comparison_l657_65759

/-- Represents the rice field contract information -/
structure RiceContract where
  acres : ℕ
  yieldPerAcre : ℕ

/-- Calculates the total yield for a given contract -/
def totalYield (contract : RiceContract) : ℕ :=
  contract.acres * contract.yieldPerAcre

theorem rice_yield_comparison 
  (uncleLi : RiceContract)
  (auntLin : RiceContract)
  (h1 : uncleLi.acres = 12)
  (h2 : uncleLi.yieldPerAcre = 660)
  (h3 : auntLin.acres = uncleLi.acres - 2)
  (h4 : totalYield auntLin = totalYield uncleLi - 420) :
  totalYield uncleLi = 7920 ∧ 
  uncleLi.yieldPerAcre + 90 = auntLin.yieldPerAcre :=
by sorry

end NUMINAMATH_CALUDE_rice_yield_comparison_l657_65759


namespace NUMINAMATH_CALUDE_octal_to_decimal_conversion_l657_65707

-- Define the octal number
def octal_age : ℕ := 536

-- Define the decimal equivalent
def decimal_age : ℕ := 350

-- Theorem to prove the equivalence
theorem octal_to_decimal_conversion :
  (5 * 8^2 + 3 * 8^1 + 6 * 8^0) = decimal_age :=
by sorry

end NUMINAMATH_CALUDE_octal_to_decimal_conversion_l657_65707


namespace NUMINAMATH_CALUDE_three_oclock_angle_l657_65768

/-- The angle between the hour hand and minute hand at a given time -/
def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  sorry

theorem three_oclock_angle :
  clock_angle 3 0 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_three_oclock_angle_l657_65768


namespace NUMINAMATH_CALUDE_jesse_stamp_collection_l657_65718

theorem jesse_stamp_collection (total : ℕ) (european : ℕ) (asian : ℕ) 
  (h1 : total = 444)
  (h2 : european = 3 * asian)
  (h3 : total = european + asian) :
  european = 333 := by
sorry

end NUMINAMATH_CALUDE_jesse_stamp_collection_l657_65718


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l657_65776

theorem arithmetic_expression_equality : 1874 + 230 / 46 - 874 * 2 = 131 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l657_65776


namespace NUMINAMATH_CALUDE_other_communities_count_l657_65783

/-- The number of students belonging to other communities in a school with given demographics -/
theorem other_communities_count (total : ℕ) (muslim hindu sikh buddhist christian jew : ℚ) 
  (h_total : total = 2500)
  (h_muslim : muslim = 28/100)
  (h_hindu : hindu = 26/100)
  (h_sikh : sikh = 12/100)
  (h_buddhist : buddhist = 10/100)
  (h_christian : christian = 6/100)
  (h_jew : jew = 4/100) :
  ↑total * (1 - (muslim + hindu + sikh + buddhist + christian + jew)) = 350 :=
by sorry

end NUMINAMATH_CALUDE_other_communities_count_l657_65783


namespace NUMINAMATH_CALUDE_chord_minimum_value_l657_65738

theorem chord_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let line := {p : ℝ × ℝ | a * p.1 - b * p.2 + 2 = 0}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - 4*p.2 + 1 = 0}
  let chord_length := 4
  (∃ (p q : ℝ × ℝ), p ∈ line ∧ q ∈ line ∧ p ∈ circle ∧ q ∈ circle ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) →
  (2/a + 3/b ≥ 4 + 2 * Real.sqrt 3) ∧ 
  (∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ 2/a' + 3/b' = 4 + 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_chord_minimum_value_l657_65738


namespace NUMINAMATH_CALUDE_mrs_hilt_pencil_purchase_l657_65778

/-- Given Mrs. Hilt's purchases at the school store, prove the number of pencils she bought. -/
theorem mrs_hilt_pencil_purchase
  (total_spent : ℕ)
  (notebook_cost : ℕ)
  (ruler_cost : ℕ)
  (pencil_cost : ℕ)
  (h1 : total_spent = 74)
  (h2 : notebook_cost = 35)
  (h3 : ruler_cost = 18)
  (h4 : pencil_cost = 7)
  : (total_spent - notebook_cost - ruler_cost) / pencil_cost = 3 :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_pencil_purchase_l657_65778


namespace NUMINAMATH_CALUDE_rectangular_parking_lot_perimeter_l657_65700

theorem rectangular_parking_lot_perimeter
  (diagonal : ℝ)
  (area : ℝ)
  (h_diagonal : diagonal = 28)
  (h_area : area = 180) :
  ∃ (length width : ℝ),
    length > 0 ∧
    width > 0 ∧
    length * width = area ∧
    length^2 + width^2 = diagonal^2 ∧
    2 * (length + width) = 68 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_parking_lot_perimeter_l657_65700


namespace NUMINAMATH_CALUDE_circle_equation_l657_65747

/-- A circle C with points A and B, and a chord intercepted by a line --/
structure CircleWithPointsAndChord where
  -- Center of the circle
  center : ℝ × ℝ
  -- Radius of the circle
  radius : ℝ
  -- Point A on the circle
  pointA : ℝ × ℝ
  -- Point B on the circle
  pointB : ℝ × ℝ
  -- Length of the chord intercepted by the line x-y-2=0
  chordLength : ℝ
  -- Ensure A and B are on the circle
  h_pointA_on_circle : (pointA.1 - center.1)^2 + (pointA.2 - center.2)^2 = radius^2
  h_pointB_on_circle : (pointB.1 - center.1)^2 + (pointB.2 - center.2)^2 = radius^2
  -- Ensure the chord length is correct
  h_chord_length : chordLength = Real.sqrt 2

/-- The theorem stating that the circle satisfying the given conditions has the equation (x-1)² + y² = 1 --/
theorem circle_equation (c : CircleWithPointsAndChord) 
  (h_pointA : c.pointA = (1, 1)) 
  (h_pointB : c.pointB = (2, 0)) :
  c.center = (1, 0) ∧ c.radius = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l657_65747


namespace NUMINAMATH_CALUDE_focus_coordinates_min_distance_midpoint_perpendicular_product_l657_65795

-- Define the parabola C: x^2 = 4y
def C (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus F
def F : ℝ × ℝ := (0, 1)

-- Define points A and B on the parabola
def A (x₁ y₁ : ℝ) : Prop := C x₁ y₁
def B (x₂ y₂ : ℝ) : Prop := C x₂ y₂

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem statements
theorem focus_coordinates : F = (0, 1) := by sorry

theorem min_distance_midpoint (x₁ y₁ x₂ y₂ : ℝ) (hA : A x₁ y₁) (hB : B x₂ y₂) :
  let d := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)
  d = 8 → (((y₁ + y₂) / 2) - 0) ≥ 3 := by sorry

theorem perpendicular_product (x₁ y₁ x₂ y₂ : ℝ) (hA : A x₁ y₁) (hB : B x₂ y₂) :
  (x₁ * x₂ + y₁ * y₂ = 0) → 
  (Real.sqrt (x₁^2 + y₁^2)) * (Real.sqrt (x₂^2 + y₂^2)) ≥ 32 := by sorry

end NUMINAMATH_CALUDE_focus_coordinates_min_distance_midpoint_perpendicular_product_l657_65795


namespace NUMINAMATH_CALUDE_equal_roots_condition_l657_65774

theorem equal_roots_condition (m : ℝ) : 
  (∃! x : ℝ, (x * (x - 2) - (m + 2)) / ((x - 2) * (m - 2)) = (x + 1) / (m + 1)) ↔ 
  (m = -1 ∨ m = -5) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l657_65774


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l657_65751

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 5 ∧ min = -15 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l657_65751


namespace NUMINAMATH_CALUDE_min_omega_value_l657_65740

open Real

theorem min_omega_value (ω φ : ℝ) (f : ℝ → ℝ) : 
  ω > 0 → 
  abs φ < π / 2 →
  (∀ x, f x = sin (ω * x + φ)) →
  f 0 = 1 / 2 →
  (∀ x, f x ≤ f (π / 12)) →
  (∀ ω' > 0, (∀ x, sin (ω' * x + φ) ≤ sin (ω' * π / 12 + φ)) → ω' ≥ ω) →
  ω = 4 := by
sorry

end NUMINAMATH_CALUDE_min_omega_value_l657_65740


namespace NUMINAMATH_CALUDE_arcsin_sin_eq_x_div_3_solutions_l657_65729

theorem arcsin_sin_eq_x_div_3_solutions (x : ℝ) :
  -((3 * π) / 2) ≤ x ∧ x ≤ (3 * π) / 2 →
  (Real.arcsin (Real.sin x) = x / 3) ↔ 
  x ∈ ({-3*π, -2*π, -π, 0, π, 2*π, 3*π} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_sin_eq_x_div_3_solutions_l657_65729


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l657_65730

theorem roots_of_quadratic (x : ℝ) : x^2 = 2*x ↔ x = 0 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l657_65730


namespace NUMINAMATH_CALUDE_trigonometric_and_algebraic_identities_l657_65746

theorem trigonometric_and_algebraic_identities :
  (2 * Real.sin (45 * π / 180) ^ 2 + Real.tan (60 * π / 180) * Real.tan (30 * π / 180) - Real.cos (60 * π / 180) = 3 / 2) ∧
  (Real.sqrt 12 - 2 * Real.cos (30 * π / 180) + (3 - Real.pi) ^ 0 + |1 - Real.sqrt 3| = 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_and_algebraic_identities_l657_65746


namespace NUMINAMATH_CALUDE_angle_through_point_l657_65720

/-- 
Given an angle α whose terminal side passes through point P(1, 2) in a plane coordinate system,
prove that:
1. tan α = 2
2. (sin α + 2 cos α) / (2 sin α - cos α) = 4/3
-/
theorem angle_through_point (α : Real) : 
  (∃ (x y : Real), x = 1 ∧ y = 2 ∧ x = Real.cos α ∧ y = Real.sin α) →
  Real.tan α = 2 ∧ (Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_through_point_l657_65720


namespace NUMINAMATH_CALUDE_toothpicks_150th_stage_l657_65797

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ := 4 + 4 * (n - 1)

/-- Theorem: The 150th stage of the pattern contains 600 toothpicks -/
theorem toothpicks_150th_stage : toothpicks 150 = 600 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_150th_stage_l657_65797


namespace NUMINAMATH_CALUDE_stock_price_decrease_l657_65760

theorem stock_price_decrease (a : ℝ) (n : ℕ) (h₁ : a > 0) : a * (0.99 ^ n) < a := by
  sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l657_65760


namespace NUMINAMATH_CALUDE_no_valid_score_l657_65764

/-- Represents a player in the hockey match -/
inductive Player
| Anton
| Ilya
| Sergey

/-- Represents the statements made by each player -/
def Statement : Type := Player → ℕ

/-- The statements made by Anton -/
def AntonStatement : Statement :=
  fun p => match p with
  | Player.Anton => 3
  | Player.Ilya => 1
  | Player.Sergey => 0

/-- The statements made by Ilya -/
def IlyaStatement : Statement :=
  fun p => match p with
  | Player.Anton => 0
  | Player.Ilya => 4
  | Player.Sergey => 5

/-- The statements made by Sergey -/
def SergeyStatement : Statement :=
  fun p => match p with
  | Player.Anton => 2
  | Player.Ilya => 0
  | Player.Sergey => 6

/-- Checks if a given score satisfies the conditions -/
def satisfiesConditions (score : Player → ℕ) : Prop :=
  (score Player.Anton + score Player.Ilya + score Player.Sergey = 10) ∧
  (∃ (p : Player), AntonStatement p = score p) ∧
  (∃ (p : Player), AntonStatement p ≠ score p) ∧
  (∃ (p : Player), IlyaStatement p = score p) ∧
  (∃ (p : Player), IlyaStatement p ≠ score p) ∧
  (∃ (p : Player), SergeyStatement p = score p) ∧
  (∃ (p : Player), SergeyStatement p ≠ score p)

/-- Theorem stating that no score satisfies all conditions -/
theorem no_valid_score : ¬∃ (score : Player → ℕ), satisfiesConditions score := by
  sorry


end NUMINAMATH_CALUDE_no_valid_score_l657_65764


namespace NUMINAMATH_CALUDE_negation_of_proposition_l657_65711

theorem negation_of_proposition :
  (∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) →
  (¬(∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) ↔ ∃ x : ℝ, 2^x - 2*x - 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l657_65711


namespace NUMINAMATH_CALUDE_endomorphism_characterization_l657_65798

/-- An endomorphism of ℤ² --/
def Endomorphism : Type := ℤ × ℤ → ℤ × ℤ

/-- The group operation on ℤ² --/
def add : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  λ (a b : ℤ × ℤ) => (a.1 + b.1, a.2 + b.2)

/-- A homomorphism respects the group operation --/
def is_homomorphism (φ : Endomorphism) : Prop :=
  ∀ a b : ℤ × ℤ, φ (add a b) = add (φ a) (φ b)

/-- Linear representation of an endomorphism --/
def linear_form (u v : ℤ × ℤ) : Endomorphism :=
  λ (x : ℤ × ℤ) => (x.1 * u.1 + x.2 * v.1, x.1 * u.2 + x.2 * v.2)

/-- Main theorem: Characterization of endomorphisms of ℤ² --/
theorem endomorphism_characterization :
  ∀ φ : Endomorphism, 
    is_homomorphism φ ↔ ∃ u v : ℤ × ℤ, φ = linear_form u v :=
by sorry

end NUMINAMATH_CALUDE_endomorphism_characterization_l657_65798
