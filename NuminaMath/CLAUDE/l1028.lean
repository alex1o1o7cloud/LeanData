import Mathlib

namespace NUMINAMATH_CALUDE_circle_equation_with_given_diameter_l1028_102847

/-- The standard equation of a circle with diameter endpoints A(-1, 2) and B(5, -6) -/
theorem circle_equation_with_given_diameter :
  ∃ (f : ℝ × ℝ → ℝ),
    (∀ x y : ℝ, f (x, y) = (x - 2)^2 + (y + 2)^2) ∧
    (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | f p = 25} ↔ 
      ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
        x = -1 + 6*t ∧ 
        y = 2 - 8*t) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_with_given_diameter_l1028_102847


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l1028_102883

/-- Proves that the given function satisfies the differential equation -/
theorem function_satisfies_equation (x a : ℝ) :
  let y := a + (7 * x) / (a * x + 1)
  let y' := 7 / ((a * x + 1) ^ 2)
  y - x * y' = a * (1 + x^2 * y') := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l1028_102883


namespace NUMINAMATH_CALUDE_marks_vaccine_wait_l1028_102862

/-- Theorem: Mark's wait for first vaccine appointment
Given:
- The total waiting time is 38 days
- There's a 20-day wait between appointments
- There's a 14-day wait for full effectiveness after the second appointment
Prove: The wait for the first appointment is 4 days
-/
theorem marks_vaccine_wait (total_wait : ℕ) (between_appointments : ℕ) (full_effectiveness : ℕ) :
  total_wait = 38 →
  between_appointments = 20 →
  full_effectiveness = 14 →
  total_wait = between_appointments + full_effectiveness + 4 :=
by sorry

end NUMINAMATH_CALUDE_marks_vaccine_wait_l1028_102862


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_l1028_102869

/-- Given an ellipse with equation x²/(8-m) + y²/(m-2) = 1, 
    where the major axis is on the y-axis and the focal distance is 4,
    prove that the value of m is 7. -/
theorem ellipse_focal_distance (m : ℝ) : 
  (∀ x y : ℝ, x^2 / (8 - m) + y^2 / (m - 2) = 1) →  -- Ellipse equation
  (8 - m < m - 2) →                                -- Major axis on y-axis
  (m - 2 - (8 - m) = 4) →                          -- Focal distance is 4
  m = 7 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focal_distance_l1028_102869


namespace NUMINAMATH_CALUDE_second_divisor_problem_l1028_102876

theorem second_divisor_problem (initial : ℝ) (first_divisor : ℝ) (final_result : ℝ) (x : ℝ) :
  initial = 8900 →
  first_divisor = 6 →
  final_result = 370.8333333333333 →
  (initial / first_divisor) / x = final_result →
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l1028_102876


namespace NUMINAMATH_CALUDE_constant_for_max_n_l1028_102800

theorem constant_for_max_n (c : ℝ) : 
  (∀ n : ℤ, c * n^2 ≤ 6400 → n ≤ 7) ∧ 
  (∃ n : ℤ, c * n^2 ≤ 6400 ∧ n = 7) →
  c = 6400 / 49 :=
sorry

end NUMINAMATH_CALUDE_constant_for_max_n_l1028_102800


namespace NUMINAMATH_CALUDE_max_k_inequality_k_max_is_tight_l1028_102877

theorem max_k_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ∀ k : ℝ, k ≤ 174960 →
    (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2*d)^5) ≥ k * a * b * c * d^3 :=
by sorry

theorem k_max_is_tight :
  ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2*d)^5) = 174960 * a * b * c * d^3 :=
by sorry

end NUMINAMATH_CALUDE_max_k_inequality_k_max_is_tight_l1028_102877


namespace NUMINAMATH_CALUDE_quadratic_complex_roots_l1028_102896

theorem quadratic_complex_roots : ∃ (z₁ z₂ : ℂ),
  z₁ = (3 + Real.sqrt 14) / 2 + Complex.I * Real.sqrt 14 / 7 ∧
  z₂ = (3 - Real.sqrt 14) / 2 - Complex.I * Real.sqrt 14 / 7 ∧
  z₁^2 - 3*z₁ + 2 = 3 - 2*Complex.I ∧
  z₂^2 - 3*z₂ + 2 = 3 - 2*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complex_roots_l1028_102896


namespace NUMINAMATH_CALUDE_subset_implies_a_leq_two_l1028_102887

def A : Set ℝ := {x | x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

theorem subset_implies_a_leq_two (a : ℝ) (h : A ⊆ B a) : a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_leq_two_l1028_102887


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1028_102852

theorem unique_integer_solution : ∃! n : ℤ, n + 15 > 16 ∧ -3*n > -9 :=
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1028_102852


namespace NUMINAMATH_CALUDE_cubic_roots_proof_l1028_102837

theorem cubic_roots_proof (k : ℝ) (p q r : ℝ) : 
  (2 * p^3 + k * p^2 - 6 * p - 3 = 0) →
  (p = 3) →
  (p + q + r = 5) →
  (p * q * r = -6) →
  ({q, r} : Set ℝ) = {1 + Real.sqrt 3, 1 - Real.sqrt 3} :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_proof_l1028_102837


namespace NUMINAMATH_CALUDE_tangent_roots_sum_l1028_102810

theorem tangent_roots_sum (α β : Real) :
  (∃ (x y : Real), x^2 + 3 * Real.sqrt 3 * x + 4 = 0 ∧ y^2 + 3 * Real.sqrt 3 * y + 4 = 0 ∧ 
   x = Real.tan α ∧ y = Real.tan β) →
  α > -π/2 ∧ α < π/2 ∧ β > -π/2 ∧ β < π/2 →
  α + β = -2*π/3 := by
sorry

end NUMINAMATH_CALUDE_tangent_roots_sum_l1028_102810


namespace NUMINAMATH_CALUDE_line_at_0_l1028_102805

/-- A line parameterized by t -/
def line (t : ℝ) : ℝ × ℝ := sorry

/-- The vector on the line at t = 1 is (2, 3) -/
axiom line_at_1 : line 1 = (2, 3)

/-- The vector on the line at t = 4 is (8, -5) -/
axiom line_at_4 : line 4 = (8, -5)

/-- The vector on the line at t = 5 is (10, -9) -/
axiom line_at_5 : line 5 = (10, -9)

/-- The vector on the line at t = 0 is (0, 17/3) -/
theorem line_at_0 : line 0 = (0, 17/3) := by sorry

end NUMINAMATH_CALUDE_line_at_0_l1028_102805


namespace NUMINAMATH_CALUDE_discount_difference_l1028_102854

def original_bill : ℝ := 12000

def single_discount (bill : ℝ) : ℝ := bill * 0.7

def successive_discounts (bill : ℝ) : ℝ := bill * 0.75 * 0.95

theorem discount_difference :
  successive_discounts original_bill - single_discount original_bill = 150 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l1028_102854


namespace NUMINAMATH_CALUDE_smallest_area_of_2020th_square_l1028_102898

theorem smallest_area_of_2020th_square :
  ∀ (n : ℕ),
  (∃ (a : ℕ), n^2 = 2019 + a ∧ a ≠ 1) →
  (∀ (a : ℕ), n^2 = 2019 + a ∧ a ≠ 1 → a ≥ 112225) :=
by sorry

end NUMINAMATH_CALUDE_smallest_area_of_2020th_square_l1028_102898


namespace NUMINAMATH_CALUDE_tina_postcards_per_day_l1028_102857

/-- The number of postcards Tina can make in a day -/
def postcards_per_day : ℕ := 30

/-- The price of each postcard in dollars -/
def price_per_postcard : ℕ := 5

/-- The number of days Tina sold postcards -/
def days_sold : ℕ := 6

/-- The total amount earned in dollars -/
def total_earned : ℕ := 900

theorem tina_postcards_per_day :
  postcards_per_day * price_per_postcard * days_sold = total_earned :=
by sorry

end NUMINAMATH_CALUDE_tina_postcards_per_day_l1028_102857


namespace NUMINAMATH_CALUDE_joans_attendance_l1028_102860

/-- The number of football games Joan attended -/
structure FootballAttendance where
  total : ℕ
  lastYear : ℕ
  thisYear : ℕ

/-- Theorem stating that Joan's attendance this year is 4 games -/
theorem joans_attendance (joan : FootballAttendance) 
  (h1 : joan.total = 13)
  (h2 : joan.lastYear = 9)
  (h3 : joan.total = joan.lastYear + joan.thisYear) :
  joan.thisYear = 4 := by
  sorry

end NUMINAMATH_CALUDE_joans_attendance_l1028_102860


namespace NUMINAMATH_CALUDE_stuart_initial_marbles_l1028_102861

def betty_initial_marbles : ℕ := 150
def tom_initial_marbles : ℕ := 30
def susan_initial_marbles : ℕ := 20
def stuart_final_marbles : ℕ := 80

def marbles_to_tom : ℕ := (betty_initial_marbles * 20) / 100
def marbles_to_susan : ℕ := (betty_initial_marbles * 10) / 100
def marbles_to_stuart : ℕ := (betty_initial_marbles * 40) / 100

theorem stuart_initial_marbles :
  stuart_final_marbles - marbles_to_stuart = 20 :=
by sorry

end NUMINAMATH_CALUDE_stuart_initial_marbles_l1028_102861


namespace NUMINAMATH_CALUDE_f_properties_l1028_102865

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 + 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x

-- Theorem stating the properties of f(x)
theorem f_properties :
  (f 0 = 1) ∧ 
  (f' 1 = 1) ∧
  (∀ x : ℝ, f x ≤ 1) ∧
  (f 0 = 1) ∧
  (∀ x : ℝ, f x ≥ 23/27) ∧
  (f (2/3) = 23/27) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1028_102865


namespace NUMINAMATH_CALUDE_unique_solution_system_l1028_102853

theorem unique_solution_system :
  ∃! (x y z : ℝ),
    x^2 - 2*x - 4*z = 3 ∧
    y^2 - 2*y - 2*x = -14 ∧
    z^2 - 4*y - 4*z = -18 ∧
    x = 2 ∧ y = 3 ∧ z = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1028_102853


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_area_formula_l1028_102820

/-- A cyclic quadrilateral is a quadrilateral inscribed in a circle -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  φ : ℝ
  S : ℝ

/-- The area formula for a cyclic quadrilateral -/
theorem cyclic_quadrilateral_area_formula (Q : CyclicQuadrilateral) :
  Q.S = Real.sqrt (Q.a * Q.b * Q.c * Q.d) * Real.sin Q.φ := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_area_formula_l1028_102820


namespace NUMINAMATH_CALUDE_computer_upgrade_cost_l1028_102808

/-- Calculates the total amount spent on a computer after upgrading the video card -/
def totalSpent (initialCost salePrice newCardCost : ℕ) : ℕ :=
  initialCost + newCardCost - salePrice

/-- Theorem stating the total amount spent on the computer -/
theorem computer_upgrade_cost :
  ∀ (initialCost salePrice newCardCost : ℕ),
    initialCost = 1200 →
    salePrice = 300 →
    newCardCost = 500 →
    totalSpent initialCost salePrice newCardCost = 1400 :=
by
  sorry

end NUMINAMATH_CALUDE_computer_upgrade_cost_l1028_102808


namespace NUMINAMATH_CALUDE_haircuts_to_goal_l1028_102864

/-- Given a person who has gotten 8 haircuts and is 80% towards their goal,
    prove that the number of additional haircuts needed to reach 100% of the goal is 2. -/
theorem haircuts_to_goal (current_haircuts : ℕ) (current_percentage : ℚ) : 
  current_haircuts = 8 → current_percentage = 80/100 → 
  (100/100 - current_percentage) / (current_percentage / current_haircuts) = 2 := by
sorry

end NUMINAMATH_CALUDE_haircuts_to_goal_l1028_102864


namespace NUMINAMATH_CALUDE_probability_white_then_red_l1028_102873

/-- The probability of drawing a white marble first and then a red marble second, without replacement, from a bag containing 4 red marbles and 6 white marbles. -/
theorem probability_white_then_red (red_marbles white_marbles : ℕ) 
  (h_red : red_marbles = 4) 
  (h_white : white_marbles = 6) : 
  (white_marbles : ℚ) / (red_marbles + white_marbles) * 
  (red_marbles : ℚ) / (red_marbles + white_marbles - 1) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_then_red_l1028_102873


namespace NUMINAMATH_CALUDE_parabola_vertex_l1028_102858

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = 2 * (x - 3)^2 - 7

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (3, -7)

/-- Theorem: The vertex of the parabola y = 2(x-3)^2 - 7 is (3, -7) -/
theorem parabola_vertex : 
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1028_102858


namespace NUMINAMATH_CALUDE_negative_two_cubed_l1028_102834

theorem negative_two_cubed : (-2 : ℤ)^3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_cubed_l1028_102834


namespace NUMINAMATH_CALUDE_complex_power_48_l1028_102878

theorem complex_power_48 :
  (Complex.exp (Complex.I * Real.pi * (125 / 180)))^48 = Complex.ofReal (-1/2) + Complex.I * Complex.ofReal (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_48_l1028_102878


namespace NUMINAMATH_CALUDE_prime_congruence_problem_l1028_102891

theorem prime_congruence_problem (p q : Nat) (n : Nat) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : n > 1)
  (hpOdd : Odd p) (hqOdd : Odd q)
  (hcong1 : q^(n+2) ≡ 3^(n+2) [MOD p^n])
  (hcong2 : p^(n+2) ≡ 3^(n+2) [MOD q^n]) :
  p = 3 ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_prime_congruence_problem_l1028_102891


namespace NUMINAMATH_CALUDE_cost_of_pencils_and_pens_l1028_102831

/-- Given the cost of pencils and pens, prove the cost of 4 pencils and 4 pens -/
theorem cost_of_pencils_and_pens 
  (pencil_cost pen_cost : ℝ)
  (h1 : 6 * pencil_cost + 3 * pen_cost = 5.40)
  (h2 : 3 * pencil_cost + 5 * pen_cost = 4.80) :
  4 * pencil_cost + 4 * pen_cost = 4.80 := by
sorry


end NUMINAMATH_CALUDE_cost_of_pencils_and_pens_l1028_102831


namespace NUMINAMATH_CALUDE_group_frequency_l1028_102806

theorem group_frequency (sample_capacity : ℕ) (group_frequency : ℚ) :
  sample_capacity = 80 →
  group_frequency = 0.125 →
  (sample_capacity : ℚ) * group_frequency = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_frequency_l1028_102806


namespace NUMINAMATH_CALUDE_bonus_interval_proof_l1028_102856

/-- The number of cards after which Brady gets a bonus -/
def bonus_interval : ℕ := sorry

/-- The pay per card in cents -/
def pay_per_card : ℕ := 70

/-- The bonus amount in cents -/
def bonus_amount : ℕ := 1000

/-- The total number of cards transcribed -/
def total_cards : ℕ := 200

/-- The total earnings in cents including bonuses -/
def total_earnings : ℕ := 16000

theorem bonus_interval_proof : 
  bonus_interval = 100 ∧
  total_earnings = total_cards * pay_per_card + 
    (total_cards / bonus_interval) * bonus_amount :=
by sorry

end NUMINAMATH_CALUDE_bonus_interval_proof_l1028_102856


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l1028_102888

theorem consecutive_integers_square_sum : ∃ (a : ℕ), 
  (a > 0) ∧ 
  ((a - 1) * a * (a + 1) = 8 * (3 * a)) ∧ 
  ((a - 1)^2 + a^2 + (a + 1)^2 = 77) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l1028_102888


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1028_102807

theorem arithmetic_geometric_sequence_ratio 
  (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d) 
  (h3 : (a 3 - a 1) * (a 9 - a 3) = (a 3 - a 1)^2) : 
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13/16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1028_102807


namespace NUMINAMATH_CALUDE_tank_width_l1028_102836

/-- The width of a tank given its dimensions and plastering costs -/
theorem tank_width (length : ℝ) (depth : ℝ) (plaster_rate : ℝ) (total_cost : ℝ) 
  (h1 : length = 25)
  (h2 : depth = 6)
  (h3 : plaster_rate = 0.75)
  (h4 : total_cost = 558)
  (h5 : total_cost = plaster_rate * (length * width + 2 * length * depth + 2 * width * depth)) :
  width = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_width_l1028_102836


namespace NUMINAMATH_CALUDE_third_day_breath_holding_l1028_102844

def breath_holding_sequence (n : ℕ) : ℕ :=
  10 * n

theorem third_day_breath_holding :
  let seq := breath_holding_sequence
  seq 1 = 10 ∧ 
  seq 2 = 20 ∧ 
  seq 6 = 90 →
  seq 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_third_day_breath_holding_l1028_102844


namespace NUMINAMATH_CALUDE_salary_restoration_l1028_102833

theorem salary_restoration (initial_salary : ℝ) (h : initial_salary > 0) :
  let reduced_salary := 0.7 * initial_salary
  (initial_salary / reduced_salary - 1) * 100 = 42.86 := by
sorry

end NUMINAMATH_CALUDE_salary_restoration_l1028_102833


namespace NUMINAMATH_CALUDE_patio_perimeter_l1028_102889

/-- A rectangular patio with length 40 feet and width equal to one-fourth of its length has a perimeter of 100 feet. -/
theorem patio_perimeter : 
  ∀ (length width : ℝ), 
  length = 40 → 
  width = length / 4 → 
  2 * length + 2 * width = 100 := by
sorry

end NUMINAMATH_CALUDE_patio_perimeter_l1028_102889


namespace NUMINAMATH_CALUDE_complement_of_hit_at_least_once_l1028_102897

/-- Represents the outcome of a single shot -/
inductive ShotOutcome
| Hit
| Miss

/-- Represents the outcomes of two shots -/
def TwoShots := (ShotOutcome × ShotOutcome)

/-- The event of hitting the target at least once in two shots -/
def HitAtLeastOnce (shots : TwoShots) : Prop :=
  shots.1 = ShotOutcome.Hit ∨ shots.2 = ShotOutcome.Hit

/-- The event of missing the target both times -/
def MissBothTimes (shots : TwoShots) : Prop :=
  shots.1 = ShotOutcome.Miss ∧ shots.2 = ShotOutcome.Miss

/-- Theorem stating that MissBothTimes is the complement of HitAtLeastOnce -/
theorem complement_of_hit_at_least_once :
  ∀ (shots : TwoShots), ¬(HitAtLeastOnce shots) ↔ MissBothTimes shots :=
sorry


end NUMINAMATH_CALUDE_complement_of_hit_at_least_once_l1028_102897


namespace NUMINAMATH_CALUDE_product_with_9999_l1028_102863

theorem product_with_9999 : ∃ x : ℝ, x * 9999 = 4690910862 ∧ x = 469.1 := by
  sorry

end NUMINAMATH_CALUDE_product_with_9999_l1028_102863


namespace NUMINAMATH_CALUDE_sum_of_powers_l1028_102875

theorem sum_of_powers : -2^2003 + (-2)^2004 + 2^2005 - 2^2006 = -3 * 2^2003 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1028_102875


namespace NUMINAMATH_CALUDE_sum_squares_product_l1028_102841

theorem sum_squares_product (m n : ℝ) (h : m + n = -2) : 5*m^2 + 5*n^2 + 10*m*n = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_product_l1028_102841


namespace NUMINAMATH_CALUDE_board_pair_positive_l1028_102848

inductive BoardPair : ℚ × ℚ → Prop where
  | initial : BoardPair (1, 1)
  | trans1a (x y : ℚ) : BoardPair (x, y - 1) → BoardPair (x + y, y + 1)
  | trans1b (x y : ℚ) : BoardPair (x + y, y + 1) → BoardPair (x, y - 1)
  | trans2a (x y : ℚ) : BoardPair (x, x * y) → BoardPair (1 / x, y)
  | trans2b (x y : ℚ) : BoardPair (1 / x, y) → BoardPair (x, x * y)

theorem board_pair_positive (a b : ℚ) : BoardPair (a, b) → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_board_pair_positive_l1028_102848


namespace NUMINAMATH_CALUDE_cube_sum_product_l1028_102828

theorem cube_sum_product : ∃ x y : ℤ, x^3 + y^3 = 189 ∧ x * y = 20 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_product_l1028_102828


namespace NUMINAMATH_CALUDE_solution_set_trig_equation_l1028_102817

theorem solution_set_trig_equation :
  {x : ℝ | 3 * Real.sin x = 1 + Real.cos (2 * x)} =
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + (-1)^k * Real.pi / 6} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_trig_equation_l1028_102817


namespace NUMINAMATH_CALUDE_canoe_oar_probability_l1028_102845

theorem canoe_oar_probability (p : ℝ) :
  p ≥ 0 ∧ p ≤ 1 →
  2 * p - p^2 = 0.84 →
  p = 0.6 := by
sorry

end NUMINAMATH_CALUDE_canoe_oar_probability_l1028_102845


namespace NUMINAMATH_CALUDE_expansion_terms_count_l1028_102827

theorem expansion_terms_count (N : ℕ+) : 
  (Nat.choose N 5 = 2002) ↔ (N = 16) := by sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l1028_102827


namespace NUMINAMATH_CALUDE_minimum_distance_point_l1028_102880

/-- The point that minimizes the sum of distances to two fixed points lies on the line connecting those points -/
theorem minimum_distance_point (P Q R : ℝ × ℝ) :
  P.1 = -2 ∧ P.2 = -3 ∧
  Q.1 = 5 ∧ Q.2 = 3 ∧
  R.1 = 2 →
  (∀ m : ℝ, (Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) + 
              Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)) ≥
             (Real.sqrt ((R.1 - P.1)^2 + ((3/7) - P.2)^2) + 
              Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - (3/7))^2))) →
  R.2 = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_minimum_distance_point_l1028_102880


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1028_102803

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, |x - 1| < 2 → x^2 - 5*x - 6 < 0) ∧
  (∃ x : ℝ, x^2 - 5*x - 6 < 0 ∧ ¬(|x - 1| < 2)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1028_102803


namespace NUMINAMATH_CALUDE_sin_m_eq_cos_810_l1028_102804

theorem sin_m_eq_cos_810 (m : ℤ) (h1 : -180 ≤ m) (h2 : m ≤ 180) (h3 : Real.sin (m * π / 180) = Real.cos (810 * π / 180)) :
  m = 0 ∨ m = 180 := by
  sorry

end NUMINAMATH_CALUDE_sin_m_eq_cos_810_l1028_102804


namespace NUMINAMATH_CALUDE_daphnes_collection_height_l1028_102809

/-- Represents the height of a book collection in inches and pages -/
structure BookCollection where
  inches : ℝ
  pages : ℝ
  pages_per_inch : ℝ

/-- The problem statement -/
theorem daphnes_collection_height 
  (miles : BookCollection)
  (daphne : BookCollection)
  (longest_collection_pages : ℝ)
  (h1 : miles.pages_per_inch = 5)
  (h2 : daphne.pages_per_inch = 50)
  (h3 : miles.inches = 240)
  (h4 : longest_collection_pages = 1250)
  (h5 : longest_collection_pages ≥ miles.pages)
  (h6 : longest_collection_pages ≥ daphne.pages)
  (h7 : daphne.pages = longest_collection_pages) :
  daphne.inches = 25 := by
sorry

end NUMINAMATH_CALUDE_daphnes_collection_height_l1028_102809


namespace NUMINAMATH_CALUDE_product_inequality_l1028_102815

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1028_102815


namespace NUMINAMATH_CALUDE_parabola_intersection_l1028_102802

-- Define the two parabolas
def f (x : ℝ) : ℝ := 4 * x^2 + 6 * x - 7
def g (x : ℝ) : ℝ := 2 * x^2 + 5

-- Define the intersection points
def p1 : ℝ × ℝ := (-4, 33)
def p2 : ℝ × ℝ := (1.5, 11)

-- Theorem statement
theorem parabola_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x, y) = p1 ∨ (x, y) = p2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1028_102802


namespace NUMINAMATH_CALUDE_M_superset_P_l1028_102814

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x, y = x^2 - 4}
def P : Set ℝ := {y | |y - 3| ≤ 1}

-- State the theorem
theorem M_superset_P : M ⊇ P := by
  sorry

end NUMINAMATH_CALUDE_M_superset_P_l1028_102814


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l1028_102835

theorem geometric_arithmetic_sequence_sum (x y z : ℝ) 
  (h1 : (4*y)^2 = (3*x)*(5*z))  -- Geometric sequence condition
  (h2 : 2/y = 1/x + 1/z)        -- Arithmetic sequence condition
  : x/z + z/x = 34/15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l1028_102835


namespace NUMINAMATH_CALUDE_prob_X_or_Y_or_Z_wins_l1028_102871

-- Define the probabilities
def prob_X : ℚ := 1/4
def prob_Y : ℚ := 1/8
def prob_Z : ℚ := 1/12

-- Define the total number of cars
def total_cars : ℕ := 15

-- Theorem statement
theorem prob_X_or_Y_or_Z_wins : 
  prob_X + prob_Y + prob_Z = 11/24 := by sorry

end NUMINAMATH_CALUDE_prob_X_or_Y_or_Z_wins_l1028_102871


namespace NUMINAMATH_CALUDE_six_heads_before_tail_l1028_102838

/-- The probability of getting exactly n consecutive heads when flipping a fair coin -/
def prob_n_heads (n : ℕ) : ℚ :=
  1 / 2^n

/-- The probability of getting at least n consecutive heads before a tail when flipping a fair coin -/
def prob_at_least_n_heads (n : ℕ) : ℚ :=
  prob_n_heads n

theorem six_heads_before_tail (q : ℚ) :
  (q = prob_at_least_n_heads 6) → (q = 1 / 64) :=
by sorry

#eval (1 : ℕ) + (64 : ℕ)  -- Should output 65

end NUMINAMATH_CALUDE_six_heads_before_tail_l1028_102838


namespace NUMINAMATH_CALUDE_five_by_seven_double_covered_cells_l1028_102859

/-- Represents a rectangular grid with fold lines -/
structure FoldableGrid :=
  (rows : ℕ)
  (cols : ℕ)
  (foldLines : List (ℕ × ℕ × ℕ × ℕ))  -- List of start and end points of fold lines

/-- Counts the number of cells covered exactly twice after folding -/
def countDoubleCoveredCells (grid : FoldableGrid) : ℕ :=
  sorry

/-- The main theorem stating that a 5x7 grid with specific fold lines has 9 double-covered cells -/
theorem five_by_seven_double_covered_cells :
  ∃ (foldLines : List (ℕ × ℕ × ℕ × ℕ)),
    let grid := FoldableGrid.mk 5 7 foldLines
    countDoubleCoveredCells grid = 9 :=
  sorry

end NUMINAMATH_CALUDE_five_by_seven_double_covered_cells_l1028_102859


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l1028_102894

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l1028_102894


namespace NUMINAMATH_CALUDE_weighted_average_percentage_l1028_102866

def bag1_popped : ℕ := 60
def bag1_total : ℕ := 75

def bag2_popped : ℕ := 42
def bag2_total : ℕ := 50

def bag3_popped : ℕ := 112
def bag3_total : ℕ := 130

def bag4_popped : ℕ := 68
def bag4_total : ℕ := 90

def bag5_popped : ℕ := 82
def bag5_total : ℕ := 100

def total_kernels : ℕ := bag1_total + bag2_total + bag3_total + bag4_total + bag5_total

def weighted_sum : ℚ :=
  (bag1_popped : ℚ) / (bag1_total : ℚ) * (bag1_total : ℚ) +
  (bag2_popped : ℚ) / (bag2_total : ℚ) * (bag2_total : ℚ) +
  (bag3_popped : ℚ) / (bag3_total : ℚ) * (bag3_total : ℚ) +
  (bag4_popped : ℚ) / (bag4_total : ℚ) * (bag4_total : ℚ) +
  (bag5_popped : ℚ) / (bag5_total : ℚ) * (bag5_total : ℚ)

theorem weighted_average_percentage (ε : ℚ) (hε : ε = 1 / 10000) :
  ∃ (x : ℚ), abs (x - (weighted_sum / (total_kernels : ℚ))) < ε ∧ x = 7503 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_percentage_l1028_102866


namespace NUMINAMATH_CALUDE_indigo_restaurant_reviews_l1028_102899

theorem indigo_restaurant_reviews :
  let five_star : ℕ := 6
  let four_star : ℕ := 7
  let three_star : ℕ := 4
  let two_star : ℕ := 1
  let average_rating : ℚ := 4
  let total_reviews := five_star + four_star + three_star + two_star
  let total_stars := 5 * five_star + 4 * four_star + 3 * three_star + 2 * two_star
  (total_stars : ℚ) / total_reviews = average_rating →
  total_reviews = 18 := by
sorry

end NUMINAMATH_CALUDE_indigo_restaurant_reviews_l1028_102899


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l1028_102832

theorem smallest_number_with_remainders : ∃! N : ℕ,
  (N > 0) ∧
  (N % 13 = 2) ∧
  (N % 15 = 4) ∧
  (N % 17 = 6) ∧
  (N % 19 = 8) ∧
  (∀ M : ℕ, M > 0 ∧ M % 13 = 2 ∧ M % 15 = 4 ∧ M % 17 = 6 ∧ M % 19 = 8 → M ≥ N) ∧
  N = 1070747 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l1028_102832


namespace NUMINAMATH_CALUDE_stating_rowing_speed_calculation_l1028_102881

/-- Represents the speed of the river current in km/h -/
def stream_speed : ℝ := 12

/-- Represents the man's rowing speed in still water in km/h -/
def rowing_speed : ℝ := 24

/-- 
Theorem stating that if it takes thrice as long to row up as to row down the river,
given the stream speed, then the rowing speed in still water is 24 km/h
-/
theorem rowing_speed_calculation (distance : ℝ) (h : distance > 0) :
  (distance / (rowing_speed - stream_speed)) = 3 * (distance / (rowing_speed + stream_speed)) →
  rowing_speed = 24 := by
sorry

end NUMINAMATH_CALUDE_stating_rowing_speed_calculation_l1028_102881


namespace NUMINAMATH_CALUDE_fruit_drawing_orders_l1028_102821

def basket : Finset String := {"apple", "peach", "pear", "melon"}

theorem fruit_drawing_orders :
  (basket.card * (basket.card - 1) : ℕ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_fruit_drawing_orders_l1028_102821


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1028_102879

/-- 
Given an arithmetic sequence where:
- The first term is 3x - 4
- The second term is 6x - 15
- The third term is 4x + 3
- The nth term is 4021

Prove that n = 627
-/
theorem arithmetic_sequence_nth_term (x : ℚ) (n : ℕ) :
  (3 * x - 4 : ℚ) = (6 * x - 15 : ℚ) - (3 * x - 4 : ℚ) ∧
  (4 * x + 3 : ℚ) = (6 * x - 15 : ℚ) + ((6 * x - 15 : ℚ) - (3 * x - 4 : ℚ)) ∧
  (3 * x - 4 : ℚ) + (n - 1 : ℕ) * ((6 * x - 15 : ℚ) - (3 * x - 4 : ℚ)) = 4021 →
  n = 627 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1028_102879


namespace NUMINAMATH_CALUDE_arnold_protein_consumption_l1028_102822

-- Define the protein content of each food item
def collagen_protein_per_2_scoops : ℕ := 18
def protein_powder_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

-- Define Arnold's consumption
def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 1
def steak_portions : ℕ := 1

-- Theorem to prove
theorem arnold_protein_consumption :
  (collagen_scoops * collagen_protein_per_2_scoops / 2) +
  (protein_powder_scoops * protein_powder_per_scoop) +
  (steak_portions * steak_protein) = 86 := by
  sorry

end NUMINAMATH_CALUDE_arnold_protein_consumption_l1028_102822


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1028_102812

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →
  (S 1 + 2 * S 5 = 3 * S 3) →
  q = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1028_102812


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1028_102818

theorem sum_of_coefficients (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (2 - x) * (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 →
  a₀ + a₆ = -30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1028_102818


namespace NUMINAMATH_CALUDE_symmetric_abs_sum_l1028_102882

/-- A function f is symmetric about a point c if f(c+x) = f(c-x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- The main theorem -/
theorem symmetric_abs_sum (a : ℝ) :
  SymmetricAbout (fun x ↦ |x + 1| + |x - a|) 1 → a = 3 := by
  sorry


end NUMINAMATH_CALUDE_symmetric_abs_sum_l1028_102882


namespace NUMINAMATH_CALUDE_mark_fish_problem_l1028_102840

/-- Given the number of tanks, pregnant fish per tank, and young per fish, 
    calculate the total number of young fish. -/
def total_young_fish (num_tanks : ℕ) (fish_per_tank : ℕ) (young_per_fish : ℕ) : ℕ :=
  num_tanks * fish_per_tank * young_per_fish

/-- Theorem stating that with 3 tanks, 4 pregnant fish per tank, and 20 young per fish, 
    the total number of young fish is 240. -/
theorem mark_fish_problem : 
  total_young_fish 3 4 20 = 240 := by
  sorry

end NUMINAMATH_CALUDE_mark_fish_problem_l1028_102840


namespace NUMINAMATH_CALUDE_test_score_ranges_l1028_102872

/-- Given three ranges of test scores, prove that R1 is 30 -/
theorem test_score_ranges (R1 R2 R3 : ℕ) : 
  R2 = 26 → 
  R3 = 32 → 
  (min R1 (min R2 R3) = 30) → 
  R1 = 30 := by
sorry

end NUMINAMATH_CALUDE_test_score_ranges_l1028_102872


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1028_102846

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1028_102846


namespace NUMINAMATH_CALUDE_sum_of_multiples_is_even_l1028_102870

theorem sum_of_multiples_is_even (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  ∃ n : ℤ, a + b = 2 * n :=
sorry

end NUMINAMATH_CALUDE_sum_of_multiples_is_even_l1028_102870


namespace NUMINAMATH_CALUDE_expression_value_l1028_102829

theorem expression_value (a b c d e f : ℝ) 
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : |e| = Real.sqrt 2)
  (h4 : Real.sqrt f = 8) :
  1/2 * a * b + (c + d) / 5 + e^2 + (f^(1/3)) = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1028_102829


namespace NUMINAMATH_CALUDE_unique_balance_point_iff_m_eq_two_or_neg_one_or_one_l1028_102826

/-- A function f : ℝ → ℝ has a balance point at t if f(t) = t -/
def HasBalancePoint (f : ℝ → ℝ) (t : ℝ) : Prop :=
  f t = t

/-- A function f : ℝ → ℝ has a unique balance point if there exists exactly one t such that f(t) = t -/
def HasUniqueBalancePoint (f : ℝ → ℝ) : Prop :=
  ∃! t, HasBalancePoint f t

/-- The function we're considering -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 - 3 * x + 2 * m

theorem unique_balance_point_iff_m_eq_two_or_neg_one_or_one :
  ∀ m : ℝ, HasUniqueBalancePoint (f m) ↔ m = 2 ∨ m = -1 ∨ m = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_balance_point_iff_m_eq_two_or_neg_one_or_one_l1028_102826


namespace NUMINAMATH_CALUDE_no_divisible_by_seven_l1028_102801

theorem no_divisible_by_seven : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2009 → ¬(7 ∣ (4 * n^6 + n^3 + 5)) := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_by_seven_l1028_102801


namespace NUMINAMATH_CALUDE_number_of_baskets_l1028_102868

def apples_per_basket : ℕ := 17
def total_apples : ℕ := 629

theorem number_of_baskets : 
  total_apples / apples_per_basket = 37 := by sorry

end NUMINAMATH_CALUDE_number_of_baskets_l1028_102868


namespace NUMINAMATH_CALUDE_audrey_peaches_l1028_102849

def paul_peaches : ℕ := 48
def peach_difference : ℤ := 22

theorem audrey_peaches :
  ∃ (audrey : ℕ), (audrey : ℤ) - paul_peaches = peach_difference ∧ audrey = 70 := by
  sorry

end NUMINAMATH_CALUDE_audrey_peaches_l1028_102849


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l1028_102839

theorem wire_ratio_proof (total_length longer_piece shorter_piece : ℤ) : 
  total_length = 90 ∧ shorter_piece = 20 ∧ longer_piece = total_length - shorter_piece →
  shorter_piece / longer_piece = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l1028_102839


namespace NUMINAMATH_CALUDE_no_real_solutions_l1028_102855

theorem no_real_solutions :
  ∀ x y : ℝ, 3 * x^2 + 4 * y^2 - 12 * x - 16 * y + 36 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1028_102855


namespace NUMINAMATH_CALUDE_line_segment_intersection_l1028_102867

/-- Given a line ax + y + 2 = 0 and points P(-2, 1) and Q(3, 2), 
    if the line intersects with the line segment PQ, 
    then a ≤ -4/3 or a ≥ 3/2 -/
theorem line_segment_intersection (a : ℝ) : 
  (∃ (x y : ℝ), a * x + y + 2 = 0 ∧ 
    ((x = -2 ∧ y = 1) ∨ 
     (x = 3 ∧ y = 2) ∨ 
     (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ 
       x = -2 + 5*t ∧ 
       y = 1 + t))) → 
  (a ≤ -4/3 ∨ a ≥ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_intersection_l1028_102867


namespace NUMINAMATH_CALUDE_multiplication_grid_problem_l1028_102825

theorem multiplication_grid_problem :
  ∃ (a b : ℕ+), 
    a * b = 1843 ∧ 
    (1843 % 10 = 3) ∧ 
    ((1843 / 10) % 10 = 8) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_grid_problem_l1028_102825


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l1028_102816

/-- Given a geometric sequence {a_n} with common ratio q = 3,
    if S_3 + S_4 = 53/3, then a_3 = 3 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = 3 * a n) →  -- common ratio q = 3
  (∀ n, S n = (a 1) * (3^n - 1) / 2) →  -- sum formula for geometric sequence
  S 3 + S 4 = 53 / 3 →  -- given condition
  a 3 = 3 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_a3_l1028_102816


namespace NUMINAMATH_CALUDE_lauren_change_l1028_102819

/-- Represents the grocery items with their prices and discounts --/
structure GroceryItems where
  hamburger_meat_price : ℝ
  hamburger_meat_discount : ℝ
  hamburger_buns_price : ℝ
  lettuce_price : ℝ
  tomato_price : ℝ
  tomato_weight : ℝ
  onion_price : ℝ
  onion_weight : ℝ
  pickles_price : ℝ
  pickles_coupon : ℝ
  potatoes_price : ℝ
  soda_price : ℝ
  soda_discount : ℝ

/-- Calculates the total cost of the grocery items including tax --/
def calculateTotalCost (items : GroceryItems) (tax_rate : ℝ) : ℝ :=
  let hamburger_meat_cost := 2 * items.hamburger_meat_price * (1 - items.hamburger_meat_discount)
  let hamburger_buns_cost := items.hamburger_buns_price
  let tomato_cost := items.tomato_price * items.tomato_weight
  let onion_cost := items.onion_price * items.onion_weight
  let pickles_cost := items.pickles_price - items.pickles_coupon
  let soda_cost := items.soda_price * (1 - items.soda_discount)
  let subtotal := hamburger_meat_cost + hamburger_buns_cost + items.lettuce_price + 
                  tomato_cost + onion_cost + pickles_cost + items.potatoes_price + soda_cost
  subtotal * (1 + tax_rate)

/-- Proves that Lauren's change from a $50 bill is $24.67 --/
theorem lauren_change (items : GroceryItems) (tax_rate : ℝ) :
  items.hamburger_meat_price = 3.5 →
  items.hamburger_meat_discount = 0.15 →
  items.hamburger_buns_price = 1.5 →
  items.lettuce_price = 1 →
  items.tomato_price = 2 →
  items.tomato_weight = 1.5 →
  items.onion_price = 0.75 →
  items.onion_weight = 0.5 →
  items.pickles_price = 2.5 →
  items.pickles_coupon = 1 →
  items.potatoes_price = 4 →
  items.soda_price = 5.99 →
  items.soda_discount = 0.07 →
  tax_rate = 0.06 →
  50 - calculateTotalCost items tax_rate = 24.67 := by
  sorry

end NUMINAMATH_CALUDE_lauren_change_l1028_102819


namespace NUMINAMATH_CALUDE_shielas_drawing_distribution_l1028_102842

/-- Represents the number of animal drawings each neighbor receives. -/
def drawings_per_neighbor (total_drawings : ℕ) (num_neighbors : ℕ) : ℕ :=
  total_drawings / num_neighbors

/-- Proves that Shiela's neighbors each receive 9 animal drawings. -/
theorem shielas_drawing_distribution :
  let total_drawings : ℕ := 54
  let num_neighbors : ℕ := 6
  drawings_per_neighbor total_drawings num_neighbors = 9 := by
  sorry

end NUMINAMATH_CALUDE_shielas_drawing_distribution_l1028_102842


namespace NUMINAMATH_CALUDE_fantasy_books_per_day_l1028_102813

/-- Proves that the number of fantasy books sold per day is 5 --/
theorem fantasy_books_per_day 
  (fantasy_price : ℝ)
  (literature_price : ℝ)
  (literature_per_day : ℕ)
  (total_earnings : ℝ)
  (h1 : fantasy_price = 4)
  (h2 : literature_price = fantasy_price / 2)
  (h3 : literature_per_day = 8)
  (h4 : total_earnings = 180) :
  ∃ (fantasy_per_day : ℕ), 
    fantasy_per_day * fantasy_price * 5 + 
    literature_per_day * literature_price * 5 = total_earnings ∧ 
    fantasy_per_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_fantasy_books_per_day_l1028_102813


namespace NUMINAMATH_CALUDE_wilsons_theorem_l1028_102893

theorem wilsons_theorem (p : ℕ) (h : p > 1) :
  Nat.Prime p ↔ p ∣ (Nat.factorial (p - 1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l1028_102893


namespace NUMINAMATH_CALUDE_order_of_numbers_l1028_102811

theorem order_of_numbers : 
  let a := 2 / Real.exp 2
  let b := Real.log (Real.sqrt 2)
  let c := Real.log 3 / 3
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l1028_102811


namespace NUMINAMATH_CALUDE_cereal_eating_time_l1028_102874

def fat_rate : ℚ := 1 / 25
def thin_rate : ℚ := 1 / 35
def medium_rate : ℚ := 1 / 28
def total_cereal : ℚ := 5

def combined_rate : ℚ := fat_rate + thin_rate + medium_rate

def time_taken : ℚ := total_cereal / combined_rate

theorem cereal_eating_time : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ time_taken - 48 < ε ∧ 48 - time_taken < ε :=
sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l1028_102874


namespace NUMINAMATH_CALUDE_method_one_saves_more_money_l1028_102824

/-- Represents the discount methods available at the store -/
inductive DiscountMethod
  | BuyRacketGetShuttlecock
  | PayPercentage

/-- Calculates the cost of purchase using the given discount method -/
def calculateCost (racketPrice shuttlecockPrice : ℕ) (racketCount shuttlecockCount : ℕ) (method : DiscountMethod) : ℚ :=
  match method with
  | DiscountMethod.BuyRacketGetShuttlecock =>
      (racketCount * racketPrice + (shuttlecockCount - racketCount) * shuttlecockPrice : ℚ)
  | DiscountMethod.PayPercentage =>
      ((racketCount * racketPrice + shuttlecockCount * shuttlecockPrice) * 92 / 100 : ℚ)

/-- Theorem stating that discount method ① saves more money than method ② -/
theorem method_one_saves_more_money (racketPrice shuttlecockPrice : ℕ) (racketCount shuttlecockCount : ℕ)
    (h1 : racketPrice = 20)
    (h2 : shuttlecockPrice = 5)
    (h3 : racketCount = 4)
    (h4 : shuttlecockCount = 30) :
    calculateCost racketPrice shuttlecockPrice racketCount shuttlecockCount DiscountMethod.BuyRacketGetShuttlecock <
    calculateCost racketPrice shuttlecockPrice racketCount shuttlecockCount DiscountMethod.PayPercentage :=
  sorry

end NUMINAMATH_CALUDE_method_one_saves_more_money_l1028_102824


namespace NUMINAMATH_CALUDE_log_sum_equality_l1028_102885

theorem log_sum_equality : Real.log 4 / Real.log 10 + 2 * (Real.log 5 / Real.log 10) + 8^(2/3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l1028_102885


namespace NUMINAMATH_CALUDE_saras_golf_balls_l1028_102892

-- Define the number of dozens Sara has
def saras_dozens : ℕ := 9

-- Define the number of items in a dozen
def items_per_dozen : ℕ := 12

-- Theorem stating that Sara's total number of golf balls is 108
theorem saras_golf_balls : saras_dozens * items_per_dozen = 108 := by
  sorry

end NUMINAMATH_CALUDE_saras_golf_balls_l1028_102892


namespace NUMINAMATH_CALUDE_total_travel_ways_problem_solution_l1028_102890

/-- Represents the number of transportation options between two cities -/
structure TransportOptions where
  buses : Nat
  trains : Nat
  ferries : Nat

/-- Calculates the total number of ways to travel between two cities -/
def totalWays (options : TransportOptions) : Nat :=
  options.buses + options.trains + options.ferries

/-- Theorem: The total number of ways to travel from A to C via B is the product
    of the number of ways to travel from A to B and from B to C -/
theorem total_travel_ways
  (optionsAB : TransportOptions)
  (optionsBC : TransportOptions) :
  totalWays optionsAB * totalWays optionsBC =
  (optionsAB.buses + optionsAB.trains) * (optionsBC.buses + optionsBC.ferries) :=
by sorry

/-- Given the specific transportation options in the problem -/
def morningOptions : TransportOptions :=
  { buses := 5, trains := 2, ferries := 0 }

def afternoonOptions : TransportOptions :=
  { buses := 3, trains := 0, ferries := 2 }

/-- The main theorem that proves the total number of ways for the specific problem -/
theorem problem_solution :
  totalWays morningOptions * totalWays afternoonOptions = 35 :=
by sorry

end NUMINAMATH_CALUDE_total_travel_ways_problem_solution_l1028_102890


namespace NUMINAMATH_CALUDE_paving_cost_example_l1028_102886

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

theorem paving_cost_example : 
  paving_cost 5.5 3.75 1200 = 24750 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_example_l1028_102886


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1028_102850

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → ∃ r : ℝ, a (n + 1) = a n * r

-- Define the problem statement
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, n > 0 → a n > 0) →
  a 1 * a 99 = 16 →
  a 1 + a 99 = 10 →
  a 40 * a 50 * a 60 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1028_102850


namespace NUMINAMATH_CALUDE_uncertain_relationship_l1028_102830

-- Define a type for planes in 3D space
variable (Plane : Type)

-- Define the perpendicular relation between planes
variable (perp : Plane → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Define the intersecting (neither perpendicular nor parallel) relation between planes
variable (intersecting : Plane → Plane → Prop)

-- State the theorem
theorem uncertain_relationship 
  (a₁ a₂ a₃ a₄ : Plane) 
  (distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄) 
  (h₁ : perp a₁ a₂) 
  (h₂ : perp a₂ a₃) 
  (h₃ : perp a₃ a₄) : 
  ¬(∀ a₁ a₄, perp a₁ a₄ ∨ parallel a₁ a₄ ∨ intersecting a₁ a₄) :=
by sorry

end NUMINAMATH_CALUDE_uncertain_relationship_l1028_102830


namespace NUMINAMATH_CALUDE_businessman_travel_l1028_102895

theorem businessman_travel (morning_bike : ℕ) (evening_bike : ℕ) (car_trips : ℕ) :
  morning_bike = 10 →
  evening_bike = 12 →
  car_trips = 8 →
  morning_bike + evening_bike + car_trips - 15 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_businessman_travel_l1028_102895


namespace NUMINAMATH_CALUDE_total_cost_is_thirteen_l1028_102823

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := 2

/-- The additional cost of a pen compared to a pencil in dollars -/
def pen_additional_cost : ℝ := 9

/-- The cost of a pen in dollars -/
def pen_cost : ℝ := pencil_cost + pen_additional_cost

/-- The total cost of both items in dollars -/
def total_cost : ℝ := pen_cost + pencil_cost

theorem total_cost_is_thirteen : total_cost = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_thirteen_l1028_102823


namespace NUMINAMATH_CALUDE_pizza_time_is_ten_minutes_l1028_102843

/-- Represents the pizza-making scenario --/
structure PizzaScenario where
  totalTime : ℕ        -- Total time in hours
  initialFlour : ℕ     -- Initial flour in kg
  flourPerPizza : ℚ    -- Flour required per pizza in kg
  remainingPizzas : ℕ  -- Number of pizzas that can be made with remaining flour

/-- Calculates the time taken to make each pizza --/
def timeTakenPerPizza (scenario : PizzaScenario) : ℚ :=
  let totalMinutes := scenario.totalTime * 60
  let usedFlour := scenario.initialFlour - (scenario.remainingPizzas * scenario.flourPerPizza)
  let pizzasMade := usedFlour / scenario.flourPerPizza
  totalMinutes / pizzasMade

/-- Theorem stating that the time taken per pizza is 10 minutes --/
theorem pizza_time_is_ten_minutes (scenario : PizzaScenario) 
    (h1 : scenario.totalTime = 7)
    (h2 : scenario.initialFlour = 22)
    (h3 : scenario.flourPerPizza = 1/2)
    (h4 : scenario.remainingPizzas = 2) :
    timeTakenPerPizza scenario = 10 := by
  sorry


end NUMINAMATH_CALUDE_pizza_time_is_ten_minutes_l1028_102843


namespace NUMINAMATH_CALUDE_factorize_expression_1_l1028_102884

theorem factorize_expression_1 (x : ℝ) :
  (x^2 - 1 + x) * (x^2 - 1 + 3*x) + x^2 = x^4 + 4*x^3 + 4*x^2 - 4*x - 1 := by
sorry

end NUMINAMATH_CALUDE_factorize_expression_1_l1028_102884


namespace NUMINAMATH_CALUDE_door_open_probability_l1028_102851

def num_keys : ℕ := 5

def probability_open_on_third_attempt : ℚ := 1 / 5

theorem door_open_probability :
  probability_open_on_third_attempt = 0.2 := by sorry

end NUMINAMATH_CALUDE_door_open_probability_l1028_102851
