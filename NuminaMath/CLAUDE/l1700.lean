import Mathlib

namespace NUMINAMATH_CALUDE_fraction_equals_decimal_l1700_170067

theorem fraction_equals_decimal : (1 : ℚ) / 4 = 0.25 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_decimal_l1700_170067


namespace NUMINAMATH_CALUDE_cars_time_passage_l1700_170024

/-- Given that a car comes down the road every 20 minutes, 
    prove that the time passed for 30 cars is 10 hours. -/
theorem cars_time_passage (interval : ℕ) (num_cars : ℕ) (hours_per_day : ℕ) :
  interval = 20 →
  num_cars = 30 →
  hours_per_day = 24 →
  (interval * num_cars) / 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cars_time_passage_l1700_170024


namespace NUMINAMATH_CALUDE_solve_equation_l1700_170065

theorem solve_equation : ∃ x : ℝ, x + 2*x = 400 - (3*x + 4*x) ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1700_170065


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l1700_170031

theorem right_triangle_leg_length : ∀ (a b c : ℝ),
  a = 8 →
  c = 17 →
  a^2 + b^2 = c^2 →
  b = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l1700_170031


namespace NUMINAMATH_CALUDE_smallest_constant_D_l1700_170084

theorem smallest_constant_D :
  ∃ (D : ℝ), D = Real.sqrt (8 / 17) ∧
  (∀ (x y : ℝ), x^2 + 2*y^2 + 5 ≥ D*(2*x + 3*y) + 4) ∧
  (∀ (D' : ℝ), (∀ (x y : ℝ), x^2 + 2*y^2 + 5 ≥ D'*(2*x + 3*y) + 4) → D' ≥ D) :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_D_l1700_170084


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1700_170010

/-- The solution set of the inequality |x| + |x - 1| < 2 -/
theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x| + |x - 1| < 2} = Set.Ioo (-1/2 : ℝ) (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1700_170010


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l1700_170013

/-- The function f(x) = |2x+1| + |2x-3| -/
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

/-- Theorem for the range of a -/
theorem range_of_a (a : ℝ) : (∀ x, f x > |1 - 3*a|) → -1 < a ∧ a < 5/3 := by sorry

/-- Theorem for the range of m -/
theorem range_of_m (m : ℝ) : 
  (∃ t : ℝ, t^2 - 4*Real.sqrt 2*t + f m = 0) → 
  -3/2 ≤ m ∧ m ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l1700_170013


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l1700_170043

/-- Calculates the total bill given the number of people and the amount each person paid -/
def totalBill (numPeople : ℕ) (amountPerPerson : ℕ) : ℕ :=
  numPeople * amountPerPerson

/-- Proves that if three people divide a bill evenly and each pays $45, then the total bill is $135 -/
theorem restaurant_bill_proof :
  totalBill 3 45 = 135 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l1700_170043


namespace NUMINAMATH_CALUDE_nail_container_problem_l1700_170078

theorem nail_container_problem (N : ℝ) : 
  (N > 0) →
  (0.7 * N - 0.7 * (0.7 * N) = 84) →
  N = 400 := by
sorry

end NUMINAMATH_CALUDE_nail_container_problem_l1700_170078


namespace NUMINAMATH_CALUDE_lcm_of_385_and_180_l1700_170022

theorem lcm_of_385_and_180 :
  let a := 385
  let b := 180
  let hcf := 30
  Nat.lcm a b = 2310 :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_385_and_180_l1700_170022


namespace NUMINAMATH_CALUDE_rope_average_length_l1700_170021

/-- Given 6 ropes where one third have an average length of 70 cm and the rest have an average length of 85 cm, prove that the overall average length is 80 cm. -/
theorem rope_average_length : 
  let total_ropes : ℕ := 6
  let third_ropes : ℕ := total_ropes / 3
  let remaining_ropes : ℕ := total_ropes - third_ropes
  let third_avg_length : ℝ := 70
  let remaining_avg_length : ℝ := 85
  let total_length : ℝ := (third_ropes : ℝ) * third_avg_length + (remaining_ropes : ℝ) * remaining_avg_length
  let overall_avg_length : ℝ := total_length / (total_ropes : ℝ)
  overall_avg_length = 80 := by
sorry

end NUMINAMATH_CALUDE_rope_average_length_l1700_170021


namespace NUMINAMATH_CALUDE_probability_multiple_of_three_l1700_170070

/-- A fair cubic die with 6 faces numbered 1 to 6 -/
def FairDie : Finset ℕ := Finset.range 6

/-- The set of outcomes that are multiples of 3 -/
def MultiplesOfThree : Finset ℕ := Finset.filter (fun n => n % 3 = 0) FairDie

/-- The probability of an event in a finite sample space -/
def probability (event : Finset ℕ) (sampleSpace : Finset ℕ) : ℚ :=
  (event.card : ℚ) / (sampleSpace.card : ℚ)

theorem probability_multiple_of_three : 
  probability MultiplesOfThree FairDie = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_of_three_l1700_170070


namespace NUMINAMATH_CALUDE_last_mile_speed_l1700_170054

/-- Represents the problem of calculating the required speed for the last mile of a journey --/
theorem last_mile_speed (total_distance : ℝ) (normal_speed : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) (last_part_distance : ℝ) : 
  total_distance = 3 →
  normal_speed = 10 →
  first_part_distance = 2 →
  first_part_speed = 5 →
  last_part_distance = 1 →
  (total_distance / normal_speed = first_part_distance / first_part_speed + last_part_distance / 10) := by
  sorry

end NUMINAMATH_CALUDE_last_mile_speed_l1700_170054


namespace NUMINAMATH_CALUDE_prime_congruent_three_mod_four_divides_x_l1700_170023

theorem prime_congruent_three_mod_four_divides_x (p : ℕ) (x₀ y₀ : ℕ) :
  Prime p →
  p % 4 = 3 →
  x₀ > 0 →
  y₀ > 0 →
  (p + 2) * x₀^2 - (p + 1) * y₀^2 + p * x₀ + (p + 2) * y₀ = 1 →
  p ∣ x₀ := by
  sorry

end NUMINAMATH_CALUDE_prime_congruent_three_mod_four_divides_x_l1700_170023


namespace NUMINAMATH_CALUDE_turtle_arrival_time_l1700_170057

/-- Represents the speeds and distances of the animals -/
structure AnimalData where
  turtle_speed : ℝ
  lion1_speed : ℝ
  lion2_speed : ℝ
  turtle_distance : ℝ
  lion1_distance : ℝ

/-- Represents the time intervals between events -/
structure TimeIntervals where
  between_encounters : ℝ
  after_second_encounter : ℝ

/-- The main theorem stating the time for the turtle to reach the watering hole -/
theorem turtle_arrival_time 
  (data : AnimalData) 
  (time : TimeIntervals) 
  (h1 : data.lion1_distance = 6 * data.lion1_speed)
  (h2 : data.lion2_speed = 1.5 * data.lion1_speed)
  (h3 : data.turtle_distance = 32 * data.turtle_speed)
  (h4 : time.between_encounters = 2.4)
  (h5 : (data.lion1_distance - data.turtle_distance) / (data.lion1_speed - data.turtle_speed) + 
        time.between_encounters = 
        data.turtle_distance / (data.turtle_speed + data.lion2_speed))
  : time.after_second_encounter = 28.8 := by
  sorry

end NUMINAMATH_CALUDE_turtle_arrival_time_l1700_170057


namespace NUMINAMATH_CALUDE_right_triangle_condition_l1700_170081

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*cos(B) + a*cos(C) = b + c, then the triangle is right-angled. -/
theorem right_triangle_condition (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.cos B + a * Real.cos C = b + c →
  a^2 = b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l1700_170081


namespace NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l1700_170008

-- Problem 1
theorem simplify_fraction_1 (x y : ℝ) (h : y ≠ 0) :
  (x^2 - 1) / y / ((x + 1) / y^2) = y * (x - 1) := by sorry

-- Problem 2
theorem simplify_fraction_2 (m n : ℝ) (h1 : m ≠ n) (h2 : m ≠ -n) :
  m / (m + n) + n / (m - n) - 2 * m^2 / (m^2 - n^2) = -1 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l1700_170008


namespace NUMINAMATH_CALUDE_wall_width_proof_l1700_170080

/-- Proves that the width of a wall is 2 meters given specific brick and wall dimensions --/
theorem wall_width_proof (brick_length : Real) (brick_width : Real) (brick_height : Real)
  (wall_length : Real) (wall_height : Real) (num_bricks : Nat) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_length = 27 →
  wall_height = 0.75 →
  num_bricks = 27000 →
  ∃ (wall_width : Real), wall_width = 2 ∧
    brick_length * brick_width * brick_height * num_bricks =
    wall_length * wall_width * wall_height := by
  sorry

end NUMINAMATH_CALUDE_wall_width_proof_l1700_170080


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1700_170005

/-- Given a geometric sequence with first term a₁ = 1, 
    the minimum value of 6a₂ + 7a₃ is -9/7 -/
theorem min_value_geometric_sequence (a₁ a₂ a₃ : ℝ) : 
  a₁ = 1 → 
  (∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) → 
  (∀ x y : ℝ, x = a₂ ∧ y = a₃ → 6*x + 7*y ≥ -9/7) ∧ 
  (∃ x y : ℝ, x = a₂ ∧ y = a₃ ∧ 6*x + 7*y = -9/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1700_170005


namespace NUMINAMATH_CALUDE_exists_function_satisfying_condition_l1700_170032

theorem exists_function_satisfying_condition :
  ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, (n : ℝ)^2 - 1 < (f (f n) : ℝ) ∧ (f (f n) : ℝ) < (n : ℝ)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_function_satisfying_condition_l1700_170032


namespace NUMINAMATH_CALUDE_john_remaining_money_l1700_170063

def calculate_remaining_money (base_income : ℝ) (bonus_rate : ℝ) (transport_rate : ℝ)
  (rent : ℝ) (utilities : ℝ) (food : ℝ) (misc_rate : ℝ) (emergency_rate : ℝ)
  (retirement_rate : ℝ) (medical_expense : ℝ) (tax_rate : ℝ) : ℝ :=
  let total_income := base_income * (1 + bonus_rate)
  let after_tax_income := total_income - (base_income * tax_rate)
  let fixed_expenses := rent + utilities + food
  let variable_expenses := (total_income * transport_rate) + (total_income * misc_rate)
  let savings_and_investments := (total_income * emergency_rate) + (total_income * retirement_rate)
  let total_expenses := fixed_expenses + variable_expenses + medical_expense + savings_and_investments
  after_tax_income - total_expenses

theorem john_remaining_money :
  calculate_remaining_money 2000 0.15 0.05 500 100 300 0.10 0.07 0.05 250 0.15 = 229 := by
  sorry

end NUMINAMATH_CALUDE_john_remaining_money_l1700_170063


namespace NUMINAMATH_CALUDE_first_column_is_seven_l1700_170091

/-- Represents a 5x2 grid with one empty cell -/
def Grid := Fin 9 → Fin 9

/-- The sum of a column in the grid -/
def column_sum (g : Grid) (col : Fin 5) : ℕ :=
  if col = 0 then g 0
  else if col = 1 then g 1 + g 2
  else if col = 2 then g 3 + g 4
  else if col = 3 then g 5 + g 6
  else g 7 + g 8

/-- Predicate for a valid grid arrangement -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j : Fin 9, i ≠ j → g i ≠ g j) ∧
  (∀ col : Fin 4, column_sum g (col + 1) = column_sum g col + 1)

theorem first_column_is_seven (g : Grid) (h : is_valid_grid g) : g 0 = 7 := by
  sorry

end NUMINAMATH_CALUDE_first_column_is_seven_l1700_170091


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1700_170003

-- Define the property of having at most finitely many zeros
def HasFinitelyManyZeros (f : ℝ → ℝ) : Prop :=
  ∃ (S : Finset ℝ), ∀ x, f x = 0 → x ∈ S

-- Define the functional equation
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x^4 + y) = x^3 * f x + f (f y)

-- Theorem statement
theorem unique_function_satisfying_conditions :
  ∃! f : ℝ → ℝ, HasFinitelyManyZeros f ∧ SatisfiesFunctionalEquation f ∧ (∀ x, f x = x) :=
sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1700_170003


namespace NUMINAMATH_CALUDE_problem_1_l1700_170027

theorem problem_1 (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 2 * (deriv f 1) * x) :
  deriv f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1700_170027


namespace NUMINAMATH_CALUDE_inverse_sum_lower_bound_l1700_170006

theorem inverse_sum_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (hab_sum : a + b = 1) :
  1 / a + 1 / b > 4 := by
sorry

end NUMINAMATH_CALUDE_inverse_sum_lower_bound_l1700_170006


namespace NUMINAMATH_CALUDE_problem_solution_l1700_170033

theorem problem_solution (x y a : ℝ) 
  (h1 : |x + 1| + (y + 2)^2 = 0)
  (h2 : a * x - 3 * a * y = 1) : 
  a = 0.2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1700_170033


namespace NUMINAMATH_CALUDE_system_a_l1700_170026

theorem system_a (x y : ℝ) : 
  y^4 + x*y^2 - 2*x^2 = 0 ∧ x + y = 6 →
  (x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = -3) :=
sorry

end NUMINAMATH_CALUDE_system_a_l1700_170026


namespace NUMINAMATH_CALUDE_negative_inequality_l1700_170053

theorem negative_inequality (a b : ℝ) (h : a > b) : -a < -b := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l1700_170053


namespace NUMINAMATH_CALUDE_square_root_of_one_fourth_l1700_170073

theorem square_root_of_one_fourth : ∃ x : ℚ, x^2 = (1/4 : ℚ) ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_one_fourth_l1700_170073


namespace NUMINAMATH_CALUDE_tan_negative_fifty_five_sixths_pi_l1700_170056

theorem tan_negative_fifty_five_sixths_pi : 
  Real.tan (-55 / 6 * Real.pi) = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_negative_fifty_five_sixths_pi_l1700_170056


namespace NUMINAMATH_CALUDE_triangle_inequality_l1700_170015

/-- For any triangle ABC with semiperimeter p and inradius r, 
    the sum of the reciprocals of the square roots of twice the sines of its angles 
    is less than or equal to the square root of the ratio of its semiperimeter to its inradius. -/
theorem triangle_inequality (A B C : Real) (p r : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π ∧ 0 < p ∧ 0 < r → 
  1 / Real.sqrt (2 * Real.sin A) + 1 / Real.sqrt (2 * Real.sin B) + 1 / Real.sqrt (2 * Real.sin C) ≤ Real.sqrt (p / r) := by
  sorry


end NUMINAMATH_CALUDE_triangle_inequality_l1700_170015


namespace NUMINAMATH_CALUDE_fabric_length_l1700_170038

/-- Given a rectangular piece of fabric with width 3 cm and area 24 cm², prove its length is 8 cm. -/
theorem fabric_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 3 → area = 24 → area = length * width → length = 8 := by
sorry

end NUMINAMATH_CALUDE_fabric_length_l1700_170038


namespace NUMINAMATH_CALUDE_equation_proof_l1700_170039

theorem equation_proof : 529 + 2 * 23 * 11 + 121 = 1156 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1700_170039


namespace NUMINAMATH_CALUDE_basketball_wins_l1700_170055

/-- The total number of wins for a basketball team over four competitions -/
def total_wins (first_wins : ℕ) : ℕ :=
  let second_wins := (first_wins * 5) / 8
  let third_wins := first_wins + second_wins
  let fourth_wins := ((first_wins + second_wins + third_wins) * 3) / 5
  first_wins + second_wins + third_wins + fourth_wins

/-- Theorem stating that given 40 wins in the first competition, the total wins over four competitions is 208 -/
theorem basketball_wins : total_wins 40 = 208 := by
  sorry

end NUMINAMATH_CALUDE_basketball_wins_l1700_170055


namespace NUMINAMATH_CALUDE_probability_one_girl_in_pair_l1700_170068

theorem probability_one_girl_in_pair (n_boys n_girls : ℕ) (h_boys : n_boys = 4) (h_girls : n_girls = 2) :
  let total := n_boys + n_girls
  let total_pairs := total.choose 2
  let favorable_outcomes := n_boys * n_girls
  (favorable_outcomes : ℚ) / total_pairs = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_girl_in_pair_l1700_170068


namespace NUMINAMATH_CALUDE_cost_per_flower_is_15_l1700_170014

/-- Represents the number of centerpieces -/
def num_centerpieces : ℕ := 6

/-- Represents the number of roses per centerpiece -/
def roses_per_centerpiece : ℕ := 8

/-- Represents the number of lilies per centerpiece -/
def lilies_per_centerpiece : ℕ := 6

/-- Represents the total budget in dollars -/
def total_budget : ℕ := 2700

/-- Calculates the total number of roses -/
def total_roses : ℕ := num_centerpieces * roses_per_centerpiece

/-- Calculates the total number of orchids -/
def total_orchids : ℕ := 2 * total_roses

/-- Calculates the total number of lilies -/
def total_lilies : ℕ := num_centerpieces * lilies_per_centerpiece

/-- Calculates the total number of flowers -/
def total_flowers : ℕ := total_roses + total_orchids + total_lilies

/-- Theorem: The cost per flower is $15 -/
theorem cost_per_flower_is_15 : total_budget / total_flowers = 15 := by
  sorry


end NUMINAMATH_CALUDE_cost_per_flower_is_15_l1700_170014


namespace NUMINAMATH_CALUDE_sqrt_calculations_l1700_170096

theorem sqrt_calculations :
  (∃ (x y : ℝ), x = Real.sqrt 3 ∧ y = Real.sqrt 2 ∧
    x * y - Real.sqrt 12 / Real.sqrt 8 = Real.sqrt 6 / 2) ∧
  ((Real.sqrt 2 - 3)^2 - Real.sqrt 2^2 - Real.sqrt (2^2) - Real.sqrt 2 = 7 - 7 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l1700_170096


namespace NUMINAMATH_CALUDE_nonzero_matrix_squared_zero_l1700_170002

theorem nonzero_matrix_squared_zero : 
  ∃ (A : Matrix (Fin 2) (Fin 2) ℝ), A ≠ 0 ∧ A ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_matrix_squared_zero_l1700_170002


namespace NUMINAMATH_CALUDE_polynomial_identity_l1700_170025

theorem polynomial_identity (x : ℝ) : 
  (x + 1)^4 + 4*(x + 1)^3 + 6*(x + 1)^2 + 4*(x + 1) + 1 = (x + 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1700_170025


namespace NUMINAMATH_CALUDE_rectangle_area_l1700_170041

/-- The area of a rectangle with width 2 feet and length 5 feet is 10 square feet. -/
theorem rectangle_area : 
  let width : ℝ := 2
  let length : ℝ := 5
  width * length = 10 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1700_170041


namespace NUMINAMATH_CALUDE_cabinet_area_l1700_170007

theorem cabinet_area : 
  ∀ (width length area : ℝ),
  width = 1.2 →
  length = 1.8 →
  area = width * length →
  area = 2.16 := by
sorry

end NUMINAMATH_CALUDE_cabinet_area_l1700_170007


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1700_170034

theorem inequality_solution_set (t : ℝ) (h : 0 < t ∧ t < 1) :
  {x : ℝ | (t - x) * (x - 1/t) > 0} = {x : ℝ | t < x ∧ x < 1/t} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1700_170034


namespace NUMINAMATH_CALUDE_sandwich_non_filler_percentage_l1700_170089

/-- Given a sandwich weighing 180 grams with 45 grams of fillers,
    prove that the percentage of the sandwich that is not filler is 75%. -/
theorem sandwich_non_filler_percentage
  (total_weight : ℝ)
  (filler_weight : ℝ)
  (h1 : total_weight = 180)
  (h2 : filler_weight = 45) :
  (total_weight - filler_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_non_filler_percentage_l1700_170089


namespace NUMINAMATH_CALUDE_arcMTN_constant_l1700_170064

/-- Represents an equilateral triangle ABC with a circle rolling along side AB -/
structure RollingCircleTriangle where
  /-- Side length of the equilateral triangle -/
  side : ℝ
  /-- Radius of the circle, equal to the triangle's altitude -/
  radius : ℝ
  /-- The circle's radius is equal to the triangle's altitude -/
  radius_eq_altitude : radius = side * Real.sqrt 3 / 2

/-- The measure of arc MTN in degrees -/
def arcMTN (rct : RollingCircleTriangle) : ℝ :=
  60

/-- Theorem stating that arc MTN always measures 60° -/
theorem arcMTN_constant (rct : RollingCircleTriangle) :
  arcMTN rct = 60 := by
  sorry

end NUMINAMATH_CALUDE_arcMTN_constant_l1700_170064


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l1700_170011

def f (x : ℝ) : ℝ := -x + 1

theorem f_satisfies_conditions :
  (∃ x y, x < 0 ∧ y > 0 ∧ f x = y) ∧
  (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂) :=
sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l1700_170011


namespace NUMINAMATH_CALUDE_clock_angle_at_9am_l1700_170045

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees each hour represents -/
def degrees_per_hour : ℕ := 30

/-- The position of the minute hand at 9:00 a.m. in degrees -/
def minute_hand_position : ℕ := 0

/-- The position of the hour hand at 9:00 a.m. in degrees -/
def hour_hand_position : ℕ := 270

/-- The smaller angle between the minute hand and hour hand at 9:00 a.m. -/
def smaller_angle : ℕ := 90

/-- Theorem stating that the smaller angle between the minute hand and hour hand at 9:00 a.m. is 90 degrees -/
theorem clock_angle_at_9am :
  smaller_angle = min (hour_hand_position - minute_hand_position) (360 - (hour_hand_position - minute_hand_position)) :=
by sorry

end NUMINAMATH_CALUDE_clock_angle_at_9am_l1700_170045


namespace NUMINAMATH_CALUDE_registration_methods_count_l1700_170020

/-- Represents the number of courses -/
def num_courses : ℕ := 3

/-- Represents the number of students -/
def num_students : ℕ := 5

/-- 
Calculates the number of ways to distribute n distinct objects into k distinct boxes,
where each box must contain at least one object.
-/
def distribution_count (n k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the number of registration methods is 150 -/
theorem registration_methods_count : distribution_count num_students num_courses = 150 :=
  sorry

end NUMINAMATH_CALUDE_registration_methods_count_l1700_170020


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l1700_170001

theorem trigonometric_product_equals_one :
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let sin30 := 1 / 2
  let cos60 := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l1700_170001


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_for_given_equation_l1700_170083

theorem sum_of_x_and_y_for_given_equation (x y : ℝ) : 
  2 * x^2 - 4 * x * y + 4 * y^2 + 6 * x + 9 = 0 → x + y = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_for_given_equation_l1700_170083


namespace NUMINAMATH_CALUDE_comparison_of_rational_numbers_l1700_170044

theorem comparison_of_rational_numbers :
  (- (- (1 / 5 : ℚ)) > - (1 / 5 : ℚ)) ∧
  (- (- (17 / 5 : ℚ)) > - (17 / 5 : ℚ)) ∧
  (- (4 : ℚ) < (4 : ℚ)) ∧
  ((- (11 / 10 : ℚ)) < 0) :=
by sorry

end NUMINAMATH_CALUDE_comparison_of_rational_numbers_l1700_170044


namespace NUMINAMATH_CALUDE_jumping_contest_l1700_170079

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_jump = 15)
  (h3 : mouse_jump + 44 = frog_jump) :
  grasshopper_jump - frog_jump = 4 := by
  sorry


end NUMINAMATH_CALUDE_jumping_contest_l1700_170079


namespace NUMINAMATH_CALUDE_earth_total_area_l1700_170099

/-- The ocean area on Earth's surface in million square kilometers -/
def ocean_area : ℝ := 361

/-- The difference between ocean and land area in million square kilometers -/
def area_difference : ℝ := 2.12

/-- The total area of the Earth in million square kilometers -/
def total_area : ℝ := ocean_area + (ocean_area - area_difference)

theorem earth_total_area :
  total_area = 5.10 := by
  sorry

end NUMINAMATH_CALUDE_earth_total_area_l1700_170099


namespace NUMINAMATH_CALUDE_number_equation_l1700_170086

theorem number_equation (x : ℝ) : (1/4 : ℝ) * x + 15 = 27 ↔ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1700_170086


namespace NUMINAMATH_CALUDE_probability_both_above_400_l1700_170075

def total_students : ℕ := 600
def male_students : ℕ := 220
def female_students : ℕ := 380
def selected_students : ℕ := 10
def selected_females : ℕ := 6
def females_above_400 : ℕ := 3
def discussion_group_size : ℕ := 2

theorem probability_both_above_400 :
  (female_students = total_students - male_students) →
  (selected_females ≤ selected_students) →
  (females_above_400 ≤ selected_females) →
  (discussion_group_size ≤ selected_females) →
  (Nat.choose females_above_400 discussion_group_size) / (Nat.choose selected_females discussion_group_size) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_above_400_l1700_170075


namespace NUMINAMATH_CALUDE_special_arrangements_count_l1700_170017

/-- The number of ways to arrange 3 boys and 3 girls in a row, 
    where one specific boy is not adjacent to the other two boys -/
def special_arrangements : ℕ :=
  let n_boys := 3
  let n_girls := 3
  let arrangements_with_boys_separated := n_girls.factorial * (n_girls + 1).factorial
  let arrangements_with_two_boys_adjacent := 2 * (n_girls + 1).factorial * n_girls.factorial
  arrangements_with_boys_separated + arrangements_with_two_boys_adjacent

theorem special_arrangements_count : special_arrangements = 288 := by
  sorry

end NUMINAMATH_CALUDE_special_arrangements_count_l1700_170017


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1700_170052

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  passesThrough : ℝ × ℝ

/-- The equation of a circle. -/
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = 
    (c.passesThrough.1 - c.center.1)^2 + (c.passesThrough.2 - c.center.2)^2

/-- The specific circle from the problem. -/
def C : Circle :=
  { center := (2, -3)
  , passesThrough := (0, 0) }

theorem circle_equation_proof :
  ∀ x y : ℝ, circleEquation C x y ↔ (x - 2)^2 + (y + 3)^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l1700_170052


namespace NUMINAMATH_CALUDE_field_trip_van_occupancy_l1700_170009

theorem field_trip_van_occupancy (num_vans num_buses people_per_bus total_people : ℕ) 
  (h1 : num_vans = 9)
  (h2 : num_buses = 10)
  (h3 : people_per_bus = 27)
  (h4 : total_people = 342) :
  (total_people - num_buses * people_per_bus) / num_vans = 8 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_van_occupancy_l1700_170009


namespace NUMINAMATH_CALUDE_max_trees_cut_2001_l1700_170085

/-- Represents a square grid of trees -/
structure TreeGrid where
  size : Nat
  is_square : size * size = size * size

/-- Represents the maximum number of trees that can be cut down -/
def max_trees_cut (grid : TreeGrid) : Nat :=
  (grid.size / 2) * (grid.size / 2) + 1

/-- The theorem to be proved -/
theorem max_trees_cut_2001 :
  ∀ (grid : TreeGrid),
    grid.size = 2001 →
    max_trees_cut grid = 1001001 := by
  sorry

end NUMINAMATH_CALUDE_max_trees_cut_2001_l1700_170085


namespace NUMINAMATH_CALUDE_initial_volume_calculation_l1700_170050

theorem initial_volume_calculation (initial_milk_percentage : Real)
                                   (final_milk_percentage : Real)
                                   (added_water : Real) :
  initial_milk_percentage = 0.84 →
  final_milk_percentage = 0.64 →
  added_water = 18.75 →
  ∃ (initial_volume : Real),
    initial_volume * initial_milk_percentage = 
    final_milk_percentage * (initial_volume + added_water) ∧
    initial_volume = 225 := by
  sorry

end NUMINAMATH_CALUDE_initial_volume_calculation_l1700_170050


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l1700_170035

/-- Represents a quadratic function of the form y = a(x-m)(x-m-k) -/
def quadratic_function (a m k x : ℝ) : ℝ := a * (x - m) * (x - m - k)

/-- The minimum value of the quadratic function when k = 2 -/
def min_value (a m : ℝ) : ℝ := -a

theorem quadratic_minimum_value (a m : ℝ) (h : a > 0) :
  ∃ x, quadratic_function a m 2 x = min_value a m ∧
  ∀ y, quadratic_function a m 2 y ≥ min_value a m :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l1700_170035


namespace NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l1700_170047

/-- Represents a quiz with specified scoring rules -/
structure Quiz where
  total_questions : ℕ
  correct_points : ℕ
  incorrect_deduction : ℕ

/-- Calculates the score for a given number of correct answers -/
def score (q : Quiz) (correct_answers : ℕ) : ℤ :=
  (q.correct_points * correct_answers : ℤ) - 
  (q.incorrect_deduction * (q.total_questions - correct_answers) : ℤ)

/-- Theorem stating the minimum number of correct answers needed to achieve a target score -/
theorem min_correct_answers_for_target_score 
  (q : Quiz) 
  (target_score : ℤ) 
  (correct_answers : ℕ) :
  q.total_questions = 20 →
  q.correct_points = 5 →
  q.incorrect_deduction = 1 →
  target_score = 88 →
  score q correct_answers ≥ target_score →
  (5 : ℤ) * correct_answers - (20 - correct_answers) ≥ 88 := by
  sorry

#check min_correct_answers_for_target_score

end NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l1700_170047


namespace NUMINAMATH_CALUDE_time_between_ticks_at_6_l1700_170040

/-- The number of ticks at 6 o'clock -/
def ticks_at_6 : ℕ := 6

/-- The number of ticks at 8 o'clock -/
def ticks_at_8 : ℕ := 8

/-- The time between the first and last ticks at 8 o'clock in seconds -/
def time_at_8 : ℕ := 42

/-- The theorem stating the time between the first and last ticks at 6 o'clock -/
theorem time_between_ticks_at_6 : ℕ := by
  -- Assume the time between each tick is constant for any hour
  -- Calculate the time between ticks at 6 o'clock
  sorry

end NUMINAMATH_CALUDE_time_between_ticks_at_6_l1700_170040


namespace NUMINAMATH_CALUDE_simplify_expression_l1700_170093

theorem simplify_expression (x : ℚ) : 
  ((3 * x + 6) - 5 * x) / 3 = -2 * x / 3 + 2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1700_170093


namespace NUMINAMATH_CALUDE_ricardo_coin_value_difference_l1700_170049

/-- The total number of coins Ricardo has -/
def total_coins : ℕ := 3030

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Calculate the total value in cents given the number of pennies -/
def total_value (num_pennies : ℕ) : ℕ :=
  num_pennies * penny_value + (total_coins - num_pennies) * nickel_value

/-- The minimum number of pennies Ricardo can have -/
def min_pennies : ℕ := 1

/-- The maximum number of pennies Ricardo can have -/
def max_pennies : ℕ := total_coins - 1

theorem ricardo_coin_value_difference :
  (total_value min_pennies) - (total_value max_pennies) = 12112 := by
  sorry

end NUMINAMATH_CALUDE_ricardo_coin_value_difference_l1700_170049


namespace NUMINAMATH_CALUDE_cube_edge_length_l1700_170030

-- Define the cube
structure Cube where
  edge_length : ℝ
  sum_of_edges : ℝ

-- State the theorem
theorem cube_edge_length (c : Cube) (h : c.sum_of_edges = 108) : c.edge_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1700_170030


namespace NUMINAMATH_CALUDE_sum_357_eq_42_l1700_170051

/-- A geometric sequence with first term 3 and the sum of the first, third, and fifth terms equal to 21 -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  first_term : a 1 = 3
  sum_135 : a 1 + a 3 + a 5 = 21

/-- The sum of the third, fifth, and seventh terms of the geometric sequence is 42 -/
theorem sum_357_eq_42 (seq : GeometricSequence) : seq.a 3 + seq.a 5 + seq.a 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_357_eq_42_l1700_170051


namespace NUMINAMATH_CALUDE_unused_sector_angle_l1700_170059

/-- Given a circular piece of paper with radius 20 cm, from which a sector is removed
    to form a cone with radius 15 cm and volume 900π cubic cm,
    prove that the measure of the angle of the unused sector is 90°. -/
theorem unused_sector_angle (r_paper : ℝ) (r_cone : ℝ) (v_cone : ℝ) :
  r_paper = 20 →
  r_cone = 15 →
  v_cone = 900 * Real.pi →
  ∃ (h : ℝ) (s : ℝ),
    v_cone = (1/3) * Real.pi * r_cone^2 * h ∧
    s^2 = r_cone^2 + h^2 ∧
    s ≤ r_paper ∧
    (2 * Real.pi * r_cone) / (2 * Real.pi * r_paper) * 360 = 270 :=
by sorry

end NUMINAMATH_CALUDE_unused_sector_angle_l1700_170059


namespace NUMINAMATH_CALUDE_initial_girls_count_l1700_170037

theorem initial_girls_count (initial_total : ℕ) (initial_girls : ℕ) : 
  initial_girls = 12 ∧ initial_total = 24 :=
  by
  have h1 : initial_girls = initial_total / 2 := by sorry
  have h2 : (initial_girls - 2) * 100 = 40 * (initial_total + 1) := by sorry
  have h3 : initial_girls * 100 = 45 * (initial_total - 1) := by sorry
  sorry

#check initial_girls_count

end NUMINAMATH_CALUDE_initial_girls_count_l1700_170037


namespace NUMINAMATH_CALUDE_minimize_y_l1700_170069

variable (a b : ℝ)
def y (x : ℝ) := (x - a)^2 + (x - b)^2

theorem minimize_y :
  ∃ (x : ℝ), ∀ (z : ℝ), y a b x ≤ y a b z ∧ x = (a + b) / 2 :=
sorry

end NUMINAMATH_CALUDE_minimize_y_l1700_170069


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1700_170090

theorem triangle_angle_measure (A B C : Real) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) (h3 : A + B + C = Real.pi) :
  C = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1700_170090


namespace NUMINAMATH_CALUDE_second_project_length_l1700_170082

/-- Represents a digging project -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of earth dug in a project -/
def volume (p : DiggingProject) : ℝ := p.depth * p.length * p.breadth

/-- The first digging project -/
def project1 : DiggingProject := ⟨100, 25, 30⟩

/-- The second digging project with unknown length -/
def project2 (l : ℝ) : DiggingProject := ⟨75, l, 50⟩

/-- The theorem stating that the length of the second project is 20 meters -/
theorem second_project_length :
  ∃ l : ℝ, volume project1 = volume (project2 l) ∧ l = 20 := by
  sorry


end NUMINAMATH_CALUDE_second_project_length_l1700_170082


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l1700_170004

theorem r_value_when_n_is_3 : 
  ∀ (n s r : ℕ), 
    n = 3 → 
    s = 2^n + 2 → 
    r = 4^s + 3*s → 
    r = 1048606 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l1700_170004


namespace NUMINAMATH_CALUDE_parabola_point_M_x_coordinate_l1700_170074

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define line l passing through F and intersecting the parabola at A and B
def line_l (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  ∃ k : ℝ, A.2 = k * (A.1 - 1) ∧ B.2 = k * (B.1 - 1)

-- Define point M as the midpoint of A and B
def point_M (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define point P on the parabola
def point_P (P : ℝ × ℝ) : Prop := parabola P.1 P.2

-- Define the distance between P and F is 2
def PF_distance (P : ℝ × ℝ) : Prop :=
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = 4

theorem parabola_point_M_x_coordinate 
  (A B M P : ℝ × ℝ) 
  (h1 : line_l A B) 
  (h2 : point_M A B M) 
  (h3 : point_P P) 
  (h4 : PF_distance P) :
  M.1 = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_point_M_x_coordinate_l1700_170074


namespace NUMINAMATH_CALUDE_sqrt_product_equals_two_l1700_170012

theorem sqrt_product_equals_two : Real.sqrt 12 * Real.sqrt (1/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_two_l1700_170012


namespace NUMINAMATH_CALUDE_arctan_sum_property_l1700_170088

theorem arctan_sum_property (a b c : ℝ) 
  (h : Real.arctan a + Real.arctan b + Real.arctan c + π / 2 = 0) : 
  (a * b + b * c + c * a = 1) ∧ (a + b + c < a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_property_l1700_170088


namespace NUMINAMATH_CALUDE_expand_and_equate_l1700_170016

theorem expand_and_equate : 
  (∀ x : ℝ, (x - 5) * (x + 2) = x^2 + p * x + q) → p = -3 ∧ q = -10 := by
sorry

end NUMINAMATH_CALUDE_expand_and_equate_l1700_170016


namespace NUMINAMATH_CALUDE_smallest_equal_distribution_l1700_170042

def apple_per_box : ℕ := 18
def grapes_per_container : ℕ := 9
def orange_per_container : ℕ := 12
def cherries_per_bag : ℕ := 6

theorem smallest_equal_distribution (n : ℕ) :
  (n % apple_per_box = 0) ∧
  (n % grapes_per_container = 0) ∧
  (n % orange_per_container = 0) ∧
  (n % cherries_per_bag = 0) ∧
  (∀ m : ℕ, m < n →
    ¬((m % apple_per_box = 0) ∧
      (m % grapes_per_container = 0) ∧
      (m % orange_per_container = 0) ∧
      (m % cherries_per_bag = 0))) →
  n = 36 := by
sorry

end NUMINAMATH_CALUDE_smallest_equal_distribution_l1700_170042


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1700_170058

/-- The y-intercept of the line 4x + 7y = 28 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → x = 0 → y = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1700_170058


namespace NUMINAMATH_CALUDE_consecutive_composites_under_40_l1700_170077

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem consecutive_composites_under_40 :
  ∃ (a : ℕ),
    (∀ i : Fin 6, isTwoDigit (a + i) ∧ a + i < 40) ∧
    (∀ i : Fin 6, ¬ isPrime (a + i)) ∧
    (∀ n : ℕ, n > a + 5 →
      ¬(∀ i : Fin 6, isTwoDigit (n - i) ∧ n - i < 40 ∧ ¬ isPrime (n - i))) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_composites_under_40_l1700_170077


namespace NUMINAMATH_CALUDE_multiply_by_hundred_l1700_170097

theorem multiply_by_hundred (x : ℝ) : x = 15.46 → x * 100 = 1546 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_hundred_l1700_170097


namespace NUMINAMATH_CALUDE_sapling_planting_equation_l1700_170094

theorem sapling_planting_equation (x : ℤ) : 
  (∀ (total : ℤ), (5 * x + 3 = total) ↔ (6 * x = total + 4)) :=
by sorry

end NUMINAMATH_CALUDE_sapling_planting_equation_l1700_170094


namespace NUMINAMATH_CALUDE_car_distance_proof_l1700_170019

theorem car_distance_proof (D : ℝ) : 
  (D / 60 = D / 90 + 1/2) → D = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l1700_170019


namespace NUMINAMATH_CALUDE_set_operations_l1700_170098

-- Define the universal set U
def U : Set Int := {-3, -1, 0, 1, 2, 3, 4, 6}

-- Define set A
def A : Set Int := {0, 2, 4, 6}

-- Define the complement of A in U
def C_UA : Set Int := {-1, -3, 1, 3}

-- Define the complement of B in U
def C_UB : Set Int := {-1, 0, 2}

-- Define set B
def B : Set Int := U \ C_UB

-- Theorem to prove
theorem set_operations :
  (A ∩ B = {4, 6}) ∧ (A ∪ B = {-3, 0, 1, 2, 3, 4, 6}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1700_170098


namespace NUMINAMATH_CALUDE_red_tile_cost_courtyard_red_tile_cost_l1700_170046

/-- Calculates the cost of each red tile in a courtyard tiling project. -/
theorem red_tile_cost (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (tiles_per_sqft : ℝ) (green_tile_percentage : ℝ) (green_tile_cost : ℝ) 
  (total_cost : ℝ) : ℝ :=
  let total_area := courtyard_length * courtyard_width
  let total_tiles := total_area * tiles_per_sqft
  let green_tiles := green_tile_percentage * total_tiles
  let red_tiles := total_tiles - green_tiles
  let green_cost := green_tiles * green_tile_cost
  let red_cost := total_cost - green_cost
  red_cost / red_tiles

/-- The cost of each red tile in the given courtyard tiling project is $1.50. -/
theorem courtyard_red_tile_cost : 
  red_tile_cost 25 10 4 0.4 3 2100 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_red_tile_cost_courtyard_red_tile_cost_l1700_170046


namespace NUMINAMATH_CALUDE_f_positive_iff_m_range_f_root_in_zero_one_iff_m_range_l1700_170000

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (m-1)*x + 2*m

theorem f_positive_iff_m_range (m : ℝ) :
  (∀ x > 0, f m x > 0) ↔ -2*Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2*Real.sqrt 6 + 5 := by
  sorry

theorem f_root_in_zero_one_iff_m_range (m : ℝ) :
  (∃ x ∈ Set.Ioo 0 1, f m x = 0) ↔ m ∈ Set.Ioo (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_iff_m_range_f_root_in_zero_one_iff_m_range_l1700_170000


namespace NUMINAMATH_CALUDE_division_problem_l1700_170036

theorem division_problem (x y : ℕ+) 
  (h1 : (x : ℝ) / (y : ℝ) = 96.12) 
  (h2 : (x : ℝ) % (y : ℝ) = 5.76) : 
  y = 48 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1700_170036


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_387420501_l1700_170066

-- Define the polynomial
def polynomial (x y : ℤ) : ℤ := (3*x + 4*y)^9 + (2*x - 5*y)^9

-- Define the sum of coefficients function
def sum_of_coefficients (p : ℤ → ℤ → ℤ) (x y : ℤ) : ℤ := p x y

-- Theorem statement
theorem sum_of_coefficients_equals_387420501 :
  sum_of_coefficients polynomial 2 (-1) = 387420501 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_387420501_l1700_170066


namespace NUMINAMATH_CALUDE_max_value_expression_l1700_170061

theorem max_value_expression (a b c d : ℕ) : 
  a ∈ ({0, 1, 2, 3} : Set ℕ) →
  b ∈ ({0, 1, 2, 3} : Set ℕ) →
  c ∈ ({0, 1, 2, 3} : Set ℕ) →
  d ∈ ({0, 1, 2, 3} : Set ℕ) →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  d ≠ 0 →
  c * a^b - d ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1700_170061


namespace NUMINAMATH_CALUDE_ellipse_sum_bounds_l1700_170092

theorem ellipse_sum_bounds (x y : ℝ) : 
  x^2 / 2 + y^2 / 3 = 1 → 
  ∃ (S : ℝ), S = x + y ∧ -Real.sqrt 5 ≤ S ∧ S ≤ Real.sqrt 5 ∧
  (∃ (x₁ y₁ : ℝ), x₁^2 / 2 + y₁^2 / 3 = 1 ∧ x₁ + y₁ = -Real.sqrt 5) ∧
  (∃ (x₂ y₂ : ℝ), x₂^2 / 2 + y₂^2 / 3 = 1 ∧ x₂ + y₂ = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_sum_bounds_l1700_170092


namespace NUMINAMATH_CALUDE_time_to_school_building_l1700_170095

/-- Proves that the time to get from the school gate to the school building is 6 minutes -/
theorem time_to_school_building 
  (total_time : ℕ) 
  (time_to_gate : ℕ) 
  (time_to_room : ℕ) 
  (h1 : total_time = 30) 
  (h2 : time_to_gate = 15) 
  (h3 : time_to_room = 9) : 
  total_time - time_to_gate - time_to_room = 6 := by
  sorry

#check time_to_school_building

end NUMINAMATH_CALUDE_time_to_school_building_l1700_170095


namespace NUMINAMATH_CALUDE_ellipse_and_fixed_point_l1700_170076

/-- Ellipse C₁ -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Parabola C₂ -/
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

/-- Tangent line to parabola -/
def tangent_line (b : ℝ) (x y : ℝ) : Prop :=
  y = x + b

/-- Circle with diameter AB passing through T -/
def circle_passes_through (A B T : ℝ × ℝ) : Prop :=
  (T.1 - A.1) * (T.1 - B.1) + (T.2 - A.2) * (T.2 - B.2) = 0

theorem ellipse_and_fixed_point 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : a^2 - b^2 = a^2 / 2) -- Eccentricity condition
  (h4 : ∃ (x y : ℝ), tangent_line 1 x y ∧ parabola x y) :
  (∀ (x y : ℝ), ellipse a b x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∀ (A B : ℝ × ℝ), 
    ellipse a b A.1 A.2 → 
    ellipse a b B.1 B.2 → 
    (∃ (k : ℝ), A.2 = k * A.1 - 1/3 ∧ B.2 = k * B.1 - 1/3) →
    circle_passes_through A B (0, 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_fixed_point_l1700_170076


namespace NUMINAMATH_CALUDE_alice_rearrangement_time_l1700_170028

/-- The time in hours required to write all rearrangements of a name -/
def time_to_write_rearrangements (name_length : ℕ) (rearrangements_per_minute : ℕ) : ℚ :=
  (Nat.factorial name_length : ℚ) / (rearrangements_per_minute : ℚ) / 60

/-- Theorem: Given a name with 5 unique letters and the ability to write 12 rearrangements per minute,
    it takes 1/6 hours to write all possible rearrangements -/
theorem alice_rearrangement_time :
  time_to_write_rearrangements 5 12 = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_alice_rearrangement_time_l1700_170028


namespace NUMINAMATH_CALUDE_square_difference_fourth_power_l1700_170071

theorem square_difference_fourth_power : (7^2 - 5^2)^4 = 331776 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fourth_power_l1700_170071


namespace NUMINAMATH_CALUDE_nine_trailing_zeros_l1700_170072

def binary_trailing_zeros (n : ℕ) : ℕ :=
  (n.digits 2).reverse.takeWhile (· = 0) |>.length

theorem nine_trailing_zeros (n : ℕ) : binary_trailing_zeros (n * 1024 + 4 * 64 + 2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_trailing_zeros_l1700_170072


namespace NUMINAMATH_CALUDE_division_problem_l1700_170018

theorem division_problem (x : ℝ) : 100 / x = 400 → x = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1700_170018


namespace NUMINAMATH_CALUDE_eight_number_sequence_proof_l1700_170029

theorem eight_number_sequence_proof :
  ∀ (a : Fin 8 → ℕ),
  (a 0 = 20) →
  (a 7 = 16) →
  (∀ i : Fin 6, a i + a (i + 1) + a (i + 2) = 100) →
  (∀ i : Fin 8, a i = [20, 16, 64, 20, 16, 64, 20, 16].get i) :=
by
  sorry

end NUMINAMATH_CALUDE_eight_number_sequence_proof_l1700_170029


namespace NUMINAMATH_CALUDE_weight_of_calcium_hydride_l1700_170087

/-- The atomic weight of calcium in g/mol -/
def Ca_weight : ℝ := 40.08

/-- The atomic weight of hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- The molecular weight of calcium hydride (CaH2) in g/mol -/
def CaH2_weight : ℝ := Ca_weight + 2 * H_weight

/-- The number of moles of calcium hydride -/
def moles : ℝ := 6

/-- Theorem: The weight of 6 moles of calcium hydride (CaH2) is 252.576 grams -/
theorem weight_of_calcium_hydride : moles * CaH2_weight = 252.576 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_calcium_hydride_l1700_170087


namespace NUMINAMATH_CALUDE_function_characterization_l1700_170060

def iterate (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f (iterate f n x)

theorem function_characterization (f : ℕ → ℕ) : 
  (∀ a b : ℕ, a > 0 → b > 0 → 
    (iterate f a b + iterate f b a) ∣ (2 * (f (a * b) + b^2 - 1))) → 
  ((∀ x : ℕ, f x = x + 1) ∨ (f 1 ∣ 4)) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l1700_170060


namespace NUMINAMATH_CALUDE_train_meeting_point_l1700_170048

theorem train_meeting_point 
  (route_length : ℝ) 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (h1 : route_length = 75)
  (h2 : speed_A = 25)
  (h3 : speed_B = 37.5)
  : (route_length * speed_A) / (speed_A + speed_B) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_point_l1700_170048


namespace NUMINAMATH_CALUDE_sum_of_k_values_l1700_170062

theorem sum_of_k_values : ∃ (S : Finset ℤ), 
  (∀ k ∈ S, ∃ x y : ℤ, x ≠ y ∧ 3 * x^2 - k * x + 9 = 0 ∧ 3 * y^2 - k * y + 9 = 0) ∧
  (∀ k : ℤ, (∃ x y : ℤ, x ≠ y ∧ 3 * x^2 - k * x + 9 = 0 ∧ 3 * y^2 - k * y + 9 = 0) → k ∈ S) ∧
  (S.sum id = 0) :=
sorry

end NUMINAMATH_CALUDE_sum_of_k_values_l1700_170062
