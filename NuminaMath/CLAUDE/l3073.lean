import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l3073_307375

def f : Set ℝ → Set ℝ := sorry

def A : Set ℝ := {x | ∃ y ∈ Set.Icc 7 15, f {y} = {2 * x + 1}}

def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ A ∨ x ∈ B a) ↔ 3 ≤ a ∧ a < 6 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3073_307375


namespace NUMINAMATH_CALUDE_solve_rental_problem_l3073_307338

def rental_problem (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : Prop :=
  daily_rate * days + mileage_rate * miles = 275

theorem solve_rental_problem :
  rental_problem 30 0.25 5 500 := by
  sorry

end NUMINAMATH_CALUDE_solve_rental_problem_l3073_307338


namespace NUMINAMATH_CALUDE_max_sum_squares_fibonacci_l3073_307387

theorem max_sum_squares_fibonacci (m n : ℕ) : 
  m ∈ Finset.range 1982 → 
  n ∈ Finset.range 1982 → 
  (n^2 - m*n - m^2)^2 = 1 → 
  m^2 + n^2 ≤ 3524578 := by
sorry

end NUMINAMATH_CALUDE_max_sum_squares_fibonacci_l3073_307387


namespace NUMINAMATH_CALUDE_triangular_array_coins_l3073_307380

-- Define the sum of the first N natural numbers
def triangular_sum (N : ℕ) : ℕ := N * (N + 1) / 2

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_coins :
  ∃ N : ℕ, triangular_sum N = 5050 ∧ sum_of_digits N = 1 :=
sorry

end NUMINAMATH_CALUDE_triangular_array_coins_l3073_307380


namespace NUMINAMATH_CALUDE_right_triangle_ratio_range_l3073_307326

theorem right_triangle_ratio_range (a b c h : ℝ) :
  a > 0 → b > 0 →
  c = (a^2 + b^2).sqrt →
  h = (a * b) / c →
  1 < (c + 2 * h) / (a + b) ∧ (c + 2 * h) / (a + b) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_range_l3073_307326


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3073_307314

theorem consecutive_integers_sum (n : ℕ) (h : n > 0) :
  (7 * n + 21 = 2821) → (n + 6 = 406) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3073_307314


namespace NUMINAMATH_CALUDE_triangle_inequality_l3073_307394

theorem triangle_inequality (a b c p : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_half_perimeter : p = (a + b + c) / 2) :
  a^2 * (p - a) * (p - b) + b^2 * (p - b) * (p - c) + c^2 * (p - c) * (p - a) ≤ (4 / 27) * p^4 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3073_307394


namespace NUMINAMATH_CALUDE_parabola_shift_l3073_307398

-- Define the original parabola
def original_parabola (x y : ℝ) : Prop := y = -(x - 2)^2 - 5

-- Define the shifted parabola
def shifted_parabola (x y : ℝ) : Prop := y = -x^2

-- Theorem stating that shifting the original parabola results in the shifted parabola
theorem parabola_shift :
  ∀ (x y : ℝ), original_parabola (x - 2) (y - 5) ↔ shifted_parabola x y :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l3073_307398


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l3073_307339

theorem least_three_digit_multiple_of_13 : ∃ n : ℕ, 
  n = 104 ∧ 
  n % 13 = 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  ∀ m : ℕ, (m % 13 = 0 ∧ 100 ≤ m ∧ m < 1000) → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l3073_307339


namespace NUMINAMATH_CALUDE_solve_journey_l3073_307382

def journey_problem (total_distance : ℝ) (cycling_speed : ℝ) (walking_speed : ℝ) (total_time : ℝ) : Prop :=
  let cycling_distance : ℝ := (2/3) * total_distance
  let walking_distance : ℝ := total_distance - cycling_distance
  let cycling_time : ℝ := cycling_distance / cycling_speed
  let walking_time : ℝ := walking_distance / walking_speed
  (cycling_time + walking_time = total_time) → (walking_distance = 6)

theorem solve_journey :
  journey_problem 18 20 4 (70/60) := by
  sorry

end NUMINAMATH_CALUDE_solve_journey_l3073_307382


namespace NUMINAMATH_CALUDE_area_30_60_90_triangle_l3073_307355

theorem area_30_60_90_triangle (a : ℝ) (h : a > 0) :
  let triangle_area := (1/2) * a * (a / Real.sqrt 3)
  triangle_area = (32 * Real.sqrt 3) / 3 ↔ a = 8 := by
sorry

end NUMINAMATH_CALUDE_area_30_60_90_triangle_l3073_307355


namespace NUMINAMATH_CALUDE_football_playtime_l3073_307320

def total_playtime_hours : ℝ := 1.5
def basketball_playtime_minutes : ℕ := 60

theorem football_playtime (total_playtime_minutes : ℕ) 
  (h1 : total_playtime_minutes = Int.floor (total_playtime_hours * 60)) 
  (h2 : total_playtime_minutes ≥ basketball_playtime_minutes) : 
  total_playtime_minutes - basketball_playtime_minutes = 30 := by
  sorry

end NUMINAMATH_CALUDE_football_playtime_l3073_307320


namespace NUMINAMATH_CALUDE_solve_clubsuit_equation_l3073_307378

-- Define the ♣ operation
def clubsuit (A B : ℝ) : ℝ := 3 * A^2 + 2 * B + 7

-- State the theorem
theorem solve_clubsuit_equation :
  ∃ A : ℝ, (clubsuit A 7 = 61) ∧ (A = 2 * Real.sqrt 30 / 3) :=
by sorry

end NUMINAMATH_CALUDE_solve_clubsuit_equation_l3073_307378


namespace NUMINAMATH_CALUDE_wire_cutting_l3073_307307

theorem wire_cutting (total_length : ℝ) (ratio : ℚ) (shorter_piece : ℝ) :
  total_length = 60 →
  ratio = 2/4 →
  shorter_piece + ratio * shorter_piece = total_length →
  shorter_piece = 20 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l3073_307307


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l3073_307329

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^3 = 2) :
  a^4 + 1/a^4 = (Real.rpow 4 (1/3) - 2)^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l3073_307329


namespace NUMINAMATH_CALUDE_gear_revolution_theorem_l3073_307337

/-- The number of revolutions per minute for gear p -/
def p_rpm : ℝ := 10

/-- The duration in minutes -/
def duration : ℝ := 0.5

/-- The difference in revolutions between gear q and gear p after the given duration -/
def revolution_difference : ℝ := 15

/-- The number of revolutions per minute for gear q -/
def q_rpm : ℝ := 40

theorem gear_revolution_theorem :
  q_rpm = 2 * (p_rpm * duration + revolution_difference) := by
  sorry

end NUMINAMATH_CALUDE_gear_revolution_theorem_l3073_307337


namespace NUMINAMATH_CALUDE_vaccine_effectiveness_theorem_l3073_307395

/-- Represents the data for a vaccine experiment -/
structure VaccineExperiment where
  total_participants : ℕ
  vaccinated_infected : ℕ
  placebo_infected : ℕ

/-- Calculates the vaccine effectiveness -/
def vaccine_effectiveness (exp : VaccineExperiment) : ℚ :=
  let p := exp.vaccinated_infected / (exp.total_participants / 2 : ℚ)
  let q := exp.placebo_infected / (exp.total_participants / 2 : ℚ)
  1 - p / q

/-- The main theorem about vaccine effectiveness -/
theorem vaccine_effectiveness_theorem (exp_A exp_B : VaccineExperiment) :
  exp_A.total_participants = 30000 →
  exp_A.vaccinated_infected = 50 →
  exp_A.placebo_infected = 500 →
  vaccine_effectiveness exp_A = 9/10 ∧
  ∃ (exp_B : VaccineExperiment),
    vaccine_effectiveness exp_B > 9/10 ∧
    exp_B.vaccinated_infected ≥ exp_A.vaccinated_infected :=
by sorry

end NUMINAMATH_CALUDE_vaccine_effectiveness_theorem_l3073_307395


namespace NUMINAMATH_CALUDE_joan_spent_four_half_dollars_on_wednesday_l3073_307396

/-- The number of half-dollars Joan spent on Wednesday -/
def wednesday_half_dollars : ℕ := sorry

/-- The number of half-dollars Joan spent on Thursday -/
def thursday_half_dollars : ℕ := 14

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 1/2

/-- The total amount Joan spent in dollars -/
def total_spent : ℚ := 9

/-- Theorem: Joan spent 4 half-dollars on Wednesday -/
theorem joan_spent_four_half_dollars_on_wednesday :
  wednesday_half_dollars = 4 :=
by
  have h1 : (wednesday_half_dollars : ℚ) * half_dollar_value + 
            (thursday_half_dollars : ℚ) * half_dollar_value = total_spent :=
    sorry
  sorry

end NUMINAMATH_CALUDE_joan_spent_four_half_dollars_on_wednesday_l3073_307396


namespace NUMINAMATH_CALUDE_shortest_tangent_is_sqrt_449_l3073_307369

/-- Circle C₁ with center (8, 3) and radius 7 -/
def C₁ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 8)^2 + (p.2 - 3)^2 = 49}

/-- Circle C₂ with center (-12, -4) and radius 5 -/
def C₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + 12)^2 + (p.2 + 4)^2 = 25}

/-- The length of the shortest line segment PQ tangent to C₁ at P and C₂ at Q -/
def shortest_tangent_length (C₁ C₂ : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem stating that the shortest tangent length between C₁ and C₂ is √449 -/
theorem shortest_tangent_is_sqrt_449 : 
  shortest_tangent_length C₁ C₂ = Real.sqrt 449 := by sorry

end NUMINAMATH_CALUDE_shortest_tangent_is_sqrt_449_l3073_307369


namespace NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_polar_equation_is_line_l3073_307376

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 1 / (Real.sin θ + Real.cos θ)

-- Define the Cartesian equation of a line
def line_equation (x y : ℝ) : Prop := y + x = 1

-- Theorem statement
theorem polar_to_cartesian_equivalence :
  ∀ (r θ x y : ℝ), 
    polar_equation r θ → 
    x = r * Real.cos θ → 
    y = r * Real.sin θ → 
    line_equation x y := by
  sorry

-- The main theorem stating that the polar equation represents a line
theorem polar_equation_is_line :
  ∃ (a b c : ℝ), ∀ (r θ : ℝ), 
    polar_equation r θ → 
    ∃ (x y : ℝ), x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ a*x + b*y = c := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_polar_equation_is_line_l3073_307376


namespace NUMINAMATH_CALUDE_k_range_l3073_307341

theorem k_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) → 
  -1 < k ∧ k < 0 := by
sorry

end NUMINAMATH_CALUDE_k_range_l3073_307341


namespace NUMINAMATH_CALUDE_factorial_plus_24_equals_square_l3073_307344

theorem factorial_plus_24_equals_square (n m : ℕ) : n.factorial + 24 = m ^ 2 ↔ (n = 1 ∧ m = 5) ∨ (n = 5 ∧ m = 12) := by
  sorry

end NUMINAMATH_CALUDE_factorial_plus_24_equals_square_l3073_307344


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3073_307386

/-- The amount of money Rachel and Sarah had when they left home -/
def initial_money : ℝ := 50

/-- The amount spent on gasoline -/
def gasoline_cost : ℝ := 8

/-- The amount spent on lunch -/
def lunch_cost : ℝ := 15.65

/-- The amount spent on gifts for grandma (per person) -/
def gift_cost : ℝ := 5

/-- The amount received from grandma (per person) -/
def grandma_gift : ℝ := 10

/-- The amount of money they have for the return trip -/
def return_trip_money : ℝ := 36.35

/-- The number of people (Rachel and Sarah) -/
def num_people : ℕ := 2

theorem initial_money_calculation :
  initial_money = 
    return_trip_money + 
    (gasoline_cost + lunch_cost + num_people * gift_cost) - 
    (num_people * grandma_gift) := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l3073_307386


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3073_307361

theorem rectangle_perimeter (x y : ℝ) 
  (h1 : 2*x + 2*y = x/2 + 2*y + 18) 
  (h2 : x*y = x*y/4 + 18) : 
  2*x + 2*y = 28 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3073_307361


namespace NUMINAMATH_CALUDE_exponential_comparison_l3073_307309

theorem exponential_comparison : 0.3^2.1 < 2.1^0.3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_comparison_l3073_307309


namespace NUMINAMATH_CALUDE_pencil_distribution_l3073_307317

theorem pencil_distribution (num_pens : ℕ) (num_pencils : ℕ) (num_students : ℕ) :
  num_pens = 1001 →
  num_students = 91 →
  (num_pens % num_students = 0) →
  (num_pencils % num_students = 0) →
  ∃ k : ℕ, num_pencils = 91 * k :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3073_307317


namespace NUMINAMATH_CALUDE_total_onions_grown_l3073_307366

/-- The total number of onions grown by Sara, Sally, and Fred is 18. -/
theorem total_onions_grown (sara_onions : ℕ) (sally_onions : ℕ) (fred_onions : ℕ)
  (h1 : sara_onions = 4)
  (h2 : sally_onions = 5)
  (h3 : fred_onions = 9) :
  sara_onions + sally_onions + fred_onions = 18 :=
by sorry

end NUMINAMATH_CALUDE_total_onions_grown_l3073_307366


namespace NUMINAMATH_CALUDE_order_of_abc_l3073_307330

theorem order_of_abc (a b c : ℝ) : 
  a = 2 * Real.log 1.01 →
  b = Real.log 1.02 →
  c = Real.sqrt 1.04 - 1 →
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_order_of_abc_l3073_307330


namespace NUMINAMATH_CALUDE_juice_distribution_l3073_307351

theorem juice_distribution (container_capacity : ℝ) : 
  container_capacity > 0 → 
  let total_juice := (3 / 4) * container_capacity
  let num_cups := 5
  let juice_per_cup := total_juice / num_cups
  (juice_per_cup / container_capacity) * 100 = 15 := by sorry

end NUMINAMATH_CALUDE_juice_distribution_l3073_307351


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l3073_307358

theorem sum_of_coefficients_equals_one 
  (a a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x : ℝ, (2*x - 3)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) : 
  a + a₁ + a₂ + a₃ + a₄ = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l3073_307358


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l3073_307374

/-- The capacity of a tank with specific inlet and outlet pipe characteristics -/
def tank_capacity : ℝ := 1280

/-- The time it takes for the outlet pipe to empty the full tank -/
def outlet_time : ℝ := 10

/-- The rate at which the inlet pipe fills the tank in litres per minute -/
def inlet_rate : ℝ := 8

/-- The additional time it takes to empty the tank when the inlet pipe is open -/
def additional_time : ℝ := 6

theorem tank_capacity_proof :
  tank_capacity = outlet_time * inlet_rate * 60 * (outlet_time + additional_time) / additional_time :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l3073_307374


namespace NUMINAMATH_CALUDE_change_in_expression_l3073_307365

/-- The original function -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

/-- The change in f when x is replaced by x + b -/
def delta_plus (x b : ℝ) : ℝ := f (x + b) - f x

/-- The change in f when x is replaced by x - b -/
def delta_minus (x b : ℝ) : ℝ := f (x - b) - f x

theorem change_in_expression (x b : ℝ) (h : b > 0) :
  (delta_plus x b = 3*x^2*b + 3*x*b^2 + b^3 - 2*b) ∧
  (delta_minus x b = -3*x^2*b + 3*x*b^2 - b^3 + 2*b) := by
  sorry

end NUMINAMATH_CALUDE_change_in_expression_l3073_307365


namespace NUMINAMATH_CALUDE_smallest_m_divisibility_l3073_307308

def a : ℕ → ℕ
  | 0 => 3
  | n + 1 => (n + 2) * a n - (n + 1)

def divisible (x y : ℕ) : Prop := ∃ k, y = k * x

theorem smallest_m_divisibility :
  ∀ m : ℕ, m ≥ 2005 →
    (divisible (a (m + 1) - 1) (a m ^ 2 - 1) ∧
     ∀ k : ℕ, 2005 ≤ k ∧ k < m →
       ¬divisible (a (k + 1) - 1) (a k ^ 2 - 1)) ↔
    m = 2010 := by sorry

end NUMINAMATH_CALUDE_smallest_m_divisibility_l3073_307308


namespace NUMINAMATH_CALUDE_quadratic_roots_opposite_l3073_307388

theorem quadratic_roots_opposite (a : ℝ) : 
  (∃ x y : ℝ, x^2 + (a^2 - 2*a)*x + (a - 1) = 0 ∧ 
               y^2 + (a^2 - 2*a)*y + (a - 1) = 0 ∧ 
               x = -y) → 
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_opposite_l3073_307388


namespace NUMINAMATH_CALUDE_no_integer_arithmetic_progression_l3073_307373

theorem no_integer_arithmetic_progression : 
  ¬ ∃ (a b : ℤ), (b - a = a - 6) ∧ (ab + 3 - b = b - a) := by sorry

end NUMINAMATH_CALUDE_no_integer_arithmetic_progression_l3073_307373


namespace NUMINAMATH_CALUDE_imaginary_unit_power_fraction_l3073_307332

theorem imaginary_unit_power_fraction (i : ℂ) (h : i^2 = -1) : 
  i^2015 / (1 + i) = (-1 - i) / 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_fraction_l3073_307332


namespace NUMINAMATH_CALUDE_four_type_B_in_rewards_l3073_307340

/-- Represents the cost and purchasing details of appliances A and B -/
structure AppliancePurchase where
  cost_A : ℕ  -- Cost of appliance A in yuan
  cost_B : ℕ  -- Cost of appliance B in yuan
  total_units : ℕ  -- Total units to purchase
  max_amount : ℕ  -- Maximum amount to spend in yuan
  max_A : ℕ  -- Maximum units of appliance A
  sell_A : ℕ  -- Selling price of appliance A in yuan
  sell_B : ℕ  -- Selling price of appliance B in yuan
  reward_units : ℕ  -- Units taken out for employee rewards
  profit : ℕ  -- Profit after selling remaining appliances in yuan

/-- Theorem stating that given the conditions, 4 units of type B are among the 10 reward units -/
theorem four_type_B_in_rewards (p : AppliancePurchase) 
  (h1 : p.cost_B = p.cost_A + 100)
  (h2 : 10000 / p.cost_A = 12000 / p.cost_B)
  (h3 : p.total_units = 100)
  (h4 : p.cost_A * 67 + p.cost_B * 33 ≤ p.max_amount)
  (h5 : p.max_A = 67)
  (h6 : p.sell_A = 600)
  (h7 : p.sell_B = 750)
  (h8 : p.reward_units = 10)
  (h9 : p.profit = 5050) :
  ∃ (a b : ℕ), a + b = p.reward_units ∧ b = 4 := by sorry

end NUMINAMATH_CALUDE_four_type_B_in_rewards_l3073_307340


namespace NUMINAMATH_CALUDE_jame_card_tearing_l3073_307306

/-- The number of cards in a new deck -/
def cards_per_deck : ℕ := 55

/-- The number of times Jame tears cards per week -/
def tear_times_per_week : ℕ := 3

/-- The number of decks Jame buys -/
def decks_bought : ℕ := 18

/-- The number of weeks Jame can go with the bought decks -/
def weeks_lasted : ℕ := 11

/-- The number of cards Jame can tear at a time -/
def cards_torn_at_once : ℕ := decks_bought * cards_per_deck / (weeks_lasted * tear_times_per_week)

theorem jame_card_tearing :
  cards_torn_at_once = 30 :=
sorry

end NUMINAMATH_CALUDE_jame_card_tearing_l3073_307306


namespace NUMINAMATH_CALUDE_sum_of_cubes_constraint_l3073_307371

theorem sum_of_cubes_constraint (a b : ℝ) :
  a^3 + b^3 = 1 - 3*a*b → (a + b = 1 ∨ a + b = -2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_constraint_l3073_307371


namespace NUMINAMATH_CALUDE_optimal_numbering_scheme_l3073_307353

/-- Represents a numbering scheme for a population --/
structure NumberingScheme where
  start : Nat
  digits : Nat

/-- Checks if a numbering scheme is valid for a given population size --/
def isValidScheme (populationSize : Nat) (scheme : NumberingScheme) : Prop :=
  scheme.start = 0 ∧
  scheme.digits = 3 ∧
  10 ^ scheme.digits > populationSize

/-- Theorem stating the optimal numbering scheme for the given conditions --/
theorem optimal_numbering_scheme
  (populationSize : Nat)
  (sampleSize : Nat)
  (h1 : populationSize = 106)
  (h2 : sampleSize = 10)
  (h3 : sampleSize < populationSize) :
  ∃ (scheme : NumberingScheme),
    isValidScheme populationSize scheme ∧
    scheme.start = 0 ∧
    scheme.digits = 3 :=
  sorry

end NUMINAMATH_CALUDE_optimal_numbering_scheme_l3073_307353


namespace NUMINAMATH_CALUDE_pyramid_sphere_radius_l3073_307392

-- Define the pyramid
structure RegularQuadrilateralPyramid where
  base_side : ℝ
  lateral_edge : ℝ

-- Define the spheres
structure Sphere where
  radius : ℝ

-- Define the problem
def pyramid_problem (p : RegularQuadrilateralPyramid) (q1 q2 : Sphere) : Prop :=
  p.base_side = 12 ∧
  p.lateral_edge = 10 ∧
  -- Q1 is inscribed in the pyramid (this is implied, not explicitly stated in Lean)
  -- Q2 touches Q1 and all lateral faces (this is implied, not explicitly stated in Lean)
  q2.radius = 6 * Real.sqrt 7 / 49

-- Theorem statement
theorem pyramid_sphere_radius 
  (p : RegularQuadrilateralPyramid) 
  (q1 q2 : Sphere) 
  (h : pyramid_problem p q1 q2) : 
  q2.radius = 6 * Real.sqrt 7 / 49 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_sphere_radius_l3073_307392


namespace NUMINAMATH_CALUDE_area_ratio_concentric_spheres_specific_sphere_areas_l3073_307313

/-- Given two concentric spheres with radii R₁ and R₂, if a region on the smaller sphere
    has an area A₁, then the corresponding region on the larger sphere has an area A₂. -/
theorem area_ratio_concentric_spheres (R₁ R₂ A₁ A₂ : ℝ) 
    (h₁ : R₁ > 0) (h₂ : R₂ > 0) (h₃ : A₁ > 0) :
  R₁ = 4 → R₂ = 6 → A₁ = 37 → A₂ = (R₂ / R₁)^2 * A₁ → A₂ = 83.25 := by
  sorry

/-- The specific case mentioned in the problem -/
theorem specific_sphere_areas :
  let R₁ : ℝ := 4
  let R₂ : ℝ := 6
  let A₁ : ℝ := 37
  let A₂ : ℝ := (R₂ / R₁)^2 * A₁
  A₂ = 83.25 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_concentric_spheres_specific_sphere_areas_l3073_307313


namespace NUMINAMATH_CALUDE_trisha_take_home_pay_l3073_307368

/-- Calculates the annual take-home pay for an hourly worker. -/
def annual_take_home_pay (hourly_rate : ℚ) (hours_per_week : ℕ) (weeks_per_year : ℕ) (withholding_rate : ℚ) : ℚ :=
  let gross_pay := hourly_rate * hours_per_week * weeks_per_year
  let withholding := withholding_rate * gross_pay
  gross_pay - withholding

/-- Proves that Trisha's annual take-home pay is $24,960 given the specified conditions. -/
theorem trisha_take_home_pay :
  annual_take_home_pay 15 40 52 (1/5) = 24960 := by
  sorry

#eval annual_take_home_pay 15 40 52 (1/5)

end NUMINAMATH_CALUDE_trisha_take_home_pay_l3073_307368


namespace NUMINAMATH_CALUDE_problem_solution_l3073_307385

/-- If 2a - b + 3 = 0, then 2(2a + b) - 4b = -6 -/
theorem problem_solution (a b : ℝ) (h : 2*a - b + 3 = 0) : 
  2*(2*a + b) - 4*b = -6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3073_307385


namespace NUMINAMATH_CALUDE_product_of_y_coordinates_l3073_307349

/-- Theorem: Product of y-coordinates for point Q -/
theorem product_of_y_coordinates (y₁ y₂ : ℝ) : 
  (((4 - (-2))^2 + (y₁ - (-3))^2 = 7^2) ∧
   ((4 - (-2))^2 + (y₂ - (-3))^2 = 7^2)) →
  y₁ * y₂ = -4 := by
sorry

end NUMINAMATH_CALUDE_product_of_y_coordinates_l3073_307349


namespace NUMINAMATH_CALUDE_two_tangent_lines_l3073_307391

-- Define the point P
def P : ℝ × ℝ := (-4, 1)

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2 = 1

-- Define a line passing through P
def line_through_P (m : ℝ) (x y : ℝ) : Prop :=
  y - P.2 = m * (x - P.1)

-- Define the condition for a line to intersect the hyperbola at only one point
def intersects_at_one_point (m : ℝ) : Prop :=
  ∃! (x y : ℝ), hyperbola x y ∧ line_through_P m x y

-- The main theorem
theorem two_tangent_lines :
  ∃! (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧ 
    intersects_at_one_point m₁ ∧ 
    intersects_at_one_point m₂ ∧
    ∀ m, intersects_at_one_point m → m = m₁ ∨ m = m₂ :=
  sorry

end NUMINAMATH_CALUDE_two_tangent_lines_l3073_307391


namespace NUMINAMATH_CALUDE_det_problem_l3073_307331

def det (a b d c : ℕ) : ℤ := a * c - b * d

theorem det_problem (b d : ℕ) (h : det 2 b d 4 = 2) : b + d = 5 ∨ b + d = 7 := by
  sorry

end NUMINAMATH_CALUDE_det_problem_l3073_307331


namespace NUMINAMATH_CALUDE_unique_prime_between_30_and_40_with_remainder_4_mod_9_l3073_307304

theorem unique_prime_between_30_and_40_with_remainder_4_mod_9 :
  ∃! n : ℕ, 30 < n ∧ n < 40 ∧ Prime n ∧ n % 9 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_between_30_and_40_with_remainder_4_mod_9_l3073_307304


namespace NUMINAMATH_CALUDE_kats_training_hours_l3073_307370

/-- The number of hours Kat trains per week -/
def total_training_hours (strength_sessions : ℕ) (strength_hours : ℝ) 
  (boxing_sessions : ℕ) (boxing_hours : ℝ) : ℝ :=
  (strength_sessions : ℝ) * strength_hours + (boxing_sessions : ℝ) * boxing_hours

/-- Theorem stating that Kat's total training hours per week is 9 -/
theorem kats_training_hours :
  total_training_hours 3 1 4 1.5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_kats_training_hours_l3073_307370


namespace NUMINAMATH_CALUDE_function_identity_l3073_307322

theorem function_identity (f : ℕ → ℕ) :
  (∀ x y : ℕ, 0 < x → 0 < y → f x + y * f (f x) ≤ x * (1 + f y)) →
  ∀ x : ℕ, f x = x :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l3073_307322


namespace NUMINAMATH_CALUDE_complement_intersection_real_l3073_307384

open Set

theorem complement_intersection_real (A B : Set ℝ) 
  (hA : A = {x : ℝ | 3 ≤ x ∧ x < 7})
  (hB : B = {x : ℝ | 2 < x ∧ x < 10}) :
  (A ∩ B)ᶜ = {x : ℝ | x < 3 ∨ 7 ≤ x} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_real_l3073_307384


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l3073_307342

theorem ratio_equation_solution : 
  let x : ℚ := 7 / 15
  (3 : ℚ) / 5 / ((6 : ℚ) / 7) = x / ((2 : ℚ) / 3) :=
by sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l3073_307342


namespace NUMINAMATH_CALUDE_perimeter_after_increase_l3073_307363

/-- Represents a triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_pos : 0 < a ∧ 0 < b ∧ 0 < c
  h_ineq : a < b + c ∧ b < a + c ∧ c < a + b

/-- The perimeter of a triangle. -/
def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Given a triangle, returns a new triangle with two sides increased by 4 and one by 1. -/
def increaseSides (t : Triangle) : Triangle where
  a := t.a + 4
  b := t.b + 4
  c := t.c + 1
  h_pos := sorry
  h_ineq := sorry

theorem perimeter_after_increase (t : Triangle) 
    (h1 : t.a = 8)
    (h2 : t.b = 5)
    (h3 : t.c = 6) :
    (increaseSides t).perimeter = 28 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_after_increase_l3073_307363


namespace NUMINAMATH_CALUDE_julie_school_year_hours_l3073_307377

/-- Given Julie's summer work details and school year earnings goal, calculate her required weekly hours during the school year. -/
theorem julie_school_year_hours 
  (summer_weeks : ℕ) 
  (summer_hours_per_week : ℕ) 
  (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) 
  (school_year_earnings : ℕ) 
  (h1 : summer_weeks = 12)
  (h2 : summer_hours_per_week = 40)
  (h3 : summer_earnings = 6000)
  (h4 : school_year_weeks = 36)
  (h5 : school_year_earnings = 9000) :
  (school_year_earnings * summer_weeks * summer_hours_per_week) / 
  (summer_earnings * school_year_weeks) = 20 := by
sorry

end NUMINAMATH_CALUDE_julie_school_year_hours_l3073_307377


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l3073_307327

theorem product_remainder_mod_five :
  (1024 * 1455 * 1776 * 2018 * 2222) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l3073_307327


namespace NUMINAMATH_CALUDE_simplify_expression_l3073_307345

theorem simplify_expression (y : ℝ) :
  3 * y - 7 * y^2 + 4 - (5 - 3 * y + 7 * y^2) = -14 * y^2 + 6 * y - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3073_307345


namespace NUMINAMATH_CALUDE_square_area_l3073_307321

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The line function -/
def line : ℝ := 8

theorem square_area : ∃ (x₁ x₂ : ℝ), 
  parabola x₁ = line ∧ 
  parabola x₂ = line ∧ 
  (x₂ - x₁)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l3073_307321


namespace NUMINAMATH_CALUDE_shaded_square_area_fraction_l3073_307389

/-- The area of a square with vertices at (2,1), (4,3), (2,5), and (0,3) divided by the area of a 5x5 square -/
theorem shaded_square_area_fraction : 
  let vertices : List (ℤ × ℤ) := [(2,1), (4,3), (2,5), (0,3)]
  let side_length := Real.sqrt ((4 - 2)^2 + (3 - 1)^2)
  let shaded_area := side_length ^ 2
  let grid_area := 5^2
  shaded_area / grid_area = 8 / 25 := by sorry

end NUMINAMATH_CALUDE_shaded_square_area_fraction_l3073_307389


namespace NUMINAMATH_CALUDE_remainder_problem_l3073_307362

theorem remainder_problem : (1989 * 1990 * 1991 + 1992^2) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3073_307362


namespace NUMINAMATH_CALUDE_tangent_line_quadratic_l3073_307346

theorem tangent_line_quadratic (a b : ℝ) : 
  (∀ x y : ℝ, y = x^2 + a*x + b) →
  (∀ x : ℝ, x + 1 = (0^2 + a*0 + b) + (2*0 + a)*x) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_quadratic_l3073_307346


namespace NUMINAMATH_CALUDE_initial_volume_calculation_l3073_307300

/-- The initial volume of a solution in liters -/
def initial_volume : ℝ := 6

/-- The percentage of alcohol in the initial solution -/
def initial_alcohol_percentage : ℝ := 0.30

/-- The volume of pure alcohol added in liters -/
def added_alcohol : ℝ := 2.4

/-- The percentage of alcohol in the final solution -/
def final_alcohol_percentage : ℝ := 0.50

theorem initial_volume_calculation :
  initial_volume * initial_alcohol_percentage + added_alcohol =
  final_alcohol_percentage * (initial_volume + added_alcohol) :=
by sorry

end NUMINAMATH_CALUDE_initial_volume_calculation_l3073_307300


namespace NUMINAMATH_CALUDE_non_shaded_area_of_square_with_semicircles_l3073_307336

/-- The area of the non-shaded part of a square with side length 4 and eight congruent semicircles --/
theorem non_shaded_area_of_square_with_semicircles :
  let square_side : ℝ := 4
  let num_semicircles : ℕ := 8
  let square_area : ℝ := square_side ^ 2
  let semicircle_radius : ℝ := square_side / 2
  let semicircle_area : ℝ := π * semicircle_radius ^ 2 / 2
  let total_shaded_area : ℝ := num_semicircles * semicircle_area
  let non_shaded_area : ℝ := square_area - total_shaded_area
  non_shaded_area = 8 := by sorry

end NUMINAMATH_CALUDE_non_shaded_area_of_square_with_semicircles_l3073_307336


namespace NUMINAMATH_CALUDE_negative_expressions_l3073_307359

/-- Represents a real number with an approximate value -/
structure ApproxReal where
  value : ℝ
  approx : ℝ
  is_close : |value - approx| < 0.1

/-- Given approximate values for U, V, W, X, and Y, prove which expressions are negative -/
theorem negative_expressions 
  (U : ApproxReal) (hU : U.approx = -4.6)
  (V : ApproxReal) (hV : V.approx = -2.0)
  (W : ApproxReal) (hW : W.approx = 0.2)
  (X : ApproxReal) (hX : X.approx = 1.3)
  (Y : ApproxReal) (hY : Y.approx = 2.2) :
  U.value - V.value < 0 ∧ 
  (X.value / V.value) * U.value < 0 ∧ 
  U.value * V.value ≥ 0 ∧ 
  W.value / (U.value * V.value) ≥ 0 ∧ 
  (X.value + Y.value) / W.value ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_expressions_l3073_307359


namespace NUMINAMATH_CALUDE_cupcake_flour_requirement_l3073_307333

-- Define the given quantities
def total_flour : ℝ := 6
def flour_for_cakes : ℝ := 4
def flour_per_cake : ℝ := 0.5
def flour_for_cupcakes : ℝ := 2
def price_per_cake : ℝ := 2.5
def price_per_cupcake : ℝ := 1
def total_earnings : ℝ := 30

-- Define the theorem
theorem cupcake_flour_requirement :
  ∃ (flour_per_cupcake : ℝ),
    flour_per_cupcake * (flour_for_cupcakes / flour_per_cupcake) = 
      total_earnings - (flour_for_cakes / flour_per_cake) * price_per_cake ∧
    flour_per_cupcake = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_flour_requirement_l3073_307333


namespace NUMINAMATH_CALUDE_c_share_calculation_l3073_307399

theorem c_share_calculation (total : ℚ) (a b c : ℚ) : 
  total = 392 →
  a = (1/2) * b →
  b = (1/2) * c →
  a + b + c = total →
  c = 224 := by
sorry

end NUMINAMATH_CALUDE_c_share_calculation_l3073_307399


namespace NUMINAMATH_CALUDE_journey_speed_problem_l3073_307381

/-- Proves that given a journey of 3 km, if traveling at speed v km/hr results in arriving 7 minutes late, 
    and traveling at 12 km/hr results in arriving 8 minutes early, then v = 6 km/hr. -/
theorem journey_speed_problem (v : ℝ) : 
  (3 / v - 3 / 12 = 15 / 60) → v = 6 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_problem_l3073_307381


namespace NUMINAMATH_CALUDE_no_upper_bound_expression_l3073_307383

/-- The expression has no upper bound -/
theorem no_upper_bound_expression (a b c d : ℝ) (h : a * d - b * c = 1) :
  ∀ M : ℝ, ∃ a' b' c' d' : ℝ, 
    a' * d' - b' * c' = 1 ∧ 
    a'^2 + b'^2 + c'^2 + d'^2 + a' * b' + c' * d' > M :=
by sorry

end NUMINAMATH_CALUDE_no_upper_bound_expression_l3073_307383


namespace NUMINAMATH_CALUDE_extreme_value_and_range_l3073_307312

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x + (1 - Real.log x) / x

theorem extreme_value_and_range (a : ℝ) :
  (∀ x > 0, f 0 x ≥ -1 / Real.exp 2) ∧
  (∀ x > 0, f 0 x = -1 / Real.exp 2 → x = Real.exp 2) ∧
  (∀ x > 0, f a x ≥ 1 ↔ a ≥ 1 / Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_and_range_l3073_307312


namespace NUMINAMATH_CALUDE_election_votes_l3073_307343

theorem election_votes :
  ∀ (V : ℕ) (geoff_votes : ℕ),
    geoff_votes = V / 100 →                     -- Geoff received 1% of votes
    geoff_votes + 3000 > V * 51 / 100 →         -- With 3000 more votes, Geoff would win
    geoff_votes + 3000 ≤ V * 51 / 100 + 1 →     -- Geoff needed exactly 3000 more votes to win
    V = 6000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l3073_307343


namespace NUMINAMATH_CALUDE_lindsay_workout_weight_l3073_307311

/-- Represents the resistance of exercise bands in pounds -/
structure Band where
  resistance : ℕ

/-- Represents a workout exercise with associated weights -/
structure Exercise where
  bands : List Band
  legWeights : ℕ
  additionalWeight : ℕ

/-- Calculates the total weight for an exercise -/
def totalWeight (e : Exercise) : ℕ :=
  (e.bands.map (λ b => b.resistance)).sum + 2 * e.legWeights + e.additionalWeight

/-- Lindsey's workout session -/
def lindseyWorkout : Prop :=
  let bandA : Band := ⟨7⟩
  let bandB : Band := ⟨5⟩
  let bandC : Band := ⟨3⟩
  let squats : Exercise := ⟨[bandA, bandB, bandC], 10, 15⟩
  let lunges : Exercise := ⟨[bandA, bandC], 8, 18⟩
  totalWeight squats + totalWeight lunges = 94

theorem lindsay_workout_weight : lindseyWorkout := by
  sorry

end NUMINAMATH_CALUDE_lindsay_workout_weight_l3073_307311


namespace NUMINAMATH_CALUDE_minimum_children_for_shared_birthday_l3073_307364

theorem minimum_children_for_shared_birthday (n : ℕ) : 
  (∀ f : Fin n → Fin 366, ∃ d : Fin 366, (∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ f i = f j ∧ f j = f k)) ↔ 
  n ≥ 733 :=
sorry

end NUMINAMATH_CALUDE_minimum_children_for_shared_birthday_l3073_307364


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3073_307324

theorem lcm_from_product_and_hcf (a b : ℕ+) :
  a * b = 45276 →
  Nat.gcd a b = 22 →
  Nat.lcm a b = 2058 := by
sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3073_307324


namespace NUMINAMATH_CALUDE_expression_simplification_l3073_307393

theorem expression_simplification (b y : ℝ) (hb : b > 0) (hy : y > 0) :
  (Real.sqrt (b^2 + y^2) - (y^2 - b^2) / Real.sqrt (b^2 + y^2)) / (b + y^2) = 2 * b^2 / (b + y^2) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3073_307393


namespace NUMINAMATH_CALUDE_division_reduction_l3073_307305

theorem division_reduction (original : ℕ) (divisor : ℕ) (reduction : ℕ) : 
  original = 72 → divisor = 3 → reduction = 48 → 
  (original : ℚ) / divisor = original - reduction :=
by
  sorry

end NUMINAMATH_CALUDE_division_reduction_l3073_307305


namespace NUMINAMATH_CALUDE_pen_price_relationship_l3073_307325

/-- The relationship between the number of pens and their total price -/
theorem pen_price_relationship (x y : ℝ) : y = (3/2) * x ↔ 
  ∃ (boxes : ℝ), 
    x = 12 * boxes ∧ 
    y = 18 * boxes :=
by sorry

end NUMINAMATH_CALUDE_pen_price_relationship_l3073_307325


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_4_l3073_307323

/-- A geometric sequence with its sum -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1  -- Geometric sequence property

/-- Theorem stating the conditions and the result to be proved -/
theorem geometric_sequence_sum_4 (seq : GeometricSequence) 
  (h1 : seq.a 1 - seq.a 2 = 2)
  (h2 : seq.a 2 - seq.a 3 = 6) :
  seq.S 4 = -40 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_4_l3073_307323


namespace NUMINAMATH_CALUDE_oil_bill_ratio_l3073_307335

/-- The oil bill problem -/
theorem oil_bill_ratio : 
  ∀ (feb_bill jan_bill : ℚ),
  jan_bill = 180 →
  (feb_bill + 45) / jan_bill = 3 / 2 →
  feb_bill / jan_bill = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_oil_bill_ratio_l3073_307335


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l3073_307354

/-- For the equation 7x^2-(m+13)x+m^2-m-2=0 to have one root greater than 1 
    and one root less than 1, m must satisfy -2 < m < 4 -/
theorem quadratic_root_condition (m : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ > 1 ∧ x₂ < 1 ∧ 
    7 * x₁^2 - (m + 13) * x₁ + m^2 - m - 2 = 0 ∧
    7 * x₂^2 - (m + 13) * x₂ + m^2 - m - 2 = 0) ↔
  -2 < m ∧ m < 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l3073_307354


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3073_307318

theorem negation_of_universal_statement :
  ¬(∀ a b : ℝ, a > b → a^2 > b^2) ↔ ∃ a b : ℝ, a ≤ b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3073_307318


namespace NUMINAMATH_CALUDE_great_wall_precision_l3073_307390

/-- The precision of a number in scientific notation is determined by the place value of its last significant digit. -/
def precision_scientific_notation (mantissa : ℝ) (exponent : ℤ) : ℕ :=
  sorry

/-- The Great Wall's length in scientific notation -/
def great_wall_length : ℝ := 6.7

/-- The exponent in the scientific notation of the Great Wall's length -/
def great_wall_exponent : ℤ := 6

/-- Hundred thousands place value -/
def hundred_thousands : ℕ := 100000

theorem great_wall_precision :
  precision_scientific_notation great_wall_length great_wall_exponent = hundred_thousands :=
sorry

end NUMINAMATH_CALUDE_great_wall_precision_l3073_307390


namespace NUMINAMATH_CALUDE_alyssa_balloons_l3073_307397

theorem alyssa_balloons (sandy_balloons sally_balloons total_balloons : ℕ) 
  (h1 : sandy_balloons = 28)
  (h2 : sally_balloons = 39)
  (h3 : total_balloons = 104) :
  total_balloons - (sandy_balloons + sally_balloons) = 37 :=
by sorry

end NUMINAMATH_CALUDE_alyssa_balloons_l3073_307397


namespace NUMINAMATH_CALUDE_power_relation_l3073_307302

theorem power_relation (a m n : ℝ) (h1 : a^(m+n) = 8) (h2 : a^(m-n) = 2) : a^(2*n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l3073_307302


namespace NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l3073_307348

theorem complex_equation_sum_of_squares (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a - 2 * i) * i = b - i →
  a^2 + b^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l3073_307348


namespace NUMINAMATH_CALUDE_boxer_weight_loss_l3073_307319

/-- Given a boxer's initial weight, monthly weight loss, and number of months until the fight,
    calculate the boxer's weight on the day of the fight. -/
def boxerFinalWeight (initialWeight monthlyLoss months : ℕ) : ℕ :=
  initialWeight - monthlyLoss * months

/-- Theorem stating that a boxer weighing 97 kg and losing 3 kg per month for 4 months
    will weigh 85 kg on the day of the fight. -/
theorem boxer_weight_loss : boxerFinalWeight 97 3 4 = 85 := by
  sorry

end NUMINAMATH_CALUDE_boxer_weight_loss_l3073_307319


namespace NUMINAMATH_CALUDE_birthday_puzzle_l3073_307360

theorem birthday_puzzle :
  ∃! (X Y : ℕ), 
    31 * X + 12 * Y = 376 ∧
    1 ≤ X ∧ X ≤ 12 ∧
    1 ≤ Y ∧ Y ≤ 31 ∧
    X = 9 ∧ Y = 8 := by
  sorry

end NUMINAMATH_CALUDE_birthday_puzzle_l3073_307360


namespace NUMINAMATH_CALUDE_no_valid_triples_l3073_307310

theorem no_valid_triples :
  ¬∃ (x y z : ℕ+), 
    (Nat.lcm x.val y.val = 180) ∧
    (Nat.lcm x.val z.val = 450) ∧
    (Nat.lcm y.val z.val = 600) ∧
    (x.val + y.val + z.val = 120) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_triples_l3073_307310


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l3073_307303

theorem complement_intersection_problem (U A B : Set ℕ) : 
  U = {0, 1, 2, 3} → 
  A = {1, 2} → 
  B = {3, 4} → 
  (U \ A) ∩ B = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l3073_307303


namespace NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l3073_307367

theorem integer_solutions_quadratic_equation :
  ∀ x y : ℤ, x + y = x^2 - x*y + y^2 ↔ 
    (x = 0 ∧ y = 0) ∨ 
    (x = 0 ∧ y = 1) ∨ 
    (x = 1 ∧ y = 0) ∨ 
    (x = 1 ∧ y = 2) ∨ 
    (x = 2 ∧ y = 1) ∨ 
    (x = 2 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l3073_307367


namespace NUMINAMATH_CALUDE_first_year_exceeding_target_l3073_307328

-- Define the initial investment and growth rate
def initial_investment : ℝ := 1.3
def growth_rate : ℝ := 0.12

-- Define the target investment
def target_investment : ℝ := 2.0

-- Define the function to calculate the investment for a given year
def investment (year : ℕ) : ℝ := initial_investment * (1 + growth_rate) ^ (year - 2015)

-- Theorem statement
theorem first_year_exceeding_target : 
  (∀ y : ℕ, y < 2019 → investment y ≤ target_investment) ∧ 
  investment 2019 > target_investment :=
sorry

end NUMINAMATH_CALUDE_first_year_exceeding_target_l3073_307328


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3073_307352

/-- The minimum distance from the origin (0, 0) to the line 2x + y + 5 = 0 is √5 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | 2 * p.1 + p.2 + 5 = 0}
  ∃ d : ℝ, d = Real.sqrt 5 ∧ ∀ p ∈ line, d ≤ Real.sqrt (p.1^2 + p.2^2) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3073_307352


namespace NUMINAMATH_CALUDE_sparrow_distribution_l3073_307372

theorem sparrow_distribution (a b c : ℕ) : 
  a + b + c = 24 →
  a - 4 = b + 1 →
  b + 1 = c + 3 →
  (a, b, c) = (12, 7, 5) := by
sorry

end NUMINAMATH_CALUDE_sparrow_distribution_l3073_307372


namespace NUMINAMATH_CALUDE_milan_phone_rate_l3073_307356

/-- Calculates the rate per minute for a phone service given the total bill, monthly fee, and minutes used. -/
def rate_per_minute (total_bill : ℚ) (monthly_fee : ℚ) (minutes : ℕ) : ℚ :=
  (total_bill - monthly_fee) / minutes

/-- Proves that given the specific conditions, the rate per minute is $0.12 -/
theorem milan_phone_rate :
  let total_bill : ℚ := 23.36
  let monthly_fee : ℚ := 2
  let minutes : ℕ := 178
  rate_per_minute total_bill monthly_fee minutes = 0.12 := by
sorry


end NUMINAMATH_CALUDE_milan_phone_rate_l3073_307356


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3073_307316

theorem inscribed_squares_ratio : 
  let triangle1 : ℝ × ℝ × ℝ := (5, 12, 13)
  let triangle2 : ℝ × ℝ × ℝ := (5, 12, 13)
  let a := (60 : ℝ) / 17  -- side length of square in triangle1
  let b := (65 : ℝ) / 17  -- side length of square in triangle2
  (a^2) / (b^2) = 3600 / 4225 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3073_307316


namespace NUMINAMATH_CALUDE_frog_escape_probability_l3073_307350

/-- Probability of the frog surviving when starting at pad N -/
noncomputable def P (N : ℕ) : ℝ :=
  sorry

/-- The number of lilypads -/
def num_pads : ℕ := 21

/-- The starting position of the frog -/
def start_pos : ℕ := 3

theorem frog_escape_probability :
  (∀ N : ℕ, 0 < N → N < num_pads - 1 →
    P N = (2 * N : ℝ) / 20 * P (N - 1) + (1 - (2 * N : ℝ) / 20) * P (N + 1)) →
  P 0 = 0 →
  P (num_pads - 1) = 1 →
  P start_pos = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_frog_escape_probability_l3073_307350


namespace NUMINAMATH_CALUDE_x_intercepts_count_l3073_307315

theorem x_intercepts_count (x : ℝ) : 
  (∃! x, (x - 4) * (x^2 + 4*x + 8) = 0) := by
  sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l3073_307315


namespace NUMINAMATH_CALUDE_monotonic_increasing_not_implies_positive_derivative_l3073_307334

theorem monotonic_increasing_not_implies_positive_derivative :
  ∃ (f : ℝ → ℝ) (a b : ℝ), a < b ∧
    (∀ x y, a < x ∧ x < y ∧ y < b → f x ≤ f y) ∧
    ¬(∀ x, a < x ∧ x < b → (deriv f x) > 0) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_not_implies_positive_derivative_l3073_307334


namespace NUMINAMATH_CALUDE_min_value_sum_l3073_307379

theorem min_value_sum (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h_sum : x₁^3 + x₂^3 + x₃^3 + x₄^3 + x₅^3 = 1) : 
  x₁/(1 - x₁^2) + x₂/(1 - x₂^2) + x₃/(1 - x₃^2) + x₄/(1 - x₄^2) + x₅/(1 - x₅^2) ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l3073_307379


namespace NUMINAMATH_CALUDE_square_division_l3073_307357

theorem square_division (a n : ℕ) : 
  a > 0 → 
  n > 1 → 
  a^2 = 88 + n^2 → 
  (a = 13 ∧ n = 9) ∨ (a = 23 ∧ n = 21) :=
by sorry

end NUMINAMATH_CALUDE_square_division_l3073_307357


namespace NUMINAMATH_CALUDE_unique_desk_arrangement_l3073_307301

theorem unique_desk_arrangement (total_desks : ℕ) (h_total : total_desks = 49) :
  ∃! (rows columns : ℕ),
    rows * columns = total_desks ∧
    rows ≥ 2 ∧
    columns ≥ 2 ∧
    (∀ r c : ℕ, r * c = total_desks → r ≥ 2 → c ≥ 2 → r = rows ∧ c = columns) :=
by sorry

end NUMINAMATH_CALUDE_unique_desk_arrangement_l3073_307301


namespace NUMINAMATH_CALUDE_books_per_bookshelf_l3073_307347

theorem books_per_bookshelf (total_books : ℕ) (num_bookshelves : ℕ) 
  (h1 : total_books = 621) 
  (h2 : num_bookshelves = 23) :
  total_books / num_bookshelves = 27 := by
  sorry

end NUMINAMATH_CALUDE_books_per_bookshelf_l3073_307347
