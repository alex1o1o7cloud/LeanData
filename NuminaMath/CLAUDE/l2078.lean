import Mathlib

namespace NUMINAMATH_CALUDE_sandy_molly_age_ratio_l2078_207829

/-- The ratio of Sandy's age to Molly's age -/
def age_ratio (sandy_age molly_age : ℕ) : ℚ :=
  sandy_age / molly_age

/-- Theorem stating that the ratio of Sandy's age to Molly's age is 7/9 -/
theorem sandy_molly_age_ratio :
  let sandy_age : ℕ := 63
  let molly_age : ℕ := sandy_age + 18
  age_ratio sandy_age molly_age = 7/9 := by
sorry

end NUMINAMATH_CALUDE_sandy_molly_age_ratio_l2078_207829


namespace NUMINAMATH_CALUDE_min_value_of_f_l2078_207844

/-- The quadratic function f(x) = x^2 + 6x + 8 -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 8

/-- The minimum value of f(x) is -1 -/
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2078_207844


namespace NUMINAMATH_CALUDE_no_valid_n_l2078_207898

theorem no_valid_n : ¬∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (y : ℤ), n^2 - 18*n + 80 = y^2) ∧ 
  (∃ (k : ℤ), 15 = n * k) := by
sorry

end NUMINAMATH_CALUDE_no_valid_n_l2078_207898


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l2078_207887

/-- The area between two concentric circles, where a chord of length 100 units
    is tangent to the smaller circle, is equal to 2500π square units. -/
theorem area_between_concentric_circles (R r : ℝ) : 
  R > r → r > 0 → R^2 - r^2 = 2500 → π * (R^2 - r^2) = 2500 * π := by
  sorry

#check area_between_concentric_circles

end NUMINAMATH_CALUDE_area_between_concentric_circles_l2078_207887


namespace NUMINAMATH_CALUDE_lemonade_stand_lemons_cost_l2078_207843

/-- Proves that the amount spent on lemons is $10 given the lemonade stand conditions --/
theorem lemonade_stand_lemons_cost (sugar_cost cups_cost : ℕ) 
  (cups_sold price_per_cup : ℕ) (profit : ℕ) :
  sugar_cost = 5 →
  cups_cost = 3 →
  cups_sold = 21 →
  price_per_cup = 4 →
  profit = 66 →
  ∃ (lemons_cost : ℕ),
    lemons_cost = 10 ∧
    profit = cups_sold * price_per_cup - (lemons_cost + sugar_cost + cups_cost) :=
by sorry

end NUMINAMATH_CALUDE_lemonade_stand_lemons_cost_l2078_207843


namespace NUMINAMATH_CALUDE_inequality_of_roots_l2078_207838

theorem inequality_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_roots_l2078_207838


namespace NUMINAMATH_CALUDE_set_equals_interval_l2078_207835

-- Define the set {x | x ≥ 2}
def S : Set ℝ := {x : ℝ | x ≥ 2}

-- Define the interval [2, +∞)
def I : Set ℝ := Set.Ici 2

-- Theorem stating that S is equal to I
theorem set_equals_interval : S = I := by sorry

end NUMINAMATH_CALUDE_set_equals_interval_l2078_207835


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l2078_207877

theorem opposite_of_negative_fraction :
  -(-(7 : ℚ) / 3) = 7 / 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l2078_207877


namespace NUMINAMATH_CALUDE_veranda_width_l2078_207891

/-- Veranda width problem -/
theorem veranda_width (room_length room_width veranda_area : ℝ) 
  (h1 : room_length = 17)
  (h2 : room_width = 12)
  (h3 : veranda_area = 132)
  (h4 : veranda_area = (room_length + 2 * w) * (room_width + 2 * w) - room_length * room_width)
  : w = 2 := by
  sorry

end NUMINAMATH_CALUDE_veranda_width_l2078_207891


namespace NUMINAMATH_CALUDE_tan_70_cos_10_identity_l2078_207865

theorem tan_70_cos_10_identity : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_70_cos_10_identity_l2078_207865


namespace NUMINAMATH_CALUDE_power_of_two_divisibility_l2078_207833

theorem power_of_two_divisibility (n : ℕ+) :
  (∃ k : ℕ, 2^n.val - 1 = 7 * k) ↔ (∃ m : ℕ, n.val = 3 * m) ∧
  ¬(∃ k : ℕ, 2^n.val + 1 = 7 * k) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_divisibility_l2078_207833


namespace NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l2078_207822

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x^2 = 1 is a quadratic equation -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f :=
sorry

end NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l2078_207822


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2078_207866

theorem algebraic_expression_value (x : ℝ) : 
  2 * x^2 + 3 * x + 7 = 8 → 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2078_207866


namespace NUMINAMATH_CALUDE_ping_pong_games_l2078_207819

theorem ping_pong_games (total_games : ℕ) (frankie_games carla_games : ℕ) : 
  total_games = 30 →
  frankie_games + carla_games = total_games →
  frankie_games = carla_games / 2 →
  carla_games = 20 := by
sorry

end NUMINAMATH_CALUDE_ping_pong_games_l2078_207819


namespace NUMINAMATH_CALUDE_convention_handshakes_l2078_207828

/-- The number of companies at the convention -/
def num_companies : ℕ := 5

/-- The number of representatives from each company -/
def reps_per_company : ℕ := 5

/-- The total number of people at the convention -/
def total_people : ℕ := num_companies * reps_per_company

/-- The number of people each person shakes hands with -/
def handshakes_per_person : ℕ := total_people - reps_per_company - 1

/-- The total number of handshakes at the convention -/
def total_handshakes : ℕ := (total_people * handshakes_per_person) / 2

theorem convention_handshakes :
  total_handshakes = 250 :=
by sorry

end NUMINAMATH_CALUDE_convention_handshakes_l2078_207828


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2078_207886

theorem algebraic_expression_value (a b : ℝ) (h : a - 3 * b = 0) :
  (a - (2 * a * b - b^2) / a) / ((a^2 - b^2) / a) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2078_207886


namespace NUMINAMATH_CALUDE_prob_different_given_alone_is_half_l2078_207868

/-- The number of people visiting tourist spots -/
def num_people : ℕ := 3

/-- The number of tourist spots -/
def num_spots : ℕ := 3

/-- The number of ways person A can visit a spot alone -/
def ways_A_alone : ℕ := num_spots * (num_spots - 1) * (num_spots - 1)

/-- The number of ways all three people can visit different spots -/
def ways_all_different : ℕ := num_spots * (num_spots - 1) * (num_spots - 2)

/-- The probability that three people visit different spots given that one person visits a spot alone -/
def prob_different_given_alone : ℚ := ways_all_different / ways_A_alone

theorem prob_different_given_alone_is_half : prob_different_given_alone = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_given_alone_is_half_l2078_207868


namespace NUMINAMATH_CALUDE_polynomial_coefficient_C_l2078_207869

theorem polynomial_coefficient_C (A B C D : ℤ) : 
  (∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ z : ℂ, z^6 - 15*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36 = 
      (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆)) ∧
    (r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 15)) → 
  C = -92 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_C_l2078_207869


namespace NUMINAMATH_CALUDE_fraction_simplification_l2078_207857

theorem fraction_simplification : (5 * 6 - 3 * 4) / (6 + 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2078_207857


namespace NUMINAMATH_CALUDE_cost_effective_plan_l2078_207858

/-- Represents the ticket purchasing scenario for a group of employees visiting a scenic spot. -/
structure TicketScenario where
  totalEmployees : ℕ
  regularPrice : ℕ
  groupDiscountRate : ℚ
  womenDiscountRate : ℚ
  minGroupSize : ℕ

/-- Calculates the cost of tickets with women's discount applied. -/
def womenDiscountCost (s : TicketScenario) (numWomen : ℕ) : ℚ :=
  s.regularPrice * s.womenDiscountRate * numWomen + s.regularPrice * (s.totalEmployees - numWomen)

/-- Calculates the cost of tickets with group discount applied. -/
def groupDiscountCost (s : TicketScenario) : ℚ :=
  s.totalEmployees * s.regularPrice * (1 - s.groupDiscountRate)

/-- Theorem stating the conditions for the most cost-effective ticket purchasing plan. -/
theorem cost_effective_plan (s : TicketScenario) (numWomen : ℕ) :
  s.totalEmployees = 30 ∧
  s.regularPrice = 80 ∧
  s.groupDiscountRate = 1/5 ∧
  s.womenDiscountRate = 1/2 ∧
  s.minGroupSize = 30 ∧
  numWomen ≤ s.totalEmployees →
  (numWomen < 12 → groupDiscountCost s < womenDiscountCost s numWomen) ∧
  (numWomen = 12 → groupDiscountCost s = womenDiscountCost s numWomen) ∧
  (numWomen > 12 → groupDiscountCost s > womenDiscountCost s numWomen) :=
by sorry

end NUMINAMATH_CALUDE_cost_effective_plan_l2078_207858


namespace NUMINAMATH_CALUDE_pencil_pen_difference_l2078_207847

-- Define the given conditions
def paige_pencils_home : ℕ := 15
def paige_pens_backpack : ℕ := 7

-- Define the theorem
theorem pencil_pen_difference : 
  paige_pencils_home - paige_pens_backpack = 8 := by
  sorry


end NUMINAMATH_CALUDE_pencil_pen_difference_l2078_207847


namespace NUMINAMATH_CALUDE_cost_of_green_pill_l2078_207854

/-- Prove that the cost of one green pill is $20 -/
theorem cost_of_green_pill (treatment_duration : ℕ) (daily_green_pills : ℕ) (daily_pink_pills : ℕ) 
  (total_cost : ℕ) : ℕ :=
by
  sorry

#check cost_of_green_pill 3 1 1 819

end NUMINAMATH_CALUDE_cost_of_green_pill_l2078_207854


namespace NUMINAMATH_CALUDE_lcm_36_65_l2078_207889

theorem lcm_36_65 : Nat.lcm 36 65 = 2340 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_65_l2078_207889


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l2078_207812

theorem sum_remainder_mod_nine : (88000 + 88002 + 87999 + 88001 + 88003 + 87998) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l2078_207812


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l2078_207845

-- Define the function f
def f (a x : ℝ) := |x - a| + |2*x - 2|

-- Theorem 1: Solution set of f(x) > 2 when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f 2 x > 2} = {x : ℝ | x < 2/3 ∨ x > 2} :=
sorry

-- Theorem 2: Range of a when f(x) ≥ 2 for all x
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 2) ↔ (a ≤ -1 ∨ a ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l2078_207845


namespace NUMINAMATH_CALUDE_holmium166_neutron_proton_difference_l2078_207861

/-- Properties of Holmium-166 isotope -/
structure Holmium166 where
  mass_number : ℕ
  proton_number : ℕ
  mass_number_eq : mass_number = 166
  proton_number_eq : proton_number = 67

/-- Theorem: The difference between neutrons and protons in Holmium-166 is 32 -/
theorem holmium166_neutron_proton_difference (ho : Holmium166) :
  ho.mass_number - ho.proton_number - ho.proton_number = 32 := by
  sorry

#check holmium166_neutron_proton_difference

end NUMINAMATH_CALUDE_holmium166_neutron_proton_difference_l2078_207861


namespace NUMINAMATH_CALUDE_gcd_product_l2078_207808

theorem gcd_product (a b A B : ℕ) (d D : ℕ+) :
  d = Nat.gcd a b →
  D = Nat.gcd A B →
  (d * D : ℕ) = Nat.gcd (a * A) (Nat.gcd (a * B) (Nat.gcd (b * A) (b * B))) :=
by sorry

end NUMINAMATH_CALUDE_gcd_product_l2078_207808


namespace NUMINAMATH_CALUDE_most_cost_effective_plan_l2078_207800

-- Define the prices of A and B type devices
def price_A : ℕ := 12000
def price_B : ℕ := 10000

-- Define the production capacities
def capacity_A : ℕ := 240
def capacity_B : ℕ := 180

-- Define the total number of devices to purchase
def total_devices : ℕ := 10

-- Define the budget constraint
def budget : ℕ := 110000

-- Define the minimum required production capacity
def min_capacity : ℕ := 2040

-- Theorem statement
theorem most_cost_effective_plan :
  ∃ (num_A num_B : ℕ),
    -- The total number of devices is 10
    num_A + num_B = total_devices ∧
    -- The total cost is within budget
    num_A * price_A + num_B * price_B ≤ budget ∧
    -- The total production capacity meets the minimum requirement
    num_A * capacity_A + num_B * capacity_B ≥ min_capacity ∧
    -- This is the most cost-effective plan
    ∀ (other_A other_B : ℕ),
      other_A + other_B = total_devices →
      other_A * capacity_A + other_B * capacity_B ≥ min_capacity →
      other_A * price_A + other_B * price_B ≥ num_A * price_A + num_B * price_B :=
by
  -- The proof goes here
  sorry

#check most_cost_effective_plan

end NUMINAMATH_CALUDE_most_cost_effective_plan_l2078_207800


namespace NUMINAMATH_CALUDE_power_three_1234_mod_5_l2078_207875

theorem power_three_1234_mod_5 : 3^1234 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_three_1234_mod_5_l2078_207875


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l2078_207851

theorem infinitely_many_solutions (b : ℝ) : 
  (∀ x, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l2078_207851


namespace NUMINAMATH_CALUDE_circle_center_and_radius_circle_properties_l2078_207840

theorem circle_center_and_radius 
  (x y : ℝ) : 
  x^2 + y^2 + 4*x - 6*y = 11 ↔ 
  (x + 2)^2 + (y - 3)^2 = 24 :=
by sorry

theorem circle_properties : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  center = (-2, 3) ∧ 
  radius = 2 * Real.sqrt 6 ∧
  ∀ (x y : ℝ), x^2 + y^2 + 4*x - 6*y = 11 ↔ 
  (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_circle_properties_l2078_207840


namespace NUMINAMATH_CALUDE_snack_packs_needed_l2078_207881

def trail_mix_pack_size : ℕ := 6
def granola_bar_pack_size : ℕ := 8
def fruit_cup_pack_size : ℕ := 4
def total_people : ℕ := 18

def min_packs_needed (pack_size : ℕ) (people : ℕ) : ℕ :=
  (people + pack_size - 1) / pack_size

theorem snack_packs_needed :
  (min_packs_needed trail_mix_pack_size total_people = 3) ∧
  (min_packs_needed granola_bar_pack_size total_people = 3) ∧
  (min_packs_needed fruit_cup_pack_size total_people = 5) :=
by sorry

end NUMINAMATH_CALUDE_snack_packs_needed_l2078_207881


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_neg_three_range_of_a_given_condition_l2078_207816

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 2|

-- Theorem for part 1
theorem solution_set_when_a_is_neg_three :
  {x : ℝ | f (-3) x ≥ 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

-- Theorem for part 2
theorem range_of_a_given_condition :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_neg_three_range_of_a_given_condition_l2078_207816


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2078_207831

theorem fraction_to_decimal : (7 : ℚ) / 125 = 0.056 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2078_207831


namespace NUMINAMATH_CALUDE_scientific_notation_of_1_5_million_l2078_207872

/-- Expresses 1.5 million in scientific notation -/
theorem scientific_notation_of_1_5_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1500000 = a * (10 : ℝ) ^ n ∧ a = 1.5 ∧ n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1_5_million_l2078_207872


namespace NUMINAMATH_CALUDE_min_perimeter_is_18_l2078_207860

/-- Represents a triangle with side lengths a and b, where a = AB = BC and b = AC -/
structure IsoscelesTriangle where
  a : ℕ
  b : ℕ

/-- Represents the incircle and excircles of the triangle -/
structure TriangleCircles (t : IsoscelesTriangle) where
  inradius : ℝ
  exradius_A : ℝ
  exradius_B : ℝ
  exradius_C : ℝ

/-- Represents the smaller circle φ -/
structure SmallerCircle (t : IsoscelesTriangle) (c : TriangleCircles t) where
  radius : ℝ

/-- Checks if the given triangle satisfies all the tangency conditions -/
def satisfiesTangencyConditions (t : IsoscelesTriangle) (c : TriangleCircles t) (φ : SmallerCircle t c) : Prop :=
  c.exradius_A = c.inradius + c.exradius_A ∧
  c.exradius_B = c.inradius + c.exradius_B ∧
  c.exradius_C = c.inradius + c.exradius_C ∧
  φ.radius = c.inradius - c.exradius_A

/-- The main theorem stating the minimum perimeter -/
theorem min_perimeter_is_18 :
  ∃ (t : IsoscelesTriangle) (c : TriangleCircles t) (φ : SmallerCircle t c),
    satisfiesTangencyConditions t c φ ∧
    ∀ (t' : IsoscelesTriangle) (c' : TriangleCircles t') (φ' : SmallerCircle t' c'),
      satisfiesTangencyConditions t' c' φ' →
      2 * t.a + t.b ≤ 2 * t'.a + t'.b ∧
      2 * t.a + t.b = 18 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_is_18_l2078_207860


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2078_207811

-- Problem 1
theorem problem_1 (a : ℝ) : a * a^3 - 5 * a^4 + (2 * a^2)^2 = 0 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : (2 * a + 3 * b) * (a - 2 * b) - 1/8 * a * (4 * a - 3 * b) = 3/2 * a^2 - 5/8 * a * b - 6 * b^2 := by sorry

-- Problem 3
theorem problem_3 : (-0.125)^2023 * 2^2024 * 4^2024 = -8 := by sorry

-- Problem 4
theorem problem_4 : (2 * (1/2 : ℝ) - (-1))^2 + ((1/2 : ℝ) - (-1)) * ((1/2 : ℝ) + (-1)) - 5 * (1/2 : ℝ) * ((1/2 : ℝ) - 2 * (-1)) = -3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2078_207811


namespace NUMINAMATH_CALUDE_probability_need_change_is_six_sevenths_l2078_207832

/-- Represents the cost of a toy in cents -/
def ToyCost := Fin 8 → Nat

/-- The machine with 8 toys -/
structure ToyMachine where
  toys : Fin 8
  costs : ToyCost
  favorite_toy_cost : costs 3 = 175  -- $1.75 is the 4th most expensive toy (index 3)

/-- Sam's initial money in quarters -/
def initial_quarters : Nat := 8

/-- Probability of needing to get change -/
def probability_need_change (m : ToyMachine) : Rat :=
  1 - (1 : Rat) / 7

/-- Main theorem: The probability of needing change is 6/7 -/
theorem probability_need_change_is_six_sevenths (m : ToyMachine) :
  probability_need_change m = 6 / 7 := by
  sorry

/-- All costs are between 25 cents and 2 dollars, decreasing by 25 cents each time -/
axiom cost_constraint (m : ToyMachine) :
  ∀ i : Fin 8, m.costs i = 200 - 25 * i.val

/-- The machine randomly selects one of the remaining toys each time -/
axiom random_selection (m : ToyMachine) : True

/-- The machine only accepts quarters -/
axiom quarters_only (m : ToyMachine) : True

end NUMINAMATH_CALUDE_probability_need_change_is_six_sevenths_l2078_207832


namespace NUMINAMATH_CALUDE_log_eight_x_equals_three_halves_l2078_207859

theorem log_eight_x_equals_three_halves (x : ℝ) :
  Real.log x / Real.log 8 = 3/2 → x = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_eight_x_equals_three_halves_l2078_207859


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l2078_207823

theorem arithmetic_geometric_sequence_product (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-9 : ℝ) < a₁ ∧ a₁ < a₂ ∧ a₂ < (-1 : ℝ)) →  -- Arithmetic sequence condition
  ((-9 : ℝ) < b₁ ∧ b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < (-1 : ℝ)) →  -- Geometric sequence condition
  (a₂ - a₁ = a₁ - (-9 : ℝ)) →  -- Arithmetic sequence property
  (b₂ * b₂ = b₁ * b₃) →  -- Geometric sequence property
  (b₂ * (a₂ - a₁) = (-8 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l2078_207823


namespace NUMINAMATH_CALUDE_lexus_cars_sold_l2078_207862

def total_cars : ℕ := 300

def audi_percent : ℚ := 10 / 100
def toyota_percent : ℚ := 25 / 100
def bmw_percent : ℚ := 15 / 100
def acura_percent : ℚ := 30 / 100

def other_brands_percent : ℚ := audi_percent + toyota_percent + bmw_percent + acura_percent

def lexus_percent : ℚ := 1 - other_brands_percent

theorem lexus_cars_sold : 
  ⌊(lexus_percent * total_cars : ℚ)⌋ = 60 := by
  sorry

end NUMINAMATH_CALUDE_lexus_cars_sold_l2078_207862


namespace NUMINAMATH_CALUDE_problem_solution_l2078_207842

theorem problem_solution (a b : ℕ+) 
  (sum_constraint : a + b = 30)
  (equation_constraint : 2 * a * b + 12 * a = 3 * b + 270) :
  a * b = 216 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2078_207842


namespace NUMINAMATH_CALUDE_two_layer_wallpaper_area_l2078_207834

/-- Given the total wallpaper area, wall area, and area covered by three layers,
    calculate the area covered by exactly two layers of wallpaper. -/
theorem two_layer_wallpaper_area
  (total_area : ℝ)
  (wall_area : ℝ)
  (three_layer_area : ℝ)
  (h1 : total_area = 300)
  (h2 : wall_area = 180)
  (h3 : three_layer_area = 40) :
  total_area - wall_area - three_layer_area = 80 := by
sorry

end NUMINAMATH_CALUDE_two_layer_wallpaper_area_l2078_207834


namespace NUMINAMATH_CALUDE_sand_amount_l2078_207846

/-- The amount of gravel bought by the company in tons -/
def gravel : ℝ := 5.91

/-- The total amount of material bought by the company in tons -/
def total_material : ℝ := 14.02

/-- The amount of sand bought by the company in tons -/
def sand : ℝ := total_material - gravel

theorem sand_amount : sand = 8.11 := by
  sorry

end NUMINAMATH_CALUDE_sand_amount_l2078_207846


namespace NUMINAMATH_CALUDE_unique_k_exists_l2078_207821

-- Define the sequence sum function
def S (n : ℕ) : ℤ := n^2 - 9*n

-- Define the k-th term of the sequence
def a (k : ℕ) : ℤ := S k - S (k-1)

-- State the theorem
theorem unique_k_exists (k : ℕ) :
  (∃ k, 5 < a k ∧ a k < 8) → k = 8 :=
sorry

end NUMINAMATH_CALUDE_unique_k_exists_l2078_207821


namespace NUMINAMATH_CALUDE_divisors_product_prime_factors_l2078_207817

theorem divisors_product_prime_factors :
  let divisors : List Nat := [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
  let A : Nat := divisors.prod
  (Nat.factors A).toFinset.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisors_product_prime_factors_l2078_207817


namespace NUMINAMATH_CALUDE_sqrt_2_power_n_equals_64_l2078_207873

theorem sqrt_2_power_n_equals_64 (n : ℕ) : Real.sqrt (2^n) = 64 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_power_n_equals_64_l2078_207873


namespace NUMINAMATH_CALUDE_seashell_collection_l2078_207837

/-- Theorem: Given an initial collection of 19 seashells and adding 6 more,
    the total number of seashells is 25. -/
theorem seashell_collection (initial : Nat) (added : Nat) (total : Nat) : 
  initial = 19 → added = 6 → total = initial + added → total = 25 := by
  sorry

end NUMINAMATH_CALUDE_seashell_collection_l2078_207837


namespace NUMINAMATH_CALUDE_wicket_keeper_age_difference_l2078_207850

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  total_members : Nat
  average_age : Nat
  wicket_keeper_age : Nat
  haveProperty : (total_members - 2) * (average_age - 1) = (total_members * average_age - wicket_keeper_age - average_age)

/-- Theorem stating the age difference between the wicket keeper and the team average -/
theorem wicket_keeper_age_difference (team : CricketTeam)
  (h1 : team.total_members = 11)
  (h2 : team.average_age = 23) :
  team.wicket_keeper_age - team.average_age = 9 := by
  sorry

end NUMINAMATH_CALUDE_wicket_keeper_age_difference_l2078_207850


namespace NUMINAMATH_CALUDE_distance_between_points_l2078_207864

theorem distance_between_points (a b c : ℝ) : 
  Real.sqrt ((a - (a + 3))^2 + (b - (b + 7))^2 + (c - (c + 1))^2) = Real.sqrt 59 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2078_207864


namespace NUMINAMATH_CALUDE_horner_method_v3_l2078_207815

def horner_polynomial (x : ℤ) : ℤ := 12 + 35*x - 8*x^2 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v0 : ℤ := 3
def horner_v1 (x : ℤ) : ℤ := horner_v0 * x + 5
def horner_v2 (x : ℤ) : ℤ := horner_v1 x * x + 6
def horner_v3 (x : ℤ) : ℤ := horner_v2 x * x + 0

theorem horner_method_v3 :
  horner_v3 (-4) = -57 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v3_l2078_207815


namespace NUMINAMATH_CALUDE_sugar_for_muffins_sugar_for_muffins_proof_l2078_207809

/-- Given that 45 muffins require 3 cups of sugar, 
    prove that 135 muffins require 9 cups of sugar. -/
theorem sugar_for_muffins : ℝ → ℝ → ℝ → Prop :=
  fun muffins_base sugar_base muffins_target =>
    (muffins_base = 45 ∧ sugar_base = 3) →
    (muffins_target = 135) →
    (muffins_target * sugar_base / muffins_base = 9)

/-- Proof of the theorem -/
theorem sugar_for_muffins_proof : sugar_for_muffins 45 3 135 := by
  sorry

end NUMINAMATH_CALUDE_sugar_for_muffins_sugar_for_muffins_proof_l2078_207809


namespace NUMINAMATH_CALUDE_soccer_league_games_l2078_207876

theorem soccer_league_games (n : ℕ) (h : n = 12) : 
  (n * (n - 1)) / 2 = 66 := by
  sorry

#check soccer_league_games

end NUMINAMATH_CALUDE_soccer_league_games_l2078_207876


namespace NUMINAMATH_CALUDE_all_fruits_fallen_on_day_10_l2078_207825

/-- Represents the number of fruits that fall on a given day -/
def fruitsFallingOnDay (day : ℕ) : ℕ :=
  if day % 9 = 0 then 9 else day % 9

/-- Represents the total number of fruits that have fallen up to and including a given day -/
def totalFruitsFallen (day : ℕ) : ℕ :=
  (day / 9) * 45 + (day % 9) * (day % 9 + 1) / 2

/-- The theorem stating that all fruits will have fallen after 10 days -/
theorem all_fruits_fallen_on_day_10 (initial_fruits : ℕ) (h : initial_fruits = 46) :
  totalFruitsFallen 10 = initial_fruits :=
sorry

end NUMINAMATH_CALUDE_all_fruits_fallen_on_day_10_l2078_207825


namespace NUMINAMATH_CALUDE_ezekiel_new_shoes_l2078_207899

/-- The number of pairs of shoes Ezekiel bought -/
def pairs_bought : ℕ := 3

/-- The number of shoes in each pair -/
def shoes_per_pair : ℕ := 2

/-- The total number of new shoes Ezekiel has now -/
def total_new_shoes : ℕ := pairs_bought * shoes_per_pair

theorem ezekiel_new_shoes : total_new_shoes = 6 := by
  sorry

end NUMINAMATH_CALUDE_ezekiel_new_shoes_l2078_207899


namespace NUMINAMATH_CALUDE_bus_arrival_probability_l2078_207863

-- Define the probability of the bus arriving on time
def p : ℚ := 3/5

-- Define the probability of the bus not arriving on time
def q : ℚ := 1 - p

-- Define the function to calculate the probability of exactly k successes in n trials
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

-- State the theorem
theorem bus_arrival_probability :
  binomial_probability 3 2 p + binomial_probability 3 3 p = 81/125 := by
  sorry

end NUMINAMATH_CALUDE_bus_arrival_probability_l2078_207863


namespace NUMINAMATH_CALUDE_average_weight_l2078_207824

/-- Given three weights a, b, and c, prove that their average is 42 kg
    under the specified conditions. -/
theorem average_weight (a b c : ℝ) : 
  (a + b) / 2 = 40 →   -- The average weight of a and b is 40 kg
  (b + c) / 2 = 43 →   -- The average weight of b and c is 43 kg
  b = 40 →             -- The weight of b is 40 kg
  (a + b + c) / 3 = 42 -- The average weight of a, b, and c is 42 kg
  := by sorry

end NUMINAMATH_CALUDE_average_weight_l2078_207824


namespace NUMINAMATH_CALUDE_quadratic_roots_inequality_l2078_207807

theorem quadratic_roots_inequality (t : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 - t*x₁ + t = 0) → 
  (x₂^2 - t*x₂ + t = 0) → 
  (x₁^2 + x₂^2 ≥ 2*(x₁ + x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_inequality_l2078_207807


namespace NUMINAMATH_CALUDE_at_least_two_fever_probability_l2078_207879

def vaccine_fever_prob : ℝ := 0.80

def num_people : ℕ := 3

def at_least_two_fever_prob : ℝ := 
  (Nat.choose num_people 2) * (vaccine_fever_prob ^ 2) * (1 - vaccine_fever_prob) +
  vaccine_fever_prob ^ num_people

theorem at_least_two_fever_probability :
  at_least_two_fever_prob = 0.896 := by sorry

end NUMINAMATH_CALUDE_at_least_two_fever_probability_l2078_207879


namespace NUMINAMATH_CALUDE_jane_earnings_l2078_207882

/-- Represents the number of flower bulbs planted for each type --/
structure FlowerBulbs where
  tulips : ℕ
  iris : ℕ
  daffodils : ℕ
  crocus : ℕ

/-- Calculates the total earnings from planting flower bulbs --/
def calculate_earnings (bulbs : FlowerBulbs) (price_per_bulb : ℚ) : ℚ :=
  price_per_bulb * (bulbs.tulips + bulbs.iris + bulbs.daffodils + bulbs.crocus)

/-- The main theorem stating Jane's earnings --/
theorem jane_earnings : ∃ (bulbs : FlowerBulbs),
  bulbs.tulips = 20 ∧
  bulbs.iris = bulbs.tulips / 2 ∧
  bulbs.daffodils = 30 ∧
  bulbs.crocus = 3 * bulbs.daffodils ∧
  calculate_earnings bulbs (1/2) = 75 := by
  sorry


end NUMINAMATH_CALUDE_jane_earnings_l2078_207882


namespace NUMINAMATH_CALUDE_trailingZeros_2017_factorial_l2078_207884

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (Finset.range 4).sum fun i => (n / 5^(i + 1) : ℕ)

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => i + 1)

theorem trailingZeros_2017_factorial :
  trailingZeros 2017 = 502 :=
sorry

end NUMINAMATH_CALUDE_trailingZeros_2017_factorial_l2078_207884


namespace NUMINAMATH_CALUDE_fundraiser_customers_l2078_207895

/-- The number of customers who participated in the fundraiser -/
def num_customers : ℕ := 40

/-- The restaurant's donation ratio -/
def restaurant_ratio : ℚ := 2 / 10

/-- The average donation per customer -/
def avg_donation : ℚ := 3

/-- The total donation by the restaurant -/
def total_restaurant_donation : ℚ := 24

/-- Theorem stating that the number of customers is correct given the conditions -/
theorem fundraiser_customers :
  restaurant_ratio * (↑num_customers * avg_donation) = total_restaurant_donation :=
by sorry

end NUMINAMATH_CALUDE_fundraiser_customers_l2078_207895


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l2078_207820

/-- Given a circle with center (a, b) and radius r, returns the equation of the circle symmetric to it with respect to the line y = x -/
def symmetricCircle (a b r : ℝ) : (ℝ × ℝ → Prop) :=
  fun p => (p.1 - b)^2 + (p.2 - a)^2 = r^2

/-- The original circle (x-1)^2 + (y-2)^2 = 1 -/
def originalCircle : (ℝ × ℝ → Prop) :=
  fun p => (p.1 - 1)^2 + (p.2 - 2)^2 = 1

theorem symmetric_circle_equation :
  symmetricCircle 1 2 1 = fun p => (p.1 - 2)^2 + (p.2 - 1)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l2078_207820


namespace NUMINAMATH_CALUDE_tangent_slope_angle_sin_plus_cos_l2078_207803

theorem tangent_slope_angle_sin_plus_cos (x : Real) : 
  let f : Real → Real := λ x => Real.sin x + Real.cos x
  let f' : Real → Real := λ x => -Real.sin x + Real.cos x
  let slope : Real := f' (π/4)
  let slope_angle : Real := Real.arctan slope
  x = π/4 → slope_angle = 0 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_sin_plus_cos_l2078_207803


namespace NUMINAMATH_CALUDE_rhind_papyrus_bread_division_l2078_207813

theorem rhind_papyrus_bread_division :
  ∀ (a d : ℚ),
    d > 0 →
    5 * a = 100 →
    (1 / 7) * (a + (a + d) + (a + 2 * d)) = (a - 2 * d) + (a - d) →
    a - 2 * d = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_rhind_papyrus_bread_division_l2078_207813


namespace NUMINAMATH_CALUDE_expression_evaluation_l2078_207806

theorem expression_evaluation (a b c : ℝ) (ha : a = 14) (hb : b = 19) (hc : c = 13) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2078_207806


namespace NUMINAMATH_CALUDE_opposite_of_negative_half_l2078_207826

theorem opposite_of_negative_half :
  -(-(1/2 : ℚ)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_half_l2078_207826


namespace NUMINAMATH_CALUDE_f_of_g_5_l2078_207827

def g (x : ℝ) : ℝ := 3 * x - 4

def f (x : ℝ) : ℝ := 2 * x + 5

theorem f_of_g_5 : f (g 5) = 27 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_5_l2078_207827


namespace NUMINAMATH_CALUDE_school_trip_theorem_l2078_207818

/-- Represents the number of people initially planned per bus -/
def initial_people_per_bus : ℕ := 28

/-- Represents the number of students who couldn't get on the buses -/
def students_left_behind : ℕ := 13

/-- Represents the final number of people per bus -/
def final_people_per_bus : ℕ := 32

/-- Represents the number of empty seats per bus after redistribution -/
def empty_seats_per_bus : ℕ := 3

/-- Proves that the number of third-grade students is 125 and the number of buses rented is 4 -/
theorem school_trip_theorem :
  ∃ (num_students num_buses : ℕ),
    num_students = 125 ∧
    num_buses = 4 ∧
    num_students = initial_people_per_bus * num_buses + students_left_behind ∧
    num_students = final_people_per_bus * num_buses - empty_seats_per_bus * num_buses :=
by
  sorry

end NUMINAMATH_CALUDE_school_trip_theorem_l2078_207818


namespace NUMINAMATH_CALUDE_unique_interior_point_is_centroid_l2078_207855

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Predicate to check if a point is inside a triangle -/
def isInside (P : LatticePoint) (T : LatticeTriangle) : Prop :=
  sorry

/-- Predicate to check if a point is on the boundary of a triangle -/
def isOnBoundary (P : LatticePoint) (T : LatticeTriangle) : Prop :=
  sorry

/-- The centroid of a triangle -/
def centroid (T : LatticeTriangle) : LatticePoint :=
  sorry

/-- Main theorem -/
theorem unique_interior_point_is_centroid (T : LatticeTriangle) (P : LatticePoint) :
  (∀ Q : LatticePoint, isOnBoundary Q T → (Q = T.A ∨ Q = T.B ∨ Q = T.C)) →
  isInside P T →
  (∀ Q : LatticePoint, isInside Q T → Q = P) →
  P = centroid T :=
by sorry

end NUMINAMATH_CALUDE_unique_interior_point_is_centroid_l2078_207855


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_five_l2078_207839

theorem unique_square_divisible_by_five (y : ℕ) : 
  (∃ n : ℕ, y = n^2) ∧ 
  y % 5 = 0 ∧ 
  50 < y ∧ 
  y < 120 → 
  y = 100 := by
sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_five_l2078_207839


namespace NUMINAMATH_CALUDE_initial_women_count_l2078_207805

theorem initial_women_count (x y : ℕ) : 
  (y / (x - 15) = 2) → 
  ((y - 45) / (x - 15) = 1 / 5) → 
  x = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_women_count_l2078_207805


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_range_l2078_207849

-- Define the property of x that makes √(x-1) meaningful
def is_meaningful (x : ℝ) : Prop := x - 1 ≥ 0

-- Theorem stating the range of x where √(x-1) is meaningful
theorem sqrt_x_minus_one_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_range_l2078_207849


namespace NUMINAMATH_CALUDE_point_distance_range_l2078_207896

/-- Given points A(0,1) and B(0,4), and a point P on the line 2x-y+m=0 such that |PA| = 1/2|PB|,
    the range of values for m is -2√5 ≤ m ≤ 2√5. -/
theorem point_distance_range (m : ℝ) : 
  (∃ (x y : ℝ), 2*x - y + m = 0 ∧ 
    (x^2 + (y-1)^2)^(1/2) = 1/2 * (x^2 + (y-4)^2)^(1/2)) 
  ↔ -2 * Real.sqrt 5 ≤ m ∧ m ≤ 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_point_distance_range_l2078_207896


namespace NUMINAMATH_CALUDE_g_of_3_equals_10_l2078_207878

def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem g_of_3_equals_10 : g 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_10_l2078_207878


namespace NUMINAMATH_CALUDE_infinite_solution_equation_non_solutions_l2078_207801

/-- Given an equation with infinitely many solutions, prove the number and sum of non-solutions -/
theorem infinite_solution_equation_non_solutions (A B C : ℚ) : 
  (∀ x, (x + B) * (A * x + 42) = 3 * (x + C) * (x + 9)) →
  (∃! s : Finset ℚ, s.card = 2 ∧ 
    (∀ x ∈ s, (x + B) * (A * x + 42) ≠ 3 * (x + C) * (x + 9)) ∧
    (∀ x ∉ s, (x + B) * (A * x + 42) = 3 * (x + C) * (x + 9)) ∧
    s.sum id = -187/13) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solution_equation_non_solutions_l2078_207801


namespace NUMINAMATH_CALUDE_better_performance_criterion_l2078_207871

/-- Represents a shooter's performance statistics -/
structure ShooterStats where
  average_score : ℝ
  standard_deviation : ℝ

/-- Defines when a shooter has better performance than another -/
def better_performance (a b : ShooterStats) : Prop :=
  a.average_score > b.average_score ∧ a.standard_deviation < b.standard_deviation

/-- Theorem stating that a shooter with higher average score and lower standard deviation
    has better performance -/
theorem better_performance_criterion (shooter_a shooter_b : ShooterStats)
  (h1 : shooter_a.average_score > shooter_b.average_score)
  (h2 : shooter_a.standard_deviation < shooter_b.standard_deviation) :
  better_performance shooter_a shooter_b := by
  sorry

end NUMINAMATH_CALUDE_better_performance_criterion_l2078_207871


namespace NUMINAMATH_CALUDE_intersection_point_inside_circle_l2078_207836

theorem intersection_point_inside_circle (a : ℝ) : 
  let line1 : ℝ → ℝ := λ x => x + 2 * a
  let line2 : ℝ → ℝ := λ x => 2 * x + a + 1
  let P : ℝ × ℝ := (a - 1, 3 * a - 1)
  (∀ x y, y = line1 x ∧ y = line2 x → (x, y) = P) →
  P.1^2 + P.2^2 < 4 →
  -1/5 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_inside_circle_l2078_207836


namespace NUMINAMATH_CALUDE_factorization_proof_l2078_207890

theorem factorization_proof (a : ℝ) : 74 * a^2 + 222 * a + 148 = 74 * (a + 2) * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2078_207890


namespace NUMINAMATH_CALUDE_inequality_solution_l2078_207856

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -7/6 ∨ x > -4/3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2078_207856


namespace NUMINAMATH_CALUDE_complex_abs_3_minus_10i_l2078_207804

theorem complex_abs_3_minus_10i :
  let z : ℂ := 3 - 10*I
  Complex.abs z = Real.sqrt 109 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_3_minus_10i_l2078_207804


namespace NUMINAMATH_CALUDE_work_completion_time_l2078_207830

/-- Given two workers A and B, where A can complete a work in 10 days and B can complete the same work in 7 days, 
    this theorem proves that A and B working together can complete the work in 70/17 days. -/
theorem work_completion_time 
  (work : ℝ) -- Total amount of work
  (a_rate : ℝ) -- A's work rate
  (b_rate : ℝ) -- B's work rate
  (ha : a_rate = work / 10) -- A completes the work in 10 days
  (hb : b_rate = work / 7)  -- B completes the work in 7 days
  : work / (a_rate + b_rate) = 70 / 17 := by
sorry


end NUMINAMATH_CALUDE_work_completion_time_l2078_207830


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_two_l2078_207853

theorem least_subtraction_for_divisibility_by_two : 
  ∃ (n : ℕ), n = 1 ∧ 
  (∀ m : ℕ, (9671 - m) % 2 = 0 → m ≥ n) ∧
  (9671 - n) % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_two_l2078_207853


namespace NUMINAMATH_CALUDE_number_of_boys_l2078_207852

/-- The number of boys in a class, given the average weights and number of students. -/
theorem number_of_boys (avg_weight_boys : ℝ) (avg_weight_class : ℝ) (total_students : ℕ)
  (num_girls : ℕ) (avg_weight_girls : ℝ)
  (h1 : avg_weight_boys = 48)
  (h2 : avg_weight_class = 45)
  (h3 : total_students = 25)
  (h4 : num_girls = 15)
  (h5 : avg_weight_girls = 40.5) :
  total_students - num_girls = 10 := by
  sorry

#check number_of_boys

end NUMINAMATH_CALUDE_number_of_boys_l2078_207852


namespace NUMINAMATH_CALUDE_purchase_cost_l2078_207892

theorem purchase_cost (x y z : ℚ) 
  (eq1 : 4 * x + 9/2 * y + 12 * z = 6)
  (eq2 : 12 * x + 6 * y + 6 * z = 8) :
  4 * x + 3 * y + 6 * z = 4 := by
sorry

end NUMINAMATH_CALUDE_purchase_cost_l2078_207892


namespace NUMINAMATH_CALUDE_egg_roll_ratio_l2078_207810

def matthew_egg_rolls : ℕ := 6
def alvin_egg_rolls : ℕ := 4
def patrick_egg_rolls : ℕ := alvin_egg_rolls / 2

theorem egg_roll_ratio :
  matthew_egg_rolls / patrick_egg_rolls = 3 := by
  sorry

end NUMINAMATH_CALUDE_egg_roll_ratio_l2078_207810


namespace NUMINAMATH_CALUDE_disease_test_probability_l2078_207893

theorem disease_test_probability (disease_prevalence : ℝ) 
  (test_sensitivity : ℝ) (test_specificity : ℝ) : 
  disease_prevalence = 1/1000 →
  test_sensitivity = 1 →
  test_specificity = 0.95 →
  (disease_prevalence * test_sensitivity) / 
  (disease_prevalence * test_sensitivity + 
   (1 - disease_prevalence) * (1 - test_specificity)) = 100/5095 := by
sorry

end NUMINAMATH_CALUDE_disease_test_probability_l2078_207893


namespace NUMINAMATH_CALUDE_new_barbell_cost_l2078_207888

/-- The cost of a new barbell that is 30% more expensive than an old barbell priced at $250 is $325. -/
theorem new_barbell_cost (old_price : ℝ) (new_price : ℝ) : 
  old_price = 250 →
  new_price = old_price * 1.3 →
  new_price = 325 := by
  sorry

end NUMINAMATH_CALUDE_new_barbell_cost_l2078_207888


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l2078_207867

theorem geometric_arithmetic_sequence :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive numbers
    a = 2 ∧  -- First number is 2
    b / a = c / b ∧  -- Geometric sequence
    (b + 4 - a = c - (b + 4)) ∧  -- Arithmetic sequence when 4 is added to b
    a = 2 ∧ b = 6 ∧ c = 18 :=  -- The solution
by
  sorry


end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l2078_207867


namespace NUMINAMATH_CALUDE_coin_difference_l2078_207897

theorem coin_difference (total_coins quarters : ℕ) 
  (h1 : total_coins = 77)
  (h2 : quarters = 29) : 
  total_coins - quarters = 48 := by
  sorry

end NUMINAMATH_CALUDE_coin_difference_l2078_207897


namespace NUMINAMATH_CALUDE_additional_grazing_area_l2078_207814

theorem additional_grazing_area (π : ℝ) (h : π > 0) : 
  π * 23^2 - π * 16^2 = 273 * π := by
  sorry

end NUMINAMATH_CALUDE_additional_grazing_area_l2078_207814


namespace NUMINAMATH_CALUDE_don_buys_150_from_shop_A_l2078_207880

/-- The number of bottles Don buys from each shop -/
structure BottlePurchase where
  total : ℕ
  shopA : ℕ
  shopB : ℕ
  shopC : ℕ

/-- Don's bottle purchase satisfies the given conditions -/
def valid_purchase (p : BottlePurchase) : Prop :=
  p.total = 550 ∧ p.shopB = 180 ∧ p.shopC = 220 ∧ p.total = p.shopA + p.shopB + p.shopC

/-- Theorem: Don buys 150 bottles from Shop A -/
theorem don_buys_150_from_shop_A (p : BottlePurchase) (h : valid_purchase p) : p.shopA = 150 := by
  sorry

end NUMINAMATH_CALUDE_don_buys_150_from_shop_A_l2078_207880


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l2078_207848

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : (Real.sin α) / (Real.tan α) > 0) 
  (h2 : (Real.tan α) / (Real.cos α) < 0) : 
  0 < α ∧ α < π / 2 ∧ Real.sin α < 0 ∧ Real.cos α > 0 :=
sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l2078_207848


namespace NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l2078_207841

theorem circle_area_equilateral_triangle (s : ℝ) (h : s = 12) :
  let R := s / Real.sqrt 3
  (π * R^2) = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l2078_207841


namespace NUMINAMATH_CALUDE_ice_cream_parlor_distance_l2078_207894

/-- The distance to the ice cream parlor satisfies the equation relating to Rita's canoe trip --/
theorem ice_cream_parlor_distance :
  ∃ D : ℝ, (D / (3 - 2)) + (D / (9 + 4)) = 8 - 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_parlor_distance_l2078_207894


namespace NUMINAMATH_CALUDE_video_game_map_width_l2078_207874

/-- Represents the dimensions of a rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℝ :=
  prism.length * prism.width * prism.height

theorem video_game_map_width :
  ∀ (prism : RectangularPrism),
    volume prism = 50 →
    prism.length = 5 →
    prism.height = 2 →
    prism.width = 5 := by
  sorry

end NUMINAMATH_CALUDE_video_game_map_width_l2078_207874


namespace NUMINAMATH_CALUDE_max_arithmetic_mean_for_special_pair_l2078_207883

theorem max_arithmetic_mean_for_special_pair : ∃ (a b : ℕ), 
  10 ≤ a ∧ a ≤ 99 ∧ 
  10 ≤ b ∧ b ≤ 99 ∧ 
  a > b ∧
  (a + b) / 2 = (25 / 24) * Real.sqrt (a * b) ∧
  ∀ (c d : ℕ), 
    10 ≤ c ∧ c ≤ 99 ∧ 
    10 ≤ d ∧ d ≤ 99 ∧ 
    c > d ∧
    (c + d) / 2 = (25 / 24) * Real.sqrt (c * d) →
    (a + b) / 2 ≥ (c + d) / 2 ∧
  (a + b) / 2 = 75 := by
sorry

end NUMINAMATH_CALUDE_max_arithmetic_mean_for_special_pair_l2078_207883


namespace NUMINAMATH_CALUDE_f_one_half_equals_sixteen_l2078_207802

-- Define the function f
noncomputable def f : ℝ → ℝ := fun t => 1 / ((1 - t) / 2)^2

-- State the theorem
theorem f_one_half_equals_sixteen :
  (∀ x, f (1 - 2 * x) = 1 / x^2) → f (1/2) = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_f_one_half_equals_sixteen_l2078_207802


namespace NUMINAMATH_CALUDE_acme_vowel_soup_sequences_l2078_207885

/-- The number of vowels in the soup -/
def num_vowels : ℕ := 5

/-- The length of each sequence -/
def sequence_length : ℕ := 5

/-- The minimum number of times any vowel appears -/
def min_vowel_count : ℕ := 3

/-- The maximum number of times any vowel appears -/
def max_vowel_count : ℕ := 7

/-- The number of five-letter sequences that can be formed -/
def num_sequences : ℕ := num_vowels ^ sequence_length

theorem acme_vowel_soup_sequences :
  num_sequences = 3125 :=
sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_sequences_l2078_207885


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l2078_207870

theorem quadratic_form_equivalence (b : ℝ) (n : ℝ) :
  b < 0 →
  (∀ x, x^2 + b*x - 36 = (x + n)^2 - 20) →
  b = -8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l2078_207870
