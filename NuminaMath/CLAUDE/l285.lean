import Mathlib

namespace NUMINAMATH_CALUDE_arctg_sum_quarter_pi_l285_28518

theorem arctg_sum_quarter_pi (a b : ℝ) : 
  a = (1 : ℝ) / 2 → 
  (a + 1) * (b + 1) = 2 → 
  Real.arctan a + Real.arctan b = π / 4 := by
sorry

end NUMINAMATH_CALUDE_arctg_sum_quarter_pi_l285_28518


namespace NUMINAMATH_CALUDE_profit_starts_third_year_option_one_more_profitable_l285_28568

/-- Represents the financial model of the fishing boat -/
structure FishingBoat where
  initialCost : ℕ
  firstYearExpenses : ℕ
  annualExpenseIncrease : ℕ
  annualIncome : ℕ

/-- Calculates the cumulative profit after n years -/
def cumulativeProfit (boat : FishingBoat) (n : ℕ) : ℤ :=
  n * boat.annualIncome - boat.initialCost - boat.firstYearExpenses
    - (n - 1) * boat.annualExpenseIncrease * n / 2

/-- Calculates the average profit after n years -/
def averageProfit (boat : FishingBoat) (n : ℕ) : ℚ :=
  (cumulativeProfit boat n : ℚ) / n

/-- The boat configuration from the problem -/
def problemBoat : FishingBoat :=
  { initialCost := 980000
    firstYearExpenses := 120000
    annualExpenseIncrease := 40000
    annualIncome := 500000 }

theorem profit_starts_third_year :
  ∀ n : ℕ, n < 3 → cumulativeProfit problemBoat n ≤ 0
  ∧ cumulativeProfit problemBoat 3 > 0 := by sorry

theorem option_one_more_profitable :
  let optionOne := cumulativeProfit problemBoat 7 + 260000
  let optionTwo := cumulativeProfit problemBoat 10 + 80000
  optionOne = optionTwo ∧ 7 < 10 := by sorry

end NUMINAMATH_CALUDE_profit_starts_third_year_option_one_more_profitable_l285_28568


namespace NUMINAMATH_CALUDE_orchid_planting_problem_l285_28536

/-- Calculates the number of orchid bushes to be planted -/
def orchids_to_plant (current : ℕ) (after : ℕ) : ℕ :=
  after - current

theorem orchid_planting_problem :
  let current_orchids : ℕ := 22
  let total_after_planting : ℕ := 35
  orchids_to_plant current_orchids total_after_planting = 13 := by
  sorry

end NUMINAMATH_CALUDE_orchid_planting_problem_l285_28536


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l285_28593

theorem sum_of_two_numbers (x y : ℝ) : x * y = 437 ∧ |x - y| = 4 → x + y = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l285_28593


namespace NUMINAMATH_CALUDE_full_house_prob_modified_deck_l285_28523

/-- Represents a modified deck of cards -/
structure ModifiedDeck :=
  (ranks : Nat)
  (cards_per_rank : Nat)
  (hand_size : Nat)

/-- Calculate the number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculate the probability of drawing a full house -/
def full_house_probability (deck : ModifiedDeck) : Rat :=
  let total_cards := deck.ranks * deck.cards_per_rank
  let total_combinations := choose total_cards deck.hand_size
  let full_house_combinations := 
    deck.ranks * choose deck.cards_per_rank 3 * (deck.ranks - 1) * choose deck.cards_per_rank 2
  full_house_combinations / total_combinations

/-- Theorem: The probability of drawing a full house in the given modified deck is 40/1292 -/
theorem full_house_prob_modified_deck :
  full_house_probability ⟨5, 4, 5⟩ = 40 / 1292 := by
  sorry


end NUMINAMATH_CALUDE_full_house_prob_modified_deck_l285_28523


namespace NUMINAMATH_CALUDE_cylinder_to_cone_volume_l285_28561

/-- Given a cylindrical block carved into the largest possible cone, 
    if the volume of the part removed is 25.12 cubic centimeters, 
    then the volume of the original cylindrical block is 37.68 cubic centimeters 
    and the volume of the cone-shaped block is 12.56 cubic centimeters. -/
theorem cylinder_to_cone_volume (removed_volume : ℝ) 
  (h : removed_volume = 25.12) : 
  ∃ (cylinder_volume cone_volume : ℝ),
    cylinder_volume = 37.68 ∧ 
    cone_volume = 12.56 ∧
    removed_volume = cylinder_volume - cone_volume := by
  sorry

end NUMINAMATH_CALUDE_cylinder_to_cone_volume_l285_28561


namespace NUMINAMATH_CALUDE_green_and_yellow_peaches_count_l285_28547

/-- Given a basket of peaches with different colors, this theorem proves
    that the sum of green and yellow peaches is 20. -/
theorem green_and_yellow_peaches_count (red : ℕ) (yellow : ℕ) (green : ℕ) 
    (h1 : red = 5) (h2 : yellow = 14) (h3 : green = 6) : 
    yellow + green = 20 := by
  sorry

end NUMINAMATH_CALUDE_green_and_yellow_peaches_count_l285_28547


namespace NUMINAMATH_CALUDE_max_sum_squares_given_sum_cubes_l285_28567

theorem max_sum_squares_given_sum_cubes 
  (a b c d : ℝ) 
  (h : a^3 + b^3 + c^3 + d^3 = 8) : 
  ∃ (m : ℝ), m = 4 ∧ ∀ (x y z w : ℝ), x^3 + y^3 + z^3 + w^3 = 8 → x^2 + y^2 + z^2 + w^2 ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_sum_squares_given_sum_cubes_l285_28567


namespace NUMINAMATH_CALUDE_average_race_time_l285_28583

/-- Calculates the average time in seconds for two racers to complete a block,
    given the time for one racer to complete the full block and the time for
    the other racer to complete half the block. -/
theorem average_race_time (carlos_time : ℝ) (diego_half_time : ℝ) : 
  carlos_time = 3 →
  diego_half_time = 2.5 →
  (carlos_time + 2 * diego_half_time) / 2 * 60 = 240 := by
  sorry

#check average_race_time

end NUMINAMATH_CALUDE_average_race_time_l285_28583


namespace NUMINAMATH_CALUDE_three_digit_factorial_sum_l285_28597

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def digit_factorial_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.map factorial).sum

def contains_digits (n : ℕ) (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d ∈ n.digits 10

theorem three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧
    contains_digits n [7, 2, 1] ∧
    n = digit_factorial_sum n :=
  sorry

end NUMINAMATH_CALUDE_three_digit_factorial_sum_l285_28597


namespace NUMINAMATH_CALUDE_tutors_next_meeting_l285_28575

/-- Anthony's work schedule in days -/
def anthony : ℕ := 5

/-- Beth's work schedule in days -/
def beth : ℕ := 6

/-- Carlos's work schedule in days -/
def carlos : ℕ := 8

/-- Diana's work schedule in days -/
def diana : ℕ := 10

/-- The number of days until all tutors work together again -/
def next_meeting : ℕ := 120

theorem tutors_next_meeting :
  Nat.lcm anthony (Nat.lcm beth (Nat.lcm carlos diana)) = next_meeting := by
  sorry

end NUMINAMATH_CALUDE_tutors_next_meeting_l285_28575


namespace NUMINAMATH_CALUDE_regression_line_equation_l285_28552

/-- Proves that the regression line equation is y = -x + 3 given the conditions -/
theorem regression_line_equation (b : ℝ) :
  (∃ (x y : ℝ), y = b * x + 3 ∧ (1, 2) = (x, y)) →
  (∀ (x y : ℝ), y = -x + 3) :=
by sorry

end NUMINAMATH_CALUDE_regression_line_equation_l285_28552


namespace NUMINAMATH_CALUDE_expression_evaluation_l285_28517

theorem expression_evaluation : 8 - 5 * (9 - (4 - 2)^2) * 2 = -42 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l285_28517


namespace NUMINAMATH_CALUDE_gcd_problem_l285_28529

theorem gcd_problem (a b : ℕ+) (h : Nat.gcd a b = 10) :
  (∃ (x y : ℕ+), Nat.gcd x y = 10 ∧ Nat.gcd (12 * x) (18 * y) = 60) ∧
  (∀ (c d : ℕ+), Nat.gcd c d = 10 → Nat.gcd (12 * c) (18 * d) ≥ 60) :=
sorry

end NUMINAMATH_CALUDE_gcd_problem_l285_28529


namespace NUMINAMATH_CALUDE_simultaneous_divisibility_by_17_l285_28594

theorem simultaneous_divisibility_by_17 : ∃ (x y : ℤ), 
  (17 ∣ (2*x + 3*y)) ∧ (17 ∣ (9*x + 5*y)) := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_divisibility_by_17_l285_28594


namespace NUMINAMATH_CALUDE_tan_double_angle_l285_28565

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) : 
  Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l285_28565


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l285_28585

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  -- Half of the base length
  b : ℝ
  -- Length of each equal side
  s : ℝ
  -- Altitude to the base
  h : ℝ
  -- Perimeter constraint
  perimeter_eq : 2 * s + 2 * b = 40
  -- Altitude constraint
  altitude_eq : h = 12
  -- Pythagorean theorem constraint
  pythagorean_eq : b^2 + h^2 = s^2

/-- The area of an isosceles triangle -/
def triangle_area (t : IsoscelesTriangle) : ℝ := t.b * t.h

/-- Theorem: The area of the specific isosceles triangle is 76.8 -/
theorem isosceles_triangle_area :
  ∀ t : IsoscelesTriangle, triangle_area t = 76.8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l285_28585


namespace NUMINAMATH_CALUDE_power_eleven_mod_120_l285_28569

theorem power_eleven_mod_120 : 11^2023 % 120 = 11 := by sorry

end NUMINAMATH_CALUDE_power_eleven_mod_120_l285_28569


namespace NUMINAMATH_CALUDE_su_buqing_star_distance_l285_28573

theorem su_buqing_star_distance (d : ℝ) : d = 218000000 → d = 2.18 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_su_buqing_star_distance_l285_28573


namespace NUMINAMATH_CALUDE_disjunction_implies_conjunction_false_l285_28590

theorem disjunction_implies_conjunction_false :
  ¬(∀ (p q : Prop), (p ∨ q) → (p ∧ q)) := by
  sorry

end NUMINAMATH_CALUDE_disjunction_implies_conjunction_false_l285_28590


namespace NUMINAMATH_CALUDE_janes_babysitting_ratio_l285_28507

/-- Represents the age ratio between a babysitter and a child -/
structure AgeRatio where
  babysitter : ℕ
  child : ℕ

/-- The problem setup for Jane's babysitting scenario -/
structure BabysittingScenario where
  jane_current_age : ℕ
  years_since_stopped : ℕ
  oldest_child_current_age : ℕ

/-- Calculates the age ratio between Jane and the oldest child she could have babysat -/
def calculate_age_ratio (scenario : BabysittingScenario) : AgeRatio :=
  { babysitter := scenario.jane_current_age - scenario.years_since_stopped,
    child := scenario.oldest_child_current_age - scenario.years_since_stopped }

/-- The main theorem to prove -/
theorem janes_babysitting_ratio :
  let scenario : BabysittingScenario := {
    jane_current_age := 34,
    years_since_stopped := 12,
    oldest_child_current_age := 25
  }
  let ratio := calculate_age_ratio scenario
  ratio.babysitter = 22 ∧ ratio.child = 13 := by sorry

end NUMINAMATH_CALUDE_janes_babysitting_ratio_l285_28507


namespace NUMINAMATH_CALUDE_remaining_balloons_l285_28506

def initial_balloons : ℕ := 709
def given_away : ℕ := 221

theorem remaining_balloons : initial_balloons - given_away = 488 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balloons_l285_28506


namespace NUMINAMATH_CALUDE_valid_allocations_count_l285_28500

/-- The number of male volunteers -/
def num_males : ℕ := 4

/-- The number of female volunteers -/
def num_females : ℕ := 3

/-- The total number of volunteers -/
def total_volunteers : ℕ := num_males + num_females

/-- The maximum number of people allowed in a group -/
def max_group_size : ℕ := 5

/-- A function to calculate the number of valid allocation plans -/
def count_valid_allocations : ℕ :=
  let three_four_split := (Nat.choose total_volunteers 3 - 1) * 2
  let two_five_split := (Nat.choose total_volunteers 2 - Nat.choose num_females 2) * 2
  three_four_split + two_five_split

/-- Theorem stating that the number of valid allocation plans is 104 -/
theorem valid_allocations_count : count_valid_allocations = 104 := by
  sorry


end NUMINAMATH_CALUDE_valid_allocations_count_l285_28500


namespace NUMINAMATH_CALUDE_at_least_one_geq_quarter_l285_28560

theorem at_least_one_geq_quarter (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) 
  (h_eq : x * y * z = (1 - x) * (1 - y) * (1 - z)) : 
  (1 - x) * y ≥ 1/4 ∨ (1 - y) * z ≥ 1/4 ∨ (1 - z) * x ≥ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_geq_quarter_l285_28560


namespace NUMINAMATH_CALUDE_unique_function_solution_l285_28554

/-- A function f: ℕ → ℤ is an increasing function that satisfies the given conditions -/
def IsValidFunction (f : ℕ → ℤ) : Prop :=
  (∀ m n : ℕ, m < n → f m < f n) ∧ 
  (f 2 = 7) ∧
  (∀ m n : ℕ, f (m * n) = f m + f n + f m * f n)

/-- The theorem stating that the only function satisfying the conditions is f(n) = n³ - 1 -/
theorem unique_function_solution :
  ∀ f : ℕ → ℤ, IsValidFunction f → ∀ n : ℕ, f n = n^3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_function_solution_l285_28554


namespace NUMINAMATH_CALUDE_maria_ate_two_cookies_l285_28595

/-- Given Maria's cookie distribution, prove she ate 2 cookies. -/
theorem maria_ate_two_cookies
  (initial_cookies : ℕ)
  (friend_cookies : ℕ)
  (final_cookies : ℕ)
  (h1 : initial_cookies = 19)
  (h2 : friend_cookies = 5)
  (h3 : final_cookies = 5)
  (h4 : ∃ (family_cookies : ℕ), 
    2 * family_cookies = initial_cookies - friend_cookies) :
  initial_cookies - friend_cookies - 
    ((initial_cookies - friend_cookies) / 2) - final_cookies = 2 :=
by sorry


end NUMINAMATH_CALUDE_maria_ate_two_cookies_l285_28595


namespace NUMINAMATH_CALUDE_video_game_lives_l285_28584

theorem video_game_lives (initial_lives lives_lost lives_gained : ℕ) :
  initial_lives - lives_lost + lives_gained = initial_lives + lives_gained - lives_lost :=
by sorry

#check video_game_lives 43 14 27

end NUMINAMATH_CALUDE_video_game_lives_l285_28584


namespace NUMINAMATH_CALUDE_car_speed_problem_l285_28576

/-- Proves that if a car traveling at 600 km/h takes 2 seconds longer to cover 1 km 
    than it would at speed v km/h, then v = 900 km/h. -/
theorem car_speed_problem (v : ℝ) : 
  (1 / (600 / 3600) - 1 / (v / 3600) = 2) → v = 900 := by
  sorry

#check car_speed_problem

end NUMINAMATH_CALUDE_car_speed_problem_l285_28576


namespace NUMINAMATH_CALUDE_even_n_with_specific_digit_sums_l285_28549

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem even_n_with_specific_digit_sums 
  (n : ℕ) 
  (n_positive : 0 < n) 
  (sum_n : sum_of_digits n = 2014) 
  (sum_5n : sum_of_digits (5 * n) = 1007) : 
  Even n := by sorry

end NUMINAMATH_CALUDE_even_n_with_specific_digit_sums_l285_28549


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l285_28592

/-- Given a person's income and savings, calculate the ratio of income to expenditure -/
theorem income_expenditure_ratio (income savings : ℕ) (h1 : income = 18000) (h2 : savings = 3600) :
  (income : ℚ) / (income - savings) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l285_28592


namespace NUMINAMATH_CALUDE_doctor_assignment_theorem_l285_28511

/-- Represents the number of doctors -/
def num_doctors : ℕ := 4

/-- Represents the number of companies -/
def num_companies : ℕ := 3

/-- Calculates the total number of valid assignment schemes -/
def total_assignments : ℕ := sorry

/-- Calculates the number of assignments when one doctor is fixed to a company -/
def fixed_doctor_assignments : ℕ := sorry

/-- Calculates the number of assignments when two doctors cannot be in the same company -/
def separated_doctors_assignments : ℕ := sorry

theorem doctor_assignment_theorem :
  (total_assignments = 36) ∧
  (fixed_doctor_assignments = 12) ∧
  (separated_doctors_assignments = 30) := by sorry

end NUMINAMATH_CALUDE_doctor_assignment_theorem_l285_28511


namespace NUMINAMATH_CALUDE_inscribed_parallelepiped_surface_area_l285_28532

/-- A parallelepiped inscribed in a sphere -/
structure InscribedParallelepiped where
  /-- The radius of the circumscribed sphere -/
  sphere_radius : ℝ
  /-- The volume of the parallelepiped -/
  volume : ℝ
  /-- The edges of the parallelepiped -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- The sphere radius is √3 -/
  sphere_radius_eq : sphere_radius = Real.sqrt 3
  /-- The volume is 8 -/
  volume_eq : volume = 8
  /-- The volume is the product of the edges -/
  volume_product : volume = a * b * c
  /-- The diagonal of the parallelepiped equals the diameter of the sphere -/
  diagonal_eq : a^2 + b^2 + c^2 = 4 * sphere_radius^2

/-- The theorem stating that the surface area of the inscribed parallelepiped is 24 -/
theorem inscribed_parallelepiped_surface_area
  (p : InscribedParallelepiped) : 2 * (p.a * p.b + p.b * p.c + p.c * p.a) = 24 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_parallelepiped_surface_area_l285_28532


namespace NUMINAMATH_CALUDE_translation_right_2_units_l285_28559

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point to the right by a given distance -/
def translateRight (p : Point) (d : ℝ) : Point :=
  { x := p.x + d, y := p.y }

theorem translation_right_2_units :
  let A : Point := { x := 1, y := 2 }
  let A' : Point := translateRight A 2
  A'.x = 3 ∧ A'.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_translation_right_2_units_l285_28559


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l285_28550

theorem unique_three_digit_number : ∃! n : ℕ,
  (100 ≤ n ∧ n ≤ 999) ∧
  (n % 11 = 0) ∧
  (n / 11 = (n / 100)^2 + ((n / 10) % 10)^2 + (n % 10)^2) ∧
  (n = 550) := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l285_28550


namespace NUMINAMATH_CALUDE_scooter_safety_gear_cost_increase_l285_28555

/-- The percent increase in the combined cost of a scooter and safety gear set --/
theorem scooter_safety_gear_cost_increase (scooter_cost safety_gear_cost : ℝ)
  (scooter_increase safety_gear_increase : ℝ) :
  scooter_cost = 200 →
  safety_gear_cost = 50 →
  scooter_increase = 0.08 →
  safety_gear_increase = 0.15 →
  let new_scooter_cost := scooter_cost * (1 + scooter_increase)
  let new_safety_gear_cost := safety_gear_cost * (1 + safety_gear_increase)
  let total_original_cost := scooter_cost + safety_gear_cost
  let total_new_cost := new_scooter_cost + new_safety_gear_cost
  let percent_increase := (total_new_cost - total_original_cost) / total_original_cost * 100
  ∃ ε > 0, |percent_increase - 9| < ε :=
by sorry

end NUMINAMATH_CALUDE_scooter_safety_gear_cost_increase_l285_28555


namespace NUMINAMATH_CALUDE_b_eventually_constant_iff_square_l285_28579

/-- The greatest integer m such that m^2 ≤ n -/
def m (n : ℕ) : ℕ := Nat.sqrt n

/-- d(n) = n - m^2, where m is the greatest integer such that m^2 ≤ n -/
def d (n : ℕ) : ℕ := n - (m n)^2

/-- The sequence b_i defined by b_{k+1} = b_k + d(b_k) -/
def b : ℕ → ℕ → ℕ
  | b_0, 0 => b_0
  | b_0, k + 1 => b b_0 k + d (b b_0 k)

/-- A sequence is eventually constant if there exists an N such that
    for all i ≥ N, the i-th term equals the N-th term -/
def EventuallyConstant (s : ℕ → ℕ) : Prop :=
  ∃ N, ∀ i, N ≤ i → s i = s N

/-- Main theorem: b_i is eventually constant iff b_0 is a perfect square -/
theorem b_eventually_constant_iff_square (b_0 : ℕ) :
  EventuallyConstant (b b_0) ↔ ∃ k, b_0 = k^2 := by sorry

end NUMINAMATH_CALUDE_b_eventually_constant_iff_square_l285_28579


namespace NUMINAMATH_CALUDE_sum_of_numbers_l285_28524

theorem sum_of_numbers : (6 / 5 : ℚ) + (1 / 10 : ℚ) + (156 / 100 : ℚ) = 286 / 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l285_28524


namespace NUMINAMATH_CALUDE_josh_marbles_l285_28586

/-- The number of marbles Josh has after losing some and giving away half of the remainder --/
def final_marbles (initial : ℕ) (lost : ℕ) : ℕ :=
  let remaining := initial - lost
  remaining - (remaining / 2)

/-- Theorem stating that Josh ends up with 103 marbles --/
theorem josh_marbles : final_marbles 320 115 = 103 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l285_28586


namespace NUMINAMATH_CALUDE_completing_square_transformation_l285_28520

theorem completing_square_transformation (x : ℝ) :
  x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l285_28520


namespace NUMINAMATH_CALUDE_tangent_ratio_l285_28510

/-- A cube with an inscribed sphere -/
structure CubeWithSphere where
  edge_length : ℝ
  sphere_radius : ℝ
  /- The sphere radius is half the edge length -/
  sphere_radius_eq : sphere_radius = edge_length / 2

/-- A point on the edge of a cube -/
structure EdgePoint (c : CubeWithSphere) where
  x : ℝ
  y : ℝ
  z : ℝ
  /- The point is on an edge -/
  on_edge : (x = 0 ∧ y = 0) ∨ (x = 0 ∧ z = 0) ∨ (y = 0 ∧ z = 0)

/-- A point on the inscribed sphere -/
structure SpherePoint (c : CubeWithSphere) where
  x : ℝ
  y : ℝ
  z : ℝ
  /- The point is on the sphere -/
  on_sphere : x^2 + y^2 + z^2 = c.sphere_radius^2

/-- Theorem: The ratio KE:EF is 4:5 -/
theorem tangent_ratio 
  (c : CubeWithSphere) 
  (K : EdgePoint c) 
  (E : SpherePoint c) 
  (F : EdgePoint c) 
  (h_K_midpoint : K.x = c.edge_length / 2 ∨ K.y = c.edge_length / 2 ∨ K.z = c.edge_length / 2)
  (h_tangent : ∃ t : ℝ, K.x + t * (E.x - K.x) = F.x ∧ 
                        K.y + t * (E.y - K.y) = F.y ∧ 
                        K.z + t * (E.z - K.z) = F.z)
  (h_skew : (F.x ≠ K.x ∨ F.y ≠ K.y) ∧ (F.y ≠ K.y ∨ F.z ≠ K.z) ∧ (F.x ≠ K.x ∨ F.z ≠ K.z)) :
  ∃ (ke ef : ℝ), ke / ef = 4 / 5 ∧ 
    ke^2 = (E.x - K.x)^2 + (E.y - K.y)^2 + (E.z - K.z)^2 ∧
    ef^2 = (F.x - E.x)^2 + (F.y - E.y)^2 + (F.z - E.z)^2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_ratio_l285_28510


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l285_28589

theorem consecutive_even_integers_sum (n : ℤ) : 
  (n + (n + 4) = 156) → (n + (n + 2) + (n + 4) = 234) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l285_28589


namespace NUMINAMATH_CALUDE_inequality_condition_l285_28512

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 + x) = f (2 - x)) ∧
  (∀ x y, x < y → x ≤ 2 → y ≤ 2 → f x < f y)

/-- The main theorem -/
theorem inequality_condition (f : ℝ → ℝ) (a : ℝ) 
  (h : special_function f) : 
  f (a^2 + 3*a + 2) < f (a^2 - a + 2) ↔ a > -1 ∧ a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l285_28512


namespace NUMINAMATH_CALUDE_quadratic_inequality_l285_28534

theorem quadratic_inequality (x : ℝ) : x^2 + 3*x - 18 < 0 ↔ -6 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l285_28534


namespace NUMINAMATH_CALUDE_range_of_p_l285_28535

-- Define set A
def A (p : ℝ) : Set ℝ := {x : ℝ | x^2 + (p+2)*x + 1 = 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x > 0}

-- Theorem statement
theorem range_of_p (p : ℝ) : (A p ∩ B = ∅) → p > -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l285_28535


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l285_28596

/-- The perimeter of a regular hexagon given its radius -/
theorem regular_hexagon_perimeter (radius : ℝ) : 
  radius = 3 → 6 * radius = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l285_28596


namespace NUMINAMATH_CALUDE_sum_to_60_l285_28599

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of integers from 1 to 60 is equal to 1830 -/
theorem sum_to_60 : sum_to_n 60 = 1830 := by
  sorry

end NUMINAMATH_CALUDE_sum_to_60_l285_28599


namespace NUMINAMATH_CALUDE_difference_of_percentages_l285_28502

theorem difference_of_percentages (x y : ℝ) : 
  0.60 * (50 + x) - 0.45 * (30 + y) = 16.5 + 0.60 * x - 0.45 * y :=
by sorry

end NUMINAMATH_CALUDE_difference_of_percentages_l285_28502


namespace NUMINAMATH_CALUDE_wire_service_reporters_l285_28503

theorem wire_service_reporters (total : ℝ) (x y both other_politics : ℝ) :
  x = 0.3 * total →
  y = 0.1 * total →
  both = 0.1 * total →
  other_politics = 0.25 * (x + y - both + other_politics) →
  total - (x + y - both + other_politics) = 0.45 * total :=
by sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l285_28503


namespace NUMINAMATH_CALUDE_bryans_collection_total_l285_28543

/-- Calculates the total number of reading materials in Bryan's collection --/
def total_reading_materials (
  num_shelves : ℕ
) (
  books_per_shelf : ℕ
) (
  magazines_per_shelf : ℕ
) (
  newspapers_per_shelf : ℕ
) (
  graphic_novels_per_shelf : ℕ
) : ℕ :=
  num_shelves * (books_per_shelf + magazines_per_shelf + newspapers_per_shelf + graphic_novels_per_shelf)

/-- Proves that Bryan's collection contains 4810 reading materials --/
theorem bryans_collection_total :
  total_reading_materials 37 23 61 17 29 = 4810 := by
  sorry

end NUMINAMATH_CALUDE_bryans_collection_total_l285_28543


namespace NUMINAMATH_CALUDE_right_triangle_angle_calculation_l285_28540

theorem right_triangle_angle_calculation (A B C : Real) : 
  A = 35 → C = 90 → A + B + C = 180 → B = 55 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_calculation_l285_28540


namespace NUMINAMATH_CALUDE_problem_solution_l285_28528

theorem problem_solution (A B : Set ℝ) (a b : ℝ) : 
  A = {2, 3} →
  B = {x : ℝ | x^2 + a*x + b = 0} →
  A ∩ B = {2} →
  A ∪ B = A →
  (a + b = 0 ∨ a + b = 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l285_28528


namespace NUMINAMATH_CALUDE_find_q_l285_28562

-- Define the polynomial g(x)
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p*x^4 + q*x^3 + r*x^2 + s*x + t

-- State the theorem
theorem find_q :
  ∀ p q r s t : ℝ,
  (∀ x : ℝ, g p q r s t x = 0 ↔ x = -2 ∨ x = 0 ∨ x = 1 ∨ x = 3) →
  g p q r s t 2 = -24 →
  q = 12 := by
sorry


end NUMINAMATH_CALUDE_find_q_l285_28562


namespace NUMINAMATH_CALUDE_clara_age_in_five_years_l285_28541

/-- Given the conditions about Alice and Clara's pens and ages, prove Clara's age in 5 years. -/
theorem clara_age_in_five_years
  (alice_pens : ℕ)
  (clara_pens_ratio : ℚ)
  (alice_age : ℕ)
  (clara_older : Prop)
  (pen_diff_equals_age_diff : Prop)
  (h1 : alice_pens = 60)
  (h2 : clara_pens_ratio = 2 / 5)
  (h3 : alice_age = 20)
  (h4 : clara_older)
  (h5 : pen_diff_equals_age_diff) :
  ∃ (clara_age : ℕ), clara_age + 5 = 61 :=
by sorry

end NUMINAMATH_CALUDE_clara_age_in_five_years_l285_28541


namespace NUMINAMATH_CALUDE_stating_table_tennis_outcomes_count_l285_28558

/-- Represents the number of possible outcomes in a table tennis match --/
def table_tennis_outcomes : ℕ := 30

/-- 
Theorem stating that the number of possible outcomes in a table tennis match,
where the first to win 3 games wins the match, is equal to 30.
--/
theorem table_tennis_outcomes_count :
  table_tennis_outcomes = 30 := by sorry

end NUMINAMATH_CALUDE_stating_table_tennis_outcomes_count_l285_28558


namespace NUMINAMATH_CALUDE_ophelia_pay_reaches_93_l285_28546

/-- Ophelia's weekly earnings function -/
def earnings (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 51
  else 51 + 100 * (n - 1)

/-- Average weekly pay after n weeks -/
def average_pay (n : ℕ) : ℚ :=
  if n = 0 then 0 else (earnings n) / n

/-- Theorem: Ophelia's average weekly pay reaches $93 after 7 weeks -/
theorem ophelia_pay_reaches_93 :
  ∃ n : ℕ, n > 0 ∧ average_pay n = 93 ∧ ∀ m : ℕ, m > 0 ∧ m < n → average_pay m < 93 :=
sorry

end NUMINAMATH_CALUDE_ophelia_pay_reaches_93_l285_28546


namespace NUMINAMATH_CALUDE_years_since_stopped_babysitting_l285_28519

/-- Represents the age when Jane started babysitting -/
def start_age : ℕ := 18

/-- Represents Jane's current age -/
def current_age : ℕ := 32

/-- Represents the current age of the oldest person Jane could have babysat -/
def oldest_babysat_current_age : ℕ := 23

/-- Represents the maximum age ratio between Jane and the children she babysat -/
def max_age_ratio : ℚ := 1/2

/-- Theorem stating that Jane stopped babysitting 14 years ago -/
theorem years_since_stopped_babysitting :
  current_age - (oldest_babysat_current_age - (start_age * max_age_ratio).floor) = 14 := by
  sorry

end NUMINAMATH_CALUDE_years_since_stopped_babysitting_l285_28519


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l285_28501

theorem arithmetic_series_sum (A : ℕ) : A = 380 := by
  sorry

#check arithmetic_series_sum

end NUMINAMATH_CALUDE_arithmetic_series_sum_l285_28501


namespace NUMINAMATH_CALUDE_room_occupancy_l285_28542

theorem room_occupancy (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) : 
  (5 : ℚ) / 6 * total_people = seated_people →
  (5 : ℚ) / 6 * total_chairs = seated_people →
  total_chairs - seated_people = 10 →
  total_people = 60 := by
sorry

end NUMINAMATH_CALUDE_room_occupancy_l285_28542


namespace NUMINAMATH_CALUDE_treats_calculation_l285_28563

/-- Calculates the number of treats per child per house -/
def treats_per_child_per_house (num_children : ℕ) (num_hours : ℕ) (houses_per_hour : ℕ) (total_treats : ℕ) : ℚ :=
  (total_treats : ℚ) / ((num_children : ℚ) * (num_hours * houses_per_hour))

/-- Theorem: Given the conditions from the problem, the number of treats per child per house is 3 -/
theorem treats_calculation :
  let num_children : ℕ := 3
  let num_hours : ℕ := 4
  let houses_per_hour : ℕ := 5
  let total_treats : ℕ := 180
  treats_per_child_per_house num_children num_hours houses_per_hour total_treats = 3 := by
  sorry

#eval treats_per_child_per_house 3 4 5 180

end NUMINAMATH_CALUDE_treats_calculation_l285_28563


namespace NUMINAMATH_CALUDE_max_abs_z_l285_28580

theorem max_abs_z (z : ℂ) (h : Complex.abs (z - (0 : ℂ) + 2 * Complex.I) = 1) :
  ∃ (M : ℝ), M = 3 ∧ ∀ w : ℂ, Complex.abs (w - (0 : ℂ) + 2 * Complex.I) = 1 → Complex.abs w ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_l285_28580


namespace NUMINAMATH_CALUDE_domain_of_composition_l285_28570

def f : Set ℝ → Prop := λ S => ∀ x ∈ S, 0 ≤ x ∧ x ≤ 1

theorem domain_of_composition (f : Set ℝ → Prop) (h : f (Set.Icc 0 1)) :
  f (Set.Icc 0 (1/2)) :=
sorry

end NUMINAMATH_CALUDE_domain_of_composition_l285_28570


namespace NUMINAMATH_CALUDE_function_zeros_sum_l285_28587

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - Real.exp (1 - x) - a * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x + Real.exp (1 - x) - a

theorem function_zeros_sum (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : g a x₁ = 0) 
  (h₂ : g a x₂ = 0) 
  (h₃ : f a x₁ + f a x₂ = -4) : 
  a = 4 := by sorry

end NUMINAMATH_CALUDE_function_zeros_sum_l285_28587


namespace NUMINAMATH_CALUDE_prob_box1_given_defective_l285_28578

-- Define the number of components and defective components in each box
def total_box1 : ℕ := 10
def defective_box1 : ℕ := 3
def total_box2 : ℕ := 20
def defective_box2 : ℕ := 2

-- Define the probability of selecting each box
def prob_select_box1 : ℚ := 1/2
def prob_select_box2 : ℚ := 1/2

-- Define the probability of selecting a defective component from each box
def prob_defective_given_box1 : ℚ := defective_box1 / total_box1
def prob_defective_given_box2 : ℚ := defective_box2 / total_box2

-- Define the overall probability of selecting a defective component
def prob_defective : ℚ := 
  prob_select_box1 * prob_defective_given_box1 + 
  prob_select_box2 * prob_defective_given_box2

-- State the theorem
theorem prob_box1_given_defective : 
  (prob_select_box1 * prob_defective_given_box1) / prob_defective = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_box1_given_defective_l285_28578


namespace NUMINAMATH_CALUDE_exp_sum_rule_l285_28508

theorem exp_sum_rule (a b : ℝ) : Real.exp a * Real.exp b = Real.exp (a + b) := by
  sorry

end NUMINAMATH_CALUDE_exp_sum_rule_l285_28508


namespace NUMINAMATH_CALUDE_sams_dimes_l285_28527

/-- Sam's dimes problem -/
theorem sams_dimes (initial_dimes given_away_dimes : ℕ) 
  (h1 : initial_dimes = 9)
  (h2 : given_away_dimes = 7) :
  initial_dimes - given_away_dimes = 2 := by
  sorry

end NUMINAMATH_CALUDE_sams_dimes_l285_28527


namespace NUMINAMATH_CALUDE_focus_coordinates_l285_28588

/-- A parabola is defined by the equation x^2 = 4y -/
structure Parabola where
  equation : ∀ x y : ℝ, x^2 = 4*y

/-- The focus of a parabola is a point (h, k) on its axis of symmetry -/
structure Focus (p : Parabola) where
  h : ℝ
  k : ℝ

/-- Theorem: The focus of the parabola x^2 = 4y has coordinates (0, 1) -/
theorem focus_coordinates (p : Parabola) : 
  ∃ f : Focus p, f.h = 0 ∧ f.k = 1 := by
  sorry

end NUMINAMATH_CALUDE_focus_coordinates_l285_28588


namespace NUMINAMATH_CALUDE_special_function_properties_l285_28530

/-- A function satisfying f(x+y) = f(x) + f(y) for all x, y, and f(x) > 0 for x > 0 -/
class SpecialFunction (f : ℝ → ℝ) :=
  (add : ∀ x y : ℝ, f (x + y) = f x + f y)
  (pos : ∀ x : ℝ, x > 0 → f x > 0)

/-- The main theorem stating that a SpecialFunction is odd and monotonically increasing -/
theorem special_function_properties (f : ℝ → ℝ) [SpecialFunction f] :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l285_28530


namespace NUMINAMATH_CALUDE_problem_solution_l285_28516

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x^3 - x^2 + 5*x + (1 - a) * Real.log x

theorem problem_solution :
  (∃ a : ℝ, (∀ x : ℝ, (deriv (f a)) x = 0 ↔ x = 1) ∧ a = 1) ∧
  (∃ x : ℝ, (deriv (f 0)) x = -1) ∧
  (¬∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧
    ∃ d : ℝ, x₂ = x₁ + d ∧ x₃ = x₂ + d ∧
    (deriv (f 2)) x₂ = (f 2 x₃ - f 2 x₁) / (x₃ - x₁)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l285_28516


namespace NUMINAMATH_CALUDE_easter_egg_distribution_l285_28574

theorem easter_egg_distribution (baskets : ℕ) (eggs_per_basket : ℕ) (people : ℕ) :
  baskets = 15 →
  eggs_per_basket = 12 →
  people = 20 →
  (baskets * eggs_per_basket) / people = 9 := by
sorry

end NUMINAMATH_CALUDE_easter_egg_distribution_l285_28574


namespace NUMINAMATH_CALUDE_trip_duration_l285_28551

theorem trip_duration (duration_first : ℝ) 
  (h1 : duration_first ≥ 0)
  (h2 : duration_first + 2 * duration_first + 2 * duration_first = 10) :
  duration_first = 2 := by
sorry

end NUMINAMATH_CALUDE_trip_duration_l285_28551


namespace NUMINAMATH_CALUDE_expression_factorization_l285_28509

theorem expression_factorization (x y z : ℝ) :
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / 
  ((x - y)^3 + (y - z)^3 + (z - x)^3) = 
  (x + y) * (y + z) * (z + x) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l285_28509


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_nine_l285_28539

theorem smallest_four_digit_mod_nine : ∃ n : ℕ, 
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 9 = 5) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 9 = 5 → m ≥ n) ∧ 
  (n = 1004) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_nine_l285_28539


namespace NUMINAMATH_CALUDE_expression_value_l285_28515

theorem expression_value : 
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 4
  x^2 * y * z - x * y * z^2 = 48 := by sorry

end NUMINAMATH_CALUDE_expression_value_l285_28515


namespace NUMINAMATH_CALUDE_sum_distinct_prime_divisors_of_1728_l285_28564

theorem sum_distinct_prime_divisors_of_1728 : 
  (Finset.sum (Finset.filter Nat.Prime (Nat.divisors 1728)) id) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_distinct_prime_divisors_of_1728_l285_28564


namespace NUMINAMATH_CALUDE_fifth_term_is_89_l285_28544

def sequence_rule (seq : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → seq (n + 1) = (seq n + seq (n + 2)) / 3

theorem fifth_term_is_89 (seq : ℕ → ℕ) (h_rule : sequence_rule seq) 
  (h_first : seq 1 = 2) (h_fourth : seq 4 = 34) : seq 5 = 89 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_89_l285_28544


namespace NUMINAMATH_CALUDE_sets_problem_l285_28545

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}
def B : Set ℝ := {x | 0 < x ∧ x < 5}
def C (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ 2*m}

-- Theorem statement
theorem sets_problem :
  (∀ x : ℝ, x ∈ A ∩ B ↔ 4 ≤ x ∧ x < 5) ∧
  (∀ x : ℝ, x ∈ (Set.univ \ A) ∪ B ↔ -1 < x ∧ x < 5) ∧
  (∀ m : ℝ, B ∩ C m = C m ↔ m < -2 ∨ (2 < m ∧ m < 5/2)) :=
sorry

end NUMINAMATH_CALUDE_sets_problem_l285_28545


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l285_28513

/-- Represents the number of distinct balls -/
def num_balls : ℕ := 4

/-- Represents the number of distinct boxes -/
def num_boxes : ℕ := 4

/-- Calculates the number of ways to place all balls into boxes leaving exactly one box empty -/
def ways_one_empty : ℕ := sorry

/-- Calculates the number of ways to place all balls into boxes with exactly one box containing two balls -/
def ways_one_two_balls : ℕ := sorry

/-- Calculates the number of ways to place all balls into boxes leaving exactly two boxes empty -/
def ways_two_empty : ℕ := sorry

theorem ball_placement_theorem :
  ways_one_empty = 144 ∧
  ways_one_two_balls = 144 ∧
  ways_two_empty = 84 := by sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l285_28513


namespace NUMINAMATH_CALUDE_expression_reduction_l285_28514

theorem expression_reduction (a b c : ℝ) 
  (h1 : a^2 + c^2 - b^2 - 2*a*c ≠ 0)
  (h2 : a - b + c ≠ 0)
  (h3 : a - c + b ≠ 0) :
  (a^2 + b^2 - c^2 - 2*a*b) / (a^2 + c^2 - b^2 - 2*a*c) = 
  ((a - b + c) * (a - b - c)) / ((a - c + b) * (a - c - b)) := by
  sorry

end NUMINAMATH_CALUDE_expression_reduction_l285_28514


namespace NUMINAMATH_CALUDE_factor_expression_l285_28556

theorem factor_expression (a b : ℝ) : 9 * a^2 * b - b = b * (3 * a + 1) * (3 * a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l285_28556


namespace NUMINAMATH_CALUDE_machine_work_time_l285_28598

theorem machine_work_time (x : ℝ) 
  (h1 : (1 / (x + 6) + 1 / (x + 1) + 1 / (2 * x)) = 1 / x) 
  (h2 : x > 0) : x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_machine_work_time_l285_28598


namespace NUMINAMATH_CALUDE_red_stamp_price_l285_28525

theorem red_stamp_price 
  (red_count blue_count yellow_count : ℕ)
  (blue_price yellow_price : ℚ)
  (total_earnings : ℚ) :
  red_count = 20 →
  blue_count = 80 →
  yellow_count = 7 →
  blue_price = 4/5 →
  yellow_price = 2 →
  total_earnings = 100 →
  (red_count : ℚ) * (total_earnings - blue_count * blue_price - yellow_count * yellow_price) / red_count = 11/10 :=
by sorry

end NUMINAMATH_CALUDE_red_stamp_price_l285_28525


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l285_28553

theorem quadratic_roots_relation (a b c : ℚ) : 
  (∃ r s : ℚ, (5 * r^2 + 2 * r - 4 = 0) ∧ (5 * s^2 + 2 * s - 4 = 0) ∧ 
   (a * (r - 3)^2 + b * (r - 3) + c = 0) ∧ (a * (s - 3)^2 + b * (s - 3) + c = 0)) →
  c = 47/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l285_28553


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l285_28504

/-- Given a rectangle with sides a and b (in decimeters), prove that its perimeter is 20 decimeters
    if the sum of two sides is 10 and the sum of three sides is 14. -/
theorem rectangle_perimeter (a b : ℝ) : 
  a + b = 10 → a + a + b = 14 → 2 * (a + b) = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l285_28504


namespace NUMINAMATH_CALUDE_f_zero_gt_f_one_l285_28581

/-- A quadratic function f(x) = x^2 - 4x + m, where m is a real constant. -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + m

/-- Theorem stating that f(0) > f(1) for any real m. -/
theorem f_zero_gt_f_one (m : ℝ) : f m 0 > f m 1 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_gt_f_one_l285_28581


namespace NUMINAMATH_CALUDE_num_cases_hearts_D_l285_28531

/-- The number of cards in a standard deck without jokers -/
def totalCards : ℕ := 52

/-- The number of people among whom the cards are distributed -/
def numPeople : ℕ := 4

/-- The total number of hearts in the deck -/
def totalHearts : ℕ := 13

/-- The number of hearts A has -/
def heartsA : ℕ := 5

/-- The number of hearts B has -/
def heartsB : ℕ := 4

/-- Theorem stating the number of possible cases for D's hearts -/
theorem num_cases_hearts_D : 
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ k : ℕ, k ≤ totalHearts - heartsA - heartsB → 
    (∃ (heartsC heartsD : ℕ), 
      heartsC + heartsD = totalHearts - heartsA - heartsB ∧
      heartsD = k)) ∧
  (∀ k : ℕ, k > totalHearts - heartsA - heartsB → 
    ¬∃ (heartsC heartsD : ℕ), 
      heartsC + heartsD = totalHearts - heartsA - heartsB ∧
      heartsD = k) :=
by sorry

end NUMINAMATH_CALUDE_num_cases_hearts_D_l285_28531


namespace NUMINAMATH_CALUDE_large_cube_surface_area_l285_28572

-- Define the volume of a small cube
def small_cube_volume : ℝ := 512

-- Define the number of small cubes
def num_small_cubes : ℕ := 8

-- Define the function to calculate the side length of a cube given its volume
def side_length (volume : ℝ) : ℝ := volume ^ (1/3)

-- Define the function to calculate the surface area of a cube given its side length
def surface_area (side : ℝ) : ℝ := 6 * side^2

-- Theorem statement
theorem large_cube_surface_area :
  let small_side := side_length small_cube_volume
  let large_side := small_side * (num_small_cubes ^ (1/3))
  surface_area large_side = 1536 := by
  sorry

end NUMINAMATH_CALUDE_large_cube_surface_area_l285_28572


namespace NUMINAMATH_CALUDE_gum_per_nickel_l285_28522

/-- 
Given:
- initial_nickels: The number of nickels Quentavious started with
- remaining_nickels: The number of nickels Quentavious had left
- total_gum: The total number of gum pieces Quentavious received

Prove: The number of gum pieces per nickel is 2
-/
theorem gum_per_nickel 
  (initial_nickels : ℕ) 
  (remaining_nickels : ℕ) 
  (total_gum : ℕ) 
  (h1 : initial_nickels = 5)
  (h2 : remaining_nickels = 2)
  (h3 : total_gum = 6)
  : (total_gum : ℚ) / (initial_nickels - remaining_nickels : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gum_per_nickel_l285_28522


namespace NUMINAMATH_CALUDE_min_value_expression_l285_28548

theorem min_value_expression (x y : ℝ) : (x * y + 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l285_28548


namespace NUMINAMATH_CALUDE_problem_statement_l285_28521

theorem problem_statement (x y : ℝ) 
  (h1 : (4 : ℝ)^y = 1 / (8 * (Real.sqrt 2)^(x + 2)))
  (h2 : (9 : ℝ)^x * (3 : ℝ)^y = 3 * Real.sqrt 3) :
  (5 : ℝ)^(x + y) = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l285_28521


namespace NUMINAMATH_CALUDE_sunflower_majority_on_day_two_l285_28577

/-- Represents the proportion of sunflower seeds in the feeder on a given day -/
def sunflower_proportion (day : ℕ) : ℝ :=
  1 - (0.7 : ℝ) ^ day

/-- The day when more than half of the seeds are sunflower seeds -/
def target_day : ℕ := 2

theorem sunflower_majority_on_day_two :
  sunflower_proportion target_day > (1/2 : ℝ) :=
by sorry

#check sunflower_majority_on_day_two

end NUMINAMATH_CALUDE_sunflower_majority_on_day_two_l285_28577


namespace NUMINAMATH_CALUDE_students_taking_both_courses_l285_28537

/-- Given a class with the following properties:
  * total_students: The total number of students in the class
  * french_students: The number of students taking French
  * german_students: The number of students taking German
  * neither_students: The number of students taking neither French nor German
  
  Prove that the number of students taking both French and German is equal to
  french_students + german_students - (total_students - neither_students) -/
theorem students_taking_both_courses
  (total_students : ℕ)
  (french_students : ℕ)
  (german_students : ℕ)
  (neither_students : ℕ)
  (h1 : total_students = 87)
  (h2 : french_students = 41)
  (h3 : german_students = 22)
  (h4 : neither_students = 33) :
  french_students + german_students - (total_students - neither_students) = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_both_courses_l285_28537


namespace NUMINAMATH_CALUDE_max_intersections_circle_sine_l285_28557

/-- The maximum number of intersection points between a circle and sine curve --/
theorem max_intersections_circle_sine (h k : ℝ) : 
  (k ≥ -2 ∧ k ≤ 2) → 
  (∃ (n : ℕ), n ≤ 8 ∧ 
    (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = 4 ∧ y = Real.sin x → 
      (∃ (m : ℕ), m ≤ n ∧ 
        (∀ (p q : ℝ), (p - h)^2 + (q - k)^2 = 4 ∧ q = Real.sin p → 
          (x = p ∧ y = q) ∨ m > 1)))) ∧
  (∀ (m : ℕ), m > 8 → 
    (∃ (x y : ℝ), (x - h)^2 + (y - k)^2 = 4 ∧ y = Real.sin x ∧
      (∀ (p q : ℝ), (p - h)^2 + (q - k)^2 = 4 ∧ q = Real.sin p → 
        (x ≠ p ∨ y ≠ q)))) := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_circle_sine_l285_28557


namespace NUMINAMATH_CALUDE_holiday_savings_l285_28571

theorem holiday_savings (sam_savings : ℕ) (total_savings : ℕ) (victory_savings : ℕ) : 
  sam_savings = 1000 →
  total_savings = 1900 →
  victory_savings < sam_savings →
  victory_savings = total_savings - sam_savings →
  sam_savings - victory_savings = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_holiday_savings_l285_28571


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l285_28533

theorem ellipse_eccentricity (a b m n c : ℝ) : 
  a > b ∧ b > 0 ∧ m > 0 ∧ n > 0 →
  c^2 = a^2 - b^2 →
  c^2 = m^2 + n^2 →
  c^2 = a * m →
  n^2 = (2 * m^2 + c^2) / 2 →
  c / a = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l285_28533


namespace NUMINAMATH_CALUDE_tan_585_degrees_l285_28526

theorem tan_585_degrees : Real.tan (585 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_585_degrees_l285_28526


namespace NUMINAMATH_CALUDE_logical_equivalence_l285_28582

theorem logical_equivalence (P Q : Prop) :
  (¬P → Q) ↔ (¬Q → P) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l285_28582


namespace NUMINAMATH_CALUDE_parabola_circle_tangent_l285_28591

/-- Given a parabola and a circle, if the parabola's axis is tangent to the circle, then p = 2 -/
theorem parabola_circle_tangent (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- Parabola equation
  (∀ x y : ℝ, x^2 + y^2 - 6*x - 7 = 0) →  -- Circle equation
  (∀ x y : ℝ, x = -p/2) →  -- Parabola's axis equation
  (abs (-p/2 + 3) = 4) →  -- Tangency condition (distance from circle center to axis equals radius)
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangent_l285_28591


namespace NUMINAMATH_CALUDE_equation_solution_l285_28566

theorem equation_solution : ∃! x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ∧ x = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l285_28566


namespace NUMINAMATH_CALUDE_four_digit_difference_l285_28538

def original_number : ℕ := 201312210840

def is_valid_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ (d1 d2 d3 d4 : ℕ), 
    d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n ∧
    (d1 = 2 ∨ d1 = 0 ∨ d1 = 1 ∨ d1 = 3 ∨ d1 = 8 ∨ d1 = 4) ∧
    (d2 = 2 ∨ d2 = 0 ∨ d2 = 1 ∨ d2 = 3 ∨ d2 = 8 ∨ d2 = 4) ∧
    (d3 = 2 ∨ d3 = 0 ∨ d3 = 1 ∨ d3 = 3 ∨ d3 = 8 ∨ d3 = 4) ∧
    (d4 = 2 ∨ d4 = 0 ∨ d4 = 1 ∨ d4 = 3 ∨ d4 = 8 ∨ d4 = 4))

theorem four_digit_difference :
  ∃ (max min : ℕ), 
    is_valid_four_digit max ∧
    is_valid_four_digit min ∧
    (∀ n, is_valid_four_digit n → n ≤ max) ∧
    (∀ n, is_valid_four_digit n → min ≤ n) ∧
    max - min = 2800 := by sorry

end NUMINAMATH_CALUDE_four_digit_difference_l285_28538


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l285_28505

/-- Given a circle with equation x^2 + y^2 - 6x + 14y = -28, 
    the sum of the x-coordinate and y-coordinate of its center is -4 -/
theorem circle_center_coordinate_sum : 
  ∃ (h k : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 6*x + 14*y = -28 ↔ (x - h)^2 + (y - k)^2 = 30) ∧ h + k = -4 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l285_28505
