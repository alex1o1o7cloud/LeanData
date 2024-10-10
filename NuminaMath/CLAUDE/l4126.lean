import Mathlib

namespace xy_problem_l4126_412617

theorem xy_problem (x y : ℝ) (h1 : x * y = 9) (h2 : x / y = 36) (hx : x > 0) (hy : y > 0) : y = 1 / 2 := by
  sorry

end xy_problem_l4126_412617


namespace B_power_60_is_identity_l4126_412685

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 0, 0],
    ![0, 0, -1],
    ![0, 1, 0]]

theorem B_power_60_is_identity :
  B ^ 60 = 1 := by sorry

end B_power_60_is_identity_l4126_412685


namespace barry_sotter_magic_l4126_412624

-- Define the length increase factor for day k
def increase_factor (k : ℕ) : ℚ := (2 * k + 2) / (2 * k + 1)

-- Define the total increase factor after n days
def total_increase (n : ℕ) : ℚ := (2 * n + 2) / 2

-- Theorem statement
theorem barry_sotter_magic (n : ℕ) : total_increase n = 50 ↔ n = 49 := by
  sorry

end barry_sotter_magic_l4126_412624


namespace tangent_perpendicular_and_inequality_l4126_412682

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((4*x + a) * Real.log x) / (3*x + 1)

theorem tangent_perpendicular_and_inequality (a : ℝ) :
  (∃ m : ℝ, ∀ x : ℝ, x ≥ 1 → f a x ≤ m * (x - 1)) →
  (a = 0 ∧ ∀ m : ℝ, (∀ x : ℝ, x ≥ 1 → f a x ≤ m * (x - 1)) → m ≥ 1) :=
sorry

end tangent_perpendicular_and_inequality_l4126_412682


namespace quadratic_function_value_at_three_l4126_412636

/-- A quadratic function f(x) = ax^2 + bx + c with roots at x=1 and x=5, and minimum value 36 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_value_at_three
  (a b c : ℝ)
  (root_one : QuadraticFunction a b c 1 = 0)
  (root_five : QuadraticFunction a b c 5 = 0)
  (min_value : ∀ x, QuadraticFunction a b c x ≥ 36)
  (attains_min : ∃ x, QuadraticFunction a b c x = 36) :
  QuadraticFunction a b c 3 = 36 := by
  sorry

end quadratic_function_value_at_three_l4126_412636


namespace spaceship_journey_theorem_l4126_412601

/-- Represents the travel schedule of a spaceship --/
structure SpaceshipJourney where
  totalJourneyTime : ℕ
  firstDayTravelTime1 : ℕ
  firstDayBreakTime1 : ℕ
  firstDayTravelTime2 : ℕ
  firstDayBreakTime2 : ℕ
  routineTravelTime : ℕ
  routineBreakTime : ℕ

/-- Calculates the total time the spaceship was not moving during its journey --/
def totalNotMovingTime (journey : SpaceshipJourney) : ℕ :=
  let firstDayBreakTime := journey.firstDayBreakTime1 + journey.firstDayBreakTime2
  let firstDayTotalTime := journey.firstDayTravelTime1 + journey.firstDayTravelTime2 + firstDayBreakTime
  let remainingTime := journey.totalJourneyTime - firstDayTotalTime
  let routineBlockTime := journey.routineTravelTime + journey.routineBreakTime
  let routineBlocks := remainingTime / routineBlockTime
  firstDayBreakTime + routineBlocks * journey.routineBreakTime

theorem spaceship_journey_theorem (journey : SpaceshipJourney) 
  (h1 : journey.totalJourneyTime = 72)
  (h2 : journey.firstDayTravelTime1 = 10)
  (h3 : journey.firstDayBreakTime1 = 3)
  (h4 : journey.firstDayTravelTime2 = 10)
  (h5 : journey.firstDayBreakTime2 = 1)
  (h6 : journey.routineTravelTime = 11)
  (h7 : journey.routineBreakTime = 1) :
  totalNotMovingTime journey = 8 := by
  sorry

end spaceship_journey_theorem_l4126_412601


namespace particle_probability_l4126_412689

/-- Probability of reaching (0,0) from position (x,y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3

/-- The probability of reaching (0,0) from (3,5) is 1385/19683 -/
theorem particle_probability : P 3 5 = 1385 / 19683 := by
  sorry

end particle_probability_l4126_412689


namespace jack_and_jill_speed_jack_and_jill_speed_proof_l4126_412691

/-- The common speed of Jack and Jill given their walking conditions -/
theorem jack_and_jill_speed : ℝ → Prop :=
  fun (x : ℝ) ↦ 
    let jack_speed := x^2 - 11*x - 22
    let jill_distance := x^2 - 3*x - 54
    let jill_time := x + 6
    let jill_speed := jill_distance / jill_time
    (jack_speed = jill_speed) → (jack_speed = 4)

/-- Proof of the theorem -/
theorem jack_and_jill_speed_proof : ∃ x : ℝ, jack_and_jill_speed x :=
  sorry

end jack_and_jill_speed_jack_and_jill_speed_proof_l4126_412691


namespace nonagon_diagonals_l4126_412659

/-- The number of distinct diagonals in a convex nonagon -/
def diagonals_in_nonagon : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals : diagonals_in_nonagon = 27 := by sorry

end nonagon_diagonals_l4126_412659


namespace base_four_20314_equals_568_l4126_412618

def base_four_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base_four_20314_equals_568 :
  base_four_to_decimal [4, 1, 3, 0, 2] = 568 := by
  sorry

end base_four_20314_equals_568_l4126_412618


namespace square_of_198_l4126_412674

theorem square_of_198 : 
  (198 : ℕ)^2 = 200^2 - 2 * 200 * 2 + 2^2 := by
  have h1 : 198 = 200 - 2 := by sorry
  have h2 : ∀ (a b : ℕ), (a - b)^2 = a^2 - 2*a*b + b^2 := by sorry
  sorry

end square_of_198_l4126_412674


namespace two_colonies_reach_limit_same_time_l4126_412614

/-- Represents the number of days it takes for a bacteria colony to reach its habitat limit -/
def habitat_limit_days : ℕ := 22

/-- Represents the daily growth factor of the bacteria colony -/
def daily_growth_factor : ℕ := 2

/-- Theorem stating that two bacteria colonies starting simultaneously will reach the habitat limit in the same number of days as a single colony -/
theorem two_colonies_reach_limit_same_time (initial_population : ℕ) :
  (initial_population * daily_growth_factor ^ habitat_limit_days) =
  (2 * initial_population * daily_growth_factor ^ habitat_limit_days) / 2 :=
by
  sorry

#check two_colonies_reach_limit_same_time

end two_colonies_reach_limit_same_time_l4126_412614


namespace evaluate_expression_l4126_412609

theorem evaluate_expression : 3000 * (3000^1500 + 3000^1500) = 2 * 3000^1501 := by
  sorry

end evaluate_expression_l4126_412609


namespace max_sum_given_constraints_l4126_412698

theorem max_sum_given_constraints (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) :
  x + y ≤ 6 * Real.sqrt 5 := by
sorry

end max_sum_given_constraints_l4126_412698


namespace butterfingers_count_l4126_412626

theorem butterfingers_count (total : ℕ) (mars : ℕ) (snickers : ℕ) (butterfingers : ℕ)
  (h1 : total = 12)
  (h2 : mars = 2)
  (h3 : snickers = 3)
  (h4 : total = mars + snickers + butterfingers) :
  butterfingers = 7 := by
  sorry

end butterfingers_count_l4126_412626


namespace quadratic_inequality_solution_set_l4126_412627

theorem quadratic_inequality_solution_set (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 8*x + c < 0) ↔ (c > 0 ∧ c < 16) :=
by sorry

end quadratic_inequality_solution_set_l4126_412627


namespace normal_distribution_std_dev_l4126_412673

/-- Given a normal distribution with mean 12 and a value 9.6 that is 2 standard deviations
    below the mean, the standard deviation is 1.2 -/
theorem normal_distribution_std_dev (μ σ : ℝ) (h1 : μ = 12) (h2 : μ - 2 * σ = 9.6) :
  σ = 1.2 := by
  sorry

end normal_distribution_std_dev_l4126_412673


namespace arithmetic_sequence_problem_l4126_412686

/-- An arithmetic sequence {a_n} where a_1 = 1/3, a_2 + a_5 = 4, and a_n = 33 has n = 50 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) (n : ℕ) 
  (h_arith : ∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) 
  (h_a1 : a 1 = 1/3)
  (h_sum : a 2 + a 5 = 4)
  (h_an : a n = 33) :
  n = 50 := by
  sorry

end arithmetic_sequence_problem_l4126_412686


namespace pentagon_smallest_angle_l4126_412664

theorem pentagon_smallest_angle (a b c d e : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  a + b + c + d + e = 540 →
  b = 4/3 * a →
  c = 5/3 * a →
  d = 2 * a →
  e = 7/3 * a →
  a = 64.8 :=
by sorry

end pentagon_smallest_angle_l4126_412664


namespace semicircular_cubicle_perimeter_l4126_412670

/-- The perimeter of a semicircular cubicle with radius 14 is approximately 72 units. -/
theorem semicircular_cubicle_perimeter : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |14 * Real.pi + 28 - 72| < ε := by
  sorry

end semicircular_cubicle_perimeter_l4126_412670


namespace complementary_implies_mutually_exclusive_l4126_412607

/-- Two events are complementary if one event occurs if and only if the other does not occur -/
def complementary_events (Ω : Type*) (A B : Set Ω) : Prop :=
  A = (Bᶜ)

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutually_exclusive (Ω : Type*) (A B : Set Ω) : Prop :=
  A ∩ B = ∅

/-- The probability of an event is a number between 0 and 1 inclusive -/
axiom probability_range (Ω : Type*) (A : Set Ω) :
  ∃ (P : Set Ω → ℝ), 0 ≤ P A ∧ P A ≤ 1

theorem complementary_implies_mutually_exclusive (Ω : Type*) (A B : Set Ω) :
  complementary_events Ω A B → mutually_exclusive Ω A B :=
sorry

end complementary_implies_mutually_exclusive_l4126_412607


namespace quadratic_root_implies_m_value_l4126_412684

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (1 : ℝ)^2 - 2*m*(1 : ℝ) + 1 = 0 → m = 1 := by
  sorry

end quadratic_root_implies_m_value_l4126_412684


namespace tobys_friends_l4126_412665

theorem tobys_friends (total_friends : ℕ) (boy_friends : ℕ) (girl_friends : ℕ) : 
  (boy_friends : ℚ) / total_friends = 55 / 100 →
  boy_friends = 33 →
  girl_friends = 27 :=
by sorry

end tobys_friends_l4126_412665


namespace inequality_proof_l4126_412619

theorem inequality_proof (n : ℕ) (h : n > 2) :
  (2*n - 1)^n + (2*n)^n < (2*n + 1)^n := by
  sorry

end inequality_proof_l4126_412619


namespace tan_graph_property_l4126_412676

theorem tan_graph_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + 2 * Real.pi / 5))) →
  a * Real.tan (b * Real.pi / 10) = 1 →
  a * b = 5 / 2 := by
  sorry

end tan_graph_property_l4126_412676


namespace greatest_divisor_with_remainders_l4126_412642

theorem greatest_divisor_with_remainders : ∃ (n : ℕ), 
  (n ∣ (150 - 50)) ∧ 
  (n ∣ (230 - 5)) ∧ 
  (n ∣ (175 - 25)) ∧ 
  (∀ m : ℕ, m > n → (m ∣ (150 - 50)) → (m ∣ (230 - 5)) → ¬(m ∣ (175 - 25))) := by
  sorry

end greatest_divisor_with_remainders_l4126_412642


namespace car_production_is_four_l4126_412640

/-- Represents the factory's production and profit data -/
structure FactoryData where
  car_material_cost : ℕ
  car_selling_price : ℕ
  motorcycle_material_cost : ℕ
  motorcycle_count : ℕ
  motorcycle_selling_price : ℕ
  profit_difference : ℕ

/-- Calculates the number of cars that could be produced per month -/
def calculate_car_production (data : FactoryData) : ℕ :=
  let motorcycle_profit := data.motorcycle_count * data.motorcycle_selling_price - data.motorcycle_material_cost
  let car_profit := fun c => c * data.car_selling_price - data.car_material_cost
  (motorcycle_profit - data.profit_difference + data.car_material_cost) / data.car_selling_price

theorem car_production_is_four (data : FactoryData) 
  (h1 : data.car_material_cost = 100)
  (h2 : data.car_selling_price = 50)
  (h3 : data.motorcycle_material_cost = 250)
  (h4 : data.motorcycle_count = 8)
  (h5 : data.motorcycle_selling_price = 50)
  (h6 : data.profit_difference = 50) :
  calculate_car_production data = 4 := by
  sorry

end car_production_is_four_l4126_412640


namespace units_digit_G_1000_l4126_412680

-- Define G_n
def G (n : ℕ) : ℕ := 3 * 2^(2^n) + 4

-- Theorem statement
theorem units_digit_G_1000 : G 1000 % 10 = 6 := by
  sorry

end units_digit_G_1000_l4126_412680


namespace function_equality_l4126_412679

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x => 2 * (x - 1)^2 + 1

-- State the theorem
theorem function_equality (x : ℝ) (h : x ≥ 1) : 
  f (1 + Real.sqrt x) = 2 * x + 1 ∧ f x = 2 * x^2 - 4 * x + 3 := by
  sorry

end function_equality_l4126_412679


namespace geometric_progression_sufficient_not_necessary_l4126_412652

/-- A sequence of three real numbers forms a geometric progression --/
def is_geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

theorem geometric_progression_sufficient_not_necessary :
  (∀ a b c : ℝ, is_geometric_progression a b c → b^2 = a*c) ∧
  (∃ a b c : ℝ, b^2 = a*c ∧ ¬is_geometric_progression a b c) := by
  sorry


end geometric_progression_sufficient_not_necessary_l4126_412652


namespace kids_difference_l4126_412641

theorem kids_difference (camp_kids home_kids : ℕ) 
  (h1 : camp_kids = 202958) 
  (h2 : home_kids = 777622) : 
  home_kids - camp_kids = 574664 := by
sorry

end kids_difference_l4126_412641


namespace exists_line_through_P_intersecting_hyperbola_l4126_412671

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define a line passing through P with slope k
def line (k : ℝ) (x y : ℝ) : Prop := y - P.2 = k * (x - P.1)

-- Define the midpoint condition
def is_midpoint (p a b : ℝ × ℝ) : Prop :=
  p.1 = (a.1 + b.1) / 2 ∧ p.2 = (a.2 + b.2) / 2

-- Theorem statement
theorem exists_line_through_P_intersecting_hyperbola :
  ∃ (k : ℝ) (A B : ℝ × ℝ),
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    line k A.1 A.2 ∧
    line k B.1 B.2 ∧
    is_midpoint P A B :=
  sorry

end exists_line_through_P_intersecting_hyperbola_l4126_412671


namespace square_roots_equality_l4126_412678

theorem square_roots_equality (x a : ℝ) (hx : x > 0) :
  (3 * a - 14) ^ 2 = x ∧ (a - 2) ^ 2 = x → a = 4 ∧ x = 4 :=
by sorry

end square_roots_equality_l4126_412678


namespace proposition_truth_l4126_412649

theorem proposition_truth (x y : ℝ) : x + y ≠ 3 → x ≠ 2 ∨ y ≠ 1 := by
  sorry

end proposition_truth_l4126_412649


namespace haleys_extra_tickets_l4126_412630

theorem haleys_extra_tickets (ticket_price : ℕ) (initial_tickets : ℕ) (total_spent : ℕ) : 
  ticket_price = 4 →
  initial_tickets = 3 →
  total_spent = 32 →
  total_spent / ticket_price - initial_tickets = 5 :=
by
  sorry

end haleys_extra_tickets_l4126_412630


namespace roses_remaining_is_nine_l4126_412623

/-- Represents the number of roses in a dozen -/
def dozen : ℕ := 12

/-- Calculates the number of unwilted roses remaining after a series of events -/
def remaining_roses (initial_dozens : ℕ) (traded_dozens : ℕ) : ℕ :=
  let initial_roses := initial_dozens * dozen
  let after_trade := initial_roses + traded_dozens * dozen
  let after_first_wilt := after_trade / 2
  after_first_wilt / 2

/-- Proves that given the initial conditions and subsequent events, 
    the number of unwilted roses remaining is 9 -/
theorem roses_remaining_is_nine :
  remaining_roses 2 1 = 9 := by
  sorry

#eval remaining_roses 2 1

end roses_remaining_is_nine_l4126_412623


namespace joint_purchase_popularity_l4126_412651

structure JointPurchase where
  scale : ℝ
  cost_savings : ℝ
  quality_assessment : ℝ
  community_trust : ℝ
  transaction_costs : ℝ
  organizational_efforts : ℝ
  convenience : ℝ
  dispute_potential : ℝ

def benefits (jp : JointPurchase) : ℝ :=
  jp.cost_savings + jp.quality_assessment + jp.community_trust

def drawbacks (jp : JointPurchase) : ℝ :=
  jp.transaction_costs + jp.organizational_efforts + jp.convenience + jp.dispute_potential

theorem joint_purchase_popularity (jp : JointPurchase) :
  jp.scale > 1 → benefits jp > drawbacks jp ∧
  jp.scale ≤ 1 → benefits jp ≤ drawbacks jp :=
sorry

end joint_purchase_popularity_l4126_412651


namespace average_speed_calculation_l4126_412681

theorem average_speed_calculation (speed1 speed2 : ℝ) (time : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 30)
  (h3 : time = 2) :
  (speed1 + speed2) / time = 25 :=
by sorry

end average_speed_calculation_l4126_412681


namespace linear_relationship_l4126_412647

/-- Given a linear relationship where an increase of 4 units in x corresponds to an increase of 6 units in y,
    prove that an increase of 12 units in x will result in an increase of 18 units in y. -/
theorem linear_relationship (f : ℝ → ℝ) (x₀ : ℝ) :
  (f (x₀ + 4) - f x₀ = 6) → (f (x₀ + 12) - f x₀ = 18) := by
  sorry

end linear_relationship_l4126_412647


namespace quadratic_root_implies_a_bound_l4126_412692

theorem quadratic_root_implies_a_bound (a : ℝ) (h1 : a > 0) 
  (h2 : 3^2 - 5/3 * a * 3 - a^2 = 0) : 1 < a ∧ a < 3/2 := by
  sorry

end quadratic_root_implies_a_bound_l4126_412692


namespace september_percentage_l4126_412656

-- Define the total number of people surveyed
def total_people : ℕ := 150

-- Define the number of people born in September
def september_births : ℕ := 12

-- Define the percentage calculation function
def percentage (part : ℕ) (whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

-- State the theorem
theorem september_percentage : percentage september_births total_people = 8 := by
  sorry

end september_percentage_l4126_412656


namespace largest_n_for_equation_l4126_412606

theorem largest_n_for_equation : ∃ (x y z : ℕ+), 
  10^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10 ∧
  ∀ (n : ℕ+), n > 10 → ¬∃ (a b c : ℕ+), 
    n^2 = a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a + 5*a + 5*b + 5*c - 10 :=
by sorry

end largest_n_for_equation_l4126_412606


namespace blue_balls_unchanged_l4126_412621

/-- The number of blue balls remains unchanged when red balls are removed from a box -/
theorem blue_balls_unchanged (initial_blue : ℕ) (initial_red : ℕ) (removed_red : ℕ) :
  initial_blue = initial_blue :=
by sorry

end blue_balls_unchanged_l4126_412621


namespace inequality_proof_l4126_412602

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ 2 / (1 - x*y) := by
  sorry

end inequality_proof_l4126_412602


namespace two_eyes_for_dog_l4126_412632

/-- Given a family that catches and distributes fish, calculate the number of fish eyes left for the dog. -/
def fish_eyes_for_dog (family_size : ℕ) (fish_per_person : ℕ) (eyes_per_fish : ℕ) (eyes_eaten : ℕ) : ℕ :=
  let total_fish := family_size * fish_per_person
  let total_eyes := total_fish * eyes_per_fish
  total_eyes - eyes_eaten

/-- Theorem stating that under the given conditions, 2 fish eyes remain for the dog. -/
theorem two_eyes_for_dog :
  fish_eyes_for_dog 3 4 2 22 = 2 :=
by sorry

end two_eyes_for_dog_l4126_412632


namespace can_display_rows_l4126_412635

/-- Represents a display of cans arranged in rows. -/
structure CanDisplay where
  firstRowCans : ℕ  -- Number of cans in the first row
  rowIncrement : ℕ  -- Increment in number of cans for each subsequent row
  totalCans : ℕ     -- Total number of cans in the display

/-- Calculates the number of rows in a can display. -/
def numberOfRows (display : CanDisplay) : ℕ :=
  sorry

/-- Theorem stating that a display with 2 cans in the first row,
    incrementing by 3 cans each row, and totaling 120 cans has 9 rows. -/
theorem can_display_rows :
  let display : CanDisplay := {
    firstRowCans := 2,
    rowIncrement := 3,
    totalCans := 120
  }
  numberOfRows display = 9 := by sorry

end can_display_rows_l4126_412635


namespace angle_bisector_length_l4126_412662

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle bisector of B
def angleBisector (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the measure of an angle
def angleMeasure (p q r : ℝ × ℝ) : ℝ := sorry

theorem angle_bisector_length 
  (t : Triangle) 
  (h1 : angleMeasure t.A t.B t.C = 20)
  (h2 : angleMeasure t.C t.A t.B = 40)
  (h3 : length t.A t.C - length t.A t.B = 5) :
  length t.B (angleBisector t) = 5 := by sorry

end angle_bisector_length_l4126_412662


namespace train_length_l4126_412646

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 50 → -- speed in km/h
  time = 5.399568034557236 → -- time in seconds
  length = speed * 1000 / 3600 * time → -- length in meters
  length = 75 := by sorry

end train_length_l4126_412646


namespace concert_attendance_l4126_412611

theorem concert_attendance (total_tickets : ℕ) 
  (h1 : total_tickets = 2465)
  (before_start : ℕ) 
  (h2 : before_start = (7 * total_tickets) / 8)
  (after_first_song : ℕ) 
  (h3 : after_first_song = (13 * (total_tickets - before_start)) / 17)
  (last_performances : ℕ) 
  (h4 : last_performances = 47) : 
  total_tickets - before_start - after_first_song - last_performances = 26 := by
sorry

end concert_attendance_l4126_412611


namespace jason_average_messages_l4126_412644

/-- The average number of text messages sent over five days -/
def average_messages (monday : ℕ) (tuesday : ℕ) (wed_to_fri : ℕ) (days : ℕ) : ℚ :=
  (monday + tuesday + 3 * wed_to_fri : ℚ) / days

theorem jason_average_messages :
  let monday := 220
  let tuesday := monday / 2
  let wed_to_fri := 50
  let days := 5
  average_messages monday tuesday wed_to_fri days = 96 := by
sorry

end jason_average_messages_l4126_412644


namespace equation_solution_l4126_412645

theorem equation_solution (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ -1) :
  (x / (x - 1) = 4 / (x^2 - 1) + 1) ↔ (x = 3) := by
sorry

end equation_solution_l4126_412645


namespace simplify_expression_l4126_412690

theorem simplify_expression (x y : ℝ) : (3 * x^2 * y^3)^4 = 81 * x^8 * y^12 := by
  sorry

end simplify_expression_l4126_412690


namespace range_equivalence_l4126_412648

/-- The range of real numbers a for which at least one of the given equations has real roots -/
def range_with_real_roots (a : ℝ) : Prop :=
  ∃ x : ℝ, (x^2 + 4*a*x - 4*a + 3 = 0) ∨ 
            (x^2 + (a-1)*x + a^2 = 0) ∨ 
            (x^2 + 2*a*x - 2*a = 0)

/-- The range of real numbers a for which none of the given equations have real roots -/
def range_without_real_roots (a : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 + 4*a*x - 4*a + 3 ≠ 0) ∧ 
            (x^2 + (a-1)*x + a^2 ≠ 0) ∧ 
            (x^2 + 2*a*x - 2*a ≠ 0)

/-- The theorem stating that the range with real roots is the complement of the range without real roots -/
theorem range_equivalence : 
  ∀ a : ℝ, range_with_real_roots a ↔ ¬(range_without_real_roots a) :=
sorry

end range_equivalence_l4126_412648


namespace average_weight_decrease_l4126_412658

/-- Proves that replacing a 72 kg student with a 12 kg student in a group of 5 decreases the average weight by 12 kg -/
theorem average_weight_decrease (initial_average : ℝ) : 
  let total_weight := 5 * initial_average
  let new_total_weight := total_weight - 72 + 12
  let new_average := new_total_weight / 5
  initial_average - new_average = 12 := by
sorry

end average_weight_decrease_l4126_412658


namespace isosceles_triangle_vertex_angle_l4126_412637

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We don't need to specify all properties of an isosceles triangle,
  -- just that it has a vertex angle
  vertexAngle : ℝ

-- Define our theorem
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (has_40_degree_angle : ∃ (angle : ℝ), angle = 40 ∧ 
    (angle = triangle.vertexAngle ∨ 
     2 * angle + triangle.vertexAngle = 180)) :
  triangle.vertexAngle = 40 ∨ triangle.vertexAngle = 100 := by
sorry


end isosceles_triangle_vertex_angle_l4126_412637


namespace time_to_weave_cloth_l4126_412654

/-- Represents the industrial loom's weaving rate and characteristics -/
structure Loom where
  rate : Real
  sample_time : Real
  sample_cloth : Real

/-- Theorem: Time to weave cloth -/
theorem time_to_weave_cloth (loom : Loom) (x : Real) :
  loom.rate = 0.128 ∧ 
  loom.sample_time = 195.3125 ∧ 
  loom.sample_cloth = 25 →
  x / loom.rate = x / 0.128 := by
  sorry

#check time_to_weave_cloth

end time_to_weave_cloth_l4126_412654


namespace abs_2y_minus_7_zero_l4126_412612

theorem abs_2y_minus_7_zero (y : ℚ) : |2 * y - 7| = 0 ↔ y = 7/2 := by
  sorry

end abs_2y_minus_7_zero_l4126_412612


namespace min_value_theorem_l4126_412620

theorem min_value_theorem (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) 
  (h_inequality : b + c ≥ a + d) : 
  (b / (c + d)) + (c / (a + b)) ≥ Real.sqrt 2 - 1/2 := by
  sorry

end min_value_theorem_l4126_412620


namespace union_equality_implies_t_value_l4126_412694

def M (t : ℝ) : Set ℝ := {1, 3, t}
def N (t : ℝ) : Set ℝ := {t^2 - t + 1}

theorem union_equality_implies_t_value (t : ℝ) :
  M t ∪ N t = M t → t = 0 ∨ t = 2 ∨ t = -1 := by
  sorry

end union_equality_implies_t_value_l4126_412694


namespace sum_of_numbers_greater_than_threshold_l4126_412693

def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]
def threshold : ℚ := 11/10

theorem sum_of_numbers_greater_than_threshold :
  (numbers.filter (λ x => x > threshold)).sum = 39/10 := by
  sorry

end sum_of_numbers_greater_than_threshold_l4126_412693


namespace euler_formula_second_quadrant_l4126_412688

theorem euler_formula_second_quadrant :
  let z : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 3))
  z.re < 0 ∧ z.im > 0 :=
by sorry

end euler_formula_second_quadrant_l4126_412688


namespace angle_Y_measure_l4126_412695

-- Define the structure for lines and angles
structure Geometry where
  Line : Type
  Angle : Type
  measure : Angle → ℝ
  parallel : Line → Line → Prop
  intersect : Line → Line → Prop
  angleOn : Line → Angle → Prop
  transversal : Line → Line → Line → Prop

-- State the theorem
theorem angle_Y_measure (G : Geometry) 
  (p q t yz : G.Line) (X Z Y : G.Angle) :
  G.parallel p q →
  G.parallel p yz →
  G.parallel q yz →
  G.transversal t p q →
  G.intersect t yz →
  G.angleOn p X →
  G.angleOn q Z →
  G.measure X = 100 →
  G.measure Z = 110 →
  G.measure Y = 40 := by
  sorry


end angle_Y_measure_l4126_412695


namespace inequality_solution_l4126_412603

theorem inequality_solution (x : ℝ) : 2 * (3 * x - 2) > x + 1 ↔ x > 1 := by
  sorry

end inequality_solution_l4126_412603


namespace siblings_total_age_l4126_412643

/-- Represents the ages of six siblings -/
structure SiblingAges where
  susan : ℕ
  arthur : ℕ
  tom : ℕ
  bob : ℕ
  emily : ℕ
  david : ℕ

/-- Calculates the total age of all siblings -/
def totalAge (ages : SiblingAges) : ℕ :=
  ages.susan + ages.arthur + ages.tom + ages.bob + ages.emily + ages.david

/-- Theorem stating the total age of the siblings -/
theorem siblings_total_age :
  ∀ (ages : SiblingAges),
    ages.susan = 15 →
    ages.bob = 11 →
    ages.arthur = ages.susan + 2 →
    ages.tom = ages.bob - 3 →
    ages.emily = ages.susan / 2 →
    ages.david = (ages.arthur + ages.tom + ages.emily) / 3 →
    totalAge ages = 70 := by
  sorry


end siblings_total_age_l4126_412643


namespace compound_oxygen_count_l4126_412672

/-- Represents a chemical compound with a given number of Carbon, Hydrogen, and Oxygen atoms -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight hydrogenWeight oxygenWeight : ℝ) : ℝ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

/-- Theorem: A compound with 4 Carbon atoms, 1 Hydrogen atom, and a molecular weight of 65 g/mol contains 1 Oxygen atom -/
theorem compound_oxygen_count :
  ∃ (c : Compound),
    c.carbon = 4 ∧
    c.hydrogen = 1 ∧
    c.oxygen = 1 ∧
    molecularWeight c 12.01 1.008 16.00 = 65 := by
  sorry


end compound_oxygen_count_l4126_412672


namespace friend_name_probability_l4126_412699

def total_cards : ℕ := 15
def cybil_cards : ℕ := 6
def ronda_cards : ℕ := 9
def cards_drawn : ℕ := 3

theorem friend_name_probability : 
  (1 : ℚ) - (Nat.choose ronda_cards cards_drawn : ℚ) / (Nat.choose total_cards cards_drawn)
         - (Nat.choose cybil_cards cards_drawn : ℚ) / (Nat.choose total_cards cards_drawn)
  = 351 / 455 := by sorry

end friend_name_probability_l4126_412699


namespace hen_count_l4126_412639

theorem hen_count (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) 
  (h1 : total_animals = 48) 
  (h2 : total_feet = 136) 
  (h3 : hen_feet = 2) 
  (h4 : cow_feet = 4) :
  ∃ (hens cows : ℕ), 
    hens + cows = total_animals ∧ 
    hen_feet * hens + cow_feet * cows = total_feet ∧ 
    hens = 28 := by
  sorry

end hen_count_l4126_412639


namespace quadratic_integer_roots_count_l4126_412653

theorem quadratic_integer_roots_count : 
  let f (m : ℤ) := (∃ x₁ x₂ : ℤ, x₁^2 - m*x₁ + 36 = 0 ∧ x₂^2 - m*x₂ + 36 = 0 ∧ x₁ ≠ x₂)
  (∃! (s : Finset ℤ), (∀ m ∈ s, f m) ∧ s.card = 10) :=
by sorry

end quadratic_integer_roots_count_l4126_412653


namespace calculation_proof_l4126_412615

theorem calculation_proof : |-4| + (1/3)⁻¹ - (Real.sqrt 2)^2 + 2035^0 = 6 := by
  sorry

end calculation_proof_l4126_412615


namespace n2o3_molecular_weight_l4126_412633

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of N2O3 in g/mol -/
def n2o3_weight : ℝ := 2 * nitrogen_weight + 3 * oxygen_weight

/-- Theorem stating that the molecular weight of N2O3 is 76.02 g/mol -/
theorem n2o3_molecular_weight : n2o3_weight = 76.02 := by
  sorry

end n2o3_molecular_weight_l4126_412633


namespace museum_trip_ratio_l4126_412605

theorem museum_trip_ratio : 
  ∀ (p1 p2 p3 p4 : ℕ),
  p1 = 12 →
  p3 = p2 - 6 →
  p4 = p1 + 9 →
  p1 + p2 + p3 + p4 = 75 →
  p2 / p1 = 2 := by
sorry

end museum_trip_ratio_l4126_412605


namespace planes_parallel_from_skew_lines_parallel_l4126_412628

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (skew_lines : Line → Line → Prop)

-- Define the planes and lines
variable (α β : Plane)
variable (a b : Line)

-- State the theorem
theorem planes_parallel_from_skew_lines_parallel 
  (h_distinct : α ≠ β)
  (h_different : a ≠ b)
  (h_skew : skew_lines a b)
  (h_a_alpha : parallel_line_plane a α)
  (h_b_alpha : parallel_line_plane b α)
  (h_a_beta : parallel_line_plane a β)
  (h_b_beta : parallel_line_plane b β) :
  parallel_planes α β :=
sorry

end planes_parallel_from_skew_lines_parallel_l4126_412628


namespace other_root_of_quadratic_l4126_412613

theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, 7 * x^2 + m * x - 6 = 0 ∧ x = -3) →
  (7 * (2/7)^2 + m * (2/7) - 6 = 0) :=
by sorry

end other_root_of_quadratic_l4126_412613


namespace expression_simplification_l4126_412668

theorem expression_simplification (a : ℝ) (h : a^2 + 2*a - 8 = 0) :
  ((a^2 - 4) / (a^2 - 4*a + 4) - a / (a - 2)) / ((a^2 + 2*a) / (a - 2)) = 1/4 := by
  sorry

end expression_simplification_l4126_412668


namespace sin_2alpha_over_cos_squared_l4126_412604

theorem sin_2alpha_over_cos_squared (α : Real) 
  (h : Real.sin α = 3 * Real.cos α) : 
  Real.sin (2 * α) / (Real.cos α)^2 = 6 := by
  sorry

end sin_2alpha_over_cos_squared_l4126_412604


namespace tennis_players_count_l4126_412697

theorem tennis_players_count (total : ℕ) (badminton : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 35)
  (h2 : badminton = 15)
  (h3 : neither = 5)
  (h4 : both = 3) :
  ∃ tennis : ℕ, tennis = 18 ∧ 
  tennis = total - neither - (badminton - both) := by
  sorry

end tennis_players_count_l4126_412697


namespace nadines_pebbles_l4126_412608

/-- The number of pebbles Nadine has -/
def total_pebbles (white red blue green : ℕ) : ℕ := white + red + blue + green

/-- Theorem stating the total number of pebbles Nadine has -/
theorem nadines_pebbles :
  ∀ (white red blue green : ℕ),
  white = 20 →
  red = white / 2 →
  blue = red / 3 →
  green = blue + 5 →
  total_pebbles white red blue green = 41 := by
sorry

#eval total_pebbles 20 10 3 8

end nadines_pebbles_l4126_412608


namespace four_solutions_l4126_412660

def is_solution (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ (4 : ℚ) / m + (2 : ℚ) / n = 1

def solution_count : ℕ := 4

theorem four_solutions :
  ∃ (S : Finset (ℕ × ℕ)), S.card = solution_count ∧
    (∀ (p : ℕ × ℕ), p ∈ S ↔ is_solution p.1 p.2) :=
sorry

end four_solutions_l4126_412660


namespace robin_extra_gum_l4126_412631

/-- Represents the number of extra pieces of gum Robin has -/
def extra_gum (total_pieces packages pieces_per_package : ℕ) : ℕ :=
  total_pieces - packages * pieces_per_package

/-- Proves that Robin has 6 extra pieces of gum given the conditions -/
theorem robin_extra_gum :
  extra_gum 41 5 7 = 6 := by
  sorry

end robin_extra_gum_l4126_412631


namespace max_value_inequality_l4126_412638

theorem max_value_inequality (x y : ℝ) :
  (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 35 := by
  sorry

end max_value_inequality_l4126_412638


namespace chris_age_l4126_412600

/-- The ages of Amy, Ben, and Chris -/
structure Ages where
  amy : ℝ
  ben : ℝ
  chris : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 10
  (ages.amy + ages.ben + ages.chris) / 3 = 10 ∧
  -- Five years ago, Chris was twice Amy's age
  ages.chris - 5 = 2 * (ages.amy - 5) ∧
  -- In 5 years, Ben's age will be half of Amy's age
  ages.ben + 5 = (ages.amy + 5) / 2

/-- The theorem to prove -/
theorem chris_age (ages : Ages) (h : satisfies_conditions ages) : 
  ∃ (ε : ℝ), ages.chris = 16 + ε ∧ abs ε < 1 := by
  sorry

end chris_age_l4126_412600


namespace solution_set_f_geq_1_max_value_f_minus_x_squared_plus_x_l4126_412687

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem 1: The solution set of f(x) ≥ 1 is {x | x ≥ 1}
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

-- Theorem 2: The maximum value of f(x) - x^2 + x is 5/4
theorem max_value_f_minus_x_squared_plus_x :
  ∃ (x : ℝ), ∀ (y : ℝ), f y - y^2 + y ≤ f x - x^2 + x ∧ f x - x^2 + x = 5/4 := by sorry

end solution_set_f_geq_1_max_value_f_minus_x_squared_plus_x_l4126_412687


namespace second_chapter_pages_count_l4126_412677

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  first_chapter_pages : ℕ
  second_chapter_pages : ℕ
  two_chapters : first_chapter_pages + second_chapter_pages = total_pages

/-- The specific book described in the problem -/
def problem_book : Book where
  total_pages := 81
  first_chapter_pages := 13
  second_chapter_pages := 68
  two_chapters := by sorry

theorem second_chapter_pages_count :
  problem_book.second_chapter_pages = 68 := by sorry

end second_chapter_pages_count_l4126_412677


namespace license_plate_count_l4126_412696

/-- The number of possible letters in each position of the license plate -/
def num_letters : ℕ := 26

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible even digits -/
def num_even_digits : ℕ := 5

/-- The number of possible odd digits -/
def num_odd_digits : ℕ := 5

/-- The total number of license plates with 3 letters followed by 2 digits,
    where one digit is odd and the other is even -/
def total_license_plates : ℕ := num_letters^3 * num_digits * num_even_digits

theorem license_plate_count :
  total_license_plates = 878800 := by
  sorry

end license_plate_count_l4126_412696


namespace optimal_tank_design_l4126_412683

/-- Represents the dimensions and cost of a rectangular open-top water storage tank. -/
structure Tank where
  length : ℝ
  width : ℝ
  depth : ℝ
  base_cost : ℝ
  wall_cost : ℝ

/-- Calculates the volume of the tank. -/
def volume (t : Tank) : ℝ := t.length * t.width * t.depth

/-- Calculates the total construction cost of the tank. -/
def construction_cost (t : Tank) : ℝ :=
  t.base_cost * t.length * t.width + t.wall_cost * 2 * (t.length + t.width) * t.depth

/-- Theorem stating the optimal dimensions and minimum cost for the tank. -/
theorem optimal_tank_design :
  ∃ (t : Tank),
    t.depth = 3 ∧
    volume t = 4800 ∧
    t.base_cost = 150 ∧
    t.wall_cost = 120 ∧
    t.length = t.width ∧
    t.length = 40 ∧
    construction_cost t = 297600 ∧
    ∀ (t' : Tank),
      t'.depth = 3 →
      volume t' = 4800 →
      t'.base_cost = 150 →
      t'.wall_cost = 120 →
      construction_cost t' ≥ construction_cost t :=
by sorry

end optimal_tank_design_l4126_412683


namespace percent_democrat_voters_l4126_412629

theorem percent_democrat_voters (D R : ℝ) : 
  D + R = 100 →
  0.75 * D + 0.30 * R = 57 →
  D = 60 := by
sorry

end percent_democrat_voters_l4126_412629


namespace sum_of_last_two_digits_of_series_l4126_412610

def modified_fibonacci_factorial_series : List Nat :=
  [1, 2, 3, 4, 7, 11, 18, 29, 47, 76]

def last_two_digits (n : Nat) : Nat :=
  n % 100

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_of_last_two_digits_of_series :
  (modified_fibonacci_factorial_series.map (λ n => last_two_digits (factorial n))).sum = 73 := by
  sorry

end sum_of_last_two_digits_of_series_l4126_412610


namespace binary_octal_conversion_l4126_412616

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- The binary representation of the number in question -/
def binary_number : List Bool :=
  [true, false, true, true, false, true, true, true, false]

theorem binary_octal_conversion :
  (binary_to_decimal binary_number = 54) ∧
  (decimal_to_octal 54 = [6, 6]) :=
by sorry

end binary_octal_conversion_l4126_412616


namespace g_difference_l4126_412625

-- Define the sum of divisors function
def sigma (n : ℕ) : ℕ := sorry

-- Define the g function
def g (n : ℕ) : ℚ :=
  (sigma n : ℚ) / n

-- Theorem statement
theorem g_difference : g 432 - g 216 = 5 / 54 := by sorry

end g_difference_l4126_412625


namespace tank_capacity_proof_l4126_412634

/-- Represents the filling rate of pipe A in liters per minute -/
def pipe_A_rate : ℕ := 40

/-- Represents the filling rate of pipe B in liters per minute -/
def pipe_B_rate : ℕ := 30

/-- Represents the draining rate of pipe C in liters per minute -/
def pipe_C_rate : ℕ := 20

/-- Represents the time in minutes it takes to fill the tank -/
def fill_time : ℕ := 48

/-- Represents the capacity of the tank in liters -/
def tank_capacity : ℕ := 780

/-- Theorem stating that given the pipe rates and fill time, the tank capacity is 780 liters -/
theorem tank_capacity_proof :
  pipe_A_rate = 40 →
  pipe_B_rate = 30 →
  pipe_C_rate = 20 →
  fill_time = 48 →
  tank_capacity = 780 :=
by sorry

end tank_capacity_proof_l4126_412634


namespace quadratic_equations_solutions_l4126_412661

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 6*x + 4 = 0) ∧
  (∃ x : ℝ, (3*x - 1)^2 - 4*x^2 = 0) ∧
  (∀ x : ℝ, x^2 - 6*x + 4 = 0 ↔ x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5) ∧
  (∀ x : ℝ, (3*x - 1)^2 - 4*x^2 = 0 ↔ x = 1/5 ∨ x = 1) := by
  sorry

end quadratic_equations_solutions_l4126_412661


namespace zack_traveled_18_countries_l4126_412650

-- Define the number of countries each person traveled to
def george_countries : ℕ := 6
def joseph_countries : ℕ := george_countries / 2
def patrick_countries : ℕ := joseph_countries * 3
def zack_countries : ℕ := patrick_countries * 2

-- Theorem to prove
theorem zack_traveled_18_countries : zack_countries = 18 := by
  sorry

end zack_traveled_18_countries_l4126_412650


namespace scientists_born_in_july_percentage_l4126_412667

theorem scientists_born_in_july_percentage :
  let total_scientists : ℕ := 120
  let born_in_july : ℕ := 20
  let percentage : ℚ := (born_in_july : ℚ) / total_scientists * 100
  percentage = 50 / 3 := by sorry

end scientists_born_in_july_percentage_l4126_412667


namespace distance_climbed_l4126_412655

/-- The number of staircases John climbs -/
def num_staircases : ℕ := 3

/-- The number of steps in the first staircase -/
def first_staircase_steps : ℕ := 24

/-- The number of steps in the second staircase -/
def second_staircase_steps : ℕ := 3 * first_staircase_steps

/-- The number of steps in the third staircase -/
def third_staircase_steps : ℕ := second_staircase_steps - 20

/-- The height of each step in feet -/
def step_height : ℚ := 6/10

/-- The total number of steps climbed -/
def total_steps : ℕ := first_staircase_steps + second_staircase_steps + third_staircase_steps

/-- The total distance climbed in feet -/
def total_distance : ℚ := (total_steps : ℚ) * step_height

theorem distance_climbed : total_distance = 888/10 := by
  sorry

end distance_climbed_l4126_412655


namespace smallest_number_with_digit_sum_47_l4126_412675

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_number (n : ℕ) : Prop :=
  sum_of_digits n = 47

theorem smallest_number_with_digit_sum_47 :
  ∀ n : ℕ, is_valid_number n → n ≥ 299999 :=
sorry

end smallest_number_with_digit_sum_47_l4126_412675


namespace refrigerator_profit_theorem_l4126_412666

/-- Represents the financial details of a refrigerator sale --/
structure RefrigeratorSale where
  costPrice : ℝ
  markedPrice : ℝ
  discountPercentage : ℝ

/-- Calculates the profit from a refrigerator sale --/
def calculateProfit (sale : RefrigeratorSale) : ℝ :=
  sale.markedPrice * (1 - sale.discountPercentage) - sale.costPrice

/-- Theorem stating the profit for a specific refrigerator sale scenario --/
theorem refrigerator_profit_theorem (sale : RefrigeratorSale) 
  (h1 : sale.costPrice = 2000)
  (h2 : sale.markedPrice = 2750)
  (h3 : sale.discountPercentage = 0.15) :
  calculateProfit sale = 337.5 := by
  sorry

#eval calculateProfit { costPrice := 2000, markedPrice := 2750, discountPercentage := 0.15 }

end refrigerator_profit_theorem_l4126_412666


namespace geometric_sequence_sum_l4126_412663

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 2 = 4/9 →
  a 3 + a 4 + a 5 + a 6 = 40 →
  (a 7 + a 8 + a 9) / 9 = 117 :=
by
  sorry

end geometric_sequence_sum_l4126_412663


namespace ratio_of_segments_l4126_412657

/-- Given points A, B, C, and D on a line in that order, with AB : AC = 1 : 5 and BC : CD = 2 : 1, prove AB : CD = 1 : 2 -/
theorem ratio_of_segments (A B C D : ℝ) (h_order : A < B ∧ B < C ∧ C < D) 
  (h_ratio1 : (B - A) / (C - A) = 1 / 5)
  (h_ratio2 : (C - B) / (D - C) = 2 / 1) :
  (B - A) / (D - C) = 1 / 2 := by
sorry

end ratio_of_segments_l4126_412657


namespace carla_karen_age_difference_l4126_412669

-- Define the current ages
def karen_age : ℕ := 2
def frank_future_age : ℕ := 36
def years_until_frank_future : ℕ := 5

-- Define relationships between ages
def frank_age : ℕ := frank_future_age - years_until_frank_future
def ty_age : ℕ := frank_future_age / 3
def carla_age : ℕ := (ty_age - 4) / 2

-- Theorem to prove
theorem carla_karen_age_difference : carla_age - karen_age = 2 := by
  sorry

end carla_karen_age_difference_l4126_412669


namespace expression_evaluation_l4126_412622

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end expression_evaluation_l4126_412622
