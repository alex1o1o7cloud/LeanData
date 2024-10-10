import Mathlib

namespace rice_and_flour_weights_l1424_142498

/-- The weight of a bag of rice in kilograms -/
def rice_weight : ℝ := 50

/-- The weight of a bag of flour in kilograms -/
def flour_weight : ℝ := 25

/-- The total weight of 8 bags of rice and 6 bags of flour in kilograms -/
def weight1 : ℝ := 550

/-- The total weight of 4 bags of rice and 7 bags of flour in kilograms -/
def weight2 : ℝ := 375

theorem rice_and_flour_weights :
  (8 * rice_weight + 6 * flour_weight = weight1) ∧
  (4 * rice_weight + 7 * flour_weight = weight2) := by
  sorry

end rice_and_flour_weights_l1424_142498


namespace pet_store_cages_l1424_142401

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 45)
  (h2 : sold_puppies = 39)
  (h3 : puppies_per_cage = 2) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := by
  sorry

end pet_store_cages_l1424_142401


namespace expression_evaluation_l1424_142412

theorem expression_evaluation : (35 * 100) / (0.07 * 100) = 500 := by
  sorry

end expression_evaluation_l1424_142412


namespace edward_candy_cost_l1424_142483

def edward_problem (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candies : ℕ) : Prop :=
  let total_tickets := whack_a_mole_tickets + skee_ball_tickets
  let cost_per_candy := total_tickets / candies
  whack_a_mole_tickets = 3 ∧ 
  skee_ball_tickets = 5 ∧ 
  candies = 2 → 
  cost_per_candy = 4

theorem edward_candy_cost : edward_problem 3 5 2 := by
  sorry

end edward_candy_cost_l1424_142483


namespace draw_with_min_black_balls_l1424_142415

def white_balls : ℕ := 6
def black_balls : ℕ := 4
def total_draw : ℕ := 4
def min_black : ℕ := 2

theorem draw_with_min_black_balls (white_balls black_balls total_draw min_black : ℕ) :
  (white_balls = 6) → (black_balls = 4) → (total_draw = 4) → (min_black = 2) →
  (Finset.sum (Finset.range (black_balls - min_black + 1))
    (λ i => Nat.choose black_balls (min_black + i) * Nat.choose white_balls (total_draw - (min_black + i)))) = 115 := by
  sorry

end draw_with_min_black_balls_l1424_142415


namespace list_property_l1424_142456

theorem list_property (S : ℝ) (n : ℝ) (list_size : ℕ) (h1 : list_size = 21) 
  (h2 : n = 4 * ((S - n) / (list_size - 1))) 
  (h3 : n = S / 6) : 
  list_size - 1 = 20 := by
sorry

end list_property_l1424_142456


namespace average_of_rst_l1424_142404

theorem average_of_rst (r s t : ℝ) (h : (5 / 4) * (r + s + t) = 20) :
  (r + s + t) / 3 = 16 / 3 := by
  sorry

end average_of_rst_l1424_142404


namespace jack_remaining_money_l1424_142490

def calculate_remaining_money (initial_amount : ℝ) 
  (sparkling_water_bottles : ℕ) (sparkling_water_cost : ℝ)
  (still_water_multiplier : ℕ) (still_water_cost : ℝ)
  (cheddar_cheese_pounds : ℝ) (cheddar_cheese_cost : ℝ)
  (swiss_cheese_pounds : ℝ) (swiss_cheese_cost : ℝ) : ℝ :=
  let sparkling_water_total := sparkling_water_bottles * sparkling_water_cost
  let still_water_total := (sparkling_water_bottles * still_water_multiplier) * still_water_cost
  let cheddar_cheese_total := cheddar_cheese_pounds * cheddar_cheese_cost
  let swiss_cheese_total := swiss_cheese_pounds * swiss_cheese_cost
  let total_cost := sparkling_water_total + still_water_total + cheddar_cheese_total + swiss_cheese_total
  initial_amount - total_cost

theorem jack_remaining_money :
  calculate_remaining_money 150 4 3 3 2.5 2 8.5 1.5 11 = 74.5 := by
  sorry

end jack_remaining_money_l1424_142490


namespace amanda_earnings_l1424_142400

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℝ := 20

/-- Number of appointments on Monday -/
def monday_appointments : ℕ := 5

/-- Duration of each Monday appointment in hours -/
def monday_appointment_duration : ℝ := 1.5

/-- Duration of Tuesday appointment in hours -/
def tuesday_appointment_duration : ℝ := 3

/-- Number of appointments on Thursday -/
def thursday_appointments : ℕ := 2

/-- Duration of each Thursday appointment in hours -/
def thursday_appointment_duration : ℝ := 2

/-- Duration of Saturday appointment in hours -/
def saturday_appointment_duration : ℝ := 6

/-- Total earnings for the week -/
def total_earnings : ℝ :=
  hourly_rate * (monday_appointments * monday_appointment_duration +
                 tuesday_appointment_duration +
                 thursday_appointments * thursday_appointment_duration +
                 saturday_appointment_duration)

theorem amanda_earnings : total_earnings = 410 := by
  sorry

end amanda_earnings_l1424_142400


namespace events_complementary_l1424_142469

-- Define the sample space for a fair die
def DieOutcome := Fin 6

-- Define Event 1: odd numbers
def Event1 (outcome : DieOutcome) : Prop :=
  outcome.val % 2 = 1

-- Define Event 2: even numbers
def Event2 (outcome : DieOutcome) : Prop :=
  outcome.val % 2 = 0

-- Theorem stating that Event1 and Event2 are complementary
theorem events_complementary :
  ∀ (outcome : DieOutcome), Event1 outcome ↔ ¬Event2 outcome :=
sorry

end events_complementary_l1424_142469


namespace quiz_competition_outcomes_quiz_competition_proof_l1424_142472

def participants : Nat := 6

theorem quiz_competition_outcomes (rita_not_third : Bool) : Nat :=
  participants * (participants - 1) * (participants - 2)

theorem quiz_competition_proof :
  quiz_competition_outcomes true = 120 := by sorry

end quiz_competition_outcomes_quiz_competition_proof_l1424_142472


namespace absolute_value_fraction_l1424_142478

theorem absolute_value_fraction (i : ℂ) : i * i = -1 → Complex.abs ((3 - i) / (i + 2)) = Real.sqrt 2 := by
  sorry

end absolute_value_fraction_l1424_142478


namespace problem_solution_l1424_142496

theorem problem_solution (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 6) : 
  (a^2 + b^2 = 13) ∧ ((a - b)^2 = 1) := by
sorry

end problem_solution_l1424_142496


namespace bus_driver_regular_rate_l1424_142470

/-- Calculates the regular hourly rate for a bus driver given their total hours worked,
    total compensation, and overtime policy. -/
def calculate_regular_rate (total_hours : ℕ) (total_compensation : ℚ) : ℚ :=
  let regular_hours := min total_hours 40
  let overtime_hours := total_hours - regular_hours
  let rate := total_compensation / (regular_hours + 1.75 * overtime_hours)
  rate

/-- Theorem stating that given the specific conditions of the bus driver's work week,
    their regular hourly rate is $16. -/
theorem bus_driver_regular_rate :
  calculate_regular_rate 54 1032 = 16 := by
  sorry

end bus_driver_regular_rate_l1424_142470


namespace Ca_concentration_after_mixing_l1424_142446

-- Define the constants
def K_sp : ℝ := 4.96e-9
def c_Na2CO3 : ℝ := 0.40
def c_CaCl2 : ℝ := 0.20

-- Define the theorem
theorem Ca_concentration_after_mixing :
  let c_CO3_remaining : ℝ := (c_Na2CO3 - c_CaCl2) / 2
  let c_Ca : ℝ := K_sp / c_CO3_remaining
  c_Ca = 4.96e-8 := by sorry

end Ca_concentration_after_mixing_l1424_142446


namespace arrangement_with_A_middle_or_sides_arrangement_with_males_grouped_arrangement_with_males_not_grouped_arrangement_with_ABC_order_fixed_arrangement_A_not_left_B_not_right_arrangement_with_extra_female_no_adjacent_arrangement_in_two_rows_arrangement_with_person_between_A_and_B_l1424_142494

-- Common definitions
def male_students : ℕ := 3
def female_students : ℕ := 2
def total_students : ℕ := male_students + female_students

-- (1)
theorem arrangement_with_A_middle_or_sides :
  (3 * (total_students - 1).factorial) = 72 := by sorry

-- (2)
theorem arrangement_with_males_grouped :
  (male_students.factorial * (total_students - male_students + 1).factorial) = 36 := by sorry

-- (3)
theorem arrangement_with_males_not_grouped :
  (female_students.factorial * male_students.factorial) = 12 := by sorry

-- (4)
theorem arrangement_with_ABC_order_fixed :
  (total_students.factorial / male_students.factorial) = 20 := by sorry

-- (5)
theorem arrangement_A_not_left_B_not_right :
  ((total_students - 1) * (total_students - 1).factorial - 
   (total_students - 2) * (total_students - 2).factorial) = 78 := by sorry

-- (6)
def extra_female_student : ℕ := 1
def new_total_students : ℕ := total_students + extra_female_student

theorem arrangement_with_extra_female_no_adjacent :
  (male_students.factorial * (new_total_students - male_students + 1).factorial) = 144 := by sorry

-- (7)
theorem arrangement_in_two_rows :
  total_students.factorial = 120 := by sorry

-- (8)
theorem arrangement_with_person_between_A_and_B :
  (3 * 2 * male_students.factorial) = 36 := by sorry

end arrangement_with_A_middle_or_sides_arrangement_with_males_grouped_arrangement_with_males_not_grouped_arrangement_with_ABC_order_fixed_arrangement_A_not_left_B_not_right_arrangement_with_extra_female_no_adjacent_arrangement_in_two_rows_arrangement_with_person_between_A_and_B_l1424_142494


namespace f_bound_l1424_142428

def f (x : ℝ) : ℝ := x^2 - x + 13

theorem f_bound (a x : ℝ) (h : |x - a| < 1) : |f x - f a| < 2 * (|a| + 1) := by
  sorry

end f_bound_l1424_142428


namespace octagons_700_sticks_4901_l1424_142440

/-- The number of sticks required to construct a series of octagons -/
def sticks_for_octagons (n : ℕ) : ℕ :=
  if n = 0 then 0
  else 8 + 7 * (n - 1)

/-- Theorem stating that 700 octagons require 4901 sticks -/
theorem octagons_700_sticks_4901 : sticks_for_octagons 700 = 4901 := by
  sorry

end octagons_700_sticks_4901_l1424_142440


namespace pump_out_time_for_specific_basement_l1424_142473

/-- Represents the dimensions and flooding of a basement -/
structure Basement :=
  (length : ℝ)
  (width : ℝ)
  (depth_inches : ℝ)

/-- Represents a water pump -/
structure Pump :=
  (rate : ℝ)  -- gallons per minute

/-- Calculates the time required to pump out a flooded basement -/
def pump_out_time (b : Basement) (pumps : List Pump) (cubic_foot_to_gallon : ℝ) : ℝ :=
  sorry

/-- Theorem stating the time required to pump out the specific basement -/
theorem pump_out_time_for_specific_basement :
  let basement := Basement.mk 40 20 24
  let pumps := [Pump.mk 10, Pump.mk 10, Pump.mk 10]
  pump_out_time basement pumps 7.5 = 400 := by
  sorry

end pump_out_time_for_specific_basement_l1424_142473


namespace rectangle_area_increase_l1424_142474

theorem rectangle_area_increase (x y : ℝ) 
  (area_eq : x * y = 180)
  (perimeter_eq : 2 * x + 2 * y = 54) :
  (x + 6) * (y + 6) = 378 := by
  sorry

end rectangle_area_increase_l1424_142474


namespace cubic_roots_relation_l1424_142423

theorem cubic_roots_relation (m n p q : ℝ) : 
  (∃ α β : ℝ, α^2 + m*α + n = 0 ∧ β^2 + m*β + n = 0) →
  (∃ γ δ : ℝ, γ^2 + p*γ + q = 0 ∧ δ^2 + p*δ + q = 0) →
  (∀ α β γ δ : ℝ, 
    (α^2 + m*α + n = 0 ∧ β^2 + m*β + n = 0) →
    (γ^2 + p*γ + q = 0 ∧ δ^2 + p*δ + q = 0) →
    (γ = α^3 ∧ δ = β^3 ∨ γ = β^3 ∧ δ = α^3)) →
  p = m^3 - 3*m*n :=
by sorry

end cubic_roots_relation_l1424_142423


namespace other_divisor_problem_l1424_142427

theorem other_divisor_problem (n : ℕ) (h1 : n = 174) : 
  ∃ (x : ℕ), x ≠ 5 ∧ x < 170 ∧ 
  (∀ y : ℕ, y < 170 → y ≠ 5 → n % y = 4 → y ≤ x) ∧
  n % x = 4 ∧ n % 5 = 4 := by
  sorry

end other_divisor_problem_l1424_142427


namespace third_bed_theorem_l1424_142411

/-- Represents the number of carrots harvested from each bed and the total harvest weight -/
structure CarrotHarvest where
  first_bed : ℕ
  second_bed : ℕ
  total_weight : ℕ
  carrots_per_pound : ℕ

/-- Calculates the number of carrots in the third bed given the harvest information -/
def third_bed_carrots (harvest : CarrotHarvest) : ℕ :=
  harvest.total_weight * harvest.carrots_per_pound - (harvest.first_bed + harvest.second_bed)

/-- Theorem stating that given the specific harvest conditions, the third bed contains 78 carrots -/
theorem third_bed_theorem (harvest : CarrotHarvest)
  (h1 : harvest.first_bed = 55)
  (h2 : harvest.second_bed = 101)
  (h3 : harvest.total_weight = 39)
  (h4 : harvest.carrots_per_pound = 6) :
  third_bed_carrots harvest = 78 := by
  sorry

#eval third_bed_carrots { first_bed := 55, second_bed := 101, total_weight := 39, carrots_per_pound := 6 }

end third_bed_theorem_l1424_142411


namespace books_initially_l1424_142471

/-- Given that Paul bought some books and ended up with a certain total, 
    this theorem proves how many books he had initially. -/
theorem books_initially (bought : ℕ) (total_after : ℕ) (h : bought = 101) (h' : total_after = 151) :
  total_after - bought = 50 := by
  sorry

end books_initially_l1424_142471


namespace unique_integer_expression_l1424_142495

/-- The function representing the given expression -/
def f (x y : ℕ+) : ℚ := (x^2 + y) / (x * y + 1)

/-- The theorem stating that 1 is the only positive integer expressible
    by the function for at least two distinct pairs of positive integers -/
theorem unique_integer_expression :
  ∀ n : ℕ+, (∃ (x₁ y₁ x₂ y₂ : ℕ+), (x₁, y₁) ≠ (x₂, y₂) ∧ f x₁ y₁ = n ∧ f x₂ y₂ = n) ↔ n = 1 := by
  sorry

end unique_integer_expression_l1424_142495


namespace base5_calculation_l1424_142460

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 5 --/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Theorem: 231₅ × 24₅ - 12₅ = 12132₅ in base 5 --/
theorem base5_calculation : 
  base10ToBase5 (base5ToBase10 231 * base5ToBase10 24 - base5ToBase10 12) = 12132 := by sorry

end base5_calculation_l1424_142460


namespace books_read_l1424_142422

theorem books_read (total_books : ℕ) (unread_books : ℕ) (h1 : total_books = 20) (h2 : unread_books = 5) :
  total_books - unread_books = 15 := by
  sorry

end books_read_l1424_142422


namespace sandy_work_hours_l1424_142493

/-- Sandy's work schedule -/
structure WorkSchedule where
  total_hours : ℕ
  num_days : ℕ
  hours_per_day : ℕ
  equal_hours : total_hours = num_days * hours_per_day

/-- Theorem: Sandy worked 9 hours per day -/
theorem sandy_work_hours (schedule : WorkSchedule)
  (h1 : schedule.total_hours = 45)
  (h2 : schedule.num_days = 5) :
  schedule.hours_per_day = 9 := by
  sorry

end sandy_work_hours_l1424_142493


namespace intersection_nonempty_implies_a_positive_l1424_142425

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = Real.log p.1}
def B (a : ℝ) : Set (ℝ × ℝ) := {p | p.1 = a}

-- State the theorem
theorem intersection_nonempty_implies_a_positive (a : ℝ) :
  (A ∩ B a).Nonempty → a > 0 := by
  sorry

end intersection_nonempty_implies_a_positive_l1424_142425


namespace sqrt_equation_solution_l1424_142419

theorem sqrt_equation_solution :
  ∀ a b : ℕ+,
    a < b →
    (Real.sqrt (1 + Real.sqrt (25 + 20 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b) →
    (a = 2 ∧ b = 6) := by
  sorry

end sqrt_equation_solution_l1424_142419


namespace smaller_exterior_angle_implies_obtuse_l1424_142484

-- Define a triangle
structure Triangle where
  -- We don't need to specify the exact properties of a triangle here

-- Define the property of having an exterior angle smaller than its adjacent interior angle
def has_smaller_exterior_angle (t : Triangle) : Prop :=
  ∃ (exterior_angle interior_angle : ℝ), exterior_angle < interior_angle

-- Define an obtuse triangle
def is_obtuse (t : Triangle) : Prop :=
  ∃ (angle : ℝ), angle > Real.pi / 2

-- State the theorem
theorem smaller_exterior_angle_implies_obtuse (t : Triangle) :
  has_smaller_exterior_angle t → is_obtuse t :=
sorry

end smaller_exterior_angle_implies_obtuse_l1424_142484


namespace tangent_sine_equality_l1424_142458

open Real

theorem tangent_sine_equality (α : ℝ) :
  (∃ k : ℤ, -π/2 + 2*π*(k : ℝ) < α ∧ α < π/2 + 2*π*(k : ℝ)) ↔
  Real.sqrt ((tan α)^2 - (sin α)^2) = tan α * sin α :=
sorry

end tangent_sine_equality_l1424_142458


namespace triple_hash_100_l1424_142466

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.4 * N + 3

-- State the theorem
theorem triple_hash_100 : hash (hash (hash 100)) = 11.08 := by
  sorry

end triple_hash_100_l1424_142466


namespace sin_60_abs_5_pi_sqrt2_equality_l1424_142491

theorem sin_60_abs_5_pi_sqrt2_equality : 
  2 * Real.sin (π / 3) + |-5| - (π - Real.sqrt 2) ^ 0 = Real.sqrt 3 + 4 := by
  sorry

end sin_60_abs_5_pi_sqrt2_equality_l1424_142491


namespace smaller_number_proof_l1424_142455

theorem smaller_number_proof (x y : ℝ) : 
  x - y = 9 → x + y = 46 → min x y = 18.5 := by
sorry

end smaller_number_proof_l1424_142455


namespace jonathan_social_media_time_l1424_142437

/-- Calculates the total time spent on social media in a week -/
def social_media_time_per_week (daily_time : ℕ) (days_in_week : ℕ) : ℕ :=
  daily_time * days_in_week

/-- Proves that Jonathan spends 21 hours on social media in a week -/
theorem jonathan_social_media_time :
  social_media_time_per_week 3 7 = 21 := by
  sorry

end jonathan_social_media_time_l1424_142437


namespace square_ratio_theorem_l1424_142406

theorem square_ratio_theorem : 
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (75 : ℚ) / 27 = (a * (b.sqrt : ℚ) / c) ^ 2 ∧
    a + b + c = 9 := by
  sorry

end square_ratio_theorem_l1424_142406


namespace equation_solution_l1424_142464

theorem equation_solution : ∃ x : ℝ, (23 - 5 = 3 + x) ∧ (x = 15) := by
  sorry

end equation_solution_l1424_142464


namespace warrior_truth_count_l1424_142418

/-- Represents the types of weapons a warrior can have as their favorite. -/
inductive Weapon
| sword
| spear
| axe
| bow

/-- Represents a warrior's truthfulness. -/
inductive Truthfulness
| truthful
| liar

/-- Represents the problem setup. -/
structure WarriorProblem where
  totalWarriors : Nat
  swordYes : Nat
  spearYes : Nat
  axeYes : Nat
  bowYes : Nat

/-- The main theorem to prove. -/
theorem warrior_truth_count (problem : WarriorProblem)
  (h_total : problem.totalWarriors = 33)
  (h_sword : problem.swordYes = 13)
  (h_spear : problem.spearYes = 15)
  (h_axe : problem.axeYes = 20)
  (h_bow : problem.bowYes = 27)
  : { truthfulCount : Nat // 
      truthfulCount = 12 ∧
      truthfulCount + (problem.totalWarriors - truthfulCount) * 3 = 
        problem.swordYes + problem.spearYes + problem.axeYes + problem.bowYes } :=
  sorry


end warrior_truth_count_l1424_142418


namespace min_cut_length_for_non_triangle_l1424_142481

theorem min_cut_length_for_non_triangle (a b c : ℕ) (ha : a = 9) (hb : b = 16) (hc : c = 18) :
  ∃ x : ℕ, x = 8 ∧
  (∀ y : ℕ, y < x → (a - y) + (b - y) > (c - y)) ∧
  (a - x) + (b - x) ≤ (c - x) :=
by sorry

end min_cut_length_for_non_triangle_l1424_142481


namespace complex_equality_implies_a_value_l1424_142448

theorem complex_equality_implies_a_value (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (2 + Complex.I)
  Complex.re z = Complex.im z → a = 3 := by
sorry

end complex_equality_implies_a_value_l1424_142448


namespace sector_area_l1424_142497

/-- The area of a circular sector with central angle π/3 and radius 4 is 8π/3 -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 4) :
  (1 / 2) * θ * r^2 = 8 * π / 3 := by
  sorry

end sector_area_l1424_142497


namespace fifteen_point_seven_billion_in_scientific_notation_l1424_142485

-- Define the number of billions
def billions : ℝ := 15.7

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.57 * (10 ^ 9)

-- Theorem statement
theorem fifteen_point_seven_billion_in_scientific_notation :
  billions * (10 ^ 9) = scientific_notation := by
  sorry

end fifteen_point_seven_billion_in_scientific_notation_l1424_142485


namespace fraction_ordering_l1424_142476

theorem fraction_ordering : (6 : ℚ) / 29 < 8 / 25 ∧ 8 / 25 < 10 / 31 := by
  sorry

end fraction_ordering_l1424_142476


namespace polynomial_evaluation_l1424_142430

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 5 = 5 := by
  sorry

end polynomial_evaluation_l1424_142430


namespace cos_sin_thirty_squared_difference_l1424_142408

theorem cos_sin_thirty_squared_difference :
  let cos_thirty : ℝ := Real.sqrt 3 / 2
  let sin_thirty : ℝ := 1 / 2
  cos_thirty ^ 2 - sin_thirty ^ 2 = 1 / 2 := by
  sorry

end cos_sin_thirty_squared_difference_l1424_142408


namespace optionC_is_most_suitable_l1424_142461

structure SamplingMethod where
  method : String
  representativeOfAllStudents : Bool
  includesAllGrades : Bool
  unbiased : Bool

def cityJuniorHighSchools : Set String := sorry

def isMostSuitableSamplingMethod (m : SamplingMethod) : Prop :=
  m.representativeOfAllStudents ∧ m.includesAllGrades ∧ m.unbiased

def optionC : SamplingMethod := {
  method := "Randomly select 1000 students from each of the three grades in junior high schools in the city",
  representativeOfAllStudents := true,
  includesAllGrades := true,
  unbiased := true
}

theorem optionC_is_most_suitable :
  isMostSuitableSamplingMethod optionC :=
sorry

end optionC_is_most_suitable_l1424_142461


namespace largest_fourth_number_l1424_142414

/-- Represents a two-digit number -/
def TwoDigitNumber := {n : ℕ // 10 ≤ n ∧ n < 100}

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The problem setup -/
def fourthNumberProblem (a b c d : TwoDigitNumber) : Prop :=
  a.val = 34 ∧ b.val = 21 ∧ c.val = 65 ∧ 
  (∃ (x : ℕ), d.val = 40 + x ∧ x < 10) ∧
  4 * (sumOfDigits a.val + sumOfDigits b.val + sumOfDigits c.val + sumOfDigits d.val) = 
    a.val + b.val + c.val + d.val

/-- The theorem to be proved -/
theorem largest_fourth_number : 
  ∀ (a b c d : TwoDigitNumber), 
    fourthNumberProblem a b c d → d.val ≤ 49 := by sorry

end largest_fourth_number_l1424_142414


namespace tax_calculation_l1424_142450

/-- Given gross pay and net pay, calculates the tax amount -/
def calculate_tax (gross_pay : ℝ) (net_pay : ℝ) : ℝ :=
  gross_pay - net_pay

theorem tax_calculation :
  let gross_pay : ℝ := 450
  let net_pay : ℝ := 315
  calculate_tax gross_pay net_pay = 135 := by
sorry

end tax_calculation_l1424_142450


namespace half_sum_abs_diff_squares_l1424_142442

theorem half_sum_abs_diff_squares : 
  (1/2 : ℝ) * (|20^2 - 15^2| + |15^2 - 20^2|) = 175 := by
  sorry

end half_sum_abs_diff_squares_l1424_142442


namespace johns_annual_epipen_cost_l1424_142486

/-- Calculates the annual cost of EpiPens for John given the replacement frequency,
    cost per EpiPen, and insurance coverage percentage. -/
def annual_epipen_cost (replacement_months : ℕ) (cost_per_epipen : ℕ) (insurance_coverage_percent : ℕ) : ℕ :=
  let epipens_per_year : ℕ := 12 / replacement_months
  let insurance_coverage : ℕ := cost_per_epipen * insurance_coverage_percent / 100
  let cost_after_insurance : ℕ := cost_per_epipen - insurance_coverage
  epipens_per_year * cost_after_insurance

/-- Theorem stating that John's annual cost for EpiPens is $250 -/
theorem johns_annual_epipen_cost :
  annual_epipen_cost 6 500 75 = 250 := by
  sorry

end johns_annual_epipen_cost_l1424_142486


namespace convention_delegates_l1424_142432

theorem convention_delegates (total : ℕ) 
  (h1 : 16 ≤ total) 
  (h2 : (total - 16) % 2 = 0) 
  (h3 : 10 ≤ total - 16 - (total - 16) / 2) : 
  total = 36 := by
  sorry

end convention_delegates_l1424_142432


namespace velocity_from_similarity_l1424_142417

/-- Given a, T, R, L, and x as real numbers, where x represents a distance,
    and assuming the equation (a * T) / (a * T - R) = (L + x) / x holds,
    prove that the velocity of the point described by x is a * (L / R). -/
theorem velocity_from_similarity (a T R L x : ℝ) (h : (a * T) / (a * T - R) = (L + x) / x) :
  x / T = a * L / R := by
  sorry

end velocity_from_similarity_l1424_142417


namespace triangle_expression_negative_l1424_142420

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

-- Theorem statement
theorem triangle_expression_negative (t : Triangle) : (t.a - t.c)^2 - t.b^2 < 0 := by
  sorry

end triangle_expression_negative_l1424_142420


namespace contractor_problem_l1424_142441

/-- Proves that the original number of men employed is 12 --/
theorem contractor_problem (initial_days : ℕ) (absent_men : ℕ) (actual_days : ℕ) 
  (h1 : initial_days = 5)
  (h2 : absent_men = 8)
  (h3 : actual_days = 15) :
  ∃ (original_men : ℕ), 
    original_men * initial_days = (original_men - absent_men) * actual_days ∧ 
    original_men = 12 := by
  sorry

end contractor_problem_l1424_142441


namespace rearranged_cube_surface_area_l1424_142487

def slice_heights : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

def last_slice_height (heights : List ℚ) : ℚ :=
  1 - (heights.sum)

def surface_area (heights : List ℚ) : ℚ :=
  2 + 2 + 2  -- top/bottom + sides + front/back

theorem rearranged_cube_surface_area :
  surface_area slice_heights = 6 := by
  sorry

end rearranged_cube_surface_area_l1424_142487


namespace remainder_product_mod_twelve_l1424_142492

theorem remainder_product_mod_twelve : (1425 * 1427 * 1429) % 12 = 3 := by
  sorry

end remainder_product_mod_twelve_l1424_142492


namespace power_comparison_l1424_142457

theorem power_comparison (h1 : 2 > 1) (h2 : -1.1 > -1.2) : 2^(-1.1) > 2^(-1.2) := by
  sorry

end power_comparison_l1424_142457


namespace fourth_root_over_seventh_root_of_seven_l1424_142426

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (h : x > 0) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
by
  sorry

end fourth_root_over_seventh_root_of_seven_l1424_142426


namespace mark_increase_ratio_l1424_142482

/-- Proves the ratio of increase in average marks to original average marks
    when one pupil's mark is increased by 40 in a class of 80 pupils. -/
theorem mark_increase_ratio (T : ℝ) (A : ℝ) (h1 : A = T / 80) :
  let new_average := (T + 40) / 80
  let increase := new_average - A
  increase / A = 1 / (2 * A) :=
by sorry

end mark_increase_ratio_l1424_142482


namespace pear_vendor_theorem_l1424_142424

/-- Represents the actions of a pear vendor over two days --/
def pear_vendor_problem (initial_pears : ℝ) : Prop :=
  let day1_sold := 0.8 * initial_pears
  let day1_remaining := initial_pears - day1_sold
  let day1_thrown := 0.5 * day1_remaining
  let day2_start := day1_remaining - day1_thrown
  let day2_sold := 0.8 * day2_start
  let day2_thrown := day2_start - day2_sold
  let total_thrown := day1_thrown + day2_thrown
  (total_thrown / initial_pears) * 100 = 12

/-- Theorem stating that the pear vendor throws away 12% of the initial pears --/
theorem pear_vendor_theorem :
  ∀ initial_pears : ℝ, initial_pears > 0 → pear_vendor_problem initial_pears :=
by
  sorry

end pear_vendor_theorem_l1424_142424


namespace max_free_squares_l1424_142409

/-- Represents a chessboard with bugs -/
structure BugChessboard (n : ℕ) where
  size : ℕ
  size_eq : size = n

/-- Represents a valid move of bugs on the chessboard -/
def ValidMove (board : BugChessboard n) : Prop :=
  ∀ i j : ℕ, i < n ∧ j < n →
    ∃ (i₁ j₁ i₂ j₂ : ℕ), 
      (i₁ < n ∧ j₁ < n ∧ i₂ < n ∧ j₂ < n) ∧
      ((i₁ = i ∧ (j₁ = j + 1 ∨ j₁ = j - 1)) ∨ (j₁ = j ∧ (i₁ = i + 1 ∨ i₁ = i - 1))) ∧
      ((i₂ = i ∧ (j₂ = j + 1 ∨ j₂ = j - 1)) ∨ (j₂ = j ∧ (i₂ = i + 1 ∨ i₂ = i - 1))) ∧
      (i₁ ≠ i₂ ∨ j₁ ≠ j₂)

/-- The number of free squares after a valid move -/
def FreeSquares (board : BugChessboard n) (move : ValidMove board) : ℕ := sorry

/-- The main theorem: the maximal number of free squares after one move is n^2 -/
theorem max_free_squares (n : ℕ) (board : BugChessboard n) :
  ∃ (move : ValidMove board), FreeSquares board move = n^2 :=
sorry

end max_free_squares_l1424_142409


namespace minimum_value_f_l1424_142454

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

theorem minimum_value_f (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ f a 1 ∧ f a 1 = 1) ∨
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ f a (Real.sqrt (-a/2)) ∧
    f a (Real.sqrt (-a/2)) = a/2 * Real.log (-a/2) - a/2 ∧
    -2*(Real.exp 1)^2 < a ∧ a < -2) ∨
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ f a (Real.exp 1) ∧
    f a (Real.exp 1) = a + (Real.exp 1)^2) := by
  sorry

end minimum_value_f_l1424_142454


namespace percentage_less_than_150000_l1424_142499

/-- Represents the percentage of counties in a specific population range -/
structure PopulationRange where
  percentage : ℝ
  lower_bound : ℕ
  upper_bound : Option ℕ

/-- Proves that the percentage of counties with fewer than 150,000 residents is 83% -/
theorem percentage_less_than_150000 
  (less_than_10000 : PopulationRange)
  (between_10000_and_49999 : PopulationRange)
  (between_50000_and_149999 : PopulationRange)
  (more_than_150000 : PopulationRange)
  (h1 : less_than_10000.percentage = 21)
  (h2 : between_10000_and_49999.percentage = 44)
  (h3 : between_50000_and_149999.percentage = 18)
  (h4 : more_than_150000.percentage = 17)
  (h5 : less_than_10000.upper_bound = some 9999)
  (h6 : between_10000_and_49999.lower_bound = 10000 ∧ between_10000_and_49999.upper_bound = some 49999)
  (h7 : between_50000_and_149999.lower_bound = 50000 ∧ between_50000_and_149999.upper_bound = some 149999)
  (h8 : more_than_150000.lower_bound = 150000 ∧ more_than_150000.upper_bound = none)
  (h9 : less_than_10000.percentage + between_10000_and_49999.percentage + between_50000_and_149999.percentage + more_than_150000.percentage = 100) :
  less_than_10000.percentage + between_10000_and_49999.percentage + between_50000_and_149999.percentage = 83 := by
  sorry


end percentage_less_than_150000_l1424_142499


namespace x_in_interval_l1424_142479

theorem x_in_interval (x : ℝ) (hx : x ≠ 0) : x = 2 * (1 / x) * (-x) → -4 < x ∧ x ≤ 0 := by
  sorry

end x_in_interval_l1424_142479


namespace grasshopper_jump_distance_l1424_142429

theorem grasshopper_jump_distance (frog_jump : ℕ) (difference : ℕ) : 
  frog_jump = 40 → difference = 15 → frog_jump - difference = 25 := by
  sorry

end grasshopper_jump_distance_l1424_142429


namespace king_game_winner_l1424_142489

/-- Represents the result of the game -/
inductive GameResult
  | PlayerAWins
  | PlayerBWins

/-- Represents a chessboard of size m × n -/
structure Chessboard where
  m : Nat
  n : Nat

/-- Determines the winner of the game based on the chessboard size -/
def determineWinner (board : Chessboard) : GameResult :=
  if board.m * board.n % 2 == 0 then
    GameResult.PlayerAWins
  else
    GameResult.PlayerBWins

/-- Theorem stating the winning condition for the game -/
theorem king_game_winner (board : Chessboard) :
  determineWinner board = GameResult.PlayerAWins ↔ board.m * board.n % 2 == 0 := by
  sorry

end king_game_winner_l1424_142489


namespace frank_reading_time_l1424_142468

/-- Calculates the number of days needed to read a book -/
def days_to_read (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- Proves that it takes 569 days to read a book with 12518 pages at 22 pages per day -/
theorem frank_reading_time : days_to_read 12518 22 = 569 := by
  sorry

end frank_reading_time_l1424_142468


namespace min_sum_is_negative_442_l1424_142451

/-- An arithmetic progression with sum S_n for the first n terms. -/
structure ArithmeticProgression where
  S : ℕ → ℤ
  sum_3 : S 3 = -141
  sum_35 : S 35 = 35

/-- The minimum value of S_n for an arithmetic progression satisfying the given conditions. -/
def min_sum (ap : ArithmeticProgression) : ℤ :=
  sorry

/-- Theorem stating that the minimum value of S_n is -442. -/
theorem min_sum_is_negative_442 (ap : ArithmeticProgression) :
  min_sum ap = -442 := by
  sorry

end min_sum_is_negative_442_l1424_142451


namespace rectangle_length_l1424_142410

/-- Given a rectangle and a square, prove that the length of the rectangle is 15 cm. -/
theorem rectangle_length (w l : ℝ) (square_side : ℝ) : 
  w = 9 → 
  square_side = 12 → 
  4 * square_side = 2 * w + 2 * l → 
  l = 15 := by
  sorry

end rectangle_length_l1424_142410


namespace no_fascinating_function_l1424_142480

theorem no_fascinating_function : ¬ ∃ (F : ℤ → ℤ), 
  (∀ c : ℤ, ∃ x : ℤ, F x ≠ c) ∧ 
  (∀ x : ℤ, F x = F (412 - x)) ∧
  (∀ x : ℤ, F x = F (414 - x)) ∧
  (∀ x : ℤ, F x = F (451 - x)) :=
by sorry

end no_fascinating_function_l1424_142480


namespace pencils_across_diameter_l1424_142436

theorem pencils_across_diameter (radius : ℝ) (pencil_length : ℝ) :
  radius = 14 →
  pencil_length = 0.5 →
  (2 * radius) / pencil_length = 56 :=
by
  sorry

end pencils_across_diameter_l1424_142436


namespace divide_twelve_by_repeating_third_l1424_142439

/-- The repeating decimal 0.3333... --/
def repeating_third : ℚ := 1 / 3

/-- The result of dividing 12 by the repeating decimal 0.3333... --/
theorem divide_twelve_by_repeating_third : 12 / repeating_third = 36 := by sorry

end divide_twelve_by_repeating_third_l1424_142439


namespace regular_polygon_angle_characterization_l1424_142488

def is_regular_polygon_angle (angle : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 3 ∧ angle = 180 - 360 / n

def regular_polygon_angles : Set ℕ :=
  {60, 90, 108, 120, 135, 140, 144, 150, 156, 160, 162, 165, 168, 170, 171, 172, 174, 175, 176, 177, 178, 179}

theorem regular_polygon_angle_characterization :
  ∀ angle : ℕ, is_regular_polygon_angle angle ↔ angle ∈ regular_polygon_angles :=
by sorry

end regular_polygon_angle_characterization_l1424_142488


namespace line_x_axis_intersection_l1424_142475

/-- The line equation: 5y + 3x = 15 -/
def line_equation (x y : ℝ) : Prop := 5 * y + 3 * x = 15

/-- A point on the x-axis has y-coordinate equal to 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (5, 0)

theorem line_x_axis_intersection :
  let (x, y) := intersection_point
  line_equation x y ∧ on_x_axis x y :=
by sorry

end line_x_axis_intersection_l1424_142475


namespace average_speed_calculation_l1424_142452

theorem average_speed_calculation (total_distance : ℝ) (first_half_distance : ℝ) (second_half_distance : ℝ) 
  (first_half_speed : ℝ) (second_half_speed : ℝ) 
  (h1 : total_distance = 50)
  (h2 : first_half_distance = 25)
  (h3 : second_half_distance = 25)
  (h4 : first_half_speed = 60)
  (h5 : second_half_speed = 30)
  (h6 : total_distance = first_half_distance + second_half_distance) :
  (total_distance) / ((first_half_distance / first_half_speed) + (second_half_distance / second_half_speed)) = 40 := by
  sorry

end average_speed_calculation_l1424_142452


namespace triangle_properties_l1424_142477

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the following statements under the given conditions. -/
theorem triangle_properties (A B C a b c : Real) :
  -- Given conditions
  (4 * Real.sin (A / 2 - B / 2) ^ 2 + 4 * Real.sin A * Real.sin B = 2 + Real.sqrt 2) →
  (b = 4) →
  (1 / 2 * a * b * Real.sin C = 6) →
  -- Triangle inequality and angle sum
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (A + B + C = π) →
  (A > 0 ∧ B > 0 ∧ C > 0) →
  -- Statements to prove
  (C = π / 4 ∧
   c = Real.sqrt 10 ∧
   Real.tan (2 * B - C) = 7) :=
by sorry

end triangle_properties_l1424_142477


namespace count_perimeters_eq_42_l1424_142433

/-- Represents a quadrilateral EFGH with specific properties -/
structure Quadrilateral where
  ef : ℕ+
  fg : ℕ+
  gh : ℕ+
  eh : ℕ+
  perimeter_lt_1200 : ef.val + fg.val + gh.val + eh.val < 1200
  right_angle_f : True
  right_angle_g : True
  ef_eq_3 : ef = 3
  gh_eq_eh : gh = eh

/-- The number of different possible perimeter values -/
def count_perimeters : ℕ := sorry

/-- Main theorem stating the number of different possible perimeter values -/
theorem count_perimeters_eq_42 : count_perimeters = 42 := by sorry

end count_perimeters_eq_42_l1424_142433


namespace pen_price_is_14_l1424_142407

/-- The price of a pen in yuan -/
def pen_price : ℝ := 14

/-- The price of a ballpoint pen in yuan -/
def ballpoint_price : ℝ := 7

/-- The total cost of the pens in yuan -/
def total_cost : ℝ := 49

theorem pen_price_is_14 :
  (2 * pen_price + 3 * ballpoint_price = total_cost) ∧
  (3 * pen_price + ballpoint_price = total_cost) →
  pen_price = 14 := by
sorry

end pen_price_is_14_l1424_142407


namespace symmetric_point_on_circle_l1424_142416

/-- The circle equation: x^2 + y^2 + 2x - 4y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line equation: x - ay + 2 = 0 -/
def line_equation (a x y : ℝ) : Prop :=
  x - a*y + 2 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

theorem symmetric_point_on_circle (a : ℝ) :
  line_equation a (circle_center.1) (circle_center.2) →
  a = -1/2 := by
  sorry

end symmetric_point_on_circle_l1424_142416


namespace pages_written_first_week_pages_written_first_week_proof_l1424_142467

/-- Calculates the number of pages written in the first week of a 500-page book -/
theorem pages_written_first_week : ℕ :=
  let total_pages : ℕ := 500
  let second_week_write_ratio : ℚ := 30 / 100
  let coffee_damage_ratio : ℚ := 20 / 100
  let remaining_empty_pages : ℕ := 196
  
  -- Define a function to calculate pages written in first week
  let pages_written (x : ℕ) : Prop :=
    let remaining_after_first := total_pages - x
    let remaining_after_second := remaining_after_first - (second_week_write_ratio * remaining_after_first).floor
    let damaged_pages := (coffee_damage_ratio * remaining_after_second).floor
    remaining_after_second - damaged_pages = remaining_empty_pages

  -- The theorem states that 150 satisfies the conditions
  150

/-- Proof of the theorem -/
theorem pages_written_first_week_proof : pages_written_first_week = 150 := by
  sorry

end pages_written_first_week_pages_written_first_week_proof_l1424_142467


namespace even_function_range_l1424_142459

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f is increasing on (0, +∞) if f(x) ≤ f(y) for all 0 < x < y -/
def IncreasingOnPositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x ≤ f y

theorem even_function_range (f : ℝ → ℝ) (a : ℝ) 
  (h_even : IsEven f)
  (h_incr : IncreasingOnPositive f)
  (h_cond : f a ≥ f 2) :
  a ∈ Set.Iic (-2) ∪ Set.Ici 2 := by
  sorry

end even_function_range_l1424_142459


namespace manny_purchase_theorem_l1424_142444

/-- The cost of a plastic chair in dollars -/
def chair_cost : ℚ := 55 / 5

/-- The cost of a portable table in dollars -/
def table_cost : ℚ := 3 * chair_cost

/-- Manny's initial amount in dollars -/
def initial_amount : ℚ := 100

/-- The cost of Manny's purchase (one table and two chairs) in dollars -/
def purchase_cost : ℚ := table_cost + 2 * chair_cost

/-- The amount left after Manny's purchase in dollars -/
def amount_left : ℚ := initial_amount - purchase_cost

theorem manny_purchase_theorem : amount_left = 45 := by
  sorry

end manny_purchase_theorem_l1424_142444


namespace problem_solution_l1424_142403

theorem problem_solution (x y : ℝ) 
  (h1 : 5 + x = 3 - y) 
  (h2 : 2 + y = 6 + x) : 
  5 - x = 8 := by
  sorry

end problem_solution_l1424_142403


namespace line_ab_equals_ba_infinite_lines_through_point_ray_ab_not_equal_ba_ray_line_length_incomparable_l1424_142434

-- Define basic geometric structures
structure Point : Type :=
  (x : ℝ) (y : ℝ)

structure Line : Type :=
  (p1 : Point) (p2 : Point)

structure Ray : Type :=
  (start : Point) (through : Point)

-- Define equality for lines
def line_eq (l1 l2 : Line) : Prop :=
  (l1.p1 = l2.p1 ∧ l1.p2 = l2.p2) ∨ (l1.p1 = l2.p2 ∧ l1.p2 = l2.p1)

-- Define inequality for rays
def ray_neq (r1 r2 : Ray) : Prop :=
  r1.start ≠ r2.start ∨ r1.through ≠ r2.through

-- Theorem statements
theorem line_ab_equals_ba (A B : Point) : 
  line_eq (Line.mk A B) (Line.mk B A) :=
sorry

theorem infinite_lines_through_point (P : Point) :
  ∀ n : ℕ, ∃ (lines : Fin n → Line), ∀ i : Fin n, (lines i).p1 = P :=
sorry

theorem ray_ab_not_equal_ba (A B : Point) :
  ray_neq (Ray.mk A B) (Ray.mk B A) :=
sorry

theorem ray_line_length_incomparable :
  ¬∃ (f : Ray → ℝ) (g : Line → ℝ), ∀ (r : Ray) (l : Line), f r < g l :=
sorry

end line_ab_equals_ba_infinite_lines_through_point_ray_ab_not_equal_ba_ray_line_length_incomparable_l1424_142434


namespace leila_cake_consumption_l1424_142462

def monday_cakes : ℕ := 6
def friday_cakes : ℕ := 9
def saturday_cakes : ℕ := 3 * monday_cakes

theorem leila_cake_consumption : 
  monday_cakes + friday_cakes + saturday_cakes = 33 := by
  sorry

end leila_cake_consumption_l1424_142462


namespace tan_3x_domain_l1424_142431

theorem tan_3x_domain (x : ℝ) : 
  ∃ y, y = Real.tan (3 * x) ↔ ∀ k : ℤ, x ≠ π / 6 + k * π / 3 :=
by sorry

end tan_3x_domain_l1424_142431


namespace circle_and_tangent_lines_l1424_142465

-- Define the circle with center (8, -3) passing through (5, 1)
def circle1 (x y : ℝ) : Prop := (x - 8)^2 + (y + 3)^2 = 25

-- Define the circle x^2 + y^2 = 25
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the two tangent lines
def tangentLine1 (x y : ℝ) : Prop := y = (4/3) * x - 25/3
def tangentLine2 (x y : ℝ) : Prop := y = (-3/4) * x - 25/4

theorem circle_and_tangent_lines :
  (∀ x y, circle1 x y ↔ ((x = 8 ∧ y = -3) ∨ (x = 5 ∧ y = 1))) ∧
  (∀ x y, tangentLine1 x y → circle2 x y → x = 1 ∧ y = -7) ∧
  (∀ x y, tangentLine2 x y → circle2 x y → x = 1 ∧ y = -7) :=
by sorry

end circle_and_tangent_lines_l1424_142465


namespace rita_backstroke_hours_l1424_142447

/-- Calculates the number of backstroke hours completed by Rita --/
def backstroke_hours (total_required : ℕ) (breaststroke : ℕ) (butterfly : ℕ) 
  (freestyle_sidestroke_per_month : ℕ) (months : ℕ) : ℕ :=
  total_required - (breaststroke + butterfly + freestyle_sidestroke_per_month * months)

/-- Theorem stating that Rita completed 50 hours of backstroke --/
theorem rita_backstroke_hours : 
  backstroke_hours 1500 9 121 220 6 = 50 := by
  sorry

end rita_backstroke_hours_l1424_142447


namespace trig_values_of_α_l1424_142435

-- Define the angle α and its properties
def α : Real := sorry

-- Define that the terminal side of α passes through (3, 4)
axiom terminal_point : ∃ (r : Real), r * Real.cos α = 3 ∧ r * Real.sin α = 4

-- Theorem to prove
theorem trig_values_of_α :
  Real.sin α = 4/5 ∧ Real.cos α = 3/5 ∧ Real.tan α = 4/3 := by
  sorry

end trig_values_of_α_l1424_142435


namespace division_remainder_problem_l1424_142443

theorem division_remainder_problem (L S : ℕ) (h1 : L - S = 1370) (h2 : L = 1626) 
  (h3 : L / S = 6) : L % S = 90 := by
  sorry

end division_remainder_problem_l1424_142443


namespace november_to_december_ratio_l1424_142413

/-- Represents the revenue of a toy store in a given month -/
structure Revenue where
  amount : ℝ
  amount_pos : amount > 0

/-- The toy store's revenues for November, December, and January -/
structure StoreRevenue where
  november : Revenue
  december : Revenue
  january : Revenue
  january_is_third_of_november : january.amount = (1/3) * november.amount
  december_is_average_multiple : december.amount = 2.5 * ((november.amount + january.amount) / 2)

/-- The ratio of November's revenue to December's revenue is 3:5 -/
theorem november_to_december_ratio (s : StoreRevenue) :
  s.november.amount / s.december.amount = 3/5 := by
  sorry

end november_to_december_ratio_l1424_142413


namespace gcd_repeated_six_digit_l1424_142463

def is_repeated_six_digit (n : ℕ) : Prop :=
  ∃ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ n = 1001 * m

theorem gcd_repeated_six_digit :
  ∃ g : ℕ, ∀ n : ℕ, is_repeated_six_digit n → Nat.gcd n g = g ∧ g = 1001 :=
sorry

end gcd_repeated_six_digit_l1424_142463


namespace house_number_painting_cost_l1424_142453

/-- Represents a side of the street with houses -/
structure StreetSide where
  start : ℕ
  diff : ℕ
  count : ℕ

/-- Calculates the cost of painting numbers for a given street side -/
def paintCost (side : StreetSide) : ℕ := sorry

/-- The problem statement -/
theorem house_number_painting_cost :
  let southSide : StreetSide := { start := 5, diff := 7, count := 25 }
  let northSide : StreetSide := { start := 2, diff := 8, count := 25 }
  paintCost southSide + paintCost northSide = 123 := by sorry

end house_number_painting_cost_l1424_142453


namespace rectangular_field_length_l1424_142449

/-- Proves the length of a rectangular field given specific conditions -/
theorem rectangular_field_length : ∀ w : ℝ,
  w > 0 →  -- width is positive
  (w + 10) * w = 171 →  -- area equation
  w + 10 = 19 :=  -- length equation
by
  sorry

#check rectangular_field_length

end rectangular_field_length_l1424_142449


namespace pentagon_area_l1424_142405

/-- The area of a pentagon with specific properties -/
theorem pentagon_area : 
  ∀ (s₁ s₂ s₃ s₄ s₅ : ℝ),
  s₁ = 16 ∧ s₂ = 25 ∧ s₃ = 30 ∧ s₄ = 26 ∧ s₅ = 25 →
  ∃ (triangle_area trapezoid_area : ℝ),
    triangle_area = (1/2) * s₁ * s₂ ∧
    trapezoid_area = (1/2) * (s₄ + s₅) * s₃ ∧
    triangle_area + trapezoid_area = 965 :=
by sorry


end pentagon_area_l1424_142405


namespace cost_of_tax_free_items_l1424_142402

/-- Given a total cost, sales tax, and tax rate, calculate the cost of tax-free items -/
theorem cost_of_tax_free_items
  (total_cost : ℝ)
  (sales_tax : ℝ)
  (tax_rate : ℝ)
  (h1 : total_cost = 25)
  (h2 : sales_tax = 0.30)
  (h3 : tax_rate = 0.05)
  : ∃ (tax_free_cost : ℝ), tax_free_cost = 19 :=
by sorry

end cost_of_tax_free_items_l1424_142402


namespace consecutive_even_numbers_properties_l1424_142421

/-- Represents a sequence of seven consecutive even numbers -/
structure ConsecutiveEvenNumbers where
  middle : ℤ
  sum : ℤ
  sum_eq : sum = 7 * middle

/-- Properties of the sequence of consecutive even numbers -/
theorem consecutive_even_numbers_properties (seq : ConsecutiveEvenNumbers)
  (h : seq.sum = 686) :
  let smallest := seq.middle - 6
  let median := seq.middle
  let mean := seq.sum / 7
  (smallest = 92) ∧ (median = 98) ∧ (mean = 98) := by
  sorry

#check consecutive_even_numbers_properties

end consecutive_even_numbers_properties_l1424_142421


namespace mike_picked_52_peaches_l1424_142445

/-- The number of peaches Mike initially had -/
def initial_peaches : ℕ := 34

/-- The total number of peaches Mike has after picking more -/
def total_peaches : ℕ := 86

/-- The number of peaches Mike picked -/
def picked_peaches : ℕ := total_peaches - initial_peaches

theorem mike_picked_52_peaches : picked_peaches = 52 := by
  sorry

end mike_picked_52_peaches_l1424_142445


namespace inequalities_proof_l1424_142438

theorem inequalities_proof :
  (∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2) ∧
  (∀ a b : ℝ, a > 0 ∧ 0 > b → a * b < a^2) ∧
  (∃ a b : ℝ, a * b = 4 ∧ a + b < 4) ∧
  (∀ a b c d : ℝ, a > b ∧ c > d → a - d > b - c) :=
sorry

end inequalities_proof_l1424_142438
