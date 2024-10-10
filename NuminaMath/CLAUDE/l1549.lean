import Mathlib

namespace odd_power_divisibility_l1549_154911

theorem odd_power_divisibility (a b : ℕ) (ha : Odd a) (hb : Odd b) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  ∀ n : ℕ, 0 < n → ∃ m : ℕ, (2^n ∣ a^m * b^2 - 1) ∨ (2^n ∣ b^m * a^2 - 1) := by
  sorry

end odd_power_divisibility_l1549_154911


namespace angle4_measure_l1549_154937

-- Define the angles
variable (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)

-- State the theorem
theorem angle4_measure
  (h1 : angle1 = 82)
  (h2 : angle2 = 34)
  (h3 : angle3 = 19)
  (h4 : angle5 = angle6 + 10)
  (h5 : angle1 + angle2 + angle3 + angle5 + angle6 = 180)
  (h6 : angle4 + angle5 + angle6 = 180) :
  angle4 = 135 := by
sorry

end angle4_measure_l1549_154937


namespace units_digit_sum_factorials_plus_1000_l1549_154963

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_plus_1000 :
  units_digit (sum_factorials 10 + 1000) = 3 := by sorry

end units_digit_sum_factorials_plus_1000_l1549_154963


namespace remaining_candy_l1549_154927

/-- Given a group of people who collected candy and ate some, calculate the remaining candy. -/
theorem remaining_candy (total_candy : ℕ) (num_people : ℕ) (candy_eaten_per_person : ℕ) :
  total_candy = 120 →
  num_people = 3 →
  candy_eaten_per_person = 6 →
  total_candy - (num_people * candy_eaten_per_person) = 102 := by
  sorry

end remaining_candy_l1549_154927


namespace doughnut_boxes_l1549_154902

theorem doughnut_boxes (total_doughnuts : ℕ) (doughnuts_per_box : ℕ) (h1 : total_doughnuts = 48) (h2 : doughnuts_per_box = 12) :
  total_doughnuts / doughnuts_per_box = 4 := by
  sorry

end doughnut_boxes_l1549_154902


namespace wednesdays_temperature_l1549_154929

theorem wednesdays_temperature (monday tuesday wednesday : ℤ) : 
  tuesday = monday + 4 →
  wednesday = monday - 6 →
  tuesday = 22 →
  wednesday = 12 := by
sorry

end wednesdays_temperature_l1549_154929


namespace probability_three_heads_in_eight_tosses_l1549_154926

theorem probability_three_heads_in_eight_tosses (n : Nat) (k : Nat) :
  n = 8 → k = 3 →
  (Nat.choose n k : Rat) / (2 ^ n : Rat) = 7 / 32 := by
  sorry

end probability_three_heads_in_eight_tosses_l1549_154926


namespace parking_cost_theorem_l1549_154935

/-- Calculates the average hourly parking cost for a given duration -/
def averageHourlyCost (baseCost : ℚ) (baseHours : ℚ) (additionalHourlyRate : ℚ) (totalHours : ℚ) : ℚ :=
  let totalCost := baseCost + (totalHours - baseHours) * additionalHourlyRate
  totalCost / totalHours

/-- Proves that the average hourly cost for 9 hours of parking is $3.03 -/
theorem parking_cost_theorem :
  let baseCost : ℚ := 15
  let baseHours : ℚ := 2
  let additionalHourlyRate : ℚ := 1.75
  let totalHours : ℚ := 9
  averageHourlyCost baseCost baseHours additionalHourlyRate totalHours = 3.03 := by
  sorry

#eval averageHourlyCost 15 2 1.75 9

end parking_cost_theorem_l1549_154935


namespace prob_shortest_diagonal_nonagon_l1549_154981

/-- A regular polygon with n sides. -/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of diagonals in a polygon with n sides. -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in a regular polygon with n sides. -/
def num_shortest_diagonals (n : ℕ) : ℕ := n

/-- The probability of selecting a shortest diagonal from all diagonals in a regular polygon. -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  (num_shortest_diagonals n : ℚ) / (num_diagonals n : ℚ)

theorem prob_shortest_diagonal_nonagon :
  prob_shortest_diagonal 9 = 1/3 := by sorry

end prob_shortest_diagonal_nonagon_l1549_154981


namespace boris_early_theorem_l1549_154936

/-- Represents the distance between two points -/
structure Distance where
  value : ℝ
  nonneg : 0 ≤ value

/-- Represents a speed (distance per unit time) -/
structure Speed where
  value : ℝ
  pos : 0 < value

/-- Represents a time duration -/
structure Time where
  value : ℝ
  nonneg : 0 ≤ value

/-- The scenario of Anna and Boris walking towards each other -/
structure WalkingScenario where
  d : Distance  -- distance between villages A and B
  v_A : Speed   -- Anna's speed
  v_B : Speed   -- Boris's speed
  t : Time      -- time they meet when starting simultaneously

variable (scenario : WalkingScenario)

/-- The distance Anna walks in the original scenario -/
def anna_distance : ℝ := scenario.v_A.value * scenario.t.value

/-- The distance Boris walks in the original scenario -/
def boris_distance : ℝ := scenario.v_B.value * scenario.t.value

/-- Condition: Anna and Boris meet when they start simultaneously -/
axiom meet_condition : anna_distance scenario + boris_distance scenario = scenario.d.value

/-- Condition: If Anna starts 30 minutes earlier, they meet 2 km closer to village B -/
axiom anna_early_condition : 
  scenario.v_A.value * (scenario.t.value + 0.5) + scenario.v_B.value * scenario.t.value 
  = scenario.d.value - 2

/-- Theorem: If Boris starts 30 minutes earlier, they meet 2 km closer to village A -/
theorem boris_early_theorem : 
  scenario.v_A.value * scenario.t.value + scenario.v_B.value * (scenario.t.value + 0.5) 
  = scenario.d.value + 2 := by
  sorry

end boris_early_theorem_l1549_154936


namespace same_grade_probability_l1549_154933

/-- Represents the grades in the school -/
inductive Grade
| A
| B
| C

/-- Represents a student volunteer -/
structure Student where
  grade : Grade

/-- The total number of student volunteers -/
def total_students : Nat := 560

/-- The number of students in each grade -/
def students_per_grade (g : Grade) : Nat :=
  match g with
  | Grade.A => 240
  | Grade.B => 160
  | Grade.C => 160

/-- The number of students selected from each grade for the charity event -/
def selected_per_grade (g : Grade) : Nat :=
  match g with
  | Grade.A => 3
  | Grade.B => 2
  | Grade.C => 2

/-- The total number of students selected for the charity event -/
def total_selected : Nat := 7

/-- The number of students to be selected for sanitation work -/
def sanitation_workers : Nat := 2

/-- Theorem: The probability of selecting 2 students from the same grade for sanitation work is 5/21 -/
theorem same_grade_probability :
  (Nat.choose total_selected sanitation_workers) = 21 ∧
  (Nat.choose (selected_per_grade Grade.A) sanitation_workers +
   Nat.choose (selected_per_grade Grade.B) sanitation_workers +
   Nat.choose (selected_per_grade Grade.C) sanitation_workers) = 5 :=
by sorry


end same_grade_probability_l1549_154933


namespace bill_amount_calculation_l1549_154988

/-- Calculates the face value of a bill given the true discount, interest rate, and time to maturity. -/
def faceBill (trueDiscount : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  trueDiscount * (1 + rate * time)

/-- Theorem: Given a true discount of 150 on a bill due in 9 months at 16% per annum, the amount of the bill is 168. -/
theorem bill_amount_calculation : 
  let trueDiscount : ℝ := 150
  let rate : ℝ := 0.16  -- 16% per annum
  let time : ℝ := 0.75  -- 9 months = 9/12 years = 0.75 years
  faceBill trueDiscount rate time = 168 := by
  sorry


end bill_amount_calculation_l1549_154988


namespace cougar_ratio_l1549_154960

theorem cougar_ratio (lions tigers total : ℕ) 
  (h1 : lions = 12)
  (h2 : tigers = 14)
  (h3 : total = 39) :
  (total - (lions + tigers)) * 2 = lions + tigers :=
by sorry

end cougar_ratio_l1549_154960


namespace sqrt_nested_roots_l1549_154948

theorem sqrt_nested_roots (N : ℝ) (h : N > 1) : 
  Real.sqrt (N * Real.sqrt (N * Real.sqrt N)) = N^(7/8) := by
  sorry

end sqrt_nested_roots_l1549_154948


namespace max_points_at_distance_l1549_154917

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Whether a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop := sorry

/-- The number of points on a circle that are at a given distance from a point -/
def numPointsAtDistance (c : Circle) (p : Point) (d : ℝ) : ℕ := sorry

theorem max_points_at_distance (C : Circle) (P : Point) :
  isOutside P C →
  (∃ (n : ℕ), numPointsAtDistance C P 5 = n ∧ 
    ∀ (m : ℕ), numPointsAtDistance C P 5 ≤ m → n ≤ m) →
  numPointsAtDistance C P 5 = 2 := by sorry

end max_points_at_distance_l1549_154917


namespace intersection_and_union_union_equality_condition_l1549_154943

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 6}
def C (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m}

-- Theorem for part (I)
theorem intersection_and_union :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 6}) ∧
  ((Aᶜ) ∪ B = {x | -1 < x ∧ x ≤ 6}) :=
sorry

-- Theorem for part (II)
theorem union_equality_condition (m : ℝ) :
  (B ∪ C m = B) ↔ m ≤ 3 :=
sorry

end intersection_and_union_union_equality_condition_l1549_154943


namespace tuesday_poodles_l1549_154910

/-- Represents the number of hours it takes to walk a dog of a specific breed --/
def walkTime (breed : String) : ℕ :=
  match breed with
  | "Poodle" => 2
  | "Chihuahua" => 1
  | "Labrador" => 3
  | _ => 0

/-- Represents the schedule for a specific day --/
structure DaySchedule where
  poodles : ℕ
  chihuahuas : ℕ
  labradors : ℕ

def monday : DaySchedule := { poodles := 4, chihuahuas := 2, labradors := 0 }
def wednesday : DaySchedule := { poodles := 0, chihuahuas := 0, labradors := 4 }

def totalHours : ℕ := 32

theorem tuesday_poodles :
  ∃ (tuesday : DaySchedule),
    tuesday.chihuahuas = monday.chihuahuas ∧
    totalHours =
      (monday.poodles * walkTime "Poodle" +
       monday.chihuahuas * walkTime "Chihuahua" +
       wednesday.labradors * walkTime "Labrador" +
       tuesday.poodles * walkTime "Poodle" +
       tuesday.chihuahuas * walkTime "Chihuahua") ∧
    tuesday.poodles = 4 :=
  sorry

end tuesday_poodles_l1549_154910


namespace popsicle_melting_rate_l1549_154904

/-- Given a sequence of 6 terms where each term is twice the previous term and the first term is 1,
    prove that the last term is equal to 32. -/
theorem popsicle_melting_rate (seq : Fin 6 → ℕ) 
    (h1 : seq 0 = 1)
    (h2 : ∀ i : Fin 5, seq (i.succ) = 2 * seq i) : 
  seq 5 = 32 := by
  sorry

end popsicle_melting_rate_l1549_154904


namespace arithmetic_sequence_solution_l1549_154978

theorem arithmetic_sequence_solution (y : ℝ) (h : y > 0) :
  (2^2 + 5^2) / 2 = y^2 → y = Real.sqrt (29 / 2) :=
by sorry

end arithmetic_sequence_solution_l1549_154978


namespace complement_union_equals_singleton_l1549_154966

def M : Set (ℝ × ℝ) := {p | p.2 ≠ p.1 + 2}

def N : Set (ℝ × ℝ) := {p | p.2 ≠ -p.1}

def I : Set (ℝ × ℝ) := Set.univ

theorem complement_union_equals_singleton : 
  (I \ (M ∪ N)) = {(-1, 1)} := by sorry

end complement_union_equals_singleton_l1549_154966


namespace jade_transactions_l1549_154976

theorem jade_transactions 
  (mabel_transactions : ℕ)
  (anthony_transactions : ℕ)
  (cal_transactions : ℕ)
  (jade_transactions : ℕ)
  (h1 : mabel_transactions = 90)
  (h2 : anthony_transactions = mabel_transactions + mabel_transactions / 10)
  (h3 : cal_transactions = anthony_transactions * 2 / 3)
  (h4 : jade_transactions = cal_transactions + 19) :
  jade_transactions = 85 := by
  sorry

end jade_transactions_l1549_154976


namespace equation_solution_l1549_154956

theorem equation_solution : ∃ x : ℚ, (1/6 : ℚ) + 2/x = 3/x + (1/15 : ℚ) ∧ x = 10 := by
  sorry

end equation_solution_l1549_154956


namespace cookie_radius_l1549_154989

/-- Given a circle described by the equation x^2 + y^2 + 2x - 4y - 7 = 0, its radius is 2√3 -/
theorem cookie_radius (x y : ℝ) : 
  (x^2 + y^2 + 2*x - 4*y - 7 = 0) → 
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ r = 2 * Real.sqrt 3 := by
sorry

end cookie_radius_l1549_154989


namespace weight_loss_problem_l1549_154982

theorem weight_loss_problem (total_loss weight_loss_2 weight_loss_3 weight_loss_4 : ℕ) 
  (h1 : total_loss = 103)
  (h2 : weight_loss_3 = 28)
  (h3 : weight_loss_4 = 28)
  (h4 : weight_loss_2 = weight_loss_3 + weight_loss_4 - 7) :
  ∃ (weight_loss_1 : ℕ), 
    weight_loss_1 + weight_loss_2 + weight_loss_3 + weight_loss_4 = total_loss ∧ 
    weight_loss_1 = 27 := by
  sorry

end weight_loss_problem_l1549_154982


namespace product_and_square_calculation_l1549_154906

theorem product_and_square_calculation :
  (99 * 101 = 9999) ∧ (98^2 = 9604) := by
  sorry

end product_and_square_calculation_l1549_154906


namespace count_numbers_eq_243_l1549_154945

/-- The count of three-digit numbers less than 500 that do not contain the digit 1 -/
def count_numbers : Nat :=
  let hundreds := {2, 3, 4}
  let other_digits := {0, 2, 3, 4, 5, 6, 7, 8, 9}
  (Finset.card hundreds) * (Finset.card other_digits) * (Finset.card other_digits)

/-- Theorem stating that the count of three-digit numbers less than 500 
    that do not contain the digit 1 is equal to 243 -/
theorem count_numbers_eq_243 : count_numbers = 243 := by
  sorry

end count_numbers_eq_243_l1549_154945


namespace faye_age_l1549_154959

/-- Given the ages of Chad, Diana, Eduardo, and Faye, prove that Faye is 18 years old. -/
theorem faye_age (C D E F : ℕ) 
  (h1 : D = E - 2)
  (h2 : E = C + 3)
  (h3 : F = C + 4)
  (h4 : D = 15) :
  F = 18 := by
  sorry

end faye_age_l1549_154959


namespace magnitude_of_scaled_complex_l1549_154901

theorem magnitude_of_scaled_complex (z : ℂ) :
  z = 3 - 2 * Complex.I →
  Complex.abs (-1/3 * z) = Real.sqrt 13 / 3 := by
  sorry

end magnitude_of_scaled_complex_l1549_154901


namespace journey_possible_l1549_154955

/-- Represents the journey parameters and conditions -/
structure JourneyParams where
  total_distance : ℝ
  motorcycle_speed : ℝ
  baldwin_speed : ℝ
  clark_speed : ℝ
  (total_distance_positive : total_distance > 0)
  (speeds_positive : motorcycle_speed > 0 ∧ baldwin_speed > 0 ∧ clark_speed > 0)
  (motorcycle_fastest : motorcycle_speed > baldwin_speed ∧ motorcycle_speed > clark_speed)

/-- Represents a valid journey plan -/
structure JourneyPlan where
  params : JourneyParams
  baldwin_pickup : ℝ
  clark_pickup : ℝ
  (valid_pickups : 0 ≤ baldwin_pickup ∧ baldwin_pickup ≤ params.total_distance ∧
                   0 ≤ clark_pickup ∧ clark_pickup ≤ params.total_distance)

/-- Calculates the total time for a given journey plan -/
def totalTime (plan : JourneyPlan) : ℝ :=
  sorry

/-- Theorem stating that there exists a journey plan where everyone arrives in 5 hours -/
theorem journey_possible (params : JourneyParams) 
  (h1 : params.total_distance = 52)
  (h2 : params.motorcycle_speed = 20)
  (h3 : params.baldwin_speed = 5)
  (h4 : params.clark_speed = 4) :
  ∃ (plan : JourneyPlan), totalTime plan = 5 :=
sorry

end journey_possible_l1549_154955


namespace total_tax_percentage_l1549_154992

/-- Calculates the total tax percentage given spending percentages and tax rates -/
theorem total_tax_percentage
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (clothing_tax_rate : ℝ)
  (food_tax_rate : ℝ)
  (other_tax_rate : ℝ)
  (h1 : clothing_percent = 0.5)
  (h2 : food_percent = 0.2)
  (h3 : other_percent = 0.3)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax_rate = 0.04)
  (h6 : food_tax_rate = 0)
  (h7 : other_tax_rate = 0.1) :
  clothing_percent * clothing_tax_rate +
  food_percent * food_tax_rate +
  other_percent * other_tax_rate = 0.05 := by
sorry


end total_tax_percentage_l1549_154992


namespace max_garden_area_l1549_154941

/-- The maximum area of a rectangular garden with one side along a wall and 400 feet of fencing for the other three sides. -/
theorem max_garden_area : 
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ l + 2*w = 400 ∧
  (∀ (l' w' : ℝ), l' > 0 → w' > 0 → l' + 2*w' = 400 → l'*w' ≤ l*w) ∧
  l*w = 20000 :=
by sorry

end max_garden_area_l1549_154941


namespace burger_distance_is_two_l1549_154909

/-- Represents the distance driven to various locations --/
structure Distances where
  school : ℕ
  softball : ℕ
  friend : ℕ
  home : ℕ

/-- Calculates the distance to the burger restaurant given the car's efficiency,
    initial gas, and distances driven to other locations --/
def distance_to_burger (efficiency : ℕ) (initial_gas : ℕ) (distances : Distances) : ℕ :=
  efficiency * initial_gas - (distances.school + distances.softball + distances.friend + distances.home)

/-- Theorem stating that the distance to the burger restaurant is 2 miles --/
theorem burger_distance_is_two :
  let efficiency := 19
  let initial_gas := 2
  let distances := Distances.mk 15 6 4 11
  distance_to_burger efficiency initial_gas distances = 2 := by
  sorry

#check burger_distance_is_two

end burger_distance_is_two_l1549_154909


namespace expression_simplification_l1549_154995

theorem expression_simplification (a b c x y z : ℝ) :
  (c * x * (b * x^2 + 3 * a^2 * y^2 + c^2 * z^2) + b * z * (b * x^2 + 3 * c^2 * x^2 + a^2 * y^2)) / (c * x + b * z) = 
  b * x^2 + a^2 * y^2 + c^2 * z^2 :=
by sorry

end expression_simplification_l1549_154995


namespace total_sugar_needed_l1549_154971

def sugar_for_frosting : ℝ := 0.6
def sugar_for_cake : ℝ := 0.2

theorem total_sugar_needed : sugar_for_frosting + sugar_for_cake = 0.8 := by
  sorry

end total_sugar_needed_l1549_154971


namespace diophantine_equation_solutions_count_l1549_154980

theorem diophantine_equation_solutions_count : 
  ∃ (S : Finset ℤ), 
    (∀ p ∈ S, 1 ≤ p ∧ p ≤ 15) ∧ 
    (∀ p ∈ S, ∃ q : ℤ, p * q - 8 * p - 3 * q = 15) ∧
    S.card = 4 := by
  sorry

end diophantine_equation_solutions_count_l1549_154980


namespace unique_solution_quadratic_l1549_154921

theorem unique_solution_quadratic (p : ℝ) (hp : p ≠ 0) :
  (∃! x : ℝ, p * x^2 - 10 * x + 2 = 0) ↔ p = 25/2 := by
  sorry

end unique_solution_quadratic_l1549_154921


namespace unique_solution_cube_root_equation_l1549_154938

theorem unique_solution_cube_root_equation :
  ∃! x : ℝ, x^(3/5) - 4 = 32 - x^(2/5) := by sorry

end unique_solution_cube_root_equation_l1549_154938


namespace arithmetic_problem_l1549_154983

theorem arithmetic_problem : 
  ((2 * 4 * 6) / (1 + 3 + 5 + 7) - (1 * 3 * 5) / (2 + 4 + 6)) / (1/2) = 3.5 := by
  sorry

end arithmetic_problem_l1549_154983


namespace merry_go_round_time_l1549_154972

theorem merry_go_round_time (dave chuck erica : ℝ) : 
  chuck = 5 * dave →
  erica = chuck + 0.3 * chuck →
  erica = 65 →
  dave = 10 :=
by sorry

end merry_go_round_time_l1549_154972


namespace area_of_curve_l1549_154973

/-- The curve defined by the equation x^2 + y^2 = 2(|x| + |y|) -/
def curve (x y : ℝ) : Prop := x^2 + y^2 = 2 * (abs x + abs y)

/-- The region enclosed by the curve -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ curve x y}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

theorem area_of_curve : area enclosed_region = 2 * Real.pi := by sorry

end area_of_curve_l1549_154973


namespace no_real_roots_of_quartic_equation_l1549_154957

theorem no_real_roots_of_quartic_equation :
  ∀ x : ℝ, 5 * x^4 - 28 * x^3 + 57 * x^2 - 28 * x + 5 ≠ 0 :=
by sorry

end no_real_roots_of_quartic_equation_l1549_154957


namespace pizza_toppings_combinations_l1549_154997

/-- The number of combinations of k items chosen from n items -/
def combination (n k : ℕ) : ℕ := Nat.choose n k

/-- There are 10 available toppings -/
def num_toppings : ℕ := 10

/-- We want to choose 3 toppings -/
def toppings_to_choose : ℕ := 3

theorem pizza_toppings_combinations :
  combination num_toppings toppings_to_choose = 120 := by
  sorry

end pizza_toppings_combinations_l1549_154997


namespace magical_red_knights_fraction_l1549_154953

theorem magical_red_knights_fraction (total : ℚ) (red : ℚ) (blue : ℚ) (magical : ℚ) 
  (h1 : red = 3 / 8 * total)
  (h2 : blue = total - red)
  (h3 : magical = 1 / 4 * total)
  (h4 : ∃ (x y : ℚ), x / y > 0 ∧ red * (x / y) = 3 * (blue * (x / (3 * y))) ∧ red * (x / y) + blue * (x / (3 * y)) = magical) :
  ∃ (x y : ℚ), x / y = 3 / 7 ∧ red * (x / y) = magical := by
  sorry

end magical_red_knights_fraction_l1549_154953


namespace smallest_set_size_l1549_154998

theorem smallest_set_size (n : ℕ) (hn : n > 0) :
  let S := {S : Finset ℕ | S ⊆ Finset.range n ∧
    ∀ β : ℝ, β > 0 → (∀ s ∈ S, ∃ m : ℕ, s = ⌊β * m⌋) →
      ∀ k ∈ Finset.range n, ∃ m : ℕ, k = ⌊β * m⌋}
  ∃ S₀ ∈ S, S₀.card = n / 2 + 1 ∧ ∀ S' ∈ S, S'.card ≥ S₀.card :=
sorry

end smallest_set_size_l1549_154998


namespace short_trees_planted_l1549_154990

theorem short_trees_planted (initial_short : ℕ) (final_short : ℕ) :
  initial_short = 31 →
  final_short = 95 →
  final_short - initial_short = 64 := by
sorry

end short_trees_planted_l1549_154990


namespace floor_sqrt_50_squared_l1549_154986

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end floor_sqrt_50_squared_l1549_154986


namespace no_functions_satisfying_conditions_l1549_154996

theorem no_functions_satisfying_conditions :
  ¬ (∃ (f g : ℝ → ℝ), 
    (∀ (x y : ℝ), f (x^2 + g y) - f (x^2) + g y - g x ≤ 2 * y) ∧ 
    (∀ (x : ℝ), f x ≥ x^2)) := by
  sorry

end no_functions_satisfying_conditions_l1549_154996


namespace scorpion_daily_segments_l1549_154952

/-- The number of body segments a cave scorpion needs to eat daily -/
def daily_segments : ℕ :=
  let segments_first_millipede := 60
  let segments_long_millipede := 2 * segments_first_millipede
  let segments_eaten := segments_first_millipede + 2 * segments_long_millipede
  let segments_to_eat := 10 * 50
  segments_eaten + segments_to_eat

theorem scorpion_daily_segments : daily_segments = 800 := by
  sorry

end scorpion_daily_segments_l1549_154952


namespace slower_train_speed_l1549_154914

/-- Proves that the speed of the slower train is 36 km/hr given the specified conditions -/
theorem slower_train_speed 
  (train_length : ℝ) 
  (faster_train_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 25) 
  (h2 : faster_train_speed = 46) 
  (h3 : passing_time = 18) : 
  ∃ (slower_train_speed : ℝ), 
    slower_train_speed = 36 ∧ 
    (faster_train_speed - slower_train_speed) * (5 / 18) * passing_time = 2 * train_length :=
by sorry

end slower_train_speed_l1549_154914


namespace total_production_8_minutes_l1549_154908

/-- Represents a machine type in the factory -/
inductive MachineType
| A
| B
| C

/-- Represents the state of the factory at a given time -/
structure FactoryState where
  machineCount : MachineType → ℕ
  productionRate : MachineType → ℕ

/-- Calculates the total production for a given time interval -/
def totalProduction (state : FactoryState) (minutes : ℕ) : ℕ :=
  (state.machineCount MachineType.A * state.productionRate MachineType.A +
   state.machineCount MachineType.B * state.productionRate MachineType.B +
   state.machineCount MachineType.C * state.productionRate MachineType.C) * minutes

/-- The initial state of the factory -/
def initialState : FactoryState := {
  machineCount := λ t => match t with
    | MachineType.A => 6
    | MachineType.B => 4
    | MachineType.C => 3
  productionRate := λ t => match t with
    | MachineType.A => 270
    | MachineType.B => 200
    | MachineType.C => 150
}

/-- The state of the factory after 2 minutes -/
def stateAfter2Min : FactoryState := {
  machineCount := λ t => match t with
    | MachineType.A => 4
    | MachineType.B => 7
    | MachineType.C => 3
  productionRate := λ t => match t with
    | MachineType.A => 270
    | MachineType.B => 200
    | MachineType.C => 150
}

/-- The state of the factory after 4 minutes -/
def stateAfter4Min : FactoryState := {
  machineCount := λ t => match t with
    | MachineType.A => 6
    | MachineType.B => 9
    | MachineType.C => 3
  productionRate := λ t => match t with
    | MachineType.A => 300
    | MachineType.B => 180
    | MachineType.C => 170
}

/-- Theorem stating the total production over 8 minutes -/
theorem total_production_8_minutes :
  totalProduction initialState 2 +
  totalProduction stateAfter2Min 2 +
  totalProduction stateAfter4Min 4 = 27080 := by
  sorry


end total_production_8_minutes_l1549_154908


namespace scenario_equivalence_l1549_154950

/-- Represents the cost of trees in yuan -/
structure TreeCost where
  pine : ℝ
  cypress : ℝ

/-- Represents the given scenario for tree costs -/
def scenario (cost : TreeCost) : Prop :=
  2 * cost.pine + 3 * cost.cypress = 120 ∧
  2 * cost.pine - cost.cypress = 20

/-- The correct system of equations for the scenario -/
def correct_system (x y : ℝ) : Prop :=
  2 * x + 3 * y = 120 ∧
  2 * x - y = 20

/-- Theorem stating that the correct system accurately represents the scenario -/
theorem scenario_equivalence :
  ∀ (cost : TreeCost), scenario cost ↔ correct_system cost.pine cost.cypress :=
by sorry

end scenario_equivalence_l1549_154950


namespace parallelogram_zk_product_l1549_154951

structure Parallelogram where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ

def is_valid_parallelogram (p : Parallelogram) (z k : ℝ) : Prop :=
  p.EF = 5 * z + 5 ∧
  p.FG = 4 * k^2 ∧
  p.GH = 40 ∧
  p.HE = k + 20

theorem parallelogram_zk_product (p : Parallelogram) (z k : ℝ) :
  is_valid_parallelogram p z k → z * k = (7 + 7 * Real.sqrt 321) / 8 := by
  sorry

end parallelogram_zk_product_l1549_154951


namespace fish_total_weight_l1549_154994

/-- The weight of a fish with specific weight relationships between its parts -/
def fish_weight (head body tail : ℝ) : Prop :=
  tail = 1 ∧ 
  head = tail + body / 2 ∧ 
  body = head + tail

theorem fish_total_weight : 
  ∀ (head body tail : ℝ), 
  fish_weight head body tail → 
  head + body + tail = 8 := by
  sorry

end fish_total_weight_l1549_154994


namespace exponent_division_l1549_154964

theorem exponent_division (a : ℝ) : a^3 / a = a^2 := by
  sorry

end exponent_division_l1549_154964


namespace intersection_implies_y_zero_l1549_154939

theorem intersection_implies_y_zero (x y : ℝ) : 
  let A : Set ℝ := {2, Real.log x}
  let B : Set ℝ := {x, y}
  A ∩ B = {0} → y = 0 := by
  sorry

end intersection_implies_y_zero_l1549_154939


namespace hyperbola_asymptotes_l1549_154987

/-- The asymptotes of the hyperbola x^2 - y^2/3 = 1 are y = ±√3 x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2/3 = 1
  ∃ (k : ℝ), k^2 = 3 ∧
    (∀ x y, h x y → (y = k*x ∨ y = -k*x))
    ∧ (∀ ε > 0, ∃ δ > 0, ∀ x y, h x y → (|x| > δ → min (|y - k*x|) (|y + k*x|) < ε * |x|)) :=
by sorry

end hyperbola_asymptotes_l1549_154987


namespace binary_addition_theorem_l1549_154940

/-- Represents a binary number as a list of bits (0 or 1) in little-endian order -/
def BinaryNumber := List Bool

/-- Converts a decimal number to its binary representation -/
def decimalToBinary (n : Int) : BinaryNumber :=
  sorry

/-- Converts a binary number to its decimal representation -/
def binaryToDecimal (b : BinaryNumber) : Int :=
  sorry

/-- Adds two binary numbers -/
def addBinary (a b : BinaryNumber) : BinaryNumber :=
  sorry

/-- Negates a binary number (two's complement) -/
def negateBinary (b : BinaryNumber) : BinaryNumber :=
  sorry

theorem binary_addition_theorem :
  let b1 := decimalToBinary 13  -- 1101₂
  let b2 := decimalToBinary 10  -- 1010₂
  let b3 := decimalToBinary 7   -- 111₂
  let b4 := negateBinary (decimalToBinary 11)  -- -1011₂
  let sum := addBinary b1 (addBinary b2 (addBinary b3 b4))
  binaryToDecimal sum = 35  -- 100011₂
  := by sorry

end binary_addition_theorem_l1549_154940


namespace balanced_numbers_count_l1549_154928

/-- A four-digit number abcd is balanced if a + b = c + d -/
def is_balanced (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a + b = c + d

/-- Count of balanced four-digit numbers with sum 8 -/
def balanced_sum_8_count : ℕ := 72

/-- Count of balanced four-digit numbers with sum 16 -/
def balanced_sum_16_count : ℕ := 9

/-- Total count of balanced four-digit numbers -/
def total_balanced_count : ℕ := 615

/-- Theorem stating the counts of balanced numbers -/
theorem balanced_numbers_count :
  (balanced_sum_8_count = 72) ∧
  (balanced_sum_16_count = 9) ∧
  (total_balanced_count = 615) :=
sorry

end balanced_numbers_count_l1549_154928


namespace geometric_series_problem_l1549_154934

theorem geometric_series_problem (q : ℝ) (b₁ : ℝ) (h₁ : |q| < 1) 
  (h₂ : b₁ / (1 - q) = 16) (h₃ : b₁^2 / (1 - q^2) = 153.6) :
  b₁ * q^3 = 3/16 ∧ q = 1/4 := by
  sorry

end geometric_series_problem_l1549_154934


namespace problem_solution_l1549_154974

theorem problem_solution (a b : ℚ) 
  (h1 : a + b = 8/15) 
  (h2 : a - b = 2/15) : 
  a^2 - b^2 = 16/225 ∧ a * b = 1/25 := by
sorry

end problem_solution_l1549_154974


namespace all_terms_perfect_squares_l1549_154918

/-- A sequence of integers satisfying specific conditions -/
def SpecialSequence (a : ℕ → ℤ) : Prop :=
  (∀ n ≥ 2, a (n + 1) = 3 * a n - 3 * a (n - 1) + a (n - 2)) ∧
  (2 * a 1 = a 0 + a 2 - 2) ∧
  (∀ m : ℕ, ∃ k : ℕ, ∀ i < m, ∃ j : ℤ, a (k + i) = j ^ 2)

/-- All terms in the special sequence are perfect squares -/
theorem all_terms_perfect_squares (a : ℕ → ℤ) (h : SpecialSequence a) :
  ∀ n : ℕ, ∃ k : ℤ, a n = k ^ 2 :=
by sorry

end all_terms_perfect_squares_l1549_154918


namespace total_boys_in_assembly_l1549_154919

/-- Represents the assembly of boys in two rows --/
structure Assembly where
  first_row : ℕ
  second_row : ℕ

/-- The position of a boy in a row --/
structure Position where
  from_left : ℕ
  from_right : ℕ

/-- Represents the assembly with given conditions --/
def school_assembly : Assembly where
  first_row := 24
  second_row := 24

/-- Rajan's position in the first row --/
def rajan_position : Position where
  from_left := 6
  from_right := school_assembly.first_row - 5

/-- Vinay's position in the first row --/
def vinay_position : Position where
  from_left := school_assembly.first_row - 9
  from_right := 10

/-- Number of boys between Rajan and Vinay --/
def boys_between : ℕ := 8

/-- Suresh's position in the second row --/
def suresh_position : Position where
  from_left := 5
  from_right := school_assembly.second_row - 4

theorem total_boys_in_assembly :
  school_assembly.first_row + school_assembly.second_row = 48 ∧
  school_assembly.first_row = school_assembly.second_row ∧
  rajan_position.from_left = 6 ∧
  vinay_position.from_right = 10 ∧
  vinay_position.from_left - rajan_position.from_left - 1 = boys_between ∧
  suresh_position.from_left = 5 :=
by sorry

end total_boys_in_assembly_l1549_154919


namespace minimum_value_of_reciprocal_sum_l1549_154903

theorem minimum_value_of_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a : ℝ × ℝ := (m, 1 - n)
  let b : ℝ × ℝ := (1, 2)
  (∃ (k : ℝ), a = k • b) →
  (∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y ≥ 1/m + 1/n) →
  1/m + 1/n = 3 + 2 * Real.sqrt 2 := by
  sorry

end minimum_value_of_reciprocal_sum_l1549_154903


namespace largest_number_of_piles_l1549_154905

theorem largest_number_of_piles (apples : Nat) (apricots : Nat) (cherries : Nat)
  (h1 : apples = 42)
  (h2 : apricots = 60)
  (h3 : cherries = 90) :
  Nat.gcd apples (Nat.gcd apricots cherries) = 6 := by
  sorry

end largest_number_of_piles_l1549_154905


namespace power_two_greater_than_square_l1549_154968

theorem power_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end power_two_greater_than_square_l1549_154968


namespace batsman_average_increase_l1549_154924

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalScore : Nat
  lastInningsScore : Nat

/-- Calculates the average score of a batsman -/
def average (b : Batsman) : ℚ :=
  b.totalScore / b.innings

/-- The increase in average for a batsman after their last innings -/
def averageIncrease (b : Batsman) : ℚ :=
  average b - average { b with
    innings := b.innings - 1
    totalScore := b.totalScore - b.lastInningsScore
  }

theorem batsman_average_increase :
  ∀ b : Batsman,
    b.innings = 12 ∧
    b.lastInningsScore = 70 ∧
    average b = 37 →
    averageIncrease b = 3 := by
  sorry

end batsman_average_increase_l1549_154924


namespace basil_cookie_boxes_l1549_154916

/-- The number of cookies Basil gets in the morning and before bed -/
def morning_night_cookies : ℚ := 1/2 + 1/2

/-- The number of whole cookies Basil gets during the day -/
def day_cookies : ℕ := 2

/-- The number of cookies per box -/
def cookies_per_box : ℕ := 45

/-- The number of days Basil needs cookies for -/
def days : ℕ := 30

/-- Theorem stating the number of boxes Basil needs for 30 days -/
theorem basil_cookie_boxes : 
  ⌈(days * (morning_night_cookies + day_cookies)) / cookies_per_box⌉ = 2 := by
  sorry

end basil_cookie_boxes_l1549_154916


namespace odd_even_function_problem_l1549_154977

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem odd_even_function_problem (f g : ℝ → ℝ) 
  (h_odd : IsOdd f) (h_even : IsEven g)
  (h1 : f (-3) + g 3 = 2) (h2 : f 3 + g (-3) = 4) : 
  g 3 = 3 := by
  sorry

end odd_even_function_problem_l1549_154977


namespace parallelogram_fourth_vertex_l1549_154970

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if four points form a parallelogram -/
def is_parallelogram (p : Parallelogram) : Prop :=
  (p.A.x + p.C.x = p.B.x + p.D.x) ∧ (p.A.y + p.C.y = p.B.y + p.D.y)

theorem parallelogram_fourth_vertex :
  ∀ (p : Parallelogram),
  p.A = Point.mk (-2) 1 →
  p.B = Point.mk (-1) 3 →
  p.C = Point.mk 3 4 →
  is_parallelogram p →
  (p.D = Point.mk 2 2 ∨ p.D = Point.mk (-6) 0 ∨ p.D = Point.mk 4 6) :=
by sorry

end parallelogram_fourth_vertex_l1549_154970


namespace root_exists_in_interval_l1549_154925

theorem root_exists_in_interval : ∃ x : ℝ, x ∈ Set.Ioo (-2) (-1) ∧ 2^x - x - 2 = 0 := by
  sorry

end root_exists_in_interval_l1549_154925


namespace rational_power_difference_integer_implies_integer_l1549_154913

theorem rational_power_difference_integer_implies_integer 
  (a b : ℚ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_distinct : a ≠ b) 
  (h_inf_int : ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → ∃ (k : ℤ), k > 0 ∧ a^n - b^n = k) :
  ∃ (m n : ℕ), (m : ℚ) = a ∧ (n : ℚ) = b :=
sorry

end rational_power_difference_integer_implies_integer_l1549_154913


namespace sum_of_squares_inequality_l1549_154947

theorem sum_of_squares_inequality (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (2 + a) * (2 + b) ≥ c * d := by
  sorry

end sum_of_squares_inequality_l1549_154947


namespace basketball_team_starters_count_l1549_154930

theorem basketball_team_starters_count :
  let total_players : ℕ := 16
  let quadruplets : ℕ := 4
  let starters : ℕ := 7
  let quadruplets_in_lineup : ℕ := 2
  let remaining_players : ℕ := total_players - quadruplets
  let remaining_starters : ℕ := starters - quadruplets_in_lineup

  (Nat.choose quadruplets quadruplets_in_lineup) *
  (Nat.choose remaining_players remaining_starters) = 4752 :=
by sorry

end basketball_team_starters_count_l1549_154930


namespace ajay_ride_distance_l1549_154915

/-- Ajay's riding speed in km/hour -/
def riding_speed : ℝ := 50

/-- Time taken for the ride in hours -/
def ride_time : ℝ := 18

/-- The distance Ajay can ride in the given time -/
def ride_distance : ℝ := riding_speed * ride_time

theorem ajay_ride_distance : ride_distance = 900 := by
  sorry

end ajay_ride_distance_l1549_154915


namespace diamond_720_1001_cubed_l1549_154944

/-- The diamond operation on positive integers -/
def diamond (a b : ℕ+) : ℕ := sorry

/-- The number of distinct prime factors of a positive integer -/
def num_distinct_prime_factors (n : ℕ+) : ℕ := sorry

theorem diamond_720_1001_cubed : 
  (diamond 720 1001)^3 = 216 := by sorry

end diamond_720_1001_cubed_l1549_154944


namespace harry_earnings_theorem_l1549_154923

/-- Harry's weekly dog-walking earnings -/
def harry_weekly_earnings : ℕ :=
  let monday_wednesday_friday_dogs := 7
  let tuesday_dogs := 12
  let thursday_dogs := 9
  let pay_per_dog := 5
  let days_with_7_dogs := 3
  
  (monday_wednesday_friday_dogs * days_with_7_dogs + tuesday_dogs + thursday_dogs) * pay_per_dog

/-- Theorem stating Harry's weekly earnings -/
theorem harry_earnings_theorem : harry_weekly_earnings = 210 := by
  sorry

end harry_earnings_theorem_l1549_154923


namespace plot_length_is_60_l1549_154975

/-- Represents a rectangular plot with given properties --/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  cost_per_meter : ℝ
  total_cost : ℝ
  length_breadth_relation : length = breadth + 20
  perimeter_cost_relation : 2 * (length + breadth) = total_cost / cost_per_meter

/-- The length of the plot is 60 meters given the specified conditions --/
theorem plot_length_is_60 (plot : RectangularPlot) 
    (h1 : plot.cost_per_meter = 26.5)
    (h2 : plot.total_cost = 5300) : 
  plot.length = 60 := by
  sorry

end plot_length_is_60_l1549_154975


namespace no_subdivision_for_1986_plots_l1549_154979

theorem no_subdivision_for_1986_plots : ¬ ∃ (n : ℕ), 8 * n + 9 = 1986 := by
  sorry

end no_subdivision_for_1986_plots_l1549_154979


namespace circle_radius_proof_l1549_154961

theorem circle_radius_proof (r : ℝ) (x y : ℝ) : 
  x = π * r^2 →
  y = 2 * π * r - 6 →
  x + y = 94 * π →
  r = 10 := by
sorry

end circle_radius_proof_l1549_154961


namespace periodic_sequence_properties_l1549_154993

/-- A periodic sequence with period T -/
def is_periodic (a : ℕ → ℕ) (T : ℕ) : Prop :=
  ∀ n, a (n + T) = a n

/-- The smallest period of a sequence -/
def smallest_period (a : ℕ → ℕ) (t : ℕ) : Prop :=
  is_periodic a t ∧ ∀ s, is_periodic a s → t ≤ s

theorem periodic_sequence_properties {a : ℕ → ℕ} {T : ℕ} (h : is_periodic a T) :
  (∃ t, smallest_period a t) ∧ (∀ t, smallest_period a t → T % t = 0) := by
  sorry

end periodic_sequence_properties_l1549_154993


namespace english_test_percentage_l1549_154954

theorem english_test_percentage (math_questions : ℕ) (english_questions : ℕ) 
  (math_percentage : ℚ) (total_correct : ℕ) : 
  math_questions = 40 →
  english_questions = 50 →
  math_percentage = 3/4 →
  total_correct = 79 →
  (total_correct - (math_percentage * math_questions).num) / english_questions = 49/50 := by
sorry

end english_test_percentage_l1549_154954


namespace fraction_of_third_is_eighth_l1549_154949

theorem fraction_of_third_is_eighth (x : ℚ) : x * (1/3 : ℚ) = 1/8 → x = 3/8 := by
  sorry

end fraction_of_third_is_eighth_l1549_154949


namespace average_age_of_contestants_l1549_154912

/-- Represents an age in years and months -/
structure Age where
  years : ℕ
  months : ℕ
  valid : months < 12

/-- Converts an age to total months -/
def ageToMonths (a : Age) : ℕ := a.years * 12 + a.months

/-- Converts total months to an age -/
def monthsToAge (m : ℕ) : Age :=
  { years := m / 12
  , months := m % 12
  , valid := by exact Nat.mod_lt m (by norm_num) }

/-- Calculates the average age of three contestants -/
def averageAge (a1 a2 a3 : Age) : Age :=
  monthsToAge ((ageToMonths a1 + ageToMonths a2 + ageToMonths a3) / 3)

theorem average_age_of_contestants :
  let age1 : Age := { years := 15, months := 9, valid := by norm_num }
  let age2 : Age := { years := 16, months := 1, valid := by norm_num }
  let age3 : Age := { years := 15, months := 8, valid := by norm_num }
  let avgAge := averageAge age1 age2 age3
  avgAge.years = 15 ∧ avgAge.months = 10 := by
  sorry

end average_age_of_contestants_l1549_154912


namespace sin_40_tan_10_minus_sqrt_3_l1549_154991

theorem sin_40_tan_10_minus_sqrt_3 : 
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end sin_40_tan_10_minus_sqrt_3_l1549_154991


namespace camp_children_count_l1549_154985

/-- The initial number of children in the camp -/
def initial_children : ℕ := 50

/-- The fraction of boys in the initial group -/
def boys_fraction : ℚ := 4/5

/-- The number of boys added -/
def boys_added : ℕ := 50

/-- The fraction of girls in the final group -/
def final_girls_fraction : ℚ := 1/10

theorem camp_children_count :
  (initial_children : ℚ) * (1 - boys_fraction) = 
    final_girls_fraction * (initial_children + boys_added) := by
  sorry

end camp_children_count_l1549_154985


namespace max_a_for_increasing_f_l1549_154965

def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

theorem max_a_for_increasing_f :
  (∃ a : ℝ, ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ ∧ x₂ ≤ a → f x₁ ≤ f x₂) ∧
  (∀ b : ℝ, b > 1 → ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ b ∧ f x₁ > f x₂) :=
by sorry

end max_a_for_increasing_f_l1549_154965


namespace museum_ticket_price_l1549_154920

theorem museum_ticket_price (group_size : ℕ) (total_with_tax : ℚ) (tax_rate : ℚ) :
  group_size = 25 →
  total_with_tax = 945 →
  tax_rate = 5 / 100 →
  ∃ (ticket_price : ℚ),
    ticket_price * group_size * (1 + tax_rate) = total_with_tax ∧
    ticket_price = 36 :=
by sorry

end museum_ticket_price_l1549_154920


namespace new_person_weight_l1549_154969

/-- Given a group of 8 people, where one person weighing 70 kg is replaced by a new person,
    causing the average weight to increase by 2.5 kg, 
    prove that the weight of the new person is 90 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 70 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 90 := by
  sorry

end new_person_weight_l1549_154969


namespace equation_solution_exists_l1549_154962

theorem equation_solution_exists (a : ℝ) : 
  (∃ x : ℝ, 4^x - a * 2^x - a + 3 = 0) ↔ a ≥ 2 := by
  sorry

end equation_solution_exists_l1549_154962


namespace job_filling_combinations_l1549_154967

def num_resumes : ℕ := 30
def num_unsuitable : ℕ := 20
def num_job_openings : ℕ := 5

theorem job_filling_combinations :
  (num_resumes - num_unsuitable).factorial / (num_resumes - num_unsuitable - num_job_openings).factorial = 30240 :=
by sorry

end job_filling_combinations_l1549_154967


namespace sigma_phi_inequality_l1549_154946

open Nat

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- A natural number is prime if it has exactly two divisors -/
def isPrime (n : ℕ) : Prop := sorry

theorem sigma_phi_inequality (n : ℕ) (h : n > 1) :
  sigma n * phi n ≤ n^2 - 1 ∧ (sigma n * phi n = n^2 - 1 ↔ isPrime n) := by sorry

end sigma_phi_inequality_l1549_154946


namespace sum_of_constants_l1549_154922

/-- Given constants a, b, and c satisfying the specified conditions, prove that a + 2b + 3c = 64 -/
theorem sum_of_constants (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ x < -4 ∨ |x - 25| ≤ 1)
  (h2 : a < b) : 
  a + 2*b + 3*c = 64 := by
  sorry

end sum_of_constants_l1549_154922


namespace logarithm_sum_equality_l1549_154907

-- Define the logarithm function (base 2)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem logarithm_sum_equality (a b : ℝ) 
  (h1 : a > 1) (h2 : b > 1) (h3 : lg (a + b) = lg a + lg b) : 
  lg (a - 1) + lg (b - 1) = 0 := by
  sorry

end logarithm_sum_equality_l1549_154907


namespace opposite_of_one_half_l1549_154999

theorem opposite_of_one_half : -(1 / 2 : ℚ) = -1 / 2 := by
  sorry

end opposite_of_one_half_l1549_154999


namespace wage_problem_l1549_154958

/-- Given a sum of money S that can pay q's wages for 40 days and both p and q's wages for 15 days,
    prove that S can pay p's wages for 24 days. -/
theorem wage_problem (S P Q : ℝ) (hS_positive : S > 0) (hP_positive : P > 0) (hQ_positive : Q > 0)
  (hS_q : S = 40 * Q) (hS_pq : S = 15 * (P + Q)) :
  S = 24 * P := by
  sorry

end wage_problem_l1549_154958


namespace crew_size_proof_l1549_154931

/-- The number of laborers present on a certain day -/
def present_laborers : ℕ := 10

/-- The percentage of laborers that showed up for work (as a rational number) -/
def attendance_percentage : ℚ := 385 / 1000

/-- The total number of laborers in the crew -/
def total_laborers : ℕ := 26

theorem crew_size_proof :
  (present_laborers : ℚ) / attendance_percentage = total_laborers := by
  sorry

end crew_size_proof_l1549_154931


namespace sales_solution_l1549_154900

def sales_problem (month1 month3 month4 month5 month6 average_sale : ℕ) : Prop :=
  ∃ (month2 : ℕ),
    (month1 + month2 + month3 + month4 + month5 + month6) / 6 = average_sale ∧
    month2 = 6927

theorem sales_solution :
  sales_problem 6435 6855 7230 6562 7991 7000 :=
sorry

end sales_solution_l1549_154900


namespace max_area_PQR_max_area_incenters_l1549_154932

-- Define the equilateral triangle ABC with unit area
def triangle_ABC : Set (ℝ × ℝ) := sorry

-- Define the external equilateral triangles
def triangle_APB : Set (ℝ × ℝ) := sorry
def triangle_BQC : Set (ℝ × ℝ) := sorry
def triangle_CRA : Set (ℝ × ℝ) := sorry

-- Define the angles
def angle_APB : ℝ := 60
def angle_BQC : ℝ := 60
def angle_CRA : ℝ := 60

-- Define the points P, Q, R
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry
def R : ℝ × ℝ := sorry

-- Define the incenters
def incenter_APB : ℝ × ℝ := sorry
def incenter_BQC : ℝ × ℝ := sorry
def incenter_CRA : ℝ × ℝ := sorry

-- Define the area function
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem for the maximum area of triangle PQR
theorem max_area_PQR :
  ∀ P Q R,
    P ∈ triangle_APB ∧ Q ∈ triangle_BQC ∧ R ∈ triangle_CRA →
    area {P, Q, R} ≤ 4 * Real.sqrt 3 :=
sorry

-- Theorem for the maximum area of triangle formed by incenters
theorem max_area_incenters :
  area {incenter_APB, incenter_BQC, incenter_CRA} ≤ 1 :=
sorry

end max_area_PQR_max_area_incenters_l1549_154932


namespace class_size_roses_class_size_l1549_154942

theorem class_size (girls_present : ℕ) (boys_absent : ℕ) : ℕ :=
  let boys_present := girls_present / 2
  let total_boys := boys_present + boys_absent
  let total_students := girls_present + total_boys
  total_students

theorem roses_class_size : class_size 140 40 = 250 := by
  sorry

end class_size_roses_class_size_l1549_154942


namespace exchange_indifference_l1549_154984

/-- Represents the number of rubles a tourist plans to exchange. -/
def rubles : ℕ := 140

/-- Represents the exchange rate (in tugriks) for the first office. -/
def rate1 : ℕ := 3000

/-- Represents the exchange rate (in tugriks) for the second office. -/
def rate2 : ℕ := 2950

/-- Represents the commission fee (in tugriks) for the first office. -/
def commission : ℕ := 7000

theorem exchange_indifference :
  rate1 * rubles - commission = rate2 * rubles :=
by sorry

end exchange_indifference_l1549_154984
