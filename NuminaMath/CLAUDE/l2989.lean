import Mathlib

namespace cookie_revenue_l2989_298966

/-- Calculates the total revenue from selling chocolate and vanilla cookies -/
theorem cookie_revenue (chocolate_count : ℕ) (vanilla_count : ℕ) 
  (chocolate_price : ℚ) (vanilla_price : ℚ) : 
  chocolate_count * chocolate_price + vanilla_count * vanilla_price = 360 :=
by
  -- Assuming chocolate_count = 220, vanilla_count = 70, 
  -- chocolate_price = 1, and vanilla_price = 2
  have h1 : chocolate_count = 220 := by sorry
  have h2 : vanilla_count = 70 := by sorry
  have h3 : chocolate_price = 1 := by sorry
  have h4 : vanilla_price = 2 := by sorry
  
  -- The proof would go here
  sorry

end cookie_revenue_l2989_298966


namespace rachel_reading_homework_l2989_298902

/-- The number of pages of math homework Rachel had to complete -/
def math_homework_pages : ℕ := 8

/-- The additional pages of reading homework compared to math homework -/
def additional_reading_pages : ℕ := 6

/-- The total number of pages of reading homework Rachel had to complete -/
def reading_homework_pages : ℕ := math_homework_pages + additional_reading_pages

theorem rachel_reading_homework : reading_homework_pages = 14 := by
  sorry

end rachel_reading_homework_l2989_298902


namespace workshop_production_theorem_l2989_298939

/-- Represents the factory workshop setup and production requirements -/
structure Workshop where
  total_workers : ℕ
  type_a_production : ℕ
  type_b_production : ℕ
  type_a_required : ℕ
  type_b_required : ℕ
  type_a_cost : ℕ
  type_b_cost : ℕ

/-- Calculates the number of workers assigned to type A parts -/
def workers_for_type_a (w : Workshop) : ℕ :=
  sorry

/-- Calculates the total processing cost for all workers in one day -/
def total_processing_cost (w : Workshop) : ℕ :=
  sorry

/-- The main theorem stating the correct number of workers for type A and total cost -/
theorem workshop_production_theorem (w : Workshop) 
  (h1 : w.total_workers = 50)
  (h2 : w.type_a_production = 30)
  (h3 : w.type_b_production = 20)
  (h4 : w.type_a_required = 7)
  (h5 : w.type_b_required = 2)
  (h6 : w.type_a_cost = 10)
  (h7 : w.type_b_cost = 12) :
  workers_for_type_a w = 35 ∧ total_processing_cost w = 14100 :=
by sorry

end workshop_production_theorem_l2989_298939


namespace max_towns_is_four_l2989_298915

/-- Represents the type of link between two towns -/
inductive LinkType
| Air
| Bus
| Train

/-- Represents a town -/
structure Town where
  id : Nat

/-- Represents a link between two towns -/
structure Link where
  town1 : Town
  town2 : Town
  linkType : LinkType

/-- A network of towns and their connections -/
structure TownNetwork where
  towns : List Town
  links : List Link

/-- Checks if a given network satisfies all the required conditions -/
def isValidNetwork (network : TownNetwork) : Prop :=
  -- Each pair of towns is linked by exactly one type of link
  (∀ t1 t2 : Town, t1 ∈ network.towns → t2 ∈ network.towns → t1 ≠ t2 →
    ∃! link : Link, link ∈ network.links ∧ 
    ((link.town1 = t1 ∧ link.town2 = t2) ∨ (link.town1 = t2 ∧ link.town2 = t1))) ∧
  -- At least one pair is linked by each type
  (∃ link : Link, link ∈ network.links ∧ link.linkType = LinkType.Air) ∧
  (∃ link : Link, link ∈ network.links ∧ link.linkType = LinkType.Bus) ∧
  (∃ link : Link, link ∈ network.links ∧ link.linkType = LinkType.Train) ∧
  -- No town has all three types of links
  (∀ t : Town, t ∈ network.towns →
    ¬(∃ l1 l2 l3 : Link, l1 ∈ network.links ∧ l2 ∈ network.links ∧ l3 ∈ network.links ∧
      (l1.town1 = t ∨ l1.town2 = t) ∧ (l2.town1 = t ∨ l2.town2 = t) ∧ (l3.town1 = t ∨ l3.town2 = t) ∧
      l1.linkType = LinkType.Air ∧ l2.linkType = LinkType.Bus ∧ l3.linkType = LinkType.Train)) ∧
  -- No three towns form a triangle with all sides of the same type
  (∀ t1 t2 t3 : Town, t1 ∈ network.towns → t2 ∈ network.towns → t3 ∈ network.towns →
    t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 →
    ¬(∃ l1 l2 l3 : Link, l1 ∈ network.links ∧ l2 ∈ network.links ∧ l3 ∈ network.links ∧
      ((l1.town1 = t1 ∧ l1.town2 = t2) ∨ (l1.town1 = t2 ∧ l1.town2 = t1)) ∧
      ((l2.town1 = t2 ∧ l2.town2 = t3) ∨ (l2.town1 = t3 ∧ l2.town2 = t2)) ∧
      ((l3.town1 = t3 ∧ l3.town2 = t1) ∨ (l3.town1 = t1 ∧ l3.town2 = t3)) ∧
      l1.linkType = l2.linkType ∧ l2.linkType = l3.linkType))

/-- The theorem stating that the maximum number of towns in a valid network is 4 -/
theorem max_towns_is_four :
  (∃ network : TownNetwork, isValidNetwork network ∧ network.towns.length = 4) ∧
  (∀ network : TownNetwork, isValidNetwork network → network.towns.length ≤ 4) :=
sorry

end max_towns_is_four_l2989_298915


namespace balloons_left_after_distribution_l2989_298924

def red_balloons : ℕ := 23
def blue_balloons : ℕ := 39
def green_balloons : ℕ := 71
def yellow_balloons : ℕ := 89
def num_friends : ℕ := 10

theorem balloons_left_after_distribution :
  (red_balloons + blue_balloons + green_balloons + yellow_balloons) % num_friends = 2 :=
by sorry

end balloons_left_after_distribution_l2989_298924


namespace product_from_lcm_gcd_l2989_298908

theorem product_from_lcm_gcd : 
  ∀ (a b : ℕ+), 
    Nat.lcm a b = 72 → 
    Nat.gcd a b = 8 → 
    (a : ℕ) * b = 576 := by
  sorry

end product_from_lcm_gcd_l2989_298908


namespace initial_balance_was_200_l2989_298961

/-- Represents the balance of Yasmin's bank account throughout the week --/
structure BankAccount where
  initial : ℝ
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ

/-- Calculates the final balance of Yasmin's account after all transactions --/
def finalBalance (account : BankAccount) : ℝ :=
  account.thursday

/-- Theorem stating that the initial balance was $200 --/
theorem initial_balance_was_200 (account : BankAccount) :
  account.initial = 200 ∧
  account.monday = account.initial / 2 ∧
  account.tuesday = account.monday + 30 ∧
  account.wednesday = 200 ∧
  account.thursday = account.wednesday - 20 ∧
  finalBalance account = 160 :=
by sorry

end initial_balance_was_200_l2989_298961


namespace profit_percentage_previous_year_l2989_298920

theorem profit_percentage_previous_year 
  (revenue_prev : ℝ) 
  (profit_prev : ℝ) 
  (revenue_1999 : ℝ) 
  (profit_1999 : ℝ) 
  (h1 : revenue_1999 = 0.7 * revenue_prev) 
  (h2 : profit_1999 = 0.15 * revenue_1999) 
  (h3 : profit_1999 = 1.0499999999999999 * profit_prev) : 
  profit_prev / revenue_prev = 0.1 := by
sorry

end profit_percentage_previous_year_l2989_298920


namespace intersection_of_A_and_B_l2989_298975

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B : Set ℝ := {x | x^2 + x - 2 > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end intersection_of_A_and_B_l2989_298975


namespace race_permutations_eq_24_l2989_298974

/-- The number of different possible orders for a race with 4 distinct participants and no ties -/
def race_permutations : ℕ := 24

/-- The number of participants in the race -/
def num_participants : ℕ := 4

/-- Theorem: The number of different possible orders for a race with 4 distinct participants and no ties is 24 -/
theorem race_permutations_eq_24 : race_permutations = Nat.factorial num_participants := by
  sorry

end race_permutations_eq_24_l2989_298974


namespace hyperbola_dimensions_l2989_298993

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the length of the real axis is 2 units greater than the length of the imaginary axis
    and the focal length is 10, then a = 4 and b = 3. -/
theorem hyperbola_dimensions (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2*a - 2*b = 2 → a^2 + b^2 = 25 → a = 4 ∧ b = 3 := by sorry

end hyperbola_dimensions_l2989_298993


namespace emily_roses_purchase_l2989_298948

theorem emily_roses_purchase (flower_cost : ℕ) (total_spent : ℕ) : 
  flower_cost = 3 →
  total_spent = 12 →
  ∃ (roses : ℕ), roses * 2 * flower_cost = total_spent ∧ roses = 2 :=
by sorry

end emily_roses_purchase_l2989_298948


namespace nancy_savings_l2989_298953

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The number of coins in a dozen -/
def dozen : ℕ := 12

/-- The number of quarters Nancy has -/
def nancy_quarters : ℕ := 3 * dozen

/-- The number of dimes Nancy has -/
def nancy_dimes : ℕ := 2 * dozen

/-- The number of nickels Nancy has -/
def nancy_nickels : ℕ := 5 * dozen

/-- The total monetary value of Nancy's coins -/
def nancy_total : ℚ := 
  (nancy_quarters : ℚ) * quarter_value + 
  (nancy_dimes : ℚ) * dime_value + 
  (nancy_nickels : ℚ) * nickel_value

theorem nancy_savings : nancy_total = 14.40 := by
  sorry

end nancy_savings_l2989_298953


namespace vector_problem_l2989_298972

def a : ℝ × ℝ := (3, -1)

theorem vector_problem (b : ℝ × ℝ) (x : ℝ) :
  let c := λ x => x • a + (1 - x) • b
  let dot_product := λ u v : ℝ × ℝ => u.1 * v.1 + u.2 * v.2
  dot_product a b = -5 ∧ ‖b‖ = Real.sqrt 5 →
  (dot_product a (c x) = 0 → x = 1/3) ∧
  (∃ x₀, ∀ x, ‖c x₀‖ ≤ ‖c x‖ ∧ ‖c x₀‖ = 1) :=
by sorry

end vector_problem_l2989_298972


namespace fifteen_fishers_tomorrow_l2989_298910

/-- Represents the fishing schedule in the coastal village -/
structure FishingSchedule where
  daily : ℕ
  everyOtherDay : ℕ
  everyThreeDay : ℕ
  yesterdayCount : ℕ
  todayCount : ℕ

/-- Calculates the number of people fishing tomorrow given the fishing schedule -/
def tomorrowFishers (schedule : FishingSchedule) : ℕ :=
  schedule.daily + schedule.everyThreeDay + (schedule.everyOtherDay - (schedule.yesterdayCount - schedule.daily))

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_fishers_tomorrow (schedule : FishingSchedule) 
  (h1 : schedule.daily = 7)
  (h2 : schedule.everyOtherDay = 8)
  (h3 : schedule.everyThreeDay = 3)
  (h4 : schedule.yesterdayCount = 12)
  (h5 : schedule.todayCount = 10) :
  tomorrowFishers schedule = 15 := by
  sorry

#eval tomorrowFishers { daily := 7, everyOtherDay := 8, everyThreeDay := 3, yesterdayCount := 12, todayCount := 10 }

end fifteen_fishers_tomorrow_l2989_298910


namespace intersection_equals_N_l2989_298921

def U := ℝ

def M : Set ℝ := {x : ℝ | x < 1}

def N : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

theorem intersection_equals_N : M ∩ N = N := by
  sorry

end intersection_equals_N_l2989_298921


namespace problem_solution_l2989_298905

def f (t : ℝ) : ℝ := t^2003 + 2002*t

theorem problem_solution (x y : ℝ) 
  (h1 : f (x - 1) = -1)
  (h2 : f (y - 2) = 1) : 
  x + y = 3 := by
  sorry

end problem_solution_l2989_298905


namespace test_scores_l2989_298984

theorem test_scores (joao_score claudia_score : ℕ) : 
  (10 ≤ joao_score ∧ joao_score < 100) →  -- João's score is a two-digit number
  (10 ≤ claudia_score ∧ claudia_score < 100) →  -- Cláudia's score is a two-digit number
  claudia_score = joao_score + 13 →  -- Cláudia scored 13 points more than João
  joao_score + claudia_score = 149 →  -- Their combined score is 149
  joao_score = 68 ∧ claudia_score = 81 :=
by sorry

end test_scores_l2989_298984


namespace prob_three_same_color_l2989_298970

def total_marbles : ℕ := 23
def red_marbles : ℕ := 6
def white_marbles : ℕ := 8
def blue_marbles : ℕ := 9

def prob_same_color : ℚ := 160 / 1771

theorem prob_three_same_color :
  let prob_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
  let prob_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
  let prob_blue := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2))
  prob_red + prob_white + prob_blue = prob_same_color := by
sorry

end prob_three_same_color_l2989_298970


namespace frog_climb_time_l2989_298913

/-- Represents the frog's climbing problem in the well -/
structure FrogClimb where
  well_depth : ℕ := 12
  climb_distance : ℕ := 3
  slide_distance : ℕ := 1
  time_to_climb : ℕ := 3
  time_to_slide : ℕ := 1
  time_at_3m_from_top : ℕ := 17

/-- Calculates the total time for the frog to reach the top of the well -/
def total_climb_time (f : FrogClimb) : ℕ :=
  sorry

/-- Theorem stating that the total climb time is 22 minutes -/
theorem frog_climb_time (f : FrogClimb) : total_climb_time f = 22 :=
  sorry

end frog_climb_time_l2989_298913


namespace exactly_one_absent_probability_l2989_298918

/-- The probability of an employee being absent on a given day -/
def p_absent : ℚ := 1 / 30

/-- The probability of an employee being present on a given day -/
def p_present : ℚ := 1 - p_absent

/-- The number of employees selected -/
def n : ℕ := 3

/-- The number of employees that should be absent -/
def k : ℕ := 1

theorem exactly_one_absent_probability :
  (n.choose k : ℚ) * p_absent^k * p_present^(n - k) = 841 / 9000 := by
  sorry

end exactly_one_absent_probability_l2989_298918


namespace geometric_sequence_a5_l2989_298904

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a3 : a 3 = -4) 
  (h_a7 : a 7 = -16) : 
  a 5 = -8 := by
sorry

end geometric_sequence_a5_l2989_298904


namespace prob_at_most_one_value_l2989_298980

/-- The probability that A hits the target -/
def prob_A : ℝ := 0.6

/-- The probability that B hits the target -/
def prob_B : ℝ := 0.7

/-- The probability that at most one of A and B hits the target -/
def prob_at_most_one : ℝ := 1 - prob_A * prob_B

theorem prob_at_most_one_value : prob_at_most_one = 0.58 := by
  sorry

end prob_at_most_one_value_l2989_298980


namespace art_museum_survey_l2989_298900

theorem art_museum_survey (V : ℕ) (E U : ℕ) : 
  E = U →                                     -- Number who enjoyed equals number who understood
  (3 : ℚ) / 4 * V = E →                       -- 3/4 of visitors both enjoyed and understood
  V = 520 →                                   -- Total number of visitors
  V - E = 130                                 -- Number who didn't enjoy and didn't understand
  := by sorry

end art_museum_survey_l2989_298900


namespace sqrt_sum_difference_equals_2sqrt3_quadratic_equation_solutions_l2989_298996

-- Problem 1
theorem sqrt_sum_difference_equals_2sqrt3 :
  Real.sqrt 12 + Real.sqrt 27 / 9 - Real.sqrt (1/3) = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem quadratic_equation_solutions (x : ℝ) :
  x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 := by sorry

end sqrt_sum_difference_equals_2sqrt3_quadratic_equation_solutions_l2989_298996


namespace reciprocal_roots_l2989_298947

theorem reciprocal_roots (a b c : ℝ) (x y : ℝ) : 
  (a * x^2 + b * x + c = 0 ↔ c * (1/x)^2 + b * (1/x) + a = 0) ∧ 
  (c * y^2 + b * y + a = 0 ↔ a * (1/y)^2 + b * (1/y) + c = 0) := by
sorry

end reciprocal_roots_l2989_298947


namespace smaller_number_in_ratio_l2989_298973

/-- Given two positive integers in ratio 4:5 with LCM 180, prove the smaller number is 144 -/
theorem smaller_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / b = 4 / 5 →
  Nat.lcm a b = 180 →
  a = 144 := by
sorry

end smaller_number_in_ratio_l2989_298973


namespace toms_journey_to_virgo_l2989_298998

theorem toms_journey_to_virgo (
  train_ride : ℝ)
  (first_layover : ℝ)
  (bus_ride : ℝ)
  (second_layover : ℝ)
  (first_flight : ℝ)
  (third_layover : ℝ)
  (fourth_layover : ℝ)
  (car_drive : ℝ)
  (first_boat_ride : ℝ)
  (fifth_layover : ℝ)
  (final_walk : ℝ)
  (h1 : train_ride = 5)
  (h2 : first_layover = 1.5)
  (h3 : bus_ride = 4)
  (h4 : second_layover = 0.5)
  (h5 : first_flight = 6)
  (h6 : third_layover = 2)
  (h7 : fourth_layover = 3)
  (h8 : car_drive = 3.5)
  (h9 : first_boat_ride = 1.5)
  (h10 : fifth_layover = 0.75)
  (h11 : final_walk = 1.25) :
  train_ride + first_layover + bus_ride + second_layover + first_flight + 
  third_layover + (3 * bus_ride) + fourth_layover + car_drive + 
  first_boat_ride + fifth_layover + (2 * first_boat_ride - 0.5) + final_walk = 44 := by
  sorry


end toms_journey_to_virgo_l2989_298998


namespace min_ab_value_l2989_298990

theorem min_ab_value (a b : ℕ+) (h : (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = (9 : ℚ)⁻¹) :
  (a * b : ℕ) ≥ 60 ∧ ∃ (a₀ b₀ : ℕ+), (a₀ : ℚ)⁻¹ + (3 * b₀ : ℚ)⁻¹ = (9 : ℚ)⁻¹ ∧ (a₀ * b₀ : ℕ) = 60 :=
by sorry

end min_ab_value_l2989_298990


namespace smallest_k_with_remainders_l2989_298942

theorem smallest_k_with_remainders : ∃! k : ℕ,
  k > 1 ∧
  k % 19 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 → m % 19 = 1 → m % 7 = 1 → m % 3 = 1 → k ≤ m :=
by
  use 400
  sorry

end smallest_k_with_remainders_l2989_298942


namespace trapezoid_area_trapezoid_area_proof_l2989_298950

/-- The area of a trapezoid bounded by y = x + 1, y = 12, y = 7, and the y-axis -/
theorem trapezoid_area : ℝ :=
  let line1 : ℝ → ℝ := λ x ↦ x + 1
  let line2 : ℝ → ℝ := λ _ ↦ 12
  let line3 : ℝ → ℝ := λ _ ↦ 7
  let y_axis : ℝ → ℝ := λ _ ↦ 0
  42.5
  
#check trapezoid_area

/-- Proof that the area of the trapezoid is 42.5 square units -/
theorem trapezoid_area_proof : trapezoid_area = 42.5 := by
  sorry

end trapezoid_area_trapezoid_area_proof_l2989_298950


namespace functions_characterization_l2989_298964

variable (f g : ℚ → ℚ)

-- Define the conditions
axiom condition1 : ∀ x y : ℚ, f (g x + g y) = f (g x) + y
axiom condition2 : ∀ x y : ℚ, g (f x + f y) = g (f x) + y

-- Define the theorem
theorem functions_characterization :
  ∃ a b : ℚ, (a * b = 1) ∧ (∀ x : ℚ, f x = a * x ∧ g x = b * x) :=
sorry

end functions_characterization_l2989_298964


namespace equation_solution_l2989_298937

theorem equation_solution : 
  ∃ y : ℚ, (5 * y - 2) / (6 * y - 6) = 3 / 4 ∧ y = -5 := by
  sorry

end equation_solution_l2989_298937


namespace inequality_solution_set_l2989_298927

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
sorry

end inequality_solution_set_l2989_298927


namespace triangle_perimeter_l2989_298914

/-- Given a triangle with inradius 2.5 cm and area 25 cm², prove its perimeter is 20 cm -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) 
  (h1 : r = 2.5)
  (h2 : A = 25)
  (h3 : A = r * (p / 2)) :
  p = 20 := by
  sorry

end triangle_perimeter_l2989_298914


namespace first_month_bill_is_50_l2989_298907

/-- Represents Elvin's monthly telephone bill --/
structure PhoneBill where
  callCharge : ℝ
  internetCharge : ℝ

/-- The total bill is the sum of call charge and internet charge --/
def PhoneBill.total (bill : PhoneBill) : ℝ :=
  bill.callCharge + bill.internetCharge

theorem first_month_bill_is_50 
  (firstMonth secondMonth : PhoneBill)
  (h1 : firstMonth.total = 50)
  (h2 : secondMonth.total = 76)
  (h3 : secondMonth.callCharge = 2 * firstMonth.callCharge)
  (h4 : firstMonth.internetCharge = secondMonth.internetCharge) :
  firstMonth.total = 50 := by
  sorry

#check first_month_bill_is_50

end first_month_bill_is_50_l2989_298907


namespace congruent_triangles_corresponding_angles_l2989_298988

-- Define a triangle
def Triangle := ℝ × ℝ × ℝ

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define the property of corresponding angles being congruent
def corresponding_angles_congruent (t1 t2 : Triangle) : Prop := sorry

-- Theorem statement
theorem congruent_triangles_corresponding_angles 
  (t1 t2 : Triangle) : congruent t1 t2 → corresponding_angles_congruent t1 t2 := by
  sorry

end congruent_triangles_corresponding_angles_l2989_298988


namespace garden_perimeter_l2989_298991

/-- The perimeter of a rectangular garden with given length and breadth -/
theorem garden_perimeter (length breadth : ℝ) (h1 : length = 360) (h2 : breadth = 240) :
  2 * (length + breadth) = 1200 := by
  sorry

#check garden_perimeter

end garden_perimeter_l2989_298991


namespace tommy_house_price_l2989_298926

/-- The original price of Tommy's first house -/
def original_price : ℝ := 100000

/-- The increased value of Tommy's first house -/
def increased_value : ℝ := original_price * 1.25

/-- The cost of Tommy's new house -/
def new_house_cost : ℝ := 500000

/-- The percentage Tommy paid for the new house from his own funds -/
def own_funds_percentage : ℝ := 0.25

theorem tommy_house_price :
  original_price = 100000 ∧
  increased_value = original_price * 1.25 ∧
  new_house_cost = 500000 ∧
  own_funds_percentage = 0.25 ∧
  new_house_cost * own_funds_percentage = increased_value - original_price :=
by sorry

end tommy_house_price_l2989_298926


namespace b_power_sum_l2989_298994

theorem b_power_sum (b : ℝ) (h : 5 = b + b⁻¹) : b^6 + b⁻¹^6 = 12239 := by sorry

end b_power_sum_l2989_298994


namespace sqrt_32_div_sqrt_8_eq_2_l2989_298919

theorem sqrt_32_div_sqrt_8_eq_2 : Real.sqrt 32 / Real.sqrt 8 = 2 := by
  sorry

end sqrt_32_div_sqrt_8_eq_2_l2989_298919


namespace rebus_solution_l2989_298957

theorem rebus_solution :
  ∃! (A B C : ℕ),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) = (100 * A + 10 * C + C) ∧
    100 * A + 10 * C + C = 1416 :=
by sorry

end rebus_solution_l2989_298957


namespace yellow_straight_probability_l2989_298912

structure Garden where
  roses : ℝ
  daffodils : ℝ
  tulips : ℝ
  green_prob : ℝ
  straight_prob : ℝ
  rose_straight_prob : ℝ
  daffodil_curved_prob : ℝ
  tulip_straight_prob : ℝ

def is_valid_garden (g : Garden) : Prop :=
  g.roses + g.daffodils + g.tulips = 1 ∧
  g.roses = 1/4 ∧
  g.daffodils = 1/2 ∧
  g.tulips = 1/4 ∧
  g.green_prob = 2/3 ∧
  g.straight_prob = 1/2 ∧
  g.rose_straight_prob = 1/6 ∧
  g.daffodil_curved_prob = 1/3 ∧
  g.tulip_straight_prob = 1/8

theorem yellow_straight_probability (g : Garden) 
  (h : is_valid_garden g) : 
  (1 - g.green_prob) * g.straight_prob = 1/6 := by
  sorry

end yellow_straight_probability_l2989_298912


namespace complement_of_P_in_U_l2989_298933

def U : Set Int := {-1, 0, 1, 2}

def P : Set Int := {x : Int | x^2 < 2}

theorem complement_of_P_in_U : {2} = U \ P := by sorry

end complement_of_P_in_U_l2989_298933


namespace cube_root_function_l2989_298901

theorem cube_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * (27 : ℝ)^(1/3) ∧ y = 3 * Real.sqrt 3) →
  k * (8 : ℝ)^(1/3) = 2 * Real.sqrt 3 :=
by sorry

end cube_root_function_l2989_298901


namespace unique_n_value_l2989_298971

theorem unique_n_value (n : ℤ) 
  (h1 : 50 ≤ n ∧ n ≤ 120) 
  (h2 : ∃ k : ℤ, n = 5 * k) 
  (h3 : n % 6 = 3) 
  (h4 : n % 7 = 4) : 
  n = 165 := by sorry

end unique_n_value_l2989_298971


namespace well_volume_l2989_298965

/-- The volume of a cylindrical well with diameter 2 meters and depth 14 meters is π * 14 cubic meters -/
theorem well_volume (π : ℝ) (h : π = Real.pi) :
  let diameter : ℝ := 2
  let depth : ℝ := 14
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * depth
  volume = π * 14 := by sorry

end well_volume_l2989_298965


namespace conner_start_rocks_l2989_298976

/-- Represents the number of rocks collected by each person on each day -/
structure RockCollection where
  sydney_start : ℕ
  conner_start : ℕ
  sydney_day1 : ℕ
  conner_day1 : ℕ
  sydney_day2 : ℕ
  conner_day2 : ℕ
  sydney_day3 : ℕ
  conner_day3 : ℕ

/-- The rock collecting contest scenario -/
def contest_scenario : RockCollection where
  sydney_start := 837
  conner_start := 723  -- This is what we want to prove
  sydney_day1 := 4
  conner_day1 := 8 * 4
  sydney_day2 := 0
  conner_day2 := 123
  sydney_day3 := 2 * (8 * 4)
  conner_day3 := 27

/-- Calculates the total rocks for each person at the end of the contest -/
def total_rocks (rc : RockCollection) : ℕ × ℕ :=
  (rc.sydney_start + rc.sydney_day1 + rc.sydney_day2 + rc.sydney_day3,
   rc.conner_start + rc.conner_day1 + rc.conner_day2 + rc.conner_day3)

/-- Theorem stating that Conner must have started with 723 rocks to at least tie Sydney -/
theorem conner_start_rocks : 
  let (sydney_total, conner_total) := total_rocks contest_scenario
  conner_total ≥ sydney_total ∧ contest_scenario.conner_start = 723 := by
  sorry


end conner_start_rocks_l2989_298976


namespace orange_cost_theorem_l2989_298931

/-- The rate at which oranges are sold in dollars per kilogram -/
def orange_rate : ℚ := 5 / 3

/-- The amount of oranges in kilograms to be purchased -/
def amount_to_buy : ℚ := 12

/-- The cost of buying a given amount of oranges in dollars -/
def cost (kg : ℚ) : ℚ := kg * orange_rate

theorem orange_cost_theorem : cost amount_to_buy = 20 := by
  sorry

end orange_cost_theorem_l2989_298931


namespace old_barbell_cost_l2989_298935

theorem old_barbell_cost (new_barbell_cost : ℝ) (percentage_increase : ℝ) : 
  new_barbell_cost = 325 →
  percentage_increase = 0.30 →
  new_barbell_cost = (1 + percentage_increase) * (new_barbell_cost / (1 + percentage_increase)) →
  new_barbell_cost / (1 + percentage_increase) = 250 := by
sorry

end old_barbell_cost_l2989_298935


namespace function_value_at_three_pi_four_l2989_298969

noncomputable def f (A φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (x + φ)

theorem function_value_at_three_pi_four
  (A φ : ℝ)
  (h1 : A > 0)
  (h2 : 0 < φ)
  (h3 : φ < Real.pi)
  (h4 : ∀ x, f A φ x ≤ 1)
  (h5 : ∃ x, f A φ x = 1)
  (h6 : f A φ (Real.pi / 3) = 1 / 2) :
  f A φ (3 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
sorry

end function_value_at_three_pi_four_l2989_298969


namespace saree_price_calculation_l2989_298938

theorem saree_price_calculation (final_price : ℝ) 
  (h : final_price = 378.675) : ∃ (original_price : ℝ), 
  original_price * 0.85 * 0.90 = final_price ∧ 
  original_price = 495 := by
  sorry

end saree_price_calculation_l2989_298938


namespace no_solution_to_system_l2989_298968

theorem no_solution_to_system :
  ¬ ∃ (x y : ℝ), (2 * x - 3 * y = 8) ∧ (6 * y - 4 * x = 9) := by
  sorry

end no_solution_to_system_l2989_298968


namespace tin_silver_ratio_l2989_298956

/-- Represents the composition of a metal bar made of tin and silver -/
structure MetalBar where
  tin : ℝ
  silver : ℝ

/-- Properties of the metal bar -/
def bar_properties (bar : MetalBar) : Prop :=
  bar.tin + bar.silver = 40 ∧
  0.1375 * bar.tin + 0.075 * bar.silver = 4

/-- The ratio of tin to silver in the bar is 2:3 -/
theorem tin_silver_ratio (bar : MetalBar) :
  bar_properties bar → bar.tin / bar.silver = 2 / 3 := by
  sorry

end tin_silver_ratio_l2989_298956


namespace max_value_of_f_l2989_298982

/-- The function f(x) = x(1-2x) -/
def f (x : ℝ) := x * (1 - 2 * x)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1/8 ∧ ∀ x, 0 < x → x < 1/2 → f x ≤ M :=
sorry

end max_value_of_f_l2989_298982


namespace integer_solution_system_l2989_298925

theorem integer_solution_system (m n : ℤ) : 
  m * (m + n) = n * 12 ∧ n * (m + n) = m * 3 → m = 4 ∧ n = 2 := by
  sorry

end integer_solution_system_l2989_298925


namespace darry_total_steps_l2989_298916

/-- The number of steps Darry climbed in total -/
def total_steps (full_ladder_steps : ℕ) (full_ladder_climbs : ℕ) 
                (small_ladder_steps : ℕ) (small_ladder_climbs : ℕ) : ℕ :=
  full_ladder_steps * full_ladder_climbs + small_ladder_steps * small_ladder_climbs

/-- Proof that Darry climbed 152 steps in total -/
theorem darry_total_steps : 
  total_steps 11 10 6 7 = 152 := by
  sorry

end darry_total_steps_l2989_298916


namespace point_on_line_with_distance_l2989_298903

theorem point_on_line_with_distance (x₀ y₀ : ℝ) :
  (3 * x₀ + y₀ - 5 = 0) →
  (|x₀ - y₀ - 1| / Real.sqrt 2 = Real.sqrt 2) →
  ((x₀ = 1 ∧ y₀ = 2) ∨ (x₀ = 2 ∧ y₀ = -1)) :=
by sorry

end point_on_line_with_distance_l2989_298903


namespace logan_corn_purchase_l2989_298963

/-- Proves that Logan bought 15.0 pounds of corn given the problem conditions -/
theorem logan_corn_purchase 
  (corn_price : ℝ) 
  (bean_price : ℝ) 
  (total_weight : ℝ) 
  (total_cost : ℝ) 
  (h1 : corn_price = 1.20)
  (h2 : bean_price = 0.60)
  (h3 : total_weight = 30)
  (h4 : total_cost = 27.00) : 
  ∃ (corn_weight : ℝ) (bean_weight : ℝ),
    corn_weight + bean_weight = total_weight ∧ 
    corn_price * corn_weight + bean_price * bean_weight = total_cost ∧ 
    corn_weight = 15.0 := by
  sorry

end logan_corn_purchase_l2989_298963


namespace prob_two_empty_given_at_least_one_empty_l2989_298999

/-- The number of balls -/
def num_balls : ℕ := 4

/-- The number of boxes -/
def num_boxes : ℕ := 4

/-- The number of ways to place balls into boxes with exactly one empty box -/
def ways_one_empty : ℕ := 144

/-- The number of ways to place balls into boxes with exactly two empty boxes -/
def ways_two_empty : ℕ := 84

/-- The number of ways to place balls into boxes with exactly three empty boxes -/
def ways_three_empty : ℕ := 4

/-- The probability of exactly two boxes being empty given at least one box is empty -/
theorem prob_two_empty_given_at_least_one_empty :
  (ways_two_empty : ℚ) / (ways_one_empty + ways_two_empty + ways_three_empty) = 21 / 58 := by
  sorry

end prob_two_empty_given_at_least_one_empty_l2989_298999


namespace minyoung_fruit_sale_l2989_298978

theorem minyoung_fruit_sale :
  ∀ (tangerines apples : ℕ),
    tangerines = 2 →
    apples = 7 →
    tangerines + apples = 9 :=
by
  sorry

end minyoung_fruit_sale_l2989_298978


namespace painted_cube_theorem_l2989_298962

/-- Represents a cube composed of unit cubes -/
structure PaintedCube where
  size : ℕ
  totalCubes : ℕ
  surfacePainted : Bool

/-- Counts the number of unit cubes with a specific number of faces painted -/
def countPaintedFaces (cube : PaintedCube) (numFaces : ℕ) : ℕ :=
  match numFaces with
  | 3 => 8
  | 2 => 12 * (cube.size - 2)
  | 1 => 6 * (cube.size - 2)^2
  | 0 => (cube.size - 2)^3
  | _ => 0

theorem painted_cube_theorem (cube : PaintedCube) 
  (h1 : cube.size = 10) 
  (h2 : cube.totalCubes = 1000) 
  (h3 : cube.surfacePainted = true) :
  (countPaintedFaces cube 3 = 8) ∧
  (countPaintedFaces cube 2 = 96) ∧
  (countPaintedFaces cube 1 = 384) ∧
  (countPaintedFaces cube 0 = 512) := by
  sorry

end painted_cube_theorem_l2989_298962


namespace parallel_lines_circle_intersection_l2989_298986

theorem parallel_lines_circle_intersection (r : ℝ) : 
  ∀ d : ℝ, 
    (17682 + (21/4) * d^2 = 42 * r^2) ∧ 
    (4394 + (117/4) * d^2 = 26 * r^2) → 
    d = Real.sqrt 127 := by
  sorry

end parallel_lines_circle_intersection_l2989_298986


namespace triangle_abc_properties_l2989_298945

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  C = π / 3 →
  (1/2 * a * b * Real.sin C = Real.sqrt 3 → a = 2 ∧ b = 2) ∧
  (Real.sin B = 2 * Real.sin A → 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3 / 3) :=
by sorry

end triangle_abc_properties_l2989_298945


namespace rod_cutting_l2989_298997

theorem rod_cutting (total_length : Real) (num_pieces : Nat) :
  total_length = 42.5 → num_pieces = 50 →
  (total_length / num_pieces) * 100 = 85 := by
  sorry

end rod_cutting_l2989_298997


namespace fencing_cost_per_foot_l2989_298959

/-- The cost of fencing per foot -/
def cost_per_foot (side_length back_length total_cost : ℚ) : ℚ :=
  let total_length := 2 * side_length + back_length
  let neighbor_back_contribution := back_length / 2
  let neighbor_left_contribution := side_length / 3
  let cole_length := total_length - neighbor_back_contribution - neighbor_left_contribution
  total_cost / cole_length

/-- Theorem stating that the cost per foot of fencing is $3 -/
theorem fencing_cost_per_foot :
  cost_per_foot 9 18 72 = 3 :=
sorry

end fencing_cost_per_foot_l2989_298959


namespace sum_of_square_roots_inequality_l2989_298960

theorem sum_of_square_roots_inequality (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_eq_four : a + b + c + d = 4) : 
  Real.sqrt (a + b + c) + Real.sqrt (b + c + d) + Real.sqrt (c + d + a) + Real.sqrt (d + a + b) ≥ 6 := by
  sorry

end sum_of_square_roots_inequality_l2989_298960


namespace rectangular_prism_sum_l2989_298911

/-- A rectangular prism is a three-dimensional shape with 6 faces, 12 edges, and 8 vertices. -/
structure RectangularPrism where
  faces : Nat
  edges : Nat
  vertices : Nat
  faces_eq : faces = 6
  edges_eq : edges = 12
  vertices_eq : vertices = 8

/-- The sum of faces, edges, and vertices of a rectangular prism is 26. -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  rp.faces + rp.edges + rp.vertices = 26 := by
  sorry

end rectangular_prism_sum_l2989_298911


namespace mustard_total_l2989_298952

theorem mustard_total (table1 table2 table3 : ℚ) 
  (h1 : table1 = 0.25)
  (h2 : table2 = 0.25)
  (h3 : table3 = 0.38) :
  table1 + table2 + table3 = 0.88 := by
  sorry

end mustard_total_l2989_298952


namespace circle_intersection_and_tangent_line_l2989_298979

theorem circle_intersection_and_tangent_line :
  ∃ (A B C : ℝ),
    (∀ x y : ℝ, A * x^2 + A * y^2 + B * x + C = 0) ∧
    (∀ x y : ℝ, (A * x^2 + A * y^2 + B * x + C = 0) →
      ((x^2 + y^2 - 1 = 0) ∧ (x^2 - 4*x + y^2 = 0)) ∨
      ((x^2 + y^2 - 1 ≠ 0) ∧ (x^2 - 4*x + y^2 ≠ 0))) ∧
    (∃ x₀ y₀ : ℝ,
      A * x₀^2 + A * y₀^2 + B * x₀ + C = 0 ∧
      x₀ - Real.sqrt 3 * y₀ - 6 = 0 ∧
      ∀ x y : ℝ, A * x^2 + A * y^2 + B * x + C = 0 →
        (x - Real.sqrt 3 * y - 6)^2 ≥ (x₀ - Real.sqrt 3 * y₀ - 6)^2) :=
by sorry

end circle_intersection_and_tangent_line_l2989_298979


namespace inscribed_circle_ratio_l2989_298936

/-- A circle inscribed in a semicircle -/
structure InscribedCircle where
  R : ℝ  -- Radius of the semicircle
  r : ℝ  -- Radius of the inscribed circle
  O : ℝ × ℝ  -- Center of the semicircle
  A : ℝ × ℝ  -- One end of the semicircle's diameter
  P : ℝ × ℝ  -- Center of the inscribed circle
  h₁ : R > 0  -- Radius of semicircle is positive
  h₂ : r > 0  -- Radius of inscribed circle is positive
  h₃ : A = (O.1 - R, O.2)  -- A is R units to the left of O
  h₄ : dist P O = dist P A  -- P is equidistant from O and A

/-- The ratio of radii in an inscribed circle is 3:8 -/
theorem inscribed_circle_ratio (c : InscribedCircle) : c.r / c.R = 3 / 8 := by
  sorry

end inscribed_circle_ratio_l2989_298936


namespace dave_apps_problem_l2989_298929

theorem dave_apps_problem (initial_apps final_apps : ℕ) 
  (h1 : initial_apps = 15)
  (h2 : final_apps = 14)
  (h3 : ∃ (added deleted : ℕ), initial_apps + added - deleted = final_apps ∧ deleted = added + 1) :
  ∃ (added : ℕ), added = 0 ∧ initial_apps + added - (added + 1) = final_apps :=
by sorry

end dave_apps_problem_l2989_298929


namespace arithmetic_progression_rth_term_l2989_298951

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℕ := 5 * n + 4 * n^2

/-- The r-th term of the arithmetic progression -/
def a (r : ℕ) : ℕ := S r - S (r - 1)

theorem arithmetic_progression_rth_term (r : ℕ) (h : r > 0) : a r = 8 * r + 1 := by
  sorry

end arithmetic_progression_rth_term_l2989_298951


namespace davis_remaining_sticks_l2989_298932

/-- The number of popsicle sticks Miss Davis had initially -/
def initial_sticks : ℕ := 170

/-- The number of groups in Miss Davis's class -/
def num_groups : ℕ := 10

/-- The number of popsicle sticks given to each group -/
def sticks_per_group : ℕ := 15

/-- The number of popsicle sticks Miss Davis has left -/
def remaining_sticks : ℕ := initial_sticks - (num_groups * sticks_per_group)

theorem davis_remaining_sticks :
  remaining_sticks = 20 := by sorry

end davis_remaining_sticks_l2989_298932


namespace not_always_intersects_x_axis_l2989_298917

/-- Represents a circle in a 2D plane -/
structure Circle where
  a : ℝ  -- x-coordinate of the center
  b : ℝ  -- y-coordinate of the center
  r : ℝ  -- radius
  r_pos : r > 0

/-- Predicate to check if a circle intersects the x-axis -/
def intersects_x_axis (c : Circle) : Prop :=
  ∃ x : ℝ, (x - c.a)^2 + c.b^2 = c.r^2

/-- Theorem stating that b < r does not always imply intersection with x-axis -/
theorem not_always_intersects_x_axis :
  ¬ (∀ c : Circle, c.b < c.r → intersects_x_axis c) :=
sorry

end not_always_intersects_x_axis_l2989_298917


namespace train_length_problem_l2989_298949

/-- Given a platform length, time to pass, and train speed, calculates the length of the train -/
def train_length (platform_length time_to_pass train_speed : ℝ) : ℝ :=
  train_speed * time_to_pass - platform_length

/-- Theorem stating that under the given conditions, the train length is 50 meters -/
theorem train_length_problem :
  let platform_length : ℝ := 100
  let time_to_pass : ℝ := 10
  let train_speed : ℝ := 15
  train_length platform_length time_to_pass train_speed = 50 := by
sorry

#eval train_length 100 10 15

end train_length_problem_l2989_298949


namespace max_fraction_value_l2989_298943

theorem max_fraction_value (a b : ℝ) 
  (ha : 100 ≤ a ∧ a ≤ 500) (hb : 500 ≤ b ∧ b ≤ 1500) : 
  (b - 100) / (a + 50) ≤ 28/3 := by
  sorry

end max_fraction_value_l2989_298943


namespace truck_rental_miles_driven_l2989_298981

theorem truck_rental_miles_driven 
  (rental_fee : ℚ) 
  (charge_per_mile : ℚ) 
  (total_paid : ℚ) 
  (h1 : rental_fee = 2099 / 100)
  (h2 : charge_per_mile = 25 / 100)
  (h3 : total_paid = 9574 / 100) : 
  (total_paid - rental_fee) / charge_per_mile = 299 := by
sorry

#eval (9574 / 100 - 2099 / 100) / (25 / 100)

end truck_rental_miles_driven_l2989_298981


namespace yellow_last_probability_l2989_298940

/-- Represents a bag of marbles -/
structure Bag where
  yellow : ℕ
  blue : ℕ
  white : ℕ
  black : ℕ
  green : ℕ
  red : ℕ

/-- The probability of drawing a yellow marble as the last marble -/
def last_yellow_probability (bagA bagB bagC bagD : Bag) : ℚ :=
  sorry

/-- The theorem stating the probability of drawing a yellow marble last -/
theorem yellow_last_probability :
  let bagA : Bag := { yellow := 0, blue := 0, white := 5, black := 5, green := 0, red := 0 }
  let bagB : Bag := { yellow := 8, blue := 6, white := 0, black := 0, green := 0, red := 0 }
  let bagC : Bag := { yellow := 3, blue := 7, white := 0, black := 0, green := 0, red := 0 }
  let bagD : Bag := { yellow := 0, blue := 0, white := 0, black := 0, green := 4, red := 6 }
  last_yellow_probability bagA bagB bagC bagD = 73 / 140 := by
  sorry

end yellow_last_probability_l2989_298940


namespace six_grade_assignments_l2989_298941

/-- Number of ways to assign n grades, where each grade is 2, 3, or 4, and no two consecutive grades can both be 2 -/
def gradeAssignments : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => 2 * gradeAssignments (n + 1) + 2 * gradeAssignments n

/-- The number of ways to assign 6 grades under the given conditions is 448 -/
theorem six_grade_assignments : gradeAssignments 6 = 448 := by
  sorry

end six_grade_assignments_l2989_298941


namespace dale_had_two_eggs_l2989_298985

/-- The cost of breakfast for Dale and Andrew -/
def breakfast_cost (dale_eggs : ℕ) : ℝ :=
  (2 * 1 + dale_eggs * 3) + (1 * 1 + 2 * 3)

/-- Theorem: Dale had 2 eggs -/
theorem dale_had_two_eggs : 
  ∃ (dale_eggs : ℕ), breakfast_cost dale_eggs = 15 ∧ dale_eggs = 2 :=
by
  sorry

end dale_had_two_eggs_l2989_298985


namespace matrix_power_1000_l2989_298955

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 2, 1]

theorem matrix_power_1000 :
  A ^ 1000 = !![1, 0; 2000, 1] := by sorry

end matrix_power_1000_l2989_298955


namespace dining_bill_share_l2989_298983

/-- Given a total bill, number of people, and tip percentage, calculates each person's share --/
def calculate_share (total_bill : ℚ) (num_people : ℕ) (tip_percentage : ℚ) : ℚ :=
  (total_bill * (1 + tip_percentage)) / num_people

/-- Proves that the calculated share for the given conditions is approximately $48.53 --/
theorem dining_bill_share :
  let total_bill : ℚ := 211
  let num_people : ℕ := 5
  let tip_percentage : ℚ := 15 / 100
  abs (calculate_share total_bill num_people tip_percentage - 48.53) < 0.01 := by
  sorry

end dining_bill_share_l2989_298983


namespace chess_team_girls_l2989_298958

theorem chess_team_girls (total : ℕ) (attended : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 26 →
  attended = 16 →
  boys + girls = total →
  boys + (girls / 2) = attended →
  girls = 20 := by
sorry

end chess_team_girls_l2989_298958


namespace triangle_properties_l2989_298934

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin (2 * B) = Real.sqrt 3 * b * Real.sin A →
  Real.cos A = 1 / 3 →
  B = π / 6 ∧ Real.sin C = (2 * Real.sqrt 6 + 1) / 6 := by
sorry

end triangle_properties_l2989_298934


namespace max_value_expression_l2989_298906

theorem max_value_expression (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  (x^2 - 2*x*y + 2*y^2) * (x^2 - 2*x*z + 2*z^2) * (y^2 - 2*y*z + 2*z^2) ≤ 12 ∧
  ∃ x y z, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 3 ∧
    (x^2 - 2*x*y + 2*y^2) * (x^2 - 2*x*z + 2*z^2) * (y^2 - 2*y*z + 2*z^2) = 12 :=
by sorry

end max_value_expression_l2989_298906


namespace hex_pattern_theorem_l2989_298928

/-- Represents a hexagonal tile pattern -/
structure HexPattern where
  blue_tiles : ℕ
  green_tiles : ℕ
  red_tiles : ℕ

/-- Creates a new pattern by adding green and red tiles -/
def add_border (initial : HexPattern) (green_layers : ℕ) : HexPattern :=
  let new_green := initial.green_tiles + green_layers * 24
  let new_red := 12
  { blue_tiles := initial.blue_tiles,
    green_tiles := new_green,
    red_tiles := new_red }

theorem hex_pattern_theorem (initial : HexPattern) :
  initial.blue_tiles = 20 →
  initial.green_tiles = 9 →
  let new_pattern := add_border initial 2
  new_pattern.red_tiles = 12 ∧
  new_pattern.green_tiles + new_pattern.red_tiles - new_pattern.blue_tiles = 25 := by
  sorry

end hex_pattern_theorem_l2989_298928


namespace inverse_proportion_order_l2989_298944

theorem inverse_proportion_order (y₁ y₂ y₃ : ℝ) : 
  y₁ = -12 / (-3) → 
  y₂ = -12 / (-2) → 
  y₃ = -12 / 2 → 
  y₃ < y₁ ∧ y₁ < y₂ := by
sorry

end inverse_proportion_order_l2989_298944


namespace saree_discount_problem_l2989_298987

/-- Proves that the first discount percentage is 10% given the conditions of the saree pricing problem -/
theorem saree_discount_problem (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 600 →
  second_discount = 5 →
  final_price = 513 →
  ∃ (first_discount : ℝ),
    first_discount = 10 ∧
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end saree_discount_problem_l2989_298987


namespace inequality_proof_l2989_298992

theorem inequality_proof (x y z p q : ℝ) (n : Nat) (h1 : y = x^n + p*x + q) (h2 : z = y^n + p*y + q) (h3 : x = z^n + p*z + q) (h4 : n = 2 ∨ n = 2010) :
  x^2*y + y^2*z + z^2*x ≥ x^2*z + y^2*x + z^2*y := by
  sorry

end inequality_proof_l2989_298992


namespace sqrt_eight_div_sqrt_two_eq_two_l2989_298967

theorem sqrt_eight_div_sqrt_two_eq_two : Real.sqrt 8 / Real.sqrt 2 = 2 := by
  sorry

end sqrt_eight_div_sqrt_two_eq_two_l2989_298967


namespace sample_size_is_80_l2989_298977

/-- Represents the ratio of quantities for products A, B, and C -/
structure ProductRatio where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a stratified sample -/
structure StratifiedSample where
  ratio : ProductRatio
  units_of_a : ℕ

/-- Theorem stating that given the specific conditions, the sample size is 80 -/
theorem sample_size_is_80 (sample : StratifiedSample) 
  (h_ratio : sample.ratio = ProductRatio.mk 2 3 5)
  (h_units_a : sample.units_of_a = 16) : 
  (sample.units_of_a / sample.ratio.a) * (sample.ratio.a + sample.ratio.b + sample.ratio.c) = 80 := by
  sorry

#check sample_size_is_80

end sample_size_is_80_l2989_298977


namespace decimal_representation_of_fraction_l2989_298909

theorem decimal_representation_of_fraction (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 16 / 50 → (n : ℚ) / d = 0.32 := by
  sorry

end decimal_representation_of_fraction_l2989_298909


namespace cyclist_problem_l2989_298995

/-- Proves that given the conditions of the cyclist problem, the speed of cyclist A is 10 mph --/
theorem cyclist_problem (distance : ℝ) (speed_difference : ℝ) (meeting_distance : ℝ)
  (h1 : distance = 100)
  (h2 : speed_difference = 5)
  (h3 : meeting_distance = 20) :
  ∃ (speed_a : ℝ), speed_a = 10 ∧ 
    (distance - meeting_distance) / speed_a = 
    (distance + meeting_distance) / (speed_a + speed_difference) :=
by
  sorry


end cyclist_problem_l2989_298995


namespace modulus_of_z_l2989_298923

theorem modulus_of_z (z : ℂ) (h : z / (Real.sqrt 3 - Complex.I) = 1 + Real.sqrt 3 * Complex.I) : 
  Complex.abs z = 4 := by
sorry

end modulus_of_z_l2989_298923


namespace trapezoid_median_l2989_298930

/-- Given a triangle with base 24 inches and area 192 square inches, and a trapezoid with the same 
    height and area as the triangle, the median of the trapezoid is 12 inches. -/
theorem trapezoid_median (triangle_base : ℝ) (triangle_area : ℝ) (trapezoid_height : ℝ) 
  (trapezoid_median : ℝ) : 
  triangle_base = 24 → 
  triangle_area = 192 → 
  triangle_area = (1/2) * triangle_base * trapezoid_height → 
  triangle_area = trapezoid_median * trapezoid_height → 
  trapezoid_median = 12 := by
  sorry

#check trapezoid_median

end trapezoid_median_l2989_298930


namespace pepsi_amount_l2989_298989

/-- Represents the drink inventory and packing constraints -/
structure DrinkInventory where
  maaza : ℕ
  sprite : ℕ
  total_cans : ℕ
  pepsi : ℕ

/-- Calculates the greatest common divisor of two natural numbers -/
def gcd (a b : ℕ) : ℕ := sorry

/-- Theorem: Given the inventory and constraints, the amount of Pepsi is 144 liters -/
theorem pepsi_amount (inventory : DrinkInventory) 
  (h1 : inventory.maaza = 80)
  (h2 : inventory.sprite = 368)
  (h3 : inventory.total_cans = 37)
  (h4 : ∃ (can_size : ℕ), can_size > 0 ∧ 
        inventory.maaza % can_size = 0 ∧ 
        inventory.sprite % can_size = 0 ∧
        inventory.pepsi % can_size = 0 ∧
        inventory.total_cans = inventory.maaza / can_size + inventory.sprite / can_size + inventory.pepsi / can_size)
  : inventory.pepsi = 144 := by
  sorry

end pepsi_amount_l2989_298989


namespace pen_bag_discount_l2989_298946

theorem pen_bag_discount (price : ℝ) (discount : ℝ) (savings : ℝ) :
  price = 18 →
  discount = 0.1 →
  savings = 36 →
  ∃ (x : ℝ),
    price * (x + 1) * (1 - discount) = price * x - savings ∧
    x = 30 ∧
    price * (x + 1) * (1 - discount) = 486 :=
by
  sorry

end pen_bag_discount_l2989_298946


namespace unique_pair_satisfying_conditions_l2989_298922

theorem unique_pair_satisfying_conditions :
  ∀ a b : ℕ+,
  a + b + (Nat.gcd a b)^2 = Nat.lcm a b ∧
  Nat.lcm a b = 2 * Nat.lcm (a - 1) b →
  a = 6 ∧ b = 15 := by
sorry

end unique_pair_satisfying_conditions_l2989_298922


namespace equation_relationship_l2989_298954

/-- Represents a relationship between x and y --/
inductive Relationship
  | Direct
  | Inverse
  | Neither

/-- Determines the relationship between x and y in the equation 2x + 3y = 15 --/
def relationshipInEquation : Relationship := sorry

/-- Theorem stating that the relationship in the equation 2x + 3y = 15 is neither direct nor inverse proportionality --/
theorem equation_relationship :
  relationshipInEquation = Relationship.Neither := by sorry

end equation_relationship_l2989_298954
