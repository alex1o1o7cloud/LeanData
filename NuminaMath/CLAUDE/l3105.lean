import Mathlib

namespace NUMINAMATH_CALUDE_fraction_evaluation_l3105_310567

theorem fraction_evaluation : 
  (((1 : ℚ) / 2 + (1 : ℚ) / 5) / ((3 : ℚ) / 7 - (1 : ℚ) / 14)) / ((3 : ℚ) / 4) = (196 : ℚ) / 75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3105_310567


namespace NUMINAMATH_CALUDE_correct_num_teams_l3105_310524

/-- The number of teams in a league where each team plays every other team exactly once -/
def num_teams : ℕ := 14

/-- The total number of games played in the league -/
def total_games : ℕ := 91

/-- Theorem stating that the number of teams is correct given the conditions -/
theorem correct_num_teams :
  (num_teams * (num_teams - 1)) / 2 = total_games :=
by sorry

end NUMINAMATH_CALUDE_correct_num_teams_l3105_310524


namespace NUMINAMATH_CALUDE_money_distribution_l3105_310561

theorem money_distribution (a b c : ℕ) : 
  a = 3 * b →
  b > c →
  a + b + c = 645 →
  b = 134 →
  b - c = 25 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l3105_310561


namespace NUMINAMATH_CALUDE_negation_equivalence_l3105_310581

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x / (x - 1) > 0) ↔ (∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3105_310581


namespace NUMINAMATH_CALUDE_g_neg_three_eq_four_l3105_310505

/-- The function g is defined as g(x) = x^2 + 2x + 1 for all real x. -/
def g (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- Theorem: The value of g(-3) is equal to 4. -/
theorem g_neg_three_eq_four : g (-3) = 4 := by sorry

end NUMINAMATH_CALUDE_g_neg_three_eq_four_l3105_310505


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3105_310512

theorem quadratic_inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, x = 1 → k^2 * x^2 - 6*k*x + 8 ≥ 0) → 
  (k ≥ 4 ∨ k ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3105_310512


namespace NUMINAMATH_CALUDE_slipper_cost_l3105_310515

theorem slipper_cost (total_items : ℕ) (slipper_count : ℕ) (lipstick_count : ℕ) (lipstick_price : ℚ) 
  (hair_color_count : ℕ) (hair_color_price : ℚ) (total_paid : ℚ) :
  total_items = slipper_count + lipstick_count + hair_color_count →
  total_items = 18 →
  slipper_count = 6 →
  lipstick_count = 4 →
  lipstick_price = 5/4 →
  hair_color_count = 8 →
  hair_color_price = 3 →
  total_paid = 44 →
  (total_paid - (lipstick_count * lipstick_price + hair_color_count * hair_color_price)) / slipper_count = 5/2 :=
by
  sorry

#check slipper_cost

end NUMINAMATH_CALUDE_slipper_cost_l3105_310515


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l3105_310596

/-- Represents the number of roots available to the wizard. -/
def num_roots : ℕ := 4

/-- Represents the number of minerals available to the wizard. -/
def num_minerals : ℕ := 5

/-- Represents the number of incompatible pairs of roots and minerals. -/
def num_incompatible_pairs : ℕ := 3

/-- Theorem stating the number of possible combinations for the wizard's elixir. -/
theorem wizard_elixir_combinations : 
  num_roots * num_minerals - num_incompatible_pairs = 17 := by
  sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l3105_310596


namespace NUMINAMATH_CALUDE_sin_400_lt_cos_40_l3105_310526

theorem sin_400_lt_cos_40 : 
  Real.sin (400 * Real.pi / 180) < Real.cos (40 * Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_400_lt_cos_40_l3105_310526


namespace NUMINAMATH_CALUDE_min_value_3m_plus_n_l3105_310572

/-- The minimum value of 3m + n given the conditions -/
theorem min_value_3m_plus_n (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hm : m > 0) (hn : n > 0) : 
  let f := fun x => a^(x + 3) - 2
  let A := (-3, -1)
  (A.1 / m + A.2 / n + 1 = 0) →
  ∀ m' n', m' > 0 → n' > 0 → 
    (m' / m' + n' / n' + 1 = 0) → 
    (3 * m + n ≤ 3 * m' + n') :=
by sorry

end NUMINAMATH_CALUDE_min_value_3m_plus_n_l3105_310572


namespace NUMINAMATH_CALUDE_angle_A_measure_l3105_310559

theorem angle_A_measure :
  ∀ (A B C : ℝ) (small_angle : ℝ),
  B = 120 →
  B + C = 180 →
  small_angle = 50 →
  small_angle + C + 70 = 180 →
  A + B = 180 →
  A = 60 :=
by sorry

end NUMINAMATH_CALUDE_angle_A_measure_l3105_310559


namespace NUMINAMATH_CALUDE_star_one_neg_three_l3105_310518

-- Define the ※ operation
def star (a b : ℝ) : ℝ := 2 * a * b - b^2

-- Theorem statement
theorem star_one_neg_three : star 1 (-3) = -15 := by sorry

end NUMINAMATH_CALUDE_star_one_neg_three_l3105_310518


namespace NUMINAMATH_CALUDE_heaviest_lightest_difference_l3105_310574

def pumpkin_contest (brad_weight jessica_weight betty_weight : ℝ) : Prop :=
  jessica_weight = brad_weight / 2 ∧
  betty_weight = 4 * jessica_weight ∧
  brad_weight = 54

theorem heaviest_lightest_difference (brad_weight jessica_weight betty_weight : ℝ) 
  (h : pumpkin_contest brad_weight jessica_weight betty_weight) :
  max betty_weight (max brad_weight jessica_weight) - 
  min betty_weight (min brad_weight jessica_weight) = 81 := by
  sorry

end NUMINAMATH_CALUDE_heaviest_lightest_difference_l3105_310574


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l3105_310570

theorem halfway_between_fractions :
  (1 / 12 + 1 / 20) / 2 = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l3105_310570


namespace NUMINAMATH_CALUDE_distance_after_pie_is_18_l3105_310538

/-- Calculates the distance driven after buying pie and before stopping for gas -/
def distance_after_pie (total_distance : ℕ) (distance_before_pie : ℕ) (remaining_distance : ℕ) : ℕ :=
  total_distance - distance_before_pie - remaining_distance

/-- Proves that the distance driven after buying pie and before stopping for gas is 18 miles -/
theorem distance_after_pie_is_18 :
  distance_after_pie 78 35 25 = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_pie_is_18_l3105_310538


namespace NUMINAMATH_CALUDE_choose_two_from_three_l3105_310523

theorem choose_two_from_three (n : ℕ) (k : ℕ) : n = 3 ∧ k = 2 → Nat.choose n k = 3 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_three_l3105_310523


namespace NUMINAMATH_CALUDE_intersection_equality_l3105_310599

theorem intersection_equality (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
  let B : Set ℝ := {x | a*x - 1 = 0}
  (A ∩ B = B) → (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l3105_310599


namespace NUMINAMATH_CALUDE_four_digit_number_property_l3105_310573

theorem four_digit_number_property (m : ℕ) : 
  1000 ≤ m ∧ m ≤ 2025 →
  ∃ (n : ℕ), n > 0 ∧ Nat.Prime (m - n) ∧ ∃ (k : ℕ), m * n = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_property_l3105_310573


namespace NUMINAMATH_CALUDE_square_to_rectangle_area_increase_l3105_310536

theorem square_to_rectangle_area_increase (s : ℝ) (h : s > 0) :
  let original_area := s^2
  let new_length := 1.3 * s
  let new_width := 1.2 * s
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.56 := by
sorry

end NUMINAMATH_CALUDE_square_to_rectangle_area_increase_l3105_310536


namespace NUMINAMATH_CALUDE_system_solution_l3105_310551

theorem system_solution (x y t : ℝ) :
  (x^2 + t = 1 ∧ (x + y) * t = 0 ∧ y^2 + t = 1) ↔
  ((t = 0 ∧ ((x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1))) ∨
   (0 < t ∧ t < 1 ∧ ((x = Real.sqrt (1 - t) ∧ y = -Real.sqrt (1 - t)) ∨
                     (x = -Real.sqrt (1 - t) ∧ y = Real.sqrt (1 - t))))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3105_310551


namespace NUMINAMATH_CALUDE_cistern_length_is_nine_l3105_310513

/-- Represents a rectangular cistern with water -/
structure WaterCistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  totalWetArea : ℝ

/-- Calculates the wet surface area of a cistern -/
def wetSurfaceArea (c : WaterCistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem: The length of the cistern with given parameters is 9 meters -/
theorem cistern_length_is_nine :
  ∃ (c : WaterCistern),
    c.width = 4 ∧
    c.depth = 1.25 ∧
    c.totalWetArea = 68.5 ∧
    wetSurfaceArea c = c.totalWetArea ∧
    c.length = 9 := by
  sorry

end NUMINAMATH_CALUDE_cistern_length_is_nine_l3105_310513


namespace NUMINAMATH_CALUDE_flight_time_theorem_l3105_310544

/-- Represents the flight scenario between two towns -/
structure FlightScenario where
  d : ℝ  -- distance between towns
  p : ℝ  -- speed of plane in still air
  w : ℝ  -- speed of wind
  time_against : ℝ  -- time for flight against wind
  time_diff : ℝ  -- difference in time between with-wind and still air flights

/-- The theorem stating the conditions and the result to be proved -/
theorem flight_time_theorem (scenario : FlightScenario) 
  (h1 : scenario.time_against = 84)
  (h2 : scenario.time_diff = 9)
  (h3 : scenario.d = scenario.time_against * (scenario.p - scenario.w))
  (h4 : scenario.d / (scenario.p + scenario.w) = scenario.d / scenario.p - scenario.time_diff) :
  scenario.d / (scenario.p + scenario.w) = 63 ∨ scenario.d / (scenario.p + scenario.w) = 12 := by
  sorry

end NUMINAMATH_CALUDE_flight_time_theorem_l3105_310544


namespace NUMINAMATH_CALUDE_evaluate_expression_l3105_310510

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 2/3) (hz : z = -3) :
  x^3 * y^2 * z^2 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3105_310510


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l3105_310575

theorem fraction_product_simplification : 
  (36 : ℚ) / 34 * 26 / 48 * 136 / 78 * 9 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l3105_310575


namespace NUMINAMATH_CALUDE_fish_weight_l3105_310532

theorem fish_weight : 
  ∀ w : ℝ, w = 2 + w / 3 → w = 3 := by sorry

end NUMINAMATH_CALUDE_fish_weight_l3105_310532


namespace NUMINAMATH_CALUDE_sum_nth_group_is_cube_l3105_310566

/-- Returns the nth odd number -/
def nthOdd (n : ℕ) : ℕ := 2 * n - 1

/-- Returns the sum of the first n odd numbers -/
def sumFirstNOdds (n : ℕ) : ℕ := n^2

/-- Returns the sum of odd numbers in the nth group -/
def sumNthGroup (n : ℕ) : ℕ :=
  sumFirstNOdds (sumFirstNOdds n) - sumFirstNOdds (sumFirstNOdds (n - 1))

theorem sum_nth_group_is_cube (n : ℕ) (h : 1 ≤ n ∧ n ≤ 5) : sumNthGroup n = n^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_nth_group_is_cube_l3105_310566


namespace NUMINAMATH_CALUDE_correct_distribution_l3105_310503

/-- The number of ways to distribute men and women into groups --/
def distribute_people (num_men num_women : ℕ) : ℕ :=
  let group1 := Nat.choose num_men 2 * Nat.choose num_women 1
  let group2 := Nat.choose (num_men - 2) 1 * Nat.choose (num_women - 1) 2
  let group3 := Nat.choose 1 1 * Nat.choose 2 2
  (group1 * group2 * group3) / 2

/-- Theorem stating the correct number of distributions --/
theorem correct_distribution : distribute_people 4 5 = 180 := by
  sorry

end NUMINAMATH_CALUDE_correct_distribution_l3105_310503


namespace NUMINAMATH_CALUDE_square_area_equals_perimeter_implies_perimeter_16_l3105_310595

theorem square_area_equals_perimeter_implies_perimeter_16 :
  ∀ s : ℝ, s > 0 → s^2 = 4*s → 4*s = 16 := by sorry

end NUMINAMATH_CALUDE_square_area_equals_perimeter_implies_perimeter_16_l3105_310595


namespace NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l3105_310555

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l3105_310555


namespace NUMINAMATH_CALUDE_no_solution_exists_l3105_310584

theorem no_solution_exists : ¬ ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a + b ≠ 0 ∧ 1 / a + 2 / b = 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3105_310584


namespace NUMINAMATH_CALUDE_f_of_2_equals_5_l3105_310537

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x - 1

-- State the theorem
theorem f_of_2_equals_5 : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_5_l3105_310537


namespace NUMINAMATH_CALUDE_range_of_m_l3105_310585

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - m ≥ 0

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m*x + 1 > 0

-- Theorem statement
theorem range_of_m (m : ℝ) :
  p m ∧ q m → -2 < m ∧ m ≤ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_m_l3105_310585


namespace NUMINAMATH_CALUDE_school_fundraising_admin_fee_percentage_l3105_310545

/-- Proves that the percentage deducted for administration fees is 2% --/
theorem school_fundraising_admin_fee_percentage 
  (johnson_amount : ℝ)
  (sutton_amount : ℝ)
  (rollin_amount : ℝ)
  (total_amount : ℝ)
  (remaining_amount : ℝ)
  (h1 : johnson_amount = 2300)
  (h2 : johnson_amount = 2 * sutton_amount)
  (h3 : rollin_amount = 8 * sutton_amount)
  (h4 : rollin_amount = total_amount / 3)
  (h5 : remaining_amount = 27048) :
  (total_amount - remaining_amount) / total_amount * 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_school_fundraising_admin_fee_percentage_l3105_310545


namespace NUMINAMATH_CALUDE_slope_of_l3_l3105_310582

/-- Line passing through two points -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Find the intersection point of two lines -/
def lineIntersection (l1 l2 : Line) : Point := sorry

/-- Calculate the slope between two points -/
def slopeBetweenPoints (p1 p2 : Point) : ℝ := sorry

theorem slope_of_l3 (l1 l2 l3 : Line) (A B C : Point) :
  l1.slope = 4/3 ∧ l1.yIntercept = 2/3 ∧
  pointOnLine A l1 ∧ A.x = -2 ∧ A.y = -3 ∧
  l2.slope = 0 ∧ l2.yIntercept = 2 ∧
  B = lineIntersection l1 l2 ∧
  pointOnLine A l3 ∧ pointOnLine C l3 ∧
  pointOnLine C l2 ∧
  l3.slope > 0 ∧
  triangleArea ⟨A, B, C⟩ = 5 →
  l3.slope = 5/6 := by sorry

end NUMINAMATH_CALUDE_slope_of_l3_l3105_310582


namespace NUMINAMATH_CALUDE_go_out_is_better_l3105_310577

/-- Represents the decision of the fishing boat -/
inductive Decision
| GoOut
| StayIn

/-- Represents the weather conditions -/
inductive Weather
| Good
| Bad

/-- The profit or loss for each scenario -/
def profit (d : Decision) (w : Weather) : ℝ :=
  match d, w with
  | Decision.GoOut, Weather.Good => 6000
  | Decision.GoOut, Weather.Bad => -8000
  | Decision.StayIn, _ => -1000

/-- The probability of each weather condition -/
def weather_prob (w : Weather) : ℝ :=
  match w with
  | Weather.Good => 0.6
  | Weather.Bad => 0.4

/-- The expected value of a decision -/
def expected_value (d : Decision) : ℝ :=
  (profit d Weather.Good * weather_prob Weather.Good) +
  (profit d Weather.Bad * weather_prob Weather.Bad)

/-- Theorem stating that going out to sea has a higher expected value -/
theorem go_out_is_better :
  expected_value Decision.GoOut > expected_value Decision.StayIn :=
by sorry

end NUMINAMATH_CALUDE_go_out_is_better_l3105_310577


namespace NUMINAMATH_CALUDE_distributor_profit_percentage_l3105_310554

/-- The commission rate of the online store -/
def commission_rate : ℚ := 1/5

/-- The price at which the distributor obtains the product from the producer -/
def producer_price : ℚ := 18

/-- The price observed by the buyer on the online store -/
def buyer_price : ℚ := 27

/-- The selling price of the distributor to the online store -/
def selling_price : ℚ := buyer_price / (1 + commission_rate)

/-- The profit made by the distributor per item -/
def profit : ℚ := selling_price - producer_price

/-- The profit percentage of the distributor -/
def profit_percentage : ℚ := profit / producer_price * 100

theorem distributor_profit_percentage :
  profit_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_distributor_profit_percentage_l3105_310554


namespace NUMINAMATH_CALUDE_max_constant_C_l3105_310543

theorem max_constant_C : ∃ (C : ℝ), C = Real.sqrt 2 ∧
  (∀ (x y : ℝ), x^2 + y^2 + 1 ≥ C*(x + y)) ∧
  (∀ (x y : ℝ), x^2 + y^2 + x*y + 1 ≥ C*(x + y)) ∧
  (∀ (C' : ℝ), C' > C →
    (∃ (x y : ℝ), x^2 + y^2 + 1 < C'*(x + y) ∨ x^2 + y^2 + x*y + 1 < C'*(x + y))) := by
  sorry

end NUMINAMATH_CALUDE_max_constant_C_l3105_310543


namespace NUMINAMATH_CALUDE_other_ticket_cost_l3105_310507

/-- Given a total of 29 tickets, with 11 tickets costing $9 each,
    and a total cost of $225 for all tickets,
    prove that the remaining tickets cost $7 each. -/
theorem other_ticket_cost (total_tickets : ℕ) (nine_dollar_tickets : ℕ) 
  (total_cost : ℕ) (h1 : total_tickets = 29) (h2 : nine_dollar_tickets = 11) 
  (h3 : total_cost = 225) : 
  (total_cost - nine_dollar_tickets * 9) / (total_tickets - nine_dollar_tickets) = 7 :=
by sorry

end NUMINAMATH_CALUDE_other_ticket_cost_l3105_310507


namespace NUMINAMATH_CALUDE_wire_service_reporters_l3105_310504

theorem wire_service_reporters (total_reporters : ℝ) 
  (local_politics_percentage : ℝ) (non_local_politics_percentage : ℝ) :
  local_politics_percentage = 18 / 100 →
  non_local_politics_percentage = 40 / 100 →
  total_reporters > 0 →
  (total_reporters - (total_reporters * local_politics_percentage / (1 - non_local_politics_percentage))) / total_reporters = 70 / 100 := by
  sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l3105_310504


namespace NUMINAMATH_CALUDE_tenth_place_is_unnamed_l3105_310565

/-- Represents a racer in the race --/
inductive Racer
| Eda
| Simon
| Jacob
| Naomi
| Cal
| Iris
| Unnamed

/-- Represents the finishing position of a racer --/
def Position := Fin 15

/-- The race results, mapping each racer to their position --/
def RaceResult := Racer → Position

def valid_race_result (result : RaceResult) : Prop :=
  (result Racer.Jacob).val + 4 = (result Racer.Eda).val
  ∧ (result Racer.Naomi).val = (result Racer.Simon).val + 1
  ∧ (result Racer.Jacob).val = (result Racer.Cal).val + 3
  ∧ (result Racer.Simon).val = (result Racer.Iris).val + 2
  ∧ (result Racer.Cal).val + 2 = (result Racer.Iris).val
  ∧ (result Racer.Naomi).val = 7

theorem tenth_place_is_unnamed (result : RaceResult) 
  (h : valid_race_result result) : 
  ∀ r : Racer, r ≠ Racer.Unnamed → (result r).val ≠ 10 := by
  sorry

end NUMINAMATH_CALUDE_tenth_place_is_unnamed_l3105_310565


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_l3105_310588

theorem polygon_interior_angle_sum (n : ℕ) (h : n * 36 = 360) :
  (n - 2) * 180 = 1440 :=
sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_l3105_310588


namespace NUMINAMATH_CALUDE_dogsled_race_time_difference_l3105_310556

/-- Proves that the difference in time taken to complete a 300-mile course between two teams,
    where one team's average speed is 5 miles per hour greater than the other team's speed
    of 20 miles per hour, is 3 hours. -/
theorem dogsled_race_time_difference :
  let course_length : ℝ := 300
  let team_b_speed : ℝ := 20
  let team_a_speed : ℝ := team_b_speed + 5
  let team_b_time : ℝ := course_length / team_b_speed
  let team_a_time : ℝ := course_length / team_a_speed
  team_b_time - team_a_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_dogsled_race_time_difference_l3105_310556


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3105_310558

theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3105_310558


namespace NUMINAMATH_CALUDE_sea_world_trip_savings_l3105_310569

def trip_cost (parking : ℕ) (entrance : ℕ) (meal : ℕ) (souvenirs : ℕ) (hotel : ℕ) : ℕ :=
  parking + entrance + meal + souvenirs + hotel

def gas_cost (distance : ℕ) (mpg : ℕ) (price_per_gallon : ℕ) : ℕ :=
  (2 * distance / mpg) * price_per_gallon

def additional_savings (total_cost : ℕ) (current_savings : ℕ) : ℕ :=
  total_cost - current_savings

theorem sea_world_trip_savings : 
  let current_savings : ℕ := 28
  let parking : ℕ := 10
  let entrance : ℕ := 55
  let meal : ℕ := 25
  let souvenirs : ℕ := 40
  let hotel : ℕ := 80
  let distance : ℕ := 165
  let mpg : ℕ := 30
  let price_per_gallon : ℕ := 3
  
  let total_trip_cost := trip_cost parking entrance meal souvenirs hotel
  let total_gas_cost := gas_cost distance mpg price_per_gallon
  let total_cost := total_trip_cost + total_gas_cost
  
  additional_savings total_cost current_savings = 215 := by
  sorry

end NUMINAMATH_CALUDE_sea_world_trip_savings_l3105_310569


namespace NUMINAMATH_CALUDE_bicycle_average_speed_l3105_310560

theorem bicycle_average_speed (total_distance : ℝ) (first_distance : ℝ) (second_distance : ℝ)
  (first_speed : ℝ) (second_speed : ℝ) (h1 : total_distance = 250)
  (h2 : first_distance = 100) (h3 : second_distance = 150)
  (h4 : first_speed = 20) (h5 : second_speed = 15)
  (h6 : total_distance = first_distance + second_distance) :
  (total_distance / (first_distance / first_speed + second_distance / second_speed)) =
  (250 : ℝ) / ((100 : ℝ) / 20 + (150 : ℝ) / 15) := by
  sorry

end NUMINAMATH_CALUDE_bicycle_average_speed_l3105_310560


namespace NUMINAMATH_CALUDE_abigail_typing_speed_l3105_310590

/-- The number of words Abigail can type in half an hour -/
def words_per_half_hour : ℕ := sorry

/-- The total length of the report in words -/
def total_report_length : ℕ := 1000

/-- The number of words Abigail has already written -/
def words_already_written : ℕ := 200

/-- The number of minutes Abigail needs to finish the report -/
def minutes_to_finish : ℕ := 80

theorem abigail_typing_speed :
  words_per_half_hour = 300 := by sorry

end NUMINAMATH_CALUDE_abigail_typing_speed_l3105_310590


namespace NUMINAMATH_CALUDE_prime_fraction_sum_of_reciprocals_l3105_310528

theorem prime_fraction_sum_of_reciprocals (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ (m : ℕ) (x y : ℕ+), 3 ≤ m ∧ m ≤ p - 2 ∧ (m : ℚ) / (p^2 : ℚ) = (1 : ℚ) / (x : ℚ) + (1 : ℚ) / (y : ℚ) :=
sorry

end NUMINAMATH_CALUDE_prime_fraction_sum_of_reciprocals_l3105_310528


namespace NUMINAMATH_CALUDE_same_color_probability_l3105_310509

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def total_plates : ℕ := red_plates + blue_plates
def selected_plates : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates selected_plates + Nat.choose blue_plates selected_plates) / 
  Nat.choose total_plates selected_plates = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3105_310509


namespace NUMINAMATH_CALUDE_fraction_simplification_fraction_value_at_one_l3105_310598

theorem fraction_simplification (x : ℤ) (h1 : -2 < x) (h2 : x < 2) (h3 : x ≠ 0) :
  (((x^2 - 1) / (x^2 + 2*x + 1)) / ((1 / (x + 1)) - 1)) = -(x - 1) / x :=
sorry

theorem fraction_value_at_one :
  (((1^2 - 1) / (1^2 + 2*1 + 1)) / ((1 / (1 + 1)) - 1)) = 0 :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_fraction_value_at_one_l3105_310598


namespace NUMINAMATH_CALUDE_solve_bracket_equation_l3105_310501

-- Define the bracket function
def bracket (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 + 1 else 2 * x + 1

-- State the theorem
theorem solve_bracket_equation :
  ∃ x : ℤ, (bracket 6) * (bracket x) = 28 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_bracket_equation_l3105_310501


namespace NUMINAMATH_CALUDE_victor_trips_l3105_310593

/-- Calculate the number of trips needed to carry a given number of trays -/
def tripsNeeded (trays : ℕ) (capacity : ℕ) : ℕ :=
  (trays + capacity - 1) / capacity

/-- The problem setup -/
def victorProblem : Prop :=
  let capacity := 6
  let table1 := 23
  let table2 := 5
  let table3 := 12
  let table4 := 18
  let table5 := 27
  let totalTrips := tripsNeeded table1 capacity + tripsNeeded table2 capacity +
                    tripsNeeded table3 capacity + tripsNeeded table4 capacity +
                    tripsNeeded table5 capacity
  totalTrips = 15

theorem victor_trips : victorProblem := by
  sorry

end NUMINAMATH_CALUDE_victor_trips_l3105_310593


namespace NUMINAMATH_CALUDE_li_ming_on_time_probability_l3105_310534

structure TransportationProbabilities where
  bike_prob : ℝ
  bus_prob : ℝ
  bike_on_time_prob : ℝ
  bus_on_time_prob : ℝ

def probability_on_time (p : TransportationProbabilities) : ℝ :=
  p.bike_prob * p.bike_on_time_prob + p.bus_prob * p.bus_on_time_prob

theorem li_ming_on_time_probability :
  ∀ (p : TransportationProbabilities),
    p.bike_prob = 0.7 →
    p.bus_prob = 0.3 →
    p.bike_on_time_prob = 0.9 →
    p.bus_on_time_prob = 0.8 →
    probability_on_time p = 0.87 := by
  sorry

end NUMINAMATH_CALUDE_li_ming_on_time_probability_l3105_310534


namespace NUMINAMATH_CALUDE_hall_seats_l3105_310587

theorem hall_seats (total_seats : ℕ) : 
  (total_seats : ℝ) / 2 = 300 → total_seats = 600 := by
  sorry

end NUMINAMATH_CALUDE_hall_seats_l3105_310587


namespace NUMINAMATH_CALUDE_triangle_determination_l3105_310514

/-- Represents the different combinations of triangle information --/
inductive TriangleInfo
  | SSS  -- Three sides
  | SAS  -- Two sides and included angle
  | ASA  -- Two angles and included side
  | SSA  -- Two sides and angle opposite one of them

/-- Predicate to determine if a given combination of triangle information can uniquely determine a triangle --/
def uniquely_determines_triangle (info : TriangleInfo) : Prop :=
  match info with
  | TriangleInfo.SSS => true
  | TriangleInfo.SAS => true
  | TriangleInfo.ASA => true
  | TriangleInfo.SSA => false

/-- Theorem stating which combinations of triangle information can uniquely determine a triangle --/
theorem triangle_determination :
  (uniquely_determines_triangle TriangleInfo.SSS) ∧
  (uniquely_determines_triangle TriangleInfo.SAS) ∧
  (uniquely_determines_triangle TriangleInfo.ASA) ∧
  ¬(uniquely_determines_triangle TriangleInfo.SSA) :=
by sorry

end NUMINAMATH_CALUDE_triangle_determination_l3105_310514


namespace NUMINAMATH_CALUDE_goldfish_cost_graph_l3105_310506

/-- Represents the cost function for buying goldfish -/
def cost (n : ℕ) : ℚ :=
  18 * n + 3

/-- Represents the set of points on the graph -/
def graph : Set (ℕ × ℚ) :=
  {p | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 20 ∧ p = (n, cost n)}

theorem goldfish_cost_graph :
  (∃ (S : Set (ℕ × ℚ)), S.Finite ∧ (∀ p ∈ S, ∃ q ∈ S, p ≠ q) ∧ S = graph) :=
sorry

end NUMINAMATH_CALUDE_goldfish_cost_graph_l3105_310506


namespace NUMINAMATH_CALUDE_cycle_iteration_equivalence_l3105_310530

/-- A function that represents the k-th iteration of f -/
def iterate (f : α → α) : ℕ → α → α
  | 0, x => x
  | n + 1, x => f (iterate f n x)

/-- The main theorem -/
theorem cycle_iteration_equivalence
  {α : Type*} (f : α → α) (x₀ : α) (s k : ℕ) :
  (∃ (n : ℕ), iterate f s x₀ = x₀) →  -- x₀ belongs to a cycle of length s
  (k % s = 0 ↔ iterate f k x₀ = x₀) :=
sorry

end NUMINAMATH_CALUDE_cycle_iteration_equivalence_l3105_310530


namespace NUMINAMATH_CALUDE_no_integer_roots_for_odd_coefficients_l3105_310553

theorem no_integer_roots_for_odd_coefficients (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) : 
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_for_odd_coefficients_l3105_310553


namespace NUMINAMATH_CALUDE_employee_count_l3105_310531

theorem employee_count :
  ∀ (E : ℕ) (M : ℝ),
    M = 0.99 * (E : ℝ) →
    M - 299.9999999999997 = 0.98 * (E : ℝ) →
    E = 30000 :=
by
  sorry

end NUMINAMATH_CALUDE_employee_count_l3105_310531


namespace NUMINAMATH_CALUDE_knights_in_gamma_quarter_l3105_310539

/-- Represents a resident of the town -/
inductive Resident
| Knight
| Liar

/-- The total number of residents in the town -/
def total_residents : ℕ := 200

/-- The total number of affirmative answers received -/
def total_affirmative_answers : ℕ := 430

/-- The number of affirmative answers received in quarter Γ -/
def gamma_quarter_affirmative_answers : ℕ := 119

/-- The number of affirmative answers a knight gives to every four questions -/
def knight_affirmative_rate : ℚ := 1/4

/-- The number of affirmative answers a liar gives to every four questions -/
def liar_affirmative_rate : ℚ := 3/4

/-- The total number of liars in the town -/
def total_liars : ℕ := (total_affirmative_answers - total_residents) / 2

theorem knights_in_gamma_quarter : 
  ∃ (k : ℕ), k = 4 ∧ 
  k ≤ gamma_quarter_affirmative_answers ∧
  (gamma_quarter_affirmative_answers - k : ℤ) = total_liars - (k - 4 : ℤ) ∧
  ∀ (other_quarter : ℕ), other_quarter ≠ gamma_quarter_affirmative_answers →
    (other_quarter : ℤ) - (total_residents - total_liars) > (total_residents - total_liars : ℤ) :=
sorry

end NUMINAMATH_CALUDE_knights_in_gamma_quarter_l3105_310539


namespace NUMINAMATH_CALUDE_binomial_square_constant_l3105_310508

theorem binomial_square_constant (a : ℝ) : 
  (∃ b c : ℝ, ∀ x, 9*x^2 - 27*x + a = (b*x + c)^2) → a = 20.25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l3105_310508


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3105_310529

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g : ℚ → ℚ := λ x => c * x^3 - 7 * x^2 + d * x - 8
  (g 2 = -8) ∧ (g (-3) = -80) → c = 107/7 ∧ d = -302/7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3105_310529


namespace NUMINAMATH_CALUDE_max_intersections_square_decagon_l3105_310549

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  vertices : Finset (ℝ × ℝ)
  edges : Finset ((ℝ × ℝ) × (ℝ × ℝ))
  convex : Bool
  planar : Bool

/-- Represents the number of sides in a polygon -/
def numSides (p : ConvexPolygon) : ℕ := p.edges.card

/-- Determines if one polygon is inscribed in another -/
def isInscribed (p₁ p₂ : ConvexPolygon) : Prop := sorry

/-- Counts the number of shared vertices between two polygons -/
def sharedVertices (p₁ p₂ : ConvexPolygon) : ℕ := sorry

/-- Counts the number of intersections between edges of two polygons -/
def countIntersections (p₁ p₂ : ConvexPolygon) : ℕ := sorry

theorem max_intersections_square_decagon (p₁ p₂ : ConvexPolygon) : 
  numSides p₁ = 4 →
  numSides p₂ = 10 →
  p₁.convex →
  p₂.convex →
  p₁.planar →
  p₂.planar →
  isInscribed p₁ p₂ →
  sharedVertices p₁ p₂ = 4 →
  countIntersections p₁ p₂ ≤ 8 ∧ 
  ∃ (q₁ q₂ : ConvexPolygon), 
    numSides q₁ = 4 ∧
    numSides q₂ = 10 ∧
    q₁.convex ∧
    q₂.convex ∧
    q₁.planar ∧
    q₂.planar ∧
    isInscribed q₁ q₂ ∧
    sharedVertices q₁ q₂ = 4 ∧
    countIntersections q₁ q₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_square_decagon_l3105_310549


namespace NUMINAMATH_CALUDE_exam_mean_score_l3105_310563

theorem exam_mean_score (SD : ℝ) :
  (∃ M : ℝ, (58 = M - 2 * SD) ∧ (98 = M + 3 * SD)) → 
  (∃ M : ℝ, (58 = M - 2 * SD) ∧ (98 = M + 3 * SD) ∧ M = 74) :=
by sorry

end NUMINAMATH_CALUDE_exam_mean_score_l3105_310563


namespace NUMINAMATH_CALUDE_thursday_rainfall_thursday_rainfall_proof_l3105_310519

/-- Calculates the total rainfall on Thursday given the rainfall patterns of the week --/
theorem thursday_rainfall (monday_rain : Real) (tuesday_decrease : Real) 
  (wednesday_increase_percent : Real) (thursday_decrease_percent : Real) 
  (thursday_additional_rain : Real) : Real :=
  let tuesday_rain := monday_rain - tuesday_decrease
  let wednesday_rain := tuesday_rain * (1 + wednesday_increase_percent)
  let thursday_rain_before_system := wednesday_rain * (1 - thursday_decrease_percent)
  let thursday_total_rain := thursday_rain_before_system + thursday_additional_rain
  thursday_total_rain

/-- Proves that the total rainfall on Thursday is 0.54 inches given the specific conditions --/
theorem thursday_rainfall_proof :
  thursday_rainfall 0.9 0.7 0.5 0.2 0.3 = 0.54 := by
  sorry

end NUMINAMATH_CALUDE_thursday_rainfall_thursday_rainfall_proof_l3105_310519


namespace NUMINAMATH_CALUDE_relic_age_conversion_l3105_310552

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

/-- The octal representation of the relic's age --/
def relic_age_octal : List Nat := [4, 6, 5, 7]

theorem relic_age_conversion :
  octal_to_decimal relic_age_octal = 3956 := by
  sorry

end NUMINAMATH_CALUDE_relic_age_conversion_l3105_310552


namespace NUMINAMATH_CALUDE_calculate_b_amount_l3105_310589

/-- Given a total amount and the ratio between two parts, calculate the second part -/
theorem calculate_b_amount (total : ℚ) (a b : ℚ) (h1 : a + b = total) (h2 : 2/3 * a = 1/2 * b) : 
  b = 691.43 := by
  sorry

end NUMINAMATH_CALUDE_calculate_b_amount_l3105_310589


namespace NUMINAMATH_CALUDE_island_length_calculation_l3105_310586

/-- Represents the dimensions of a rectangular island -/
structure IslandDimensions where
  width : ℝ
  length : ℝ
  perimeter : ℝ

/-- Theorem: An island with width 4 miles and perimeter 22 miles has a length of 7 miles -/
theorem island_length_calculation (island : IslandDimensions) 
    (h1 : island.width = 4)
    (h2 : island.perimeter = 22)
    (h3 : island.perimeter = 2 * (island.length + island.width)) : 
  island.length = 7 := by
  sorry


end NUMINAMATH_CALUDE_island_length_calculation_l3105_310586


namespace NUMINAMATH_CALUDE_hypotenuse_squared_of_complex_zeros_l3105_310591

-- Define the polynomial P(z)
def P (z : ℂ) : ℂ := z^3 - 2*z^2 + 2*z + 4

-- State the theorem
theorem hypotenuse_squared_of_complex_zeros (a b c : ℂ) :
  P a = 0 → P b = 0 → P c = 0 →
  Complex.abs a ^ 2 + Complex.abs b ^ 2 + Complex.abs c ^ 2 = 300 →
  (a - b).re * (c - b).re + (a - b).im * (c - b).im = 0 →
  (Complex.abs (b - c)) ^ 2 = 450 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_squared_of_complex_zeros_l3105_310591


namespace NUMINAMATH_CALUDE_right_triangle_sets_l3105_310562

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that among the given sets, only (6, 8, 10) forms a right triangle --/
theorem right_triangle_sets :
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle (Real.sqrt 7) 3 5 ∧
  is_right_triangle 6 8 10 ∧
  ¬ is_right_triangle 5 12 12 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l3105_310562


namespace NUMINAMATH_CALUDE_weight_problem_l3105_310547

/-- The weight problem -/
theorem weight_problem (student_weight sister_weight brother_weight : ℝ) : 
  (student_weight - 8 = sister_weight + brother_weight) →
  (brother_weight = sister_weight + 5) →
  (sister_weight + brother_weight = 180) →
  (student_weight = 188) :=
by
  sorry

end NUMINAMATH_CALUDE_weight_problem_l3105_310547


namespace NUMINAMATH_CALUDE_triangle_trig_max_value_l3105_310542

theorem triangle_trig_max_value (A C : ℝ) (h1 : 0 ≤ A ∧ A ≤ π) (h2 : 0 ≤ C ∧ C ≤ π) 
  (h3 : Real.sin A + Real.sin C = 3/2) :
  let t := 2 * Real.sin A * Real.sin C
  (∃ (x : ℝ), x = t * Real.sqrt ((9/4 - t) * (t - 1/4))) ∧
  (∀ (y : ℝ), y = t * Real.sqrt ((9/4 - t) * (t - 1/4)) → y ≤ 27 * Real.sqrt 7 / 64) ∧
  (∃ (z : ℝ), z = t * Real.sqrt ((9/4 - t) * (t - 1/4)) ∧ z = 27 * Real.sqrt 7 / 64) :=
by sorry

end NUMINAMATH_CALUDE_triangle_trig_max_value_l3105_310542


namespace NUMINAMATH_CALUDE_quadratic_vertex_on_line_quadratic_intersects_line_l3105_310520

/-- The quadratic function parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + m - 1

/-- The line y = x - 1 -/
def g (x : ℝ) : ℝ := x - 1

/-- The line y = x + b parameterized by b -/
def h (b : ℝ) (x : ℝ) : ℝ := x + b

/-- The vertex of a quadratic function ax^2 + bx + c is at (-b/(2a), f(-b/(2a))) -/
def vertex (m : ℝ) : ℝ × ℝ := (m, f m m)

theorem quadratic_vertex_on_line (m : ℝ) : 
  g (vertex m).1 = (vertex m).2 :=
sorry

theorem quadratic_intersects_line (m b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = h b x₁ ∧ f m x₂ = h b x₂) ↔ b > -5/4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_on_line_quadratic_intersects_line_l3105_310520


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3105_310597

theorem max_sum_of_factors (diamond delta : ℕ) : 
  diamond * delta = 36 → (∀ x y : ℕ, x * y = 36 → x + y ≤ diamond + delta) → diamond + delta = 37 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3105_310597


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3105_310511

/-- Given a hyperbola with equation x²/a - y²/2 = 1 and one asymptote 2x - y = 0, 
    prove that a = 1/2 -/
theorem hyperbola_asymptote (a : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 / a - y^2 / 2 = 1 → (2*x - y = 0 ∨ 2*x + y = 0)) : a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3105_310511


namespace NUMINAMATH_CALUDE_x_minus_y_equals_pi_over_three_l3105_310516

theorem x_minus_y_equals_pi_over_three (x y : Real) 
  (h1 : 0 < y) (h2 : y < x) (h3 : x < π)
  (h4 : Real.tan x * Real.tan y = 2)
  (h5 : Real.sin x * Real.sin y = 1/3) : 
  x - y = π/3 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_pi_over_three_l3105_310516


namespace NUMINAMATH_CALUDE_road_travel_cost_l3105_310548

/-- The cost of traveling two intersecting roads on a rectangular lawn -/
theorem road_travel_cost (lawn_length lawn_width road_width cost_per_sqm : ℕ) : 
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  cost_per_sqm = 3 → 
  (road_width * lawn_width + road_width * lawn_length - road_width * road_width) * cost_per_sqm = 3900 :=
by sorry

end NUMINAMATH_CALUDE_road_travel_cost_l3105_310548


namespace NUMINAMATH_CALUDE_cistern_fill_time_l3105_310579

/-- Represents the time to fill a cistern with two taps -/
def time_to_fill_cistern (fill_rate : ℚ) (empty_rate : ℚ) : ℚ :=
  1 / (fill_rate - empty_rate)

/-- Theorem: The time to fill the cistern is 12 hours -/
theorem cistern_fill_time :
  let fill_rate : ℚ := 1/6
  let empty_rate : ℚ := 1/12
  time_to_fill_cistern fill_rate empty_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l3105_310579


namespace NUMINAMATH_CALUDE_pentomino_tiling_l3105_310535

-- Define the pentomino types
inductive Pentomino
| UShaped
| CrossShaped

-- Define a function to check if a rectangle can be tiled
def canTile (width height : ℕ) : Prop :=
  ∃ (arrangement : ℕ → ℕ → Pentomino), 
    (∀ x y, x < width ∧ y < height → ∃ (px py : ℕ) (p : Pentomino), 
      arrangement px py = p ∧ 
      (px ≤ x ∧ x < px + 5) ∧ 
      (py ≤ y ∧ y < py + 5))

-- State the theorem
theorem pentomino_tiling (n : ℕ) :
  n > 1 ∧ canTile 15 n ↔ n ≠ 2 ∧ n ≠ 4 ∧ n ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_pentomino_tiling_l3105_310535


namespace NUMINAMATH_CALUDE_sum_of_four_digit_primes_and_multiples_of_three_l3105_310517

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def count_four_digit_primes : ℕ := sorry

def count_four_digit_multiples_of_three : ℕ := sorry

theorem sum_of_four_digit_primes_and_multiples_of_three :
  count_four_digit_primes + count_four_digit_multiples_of_three = 4061 := by sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_primes_and_multiples_of_three_l3105_310517


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l3105_310576

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l3105_310576


namespace NUMINAMATH_CALUDE_cube_root_of_three_times_five_to_seven_l3105_310580

theorem cube_root_of_three_times_five_to_seven (x : ℝ) :
  x = (5^7 + 5^7 + 5^7)^(1/3) → x = 3^(1/3) * 5^(7/3) := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_three_times_five_to_seven_l3105_310580


namespace NUMINAMATH_CALUDE_second_divisor_problem_l3105_310568

theorem second_divisor_problem : ∃ (D : ℕ+) (N : ℕ), N % 35 = 25 ∧ N % D = 4 ∧ D = 17 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l3105_310568


namespace NUMINAMATH_CALUDE_seven_people_round_table_l3105_310550

def factorial (n : ℕ) : ℕ := Nat.factorial n

def roundTableArrangements (n : ℕ) : ℕ := factorial (n - 1)

theorem seven_people_round_table :
  roundTableArrangements 7 = 720 := by
  sorry

end NUMINAMATH_CALUDE_seven_people_round_table_l3105_310550


namespace NUMINAMATH_CALUDE_bakers_cakes_l3105_310571

/-- Baker's cake problem -/
theorem bakers_cakes (initial_cakes : ℕ) : 
  (initial_cakes - 91 + 154 = initial_cakes + 63) →
  initial_cakes = 182 := by
  sorry

#check bakers_cakes

end NUMINAMATH_CALUDE_bakers_cakes_l3105_310571


namespace NUMINAMATH_CALUDE_wrapping_paper_area_is_8lh_l3105_310500

/-- Represents a rectangular box -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the area of wrapping paper needed for a given box -/
def wrappingPaperArea (box : Box) : ℝ :=
  8 * box.length * box.height

/-- Theorem stating that the area of wrapping paper needed is 8lh -/
theorem wrapping_paper_area_is_8lh (box : Box) :
  wrappingPaperArea box = 8 * box.length * box.height :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_is_8lh_l3105_310500


namespace NUMINAMATH_CALUDE_negation_inequality_statement_l3105_310541

theorem negation_inequality_statement :
  ¬(∀ (x : ℝ), x^2 + 1 > 0) ≠ (∃ (x : ℝ), x^2 + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_inequality_statement_l3105_310541


namespace NUMINAMATH_CALUDE_surface_polygon_angle_sum_sum_all_defects_l3105_310583

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  
/-- An m-gon on the surface of a polyhedron -/
structure SurfacePolygon (P : ConvexPolyhedron) where
  m : ℕ  -- number of sides
  -- Add other necessary fields here

/-- The defect of a polyhedral angle -/
def defect (P : ConvexPolyhedron) (v : ℝ × ℝ × ℝ) : ℝ := sorry

/-- The sum of angles of a surface polygon -/
def sumAngles (P : ConvexPolyhedron) (S : SurfacePolygon P) : ℝ := sorry

/-- The sum of defects of vertices inside a surface polygon -/
def sumDefectsInside (P : ConvexPolyhedron) (S : SurfacePolygon P) : ℝ := sorry

/-- The sum of defects of all vertices of a polyhedron -/
def sumAllDefects (P : ConvexPolyhedron) : ℝ := sorry

theorem surface_polygon_angle_sum (P : ConvexPolyhedron) (S : SurfacePolygon P) :
  sumAngles P S = 2 * Real.pi * (S.m - 2 : ℝ) + sumDefectsInside P S := by sorry

theorem sum_all_defects (P : ConvexPolyhedron) :
  sumAllDefects P = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_surface_polygon_angle_sum_sum_all_defects_l3105_310583


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3105_310578

/-- A quadratic function with vertex (2, 5) passing through (0, 0) has a = -5/4 --/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- quadratic function definition
  (2, 5) = (2, a * 2^2 + b * 2 + c) →     -- vertex condition
  (0, 0) = (0, a * 0^2 + b * 0 + c) →     -- point condition
  a = -5/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3105_310578


namespace NUMINAMATH_CALUDE_brendas_age_l3105_310594

theorem brendas_age (addison brenda janet : ℚ) 
  (h1 : addison = 4 * brenda)
  (h2 : janet = brenda + 8)
  (h3 : addison = janet + 2) :
  brenda = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_brendas_age_l3105_310594


namespace NUMINAMATH_CALUDE_flowchart_transformation_l3105_310557

def transform (a b c : ℕ) : ℕ × ℕ × ℕ :=
  (c, a, b)

theorem flowchart_transformation :
  transform 21 32 75 = (75, 21, 32) := by
  sorry

end NUMINAMATH_CALUDE_flowchart_transformation_l3105_310557


namespace NUMINAMATH_CALUDE_ellipse_condition_iff_l3105_310525

-- Define the condition
def condition (m n : ℝ) : Prop := m > n ∧ n > 0

-- Define what it means for the equation to represent an ellipse with foci on the y-axis
def is_ellipse_with_foci_on_y_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ m = 1 / (a^2) ∧ n = 1 / (b^2)

-- State the theorem
theorem ellipse_condition_iff (m n : ℝ) :
  condition m n ↔ is_ellipse_with_foci_on_y_axis m n := by
  sorry

end NUMINAMATH_CALUDE_ellipse_condition_iff_l3105_310525


namespace NUMINAMATH_CALUDE_julia_payment_l3105_310540

def snickers_price : ℝ := 1.5
def snickers_quantity : ℕ := 2
def mm_quantity : ℕ := 3
def change : ℝ := 8

def mm_price : ℝ := 2 * snickers_price

def total_cost : ℝ := snickers_price * snickers_quantity + mm_price * mm_quantity

theorem julia_payment : total_cost + change = 20 := by
  sorry

end NUMINAMATH_CALUDE_julia_payment_l3105_310540


namespace NUMINAMATH_CALUDE_female_employees_with_advanced_degrees_l3105_310546

theorem female_employees_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (total_advanced_degrees : ℕ) 
  (males_college_only : ℕ) 
  (h1 : total_employees = 160)
  (h2 : total_females = 90)
  (h3 : total_advanced_degrees = 80)
  (h4 : males_college_only = 40) :
  total_advanced_degrees - (total_employees - total_females - males_college_only) = 50 := by
  sorry

end NUMINAMATH_CALUDE_female_employees_with_advanced_degrees_l3105_310546


namespace NUMINAMATH_CALUDE_sum_of_coefficients_without_x_l3105_310522

theorem sum_of_coefficients_without_x (x y : ℝ) : 
  (fun x y => (1 - x - 5*y)^5) 0 1 = -1024 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_without_x_l3105_310522


namespace NUMINAMATH_CALUDE_hunter_has_ten_rats_l3105_310592

/-- The number of rats Hunter has -/
def hunter_rats : ℕ := sorry

/-- The number of rats Elodie has -/
def elodie_rats : ℕ := hunter_rats + 30

/-- The number of rats Kenia has -/
def kenia_rats : ℕ := 3 * (hunter_rats + elodie_rats)

/-- The total number of pets -/
def total_pets : ℕ := 200

theorem hunter_has_ten_rats :
  hunter_rats + elodie_rats + kenia_rats = total_pets →
  hunter_rats = 10 := by sorry

end NUMINAMATH_CALUDE_hunter_has_ten_rats_l3105_310592


namespace NUMINAMATH_CALUDE_square_field_area_l3105_310533

theorem square_field_area (side : ℝ) (h1 : 4 * side = 36) 
  (h2 : 6 * (side * side) = 6 * (2 * (4 * side) + 9)) : side * side = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l3105_310533


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3105_310564

theorem division_remainder_problem (dividend quotient divisor remainder : ℕ) : 
  dividend = 95 →
  quotient = 6 →
  divisor = 15 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3105_310564


namespace NUMINAMATH_CALUDE_cedarwood_earnings_theorem_l3105_310527

/-- Represents the data for each school's participation in the community project -/
structure SchoolData where
  name : String
  students : ℕ
  days : ℕ

/-- Calculates the total earnings for Cedarwood school given the project data -/
def cedarwoodEarnings (ashwood briarwood cedarwood : SchoolData) (totalPaid : ℚ) : ℚ :=
  let totalStudentDays := ashwood.students * ashwood.days + briarwood.students * briarwood.days + cedarwood.students * cedarwood.days
  let dailyWage := totalPaid / totalStudentDays
  dailyWage * (cedarwood.students * cedarwood.days)

/-- Theorem stating that Cedarwood school's earnings are 454.74 given the project conditions -/
theorem cedarwood_earnings_theorem (ashwood briarwood cedarwood : SchoolData) (totalPaid : ℚ) :
  ashwood.name = "Ashwood" ∧ ashwood.students = 9 ∧ ashwood.days = 4 ∧
  briarwood.name = "Briarwood" ∧ briarwood.students = 5 ∧ briarwood.days = 6 ∧
  cedarwood.name = "Cedarwood" ∧ cedarwood.students = 6 ∧ cedarwood.days = 8 ∧
  totalPaid = 1080 →
  cedarwoodEarnings ashwood briarwood cedarwood totalPaid = 454.74 := by
  sorry

#eval cedarwoodEarnings
  { name := "Ashwood", students := 9, days := 4 }
  { name := "Briarwood", students := 5, days := 6 }
  { name := "Cedarwood", students := 6, days := 8 }
  1080

end NUMINAMATH_CALUDE_cedarwood_earnings_theorem_l3105_310527


namespace NUMINAMATH_CALUDE_no_divisibility_pairs_l3105_310521

theorem no_divisibility_pairs : ¬∃ (m n : ℕ+), (m.val * n.val ∣ 3^m.val + 1) ∧ (m.val * n.val ∣ 3^n.val + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_divisibility_pairs_l3105_310521


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3105_310502

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_prod : a 2 * a 5 = -3/4)
  (h_sum : a 2 + a 3 + a 4 + a 5 = 5/4) :
  1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 = -5/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3105_310502
