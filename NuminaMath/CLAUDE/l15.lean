import Mathlib

namespace largest_gold_coins_l15_1559

theorem largest_gold_coins : 
  ∃ (n : ℕ), n = 146 ∧ 
  (∃ (k : ℕ), n = 13 * k + 3) ∧ 
  n < 150 ∧
  ∀ (m : ℕ), (∃ (j : ℕ), m = 13 * j + 3) → m < 150 → m ≤ n :=
by sorry

end largest_gold_coins_l15_1559


namespace seven_sum_problem_l15_1551

theorem seven_sum_problem :
  ∃ (S : Finset ℕ), (Finset.card S = 108) ∧ 
  (∀ n : ℕ, n ∈ S ↔ 
    ∃ a b c : ℕ, (7 * a + 77 * b + 777 * c = 7000) ∧ 
                 (a + 2 * b + 3 * c = n)) :=
sorry

end seven_sum_problem_l15_1551


namespace euler_totient_equation_solution_l15_1549

def euler_totient (n : ℕ) : ℕ := sorry

theorem euler_totient_equation_solution :
  ∀ n : ℕ, n > 0 → (n = euler_totient n + 402 ↔ n = 802 ∨ n = 546) := by
  sorry

end euler_totient_equation_solution_l15_1549


namespace saree_stripes_l15_1525

theorem saree_stripes (brown_stripes : ℕ) (gold_stripes : ℕ) (blue_stripes : ℕ) 
  (h1 : gold_stripes = 3 * brown_stripes)
  (h2 : blue_stripes = 5 * gold_stripes)
  (h3 : brown_stripes = 4) : 
  blue_stripes = 60 := by
  sorry

end saree_stripes_l15_1525


namespace consecutive_integers_cube_sum_l15_1567

theorem consecutive_integers_cube_sum : 
  ∃ (a : ℕ), 
    (a > 0) ∧ 
    ((a - 1) * a * (a + 1) * (a + 2) = 12 * (4 * a + 2)) ∧ 
    ((a - 1)^3 + a^3 + (a + 1)^3 + (a + 2)^3 = 224) := by
  sorry

end consecutive_integers_cube_sum_l15_1567


namespace exists_interest_rate_unique_interest_rate_l15_1560

/-- The interest rate that satisfies the given conditions --/
def interest_rate_equation (r : ℝ) : Prop :=
  1200 * ((1 + r/2)^2 - 1 - r) = 3

/-- Theorem stating that there exists an interest rate satisfying the equation --/
theorem exists_interest_rate : ∃ r : ℝ, interest_rate_equation r ∧ r > 0 ∧ r < 1 := by
  sorry

/-- Theorem stating that the interest rate solution is unique --/
theorem unique_interest_rate : ∃! r : ℝ, interest_rate_equation r ∧ r > 0 ∧ r < 1 := by
  sorry

end exists_interest_rate_unique_interest_rate_l15_1560


namespace min_positive_temperatures_l15_1542

theorem min_positive_temperatures (x : ℕ) (y : ℕ) : 
  x * (x - 1) = 110 → 
  y * (y - 1) + (x - y) * (x - 1 - y) = 50 → 
  y ≥ 5 :=
by sorry

end min_positive_temperatures_l15_1542


namespace five_hour_study_score_l15_1506

/-- Represents a student's test score based on study time -/
structure TestScore where
  studyTime : ℝ
  score : ℝ

/-- The maximum possible score on a test -/
def maxScore : ℝ := 100

/-- Calculates the potential score based on study time and effectiveness -/
def potentialScore (effectiveness : ℝ) (studyTime : ℝ) : ℝ :=
  effectiveness * studyTime

/-- Theorem: Given the conditions, the score for 5 hours of study is 100 -/
theorem five_hour_study_score :
  ∀ (effectiveness : ℝ),
  effectiveness > 0 →
  potentialScore effectiveness 2 = 80 →
  min (potentialScore effectiveness 5) maxScore = 100 := by
sorry

end five_hour_study_score_l15_1506


namespace expression_equality_l15_1526

theorem expression_equality : -1^2023 + |Real.sqrt 3 - 2| - 3 * Real.tan (π / 3) = 1 - 4 * Real.sqrt 3 := by
  sorry

end expression_equality_l15_1526


namespace min_value_sqrt_expression_l15_1564

theorem min_value_sqrt_expression (x : ℝ) (hx : x > 0) :
  4 * Real.sqrt x + 2 / Real.sqrt x ≥ 6 ∧
  ∃ y : ℝ, y > 0 ∧ 4 * Real.sqrt y + 2 / Real.sqrt y = 6 :=
by sorry

end min_value_sqrt_expression_l15_1564


namespace journey_speed_proof_l15_1519

/-- Proves that given a journey of 336 km completed in 15 hours, where the second half is traveled at 24 km/hr, the speed for the first half of the journey is 21 km/hr. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 336 →
  total_time = 15 →
  second_half_speed = 24 →
  let first_half_distance : ℝ := total_distance / 2
  let second_half_distance : ℝ := total_distance / 2
  let second_half_time : ℝ := second_half_distance / second_half_speed
  let first_half_time : ℝ := total_time - second_half_time
  let first_half_speed : ℝ := first_half_distance / first_half_time
  first_half_speed = 21 :=
by sorry

end journey_speed_proof_l15_1519


namespace percentage_problem_l15_1586

theorem percentage_problem (x : ℝ) (h : 150 = 250 / 100 * x) : x = 60 := by
  sorry

end percentage_problem_l15_1586


namespace angle_and_function_properties_l15_1509

-- Define the angle equivalence relation
def angle_equiv (a b : ℝ) : Prop := ∃ k : ℤ, a = b + k * 360

-- Define evenness for functions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Theorem statement
theorem angle_and_function_properties :
  (angle_equiv (-497) 2023) ∧
  (is_even_function (λ x => Real.sin ((2/3)*x - 7*Real.pi/2))) :=
by sorry

end angle_and_function_properties_l15_1509


namespace point_coordinates_l15_1590

/-- A point in a plane rectangular coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of a plane rectangular coordinate system -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates (p : Point) 
  (h1 : fourth_quadrant p) 
  (h2 : distance_to_x_axis p = 2) 
  (h3 : distance_to_y_axis p = 3) : 
  p = Point.mk 3 (-2) := by
  sorry

end point_coordinates_l15_1590


namespace janet_clarinet_lessons_l15_1548

/-- Proves that Janet takes 3 hours of clarinet lessons per week -/
theorem janet_clarinet_lessons :
  let clarinet_hourly_rate : ℕ := 40
  let piano_hourly_rate : ℕ := 28
  let piano_hours_per_week : ℕ := 5
  let weeks_per_year : ℕ := 52
  let annual_cost_difference : ℕ := 1040
  ∃ (clarinet_hours_per_week : ℕ),
    clarinet_hours_per_week = 3 ∧
    weeks_per_year * piano_hourly_rate * piano_hours_per_week - 
    weeks_per_year * clarinet_hourly_rate * clarinet_hours_per_week = 
    annual_cost_difference :=
by
  sorry

end janet_clarinet_lessons_l15_1548


namespace wednesday_thursday_miles_l15_1587

/-- Represents the mileage reimbursement rate in dollars per mile -/
def reimbursement_rate : ℚ := 36 / 100

/-- Represents the total reimbursement amount in dollars -/
def total_reimbursement : ℚ := 36

/-- Represents the miles driven on Monday -/
def monday_miles : ℕ := 18

/-- Represents the miles driven on Tuesday -/
def tuesday_miles : ℕ := 26

/-- Represents the miles driven on Friday -/
def friday_miles : ℕ := 16

/-- Theorem stating that the miles driven on Wednesday and Thursday combined is 40 -/
theorem wednesday_thursday_miles : 
  (total_reimbursement / reimbursement_rate : ℚ) - 
  (monday_miles + tuesday_miles + friday_miles : ℚ) = 40 := by sorry

end wednesday_thursday_miles_l15_1587


namespace postcard_selling_price_l15_1522

/-- Proves that the selling price per postcard is $10 --/
theorem postcard_selling_price 
  (initial_postcards : ℕ)
  (sold_postcards : ℕ)
  (new_postcard_price : ℚ)
  (final_postcard_count : ℕ)
  (h1 : initial_postcards = 18)
  (h2 : sold_postcards = initial_postcards / 2)
  (h3 : new_postcard_price = 5)
  (h4 : final_postcard_count = 36)
  : (sold_postcards : ℚ) * (final_postcard_count - initial_postcards) * new_postcard_price / sold_postcards = 10 := by
  sorry

end postcard_selling_price_l15_1522


namespace fourth_boy_payment_l15_1546

theorem fourth_boy_payment (a b c d : ℝ) : 
  a + b + c + d = 80 →
  a = (1/2) * (b + c + d) →
  b = (1/4) * (a + c + d) →
  c = (1/3) * (a + b + d) →
  d + 5 = 23 := by
  sorry

end fourth_boy_payment_l15_1546


namespace range_of_m_l15_1571

-- Define the linear function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x + m + 1

-- Define the condition for passing through first, second, and fourth quadrants
def passes_through_quadrants (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₄ : ℝ), 
    x₁ > 0 ∧ f m x₁ > 0 ∧  -- First quadrant
    x₂ < 0 ∧ f m x₂ > 0 ∧  -- Second quadrant
    x₄ > 0 ∧ f m x₄ < 0    -- Fourth quadrant

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  passes_through_quadrants m → -1 < m ∧ m < 3 :=
by sorry

end range_of_m_l15_1571


namespace servant_cash_compensation_l15_1508

/-- Calculates the cash compensation for a servant given the annual salary, work period, and value of a non-cash item received. -/
def servant_compensation (annual_salary : ℚ) (work_months : ℕ) (item_value : ℚ) : ℚ :=
  annual_salary * (work_months / 12 : ℚ) - item_value

/-- Proves that the cash compensation for the servant is 57.5 given the problem conditions. -/
theorem servant_cash_compensation : 
  servant_compensation 90 9 10 = 57.5 := by
  sorry

end servant_cash_compensation_l15_1508


namespace kylies_coins_l15_1593

/-- Kylie's coin collection problem -/
theorem kylies_coins (coins_from_piggy_bank coins_from_father coins_to_laura coins_left : ℕ) 
  (h1 : coins_from_piggy_bank = 15)
  (h2 : coins_from_father = 8)
  (h3 : coins_to_laura = 21)
  (h4 : coins_left = 15) :
  coins_from_piggy_bank + coins_from_father + coins_to_laura - coins_left = 13 := by
  sorry

#check kylies_coins

end kylies_coins_l15_1593


namespace equation_one_real_solution_l15_1544

theorem equation_one_real_solution :
  ∃! x : ℝ, (3 * x) / (x^2 + 2 * x + 4) + (4 * x) / (x^2 - 4 * x + 4) = 1 := by
  sorry

end equation_one_real_solution_l15_1544


namespace company_z_employees_l15_1569

/-- The number of employees in Company Z having birthdays on Wednesday -/
def wednesday_birthdays : ℕ := 12

/-- The number of employees in Company Z having birthdays on any day other than Wednesday -/
def other_day_birthdays : ℕ := 11

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem company_z_employees :
  let total_employees := wednesday_birthdays + (days_in_week - 1) * other_day_birthdays
  wednesday_birthdays > other_day_birthdays →
  total_employees = 78 := by
  sorry

end company_z_employees_l15_1569


namespace mrs_wong_valentines_l15_1534

def valentines_problem (initial : ℕ) (given_away : ℕ) : Prop :=
  initial - given_away = 22

theorem mrs_wong_valentines : valentines_problem 30 8 := by
  sorry

end mrs_wong_valentines_l15_1534


namespace factorization_proof_l15_1513

theorem factorization_proof (x y : ℝ) : x * (y - 1) + 4 * (1 - y) = (y - 1) * (x - 4) := by
  sorry

end factorization_proof_l15_1513


namespace video_recorder_price_l15_1574

/-- Given a wholesale cost, markup percentage, and discount percentage,
    calculate the final price after markup and discount. -/
def finalPrice (wholesaleCost markup discount : ℝ) : ℝ :=
  wholesaleCost * (1 + markup) * (1 - discount)

/-- Theorem stating that for a video recorder with a $200 wholesale cost,
    20% markup, and 25% employee discount, the final price is $180. -/
theorem video_recorder_price :
  finalPrice 200 0.20 0.25 = 180 := by
  sorry

end video_recorder_price_l15_1574


namespace range_of_a_and_m_l15_1550

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

theorem range_of_a_and_m :
  ∀ (a m : ℝ), A ∪ B a = A → A ∩ C m = C m →
  (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) :=
by sorry

end range_of_a_and_m_l15_1550


namespace prob_even_heads_is_17_25_l15_1543

/-- Represents an unfair coin where the probability of heads is 4 times the probability of tails -/
structure UnfairCoin where
  p_tails : ℝ
  p_heads : ℝ
  p_tails_pos : 0 < p_tails
  p_heads_pos : 0 < p_heads
  p_sum_one : p_tails + p_heads = 1
  p_heads_four_times : p_heads = 4 * p_tails

/-- The probability of getting an even number of heads when flipping the unfair coin twice -/
def prob_even_heads (c : UnfairCoin) : ℝ :=
  c.p_tails^2 + c.p_heads^2

/-- Theorem stating that the probability of getting an even number of heads
    when flipping the unfair coin twice is 17/25 -/
theorem prob_even_heads_is_17_25 (c : UnfairCoin) :
  prob_even_heads c = 17/25 := by
  sorry

end prob_even_heads_is_17_25_l15_1543


namespace square_side_increase_l15_1572

theorem square_side_increase (s : ℝ) (h : s > 0) :
  let new_area := s^2 * 1.5625
  let new_side := s * 1.25
  new_area = new_side^2 := by sorry

end square_side_increase_l15_1572


namespace intersection_condition_l15_1575

/-- The set M in ℝ² defined by y ≥ x² -/
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ≥ p.1^2}

/-- The set N in ℝ² defined by x² + (y-a)² ≤ 1 -/
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - a)^2 ≤ 1}

/-- Theorem stating the necessary and sufficient condition for M ∩ N = N -/
theorem intersection_condition (a : ℝ) : M ∩ N a = N a ↔ a ≥ 5/4 := by sorry

end intersection_condition_l15_1575


namespace binomial_prob_theorem_l15_1540

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- The probability mass function of a binomial distribution -/
def pmf (X : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose X.n k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

/-- Theorem: If X ~ B(10,p) with D(X) = 2.4 and P(X=4) > P(X=6), then p = 0.4 -/
theorem binomial_prob_theorem (X : BinomialRV) 
  (h_n : X.n = 10)
  (h_var : variance X = 2.4)
  (h_prob : pmf X 4 > pmf X 6) :
  X.p = 0.4 := by
  sorry

end binomial_prob_theorem_l15_1540


namespace probability_two_non_defective_pens_l15_1580

theorem probability_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 12) (h2 : defective_pens = 4) :
  let non_defective_pens := total_pens - defective_pens
  let prob_first := non_defective_pens / total_pens
  let prob_second := (non_defective_pens - 1) / (total_pens - 1)
  prob_first * prob_second = 14 / 33 := by
  sorry

end probability_two_non_defective_pens_l15_1580


namespace share_price_increase_l15_1570

theorem share_price_increase (P : ℝ) (h : P > 0) : 
  let first_quarter_price := P * 1.30
  let second_quarter_increase := 0.15384615384615374
  let second_quarter_price := first_quarter_price * (1 + second_quarter_increase)
  (second_quarter_price - P) / P = 0.50 := by sorry

end share_price_increase_l15_1570


namespace half_sum_sequence_common_ratio_l15_1539

/-- A geometric sequence where each term is half the sum of its next two terms -/
def HalfSumSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ a n = (a (n + 1) + a (n + 2)) / 2

/-- The common ratio of a geometric sequence -/
def CommonRatio (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem half_sum_sequence_common_ratio (a : ℕ → ℝ) (r : ℝ) :
  HalfSumSequence a → CommonRatio a r → r = 1 := by sorry

end half_sum_sequence_common_ratio_l15_1539


namespace max_consecutive_even_sum_l15_1532

/-- The sum of k consecutive even integers starting from 2n is 156 -/
def ConsecutiveEvenSum (n k : ℕ) : Prop :=
  2 * k * n + k * (k - 1) = 156

/-- The proposition that 4 is the maximum number of consecutive even integers summing to 156 -/
theorem max_consecutive_even_sum :
  (∃ n : ℕ, ConsecutiveEvenSum n 4) ∧
  (∀ k : ℕ, k > 4 → ¬∃ n : ℕ, ConsecutiveEvenSum n k) :=
sorry

end max_consecutive_even_sum_l15_1532


namespace union_M_N_equals_reals_l15_1500

-- Define the sets M and N
def M : Set ℝ := {x | x > 0}
def N : Set ℝ := {x | x^2 ≥ x}

-- State the theorem
theorem union_M_N_equals_reals : M ∪ N = Set.univ := by sorry

end union_M_N_equals_reals_l15_1500


namespace reims_to_chaumont_distance_l15_1552

/-- Represents a city in the polygon -/
inductive City
  | Chalons
  | Vitry
  | Chaumont
  | SaintQuentin
  | Reims

/-- Represents the distance between two cities -/
def distance (a b : City) : ℕ :=
  match a, b with
  | City.Chalons, City.Vitry => 30
  | City.Vitry, City.Chaumont => 80
  | City.Chaumont, City.SaintQuentin => 236
  | City.SaintQuentin, City.Reims => 86
  | City.Reims, City.Chalons => 40
  | _, _ => 0  -- For simplicity, we set other distances to 0

/-- The theorem stating the distance from Reims to Chaumont -/
theorem reims_to_chaumont_distance :
  distance City.Reims City.Chaumont = 150 :=
by sorry

end reims_to_chaumont_distance_l15_1552


namespace boxer_weight_loss_l15_1515

/-- Given a boxer's initial weight, monthly weight loss, and number of months until the fight,
    calculate the boxer's weight on the day of the fight. -/
def boxerFinalWeight (initialWeight monthlyLoss months : ℕ) : ℕ :=
  initialWeight - monthlyLoss * months

/-- Theorem stating that a boxer weighing 97 kg and losing 3 kg per month for 4 months
    will weigh 85 kg on the day of the fight. -/
theorem boxer_weight_loss : boxerFinalWeight 97 3 4 = 85 := by
  sorry

end boxer_weight_loss_l15_1515


namespace range_of_fraction_l15_1599

theorem range_of_fraction (a b : ℝ) 
  (ha : 0 < a ∧ a ≤ 2) 
  (hb : b ≥ 1)
  (hba : b ≤ a^2) :
  ∃ (t : ℝ), t = b / a ∧ 1/2 ≤ t ∧ t ≤ 2 :=
by sorry

end range_of_fraction_l15_1599


namespace three_numbers_sum_to_50_l15_1533

def number_list : List Nat := [21, 19, 30, 25, 3, 12, 9, 15, 6, 27]

theorem three_numbers_sum_to_50 :
  ∃ (a b c : Nat), a ∈ number_list ∧ b ∈ number_list ∧ c ∈ number_list ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = 50 :=
by sorry

end three_numbers_sum_to_50_l15_1533


namespace cupcake_packages_l15_1517

theorem cupcake_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) :
  initial_cupcakes = 39 →
  eaten_cupcakes = 21 →
  cupcakes_per_package = 3 →
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package = 6 :=
by sorry

end cupcake_packages_l15_1517


namespace frank_planted_two_seeds_per_orange_l15_1598

/-- The number of oranges Betty picked -/
def betty_oranges : ℕ := 15

/-- The number of oranges Bill picked -/
def bill_oranges : ℕ := 12

/-- The number of oranges Frank picked -/
def frank_oranges : ℕ := 3 * (betty_oranges + bill_oranges)

/-- The number of oranges each tree contains -/
def oranges_per_tree : ℕ := 5

/-- The total number of oranges Philip can pick -/
def philip_total_oranges : ℕ := 810

/-- The number of seeds Frank planted from each of his oranges -/
def seeds_per_orange : ℕ := philip_total_oranges / oranges_per_tree / frank_oranges

theorem frank_planted_two_seeds_per_orange : seeds_per_orange = 2 := by
  sorry

end frank_planted_two_seeds_per_orange_l15_1598


namespace complex_power_four_l15_1524

theorem complex_power_four (i : ℂ) : i * i = -1 → (1 - i)^4 = -4 := by sorry

end complex_power_four_l15_1524


namespace certain_number_proof_l15_1520

theorem certain_number_proof : ∃ n : ℕ, n = 213 * 16 ∧ n = 3408 := by
  -- Given condition: 0.016 * 2.13 = 0.03408
  have h : (0.016 : ℝ) * 2.13 = 0.03408 := by sorry
  
  -- Proof that 213 * 16 = 3408
  sorry


end certain_number_proof_l15_1520


namespace stream_speed_l15_1555

/-- Proves that given a boat with a speed of 22 km/hr in still water, 
    traveling 54 km downstream in 2 hours, the speed of the stream is 5 km/hr. -/
theorem stream_speed 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 22)
  (h2 : distance = 54)
  (h3 : time = 2)
  : ∃ (stream_speed : ℝ), 
    distance = (boat_speed + stream_speed) * time ∧ 
    stream_speed = 5 :=
by
  sorry

end stream_speed_l15_1555


namespace tank_fraction_problem_l15_1562

/-- The problem of determining the fraction of the first tank's capacity that is filled. -/
theorem tank_fraction_problem (tank1_capacity tank2_capacity tank3_capacity : ℚ)
  (tank2_fill_fraction tank3_fill_fraction : ℚ)
  (total_water : ℚ) :
  tank1_capacity = 7000 →
  tank2_capacity = 5000 →
  tank3_capacity = 3000 →
  tank2_fill_fraction = 4/5 →
  tank3_fill_fraction = 1/2 →
  total_water = 10850 →
  total_water = tank1_capacity * (107/140) + tank2_capacity * tank2_fill_fraction + tank3_capacity * tank3_fill_fraction :=
by sorry

end tank_fraction_problem_l15_1562


namespace problem_3_l15_1579

theorem problem_3 : (-48) / ((-2)^3) - (-25) * (-4) + (-2)^3 = -102 := by
  sorry

end problem_3_l15_1579


namespace cemc_employee_change_l15_1507

/-- The net change in employees for Canadian Excellent Mathematics Corporation in 2018 -/
theorem cemc_employee_change (t : ℕ) (h : t = 120) : 
  (((t : ℚ) * (1 + 0.25) + (40 : ℚ) * (1 - 0.35)) - (t + 40 : ℚ)).floor = 16 := by
  sorry

end cemc_employee_change_l15_1507


namespace sum_denominator_power_of_two_l15_1536

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def sum_series : ℕ → ℚ
  | 0 => 0
  | n + 1 => sum_series n + (double_factorial (2 * (n + 1) - 1) : ℚ) / (double_factorial (2 * (n + 1)) : ℚ)

theorem sum_denominator_power_of_two : 
  ∃ (numerator : ℕ), sum_series 11 = (numerator : ℚ) / 2^8 := by sorry

end sum_denominator_power_of_two_l15_1536


namespace percentage_commutation_l15_1565

theorem percentage_commutation (x : ℝ) : 
  (0.4 * (0.3 * x) = 24) → (0.3 * (0.4 * x) = 24) := by
  sorry

end percentage_commutation_l15_1565


namespace baseball_game_attendance_l15_1584

theorem baseball_game_attendance (total : ℕ) (first_team_percent : ℚ) (second_team_percent : ℚ) 
  (h1 : total = 50)
  (h2 : first_team_percent = 40 / 100)
  (h3 : second_team_percent = 34 / 100) :
  total - (total * first_team_percent).floor - (total * second_team_percent).floor = 13 := by
  sorry

end baseball_game_attendance_l15_1584


namespace integral_exp_plus_2x_equals_e_l15_1595

theorem integral_exp_plus_2x_equals_e :
  ∫ x in (0:ℝ)..1, (Real.exp x + 2 * x) = Real.exp 1 - 1 := by
  sorry

end integral_exp_plus_2x_equals_e_l15_1595


namespace largest_perimeter_l15_1510

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ

/-- Checks if the given lengths can form a valid triangle --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem: The largest possible perimeter of a triangle with sides 7, 9, and an integer y is 31 --/
theorem largest_perimeter :
  ∀ y : ℕ,
    is_valid_triangle ⟨7, 9, y⟩ →
    perimeter ⟨7, 9, y⟩ ≤ 31 ∧
    ∃ (y' : ℕ), is_valid_triangle ⟨7, 9, y'⟩ ∧ perimeter ⟨7, 9, y'⟩ = 31 :=
by sorry


end largest_perimeter_l15_1510


namespace parabola_hyperbola_tangent_l15_1547

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The hyperbola equation -/
def hyperbola (m : ℝ) (x y : ℝ) : Prop := y^2 - m*x^2 = 1

/-- The tangency condition -/
def are_tangent (m : ℝ) : Prop :=
  ∃ x : ℝ, (hyperbola m x (parabola x)) ∧
    (∀ x' : ℝ, x' ≠ x → ¬(hyperbola m x' (parabola x')))

theorem parabola_hyperbola_tangent :
  are_tangent 1 := by sorry

end parabola_hyperbola_tangent_l15_1547


namespace original_amount_is_1160_l15_1578

/-- Given an initial principal, time period, and interest rates, calculate the final amount using simple interest. -/
def simple_interest_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Prove that under the given conditions, the amount after 3 years at the original interest rate is $1160. -/
theorem original_amount_is_1160 
  (principal : ℝ) 
  (original_rate : ℝ) 
  (time : ℝ) 
  (h_principal : principal = 800) 
  (h_time : time = 3) 
  (h_increased_amount : simple_interest_amount principal (original_rate + 0.03) time = 992) :
  simple_interest_amount principal original_rate time = 1160 := by
sorry

end original_amount_is_1160_l15_1578


namespace sphere_volume_from_cylinder_volume_l15_1530

/-- The volume of a sphere with the same radius as a cylinder of volume 72π -/
theorem sphere_volume_from_cylinder_volume (r : ℝ) (h : ℝ) :
  (π * r^2 * h = 72 * π) →
  ((4 / 3) * π * r^3 = 48 * π) :=
by sorry

end sphere_volume_from_cylinder_volume_l15_1530


namespace sum_of_multiples_l15_1502

theorem sum_of_multiples (a b : ℤ) (ha : 6 ∣ a) (hb : 9 ∣ b) : 3 ∣ (a + b) := by
  sorry

end sum_of_multiples_l15_1502


namespace zero_in_interval_l15_1568

-- Define the function f(x) = x³ + x - 3
def f (x : ℝ) : ℝ := x^3 + x - 3

-- Theorem statement
theorem zero_in_interval : ∃ x : ℝ, x > 1 ∧ x < 2 ∧ f x = 0 := by
  sorry

end zero_in_interval_l15_1568


namespace unique_perfect_square_l15_1558

theorem unique_perfect_square (n : ℕ+) : n^2 - 19*n - 99 = m^2 ↔ n = 199 :=
  sorry

end unique_perfect_square_l15_1558


namespace plant_growth_theorem_l15_1514

structure PlantType where
  seedsPerPacket : ℕ
  growthRate : ℕ
  initialPackets : ℕ

def totalPlants (p : PlantType) : ℕ :=
  p.seedsPerPacket * p.initialPackets

def additionalPacketsNeeded (p : PlantType) (targetPlants : ℕ) : ℕ :=
  max 0 ((targetPlants - totalPlants p + p.seedsPerPacket - 1) / p.seedsPerPacket)

def growthTime (p : PlantType) (targetPlants : ℕ) : ℕ :=
  p.growthRate * max 0 (targetPlants - totalPlants p)

theorem plant_growth_theorem (targetPlants : ℕ) 
  (typeA typeB typeC : PlantType)
  (h1 : typeA = { seedsPerPacket := 3, growthRate := 5, initialPackets := 2 })
  (h2 : typeB = { seedsPerPacket := 6, growthRate := 7, initialPackets := 3 })
  (h3 : typeC = { seedsPerPacket := 9, growthRate := 4, initialPackets := 3 })
  (h4 : targetPlants = 12) : 
  additionalPacketsNeeded typeA targetPlants = 2 ∧ 
  growthTime typeA targetPlants = 5 ∧
  additionalPacketsNeeded typeB targetPlants = 0 ∧
  additionalPacketsNeeded typeC targetPlants = 0 := by
  sorry

end plant_growth_theorem_l15_1514


namespace sqrt_equation_difference_l15_1501

theorem sqrt_equation_difference (a b : ℕ+) 
  (h1 : Real.sqrt 18 = (a : ℝ) * Real.sqrt 2) 
  (h2 : Real.sqrt 8 = 2 * Real.sqrt (b : ℝ)) : 
  (a : ℤ) - (b : ℤ) = 1 := by
  sorry

end sqrt_equation_difference_l15_1501


namespace total_cakes_served_l15_1557

theorem total_cakes_served (lunch_cakes dinner_cakes : ℕ) 
  (h1 : lunch_cakes = 6) 
  (h2 : dinner_cakes = 9) : 
  lunch_cakes + dinner_cakes = 15 := by
sorry

end total_cakes_served_l15_1557


namespace min_value_theorem_l15_1585

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a + 8) * x + a^2 + a - 12

-- Define the theorem
theorem min_value_theorem (a : ℝ) (h1 : a < 0) 
  (h2 : f a (a^2 - 4) = f a (2*a - 8)) :
  ∀ n : ℕ+, (f a n - 4*a) / (n + 1) ≥ 35/8 ∧ 
  ∃ n : ℕ+, (f a n - 4*a) / (n + 1) = 35/8 := by
sorry


end min_value_theorem_l15_1585


namespace different_orders_eq_120_l15_1523

/-- The number of ways to arrange n elements. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of students who won awards. -/
def total_students : ℕ := 6

/-- The number of students whose order is fixed. -/
def fixed_order_students : ℕ := 3

/-- The number of different orders for all students to go on stage. -/
def different_orders : ℕ := permutations total_students / permutations fixed_order_students

theorem different_orders_eq_120 : different_orders = 120 := by
  sorry

end different_orders_eq_120_l15_1523


namespace football_playtime_l15_1516

def total_playtime_hours : ℝ := 1.5
def basketball_playtime_minutes : ℕ := 60

theorem football_playtime (total_playtime_minutes : ℕ) 
  (h1 : total_playtime_minutes = Int.floor (total_playtime_hours * 60)) 
  (h2 : total_playtime_minutes ≥ basketball_playtime_minutes) : 
  total_playtime_minutes - basketball_playtime_minutes = 30 := by
  sorry

end football_playtime_l15_1516


namespace rectangle_area_preservation_l15_1582

theorem rectangle_area_preservation (original_length original_width : ℝ) 
  (h_length : original_length = 140)
  (h_width : original_width = 40)
  (length_increase_percent : ℝ) 
  (h_increase : length_increase_percent = 30) :
  let new_length := original_length * (1 + length_increase_percent / 100)
  let new_width := (original_length * original_width) / new_length
  let width_decrease_percent := (original_width - new_width) / original_width * 100
  ∃ ε > 0, abs (width_decrease_percent - 23.08) < ε :=
sorry

end rectangle_area_preservation_l15_1582


namespace power_equality_l15_1573

theorem power_equality (n b : ℝ) : n = 2^(1/4) → n^b = 16 → b = 16 := by
  sorry

end power_equality_l15_1573


namespace right_triangle_sin_I_l15_1538

theorem right_triangle_sin_I (G H I : Real) :
  -- GHI is a right triangle with ∠G = 90°
  G + H + I = Real.pi →
  G = Real.pi / 2 →
  -- sin H = 3/5
  Real.sin H = 3 / 5 →
  -- Prove: sin I = 4/5
  Real.sin I = 4 / 5 := by
sorry

end right_triangle_sin_I_l15_1538


namespace unique_solution_cube_difference_square_l15_1563

theorem unique_solution_cube_difference_square (x y z : ℕ+) : 
  (x.val : ℤ)^3 - (y.val : ℤ)^3 = (z.val : ℤ)^2 →
  Nat.Prime y.val →
  ¬(3 ∣ z.val) →
  ¬(y.val ∣ z.val) →
  x = 8 ∧ y = 7 ∧ z = 13 :=
by sorry

end unique_solution_cube_difference_square_l15_1563


namespace unique_solution_system_l15_1591

theorem unique_solution_system (x y : ℚ) : 
  (3 * x + 2 * y = 7 ∧ 6 * x - 5 * y = 4) ↔ (x = 43/27 ∧ y = 10/9) :=
by sorry

end unique_solution_system_l15_1591


namespace jacob_breakfast_calories_l15_1588

theorem jacob_breakfast_calories 
  (daily_limit : ℕ) 
  (lunch_calories : ℕ) 
  (dinner_calories : ℕ) 
  (exceeded_calories : ℕ) 
  (h1 : daily_limit = 1800)
  (h2 : lunch_calories = 900)
  (h3 : dinner_calories = 1100)
  (h4 : exceeded_calories = 600) :
  daily_limit + exceeded_calories - (lunch_calories + dinner_calories) = 400 := by
sorry

end jacob_breakfast_calories_l15_1588


namespace expected_rainfall_is_50_4_l15_1576

/-- Weather forecast probabilities and rainfall amounts -/
structure WeatherForecast where
  days : ℕ
  prob_sun : ℝ
  prob_rain_3 : ℝ
  prob_rain_8 : ℝ
  amount_rain_3 : ℝ
  amount_rain_8 : ℝ

/-- Expected total rainfall over the forecast period -/
def expectedTotalRainfall (forecast : WeatherForecast) : ℝ :=
  forecast.days * (forecast.prob_rain_3 * forecast.amount_rain_3 + 
                   forecast.prob_rain_8 * forecast.amount_rain_8)

/-- Theorem: The expected total rainfall for the given forecast is 50.4 inches -/
theorem expected_rainfall_is_50_4 (forecast : WeatherForecast) 
  (h1 : forecast.days = 14)
  (h2 : forecast.prob_sun = 0.3)
  (h3 : forecast.prob_rain_3 = 0.4)
  (h4 : forecast.prob_rain_8 = 0.3)
  (h5 : forecast.amount_rain_3 = 3)
  (h6 : forecast.amount_rain_8 = 8)
  (h7 : forecast.prob_sun + forecast.prob_rain_3 + forecast.prob_rain_8 = 1) :
  expectedTotalRainfall forecast = 50.4 := by
  sorry

#eval expectedTotalRainfall { 
  days := 14, 
  prob_sun := 0.3, 
  prob_rain_3 := 0.4, 
  prob_rain_8 := 0.3, 
  amount_rain_3 := 3, 
  amount_rain_8 := 8 
}

end expected_rainfall_is_50_4_l15_1576


namespace knicks_knacks_knocks_conversion_l15_1531

/-- Given the conversions between knicks, knacks, and knocks, 
    prove that 80 knocks is equal to 192 knicks. -/
theorem knicks_knacks_knocks_conversion : 
  ∀ (knicks knacks knocks : ℚ),
    (9 * knicks = 3 * knacks) →
    (4 * knacks = 5 * knocks) →
    (80 * knocks = 192 * knicks) :=
by
  sorry

end knicks_knacks_knocks_conversion_l15_1531


namespace wang_yue_more_stable_l15_1503

def li_na_scores : List ℝ := [80, 70, 90, 70]
def wang_yue_scores (a : ℝ) : List ℝ := [80, a, 70, 90]

def median (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem wang_yue_more_stable (a : ℝ) :
  a ≥ 70 →
  median li_na_scores + 5 = median (wang_yue_scores a) →
  variance (wang_yue_scores a) < variance li_na_scores :=
sorry

end wang_yue_more_stable_l15_1503


namespace train_length_l15_1577

theorem train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 300 →
  crossing_time = 45 →
  train_speed = 47.99999999999999 →
  (train_speed * crossing_time) - bridge_length = 1860 :=
by sorry

end train_length_l15_1577


namespace least_positive_value_cubic_equation_l15_1527

/-- The least positive integer value of a cubic equation with prime number constraints -/
theorem least_positive_value_cubic_equation (x y z w : ℕ) : 
  Prime x → Prime y → Prime z → Prime w →
  x + y + z + w < 50 →
  (∀ a b c d : ℕ, Prime a → Prime b → Prime c → Prime d → 
    a + b + c + d < 50 → 
    24 * a^3 + 16 * b^3 - 7 * c^3 + 5 * d^3 ≥ 24 * x^3 + 16 * y^3 - 7 * z^3 + 5 * w^3) →
  24 * x^3 + 16 * y^3 - 7 * z^3 + 5 * w^3 = 1464 := by
  sorry

end least_positive_value_cubic_equation_l15_1527


namespace lees_friend_money_l15_1545

/-- 
Given:
- Lee had $10
- The total cost of the meal was $15 (including tax)
- They received $3 in change
- The total amount paid was $18

Prove that Lee's friend had $8 initially.
-/
theorem lees_friend_money (lee_money : ℕ) (meal_cost : ℕ) (change : ℕ) (total_paid : ℕ)
  (h1 : lee_money = 10)
  (h2 : meal_cost = 15)
  (h3 : change = 3)
  (h4 : total_paid = 18)
  : total_paid - lee_money = 8 := by
  sorry

end lees_friend_money_l15_1545


namespace roberta_record_listening_time_l15_1537

theorem roberta_record_listening_time :
  let x : ℕ := 8  -- initial number of records
  let y : ℕ := 12 -- additional records received
  let z : ℕ := 30 -- records bought
  let t : ℕ := 2  -- time needed to listen to each record in days
  (x + y + z) * t = 100 := by sorry

end roberta_record_listening_time_l15_1537


namespace monthly_donation_proof_l15_1583

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total annual donation in dollars -/
def annual_donation : ℕ := 17436

/-- The monthly donation in dollars -/
def monthly_donation : ℕ := annual_donation / months_in_year

theorem monthly_donation_proof : monthly_donation = 1453 := by
  sorry

end monthly_donation_proof_l15_1583


namespace tommys_estimate_l15_1505

theorem tommys_estimate (x y ε : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : ε > 0) :
  (x + ε) - (y - ε) > x - y := by
  sorry

end tommys_estimate_l15_1505


namespace boat_speed_in_still_water_l15_1581

/-- Proves that the speed of a boat in still water is 22 km/hr, given that it travels 54 km downstream in 2 hours with a stream speed of 5 km/hr. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 54)
  (h3 : downstream_time = 2)
  : ∃ (still_water_speed : ℝ),
    still_water_speed = 22 ∧
    downstream_distance = (still_water_speed + stream_speed) * downstream_time :=
by
  sorry

end boat_speed_in_still_water_l15_1581


namespace smallest_possible_d_l15_1511

theorem smallest_possible_d (c d : ℝ) : 
  (1 < c) → 
  (c < d) → 
  (1 + c ≤ d) → 
  (1 / c + 1 / d ≤ 1) → 
  d ≥ (3 + Real.sqrt 5) / 2 :=
by sorry

end smallest_possible_d_l15_1511


namespace women_doubles_tournament_handshakes_l15_1597

/-- The number of handshakes in a women's doubles tennis tournament -/
def num_handshakes (num_teams : ℕ) (team_size : ℕ) : ℕ :=
  let total_players := num_teams * team_size
  let handshakes_per_player := total_players - team_size
  (total_players * handshakes_per_player) / 2

theorem women_doubles_tournament_handshakes :
  num_handshakes 4 2 = 24 := by
  sorry

end women_doubles_tournament_handshakes_l15_1597


namespace right_triangle_with_constraint_l15_1512

-- Define the triangle sides
def side1 (p q : ℝ) : ℝ := p
def side2 (p q : ℝ) : ℝ := p + q
def side3 (p q : ℝ) : ℝ := p + 2*q

-- Define the conditions
def is_right_triangle (p q : ℝ) : Prop :=
  (side3 p q)^2 = (side1 p q)^2 + (side2 p q)^2

def longest_side_constraint (p q : ℝ) : Prop :=
  side3 p q ≤ 12

-- Theorem statement
theorem right_triangle_with_constraint :
  ∃ (p q : ℝ),
    is_right_triangle p q ∧
    longest_side_constraint p q ∧
    p = (1 + Real.sqrt 7) / 2 ∧
    q = 1 :=
by sorry

end right_triangle_with_constraint_l15_1512


namespace john_sells_20_woodburnings_l15_1594

/-- The number of woodburnings John sells -/
def num_woodburnings : ℕ := 20

/-- The selling price of each woodburning in dollars -/
def selling_price : ℕ := 15

/-- The cost of wood in dollars -/
def wood_cost : ℕ := 100

/-- John's profit in dollars -/
def profit : ℕ := 200

/-- Theorem stating that the number of woodburnings John sells is 20 -/
theorem john_sells_20_woodburnings :
  num_woodburnings = 20 ∧
  selling_price * num_woodburnings = wood_cost + profit := by
  sorry

end john_sells_20_woodburnings_l15_1594


namespace sphere_surface_area_ratio_l15_1553

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 1 / 27) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 9 := by
  sorry

end sphere_surface_area_ratio_l15_1553


namespace power_division_nineteen_l15_1535

theorem power_division_nineteen : 19^12 / 19^10 = 361 := by
  sorry

end power_division_nineteen_l15_1535


namespace price_equation_system_l15_1504

/-- Represents the price of a basketball in yuan -/
def basketball_price : ℝ := sorry

/-- Represents the price of a soccer ball in yuan -/
def soccer_ball_price : ℝ := sorry

/-- The total cost of 3 basketballs and 4 soccer balls is 330 yuan -/
axiom total_cost : 3 * basketball_price + 4 * soccer_ball_price = 330

/-- The price of a basketball is 5 yuan less than the price of a soccer ball -/
axiom price_difference : basketball_price = soccer_ball_price - 5

/-- The system of equations accurately represents the given conditions -/
theorem price_equation_system : 
  (3 * basketball_price + 4 * soccer_ball_price = 330) ∧ 
  (basketball_price = soccer_ball_price - 5) :=
by sorry

end price_equation_system_l15_1504


namespace factorial_equation_sum_l15_1528

theorem factorial_equation_sum : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, ∃ k l : ℕ, n.factorial / 2 = k.factorial + l.factorial) ∧
  (∀ n : ℕ, n ∉ S → ¬∃ k l : ℕ, n.factorial / 2 = k.factorial + l.factorial) ∧
  S.sum id = 10 := by
  sorry

end factorial_equation_sum_l15_1528


namespace words_in_page_l15_1554

/-- The number of words Tom can type per minute -/
def words_per_minute : ℕ := 90

/-- The number of minutes it takes Tom to type 10 pages -/
def minutes_for_ten_pages : ℕ := 50

/-- The number of pages Tom types in the given time -/
def number_of_pages : ℕ := 10

/-- Calculates the number of words in a page -/
def words_per_page : ℕ := (words_per_minute * minutes_for_ten_pages) / number_of_pages

/-- Theorem stating that there are 450 words in a page -/
theorem words_in_page : words_per_page = 450 := by
  sorry

end words_in_page_l15_1554


namespace min_green_surface_fraction_l15_1592

/-- Represents a cube with given edge length -/
structure Cube where
  edge : ℕ
  deriving Repr

/-- Represents the composition of a large cube -/
structure CubeComposition where
  large_cube : Cube
  small_cube : Cube
  blue_count : ℕ
  green_count : ℕ
  deriving Repr

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℕ := 6 * c.edge^2

/-- Calculates the volume of a cube -/
def volume (c : Cube) : ℕ := c.edge^3

/-- Theorem: Minimum green surface area fraction -/
theorem min_green_surface_fraction (cc : CubeComposition) 
  (h1 : cc.large_cube.edge = 4)
  (h2 : cc.small_cube.edge = 1)
  (h3 : volume cc.large_cube = cc.blue_count + cc.green_count)
  (h4 : cc.blue_count = 50)
  (h5 : cc.green_count = 14) :
  (6 : ℚ) / surface_area cc.large_cube = 1/16 := by
  sorry

end min_green_surface_fraction_l15_1592


namespace oil_purchase_amount_l15_1561

/-- Represents the price and quantity of oil before and after a price reduction --/
structure OilPurchase where
  original_price : ℝ
  reduced_price : ℝ
  original_quantity : ℝ
  additional_quantity : ℝ
  price_reduction_percent : ℝ

/-- Calculates the total amount spent on oil after the price reduction --/
def total_spent (purchase : OilPurchase) : ℝ :=
  purchase.reduced_price * (purchase.original_quantity + purchase.additional_quantity)

/-- Theorem stating the total amount spent on oil after the price reduction --/
theorem oil_purchase_amount (purchase : OilPurchase) 
  (h1 : purchase.price_reduction_percent = 25)
  (h2 : purchase.additional_quantity = 5)
  (h3 : purchase.reduced_price = 60)
  (h4 : purchase.reduced_price = purchase.original_price * (1 - purchase.price_reduction_percent / 100)) :
  total_spent purchase = 1200 := by
  sorry

#eval total_spent { original_price := 80, reduced_price := 60, original_quantity := 15, additional_quantity := 5, price_reduction_percent := 25 }

end oil_purchase_amount_l15_1561


namespace same_color_prob_six_green_seven_white_l15_1529

/-- The probability of drawing two balls of the same color from a bag containing 
    6 green balls and 7 white balls. -/
def same_color_probability (green : ℕ) (white : ℕ) : ℚ :=
  let total := green + white
  let p_green := (green / total) * ((green - 1) / (total - 1))
  let p_white := (white / total) * ((white - 1) / (total - 1))
  p_green + p_white

/-- Theorem stating that the probability of drawing two balls of the same color 
    from a bag with 6 green and 7 white balls is 6/13. -/
theorem same_color_prob_six_green_seven_white : 
  same_color_probability 6 7 = 6 / 13 := by
  sorry

end same_color_prob_six_green_seven_white_l15_1529


namespace parabola_translation_l15_1589

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  equation : ℝ → ℝ

/-- Represents a translation in the xy-plane -/
structure Translation where
  horizontal : ℝ
  vertical : ℝ

/-- Applies a translation to a parabola -/
def apply_translation (p : Parabola) (t : Translation) : Parabola :=
  { equation := fun x => p.equation (x - t.horizontal) + t.vertical }

theorem parabola_translation :
  let original : Parabola := { equation := fun x => x^2 }
  let translation : Translation := { horizontal := 3, vertical := -4 }
  let transformed := apply_translation original translation
  ∀ x, transformed.equation x = (x + 3)^2 - 4 := by
  sorry

end parabola_translation_l15_1589


namespace complex_number_problem_l15_1556

theorem complex_number_problem (a : ℝ) :
  let z : ℂ := a + (10 * Complex.I) / (3 - Complex.I)
  (∃ b : ℝ, z = Complex.I * b) → Complex.abs (a - 2 * Complex.I) = Real.sqrt 5 := by
  sorry

end complex_number_problem_l15_1556


namespace inequality_solution_l15_1518

theorem inequality_solution (x : ℝ) : 
  (5 * x^2 + 10 * x - 34) / ((x - 2) * (3 * x + 5)) < 2 ↔ 
  x < -5/3 ∨ x > 2 := by
  sorry

end inequality_solution_l15_1518


namespace partial_fraction_decomposition_l15_1521

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (P = -4/15 ∧ Q = -11/6 ∧ R = 31/10) ∧
    ∀ (x : ℝ), x ≠ 1 → x ≠ 4 → x ≠ 6 →
      (x^2 - 5) / ((x - 1) * (x - 4) * (x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6) := by
  sorry

end partial_fraction_decomposition_l15_1521


namespace base_10_729_equals_base_7_2061_l15_1596

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Theorem: 729 in base-10 is equal to 2061 in base-7 --/
theorem base_10_729_equals_base_7_2061 :
  729 = base7ToBase10 [1, 6, 0, 2] := by
  sorry

end base_10_729_equals_base_7_2061_l15_1596


namespace truck_and_goods_problem_l15_1541

theorem truck_and_goods_problem (x : ℕ) (total_goods : ℕ) :
  (3 * x + 5 = total_goods) →  -- Condition 1
  (4 * (x - 5) = total_goods) →  -- Condition 2
  (x = 25 ∧ total_goods = 80) :=  -- Conclusion
by sorry

end truck_and_goods_problem_l15_1541


namespace problem_solution_l15_1566

-- Define the function f
def f (a x : ℝ) : ℝ := |a - x|

-- Define the set A
def A : Set ℝ := {x | f (3/2) (2*x - 3/2) > 2 * f (3/2) (x + 2) + 2}

theorem problem_solution :
  (A = Set.Iio 0) ∧
  (∀ x₀ ∈ A, ∀ x : ℝ, f (3/2) (x₀ * x) ≥ x₀ * f (3/2) x + f (3/2) ((3/2) * x₀)) := by
  sorry


end problem_solution_l15_1566
