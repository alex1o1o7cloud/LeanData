import Mathlib

namespace NUMINAMATH_CALUDE_log_equation_solution_l544_54441

theorem log_equation_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l544_54441


namespace NUMINAMATH_CALUDE_principal_interest_difference_l544_54486

/-- Calculate the difference between principal and interest for a simple interest loan. -/
theorem principal_interest_difference
  (principal : ℕ)
  (rate : ℕ)
  (time : ℕ)
  (h1 : principal = 6200)
  (h2 : rate = 5)
  (h3 : time = 10) :
  principal - (principal * rate * time) / 100 = 3100 :=
by
  sorry

end NUMINAMATH_CALUDE_principal_interest_difference_l544_54486


namespace NUMINAMATH_CALUDE_equation_solution_l544_54454

theorem equation_solution :
  ∃ (t₁ t₂ : ℝ), t₁ > t₂ ∧
  (∀ t : ℝ, t ≠ 10 → (t^2 - 3*t - 70) / (t - 10) = 7 / (t + 4) ↔ t = t₁ ∨ t = t₂) ∧
  t₁ = -3 ∧ t₂ = -7 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l544_54454


namespace NUMINAMATH_CALUDE_chocolate_distribution_chocolate_problem_l544_54425

theorem chocolate_distribution (initial_bars : ℕ) (sisters : ℕ) (father_ate : ℕ) (father_left : ℕ) : ℕ :=
  let total_people := sisters + 1
  let bars_per_person := initial_bars / total_people
  let bars_given_to_father := (bars_per_person / 2) * total_people
  let bars_father_had := bars_given_to_father - father_ate
  bars_father_had - father_left

theorem chocolate_problem : 
  chocolate_distribution 20 4 2 5 = 3 := by sorry

end NUMINAMATH_CALUDE_chocolate_distribution_chocolate_problem_l544_54425


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l544_54457

open Real

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 - Real.sqrt 3 * cos x * cos (x + π / 2)

theorem f_monotone_increasing :
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 3 → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l544_54457


namespace NUMINAMATH_CALUDE_parking_lot_car_difference_l544_54460

theorem parking_lot_car_difference (initial_cars : ℕ) (cars_left : ℕ) (current_cars : ℕ) : 
  initial_cars = 80 → cars_left = 13 → current_cars = 85 → 
  (current_cars - initial_cars) + cars_left = 18 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_car_difference_l544_54460


namespace NUMINAMATH_CALUDE_election_winner_percentage_l544_54401

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 868 → 
  margin = 336 → 
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 62 / 100 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l544_54401


namespace NUMINAMATH_CALUDE_book_distribution_l544_54448

theorem book_distribution (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 6 →
  k = 3 →
  m = 2 →
  (Nat.choose n m) * (Nat.choose (n - m) m) * (Nat.choose (n - 2*m) m) = 90 :=
by sorry

end NUMINAMATH_CALUDE_book_distribution_l544_54448


namespace NUMINAMATH_CALUDE_cookie_cost_claire_cookie_cost_l544_54407

/-- The cost of a cookie given Claire's spending habits and gift card balance --/
theorem cookie_cost (gift_card : ℝ) (latte_cost : ℝ) (croissant_cost : ℝ) 
  (days : ℕ) (num_cookies : ℕ) (remaining_balance : ℝ) : ℝ :=
  let daily_treat_cost := latte_cost + croissant_cost
  let weekly_treat_cost := daily_treat_cost * days
  let total_spent := gift_card - remaining_balance
  let cookie_total_cost := total_spent - weekly_treat_cost
  cookie_total_cost / num_cookies

/-- Proof that each cookie costs $1.25 given Claire's spending habits --/
theorem claire_cookie_cost : 
  cookie_cost 100 3.75 3.50 7 5 43 = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cost_claire_cookie_cost_l544_54407


namespace NUMINAMATH_CALUDE_replaced_person_weight_l544_54474

def group_size : ℕ := 8
def average_weight_increase : ℝ := 2.5
def new_person_weight : ℝ := 90

theorem replaced_person_weight :
  let total_weight_increase : ℝ := group_size * average_weight_increase
  let replaced_weight : ℝ := new_person_weight - total_weight_increase
  replaced_weight = 70 := by sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l544_54474


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l544_54468

theorem triangle_area_in_circle (r : ℝ) : 
  r > 0 → 
  let a := 5 * (10 / 13)
  let b := 12 * (10 / 13)
  let c := 13 * (10 / 13)
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  c = 2 * r → -- diameter of the circle
  (1/2) * a * b = 3000/169 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_in_circle_l544_54468


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l544_54438

/-- The compound interest formula: A = P * (1 + r)^n -/
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem interest_rate_calculation
  (P : ℝ)  -- Principal amount
  (r : ℝ)  -- Rate of interest (as a decimal)
  (h1 : compound_interest P r 3 = 800)
  (h2 : compound_interest P r 4 = 820) :
  r = 0.025 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l544_54438


namespace NUMINAMATH_CALUDE_total_savings_three_months_l544_54476

def savings (n : ℕ) : ℕ := 10 + 30 * n

theorem total_savings_three_months :
  savings 0 + savings 1 + savings 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_three_months_l544_54476


namespace NUMINAMATH_CALUDE_problems_solved_l544_54434

theorem problems_solved (first last : ℕ) (h : first = 55) (h' : last = 150) :
  (Finset.range (last - first + 1)).card = 96 := by
  sorry

end NUMINAMATH_CALUDE_problems_solved_l544_54434


namespace NUMINAMATH_CALUDE_video_release_week_l544_54461

/-- Proves that the number of days in a week is 7, given John's video release schedule --/
theorem video_release_week (short_video_length : ℕ) (long_video_multiplier : ℕ) 
  (videos_per_day : ℕ) (short_videos_per_day : ℕ) (total_weekly_minutes : ℕ) :
  short_video_length = 2 →
  long_video_multiplier = 6 →
  videos_per_day = 3 →
  short_videos_per_day = 2 →
  total_weekly_minutes = 112 →
  (total_weekly_minutes / (short_videos_per_day * short_video_length + 
    (videos_per_day - short_videos_per_day) * (long_video_multiplier * short_video_length))) = 7 := by
  sorry

#check video_release_week

end NUMINAMATH_CALUDE_video_release_week_l544_54461


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l544_54499

theorem arithmetic_calculation : 
  (1 + 0.23 + 0.34) * (0.23 + 0.34 + 0.45) - (1 + 0.23 + 0.34 + 0.45) * (0.23 + 0.34) = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l544_54499


namespace NUMINAMATH_CALUDE_house_price_calculation_house_price_proof_l544_54444

theorem house_price_calculation (selling_price : ℝ) 
  (profit_rate : ℝ) (commission_rate : ℝ) : ℝ :=
  let original_price := selling_price / (1 + profit_rate - commission_rate)
  original_price

theorem house_price_proof :
  house_price_calculation 100000 0.2 0.05 = 100000 / 1.15 := by
  sorry

end NUMINAMATH_CALUDE_house_price_calculation_house_price_proof_l544_54444


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l544_54405

theorem solution_set_of_inequality (x : ℝ) :
  (2 / (x - 1) ≥ 1) ↔ (1 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l544_54405


namespace NUMINAMATH_CALUDE_magnitude_of_z_l544_54446

def z : ℂ := (1 + Complex.I) * (2 - Complex.I)

theorem magnitude_of_z : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l544_54446


namespace NUMINAMATH_CALUDE_martin_improvement_l544_54456

/-- Represents Martin's cycling performance --/
structure CyclingPerformance where
  laps : ℕ
  time : ℕ

/-- Calculates the time per lap given a cycling performance --/
def timePerLap (performance : CyclingPerformance) : ℚ :=
  performance.time / performance.laps

/-- Martin's initial cycling performance --/
def initialPerformance : CyclingPerformance :=
  { laps := 15, time := 45 }

/-- Martin's improved cycling performance --/
def improvedPerformance : CyclingPerformance :=
  { laps := 18, time := 42 }

/-- Theorem stating the improvement in Martin's per-lap time --/
theorem martin_improvement :
  timePerLap initialPerformance - timePerLap improvedPerformance = 2/3 := by
  sorry

#eval timePerLap initialPerformance - timePerLap improvedPerformance

end NUMINAMATH_CALUDE_martin_improvement_l544_54456


namespace NUMINAMATH_CALUDE_tiffanys_bags_collection_l544_54470

/-- Calculates the total number of bags collected over three days -/
def totalBags (initialBags dayTwoBags dayThreeBags : ℕ) : ℕ :=
  initialBags + dayTwoBags + dayThreeBags

/-- Proves that the total number of bags is correct for Tiffany's collection -/
theorem tiffanys_bags_collection :
  totalBags 10 3 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tiffanys_bags_collection_l544_54470


namespace NUMINAMATH_CALUDE_nine_pouches_sufficient_l544_54416

-- Define the number of coins and pouches
def totalCoins : ℕ := 60
def numPouches : ℕ := 9

-- Define a type for pouch distributions
def PouchDistribution := List ℕ

-- Function to check if a distribution is valid
def isValidDistribution (d : PouchDistribution) : Prop :=
  d.length = numPouches ∧ d.sum = totalCoins

-- Function to check if a distribution can be equally split among a given number of sailors
def canSplitEqually (d : PouchDistribution) (sailors : ℕ) : Prop :=
  ∃ (groups : List (List ℕ)), 
    groups.length = sailors ∧ 
    (∀ g ∈ groups, g.sum = totalCoins / sailors) ∧
    groups.join.toFinset = d.toFinset

-- The main theorem
theorem nine_pouches_sufficient :
  ∃ (d : PouchDistribution),
    isValidDistribution d ∧
    (∀ sailors ∈ [2, 3, 4, 5], canSplitEqually d sailors) :=
sorry

end NUMINAMATH_CALUDE_nine_pouches_sufficient_l544_54416


namespace NUMINAMATH_CALUDE_lunch_cost_proof_l544_54495

theorem lunch_cost_proof (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 22 → difference = 5 → 
  (∃ (your_cost : ℝ), your_cost + (your_cost + difference) = total) →
  friend_cost = 13.5 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_proof_l544_54495


namespace NUMINAMATH_CALUDE_circle_equation_l544_54422

-- Define the line L1: x + y + 2 = 0
def L1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 + 2 = 0}

-- Define the circle C1: x² + y² = 4
def C1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define the line L2: 2x - y - 3 = 0
def L2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2*p.1 - p.2 - 3 = 0}

-- Define the circle C we're looking for
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 - 6*p.2 - 16 = 0}

theorem circle_equation :
  (∀ p ∈ L1 ∩ C1, p ∈ C) ∧
  (∃ center ∈ L2, ∀ p ∈ C, (p.1 - center.1)^2 + (p.2 - center.2)^2 = (6^2 + 6^2) / 4) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l544_54422


namespace NUMINAMATH_CALUDE_add_million_minutes_to_start_date_l544_54490

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The start date and time -/
def startDateTime : DateTime :=
  { year := 2007, month := 4, day := 15, hour := 12, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 1000000

/-- The expected end date and time -/
def expectedEndDateTime : DateTime :=
  { year := 2009, month := 3, day := 10, hour := 10, minute := 40 }

theorem add_million_minutes_to_start_date :
  addMinutes startDateTime minutesToAdd = expectedEndDateTime :=
sorry

end NUMINAMATH_CALUDE_add_million_minutes_to_start_date_l544_54490


namespace NUMINAMATH_CALUDE_bridgette_dogs_l544_54464

/-- Represents the number of baths given to an animal in a year. -/
def baths_per_year (frequency : ℕ) : ℕ := 12 / frequency

/-- Represents the total number of baths given to a group of animals in a year. -/
def total_baths (num_animals : ℕ) (frequency : ℕ) : ℕ :=
  num_animals * baths_per_year frequency

theorem bridgette_dogs :
  ∃ (num_dogs : ℕ),
    total_baths num_dogs 2 + -- Dogs bathed twice a month
    total_baths 3 1 + -- 3 cats bathed once a month
    total_baths 4 4 = 96 ∧ -- 4 birds bathed once every 4 months
    num_dogs = 2 := by
  sorry

end NUMINAMATH_CALUDE_bridgette_dogs_l544_54464


namespace NUMINAMATH_CALUDE_fifth_power_minus_fifth_power_equals_sixteen_product_l544_54403

theorem fifth_power_minus_fifth_power_equals_sixteen_product (m n : ℤ) :
  m^5 - n^5 = 16*m*n ↔ (m = 0 ∧ n = 0) ∨ (m = -2 ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_fifth_power_minus_fifth_power_equals_sixteen_product_l544_54403


namespace NUMINAMATH_CALUDE_committee_selection_l544_54411

theorem committee_selection (n : ℕ) : 
  (n.choose 3 = 20) → (n.choose 4 = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l544_54411


namespace NUMINAMATH_CALUDE_angle_Y_is_50_l544_54492

-- Define the angles in the geometric figure
def angle_X : ℝ := 120
def angle_Y : ℝ := 50
def angle_Z : ℝ := 180 - angle_X

-- Theorem statement
theorem angle_Y_is_50 : 
  angle_X = 120 →
  angle_Y = 50 →
  angle_Z = 180 - angle_X →
  angle_Y = 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_Y_is_50_l544_54492


namespace NUMINAMATH_CALUDE_james_annual_training_hours_l544_54496

/-- Calculates the total training hours per year for an athlete with a specific schedule. -/
def training_hours_per_year (sessions_per_day : ℕ) (hours_per_session : ℕ) (training_days_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  sessions_per_day * hours_per_session * training_days_per_week * weeks_per_year

/-- Proves that James' training schedule results in 2080 hours per year. -/
theorem james_annual_training_hours :
  training_hours_per_year 2 4 5 52 = 2080 := by
  sorry

#eval training_hours_per_year 2 4 5 52

end NUMINAMATH_CALUDE_james_annual_training_hours_l544_54496


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l544_54478

theorem isosceles_right_triangle_hypotenuse (a c : ℝ) :
  a > 0 →  -- Ensure positive side length
  c > 0 →  -- Ensure positive hypotenuse length
  c^2 = 2 * a^2 →  -- Pythagorean theorem for isosceles right triangle
  2 * a + c = 4 + 4 * Real.sqrt 2 →  -- Perimeter condition
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l544_54478


namespace NUMINAMATH_CALUDE_evaluate_dagger_l544_54430

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (q / n)

-- Theorem statement
theorem evaluate_dagger : dagger (5/16) (12/5) = 75/4 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_dagger_l544_54430


namespace NUMINAMATH_CALUDE_date_statistics_order_l544_54421

def date_counts : List (Nat × Nat) := 
  (List.range 30).map (fun n => (n + 1, 12)) ++ [(31, 7)]

def total_count : Nat := date_counts.foldl (fun acc (_, count) => acc + count) 0

def sum_of_values : Nat := date_counts.foldl (fun acc (date, count) => acc + date * count) 0

def mean : ℚ := sum_of_values / total_count

def median : Nat := 16

def median_of_modes : ℚ := 15.5

theorem date_statistics_order : median_of_modes < mean ∧ mean < median := by sorry

end NUMINAMATH_CALUDE_date_statistics_order_l544_54421


namespace NUMINAMATH_CALUDE_triangle_ratio_l544_54488

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  A = Real.pi / 3 →
  b = 1 →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_l544_54488


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l544_54482

/-- Theorem: Volume of a tetrahedron
  Given a tetrahedron with:
  - a, b: lengths of opposite edges
  - α: angle between these edges
  - c: distance between the lines containing these edges
  The volume V of the tetrahedron is given by V = (1/6) * a * b * c * sin(α)
-/
theorem tetrahedron_volume 
  (a b c : ℝ) 
  (α : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hα : 0 < α ∧ α < π) :
  ∃ V : ℝ, V = (1/6) * a * b * c * Real.sin α := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_volume_l544_54482


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l544_54455

theorem sqrt_product_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (8 * x) = 30 * x * Real.sqrt (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l544_54455


namespace NUMINAMATH_CALUDE_seven_power_plus_one_prime_divisors_l544_54432

theorem seven_power_plus_one_prime_divisors (n : ℕ) :
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ 
    (∀ p ∈ S, p ∣ (7^(7^n) + 1)) ∧ 
    (Finset.card S ≥ 2*n + 3) :=
by sorry

end NUMINAMATH_CALUDE_seven_power_plus_one_prime_divisors_l544_54432


namespace NUMINAMATH_CALUDE_total_handshakes_l544_54406

def number_of_couples : ℕ := 15

-- Define the number of handshakes between men
def handshakes_between_men (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define the number of handshakes between women
def handshakes_between_women (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define the number of handshakes between men and women (excluding spouses)
def handshakes_men_women (n : ℕ) : ℕ := n * (n - 1)

theorem total_handshakes :
  handshakes_between_men number_of_couples +
  handshakes_between_women number_of_couples +
  handshakes_men_women number_of_couples = 420 := by
  sorry

end NUMINAMATH_CALUDE_total_handshakes_l544_54406


namespace NUMINAMATH_CALUDE_simplify_expression_l544_54472

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l544_54472


namespace NUMINAMATH_CALUDE_find_y_value_l544_54417

theorem find_y_value : ∃ y : ℚ, 
  (1/4 : ℚ) * ((y + 8) + (7*y + 4) + (3*y + 9) + (4*y + 5)) = 6*y - 10 → y = 22/3 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l544_54417


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_sum_l544_54447

/-- The sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a : ℕ → ℚ) : ℚ := (n : ℚ) * (a 1 + a n) / 2

/-- An arithmetic sequence with common difference -1 -/
def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = a n - 1

/-- S_1, S_2, S_4 form a geometric sequence -/
def geometricSequence (a : ℕ → ℚ) : Prop :=
  (S 2 a) ^ 2 = (S 1 a) * (S 4 a)

theorem arithmetic_sequence_with_geometric_sum 
  (a : ℕ → ℚ) 
  (h1 : arithmeticSequence a) 
  (h2 : geometricSequence a) : 
  ∀ n, a n = 1/2 - n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_sum_l544_54447


namespace NUMINAMATH_CALUDE_unique_solution_tan_equation_l544_54420

theorem unique_solution_tan_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan (150 * π / 180 - x * π / 180) =
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 110 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_tan_equation_l544_54420


namespace NUMINAMATH_CALUDE_more_polygons_with_specific_point_l544_54494

theorem more_polygons_with_specific_point (n : ℕ) (h : n = 16) :
  let total_polygons := 2^n - (Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2)
  let polygons_with_point := 2^(n-1) - (Nat.choose (n-1) 0 + Nat.choose (n-1) 1)
  let polygons_without_point := total_polygons - polygons_with_point
  polygons_with_point > polygons_without_point := by
sorry

end NUMINAMATH_CALUDE_more_polygons_with_specific_point_l544_54494


namespace NUMINAMATH_CALUDE_limes_remaining_l544_54433

/-- The number of limes Mike picked -/
def mike_limes : ℝ := 32.0

/-- The number of limes Alyssa ate -/
def alyssa_limes : ℝ := 25.0

/-- The number of limes left -/
def limes_left : ℝ := mike_limes - alyssa_limes

theorem limes_remaining : limes_left = 7.0 := by sorry

end NUMINAMATH_CALUDE_limes_remaining_l544_54433


namespace NUMINAMATH_CALUDE_five_by_seven_domino_five_by_seven_minus_corner_domino_five_by_seven_minus_second_row_domino_six_by_six_tetromino_l544_54426

-- Define the types of tiles
inductive Tile
| Domino    -- 2x1 tile
| Tetromino -- 4x1 tile

-- Define a board
structure Board :=
(rows : ℕ)
(cols : ℕ)
(removed_cells : List (ℕ × ℕ)) -- List of removed cells' coordinates

-- Define a function to check if a board can be tiled
def can_be_tiled (b : Board) (t : Tile) : Prop :=
  match t with
  | Tile.Domino    => sorry
  | Tile.Tetromino => sorry

-- Theorem 1: A 5x7 board cannot be tiled with dominoes
theorem five_by_seven_domino :
  ¬ can_be_tiled { rows := 5, cols := 7, removed_cells := [] } Tile.Domino :=
sorry

-- Theorem 2: A 5x7 board with bottom left corner removed can be tiled with dominoes
theorem five_by_seven_minus_corner_domino :
  can_be_tiled { rows := 5, cols := 7, removed_cells := [(1, 1)] } Tile.Domino :=
sorry

-- Theorem 3: A 5x7 board with leftmost cell on second row removed cannot be tiled with dominoes
theorem five_by_seven_minus_second_row_domino :
  ¬ can_be_tiled { rows := 5, cols := 7, removed_cells := [(2, 1)] } Tile.Domino :=
sorry

-- Theorem 4: A 6x6 board can be tiled with tetrominoes
theorem six_by_six_tetromino :
  can_be_tiled { rows := 6, cols := 6, removed_cells := [] } Tile.Tetromino :=
sorry

end NUMINAMATH_CALUDE_five_by_seven_domino_five_by_seven_minus_corner_domino_five_by_seven_minus_second_row_domino_six_by_six_tetromino_l544_54426


namespace NUMINAMATH_CALUDE_max_area_triangle_OAB_l544_54410

/-- The maximum area of triangle OAB in the complex plane -/
theorem max_area_triangle_OAB :
  ∀ (α β : ℂ),
  β = (1 + Complex.I) * α →
  Complex.abs (α - 2) = 1 →
  (∀ (S : ℝ),
    S = (Complex.abs α * Complex.abs β * Real.sin (Real.pi / 4)) / 2 →
    S ≤ 9 / 2) ∧
  ∃ (α₀ β₀ : ℂ),
    β₀ = (1 + Complex.I) * α₀ ∧
    Complex.abs (α₀ - 2) = 1 ∧
    (Complex.abs α₀ * Complex.abs β₀ * Real.sin (Real.pi / 4)) / 2 = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_OAB_l544_54410


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l544_54453

/-- Given a line L1 with equation 2x - y - 1 = 0, prove that the line L2 passing through
    the point (1, 2) and parallel to L1 has the equation 2x - y = 0. -/
theorem parallel_line_through_point (x y : ℝ) :
  (∃ (m b : ℝ), 2 * x - y - 1 = m * x + b) →  -- L1 exists
  (∃ (k : ℝ), 2 * 1 - 2 = k) →                -- L2 passes through (1, 2)
  (∃ (c : ℝ), 2 * x - y + c = 0) →            -- L2 is parallel to L1
  (2 * x - y = 0) :=                          -- L2 equation
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l544_54453


namespace NUMINAMATH_CALUDE_second_number_proof_l544_54466

theorem second_number_proof (x : ℕ) : x > 1428 ∧ 
  x % 129 = 13 ∧ 
  1428 % 129 = 9 ∧ 
  (∀ y, y > 1428 ∧ y % 129 = 13 → y ≥ x) ∧ 
  x = 1561 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l544_54466


namespace NUMINAMATH_CALUDE_frog_expected_returns_l544_54415

/-- Represents the probability of moving in a certain direction or getting eaten -/
def move_probability : ℚ := 1 / 3

/-- Represents the frog's position on the number line -/
def Position : Type := ℤ

/-- Calculates the probability of returning to the starting position from a given position -/
noncomputable def prob_return_to_start (pos : Position) : ℝ :=
  sorry

/-- Calculates the expected number of returns to the starting position before getting eaten -/
noncomputable def expected_returns : ℝ :=
  sorry

/-- The main theorem stating the expected number of returns -/
theorem frog_expected_returns :
  expected_returns = (3 * Real.sqrt 5 - 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_frog_expected_returns_l544_54415


namespace NUMINAMATH_CALUDE_largest_zero_correct_l544_54483

/-- Sequence S defined recursively -/
def S : ℕ → ℤ
  | 0 => 0
  | (n + 1) => S n + (n + 1) * (if S n < n + 1 then 1 else -1)

/-- Predicate for S[k] = 0 -/
def is_zero (k : ℕ) : Prop := S k = 0

/-- The largest k ≤ 2010 such that S[k] = 0 -/
def largest_zero : ℕ := 1092

theorem largest_zero_correct :
  is_zero largest_zero ∧
  ∀ k, k ≤ 2010 → is_zero k → k ≤ largest_zero :=
by sorry

end NUMINAMATH_CALUDE_largest_zero_correct_l544_54483


namespace NUMINAMATH_CALUDE_range_of_a_range_of_t_l544_54418

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Statement for the range of a
theorem range_of_a : 
  {a : ℝ | ∃ x, a ≥ f x} = {a : ℝ | a ≥ -5/2} := by sorry

-- Statement for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ -t^2 - 5/2*t - 1} = 
  {t : ℝ | t ≥ 1/2 ∨ t ≤ -3} := by sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_t_l544_54418


namespace NUMINAMATH_CALUDE_water_park_admission_charge_l544_54473

/-- Calculates the total admission charge for an adult and accompanying children in a water park. -/
def totalAdmissionCharge (adultCharge childCharge : ℚ) (numChildren : ℕ) : ℚ :=
  adultCharge + childCharge * numChildren

/-- Proves that the total admission charge for an adult and 3 children is $3.25 -/
theorem water_park_admission_charge :
  let adultCharge : ℚ := 1
  let childCharge : ℚ := 3/4
  let numChildren : ℕ := 3
  totalAdmissionCharge adultCharge childCharge numChildren = 13/4 := by
sorry

#eval totalAdmissionCharge 1 (3/4) 3

end NUMINAMATH_CALUDE_water_park_admission_charge_l544_54473


namespace NUMINAMATH_CALUDE_picture_area_l544_54402

theorem picture_area (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (h_frame_area : (2 * x + 3) * (y + 2) - x * y = 34) : x * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l544_54402


namespace NUMINAMATH_CALUDE_calculate_product_l544_54452

theorem calculate_product : 200 * 375 * 0.0375 * 5 = 14062.5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_product_l544_54452


namespace NUMINAMATH_CALUDE_triangle_area_scaling_l544_54442

theorem triangle_area_scaling (original_area new_area : ℝ) : 
  new_area = 54 → 
  new_area = 9 * original_area → 
  original_area = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_scaling_l544_54442


namespace NUMINAMATH_CALUDE_line_tangent_to_curve_l544_54450

/-- The line equation: kx - y + 1 = 0 -/
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y + 1 = 0

/-- The curve equation: y² = 4x -/
def curve_equation (x y : ℝ) : Prop :=
  y^2 = 4 * x

/-- The tangency condition: the discriminant of the resulting quadratic equation is zero -/
def tangency_condition (k : ℝ) : Prop :=
  (4 * k - 8)^2 - 16 * k^2 = 0

theorem line_tangent_to_curve (k : ℝ) :
  (∀ x y : ℝ, line_equation k x y ∧ curve_equation x y → tangency_condition k) →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_curve_l544_54450


namespace NUMINAMATH_CALUDE_solution_difference_l544_54414

theorem solution_difference (r s : ℝ) : 
  ((r - 4) * (r + 4) = 24 * r - 96) →
  ((s - 4) * (s + 4) = 24 * s - 96) →
  r ≠ s →
  r > s →
  r - s = 16 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l544_54414


namespace NUMINAMATH_CALUDE_next_perfect_square_sum_l544_54439

def children_ages : List ℕ := [1, 3, 5, 7, 9, 11, 13]

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def sum_ages (years_later : ℕ) : ℕ :=
  List.sum (List.map (· + years_later) children_ages)

theorem next_perfect_square_sum :
  (∃ (x : ℕ), x > 0 ∧ 
    is_perfect_square (sum_ages x) ∧
    (∀ y : ℕ, 0 < y ∧ y < x → ¬is_perfect_square (sum_ages y))) →
  (∃ (x : ℕ), x = 21 ∧ 
    is_perfect_square (sum_ages x) ∧
    (List.head! children_ages + x) + sum_ages x = 218) :=
by sorry

end NUMINAMATH_CALUDE_next_perfect_square_sum_l544_54439


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l544_54408

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 33 = 14 % 33 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 33 = 14 % 33 → x ≤ y :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l544_54408


namespace NUMINAMATH_CALUDE_max_prime_difference_l544_54429

theorem max_prime_difference (a b c d : ℕ) : 
  a.Prime ∧ b.Prime ∧ c.Prime ∧ d.Prime ∧
  (a + b + c + 18 + d).Prime ∧ (a + b + c + 18 - d).Prime ∧
  (b + c).Prime ∧ (c + d).Prime ∧
  (a + b + c = 2010) ∧
  (a ≠ 3 ∧ b ≠ 3 ∧ c ≠ 3 ∧ d ≠ 3) ∧
  (d ≤ 50) →
  (∃ (p q : ℕ), (p.Prime ∧ q.Prime ∧ 
    (p = a ∨ p = b ∨ p = c ∨ p = d ∨ 
     p = a + b + c + 18 + d ∨ p = a + b + c + 18 - d ∨
     p = b + c ∨ p = c + d) ∧
    (q = a ∨ q = b ∨ q = c ∨ q = d ∨ 
     q = a + b + c + 18 + d ∨ q = a + b + c + 18 - d ∨
     q = b + c ∨ q = c + d) ∧
    p - q ≤ 2067) ∧
   ∀ (r s : ℕ), (r.Prime ∧ s.Prime ∧ 
    (r = a ∨ r = b ∨ r = c ∨ r = d ∨ 
     r = a + b + c + 18 + d ∨ r = a + b + c + 18 - d ∨
     r = b + c ∨ r = c + d) ∧
    (s = a ∨ s = b ∨ s = c ∨ s = d ∨ 
     s = a + b + c + 18 + d ∨ s = a + b + c + 18 - d ∨
     s = b + c ∨ s = c + d) →
    r - s ≤ 2067)) :=
by sorry

end NUMINAMATH_CALUDE_max_prime_difference_l544_54429


namespace NUMINAMATH_CALUDE_sisters_birth_year_l544_54435

/-- Represents the birth years of family members --/
structure FamilyBirthYears where
  brother : Nat
  sister : Nat
  grandmother : Nat

/-- Checks if the birth years satisfy the given conditions --/
def validBirthYears (years : FamilyBirthYears) : Prop :=
  years.brother = 1932 ∧
  years.grandmother = 1944 ∧
  (years.grandmother - years.sister) = 2 * (years.sister - years.brother)

/-- Theorem stating that the grandmother's older sister was born in 1936 --/
theorem sisters_birth_year (years : FamilyBirthYears) 
  (h : validBirthYears years) : years.sister = 1936 := by
  sorry

#check sisters_birth_year

end NUMINAMATH_CALUDE_sisters_birth_year_l544_54435


namespace NUMINAMATH_CALUDE_triangle_ratio_l544_54469

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin A - b * Real.sin B = 4 * c * Real.sin C →
  Real.cos A = -1/4 →
  b / c = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l544_54469


namespace NUMINAMATH_CALUDE_expression_evaluation_l544_54467

theorem expression_evaluation : 2^3 + 15 * 2 - 4 + 10 * 5 / 2 = 59 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l544_54467


namespace NUMINAMATH_CALUDE_quadratic_point_relation_l544_54487

/-- The quadratic function f(x) = (x - 1)^2 -/
def f (x : ℝ) : ℝ := (x - 1)^2

theorem quadratic_point_relation (m : ℝ) :
  f m < f (m + 1) → m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_relation_l544_54487


namespace NUMINAMATH_CALUDE_leopards_score_l544_54423

theorem leopards_score (total_points margin : ℕ) 
  (h1 : total_points = 50)
  (h2 : margin = 28) : 
  ∃ (jaguars_score leopards_score : ℕ),
    jaguars_score + leopards_score = total_points ∧
    jaguars_score - leopards_score = margin ∧
    leopards_score = 11 := by
  sorry

end NUMINAMATH_CALUDE_leopards_score_l544_54423


namespace NUMINAMATH_CALUDE_complex_number_properties_l544_54431

theorem complex_number_properties (z : ℂ) (h : Complex.I * (z + 1) = -2 + 2 * Complex.I) :
  (Complex.im z = 2) ∧ (let ω := z / (1 - 2 * Complex.I); Complex.abs ω ^ 2015 = 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l544_54431


namespace NUMINAMATH_CALUDE_factorization_equality_l544_54462

theorem factorization_equality (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) = (3 * y + 5) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l544_54462


namespace NUMINAMATH_CALUDE_ramu_car_profit_percent_l544_54475

/-- Calculates the profit percent given the purchase price, repair cost, and selling price of a car. -/
def profit_percent (purchase_price repair_cost selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem stating that the profit percent for Ramu's car transaction is 29.8% -/
theorem ramu_car_profit_percent :
  profit_percent 42000 8000 64900 = 29.8 :=
by sorry

end NUMINAMATH_CALUDE_ramu_car_profit_percent_l544_54475


namespace NUMINAMATH_CALUDE_quadratic_sum_l544_54458

theorem quadratic_sum (x : ℝ) : 
  let f : ℝ → ℝ := λ x => -3 * x^2 + 27 * x - 153
  ∃ (a b c : ℝ), (∀ x, f x = a * (x + b)^2 + c) ∧ (a + b + c = -99.75) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l544_54458


namespace NUMINAMATH_CALUDE_park_legs_count_l544_54445

/-- Calculate the total number of legs for given numbers of dogs, cats, birds, and spiders -/
def totalLegs (dogs cats birds spiders : ℕ) : ℕ :=
  4 * dogs + 4 * cats + 2 * birds + 8 * spiders

/-- Theorem stating that the total number of legs for 109 dogs, 37 cats, 52 birds, and 19 spiders is 840 -/
theorem park_legs_count : totalLegs 109 37 52 19 = 840 := by
  sorry

end NUMINAMATH_CALUDE_park_legs_count_l544_54445


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l544_54465

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  2 * (2*b - 3*a) + 3 * (2*a - 3*b) = -5*b := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  4*a^2 + 2*(3*a*b - 2*a^2) - (7*a*b - 1) = -a*b + 1 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l544_54465


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l544_54491

/-- Represents the systematic sampling problem --/
theorem systematic_sampling_problem 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (num_groups : ℕ) 
  (group_size : ℕ) 
  (sixteenth_group_num : ℕ) :
  total_students = 160 →
  sample_size = 20 →
  num_groups = 20 →
  group_size = total_students / num_groups →
  sixteenth_group_num = 126 →
  ∃ (first_group_num : ℕ), first_group_num = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_problem_l544_54491


namespace NUMINAMATH_CALUDE_m_range_proof_l544_54437

theorem m_range_proof (m : ℝ) (h_pos : m > 0) : 
  (∀ x : ℝ, x ≤ -1 → (3 * m - 1) * 2^x < 1) → 
  m < 1 :=
by sorry

end NUMINAMATH_CALUDE_m_range_proof_l544_54437


namespace NUMINAMATH_CALUDE_fraction_problem_l544_54497

theorem fraction_problem (F : ℝ) :
  (0.4 * F * 150 = 36) → F = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l544_54497


namespace NUMINAMATH_CALUDE_arithmetic_progression_theorem_l544_54436

/-- Represents an arithmetic progression of five terms. -/
structure ArithmeticProgression :=
  (a : ℝ)  -- First term
  (d : ℝ)  -- Common difference

/-- Checks if the arithmetic progression is decreasing. -/
def ArithmeticProgression.isDecreasing (ap : ArithmeticProgression) : Prop :=
  ap.d > 0

/-- Calculates the sum of cubes of the terms in the arithmetic progression. -/
def ArithmeticProgression.sumOfCubes (ap : ArithmeticProgression) : ℝ :=
  ap.a^3 + (ap.a - ap.d)^3 + (ap.a - 2*ap.d)^3 + (ap.a - 3*ap.d)^3 + (ap.a - 4*ap.d)^3

/-- Calculates the sum of fourth powers of the terms in the arithmetic progression. -/
def ArithmeticProgression.sumOfFourthPowers (ap : ArithmeticProgression) : ℝ :=
  ap.a^4 + (ap.a - ap.d)^4 + (ap.a - 2*ap.d)^4 + (ap.a - 3*ap.d)^4 + (ap.a - 4*ap.d)^4

/-- The main theorem stating the properties of the required arithmetic progression. -/
theorem arithmetic_progression_theorem (ap : ArithmeticProgression) :
  ap.isDecreasing ∧
  ap.sumOfCubes = 0 ∧
  ap.sumOfFourthPowers = 306 →
  ap.a - 4*ap.d = -2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_theorem_l544_54436


namespace NUMINAMATH_CALUDE_student_A_pass_probability_l544_54409

/-- Probability that student A passes the exam --/
def prob_pass (pA pB pDAC pDB : ℝ) : ℝ :=
  pA * pDAC + pB * pDB + (1 - pA - pB) * pDAC

theorem student_A_pass_probability :
  let pA := 0.3
  let pB := 0.3
  let pDAC := 0.8
  let pDB := 0.6
  prob_pass pA pB pDAC pDB = 0.74 := by
  sorry

#eval prob_pass 0.3 0.3 0.8 0.6

end NUMINAMATH_CALUDE_student_A_pass_probability_l544_54409


namespace NUMINAMATH_CALUDE_horse_race_equation_l544_54484

/-- Represents the scenario of two horses racing --/
structure HorseRace where
  fast_speed : ℕ  -- Speed of the fast horse in miles per day
  slow_speed : ℕ  -- Speed of the slow horse in miles per day
  head_start : ℕ  -- Number of days the slow horse starts earlier

/-- The equation for when the fast horse catches up to the slow horse --/
def catch_up_equation (race : HorseRace) (x : ℕ) : Prop :=
  race.slow_speed * (x + race.head_start) = race.fast_speed * x

/-- The theorem stating the correct equation for the given scenario --/
theorem horse_race_equation :
  let race := HorseRace.mk 240 150 12
  ∀ x, catch_up_equation race x ↔ 150 * (x + 12) = 240 * x :=
by sorry

end NUMINAMATH_CALUDE_horse_race_equation_l544_54484


namespace NUMINAMATH_CALUDE_total_balls_in_box_l544_54489

theorem total_balls_in_box (white_balls black_balls : ℕ) : 
  white_balls = 6 * black_balls →
  black_balls = 8 →
  white_balls + black_balls = 56 := by
sorry

end NUMINAMATH_CALUDE_total_balls_in_box_l544_54489


namespace NUMINAMATH_CALUDE_coin_representation_l544_54400

def is_representable (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 3 * a + 5 * b

theorem coin_representation :
  ∀ n : ℕ, n > 0 → (is_representable n ↔ n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 4 ∧ n ≠ 7) :=
by sorry

end NUMINAMATH_CALUDE_coin_representation_l544_54400


namespace NUMINAMATH_CALUDE_max_a_for_monotonic_f_l544_54459

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem max_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x ≤ y → f a x ≤ f a y) →
  a ≤ 0 ∧ ∀ b : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x ≤ y → f b x ≤ f b y) → b ≤ a :=
sorry

end NUMINAMATH_CALUDE_max_a_for_monotonic_f_l544_54459


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l544_54498

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h
  , c := p.a * h^2 + k }

theorem parabola_shift_theorem (original : Parabola) (h k : ℝ) :
  original.a = 3 ∧ original.b = 0 ∧ original.c = 0 ∧ h = 1 ∧ k = 2 →
  let shifted := shift_parabola original h k
  shifted.a = 3 ∧ shifted.b = -6 ∧ shifted.c = 5 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l544_54498


namespace NUMINAMATH_CALUDE_amount_saved_christine_savings_l544_54477

/-- Calculates the amount saved by a salesperson given their commission rate, total sales, and allocation for personal needs. -/
theorem amount_saved 
  (commission_rate : ℚ) 
  (total_sales : ℚ) 
  (personal_needs_allocation : ℚ) : ℚ :=
  let commission_earned := commission_rate * total_sales
  let amount_for_personal_needs := personal_needs_allocation * commission_earned
  commission_earned - amount_for_personal_needs

/-- Proves that given the specific conditions, Christine saved $1152. -/
theorem christine_savings : 
  amount_saved (12/100) 24000 (60/100) = 1152 := by
  sorry

end NUMINAMATH_CALUDE_amount_saved_christine_savings_l544_54477


namespace NUMINAMATH_CALUDE_afternoon_fish_count_l544_54471

/-- Proves that the number of fish caught in the afternoon is 3 --/
theorem afternoon_fish_count (morning_a : ℕ) (morning_b : ℕ) (total : ℕ)
  (h1 : morning_a = 4)
  (h2 : morning_b = 3)
  (h3 : total = 10) :
  total - (morning_a + morning_b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_fish_count_l544_54471


namespace NUMINAMATH_CALUDE_corner_cut_pentagon_area_corner_cut_pentagon_area_is_correct_l544_54424

/-- A pentagon formed by cutting a triangular corner from a rectangle -/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {14, 21, 22, 28, 37}

/-- The area of a CornerCutPentagon is 826 -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : ℕ :=
  826

/-- The area of a CornerCutPentagon is correct -/
theorem corner_cut_pentagon_area_is_correct (p : CornerCutPentagon) :
  corner_cut_pentagon_area p = 826 := by sorry

end NUMINAMATH_CALUDE_corner_cut_pentagon_area_corner_cut_pentagon_area_is_correct_l544_54424


namespace NUMINAMATH_CALUDE_team_a_win_probability_l544_54480

def number_of_matches : ℕ := 7
def wins_required : ℕ := 4
def win_probability : ℚ := 1/2

theorem team_a_win_probability :
  (number_of_matches.choose wins_required) * win_probability ^ wins_required * (1 - win_probability) ^ (number_of_matches - wins_required) = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_team_a_win_probability_l544_54480


namespace NUMINAMATH_CALUDE_square_root_sum_equals_four_root_six_l544_54481

theorem square_root_sum_equals_four_root_six :
  Real.sqrt (16 - 8 * Real.sqrt 6) + Real.sqrt (16 + 8 * Real.sqrt 6) = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_four_root_six_l544_54481


namespace NUMINAMATH_CALUDE_point_not_in_A_when_a_negative_l544_54451

-- Define the set A
def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 ≥ 0 ∧ a * p.1 + p.2 ≥ 2 ∧ p.1 - a * p.2 ≤ 2}

-- Theorem statement
theorem point_not_in_A_when_a_negative :
  ∀ a : ℝ, a < 0 → (1, 1) ∉ A a :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_A_when_a_negative_l544_54451


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l544_54479

/-- Given 60 feet of fencing for a rectangular pen where the length is exactly twice the width,
    the maximum possible area is 200 square feet. -/
theorem max_area_rectangular_pen (perimeter : ℝ) (width : ℝ) (length : ℝ) (area : ℝ) :
  perimeter = 60 →
  length = 2 * width →
  perimeter = 2 * length + 2 * width →
  area = length * width →
  area ≤ 200 ∧ ∃ w l, width = w ∧ length = l ∧ area = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l544_54479


namespace NUMINAMATH_CALUDE_vertical_shift_graph_l544_54463

-- Define a type for functions from real numbers to real numbers
def RealFunction := ℝ → ℝ

-- Define a vertical shift operation on functions
def verticalShift (f : RealFunction) (k : ℝ) : RealFunction :=
  λ x => f x + k

-- State the theorem
theorem vertical_shift_graph (f : RealFunction) (k : ℝ) :
  ∀ x y, y = f x + k ↔ y - k = f x :=
sorry

end NUMINAMATH_CALUDE_vertical_shift_graph_l544_54463


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l544_54427

theorem sum_of_four_numbers : 1357 + 7531 + 3175 + 5713 = 17776 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l544_54427


namespace NUMINAMATH_CALUDE_range_of_a_l544_54493

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 2| - |x - 1| ≥ a^3 - 4*a^2 - 3) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l544_54493


namespace NUMINAMATH_CALUDE_expression_equality_l544_54419

theorem expression_equality (v u w : ℝ) 
  (h1 : u = 3 * v) 
  (h2 : w = 5 * u) : 
  2 * v + u + w = 20 * v := by sorry

end NUMINAMATH_CALUDE_expression_equality_l544_54419


namespace NUMINAMATH_CALUDE_solution_property_l544_54449

theorem solution_property (m n : ℝ) (hm : m ≠ 0) 
  (h : m^2 + n*m - m = 0) : m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_property_l544_54449


namespace NUMINAMATH_CALUDE_exists_two_equal_types_l544_54404

/-- Represents the types of sweets -/
inductive SweetType
  | Blackberry
  | Coconut
  | Chocolate

/-- Represents the number of sweets for each type -/
structure Sweets where
  blackberry : Nat
  coconut : Nat
  chocolate : Nat

/-- The initial number of sweets -/
def initialSweets : Sweets :=
  { blackberry := 7, coconut := 6, chocolate := 3 }

/-- The number of sweets Sofia eats -/
def eatenSweets : Nat := 2

/-- Checks if two types of sweets have the same number -/
def hasTwoEqualTypes (s : Sweets) : Prop :=
  (s.blackberry = s.coconut) ∨ (s.blackberry = s.chocolate) ∨ (s.coconut = s.chocolate)

/-- Theorem: It's possible for grandmother to receive the same number of sweets for two varieties -/
theorem exists_two_equal_types :
  ∃ (finalSweets : Sweets),
    finalSweets.blackberry + finalSweets.coconut + finalSweets.chocolate =
      initialSweets.blackberry + initialSweets.coconut + initialSweets.chocolate - eatenSweets ∧
    finalSweets.blackberry ≤ initialSweets.blackberry ∧
    finalSweets.coconut ≤ initialSweets.coconut ∧
    finalSweets.chocolate ≤ initialSweets.chocolate ∧
    hasTwoEqualTypes finalSweets :=
  sorry

end NUMINAMATH_CALUDE_exists_two_equal_types_l544_54404


namespace NUMINAMATH_CALUDE_triangle_count_l544_54412

def stick_lengths : List ℕ := [1, 2, 3, 4, 5]

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def count_valid_triangles (lengths : List ℕ) : ℕ :=
  (lengths.toFinset.powerset.filter (fun s => s.card = 3)).card

theorem triangle_count : count_valid_triangles stick_lengths = 22 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l544_54412


namespace NUMINAMATH_CALUDE_least_possible_average_speed_l544_54443

/-- Represents a palindromic number -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The drive duration in hours -/
def driveDuration : ℕ := 5

/-- The speed limit in miles per hour -/
def speedLimit : ℕ := 65

/-- The initial odometer reading -/
def initialReading : ℕ := 123321

/-- Theorem: The least possible average speed is 20 miles per hour -/
theorem least_possible_average_speed :
  ∃ (finalReading : ℕ),
    isPalindrome initialReading ∧
    isPalindrome finalReading ∧
    finalReading > initialReading ∧
    finalReading - initialReading ≤ driveDuration * speedLimit ∧
    (finalReading - initialReading) / driveDuration = 20 ∧
    ∀ (otherFinalReading : ℕ),
      isPalindrome otherFinalReading →
      otherFinalReading > initialReading →
      otherFinalReading - initialReading ≤ driveDuration * speedLimit →
      (otherFinalReading - initialReading) / driveDuration ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_least_possible_average_speed_l544_54443


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_count_l544_54413

/-- Represents a club with members and their characteristics -/
structure Club where
  total_members : ℕ
  left_handed : ℕ
  jazz_lovers : ℕ
  right_handed_non_jazz : ℕ

/-- The number of left-handed jazz lovers in the club -/
def left_handed_jazz_lovers (c : Club) : ℕ :=
  c.total_members + c.right_handed_non_jazz - c.left_handed - c.jazz_lovers

/-- Theorem stating the number of left-handed jazz lovers in the given club -/
theorem left_handed_jazz_lovers_count (c : Club) 
  (h1 : c.total_members = 30)
  (h2 : c.left_handed = 12)
  (h3 : c.jazz_lovers = 22)
  (h4 : c.right_handed_non_jazz = 4)
  (h5 : c.left_handed + (c.total_members - c.left_handed) = c.total_members) :
  left_handed_jazz_lovers c = 8 := by
  sorry

#eval left_handed_jazz_lovers ⟨30, 12, 22, 4⟩

end NUMINAMATH_CALUDE_left_handed_jazz_lovers_count_l544_54413


namespace NUMINAMATH_CALUDE_sqrt_equation_sum_l544_54440

theorem sqrt_equation_sum (n a t : ℝ) (hn : n ≥ 2) (ha : a > 0) (ht : t > 0) :
  Real.sqrt (n + a / t) = n * Real.sqrt (a / t) → a + t = n^2 + n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_sum_l544_54440


namespace NUMINAMATH_CALUDE_F_properties_l544_54485

-- Define the function f
def f (a b : ℝ) (h1 : 0 < b) (h2 : b < -a) : ℝ → ℝ := sorry

-- Define the function F
def F (a b : ℝ) (h1 : 0 < b) (h2 : b < -a) (x : ℝ) : ℝ :=
  (f a b h1 h2 x)^2 - (f a b h1 h2 (-x))^2

-- State the theorem
theorem F_properties (a b : ℝ) (h1 : 0 < b) (h2 : b < -a) :
  (∀ x, F a b h1 h2 x ≠ 0) →
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f a b h1 h2 x < f a b h1 h2 y) →
  (∀ x, F a b h1 h2 x = 0 ∨ -b ≤ x ∧ x ≤ b) ∧
  (∀ x, F a b h1 h2 (-x) = -(F a b h1 h2 x)) :=
by sorry

end NUMINAMATH_CALUDE_F_properties_l544_54485


namespace NUMINAMATH_CALUDE_cube_sum_divisible_by_nine_l544_54428

theorem cube_sum_divisible_by_nine (n : ℕ+) : 
  9 ∣ (n.val^3 + (n.val + 1)^3 + (n.val + 2)^3) := by
sorry

end NUMINAMATH_CALUDE_cube_sum_divisible_by_nine_l544_54428
