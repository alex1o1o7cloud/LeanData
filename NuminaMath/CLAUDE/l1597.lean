import Mathlib

namespace first_discount_is_twenty_percent_l1597_159723

def initial_price : ℝ := 12000
def final_price : ℝ := 7752
def second_discount : ℝ := 0.15
def third_discount : ℝ := 0.05

def first_discount_percentage (x : ℝ) : Prop :=
  final_price = initial_price * (1 - x / 100) * (1 - second_discount) * (1 - third_discount)

theorem first_discount_is_twenty_percent :
  first_discount_percentage 20 := by
  sorry

end first_discount_is_twenty_percent_l1597_159723


namespace negation_quadratic_inequality_l1597_159766

theorem negation_quadratic_inequality (x : ℝ) : 
  ¬(x^2 - x + 3 > 0) ↔ x^2 - x + 3 ≤ 0 := by sorry

end negation_quadratic_inequality_l1597_159766


namespace y1_greater_than_y2_l1597_159799

/-- Given two points M(-3, y₁) and N(2, y₂) on the line y = -3x + 1, prove that y₁ > y₂ -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : 
  (y₁ = -3 * (-3) + 1) → (y₂ = -3 * 2 + 1) → y₁ > y₂ := by
  sorry

end y1_greater_than_y2_l1597_159799


namespace special_trapezoid_base_ratio_l1597_159783

/-- A trapezoid with an inscribed and circumscribed circle, and one angle of 60 degrees -/
structure SpecialTrapezoid where
  /-- The trapezoid has an inscribed circle -/
  has_inscribed_circle : Bool
  /-- The trapezoid has a circumscribed circle -/
  has_circumscribed_circle : Bool
  /-- One angle of the trapezoid is 60 degrees -/
  has_60_degree_angle : Bool
  /-- The length of the longer base of the trapezoid -/
  longer_base : ℝ
  /-- The length of the shorter base of the trapezoid -/
  shorter_base : ℝ

/-- The ratio of the bases of a special trapezoid is 3:1 -/
theorem special_trapezoid_base_ratio (t : SpecialTrapezoid) :
  t.has_inscribed_circle ∧ t.has_circumscribed_circle ∧ t.has_60_degree_angle →
  t.longer_base / t.shorter_base = 3 := by
  sorry

end special_trapezoid_base_ratio_l1597_159783


namespace y_value_l1597_159730

/-- In an acute triangle, two altitudes divide the sides into segments of lengths 7, 3, 6, and y units. -/
structure AcuteTriangle where
  -- Define the segment lengths
  a : ℝ
  b : ℝ
  c : ℝ
  y : ℝ
  -- Conditions on the segment lengths
  ha : a = 7
  hb : b = 3
  hc : c = 6
  -- The triangle is acute (we don't use this directly, but it's part of the problem statement)
  acute : True

/-- The value of y in the acute triangle with given segment lengths is 7. -/
theorem y_value (t : AcuteTriangle) : t.y = 7 := by
  sorry

end y_value_l1597_159730


namespace original_price_after_discount_l1597_159716

theorem original_price_after_discount (a : ℝ) (h : a > 0) : 
  (4/5 : ℝ) * ((5/4 : ℝ) * a) = a := by sorry

end original_price_after_discount_l1597_159716


namespace unique_solution_condition_l1597_159704

/-- The equation (x+5)(x+2) = m + 3x has exactly one real solution if and only if m = 6 -/
theorem unique_solution_condition (m : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = m + 3 * x) ↔ m = 6 := by
  sorry

end unique_solution_condition_l1597_159704


namespace geometric_series_first_term_l1597_159725

theorem geometric_series_first_term (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : a = 20 / 3 := by
  sorry

end geometric_series_first_term_l1597_159725


namespace cost_of_45_lilies_l1597_159746

/-- The cost of a bouquet of lilies at Lila's Lily Shop -/
def bouquet_cost (n : ℕ) : ℚ :=
  let base_price := 2 * n  -- $2 per lily
  if n ≤ 30 then base_price else base_price * (1 - 1/10)  -- 10% discount if > 30 lilies

/-- Theorem: The cost of a bouquet with 45 lilies is $81 -/
theorem cost_of_45_lilies :
  bouquet_cost 45 = 81 := by sorry

end cost_of_45_lilies_l1597_159746


namespace lemonade_price_calculation_l1597_159772

theorem lemonade_price_calculation (glasses_per_gallon : ℕ) (cost_per_gallon : ℚ) 
  (gallons_made : ℕ) (glasses_drunk : ℕ) (glasses_unsold : ℕ) (net_profit : ℚ) :
  glasses_per_gallon = 16 →
  cost_per_gallon = 7/2 →
  gallons_made = 2 →
  glasses_drunk = 5 →
  glasses_unsold = 6 →
  net_profit = 14 →
  (gallons_made * cost_per_gallon + net_profit) / 
    (gallons_made * glasses_per_gallon - glasses_drunk - glasses_unsold) = 1 := by
  sorry

#eval (2 * (7/2 : ℚ) + 14) / (2 * 16 - 5 - 6)

end lemonade_price_calculation_l1597_159772


namespace game_sequence_repeats_a_2009_equals_65_l1597_159779

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The sequence defined by the game rules -/
def game_sequence (i : ℕ) : ℕ := 
  match i with
  | 0 => 5  -- n₁ = 5
  | i + 1 => sum_of_digits ((game_sequence i)^2 + 1)

/-- The a_i values in the sequence -/
def a_sequence (i : ℕ) : ℕ := (game_sequence i)^2 + 1

theorem game_sequence_repeats : 
  ∀ k : ℕ, k ≥ 3 → game_sequence k = game_sequence (k % 3) := sorry

theorem a_2009_equals_65 : a_sequence 2009 = 65 := by sorry

end game_sequence_repeats_a_2009_equals_65_l1597_159779


namespace max_cos_a_value_l1597_159732

theorem max_cos_a_value (a b c : Real) 
  (h1 : Real.sin a = Real.cos b)
  (h2 : Real.sin b = Real.cos c)
  (h3 : Real.sin c = Real.cos a) :
  ∃ (max_cos_a : Real), max_cos_a = Real.sqrt 2 / 2 ∧ 
    ∀ x, Real.cos a ≤ x → x ≤ max_cos_a :=
by sorry

end max_cos_a_value_l1597_159732


namespace complement_A_intersect_B_l1597_159771

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x > 0}
def B : Set ℝ := {x : ℝ | x > 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 4} := by sorry

end complement_A_intersect_B_l1597_159771


namespace median_salary_is_25000_l1597_159710

/-- Represents the salary distribution of a company -/
structure SalaryDistribution where
  ceo : ℕ × ℕ
  senior_manager : ℕ × ℕ
  manager : ℕ × ℕ
  assistant_manager : ℕ × ℕ
  clerk : ℕ × ℕ

/-- The total number of employees in the company -/
def total_employees (sd : SalaryDistribution) : ℕ :=
  sd.ceo.1 + sd.senior_manager.1 + sd.manager.1 + sd.assistant_manager.1 + sd.clerk.1

/-- The median index in a list of salaries -/
def median_index (n : ℕ) : ℕ :=
  (n + 1) / 2

/-- Finds the median salary given a salary distribution -/
def median_salary (sd : SalaryDistribution) : ℕ :=
  let total := total_employees sd
  let median_idx := median_index total
  if median_idx ≤ sd.ceo.1 then sd.ceo.2
  else if median_idx ≤ sd.ceo.1 + sd.senior_manager.1 then sd.senior_manager.2
  else if median_idx ≤ sd.ceo.1 + sd.senior_manager.1 + sd.manager.1 then sd.manager.2
  else if median_idx ≤ sd.ceo.1 + sd.senior_manager.1 + sd.manager.1 + sd.assistant_manager.1 then sd.assistant_manager.2
  else sd.clerk.2

/-- The company's salary distribution -/
def company_salaries : SalaryDistribution := {
  ceo := (1, 140000),
  senior_manager := (4, 95000),
  manager := (15, 80000),
  assistant_manager := (7, 55000),
  clerk := (40, 25000)
}

theorem median_salary_is_25000 :
  median_salary company_salaries = 25000 := by
  sorry

end median_salary_is_25000_l1597_159710


namespace p_20_equals_neg_8_l1597_159780

/-- A quadratic function with specific properties -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating that p(20) = -8 given the conditions -/
theorem p_20_equals_neg_8 (a b c : ℝ) :
  (∀ x, p a b c (19 - x) = p a b c x) →  -- Axis of symmetry at x = 9.5
  p a b c 0 = -8 →                       -- p(0) = -8
  p a b c 20 = -8 := by
  sorry

end p_20_equals_neg_8_l1597_159780


namespace min_value_2x_plus_y_l1597_159713

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/(y+1) = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 2/(b+1) = 2 → 2*x + y ≤ 2*a + b :=
by sorry

end min_value_2x_plus_y_l1597_159713


namespace exponential_function_not_multiplicative_l1597_159794

theorem exponential_function_not_multiplicative : ¬∀ a b : ℝ, Real.exp (a * b) = Real.exp a * Real.exp b := by
  sorry

end exponential_function_not_multiplicative_l1597_159794


namespace infinite_triples_exist_l1597_159728

/-- An infinite, strictly increasing sequence of positive integers -/
def StrictlyIncreasingSeq (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- The condition satisfied by the sequence -/
def SequenceCondition (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (a n) ≤ a n + a (n + 3)

/-- The existence of infinitely many triples satisfying the condition -/
def InfinitelyManyTriples (a : ℕ → ℕ) : Prop :=
  ∀ N : ℕ, ∃ k l m : ℕ, k > N ∧ l > k ∧ m > l ∧ a k + a m = 2 * a l

/-- The main theorem -/
theorem infinite_triples_exist (a : ℕ → ℕ) 
  (h1 : StrictlyIncreasingSeq a) 
  (h2 : SequenceCondition a) : 
  InfinitelyManyTriples a :=
sorry

end infinite_triples_exist_l1597_159728


namespace polar_to_cartesian_l1597_159707

theorem polar_to_cartesian (x y : ℝ) : 
  (∃ (ρ θ : ℝ), ρ = 3 ∧ θ = π/6 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) → 
  x = 3 * Real.sqrt 3 / 2 ∧ y = 3 / 2 := by
sorry

end polar_to_cartesian_l1597_159707


namespace max_notebooks_lucy_can_buy_l1597_159703

def lucy_money : ℕ := 2145
def notebook_cost : ℕ := 230

theorem max_notebooks_lucy_can_buy :
  (lucy_money / notebook_cost : ℕ) = 9 :=
by sorry

end max_notebooks_lucy_can_buy_l1597_159703


namespace earth_sun_max_distance_l1597_159760

/-- The semi-major axis of Earth's orbit in kilometers -/
def semi_major_axis : ℝ := 1.5e8

/-- The semi-minor axis of Earth's orbit in kilometers -/
def semi_minor_axis : ℝ := 3e6

/-- The maximum distance from Earth to Sun in kilometers -/
def max_distance : ℝ := semi_major_axis + semi_minor_axis

theorem earth_sun_max_distance :
  max_distance = 1.53e8 := by sorry

end earth_sun_max_distance_l1597_159760


namespace equation_root_in_interval_l1597_159769

theorem equation_root_in_interval : ∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2^x = 2 - x := by sorry

end equation_root_in_interval_l1597_159769


namespace min_fraction_value_l1597_159702

theorem min_fraction_value (x y : ℝ) (h : Real.sqrt (x - 1) + Real.sqrt (y - 1) = 1) :
  ∃ (min : ℝ), min = 1/2 ∧ ∀ (z : ℝ), z = x/y → z ≥ min :=
by sorry

end min_fraction_value_l1597_159702


namespace cos_alpha_value_l1597_159773

theorem cos_alpha_value (α : ℝ) (h : Real.sin (π / 2 + α) = 3 / 5) : 
  Real.cos α = 3 / 5 := by
sorry

end cos_alpha_value_l1597_159773


namespace valid_divisions_count_l1597_159720

/-- Represents a rectangle on the grid -/
structure Rectangle where
  x : Nat
  y : Nat
  width : Nat
  height : Nat

/-- Represents a division of the grid into 5 rectangles -/
structure GridDivision where
  center : Rectangle
  top : Rectangle
  bottom : Rectangle
  left : Rectangle
  right : Rectangle

/-- Checks if a rectangle touches the perimeter of an 11x11 grid -/
def touchesPerimeter (r : Rectangle) : Bool :=
  r.x = 0 || r.y = 0 || r.x + r.width = 11 || r.y + r.height = 11

/-- Checks if a grid division is valid -/
def isValidDivision (d : GridDivision) : Bool :=
  d.center.x > 0 && d.center.y > 0 && 
  d.center.x + d.center.width < 11 && 
  d.center.y + d.center.height < 11 &&
  touchesPerimeter d.top &&
  touchesPerimeter d.bottom &&
  touchesPerimeter d.left &&
  touchesPerimeter d.right

/-- Counts the number of valid grid divisions -/
def countValidDivisions : Nat :=
  sorry

theorem valid_divisions_count : countValidDivisions = 81 := by
  sorry

end valid_divisions_count_l1597_159720


namespace passengers_gained_at_halfway_l1597_159755

theorem passengers_gained_at_halfway (num_cars : ℕ) (initial_people_per_car : ℕ) (total_people_at_end : ℕ) : 
  num_cars = 20 →
  initial_people_per_car = 3 →
  total_people_at_end = 80 →
  (total_people_at_end - num_cars * initial_people_per_car) / num_cars = 1 :=
by sorry

end passengers_gained_at_halfway_l1597_159755


namespace time_to_work_calculation_l1597_159774

-- Define the problem parameters
def speed_to_work : ℝ := 50
def speed_to_home : ℝ := 110
def total_time : ℝ := 2

-- Define the theorem
theorem time_to_work_calculation :
  ∃ (distance : ℝ) (time_to_work : ℝ),
    distance / speed_to_work + distance / speed_to_home = total_time ∧
    time_to_work = distance / speed_to_work ∧
    time_to_work * 60 = 82.5 := by
  sorry


end time_to_work_calculation_l1597_159774


namespace mona_unique_players_l1597_159712

/-- The number of unique players Mona grouped with in a video game --/
def unique_players (total_groups : ℕ) (players_per_group : ℕ) (repeated_players : ℕ) : ℕ :=
  total_groups * players_per_group - repeated_players

/-- Theorem: Mona grouped with 33 unique players --/
theorem mona_unique_players :
  unique_players 9 4 3 = 33 := by
  sorry

end mona_unique_players_l1597_159712


namespace factorization_equality_l1597_159742

theorem factorization_equality (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) := by
  sorry

end factorization_equality_l1597_159742


namespace train_length_problem_l1597_159701

/-- Proves that the length of each train is 25 meters given the specified conditions -/
theorem train_length_problem (speed_fast speed_slow : ℝ) (passing_time : ℝ) :
  speed_fast = 46 →
  speed_slow = 36 →
  passing_time = 18 →
  let relative_speed := (speed_fast - speed_slow) * (5 / 18)
  let train_length := (relative_speed * passing_time) / 2
  train_length = 25 := by
sorry


end train_length_problem_l1597_159701


namespace parallelogram_height_l1597_159785

/-- Theorem: Height of a parallelogram with given area and base -/
theorem parallelogram_height (area base height : ℝ) : 
  area = 448 ∧ base = 32 ∧ area = base * height → height = 14 := by
  sorry

end parallelogram_height_l1597_159785


namespace remainder_8_900_mod_29_l1597_159787

theorem remainder_8_900_mod_29 : (8 : Nat)^900 % 29 = 7 := by
  sorry

end remainder_8_900_mod_29_l1597_159787


namespace card_collection_problem_l1597_159705

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of squares of the first n natural numbers -/
def sum_squares_first_n (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The average value of cards in a collection where each number k from 1 to n appears k times -/
def average_card_value (n : ℕ) : ℚ :=
  (sum_squares_first_n n : ℚ) / (sum_first_n n : ℚ)

theorem card_collection_problem :
  ∃ m : ℕ, average_card_value m = 56 ∧ m = 84 := by
  sorry

end card_collection_problem_l1597_159705


namespace hit_probability_theorem_l1597_159717

/-- The probability of hitting a target with one shot -/
def hit_probability : ℚ := 1 / 2

/-- The number of shots taken -/
def total_shots : ℕ := 6

/-- The number of hits required -/
def required_hits : ℕ := 3

/-- The number of consecutive hits required -/
def consecutive_hits : ℕ := 2

/-- The probability of hitting the target 3 times with exactly 2 consecutive hits out of 6 shots -/
def target_probability : ℚ := (Nat.choose 4 2 : ℚ) * (hit_probability ^ total_shots)

theorem hit_probability_theorem : 
  target_probability = (Nat.choose 4 2 : ℚ) * ((1 : ℚ) / 2) ^ 6 := by sorry

end hit_probability_theorem_l1597_159717


namespace function_value_at_inverse_l1597_159757

/-- Given a function f(x) = kx + 2/x^3 - 3 where k is a real number,
    if f(ln 6) = 1, then f(ln(1/6)) = -7 -/
theorem function_value_at_inverse (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ k * x + 2 / x^3 - 3
  f (Real.log 6) = 1 → f (Real.log (1/6)) = -7 := by
  sorry

end function_value_at_inverse_l1597_159757


namespace last_three_digits_of_3_to_12000_l1597_159709

theorem last_three_digits_of_3_to_12000 (h : 3^400 ≡ 1 [ZMOD 1000]) :
  3^12000 ≡ 1 [ZMOD 1000] := by
  sorry

end last_three_digits_of_3_to_12000_l1597_159709


namespace x_intercept_of_line_l1597_159743

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end x_intercept_of_line_l1597_159743


namespace correct_multiplication_l1597_159765

theorem correct_multiplication (x : ℚ) : 14 * x = 42 → 12 * x = 36 := by
  sorry

end correct_multiplication_l1597_159765


namespace farm_horses_cows_l1597_159777

theorem farm_horses_cows (initial_horses : ℕ) (initial_cows : ℕ) : 
  (initial_horses = 4 * initial_cows) →
  ((initial_horses - 15) / (initial_cows + 15) = 7 / 3) →
  (initial_horses - 15) - (initial_cows + 15) = 60 := by
  sorry

end farm_horses_cows_l1597_159777


namespace power_difference_equals_one_l1597_159792

theorem power_difference_equals_one (x y : ℕ) : 
  (2^x ∣ 180) ∧ 
  (3^y ∣ 180) ∧ 
  (∀ z : ℕ, z > x → ¬(2^z ∣ 180)) ∧ 
  (∀ w : ℕ, w > y → ¬(3^w ∣ 180)) → 
  (1/3 : ℚ)^(y - x) = 1 := by
sorry

end power_difference_equals_one_l1597_159792


namespace expression_value_l1597_159729

theorem expression_value : 
  let x : ℝ := 4
  (x^2 - 2*x - 15) / (x - 5) = 7 := by
sorry

end expression_value_l1597_159729


namespace two_std_dev_less_than_mean_example_l1597_159752

/-- For a normal distribution with given mean and standard deviation,
    calculate the value that is exactly 2 standard deviations less than the mean -/
def twoStdDevLessThanMean (mean : ℝ) (stdDev : ℝ) : ℝ :=
  mean - 2 * stdDev

/-- Theorem stating that for a normal distribution with mean 12 and standard deviation 1.2,
    the value exactly 2 standard deviations less than the mean is 9.6 -/
theorem two_std_dev_less_than_mean_example :
  twoStdDevLessThanMean 12 1.2 = 9.6 := by
  sorry

end two_std_dev_less_than_mean_example_l1597_159752


namespace time_after_1750_minutes_l1597_159756

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

/-- Converts a number of minutes to hours and minutes -/
def minutesToTime (m : Nat) : Time :=
  sorry

theorem time_after_1750_minutes :
  let start_time : Time := ⟨8, 0, by sorry, by sorry⟩
  let added_time : Time := minutesToTime 1750
  let end_time : Time := addMinutes start_time 1750
  end_time = ⟨13, 10, by sorry, by sorry⟩ := by
  sorry

end time_after_1750_minutes_l1597_159756


namespace consecutive_integers_sum_of_cubes_l1597_159767

theorem consecutive_integers_sum_of_cubes (n : ℤ) : 
  n > 0 ∧ (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 11534 →
  (n - 1)^3 + n^3 + (n + 1)^3 + (n + 2)^3 = 74836 :=
by sorry

end consecutive_integers_sum_of_cubes_l1597_159767


namespace kendra_sunday_shirts_l1597_159715

/-- The number of shirts Kendra wears in two weeks -/
def total_shirts : ℕ := 22

/-- The number of weekdays in two weeks -/
def weekdays : ℕ := 10

/-- The number of days Kendra changes shirts for after-school club in two weeks -/
def club_days : ℕ := 6

/-- The number of Saturdays in two weeks -/
def saturdays : ℕ := 2

/-- The number of Sundays in two weeks -/
def sundays : ℕ := 2

/-- The number of shirts Kendra wears on weekdays for school and club in two weeks -/
def weekday_shirts : ℕ := weekdays + club_days

/-- The number of shirts Kendra wears on Saturdays in two weeks -/
def saturday_shirts : ℕ := saturdays

theorem kendra_sunday_shirts :
  total_shirts - (weekday_shirts + saturday_shirts) = 4 := by
sorry

end kendra_sunday_shirts_l1597_159715


namespace imaginary_unit_sum_zero_l1597_159724

theorem imaginary_unit_sum_zero (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end imaginary_unit_sum_zero_l1597_159724


namespace quadrilateral_angle_equality_l1597_159727

-- Define the points
variable (A B C D E F P : Point)

-- Define the quadrilateral
def is_quadrilateral (A B C D : Point) : Prop := sorry

-- Define that a point is on a line segment
def on_segment (X Y Z : Point) : Prop := sorry

-- Define that two lines intersect at a point
def lines_intersect_at (W X Y Z P : Point) : Prop := sorry

-- Define angle equality
def angle_eq (A B C D E F : Point) : Prop := sorry

-- State the theorem
theorem quadrilateral_angle_equality 
  (h1 : is_quadrilateral A B C D)
  (h2 : on_segment B E C)
  (h3 : on_segment C F D)
  (h4 : lines_intersect_at B F D E P)
  (h5 : angle_eq B A E F A D) :
  angle_eq B A P C A D := by sorry

end quadrilateral_angle_equality_l1597_159727


namespace first_valid_year_is_2049_l1597_159759

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2010 ∧ year < 3000 ∧ sum_of_digits year = 15

theorem first_valid_year_is_2049 :
  (∀ year : ℕ, year < 2049 → ¬(is_valid_year year)) ∧ 
  is_valid_year 2049 :=
sorry

end first_valid_year_is_2049_l1597_159759


namespace goldfish_count_l1597_159741

/-- The number of goldfish in the fish tank -/
def num_goldfish : ℕ := sorry

/-- The number of platyfish in the fish tank -/
def num_platyfish : ℕ := 10

/-- The number of red balls each goldfish plays with -/
def red_balls_per_goldfish : ℕ := 10

/-- The number of white balls each platyfish plays with -/
def white_balls_per_platyfish : ℕ := 5

/-- The total number of balls in the fish tank -/
def total_balls : ℕ := 80

theorem goldfish_count : num_goldfish = 3 := by
  have h1 : num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish = total_balls := sorry
  sorry

end goldfish_count_l1597_159741


namespace roots_sum_powers_l1597_159726

theorem roots_sum_powers (γ δ : ℝ) : 
  γ^2 - 5*γ + 6 = 0 → δ^2 - 5*δ + 6 = 0 → 8*γ^5 + 15*δ^4 = 8425 := by
  sorry

end roots_sum_powers_l1597_159726


namespace borrowed_amount_l1597_159761

/-- Proves that the amount borrowed is 5000 given the specified conditions --/
theorem borrowed_amount (loan_duration : ℕ) (borrow_rate lend_rate : ℚ) (gain_per_year : ℕ) : 
  loan_duration = 2 →
  borrow_rate = 4 / 100 →
  lend_rate = 8 / 100 →
  gain_per_year = 200 →
  ∃ (amount : ℕ), amount = 5000 ∧ 
    (amount * lend_rate * loan_duration) - (amount * borrow_rate * loan_duration) = gain_per_year * loan_duration :=
by sorry

end borrowed_amount_l1597_159761


namespace quadratic_function_and_area_bisection_l1597_159770

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  equal_roots : ∃ r : ℝ, (∀ x, f x = 0 ↔ x = r)
  derivative : ∀ x, HasDerivAt f (2 * x + 2) x

/-- The main theorem about the quadratic function and area bisection -/
theorem quadratic_function_and_area_bisection (qf : QuadraticFunction) :
  (∀ x, qf.f x = x^2 + 2*x + 1) ∧
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧
    (∫ x in (-1)..(-t), qf.f x) = (∫ x in (-t)..0, qf.f x) ∧
    t = 1 - 1 / Real.sqrt 2 := by sorry

end quadratic_function_and_area_bisection_l1597_159770


namespace extreme_value_implies_a_increasing_implies_a_nonnegative_l1597_159762

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

-- Part 1: Extreme value at x = 3 implies a = 3
theorem extreme_value_implies_a (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (3 - ε) (3 + ε), f a x ≤ f a 3) →
  a = 3 :=
sorry

-- Part 2: Increasing on (-∞, 0) implies a ∈ [0, +∞)
theorem increasing_implies_a_nonnegative (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y < 0 → f a x < f a y) →
  a ≥ 0 :=
sorry

end extreme_value_implies_a_increasing_implies_a_nonnegative_l1597_159762


namespace first_number_proof_l1597_159744

theorem first_number_proof (x y : ℝ) : 
  x + y = 10 → 2 * x = 3 * y + 5 → x = 7 := by sorry

end first_number_proof_l1597_159744


namespace binomial_expansion_example_l1597_159700

theorem binomial_expansion_example : 57^3 + 3*(57^2)*4 + 3*57*(4^2) + 4^3 = 226981 := by
  sorry

end binomial_expansion_example_l1597_159700


namespace equation_solution_l1597_159722

theorem equation_solution (a b : ℝ) (h : a ≠ 0) :
  let x : ℝ := (a^2 - b^2) / a
  x^2 + 4 * b^2 = (2 * a - x)^2 :=
by sorry

end equation_solution_l1597_159722


namespace eleven_integer_chords_l1597_159740

/-- Represents a circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distance_to_p : ℝ

/-- Counts the number of integer-length chords containing P -/
def count_integer_chords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem -/
theorem eleven_integer_chords :
  let c := CircleWithPoint.mk 17 12
  count_integer_chords c = 11 := by sorry

end eleven_integer_chords_l1597_159740


namespace oil_bill_problem_l1597_159786

/-- The oil bill problem -/
theorem oil_bill_problem (jan_bill feb_bill : ℝ) 
  (h1 : feb_bill / jan_bill = 5 / 4)
  (h2 : (feb_bill + 45) / jan_bill = 3 / 2) :
  jan_bill = 180 := by
  sorry

end oil_bill_problem_l1597_159786


namespace pure_imaginary_product_l1597_159711

theorem pure_imaginary_product (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (y : ℝ), (3 - 4 * Complex.I) * (a + b * Complex.I) = y * Complex.I) : 
  a / b = -4 / 3 := by
sorry

end pure_imaginary_product_l1597_159711


namespace sum_of_special_integers_l1597_159753

theorem sum_of_special_integers (n : ℕ) : 
  (∃ (S : Finset ℕ), 
    (∀ m ∈ S, m < 100 ∧ m > 0 ∧ ∃ k : ℤ, 5 * m^2 + 3 * m - 5 = 15 * k) ∧ 
    (∀ m : ℕ, m < 100 ∧ m > 0 ∧ (∃ k : ℤ, 5 * m^2 + 3 * m - 5 = 15 * k) → m ∈ S) ∧
    (Finset.sum S id = 635)) :=
by sorry

end sum_of_special_integers_l1597_159753


namespace problem_solution_l1597_159714

theorem problem_solution : 
  (1 - 1^2022 - (3 * (2/3)^2 - 8/3 / (-2)^3) = -8/3) ∧ 
  (2^3 / 3 * (-1/4 + 7/12 - 5/6) / (-1/18) = 24) := by
  sorry

end problem_solution_l1597_159714


namespace four_digit_divisible_by_18_l1597_159789

theorem four_digit_divisible_by_18 : 
  ∀ n : ℕ, n < 10 → (4150 + n) % 18 = 0 ↔ n = 8 := by sorry

end four_digit_divisible_by_18_l1597_159789


namespace no_inverse_implies_x_equals_five_l1597_159748

def M (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![x, 5],
    ![6, 6]]

theorem no_inverse_implies_x_equals_five :
  ∀ x : ℝ, ¬(IsUnit (M x)) → x = 5 := by
  sorry

end no_inverse_implies_x_equals_five_l1597_159748


namespace book_price_increase_l1597_159747

theorem book_price_increase (new_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) : 
  new_price = 390 ∧ increase_percentage = 30 →
  original_price * (1 + increase_percentage / 100) = new_price →
  original_price = 300 := by
sorry

end book_price_increase_l1597_159747


namespace monthly_income_p_l1597_159798

/-- Given the average monthly incomes of pairs of individuals, prove that the monthly income of p is 4000. -/
theorem monthly_income_p (p q r : ℕ) : 
  (p + q) / 2 = 5050 →
  (q + r) / 2 = 6250 →
  (p + r) / 2 = 5200 →
  p = 4000 := by
sorry

end monthly_income_p_l1597_159798


namespace intersection_and_parallel_line_l1597_159788

-- Define the two lines
def l₁ (x y : ℝ) : Prop := x + 8*y + 7 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := x + y + 1 = 0

theorem intersection_and_parallel_line :
  -- Part 1: Prove that (1, -1) is the intersection point
  (l₁ 1 (-1) ∧ l₂ 1 (-1)) ∧
  -- Part 2: Prove that x + y = 0 is the equation of the line passing through
  -- the intersection point and parallel to x + y + 1 = 0
  (∃ c : ℝ, ∀ x y : ℝ, (l₁ x y ∧ l₂ x y) → x + y = c) ∧
  (∀ x y : ℝ, (x + y = 0) ↔ (∃ k : ℝ, x = 1 + k ∧ y = -1 - k ∧ parallel_line (1 + k) (-1 - k))) :=
sorry

end intersection_and_parallel_line_l1597_159788


namespace jills_peaches_l1597_159781

/-- Given that Steven has 19 peaches and 13 more peaches than Jill,
    prove that Jill has 6 peaches. -/
theorem jills_peaches (steven_peaches : ℕ) (steven_jill_diff : ℕ) 
  (h1 : steven_peaches = 19)
  (h2 : steven_peaches = steven_jill_diff + jill_peaches) :
  jill_peaches = 6 :=
by
  sorry

end jills_peaches_l1597_159781


namespace train_length_l1597_159736

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 144 →
  time_s = 8.7493 →
  length_m = speed_kmh * (1000 / 3600) * time_s →
  length_m = 350 := by
sorry

end train_length_l1597_159736


namespace smallest_candy_number_l1597_159718

theorem smallest_candy_number : ∃ (n : ℕ), 
  100 ≤ n ∧ n < 1000 ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 9) % 7 = 0 ∧
  ∀ m, 100 ≤ m ∧ m < n → ¬((m + 7) % 9 = 0 ∧ (m - 9) % 7 = 0) :=
by
  use 110
  sorry

end smallest_candy_number_l1597_159718


namespace tims_earnings_per_visit_l1597_159784

theorem tims_earnings_per_visit
  (visitors_per_day : ℕ)
  (regular_days : ℕ)
  (total_earnings : ℚ)
  (h1 : visitors_per_day = 100)
  (h2 : regular_days = 6)
  (h3 : total_earnings = 18)
  : 
  let total_visitors := visitors_per_day * regular_days + 2 * (visitors_per_day * regular_days)
  total_earnings / total_visitors = 1 / 100 := by
  sorry

end tims_earnings_per_visit_l1597_159784


namespace soccer_field_kids_l1597_159745

theorem soccer_field_kids (initial_kids joining_kids : ℕ) : 
  initial_kids = 14 → joining_kids = 22 → initial_kids + joining_kids = 36 := by
sorry

end soccer_field_kids_l1597_159745


namespace find_number_l1597_159758

theorem find_number (x : ℝ) : ((4 * x - 28) / 7 + 12 = 36) → x = 49 := by
  sorry

end find_number_l1597_159758


namespace square_circle_equal_area_l1597_159750

theorem square_circle_equal_area (r : ℝ) (s : ℝ) : 
  r = 5 →
  s = 2 * r →
  s^2 = π * r^2 →
  s = 5 * Real.sqrt π :=
by sorry

end square_circle_equal_area_l1597_159750


namespace opposite_colors_in_prism_l1597_159731

-- Define the set of colors
inductive Color
  | Red
  | Yellow
  | Blue
  | Black
  | White
  | Green

-- Define a cube as a function from faces to colors
def Cube := Fin 6 → Color

-- Define the property of having all different colors
def allDifferentColors (c : Cube) : Prop :=
  ∀ i j : Fin 6, i ≠ j → c i ≠ c j

-- Define the property of opposite faces having the same color in a rectangular prism
def oppositeColorsSame (c : Cube) : Prop :=
  (c 0 = Color.Red ∧ c 5 = Color.Green) ∨
  (c 0 = Color.Green ∧ c 5 = Color.Red) ∧
  (c 1 = Color.Yellow ∧ c 4 = Color.Blue) ∨
  (c 1 = Color.Blue ∧ c 4 = Color.Yellow) ∧
  (c 2 = Color.Black ∧ c 3 = Color.White) ∨
  (c 2 = Color.White ∧ c 3 = Color.Black)

-- Theorem stating the opposite colors in the rectangular prism
theorem opposite_colors_in_prism (c : Cube) 
  (h1 : allDifferentColors c) 
  (h2 : oppositeColorsSame c) :
  (c 0 = Color.Red → c 5 = Color.Green) ∧
  (c 1 = Color.Yellow → c 4 = Color.Blue) ∧
  (c 2 = Color.Black → c 3 = Color.White) :=
by sorry

end opposite_colors_in_prism_l1597_159731


namespace cube_edge_length_range_l1597_159795

theorem cube_edge_length_range (V : ℝ) (a : ℝ) (h1 : V = 9) (h2 : V = a^3) :
  2 < a ∧ a < 2.5 := by
  sorry

end cube_edge_length_range_l1597_159795


namespace subtraction_preserves_inequality_l1597_159739

theorem subtraction_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end subtraction_preserves_inequality_l1597_159739


namespace chess_tournament_participants_l1597_159737

/-- The number of participants in the chess tournament. -/
def n : ℕ := 19

/-- The number of matches played in a round-robin tournament. -/
def matches_played (x : ℕ) : ℕ := x * (x - 1) / 2

/-- The number of matches played after three players dropped out. -/
def matches_after_dropout (x : ℕ) : ℕ := (x - 3) * (x - 4) / 2

/-- Theorem stating that the number of participants in the chess tournament is 19. -/
theorem chess_tournament_participants :
  matches_played n - matches_after_dropout n = 130 :=
by sorry

end chess_tournament_participants_l1597_159737


namespace square_of_binomial_l1597_159791

theorem square_of_binomial (x : ℝ) : ∃ (a : ℝ), x^2 - 20*x + 100 = (x - a)^2 := by
  sorry

end square_of_binomial_l1597_159791


namespace smallest_non_prime_a_l1597_159778

def a : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * a n + 1

theorem smallest_non_prime_a (n : ℕ) : 
  (∀ k < n, Nat.Prime (a k)) ∧ ¬Nat.Prime (a n) → n = 5 ∧ a n = 95 := by
  sorry

end smallest_non_prime_a_l1597_159778


namespace mans_age_to_sons_age_ratio_l1597_159790

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given that the man is 37 years older than his son and the son's current age is 35. -/
theorem mans_age_to_sons_age_ratio :
  let sons_current_age : ℕ := 35
  let mans_current_age : ℕ := sons_current_age + 37
  let sons_age_in_two_years : ℕ := sons_current_age + 2
  let mans_age_in_two_years : ℕ := mans_current_age + 2
  (mans_age_in_two_years : ℚ) / (sons_age_in_two_years : ℚ) = 2 := by
sorry

end mans_age_to_sons_age_ratio_l1597_159790


namespace normal_distribution_probability_l1597_159793

/-- A random variable following a normal distribution with mean 2 and some variance σ² -/
noncomputable def ξ : Real → ℝ := sorry

/-- The probability density function of ξ -/
noncomputable def pdf_ξ : ℝ → ℝ := sorry

/-- The cumulative distribution function of ξ -/
noncomputable def cdf_ξ : ℝ → ℝ := sorry

/-- The condition that P(ξ < 4) = 0.8 -/
axiom cdf_at_4 : cdf_ξ 4 = 0.8

/-- The theorem to prove -/
theorem normal_distribution_probability :
  cdf_ξ 2 - cdf_ξ 0 = 0.3 := by sorry

end normal_distribution_probability_l1597_159793


namespace solve_cube_equation_l1597_159754

theorem solve_cube_equation : ∃ x : ℝ, (x - 5)^3 = -(1/27)⁻¹ ∧ x = 2 := by
  sorry

end solve_cube_equation_l1597_159754


namespace range_of_a_l1597_159763

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 + 2*a*x - 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 < 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
  (a > 0) →
  (∀ x : ℝ, p x a → q x) →
  (∃ x : ℝ, q x ∧ ¬(p x a)) →
  (0 < a ∧ a ≤ 4/3) :=
by sorry

end range_of_a_l1597_159763


namespace largest_valid_number_l1597_159776

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (n % 10 = (n / 10 % 10 + n / 100 % 10) % 10) ∧
  (n / 10 % 10 = (n / 100 % 10 + n / 1000 % 10) % 10)

theorem largest_valid_number : 
  is_valid_number 9099 ∧ ∀ m : ℕ, is_valid_number m → m ≤ 9099 :=
by sorry

end largest_valid_number_l1597_159776


namespace odd_symmetric_function_property_l1597_159751

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ is symmetric about x = 3 if f(3+x) = f(3-x) for all x -/
def SymmetricAboutThree (f : ℝ → ℝ) : Prop :=
  ∀ x, f (3 + x) = f (3 - x)

theorem odd_symmetric_function_property (f : ℝ → ℝ) 
  (h_odd : OddFunction f)
  (h_sym : SymmetricAboutThree f)
  (h_def : ∀ x ∈ Set.Ioo 0 3, f x = 2^x) :
  ∀ x ∈ Set.Ioo (-6) (-3), f x = -(2^(x + 6)) := by
  sorry

end odd_symmetric_function_property_l1597_159751


namespace problem_solution_l1597_159708

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : y = x / (2 * x + 1)) :
  (3 * x - 3 * y + x * y) / (4 * x * y) = 7 / 4 := by
sorry

end problem_solution_l1597_159708


namespace median_sum_lower_bound_l1597_159775

/-- Given a triangle ABC with sides a, b, c and medians ma, mb, mc, 
    the sum of the lengths of the medians is at least three quarters of its perimeter. -/
theorem median_sum_lower_bound (a b c ma mb mc : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_pos_ma : ma > 0) (h_pos_mb : mb > 0) (h_pos_mc : mc > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_ma : ma^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_mb : mb^2 = (2*c^2 + 2*a^2 - b^2) / 4)
  (h_mc : mc^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  ma + mb + mc ≥ 3/4 * (a + b + c) := by
  sorry

end median_sum_lower_bound_l1597_159775


namespace sum_square_values_l1597_159782

theorem sum_square_values (K M : ℕ) : 
  K * (K + 1) / 2 = M^2 →
  M < 200 →
  K > M →
  (K = 8 ∨ K = 49) ∧ 
  (∀ n : ℕ, n * (n + 1) / 2 = M^2 ∧ M < 200 ∧ n > M → n = 8 ∨ n = 49) :=
by sorry

end sum_square_values_l1597_159782


namespace five_digit_with_four_or_five_l1597_159764

/-- The number of five-digit positive integers -/
def total_five_digit : ℕ := 90000

/-- The number of digits that are not 4 or 5 -/
def non_four_five_digits : ℕ := 8

/-- The number of options for the first digit (excluding 0, 4, and 5) -/
def first_digit_options : ℕ := 7

/-- The number of five-digit positive integers without 4 or 5 -/
def without_four_five : ℕ := first_digit_options * non_four_five_digits^4

theorem five_digit_with_four_or_five :
  total_five_digit - without_four_five = 61328 := by
  sorry

end five_digit_with_four_or_five_l1597_159764


namespace home_appliances_promotion_l1597_159733

theorem home_appliances_promotion (salespersons technicians : ℕ) 
  (h1 : salespersons = 5)
  (h2 : technicians = 4)
  (h3 : salespersons + technicians = 9) :
  (Nat.choose 9 3) - (Nat.choose 5 3) - (Nat.choose 4 3) = 70 := by
  sorry

end home_appliances_promotion_l1597_159733


namespace cricketer_average_after_22nd_inning_l1597_159719

/-- Represents the average score of a cricketer before the 22nd inning -/
def initial_average : ℝ := sorry

/-- The score made by the cricketer in the 22nd inning -/
def score_22nd_inning : ℝ := 134

/-- The increase in average after the 22nd inning -/
def average_increase : ℝ := 3.5

/-- The number of innings played before the 22nd inning -/
def previous_innings : ℕ := 21

/-- The total number of innings including the 22nd inning -/
def total_innings : ℕ := 22

/-- Calculates the new average after the 22nd inning -/
def new_average : ℝ := initial_average + average_increase

/-- Theorem stating that the new average after the 22nd inning is 60.5 -/
theorem cricketer_average_after_22nd_inning : 
  (previous_innings : ℝ) * initial_average + score_22nd_inning = 
    new_average * (total_innings : ℝ) ∧ new_average = 60.5 := by sorry

end cricketer_average_after_22nd_inning_l1597_159719


namespace function_satisfies_conditions_l1597_159768

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 + 3 * x

-- State the theorem
theorem function_satisfies_conditions :
  (f 1 = 1) ∧ 
  (∃ x y, x > 0 ∧ y < 0 ∧ f x = y) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) := by
  sorry

end function_satisfies_conditions_l1597_159768


namespace power_of_three_in_product_l1597_159735

theorem power_of_three_in_product (w : ℕ+) : 
  (∃ k : ℕ, 936 * w = 2^5 * 11^2 * k) → 
  (132 ≤ w) →
  (∃ m : ℕ, 936 * w = 3^3 * m ∧ ∀ n > 3, ¬(∃ l : ℕ, 936 * w = 3^n * l)) := by
  sorry

end power_of_three_in_product_l1597_159735


namespace total_length_climbed_50_30_6_25_l1597_159796

/-- The total length of ladders climbed by two workers in centimeters -/
def total_length_climbed (keaton_ladder_height : ℕ) (keaton_climbs : ℕ) 
  (reece_ladder_diff : ℕ) (reece_climbs : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - reece_ladder_diff
  let keaton_total := keaton_ladder_height * keaton_climbs
  let reece_total := reece_ladder_height * reece_climbs
  (keaton_total + reece_total) * 100

/-- Theorem stating the total length climbed by both workers -/
theorem total_length_climbed_50_30_6_25 :
  total_length_climbed 50 30 6 25 = 260000 := by
  sorry

end total_length_climbed_50_30_6_25_l1597_159796


namespace direction_cannot_determine_position_l1597_159734

-- Define a type for positions
structure Position where
  x : ℝ
  y : ℝ

-- Define a type for directions
structure Direction where
  angle : ℝ

-- Define a function to check if a piece of data can determine a position
def canDeterminePosition (data : Type) : Prop :=
  ∃ (f : data → Position), Function.Injective f

-- Theorem statement
theorem direction_cannot_determine_position :
  ¬ (canDeterminePosition Direction) :=
sorry

end direction_cannot_determine_position_l1597_159734


namespace tangent_line_equation_l1597_159749

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x - 4

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -3
  let k : ℝ := f' x₀
  (f x₀ = y₀) ∧ 
  (∀ x y : ℝ, y = k * (x - x₀) + y₀ ↔ 5*x + y - 2 = 0) :=
sorry

end tangent_line_equation_l1597_159749


namespace expression_value_l1597_159738

theorem expression_value : 50 * (50 - 5) - (50 * 50 - 5) = -245 := by
  sorry

end expression_value_l1597_159738


namespace abs_sum_reciprocals_ge_two_l1597_159721

theorem abs_sum_reciprocals_ge_two (a b : ℝ) (h : a * b ≠ 0) :
  |a / b + b / a| ≥ 2 := by
  sorry

end abs_sum_reciprocals_ge_two_l1597_159721


namespace binomial_coefficient_divisibility_l1597_159797

theorem binomial_coefficient_divisibility (p : ℕ) (hp : Nat.Prime p) :
  (Nat.choose (2 * p) p - 2) % (p^2) = 0 := by
  sorry

end binomial_coefficient_divisibility_l1597_159797


namespace game_probability_l1597_159706

/-- Represents a player in the game -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Cindy : Player
| Dave : Player

/-- The game state is represented by a function from Player to ℕ (natural numbers) -/
def GameState : Type := Player → ℕ

/-- The initial state of the game where each player has 2 units -/
def initialState : GameState :=
  fun _ => 2

/-- A single round of the game -/
def gameRound (state : GameState) : GameState :=
  sorry -- Implementation details omitted

/-- The probability of a specific outcome after one round -/
def roundProbability (initialState finalState : GameState) : ℚ :=
  sorry -- Implementation details omitted

/-- The probability of all players having 2 units after 5 rounds -/
def finalProbability : ℚ :=
  sorry -- Implementation details omitted

/-- The main theorem stating the probability of all players having 2 units after 5 rounds -/
theorem game_probability : finalProbability = 4 / 81^5 := by
  sorry

end game_probability_l1597_159706
