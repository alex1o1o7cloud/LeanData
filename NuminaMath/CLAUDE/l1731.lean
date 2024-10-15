import Mathlib

namespace NUMINAMATH_CALUDE_shooting_is_impossible_coin_toss_is_random_triangle_angles_is_certain_l1731_173101

-- Define the types of events
inductive EventType
  | Impossible
  | Random
  | Certain

-- Define the events
def shooting_event : EventType := EventType.Impossible
def coin_toss_event : EventType := EventType.Random
def triangle_angles_event : EventType := EventType.Certain

-- Theorem statements
theorem shooting_is_impossible : shooting_event = EventType.Impossible := by sorry

theorem coin_toss_is_random : coin_toss_event = EventType.Random := by sorry

theorem triangle_angles_is_certain : triangle_angles_event = EventType.Certain := by sorry

end NUMINAMATH_CALUDE_shooting_is_impossible_coin_toss_is_random_triangle_angles_is_certain_l1731_173101


namespace NUMINAMATH_CALUDE_binary_property_l1731_173164

-- Define the number in base 10
def base10Number : Nat := 235

-- Define a function to convert a number to binary
def toBinary (n : Nat) : List Nat := sorry

-- Define a function to count zeros in a binary representation
def countZeros (binary : List Nat) : Nat := sorry

-- Define a function to count ones in a binary representation
def countOnes (binary : List Nat) : Nat := sorry

-- Theorem statement
theorem binary_property :
  let binary := toBinary base10Number
  let x := countZeros binary
  let y := countOnes binary
  y^2 - 2*x = 32 := by sorry

end NUMINAMATH_CALUDE_binary_property_l1731_173164


namespace NUMINAMATH_CALUDE_junior_count_in_club_l1731_173153

theorem junior_count_in_club (total_students : ℕ) 
  (junior_selection_rate : ℚ) (senior_selection_rate : ℚ) : ℕ :=
by
  sorry

#check junior_count_in_club 30 (2/5) (1/4) = 11

end NUMINAMATH_CALUDE_junior_count_in_club_l1731_173153


namespace NUMINAMATH_CALUDE_geometric_mean_problem_l1731_173126

/-- Given a geometric sequence {a_n} where a_2 = 9 and a_5 = 243,
    the geometric mean of a_1 and a_7 is ±81. -/
theorem geometric_mean_problem (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 = 9 →
  a 5 = 243 →
  (∃ x : ℝ, x ^ 2 = a 1 * a 7 ∧ (x = 81 ∨ x = -81)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_problem_l1731_173126


namespace NUMINAMATH_CALUDE_gw_to_w_conversion_l1731_173142

/-- Conversion factor from gigawatts to watts -/
def gw_to_w : ℝ := 1000000000

/-- The newly installed capacity in gigawatts -/
def installed_capacity : ℝ := 125

/-- Theorem stating that 125 gigawatts is equal to 1.25 × 10^11 watts -/
theorem gw_to_w_conversion :
  installed_capacity * gw_to_w = 1.25 * (10 : ℝ) ^ 11 := by sorry

end NUMINAMATH_CALUDE_gw_to_w_conversion_l1731_173142


namespace NUMINAMATH_CALUDE_product_difference_square_l1731_173168

theorem product_difference_square : 2012 * 2016 - 2014^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_square_l1731_173168


namespace NUMINAMATH_CALUDE_max_distance_complex_numbers_l1731_173120

theorem max_distance_complex_numbers :
  ∃ (M : ℝ), M = 81 + 9 * Real.sqrt 5 ∧
  ∀ (z : ℂ), Complex.abs z = 3 →
  Complex.abs ((1 + 2*Complex.I) * z^2 - z^4) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_distance_complex_numbers_l1731_173120


namespace NUMINAMATH_CALUDE_smallest_green_points_l1731_173176

/-- The total number of points in the plane -/
def total_points : ℕ := 2020

/-- The distance between a black point and its two associated green points -/
def distance : ℕ := 2020

/-- The property that for each black point, there are exactly two green points at the specified distance -/
def black_point_property (n : ℕ) : Prop :=
  ∀ b : ℕ, b ≤ n * (n - 1)

/-- The theorem stating the smallest number of green points -/
theorem smallest_green_points :
  ∃ n : ℕ, n = 45 ∧ 
    black_point_property n ∧
    n + (total_points - n) = total_points ∧
    ∀ m : ℕ, m < n → ¬(black_point_property m ∧ m + (total_points - m) = total_points) :=
by sorry

end NUMINAMATH_CALUDE_smallest_green_points_l1731_173176


namespace NUMINAMATH_CALUDE_ashton_pencils_left_l1731_173194

def pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (given_away : ℕ) : ℕ :=
  boxes * pencils_per_box - given_away

theorem ashton_pencils_left : pencils_left 2 14 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_ashton_pencils_left_l1731_173194


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1731_173186

-- Define the universal set U
def U : Set Int := {-1, 0, 1, 2}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {0, 2}

-- Theorem statement
theorem complement_A_intersect_B : (U \ A) ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1731_173186


namespace NUMINAMATH_CALUDE_fold_line_length_l1731_173160

-- Define the triangle
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let a := dist B C
  let b := dist A C
  let c := dist A B
  a = 5 ∧ b = 12 ∧ c = 13

-- Define the right angle at C
def right_angle_at_C (A B C : ℝ × ℝ) : Prop :=
  (dist B C)^2 + (dist A C)^2 = (dist A B)^2

-- Define the perpendicular bisector of AB
def perp_bisector_AB (A B : ℝ × ℝ) (D : ℝ × ℝ) : Prop :=
  dist A D = dist B D ∧ 
  (A.1 - B.1) * (D.1 - A.1) + (A.2 - B.2) * (D.2 - A.2) = 0

-- Theorem statement
theorem fold_line_length 
  (A B C : ℝ × ℝ) 
  (h1 : triangle_ABC A B C) 
  (h2 : right_angle_at_C A B C) :
  ∃ D : ℝ × ℝ, perp_bisector_AB A B D ∧ dist C D = Real.sqrt 7.33475 :=
sorry

end NUMINAMATH_CALUDE_fold_line_length_l1731_173160


namespace NUMINAMATH_CALUDE_girls_on_playground_l1731_173154

theorem girls_on_playground (total_children boys : ℕ) 
  (h1 : total_children = 63) 
  (h2 : boys = 35) : 
  total_children - boys = 28 := by
sorry

end NUMINAMATH_CALUDE_girls_on_playground_l1731_173154


namespace NUMINAMATH_CALUDE_grade_10_sample_size_l1731_173130

/-- Represents the number of students to be sampled from a grade in a stratified sampling scenario -/
def stratified_sample (total_sample : ℕ) (grade_ratio : ℕ) (total_ratio : ℕ) : ℕ :=
  (grade_ratio * total_sample) / total_ratio

/-- Theorem stating that in a stratified sampling of 65 students from three grades with a ratio of 4:4:5, 
    the number of students to be sampled from the first grade is 20 -/
theorem grade_10_sample_size :
  stratified_sample 65 4 (4 + 4 + 5) = 20 := by
  sorry

end NUMINAMATH_CALUDE_grade_10_sample_size_l1731_173130


namespace NUMINAMATH_CALUDE_dot_product_equals_negative_31_l1731_173181

def vector1 : Fin 2 → ℝ
  | 0 => -3
  | 1 => 2

def vector2 : Fin 2 → ℝ
  | 0 => 7
  | 1 => -5

theorem dot_product_equals_negative_31 :
  (Finset.univ.sum fun i => vector1 i * vector2 i) = -31 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_equals_negative_31_l1731_173181


namespace NUMINAMATH_CALUDE_strictly_increasing_implies_monotone_increasing_l1731_173179

-- Define a function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property that for any x₁ < x₂, f(x₁) < f(x₂)
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

-- Theorem statement
theorem strictly_increasing_implies_monotone_increasing
  (h : StrictlyIncreasing f) : MonotoneOn f Set.univ :=
sorry

end NUMINAMATH_CALUDE_strictly_increasing_implies_monotone_increasing_l1731_173179


namespace NUMINAMATH_CALUDE_coefficient_expansion_l1731_173110

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def coefficient_x2y3z2 (a b c : ℕ → ℕ → ℕ → ℕ) : ℕ :=
  2^3 * binomial 6 3 * binomial 3 2 - 2^2 * binomial 6 4 * binomial 4 2

theorem coefficient_expansion :
  ∀ (a b c : ℕ → ℕ → ℕ → ℕ),
  (∀ x y z, a x y z = x - y) →
  (∀ x y z, b x y z = x + 2*y + z) →
  (∀ x y z, c x y z = a x y z * (b x y z)^6) →
  coefficient_x2y3z2 a b c = 120 :=
sorry

end NUMINAMATH_CALUDE_coefficient_expansion_l1731_173110


namespace NUMINAMATH_CALUDE_german_team_goals_l1731_173151

def journalist1_statement (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2_statement (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3_statement (x : ℕ) : Prop := x % 2 = 1

def exactly_two_correct (x : ℕ) : Prop :=
  (journalist1_statement x ∧ journalist2_statement x ∧ ¬journalist3_statement x) ∨
  (journalist1_statement x ∧ ¬journalist2_statement x ∧ journalist3_statement x) ∨
  (¬journalist1_statement x ∧ journalist2_statement x ∧ journalist3_statement x)

theorem german_team_goals :
  {x : ℕ | exactly_two_correct x} = {11, 12, 14, 16, 17} := by sorry

end NUMINAMATH_CALUDE_german_team_goals_l1731_173151


namespace NUMINAMATH_CALUDE_R_calculation_l1731_173192

/-- R_k is the integer composed of k repeating digits of 1 in decimal form -/
def R (k : ℕ) : ℕ := (10^k - 1) / 9

/-- The result of the calculation R_36/R_6 - R_3 -/
def result : ℕ := 100000100000100000100000100000099989

/-- Theorem stating that R_36/R_6 - R_3 equals the specified result -/
theorem R_calculation : R 36 / R 6 - R 3 = result := by
  sorry

end NUMINAMATH_CALUDE_R_calculation_l1731_173192


namespace NUMINAMATH_CALUDE_renatas_final_amount_is_77_l1731_173165

/-- Calculates Renata's final amount after a series of transactions --/
def renatas_final_amount (initial_amount charity_donation charity_prize 
  slot_loss1 slot_loss2 slot_loss3 sunglasses_price sunglasses_discount
  water_price lottery_ticket_price lottery_prize sandwich_price 
  sandwich_discount latte_price : ℚ) : ℚ :=
  let after_charity := initial_amount - charity_donation + charity_prize
  let after_slots := after_charity - slot_loss1 - slot_loss2 - slot_loss3
  let sunglasses_cost := sunglasses_price * (1 - sunglasses_discount)
  let after_sunglasses := after_slots - sunglasses_cost
  let after_water_lottery := after_sunglasses - water_price - lottery_ticket_price + lottery_prize
  let meal_cost := (sandwich_price * (1 - sandwich_discount) + latte_price) / 2
  after_water_lottery - meal_cost

/-- Theorem stating that Renata's final amount is $77 --/
theorem renatas_final_amount_is_77 :
  renatas_final_amount 10 4 90 50 10 5 15 0.2 1 1 65 8 0.25 4 = 77 := by
  sorry

end NUMINAMATH_CALUDE_renatas_final_amount_is_77_l1731_173165


namespace NUMINAMATH_CALUDE_toy_factory_daily_production_l1731_173148

/-- A factory produces toys with the following conditions:
    - The factory produces 5500 toys per week
    - Workers work 5 days a week
    - The same number of toys is made every day -/
def ToyFactory (weekly_production : ℕ) (work_days : ℕ) (daily_production : ℕ) : Prop :=
  weekly_production = 5500 ∧ work_days = 5 ∧ daily_production * work_days = weekly_production

/-- Theorem: Given the conditions of the toy factory, the daily production is 1100 toys -/
theorem toy_factory_daily_production :
  ∀ (weekly_production work_days daily_production : ℕ),
  ToyFactory weekly_production work_days daily_production →
  daily_production = 1100 := by
  sorry

end NUMINAMATH_CALUDE_toy_factory_daily_production_l1731_173148


namespace NUMINAMATH_CALUDE_min_value_expression_l1731_173105

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) + 
  (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) ≥ 2 ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → 
    (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) + 
    (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1731_173105


namespace NUMINAMATH_CALUDE_ice_cream_cost_is_3_l1731_173163

/-- The cost of items in dollars -/
structure Costs where
  coffee : ℕ
  cake : ℕ
  total : ℕ

/-- The number of items ordered -/
structure Order where
  coffee : ℕ
  cake : ℕ
  icecream : ℕ

def ice_cream_cost (c : Costs) (mell_order : Order) (friend_order : Order) : ℕ :=
  (c.total - (mell_order.coffee * c.coffee + mell_order.cake * c.cake +
    2 * (friend_order.coffee * c.coffee + friend_order.cake * c.cake))) / (2 * friend_order.icecream)

theorem ice_cream_cost_is_3 (c : Costs) (mell_order : Order) (friend_order : Order) :
  c.coffee = 4 →
  c.cake = 7 →
  c.total = 51 →
  mell_order = { coffee := 2, cake := 1, icecream := 0 } →
  friend_order = { coffee := 2, cake := 1, icecream := 1 } →
  ice_cream_cost c mell_order friend_order = 3 := by
  sorry

#check ice_cream_cost_is_3

end NUMINAMATH_CALUDE_ice_cream_cost_is_3_l1731_173163


namespace NUMINAMATH_CALUDE_line_through_point_l1731_173104

/-- The value of k for which the line -3/4 - 3kx = 7y passes through (1/3, -8) -/
theorem line_through_point (k : ℝ) : 
  (-3/4 : ℝ) - 3 * k * (1/3) = 7 * (-8) → k = 55.25 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l1731_173104


namespace NUMINAMATH_CALUDE_rent_to_expenses_ratio_l1731_173145

/-- Given Kathryn's monthly finances, prove the ratio of rent to food and travel expenses -/
theorem rent_to_expenses_ratio 
  (rent : ℕ) 
  (salary : ℕ) 
  (remaining : ℕ) 
  (h1 : rent = 1200)
  (h2 : salary = 5000)
  (h3 : remaining = 2000) :
  (rent : ℚ) / ((salary - remaining) - rent) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rent_to_expenses_ratio_l1731_173145


namespace NUMINAMATH_CALUDE_writer_productivity_l1731_173157

/-- Calculates the average words per hour for a writer given total words, total hours, and break hours. -/
def averageWordsPerHour (totalWords : ℕ) (totalHours : ℕ) (breakHours : ℕ) : ℚ :=
  totalWords / (totalHours - breakHours)

/-- Theorem stating that for a writer completing 60,000 words in 100 hours with 20 hours of breaks,
    the average words per hour when actually working is 750. -/
theorem writer_productivity : averageWordsPerHour 60000 100 20 = 750 := by
  sorry

end NUMINAMATH_CALUDE_writer_productivity_l1731_173157


namespace NUMINAMATH_CALUDE_election_votes_calculation_l1731_173170

theorem election_votes_calculation (total_votes : ℕ) : 
  let valid_votes := (85 : ℚ) / 100 * total_votes
  let candidate_a_votes := (70 : ℚ) / 100 * valid_votes
  candidate_a_votes = 333200 →
  total_votes = 560000 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l1731_173170


namespace NUMINAMATH_CALUDE_driving_distance_proof_l1731_173149

/-- Proves that under the given driving conditions, the distance must be 60 miles -/
theorem driving_distance_proof (D x : ℝ) (h1 : D > 0) (h2 : x > 0) : 
  (32 / (2 * x) + (D - 32) / (x / 2) = D / x * 1.2) → D = 60 := by
  sorry

end NUMINAMATH_CALUDE_driving_distance_proof_l1731_173149


namespace NUMINAMATH_CALUDE_find_x_l1731_173169

theorem find_x : ∃ x : ℝ, 
  (3 + 7 + 10 + 15) / 4 = 2 * ((x + 20 + 6) / 3) ∧ 
  x = -12.875 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1731_173169


namespace NUMINAMATH_CALUDE_polygon_with_equal_angle_sums_l1731_173125

theorem polygon_with_equal_angle_sums (n : ℕ) : 
  (n - 2) * 180 = 360 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_equal_angle_sums_l1731_173125


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l1731_173189

theorem units_digit_of_seven_to_six_to_five (n : ℕ) : n = 7^(6^5) → n % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l1731_173189


namespace NUMINAMATH_CALUDE_equation_solution_l1731_173198

theorem equation_solution : ∃! x : ℚ, (2 * x) / (x + 3) + 1 = 7 / (2 * x + 6) :=
  by
    use 1/6
    sorry

end NUMINAMATH_CALUDE_equation_solution_l1731_173198


namespace NUMINAMATH_CALUDE_razorback_tshirt_shop_profit_l1731_173129

/-- Calculate the net profit for the Razorback T-shirt Shop on game day -/
theorem razorback_tshirt_shop_profit :
  let regular_price : ℚ := 15
  let production_cost : ℚ := 4
  let first_event_quantity : ℕ := 150
  let second_event_quantity : ℕ := 175
  let first_event_discount : ℚ := 0.1
  let second_event_discount : ℚ := 0.15
  let overhead_expense : ℚ := 200
  let sales_tax_rate : ℚ := 0.05

  let first_event_revenue := (regular_price * (1 - first_event_discount)) * first_event_quantity
  let second_event_revenue := (regular_price * (1 - second_event_discount)) * second_event_quantity
  let total_revenue := first_event_revenue + second_event_revenue
  let total_quantity := first_event_quantity + second_event_quantity
  let total_production_cost := production_cost * total_quantity
  let sales_tax := sales_tax_rate * total_revenue
  let total_expenses := total_production_cost + overhead_expense + sales_tax
  let net_profit := total_revenue - total_expenses

  net_profit = 2543.4375 := by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_shop_profit_l1731_173129


namespace NUMINAMATH_CALUDE_parabola_symmetric_points_m_value_l1731_173140

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  h_pos : a > 0

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y = p.a * x^2

/-- Theorem: Value of m for symmetric points on parabola -/
theorem parabola_symmetric_points_m_value
  (p : Parabola)
  (h_focus_directrix : (1 : ℝ) / (4 * p.a) = 1/4)
  (A B : ParabolaPoint p)
  (h_symmetric : ∃ m : ℝ, (A.y + B.y) / 2 = ((A.x + B.x) / 2) + m ∧
                           (B.y - A.y) = (B.x - A.x))
  (h_product : A.x * B.x = -1/2) :
  ∃ m : ℝ, (A.y + B.y) / 2 = ((A.x + B.x) / 2) + m ∧ m = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_symmetric_points_m_value_l1731_173140


namespace NUMINAMATH_CALUDE_fault_line_movement_l1731_173122

theorem fault_line_movement (total_movement : ℝ) (past_year_movement : ℝ) 
  (h1 : total_movement = 6.5)
  (h2 : past_year_movement = 1.25) :
  total_movement - past_year_movement = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_movement_l1731_173122


namespace NUMINAMATH_CALUDE_season_games_l1731_173107

/-- Represents the basketball season statistics --/
structure SeasonStats where
  total_points : ℕ
  avg_free_throws : ℕ
  avg_two_pointers : ℕ
  avg_three_pointers : ℕ

/-- Calculates the number of games in the season --/
def calculate_games (stats : SeasonStats) : ℕ :=
  stats.total_points / (stats.avg_free_throws + 2 * stats.avg_two_pointers + 3 * stats.avg_three_pointers)

/-- Theorem stating that the number of games in the season is 15 --/
theorem season_games (stats : SeasonStats) 
  (h1 : stats.total_points = 345)
  (h2 : stats.avg_free_throws = 4)
  (h3 : stats.avg_two_pointers = 5)
  (h4 : stats.avg_three_pointers = 3) :
  calculate_games stats = 15 := by
  sorry

end NUMINAMATH_CALUDE_season_games_l1731_173107


namespace NUMINAMATH_CALUDE_contest_paths_count_l1731_173133

/-- Represents a grid where the word "CONTEST" can be spelled out -/
structure ContestGrid where
  word : String
  start_letter : Char
  end_letter : Char

/-- Calculates the number of valid paths to spell out the word on the grid -/
def count_paths (grid : ContestGrid) : ℕ :=
  2^(grid.word.length - 1) - 1

/-- Theorem stating that the number of valid paths to spell "CONTEST" is 127 -/
theorem contest_paths_count :
  ∀ (grid : ContestGrid),
    grid.word = "CONTEST" →
    grid.start_letter = 'C' →
    grid.end_letter = 'T' →
    count_paths grid = 127 := by
  sorry


end NUMINAMATH_CALUDE_contest_paths_count_l1731_173133


namespace NUMINAMATH_CALUDE_local_tax_deduction_l1731_173121

/-- Alicia's hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- Local tax rate as a decimal -/
def tax_rate : ℝ := 0.024

/-- Conversion rate from dollars to cents -/
def dollars_to_cents : ℝ := 100

theorem local_tax_deduction :
  hourly_wage * tax_rate * dollars_to_cents = 60 := by
  sorry

end NUMINAMATH_CALUDE_local_tax_deduction_l1731_173121


namespace NUMINAMATH_CALUDE_min_value_theorem_l1731_173156

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 3 * m + n = 1) :
  (1 / m + 2 / n) ≥ 5 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1731_173156


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1731_173173

theorem polynomial_coefficient_sum (a₄ a₃ a₂ a₁ a : ℝ) :
  (∀ x : ℝ, a₄ * (x + 1)^4 + a₃ * (x + 1)^3 + a₂ * (x + 1)^2 + a₁ * (x + 1) + a = x^4) →
  a₃ - a₂ + a₁ = -14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1731_173173


namespace NUMINAMATH_CALUDE_even_function_inequality_l1731_173183

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → x₂ ≤ 0 → f x₁ < f x₂

theorem even_function_inequality (f : ℝ → ℝ) (n : ℕ) 
  (h_even : is_even_function f)
  (h_incr : increasing_on_nonpositive f) :
  f (n + 1) < f (-n) ∧ f (-n) < f (n - 1) :=
sorry

end NUMINAMATH_CALUDE_even_function_inequality_l1731_173183


namespace NUMINAMATH_CALUDE_novel_essay_arrangement_l1731_173195

/-- The number of ways to arrange 2 novels (which must be placed next to each other) and 3 essays on a bookshelf -/
def arrangement_count : ℕ := 48

/-- The number of novels -/
def novel_count : ℕ := 2

/-- The number of essays -/
def essay_count : ℕ := 3

/-- The total number of items to arrange (treating the novels as a single unit) -/
def total_units : ℕ := essay_count + 1

theorem novel_essay_arrangement :
  arrangement_count = (Nat.factorial total_units) * (Nat.factorial novel_count) :=
sorry

end NUMINAMATH_CALUDE_novel_essay_arrangement_l1731_173195


namespace NUMINAMATH_CALUDE_alpha_plus_beta_value_l1731_173135

theorem alpha_plus_beta_value (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sqrt 3 * (Real.cos (α/2))^2 + Real.sqrt 2 * (Real.sin (β/2))^2 = Real.sqrt 2 / 2 + Real.sqrt 3 / 2)
  (h4 : Real.sin (2017 * π - α) = Real.sqrt 2 * Real.cos (5 * π / 2 - β)) :
  α + β = 5 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_value_l1731_173135


namespace NUMINAMATH_CALUDE_min_A_over_B_l1731_173124

theorem min_A_over_B (A B x : ℝ) (hA : A > 0) (hB : B > 0) (hx : x > 0)
  (h1 : x^2 + 1/x^2 = A) (h2 : x - 1/x = B) :
  A / B ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_A_over_B_l1731_173124


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1731_173161

theorem sphere_surface_area_ratio : 
  let r₁ : ℝ := 6
  let r₂ : ℝ := 3
  let surface_area (r : ℝ) := 4 * Real.pi * r^2
  (surface_area r₁) / (surface_area r₂) = 4 := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1731_173161


namespace NUMINAMATH_CALUDE_triangle_point_collinearity_l1731_173166

/-- Given a triangle ABC and a point P on the same plane, 
    if BC + BA = 2BP, then P, A, and C are collinear -/
theorem triangle_point_collinearity 
  (A B C P : EuclideanSpace ℝ (Fin 2)) 
  (h : (C - B) + (A - B) = 2 • (P - B)) : 
  Collinear ℝ ({P, A, C} : Set (EuclideanSpace ℝ (Fin 2))) :=
sorry

end NUMINAMATH_CALUDE_triangle_point_collinearity_l1731_173166


namespace NUMINAMATH_CALUDE_building_units_l1731_173106

theorem building_units (total : ℕ) (restaurants : ℕ) : 
  (2 * restaurants = total / 4) →
  (restaurants = 75) →
  (total = 300) := by
sorry

end NUMINAMATH_CALUDE_building_units_l1731_173106


namespace NUMINAMATH_CALUDE_percent_of_percent_l1731_173199

theorem percent_of_percent (x : ℝ) : (30 / 100) * (70 / 100) * x = (21 / 100) * x := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l1731_173199


namespace NUMINAMATH_CALUDE_range_of_a_when_quadratic_nonnegative_l1731_173109

theorem range_of_a_when_quadratic_nonnegative (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + a ≥ 0) → a ∈ Set.Icc 0 4 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_when_quadratic_nonnegative_l1731_173109


namespace NUMINAMATH_CALUDE_new_acute_angle_l1731_173150

theorem new_acute_angle (initial_angle : ℝ) (net_rotation : ℝ) : 
  initial_angle = 60 → net_rotation = 90 → 
  (180 - (initial_angle + net_rotation)) % 180 = 30 := by
sorry

end NUMINAMATH_CALUDE_new_acute_angle_l1731_173150


namespace NUMINAMATH_CALUDE_bacteriophage_and_transformation_principle_correct_biological_experiment_description_l1731_173185

/-- Represents a biological experiment --/
structure BiologicalExperiment where
  name : String
  description : String

/-- Represents the principle behind an experiment --/
inductive ExperimentPrinciple
  | GeneticContinuity
  | Other

/-- Function to determine the principle of an experiment --/
def experimentPrinciple (exp : BiologicalExperiment) : ExperimentPrinciple :=
  if exp.name = "Bacteriophage Infection" || exp.name = "Bacterial Transformation" then
    ExperimentPrinciple.GeneticContinuity
  else
    ExperimentPrinciple.Other

/-- Theorem stating that bacteriophage infection and bacterial transformation 
    experiments are based on the same principle of genetic continuity --/
theorem bacteriophage_and_transformation_principle :
  ∀ (exp1 exp2 : BiologicalExperiment),
    exp1.name = "Bacteriophage Infection" →
    exp2.name = "Bacterial Transformation" →
    experimentPrinciple exp1 = experimentPrinciple exp2 :=
by
  sorry

/-- Main theorem proving the correctness of the statement --/
theorem correct_biological_experiment_description :
  ∃ (exp1 exp2 : BiologicalExperiment),
    exp1.name = "Bacteriophage Infection" ∧
    exp2.name = "Bacterial Transformation" ∧
    experimentPrinciple exp1 = ExperimentPrinciple.GeneticContinuity ∧
    experimentPrinciple exp2 = ExperimentPrinciple.GeneticContinuity :=
by
  sorry

end NUMINAMATH_CALUDE_bacteriophage_and_transformation_principle_correct_biological_experiment_description_l1731_173185


namespace NUMINAMATH_CALUDE_simplify_expression_find_value_evaluate_compound_l1731_173177

-- Part 1
theorem simplify_expression (a b : ℝ) :
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 := by sorry

-- Part 2
theorem find_value (x y : ℝ) (h : x^2 - 2*y = 4) :
  3 * x^2 - 6 * y - 21 = -9 := by sorry

-- Part 3
theorem evaluate_compound (a b c d : ℝ) 
  (h1 : a - 2*b = 6) (h2 : 2*b - c = -8) (h3 : c - d = 9) :
  (a - c) + (2*b - d) - (2*b - c) = 7 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_find_value_evaluate_compound_l1731_173177


namespace NUMINAMATH_CALUDE_range_of_a_l1731_173182

def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem range_of_a (a : ℝ) (h : A a ∪ B a = Set.univ) : a ∈ Set.Iic 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1731_173182


namespace NUMINAMATH_CALUDE_translated_line_through_point_l1731_173178

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically -/
def translate_line (l : Line) (units : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + units }

/-- Check if a point lies on a line -/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem translated_line_through_point (m : ℝ) :
  let original_line : Line := { slope := 1, intercept := 0 }
  let translated_line := translate_line original_line 3
  point_on_line translated_line 2 m → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_translated_line_through_point_l1731_173178


namespace NUMINAMATH_CALUDE_modified_square_perimeter_l1731_173159

/-- The perimeter of a modified square with an isosceles right triangle repositioned -/
theorem modified_square_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 64) :
  let side_length := square_perimeter / 4
  let hypotenuse := Real.sqrt (2 * side_length ^ 2)
  square_perimeter + hypotenuse = 80 + 16 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_modified_square_perimeter_l1731_173159


namespace NUMINAMATH_CALUDE_tangerines_left_l1731_173112

theorem tangerines_left (total : ℕ) (given : ℕ) (h1 : total = 27) (h2 : given = 18) :
  total - given = 9 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_left_l1731_173112


namespace NUMINAMATH_CALUDE_range_of_x_range_of_a_l1731_173171

-- Define the conditions
def p (x : ℝ) : Prop := (x + 1) / (x - 2) > 2
def q (x a : ℝ) : Prop := x^2 - a*x + 5 > 0

-- Theorem 1: If p is true, then x is in the open interval (2,5)
theorem range_of_x (x : ℝ) : p x → x ∈ Set.Ioo 2 5 := by sorry

-- Theorem 2: If p is a sufficient but not necessary condition for q,
-- then a is in the interval (-∞, 2√5)
theorem range_of_a (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x) →
  a ∈ Set.Iio (2 * Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_a_l1731_173171


namespace NUMINAMATH_CALUDE_green_light_probability_l1731_173197

def traffic_light_cycle : ℕ := 60
def red_light_duration : ℕ := 30
def green_light_duration : ℕ := 25
def yellow_light_duration : ℕ := 5

theorem green_light_probability :
  (green_light_duration : ℚ) / traffic_light_cycle = 5 / 12 := by
sorry

end NUMINAMATH_CALUDE_green_light_probability_l1731_173197


namespace NUMINAMATH_CALUDE_irrational_zero_one_sequence_exists_l1731_173139

/-- A sequence representing the decimal digits of a number -/
def DecimalSequence := ℕ → Fin 10

/-- Checks if a decimal sequence contains only 0 and 1 -/
def OnlyZeroOne (s : DecimalSequence) : Prop :=
  ∀ n, s n = 0 ∨ s n = 1

/-- Checks if a decimal sequence has no two adjacent 1s -/
def NoAdjacentOnes (s : DecimalSequence) : Prop :=
  ∀ n, ¬(s n = 1 ∧ s (n + 1) = 1)

/-- Checks if a decimal sequence has no more than two adjacent 0s -/
def NoMoreThanTwoZeros (s : DecimalSequence) : Prop :=
  ∀ n, ¬(s n = 0 ∧ s (n + 1) = 0 ∧ s (n + 2) = 0)

/-- Checks if a decimal sequence represents an irrational number -/
def IsIrrational (s : DecimalSequence) : Prop :=
  ∀ k p, ∃ n ≥ k, s n ≠ s (n + p)

/-- There exists an irrational number whose decimal representation
    contains only 0 and 1, with no two adjacent 1s and no more than two adjacent 0s -/
theorem irrational_zero_one_sequence_exists : 
  ∃ s : DecimalSequence, 
    OnlyZeroOne s ∧ 
    NoAdjacentOnes s ∧ 
    NoMoreThanTwoZeros s ∧ 
    IsIrrational s := by
  sorry

end NUMINAMATH_CALUDE_irrational_zero_one_sequence_exists_l1731_173139


namespace NUMINAMATH_CALUDE_time_to_drain_pool_l1731_173103

/-- The time it takes to drain a rectangular pool given its dimensions, capacity, and drainage rate. -/
theorem time_to_drain_pool 
  (length width depth : ℝ) 
  (capacity : ℝ) 
  (drainage_rate : ℝ) 
  (h1 : length = 150)
  (h2 : width = 50)
  (h3 : depth = 10)
  (h4 : capacity = 0.8)
  (h5 : drainage_rate = 60) :
  (length * width * depth * capacity) / drainage_rate = 1000 := by
  sorry


end NUMINAMATH_CALUDE_time_to_drain_pool_l1731_173103


namespace NUMINAMATH_CALUDE_inverse_negation_correct_l1731_173162

/-- The original statement -/
def original_statement (x : ℝ) : Prop := x ≥ 3 → x < 0

/-- The inverse negation of the original statement -/
def inverse_negation (x : ℝ) : Prop := x ≥ 0 → x < 3

/-- Theorem stating that the inverse_negation is correct -/
theorem inverse_negation_correct : 
  (∀ x, original_statement x) ↔ (∀ x, inverse_negation x) :=
sorry

end NUMINAMATH_CALUDE_inverse_negation_correct_l1731_173162


namespace NUMINAMATH_CALUDE_two_week_jogging_time_l1731_173117

/-- The total jogging time in hours after a given number of days, 
    given a fixed daily jogging time in hours -/
def total_jogging_time (daily_time : ℝ) (days : ℕ) : ℝ :=
  daily_time * days

/-- Theorem stating that jogging 1.5 hours daily for 14 days results in 21 hours total -/
theorem two_week_jogging_time :
  total_jogging_time 1.5 14 = 21 := by
  sorry

end NUMINAMATH_CALUDE_two_week_jogging_time_l1731_173117


namespace NUMINAMATH_CALUDE_exists_cell_with_same_color_in_all_directions_l1731_173180

/-- A color type with four possible values -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- A type representing a 50x50 grid colored with four colors -/
def ColoredGrid := Fin 50 → Fin 50 → Color

/-- A function to check if a cell has the same color in all four directions -/
def hasSameColorInAllDirections (grid : ColoredGrid) (row col : Fin 50) : Prop :=
  ∃ (r1 r2 : Fin 50) (c1 c2 : Fin 50),
    r1 < row ∧ row < r2 ∧ c1 < col ∧ col < c2 ∧
    grid row col = grid r1 col ∧
    grid row col = grid r2 col ∧
    grid row col = grid row c1 ∧
    grid row col = grid row c2

/-- Theorem stating that there exists a cell with the same color in all four directions -/
theorem exists_cell_with_same_color_in_all_directions (grid : ColoredGrid) :
  ∃ (row col : Fin 50), hasSameColorInAllDirections grid row col := by
  sorry


end NUMINAMATH_CALUDE_exists_cell_with_same_color_in_all_directions_l1731_173180


namespace NUMINAMATH_CALUDE_f_1_eq_0_f_decreasing_f_abs_lt_neg_2_iff_l1731_173187

noncomputable section

variable (f : ℝ → ℝ)

axiom f_domain : ∀ x, x > 0 → f x ≠ 0

axiom f_functional_equation : ∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y

axiom f_negative_when_gt_one : ∀ x, x > 1 → f x < 0

axiom f_3_eq_neg_1 : f 3 = -1

theorem f_1_eq_0 : f 1 = 0 := by sorry

theorem f_decreasing : ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ > x₂ → f x₁ < f x₂ := by sorry

theorem f_abs_lt_neg_2_iff : ∀ x, f (|x|) < -2 ↔ x < -9 ∨ x > 9 := by sorry

end

end NUMINAMATH_CALUDE_f_1_eq_0_f_decreasing_f_abs_lt_neg_2_iff_l1731_173187


namespace NUMINAMATH_CALUDE_salary_problem_l1731_173146

-- Define the salaries and total
variable (A B : ℝ)
def total : ℝ := 6000

-- Define the spending percentages
def A_spend_percent : ℝ := 0.95
def B_spend_percent : ℝ := 0.85

-- Define the theorem
theorem salary_problem :
  A + B = total ∧ 
  (1 - A_spend_percent) * A = (1 - B_spend_percent) * B →
  A = 4500 := by
sorry

end NUMINAMATH_CALUDE_salary_problem_l1731_173146


namespace NUMINAMATH_CALUDE_toy_store_revenue_l1731_173102

theorem toy_store_revenue (december : ℝ) (november : ℝ) (january : ℝ) 
  (h1 : november = (3/5) * december) 
  (h2 : january = (1/6) * november) : 
  december = (20/7) * ((november + january) / 2) := by
sorry

end NUMINAMATH_CALUDE_toy_store_revenue_l1731_173102


namespace NUMINAMATH_CALUDE_smallest_transactions_to_exceed_fee_l1731_173131

/-- Represents the types of transactions --/
inductive TransactionType
| Autodebit
| Cheque
| CashWithdrawal

/-- Represents the cost of each transaction type --/
def transactionCost : TransactionType → ℚ
| TransactionType.Autodebit => 0.60
| TransactionType.Cheque => 0.50
| TransactionType.CashWithdrawal => 0.45

/-- Calculates the total cost for the first 25 transactions --/
def firstTwentyFiveCost : ℚ := 15 * transactionCost TransactionType.Autodebit +
                                5 * transactionCost TransactionType.Cheque +
                                5 * transactionCost TransactionType.CashWithdrawal

/-- Theorem stating that 29 is the smallest number of transactions to exceed $15.95 --/
theorem smallest_transactions_to_exceed_fee :
  ∀ n : ℕ, n ≥ 29 ↔ 
    firstTwentyFiveCost + (n - 25 : ℕ) * transactionCost TransactionType.Autodebit > 15.95 :=
by sorry

end NUMINAMATH_CALUDE_smallest_transactions_to_exceed_fee_l1731_173131


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l1731_173196

theorem quadratic_distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 6 * x + 9 = 0 ∧ k * y^2 - 6 * y + 9 = 0) ↔ 
  (k < 1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l1731_173196


namespace NUMINAMATH_CALUDE_missing_shirts_count_l1731_173175

theorem missing_shirts_count (trousers : ℕ) (total_bill : ℕ) (shirt_cost : ℕ) (trouser_cost : ℕ) (claimed_shirts : ℕ) : 
  trousers = 10 →
  total_bill = 140 →
  shirt_cost = 5 →
  trouser_cost = 9 →
  claimed_shirts = 2 →
  (total_bill - trousers * trouser_cost) / shirt_cost - claimed_shirts = 8 := by
sorry

end NUMINAMATH_CALUDE_missing_shirts_count_l1731_173175


namespace NUMINAMATH_CALUDE_matrix_equation_l1731_173191

theorem matrix_equation (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = !![5, 2; -3, 9]) : 
  B * A = !![5, 2; -3, 9] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_l1731_173191


namespace NUMINAMATH_CALUDE_vertical_asymptotes_sum_l1731_173193

theorem vertical_asymptotes_sum (a b c : ℝ) (h : a ≠ 0) :
  let p := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let q := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 4 ∧ b = 6 ∧ c = 3 → p + q = -1.75 := by
  sorry

#check vertical_asymptotes_sum

end NUMINAMATH_CALUDE_vertical_asymptotes_sum_l1731_173193


namespace NUMINAMATH_CALUDE_triangle_inequality_for_given_sides_l1731_173188

theorem triangle_inequality_for_given_sides (x : ℝ) (h : x > 1) :
  let a := x^4 + x^3 + 2*x^2 + x + 1
  let b := 2*x^3 + x^2 + 2*x + 1
  let c := x^4 - 1
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_for_given_sides_l1731_173188


namespace NUMINAMATH_CALUDE_slate_rock_probability_l1731_173137

/-- The probability of choosing two slate rocks without replacement from a field of rocks. -/
theorem slate_rock_probability (slate_rocks pumice_rocks granite_rocks : ℕ) 
  (h_slate : slate_rocks = 10)
  (h_pumice : pumice_rocks = 11)
  (h_granite : granite_rocks = 4) :
  let total_rocks := slate_rocks + pumice_rocks + granite_rocks
  (slate_rocks : ℚ) / total_rocks * ((slate_rocks - 1) : ℚ) / (total_rocks - 1) = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_slate_rock_probability_l1731_173137


namespace NUMINAMATH_CALUDE_shopkeeper_net_loss_percent_l1731_173134

/-- Calculates the net profit or loss percentage for a shopkeeper's transactions -/
theorem shopkeeper_net_loss_percent : 
  let cost_price : ℝ := 1000
  let num_articles : ℕ := 4
  let profit_percent1 : ℝ := 10
  let loss_percent2 : ℝ := 10
  let profit_percent3 : ℝ := 20
  let loss_percent4 : ℝ := 25
  
  let selling_price1 : ℝ := cost_price * (1 + profit_percent1 / 100)
  let selling_price2 : ℝ := cost_price * (1 - loss_percent2 / 100)
  let selling_price3 : ℝ := cost_price * (1 + profit_percent3 / 100)
  let selling_price4 : ℝ := cost_price * (1 - loss_percent4 / 100)
  
  let total_cost : ℝ := cost_price * num_articles
  let total_selling : ℝ := selling_price1 + selling_price2 + selling_price3 + selling_price4
  
  let net_loss : ℝ := total_cost - total_selling
  let net_loss_percent : ℝ := (net_loss / total_cost) * 100
  
  net_loss_percent = 1.25 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_net_loss_percent_l1731_173134


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l1731_173111

theorem min_distance_between_curves (a : ℝ) (h : a > 0) : 
  ∃ (min_val : ℝ), min_val = 12 ∧ ∀ x > 0, |16 / x + x^2| ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l1731_173111


namespace NUMINAMATH_CALUDE_ab_greater_than_a_plus_b_l1731_173113

theorem ab_greater_than_a_plus_b (a b : ℝ) (ha : a > 2) (hb : b > 2) :
  a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_than_a_plus_b_l1731_173113


namespace NUMINAMATH_CALUDE_cookie_cost_l1731_173138

/-- The cost of cookies is equal to the sum of money Diane has and the additional money she needs. -/
theorem cookie_cost (money_has : ℕ) (money_needs : ℕ) (cost : ℕ) : 
  money_has = 27 → money_needs = 38 → cost = money_has + money_needs := by
  sorry

end NUMINAMATH_CALUDE_cookie_cost_l1731_173138


namespace NUMINAMATH_CALUDE_smallest_positive_real_l1731_173155

theorem smallest_positive_real : ∃ (x : ℝ), x > 0 ∧ x + 1 > 1 * x ∧ ∀ (y : ℝ), y > 0 ∧ y + 1 > 1 * y → x ≤ y :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_real_l1731_173155


namespace NUMINAMATH_CALUDE_markup_percentage_is_20_l1731_173100

/-- Calculate the markup percentage given cost price, discount, and profit percentage --/
def calculate_markup_percentage (cost_price discount : ℕ) (profit_percentage : ℚ) : ℚ :=
  let selling_price := cost_price + (cost_price * profit_percentage / 100)
  let marked_price := selling_price + discount
  (marked_price - cost_price) / cost_price * 100

/-- Theorem stating that the markup percentage is 20% given the specified conditions --/
theorem markup_percentage_is_20 :
  calculate_markup_percentage 180 50 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_markup_percentage_is_20_l1731_173100


namespace NUMINAMATH_CALUDE_number_of_cats_l1731_173115

/-- The number of cats on a farm, given the number of dogs, fish, and total pets -/
theorem number_of_cats (dogs : ℕ) (fish : ℕ) (total_pets : ℕ) (h1 : dogs = 43) (h2 : fish = 72) (h3 : total_pets = 149) :
  total_pets - dogs - fish = 34 := by
sorry

end NUMINAMATH_CALUDE_number_of_cats_l1731_173115


namespace NUMINAMATH_CALUDE_count_master_sudokus_master_sudoku_count_l1731_173144

/-- The number of Master Sudokus for a given n -/
def masterSudokuCount (n : ℕ) : ℕ :=
  2^(n-1)

/-- Theorem: The number of Master Sudokus for n is 2^(n-1) -/
theorem count_master_sudokus (n : ℕ) :
  (∀ k : ℕ, k < n → masterSudokuCount k = 2^(k-1)) →
  masterSudokuCount n = 2^(n-1) := by
  sorry

/-- The main theorem stating the number of Master Sudokus -/
theorem master_sudoku_count (n : ℕ) :
  masterSudokuCount n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_count_master_sudokus_master_sudoku_count_l1731_173144


namespace NUMINAMATH_CALUDE_circle_area_tripled_l1731_173118

theorem circle_area_tripled (r n : ℝ) : 
  (r > 0) →
  (n > 0) →
  (π * (r + n)^2 = 3 * π * r^2) →
  (r = n * (Real.sqrt 3 + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l1731_173118


namespace NUMINAMATH_CALUDE_find_n_l1731_173127

theorem find_n : ∃ n : ℚ, (1 / (n + 2) + 2 / (n + 2) + 3 * n / (n + 2) = 5) ∧ (n = -7/2) := by
  sorry

end NUMINAMATH_CALUDE_find_n_l1731_173127


namespace NUMINAMATH_CALUDE_non_monotonic_values_l1731_173158

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -2 * x^2 + a * x

-- Define the interval
def interval : Set ℝ := Set.Ioo (-1) 2

-- Define the property of non-monotonicity
def is_non_monotonic (g : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∃ (x y z : ℝ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x < y ∧ y < z ∧
    ((g x < g y ∧ g y > g z) ∨ (g x > g y ∧ g y < g z))

-- State the theorem
theorem non_monotonic_values :
  {a : ℝ | is_non_monotonic (f a) interval} = {-2, 4} := by sorry

end NUMINAMATH_CALUDE_non_monotonic_values_l1731_173158


namespace NUMINAMATH_CALUDE_square_root_of_nine_l1731_173141

theorem square_root_of_nine (x : ℝ) : x ^ 2 = 9 ↔ x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l1731_173141


namespace NUMINAMATH_CALUDE_consecutive_sum_smallest_l1731_173167

theorem consecutive_sum_smallest (a : ℤ) (h : a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = 210) : a = 40 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_smallest_l1731_173167


namespace NUMINAMATH_CALUDE_tangent_line_determines_function_l1731_173132

/-- Given a function f(x) = (mx-6)/(x^2+n), prove that if the tangent line
    at P(-1,f(-1)) is x+2y+5=0, then f(x) = (2x-6)/(x^2+3) -/
theorem tangent_line_determines_function (m n : ℝ) :
  let f : ℝ → ℝ := λ x => (m*x - 6) / (x^2 + n)
  let tangent_line : ℝ → ℝ := λ x => -(1/2)*x - 5/2
  (f (-1) = tangent_line (-1) ∧ 
   (deriv f) (-1) = (deriv tangent_line) (-1)) →
  f = λ x => (2*x - 6) / (x^2 + 3) :=
by
  sorry


end NUMINAMATH_CALUDE_tangent_line_determines_function_l1731_173132


namespace NUMINAMATH_CALUDE_outfit_combinations_l1731_173147

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (ties : ℕ) (jackets : ℕ) :
  shirts = 8 →
  pants = 5 →
  ties = 4 →
  jackets = 3 →
  shirts * pants * (ties + 1) * (jackets + 1) = 800 :=
by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1731_173147


namespace NUMINAMATH_CALUDE_negative_two_plus_three_equals_one_l1731_173123

theorem negative_two_plus_three_equals_one : (-2 : ℤ) + 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_plus_three_equals_one_l1731_173123


namespace NUMINAMATH_CALUDE_line_properties_l1731_173128

-- Define the line equation
def line_equation (a x y : ℝ) : Prop :=
  (a - 1) * x + y - a - 5 = 0

-- Define the fixed point
def fixed_point (A : ℝ × ℝ) : Prop :=
  ∀ a : ℝ, line_equation a A.1 A.2

-- Define the condition for not passing through the second quadrant
def not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation a x y → (x ≤ 0 ∧ y > 0 → False)

-- Theorem statement
theorem line_properties :
  (∃ A : ℝ × ℝ, fixed_point A ∧ A = (1, 6)) ∧
  (∀ a : ℝ, not_in_second_quadrant a ↔ a ≤ -5) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l1731_173128


namespace NUMINAMATH_CALUDE_batsman_average_l1731_173174

/-- Represents a batsman's cricket performance -/
structure Batsman where
  innings : ℕ
  lastScore : ℕ
  averageIncrease : ℕ
  neverNotOut : Prop

/-- Calculates the average score after the latest innings -/
def averageAfterLatestInnings (b : Batsman) : ℕ :=
  sorry

/-- Theorem: Given the conditions, the batsman's average after the 12th innings is 37 runs -/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.lastScore = 70)
  (h3 : b.averageIncrease = 3)
  (h4 : b.neverNotOut) :
  averageAfterLatestInnings b = 37 :=
sorry

end NUMINAMATH_CALUDE_batsman_average_l1731_173174


namespace NUMINAMATH_CALUDE_exactly_five_cheaper_values_l1731_173172

/-- The cost function for books, including the discount -/
def C (n : ℕ) : ℚ :=
  let base := if n ≤ 20 then 15 * n
              else if n ≤ 40 then 14 * n - 5
              else 13 * n
  base - 10 * (n / 10 : ℚ)

/-- Predicate for when it's cheaper to buy n+1 books than n books -/
def cheaper_to_buy_more (n : ℕ) : Prop :=
  C (n + 1) < C n

/-- The main theorem stating there are exactly 5 values where it's cheaper to buy more -/
theorem exactly_five_cheaper_values :
  (∃ (s : Finset ℕ), s.card = 5 ∧ ∀ n, n ∈ s ↔ cheaper_to_buy_more n) :=
sorry

end NUMINAMATH_CALUDE_exactly_five_cheaper_values_l1731_173172


namespace NUMINAMATH_CALUDE_kyle_driving_time_l1731_173152

/-- Given the conditions of Joseph and Kyle's driving, prove that Kyle's driving time is 2 hours. -/
theorem kyle_driving_time :
  let joseph_speed : ℝ := 50
  let joseph_time : ℝ := 2.5
  let kyle_speed : ℝ := 62
  let joseph_distance : ℝ := joseph_speed * joseph_time
  let kyle_distance : ℝ := joseph_distance - 1
  kyle_distance / kyle_speed = 2 := by sorry

end NUMINAMATH_CALUDE_kyle_driving_time_l1731_173152


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_sum_l1731_173119

theorem imaginary_part_of_complex_sum (i : ℂ) (h : i * i = -1) :
  Complex.im ((1 / (i - 2)) + (2 / (1 - 2*i))) = 3/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_sum_l1731_173119


namespace NUMINAMATH_CALUDE_power_equation_l1731_173190

theorem power_equation (m n : ℕ) (h1 : 2^m = 3) (h2 : 2^n = 4) : 2^(3*m - 2*n) = 27/16 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l1731_173190


namespace NUMINAMATH_CALUDE_train_passing_time_l1731_173116

/-- Time for a train to pass a trolley moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (trolley_speed : ℝ) :
  train_length = 110 →
  train_speed = 60 * (1000 / 3600) →
  trolley_speed = 12 * (1000 / 3600) →
  (train_length / (train_speed + trolley_speed)) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1731_173116


namespace NUMINAMATH_CALUDE_ratio_is_two_l1731_173136

/-- An isosceles right triangle with an inscribed square -/
structure IsoscelesRightTriangleWithSquare where
  /-- Length of OP -/
  a : ℝ
  /-- Length of OQ -/
  b : ℝ
  /-- Assumption that a and b are positive -/
  a_pos : a > 0
  b_pos : b > 0
  /-- The area of the square PQRS is 2/5 of the area of triangle AOB -/
  area_ratio : (a^2 + b^2) / ((2*a + b)^2 / 2) = 2/5

/-- The main theorem -/
theorem ratio_is_two (t : IsoscelesRightTriangleWithSquare) : t.a / t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_is_two_l1731_173136


namespace NUMINAMATH_CALUDE_car_speed_problem_l1731_173108

theorem car_speed_problem (highway_length : ℝ) (meeting_time : ℝ) (second_car_speed : ℝ) :
  highway_length = 45 ∧ 
  meeting_time = 1.5 ∧ 
  second_car_speed = 16 →
  ∃ (first_car_speed : ℝ), 
    first_car_speed * meeting_time + second_car_speed * meeting_time = highway_length ∧ 
    first_car_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1731_173108


namespace NUMINAMATH_CALUDE_money_distribution_l1731_173114

/-- Given three people A, B, and C with money, prove that A and C together have 200 units of money -/
theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 →  -- Total money between A, B, and C
  b + c = 360 →      -- Money B and C have together
  c = 60 →           -- Money C has
  a + c = 200 :=     -- Prove A and C have 200 together
by
  sorry


end NUMINAMATH_CALUDE_money_distribution_l1731_173114


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l1731_173143

/-- Represents different income levels of families -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic

/-- Represents a survey with its population and sample size -/
structure Survey where
  population : ℕ
  sampleSize : ℕ
  incomeGroups : Option (ℕ × ℕ × ℕ)

/-- Determines the appropriate sampling method for a given survey -/
def appropriateSamplingMethod (s : Survey) : SamplingMethod :=
  sorry

/-- The first survey from the problem -/
def survey1 : Survey :=
  { population := 430 + 980 + 290
  , sampleSize := 170
  , incomeGroups := some (430, 980, 290) }

/-- The second survey from the problem -/
def survey2 : Survey :=
  { population := 12
  , sampleSize := 5
  , incomeGroups := none }

/-- Theorem stating the appropriate sampling methods for the two surveys -/
theorem appropriate_sampling_methods :
  appropriateSamplingMethod survey1 = SamplingMethod.Stratified ∧
  appropriateSamplingMethod survey2 = SamplingMethod.SimpleRandom :=
sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l1731_173143


namespace NUMINAMATH_CALUDE_hair_cut_length_l1731_173184

def hair_problem (initial_length growth_length final_length : ℕ) : ℕ :=
  initial_length + growth_length - final_length

theorem hair_cut_length :
  hair_problem 14 8 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_length_l1731_173184
