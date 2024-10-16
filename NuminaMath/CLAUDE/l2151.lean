import Mathlib

namespace NUMINAMATH_CALUDE_pencils_purchased_l2151_215161

/-- The number of pens purchased -/
def num_pens : ℕ := 30

/-- The total cost of pens and pencils -/
def total_cost : ℚ := 630

/-- The average price of a pencil -/
def pencil_price : ℚ := 2

/-- The average price of a pen -/
def pen_price : ℚ := 16

/-- The number of pencils purchased -/
def num_pencils : ℕ := 75

theorem pencils_purchased : 
  (num_pens : ℚ) * pen_price + (num_pencils : ℚ) * pencil_price = total_cost := by
  sorry

end NUMINAMATH_CALUDE_pencils_purchased_l2151_215161


namespace NUMINAMATH_CALUDE_grape_difference_l2151_215122

/-- The number of grapes in Rob's bowl -/
def robs_grapes : ℕ := 25

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := 83

/-- The number of grapes in Allie's bowl -/
def allies_grapes : ℕ := (total_grapes - robs_grapes - 4) / 2

/-- The number of grapes in Allyn's bowl -/
def allyns_grapes : ℕ := allies_grapes + 4

theorem grape_difference : allies_grapes - robs_grapes = 2 := by
  sorry

end NUMINAMATH_CALUDE_grape_difference_l2151_215122


namespace NUMINAMATH_CALUDE_system_solution_correct_l2151_215198

theorem system_solution_correct (x₁ x₂ x₃ : ℝ) : 
  x₁ = 4 ∧ x₂ = 3 ∧ x₃ = 5 →
  x₁ + 2*x₂ = 10 ∧
  3*x₁ + 2*x₂ + x₃ = 23 ∧
  x₂ + 2*x₃ = 13 := by
sorry

end NUMINAMATH_CALUDE_system_solution_correct_l2151_215198


namespace NUMINAMATH_CALUDE_power_mod_seven_l2151_215100

theorem power_mod_seven : 76^77 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seven_l2151_215100


namespace NUMINAMATH_CALUDE_ball_arrangement_theorem_l2151_215119

-- Define the number of balls and boxes
def n : ℕ := 4

-- Define the function for the number of arrangements with each box containing one ball
def arrangements_full (n : ℕ) : ℕ := n.factorial

-- Define the function for the number of arrangements with exactly one box empty
def arrangements_one_empty (n : ℕ) : ℕ := n.choose 2 * (n - 1).factorial

-- State the theorem
theorem ball_arrangement_theorem :
  arrangements_full n = 24 ∧ arrangements_one_empty n = 144 := by
  sorry


end NUMINAMATH_CALUDE_ball_arrangement_theorem_l2151_215119


namespace NUMINAMATH_CALUDE_tysons_age_l2151_215126

/-- Given the ages and relationships between Kyle, Julian, Frederick, and Tyson, prove Tyson's age --/
theorem tysons_age (kyle_age julian_age frederick_age tyson_age : ℕ) : 
  kyle_age = 25 →
  kyle_age = julian_age + 5 →
  frederick_age = julian_age + 20 →
  frederick_age = 2 * tyson_age →
  tyson_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_tysons_age_l2151_215126


namespace NUMINAMATH_CALUDE_road_trip_gas_cost_jennas_road_trip_cost_l2151_215174

/-- Calculates the cost of a road trip given driving times, speeds, and gas efficiency --/
theorem road_trip_gas_cost 
  (time1 : ℝ) (speed1 : ℝ) (time2 : ℝ) (speed2 : ℝ) 
  (gas_efficiency : ℝ) (gas_price : ℝ) : ℝ :=
  let distance1 := time1 * speed1
  let distance2 := time2 * speed2
  let total_distance := distance1 + distance2
  let gas_used := total_distance / gas_efficiency
  let total_cost := gas_used * gas_price
  total_cost

/-- Proves that Jenna's road trip gas cost is $18 --/
theorem jennas_road_trip_cost : 
  road_trip_gas_cost 2 60 3 50 30 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_gas_cost_jennas_road_trip_cost_l2151_215174


namespace NUMINAMATH_CALUDE_max_distinct_count_is_five_l2151_215156

/-- A type representing a circular arrangement of nine natural numbers -/
def CircularArrangement := Fin 9 → ℕ

/-- Checks if a number is prime -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The condition that all adjacent triples in the circle form prime sums -/
def AllAdjacentTriplesPrime (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 9, IsPrime (arr i + arr (i + 1) ^ (arr (i + 2)))

/-- The number of distinct elements in the circular arrangement -/
def DistinctCount (arr : CircularArrangement) : ℕ :=
  Finset.card (Finset.image arr Finset.univ)

/-- The main theorem statement -/
theorem max_distinct_count_is_five (arr : CircularArrangement) 
  (h : AllAdjacentTriplesPrime arr) : 
  DistinctCount arr ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_distinct_count_is_five_l2151_215156


namespace NUMINAMATH_CALUDE_percent_problem_l2151_215138

theorem percent_problem (x : ℝ) (h : 120 = 0.75 * x) : x = 160 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l2151_215138


namespace NUMINAMATH_CALUDE_max_value_theorem_l2151_215123

theorem max_value_theorem (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_constraint : x + y + z = 3) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 243/16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2151_215123


namespace NUMINAMATH_CALUDE_ben_spending_correct_l2151_215137

/-- Calculates Ben's spending at the bookstore with given prices and discounts --/
def benSpending (notebookPrice magazinePrice penPrice bookPrice : ℚ)
                (notebookCount magazineCount penCount bookCount : ℕ)
                (penDiscount membershipDiscount membershipThreshold : ℚ) : ℚ :=
  let subtotal := notebookPrice * notebookCount +
                  magazinePrice * magazineCount +
                  penPrice * (1 - penDiscount) * penCount +
                  bookPrice * bookCount
  if subtotal ≥ membershipThreshold then
    subtotal - membershipDiscount
  else
    subtotal

/-- Theorem stating that Ben's spending matches the calculated amount --/
theorem ben_spending_correct :
  benSpending 2 6 1.5 12 4 3 5 2 0.25 10 50 = 45.625 := by sorry

end NUMINAMATH_CALUDE_ben_spending_correct_l2151_215137


namespace NUMINAMATH_CALUDE_sin_x_minus_pi_third_l2151_215154

theorem sin_x_minus_pi_third (x : ℝ) (h : Real.cos (x + π / 6) = 1 / 3) :
  Real.sin (x - π / 3) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_x_minus_pi_third_l2151_215154


namespace NUMINAMATH_CALUDE_unique_function_solution_l2151_215117

theorem unique_function_solution (f : ℕ → ℕ) :
  (∀ a b : ℕ, f (f a + f b) = a + b) ↔ (∀ n : ℕ, f n = n) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_solution_l2151_215117


namespace NUMINAMATH_CALUDE_pinecone_count_l2151_215147

theorem pinecone_count (initial : ℕ) : 
  (initial : ℝ) * 0.2 = initial * 0.2 ∧  -- 20% eaten by reindeer
  (initial : ℝ) * 0.4 = 2 * (initial * 0.2) ∧  -- Twice as many eaten by squirrels
  (initial : ℝ) * 0.25 * 0.4 = initial * 0.1 ∧  -- 25% of remainder collected for fires
  (initial : ℝ) * 0.3 = 600 →  -- 600 pinecones left
  initial = 2000 := by
sorry

end NUMINAMATH_CALUDE_pinecone_count_l2151_215147


namespace NUMINAMATH_CALUDE_weight_loss_per_month_l2151_215179

def initial_weight : ℝ := 250
def final_weight : ℝ := 154
def months_in_year : ℕ := 12

theorem weight_loss_per_month :
  (initial_weight - final_weight) / months_in_year = 8 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_per_month_l2151_215179


namespace NUMINAMATH_CALUDE_tp_supply_duration_l2151_215190

/-- Represents the toilet paper usage of a family member --/
structure TPUsage where
  weekdayTimes : ℕ
  weekdaySquares : ℕ
  weekendTimes : ℕ
  weekendSquares : ℕ

/-- Calculates the total squares used per week for a family member --/
def weeklyUsage (usage : TPUsage) : ℕ :=
  5 * usage.weekdayTimes * usage.weekdaySquares +
  2 * usage.weekendTimes * usage.weekendSquares

/-- Represents the family's toilet paper situation --/
structure TPFamily where
  bill : TPUsage
  wife : TPUsage
  kid : TPUsage
  kidCount : ℕ
  rollCount : ℕ
  squaresPerRoll : ℕ

/-- Calculates the total squares used per week for the entire family --/
def familyWeeklyUsage (family : TPFamily) : ℕ :=
  weeklyUsage family.bill +
  weeklyUsage family.wife +
  family.kidCount * weeklyUsage family.kid

/-- Calculates how many days the toilet paper supply will last --/
def supplyDuration (family : TPFamily) : ℕ :=
  let totalSquares := family.rollCount * family.squaresPerRoll
  let weeksSupply := totalSquares / familyWeeklyUsage family
  7 * weeksSupply

/-- The main theorem stating how long the toilet paper supply will last --/
theorem tp_supply_duration : 
  let family : TPFamily := {
    bill := { weekdayTimes := 3, weekdaySquares := 5, weekendTimes := 4, weekendSquares := 6 },
    wife := { weekdayTimes := 4, weekdaySquares := 8, weekendTimes := 5, weekendSquares := 10 },
    kid := { weekdayTimes := 5, weekdaySquares := 6, weekendTimes := 6, weekendSquares := 5 },
    kidCount := 2,
    rollCount := 1000,
    squaresPerRoll := 300
  }
  ∃ (d : ℕ), d ≥ 2615 ∧ d ≤ 2616 ∧ supplyDuration family = d :=
by sorry


end NUMINAMATH_CALUDE_tp_supply_duration_l2151_215190


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2151_215135

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 5 + 1
  let y : ℝ := Real.sqrt 5 - 1
  ((5 * x + 3 * y) / (x^2 - y^2) + (2 * x) / (y^2 - x^2)) / (1 / (x^2 * y - x * y^2)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2151_215135


namespace NUMINAMATH_CALUDE_sin_cos_product_for_specific_tan_l2151_215163

theorem sin_cos_product_for_specific_tan (α : Real) (h : Real.tan α = 1/2) :
  Real.sin α * Real.cos α = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_for_specific_tan_l2151_215163


namespace NUMINAMATH_CALUDE_circular_pool_area_l2151_215142

theorem circular_pool_area (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) : ∃ (r : ℝ), r^2 = 244 ∧ π * r^2 = 244 * π := by
  sorry

end NUMINAMATH_CALUDE_circular_pool_area_l2151_215142


namespace NUMINAMATH_CALUDE_log_pieces_after_ten_cuts_l2151_215124

/-- The number of pieces obtained after cutting a log -/
def numPieces (cuts : ℕ) : ℕ := cuts + 1

/-- Theorem: The number of pieces obtained after 10 cuts on a log is 11 -/
theorem log_pieces_after_ten_cuts : numPieces 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_pieces_after_ten_cuts_l2151_215124


namespace NUMINAMATH_CALUDE_lesser_solution_quadratic_l2151_215184

theorem lesser_solution_quadratic (x : ℝ) : 
  x^2 + 9*x - 22 = 0 ∧ (∀ y : ℝ, y^2 + 9*y - 22 = 0 → x ≤ y) → x = -11 :=
by sorry

end NUMINAMATH_CALUDE_lesser_solution_quadratic_l2151_215184


namespace NUMINAMATH_CALUDE_bill_more_sticks_than_ted_l2151_215151

/-- Represents the number of objects thrown by a person -/
structure ThrowCount where
  sticks : ℕ
  rocks : ℕ

/-- Calculates the total number of objects thrown -/
def ThrowCount.total (t : ThrowCount) : ℕ := t.sticks + t.rocks

theorem bill_more_sticks_than_ted (bill : ThrowCount) (ted : ThrowCount) : 
  bill.total = 21 → 
  ted.rocks = 2 * bill.rocks → 
  ted.sticks = 10 → 
  ted.rocks = 10 → 
  bill.sticks - ted.sticks = 6 := by
sorry

end NUMINAMATH_CALUDE_bill_more_sticks_than_ted_l2151_215151


namespace NUMINAMATH_CALUDE_original_number_proof_l2151_215133

theorem original_number_proof (x : ℚ) : 1 + (1 / x) = 9 / 5 → x = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2151_215133


namespace NUMINAMATH_CALUDE_age_difference_l2151_215172

theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 23 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2151_215172


namespace NUMINAMATH_CALUDE_sum_remainder_l2151_215166

theorem sum_remainder (f y : ℤ) (hf : f % 5 = 3) (hy : y % 5 = 4) : (f + y) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l2151_215166


namespace NUMINAMATH_CALUDE_factorization_equality_l2151_215153

theorem factorization_equality (x y : ℝ) : 25*x - x*y^2 = x*(5+y)*(5-y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2151_215153


namespace NUMINAMATH_CALUDE_digit_addition_puzzle_l2151_215111

/-- Represents a four-digit number ABCD --/
def FourDigitNumber (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

theorem digit_addition_puzzle :
  ∃ (possible_d : Finset ℕ),
    (∀ a b c d : ℕ,
      a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧  -- Digits are less than 10
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- Digits are distinct
      FourDigitNumber a a b c + FourDigitNumber b c a d = FourDigitNumber d b c d →  -- AABC + BCAD = DBCD
      d ∈ possible_d) ∧
    possible_d.card = 9  -- There are 9 possible values for D
  := by sorry

end NUMINAMATH_CALUDE_digit_addition_puzzle_l2151_215111


namespace NUMINAMATH_CALUDE_hammingDistance_bounds_hammingDistance_triangle_inequality_l2151_215141

/-- A byte is a list of booleans representing binary digits. -/
def Byte := List Bool

/-- The Hamming distance between two bytes is the number of positions at which they differ. -/
def hammingDistance (u v : Byte) : Nat :=
  (u.zip v).filter (fun (a, b) => a ≠ b) |>.length

/-- Theorem stating that the Hamming distance between two bytes is bounded by 0 and the length of the bytes. -/
theorem hammingDistance_bounds (u v : Byte) (h : u.length = v.length) :
    0 ≤ hammingDistance u v ∧ hammingDistance u v ≤ u.length := by
  sorry

/-- Theorem stating the triangle inequality for Hamming distance. -/
theorem hammingDistance_triangle_inequality (u v w : Byte) 
    (hu : u.length = v.length) (hv : v.length = w.length) :
    hammingDistance u v ≤ hammingDistance w u + hammingDistance w v := by
  sorry

end NUMINAMATH_CALUDE_hammingDistance_bounds_hammingDistance_triangle_inequality_l2151_215141


namespace NUMINAMATH_CALUDE_problem_statement_l2151_215145

theorem problem_statement : 7^2 - 2 * 6 + (3^2 - 1) = 45 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2151_215145


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_tangent_line_k_value_l2151_215139

/-- A line is tangent to a parabola if and only if the discriminant of their intersection equation is zero -/
theorem line_tangent_to_parabola (k : ℝ) : 
  (∃ x y : ℝ, 4*x - 3*y + k = 0 ∧ y^2 = 16*x ∧ 
   ∀ x' y' : ℝ, (4*x' - 3*y' + k = 0 ∧ y'^2 = 16*x') → (x' = x ∧ y' = y)) ↔ 
  k = 9 := by
  sorry

/-- The value of k for which the line 4x - 3y + k = 0 is tangent to the parabola y² = 16x is 9 -/
theorem tangent_line_k_value : 
  ∃! k : ℝ, ∃ x y : ℝ, 4*x - 3*y + k = 0 ∧ y^2 = 16*x ∧ 
  ∀ x' y' : ℝ, (4*x' - 3*y' + k = 0 ∧ y'^2 = 16*x') → (x' = x ∧ y' = y) := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_tangent_line_k_value_l2151_215139


namespace NUMINAMATH_CALUDE_ivan_work_and_charity_l2151_215132

/-- Represents Ivan Petrovich's daily work and financial situation --/
structure IvanPetrovich where
  workDays : ℕ -- number of working days per month
  sleepHours : ℕ -- hours of sleep per day
  workHours : ℝ -- hours of work per day
  hobbyRatio : ℝ -- ratio of hobby time to work time
  hourlyRate : ℝ -- rubles earned per hour of work
  rentalIncome : ℝ -- monthly rental income in rubles
  charityRatio : ℝ -- ratio of charity donation to rest hours
  monthlyExpenses : ℝ -- monthly expenses excluding charity in rubles

/-- Theorem stating Ivan Petrovich's work hours and charity donation --/
theorem ivan_work_and_charity 
  (ivan : IvanPetrovich)
  (h1 : ivan.workDays = 21)
  (h2 : ivan.sleepHours = 8)
  (h3 : ivan.hobbyRatio = 2)
  (h4 : ivan.hourlyRate = 3000)
  (h5 : ivan.rentalIncome = 14000)
  (h6 : ivan.charityRatio = 1/3)
  (h7 : ivan.monthlyExpenses = 70000)
  (h8 : 24 = ivan.sleepHours + ivan.workHours + ivan.hobbyRatio * ivan.workHours + (24 - ivan.sleepHours - ivan.workHours * (1 + ivan.hobbyRatio)))
  (h9 : ivan.workDays * (ivan.hourlyRate * ivan.workHours + ivan.charityRatio * (24 - ivan.sleepHours - ivan.workHours * (1 + ivan.hobbyRatio)) * 1000) + ivan.rentalIncome = ivan.monthlyExpenses + ivan.workDays * ivan.charityRatio * (24 - ivan.sleepHours - ivan.workHours * (1 + ivan.hobbyRatio)) * 1000) :
  ivan.workHours = 2 ∧ ivan.workDays * ivan.charityRatio * (24 - ivan.sleepHours - ivan.workHours * (1 + ivan.hobbyRatio)) * 1000 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_ivan_work_and_charity_l2151_215132


namespace NUMINAMATH_CALUDE_cakes_sold_is_six_l2151_215159

/-- The number of cakes sold during dinner today, given the number of cakes
    baked today, yesterday, and the number of cakes left. -/
def cakes_sold_during_dinner (cakes_baked_today cakes_baked_yesterday cakes_left : ℕ) : ℕ :=
  cakes_baked_today + cakes_baked_yesterday - cakes_left

/-- Theorem stating that the number of cakes sold during dinner today is 6,
    given the specific conditions of the problem. -/
theorem cakes_sold_is_six :
  cakes_sold_during_dinner 5 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cakes_sold_is_six_l2151_215159


namespace NUMINAMATH_CALUDE_javier_speech_time_l2151_215108

theorem javier_speech_time (outline_time writing_time practice_time total_time : ℕ) : 
  outline_time = 30 →
  writing_time = outline_time + 28 →
  practice_time = writing_time / 2 →
  total_time = outline_time + writing_time + practice_time →
  total_time = 117 :=
by sorry

end NUMINAMATH_CALUDE_javier_speech_time_l2151_215108


namespace NUMINAMATH_CALUDE_rational_function_value_l2151_215106

theorem rational_function_value (x : ℝ) (h : x ≠ 5) :
  x = 4 → (x^2 - 3*x - 10) / (x - 5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l2151_215106


namespace NUMINAMATH_CALUDE_return_speed_calculation_l2151_215192

/-- Proves that given a round trip with specified conditions, the return speed is 30 km/hr -/
theorem return_speed_calculation (distance : ℝ) (speed_going : ℝ) (average_speed : ℝ) 
  (h1 : distance = 150)
  (h2 : speed_going = 50)
  (h3 : average_speed = 37.5) : 
  (2 * distance) / ((distance / speed_going) + (distance / ((2 * distance) / average_speed - distance / speed_going))) = 30 :=
by sorry

end NUMINAMATH_CALUDE_return_speed_calculation_l2151_215192


namespace NUMINAMATH_CALUDE_a_beats_b_by_seven_seconds_l2151_215150

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ
  time : ℝ

/-- The race scenario -/
def race_scenario (a b : Runner) : Prop :=
  a.distance = 280 ∧
  b.distance = 224 ∧
  a.time = 28 ∧
  a.speed = a.distance / a.time ∧
  b.speed = b.distance / a.time

/-- Theorem stating that A beats B by 7 seconds -/
theorem a_beats_b_by_seven_seconds (a b : Runner) (h : race_scenario a b) :
  b.distance / b.speed - a.time = 7 := by
  sorry


end NUMINAMATH_CALUDE_a_beats_b_by_seven_seconds_l2151_215150


namespace NUMINAMATH_CALUDE_cubic_derivative_odd_implies_nonzero_l2151_215180

/-- A cubic function with a constant term of 2 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- A function is odd if f(-x) = -f(x) for all x -/
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem cubic_derivative_odd_implies_nonzero (a b c : ℝ) :
  is_odd (f' a b c) → a^2 + c^2 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_cubic_derivative_odd_implies_nonzero_l2151_215180


namespace NUMINAMATH_CALUDE_function_bounded_by_identity_l2151_215185

/-- For a differentiable function f: ℝ → ℝ, if f(x) ≤ f'(x) for all x in ℝ, then f(x) ≤ x for all x in ℝ. -/
theorem function_bounded_by_identity (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x ≤ deriv f x) : ∀ x, f x ≤ x := by
  sorry

end NUMINAMATH_CALUDE_function_bounded_by_identity_l2151_215185


namespace NUMINAMATH_CALUDE_bakery_doughnuts_given_away_l2151_215102

theorem bakery_doughnuts_given_away (total_doughnuts : ℕ) (box_capacity : ℕ) (boxes_sold : ℕ) : 
  total_doughnuts = 300 → 
  box_capacity = 10 → 
  boxes_sold = 27 → 
  (total_doughnuts - boxes_sold * box_capacity) = 30 := by
sorry

end NUMINAMATH_CALUDE_bakery_doughnuts_given_away_l2151_215102


namespace NUMINAMATH_CALUDE_gift_wrapping_l2151_215191

theorem gift_wrapping (total_rolls total_gifts first_roll_gifts second_roll_gifts : ℕ) :
  total_rolls = 3 →
  total_gifts = 12 →
  first_roll_gifts = 3 →
  second_roll_gifts = 5 →
  total_gifts = first_roll_gifts + second_roll_gifts + (total_gifts - (first_roll_gifts + second_roll_gifts)) →
  (total_gifts - (first_roll_gifts + second_roll_gifts)) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_l2151_215191


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2151_215109

theorem inequality_proof (a b n : ℕ) (h1 : a > b) (h2 : a * b - 1 = n^2) :
  a - b ≥ Real.sqrt (4 * n - 3) := by
  sorry

theorem equality_condition (a b n : ℕ) (h1 : a > b) (h2 : a * b - 1 = n^2) :
  (a - b = Real.sqrt (4 * n - 3)) ↔ 
  (∃ u : ℕ, a = u^2 + 2*u + 2 ∧ b = u^2 + 1 ∧ n = u^2 + u + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2151_215109


namespace NUMINAMATH_CALUDE_choir_size_l2151_215121

/-- Given an orchestra with female and male students, and a choir with three times
    the number of people in the orchestra, calculate the number of people in the choir. -/
theorem choir_size (female_students male_students : ℕ) 
  (h1 : female_students = 18) 
  (h2 : male_students = 25) : 
  3 * (female_students + male_students) = 129 := by
  sorry

end NUMINAMATH_CALUDE_choir_size_l2151_215121


namespace NUMINAMATH_CALUDE_money_division_l2151_215152

theorem money_division (p q r : ℕ) (total : ℝ) (h1 : p + q + r = 22) (h2 : 12 * total / 22 - 7 * total / 22 = 5000) :
  7 * total / 22 - 3 * total / 22 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l2151_215152


namespace NUMINAMATH_CALUDE_common_ratio_is_two_l2151_215173

/-- An increasing geometric sequence with specific conditions -/
structure IncreasingGeometricSequence where
  a : ℕ → ℝ
  is_increasing : ∀ n, a n < a (n + 1)
  is_geometric : ∃ q : ℝ, q > 1 ∧ ∀ n, a (n + 1) = a n * q
  a2_eq_2 : a 2 = 2
  a4_minus_a3_eq_4 : a 4 - a 3 = 4

/-- The common ratio of the increasing geometric sequence is 2 -/
theorem common_ratio_is_two (seq : IncreasingGeometricSequence) : 
  ∃ q : ℝ, (∀ n, seq.a (n + 1) = seq.a n * q) ∧ q = 2 :=
sorry

end NUMINAMATH_CALUDE_common_ratio_is_two_l2151_215173


namespace NUMINAMATH_CALUDE_raduzhny_population_l2151_215188

/-- The number of villages in Sunny Valley -/
def num_villages : ℕ := 10

/-- The population of Znoynoe village -/
def znoynoe_population : ℕ := 1000

/-- The amount by which Znoynoe's population exceeds the average -/
def excess_population : ℕ := 90

/-- The total population of all villages in Sunny Valley -/
def total_population : ℕ := znoynoe_population + (num_villages - 1) * (znoynoe_population - excess_population)

/-- The average population of villages in Sunny Valley -/
def average_population : ℕ := total_population / num_villages

theorem raduzhny_population : 
  ∃ (raduzhny_pop : ℕ), 
    raduzhny_pop = average_population ∧ 
    raduzhny_pop = 900 :=
sorry

end NUMINAMATH_CALUDE_raduzhny_population_l2151_215188


namespace NUMINAMATH_CALUDE_min_abs_phi_l2151_215170

/-- Given a function y = 3cos(2x + φ) with its graph symmetric about (2π/3, 0),
    the minimum value of |φ| is π/6 -/
theorem min_abs_phi (φ : ℝ) : 
  (∀ x, 3 * Real.cos (2 * x + φ) = 3 * Real.cos (2 * (4 * π / 3 - x) + φ)) →
  (∃ k : ℤ, φ = k * π - 5 * π / 6) →
  π / 6 ≤ |φ| ∧ (∃ φ₀, |φ₀| = π / 6 ∧ 
    (∀ x, 3 * Real.cos (2 * x + φ₀) = 3 * Real.cos (2 * (4 * π / 3 - x) + φ₀))) :=
by sorry

end NUMINAMATH_CALUDE_min_abs_phi_l2151_215170


namespace NUMINAMATH_CALUDE_max_ratio_squared_l2151_215197

theorem max_ratio_squared (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a^2 - 2*b^2 = 0)
  (h2 : a^2 + y^2 = b^2 + x^2)
  (h3 : b^2 + x^2 = (a - x)^2 + (b - y)^2)
  (h4 : 0 ≤ x ∧ x < a)
  (h5 : 0 ≤ y ∧ y < b)
  (h6 : x^2 + y^2 = b^2) :
  ∃ (ρ : ℝ), ρ^2 = 2 ∧ ∀ (c : ℝ), (c = a / b → c^2 ≤ ρ^2) :=
sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l2151_215197


namespace NUMINAMATH_CALUDE_positive_number_equality_l2151_215118

theorem positive_number_equality (x : ℝ) (h1 : x > 0) 
  (h2 : (2/3) * x = (64/216) * (1/x)) : x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_equality_l2151_215118


namespace NUMINAMATH_CALUDE_fifth_term_is_x_l2151_215115

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define our specific sequence
def our_sequence (x y : ℝ) : ℕ → ℝ
| 0 => x + 2*y
| 1 => x - 2*y
| 2 => x + y
| 3 => x - y
| n + 4 => our_sequence x y 3 + (n + 1) * (our_sequence x y 1 - our_sequence x y 0)

theorem fifth_term_is_x (x y : ℝ) :
  is_arithmetic_sequence (our_sequence x y) →
  our_sequence x y 4 = x :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_x_l2151_215115


namespace NUMINAMATH_CALUDE_total_cartridge_cost_l2151_215116

def black_and_white_cost : ℕ := 27
def color_cost : ℕ := 32
def num_black_and_white : ℕ := 1
def num_color : ℕ := 3

theorem total_cartridge_cost :
  num_black_and_white * black_and_white_cost + num_color * color_cost = 123 :=
by sorry

end NUMINAMATH_CALUDE_total_cartridge_cost_l2151_215116


namespace NUMINAMATH_CALUDE_sample_size_calculation_l2151_215182

/-- Given a sample divided into groups, this theorem proves that when one group
    has a frequency of 36 and a rate of 0.25, the total sample size is 144. -/
theorem sample_size_calculation (n : ℕ) (f : ℕ) (r : ℚ)
  (h1 : f = 36)
  (h2 : r = 1/4)
  (h3 : r = f / n) :
  n = 144 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l2151_215182


namespace NUMINAMATH_CALUDE_circle_area_is_one_l2151_215131

theorem circle_area_is_one (r : ℝ) (h : r > 0) :
  (4 * (1 / (2 * Real.pi * r)) = 2 * r) → (Real.pi * r^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_is_one_l2151_215131


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l2151_215103

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l2151_215103


namespace NUMINAMATH_CALUDE_isosceles_triangle_other_side_l2151_215105

structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  is_isosceles : side1 = side2
  perimeter : side1 + side2 + base = 15

def has_side_6 (t : IsoscelesTriangle) : Prop :=
  t.side1 = 6 ∨ t.side2 = 6 ∨ t.base = 6

theorem isosceles_triangle_other_side (t : IsoscelesTriangle) 
  (h : has_side_6 t) : 
  (t.side1 = 3 ∧ t.side2 = 3) ∨ (t.side1 = 4.5 ∧ t.side2 = 4.5) ∨ t.base = 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_other_side_l2151_215105


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_large_rectangle_perimeter_proof_l2151_215110

/-- The perimeter of a rectangle composed of three squares with perimeter 24 each
    and three rectangles with perimeter 16 each is 52. -/
theorem large_rectangle_perimeter : ℝ → Prop :=
  fun (perimeter : ℝ) =>
    let square_perimeter := 24
    let small_rectangle_perimeter := 16
    let square_side := square_perimeter / 4
    let small_rectangle_width := (small_rectangle_perimeter / 2) - square_side
    let large_rectangle_height := square_side + small_rectangle_width
    let large_rectangle_width := 3 * square_side
    perimeter = 2 * (large_rectangle_height + large_rectangle_width) ∧
    perimeter = 52

/-- Proof of the theorem -/
theorem large_rectangle_perimeter_proof : large_rectangle_perimeter 52 := by
  sorry

#check large_rectangle_perimeter_proof

end NUMINAMATH_CALUDE_large_rectangle_perimeter_large_rectangle_perimeter_proof_l2151_215110


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2151_215158

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) :
  1/a + 1/b + 4/c + 9/d + 16/e + 25/f ≥ 25.6 ∧ 
  ∃ (a' b' c' d' e' f' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 0 < f' ∧
    a' + b' + c' + d' + e' + f' = 10 ∧
    1/a' + 1/b' + 4/c' + 9/d' + 16/e' + 25/f' = 25.6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2151_215158


namespace NUMINAMATH_CALUDE_negation_equivalence_l2151_215143

/-- A triangle is a geometric shape with three sides and three angles. -/
structure Triangle where
  -- We don't need to define the specifics of a triangle for this problem

/-- An angle is obtuse if it is greater than 90 degrees. -/
def is_obtuse_angle (angle : ℝ) : Prop := angle > 90

/-- The original statement: Every triangle has at least two obtuse angles. -/
def original_statement : Prop :=
  ∀ t : Triangle, ∃ a b : ℝ, is_obtuse_angle a ∧ is_obtuse_angle b ∧ a ≠ b

/-- The negation: There exists a triangle that has at most one obtuse angle. -/
def negation : Prop :=
  ∃ t : Triangle, ∀ a b : ℝ, is_obtuse_angle a ∧ is_obtuse_angle b → a = b

/-- The negation of the original statement is equivalent to the given negation. -/
theorem negation_equivalence : ¬original_statement ↔ negation := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2151_215143


namespace NUMINAMATH_CALUDE_sum_not_prime_l2151_215164

theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_sum_not_prime_l2151_215164


namespace NUMINAMATH_CALUDE_payment_plan_difference_l2151_215186

def purchase_price : ℕ := 1500
def down_payment : ℕ := 200
def num_monthly_payments : ℕ := 24
def monthly_payment : ℕ := 65

theorem payment_plan_difference :
  (down_payment + num_monthly_payments * monthly_payment) - purchase_price = 260 := by
  sorry

end NUMINAMATH_CALUDE_payment_plan_difference_l2151_215186


namespace NUMINAMATH_CALUDE_unique_m_solution_l2151_215149

def S (n : ℕ) : ℕ := n^2

def a (n : ℕ) : ℕ := 2*n - 1

theorem unique_m_solution :
  ∃! m : ℕ+, 
    (∀ n : ℕ, S n = n^2) ∧ 
    (S m = (a m.val + a (m.val + 1)) / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_m_solution_l2151_215149


namespace NUMINAMATH_CALUDE_max_value_at_two_a_is_max_point_l2151_215160

-- Define the function f(x) = -x^3 + 12x
def f (x : ℝ) : ℝ := -x^3 + 12*x

-- State the theorem
theorem max_value_at_two : 
  ∀ x : ℝ, f x ≤ f 2 := by
sorry

-- Define a as the point where f reaches its maximum value
def a : ℝ := 2

-- State that a is indeed the point of maximum value
theorem a_is_max_point : 
  ∀ x : ℝ, f x ≤ f a := by
sorry

end NUMINAMATH_CALUDE_max_value_at_two_a_is_max_point_l2151_215160


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2151_215199

/-- For a quadratic equation qx^2 - 8x + 2 = 0 with q ≠ 0, it has only one solution iff q = 8 -/
theorem unique_solution_quadratic (q : ℝ) (hq : q ≠ 0) :
  (∃! x : ℝ, q * x^2 - 8 * x + 2 = 0) ↔ q = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2151_215199


namespace NUMINAMATH_CALUDE_smallest_N_for_Q_condition_l2151_215196

def Q (N : ℕ) : ℚ := ((2 * N + 3) / 3 : ℚ) / (N + 1 : ℚ)

theorem smallest_N_for_Q_condition : 
  ∀ N : ℕ, 
    N > 0 → 
    N % 6 = 0 → 
    (∀ k : ℕ, k > 0 → k % 6 = 0 → k < N → Q k ≥ 7/10) → 
    Q N < 7/10 → 
    N = 12 := by sorry

end NUMINAMATH_CALUDE_smallest_N_for_Q_condition_l2151_215196


namespace NUMINAMATH_CALUDE_sum_of_common_elements_l2151_215189

/-- Arithmetic progression with first term 4 and common difference 3 -/
def arithmetic_seq (n : ℕ) : ℕ := 4 + 3 * n

/-- Geometric progression with first term 10 and common ratio 2 -/
def geometric_seq (k : ℕ) : ℕ := 10 * 2^k

/-- Sequence of common elements in both progressions -/
def common_seq (m : ℕ) : ℕ := 10 * 4^m

theorem sum_of_common_elements : 
  (Finset.range 10).sum common_seq = 3495250 := by sorry

end NUMINAMATH_CALUDE_sum_of_common_elements_l2151_215189


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2151_215176

def A : Set ℝ := {x | (x - 2) / (x + 3) ≤ 0}
def B : Set ℝ := {x | x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x | -3 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2151_215176


namespace NUMINAMATH_CALUDE_inequality_proof_l2151_215187

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_condition : a + b + c ≤ 3) : 
  (3 > (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1)) ∧ 
   (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1)) ≥ 3 / 2) ∧
  ((a + 1) / (a * (a + 2)) + (b + 1) / (b * (b + 2)) + (c + 1) / (c * (c + 2)) ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2151_215187


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2151_215120

theorem smallest_number_with_given_remainders : ∃ (a : ℕ), (
  (a % 3 = 1) ∧
  (a % 6 = 3) ∧
  (a % 7 = 4) ∧
  (∀ b : ℕ, b < a → (b % 3 ≠ 1 ∨ b % 6 ≠ 3 ∨ b % 7 ≠ 4))
) ∧ a = 39 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2151_215120


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2151_215181

theorem diophantine_equation_solution (a b c : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : ∃ (x y z : ℤ), (x, y, z) ≠ (0, 0, 0) ∧ a * x^2 + b * y^2 + c * z^2 = 0) :
  ∃ (x y z : ℚ), a * x^2 + b * y^2 + c * z^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2151_215181


namespace NUMINAMATH_CALUDE_initial_boxes_count_l2151_215183

theorem initial_boxes_count (ali_boxes_per_circle ernie_boxes_per_circle ali_circles ernie_circles : ℕ) 
  (h1 : ali_boxes_per_circle = 8)
  (h2 : ernie_boxes_per_circle = 10)
  (h3 : ali_circles = 5)
  (h4 : ernie_circles = 4) :
  ali_boxes_per_circle * ali_circles + ernie_boxes_per_circle * ernie_circles = 80 :=
by sorry

end NUMINAMATH_CALUDE_initial_boxes_count_l2151_215183


namespace NUMINAMATH_CALUDE_triangle_side_sum_range_l2151_215101

/-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively,
    if b^2 + c^2 - a^2 = bc, AB · BC > 0, and a = √3/2,
    then √3/2 < b + c < 3/2. -/
theorem triangle_side_sum_range (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- positive angles
  A + B + C = π →  -- angles sum to π
  b^2 + c^2 - a^2 = b * c →  -- given condition
  (b * c * Real.cos A) > 0 →  -- AB · BC > 0
  a = Real.sqrt 3 / 2 →  -- given condition
  Real.sqrt 3 / 2 < b + c ∧ b + c < 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_sum_range_l2151_215101


namespace NUMINAMATH_CALUDE_triangle_theorem_l2151_215178

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (2 * t.b - t.a) * Real.cos (t.A + t.B) = -t.c * Real.cos t.A)
  (h2 : t.c = 3)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (4 * Real.sqrt 3) / 3) :
  t.C = π/3 ∧ t.a + t.b = 5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2151_215178


namespace NUMINAMATH_CALUDE_altitude_difference_l2151_215112

theorem altitude_difference (a b c : ℤ) (ha : a = -112) (hb : b = -80) (hc : c = -25) :
  (max a (max b c) - min a (min b c) : ℤ) = 87 := by
  sorry

end NUMINAMATH_CALUDE_altitude_difference_l2151_215112


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_pi_over_2_l2151_215144

theorem cos_2alpha_minus_pi_over_2 (α : ℝ) :
  (Real.cos α = -5/13) → (Real.sin α = 12/13) → Real.cos (2*α - π/2) = -120/169 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_pi_over_2_l2151_215144


namespace NUMINAMATH_CALUDE_arithmetic_sum_equals_square_l2151_215114

theorem arithmetic_sum_equals_square (n : ℕ) :
  let first_term := 1
  let last_term := 2*n + 3
  let num_terms := n + 2
  (num_terms * (first_term + last_term)) / 2 = (n + 2)^2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sum_equals_square_l2151_215114


namespace NUMINAMATH_CALUDE_water_distribution_l2151_215130

/-- Proves that given 122 ounces of water, after filling six 5-ounce glasses and four 8-ounce glasses,
    the remaining water can fill exactly 15 four-ounce glasses. -/
theorem water_distribution (total_water : ℕ) (five_oz_glasses : ℕ) (eight_oz_glasses : ℕ) 
  (four_oz_glasses : ℕ) : 
  total_water = 122 ∧ 
  five_oz_glasses = 6 ∧ 
  eight_oz_glasses = 4 ∧ 
  four_oz_glasses * 4 = total_water - (five_oz_glasses * 5 + eight_oz_glasses * 8) → 
  four_oz_glasses = 15 :=
by sorry

end NUMINAMATH_CALUDE_water_distribution_l2151_215130


namespace NUMINAMATH_CALUDE_factor_expression_l2151_215127

theorem factor_expression (x : ℝ) : 16 * x^2 + 8 * x = 8 * x * (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2151_215127


namespace NUMINAMATH_CALUDE_dress_design_count_l2151_215175

/-- The number of available fabric colors -/
def num_colors : ℕ := 5

/-- The number of available patterns -/
def num_patterns : ℕ := 4

/-- The number of available sleeve styles -/
def num_sleeve_styles : ℕ := 3

/-- Each dress design requires exactly one color, one pattern, and one sleeve style -/
axiom dress_design_composition : True

/-- The total number of possible dress designs -/
def total_dress_designs : ℕ := num_colors * num_patterns * num_sleeve_styles

theorem dress_design_count : total_dress_designs = 60 := by
  sorry

end NUMINAMATH_CALUDE_dress_design_count_l2151_215175


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2151_215168

theorem unique_solution_quadratic (k : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 3) = k + 3 * x) ↔ k = 35 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2151_215168


namespace NUMINAMATH_CALUDE_nancy_insurance_percentage_l2151_215125

/-- Given a monthly insurance cost and an annual payment, 
    calculate the percentage of the total cost being paid. -/
def insurance_percentage (monthly_cost : ℚ) (annual_payment : ℚ) : ℚ :=
  (annual_payment / (monthly_cost * 12)) * 100

/-- Theorem stating that for a monthly cost of $80 and an annual payment of $384,
    the percentage paid is 40% of the total cost. -/
theorem nancy_insurance_percentage :
  insurance_percentage 80 384 = 40 := by
  sorry

end NUMINAMATH_CALUDE_nancy_insurance_percentage_l2151_215125


namespace NUMINAMATH_CALUDE_highest_probability_high_speed_rail_l2151_215134

theorem highest_probability_high_speed_rail (beidou tianyan high_speed_rail : ℕ) :
  beidou = 3 →
  tianyan = 2 →
  high_speed_rail = 5 →
  let total := beidou + tianyan + high_speed_rail
  (high_speed_rail : ℚ) / total > (beidou : ℚ) / total ∧
  (high_speed_rail : ℚ) / total > (tianyan : ℚ) / total :=
by sorry

end NUMINAMATH_CALUDE_highest_probability_high_speed_rail_l2151_215134


namespace NUMINAMATH_CALUDE_trig_identity_l2151_215155

theorem trig_identity : 
  Real.cos (54 * π / 180) * Real.cos (24 * π / 180) + 
  2 * Real.sin (12 * π / 180) * Real.cos (12 * π / 180) * Real.sin (126 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l2151_215155


namespace NUMINAMATH_CALUDE_inequality_solution_l2151_215113

theorem inequality_solution :
  let ineq1 : ℝ → Prop := λ x => x > 1
  let ineq2 : ℝ → Prop := λ x => x > 4
  let ineq3 : ℝ → Prop := λ x => 2 - x > -1
  let ineq4 : ℝ → Prop := λ x => x < 2
  (∀ x : ℤ, (ineq1 x ∧ ineq3 x) ↔ x = 2) ∧
  (∀ x : ℤ, ¬(ineq1 x ∧ ineq2 x ∧ x = 2)) ∧
  (∀ x : ℤ, ¬(ineq1 x ∧ ineq4 x ∧ x = 2)) ∧
  (∀ x : ℤ, ¬(ineq2 x ∧ ineq3 x ∧ x = 2)) ∧
  (∀ x : ℤ, ¬(ineq2 x ∧ ineq4 x ∧ x = 2)) ∧
  (∀ x : ℤ, ¬(ineq3 x ∧ ineq4 x ∧ x = 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2151_215113


namespace NUMINAMATH_CALUDE_root_product_equals_27_l2151_215157

theorem root_product_equals_27 : 
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by sorry

end NUMINAMATH_CALUDE_root_product_equals_27_l2151_215157


namespace NUMINAMATH_CALUDE_traffic_light_theorem_l2151_215167

structure TrafficLightSystem where
  p1 : ℝ
  p2 : ℝ
  p3 : ℝ
  h1 : 0 ≤ p1 ∧ p1 ≤ 1
  h2 : 0 ≤ p2 ∧ p2 ≤ 1
  h3 : 0 ≤ p3 ∧ p3 ≤ 1
  h4 : p1 < p2
  h5 : p2 < p3
  h6 : p1 = 1/2
  h7 : (1 - p1) * (1 - p2) * (1 - p3) = 1/24
  h8 : p1 * p2 * p3 = 1/4

def prob_first_red_at_third (s : TrafficLightSystem) : ℝ :=
  (1 - s.p1) * (1 - s.p2) * s.p3

def expected_red_lights (s : TrafficLightSystem) : ℝ :=
  s.p1 + s.p2 + s.p3

theorem traffic_light_theorem (s : TrafficLightSystem) :
  prob_first_red_at_third s = 1/8 ∧ expected_red_lights s = 23/12 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_theorem_l2151_215167


namespace NUMINAMATH_CALUDE_fast_reader_time_l2151_215148

/-- Given two people, where one reads 4 times faster than the other, 
    prove that if the slower reader takes 90 minutes to read a book, 
    the faster reader will take 22.5 minutes to read the same book. -/
theorem fast_reader_time (slow_reader_time : ℝ) (speed_ratio : ℝ) 
    (h1 : slow_reader_time = 90) 
    (h2 : speed_ratio = 4) : 
  slow_reader_time / speed_ratio = 22.5 := by
  sorry

#check fast_reader_time

end NUMINAMATH_CALUDE_fast_reader_time_l2151_215148


namespace NUMINAMATH_CALUDE_weaving_problem_l2151_215107

def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem weaving_problem (a₁ d : ℕ) (h₁ : a₁ > 0) (h₂ : d > 0) :
  (arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 2 + 
   arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d 4 = 24) →
  (arithmetic_sequence a₁ d 7 = arithmetic_sequence a₁ d 1 * arithmetic_sequence a₁ d 2) →
  arithmetic_sequence a₁ d 10 = 21 := by
  sorry

end NUMINAMATH_CALUDE_weaving_problem_l2151_215107


namespace NUMINAMATH_CALUDE_jason_initial_cards_l2151_215177

theorem jason_initial_cards (initial_cards final_cards bought_cards : ℕ) 
  (h1 : bought_cards = 224)
  (h2 : final_cards = 900)
  (h3 : final_cards = initial_cards + bought_cards) :
  initial_cards = 676 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l2151_215177


namespace NUMINAMATH_CALUDE_equation_solutions_l2151_215194

def equation (x y : ℝ) : Prop :=
  x^2 - x*y + y^2 - x + 3*y - 7 = 0

def solution_set : Set (ℝ × ℝ) :=
  {(3,1), (-1,1), (3,-1), (-3,-1), (-1,-5)}

theorem equation_solutions :
  (∀ (x y : ℝ), (x, y) ∈ solution_set ↔ equation x y) ∧
  equation 3 1 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2151_215194


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2151_215165

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 2 + a 12 + a 13 = 24) : 
  a 7 = 6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2151_215165


namespace NUMINAMATH_CALUDE_distribution_count_l2151_215193

-- Define the number of balls and boxes
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Define the Stirling number of the second kind function
noncomputable def stirling_second (n k : ℕ) : ℕ :=
  sorry  -- Implementation of Stirling number of the second kind

-- Theorem statement
theorem distribution_count :
  (stirling_second num_balls num_boxes) = 25 :=
sorry

end NUMINAMATH_CALUDE_distribution_count_l2151_215193


namespace NUMINAMATH_CALUDE_value_of_4x2y2_l2151_215136

theorem value_of_4x2y2 (x y : ℤ) (h : y^2 + 4*x^2*y^2 = 40*x^2 + 817) : 
  4*x^2*y^2 = 3484 := by sorry

end NUMINAMATH_CALUDE_value_of_4x2y2_l2151_215136


namespace NUMINAMATH_CALUDE_count_valid_house_numbers_l2151_215140

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_valid_house_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  is_prime (n / 1000) ∧ 
  is_prime (n % 1000) ∧
  (n / 1000) < 100 ∧
  (n % 1000) < 500 ∧
  ∀ d : ℕ, d < 5 → (n / 10^d % 10) ≠ 0

theorem count_valid_house_numbers :
  ∃ (s : Finset ℕ), (∀ n ∈ s, is_valid_house_number n) ∧ s.card = 1302 :=
sorry

end NUMINAMATH_CALUDE_count_valid_house_numbers_l2151_215140


namespace NUMINAMATH_CALUDE_min_distance_point_to_circle_l2151_215104

/-- The minimum distance between the point (3,4) and any point on the circle x^2 + y^2 = 1 is 4 -/
theorem min_distance_point_to_circle : ∃ (d : ℝ),
  d = 4 ∧
  ∀ (x y : ℝ), x^2 + y^2 = 1 → 
  d ≤ Real.sqrt ((x - 3)^2 + (y - 4)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_to_circle_l2151_215104


namespace NUMINAMATH_CALUDE_second_half_speed_l2151_215129

/-- Proves that given a journey of 224 km completed in 10 hours, where the first half is traveled at 21 km/hr, the speed for the second half of the journey is 24 km/hr. -/
theorem second_half_speed (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) :
  total_distance = 224 →
  total_time = 10 →
  first_half_speed = 21 →
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := total_time - first_half_time
  let second_half_speed := first_half_distance / second_half_time
  second_half_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_second_half_speed_l2151_215129


namespace NUMINAMATH_CALUDE_x_plus_y_equals_30_l2151_215146

theorem x_plus_y_equals_30 (x y : ℝ) 
  (h1 : |x| - x + y = 6) 
  (h2 : x + |y| + y = 8) : 
  x + y = 30 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_30_l2151_215146


namespace NUMINAMATH_CALUDE_length_of_segment_l2151_215171

/-- Given a line segment AB divided by points P and Q, prove that AB has length 25 -/
theorem length_of_segment (A B P Q : ℝ) : 
  (P - A) / (B - A) = 3 / 5 →  -- P divides AB in ratio 3:2
  (Q - A) / (B - A) = 2 / 5 →  -- Q divides AB in ratio 2:3
  Q - P = 5 →                  -- Distance between P and Q is 5 units
  B - A = 25 := by             -- Length of AB is 25 units
sorry

end NUMINAMATH_CALUDE_length_of_segment_l2151_215171


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2151_215195

theorem polynomial_simplification (r : ℝ) : 
  (2 * r^3 + 5 * r^2 + 4 * r - 3) - (r^3 + 4 * r^2 + 6 * r - 8) = r^3 + r^2 - 2 * r + 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2151_215195


namespace NUMINAMATH_CALUDE_student_count_l2151_215128

/-- If a student is ranked 17th from the right and 5th from the left in a line of students,
    then the total number of students is 21. -/
theorem student_count (n : ℕ) (rank_right rank_left : ℕ) 
  (h1 : rank_right = 17)
  (h2 : rank_left = 5)
  (h3 : n = rank_right + rank_left - 1) :
  n = 21 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2151_215128


namespace NUMINAMATH_CALUDE_triangle_sum_formula_l2151_215162

def triangleSum (n : ℕ) : ℕ := 8 * 2^n - 4

theorem triangle_sum_formula (n : ℕ) : 
  n ≥ 1 → 
  (∀ k, k ≥ 2 → triangleSum k = 2 * triangleSum (k-1) + 4) → 
  triangleSum 1 = 4 → 
  triangleSum n = 8 * 2^n - 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_formula_l2151_215162


namespace NUMINAMATH_CALUDE_function_extrema_implies_a_range_l2151_215169

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

theorem function_extrema_implies_a_range (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ x, f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_implies_a_range_l2151_215169
