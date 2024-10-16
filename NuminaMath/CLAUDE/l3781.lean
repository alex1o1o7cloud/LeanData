import Mathlib

namespace NUMINAMATH_CALUDE_train_crossing_time_l3781_378124

/-- The time taken for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 120 →
  train_speed = 67 * (1000 / 3600) →
  man_speed = 5 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 6 := by
sorry


end NUMINAMATH_CALUDE_train_crossing_time_l3781_378124


namespace NUMINAMATH_CALUDE_student_assignment_l3781_378197

theorem student_assignment (n : ℕ) (m : ℕ) (h1 : n = 4) (h2 : m = 3) :
  (Nat.choose n 2) * (Nat.factorial m) = 36 := by
  sorry

end NUMINAMATH_CALUDE_student_assignment_l3781_378197


namespace NUMINAMATH_CALUDE_factorial_1500_trailing_zeros_l3781_378115

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (Finset.range 13).sum fun i => n / (5 ^ (i + 1))

/-- 1500! has 374 trailing zeros -/
theorem factorial_1500_trailing_zeros :
  trailingZeros 1500 = 374 := by
  sorry

end NUMINAMATH_CALUDE_factorial_1500_trailing_zeros_l3781_378115


namespace NUMINAMATH_CALUDE_worm_distance_after_15_days_l3781_378189

/-- Represents the daily movement of a worm -/
structure WormMovement where
  forward : ℝ
  backward : ℝ

/-- Calculates the net daily distance traveled by the worm -/
def net_daily_distance (movement : WormMovement) : ℝ :=
  movement.forward - movement.backward

/-- Calculates the total distance traveled over a number of days -/
def total_distance (movement : WormMovement) (days : ℕ) : ℝ :=
  (net_daily_distance movement) * days

/-- The theorem to be proved -/
theorem worm_distance_after_15_days (worm_movement : WormMovement)
    (h1 : worm_movement.forward = 5)
    (h2 : worm_movement.backward = 3)
    : total_distance worm_movement 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_worm_distance_after_15_days_l3781_378189


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3781_378117

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3781_378117


namespace NUMINAMATH_CALUDE_sort_three_integers_correct_l3781_378152

/-- Algorithm to sort three positive integers in descending order -/
def sort_three_integers (a b c : ℕ+) : ℕ+ × ℕ+ × ℕ+ :=
  let step2 := if a ≤ b then (b, a, c) else (a, b, c)
  let step3 := let (x, y, z) := step2
                if x ≤ z then (z, y, x) else (x, y, z)
  let step4 := let (x, y, z) := step3
                if y ≤ z then (x, z, y) else (x, y, z)
  step4

/-- Theorem stating that the sorting algorithm produces a descending order result -/
theorem sort_three_integers_correct (a b c : ℕ+) :
  let (x, y, z) := sort_three_integers a b c
  x ≥ y ∧ y ≥ z :=
by
  sorry

end NUMINAMATH_CALUDE_sort_three_integers_correct_l3781_378152


namespace NUMINAMATH_CALUDE_iceland_visitors_iceland_visitor_count_l3781_378132

theorem iceland_visitors (total : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 50)
  (h2 : norway = 23)
  (h3 : both = 21)
  (h4 : neither = 23) :
  total = (total - norway - neither + both) + norway - both + neither :=
by sorry

theorem iceland_visitor_count : 
  ∃ (iceland : ℕ), iceland = 50 - 23 - 23 + 21 ∧ iceland = 25 :=
by sorry

end NUMINAMATH_CALUDE_iceland_visitors_iceland_visitor_count_l3781_378132


namespace NUMINAMATH_CALUDE_vector_decomposition_l3781_378179

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![0, -8, 9]
def p : Fin 3 → ℝ := ![0, -2, 1]
def q : Fin 3 → ℝ := ![3, 1, -1]
def r : Fin 3 → ℝ := ![4, 0, 1]

/-- Theorem stating that x can be expressed as a linear combination of p, q, and r -/
theorem vector_decomposition :
  x = (2 : ℝ) • p + (-4 : ℝ) • q + (3 : ℝ) • r :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l3781_378179


namespace NUMINAMATH_CALUDE_zeros_of_f_l3781_378111

def f (x : ℝ) : ℝ := x * (x^2 - 16)

theorem zeros_of_f :
  ∀ x : ℝ, f x = 0 ↔ x = -4 ∨ x = 0 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_f_l3781_378111


namespace NUMINAMATH_CALUDE_sum_product_over_sum_squares_l3781_378195

theorem sum_product_over_sum_squares (a b c : ℝ) (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h2 : a + b + c = 1) :
  (a * b + b * c + c * a) / (a^2 + b^2 + c^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_over_sum_squares_l3781_378195


namespace NUMINAMATH_CALUDE_cubic_less_than_square_l3781_378106

theorem cubic_less_than_square (x : ℚ) : 
  (x = 3/4 → x^3 < x^2) ∧ 
  (x = 5/3 → x^3 ≥ x^2) ∧ 
  (x = 1 → x^3 ≥ x^2) ∧ 
  (x = 3/2 → x^3 ≥ x^2) ∧ 
  (x = 21/20 → x^3 ≥ x^2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_less_than_square_l3781_378106


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3781_378177

theorem max_value_sqrt_sum (x : ℝ) (h : x ∈ Set.Icc (-49 : ℝ) 49) :
  ∃ (M : ℝ), M = 14 ∧ Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ M ∧
  ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (-49 : ℝ) 49 ∧ Real.sqrt (49 + x₀) + Real.sqrt (49 - x₀) = M :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3781_378177


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocal_l3781_378168

theorem quadratic_roots_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ - 7 = 0 →
  x₂^2 - 4*x₂ - 7 = 0 →
  x₁ ≠ 0 →
  x₂ ≠ 0 →
  1/x₁ + 1/x₂ = -4/7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocal_l3781_378168


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3781_378183

def complex_equation (z : ℂ) : Prop := (1 + Complex.I) * z = 2 * Complex.I

theorem z_in_fourth_quadrant (z : ℂ) (h : complex_equation z) : 
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3781_378183


namespace NUMINAMATH_CALUDE_alyssa_chicken_nuggets_l3781_378136

/-- Given 100 total chicken nuggets and two people eating twice as much as Alyssa,
    prove that Alyssa ate 20 chicken nuggets. -/
theorem alyssa_chicken_nuggets :
  ∀ (total : ℕ) (alyssa : ℕ),
    total = 100 →
    total = alyssa + 2 * alyssa + 2 * alyssa →
    alyssa = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_alyssa_chicken_nuggets_l3781_378136


namespace NUMINAMATH_CALUDE_rice_distribution_l3781_378142

theorem rice_distribution (R : ℚ) : 
  (7/10 : ℚ) * R - (3/10 : ℚ) * R = 20 → R = 50 := by
sorry

end NUMINAMATH_CALUDE_rice_distribution_l3781_378142


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3781_378148

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x < 0}
def N : Set ℝ := {x | |x| < 2}

-- Define propositions p and q
def p (a : ℝ) : Prop := a ∈ M
def q (a : ℝ) : Prop := a ∈ N

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬p a) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3781_378148


namespace NUMINAMATH_CALUDE_proportion_fourth_term_l3781_378199

theorem proportion_fourth_term (x y : ℝ) : 
  (0.25 / x = 2 / y) → x = 0.75 → y = 6 := by sorry

end NUMINAMATH_CALUDE_proportion_fourth_term_l3781_378199


namespace NUMINAMATH_CALUDE_james_tylenol_frequency_l3781_378161

/-- Proves that James takes Tylenol tablets every 6 hours given the conditions --/
theorem james_tylenol_frequency 
  (tablets_per_dose : ℕ)
  (mg_per_tablet : ℕ)
  (total_mg_per_day : ℕ)
  (hours_per_day : ℕ)
  (h1 : tablets_per_dose = 2)
  (h2 : mg_per_tablet = 375)
  (h3 : total_mg_per_day = 3000)
  (h4 : hours_per_day = 24) :
  (hours_per_day : ℚ) / ((total_mg_per_day : ℚ) / ((tablets_per_dose : ℚ) * mg_per_tablet)) = 6 := by
  sorry

#check james_tylenol_frequency

end NUMINAMATH_CALUDE_james_tylenol_frequency_l3781_378161


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_element5_l3781_378121

theorem pascal_triangle_row20_element5 : Nat.choose 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_element5_l3781_378121


namespace NUMINAMATH_CALUDE_cricket_count_l3781_378145

theorem cricket_count (initial : Float) (additional : Float) :
  initial = 7.0 → additional = 11.0 → initial + additional = 18.0 := by sorry

end NUMINAMATH_CALUDE_cricket_count_l3781_378145


namespace NUMINAMATH_CALUDE_pythagorean_triple_double_l3781_378147

/-- If (a, b, c) is a Pythagorean triple, then (2a, 2b, 2c) is also a Pythagorean triple. -/
theorem pythagorean_triple_double {a b c : ℕ} (h : a ^ 2 + b ^ 2 = c ^ 2) :
  (2 * a) ^ 2 + (2 * b) ^ 2 = (2 * c) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_double_l3781_378147


namespace NUMINAMATH_CALUDE_joan_football_games_l3781_378130

/-- The number of football games Joan attended this year -/
def games_this_year : ℕ := 4

/-- The number of football games Joan attended last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan attended -/
def total_games : ℕ := games_this_year + games_last_year

theorem joan_football_games :
  total_games = 13 := by sorry

end NUMINAMATH_CALUDE_joan_football_games_l3781_378130


namespace NUMINAMATH_CALUDE_isabella_hair_length_l3781_378185

/-- Calculates the length of hair after a given time period. -/
def hair_length (initial_length : ℝ) (growth_rate : ℝ) (months : ℝ) : ℝ :=
  initial_length + growth_rate * months

/-- Theorem stating that Isabella's hair length after y months is 18 + xy -/
theorem isabella_hair_length (x y : ℝ) :
  hair_length 18 x y = 18 + x * y := by
  sorry

end NUMINAMATH_CALUDE_isabella_hair_length_l3781_378185


namespace NUMINAMATH_CALUDE_worker_efficiency_l3781_378162

theorem worker_efficiency (p q : ℝ) (hp : p > 0) (hq : q > 0) : 
  p = 1 / 22 → p + q = 1 / 12 → p / q = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_worker_efficiency_l3781_378162


namespace NUMINAMATH_CALUDE_carolyn_practice_ratio_l3781_378180

/-- Represents Carolyn's music practice schedule and calculates the ratio of violin to piano practice time -/
theorem carolyn_practice_ratio :
  let piano_daily := 20 -- minutes of piano practice per day
  let days_per_week := 6 -- number of practice days per week
  let weeks_per_month := 4 -- number of weeks in a month
  let total_monthly := 1920 -- total practice time in minutes per month

  let piano_monthly := piano_daily * days_per_week * weeks_per_month
  let violin_monthly := total_monthly - piano_monthly
  let violin_daily := violin_monthly / (days_per_week * weeks_per_month)

  (violin_daily : ℚ) / piano_daily = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_practice_ratio_l3781_378180


namespace NUMINAMATH_CALUDE_prob_three_tails_in_eight_flips_l3781_378155

/-- The probability of flipping a tail -/
def p_tail : ℚ := 3/4

/-- The probability of flipping a head -/
def p_head : ℚ := 1/4

/-- The number of coin flips -/
def n_flips : ℕ := 8

/-- The number of tails we want to get -/
def n_tails : ℕ := 3

/-- The probability of getting exactly n_tails in n_flips of an unfair coin -/
def prob_exact_tails (n_flips n_tails : ℕ) (p_tail : ℚ) : ℚ :=
  (n_flips.choose n_tails) * (p_tail ^ n_tails) * ((1 - p_tail) ^ (n_flips - n_tails))

theorem prob_three_tails_in_eight_flips : 
  prob_exact_tails n_flips n_tails p_tail = 189/512 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_tails_in_eight_flips_l3781_378155


namespace NUMINAMATH_CALUDE_legos_in_box_l3781_378110

theorem legos_in_box (total : ℕ) (used : ℕ) (missing : ℕ) (in_box : ℕ) : 
  total = 500 → 
  used = total / 2 → 
  missing = 5 → 
  in_box = total - used - missing → 
  in_box = 245 := by
sorry

end NUMINAMATH_CALUDE_legos_in_box_l3781_378110


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3781_378149

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) :
  (1 / x + 1 / y) ≥ 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3781_378149


namespace NUMINAMATH_CALUDE_zainab_works_two_hours_per_day_l3781_378156

/-- Represents Zainab's flyer distribution job --/
structure FlyerJob where
  hourly_rate : ℕ
  days_per_week : ℕ
  total_weeks : ℕ
  total_earnings : ℕ

/-- Calculates the number of hours worked per day --/
def hours_per_day (job : FlyerJob) : ℚ :=
  job.total_earnings / (job.hourly_rate * job.days_per_week * job.total_weeks)

/-- Theorem stating that Zainab works 2 hours per day --/
theorem zainab_works_two_hours_per_day :
  let job := FlyerJob.mk 2 3 4 96
  hours_per_day job = 2 := by sorry

end NUMINAMATH_CALUDE_zainab_works_two_hours_per_day_l3781_378156


namespace NUMINAMATH_CALUDE_working_days_is_twenty_main_theorem_l3781_378150

/-- Represents the commute data for a period of working days -/
structure CommuteData where
  car_to_work : ℕ
  train_from_work : ℕ
  total_train_trips : ℕ

/-- Calculates the total number of working days based on commute data -/
def calculate_working_days (data : CommuteData) : ℕ :=
  data.car_to_work + data.total_train_trips

/-- Theorem stating that the number of working days is 20 given the specific commute data -/
theorem working_days_is_twenty (data : CommuteData) 
  (h1 : data.car_to_work = 12)
  (h2 : data.train_from_work = 11)
  (h3 : data.total_train_trips = 8)
  (h4 : data.car_to_work = data.train_from_work + 1) :
  calculate_working_days data = 20 := by
  sorry

/-- Main theorem to be proved -/
theorem main_theorem : ∃ (data : CommuteData), 
  data.car_to_work = 12 ∧ 
  data.train_from_work = 11 ∧ 
  data.total_train_trips = 8 ∧ 
  data.car_to_work = data.train_from_work + 1 ∧
  calculate_working_days data = 20 := by
  sorry

end NUMINAMATH_CALUDE_working_days_is_twenty_main_theorem_l3781_378150


namespace NUMINAMATH_CALUDE_remainder_problem_l3781_378103

theorem remainder_problem (G : ℕ) (h1 : G = 144) (h2 : 6215 % G = 23) : 7373 % G = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3781_378103


namespace NUMINAMATH_CALUDE_value_two_std_dev_below_mean_l3781_378182

/-- Given a normal distribution with mean 14.5 and standard deviation 1.7,
    the value that is exactly 2 standard deviations less than the mean is 11.1 -/
theorem value_two_std_dev_below_mean :
  let μ : ℝ := 14.5  -- mean
  let σ : ℝ := 1.7   -- standard deviation
  μ - 2 * σ = 11.1 := by
  sorry

end NUMINAMATH_CALUDE_value_two_std_dev_below_mean_l3781_378182


namespace NUMINAMATH_CALUDE_expression_evaluation_l3781_378178

theorem expression_evaluation (a b : ℝ) 
  (h : |a + 1| + (b - 2)^2 = 0) : 
  2 * (3 * a^2 - a * b + 1) - (-a^2 + 2 * a * b + 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3781_378178


namespace NUMINAMATH_CALUDE_tenth_largest_number_l3781_378167

/-- Given a list of digits, generate all possible three-digit numbers -/
def generateThreeDigitNumbers (digits : List Nat) : List Nat :=
  sorry

/-- Sort a list of numbers in descending order -/
def sortDescending (numbers : List Nat) : List Nat :=
  sorry

theorem tenth_largest_number : 
  let digits : List Nat := [5, 3, 1, 9]
  let threeDigitNumbers := generateThreeDigitNumbers digits
  let sortedNumbers := sortDescending threeDigitNumbers
  List.get! sortedNumbers 9 = 531 := by
  sorry

end NUMINAMATH_CALUDE_tenth_largest_number_l3781_378167


namespace NUMINAMATH_CALUDE_cloth_cost_price_theorem_l3781_378122

/-- Represents the cost price of one meter of cloth given the selling conditions --/
def cost_price_per_meter (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_meters) / total_meters

/-- Theorem stating that under the given conditions, the cost price per meter is 88 --/
theorem cloth_cost_price_theorem (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ)
    (h1 : total_meters = 45)
    (h2 : selling_price = 4500)
    (h3 : profit_per_meter = 12) :
    cost_price_per_meter total_meters selling_price profit_per_meter = 88 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_theorem_l3781_378122


namespace NUMINAMATH_CALUDE_sets_intersection_theorem_l3781_378169

def A (p q : ℝ) : Set ℝ := {x | x^2 + p*x + q = 0}
def B (p q : ℝ) : Set ℝ := {x | q*x^2 + p*x + 1 = 0}

theorem sets_intersection_theorem (p q : ℝ) :
  p ≠ 0 ∧ q ≠ 0 ∧ (A p q ∩ B p q).Nonempty ∧ (-2 ∈ A p q) →
  ((p = 1 ∧ q = -2) ∨ (p = 3 ∧ q = 2) ∨ (p = 5/2 ∧ q = 1)) :=
by sorry

end NUMINAMATH_CALUDE_sets_intersection_theorem_l3781_378169


namespace NUMINAMATH_CALUDE_square_cut_divisible_by_four_l3781_378129

/-- A rectangle on a grid --/
structure GridRectangle where
  length : ℕ
  width : ℕ

/-- A square on a grid --/
structure GridSquare where
  side : ℕ

/-- Function to cut a square into rectangles along grid lines --/
def cutSquareIntoRectangles (square : GridSquare) : List GridRectangle :=
  sorry

/-- Function to calculate the perimeter of a rectangle --/
def rectanglePerimeter (rect : GridRectangle) : ℕ :=
  2 * (rect.length + rect.width)

theorem square_cut_divisible_by_four (square : GridSquare) 
    (h : square.side = 2009) :
    ∃ (rect : GridRectangle), rect ∈ cutSquareIntoRectangles square ∧ 
    (rectanglePerimeter rect) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_cut_divisible_by_four_l3781_378129


namespace NUMINAMATH_CALUDE_car_engine_part_cost_l3781_378194

/-- Calculates the cost of a car engine part given labor and total cost information --/
theorem car_engine_part_cost
  (labor_rate : ℕ)
  (labor_hours : ℕ)
  (total_cost : ℕ)
  (h1 : labor_rate = 75)
  (h2 : labor_hours = 16)
  (h3 : total_cost = 2400) :
  total_cost - (labor_rate * labor_hours) = 1200 := by
  sorry

#check car_engine_part_cost

end NUMINAMATH_CALUDE_car_engine_part_cost_l3781_378194


namespace NUMINAMATH_CALUDE_cosine_sum_simplification_l3781_378166

theorem cosine_sum_simplification :
  Real.cos (π / 15) + Real.cos (4 * π / 15) + Real.cos (14 * π / 15) = (Real.sqrt 21 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_simplification_l3781_378166


namespace NUMINAMATH_CALUDE_triangle_properties_l3781_378187

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.cos t.C * (t.a * Real.cos t.B + t.b * Real.cos t.A) = t.c)
  (h2 : t.c = Real.sqrt 7)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2) :
  t.C = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3781_378187


namespace NUMINAMATH_CALUDE_digit_sum_is_two_l3781_378101

/-- Given a four-digit number abcd and a three-digit number bcd, where a, b, c, d are distinct digits 
    and abcd - bcd is a two-digit number, the sum of a, b, c, and d is 2. -/
theorem digit_sum_is_two (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  1000 * a + 100 * b + 10 * c + d > 999 →
  1000 * a + 100 * b + 10 * c + d - (100 * b + 10 * c + d) < 100 →
  a + b + c + d = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_is_two_l3781_378101


namespace NUMINAMATH_CALUDE_sequence_problem_l3781_378116

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_sum_a : a 1 + a 3 + a 5 + a 7 + a 9 = 50)
  (h_prod_b : b 4 * b 6 * b 14 * b 16 = 625) :
  (a 2 + a 8) / b 10 = 4 ∨ (a 2 + a 8) / b 10 = -4 :=
sorry

end NUMINAMATH_CALUDE_sequence_problem_l3781_378116


namespace NUMINAMATH_CALUDE_cake_ratio_correct_l3781_378114

/-- The ratio of cakes made each day compared to the previous day -/
def cake_ratio : ℝ := 2

/-- The number of cakes made on the first day -/
def first_day_cakes : ℕ := 10

/-- The number of cakes made on the sixth day -/
def sixth_day_cakes : ℕ := 320

/-- Theorem stating that the cake ratio is correct given the conditions -/
theorem cake_ratio_correct :
  (first_day_cakes : ℝ) * cake_ratio ^ 5 = sixth_day_cakes := by sorry

end NUMINAMATH_CALUDE_cake_ratio_correct_l3781_378114


namespace NUMINAMATH_CALUDE_machine_selling_price_l3781_378154

/-- Calculates the selling price of a machine given its costs and desired profit percentage -/
def selling_price (purchase_price repair_cost transport_cost profit_percent : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := total_cost * profit_percent / 100
  total_cost + profit

/-- Theorem stating that the selling price of the machine is 30000 Rs -/
theorem machine_selling_price :
  selling_price 14000 5000 1000 50 = 30000 := by
  sorry

end NUMINAMATH_CALUDE_machine_selling_price_l3781_378154


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3781_378128

theorem solution_set_of_inequality (x : ℝ) :
  (2 - x) / (x + 4) > 0 ↔ -4 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3781_378128


namespace NUMINAMATH_CALUDE_movie_theatre_attendance_l3781_378196

theorem movie_theatre_attendance (total_seats : ℕ) (adult_price child_price : ℚ) 
  (total_revenue : ℚ) (h_seats : total_seats = 250) (h_adult_price : adult_price = 6)
  (h_child_price : child_price = 4) (h_revenue : total_revenue = 1124) :
  ∃ (children : ℕ), children = 188 ∧ 
    (∃ (adults : ℕ), adults + children = total_seats ∧
      adult_price * adults + child_price * children = total_revenue) :=
by sorry

end NUMINAMATH_CALUDE_movie_theatre_attendance_l3781_378196


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3781_378112

theorem right_triangle_hypotenuse (PQ PR PS SQ PT TR QT SR : ℝ) :
  PS / SQ = 1 / 3 →
  PT / TR = 1 / 3 →
  QT = 20 →
  SR = 36 →
  PQ^2 + PR^2 = 1085.44 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3781_378112


namespace NUMINAMATH_CALUDE_combined_return_percentage_l3781_378104

theorem combined_return_percentage 
  (investment1 : ℝ) 
  (investment2 : ℝ) 
  (return1 : ℝ) 
  (return2 : ℝ) 
  (h1 : investment1 = 500)
  (h2 : investment2 = 1500)
  (h3 : return1 = 0.07)
  (h4 : return2 = 0.09) :
  (investment1 * return1 + investment2 * return2) / (investment1 + investment2) = 0.085 := by
sorry

end NUMINAMATH_CALUDE_combined_return_percentage_l3781_378104


namespace NUMINAMATH_CALUDE_sum_of_ten_consecutive_squares_not_perfect_square_l3781_378107

theorem sum_of_ten_consecutive_squares_not_perfect_square (x : ℤ) :
  ∃ (y : ℤ), 5 * (2 * x^2 + 10 * x + 29) ≠ y^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ten_consecutive_squares_not_perfect_square_l3781_378107


namespace NUMINAMATH_CALUDE_expression_value_l3781_378119

theorem expression_value : 
  let a : ℤ := 12
  let b : ℤ := 8
  let c : ℤ := 3
  ((a - b + c) - (a - (b + c))) = 6 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3781_378119


namespace NUMINAMATH_CALUDE_no_real_roots_for_equation_l3781_378175

theorem no_real_roots_for_equation : ¬∃ x : ℝ, x + Real.sqrt (2*x - 5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_for_equation_l3781_378175


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3781_378164

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity :
  ∀ (a : ℝ), a > 0 →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / 4 = 1) →
  (∀ (x y : ℝ), y^2 = 12*x) →
  (∃ (xf : ℝ), xf = 3 ∧ (∀ (y : ℝ), x^2 / a^2 - y^2 / 4 = 1 → (x - xf)^2 + y^2 = (3*a/5)^2)) →
  3 * Real.sqrt 5 / 5 = 3 / Real.sqrt (a^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3781_378164


namespace NUMINAMATH_CALUDE_convention_handshakes_specific_l3781_378158

/-- The number of handshakes in a convention with multiple companies --/
def convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating the number of handshakes for the specific convention described --/
theorem convention_handshakes_specific : convention_handshakes 5 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_specific_l3781_378158


namespace NUMINAMATH_CALUDE_square_area_rational_l3781_378113

theorem square_area_rational (s : ℚ) : ∃ (a : ℚ), a = s^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_rational_l3781_378113


namespace NUMINAMATH_CALUDE_square_difference_501_499_l3781_378125

theorem square_difference_501_499 : 501^2 - 499^2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_501_499_l3781_378125


namespace NUMINAMATH_CALUDE_competition_results_l3781_378105

/-- Represents the categories of safety questions -/
inductive Category
  | TrafficSafety
  | FireSafety
  | WaterSafety

/-- Represents the scoring system for the competition -/
structure ScoringSystem where
  correct_points : ℕ
  incorrect_points : ℕ

/-- Represents the correct rates for each category -/
def correct_rates : Category → ℚ
  | Category.TrafficSafety => 2/3
  | Category.FireSafety => 1/2
  | Category.WaterSafety => 1/3

/-- The scoring system used in the competition -/
def competition_scoring : ScoringSystem :=
  { correct_points := 5, incorrect_points := 1 }

/-- Calculates the probability of scoring at least 6 points for two questions -/
def prob_at_least_6_points (s : ScoringSystem) : ℚ :=
  let p_traffic := correct_rates Category.TrafficSafety
  let p_fire := correct_rates Category.FireSafety
  p_traffic * p_fire + p_traffic * (1 - p_fire) + (1 - p_traffic) * p_fire

/-- Calculates the expected value of the total score for three questions from different categories -/
def expected_score_three_questions (s : ScoringSystem) : ℚ :=
  let p_traffic := correct_rates Category.TrafficSafety
  let p_fire := correct_rates Category.FireSafety
  let p_water := correct_rates Category.WaterSafety
  let p_all_correct := p_traffic * p_fire * p_water
  let p_two_correct := p_traffic * p_fire * (1 - p_water) +
                       p_traffic * (1 - p_fire) * p_water +
                       (1 - p_traffic) * p_fire * p_water
  let p_one_correct := p_traffic * (1 - p_fire) * (1 - p_water) +
                       (1 - p_traffic) * p_fire * (1 - p_water) +
                       (1 - p_traffic) * (1 - p_fire) * p_water
  let p_all_incorrect := (1 - p_traffic) * (1 - p_fire) * (1 - p_water)
  3 * s.correct_points * p_all_correct +
  (2 * s.correct_points + s.incorrect_points) * p_two_correct +
  (s.correct_points + 2 * s.incorrect_points) * p_one_correct +
  3 * s.incorrect_points * p_all_incorrect

theorem competition_results :
  prob_at_least_6_points competition_scoring = 5/6 ∧
  expected_score_three_questions competition_scoring = 9 := by
  sorry

end NUMINAMATH_CALUDE_competition_results_l3781_378105


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l3781_378186

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) :
  x₁^2 - 3*x₁ + 2 = 0 → x₂^2 - 3*x₂ + 2 = 0 → x₁ * x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_l3781_378186


namespace NUMINAMATH_CALUDE_smallest_n_perfect_square_and_fifth_power_l3781_378172

theorem smallest_n_perfect_square_and_fifth_power : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^5) ∧
  (∀ (x : ℕ), x > 0 → 
    ((∃ (y : ℕ), 4 * x = y^2) ∧ (∃ (z : ℕ), 5 * x = z^5)) → 
    x ≥ 625) ∧
  n = 625 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_square_and_fifth_power_l3781_378172


namespace NUMINAMATH_CALUDE_b_invested_after_six_months_l3781_378123

/-- Represents the investment scenario and calculates when B invested -/
def calculate_b_investment_time (a_investment : ℕ) (b_investment : ℕ) (total_profit : ℕ) (a_profit : ℕ) : ℕ :=
  let a_time := 12
  let b_time := 12 - (a_investment * a_time * total_profit) / (a_profit * (a_investment + b_investment))
  b_time

/-- Theorem stating that B invested 6 months after A, given the problem conditions -/
theorem b_invested_after_six_months :
  calculate_b_investment_time 300 200 100 75 = 6 := by
  sorry

end NUMINAMATH_CALUDE_b_invested_after_six_months_l3781_378123


namespace NUMINAMATH_CALUDE_circle_center_l3781_378192

/-- A circle passes through (0,0) and is tangent to y = x^2 at (1,1). Its center is (-1, 2). -/
theorem circle_center (c : ℝ × ℝ) : 
  (∀ (x y : ℝ), (x - c.1)^2 + (y - c.2)^2 = (c.1 - 1)^2 + (c.2 - 1)^2) → -- circle equation
  ((0 : ℝ) - c.1)^2 + ((0 : ℝ) - c.2)^2 = (c.1 - 1)^2 + (c.2 - 1)^2 → -- (0,0) is on the circle
  (∀ (x : ℝ), x ≠ 1 → (x^2 - c.2) / (x - c.1) ≠ 2 * x) → -- circle is tangent to y = x^2 at (1,1)
  c = (-1, 2) :=
by sorry


end NUMINAMATH_CALUDE_circle_center_l3781_378192


namespace NUMINAMATH_CALUDE_power_inequality_l3781_378165

theorem power_inequality (a b m n : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0) (hn : n > 0) :
  a^(m+n) + b^(m+n) ≥ a^m * b^n + a^n * b^m := by sorry

end NUMINAMATH_CALUDE_power_inequality_l3781_378165


namespace NUMINAMATH_CALUDE_simplify_expression_l3781_378191

theorem simplify_expression : (27 * (10 ^ 9)) / (9 * (10 ^ 5)) = 30000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3781_378191


namespace NUMINAMATH_CALUDE_expression_evaluation_l3781_378188

theorem expression_evaluation : 
  |Real.sqrt 2 - Real.sqrt 3| + 2 * Real.cos (π / 4) - Real.sqrt 2 * Real.sqrt 6 = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3781_378188


namespace NUMINAMATH_CALUDE_prob_two_adjacent_is_one_fifth_l3781_378139

def num_knights : ℕ := 30
def num_selected : ℕ := 3

def prob_at_least_two_adjacent : ℚ :=
  1 - (num_knights * (num_knights - 3) * (num_knights - 4) - num_knights * 2 * (num_knights - 3)) / (num_knights.choose num_selected)

theorem prob_two_adjacent_is_one_fifth :
  prob_at_least_two_adjacent = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_adjacent_is_one_fifth_l3781_378139


namespace NUMINAMATH_CALUDE_discount_is_25_percent_l3781_378181

-- Define the cost of one photocopy
def cost_per_copy : ℚ := 2 / 100

-- Define the number of copies for each person
def copies_per_person : ℕ := 80

-- Define the total number of copies in the combined order
def total_copies : ℕ := 2 * copies_per_person

-- Define the savings per person
def savings_per_person : ℚ := 40 / 100

-- Define the total savings
def total_savings : ℚ := 2 * savings_per_person

-- Define the total cost without discount
def total_cost_without_discount : ℚ := total_copies * cost_per_copy

-- Define the total cost with discount
def total_cost_with_discount : ℚ := total_cost_without_discount - total_savings

-- Define the discount percentage
def discount_percentage : ℚ := (total_savings / total_cost_without_discount) * 100

-- Theorem statement
theorem discount_is_25_percent : discount_percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_discount_is_25_percent_l3781_378181


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3781_378143

theorem fixed_point_on_line (m : ℝ) : 
  (3 * m - 2) * (-3/4 : ℝ) - (m - 2) * (-13/4 : ℝ) - (m - 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3781_378143


namespace NUMINAMATH_CALUDE_print_350_pages_time_l3781_378138

/-- Calculates the time needed to print a given number of pages with a printer that has a specified
printing rate and pause interval. -/
def print_time (total_pages : ℕ) (pages_per_minute : ℕ) (pause_interval : ℕ) (pause_duration : ℕ) : ℕ :=
  let num_pauses := (total_pages / pause_interval) - 1
  let pause_time := num_pauses * pause_duration
  let print_time := (total_pages + pages_per_minute - 1) / pages_per_minute
  print_time + pause_time

/-- Theorem stating that printing 350 pages with the given printer specifications
takes approximately 27 minutes. -/
theorem print_350_pages_time :
  print_time 350 23 50 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_print_350_pages_time_l3781_378138


namespace NUMINAMATH_CALUDE_triangle_sin_c_l3781_378126

theorem triangle_sin_c (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a = 1 →
  b = Real.sqrt 2 →
  A + C = 2 * B →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  Real.sin C = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sin_c_l3781_378126


namespace NUMINAMATH_CALUDE_survey_change_bounds_l3781_378127

theorem survey_change_bounds (initial_yes initial_no final_yes final_no : ℚ) 
  (h1 : initial_yes = 1/2)
  (h2 : initial_no = 1/2)
  (h3 : final_yes = 7/10)
  (h4 : final_no = 3/10)
  (h5 : initial_yes + initial_no = 1)
  (h6 : final_yes + final_no = 1) :
  ∃ (x : ℚ), 1/5 ≤ x ∧ x ≤ 4/5 ∧ 
  (∃ (a b c d : ℚ), 
    a + c = initial_yes ∧
    b + d = initial_no ∧
    a + d = final_yes ∧
    b + c = final_no ∧
    c + d = x) :=
by sorry

end NUMINAMATH_CALUDE_survey_change_bounds_l3781_378127


namespace NUMINAMATH_CALUDE_function_equality_implies_zero_l3781_378163

/-- Given a function f(x, y) = kx + 1/y, prove that if f(a, b) = f(b, a) for a ≠ b, then f(ab, 1) = 0 -/
theorem function_equality_implies_zero (k : ℝ) (a b : ℝ) (h1 : a ≠ b) :
  (k * a + 1 / b = k * b + 1 / a) → (k * (a * b) + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_equality_implies_zero_l3781_378163


namespace NUMINAMATH_CALUDE_first_play_duration_is_20_l3781_378160

/-- Represents the duration of a soccer game in minutes -/
def game_duration : ℕ := 90

/-- Represents the duration of the second part of play in minutes -/
def second_play_duration : ℕ := 35

/-- Represents the duration of sideline time in minutes -/
def sideline_duration : ℕ := 35

/-- Calculates the duration of the first part of play given the total game duration,
    second part play duration, and sideline duration -/
def first_play_duration (total : ℕ) (second : ℕ) (sideline : ℕ) : ℕ :=
  total - second - sideline

theorem first_play_duration_is_20 :
  first_play_duration game_duration second_play_duration sideline_duration = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_play_duration_is_20_l3781_378160


namespace NUMINAMATH_CALUDE_value_of_a_l3781_378151

def A (a : ℝ) : Set ℝ := {-1, 0, a}
def B (a : ℝ) : Set ℝ := {0, Real.sqrt a}

theorem value_of_a (a : ℝ) (h : B a ⊆ A a) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3781_378151


namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l3781_378176

theorem quadratic_rational_root_even_coefficient
  (a b c : ℤ) (h_a : a ≠ 0)
  (h_root : ∃ (x : ℚ), a * x^2 + b * x + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l3781_378176


namespace NUMINAMATH_CALUDE_park_hikers_l3781_378174

theorem park_hikers (total : ℕ) (difference : ℕ) (hikers : ℕ) (bikers : ℕ) : 
  total = 676 → 
  difference = 178 → 
  total = hikers + bikers → 
  hikers = bikers + difference → 
  hikers = 427 := by
sorry

end NUMINAMATH_CALUDE_park_hikers_l3781_378174


namespace NUMINAMATH_CALUDE_largest_number_l3781_378159

theorem largest_number (a b c d e : ℝ) : 
  a = 0.9891 → b = 0.9799 → c = 0.989 → d = 0.978 → e = 0.979 →
  (a ≥ b ∧ a ≥ c ∧ a ≥ d ∧ a ≥ e) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3781_378159


namespace NUMINAMATH_CALUDE_equation_solution_pairs_l3781_378137

theorem equation_solution_pairs : 
  {(p, q) : ℕ × ℕ | (p + 1)^(p - 1) + (p - 1)^(p + 1) = q^q} = {(1, 1), (2, 2)} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_pairs_l3781_378137


namespace NUMINAMATH_CALUDE_fresh_grapes_weight_calculation_l3781_378131

/-- The weight of dried grapes in kilograms -/
def dried_grapes_weight : ℝ := 66.67

/-- The fraction of water in fresh grapes by weight -/
def fresh_water_fraction : ℝ := 0.75

/-- The fraction of water in dried grapes by weight -/
def dried_water_fraction : ℝ := 0.25

/-- The weight of fresh grapes in kilograms -/
def fresh_grapes_weight : ℝ := 200.01

theorem fresh_grapes_weight_calculation :
  fresh_grapes_weight = dried_grapes_weight * (1 - dried_water_fraction) / (1 - fresh_water_fraction) :=
by sorry

end NUMINAMATH_CALUDE_fresh_grapes_weight_calculation_l3781_378131


namespace NUMINAMATH_CALUDE_circle_area_with_perimeter_of_square_l3781_378146

theorem circle_area_with_perimeter_of_square (π : ℝ) (h_pi : π > 0) : 
  let square_area : ℝ := 121
  let square_side : ℝ := Real.sqrt square_area
  let square_perimeter : ℝ := 4 * square_side
  let circle_radius : ℝ := square_perimeter / (2 * π)
  let circle_area : ℝ := π * circle_radius^2
  circle_area = 484 / π := by
sorry

end NUMINAMATH_CALUDE_circle_area_with_perimeter_of_square_l3781_378146


namespace NUMINAMATH_CALUDE_same_color_combination_probability_l3781_378157

def total_candies : ℕ := 20
def red_candies : ℕ := 12
def blue_candies : ℕ := 8

def lucy_picks : ℕ := 2
def john_picks : ℕ := 2

theorem same_color_combination_probability :
  let probability_same_combination := (2 * (Nat.choose red_candies 2 * Nat.choose (red_candies - 2) 2 +
                                            Nat.choose blue_candies 2 * Nat.choose (blue_candies - 2) 2) +
                                       Nat.choose red_candies 2 * Nat.choose blue_candies 2 +
                                       Nat.choose blue_candies 2 * Nat.choose red_candies 2) /
                                      (Nat.choose total_candies 2 * Nat.choose (total_candies - 2) 2)
  probability_same_combination = 184 / 323 := by
  sorry

end NUMINAMATH_CALUDE_same_color_combination_probability_l3781_378157


namespace NUMINAMATH_CALUDE_vote_count_l3781_378118

theorem vote_count (U A B : Finset Nat) (h1 : Finset.card U = 232)
  (h2 : Finset.card A = 172) (h3 : Finset.card B = 143)
  (h4 : Finset.card (U \ (A ∪ B)) = 37) :
  Finset.card (A ∩ B) = 120 := by
  sorry

end NUMINAMATH_CALUDE_vote_count_l3781_378118


namespace NUMINAMATH_CALUDE_range_of_f_l3781_378173

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x)
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (2 * x + φ) + 1

theorem range_of_f (ω : ℝ) (h_ω : ω > 0) (φ : ℝ) 
  (h_symmetry : ∀ x : ℝ, ∃ c : ℝ, f ω (c - x) = f ω (c + x) ∧ g φ (c - x) = g φ (c + x)) :
  Set.range (f ω) = Set.Icc (-3) 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l3781_378173


namespace NUMINAMATH_CALUDE_parabolas_intersection_l3781_378141

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 2 * x^2 - 10 * x - 10
def parabola2 (x : ℝ) : ℝ := x^2 - 4 * x + 6

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(-2, 18), (8, 38)}

-- Theorem statement
theorem parabolas_intersection :
  ∀ x y : ℝ, parabola1 x = parabola2 x ∧ y = parabola1 x ↔ (x, y) ∈ intersection_points :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l3781_378141


namespace NUMINAMATH_CALUDE_farm_animals_l3781_378135

theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 8) 
  (h2 : total_legs = 24) : 
  ∃ (ducks dogs : ℕ), 
    ducks + dogs = total_animals ∧ 
    2 * ducks + 4 * dogs = total_legs ∧ 
    ducks = 4 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l3781_378135


namespace NUMINAMATH_CALUDE_white_mice_count_l3781_378120

theorem white_mice_count (total : ℕ) (white : ℕ) (brown : ℕ) : 
  (white = 2 * total / 3) →  -- 2/3 of the mice are white
  (brown = 7) →              -- There are 7 brown mice
  (total = white + brown) →  -- Total mice is the sum of white and brown mice
  (white > 0) →              -- There are some white mice
  (white = 14) :=            -- The number of white mice is 14
by
  sorry

end NUMINAMATH_CALUDE_white_mice_count_l3781_378120


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_six_l3781_378170

theorem smallest_digit_divisible_by_six : 
  ∃ (N : ℕ), N < 10 ∧ (1453 * 10 + N) % 6 = 0 ∧ 
  ∀ (M : ℕ), M < N → M < 10 → (1453 * 10 + M) % 6 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_six_l3781_378170


namespace NUMINAMATH_CALUDE_unique_solution_linear_equation_l3781_378190

theorem unique_solution_linear_equation (a b c : ℝ) (h1 : c ≠ 0) (h2 : b ≠ 2) :
  ∃! x : ℝ, 4 * x - 7 + a = 2 * b * x + c ∧ x = (c + 7 - a) / (4 - 2 * b) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_linear_equation_l3781_378190


namespace NUMINAMATH_CALUDE_custom_op_seven_three_l3781_378198

-- Define the custom operation
def custom_op (a b : ℤ) : ℤ := 4*a + 5*b - a*b

-- Theorem statement
theorem custom_op_seven_three :
  custom_op 7 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_seven_three_l3781_378198


namespace NUMINAMATH_CALUDE_total_amount_divided_l3781_378140

/-- The total amount divided among A, B, and C is 3366.00000000000006 given the conditions. -/
theorem total_amount_divided (a b c : ℝ) 
  (h1 : a = (2/3) * b)
  (h2 : b = (1/4) * c)
  (h3 : a = 396.00000000000006) : 
  a + b + c = 3366.00000000000006 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_divided_l3781_378140


namespace NUMINAMATH_CALUDE_factor_calculation_l3781_378100

theorem factor_calculation (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 18 → 
  factor * (2 * initial_number + 5) = 123 → 
  factor = 3 := by sorry

end NUMINAMATH_CALUDE_factor_calculation_l3781_378100


namespace NUMINAMATH_CALUDE_min_value_abc_min_value_abc_attainable_l3781_378102

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + 2*b + 3*c = a*b*c) : 
  a*b*c ≥ 9*Real.sqrt 2 := by
sorry

theorem min_value_abc_attainable : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + 2*b + 3*c = a*b*c ∧ a*b*c = 9*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_abc_min_value_abc_attainable_l3781_378102


namespace NUMINAMATH_CALUDE_class_size_l3781_378134

theorem class_size (n : ℕ) 
  (h1 : n < 50) 
  (h2 : n % 8 = 5) 
  (h3 : n % 6 = 4) : 
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_class_size_l3781_378134


namespace NUMINAMATH_CALUDE_other_sides_equations_l3781_378153

/-- An isosceles right triangle with one leg on the line 2x - y = 0 and hypotenuse midpoint (4, 2) -/
structure IsoscelesRightTriangle where
  /-- The line containing one leg of the triangle -/
  leg_line : Set (ℝ × ℝ)
  /-- The midpoint of the hypotenuse -/
  hypotenuse_midpoint : ℝ × ℝ
  /-- The triangle is isosceles and right-angled -/
  is_isosceles_right : Bool
  /-- The leg line equation is 2x - y = 0 -/
  leg_line_eq : leg_line = {(x, y) : ℝ × ℝ | 2 * x - y = 0}
  /-- The hypotenuse midpoint is (4, 2) -/
  midpoint_coords : hypotenuse_midpoint = (4, 2)

/-- The theorem stating the equations of the other two sides -/
theorem other_sides_equations (t : IsoscelesRightTriangle) :
  ∃ (side1 side2 : Set (ℝ × ℝ)),
    side1 = {(x, y) : ℝ × ℝ | x + 2 * y - 2 = 0} ∧
    side2 = {(x, y) : ℝ × ℝ | x + 2 * y - 14 = 0} :=
  sorry

end NUMINAMATH_CALUDE_other_sides_equations_l3781_378153


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3781_378184

theorem solution_set_of_inequality (x : ℝ) : 
  (x - 3) / (x + 2) < 0 ↔ -2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3781_378184


namespace NUMINAMATH_CALUDE_tan_105_degrees_l3781_378133

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l3781_378133


namespace NUMINAMATH_CALUDE_prism_volume_l3781_378171

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines the properties of our specific rectangular prism -/
def RectangularPrism (d : PrismDimensions) : Prop :=
  d.x * d.y = 18 ∧ 
  d.y * d.z = 12 ∧ 
  d.x * d.z = 8 ∧
  d.y = 2 * min d.x d.z

theorem prism_volume (d : PrismDimensions) 
  (h : RectangularPrism d) : d.x * d.y * d.z = 16 := by
  sorry

#check prism_volume

end NUMINAMATH_CALUDE_prism_volume_l3781_378171


namespace NUMINAMATH_CALUDE_fifteen_machines_six_minutes_l3781_378193

/-- The number of paperclips produced by a given number of machines in a given time -/
def paperclips_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  let base_machines := 8
  let base_production := 560
  let production_per_machine := base_production / base_machines
  machines * production_per_machine * minutes

/-- Theorem stating that 15 machines will produce 6300 paperclips in 6 minutes -/
theorem fifteen_machines_six_minutes :
  paperclips_produced 15 6 = 6300 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_machines_six_minutes_l3781_378193


namespace NUMINAMATH_CALUDE_sugar_cube_weight_l3781_378144

/-- The weight of sugar cubes in the first group -/
def weight_first_group : ℝ := 10

/-- The number of ants in the first group -/
def ants_first : ℕ := 15

/-- The number of sugar cubes moved by the first group -/
def cubes_first : ℕ := 600

/-- The time taken by the first group (in hours) -/
def time_first : ℝ := 5

/-- The number of ants in the second group -/
def ants_second : ℕ := 20

/-- The number of sugar cubes moved by the second group -/
def cubes_second : ℕ := 960

/-- The time taken by the second group (in hours) -/
def time_second : ℝ := 3

/-- The weight of sugar cubes in the second group -/
def weight_second : ℝ := 5

theorem sugar_cube_weight :
  (ants_first : ℝ) * cubes_second * time_first * weight_second =
  (ants_second : ℝ) * cubes_first * time_second * weight_first_group :=
by sorry

end NUMINAMATH_CALUDE_sugar_cube_weight_l3781_378144


namespace NUMINAMATH_CALUDE_exists_rational_less_than_neg_half_l3781_378109

theorem exists_rational_less_than_neg_half : ∃ q : ℚ, q < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_exists_rational_less_than_neg_half_l3781_378109


namespace NUMINAMATH_CALUDE_fraction_power_product_l3781_378108

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l3781_378108
