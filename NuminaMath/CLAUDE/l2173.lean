import Mathlib

namespace twelve_lines_formed_l2173_217301

/-- A configuration of points in a plane -/
structure PointConfiguration where
  total_points : ℕ
  collinear_points : ℕ
  noncollinear_points : ℕ
  h_total : total_points = collinear_points + noncollinear_points
  h_collinear : collinear_points ≥ 3
  h_noncollinear : noncollinear_points ≥ 0

/-- The number of lines formed by a given point configuration -/
def num_lines (config : PointConfiguration) : ℕ :=
  1 + config.collinear_points * config.noncollinear_points + 
  (config.noncollinear_points * (config.noncollinear_points - 1)) / 2

/-- Theorem: In the given configuration, 12 lines can be formed -/
theorem twelve_lines_formed (config : PointConfiguration) 
  (h1 : config.total_points = 7)
  (h2 : config.collinear_points = 5)
  (h3 : config.noncollinear_points = 2) :
  num_lines config = 12 := by
  sorry

end twelve_lines_formed_l2173_217301


namespace subset_coloring_existence_l2173_217372

theorem subset_coloring_existence (S : Type) [Fintype S] (h : Fintype.card S = 2002) (N : ℕ) (hN : N ≤ 2^2002) :
  ∃ f : Set S → Bool,
    (∀ A B : Set S, f A = true → f B = true → f (A ∪ B) = true) ∧
    (∀ A B : Set S, f A = false → f B = false → f (A ∪ B) = false) ∧
    (Fintype.card {A : Set S | f A = true} = N) :=
by sorry

end subset_coloring_existence_l2173_217372


namespace cos_48_degrees_l2173_217318

theorem cos_48_degrees : Real.cos (48 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_48_degrees_l2173_217318


namespace quinn_reading_rate_l2173_217394

/-- A reading challenge that lasts for a certain number of weeks -/
structure ReadingChallenge where
  duration : ℕ  -- Duration of the challenge in weeks
  books_per_coupon : ℕ  -- Number of books required for one coupon

/-- A participant in the reading challenge -/
structure Participant where
  challenge : ReadingChallenge
  coupons_earned : ℕ  -- Number of coupons earned

def books_per_week (p : Participant) : ℚ :=
  (p.coupons_earned * p.challenge.books_per_coupon : ℚ) / p.challenge.duration

theorem quinn_reading_rate (c : ReadingChallenge) (p : Participant) :
    c.duration = 10 ∧ c.books_per_coupon = 5 ∧ p.challenge = c ∧ p.coupons_earned = 4 →
    books_per_week p = 2 := by
  sorry

end quinn_reading_rate_l2173_217394


namespace squirrel_travel_time_l2173_217327

/-- Proves that a squirrel traveling at 6 miles per hour takes 30 minutes to travel 3 miles -/
theorem squirrel_travel_time :
  let speed : ℝ := 6  -- Speed in miles per hour
  let distance : ℝ := 3  -- Distance in miles
  let time_hours : ℝ := distance / speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 30 := by sorry

end squirrel_travel_time_l2173_217327


namespace exam_scores_l2173_217336

theorem exam_scores (full_marks : ℝ) (a b c d : ℝ) : 
  full_marks = 500 →
  a = b * 0.9 →
  b = c * 1.25 →
  c = d * 0.8 →
  a = 360 →
  d / full_marks = 0.8 :=
by sorry

end exam_scores_l2173_217336


namespace find_a_solve_inequality_l2173_217302

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1|

-- Theorem 1: Prove the value of a
theorem find_a : 
  ∀ a : ℝ, (∀ x : ℝ, f a x ≤ 2 ↔ -1/2 ≤ x ∧ x ≤ 3/2) → a = 2 :=
by sorry

-- Theorem 2: Prove the solution set of the inequality
theorem solve_inequality :
  ∀ x : ℝ, f 2 x + f 2 (x/2 - 1) ≥ 5 ↔ x ≥ 3 ∨ x ≤ -1/3 :=
by sorry

end find_a_solve_inequality_l2173_217302


namespace trig_inequality_implies_range_l2173_217303

open Real

theorem trig_inequality_implies_range (θ : ℝ) :
  θ ∈ Set.Icc 0 (2 * π) →
  (cos θ)^5 - (sin θ)^5 < 7 * ((sin θ)^3 - (cos θ)^3) →
  θ ∈ Set.Ioo (π / 4) (5 * π / 4) :=
by
  sorry

end trig_inequality_implies_range_l2173_217303


namespace x_over_y_equals_one_l2173_217347

-- Define a function that represents the nested absolute value expression
def nestedAbs (x y : ℝ) : ℕ → ℝ
  | 0 => x
  | n + 1 => |nestedAbs y x n - x|

-- State the theorem
theorem x_over_y_equals_one
  (x y : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (h : nestedAbs x y 2019 = nestedAbs y x 2019) :
  x / y = 1 :=
sorry

end x_over_y_equals_one_l2173_217347


namespace prob_two_sixes_is_one_thirty_sixth_l2173_217362

/-- A fair six-sided die -/
structure FairDie :=
  (sides : Fin 6)

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll (d : FairDie) (n : Fin 6) : ℚ := 1 / 6

/-- The probability of rolling two consecutive sixes -/
def prob_two_sixes (d : FairDie) : ℚ :=
  (prob_single_roll d 5) * (prob_single_roll d 5)

/-- Theorem: The probability of rolling two consecutive sixes with a fair six-sided die is 1/36 -/
theorem prob_two_sixes_is_one_thirty_sixth (d : FairDie) :
  prob_two_sixes d = 1 / 36 := by
  sorry

end prob_two_sixes_is_one_thirty_sixth_l2173_217362


namespace combined_new_weight_theorem_l2173_217389

/-- Calculates the new weight of fruit after water loss -/
def new_weight (initial_weight : ℝ) (initial_water_percent : ℝ) (evaporation_loss : ℝ) (skin_loss : ℝ) : ℝ :=
  let initial_water := initial_weight * initial_water_percent
  let pulp := initial_weight - initial_water
  let water_loss := initial_water * (evaporation_loss + skin_loss)
  let new_water := initial_water - water_loss
  pulp + new_water

/-- The combined new weight of oranges and apples after water loss -/
theorem combined_new_weight_theorem (orange_weight : ℝ) (apple_weight : ℝ) 
    (orange_water_percent : ℝ) (apple_water_percent : ℝ)
    (orange_evaporation_loss : ℝ) (orange_skin_loss : ℝ)
    (apple_evaporation_loss : ℝ) (apple_skin_loss : ℝ) :
  orange_weight = 5 →
  apple_weight = 3 →
  orange_water_percent = 0.95 →
  apple_water_percent = 0.90 →
  orange_evaporation_loss = 0.05 →
  orange_skin_loss = 0.02 →
  apple_evaporation_loss = 0.03 →
  apple_skin_loss = 0.01 →
  (new_weight orange_weight orange_water_percent orange_evaporation_loss orange_skin_loss +
   new_weight apple_weight apple_water_percent apple_evaporation_loss apple_skin_loss) = 7.5595 := by
  sorry

end combined_new_weight_theorem_l2173_217389


namespace yogurt_price_is_2_5_l2173_217367

/-- The price of a pack of yogurt in yuan -/
def yogurt_price : ℝ := 2.5

/-- The price of a pack of fresh milk in yuan -/
def milk_price : ℝ := 1

/-- The total cost of 4 packs of yogurt and 4 packs of fresh milk is 14 yuan -/
axiom first_purchase : 4 * yogurt_price + 4 * milk_price = 14

/-- The total cost of 2 packs of yogurt and 8 packs of fresh milk is 13 yuan -/
axiom second_purchase : 2 * yogurt_price + 8 * milk_price = 13

/-- The price of each pack of yogurt is 2.5 yuan -/
theorem yogurt_price_is_2_5 : yogurt_price = 2.5 := by
  sorry

end yogurt_price_is_2_5_l2173_217367


namespace correct_product_l2173_217365

/-- Given two positive integers a and b, where a is a two-digit number,
    if the product of the reversed digits of a and b is 161,
    then the product of a and b is 224. -/
theorem correct_product (a b : ℕ) : 
  a ≥ 10 ∧ a ≤ 99 →  -- a is a two-digit number
  b > 0 →  -- b is positive
  (10 * (a % 10) + a / 10) * b = 161 →  -- reversed a * b = 161
  a * b = 224 :=
by sorry

end correct_product_l2173_217365


namespace polynomial_division_remainder_l2173_217304

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 + 2*x^3 = (x^2 + 7*x + 2) * q + (33*x^2 + 10*x) := by
  sorry

end polynomial_division_remainder_l2173_217304


namespace sin_addition_equality_l2173_217334

theorem sin_addition_equality (y : Real) : 
  (y ∈ Set.Icc 0 (Real.pi / 2)) → 
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.sin (x + y) = Real.sin x + Real.sin y) → 
  y = 0 :=
by sorry

end sin_addition_equality_l2173_217334


namespace jesses_room_difference_l2173_217314

/-- Jesse's room dimensions and length-width difference --/
theorem jesses_room_difference :
  ∀ (length width : ℝ),
  length = 20 →
  width = 19 →
  length - width = 1 :=
by
  sorry

end jesses_room_difference_l2173_217314


namespace unique_solution_for_k_squared_minus_2016_equals_3_to_n_l2173_217310

theorem unique_solution_for_k_squared_minus_2016_equals_3_to_n :
  ∃! (k n : ℕ), k > 0 ∧ n > 0 ∧ k^2 - 2016 = 3^n :=
by
  -- The proof goes here
  sorry

end unique_solution_for_k_squared_minus_2016_equals_3_to_n_l2173_217310


namespace thirteen_binary_l2173_217323

/-- Converts a natural number to its binary representation as a list of booleans -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Checks if a list of booleans represents a given natural number in binary -/
def is_binary_rep (n : ℕ) (l : List Bool) : Prop :=
  to_binary n = l.reverse

theorem thirteen_binary :
  is_binary_rep 13 [true, false, true, true] := by sorry

end thirteen_binary_l2173_217323


namespace largest_valid_number_is_valid_853_largest_valid_number_is_853_l2173_217341

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- Three-digit integer
  (n / 100 = 8) ∧  -- Starts with 8
  (∀ d, d ≠ 0 → d ∣ n → n % d = 0) ∧  -- Divisible by each non-zero digit
  (n % (n / 100 + (n / 10) % 10 + n % 10) = 0)  -- Divisible by sum of digits

theorem largest_valid_number :
  ∀ m, is_valid_number m → m ≤ 853 :=
by sorry

theorem is_valid_853 : is_valid_number 853 :=
by sorry

theorem largest_valid_number_is_853 :
  ∀ n, is_valid_number n ∧ n ≠ 853 → n < 853 :=
by sorry

end largest_valid_number_is_valid_853_largest_valid_number_is_853_l2173_217341


namespace quadratic_function_properties_l2173_217360

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h_no_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) :=
sorry

end quadratic_function_properties_l2173_217360


namespace rectangular_field_area_l2173_217396

theorem rectangular_field_area (width length perimeter area : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  area = width * length →
  area = 243 := by
  sorry

end rectangular_field_area_l2173_217396


namespace half_of_third_of_sixth_of_90_l2173_217325

theorem half_of_third_of_sixth_of_90 : (1 / 2 : ℚ) * (1 / 3) * (1 / 6) * 90 = 5 / 2 := by
  sorry

end half_of_third_of_sixth_of_90_l2173_217325


namespace bug_meeting_point_l2173_217326

/-- Represents a triangle with side lengths -/
structure Triangle where
  pq : ℝ
  qr : ℝ
  pr : ℝ

/-- Represents a bug crawling on the triangle -/
structure Bug where
  speed : ℝ
  clockwise : Bool

/-- The meeting point of two bugs on a triangle -/
def meetingPoint (t : Triangle) (b1 b2 : Bug) : ℝ := sorry

theorem bug_meeting_point (t : Triangle) (b1 b2 : Bug) :
  t.pq = 8 ∧ t.qr = 10 ∧ t.pr = 12 ∧
  b1.speed = 2 ∧ b1.clockwise = true ∧
  b2.speed = 3 ∧ b2.clockwise = false →
  t.qr - meetingPoint t b1 b2 = 6 := by sorry

end bug_meeting_point_l2173_217326


namespace shopping_tax_calculation_l2173_217385

-- Define the percentages of spending
def clothing_percent : ℝ := 0.50
def food_percent : ℝ := 0.20
def other_percent : ℝ := 0.30

-- Define the tax rates
def clothing_tax_rate : ℝ := 0.04
def food_tax_rate : ℝ := 0
def total_tax_rate : ℝ := 0.044

-- Define the unknown tax rate on other items
def other_tax_rate : ℝ := sorry

theorem shopping_tax_calculation :
  let total_spent := 100  -- Assume total spent is 100 for simplicity
  let clothing_tax := clothing_percent * total_spent * clothing_tax_rate
  let other_tax := other_percent * total_spent * other_tax_rate
  clothing_tax + other_tax = total_tax_rate * total_spent →
  other_tax_rate = 0.08 := by sorry

end shopping_tax_calculation_l2173_217385


namespace equation_solution_l2173_217306

theorem equation_solution : 
  ∃! x : ℝ, (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ x = -9 := by
  sorry

end equation_solution_l2173_217306


namespace sum_of_digits_in_period_of_one_over_98_squared_l2173_217375

/-- The sum of all digits in one period of the repeating decimal expansion of 1/(98^2) -/
def sum_of_digits_in_period (n : ℕ) : ℕ :=
  sorry

/-- The period length of the repeating decimal expansion of 1/(98^2) -/
def period_length : ℕ := 196

/-- Theorem: The sum of all digits in one period of the repeating decimal expansion of 1/(98^2) is 882 -/
theorem sum_of_digits_in_period_of_one_over_98_squared :
  sum_of_digits_in_period period_length = 882 := by
  sorry

end sum_of_digits_in_period_of_one_over_98_squared_l2173_217375


namespace swimmer_speed_l2173_217308

/-- The speed of a swimmer in still water, given downstream and upstream distances and times. -/
theorem swimmer_speed (downstream_distance upstream_distance : ℝ) 
  (downstream_time upstream_time : ℝ) (h1 : downstream_distance = 36)
  (h2 : upstream_distance = 26) (h3 : downstream_time = 2) (h4 : upstream_time = 2) :
  ∃ (speed_still : ℝ), speed_still = 15.5 ∧ 
  downstream_distance / downstream_time = speed_still + (downstream_distance - upstream_distance) / (downstream_time + upstream_time) ∧
  upstream_distance / upstream_time = speed_still - (downstream_distance - upstream_distance) / (downstream_time + upstream_time) :=
by sorry

end swimmer_speed_l2173_217308


namespace log_five_negative_one_l2173_217319

theorem log_five_negative_one (x : ℝ) (h1 : x > 0) (h2 : Real.log x / Real.log 5 = -1) : x = 0.2 := by
  sorry

end log_five_negative_one_l2173_217319


namespace octagon_arc_length_l2173_217397

/-- The length of the arc intercepted by one side of a regular octagon inscribed in a circle -/
theorem octagon_arc_length (s : ℝ) (h : s = 4) :
  let R := s / (2 * Real.sin (π / 8))
  let C := 2 * π * R
  C / 8 = (Real.sqrt 2 * π) / 2 := by sorry

end octagon_arc_length_l2173_217397


namespace oil_to_add_l2173_217340

/-- The amount of oil Scarlett needs to add to her measuring cup -/
theorem oil_to_add (current : ℚ) (desired : ℚ) : 
  current = 0.16666666666666666 →
  desired = 0.8333333333333334 →
  desired - current = 0.6666666666666667 := by
  sorry

end oil_to_add_l2173_217340


namespace arithmetic_sequence_a7_l2173_217316

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℚ) 
    (h_arith : is_arithmetic_sequence a)
    (h_a3 : a 3 = 2)
    (h_a5 : a 5 = 7) : 
  a 7 = 12 := by
sorry

end arithmetic_sequence_a7_l2173_217316


namespace square_root_and_square_operations_l2173_217363

theorem square_root_and_square_operations : 
  (∃ (x : ℝ), x ^ 2 = 4 ∧ x = 2) ∧ 
  (∀ (a : ℝ), (-3 * a) ^ 2 = 9 * a ^ 2) := by
  sorry

end square_root_and_square_operations_l2173_217363


namespace problem_solution_l2173_217380

-- Define the sets A and B
def A (a b c : ℝ) : Prop := a^2 - b*c - 8*a + 7 = 0
def B (a b c : ℝ) : Prop := b^2 + c^2 + b*c - b*a + b = 0

-- Define the function y
def y (a b c : ℝ) : ℝ := a*b + b*c + a*c

-- Theorem statement
theorem problem_solution :
  ∃ (a b c : ℝ), A a b c ∧ B a b c →
  (∀ a : ℝ, (∃ b c : ℝ, A a b c ∧ B a b c) → 1 ≤ a ∧ a ≤ 9) ∧
  (∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, 
    A a₁ b₁ c₁ ∧ B a₁ b₁ c₁ ∧ A a₂ b₂ c₂ ∧ B a₂ b₂ c₂ ∧
    y a₁ b₁ c₁ = 88 ∧ y a₂ b₂ c₂ = -56 ∧
    ∀ a b c : ℝ, A a b c ∧ B a b c → -56 ≤ y a b c ∧ y a b c ≤ 88) :=
by
  sorry

end problem_solution_l2173_217380


namespace total_apples_picked_l2173_217333

-- Define the number of apples picked by each person
def benny_apples : ℕ := 2 * 4
def dan_apples : ℕ := 9 * 5
def sarah_apples : ℕ := (dan_apples + 1) / 2  -- Rounding up
def lisa_apples : ℕ := ((3 * (benny_apples + dan_apples) + 4) / 5)  -- Rounding up

-- Theorem to prove
theorem total_apples_picked : 
  benny_apples + dan_apples + sarah_apples + lisa_apples = 108 := by
  sorry


end total_apples_picked_l2173_217333


namespace range_of_a_l2173_217312

/-- A linear function y = (2a-3)x + a + 2 that is above the x-axis for -2 ≤ x ≤ 1 -/
def LinearFunction (a : ℝ) (x : ℝ) : ℝ := (2*a - 3)*x + a + 2

/-- The function is above the x-axis for -2 ≤ x ≤ 1 -/
def AboveXAxis (a : ℝ) : Prop :=
  ∀ x, -2 ≤ x ∧ x ≤ 1 → LinearFunction a x > 0

theorem range_of_a (a : ℝ) (h : AboveXAxis a) :
  1/3 < a ∧ a < 8/3 ∧ a ≠ 3/2 :=
sorry

end range_of_a_l2173_217312


namespace square_sum_value_l2173_217364

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 10) :
  x^2 + y^2 = 29 := by
sorry

end square_sum_value_l2173_217364


namespace periodic_function_extension_l2173_217398

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function_extension
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 2)
  (h_smallest_period : ∀ p, 0 < p → p < 2 → ¬ is_periodic f p)
  (h_def : ∀ x, 0 ≤ x → x < 2 → f x = x^3 - x) :
  ∀ x, -2 ≤ x → x < 0 → f x = x^3 + 6*x^2 + 11*x + 6 :=
sorry

end periodic_function_extension_l2173_217398


namespace triangle_perimeter_max_l2173_217379

open Real

theorem triangle_perimeter_max (x : ℝ) (h : 0 < x ∧ x < 2 * π / 3) :
  let y := 4 * Real.sqrt 3 * sin (x + π / 6) + 2 * Real.sqrt 3
  ∃ (y_max : ℝ), y ≤ y_max ∧ y_max = 6 * Real.sqrt 3 := by
  sorry

#check triangle_perimeter_max

end triangle_perimeter_max_l2173_217379


namespace river_current_speed_l2173_217320

/-- Proves that given a ship with a maximum speed of 20 km/h in still water,
    if it takes the same time to travel 100 km downstream as it does to travel 60 km upstream,
    then the speed of the river current is 5 km/h. -/
theorem river_current_speed :
  let ship_speed : ℝ := 20
  let downstream_distance : ℝ := 100
  let upstream_distance : ℝ := 60
  ∀ current_speed : ℝ,
    (downstream_distance / (ship_speed + current_speed) = upstream_distance / (ship_speed - current_speed)) →
    current_speed = 5 := by
  sorry

end river_current_speed_l2173_217320


namespace chad_earnings_problem_l2173_217392

/-- Chad's earnings and savings problem -/
theorem chad_earnings_problem (mowing_earnings : ℝ) : 
  (mowing_earnings + 250 + 150 + 150) * 0.4 = 460 → mowing_earnings = 600 := by
  sorry

end chad_earnings_problem_l2173_217392


namespace student_grade_average_l2173_217386

theorem student_grade_average (grade1 grade2 : ℝ) 
  (h1 : grade1 = 70)
  (h2 : grade2 = 80) : 
  ∃ (grade3 : ℝ), (grade1 + grade2 + grade3) / 3 = grade3 ∧ grade3 = 75 := by
sorry

end student_grade_average_l2173_217386


namespace computer_pricing_l2173_217349

theorem computer_pricing (selling_price_40 : ℝ) (profit_percentage_40 : ℝ) 
  (selling_price_50 : ℝ) (profit_percentage_50 : ℝ) :
  selling_price_40 = 2240 ∧ 
  profit_percentage_40 = 0.4 ∧ 
  selling_price_50 = 2400 ∧ 
  profit_percentage_50 = 0.5 →
  let cost := selling_price_40 / (1 + profit_percentage_40)
  selling_price_50 = cost * (1 + profit_percentage_50) := by
  sorry


end computer_pricing_l2173_217349


namespace solve_congruence_l2173_217378

theorem solve_congruence :
  ∃ n : ℕ, 0 ≤ n ∧ n < 43 ∧ (11 * n) % 43 = 7 % 43 ∧ n = 28 := by
  sorry

end solve_congruence_l2173_217378


namespace bakery_pie_division_l2173_217351

theorem bakery_pie_division (pie_leftover : ℚ) (num_people : ℕ) : 
  pie_leftover = 8 / 9 → num_people = 3 → 
  pie_leftover / num_people = 8 / 27 := by
  sorry

end bakery_pie_division_l2173_217351


namespace xyz_inequality_l2173_217387

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2*x*y*z = 1) :
  8*x*y*z ≤ 1 ∧
  (8*x*y*z = 1 ↔
    (x, y, z) = (1/2, 1/2, 1/2) ∨
    (x, y, z) = (-1/2, -1/2, 1/2) ∨
    (x, y, z) = (-1/2, 1/2, -1/2) ∨
    (x, y, z) = (1/2, -1/2, -1/2)) :=
by sorry

end xyz_inequality_l2173_217387


namespace regular_polygon_interior_angle_sum_l2173_217383

theorem regular_polygon_interior_angle_sum :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 0 →
    exterior_angle = 20 →
    360 / exterior_angle = n →
    (n - 2) * 180 = 2880 :=
by
  sorry

end regular_polygon_interior_angle_sum_l2173_217383


namespace half_merit_scholarship_percentage_l2173_217322

/-- Given a group of senior students, prove the percentage who received
    a half merit scholarship. -/
theorem half_merit_scholarship_percentage
  (total_students : ℕ)
  (full_scholarship_percentage : ℚ)
  (no_scholarship_count : ℕ)
  (h1 : total_students = 300)
  (h2 : full_scholarship_percentage = 5 / 100)
  (h3 : no_scholarship_count = 255) :
  (total_students - no_scholarship_count - 
   (full_scholarship_percentage * total_students).num) / total_students = 1 / 10 := by
  sorry

end half_merit_scholarship_percentage_l2173_217322


namespace distance_from_point_to_line_l2173_217329

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  rho : ℝ
  theta : ℝ

/-- Represents a line in polar form ρ sin(θ - α) = k -/
structure PolarLine where
  alpha : ℝ
  k : ℝ

/-- Calculates the distance from a point in polar coordinates to a line in polar form -/
noncomputable def distanceFromPointToLine (p : PolarPoint) (l : PolarLine) : ℝ :=
  sorry

/-- Theorem stating that the distance from P(2, -π/6) to the line ρ sin(θ - π/6) = 1 is √3 + 1 -/
theorem distance_from_point_to_line :
  let p : PolarPoint := ⟨2, -π/6⟩
  let l : PolarLine := ⟨π/6, 1⟩
  distanceFromPointToLine p l = Real.sqrt 3 + 1 := by
  sorry

end distance_from_point_to_line_l2173_217329


namespace cos_2alpha_value_l2173_217353

theorem cos_2alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α - π/4) = 1/4) : 
  Real.cos (2 * α) = -Real.sqrt 15 / 8 := by
  sorry

end cos_2alpha_value_l2173_217353


namespace largest_decimal_l2173_217388

def circle_digits : List Nat := [1, 3, 9, 5, 7, 9, 1, 3, 9, 5, 7, 1]

def is_valid_decimal (d : ℚ) : Prop :=
  ∃ (n : ℕ) (l : List ℕ),
    l.length = 6 ∧
    l.all (λ x => x ∈ circle_digits) ∧
    d = n + (l.foldl (λ acc x => (acc + x) / 10) 0 : ℚ)

def is_largest_decimal (d : ℚ) : Prop :=
  is_valid_decimal d ∧
  ∀ d', is_valid_decimal d' → d' ≤ d

theorem largest_decimal :
  is_largest_decimal (9 + 579139 / 1000000 : ℚ) :=
sorry

end largest_decimal_l2173_217388


namespace absolute_value_inequality_l2173_217348

theorem absolute_value_inequality (x : ℝ) :
  |x - 1| - |x - 5| < 2 ↔ x < 4 := by sorry

end absolute_value_inequality_l2173_217348


namespace horseback_trip_distance_l2173_217368

/-- Calculates the total distance traveled during a horseback riding trip -/
def total_distance : ℝ :=
  let day1_segment1 := 5 * 7
  let day1_segment2 := 3 * 2
  let day2_segment1 := 6 * 6
  let day2_segment2 := 3 * 3
  let day3_segment1 := 4 * 3
  let day3_segment2 := 7 * 5
  day1_segment1 + day1_segment2 + day2_segment1 + day2_segment2 + day3_segment1 + day3_segment2

theorem horseback_trip_distance : total_distance = 133 := by
  sorry

end horseback_trip_distance_l2173_217368


namespace total_cars_on_train_l2173_217331

/-- The number of cars Rita counted in the first 15 seconds -/
def initial_cars : ℕ := 9

/-- The time in seconds during which Rita counted the initial cars -/
def initial_time : ℕ := 15

/-- The total time in seconds for the train to pass -/
def total_time : ℕ := 195

/-- The rate of cars passing per second -/
def rate : ℚ := initial_cars / initial_time

/-- The theorem stating the total number of cars on the train -/
theorem total_cars_on_train : ⌊rate * total_time⌋ = 117 := by sorry

end total_cars_on_train_l2173_217331


namespace change_eight_dollars_theorem_l2173_217356

theorem change_eight_dollars_theorem :
  ∃ (n : ℕ), n > 0 ∧
  (∃ (combinations : List (ℕ × ℕ × ℕ)),
    combinations.length = n ∧
    ∀ (c : ℕ × ℕ × ℕ), c ∈ combinations →
      let (nickels, dimes, quarters) := c
      nickels > 0 ∧ dimes > 0 ∧ quarters > 0 ∧
      5 * nickels + 10 * dimes + 25 * quarters = 800) :=
by sorry

end change_eight_dollars_theorem_l2173_217356


namespace average_age_of_five_students_l2173_217338

theorem average_age_of_five_students
  (total_students : Nat)
  (avg_age_all : ℝ)
  (num_group1 : Nat)
  (avg_age_group1 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 20)
  (h2 : avg_age_all = 20)
  (h3 : num_group1 = 9)
  (h4 : avg_age_group1 = 16)
  (h5 : age_last_student = 186)
  : ∃ (avg_age_group2 : ℝ),
    avg_age_group2 = 14 ∧
    avg_age_group2 * (total_students - num_group1 - 1) =
      total_students * avg_age_all - num_group1 * avg_age_group1 - age_last_student :=
by sorry

end average_age_of_five_students_l2173_217338


namespace students_in_both_band_and_chorus_l2173_217354

theorem students_in_both_band_and_chorus 
  (total_students : ℕ) 
  (band_students : ℕ) 
  (chorus_students : ℕ) 
  (either_band_or_chorus : ℕ) 
  (h1 : total_students = 300) 
  (h2 : band_students = 150) 
  (h3 : chorus_students = 180) 
  (h4 : either_band_or_chorus = 250) : 
  band_students + chorus_students - either_band_or_chorus = 80 := by
sorry

end students_in_both_band_and_chorus_l2173_217354


namespace product_in_first_quadrant_l2173_217371

def complex_multiply (a b c d : ℝ) : ℂ :=
  Complex.mk (a * c - b * d) (a * d + b * c)

theorem product_in_first_quadrant :
  let z : ℂ := complex_multiply 1 3 3 (-1)
  0 < z.re ∧ 0 < z.im :=
by sorry

end product_in_first_quadrant_l2173_217371


namespace peters_pants_purchase_l2173_217335

theorem peters_pants_purchase (shirt_price : ℕ) (pants_price : ℕ) (total_cost : ℕ) :
  shirt_price * 2 = 20 →
  pants_price = 6 →
  ∃ (num_pants : ℕ), shirt_price * 5 + pants_price * num_pants = 62 →
  num_pants = 2 := by
sorry

end peters_pants_purchase_l2173_217335


namespace parallel_vectors_x_value_l2173_217373

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, if they are parallel, then x = -4 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (-12, x - 4)
  parallel a b → x = -4 := by
  sorry

end parallel_vectors_x_value_l2173_217373


namespace g_of_5_equals_15_l2173_217384

def g (x : ℝ) : ℝ := x^2 - 2*x

theorem g_of_5_equals_15 : g 5 = 15 := by
  sorry

end g_of_5_equals_15_l2173_217384


namespace cody_dumplings_l2173_217315

def dumplings_problem (first_batch second_batch eaten_first shared_first shared_second additional_eaten : ℕ) : Prop :=
  let remaining_first := first_batch - eaten_first - shared_first
  let remaining_second := second_batch - shared_second
  let total_remaining := remaining_first + remaining_second - additional_eaten
  total_remaining = 10

theorem cody_dumplings :
  dumplings_problem 14 20 7 5 8 4 := by sorry

end cody_dumplings_l2173_217315


namespace harrys_book_pages_l2173_217350

theorem harrys_book_pages (selenas_pages : ℕ) (harrys_pages : ℕ) : 
  selenas_pages = 400 →
  harrys_pages = selenas_pages / 2 - 20 →
  harrys_pages = 180 :=
by
  sorry

end harrys_book_pages_l2173_217350


namespace percentage_equality_l2173_217343

theorem percentage_equality (x : ℝ) (h : 0.3 * (0.4 * x) = 60) : 0.4 * (0.3 * x) = 60 := by
  sorry

end percentage_equality_l2173_217343


namespace point_on_h_graph_l2173_217307

-- Define the function g
def g : ℝ → ℝ := sorry

-- Define the function h in terms of g
def h (x : ℝ) : ℝ := (g x)^3

-- State the theorem
theorem point_on_h_graph :
  ∃ (x y : ℝ), g 2 = -5 ∧ h x = y ∧ x + y = -123 := by sorry

end point_on_h_graph_l2173_217307


namespace cheerleader_group_composition_l2173_217369

theorem cheerleader_group_composition :
  let total_males : ℕ := 10
  let males_chose_malt : ℕ := 6
  let females_chose_malt : ℕ := 8
  let total_chose_malt : ℕ := males_chose_malt + females_chose_malt
  let total_chose_coke : ℕ := total_chose_malt / 2
  let females_chose_coke : ℕ := total_chose_coke
  total_males = 10 →
  males_chose_malt = 6 →
  females_chose_malt = 8 →
  total_chose_malt = 2 * total_chose_coke →
  (females_chose_malt + females_chose_coke : ℕ) = 15
  := by sorry

end cheerleader_group_composition_l2173_217369


namespace initial_ball_count_is_three_l2173_217352

def bat_cost : ℕ := 500
def ball_cost : ℕ := 100

def initial_purchase_cost : ℕ := 3800
def initial_bat_count : ℕ := 7

def second_purchase_cost : ℕ := 1750
def second_bat_count : ℕ := 3
def second_ball_count : ℕ := 5

theorem initial_ball_count_is_three : 
  ∃ (x : ℕ), 
    initial_bat_count * bat_cost + x * ball_cost = initial_purchase_cost ∧
    second_bat_count * bat_cost + second_ball_count * ball_cost = second_purchase_cost ∧
    x = 3 := by
  sorry

end initial_ball_count_is_three_l2173_217352


namespace smallest_perfect_square_sum_l2173_217337

/-- The sum of 20 consecutive positive integers starting from n -/
def sum_20_consecutive (n : ℕ) : ℕ := 10 * (2 * n + 19)

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_perfect_square_sum :
  (∃ n : ℕ, sum_20_consecutive n = 250) ∧
  (∀ m : ℕ, m < 250 → ¬∃ n : ℕ, sum_20_consecutive n = m ∧ is_perfect_square m) :=
sorry

end smallest_perfect_square_sum_l2173_217337


namespace cow_count_is_six_l2173_217382

/-- Represents the number of animals in a group -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of heads in the group -/
def totalHeads (g : AnimalGroup) : ℕ := g.ducks + g.cows

/-- Calculates the total number of legs in the group -/
def totalLegs (g : AnimalGroup) : ℕ := 2 * g.ducks + 4 * g.cows

/-- Theorem: If the total number of legs is 12 more than twice the number of heads,
    then the number of cows is 6 -/
theorem cow_count_is_six (g : AnimalGroup) :
  totalLegs g = 2 * totalHeads g + 12 → g.cows = 6 := by
  sorry

end cow_count_is_six_l2173_217382


namespace production_days_l2173_217328

theorem production_days (n : ℕ) 
  (h1 : (50 * n) / n = 50)  -- Average production for past n days
  (h2 : ((50 * n + 60) : ℝ) / (n + 1) = 55)  -- New average including today
  : n = 1 := by
  sorry

end production_days_l2173_217328


namespace computer_screen_height_l2173_217311

theorem computer_screen_height (side : ℝ) (height : ℝ) : 
  side = 20 →
  height = 4 * side + 20 →
  height = 100 := by
sorry

end computer_screen_height_l2173_217311


namespace complex_magnitude_product_l2173_217391

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 2 - 3 * Complex.I) * (2 * Real.sqrt 3 + 4 * Complex.I)) = 6 * Real.sqrt 21 := by
  sorry

end complex_magnitude_product_l2173_217391


namespace power_division_sum_equality_l2173_217355

theorem power_division_sum_equality : (-6)^5 / 6^2 + 4^3 - 7^2 = -201 := by
  sorry

end power_division_sum_equality_l2173_217355


namespace prayer_difference_l2173_217345

/-- Represents the number of prayers for a pastor in a week -/
structure WeeklyPrayers where
  regularDays : ℕ
  sunday : ℕ

/-- Calculates the total number of prayers in a week -/
def totalPrayers (wp : WeeklyPrayers) : ℕ :=
  wp.regularDays * 6 + wp.sunday

/-- Pastor Paul's prayer schedule -/
def paulPrayers : WeeklyPrayers where
  regularDays := 20
  sunday := 40

/-- Pastor Bruce's prayer schedule -/
def brucePrayers : WeeklyPrayers where
  regularDays := paulPrayers.regularDays / 2
  sunday := paulPrayers.sunday * 2

theorem prayer_difference : 
  totalPrayers paulPrayers - totalPrayers brucePrayers = 20 := by
  sorry


end prayer_difference_l2173_217345


namespace quadratic_function_max_a_l2173_217399

theorem quadratic_function_max_a (a b c m : ℝ) : 
  a < 0 →
  a * m^2 + b * m + c = b →
  a * (m + 1)^2 + b * (m + 1) + c = a →
  b ≥ a →
  m < 0 →
  (∀ x, a * x^2 + b * x + c ≤ -2) →
  (∀ a', a' < 0 → 
    (∀ x, a' * x^2 + (-a' * m) * x + (-a' * m) ≤ -2) → 
    a' ≤ a) →
  a = -8/3 := by
sorry

end quadratic_function_max_a_l2173_217399


namespace complex_number_fourth_quadrant_range_l2173_217366

theorem complex_number_fourth_quadrant_range (a : ℝ) : 
  let z : ℂ := (2 + Complex.I) * (a + 2 * Complex.I^3)
  (z.re > 0 ∧ z.im < 0) → -1 < a ∧ a < 4 :=
by sorry

end complex_number_fourth_quadrant_range_l2173_217366


namespace circle_radius_l2173_217332

theorem circle_radius (x y : ℝ) (h : x + y = 100 * Real.pi) :
  let r := Real.sqrt 101 - 1
  x = Real.pi * r^2 ∧ y = 2 * Real.pi * r := by
  sorry

end circle_radius_l2173_217332


namespace volume_maximized_at_two_l2173_217390

/-- The volume function of the box -/
def volume (x : ℝ) : ℝ := 4 * x * (6 - x)^2

/-- The side length of the original square sheet -/
def original_side : ℝ := 12

/-- Theorem stating that the volume is maximized when x = 2 -/
theorem volume_maximized_at_two :
  ∃ (max_x : ℝ), max_x = 2 ∧
  ∀ (x : ℝ), 0 < x ∧ x < original_side / 2 → volume x ≤ volume max_x :=
sorry

end volume_maximized_at_two_l2173_217390


namespace sum_of_max_min_g_l2173_217317

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 8| + 3

theorem sum_of_max_min_g :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 1 10, g x ≤ max) ∧
    (∃ x ∈ Set.Icc 1 10, g x = max) ∧
    (∀ x ∈ Set.Icc 1 10, min ≤ g x) ∧
    (∃ x ∈ Set.Icc 1 10, g x = min) ∧
    max + min = -1 :=
by
  sorry

end sum_of_max_min_g_l2173_217317


namespace base_10_to_base_12_153_l2173_217374

def base_12_digit (n : ℕ) : Char :=
  if n < 10 then Char.ofNat (n + 48)
  else if n = 10 then 'A'
  else 'B'

def to_base_12 (n : ℕ) : String :=
  let d₁ := n / 12
  let d₀ := n % 12
  String.mk [base_12_digit d₁, base_12_digit d₀]

theorem base_10_to_base_12_153 :
  to_base_12 153 = "B9" := by
  sorry

end base_10_to_base_12_153_l2173_217374


namespace cash_realized_before_brokerage_l2173_217324

/-- The cash realized on selling a stock before brokerage, given the total amount and brokerage rate -/
theorem cash_realized_before_brokerage 
  (total_amount : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : total_amount = 104)
  (h2 : brokerage_rate = 1 / 400) : 
  ∃ (cash_before_brokerage : ℝ), 
    cash_before_brokerage + cash_before_brokerage * brokerage_rate = total_amount ∧ 
    cash_before_brokerage = 41600 / 401 := by
  sorry

end cash_realized_before_brokerage_l2173_217324


namespace computer_table_cost_calculation_l2173_217309

/-- The cost price of the computer table -/
def computer_table_cost : ℝ := 4813.58

/-- The cost price of the office chair -/
def office_chair_cost : ℝ := 5000

/-- The markup percentage -/
def markup_percentage : ℝ := 0.24

/-- The discount percentage -/
def discount_percentage : ℝ := 0.05

/-- The total amount paid by the customer -/
def total_paid : ℝ := 11560

theorem computer_table_cost_calculation :
  let total_before_discount := (1 + markup_percentage) * (computer_table_cost + office_chair_cost)
  (1 - discount_percentage) * total_before_discount = total_paid := by
  sorry

#eval computer_table_cost

end computer_table_cost_calculation_l2173_217309


namespace binomial_threshold_l2173_217376

theorem binomial_threshold (n : ℕ) : 
  (n ≥ 82 → Nat.choose (2*n) n < 4^(n-2)) ∧ 
  (n ≥ 1305 → Nat.choose (2*n) n < 4^(n-3)) := by
  sorry

end binomial_threshold_l2173_217376


namespace raffle_probabilities_l2173_217321

structure Raffle :=
  (white_balls : ℕ)
  (black_balls : ℕ)
  (num_people : ℕ)

def first_person_wins (r : Raffle) : ℚ :=
  r.black_balls / (r.white_balls + r.black_balls)

def last_person_wins (r : Raffle) : ℚ :=
  (r.white_balls / (r.white_balls + r.black_balls)) *
  ((r.white_balls - 1) / (r.white_balls + r.black_balls - 1)) *
  ((r.white_balls - 2) / (r.white_balls + r.black_balls - 2)) *
  (r.black_balls / (r.white_balls + r.black_balls - 3))

def first_person_wins_continued (r : Raffle) : ℚ :=
  first_person_wins r +
  (r.white_balls / (r.white_balls + r.black_balls)) *
  ((r.white_balls - 1) / (r.white_balls + r.black_balls - 1)) *
  ((r.white_balls - 2) / (r.white_balls + r.black_balls - 2)) *
  ((r.white_balls - 3) / (r.white_balls + r.black_balls - 3)) *
  (r.black_balls / (r.white_balls + r.black_balls - 4))

def last_person_wins_continued (r : Raffle) : ℚ :=
  (r.white_balls / (r.white_balls + r.black_balls)) *
  ((r.white_balls - 1) / (r.white_balls + r.black_balls - 1)) *
  ((r.white_balls - 2) / (r.white_balls + r.black_balls - 2)) *
  (r.black_balls / (r.white_balls + r.black_balls - 3))

theorem raffle_probabilities :
  let r1 : Raffle := ⟨3, 1, 4⟩
  let r2 : Raffle := ⟨6, 2, 4⟩
  (first_person_wins r1 = 1/4) ∧
  (last_person_wins r1 = 1/4) ∧
  (first_person_wins_continued r2 = 5/14) ∧
  (last_person_wins_continued r2 = 1/7) :=
sorry

end raffle_probabilities_l2173_217321


namespace min_value_of_function_l2173_217395

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  ∃ (m : ℝ), m = 8 ∧ ∀ y, y = x + 1/x + 16*x/(x^2 + 1) → y ≥ m :=
by sorry

end min_value_of_function_l2173_217395


namespace square_area_is_49_l2173_217313

-- Define the right triangle ABC
structure RightTriangle :=
  (AB : ℝ)
  (BC : ℝ)
  (is_right : True)  -- Placeholder for the right angle condition

-- Define the square BDEF
structure Square :=
  (side : ℝ)

-- Define the triangle EMN
structure TriangleEMN :=
  (EH : ℝ)

-- Main theorem
theorem square_area_is_49 
  (triangle : RightTriangle)
  (square : Square)
  (triangle_EMN : TriangleEMN)
  (h1 : triangle.AB = 15)
  (h2 : triangle.BC = 20)
  (h3 : triangle_EMN.EH = 2) :
  square.side ^ 2 = 49 := by
  sorry

end square_area_is_49_l2173_217313


namespace ned_remaining_games_l2173_217300

/-- The number of games Ned initially had -/
def initial_games : ℕ := 19

/-- The number of games Ned gave away -/
def games_given_away : ℕ := 13

/-- The number of games Ned has now -/
def remaining_games : ℕ := initial_games - games_given_away

theorem ned_remaining_games : remaining_games = 6 := by
  sorry

end ned_remaining_games_l2173_217300


namespace largest_integer_satisfying_inequality_l2173_217359

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (6*x - 5 < 3*x + 4) → x ≤ 2 ∧ (6*2 - 5 < 3*2 + 4) :=
by
  sorry

end largest_integer_satisfying_inequality_l2173_217359


namespace even_function_implies_a_equals_two_l2173_217346

/-- Given a function f(x) = (x * e^x) / (e^(ax) - 1), prove that if f is even, then a = 2 -/
theorem even_function_implies_a_equals_two (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (x * Real.exp x) / (Real.exp (a * x) - 1) = 
    (-x * Real.exp (-x)) / (Real.exp (-a * x) - 1)) →
  a = 2 := by
sorry


end even_function_implies_a_equals_two_l2173_217346


namespace regular_polygon_15_diagonals_l2173_217381

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 15 diagonals has 7 sides -/
theorem regular_polygon_15_diagonals :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 15 ∧ n = 7 := by
  sorry

end regular_polygon_15_diagonals_l2173_217381


namespace lottery_first_prize_probability_l2173_217357

/-- The number of balls in the MegaBall drawing -/
def megaBallCount : ℕ := 30

/-- The number of balls in the WinnerBalls drawing -/
def winnerBallCount : ℕ := 50

/-- The number of WinnerBalls picked -/
def winnerBallsPicked : ℕ := 5

/-- The probability of winning the first prize in the lottery game -/
def firstPrizeProbability : ℚ := 1 / 127125600

/-- Theorem stating the probability of winning the first prize in the lottery game -/
theorem lottery_first_prize_probability :
  firstPrizeProbability = 1 / (megaBallCount * 2 * Nat.choose winnerBallCount winnerBallsPicked) :=
by sorry

end lottery_first_prize_probability_l2173_217357


namespace percentage_increase_in_workers_l2173_217358

theorem percentage_increase_in_workers (original : ℕ) (new : ℕ) : 
  original = 852 → new = 1065 → (new - original) / original * 100 = 25 := by
  sorry

end percentage_increase_in_workers_l2173_217358


namespace exam_selection_difference_l2173_217377

theorem exam_selection_difference (total_candidates : ℕ) 
  (selection_rate_A selection_rate_B : ℚ) : 
  total_candidates = 8200 →
  selection_rate_A = 6 / 100 →
  selection_rate_B = 7 / 100 →
  (selection_rate_B * total_candidates : ℚ).floor - 
  (selection_rate_A * total_candidates : ℚ).floor = 82 :=
by sorry

end exam_selection_difference_l2173_217377


namespace lords_partition_l2173_217330

/-- A graph with vertices of type α -/
structure Graph (α : Type) where
  adj : α → α → Prop

/-- The degree of a vertex in a graph -/
def degree {α : Type} (G : Graph α) (v : α) : ℕ := 
  sorry

/-- A partition of a set into two subsets -/
def Partition (α : Type) := (α → Bool)

/-- The number of adjacent vertices in the same partition -/
def samePartitionDegree {α : Type} (G : Graph α) (p : Partition α) (v : α) : ℕ := 
  sorry

theorem lords_partition {α : Type} (G : Graph α) :
  (∀ v : α, degree G v ≤ 3) →
  ∃ p : Partition α, ∀ v : α, samePartitionDegree G p v ≤ 1 := by
  sorry

end lords_partition_l2173_217330


namespace min_value_a_squared_minus_b_l2173_217344

/-- The function f(x) = x^4 + ax^3 + bx^2 + ax + 1 -/
def f (a b x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + a*x + 1

/-- Theorem: If f(x) has at least one root, then a^2 - b ≥ 1 -/
theorem min_value_a_squared_minus_b (a b : ℝ) :
  (∃ x, f a b x = 0) → a^2 - b ≥ 1 := by
  sorry

end min_value_a_squared_minus_b_l2173_217344


namespace midpoint_arrival_time_l2173_217393

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Represents a hiking event -/
structure HikingEvent where
  planned_start : Time
  planned_end : Time
  actual_start_delay : Nat
  actual_end_early : Nat

def midpoint_time (event : HikingEvent) : Time :=
  sorry

theorem midpoint_arrival_time (event : HikingEvent) : 
  event.planned_start = { hours := 10, minutes := 10 } →
  event.planned_end = { hours := 13, minutes := 10 } →
  event.actual_start_delay = 5 →
  event.actual_end_early = 4 →
  midpoint_time event = { hours := 11, minutes := 50 } :=
sorry

end midpoint_arrival_time_l2173_217393


namespace work_completion_time_l2173_217361

/-- Given that person A can complete a work in 30 days, and persons A and B together complete 1/9 of the work in 2 days, prove that person B can complete the work alone in 45 days. -/
theorem work_completion_time (a b : ℝ) (h1 : a = 30) 
  (h2 : 2 * (1 / a + 1 / b) = 1 / 9) : b = 45 := by
  sorry

end work_completion_time_l2173_217361


namespace original_number_is_22_l2173_217370

theorem original_number_is_22 (N : ℕ) : 
  (∀ k < 6, ¬ (16 ∣ (N - k))) →  -- Condition 1: 6 is the least number
  (16 ∣ (N - 6)) →               -- Condition 2: N - 6 is divisible by 16
  N = 22 := by                   -- Conclusion: The original number is 22
sorry

end original_number_is_22_l2173_217370


namespace jo_bob_balloon_ride_max_height_l2173_217342

/-- Represents the state of a hot air balloon ride -/
structure BalloonRide where
  ascent_rate : ℝ  -- Rate of ascent when chain is pulled (feet per minute)
  descent_rate : ℝ  -- Rate of descent when chain is released (feet per minute)
  first_pull_duration : ℝ  -- Duration of first chain pull (minutes)
  release_duration : ℝ  -- Duration of chain release (minutes)
  second_pull_duration : ℝ  -- Duration of second chain pull (minutes)

/-- Calculates the maximum height reached during a balloon ride -/
def max_height (ride : BalloonRide) : ℝ :=
  (ride.ascent_rate * ride.first_pull_duration) -
  (ride.descent_rate * ride.release_duration) +
  (ride.ascent_rate * ride.second_pull_duration)

/-- Theorem stating the maximum height reached during Jo-Bob's balloon ride -/
theorem jo_bob_balloon_ride_max_height :
  let ride : BalloonRide := {
    ascent_rate := 50,
    descent_rate := 10,
    first_pull_duration := 15,
    release_duration := 10,
    second_pull_duration := 15
  }
  max_height ride = 1400 := by sorry

end jo_bob_balloon_ride_max_height_l2173_217342


namespace x_squared_gt_4_necessary_not_sufficient_for_x_cubed_lt_neg_8_l2173_217305

theorem x_squared_gt_4_necessary_not_sufficient_for_x_cubed_lt_neg_8 :
  (∀ x : ℝ, x^3 < -8 → x^2 > 4) ∧
  (∃ x : ℝ, x^2 > 4 ∧ x^3 ≥ -8) :=
by sorry

end x_squared_gt_4_necessary_not_sufficient_for_x_cubed_lt_neg_8_l2173_217305


namespace joe_caught_23_times_l2173_217339

/-- The number of times Joe caught the ball -/
def joe_catches : ℕ := 23

/-- The number of times Derek caught the ball -/
def derek_catches (j : ℕ) : ℕ := 2 * j - 4

/-- The number of times Tammy caught the ball -/
def tammy_catches (d : ℕ) : ℕ := d / 3 + 16

theorem joe_caught_23_times :
  joe_catches = 23 ∧
  derek_catches joe_catches = 2 * joe_catches - 4 ∧
  tammy_catches (derek_catches joe_catches) = 30 :=
sorry

end joe_caught_23_times_l2173_217339
