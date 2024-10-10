import Mathlib

namespace alices_number_l3001_300103

theorem alices_number : ∃ n : ℕ, 
  (180 ∣ n) ∧ 
  (45 ∣ n) ∧ 
  1000 ≤ n ∧ 
  n < 3000 ∧ 
  (∀ m : ℕ, (180 ∣ m) ∧ (45 ∣ m) ∧ 1000 ≤ m ∧ m < 3000 → n ≤ m) ∧
  n = 1260 :=
by sorry

end alices_number_l3001_300103


namespace consecutive_primes_sum_composite_l3001_300130

theorem consecutive_primes_sum_composite (p₁ p₂ q : ℕ) : 
  Nat.Prime p₁ → Nat.Prime p₂ → 
  Odd p₁ → Odd p₂ → 
  p₁ < p₂ → 
  ¬∃k, Nat.Prime k ∧ p₁ < k ∧ k < p₂ →
  p₁ + p₂ = 2 * q → 
  ¬(Nat.Prime q) := by
sorry

end consecutive_primes_sum_composite_l3001_300130


namespace payment_for_C_is_250_l3001_300117

/-- Calculates the payment for worker C given the work rates and total payment --/
def calculate_payment_C (a_rate : ℚ) (b_rate : ℚ) (total_rate : ℚ) (total_payment : ℚ) : ℚ :=
  let c_rate := total_rate - (a_rate + b_rate)
  c_rate * total_payment

/-- Theorem stating that C should be paid 250 given the problem conditions --/
theorem payment_for_C_is_250 
  (a_rate : ℚ) 
  (b_rate : ℚ) 
  (total_rate : ℚ) 
  (total_payment : ℚ) 
  (h1 : a_rate = 1/6) 
  (h2 : b_rate = 1/8) 
  (h3 : total_rate = 1/3) 
  (h4 : total_payment = 6000) : 
  calculate_payment_C a_rate b_rate total_rate total_payment = 250 := by
  sorry

#eval calculate_payment_C (1/6) (1/8) (1/3) 6000

end payment_for_C_is_250_l3001_300117


namespace min_red_pieces_l3001_300126

theorem min_red_pieces (w b r : ℕ) : 
  b ≥ w / 2 →
  b ≤ r / 3 →
  w + b ≥ 55 →
  r ≥ 57 ∧ ∀ r', (∃ w' b', b' ≥ w' / 2 ∧ b' ≤ r' / 3 ∧ w' + b' ≥ 55) → r' ≥ r :=
by sorry

end min_red_pieces_l3001_300126


namespace expected_different_faces_formula_l3001_300124

/-- The number of sides on a fair die -/
def numSides : ℕ := 6

/-- The number of times the die is rolled -/
def numRolls : ℕ := 6

/-- The probability of a specific face not appearing in a single roll -/
def probNotAppear : ℚ := (numSides - 1) / numSides

/-- The expected number of different faces that appear when rolling a fair die -/
def expectedDifferentFaces : ℚ := numSides * (1 - probNotAppear ^ numRolls)

/-- Theorem stating the expected number of different faces when rolling a fair die -/
theorem expected_different_faces_formula :
  expectedDifferentFaces = (numSides^numRolls - (numSides - 1)^numRolls) / numSides^(numRolls - 1) :=
sorry

end expected_different_faces_formula_l3001_300124


namespace lillian_initial_candies_l3001_300100

/-- The number of candies Lillian's father gave her -/
def candies_from_father : ℕ := 5

/-- The total number of candies Lillian has after receiving candies from her father -/
def total_candies : ℕ := 93

/-- The number of candies Lillian collected initially -/
def initial_candies : ℕ := total_candies - candies_from_father

theorem lillian_initial_candies :
  initial_candies = 88 :=
by sorry

end lillian_initial_candies_l3001_300100


namespace mika_stickers_l3001_300110

/-- The number of stickers Mika has left after various transactions --/
def stickers_left (initial bought birthday given_away used : ℕ) : ℕ :=
  initial + bought + birthday - given_away - used

/-- Theorem stating that Mika is left with 2 stickers --/
theorem mika_stickers :
  stickers_left 20 26 20 6 58 = 2 := by
  sorry

end mika_stickers_l3001_300110


namespace elise_comic_book_expense_l3001_300120

theorem elise_comic_book_expense (initial_amount : ℤ) (saved_amount : ℤ) 
  (puzzle_cost : ℤ) (amount_left : ℤ) :
  initial_amount = 8 →
  saved_amount = 13 →
  puzzle_cost = 18 →
  amount_left = 1 →
  initial_amount + saved_amount - puzzle_cost - amount_left = 2 :=
by
  sorry

end elise_comic_book_expense_l3001_300120


namespace exam_max_marks_l3001_300146

theorem exam_max_marks :
  let pass_percentage : ℚ := 60 / 100
  let failing_score : ℕ := 210
  let failing_margin : ℕ := 90
  let max_marks : ℕ := 500
  (pass_percentage * max_marks : ℚ) = failing_score + failing_margin ∧
  max_marks = 500 := by
  sorry

end exam_max_marks_l3001_300146


namespace probability_at_least_one_head_l3001_300127

theorem probability_at_least_one_head (p : ℝ) (n : ℕ) : 
  p = 1/2 → n = 3 → 1 - (1 - p)^n = 7/8 := by
  sorry

end probability_at_least_one_head_l3001_300127


namespace dividend_rate_is_14_percent_l3001_300163

/-- Calculates the rate of dividend given investment details and annual income -/
def rate_of_dividend (total_investment : ℚ) (share_face_value : ℚ) (share_quoted_price : ℚ) (annual_income : ℚ) : ℚ :=
  let number_of_shares := total_investment / share_quoted_price
  let dividend_per_share := annual_income / number_of_shares
  (dividend_per_share / share_face_value) * 100

/-- Theorem stating that given the specific investment details and annual income, the rate of dividend is 14% -/
theorem dividend_rate_is_14_percent :
  rate_of_dividend 4940 10 9.5 728 = 14 := by
  sorry

end dividend_rate_is_14_percent_l3001_300163


namespace jason_initial_cards_l3001_300188

/-- The number of Pokemon cards Jason gave away. -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left. -/
def cards_left : ℕ := 4

/-- The initial number of Pokemon cards Jason had. -/
def initial_cards : ℕ := cards_given_away + cards_left

/-- Theorem stating that Jason initially had 13 Pokemon cards. -/
theorem jason_initial_cards : initial_cards = 13 := by
  sorry

end jason_initial_cards_l3001_300188


namespace shopping_discount_theorem_l3001_300158

def shoe_price : ℝ := 60
def dress_price : ℝ := 120
def accessory_price : ℝ := 25

def shoe_discount : ℝ := 0.3
def dress_discount : ℝ := 0.15
def accessory_discount : ℝ := 0.5
def additional_discount : ℝ := 0.1

def shoe_quantity : ℕ := 3
def dress_quantity : ℕ := 2
def accessory_quantity : ℕ := 3

def discount_threshold : ℝ := 200

theorem shopping_discount_theorem :
  let total_before_discount := shoe_price * shoe_quantity + dress_price * dress_quantity + accessory_price * accessory_quantity
  let shoe_discounted := shoe_price * shoe_quantity * (1 - shoe_discount)
  let dress_discounted := dress_price * dress_quantity * (1 - dress_discount)
  let accessory_discounted := accessory_price * accessory_quantity * (1 - accessory_discount)
  let total_after_category_discounts := shoe_discounted + dress_discounted + accessory_discounted
  let final_total := 
    if total_before_discount > discount_threshold
    then total_after_category_discounts * (1 - additional_discount)
    else total_after_category_discounts
  final_total = 330.75 := by
  sorry

end shopping_discount_theorem_l3001_300158


namespace solution_set_f_geq_1_minus_x_sq_range_of_a_for_nonempty_solution_l3001_300195

-- Define the function f(x) = |x-1|
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part (1)
theorem solution_set_f_geq_1_minus_x_sq :
  {x : ℝ | f x ≥ 1 - x^2} = {x : ℝ | x ≤ 0 ∨ x ≥ 1} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_nonempty_solution (a : ℝ) :
  (∃ x : ℝ, f x < a - x^2 + |x + 1|) ↔ a > -1 := by sorry

end solution_set_f_geq_1_minus_x_sq_range_of_a_for_nonempty_solution_l3001_300195


namespace bill_and_caroline_ages_l3001_300134

/-- Given that Bill is 17 years old and 1 year less than twice as old as his sister Caroline,
    prove that the sum of their ages is 26. -/
theorem bill_and_caroline_ages : ∀ (caroline_age : ℕ),
  17 = 2 * caroline_age - 1 →
  17 + caroline_age = 26 := by
  sorry

end bill_and_caroline_ages_l3001_300134


namespace toms_running_days_l3001_300138

/-- Proves that Tom runs 5 days a week given his running schedule and total distance covered -/
theorem toms_running_days 
  (hours_per_day : ℝ) 
  (speed : ℝ) 
  (total_miles_per_week : ℝ) 
  (h1 : hours_per_day = 1.5)
  (h2 : speed = 8)
  (h3 : total_miles_per_week = 60) :
  (total_miles_per_week / (speed * hours_per_day)) = 5 := by
  sorry


end toms_running_days_l3001_300138


namespace function_inequality_l3001_300178

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x > 0, (x * log x) * deriv f x < f x) : 
  2 * f (sqrt e) > f e := by
  sorry

end function_inequality_l3001_300178


namespace floor_equality_iff_interval_l3001_300143

theorem floor_equality_iff_interval (x : ℝ) :
  ⌊⌊2 * x⌋ - (1/2 : ℝ)⌋ = ⌊x + 3⌋ ↔ 3 ≤ x ∧ x < 4 :=
by sorry

end floor_equality_iff_interval_l3001_300143


namespace total_tickets_sold_l3001_300113

theorem total_tickets_sold (student_price general_price total_amount general_tickets : ℕ)
  (h1 : student_price = 4)
  (h2 : general_price = 6)
  (h3 : total_amount = 2876)
  (h4 : general_tickets = 388)
  (h5 : ∃ student_tickets : ℕ, student_price * student_tickets + general_price * general_tickets = total_amount) :
  ∃ total_tickets : ℕ, total_tickets = general_tickets + (total_amount - general_price * general_tickets) / student_price ∧ total_tickets = 525 :=
by
  sorry

end total_tickets_sold_l3001_300113


namespace parabola_symmetry_l3001_300186

/-- Prove that if (2, 3) lies on the parabola y = ax^2 + 2ax + c, then (-4, 3) also lies on it -/
theorem parabola_symmetry (a c : ℝ) : 
  (3 = a * 2^2 + 2 * a * 2 + c) → (3 = a * (-4)^2 + 2 * a * (-4) + c) := by
  sorry

end parabola_symmetry_l3001_300186


namespace ball_count_proof_l3001_300122

/-- Given a box with balls where some are red, prove that if there are 12 red balls
    and the probability of drawing a red ball is 0.6, then the total number of balls is 20. -/
theorem ball_count_proof (total_balls : ℕ) (red_balls : ℕ) (prob_red : ℚ) 
    (h1 : red_balls = 12)
    (h2 : prob_red = 6/10)
    (h3 : (red_balls : ℚ) / total_balls = prob_red) : 
  total_balls = 20 := by
  sorry

end ball_count_proof_l3001_300122


namespace equation_solutions_l3001_300149

theorem equation_solutions :
  (∀ x : ℝ, x * (x + 2) - (x + 2) = 0 ↔ x = -2 ∨ x = 1) ∧
  (∀ x : ℝ, x^2 - 4*x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) := by
  sorry

end equation_solutions_l3001_300149


namespace well_diameter_is_six_l3001_300132

def well_depth : ℝ := 24
def well_volume : ℝ := 678.5840131753953

theorem well_diameter_is_six :
  ∃ (d : ℝ), d = 6 ∧ well_volume = π * (d / 2)^2 * well_depth := by sorry

end well_diameter_is_six_l3001_300132


namespace bob_cleaning_time_l3001_300187

/-- Given that Alice takes 40 minutes to clean her room and Bob spends 3/8 of Alice's time,
    prove that Bob's cleaning time is 15 minutes. -/
theorem bob_cleaning_time (alice_time : ℕ) (bob_fraction : ℚ) :
  alice_time = 40 →
  bob_fraction = 3 / 8 →
  (bob_fraction * alice_time : ℚ) = 15 := by
  sorry

end bob_cleaning_time_l3001_300187


namespace percentage_relation_l3001_300140

theorem percentage_relation (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x / 100 * y = 12) (h2 : y / 100 * x = 9) : x = 400 / 3 := by
  sorry

end percentage_relation_l3001_300140


namespace jerry_current_average_l3001_300179

def jerry_average_score (current_average : ℝ) (raise_by : ℝ) (fourth_test_score : ℝ) : Prop :=
  let total_score_needed := 4 * (current_average + raise_by)
  3 * current_average + fourth_test_score = total_score_needed

theorem jerry_current_average : 
  ∃ (A : ℝ), jerry_average_score A 2 89 ∧ A = 81 :=
sorry

end jerry_current_average_l3001_300179


namespace intersecting_line_equations_l3001_300109

/-- A line passing through a point and intersecting a circle --/
structure IntersectingLine where
  -- The point through which the line passes
  point : ℝ × ℝ
  -- The center of the circle
  center : ℝ × ℝ
  -- The radius of the circle
  radius : ℝ
  -- The length of the chord formed by the intersection
  chord_length : ℝ

/-- The equations of the line given the conditions --/
def line_equations (l : IntersectingLine) : Set (ℝ → ℝ → Prop) :=
  { (λ x y => x = -4),
    (λ x y => 4*x + 3*y + 25 = 0) }

/-- Theorem stating that the given conditions result in the specified line equations --/
theorem intersecting_line_equations 
  (l : IntersectingLine)
  (h1 : l.point = (-4, -3))
  (h2 : l.center = (-1, -2))
  (h3 : l.radius = 5)
  (h4 : l.chord_length = 8) :
  ∃ (eq : ℝ → ℝ → Prop), eq ∈ line_equations l ∧ 
    ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | eq p.1 p.2} → 
      ((x + 1)^2 + (y + 2)^2 = 25 ∨ (x, y) = l.point) :=
sorry

end intersecting_line_equations_l3001_300109


namespace simplify_expression_l3001_300172

theorem simplify_expression (x : ℝ) : (3*x)^4 + (4*x)*(x^5) = 81*x^4 + 4*x^6 := by
  sorry

end simplify_expression_l3001_300172


namespace candy_bar_sales_l3001_300175

/-- The number of additional candy bars sold each day -/
def additional_candy_bars : ℕ := sorry

/-- The cost of each candy bar in cents -/
def candy_bar_cost : ℕ := 10

/-- The number of days Sol sells candy bars in a week -/
def selling_days : ℕ := 6

/-- The number of candy bars sold on the first day -/
def first_day_sales : ℕ := 10

/-- The total earnings in cents for the week -/
def total_earnings : ℕ := 1200

theorem candy_bar_sales :
  (first_day_sales * selling_days + 
   additional_candy_bars * (selling_days * (selling_days - 1) / 2)) * 
  candy_bar_cost = total_earnings :=
sorry

end candy_bar_sales_l3001_300175


namespace cargo_weight_calculation_l3001_300193

/-- Calculates the total cargo weight after loading and unloading activities -/
def total_cargo_weight (initial_cargo : Real) (additional_cargo : Real) (unloaded_cargo : Real) 
  (short_ton_to_kg : Real) (pound_to_kg : Real) : Real :=
  (initial_cargo * short_ton_to_kg) + (additional_cargo * short_ton_to_kg) - (unloaded_cargo * pound_to_kg)

/-- Theorem stating the total cargo weight after loading and unloading activities -/
theorem cargo_weight_calculation :
  let initial_cargo : Real := 5973.42
  let additional_cargo : Real := 8723.18
  let unloaded_cargo : Real := 2256719.55
  let short_ton_to_kg : Real := 907.18474
  let pound_to_kg : Real := 0.45359237
  total_cargo_weight initial_cargo additional_cargo unloaded_cargo short_ton_to_kg pound_to_kg = 12302024.7688159 := by
  sorry


end cargo_weight_calculation_l3001_300193


namespace number_problem_l3001_300121

theorem number_problem (N : ℝ) 
  (h1 : (1/4) * (1/3) * (2/5) * N = 17) 
  (h2 : Real.sqrt (0.6 * N) = (N^(1/3)) / 2) : 
  0.4 * N = 204 := by sorry

end number_problem_l3001_300121


namespace video_votes_l3001_300180

theorem video_votes (up_votes : ℕ) (ratio_up : ℕ) (ratio_down : ℕ) (down_votes : ℕ) : 
  up_votes = 18 →
  ratio_up = 9 →
  ratio_down = 2 →
  up_votes * ratio_down = down_votes * ratio_up →
  down_votes = 4 := by
sorry

end video_votes_l3001_300180


namespace problem_1_l3001_300141

theorem problem_1 : -20 - (-14) - |(-18)| - 13 = -37 := by
  sorry

end problem_1_l3001_300141


namespace stone_slab_length_l3001_300152

theorem stone_slab_length (total_area : ℝ) (num_slabs : ℕ) (slab_length : ℝ) :
  total_area = 72 →
  num_slabs = 50 →
  (slab_length ^ 2) * num_slabs = total_area →
  slab_length = 1.2 := by
sorry

end stone_slab_length_l3001_300152


namespace angle_between_planes_l3001_300131

def plane1 : ℝ → ℝ → ℝ → ℝ := fun x y z ↦ 3 * x - 4 * y + z - 8
def plane2 : ℝ → ℝ → ℝ → ℝ := fun x y z ↦ 9 * x - 12 * y - 4 * z + 6

def normal1 : Fin 3 → ℝ := fun i ↦ match i with
  | 0 => 3
  | 1 => -4
  | 2 => 1

def normal2 : Fin 3 → ℝ := fun i ↦ match i with
  | 0 => 9
  | 1 => -12
  | 2 => -4

theorem angle_between_planes :
  let dot_product := (normal1 0 * normal2 0 + normal1 1 * normal2 1 + normal1 2 * normal2 2)
  let magnitude1 := Real.sqrt (normal1 0 ^ 2 + normal1 1 ^ 2 + normal1 2 ^ 2)
  let magnitude2 := Real.sqrt (normal2 0 ^ 2 + normal2 1 ^ 2 + normal2 2 ^ 2)
  dot_product / (magnitude1 * magnitude2) = 71 / (Real.sqrt 26 * Real.sqrt 241) := by
sorry

end angle_between_planes_l3001_300131


namespace cosine_sum_zero_l3001_300135

theorem cosine_sum_zero (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos (y + 2 * Real.pi / 3) + Real.cos (z + 4 * Real.pi / 3) = 0)
  (h2 : Real.sin x + Real.sin (y + 2 * Real.pi / 3) + Real.sin (z + 4 * Real.pi / 3) = 0) :
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 0 := by
sorry

end cosine_sum_zero_l3001_300135


namespace remainder_theorem_l3001_300157

-- Define the polynomial p(x)
variable (p : ℝ → ℝ)

-- Define the conditions
axiom p_div_x_minus_2 : ∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * q x + 2
axiom p_div_x_minus_3 : ∃ q : ℝ → ℝ, ∀ x, p x = (x - 3) * q x + 6

-- State the theorem
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * (x - 3) * q x + (4 * x - 6) := by
  sorry

end remainder_theorem_l3001_300157


namespace min_hours_sixth_week_l3001_300151

/-- The required average number of hours per week -/
def required_average : ℚ := 12

/-- The number of weeks -/
def total_weeks : ℕ := 6

/-- The hours worked in the first 5 weeks -/
def first_five_weeks : List ℕ := [9, 10, 14, 11, 8]

/-- The sum of hours worked in the first 5 weeks -/
def sum_first_five : ℕ := first_five_weeks.sum

theorem min_hours_sixth_week : 
  ∀ x : ℕ, 
    (sum_first_five + x : ℚ) / total_weeks ≥ required_average → 
    x ≥ 20 := by
  sorry

end min_hours_sixth_week_l3001_300151


namespace repeating_decimal_as_fraction_l3001_300125

-- Define the repeating decimal 4.666...
def repeating_decimal : ℚ :=
  4 + (2 / 3)

-- Theorem statement
theorem repeating_decimal_as_fraction :
  repeating_decimal = 14 / 3 := by
  sorry

end repeating_decimal_as_fraction_l3001_300125


namespace greatest_prime_factor_of_2_8_plus_5_5_l3001_300116

theorem greatest_prime_factor_of_2_8_plus_5_5 :
  ∃ p : ℕ, p.Prime ∧ p ∣ (2^8 + 5^5) ∧ ∀ q : ℕ, q.Prime → q ∣ (2^8 + 5^5) → q ≤ p :=
by sorry

end greatest_prime_factor_of_2_8_plus_5_5_l3001_300116


namespace feasible_measures_correct_l3001_300154

-- Define the set of all proposed measures
def AllMeasures : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the set of infeasible measures
def InfeasibleMeasures : Set ℕ := {4, 5, 6, 8}

-- Define a predicate for feasible measures
def IsFeasibleMeasure (m : ℕ) : Prop :=
  m ∈ AllMeasures ∧ m ∉ InfeasibleMeasures

-- Define the set of feasible measures
def FeasibleMeasures : Set ℕ := {m ∈ AllMeasures | IsFeasibleMeasure m}

-- Theorem statement
theorem feasible_measures_correct :
  FeasibleMeasures = AllMeasures \ InfeasibleMeasures :=
sorry

end feasible_measures_correct_l3001_300154


namespace salt_mixture_price_l3001_300104

theorem salt_mixture_price (salt_price_1 : ℚ) (salt_weight_1 : ℚ)
  (salt_weight_2 : ℚ) (selling_price : ℚ) (profit_percentage : ℚ) :
  salt_price_1 = 50 / 100 →
  salt_weight_1 = 8 →
  salt_weight_2 = 40 →
  selling_price = 48 / 100 →
  profit_percentage = 20 / 100 →
  ∃ (salt_price_2 : ℚ),
    salt_price_2 * salt_weight_2 + salt_price_1 * salt_weight_1 =
      (selling_price * (salt_weight_1 + salt_weight_2)) / (1 + profit_percentage) ∧
    salt_price_2 = 38 / 100 :=
by sorry

end salt_mixture_price_l3001_300104


namespace road_trash_cans_l3001_300177

/-- The number of trash cans on both sides of a road -/
def trashCanCount (roadLength : ℕ) (interval : ℕ) : ℕ :=
  2 * ((roadLength / interval) - 1)

/-- Theorem: The total number of trash cans on a 400-meter road with 20-meter intervals is 38 -/
theorem road_trash_cans :
  trashCanCount 400 20 = 38 := by
  sorry

end road_trash_cans_l3001_300177


namespace intersection_point_pq_rs_l3001_300142

/-- The intersection point of two lines in 3D space --/
def intersection_point (p q r s : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Given points in 3D space --/
def P : ℝ × ℝ × ℝ := (4, -3, 6)
def Q : ℝ × ℝ × ℝ := (20, -23, 14)
def R : ℝ × ℝ × ℝ := (-2, 7, -10)
def S : ℝ × ℝ × ℝ := (6, -11, 16)

/-- Theorem stating that the intersection point of lines PQ and RS is (180/19, -283/19, 202/19) --/
theorem intersection_point_pq_rs :
  intersection_point P Q R S = (180/19, -283/19, 202/19) := by sorry

end intersection_point_pq_rs_l3001_300142


namespace veronica_brown_balls_l3001_300196

/-- Given that Veronica carried 27 yellow balls and 45% of the total balls were yellow,
    prove that she carried 33 brown balls. -/
theorem veronica_brown_balls :
  ∀ (total_balls : ℕ) (yellow_balls : ℕ) (brown_balls : ℕ),
    yellow_balls = 27 →
    (yellow_balls : ℚ) / (total_balls : ℚ) = 45 / 100 →
    total_balls = yellow_balls + brown_balls →
    brown_balls = 33 := by
  sorry

end veronica_brown_balls_l3001_300196


namespace square_of_binomial_coefficient_l3001_300119

/-- If bx^2 + 18x + 9 is the square of a binomial, then b = 9 -/
theorem square_of_binomial_coefficient (b : ℝ) : 
  (∃ t u : ℝ, ∀ x : ℝ, bx^2 + 18*x + 9 = (t*x + u)^2) → b = 9 :=
by sorry

end square_of_binomial_coefficient_l3001_300119


namespace tangent_line_equation_l3001_300115

/-- The equation of the tangent line to the circle x^2 + y^2 - 4x = 0 at the point (1, √3) is x - √3y + 2 = 0. -/
theorem tangent_line_equation (x y : ℝ) :
  let circle_equation := (x^2 + y^2 - 4*x = 0)
  let point_on_circle := (1, Real.sqrt 3)
  let tangent_line := (x - Real.sqrt 3 * y + 2 = 0)
  circle_equation ∧ (x, y) = point_on_circle → tangent_line := by
  sorry

end tangent_line_equation_l3001_300115


namespace consecutive_product_prime_power_and_perfect_power_l3001_300160

theorem consecutive_product_prime_power_and_perfect_power (m : ℕ) : m ≥ 1 → (
  (∃ (p : ℕ) (k : ℕ), Prime p ∧ m * (m + 1) = p^k) ↔ m = 1
) ∧ 
¬(∃ (a k : ℕ), a ≥ 1 ∧ k ≥ 2 ∧ m * (m + 1) = a^k) := by
  sorry

end consecutive_product_prime_power_and_perfect_power_l3001_300160


namespace fraction_equality_l3001_300128

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 2 / 3) :
  t / q = 1 := by
  sorry

end fraction_equality_l3001_300128


namespace target_probability_l3001_300107

theorem target_probability (p : ℝ) : 
  (1 - (1 - p)^3 = 0.875) → p = 0.5 := by
  sorry

end target_probability_l3001_300107


namespace statement_2_statement_4_l3001_300192

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Statement ②
theorem statement_2 
  (l m : Line) (α β : Plane) 
  (h1 : subset l α) 
  (h2 : parallel_line_plane l β) 
  (h3 : intersect α β m) : 
  parallel l m :=
sorry

-- Statement ④
theorem statement_4 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : parallel l m) 
  (h3 : parallel_plane α β) : 
  perpendicular m β :=
sorry

end statement_2_statement_4_l3001_300192


namespace second_year_cost_difference_l3001_300101

/-- Proves that the difference between second and first year payments is 2 --/
theorem second_year_cost_difference (total_payments : ℕ) (first_year : ℕ) (x : ℕ) :
  total_payments = 96 →
  first_year = 20 →
  total_payments = first_year + (first_year + x) + (first_year + x + 3) + (first_year + x + 3 + 4) →
  x = 2 := by
  sorry

end second_year_cost_difference_l3001_300101


namespace diplomats_language_theorem_l3001_300181

theorem diplomats_language_theorem (total : ℕ) (japanese : ℕ) (not_russian : ℕ) (both_percent : ℚ) :
  total = 120 →
  japanese = 20 →
  not_russian = 32 →
  both_percent = 1/10 →
  (↑(total - (japanese + (total - not_russian) - (both_percent * ↑total).num)) / ↑total : ℚ) = 1/5 := by
  sorry

end diplomats_language_theorem_l3001_300181


namespace absent_students_percentage_l3001_300162

theorem absent_students_percentage
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (boys_absent_fraction : ℚ)
  (girls_absent_fraction : ℚ)
  (h1 : total_students = 240)
  (h2 : boys = 150)
  (h3 : girls = 90)
  (h4 : boys_absent_fraction = 1 / 5)
  (h5 : girls_absent_fraction = 1 / 2)
  (h6 : total_students = boys + girls) :
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students = 5 / 16 :=
by sorry

end absent_students_percentage_l3001_300162


namespace problem_statement_l3001_300190

theorem problem_statement (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) (h : x = 1 / z^2) :
  (x - 1/x) * (z^2 + 1/z^2) = x^2 - z^4 := by
  sorry

end problem_statement_l3001_300190


namespace function_passes_through_point_l3001_300148

theorem function_passes_through_point (a : ℝ) (h : 0 < a ∧ a < 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a * x - 1
  f 1 = 1 := by sorry

end function_passes_through_point_l3001_300148


namespace equation_decomposition_l3001_300197

-- Define the equation
def equation (x y : ℝ) : Prop := y^6 - 9*x^6 = 3*y^3 - 1

-- Define a parabola
def is_parabola (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Theorem statement
theorem equation_decomposition :
  ∃ f g : ℝ → ℝ, 
    (∀ x y, equation x y ↔ (y = f x ∨ y = g x)) ∧
    is_parabola f ∧ is_parabola g :=
sorry

end equation_decomposition_l3001_300197


namespace cream_cheese_cost_l3001_300111

/-- Cost of items for staff meetings -/
theorem cream_cheese_cost (bagel_cost cream_cheese_cost : ℝ) : 
  2 * bagel_cost + 3 * cream_cheese_cost = 12 →
  4 * bagel_cost + 2 * cream_cheese_cost = 14 →
  cream_cheese_cost = 2.5 := by
sorry

end cream_cheese_cost_l3001_300111


namespace intersection_of_A_and_B_l3001_300164

def A : Set ℝ := {x | x^2 - 2*x - 8 > 0}
def B : Set ℝ := {-3, -1, 1, 3, 5}

theorem intersection_of_A_and_B : A ∩ B = {-3, 5} := by sorry

end intersection_of_A_and_B_l3001_300164


namespace optimal_move_is_MN_l3001_300102

/-- Represents a move in the game -/
inductive Move
| FG
| MN

/-- Represents the outcome of the game -/
structure Outcome :=
(player_score : ℕ)
(opponent_score : ℕ)

/-- The game state after 12 moves (6 by each player) -/
def initial_state : ℕ := 12

/-- Simulates the game outcome based on the chosen move -/
def simulate_game (move : Move) : Outcome :=
  match move with
  | Move.FG => ⟨1, 8⟩
  | Move.MN => ⟨8, 1⟩

/-- Determines if one outcome is better than another for the player -/
def is_better_outcome (o1 o2 : Outcome) : Prop :=
  o1.player_score > o2.player_score

theorem optimal_move_is_MN :
  let fg_outcome := simulate_game Move.FG
  let mn_outcome := simulate_game Move.MN
  is_better_outcome mn_outcome fg_outcome :=
by sorry


end optimal_move_is_MN_l3001_300102


namespace mashed_potatoes_count_l3001_300108

theorem mashed_potatoes_count : ∀ (bacon_count mashed_count : ℕ),
  bacon_count = 489 →
  bacon_count = mashed_count + 10 →
  mashed_count = 479 := by
  sorry

end mashed_potatoes_count_l3001_300108


namespace revision_cost_per_page_revision_cost_is_four_l3001_300156

/-- The cost per page for revision in a manuscript typing service --/
theorem revision_cost_per_page : ℝ → Prop :=
  fun x =>
    let total_pages : ℕ := 100
    let pages_revised_once : ℕ := 35
    let pages_revised_twice : ℕ := 15
    let pages_not_revised : ℕ := total_pages - pages_revised_once - pages_revised_twice
    let initial_typing_cost_per_page : ℝ := 6
    let total_cost : ℝ := 860
    (initial_typing_cost_per_page * total_pages + x * pages_revised_once + 2 * x * pages_revised_twice = total_cost) →
    x = 4

theorem revision_cost_is_four : revision_cost_per_page 4 := by
  sorry

end revision_cost_per_page_revision_cost_is_four_l3001_300156


namespace rectangle_area_l3001_300199

/-- The area of a rectangle with width 7 meters and length 2 meters longer than the width is 63 square meters. -/
theorem rectangle_area : ℝ → ℝ → ℝ → Prop :=
  fun width length area =>
    width = 7 ∧ length = width + 2 → area = width * length → area = 63

/-- Proof of the theorem -/
lemma rectangle_area_proof : rectangle_area 7 9 63 := by
  sorry

end rectangle_area_l3001_300199


namespace inequality_proof_l3001_300144

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end inequality_proof_l3001_300144


namespace square_of_complex_2_minus_i_l3001_300145

theorem square_of_complex_2_minus_i :
  let z : ℂ := 2 - I
  z^2 = 3 - 4*I :=
by sorry

end square_of_complex_2_minus_i_l3001_300145


namespace adult_meal_cost_l3001_300150

/-- Calculates the cost of an adult meal given the total number of people,
    number of kids, and total cost for a group at a restaurant where kids eat free. -/
theorem adult_meal_cost (total_people : ℕ) (num_kids : ℕ) (total_cost : ℚ) :
  total_people = 12 →
  num_kids = 7 →
  total_cost = 15 →
  (total_cost / (total_people - num_kids) : ℚ) = 3 := by
  sorry

#check adult_meal_cost

end adult_meal_cost_l3001_300150


namespace gumball_ratio_l3001_300191

/-- The number of red gumballs in the machine -/
def red_gumballs : ℕ := 16

/-- The total number of gumballs in the machine -/
def total_gumballs : ℕ := 56

/-- The number of blue gumballs in the machine -/
def blue_gumballs : ℕ := red_gumballs / 2

/-- The number of green gumballs in the machine -/
def green_gumballs : ℕ := total_gumballs - red_gumballs - blue_gumballs

/-- The ratio of green gumballs to blue gumballs is 4:1 -/
theorem gumball_ratio : 
  green_gumballs / blue_gumballs = 4 := by sorry

end gumball_ratio_l3001_300191


namespace quadratic_equation_solution_l3001_300136

theorem quadratic_equation_solution (x : ℝ) : 
  (x - 2) * (x + 3) = 0 ↔ x = 2 ∨ x = -3 := by
  sorry

end quadratic_equation_solution_l3001_300136


namespace least_positive_integer_multiple_of_53_l3001_300155

theorem least_positive_integer_multiple_of_53 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ ((3*y)^2 + 2*43*(3*y) + 43^2))) ∧
  (53 ∣ ((3*x)^2 + 2*43*(3*x) + 43^2)) ∧
  x = 21 := by
  sorry

end least_positive_integer_multiple_of_53_l3001_300155


namespace test_has_ten_four_point_questions_l3001_300198

/-- Represents a test with two-point and four-point questions -/
structure Test where
  total_points : ℕ
  total_questions : ℕ
  two_point_questions : ℕ
  four_point_questions : ℕ

/-- Checks if a test configuration is valid -/
def is_valid_test (t : Test) : Prop :=
  t.total_questions = t.two_point_questions + t.four_point_questions ∧
  t.total_points = 2 * t.two_point_questions + 4 * t.four_point_questions

/-- Theorem: A test with 100 points and 40 questions has 10 four-point questions -/
theorem test_has_ten_four_point_questions (t : Test) 
  (h1 : t.total_points = 100) 
  (h2 : t.total_questions = 40) 
  (h3 : is_valid_test t) : 
  t.four_point_questions = 10 := by
  sorry

end test_has_ten_four_point_questions_l3001_300198


namespace milk_container_problem_l3001_300147

/-- The initial quantity of milk in container A --/
def initial_quantity : ℝ := 1232

/-- The fraction of container A's capacity that goes into container B --/
def fraction_in_B : ℝ := 0.375

/-- The amount transferred from C to B to equalize the quantities --/
def transfer_amount : ℝ := 154

theorem milk_container_problem :
  -- Container A is filled to its brim
  -- All milk from A is poured into B and C
  -- Quantity in B is 62.5% less than A (which means it's 37.5% of A)
  -- If 154L is transferred from C to B, they become equal
  (fraction_in_B * initial_quantity + transfer_amount = 
   (1 - fraction_in_B) * initial_quantity - transfer_amount) →
  -- Then the initial quantity in A is 1232 liters
  initial_quantity = 1232 := by
  sorry

end milk_container_problem_l3001_300147


namespace sin_585_degrees_l3001_300153

theorem sin_585_degrees :
  Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_585_degrees_l3001_300153


namespace tangent_line_at_1_l3001_300118

-- Define the function f
def f (x : ℝ) : ℝ := x^2 * (x - 2) + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

-- Theorem statement
theorem tangent_line_at_1 : 
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m*x + b ↔ x - y + 1 = 0) ∧ 
  (m = f' 1) ∧ 
  (f 1 = m*1 + b) := by
  sorry

end tangent_line_at_1_l3001_300118


namespace lemonade_sold_l3001_300161

/-- Represents the number of cups of lemonade sold -/
def lemonade : ℕ := sorry

/-- Represents the number of cups of hot chocolate sold -/
def hotChocolate : ℕ := sorry

/-- The total number of cups sold -/
def totalCups : ℕ := 400

/-- The total money earned in yuan -/
def totalMoney : ℕ := 546

/-- The price of a cup of lemonade in yuan -/
def lemonadePrice : ℕ := 1

/-- The price of a cup of hot chocolate in yuan -/
def hotChocolatePrice : ℕ := 2

theorem lemonade_sold : 
  lemonade = 254 ∧ 
  lemonade + hotChocolate = totalCups ∧ 
  lemonade * lemonadePrice + hotChocolate * hotChocolatePrice = totalMoney :=
sorry

end lemonade_sold_l3001_300161


namespace y_intercept_of_line_l3001_300174

/-- The y-intercept of the line 4x + 7y = 28 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) : 
  4 * x + 7 * y = 28 → (x = 0 ∧ y = 4) → (0, 4).fst = x ∧ (0, 4).snd = y := by
  sorry

end y_intercept_of_line_l3001_300174


namespace cylinder_volume_relationship_l3001_300139

/-- Theorem about the volumes of two cylinders with specific relationships -/
theorem cylinder_volume_relationship (h r_C r_D h_C h_D : ℝ) 
  (h_positive : h > 0)
  (cylinder_C : r_C = h ∧ h_C = 3 * r_D)
  (cylinder_D : r_D = h / 3 ∧ h_D = h)
  (volume_ratio : π * r_D^2 * h_D = 3 * (π * r_C^2 * h_C)) :
  π * r_D^2 * h_D = 3 * π * h^3 :=
sorry

end cylinder_volume_relationship_l3001_300139


namespace arithmetic_sequence_sum_ratio_l3001_300106

theorem arithmetic_sequence_sum_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h1 : ∀ n, S n = n / 2 * (a 1 + a n))
  (h2 : a 7 / a 4 = 2) :
  S 13 / S 7 = 26 / 7 := by
sorry

end arithmetic_sequence_sum_ratio_l3001_300106


namespace lucky_number_properties_l3001_300123

def is_lucky_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  let ab := n / 100
  let cd := n % 100
  ab ≠ cd ∧ cd % ab = 0 ∧ n % cd = 0

def count_lucky_numbers : ℕ := sorry

def largest_odd_lucky_number : ℕ := sorry

theorem lucky_number_properties :
  count_lucky_numbers = 65 ∧
  largest_odd_lucky_number = 1995 ∧
  is_lucky_number largest_odd_lucky_number ∧
  (∀ n, is_lucky_number n → n % 2 = 1 → n ≤ largest_odd_lucky_number) := by sorry

end lucky_number_properties_l3001_300123


namespace omars_kite_height_l3001_300171

/-- Omar's kite raising problem -/
theorem omars_kite_height 
  (omar_time : ℝ) 
  (jasper_rate_multiplier : ℝ) 
  (jasper_height : ℝ) 
  (jasper_time : ℝ) :
  omar_time = 12 →
  jasper_rate_multiplier = 3 →
  jasper_height = 600 →
  jasper_time = 10 →
  (omar_time * (jasper_height / jasper_time) / jasper_rate_multiplier) = 240 := by
  sorry

#check omars_kite_height

end omars_kite_height_l3001_300171


namespace regular_polyhedra_similarity_l3001_300129

/-- A regular polyhedron -/
structure RegularPolyhedron where
  -- Add necessary fields here
  -- For example:
  vertices : Set (ℝ × ℝ × ℝ)
  faces : Set (Set (ℝ × ℝ × ℝ))
  -- Add more fields as needed

/-- Defines what it means for two polyhedra to be of the same combinatorial type -/
def same_combinatorial_type (P Q : RegularPolyhedron) : Prop :=
  sorry

/-- Defines what it means for faces to be of the same kind -/
def same_kind_faces (P Q : RegularPolyhedron) : Prop :=
  sorry

/-- Defines what it means for polyhedral angles to be of the same kind -/
def same_kind_angles (P Q : RegularPolyhedron) : Prop :=
  sorry

/-- Defines similarity between two polyhedra -/
def similar (P Q : RegularPolyhedron) : Prop :=
  sorry

/-- The main theorem: two regular polyhedra of the same combinatorial type
    with faces and polyhedral angles of the same kind are similar -/
theorem regular_polyhedra_similarity (P Q : RegularPolyhedron)
  (h1 : same_combinatorial_type P Q)
  (h2 : same_kind_faces P Q)
  (h3 : same_kind_angles P Q) :
  similar P Q :=
by
  sorry

end regular_polyhedra_similarity_l3001_300129


namespace prob_event_A_is_three_eighths_l3001_300167

/-- Represents the faces of the tetrahedron -/
inductive Face : Type
  | zero : Face
  | one : Face
  | two : Face
  | three : Face

/-- Converts a Face to its numerical value -/
def faceValue : Face → ℕ
  | Face.zero => 0
  | Face.one => 1
  | Face.two => 2
  | Face.three => 3

/-- Defines the event A: m^2 + n^2 ≤ 4 -/
def eventA (m n : Face) : Prop :=
  (faceValue m)^2 + (faceValue n)^2 ≤ 4

/-- The probability of event A occurring -/
def probEventA : ℚ := 3/8

/-- Theorem stating that the probability of event A is 3/8 -/
theorem prob_event_A_is_three_eighths :
  probEventA = 3/8 := by sorry

end prob_event_A_is_three_eighths_l3001_300167


namespace vector_equation_l3001_300176

theorem vector_equation (a b : ℝ × ℝ) : 
  a = (1, 2) → 2 • a + b = (3, 2) → b = (1, -2) := by sorry

end vector_equation_l3001_300176


namespace max_x_implies_a_value_l3001_300184

/-- Given that the maximum value of x satisfying (x² - 4x + a) + |x - 3| ≤ 5 is 3, prove that a = 8 -/
theorem max_x_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, (x^2 - 4*x + a) + |x - 3| ≤ 5 → x ≤ 3) ∧ 
  (∃ x : ℝ, x = 3 ∧ (x^2 - 4*x + a) + |x - 3| = 5) → 
  a = 8 := by
sorry

end max_x_implies_a_value_l3001_300184


namespace log_relation_l3001_300183

theorem log_relation (a b : ℝ) : 
  a = Real.log 343 / Real.log 6 → 
  b = Real.log 18 / Real.log 7 → 
  a = 6 / (b + 2 * Real.log 2 / Real.log 7) := by
sorry

end log_relation_l3001_300183


namespace gcf_of_48_180_120_l3001_300168

theorem gcf_of_48_180_120 : Nat.gcd 48 (Nat.gcd 180 120) = 12 := by
  sorry

end gcf_of_48_180_120_l3001_300168


namespace arithmetic_sequence_sum_l3001_300105

/-- Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
    prove that S₉ = 81 when a₂ = 3 and S₄ = 16. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →
  (∀ n, a (n + 1) - a n = a 2 - a 1) →
  a 2 = 3 →
  S 4 = 16 →
  S 9 = 81 := by
sorry

end arithmetic_sequence_sum_l3001_300105


namespace suit_pants_cost_l3001_300194

theorem suit_pants_cost (budget initial_budget remaining : ℕ) 
  (shirt_cost coat_cost socks_cost belt_cost shoes_cost : ℕ) :
  initial_budget = 200 →
  shirt_cost = 30 →
  coat_cost = 38 →
  socks_cost = 11 →
  belt_cost = 18 →
  shoes_cost = 41 →
  remaining = 16 →
  ∃ (pants_cost : ℕ),
    pants_cost = initial_budget - (shirt_cost + coat_cost + socks_cost + belt_cost + shoes_cost + remaining) ∧
    pants_cost = 46 :=
by sorry

end suit_pants_cost_l3001_300194


namespace card_game_unfair_l3001_300114

/-- Represents a playing card with a rank and suit -/
structure Card :=
  (rank : Nat)
  (suit : Nat)

/-- Represents the deck of cards -/
def Deck : Finset Card := sorry

/-- The number of cards in the deck -/
def deckSize : Nat := Finset.card Deck

/-- Volodya's draw -/
def volodyaDraw : Deck → Card := sorry

/-- Masha's draw -/
def mashaDraw : Deck → Card → Card := sorry

/-- Masha wins if her card rank is higher than Volodya's -/
def mashaWins (vCard mCard : Card) : Prop := mCard.rank > vCard.rank

/-- The probability of Masha winning -/
def probMashaWins : ℝ := sorry

/-- Theorem: The card game is unfair (biased against Masha) -/
theorem card_game_unfair : probMashaWins < 1/2 := by sorry

end card_game_unfair_l3001_300114


namespace money_distribution_l3001_300189

/-- Given the ratios of money between Ram and Gopal (7:17) and between Gopal and Krishan (7:17),
    and that Ram has Rs. 686, prove that Krishan has Rs. 4046. -/
theorem money_distribution (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 686 →
  krishan = 4046 := by
sorry

end money_distribution_l3001_300189


namespace probability_of_two_distinct_roots_l3001_300137

/-- Represents the outcome of rolling two dice where at least one die shows 4 -/
inductive DiceRoll
  | first_four (second : Nat)
  | second_four (first : Nat)
  | both_four

/-- The set of all possible outcomes when rolling two dice and at least one is 4 -/
def all_outcomes : Finset DiceRoll :=
  sorry

/-- Checks if the quadratic equation x^2 + mx + n = 0 has two distinct real roots -/
def has_two_distinct_roots (roll : DiceRoll) : Bool :=
  sorry

/-- The set of outcomes where the equation has two distinct real roots -/
def favorable_outcomes : Finset DiceRoll :=
  sorry

theorem probability_of_two_distinct_roots :
  (Finset.card favorable_outcomes) / (Finset.card all_outcomes) = 5 / 11 :=
sorry

end probability_of_two_distinct_roots_l3001_300137


namespace expression_value_l3001_300185

theorem expression_value (a b c d x : ℝ) : 
  (a = b) →  -- a and -b are opposite numbers
  (c * d = -1) →  -- c and -d are reciprocals
  (abs x = 3) →  -- absolute value of x is 3
  (x^3 + c * d * x^2 - (a - b) / 2 = 18 ∨ x^3 + c * d * x^2 - (a - b) / 2 = -36) :=
by sorry

end expression_value_l3001_300185


namespace specific_tetrahedron_volume_l3001_300133

/-- Tetrahedron PQRS with given properties -/
structure Tetrahedron where
  -- Edge length QR
  qr : ℝ
  -- Area of face PQR
  area_pqr : ℝ
  -- Area of face QRS
  area_qrs : ℝ
  -- Angle between faces PQR and QRS (in radians)
  angle_pqr_qrs : ℝ

/-- The volume of the tetrahedron PQRS -/
def tetrahedron_volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the specific tetrahedron -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    qr := 15,
    area_pqr := 150,
    area_qrs := 90,
    angle_pqr_qrs := π / 4  -- 45° in radians
  }
  tetrahedron_volume t = 300 * Real.sqrt 2 := by sorry

end specific_tetrahedron_volume_l3001_300133


namespace solve_for_a_l3001_300166

theorem solve_for_a : ∃ a : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = -3 → a * x - y = 1) → a = -2 := by
  sorry

end solve_for_a_l3001_300166


namespace log_eight_x_equals_three_point_two_five_l3001_300159

theorem log_eight_x_equals_three_point_two_five (x : ℝ) :
  Real.log x / Real.log 8 = 3.25 → x = 32 * (2 : ℝ)^(1/4) := by
  sorry

end log_eight_x_equals_three_point_two_five_l3001_300159


namespace complex_minimum_value_l3001_300169

theorem complex_minimum_value (z : ℂ) (h : Complex.abs (z - (5 + I)) = 5) :
  Complex.abs (z - (1 - 2*I))^2 + Complex.abs (z - (9 + 4*I))^2 = 100 :=
by sorry

end complex_minimum_value_l3001_300169


namespace sqrt_mixed_number_simplification_l3001_300173

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 := by
  sorry

end sqrt_mixed_number_simplification_l3001_300173


namespace quadratic_root_implies_k_l3001_300112

theorem quadratic_root_implies_k (k : ℝ) : 
  (1 : ℝ)^2 + k*(1 : ℝ) - 3 = 0 → k = 2 := by
  sorry

end quadratic_root_implies_k_l3001_300112


namespace least_possible_y_l3001_300182

/-- Given that x is an even integer, y and z are odd integers,
    y - x > 5, and the least possible value of z - x is 9,
    prove that the least possible value of y is 7. -/
theorem least_possible_y (x y z : ℤ) 
  (h_x_even : Even x)
  (h_y_odd : Odd y)
  (h_z_odd : Odd z)
  (h_y_minus_x : y - x > 5)
  (h_z_minus_x_min : ∀ w, z - x ≤ w - x → w - x ≥ 9) :
  y ≥ 7 ∧ ∀ w, (Odd w ∧ w - x > 5) → y ≤ w := by
  sorry

end least_possible_y_l3001_300182


namespace horner_v1_value_l3001_300170

def horner_polynomial (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

def horner_v1 (x : ℝ) : ℝ := x * 1 - 5

theorem horner_v1_value :
  horner_v1 (-2) = -7 :=
by sorry

end horner_v1_value_l3001_300170


namespace smallest_multiple_thirty_two_satisfies_smallest_satisfying_integer_l3001_300165

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 900 * x % 640 = 0 → x ≥ 32 := by
  sorry

theorem thirty_two_satisfies : 900 * 32 % 640 = 0 := by
  sorry

theorem smallest_satisfying_integer : ∃! x : ℕ, x > 0 ∧ 900 * x % 640 = 0 ∧ ∀ y : ℕ, (y > 0 ∧ 900 * y % 640 = 0 → y ≥ x) := by
  sorry

end smallest_multiple_thirty_two_satisfies_smallest_satisfying_integer_l3001_300165
