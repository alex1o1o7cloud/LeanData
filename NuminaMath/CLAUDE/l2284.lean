import Mathlib

namespace NUMINAMATH_CALUDE_decimal_to_fraction_035_l2284_228405

def decimal_to_fraction (d : ℚ) : ℕ × ℕ := sorry

theorem decimal_to_fraction_035 :
  (decimal_to_fraction 0.35).1 = 7 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_035_l2284_228405


namespace NUMINAMATH_CALUDE_complex_roots_to_real_pair_l2284_228477

theorem complex_roots_to_real_pair :
  ∀ (a b : ℝ),
  (Complex.I : ℂ) ^ 2 = -1 →
  (a + 3 * Complex.I) * (a + 3 * Complex.I) - (12 + 15 * Complex.I) * (a + 3 * Complex.I) + (50 + 29 * Complex.I) = 0 →
  (b + 6 * Complex.I) * (b + 6 * Complex.I) - (12 + 15 * Complex.I) * (b + 6 * Complex.I) + (50 + 29 * Complex.I) = 0 →
  a = 5 / 3 ∧ b = 31 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_to_real_pair_l2284_228477


namespace NUMINAMATH_CALUDE_find_h_of_x_l2284_228430

theorem find_h_of_x (x : ℝ) (h : ℝ → ℝ) : 
  (9 * x^3 - 3 * x + 1 + h x = 3 * x^2 - 5 * x + 3) → 
  (h x = -9 * x^3 + 3 * x^2 - 2 * x + 2) := by
sorry

end NUMINAMATH_CALUDE_find_h_of_x_l2284_228430


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l2284_228495

/-- 
Proves that the profit percent is 32% when selling an article at a certain price, 
given that selling at 2/3 of that price results in a 12% loss.
-/
theorem profit_percent_calculation 
  (P : ℝ) -- The selling price
  (C : ℝ) -- The cost price
  (h : (2/3) * P = 0.88 * C) -- Condition: selling at 2/3 of P results in a 12% loss
  : (P - C) / C * 100 = 32 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l2284_228495


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2284_228466

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 4 < 0 ↔ -1 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2284_228466


namespace NUMINAMATH_CALUDE_sum_of_palindromic_primes_less_than_70_l2284_228437

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def reverseDigits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def isPalindromicPrime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 70 ∧ isPrime n ∧ isPrime (reverseDigits n) ∧ reverseDigits n < 70

def sumOfPalindromicPrimes : ℕ := sorry

theorem sum_of_palindromic_primes_less_than_70 :
  sumOfPalindromicPrimes = 92 := by sorry

end NUMINAMATH_CALUDE_sum_of_palindromic_primes_less_than_70_l2284_228437


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l2284_228435

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (bounces : ℕ) : ℝ :=
  let descendDistances := (List.range (bounces + 1)).map (λ i => initialHeight * bounceRatio^i)
  let ascendDistances := (List.range bounces).map (λ i => initialHeight * bounceRatio^(i+1))
  (descendDistances.sum + ascendDistances.sum)

/-- Theorem: A ball dropped from 25 meters, bouncing 2/3 of its previous height each time,
    and caught after the 4th bounce, travels approximately 88 meters. -/
theorem bouncing_ball_distance :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |totalDistance 25 (2/3) 4 - 88| < ε :=
sorry

end NUMINAMATH_CALUDE_bouncing_ball_distance_l2284_228435


namespace NUMINAMATH_CALUDE_community_center_tables_l2284_228425

/-- The number of chairs per table -/
def chairs_per_table : ℕ := 8

/-- The number of legs per chair -/
def legs_per_chair : ℕ := 4

/-- The number of legs per table -/
def legs_per_table : ℕ := 3

/-- The total number of legs from all chairs and tables -/
def total_legs : ℕ := 759

/-- The number of tables in the community center -/
def num_tables : ℕ := 22

theorem community_center_tables :
  chairs_per_table * num_tables * legs_per_chair + num_tables * legs_per_table = total_legs :=
sorry

end NUMINAMATH_CALUDE_community_center_tables_l2284_228425


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2284_228462

theorem arithmetic_sequence_length : 
  ∀ (a₁ : ℕ) (aₙ : ℕ) (d : ℕ),
    a₁ = 4 →
    aₙ = 130 →
    d = 2 →
    (aₙ - a₁) / d + 1 = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2284_228462


namespace NUMINAMATH_CALUDE_keith_card_spending_l2284_228463

/-- Represents the total amount spent on trading cards -/
def total_spent (digimon_packs pokemon_packs yugioh_packs magic_packs : ℕ) 
  (digimon_price pokemon_price yugioh_price magic_price baseball_price : ℚ) : ℚ :=
  digimon_packs * digimon_price + 
  pokemon_packs * pokemon_price + 
  yugioh_packs * yugioh_price + 
  magic_packs * magic_price + 
  baseball_price

/-- Theorem stating the total amount Keith spent on cards -/
theorem keith_card_spending :
  total_spent 4 3 6 2 4.45 5.25 3.99 6.75 6.06 = 77.05 := by
  sorry

end NUMINAMATH_CALUDE_keith_card_spending_l2284_228463


namespace NUMINAMATH_CALUDE_charity_fundraising_l2284_228455

theorem charity_fundraising 
  (total_amount : ℝ) 
  (sponsor_contribution : ℝ) 
  (number_of_people : ℕ) :
  total_amount = 2400 →
  sponsor_contribution = 300 →
  number_of_people = 8 →
  (total_amount - sponsor_contribution) / number_of_people = 262.5 := by
sorry

end NUMINAMATH_CALUDE_charity_fundraising_l2284_228455


namespace NUMINAMATH_CALUDE_trees_planted_today_is_41_l2284_228457

-- Define the initial number of trees
def initial_trees : Nat := 39

-- Define the number of trees to be planted tomorrow
def trees_tomorrow : Nat := 20

-- Define the final number of trees
def final_trees : Nat := 100

-- Define the number of trees planted today
def trees_planted_today : Nat := final_trees - initial_trees - trees_tomorrow

-- Theorem to prove
theorem trees_planted_today_is_41 : trees_planted_today = 41 := by
  sorry

end NUMINAMATH_CALUDE_trees_planted_today_is_41_l2284_228457


namespace NUMINAMATH_CALUDE_distance_swum_back_l2284_228456

/-- Calculates the distance swum against the current given swimming speed, water speed, and time -/
def distance_against_current (swimming_speed water_speed : ℝ) (time : ℝ) : ℝ :=
  (swimming_speed - water_speed) * time

/-- Proves that the distance swum against the current is 8 km given the specified conditions -/
theorem distance_swum_back (swimming_speed water_speed time : ℝ) 
  (h1 : swimming_speed = 12)
  (h2 : water_speed = 10)
  (h3 : time = 4) :
  distance_against_current swimming_speed water_speed time = 8 := by
  sorry

#check distance_swum_back

end NUMINAMATH_CALUDE_distance_swum_back_l2284_228456


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l2284_228433

/-- The equation (x+y)^2 = x^2 + y^2 + 4 represents a hyperbola in the xy-plane. -/
theorem equation_represents_hyperbola :
  ∃ (f : ℝ → ℝ → Prop), (∀ x y : ℝ, f x y ↔ (x + y)^2 = x^2 + y^2 + 4) ∧
  (∃ a : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, f x y ↔ x * y = a) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l2284_228433


namespace NUMINAMATH_CALUDE_ceiling_of_negative_three_point_six_l2284_228438

theorem ceiling_of_negative_three_point_six :
  ⌈(-3.6 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_three_point_six_l2284_228438


namespace NUMINAMATH_CALUDE_probability_independent_events_l2284_228411

theorem probability_independent_events (a b : Set α) (p : Set α → ℝ) 
  (h1 : p a = 4/7)
  (h2 : p (a ∩ b) = 0.22857142857142856)
  (h3 : p (a ∩ b) = p a * p b) : 
  p b = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_probability_independent_events_l2284_228411


namespace NUMINAMATH_CALUDE_onions_sum_is_eighteen_l2284_228427

/-- The total number of onions grown by Sara, Sally, and Fred -/
def total_onions (sara_onions sally_onions fred_onions : ℕ) : ℕ :=
  sara_onions + sally_onions + fred_onions

/-- Theorem stating that the total number of onions grown is 18 -/
theorem onions_sum_is_eighteen :
  total_onions 4 5 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_onions_sum_is_eighteen_l2284_228427


namespace NUMINAMATH_CALUDE_walts_investment_l2284_228484

/-- Proves that given the conditions of Walt's investment, the unknown interest rate is 8% -/
theorem walts_investment (total_investment : ℝ) (known_rate : ℝ) (total_interest : ℝ) (unknown_investment : ℝ) :
  total_investment = 9000 →
  known_rate = 0.09 →
  total_interest = 770 →
  unknown_investment = 4000 →
  ∃ (unknown_rate : ℝ),
    unknown_rate * unknown_investment + known_rate * (total_investment - unknown_investment) = total_interest ∧
    unknown_rate = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_walts_investment_l2284_228484


namespace NUMINAMATH_CALUDE_inequality_not_hold_l2284_228415

theorem inequality_not_hold (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(1 / (a - b) > 1 / a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_hold_l2284_228415


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l2284_228439

/-- The radius of a circle tangent to eight semicircles lining the inside of a square --/
theorem tangent_circle_radius (square_side : ℝ) (h : square_side = 4) :
  let semicircle_radius : ℝ := square_side / 4
  let diagonal : ℝ := Real.sqrt (square_side ^ 2 / 4 + (square_side / 4) ^ 2)
  diagonal - semicircle_radius = Real.sqrt 5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l2284_228439


namespace NUMINAMATH_CALUDE_reshuffling_theorem_l2284_228406

def total_employees : ℕ := 10000

def current_proportions : List (String × ℚ) := [
  ("Senior Managers", 2/5),
  ("Junior Managers", 3/10),
  ("Engineers", 1/5),
  ("Marketing Team", 1/10)
]

def desired_proportions : List (String × ℚ) := [
  ("Senior Managers", 7/20),
  ("Junior Managers", 1/5),
  ("Engineers", 1/4),
  ("Marketing Team", 1/5)
]

def calculate_changes (current : List (String × ℚ)) (desired : List (String × ℚ)) (total : ℕ) : 
  List (String × ℤ) :=
  sorry

theorem reshuffling_theorem : 
  calculate_changes current_proportions desired_proportions total_employees = 
    [("Senior Managers", -500), 
     ("Junior Managers", -1000), 
     ("Engineers", 500), 
     ("Marketing Team", 1000)] :=
by sorry

end NUMINAMATH_CALUDE_reshuffling_theorem_l2284_228406


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l2284_228465

theorem average_of_three_numbers (a : ℝ) : 
  (3 + a + 10) / 3 = 5 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l2284_228465


namespace NUMINAMATH_CALUDE_projection_area_l2284_228407

/-- A polygon in 3D space -/
structure Polygon3D where
  -- Define the polygon structure (this is a simplification)
  area : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  -- Define the plane structure (this is a simplification)

/-- The angle between two planes -/
def angle_between_planes (p1 p2 : Plane3D) : ℝ := sorry

/-- The projection of a polygon onto a plane -/
def project_polygon (poly : Polygon3D) (plane : Plane3D) : Polygon3D := sorry

/-- Theorem: The area of a polygon's projection is the original area times the cosine of the angle between planes -/
theorem projection_area (poly : Polygon3D) (plane : Plane3D) : 
  (project_polygon poly plane).area = poly.area * Real.cos (angle_between_planes (Plane3D.mk) plane) := by
  sorry

end NUMINAMATH_CALUDE_projection_area_l2284_228407


namespace NUMINAMATH_CALUDE_roundness_of_eight_million_l2284_228445

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 8,000,000 is 15 -/
theorem roundness_of_eight_million : roundness 8000000 = 15 := by sorry

end NUMINAMATH_CALUDE_roundness_of_eight_million_l2284_228445


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_60_l2284_228444

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the general term of the expansion
def generalTerm (r : ℕ) : ℚ :=
  (-1)^r * binomial 6 r * 2^r

-- Theorem statement
theorem coefficient_x_squared_is_60 :
  generalTerm 2 = 60 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_60_l2284_228444


namespace NUMINAMATH_CALUDE_remainder_theorem_l2284_228491

theorem remainder_theorem (m : ℤ) (h : m % 9 = 3) : (3 * m + 2436) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2284_228491


namespace NUMINAMATH_CALUDE_balance_weights_l2284_228483

def weights : List ℕ := [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def target_weight : ℕ := 1998

theorem balance_weights :
  (∀ w ∈ weights, is_power_of_two w) →
  (∃ subset : List ℕ, subset.Subset weights ∧ subset.sum = target_weight ∧ subset.length = 8) ∧
  (∀ subset : List ℕ, subset.Subset weights → subset.sum = target_weight → subset.length ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_balance_weights_l2284_228483


namespace NUMINAMATH_CALUDE_solution_count_l2284_228460

/-- The number of positive integer solutions to the equation 3x + 4y = 1024 -/
def num_solutions : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 4 * p.2 = 1024 ∧ p.1 > 0 ∧ p.2 > 0)
    (Finset.product (Finset.range 1025) (Finset.range 1025))).card

theorem solution_count : num_solutions = 85 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_l2284_228460


namespace NUMINAMATH_CALUDE_tank_fill_time_l2284_228434

/-- Represents the time (in hours) it takes for a pipe to empty or fill the tank when working alone -/
structure PipeTime where
  A : ℝ  -- Time for pipe A to empty the tank
  B : ℝ  -- Time for pipe B to empty the tank
  C : ℝ  -- Time for pipe C to fill the tank

/-- Conditions for the tank filling problem -/
def TankConditions (t : PipeTime) : Prop :=
  (1 / t.C - 1 / t.A) * 2 = 1 ∧
  (1 / t.C - 1 / t.B) * 4 = 1 ∧
  1 / t.C * 5 - (1 / t.A + 1 / t.B) * 8 = 0

/-- The main theorem stating the time to fill the tank using only pipe C -/
theorem tank_fill_time (t : PipeTime) (h : TankConditions t) : t.C = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l2284_228434


namespace NUMINAMATH_CALUDE_units_digit_of_2_power_2018_l2284_228409

theorem units_digit_of_2_power_2018 : ∃ (f : ℕ → ℕ), 
  (∀ n, f n = n % 4) ∧ 
  (∀ n, n > 0 → (2^n % 10 = 2 ∨ 2^n % 10 = 4 ∨ 2^n % 10 = 8 ∨ 2^n % 10 = 6)) ∧
  (2^2018 % 10 = 4) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_power_2018_l2284_228409


namespace NUMINAMATH_CALUDE_share_calculation_l2284_228480

/-- Proves that given Debby takes 25% of a total sum and Maggie takes the rest,
if Maggie's share is $4,500, then the total sum is $6,000. -/
theorem share_calculation (total : ℝ) (debby_share : ℝ) (maggie_share : ℝ) : 
  debby_share = 0.25 * total →
  maggie_share = total - debby_share →
  maggie_share = 4500 →
  total = 6000 := by
  sorry

end NUMINAMATH_CALUDE_share_calculation_l2284_228480


namespace NUMINAMATH_CALUDE_exactly_one_correct_probability_l2284_228487

theorem exactly_one_correct_probability
  (probA : ℝ) (probB : ℝ) (probC : ℝ)
  (hprobA : probA = 3/4)
  (hprobB : probB = 2/3)
  (hprobC : probC = 2/3)
  (hprobA_bounds : 0 ≤ probA ∧ probA ≤ 1)
  (hprobB_bounds : 0 ≤ probB ∧ probB ≤ 1)
  (hprobC_bounds : 0 ≤ probC ∧ probC ≤ 1) :
  probA * (1 - probB) * (1 - probC) +
  (1 - probA) * probB * (1 - probC) +
  (1 - probA) * (1 - probB) * probC = 7/36 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_correct_probability_l2284_228487


namespace NUMINAMATH_CALUDE_radical_axis_properties_l2284_228400

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def power (p : ℝ × ℝ) (c : Circle) : ℝ :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 - c.radius^2

def perpendicular (l1 l2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let ((x1, y1), (x2, y2)) := l1
  let ((x3, y3), (x4, y4)) := l2
  (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3) = 0

def on_line (p : ℝ × ℝ) (l : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (x, y) := p
  let ((x1, y1), (x2, y2)) := l
  (y2 - y1) * (x - x1) = (x2 - x1) * (y - y1)

-- Define the theorem
theorem radical_axis_properties 
  (k₁ k₂ : Circle) 
  (P Q : ℝ × ℝ) 
  (h_non_intersect : k₁ ≠ k₂) 
  (h_power_P : power P k₁ = power P k₂)
  (h_power_Q : power Q k₁ = power Q k₂) :
  let O₁ := k₁.center
  let O₂ := k₂.center
  (perpendicular (P, Q) (O₁, O₂)) ∧ 
  (∀ S, (power S k₁ = power S k₂) ↔ on_line S (P, Q)) ∧
  (∀ k : Circle, 
    (∃ x y, power (x, y) k = power (x, y) k₁ ∧ power (x, y) k = power (x, y) k₂) →
    (∃ M, (power M k = power M k₁) ∧ (power M k = power M k₂) ∧ 
          (on_line M (P, Q) ∨ perpendicular (P, Q) (k.center, M)))) := by
  sorry

end NUMINAMATH_CALUDE_radical_axis_properties_l2284_228400


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l2284_228453

theorem greatest_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), y - 5 > 4*y - 1 → y ≤ x) ∧ (x - 5 > 4*x - 1) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l2284_228453


namespace NUMINAMATH_CALUDE_rocky_path_trail_length_l2284_228401

/-- Represents the length of Phoenix's hike on the Rocky Path Trail -/
def rocky_path_trail (a b c d e : ℝ) : Prop :=
  a + b = 24 ∧
  b + c = 28 ∧
  c + d + e = 36 ∧
  a + c = 28

theorem rocky_path_trail_length :
  ∀ a b c d e : ℝ, rocky_path_trail a b c d e → a + b + c + d + e = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_rocky_path_trail_length_l2284_228401


namespace NUMINAMATH_CALUDE_min_value_circle_l2284_228486

theorem min_value_circle (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  ∃ (m : ℝ), (∀ (a b : ℝ), a^2 + b^2 - 4*a + 1 = 0 → x^2 + y^2 ≤ a^2 + b^2) ∧ 
  m = x^2 + y^2 ∧ m = 7 - 4*Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_circle_l2284_228486


namespace NUMINAMATH_CALUDE_girls_exceed_boys_by_69_l2284_228403

/-- Proves that in a class of 485 students with 208 boys, the number of girls exceeds the number of boys by 69 -/
theorem girls_exceed_boys_by_69 :
  let total_students : ℕ := 485
  let num_boys : ℕ := 208
  let num_girls : ℕ := total_students - num_boys
  num_girls - num_boys = 69 := by sorry

end NUMINAMATH_CALUDE_girls_exceed_boys_by_69_l2284_228403


namespace NUMINAMATH_CALUDE_probability_same_number_l2284_228404

def emily_options : ℕ := 250 / 20
def eli_options : ℕ := 250 / 30
def common_options : ℕ := 250 / 60

theorem probability_same_number : 
  (emily_options : ℚ) * eli_options ≠ 0 →
  (common_options : ℚ) / (emily_options * eli_options) = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_number_l2284_228404


namespace NUMINAMATH_CALUDE_find_n_l2284_228431

theorem find_n : ∃ n : ℕ, 
  50 ≤ n ∧ n ≤ 120 ∧ 
  n % 8 = 0 ∧ 
  n % 12 = 4 ∧ 
  n % 7 = 4 ∧
  n = 88 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l2284_228431


namespace NUMINAMATH_CALUDE_digits_of_3_pow_15_times_5_pow_10_l2284_228410

theorem digits_of_3_pow_15_times_5_pow_10 : 
  (Nat.log 10 (3^15 * 5^10) + 1 : ℕ) = 18 :=
sorry

end NUMINAMATH_CALUDE_digits_of_3_pow_15_times_5_pow_10_l2284_228410


namespace NUMINAMATH_CALUDE_unique_bases_sum_l2284_228459

theorem unique_bases_sum : ∃! (R₃ R₄ : ℕ), 
  (R₃ > 0 ∧ R₄ > 0) ∧
  ((4 * R₃ + 6) * (R₄^2 - 1) = (4 * R₄ + 9) * (R₃^2 - 1)) ∧
  ((6 * R₃ + 4) * (R₄^2 - 1) = (9 * R₄ + 4) * (R₃^2 - 1)) ∧
  (R₃ + R₄ = 23) := by
  sorry

end NUMINAMATH_CALUDE_unique_bases_sum_l2284_228459


namespace NUMINAMATH_CALUDE_parallelogram_probability_theorem_l2284_228408

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- The probability of a point in a parallelogram being not below the x-axis -/
def probability_not_below_x_axis (para : Parallelogram) : ℝ := 
  sorry

theorem parallelogram_probability_theorem (para : Parallelogram) :
  para.P = Point.mk 4 4 →
  para.Q = Point.mk (-2) (-2) →
  para.R = Point.mk (-8) (-2) →
  para.S = Point.mk (-2) 4 →
  probability_not_below_x_axis para = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_probability_theorem_l2284_228408


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l2284_228458

theorem sum_of_roots_cubic_equation : 
  let f (x : ℝ) := (x^3 - 3*x^2 - 12*x) / (x + 3)
  ∃ (x₁ x₂ : ℝ), (f x₁ = 7 ∧ f x₂ = 7 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l2284_228458


namespace NUMINAMATH_CALUDE_triangle_area_combinations_l2284_228426

theorem triangle_area_combinations (a b c : ℝ) (A B C : ℝ) :
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (A > 0 ∧ B > 0 ∧ C > 0) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a = Real.sqrt 3 ∧ b = 2 ∧ (Real.sin B + Real.sin C) / Real.sin A = (a + c) / (b - c) →
    1/2 * a * c * Real.sin B = 3 * (Real.sqrt 7 - Real.sqrt 3) / 8) ∧
  (a = Real.sqrt 3 ∧ b = 2 ∧ Real.cos ((B - C) / 2)^2 - Real.sin B * Real.sin C = 1/4 →
    1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_combinations_l2284_228426


namespace NUMINAMATH_CALUDE_f_of_2_eq_0_l2284_228472

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2 - 1

-- State the theorem
theorem f_of_2_eq_0 : f 2 = 0 := by sorry

end NUMINAMATH_CALUDE_f_of_2_eq_0_l2284_228472


namespace NUMINAMATH_CALUDE_coffee_break_theorem_coffee_break_converse_l2284_228424

/-- Represents the number of participants who went for coffee -/
def coffee_drinkers : Finset ℕ := {6, 8, 10, 12}

/-- Represents the total number of participants -/
def total_participants : ℕ := 14

theorem coffee_break_theorem (n : ℕ) (hn : n ∈ coffee_drinkers) :
  ∃ (k : ℕ),
    -- k represents the number of pairs of participants who stayed
    0 < k ∧ 
    k < total_participants / 2 ∧
    -- n is the number of participants who left
    n = total_participants - 2 * k ∧
    -- Each remaining participant has exactly one neighbor who left
    ∀ (i : ℕ), i < total_participants → 
      (i % 2 = 0 → (i + 1) % total_participants < n) ∧
      (i % 2 = 1 → i < n) :=
by sorry

theorem coffee_break_converse :
  ∀ (n : ℕ),
    (∃ (k : ℕ),
      0 < k ∧ 
      k < total_participants / 2 ∧
      n = total_participants - 2 * k ∧
      ∀ (i : ℕ), i < total_participants → 
        (i % 2 = 0 → (i + 1) % total_participants < n) ∧
        (i % 2 = 1 → i < n)) →
    n ∈ coffee_drinkers :=
by sorry

end NUMINAMATH_CALUDE_coffee_break_theorem_coffee_break_converse_l2284_228424


namespace NUMINAMATH_CALUDE_harriet_ran_approximately_45_miles_l2284_228499

/-- The total distance run by six runners -/
def total_distance : ℝ := 378.5

/-- The distance run by Katarina -/
def katarina_distance : ℝ := 47.5

/-- The distance run by Adriana -/
def adriana_distance : ℝ := 83.25

/-- The distance run by Jeremy -/
def jeremy_distance : ℝ := 92.75

/-- The difference in distance between Tomas, Tyler, and Harriet -/
def difference : ℝ := 6.5

/-- Harriet's approximate distance -/
def harriet_distance : ℕ := 45

theorem harriet_ran_approximately_45_miles :
  ∃ (tomas_distance tyler_distance harriet_exact_distance : ℝ),
    tomas_distance ≠ tyler_distance ∧
    tyler_distance ≠ harriet_exact_distance ∧
    tomas_distance ≠ harriet_exact_distance ∧
    (tomas_distance = tyler_distance + difference ∨ tyler_distance = tomas_distance + difference) ∧
    (tyler_distance = harriet_exact_distance + difference ∨ harriet_exact_distance = tyler_distance + difference) ∧
    tomas_distance + tyler_distance + harriet_exact_distance + katarina_distance + adriana_distance + jeremy_distance = total_distance ∧
    harriet_distance = round harriet_exact_distance :=
by
  sorry

end NUMINAMATH_CALUDE_harriet_ran_approximately_45_miles_l2284_228499


namespace NUMINAMATH_CALUDE_triangle_max_area_l2284_228476

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  (2 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C →
  ∀ (area : ℝ), area = (1/2) * b * c * Real.sin A → area ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2284_228476


namespace NUMINAMATH_CALUDE_odd_k_triple_f_81_l2284_228418

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then 2 * n + 3
  else if n % 3 = 0 ∧ n % 2 ≠ 0 then n / 3
  else n  -- This case is not specified in the original problem, so we leave n unchanged

theorem odd_k_triple_f_81 (k : ℤ) (h_odd : k % 2 = 1) (h_triple_f : f (f (f k)) = 81) : k = 57 := by
  sorry

end NUMINAMATH_CALUDE_odd_k_triple_f_81_l2284_228418


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l2284_228428

theorem unique_x_with_three_prime_divisors (x n : ℕ) : 
  x = 8^n - 1 →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 31 ∧ q ≠ 31 ∧ x = 31 * p * q) →
  (∀ r : ℕ, Prime r ∧ r ∣ x → r = 31 ∨ r = p ∨ r = q) →
  x = 32767 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l2284_228428


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_one_l2284_228482

theorem sqrt_expression_equals_one :
  1 + (Real.sqrt 2 - Real.sqrt 3) + |Real.sqrt 2 - Real.sqrt 3| = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_one_l2284_228482


namespace NUMINAMATH_CALUDE_cake_division_l2284_228496

theorem cake_division (x y z : ℚ) :
  x + y + z = 1 →
  2 * z = x →
  z = (1/2) * (y + (2/3) * x) →
  (2/3) * x = 4/11 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_division_l2284_228496


namespace NUMINAMATH_CALUDE_recipe_eggs_l2284_228449

theorem recipe_eggs (total_eggs : ℕ) (rotten_eggs : ℕ) (prob_all_rotten : ℝ) :
  total_eggs = 36 →
  rotten_eggs = 3 →
  prob_all_rotten = 0.0047619047619047615 →
  ∃ (n : ℕ), (rotten_eggs : ℝ) / (total_eggs : ℝ) ^ n = prob_all_rotten ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_recipe_eggs_l2284_228449


namespace NUMINAMATH_CALUDE_factorization_equality_l2284_228478

theorem factorization_equality (a b : ℝ) : 9 * a * b - a^3 * b = a * b * (3 + a) * (3 - a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2284_228478


namespace NUMINAMATH_CALUDE_abs_neg_reciprocal_of_two_l2284_228470

theorem abs_neg_reciprocal_of_two : |-(1 / 2)| = |1 / 2| := by sorry

end NUMINAMATH_CALUDE_abs_neg_reciprocal_of_two_l2284_228470


namespace NUMINAMATH_CALUDE_parameter_range_l2284_228469

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (a * x / 3) else 3 * Real.log x / x

/-- The maximum value of f(x) on [-3, 3] is 3/e -/
axiom max_value (a : ℝ) : ∀ x ∈ Set.Icc (-3) 3, f a x ≤ 3 / Real.exp 1

/-- The range of parameter a is [1 - ln(3), +∞) -/
theorem parameter_range :
  {a : ℝ | ∀ x ∈ Set.Icc (-3) 3, f a x ≤ 3 / Real.exp 1} = Set.Ici (1 - Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_parameter_range_l2284_228469


namespace NUMINAMATH_CALUDE_medicine_A_count_l2284_228417

/-- The number of tablets of medicine B in the box -/
def medicine_B : ℕ := 16

/-- The minimum number of tablets to extract to ensure at least two of each kind -/
def min_extract : ℕ := 18

/-- The number of tablets of medicine A in the box -/
def medicine_A : ℕ := 3

theorem medicine_A_count : 
  ∀ (x : ℕ), 
  (x = medicine_A) ↔ 
  (x > 0 ∧ 
   x + medicine_B ≥ min_extract ∧ 
   x - 1 + medicine_B < min_extract) :=
by sorry

end NUMINAMATH_CALUDE_medicine_A_count_l2284_228417


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_l2284_228413

theorem imaginary_part_of_complex (z : ℂ) : z = 1 - 2*I → Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_l2284_228413


namespace NUMINAMATH_CALUDE_basis_from_noncoplanar_vectors_l2284_228443

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem basis_from_noncoplanar_vectors (a b c : V) 
  (h : LinearIndependent ℝ ![a, b, c]) :
  LinearIndependent ℝ ![a + b, b - a, c] :=
sorry

end NUMINAMATH_CALUDE_basis_from_noncoplanar_vectors_l2284_228443


namespace NUMINAMATH_CALUDE_adam_katie_miles_difference_l2284_228473

/-- Proves that Adam ran 25 miles more than Katie -/
theorem adam_katie_miles_difference :
  let adam_miles : ℕ := 35
  let katie_miles : ℕ := 10
  adam_miles - katie_miles = 25 := by
  sorry

end NUMINAMATH_CALUDE_adam_katie_miles_difference_l2284_228473


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2284_228450

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 5 * a 7 = 2) →
  (a 2 + a 10 = 3) →
  (a 12 / a 4 = 2 ∨ a 12 / a 4 = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2284_228450


namespace NUMINAMATH_CALUDE_recipe_total_cups_l2284_228489

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients used in a recipe given the ratio and cups of sugar -/
def total_cups (ratio : RecipeRatio) (sugar_cups : ℕ) : ℕ :=
  let part_size := sugar_cups / ratio.sugar
  part_size * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem stating that for a recipe with ratio 1:8:5 and 10 cups of sugar, the total cups used is 28 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := ⟨1, 8, 5⟩
  let sugar_cups : ℕ := 10
  total_cups ratio sugar_cups = 28 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l2284_228489


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_and_x_cubed_plus_reciprocal_l2284_228451

theorem x_squared_plus_reciprocal_and_x_cubed_plus_reciprocal (x : ℝ) (hx : x ≠ 0) :
  let y := x + 1/x
  (x^2 + 1/x^2 = y^2 - 2) ∧ (x^3 + 1/x^3 = y^3 - 3*y) := by
  sorry

#check x_squared_plus_reciprocal_and_x_cubed_plus_reciprocal

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_and_x_cubed_plus_reciprocal_l2284_228451


namespace NUMINAMATH_CALUDE_equation_solution_l2284_228488

theorem equation_solution (x : ℝ) : 
  Real.sqrt (9 + Real.sqrt (27 + 3*x)) + Real.sqrt (3 + Real.sqrt (9 + x)) = 3 + 3 * Real.sqrt 3 →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2284_228488


namespace NUMINAMATH_CALUDE_infinite_primes_dividing_x_l2284_228468

/-- A polynomial with non-negative integer coefficients -/
def NonNegIntPoly := ℕ → ℕ

/-- Definition of x_n -/
def x (P Q : NonNegIntPoly) (n : ℕ) : ℕ := 2016^(P n) + Q n

/-- A number is squarefree if it's not divisible by any prime square -/
def IsSquarefree (m : ℕ) : Prop := ∀ p : ℕ, Nat.Prime p → (p^2 ∣ m) → False

theorem infinite_primes_dividing_x (P Q : NonNegIntPoly) 
  (hP : ¬ ∀ n : ℕ, P n = P 0) 
  (hQ : ¬ ∀ n : ℕ, Q n = Q 0) : 
  ∃ S : Set ℕ, (S.Infinite) ∧ 
  (∀ p ∈ S, Nat.Prime p ∧ ∃ m : ℕ, IsSquarefree m ∧ (p ∣ x P Q m)) := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_dividing_x_l2284_228468


namespace NUMINAMATH_CALUDE_algorithm_swaps_values_l2284_228423

-- Define the algorithm steps
def algorithm (x y : ℝ) : ℝ × ℝ :=
  let z := x
  let x' := y
  let y' := z
  (x', y')

-- Theorem statement
theorem algorithm_swaps_values (x y : ℝ) :
  algorithm x y = (y, x) := by sorry

end NUMINAMATH_CALUDE_algorithm_swaps_values_l2284_228423


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2284_228402

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let special_number := 1 + 1 / n
  let regular_number := 1
  let sum := special_number + (n - 1) * regular_number
  sum / n = 1 + 1 / n^2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2284_228402


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l2284_228474

theorem square_garden_perimeter (q p : ℝ) (h1 : q > 0) (h2 : p > 0) (h3 : q = p + 21) :
  p = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l2284_228474


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2284_228497

theorem consecutive_integers_square_sum (n : ℕ) : 
  (n > 0) → (n^2 + (n+1)^2 = n*(n+1) + 91) → (n+1 = 10) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2284_228497


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l2284_228416

-- Define a triangle with sides and angles
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_side_calculation (t : Triangle) 
  (h1 : t.A = 30 * π / 180)
  (h2 : t.C = 45 * π / 180)
  (h3 : t.a = 4) :
  t.c = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l2284_228416


namespace NUMINAMATH_CALUDE_equation_condition_l2284_228485

theorem equation_condition (a b c d : ℝ) :
  (a^2 + b) / (b + c^2) = (c^2 + d) / (d + a^2) →
  (a = c ∨ a^2 + d + 2*b = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_condition_l2284_228485


namespace NUMINAMATH_CALUDE_not_all_triangles_divisible_to_square_l2284_228452

/-- A triangle with base and height -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- A square with side length -/
structure Square where
  side : ℝ

/-- Represents a division of a shape into parts -/
structure Division where
  parts : ℕ

/-- Represents the ability to form a shape from parts -/
def can_form (d : Division) (s : Square) : Prop := sorry

/-- The theorem stating that not all triangles can be divided into 1000 parts to form a square -/
theorem not_all_triangles_divisible_to_square :
  ∃ t : Triangle, ¬ ∃ (d : Division) (s : Square), d.parts = 1000 ∧ can_form d s := by sorry

end NUMINAMATH_CALUDE_not_all_triangles_divisible_to_square_l2284_228452


namespace NUMINAMATH_CALUDE_fifth_row_dots_l2284_228421

/-- Represents the number of green dots in a row -/
def greenDots : ℕ → ℕ
  | 0 => 3  -- First row (index 0) has 3 dots
  | n + 1 => greenDots n + 3  -- Each subsequent row increases by 3 dots

/-- The theorem stating that the fifth row has 15 green dots -/
theorem fifth_row_dots : greenDots 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifth_row_dots_l2284_228421


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l2284_228464

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  e : ℝ  -- eccentricity
  a : ℝ  -- semi-major axis
  f1 : Point  -- left focus
  f2 : Point  -- right focus

/-- Represents a parabola -/
structure Parabola where
  vertex : Point
  focus : Point

/-- The main theorem -/
theorem hyperbola_parabola_intersection (h : Hyperbola) (p : Parabola) (P : Point) (a c : ℝ) :
  h.f1 = p.focus →
  h.f2 = p.vertex →
  (P.x - h.f2.x) ^ 2 + P.y ^ 2 = (h.e * h.a) ^ 2 →  -- P is on the right branch of the hyperbola
  P.y ^ 2 = 2 * h.a * (P.x - h.f2.x) →  -- P is on the parabola
  a * |P.x - h.f2.x| + c * |h.f1.x - P.x| = 8 * a ^ 2 →
  h.e = 8 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l2284_228464


namespace NUMINAMATH_CALUDE_max_value_expression_l2284_228475

theorem max_value_expression (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) : 
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) * (x - y + z) ≤ 2187/216 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2284_228475


namespace NUMINAMATH_CALUDE_fifth_number_eighth_row_l2284_228490

/-- Represents the end number of the n-th row in the table -/
def end_of_row (n : ℕ) : ℕ := n * n

/-- Represents the first number in the n-th row -/
def start_of_row (n : ℕ) : ℕ := end_of_row (n - 1) + 1

/-- The theorem stating that the 5th number from the left in the 8th row is 54 -/
theorem fifth_number_eighth_row : start_of_row 8 + 4 = 54 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_eighth_row_l2284_228490


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l2284_228429

/-- The focal length of the ellipse 2x^2 + 3y^2 = 6 is 2 -/
theorem ellipse_focal_length : 
  let ellipse := {(x, y) : ℝ × ℝ | 2 * x^2 + 3 * y^2 = 6}
  ∃ (f : ℝ), f = 2 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ ellipse → 
      ∃ (c₁ c₂ : ℝ × ℝ), 
        (c₁.1 - x)^2 + (c₁.2 - y)^2 + (c₂.1 - x)^2 + (c₂.2 - y)^2 = f^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l2284_228429


namespace NUMINAMATH_CALUDE_probability_two_females_selected_l2284_228432

def total_contestants : ℕ := 7
def female_contestants : ℕ := 4
def male_contestants : ℕ := 3

theorem probability_two_females_selected :
  (Nat.choose female_contestants 2 : ℚ) / (Nat.choose total_contestants 2 : ℚ) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_females_selected_l2284_228432


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l2284_228422

theorem two_digit_number_problem : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10) * 2 = (n % 10) * 3 ∧
  (n / 10) = (n % 10) + 3 ∧
  n = 63 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l2284_228422


namespace NUMINAMATH_CALUDE_carpet_square_cost_l2284_228419

/-- Calculates the cost of each carpet square given the floor dimensions, carpet square dimensions, and total cost. -/
theorem carpet_square_cost
  (floor_length : ℝ)
  (floor_width : ℝ)
  (square_side : ℝ)
  (total_cost : ℝ)
  (h1 : floor_length = 24)
  (h2 : floor_width = 64)
  (h3 : square_side = 8)
  (h4 : total_cost = 576)
  : (total_cost / ((floor_length * floor_width) / (square_side * square_side))) = 24 :=
by
  sorry

#check carpet_square_cost

end NUMINAMATH_CALUDE_carpet_square_cost_l2284_228419


namespace NUMINAMATH_CALUDE_complement_union_problem_l2284_228481

def U : Set ℝ := {-2, -8, 0, Real.pi, 6, 10}
def A : Set ℝ := {-2, Real.pi, 6}
def B : Set ℝ := {1}

theorem complement_union_problem : (U \ A) ∪ B = {0, 1, -8, 10} := by sorry

end NUMINAMATH_CALUDE_complement_union_problem_l2284_228481


namespace NUMINAMATH_CALUDE_largest_x_satisfying_equation_l2284_228420

theorem largest_x_satisfying_equation : ∃ (x : ℝ), 
  (∀ y : ℝ, ⌊y⌋ / y = 8 / 9 → y ≤ x) ∧ 
  ⌊x⌋ / x = 8 / 9 ∧ 
  x = 63 / 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_satisfying_equation_l2284_228420


namespace NUMINAMATH_CALUDE_prob_three_primes_l2284_228461

def num_dice : ℕ := 6
def sides_per_die : ℕ := 12
def prob_prime : ℚ := 5/12

theorem prob_three_primes :
  let choose_three := Nat.choose num_dice 3
  let prob_three_prime := (prob_prime ^ 3 : ℚ)
  let prob_three_non_prime := ((1 - prob_prime) ^ 3 : ℚ)
  choose_three * prob_three_prime * prob_three_non_prime = 312500/248832 := by
sorry

end NUMINAMATH_CALUDE_prob_three_primes_l2284_228461


namespace NUMINAMATH_CALUDE_two_statements_correct_l2284_228447

-- Define a structure for a line in 2D plane
structure Line where
  slope : Option ℝ
  angle_of_inclination : ℝ

-- Define parallel and perpendicular relations
def parallel (l₁ l₂ : Line) : Prop := sorry

def perpendicular (l₁ l₂ : Line) : Prop := sorry

-- Define the four statements
def statement1 (l₁ l₂ : Line) : Prop :=
  (l₁.slope.isSome ∧ l₂.slope.isSome ∧ l₁.slope = l₂.slope) → parallel l₁ l₂

def statement2 (l₁ l₂ : Line) : Prop :=
  perpendicular l₁ l₂ →
    (l₁.slope.isSome ∧ l₂.slope.isSome ∧
     ∃ (s₁ s₂ : ℝ), l₁.slope = some s₁ ∧ l₂.slope = some s₂ ∧ s₁ * s₂ = -1)

def statement3 (l₁ l₂ : Line) : Prop :=
  l₁.angle_of_inclination = l₂.angle_of_inclination → parallel l₁ l₂

def statement4 : Prop :=
  ∀ (l₁ l₂ : Line), parallel l₁ l₂ → (l₁.slope.isSome ∧ l₂.slope.isSome ∧ l₁.slope = l₂.slope)

theorem two_statements_correct (l₁ l₂ : Line) (h : l₁ ≠ l₂) :
  (statement1 l₁ l₂ ∧ statement3 l₁ l₂ ∧ ¬statement2 l₁ l₂ ∧ ¬statement4) := by
  sorry

end NUMINAMATH_CALUDE_two_statements_correct_l2284_228447


namespace NUMINAMATH_CALUDE_sum_distances_specific_triangle_l2284_228454

/-- The sum of distances from a point to the vertices of a triangle, expressed as x + y√z --/
def sum_distances (A B C P : ℝ × ℝ) : ℝ × ℝ × ℕ :=
  sorry

/-- Theorem stating the sum of distances for specific triangle and point --/
theorem sum_distances_specific_triangle :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (10, 2)
  let C : ℝ × ℝ := (5, 4)
  let P : ℝ × ℝ := (3, 1)
  let (x, y, z) := sum_distances A B C P
  x + y + z = 16 :=
by sorry

end NUMINAMATH_CALUDE_sum_distances_specific_triangle_l2284_228454


namespace NUMINAMATH_CALUDE_carwash_solution_l2284_228467

/-- Represents the carwash problem --/
structure CarWash where
  car_price : ℕ
  truck_price : ℕ
  suv_price : ℕ
  total_raised : ℕ
  num_suvs : ℕ
  num_trucks : ℕ

/-- Calculates the number of cars washed --/
def cars_washed (cw : CarWash) : ℕ :=
  (cw.total_raised - (cw.suv_price * cw.num_suvs + cw.truck_price * cw.num_trucks)) / cw.car_price

/-- Theorem stating the solution to the carwash problem --/
theorem carwash_solution (cw : CarWash) 
  (h1 : cw.car_price = 5)
  (h2 : cw.truck_price = 6)
  (h3 : cw.suv_price = 7)
  (h4 : cw.total_raised = 100)
  (h5 : cw.num_suvs = 5)
  (h6 : cw.num_trucks = 5) :
  cars_washed cw = 7 := by
  sorry

#eval cars_washed ⟨5, 6, 7, 100, 5, 5⟩

end NUMINAMATH_CALUDE_carwash_solution_l2284_228467


namespace NUMINAMATH_CALUDE_group_size_proof_l2284_228494

theorem group_size_proof (W : ℝ) (n : ℕ) : 
  (W + 15) / n = W / n + 2.5 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l2284_228494


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2284_228493

theorem perfect_square_condition (n : ℕ) (h : n > 0) :
  (∃ k : ℤ, 2 + 2 * Real.sqrt (1 + 12 * n^2) = k) →
  ∃ m : ℕ, n = m^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2284_228493


namespace NUMINAMATH_CALUDE_total_apples_l2284_228442

def marin_apples : ℕ := 8

def david_apples : ℚ := (3/4) * marin_apples

def amanda_apples : ℚ := 1.5 * david_apples + 2

theorem total_apples : marin_apples + david_apples + amanda_apples = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_l2284_228442


namespace NUMINAMATH_CALUDE_bruce_payment_l2284_228492

/-- The total amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1165 to the shopkeeper -/
theorem bruce_payment : total_amount 8 70 11 55 = 1165 := by
  sorry

end NUMINAMATH_CALUDE_bruce_payment_l2284_228492


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2284_228448

theorem complex_fraction_simplification :
  (2 + 2 * Complex.I) / (-3 + 4 * Complex.I) = -14/25 - 14/25 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2284_228448


namespace NUMINAMATH_CALUDE_octagon_side_length_l2284_228479

/-- Given an octagon-shaped box with a perimeter of 72 cm, prove that each side length is 9 cm. -/
theorem octagon_side_length (perimeter : ℝ) (num_sides : ℕ) : 
  perimeter = 72 ∧ num_sides = 8 → perimeter / num_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_octagon_side_length_l2284_228479


namespace NUMINAMATH_CALUDE_rain_both_days_no_snow_l2284_228498

theorem rain_both_days_no_snow (rain_sat rain_sun snow_sat : ℝ) 
  (h_rain_sat : rain_sat = 0.7)
  (h_rain_sun : rain_sun = 0.5)
  (h_snow_sat : snow_sat = 0.2)
  (h_independence : True) -- Assumption of independence
  : rain_sat * rain_sun * (1 - snow_sat) = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_rain_both_days_no_snow_l2284_228498


namespace NUMINAMATH_CALUDE_gcf_of_40_120_80_l2284_228446

theorem gcf_of_40_120_80 : Nat.gcd 40 (Nat.gcd 120 80) = 40 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_40_120_80_l2284_228446


namespace NUMINAMATH_CALUDE_train_crossing_time_l2284_228412

/-- Proves the time it takes for a train to cross a stationary man on a platform --/
theorem train_crossing_time (train_speed_kmph : ℝ) (train_speed_mps : ℝ) 
  (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed_kmph = 72 →
  train_speed_mps = 20 →
  platform_length = 300 →
  platform_crossing_time = 33 →
  ∃ (train_length : ℝ),
    train_length = train_speed_mps * platform_crossing_time - platform_length ∧
    train_length / train_speed_mps = 18 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2284_228412


namespace NUMINAMATH_CALUDE_doll_count_difference_l2284_228471

/-- The number of dolls Geraldine has -/
def geraldine_dolls : ℝ := 2186.25

/-- The number of dolls Jazmin has -/
def jazmin_dolls : ℝ := 1209.73

/-- The number of dolls Felicia has -/
def felicia_dolls : ℝ := 1530.48

/-- The difference between Geraldine's dolls and the sum of Jazmin's and Felicia's dolls -/
def doll_difference : ℝ := geraldine_dolls - (jazmin_dolls + felicia_dolls)

theorem doll_count_difference : doll_difference = -553.96 := by
  sorry

end NUMINAMATH_CALUDE_doll_count_difference_l2284_228471


namespace NUMINAMATH_CALUDE_system_1_solution_l2284_228414

theorem system_1_solution (x y : ℝ) :
  x = y + 1 ∧ 4 * x - 3 * y = 5 → x = 2 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_system_1_solution_l2284_228414


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2284_228441

theorem multiplication_puzzle :
  ∃ (AB C : ℕ),
    10 ≤ AB ∧ AB < 100 ∧
    1 ≤ C ∧ C < 10 ∧
    100 ≤ AB * 8 ∧ AB * 8 < 1000 ∧
    1000 ≤ AB * 9 ∧
    AB * C = 1068 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2284_228441


namespace NUMINAMATH_CALUDE_car_distance_theorem_l2284_228436

/-- Calculates the total distance traveled by a car with increasing speed over a given number of hours. -/
def total_distance (initial_speed : ℕ) (speed_increase : ℕ) (hours : ℕ) : ℕ :=
  (List.range hours).foldl (fun acc h => acc + (initial_speed + h * speed_increase)) 0

/-- Theorem stating that a car with initial speed 50 km/h, increasing by 2 km/h each hour, travels 732 km in 12 hours. -/
theorem car_distance_theorem : total_distance 50 2 12 = 732 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l2284_228436


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l2284_228440

/-- A quadratic function passing through the points (1, 8), (3, -1), and (5, 8) -/
def quadratic_function (x : ℝ) : ℝ := sorry

/-- The axis of symmetry of the quadratic function -/
def axis_of_symmetry : ℝ := 3

theorem quadratic_symmetry :
  (quadratic_function 1 = 8) ∧
  (quadratic_function 3 = -1) ∧
  (quadratic_function 5 = 8) →
  axis_of_symmetry = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l2284_228440
