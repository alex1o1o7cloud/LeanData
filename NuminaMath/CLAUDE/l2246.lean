import Mathlib

namespace treasure_probability_value_l2246_224637

/-- The probability of finding exactly 4 islands with treasure and no traps out of 8 islands -/
def treasure_probability : ℚ :=
  let n : ℕ := 8  -- Total number of islands
  let k : ℕ := 4  -- Number of islands with treasure
  let p_treasure : ℚ := 1/5  -- Probability of treasure and no traps
  let p_neither : ℚ := 7/10  -- Probability of neither treasure nor traps
  Nat.choose n k * p_treasure^k * p_neither^(n-k)

/-- The probability of finding exactly 4 islands with treasure and no traps out of 8 islands
    is equal to 673/25000 -/
theorem treasure_probability_value : treasure_probability = 673/25000 := by
  sorry

end treasure_probability_value_l2246_224637


namespace parallel_vectors_x_value_l2246_224643

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (6, 2) (x, 3) → x = 9 := by
  sorry

end parallel_vectors_x_value_l2246_224643


namespace divisibility_implies_equality_l2246_224614

theorem divisibility_implies_equality (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by
sorry

end divisibility_implies_equality_l2246_224614


namespace factorization_mn_minus_9m_l2246_224699

theorem factorization_mn_minus_9m (m n : ℝ) : m * n - 9 * m = m * (n - 9) := by
  sorry

end factorization_mn_minus_9m_l2246_224699


namespace registration_methods_count_l2246_224645

theorem registration_methods_count :
  let num_students : ℕ := 4
  let num_activities : ℕ := 3
  let students_choose_one (s : ℕ) (a : ℕ) : ℕ := a^s
  students_choose_one num_students num_activities = 81 := by
  sorry

end registration_methods_count_l2246_224645


namespace wendy_lost_lives_l2246_224672

/-- Represents the number of lives in Wendy's video game scenario -/
structure GameLives where
  initial : ℕ
  gained : ℕ
  final : ℕ
  lost : ℕ

/-- Theorem stating that Wendy lost 6 lives given the initial conditions -/
theorem wendy_lost_lives (game : GameLives) 
  (h1 : game.initial = 10)
  (h2 : game.gained = 37)
  (h3 : game.final = 41)
  (h4 : game.final = game.initial - game.lost + game.gained) :
  game.lost = 6 := by
  sorry

end wendy_lost_lives_l2246_224672


namespace cyclist_speed_north_l2246_224692

/-- The speed of the cyclist going north -/
def speed_north : ℝ := 10

/-- The speed of the cyclist going south -/
def speed_south : ℝ := 40

/-- The time taken -/
def time : ℝ := 1

/-- The distance between the cyclists after the given time -/
def distance : ℝ := 50

/-- Theorem stating that the speed of the cyclist going north is 10 km/h -/
theorem cyclist_speed_north : 
  speed_north + speed_south = distance / time :=
by sorry

end cyclist_speed_north_l2246_224692


namespace b_join_time_correct_l2246_224611

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- A's initial investment in Rupees -/
def aInvestment : ℕ := 36000

/-- B's initial investment in Rupees -/
def bInvestment : ℕ := 54000

/-- Profit sharing ratio of A to B -/
def profitRatio : ℚ := 2 / 1

/-- Calculates the time B joined the business in months -/
def bJoinTime : ℕ := monthsInYear - 8

theorem b_join_time_correct :
  (aInvestment * monthsInYear : ℚ) / (bInvestment * bJoinTime) = profitRatio :=
sorry

end b_join_time_correct_l2246_224611


namespace two_digit_product_sum_l2246_224632

theorem two_digit_product_sum (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8060 → 
  a + b = 127 := by
sorry

end two_digit_product_sum_l2246_224632


namespace num_valid_sequences_is_377_l2246_224679

/-- Represents a sequence of A's and B's -/
inductive ABSequence
  | A : ABSequence
  | B : ABSequence
  | cons : ABSequence → ABSequence → ABSequence

/-- Returns true if the given sequence satisfies the run length conditions -/
def validSequence : ABSequence → Bool :=
  sorry

/-- Returns the length of the given sequence -/
def sequenceLength : ABSequence → Nat :=
  sorry

/-- Returns true if the given sequence has length 15 and satisfies the run length conditions -/
def validSequenceOfLength15 (s : ABSequence) : Bool :=
  validSequence s ∧ sequenceLength s = 15

/-- The number of valid sequences of length 15 -/
def numValidSequences : Nat :=
  sorry

theorem num_valid_sequences_is_377 : numValidSequences = 377 := by
  sorry

end num_valid_sequences_is_377_l2246_224679


namespace circle_area_increase_l2246_224626

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end circle_area_increase_l2246_224626


namespace promotion_savings_l2246_224667

/-- The price of a single pair of shoes -/
def shoe_price : ℕ := 40

/-- The discount amount for Promotion B -/
def promotion_b_discount : ℕ := 15

/-- Calculate the total cost using Promotion A -/
def cost_promotion_a (price : ℕ) : ℕ :=
  price + price / 2

/-- Calculate the total cost using Promotion B -/
def cost_promotion_b (price : ℕ) (discount : ℕ) : ℕ :=
  price + (price - discount)

/-- Theorem: The difference in cost between Promotion B and Promotion A is $5 -/
theorem promotion_savings : 
  cost_promotion_b shoe_price promotion_b_discount - cost_promotion_a shoe_price = 5 := by
  sorry

end promotion_savings_l2246_224667


namespace birdseed_mix_problem_l2246_224651

/-- Proves that Brand A contains 60% sunflower given the conditions of the birdseed mix problem -/
theorem birdseed_mix_problem (brand_a_millet : ℝ) (brand_b_millet : ℝ) (brand_b_safflower : ℝ)
  (mix_millet : ℝ) (mix_brand_a : ℝ) :
  brand_a_millet = 0.4 →
  brand_b_millet = 0.65 →
  brand_b_safflower = 0.35 →
  mix_millet = 0.5 →
  mix_brand_a = 0.6 →
  ∃ (brand_a_sunflower : ℝ),
    brand_a_sunflower = 0.6 ∧
    brand_a_millet + brand_a_sunflower = 1 ∧
    mix_brand_a * brand_a_millet + (1 - mix_brand_a) * brand_b_millet = mix_millet :=
by sorry

end birdseed_mix_problem_l2246_224651


namespace sum_A_and_B_l2246_224633

theorem sum_A_and_B : 
  let B := 278 + 365 * 3
  let A := 20 * 100 + 87 * 10
  A + B = 4243 := by
sorry

end sum_A_and_B_l2246_224633


namespace cubic_local_min_implies_a_range_l2246_224625

/-- A function f has a local minimum in the interval (1, 2) -/
def has_local_min_in_interval (f : ℝ → ℝ) : Prop :=
  ∃ x, 1 < x ∧ x < 2 ∧ ∀ y, 1 < y ∧ y < 2 → f x ≤ f y

/-- The cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + a

theorem cubic_local_min_implies_a_range :
  ∀ a : ℝ, has_local_min_in_interval (f a) → 1 < a ∧ a < 4 :=
sorry

end cubic_local_min_implies_a_range_l2246_224625


namespace yellow_marble_fraction_l2246_224666

theorem yellow_marble_fraction (n : ℝ) (h : n > 0) : 
  let initial_green := (2/3) * n
  let initial_yellow := n - initial_green
  let new_yellow := 3 * initial_yellow
  let new_total := initial_green + new_yellow
  new_yellow / new_total = 3/5 := by sorry

end yellow_marble_fraction_l2246_224666


namespace fuel_mixture_problem_l2246_224693

/-- Proves the volume of fuel A in a partially filled tank --/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 212 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (volume_a : ℝ), volume_a = 98 ∧
    ∃ (volume_b : ℝ), volume_a + volume_b = tank_capacity ∧
      ethanol_a * volume_a + ethanol_b * volume_b = total_ethanol :=
by
  sorry

end fuel_mixture_problem_l2246_224693


namespace smallest_angle_for_complete_circle_sufficient_angle_for_complete_circle_exact_smallest_angle_for_complete_circle_l2246_224677

/-- The smallest angle needed to plot the entire circle for r = sin θ -/
theorem smallest_angle_for_complete_circle : 
  ∀ t : ℝ, (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = Real.sin θ ∧ 
    (∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ)) →
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = (Real.sin θ) * (Real.sin θ)) →
  t ≥ 3 * π / 2 :=
by sorry

/-- 3π/2 is sufficient to plot the entire circle for r = sin θ -/
theorem sufficient_angle_for_complete_circle :
  ∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 3 * π / 2 ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = (Real.sin θ) * (Real.sin θ) :=
by sorry

/-- The smallest angle needed to plot the entire circle for r = sin θ is exactly 3π/2 -/
theorem exact_smallest_angle_for_complete_circle :
  (∀ t : ℝ, t < 3 * π / 2 → 
    ∃ x y : ℝ, x^2 + y^2 ≤ 1 ∧ 
      ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → x ≠ (Real.sin θ) * (Real.cos θ) ∨ y ≠ (Real.sin θ) * (Real.sin θ)) ∧
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 3 * π / 2 ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = (Real.sin θ) * (Real.sin θ)) :=
by sorry

end smallest_angle_for_complete_circle_sufficient_angle_for_complete_circle_exact_smallest_angle_for_complete_circle_l2246_224677


namespace point_in_second_quadrant_l2246_224612

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

-- Theorem statement
theorem point_in_second_quadrant (m n : ℝ) 
  (hm : quadratic_eq m) (hn : quadratic_eq n) (hlt : m < n) : 
  m < 0 ∧ n > 0 := by
  sorry

end point_in_second_quadrant_l2246_224612


namespace fields_medal_stats_l2246_224630

def data_set : List ℕ := [29, 32, 33, 35, 35, 40]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem fields_medal_stats : 
  mode data_set = 35 ∧ median data_set = 34 := by sorry

end fields_medal_stats_l2246_224630


namespace symmetry_coordinates_l2246_224609

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

theorem symmetry_coordinates :
  let A : Point := ⟨3, -2⟩
  let A' : Point := ⟨-3, 2⟩
  symmetricToOrigin A A' → A'.x = -3 ∧ A'.y = 2 := by
  sorry

end symmetry_coordinates_l2246_224609


namespace binomial_coefficients_10_l2246_224669

theorem binomial_coefficients_10 : (Nat.choose 10 10 = 1) ∧ (Nat.choose 10 9 = 10) := by
  sorry

end binomial_coefficients_10_l2246_224669


namespace tshirt_cost_l2246_224649

theorem tshirt_cost (initial_amount : ℝ) (sweater_cost : ℝ) (shoes_cost : ℝ) 
                    (refund_percentage : ℝ) (final_amount : ℝ) :
  initial_amount = 74 →
  sweater_cost = 9 →
  shoes_cost = 30 →
  refund_percentage = 0.9 →
  final_amount = 51 →
  ∃ (tshirt_cost : ℝ),
    tshirt_cost = 14 ∧
    final_amount = initial_amount - sweater_cost - tshirt_cost - shoes_cost + refund_percentage * shoes_cost :=
by sorry

end tshirt_cost_l2246_224649


namespace cubic_factorization_l2246_224675

theorem cubic_factorization (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by
  sorry

end cubic_factorization_l2246_224675


namespace orange_marbles_count_l2246_224615

def total_marbles : ℕ := 24
def blue_marbles : ℕ := total_marbles / 2
def red_marbles : ℕ := 6

theorem orange_marbles_count : total_marbles - blue_marbles - red_marbles = 6 := by
  sorry

end orange_marbles_count_l2246_224615


namespace original_number_l2246_224660

theorem original_number : ∃ x : ℝ, 3 * (2 * x + 9) = 51 ∧ x = 4 := by
  sorry

end original_number_l2246_224660


namespace total_weekly_time_l2246_224659

def parking_time : ℕ := 5
def walking_time : ℕ := 3
def long_wait_days : ℕ := 2
def short_wait_days : ℕ := 3
def long_wait_time : ℕ := 30
def short_wait_time : ℕ := 10
def work_days : ℕ := 5

theorem total_weekly_time :
  (parking_time + walking_time) * work_days +
  long_wait_days * long_wait_time +
  short_wait_days * short_wait_time = 130 := by
sorry

end total_weekly_time_l2246_224659


namespace soda_cans_with_tax_l2246_224610

/-- Given:
  S : number of cans bought for Q quarters
  Q : number of quarters for S cans
  t : tax rate as a fraction of 1
  D : number of dollars available
-/
theorem soda_cans_with_tax (S Q : ℕ) (t : ℚ) (D : ℕ) :
  let cans_purchasable := (4 * D * S * (1 + t)) / Q
  cans_purchasable = (4 * D * S * (1 + t)) / Q :=
by sorry

end soda_cans_with_tax_l2246_224610


namespace wine_equation_correctness_l2246_224602

/-- Represents the wine consumption and intoxication scenario --/
def wine_scenario (x y : ℚ) : Prop :=
  -- Total bottles of wine
  x + y = 19 ∧
  -- Intoxication effect
  3 * x + (1/3) * y = 33 ∧
  -- x represents good wine bottles
  x ≥ 0 ∧
  -- y represents inferior wine bottles
  y ≥ 0

/-- The system of equations correctly represents the wine scenario --/
theorem wine_equation_correctness :
  ∃ x y : ℚ, wine_scenario x y :=
sorry

end wine_equation_correctness_l2246_224602


namespace find_m_value_l2246_224641

theorem find_m_value : ∃ m : ℝ, 
  (∀ x : ℝ, (m - 1) * x^2 + 5 * x + m^2 - 3 * m + 2 = 0) ∧ 
  (m^2 - 3 * m + 2 = 0) ∧ 
  (m - 1 ≠ 0) ∧ 
  (m = 2) := by
sorry

end find_m_value_l2246_224641


namespace b_investment_is_4200_l2246_224681

/-- Represents the investment and profit details of a partnership business -/
structure BusinessPartnership where
  a_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- Calculates B's investment given the partnership details -/
def calculate_b_investment (bp : BusinessPartnership) : ℕ :=
  bp.total_profit * bp.a_investment / bp.a_profit_share - bp.a_investment - bp.c_investment

/-- Theorem stating that B's investment is 4200 given the specified conditions -/
theorem b_investment_is_4200 (bp : BusinessPartnership) 
  (h1 : bp.a_investment = 6300)
  (h2 : bp.c_investment = 10500)
  (h3 : bp.total_profit = 13000)
  (h4 : bp.a_profit_share = 3900) :
  calculate_b_investment bp = 4200 := by
  sorry

#eval calculate_b_investment ⟨6300, 10500, 13000, 3900⟩

end b_investment_is_4200_l2246_224681


namespace sequence_difference_l2246_224695

theorem sequence_difference (x : ℕ → ℕ)
  (h1 : x 1 = 1)
  (h2 : ∀ n, x n < x (n + 1))
  (h3 : ∀ n, x (n + 1) ≤ 2 * n) :
  ∀ k : ℕ, k > 0 → ∃ i j, k = x i - x j :=
by sorry

end sequence_difference_l2246_224695


namespace complement_of_A_l2246_224674

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | 1 / x ≥ 1}

theorem complement_of_A : 
  Set.compl A = {x : ℝ | x ≤ 0 ∨ x > 1} := by sorry

end complement_of_A_l2246_224674


namespace purely_imaginary_complex_number_l2246_224661

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 := by
  sorry

end purely_imaginary_complex_number_l2246_224661


namespace perpendicular_to_oblique_implies_perpendicular_to_projection_l2246_224647

/-- A plane in which we consider lines and their projections. -/
structure Plane where
  -- Add necessary fields here

/-- Represents a line in the plane. -/
structure Line (P : Plane) where
  -- Add necessary fields here

/-- Indicates that a line is oblique (not parallel or perpendicular to some reference). -/
def isOblique (P : Plane) (l : Line P) : Prop :=
  sorry

/-- The projection of a line onto the plane. -/
def projection (P : Plane) (l : Line P) : Line P :=
  sorry

/-- Indicates that two lines are perpendicular. -/
def isPerpendicular (P : Plane) (l1 l2 : Line P) : Prop :=
  sorry

/-- 
The main theorem: If a line is perpendicular to an oblique line in a plane,
then it is also perpendicular to the projection of the oblique line in this plane.
-/
theorem perpendicular_to_oblique_implies_perpendicular_to_projection
  (P : Plane) (l1 l2 : Line P) (h1 : isOblique P l1) (h2 : isPerpendicular P l1 l2) :
  isPerpendicular P (projection P l1) l2 :=
sorry

end perpendicular_to_oblique_implies_perpendicular_to_projection_l2246_224647


namespace smallest_k_carboxylic_for_8002_l2246_224600

/-- A function that checks if a number has all digits the same --/
def allDigitsSame (n : ℕ) : Prop := sorry

/-- A function that checks if a list of natural numbers are all distinct --/
def allDistinct (list : List ℕ) : Prop := sorry

/-- A function that checks if all numbers in a list are greater than 9 --/
def allGreaterThan9 (list : List ℕ) : Prop := sorry

/-- A function that checks if a number is k-carboxylic --/
def isKCarboxylic (n k : ℕ) : Prop :=
  ∃ (list : List ℕ), 
    list.length = k ∧ 
    list.sum = n ∧ 
    allDistinct list ∧ 
    allGreaterThan9 list ∧ 
    ∀ m ∈ list, allDigitsSame m

/-- The main theorem --/
theorem smallest_k_carboxylic_for_8002 :
  (isKCarboxylic 8002 14) ∧ ∀ k < 14, ¬(isKCarboxylic 8002 k) := by sorry

end smallest_k_carboxylic_for_8002_l2246_224600


namespace train_speed_l2246_224629

/-- Given a train and platform with the following properties:
  * The train and platform have equal length
  * The train is 750 meters long
  * The train crosses the platform in one minute
  Prove that the speed of the train is 90 km/hr -/
theorem train_speed (train_length : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_length = 750 →
  platform_length = train_length →
  crossing_time = 1 →
  (train_length + platform_length) / crossing_time * 60 / 1000 = 90 := by
  sorry

end train_speed_l2246_224629


namespace quadratic_roots_divisibility_l2246_224622

theorem quadratic_roots_divisibility
  (a b : ℤ) (u v : ℂ) (h1 : u^2 + a*u + b = 0)
  (h2 : v^2 + a*v + b = 0) (h3 : ∃ k : ℤ, a^2 = k * b) :
  ∀ n : ℕ, ∃ m : ℤ, u^(2*n) + v^(2*n) = m * b^n :=
sorry

end quadratic_roots_divisibility_l2246_224622


namespace special_triangle_angle_difference_l2246_224654

/-- A triangle with special angle properties -/
structure SpecialTriangle where
  /-- The smallest angle of the triangle -/
  a : ℕ
  /-- The middle angle of the triangle -/
  b : ℕ
  /-- The largest angle of the triangle -/
  c : ℕ
  /-- One of the angles is a prime number -/
  h1 : Prime a ∨ Prime b ∨ Prime c
  /-- Two of the angles are squares of prime numbers -/
  h2 : ∃ p q : ℕ, Prime p ∧ Prime q ∧ 
       ((b = p^2 ∧ c = q^2) ∨ (a = p^2 ∧ c = q^2) ∨ (a = p^2 ∧ b = q^2))
  /-- The sum of the angles is 180 degrees -/
  h3 : a + b + c = 180
  /-- The angles are in ascending order -/
  h4 : a ≤ b ∧ b ≤ c

/-- The theorem stating the difference between the largest and smallest angles -/
theorem special_triangle_angle_difference (t : SpecialTriangle) : t.c - t.a = 167 := by
  sorry

end special_triangle_angle_difference_l2246_224654


namespace inequality_solution_range_l2246_224670

theorem inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, |x + 2| - |x + 3| > m) → m < -1 := by
  sorry

end inequality_solution_range_l2246_224670


namespace sqrt_36_equals_6_l2246_224653

theorem sqrt_36_equals_6 : Real.sqrt 36 = 6 := by
  sorry

end sqrt_36_equals_6_l2246_224653


namespace decimal_49_to_binary_l2246_224676

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Convert a list of booleans to a natural number in binary representation -/
def fromBinary (l : List Bool) : ℕ :=
  l.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem decimal_49_to_binary :
  toBinary 49 = [true, true, false, false, false, true] :=
by sorry

end decimal_49_to_binary_l2246_224676


namespace divisible_by_15_20_25_between_1000_and_2000_l2246_224613

theorem divisible_by_15_20_25_between_1000_and_2000 : 
  ∃! n : ℕ, (∀ k : ℕ, 1000 < k ∧ k < 2000 ∧ 15 ∣ k ∧ 20 ∣ k ∧ 25 ∣ k → k ∈ Finset.range n) ∧ 
  (∀ k : ℕ, k ∈ Finset.range n → 1000 < k ∧ k < 2000 ∧ 15 ∣ k ∧ 20 ∣ k ∧ 25 ∣ k) :=
by sorry

end divisible_by_15_20_25_between_1000_and_2000_l2246_224613


namespace insurance_compensation_l2246_224640

/-- Insurance compensation calculation --/
theorem insurance_compensation
  (insured_amount : ℝ)
  (deductible_percentage : ℝ)
  (actual_damage : ℝ)
  (h1 : insured_amount = 500000)
  (h2 : deductible_percentage = 0.01)
  (h3 : actual_damage = 4000)
  : min (max (actual_damage - insured_amount * deductible_percentage) 0) insured_amount = 0 :=
by
  sorry

end insurance_compensation_l2246_224640


namespace no_solution_iff_m_equals_six_l2246_224665

theorem no_solution_iff_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (2 * x + m) / (x + 3) ≠ 1) ↔ m = 6 := by
  sorry

end no_solution_iff_m_equals_six_l2246_224665


namespace angle_sum_in_circle_l2246_224682

theorem angle_sum_in_circle (x : ℝ) : 
  3 * x + 6 * x + 2 * x + x = 360 → x = 30 := by
  sorry

#check angle_sum_in_circle

end angle_sum_in_circle_l2246_224682


namespace volcano_count_l2246_224601

theorem volcano_count (total : ℕ) (intact : ℕ) : 
  (intact : ℝ) = total * (1 - 0.2) * (1 - 0.4) * (1 - 0.5) ∧ intact = 48 → 
  total = 200 := by
sorry

end volcano_count_l2246_224601


namespace fraction_sum_difference_l2246_224696

theorem fraction_sum_difference : 7/6 + 5/4 - 3/2 = 11/12 := by
  sorry

end fraction_sum_difference_l2246_224696


namespace erased_odd_number_l2246_224638

theorem erased_odd_number (n : ℕ) (erased : ℕ) :
  (∃ k, n = k^2) ∧
  (∃ m, erased = 2*m - 1) ∧
  (n^2 - erased = 2008) →
  erased = 17 := by
sorry

end erased_odd_number_l2246_224638


namespace cubic_identity_l2246_224604

theorem cubic_identity (y : ℝ) (h : y^3 + 1/y^3 = 110) : y + 1/y = 5 := by
  sorry

end cubic_identity_l2246_224604


namespace part_one_part_two_l2246_224624

noncomputable section

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hsum : A + B + C = π)
  (law_of_sines : a / Real.sin A = b / Real.sin B)
  (law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)

-- Define the specific triangle with the given condition
def SpecialTriangle (t : Triangle) : Prop :=
  3 * t.a = 2 * t.b

-- Part I
theorem part_one (t : Triangle) (h : SpecialTriangle t) (hB : t.B = π/3) :
  Real.sin t.C = (Real.sqrt 3 + 3 * Real.sqrt 2) / 6 :=
sorry

-- Part II
theorem part_two (t : Triangle) (h : SpecialTriangle t) (hC : Real.cos t.C = 2/3) :
  Real.sin (t.A - t.B) = -Real.sqrt 5 / 3 :=
sorry

end part_one_part_two_l2246_224624


namespace line_symmetry_l2246_224691

-- Define the lines
def original_line (x y : ℝ) : Prop := 2*x - y + 3 = 0
def reference_line (x y : ℝ) : Prop := x - y + 2 = 0
def symmetric_line (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (l1 l2 l_ref : (ℝ → ℝ → Prop)) : Prop :=
  ∀ (x y : ℝ), l1 x y → ∃ (x' y' : ℝ), l2 x' y' ∧
    (x + x') / 2 = (y + y') / 2 + 2 ∧ -- Point on reference line
    (y' - y) = (x' - x) -- Perpendicular to reference line

-- Theorem statement
theorem line_symmetry :
  symmetric_wrt original_line symmetric_line reference_line :=
sorry

end line_symmetry_l2246_224691


namespace cow_ratio_proof_l2246_224662

theorem cow_ratio_proof (initial_cows : ℕ) (added_cows : ℕ) (remaining_cows : ℕ) : 
  initial_cows = 51 → 
  added_cows = 5 → 
  remaining_cows = 42 → 
  (initial_cows + added_cows - remaining_cows) / (initial_cows + added_cows) = 1/4 :=
by
  sorry

end cow_ratio_proof_l2246_224662


namespace ping_pong_meeting_l2246_224603

theorem ping_pong_meeting (total_legs : ℕ) (square_stool_legs round_stool_legs : ℕ) :
  total_legs = 33 ∧ square_stool_legs = 4 ∧ round_stool_legs = 3 →
  ∃ (total_members square_stools round_stools : ℕ),
    total_members = square_stools + round_stools ∧
    total_members * 2 + square_stools * square_stool_legs + round_stools * round_stool_legs = total_legs ∧
    total_members = 6 :=
by sorry

end ping_pong_meeting_l2246_224603


namespace nine_gon_diagonals_l2246_224678

/-- The number of diagonals in a regular nine-sided polygon -/
def num_diagonals_nine_gon : ℕ :=
  (9 * (9 - 1)) / 2 - 9

theorem nine_gon_diagonals :
  num_diagonals_nine_gon = 27 := by
  sorry

end nine_gon_diagonals_l2246_224678


namespace quadratic_roots_d_value_l2246_224636

theorem quadratic_roots_d_value (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) →
  d = 9.8 :=
by sorry

end quadratic_roots_d_value_l2246_224636


namespace fraction_of_male_birds_l2246_224673

theorem fraction_of_male_birds (T : ℚ) (h1 : T > 0) : 
  let robins := (2 / 5 : ℚ) * T
  let bluejays := T - robins
  let female_robins := (1 / 3 : ℚ) * robins
  let female_bluejays := (2 / 3 : ℚ) * bluejays
  let male_birds := T - female_robins - female_bluejays
  male_birds / T = 7 / 15 := by
sorry

end fraction_of_male_birds_l2246_224673


namespace recruitment_probability_one_pass_reinspection_probability_l2246_224652

/-- Probabilities of passing re-inspection for students A, B, and C -/
def p_reinspect_A : ℝ := 0.5
def p_reinspect_B : ℝ := 0.6
def p_reinspect_C : ℝ := 0.75

/-- Probabilities of passing cultural examination for students A, B, and C -/
def p_cultural_A : ℝ := 0.6
def p_cultural_B : ℝ := 0.5
def p_cultural_C : ℝ := 0.4

/-- All students pass political review -/
def p_political : ℝ := 1

/-- Assumption: Outcomes of the last three stages are independent -/
axiom independence : True

theorem recruitment_probability :
  p_reinspect_A * p_cultural_A * p_political = 0.3 :=
sorry

theorem one_pass_reinspection_probability :
  p_reinspect_A * (1 - p_reinspect_B) * (1 - p_reinspect_C) +
  (1 - p_reinspect_A) * p_reinspect_B * (1 - p_reinspect_C) +
  (1 - p_reinspect_A) * (1 - p_reinspect_B) * p_reinspect_C = 0.275 :=
sorry

end recruitment_probability_one_pass_reinspection_probability_l2246_224652


namespace pear_problem_solution_l2246_224621

/-- Represents the pear selling problem --/
def PearProblem (initial_pears : ℝ) : Prop :=
  let sold_day1 := 0.20 * initial_pears
  let remaining_after_sale := initial_pears - sold_day1
  let thrown_day1 := 0.50 * remaining_after_sale
  let remaining_day2 := remaining_after_sale - thrown_day1
  let total_thrown := 0.72 * initial_pears
  let thrown_day2 := total_thrown - thrown_day1
  let sold_day2 := remaining_day2 - thrown_day2
  (sold_day2 / remaining_day2) = 0.20

/-- Theorem stating that the percentage of remaining pears sold on day 2 is 20% --/
theorem pear_problem_solution : 
  ∀ initial_pears : ℝ, initial_pears > 0 → PearProblem initial_pears :=
by
  sorry


end pear_problem_solution_l2246_224621


namespace like_terms_exponent_l2246_224619

theorem like_terms_exponent (a b : ℕ) : 
  (∀ x y : ℝ, ∃ k : ℝ, 2 * x^a * y^3 = k * (-x^2 * y^b)) → a^b = 8 := by
  sorry

end like_terms_exponent_l2246_224619


namespace quadratic_factorization_l2246_224639

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 12 * x + 18 = 2 * (x - 3)^2 := by
  sorry

end quadratic_factorization_l2246_224639


namespace inequality_proof_l2246_224683

theorem inequality_proof (x : ℝ) (h1 : x > -1) (h2 : x ≠ 0) :
  (2 * |x|) / (2 + x) < |Real.log (1 + x)| ∧ |Real.log (1 + x)| < |x| / Real.sqrt (1 + x) := by
  sorry

end inequality_proof_l2246_224683


namespace green_bows_count_l2246_224628

theorem green_bows_count (total : ℕ) (white : ℕ) :
  white = 40 →
  (1 : ℚ) / 4 + (1 : ℚ) / 3 + (1 : ℚ) / 6 + (white : ℚ) / total = 1 →
  (1 : ℚ) / 6 * total = 27 :=
by sorry

end green_bows_count_l2246_224628


namespace puppies_given_away_l2246_224620

def initial_puppies : ℕ := 12
def current_puppies : ℕ := 5

theorem puppies_given_away : initial_puppies - current_puppies = 7 := by
  sorry

end puppies_given_away_l2246_224620


namespace reeya_second_subject_score_l2246_224684

/-- Given Reeya's scores in 4 subjects and her average score, prove the score of the second subject. -/
theorem reeya_second_subject_score (score1 score2 score3 score4 : ℕ) (average : ℚ) :
  score1 = 55 →
  score3 = 82 →
  score4 = 55 →
  average = 67 →
  (score1 + score2 + score3 + score4 : ℚ) / 4 = average →
  score2 = 76 := by
sorry

end reeya_second_subject_score_l2246_224684


namespace initial_men_employed_is_300_l2246_224688

/-- Represents the highway construction scenario --/
structure HighwayConstruction where
  totalLength : ℝ
  totalDays : ℕ
  initialHoursPerDay : ℕ
  daysWorked : ℕ
  workCompleted : ℝ
  additionalMen : ℕ
  newHoursPerDay : ℕ

/-- Calculates the initial number of men employed --/
def initialMenEmployed (h : HighwayConstruction) : ℕ :=
  sorry

/-- Theorem stating that the initial number of men employed is 300 --/
theorem initial_men_employed_is_300 (h : HighwayConstruction) 
  (h_total_length : h.totalLength = 2)
  (h_total_days : h.totalDays = 50)
  (h_initial_hours : h.initialHoursPerDay = 8)
  (h_days_worked : h.daysWorked = 25)
  (h_work_completed : h.workCompleted = 1/3)
  (h_additional_men : h.additionalMen = 60)
  (h_new_hours : h.newHoursPerDay = 10) :
  initialMenEmployed h = 300 :=
sorry

end initial_men_employed_is_300_l2246_224688


namespace intersection_M_N_l2246_224646

def M : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x : ℝ | -5 < x ∧ x < 5}

theorem intersection_M_N : M ∩ N = {x : ℝ | -3 < x ∧ x < 5} := by sorry

end intersection_M_N_l2246_224646


namespace common_root_is_one_l2246_224617

/-- Given two quadratic equations with coefficients a and b that have exactly one common root, prove that this root is 1 -/
theorem common_root_is_one (a b : ℝ) 
  (h : ∃! x : ℝ, (x^2 + a*x + b = 0) ∧ (x^2 + b*x + a = 0)) : 
  ∃ x : ℝ, (x^2 + a*x + b = 0) ∧ (x^2 + b*x + a = 0) ∧ x = 1 :=
by
  sorry


end common_root_is_one_l2246_224617


namespace complex_modulus_problem_l2246_224663

theorem complex_modulus_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 9)
  (h3 : Complex.abs (z + w) = 5) :
  Complex.abs z = 4 := by sorry

end complex_modulus_problem_l2246_224663


namespace merchant_pricing_strategy_l2246_224635

/-- Represents the pricing strategy of a merchant --/
structure MerchantPricing where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  selling_discount : ℝ
  profit_margin : ℝ

/-- Calculates the purchase price given the list price and purchase discount --/
def purchase_price (mp : MerchantPricing) : ℝ :=
  mp.list_price * (1 - mp.purchase_discount)

/-- Calculates the selling price given the marked price and selling discount --/
def selling_price (mp : MerchantPricing) : ℝ :=
  mp.marked_price * (1 - mp.selling_discount)

/-- Checks if the pricing strategy satisfies the profit margin requirement --/
def satisfies_profit_margin (mp : MerchantPricing) : Prop :=
  selling_price mp - purchase_price mp = mp.profit_margin * selling_price mp

/-- The main theorem to prove --/
theorem merchant_pricing_strategy (mp : MerchantPricing) 
  (h1 : mp.purchase_discount = 0.3)
  (h2 : mp.selling_discount = 0.2)
  (h3 : mp.profit_margin = 0.2)
  (h4 : satisfies_profit_margin mp) :
  mp.marked_price / mp.list_price = 1.09375 := by
  sorry

end merchant_pricing_strategy_l2246_224635


namespace mystery_number_sum_l2246_224685

theorem mystery_number_sum : Int → Prop :=
  fun result =>
    let mystery_number : Int := 47
    let added_number : Int := 45
    result = mystery_number + added_number

#check mystery_number_sum 92

end mystery_number_sum_l2246_224685


namespace m_range_l2246_224605

/-- A circle in a 2D Cartesian coordinate system --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point A on circle C --/
def A : ℝ × ℝ := (3, 2)

/-- Circle C --/
def C : Circle := { center := (3, 4), radius := 2 }

/-- First fold line equation --/
def foldLine1 (x y : ℝ) : Prop := x - y + 1 = 0

/-- Second fold line equation --/
def foldLine2 (x y : ℝ) : Prop := x + y - 7 = 0

/-- Point M on x-axis --/
def M (m : ℝ) : ℝ × ℝ := (-m, 0)

/-- Point N on x-axis --/
def N (m : ℝ) : ℝ × ℝ := (m, 0)

/-- Theorem stating the range of m --/
theorem m_range : 
  ∀ m : ℝ, 
  (∃ P : ℝ × ℝ, 
    (P.1 - C.center.1)^2 + (P.2 - C.center.2)^2 = C.radius^2 ∧ 
    (P.1 - (M m).1)^2 + (P.2 - (M m).2)^2 = (P.1 - (N m).1)^2 + (P.2 - (N m).2)^2
  ) ↔ 3 ≤ m ∧ m ≤ 7 := by sorry

end m_range_l2246_224605


namespace digit_equation_solution_l2246_224680

theorem digit_equation_solution :
  ∃ (a b c d e : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    (5^a) + (100*b + 10*c + 3) = (1000*d + 100*e + 1) := by
  sorry

end digit_equation_solution_l2246_224680


namespace comic_book_stacking_theorem_l2246_224607

def num_spiderman : ℕ := 7
def num_archie : ℕ := 6
def num_garfield : ℕ := 4

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def permutations_within_groups : ℕ := 
  factorial num_spiderman * factorial num_archie * factorial num_garfield

def group_arrangements : ℕ := 2 * 2

theorem comic_book_stacking_theorem :
  permutations_within_groups * group_arrangements = 19353600 := by
  sorry

end comic_book_stacking_theorem_l2246_224607


namespace flower_seedling_problem_l2246_224634

/-- Represents the unit price of flower seedlings --/
structure FlowerPrice where
  typeA : ℝ
  typeB : ℝ

/-- Represents the cost function for purchasing flower seedlings --/
def cost_function (p : FlowerPrice) (a : ℝ) : ℝ :=
  p.typeA * (12 - a) + (p.typeB - a) * a

/-- The theorem statement for the flower seedling problem --/
theorem flower_seedling_problem (p : FlowerPrice) :
  (3 * p.typeA + 5 * p.typeB = 210) →
  (4 * p.typeA + 10 * p.typeB = 380) →
  p.typeA = 20 ∧ p.typeB = 30 ∧
  ∃ (a_min a_max : ℝ), 0 < a_min ∧ a_min < 12 ∧ 0 < a_max ∧ a_max < 12 ∧
    ∀ (a : ℝ), 0 < a ∧ a < 12 →
      229 ≤ cost_function p a ∧ cost_function p a ≤ 265 ∧
      cost_function p a_min = 229 ∧ cost_function p a_max = 265 := by
  sorry

end flower_seedling_problem_l2246_224634


namespace coffee_maker_capacity_l2246_224648

/-- Represents a cylindrical coffee maker -/
structure CoffeeMaker :=
  (capacity : ℝ)

/-- The coffee maker contains 45 cups when it is 36% full -/
def partially_filled (cm : CoffeeMaker) : Prop :=
  0.36 * cm.capacity = 45

/-- Theorem: A cylindrical coffee maker that contains 45 cups when 36% full has a capacity of 125 cups -/
theorem coffee_maker_capacity (cm : CoffeeMaker) 
  (h : partially_filled cm) : cm.capacity = 125 := by
  sorry

end coffee_maker_capacity_l2246_224648


namespace interesting_iff_prime_power_l2246_224687

def is_interesting (n : ℕ) : Prop :=
  n > 1 ∧ ∀ x y : ℕ, (Nat.gcd x n ≠ 1 ∧ Nat.gcd y n ≠ 1) → Nat.gcd (x + y) n ≠ 1

theorem interesting_iff_prime_power (n : ℕ) :
  is_interesting n ↔ ∃ p k : ℕ, Nat.Prime p ∧ k > 0 ∧ n = p^k :=
sorry

end interesting_iff_prime_power_l2246_224687


namespace intra_division_games_is_56_l2246_224694

/-- Represents a basketball league with specific conditions -/
structure BasketballLeague where
  N : ℕ  -- Number of times teams within the same division play each other
  M : ℕ  -- Number of times teams from different divisions play each other
  division_size : ℕ  -- Number of teams in each division
  total_games : ℕ  -- Total number of games each team plays in the season
  h1 : 3 * N = 5 * M + 8
  h2 : M > 6
  h3 : division_size = 5
  h4 : total_games = 82
  h5 : (division_size - 1) * N + division_size * M = total_games

/-- The number of games a team plays within its own division -/
def intra_division_games (league : BasketballLeague) : ℕ :=
  (league.division_size - 1) * league.N

/-- Theorem stating that each team plays 56 games within its own division -/
theorem intra_division_games_is_56 (league : BasketballLeague) :
  intra_division_games league = 56 := by
  sorry

end intra_division_games_is_56_l2246_224694


namespace min_sum_of_squares_l2246_224608

theorem min_sum_of_squares (x y : ℝ) (h : (x + 3) * (y - 3) = 0) :
  ∃ (m : ℝ), (∀ a b : ℝ, (a + 3) * (b - 3) = 0 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 18 :=
sorry

end min_sum_of_squares_l2246_224608


namespace not_penetrating_function_l2246_224689

/-- Definition of a penetrating function -/
def isPenetratingFunction (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, a > 0 ∧ a ≠ 1 → ∀ x : ℝ, f (a * x) = a * f x

/-- The function f(x) = x + 1 -/
def f (x : ℝ) : ℝ := x + 1

/-- Theorem: f(x) = x + 1 is not a penetrating function -/
theorem not_penetrating_function : ¬ isPenetratingFunction f := by
  sorry

end not_penetrating_function_l2246_224689


namespace digit_puzzle_proof_l2246_224658

theorem digit_puzzle_proof (P Q R S : ℕ) : 
  (P < 10 ∧ Q < 10 ∧ R < 10 ∧ S < 10) →
  (10 * P + Q) + (10 * R + P) = 10 * S + P →
  (10 * P + Q) - (10 * R + P) = P →
  S = 0 := by
  sorry

end digit_puzzle_proof_l2246_224658


namespace birthday_party_attendees_l2246_224697

theorem birthday_party_attendees :
  ∀ (n : ℕ),
  (12 * (n + 2) = 16 * n) →
  n = 6 :=
by
  sorry

end birthday_party_attendees_l2246_224697


namespace midpoint_calculation_l2246_224698

/-- Given two points A and B in a 2D plane, prove that 3x - 5y = -13.5,
    where (x, y) is the midpoint of segment AB. -/
theorem midpoint_calculation (A B : ℝ × ℝ) (h : A = (20, 12) ∧ B = (-4, 3)) :
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  3 * C.1 - 5 * C.2 = -13.5 := by
sorry

end midpoint_calculation_l2246_224698


namespace library_charge_calculation_l2246_224668

/-- Calculates the total amount paid for borrowed books --/
def total_amount_paid (daily_rate : ℚ) (book1_days : ℕ) (book2_days : ℕ) (num_books2 : ℕ) : ℚ :=
  daily_rate * book1_days + daily_rate * book2_days * num_books2

theorem library_charge_calculation :
  let daily_rate : ℚ := 50 / 100  -- 50 cents in dollars
  let book1_days : ℕ := 20
  let book2_days : ℕ := 31
  let num_books2 : ℕ := 2
  total_amount_paid daily_rate book1_days book2_days num_books2 = 41 := by
sorry

#eval total_amount_paid (50 / 100) 20 31 2

end library_charge_calculation_l2246_224668


namespace num_new_candles_l2246_224690

/-- The amount of wax left in a candle as a percentage of its original weight -/
def waxLeftPercentage : ℚ := 1 / 10

/-- The weight of a large candle in ounces -/
def largeCandle : ℚ := 20

/-- The weight of a medium candle in ounces -/
def mediumCandle : ℚ := 5

/-- The weight of a small candle in ounces -/
def smallCandle : ℚ := 1

/-- The number of large candles -/
def numLargeCandles : ℕ := 5

/-- The number of medium candles -/
def numMediumCandles : ℕ := 5

/-- The number of small candles -/
def numSmallCandles : ℕ := 25

/-- The weight of a new candle to be made in ounces -/
def newCandleWeight : ℚ := 5

/-- Theorem: The number of new candles that can be made is 3 -/
theorem num_new_candles :
  (waxLeftPercentage * (numLargeCandles * largeCandle + 
                        numMediumCandles * mediumCandle + 
                        numSmallCandles * smallCandle)) / newCandleWeight = 3 := by
  sorry

end num_new_candles_l2246_224690


namespace existence_of_index_l2246_224606

theorem existence_of_index (n : ℕ) (x : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_x : ∀ i, i ≤ n → 0 ≤ x i ∧ x i ≤ 1) :
  ∃ i, 1 ≤ i ∧ i ≤ n - 1 ∧ x i * (1 - x (i + 1)) ≥ (1/4) * x 1 * (1 - x n) := by
sorry

end existence_of_index_l2246_224606


namespace iron_conductivity_is_deductive_l2246_224631

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Metal : U → Prop)
variable (ConductsElectricity : U → Prop)

-- Define iron as a constant in our universe
variable (iron : U)

-- State the premises and conclusion
variable (all_metals_conduct : ∀ x, Metal x → ConductsElectricity x)
variable (iron_is_metal : Metal iron)
variable (iron_conducts : ConductsElectricity iron)

-- Define deductive reasoning
def is_deductive_reasoning (premise1 premise2 conclusion : Prop) : Prop :=
  (premise1 ∧ premise2) → conclusion

-- Theorem stating that the given reasoning is deductive
theorem iron_conductivity_is_deductive :
  is_deductive_reasoning 
    (∀ x, Metal x → ConductsElectricity x)
    (Metal iron)
    (ConductsElectricity iron) :=
by sorry

end iron_conductivity_is_deductive_l2246_224631


namespace complex_power_four_l2246_224650

theorem complex_power_four (i : ℂ) (h : i^2 = -1) : (2 + i)^4 = -7 + 24*i := by
  sorry

end complex_power_four_l2246_224650


namespace regression_equation_proof_l2246_224686

theorem regression_equation_proof (x y z : ℝ) (b a : ℝ) :
  (y = Real.exp (b * x + a)) →
  (z = Real.log y) →
  (z = 0.25 * x - 2.58) →
  (y = Real.exp (0.25 * x - 2.58)) := by
  sorry

end regression_equation_proof_l2246_224686


namespace circle_center_radius_sum_l2246_224627

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 - 8 * y - 6 = -2 * y^2 - 8 * x

-- Define the center and radius
def is_center_radius (c d s : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_equation x y ↔ (x - c)^2 + (y - d)^2 = s^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ (c d s : ℝ), is_center_radius c d s ∧ c + d + s = Real.sqrt 7 :=
sorry

end circle_center_radius_sum_l2246_224627


namespace bagel_count_l2246_224644

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the cost of a bagel in cents -/
def bagel_cost : ℕ := 65

/-- Represents the cost of a muffin in cents -/
def muffin_cost : ℕ := 40

/-- Represents the number of days in the week -/
def days_in_week : ℕ := 7

/-- 
Given a 7-day period where either a 40-cent muffin or a 65-cent bagel is bought each day, 
and the total spending is a whole number of dollars, the number of bagels bought must be 4.
-/
theorem bagel_count : 
  ∀ (b : ℕ), 
  b ≤ days_in_week → 
  (bagel_cost * b + muffin_cost * (days_in_week - b)) % cents_per_dollar = 0 → 
  b = 4 := by
sorry

end bagel_count_l2246_224644


namespace ab_value_l2246_224664

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 * b^2 + a^2 * b^3 = 20) : a * b = 2 ∨ a * b = -2 := by
  sorry

end ab_value_l2246_224664


namespace polynomial_equality_l2246_224623

theorem polynomial_equality : 110^5 - 5 * 110^4 + 10 * 110^3 - 10 * 110^2 + 5 * 110 - 1 = 161051000 := by
  sorry

end polynomial_equality_l2246_224623


namespace rocket_max_altitude_l2246_224616

/-- The altitude function of the rocket -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 25

/-- Theorem: The maximum altitude reached by the rocket is 45 meters -/
theorem rocket_max_altitude :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 45 := by
  sorry

end rocket_max_altitude_l2246_224616


namespace heather_counts_209_l2246_224655

-- Define the range of numbers
def range : Set ℕ := {n | 1 ≤ n ∧ n ≤ 500}

-- Define Alice's skipping pattern
def aliceSkips (n : ℕ) : Prop := ∃ k, n = 5 * k - 2 ∧ 1 ≤ k ∧ k ≤ 100

-- Define the general skipping pattern for Barbara and the next 5 students
def otherSkips (n : ℕ) : Prop := ∃ m, n = 3 * m - 1 ∧ ¬(aliceSkips n)

-- Define Heather's number
def heatherNumber : ℕ := 209

-- Theorem statement
theorem heather_counts_209 :
  heatherNumber ∈ range ∧
  ¬(aliceSkips heatherNumber) ∧
  ¬(otherSkips heatherNumber) ∧
  ∀ n ∈ range, n ≠ heatherNumber → aliceSkips n ∨ otherSkips n :=
sorry

end heather_counts_209_l2246_224655


namespace f_passes_through_2_8_f_neg_one_eq_neg_one_l2246_224642

/-- A power function passing through (2, 8) -/
def f (x : ℝ) : ℝ := x^3

/-- The function f passes through (2, 8) -/
theorem f_passes_through_2_8 : f 2 = 8 := by sorry

/-- The value of f(-1) is -1 -/
theorem f_neg_one_eq_neg_one : f (-1) = -1 := by sorry

end f_passes_through_2_8_f_neg_one_eq_neg_one_l2246_224642


namespace enclosing_polygons_sides_l2246_224656

theorem enclosing_polygons_sides (m : ℕ) (n : ℕ) : 
  m = 12 →
  (360 / m : ℚ) / 2 = 360 / n →
  n = 24 := by
  sorry

end enclosing_polygons_sides_l2246_224656


namespace min_value_quadratic_min_value_achieved_l2246_224657

theorem min_value_quadratic (x : ℝ) : x^2 - 6*x + 10 ≥ 1 := by sorry

theorem min_value_achieved : ∃ x : ℝ, x^2 - 6*x + 10 = 1 := by sorry

end min_value_quadratic_min_value_achieved_l2246_224657


namespace solution_set_inequality_l2246_224671

/-- The solution set of the inequality (x-2)(3-x) > 0 is the open interval (2, 3). -/
theorem solution_set_inequality (x : ℝ) : 
  (x - 2) * (3 - x) > 0 ↔ x ∈ Set.Ioo 2 3 := by sorry

end solution_set_inequality_l2246_224671


namespace question_1_question_2_l2246_224618

-- Define propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (2 - x) ≥ 0
def q (m : ℝ) (x : ℝ) : Prop := x^2 + 2*m*x - m + 6 > 0

-- Theorem for question 1
theorem question_1 (m : ℝ) : (∀ x, q m x) → m ∈ Set.Ioo (-3 : ℝ) 2 :=
sorry

-- Theorem for question 2
theorem question_2 (m : ℝ) : 
  ((∀ x, p x → q m x) ∧ (∃ x, q m x ∧ ¬p x)) → m ∈ Set.Ioc (-3 : ℝ) 2 :=
sorry

end question_1_question_2_l2246_224618
