import Mathlib

namespace kateDisprovesPeter_l2686_268645

/-- Represents a card with a character on one side and a natural number on the other -/
structure Card where
  letter : Char
  number : Nat

/-- Checks if a given character is a vowel -/
def isVowel (c : Char) : Bool :=
  c ∈ ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']

/-- Checks if a given natural number is even -/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/-- Represents Peter's statement about vowels and even numbers -/
def petersStatement (c : Card) : Bool :=
  isVowel c.letter → isEven c.number

/-- The set of cards on the table -/
def cardsOnTable : List Card := [
  ⟨'A', 0⟩,  -- Placeholder number
  ⟨'B', 0⟩,  -- Placeholder number
  ⟨'C', 1⟩,  -- Assuming 'C' for the third card
  ⟨'D', 7⟩,  -- The fourth card we know about
  ⟨'U', 0⟩   -- Placeholder number
]

theorem kateDisprovesPeter :
  ∃ (c : Card), c ∈ cardsOnTable ∧ ¬(petersStatement c) ∧ c.number = 7 := by
  sorry

#check kateDisprovesPeter

end kateDisprovesPeter_l2686_268645


namespace equation_roots_imply_m_range_l2686_268622

theorem equation_roots_imply_m_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    4^x₁ - m * 2^(x₁ + 1) + 2 - m = 0 ∧
    4^x₂ - m * 2^(x₂ + 1) + 2 - m = 0) →
  1 < m ∧ m < 2 :=
by sorry

end equation_roots_imply_m_range_l2686_268622


namespace fraction_product_simplification_l2686_268654

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end fraction_product_simplification_l2686_268654


namespace polynomial_value_theorem_l2686_268614

/-- Given a polynomial function g(x) = px^4 + qx^3 + rx^2 + sx + t 
    where g(-1) = 4, prove that 16p - 8q + 4r - 2s + t = 64 -/
theorem polynomial_value_theorem 
  (p q r s t : ℝ) 
  (g : ℝ → ℝ)
  (h1 : ∀ x, g x = p * x^4 + q * x^3 + r * x^2 + s * x + t)
  (h2 : g (-1) = 4) :
  16 * p - 8 * q + 4 * r - 2 * s + t = 64 := by
sorry

end polynomial_value_theorem_l2686_268614


namespace arithmetic_progression_zero_term_l2686_268644

/-- An arithmetic progression with a_{2n} / a_{2m} = -1 has a zero term at position n+m -/
theorem arithmetic_progression_zero_term 
  (a : ℕ → ℝ) 
  (h_arith : ∃ (d : ℝ), ∀ (k : ℕ), a (k + 1) = a k + d) 
  (n m : ℕ) 
  (h_ratio : a (2 * n) / a (2 * m) = -1) :
  ∃ (k : ℕ), k = n + m ∧ a k = 0 := by
sorry

end arithmetic_progression_zero_term_l2686_268644


namespace flight_cost_X_to_Y_l2686_268696

/-- Represents a city in the travel problem -/
inductive City : Type
| X : City
| Y : City
| Z : City

/-- The distance between two cities in kilometers -/
def distance : City → City → ℝ
| City.X, City.Y => 4800
| City.X, City.Z => 4000
| _, _ => 0  -- We don't need other distances for this problem

/-- The cost per kilometer for bus travel -/
def busCostPerKm : ℝ := 0.15

/-- The cost per kilometer for air travel -/
def airCostPerKm : ℝ := 0.12

/-- The booking fee for air travel -/
def airBookingFee : ℝ := 150

/-- The cost of flying between two cities -/
def flightCost (c1 c2 : City) : ℝ :=
  airBookingFee + airCostPerKm * distance c1 c2

/-- The main theorem: The cost of flying from X to Y is $726 -/
theorem flight_cost_X_to_Y : flightCost City.X City.Y = 726 := by
  sorry

end flight_cost_X_to_Y_l2686_268696


namespace distance_covered_l2686_268611

/-- Proves that the distance covered is 30 km given the conditions of the problem -/
theorem distance_covered (D : ℝ) (S : ℝ) : 
  (D / 5 = D / S + 2) →    -- Abhay takes 2 hours more than Sameer
  (D / 10 = D / S - 1) →   -- If Abhay doubles his speed, he takes 1 hour less than Sameer
  D = 30 := by             -- The distance covered is 30 km
sorry

end distance_covered_l2686_268611


namespace mistaken_calculation_correction_l2686_268667

theorem mistaken_calculation_correction (x : ℝ) : 
  x * 4 = 166.08 → x / 4 = 10.38 := by
sorry

end mistaken_calculation_correction_l2686_268667


namespace quadratic_max_value_l2686_268638

/-- The quadratic function f(x) = ax² + 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

/-- The maximum value of f(x) on the interval [-3, 2] -/
def max_value : ℝ := 9

/-- The lower bound of the interval -/
def lower_bound : ℝ := -3

/-- The upper bound of the interval -/
def upper_bound : ℝ := 2

theorem quadratic_max_value (a : ℝ) :
  (∀ x, lower_bound ≤ x ∧ x ≤ upper_bound → f a x ≤ max_value) ∧
  (∃ x, lower_bound ≤ x ∧ x ≤ upper_bound ∧ f a x = max_value) →
  a = 1 ∨ a = -8 :=
sorry

end quadratic_max_value_l2686_268638


namespace sum_of_sixth_powers_l2686_268625

theorem sum_of_sixth_powers (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^3 + b^3 + c^3 = 8)
  (h3 : a^5 + b^5 + c^5 = 32) :
  a^6 + b^6 + c^6 = 64 := by
sorry

end sum_of_sixth_powers_l2686_268625


namespace sum_of_ab_is_fifteen_l2686_268604

-- Define the set of digits
def Digit := Fin 10

-- Define the property of being four different digits
def FourDifferentDigits (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Define the property of (A+B)/(C+D) being an integer
def SumRatioIsInteger (a b c d : Digit) : Prop :=
  ∃ (k : ℕ), (a.val + b.val : ℕ) = k * (c.val + d.val)

-- Define the property of C and D being non-zero
def NonZeroCD (c d : Digit) : Prop :=
  c.val ≠ 0 ∧ d.val ≠ 0

-- Define the property of C and D being as small as possible
def MinimalCD (c d : Digit) : Prop :=
  ∀ (c' d' : Digit), NonZeroCD c' d' → c.val + d.val ≤ c'.val + d'.val

theorem sum_of_ab_is_fifteen :
  ∀ (a b c d : Digit),
    FourDifferentDigits a b c d →
    SumRatioIsInteger a b c d →
    NonZeroCD c d →
    MinimalCD c d →
    a.val + b.val = 15 := by
  sorry

end sum_of_ab_is_fifteen_l2686_268604


namespace oreo_cheesecake_graham_crackers_l2686_268607

theorem oreo_cheesecake_graham_crackers :
  ∀ (G : ℕ) (oreos : ℕ),
  oreos = 15 →
  (∃ (cheesecakes : ℕ),
    cheesecakes * 2 = G - 4 ∧
    cheesecakes * 3 ≤ oreos ∧
    ∀ (c : ℕ), c * 2 ≤ G - 4 ∧ c * 3 ≤ oreos → c ≤ cheesecakes) →
  G = 14 := by sorry

end oreo_cheesecake_graham_crackers_l2686_268607


namespace hayes_laundry_loads_l2686_268635

/-- The number of detergent pods in a pack -/
def pods_per_pack : ℕ := 39

/-- The number of packs Hayes needs for a full year -/
def packs_per_year : ℕ := 4

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The number of loads of laundry Hayes does in a week -/
def loads_per_week : ℕ := (pods_per_pack * packs_per_year) / weeks_per_year

theorem hayes_laundry_loads : loads_per_week = 3 := by sorry

end hayes_laundry_loads_l2686_268635


namespace min_sum_of_primes_l2686_268661

theorem min_sum_of_primes (p q r s : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
  (30 ∣ p * q - r * s) →
  54 ≤ p + q + r + s :=
by sorry

end min_sum_of_primes_l2686_268661


namespace least_subtraction_for_divisibility_l2686_268606

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 33 ∧ 
  (∀ (m : ℕ), m < n → ¬(87 ∣ (13605 - m))) ∧ 
  (87 ∣ (13605 - n)) := by
  sorry

end least_subtraction_for_divisibility_l2686_268606


namespace equation_solution_l2686_268617

theorem equation_solution : ∃ x : ℝ, 
  ((3^2 - 5) / (0.08 * 7 + 2)) + Real.sqrt x = 10 ∧ x = 71.2715625 := by
  sorry

end equation_solution_l2686_268617


namespace sin_graph_shift_l2686_268698

/-- Theorem: Shifting the graph of y = 3sin(2x) to the right by π/16 units 
    results in the graph of y = 3sin(2x - π/8) -/
theorem sin_graph_shift (x : ℝ) : 
  3 * Real.sin (2 * (x - π/16)) = 3 * Real.sin (2 * x - π/8) := by
  sorry

end sin_graph_shift_l2686_268698


namespace prime_sum_2019_power_l2686_268626

theorem prime_sum_2019_power (p q : ℕ) : 
  Prime p → Prime q → p + q = 2019 → (p - 1)^(q - 1) = 1 ∨ (p - 1)^(q - 1) = 2016 := by
  sorry

end prime_sum_2019_power_l2686_268626


namespace inequality_holds_l2686_268620

theorem inequality_holds (a b c : ℝ) (h : a > b) : a * |c| ≥ b * |c| := by
  sorry

end inequality_holds_l2686_268620


namespace right_triangle_ratio_l2686_268662

/-- Given a right triangle with legs a and b, and hypotenuse c, where a:b = 2:5,
    if a perpendicular from the right angle to the hypotenuse divides it into
    segments r (adjacent to a) and s (adjacent to b), then r/s = 4/25. -/
theorem right_triangle_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a ^ 2 + b ^ 2 = c ^ 2 →  -- Pythagorean theorem
  a / b = 2 / 5 →  -- given ratio of legs
  r * s = a * b →  -- geometric mean theorem
  r + s = c →  -- sum of segments equals hypotenuse
  r / s = 4 / 25 := by
sorry

end right_triangle_ratio_l2686_268662


namespace car_speed_proof_l2686_268649

/-- Proves that a car traveling at speed v km/h takes 15 seconds longer to travel 1 kilometer
    than it would at 48 km/h if and only if v = 40 km/h. -/
theorem car_speed_proof (v : ℝ) : v > 0 →
  (1 / v) * 3600 = (1 / 48) * 3600 + 15 ↔ v = 40 := by
  sorry

end car_speed_proof_l2686_268649


namespace first_reduction_percentage_l2686_268634

theorem first_reduction_percentage (x : ℝ) :
  (1 - x / 100) * (1 - 50 / 100) = 1 - 62.5 / 100 →
  x = 25 := by
sorry

end first_reduction_percentage_l2686_268634


namespace arithmetic_sequence_sum_l2686_268602

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The theorem to prove -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 2 + a 4 + a 6 + a 8 + a 10 = 80 →
  a 6 = 16 := by
  sorry

end arithmetic_sequence_sum_l2686_268602


namespace distance_to_point_l2686_268664

theorem distance_to_point : Real.sqrt (8^2 + (-15)^2) = 17 := by sorry

end distance_to_point_l2686_268664


namespace smallest_special_number_l2686_268685

def unit_digit (n : ℕ) : ℕ := n % 10

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem smallest_special_number : 
  ∀ n : ℕ, 
    (unit_digit n = 5 ∧ 
     is_perfect_square n ∧ 
     (∃ k : ℕ, k * k = n ∧ digit_sum k = 9)) → 
    n ≥ 2025 :=
sorry

end smallest_special_number_l2686_268685


namespace fraction_equality_l2686_268683

theorem fraction_equality (x y : ℝ) (h : y / x = 3 / 7) : (x - y) / x = 4 / 7 := by
  sorry

end fraction_equality_l2686_268683


namespace min_points_on_circle_l2686_268655

theorem min_points_on_circle (n : ℕ) (h : n ≥ 3) :
  let N := if (2*n - 1) % 3 = 0 then n else n - 1
  ∀ (S : Finset (Fin (2*n - 1))),
    S.card ≥ N →
    ∃ (i j : Fin (2*n - 1)), i ∈ S ∧ j ∈ S ∧
      (((j - i : ℤ) + (2*n - 1)) % (2*n - 1) = n ∨
       ((i - j : ℤ) + (2*n - 1)) % (2*n - 1) = n) :=
by sorry

end min_points_on_circle_l2686_268655


namespace percentage_increase_l2686_268694

theorem percentage_increase (initial_earnings new_earnings : ℝ) (h1 : initial_earnings = 60) (h2 : new_earnings = 72) :
  (new_earnings - initial_earnings) / initial_earnings * 100 = 20 := by
  sorry

end percentage_increase_l2686_268694


namespace smaller_two_digit_factor_of_4680_l2686_268678

theorem smaller_two_digit_factor_of_4680 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4680 → min a b = 52 := by
  sorry

end smaller_two_digit_factor_of_4680_l2686_268678


namespace problem_solution_l2686_268615

def f (a : ℝ) (x : ℝ) : ℝ := |3*x - a| - 2*|x - 1|

theorem problem_solution :
  (∀ x : ℝ, f (-3) x > 1 ↔ (x < -6 ∨ x > 1)) ∧
  (∃ x : ℝ, f a x ≥ 6 + |x - 1| → (a ≥ 9 ∨ a < -3)) :=
by sorry

end problem_solution_l2686_268615


namespace fifth_group_size_l2686_268672

/-- Represents a choir split into groups -/
structure Choir :=
  (total_members : ℕ)
  (group1 : ℕ)
  (group2 : ℕ)
  (group3 : ℕ)
  (group4 : ℕ)
  (group5 : ℕ)

/-- The choir satisfies the given conditions -/
def choir_conditions (c : Choir) : Prop :=
  c.total_members = 150 ∧
  c.group1 = 18 ∧
  c.group2 = 29 ∧
  c.group3 = 34 ∧
  c.group4 = 23 ∧
  c.total_members = c.group1 + c.group2 + c.group3 + c.group4 + c.group5

/-- Theorem: The fifth group has 46 members -/
theorem fifth_group_size (c : Choir) (h : choir_conditions c) : c.group5 = 46 := by
  sorry

end fifth_group_size_l2686_268672


namespace factor_tree_value_l2686_268686

theorem factor_tree_value (F G H J X : ℕ) : 
  H = 2 * 5 →
  J = 3 * 7 →
  F = 7 * H →
  G = 11 * J →
  X = F * G →
  X = 16170 :=
by
  sorry

end factor_tree_value_l2686_268686


namespace arithmetic_calculation_l2686_268639

theorem arithmetic_calculation : 1435 + 180 / 60 * 3 - 435 = 1009 := by
  sorry

end arithmetic_calculation_l2686_268639


namespace sheet_difference_l2686_268603

theorem sheet_difference : ∀ (tommy jimmy : ℕ), 
  jimmy = 32 →
  jimmy + 40 = tommy + 30 →
  tommy - jimmy = 10 := by
    sorry

end sheet_difference_l2686_268603


namespace small_boxes_count_l2686_268668

theorem small_boxes_count (chocolate_per_small_box : ℕ) (total_chocolate : ℕ) : 
  chocolate_per_small_box = 25 → total_chocolate = 475 → 
  total_chocolate / chocolate_per_small_box = 19 := by
  sorry

end small_boxes_count_l2686_268668


namespace paving_stone_width_l2686_268684

/-- The width of a paving stone given the dimensions of a rectangular courtyard and the number of stones required to pave it. -/
theorem paving_stone_width 
  (courtyard_length : ℝ) 
  (courtyard_width : ℝ) 
  (num_stones : ℕ) 
  (stone_length : ℝ) 
  (h1 : courtyard_length = 50) 
  (h2 : courtyard_width = 16.5) 
  (h3 : num_stones = 165) 
  (h4 : stone_length = 2.5) : 
  ∃ (stone_width : ℝ), stone_width = 2 ∧ 
    courtyard_length * courtyard_width = ↑num_stones * stone_length * stone_width := by
  sorry

end paving_stone_width_l2686_268684


namespace cloth_sold_calculation_l2686_268628

/-- The number of meters of cloth sold by a trader -/
def meters_of_cloth : ℕ := 40

/-- The profit per meter of cloth in rupees -/
def profit_per_meter : ℕ := 30

/-- The total profit earned by the trader in rupees -/
def total_profit : ℕ := 1200

/-- Theorem stating that the number of meters of cloth sold is 40 -/
theorem cloth_sold_calculation :
  meters_of_cloth * profit_per_meter = total_profit :=
by sorry

end cloth_sold_calculation_l2686_268628


namespace five_integers_average_l2686_268621

theorem five_integers_average (a b c d e : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (a + b + c + d + e : ℚ) / 5 = 7 ∧
  ∀ (x y z w v : ℕ+), 
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧
    y ≠ z ∧ y ≠ w ∧ y ≠ v ∧
    z ≠ w ∧ z ≠ v ∧
    w ≠ v ∧
    (x + y + z + w + v : ℚ) / 5 = 7 →
    (max a b - min a b : ℤ) ≥ (max x y - min x y : ℤ) ∧
    (max a c - min a c : ℤ) ≥ (max x z - min x z : ℤ) ∧
    (max a d - min a d : ℤ) ≥ (max x w - min x w : ℤ) ∧
    (max a e - min a e : ℤ) ≥ (max x v - min x v : ℤ) ∧
    (max b c - min b c : ℤ) ≥ (max y z - min y z : ℤ) ∧
    (max b d - min b d : ℤ) ≥ (max y w - min y w : ℤ) ∧
    (max b e - min b e : ℤ) ≥ (max y v - min y v : ℤ) ∧
    (max c d - min c d : ℤ) ≥ (max z w - min z w : ℤ) ∧
    (max c e - min c e : ℤ) ≥ (max z v - min z v : ℤ) ∧
    (max d e - min d e : ℤ) ≥ (max w v - min w v : ℤ) →
  (b + c + d : ℚ) / 3 = 3 := by
sorry

end five_integers_average_l2686_268621


namespace prob_one_or_two_sunny_days_l2686_268636

-- Define the probability of rain
def rain_prob : ℚ := 3/5

-- Define the number of days
def num_days : ℕ := 5

-- Function to calculate the probability of exactly k sunny days
def prob_k_sunny_days (k : ℕ) : ℚ :=
  (num_days.choose k) * (1 - rain_prob)^k * rain_prob^(num_days - k)

-- Theorem statement
theorem prob_one_or_two_sunny_days :
  prob_k_sunny_days 1 + prob_k_sunny_days 2 = 378/625 := by
  sorry

end prob_one_or_two_sunny_days_l2686_268636


namespace total_songs_bought_l2686_268600

/-- The number of country albums Megan bought -/
def country_albums : ℕ := 2

/-- The number of pop albums Megan bought -/
def pop_albums : ℕ := 8

/-- The number of songs in each album -/
def songs_per_album : ℕ := 7

/-- The total number of albums Megan bought -/
def total_albums : ℕ := country_albums + pop_albums

/-- Theorem: The total number of songs Megan bought is 70 -/
theorem total_songs_bought : total_albums * songs_per_album = 70 := by
  sorry

end total_songs_bought_l2686_268600


namespace equation_holds_for_all_x_l2686_268653

theorem equation_holds_for_all_x : ∃ (a b c : ℝ), ∀ (x : ℝ), 
  (x + a)^2 + (2*x + b)^2 + (2*x + c)^2 = (3*x + 1)^2 := by
  sorry

end equation_holds_for_all_x_l2686_268653


namespace bumper_car_line_l2686_268643

/-- The number of people initially in line for bumper cars -/
def initial_people : ℕ := sorry

/-- The number of people in line after 2 leave and 2 join -/
def final_people : ℕ := 10

/-- The condition that if 2 people leave and 2 join, there are 10 people in line -/
axiom condition : initial_people = final_people

theorem bumper_car_line : initial_people = 10 := by sorry

end bumper_car_line_l2686_268643


namespace product_of_roots_plus_one_l2686_268624

theorem product_of_roots_plus_one (a b c : ℝ) : 
  (x^3 - 15*x^2 + 25*x - 12 = 0 → x = a ∨ x = b ∨ x = c) →
  (1 + a) * (1 + b) * (1 + c) = 53 := by
  sorry

end product_of_roots_plus_one_l2686_268624


namespace geometric_sequence_ratio_l2686_268632

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n ∧ a n > 0

/-- Three terms form an arithmetic sequence -/
def ArithmeticSequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticSequence (3 * a 1) ((1 / 2) * a 3) (2 * a 2) →
  (a 11 + a 13) / (a 8 + a 10) = 27 := by
  sorry

end geometric_sequence_ratio_l2686_268632


namespace sqrt_sum_equality_l2686_268650

theorem sqrt_sum_equality (a b c k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k > 0)
  (h : 2 * a * b * c + k * (a^2 + b^2 + c^2) = k^3) :
  Real.sqrt ((k - a) * (k - b) / ((k + a) * (k + b))) +
  Real.sqrt ((k - b) * (k - c) / ((k + b) * (k + c))) +
  Real.sqrt ((k - c) * (k - a) / ((k + c) * (k + a))) = 1 := by
  sorry

end sqrt_sum_equality_l2686_268650


namespace new_average_after_adding_l2686_268640

theorem new_average_after_adding (n : ℕ) (original_avg : ℝ) (added_value : ℝ) :
  n > 0 →
  n = 15 →
  original_avg = 40 →
  added_value = 14 →
  (n * original_avg + n * added_value) / n = 54 := by
  sorry

end new_average_after_adding_l2686_268640


namespace sets_properties_l2686_268679

def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) = 0}

theorem sets_properties :
  (A = {-1, 3}) ∧
  (∀ a : ℝ, {-1, 1, 3} ⊆ A ∪ B a) ∧
  (∀ a : ℝ, a ≠ -1 ∧ a ≠ 1 ∧ a ≠ 3 → A ∪ B a = {-1, 1, 3, a}) ∧
  (A ∩ B 1 = ∅) ∧
  (∀ a : ℝ, a ≠ -1 ∧ a ≠ 1 ∧ a ≠ 3 → A ∩ B a = ∅) ∧
  (A ∩ B (-1) = {-1}) ∧
  (A ∩ B 3 = {3}) := by
  sorry

end sets_properties_l2686_268679


namespace cube_root_always_real_l2686_268663

theorem cube_root_always_real : 
  ∀ x : ℝ, ∃ y : ℝ, y^3 = -(x + 3)^3 :=
by
  sorry

end cube_root_always_real_l2686_268663


namespace orange_juice_percentage_l2686_268674

theorem orange_juice_percentage
  (total_volume : ℝ)
  (watermelon_percentage : ℝ)
  (grape_volume : ℝ)
  (h1 : total_volume = 300)
  (h2 : watermelon_percentage = 40)
  (h3 : grape_volume = 105) :
  (total_volume - watermelon_percentage / 100 * total_volume - grape_volume) / total_volume * 100 = 25 := by
  sorry

end orange_juice_percentage_l2686_268674


namespace equal_sum_sequence_properties_l2686_268618

/-- An equal sum sequence is a sequence where each term plus the previous term
    equals the same constant, starting from the second term. -/
def EqualSumSequence (a : ℕ → ℝ) :=
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = k

theorem equal_sum_sequence_properties (a : ℕ → ℝ) (h : EqualSumSequence a) :
  (∀ n : ℕ, n ≥ 1 → a n = a (n + 2)) ∧
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → Odd m ∧ Odd n → a m = a n) ∧
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → Even m ∧ Even n → a m = a n) :=
by sorry

end equal_sum_sequence_properties_l2686_268618


namespace protons_equal_atomic_number_oxygen16_protons_l2686_268699

/-- Represents an atom with mass number and atomic number -/
structure Atom where
  mass_number : ℕ
  atomic_number : ℕ

/-- The oxygen-16 atom -/
def oxygen16 : Atom := { mass_number := 16, atomic_number := 8 }

/-- The number of protons in an atom is equal to its atomic number -/
theorem protons_equal_atomic_number (a : Atom) : a.atomic_number = a.atomic_number := by sorry

theorem oxygen16_protons : oxygen16.atomic_number = 8 := by sorry

end protons_equal_atomic_number_oxygen16_protons_l2686_268699


namespace legos_won_l2686_268647

def initial_legos : ℕ := 2080
def final_legos : ℕ := 2097

theorem legos_won : final_legos - initial_legos = 17 := by
  sorry

end legos_won_l2686_268647


namespace parallelogram_condition_l2686_268608

/-- The condition for the existence of a parallelogram inscribed in an ellipse and tangent to a circle -/
theorem parallelogram_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∀ (x y : ℝ), x^2 + y^2 = 1 →
    ∃ (P : ℝ × ℝ), P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧
      ∃ (Q R S : ℝ × ℝ),
        Q.1^2 / a^2 + Q.2^2 / b^2 = 1 ∧
        R.1^2 / a^2 + R.2^2 / b^2 = 1 ∧
        S.1^2 / a^2 + S.2^2 / b^2 = 1 ∧
        (P.1 - Q.1) * (R.1 - S.1) + (P.2 - Q.2) * (R.2 - S.2) = 0 ∧
        (P.1 - R.1) * (Q.1 - S.1) + (P.2 - R.2) * (Q.2 - S.2) = 0 ∧
        ((P.1 - x)^2 + (P.2 - y)^2 = 1 ∨
         (Q.1 - x)^2 + (Q.2 - y)^2 = 1 ∨
         (R.1 - x)^2 + (R.2 - y)^2 = 1 ∨
         (S.1 - x)^2 + (S.2 - y)^2 = 1)) ↔
  1 / a^2 + 1 / b^2 = 1 :=
by sorry

end parallelogram_condition_l2686_268608


namespace dance_relationship_l2686_268609

/-- The number of girls that the nth boy dances with -/
def girls_danced (n : ℕ) : ℕ := n + 7

/-- The relationship between the number of boys (b) and girls (g) at a school dance -/
theorem dance_relationship (b g : ℕ) : 
  (∀ n : ℕ, n ≤ b → girls_danced n ≤ g) → 
  girls_danced b = g → 
  b = g - 7 :=
by sorry

end dance_relationship_l2686_268609


namespace crayon_production_in_four_hours_l2686_268651

/-- Represents a crayon factory with given specifications -/
structure CrayonFactory where
  colors : Nat
  crayonsPerColorPerBox : Nat
  boxesPerHour : Nat

/-- Calculates the total number of crayons produced in a given number of hours -/
def totalCrayonsProduced (factory : CrayonFactory) (hours : Nat) : Nat :=
  factory.colors * factory.crayonsPerColorPerBox * factory.boxesPerHour * hours

/-- Theorem stating that a factory with given specifications produces 160 crayons in 4 hours -/
theorem crayon_production_in_four_hours :
  ∀ (factory : CrayonFactory),
    factory.colors = 4 →
    factory.crayonsPerColorPerBox = 2 →
    factory.boxesPerHour = 5 →
    totalCrayonsProduced factory 4 = 160 :=
by sorry

end crayon_production_in_four_hours_l2686_268651


namespace tank_capacity_l2686_268646

theorem tank_capacity (C : ℚ) : 
  (C > 0) →  -- The capacity is positive
  ((117 / 200) * C = 4680) →  -- Final volume equation
  (C = 8000) := by
sorry

end tank_capacity_l2686_268646


namespace greatest_integer_with_gcf_two_ninety_eight_satisfies_conditions_ninety_eight_is_greatest_l2686_268690

theorem greatest_integer_with_gcf_two (n : ℕ) : n < 100 → Nat.gcd n 12 = 2 → n ≤ 98 := by
  sorry

theorem ninety_eight_satisfies_conditions : 
  98 < 100 ∧ Nat.gcd 98 12 = 2 := by
  sorry

theorem ninety_eight_is_greatest : 
  ∀ (m : ℕ), m < 100 → Nat.gcd m 12 = 2 → m ≤ 98 := by
  sorry

end greatest_integer_with_gcf_two_ninety_eight_satisfies_conditions_ninety_eight_is_greatest_l2686_268690


namespace david_started_with_at_least_six_iphones_l2686_268631

/-- Represents the number of cell phones in various categories -/
structure CellPhoneInventory where
  samsung_end : ℕ
  iphone_end : ℕ
  samsung_damaged : ℕ
  iphone_defective : ℕ
  total_sold : ℕ

/-- Given the end-of-day inventory and sales data, proves that David started with at least 6 iPhones -/
theorem david_started_with_at_least_six_iphones 
  (inventory : CellPhoneInventory)
  (h1 : inventory.samsung_end = 10)
  (h2 : inventory.iphone_end = 5)
  (h3 : inventory.samsung_damaged = 2)
  (h4 : inventory.iphone_defective = 1)
  (h5 : inventory.total_sold = 4) :
  ∃ (initial_iphones : ℕ), initial_iphones ≥ 6 ∧ 
    initial_iphones ≥ inventory.iphone_end + inventory.iphone_defective :=
by sorry

end david_started_with_at_least_six_iphones_l2686_268631


namespace village_language_problem_l2686_268612

theorem village_language_problem (total_population : ℕ) 
  (tamil_speakers : ℕ) (english_speakers : ℕ) (hindi_probability : ℚ) :
  total_population = 1024 →
  tamil_speakers = 720 →
  english_speakers = 562 →
  hindi_probability = 0.0859375 →
  ∃ (both_speakers : ℕ),
    both_speakers = 434 ∧
    total_population = tamil_speakers + english_speakers - both_speakers + 
      (↑total_population * hindi_probability).floor := by
  sorry

end village_language_problem_l2686_268612


namespace paving_cost_l2686_268695

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) : 
  length = 6.5 → 
  width = 2.75 → 
  rate = 600 → 
  length * width * rate = 10725 := by
sorry

end paving_cost_l2686_268695


namespace younger_son_age_in_30_years_l2686_268675

/-- Given an elder son's age and the age difference between two sons, 
    calculate the younger son's age after a certain number of years. -/
def younger_son_future_age (elder_son_age : ℕ) (age_difference : ℕ) (years_from_now : ℕ) : ℕ :=
  (elder_son_age - age_difference) + years_from_now

theorem younger_son_age_in_30_years :
  younger_son_future_age 40 10 30 = 60 := by
  sorry

end younger_son_age_in_30_years_l2686_268675


namespace sum_three_numbers_l2686_268682

theorem sum_three_numbers (a b c N : ℝ) : 
  a + b + c = 60 ∧ 
  a - 7 = N ∧ 
  b + 7 = N ∧ 
  7 * c = N → 
  N = 28 := by
sorry

end sum_three_numbers_l2686_268682


namespace circle_radius_proof_l2686_268687

theorem circle_radius_proof (r p q : ℕ) (m n : ℕ+) :
  -- r is an odd integer
  Odd r →
  -- p and q are prime numbers
  Nat.Prime p →
  Nat.Prime q →
  -- (p^m, q^n) is on the circle with radius r
  p^(m:ℕ) * p^(m:ℕ) + q^(n:ℕ) * q^(n:ℕ) = r * r →
  -- The radius r is equal to 5
  r = 5 := by sorry

end circle_radius_proof_l2686_268687


namespace correct_proposition_l2686_268627

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a > b → a * c^2 > b * c^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 - Real.log x₀ = 0

-- Theorem to prove
theorem correct_proposition : ¬p ∧ q := by sorry

end correct_proposition_l2686_268627


namespace parabola_midpoint_distance_l2686_268630

/-- Given a parabola y² = 4x and a line passing through its focus,
    intersecting the parabola at points A and B with |AB| = 7,
    the distance from the midpoint M of AB to the directrix is 7/2. -/
theorem parabola_midpoint_distance (x₁ y₁ x₂ y₂ : ℝ) :
  y₁^2 = 4*x₁ →
  y₂^2 = 4*x₂ →
  (x₁ - 1)^2 + y₁^2 = (x₂ - 1)^2 + y₂^2 →  -- line passes through focus (1, 0)
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 49 →         -- |AB| = 7
  (((x₁ + x₂)/2 + 1) : ℝ) = 7/2 :=
by sorry

end parabola_midpoint_distance_l2686_268630


namespace second_expression_proof_l2686_268681

theorem second_expression_proof (a x : ℝ) (h1 : ((2 * a + 16) + x) / 2 = 74) (h2 : a = 28) : x = 76 := by
  sorry

end second_expression_proof_l2686_268681


namespace S_is_three_rays_l2686_268641

/-- The set S of points (x,y) in the coordinate plane satisfying the given conditions -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (4 = x + 1 ∧ y - 3 ≤ 4) ∨
               (4 = y - 3 ∧ x + 1 ≤ 4) ∨
               (x + 1 = y - 3 ∧ 4 ≤ x + 1)}

/-- A ray starting from a point in a given direction -/
def Ray (start : ℝ × ℝ) (direction : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (start.1 + t * direction.1, start.2 + t * direction.2)}

/-- The theorem stating that S consists of three rays with a common point -/
theorem S_is_three_rays :
  ∃ (r₁ r₂ r₃ : Set (ℝ × ℝ)) (common_point : ℝ × ℝ),
    S = r₁ ∪ r₂ ∪ r₃ ∧
    (∃ d₁ d₂ d₃ : ℝ × ℝ, r₁ = Ray common_point d₁ ∧
                         r₂ = Ray common_point d₂ ∧
                         r₃ = Ray common_point d₃) ∧
    common_point = (3, 7) := by
  sorry

end S_is_three_rays_l2686_268641


namespace eric_pencil_boxes_l2686_268656

theorem eric_pencil_boxes (pencils_per_box : ℕ) (total_pencils : ℕ) (h1 : pencils_per_box = 9) (h2 : total_pencils = 27) :
  total_pencils / pencils_per_box = 3 :=
by sorry

end eric_pencil_boxes_l2686_268656


namespace intersection_implies_a_equals_three_l2686_268633

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {3, 4, 2*a - 4}
def B (a : ℝ) : Set ℝ := {a}

-- State the theorem
theorem intersection_implies_a_equals_three (a : ℝ) :
  (A a ∩ B a).Nonempty → a = 3 := by
  sorry

end intersection_implies_a_equals_three_l2686_268633


namespace unique_zero_iff_a_nonpositive_l2686_268605

/-- A function f(x) = x^3 - 3ax has a unique zero if and only if a ≤ 0 -/
theorem unique_zero_iff_a_nonpositive (a : ℝ) :
  (∃! x, x^3 - 3*a*x = 0) ↔ a ≤ 0 := by sorry

end unique_zero_iff_a_nonpositive_l2686_268605


namespace card_game_result_l2686_268677

/-- Represents the number of cards in each pile -/
structure CardPiles :=
  (left : ℕ)
  (middle : ℕ)
  (right : ℕ)

/-- The card game operations -/
def card_game_operations (initial : CardPiles) : CardPiles :=
  let step1 := initial
  let step2 := CardPiles.mk (step1.left - 2) (step1.middle + 2) step1.right
  let step3 := CardPiles.mk step2.left (step2.middle + 1) (step2.right - 1)
  CardPiles.mk step3.left.succ (step3.middle - step3.left) step3.right

theorem card_game_result (initial : CardPiles) 
  (h1 : initial.left = initial.middle)
  (h2 : initial.middle = initial.right)
  (h3 : initial.left ≥ 2) :
  (card_game_operations initial).middle = 5 :=
sorry

end card_game_result_l2686_268677


namespace difference_of_squares_l2686_268689

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end difference_of_squares_l2686_268689


namespace solution_values_solution_set_when_a_negative_l2686_268671

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - a

-- Define the solution set condition
def hasSolutionSet (a b : ℝ) : Prop :=
  ∀ x, f a x < b ↔ x < -1 ∨ x > 3

-- Theorem statement
theorem solution_values (a b : ℝ) (h : hasSolutionSet a b) : a = -1/2 ∧ b = -1 := by
  sorry

-- Additional theorem for part 2
theorem solution_set_when_a_negative (a : ℝ) (h : a < 0) :
  (∀ x, f a x > 1 ↔ 
    (a < -1/2 ∧ -((a+1)/a) < x ∧ x < 1) ∨
    (a = -1/2 ∧ False) ∨
    (-1/2 < a ∧ a < 0 ∧ 1 < x ∧ x < -((a+1)/a))) := by
  sorry

end solution_values_solution_set_when_a_negative_l2686_268671


namespace fraction_sum_equality_l2686_268623

theorem fraction_sum_equality : ∃ (a b c d e f : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
   d ≠ e ∧ d ≠ f ∧
   e ≠ f) ∧
  (a ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   b ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   c ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   d ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   e ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   f ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ)) ∧
  (Nat.gcd a b = 1 ∧ Nat.gcd c d = 1 ∧ Nat.gcd e f = 1) ∧
  (a * d * f + c * b * f = e * b * d) ∧
  (b ≠ 0 ∧ d ≠ 0 ∧ f ≠ 0) :=
by sorry

end fraction_sum_equality_l2686_268623


namespace curve_is_line_l2686_268666

/-- The curve defined by the polar equation θ = 5π/6 is a line -/
theorem curve_is_line : ∀ (r : ℝ) (θ : ℝ), 
  θ = (5 * Real.pi) / 6 → 
  ∃ (a b : ℝ), ∀ (x y : ℝ), x = r * Real.cos θ ∧ y = r * Real.sin θ → 
  a * x + b * y = 0 :=
sorry

end curve_is_line_l2686_268666


namespace cos_alpha_plus_pi_fourth_l2686_268616

theorem cos_alpha_plus_pi_fourth (α : Real) 
  (h1 : π/2 < α ∧ α < π)  -- α is an obtuse angle
  (h2 : Real.sin (α - 3*π/4) = 3/5) :
  Real.cos (α + π/4) = -4/5 := by
sorry

end cos_alpha_plus_pi_fourth_l2686_268616


namespace parade_probability_l2686_268669

/-- The number of possible permutations of n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The probability of an event occurring, given the number of favorable outcomes and total outcomes -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := 
  if total = 0 then 0 else (favorable : ℚ) / (total : ℚ)

/-- The number of formations in the parade -/
def num_formations : ℕ := 3

/-- The number of favorable outcomes (B passes before both A and C) -/
def favorable_outcomes : ℕ := 2

theorem parade_probability :
  probability favorable_outcomes (factorial num_formations) = 1 / 3 := by
  sorry


end parade_probability_l2686_268669


namespace sum_equals_200_l2686_268665

theorem sum_equals_200 : 148 + 32 + 18 + 2 = 200 := by
  sorry

end sum_equals_200_l2686_268665


namespace expression_simplification_l2686_268680

theorem expression_simplification (y : ℝ) : 
  3*y - 7*y^2 + 15 - (6 - 5*y + 7*y^2) = -14*y^2 + 8*y + 9 := by
  sorry

end expression_simplification_l2686_268680


namespace teacher_pen_cost_l2686_268660

/-- The total cost of pens purchased by a teacher -/
theorem teacher_pen_cost : 
  let black_pens : ℕ := 7
  let blue_pens : ℕ := 9
  let red_pens : ℕ := 5
  let black_pen_cost : ℚ := 125/100
  let blue_pen_cost : ℚ := 150/100
  let red_pen_cost : ℚ := 175/100
  (black_pens : ℚ) * black_pen_cost + 
  (blue_pens : ℚ) * blue_pen_cost + 
  (red_pens : ℚ) * red_pen_cost = 31 :=
by sorry

end teacher_pen_cost_l2686_268660


namespace power_45_equals_a_squared_b_l2686_268652

theorem power_45_equals_a_squared_b (x a b : ℝ) (h1 : 3^x = a) (h2 : 5^x = b) : 45^x = a^2 * b := by
  sorry

end power_45_equals_a_squared_b_l2686_268652


namespace total_payment_proof_l2686_268673

def apple_quantity : ℕ := 15
def apple_price : ℕ := 85
def mango_quantity : ℕ := 12
def mango_price : ℕ := 60
def grape_quantity : ℕ := 10
def grape_price : ℕ := 75
def strawberry_quantity : ℕ := 6
def strawberry_price : ℕ := 150

def total_cost : ℕ := 
  apple_quantity * apple_price + 
  mango_quantity * mango_price + 
  grape_quantity * grape_price + 
  strawberry_quantity * strawberry_price

theorem total_payment_proof : total_cost = 3645 := by
  sorry

end total_payment_proof_l2686_268673


namespace inverse_variation_problem_l2686_268657

theorem inverse_variation_problem (k : ℝ) (x y : ℝ → ℝ) (h1 : ∀ t, 5 * y t = k / (x t)^2)
  (h2 : y 1 = 16) (h3 : x 1 = 1) : y 8 = 1/4 := by
  sorry

end inverse_variation_problem_l2686_268657


namespace min_sum_of_squares_l2686_268693

theorem min_sum_of_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 3 * x₂ + 5 * x₃ = 100) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 2000 / 7 ∧ 
  ∃ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ 
    y₁ + 3 * y₂ + 5 * y₃ = 100 ∧ 
    y₁^2 + y₂^2 + y₃^2 = 2000 / 7 :=
by sorry

end min_sum_of_squares_l2686_268693


namespace gcd_228_1995_base_conversion_l2686_268629

-- Problem 1: GCD of 228 and 1995
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by sorry

-- Problem 2: Base conversion
theorem base_conversion :
  (1 * 3^4 + 1 * 3^3 + 1 * 3^2 + 0 * 3^1 + 2 * 3^0) = (3 * 6^2 + 1 * 6^1 + 5 * 6^0) := by sorry

end gcd_228_1995_base_conversion_l2686_268629


namespace sin_90_degrees_l2686_268637

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end sin_90_degrees_l2686_268637


namespace arrangements_count_l2686_268601

/-- The number of candidates -/
def total_candidates : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The number of students who can be assigned to swimming -/
def swimming_candidates : ℕ := total_candidates - 1

/-- The number of different arrangements -/
def arrangements : ℕ := swimming_candidates * (total_candidates - 1) * (total_candidates - 2)

theorem arrangements_count : arrangements = 48 := by
  sorry

end arrangements_count_l2686_268601


namespace weight_puzzle_l2686_268648

theorem weight_puzzle (w₁ w₂ w₃ w₄ : ℕ) 
  (h1 : w₁ + w₂ = 1700 ∨ w₁ + w₃ = 1700 ∨ w₁ + w₄ = 1700 ∨ w₂ + w₃ = 1700 ∨ w₂ + w₄ = 1700 ∨ w₃ + w₄ = 1700)
  (h2 : w₁ + w₂ = 1870 ∨ w₁ + w₃ = 1870 ∨ w₁ + w₄ = 1870 ∨ w₂ + w₃ = 1870 ∨ w₂ + w₄ = 1870 ∨ w₃ + w₄ = 1870)
  (h3 : w₁ + w₂ = 2110 ∨ w₁ + w₃ = 2110 ∨ w₁ + w₄ = 2110 ∨ w₂ + w₃ = 2110 ∨ w₂ + w₄ = 2110 ∨ w₃ + w₄ = 2110)
  (h4 : w₁ + w₂ = 2330 ∨ w₁ + w₃ = 2330 ∨ w₁ + w₄ = 2330 ∨ w₂ + w₃ = 2330 ∨ w₂ + w₄ = 2330 ∨ w₃ + w₄ = 2330)
  (h5 : w₁ + w₂ = 2500 ∨ w₁ + w₃ = 2500 ∨ w₁ + w₄ = 2500 ∨ w₂ + w₃ = 2500 ∨ w₂ + w₄ = 2500 ∨ w₃ + w₄ = 2500)
  (h_distinct : w₁ ≠ w₂ ∧ w₁ ≠ w₃ ∧ w₁ ≠ w₄ ∧ w₂ ≠ w₃ ∧ w₂ ≠ w₄ ∧ w₃ ≠ w₄) :
  w₁ + w₂ = 2090 ∨ w₁ + w₃ = 2090 ∨ w₁ + w₄ = 2090 ∨ w₂ + w₃ = 2090 ∨ w₂ + w₄ = 2090 ∨ w₃ + w₄ = 2090 :=
by sorry

end weight_puzzle_l2686_268648


namespace intersection_A_B_union_A_C_R_B_l2686_268691

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}

-- Define the complement of B in ℝ
def C_R_B : Set ℝ := {x | ¬ (x ∈ B)}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

-- Theorem for the union of A and complement of B
theorem union_A_C_R_B : A ∪ C_R_B = {x : ℝ | -4 < x ∧ x < 3} := by sorry

end intersection_A_B_union_A_C_R_B_l2686_268691


namespace discounted_price_is_nine_l2686_268642

/-- The final price after applying a discount --/
def final_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

/-- Theorem: The final price of a $10 item after a 10% discount is $9 --/
theorem discounted_price_is_nine :
  final_price 10 0.1 = 9 := by
  sorry

end discounted_price_is_nine_l2686_268642


namespace tangent_line_at_zero_l2686_268613

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / (Real.exp x)

theorem tangent_line_at_zero (x y : ℝ) :
  (∃ (m : ℝ), HasDerivAt f m 0 ∧ m = -1) →
  f 0 = 1 →
  (x + y - 1 = 0 ↔ y - f 0 = m * (x - 0)) :=
by sorry

end tangent_line_at_zero_l2686_268613


namespace age_problem_l2686_268658

theorem age_problem (P R J M : ℕ) : 
  P = R / 2 →
  R + 12 = J + 12 + 7 →
  J + 12 = 3 * P →
  M + 8 = J + 8 + 9 →
  M + 4 = 2 * (R + 4) →
  P = 5 ∧ R = 10 ∧ J = 3 ∧ M = 24 := by
sorry

end age_problem_l2686_268658


namespace inequality_proof_l2686_268619

theorem inequality_proof (w x y z : ℝ) (h : w^2 + y^2 ≤ 1) :
  (w*x + y*z - 1)^2 ≥ (w^2 + y^2 - 1)*(x^2 + z^2 - 1) := by
  sorry

end inequality_proof_l2686_268619


namespace triangle_perimeter_lower_bound_l2686_268659

theorem triangle_perimeter_lower_bound 
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (h : ℝ) -- Height on side BC
  (ha : a = 1) -- Side a equals 1
  (hh : h = Real.tan A) -- Height equals tan A
  (hA : 0 < A ∧ A < Real.pi / 2) -- A is in the range (0, π/2)
  (hS : (1/2) * a * h = (1/2) * b * c * Real.sin A) -- Area formula
  (hC : c^2 = a^2 + b^2 - 2*a*b*Real.cos C) -- Cosine rule
  : a + b + c > Real.sqrt 5 + 1 := by sorry

end triangle_perimeter_lower_bound_l2686_268659


namespace non_intersecting_chords_eq_catalan_number_l2686_268688

/-- The number of ways to draw n non-intersecting chords joining 2n points on a circle's circumference -/
def numberOfNonIntersectingChords (n : ℕ) : ℕ :=
  Nat.choose (2 * n) n / (n + 1)

/-- The nth Catalan number -/
def catalanNumber (n : ℕ) : ℕ :=
  Nat.choose (2 * n) n / (n + 1)

theorem non_intersecting_chords_eq_catalan_number :
  numberOfNonIntersectingChords 6 = catalanNumber 6 := by
  sorry

end non_intersecting_chords_eq_catalan_number_l2686_268688


namespace carol_weight_l2686_268676

/-- Given that Alice and Carol have a combined weight of 280 pounds,
    and the difference between Carol's and Alice's weights is one-third of Carol's weight,
    prove that Carol weighs 168 pounds. -/
theorem carol_weight (alice_weight carol_weight : ℝ) 
    (h1 : alice_weight + carol_weight = 280)
    (h2 : carol_weight - alice_weight = carol_weight / 3) :
    carol_weight = 168 := by
  sorry

end carol_weight_l2686_268676


namespace max_sum_squared_distances_l2686_268610

theorem max_sum_squared_distances (a b c d : Fin 4 → ℝ) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  (‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2) ≤ 24 :=
by sorry

end max_sum_squared_distances_l2686_268610


namespace rectangle_division_l2686_268692

theorem rectangle_division (original_width original_height : ℕ) 
  (piece1_width piece1_height : ℕ) (piece2_width piece2_height : ℕ) 
  (piece3_width piece3_height : ℕ) (piece4_width piece4_height : ℕ) :
  original_width = 15 ∧ original_height = 7 ∧
  piece1_width = 7 ∧ piece1_height = 7 ∧
  piece2_width = 8 ∧ piece2_height = 3 ∧
  piece3_width = 7 ∧ piece3_height = 4 ∧
  piece4_width = 8 ∧ piece4_height = 4 →
  original_width * original_height = 
    piece1_width * piece1_height + 
    piece2_width * piece2_height + 
    piece3_width * piece3_height + 
    piece4_width * piece4_height :=
by
  sorry

end rectangle_division_l2686_268692


namespace smallest_natural_number_divisibility_l2686_268697

theorem smallest_natural_number_divisibility : ∃! n : ℕ,
  (∀ m : ℕ, m < n → ¬((m + 2018) % 2020 = 0 ∧ (m + 2020) % 2018 = 0)) ∧
  (n + 2018) % 2020 = 0 ∧
  (n + 2020) % 2018 = 0 ∧
  n = 2030102 := by
  sorry

end smallest_natural_number_divisibility_l2686_268697


namespace rectangular_prism_problem_l2686_268670

/-- The number of valid triples (a, b, c) for the rectangular prism problem -/
def valid_triples : Nat :=
  (Finset.filter (fun a => a < 1995 ∧ (1995 * 1995) % a = 0)
    (Finset.range 1995)).card

/-- The theorem stating that there are exactly 40 valid triples -/
theorem rectangular_prism_problem :
  valid_triples = 40 := by
  sorry

end rectangular_prism_problem_l2686_268670
