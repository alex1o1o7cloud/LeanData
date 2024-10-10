import Mathlib

namespace intersection_and_perpendicular_line_l481_48131

/-- Given three lines in the xy-plane:
    L₁: x + y - 2 = 0
    L₂: 3x + 2y - 5 = 0
    L₃: 3x + 4y - 12 = 0
    Prove that the line L: 4x - 3y - 1 = 0 passes through the intersection of L₁ and L₂,
    and is perpendicular to L₃. -/
theorem intersection_and_perpendicular_line 
  (L₁ : Set (ℝ × ℝ) := {p | p.1 + p.2 - 2 = 0})
  (L₂ : Set (ℝ × ℝ) := {p | 3 * p.1 + 2 * p.2 - 5 = 0})
  (L₃ : Set (ℝ × ℝ) := {p | 3 * p.1 + 4 * p.2 - 12 = 0})
  (L : Set (ℝ × ℝ) := {p | 4 * p.1 - 3 * p.2 - 1 = 0}) :
  (∃ p, p ∈ L₁ ∩ L₂ ∧ p ∈ L) ∧
  (∀ p q : ℝ × ℝ, p ≠ q → p ∈ L → q ∈ L → p ∈ L₃ → q ∈ L₃ → 
    (p.1 - q.1) * (p.1 - q.1) + (p.2 - q.2) * (p.2 - q.2) = 0) :=
by sorry

end intersection_and_perpendicular_line_l481_48131


namespace base_conversion_sum_l481_48112

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The value of C in base 13 -/
def C : ℕ := 12

/-- The first number in base 9 -/
def num1 : List ℕ := [7, 5, 2]

/-- The second number in base 13 -/
def num2 : List ℕ := [6, C, 3]

theorem base_conversion_sum :
  to_base_10 num1 9 + to_base_10 num2 13 = 1787 := by
  sorry


end base_conversion_sum_l481_48112


namespace some_value_is_zero_l481_48136

theorem some_value_is_zero (x y w : ℝ) (some_value : ℝ) 
  (h1 : some_value + 3 / x = 3 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 1 / 2) :
  some_value = 0 := by
sorry

end some_value_is_zero_l481_48136


namespace smallest_bob_number_l481_48111

def alice_number : ℕ := 30

def has_all_prime_factors (a b : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ a → p ∣ b)

theorem smallest_bob_number (bob_number : ℕ) 
  (h1 : has_all_prime_factors alice_number bob_number)
  (h2 : 5 ∣ bob_number) :
  bob_number ≥ 30 := by
sorry

end smallest_bob_number_l481_48111


namespace ceiling_floor_difference_l481_48130

theorem ceiling_floor_difference (x : ℝ) 
  (h : ⌈x⌉ - ⌊x⌋ = 2) : 
  3 * (⌈x⌉ - x) = 6 - 3 * (x - ⌊x⌋) := by
  sorry

end ceiling_floor_difference_l481_48130


namespace notebook_distribution_l481_48105

theorem notebook_distribution (S : ℕ) : 
  (S / 8 : ℚ) = 16 → S * (S / 8 : ℚ) = 2048 := by
  sorry

end notebook_distribution_l481_48105


namespace complex_magnitude_l481_48144

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 1 + Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l481_48144


namespace monotonic_increasing_condition_l481_48127

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x - 5

-- State the theorem
theorem monotonic_increasing_condition (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) → m ≥ 4/3 := by sorry

end monotonic_increasing_condition_l481_48127


namespace expression_evaluation_l481_48107

theorem expression_evaluation (a b c : ℝ) (h1 : a = 12) (h2 : b = 14) (h3 : c = 20) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end expression_evaluation_l481_48107


namespace locus_equals_homothety_image_l481_48117

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- Represents a rotational homothety transformation -/
structure RotationalHomothety where
  center : Point
  angle : ℝ
  factor : ℝ

/-- The locus of points Y for a given semicircle and constant k -/
def locusOfY (s : Semicircle) (k : ℝ) : Set Point :=
  sorry

/-- The image of a semicircle under rotational homothety -/
def imageUnderHomothety (s : Semicircle) (h : RotationalHomothety) : Set Point :=
  sorry

/-- Main theorem: The locus of Y is the image of the semicircle under rotational homothety -/
theorem locus_equals_homothety_image (s : Semicircle) (k : ℝ) (h0 : k > 0) :
  locusOfY s k = imageUnderHomothety s ⟨s.center, Real.arctan k, Real.sqrt (k^2 + 1)⟩ :=
sorry

end locus_equals_homothety_image_l481_48117


namespace decimal_subtraction_l481_48134

theorem decimal_subtraction :
  let largest_three_digit := 0.999
  let smallest_four_digit := 0.0001
  largest_three_digit - smallest_four_digit = 0.9989 := by
  sorry

end decimal_subtraction_l481_48134


namespace benny_market_money_l481_48156

/-- The amount of money Benny took to the market --/
def money_taken : ℕ → ℕ → ℕ → ℕ
  | num_kids, apples_per_kid, cost_per_apple =>
    num_kids * apples_per_kid * cost_per_apple

theorem benny_market_money :
  money_taken 18 5 4 = 360 := by
  sorry

end benny_market_money_l481_48156


namespace efficient_coin_labeling_theorem_l481_48163

/-- A coin labeling is a list of 8 positive integers representing coin values in cents -/
def CoinLabeling := List Nat

/-- Checks if a given coin labeling is n-efficient -/
def is_n_efficient (labeling : CoinLabeling) (n : Nat) : Prop :=
  (labeling.length = 8) ∧
  (∀ (k : Nat), 1 ≤ k ∧ k ≤ n → ∃ (buyer_coins seller_coins : List Nat),
    buyer_coins ⊆ labeling.take 4 ∧
    seller_coins ⊆ labeling.drop 4 ∧
    buyer_coins.sum - seller_coins.sum = k)

/-- The maximum n for which an n-efficient labeling exists -/
def max_efficient_n : Nat := 240

/-- Theorem stating the existence of a 240-efficient labeling and that it's the maximum -/
theorem efficient_coin_labeling_theorem :
  (∃ (labeling : CoinLabeling), is_n_efficient labeling max_efficient_n) ∧
  (∀ (n : Nat), n > max_efficient_n → ¬∃ (labeling : CoinLabeling), is_n_efficient labeling n) :=
sorry

end efficient_coin_labeling_theorem_l481_48163


namespace two_digit_numbers_equal_three_times_product_of_digits_l481_48168

theorem two_digit_numbers_equal_three_times_product_of_digits :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n = 3 * (n / 10) * (n % 10)} = {15, 24} := by
  sorry

end two_digit_numbers_equal_three_times_product_of_digits_l481_48168


namespace middle_integer_of_consecutive_sum_l481_48194

theorem middle_integer_of_consecutive_sum (n : ℤ) : 
  (n - 1) + n + (n + 1) = 180 → n = 60 := by
  sorry

end middle_integer_of_consecutive_sum_l481_48194


namespace people_to_lift_car_l481_48170

theorem people_to_lift_car : ℕ :=
  let people_for_car : ℕ := sorry
  let people_for_truck : ℕ := 2 * people_for_car
  have h1 : 6 * people_for_car + 3 * people_for_truck = 60 := by sorry
  have h2 : people_for_car = 5 := by sorry
  5

#check people_to_lift_car

end people_to_lift_car_l481_48170


namespace series_sum_equals_half_l481_48185

/-- The sum of the series defined by the nth term 1/((n+1)(n+2)) - 1/((n+2)(n+3)) for n ≥ 1 is equal to 1/2. -/
theorem series_sum_equals_half :
  (∑' n : ℕ, (1 : ℝ) / ((n + 1) * (n + 2)) - 1 / ((n + 2) * (n + 3))) = 1 / 2 := by
  sorry

end series_sum_equals_half_l481_48185


namespace mean_calculation_l481_48157

theorem mean_calculation (x : ℝ) :
  (28 + x + 50 + 78 + 104) / 5 = 62 →
  (48 + 62 + 98 + 124 + x) / 5 = 76.4 := by
sorry

end mean_calculation_l481_48157


namespace sum_1_to_140_mod_7_l481_48151

theorem sum_1_to_140_mod_7 : 
  (List.range 140).sum % 7 = 0 := by
  sorry

end sum_1_to_140_mod_7_l481_48151


namespace angle_CAD_measure_l481_48171

/-- Represents a point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a pentagon -/
structure Pentagon :=
  (B : Point)
  (C : Point)
  (D : Point)
  (E : Point)
  (G : Point)

/-- Checks if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Checks if a pentagon is regular -/
def is_regular_pentagon (p : Pentagon) : Prop := sorry

/-- Calculates the angle between three points in degrees -/
def angle_deg (A B C : Point) : ℝ := sorry

/-- Main theorem -/
theorem angle_CAD_measure 
  (t : Triangle) 
  (p : Pentagon) 
  (h1 : is_equilateral t)
  (h2 : is_regular_pentagon p)
  (h3 : t.B = p.B)
  (h4 : t.C = p.C) :
  angle_deg t.A p.D t.C = 24 := by sorry

end angle_CAD_measure_l481_48171


namespace no_two_digit_sum_reverse_21_l481_48184

theorem no_two_digit_sum_reverse_21 : 
  ¬ ∃ (N : ℕ), 
    10 ≤ N ∧ N < 100 ∧ 
    (N + (10 * (N % 10) + N / 10) = 21) :=
sorry

end no_two_digit_sum_reverse_21_l481_48184


namespace circle_center_l481_48123

-- Define the equation
def circle_equation (a x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0

-- Define what it means for the equation to represent a circle
def is_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), ∀ (x y : ℝ), 
    circle_equation a x y ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem circle_center : 
  ∃ (a : ℝ), is_circle a ∧ 
  ∀ (h k : ℝ), (∀ (x y : ℝ), circle_equation a x y ↔ (x - h)^2 + (y - k)^2 = 25) → 
  h = -2 ∧ k = -4 :=
sorry

end circle_center_l481_48123


namespace annular_area_l481_48108

/-- The area of an annular region formed by two concentric circles -/
theorem annular_area (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 10) :
  π * r₂^2 - π * r₁^2 = 84 * π := by
  sorry

end annular_area_l481_48108


namespace toms_marbles_pairs_l481_48139

/-- Represents the set of marbles Tom has --/
structure MarbleSet where
  unique_colors : ℕ
  yellow_count : ℕ
  orange_count : ℕ

/-- Calculates the number of distinct pairs of marbles that can be chosen --/
def distinct_pairs (ms : MarbleSet) : ℕ :=
  let yellow_pairs := if ms.yellow_count ≥ 2 then 1 else 0
  let orange_pairs := if ms.orange_count ≥ 2 then 1 else 0
  let diff_color_pairs := ms.unique_colors.choose 2
  let yellow_other_pairs := ms.unique_colors * ms.yellow_count
  let orange_other_pairs := ms.unique_colors * ms.orange_count
  yellow_pairs + orange_pairs + diff_color_pairs + yellow_other_pairs + orange_other_pairs

/-- Theorem stating that Tom's marble set results in 36 distinct pairs --/
theorem toms_marbles_pairs :
  distinct_pairs { unique_colors := 4, yellow_count := 4, orange_count := 3 } = 36 := by
  sorry

end toms_marbles_pairs_l481_48139


namespace people_per_car_l481_48182

/-- Proves that if 63 people are equally divided among 9 cars, then the number of people in each car is 7. -/
theorem people_per_car (total_people : ℕ) (num_cars : ℕ) (people_per_car : ℕ) 
  (h1 : total_people = 63) 
  (h2 : num_cars = 9) 
  (h3 : total_people = num_cars * people_per_car) : 
  people_per_car = 7 := by
  sorry

end people_per_car_l481_48182


namespace problem_statement_l481_48118

theorem problem_statement : 3^(1 + Real.log 2 / Real.log 3) + Real.log 5 + (Real.log 2 / Real.log 3) * (Real.log 3 / Real.log 2) * Real.log 2 = 7 := by
  sorry

end problem_statement_l481_48118


namespace percentage_silver_cars_l481_48181

/-- Calculates the percentage of silver cars after a new shipment -/
theorem percentage_silver_cars (initial_cars : ℕ) (initial_silver_percentage : ℚ) 
  (new_cars : ℕ) (new_non_silver_percentage : ℚ) :
  initial_cars = 40 →
  initial_silver_percentage = 1/5 →
  new_cars = 80 →
  new_non_silver_percentage = 1/2 →
  let initial_silver := initial_cars * initial_silver_percentage
  let new_silver := new_cars * (1 - new_non_silver_percentage)
  let total_silver := initial_silver + new_silver
  let total_cars := initial_cars + new_cars
  (total_silver / total_cars : ℚ) = 2/5 := by
  sorry

end percentage_silver_cars_l481_48181


namespace equation_solution_l481_48128

theorem equation_solution :
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 :=
by sorry

end equation_solution_l481_48128


namespace num_cows_is_24_l481_48153

/-- Represents the number of ducks in the group -/
def num_ducks : ℕ := sorry

/-- Represents the number of cows in the group -/
def num_cows : ℕ := sorry

/-- The total number of legs in the group -/
def total_legs : ℕ := 2 * num_ducks + 4 * num_cows

/-- The total number of heads in the group -/
def total_heads : ℕ := num_ducks + num_cows

/-- Theorem stating that the number of cows is 24 given the conditions -/
theorem num_cows_is_24 : 
  (total_legs = 2 * total_heads + 48) → num_cows = 24 := by
  sorry

end num_cows_is_24_l481_48153


namespace max_m_value_l481_48199

/-- A point in the coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Definition of a valid configuration -/
def ValidConfig (n : ℕ) (m : ℕ) (points : Fin (m + 2) → Point) : Prop :=
  (n % 2 = 1) ∧ 
  (points 0 = ⟨0, 1⟩) ∧ 
  (points (Fin.last m) = ⟨n + 1, n⟩) ∧ 
  (∀ i : Fin m, 1 ≤ (points i.succ).x ∧ (points i.succ).x ≤ n ∧ 
                1 ≤ (points i.succ).y ∧ (points i.succ).y ≤ n) ∧
  (∀ i : Fin (m + 1), i.val % 2 = 0 → (points i).y = (points i.succ).y) ∧
  (∀ i : Fin (m + 1), i.val % 2 = 1 → (points i).x = (points i.succ).x) ∧
  (∀ i j : Fin (m + 1), i < j → 
    ((points i).x = (points i.succ).x ∧ (points j).x = (points j.succ).x → 
      (points i).x ≠ (points j).x) ∨
    ((points i).y = (points i.succ).y ∧ (points j).y = (points j.succ).y → 
      (points i).y ≠ (points j).y))

/-- The main theorem -/
theorem max_m_value (n : ℕ) : 
  (n % 2 = 1) → (∃ m : ℕ, ∃ points : Fin (m + 2) → Point, ValidConfig n m points) → 
  (∀ k : ℕ, ∀ points : Fin (k + 2) → Point, ValidConfig n k points → k ≤ n * (n - 1)) :=
sorry

end max_m_value_l481_48199


namespace jafari_candy_count_l481_48149

theorem jafari_candy_count (total candy_taquon candy_mack : ℕ) 
  (h1 : total = candy_taquon + candy_mack + (total - candy_taquon - candy_mack))
  (h2 : candy_taquon = 171)
  (h3 : candy_mack = 171)
  (h4 : total = 418) :
  total - candy_taquon - candy_mack = 76 := by
sorry

end jafari_candy_count_l481_48149


namespace intersection_A_complement_B_l481_48197

open Set

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 1}

theorem intersection_A_complement_B :
  A ∩ (𝒰 \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_A_complement_B_l481_48197


namespace stratified_sample_size_l481_48146

theorem stratified_sample_size 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (sample_female : ℕ) 
  (h1 : total_male = 42) 
  (h2 : total_female = 30) 
  (h3 : sample_female = 5) :
  ∃ (sample_male : ℕ), 
    (sample_male : ℚ) / (sample_female : ℚ) = (total_male : ℚ) / (total_female : ℚ) ∧
    sample_male + sample_female = 12 :=
by sorry

end stratified_sample_size_l481_48146


namespace inequality_solution_l481_48100

theorem inequality_solution :
  ∀ x y : ℝ,
  (y^2)^2 < (x + 1)^2 ∧ (x + 1)^2 = y^4 + y^2 + 1 ∧ y^4 + y^2 + 1 ≤ (y^2 + 1)^2 →
  (x = 0 ∧ y = 0) ∨ (x = -2 ∧ y = 0) :=
by sorry

end inequality_solution_l481_48100


namespace sallys_cards_l481_48166

/-- Sally's card counting problem -/
theorem sallys_cards (initial : ℕ) (dans_gift : ℕ) (sallys_purchase : ℕ) : 
  initial = 27 → dans_gift = 41 → sallys_purchase = 20 → 
  initial + dans_gift + sallys_purchase = 88 := by
  sorry

end sallys_cards_l481_48166


namespace fence_painting_time_l481_48114

theorem fence_painting_time (taimour_time : ℝ) (h1 : taimour_time = 21) :
  let jamshid_time := taimour_time / 2
  let combined_rate := 1 / taimour_time + 1 / jamshid_time
  1 / combined_rate = 7 := by sorry

end fence_painting_time_l481_48114


namespace min_volleyballs_l481_48109

/-- The price of 3 basketballs and 2 volleyballs -/
def price_3b_2v : ℕ := 520

/-- The price of 2 basketballs and 5 volleyballs -/
def price_2b_5v : ℕ := 640

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 50

/-- The total budget in yuan -/
def total_budget : ℕ := 5500

/-- The price of a basketball in yuan -/
def basketball_price : ℕ := 120

/-- The price of a volleyball in yuan -/
def volleyball_price : ℕ := 80

theorem min_volleyballs (b v : ℕ) :
  3 * b + 2 * v = price_3b_2v →
  2 * b + 5 * v = price_2b_5v →
  b = basketball_price →
  v = volleyball_price →
  (∀ x y : ℕ, x + y = total_balls → basketball_price * x + volleyball_price * y ≤ total_budget →
    y ≥ 13) :=
by sorry

end min_volleyballs_l481_48109


namespace b_initial_investment_l481_48138

/-- Given A's investment and doubling conditions, proves B's initial investment --/
theorem b_initial_investment 
  (a_initial : ℕ) 
  (a_doubles_after_six_months : Bool) 
  (equal_yearly_investment : Bool) : ℕ :=
by
  -- Assuming a_initial = 3000, a_doubles_after_six_months = true, and equal_yearly_investment = true
  sorry

#check b_initial_investment

end b_initial_investment_l481_48138


namespace mango_juice_cost_l481_48188

/-- The cost of a big bottle of mango juice in pesetas -/
def big_bottle_cost : ℕ := 2700

/-- The volume of a big bottle in ounces -/
def big_bottle_volume : ℕ := 30

/-- The volume of a small bottle in ounces -/
def small_bottle_volume : ℕ := 6

/-- The cost of a small bottle in pesetas -/
def small_bottle_cost : ℕ := 600

/-- The amount saved by buying a big bottle instead of equivalent small bottles in pesetas -/
def saving : ℕ := 300

theorem mango_juice_cost :
  big_bottle_cost = 
    (big_bottle_volume / small_bottle_volume) * small_bottle_cost - saving :=
by sorry

end mango_juice_cost_l481_48188


namespace airline_rows_calculation_l481_48110

/-- Represents an airline company with a fleet of airplanes -/
structure AirlineCompany where
  num_airplanes : ℕ
  rows_per_airplane : ℕ
  seats_per_row : ℕ
  flights_per_day : ℕ
  total_daily_capacity : ℕ

/-- Theorem: Given the airline company's specifications, prove that each airplane has 20 rows -/
theorem airline_rows_calculation (airline : AirlineCompany)
    (h1 : airline.num_airplanes = 5)
    (h2 : airline.seats_per_row = 7)
    (h3 : airline.flights_per_day = 2)
    (h4 : airline.total_daily_capacity = 1400) :
    airline.rows_per_airplane = 20 := by
  sorry

/-- Example airline company satisfying the given conditions -/
def example_airline : AirlineCompany :=
  { num_airplanes := 5
    rows_per_airplane := 20  -- This is what we're proving
    seats_per_row := 7
    flights_per_day := 2
    total_daily_capacity := 1400 }

end airline_rows_calculation_l481_48110


namespace cars_meeting_time_l481_48125

/-- Two cars meeting on a highway -/
theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) (h1 : highway_length = 105)
    (h2 : speed1 = 15) (h3 : speed2 = 20) : 
  (highway_length / (speed1 + speed2)) = 3 := by
  sorry

end cars_meeting_time_l481_48125


namespace unintended_texts_per_week_l481_48132

theorem unintended_texts_per_week 
  (old_daily_texts : ℕ) 
  (new_daily_texts : ℕ) 
  (days_in_week : ℕ) 
  (h1 : old_daily_texts = 20)
  (h2 : new_daily_texts = 55)
  (h3 : days_in_week = 7) :
  (new_daily_texts - old_daily_texts) * days_in_week = 245 :=
by sorry

end unintended_texts_per_week_l481_48132


namespace line_perpendicular_to_plane_and_line_in_plane_l481_48113

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_and_line_in_plane
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : contained_in n α) :
  perpendicular_lines m n :=
sorry

end line_perpendicular_to_plane_and_line_in_plane_l481_48113


namespace sin_cos_shift_l481_48155

theorem sin_cos_shift (x : ℝ) : Real.cos (2 * (x - Real.pi / 8) - Real.pi / 4) = Real.sin (2 * x) := by
  sorry

end sin_cos_shift_l481_48155


namespace shirt_cost_calculation_l481_48192

/-- The amount Mary spent on clothing -/
def total_spent : ℝ := 25.31

/-- The amount Mary spent on the jacket -/
def jacket_cost : ℝ := 12.27

/-- The number of shops Mary visited -/
def shops_visited : ℕ := 2

/-- The amount Mary spent on the shirt -/
def shirt_cost : ℝ := total_spent - jacket_cost

theorem shirt_cost_calculation : 
  shirt_cost = total_spent - jacket_cost :=
by sorry

end shirt_cost_calculation_l481_48192


namespace sin_330_degrees_l481_48169

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l481_48169


namespace min_value_expression_min_value_achievable_l481_48165

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2*x + 1/(3*y)) * (2*x + 1/(3*y) - 2023) + (3*y + 1/(2*x)) * (3*y + 1/(2*x) - 2023) ≥ -2050529.5 :=
by sorry

theorem min_value_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧
  (2*x + 1/(3*y)) * (2*x + 1/(3*y) - 2023) + (3*y + 1/(2*x)) * (3*y + 1/(2*x) - 2023) = -2050529.5 :=
by sorry

end min_value_expression_min_value_achievable_l481_48165


namespace prize_probability_after_addition_l481_48180

/-- Given a box with prizes, this function calculates the probability of pulling a prize -/
def prizeProbability (favorable : ℕ) (unfavorable : ℕ) : ℚ :=
  (favorable : ℚ) / (favorable + unfavorable : ℚ)

theorem prize_probability_after_addition (initial_favorable : ℕ) (initial_unfavorable : ℕ) 
  (h_initial_odds : initial_favorable = 5 ∧ initial_unfavorable = 6) 
  (added_prizes : ℕ) (h_added_prizes : added_prizes = 2) :
  prizeProbability (initial_favorable + added_prizes) initial_unfavorable = 7 / 13 := by
  sorry

#check prize_probability_after_addition

end prize_probability_after_addition_l481_48180


namespace min_value_expression_l481_48173

theorem min_value_expression (a b c : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (5^(1/4) - 1/2)^2 := by
  sorry

end min_value_expression_l481_48173


namespace parabola_axis_of_symmetry_l481_48145

/-- The axis of symmetry for the parabola x = -4y² is x = 1/16 -/
theorem parabola_axis_of_symmetry :
  let f : ℝ → ℝ := fun y ↦ -4 * y^2
  ∃ x₀ : ℝ, x₀ = 1/16 ∧ ∀ y : ℝ, f y = f (-y) → x₀ = f y :=
by sorry

end parabola_axis_of_symmetry_l481_48145


namespace leader_secretary_selection_count_l481_48115

/-- The number of ways to select a team leader and a secretary from a team of 5 members -/
def select_leader_and_secretary (team_size : ℕ) : ℕ :=
  team_size * (team_size - 1)

/-- Theorem: The number of ways to select a team leader and a secretary from a team of 5 members is 20 -/
theorem leader_secretary_selection_count :
  select_leader_and_secretary 5 = 20 := by
  sorry

end leader_secretary_selection_count_l481_48115


namespace photo_arrangements_l481_48176

def number_of_students : ℕ := 7
def number_of_bound_students : ℕ := 2
def number_of_separated_students : ℕ := 2

def arrangements (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem photo_arrangements :
  let bound_ways := number_of_bound_students
  let remaining_elements := number_of_students - number_of_bound_students - number_of_separated_students + 1
  let gaps := remaining_elements + 1
  bound_ways * arrangements remaining_elements remaining_elements * arrangements gaps number_of_separated_students = 960 := by
  sorry

end photo_arrangements_l481_48176


namespace ball_distribution_ratio_l481_48177

/-- The number of balls -/
def n : ℕ := 30

/-- The number of bins -/
def m : ℕ := 6

/-- The probability of one bin having 6 balls, one having 3 balls, and four having 5 balls each -/
noncomputable def p' : ℝ := sorry

/-- The probability of all bins having exactly 5 balls -/
noncomputable def q' : ℝ := sorry

/-- The theorem stating that the ratio of p' to q' is 5 -/
theorem ball_distribution_ratio : p' / q' = 5 := by sorry

end ball_distribution_ratio_l481_48177


namespace polynomial_sum_l481_48135

-- Define the polynomials
def f (x : ℝ) : ℝ := -3 * x^3 - 3 * x^2 + x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 5 * x^2 + 6 * x + 2

-- State the theorem
theorem polynomial_sum (x : ℝ) : f x + g x + h x = -3 * x^3 - 4 * x^2 + 11 * x - 12 := by
  sorry

end polynomial_sum_l481_48135


namespace sum_of_squares_zero_implies_sum_l481_48175

theorem sum_of_squares_zero_implies_sum (a b c : ℝ) :
  (a - 2)^2 + (b - 6)^2 + (c - 8)^2 = 0 → a + b + c = 16 := by
  sorry

end sum_of_squares_zero_implies_sum_l481_48175


namespace tangent_sum_l481_48161

theorem tangent_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 10/9 := by
  sorry

end tangent_sum_l481_48161


namespace cubic_inequality_l481_48190

theorem cubic_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end cubic_inequality_l481_48190


namespace sector_max_area_l481_48148

/-- Given a sector of a circle with radius R, central angle α, and fixed perimeter c,
    the maximum area of the sector is c²/16. -/
theorem sector_max_area (R α c : ℝ) (h_pos_R : R > 0) (h_pos_α : α > 0) (h_pos_c : c > 0)
  (h_perimeter : c = 2 * R + R * α) :
  ∃ (A : ℝ), A ≤ c^2 / 16 ∧ 
  (∀ (R' α' : ℝ), R' > 0 → α' > 0 → c = 2 * R' + R' * α' → 
    (1/2) * R' * R' * α' ≤ A) :=
sorry

end sector_max_area_l481_48148


namespace greatest_power_of_two_factor_l481_48106

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 2^351 * k = 15^702 - 6^351) ∧ 
  (∀ m : ℕ, m > 351 → ¬(∃ k : ℕ, 2^m * k = 15^702 - 6^351)) := by
  sorry

end greatest_power_of_two_factor_l481_48106


namespace train_acceleration_equation_l481_48164

theorem train_acceleration_equation 
  (v : ℝ) (s : ℝ) (x : ℝ) 
  (h1 : v > 0) 
  (h2 : s > 0) 
  (h3 : x > v) :
  s / (x - v) = (s + 50) / x :=
by sorry

end train_acceleration_equation_l481_48164


namespace carpet_shaded_area_l481_48179

/-- Represents the carpet configuration with shaded squares -/
structure CarpetConfig where
  carpet_side : ℝ
  large_square_side : ℝ
  small_square_side : ℝ
  large_square_count : ℕ
  small_square_count : ℕ

/-- Calculates the total shaded area of the carpet -/
def total_shaded_area (config : CarpetConfig) : ℝ :=
  config.large_square_count * config.large_square_side^2 +
  config.small_square_count * config.small_square_side^2

/-- Theorem stating the total shaded area of the carpet with given conditions -/
theorem carpet_shaded_area :
  ∀ (config : CarpetConfig),
    config.carpet_side = 12 →
    config.carpet_side / config.large_square_side = 4 →
    config.large_square_side / config.small_square_side = 3 →
    config.large_square_count = 1 →
    config.small_square_count = 8 →
    total_shaded_area config = 17 := by
  sorry


end carpet_shaded_area_l481_48179


namespace chicken_wings_distribution_l481_48137

theorem chicken_wings_distribution (num_friends : ℕ) (initial_wings : ℕ) (additional_wings : ℕ) :
  num_friends = 4 →
  initial_wings = 9 →
  additional_wings = 7 →
  (initial_wings + additional_wings) % num_friends = 0 →
  (initial_wings + additional_wings) / num_friends = 4 :=
by sorry

end chicken_wings_distribution_l481_48137


namespace ryan_spanish_hours_l481_48143

/-- Ryan's daily study hours -/
structure StudyHours where
  english : ℕ
  chinese : ℕ
  spanish : ℕ

/-- Ryan's study schedule satisfies the given conditions -/
def validSchedule (h : StudyHours) : Prop :=
  h.english = 7 ∧ h.chinese = 2 ∧ h.english = h.spanish + 3

theorem ryan_spanish_hours (h : StudyHours) (hvalid : validSchedule h) : h.spanish = 4 := by
  sorry

end ryan_spanish_hours_l481_48143


namespace digits_divisible_by_3_in_base_4_of_375_l481_48186

def base_4_representation (n : ℕ) : List ℕ :=
  sorry

def count_divisible_by_3 (digits : List ℕ) : ℕ :=
  sorry

theorem digits_divisible_by_3_in_base_4_of_375 :
  count_divisible_by_3 (base_4_representation 375) = 2 :=
sorry

end digits_divisible_by_3_in_base_4_of_375_l481_48186


namespace polygon_sides_l481_48122

theorem polygon_sides (n : ℕ) (sum_angles : ℝ) : sum_angles = 900 → n = 5 := by
  sorry

end polygon_sides_l481_48122


namespace alberts_remaining_laps_l481_48196

/-- Calculates the remaining laps for Albert's run -/
theorem alberts_remaining_laps 
  (total_distance : ℕ) 
  (track_length : ℕ) 
  (laps_run : ℕ) 
  (h1 : total_distance = 99) 
  (h2 : track_length = 9) 
  (h3 : laps_run = 6) : 
  total_distance / track_length - laps_run = 5 := by
  sorry

#check alberts_remaining_laps

end alberts_remaining_laps_l481_48196


namespace particle_paths_l481_48126

theorem particle_paths (n k : ℕ) : 
  (n = 5 ∧ k = 3) → (Nat.choose n ((n + k) / 2) = 5) ∧
  (n = 20 ∧ k = 16) → (Nat.choose n ((n + k) / 2) = 190) :=
by sorry

end particle_paths_l481_48126


namespace system_solution_l481_48119

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop :=
  7 / (2 * x - 3) - 2 / (10 * z - 3 * y) + 3 / (3 * y - 8 * z) = 8

def equation2 (x y z : ℝ) : Prop :=
  2 / (2 * x - 3 * y) - 3 / (10 * z - 3 * y) + 1 / (3 * y - 8 * z) = 0

def equation3 (x y z : ℝ) : Prop :=
  5 / (2 * x - 3 * y) - 4 / (10 * z - 3 * y) + 7 / (3 * y - 8 * z) = 8

-- Define the solution
def solution : ℝ × ℝ × ℝ := (5, 3, 1)

-- Theorem statement
theorem system_solution :
  ∀ x y z : ℝ,
  2 * x ≠ 3 * y →
  10 * z ≠ 3 * y →
  8 * z ≠ 3 * y →
  equation1 x y z ∧ equation2 x y z ∧ equation3 x y z →
  (x, y, z) = solution :=
by
  sorry

end system_solution_l481_48119


namespace both_languages_difference_l481_48191

/-- The total number of students in the school -/
def total_students : ℕ := 2500

/-- The minimum percentage of students studying Italian -/
def min_italian_percent : ℚ := 70 / 100

/-- The maximum percentage of students studying Italian -/
def max_italian_percent : ℚ := 75 / 100

/-- The minimum percentage of students studying German -/
def min_german_percent : ℚ := 35 / 100

/-- The maximum percentage of students studying German -/
def max_german_percent : ℚ := 45 / 100

/-- The number of students studying Italian -/
def italian_students (n : ℕ) : Prop :=
  ⌈(min_italian_percent * total_students : ℚ)⌉ ≤ n ∧ n ≤ ⌊(max_italian_percent * total_students : ℚ)⌋

/-- The number of students studying German -/
def german_students (n : ℕ) : Prop :=
  ⌈(min_german_percent * total_students : ℚ)⌉ ≤ n ∧ n ≤ ⌊(max_german_percent * total_students : ℚ)⌋

/-- The theorem stating the difference between max and min number of students studying both languages -/
theorem both_languages_difference :
  ∃ (max min : ℕ),
    (∀ i g b, italian_students i → german_students g → i + g - b = total_students → b ≤ max) ∧
    (∀ i g b, italian_students i → german_students g → i + g - b = total_students → min ≤ b) ∧
    max - min = 375 := by
  sorry

end both_languages_difference_l481_48191


namespace distinct_reciprocals_inequality_l481_48162

theorem distinct_reciprocals_inequality (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h_sum : 2 * b = a + c) : 
  2 / b ≠ 1 / a + 1 / c := by
sorry

end distinct_reciprocals_inequality_l481_48162


namespace inspector_examination_l481_48120

/-- Given an inspector who rejects 0.02% of meters as defective and examined 10,000 meters to reject 2 meters,
    prove that to reject x meters, the inspector needs to examine 5000x meters. -/
theorem inspector_examination (x : ℝ) : 
  (2 / 10000 = x / (5000 * x)) → 5000 * x = (x * 10000) / 2 := by
sorry

end inspector_examination_l481_48120


namespace remaining_area_formula_l481_48167

/-- The remaining area of a rectangle with a hole -/
def remaining_area (x : ℝ) : ℝ :=
  (2*x + 5) * (x + 8) - (3*x - 2) * (x + 1)

/-- Theorem: The remaining area is equal to -x^2 + 20x + 42 -/
theorem remaining_area_formula (x : ℝ) :
  remaining_area x = -x^2 + 20*x + 42 := by
  sorry

end remaining_area_formula_l481_48167


namespace sequence_average_bound_l481_48174

theorem sequence_average_bound (n : ℕ) (a : ℕ → ℝ) 
  (h1 : a 1 = 0)
  (h2 : ∀ k ∈ Finset.range n, k > 1 → |a k| = |a (k-1) + 1|) :
  (Finset.sum (Finset.range n) (λ i => a (i+1))) / n ≥ -1/2 := by
  sorry

end sequence_average_bound_l481_48174


namespace apples_per_pie_l481_48158

theorem apples_per_pie 
  (initial_apples : ℕ) 
  (handed_out : ℕ) 
  (num_pies : ℕ) 
  (h1 : initial_apples = 75) 
  (h2 : handed_out = 19) 
  (h3 : num_pies = 7) :
  (initial_apples - handed_out) / num_pies = 8 := by
  sorry

end apples_per_pie_l481_48158


namespace quadratic_rational_roots_unique_b_l481_48150

theorem quadratic_rational_roots_unique_b : 
  ∃! b : ℕ+, (∃ x y : ℚ, 3 * x^2 + 6 * x + b.val = 0 ∧ 3 * y^2 + 6 * y + b.val = 0 ∧ x ≠ y) ∧ b = 3 := by
  sorry

end quadratic_rational_roots_unique_b_l481_48150


namespace mary_zoom_time_l481_48193

def total_time (mac_download : ℕ) (windows_download_factor : ℕ) 
               (audio_glitch_duration : ℕ) (audio_glitch_count : ℕ)
               (video_glitch_duration : ℕ) : ℕ :=
  let windows_download := mac_download * windows_download_factor
  let total_download := mac_download + windows_download
  let audio_glitch_time := audio_glitch_duration * audio_glitch_count
  let total_glitch_time := audio_glitch_time + video_glitch_duration
  let glitch_free_time := 2 * total_glitch_time
  total_download + total_glitch_time + glitch_free_time

theorem mary_zoom_time : 
  total_time 10 3 4 2 6 = 82 := by
  sorry

end mary_zoom_time_l481_48193


namespace ellipse_through_six_points_l481_48178

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Check if a point lies on an ellipse with center (h, k), semi-major axis a, and semi-minor axis b -/
def onEllipse (p : Point) (h k a b : ℝ) : Prop :=
  ((p.x - h) ^ 2 / a ^ 2) + ((p.y - k) ^ 2 / b ^ 2) = 1

theorem ellipse_through_six_points :
  let p1 : Point := ⟨-3, 2⟩
  let p2 : Point := ⟨0, 0⟩
  let p3 : Point := ⟨0, 4⟩
  let p4 : Point := ⟨6, 0⟩
  let p5 : Point := ⟨6, 4⟩
  let p6 : Point := ⟨-3, 0⟩
  let points := [p1, p2, p3, p4, p5, p6]
  (∀ (a b c : Point), a ∈ points → b ∈ points → c ∈ points → a ≠ b → b ≠ c → a ≠ c → ¬collinear a b c) →
  ∃ (h k a b : ℝ), 
    a = 6 ∧ 
    b = 1 ∧ 
    (∀ p ∈ points, onEllipse p h k a b) :=
by sorry

end ellipse_through_six_points_l481_48178


namespace sqrt_equation_solution_l481_48159

theorem sqrt_equation_solution (a b : ℝ) : 
  Real.sqrt (a - 5) + Real.sqrt (5 - a) = b + 3 → 
  a = 5 ∧ (Real.sqrt (a^2 - b^2) = 4 ∨ Real.sqrt (a^2 - b^2) = -4) :=
by sorry

end sqrt_equation_solution_l481_48159


namespace age_difference_l481_48133

/-- Proves that the age difference between a man and his student is 26 years -/
theorem age_difference (student_age man_age : ℕ) : 
  student_age = 24 →
  man_age + 2 = 2 * (student_age + 2) →
  man_age - student_age = 26 := by
sorry

end age_difference_l481_48133


namespace sequence1_correct_sequence2_correct_l481_48183

-- Sequence 1
def sequence1 (n : ℕ) : ℚ :=
  (-5^n + (-1)^(n-1) * 3 * 2^(n+1)) / (2 * 5^n + (-1)^(n-1) * 2^(n+1))

def sequence1_recurrence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/2 ∧ ∀ n, n ≥ 1 → a (n+1) = (a n + 3) / (2 * a n - 4)

theorem sequence1_correct :
  sequence1_recurrence sequence1 := by sorry

-- Sequence 2
def sequence2 (n : ℕ) : ℚ :=
  (6*n - 11) / (3*n - 4)

def sequence2_recurrence (a : ℕ → ℚ) : Prop :=
  a 1 = 5 ∧ ∀ n, n ≥ 1 → a (n+1) = (a n - 4) / (a n - 3)

theorem sequence2_correct :
  sequence2_recurrence sequence2 := by sorry

end sequence1_correct_sequence2_correct_l481_48183


namespace alicia_remaining_masks_l481_48142

/-- The number of sets of masks Alicia had initially -/
def initial_sets : ℕ := 90

/-- The number of sets of masks Alicia gave away -/
def given_away : ℕ := 51

/-- The number of sets of masks left in Alicia's collection -/
def remaining_sets : ℕ := initial_sets - given_away

theorem alicia_remaining_masks : remaining_sets = 39 := by
  sorry

end alicia_remaining_masks_l481_48142


namespace event_attendees_l481_48154

/-- Represents the number of men at the event -/
def num_men : ℕ := 15

/-- Represents the number of women each man danced with -/
def dances_per_man : ℕ := 4

/-- Represents the number of men each woman danced with -/
def dances_per_woman : ℕ := 3

/-- Calculates the number of women at the event -/
def num_women : ℕ := (num_men * dances_per_man) / dances_per_woman

theorem event_attendees :
  num_women = 20 := by
  sorry

end event_attendees_l481_48154


namespace innings_count_l481_48152

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  average : ℝ
  highestScore : ℕ
  scoreDifference : ℕ
  averageExcludingExtremes : ℝ

/-- Calculates the number of innings played by a batsman given their stats -/
def calculateInnings (stats : BatsmanStats) : ℕ :=
  sorry

/-- Theorem stating that for the given batsman stats, the number of innings is 104 -/
theorem innings_count (stats : BatsmanStats) 
  (h1 : stats.average = 62)
  (h2 : stats.highestScore = 225)
  (h3 : stats.scoreDifference = 150)
  (h4 : stats.averageExcludingExtremes = 58) :
  calculateInnings stats = 104 := by
  sorry

end innings_count_l481_48152


namespace truck_distance_l481_48160

theorem truck_distance (north_distance east_distance : ℝ) 
  (h1 : north_distance = 40)
  (h2 : east_distance = 30) :
  Real.sqrt (north_distance ^ 2 + east_distance ^ 2) = 50 :=
by sorry

end truck_distance_l481_48160


namespace triangle_area_triangle_area_proof_l481_48172

/-- The area of a triangle with sides 16, 30, and 34 is 240 -/
theorem triangle_area : ℝ → Prop :=
  fun a : ℝ =>
    let s1 : ℝ := 16
    let s2 : ℝ := 30
    let s3 : ℝ := 34
    (s1 * s1 + s2 * s2 = s3 * s3) →  -- Pythagorean theorem condition
    (a = (1 / 2) * s1 * s2) →        -- Area formula for right triangle
    a = 240

/-- Proof of the theorem -/
theorem triangle_area_proof : triangle_area 240 := by
  sorry

end triangle_area_triangle_area_proof_l481_48172


namespace five_million_squared_l481_48141

theorem five_million_squared (five_million : ℕ) (h : five_million = 5 * 10^6) :
  five_million^2 = 25 * 10^12 := by
  sorry

end five_million_squared_l481_48141


namespace trapezoid_shorter_base_l481_48198

/-- A trapezoid with the given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ
  longer_base_length : longer_base = 117
  midpoint_segment_length : midpoint_segment = 5
  midpoint_segment_property : midpoint_segment = (longer_base - shorter_base) / 2

/-- Theorem stating that the shorter base of the trapezoid is 107 -/
theorem trapezoid_shorter_base (t : Trapezoid) : t.shorter_base = 107 := by
  sorry

end trapezoid_shorter_base_l481_48198


namespace smallest_sum_B_plus_c_l481_48101

/-- Given that B is a digit in base 5 and c is a base greater than 6, 
    if BBB₅ = 44ₖ, then the smallest possible sum of B + c is 34. -/
theorem smallest_sum_B_plus_c : 
  ∀ (B c : ℕ), 
    0 ≤ B ∧ B ≤ 4 →  -- B is a digit in base 5
    c > 6 →  -- c is a base greater than 6
    31 * B = 4 * c + 4 →  -- BBB₅ = 44ₖ
    ∀ (B' c' : ℕ), 
      0 ≤ B' ∧ B' ≤ 4 →
      c' > 6 →
      31 * B' = 4 * c' + 4 →
      B + c ≤ B' + c' →
      B + c = 34 :=
by sorry

end smallest_sum_B_plus_c_l481_48101


namespace area_enclosed_l481_48140

noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => fun x => |x|
  | k + 1 => fun x => |f k x - (k + 1)|

theorem area_enclosed (n : ℕ) : 
  ∃ (a : ℝ), a > 0 ∧ 
  (∫ (x : ℝ) in -a..a, f n x) = (4 * n^3 + 6 * n^2 - 1 + (-1)^n) / 8 :=
sorry

end area_enclosed_l481_48140


namespace arithmetic_sqrt_of_nine_l481_48121

/-- The arithmetic square root of a non-negative real number -/
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

/-- The arithmetic square root is non-negative -/
axiom arithmetic_sqrt_nonneg (x : ℝ) : x ≥ 0 → arithmetic_sqrt x ≥ 0

/-- The arithmetic square root of 9 is 3 -/
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 := by
  sorry

end arithmetic_sqrt_of_nine_l481_48121


namespace max_m_value_l481_48102

theorem max_m_value (m : ℕ+) 
  (h : ∃ (k : ℕ), (m.val ^ 4 + 16 * m.val + 8 : ℕ) = (k * (k + 1))) : 
  m ≤ 2 := by
sorry

end max_m_value_l481_48102


namespace triangle_abc_properties_l481_48103

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (a^2 - 2 * Real.sqrt 3 * a + 2 = 0) →
  (b^2 - 2 * Real.sqrt 3 * b + 2 = 0) →
  (2 * Real.cos (A + B) = -1) →
  -- Definitions of triangle
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  -- Law of cosines
  (Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) →
  -- Conclusions
  (C = π / 3) ∧ 
  (c = Real.sqrt 6) ∧ 
  (1/2 * a * b * Real.sin C = Real.sqrt 3 / 2) :=
by sorry

end triangle_abc_properties_l481_48103


namespace unique_x_with_703_factors_l481_48129

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- x^x has exactly 703 positive factors -/
def has_703_factors (x : ℕ) : Prop :=
  num_factors (x^x) = 703

theorem unique_x_with_703_factors :
  ∃! x : ℕ, x > 0 ∧ has_703_factors x ∧ x = 18 := by sorry

end unique_x_with_703_factors_l481_48129


namespace m_range_l481_48187

def p (m : ℝ) : Prop := ∀ x : ℝ, |x| + |x - 1| > m

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(7 - 3*m))^x > (-(7 - 3*m))^y

theorem m_range : 
  (∃ m : ℝ, (p m ∧ ¬q m) ∨ (¬p m ∧ q m)) ∧ 
  (∀ m : ℝ, (p m ∧ ¬q m) ∨ (¬p m ∧ q m) → m ∈ Set.Icc 1 2) ∧
  (∀ m : ℝ, m ∈ Set.Icc 1 2 → (p m ∧ ¬q m) ∨ (¬p m ∧ q m)) :=
sorry

end m_range_l481_48187


namespace triangle_side_length_l481_48147

/-- Given a triangle ABC with area √3, angle B = 60°, and a² + c² = 3ac, prove that the length of side b is 2√2 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →   -- Area of the triangle is √3
  (B = π/3) →                                 -- Angle B is 60°
  (a^2 + c^2 = 3*a*c) →                        -- Given condition
  (b = 2 * Real.sqrt 2) :=                     -- Conclusion to prove
by sorry

end triangle_side_length_l481_48147


namespace nine_digit_increasing_integers_mod_1000_l481_48189

/-- The number of ways to select 9 items from 10 items with replacement and order matters -/
def M : ℕ := Nat.choose 18 9

/-- The theorem to prove -/
theorem nine_digit_increasing_integers_mod_1000 :
  M % 1000 = 620 := by
  sorry

end nine_digit_increasing_integers_mod_1000_l481_48189


namespace certain_number_problem_l481_48104

theorem certain_number_problem :
  ∃! x : ℝ, ((7 * (x + 10)) / 5) - 5 = 88 / 2 := by
  sorry

end certain_number_problem_l481_48104


namespace not_cheap_necessary_for_good_quality_l481_48124

-- Define the universe of items
variable (Item : Type)

-- Define the properties
variable (not_cheap : Item → Prop)
variable (good_quality : Item → Prop)

-- Define the "You get what you pay for" principle
variable (you_get_what_you_pay_for : ∀ (x : Item), good_quality x → ¬(not_cheap x) → False)

-- Theorem: "not cheap" is a necessary condition for "good quality"
theorem not_cheap_necessary_for_good_quality :
  ∀ (x : Item), good_quality x → not_cheap x :=
by
  sorry

end not_cheap_necessary_for_good_quality_l481_48124


namespace simple_interest_rate_for_doubling_in_8_years_l481_48116

/-- 
Given a sum of money that doubles itself in 8 years at simple interest,
this theorem proves that the interest rate is 12.5% per annum.
-/
theorem simple_interest_rate_for_doubling_in_8_years : 
  ∀ (P : ℝ), P > 0 → 
  ∃ (R : ℝ), 
    (P + (P * R * 8) / 100 = 2 * P) ∧ 
    R = 12.5 := by
  sorry

end simple_interest_rate_for_doubling_in_8_years_l481_48116


namespace root_in_interval_l481_48195

def f (x : ℝ) := 2*x + 3*x - 7

theorem root_in_interval :
  ∃ r ∈ Set.Ioo 1 2, f r = 0 :=
sorry

end root_in_interval_l481_48195
