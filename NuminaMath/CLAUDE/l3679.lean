import Mathlib

namespace NUMINAMATH_CALUDE_no_real_roots_l3679_367905

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 5) - Real.sqrt (x - 2) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3679_367905


namespace NUMINAMATH_CALUDE_juice_consumption_l3679_367912

theorem juice_consumption (total_juice : ℚ) (sam_fraction : ℚ) (alex_fraction : ℚ) :
  total_juice = 3/4 ∧ sam_fraction = 1/2 ∧ alex_fraction = 1/4 →
  sam_fraction * total_juice + alex_fraction * total_juice = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_juice_consumption_l3679_367912


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l3679_367943

/-- Calculate the corrected mean of a dataset with misrecorded observations -/
theorem corrected_mean_calculation (n : ℕ) (incorrect_mean : ℚ) 
  (actual_values : List ℚ) (recorded_values : List ℚ) : 
  n = 25 ∧ 
  incorrect_mean = 50 ∧ 
  actual_values = [20, 35, 70] ∧
  recorded_values = [40, 55, 80] →
  (n * incorrect_mean - (recorded_values.sum - actual_values.sum)) / n = 48 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_calculation_l3679_367943


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l3679_367913

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (3/5 : ℂ) + (2/5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (3/5 : ℂ) - (2/5 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l3679_367913


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3679_367986

theorem inequality_equivalence (x : ℝ) : 
  (-2 < (x^2 - 16*x + 15) / (x^2 - 4*x + 5) ∧ (x^2 - 16*x + 15) / (x^2 - 4*x + 5) < 2) ↔ 
  (4 - Real.sqrt 21 < x ∧ x < 4 + Real.sqrt 21) := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3679_367986


namespace NUMINAMATH_CALUDE_nth_equation_l3679_367904

theorem nth_equation (n : ℕ) :
  (n + 1 : ℚ) / ((n + 1)^2 - 1) - 1 / (n * (n + 1) * (n + 2)) = 1 / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l3679_367904


namespace NUMINAMATH_CALUDE_april_flower_sale_earnings_l3679_367902

theorem april_flower_sale_earnings 
  (rose_price : ℕ)
  (initial_roses : ℕ)
  (remaining_roses : ℕ)
  (h1 : rose_price = 7)
  (h2 : initial_roses = 9)
  (h3 : remaining_roses = 4) :
  (initial_roses - remaining_roses) * rose_price = 35 :=
by sorry

end NUMINAMATH_CALUDE_april_flower_sale_earnings_l3679_367902


namespace NUMINAMATH_CALUDE_modulus_of_complex_l3679_367992

theorem modulus_of_complex (z : ℂ) (h : z + 3*I = 3 - I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l3679_367992


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3679_367942

theorem units_digit_of_product (n : ℕ) : (4^101 * 5^204 * 9^303 * 11^404) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3679_367942


namespace NUMINAMATH_CALUDE_gcd_228_1995_l3679_367907

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l3679_367907


namespace NUMINAMATH_CALUDE_triangle_height_l3679_367951

/-- Proves that a triangle with area 46 cm² and base 10 cm has a height of 9.2 cm -/
theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 46 → base = 10 → area = (base * height) / 2 → height = 9.2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l3679_367951


namespace NUMINAMATH_CALUDE_unique_solution_system_l3679_367932

theorem unique_solution_system (x y : ℝ) :
  (x - 2*y = 1 ∧ 2*x - y = 11) ↔ (x = 7 ∧ y = 3) := by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3679_367932


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3679_367953

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 4*x + 3 < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 6*x + 8 < 0}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3679_367953


namespace NUMINAMATH_CALUDE_fibLastDigitsCyclic_l3679_367975

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Sequence of last digits of Fibonacci numbers -/
def fibLastDigits : ℕ → ℕ := λ n => lastDigit (fib n)

/-- Period of a sequence -/
def isPeriodic (f : ℕ → ℕ) (p : ℕ) : Prop :=
  ∀ n, f (n + p) = f n

/-- Theorem: The sequence of last digits of Fibonacci numbers is cyclic -/
theorem fibLastDigitsCyclic : ∃ p : ℕ, p > 0 ∧ isPeriodic fibLastDigits p :=
  sorry

end NUMINAMATH_CALUDE_fibLastDigitsCyclic_l3679_367975


namespace NUMINAMATH_CALUDE_jenga_initial_blocks_jenga_game_proof_l3679_367906

theorem jenga_initial_blocks (players : ℕ) (complete_rounds : ℕ) (blocks_removed_last_round : ℕ) (blocks_remaining : ℕ) : ℕ :=
  let blocks_removed_complete_rounds := players * complete_rounds
  let total_blocks_removed := blocks_removed_complete_rounds + blocks_removed_last_round
  let initial_blocks := total_blocks_removed + blocks_remaining
  initial_blocks

theorem jenga_game_proof :
  jenga_initial_blocks 5 5 1 28 = 54 := by
  sorry

end NUMINAMATH_CALUDE_jenga_initial_blocks_jenga_game_proof_l3679_367906


namespace NUMINAMATH_CALUDE_balloon_problem_l3679_367921

theorem balloon_problem (x : ℝ) : x + 5.0 = 12 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_balloon_problem_l3679_367921


namespace NUMINAMATH_CALUDE_original_number_l3679_367981

theorem original_number (x : ℝ) (h : 5 * x - 9 = 51) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l3679_367981


namespace NUMINAMATH_CALUDE_constant_term_equals_twenty_implies_n_equals_three_l3679_367925

/-- The constant term in the expansion of (x + 2 + 1/x)^n -/
def constant_term (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- The theorem stating that if the constant term is 20, then n = 3 -/
theorem constant_term_equals_twenty_implies_n_equals_three :
  ∃ n : ℕ, constant_term n = 20 ∧ n = 3 :=
sorry

end NUMINAMATH_CALUDE_constant_term_equals_twenty_implies_n_equals_three_l3679_367925


namespace NUMINAMATH_CALUDE_investment_rate_problem_l3679_367908

theorem investment_rate_problem (total_investment : ℝ) (first_investment : ℝ) (second_rate : ℝ) (total_interest : ℝ) :
  total_investment = 10000 →
  first_investment = 6000 →
  second_rate = 0.09 →
  total_interest = 840 →
  ∃ (r : ℝ),
    r * first_investment + second_rate * (total_investment - first_investment) = total_interest ∧
    r = 0.08 :=
by sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l3679_367908


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3679_367923

theorem simplify_fraction_product : 24 * (3 / 4) * (2 / 11) * (5 / 8) = 45 / 22 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3679_367923


namespace NUMINAMATH_CALUDE_custom_mult_solution_l3679_367946

/-- Custom multiplication operation -/
def star_mult (a b : ℝ) : ℝ := a * b + a + b

/-- Theorem stating that if 3 * x = 27 under the custom multiplication, then x = 6 -/
theorem custom_mult_solution :
  (∀ a b : ℝ, star_mult a b = a * b + a + b) →
  star_mult 3 x = 27 →
  x = 6 := by sorry

end NUMINAMATH_CALUDE_custom_mult_solution_l3679_367946


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3679_367966

theorem geometric_series_sum : 
  let a : ℚ := 2/3
  let r : ℚ := 2/3
  let n : ℕ := 12
  let series_sum : ℚ := (a * (1 - r^n)) / (1 - r)
  series_sum = 1054690/531441 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3679_367966


namespace NUMINAMATH_CALUDE_ninth_minus_eighth_square_tiles_l3679_367934

/-- The side length of the nth square in the sequence -/
def L (n : ℕ) : ℕ := 2 * n + 1

/-- The number of tiles in the nth square -/
def tiles (n : ℕ) : ℕ := (L n) ^ 2

theorem ninth_minus_eighth_square_tiles : tiles 9 - tiles 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ninth_minus_eighth_square_tiles_l3679_367934


namespace NUMINAMATH_CALUDE_charm_cost_calculation_l3679_367958

/-- The cost of a single charm used in Tim's necklace business -/
def charm_cost : ℚ := 15

/-- The number of charms used in each necklace -/
def charms_per_necklace : ℕ := 10

/-- The selling price of each necklace -/
def necklace_price : ℚ := 200

/-- The number of necklaces sold -/
def necklaces_sold : ℕ := 30

/-- The total profit from selling 30 necklaces -/
def total_profit : ℚ := 1500

theorem charm_cost_calculation :
  charm_cost * (charms_per_necklace : ℚ) * (necklaces_sold : ℚ) =
  necklace_price * (necklaces_sold : ℚ) - total_profit := by sorry

end NUMINAMATH_CALUDE_charm_cost_calculation_l3679_367958


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3679_367989

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 5 * p - 7 = 0) → 
  (3 * q^2 + 5 * q - 7 = 0) → 
  (p - 2) * (q - 2) = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3679_367989


namespace NUMINAMATH_CALUDE_parallel_line_intercepts_l3679_367941

/-- A line parallel to y = 3x - 2 passing through (5, -1) has y-intercept -16 and x-intercept 16/3 -/
theorem parallel_line_intercepts :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b y = 3 * x + (b 0 - 3 * 0)) →  -- b is parallel to y = 3x - 2
  b (-1) = 5 →  -- b passes through (5, -1)
  b 0 = -16 ∧ b (16/3) = 0 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_intercepts_l3679_367941


namespace NUMINAMATH_CALUDE_complex_number_location_l3679_367999

theorem complex_number_location :
  let z : ℂ := (2 - Complex.I) / (1 + Complex.I)
  (0 < z.re) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3679_367999


namespace NUMINAMATH_CALUDE_dino_expenses_l3679_367945

/-- Calculates Dino's monthly expenses based on his work hours, hourly rates, and remaining money --/
theorem dino_expenses (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (remaining : ℕ) : 
  hours1 = 20 → hours2 = 30 → hours3 = 5 →
  rate1 = 10 → rate2 = 20 → rate3 = 40 →
  remaining = 500 →
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3 - remaining = 500 := by
  sorry

#check dino_expenses

end NUMINAMATH_CALUDE_dino_expenses_l3679_367945


namespace NUMINAMATH_CALUDE_garden_ants_approximation_l3679_367977

/-- The number of ants in a rectangular garden --/
def number_of_ants (width_feet : ℝ) (length_feet : ℝ) (ants_per_sq_inch : ℝ) : ℝ :=
  width_feet * length_feet * (12 * 12) * ants_per_sq_inch

/-- Theorem stating that the number of ants in the garden is approximately 72 million --/
theorem garden_ants_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1000000 ∧ 
  abs (number_of_ants 200 500 5 - 72000000) < ε :=
sorry

end NUMINAMATH_CALUDE_garden_ants_approximation_l3679_367977


namespace NUMINAMATH_CALUDE_friend_team_assignment_l3679_367919

theorem friend_team_assignment (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k^n = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friend_team_assignment_l3679_367919


namespace NUMINAMATH_CALUDE_greene_nursery_roses_l3679_367901

/-- The Greene Nursery flower counting problem -/
theorem greene_nursery_roses (total_flowers yellow_carnations white_roses : ℕ) 
  (h1 : total_flowers = 6284)
  (h2 : yellow_carnations = 3025)
  (h3 : white_roses = 1768) :
  total_flowers - yellow_carnations - white_roses = 1491 := by
  sorry

end NUMINAMATH_CALUDE_greene_nursery_roses_l3679_367901


namespace NUMINAMATH_CALUDE_remaining_candy_l3679_367970

def initial_candy : Real := 520.75
def given_away : Real := 234.56

theorem remaining_candy : 
  (initial_candy / 2) - given_away = 25.815 := by sorry

end NUMINAMATH_CALUDE_remaining_candy_l3679_367970


namespace NUMINAMATH_CALUDE_meaningful_range_l3679_367911

def is_meaningful (x : ℝ) : Prop :=
  1 - x ≥ 0 ∧ 2 + x ≠ 0

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≤ 1 ∧ x ≠ -2 := by
sorry

end NUMINAMATH_CALUDE_meaningful_range_l3679_367911


namespace NUMINAMATH_CALUDE_line_y_coordinate_at_x_10_l3679_367928

/-- Given a line passing through points (4, 0) and (-2, -3),
    prove that the y-coordinate of the point on this line with x-coordinate 10 is 3. -/
theorem line_y_coordinate_at_x_10 :
  let m : ℚ := (0 - (-3)) / (4 - (-2))  -- Slope of the line
  let b : ℚ := 0 - m * 4                -- y-intercept of the line
  m * 10 + b = 3 := by sorry

end NUMINAMATH_CALUDE_line_y_coordinate_at_x_10_l3679_367928


namespace NUMINAMATH_CALUDE_honda_sales_calculation_l3679_367949

/-- Given a car dealer's sales data, calculate the number of Hondas sold. -/
theorem honda_sales_calculation (total_cars : ℕ) 
  (audi_percent toyota_percent acura_percent : ℚ) : 
  total_cars = 200 →
  audi_percent = 15/100 →
  toyota_percent = 22/100 →
  acura_percent = 28/100 →
  (total_cars : ℚ) * (1 - (audi_percent + toyota_percent + acura_percent)) = 70 :=
by sorry

end NUMINAMATH_CALUDE_honda_sales_calculation_l3679_367949


namespace NUMINAMATH_CALUDE_investment_split_l3679_367961

theorem investment_split (initial_investment : ℝ) (rate1 rate2 : ℝ) (years : ℕ) (final_amount : ℝ) :
  initial_investment = 2000 ∧
  rate1 = 0.04 ∧
  rate2 = 0.06 ∧
  years = 3 ∧
  final_amount = 2436.29 →
  ∃ (x : ℝ),
    x * (1 + rate1) ^ years + (initial_investment - x) * (1 + rate2) ^ years = final_amount ∧
    x = 820 := by
  sorry

end NUMINAMATH_CALUDE_investment_split_l3679_367961


namespace NUMINAMATH_CALUDE_cherry_pie_angle_l3679_367990

theorem cherry_pie_angle (total_students : ℕ) (chocolate_pref : ℕ) (apple_pref : ℕ) (blueberry_pref : ℕ) :
  total_students = 45 →
  chocolate_pref = 15 →
  apple_pref = 10 →
  blueberry_pref = 9 →
  let remaining := total_students - (chocolate_pref + apple_pref + blueberry_pref)
  let cherry_pref := remaining / 2
  let lemon_pref := remaining / 3
  let pecan_pref := remaining - cherry_pref - lemon_pref
  (cherry_pref : ℚ) / total_students * 360 = 40 :=
by sorry

end NUMINAMATH_CALUDE_cherry_pie_angle_l3679_367990


namespace NUMINAMATH_CALUDE_equal_distances_l3679_367967

/-- Two circles in a plane -/
structure TwoCircles where
  Γ₁ : Set (ℝ × ℝ)
  Γ₂ : Set (ℝ × ℝ)

/-- Points of intersection of two circles -/
structure IntersectionPoints (tc : TwoCircles) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h₁ : A ∈ tc.Γ₁ ∧ A ∈ tc.Γ₂
  h₂ : B ∈ tc.Γ₁ ∧ B ∈ tc.Γ₂

/-- Common tangent line to two circles -/
structure CommonTangent (tc : TwoCircles) where
  Δ : Set (ℝ × ℝ)
  C : ℝ × ℝ
  D : ℝ × ℝ
  h₁ : C ∈ tc.Γ₁ ∧ C ∈ Δ
  h₂ : D ∈ tc.Γ₂ ∧ D ∈ Δ

/-- Intersection point of lines AB and CD -/
def intersectionPoint (ip : IntersectionPoints tc) (ct : CommonTangent tc) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The distances PC and PD are equal -/
theorem equal_distances (tc : TwoCircles) (ip : IntersectionPoints tc) (ct : CommonTangent tc) :
  let P := intersectionPoint ip ct
  distance P ct.C = distance P ct.D := by sorry

end NUMINAMATH_CALUDE_equal_distances_l3679_367967


namespace NUMINAMATH_CALUDE_unique_valid_number_l3679_367955

def is_valid_number (n : Fin 10 → Nat) : Prop :=
  (∀ i : Fin 8, n i * n (i + 1) * n (i + 2) = 24) ∧
  n 4 = 2 ∧
  n 8 = 3

theorem unique_valid_number :
  ∃! n : Fin 10 → Nat, is_valid_number n ∧ 
    (∀ i : Fin 10, n i = ([4, 2, 3, 4, 2, 3, 4, 2, 3, 4] : List Nat)[i]) :=
by sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3679_367955


namespace NUMINAMATH_CALUDE_train_passenger_count_l3679_367929

def train_problem (initial_passengers : ℕ) (first_station_pickup : ℕ) (final_passengers : ℕ) : ℕ :=
  let after_first_drop := initial_passengers - (initial_passengers / 3)
  let after_first_pickup := after_first_drop + first_station_pickup
  let after_second_drop := after_first_pickup / 2
  final_passengers - after_second_drop

theorem train_passenger_count :
  train_problem 288 280 248 = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_passenger_count_l3679_367929


namespace NUMINAMATH_CALUDE_min_value_f_l3679_367965

/-- The function f(x) = (x^2 + 2) / x has a minimum value of 2√2 for x > 1 -/
theorem min_value_f (x : ℝ) (h : x > 1) : (x^2 + 2) / x ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_l3679_367965


namespace NUMINAMATH_CALUDE_road_construction_equation_l3679_367940

theorem road_construction_equation (x : ℝ) : 
  x > 0 →
  (9 : ℝ) / x - 12 / (x + 1) = (1 : ℝ) / 2 ↔
  (9 / x = 12 / (x + 1) + 1 / 2 ∧
   9 = x * (12 / (x + 1) + 1 / 2) ∧
   12 = (x + 1) * (9 / x - 1 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_road_construction_equation_l3679_367940


namespace NUMINAMATH_CALUDE_bicycle_spokes_front_wheel_l3679_367997

/-- Proves that a bicycle with 60 total spokes and twice as many spokes on the back wheel as on the front wheel has 20 spokes on the front wheel. -/
theorem bicycle_spokes_front_wheel : 
  ∀ (front back : ℕ), 
  front + back = 60 → 
  back = 2 * front → 
  front = 20 := by
sorry

end NUMINAMATH_CALUDE_bicycle_spokes_front_wheel_l3679_367997


namespace NUMINAMATH_CALUDE_two_digit_sum_l3679_367956

/-- Given two single-digit natural numbers A and B, if 6A + B2 = 77, then B = 1 -/
theorem two_digit_sum (A B : ℕ) : 
  A < 10 → B < 10 → (60 + A) + (10 * B + 2) = 77 → B = 1 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_sum_l3679_367956


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3679_367993

theorem regular_polygon_interior_angle_sum (n : ℕ) (h1 : n > 2) (h2 : 360 / n = 18) :
  (n - 2) * 180 = 3240 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3679_367993


namespace NUMINAMATH_CALUDE_floor_negative_seven_halves_l3679_367985

theorem floor_negative_seven_halves : 
  ⌊(-7 : ℚ) / 2⌋ = -4 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_halves_l3679_367985


namespace NUMINAMATH_CALUDE_orange_face_probability_l3679_367910

/-- Represents a die with a specific number of sides and orange faces. -/
structure Die where
  totalSides : ℕ
  orangeFaces : ℕ
  orangeFaces_le_totalSides : orangeFaces ≤ totalSides

/-- Calculates the probability of rolling an orange face on a given die. -/
def probabilityOrangeFace (d : Die) : ℚ :=
  d.orangeFaces / d.totalSides

/-- The specific 10-sided die with 4 orange faces. -/
def tenSidedDie : Die where
  totalSides := 10
  orangeFaces := 4
  orangeFaces_le_totalSides := by norm_num

/-- Theorem stating that the probability of rolling an orange face on the 10-sided die is 2/5. -/
theorem orange_face_probability :
  probabilityOrangeFace tenSidedDie = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_orange_face_probability_l3679_367910


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l3679_367976

theorem cubic_root_sum_product (p q r : ℝ) : 
  (6 * p^3 - 4 * p^2 + 7 * p - 3 = 0) ∧ 
  (6 * q^3 - 4 * q^2 + 7 * q - 3 = 0) ∧ 
  (6 * r^3 - 4 * r^2 + 7 * r - 3 = 0) → 
  p * q + q * r + r * p = 7/6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l3679_367976


namespace NUMINAMATH_CALUDE_linear_function_domain_range_l3679_367998

def LinearFunction (k b : ℚ) : ℝ → ℝ := fun x ↦ k * x + b

theorem linear_function_domain_range 
  (k b : ℚ) 
  (h_domain : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → LinearFunction k b x ∈ Set.Icc (-4 : ℝ) 1) 
  (h_range : Set.Icc (-4 : ℝ) 1 ⊆ Set.range (LinearFunction k b)) :
  (k = 5/6 ∧ b = -3/2) ∨ (k = -5/6 ∧ b = -3/2) :=
sorry

end NUMINAMATH_CALUDE_linear_function_domain_range_l3679_367998


namespace NUMINAMATH_CALUDE_man_crossing_bridge_l3679_367916

/-- Proves that a man walking at 6 km/hr will take 15 minutes to cross a bridge of 1500 meters in length. -/
theorem man_crossing_bridge (walking_speed : Real) (bridge_length : Real) (crossing_time : Real) : 
  walking_speed = 6 → bridge_length = 1500 → crossing_time = 15 → 
  crossing_time * (walking_speed * 1000 / 60) = bridge_length := by
  sorry

#check man_crossing_bridge

end NUMINAMATH_CALUDE_man_crossing_bridge_l3679_367916


namespace NUMINAMATH_CALUDE_chebyshev_roots_l3679_367930

def T : ℕ → (Real → Real)
  | 0 => λ x => 1
  | 1 => λ x => x
  | (n + 2) => λ x => 2 * x * T (n + 1) x + T n x

theorem chebyshev_roots (n : ℕ) (k : ℕ) (h : 1 ≤ k ∧ k ≤ n) :
  T n (Real.cos ((2 * k - 1 : ℝ) * Real.pi / (2 * n : ℝ))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_chebyshev_roots_l3679_367930


namespace NUMINAMATH_CALUDE_spaceship_age_conversion_l3679_367917

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The octal representation of the spaceship's age -/
def spaceship_age_octal : List Nat := [3, 4, 7]

theorem spaceship_age_conversion :
  octal_to_decimal spaceship_age_octal = 483 := by
  sorry

#eval octal_to_decimal spaceship_age_octal

end NUMINAMATH_CALUDE_spaceship_age_conversion_l3679_367917


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_three_l3679_367964

theorem gcd_of_powers_of_three :
  Nat.gcd (3^1007 - 1) (3^1018 - 1) = 3^11 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_three_l3679_367964


namespace NUMINAMATH_CALUDE_product_of_sums_zero_l3679_367927

theorem product_of_sums_zero (x y z w : ℝ) 
  (sum_zero : x + y + z + w = 0)
  (sum_seventh_power_zero : x^7 + y^7 + z^7 + w^7 = 0) : 
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_zero_l3679_367927


namespace NUMINAMATH_CALUDE_right_angled_triangle_isosceles_triangle_isosceles_perimeter_l3679_367987

/-- Definition of the triangle ABC with side lengths based on the quadratic equation -/
def Triangle (k : ℝ) : Prop :=
  ∃ (a b : ℝ),
    a^2 - (2*k + 3)*a + k^2 + 3*k + 2 = 0 ∧
    b^2 - (2*k + 3)*b + k^2 + 3*k + 2 = 0 ∧
    a ≠ b

/-- The length of side BC is 5 -/
def BC_length (k : ℝ) : ℝ := 5

/-- Theorem: If ABC is a right-angled triangle with BC as the hypotenuse, then k = 2 -/
theorem right_angled_triangle (k : ℝ) :
  Triangle k → (∃ (a b : ℝ), a^2 + b^2 = (BC_length k)^2) → k = 2 :=
sorry

/-- Theorem: If ABC is an isosceles triangle, then k = 3 or k = 4 -/
theorem isosceles_triangle (k : ℝ) :
  Triangle k → (∃ (a b : ℝ), (a = b ∧ a ≠ BC_length k) ∨ (a = BC_length k ∧ b ≠ BC_length k) ∨ (b = BC_length k ∧ a ≠ BC_length k)) →
  k = 3 ∨ k = 4 :=
sorry

/-- Theorem: If ABC is an isosceles triangle, then its perimeter is 14 or 16 -/
theorem isosceles_perimeter (k : ℝ) :
  Triangle k → (∃ (a b : ℝ), (a = b ∧ a ≠ BC_length k) ∨ (a = BC_length k ∧ b ≠ BC_length k) ∨ (b = BC_length k ∧ a ≠ BC_length k)) →
  (∃ (p : ℝ), p = a + b + BC_length k ∧ (p = 14 ∨ p = 16)) :=
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_isosceles_triangle_isosceles_perimeter_l3679_367987


namespace NUMINAMATH_CALUDE_two_x_times_x_squared_l3679_367935

theorem two_x_times_x_squared (x : ℝ) : 2 * x * x^2 = 2 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_two_x_times_x_squared_l3679_367935


namespace NUMINAMATH_CALUDE_consecutive_divisibility_l3679_367982

theorem consecutive_divisibility (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  ∀ (start : ℕ), ∃ (x y z : ℕ), 
    (x ∈ Finset.range (2 * c) ∧ y ∈ Finset.range (2 * c) ∧ z ∈ Finset.range (2 * c)) ∧
    (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
    ((a * b * c) ∣ (x * y * z)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_divisibility_l3679_367982


namespace NUMINAMATH_CALUDE_stuffed_animals_gcd_l3679_367973

theorem stuffed_animals_gcd : Nat.gcd 26 (Nat.gcd 14 (Nat.gcd 18 22)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_gcd_l3679_367973


namespace NUMINAMATH_CALUDE_amicable_pairs_l3679_367962

/-- Sum of proper divisors of a natural number -/
def sumProperDivisors (n : ℕ) : ℕ := sorry

/-- Two numbers are amicable if the sum of proper divisors of each equals the other -/
def areAmicable (a b : ℕ) : Prop :=
  sumProperDivisors a = b ∧ sumProperDivisors b = a

theorem amicable_pairs :
  (areAmicable 284 220) ∧ (areAmicable 76084 63020) := by sorry

end NUMINAMATH_CALUDE_amicable_pairs_l3679_367962


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l3679_367947

/-- Represents a rectangular floor pattern with black and white tiles -/
structure FloorPattern where
  width : ℕ
  height : ℕ
  blackTiles : ℕ
  whiteTiles : ℕ

/-- Adds a border of white tiles to a floor pattern -/
def addWhiteBorder (pattern : FloorPattern) : FloorPattern :=
  { width := pattern.width + 2
  , height := pattern.height + 2
  , blackTiles := pattern.blackTiles
  , whiteTiles := pattern.whiteTiles + (pattern.width + 2) * (pattern.height + 2) - (pattern.width * pattern.height)
  }

/-- Calculates the ratio of black tiles to white tiles -/
def tileRatio (pattern : FloorPattern) : ℚ :=
  pattern.blackTiles / pattern.whiteTiles

theorem extended_pattern_ratio :
  let initialPattern : FloorPattern :=
    { width := 5
    , height := 7
    , blackTiles := 14
    , whiteTiles := 21
    }
  let extendedPattern := addWhiteBorder initialPattern
  tileRatio extendedPattern = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l3679_367947


namespace NUMINAMATH_CALUDE_special_pentagon_angles_l3679_367963

/-- A pentagon that is a cross-section of a parallelepiped with side ratio constraints -/
structure SpecialPentagon where
  -- The pentagon is a cross-section of a parallelepiped
  is_cross_section : Bool
  -- The ratio of any two sides is either 1, 2, or 1/2
  side_ratio_constraint : ∀ (s1 s2 : ℝ), s1 > 0 → s2 > 0 → s1 / s2 ∈ ({1, 2, 1/2} : Set ℝ)

/-- The interior angles of the special pentagon -/
def interior_angles (p : SpecialPentagon) : List ℝ := sorry

/-- Theorem stating the interior angles of the special pentagon -/
theorem special_pentagon_angles (p : SpecialPentagon) :
  ∃ (angles : List ℝ), angles = interior_angles p ∧ angles.length = 5 ∧
  (angles.count 120 = 4 ∧ angles.count 60 = 1) := by sorry

end NUMINAMATH_CALUDE_special_pentagon_angles_l3679_367963


namespace NUMINAMATH_CALUDE_roots_on_circle_l3679_367984

theorem roots_on_circle : ∃ (r : ℝ), r = 2/3 ∧
  ∀ (z : ℂ), (z - 2)^6 = 64*z^6 → Complex.abs (z - 2/3) = r := by
  sorry

end NUMINAMATH_CALUDE_roots_on_circle_l3679_367984


namespace NUMINAMATH_CALUDE_algebraic_division_l3679_367957

theorem algebraic_division (m : ℝ) : -20 * m^6 / (5 * m^2) = -4 * m^4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_division_l3679_367957


namespace NUMINAMATH_CALUDE_nail_positions_symmetry_l3679_367920

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the shape of the flag -/
structure FlagShape where
  width : ℝ
  height : ℝ
  -- Additional parameters could be added to describe the specific shape

/-- Predicate to check if a nail position allows the flag to cover the hole -/
def covers (hole : Point) (nail : Point) (flag : FlagShape) : Prop :=
  -- This would involve checking if the hole is within the bounds of the flag
  -- when placed at the nail position
  sorry

/-- The set of all valid nail positions for a given hole and flag shape -/
def validNailPositions (hole : Point) (flag : FlagShape) : Set Point :=
  {nail : Point | covers hole nail flag}

theorem nail_positions_symmetry (hole : Point) (flag : FlagShape) :
  ∃ (center : Point), ∀ (nail : Point),
    nail ∈ validNailPositions hole flag →
    ∃ (symmetricNail : Point),
      symmetricNail ∈ validNailPositions hole flag ∧
      center.x = (nail.x + symmetricNail.x) / 2 ∧
      center.y = (nail.y + symmetricNail.y) / 2 :=
  sorry

end NUMINAMATH_CALUDE_nail_positions_symmetry_l3679_367920


namespace NUMINAMATH_CALUDE_sqrt_difference_complex_expression_system_of_equations_l3679_367971

-- Problem 1
theorem sqrt_difference : Real.sqrt 8 - Real.sqrt 50 = -3 * Real.sqrt 2 := by sorry

-- Problem 2
theorem complex_expression : 
  Real.sqrt 27 * Real.sqrt (1/3) - (Real.sqrt 3 - Real.sqrt 2)^2 = 2 * Real.sqrt 6 - 2 := by sorry

-- Problem 3
theorem system_of_equations :
  ∃ (x y : ℝ), x + y = 2 ∧ x + 2*y = 6 ∧ x = -2 ∧ y = 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_difference_complex_expression_system_of_equations_l3679_367971


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_l3679_367924

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (subset : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel m n → 
  subset n β → 
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_l3679_367924


namespace NUMINAMATH_CALUDE_approximate_root_exists_l3679_367950

def f (x : ℝ) := x^3 + x^2 - 2*x - 2

theorem approximate_root_exists :
  ∃ (r : ℝ), r ∈ Set.Icc 1.375 1.4375 ∧ f r = 0 ∧ 
  ∀ (x : ℝ), x ∈ Set.Icc 1.375 1.4375 → |x - 1.42| ≤ 0.05 := by
  sorry

#check approximate_root_exists

end NUMINAMATH_CALUDE_approximate_root_exists_l3679_367950


namespace NUMINAMATH_CALUDE_prob_different_colors_example_l3679_367948

/-- A box containing colored balls. -/
structure Box where
  white : ℕ
  black : ℕ

/-- The probability of drawing two balls of different colors with replacement. -/
def prob_different_colors (b : Box) : ℚ :=
  (b.white * b.black + b.black * b.white) / ((b.white + b.black) * (b.white + b.black))

/-- Theorem: The probability of drawing two balls of different colors from a box 
    containing 2 white balls and 3 black balls, with replacement, is 12/25. -/
theorem prob_different_colors_example : 
  prob_different_colors ⟨2, 3⟩ = 12 / 25 := by
  sorry

#eval prob_different_colors ⟨2, 3⟩

end NUMINAMATH_CALUDE_prob_different_colors_example_l3679_367948


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a10_l3679_367937

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a10 (a : ℕ → ℤ) :
  arithmetic_sequence a → a 7 = 4 → a 8 = 1 → a 10 = -5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a10_l3679_367937


namespace NUMINAMATH_CALUDE_surfers_count_l3679_367980

/-- The number of surfers on Santa Monica beach -/
def santa_monica_surfers : ℕ := 20

/-- The number of surfers on Malibu beach -/
def malibu_surfers : ℕ := 2 * santa_monica_surfers

/-- The total number of surfers on both beaches -/
def total_surfers : ℕ := malibu_surfers + santa_monica_surfers

theorem surfers_count : total_surfers = 60 := by
  sorry

end NUMINAMATH_CALUDE_surfers_count_l3679_367980


namespace NUMINAMATH_CALUDE_perfume_price_problem_l3679_367933

/-- Proves that given the conditions of the perfume price changes, the original price must be $1200 -/
theorem perfume_price_problem (P : ℝ) : 
  (P * 1.10 * 0.85 = P - 78) → P = 1200 := by
  sorry

end NUMINAMATH_CALUDE_perfume_price_problem_l3679_367933


namespace NUMINAMATH_CALUDE_good_set_properties_l3679_367936

def GoodSet (S : Set ℝ) : Prop :=
  ∀ a : ℝ, a ∈ S → (8 - a) ∈ S

theorem good_set_properties :
  (¬ GoodSet {1, 2}) ∧
  GoodSet {1, 4, 7} ∧
  GoodSet {4} ∧
  GoodSet {3, 4, 5} ∧
  GoodSet {2, 6} ∧
  GoodSet {1, 2, 4, 6, 7} ∧
  GoodSet {0, 8} :=
by sorry

end NUMINAMATH_CALUDE_good_set_properties_l3679_367936


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_A_l3679_367988

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Define the universal set R (real numbers)
def R : Set ℝ := univ

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 6} := by sorry

-- Theorem for the union of complement of B and A
theorem union_complement_B_A : (R \ B) ∪ A = {x | x < 6 ∨ 9 ≤ x} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_A_l3679_367988


namespace NUMINAMATH_CALUDE_kayak_production_sum_l3679_367903

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem kayak_production_sum :
  let a := 6  -- First term (February production)
  let r := 3  -- Common ratio
  let n := 5  -- Number of months (February to June)
  geometric_sum a r n = 726 := by
sorry

end NUMINAMATH_CALUDE_kayak_production_sum_l3679_367903


namespace NUMINAMATH_CALUDE_front_view_of_given_stack_map_l3679_367979

/-- Represents a stack map with four columns --/
structure StackMap :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)
  (D : ℕ)

/-- Represents the front view of a stack map --/
def FrontView := List ℕ

/-- Computes the front view of a given stack map --/
def computeFrontView (sm : StackMap) : FrontView :=
  [sm.A, sm.B, sm.C, sm.D]

/-- Theorem stating that the front view of the given stack map is [3, 5, 2, 4] --/
theorem front_view_of_given_stack_map :
  let sm : StackMap := { A := 3, B := 5, C := 2, D := 4 }
  computeFrontView sm = [3, 5, 2, 4] := by sorry

end NUMINAMATH_CALUDE_front_view_of_given_stack_map_l3679_367979


namespace NUMINAMATH_CALUDE_negative_cube_equality_l3679_367918

theorem negative_cube_equality : (-3)^3 = -3^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_equality_l3679_367918


namespace NUMINAMATH_CALUDE_angle_difference_range_l3679_367996

theorem angle_difference_range (α β : ℝ) 
  (h1 : -π < α ∧ α < β ∧ β < π) : 
  -2*π < α - β ∧ α - β < 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_difference_range_l3679_367996


namespace NUMINAMATH_CALUDE_b_finishes_in_two_days_l3679_367991

/-- The number of days A takes to finish the work alone -/
def a_days : ℚ := 4

/-- The number of days B takes to finish the work alone -/
def b_days : ℚ := 8

/-- The number of days A and B work together before A leaves -/
def days_together : ℚ := 2

/-- The fraction of work completed per day when A and B work together -/
def combined_work_rate : ℚ := 1 / a_days + 1 / b_days

/-- The fraction of work completed when A and B work together for 2 days -/
def work_completed_together : ℚ := days_together * combined_work_rate

/-- The fraction of work remaining after A leaves -/
def remaining_work : ℚ := 1 - work_completed_together

/-- The number of days B takes to finish the remaining work alone -/
def days_for_b_to_finish : ℚ := remaining_work / (1 / b_days)

theorem b_finishes_in_two_days : days_for_b_to_finish = 2 := by
  sorry

end NUMINAMATH_CALUDE_b_finishes_in_two_days_l3679_367991


namespace NUMINAMATH_CALUDE_arithmetic_computation_l3679_367974

theorem arithmetic_computation : -7 * 5 - (-4 * -2) + (-9 * -6) = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l3679_367974


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3679_367939

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 28 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3679_367939


namespace NUMINAMATH_CALUDE_correct_match_probability_l3679_367972

theorem correct_match_probability (n : Nat) (h : n = 4) :
  (1 : ℚ) / n.factorial = (1 : ℚ) / 24 := by
  sorry

#check correct_match_probability

end NUMINAMATH_CALUDE_correct_match_probability_l3679_367972


namespace NUMINAMATH_CALUDE_factorial_80_mod_7_l3679_367983

def last_three_nonzero_digits (n : ℕ) : ℕ := sorry

theorem factorial_80_mod_7 : 
  last_three_nonzero_digits (Nat.factorial 80) % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_factorial_80_mod_7_l3679_367983


namespace NUMINAMATH_CALUDE_solve_for_x_l3679_367915

theorem solve_for_x (x y : ℝ) (h1 : x + 2*y = 100) (h2 : y = 25) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3679_367915


namespace NUMINAMATH_CALUDE_intersection_distance_l3679_367944

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- Define the horizontal line
def g (x : ℝ) : ℝ := 2

-- Define the intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Theorem statement
theorem intersection_distance :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ intersection_points ∧ x₂ ∈ intersection_points ∧ x₁ ≠ x₂ ∧
  |x₁ - x₂| = 2 * Real.sqrt 22 / 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l3679_367944


namespace NUMINAMATH_CALUDE_marcus_baseball_cards_l3679_367938

theorem marcus_baseball_cards 
  (initial_cards : ℝ) 
  (additional_cards : ℝ) 
  (h1 : initial_cards = 210.0) 
  (h2 : additional_cards = 58.0) : 
  initial_cards + additional_cards = 268.0 := by
  sorry

end NUMINAMATH_CALUDE_marcus_baseball_cards_l3679_367938


namespace NUMINAMATH_CALUDE_amanda_loan_l3679_367968

/-- Calculates the earnings for a given number of hours based on a cyclic payment structure -/
def calculateEarnings (hours : ℕ) : ℕ :=
  let fullCycles := hours / 4
  let remainingHours := hours % 4
  let earningsPerCycle := 10
  let earningsFromFullCycles := fullCycles * earningsPerCycle
  let earningsFromRemainingHours := 
    if remainingHours = 1 then 2
    else if remainingHours = 2 then 5
    else if remainingHours = 3 then 7
    else 0
  earningsFromFullCycles + earningsFromRemainingHours

theorem amanda_loan (x : ℕ) : 
  (x = calculateEarnings 50) → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_amanda_loan_l3679_367968


namespace NUMINAMATH_CALUDE_most_likely_outcome_l3679_367969

/-- The probability of a child being a girl -/
def p_girl : ℚ := 3/5

/-- The probability of a child being a boy -/
def p_boy : ℚ := 2/5

/-- The number of children born -/
def n : ℕ := 3

/-- The probability of having 2 girls and 1 boy out of 3 children -/
def p_two_girls_one_boy : ℚ := 54/125

theorem most_likely_outcome :
  p_two_girls_one_boy = Nat.choose n 2 * p_girl^2 * p_boy ∧
  p_two_girls_one_boy > p_boy^n ∧
  p_two_girls_one_boy > p_girl^n ∧
  p_two_girls_one_boy > Nat.choose n 1 * p_girl * p_boy^2 :=
by sorry

end NUMINAMATH_CALUDE_most_likely_outcome_l3679_367969


namespace NUMINAMATH_CALUDE_pie_remainder_l3679_367926

theorem pie_remainder (carlos_share maria_share remainder : ℝ) : 
  carlos_share = 65 ∧ 
  maria_share = (100 - carlos_share) / 2 ∧ 
  remainder = 100 - carlos_share - maria_share →
  remainder = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_pie_remainder_l3679_367926


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3679_367954

/-- The equation of the line tangent to a circle at two points, which also passes through a given point -/
theorem tangent_line_equation (x y : ℝ → ℝ) :
  -- Given circle equation
  (∀ t, x t ^ 2 + (y t - 2) ^ 2 = 4) →
  -- Circle passes through (-2, 6)
  (∃ t, x t = -2 ∧ y t = 6) →
  -- Line equation
  (∃ a b c : ℝ, ∀ t, a * x t + b * y t + c = 0) →
  -- The line equation is x - 2y + 6 = 0
  (∃ t₁ t₂, t₁ ≠ t₂ ∧ x t₁ - 2 * y t₁ + 6 = 0 ∧ x t₂ - 2 * y t₂ + 6 = 0) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3679_367954


namespace NUMINAMATH_CALUDE_softball_team_ratio_l3679_367994

/-- Proves that for a team with 4 more women than men and 20 total players, the ratio of men to women is 2:3 -/
theorem softball_team_ratio : 
  ∀ (men women : ℕ), 
  women = men + 4 →
  men + women = 20 →
  (men : ℚ) / women = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l3679_367994


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3679_367931

theorem simple_interest_rate_calculation (P A T : ℝ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 10) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = 2 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3679_367931


namespace NUMINAMATH_CALUDE_pencil_buyers_difference_l3679_367900

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The total amount spent by eighth graders in cents -/
def eighth_grade_total : ℕ := 162

/-- The total amount spent by fifth graders in cents -/
def fifth_grade_total : ℕ := 216

/-- The cost of each pencil in cents -/
def pencil_cost : ℕ := 18

theorem pencil_buyers_difference : 
  ∃ (eighth_buyers fifth_buyers : ℕ),
    eighth_grade_total = eighth_buyers * pencil_cost ∧
    fifth_grade_total = fifth_buyers * pencil_cost ∧
    fifth_buyers - eighth_buyers = 3 :=
by sorry

end NUMINAMATH_CALUDE_pencil_buyers_difference_l3679_367900


namespace NUMINAMATH_CALUDE_exists_self_intersecting_net_l3679_367952

/-- A tetrahedron is represented by its four vertices in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- A net of a tetrahedron is represented by the 2D coordinates of its vertices -/
structure TetrahedronNet where
  vertices : Fin 4 → ℝ × ℝ

/-- A function that determines if a tetrahedron net self-intersects -/
def self_intersects (net : TetrahedronNet) : Prop :=
  sorry

/-- A function that cuts a tetrahedron along three edges not belonging to the same face -/
def cut_tetrahedron (t : Tetrahedron) : TetrahedronNet :=
  sorry

/-- The main theorem: there exists a tetrahedron whose net self-intersects -/
theorem exists_self_intersecting_net :
  ∃ t : Tetrahedron, self_intersects (cut_tetrahedron t) :=
sorry

end NUMINAMATH_CALUDE_exists_self_intersecting_net_l3679_367952


namespace NUMINAMATH_CALUDE_quadratic_constant_term_l3679_367959

theorem quadratic_constant_term (m : ℝ) : 
  (∀ x, m * x^2 + 2 * x + m^2 - 1 = 0) → m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_constant_term_l3679_367959


namespace NUMINAMATH_CALUDE_carol_has_62_pennies_l3679_367914

/-- The number of pennies Alex currently has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Carol currently has -/
def carol_pennies : ℕ := sorry

/-- If Alex gives Carol two pennies, Carol will have four times as many pennies as Alex has -/
axiom condition1 : carol_pennies + 2 = 4 * (alex_pennies - 2)

/-- If Carol gives Alex two pennies, Carol will have three times as many pennies as Alex has -/
axiom condition2 : carol_pennies - 2 = 3 * (alex_pennies + 2)

/-- Carol has 62 pennies -/
theorem carol_has_62_pennies : carol_pennies = 62 := by sorry

end NUMINAMATH_CALUDE_carol_has_62_pennies_l3679_367914


namespace NUMINAMATH_CALUDE_prob_B_draws_1_given_A_wins_l3679_367978

/-- Represents the possible outcomes of drawing a ball -/
inductive Ball : Type
| zero : Ball
| one : Ball
| two : Ball

/-- The probability of drawing each ball -/
def prob_draw (b : Ball) : ℚ :=
  match b with
  | Ball.zero => 1/4
  | Ball.one => 1/4
  | Ball.two => 1/2

/-- Player A wins if their ball is greater than Player B's ball -/
def A_wins (a b : Ball) : Prop :=
  match a, b with
  | Ball.two, Ball.zero => True
  | Ball.two, Ball.one => True
  | Ball.one, Ball.zero => True
  | _, _ => False

/-- The probability that Player A wins -/
def prob_A_wins : ℚ := 5/16

/-- The probability that Player B draws ball 1 and Player A wins -/
def prob_B_draws_1_A_wins : ℚ := 1/8

/-- The main theorem: probability that Player B draws ball 1 given Player A wins -/
theorem prob_B_draws_1_given_A_wins :
  prob_B_draws_1_A_wins / prob_A_wins = 2/5 := by sorry

end NUMINAMATH_CALUDE_prob_B_draws_1_given_A_wins_l3679_367978


namespace NUMINAMATH_CALUDE_rhombus_area_l3679_367960

/-- The area of a rhombus with diagonals of 15 cm and 21 cm is 157.5 cm². -/
theorem rhombus_area (d1 d2 area : ℝ) (h1 : d1 = 15) (h2 : d2 = 21) 
    (h3 : area = (d1 * d2) / 2) : area = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3679_367960


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l3679_367909

/-- A coloring of integers from 1 to 2014 using four colors -/
def Coloring := Fin 2014 → Fin 4

/-- An arithmetic progression of length 11 within the range 1 to 2014 -/
structure ArithmeticProgression :=
  (start : Fin 2014)
  (step : Nat)
  (h : ∀ i : Fin 11, (start.val : ℕ) + i.val * step ≤ 2014)

/-- A coloring is valid if no arithmetic progression of length 11 is monochromatic -/
def ValidColoring (c : Coloring) : Prop :=
  ∀ ap : ArithmeticProgression, ∃ i j : Fin 11, i ≠ j ∧ 
    c ⟨(ap.start.val + i.val * ap.step : ℕ), by sorry⟩ ≠ 
    c ⟨(ap.start.val + j.val * ap.step : ℕ), by sorry⟩

/-- There exists a valid coloring of integers from 1 to 2014 using four colors -/
theorem exists_valid_coloring : ∃ c : Coloring, ValidColoring c := by sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l3679_367909


namespace NUMINAMATH_CALUDE_equation_relation_l3679_367995

theorem equation_relation (x y z w : ℝ) :
  (x + 2 * y) / (2 * y + 3 * z) = (3 * z + 4 * w) / (4 * w + x) →
  x = 3 * z ∨ x + 2 * y + 4 * w + 3 * z = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_relation_l3679_367995


namespace NUMINAMATH_CALUDE_problem_statement_l3679_367922

theorem problem_statement : (-24 : ℚ) * (5/6 - 4/3 + 5/8) = -3 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3679_367922
