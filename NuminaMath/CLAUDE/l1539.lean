import Mathlib

namespace NUMINAMATH_CALUDE_polar_to_rectangular_coordinates_l1539_153923

theorem polar_to_rectangular_coordinates (r : ℝ) (θ : ℝ) :
  r = 2 ∧ θ = π / 6 →
  ∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ x = Real.sqrt 3 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_coordinates_l1539_153923


namespace NUMINAMATH_CALUDE_point_above_line_l1539_153977

/-- A point (x, y) is above a line ax + by + c = 0 if by < -ax - c -/
def IsAboveLine (x y a b c : ℝ) : Prop := b * y < -a * x - c

/-- The range of t for which (-2, t) is above the line 2x - 3y + 6 = 0 -/
theorem point_above_line (t : ℝ) : 
  IsAboveLine (-2) t 2 (-3) 6 → t > 2/3 := by
  sorry

end NUMINAMATH_CALUDE_point_above_line_l1539_153977


namespace NUMINAMATH_CALUDE_system_solution_l1539_153922

theorem system_solution :
  ∃ (x y : ℝ), x + y = 1 ∧ 4 * x + y = 10 ∧ x = 3 ∧ y = -2 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1539_153922


namespace NUMINAMATH_CALUDE_completing_square_transform_l1539_153967

theorem completing_square_transform (x : ℝ) :
  x^2 - 2*x - 5 = 0 ↔ (x - 1)^2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transform_l1539_153967


namespace NUMINAMATH_CALUDE_coefficient_x4_equals_240_l1539_153909

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the coefficient of x^4 in (1+2x)^6
def coefficient_x4 : ℕ := binomial 6 4 * 2^4

-- Theorem statement
theorem coefficient_x4_equals_240 : coefficient_x4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_equals_240_l1539_153909


namespace NUMINAMATH_CALUDE_shower_frequency_l1539_153944

/-- Represents the duration of each shower in minutes -/
def shower_duration : ℝ := 10

/-- Represents the water usage rate in gallons per minute -/
def water_usage_rate : ℝ := 2

/-- Represents the total water usage in 4 weeks in gallons -/
def total_water_usage : ℝ := 280

/-- Represents the number of weeks -/
def num_weeks : ℝ := 4

/-- Theorem stating the frequency of John's showers -/
theorem shower_frequency :
  (total_water_usage / (shower_duration * water_usage_rate)) / num_weeks = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_shower_frequency_l1539_153944


namespace NUMINAMATH_CALUDE_marbles_left_l1539_153935

/-- The number of red marbles -/
def red_marbles : ℕ := 156

/-- The number of blue marbles -/
def blue_marbles : ℕ := 267

/-- The number of marbles that fell off and broke -/
def broken_marbles : ℕ := 115

/-- The total number of marbles initially -/
def total_marbles : ℕ := red_marbles + blue_marbles

/-- Theorem: The number of marbles left is 308 -/
theorem marbles_left : total_marbles - broken_marbles = 308 := by
  sorry

end NUMINAMATH_CALUDE_marbles_left_l1539_153935


namespace NUMINAMATH_CALUDE_cherry_pie_count_l1539_153938

theorem cherry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h1 : total_pies = 24)
  (h2 : apple_ratio = 1)
  (h3 : blueberry_ratio = 4)
  (h4 : cherry_ratio = 3) : 
  (total_pies * cherry_ratio) / (apple_ratio + blueberry_ratio + cherry_ratio) = 9 := by
sorry

end NUMINAMATH_CALUDE_cherry_pie_count_l1539_153938


namespace NUMINAMATH_CALUDE_common_terms_arithmetic_progression_l1539_153918

/-- Definition of the first arithmetic progression -/
def a (n : ℕ) : ℤ := 4*n - 3

/-- Definition of the second arithmetic progression -/
def b (n : ℕ) : ℤ := 3*n - 1

/-- Function to generate the sequence of common terms -/
def common_terms (m : ℕ) : ℤ := 12*m + 5

/-- Theorem stating that the sequence of common terms forms an arithmetic progression with common difference 12 -/
theorem common_terms_arithmetic_progression :
  ∀ m : ℕ, ∃ n k : ℕ, 
    a n = b k ∧ 
    a n = common_terms m ∧ 
    common_terms (m + 1) - common_terms m = 12 :=
sorry

end NUMINAMATH_CALUDE_common_terms_arithmetic_progression_l1539_153918


namespace NUMINAMATH_CALUDE_total_distance_apart_l1539_153920

/-- Represents the speeds of a skater for three hours -/
structure SkaterSpeeds where
  hour1 : ℝ
  hour2 : ℝ
  hour3 : ℝ

/-- Calculates the total distance traveled by a skater given their speeds -/
def totalDistance (speeds : SkaterSpeeds) : ℝ :=
  speeds.hour1 + speeds.hour2 + speeds.hour3

/-- Ann's skating speeds for each hour -/
def annSpeeds : SkaterSpeeds :=
  { hour1 := 6, hour2 := 8, hour3 := 4 }

/-- Glenda's skating speeds for each hour -/
def glendaSpeeds : SkaterSpeeds :=
  { hour1 := 8, hour2 := 5, hour3 := 9 }

/-- Theorem stating the total distance between Ann and Glenda after three hours -/
theorem total_distance_apart : totalDistance annSpeeds + totalDistance glendaSpeeds = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_apart_l1539_153920


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l1539_153968

theorem simultaneous_equations_solution (m : ℝ) :
  (m ≠ 3) ↔ (∃ x y : ℝ, y = 3 * m * x + 6 ∧ y = (4 * m - 3) * x + 9) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l1539_153968


namespace NUMINAMATH_CALUDE_square_division_perimeters_l1539_153974

theorem square_division_perimeters (p : ℚ) : 
  (∃ a b c d e f : ℚ, 
    a + b + c = 1 ∧ 
    d + e + f = 1 ∧ 
    2 * (a + d) = p ∧ 
    2 * (b + e) = p ∧ 
    2 * (c + f) = p) → 
  (p = 8/3 ∨ p = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_square_division_perimeters_l1539_153974


namespace NUMINAMATH_CALUDE_quadrilateral_angle_sum_l1539_153970

theorem quadrilateral_angle_sum (a b c d : ℕ) : 
  50 ≤ a ∧ a ≤ 200 ∧
  50 ≤ b ∧ b ≤ 200 ∧
  50 ≤ c ∧ c ≤ 200 ∧
  50 ≤ d ∧ d ≤ 200 ∧
  b = 75 ∧ c = 80 ∧ d = 120 ∧
  a + b + c + d = 360 →
  a = 85 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_sum_l1539_153970


namespace NUMINAMATH_CALUDE_marble_probability_l1539_153973

theorem marble_probability (b : ℕ) : 
  2 * (2 / (2 + b)) * (1 / (1 + b)) = 1/3 → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l1539_153973


namespace NUMINAMATH_CALUDE_inverse_in_S_l1539_153930

-- Define the set S
variable (S : Set ℝ)

-- Define the properties of S
variable (h1 : Set.Subset (Set.range (Int.cast : ℤ → ℝ)) S)
variable (h2 : (Real.sqrt 2 + Real.sqrt 3) ∈ S)
variable (h3 : ∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S)
variable (h4 : ∀ x y, x ∈ S → y ∈ S → (x * y) ∈ S)

-- Theorem statement
theorem inverse_in_S : (Real.sqrt 2 + Real.sqrt 3)⁻¹ ∈ S := by
  sorry

end NUMINAMATH_CALUDE_inverse_in_S_l1539_153930


namespace NUMINAMATH_CALUDE_geometric_series_relation_l1539_153926

/-- Given two infinite geometric series with specified terms, prove that if the sum of the second series
    is three times the sum of the first series, then n = 4. -/
theorem geometric_series_relation (n : ℝ) : 
  let first_series_term1 : ℝ := 15
  let first_series_term2 : ℝ := 9
  let second_series_term1 : ℝ := 15
  let second_series_term2 : ℝ := 9 + n
  let first_series_sum : ℝ := first_series_term1 / (1 - (first_series_term2 / first_series_term1))
  let second_series_sum : ℝ := second_series_term1 / (1 - (second_series_term2 / second_series_term1))
  second_series_sum = 3 * first_series_sum → n = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_series_relation_l1539_153926


namespace NUMINAMATH_CALUDE_circle_tangent_probability_main_theorem_l1539_153956

/-- The probability that two circles have exactly two common tangent lines -/
theorem circle_tangent_probability : Real → Prop := fun p =>
  let r_min : Real := 4
  let r_max : Real := 9
  let circle1_center : Real × Real := (2, -1)
  let circle2_center : Real × Real := (-1, 3)
  let circle1_radius : Real := 2
  let valid_r_min : Real := 3
  let valid_r_max : Real := 7
  p = (valid_r_max - valid_r_min) / (r_max - r_min)

/-- The main theorem -/
theorem main_theorem : circle_tangent_probability (4/5) := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_probability_main_theorem_l1539_153956


namespace NUMINAMATH_CALUDE_prime_sum_product_93_178_l1539_153943

theorem prime_sum_product_93_178 : 
  ∃ p q : ℕ, 
    Prime p ∧ Prime q ∧ p + q = 93 ∧ p * q = 178 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_93_178_l1539_153943


namespace NUMINAMATH_CALUDE_sum_consecutive_odds_not_even_not_div_four_l1539_153946

theorem sum_consecutive_odds_not_even_not_div_four (n : ℤ) (m : ℤ) :
  ¬(∃ (k : ℤ), 4 * (n + 1) = 2 * k ∧ ¬(∃ (l : ℤ), 2 * k = 4 * l)) :=
by sorry

end NUMINAMATH_CALUDE_sum_consecutive_odds_not_even_not_div_four_l1539_153946


namespace NUMINAMATH_CALUDE_problem_statements_l1539_153931

theorem problem_statements :
  (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) ∧
  (∃ x y : ℝ, (abs x > abs y ∧ x ≤ y) ∧ ∃ x y : ℝ, (x > y ∧ abs x ≤ abs y)) ∧
  (∃ x : ℤ, x^2 ≤ 0) ∧
  (∀ m : ℝ, (∃ x y : ℝ, x^2 - 2*x + m = 0 ∧ x > 0 ∧ y < 0) ↔ m < 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l1539_153931


namespace NUMINAMATH_CALUDE_inequality_proof_l1539_153998

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  1 / a + 4 / (1 - a) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1539_153998


namespace NUMINAMATH_CALUDE_x_value_l1539_153984

theorem x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^3) (h2 : x/9 = 9*y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1539_153984


namespace NUMINAMATH_CALUDE_quadratic_inequality_domain_l1539_153917

theorem quadratic_inequality_domain (a : ℝ) :
  (∀ x : ℝ, (x < 1 ∨ x > 5) → x^2 - 2*(a-2)*x + a > 0) ↔ a ∈ Set.Ioo 1 5 ∪ Set.Ioc 5 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_domain_l1539_153917


namespace NUMINAMATH_CALUDE_daily_average_is_40_l1539_153983

/-- Represents the daily average of borrowed books -/
def daily_average : ℝ := 40

/-- Represents the total number of books borrowed in a week -/
def total_weekly_books : ℕ := 216

/-- Represents the borrowing rate on Friday as a multiplier of the daily average -/
def friday_rate : ℝ := 1.4

/-- Theorem stating that given the conditions, the daily average of borrowed books is 40 -/
theorem daily_average_is_40 :
  daily_average * 4 + daily_average * friday_rate = total_weekly_books :=
by sorry

end NUMINAMATH_CALUDE_daily_average_is_40_l1539_153983


namespace NUMINAMATH_CALUDE_product_of_special_integers_l1539_153961

theorem product_of_special_integers (A B C D : ℕ+) 
  (sum_eq : A + B + C + D = 70)
  (def_A : A = 3 * C + 1)
  (def_B : B = 3 * C + 5)
  (def_D : D = 3 * C * C) :
  A * B * C * D = 16896 := by
  sorry

end NUMINAMATH_CALUDE_product_of_special_integers_l1539_153961


namespace NUMINAMATH_CALUDE_minimum_j_10_l1539_153906

/-- A function is stringent if it satisfies the given inequality for all positive integers x and y. -/
def Stringent (f : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, f x + f y ≥ 2 * x.val ^ 2 - y.val

/-- The sum of j from 1 to 15 -/
def SumJ (j : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (λ i => j ⟨i + 1, Nat.succ_pos i⟩)

theorem minimum_j_10 :
  ∃ j : ℕ+ → ℤ,
    Stringent j ∧
    (∀ k : ℕ+ → ℤ, Stringent k → SumJ j ≤ SumJ k) ∧
    j ⟨10, by norm_num⟩ = 137 ∧
    (∀ k : ℕ+ → ℤ, Stringent k → (∀ i : ℕ+, SumJ j ≤ SumJ k) → j ⟨10, by norm_num⟩ ≤ k ⟨10, by norm_num⟩) :=
by sorry

end NUMINAMATH_CALUDE_minimum_j_10_l1539_153906


namespace NUMINAMATH_CALUDE_profit_maximizing_volume_l1539_153908

/-- Annual fixed cost in ten thousand dollars -/
def fixed_cost : ℝ := 10

/-- Variable cost per thousand items in ten thousand dollars -/
def variable_cost : ℝ := 2.7

/-- Revenue function in ten thousand dollars -/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then
    10.8 - x^2 / 30
  else if x > 10 then
    108 / x - 1000 / (3 * x^2)
  else
    0

/-- Profit function in ten thousand dollars -/
noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then
    x * R x - (fixed_cost + variable_cost * x)
  else if x > 10 then
    x * R x - (fixed_cost + variable_cost * x)
  else
    0

/-- Theorem stating that the profit-maximizing production volume is 9 thousand items -/
theorem profit_maximizing_volume :
  ∃ (max_profit : ℝ), W 9 = max_profit ∧ ∀ x, W x ≤ max_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_maximizing_volume_l1539_153908


namespace NUMINAMATH_CALUDE_interior_angle_regular_pentagon_interior_angle_regular_pentagon_is_108_l1539_153980

/-- The measure of one interior angle of a regular pentagon is 108 degrees. -/
theorem interior_angle_regular_pentagon : ℝ :=
  let n : ℕ := 5  -- number of sides in a pentagon
  let S : ℝ := 180 * (n - 2)  -- sum of interior angles formula
  S / n

/-- Proof that the measure of one interior angle of a regular pentagon is 108 degrees. -/
theorem interior_angle_regular_pentagon_is_108 :
  interior_angle_regular_pentagon = 108 := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_regular_pentagon_interior_angle_regular_pentagon_is_108_l1539_153980


namespace NUMINAMATH_CALUDE_max_revenue_price_l1539_153910

/-- The revenue function for the toy shop -/
def revenue (p : ℝ) : ℝ := p * (100 - 4 * p)

/-- The theorem stating the price that maximizes revenue -/
theorem max_revenue_price :
  ∃ (p : ℝ), p ≤ 20 ∧ ∀ (q : ℝ), q ≤ 20 → revenue p ≥ revenue q ∧ p = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_max_revenue_price_l1539_153910


namespace NUMINAMATH_CALUDE_probability_ace_king_queen_standard_deck_l1539_153959

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_aces : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)

/-- Calculates the probability of drawing an Ace, then a King, then a Queen from a standard deck without replacement -/
def probability_ace_king_queen (d : Deck) : ℚ :=
  (d.num_aces : ℚ) / d.total_cards *
  (d.num_kings : ℚ) / (d.total_cards - 1) *
  (d.num_queens : ℚ) / (d.total_cards - 2)

/-- Theorem stating the probability of drawing an Ace, then a King, then a Queen from a standard 52-card deck -/
theorem probability_ace_king_queen_standard_deck :
  probability_ace_king_queen ⟨52, 4, 4, 4⟩ = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_probability_ace_king_queen_standard_deck_l1539_153959


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1539_153964

theorem perfect_square_trinomial (a b : ℝ) : 
  (b - a = -7) → 
  (∃ k : ℝ, ∀ x : ℝ, 16 * x^2 + 144 * x + (a + b) = (k * x + (a + b) / (2 * k))^2) ↔ 
  (a = 165.5 ∧ b = 158.5) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1539_153964


namespace NUMINAMATH_CALUDE_lisa_caffeine_consumption_l1539_153927

/-- Represents the number of beverages Lisa drinks -/
structure BeverageCount where
  coffee : ℕ
  soda : ℕ
  tea : ℕ

/-- Represents the caffeine content of each beverage in milligrams -/
structure CaffeineContent where
  coffee : ℕ
  soda : ℕ
  tea : ℕ

/-- Calculates the total caffeine consumed based on beverage count and caffeine content -/
def totalCaffeine (count : BeverageCount) (content : CaffeineContent) : ℕ :=
  count.coffee * content.coffee + count.soda * content.soda + count.tea * content.tea

/-- Lisa's daily caffeine goal in milligrams -/
def caffeineGoal : ℕ := 200

/-- Theorem stating Lisa's caffeine consumption and excess -/
theorem lisa_caffeine_consumption 
  (lisas_beverages : BeverageCount)
  (caffeine_per_beverage : CaffeineContent)
  (h1 : lisas_beverages.coffee = 3)
  (h2 : lisas_beverages.soda = 1)
  (h3 : lisas_beverages.tea = 2)
  (h4 : caffeine_per_beverage.coffee = 80)
  (h5 : caffeine_per_beverage.soda = 40)
  (h6 : caffeine_per_beverage.tea = 50) :
  totalCaffeine lisas_beverages caffeine_per_beverage = 380 ∧
  totalCaffeine lisas_beverages caffeine_per_beverage - caffeineGoal = 180 := by
  sorry

end NUMINAMATH_CALUDE_lisa_caffeine_consumption_l1539_153927


namespace NUMINAMATH_CALUDE_cars_per_row_section_h_l1539_153919

/-- Prove that the number of cars in each row of Section H is 9 --/
theorem cars_per_row_section_h (
  section_g_rows : ℕ)
  (section_g_cars_per_row : ℕ)
  (section_h_rows : ℕ)
  (cars_per_minute : ℕ)
  (search_time : ℕ)
  (h_section_g_rows : section_g_rows = 15)
  (h_section_g_cars_per_row : section_g_cars_per_row = 10)
  (h_section_h_rows : section_h_rows = 20)
  (h_cars_per_minute : cars_per_minute = 11)
  (h_search_time : search_time = 30)
  : (cars_per_minute * search_time - section_g_rows * section_g_cars_per_row) / section_h_rows = 9 := by
  sorry

end NUMINAMATH_CALUDE_cars_per_row_section_h_l1539_153919


namespace NUMINAMATH_CALUDE_circle_symmetry_minimum_l1539_153904

theorem circle_symmetry_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 + y^2 - 4*a*x - 2*b*y - 5 = 0 → x + 2*y - 1 = 0) →
  (∃ m : ℝ, m = 4/a + 1/b ∧ ∀ k : ℝ, k = 4/a + 1/b → m ≤ k) →
  4/a + 1/b = 18 := by
sorry

end NUMINAMATH_CALUDE_circle_symmetry_minimum_l1539_153904


namespace NUMINAMATH_CALUDE_point_motion_time_l1539_153972

/-- 
Given two points A and B initially separated by distance a, moving along different sides of a right angle 
towards its vertex with constant speed v, where B reaches the vertex t units of time before A, 
this theorem states the time x that A takes to reach the vertex.
-/
theorem point_motion_time (a v t : ℝ) (h : a > v * t) : 
  ∃ x : ℝ, x = (t * v + Real.sqrt (2 * a^2 - v^2 * t^2)) / (2 * v) ∧ 
  x * v = Real.sqrt ((x * v)^2 + ((x - t) * v)^2) :=
sorry

end NUMINAMATH_CALUDE_point_motion_time_l1539_153972


namespace NUMINAMATH_CALUDE_annie_future_age_l1539_153916

theorem annie_future_age (anna_current_age : ℕ) (annie_current_age : ℕ) : 
  anna_current_age = 13 →
  annie_current_age = 3 * anna_current_age →
  (3 * anna_current_age + (annie_current_age - anna_current_age) = 65) :=
by
  sorry


end NUMINAMATH_CALUDE_annie_future_age_l1539_153916


namespace NUMINAMATH_CALUDE_ratio_equality_l1539_153905

theorem ratio_equality (x : ℚ) : (1 : ℚ) / 3 = (5 : ℚ) / (3 * x) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1539_153905


namespace NUMINAMATH_CALUDE_worker_wages_l1539_153976

theorem worker_wages (workers1 workers2 days1 days2 total_wages2 : ℕ)
  (hw1 : workers1 = 15)
  (hw2 : workers2 = 19)
  (hd1 : days1 = 6)
  (hd2 : days2 = 5)
  (ht2 : total_wages2 = 9975) :
  workers1 * days1 * (total_wages2 / (workers2 * days2)) = 9450 := by
  sorry

end NUMINAMATH_CALUDE_worker_wages_l1539_153976


namespace NUMINAMATH_CALUDE_dans_initial_limes_l1539_153903

/-- 
Given that Dan gave Sara some limes and has some limes left, 
this theorem proves the initial number of limes Dan picked.
-/
theorem dans_initial_limes 
  (limes_given_to_sara : ℕ) 
  (limes_left_with_dan : ℕ) 
  (h1 : limes_given_to_sara = 4) 
  (h2 : limes_left_with_dan = 5) : 
  limes_given_to_sara + limes_left_with_dan = 9 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_limes_l1539_153903


namespace NUMINAMATH_CALUDE_male_outnumber_female_l1539_153929

theorem male_outnumber_female (total : ℕ) (male : ℕ) 
  (h1 : total = 928) 
  (h2 : male = 713) : 
  male - (total - male) = 498 := by
  sorry

end NUMINAMATH_CALUDE_male_outnumber_female_l1539_153929


namespace NUMINAMATH_CALUDE_f_properties_l1539_153933

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem f_properties :
  (f 0 = 1) ∧
  (f 1 = 1/2) ∧
  (∀ x : ℝ, 0 < f x ∧ f x ≤ 1) ∧
  (∀ y : ℝ, 0 < y ∧ y ≤ 1 → ∃ x : ℝ, f x = y) := by sorry

end NUMINAMATH_CALUDE_f_properties_l1539_153933


namespace NUMINAMATH_CALUDE_parity_of_expression_l1539_153951

theorem parity_of_expression (p m : ℤ) (h_p_odd : Odd p) :
  Odd (p^2 + 3*m*p) ↔ Even m := by
sorry

end NUMINAMATH_CALUDE_parity_of_expression_l1539_153951


namespace NUMINAMATH_CALUDE_symmetry_propositions_l1539_153924

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Proposition ①
def prop1 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x - 1) = f (x + 1)) →
  (∀ x y : ℝ, f (1 + (x - 1)) = f (1 - (x - 1)) ∧ y = f (x - 1) ↔ y = f (2 - x))

-- Proposition ②
def prop2 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f x = -f (-x)) →
  (∀ x y : ℝ, y = f (x - 1) ↔ -y = f (2 - x))

-- Proposition ③
def prop3 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 1) + f (1 - x) = 0) →
  (∀ x y : ℝ, y = f x ↔ -y = f (2 - x))

-- Proposition ④
def prop4 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, y = f (x - 1) ↔ y = f (1 - x)

-- Theorem stating which propositions are correct
theorem symmetry_propositions :
  ¬ (∀ f : ℝ → ℝ, prop1 f) ∧
  (∀ f : ℝ → ℝ, prop2 f) ∧
  (∀ f : ℝ → ℝ, prop3 f) ∧
  ¬ (∀ f : ℝ → ℝ, prop4 f) :=
sorry

end NUMINAMATH_CALUDE_symmetry_propositions_l1539_153924


namespace NUMINAMATH_CALUDE_book_sale_profit_l1539_153997

/-- Calculates the percent profit for a book sale given the cost, markup percentage, and discount percentage. -/
theorem book_sale_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) :
  cost = 50 ∧ markup_percent = 30 ∧ discount_percent = 10 →
  (((cost * (1 + markup_percent / 100)) * (1 - discount_percent / 100) - cost) / cost) * 100 = 17 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_profit_l1539_153997


namespace NUMINAMATH_CALUDE_permutation_sum_squares_values_l1539_153934

theorem permutation_sum_squares_values (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) : 
  ∃! (s : Finset ℝ), 
    s.card = 3 ∧ 
    (∀ (x y z t : ℝ), ({x, y, z, t} : Finset ℝ) = {a, b, c, d} → 
      ((x - y)^2 + (y - z)^2 + (z - t)^2 + (t - x)^2) ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_permutation_sum_squares_values_l1539_153934


namespace NUMINAMATH_CALUDE_truth_and_lie_l1539_153954

/-- Represents a person who either always tells the truth or always lies -/
inductive Person
| Truthful
| Liar

/-- The setup of three people sitting side by side -/
structure Setup :=
  (left : Person)
  (middle : Person)
  (right : Person)

/-- The statement made by the left person about the middle person's response -/
def leftStatement (s : Setup) : Prop :=
  s.middle = Person.Truthful

/-- The statement made by the right person about the middle person's response -/
def rightStatement (s : Setup) : Prop :=
  s.middle = Person.Liar

theorem truth_and_lie (s : Setup) :
  (leftStatement s = true ↔ s.left = Person.Truthful) ∧
  (rightStatement s = false ↔ s.right = Person.Liar) :=
sorry

end NUMINAMATH_CALUDE_truth_and_lie_l1539_153954


namespace NUMINAMATH_CALUDE_min_value_sum_cubic_ratios_l1539_153952

theorem min_value_sum_cubic_ratios (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 9) :
  (x^3 + y^3) / (x + y) + (x^3 + z^3) / (x + z) + (y^3 + z^3) / (y + z) ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_cubic_ratios_l1539_153952


namespace NUMINAMATH_CALUDE_equal_real_roots_no_real_solutions_l1539_153915

-- Define the quadratic equation
def quadratic_equation (a b x : ℝ) : Prop := a * x^2 + b * x + (1/4 : ℝ) = 0

-- Part 1: Equal real roots condition
theorem equal_real_roots (a b : ℝ) (h : a ≠ 0) :
  a = 1 ∧ b = 1 → ∃! x : ℝ, quadratic_equation a b x :=
sorry

-- Part 2: No real solutions condition
theorem no_real_solutions (a b : ℝ) (h1 : a > 1) (h2 : 0 < b) (h3 : b < 1) :
  ¬∃ x : ℝ, quadratic_equation a b x :=
sorry

end NUMINAMATH_CALUDE_equal_real_roots_no_real_solutions_l1539_153915


namespace NUMINAMATH_CALUDE_complex_problem_l1539_153950

theorem complex_problem (b : ℝ) (z : ℂ) (h1 : z = 3 + b * I) 
  (h2 : (Complex.I * (Complex.I * ((1 + 3 * I) * z))).re = 0) : 
  z = 3 + I ∧ Complex.abs (z / (2 + I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_problem_l1539_153950


namespace NUMINAMATH_CALUDE_initial_marbles_l1539_153942

/-- Proves that if a person has 7 marbles left after sharing 3 marbles, 
    then they initially had 10 marbles. -/
theorem initial_marbles (shared : ℕ) (left : ℕ) (initial : ℕ) : 
  shared = 3 → left = 7 → initial = shared + left → initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_l1539_153942


namespace NUMINAMATH_CALUDE_unique_prime_n_l1539_153948

def f (n : ℕ+) : ℤ := -n^4 + n^3 - 4*n^2 + 18*n - 19

theorem unique_prime_n : ∃! (n : ℕ+), Nat.Prime (Int.natAbs (f n)) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_n_l1539_153948


namespace NUMINAMATH_CALUDE_sqrt_four_twos_to_fourth_l1539_153994

theorem sqrt_four_twos_to_fourth : Real.sqrt (2^4 + 2^4 + 2^4 + 2^4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_twos_to_fourth_l1539_153994


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1539_153949

theorem fraction_sum_equality (p q m n : ℕ+) (x : ℚ) 
  (h1 : (p : ℚ) / q = (m : ℚ) / n)
  (h2 : (p : ℚ) / q = 4 / 5)
  (h3 : x = 1 / 7) :
  x + ((2 * q - p + 3 * m - 2 * n) : ℚ) / (2 * q + p - m + n) = 71 / 105 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1539_153949


namespace NUMINAMATH_CALUDE_diana_remaining_paint_l1539_153947

/-- The amount of paint required for one statue in gallons -/
def paint_per_statue : ℚ := 1/8

/-- The number of statues Diana can paint with the remaining paint -/
def statues_to_paint : ℕ := 7

/-- The total amount of paint Diana has remaining in gallons -/
def remaining_paint : ℚ := paint_per_statue * statues_to_paint

theorem diana_remaining_paint : remaining_paint = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_diana_remaining_paint_l1539_153947


namespace NUMINAMATH_CALUDE_inequality_proof_l1539_153955

theorem inequality_proof (a b c d : ℝ) 
  (non_neg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (sum_condition : a*b + a*c + a*d + b*c + b*d + c*d = 6) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1) + 1 / (d^2 + 1) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1539_153955


namespace NUMINAMATH_CALUDE_three_digit_prime_discriminant_not_square_l1539_153958

theorem three_digit_prime_discriminant_not_square (A B C : ℕ) : 
  (100 * A + 10 * B + C).Prime → 
  ¬∃ (n : ℤ), B^2 - 4*A*C = n^2 := by
sorry

end NUMINAMATH_CALUDE_three_digit_prime_discriminant_not_square_l1539_153958


namespace NUMINAMATH_CALUDE_elevator_optimal_stop_l1539_153936

def total_floors : ℕ := 12
def num_people : ℕ := 11

def dissatisfaction (n : ℕ) : ℕ :=
  let down_sum := (n - 2) * (n - 1) / 2
  let up_sum := (total_floors - n) * (total_floors - n + 1)
  down_sum + 2 * up_sum

theorem elevator_optimal_stop :
  ∀ k : ℕ, 2 ≤ k ∧ k ≤ total_floors →
    dissatisfaction 9 ≤ dissatisfaction k :=
sorry

end NUMINAMATH_CALUDE_elevator_optimal_stop_l1539_153936


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1539_153999

-- Define the concept of opposite
def opposite (x : ℤ) : ℤ := -x

-- State the theorem
theorem opposite_of_negative_two : opposite (-2) = 2 := by
  -- The proof would go here, but we're skipping it as requested
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1539_153999


namespace NUMINAMATH_CALUDE_rogers_allowance_theorem_l1539_153985

/-- Roger's weekly allowance problem -/
theorem rogers_allowance_theorem (B : ℝ) (m s p : ℝ) : 
  (m = (1/4) * (B - s)) → 
  (s = (1/10) * (B - m)) → 
  (p = (1/10) * (m + s)) → 
  (m + s + p) / B = 22 / 65 := by
  sorry

end NUMINAMATH_CALUDE_rogers_allowance_theorem_l1539_153985


namespace NUMINAMATH_CALUDE_geese_in_marsh_l1539_153957

theorem geese_in_marsh (total_birds ducks : ℕ) (h1 : total_birds = 95) (h2 : ducks = 37) :
  total_birds - ducks = 58 := by
  sorry

end NUMINAMATH_CALUDE_geese_in_marsh_l1539_153957


namespace NUMINAMATH_CALUDE_spermatogenesis_experiment_verification_l1539_153921

-- Define the available materials and tools
inductive Material
| MouseLiver
| Testes
| Kidneys

inductive Stain
| SudanIII
| AceticOrcein
| JanusGreen

inductive Tool
| DissociationFixative

-- Define the experiment steps
structure ExperimentSteps where
  material : Material
  fixative : Tool
  stain : Stain

-- Define the experiment result
structure ExperimentResult where
  cellTypesObserved : Nat

-- Define the correct experiment setup and result
def correctExperiment : ExperimentSteps := {
  material := Material.Testes,
  fixative := Tool.DissociationFixative,
  stain := Stain.AceticOrcein
}

def correctResult : ExperimentResult := {
  cellTypesObserved := 3
}

-- Theorem statement
theorem spermatogenesis_experiment_verification :
  ∀ (setup : ExperimentSteps) (result : ExperimentResult),
  setup = correctExperiment ∧ result = correctResult →
  setup.material = Material.Testes ∧
  setup.fixative = Tool.DissociationFixative ∧
  setup.stain = Stain.AceticOrcein ∧
  result.cellTypesObserved = 3 :=
by sorry

end NUMINAMATH_CALUDE_spermatogenesis_experiment_verification_l1539_153921


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1539_153925

/-- The distance between the vertices of the hyperbola x²/64 - y²/49 = 1 is 16 -/
theorem hyperbola_vertex_distance : 
  let h : ℝ → ℝ → Prop := fun x y => x^2/64 - y^2/49 = 1
  ∃ (x₁ x₂ : ℝ), h x₁ 0 ∧ h x₂ 0 ∧ |x₁ - x₂| = 16 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1539_153925


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_non_negative_condition_two_roots_condition_l1539_153969

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2*a + 6

/-- Theorem for part I -/
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ≥ 4, ∀ y ≥ 4, x < y → f a x < f a y) ↔ a ≥ -3 :=
sorry

/-- Theorem for part II -/
theorem non_negative_condition (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) ↔ -1 ≤ a ∧ a ≤ 5 :=
sorry

/-- Theorem for part III -/
theorem two_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x > 1 ∧ y > 1 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ -5/4 < a ∧ a < -1 :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_non_negative_condition_two_roots_condition_l1539_153969


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1539_153941

theorem necessary_not_sufficient_condition (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (∃ y, 0 < y ∧ y < Real.pi / 2 ∧ (Real.sqrt y - 1 / Real.sin y < 0) ∧ ¬(1 / Real.sin y - y > 0)) ∧
  (∀ z, 0 < z ∧ z < Real.pi / 2 ∧ (1 / Real.sin z - z > 0) → (Real.sqrt z - 1 / Real.sin z < 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1539_153941


namespace NUMINAMATH_CALUDE_probability_of_red_bean_l1539_153932

/-- The probability of choosing a red bean from a bag -/
theorem probability_of_red_bean 
  (initial_red : ℕ) 
  (initial_black : ℕ) 
  (added_red : ℕ) 
  (added_black : ℕ) 
  (h1 : initial_red = 5)
  (h2 : initial_black = 9)
  (h3 : added_red = 3)
  (h4 : added_black = 3) : 
  (initial_red + added_red : ℚ) / (initial_red + initial_black + added_red + added_black) = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_of_red_bean_l1539_153932


namespace NUMINAMATH_CALUDE_x_values_l1539_153990

theorem x_values (A : Set ℝ) (x : ℝ) (h1 : A = {0, 1, x^2 - 5*x}) (h2 : -4 ∈ A) :
  x = 1 ∨ x = 4 := by
sorry

end NUMINAMATH_CALUDE_x_values_l1539_153990


namespace NUMINAMATH_CALUDE_rachels_homework_l1539_153911

theorem rachels_homework (math_pages reading_pages total_pages : ℕ) : 
  reading_pages = math_pages + 3 →
  total_pages = math_pages + reading_pages →
  total_pages = 23 →
  math_pages = 10 := by
sorry

end NUMINAMATH_CALUDE_rachels_homework_l1539_153911


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1539_153902

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) : 
  i ^ 2 = -1 →
  z * (2 - i) = i ^ 3 →
  z.im = -2/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1539_153902


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_l1539_153993

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

-- Define the right vertex M
def right_vertex (M : ℝ × ℝ) : Prop := 
  M.1 = 2 ∧ M.2 = 0 ∧ ellipse_C M.1 M.2

-- Define points A and B on the ellipse
def on_ellipse (P : ℝ × ℝ) : Prop := 
  ellipse_C P.1 P.2 ∧ P ≠ (2, 0)

-- Define the product of slopes condition
def slope_product (M A B : ℝ × ℝ) : Prop :=
  (A.2 / (A.1 - M.1)) * (B.2 / (B.1 - M.1)) = 1/4

-- Theorem statement
theorem ellipse_fixed_point 
  (M A B : ℝ × ℝ) 
  (hM : right_vertex M) 
  (hA : on_ellipse A) 
  (hB : on_ellipse B) 
  (hAB : A ≠ B) 
  (hSlope : slope_product M A B) :
  ∃ (k : ℝ), A.2 - B.2 = k * (A.1 - B.1) ∧ 
             A.2 = k * (A.1 + 4) ∧ 
             B.2 = k * (B.1 + 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_l1539_153993


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l1539_153945

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 10 = 36 → Nat.gcd n 10 = 5 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l1539_153945


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1539_153960

theorem simplify_polynomial (x : ℝ) : (3 * x^2 + 6 * x - 5) - (2 * x^2 + 4 * x - 8) = x^2 + 2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1539_153960


namespace NUMINAMATH_CALUDE_intersection_complement_equals_half_open_interval_l1539_153978

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- State the theorem
theorem intersection_complement_equals_half_open_interval :
  M ∩ (Set.compl N) = Set.Icc (-2) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_half_open_interval_l1539_153978


namespace NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l1539_153966

-- Define the necessary types
structure Point :=
  (x y : ℝ)

structure Triangle :=
  (A B C : Point)

-- Define the given information
def B : Point := sorry
def G : Point := sorry  -- centroid
def L : Point := sorry  -- intersection of symmedian from B with circumcircle

-- Define the necessary concepts
def isCentroid (G : Point) (t : Triangle) : Prop := sorry
def isSymmedianIntersection (L : Point) (t : Triangle) : Prop := sorry

-- The theorem statement
theorem triangle_reconstruction_uniqueness :
  ∃! (t : Triangle), 
    t.B = B ∧ 
    isCentroid G t ∧ 
    isSymmedianIntersection L t :=
sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l1539_153966


namespace NUMINAMATH_CALUDE_n_sticks_ge_n_plus_one_minos_l1539_153992

/-- An n-stick is a connected figure of n matches of length 1, placed horizontally or vertically, no two touching except at ends. -/
def NStick (n : ℕ) : Type := sorry

/-- An n-mino is a shape built by connecting n squares of side length 1 on their sides, with a path between each two squares. -/
def NMino (n : ℕ) : Type := sorry

/-- S_n is the number of n-sticks -/
def S (n : ℕ) : ℕ := sorry

/-- M_n is the number of n-minos -/
def M (n : ℕ) : ℕ := sorry

/-- For any natural number n, the number of n-sticks is greater than or equal to the number of (n+1)-minos. -/
theorem n_sticks_ge_n_plus_one_minos (n : ℕ) : S n ≥ M (n + 1) := by sorry

end NUMINAMATH_CALUDE_n_sticks_ge_n_plus_one_minos_l1539_153992


namespace NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l1539_153986

/-- The ellipse with equation x²/4 + y² = 1 and eccentricity √3/2 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + p.2^2 = 1}

/-- The line x = ky - 1 -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1 = k * p.2 - 1}

/-- Point M is the reflection of A across the x-axis -/
def ReflectAcrossXAxis (A M : ℝ × ℝ) : Prop :=
  M.1 = A.1 ∧ M.2 = -A.2

/-- The line passing through two points -/
def LineThroughPoints (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r | ∃ t : ℝ, r = (1 - t) • p + t • q}

theorem ellipse_intersection_fixed_point (k : ℝ) 
  (A B : ℝ × ℝ) (hA : A ∈ Ellipse ∩ Line k) (hB : B ∈ Ellipse ∩ Line k) 
  (M : ℝ × ℝ) (hM : ReflectAcrossXAxis A M) (hAB : A ≠ B) :
  ∃ P : ℝ × ℝ, P ∈ LineThroughPoints M B ∧ P.1 = -4 ∧ P.2 = 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l1539_153986


namespace NUMINAMATH_CALUDE_quadratic_coefficient_is_one_l1539_153913

/-- The quadratic equation x^2 - 2x + 1 = 0 -/
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x + 1 = 0

/-- The coefficient of the quadratic term in the equation x^2 - 2x + 1 = 0 -/
def quadratic_coefficient : ℝ := 1

theorem quadratic_coefficient_is_one : 
  quadratic_coefficient = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_is_one_l1539_153913


namespace NUMINAMATH_CALUDE_total_fish_weight_l1539_153987

/-- Calculates the total weight of fish in James' three tanks -/
theorem total_fish_weight (goldfish_weight guppy_weight angelfish_weight : ℝ)
  (goldfish_count_1 guppy_count_1 : ℕ)
  (goldfish_count_2 guppy_count_2 : ℕ)
  (goldfish_count_3 guppy_count_3 angelfish_count_3 : ℕ)
  (h1 : goldfish_weight = 0.08)
  (h2 : guppy_weight = 0.05)
  (h3 : angelfish_weight = 0.14)
  (h4 : goldfish_count_1 = 15)
  (h5 : guppy_count_1 = 12)
  (h6 : goldfish_count_2 = 2 * goldfish_count_1)
  (h7 : guppy_count_2 = 3 * guppy_count_1)
  (h8 : goldfish_count_3 = 3 * goldfish_count_1)
  (h9 : guppy_count_3 = 2 * guppy_count_1)
  (h10 : angelfish_count_3 = 5) :
  goldfish_weight * (goldfish_count_1 + goldfish_count_2 + goldfish_count_3 : ℝ) +
  guppy_weight * (guppy_count_1 + guppy_count_2 + guppy_count_3 : ℝ) +
  angelfish_weight * angelfish_count_3 = 11.5 := by
  sorry


end NUMINAMATH_CALUDE_total_fish_weight_l1539_153987


namespace NUMINAMATH_CALUDE_regular_hexagon_properties_l1539_153982

/-- A regular hexagon inscribed in a circle -/
structure RegularHexagon where
  /-- The side length of the hexagon -/
  side_length : ℝ
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The circumference of the circumscribed circle -/
  circumference : ℝ
  /-- The arc length corresponding to one side of the hexagon -/
  arc_length : ℝ
  /-- The area of the hexagon -/
  area : ℝ

/-- Properties of a regular hexagon with side length 6 -/
theorem regular_hexagon_properties :
  ∃ (h : RegularHexagon),
    h.side_length = 6 ∧
    h.radius = 6 ∧
    h.circumference = 12 * Real.pi ∧
    h.arc_length = 2 * Real.pi ∧
    h.area = 54 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_properties_l1539_153982


namespace NUMINAMATH_CALUDE_total_rainfall_2004_l1539_153995

/-- The average monthly rainfall in Mathborough in 2003 (in mm) -/
def avg_rainfall_2003 : ℝ := 41.5

/-- The increase in average monthly rainfall from 2003 to 2004 (in mm) -/
def rainfall_increase : ℝ := 2

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: The total amount of rain that fell in Mathborough in 2004 was 522 mm -/
theorem total_rainfall_2004 : 
  (avg_rainfall_2003 + rainfall_increase) * months_in_year = 522 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_2004_l1539_153995


namespace NUMINAMATH_CALUDE_matts_stair_climbing_rate_l1539_153914

theorem matts_stair_climbing_rate 
  (M : ℝ)  -- Matt's rate of climbing stairs in steps per minute
  (h1 : M > 0)  -- Matt's rate is positive
  (h2 : ∃ t : ℝ, t > 0 ∧ M * t = 220 ∧ (M + 5) * t = 275)  -- Condition when Matt reaches 220 steps and Tom reaches 275 steps
  : M = 20 := by
  sorry

end NUMINAMATH_CALUDE_matts_stair_climbing_rate_l1539_153914


namespace NUMINAMATH_CALUDE_family_reunion_attendance_l1539_153975

/-- The number of people at a family reunion --/
def family_reunion (male_adults female_adults children : ℕ) : ℕ :=
  male_adults + female_adults + children

/-- Theorem: Given the conditions, the family reunion has 750 people --/
theorem family_reunion_attendance :
  ∀ (male_adults female_adults children : ℕ),
  male_adults = 100 →
  female_adults = male_adults + 50 →
  children = 2 * (male_adults + female_adults) →
  family_reunion male_adults female_adults children = 750 :=
by
  sorry

end NUMINAMATH_CALUDE_family_reunion_attendance_l1539_153975


namespace NUMINAMATH_CALUDE_sum_of_2008th_powers_l1539_153991

theorem sum_of_2008th_powers (a b c : ℝ) 
  (sum_eq_3 : a + b + c = 3) 
  (sum_squares_eq_3 : a^2 + b^2 + c^2 = 3) : 
  a^2008 + b^2008 + c^2008 = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_2008th_powers_l1539_153991


namespace NUMINAMATH_CALUDE_one_third_of_product_l1539_153901

theorem one_third_of_product : (1 / 3 : ℚ) * 7 * 9 * 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_product_l1539_153901


namespace NUMINAMATH_CALUDE_sum_of_roots_l1539_153979

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a - 1 = 0) 
  (hb : b^3 - 3*b^2 + 5*b - 5 = 0) : 
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1539_153979


namespace NUMINAMATH_CALUDE_three_intersections_l1539_153981

/-- The number of intersection points between a circle and a parabola -/
def intersection_count (b : ℝ) : ℕ :=
  -- Define the count based on the intersection points
  -- This is a placeholder; the actual implementation would involve solving the system of equations
  sorry

/-- Theorem stating the condition for exactly three intersection points -/
theorem three_intersections (b : ℝ) :
  intersection_count b = 3 ↔ b > (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_three_intersections_l1539_153981


namespace NUMINAMATH_CALUDE_right_triangle_max_area_l1539_153971

theorem right_triangle_max_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (1/2) * a * b ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_max_area_l1539_153971


namespace NUMINAMATH_CALUDE_perfect_squares_between_100_and_400_l1539_153928

theorem perfect_squares_between_100_and_400 : 
  (Finset.filter (fun n => 100 < n^2 ∧ n^2 < 400) (Finset.range 20)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_between_100_and_400_l1539_153928


namespace NUMINAMATH_CALUDE_prob_two_hearts_one_spade_l1539_153937

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (cards : Fin 52)
  (ranks : Fin 13)
  (suits : Fin 4)

/-- Represents the suits in a deck -/
inductive Suit
| hearts
| diamonds
| clubs
| spades

/-- Defines the color of a suit -/
def suitColor (s : Suit) : Bool :=
  match s with
  | Suit.hearts | Suit.diamonds => true  -- Red
  | Suit.clubs | Suit.spades => false    -- Black

/-- Calculates the probability of drawing two hearts followed by a spade -/
def probabilityTwoHeartsOneSpade (d : Deck) : ℚ :=
  13 / 850

/-- Theorem stating the probability of drawing two hearts followed by a spade -/
theorem prob_two_hearts_one_spade (d : Deck) :
  probabilityTwoHeartsOneSpade d = 13 / 850 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_hearts_one_spade_l1539_153937


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l1539_153912

-- Define the function f
def f (x a : ℝ) : ℝ := 5 - |x + a| - |x - 2|

-- Theorem for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | f x 1 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} :=
sorry

-- Theorem for part (2)
theorem range_of_a_when_f_leq_one :
  {a : ℝ | ∀ x, f x a ≤ 1} = {a : ℝ | a ≤ -6 ∨ a ≥ 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l1539_153912


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l1539_153965

/-- Calculate the gain percent on a scooter sale -/
theorem scooter_gain_percent
  (purchase_price : ℝ)
  (repair_cost : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 900)
  (h2 : repair_cost = 300)
  (h3 : selling_price = 1500) :
  (selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l1539_153965


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_18_12_l1539_153963

/-- Perimeter of a parallelogram -/
def parallelogram_perimeter (side1 : ℝ) (side2 : ℝ) : ℝ :=
  2 * (side1 + side2)

/-- Theorem: The perimeter of a parallelogram with sides 18 cm and 12 cm is 60 cm -/
theorem parallelogram_perimeter_18_12 :
  parallelogram_perimeter 18 12 = 60 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_18_12_l1539_153963


namespace NUMINAMATH_CALUDE_derivative_exp_cos_l1539_153989

theorem derivative_exp_cos (x : ℝ) : 
  deriv (λ x => Real.exp x * Real.cos x) x = Real.exp x * (Real.cos x - Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_exp_cos_l1539_153989


namespace NUMINAMATH_CALUDE_quadratic_intersection_theorem_l1539_153900

/-- Quadratic function f(x) = x^2 + 3x + n -/
def f (n : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + n

/-- Predicate for exactly one positive real root -/
def has_exactly_one_positive_root (n : ℝ) : Prop :=
  ∃! x : ℝ, x > 0 ∧ f n x = 0

theorem quadratic_intersection_theorem :
  has_exactly_one_positive_root (-2) ∧
  ∀ n : ℝ, has_exactly_one_positive_root n → n = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_theorem_l1539_153900


namespace NUMINAMATH_CALUDE_tan_alpha_neg_half_implies_expression_neg_third_l1539_153996

theorem tan_alpha_neg_half_implies_expression_neg_third (α : Real) 
  (h : Real.tan α = -1/2) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_neg_half_implies_expression_neg_third_l1539_153996


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1539_153962

theorem simplify_polynomial (x : ℝ) : 
  2*x*(4*x^3 - 3*x + 1) - 4*(2*x^3 - x^2 + 3*x - 5) = 
  8*x^4 - 8*x^3 - 2*x^2 - 10*x + 20 := by sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1539_153962


namespace NUMINAMATH_CALUDE_set_union_problem_l1539_153953

theorem set_union_problem (a b l : ℝ) :
  let A : Set ℝ := {-2, a}
  let B : Set ℝ := {2015^a, b}
  A ∩ B = {l} →
  A ∪ B = {-2, 1, 2015} :=
by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l1539_153953


namespace NUMINAMATH_CALUDE_division_problem_l1539_153939

theorem division_problem (x y z : ℚ) 
  (h1 : x / y = 3) 
  (h2 : y / z = 2 / 5) : 
  z / x = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1539_153939


namespace NUMINAMATH_CALUDE_swim_club_scenario_l1539_153940

/-- Represents a swim club with members, some of whom have passed a lifesaving test
    and some of whom have taken a preparatory course. -/
structure SwimClub where
  total_members : ℕ
  passed_test : ℕ
  not_taken_course : ℕ

/-- The number of members who have taken the preparatory course but not passed the test -/
def members_taken_course_not_passed (club : SwimClub) : ℕ :=
  club.total_members - club.passed_test - club.not_taken_course

/-- Theorem stating the number of members who have taken the preparatory course
    but not passed the test in the given scenario -/
theorem swim_club_scenario :
  let club : SwimClub := {
    total_members := 60,
    passed_test := 18,  -- 30% of 60
    not_taken_course := 30
  }
  members_taken_course_not_passed club = 12 := by
  sorry

end NUMINAMATH_CALUDE_swim_club_scenario_l1539_153940


namespace NUMINAMATH_CALUDE_dorothy_initial_money_l1539_153988

-- Define the family members
inductive FamilyMember
| Dorothy
| Brother
| Parent1
| Parent2
| Grandfather

-- Define the age of a family member
def age (member : FamilyMember) : ℕ :=
  match member with
  | .Dorothy => 15
  | .Brother => 0  -- We don't know exact age, but younger than 18
  | .Parent1 => 18 -- We don't know exact age, but at least 18
  | .Parent2 => 18 -- We don't know exact age, but at least 18
  | .Grandfather => 18 -- We don't know exact age, but at least 18

-- Define the regular ticket price
def regularTicketPrice : ℕ := 10

-- Define the discount rate for young people
def youngDiscount : ℚ := 0.3

-- Define the discounted ticket price function
def ticketPrice (member : FamilyMember) : ℚ :=
  if age member ≤ 18 then
    regularTicketPrice * (1 - youngDiscount)
  else
    regularTicketPrice

-- Define the total cost of tickets for the family
def totalTicketCost : ℚ :=
  ticketPrice FamilyMember.Dorothy +
  ticketPrice FamilyMember.Brother +
  ticketPrice FamilyMember.Parent1 +
  ticketPrice FamilyMember.Parent2 +
  ticketPrice FamilyMember.Grandfather

-- Define Dorothy's remaining money after the trip
def moneyLeftAfterTrip : ℚ := 26

-- Theorem: Dorothy's initial money was $70
theorem dorothy_initial_money :
  totalTicketCost + moneyLeftAfterTrip = 70 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_initial_money_l1539_153988


namespace NUMINAMATH_CALUDE_place_value_comparison_l1539_153907

def number : ℚ := 43597.2468

theorem place_value_comparison : 
  (100 : ℚ) * (number % 1000 - number % 100) / 100 = (number % 0.1 - number % 0.01) / 0.01 := by
  sorry

end NUMINAMATH_CALUDE_place_value_comparison_l1539_153907
