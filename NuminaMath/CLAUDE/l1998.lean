import Mathlib

namespace NUMINAMATH_CALUDE_sandwich_combinations_l1998_199816

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents the number of bread options that can go with a specific meat/cheese combination. -/
def num_bread_options : ℕ := 5

/-- Represents the number of restricted combinations (ham/cheddar and turkey/swiss). -/
def num_restricted_combinations : ℕ := 2

theorem sandwich_combinations :
  (num_breads * num_meats * num_cheeses) - (num_bread_options * num_restricted_combinations) = 200 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l1998_199816


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l1998_199899

theorem logarithmic_equation_solution :
  ∃ x : ℝ, x > 0 ∧ (Real.log x / Real.log 8 + 3 * Real.log (x^2) / Real.log 2 - Real.log x / Real.log 4 = 14) ∧
  x = 2^(12/5) := by
sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l1998_199899


namespace NUMINAMATH_CALUDE_shower_water_usage_l1998_199813

/-- Calculates the total water usage for showers over a given period --/
def total_water_usage (weeks : ℕ) (shower_duration : ℕ) (water_per_minute : ℕ) : ℕ :=
  let days := weeks * 7
  let showers := days / 2
  let total_minutes := showers * shower_duration
  total_minutes * water_per_minute

theorem shower_water_usage : total_water_usage 4 10 2 = 280 := by
  sorry

end NUMINAMATH_CALUDE_shower_water_usage_l1998_199813


namespace NUMINAMATH_CALUDE_transform_is_right_shift_graph_transform_is_right_shift_l1998_199896

-- Define a continuous function f from reals to reals
variable (f : ℝ → ℝ) (hf : Continuous f)

-- Define the transformation function
def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (x - 1)

-- Theorem stating that the transformation is equivalent to a right shift
theorem transform_is_right_shift :
  ∀ x y : ℝ, transform f x = y ↔ f (x - 1) = y :=
by sorry

-- Theorem stating that the graph of the transformed function
-- is equivalent to the original graph shifted 1 unit right
theorem graph_transform_is_right_shift :
  ∀ x y : ℝ, (x, y) ∈ (Set.range (λ x ↦ (x, transform f x))) ↔
             (x - 1, y) ∈ (Set.range (λ x ↦ (x, f x))) :=
by sorry

end NUMINAMATH_CALUDE_transform_is_right_shift_graph_transform_is_right_shift_l1998_199896


namespace NUMINAMATH_CALUDE_a_initial_investment_l1998_199823

/-- Proves that given the conditions, A's initial investment is 3000 units -/
theorem a_initial_investment (b_investment : ℝ) (a_doubles : ℝ → ℝ) 
  (h1 : b_investment = 4500)
  (h2 : ∀ x, a_doubles x = 2 * x)
  (h3 : ∀ x, (x + a_doubles x) / 2 = b_investment) : 
  ∃ a_initial : ℝ, a_initial = 3000 := by
  sorry

end NUMINAMATH_CALUDE_a_initial_investment_l1998_199823


namespace NUMINAMATH_CALUDE_conference_room_arrangements_count_l1998_199870

/-- The number of distinct arrangements of seats in a conference room. -/
def conference_room_arrangements : ℕ :=
  let total_seats : ℕ := 12
  let armchairs : ℕ := 6
  let benches : ℕ := 4
  let stools : ℕ := 2
  Nat.choose total_seats stools * Nat.choose (total_seats - stools) benches

theorem conference_room_arrangements_count :
  conference_room_arrangements = 13860 := by
  sorry

#eval conference_room_arrangements

end NUMINAMATH_CALUDE_conference_room_arrangements_count_l1998_199870


namespace NUMINAMATH_CALUDE_amanda_lost_notebooks_l1998_199856

/-- The number of notebooks Amanda lost -/
def notebooks_lost (initial : ℕ) (ordered : ℕ) (current : ℕ) : ℕ :=
  initial + ordered - current

theorem amanda_lost_notebooks : notebooks_lost 10 6 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_amanda_lost_notebooks_l1998_199856


namespace NUMINAMATH_CALUDE_tangent_point_condition_tangent_lines_equations_l1998_199829

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point M
def point_M (a : ℝ) : ℝ × ℝ := (1, a)

-- Theorem 1: M lies on O iff a = ±√3
theorem tangent_point_condition (a : ℝ) :
  circle_O 1 a ↔ a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
sorry

-- Theorem 2: Tangent lines when a = 2
theorem tangent_lines_equations :
  let M := point_M 2
  ∃ (l₁ l₂ : ℝ → ℝ → Prop),
    (∀ x y, l₁ x y ↔ y = 2) ∧
    (∀ x y, l₂ x y ↔ 4*x + 3*y = 10) ∧
    (∀ x y, l₁ x y → circle_O x y → x = 1 ∧ y = 2) ∧
    (∀ x y, l₂ x y → circle_O x y → x = 1 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_condition_tangent_lines_equations_l1998_199829


namespace NUMINAMATH_CALUDE_min_sum_squares_l1998_199897

theorem min_sum_squares (a b : ℝ) (h : (9 : ℝ) / a^2 + 4 / b^2 = 1) :
  ∃ (min : ℝ), min = 25 ∧ ∀ (x y : ℝ), (9 : ℝ) / x^2 + 4 / y^2 = 1 → x^2 + y^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1998_199897


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1998_199846

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((x + a) / x + (x - a) / x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1998_199846


namespace NUMINAMATH_CALUDE_fraction_division_eval_l1998_199827

theorem fraction_division_eval : (7 / 3) / (8 / 15) = 35 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_eval_l1998_199827


namespace NUMINAMATH_CALUDE_license_plate_count_l1998_199873

/-- The number of possible letters in each position of the license plate -/
def num_letters : ℕ := 26

/-- The number of odd digits available for the first position -/
def num_odd_digits : ℕ := 5

/-- The number of even digits available for the second position -/
def num_even_digits : ℕ := 5

/-- The number of digits that are multiples of 3 available for the third position -/
def num_multiples_of_3 : ℕ := 4

/-- The total number of license plates satisfying the given conditions -/
def total_license_plates : ℕ := num_letters ^ 3 * num_odd_digits * num_even_digits * num_multiples_of_3

theorem license_plate_count : total_license_plates = 878800 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1998_199873


namespace NUMINAMATH_CALUDE_article_cost_l1998_199866

theorem article_cost (sell_price_high : ℝ) (sell_price_low : ℝ) (gain_percentage : ℝ) :
  sell_price_high = 350 →
  sell_price_low = 340 →
  gain_percentage = 0.04 →
  ∃ (cost : ℝ),
    sell_price_high - cost = (1 + gain_percentage) * (sell_price_low - cost) ∧
    cost = 90 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l1998_199866


namespace NUMINAMATH_CALUDE_trig_expression_equals_negative_two_l1998_199853

theorem trig_expression_equals_negative_two :
  5 * Real.sin (π / 2) + 2 * Real.cos 0 - 3 * Real.sin (3 * π / 2) + 10 * Real.cos π = -2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_negative_two_l1998_199853


namespace NUMINAMATH_CALUDE_square_of_999_l1998_199822

theorem square_of_999 : 999 * 999 = 998001 := by
  sorry

end NUMINAMATH_CALUDE_square_of_999_l1998_199822


namespace NUMINAMATH_CALUDE_seating_arrangements_l1998_199868

/-- The number of ways to arrange n people in k seats -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose n items from k items -/
def choose (n k : ℕ) : ℕ := sorry

theorem seating_arrangements : 
  let total_seats : ℕ := 6
  let people : ℕ := 3
  let all_arrangements := arrange people total_seats
  let no_adjacent_empty := choose (total_seats - people + 1) people * arrange people people
  let all_empty_adjacent := choose (total_seats - people + 1) 1 * arrange people people
  all_arrangements - no_adjacent_empty - all_empty_adjacent = 72 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1998_199868


namespace NUMINAMATH_CALUDE_ratio_first_term_to_common_difference_l1998_199855

/-- An arithmetic progression where the sum of the first twenty terms
    is five times the sum of the first ten terms -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference
  sum_condition : (20 * a + 190 * d) = 5 * (10 * a + 45 * d)

/-- The ratio of the first term to the common difference is -7/6 -/
theorem ratio_first_term_to_common_difference
  (ap : ArithmeticProgression) : ap.a / ap.d = -7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_first_term_to_common_difference_l1998_199855


namespace NUMINAMATH_CALUDE_cereal_eating_time_l1998_199879

theorem cereal_eating_time 
  (fat_rate : ℚ) 
  (thin_rate : ℚ) 
  (total_cereal : ℚ) 
  (h1 : fat_rate = 1 / 15) 
  (h2 : thin_rate = 1 / 40) 
  (h3 : total_cereal = 5) : 
  total_cereal / (fat_rate + thin_rate) = 600 / 11 := by
  sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l1998_199879


namespace NUMINAMATH_CALUDE_next_joint_performance_l1998_199898

theorem next_joint_performance (ella_interval : Nat) (felix_interval : Nat) 
  (grace_interval : Nat) (hugo_interval : Nat) 
  (h1 : ella_interval = 5)
  (h2 : felix_interval = 6)
  (h3 : grace_interval = 9)
  (h4 : hugo_interval = 10) :
  Nat.lcm (Nat.lcm (Nat.lcm ella_interval felix_interval) grace_interval) hugo_interval = 90 := by
  sorry

end NUMINAMATH_CALUDE_next_joint_performance_l1998_199898


namespace NUMINAMATH_CALUDE_train_stop_time_l1998_199805

/-- Proves that a train with given speeds stops for 10 minutes per hour -/
theorem train_stop_time (speed_without_stops speed_with_stops : ℝ) : 
  speed_without_stops = 48 → 
  speed_with_stops = 40 → 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 10 := by
  sorry

#check train_stop_time

end NUMINAMATH_CALUDE_train_stop_time_l1998_199805


namespace NUMINAMATH_CALUDE_remainder_problem_l1998_199809

theorem remainder_problem : 123456789012 % 240 = 132 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1998_199809


namespace NUMINAMATH_CALUDE_sequence_difference_l1998_199888

theorem sequence_difference (p q : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) : 
  (∀ n, S n = n^2 - 5*n) → 
  (∀ n, a (n+1) = S (n+1) - S n) →
  p - q = 4 →
  a p - a q = 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_l1998_199888


namespace NUMINAMATH_CALUDE_range_of_m_l1998_199886

/-- The range of m satisfying the given conditions -/
def M : Set ℝ := { m | ∀ x ∈ Set.Icc 0 1, 2 * m - 1 < x * (m^2 - 1) }

/-- Theorem stating that M is equal to the open interval (-∞, 0) -/
theorem range_of_m : M = Set.Ioi 0 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1998_199886


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1998_199831

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) :
  (∃ r : ℝ, 180 * r = a ∧ a * r = 81 / 32) → a = 135 / 19 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1998_199831


namespace NUMINAMATH_CALUDE_exists_expression_for_100_l1998_199872

/-- A type representing arithmetic expressions using only the number 7 --/
inductive Expr
  | seven : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an expression to a rational number --/
def eval : Expr → ℚ
  | Expr.seven => 7
  | Expr.add e₁ e₂ => eval e₁ + eval e₂
  | Expr.sub e₁ e₂ => eval e₁ - eval e₂
  | Expr.mul e₁ e₂ => eval e₁ * eval e₂
  | Expr.div e₁ e₂ => eval e₁ / eval e₂

/-- Count the number of sevens in an expression --/
def countSevens : Expr → ℕ
  | Expr.seven => 1
  | Expr.add e₁ e₂ => countSevens e₁ + countSevens e₂
  | Expr.sub e₁ e₂ => countSevens e₁ + countSevens e₂
  | Expr.mul e₁ e₂ => countSevens e₁ + countSevens e₂
  | Expr.div e₁ e₂ => countSevens e₁ + countSevens e₂

/-- There exists an expression using fewer than 10 sevens that evaluates to 100 --/
theorem exists_expression_for_100 : ∃ e : Expr, eval e = 100 ∧ countSevens e < 10 := by
  sorry

end NUMINAMATH_CALUDE_exists_expression_for_100_l1998_199872


namespace NUMINAMATH_CALUDE_cards_given_to_jeff_l1998_199890

theorem cards_given_to_jeff (initial_cards : ℕ) (cards_to_john : ℕ) (cards_left : ℕ) :
  initial_cards = 573 →
  cards_to_john = 195 →
  cards_left = 210 →
  initial_cards - cards_to_john - cards_left = 168 :=
by sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_l1998_199890


namespace NUMINAMATH_CALUDE_gcd_153_119_l1998_199848

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_153_119_l1998_199848


namespace NUMINAMATH_CALUDE_choir_members_count_l1998_199815

theorem choir_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 300 ∧ 
  n % 6 = 1 ∧ 
  n % 8 = 3 ∧ 
  n % 9 = 5 ∧ 
  n = 193 := by sorry

end NUMINAMATH_CALUDE_choir_members_count_l1998_199815


namespace NUMINAMATH_CALUDE_airplane_cost_l1998_199837

def initial_amount : ℚ := 5.00
def change_received : ℚ := 0.72

theorem airplane_cost : initial_amount - change_received = 4.28 := by
  sorry

end NUMINAMATH_CALUDE_airplane_cost_l1998_199837


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l1998_199874

theorem cubic_sum_over_product (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a + b + c + d = 0) : 
  (a^3 + b^3 + c^3 + d^3) / (a * b * c * d) = -3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l1998_199874


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l1998_199832

/-- The area of a circle with diameter 10 meters is 25π square meters -/
theorem circle_area_with_diameter_10 :
  ∀ (A : ℝ) (π : ℝ), 
  (∃ (d : ℝ), d = 10 ∧ A = (π * d^2) / 4) →
  A = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l1998_199832


namespace NUMINAMATH_CALUDE_compound_nitrogen_percentage_l1998_199880

/-- Mass percentage of nitrogen in a compound -/
def mass_percentage_N : ℝ := 26.42

/-- Theorem stating the mass percentage of nitrogen in the compound -/
theorem compound_nitrogen_percentage : mass_percentage_N = 26.42 := by
  sorry

end NUMINAMATH_CALUDE_compound_nitrogen_percentage_l1998_199880


namespace NUMINAMATH_CALUDE_honey_percentage_l1998_199859

theorem honey_percentage (initial_honey : ℝ) (final_honey : ℝ) (repetitions : ℕ) 
  (h_initial : initial_honey = 1250)
  (h_final : final_honey = 512)
  (h_repetitions : repetitions = 4) :
  ∃ (percentage : ℝ), 
    percentage = 0.2 ∧ 
    final_honey = initial_honey * (1 - percentage) ^ repetitions :=
by sorry

end NUMINAMATH_CALUDE_honey_percentage_l1998_199859


namespace NUMINAMATH_CALUDE_c_range_l1998_199814

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def solution_set_is_real (f : ℝ → ℝ) : Prop :=
  ∀ x, f x > 1

theorem c_range (c : ℝ) (hc : c > 0) :
  let p := is_increasing (λ x => Real.log ((1 - c) * x - 1) / Real.log 10)
  let q := solution_set_is_real (λ x => x + |x - 2 * c|)
  (p ∨ q) ∧ ¬(p ∧ q) →
  c ∈ Set.Ioo 0 (1/2) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_c_range_l1998_199814


namespace NUMINAMATH_CALUDE_distance_difference_l1998_199804

-- Define the distances
def mart_to_home : ℕ := 800
def home_to_academy : ℕ := 1300  -- 1 km + 300 m = 1000 m + 300 m = 1300 m
def academy_to_restaurant : ℕ := 1700

-- Theorem to prove
theorem distance_difference :
  (mart_to_home + home_to_academy) - academy_to_restaurant = 400 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l1998_199804


namespace NUMINAMATH_CALUDE_binomial_9_choose_5_l1998_199871

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by sorry

end NUMINAMATH_CALUDE_binomial_9_choose_5_l1998_199871


namespace NUMINAMATH_CALUDE_quadratic_cubic_relation_l1998_199878

theorem quadratic_cubic_relation (x₀ : ℝ) (h : x₀^2 + x₀ - 1 = 0) :
  x₀^3 + 2*x₀^2 + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_cubic_relation_l1998_199878


namespace NUMINAMATH_CALUDE_finite_solutions_l1998_199894

/-- The function F_{n,k}(x,y) as defined in the problem -/
def F (n k x y : ℕ) : ℤ := (Nat.factorial x : ℤ) + n^k + n + 1 - y^k

/-- Theorem stating that the set of solutions is finite -/
theorem finite_solutions (n k : ℕ) (hn : n > 0) (hk : k > 1) :
  Set.Finite {p : ℕ × ℕ | F n k p.1 p.2 = 0 ∧ p.1 > 0 ∧ p.2 > 0} :=
sorry

end NUMINAMATH_CALUDE_finite_solutions_l1998_199894


namespace NUMINAMATH_CALUDE_raspberry_harvest_l1998_199877

/-- Calculates the expected raspberry harvest for a rectangular garden. -/
theorem raspberry_harvest 
  (length width : ℕ) 
  (plants_per_sqft : ℕ) 
  (raspberries_per_plant : ℕ) : 
  length = 10 → 
  width = 7 → 
  plants_per_sqft = 5 → 
  raspberries_per_plant = 12 → 
  length * width * plants_per_sqft * raspberries_per_plant = 4200 := by
  sorry

#check raspberry_harvest

end NUMINAMATH_CALUDE_raspberry_harvest_l1998_199877


namespace NUMINAMATH_CALUDE_staff_discount_price_l1998_199817

/-- Given a dress with original price d, after a 35% discount and an additional 30% staff discount,
    the final price is 0.455 times the original price. -/
theorem staff_discount_price (d : ℝ) : d * (1 - 0.35) * (1 - 0.30) = d * 0.455 := by
  sorry

#check staff_discount_price

end NUMINAMATH_CALUDE_staff_discount_price_l1998_199817


namespace NUMINAMATH_CALUDE_peter_green_notebooks_l1998_199820

/-- Represents the number of green notebooks Peter bought -/
def green_notebooks (total notebooks : ℕ) (black_notebooks pink_notebooks : ℕ) 
  (total_cost black_cost pink_cost : ℕ) : ℕ :=
  total - black_notebooks - pink_notebooks

/-- Theorem stating that Peter bought 2 green notebooks -/
theorem peter_green_notebooks : 
  green_notebooks 4 1 1 45 15 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_peter_green_notebooks_l1998_199820


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l1998_199824

theorem polynomial_multiplication (x : ℝ) :
  (2 + 3 * x^3) * (1 - 2 * x^2 + x^4) = 2 - 4 * x^2 + 3 * x^3 + 2 * x^4 - 6 * x^5 + 3 * x^7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l1998_199824


namespace NUMINAMATH_CALUDE_tony_between_paul_and_rochelle_l1998_199884

-- Define the set of people
inductive Person : Type
  | Paul : Person
  | Quincy : Person
  | Rochelle : Person
  | Surinder : Person
  | Tony : Person

-- Define the seating arrangement as a function from Person to ℕ
def SeatingArrangement := Person → ℕ

-- Define the conditions of the seating arrangement
def ValidSeatingArrangement (s : SeatingArrangement) : Prop :=
  -- Condition 1: All seats are distinct
  (∀ p q : Person, p ≠ q → s p ≠ s q) ∧
  -- Condition 2: Seats are consecutive around a circular table
  (∀ p : Person, s p < 5) ∧
  -- Condition 3: Quincy sits between Paul and Surinder
  ((s Person.Quincy = (s Person.Paul + 1) % 5 ∧ s Person.Quincy = (s Person.Surinder + 4) % 5) ∨
   (s Person.Quincy = (s Person.Paul + 4) % 5 ∧ s Person.Quincy = (s Person.Surinder + 1) % 5)) ∧
  -- Condition 4: Tony is not beside Surinder
  (s Person.Tony ≠ (s Person.Surinder + 1) % 5 ∧ s Person.Tony ≠ (s Person.Surinder + 4) % 5)

-- Theorem: In any valid seating arrangement, Paul and Rochelle must be sitting on either side of Tony
theorem tony_between_paul_and_rochelle (s : SeatingArrangement) 
  (h : ValidSeatingArrangement s) : 
  (s Person.Tony = (s Person.Paul + 1) % 5 ∧ s Person.Tony = (s Person.Rochelle + 4) % 5) ∨
  (s Person.Tony = (s Person.Paul + 4) % 5 ∧ s Person.Tony = (s Person.Rochelle + 1) % 5) :=
sorry

end NUMINAMATH_CALUDE_tony_between_paul_and_rochelle_l1998_199884


namespace NUMINAMATH_CALUDE_number_of_ways_to_choose_cards_l1998_199838

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Number of cards per suit in a standard deck -/
def CardsPerSuit : ℕ := 13

/-- Number of cards to be chosen -/
def CardsToChoose : ℕ := 4

/-- Number of cards to be chosen from one suit -/
def CardsFromOneSuit : ℕ := 2

/-- Calculate the number of ways to choose cards according to the problem conditions -/
def calculateWays : ℕ :=
  Nat.choose NumberOfSuits 3 *  -- Choose 3 suits from 4
  3 *  -- Choose which of the 3 suits will have 2 cards
  Nat.choose CardsPerSuit 2 *  -- Choose 2 cards from the chosen suit
  CardsPerSuit * CardsPerSuit  -- Choose 1 card each from the other two suits

/-- Theorem stating that the number of ways to choose cards is 158184 -/
theorem number_of_ways_to_choose_cards :
  calculateWays = 158184 := by sorry

end NUMINAMATH_CALUDE_number_of_ways_to_choose_cards_l1998_199838


namespace NUMINAMATH_CALUDE_exactly_two_out_of_three_l1998_199842

def probability_single_shot : ℚ := 2/3

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem exactly_two_out_of_three :
  binomial_probability 3 2 probability_single_shot = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_out_of_three_l1998_199842


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1998_199830

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1998_199830


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1998_199828

/-- The radius of the inscribed circle of a triangle with side lengths 8, 10, and 12 is √7 -/
theorem inscribed_circle_radius (a b c : ℝ) (h_a : a = 8) (h_b : b = 10) (h_c : c = 12) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1998_199828


namespace NUMINAMATH_CALUDE_equation_solutions_l1998_199839

/-- The equation we're solving -/
def equation (x : ℝ) : Prop :=
  (15*x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 54

/-- The set of solutions to the equation -/
def solutions : Set ℝ := {0, -1, -3, -3.5}

/-- Theorem stating that the solutions are correct -/
theorem equation_solutions :
  ∀ x : ℝ, x ∈ solutions ↔ equation x :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1998_199839


namespace NUMINAMATH_CALUDE_min_value_theorem_l1998_199821

theorem min_value_theorem (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0)
  (h_sol : ∀ x, -x^2 + 6*a*x - 3*a^2 ≥ 0 ↔ x₁ ≤ x ∧ x ≤ x₂) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 6 ∧ 
    ∀ y₁ y₂, (∀ x, -x^2 + 6*a*x - 3*a^2 ≥ 0 ↔ y₁ ≤ x ∧ x ≤ y₂) → 
      y₁ + y₂ + 3*a / (y₁ * y₂) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1998_199821


namespace NUMINAMATH_CALUDE_highway_distance_theorem_l1998_199860

/-- The distance between two points A and B on a highway -/
def distance_AB : ℝ := 198

/-- The speed of vehicles traveling from A to B -/
def speed_AB : ℝ := 50

/-- The speed of vehicles traveling from B to A -/
def speed_BA : ℝ := 60

/-- The distance from point B where car X breaks down -/
def breakdown_distance : ℝ := 30

/-- The delay in the second meeting due to the breakdown -/
def delay_time : ℝ := 1.2

theorem highway_distance_theorem :
  distance_AB = 198 :=
sorry

end NUMINAMATH_CALUDE_highway_distance_theorem_l1998_199860


namespace NUMINAMATH_CALUDE_job_completion_time_l1998_199867

/-- The time it takes for Annie to complete the job alone -/
def annie_time : ℝ := 9

/-- The time the person works before stopping -/
def person_partial_time : ℝ := 4

/-- The time it takes Annie to complete the remaining work after the person stops -/
def annie_completion_time : ℝ := 6

/-- The time it takes for the person to complete the job alone -/
def person_total_time : ℝ := 12

theorem job_completion_time :
  (person_partial_time / person_total_time) + (annie_completion_time / annie_time) = 1 :=
sorry

end NUMINAMATH_CALUDE_job_completion_time_l1998_199867


namespace NUMINAMATH_CALUDE_light_flash_duration_l1998_199857

theorem light_flash_duration (flash_interval : ℕ) (num_flashes : ℕ) (seconds_per_hour : ℕ) : 
  flash_interval = 12 →
  num_flashes = 300 →
  seconds_per_hour = 3600 →
  (flash_interval * num_flashes) / seconds_per_hour = 1 := by
  sorry

end NUMINAMATH_CALUDE_light_flash_duration_l1998_199857


namespace NUMINAMATH_CALUDE_natalie_portion_ratio_l1998_199835

def total_amount : ℝ := 10000

def third_person_amount : ℝ := 2000

def second_person_percentage : ℝ := 0.6

theorem natalie_portion_ratio (first_person_amount : ℝ) 
  (h1 : third_person_amount = total_amount - first_person_amount - second_person_percentage * (total_amount - first_person_amount))
  (h2 : first_person_amount > 0)
  (h3 : first_person_amount < total_amount) :
  first_person_amount / total_amount = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_natalie_portion_ratio_l1998_199835


namespace NUMINAMATH_CALUDE_positive_real_inequality_l1998_199892

theorem positive_real_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x + y + z = 1/x + 1/y + 1/z) :
  x + y + z ≥ Real.sqrt ((x*y + 1)/2) + Real.sqrt ((y*z + 1)/2) + Real.sqrt ((z*x + 1)/2) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l1998_199892


namespace NUMINAMATH_CALUDE_sundae_probability_l1998_199836

def ice_cream_flavors : ℕ := 3
def syrup_types : ℕ := 2
def topping_options : ℕ := 3

def total_combinations : ℕ := ice_cream_flavors * syrup_types * topping_options

def specific_combination : ℕ := 1

theorem sundae_probability :
  (specific_combination : ℚ) / total_combinations = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_sundae_probability_l1998_199836


namespace NUMINAMATH_CALUDE_intersection_midpoint_l1998_199850

/-- The midpoint of the line segment connecting the intersection points of y = x and y^2 = 4x is (2,2) -/
theorem intersection_midpoint :
  let line := {(x, y) : ℝ × ℝ | y = x}
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4*x}
  let intersection := line ∩ parabola
  ∃ (a b : ℝ × ℝ), a ∈ intersection ∧ b ∈ intersection ∧ a ≠ b ∧
    (a.1 + b.1) / 2 = 2 ∧ (a.2 + b.2) / 2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_midpoint_l1998_199850


namespace NUMINAMATH_CALUDE_bleacher_exercise_calories_l1998_199811

/-- Given the number of round trips, stairs one way, and total calories burned,
    calculate the number of calories burned per stair. -/
def calories_per_stair (round_trips : ℕ) (stairs_one_way : ℕ) (total_calories : ℕ) : ℚ :=
  total_calories / (2 * round_trips * stairs_one_way)

/-- Theorem stating that under the given conditions, each stair burns 2 calories. -/
theorem bleacher_exercise_calories :
  calories_per_stair 40 32 5120 = 2 := by
  sorry

end NUMINAMATH_CALUDE_bleacher_exercise_calories_l1998_199811


namespace NUMINAMATH_CALUDE_benzene_required_l1998_199891

-- Define the chemical reaction
structure ChemicalReaction where
  benzene : ℕ
  methane : ℕ
  toluene : ℕ
  hydrogen : ℕ

-- Define the balanced equation
def balanced_equation : ChemicalReaction :=
  { benzene := 1, methane := 1, toluene := 1, hydrogen := 1 }

-- Define the given amounts
def given_amounts : ChemicalReaction :=
  { benzene := 0, methane := 2, toluene := 2, hydrogen := 2 }

-- Theorem to prove
theorem benzene_required (r : ChemicalReaction) :
  r.methane = 2 * balanced_equation.methane ∧
  r.toluene = 2 * balanced_equation.toluene ∧
  r.hydrogen = 2 * balanced_equation.hydrogen →
  r.benzene = 2 * balanced_equation.benzene :=
by sorry

end NUMINAMATH_CALUDE_benzene_required_l1998_199891


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l1998_199881

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l1998_199881


namespace NUMINAMATH_CALUDE_chord_length_theorem_l1998_199800

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if two circles are internally tangent -/
def internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem chord_length_theorem (c1 c2 c3 : Circle) 
  (h1 : c1.radius = 6)
  (h2 : c2.radius = 8)
  (h3 : externally_tangent c1 c3)
  (h4 : internally_tangent c2 c3)
  (h5 : collinear c1.center c2.center c3.center) :
  ∃ (chord_length : ℝ), chord_length = 8 * Real.sqrt (2 * c3.radius - 8) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l1998_199800


namespace NUMINAMATH_CALUDE_red_shirt_pairs_l1998_199882

theorem red_shirt_pairs (total_students : ℕ) (blue_students : ℕ) (red_students : ℕ) 
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) 
  (h1 : total_students = 144)
  (h2 : blue_students = 63)
  (h3 : red_students = 81)
  (h4 : total_pairs = 72)
  (h5 : blue_blue_pairs = 29)
  (h6 : total_students = blue_students + red_students)
  (h7 : total_pairs * 2 = total_students) :
  (red_students - (total_students - blue_blue_pairs * 2 - blue_students)) / 2 = 38 := by
  sorry

end NUMINAMATH_CALUDE_red_shirt_pairs_l1998_199882


namespace NUMINAMATH_CALUDE_friday_return_count_l1998_199847

/-- The number of books returned on Friday -/
def books_returned_friday (initial_books : ℕ) (wed_checkout : ℕ) (thur_return : ℕ) (thur_checkout : ℕ) (final_books : ℕ) : ℕ :=
  final_books - (initial_books - wed_checkout + thur_return - thur_checkout)

/-- Proof that 7 books were returned on Friday given the conditions -/
theorem friday_return_count :
  books_returned_friday 98 43 23 5 80 = 7 := by
  sorry

#eval books_returned_friday 98 43 23 5 80

end NUMINAMATH_CALUDE_friday_return_count_l1998_199847


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l1998_199819

theorem imaginary_part_of_complex_number : 
  let z : ℂ := 1 / (2 + Complex.I)^2
  Complex.im z = -4/25 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l1998_199819


namespace NUMINAMATH_CALUDE_age_divisibility_l1998_199887

theorem age_divisibility (a : ℤ) : 10 ∣ (a^5 - a) := by
  sorry

end NUMINAMATH_CALUDE_age_divisibility_l1998_199887


namespace NUMINAMATH_CALUDE_correct_change_l1998_199851

/-- The change Bomi should receive after buying candy and chocolate -/
def bomi_change (candy_cost chocolate_cost paid : ℕ) : ℕ :=
  paid - (candy_cost + chocolate_cost)

/-- Theorem stating the correct change Bomi should receive -/
theorem correct_change : bomi_change 350 500 1000 = 150 := by
  sorry

end NUMINAMATH_CALUDE_correct_change_l1998_199851


namespace NUMINAMATH_CALUDE_pyramid_perpendicular_feet_circle_l1998_199834

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a pyramid -/
structure Pyramid where
  apex : Point3D
  base : List Point3D
  altitude_foot : Point3D

/-- Represents a circle -/
structure Circle where
  center : Point3D
  radius : ℝ

/-- Check if a list of points lies on a circle -/
def points_on_circle (points : List Point3D) (circle : Circle) : Prop := sorry

/-- Get the feet of perpendiculars from a point to the edges of a pyramid -/
def get_perpendicular_feet (pyramid : Pyramid) : List Point3D := sorry

/-- The main theorem -/
theorem pyramid_perpendicular_feet_circle (pyramid : Pyramid) (base_circle : Circle) :
  points_on_circle pyramid.base base_circle →
  ∃ (feet_circle : Circle), points_on_circle (get_perpendicular_feet pyramid) feet_circle :=
sorry

end NUMINAMATH_CALUDE_pyramid_perpendicular_feet_circle_l1998_199834


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1998_199841

/-- Given an arithmetic sequence with first term 2 and sum of second and fourth terms 10,
    the third term is 5. -/
theorem arithmetic_sequence_third_term (a d : ℚ) : 
  a = 2 ∧ (a + d) + (a + 3*d) = 10 → a + 2*d = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1998_199841


namespace NUMINAMATH_CALUDE_smallest_difference_ef_de_l1998_199844

/-- Represents a triangle with integer side lengths --/
structure Triangle where
  de : ℕ
  ef : ℕ
  fd : ℕ

/-- Checks if the given lengths satisfy the triangle inequality --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.de + t.ef > t.fd ∧ t.ef + t.fd > t.de ∧ t.fd + t.de > t.ef

/-- Theorem stating the smallest possible difference between EF and DE --/
theorem smallest_difference_ef_de (t : Triangle) : 
  t.de < t.ef ∧ t.ef ≤ t.fd ∧ 
  t.de + t.ef + t.fd = 1024 ∧
  is_valid_triangle t →
  ∀ (t' : Triangle), 
    t'.de < t'.ef ∧ t'.ef ≤ t'.fd ∧
    t'.de + t'.ef + t'.fd = 1024 ∧
    is_valid_triangle t' →
    t.ef - t.de ≤ t'.ef - t'.de ∧
    t.ef - t.de = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_ef_de_l1998_199844


namespace NUMINAMATH_CALUDE_profit_equals_cost_of_three_toys_l1998_199803

/-- Proves that the number of toys whose cost price equals the profit is 3 -/
theorem profit_equals_cost_of_three_toys 
  (total_toys : ℕ) 
  (selling_price : ℕ) 
  (cost_per_toy : ℕ) 
  (h1 : total_toys = 18)
  (h2 : selling_price = 25200)
  (h3 : cost_per_toy = 1200) :
  (selling_price - total_toys * cost_per_toy) / cost_per_toy = 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_equals_cost_of_three_toys_l1998_199803


namespace NUMINAMATH_CALUDE_abc_product_l1998_199876

theorem abc_product (a b c : ℤ) 
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 30)
  (h3 : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 504 / (a * b * c) = 1) :
  a * b * c = 1176 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l1998_199876


namespace NUMINAMATH_CALUDE_square_difference_601_597_l1998_199826

theorem square_difference_601_597 : 601^2 - 597^2 = 4792 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_601_597_l1998_199826


namespace NUMINAMATH_CALUDE_baba_yaga_hut_inhabitants_l1998_199843

/-- The number of inhabitants in Baba Yaga's hut -/
def num_inhabitants : ℕ := 3

/-- The number of Talking Cats -/
def num_cats : ℕ := 1

/-- The number of Wise Owls -/
def num_owls : ℕ := 1

/-- The number of Mustached Cockroaches -/
def num_cockroaches : ℕ := 1

/-- The total number of non-Talking Cats -/
def non_cats : ℕ := 2

/-- The total number of non-Wise Owls -/
def non_owls : ℕ := 2

theorem baba_yaga_hut_inhabitants :
  num_inhabitants = num_cats + num_owls + num_cockroaches ∧
  non_cats = num_owls + num_cockroaches ∧
  non_owls = num_cats + num_cockroaches ∧
  num_inhabitants = 3 := by
  sorry

end NUMINAMATH_CALUDE_baba_yaga_hut_inhabitants_l1998_199843


namespace NUMINAMATH_CALUDE_complete_square_equivalence_l1998_199801

theorem complete_square_equivalence :
  ∀ x y : ℝ, y = -x^2 + 2*x + 3 ↔ y = -(x - 1)^2 + 4 := by sorry

end NUMINAMATH_CALUDE_complete_square_equivalence_l1998_199801


namespace NUMINAMATH_CALUDE_min_selling_price_A_l1998_199885

/-- Represents the water purifier problem with given conditions -/
structure WaterPurifierProblem where
  total_units : ℕ
  price_A : ℕ
  price_B : ℕ
  total_cost : ℕ
  units_A : ℕ
  units_B : ℕ
  min_total_profit : ℕ

/-- The specific instance of the water purifier problem -/
def problem : WaterPurifierProblem := {
  total_units := 160,
  price_A := 150,
  price_B := 350,
  total_cost := 36000,
  units_A := 100,
  units_B := 60,
  min_total_profit := 11000
}

/-- Theorem stating the minimum selling price for model A -/
theorem min_selling_price_A (p : WaterPurifierProblem) : 
  p.total_units = p.units_A + p.units_B →
  p.total_cost = p.price_A * p.units_A + p.price_B * p.units_B →
  ∀ selling_price_A : ℕ, 
    (selling_price_A - p.price_A) * p.units_A + 
    (2 * (selling_price_A - p.price_A)) * p.units_B ≥ p.min_total_profit →
    selling_price_A ≥ 200 := by
  sorry

#check min_selling_price_A problem

end NUMINAMATH_CALUDE_min_selling_price_A_l1998_199885


namespace NUMINAMATH_CALUDE_cubes_in_figure_100_l1998_199852

/-- Represents the number of cubes in a figure at position n -/
def num_cubes (n : ℕ) : ℕ := 2 * n^3 + n^2 + 3 * n + 1

/-- The sequence of cubes follows the given pattern for the first four figures -/
axiom pattern_holds : num_cubes 0 = 1 ∧ num_cubes 1 = 7 ∧ num_cubes 2 = 25 ∧ num_cubes 3 = 63

/-- The number of cubes in figure 100 is 2010301 -/
theorem cubes_in_figure_100 : num_cubes 100 = 2010301 := by
  sorry

end NUMINAMATH_CALUDE_cubes_in_figure_100_l1998_199852


namespace NUMINAMATH_CALUDE_tangent_line_slope_l1998_199833

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - 3*x - 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 3

-- Theorem statement
theorem tangent_line_slope (k : ℝ) :
  (∃ x₀ : ℝ, f x₀ = k * x₀ + 2 ∧ f' x₀ = k) → k = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l1998_199833


namespace NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l1998_199869

theorem tan_ratio_from_sin_sum_diff (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5/8) 
  (h2 : Real.sin (a - b) = 1/4) : 
  Real.tan a / Real.tan b = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l1998_199869


namespace NUMINAMATH_CALUDE_a_range_l1998_199865

theorem a_range (a b c : ℝ) 
  (sum_eq : a + b + c = 2) 
  (sum_sq_eq : a^2 + b^2 + c^2 = 4) 
  (order : a > b ∧ b > c) : 
  2/3 < a ∧ a < 2 := by
sorry

end NUMINAMATH_CALUDE_a_range_l1998_199865


namespace NUMINAMATH_CALUDE_parabola_intersection_dot_product_l1998_199895

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line of the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

def Parabola.contains (c : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * c.p * pt.x

def Line.contains (l : Line) (pt : Point) : Prop :=
  pt.y = l.m * pt.x + l.b

def dotProduct (a b : Point) : ℝ :=
  a.x * b.x + a.y * b.y

theorem parabola_intersection_dot_product 
  (c : Parabola)
  (l : Line)
  (h1 : c.contains ⟨2, -2⟩)
  (h2 : l.m = 1 ∧ l.b = -1)
  (A B : Point)
  (h3 : c.contains A ∧ l.contains A)
  (h4 : c.contains B ∧ l.contains B)
  (h5 : A ≠ B) :
  dotProduct A B = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_dot_product_l1998_199895


namespace NUMINAMATH_CALUDE_additional_cars_needed_min_additional_cars_l1998_199845

def current_cars : ℕ := 35
def cars_per_row : ℕ := 8

theorem additional_cars_needed : 
  ∃ (n : ℕ), n > 0 ∧ (current_cars + n) % cars_per_row = 0 ∧
  ∀ (m : ℕ), m < n → (current_cars + m) % cars_per_row ≠ 0 := by
  sorry

theorem min_additional_cars : 
  ∃ (n : ℕ), n = 5 ∧ (current_cars + n) % cars_per_row = 0 ∧
  ∀ (m : ℕ), m < n → (current_cars + m) % cars_per_row ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_additional_cars_needed_min_additional_cars_l1998_199845


namespace NUMINAMATH_CALUDE_john_text_messages_l1998_199854

theorem john_text_messages 
  (total_messages_per_day : ℕ) 
  (unintended_messages_per_week : ℕ) 
  (days_per_week : ℕ) 
  (h1 : total_messages_per_day = 55) 
  (h2 : unintended_messages_per_week = 245) 
  (h3 : days_per_week = 7) : 
  total_messages_per_day - (unintended_messages_per_week / days_per_week) = 20 := by
sorry

end NUMINAMATH_CALUDE_john_text_messages_l1998_199854


namespace NUMINAMATH_CALUDE_aron_cleaning_time_l1998_199861

/-- Represents the cleaning schedule and calculates total cleaning time -/
def cleaning_schedule (vacuum_time : ℕ) (vacuum_days : ℕ) (dust_time : ℕ) (dust_days : ℕ) : ℕ :=
  vacuum_time * vacuum_days + dust_time * dust_days

/-- Theorem stating that Aron's total cleaning time per week is 130 minutes -/
theorem aron_cleaning_time : 
  cleaning_schedule 30 3 20 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_aron_cleaning_time_l1998_199861


namespace NUMINAMATH_CALUDE_pairwise_ratio_sum_bound_l1998_199883

theorem pairwise_ratio_sum_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_pairwise_ratio_sum_bound_l1998_199883


namespace NUMINAMATH_CALUDE_writing_outlining_difference_l1998_199840

/-- Represents the time spent on different activities for a speech --/
structure SpeechTime where
  outlining : ℕ
  writing : ℕ
  practicing : ℕ

/-- Defines the conditions for Javier's speech preparation --/
def javierSpeechConditions (t : SpeechTime) : Prop :=
  t.outlining = 30 ∧
  t.writing > t.outlining ∧
  t.practicing = t.writing / 2 ∧
  t.outlining + t.writing + t.practicing = 117

/-- Theorem stating the difference between writing and outlining time --/
theorem writing_outlining_difference (t : SpeechTime) 
  (h : javierSpeechConditions t) : t.writing - t.outlining = 28 := by
  sorry

#check writing_outlining_difference

end NUMINAMATH_CALUDE_writing_outlining_difference_l1998_199840


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_cos_pi_minus_2alpha_l1998_199893

theorem sin_2alpha_minus_cos_pi_minus_2alpha (α : Real) (h : Real.tan α = 2/3) :
  Real.sin (2 * α) - Real.cos (π - 2 * α) = 17/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_cos_pi_minus_2alpha_l1998_199893


namespace NUMINAMATH_CALUDE_salt_solution_concentration_l1998_199802

/-- Proves that the concentration of the salt solution is 50% given the specified conditions. -/
theorem salt_solution_concentration
  (water_volume : Real)
  (salt_solution_volume : Real)
  (total_volume : Real)
  (mixture_concentration : Real)
  (h1 : water_volume = 1)
  (h2 : salt_solution_volume = 0.25)
  (h3 : total_volume = water_volume + salt_solution_volume)
  (h4 : mixture_concentration = 0.1)
  (h5 : salt_solution_volume * (concentration / 100) = total_volume * mixture_concentration) :
  concentration = 50 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_concentration_l1998_199802


namespace NUMINAMATH_CALUDE_situp_ratio_l1998_199864

theorem situp_ratio (ken_situps : ℕ) (nathan_ratio : ℚ) (bob_situps : ℕ) :
  ken_situps = 20 →
  bob_situps = (ken_situps + nathan_ratio * ken_situps) / 2 →
  bob_situps = ken_situps + 10 →
  nathan_ratio = 2 :=
by sorry

end NUMINAMATH_CALUDE_situp_ratio_l1998_199864


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1998_199875

/-- Given a line passing through points (1,3) and (3,11), 
    prove that the sum of its slope and y-intercept equals 3. -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
  (3 : ℝ) = m * (1 : ℝ) + b → 
  (11 : ℝ) = m * (3 : ℝ) + b → 
  m + b = 3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1998_199875


namespace NUMINAMATH_CALUDE_inequality_proof_l1998_199818

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) :
  a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1998_199818


namespace NUMINAMATH_CALUDE_sum_powers_l1998_199808

theorem sum_powers (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  ¬(∀ (a b c d : ℝ), (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) := by
sorry

end NUMINAMATH_CALUDE_sum_powers_l1998_199808


namespace NUMINAMATH_CALUDE_abs_plus_one_nonzero_l1998_199825

theorem abs_plus_one_nonzero (a : ℚ) : |a| + 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_plus_one_nonzero_l1998_199825


namespace NUMINAMATH_CALUDE_min_guaranteed_meeting_distance_l1998_199858

/-- Represents the state of a player on the train -/
structure PlayerState :=
  (position : Real)
  (facing_forward : Bool)
  (at_front : Bool)
  (at_end : Bool)

/-- Represents the game state -/
structure GameState :=
  (alice : PlayerState)
  (bob : PlayerState)
  (total_distance : Real)

/-- Defines the train length -/
def train_length : Real := 1

/-- Theorem stating the minimum guaranteed meeting distance -/
theorem min_guaranteed_meeting_distance :
  ∀ (initial_state : GameState),
  ∃ (strategy : GameState → GameState),
  ∀ (final_state : GameState),
  (final_state.alice.position = final_state.bob.position) →
  (final_state.total_distance ≤ 1.5) :=
sorry

end NUMINAMATH_CALUDE_min_guaranteed_meeting_distance_l1998_199858


namespace NUMINAMATH_CALUDE_function_value_sum_l1998_199863

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem function_value_sum (f : ℝ → ℝ) 
    (h_periodic : is_periodic f 2)
    (h_odd : is_odd f)
    (h_interval : ∀ x, 0 < x → x < 1 → f x = 4^x) :
  f (-5/2) + f 2 = -2 := by
  sorry


end NUMINAMATH_CALUDE_function_value_sum_l1998_199863


namespace NUMINAMATH_CALUDE_inequality_of_positive_numbers_l1998_199807

theorem inequality_of_positive_numbers (a₁ a₂ a₃ : ℝ) (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) :
  (a₁ * a₂) / a₃ + (a₂ * a₃) / a₁ + (a₃ * a₁) / a₂ ≥ a₁ + a₂ + a₃ := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_numbers_l1998_199807


namespace NUMINAMATH_CALUDE_negation_equivalence_l1998_199889

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1998_199889


namespace NUMINAMATH_CALUDE_sphere_volume_right_triangular_pyramid_l1998_199862

/-- The volume of a sphere circumscribing a right triangular pyramid with specific edge lengths -/
theorem sphere_volume_right_triangular_pyramid :
  let edge1 : ℝ := Real.sqrt 3
  let edge2 : ℝ := 2
  let edge3 : ℝ := 3
  let sphere_volume := (4 / 3) * Real.pi * (edge1^2 + edge2^2 + edge3^2)^(3/2) / 8
  sphere_volume = 32 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_right_triangular_pyramid_l1998_199862


namespace NUMINAMATH_CALUDE_number_from_hcf_lcm_and_other_l1998_199812

theorem number_from_hcf_lcm_and_other (a b : ℕ+) : 
  Nat.gcd a b = 12 →
  Nat.lcm a b = 396 →
  b = 99 →
  a = 48 := by
sorry

end NUMINAMATH_CALUDE_number_from_hcf_lcm_and_other_l1998_199812


namespace NUMINAMATH_CALUDE_exist_unequal_triangles_with_equal_angles_and_two_sides_l1998_199849

-- Define two triangles
structure Triangle :=
  (a b c : ℝ)
  (α β γ : ℝ)

-- Define the conditions for our triangles
def triangles_satisfy_conditions (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ ∧
  ((t1.a = t2.a ∧ t1.b = t2.b) ∨ (t1.a = t2.a ∧ t1.c = t2.c) ∨ (t1.b = t2.b ∧ t1.c = t2.c))

-- Define triangle inequality
def triangles_not_congruent (t1 t2 : Triangle) : Prop :=
  t1.a ≠ t2.a ∨ t1.b ≠ t2.b ∨ t1.c ≠ t2.c

-- Theorem statement
theorem exist_unequal_triangles_with_equal_angles_and_two_sides :
  ∃ (t1 t2 : Triangle), triangles_satisfy_conditions t1 t2 ∧ triangles_not_congruent t1 t2 :=
sorry

end NUMINAMATH_CALUDE_exist_unequal_triangles_with_equal_angles_and_two_sides_l1998_199849


namespace NUMINAMATH_CALUDE_polynomial_range_open_interval_l1998_199806

theorem polynomial_range_open_interval : 
  (∀ k : ℝ, k > 0 → ∃ x y : ℝ, (1 - x * y)^2 + x^2 = k) ∧ 
  (∀ x y : ℝ, (1 - x * y)^2 + x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_range_open_interval_l1998_199806


namespace NUMINAMATH_CALUDE_bob_corn_harvest_l1998_199810

/-- Calculates the number of bushels of corn harvested given the number of rows,
    corn stalks per row, and corn stalks per bushel. -/
def corn_harvest (rows : ℕ) (stalks_per_row : ℕ) (stalks_per_bushel : ℕ) : ℕ :=
  (rows * stalks_per_row) / stalks_per_bushel

/-- Proves that Bob will harvest 50 bushels of corn given the specified conditions. -/
theorem bob_corn_harvest :
  corn_harvest 5 80 8 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bob_corn_harvest_l1998_199810
