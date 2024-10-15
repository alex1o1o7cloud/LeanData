import Mathlib

namespace NUMINAMATH_CALUDE_projectile_distance_l270_27014

theorem projectile_distance (v1 v2 t : ℝ) (h1 : v1 = 470) (h2 : v2 = 500) (h3 : t = 90 / 60) :
  v1 * t + v2 * t = 1455 :=
by sorry

end NUMINAMATH_CALUDE_projectile_distance_l270_27014


namespace NUMINAMATH_CALUDE_loan_amount_calculation_l270_27016

/-- Proves that the amount lent is 1000 given the specified conditions -/
theorem loan_amount_calculation (P : ℝ) : 
  (P * 0.115 * 3 - P * 0.10 * 3 = 45) → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_calculation_l270_27016


namespace NUMINAMATH_CALUDE_mans_swimming_speed_l270_27049

/-- Proves that a man's swimming speed in still water is 1.5 km/h given the conditions -/
theorem mans_swimming_speed 
  (stream_speed : ℝ) 
  (upstream_time downstream_time : ℝ) 
  (h1 : stream_speed = 0.5)
  (h2 : upstream_time = 2 * downstream_time) : 
  ∃ (still_water_speed : ℝ), still_water_speed = 1.5 :=
by
  sorry

#check mans_swimming_speed

end NUMINAMATH_CALUDE_mans_swimming_speed_l270_27049


namespace NUMINAMATH_CALUDE_ellipse_line_slope_l270_27047

/-- An ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e_eq : e = 1/2

/-- A line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  h_slope_pos : 0 < slope

/-- Radii of incircles of triangles formed by points on the ellipse and foci -/
structure IncircleRadii where
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  h_radii_rel : r₁ + r₃ = 2 * r₂

/-- The main theorem statement -/
theorem ellipse_line_slope (E : Ellipse) (l : Line) (R : IncircleRadii) :
  l.slope = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_line_slope_l270_27047


namespace NUMINAMATH_CALUDE_carla_smoothie_cream_l270_27078

/-- Given information about Carla's smoothie recipe, prove the amount of cream used. -/
theorem carla_smoothie_cream (watermelon_puree : ℕ) (num_servings : ℕ) (serving_size : ℕ) 
  (h1 : watermelon_puree = 500)
  (h2 : num_servings = 4)
  (h3 : serving_size = 150) :
  num_servings * serving_size - watermelon_puree = 100 := by
  sorry

#check carla_smoothie_cream

end NUMINAMATH_CALUDE_carla_smoothie_cream_l270_27078


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l270_27077

theorem sum_of_coefficients_zero 
  (a₀ a₁ a₂ a₃ a₄ : ℝ) : 
  (∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*(x + 1) + a₂*(x + 1)^2 + a₃*(x + 1)^3 + a₄*(x + 1)^4) → 
  a₁ + a₂ + a₃ + a₄ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l270_27077


namespace NUMINAMATH_CALUDE_push_ups_total_l270_27020

theorem push_ups_total (david_pushups : ℕ) (difference : ℕ) : 
  david_pushups = 51 → difference = 49 → 
  david_pushups + (david_pushups - difference) = 53 := by
  sorry

end NUMINAMATH_CALUDE_push_ups_total_l270_27020


namespace NUMINAMATH_CALUDE_delivery_time_problem_l270_27035

/-- Calculates the time needed to deliver all cars -/
def delivery_time (coal_cars iron_cars wood_cars : ℕ) 
                  (coal_deposit iron_deposit wood_deposit : ℕ) 
                  (time_between_stations : ℕ) : ℕ :=
  let coal_stations := (coal_cars + coal_deposit - 1) / coal_deposit
  let iron_stations := (iron_cars + iron_deposit - 1) / iron_deposit
  let wood_stations := (wood_cars + wood_deposit - 1) / wood_deposit
  let max_stations := max coal_stations (max iron_stations wood_stations)
  max_stations * time_between_stations

/-- Proves that the delivery time for the given problem is 100 minutes -/
theorem delivery_time_problem : 
  delivery_time 6 12 2 2 3 1 25 = 100 := by
  sorry

end NUMINAMATH_CALUDE_delivery_time_problem_l270_27035


namespace NUMINAMATH_CALUDE_min_roots_count_l270_27075

/-- A function satisfying the given symmetry conditions -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (2 - x) = f (2 + x)) ∧ (∀ x : ℝ, f (7 - x) = f (7 + x))

/-- The theorem stating the minimum number of roots -/
theorem min_roots_count
  (f : ℝ → ℝ)
  (h_symmetric : SymmetricFunction f)
  (h_root_zero : f 0 = 0) :
  (∃ (roots : Finset ℝ), 
    (∀ x ∈ roots, f x = 0 ∧ x ∈ Set.Icc (-1000) 1000) ∧
    (∀ roots' : Finset ℝ, (∀ x ∈ roots', f x = 0 ∧ x ∈ Set.Icc (-1000) 1000) → 
      roots'.card ≤ roots.card) ∧
    roots.card = 401) :=
  sorry

end NUMINAMATH_CALUDE_min_roots_count_l270_27075


namespace NUMINAMATH_CALUDE_investment_split_l270_27082

/-- Proves that given an initial investment of $1500 split between two banks with annual compound
    interest rates of 4% and 6% respectively, if the total amount after three years is $1755,
    then the initial investment in the bank with 4% interest rate is $476.5625. -/
theorem investment_split (x : ℝ) : 
  x ≥ 0 ∧ x ≤ 1500 ∧ 
  x * (1 + 0.04)^3 + (1500 - x) * (1 + 0.06)^3 = 1755 →
  x = 476.5625 := by sorry

end NUMINAMATH_CALUDE_investment_split_l270_27082


namespace NUMINAMATH_CALUDE_remainder_98_pow_50_mod_100_l270_27011

theorem remainder_98_pow_50_mod_100 : 98^50 % 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_pow_50_mod_100_l270_27011


namespace NUMINAMATH_CALUDE_baby_whales_count_l270_27025

/-- Represents the number of whales observed during Ishmael's monitoring --/
structure WhaleCount where
  first_trip_males : ℕ
  first_trip_females : ℕ
  third_trip_males : ℕ
  third_trip_females : ℕ
  total_whales : ℕ

/-- Theorem stating the number of baby whales observed on the second trip --/
theorem baby_whales_count (w : WhaleCount) 
  (h1 : w.first_trip_males = 28)
  (h2 : w.first_trip_females = 2 * w.first_trip_males)
  (h3 : w.third_trip_males = w.first_trip_males / 2)
  (h4 : w.third_trip_females = w.first_trip_females)
  (h5 : w.total_whales = 178) :
  w.total_whales - (w.first_trip_males + w.first_trip_females + w.third_trip_males + w.third_trip_females) = 24 := by
  sorry

end NUMINAMATH_CALUDE_baby_whales_count_l270_27025


namespace NUMINAMATH_CALUDE_ways_to_go_home_via_library_l270_27050

theorem ways_to_go_home_via_library (school_to_library : ℕ) (library_to_home : ℕ) : 
  school_to_library = 2 → library_to_home = 3 → school_to_library * library_to_home = 6 := by
  sorry

end NUMINAMATH_CALUDE_ways_to_go_home_via_library_l270_27050


namespace NUMINAMATH_CALUDE_smallest_hope_number_l270_27059

def hope_number (n : ℕ+) : Prop :=
  ∃ (a b c : ℕ), 
    (n / 8 : ℚ) = a^2 ∧ 
    (n / 9 : ℚ) = b^3 ∧ 
    (n / 25 : ℚ) = c^5

theorem smallest_hope_number :
  ∃ (n : ℕ+), hope_number n ∧ 
    (∀ (m : ℕ+), hope_number m → n ≤ m) ∧
    n = 2^15 * 3^20 * 5^12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_hope_number_l270_27059


namespace NUMINAMATH_CALUDE_power_equality_l270_27071

theorem power_equality (n b : ℝ) : n = 2 ^ (1/4) → n ^ b = 8 → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l270_27071


namespace NUMINAMATH_CALUDE_inequality_solution_set_l270_27006

theorem inequality_solution_set : 
  {x : ℝ | x + 7 > -2*x + 1} = {x : ℝ | x > -2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l270_27006


namespace NUMINAMATH_CALUDE_apple_count_theorem_l270_27017

def is_valid_apple_count (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ (∃ k : ℕ, n = 6 * k)

theorem apple_count_theorem : 
  ∀ n : ℕ, is_valid_apple_count n ↔ (n = 72 ∨ n = 78) :=
by sorry

end NUMINAMATH_CALUDE_apple_count_theorem_l270_27017


namespace NUMINAMATH_CALUDE_four_bottles_cost_l270_27041

/-- The cost of a certain number of bottles of mineral water -/
def cost (bottles : ℕ) : ℚ :=
  if bottles = 3 then 3/2 else (3/2 * bottles) / 3

/-- Theorem: The cost of 4 bottles of mineral water is 2 euros -/
theorem four_bottles_cost : cost 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_four_bottles_cost_l270_27041


namespace NUMINAMATH_CALUDE_smallest_x_with_remainders_l270_27060

theorem smallest_x_with_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧ 
  ∀ y : ℕ, y > 0 → y % 6 = 5 → y % 7 = 6 → y % 8 = 7 → x ≤ y :=
by
  use 167
  sorry

end NUMINAMATH_CALUDE_smallest_x_with_remainders_l270_27060


namespace NUMINAMATH_CALUDE_negation_equivalence_l270_27024

theorem negation_equivalence :
  (¬ ∃ x : ℝ, 2 * x - 3 > 1) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l270_27024


namespace NUMINAMATH_CALUDE_max_score_2079_score_2079_eq_30_unique_max_score_l270_27046

def score (x : ℕ) : ℕ :=
  (if x % 3 = 0 then 3 else 0) +
  (if x % 5 = 0 then 5 else 0) +
  (if x % 7 = 0 then 7 else 0) +
  (if x % 9 = 0 then 9 else 0) +
  (if x % 11 = 0 then 11 else 0)

theorem max_score_2079 :
  ∀ x : ℕ, 2017 ≤ x → x ≤ 2117 → score x ≤ score 2079 :=
by sorry

theorem score_2079_eq_30 : score 2079 = 30 :=
by sorry

theorem unique_max_score :
  ∀ x : ℕ, 2017 ≤ x → x ≤ 2117 → x ≠ 2079 → score x < score 2079 :=
by sorry

end NUMINAMATH_CALUDE_max_score_2079_score_2079_eq_30_unique_max_score_l270_27046


namespace NUMINAMATH_CALUDE_vacation_cost_l270_27005

theorem vacation_cost (C : ℝ) :
  (C / 4 - C / 5 = 50) → C = 1000 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l270_27005


namespace NUMINAMATH_CALUDE_adam_change_l270_27083

-- Define the given amounts
def adam_money : ℚ := 5.00
def airplane_cost : ℚ := 4.28

-- Define the change function
def change (money cost : ℚ) : ℚ := money - cost

-- Theorem statement
theorem adam_change :
  change adam_money airplane_cost = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_adam_change_l270_27083


namespace NUMINAMATH_CALUDE_animals_fiber_intake_l270_27063

-- Define the absorption rates and absorbed amounts
def koala_absorption_rate : ℝ := 0.30
def koala_absorbed_amount : ℝ := 12
def kangaroo_absorption_rate : ℝ := 0.40
def kangaroo_absorbed_amount : ℝ := 16

-- Define the theorem
theorem animals_fiber_intake :
  ∃ (koala_intake kangaroo_intake : ℝ),
    koala_intake * koala_absorption_rate = koala_absorbed_amount ∧
    kangaroo_intake * kangaroo_absorption_rate = kangaroo_absorbed_amount ∧
    koala_intake = 40 ∧
    kangaroo_intake = 40 := by
  sorry

end NUMINAMATH_CALUDE_animals_fiber_intake_l270_27063


namespace NUMINAMATH_CALUDE_distance_to_focus_l270_27069

/-- Given a parabola y² = 8x and a point P(4, y) on it, 
    the distance from P to the focus of the parabola is 6. -/
theorem distance_to_focus (y : ℝ) : 
  y^2 = 32 →  -- Point P(4, y) is on the parabola y² = 8x
  let F := (2, 0)  -- Focus of the parabola
  Real.sqrt ((4 - 2)^2 + y^2) = 6 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l270_27069


namespace NUMINAMATH_CALUDE_intersection_quadratic_equations_l270_27023

theorem intersection_quadratic_equations (p q : ℝ) : 
  let M := {x : ℝ | x^2 - p*x + 6 = 0}
  let N := {x : ℝ | x^2 + 6*x - q = 0}
  (M ∩ N = {2}) → p + q = 21 := by
sorry

end NUMINAMATH_CALUDE_intersection_quadratic_equations_l270_27023


namespace NUMINAMATH_CALUDE_coin_problem_l270_27022

theorem coin_problem (x y z : ℕ) : 
  x + y + z = 30 →
  10 * x + 15 * y + 20 * z = 500 →
  z > x :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l270_27022


namespace NUMINAMATH_CALUDE_largest_derivative_at_one_l270_27027

/-- The derivative of 2x+1 at x=1 is greater than the derivatives of -x², 1/x, and √x at x=1 -/
theorem largest_derivative_at_one :
  let f₁ (x : ℝ) := -x^2
  let f₂ (x : ℝ) := 1/x
  let f₃ (x : ℝ) := 2*x + 1
  let f₄ (x : ℝ) := Real.sqrt x
  (deriv f₃ 1 > deriv f₁ 1) ∧ 
  (deriv f₃ 1 > deriv f₂ 1) ∧ 
  (deriv f₃ 1 > deriv f₄ 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_derivative_at_one_l270_27027


namespace NUMINAMATH_CALUDE_carsons_speed_l270_27080

/-- Given information about Jerry and Carson's running times and the distance to school,
    we prove that Carson's speed is 8 miles per hour. -/
theorem carsons_speed (distance : ℝ) (jerry_time : ℝ) (carson_time : ℝ)
  (h1 : distance = 4) -- Distance to school is 4 miles
  (h2 : jerry_time = 15) -- Jerry's one-way trip time is 15 minutes
  (h3 : carson_time = 2 * jerry_time) -- Carson's one-way trip time is twice Jerry's
  : carson_time / 60 * distance = 8 := by
  sorry

end NUMINAMATH_CALUDE_carsons_speed_l270_27080


namespace NUMINAMATH_CALUDE_function_values_impossibility_l270_27058

theorem function_values_impossibility (a b c : ℝ) (d : ℤ) :
  ¬∃ (m : ℝ), (a * m^3 + b * m - c / m + d = -1) ∧
              (a * (-m)^3 + b * (-m) - c / (-m) + d = 4) := by
  sorry

end NUMINAMATH_CALUDE_function_values_impossibility_l270_27058


namespace NUMINAMATH_CALUDE_least_number_divisible_by_all_l270_27007

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 6) % 24 = 0 ∧ (n + 6) % 32 = 0 ∧ (n + 6) % 36 = 0 ∧ (n + 6) % 54 = 0

theorem least_number_divisible_by_all : 
  is_divisible_by_all 858 ∧ ∀ m : ℕ, m < 858 → ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_all_l270_27007


namespace NUMINAMATH_CALUDE_complex_equation_sum_l270_27010

theorem complex_equation_sum (a b : ℝ) :
  (Complex.I * (1 + a * Complex.I) = 1 + b * Complex.I) → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l270_27010


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l270_27002

def num_men : ℕ := 8
def num_women : ℕ := 4
def num_selected : ℕ := 4

theorem probability_at_least_one_woman :
  let total_people := num_men + num_women
  let prob_all_men := (num_men.choose num_selected : ℚ) / (total_people.choose num_selected : ℚ)
  (1 : ℚ) - prob_all_men = 85 / 99 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l270_27002


namespace NUMINAMATH_CALUDE_complex_sum_product_real_implies_a_eq_one_l270_27081

/-- Given complex numbers z₁ and z₂, prove that if their sum and product are real, then the imaginary part of z₁ is 1. -/
theorem complex_sum_product_real_implies_a_eq_one (a b : ℝ) : 
  let z₁ : ℂ := -1 + a * I
  let z₂ : ℂ := b - I
  (∃ (x : ℝ), z₁ + z₂ = x) → (∃ (y : ℝ), z₁ * z₂ = y) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_product_real_implies_a_eq_one_l270_27081


namespace NUMINAMATH_CALUDE_slope_product_l270_27076

/-- Given two lines L₁ and L₂ with equations y = mx and y = nx respectively,
    where L₁ makes three times as large an angle with the horizontal as L₂,
    L₁ has 5 times the slope of L₂, and L₁ is not horizontal,
    prove that mn = 5/7. -/
theorem slope_product (m n : ℝ) : 
  m ≠ 0 →  -- L₁ is not horizontal
  (∃ θ₁ θ₂ : ℝ, 
    θ₁ = 3 * θ₂ ∧  -- L₁ makes three times as large an angle with the horizontal as L₂
    m = Real.tan θ₁ ∧ 
    n = Real.tan θ₂ ∧
    m = 5 * n) →  -- L₁ has 5 times the slope of L₂
  m * n = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_slope_product_l270_27076


namespace NUMINAMATH_CALUDE_percentage_relationship_l270_27085

theorem percentage_relationship (A B T : ℝ) 
  (h1 : B = 0.14 * T) 
  (h2 : A = 0.5 * B) : 
  A = 0.07 * T := by
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l270_27085


namespace NUMINAMATH_CALUDE_equation_solutions_l270_27030

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁ + 3)^2 = 16 ∧ (2 * x₂ + 3)^2 = 16 ∧ x₁ = 1/2 ∧ x₂ = -7/2) ∧
  (∃ y₁ y₂ : ℝ, y₁^2 - 4*y₁ - 3 = 0 ∧ y₂^2 - 4*y₂ - 3 = 0 ∧ y₁ = 2 + Real.sqrt 7 ∧ y₂ = 2 - Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l270_27030


namespace NUMINAMATH_CALUDE_mathematics_magnet_problem_l270_27072

/-- The number of letters in 'MATHEMATICS' -/
def total_letters : ℕ := 11

/-- The number of vowels in 'MATHEMATICS' -/
def num_vowels : ℕ := 4

/-- The number of consonants in 'MATHEMATICS' -/
def num_consonants : ℕ := 7

/-- The number of vowels selected -/
def selected_vowels : ℕ := 3

/-- The number of consonants selected -/
def selected_consonants : ℕ := 4

/-- The number of distinct possible collections of letters -/
def distinct_collections : ℕ := 490

theorem mathematics_magnet_problem :
  (total_letters = num_vowels + num_consonants) →
  (distinct_collections = 490) :=
by sorry

end NUMINAMATH_CALUDE_mathematics_magnet_problem_l270_27072


namespace NUMINAMATH_CALUDE_soup_per_bag_is_three_l270_27028

-- Define the quantities
def milk_quarts : ℚ := 2
def vegetable_quarts : ℚ := 1
def num_bags : ℕ := 3

-- Define the relationship between milk and chicken stock
def chicken_stock_quarts : ℚ := 3 * milk_quarts

-- Calculate the total amount of soup
def total_soup : ℚ := milk_quarts + chicken_stock_quarts + vegetable_quarts

-- Define the amount of soup per bag
def soup_per_bag : ℚ := total_soup / num_bags

-- Theorem to prove
theorem soup_per_bag_is_three : soup_per_bag = 3 := by
  sorry

end NUMINAMATH_CALUDE_soup_per_bag_is_three_l270_27028


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_is_360_or_540_l270_27062

/-- A regular polygon with all diagonals equal -/
structure EqualDiagonalRegularPolygon where
  /-- Number of sides of the polygon -/
  n : ℕ
  /-- Condition that the polygon has at least 3 sides -/
  h_n : n ≥ 3
  /-- Condition that all diagonals are equal -/
  all_diagonals_equal : True

/-- The sum of interior angles of a regular polygon with all diagonals equal -/
def sum_of_interior_angles (p : EqualDiagonalRegularPolygon) : ℝ :=
  (p.n - 2) * 180

/-- Theorem stating that the sum of interior angles is either 360° or 540° -/
theorem sum_of_interior_angles_is_360_or_540 (p : EqualDiagonalRegularPolygon) :
  sum_of_interior_angles p = 360 ∨ sum_of_interior_angles p = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_is_360_or_540_l270_27062


namespace NUMINAMATH_CALUDE_chessboard_uniquely_determined_l270_27040

/-- Represents a chessboard with numbers 1 to 64 -/
def Chessboard := Fin 8 → Fin 8 → Fin 64

/-- The sum of numbers in a rectangle of two cells -/
def RectangleSum (board : Chessboard) (r1 c1 r2 c2 : Fin 8) : ℕ :=
  (board r1 c1).val + 1 + (board r2 c2).val + 1

/-- Predicate to check if two positions are on the same diagonal -/
def OnSameDiagonal (r1 c1 r2 c2 : Fin 8) : Prop :=
  r1 + c1 = r2 + c2 ∨ r1 + c2 = r2 + c1

/-- Main theorem -/
theorem chessboard_uniquely_determined 
  (board : Chessboard) 
  (sums_known : ∀ (r1 c1 r2 c2 : Fin 8), r1 = r2 ∧ c1.val + 1 = c2.val ∨ r1.val + 1 = r2.val ∧ c1 = c2 → 
    ∃ (s : ℕ), s = RectangleSum board r1 c1 r2 c2)
  (one_and_sixtyfour_on_diagonal : ∃ (r1 c1 r2 c2 : Fin 8), 
    board r1 c1 = 0 ∧ board r2 c2 = 63 ∧ OnSameDiagonal r1 c1 r2 c2) :
  ∀ (r c : Fin 8), ∃! (n : Fin 64), board r c = n :=
sorry

end NUMINAMATH_CALUDE_chessboard_uniquely_determined_l270_27040


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l270_27013

theorem parametric_to_ordinary_equation (t : ℝ) :
  let x := Real.exp t + Real.exp (-t)
  let y := 2 * (Real.exp t - Real.exp (-t))
  (x^2 / 4) - (y^2 / 16) = 1 ∧ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l270_27013


namespace NUMINAMATH_CALUDE_grid_toothpicks_count_l270_27088

/-- Calculates the total number of toothpicks in a rectangular grid with partitions. -/
def total_toothpicks (height width : ℕ) (partition_interval : ℕ) : ℕ :=
  let horizontal_lines := height + 1
  let vertical_lines := width + 1
  let horizontal_toothpicks := horizontal_lines * width
  let vertical_toothpicks := vertical_lines * height
  let num_partitions := (height - 1) / partition_interval
  let partition_toothpicks := num_partitions * width
  horizontal_toothpicks + vertical_toothpicks + partition_toothpicks

/-- Theorem stating that the total number of toothpicks in the specified grid is 850. -/
theorem grid_toothpicks_count :
  total_toothpicks 25 15 5 = 850 := by
  sorry

end NUMINAMATH_CALUDE_grid_toothpicks_count_l270_27088


namespace NUMINAMATH_CALUDE_filter_price_theorem_l270_27052

-- Define the number of filters and their prices
def total_filters : ℕ := 5
def kit_price : ℚ := 87.50
def price_filter_1 : ℚ := 16.45
def price_filter_2 : ℚ := 19.50
def num_filter_1 : ℕ := 2
def num_filter_2 : ℕ := 1
def num_unknown_price : ℕ := 2
def savings_percentage : ℚ := 0.08

-- Define the function to calculate the total individual price
def total_individual_price (x : ℚ) : ℚ :=
  num_filter_1 * price_filter_1 + num_unknown_price * x + num_filter_2 * price_filter_2

-- Define the theorem
theorem filter_price_theorem (x : ℚ) :
  (savings_percentage * total_individual_price x = total_individual_price x - kit_price) →
  x = 21.36 := by
  sorry

end NUMINAMATH_CALUDE_filter_price_theorem_l270_27052


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l270_27079

/-- A line in the two-dimensional plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept.1) + l.x_intercept.2)

/-- Theorem: For a line with slope -3 and x-intercept (7, 0), the y-intercept is (0, 21). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -3, x_intercept := (7, 0) }
  y_intercept l = (0, 21) := by sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l270_27079


namespace NUMINAMATH_CALUDE_marley_has_31_fruits_l270_27089

-- Define the number of fruits for Louis and Samantha
def louis_oranges : ℕ := 5
def louis_apples : ℕ := 3
def samantha_oranges : ℕ := 8
def samantha_apples : ℕ := 7

-- Define Marley's fruits in terms of Louis and Samantha
def marley_oranges : ℕ := 2 * louis_oranges
def marley_apples : ℕ := 3 * samantha_apples

-- Define the total number of Marley's fruits
def marley_total_fruits : ℕ := marley_oranges + marley_apples

-- Theorem statement
theorem marley_has_31_fruits : marley_total_fruits = 31 := by
  sorry

end NUMINAMATH_CALUDE_marley_has_31_fruits_l270_27089


namespace NUMINAMATH_CALUDE_arithmetic_sequence_logarithm_l270_27019

theorem arithmetic_sequence_logarithm (x : ℝ) : 
  (∃ r : ℝ, Real.log 2 + r = Real.log (2^x - 1) ∧ 
             Real.log (2^x - 1) + r = Real.log (2^x + 3)) → 
  x = Real.log 5 / Real.log 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_logarithm_l270_27019


namespace NUMINAMATH_CALUDE_ellipse_chord_bisector_l270_27032

def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

def point_inside_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 < 1

def bisector_line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

theorem ellipse_chord_bisector :
  ∀ x y : ℝ,
  ellipse x y →
  point_inside_ellipse 3 1 →
  bisector_line 3 4 (-13) x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_chord_bisector_l270_27032


namespace NUMINAMATH_CALUDE_sample_size_calculation_l270_27039

theorem sample_size_calculation (total_population : ℕ) (sampling_rate : ℚ) :
  total_population = 2000 →
  sampling_rate = 1/10 →
  (total_population : ℚ) * sampling_rate = 200 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l270_27039


namespace NUMINAMATH_CALUDE_expression_simplification_l270_27001

/-- Given nonzero real numbers a, b, c, and a constant real number θ,
    define x, y, z as specified, and prove that x^2 + y^2 + z^2 - xyz = 4 -/
theorem expression_simplification 
  (a b c : ℝ) (θ : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (x : ℝ := b / c + c / b + Real.sin θ)
  (y : ℝ := a / c + c / a + Real.cos θ)
  (z : ℝ := a / b + b / a + Real.tan θ) :
  x^2 + y^2 + z^2 - x*y*z = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l270_27001


namespace NUMINAMATH_CALUDE_factorial_30_trailing_zeros_l270_27097

def trailing_zeros (n : ℕ) : ℕ := 
  (n / 5) + (n / 25)

theorem factorial_30_trailing_zeros : 
  trailing_zeros 30 = 7 := by sorry

end NUMINAMATH_CALUDE_factorial_30_trailing_zeros_l270_27097


namespace NUMINAMATH_CALUDE_least_amount_to_add_l270_27070

def savings : ℕ := 642986
def children : ℕ := 9

theorem least_amount_to_add : 
  (∃ (x : ℕ), (savings + x) % children = 0 ∧ 
  ∀ (y : ℕ), y < x → (savings + y) % children ≠ 0) → 
  (∃ (x : ℕ), (savings + x) % children = 0 ∧ 
  ∀ (y : ℕ), y < x → (savings + y) % children ≠ 0 ∧ x = 1) :=
sorry

end NUMINAMATH_CALUDE_least_amount_to_add_l270_27070


namespace NUMINAMATH_CALUDE_dracula_is_alive_l270_27084

-- Define the propositions
variable (T : Prop) -- "The Transylvanian is human"
variable (D : Prop) -- "Count Dracula is alive"

-- Define the Transylvanian's statements
variable (statement1 : T)
variable (statement2 : T → D)

-- Define the Transylvanian's ability to reason logically
variable (logical_reasoning : T)

-- Theorem to prove
theorem dracula_is_alive : D := by
  sorry

end NUMINAMATH_CALUDE_dracula_is_alive_l270_27084


namespace NUMINAMATH_CALUDE_two_number_cards_totaling_twelve_probability_l270_27033

/-- Represents a standard deck of cards -/
def StandardDeck : Type := Unit

/-- Number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- Set of card values that are numbers (2 through 10) -/
def numberCards : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9, 10}

/-- Number of cards of each value in the deck -/
def cardsPerValue : ℕ := 4

/-- Predicate for two cards totaling 12 -/
def totalTwelve (card1 card2 : ℕ) : Prop := card1 + card2 = 12

/-- The probability of the event -/
def probabilityTwoNumberCardsTotalingTwelve (deck : StandardDeck) : ℚ :=
  35 / 663

theorem two_number_cards_totaling_twelve_probability 
  (deck : StandardDeck) : 
  probabilityTwoNumberCardsTotalingTwelve deck = 35 / 663 := by
  sorry

end NUMINAMATH_CALUDE_two_number_cards_totaling_twelve_probability_l270_27033


namespace NUMINAMATH_CALUDE_special_set_is_all_reals_l270_27091

/-- A subset of real numbers with a special property -/
def SpecialSet (A : Set ℝ) : Prop :=
  (∀ x y : ℝ, x + y ∈ A → x * y ∈ A) ∧ Set.Nonempty A

/-- The main theorem: Any special set of real numbers is equal to the entire set of real numbers -/
theorem special_set_is_all_reals (A : Set ℝ) (h : SpecialSet A) : A = Set.univ :=
sorry

end NUMINAMATH_CALUDE_special_set_is_all_reals_l270_27091


namespace NUMINAMATH_CALUDE_sabina_college_loan_l270_27043

theorem sabina_college_loan (college_cost savings grant_percentage : ℝ) : 
  college_cost = 30000 →
  savings = 10000 →
  grant_percentage = 0.4 →
  let remainder := college_cost - savings
  let grant_amount := grant_percentage * remainder
  let loan_amount := remainder - grant_amount
  loan_amount = 12000 := by
sorry

end NUMINAMATH_CALUDE_sabina_college_loan_l270_27043


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l270_27054

theorem baseball_card_value_decrease (x : ℝ) : 
  (1 - x / 100) * (1 - 30 / 100) = 1 - 44.00000000000001 / 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l270_27054


namespace NUMINAMATH_CALUDE_tetrahedron_inscribed_circumscribed_inequality_l270_27048

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The inscribed sphere of a tetrahedron -/
def inscribedSphere (t : Tetrahedron) : Sphere := sorry

/-- The circumscribed sphere of a tetrahedron -/
def circumscribedSphere (t : Tetrahedron) : Sphere := sorry

/-- The intersection of the planes of the remaining faces -/
def planesIntersection (t : Tetrahedron) : Point3D := sorry

/-- The intersection of a line segment with a sphere -/
def lineIntersectSphere (p1 p2 : Point3D) (s : Sphere) : Point3D := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

theorem tetrahedron_inscribed_circumscribed_inequality (t : Tetrahedron) :
  let I := (inscribedSphere t).center
  let J := planesIntersection t
  let K := lineIntersectSphere I J (circumscribedSphere t)
  distance I K > distance J K := by sorry

end NUMINAMATH_CALUDE_tetrahedron_inscribed_circumscribed_inequality_l270_27048


namespace NUMINAMATH_CALUDE_rice_distributed_in_five_days_l270_27065

/-- The amount of rice distributed in the first 5 days of dike construction --/
theorem rice_distributed_in_five_days : 
  let initial_workers : ℕ := 64
  let daily_increase : ℕ := 7
  let rice_per_worker : ℕ := 3
  let days : ℕ := 5
  let total_workers : ℕ := (days * (2 * initial_workers + (days - 1) * daily_increase)) / 2
  total_workers * rice_per_worker = 1170 := by
  sorry

end NUMINAMATH_CALUDE_rice_distributed_in_five_days_l270_27065


namespace NUMINAMATH_CALUDE_min_value_expression_l270_27026

theorem min_value_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  (1 / m + 2 / n) ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 1 ∧ 1 / m₀ + 2 / n₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l270_27026


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l270_27021

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 2*k*x^2 + (8*k+1)*x + 8*k = 0 ∧ 2*k*y^2 + (8*k+1)*y + 8*k = 0) 
  ↔ 
  (k ≥ -1/16 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l270_27021


namespace NUMINAMATH_CALUDE_paulines_dress_cost_l270_27073

theorem paulines_dress_cost (pauline ida jean patty : ℕ) 
  (h1 : patty = ida + 10)
  (h2 : ida = jean + 30)
  (h3 : jean = pauline - 10)
  (h4 : pauline + ida + jean + patty = 160) :
  pauline = 30 := by
  sorry

end NUMINAMATH_CALUDE_paulines_dress_cost_l270_27073


namespace NUMINAMATH_CALUDE_different_signs_larger_negative_l270_27009

theorem different_signs_larger_negative (a b : ℝ) : 
  a + b < 0 → a * b < 0 → 
  ((a < 0 ∧ b > 0 ∧ abs a > abs b) ∨ (a > 0 ∧ b < 0 ∧ abs b > abs a)) := by
  sorry

end NUMINAMATH_CALUDE_different_signs_larger_negative_l270_27009


namespace NUMINAMATH_CALUDE_height_prediction_approximate_l270_27093

/-- Represents a linear regression model for height prediction -/
structure HeightModel where
  slope : ℝ
  intercept : ℝ

/-- Predicts height based on the model and age -/
def predict_height (model : HeightModel) (age : ℝ) : ℝ :=
  model.slope * age + model.intercept

/-- The given height prediction model -/
def given_model : HeightModel := { slope := 7.19, intercept := 73.93 }

/-- Theorem stating that the predicted height at age 10 is approximately 145.83cm -/
theorem height_prediction_approximate :
  ∃ ε > 0, ∀ δ > 0, δ < ε → 
    |predict_height given_model 10 - 145.83| < δ :=
sorry

end NUMINAMATH_CALUDE_height_prediction_approximate_l270_27093


namespace NUMINAMATH_CALUDE_direct_sort_5_rounds_l270_27018

def initial_sequence : List Nat := [49, 38, 65, 97, 76, 13, 27]

def direct_sort_step (l : List Nat) : List Nat :=
  match l with
  | [] => []
  | _ => let max := l.maximum? |>.getD 0
         max :: (l.filter (· ≠ max))

def direct_sort (l : List Nat) (n : Nat) : List Nat :=
  match n with
  | 0 => l
  | n + 1 => direct_sort (direct_sort_step l) n

theorem direct_sort_5_rounds :
  direct_sort initial_sequence 5 = [97, 76, 65, 49, 38, 13, 27] := by
  sorry

end NUMINAMATH_CALUDE_direct_sort_5_rounds_l270_27018


namespace NUMINAMATH_CALUDE_geometric_sum_first_seven_terms_l270_27067

/-- Sum of a finite geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric sequence -/
def a : ℚ := 1/5

/-- The common ratio of the geometric sequence -/
def r : ℚ := 1/3

/-- The number of terms to sum -/
def n : ℕ := 7

theorem geometric_sum_first_seven_terms :
  geometric_sum a r n = 2186/3645 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_seven_terms_l270_27067


namespace NUMINAMATH_CALUDE_crosswalk_distance_l270_27066

/-- Given a parallelogram with one side of length 25 feet, the perpendicular distance
    between this side and its opposite side being 60 feet, and another side of length 70 feet,
    the perpendicular distance between this side and its opposite side is 150/7 feet. -/
theorem crosswalk_distance (side1 side2 height1 height2 : ℝ) : 
  side1 = 25 →
  side2 = 70 →
  height1 = 60 →
  side1 * height1 = side2 * height2 →
  height2 = 150 / 7 := by sorry

end NUMINAMATH_CALUDE_crosswalk_distance_l270_27066


namespace NUMINAMATH_CALUDE_work_days_of_a_l270_27057

/-- Represents the number of days worked by each person -/
structure WorkDays where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the daily wages of each person -/
structure DailyWages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total earnings given work days and daily wages -/
def totalEarnings (days : WorkDays) (wages : DailyWages) : ℕ :=
  days.a * wages.a + days.b * wages.b + days.c * wages.c

/-- The main theorem to prove -/
theorem work_days_of_a (days : WorkDays) (wages : DailyWages) : 
  days.b = 9 ∧ 
  days.c = 4 ∧ 
  wages.a * 4 = wages.b * 3 ∧ 
  wages.b * 5 = wages.c * 4 ∧ 
  wages.c = 125 ∧ 
  totalEarnings days wages = 1850 → 
  days.a = 6 := by
  sorry


end NUMINAMATH_CALUDE_work_days_of_a_l270_27057


namespace NUMINAMATH_CALUDE_no_two_different_three_digit_cubes_l270_27092

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i j, i < digits.length → j < digits.length → i ≠ j → digits.get ⟨i, by sorry⟩ ≠ digits.get ⟨j, by sorry⟩

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

theorem no_two_different_three_digit_cubes :
  ∀ KUB SHAR : ℕ,
  is_three_digit KUB →
  is_three_digit SHAR →
  all_digits_different KUB →
  all_digits_different SHAR →
  is_cube KUB →
  (∀ d : ℕ, d < 10 → (d ∈ KUB.digits 10 → d ∉ SHAR.digits 10) ∧ (d ∈ SHAR.digits 10 → d ∉ KUB.digits 10)) →
  ¬ is_cube SHAR :=
by sorry

end NUMINAMATH_CALUDE_no_two_different_three_digit_cubes_l270_27092


namespace NUMINAMATH_CALUDE_intersection_empty_implies_k_leq_one_l270_27029

-- Define the sets P and Q
def P (k : ℝ) : Set ℝ := {y | y = k}
def Q (a : ℝ) : Set ℝ := {y | ∃ x : ℝ, y = a^x + 1}

-- State the theorem
theorem intersection_empty_implies_k_leq_one
  (k : ℝ) (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  (P k ∩ Q a = ∅) → k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_k_leq_one_l270_27029


namespace NUMINAMATH_CALUDE_function_inequality_l270_27064

theorem function_inequality (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f x + y) * (f y + x) > 0 → f x + y = f y + x) :
  ∀ x y : ℝ, x > y → f x + y ≤ f y + x := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l270_27064


namespace NUMINAMATH_CALUDE_cos_80_cos_20_plus_sin_80_sin_20_l270_27012

theorem cos_80_cos_20_plus_sin_80_sin_20 : Real.cos (80 * π / 180) * Real.cos (20 * π / 180) + Real.sin (80 * π / 180) * Real.sin (20 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_80_cos_20_plus_sin_80_sin_20_l270_27012


namespace NUMINAMATH_CALUDE_all_three_hits_mutually_exclusive_with_at_most_two_hits_l270_27095

-- Define the sample space for three shots
inductive ShotOutcome
| Hit
| Miss

-- Define the type for a sequence of three shots
def ThreeShots := (ShotOutcome × ShotOutcome × ShotOutcome)

-- Define the event "at most two hits"
def atMostTwoHits (shots : ThreeShots) : Prop :=
  match shots with
  | (ShotOutcome.Hit, ShotOutcome.Hit, ShotOutcome.Hit) => False
  | _ => True

-- Define the event "all three hits"
def allThreeHits (shots : ThreeShots) : Prop :=
  match shots with
  | (ShotOutcome.Hit, ShotOutcome.Hit, ShotOutcome.Hit) => True
  | _ => False

-- Theorem stating that "all three hits" and "at most two hits" are mutually exclusive
theorem all_three_hits_mutually_exclusive_with_at_most_two_hits :
  ∀ (shots : ThreeShots), ¬(atMostTwoHits shots ∧ allThreeHits shots) :=
by
  sorry


end NUMINAMATH_CALUDE_all_three_hits_mutually_exclusive_with_at_most_two_hits_l270_27095


namespace NUMINAMATH_CALUDE_flowerbed_count_l270_27051

theorem flowerbed_count (total_seeds : ℕ) (seeds_per_bed : ℕ) (h1 : total_seeds = 32) (h2 : seeds_per_bed = 4) :
  total_seeds / seeds_per_bed = 8 := by
  sorry

end NUMINAMATH_CALUDE_flowerbed_count_l270_27051


namespace NUMINAMATH_CALUDE_fair_coin_three_heads_probability_l270_27087

theorem fair_coin_three_heads_probability :
  let p_head : ℝ := 1/2  -- Probability of getting heads on a fair coin
  let n : ℕ := 3        -- Number of tosses
  let p_all_heads : ℝ := p_head ^ n
  p_all_heads = 1/8 := by
sorry

end NUMINAMATH_CALUDE_fair_coin_three_heads_probability_l270_27087


namespace NUMINAMATH_CALUDE_solution_set_f_geq_2_max_a_value_l270_27056

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x - 1|

-- Theorem for the solution set of f(x) ≥ 2
theorem solution_set_f_geq_2 :
  {x : ℝ | f x ≥ 2} = {x : ℝ | x ≤ 0 ∨ x ≥ 4/3} :=
sorry

-- Theorem for the maximum value of a
theorem max_a_value (a : ℝ) :
  (∀ x : ℝ, f x ≥ a * |x|) ↔ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_2_max_a_value_l270_27056


namespace NUMINAMATH_CALUDE_min_value_theorem_l270_27008

theorem min_value_theorem (x : ℝ) (h : x > 5) : x + 1 / (x - 5) ≥ 7 ∧ ∃ y > 5, y + 1 / (y - 5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l270_27008


namespace NUMINAMATH_CALUDE_flower_planting_cost_l270_27004

/-- The cost of planting and maintaining flowers with given items -/
theorem flower_planting_cost (flower_cost : ℚ) (h1 : flower_cost = 9) : ∃ total_cost : ℚ,
  let clay_pot_cost := flower_cost + 20
  let soil_cost := flower_cost - 2
  let fertilizer_cost := flower_cost * (1 + 1/2)
  let tools_cost := clay_pot_cost * (1 - 1/4)
  total_cost = flower_cost + clay_pot_cost + soil_cost + fertilizer_cost + tools_cost ∧ 
  total_cost = 80.25 := by
  sorry

end NUMINAMATH_CALUDE_flower_planting_cost_l270_27004


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_open_zero_one_l270_27034

def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem M_intersect_N_equals_open_zero_one : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_open_zero_one_l270_27034


namespace NUMINAMATH_CALUDE_reflect_parabola_x_axis_l270_27061

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflects a parabola across the x-axis -/
def reflect_x_axis (p : Parabola) : Parabola :=
  { a := -p.a, b := -p.b, c := -p.c }

/-- The original parabola y = x^2 - x - 1 -/
def original_parabola : Parabola :=
  { a := 1, b := -1, c := -1 }

theorem reflect_parabola_x_axis :
  reflect_x_axis original_parabola = { a := -1, b := 1, c := 1 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_parabola_x_axis_l270_27061


namespace NUMINAMATH_CALUDE_complex_equation_solution_l270_27042

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 2 + Complex.I → z = 1 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l270_27042


namespace NUMINAMATH_CALUDE_remainder_of_sum_l270_27094

theorem remainder_of_sum (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_div : x = u * y + v) (h_rem : v < y) : (x + 3 * u * y) % y = v := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_l270_27094


namespace NUMINAMATH_CALUDE_angle_Q_measure_l270_27090

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

-- Define the extended sides and point Q
def extended_sides (octagon : RegularOctagon) : sorry := sorry

def point_Q (octagon : RegularOctagon) : ℝ × ℝ := sorry

-- Define the angle at Q
def angle_Q (octagon : RegularOctagon) : ℝ := sorry

-- Theorem statement
theorem angle_Q_measure (octagon : RegularOctagon) : 
  angle_Q octagon = 22.5 := by sorry

end NUMINAMATH_CALUDE_angle_Q_measure_l270_27090


namespace NUMINAMATH_CALUDE_raisin_distribution_l270_27055

/-- The number of raisins Bryce received -/
def bryce_raisins : ℕ := 15

/-- The number of raisins Carter received -/
def carter_raisins : ℕ := bryce_raisins - 10

theorem raisin_distribution : 
  (bryce_raisins = 15) ∧ 
  (carter_raisins = bryce_raisins - 10) ∧ 
  (carter_raisins = bryce_raisins / 3) := by
  sorry

end NUMINAMATH_CALUDE_raisin_distribution_l270_27055


namespace NUMINAMATH_CALUDE_cube_root_of_eight_l270_27036

theorem cube_root_of_eight : ∃ x : ℝ, x^3 = 8 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_eight_l270_27036


namespace NUMINAMATH_CALUDE_division_simplification_l270_27099

theorem division_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  6 * a^2 * b / (2 * a * b) = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l270_27099


namespace NUMINAMATH_CALUDE_unique_function_solution_l270_27098

theorem unique_function_solution :
  ∃! f : ℕ → ℕ, ∀ x y : ℕ, x > 0 ∧ y > 0 →
    f x + y * (f (f x)) < x * (1 + f y) + 2021 ∧ f = id := by
  sorry

end NUMINAMATH_CALUDE_unique_function_solution_l270_27098


namespace NUMINAMATH_CALUDE_fourth_month_sale_is_9230_l270_27096

/-- Calculates the sale in the fourth month given the sales for other months and the average sale. -/
def fourth_month_sale (first_month second_month third_month fifth_month sixth_month average_sale : ℕ) : ℕ :=
  6 * average_sale - (first_month + second_month + third_month + fifth_month + sixth_month)

/-- Theorem stating that the sale in the fourth month is 9230 given the problem conditions. -/
theorem fourth_month_sale_is_9230 :
  fourth_month_sale 8435 8927 8855 8562 6991 8500 = 9230 := by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sale_is_9230_l270_27096


namespace NUMINAMATH_CALUDE_number_problem_l270_27044

theorem number_problem (x : ℝ) (n : ℝ) (h1 : x > 0) (h2 : x / 50 + x / n = 0.06 * x) : n = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l270_27044


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_zero_l270_27045

-- Define a cubic polynomial
def cubic_polynomial (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem cubic_polynomial_sum_zero 
  (a b c d : ℝ) 
  (h1 : cubic_polynomial a b c d 0 = 2 * d)
  (h2 : cubic_polynomial a b c d 1 = 3 * d)
  (h3 : cubic_polynomial a b c d (-1) = 5 * d) :
  cubic_polynomial a b c d 3 + cubic_polynomial a b c d (-3) = 0 := by
sorry


end NUMINAMATH_CALUDE_cubic_polynomial_sum_zero_l270_27045


namespace NUMINAMATH_CALUDE_negative_four_cubed_equality_l270_27031

theorem negative_four_cubed_equality : (-4)^3 = -(4^3) := by sorry

end NUMINAMATH_CALUDE_negative_four_cubed_equality_l270_27031


namespace NUMINAMATH_CALUDE_andys_calculation_l270_27086

theorem andys_calculation (y : ℝ) : 4 * y + 5 = 57 → (y + 5) * 4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_andys_calculation_l270_27086


namespace NUMINAMATH_CALUDE_condition_analysis_l270_27038

theorem condition_analysis (x : ℝ) :
  (∀ x, -1 < x ∧ x < 3 → x^2 - 2*x < 8) ∧
  (∃ x, x^2 - 2*x < 8 ∧ ¬(-1 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_condition_analysis_l270_27038


namespace NUMINAMATH_CALUDE_weighted_average_theorem_l270_27000

def group1_avg : ℝ := 30
def group1_weight : ℝ := 2
def group2_avg : ℝ := 40
def group2_weight : ℝ := 3
def group3_avg : ℝ := 20
def group3_weight : ℝ := 1

def total_weighted_sum : ℝ := group1_avg * group1_weight + group2_avg * group2_weight + group3_avg * group3_weight
def total_weight : ℝ := group1_weight + group2_weight + group3_weight

theorem weighted_average_theorem : total_weighted_sum / total_weight = 200 / 6 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_theorem_l270_27000


namespace NUMINAMATH_CALUDE_smallest_quotient_smallest_quotient_achievable_l270_27037

def card_set : Set ℤ := {-5, -4, 0, 4, 6}

theorem smallest_quotient (a b : ℤ) (ha : a ∈ card_set) (hb : b ∈ card_set) (hab : a ≠ b) (hb_nonzero : b ≠ 0) :
  (a : ℚ) / b ≥ -3/2 :=
sorry

theorem smallest_quotient_achievable :
  ∃ (a b : ℤ), a ∈ card_set ∧ b ∈ card_set ∧ a ≠ b ∧ b ≠ 0 ∧ (a : ℚ) / b = -3/2 :=
sorry

end NUMINAMATH_CALUDE_smallest_quotient_smallest_quotient_achievable_l270_27037


namespace NUMINAMATH_CALUDE_jasmine_percentage_l270_27068

/-- Calculates the percentage of jasmine in a solution after adding jasmine and water -/
theorem jasmine_percentage
  (initial_volume : ℝ)
  (initial_jasmine_percentage : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 90)
  (h2 : initial_jasmine_percentage = 5)
  (h3 : added_jasmine = 8)
  (h4 : added_water = 2) :
  let initial_jasmine := initial_volume * (initial_jasmine_percentage / 100)
  let total_jasmine := initial_jasmine + added_jasmine
  let total_volume := initial_volume + added_jasmine + added_water
  let final_percentage := (total_jasmine / total_volume) * 100
  final_percentage = 12.5 := by
sorry


end NUMINAMATH_CALUDE_jasmine_percentage_l270_27068


namespace NUMINAMATH_CALUDE_max_length_sum_l270_27003

/-- Length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer -/
def length (n : ℕ) : ℕ := sorry

theorem max_length_sum :
  ∃ (x y : ℕ),
    x > 1 ∧
    y > 1 ∧
    x + 3 * y < 5000 ∧
    length x + length y = 20 ∧
    ∀ (a b : ℕ),
      a > 1 →
      b > 1 →
      a + 3 * b < 5000 →
      length a + length b ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_max_length_sum_l270_27003


namespace NUMINAMATH_CALUDE_dance_attendance_l270_27074

theorem dance_attendance (boys girls teachers : ℕ) : 
  (boys : ℚ) / girls = 3 / 4 →
  teachers = boys / 5 →
  boys + girls + teachers = 114 →
  girls = 60 := by
sorry

end NUMINAMATH_CALUDE_dance_attendance_l270_27074


namespace NUMINAMATH_CALUDE_kgood_existence_l270_27053

def IsKGood (k : ℕ) (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

theorem kgood_existence (k : ℕ) :
  (k ≥ 2 → ∃ f : ℕ+ → ℕ+, IsKGood k f) ∧
  (k = 1 → ¬∃ f : ℕ+ → ℕ+, IsKGood k f) :=
sorry

end NUMINAMATH_CALUDE_kgood_existence_l270_27053


namespace NUMINAMATH_CALUDE_mans_age_twice_students_l270_27015

/-- Proves that it takes 2 years for a man's age to be twice his student's age -/
theorem mans_age_twice_students (student_age : ℕ) (age_difference : ℕ) : 
  student_age = 24 → age_difference = 26 → 
  ∃ (years : ℕ), (student_age + years) * 2 = (student_age + age_difference + years) ∧ years = 2 :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_students_l270_27015
