import Mathlib

namespace NUMINAMATH_CALUDE_some_ounce_glass_size_l2280_228064

/-- Given the following conditions:
  - Claudia has 122 ounces of water
  - She fills six 5-ounce glasses and four 8-ounce glasses
  - She can fill 15 glasses of the some-ounce size with the remaining water
  Prove that the size of the some-ounce glasses is 4 ounces. -/
theorem some_ounce_glass_size (total_water : ℕ) (five_ounce_count : ℕ) (eight_ounce_count : ℕ) (some_ounce_count : ℕ)
  (h1 : total_water = 122)
  (h2 : five_ounce_count = 6)
  (h3 : eight_ounce_count = 4)
  (h4 : some_ounce_count = 15)
  (h5 : total_water = 5 * five_ounce_count + 8 * eight_ounce_count + some_ounce_count * (total_water - 5 * five_ounce_count - 8 * eight_ounce_count) / some_ounce_count) :
  (total_water - 5 * five_ounce_count - 8 * eight_ounce_count) / some_ounce_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_some_ounce_glass_size_l2280_228064


namespace NUMINAMATH_CALUDE_min_trains_for_800_passengers_l2280_228001

/-- Given a maximum capacity of passengers per train and a total number of passengers to transport,
    calculate the minimum number of trains required. -/
def min_trains (capacity : ℕ) (total_passengers : ℕ) : ℕ :=
  (total_passengers + capacity - 1) / capacity

theorem min_trains_for_800_passengers :
  min_trains 50 800 = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_trains_for_800_passengers_l2280_228001


namespace NUMINAMATH_CALUDE_no_solution_exists_l2280_228071

theorem no_solution_exists : ∀ n : ℤ, n^2022 - 2*n^2021 + 3*n^2019 ≠ 2020 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2280_228071


namespace NUMINAMATH_CALUDE_sandy_clothing_purchase_l2280_228011

/-- Represents the amount spent on clothes in a foreign currency and the exchange rate --/
structure ClothingPurchase where
  shorts : ℝ
  shirt : ℝ
  jacket : ℝ
  exchange_rate : ℝ

/-- Calculates the total amount spent in the home currency --/
def total_spent_home_currency (purchase : ClothingPurchase) : ℝ :=
  (purchase.shorts + purchase.shirt + purchase.jacket) * purchase.exchange_rate

/-- Theorem stating that the total amount spent in the home currency is 33.56 times the exchange rate --/
theorem sandy_clothing_purchase (purchase : ClothingPurchase)
  (h_shorts : purchase.shorts = 13.99)
  (h_shirt : purchase.shirt = 12.14)
  (h_jacket : purchase.jacket = 7.43) :
  total_spent_home_currency purchase = 33.56 * purchase.exchange_rate := by
  sorry

end NUMINAMATH_CALUDE_sandy_clothing_purchase_l2280_228011


namespace NUMINAMATH_CALUDE_brocard_angle_inequalities_l2280_228051

theorem brocard_angle_inequalities (α β γ φ : Real) 
  (triangle : α + β + γ = Real.pi)
  (brocard_condition : φ ≤ Real.pi / 6)
  (sin_relation : Real.sin (α - φ) * Real.sin (β - φ) * Real.sin (γ - φ) = Real.sin φ ^ 3) :
  φ ^ 3 ≤ (α - φ) * (β - φ) * (γ - φ) ∧ 8 * φ ^ 3 ≤ α * β * γ := by
  sorry

end NUMINAMATH_CALUDE_brocard_angle_inequalities_l2280_228051


namespace NUMINAMATH_CALUDE_y_intercept_of_given_line_l2280_228061

/-- A line is defined by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate where the line crosses the y-axis -/
def y_intercept (l : Line) : ℝ :=
  l.slope * (-l.point.1) + l.point.2

/-- The given line has slope 3 and passes through the point (4, 0) -/
def given_line : Line :=
  { slope := 3, point := (4, 0) }

theorem y_intercept_of_given_line :
  y_intercept given_line = -12 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_given_line_l2280_228061


namespace NUMINAMATH_CALUDE_largest_non_formable_is_correct_l2280_228039

/-- Represents the coin denominations in Limonia -/
def coin_denominations (n : ℕ) : Finset ℕ :=
  {3*n - 1, 6*n + 1, 6*n + 4, 6*n + 7}

/-- Predicate to check if an amount can be formed using the given coin denominations -/
def is_formable (n : ℕ) (amount : ℕ) : Prop :=
  ∃ (a b c d : ℕ), amount = a*(3*n - 1) + b*(6*n + 1) + c*(6*n + 4) + d*(6*n + 7)

/-- The largest non-formable amount in Limonia -/
def largest_non_formable (n : ℕ) : ℕ := 6*n^2 + 4*n - 5

/-- Theorem stating that the largest non-formable amount is correct -/
theorem largest_non_formable_is_correct (n : ℕ) :
  (∀ m : ℕ, m > largest_non_formable n → is_formable n m) ∧
  ¬is_formable n (largest_non_formable n) :=
sorry

end NUMINAMATH_CALUDE_largest_non_formable_is_correct_l2280_228039


namespace NUMINAMATH_CALUDE_triangle_side_bounds_l2280_228035

theorem triangle_side_bounds (k : ℕ) (a b c : ℕ) 
  (h1 : a + b + c = k) 
  (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : 
  (2 - (k - 2*(k/2)) ≤ a ∧ a ≤ k/3) ∧
  ((k+4)/4 ≤ b ∧ b ≤ (k-1)/2) ∧
  ((k+2)/3 ≤ c ∧ c ≤ (k-1)/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_bounds_l2280_228035


namespace NUMINAMATH_CALUDE_range_intersection_l2280_228037

theorem range_intersection (x : ℝ) : 
  (x^2 - 7*x + 10 ≤ 0) ∧ ((x - 3)*(x + 1) ≤ 0) ↔ 2 ≤ x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_intersection_l2280_228037


namespace NUMINAMATH_CALUDE_isosceles_equilateral_conditions_l2280_228010

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Checks if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- Represents the feet of perpendiculars from a point to the sides of a triangle -/
structure Perpendiculars where
  D : Point  -- foot on AB
  E : Point  -- foot on BC
  F : Point  -- foot on CA

/-- Calculates the feet of perpendiculars from a point to the sides of a triangle -/
def calculatePerpendiculars (p : Point) (t : Triangle) : Perpendiculars := sorry

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Represents an Apollonius circle of a triangle -/
structure ApolloniusCircle where
  center : Point
  radius : ℝ

/-- Calculates the Apollonius circles of a triangle -/
def calculateApolloniusCircles (t : Triangle) : List ApolloniusCircle := sorry

/-- Checks if a point lies on an Apollonius circle -/
def liesOnApolloniusCircle (p : Point) (c : ApolloniusCircle) : Prop := sorry

/-- Calculates the Fermat point of a triangle -/
def calculateFermatPoint (t : Triangle) : Point := sorry

/-- The main theorem -/
theorem isosceles_equilateral_conditions 
  (t : Triangle) 
  (h1 : isAcuteAngled t) 
  (p : Point) 
  (h2 : isInside p t) 
  (perps : Perpendiculars) 
  (h3 : perps = calculatePerpendiculars p t) :
  (isIsosceles (Triangle.mk perps.D perps.E perps.F) ↔ 
    ∃ c ∈ calculateApolloniusCircles t, liesOnApolloniusCircle p c) ∧
  (isEquilateral (Triangle.mk perps.D perps.E perps.F) ↔ 
    p = calculateFermatPoint t) := by sorry

end NUMINAMATH_CALUDE_isosceles_equilateral_conditions_l2280_228010


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2280_228056

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2280_228056


namespace NUMINAMATH_CALUDE_circle_radius_from_square_perimeter_area_equality_l2280_228023

theorem circle_radius_from_square_perimeter_area_equality (r : ℝ) : 
  (4 * (r * Real.sqrt 2)) = (Real.pi * r^2) → r = (4 * Real.sqrt 2) / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_square_perimeter_area_equality_l2280_228023


namespace NUMINAMATH_CALUDE_danica_plane_arrangement_l2280_228005

theorem danica_plane_arrangement : 
  (∃ n : ℕ, (17 + n) % 7 = 0 ∧ ∀ m : ℕ, m < n → (17 + m) % 7 ≠ 0) → 
  (∃ n : ℕ, (17 + n) % 7 = 0 ∧ ∀ m : ℕ, m < n → (17 + m) % 7 ≠ 0 ∧ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_danica_plane_arrangement_l2280_228005


namespace NUMINAMATH_CALUDE_notebook_cost_l2280_228060

/-- The cost of a notebook and pencil, given their relationship -/
theorem notebook_cost (notebook_cost pencil_cost : ℝ) 
  (total : notebook_cost + pencil_cost = 2.40)
  (difference : notebook_cost = pencil_cost + 2) :
  notebook_cost = 2.20 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l2280_228060


namespace NUMINAMATH_CALUDE_variable_equals_one_l2280_228045

/-- The operator  applied to a real number x -/
def box_operator (x : ℝ) : ℝ := x * (2 - x)

/-- Theorem stating that if y + 1 = (y + 1), then y = 1 -/
theorem variable_equals_one (y : ℝ) (h : y + 1 = box_operator (y + 1)) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_variable_equals_one_l2280_228045


namespace NUMINAMATH_CALUDE_problem_solution_l2280_228007

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x^2 else a^x - 1

theorem problem_solution (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : Monotone (f a)) 
  (h4 : f a a = 5 * a - 2) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2280_228007


namespace NUMINAMATH_CALUDE_min_smartphones_for_discount_l2280_228012

def smartphone_price : ℝ := 600
def discount_rate : ℝ := 0.05
def savings : ℝ := 90

theorem min_smartphones_for_discount :
  ∃ n : ℕ, n > 0 ∧ 
  n * smartphone_price * discount_rate = savings ∧
  ∀ m : ℕ, m > 0 → m * smartphone_price * discount_rate = savings → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_min_smartphones_for_discount_l2280_228012


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l2280_228082

theorem cryptarithmetic_puzzle (A B C : ℕ) : 
  A + B + C = 10 →
  B + A + 1 = 10 →
  A + 1 = 3 →
  (A ≠ B ∧ A ≠ C ∧ B ≠ C) →
  C = 1 := by
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l2280_228082


namespace NUMINAMATH_CALUDE_second_car_speed_l2280_228073

/-- Proves that the speed of the second car is 70 km/h given the conditions of the problem -/
theorem second_car_speed (initial_distance : ℝ) (first_car_speed : ℝ) (time : ℝ) :
  initial_distance = 60 →
  first_car_speed = 90 →
  time = 3 →
  ∃ (second_car_speed : ℝ),
    second_car_speed * time + initial_distance = first_car_speed * time ∧
    second_car_speed = 70 :=
by
  sorry


end NUMINAMATH_CALUDE_second_car_speed_l2280_228073


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_b_values_l2280_228074

noncomputable section

def f (a b x : ℝ) : ℝ := (a * Real.log x) / (x + 1) + b / x

def tangent_line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

theorem tangent_line_implies_a_b_values (a b : ℝ) :
  (∀ x, tangent_line x (f a b x)) →
  (tangent_line 1 (f a b 1)) →
  (a = 1 ∧ b = 1) := by sorry

end

end NUMINAMATH_CALUDE_tangent_line_implies_a_b_values_l2280_228074


namespace NUMINAMATH_CALUDE_diana_charge_amount_l2280_228085

/-- The simple interest formula -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

theorem diana_charge_amount :
  ∃ (P : ℝ),
    (P > 0) ∧
    (P < 80.25) ∧
    (P + simple_interest P 0.07 1 = 80.25) ∧
    (abs (P - 75) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_diana_charge_amount_l2280_228085


namespace NUMINAMATH_CALUDE_students_in_both_competitions_l2280_228033

theorem students_in_both_competitions 
  (total_students : ℕ) 
  (math_students : ℕ) 
  (physics_students : ℕ) 
  (no_competition_students : ℕ) 
  (h1 : total_students = 45) 
  (h2 : math_students = 32) 
  (h3 : physics_students = 28) 
  (h4 : no_competition_students = 5) :
  total_students - no_competition_students - 
  (math_students + physics_students - total_students + no_competition_students) = 20 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_competitions_l2280_228033


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l2280_228040

theorem stratified_sampling_problem (total_students : ℕ) (sample_size : ℕ) (major_c_students : ℕ) :
  total_students = 1000 →
  sample_size = 40 →
  major_c_students = 400 →
  (major_c_students * sample_size) / total_students = 16 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l2280_228040


namespace NUMINAMATH_CALUDE_different_winning_scores_l2280_228090

/-- Represents a cross country meet between two teams -/
structure CrossCountryMeet where
  /-- The number of runners in each team -/
  runners_per_team : Nat
  /-- The total number of runners -/
  total_runners : Nat
  /-- The sum of all positions -/
  total_sum : Nat
  /-- The lowest possible winning score -/
  min_winning_score : Nat
  /-- The highest possible winning score -/
  max_winning_score : Nat
  /-- Assertion that there are two teams -/
  two_teams : total_runners = 2 * runners_per_team
  /-- Assertion that the total sum is correct -/
  sum_correct : total_sum = (total_runners * (total_runners + 1)) / 2
  /-- Assertion that the minimum winning score is correct -/
  min_score_correct : min_winning_score = (runners_per_team * (runners_per_team + 1)) / 2
  /-- Assertion that the maximum winning score is less than half the total sum -/
  max_score_correct : max_winning_score = (total_sum / 2) - 1

/-- The main theorem stating the number of different winning scores -/
theorem different_winning_scores (meet : CrossCountryMeet) (h : meet.runners_per_team = 5) :
  (meet.max_winning_score - meet.min_winning_score + 1) = 13 := by
  sorry

end NUMINAMATH_CALUDE_different_winning_scores_l2280_228090


namespace NUMINAMATH_CALUDE_selling_price_ratio_l2280_228088

/-- Given an item with cost price c, prove that the ratio of selling prices y:x is 25:16,
    where x results in a 20% loss and y results in a 25% profit. -/
theorem selling_price_ratio (c x y : ℝ) 
  (loss : x = 0.8 * c)   -- 20% loss condition
  (profit : y = 1.25 * c) -- 25% profit condition
  : y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l2280_228088


namespace NUMINAMATH_CALUDE_factor_expression_l2280_228079

theorem factor_expression (b : ℝ) : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2280_228079


namespace NUMINAMATH_CALUDE_negation_equivalence_l2280_228096

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2280_228096


namespace NUMINAMATH_CALUDE_hash_2_5_3_equals_1_l2280_228025

-- Define the # operation
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Theorem statement
theorem hash_2_5_3_equals_1 : hash 2 5 3 = 1 := by sorry

end NUMINAMATH_CALUDE_hash_2_5_3_equals_1_l2280_228025


namespace NUMINAMATH_CALUDE_z_value_for_given_w_and_v_l2280_228041

/-- Given a relationship between z, w, and v, prove that z equals 7.5 when w = 4 and v = 8 -/
theorem z_value_for_given_w_and_v (k : ℝ) :
  (3 * 15 = k * 4 / 2^2) →  -- Initial condition
  (∀ z w v : ℝ, 3 * z = k * v / w^2) →  -- General relationship
  ∃ z : ℝ, (3 * z = k * 8 / 4^2) ∧ z = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_z_value_for_given_w_and_v_l2280_228041


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2280_228097

theorem polynomial_simplification (q : ℝ) :
  (5 * q^3 - 7 * q + 8) + (3 - 9 * q^2 + 3 * q) = 5 * q^3 - 9 * q^2 - 4 * q + 11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2280_228097


namespace NUMINAMATH_CALUDE_goods_transportable_l2280_228094

-- Define the problem parameters
def total_weight : ℝ := 13.5
def max_package_weight : ℝ := 0.35
def num_trucks : ℕ := 11
def truck_capacity : ℝ := 1.5

-- Theorem statement
theorem goods_transportable :
  total_weight ≤ (num_trucks : ℝ) * truck_capacity ∧
  ∃ (num_packages : ℕ), (num_packages : ℝ) * max_package_weight ≥ total_weight :=
by sorry

end NUMINAMATH_CALUDE_goods_transportable_l2280_228094


namespace NUMINAMATH_CALUDE_unique_three_digit_pair_l2280_228048

theorem unique_three_digit_pair : 
  ∃! (a b : ℕ), 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 ∧ 1000 * a + b = 7 * a * b :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_pair_l2280_228048


namespace NUMINAMATH_CALUDE_variety_show_theorem_l2280_228080

/-- Represents the number of acts in the variety show -/
def num_acts : ℕ := 7

/-- Represents the number of acts with adjacency restrictions -/
def num_restricted_acts : ℕ := 3

/-- Represents the number of acts without adjacency restrictions -/
def num_unrestricted_acts : ℕ := num_acts - num_restricted_acts

/-- Represents the number of spaces available for restricted acts -/
def num_spaces : ℕ := num_unrestricted_acts + 1

/-- The number of ways to arrange the variety show program -/
def variety_show_arrangements : ℕ :=
  (num_spaces.choose num_restricted_acts) * 
  (Nat.factorial num_restricted_acts) * 
  (Nat.factorial num_unrestricted_acts)

theorem variety_show_theorem : 
  variety_show_arrangements = 1440 := by
  sorry

end NUMINAMATH_CALUDE_variety_show_theorem_l2280_228080


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersect_A_C_case1_intersect_A_C_case2_intersect_A_C_case3_l2280_228069

open Set Real

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | 1 ≤ x ∧ x < 10} := by sorry

-- Theorem for (ℂR A) ∩ B
theorem complement_A_intersect_B : (Aᶜ) ∩ B = {x | 7 ≤ x ∧ x < 10} := by sorry

-- Theorems for A ∩ C in different cases
theorem intersect_A_C_case1 (a : ℝ) (h : a ≤ 1) : A ∩ C a = ∅ := by sorry

theorem intersect_A_C_case2 (a : ℝ) (h : 1 < a ∧ a ≤ 7) : 
  A ∩ C a = {x | 1 ≤ x ∧ x < a} := by sorry

theorem intersect_A_C_case3 (a : ℝ) (h : 7 < a) : 
  A ∩ C a = {x | 1 ≤ x ∧ x < 7} := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersect_A_C_case1_intersect_A_C_case2_intersect_A_C_case3_l2280_228069


namespace NUMINAMATH_CALUDE_prime_between_squares_l2280_228036

theorem prime_between_squares : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  ∃ n : ℕ, n^2 = p - 9 ∧ (n+1)^2 = p + 8 := by
  sorry

end NUMINAMATH_CALUDE_prime_between_squares_l2280_228036


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2280_228002

theorem smallest_number_with_given_remainders : ∃! n : ℕ,
  (∀ m : ℕ, m < n → ¬(m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5)) ∧
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n % 6 = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2280_228002


namespace NUMINAMATH_CALUDE_y_intercept_distance_of_intersecting_lines_l2280_228031

/-- A line in 2D space represented by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Calculate the y-intercept of a line -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- The distance between two real numbers -/
def distance (a b : ℝ) : ℝ :=
  |a - b|

theorem y_intercept_distance_of_intersecting_lines :
  let l1 : Line := { slope := -2, point := (8, 20) }
  let l2 : Line := { slope := 4, point := (8, 20) }
  distance (y_intercept l1) (y_intercept l2) = 68 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_distance_of_intersecting_lines_l2280_228031


namespace NUMINAMATH_CALUDE_rectangular_playground_area_l2280_228059

theorem rectangular_playground_area : 
  ∀ (length width : ℝ),
  length > 0 ∧ width > 0 →
  2 * (length + width) = 84 →
  length = 3 * width →
  length * width = 330.75 := by
sorry

end NUMINAMATH_CALUDE_rectangular_playground_area_l2280_228059


namespace NUMINAMATH_CALUDE_range_of_a_l2280_228027

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x + a ≥ 0) → a ≥ -8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2280_228027


namespace NUMINAMATH_CALUDE_xyz_square_equality_implies_zero_l2280_228028

theorem xyz_square_equality_implies_zero (x y z : ℤ) : 
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

-- Note: The second part of the original problem doesn't have a definitive answer,
-- so we'll omit it from the Lean statement.

end NUMINAMATH_CALUDE_xyz_square_equality_implies_zero_l2280_228028


namespace NUMINAMATH_CALUDE_can_repair_propeller_l2280_228046

/-- Represents the cost of a blade in tugriks -/
def blade_cost : ℕ := 120

/-- Represents the cost of a screw in tugriks -/
def screw_cost : ℕ := 9

/-- Represents the discount threshold in tugriks -/
def discount_threshold : ℕ := 250

/-- Represents the discount rate as a percentage -/
def discount_rate : ℚ := 20 / 100

/-- Represents Karlson's budget in tugriks -/
def budget : ℕ := 360

/-- Calculates the discounted price of an item -/
def apply_discount (price : ℕ) : ℚ :=
  (1 - discount_rate) * price

/-- Theorem stating that Karlson can repair his propeller with his budget -/
theorem can_repair_propeller : ∃ (first_purchase second_purchase : ℕ),
  first_purchase ≥ discount_threshold ∧
  first_purchase + second_purchase ≤ budget ∧
  first_purchase = 2 * blade_cost + 2 * screw_cost ∧
  second_purchase = apply_discount blade_cost :=
sorry

end NUMINAMATH_CALUDE_can_repair_propeller_l2280_228046


namespace NUMINAMATH_CALUDE_speed_time_distance_return_trip_time_l2280_228004

/-- The distance to Yinping Mountain in kilometers -/
def distance : ℝ := 240

/-- The speed of the car in km/h -/
def speed (v : ℝ) : ℝ := v

/-- The time taken for the trip in hours -/
def time (t : ℝ) : ℝ := t

/-- The relationship between distance, speed, and time -/
theorem speed_time_distance (v t : ℝ) (h : t > 0) :
  speed v * time t = distance → v = distance / t :=
sorry

/-- The time taken for the return trip at 60 km/h -/
theorem return_trip_time :
  ∃ t : ℝ, t > 0 ∧ speed 60 * time t = distance ∧ t = 4 :=
sorry

end NUMINAMATH_CALUDE_speed_time_distance_return_trip_time_l2280_228004


namespace NUMINAMATH_CALUDE_radical_expression_equality_l2280_228016

theorem radical_expression_equality : 
  (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) - (Real.sqrt 3 - Real.sqrt 2)^2 = 2 * Real.sqrt 6 - 2 := by
  sorry

end NUMINAMATH_CALUDE_radical_expression_equality_l2280_228016


namespace NUMINAMATH_CALUDE_largest_product_is_15_l2280_228015

def numbers : List ℤ := [2, -3, 4, -5]

theorem largest_product_is_15 : 
  (List.map (fun x => List.map (fun y => x * y) numbers) numbers).join.maximum? = some 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_product_is_15_l2280_228015


namespace NUMINAMATH_CALUDE_fraction_inequality_l2280_228081

theorem fraction_inequality (a b c d : ℕ+) 
  (h1 : a + c ≤ 1982)
  (h2 : (a : ℚ) / b + (c : ℚ) / d < 1) :
  1 - (a : ℚ) / b - (c : ℚ) / d > 1 / (1983 ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2280_228081


namespace NUMINAMATH_CALUDE_garage_wheels_l2280_228000

/-- The number of wheels in a garage with bicycles and cars -/
def total_wheels (num_bicycles : ℕ) (num_cars : ℕ) : ℕ :=
  num_bicycles * 2 + num_cars * 4

/-- Theorem: The total number of wheels in the garage is 82 -/
theorem garage_wheels :
  total_wheels 9 16 = 82 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheels_l2280_228000


namespace NUMINAMATH_CALUDE_root_in_interval_l2280_228018

noncomputable def f (x : ℝ) := Real.exp x - x - 2

theorem root_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_root_in_interval_l2280_228018


namespace NUMINAMATH_CALUDE_min_players_on_team_l2280_228053

theorem min_players_on_team (total_score : ℕ) (min_score max_score : ℕ) : 
  total_score = 100 →
  min_score = 7 →
  max_score = 23 →
  (∃ (num_players : ℕ), 
    num_players ≥ 1 ∧
    (∀ (player_scores : List ℕ), 
      player_scores.length = num_players →
      (∀ score ∈ player_scores, min_score ≤ score ∧ score ≤ max_score) →
      player_scores.sum = total_score) ∧
    (∀ (n : ℕ), n < num_players →
      ¬∃ (player_scores : List ℕ),
        player_scores.length = n ∧
        (∀ score ∈ player_scores, min_score ≤ score ∧ score ≤ max_score) ∧
        player_scores.sum = total_score)) →
  (∃ (num_players : ℕ), num_players = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_min_players_on_team_l2280_228053


namespace NUMINAMATH_CALUDE_fourth_day_temperature_l2280_228022

theorem fourth_day_temperature
  (temp1 temp2 temp3 : ℤ)
  (avg_temp : ℚ)
  (h1 : temp1 = -36)
  (h2 : temp2 = 13)
  (h3 : temp3 = -15)
  (h4 : avg_temp = -12)
  (h5 : (temp1 + temp2 + temp3 + temp4 : ℚ) / 4 = avg_temp) :
  temp4 = -10 :=
sorry

end NUMINAMATH_CALUDE_fourth_day_temperature_l2280_228022


namespace NUMINAMATH_CALUDE_relationship_abcd_l2280_228062

theorem relationship_abcd (a b c d : ℝ) 
  (h : (a + 2*b) / (b + 2*c) = (c + 2*d) / (d + 2*a)) :
  b = 2*a ∨ a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_relationship_abcd_l2280_228062


namespace NUMINAMATH_CALUDE_dog_walking_problem_l2280_228019

/-- Greg's dog walking business problem -/
theorem dog_walking_problem (x : ℕ) : 
  (20 + x) +                 -- Cost for one dog
  (2 * 20 + 2 * 7 * 1) +     -- Cost for two dogs for 7 minutes
  (3 * 20 + 3 * 9 * 1) = 171 -- Cost for three dogs for 9 minutes
  → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_dog_walking_problem_l2280_228019


namespace NUMINAMATH_CALUDE_probability_two_nondefective_pens_l2280_228009

/-- Given a box of 12 pens with 3 defective pens, prove that the probability
    of selecting 2 non-defective pens at random without replacement is 6/11. -/
theorem probability_two_nondefective_pens (total_pens : Nat) (defective_pens : Nat)
    (h1 : total_pens = 12)
    (h2 : defective_pens = 3) :
    (total_pens - defective_pens : ℚ) / total_pens *
    ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_nondefective_pens_l2280_228009


namespace NUMINAMATH_CALUDE_lcm_problem_l2280_228089

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 36 m = 108) (h2 : Nat.lcm m 45 = 180) : m = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2280_228089


namespace NUMINAMATH_CALUDE_remainder_equivalence_l2280_228014

theorem remainder_equivalence (N : ℤ) (k : ℤ) : 
  N % 18 = 19 → N % 242 = (18 * k + 19) % 242 := by
  sorry

end NUMINAMATH_CALUDE_remainder_equivalence_l2280_228014


namespace NUMINAMATH_CALUDE_pedestrian_meets_sixteen_buses_l2280_228047

/-- Represents the problem of a pedestrian meeting buses on a road --/
structure BusMeetingProblem where
  road_length : ℝ
  bus_speed : ℝ
  bus_interval : ℝ
  pedestrian_start_time : ℝ
  pedestrian_speed : ℝ

/-- Calculates the number of buses the pedestrian meets --/
def count_bus_meetings (problem : BusMeetingProblem) : ℕ :=
  sorry

/-- The main theorem stating that the pedestrian meets 16 buses --/
theorem pedestrian_meets_sixteen_buses :
  let problem : BusMeetingProblem := {
    road_length := 8,
    bus_speed := 12,
    bus_interval := 1/6,  -- 10 minutes in hours
    pedestrian_start_time := 81/4,  -- 8:15 AM in hours since midnight
    pedestrian_speed := 4
  }
  count_bus_meetings problem = 16 := by
  sorry

end NUMINAMATH_CALUDE_pedestrian_meets_sixteen_buses_l2280_228047


namespace NUMINAMATH_CALUDE_quadratic_sets_equal_or_disjoint_l2280_228042

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The set of f(2n) where n is an integer -/
def M (f : ℝ → ℝ) : Set ℝ := {y | ∃ n : ℤ, y = f (2 * ↑n)}

/-- The set of f(2n+1) where n is an integer -/
def N (f : ℝ → ℝ) : Set ℝ := {y | ∃ n : ℤ, y = f (2 * ↑n + 1)}

/-- Theorem: For any quadratic function, M and N are either equal or disjoint -/
theorem quadratic_sets_equal_or_disjoint (a b c : ℝ) :
  let f := QuadraticFunction a b c
  (M f = N f) ∨ (M f ∩ N f = ∅) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sets_equal_or_disjoint_l2280_228042


namespace NUMINAMATH_CALUDE_area_APRQ_is_6_25_l2280_228098

/-- A rectangle with points P, Q, and R located on its sides. -/
structure RectangleWithPoints where
  /-- The area of rectangle ABCD -/
  area : ℝ
  /-- Point P is located one-fourth the length of side AD from vertex A -/
  p_location : ℝ
  /-- Point Q is located one-fourth the length of side CD from vertex C -/
  q_location : ℝ
  /-- Point R is located one-fourth the length of side BC from vertex B -/
  r_location : ℝ

/-- The area of quadrilateral APRQ in a rectangle with given properties -/
def area_APRQ (rect : RectangleWithPoints) : ℝ := sorry

/-- Theorem stating that the area of APRQ is 6.25 square meters -/
theorem area_APRQ_is_6_25 (rect : RectangleWithPoints) 
  (h1 : rect.area = 100)
  (h2 : rect.p_location = 1/4)
  (h3 : rect.q_location = 1/4)
  (h4 : rect.r_location = 1/4) : 
  area_APRQ rect = 6.25 := by sorry

end NUMINAMATH_CALUDE_area_APRQ_is_6_25_l2280_228098


namespace NUMINAMATH_CALUDE_rectangle_area_l2280_228077

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 186) : L * B = 2030 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2280_228077


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l2280_228021

theorem positive_numbers_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ∧
  (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ≤ a^3/(b*c) + b^3/(c*a) + c^3/(a*b) :=
by sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l2280_228021


namespace NUMINAMATH_CALUDE_digit_sum_divisibility_27_l2280_228083

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem digit_sum_divisibility_27 : 
  ∃ n : ℕ, (sum_of_digits n % 27 = 0) ∧ (n % 27 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_digit_sum_divisibility_27_l2280_228083


namespace NUMINAMATH_CALUDE_intersection_A_B_l2280_228091

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | (x + 4) * (x - 1) < 0}
def B : Set ℝ := {x : ℝ | x^2 - 2*x = 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2280_228091


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2280_228020

def M : Set Nat := {1, 3, 5, 7}
def N : Set Nat := {5, 6, 7}

theorem intersection_of_M_and_N : M ∩ N = {5, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2280_228020


namespace NUMINAMATH_CALUDE_B_pow_101_eq_B_l2280_228095

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_pow_101_eq_B : B^101 = B := by sorry

end NUMINAMATH_CALUDE_B_pow_101_eq_B_l2280_228095


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2280_228058

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) := {(-4, 0), (4, 0)}

-- Define the eccentricity
def eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola x y → (x, y) ∈ foci ∨ (∀ f ∈ foci, (x - f.1)^2 + y^2 > 0)) ∧
  (∀ x y, hyperbola x y → 
    let a := 2  -- sqrt(4)
    let c := 4  -- distance from center to focus
    c / a = eccentricity) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2280_228058


namespace NUMINAMATH_CALUDE_expression_evaluation_l2280_228099

theorem expression_evaluation : 16^3 + 3*(16^2) + 3*16 + 1 = 4913 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2280_228099


namespace NUMINAMATH_CALUDE_ellipse_and_circle_theorem_l2280_228049

/-- Definition of the ellipse E -/
def is_ellipse (E : Set (ℝ × ℝ)) (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ ∀ x y, (x, y) ∈ E ↔ x^2 / a^2 + y^2 / b^2 = 1

/-- E passes through the points (2, √2) and (√6, 1) -/
def passes_through_points (E : Set (ℝ × ℝ)) : Prop :=
  (2, Real.sqrt 2) ∈ E ∧ (Real.sqrt 6, 1) ∈ E

/-- Definition of perpendicular vectors -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Main theorem -/
theorem ellipse_and_circle_theorem (E : Set (ℝ × ℝ)) (a b : ℝ) 
  (h_ellipse : is_ellipse E a b) (h_points : passes_through_points E) :
  (∃ r : ℝ, r > 0 ∧
    (∀ x y, (x, y) ∈ E ↔ x^2 / 8 + y^2 / 4 = 1) ∧
    (∀ k m : ℝ,
      (∃ A B : ℝ × ℝ,
        A ∈ E ∧ B ∈ E ∧
        A.2 = k * A.1 + m ∧
        B.2 = k * B.1 + m ∧
        perpendicular A B ∧
        A.1^2 + A.2^2 = r^2 ∧
        B.1^2 + B.2^2 = r^2) ↔
      k^2 + 1 = (8 / 3) / m^2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_theorem_l2280_228049


namespace NUMINAMATH_CALUDE_arithmetic_log_implies_square_product_converse_not_always_true_l2280_228029

-- Define a predicate for arithmetic sequence of logarithms
def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  ∃ (a d : ℝ), (Real.log x = a) ∧ (Real.log y = a + d) ∧ (Real.log z = a + 2*d)

-- Define the theorem
theorem arithmetic_log_implies_square_product (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  is_arithmetic_sequence x y z → y^2 = x*z :=
by sorry

-- Define a counterexample to show the converse is not necessarily true
theorem converse_not_always_true :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ y^2 = x*z ∧ ¬(is_arithmetic_sequence x y z) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_log_implies_square_product_converse_not_always_true_l2280_228029


namespace NUMINAMATH_CALUDE_specific_ellipse_sum_l2280_228072

/-- Represents an ellipse in a 2D Cartesian coordinate system -/
structure Ellipse where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  a : ℝ  -- length of the semi-major axis
  b : ℝ  -- length of the semi-minor axis

/-- The sum of center coordinates and axis lengths for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

/-- Theorem: For an ellipse with center (3, -5), horizontal semi-major axis 6, and vertical semi-minor axis 2, the sum h + k + a + b equals 6 -/
theorem specific_ellipse_sum :
  ∃ (e : Ellipse), e.h = 3 ∧ e.k = -5 ∧ e.a = 6 ∧ e.b = 2 ∧ ellipse_sum e = 6 := by
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_sum_l2280_228072


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2280_228006

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2280_228006


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2280_228032

theorem pure_imaginary_complex_number (m : ℝ) : 
  (((m^2 - 5*m + 6) : ℂ) + (m^2 - 3*m)*I = (0 : ℂ) + ((m^2 - 3*m) : ℝ)*I) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2280_228032


namespace NUMINAMATH_CALUDE_count_tree_frogs_l2280_228066

theorem count_tree_frogs (total_frogs poison_frogs wood_frogs : ℕ) 
  (h1 : total_frogs = 78)
  (h2 : poison_frogs = 10)
  (h3 : wood_frogs = 13)
  (h4 : ∃ tree_frogs : ℕ, total_frogs = tree_frogs + poison_frogs + wood_frogs) :
  ∃ tree_frogs : ℕ, tree_frogs = 55 ∧ total_frogs = tree_frogs + poison_frogs + wood_frogs :=
by
  sorry

end NUMINAMATH_CALUDE_count_tree_frogs_l2280_228066


namespace NUMINAMATH_CALUDE_min_value_theorem_l2280_228093

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (4 * x^2) / (y + 1) + y^2 / (2 * x + 2) ≥ 4/5 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2280_228093


namespace NUMINAMATH_CALUDE_roden_fish_purchase_cost_l2280_228038

/-- Calculate the total cost of Roden's fish purchase -/
theorem roden_fish_purchase_cost : 
  let goldfish_cost : ℕ := 15 * 3
  let blue_fish_cost : ℕ := 7 * 6
  let neon_tetra_cost : ℕ := 10 * 2
  let angelfish_cost : ℕ := 5 * 8
  let total_cost : ℕ := goldfish_cost + blue_fish_cost + neon_tetra_cost + angelfish_cost
  total_cost = 147 := by
  sorry

end NUMINAMATH_CALUDE_roden_fish_purchase_cost_l2280_228038


namespace NUMINAMATH_CALUDE_davidsons_class_as_l2280_228070

/-- Proves that given the conditions of the problem, 12 students in Mr. Davidson's class received an 'A' -/
theorem davidsons_class_as (carter_total : ℕ) (carter_as : ℕ) (davidson_total : ℕ) :
  carter_total = 20 →
  carter_as = 8 →
  davidson_total = 30 →
  ∃ davidson_as : ℕ,
    davidson_as * carter_total = carter_as * davidson_total ∧
    davidson_as = 12 :=
by sorry

end NUMINAMATH_CALUDE_davidsons_class_as_l2280_228070


namespace NUMINAMATH_CALUDE_angle_half_in_third_quadrant_l2280_228063

/-- Given an angle α in the second quadrant with |cos(α/2)| = -cos(α/2),
    prove that α/2 is in the third quadrant. -/
theorem angle_half_in_third_quadrant (α : Real) :
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (|Real.cos (α/2)| = -Real.cos (α/2)) →  -- |cos(α/2)| = -cos(α/2)
  (π < α/2 ∧ α/2 < 3*π/2) :=  -- α/2 is in the third quadrant
by sorry

end NUMINAMATH_CALUDE_angle_half_in_third_quadrant_l2280_228063


namespace NUMINAMATH_CALUDE_equation_solution_l2280_228076

theorem equation_solution : ∃ (x₁ x₂ : ℚ), 
  (x₁ = 1/9 ∧ x₂ = 1/18) ∧ 
  (∀ x : ℚ, (101*x^2 - 18*x + 1)^2 - 121*x^2*(101*x^2 - 18*x + 1) + 2020*x^4 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2280_228076


namespace NUMINAMATH_CALUDE_solve_coloring_book_problem_l2280_228055

def coloring_book_problem (book1 book2 book3 colored : ℕ) : Prop :=
  let total := book1 + book2 + book3
  total - colored = 53

theorem solve_coloring_book_problem :
  coloring_book_problem 35 45 40 67 := by
  sorry

end NUMINAMATH_CALUDE_solve_coloring_book_problem_l2280_228055


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l2280_228008

/-- The number of ways to place 4 different balls into 4 numbered boxes with exactly one empty box -/
def ball_placement_count : ℕ := 144

/-- The number of different balls -/
def num_balls : ℕ := 4

/-- The number of boxes -/
def num_boxes : ℕ := 4

theorem ball_placement_theorem :
  (num_balls = 4) →
  (num_boxes = 4) →
  (ball_placement_count = 144) :=
by sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l2280_228008


namespace NUMINAMATH_CALUDE_profit_percent_l2280_228013

/-- Given an article with cost price C and selling price P, 
    where selling at 2/3 of P results in a 14% loss,
    prove that selling at P results in a 29% profit -/
theorem profit_percent (C P : ℝ) (h : (2/3) * P = 0.86 * C) :
  (P - C) / C * 100 = 29 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_l2280_228013


namespace NUMINAMATH_CALUDE_sum_six_terms_eq_neg_24_l2280_228054

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  first_term : a 1 = 1
  common_diff : ∀ n : ℕ, a (n + 1) = a n + d
  d_nonzero : d ≠ 0
  geometric_subseq : (a 3 / a 2) = (a 6 / a 3)

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- The main theorem -/
theorem sum_six_terms_eq_neg_24 (seq : ArithmeticSequence) :
  sum_n_terms seq 6 = -24 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_terms_eq_neg_24_l2280_228054


namespace NUMINAMATH_CALUDE_hazel_received_six_l2280_228075

/-- The number of shirts Hazel received -/
def hazel_shirts : ℕ := sorry

/-- The number of shirts Razel received -/
def razel_shirts : ℕ := sorry

/-- The total number of shirts Hazel and Razel have -/
def total_shirts : ℕ := 18

/-- Razel received twice the number of shirts as Hazel -/
axiom razel_twice_hazel : razel_shirts = 2 * hazel_shirts

/-- The total number of shirts is the sum of Hazel's and Razel's shirts -/
axiom total_is_sum : total_shirts = hazel_shirts + razel_shirts

/-- Theorem: Hazel received 6 shirts -/
theorem hazel_received_six : hazel_shirts = 6 := by sorry

end NUMINAMATH_CALUDE_hazel_received_six_l2280_228075


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2280_228052

theorem complex_equation_solution (z : ℂ) : 
  z * (1 - 2*I) = 3 + 2*I → z = -1/5 + 8/5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2280_228052


namespace NUMINAMATH_CALUDE_max_salary_is_260000_l2280_228030

/-- Represents the maximum possible salary for a single player on a minor league soccer team -/
def max_player_salary (n : ℕ) (min_salary : ℕ) (total_cap : ℕ) : ℕ :=
  total_cap - (n - 1) * min_salary

/-- Theorem stating the maximum possible salary for a single player on the team -/
theorem max_salary_is_260000 :
  max_player_salary 18 20000 600000 = 260000 := by
  sorry

#eval max_player_salary 18 20000 600000

end NUMINAMATH_CALUDE_max_salary_is_260000_l2280_228030


namespace NUMINAMATH_CALUDE_middle_number_calculation_l2280_228050

theorem middle_number_calculation (n : ℕ) (total_avg first_avg last_avg : ℚ) : 
  n = 11 →
  total_avg = 9.9 →
  first_avg = 10.5 →
  last_avg = 11.4 →
  ∃ (middle : ℚ), 
    middle = 22.5 ∧
    n * total_avg = (n / 2 : ℚ) * first_avg + (n / 2 : ℚ) * last_avg - middle :=
by
  sorry

end NUMINAMATH_CALUDE_middle_number_calculation_l2280_228050


namespace NUMINAMATH_CALUDE_pension_calculation_l2280_228065

/-- Given a pension system where:
  * The annual pension is proportional to the square root of years served
  * Serving 'a' additional years increases the pension by 'p' dollars
  * Serving 'b' additional years (b ≠ a) increases the pension by 'q' dollars
This theorem proves that the annual pension can be expressed in terms of a, b, p, and q. -/
theorem pension_calculation (a b p q : ℝ) (h_ab : a ≠ b) :
  ∃ (x y k : ℝ),
    x = k * Real.sqrt y ∧
    x + p = k * Real.sqrt (y + a) ∧
    x + q = k * Real.sqrt (y + b) →
    x = (a * q^2 - b * p^2) / (2 * (b * p - a * q)) :=
sorry

end NUMINAMATH_CALUDE_pension_calculation_l2280_228065


namespace NUMINAMATH_CALUDE_range_of_m_l2280_228026

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ 2}
def B (m : ℝ) : Set ℝ := {x : ℝ | x ≥ m}

-- State the theorem
theorem range_of_m (m : ℝ) (h : A ∪ B m = A) : m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2280_228026


namespace NUMINAMATH_CALUDE_f_intersects_y_axis_l2280_228043

-- Define the function f(x) = 4x - 4
def f (x : ℝ) : ℝ := 4 * x - 4

-- Theorem: f intersects the y-axis at (0, -4)
theorem f_intersects_y_axis :
  f 0 = -4 := by sorry

end NUMINAMATH_CALUDE_f_intersects_y_axis_l2280_228043


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l2280_228034

/-- A geometric sequence with first term a₁ and common ratio q -/
def GeometricSequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := fun n ↦ a₁ * q^(n - 1)

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

/-- The condition "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem geometric_sequence_increasing_condition (a₁ q : ℝ) :
  ¬(((q > 1) → IncreasingSequence (GeometricSequence a₁ q)) ∧
    (IncreasingSequence (GeometricSequence a₁ q) → (q > 1))) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l2280_228034


namespace NUMINAMATH_CALUDE_book_store_inventory_l2280_228003

theorem book_store_inventory (initial_books : ℝ) (first_addition : ℝ) (second_addition : ℝ) :
  initial_books = 41.0 →
  first_addition = 33.0 →
  second_addition = 2.0 →
  initial_books + first_addition + second_addition = 76.0 := by
  sorry

end NUMINAMATH_CALUDE_book_store_inventory_l2280_228003


namespace NUMINAMATH_CALUDE_max_acute_angles_non_convex_polygon_l2280_228024

theorem max_acute_angles_non_convex_polygon (n : ℕ) (h : n ≥ 3) :
  let sum_interior_angles := (n - 2) * 180
  let max_acute_angles := (2 * n) / 3 + 1
  ∃ k : ℕ, k ≤ max_acute_angles ∧
    k * 90 + (n - k) * 360 < sum_interior_angles ∧
    ∀ m : ℕ, m > k → m * 90 + (n - m) * 360 ≥ sum_interior_angles :=
by sorry

end NUMINAMATH_CALUDE_max_acute_angles_non_convex_polygon_l2280_228024


namespace NUMINAMATH_CALUDE_pet_store_kittens_l2280_228084

/-- The total number of kittens after receiving more -/
def total_kittens (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: If a pet store initially has 6 kittens and receives 3 more, 
    the total number of kittens will be 9 -/
theorem pet_store_kittens : total_kittens 6 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_kittens_l2280_228084


namespace NUMINAMATH_CALUDE_cake_eating_ratio_l2280_228092

theorem cake_eating_ratio (cake_weight : ℝ) (parts : ℕ) (pierre_ate : ℝ) : 
  cake_weight = 400 →
  parts = 8 →
  pierre_ate = 100 →
  (pierre_ate / (cake_weight / parts.cast)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cake_eating_ratio_l2280_228092


namespace NUMINAMATH_CALUDE_trapezoid_median_theorem_median_is_six_l2280_228017

/-- The length of the median of a trapezoid -/
def median_length : ℝ := 6

/-- The length of the longer base of the trapezoid -/
def longer_base : ℝ := 1.5 * median_length

/-- The length of the shorter base of the trapezoid -/
def shorter_base : ℝ := median_length - 3

/-- Theorem: The median of a trapezoid is the average of its bases -/
theorem trapezoid_median_theorem (median : ℝ) (longer_base shorter_base : ℝ) 
  (h1 : longer_base = 1.5 * median) 
  (h2 : shorter_base = median - 3) : 
  median = (longer_base + shorter_base) / 2 := by sorry

/-- Proof that the median length is 6 units -/
theorem median_is_six : 
  median_length = 6 ∧ 
  longer_base = 1.5 * median_length ∧ 
  shorter_base = median_length - 3 ∧
  median_length = (longer_base + shorter_base) / 2 := by sorry

end NUMINAMATH_CALUDE_trapezoid_median_theorem_median_is_six_l2280_228017


namespace NUMINAMATH_CALUDE_min_value_of_f_l2280_228087

/-- The quadratic function f(x) = 2x^2 - 16x + 22 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 16 * x + 22

/-- Theorem: The minimum value of f(x) = 2x^2 - 16x + 22 is -10 -/
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ -10 ∧ ∃ x₀ : ℝ, f x₀ = -10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2280_228087


namespace NUMINAMATH_CALUDE_calculate_product_l2280_228044

theorem calculate_product : 150 * 22.5 * (1.5^2) * 10 = 75937.5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_product_l2280_228044


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2280_228068

theorem fractional_equation_solution : 
  ∃ x : ℝ, (x ≠ 0 ∧ x ≠ -1) ∧ (6 / (x + 1) = (x + 5) / (x * (x + 1))) ∧ x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2280_228068


namespace NUMINAMATH_CALUDE_jeremy_watermelon_consumption_l2280_228067

/-- The number of watermelons Jeremy eats per week, given the total number of watermelons,
    the number of weeks they last, and the number given away each week. -/
def watermelons_eaten_per_week (total : ℕ) (weeks : ℕ) (given_away_per_week : ℕ) : ℕ :=
  (total - weeks * given_away_per_week) / weeks

/-- Theorem stating that given 30 watermelons lasting 6 weeks, 
    with 2 given away each week, Jeremy eats 3 watermelons per week. -/
theorem jeremy_watermelon_consumption :
  watermelons_eaten_per_week 30 6 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_watermelon_consumption_l2280_228067


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l2280_228086

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) : 
  a * b = 10 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l2280_228086


namespace NUMINAMATH_CALUDE_emily_team_size_l2280_228078

/-- The number of players on Emily's team -/
def num_players : ℕ := 9

/-- The total points scored by the team -/
def total_points : ℕ := 39

/-- The points scored by Emily -/
def emily_points : ℕ := 23

/-- The points scored by each other player -/
def other_player_points : ℕ := 2

/-- Theorem stating that the number of players on Emily's team is correct -/
theorem emily_team_size :
  num_players = (total_points - emily_points) / other_player_points + 1 := by
  sorry


end NUMINAMATH_CALUDE_emily_team_size_l2280_228078


namespace NUMINAMATH_CALUDE_parallel_line_equation_l2280_228057

/-- Given a triangle ABC with vertices A(4,0), B(8,10), and C(0,6),
    the equation of the line passing through A and parallel to BC is x - 2y - 4 = 0 -/
theorem parallel_line_equation (A B C : ℝ × ℝ) : 
  A = (4, 0) → B = (8, 10) → C = (0, 6) → 
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    (x = 4 ∧ y = 0) ∨ (y - 0 = m * (x - 4)) ↔ x - 2*y - 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l2280_228057
