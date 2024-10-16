import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_property_l700_70096

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n, S n = n * (a 0 + a (n - 1)) / 2

/-- Theorem: If S_6 / S_3 = 3 for an arithmetic sequence, then S_12 / S_9 = 5/3 -/
theorem arithmetic_sequence_ratio_property (seq : ArithmeticSequence) 
  (h : seq.S 6 / seq.S 3 = 3) : seq.S 12 / seq.S 9 = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_property_l700_70096


namespace NUMINAMATH_CALUDE_vector_subtraction_l700_70022

theorem vector_subtraction (a b : ℝ × ℝ) : 
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l700_70022


namespace NUMINAMATH_CALUDE_frog_jumped_farther_l700_70014

/-- The frog's jump distance in inches -/
def frog_jump : ℕ := 39

/-- The grasshopper's jump distance in inches -/
def grasshopper_jump : ℕ := 17

/-- The difference in jump distance between the frog and the grasshopper -/
def jump_difference : ℕ := frog_jump - grasshopper_jump

theorem frog_jumped_farther : jump_difference = 22 := by
  sorry

end NUMINAMATH_CALUDE_frog_jumped_farther_l700_70014


namespace NUMINAMATH_CALUDE_ant_problem_l700_70044

theorem ant_problem (abe_ants cece_ants duke_ants beth_ants : ℕ) 
  (total_ants : ℕ) (beth_percentage : ℚ) :
  abe_ants = 4 →
  cece_ants = 2 * abe_ants →
  duke_ants = abe_ants / 2 →
  beth_ants = abe_ants + (beth_percentage / 100) * abe_ants →
  total_ants = abe_ants + beth_ants + cece_ants + duke_ants →
  total_ants = 20 →
  beth_percentage = 50 := by
sorry

end NUMINAMATH_CALUDE_ant_problem_l700_70044


namespace NUMINAMATH_CALUDE_simplify_expression_l700_70049

theorem simplify_expression (x : ℝ) : 120 * x - 52 * x = 68 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l700_70049


namespace NUMINAMATH_CALUDE_area_of_triangle_area_value_l700_70041

theorem area_of_triangle : ℝ → Prop :=
  fun area =>
    ∃ (line1 line2 : ℝ → ℝ → Prop) (x_axis : ℝ → ℝ → Prop),
      (∀ x y, line1 x y ↔ y = x) ∧
      (∀ x y, line2 x y ↔ x = -7) ∧
      (∀ x y, x_axis x y ↔ y = 0) ∧
      (∃ x y, line1 x y ∧ line2 x y) ∧
      (let base := 7
       let height := 7
       area = (1/2) * base * height)

theorem area_value : area_of_triangle 24.5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_area_value_l700_70041


namespace NUMINAMATH_CALUDE_taxi_fare_equality_l700_70075

/-- Taxi fare calculation and comparison -/
theorem taxi_fare_equality (mike_miles : ℝ) : 
  (2.5 + 0.25 * mike_miles = 2.5 + 5 + 0.25 * 26) → mike_miles = 46 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_equality_l700_70075


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l700_70087

/-- The volume of a cylinder obtained by rotating a rectangle about its longer side -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (length_positive : 0 < length) (width_positive : 0 < width) (length_longer : width ≤ length) :
  let radius := width / 2
  let height := length
  let volume := π * radius^2 * height
  (length = 10 ∧ width = 8) → volume = 160 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l700_70087


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l700_70046

theorem sqrt_sum_problem (x : ℝ) : 
  Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4 → 
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l700_70046


namespace NUMINAMATH_CALUDE_solution_exists_l700_70092

theorem solution_exists (x : ℝ) (h : x = 5) : ∃ some_number : ℝ, (x / 5) + some_number = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l700_70092


namespace NUMINAMATH_CALUDE_sum_of_digits_of_k_l700_70069

def k : ℕ := 10^30 - 36

-- Function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_k : sum_of_digits k = 262 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_k_l700_70069


namespace NUMINAMATH_CALUDE_arthurs_dinner_cost_l700_70095

/-- Calculate the total cost of Arthur's dinner --/
theorem arthurs_dinner_cost :
  let appetizer_cost : ℚ := 8
  let steak_cost : ℚ := 20
  let wine_cost : ℚ := 3
  let dessert_cost : ℚ := 6
  let wine_glasses : ℕ := 2
  let voucher_discount : ℚ := 1/2
  let tip_percentage : ℚ := 1/5

  let full_meal_cost : ℚ := appetizer_cost + steak_cost + wine_cost * wine_glasses + dessert_cost
  let discounted_meal_cost : ℚ := full_meal_cost - steak_cost * voucher_discount
  let tip : ℚ := full_meal_cost * tip_percentage

  discounted_meal_cost + tip = 38 := by sorry

end NUMINAMATH_CALUDE_arthurs_dinner_cost_l700_70095


namespace NUMINAMATH_CALUDE_cubic_increasing_and_odd_l700_70013

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem cubic_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry


end NUMINAMATH_CALUDE_cubic_increasing_and_odd_l700_70013


namespace NUMINAMATH_CALUDE_composition_equation_solution_l700_70053

theorem composition_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 4 * x + 9
  let φ : ℝ → ℝ := λ x ↦ 9 * x + 6
  ∃ x : ℝ, δ (φ x) = 10 ∧ x = -23/36 := by
  sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l700_70053


namespace NUMINAMATH_CALUDE_min_value_theorem_l700_70032

theorem min_value_theorem (x y : ℝ) (hx : x > 1) (hy : y > 1) (h_sum : x + 2*y = 5) :
  1/(x-1) + 1/(y-1) ≥ 3/2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l700_70032


namespace NUMINAMATH_CALUDE_michelle_gas_problem_l700_70090

/-- Michelle's gas problem -/
theorem michelle_gas_problem (gas_left gas_used : ℚ) 
  (h1 : gas_left = 0.17)
  (h2 : gas_used = 0.33) : 
  gas_left + gas_used = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_michelle_gas_problem_l700_70090


namespace NUMINAMATH_CALUDE_expression_value_l700_70023

theorem expression_value (p q : ℚ) (h : p / q = 4 / 5) :
  25 / 7 + (2 * q - p) / (2 * q + p) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l700_70023


namespace NUMINAMATH_CALUDE_moon_temperature_difference_l700_70047

theorem moon_temperature_difference : 
  let noon_temp : ℤ := 10
  let midnight_temp : ℤ := -150
  noon_temp - midnight_temp = 160 := by
sorry

end NUMINAMATH_CALUDE_moon_temperature_difference_l700_70047


namespace NUMINAMATH_CALUDE_square_root_equation_l700_70027

theorem square_root_equation (a : ℝ) : Real.sqrt (a^2) = 3 → a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l700_70027


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l700_70072

/-- Given a geometric sequence {a_n} with positive terms, if 3a_1, (1/2)a_3, and 2a_2 form an arithmetic sequence,
    then (a_2016 + a_2017) / (a_2015 + a_2016) = 3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : 3 * a 1 + 2 * a 2 = a 3) :
  (a 2016 + a 2017) / (a 2015 + a 2016) = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l700_70072


namespace NUMINAMATH_CALUDE_expression_not_simplifiable_to_AD_l700_70042

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (A B C D M : V)

theorem expression_not_simplifiable_to_AD :
  ∃ (BM DA MB : V), -BM - DA + MB ≠ (A - D) :=
by sorry

end NUMINAMATH_CALUDE_expression_not_simplifiable_to_AD_l700_70042


namespace NUMINAMATH_CALUDE_chandra_akiko_ratio_l700_70088

/-- Represents the points scored by each player in the basketball game -/
structure GameScores where
  chandra : ℕ
  akiko : ℕ
  michiko : ℕ
  bailey : ℕ

/-- The conditions of the basketball game -/
def gameConditions (s : GameScores) : Prop :=
  s.akiko = s.michiko + 4 ∧
  s.michiko * 2 = s.bailey ∧
  s.bailey = 14 ∧
  s.chandra + s.akiko + s.michiko + s.bailey = 54

/-- The theorem stating the ratio of Chandra's points to Akiko's points -/
theorem chandra_akiko_ratio (s : GameScores) : 
  gameConditions s → s.chandra * 1 = s.akiko * 2 := by
  sorry

#check chandra_akiko_ratio

end NUMINAMATH_CALUDE_chandra_akiko_ratio_l700_70088


namespace NUMINAMATH_CALUDE_no_prime_solutions_for_equation_l700_70036

theorem no_prime_solutions_for_equation : ¬∃ (x y z : ℕ), Prime x ∧ Prime y ∧ Prime z ∧ x^2 + y^3 = z^4 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solutions_for_equation_l700_70036


namespace NUMINAMATH_CALUDE_isosceles_triangle_larger_angle_l700_70000

/-- The measure of a right angle in degrees -/
def right_angle : ℝ := 90

/-- An isosceles triangle with one angle 20% smaller than a right angle -/
structure IsoscelesTriangle where
  /-- The measure of the smallest angle in degrees -/
  small_angle : ℝ
  /-- The measure of one of the two equal larger angles in degrees -/
  large_angle : ℝ
  /-- The triangle is isosceles with two equal larger angles -/
  isosceles : large_angle = large_angle
  /-- The small angle is 20% smaller than a right angle -/
  small_angle_def : small_angle = right_angle * (1 - 0.2)
  /-- The sum of all angles in the triangle is 180° -/
  angle_sum : small_angle + 2 * large_angle = 180

/-- Theorem: In an isosceles triangle where one angle is 20% smaller than a right angle,
    each of the two equal larger angles measures 54° -/
theorem isosceles_triangle_larger_angle (t : IsoscelesTriangle) : t.large_angle = 54 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_larger_angle_l700_70000


namespace NUMINAMATH_CALUDE_oranges_in_box_l700_70086

/-- The number of oranges left in a box after some are removed -/
def oranges_left (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem stating that 20 oranges are left in the box -/
theorem oranges_in_box : oranges_left 55 35 = 20 := by
  sorry

end NUMINAMATH_CALUDE_oranges_in_box_l700_70086


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l700_70080

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {-3} → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l700_70080


namespace NUMINAMATH_CALUDE_oliver_baseball_cards_l700_70008

theorem oliver_baseball_cards (cards_per_page new_cards old_cards : ℕ) 
  (h1 : cards_per_page = 3)
  (h2 : new_cards = 2)
  (h3 : old_cards = 10) :
  (new_cards + old_cards) / cards_per_page = 4 := by
  sorry

end NUMINAMATH_CALUDE_oliver_baseball_cards_l700_70008


namespace NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l700_70065

theorem sum_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l700_70065


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l700_70094

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 7 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l700_70094


namespace NUMINAMATH_CALUDE_at_op_four_nine_l700_70010

-- Define the operation @
def at_op (a b : ℝ) : ℝ := a * b ^ (1 / 2)

-- Theorem statement
theorem at_op_four_nine : at_op 4 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_at_op_four_nine_l700_70010


namespace NUMINAMATH_CALUDE_crayons_per_box_is_two_l700_70066

/-- Represents a crayon factory with given production parameters -/
structure CrayonFactory where
  colors : Nat
  boxes_per_hour : Nat
  total_crayons : Nat
  total_hours : Nat

/-- Calculates the number of crayons of each color in each box -/
def crayons_per_color_per_box (factory : CrayonFactory) : Nat :=
  factory.total_crayons / (factory.boxes_per_hour * factory.total_hours * factory.colors)

theorem crayons_per_box_is_two (factory : CrayonFactory) 
  (h1 : factory.colors = 4)
  (h2 : factory.boxes_per_hour = 5)
  (h3 : factory.total_crayons = 160)
  (h4 : factory.total_hours = 4) :
  crayons_per_color_per_box factory = 2 := by
  sorry

#eval crayons_per_color_per_box { colors := 4, boxes_per_hour := 5, total_crayons := 160, total_hours := 4 }

end NUMINAMATH_CALUDE_crayons_per_box_is_two_l700_70066


namespace NUMINAMATH_CALUDE_m_intersect_n_equals_open_interval_l700_70030

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 + 5*x - 14 < 0}

-- Define set N
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}

-- Theorem statement
theorem m_intersect_n_equals_open_interval :
  M ∩ N = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_m_intersect_n_equals_open_interval_l700_70030


namespace NUMINAMATH_CALUDE_equation_solution_l700_70061

theorem equation_solution :
  ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_equation_solution_l700_70061


namespace NUMINAMATH_CALUDE_problem_solution_l700_70025

theorem problem_solution (x y : ℝ) 
  (h : |9*y + 1 - x| = Real.sqrt (x - 4) * Real.sqrt (4 - x)) : 
  2*x*Real.sqrt (1/x) + Real.sqrt (9*y) - Real.sqrt x / 2 + y*Real.sqrt (1/y) = 3 + 4*Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l700_70025


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l700_70029

theorem fraction_ratio_equality : 
  ∃ (x y : ℚ), x / y = (240 : ℚ) / 1547 ∧ 
  x / y / ((2 : ℚ) / 13) = ((5 : ℚ) / 34) / ((7 : ℚ) / 48) := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l700_70029


namespace NUMINAMATH_CALUDE_fraction_relation_l700_70021

theorem fraction_relation (p r t u : ℚ) 
  (h1 : p / r = 8)
  (h2 : t / r = 5)
  (h3 : t / u = 2 / 3) :
  u / p = 15 / 16 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l700_70021


namespace NUMINAMATH_CALUDE_smallest_number_formed_by_2_and_4_l700_70002

def is_formed_by_2_and_4 (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + b ∧ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2)

theorem smallest_number_formed_by_2_and_4 :
  ∀ n : ℕ, is_formed_by_2_and_4 n → n ≥ 24 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_formed_by_2_and_4_l700_70002


namespace NUMINAMATH_CALUDE_fraction_equality_l700_70028

theorem fraction_equality (x : ℚ) : (1/5)^35 * x^18 = 1/(2*(10)^35) → x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l700_70028


namespace NUMINAMATH_CALUDE_xiao_hua_first_place_l700_70020

def fish_counts : List Nat := [23, 20, 15, 18, 13]

def xiao_hua_count : Nat := 20

def min_additional_fish (counts : List Nat) (xiao_hua : Nat) : Nat :=
  match counts.maximum? with
  | none => 0
  | some max_count => max_count - xiao_hua + 1

theorem xiao_hua_first_place (counts : List Nat) (xiao_hua : Nat) :
  counts = fish_counts ∧ xiao_hua = xiao_hua_count →
  min_additional_fish counts xiao_hua = 4 :=
by sorry

end NUMINAMATH_CALUDE_xiao_hua_first_place_l700_70020


namespace NUMINAMATH_CALUDE_fraction_value_l700_70009

theorem fraction_value (a b : ℝ) (h : 1/a - 1/b = 4) :
  (a - 2*a*b - b) / (2*a - 2*b + 7*a*b) = 6 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l700_70009


namespace NUMINAMATH_CALUDE_fraction_of_120_l700_70062

theorem fraction_of_120 : (1 / 3 : ℚ) * (1 / 4 : ℚ) * (1 / 6 : ℚ) * 120 = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_120_l700_70062


namespace NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l700_70055

def grid_size : Nat := 4

-- Define the number of ways to choose 2 items from n items
def choose_two (n : Nat) : Nat :=
  n * (n - 1) / 2

-- Define the number of rectangles in a square grid
def num_rectangles (n : Nat) : Nat :=
  (choose_two n) * (choose_two n)

-- Theorem statement
theorem rectangles_in_4x4_grid :
  num_rectangles grid_size = 36 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l700_70055


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l700_70001

theorem binomial_coefficient_divisibility 
  (p : Nat) (α : Nat) (m : Nat) 
  (hp : Nat.Prime p) 
  (hp_odd : Odd p) 
  (hα : α ≥ 2) 
  (hm : m ≥ 2) : 
  ∃ k : Nat, Nat.choose (p^(α-2)) m = k * p^(α-m) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l700_70001


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l700_70003

theorem imaginary_power_sum (i : ℂ) (hi : i^2 = -1) :
  i^13 + i^18 + i^23 + i^28 + i^33 + i^38 = 0 := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l700_70003


namespace NUMINAMATH_CALUDE_cos_alpha_value_l700_70026

-- Define the angle α
variable (α : Real)

-- Define the point P
def P : ℝ × ℝ := (4, 3)

-- Define the condition that the terminal side of α passes through P
def terminal_side_passes_through (α : Real) (p : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t > 0 ∧ p.1 = t * Real.cos α ∧ p.2 = t * Real.sin α

-- State the theorem
theorem cos_alpha_value (h : terminal_side_passes_through α P) : 
  Real.cos α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l700_70026


namespace NUMINAMATH_CALUDE_lawrence_walking_distance_l700_70073

/-- Calculates the total distance walked given the daily distance and number of days -/
def totalDistanceWalked (dailyDistance : ℝ) (days : ℝ) : ℝ :=
  dailyDistance * days

/-- Proves that walking 4.0 km a day for 3.0 days results in a total distance of 12.0 km -/
theorem lawrence_walking_distance :
  totalDistanceWalked 4.0 3.0 = 12.0 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_walking_distance_l700_70073


namespace NUMINAMATH_CALUDE_farmer_children_count_l700_70079

/-- Represents the number of apples in each child's bag -/
def apples_per_bag : ℕ := 15

/-- Represents the number of apples eaten by each of the two children -/
def apples_eaten_per_child : ℕ := 4

/-- Represents the number of apples sold by one child -/
def apples_sold : ℕ := 7

/-- Represents the number of apples left when they got home -/
def apples_left : ℕ := 60

/-- Proves that the farmer has 5 children given the conditions -/
theorem farmer_children_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_farmer_children_count_l700_70079


namespace NUMINAMATH_CALUDE_sequence_properties_l700_70024

/-- A sequence satisfying the given conditions -/
def Sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  a 1 = 2 ∧
  c ≠ 0 ∧
  (∀ n : ℕ, a (n + 1) = a n + c * n) ∧
  ∃ r : ℝ, r ≠ 0 ∧ a 2 = a 1 * r ∧ a 3 = a 2 * r

theorem sequence_properties {a : ℕ → ℝ} {c : ℝ} (h : Sequence a c) :
  c = 2 ∧
  (∀ n : ℕ, n > 0 → a n = n^2 - n + 2) ∧
  ∃ d : ℝ, ∀ n m : ℕ, n > 0 → m > 0 → (a n - c) / n - (a m - c) / m = d * (n - m) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l700_70024


namespace NUMINAMATH_CALUDE_cubic_root_sum_l700_70052

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 2*p - 2 = 0 → 
  q^3 - 2*q - 2 = 0 → 
  r^3 - 2*r - 2 = 0 → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -18 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l700_70052


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l700_70078

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

-- Define the line l
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Define the points E and F
def E : ℝ × ℝ := (1, -3)
def F : ℝ × ℝ := (0, 4)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- Theorem statement
theorem circle_and_line_intersection :
  ∃ (A B : ℝ × ℝ) (C2 : ℝ → ℝ → Prop),
    (∀ x y, C1 x y ∧ l x y ↔ (x, y) = A ∨ (x, y) = B) ∧
    (C2 E.1 E.2 ∧ C2 F.1 F.2) ∧
    (∃ D E F, ∀ x y, C2 x y ↔ x^2 + y^2 + D*x + E*y + F = 0) ∧
    (∃ k, ∀ x y, (C1 x y ∧ C2 x y) → (∃ c, x + k*y = c ∧ ∀ x' y', parallel_line x' y' → ∃ c', x' + k*y' = c')) →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 / 5 ∧
    (∀ x y, C2 x y ↔ x^2 + y^2 + 6*x - 16 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l700_70078


namespace NUMINAMATH_CALUDE_walking_distance_problem_l700_70085

theorem walking_distance_problem (D : ℝ) : 
  D / 10 = (D + 20) / 20 → D = 20 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_problem_l700_70085


namespace NUMINAMATH_CALUDE_watch_cost_price_l700_70098

theorem watch_cost_price (C : ℝ) : 
  (C * 0.9 = C * (1 - 0.1)) →  -- Selling at 90% of C is a 10% loss
  (C * 1.03 = C * (1 + 0.03)) →  -- Selling at 103% of C is a 3% gain
  (C * 1.03 - C * 0.9 = 140) →  -- Difference between selling prices is 140
  C = 1076.92 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l700_70098


namespace NUMINAMATH_CALUDE_largest_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l700_70019

theorem largest_integer_divisible_by_18_with_sqrt_between_24_and_24_5 :
  ∃ n : ℕ, n > 0 ∧ 18 ∣ n ∧ 24 < Real.sqrt n ∧ Real.sqrt n < 24.5 ∧
  ∀ m : ℕ, m > 0 → 18 ∣ m → 24 < Real.sqrt m → Real.sqrt m < 24.5 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l700_70019


namespace NUMINAMATH_CALUDE_triangle_inequality_l700_70056

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l700_70056


namespace NUMINAMATH_CALUDE_positive_solution_form_l700_70081

theorem positive_solution_form (x : ℝ) (a b : ℕ+) :
  x^2 + 14*x = 82 →
  x > 0 →
  x = Real.sqrt a - b →
  a + b = 138 :=
by
  sorry

end NUMINAMATH_CALUDE_positive_solution_form_l700_70081


namespace NUMINAMATH_CALUDE_estimate_cube_of_331_l700_70016

/-- Proves that (.331)^3 is approximately equal to 0.037, given that .331 is close to 1/3 -/
theorem estimate_cube_of_331 (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ |0.331 - (1/3)| < δ → |0.331^3 - 0.037| < ε :=
sorry

end NUMINAMATH_CALUDE_estimate_cube_of_331_l700_70016


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_l700_70031

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f(x)
def f_deriv (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem statement
theorem decreasing_interval_of_f :
  ∀ x : ℝ, (x ∈ Set.Ioo (-1) 1) ↔ (f_deriv x < 0) :=
sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_l700_70031


namespace NUMINAMATH_CALUDE_smallest_multiple_of_one_to_five_l700_70060

theorem smallest_multiple_of_one_to_five : ∃ (n : ℕ), n > 0 ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 → i ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 → i ∣ m) → n ≤ m) ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_one_to_five_l700_70060


namespace NUMINAMATH_CALUDE_power_three_times_three_l700_70054

theorem power_three_times_three (x : ℝ) : x^3 * x^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_three_times_three_l700_70054


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_inequality_l700_70043

theorem sqrt_equality_implies_inequality (x y α : ℝ) : 
  Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α) → x + y ≥ 2 * α := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_inequality_l700_70043


namespace NUMINAMATH_CALUDE_hadley_walk_distance_l700_70091

/-- The distance Hadley walked to the pet store -/
def distance_to_pet_store : ℝ := 1

/-- The distance Hadley walked to the grocery store -/
def distance_to_grocery : ℝ := 2

/-- The distance Hadley walked back home -/
def distance_back_home : ℝ := 4 - 1

/-- The total distance Hadley walked -/
def total_distance : ℝ := 6

theorem hadley_walk_distance :
  distance_to_grocery + distance_to_pet_store + distance_back_home = total_distance :=
by sorry

end NUMINAMATH_CALUDE_hadley_walk_distance_l700_70091


namespace NUMINAMATH_CALUDE_tribe_leadership_choices_l700_70045

/-- The number of ways to choose leadership in a tribe --/
def choose_leadership (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (Nat.choose (n - 3) 2) * (Nat.choose (n - 5) 2)

/-- Theorem: For a tribe of 10 members with the given leadership structure, 
    there are 151200 ways to choose the leadership --/
theorem tribe_leadership_choices :
  choose_leadership 10 = 151200 := by
  sorry

end NUMINAMATH_CALUDE_tribe_leadership_choices_l700_70045


namespace NUMINAMATH_CALUDE_proportional_distribution_theorem_l700_70076

def proportional_distribution_ratio (m : ℝ) (b_portion : ℝ) (ac_sum : ℝ) : ℝ → Prop :=
  fun r => m > 0 ∧ b_portion = 80 ∧ ac_sum = 164 ∧ 
           (80 / (1 - r) + 80 * (1 - r) = 164)

theorem proportional_distribution_theorem (m : ℝ) (b_portion : ℝ) (ac_sum : ℝ) :
  ∃ r : ℝ, proportional_distribution_ratio m b_portion ac_sum r ∧ r = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_proportional_distribution_theorem_l700_70076


namespace NUMINAMATH_CALUDE_integral_2sqrt_minus_sin_l700_70077

open MeasureTheory Interval Real

theorem integral_2sqrt_minus_sin : ∫ x in (-1)..1, (2 * Real.sqrt (1 - x^2) - Real.sin x) = π := by
  sorry

end NUMINAMATH_CALUDE_integral_2sqrt_minus_sin_l700_70077


namespace NUMINAMATH_CALUDE_somu_age_problem_l700_70033

/-- Represents the problem of finding when Somu was one-fifth of his father's age -/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_ago : ℕ) : 
  somu_age = 14 →
  somu_age = father_age / 3 →
  somu_age - years_ago = (father_age - years_ago) / 5 →
  years_ago = 7 := by
  sorry

end NUMINAMATH_CALUDE_somu_age_problem_l700_70033


namespace NUMINAMATH_CALUDE_min_cuboid_height_l700_70012

/-- Represents a cuboid with a square base -/
structure Cuboid where
  base_side : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- The minimum height of a cuboid that can contain given spheres -/
def min_height (base_side : ℝ) (small_spheres : List Sphere) (large_sphere : Sphere) : ℝ :=
  sorry

theorem min_cuboid_height :
  let cuboid : Cuboid := { base_side := 4, height := min_height 4 (List.replicate 8 { radius := 1 }) { radius := 2 } }
  let small_spheres : List Sphere := List.replicate 8 { radius := 1 }
  let large_sphere : Sphere := { radius := 2 }
  cuboid.height = 2 + 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_min_cuboid_height_l700_70012


namespace NUMINAMATH_CALUDE_jelly_beans_in_jar_X_l700_70097

/-- The number of jelly beans in jar X -/
def jarX (total : ℕ) (y : ℕ) : ℕ := 3 * y - 400

/-- The total number of jelly beans in both jars -/
def totalBeans (x y : ℕ) : ℕ := x + y

theorem jelly_beans_in_jar_X :
  ∃ (y : ℕ), totalBeans (jarX 1200 y) y = 1200 ∧ jarX 1200 y = 800 := by
  sorry

end NUMINAMATH_CALUDE_jelly_beans_in_jar_X_l700_70097


namespace NUMINAMATH_CALUDE_vector_relations_l700_70018

/-- Given plane vectors a, b, and c, prove parallel and perpendicular conditions. -/
theorem vector_relations (a b c : ℝ × ℝ) (t : ℝ) 
  (ha : a = (-2, 1)) 
  (hb : b = (4, 2)) 
  (hc : c = (2, t)) : 
  (∃ (k : ℝ), a = k • c → t = -1) ∧ 
  (b.1 * c.1 + b.2 * c.2 = 0 → t = -4) := by
  sorry


end NUMINAMATH_CALUDE_vector_relations_l700_70018


namespace NUMINAMATH_CALUDE_expression_value_l700_70063

theorem expression_value (x y : ℝ) (h : x - 3*y = 4) : 15*y - 5*x + 6 = -14 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l700_70063


namespace NUMINAMATH_CALUDE_pencil_cost_l700_70064

/-- If 120 pencils cost $40, then 3600 pencils will cost $1200. -/
theorem pencil_cost (cost_120 : ℕ) (pencils : ℕ) :
  cost_120 = 40 ∧ pencils = 3600 → pencils * cost_120 / 120 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l700_70064


namespace NUMINAMATH_CALUDE_smallest_number_minus_one_in_list_minus_one_is_smallest_l700_70017

def numbers : List ℚ := [3, 0, -1, -1/2]

theorem smallest_number (n : ℚ) (hn : n ∈ numbers) :
  -1 ≤ n := by sorry

theorem minus_one_in_list : -1 ∈ numbers := by sorry

theorem minus_one_is_smallest : ∀ n ∈ numbers, -1 ≤ n ∧ ∃ m ∈ numbers, -1 = m := by sorry

end NUMINAMATH_CALUDE_smallest_number_minus_one_in_list_minus_one_is_smallest_l700_70017


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l700_70038

theorem sum_of_x_solutions_is_zero (x y : ℝ) :
  y = 6 →
  x^2 + y^2 = 169 →
  ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0 ∧ 
    ((x = x₁ ∨ x = x₂) ↔ (y = 6 ∧ x^2 + y^2 = 169)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l700_70038


namespace NUMINAMATH_CALUDE_train_speed_l700_70040

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 300 →
  crossing_time = 36 →
  ∃ (speed : ℝ), abs (speed - (train_length + bridge_length) / crossing_time) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l700_70040


namespace NUMINAMATH_CALUDE_orange_sales_l700_70037

theorem orange_sales (alice_oranges emily_oranges total_oranges : ℕ) : 
  alice_oranges = 120 →
  alice_oranges = 2 * emily_oranges →
  total_oranges = alice_oranges + emily_oranges →
  total_oranges = 180 := by
  sorry

end NUMINAMATH_CALUDE_orange_sales_l700_70037


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l700_70004

/-- The line equation passing through a fixed point -/
def line_equation (k x y : ℝ) : Prop :=
  k * x + (1 - k) * y - 3 = 0

/-- Theorem stating that the line passes through (3, 3) for all k -/
theorem fixed_point_on_line :
  ∀ (k : ℝ), line_equation k 3 3 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l700_70004


namespace NUMINAMATH_CALUDE_sum_of_a_and_a1_is_nine_l700_70058

theorem sum_of_a_and_a1_is_nine (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x + 1)^2 + (x + 1)^11 = a + a₁*(x + 2) + a₂*(x + 2)^2 + a₃*(x + 2)^3 + 
   a₄*(x + 2)^4 + a₅*(x + 2)^5 + a₆*(x + 2)^6 + a₇*(x + 2)^7 + a₈*(x + 2)^8 + 
   a₉*(x + 2)^9 + a₁₀*(x + 2)^10 + a₁₁*(x + 2)^11) →
  a + a₁ = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_a1_is_nine_l700_70058


namespace NUMINAMATH_CALUDE_wendy_packaging_capacity_l700_70067

/-- Wendy's chocolate packaging rate -/
def wendy_rate : ℕ := 1152

/-- Time period for given rate -/
def rate_period : ℕ := 4

/-- Conversion factor from dozens to individual chocolates -/
def dozen_to_individual : ℕ := 12

/-- Calculates the number of chocolates Wendy can package in a given number of hours -/
def chocolates_packaged (hours : ℕ) : ℕ :=
  (wendy_rate * hours) / rate_period

/-- Theorem stating Wendy's packaging capacity for h hours -/
theorem wendy_packaging_capacity (h : ℕ) :
  chocolates_packaged h = 288 * h :=
sorry

end NUMINAMATH_CALUDE_wendy_packaging_capacity_l700_70067


namespace NUMINAMATH_CALUDE_only_solutions_l700_70083

/-- A function from nonnegative integers to nonnegative integers -/
def NonNegIntFunction := ℕ → ℕ

/-- The property that f(f(f(n))) = f(n+1) + 1 for all n -/
def SatisfiesEquation (f : NonNegIntFunction) : Prop :=
  ∀ n, f (f (f n)) = f (n + 1) + 1

/-- The first solution function: f(n) = n + 1 -/
def Solution1 : NonNegIntFunction :=
  λ n => n + 1

/-- The second solution function: 
    f(n) = n + 1 if n ≡ 0 (mod 4) or n ≡ 2 (mod 4),
    f(n) = n + 5 if n ≡ 1 (mod 4),
    f(n) = n - 3 if n ≡ 3 (mod 4) -/
def Solution2 : NonNegIntFunction :=
  λ n => match n % 4 with
    | 0 | 2 => n + 1
    | 1 => n + 5
    | 3 => n - 3
    | _ => n  -- This case is unreachable, but needed for exhaustiveness

/-- The main theorem: Solution1 and Solution2 are the only functions satisfying the equation -/
theorem only_solutions (f : NonNegIntFunction) :
  SatisfiesEquation f ↔ (f = Solution1 ∨ f = Solution2) := by
  sorry

end NUMINAMATH_CALUDE_only_solutions_l700_70083


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_54_l700_70068

theorem gcd_lcm_product_24_54 : Nat.gcd 24 54 * Nat.lcm 24 54 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_54_l700_70068


namespace NUMINAMATH_CALUDE_trigonometric_problem_l700_70048

theorem trigonometric_problem (α : ℝ) 
  (h1 : Real.sin (α + π/3) + Real.sin α = 9 * Real.sqrt 7 / 14)
  (h2 : 0 < α)
  (h3 : α < π/3) :
  (Real.sin α = 2 * Real.sqrt 7 / 7) ∧ 
  (Real.cos (2*α - π/4) = (4 * Real.sqrt 6 - Real.sqrt 2) / 14) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l700_70048


namespace NUMINAMATH_CALUDE_earning_amount_l700_70005

/-- Represents the earning and spending pattern over 60 days -/
def pattern_result (E : ℚ) : ℚ :=
  30 * (E - 15)

/-- Proves that the earning amount must be 17 given the conditions -/
theorem earning_amount : ∃ E : ℚ, pattern_result E = 60 ∧ E = 17 := by
  sorry

end NUMINAMATH_CALUDE_earning_amount_l700_70005


namespace NUMINAMATH_CALUDE_little_red_final_score_l700_70093

/-- Calculates the final score for the "Sunshine Sports" competition --/
def final_score (running_score fancy_jump_rope_score jump_rope_score : ℝ)
  (running_weight fancy_jump_rope_weight jump_rope_weight : ℝ) : ℝ :=
  running_score * running_weight +
  fancy_jump_rope_score * fancy_jump_rope_weight +
  jump_rope_score * jump_rope_weight

/-- Theorem stating that Little Red's final score is 83 --/
theorem little_red_final_score :
  final_score 90 80 70 0.5 0.3 0.2 = 83 := by
  sorry

#eval final_score 90 80 70 0.5 0.3 0.2

end NUMINAMATH_CALUDE_little_red_final_score_l700_70093


namespace NUMINAMATH_CALUDE_k_value_for_given_factors_l700_70039

/-- The length of an integer is the number of positive prime factors, not necessarily distinct, whose product is equal to the integer. -/
def length (n : ℕ) : ℕ := sorry

/-- The prime factors of an integer as a multiset. -/
def primeFactors (n : ℕ) : Multiset ℕ := sorry

theorem k_value_for_given_factors :
  ∀ k : ℕ,
    k > 1 →
    length k = 4 →
    primeFactors k = {2, 2, 2, 3} →
    k = 24 := by
  sorry

end NUMINAMATH_CALUDE_k_value_for_given_factors_l700_70039


namespace NUMINAMATH_CALUDE_measure_water_l700_70071

theorem measure_water (a : ℤ) (h : -1562 ≤ a ∧ a ≤ 1562) :
  ∃ (b c d e f : ℤ), 
    (b ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (c ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (d ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (e ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (f ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (a = 625*b + 125*c + 25*d + 5*e + f) :=
by sorry

end NUMINAMATH_CALUDE_measure_water_l700_70071


namespace NUMINAMATH_CALUDE_sphere_tangency_loci_l700_70059

/-- Given a sphere of radius R touching a plane, and spheres of radius r
    touching both the given sphere and the plane, this theorem proves the radii
    of the circles formed by the centers and points of tangency of the r-radius spheres. -/
theorem sphere_tangency_loci (R r : ℝ) (h : R > 0) (h' : r > 0) :
  ∃ (center_locus tangent_plane_locus tangent_sphere_locus : ℝ),
    center_locus = 2 * Real.sqrt (R * r) ∧
    tangent_plane_locus = 2 * Real.sqrt (R * r) ∧
    tangent_sphere_locus = (2 * R * Real.sqrt (R * r)) / (R + r) :=
sorry

end NUMINAMATH_CALUDE_sphere_tangency_loci_l700_70059


namespace NUMINAMATH_CALUDE_fraction_equality_implies_c_geq_one_l700_70050

theorem fraction_equality_implies_c_geq_one
  (a b : ℕ+) (c : ℝ)
  (h_c_pos : c > 0)
  (h_eq : (a + 1) / (b + c) = b / a) :
  c ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_c_geq_one_l700_70050


namespace NUMINAMATH_CALUDE_age_difference_l700_70011

/-- Given that the total age of a and b is 13 years more than the total age of b and c,
    prove that c is 13 years younger than a. -/
theorem age_difference (a b c : ℕ) (h : a + b = b + c + 13) : a = c + 13 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l700_70011


namespace NUMINAMATH_CALUDE_logarithmic_best_fit_l700_70035

-- Define the types of functions we're considering
inductive FunctionType
  | Linear
  | Quadratic
  | Exponential
  | Logarithmic

-- Define the characteristics we're looking for
structure GrowthCharacteristics where
  rapid_initial_growth : Bool
  slowing_growth_rate : Bool

-- Define a function that checks if a function type matches the desired characteristics
def matches_characteristics (f : FunctionType) (c : GrowthCharacteristics) : Prop :=
  match f with
  | FunctionType.Linear => false
  | FunctionType.Quadratic => false
  | FunctionType.Exponential => false
  | FunctionType.Logarithmic => c.rapid_initial_growth ∧ c.slowing_growth_rate

-- Theorem statement
theorem logarithmic_best_fit (c : GrowthCharacteristics) 
  (h1 : c.rapid_initial_growth = true) 
  (h2 : c.slowing_growth_rate = true) : 
  ∀ f : FunctionType, matches_characteristics f c ↔ f = FunctionType.Logarithmic :=
sorry

end NUMINAMATH_CALUDE_logarithmic_best_fit_l700_70035


namespace NUMINAMATH_CALUDE_replaced_student_weight_l700_70007

theorem replaced_student_weight
  (n : ℕ)
  (new_weight : ℝ)
  (avg_decrease : ℝ)
  (h1 : n = 6)
  (h2 : new_weight = 62)
  (h3 : avg_decrease = 3)
  : ∃ (old_weight : ℝ), old_weight = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_replaced_student_weight_l700_70007


namespace NUMINAMATH_CALUDE_multiset_permutations_eq_1680_l700_70070

/-- The number of permutations of a multiset with 9 elements, where there are 3 elements of each of 3 types -/
def multiset_permutations : ℕ :=
  Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3)

/-- Theorem stating that the number of permutations of the described multiset is 1680 -/
theorem multiset_permutations_eq_1680 : multiset_permutations = 1680 := by
  sorry

end NUMINAMATH_CALUDE_multiset_permutations_eq_1680_l700_70070


namespace NUMINAMATH_CALUDE_smallest_winning_k_l700_70074

/-- Represents the game board -/
def Board := Fin 8 → Fin 8 → Option Char

/-- Checks if a sequence "HMM" or "MMH" exists horizontally or vertically -/
def winning_sequence (board : Board) : Prop :=
  ∃ (i j : Fin 8), 
    (board i j = some 'H' ∧ board i (j+1) = some 'M' ∧ board i (j+2) = some 'M') ∨
    (board i j = some 'M' ∧ board i (j+1) = some 'M' ∧ board i (j+2) = some 'H') ∨
    (board i j = some 'H' ∧ board (i+1) j = some 'M' ∧ board (i+2) j = some 'M') ∨
    (board i j = some 'M' ∧ board (i+1) j = some 'M' ∧ board (i+2) j = some 'H')

/-- Mike's strategy for placing 'M's -/
def mike_strategy (k : ℕ) : Board := sorry

/-- Harry's strategy for placing 'H's -/
def harry_strategy (k : ℕ) (mike_board : Board) : Board := sorry

/-- The main theorem stating that 16 is the smallest k for which Mike has a winning strategy -/
theorem smallest_winning_k : 
  (∀ (k : ℕ), k < 16 → ∃ (harry_board : Board), 
    harry_board = harry_strategy k (mike_strategy k) ∧ ¬winning_sequence harry_board) ∧ 
  (∀ (harry_board : Board), 
    harry_board = harry_strategy 16 (mike_strategy 16) → winning_sequence harry_board) :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_k_l700_70074


namespace NUMINAMATH_CALUDE_sally_rum_amount_l700_70089

theorem sally_rum_amount (x : ℝ) : 
  (∀ (max_rum : ℝ), max_rum = 3 * x) →   -- Maximum amount is 3 times what Sally gave
  (∀ (earlier_rum : ℝ), earlier_rum = 12) →  -- Don already had 12 oz
  (∀ (remaining_rum : ℝ), remaining_rum = 8) →  -- Don can still have 8 oz
  (x + 12 + 8 = 3 * x) →  -- Total amount equals maximum healthy amount
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_sally_rum_amount_l700_70089


namespace NUMINAMATH_CALUDE_certain_number_proof_l700_70015

theorem certain_number_proof (x : ℝ) : 
  (0.02: ℝ)^2 + x^2 + (0.035 : ℝ)^2 = 100 * ((0.002 : ℝ)^2 + (0.052 : ℝ)^2 + (0.0035 : ℝ)^2) → 
  x = 0.52 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l700_70015


namespace NUMINAMATH_CALUDE_cube_volume_from_lateral_surface_area_l700_70099

theorem cube_volume_from_lateral_surface_area :
  ∀ (lateral_surface_area : ℝ) (volume : ℝ),
  lateral_surface_area = 100 →
  volume = (lateral_surface_area / 4) ^ (3/2) →
  volume = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_lateral_surface_area_l700_70099


namespace NUMINAMATH_CALUDE_mod_equivalence_proof_l700_70051

theorem mod_equivalence_proof : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4573 [ZMOD 8] → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_proof_l700_70051


namespace NUMINAMATH_CALUDE_solve_y_l700_70034

theorem solve_y (x y : ℝ) (h1 : x^2 = y - 3) (h2 : x = 7) : 
  y = 52 ∧ y ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_solve_y_l700_70034


namespace NUMINAMATH_CALUDE_percentage_increase_l700_70057

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 100 → final = 110 → (final - initial) / initial * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l700_70057


namespace NUMINAMATH_CALUDE_divisibility_proof_l700_70006

theorem divisibility_proof : (2 ∣ 32) ∧ (20 ∣ 320) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_proof_l700_70006


namespace NUMINAMATH_CALUDE_adult_ticket_price_l700_70084

def student_price : ℚ := 5/2

theorem adult_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (student_tickets : ℕ) 
  (h1 : total_tickets = 59) 
  (h2 : total_revenue = 445/2) 
  (h3 : student_tickets = 9) : 
  (total_revenue - student_price * student_tickets) / (total_tickets - student_tickets) = 4 := by
sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l700_70084


namespace NUMINAMATH_CALUDE_regular_hexagon_area_l700_70082

/-- The area of a regular hexagon with vertices A(0,0) and C(4,6) is 78√3 -/
theorem regular_hexagon_area : 
  let A : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (4, 6)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let hexagon_area : ℝ := 6 * (Real.sqrt 3 / 4 * AC^2)
  hexagon_area = 78 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_regular_hexagon_area_l700_70082
