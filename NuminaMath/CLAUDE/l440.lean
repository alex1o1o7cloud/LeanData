import Mathlib

namespace NUMINAMATH_CALUDE_min_queries_for_100_sets_l440_44065

/-- Represents a query operation on two sets -/
inductive Query
  | intersect : ℕ → ℕ → Query
  | union : ℕ → ℕ → Query

/-- The result of a query operation -/
def QueryResult := Set ℕ

/-- A function that performs a query on two sets -/
def performQuery : Query → (ℕ → Set ℕ) → QueryResult := sorry

/-- A strategy is a sequence of queries -/
def Strategy := List Query

/-- Checks if a strategy determines all sets -/
def determinesAllSets (s : Strategy) (n : ℕ) : Prop := sorry

/-- The main theorem: 100 queries are necessary and sufficient -/
theorem min_queries_for_100_sets :
  (∃ (s : Strategy), s.length = 100 ∧ determinesAllSets s 100) ∧
  (∀ (s : Strategy), s.length < 100 → ¬determinesAllSets s 100) := by sorry

end NUMINAMATH_CALUDE_min_queries_for_100_sets_l440_44065


namespace NUMINAMATH_CALUDE_Q_proper_subset_P_l440_44055

def P : Set ℝ := {x : ℝ | x ≥ 1}
def Q : Set ℝ := {2, 3}

theorem Q_proper_subset_P : Q ⊂ P := by sorry

end NUMINAMATH_CALUDE_Q_proper_subset_P_l440_44055


namespace NUMINAMATH_CALUDE_ten_streets_intersections_l440_44041

/-- The number of intersections created by n non-parallel straight streets -/
def intersections (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- Theorem: 10 non-parallel straight streets create 45 intersections -/
theorem ten_streets_intersections :
  intersections 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_streets_intersections_l440_44041


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l440_44040

/-- A hyperbola with given asymptotes -/
structure Hyperbola where
  /-- The asymptotes of the hyperbola are x ± 2y = 0 -/
  asymptotes : ∀ (x y : ℝ), x = 2*y ∨ x = -2*y

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := 
  sorry

/-- Theorem: The eccentricity of a hyperbola with asymptotes x ± 2y = 0 is either √5 or √5/2 -/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  eccentricity h = Real.sqrt 5 ∨ eccentricity h = (Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l440_44040


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l440_44082

def dividend (x : ℝ) : ℝ := 3 * x^5 - 2 * x^3 + 5 * x^2 - 9
def divisor (x : ℝ) : ℝ := x^2 - 2 * x + 1
def remainder (x : ℝ) : ℝ := 19 * x - 22

theorem polynomial_division_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, dividend x = divisor x * q x + remainder x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l440_44082


namespace NUMINAMATH_CALUDE_viewer_ratio_l440_44088

def voltaire_daily_viewers : ℕ := 50
def earnings_per_view : ℚ := 1/2
def leila_weekly_earnings : ℕ := 350
def days_per_week : ℕ := 7

theorem viewer_ratio : 
  let voltaire_weekly_viewers := voltaire_daily_viewers * days_per_week
  let leila_weekly_viewers := (leila_weekly_earnings : ℚ) / earnings_per_view
  (leila_weekly_viewers : ℚ) / (voltaire_weekly_viewers : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_viewer_ratio_l440_44088


namespace NUMINAMATH_CALUDE_seventh_fibonacci_is_eight_l440_44061

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem seventh_fibonacci_is_eight :
  fibonacci 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_fibonacci_is_eight_l440_44061


namespace NUMINAMATH_CALUDE_xyz_value_l440_44008

theorem xyz_value (a b c x y z : ℂ) 
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (eq1 : a = (b + c) / (x + 1))
  (eq2 : b = (a + c) / (y + 1))
  (eq3 : c = (a + b) / (z + 1))
  (sum_prod : x * y + x * z + y * z = 9)
  (sum : x + y + z = 5) :
  x * y * z = 13 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l440_44008


namespace NUMINAMATH_CALUDE_cubic_function_property_l440_44053

/-- Given a function f(x) = ax³ + bx + 1, prove that if f(a) = 8, then f(-a) = -6 -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + b * x + 1
  f a = 8 → f (-a) = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l440_44053


namespace NUMINAMATH_CALUDE_car_speed_problem_l440_44019

/-- Theorem: Given two cars starting from opposite ends of a 60-mile highway at the same time,
    with one car traveling at speed x mph and the other at 17 mph, if they meet after 2 hours,
    then the speed x of the first car is 13 mph. -/
theorem car_speed_problem (x : ℝ) :
  (x > 0) →  -- Assuming positive speed for the first car
  (2 * x + 2 * 17 = 60) →  -- Distance traveled by both cars equals highway length
  x = 13 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l440_44019


namespace NUMINAMATH_CALUDE_expression_is_square_difference_l440_44058

/-- The square difference formula -/
def square_difference (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- The expression to be checked -/
def expression (x y : ℝ) : ℝ := (-x + y) * (x + y)

/-- Theorem stating that the expression can be calculated using the square difference formula -/
theorem expression_is_square_difference (x y : ℝ) :
  ∃ a b : ℝ, expression x y = -square_difference a b :=
sorry

end NUMINAMATH_CALUDE_expression_is_square_difference_l440_44058


namespace NUMINAMATH_CALUDE_total_rabbits_l440_44034

theorem total_rabbits (initial additional : ℕ) : 
  initial + additional = (initial + additional) :=
by sorry

end NUMINAMATH_CALUDE_total_rabbits_l440_44034


namespace NUMINAMATH_CALUDE_correct_answer_l440_44092

theorem correct_answer (x : ℝ) (h : 2 * x = 60) : x / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l440_44092


namespace NUMINAMATH_CALUDE_cube_remainder_sum_quotient_l440_44020

def cube_rem_16 (n : ℕ) : ℕ := (n^3) % 16

def distinct_remainders : Finset ℕ :=
  (Finset.range 15).image cube_rem_16

theorem cube_remainder_sum_quotient :
  (Finset.sum distinct_remainders id) / 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_remainder_sum_quotient_l440_44020


namespace NUMINAMATH_CALUDE_quadratic_properties_l440_44010

def f (x : ℝ) := 2 * x^2 - 4 * x + 3

theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (f 0 > 0) ∧ 
  (∀ x : ℝ, f x ≠ 0) ∧ 
  (∀ x y : ℝ, x < y → x < 1 → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l440_44010


namespace NUMINAMATH_CALUDE_min_nSn_arithmetic_sequence_l440_44054

/-- Arithmetic sequence sum function -/
def S (a₁ d : ℚ) (n : ℕ) : ℚ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Product of n and S_n -/
def nSn (a₁ d : ℚ) (n : ℕ) : ℚ := n * S a₁ d n

theorem min_nSn_arithmetic_sequence :
  ∃ (a₁ d : ℚ),
    S a₁ d 10 = 0 ∧
    S a₁ d 15 = 25 ∧
    (∀ (n : ℕ), n > 0 → nSn a₁ d n ≥ -48) ∧
    (∃ (n : ℕ), n > 0 ∧ nSn a₁ d n = -48) := by
  sorry

end NUMINAMATH_CALUDE_min_nSn_arithmetic_sequence_l440_44054


namespace NUMINAMATH_CALUDE_expression_value_theorem_l440_44028

theorem expression_value_theorem (a b c d m : ℝ) :
  (a = -b) →
  (c * d = 1) →
  (|m| = 5) →
  (2 * (a + b) - 3 * c * d + m = 2 ∨ 2 * (a + b) - 3 * c * d + m = -8) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_theorem_l440_44028


namespace NUMINAMATH_CALUDE_simplest_fraction_sum_l440_44047

theorem simplest_fraction_sum (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  (a : ℚ) / b = 0.84375 ∧
  ∀ (c d : ℕ), c > 0 → d > 0 → (c : ℚ) / d = 0.84375 → a ≤ c ∧ b ≤ d →
  a + b = 59 := by
sorry

end NUMINAMATH_CALUDE_simplest_fraction_sum_l440_44047


namespace NUMINAMATH_CALUDE_field_area_is_fifty_l440_44012

/-- Represents a rectangular field with specific fencing conditions -/
structure FencedField where
  length : ℝ
  width : ℝ
  uncovered_side : ℝ
  fencing_length : ℝ

/-- The area of the field is 50 square feet given the specified conditions -/
theorem field_area_is_fifty (field : FencedField)
  (h1 : field.uncovered_side = 20)
  (h2 : field.fencing_length = 25)
  (h3 : field.length = field.uncovered_side)
  (h4 : field.fencing_length = field.length + 2 * field.width) :
  field.length * field.width = 50 := by
  sorry

#check field_area_is_fifty

end NUMINAMATH_CALUDE_field_area_is_fifty_l440_44012


namespace NUMINAMATH_CALUDE_max_boat_distance_xiaohu_max_distance_l440_44018

/-- Calculates the maximum distance a boat can travel in a river with given conditions --/
theorem max_boat_distance (total_time : ℝ) (boat_speed : ℝ) (current_speed : ℝ) 
  (paddle_time : ℝ) (break_time : ℝ) : ℝ :=
  let total_minutes : ℝ := total_time * 60
  let cycle_time : ℝ := paddle_time + break_time
  let num_cycles : ℝ := total_minutes / cycle_time
  let total_break_time : ℝ := num_cycles * break_time
  let effective_paddle_time : ℝ := total_minutes - total_break_time - total_break_time
  let upstream_speed : ℝ := boat_speed - current_speed
  let downstream_speed : ℝ := boat_speed + current_speed
  let downstream_ratio : ℝ := downstream_speed / (upstream_speed + downstream_speed)
  let downstream_paddle_time : ℝ := downstream_ratio * effective_paddle_time
  let downstream_distance : ℝ := downstream_speed * (downstream_paddle_time / 60)
  let drift_distance : ℝ := current_speed * (break_time / 60)
  downstream_distance + drift_distance

/-- Proves that the maximum distance Xiaohu can be from the rental place is 1.375 km --/
theorem xiaohu_max_distance : 
  max_boat_distance 2 3 1.5 30 10 = 1.375 := by sorry

end NUMINAMATH_CALUDE_max_boat_distance_xiaohu_max_distance_l440_44018


namespace NUMINAMATH_CALUDE_min_value_quadratic_l440_44090

/-- The function f(x) = (3/2)x^2 - 9x + 7 attains its minimum value when x = 3 -/
theorem min_value_quadratic (x : ℝ) : 
  (∀ y : ℝ, (3/2 : ℝ) * x^2 - 9*x + 7 ≤ (3/2 : ℝ) * y^2 - 9*y + 7) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l440_44090


namespace NUMINAMATH_CALUDE_number_solution_l440_44086

theorem number_solution : ∃ x : ℝ, (5020 - (502 / x) = 5015) ∧ x = 100.4 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l440_44086


namespace NUMINAMATH_CALUDE_car_speed_problem_l440_44050

/-- Two cars start from the same point and travel in opposite directions. -/
structure TwoCars where
  car1_speed : ℝ
  car2_speed : ℝ
  travel_time : ℝ
  total_distance : ℝ

/-- The theorem states that given the conditions of the problem, 
    the speed of the second car is 50 mph. -/
theorem car_speed_problem (cars : TwoCars) 
  (h1 : cars.car1_speed = 40)
  (h2 : cars.travel_time = 5)
  (h3 : cars.total_distance = 450) :
  cars.car2_speed = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l440_44050


namespace NUMINAMATH_CALUDE_M_inter_N_eq_N_l440_44080

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {x : ℝ | x^2 ≤ x}

-- Theorem statement
theorem M_inter_N_eq_N : M ∩ N = N := by
  sorry

end NUMINAMATH_CALUDE_M_inter_N_eq_N_l440_44080


namespace NUMINAMATH_CALUDE_two_person_subcommittees_l440_44048

/-- The number of combinations of n items taken k at a time -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The size of the original committee -/
def committeeSize : ℕ := 8

/-- The size of the sub-committee -/
def subCommitteeSize : ℕ := 2

/-- The number of different two-person sub-committees that can be selected from a committee of eight people -/
theorem two_person_subcommittees : choose committeeSize subCommitteeSize = 28 := by
  sorry

end NUMINAMATH_CALUDE_two_person_subcommittees_l440_44048


namespace NUMINAMATH_CALUDE_max_sum_of_square_roots_l440_44000

theorem max_sum_of_square_roots (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_square_roots_l440_44000


namespace NUMINAMATH_CALUDE_remainder_theorem_f_of_one_eq_four_remainder_is_four_l440_44039

-- Define the polynomial f(x) = x^15 + 3
def f (x : ℝ) : ℝ := x^15 + 3

-- State the theorem
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = (x - 1) * q x + f 1 := by
  sorry

-- Prove that f(1) = 4
theorem f_of_one_eq_four : f 1 = 4 := by
  sorry

-- Main theorem: The remainder when x^15 + 3 is divided by x-1 is 4
theorem remainder_is_four :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = (x - 1) * q x + 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_f_of_one_eq_four_remainder_is_four_l440_44039


namespace NUMINAMATH_CALUDE_sum_of_roots_of_sum_l440_44006

/-- Given two quadratic polynomials with the same leading coefficient, 
    if the sum of their four roots is p and their sum has two roots, 
    then the sum of the roots of their sum is p/2 -/
theorem sum_of_roots_of_sum (f g : ℝ → ℝ) (a b₁ b₂ c₁ c₂ p : ℝ) :
  (∀ x, f x = a * x^2 + b₁ * x + c₁) →
  (∀ x, g x = a * x^2 + b₂ * x + c₂) →
  (-b₁ / a - b₂ / a = p) →
  (∃ x y, ∀ z, f z + g z = 2 * a * (z - x) * (z - y)) →
  -(b₁ + b₂) / (2 * a) = p / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_sum_l440_44006


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l440_44046

theorem student_multiplication_problem (x : ℝ) : 30 * x - 138 = 102 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l440_44046


namespace NUMINAMATH_CALUDE_twentyFiveCentCoins_l440_44042

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  five : ℕ
  ten : ℕ
  twentyFive : ℕ

/-- Calculates the total number of coins -/
def totalCoins (c : CoinCounts) : ℕ := c.five + c.ten + c.twentyFive

/-- Calculates the number of different values that can be obtained -/
def differentValues (c : CoinCounts) : ℕ :=
  74 - 4 * c.five - 3 * c.ten

/-- Main theorem -/
theorem twentyFiveCentCoins (c : CoinCounts) :
  totalCoins c = 15 ∧ differentValues c = 30 → c.twentyFive = 2 := by
  sorry

end NUMINAMATH_CALUDE_twentyFiveCentCoins_l440_44042


namespace NUMINAMATH_CALUDE_football_exercise_calories_l440_44064

/-- Calculates the total calories burned during a stair-climbing exercise. -/
def total_calories_burned (round_trips : ℕ) (stairs_one_way : ℕ) (calories_per_stair : ℕ) : ℕ :=
  round_trips * (2 * stairs_one_way) * calories_per_stair

/-- Proves that given the specific conditions, the total calories burned is 16200. -/
theorem football_exercise_calories : 
  total_calories_burned 60 45 3 = 16200 := by
  sorry

end NUMINAMATH_CALUDE_football_exercise_calories_l440_44064


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l440_44038

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_35 :
  ∃ (n : ℕ), is_four_digit n ∧ n % 35 = 0 ∧ ∀ (m : ℕ), is_four_digit m ∧ m % 35 = 0 → n ≤ m :=
by
  use 1050
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l440_44038


namespace NUMINAMATH_CALUDE_smallest_non_representable_l440_44025

def representable (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = (2^a - 2^b) / (2^c - 2^d)

theorem smallest_non_representable : ∀ k < 11, representable k ∧ ¬ representable 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_representable_l440_44025


namespace NUMINAMATH_CALUDE_books_after_donation_l440_44073

theorem books_after_donation (boris_initial : Nat) (cameron_initial : Nat)
  (h1 : boris_initial = 24)
  (h2 : cameron_initial = 30) :
  boris_initial - boris_initial / 4 + cameron_initial - cameron_initial / 3 = 38 := by
  sorry

#check books_after_donation

end NUMINAMATH_CALUDE_books_after_donation_l440_44073


namespace NUMINAMATH_CALUDE_f_composition_of_three_l440_44002

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem f_composition_of_three : f (f (f (f 3))) = 24 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l440_44002


namespace NUMINAMATH_CALUDE_pizza_toppings_l440_44001

theorem pizza_toppings (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h_total : total_slices = 24)
  (h_pepperoni : pepperoni_slices = 15)
  (h_mushroom : mushroom_slices = 15)
  (h_at_least_one : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range pepperoni_slices ∨ slice ∈ Finset.range mushroom_slices)) :
  (Finset.range pepperoni_slices ∩ Finset.range mushroom_slices).card = 6 := by
sorry

end NUMINAMATH_CALUDE_pizza_toppings_l440_44001


namespace NUMINAMATH_CALUDE_tourist_contact_probability_l440_44083

/-- The probability that at least one tourist from the first group can contact at least one tourist from the second group -/
def contact_probability (p : ℝ) : ℝ := 1 - (1 - p)^42

/-- Theorem stating the probability of contact between two groups of tourists -/
theorem tourist_contact_probability (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  contact_probability p = 
  1 - (1 - p)^(6 * 7) :=
by sorry

end NUMINAMATH_CALUDE_tourist_contact_probability_l440_44083


namespace NUMINAMATH_CALUDE_selenes_purchase_cost_l440_44059

/-- The total cost of Selene's purchase after discount -/
def total_cost_after_discount (camera_price : ℝ) (frame_price : ℝ) (num_cameras : ℕ) (num_frames : ℕ) (discount_rate : ℝ) : ℝ :=
  let total_before_discount := camera_price * num_cameras + frame_price * num_frames
  let discount := discount_rate * total_before_discount
  total_before_discount - discount

/-- Theorem stating that Selene's total payment is $551 -/
theorem selenes_purchase_cost : 
  total_cost_after_discount 110 120 2 3 0.05 = 551 := by
  sorry

#eval total_cost_after_discount 110 120 2 3 0.05

end NUMINAMATH_CALUDE_selenes_purchase_cost_l440_44059


namespace NUMINAMATH_CALUDE_triangle_problem_l440_44074

theorem triangle_problem (a b c A B C : ℝ) 
  (h1 : 3 * a * Real.cos A = c * Real.cos B + b * Real.cos C)
  (h2 : a = 1)
  (h3 : Real.cos B + Real.cos C = 1) :
  Real.cos A = 1/3 ∧ c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l440_44074


namespace NUMINAMATH_CALUDE_min_third_side_of_triangle_l440_44068

theorem min_third_side_of_triangle (a b c : ℕ) : 
  (a + b + c) % 2 = 1 → -- perimeter is odd
  (a = b + 5 ∨ b = a + 5 ∨ a = c + 5 ∨ c = a + 5 ∨ b = c + 5 ∨ c = b + 5) → -- difference between two sides is 5
  c ≥ 6 -- minimum length of the third side is 6
  :=
by sorry

end NUMINAMATH_CALUDE_min_third_side_of_triangle_l440_44068


namespace NUMINAMATH_CALUDE_batsman_matches_l440_44013

theorem batsman_matches (x : ℕ) 
  (h1 : x > 0)
  (h2 : (30 * x + 15 * 10) / (x + 10) = 25) : 
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_batsman_matches_l440_44013


namespace NUMINAMATH_CALUDE_square_product_closed_l440_44078

def P : Set ℕ := {n : ℕ | ∃ m : ℕ+, n = m ^ 2}

theorem square_product_closed (a b : ℕ) (ha : a ∈ P) (hb : b ∈ P) : 
  a * b ∈ P := by sorry

end NUMINAMATH_CALUDE_square_product_closed_l440_44078


namespace NUMINAMATH_CALUDE_square_minus_one_roots_l440_44011

theorem square_minus_one_roots (x : ℝ) : x^2 - 1 = 0 → x = -1 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_one_roots_l440_44011


namespace NUMINAMATH_CALUDE_count_distinct_digits_eq_2688_l440_44049

/-- The number of integers between 1000 and 9999 with four distinct digits, none of which is '5' -/
def count_distinct_digits : ℕ :=
  let first_digit := 8  -- 9 digits excluding 5
  let second_digit := 8 -- 9 digits excluding 5 and the first digit
  let third_digit := 7  -- 8 digits excluding 5 and the first two digits
  let fourth_digit := 6 -- 7 digits excluding 5 and the first three digits
  first_digit * second_digit * third_digit * fourth_digit

theorem count_distinct_digits_eq_2688 : count_distinct_digits = 2688 := by
  sorry

end NUMINAMATH_CALUDE_count_distinct_digits_eq_2688_l440_44049


namespace NUMINAMATH_CALUDE_sum_of_odd_and_multiples_of_three_l440_44089

/-- The number of six-digit odd numbers -/
def A : ℕ := 450000

/-- The number of six-digit multiples of 3 -/
def B : ℕ := 300000

/-- The sum of six-digit odd numbers and six-digit multiples of 3 is 750000 -/
theorem sum_of_odd_and_multiples_of_three : A + B = 750000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_and_multiples_of_three_l440_44089


namespace NUMINAMATH_CALUDE_min_sum_with_prime_hcfs_l440_44051

/-- Given three positive integers with pairwise HCFs being distinct primes, 
    their sum is at least 31 -/
theorem min_sum_with_prime_hcfs (Q R S : ℕ+) 
  (hQR : ∃ (p : ℕ), Nat.Prime p ∧ Nat.gcd Q.val R.val = p)
  (hQS : ∃ (q : ℕ), Nat.Prime q ∧ Nat.gcd Q.val S.val = q)
  (hRS : ∃ (r : ℕ), Nat.Prime r ∧ Nat.gcd R.val S.val = r)
  (h_distinct : ∀ (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
    Nat.gcd Q.val R.val = p ∧ Nat.gcd Q.val S.val = q ∧ Nat.gcd R.val S.val = r →
    p ≠ q ∧ q ≠ r ∧ p ≠ r) :
  Q.val + R.val + S.val ≥ 31 := by
  sorry

#check min_sum_with_prime_hcfs

end NUMINAMATH_CALUDE_min_sum_with_prime_hcfs_l440_44051


namespace NUMINAMATH_CALUDE_major_selection_theorem_l440_44081

-- Define the number of majors
def total_majors : ℕ := 7

-- Define the number of majors to be selected
def selected_majors : ℕ := 3

-- Define a function to calculate the number of ways to select and order majors
def ways_to_select_and_order (total : ℕ) (select : ℕ) (excluded : ℕ) : ℕ :=
  (Nat.choose total select - Nat.choose (total - excluded) (select - excluded)) * Nat.factorial select

-- Theorem statement
theorem major_selection_theorem : 
  ways_to_select_and_order total_majors selected_majors 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_major_selection_theorem_l440_44081


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l440_44087

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (-1, 3)
  parallel a b → x = -1/3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l440_44087


namespace NUMINAMATH_CALUDE_distance_sum_property_l440_44060

/-- Linear mapping between two line segments -/
structure LinearSegmentMap (AB A'B' : ℝ) where
  scale : ℝ
  map_points : ℝ → ℝ
  map_property : ∀ x, map_points x = scale * x

/-- Representation of a point on a line segment -/
structure SegmentPoint (total_length : ℝ) where
  position : ℝ
  valid_position : 0 ≤ position ∧ position ≤ total_length

theorem distance_sum_property 
  (AB A'B' : ℝ) 
  (h_AB_pos : AB > 0)
  (h_A'B'_pos : A'B' > 0)
  (h_linear_map : LinearSegmentMap AB A'B')
  (D : SegmentPoint AB)
  (D' : SegmentPoint A'B')
  (h_D_midpoint : D.position = AB / 2)
  (h_D'_third : D'.position = A'B' / 3)
  (P : SegmentPoint AB)
  (P' : SegmentPoint A'B')
  (h_P'_mapped : P'.position = h_linear_map.map_points P.position)
  (h_AB_length : AB = 3)
  (h_A'B'_length : A'B' = 6)
  (a : ℝ)
  (h_x_eq_a : |P.position - D.position| = a) :
  |P.position - D.position| + |P'.position - D'.position| = 3 * a :=
sorry

end NUMINAMATH_CALUDE_distance_sum_property_l440_44060


namespace NUMINAMATH_CALUDE_felix_brother_lift_multiple_l440_44079

theorem felix_brother_lift_multiple :
  ∀ (felix_weight brother_weight : ℝ),
  felix_weight > 0 →
  brother_weight > 0 →
  1.5 * felix_weight = 150 →
  brother_weight = 2 * felix_weight →
  600 / brother_weight = 3 := by
  sorry

end NUMINAMATH_CALUDE_felix_brother_lift_multiple_l440_44079


namespace NUMINAMATH_CALUDE_num_dogs_correct_l440_44024

/-- The number of dogs Ella owns -/
def num_dogs : ℕ := 2

/-- The amount of food each dog eats per day (in scoops) -/
def food_per_dog : ℚ := 1/8

/-- The total amount of food eaten by all dogs per day (in scoops) -/
def total_food : ℚ := 1/4

/-- Theorem stating that the number of dogs is correct given the food consumption -/
theorem num_dogs_correct : (num_dogs : ℚ) * food_per_dog = total_food := by sorry

end NUMINAMATH_CALUDE_num_dogs_correct_l440_44024


namespace NUMINAMATH_CALUDE_rectangle_tiling_l440_44026

theorem rectangle_tiling (n : ℕ) : 
  (∃ (a b : ℕ), 3 * a + 2 * b + 3 * n = 63 ∧ b + n = 20) ↔ 
  n ∈ ({2, 5, 8, 11, 14, 17, 20} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_tiling_l440_44026


namespace NUMINAMATH_CALUDE_fraction_to_decimal_subtraction_l440_44007

theorem fraction_to_decimal_subtraction : (3 : ℚ) / 40 - 0.005 = 0.070 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_subtraction_l440_44007


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l440_44036

theorem mark_and_carolyn_money_sum : 
  let mark_money : ℚ := 3 / 4
  let carolyn_money : ℚ := 3 / 10
  mark_money + carolyn_money = 21 / 20 := by
sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l440_44036


namespace NUMINAMATH_CALUDE_degree_three_polynomial_l440_44003

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 5*x^4
def g (x : ℝ) : ℝ := 3 - 2*x - 6*x^3 + 7*x^4

-- Define the combined polynomial h
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

-- Theorem statement
theorem degree_three_polynomial (c : ℝ) :
  c = -5/7 → (∀ x, h c x = 1 + (-12 - 2*c)*x + (3*x^2) + (-4 - 6*c)*x^3) :=
by sorry

end NUMINAMATH_CALUDE_degree_three_polynomial_l440_44003


namespace NUMINAMATH_CALUDE_car_downhill_speed_l440_44062

/-- Proves that given specific conditions about a car's journey, the downhill speed is 60 km/hr -/
theorem car_downhill_speed 
  (uphill_speed : ℝ) 
  (uphill_distance : ℝ) 
  (downhill_distance : ℝ) 
  (average_speed : ℝ) 
  (h1 : uphill_speed = 30) 
  (h2 : uphill_distance = 100) 
  (h3 : downhill_distance = 50) 
  (h4 : average_speed = 36) : 
  ∃ downhill_speed : ℝ, 
    downhill_speed = 60 ∧ 
    average_speed = (uphill_distance + downhill_distance) / 
      (uphill_distance / uphill_speed + downhill_distance / downhill_speed) := by
  sorry

#check car_downhill_speed

end NUMINAMATH_CALUDE_car_downhill_speed_l440_44062


namespace NUMINAMATH_CALUDE_root_difference_l440_44096

/-- The polynomial coefficients -/
def a : ℚ := 8
def b : ℚ := -22
def c : ℚ := 15
def d : ℚ := -2

/-- The polynomial function -/
def f (x : ℚ) : ℚ := a * x^3 + b * x^2 + c * x + d

/-- The roots of the polynomial are in geometric progression -/
axiom roots_in_geometric_progression : ∃ (r₁ r₂ r₃ : ℚ), 
  (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) ∧ 
  (∃ (q : ℚ), r₂ = r₁ * q ∧ r₃ = r₂ * q)

/-- The theorem to be proved -/
theorem root_difference : 
  ∃ (r₁ r₃ : ℚ), (f r₁ = 0 ∧ f r₃ = 0) ∧ 
  (∀ r, f r = 0 → r₁ ≤ r ∧ r ≤ r₃) ∧
  (r₃ - r₁ = 33 / 14) := by
sorry

end NUMINAMATH_CALUDE_root_difference_l440_44096


namespace NUMINAMATH_CALUDE_terminating_decimal_expansion_of_7_625_l440_44063

theorem terminating_decimal_expansion_of_7_625 :
  (7 : ℚ) / 625 = (112 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_terminating_decimal_expansion_of_7_625_l440_44063


namespace NUMINAMATH_CALUDE_trajectory_of_M_center_l440_44033

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the properties of circle M
def M_externally_tangent_C₁ (M : ℝ × ℝ) : Prop :=
  ∃ r > 0, ∀ x y, C₁ x y → (x - M.1)^2 + (y - M.2)^2 = (r + 1)^2

def M_internally_tangent_C₂ (M : ℝ × ℝ) : Prop :=
  ∃ r > 0, ∀ x y, C₂ x y → (x - M.1)^2 + (y - M.2)^2 = (5 - r)^2

-- Theorem statement
theorem trajectory_of_M_center :
  ∀ M : ℝ × ℝ,
  M_externally_tangent_C₁ M →
  M_internally_tangent_C₂ M →
  M.1^2 / 9 + M.2^2 / 8 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_M_center_l440_44033


namespace NUMINAMATH_CALUDE_power_sum_and_division_simplification_l440_44093

theorem power_sum_and_division_simplification :
  3^123 + 9^5 / 9^3 = 3^123 + 81 :=
by sorry

end NUMINAMATH_CALUDE_power_sum_and_division_simplification_l440_44093


namespace NUMINAMATH_CALUDE_labourer_income_l440_44031

/-- Proves that the monthly income of a labourer is 69 given the described conditions -/
theorem labourer_income (
  avg_expenditure_6months : ℝ)
  (reduced_monthly_expense : ℝ)
  (savings : ℝ)
  (h1 : avg_expenditure_6months = 70)
  (h2 : reduced_monthly_expense = 60)
  (h3 : savings = 30)
  : ∃ (monthly_income : ℝ),
    monthly_income = 69 ∧
    6 * monthly_income < 6 * avg_expenditure_6months ∧
    4 * monthly_income = 4 * reduced_monthly_expense + (6 * avg_expenditure_6months - 6 * monthly_income) + savings :=
by
  sorry

end NUMINAMATH_CALUDE_labourer_income_l440_44031


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_negation_l440_44023

theorem necessary_not_sufficient_negation (p q : Prop) 
  (h1 : q → p)  -- p is necessary for q
  (h2 : ¬(p → q))  -- p is not sufficient for q
  : (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_negation_l440_44023


namespace NUMINAMATH_CALUDE_initial_overs_is_ten_l440_44045

/-- Represents a cricket game scenario -/
structure CricketGame where
  targetScore : ℕ
  initialRunRate : ℚ
  remainingOvers : ℕ
  requiredRunRate : ℚ

/-- Calculates the number of overs played initially -/
def initialOvers (game : CricketGame) : ℚ :=
  (game.targetScore - game.requiredRunRate * game.remainingOvers) / (game.initialRunRate - game.requiredRunRate)

/-- Theorem stating that the number of overs played initially is 10 -/
theorem initial_overs_is_ten (game : CricketGame) 
  (h1 : game.targetScore = 282)
  (h2 : game.initialRunRate = 4.8)
  (h3 : game.remainingOvers = 40)
  (h4 : game.requiredRunRate = 5.85) :
  initialOvers game = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_overs_is_ten_l440_44045


namespace NUMINAMATH_CALUDE_product_mod_sixty_l440_44085

theorem product_mod_sixty (m : ℕ) : 
  198 * 953 ≡ m [ZMOD 60] → 0 ≤ m → m < 60 → m = 54 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_sixty_l440_44085


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l440_44035

theorem quadratic_root_implies_a_value (x a : ℝ) : 
  x = 1 → x^2 + a*x - 2 = 0 → a = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l440_44035


namespace NUMINAMATH_CALUDE_cuboid_area_example_l440_44069

/-- The surface area of a cuboid -/
def cuboid_surface_area (length breadth height : ℝ) : ℝ :=
  2 * (length * breadth + breadth * height + length * height)

/-- Theorem: The surface area of a cuboid with length 12, breadth 6, and height 10 is 504 -/
theorem cuboid_area_example : cuboid_surface_area 12 6 10 = 504 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_area_example_l440_44069


namespace NUMINAMATH_CALUDE_product_equality_l440_44021

theorem product_equality (a b c : ℝ) 
  (h : ∀ x y z : ℝ, x * y * z = Real.sqrt ((x + 2) * (y + 3)) / (z + 1)) : 
  6 * 15 * 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l440_44021


namespace NUMINAMATH_CALUDE_least_faces_combined_l440_44032

/-- Represents a fair die with faces numbered from 1 to n -/
structure Die (n : ℕ) where
  faces : Fin n → ℕ
  is_fair : ∀ i : Fin n, faces i = i.val + 1

/-- Represents a pair of dice -/
structure DicePair (a b : ℕ) where
  die1 : Die a
  die2 : Die b
  die2_numbering : ∀ i : Fin b, die2.faces i = 2 * i.val + 2

/-- Probability of rolling a specific sum with a pair of dice -/
def prob_sum (d : DicePair a b) (sum : ℕ) : ℚ :=
  (Fintype.card {(i, j) : Fin a × Fin b | d.die1.faces i + d.die2.faces j = sum} : ℚ) / (a * b)

/-- The main theorem stating the least possible number of faces on two dice combined -/
theorem least_faces_combined (a b : ℕ) (d : DicePair a b) :
  (prob_sum d 8 = 2 * prob_sum d 12) →
  (prob_sum d 13 = 2 * prob_sum d 8) →
  a + b ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_least_faces_combined_l440_44032


namespace NUMINAMATH_CALUDE_quadratic_single_solution_l440_44077

theorem quadratic_single_solution (a : ℝ) (ha : a ≠ 0) :
  (∃! x : ℝ, a * x^2 + 20 * x + 7 = 0) → 
  (∀ x : ℝ, a * x^2 + 20 * x + 7 = 0 → x = -7/10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_single_solution_l440_44077


namespace NUMINAMATH_CALUDE_shearer_payment_l440_44056

/-- Given the following conditions:
  - The number of sheep is 200
  - Each sheep produces 10 pounds of wool
  - The price of wool is $20 per pound
  - The profit is $38000
  Prove that the amount paid to the shearer is $2000 -/
theorem shearer_payment (num_sheep : ℕ) (wool_per_sheep : ℕ) (wool_price : ℕ) (profit : ℕ) :
  num_sheep = 200 →
  wool_per_sheep = 10 →
  wool_price = 20 →
  profit = 38000 →
  num_sheep * wool_per_sheep * wool_price - profit = 2000 := by
  sorry

end NUMINAMATH_CALUDE_shearer_payment_l440_44056


namespace NUMINAMATH_CALUDE_function_transformation_l440_44005

/-- Given a function f such that f(x-1) = x^2 + 4x - 5 for all x,
    prove that f(x) = x^2 + 6x for all x. -/
theorem function_transformation (f : ℝ → ℝ) 
    (h : ∀ x, f (x - 1) = x^2 + 4*x - 5) : 
    ∀ x, f x = x^2 + 6*x := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l440_44005


namespace NUMINAMATH_CALUDE_salt_mixture_price_l440_44015

theorem salt_mixture_price (initial_salt_weight : ℝ) (initial_salt_price : ℝ) 
  (new_salt_weight : ℝ) (selling_price : ℝ) (profit_percentage : ℝ) :
  initial_salt_weight = 40 ∧ 
  initial_salt_price = 0.35 ∧
  new_salt_weight = 5 ∧
  selling_price = 0.48 ∧
  profit_percentage = 0.2 →
  ∃ (new_salt_price : ℝ),
    new_salt_price = 0.80 ∧
    (initial_salt_weight * initial_salt_price + new_salt_weight * new_salt_price) * 
      (1 + profit_percentage) = 
    (initial_salt_weight + new_salt_weight) * selling_price :=
by sorry

end NUMINAMATH_CALUDE_salt_mixture_price_l440_44015


namespace NUMINAMATH_CALUDE_correct_product_l440_44029

theorem correct_product (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ b ≥ 10 ∧ b < 100 →
  (a * (10 * (b % 10) + (b / 10)) = 143) →
  a * b = 341 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_product_l440_44029


namespace NUMINAMATH_CALUDE_range_a_theorem_l440_44067

/-- Proposition p: The solution set of the inequality x^2+(a-1)x+1≤0 is the empty set ∅ -/
def p (a : ℝ) : Prop :=
  ∀ x, x^2 + (a-1)*x + 1 > 0

/-- Proposition q: The function y=(a-1)^x is an increasing function -/
def q (a : ℝ) : Prop :=
  ∀ x y, x < y → (a-1)^x < (a-1)^y

/-- The range of a satisfying the given conditions -/
def range_a : Set ℝ :=
  {a | (-1 < a ∧ a ≤ 2) ∨ a ≥ 3}

/-- Theorem stating that given the conditions, the range of a is as specified -/
theorem range_a_theorem (a : ℝ) :
  (¬(p a ∧ q a)) → (p a ∨ q a) → a ∈ range_a :=
by sorry

end NUMINAMATH_CALUDE_range_a_theorem_l440_44067


namespace NUMINAMATH_CALUDE_inequality_proof_l440_44070

theorem inequality_proof (x y : ℝ) : x + y + Real.sqrt (x * y) ≤ 3 * (x + y - Real.sqrt (x * y)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l440_44070


namespace NUMINAMATH_CALUDE_refrigerator_price_l440_44084

theorem refrigerator_price (refrigerator washing_machine : ℕ) 
  (h1 : washing_machine = refrigerator - 1490)
  (h2 : refrigerator + washing_machine = 7060) : 
  refrigerator = 4275 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_price_l440_44084


namespace NUMINAMATH_CALUDE_pill_supply_duration_l440_44066

/-- Proves that a supply of pills lasts for a specific number of months -/
theorem pill_supply_duration (total_pills : ℕ) (days_per_pill : ℕ) (days_per_month : ℕ) 
  (h1 : total_pills = 120)
  (h2 : days_per_pill = 2)
  (h3 : days_per_month = 30) :
  (total_pills * days_per_pill) / days_per_month = 8 := by
  sorry

#check pill_supply_duration

end NUMINAMATH_CALUDE_pill_supply_duration_l440_44066


namespace NUMINAMATH_CALUDE_no_natural_power_pair_l440_44004

theorem no_natural_power_pair : ¬∃ (x y : ℕ), 
  (∃ (k : ℕ), x^2 + x + 1 = y^k) ∧ 
  (∃ (m : ℕ), y^2 + y + 1 = x^m) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_power_pair_l440_44004


namespace NUMINAMATH_CALUDE_church_trip_distance_l440_44097

def trip_distance (speed1 speed2 speed3 : Real) (time : Real) : Real :=
  (speed1 * time + speed2 * time + speed3 * time)

theorem church_trip_distance :
  let speed1 : Real := 16
  let speed2 : Real := 12
  let speed3 : Real := 20
  let time : Real := 15 / 60
  trip_distance speed1 speed2 speed3 time = 12 := by
  sorry

end NUMINAMATH_CALUDE_church_trip_distance_l440_44097


namespace NUMINAMATH_CALUDE_arithmetic_progression_ratio_l440_44095

/-- Given three numbers in arithmetic progression where the largest is 70
    and the difference between the smallest and largest is 40,
    prove that their ratio is 3:5:7 -/
theorem arithmetic_progression_ratio :
  ∀ (a b c : ℕ),
  c = 70 →
  c - a = 40 →
  b - a = c - b →
  ∃ (k : ℕ), k > 0 ∧ a = 3 * k ∧ b = 5 * k ∧ c = 7 * k :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_ratio_l440_44095


namespace NUMINAMATH_CALUDE_expected_red_pairs_51_17_l440_44030

/-- The expected number of red adjacent pairs in a circular arrangement of cards -/
def expected_red_pairs (total_cards : ℕ) (red_cards : ℕ) : ℚ :=
  (total_cards : ℚ) * (red_cards : ℚ) / (total_cards : ℚ) * ((red_cards - 1) : ℚ) / ((total_cards - 1) : ℚ)

/-- Theorem: Expected number of red adjacent pairs in a specific card arrangement -/
theorem expected_red_pairs_51_17 :
  expected_red_pairs 51 17 = 464 / 85 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_pairs_51_17_l440_44030


namespace NUMINAMATH_CALUDE_corrected_mean_is_89_42857142857143_l440_44099

def initial_scores : List ℝ := [85, 90, 87, 93, 89, 84, 88]

def corrected_scores : List ℝ := 
  [85, 90, 87 + 5, 93, 89, 84 + 5, 88]

theorem corrected_mean_is_89_42857142857143 : 
  (corrected_scores.sum / corrected_scores.length : ℝ) = 89.42857142857143 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_is_89_42857142857143_l440_44099


namespace NUMINAMATH_CALUDE_four_times_hash_58_l440_44017

-- Define the function #
def hash (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem four_times_hash_58 : hash (hash (hash (hash 58))) = 11.8688 := by
  sorry

end NUMINAMATH_CALUDE_four_times_hash_58_l440_44017


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l440_44016

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (4 - 5 * x) = 8 → x = -12 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l440_44016


namespace NUMINAMATH_CALUDE_corn_height_after_three_weeks_l440_44043

/-- The height of corn plants after three weeks of growth -/
def cornHeight (firstWeekGrowth : ℕ) : ℕ :=
  let secondWeekGrowth := 2 * firstWeekGrowth
  let thirdWeekGrowth := 4 * secondWeekGrowth
  firstWeekGrowth + secondWeekGrowth + thirdWeekGrowth

/-- Theorem stating that the corn plants grow to 22 inches after three weeks -/
theorem corn_height_after_three_weeks :
  cornHeight 2 = 22 := by
  sorry


end NUMINAMATH_CALUDE_corn_height_after_three_weeks_l440_44043


namespace NUMINAMATH_CALUDE_remainder_of_7n_mod_4_l440_44098

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 2) : (7 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_7n_mod_4_l440_44098


namespace NUMINAMATH_CALUDE_initial_train_distance_l440_44014

/-- Calculates the initial distance between two trains given their lengths, speeds, and time to meet. -/
theorem initial_train_distance
  (length1 : ℝ)
  (length2 : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (time : ℝ)
  (h1 : length1 = 100)
  (h2 : length2 = 200)
  (h3 : speed1 = 54)
  (h4 : speed2 = 72)
  (h5 : time = 1.999840012798976) :
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  let distance_covered := relative_speed * time * 3600
  distance_covered - (length1 + length2) = 251680.84161264498 :=
by sorry

end NUMINAMATH_CALUDE_initial_train_distance_l440_44014


namespace NUMINAMATH_CALUDE_friends_eating_pizza_l440_44009

/-- The number of friends eating pizza with Ron -/
def num_friends : ℕ := 2

/-- The number of slices in the pizza -/
def total_slices : ℕ := 12

/-- The number of slices each person ate -/
def slices_per_person : ℕ := 4

/-- The total number of people eating, including Ron -/
def total_people : ℕ := total_slices / slices_per_person

theorem friends_eating_pizza : 
  num_friends = total_people - 1 :=
sorry

end NUMINAMATH_CALUDE_friends_eating_pizza_l440_44009


namespace NUMINAMATH_CALUDE_chord_length_is_16_l440_44052

/-- Represents a line in polar form -/
structure PolarLine where
  equation : ℝ → ℝ → ℝ

/-- Represents a circle in parametric form -/
structure ParametricCircle where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Calculates the length of a chord on a circle cut by a line -/
noncomputable def chordLength (l : PolarLine) (c : ParametricCircle) : ℝ :=
  sorry

/-- The main theorem stating that the chord length is 16 -/
theorem chord_length_is_16 :
  let l : PolarLine := { equation := λ ρ θ => ρ * Real.sin (θ - Real.pi / 3) - 6 }
  let c : ParametricCircle := { x := λ θ => 10 * Real.cos θ, y := λ θ => 10 * Real.sin θ }
  chordLength l c = 16 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_is_16_l440_44052


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l440_44037

theorem fraction_equation_solution :
  ∃ x : ℚ, (5 * x + 3) / (7 * x - 4) = 4128 / 4386 ∧ x = 115 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l440_44037


namespace NUMINAMATH_CALUDE_continued_fraction_evaluation_l440_44044

theorem continued_fraction_evaluation :
  1 + 2 / (3 + 4 / (5 + 6 / 7)) = 233 / 151 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_evaluation_l440_44044


namespace NUMINAMATH_CALUDE_otimes_k_otimes_k_l440_44094

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := x^3 + y - 2*x

-- Theorem statement
theorem otimes_k_otimes_k (k : ℝ) : otimes k (otimes k k) = 2*k^3 - 3*k := by
  sorry

end NUMINAMATH_CALUDE_otimes_k_otimes_k_l440_44094


namespace NUMINAMATH_CALUDE_system_solution_l440_44076

theorem system_solution (a b c d x y z : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : x + y + z = 1)
  (h2 : a * x + b * y + c * z = d)
  (h3 : a^2 * x + b^2 * y + c^2 * z = d^2) :
  x = (d - c) * (b - d) / ((a - c) * (b - a)) ∧
  y = (a - d) * (c - d) / ((b - a) * (b - c)) ∧
  z = (a - d) * (d - b) / ((a - c) * (c - b)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l440_44076


namespace NUMINAMATH_CALUDE_cubic_equation_value_l440_44091

theorem cubic_equation_value (x : ℝ) (h : 2 * x^2 - 3 * x - 2022 = 0) :
  2 * x^3 - x^2 - 2025 * x - 2020 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l440_44091


namespace NUMINAMATH_CALUDE_time_reduction_fraction_l440_44022

theorem time_reduction_fraction (actual_speed : ℝ) (speed_increase : ℝ) : 
  actual_speed = 36.000000000000014 →
  speed_increase = 18 →
  (actual_speed / (actual_speed + speed_increase)) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_time_reduction_fraction_l440_44022


namespace NUMINAMATH_CALUDE_inequality_solution_l440_44072

theorem inequality_solution (x : ℝ) : 
  (x + 2) / (x + 3) > (4 * x + 5) / (3 * x + 10) ↔ 
  (x > -10/3 ∧ x < -3) ∨ (x > (-1 - Real.sqrt 61) / 6 ∧ x < (-1 + Real.sqrt 61) / 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l440_44072


namespace NUMINAMATH_CALUDE_x_inequality_l440_44057

theorem x_inequality (x : ℝ) : (x < 0 ∧ x < 1 / (4 * x)) ↔ (-1/2 < x ∧ x < 0) :=
sorry

end NUMINAMATH_CALUDE_x_inequality_l440_44057


namespace NUMINAMATH_CALUDE_area_under_curve_l440_44075

/-- The area enclosed by the curve y = x^2 + 1, the coordinate axes, and the line x = 1 is 4/3 -/
theorem area_under_curve : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 1
  ∫ x in (0 : ℝ)..1, f x = 4/3 := by sorry

end NUMINAMATH_CALUDE_area_under_curve_l440_44075


namespace NUMINAMATH_CALUDE_award_distribution_theorem_l440_44071

-- Define the number of awards and students
def num_awards : ℕ := 7
def num_students : ℕ := 4

-- Function to calculate the number of ways to distribute awards
def distribute_awards (awards : ℕ) (students : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

-- Theorem stating that the number of ways to distribute awards is 3920
theorem award_distribution_theorem :
  distribute_awards num_awards num_students = 3920 :=
by sorry

end NUMINAMATH_CALUDE_award_distribution_theorem_l440_44071


namespace NUMINAMATH_CALUDE_at_most_one_zero_point_l440_44027

/-- A decreasing function on a closed interval has at most one zero point -/
theorem at_most_one_zero_point 
  {f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b) (h_decr : ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x) :
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0 ∨ ∀ x, a ≤ x → x ≤ b → f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_zero_point_l440_44027
