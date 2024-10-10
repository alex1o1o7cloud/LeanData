import Mathlib

namespace percentage_not_sold_approx_l2780_278093

def initial_stock : ℕ := 1200

def monday_sold : ℕ := 75
def monday_returned : ℕ := 6

def tuesday_sold : ℕ := 50

def wednesday_sold : ℕ := 64
def wednesday_returned : ℕ := 8

def thursday_sold : ℕ := 78

def friday_sold : ℕ := 135
def friday_returned : ℕ := 5

def total_sold : ℕ := 
  (monday_sold - monday_returned) + 
  tuesday_sold + 
  (wednesday_sold - wednesday_returned) + 
  thursday_sold + 
  (friday_sold - friday_returned)

def books_not_sold : ℕ := initial_stock - total_sold

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem percentage_not_sold_approx :
  abs (percentage_not_sold - 68.08) < 0.01 := by sorry

end percentage_not_sold_approx_l2780_278093


namespace f_odd_and_decreasing_l2780_278045

def f (x : ℝ) : ℝ := -x^3 - x

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) := by
  sorry

end f_odd_and_decreasing_l2780_278045


namespace geometric_sequence_first_term_l2780_278066

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The factorial of a natural number n, denoted as n!, is the product of all positive integers less than or equal to n. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h_sixth : a 6 = factorial 9)
  (h_ninth : a 9 = factorial 10) :
  a 1 = (factorial 9 : ℝ) / (10 ^ (5/3)) :=
sorry

end geometric_sequence_first_term_l2780_278066


namespace not_both_odd_l2780_278023

theorem not_both_odd (m n : ℕ) (h : (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 2020) : 
  ¬(Odd m ∧ Odd n) := by
sorry

end not_both_odd_l2780_278023


namespace fred_weekend_earnings_l2780_278001

def newspaper_earnings : ℕ := 16
def car_washing_earnings : ℕ := 74
def lawn_mowing_earnings : ℕ := 45
def lemonade_earnings : ℕ := 22
def yard_work_earnings : ℕ := 30

theorem fred_weekend_earnings :
  newspaper_earnings + car_washing_earnings + lawn_mowing_earnings + lemonade_earnings + yard_work_earnings = 187 := by
  sorry

end fred_weekend_earnings_l2780_278001


namespace intersection_M_N_l2780_278081

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x < 0}

-- Define set N
def N : Set ℝ := {x | |x| < 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end intersection_M_N_l2780_278081


namespace terminal_point_coordinates_l2780_278030

/-- Given sin α = 3/5 and cos α = -4/5, the coordinates of the point on the terminal side of angle α are (-4, 3). -/
theorem terminal_point_coordinates (α : Real) 
  (h1 : Real.sin α = 3/5) 
  (h2 : Real.cos α = -4/5) : 
  ∃ (x y : Real), x = -4 ∧ y = 3 ∧ Real.sin α = y / Real.sqrt (x^2 + y^2) ∧ Real.cos α = x / Real.sqrt (x^2 + y^2) := by
  sorry

end terminal_point_coordinates_l2780_278030


namespace power_mod_23_l2780_278067

theorem power_mod_23 : 17^1499 % 23 = 11 := by
  sorry

end power_mod_23_l2780_278067


namespace snack_store_spending_l2780_278072

/-- The amount Ben spent at the snack store -/
def ben_spent : ℝ := 60

/-- The amount David spent at the snack store -/
def david_spent : ℝ := 45

/-- For every dollar Ben spent, David spent 25 cents less -/
axiom david_spent_less : david_spent = ben_spent - 0.25 * ben_spent

/-- Ben paid $15 more than David -/
axiom ben_paid_more : ben_spent = david_spent + 15

/-- The total amount spent by Ben and David -/
def total_spent : ℝ := ben_spent + david_spent

theorem snack_store_spending : total_spent = 105 := by
  sorry

end snack_store_spending_l2780_278072


namespace tan_value_from_ratio_l2780_278011

theorem tan_value_from_ratio (α : Real) :
  (Real.sin α + 7 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = -5 →
  Real.tan α = -2 := by
  sorry

end tan_value_from_ratio_l2780_278011


namespace power_of_two_sum_l2780_278004

theorem power_of_two_sum (m n : ℕ+) (a b : ℝ) 
  (h1 : 2^(m : ℕ) = a) 
  (h2 : 2^(n : ℕ) = b) : 
  2^((m + n : ℕ+) : ℕ) = a * b := by
  sorry

end power_of_two_sum_l2780_278004


namespace tomato_theorem_l2780_278032

def tomato_problem (plant1 plant2 plant3 plant4 plant5 plant6 plant7 plant8 plant9 : ℕ) : Prop :=
  plant1 = 15 ∧
  plant2 = 2 * plant1 - 8 ∧
  plant3 = (plant1^2) / 3 ∧
  plant4 = (plant1 + plant2) / 2 ∧
  plant5 = 3 * Int.sqrt (plant1 + plant2) ∧
  plant6 = plant5 ∧
  plant7 = (3 * (plant1 + plant2 + plant3)) / 2 ∧
  plant8 = plant7 ∧
  plant9 = plant1 + plant7 + 6 →
  plant1 + plant2 + plant3 + plant4 + plant5 + plant6 + plant7 + plant8 + plant9 = 692

theorem tomato_theorem : ∃ plant1 plant2 plant3 plant4 plant5 plant6 plant7 plant8 plant9 : ℕ,
  tomato_problem plant1 plant2 plant3 plant4 plant5 plant6 plant7 plant8 plant9 :=
by
  sorry

end tomato_theorem_l2780_278032


namespace bike_vs_drive_time_difference_l2780_278046

theorem bike_vs_drive_time_difference 
  (normal_drive_time : ℝ) 
  (normal_drive_speed : ℝ) 
  (bike_route_reduction : ℝ) 
  (min_bike_speed : ℝ) 
  (max_bike_speed : ℝ) 
  (h1 : normal_drive_time = 45) 
  (h2 : normal_drive_speed = 40) 
  (h3 : bike_route_reduction = 0.2) 
  (h4 : min_bike_speed = 12) 
  (h5 : max_bike_speed = 16) : 
  ∃ (time_difference : ℝ), time_difference = 75 := by
sorry

end bike_vs_drive_time_difference_l2780_278046


namespace cos_shift_proof_l2780_278025

theorem cos_shift_proof (φ : Real) (h1 : 0 < φ) (h2 : φ < π / 2) : 
  let f := λ x : Real => 2 * Real.cos (2 * x)
  let g := λ x : Real => 2 * Real.cos (2 * x - 2 * φ)
  (∃ x₁ x₂ : Real, |f x₁ - g x₂| = 4 ∧ |x₁ - x₂| = π / 6) → φ = π / 3 := by
  sorry

end cos_shift_proof_l2780_278025


namespace f_max_value_l2780_278039

/-- The function f(x) = 3x - x^3 -/
def f (x : ℝ) : ℝ := 3 * x - x^3

/-- The theorem stating that the maximum value of f(x) = 3x - x^3 is 2 for 0 ≤ x ≤ √3 -/
theorem f_max_value :
  ∃ (M : ℝ), M = 2 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 3 → f x ≤ M) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 3 ∧ f x = M) :=
sorry

end f_max_value_l2780_278039


namespace koi_fish_count_l2780_278037

/-- Calculates the number of koi fish after three weeks -/
def koi_fish_after_three_weeks (initial_total : ℕ) (initial_goldfish : ℕ) (koi_added_per_day : ℕ) (goldfish_added_per_day : ℕ) (days : ℕ) (final_goldfish : ℕ) : ℕ :=
  let initial_koi := initial_total - initial_goldfish
  let total_koi_added := koi_added_per_day * days
  initial_koi + total_koi_added

theorem koi_fish_count (initial_total : ℕ) (koi_added_per_day : ℕ) (goldfish_added_per_day : ℕ) (days : ℕ) (final_goldfish : ℕ) 
    (h1 : initial_total = 280)
    (h2 : koi_added_per_day = 2)
    (h3 : goldfish_added_per_day = 5)
    (h4 : days = 21)
    (h5 : final_goldfish = 200) :
  koi_fish_after_three_weeks initial_total (initial_total - (final_goldfish - goldfish_added_per_day * days)) koi_added_per_day goldfish_added_per_day days final_goldfish = 227 := by
  sorry

#eval koi_fish_after_three_weeks 280 95 2 5 21 200

end koi_fish_count_l2780_278037


namespace rectangular_box_volume_l2780_278092

theorem rectangular_box_volume (l w h : ℝ) 
  (face1 : l * w = 30)
  (face2 : w * h = 20)
  (face3 : l * h = 12) :
  l * w * h = 60 := by
  sorry

end rectangular_box_volume_l2780_278092


namespace rectangle_ratio_l2780_278091

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0) : 
  (s + 2*y = 3*s) → (x + s = 3*s) → (x/y = 2) := by
  sorry

end rectangle_ratio_l2780_278091


namespace lydia_road_trip_fuel_usage_l2780_278054

/-- Proves that given the conditions of Lydia's road trip, the fraction of fuel used in the second third is 1/3 --/
theorem lydia_road_trip_fuel_usage 
  (total_fuel : ℝ) 
  (first_third_fuel : ℝ) 
  (h1 : total_fuel = 60) 
  (h2 : first_third_fuel = 30) 
  (h3 : ∃ (second_third_fraction : ℝ), 
    first_third_fuel + second_third_fraction * total_fuel + (second_third_fraction / 2) * total_fuel = total_fuel) :
  ∃ (second_third_fraction : ℝ), second_third_fraction = 1/3 := by
sorry


end lydia_road_trip_fuel_usage_l2780_278054


namespace power_three_multiplication_l2780_278086

theorem power_three_multiplication : 6^3 * 7^3 = 74088 := by
  sorry

end power_three_multiplication_l2780_278086


namespace fixed_point_on_curve_l2780_278069

/-- The curve equation for any real m and n -/
def curve_equation (x y m n : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 2*n*y + 4*(m - n - 2) = 0

/-- Theorem stating that the point (2, -2) lies on the curve for all real m and n -/
theorem fixed_point_on_curve :
  ∀ (m n : ℝ), curve_equation 2 (-2) m n := by
  sorry

end fixed_point_on_curve_l2780_278069


namespace smallest_satisfying_congruences_l2780_278080

theorem smallest_satisfying_congruences : 
  ∃ (x : ℕ), x > 0 ∧ 
    x % 3 = 2 ∧ 
    x % 5 = 3 ∧ 
    x % 7 = 2 ∧
    (∀ (y : ℕ), y > 0 ∧ y % 3 = 2 ∧ y % 5 = 3 ∧ y % 7 = 2 → x ≤ y) ∧
    x = 23 :=
by sorry

end smallest_satisfying_congruences_l2780_278080


namespace x_plus_2y_equals_12_l2780_278014

theorem x_plus_2y_equals_12 (x y : ℝ) (h1 : x = 6) (h2 : y = 3) : x + 2*y = 12 := by
  sorry

end x_plus_2y_equals_12_l2780_278014


namespace sum_of_third_and_fourth_terms_l2780_278074

theorem sum_of_third_and_fourth_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, S n = n^2 + n) → a 3 + a 4 = 14 := by
  sorry

end sum_of_third_and_fourth_terms_l2780_278074


namespace parallel_not_sufficient_nor_necessary_l2780_278090

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def parallel (m : Line) (α : Plane) : Prop := sorry
def perpendicular (m : Line) (β : Plane) : Prop := sorry
def planes_perpendicular (α β : Plane) : Prop := sorry

-- Theorem statement
theorem parallel_not_sufficient_nor_necessary 
  (m : Line) (α β : Plane) 
  (h_perp : planes_perpendicular α β) :
  ¬(∀ m α β, parallel m α → perpendicular m β) ∧ 
  ¬(∀ m α β, perpendicular m β → parallel m α) := by
  sorry

end parallel_not_sufficient_nor_necessary_l2780_278090


namespace f_of_2_equals_7_l2780_278026

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + 2*x - 1

-- State the theorem
theorem f_of_2_equals_7 : f 2 = 7 := by
  sorry

end f_of_2_equals_7_l2780_278026


namespace sandy_dozens_of_marbles_l2780_278029

def melanie_marbles : ℕ := 84
def sandy_multiplier : ℕ := 8
def marbles_per_dozen : ℕ := 12

theorem sandy_dozens_of_marbles :
  (melanie_marbles * sandy_multiplier) / marbles_per_dozen = 56 := by
  sorry

end sandy_dozens_of_marbles_l2780_278029


namespace pencils_sold_initially_l2780_278059

-- Define the number of pencils sold at 15% gain
def pencils_at_gain : ℝ := 7.391304347826086

-- Theorem statement
theorem pencils_sold_initially (x : ℝ) :
  (0.85 * x * (1 / (1.15 * pencils_at_gain)) = 1) →
  x = 10 := by
  sorry

end pencils_sold_initially_l2780_278059


namespace antonio_age_is_51_months_l2780_278058

-- Define Isabella's age in months after 18 months
def isabella_age_after_18_months : ℕ := 10 * 12

-- Define the current time difference in months
def time_difference : ℕ := 18

-- Define Isabella's current age in months
def isabella_current_age : ℕ := isabella_age_after_18_months - time_difference

-- Define the relationship between Isabella and Antonio's ages
def antonio_age : ℕ := isabella_current_age / 2

-- Theorem to prove
theorem antonio_age_is_51_months : antonio_age = 51 := by
  sorry


end antonio_age_is_51_months_l2780_278058


namespace midpoint_locus_l2780_278008

/-- The locus of midpoints between a fixed point and points on a circle -/
theorem midpoint_locus (P : ℝ × ℝ) (c : Set (ℝ × ℝ)) :
  P = (4, -2) →
  c = {(x, y) | x^2 + y^2 = 4} →
  {(x, y) | ∃ (a : ℝ × ℝ), a ∈ c ∧ (x, y) = ((P.1 + a.1) / 2, (P.2 + a.2) / 2)} =
  {(x, y) | (x - 2)^2 + (y + 1)^2 = 1} :=
by sorry

end midpoint_locus_l2780_278008


namespace second_discount_percentage_l2780_278035

theorem second_discount_percentage 
  (initial_price : ℝ) 
  (first_discount : ℝ) 
  (final_price : ℝ) 
  (x : ℝ) :
  initial_price = 1000 →
  first_discount = 15 →
  final_price = 830 →
  initial_price * (1 - first_discount / 100) * (1 - x / 100) = final_price :=
by sorry

end second_discount_percentage_l2780_278035


namespace jerry_gabriel_toy_difference_l2780_278047

theorem jerry_gabriel_toy_difference (jerry gabriel jaxon : ℕ) 
  (h1 : jerry > gabriel)
  (h2 : gabriel = 2 * jaxon)
  (h3 : jaxon = 15)
  (h4 : jerry + gabriel + jaxon = 83) :
  jerry - gabriel = 8 := by
  sorry

end jerry_gabriel_toy_difference_l2780_278047


namespace smallest_solution_congruence_l2780_278015

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 34 = 17 % 34 ∧ ∀ (y : ℕ), y > 0 → (5 * y) % 34 = 17 % 34 → x ≤ y :=
by sorry

end smallest_solution_congruence_l2780_278015


namespace proposition_2_l2780_278040

-- Define the basic types
variable (Point : Type)
variable (Line : Type)
variable (Plane : Type)

-- Define the relationships
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the proposition we want to prove
theorem proposition_2 
  (m n : Line) (α : Plane) :
  perpendicular_plane m α → parallel m n → perpendicular_plane n α :=
sorry

end proposition_2_l2780_278040


namespace no_four_binomial_coeff_arithmetic_progression_l2780_278099

theorem no_four_binomial_coeff_arithmetic_progression :
  ∀ n m : ℕ, n > 0 → m > 0 → m + 3 ≤ n →
  ¬∃ d : ℕ, 
    (Nat.choose n (m+1) = Nat.choose n m + d) ∧
    (Nat.choose n (m+2) = Nat.choose n (m+1) + d) ∧
    (Nat.choose n (m+3) = Nat.choose n (m+2) + d) :=
by sorry

end no_four_binomial_coeff_arithmetic_progression_l2780_278099


namespace vote_count_theorem_l2780_278022

/-- The number of ways to count votes such that candidate A always leads candidate B -/
def vote_count_ways (a b : ℕ) : ℕ :=
  (Nat.factorial (a + b - 1)) / (Nat.factorial (a - 1) * Nat.factorial b) -
  (Nat.factorial (a + b - 1)) / (Nat.factorial a * Nat.factorial (b - 1))

/-- Theorem stating the number of ways for candidate A to maintain a lead throughout the counting process -/
theorem vote_count_theorem (a b : ℕ) (h : a > b) :
  vote_count_ways a b = (Nat.factorial (a + b - 1)) / (Nat.factorial (a - 1) * Nat.factorial b) -
                        (Nat.factorial (a + b - 1)) / (Nat.factorial a * Nat.factorial (b - 1)) :=
by sorry

end vote_count_theorem_l2780_278022


namespace cross_figure_sum_l2780_278089

-- Define the set of digits
def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the structure
structure CrossFigure where
  vertical : Fin 3 → Nat
  horizontal1 : Fin 3 → Nat
  horizontal2 : Fin 3 → Nat
  all_digits : List Nat
  h_vertical_sum : vertical 0 + vertical 1 + vertical 2 = 17
  h_horizontal1_sum : horizontal1 0 + horizontal1 1 + horizontal1 2 = 18
  h_horizontal2_sum : horizontal2 0 + horizontal2 1 + horizontal2 2 = 13
  h_intersection1 : vertical 0 = horizontal1 0
  h_intersection2 : vertical 2 = horizontal2 0
  h_all_digits : all_digits.length = 7
  h_all_digits_unique : all_digits.Nodup
  h_all_digits_in_set : ∀ d ∈ all_digits, d ∈ Digits
  h_all_digits_cover : (vertical 0 :: vertical 1 :: vertical 2 :: 
                        horizontal1 1 :: horizontal1 2 :: 
                        horizontal2 1 :: horizontal2 2 :: []).toFinset = all_digits.toFinset

theorem cross_figure_sum (cf : CrossFigure) : 
  cf.all_digits.sum = 34 := by
  sorry

end cross_figure_sum_l2780_278089


namespace hyperbola_equation_l2780_278048

/-- Hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_ecc : e = 5/3

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a hyperbola and a point, check if the point is on the hyperbola -/
def is_on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Given a hyperbola, return its foci -/
def foci (h : Hyperbola) : (Point × Point) :=
  let c := h.a * h.e
  (Point.mk (-c) 0, Point.mk c 0)

/-- Given two points, check if they are perpendicular with respect to the origin -/
def are_perpendicular (p1 p2 : Point) : Prop :=
  p1.x * p2.x + p1.y * p2.y = 0

/-- Main theorem -/
theorem hyperbola_equation (h : Hyperbola) (p : Point) :
  let (f1, f2) := foci h
  (is_on_hyperbola h p) ∧
  (p.x = -3 ∧ p.y = -4) ∧
  (are_perpendicular (Point.mk (p.x - f1.x) (p.y - f1.y)) (Point.mk (p.x - f2.x) (p.y - f2.y))) →
  h.a = 3 ∧ h.b = 4 :=
sorry

end hyperbola_equation_l2780_278048


namespace score_difference_l2780_278034

def blue_free_throws : ℕ := 18
def blue_two_pointers : ℕ := 25
def blue_three_pointers : ℕ := 6

def red_free_throws : ℕ := 15
def red_two_pointers : ℕ := 22
def red_three_pointers : ℕ := 5

def blue_score : ℕ := blue_free_throws + 2 * blue_two_pointers + 3 * blue_three_pointers
def red_score : ℕ := red_free_throws + 2 * red_two_pointers + 3 * red_three_pointers

theorem score_difference : blue_score - red_score = 12 := by
  sorry

end score_difference_l2780_278034


namespace parallelogram_angle_E_l2780_278000

structure Parallelogram :=
  (E F G H : Point)

def angle_FGH (p : Parallelogram) : ℝ := sorry
def angle_E (p : Parallelogram) : ℝ := sorry

theorem parallelogram_angle_E (p : Parallelogram) 
  (h : angle_FGH p = 70) : angle_E p = 110 := by sorry

end parallelogram_angle_E_l2780_278000


namespace negation_of_existence_negation_of_quadratic_inequality_l2780_278079

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x > 0, x^2 - 2*x + 1 < 0) ↔ (∀ x > 0, x^2 - 2*x + 1 ≥ 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l2780_278079


namespace pentagon_sides_solutions_l2780_278031

/-- A pentagon with side lengths satisfying the given conditions -/
structure PentagonSides where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  h_one_side : e = 30
  h_arithmetic : b = a + 2 ∧ c = b + 2 ∧ d = c + 2
  h_smallest : a ≤ 7
  h_sum : a + b + c + d + e > e

/-- The theorem stating that only three specific pentagons satisfy the conditions -/
theorem pentagon_sides_solutions :
  { sides : PentagonSides | 
    (sides.a = 5 ∧ sides.b = 7 ∧ sides.c = 9 ∧ sides.d = 11 ∧ sides.e = 30) ∨
    (sides.a = 6 ∧ sides.b = 8 ∧ sides.c = 10 ∧ sides.d = 12 ∧ sides.e = 30) ∨
    (sides.a = 7 ∧ sides.b = 9 ∧ sides.c = 11 ∧ sides.d = 13 ∧ sides.e = 30) } =
  { sides : PentagonSides | True } :=
sorry

end pentagon_sides_solutions_l2780_278031


namespace train_length_l2780_278084

/-- Calculates the length of a train given its speed and time to pass a fixed point. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 → time = 14 → speed * time * (1000 / 3600) = 280 :=
by
  sorry

#check train_length

end train_length_l2780_278084


namespace largest_integer_square_sum_l2780_278077

theorem largest_integer_square_sum : ∃ (x y z : ℤ),
  6^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10 ∧
  ∀ (n : ℤ), n > 6 → ¬∃ (x y z : ℤ),
    n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10 :=
by sorry

end largest_integer_square_sum_l2780_278077


namespace arithmetic_mean_after_removal_l2780_278087

theorem arithmetic_mean_after_removal (arr : Finset ℤ) (sum : ℤ) : 
  Finset.card arr = 40 →
  sum = Finset.sum arr id →
  sum / 40 = 45 →
  60 ∈ arr →
  70 ∈ arr →
  ((sum - 60 - 70) : ℚ) / 38 = 43.95 :=
by sorry

end arithmetic_mean_after_removal_l2780_278087


namespace first_four_super_nice_sum_l2780_278071

def is_super_nice (n : ℕ) : Prop :=
  n > 1 ∧
  (∃ (divisors : Finset ℕ),
    divisors = {d : ℕ | d ∣ n ∧ d ≠ 1 ∧ d ≠ n} ∧
    n = (Finset.prod divisors id) ∧
    n = (Finset.sum divisors id))

theorem first_four_super_nice_sum :
  ∃ (a b c d : ℕ),
    a < b ∧ b < c ∧ c < d ∧
    is_super_nice a ∧
    is_super_nice b ∧
    is_super_nice c ∧
    is_super_nice d ∧
    a + b + c + d = 45 :=
  sorry

end first_four_super_nice_sum_l2780_278071


namespace bug_probability_after_8_steps_l2780_278061

/-- Probability of being at vertex A after n steps -/
def P (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (1 - P (n - 1)) / 3

/-- The probability of being at vertex A after 8 steps in a regular tetrahedron -/
theorem bug_probability_after_8_steps :
  P 8 = 547 / 2187 := by sorry

end bug_probability_after_8_steps_l2780_278061


namespace sum_of_non_common_roots_is_zero_l2780_278073

/-- Given two quadratic equations with one common root, prove that the sum of the non-common roots is 0 -/
theorem sum_of_non_common_roots_is_zero (m : ℝ) :
  (∃ x : ℝ, x^2 + (m + 1) * x - 3 = 0 ∧ x^2 - 4 * x - m = 0) →
  (∃ α β γ : ℝ, 
    (α^2 + (m + 1) * α - 3 = 0 ∧ β^2 + (m + 1) * β - 3 = 0 ∧ α ≠ β) ∧
    (α^2 - 4 * α - m = 0 ∧ γ^2 - 4 * γ - m = 0 ∧ α ≠ γ) ∧
    β + γ = 0) :=
by sorry

end sum_of_non_common_roots_is_zero_l2780_278073


namespace walnut_trees_planted_l2780_278094

/-- The number of walnut trees planted in a park --/
theorem walnut_trees_planted (initial_trees final_trees : ℕ) :
  initial_trees < final_trees →
  final_trees - initial_trees = final_trees - initial_trees :=
by sorry

end walnut_trees_planted_l2780_278094


namespace geometric_sequence_ratio_l2780_278010

/-- Given a geometric sequence {a_n} with sum S_n = b(-2)^(n-1) - a, prove that a/b = -1/2 -/
theorem geometric_sequence_ratio (b a : ℝ) (S : ℕ → ℝ) (a_n : ℕ → ℝ) :
  (∀ n : ℕ, S n = b * (-2)^(n - 1) - a) →
  (∀ n : ℕ, a_n n = S n - S (n - 1)) →
  (a_n 1 = b - a) →
  a / b = -1 / 2 := by
sorry

end geometric_sequence_ratio_l2780_278010


namespace vectors_perpendicular_l2780_278017

def a : ℝ × ℝ := (-5, 6)
def b : ℝ × ℝ := (6, 5)

theorem vectors_perpendicular : a.1 * b.1 + a.2 * b.2 = 0 := by
  sorry

end vectors_perpendicular_l2780_278017


namespace fraction_meaningful_implies_x_not_one_l2780_278097

-- Define a function that represents the meaningfulness of the fraction 1/(x-1)
def is_meaningful (x : ℝ) : Prop := x ≠ 1

-- Theorem statement
theorem fraction_meaningful_implies_x_not_one :
  ∀ x : ℝ, is_meaningful x → x ≠ 1 :=
by
  sorry

end fraction_meaningful_implies_x_not_one_l2780_278097


namespace cos_2alpha_value_l2780_278049

theorem cos_2alpha_value (α : Real) (h : Real.tan (α + π/4) = 2) : Real.cos (2*α) = 4/5 := by
  sorry

end cos_2alpha_value_l2780_278049


namespace final_clothing_count_l2780_278043

/-- Calculate the remaining clothes after donations and purchases -/
def remaining_clothes (initial : ℕ) : ℕ :=
  let after_orphanages := initial - (initial / 10 + 3 * (initial / 10))
  let after_shelter := after_orphanages - (after_orphanages / 5)
  let after_purchase := after_shelter + (after_shelter / 5)
  after_purchase - (after_purchase / 8)

/-- Theorem stating the final number of clothing pieces -/
theorem final_clothing_count :
  remaining_clothes 500 = 252 := by
  sorry

end final_clothing_count_l2780_278043


namespace unique_lottery_ticket_l2780_278021

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n ≤ 99999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem unique_lottery_ticket (ticket : ℕ) (neighbor_age : ℕ) :
  is_five_digit ticket →
  digit_sum ticket = neighbor_age →
  (∀ m : ℕ, is_five_digit m → digit_sum m = neighbor_age → m = ticket) →
  ticket = 99999 :=
by sorry

end unique_lottery_ticket_l2780_278021


namespace interest_rate_is_one_percent_l2780_278051

/-- Calculate the interest rate given principal, time, and total simple interest -/
def calculate_interest_rate (principal : ℚ) (time : ℚ) (total_interest : ℚ) : ℚ :=
  (total_interest * 100) / (principal * time)

/-- Theorem stating that given the specific values, the interest rate is 1% -/
theorem interest_rate_is_one_percent :
  let principal : ℚ := 133875
  let time : ℚ := 3
  let total_interest : ℚ := 4016.25
  calculate_interest_rate principal time total_interest = 1 := by
  sorry

end interest_rate_is_one_percent_l2780_278051


namespace inequality_solution_set_l2780_278082

theorem inequality_solution_set (a b c : ℝ) (h1 : a > c) (h2 : b + c > 0) :
  {x : ℝ | (x - c) * (x + b) / (x - a) > 0} = {x : ℝ | -b < x ∧ x < c ∨ x > a} := by
  sorry

end inequality_solution_set_l2780_278082


namespace westeros_max_cursed_roads_l2780_278050

/-- A graph representing the Westeros Empire -/
structure WesterosGraph where
  /-- The number of cities (vertices) in the graph -/
  num_cities : Nat
  /-- The number of roads (edges) in the graph -/
  num_roads : Nat
  /-- The graph is initially connected -/
  is_connected : Bool
  /-- The number of kingdoms formed after cursing some roads -/
  num_kingdoms : Nat

/-- The maximum number of roads that can be cursed -/
def max_cursed_roads (g : WesterosGraph) : Nat :=
  g.num_roads - (g.num_cities - g.num_kingdoms)

/-- Theorem stating the maximum number of roads that can be cursed -/
theorem westeros_max_cursed_roads (g : WesterosGraph) 
  (h1 : g.num_cities = 1000)
  (h2 : g.num_roads = 2017)
  (h3 : g.is_connected = true)
  (h4 : g.num_kingdoms = 7) :
  max_cursed_roads g = 1024 := by
  sorry

#eval max_cursed_roads { num_cities := 1000, num_roads := 2017, is_connected := true, num_kingdoms := 7 }

end westeros_max_cursed_roads_l2780_278050


namespace modular_congruence_solution_l2780_278068

theorem modular_congruence_solution : ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end modular_congruence_solution_l2780_278068


namespace three_gorges_dam_capacity_l2780_278042

theorem three_gorges_dam_capacity :
  (16780000 : ℝ) = 1.678 * (10 ^ 7) := by
  sorry

end three_gorges_dam_capacity_l2780_278042


namespace profit_at_selling_price_185_l2780_278085

/-- Represents the daily sales volume as a function of price reduction --/
def sales_volume (x : ℝ) : ℝ := 4 * x + 100

/-- Represents the selling price as a function of price reduction --/
def selling_price (x : ℝ) : ℝ := 200 - x

/-- Represents the daily profit as a function of price reduction --/
def daily_profit (x : ℝ) : ℝ := (selling_price x - 100) * sales_volume x

theorem profit_at_selling_price_185 :
  ∃ x : ℝ, 
    daily_profit x = 13600 ∧ 
    selling_price x = 185 ∧ 
    selling_price x ≥ 150 := by sorry

end profit_at_selling_price_185_l2780_278085


namespace sum_of_cube_difference_l2780_278095

theorem sum_of_cube_difference (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 150 →
  a + b + c = 6 := by
  sorry

end sum_of_cube_difference_l2780_278095


namespace convex_polygon_diagonals_l2780_278018

-- Define a convex polygon type
structure ConvexPolygon where
  sides : ℕ
  is_convex : Bool
  interior_angle : ℝ
  all_angles_equal : Bool

-- Theorem statement
theorem convex_polygon_diagonals 
  (p : ConvexPolygon) 
  (h1 : p.is_convex = true) 
  (h2 : p.interior_angle = 150) 
  (h3 : p.all_angles_equal = true) : 
  (p.sides * (p.sides - 3)) / 2 = 54 := by
sorry

end convex_polygon_diagonals_l2780_278018


namespace factorial_sum_l2780_278002

theorem factorial_sum : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 6 * Nat.factorial 5 = 36600 := by
  sorry

end factorial_sum_l2780_278002


namespace equation_solution_l2780_278062

theorem equation_solution : 
  ∃ (x₁ x₂ : ℚ), x₁ = 1/3 ∧ x₂ = 1/2 ∧ 
  (∀ x : ℚ, 6*x^2 - 3*x - 1 = 2*x - 2 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end equation_solution_l2780_278062


namespace cone_base_radius_l2780_278055

-- Define the surface area of the cone
def surface_area (a : ℝ) : ℝ := a

-- Define the property that the lateral surface unfolds into a semicircle
def lateral_surface_is_semicircle (r l : ℝ) : Prop := 2 * Real.pi * r = Real.pi * l

-- Theorem statement
theorem cone_base_radius (a : ℝ) (h : a > 0) :
  ∃ (r : ℝ), r > 0 ∧ 
    (∃ (l : ℝ), l > 0 ∧ 
      lateral_surface_is_semicircle r l ∧ 
      surface_area a = Real.pi * r^2 + Real.pi * r * l) ∧
    r = Real.sqrt (a / (3 * Real.pi)) :=
sorry

end cone_base_radius_l2780_278055


namespace exclusive_math_enrollment_is_29_l2780_278096

/-- Represents the number of students in each class or combination of classes --/
structure ClassEnrollment where
  total : ℕ
  math : ℕ
  foreign : ℕ
  musicOnly : ℕ

/-- Calculates the number of students enrolled exclusively in math classes --/
def exclusiveMathEnrollment (e : ClassEnrollment) : ℕ :=
  e.math - (e.math + e.foreign - (e.total - e.musicOnly))

/-- Theorem stating that given the conditions, 29 students are enrolled exclusively in math --/
theorem exclusive_math_enrollment_is_29 (e : ClassEnrollment)
  (h1 : e.total = 120)
  (h2 : e.math = 82)
  (h3 : e.foreign = 71)
  (h4 : e.musicOnly = 20) :
  exclusiveMathEnrollment e = 29 := by
  sorry

#eval exclusiveMathEnrollment ⟨120, 82, 71, 20⟩

end exclusive_math_enrollment_is_29_l2780_278096


namespace sum_of_square_roots_l2780_278019

theorem sum_of_square_roots : 
  Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + 
  Real.sqrt (1+3+5+7+9) + Real.sqrt (1+3+5+7+9+11) = 21 := by
  sorry

end sum_of_square_roots_l2780_278019


namespace stating_not_always_two_triangles_form_rectangle_l2780_278013

/-- Represents a non-isosceles right triangle -/
structure NonIsoscelesRightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  leg1_ne_leg2 : leg1 ≠ leg2
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

/-- Represents a rectangle constructed from non-isosceles right triangles -/
structure RectangleFromTriangles where
  width : ℝ
  height : ℝ
  triangle : NonIsoscelesRightTriangle
  num_triangles : ℕ
  area_equality : width * height = num_triangles * (triangle.leg1 * triangle.leg2 / 2)

/-- 
Theorem stating that it's not always necessary for any two identical non-isosceles 
right triangles to form a rectangle when a larger rectangle is constructed from 
these triangles without gaps or overlaps
-/
theorem not_always_two_triangles_form_rectangle 
  (r : RectangleFromTriangles) : 
  ¬ ∀ (t1 t2 : NonIsoscelesRightTriangle), 
    t1 = r.triangle → t2 = r.triangle → 
    ∃ (w h : ℝ), w * h = t1.leg1 * t1.leg2 + t2.leg1 * t2.leg2 := by
  sorry

end stating_not_always_two_triangles_form_rectangle_l2780_278013


namespace max_car_distance_l2780_278028

/-- Represents the maximum distance a car can travel with one tire swap -/
def max_distance (front_tire_life : ℕ) (rear_tire_life : ℕ) : ℕ :=
  front_tire_life + min front_tire_life rear_tire_life

/-- Theorem stating the maximum distance a car can travel with given tire lifespans -/
theorem max_car_distance :
  let front_tire_life : ℕ := 24000
  let rear_tire_life : ℕ := 36000
  max_distance front_tire_life rear_tire_life = 28800 := by
  sorry

#eval max_distance 24000 36000

end max_car_distance_l2780_278028


namespace perpendicular_planes_from_perpendicular_lines_l2780_278060

-- Define the types for our geometric objects
variable (Point : Type) [NormedAddCommGroup Point] [InnerProductSpace ℝ Point]
variable (Line : Type) (Plane : Type)

-- Define the relationships between geometric objects
variable (belongs_to : Point → Line → Prop)
variable (subset_of : Line → Plane → Prop)
variable (intersect_along : Plane → Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define our specific objects
variable (α β : Plane) (l a b : Line)

-- State the theorem
theorem perpendicular_planes_from_perpendicular_lines 
  (h1 : intersect_along α β l)
  (h2 : subset_of a α)
  (h3 : subset_of b β)
  (h4 : perpendicular a l)
  (h5 : perpendicular b l) :
  ¬(plane_perpendicular α β) :=
sorry

end perpendicular_planes_from_perpendicular_lines_l2780_278060


namespace velvet_area_for_given_box_l2780_278016

/-- The total area of velvet needed to line the inside of a box with given dimensions -/
def total_velvet_area (long_side_length long_side_width short_side_length short_side_width top_bottom_area : ℕ) : ℕ :=
  2 * (long_side_length * long_side_width) +
  2 * (short_side_length * short_side_width) +
  2 * top_bottom_area

/-- Theorem stating that the total area of velvet needed for the given box dimensions is 236 square inches -/
theorem velvet_area_for_given_box : total_velvet_area 8 6 5 6 40 = 236 := by
  sorry

end velvet_area_for_given_box_l2780_278016


namespace equal_area_rectangles_length_l2780_278083

/-- Given two rectangles of equal area, where one rectangle has dimensions 2 inches by 60 inches,
    and the other has a width of 24 inches, prove that the length of the second rectangle is 5 inches. -/
theorem equal_area_rectangles_length (l : ℝ) :
  (2 : ℝ) * 60 = l * 24 → l = 5 := by
  sorry

end equal_area_rectangles_length_l2780_278083


namespace special_decimal_value_l2780_278088

/-- A two-digit decimal number with specific digit placements -/
def special_decimal (n : ℚ) : Prop :=
  ∃ (w : ℕ), w < 100 ∧ n = w + 0.55

/-- The theorem stating that the special decimal number is equal to 50.05 -/
theorem special_decimal_value :
  ∀ n : ℚ, special_decimal n → n = 50.05 := by
sorry

end special_decimal_value_l2780_278088


namespace soccer_team_lineup_count_l2780_278070

def total_players : ℕ := 18
def goalie_needed : ℕ := 1
def field_players_needed : ℕ := 10

theorem soccer_team_lineup_count :
  (total_players.choose goalie_needed) * ((total_players - goalie_needed).choose field_players_needed) = 349864 := by
  sorry

end soccer_team_lineup_count_l2780_278070


namespace equation_solution_l2780_278057

theorem equation_solution : ∃! y : ℚ, 2 * (y - 3) - 6 * (2 * y - 1) = -3 * (2 - 5 * y) ∧ y = 6 / 25 := by
  sorry

end equation_solution_l2780_278057


namespace find_m_l2780_278006

theorem find_m : ∃ m : ℝ, 
  (∀ x : ℝ, mx + 3 = x ↔ 5 - 2*x = 1) → m = -1/2 := by sorry

end find_m_l2780_278006


namespace unique_solution_condition_l2780_278003

theorem unique_solution_condition (b : ℝ) : 
  (∃! x : ℝ, x^3 - b*x^2 - 4*b*x + b^2 - 4 = 0) ↔ b < 1 := by
  sorry

end unique_solution_condition_l2780_278003


namespace polynomial_factorization_l2780_278078

theorem polynomial_factorization (x : ℝ) : 
  9 * (x + 6) * (x + 12) * (x + 5) * (x + 15) - 8 * x^2 = 
  (3 * x^2 + 52 * x + 210) * (3 * x^2 + 56 * x + 222) := by
  sorry

end polynomial_factorization_l2780_278078


namespace percentage_problem_l2780_278038

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 600 = (50 / 100) * 960 → P = 80 := by
  sorry

end percentage_problem_l2780_278038


namespace unanswered_test_theorem_l2780_278024

/-- The number of ways to complete an unanswered multiple-choice test -/
def unanswered_test_completions (num_questions : ℕ) (num_choices : ℕ) : ℕ := 1

/-- Theorem: For a test with 4 questions and 5 choices per question, 
    there is only one way to complete it if all questions are unanswered -/
theorem unanswered_test_theorem : 
  unanswered_test_completions 4 5 = 1 := by
  sorry

end unanswered_test_theorem_l2780_278024


namespace polygon_sides_from_angle_sum_l2780_278020

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 720 → (n - 2) * 180 = angle_sum → n = 6 := by
  sorry

end polygon_sides_from_angle_sum_l2780_278020


namespace bridgets_skittles_proof_l2780_278098

/-- The number of Skittles Henry has -/
def henrys_skittles : ℕ := 4

/-- The total number of Skittles after Henry gives his to Bridget -/
def total_skittles : ℕ := 8

/-- Bridget's initial number of Skittles -/
def bridgets_initial_skittles : ℕ := total_skittles - henrys_skittles

theorem bridgets_skittles_proof :
  bridgets_initial_skittles = 4 :=
by sorry

end bridgets_skittles_proof_l2780_278098


namespace quadratic_equation_solution_l2780_278005

theorem quadratic_equation_solution (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (x^2 + 10*x = 45 ∧ x = Real.sqrt a - b) → a + b = 75 :=
by
  sorry

end quadratic_equation_solution_l2780_278005


namespace inequality_solution_set_l2780_278076

theorem inequality_solution_set (x : ℝ) :
  (x^2 - |x| - 2 < 0) ↔ (-2 < x ∧ x < 2) :=
by sorry

end inequality_solution_set_l2780_278076


namespace inverse_f_27_equals_3_l2780_278007

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem inverse_f_27_equals_3 :
  ∀ f_inv : ℝ → ℝ, 
  (∀ x : ℝ, f_inv (f x) = x) ∧ (∀ y : ℝ, f (f_inv y) = y) → 
  f_inv 27 = 3 := by
sorry

end inverse_f_27_equals_3_l2780_278007


namespace simplify_expression_l2780_278052

theorem simplify_expression :
  ∃ (C : ℝ), C = 2^(1 + Real.sqrt 2) ∧
  (Real.sqrt 3 - 1)^(1 - Real.sqrt 2) / (Real.sqrt 3 + 1)^(1 + Real.sqrt 2) = (4 - 2 * Real.sqrt 3) / C :=
by sorry

end simplify_expression_l2780_278052


namespace system_solution_unique_l2780_278036

theorem system_solution_unique :
  ∃! (x y : ℝ), 3 * x - 2 * y = 3 ∧ x + 4 * y = 1 :=
by
  -- The proof goes here
  sorry

end system_solution_unique_l2780_278036


namespace percentage_of_360_equals_126_l2780_278053

theorem percentage_of_360_equals_126 : 
  (126 : ℝ) / 360 * 100 = 35 := by sorry

end percentage_of_360_equals_126_l2780_278053


namespace line_perp_plane_and_line_implies_parallel_l2780_278044

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpToPlane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem line_perp_plane_and_line_implies_parallel
  (l m : Line) (α : Plane)
  (h1 : l ≠ m)
  (h2 : perpToPlane l α)
  (h3 : perp l m) :
  parallel m α :=
sorry

end line_perp_plane_and_line_implies_parallel_l2780_278044


namespace x_value_proof_l2780_278075

theorem x_value_proof (x : ℚ) 
  (eq1 : 8 * x^2 + 7 * x - 1 = 0) 
  (eq2 : 24 * x^2 + 53 * x - 7 = 0) : 
  x = 1/8 := by sorry

end x_value_proof_l2780_278075


namespace max_cables_cut_theorem_l2780_278012

/-- Represents a computer network -/
structure ComputerNetwork where
  num_computers : ℕ
  num_cables : ℕ
  num_clusters : ℕ

/-- Calculates the maximum number of cables that can be cut -/
def max_cables_cut (network : ComputerNetwork) : ℕ :=
  network.num_cables - (network.num_clusters - 1)

/-- Theorem stating the maximum number of cables that can be cut -/
theorem max_cables_cut_theorem (network : ComputerNetwork) 
  (h1 : network.num_computers = 200)
  (h2 : network.num_cables = 345)
  (h3 : network.num_clusters = 8) :
  max_cables_cut network = 153 := by
  sorry

#eval max_cables_cut ⟨200, 345, 8⟩

end max_cables_cut_theorem_l2780_278012


namespace allison_bought_28_items_l2780_278056

/-- The number of craft supply items Allison bought -/
def allison_total (marie_glue : ℕ) (marie_paper : ℕ) (glue_diff : ℕ) (paper_ratio : ℕ) : ℕ :=
  (marie_glue + glue_diff) + (marie_paper / paper_ratio)

/-- Theorem stating the total number of craft supply items Allison bought -/
theorem allison_bought_28_items : allison_total 15 30 8 6 = 28 := by
  sorry

end allison_bought_28_items_l2780_278056


namespace expression_simplification_and_evaluation_l2780_278009

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ x ≠ 3 →
  (1 - 2 / (x - 1)) / ((x^2 - 6*x + 9) / (x^2 - 1)) = (x + 1) / (x - 3) ∧
  (2 + 1) / (2 - 3) = -3 := by
sorry

end expression_simplification_and_evaluation_l2780_278009


namespace average_increase_is_4_l2780_278064

/-- Represents the cricketer's score data -/
structure CricketerScore where
  runs_19th_inning : ℕ
  average_after_19 : ℚ

/-- Calculates the increase in average score -/
def average_increase (score : CricketerScore) : ℚ :=
  let total_runs := score.average_after_19 * 19
  let runs_before_19th := total_runs - score.runs_19th_inning
  let average_before_19th := runs_before_19th / 18
  score.average_after_19 - average_before_19th

/-- Theorem stating the increase in average score -/
theorem average_increase_is_4 (score : CricketerScore) 
  (h1 : score.runs_19th_inning = 96)
  (h2 : score.average_after_19 = 24) :
  average_increase score = 4 := by
  sorry

end average_increase_is_4_l2780_278064


namespace cos_18_degrees_l2780_278063

theorem cos_18_degrees : Real.cos (18 * π / 180) = (1 + Real.sqrt 5) / 4 := by
  sorry

end cos_18_degrees_l2780_278063


namespace complex_multiplication_l2780_278027

theorem complex_multiplication (i : ℂ) :
  i^2 = -1 →
  (6 - 7*i) * (3 + 6*i) = 60 + 15*i :=
by sorry

end complex_multiplication_l2780_278027


namespace h2o_mass_formed_l2780_278065

-- Define the chemical reaction
structure Reaction where
  hcl : ℝ
  caco3 : ℝ
  h2o : ℝ

-- Define the molar masses
def molar_mass_h : ℝ := 1.008
def molar_mass_o : ℝ := 15.999

-- Define the reaction stoichiometry
def reaction_stoichiometry (r : Reaction) : Prop :=
  r.hcl = 2 * r.caco3 ∧ r.h2o = r.caco3

-- Calculate the molar mass of H2O
def molar_mass_h2o : ℝ := 2 * molar_mass_h + molar_mass_o

-- Main theorem
theorem h2o_mass_formed (r : Reaction) : 
  reaction_stoichiometry r → r.hcl = 2 → r.caco3 = 1 → r.h2o * molar_mass_h2o = 18.015 :=
sorry

end h2o_mass_formed_l2780_278065


namespace xiaoding_distance_l2780_278033

/-- Represents the distance to school for each student in meters -/
structure SchoolDistances where
  xiaoding : ℕ
  xiaowang : ℕ
  xiaocheng : ℕ
  xiaozhang : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (d : SchoolDistances) : Prop :=
  d.xiaowang + d.xiaoding + d.xiaocheng + d.xiaozhang = 705 ∧
  d.xiaowang = 4 * d.xiaoding ∧
  d.xiaocheng = d.xiaowang / 2 + 20 ∧
  d.xiaozhang = 2 * d.xiaocheng - 15

/-- The theorem to be proved -/
theorem xiaoding_distance (d : SchoolDistances) :
  satisfiesConditions d → d.xiaoding = 60 := by
  sorry


end xiaoding_distance_l2780_278033


namespace shopkeeper_sales_l2780_278041

/-- The number of articles sold by a shopkeeper -/
def articles_sold (cost_price : ℝ) : ℕ :=
  72

/-- The profit percentage made by the shopkeeper -/
def profit_percentage : ℝ :=
  20

/-- The number of articles whose cost price equals the selling price -/
def equivalent_articles : ℕ :=
  60

theorem shopkeeper_sales :
  ∀ (cost_price : ℝ),
  cost_price > 0 →
  (articles_sold cost_price : ℝ) * cost_price =
    equivalent_articles * cost_price * (1 + profit_percentage / 100) :=
by sorry

end shopkeeper_sales_l2780_278041
