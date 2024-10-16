import Mathlib

namespace NUMINAMATH_CALUDE_series_sum_l1800_180066

/-- The series defined by the problem -/
def series (n : ℕ) : ℚ :=
  if n % 3 = 1 then 1 / (2^n)
  else if n % 3 = 0 then -1 / (2^n)
  else -1 / (2^n)

/-- The sum of the series -/
noncomputable def S : ℚ := ∑' n, series n

/-- The theorem to be proved -/
theorem series_sum : S / (10 * 81) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l1800_180066


namespace NUMINAMATH_CALUDE_range_of_m_unbounded_below_m_characterization_of_m_range_l1800_180008

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2*m - 1 }

theorem range_of_m (m : ℝ) : (A ∪ B m = A) → m ≤ 5/2 :=
by sorry

theorem unbounded_below_m : ∀ (k : ℝ), ∃ (m : ℝ), m < k ∧ (A ∪ B m = A) :=
by sorry

theorem characterization_of_m_range : 
  ∀ (m : ℝ), (A ∪ B m = A) ↔ m ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_unbounded_below_m_characterization_of_m_range_l1800_180008


namespace NUMINAMATH_CALUDE_correct_expansion_l1800_180033

theorem correct_expansion (a b : ℝ) : (a - b) * (-a - b) = -a^2 + b^2 := by
  -- Definitions based on the given conditions (equations A, B, and C)
  have h1 : (a + b) * (a - b) = a^2 - b^2 := by sorry
  have h2 : (a + b) * (-a - b) = -(a + b)^2 := by sorry
  have h3 : (a - b) * (-a + b) = -(a - b)^2 := by sorry

  -- Proof of the correct expansion
  sorry

end NUMINAMATH_CALUDE_correct_expansion_l1800_180033


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l1800_180025

/-- Proves that given a mixture with 10% water content, if 5 liters of water are added
    to make the new mixture contain 20% water, then the initial volume of the mixture was 40 liters. -/
theorem initial_mixture_volume
  (initial_water_percentage : Real)
  (added_water : Real)
  (final_water_percentage : Real)
  (h1 : initial_water_percentage = 0.10)
  (h2 : added_water = 5)
  (h3 : final_water_percentage = 0.20)
  : ∃ (initial_volume : Real),
    initial_volume * initial_water_percentage + added_water
      = (initial_volume + added_water) * final_water_percentage
    ∧ initial_volume = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l1800_180025


namespace NUMINAMATH_CALUDE_infinitely_many_inequality_holds_l1800_180096

theorem infinitely_many_inequality_holds (a : ℕ → ℝ) (h : ∀ n, a n > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, 1 + a n > a (n - 1) * (2 : ℝ) ^ (1 / n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_inequality_holds_l1800_180096


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1800_180048

theorem sufficient_but_not_necessary (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1800_180048


namespace NUMINAMATH_CALUDE_factorization_x4_plus_81_l1800_180077

theorem factorization_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6*x + 9) * (x^2 - 6*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_plus_81_l1800_180077


namespace NUMINAMATH_CALUDE_find_y_l1800_180010

-- Define the binary operation ⊕
def binary_op (a b c d : ℤ) : ℤ × ℤ := (a + d, b - c)

-- Theorem statement
theorem find_y : ∀ x y : ℤ, 
  binary_op 2 5 1 1 = binary_op x y 2 0 → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l1800_180010


namespace NUMINAMATH_CALUDE_calculator_sequence_101_l1800_180064

def calculator_sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 7
  | n + 1 => 1 / (1 - calculator_sequence n)

theorem calculator_sequence_101 : calculator_sequence 101 = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_calculator_sequence_101_l1800_180064


namespace NUMINAMATH_CALUDE_prism_with_27_edges_has_11_faces_l1800_180056

/-- A prism is a polyhedron with two congruent and parallel faces (bases) connected by lateral faces. -/
structure Prism where
  edges : ℕ
  lateral_faces : ℕ

/-- The number of edges in a prism is three times the number of lateral faces. -/
axiom prism_edge_count (p : Prism) : p.edges = 3 * p.lateral_faces

/-- The total number of faces in a prism is the number of lateral faces plus two (for the bases). -/
def total_faces (p : Prism) : ℕ := p.lateral_faces + 2

/-- Theorem: A prism with 27 edges has 11 faces. -/
theorem prism_with_27_edges_has_11_faces (p : Prism) (h : p.edges = 27) : total_faces p = 11 := by
  sorry


end NUMINAMATH_CALUDE_prism_with_27_edges_has_11_faces_l1800_180056


namespace NUMINAMATH_CALUDE_dress_designs_count_l1800_180045

/-- The number of fabric colors available -/
def num_colors : ℕ := 5

/-- The number of fabric materials available for each color -/
def num_materials : ℕ := 2

/-- The number of patterns available -/
def num_patterns : ℕ := 4

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_materials * num_patterns

theorem dress_designs_count : total_designs = 40 := by
  sorry

end NUMINAMATH_CALUDE_dress_designs_count_l1800_180045


namespace NUMINAMATH_CALUDE_sum_remainder_thirteen_l1800_180023

theorem sum_remainder_thirteen : ∃ k : ℕ, (5000 + 5001 + 5002 + 5003 + 5004 + 5005 + 5006) = 13 * k + 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_thirteen_l1800_180023


namespace NUMINAMATH_CALUDE_fraction_problem_l1800_180086

theorem fraction_problem (n : ℝ) (F : ℝ) (h1 : n = 70.58823529411765) (h2 : 0.85 * F * n = 36) : F = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1800_180086


namespace NUMINAMATH_CALUDE_sqrt_square_minus_sqrt_nine_plus_cube_root_eight_equals_one_l1800_180063

theorem sqrt_square_minus_sqrt_nine_plus_cube_root_eight_equals_one :
  (Real.sqrt 2)^2 - Real.sqrt 9 + (8 : ℝ)^(1/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_minus_sqrt_nine_plus_cube_root_eight_equals_one_l1800_180063


namespace NUMINAMATH_CALUDE_log_equation_implies_sum_of_cubes_l1800_180082

theorem log_equation_implies_sum_of_cubes (x y : ℝ) 
  (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 2)^3 + (Real.log y / Real.log 3)^3 = 
       3 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^3 + y^3 = 307 := by
sorry

end NUMINAMATH_CALUDE_log_equation_implies_sum_of_cubes_l1800_180082


namespace NUMINAMATH_CALUDE_prob_two_tails_three_coins_l1800_180091

/-- A fair coin is a coin with equal probability of heads and tails. -/
def FairCoin : Type := Unit

/-- The outcome of tossing a coin. -/
inductive CoinOutcome
| Heads
| Tails

/-- The outcome of tossing multiple coins. -/
def MultiCoinOutcome (n : ℕ) := Fin n → CoinOutcome

/-- The number of coins being tossed. -/
def numCoins : ℕ := 3

/-- The total number of possible outcomes when tossing n fair coins. -/
def totalOutcomes (n : ℕ) : ℕ := 2^n

/-- The number of ways to get exactly k tails when tossing n coins. -/
def waysToGetKTails (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes. -/
def probability (favorableOutcomes totalOutcomes : ℕ) : ℚ := favorableOutcomes / totalOutcomes

/-- The main theorem: the probability of getting exactly 2 tails when tossing 3 fair coins is 3/8. -/
theorem prob_two_tails_three_coins : 
  probability (waysToGetKTails numCoins 2) (totalOutcomes numCoins) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_tails_three_coins_l1800_180091


namespace NUMINAMATH_CALUDE_probability_yellow_ball_l1800_180011

/-- Probability of choosing a yellow ball from a bag -/
theorem probability_yellow_ball (red yellow blue : ℕ) (h : red = 2 ∧ yellow = 5 ∧ blue = 4) :
  (yellow : ℚ) / (red + yellow + blue : ℚ) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_ball_l1800_180011


namespace NUMINAMATH_CALUDE_susan_pencil_purchase_l1800_180097

/-- The number of pencils Susan bought -/
def num_pencils : ℕ := 16

/-- The number of pens Susan bought -/
def num_pens : ℕ := 36 - num_pencils

/-- The cost of a pencil in cents -/
def pencil_cost : ℕ := 25

/-- The cost of a pen in cents -/
def pen_cost : ℕ := 80

/-- The total amount Susan spent in cents -/
def total_spent : ℕ := 2000

theorem susan_pencil_purchase :
  num_pencils + num_pens = 36 ∧
  pencil_cost * num_pencils + pen_cost * num_pens = total_spent :=
by sorry

#check susan_pencil_purchase

end NUMINAMATH_CALUDE_susan_pencil_purchase_l1800_180097


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1800_180020

/-- A geometric sequence {a_n} satisfying given conditions has the general term formula a_n = 1 / (2^(n-4)) -/
theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_sum1 : a 1 + a 3 = 10) 
  (h_sum2 : a 4 + a 6 = 5/4) : 
  ∀ n : ℕ, n ≥ 1 → a n = 1 / (2^(n-4)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1800_180020


namespace NUMINAMATH_CALUDE_race_equation_l1800_180028

theorem race_equation (x : ℝ) (h : x > 0) : 
  (1000 / x : ℝ) - (1000 / (1.25 * x)) = 30 :=
by sorry

end NUMINAMATH_CALUDE_race_equation_l1800_180028


namespace NUMINAMATH_CALUDE_sum_of_number_and_reverse_divisible_by_11_l1800_180035

theorem sum_of_number_and_reverse_divisible_by_11 (A B : ℕ) : 
  A < 10 → B < 10 → A ≠ B → 
  11 ∣ ((10 * A + B) + (10 * B + A)) := by
sorry

end NUMINAMATH_CALUDE_sum_of_number_and_reverse_divisible_by_11_l1800_180035


namespace NUMINAMATH_CALUDE_horner_method_v₃_l1800_180029

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

/-- The value of v₃ using Horner's method -/
def v₃ (x : ℝ) : ℝ := (((7*x+6)*x+5)*x+4)

/-- Theorem stating that v₃(3) = 262 -/
theorem horner_method_v₃ : v₃ 3 = 262 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_v₃_l1800_180029


namespace NUMINAMATH_CALUDE_choose_four_from_six_eq_fifteen_l1800_180049

/-- The number of ways to choose 4 items from a set of 6 items, where the order doesn't matter -/
def choose_four_from_six : ℕ := Nat.choose 6 4

/-- Theorem stating that choosing 4 items from a set of 6 items results in 15 combinations -/
theorem choose_four_from_six_eq_fifteen : choose_four_from_six = 15 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_six_eq_fifteen_l1800_180049


namespace NUMINAMATH_CALUDE_tina_katya_difference_l1800_180089

/-- The number of glasses of lemonade sold by each person -/
structure LemonadeSales where
  katya : ℕ
  ricky : ℕ
  tina : ℕ

/-- The conditions of the lemonade sales problem -/
def lemonade_problem (sales : LemonadeSales) : Prop :=
  sales.katya = 8 ∧
  sales.ricky = 9 ∧
  sales.tina = 2 * (sales.katya + sales.ricky)

/-- The theorem stating the difference between Tina's and Katya's sales -/
theorem tina_katya_difference (sales : LemonadeSales) 
  (h : lemonade_problem sales) : sales.tina - sales.katya = 26 := by
  sorry

end NUMINAMATH_CALUDE_tina_katya_difference_l1800_180089


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1800_180000

open Real

theorem function_inequality_implies_a_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ > a, log x₁ + 1/x₁ ≥ x₂ + 1/(x₂ - a)) → 
  a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1800_180000


namespace NUMINAMATH_CALUDE_lucas_cleaning_days_l1800_180099

/-- Calculates the number of days Lucas took to clean windows -/
def days_to_clean (floors : ℕ) (windows_per_floor : ℕ) (payment_per_window : ℕ) 
                  (deduction_per_period : ℕ) (days_per_period : ℕ) (final_payment : ℕ) : ℕ :=
  let total_windows := floors * windows_per_floor
  let total_payment := total_windows * payment_per_window
  let deduction := total_payment - final_payment
  let periods := deduction / deduction_per_period
  periods * days_per_period

/-- Theorem stating that Lucas took 6 days to clean all windows -/
theorem lucas_cleaning_days : 
  days_to_clean 3 3 2 1 3 16 = 6 := by sorry

end NUMINAMATH_CALUDE_lucas_cleaning_days_l1800_180099


namespace NUMINAMATH_CALUDE_square_side_length_l1800_180088

/-- A square with perimeter 24 meters has sides of length 6 meters. -/
theorem square_side_length (s : ℝ) (h1 : s > 0) (h2 : 4 * s = 24) : s = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1800_180088


namespace NUMINAMATH_CALUDE_pascal_interior_sum_l1800_180015

/-- Represents the sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_interior_sum (h : interior_sum 6 = 30) : interior_sum 8 = 126 := by
  sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_l1800_180015


namespace NUMINAMATH_CALUDE_solution_set_characterization_l1800_180031

/-- A function that is even and monotonically increasing on (0,+∞) -/
def f (a b : ℝ) (x : ℝ) : ℝ := (x - 2) * (a * x + b)

/-- The property of f being an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The property of f being monotonically increasing on (0,+∞) -/
def is_increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

/-- The solution set for f(2-x) > 0 -/
def solution_set (f : ℝ → ℝ) : Set ℝ := {x | f (2 - x) > 0}

/-- The theorem stating the solution set for f(2-x) > 0 -/
theorem solution_set_characterization {a b : ℝ} (h_even : is_even (f a b))
    (h_incr : is_increasing_on_positive (f a b)) :
    solution_set (f a b) = {x | x < 0 ∨ x > 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l1800_180031


namespace NUMINAMATH_CALUDE_binomial_variance_transform_l1800_180041

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Variance of a linear transformation of a random variable -/
def varianceLinearTransform (X : BinomialRV) (a b : ℝ) : ℝ := a^2 * variance X

/-- The main theorem to prove -/
theorem binomial_variance_transform (ξ : BinomialRV) 
    (h_n : ξ.n = 100) (h_p : ξ.p = 0.3) : 
    varianceLinearTransform ξ 3 (-5) = 189 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_transform_l1800_180041


namespace NUMINAMATH_CALUDE_f_unbounded_above_l1800_180014

/-- The function f(x, y) = 2x^2 + 4xy + 5y^2 + 8x - 6y -/
def f (x y : ℝ) : ℝ := 2 * x^2 + 4 * x * y + 5 * y^2 + 8 * x - 6 * y

/-- Theorem: The function f is unbounded above -/
theorem f_unbounded_above : ∀ M : ℝ, ∃ x y : ℝ, f x y > M := by
  sorry

end NUMINAMATH_CALUDE_f_unbounded_above_l1800_180014


namespace NUMINAMATH_CALUDE_infinite_points_in_circle_l1800_180068

open Set

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 < radius^2}

-- Define the condition for point P
def SatisfiesCondition (p : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  (p.1 - a.1)^2 + (p.2 - a.2)^2 + (p.1 - b.1)^2 + (p.2 - b.2)^2 ≤ 5

-- Theorem statement
theorem infinite_points_in_circle :
  let center := (0, 0)
  let radius := 2
  let a := (-2, 0)  -- One endpoint of the diameter
  let b := (2, 0)   -- Other endpoint of the diameter
  let valid_points := {p ∈ Circle center radius | SatisfiesCondition p a b}
  Infinite valid_points := by sorry

end NUMINAMATH_CALUDE_infinite_points_in_circle_l1800_180068


namespace NUMINAMATH_CALUDE_light_wash_count_l1800_180046

/-- Represents the number of gallons of water used per load for different wash types -/
structure WaterUsage where
  heavy : ℕ
  regular : ℕ
  light : ℕ

/-- Represents the number of loads for each wash type -/
structure Loads where
  heavy : ℕ
  regular : ℕ
  light : ℕ
  bleached : ℕ

def totalWaterUsage (usage : WaterUsage) (loads : Loads) : ℕ :=
  usage.heavy * loads.heavy +
  usage.regular * loads.regular +
  usage.light * (loads.light + loads.bleached)

theorem light_wash_count (usage : WaterUsage) (loads : Loads) :
  usage.heavy = 20 →
  usage.regular = 10 →
  usage.light = 2 →
  loads.heavy = 2 →
  loads.regular = 3 →
  loads.bleached = 2 →
  totalWaterUsage usage loads = 76 →
  loads.light = 1 :=
sorry

end NUMINAMATH_CALUDE_light_wash_count_l1800_180046


namespace NUMINAMATH_CALUDE_student_ticket_price_is_correct_l1800_180084

/-- Represents the ticket sales data for a single day -/
structure DaySales where
  senior : ℕ
  student : ℕ
  adult : ℕ
  total : ℚ

/-- Represents the price changes for a day -/
structure PriceChange where
  senior : ℚ
  student : ℚ
  adult : ℚ

/-- Finds the initial price of a student ticket given the sales data and price changes -/
def find_student_ticket_price (sales : Vector DaySales 5) (day4_change : PriceChange) (day5_change : PriceChange) : ℚ :=
  sorry

/-- The main theorem stating that the initial price of a student ticket is approximately $8.83 -/
theorem student_ticket_price_is_correct (sales : Vector DaySales 5) (day4_change : PriceChange) (day5_change : PriceChange) :
  let price := find_student_ticket_price sales day4_change day5_change
  abs (price - 8.83) < 0.01 := by sorry

end NUMINAMATH_CALUDE_student_ticket_price_is_correct_l1800_180084


namespace NUMINAMATH_CALUDE_power_of_64_two_thirds_l1800_180013

theorem power_of_64_two_thirds : (64 : ℝ) ^ (2/3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_64_two_thirds_l1800_180013


namespace NUMINAMATH_CALUDE_first_group_size_l1800_180006

/-- Represents the work rate of a group of people -/
structure WorkRate where
  people : ℕ
  work : ℕ
  days : ℕ

/-- The work rate of the first group -/
def first_group : WorkRate :=
  { people := 0,  -- We don't know this value yet
    work := 3,
    days := 3 }

/-- The work rate of the second group -/
def second_group : WorkRate :=
  { people := 9,
    work := 9,
    days := 3 }

/-- Calculates the daily work rate -/
def daily_rate (wr : WorkRate) : ℚ :=
  wr.work / wr.days

theorem first_group_size :
  first_group.people = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l1800_180006


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l1800_180094

/-- Proves that when the surface area of a sphere is increased to 4 times its original size,
    its volume is increased to 8 times the original. -/
theorem sphere_volume_increase (r : ℝ) (S V : ℝ → ℝ) 
    (hS : ∃ k : ℝ, ∀ x, S x = k * x^2)  -- Surface area is proportional to radius squared
    (hV : ∃ c : ℝ, ∀ x, V x = c * x^3)  -- Volume is proportional to radius cubed
    (hS_increase : S (2 * r) = 4 * S r) : -- Surface area is increased 4 times
  V (2 * r) = 8 * V r := by
sorry


end NUMINAMATH_CALUDE_sphere_volume_increase_l1800_180094


namespace NUMINAMATH_CALUDE_special_line_properties_l1800_180079

/-- A line passing through (5, 2) with y-intercept twice its x-intercept -/
def special_line (x y : ℝ) : Prop :=
  2 * x - 5 * y + 60 = 0

theorem special_line_properties :
  (special_line 5 2) ∧ 
  (∃ (b : ℝ), special_line 0 b ∧ special_line (b/2) 0 ∧ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_special_line_properties_l1800_180079


namespace NUMINAMATH_CALUDE_right_triangle_with_35_hypotenuse_l1800_180040

theorem right_triangle_with_35_hypotenuse (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 35 →           -- Hypotenuse length
  b = a + 1 →        -- Consecutive integer legs
  a + b = 51         -- Sum of leg lengths
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_with_35_hypotenuse_l1800_180040


namespace NUMINAMATH_CALUDE_division_result_l1800_180039

theorem division_result : (32 / 8 : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1800_180039


namespace NUMINAMATH_CALUDE_book_arrangement_l1800_180076

theorem book_arrangement (n : ℕ) (a b c : ℕ) (h1 : n = a + b + c) (h2 : a = 3) (h3 : b = 2) (h4 : c = 2) :
  (n.factorial) / (a.factorial * b.factorial) = 420 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_l1800_180076


namespace NUMINAMATH_CALUDE_clara_older_than_alice_l1800_180095

/-- Represents a person with their age and number of pens -/
structure Person where
  age : ℕ
  pens : ℕ

/-- The problem statement -/
theorem clara_older_than_alice (alice clara : Person)
  (h1 : alice.pens = 60)
  (h2 : clara.pens = 2 * alice.pens / 5)
  (h3 : alice.pens - clara.pens = alice.age - clara.age)
  (h4 : alice.age = 20)
  (h5 : clara.age + 5 = 61) :
  clara.age > alice.age := by
  sorry

#check clara_older_than_alice

end NUMINAMATH_CALUDE_clara_older_than_alice_l1800_180095


namespace NUMINAMATH_CALUDE_alpha_sin_beta_lt_beta_sin_alpha_l1800_180032

theorem alpha_sin_beta_lt_beta_sin_alpha (α β : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) : 
  α * Real.sin β < β * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_alpha_sin_beta_lt_beta_sin_alpha_l1800_180032


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_iff_no_intersection_l1800_180034

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Check if a line is parallel to a plane -/
def isParallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Check if a line intersects with another line -/
def intersects (l1 l2 : Line3D) : Prop :=
  sorry

/-- Get a line in a plane -/
def lineInPlane (p : Plane3D) : Line3D :=
  sorry

theorem line_parallel_to_plane_iff_no_intersection (l : Line3D) (p : Plane3D) :
  isParallel l p ↔ ∀ (l' : Line3D), lineInPlane p = l' → ¬ intersects l l' :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_iff_no_intersection_l1800_180034


namespace NUMINAMATH_CALUDE_lawn_area_proof_l1800_180090

theorem lawn_area_proof (total_posts : ℕ) (post_spacing : ℕ) 
  (h_total_posts : total_posts = 24)
  (h_post_spacing : post_spacing = 5)
  (h_longer_side_posts : ∀ s l : ℕ, s + l = total_posts / 2 → l + 1 = 3 * (s + 1)) :
  ∃ short_side long_side : ℕ,
    short_side * long_side = 500 ∧
    short_side + 1 + long_side + 1 = total_posts ∧
    (long_side + 1) * post_spacing = (short_side + 1) * post_spacing * 3 :=
by sorry

end NUMINAMATH_CALUDE_lawn_area_proof_l1800_180090


namespace NUMINAMATH_CALUDE_smallest_divisible_by_495_l1800_180001

/-- Represents a number in the sequence with n digits of 5 -/
def sequenceNumber (n : ℕ) : ℕ :=
  (10^n - 1) / 9 * 5

/-- The target number we want to prove is the smallest divisible by 495 -/
def targetNumber : ℕ := sequenceNumber 18

/-- Checks if a number is in the sequence -/
def isInSequence (k : ℕ) : Prop :=
  ∃ n : ℕ, sequenceNumber n = k

theorem smallest_divisible_by_495 :
  (targetNumber % 495 = 0) ∧
  (∀ k : ℕ, k < targetNumber → isInSequence k → k % 495 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_495_l1800_180001


namespace NUMINAMATH_CALUDE_v3_at_neg_one_l1800_180017

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 + 0.35*x + 1.8*x^2 - 3*x^3 + 6*x^4 - 5*x^5 + x^6

/-- v3 in Horner's method for f(x) -/
def v3 (x : ℝ) : ℝ := (((x - 5)*x + 6)*x - 3)

/-- Theorem: v3 equals -15 when x = -1 -/
theorem v3_at_neg_one : v3 (-1) = -15 := by sorry

end NUMINAMATH_CALUDE_v3_at_neg_one_l1800_180017


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l1800_180019

/-- The minimum distance from any point on the line x + y - 4 = 0 to the origin (0, 0) is 2√2 -/
theorem min_distance_to_origin : 
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  ∀ p ∈ line, Real.sqrt ((p.1 ^ 2) + (p.2 ^ 2)) ≥ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l1800_180019


namespace NUMINAMATH_CALUDE_principal_amount_proof_l1800_180047

/-- Proves that given the specified conditions, the principal amount is 4000 (rs.) --/
theorem principal_amount_proof (rate : ℚ) (amount : ℚ) (time : ℚ) : 
  rate = 8 / 100 → amount = 640 → time = 2 → 
  (amount * 100) / (rate * time) = 4000 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l1800_180047


namespace NUMINAMATH_CALUDE_members_playing_both_l1800_180065

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  badminton : ℕ
  tennis : ℕ
  neither : ℕ

/-- Calculate the number of members playing both badminton and tennis -/
def playBoth (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - (club.total - club.neither)

/-- Theorem stating the number of members playing both sports in the given scenario -/
theorem members_playing_both (club : SportsClub) 
  (h1 : club.total = 30)
  (h2 : club.badminton = 18)
  (h3 : club.tennis = 19)
  (h4 : club.neither = 2) :
  playBoth club = 9 := by
  sorry

#eval playBoth { total := 30, badminton := 18, tennis := 19, neither := 2 }

end NUMINAMATH_CALUDE_members_playing_both_l1800_180065


namespace NUMINAMATH_CALUDE_magnitude_of_2_plus_i_l1800_180021

theorem magnitude_of_2_plus_i : Complex.abs (2 + Complex.I) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_2_plus_i_l1800_180021


namespace NUMINAMATH_CALUDE_min_value_expression_l1800_180073

theorem min_value_expression (x : ℝ) (h : x > 0) :
  4 * x + 9 / x^2 ≥ 3 * (36 : ℝ)^(1/3) ∧
  ∃ y > 0, 4 * y + 9 / y^2 = 3 * (36 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1800_180073


namespace NUMINAMATH_CALUDE_correct_num_cars_l1800_180055

/-- Represents the number of cars taken on the hike -/
def num_cars : ℕ := 3

/-- Represents the number of taxis taken on the hike -/
def num_taxis : ℕ := 6

/-- Represents the number of vans taken on the hike -/
def num_vans : ℕ := 2

/-- Represents the number of people in each car -/
def people_per_car : ℕ := 4

/-- Represents the number of people in each taxi -/
def people_per_taxi : ℕ := 6

/-- Represents the number of people in each van -/
def people_per_van : ℕ := 5

/-- Represents the total number of people on the hike -/
def total_people : ℕ := 58

/-- Theorem stating that the number of cars is correct given the conditions -/
theorem correct_num_cars :
  num_cars * people_per_car +
  num_taxis * people_per_taxi +
  num_vans * people_per_van = total_people :=
by sorry

end NUMINAMATH_CALUDE_correct_num_cars_l1800_180055


namespace NUMINAMATH_CALUDE_solution_problem_l1800_180087

theorem solution_problem : ∃ (x y : ℤ), x > y ∧ y > 0 ∧ x + y + x * y = 80 ∧ x = 26 := by
  sorry

end NUMINAMATH_CALUDE_solution_problem_l1800_180087


namespace NUMINAMATH_CALUDE_product_of_roots_l1800_180058

theorem product_of_roots (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂)
  (h₁ : x₁^2 - 2006*x₁ = 1)
  (h₂ : x₂^2 - 2006*x₂ = 1) : 
  x₁ * x₂ = -1 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_l1800_180058


namespace NUMINAMATH_CALUDE_class_average_height_l1800_180004

/-- The average height of a class of girls -/
theorem class_average_height 
  (total_girls : ℕ) 
  (group1_girls : ℕ) 
  (group1_avg_height : ℝ) 
  (group2_avg_height : ℝ) 
  (h1 : total_girls = 40)
  (h2 : group1_girls = 30)
  (h3 : group1_avg_height = 160)
  (h4 : group2_avg_height = 156) :
  (group1_girls * group1_avg_height + (total_girls - group1_girls) * group2_avg_height) / total_girls = 159 := by
  sorry


end NUMINAMATH_CALUDE_class_average_height_l1800_180004


namespace NUMINAMATH_CALUDE_smallest_qnnn_l1800_180074

def is_two_digit_with_equal_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 11 = 0

def is_one_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

def satisfies_condition (nn : ℕ) (n : ℕ) (qnnn : ℕ) : Prop :=
  is_two_digit_with_equal_digits nn ∧
  is_one_digit n ∧
  nn * n = qnnn ∧
  1000 ≤ qnnn ∧ qnnn ≤ 9999 ∧
  qnnn % 1000 % 100 % 10 = n ∧
  qnnn % 1000 % 100 / 10 = n ∧
  qnnn % 1000 / 100 = n

theorem smallest_qnnn :
  ∀ qnnn : ℕ, (∃ nn n : ℕ, satisfies_condition nn n qnnn) →
  2555 ≤ qnnn :=
sorry

end NUMINAMATH_CALUDE_smallest_qnnn_l1800_180074


namespace NUMINAMATH_CALUDE_spinner_probability_l1800_180080

theorem spinner_probability (p_A p_B p_C p_D p_E : ℚ) :
  p_A = 3/8 →
  p_B = 1/4 →
  p_C = p_D →
  p_C = p_E →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/8 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l1800_180080


namespace NUMINAMATH_CALUDE_horse_race_equation_l1800_180061

/-- The speed of the good horse in miles per day -/
def good_horse_speed : ℕ := 200

/-- The speed of the slow horse in miles per day -/
def slow_horse_speed : ℕ := 120

/-- The number of days the slow horse starts earlier -/
def head_start : ℕ := 10

/-- The number of days it takes for the good horse to catch up -/
def catch_up_days : ℕ := sorry

theorem horse_race_equation :
  good_horse_speed * catch_up_days = slow_horse_speed * catch_up_days + slow_horse_speed * head_start :=
by sorry

end NUMINAMATH_CALUDE_horse_race_equation_l1800_180061


namespace NUMINAMATH_CALUDE_factorization_equality_l1800_180027

theorem factorization_equality (m n : ℝ) : 2*m*n^2 - 12*m*n + 18*m = 2*m*(n-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1800_180027


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1800_180024

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1800_180024


namespace NUMINAMATH_CALUDE_crystal_barrettes_count_l1800_180092

/-- The cost of one set of barrettes in dollars -/
def barrette_cost : ℕ := 3

/-- The cost of one comb in dollars -/
def comb_cost : ℕ := 1

/-- The total amount spent by both girls in dollars -/
def total_spent : ℕ := 14

/-- The number of sets of barrettes Kristine bought -/
def kristine_barrettes : ℕ := 1

/-- The number of combs Kristine bought -/
def kristine_combs : ℕ := 1

/-- The number of combs Crystal bought -/
def crystal_combs : ℕ := 1

/-- 
Given the costs of barrettes and combs, and the purchasing information for Kristine and Crystal,
prove that Crystal bought 3 sets of barrettes.
-/
theorem crystal_barrettes_count : 
  ∃ (x : ℕ), 
    barrette_cost * (kristine_barrettes + x) + 
    comb_cost * (kristine_combs + crystal_combs) = 
    total_spent ∧ x = 3 := by
  sorry


end NUMINAMATH_CALUDE_crystal_barrettes_count_l1800_180092


namespace NUMINAMATH_CALUDE_total_classes_is_nine_l1800_180053

/-- The number of classes taught by Eduardo and Frankie -/
def total_classes (eduardo_classes : ℕ) (frankie_multiplier : ℕ) : ℕ :=
  eduardo_classes + eduardo_classes * frankie_multiplier

/-- Theorem stating that the total number of classes taught by Eduardo and Frankie is 9 -/
theorem total_classes_is_nine :
  total_classes 3 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_classes_is_nine_l1800_180053


namespace NUMINAMATH_CALUDE_karlson_candies_theorem_l1800_180018

/-- The number of ones initially on the board -/
def initial_ones : ℕ := 29

/-- The number of minutes the process continues -/
def total_minutes : ℕ := 29

/-- Calculates the number of edges in a complete graph with n vertices -/
def complete_graph_edges (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- The maximum number of candies Karlson could eat -/
def max_candies : ℕ := complete_graph_edges initial_ones

theorem karlson_candies_theorem :
  max_candies = 406 :=
sorry

end NUMINAMATH_CALUDE_karlson_candies_theorem_l1800_180018


namespace NUMINAMATH_CALUDE_function_derivative_positive_l1800_180078

/-- Given a function y = 2mx^2 + (1-4m)x + 2m - 1, prove that when m = -1 and x < 5/4, 
    the derivative of y with respect to x is positive. -/
theorem function_derivative_positive (x : ℝ) (h : x < 5/4) : 
  let m : ℝ := -1
  let y : ℝ → ℝ := λ x => 2*m*x^2 + (1-4*m)*x + 2*m - 1
  (deriv y) x > 0 := by sorry

end NUMINAMATH_CALUDE_function_derivative_positive_l1800_180078


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1800_180003

theorem polynomial_division_remainder : 
  let dividend := fun z : ℝ => 4 * z^3 + 5 * z^2 - 20 * z + 7
  let divisor := fun z : ℝ => 4 * z - 3
  let quotient := fun z : ℝ => z^2 + 2 * z + 1/4
  let remainder := fun z : ℝ => -15 * z + 31/4
  ∀ z : ℝ, dividend z = divisor z * quotient z + remainder z :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1800_180003


namespace NUMINAMATH_CALUDE_min_side_length_is_optimal_l1800_180072

/-- The minimum side length of a square piece of land with an area of at least 400 square feet -/
def min_side_length : ℝ := 20

/-- The area of the square land is at least 400 square feet -/
axiom area_constraint : min_side_length ^ 2 ≥ 400

/-- The minimum side length is optimal -/
theorem min_side_length_is_optimal :
  ∀ s : ℝ, s ^ 2 ≥ 400 → s ≥ min_side_length :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_is_optimal_l1800_180072


namespace NUMINAMATH_CALUDE_circle_radii_order_l1800_180052

theorem circle_radii_order (rA rB rC : ℝ) : 
  rA = Real.sqrt 10 →
  2 * Real.pi * rB = 10 * Real.pi →
  Real.pi * rC^2 = 25 * Real.pi →
  rA ≤ rB ∧ rB ≤ rC := by
  sorry

end NUMINAMATH_CALUDE_circle_radii_order_l1800_180052


namespace NUMINAMATH_CALUDE_pages_left_to_read_l1800_180070

theorem pages_left_to_read 
  (total_pages : ℕ) 
  (pages_read : ℕ) 
  (pages_to_skip : ℕ) 
  (h1 : total_pages = 372) 
  (h2 : pages_read = 125) 
  (h3 : pages_to_skip = 16) :
  total_pages - (pages_read + pages_to_skip) = 231 :=
by sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l1800_180070


namespace NUMINAMATH_CALUDE_composite_number_l1800_180093

theorem composite_number (n : ℕ+) : ∃ (p : ℕ), Prime p ∧ p ∣ (19 * 8^n.val + 17) ∧ 1 < p ∧ p < 19 * 8^n.val + 17 := by
  sorry

end NUMINAMATH_CALUDE_composite_number_l1800_180093


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_nonnegative_l1800_180069

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x - a > 0}
def B : Set ℝ := {x | x ≤ 0}

-- State the theorem
theorem intersection_empty_implies_a_nonnegative (a : ℝ) :
  A a ∩ B = ∅ → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_nonnegative_l1800_180069


namespace NUMINAMATH_CALUDE_maria_dimes_l1800_180038

/-- The number of dimes Maria has in her piggy bank -/
def num_dimes : ℕ := sorry

/-- The number of quarters Maria has initially -/
def initial_quarters : ℕ := 4

/-- The number of nickels Maria has -/
def nickels : ℕ := 7

/-- The number of quarters Maria's mom gives her -/
def additional_quarters : ℕ := 5

/-- The total value in Maria's piggy bank in cents -/
def total_value : ℕ := 300

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

theorem maria_dimes :
  num_dimes * dime_value + 
  (initial_quarters + additional_quarters) * quarter_value + 
  nickels * nickel_value = total_value :=
by sorry

end NUMINAMATH_CALUDE_maria_dimes_l1800_180038


namespace NUMINAMATH_CALUDE_jonah_running_time_l1800_180060

/-- Represents the problem of determining Jonah's running time. -/
theorem jonah_running_time (calories_per_hour : ℕ) (extra_time : ℕ) (extra_calories : ℕ) : 
  calories_per_hour = 30 →
  extra_time = 5 →
  extra_calories = 90 →
  ∃ (actual_time : ℕ), 
    actual_time * calories_per_hour = (actual_time + extra_time) * calories_per_hour - extra_calories ∧
    actual_time = 2 :=
by sorry

end NUMINAMATH_CALUDE_jonah_running_time_l1800_180060


namespace NUMINAMATH_CALUDE_Z_in_third_quadrant_l1800_180075

-- Define the complex number Z
def Z : ℂ := -1 + (1 - Complex.I)^2

-- Theorem stating that Z is in the third quadrant
theorem Z_in_third_quadrant :
  Real.sign (Z.re) = -1 ∧ Real.sign (Z.im) = -1 :=
sorry


end NUMINAMATH_CALUDE_Z_in_third_quadrant_l1800_180075


namespace NUMINAMATH_CALUDE_adam_father_deposit_l1800_180007

/-- Calculates the total amount after a given period, including initial deposit and interest --/
def totalAmount (initialDeposit : ℝ) (interestRate : ℝ) (years : ℝ) : ℝ :=
  initialDeposit + (initialDeposit * interestRate * years)

/-- Proves that given the specified conditions, the total amount after 2.5 years is $2400 --/
theorem adam_father_deposit :
  let initialDeposit : ℝ := 2000
  let interestRate : ℝ := 0.08
  let years : ℝ := 2.5
  totalAmount initialDeposit interestRate years = 2400 := by
  sorry

end NUMINAMATH_CALUDE_adam_father_deposit_l1800_180007


namespace NUMINAMATH_CALUDE_power_subtraction_equals_6444_l1800_180057

theorem power_subtraction_equals_6444 : 3^(1+3+4) - (3^1 * 3 + 3^3 + 3^4) = 6444 := by
  sorry

end NUMINAMATH_CALUDE_power_subtraction_equals_6444_l1800_180057


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l1800_180030

theorem quadratic_root_implies_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 3*x + a = 0) ∧ (2^2 + 3*2 + a = 0) → a = -10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l1800_180030


namespace NUMINAMATH_CALUDE_parallelogram_base_l1800_180012

/-- 
Given a parallelogram with area 320 cm² and height 16 cm, 
prove that its base is 20 cm.
-/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 320 ∧ height = 16 ∧ area = base * height → base = 20 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l1800_180012


namespace NUMINAMATH_CALUDE_original_number_is_200_l1800_180098

theorem original_number_is_200 : 
  ∃ x : ℝ, (x - 25 = 0.75 * x + 25) ∧ x = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_200_l1800_180098


namespace NUMINAMATH_CALUDE_caitlin_uniform_number_l1800_180081

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ Nat.Prime n

theorem caitlin_uniform_number
  (a b c : ℕ)
  (ha : is_two_digit_prime a)
  (hb : is_two_digit_prime b)
  (hc : is_two_digit_prime c)
  (hab : a ≠ b)
  (hac : a ≠ c)
  (hbc : b ≠ c)
  (sum_ac : a + c = 24)
  (sum_ab : a + b = 30)
  (sum_bc : b + c = 28) :
  c = 11 := by
  sorry

end NUMINAMATH_CALUDE_caitlin_uniform_number_l1800_180081


namespace NUMINAMATH_CALUDE_last_digit_2_power_2010_l1800_180083

/-- The last digit of 2^n for n ≥ 1 -/
def lastDigitPowerOfTwo (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 0 => 6
  | _ => 0  -- This case should never occur

theorem last_digit_2_power_2010 : lastDigitPowerOfTwo 2010 = 4 := by
  sorry

#eval lastDigitPowerOfTwo 2010

end NUMINAMATH_CALUDE_last_digit_2_power_2010_l1800_180083


namespace NUMINAMATH_CALUDE_oranges_thrown_away_l1800_180009

/-- Proves that 2 old oranges were thrown away given the initial, added, and final orange counts. -/
theorem oranges_thrown_away (initial : ℕ) (added : ℕ) (final : ℕ) 
    (h1 : initial = 5)
    (h2 : added = 28)
    (h3 : final = 31) :
  initial - (initial + added - final) = 2 := by
  sorry

end NUMINAMATH_CALUDE_oranges_thrown_away_l1800_180009


namespace NUMINAMATH_CALUDE_cycle_original_price_l1800_180051

/-- Given a cycle sold at a 25% loss for Rs. 1350, prove its original price was Rs. 1800. -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) :
  selling_price = 1350 →
  loss_percentage = 25 →
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧
    original_price = 1800 :=
by sorry

end NUMINAMATH_CALUDE_cycle_original_price_l1800_180051


namespace NUMINAMATH_CALUDE_wallpaper_overlap_area_l1800_180026

theorem wallpaper_overlap_area
  (total_area : ℝ)
  (double_layer_area : ℝ)
  (triple_layer_area : ℝ)
  (h1 : total_area = 300)
  (h2 : double_layer_area = 38)
  (h3 : triple_layer_area = 41) :
  total_area - 2 * double_layer_area - 3 * triple_layer_area = 101 :=
by
  sorry

end NUMINAMATH_CALUDE_wallpaper_overlap_area_l1800_180026


namespace NUMINAMATH_CALUDE_simplify_fraction_l1800_180042

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (4 * x) / (x^2 - 4) - 2 / (x - 2) - 1 = -x / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1800_180042


namespace NUMINAMATH_CALUDE_range_of_a_for_intersection_equality_l1800_180071

/-- The set A defined by the quadratic equation x^2 - 3x + 2 = 0 -/
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

/-- The set B defined by the quadratic equation x^2 - ax + 3a - 5 = 0, parameterized by a -/
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + 3*a - 5 = 0}

/-- The theorem stating the range of a for which A ∩ B = B -/
theorem range_of_a_for_intersection_equality :
  ∀ a : ℝ, (A ∩ B a = B a) → (a ∈ Set.Icc 2 10 ∪ {1}) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_intersection_equality_l1800_180071


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1800_180002

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2*x - 6| = 3*x + 6) ↔ (x = 0) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1800_180002


namespace NUMINAMATH_CALUDE_stamp_collection_value_l1800_180022

theorem stamp_collection_value (partial_value : ℚ) (partial_fraction : ℚ) (total_value : ℚ) : 
  partial_fraction = 4/7 ∧ partial_value = 28 → total_value = 49 :=
by sorry

end NUMINAMATH_CALUDE_stamp_collection_value_l1800_180022


namespace NUMINAMATH_CALUDE_company_workers_l1800_180054

theorem company_workers (total : ℕ) (men : ℕ) (women : ℕ) : 
  (3 * total / 10 = men) →  -- One-third without plan, 40% of those with plan are men
  (3 * total / 5 = men + women) →  -- Total workers
  (men = 120) →  -- Given number of men
  (women = 180) :=
sorry

end NUMINAMATH_CALUDE_company_workers_l1800_180054


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l1800_180036

theorem triangle_side_lengths 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : c = 10)
  (h2 : Real.cos A / Real.cos B = b / a)
  (h3 : b / a = 4 / 3) :
  a = 6 ∧ b = 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l1800_180036


namespace NUMINAMATH_CALUDE_train_crossing_time_l1800_180062

/-- The time taken for two trains to cross each other -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 120 →
  train_speed_kmh = 18 →
  (2 * train_length) / (2 * train_speed_kmh * (1000 / 3600)) = 24 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1800_180062


namespace NUMINAMATH_CALUDE_representations_of_2022_l1800_180085

/-- Represents a sequence of consecutive natural numbers. -/
structure ConsecutiveSequence where
  start : ℕ
  length : ℕ

/-- The sum of a consecutive sequence of natural numbers. -/
def sum_consecutive (seq : ConsecutiveSequence) : ℕ :=
  seq.length * (2 * seq.start + seq.length - 1) / 2

/-- Checks if a consecutive sequence sums to a given target. -/
def is_valid_representation (seq : ConsecutiveSequence) (target : ℕ) : Prop :=
  sum_consecutive seq = target

theorem representations_of_2022 :
  ∀ (seq : ConsecutiveSequence),
    is_valid_representation seq 2022 ↔
      (seq.start = 673 ∧ seq.length = 3) ∨
      (seq.start = 504 ∧ seq.length = 4) ∨
      (seq.start = 163 ∧ seq.length = 12) :=
by sorry

end NUMINAMATH_CALUDE_representations_of_2022_l1800_180085


namespace NUMINAMATH_CALUDE_constant_sum_of_powers_l1800_180044

theorem constant_sum_of_powers (n : ℕ) : 
  (∀ x y z : ℝ, x + y + z = 0 → x * y * z = 1 → 
    ∃ c : ℝ, ∀ x' y' z' : ℝ, x' + y' + z' = 0 → x' * y' * z' = 1 → 
      x'^n + y'^n + z'^n = c) ↔ 
  n = 1 ∨ n = 3 := by sorry

end NUMINAMATH_CALUDE_constant_sum_of_powers_l1800_180044


namespace NUMINAMATH_CALUDE_circle_tangency_l1800_180005

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

theorem circle_tangency (m : ℝ) : 
  externally_tangent (m, 0) (0, 2) (|m|) 1 → m = 3/2 ∨ m = -3/2 := by
  sorry

#check circle_tangency

end NUMINAMATH_CALUDE_circle_tangency_l1800_180005


namespace NUMINAMATH_CALUDE_kittens_at_shelter_l1800_180043

def number_of_puppies : ℕ := 32

def number_of_kittens : ℕ := 2 * number_of_puppies + 14

theorem kittens_at_shelter : number_of_kittens = 78 := by
  sorry

end NUMINAMATH_CALUDE_kittens_at_shelter_l1800_180043


namespace NUMINAMATH_CALUDE_sum_of_ab_l1800_180067

theorem sum_of_ab (a b : ℝ) (h : a^2 + b^2 + a^2*b^2 = 4*a*b - 1) :
  a + b = 2 ∨ a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ab_l1800_180067


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l1800_180016

theorem rectangular_solid_volume
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 15)
  (h_front : front_area = 10)
  (h_bottom : bottom_area = 6)
  (x y z : ℝ)
  (h_xy : x * y = side_area)
  (h_yz : y * z = front_area)
  (h_xz : x * z = bottom_area)
  (h_relation : x = 3 * y) :
  x * y * z = 6 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l1800_180016


namespace NUMINAMATH_CALUDE_E_is_top_leftmost_l1800_180050

-- Define the structure for a rectangle
structure Rectangle where
  w : Int
  x : Int
  y : Int
  z : Int

-- Define the five rectangles
def A : Rectangle := { w := 4, x := 1, y := 6, z := 9 }
def B : Rectangle := { w := 1, x := 0, y := 3, z := 6 }
def C : Rectangle := { w := 3, x := 8, y := 5, z := 2 }
def D : Rectangle := { w := 7, x := 5, y := 4, z := 8 }
def E : Rectangle := { w := 9, x := 2, y := 7, z := 0 }

-- Define the placement rules
def isLeftmost (r : Rectangle) : Bool :=
  r.w = 1 ∨ r.w = 9

def isRightmost (r : Rectangle) : Bool :=
  r.y = 6 ∨ r.y = 5

def isCenter (r : Rectangle) : Bool :=
  ¬(isLeftmost r) ∧ ¬(isRightmost r)

-- Theorem to prove
theorem E_is_top_leftmost :
  isLeftmost E ∧ 
  isRightmost A ∧ 
  isRightmost C ∧ 
  isLeftmost B ∧ 
  isCenter D :=
sorry

end NUMINAMATH_CALUDE_E_is_top_leftmost_l1800_180050


namespace NUMINAMATH_CALUDE_combined_selling_price_l1800_180059

/-- Calculate the combined selling price of three items with given costs, exchange rate, profits, discount, and tax. -/
theorem combined_selling_price (exchange_rate : ℝ) (cost_a cost_b cost_c : ℝ)
  (profit_a profit_b profit_c : ℝ) (discount_b tax : ℝ) :
  exchange_rate = 70 ∧
  cost_a = 10 ∧
  cost_b = 15 ∧
  cost_c = 20 ∧
  profit_a = 0.25 ∧
  profit_b = 0.30 ∧
  profit_c = 0.20 ∧
  discount_b = 0.10 ∧
  tax = 0.08 →
  let cost_rs_a := cost_a * exchange_rate
  let cost_rs_b := cost_b * exchange_rate * (1 - discount_b)
  let cost_rs_c := cost_c * exchange_rate
  let selling_price_a := cost_rs_a * (1 + profit_a) * (1 + tax)
  let selling_price_b := cost_rs_b * (1 + profit_b) * (1 + tax)
  let selling_price_c := cost_rs_c * (1 + profit_c) * (1 + tax)
  selling_price_a + selling_price_b + selling_price_c = 4086.18 := by
sorry


end NUMINAMATH_CALUDE_combined_selling_price_l1800_180059


namespace NUMINAMATH_CALUDE_constant_term_in_system_l1800_180037

theorem constant_term_in_system (x y C : ℝ) : 
  (5 * x + y = 19) → 
  (x + 3 * y = C) → 
  (3 * x + 2 * y = 10) → 
  C = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_in_system_l1800_180037
