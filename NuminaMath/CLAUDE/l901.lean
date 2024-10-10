import Mathlib

namespace sqrt_equation_solution_l901_90132

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (x + 8) - (4 / Real.sqrt (x + 8)) = 3 ∧ x = 8 :=
by sorry

end sqrt_equation_solution_l901_90132


namespace equation_a_solution_l901_90149

theorem equation_a_solution (x : ℝ) : 
  1/(x-1) + 3/(x-3) - 9/(x-5) + 5/(x-7) = 0 ↔ x = 2 :=
by sorry

end equation_a_solution_l901_90149


namespace park_diagonal_ratio_l901_90143

theorem park_diagonal_ratio :
  ∀ (long_side : ℝ) (short_side : ℝ) (diagonal : ℝ),
    short_side = long_side / 2 →
    long_side + short_side - diagonal = long_side / 3 →
    long_side / diagonal = 2 * Real.sqrt 5 / 5 := by
  sorry

end park_diagonal_ratio_l901_90143


namespace inequality_proof_l901_90145

def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

theorem inequality_proof (m : ℝ) (a b c : ℝ) 
  (h1 : Set.Icc 0 2 = {x | f m (x + 1) ≥ 0})
  (h2 : 1/a + 1/(2*b) + 1/(3*c) = m) : 
  a + 2*b + 3*c ≥ 9 := by sorry

end inequality_proof_l901_90145


namespace complex_power_sum_l901_90158

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the property that i^2 = -1
axiom i_squared : i ^ 2 = -1

-- Define the cyclic nature of powers of i
axiom i_cyclic (n : ℕ) : i ^ (n + 4) = i ^ n

-- State the theorem
theorem complex_power_sum : i^20 + i^33 - i^56 = i := by sorry

end complex_power_sum_l901_90158


namespace reciprocal_lcm_24_221_l901_90120

theorem reciprocal_lcm_24_221 :
  let a : ℕ := 24
  let b : ℕ := 221
  Nat.gcd a b = 1 →
  (1 : ℚ) / (Nat.lcm a b) = 1 / 5304 := by
  sorry

end reciprocal_lcm_24_221_l901_90120


namespace total_reduction_proof_l901_90122

-- Define the original price and reduction percentages
def original_price : ℝ := 500
def first_reduction : ℝ := 0.07
def second_reduction : ℝ := 0.05
def third_reduction : ℝ := 0.03

-- Define the function to calculate the price after reductions
def price_after_reductions (p : ℝ) (r1 r2 r3 : ℝ) : ℝ :=
  p * (1 - r1) * (1 - r2) * (1 - r3)

-- Theorem statement
theorem total_reduction_proof :
  original_price - price_after_reductions original_price first_reduction second_reduction third_reduction = 71.5025 := by
  sorry


end total_reduction_proof_l901_90122


namespace intersection_M_N_l901_90137

def M : Set ℕ := {0, 2, 3, 4}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0, 4} := by sorry

end intersection_M_N_l901_90137


namespace ten_factorial_minus_nine_factorial_l901_90100

-- Define factorial function
def factorial : ℕ → ℕ
| 0 => 1
| n + 1 => (n + 1) * factorial n

-- State the theorem
theorem ten_factorial_minus_nine_factorial :
  factorial 10 - factorial 9 = 3265920 := by
  sorry

end ten_factorial_minus_nine_factorial_l901_90100


namespace chessboard_one_color_l901_90101

/-- Represents the color of a square on the chessboard -/
inductive Color
| Black
| White

/-- Represents the chessboard as a function from coordinates to colors -/
def Chessboard := Fin 8 → Fin 8 → Color

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  x1 : Fin 8
  y1 : Fin 8
  x2 : Fin 8
  y2 : Fin 8

/-- Checks if a rectangle is adjacent to a corner of the board -/
def isCornerRectangle (r : Rectangle) : Prop :=
  (r.x1 = 0 ∧ r.y1 = 0) ∨
  (r.x1 = 0 ∧ r.y2 = 7) ∨
  (r.x2 = 7 ∧ r.y1 = 0) ∨
  (r.x2 = 7 ∧ r.y2 = 7)

/-- The operation of changing colors in a rectangle -/
def applyRectangle (board : Chessboard) (r : Rectangle) : Chessboard :=
  sorry

/-- Theorem stating that any chessboard can be made one color -/
theorem chessboard_one_color :
  ∀ (initial : Chessboard),
  ∃ (final : Chessboard) (steps : List Rectangle),
    (∀ r ∈ steps, isCornerRectangle r) ∧
    (final = steps.foldl applyRectangle initial) ∧
    (∃ c : Color, ∀ x y : Fin 8, final x y = c) :=
  sorry

end chessboard_one_color_l901_90101


namespace ratio_sum_to_base_l901_90198

theorem ratio_sum_to_base (a b : ℚ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end ratio_sum_to_base_l901_90198


namespace abc_inequalities_l901_90128

theorem abc_inequalities (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : 4*a^2 + b^2 + 16*c^2 = 1) : 
  (0 < a*b ∧ a*b < 1/4) ∧ 
  (1/a^2 + 1/b^2 + 1/(4*a*b*c^2) > 49) := by
  sorry

end abc_inequalities_l901_90128


namespace smallest_n_satisfying_conditions_three_million_two_hundred_thousand_satisfies_conditions_smallest_n_is_three_million_two_hundred_thousand_l901_90176

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_fifth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k^5

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by n 20 ∧ is_perfect_square (n^2) ∧ is_perfect_fifth_power (n^3)

theorem smallest_n_satisfying_conditions :
  ∀ m : ℕ, m > 0 → satisfies_conditions m → m ≥ 3200000 :=
by sorry

theorem three_million_two_hundred_thousand_satisfies_conditions :
  satisfies_conditions 3200000 :=
by sorry

theorem smallest_n_is_three_million_two_hundred_thousand :
  (∀ m : ℕ, m > 0 → satisfies_conditions m → m ≥ 3200000) ∧
  satisfies_conditions 3200000 :=
by sorry

end smallest_n_satisfying_conditions_three_million_two_hundred_thousand_satisfies_conditions_smallest_n_is_three_million_two_hundred_thousand_l901_90176


namespace trigonometric_inequality_l901_90142

theorem trigonometric_inequality (φ : Real) (h : 0 < φ ∧ φ < Real.pi / 2) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end trigonometric_inequality_l901_90142


namespace exactly_one_first_class_product_l901_90140

theorem exactly_one_first_class_product (p1 p2 : ℝ) 
  (h1 : p1 = 2/3) 
  (h2 : p2 = 3/4) : 
  p1 * (1 - p2) + (1 - p1) * p2 = 5/12 := by
sorry

end exactly_one_first_class_product_l901_90140


namespace fraction_power_product_l901_90178

theorem fraction_power_product : (3/4)^4 * (1/5) = 81/1280 := by
  sorry

end fraction_power_product_l901_90178


namespace trigonometric_equation_solution_l901_90124

theorem trigonometric_equation_solution (x : ℝ) :
  8.483 * Real.tan x - Real.sin (2 * x) - Real.cos (2 * x) + 2 * (2 * Real.cos x - 1 / Real.cos x) = 0 ↔
  ∃ k : ℤ, x = π / 4 * (2 * k + 1) := by
  sorry

end trigonometric_equation_solution_l901_90124


namespace range_of_x_l901_90157

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) 
  (h3 : Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) : 
  x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4) := by
sorry

end range_of_x_l901_90157


namespace product_of_binary_and_ternary_l901_90174

-- Define a function to convert binary to decimal
def binary_to_decimal (b : List Bool) : ℕ := sorry

-- Define a function to convert ternary to decimal
def ternary_to_decimal (t : List ℕ) : ℕ := sorry

-- Theorem statement
theorem product_of_binary_and_ternary :
  let binary_num := [true, true, false, true]  -- Represents 1101₂
  let ternary_num := [2, 0, 2]  -- Represents 202₃
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 260 := by sorry

end product_of_binary_and_ternary_l901_90174


namespace rectangle_area_measurement_error_l901_90150

theorem rectangle_area_measurement_error 
  (L W : ℝ) (L_measured W_measured : ℝ) 
  (h1 : L_measured = L * 1.2) 
  (h2 : W_measured = W * 0.9) : 
  (L_measured * W_measured - L * W) / (L * W) * 100 = 8 := by
sorry

end rectangle_area_measurement_error_l901_90150


namespace distance_between_points_l901_90102

/-- The distance between two points given rowing speed, stream speed, and round trip time -/
theorem distance_between_points (rowing_speed stream_speed : ℝ) (round_trip_time : ℝ) :
  rowing_speed = 10 →
  stream_speed = 2 →
  round_trip_time = 5 →
  ∃ (distance : ℝ),
    distance / (rowing_speed + stream_speed) + distance / (rowing_speed - stream_speed) = round_trip_time ∧
    distance = 24 := by
  sorry

end distance_between_points_l901_90102


namespace plot_length_is_52_l901_90107

/-- Represents a rectangular plot with specific fencing conditions -/
structure Plot where
  breadth : ℝ
  length : ℝ
  flatCost : ℝ
  risePercent : ℝ
  totalRise : ℝ
  totalCost : ℝ

/-- Calculates the length of the plot given the conditions -/
def calculateLength (p : Plot) : ℝ :=
  p.breadth + 20

/-- Theorem stating the length of the plot under given conditions -/
theorem plot_length_is_52 (p : Plot) 
  (h1 : p.length = p.breadth + 20)
  (h2 : p.flatCost = 26.5)
  (h3 : p.risePercent = 0.1)
  (h4 : p.totalRise = 5)
  (h5 : p.totalCost = 5300)
  (h6 : p.totalCost = 2 * (p.breadth + 20) * p.flatCost + 
        2 * p.breadth * (p.flatCost * (1 + p.risePercent * p.totalRise))) :
  calculateLength p = 52 := by
  sorry

#eval calculateLength { breadth := 32, length := 52, flatCost := 26.5, 
                        risePercent := 0.1, totalRise := 5, totalCost := 5300 }

end plot_length_is_52_l901_90107


namespace f_properties_l901_90131

-- Define the function f(x) = x^2 + ln|x|
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log (abs x)

-- State the theorem
theorem f_properties :
  -- f is defined for all non-zero real numbers
  (∀ x : ℝ, x ≠ 0 → f x ≠ 0) ∧
  -- f is an even function
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  -- f is increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) :=
by sorry

end f_properties_l901_90131


namespace fraction_calculation_l901_90125

theorem fraction_calculation :
  (1 / 5 + 1 / 7) / (3 / 8 + 2 / 9) = 864 / 1505 := by
  sorry

end fraction_calculation_l901_90125


namespace perpendicular_tangents_imply_m_value_l901_90116

/-- The original function F1 -/
def F1 (x : ℝ) : ℝ := x^2

/-- The translated function F2 -/
def F2 (m : ℝ) (x : ℝ) : ℝ := (x - m)^2 - 1

/-- The derivative of F1 -/
def F1_derivative (x : ℝ) : ℝ := 2 * x

/-- The derivative of F2 -/
def F2_derivative (m : ℝ) (x : ℝ) : ℝ := 2 * (x - m)

theorem perpendicular_tangents_imply_m_value :
  ∀ m : ℝ, (F1_derivative 1 * F2_derivative m 1 = -1) → m = 5/4 := by
  sorry

end perpendicular_tangents_imply_m_value_l901_90116


namespace no_four_digit_perfect_square_palindromes_l901_90119

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end no_four_digit_perfect_square_palindromes_l901_90119


namespace original_cyclists_l901_90141

theorem original_cyclists (total_bill : ℕ) (left_cyclists : ℕ) (extra_payment : ℕ) :
  total_bill = 80 ∧ left_cyclists = 2 ∧ extra_payment = 2 →
  ∃ x : ℕ, x > 0 ∧ (total_bill / (x - left_cyclists) = total_bill / x + extra_payment) ∧ x = 10 :=
by
  sorry

#check original_cyclists

end original_cyclists_l901_90141


namespace third_shiny_penny_probability_l901_90164

def total_pennies : ℕ := 9
def shiny_pennies : ℕ := 4
def dull_pennies : ℕ := 5

def probability_more_than_five_draws : ℚ :=
  37 / 63

theorem third_shiny_penny_probability :
  probability_more_than_five_draws =
    (Nat.choose 5 2 * Nat.choose 4 1 +
     Nat.choose 5 1 * Nat.choose 4 2 +
     Nat.choose 5 0 * Nat.choose 4 3) /
    Nat.choose total_pennies shiny_pennies :=
by sorry

end third_shiny_penny_probability_l901_90164


namespace travel_time_calculation_l901_90190

/-- Given a speed of 25 km/hr and a distance of 125 km, the time taken is 5 hours. -/
theorem travel_time_calculation (speed : ℝ) (distance : ℝ) (time : ℝ) : 
  speed = 25 → distance = 125 → time = distance / speed → time = 5 := by
  sorry

end travel_time_calculation_l901_90190


namespace not_perfect_square_zero_six_l901_90173

/-- A number composed only of digits 0 and 6 -/
def DigitsZeroSix (m : ℕ) : Prop :=
  ∀ d, d ∈ m.digits 10 → d = 0 ∨ d = 6

/-- The sum of digits of a natural number -/
def DigitSum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem not_perfect_square_zero_six (m : ℕ) (h : DigitsZeroSix m) : 
  ¬∃ k : ℕ, m = k^2 := by
  sorry

end not_perfect_square_zero_six_l901_90173


namespace success_permutations_count_l901_90118

/-- The number of distinct permutations of the multiset {S, S, S, U, C, C, E} -/
def successPermutations : ℕ :=
  Nat.factorial 7 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of SUCCESS is 420 -/
theorem success_permutations_count : successPermutations = 420 := by
  sorry

end success_permutations_count_l901_90118


namespace x_range_equivalence_l901_90193

theorem x_range_equivalence (x : ℝ) : 
  (∀ a b : ℝ, a > 0 → b > 0 → x^2 + 2*x < a/b + 16*b/a) ↔ x > -4 ∧ x < 2 := by
sorry

end x_range_equivalence_l901_90193


namespace third_median_length_l901_90169

/-- A triangle with two known medians and area -/
structure Triangle where
  median1 : ℝ
  median2 : ℝ
  area : ℝ

/-- The length of the third median in a triangle -/
def third_median (t : Triangle) : ℝ := sorry

theorem third_median_length (t : Triangle) 
  (h1 : t.median1 = 4)
  (h2 : t.median2 = 5)
  (h3 : t.area = 10 * Real.sqrt 3) :
  third_median t = 3 * Real.sqrt 10 := by sorry

end third_median_length_l901_90169


namespace wedding_cost_theorem_l901_90139

/-- Calculate the total cost of John's wedding based on given parameters. -/
def wedding_cost (venue_cost : ℕ) (cost_per_guest : ℕ) (johns_guests : ℕ) (wife_increase_percent : ℕ) : ℕ :=
  let total_guests := johns_guests + (johns_guests * wife_increase_percent) / 100
  venue_cost + total_guests * cost_per_guest

/-- Theorem stating the total cost of the wedding given the specified conditions. -/
theorem wedding_cost_theorem :
  wedding_cost 10000 500 50 60 = 50000 := by
  sorry

end wedding_cost_theorem_l901_90139


namespace quadratic_equation_roots_l901_90160

theorem quadratic_equation_roots (a : ℝ) (m : ℝ) :
  let x₁ : ℝ := Real.sqrt (a + 2) - Real.sqrt (8 - a) + Real.sqrt (-a^2)
  (∃ x₂ : ℝ, (1/2) * m * x₁^2 + Real.sqrt 2 * x₁ + m^2 = 0 ∧
             (1/2) * m * x₂^2 + Real.sqrt 2 * x₂ + m^2 = 0) →
  (m = 1 ∧ x₁ = -Real.sqrt 2 ∧ x₂ = -Real.sqrt 2) ∨
  (m = -2 ∧ x₁ = -Real.sqrt 2 ∧ x₂ = 2 * Real.sqrt 2) :=
by sorry

end quadratic_equation_roots_l901_90160


namespace virgo_island_trip_duration_l901_90168

/-- The duration of Tom's trip to "Virgo" island -/
theorem virgo_island_trip_duration :
  ∀ (boat_duration : ℝ) (plane_duration : ℝ),
    boat_duration = 2 →
    plane_duration = 4 * boat_duration →
    boat_duration + plane_duration = 10 :=
by
  sorry

end virgo_island_trip_duration_l901_90168


namespace functions_properties_l901_90181

/-- Given functions f and g with parameter a, prove monotonicity of g and range of b -/
theorem functions_properties (a : ℝ) (h : a < -1) :
  let f := fun (x : ℝ) ↦ x^3 / 3 - x^2 / 2 + a^2 / 2 - 1 / 3
  let g := fun (x : ℝ) ↦ a * Real.log (x + 1) - x^2 / 2 - a * x
  let g_deriv := fun (x : ℝ) ↦ a / (x + 1) - x - a
  let monotonic_intervals := (Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (-a - 1), Set.Ioo 0 (-a - 1))
  let b := g (-a - 1) - f 1
  (∀ x ∈ monotonic_intervals.1, g_deriv x < 0) ∧
  (∀ x ∈ monotonic_intervals.2, g_deriv x > 0) ∧
  (∀ y : ℝ, y < 0 → ∃ x : ℝ, b = y) := by
  sorry


end functions_properties_l901_90181


namespace inverse_inequality_l901_90106

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end inverse_inequality_l901_90106


namespace tan_2x_value_l901_90155

theorem tan_2x_value (f : ℝ → ℝ) (x : ℝ) :
  f x = Real.sin x + Real.cos x →
  (deriv f) x = 3 * f x →
  Real.tan (2 * x) = -4/3 := by
sorry

end tan_2x_value_l901_90155


namespace inequalities_proof_l901_90197

theorem inequalities_proof (a b : ℝ) (h1 : a > 0) (h2 : 0 > b) (h3 : a + b > 0) :
  (a^2 > b^2) ∧ (1/a > 1/b) ∧ (a^2*b < b^3) ∧ ¬(∀ a b, a > 0 → 0 > b → a + b > 0 → a^3 < a*b^2) := by
  sorry

end inequalities_proof_l901_90197


namespace distance_between_points_l901_90146

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (7, -2)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 * Real.sqrt 2 := by
sorry

end distance_between_points_l901_90146


namespace fraction_sum_minus_eight_l901_90186

theorem fraction_sum_minus_eight : 
  (7 : ℚ) / 3 + 11 / 5 + 19 / 9 + 37 / 17 - 8 = 628 / 765 := by
  sorry

end fraction_sum_minus_eight_l901_90186


namespace smallest_n_for_sqrt_difference_smallest_n_is_626_l901_90180

theorem smallest_n_for_sqrt_difference (n : ℕ) : n ≥ 626 ↔ Real.sqrt n - Real.sqrt (n - 1) < 0.02 := by
  sorry

theorem smallest_n_is_626 : ∀ k : ℕ, k < 626 → Real.sqrt k - Real.sqrt (k - 1) ≥ 0.02 := by
  sorry

end smallest_n_for_sqrt_difference_smallest_n_is_626_l901_90180


namespace special_function_at_five_l901_90195

/-- A function satisfying f(x - y) = f(x) + f(y) for all real x and y, and f(0) = 2 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x - y) = f x + f y) ∧ (f 0 = 2)

/-- Theorem: For any function satisfying the special_function property, f(5) = 1 -/
theorem special_function_at_five (f : ℝ → ℝ) (h : special_function f) : f 5 = 1 := by
  sorry

end special_function_at_five_l901_90195


namespace primer_cost_calculation_l901_90189

/-- Represents the cost of primer per gallon before discount -/
def primer_cost : ℝ := 30

/-- Number of rooms to be painted and primed -/
def num_rooms : ℕ := 5

/-- Cost of paint per gallon -/
def paint_cost : ℝ := 25

/-- Discount rate on primer -/
def primer_discount : ℝ := 0.2

/-- Total amount spent on paint and primer -/
def total_spent : ℝ := 245

theorem primer_cost_calculation : 
  (num_rooms : ℝ) * paint_cost + 
  (num_rooms : ℝ) * primer_cost * (1 - primer_discount) = total_spent :=
by sorry

end primer_cost_calculation_l901_90189


namespace squash_players_l901_90153

/-- Given a class of children with information about their sport participation,
    calculate the number of children who play squash. -/
theorem squash_players (total : ℕ) (tennis : ℕ) (neither : ℕ) (both : ℕ) :
  total = 38 →
  tennis = 19 →
  neither = 10 →
  both = 12 →
  ∃ (squash : ℕ), squash = 21 ∧ 
    squash = total - neither - (tennis - both) := by
  sorry

#check squash_players

end squash_players_l901_90153


namespace sqrt_meaningful_condition_l901_90171

theorem sqrt_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 2) ↔ x ≥ -2 :=
by sorry

end sqrt_meaningful_condition_l901_90171


namespace product_of_roots_l901_90144

theorem product_of_roots (a b c : ℂ) : 
  (3 * a^3 - 9 * a^2 + 5 * a - 15 = 0) →
  (3 * b^3 - 9 * b^2 + 5 * b - 15 = 0) →
  (3 * c^3 - 9 * c^2 + 5 * c - 15 = 0) →
  a * b * c = 5 := by sorry

end product_of_roots_l901_90144


namespace sum_of_digits_2000_l901_90184

/-- The number of digits in a positive integer n -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the number of digits in 2^2000 and 5^2000 is 2001 -/
theorem sum_of_digits_2000 : num_digits (2^2000) + num_digits (5^2000) = 2001 := by sorry

end sum_of_digits_2000_l901_90184


namespace matchstick_sequence_l901_90108

/-- 
Given a sequence where:
- The first term is 4
- Each subsequent term increases by 3
This theorem proves that the 20th term of the sequence is 61.
-/
theorem matchstick_sequence (n : ℕ) : 
  let sequence : ℕ → ℕ := λ k => 4 + 3 * (k - 1)
  sequence 20 = 61 := by
  sorry

end matchstick_sequence_l901_90108


namespace two_statements_true_l901_90166

open Real

-- Define the function f
noncomputable def f (x : ℝ) := 2 * sin x * cos (abs x)

-- Define the sequence a_n
def a (n : ℕ) (k : ℝ) := n^2 + k*n + 2

theorem two_statements_true :
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∃ w : ℝ, w > 0 ∧ w = 1 ∧ ∀ x, f (x + w) = f x) ∧
  (∀ k, (∀ n : ℕ, n > 0 → a (n+1) k > a n k) → k > -3) :=
by sorry

end two_statements_true_l901_90166


namespace cone_radii_sum_l901_90135

/-- Given a circle with radius 5 divided into three sectors with area ratios 1:2:3,
    used as lateral surfaces of three cones with base radii r₁, r₂, and r₃ respectively,
    prove that r₁ + r₂ + r₃ = 5. -/
theorem cone_radii_sum (r₁ r₂ r₃ : ℝ) : 
  (2 * π * r₁ = (1 / 6) * 2 * π * 5) → 
  (2 * π * r₂ = (2 / 6) * 2 * π * 5) → 
  (2 * π * r₃ = (3 / 6) * 2 * π * 5) → 
  r₁ + r₂ + r₃ = 5 := by
  sorry

end cone_radii_sum_l901_90135


namespace amys_chicken_soup_cans_l901_90194

/-- Amy's soup purchase problem -/
theorem amys_chicken_soup_cans (total_soups : ℕ) (tomato_soup_cans : ℕ) (chicken_soup_cans : ℕ) :
  total_soups = 9 →
  tomato_soup_cans = 3 →
  total_soups = tomato_soup_cans + chicken_soup_cans →
  chicken_soup_cans = 6 := by
  sorry

end amys_chicken_soup_cans_l901_90194


namespace find_number_l901_90188

theorem find_number : ∃ n : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := 2 * diff
  n = sum * quotient + 20 ∧ n / sum = quotient ∧ n % sum = 20 := by
  sorry

end find_number_l901_90188


namespace arithmetic_sequence_proof_l901_90148

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_proof (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h1 : ∀ n : ℕ, 2 * S n = n * a n)
    (h2 : a 2 = 1) :
  (∀ n : ℕ, n ≥ 1 → a n = n - 1) ∧ is_arithmetic_sequence a :=
by sorry

end arithmetic_sequence_proof_l901_90148


namespace soccer_team_selection_l901_90170

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def players_to_select : ℕ := 7
def max_quadruplets : ℕ := 2

theorem soccer_team_selection :
  (Nat.choose total_players players_to_select) -
  ((Nat.choose quadruplets 3) * (Nat.choose (total_players - quadruplets) (players_to_select - 3)) +
   (Nat.choose quadruplets 4) * (Nat.choose (total_players - quadruplets) (players_to_select - 4))) = 9240 :=
by sorry

end soccer_team_selection_l901_90170


namespace volume_ratio_theorem_l901_90179

/-- A right rectangular prism with edge lengths -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The set of points within distance r from any point in the prism -/
def S (B : RectangularPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The volume of S(r) -/
def volume_S (B : RectangularPrism) (r : ℝ) : ℝ :=
  sorry

/-- Coefficients of the volume polynomial -/
structure VolumeCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  all_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d

theorem volume_ratio_theorem (B : RectangularPrism) (coeff : VolumeCoefficients) :
    B.length = 2 ∧ B.width = 4 ∧ B.height = 6 →
    (∀ r : ℝ, volume_S B r = coeff.a * r^3 + coeff.b * r^2 + coeff.c * r + coeff.d) →
    coeff.b * coeff.c / (coeff.a * coeff.d) = 66 := by
  sorry

end volume_ratio_theorem_l901_90179


namespace birds_reduced_correct_l901_90117

/-- The number of birds reduced on the third day, given the initial number of birds,
    the doubling on the second day, and the total number of birds seen in three days. -/
def birds_reduced (initial : ℕ) (total : ℕ) : ℕ :=
  initial * 2 - (total - (initial + initial * 2))

/-- Theorem stating that the number of birds reduced on the third day is 200,
    given the conditions from the problem. -/
theorem birds_reduced_correct : birds_reduced 300 1300 = 200 := by
  sorry

end birds_reduced_correct_l901_90117


namespace complementary_event_is_at_most_one_wins_l901_90129

-- Define the sample space
inductive Outcome
  | BothWin
  | AWinsBLoses
  | ALosesBWins
  | BothLose

-- Define the event A
def eventA (outcome : Outcome) : Prop :=
  outcome = Outcome.BothWin

-- Define the complementary event
def complementaryEventA (outcome : Outcome) : Prop :=
  outcome = Outcome.AWinsBLoses ∨ outcome = Outcome.ALosesBWins ∨ outcome = Outcome.BothLose

-- Theorem statement
theorem complementary_event_is_at_most_one_wins :
  ∀ (outcome : Outcome), ¬(eventA outcome) ↔ complementaryEventA outcome :=
sorry

end complementary_event_is_at_most_one_wins_l901_90129


namespace augmented_matrix_problem_l901_90175

/-- Given a system of linear equations with augmented matrix
    ⎛ 3 2 1 ⎞
    ⎝ 1 1 m ⎠
    where Dx = 5, prove that m = -2 -/
theorem augmented_matrix_problem (m : ℝ) : 
  let A : Matrix (Fin 2) (Fin 3) ℝ := ![![3, 2, 1], ![1, 1, m]]
  let Dx := (A 0 2 * A 1 1 - A 0 1 * A 1 2) / (A 0 0 * A 1 1 - A 0 1 * A 1 0)
  Dx = 5 → m = -2 := by
  sorry


end augmented_matrix_problem_l901_90175


namespace log_seven_eighteen_l901_90133

theorem log_seven_eighteen (a b : ℝ) 
  (h1 : Real.log 2 / Real.log 10 = a) 
  (h2 : Real.log 3 / Real.log 10 = b) : 
  Real.log 18 / Real.log 7 = (2*a + 4*b) / (1 + 2*a) := by
  sorry

end log_seven_eighteen_l901_90133


namespace cost_39_roses_l901_90156

/-- Represents the cost of a bouquet of roses -/
def bouquet_cost (roses : ℕ) : ℚ :=
  sorry

/-- The price of a bouquet is directly proportional to the number of roses -/
axiom price_proportional (r₁ r₂ : ℕ) : 
  bouquet_cost r₁ / bouquet_cost r₂ = r₁ / r₂

/-- A bouquet of 12 roses costs $20 -/
axiom dozen_cost : bouquet_cost 12 = 20

theorem cost_39_roses : bouquet_cost 39 = 65 := by
  sorry

end cost_39_roses_l901_90156


namespace triangle_gp_ratio_lt_two_l901_90136

/-- Given a triangle with side lengths forming a geometric progression,
    prove that the common ratio of the progression is less than 2. -/
theorem triangle_gp_ratio_lt_two (b q : ℝ) (hb : b > 0) (hq : q > 0) :
  (b + b*q > b*q^2) ∧ (b + b*q^2 > b*q) ∧ (b*q + b*q^2 > b) →
  q < 2 := by
  sorry

end triangle_gp_ratio_lt_two_l901_90136


namespace initial_milk_water_ratio_l901_90183

/-- Given a mixture of milk and water, proves that the initial ratio was 4:1 --/
theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (final_ratio : ℝ) :
  total_volume = 45 →
  added_water = 3 →
  final_ratio = 3 →
  ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = total_volume ∧
    initial_milk / (initial_water + added_water) = final_ratio ∧
    initial_milk / initial_water = 4 := by
  sorry

end initial_milk_water_ratio_l901_90183


namespace min_value_theorem_l901_90126

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 1) :
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) / 2 ∧
  ∀ (z : ℝ), z > 0 → z + y = 1 → (2 / (z + 3 * y) + 1 / (z - y)) ≥ min_val :=
by sorry

end min_value_theorem_l901_90126


namespace pet_store_dogs_l901_90112

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
def calculate_dogs (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) : ℕ :=
  (num_cats / cat_ratio) * dog_ratio

/-- Theorem: Given the ratio of cats to dogs is 3:5 and there are 18 cats, prove there are 30 dogs -/
theorem pet_store_dogs :
  let cat_ratio : ℕ := 3
  let dog_ratio : ℕ := 5
  let num_cats : ℕ := 18
  calculate_dogs cat_ratio dog_ratio num_cats = 30 := by
  sorry

#eval calculate_dogs 3 5 18

end pet_store_dogs_l901_90112


namespace range_of_a_l901_90105

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x + 1| > 2) → a ∈ Set.Iio (-3) ∪ Set.Ioi 1 := by
  sorry

end range_of_a_l901_90105


namespace library_books_count_library_books_count_proof_l901_90163

theorem library_books_count : ℕ → Prop :=
  fun N =>
    let initial_issued := N / 17
    let transferred := 2000
    let new_issued := initial_issued + transferred
    (initial_issued = (N - initial_issued) / 16) ∧
    (new_issued = (N - new_issued) / 15) →
    N = 544000

-- The proof goes here
theorem library_books_count_proof : library_books_count 544000 := by
  sorry

end library_books_count_library_books_count_proof_l901_90163


namespace octal_to_decimal_l901_90113

theorem octal_to_decimal (n : ℕ) (h : n = 246) : 
  2 * 8^2 + 4 * 8^1 + 6 * 8^0 = 166 := by
  sorry

end octal_to_decimal_l901_90113


namespace find_number_l901_90110

/-- Given the equation (47% of 1442 - 36% of N) + 66 = 6, prove that N = 2049.28 --/
theorem find_number (N : ℝ) : (0.47 * 1442 - 0.36 * N) + 66 = 6 → N = 2049.28 := by
  sorry

end find_number_l901_90110


namespace quadratic_inequality_solution_l901_90152

theorem quadratic_inequality_solution (c : ℝ) :
  (∀ x : ℝ, x^2 + 5*x - 2*c ≤ 0 ↔ -6 ≤ x ∧ x ≤ 1) :=
by sorry

end quadratic_inequality_solution_l901_90152


namespace annies_final_crayons_l901_90115

/-- The number of crayons Annie has at the end, given the initial conditions. -/
def anniesCrayons : ℕ :=
  let initialCrayons : ℕ := 4
  let samsCrayons : ℕ := 36
  let matthewsCrayons : ℕ := 5 * samsCrayons
  initialCrayons + samsCrayons + matthewsCrayons

/-- Theorem stating that Annie will have 220 crayons at the end. -/
theorem annies_final_crayons : anniesCrayons = 220 := by
  sorry

end annies_final_crayons_l901_90115


namespace fraction_comparison_l901_90114

theorem fraction_comparison : 
  (14 / 10 : ℚ) = 7 / 5 ∧ 
  (1 + 2 / 5 : ℚ) = 7 / 5 ∧ 
  (1 + 4 / 20 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 2 / 6 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 28 / 20 : ℚ) ≠ 7 / 5 :=
by sorry

end fraction_comparison_l901_90114


namespace speedster_fraction_l901_90121

/-- Represents the inventory of an automobile company -/
structure Inventory where
  total : ℕ
  speedsters : ℕ
  convertibles : ℕ
  non_speedsters : ℕ

/-- Conditions for the inventory -/
def inventory_conditions (inv : Inventory) : Prop :=
  inv.convertibles = (4 * inv.speedsters) / 5 ∧
  inv.non_speedsters = 60 ∧
  inv.convertibles = 96 ∧
  inv.total = inv.speedsters + inv.non_speedsters

/-- Theorem: The fraction of Speedsters in the inventory is 2/3 -/
theorem speedster_fraction (inv : Inventory) 
  (h : inventory_conditions inv) : 
  (inv.speedsters : ℚ) / inv.total = 2 / 3 := by
  sorry

end speedster_fraction_l901_90121


namespace polynomial_factorization_l901_90172

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 5*x^4 + 10*x^2 - 4 = (x - 1) * (x + 1) * (x^2 - 2)^2 := by
sorry

end polynomial_factorization_l901_90172


namespace count_six_digit_integers_l901_90103

def digit_set : Multiset ℕ := {1, 1, 2, 3, 3, 3}

/-- The number of different positive, six-digit integers formed from the given digit set -/
def num_six_digit_integers : ℕ := sorry

theorem count_six_digit_integers : num_six_digit_integers = 60 := by sorry

end count_six_digit_integers_l901_90103


namespace sock_drawing_probability_l901_90191

theorem sock_drawing_probability : 
  ∀ (total_socks : ℕ) (colors : ℕ) (socks_per_color : ℕ) (drawn_socks : ℕ),
    total_socks = colors * socks_per_color →
    total_socks = 10 →
    colors = 5 →
    socks_per_color = 2 →
    drawn_socks = 5 →
    (Nat.choose total_socks drawn_socks : ℚ) ≠ 0 →
    (Nat.choose colors 4 * Nat.choose 4 1 * (socks_per_color ^ 3) : ℚ) / 
    (Nat.choose total_socks drawn_socks : ℚ) = 20 / 21 := by
  sorry

end sock_drawing_probability_l901_90191


namespace remainder_problem_l901_90147

theorem remainder_problem (n : ℕ) (h1 : n = 349) (h2 : n % 17 = 9) : n % 13 = 11 := by
  sorry

end remainder_problem_l901_90147


namespace total_snowman_drawings_l901_90161

/-- The number of cards Melody made -/
def num_cards : ℕ := 13

/-- The number of snowman drawings on each card -/
def drawings_per_card : ℕ := 4

/-- The total number of snowman drawings printed -/
def total_drawings : ℕ := num_cards * drawings_per_card

theorem total_snowman_drawings : total_drawings = 52 := by
  sorry

end total_snowman_drawings_l901_90161


namespace janes_reading_speed_l901_90130

theorem janes_reading_speed (total_pages : ℕ) (first_half_speed : ℕ) (total_days : ℕ) 
  (h1 : total_pages = 500)
  (h2 : first_half_speed = 10)
  (h3 : total_days = 75) :
  (total_pages / 2) / (total_days - (total_pages / 2) / first_half_speed) = 5 :=
by
  sorry

end janes_reading_speed_l901_90130


namespace sugar_theorem_l901_90187

def sugar_problem (initial : ℝ) (day1_use day1_borrow : ℝ)
  (day2_buy day2_use day2_receive : ℝ)
  (day3_buy day3_use day3_return day3_borrow : ℝ)
  (day4_use day4_receive : ℝ)
  (day5_use day5_borrow day5_return : ℝ) : Prop :=
  let day1 := initial - day1_use - day1_borrow
  let day2 := day1 + day2_buy - day2_use + day2_receive
  let day3 := day2 + day3_buy - day3_use + day3_return - day3_borrow
  let day4 := day3 - day4_use + day4_receive
  let day5 := day4 - day5_use - day5_borrow + day5_return
  day5 = 63.3

theorem sugar_theorem : sugar_problem 65 18.5 5.3 30.2 12.7 4.75 20.5 8.25 2.8 1.2 9.5 6.35 10.75 3.1 3 := by
  sorry

end sugar_theorem_l901_90187


namespace identity_is_unique_solution_l901_90134

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (x + f (y + x*y)) = (y + 1) * f (x + 1) - 1

/-- The main theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x > 0) →
  SatisfiesEquation f →
  ∀ x, x > 0 → f x = x :=
sorry

end identity_is_unique_solution_l901_90134


namespace total_toys_count_l901_90185

-- Define the number of toys for each child
def jaxon_toys : ℝ := 15
def gabriel_toys : ℝ := 2.5 * jaxon_toys
def jerry_toys : ℝ := gabriel_toys + 8.5
def sarah_toys : ℝ := jerry_toys - 5.5
def emily_toys : ℝ := 1.5 * gabriel_toys

-- Define the total number of toys
def total_toys : ℝ := jerry_toys + gabriel_toys + jaxon_toys + sarah_toys + emily_toys

-- Theorem to prove
theorem total_toys_count : total_toys = 195.25 := by
  sorry

end total_toys_count_l901_90185


namespace course_length_is_300_l901_90111

/-- Represents the dogsled race scenario -/
structure DogsledRace where
  teamT_speed : ℝ
  teamA_speed_diff : ℝ
  teamT_time : ℝ
  teamA_time_diff : ℝ

/-- Calculates the length of the dogsled race course -/
def course_length (race : DogsledRace) : ℝ :=
  race.teamT_speed * race.teamT_time

/-- Theorem stating that the course length is 300 miles given the race conditions -/
theorem course_length_is_300 (race : DogsledRace)
  (h1 : race.teamT_speed = 20)
  (h2 : race.teamA_speed_diff = 5)
  (h3 : race.teamA_time_diff = 3)
  (h4 : race.teamT_time * race.teamT_speed = (race.teamT_time - race.teamA_time_diff) * (race.teamT_speed + race.teamA_speed_diff)) :
  course_length race = 300 := by
  sorry

#eval course_length { teamT_speed := 20, teamA_speed_diff := 5, teamT_time := 15, teamA_time_diff := 3 }

end course_length_is_300_l901_90111


namespace sin_120_degrees_l901_90182

theorem sin_120_degrees (π : Real) :
  Real.sin (2 * π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_120_degrees_l901_90182


namespace circle_area_difference_l901_90162

theorem circle_area_difference (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 21) (h₂ : r₂ = 31) :
  (r₃ ^ 2 = r₂ ^ 2 - r₁ ^ 2) → r₃ = 2 * Real.sqrt 130 := by
  sorry

end circle_area_difference_l901_90162


namespace evaluate_expression_l901_90196

theorem evaluate_expression (a b : ℚ) (h1 : a = 5) (h2 : b = -3) : 3 / (a + b) = 3 / 2 := by
  sorry

end evaluate_expression_l901_90196


namespace joes_journey_time_l901_90151

/-- Represents Joe's journey from home to school with a detour -/
def joes_journey (d : ℝ) : Prop :=
  let walking_speed : ℝ := d / 3 / 9  -- Speed to walk 1/3 of d in 9 minutes
  let running_speed : ℝ := 4 * walking_speed
  let total_walking_distance : ℝ := 2 * d / 3
  let total_running_distance : ℝ := 2 * d / 3
  let total_walking_time : ℝ := total_walking_distance / walking_speed
  let total_running_time : ℝ := total_running_distance / running_speed
  total_walking_time + total_running_time = 40.5

/-- Theorem stating that Joe's journey takes 40.5 minutes -/
theorem joes_journey_time :
  ∃ d : ℝ, d > 0 ∧ joes_journey d :=
sorry

end joes_journey_time_l901_90151


namespace total_stairs_climbed_l901_90109

def samir_stairs : ℕ := 318

def veronica_stairs : ℕ := samir_stairs / 2 + 18

theorem total_stairs_climbed : samir_stairs + veronica_stairs = 495 := by
  sorry

end total_stairs_climbed_l901_90109


namespace choose_four_from_ten_l901_90199

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by sorry

end choose_four_from_ten_l901_90199


namespace triangle_area_l901_90177

theorem triangle_area (a b c : ℝ) (ha : a = 9) (hb : b = 40) (hc : c = 41) :
  (1/2) * a * b = 180 :=
by
  sorry

end triangle_area_l901_90177


namespace new_student_weight_is_62_l901_90127

/-- The weight of the new student given the conditions of the problem -/
def new_student_weight (n : ℕ) (avg_decrease : ℚ) (old_student_weight : ℚ) : ℚ :=
  old_student_weight - n * avg_decrease

/-- Theorem stating that the weight of the new student is 62 kg -/
theorem new_student_weight_is_62 :
  new_student_weight 6 3 80 = 62 := by
  sorry

end new_student_weight_is_62_l901_90127


namespace parabola_intersection_ratio_l901_90192

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (c : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * c.p * x

/-- Line passing through the focus of a parabola with slope angle 60° -/
structure FocusLine (c : Parabola) where
  slope : ℝ
  h_slope : slope = Real.sqrt 3
  focus_x : ℝ
  h_focus_x : focus_x = c.p / 2

/-- Theorem stating the ratio of AB to AP is 7/12 -/
theorem parabola_intersection_ratio (c : Parabola) (l : FocusLine c) 
  (A B : ParabolaPoint c) (P : ℝ × ℝ) :
  A.x > 0 → A.y > 0 →  -- A in first quadrant
  B.x > 0 → B.y < 0 →  -- B in fourth quadrant
  P.1 = 0 →  -- P on y-axis
  (A.y - l.focus_x) = l.slope * (A.x - l.focus_x) →  -- A on line l
  (B.y - l.focus_x) = l.slope * (B.x - l.focus_x) →  -- B on line l
  (P.2 - l.focus_x) = l.slope * (P.1 - l.focus_x) →  -- P on line l
  abs (A.x - B.x) / abs (A.x - P.1) = 7 / 12 := by
    sorry

end parabola_intersection_ratio_l901_90192


namespace correct_amount_given_to_john_l901_90167

/-- The amount given to John after one month -/
def amount_given_to_john (held_commission : ℕ) (advance_fees : ℕ) (incentive : ℕ) : ℕ :=
  (held_commission - advance_fees) + incentive

/-- Theorem stating the correct amount given to John -/
theorem correct_amount_given_to_john :
  amount_given_to_john 25000 8280 1780 = 18500 := by
  sorry

end correct_amount_given_to_john_l901_90167


namespace largest_common_divisor_of_consecutive_odd_product_l901_90159

theorem largest_common_divisor_of_consecutive_odd_product (n : ℕ) (h : Odd n) :
  (∃ (k : ℕ), k > 15 ∧ ∀ (m : ℕ), Odd m → k ∣ (m * (m + 2) * (m + 4) * (m + 6) * (m + 8))) → False :=
sorry

end largest_common_divisor_of_consecutive_odd_product_l901_90159


namespace binomial_permutation_equality_l901_90123

theorem binomial_permutation_equality (n : ℕ+) :
  3 * (Nat.choose (n.val - 1) (n.val - 5)) = 5 * (Nat.factorial (n.val - 2) / Nat.factorial (n.val - 4)) →
  n.val = 9 := by
  sorry

end binomial_permutation_equality_l901_90123


namespace f_derivative_l901_90104

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem f_derivative :
  deriv f = λ x => -2 * Real.sin (2 * x) := by sorry

end f_derivative_l901_90104


namespace kendall_driving_distance_l901_90138

theorem kendall_driving_distance (mother_distance father_distance : ℝ) 
  (h1 : mother_distance = 0.17)
  (h2 : father_distance = 0.5) :
  mother_distance + father_distance = 0.67 := by
  sorry

end kendall_driving_distance_l901_90138


namespace min_value_x_l901_90154

theorem min_value_x (x : ℝ) : 2 * (x + 1) ≥ x + 1 → x ≥ -1 ∧ ∀ y, (∀ z, 2 * (z + 1) ≥ z + 1 → z ≥ y) → y ≤ -1 := by
  sorry

end min_value_x_l901_90154


namespace sum_of_absolute_coefficients_l901_90165

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 4^7 := by
sorry

end sum_of_absolute_coefficients_l901_90165
