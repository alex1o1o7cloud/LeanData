import Mathlib

namespace NUMINAMATH_CALUDE_scoops_per_carton_is_ten_l3820_382049

/-- Represents the number of scoops in each carton of ice cream -/
def scoops_per_carton : ℕ := sorry

/-- The total number of cartons -/
def total_cartons : ℕ := 3

/-- The number of scoops Ethan wants -/
def ethan_scoops : ℕ := 2

/-- The number of people who want 2 scoops of chocolate -/
def chocolate_lovers : ℕ := 3

/-- The number of scoops Olivia wants -/
def olivia_scoops : ℕ := 2

/-- The number of scoops Shannon wants (twice as much as Olivia) -/
def shannon_scoops : ℕ := 2 * olivia_scoops

/-- The number of scoops left after everyone has taken their scoops -/
def scoops_left : ℕ := 16

/-- The total number of scoops taken -/
def total_scoops_taken : ℕ := 
  ethan_scoops + (chocolate_lovers * 2) + olivia_scoops + shannon_scoops

/-- Theorem stating that the number of scoops per carton is 10 -/
theorem scoops_per_carton_is_ten : scoops_per_carton = 10 := by
  sorry

end NUMINAMATH_CALUDE_scoops_per_carton_is_ten_l3820_382049


namespace NUMINAMATH_CALUDE_amy_muffins_l3820_382005

def muffins_series (n : ℕ) : ℕ := n * (n + 1) / 2

theorem amy_muffins :
  let days : ℕ := 5
  let start_muffins : ℕ := 1
  let leftover_muffins : ℕ := 7
  let total_brought := muffins_series days
  total_brought + leftover_muffins = 22 :=
by sorry

end NUMINAMATH_CALUDE_amy_muffins_l3820_382005


namespace NUMINAMATH_CALUDE_rectangle_area_l3820_382070

theorem rectangle_area (L W : ℝ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) :
  L * W = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3820_382070


namespace NUMINAMATH_CALUDE_sunday_occurs_five_times_in_january_l3820_382083

/-- Represents the days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a month -/
structure Month where
  days : ℕ
  first_day : DayOfWeek

/-- December of year M -/
def december : Month := {
  days := 31,
  first_day := DayOfWeek.Thursday  -- This is arbitrary, as we don't know the exact first day
}

/-- January of year M+1 -/
def january : Month := {
  days := 31,
  first_day := sorry  -- We don't know the exact first day, it depends on December
}

/-- Count occurrences of a specific day in a month -/
def count_day_occurrences (m : Month) (d : DayOfWeek) : ℕ := sorry

/-- The main theorem to prove -/
theorem sunday_occurs_five_times_in_january :
  (count_day_occurrences december DayOfWeek.Thursday = 5) →
  (count_day_occurrences january DayOfWeek.Sunday = 5) :=
sorry

end NUMINAMATH_CALUDE_sunday_occurs_five_times_in_january_l3820_382083


namespace NUMINAMATH_CALUDE_multichoose_eq_choose_l3820_382051

/-- F_n^r represents the number of ways to choose r elements from [1, n] with repetition and disregarding order -/
def F (n : ℕ) (r : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to choose r elements from [1, n] with repetition and disregarding order
    is equal to the number of ways to choose r elements from [1, n+r-1] without repetition -/
theorem multichoose_eq_choose (n : ℕ) (r : ℕ) : F n r = Nat.choose (n + r - 1) r := by sorry

end NUMINAMATH_CALUDE_multichoose_eq_choose_l3820_382051


namespace NUMINAMATH_CALUDE_parabola_point_x_coord_l3820_382065

/-- A point on a parabola with a specific distance to its focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x
  distance_to_focus : (x - 1)^2 + y^2 = 25

/-- The x-coordinate of a point on a parabola with distance 5 to its focus is 4 -/
theorem parabola_point_x_coord (M : ParabolaPoint) : M.x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_x_coord_l3820_382065


namespace NUMINAMATH_CALUDE_x_with_three_prime_divisors_l3820_382052

theorem x_with_three_prime_divisors (x n : ℕ) 
  (h1 : x = 2^n - 32)
  (h2 : (Nat.factors x).toFinset.card = 3)
  (h3 : 2 ∈ Nat.factors x) :
  x = 2016 ∨ x = 16352 := by
  sorry

end NUMINAMATH_CALUDE_x_with_three_prime_divisors_l3820_382052


namespace NUMINAMATH_CALUDE_three_fifths_of_twelve_times_ten_minus_twenty_l3820_382035

theorem three_fifths_of_twelve_times_ten_minus_twenty : 
  (3 : ℚ) / 5 * ((12 * 10) - 20) = 60 := by
  sorry

end NUMINAMATH_CALUDE_three_fifths_of_twelve_times_ten_minus_twenty_l3820_382035


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l3820_382071

/-- An arithmetic progression with sum of first n terms S_n -/
structure ArithmeticProgression where
  S : ℕ → ℚ
  sum_formula : ∀ n : ℕ, S n = n / 2 * (2 * a + (n - 1) * d)
  a : ℚ
  d : ℚ

/-- Theorem stating that if S_3 = 2 and S_6 = 6, then S_24 = 510 for an arithmetic progression -/
theorem arithmetic_progression_sum (ap : ArithmeticProgression) 
  (h1 : ap.S 3 = 2) (h2 : ap.S 6 = 6) : ap.S 24 = 510 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l3820_382071


namespace NUMINAMATH_CALUDE_quadrilateral_area_l3820_382074

/-- The area of a quadrilateral with given diagonal and offsets -/
theorem quadrilateral_area (diagonal : ℝ) (offset1 offset2 : ℝ) :
  diagonal = 10 → offset1 = 7 → offset2 = 3 →
  (diagonal * offset1 / 2) + (diagonal * offset2 / 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l3820_382074


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3820_382020

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 2, 3}

-- Define set B
def B : Finset Nat := {2, 5}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3820_382020


namespace NUMINAMATH_CALUDE_problem_statement_l3820_382046

theorem problem_statement (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a + b + c = 0) : 
  c * b^2 ≤ a * b^2 ∧ a * b > a * c := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3820_382046


namespace NUMINAMATH_CALUDE_geometry_theorems_l3820_382063

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Axioms for the properties of parallel and perpendicular
axiom parallel_planes_transitive : 
  ∀ (α β γ : Plane), parallel_planes α β → parallel_planes α γ → parallel_planes β γ

axiom perpendicular_parallel_planes : 
  ∀ (m : Line) (α β : Plane), 
    perpendicular_line_plane m α → parallel_line_plane m β → perpendicular_planes α β

-- Theorem to prove
theorem geometry_theorems :
  (∀ (α β γ : Plane), parallel_planes α β → parallel_planes α γ → parallel_planes β γ) ∧
  (∀ (m : Line) (α β : Plane), 
    perpendicular_line_plane m α → parallel_line_plane m β → perpendicular_planes α β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_theorems_l3820_382063


namespace NUMINAMATH_CALUDE_square_of_real_not_always_positive_l3820_382021

theorem square_of_real_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_of_real_not_always_positive_l3820_382021


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l3820_382036

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem f_derivative_at_one :
  (deriv f) 1 = 2 * Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l3820_382036


namespace NUMINAMATH_CALUDE_tenth_fib_is_55_l3820_382050

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- The 10th Fibonacci number is 55 -/
theorem tenth_fib_is_55 : fib 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_tenth_fib_is_55_l3820_382050


namespace NUMINAMATH_CALUDE_bob_guaranteed_victory_l3820_382099

/-- Represents a grid in the game -/
def Grid := Matrix (Fin 2011) (Fin 2011) ℕ

/-- The size of the grid -/
def gridSize : ℕ := 2011

/-- The total number of grids Alice has -/
def aliceGridCount : ℕ := 2010

/-- Checks if a grid is valid (strictly increasing across rows and down columns) -/
def isValidGrid (g : Grid) : Prop :=
  ∀ i j k, i < j → g i k < g j k ∧ g k i < g k j

/-- Checks if two grids are different -/
def areDifferentGrids (g1 g2 : Grid) : Prop :=
  ∃ i j, g1 i j ≠ g2 i j

/-- Checks if Bob wins against a given grid -/
def bobWins (bobGrid aliceGrid : Grid) : Prop :=
  ∃ i j k, aliceGrid i j = bobGrid k i ∧ aliceGrid i k = bobGrid k j

/-- Theorem: Bob can guarantee victory with at most 1 swap -/
theorem bob_guaranteed_victory :
  ∃ (initialBobGrid : Grid) (swappedBobGrid : Grid),
    isValidGrid initialBobGrid ∧
    isValidGrid swappedBobGrid ∧
    (∀ (aliceGrids : Fin aliceGridCount → Grid),
      (∀ i, isValidGrid (aliceGrids i)) →
      (∀ i j, i ≠ j → areDifferentGrids (aliceGrids i) (aliceGrids j)) →
      (bobWins initialBobGrid (aliceGrids i) ∨
       bobWins swappedBobGrid (aliceGrids i))) :=
sorry

end NUMINAMATH_CALUDE_bob_guaranteed_victory_l3820_382099


namespace NUMINAMATH_CALUDE_triangle_properties_l3820_382062

theorem triangle_properties (a b c A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  c * Real.sin C / Real.sin A - c = b * Real.sin B / Real.sin A - a →
  b = 2 →
  (B = π / 3 ∧
   (a = 2 * Real.sqrt 6 / 3 →
    1/2 * a * b * Real.sin C = 1 + Real.sqrt 3 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3820_382062


namespace NUMINAMATH_CALUDE_kerosene_cost_l3820_382026

/-- The cost of kerosene in a market with given price relationships -/
theorem kerosene_cost (rice_pound_cost : ℝ) (h1 : rice_pound_cost = 0.24) :
  let dozen_eggs_cost := rice_pound_cost
  let half_liter_kerosene_cost := dozen_eggs_cost / 2
  let liter_kerosene_cost := 2 * half_liter_kerosene_cost
  let cents_per_dollar := 100
  ⌊liter_kerosene_cost * cents_per_dollar⌋ = 24 := by sorry

end NUMINAMATH_CALUDE_kerosene_cost_l3820_382026


namespace NUMINAMATH_CALUDE_two_digit_square_with_square_digit_product_l3820_382045

/-- A function that returns true if a number is a perfect square --/
def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that returns the product of digits of a two-digit number --/
def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- The main theorem to be proved --/
theorem two_digit_square_with_square_digit_product : 
  ∃! n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ is_square n ∧ is_square (digit_product n) :=
sorry

end NUMINAMATH_CALUDE_two_digit_square_with_square_digit_product_l3820_382045


namespace NUMINAMATH_CALUDE_batsman_average_increase_l3820_382032

def average_increase (total_innings : ℕ) (final_average : ℚ) (last_score : ℕ) : ℚ :=
  final_average - (total_innings * final_average - last_score) / (total_innings - 1)

theorem batsman_average_increase :
  average_increase 17 39 87 = 3 := by sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l3820_382032


namespace NUMINAMATH_CALUDE_inverse_difference_l3820_382039

theorem inverse_difference (a : ℝ) (h : a + a⁻¹ = 6) : a - a⁻¹ = 4 * Real.sqrt 2 ∨ a - a⁻¹ = -4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_difference_l3820_382039


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3820_382022

theorem sufficient_not_necessary_condition : 
  (∀ x : ℝ, x > 3 → x^2 > 4) ∧ 
  (∃ x : ℝ, x^2 > 4 ∧ ¬(x > 3)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3820_382022


namespace NUMINAMATH_CALUDE_reappearance_line_is_lcm_l3820_382040

/-- The cycle length of the letter sequence -/
def letter_cycle_length : ℕ := 8

/-- The cycle length of the digit sequence -/
def digit_cycle_length : ℕ := 4

/-- The line number where the original sequences reappear -/
def reappearance_line : ℕ := 8

/-- Theorem stating that the reappearance line is the least common multiple of the cycle lengths -/
theorem reappearance_line_is_lcm :
  reappearance_line = Nat.lcm letter_cycle_length digit_cycle_length :=
by sorry

end NUMINAMATH_CALUDE_reappearance_line_is_lcm_l3820_382040


namespace NUMINAMATH_CALUDE_edda_magni_winning_strategy_l3820_382095

/-- Represents the hexagonal board game with n tiles on each side. -/
structure HexGame where
  n : ℕ
  n_gt_two : n > 2

/-- Represents a winning strategy for Edda and Magni. -/
def winning_strategy (game : HexGame) : Prop :=
  ∃ k : ℕ, k > 0 ∧ game.n = 3 * k + 1

/-- Theorem stating the condition for Edda and Magni to have a winning strategy. -/
theorem edda_magni_winning_strategy (game : HexGame) :
  winning_strategy game ↔ ∃ k : ℕ, k > 0 ∧ game.n = 3 * k + 1 :=
by sorry


end NUMINAMATH_CALUDE_edda_magni_winning_strategy_l3820_382095


namespace NUMINAMATH_CALUDE_largest_whole_number_times_eleven_less_than_150_l3820_382093

theorem largest_whole_number_times_eleven_less_than_150 :
  (∃ x : ℕ, x = 13 ∧ 11 * x < 150 ∧ ∀ y : ℕ, y > x → 11 * y ≥ 150) :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_times_eleven_less_than_150_l3820_382093


namespace NUMINAMATH_CALUDE_distribute_7_4_l3820_382024

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 104 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_7_4 : distribute 7 4 = 104 := by sorry

end NUMINAMATH_CALUDE_distribute_7_4_l3820_382024


namespace NUMINAMATH_CALUDE_probability_is_three_fifths_l3820_382034

-- Define the set S
def S : Finset ℤ := {-3, 0, 0, 4, 7, 8}

-- Define the function to check if a pair of integers has a product of 0
def productIsZero (x y : ℤ) : Bool :=
  x * y = 0

-- Define the probability calculation function
def probabilityOfZeroProduct (s : Finset ℤ) : ℚ :=
  let totalPairs := (s.card.choose 2 : ℚ)
  let zeroPairs := (s.filter (· = 0)).card * (s.filter (· ≠ 0)).card +
                   (if (s.filter (· = 0)).card ≥ 2 then 1 else 0)
  zeroPairs / totalPairs

-- State the theorem
theorem probability_is_three_fifths :
  probabilityOfZeroProduct S = 3/5 := by sorry

end NUMINAMATH_CALUDE_probability_is_three_fifths_l3820_382034


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3820_382090

-- Define the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the instantaneous velocity function
def v (t : ℝ) : ℝ := -1 + 2*t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3820_382090


namespace NUMINAMATH_CALUDE_regular_soda_count_l3820_382077

/-- The number of regular soda bottles in a grocery store -/
def regular_soda : ℕ := sorry

/-- The number of diet soda bottles in a grocery store -/
def diet_soda : ℕ := 40

/-- The total number of regular and diet soda bottles in a grocery store -/
def total_regular_and_diet : ℕ := 89

/-- Theorem stating that the number of regular soda bottles is 49 -/
theorem regular_soda_count : regular_soda = 49 := by
  sorry

end NUMINAMATH_CALUDE_regular_soda_count_l3820_382077


namespace NUMINAMATH_CALUDE_triangle_properties_l3820_382080

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 13 ∧ 
  t.a + t.c = 4 ∧
  Real.cos t.B / Real.cos t.C = -t.b / (2 * t.a + t.c)

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) : 
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2) * t.a * t.c * Real.sin t.B = (3/4) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3820_382080


namespace NUMINAMATH_CALUDE_investment_average_rate_l3820_382011

/-- Proves that given a total investment split between two schemes with different rates,
    if the annual returns from both parts are equal, then the average rate of interest
    on the total investment is as calculated. -/
theorem investment_average_rate
  (total_investment : ℝ)
  (rate1 rate2 : ℝ)
  (h_total : total_investment = 5000)
  (h_rates : rate1 = 0.03 ∧ rate2 = 0.05)
  (h_equal_returns : ∃ (x : ℝ), x ≥ 0 ∧ x ≤ total_investment ∧
    rate1 * (total_investment - x) = rate2 * x) :
  (rate1 * (total_investment - x) + rate2 * x) / total_investment = 0.0375 :=
sorry

end NUMINAMATH_CALUDE_investment_average_rate_l3820_382011


namespace NUMINAMATH_CALUDE_simplify_expression_l3820_382038

theorem simplify_expression (x : ℝ) : (5 - 4*x) - (7 + 5*x - x^2) = x^2 - 9*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3820_382038


namespace NUMINAMATH_CALUDE_valid_numbers_l3820_382061

def is_valid_number (n : Nat) : Prop :=
  (n ≥ 1000 ∧ n ≤ 9999) ∧  -- 4-digit number
  (n % 6 = 0) ∧ (n % 7 = 0) ∧ (n % 8 = 0) ∧  -- divisible by 6, 7, and 8
  (n % 4 ≠ 0) ∧ (n % 3 ≠ 0) ∧  -- not divisible by 4 or 3
  (n / 100 = 55) ∧  -- first two digits are 55
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 22) ∧  -- sum of digits is 22
  (∃ (a b : Nat), n = a * 1100 + b * 11)  -- two digits repeat twice

theorem valid_numbers : 
  ∀ n : Nat, is_valid_number n ↔ (n = 5566 ∨ n = 6655) := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_l3820_382061


namespace NUMINAMATH_CALUDE_carnival_ticket_cost_l3820_382019

/-- Calculate the total cost of carnival tickets --/
theorem carnival_ticket_cost (kids_ticket_price : ℚ) (kids_ticket_quantity : ℕ)
  (adult_ticket_price : ℚ) (adult_ticket_quantity : ℕ)
  (kids_tickets_bought : ℕ) (adult_tickets_bought : ℕ) :
  kids_ticket_price * (kids_tickets_bought / kids_ticket_quantity : ℚ) +
  adult_ticket_price * (adult_tickets_bought / adult_ticket_quantity : ℚ) = 9 :=
by
  sorry

#check carnival_ticket_cost (1/4) 4 (2/3) 3 12 9

end NUMINAMATH_CALUDE_carnival_ticket_cost_l3820_382019


namespace NUMINAMATH_CALUDE_f_negative_eight_equals_three_l3820_382013

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def has_period_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) = f x + 2 * f 2

theorem f_negative_eight_equals_three
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_period : has_period_property f)
  (h_f_zero : f 0 = 3) :
  f (-8) = 3 := by
sorry

end NUMINAMATH_CALUDE_f_negative_eight_equals_three_l3820_382013


namespace NUMINAMATH_CALUDE_min_sum_squares_l3820_382088

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + 3*z = 1) :
  ∃ (m : ℝ), m = (1:ℝ)/14 ∧ x^2 + y^2 + z^2 ≥ m ∧ 
  (x^2 + y^2 + z^2 = m ↔ x = (1:ℝ)/14 ∧ y = (1:ℝ)/7 ∧ z = (3:ℝ)/14) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3820_382088


namespace NUMINAMATH_CALUDE_directed_segment_length_equal_l3820_382089

-- Define a vector space
variable {V : Type*} [NormedAddCommGroup V]

-- Define two points in the vector space
variable (M N : V)

-- Define the directed line segment from M to N
def directed_segment (M N : V) : V := N - M

-- Theorem statement
theorem directed_segment_length_equal :
  ‖directed_segment M N‖ = ‖directed_segment N M‖ := by sorry

end NUMINAMATH_CALUDE_directed_segment_length_equal_l3820_382089


namespace NUMINAMATH_CALUDE_four_digit_equal_digits_l3820_382086

theorem four_digit_equal_digits (n : ℕ+) : 
  (∃ d : ℕ, d ∈ Finset.range 10 ∧ 12 * n.val^2 + 12 * n.val + 11 = d * 1111) → n = 21 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_equal_digits_l3820_382086


namespace NUMINAMATH_CALUDE_min_units_B_required_twenty_units_B_not_sufficient_l3820_382007

/-- Profit from selling one unit of model A (in thousand yuan) -/
def profit_A : ℝ := 3

/-- Profit from selling one unit of model B (in thousand yuan) -/
def profit_B : ℝ := 5

/-- Total number of units to be purchased -/
def total_units : ℕ := 30

/-- Minimum desired profit (in thousand yuan) -/
def min_profit : ℝ := 131

/-- Function to calculate the profit based on the number of model B units -/
def calculate_profit (units_B : ℕ) : ℝ :=
  profit_B * units_B + profit_A * (total_units - units_B)

/-- Theorem stating the minimum number of model B units required -/
theorem min_units_B_required :
  ∀ k : ℕ, k ≥ 21 → calculate_profit k ≥ min_profit :=
by sorry

/-- Theorem stating that 20 units of model B is not sufficient -/
theorem twenty_units_B_not_sufficient :
  calculate_profit 20 < min_profit :=
by sorry

end NUMINAMATH_CALUDE_min_units_B_required_twenty_units_B_not_sufficient_l3820_382007


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l3820_382068

/-- The capacity of a water tank in gallons. -/
def tank_capacity : ℝ := 72

/-- The difference in gallons between 40% full and 10% empty. -/
def difference : ℝ := 36

/-- Proves that the tank capacity is correct given the condition. -/
theorem tank_capacity_proof : 
  tank_capacity * 0.4 = tank_capacity * 0.9 - difference :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l3820_382068


namespace NUMINAMATH_CALUDE_partition_with_equal_product_l3820_382047

def numbers : List Nat := [2, 3, 12, 14, 15, 20, 21]

theorem partition_with_equal_product :
  ∃ (s₁ s₂ : List Nat),
    s₁ ∪ s₂ = numbers ∧
    s₁ ∩ s₂ = [] ∧
    s₁ ≠ [] ∧
    s₂ ≠ [] ∧
    (s₁.prod = 2520 ∧ s₂.prod = 2520) :=
  sorry

end NUMINAMATH_CALUDE_partition_with_equal_product_l3820_382047


namespace NUMINAMATH_CALUDE_inequality_proof_l3820_382016

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 * a * b) / (a + b) < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3820_382016


namespace NUMINAMATH_CALUDE_jim_investment_is_36000_l3820_382067

/-- Represents the investment of three individuals in a business. -/
structure Investment where
  john : ℕ
  james : ℕ
  jim : ℕ

/-- Calculates Jim's investment given the ratio and total investment. -/
def calculate_jim_investment (ratio : Investment) (total : ℕ) : ℕ :=
  let total_parts := ratio.john + ratio.james + ratio.jim
  let jim_parts := ratio.jim
  (total * jim_parts) / total_parts

/-- Theorem stating that Jim's investment is $36,000 given the conditions. -/
theorem jim_investment_is_36000 :
  let ratio : Investment := ⟨4, 7, 9⟩
  let total_investment : ℕ := 80000
  calculate_jim_investment ratio total_investment = 36000 := by
  sorry

end NUMINAMATH_CALUDE_jim_investment_is_36000_l3820_382067


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l3820_382006

/-- In a cyclic quadrilateral ABCD, if angle BAC = d°, angle BCD = 43°, angle ACD = 59°, and angle BAD = 36°, then d = 42°. -/
theorem cyclic_quadrilateral_angle (d : ℝ) : 
  d + 43 + 59 + 36 = 180 → d = 42 := by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l3820_382006


namespace NUMINAMATH_CALUDE_rescue_net_sag_l3820_382081

/-- The sag of an elastic rescue net for two different jumpers -/
theorem rescue_net_sag 
  (m₁ m₂ x₁ h₁ h₂ : ℝ) 
  (hm₁ : m₁ = 78.75)
  (hm₂ : m₂ = 45)
  (hx₁ : x₁ = 1)
  (hh₁ : h₁ = 15)
  (hh₂ : h₂ = 29)
  (x₂ : ℝ) :
  28 * x₂^2 - x₂ - 29 = 0 ↔ 
  m₂ * (h₂ + x₂) / (m₁ * (h₁ + x₁)) = x₂^2 / x₁^2 := by
sorry


end NUMINAMATH_CALUDE_rescue_net_sag_l3820_382081


namespace NUMINAMATH_CALUDE_checkerboard_probability_l3820_382031

/-- The size of one side of the checkerboard -/
def board_size : ℕ := 8

/-- The total number of squares on the checkerboard -/
def total_squares : ℕ := board_size * board_size

/-- The number of squares on the perimeter of the checkerboard -/
def perimeter_squares : ℕ := 4 * (board_size - 1)

/-- The number of squares not touching the outer edge -/
def inner_squares : ℕ := total_squares - perimeter_squares

/-- The probability of choosing a square not touching the outer edge -/
def inner_square_probability : ℚ := inner_squares / total_squares

theorem checkerboard_probability :
  inner_square_probability = 9 / 16 := by sorry

end NUMINAMATH_CALUDE_checkerboard_probability_l3820_382031


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l3820_382028

def point_A : ℝ × ℝ := (-3, 5)

theorem distance_to_x_axis : 
  let (x, y) := point_A
  |y| = 5 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l3820_382028


namespace NUMINAMATH_CALUDE_triangle_division_exists_l3820_382060

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : sorry

/-- A division of a triangle into four convex shapes -/
structure TriangleDivision where
  original : ConvexPolygon 3
  triangle : ConvexPolygon 3
  quadrilateral : ConvexPolygon 4
  pentagon : ConvexPolygon 5
  hexagon : ConvexPolygon 6
  valid_division : sorry

/-- Any triangle can be divided into a triangle, quadrilateral, pentagon, and hexagon -/
theorem triangle_division_exists : ∀ (t : ConvexPolygon 3), ∃ (d : TriangleDivision), d.original = t :=
sorry

end NUMINAMATH_CALUDE_triangle_division_exists_l3820_382060


namespace NUMINAMATH_CALUDE_min_acute_triangles_in_square_l3820_382096

/-- A triangulation of a square. -/
structure SquareTriangulation where
  /-- The number of triangles in the triangulation. -/
  num_triangles : ℕ
  /-- All triangles in the triangulation are acute-angled. -/
  all_acute : Bool
  /-- The triangulation is valid (covers the entire square without overlaps). -/
  valid : Bool

/-- The minimum number of triangles in a valid acute-angled triangulation of a square. -/
def min_acute_triangulation : ℕ := 8

/-- Theorem: The minimum number of acute-angled triangles that a square can be divided into is 8. -/
theorem min_acute_triangles_in_square :
  ∀ t : SquareTriangulation, t.valid ∧ t.all_acute → t.num_triangles ≥ min_acute_triangulation :=
by sorry

end NUMINAMATH_CALUDE_min_acute_triangles_in_square_l3820_382096


namespace NUMINAMATH_CALUDE_system_solution_sum_l3820_382085

theorem system_solution_sum (a b : ℝ) : 
  (1 : ℝ) * a + 2 = -1 ∧ 2 * (1 : ℝ) - b * 2 = 0 → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_sum_l3820_382085


namespace NUMINAMATH_CALUDE_root_in_interval_implies_a_range_l3820_382087

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 1

-- State the theorem
theorem root_in_interval_implies_a_range :
  ∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a x = 0) → a ∈ Set.Ici (-1) :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_a_range_l3820_382087


namespace NUMINAMATH_CALUDE_owen_turtle_ratio_l3820_382078

def turtle_problem (owen_initial johanna_initial owen_after_month owen_final : ℕ) : Prop :=
  -- Owen initially has 21 turtles
  owen_initial = 21 ∧
  -- Johanna initially has 5 fewer turtles than Owen
  johanna_initial = owen_initial - 5 ∧
  -- After 1 month, Owen has a certain multiple of his initial number of turtles
  ∃ k : ℕ, owen_after_month = k * owen_initial ∧
  -- After 1 month, Johanna loses half of her turtles and donates the rest to Owen
  owen_final = owen_after_month + (johanna_initial / 2) ∧
  -- After all these events, Owen has 50 turtles
  owen_final = 50

theorem owen_turtle_ratio (owen_initial johanna_initial owen_after_month owen_final : ℕ)
  (h : turtle_problem owen_initial johanna_initial owen_after_month owen_final) :
  owen_after_month = 2 * owen_initial :=
by sorry

end NUMINAMATH_CALUDE_owen_turtle_ratio_l3820_382078


namespace NUMINAMATH_CALUDE_xiao_ming_arrival_time_l3820_382003

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : minutes < 60 := by sorry

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60
  , minutes := totalMinutes % 60 }

theorem xiao_ming_arrival_time 
  (departure_time : Time)
  (journey_duration : Nat)
  (h1 : departure_time.hours = 6)
  (h2 : departure_time.minutes = 55)
  (h3 : journey_duration = 30) :
  addMinutes departure_time journey_duration = { hours := 7, minutes := 25 } := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_arrival_time_l3820_382003


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l3820_382017

theorem sum_of_five_consecutive_even_integers (n : ℤ) :
  (2*n) + (2*n + 2) + (2*n + 4) + (2*n + 6) + (2*n + 8) = 10*n + 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l3820_382017


namespace NUMINAMATH_CALUDE_company_fund_problem_l3820_382018

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  (initial_fund = 60 * n - 10) →  -- The fund initially contained $10 less than needed for $60 bonuses
  (initial_fund = 50 * n + 140) → -- Each employee received a $50 bonus, and $140 remained
  initial_fund = 890 := by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l3820_382018


namespace NUMINAMATH_CALUDE_third_group_men_l3820_382044

/-- The work rate of a man -/
def man_rate : ℝ := sorry

/-- The work rate of a woman -/
def woman_rate : ℝ := sorry

/-- The number of men in the third group -/
def x : ℕ := sorry

/-- The work rate of 3 men and 8 women equals the work rate of 6 men and 2 women -/
axiom work_rate_equality : 3 * man_rate + 8 * woman_rate = 6 * man_rate + 2 * woman_rate

/-- The work rate of x men and 2 women is 0.7142857142857143 times the work rate of 3 men and 8 women -/
axiom work_rate_fraction : 
  x * man_rate + 2 * woman_rate = 0.7142857142857143 * (3 * man_rate + 8 * woman_rate)

/-- The number of men in the third group is 4 -/
theorem third_group_men : x = 4 := by sorry

end NUMINAMATH_CALUDE_third_group_men_l3820_382044


namespace NUMINAMATH_CALUDE_square_diagonals_equal_l3820_382041

-- Define a structure for shapes with diagonals
structure ShapeWithDiagonals :=
  (diagonal1 : ℝ)
  (diagonal2 : ℝ)

-- Define rectangle and square
class Rectangle extends ShapeWithDiagonals

class Square extends Rectangle

-- State the theorem about rectangle diagonals
axiom rectangle_diagonals_equal (r : Rectangle) : r.diagonal1 = r.diagonal2

-- State that a square is a rectangle
axiom square_is_rectangle (s : Square) : Rectangle

-- Theorem to prove
theorem square_diagonals_equal (s : Square) : 
  (square_is_rectangle s).diagonal1 = (square_is_rectangle s).diagonal2 :=
by sorry

end NUMINAMATH_CALUDE_square_diagonals_equal_l3820_382041


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l3820_382014

theorem wire_ratio_proof (total_length shorter_length : ℕ) 
  (h1 : total_length = 140)
  (h2 : shorter_length = 40) :
  shorter_length * 5 = (total_length - shorter_length) * 2 := by
sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l3820_382014


namespace NUMINAMATH_CALUDE_inequality_proof_l3820_382042

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3820_382042


namespace NUMINAMATH_CALUDE_unique_pair_for_n_l3820_382030

theorem unique_pair_for_n (n : ℕ+) :
  ∃! (a b : ℕ+), n = (1/2) * ((a + b - 1) * (a + b - 2) : ℕ) + a := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_for_n_l3820_382030


namespace NUMINAMATH_CALUDE_captain_selection_count_l3820_382025

/-- The number of ways to choose k items from n items without regard to order -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of people in the team -/
def team_size : ℕ := 15

/-- The number of captains to be chosen -/
def captain_count : ℕ := 4

/-- Theorem: The number of ways to choose 4 captains from a team of 15 people is 1365 -/
theorem captain_selection_count : choose team_size captain_count = 1365 := by
  sorry

end NUMINAMATH_CALUDE_captain_selection_count_l3820_382025


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3820_382097

/-- Given a geometric sequence {a_n} with sum of first n terms S_n, prove S_5/a_5 = 31 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n, a (n + 1) = q * a n) →  -- a_n is a geometric sequence
  (a 1 + a 3 = 5/2) →                    -- First condition
  (a 2 + a 4 = 5/4) →                    -- Second condition
  (∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))) →  -- Definition of S_n
  (S 5 / a 5 = 31) :=                    -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3820_382097


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3820_382009

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 16 + Real.tan (45 * π / 180)) :
  (m + 2 + 5 / (2 - m)) * ((2 * m - 4) / (3 - m)) = -16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3820_382009


namespace NUMINAMATH_CALUDE_mani_pedi_cost_l3820_382008

/-- Calculates the total cost of mani/pedis with a discount --/
theorem mani_pedi_cost (regular_price : ℝ) (discount_percent : ℝ) (num_people : ℕ) :
  regular_price = 40 →
  discount_percent = 25 →
  num_people = 5 →
  (1 - discount_percent / 100) * regular_price * num_people = 150 := by
  sorry

end NUMINAMATH_CALUDE_mani_pedi_cost_l3820_382008


namespace NUMINAMATH_CALUDE_shaded_area_division_l3820_382004

/-- Represents a grid in the first quadrant -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)
  (shaded_squares : ℕ)

/-- Represents a line passing through (0,0) and (8,c) -/
structure Line :=
  (c : ℝ)

/-- Checks if a line divides the shaded area of a grid into two equal parts -/
def divides_equally (g : Grid) (l : Line) : Prop :=
  ∃ (area : ℝ), area > 0 ∧ area * 2 = g.shaded_squares

theorem shaded_area_division (g : Grid) (l : Line) :
  g.width = 8 ∧ g.height = 6 ∧ g.shaded_squares = 32 →
  divides_equally g l ↔ l.c = 4 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_division_l3820_382004


namespace NUMINAMATH_CALUDE_median_on_hypotenuse_l3820_382076

/-- Represents a right triangle with legs a and b, and median m on the hypotenuse -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  m : ℝ

/-- The median on the hypotenuse of a right triangle with legs 6 and 8 is 5 -/
theorem median_on_hypotenuse (t : RightTriangle) (h1 : t.a = 6) (h2 : t.b = 8) : t.m = 5 := by
  sorry

end NUMINAMATH_CALUDE_median_on_hypotenuse_l3820_382076


namespace NUMINAMATH_CALUDE_lolita_milk_consumption_l3820_382000

/-- The number of boxes of milk Lolita drinks on a weekday -/
def weekday_milk : ℕ := 3

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of boxes of milk Lolita drinks on Saturday -/
def saturday_milk : ℕ := 2 * weekday_milk

/-- The number of boxes of milk Lolita drinks on Sunday -/
def sunday_milk : ℕ := 3 * weekday_milk

/-- The total number of boxes of milk Lolita drinks in a week -/
def total_weekly_milk : ℕ := weekday_milk * weekdays + saturday_milk + sunday_milk

theorem lolita_milk_consumption :
  total_weekly_milk = 30 := by sorry

end NUMINAMATH_CALUDE_lolita_milk_consumption_l3820_382000


namespace NUMINAMATH_CALUDE_parametric_to_standard_equation_l3820_382033

theorem parametric_to_standard_equation (x y θ : ℝ) 
  (h1 : x = 1 + 2 * Real.cos θ) 
  (h2 : y = 2 * Real.sin θ) : 
  (x - 1)^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_parametric_to_standard_equation_l3820_382033


namespace NUMINAMATH_CALUDE_total_black_dots_l3820_382079

/-- The number of butterflies -/
def num_butterflies : ℕ := 397

/-- The number of black dots per butterfly -/
def black_dots_per_butterfly : ℕ := 12

/-- Theorem: The total number of black dots is 4764 -/
theorem total_black_dots : num_butterflies * black_dots_per_butterfly = 4764 := by
  sorry

end NUMINAMATH_CALUDE_total_black_dots_l3820_382079


namespace NUMINAMATH_CALUDE_lineup_combinations_l3820_382073

/-- The number of ways to choose a starting lineup -/
def choose_lineup (total_players : ℕ) (offensive_linemen : ℕ) (kickers : ℕ) : ℕ :=
  offensive_linemen * kickers * (total_players - 2) * (total_players - 3) * (total_players - 4)

/-- Theorem stating the number of ways to choose the lineup -/
theorem lineup_combinations :
  choose_lineup 12 4 2 = 5760 := by
  sorry

end NUMINAMATH_CALUDE_lineup_combinations_l3820_382073


namespace NUMINAMATH_CALUDE_max_parts_with_parallel_lines_l3820_382027

/-- The maximum number of parts a plane can be divided into by n lines -/
def max_parts (n : ℕ) : ℕ := sorry

/-- The number of additional parts created by adding a line that intersects all existing lines -/
def additional_parts (n : ℕ) : ℕ := sorry

theorem max_parts_with_parallel_lines 
  (total_lines : ℕ) 
  (parallel_lines : ℕ) 
  (h1 : total_lines = 10) 
  (h2 : parallel_lines = 4) 
  (h3 : parallel_lines ≤ total_lines) :
  max_parts total_lines = max_parts (total_lines - parallel_lines) + 
    parallel_lines * (additional_parts (total_lines - parallel_lines)) ∧
  max_parts total_lines = 50 := by sorry

end NUMINAMATH_CALUDE_max_parts_with_parallel_lines_l3820_382027


namespace NUMINAMATH_CALUDE_inequality_proof_l3820_382082

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3820_382082


namespace NUMINAMATH_CALUDE_equation_equivalence_l3820_382023

theorem equation_equivalence : ∀ x y : ℝ, (5 * x - y = 6) ↔ (y = 5 * x - 6) := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3820_382023


namespace NUMINAMATH_CALUDE_incorrect_inequality_l3820_382069

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬(5 - a > 5 - b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l3820_382069


namespace NUMINAMATH_CALUDE_circle_C_properties_l3820_382043

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 2)^2 = 9

-- Define the line L where the center of C lies
def line_L (x y : ℝ) : Prop :=
  x + 2*y + 1 = 0

-- Define the line that potentially intersects C
def intersecting_line (a x y : ℝ) : Prop :=
  a*x - y + 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (2, 0)

theorem circle_C_properties :
  -- The circle C passes through M(0,-2) and N(3,1)
  circle_C 0 (-2) ∧ circle_C 3 1 ∧
  -- The center of C lies on line L
  ∃ (cx cy : ℝ), line_L cx cy ∧ ∀ (x y : ℝ), circle_C x y ↔ (x - cx)^2 + (y - cy)^2 = 9 ∧
  -- There's no real a such that the line ax-y+1=0 intersects C at two points
  -- and is perpendicularly bisected by the line through P
  ¬ ∃ (a : ℝ), 
    (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
                          intersecting_line a x₁ y₁ ∧ intersecting_line a x₂ y₂) ∧
    (∃ (mx my : ℝ), circle_C mx my ∧ 
                    (mx - point_P.1) * (x₂ - x₁) + (my - point_P.2) * (y₂ - y₁) = 0 ∧
                    2 * mx = x₁ + x₂ ∧ 2 * my = y₁ + y₂) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l3820_382043


namespace NUMINAMATH_CALUDE_meat_remaining_l3820_382091

theorem meat_remaining (initial_meat : ℝ) (meatball_fraction : ℝ) (spring_roll_meat : ℝ) :
  initial_meat = 20 →
  meatball_fraction = 1/4 →
  spring_roll_meat = 3 →
  initial_meat - (meatball_fraction * initial_meat + spring_roll_meat) = 12 := by
  sorry

end NUMINAMATH_CALUDE_meat_remaining_l3820_382091


namespace NUMINAMATH_CALUDE_exists_hexagonal_2016_l3820_382092

/-- The n-th hexagonal number -/
def hexagonal (n : ℕ) : ℕ := 2 * n^2 - n

/-- 2016 is a hexagonal number -/
theorem exists_hexagonal_2016 : ∃ n : ℕ, n > 0 ∧ hexagonal n = 2016 := by
  sorry

end NUMINAMATH_CALUDE_exists_hexagonal_2016_l3820_382092


namespace NUMINAMATH_CALUDE_square_root_of_four_l3820_382048

theorem square_root_of_four :
  {y : ℝ | y^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l3820_382048


namespace NUMINAMATH_CALUDE_product_of_powers_equals_hundred_l3820_382002

theorem product_of_powers_equals_hundred :
  (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.35) * (10 ^ 0.05) * (10 ^ 0.85) * (10 ^ 0.35) = 100 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_hundred_l3820_382002


namespace NUMINAMATH_CALUDE_simplify_fraction_l3820_382056

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 0) :
  (1 - 1 / (x - 3)) / ((x^2 - 4*x) / (x^2 - 9)) = (x + 3) / x := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3820_382056


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l3820_382001

/-- Given a quadratic equation px² + qx + r = 0 with roots u and v,
    the quadratic equation with roots p²u - q and p²v - q is
    x² + (pq + 2q)x + (p³r + pq² + q²) = 0 -/
theorem quadratic_root_transformation (p q r u v : ℝ) :
  (p * u^2 + q * u + r = 0) →
  (p * v^2 + q * v + r = 0) →
  (u ≠ v) →
  ∀ x, x^2 + (p*q + 2*q)*x + (p^3*r + p*q^2 + q^2) = 0 ↔
       (x = p^2*u - q ∨ x = p^2*v - q) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l3820_382001


namespace NUMINAMATH_CALUDE_average_score_is_two_l3820_382066

/-- Represents the distribution of scores in a class test --/
structure ScoreDistribution where
  score3 : Real
  score2 : Real
  score1 : Real
  score0 : Real
  sum_to_one : score3 + score2 + score1 + score0 = 1

/-- Calculates the average score given a score distribution --/
def averageScore (d : ScoreDistribution) : Real :=
  3 * d.score3 + 2 * d.score2 + 1 * d.score1 + 0 * d.score0

/-- Theorem: The average score for the given distribution is 2.0 --/
theorem average_score_is_two :
  let d : ScoreDistribution := {
    score3 := 0.3,
    score2 := 0.5,
    score1 := 0.1,
    score0 := 0.1,
    sum_to_one := by norm_num
  }
  averageScore d = 2.0 := by sorry

end NUMINAMATH_CALUDE_average_score_is_two_l3820_382066


namespace NUMINAMATH_CALUDE_largest_circular_pool_diameter_l3820_382072

/-- Given a rectangular garden with area 180 square meters and length three times its width,
    the diameter of the largest circular pool that can be outlined by the garden's perimeter
    is 16√15/π meters. -/
theorem largest_circular_pool_diameter (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 3 * width →
  width * length = 180 →
  (2 * (width + length)) / π = 16 * Real.sqrt 15 / π :=
by sorry

end NUMINAMATH_CALUDE_largest_circular_pool_diameter_l3820_382072


namespace NUMINAMATH_CALUDE_problem_solution_l3820_382064

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) :
  (∀ a : ℝ, a < 1/2 → 1/x + 1/y ≥ |a + 2| - |a - 1|) ∧
  x^2 + 2*y^2 ≥ 8/3 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3820_382064


namespace NUMINAMATH_CALUDE_common_root_quadratic_equations_l3820_382058

theorem common_root_quadratic_equations (b : ℤ) :
  (∃ x : ℝ, 2 * x^2 + (3 * b - 1) * x - 3 = 0 ∧ 6 * x^2 - (2 * b - 3) * x - 1 = 0) ↔ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_common_root_quadratic_equations_l3820_382058


namespace NUMINAMATH_CALUDE_answer_key_combinations_l3820_382098

/-- The number of ways to answer a single true-false question -/
def true_false_options : ℕ := 2

/-- The number of true-false questions in the quiz -/
def num_true_false : ℕ := 4

/-- The number of ways to answer a single multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- The number of multiple-choice questions in the quiz -/
def num_multiple_choice : ℕ := 2

/-- The total number of possible answer combinations for true-false questions -/
def total_true_false_combinations : ℕ := true_false_options ^ num_true_false

/-- The number of invalid true-false combinations (all true or all false) -/
def invalid_true_false_combinations : ℕ := 2

/-- The number of valid true-false combinations -/
def valid_true_false_combinations : ℕ := total_true_false_combinations - invalid_true_false_combinations

/-- The number of ways to answer all multiple-choice questions -/
def multiple_choice_combinations : ℕ := multiple_choice_options ^ num_multiple_choice

/-- The total number of ways to create an answer key for the quiz -/
def total_answer_key_combinations : ℕ := valid_true_false_combinations * multiple_choice_combinations

theorem answer_key_combinations : total_answer_key_combinations = 224 := by
  sorry

end NUMINAMATH_CALUDE_answer_key_combinations_l3820_382098


namespace NUMINAMATH_CALUDE_orvin_balloon_purchase_l3820_382037

/-- Represents the cost of balloons in cents -/
def regular_price : ℕ := 200

/-- Represents the total amount of money Orvin has in cents -/
def total_money : ℕ := 40 * regular_price

/-- Represents the cost of a pair of balloons (one at regular price, one at half price) in cents -/
def pair_cost : ℕ := regular_price + regular_price / 2

/-- The maximum number of balloons Orvin can buy -/
def max_balloons : ℕ := 2 * (total_money / pair_cost)

theorem orvin_balloon_purchase :
  max_balloons = 52 := by sorry

end NUMINAMATH_CALUDE_orvin_balloon_purchase_l3820_382037


namespace NUMINAMATH_CALUDE_function_relationship_l3820_382094

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x ≥ -4 → y ≥ -4 → x < y → f x < f y)
variable (h2 : ∀ x, f (x - 4) = f (-x - 4))

-- State the theorem
theorem function_relationship :
  f (-4) < f (-6) ∧ f (-6) < f 0 :=
sorry

end NUMINAMATH_CALUDE_function_relationship_l3820_382094


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l3820_382084

theorem cubic_sum_problem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l3820_382084


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3820_382015

/-- The distance from point (2,1) to the line x=a is 3 -/
def distance_condition (a : ℝ) : Prop := |a - 2| = 3

/-- a=5 is a sufficient condition -/
theorem sufficient_condition : distance_condition 5 := by sorry

/-- a=5 is not a necessary condition -/
theorem not_necessary_condition : ∃ x, x ≠ 5 ∧ distance_condition x := by sorry

/-- a=5 is a sufficient but not necessary condition -/
theorem sufficient_but_not_necessary : 
  (distance_condition 5) ∧ (∃ x, x ≠ 5 ∧ distance_condition x) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3820_382015


namespace NUMINAMATH_CALUDE_function_increasing_in_interval_l3820_382075

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * sin (ω * x) + cos (ω * x)

theorem function_increasing_in_interval 
  (h_symmetry : ∀ (x : ℝ), f ω (π/6 - x) = f ω (π/6 + x))
  (h_smallest_ω : ∀ (ω' : ℝ), ω' > 0 → ω' ≥ ω)
  : StrictMonoOn f (Set.Ioo 0 (π/6)) := by sorry

end NUMINAMATH_CALUDE_function_increasing_in_interval_l3820_382075


namespace NUMINAMATH_CALUDE_interest_problem_l3820_382010

/-- Given compound and simple interest conditions, prove the principal amount -/
theorem interest_problem (P R : ℝ) : 
  P * ((1 + R / 100) ^ 2 - 1) = 11730 →
  (P * R * 2) / 100 = 10200 →
  P = 34000 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_l3820_382010


namespace NUMINAMATH_CALUDE_permutations_of_three_letter_word_is_six_l3820_382053

/-- The number of permutations of a 3-letter word with distinct letters -/
def permutations_of_three_letter_word : ℕ :=
  Nat.factorial 3

/-- Proof that the number of permutations of a 3-letter word with distinct letters is 6 -/
theorem permutations_of_three_letter_word_is_six :
  permutations_of_three_letter_word = 6 := by
  sorry

#eval permutations_of_three_letter_word

end NUMINAMATH_CALUDE_permutations_of_three_letter_word_is_six_l3820_382053


namespace NUMINAMATH_CALUDE_profit_is_27000_l3820_382059

/-- Represents the profit sharing problem between Tom and Jose -/
structure ProfitSharing where
  tom_investment : ℕ
  tom_months : ℕ
  jose_investment : ℕ
  jose_months : ℕ
  jose_profit : ℕ

/-- Calculates the total profit earned by Tom and Jose -/
def total_profit (ps : ProfitSharing) : ℕ :=
  let tom_total := ps.tom_investment * ps.tom_months
  let jose_total := ps.jose_investment * ps.jose_months
  let ratio_sum := (tom_total / (tom_total.gcd jose_total)) + (jose_total / (tom_total.gcd jose_total))
  (ratio_sum * ps.jose_profit) / (jose_total / (tom_total.gcd jose_total))

/-- Theorem stating that the total profit is 27000 for the given conditions -/
theorem profit_is_27000 (ps : ProfitSharing)
  (h1 : ps.tom_investment = 30000)
  (h2 : ps.tom_months = 12)
  (h3 : ps.jose_investment = 45000)
  (h4 : ps.jose_months = 10)
  (h5 : ps.jose_profit = 15000) :
  total_profit ps = 27000 := by
  sorry

#eval total_profit { tom_investment := 30000, tom_months := 12, jose_investment := 45000, jose_months := 10, jose_profit := 15000 }

end NUMINAMATH_CALUDE_profit_is_27000_l3820_382059


namespace NUMINAMATH_CALUDE_deck_size_proof_l3820_382055

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b : ℚ) = 1/4 →
  ((r + 6 : ℚ) / (r + b + 6 : ℚ) = 1/3) →
  r + b = 48 := by
  sorry

end NUMINAMATH_CALUDE_deck_size_proof_l3820_382055


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3820_382054

/-- Given the expression 5(x^3 - 3x^2 + 4) - 8(2x^3 - x^2 - 2), 
    the sum of the squares of its coefficients when fully simplified is 1466. -/
theorem sum_of_squared_coefficients : 
  let expr := fun (x : ℝ) => 5 * (x^3 - 3*x^2 + 4) - 8 * (2*x^3 - x^2 - 2)
  let simplified := fun (x : ℝ) => -11*x^3 - 7*x^2 + 36
  (∀ x, expr x = simplified x) → 
  (-11)^2 + (-7)^2 + 36^2 = 1466 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3820_382054


namespace NUMINAMATH_CALUDE_distinct_triangles_in_square_pyramid_l3820_382029

-- Define the number of vertices in a square pyramid
def num_vertices : ℕ := 5

-- Define the number of vertices needed to form a triangle
def vertices_per_triangle : ℕ := 3

-- Define the function to calculate combinations
def combinations (n k : ℕ) : ℕ := 
  (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem distinct_triangles_in_square_pyramid :
  combinations num_vertices vertices_per_triangle = 10 := by
  sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_square_pyramid_l3820_382029


namespace NUMINAMATH_CALUDE_max_value_and_constraint_optimization_l3820_382057

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - 2*|x + 1|

-- State the theorem
theorem max_value_and_constraint_optimization :
  (∃ m : ℝ, ∀ x : ℝ, f x ≤ m ∧ ∃ x₀ : ℝ, f x₀ = m) ∧ 
  m = 2 ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a^2 + 2*b^2 + c^2 = m → 
    ab + bc ≤ 1 ∧ ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    a₀^2 + 2*b₀^2 + c₀^2 = m ∧ a₀*b₀ + b₀*c₀ = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_max_value_and_constraint_optimization_l3820_382057


namespace NUMINAMATH_CALUDE_unoccupied_volume_is_305_l3820_382012

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def prismVolume (d : PrismDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the tank with its contents -/
structure Tank where
  dimensions : PrismDimensions
  oilFillFraction : ℝ
  numIceCubes : ℕ
  iceCubeSize : ℝ

/-- Calculates the unoccupied volume in the tank -/
def unoccupiedVolume (t : Tank) : ℝ :=
  let totalVolume := prismVolume t.dimensions
  let oilVolume := t.oilFillFraction * totalVolume
  let iceVolume := t.numIceCubes * (t.iceCubeSize ^ 3)
  totalVolume - oilVolume - iceVolume

/-- Theorem: The unoccupied volume in the specified tank is 305 cubic inches -/
theorem unoccupied_volume_is_305 :
  let t : Tank := {
    dimensions := { length := 12, width := 10, height := 8 },
    oilFillFraction := 2/3,
    numIceCubes := 15,
    iceCubeSize := 1
  }
  unoccupiedVolume t = 305 := by sorry

end NUMINAMATH_CALUDE_unoccupied_volume_is_305_l3820_382012
