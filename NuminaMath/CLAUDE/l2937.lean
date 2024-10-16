import Mathlib

namespace NUMINAMATH_CALUDE_total_cds_l2937_293711

theorem total_cds (a b : ℕ) : 
  (b + 6 = 2 * (a - 6)) →
  (a + 6 = b - 6) →
  a + b = 72 := by
sorry

end NUMINAMATH_CALUDE_total_cds_l2937_293711


namespace NUMINAMATH_CALUDE_problem_statement_l2937_293777

theorem problem_statement (a b : ℝ) (h : |a + 2| + (b - 1)^2 = 0) : 
  (a + b)^2023 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2937_293777


namespace NUMINAMATH_CALUDE_prob_not_losing_l2937_293797

/-- The probability of A not losing in a chess game -/
theorem prob_not_losing (prob_draw prob_win : ℚ) : 
  prob_draw = 1/2 → prob_win = 1/3 → prob_draw + prob_win = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_losing_l2937_293797


namespace NUMINAMATH_CALUDE_inequalities_proof_l2937_293764

theorem inequalities_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a / Real.sqrt b) + (b / Real.sqrt a) ≥ Real.sqrt a + Real.sqrt b ∧
  (a + b = 1 → (1/a) + (1/b) + (1/(a*b)) ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2937_293764


namespace NUMINAMATH_CALUDE_multiplicative_inverse_290_mod_1721_l2937_293754

theorem multiplicative_inverse_290_mod_1721 : ∃ n : ℕ, 
  51^2 + 140^2 = 149^2 → 
  n < 1721 ∧ 
  (290 * n) % 1721 = 1 ∧ 
  n = 1456 := by
sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_290_mod_1721_l2937_293754


namespace NUMINAMATH_CALUDE_chord_length_l2937_293714

/-- The length of the chord cut off by a circle on a line --/
theorem chord_length (x y : ℝ) : 
  let line := {(x, y) | x - y - 3 = 0}
  let circle := {(x, y) | (x - 2)^2 + y^2 = 4}
  let chord := line ∩ circle
  ∃ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l2937_293714


namespace NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l2937_293757

/-- Given an isosceles triangle and a rectangle with the same area, 
    prove that the height of the triangle is twice the breadth of the rectangle. -/
theorem isosceles_triangle_rectangle_equal_area 
  (l b h : ℝ) (hl : l > 0) (hb : b > 0) (hlb : l > b) : 
  (1 / 2 : ℝ) * l * h = l * b → h = 2 * b := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l2937_293757


namespace NUMINAMATH_CALUDE_fraction_meaningfulness_l2937_293746

theorem fraction_meaningfulness (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x + 2)) ↔ x ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningfulness_l2937_293746


namespace NUMINAMATH_CALUDE_min_value_theorem_l2937_293766

/-- Given positive real numbers m and n, vectors a and b, and a parallel to b,
    prove that the minimum value of 1/m + 2/n is 3 + 2√2 -/
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (a b : Fin 2 → ℝ)
  (ha : a = λ i => if i = 0 then m else 1)
  (hb : b = λ i => if i = 0 then 1 - n else 1)
  (parallel : ∃ (k : ℝ), a = λ i => k * (b i)) :
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 2/y ≥ 3 + 2 * Real.sqrt 2) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 2/y = 3 + 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2937_293766


namespace NUMINAMATH_CALUDE_probability_theorem_l2937_293780

/-- The probability that the straight-line distance between two randomly chosen points 
    on the sides of a square with side length 2 is at least 1 -/
def probability_distance_at_least_one (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- A square with side length 2 -/
def square_side_two : Set (ℝ × ℝ) :=
  sorry

theorem probability_theorem :
  let S := square_side_two
  probability_distance_at_least_one S = (22 - π) / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l2937_293780


namespace NUMINAMATH_CALUDE_sylvester_theorem_l2937_293718

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in the plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define when a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define when points are collinear
def collinear (p1 p2 p3 : Point) : Prop :=
  ∃ l : Line, pointOnLine p1 l ∧ pointOnLine p2 l ∧ pointOnLine p3 l

-- Define when a set of points is not all collinear
def notAllCollinear (E : Set Point) : Prop :=
  ∃ p1 p2 p3 : Point, p1 ∈ E ∧ p2 ∈ E ∧ p3 ∈ E ∧ ¬collinear p1 p2 p3

-- Sylvester's theorem statement
theorem sylvester_theorem (E : Set Point) (h1 : E.Finite) (h2 : notAllCollinear E) :
  ∃ l : Line, ∃ p1 p2 : Point, p1 ∈ E ∧ p2 ∈ E ∧ p1 ≠ p2 ∧
    pointOnLine p1 l ∧ pointOnLine p2 l ∧
    ∀ p3 : Point, p3 ∈ E → pointOnLine p3 l → (p3 = p1 ∨ p3 = p2) :=
  sorry

end NUMINAMATH_CALUDE_sylvester_theorem_l2937_293718


namespace NUMINAMATH_CALUDE_twelfth_term_is_fifteen_l2937_293784

/-- An arithmetic sequence {a_n} with the given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 + a 4 + a 5 = 3 ∧
  a 8 = 8

/-- Theorem stating that for an arithmetic sequence satisfying the given conditions, 
    the 12th term is equal to 15 -/
theorem twelfth_term_is_fifteen (a : ℕ → ℚ) 
  (h : arithmetic_sequence a) : a 12 = 15 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_is_fifteen_l2937_293784


namespace NUMINAMATH_CALUDE_restaurant_customer_prediction_l2937_293794

theorem restaurant_customer_prediction 
  (breakfast_customers : ℕ) 
  (lunch_customers : ℕ) 
  (dinner_customers : ℕ) 
  (h1 : breakfast_customers = 73)
  (h2 : lunch_customers = 127)
  (h3 : dinner_customers = 87) :
  2 * (breakfast_customers + lunch_customers + dinner_customers) = 574 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_customer_prediction_l2937_293794


namespace NUMINAMATH_CALUDE_point_coordinates_sum_l2937_293795

theorem point_coordinates_sum (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  let slope : ℝ := (3 - 0) / (x - 0)
  slope = 3/4 → x + 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_point_coordinates_sum_l2937_293795


namespace NUMINAMATH_CALUDE_final_piece_count_l2937_293769

/-- Represents the number of pieces after each cut -/
structure PaperCuts where
  initial : Nat
  first_cut : Nat
  second_cut : Nat
  third_cut : Nat
  fourth_cut : Nat

/-- The cutting process as described in the problem -/
def cutting_process : PaperCuts :=
  { initial := 1
  , first_cut := 10
  , second_cut := 19
  , third_cut := 28
  , fourth_cut := 37 }

/-- Theorem stating that the final number of pieces is 37 -/
theorem final_piece_count :
  (cutting_process.fourth_cut = 37) := by sorry

end NUMINAMATH_CALUDE_final_piece_count_l2937_293769


namespace NUMINAMATH_CALUDE_balloon_count_l2937_293700

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := 41

/-- The total number of blue balloons Joan and Melanie have together -/
def total_balloons : ℕ := joan_balloons + melanie_balloons

theorem balloon_count : total_balloons = 81 := by sorry

end NUMINAMATH_CALUDE_balloon_count_l2937_293700


namespace NUMINAMATH_CALUDE_merill_marble_count_l2937_293713

/-- The number of marbles each person has -/
structure MarbleCount where
  merill : ℕ
  elliot : ℕ
  selma : ℕ

/-- The conditions of the marble problem -/
def marbleProblemConditions (m : MarbleCount) : Prop :=
  m.merill = 2 * m.elliot ∧
  m.merill + m.elliot = m.selma - 5 ∧
  m.selma = 50

/-- Theorem stating that under the given conditions, Merill has 30 marbles -/
theorem merill_marble_count (m : MarbleCount) 
  (h : marbleProblemConditions m) : m.merill = 30 := by
  sorry


end NUMINAMATH_CALUDE_merill_marble_count_l2937_293713


namespace NUMINAMATH_CALUDE_window_area_ratio_l2937_293734

theorem window_area_ratio :
  let AB : ℝ := 36
  let AD : ℝ := AB * (5/3)
  let circle_area : ℝ := Real.pi * (AB/2)^2
  let rectangle_area : ℝ := AD * AB
  let square_area : ℝ := AB^2
  rectangle_area / (circle_area + square_area) = 2160 / (324 * Real.pi + 1296) :=
by sorry

end NUMINAMATH_CALUDE_window_area_ratio_l2937_293734


namespace NUMINAMATH_CALUDE_revenue_growth_equation_l2937_293763

def january_revenue : ℝ := 250
def quarter_target : ℝ := 900

theorem revenue_growth_equation (x : ℝ) :
  january_revenue + january_revenue * (1 + x) + january_revenue * (1 + x)^2 = quarter_target :=
by sorry

end NUMINAMATH_CALUDE_revenue_growth_equation_l2937_293763


namespace NUMINAMATH_CALUDE_shelf_arrangement_l2937_293798

/-- The number of ways to choose k items from n items without considering order -/
def combination (n k : ℕ) : ℕ := Nat.choose n k

/-- The problem statement -/
theorem shelf_arrangement : combination 8 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_shelf_arrangement_l2937_293798


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2937_293767

def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  is_in_second_quadrant (-7 : ℝ) (3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2937_293767


namespace NUMINAMATH_CALUDE_odd_digits_in_base4_350_l2937_293717

-- Define a function to convert a number from base 10 to base 4
def toBase4 (n : ℕ) : List ℕ := sorry

-- Define a function to count odd digits in a list of digits
def countOddDigits (digits : List ℕ) : ℕ := sorry

-- Theorem statement
theorem odd_digits_in_base4_350 :
  countOddDigits (toBase4 350) = 4 := by sorry

end NUMINAMATH_CALUDE_odd_digits_in_base4_350_l2937_293717


namespace NUMINAMATH_CALUDE_fraction_product_l2937_293783

theorem fraction_product : (2 : ℚ) / 3 * 5 / 7 * 9 / 11 * 4 / 13 = 360 / 3003 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l2937_293783


namespace NUMINAMATH_CALUDE_club_members_after_five_years_l2937_293776

/-- Represents the number of people in the club after k years -/
def club_members (k : ℕ) : ℕ :=
  if k = 0 then 18
  else 3 * club_members (k - 1) - 10

/-- The number of people in the club after 5 years is 3164 -/
theorem club_members_after_five_years :
  club_members 5 = 3164 := by
  sorry

end NUMINAMATH_CALUDE_club_members_after_five_years_l2937_293776


namespace NUMINAMATH_CALUDE_fraction_count_l2937_293756

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def satisfies_condition (n m : ℕ) : Prop :=
  is_two_digit n ∧ is_two_digit m ∧ n * (m + 19) = m * (n + 20)

theorem fraction_count : 
  ∃! (count : ℕ), ∃ (pairs : Finset (ℕ × ℕ)),
    pairs.card = count ∧
    (∀ (pair : ℕ × ℕ), pair ∈ pairs ↔ satisfies_condition pair.1 pair.2) ∧
    count = 3 :=
sorry

end NUMINAMATH_CALUDE_fraction_count_l2937_293756


namespace NUMINAMATH_CALUDE_inequality_proof_l2937_293762

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  a / (a - c) > b / (b - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2937_293762


namespace NUMINAMATH_CALUDE_sum_remainder_zero_l2937_293785

theorem sum_remainder_zero (n : ℤ) : (10 - 2*n + 4*n + 2) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_zero_l2937_293785


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2937_293710

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (5 * x₁^2 + 3 * x₁ - 7 = 0) →
  (5 * x₂^2 + 3 * x₂ - 7 = 0) →
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 79/25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2937_293710


namespace NUMINAMATH_CALUDE_sine_zeros_range_l2937_293709

open Real

theorem sine_zeros_range (ω : ℝ) : 
  (ω > 0) → 
  (∃! (z₁ z₂ : ℝ), 0 ≤ z₁ ∧ z₁ < z₂ ∧ z₂ ≤ π/4 ∧ 
    sin (2*ω*z₁ - π/6) = 0 ∧ sin (2*ω*z₂ - π/6) = 0 ∧
    ∀ z, 0 ≤ z ∧ z ≤ π/4 ∧ sin (2*ω*z - π/6) = 0 → z = z₁ ∨ z = z₂) ↔ 
  (7/3 ≤ ω ∧ ω < 13/3) :=
by sorry

end NUMINAMATH_CALUDE_sine_zeros_range_l2937_293709


namespace NUMINAMATH_CALUDE_sam_candy_bars_l2937_293715

/-- Represents the number of candy bars Sam bought -/
def candy_bars : ℕ := sorry

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of dimes Sam initially had -/
def initial_dimes : ℕ := 19

/-- The number of quarters Sam initially had -/
def initial_quarters : ℕ := 6

/-- The cost of a candy bar in dimes -/
def candy_bar_cost_dimes : ℕ := 3

/-- The cost of a lollipop in quarters -/
def lollipop_cost_quarters : ℕ := 1

/-- The amount of money Sam has left after purchases, in cents -/
def remaining_cents : ℕ := 195

theorem sam_candy_bars : 
  candy_bars = 4 ∧
  initial_dimes * dime_value + initial_quarters * quarter_value = 
  remaining_cents + candy_bars * (candy_bar_cost_dimes * dime_value) + 
  lollipop_cost_quarters * quarter_value :=
sorry

end NUMINAMATH_CALUDE_sam_candy_bars_l2937_293715


namespace NUMINAMATH_CALUDE_divisibility_by_1989_l2937_293738

theorem divisibility_by_1989 (n : ℕ) : ∃ k : ℤ, 
  13 * (-50)^n + 17 * 40^n - 30 = 1989 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1989_l2937_293738


namespace NUMINAMATH_CALUDE_arithmetic_operations_l2937_293729

theorem arithmetic_operations : 
  (12 - (-18) + (-7) - 20 = 3) ∧ 
  (-4 / (1/2) * 8 = -64) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l2937_293729


namespace NUMINAMATH_CALUDE_min_value_expression_l2937_293722

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 2 + 24 / (n : ℝ) ≥ 7 ∧ 
  ∃ m : ℕ+, (m : ℝ) / 2 + 24 / (m : ℝ) = 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2937_293722


namespace NUMINAMATH_CALUDE_apple_slices_l2937_293786

theorem apple_slices (S : ℕ) : 
  S > 0 ∧ 
  (S / 16 : ℚ) * S = 5 → 
  S = 16 := by
sorry

end NUMINAMATH_CALUDE_apple_slices_l2937_293786


namespace NUMINAMATH_CALUDE_charity_event_total_is_1080_l2937_293792

/-- Represents the total money raised from a charity event with raffle ticket sales and donations -/
def charity_event_total (a_price b_price c_price : ℚ) 
                        (a_sold b_sold c_sold : ℕ) 
                        (donations : List ℚ) : ℚ :=
  a_price * a_sold + b_price * b_sold + c_price * c_sold + donations.sum

/-- Theorem stating the total money raised from the charity event -/
theorem charity_event_total_is_1080 : 
  charity_event_total 3 5.5 10 100 50 25 [30, 30, 50, 45, 100] = 1080 := by
  sorry

end NUMINAMATH_CALUDE_charity_event_total_is_1080_l2937_293792


namespace NUMINAMATH_CALUDE_scale_multiplication_l2937_293740

theorem scale_multiplication (a b c : ℝ) (h : a * b = c) :
  (a / 100) * (b / 100) = c / 10000 := by
  sorry

end NUMINAMATH_CALUDE_scale_multiplication_l2937_293740


namespace NUMINAMATH_CALUDE_prob_one_heads_is_half_l2937_293768

/-- A coin toss outcome -/
inductive CoinToss
| Heads
| Tails

/-- Result of two successive coin tosses -/
def TwoTosses := (CoinToss × CoinToss)

/-- All possible outcomes of two successive coin tosses -/
def allOutcomes : Finset TwoTosses := sorry

/-- Outcomes with exactly one heads -/
def oneHeadsOutcomes : Finset TwoTosses := sorry

/-- Probability of an event in a finite sample space -/
def probability (event : Finset TwoTosses) : ℚ :=
  (event.card : ℚ) / (allOutcomes.card : ℚ)

theorem prob_one_heads_is_half :
  probability oneHeadsOutcomes = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_one_heads_is_half_l2937_293768


namespace NUMINAMATH_CALUDE_company_workers_count_l2937_293778

/-- Represents the hierarchical structure of a company -/
structure CompanyHierarchy where
  workers_per_lead : ℕ
  leads_per_supervisor : ℕ
  num_supervisors : ℕ

/-- Calculates the number of workers in a company given its hierarchical structure -/
def calculate_workers (ch : CompanyHierarchy) : ℕ :=
  ch.num_supervisors * ch.leads_per_supervisor * ch.workers_per_lead

/-- Theorem stating that a company with the given hierarchical structure and 13 supervisors has 390 workers -/
theorem company_workers_count :
  let ch : CompanyHierarchy := {
    workers_per_lead := 10,
    leads_per_supervisor := 3,
    num_supervisors := 13
  }
  calculate_workers ch = 390 := by sorry

end NUMINAMATH_CALUDE_company_workers_count_l2937_293778


namespace NUMINAMATH_CALUDE_land_plot_side_length_l2937_293793

theorem land_plot_side_length (area : ℝ) (side : ℝ) : 
  area = 1600 → side * side = area → side = 40 := by
  sorry

end NUMINAMATH_CALUDE_land_plot_side_length_l2937_293793


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l2937_293790

/-- The sum of digits of a natural number in base 10 -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digitSum (n / 10)

/-- The number 4444^444 -/
def bigNumber : ℕ := 4444^444

/-- A is the sum of digits of bigNumber -/
def A : ℕ := digitSum bigNumber

/-- B is the sum of digits of A -/
def B : ℕ := digitSum A

theorem sum_of_digits_of_B_is_seven : digitSum B = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l2937_293790


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l2937_293703

theorem quadratic_equation_condition (a : ℝ) : 
  (∀ x, ∃ p q r : ℝ, (a + 4) * x^(a^2 - 14) - 3 * x + 8 = p * x^2 + q * x + r) ∧ 
  (a + 4 ≠ 0) → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l2937_293703


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2937_293748

theorem max_value_quadratic (p q : ℝ) : 
  q = p - 2 → 
  ∃ (max : ℝ), max = 26 + 2/3 ∧ 
  ∀ (p : ℝ), -3 * p^2 + 24 * p - 50 + 10 * q ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2937_293748


namespace NUMINAMATH_CALUDE_sum_difference_is_thirteen_l2937_293772

def star_list : List Nat := List.range 30

def replace_three_with_two (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list : List Nat :=
  star_list.map replace_three_with_two

theorem sum_difference_is_thirteen :
  star_list.sum - emilio_list.sum = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_is_thirteen_l2937_293772


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_l2937_293775

/-- A piecewise function f(x) defined by three parts -/
noncomputable def f (a c : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then 2*a*x + 6
  else if -2 ≤ x ∧ x ≤ 2 then 3*x - 2
  else 4*x + 2*c

/-- The theorem stating that if f is continuous, then a + c = -1/2 -/
theorem continuous_piecewise_function (a c : ℝ) :
  Continuous (f a c) → a + c = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_l2937_293775


namespace NUMINAMATH_CALUDE_parsley_sprigs_remaining_l2937_293750

/-- Calculates the number of parsley sprigs remaining after decorating plates. -/
theorem parsley_sprigs_remaining 
  (initial_sprigs : ℕ) 
  (whole_sprig_plates : ℕ) 
  (half_sprig_plates : ℕ) : 
  initial_sprigs = 25 → 
  whole_sprig_plates = 8 → 
  half_sprig_plates = 12 → 
  initial_sprigs - (whole_sprig_plates + half_sprig_plates / 2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_parsley_sprigs_remaining_l2937_293750


namespace NUMINAMATH_CALUDE_max_volume_right_prism_l2937_293736

/-- 
Given a right prism with triangular bases where:
- Base triangle sides are a, b, b
- a = 2b
- Angle between sides a and b is π/2
- Sum of areas of two lateral faces and one base is 30
The maximum volume of the prism is 2.5√5.
-/
theorem max_volume_right_prism (a b h : ℝ) :
  a = 2 * b →
  4 * b * h + b^2 = 30 →
  (∀ h' : ℝ, 4 * b * h' + b^2 = 30 → b^2 * h / 2 ≤ b^2 * h' / 2) →
  b^2 * h / 2 = 2.5 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_right_prism_l2937_293736


namespace NUMINAMATH_CALUDE_pencils_per_row_l2937_293742

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (h1 : total_pencils = 30) (h2 : num_rows = 6) :
  total_pencils / num_rows = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l2937_293742


namespace NUMINAMATH_CALUDE_backyard_length_is_20_l2937_293737

-- Define the backyard and shed dimensions
def backyard_width : ℝ := 13
def shed_length : ℝ := 3
def shed_width : ℝ := 5
def sod_area : ℝ := 245

-- Theorem statement
theorem backyard_length_is_20 :
  ∃ (L : ℝ), L * backyard_width - shed_length * shed_width = sod_area ∧ L = 20 := by
  sorry

end NUMINAMATH_CALUDE_backyard_length_is_20_l2937_293737


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l2937_293782

/-- Given a line l with parametric equations x = 4 - 4t and y = -2 + 3t, where t ∈ ℝ,
    the y-intercept of line l is 1. -/
theorem y_intercept_of_line (l : Set (ℝ × ℝ)) : 
  (∀ t : ℝ, (4 - 4*t, -2 + 3*t) ∈ l) → 
  (0, 1) ∈ l := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l2937_293782


namespace NUMINAMATH_CALUDE_projection_of_A_on_Oxz_l2937_293765

/-- The projection of a point (x, y, z) onto the Oxz plane is (x, 0, z) -/
def proj_oxz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, 0, p.2.2)

/-- Point A in 3D space -/
def A : ℝ × ℝ × ℝ := (2, 3, 6)

/-- Point B is the projection of A onto the Oxz plane -/
def B : ℝ × ℝ × ℝ := proj_oxz A

theorem projection_of_A_on_Oxz :
  B = (2, 0, 6) := by sorry

end NUMINAMATH_CALUDE_projection_of_A_on_Oxz_l2937_293765


namespace NUMINAMATH_CALUDE_daria_savings_weeks_l2937_293796

/-- The number of weeks required for Daria to save enough money for a vacuum cleaner. -/
def weeks_to_save (initial_savings : ℕ) (weekly_contribution : ℕ) (vacuum_cost : ℕ) : ℕ :=
  ((vacuum_cost - initial_savings) + weekly_contribution - 1) / weekly_contribution

/-- Theorem: Daria needs 10 weeks to save for the vacuum cleaner. -/
theorem daria_savings_weeks : weeks_to_save 20 10 120 = 10 := by
  sorry

end NUMINAMATH_CALUDE_daria_savings_weeks_l2937_293796


namespace NUMINAMATH_CALUDE_citizenship_test_study_time_l2937_293712

/-- Calculates the total study time in hours for a citizenship test -/
theorem citizenship_test_study_time :
  let total_questions : ℕ := 60
  let multiple_choice_questions : ℕ := 30
  let fill_in_blank_questions : ℕ := 30
  let multiple_choice_time : ℕ := 15  -- minutes per question
  let fill_in_blank_time : ℕ := 25    -- minutes per question
  
  total_questions = multiple_choice_questions + fill_in_blank_questions →
  (multiple_choice_questions * multiple_choice_time + fill_in_blank_questions * fill_in_blank_time) / 60 = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_citizenship_test_study_time_l2937_293712


namespace NUMINAMATH_CALUDE_divisibility_by_a_squared_l2937_293721

theorem divisibility_by_a_squared (a : ℤ) (n : ℕ) :
  ∃ k : ℤ, (a * n - 1) * (a + 1)^n + 1 = a^2 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_a_squared_l2937_293721


namespace NUMINAMATH_CALUDE_minimum_at_two_implies_a_twelve_l2937_293788

/-- Given a function f(x) = x^3 - ax, prove that if f takes its minimum value at x = 2, then a = 12 -/
theorem minimum_at_two_implies_a_twelve (a : ℝ) : 
  (∀ x : ℝ, x^3 - a*x ≥ 2^3 - a*2) → a = 12 := by
  sorry

end NUMINAMATH_CALUDE_minimum_at_two_implies_a_twelve_l2937_293788


namespace NUMINAMATH_CALUDE_inequality_condition_l2937_293758

theorem inequality_condition (x y : ℝ) : 
  y - x < Real.sqrt (x^2 + 4*x*y) ↔ 
  ((y < x + Real.sqrt (x^2 + 4*x*y) ∨ y < x - Real.sqrt (x^2 + 4*x*y)) ∧ x*(x + 4*y) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l2937_293758


namespace NUMINAMATH_CALUDE_model2_best_fit_l2937_293771

/-- Represents a regression model with its coefficient of determination -/
structure RegressionModel where
  name : String
  r_squared : Float

/-- Determines if a model has the best fitting effect among a list of models -/
def has_best_fit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

/-- The list of regression models with their R² values -/
def regression_models : List RegressionModel := [
  ⟨"Model 1", 0.78⟩,
  ⟨"Model 2", 0.85⟩,
  ⟨"Model 3", 0.61⟩,
  ⟨"Model 4", 0.31⟩
]

/-- Theorem stating that Model 2 has the best fitting effect -/
theorem model2_best_fit :
  ∃ model ∈ regression_models, model.name = "Model 2" ∧ has_best_fit model regression_models :=
by
  sorry

end NUMINAMATH_CALUDE_model2_best_fit_l2937_293771


namespace NUMINAMATH_CALUDE_min_value_h_negative_reals_l2937_293704

-- Define odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the theorem
theorem min_value_h_negative_reals 
  (f g : ℝ → ℝ) 
  (hf : IsOdd f) 
  (hg : IsOdd g) 
  (h : ℝ → ℝ) 
  (h_def : ∀ x, h x = f x + g x - 2) 
  (h_max : ∃ M, M = 6 ∧ ∀ x > 0, h x ≤ M) :
  ∃ m, m = -10 ∧ ∀ x < 0, h x ≥ m := by
sorry

end NUMINAMATH_CALUDE_min_value_h_negative_reals_l2937_293704


namespace NUMINAMATH_CALUDE_twenty_fifth_in_base5_l2937_293751

/-- Converts a natural number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Checks if a list of digits is a valid base 5 number -/
def isValidBase5 (l : List ℕ) : Prop :=
  l.all (· < 5)

theorem twenty_fifth_in_base5 :
  let base5Repr := toBase5 25
  isValidBase5 base5Repr ∧ base5Repr = [1, 0, 0] := by sorry

end NUMINAMATH_CALUDE_twenty_fifth_in_base5_l2937_293751


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2937_293730

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → c = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2937_293730


namespace NUMINAMATH_CALUDE_f_decreasing_on_positive_reals_l2937_293705

/-- The function f(x) = -x^2 + 3 is decreasing on the interval (0, +∞) -/
theorem f_decreasing_on_positive_reals :
  ∀ (x₁ x₂ : ℝ), 0 < x₁ → x₁ < x₂ → (-x₁^2 + 3) > (-x₂^2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_on_positive_reals_l2937_293705


namespace NUMINAMATH_CALUDE_exactly_two_base_pairs_l2937_293747

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4 else (a * x^2 + a * x - 1) / x

-- Define what it means for two points to be symmetric about the origin
def symmetricAboutOrigin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

-- Define a base pair
def basePair (a : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  symmetricAboutOrigin p1 p2 ∧ p1.2 = f a p1.1 ∧ p2.2 = f a p2.1

-- The main theorem
theorem exactly_two_base_pairs (a : ℝ) : 
  (∃ p1 p2 p3 p4 : ℝ × ℝ, 
    basePair a p1 p2 ∧ basePair a p3 p4 ∧ 
    p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧
    (∀ p5 p6 : ℝ × ℝ, basePair a p5 p6 → (p5 = p1 ∧ p6 = p2) ∨ (p5 = p3 ∧ p6 = p4) ∨ 
                                         (p5 = p2 ∧ p6 = p1) ∨ (p5 = p4 ∧ p6 = p3))) ↔ 
  a > -6 + 2 * Real.sqrt 6 ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_base_pairs_l2937_293747


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt2_over_2_l2937_293727

theorem cos_sin_sum_equals_sqrt2_over_2 : 
  Real.cos (80 * π / 180) * Real.cos (35 * π / 180) + 
  Real.sin (80 * π / 180) * Real.cos (55 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt2_over_2_l2937_293727


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2937_293743

/-- Represents the repeating decimal 0.37246̅ -/
def repeating_decimal : ℚ := 37246 / 100000 + (246 / 100000) / (1 - 1/1000)

/-- The fraction we want to prove equality with -/
def target_fraction : ℚ := 37187378 / 99900

/-- Theorem stating that the repeating decimal is equal to the target fraction -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2937_293743


namespace NUMINAMATH_CALUDE_soccer_team_wins_solution_l2937_293719

def soccer_team_wins (total_games wins losses draws : ℕ) : Prop :=
  total_games = wins + losses + draws ∧
  losses = 2 ∧
  3 * wins + draws = 46

theorem soccer_team_wins_solution :
  ∃ (wins losses draws : ℕ),
    soccer_team_wins 20 wins losses draws ∧ wins = 14 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_wins_solution_l2937_293719


namespace NUMINAMATH_CALUDE_sequence_position_l2937_293739

/-- The general term of the sequence -/
def sequenceTerm (n : ℕ) : ℚ := (n + 3) / (n + 1)

/-- The position we want to prove -/
def position : ℕ := 14

/-- The fraction we're looking for -/
def targetFraction : ℚ := 17 / 15

theorem sequence_position :
  sequenceTerm position = targetFraction := by sorry

end NUMINAMATH_CALUDE_sequence_position_l2937_293739


namespace NUMINAMATH_CALUDE_triangle_inequality_l2937_293724

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  a * (b^2 + c^2 - a^2) + b * (c^2 + a^2 - b^2) + c * (a^2 + b^2 - c^2) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2937_293724


namespace NUMINAMATH_CALUDE_intersection_chord_length_l2937_293753

theorem intersection_chord_length :
  let line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1}
  let circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 8}
  let intersection := line ∩ circle
  ∃ (A B : ℝ × ℝ), A ∈ intersection ∧ B ∈ intersection ∧ A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l2937_293753


namespace NUMINAMATH_CALUDE_geography_english_sum_l2937_293745

/-- Represents Henry's test scores -/
structure TestScores where
  geography : ℝ
  math : ℝ
  english : ℝ
  history : ℝ

/-- Henry's test scores satisfy the given conditions -/
def satisfiesConditions (scores : TestScores) : Prop :=
  scores.math = 70 ∧
  scores.history = (scores.geography + scores.math + scores.english) / 3 ∧
  scores.geography + scores.math + scores.english + scores.history = 248

/-- The sum of Henry's Geography and English scores is 116 -/
theorem geography_english_sum (scores : TestScores) 
  (h : satisfiesConditions scores) : scores.geography + scores.english = 116 := by
  sorry

end NUMINAMATH_CALUDE_geography_english_sum_l2937_293745


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l2937_293760

theorem quadratic_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 5*x₁ + 15 = x₁ + 55) ∧
  (x₂^2 - 5*x₂ + 15 = x₂ + 55) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l2937_293760


namespace NUMINAMATH_CALUDE_range_of_a_l2937_293744

theorem range_of_a (x y z a : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y + z = 1) (heq : a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) :
  0 < a ∧ a ≤ 7 / 27 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2937_293744


namespace NUMINAMATH_CALUDE_trapezoid_longer_base_l2937_293701

/-- Represents a trapezoid with bases and height forming an arithmetic progression -/
structure ArithmeticTrapezoid where
  shorter_base : ℝ
  altitude : ℝ
  longer_base : ℝ
  is_arithmetic_progression : 
    longer_base - altitude = altitude - shorter_base
  area_formula : 
    (shorter_base + longer_base) * altitude / 2 = 63

/-- Theorem: Given a trapezoid with specific measurements, prove its longer base length -/
theorem trapezoid_longer_base 
  (t : ArithmeticTrapezoid) 
  (h1 : t.shorter_base = 5) 
  (h2 : t.altitude = 7) : 
  t.longer_base = 13 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_longer_base_l2937_293701


namespace NUMINAMATH_CALUDE_jolyn_older_than_clarisse_l2937_293726

/-- Represents an age difference in months and days -/
structure AgeDifference where
  months : ℕ
  days : ℕ

/-- Adds two age differences -/
def addAgeDifference (ad1 ad2 : AgeDifference) : AgeDifference :=
  { months := ad1.months + ad2.months + (ad1.days + ad2.days) / 30,
    days := (ad1.days + ad2.days) % 30 }

/-- Subtracts two age differences -/
def subtractAgeDifference (ad1 ad2 : AgeDifference) : AgeDifference :=
  { months := ad1.months - ad2.months - (if ad1.days < ad2.days then 1 else 0),
    days := if ad1.days < ad2.days then ad1.days + 30 - ad2.days else ad1.days - ad2.days }

theorem jolyn_older_than_clarisse
  (jolyn_therese : AgeDifference)
  (therese_aivo : AgeDifference)
  (leon_aivo : AgeDifference)
  (clarisse_leon : AgeDifference)
  (h1 : jolyn_therese = { months := 2, days := 10 })
  (h2 : therese_aivo = { months := 5, days := 15 })
  (h3 : leon_aivo = { months := 2, days := 25 })
  (h4 : clarisse_leon = { months := 3, days := 20 })
  : subtractAgeDifference (addAgeDifference jolyn_therese therese_aivo)
                          (addAgeDifference clarisse_leon leon_aivo)
    = { months := 1, days := 10 } := by
  sorry


end NUMINAMATH_CALUDE_jolyn_older_than_clarisse_l2937_293726


namespace NUMINAMATH_CALUDE_orangeade_pricing_l2937_293728

/-- Represents the amount of orange juice used each day -/
def orange_juice : ℝ := sorry

/-- Represents the amount of water used on the first day -/
def water : ℝ := sorry

/-- The price per glass on the first day -/
def price_day1 : ℝ := 0.60

/-- The price per glass on the third day -/
def price_day3 : ℝ := sorry

/-- The volume of orangeade on the first day -/
def volume_day1 : ℝ := orange_juice + water

/-- The volume of orangeade on the second day -/
def volume_day2 : ℝ := orange_juice + 2 * water

/-- The volume of orangeade on the third day -/
def volume_day3 : ℝ := orange_juice + 3 * water

theorem orangeade_pricing :
  (orange_juice > 0) →
  (water > 0) →
  (orange_juice = water) →
  (price_day1 * volume_day1 = price_day3 * volume_day3) →
  (price_day3 = price_day1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_orangeade_pricing_l2937_293728


namespace NUMINAMATH_CALUDE_line_points_product_l2937_293779

theorem line_points_product (x y : ℝ) : 
  (∃ k : Set (ℝ × ℝ), 
    (∀ p : ℝ × ℝ, p ∈ k ↔ p.2 = (1/4) * p.1) ∧ 
    (x, 8) ∈ k ∧ 
    (20, y) ∈ k ∧ 
    x * y = 160) → 
  y = 5 := by
sorry

end NUMINAMATH_CALUDE_line_points_product_l2937_293779


namespace NUMINAMATH_CALUDE_fraction_equality_l2937_293735

theorem fraction_equality : (5 * 7) / 8 = 4 + 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2937_293735


namespace NUMINAMATH_CALUDE_password_letters_count_l2937_293799

theorem password_letters_count : ∃ (n : ℕ), 
  (n ^ 4 : ℕ) - n * (n - 1) * (n - 2) * (n - 3) = 936 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_password_letters_count_l2937_293799


namespace NUMINAMATH_CALUDE_expenses_calculation_l2937_293731

/-- Represents the revenue allocation ratio -/
structure RevenueRatio :=
  (employee_salaries : ℕ)
  (stock_purchases : ℕ)
  (rent : ℕ)
  (marketing_costs : ℕ)

/-- Calculates the total amount spent on employee salaries, rent, and marketing costs -/
def calculate_expenses (revenue : ℕ) (ratio : RevenueRatio) : ℕ :=
  let total_ratio := ratio.employee_salaries + ratio.stock_purchases + ratio.rent + ratio.marketing_costs
  let unit_value := revenue / total_ratio
  (ratio.employee_salaries + ratio.rent + ratio.marketing_costs) * unit_value

/-- Theorem stating that the calculated expenses for the given revenue and ratio equal $7,800 -/
theorem expenses_calculation (revenue : ℕ) (ratio : RevenueRatio) :
  revenue = 10800 ∧ 
  ratio = { employee_salaries := 3, stock_purchases := 5, rent := 2, marketing_costs := 8 } →
  calculate_expenses revenue ratio = 7800 :=
by sorry

end NUMINAMATH_CALUDE_expenses_calculation_l2937_293731


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2937_293759

theorem geometric_arithmetic_sequence_ratio 
  (x y z r : ℝ) 
  (h1 : y = r * x) 
  (h2 : z = r * y) 
  (h3 : x ≠ y) 
  (h4 : 2 * (2 * y) = x + 3 * z) : 
  r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2937_293759


namespace NUMINAMATH_CALUDE_odd_integers_sum_13_to_45_l2937_293732

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  (a₁ + aₙ) * n / 2

theorem odd_integers_sum_13_to_45 :
  arithmetic_sum 13 45 2 = 493 := by
  sorry

end NUMINAMATH_CALUDE_odd_integers_sum_13_to_45_l2937_293732


namespace NUMINAMATH_CALUDE_painter_week_total_l2937_293720

/-- Represents the painter's work schedule and productivity --/
structure PainterSchedule where
  monday_speed : ℝ
  normal_speed : ℝ
  friday_speed : ℝ
  normal_hours : ℝ
  friday_hours : ℝ
  friday_monday_diff : ℝ

/-- Calculates the total length of fence painted over the week --/
def total_painted (schedule : PainterSchedule) : ℝ :=
  let monday_length := schedule.monday_speed * schedule.normal_hours
  let normal_day_length := schedule.normal_speed * schedule.normal_hours
  let friday_length := schedule.friday_speed * schedule.friday_hours
  monday_length + 3 * normal_day_length + friday_length

/-- Theorem stating the total length of fence painted over the week --/
theorem painter_week_total (schedule : PainterSchedule)
  (h1 : schedule.monday_speed = 0.5 * schedule.normal_speed)
  (h2 : schedule.friday_speed = 2 * schedule.normal_speed)
  (h3 : schedule.friday_hours = 6)
  (h4 : schedule.normal_hours = 8)
  (h5 : schedule.friday_speed * schedule.friday_hours - 
        schedule.monday_speed * schedule.normal_hours = schedule.friday_monday_diff)
  (h6 : schedule.friday_monday_diff = 300) :
  total_painted schedule = 1500 := by
  sorry


end NUMINAMATH_CALUDE_painter_week_total_l2937_293720


namespace NUMINAMATH_CALUDE_candies_added_l2937_293707

theorem candies_added (initial_candies final_candies : ℕ) (h1 : initial_candies = 6) (h2 : final_candies = 10) :
  final_candies - initial_candies = 4 := by
  sorry

end NUMINAMATH_CALUDE_candies_added_l2937_293707


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l2937_293781

theorem triangle_angle_sum (A B C : ℝ) (h1 : A = 75) (h2 : B = 40) : C = 65 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l2937_293781


namespace NUMINAMATH_CALUDE_vasyas_numbers_l2937_293716

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 := by
sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l2937_293716


namespace NUMINAMATH_CALUDE_x_minus_y_equals_three_l2937_293702

theorem x_minus_y_equals_three (x y : ℝ) 
  (h1 : x + y = 8) 
  (h2 : x^2 - y^2 = 24) : 
  x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_three_l2937_293702


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2937_293755

theorem inequality_solution_set (x : ℝ) : 2 * x + 3 ≤ 1 ↔ x ≤ -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2937_293755


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2937_293733

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if 2x - √3y = 0 is one of its asymptotes, then its eccentricity is √21/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → 2*x - Real.sqrt 3*y = 0) →
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 21 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2937_293733


namespace NUMINAMATH_CALUDE_callie_caught_seven_frogs_l2937_293773

def alster_frogs : ℕ := 2

def quinn_frogs (alster : ℕ) : ℕ := 2 * alster

def bret_frogs (quinn : ℕ) : ℕ := 3 * quinn

def callie_frogs (bret : ℕ) : ℕ := (5 * bret) / 8

theorem callie_caught_seven_frogs :
  callie_frogs (bret_frogs (quinn_frogs alster_frogs)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_callie_caught_seven_frogs_l2937_293773


namespace NUMINAMATH_CALUDE_two_tangent_lines_l2937_293770

/-- A line that intersects a parabola at exactly one point -/
structure TangentLine where
  slope : ℝ
  y_intercept : ℝ

/-- The parabola y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The point M(2, 4) -/
def M : Point := ⟨2, 4⟩

/-- A line passes through a point -/
def passes_through (l : TangentLine) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

/-- A line intersects the parabola at exactly one point -/
def intersects_once (l : TangentLine) : Prop :=
  ∃! (p : Point), passes_through l p ∧ parabola p.x p.y

/-- There are exactly two lines passing through M(2, 4) that intersect the parabola at exactly one point -/
theorem two_tangent_lines : ∃! (l1 l2 : TangentLine), 
  l1 ≠ l2 ∧ 
  passes_through l1 M ∧ 
  passes_through l2 M ∧ 
  intersects_once l1 ∧ 
  intersects_once l2 :=
sorry

end NUMINAMATH_CALUDE_two_tangent_lines_l2937_293770


namespace NUMINAMATH_CALUDE_no_solution_system_l2937_293708

theorem no_solution_system :
  ¬∃ (x y : ℝ), 
    (x^3 + x + y + 1 = 0) ∧ 
    (y*x^2 + x + y = 0) ∧ 
    (y^2 + y - x^2 + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_system_l2937_293708


namespace NUMINAMATH_CALUDE_wage_payment_period_l2937_293791

/-- Given a sum of money that can pay three workers' wages for different periods,
    prove that it can pay their combined wages for a specific period when working together. -/
theorem wage_payment_period (M : ℝ) (p q r : ℝ) : 
  M = 24 * p ∧ M = 40 * q ∧ M = 30 * r → M = 10 * (p + q + r) := by
sorry

end NUMINAMATH_CALUDE_wage_payment_period_l2937_293791


namespace NUMINAMATH_CALUDE_ninth_row_fourth_number_l2937_293749

def row_start (i : ℕ) : ℕ := 2 + 4 * 6 * (i - 1)

def fourth_number (i : ℕ) : ℕ := row_start i + 4 * 3

theorem ninth_row_fourth_number :
  fourth_number 9 = 206 :=
by sorry

end NUMINAMATH_CALUDE_ninth_row_fourth_number_l2937_293749


namespace NUMINAMATH_CALUDE_remainder_70_div_17_l2937_293789

theorem remainder_70_div_17 : 70 % 17 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_70_div_17_l2937_293789


namespace NUMINAMATH_CALUDE_original_price_calculation_l2937_293723

theorem original_price_calculation (P : ℝ) : 
  (P * (1 - 0.3) * (1 - 0.2) = 1120) → P = 2000 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2937_293723


namespace NUMINAMATH_CALUDE_triangle_angle_matrix_det_zero_l2937_293752

/-- The determinant of a specific matrix formed by angles of a triangle is zero -/
theorem triangle_angle_matrix_det_zero (A B C : Real) 
  (h : A + B + C = Real.pi) : -- Condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.exp A, Real.exp (-A), 1],
    ![Real.exp B, Real.exp (-B), 1],
    ![Real.exp C, Real.exp (-C), 1]
  ]
  Matrix.det M = 0 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_matrix_det_zero_l2937_293752


namespace NUMINAMATH_CALUDE_expression_lower_bound_l2937_293725

theorem expression_lower_bound (a : ℝ) (h : a > 1) :
  a + 4 / (a - 1) ≥ 5 ∧ (a + 4 / (a - 1) = 5 ↔ a = 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_lower_bound_l2937_293725


namespace NUMINAMATH_CALUDE_quadratic_roots_greater_than_one_implies_s_positive_l2937_293787

theorem quadratic_roots_greater_than_one_implies_s_positive
  (b c : ℝ)
  (h1 : ∃ x y : ℝ, x > 1 ∧ y > 1 ∧ x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0)
  : b + c + 1 > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_greater_than_one_implies_s_positive_l2937_293787


namespace NUMINAMATH_CALUDE_auction_bid_ratio_l2937_293741

/-- Auction problem statement -/
theorem auction_bid_ratio :
  -- Auction starts at $300
  let start_price : ℕ := 300
  -- Harry's first bid adds $200 to the starting value
  let harry_first_bid : ℕ := start_price + 200
  -- A third bidder adds three times Harry's bid
  let third_bid : ℕ := harry_first_bid + 3 * harry_first_bid
  -- Harry's final bid is $4,000
  let harry_final_bid : ℕ := 4000
  -- Harry's final bid exceeded the third bidder's bid by $1500
  let third_bid_final : ℕ := harry_final_bid - 1500
  -- Calculate the second bidder's bid
  let second_bid : ℕ := third_bid_final - 3 * harry_first_bid
  -- The ratio of the second bidder's bid to Harry's first bid is 2:1
  second_bid / harry_first_bid = 2 := by
  sorry

end NUMINAMATH_CALUDE_auction_bid_ratio_l2937_293741


namespace NUMINAMATH_CALUDE_coincident_centers_of_inscribed_ngons_l2937_293706

/-- A regular n-gon in a 2D plane. -/
structure RegularNGon where
  n : ℕ
  center : ℝ × ℝ
  radius : ℝ
  rotation : ℝ  -- Rotation angle of the first vertex

/-- The vertices of a regular n-gon. -/
def vertices (ngon : RegularNGon) : Finset (ℝ × ℝ) :=
  sorry

/-- Predicate to check if a point lies on the perimeter of an n-gon. -/
def on_perimeter (point : ℝ × ℝ) (ngon : RegularNGon) : Prop :=
  sorry

/-- Theorem: If the vertices of one regular n-gon lie on the perimeter of another,
    their centers coincide (for n ≥ 4). -/
theorem coincident_centers_of_inscribed_ngons
  (n : ℕ)
  (h_n : n ≥ 4)
  (ngon1 ngon2 : RegularNGon)
  (h_same_n : ngon1.n = n ∧ ngon2.n = n)
  (h_inscribed : ∀ v ∈ vertices ngon1, on_perimeter v ngon2) :
  ngon1.center = ngon2.center :=
sorry

end NUMINAMATH_CALUDE_coincident_centers_of_inscribed_ngons_l2937_293706


namespace NUMINAMATH_CALUDE_meeting_participants_l2937_293761

theorem meeting_participants :
  ∀ (F M : ℕ),
  F > 0 ∧ M > 0 →
  F / 2 + M / 4 = (F + M) / 3 →
  F / 2 = 110 →
  F + M = 330 :=
by
  sorry

end NUMINAMATH_CALUDE_meeting_participants_l2937_293761


namespace NUMINAMATH_CALUDE_factory_employee_increase_l2937_293774

theorem factory_employee_increase (initial_employees : ℕ) (increase_percentage : ℚ) 
  (h1 : initial_employees = 852)
  (h2 : increase_percentage = 25 / 100) :
  initial_employees + (increase_percentage * initial_employees).floor = 1065 := by
  sorry

end NUMINAMATH_CALUDE_factory_employee_increase_l2937_293774
