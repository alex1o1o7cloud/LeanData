import Mathlib

namespace NUMINAMATH_CALUDE_sine_shift_l2412_241287

theorem sine_shift (x : ℝ) : 3 * Real.sin (2 * x + π / 5) = 3 * Real.sin (2 * (x + π / 10)) := by
  sorry

end NUMINAMATH_CALUDE_sine_shift_l2412_241287


namespace NUMINAMATH_CALUDE_mike_office_visits_l2412_241245

/-- The number of pull-ups Mike does each time he enters his office -/
def pull_ups_per_visit : ℕ := 2

/-- The number of pull-ups Mike does in a week -/
def pull_ups_per_week : ℕ := 70

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of times Mike goes into his office each day -/
def office_visits_per_day : ℕ := 5

theorem mike_office_visits :
  office_visits_per_day * days_per_week * pull_ups_per_visit = pull_ups_per_week :=
by sorry

end NUMINAMATH_CALUDE_mike_office_visits_l2412_241245


namespace NUMINAMATH_CALUDE_smallest_common_multiple_13_8_lcm_13_8_l2412_241221

theorem smallest_common_multiple_13_8 : 
  ∀ n : ℕ, (13 ∣ n ∧ 8 ∣ n) → n ≥ 104 := by
  sorry

theorem lcm_13_8 : Nat.lcm 13 8 = 104 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_13_8_lcm_13_8_l2412_241221


namespace NUMINAMATH_CALUDE_probability_in_standard_deck_l2412_241246

/-- Represents a standard deck of 52 cards -/
structure Deck :=
  (total_cards : Nat)
  (diamonds : Nat)
  (spades : Nat)
  (hearts : Nat)

/-- The probability of drawing a diamond, then a spade, then a heart from a standard deck -/
def probability_diamond_spade_heart (d : Deck) : Rat :=
  (d.diamonds : Rat) / d.total_cards *
  (d.spades : Rat) / (d.total_cards - 1) *
  (d.hearts : Rat) / (d.total_cards - 2)

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52
  , diamonds := 13
  , spades := 13
  , hearts := 13 }

theorem probability_in_standard_deck :
  probability_diamond_spade_heart standard_deck = 13 / 780 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_standard_deck_l2412_241246


namespace NUMINAMATH_CALUDE_equation_solution_l2412_241252

theorem equation_solution (x : ℝ) :
  3 / (x - 3) + 5 / (2 * x - 6) = 11 / 2 →
  2 * x - 6 = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2412_241252


namespace NUMINAMATH_CALUDE_compare_expressions_l2412_241268

theorem compare_expressions (x y : ℝ) (h1 : x < y) (h2 : y < 0) :
  (x^2 + y^2) * (x - y) > (x^2 - y^2) * (x + y) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l2412_241268


namespace NUMINAMATH_CALUDE_group_sizes_min_group_a_size_l2412_241227

/-- Represents the ticket price based on the number of people -/
def ticket_price (m : ℕ) : ℕ :=
  if 10 ≤ m ∧ m ≤ 50 then 60
  else if 51 ≤ m ∧ m ≤ 100 then 50
  else 40

/-- The total number of people in both groups -/
def total_people : ℕ := 102

/-- The total amount paid when buying tickets separately -/
def total_amount : ℕ := 5580

/-- Theorem stating the number of people in each group -/
theorem group_sizes :
  ∃ (a b : ℕ), a < 50 ∧ b > 50 ∧ a + b = total_people ∧
  ticket_price a * a + ticket_price b * b = total_amount :=
sorry

/-- Theorem stating the minimum number of people in Group A for savings -/
theorem min_group_a_size :
  ∃ (min_a : ℕ), ∀ a : ℕ, a ≥ min_a →
  ticket_price a * a + ticket_price (total_people - a) * (total_people - a) - 
  ticket_price total_people * total_people ≥ 1200 :=
sorry

end NUMINAMATH_CALUDE_group_sizes_min_group_a_size_l2412_241227


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_for_x_gt_e_l2412_241271

theorem necessary_not_sufficient_condition_for_x_gt_e (x : ℝ) :
  (x > Real.exp 1 → x > 1) ∧ ¬(x > 1 → x > Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_for_x_gt_e_l2412_241271


namespace NUMINAMATH_CALUDE_sum_of_digits_45_times_40_l2412_241213

def product_45_40 : ℕ := 45 * 40

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_45_times_40 : sum_of_digits product_45_40 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_45_times_40_l2412_241213


namespace NUMINAMATH_CALUDE_corn_farmer_profit_l2412_241273

/-- Calculates the profit for a corn farmer given specific conditions. -/
theorem corn_farmer_profit : 
  let seeds_per_ear : ℕ := 4
  let price_per_ear : ℚ := 1/10
  let seeds_per_bag : ℕ := 100
  let price_per_bag : ℚ := 1/2
  let ears_sold : ℕ := 500
  let total_seeds : ℕ := seeds_per_ear * ears_sold
  let bags_needed : ℕ := (total_seeds + seeds_per_bag - 1) / seeds_per_bag
  let total_cost : ℚ := bags_needed * price_per_bag
  let total_revenue : ℚ := ears_sold * price_per_ear
  let profit : ℚ := total_revenue - total_cost
  profit = 40 := by sorry

end NUMINAMATH_CALUDE_corn_farmer_profit_l2412_241273


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2412_241255

/-- Given a geometric sequence with common ratio q > 0 and T_n as the product of the first n terms,
    if T_7 > T_6 > T_8, then 0 < q < 1 and T_13 > 1 > T_14 -/
theorem geometric_sequence_properties (q : ℝ) (T : ℕ → ℝ) 
    (h_q_pos : q > 0)
    (h_T : ∀ n, T n = (T 1) * q^(n-1))
    (h_ineq : T 7 > T 6 ∧ T 6 > T 8) :
    (0 < q ∧ q < 1) ∧ (T 13 > 1 ∧ 1 > T 14) := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_properties_l2412_241255


namespace NUMINAMATH_CALUDE_monotonicity_indeterminate_l2412_241236

theorem monotonicity_indeterminate 
  (f : ℝ → ℝ) 
  (h_domain : ∀ x, x ∈ Set.Icc (-1) 2 → f x ≠ 0) 
  (h_inequality : f (-1/2) < f 1) : 
  ¬ (∀ x y, x ∈ Set.Icc (-1) 2 → y ∈ Set.Icc (-1) 2 → x < y → f x < f y) ∧ 
  ¬ (∀ x y, x ∈ Set.Icc (-1) 2 → y ∈ Set.Icc (-1) 2 → x < y → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_indeterminate_l2412_241236


namespace NUMINAMATH_CALUDE_sally_quarters_l2412_241216

/-- The number of quarters Sally spent -/
def quarters_spent : ℕ := 418

/-- The number of quarters Sally has left -/
def quarters_left : ℕ := 342

/-- The initial number of quarters Sally had -/
def initial_quarters : ℕ := quarters_spent + quarters_left

theorem sally_quarters : initial_quarters = 760 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_l2412_241216


namespace NUMINAMATH_CALUDE_sequence_property_l2412_241256

-- Define the sequence type
def Sequence := ℕ+ → ℝ

-- Define the property of the sequence
def HasProperty (a : Sequence) : Prop :=
  ∀ n : ℕ+, a n * a (n + 2) = (a (n + 1))^2

-- State the theorem
theorem sequence_property (a : Sequence) 
  (h1 : HasProperty a) 
  (h2 : a 7 = 16) 
  (h3 : a 3 * a 5 = 4) : 
  a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l2412_241256


namespace NUMINAMATH_CALUDE_log_expression_equals_zero_l2412_241223

-- Define base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_zero :
  (1/2) * log10 4 + log10 5 - (π + 1)^0 = 0 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_zero_l2412_241223


namespace NUMINAMATH_CALUDE_james_investment_l2412_241284

theorem james_investment (initial_balance : ℝ) (weekly_investment : ℝ) (weeks : ℕ) (windfall_percentage : ℝ) : 
  initial_balance = 250000 ∧ 
  weekly_investment = 2000 ∧ 
  weeks = 52 ∧ 
  windfall_percentage = 0.5 →
  let final_balance := initial_balance + weekly_investment * weeks
  let windfall := windfall_percentage * final_balance
  final_balance + windfall = 531000 := by
sorry

end NUMINAMATH_CALUDE_james_investment_l2412_241284


namespace NUMINAMATH_CALUDE_count_numbers_satisfying_conditions_l2412_241278

/-- A function that returns true if n can be expressed as the sum of k consecutive positive integers starting from a -/
def is_sum_of_consecutive_integers (n k a : ℕ) : Prop :=
  n = k * a + k * (k - 1) / 2

/-- A function that checks if n satisfies all the conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  n ≤ 2000 ∧ ∃ k a : ℕ, k ≥ 60 ∧ is_sum_of_consecutive_integers n k a

/-- The main theorem stating that there are exactly 6 numbers satisfying the conditions -/
theorem count_numbers_satisfying_conditions :
  ∃! (S : Finset ℕ), (∀ n ∈ S, satisfies_conditions n) ∧ S.card = 6 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_satisfying_conditions_l2412_241278


namespace NUMINAMATH_CALUDE_job_completion_time_l2412_241298

/-- Given a job that can be completed by a man in 10 days and his son in 20/3 days,
    prove that they can complete the job together in 4 days. -/
theorem job_completion_time (man_time son_time combined_time : ℚ) : 
  man_time = 10 → son_time = 20 / 3 → 
  combined_time = 1 / (1 / man_time + 1 / son_time) → 
  combined_time = 4 := by sorry

end NUMINAMATH_CALUDE_job_completion_time_l2412_241298


namespace NUMINAMATH_CALUDE_line_equation_l2412_241228

/-- A line passing through a point and intersecting axes -/
structure Line where
  -- Point that the line passes through
  P : ℝ × ℝ
  -- x-coordinate of intersection with positive x-axis
  C : ℝ
  -- y-coordinate of intersection with negative y-axis
  D : ℝ
  -- Condition that P lies on the line
  point_on_line : (P.1 / C) + (P.2 / (-D)) = 1
  -- Condition for positive x-axis intersection
  pos_x_axis : C > 0
  -- Condition for negative y-axis intersection
  neg_y_axis : D > 0
  -- Area condition
  area_condition : (1/2) * C * D = 2

/-- Theorem stating the equation of the line -/
theorem line_equation (l : Line) (h : l.P = (1, -1)) :
  ∃ (a b : ℝ), a * l.P.1 + b * l.P.2 + 2 = 0 ∧ 
               ∀ (x y : ℝ), a * x + b * y + 2 = 0 ↔ (x / l.C) + (y / (-l.D)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l2412_241228


namespace NUMINAMATH_CALUDE_sqrt_sqrt_equation_l2412_241203

theorem sqrt_sqrt_equation (x : ℝ) : Real.sqrt (Real.sqrt x) = 3 → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sqrt_equation_l2412_241203


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2412_241212

theorem arithmetic_calculation : 24 * 36 + 18 * 24 - 12 * (36 / 6) = 1224 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2412_241212


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2412_241235

theorem opposite_of_negative_2023 : 
  ∀ x : ℤ, (x + (-2023) = 0) → x = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2412_241235


namespace NUMINAMATH_CALUDE_ln_inequality_solution_set_l2412_241218

theorem ln_inequality_solution_set (x : ℝ) : 
  Real.log (x^2 - 2*x - 2) > 0 ↔ x > 3 ∨ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_ln_inequality_solution_set_l2412_241218


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l2412_241220

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first three terms equals 3 -/
def sum_first_three (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 = 3

/-- The sum of the 5th, 6th, and 7th terms equals 9 -/
def sum_middle_three (a : ℕ → ℝ) : Prop :=
  a 5 + a 6 + a 7 = 9

theorem arithmetic_sequence_tenth_term (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : sum_first_three a) 
  (h3 : sum_middle_three a) : 
  a 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l2412_241220


namespace NUMINAMATH_CALUDE_equal_expressions_l2412_241258

theorem equal_expressions (x : ℝ) (h : x > 0) : 
  (∃! n : ℕ, n = (if x^x + x^x = 2*x^x then 1 else 0) + 
              (if x^x + x^x = x^(2*x) then 1 else 0) + 
              (if x^x + x^x = (2*x)^x then 1 else 0) + 
              (if x^x + x^x = (2*x)^(2*x) then 1 else 0)) ∧
  (x^x + x^x = 2*x^x) ∧
  (x^x + x^x ≠ x^(2*x)) ∧
  (x^x + x^x ≠ (2*x)^x) ∧
  (x^x + x^x ≠ (2*x)^(2*x)) :=
by sorry

end NUMINAMATH_CALUDE_equal_expressions_l2412_241258


namespace NUMINAMATH_CALUDE_expression_evaluation_l2412_241269

theorem expression_evaluation (a b c : ℚ) : 
  a = 6 → 
  b = 2 * a - 1 → 
  c = 2 * b - 30 → 
  a + 2 ≠ 0 → 
  b - 3 ≠ 0 → 
  c + 7 ≠ 0 → 
  (a + 3) / (a + 2) * (b + 5) / (b - 3) * (c + 10) / (c + 7) = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2412_241269


namespace NUMINAMATH_CALUDE_interest_difference_approx_l2412_241244

/-- Calculates the balance of an account with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Calculates the balance of an account with simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

/-- The difference between compound and simple interest balances -/
def interest_difference (principal : ℝ) (compound_rate : ℝ) (simple_rate : ℝ) (time : ℕ) : ℝ :=
  compound_interest principal compound_rate time - simple_interest principal simple_rate time

theorem interest_difference_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |interest_difference 15000 0.06 0.08 20 - 9107| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_difference_approx_l2412_241244


namespace NUMINAMATH_CALUDE_sale_price_markdown_l2412_241231

theorem sale_price_markdown (original_price : ℝ) (h1 : original_price > 0) : 
  let sale_price := 0.8 * original_price
  let final_price := 0.64 * original_price
  let markdown_percentage := (sale_price - final_price) / sale_price * 100
  markdown_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_sale_price_markdown_l2412_241231


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2412_241251

theorem complex_equation_solution (x y : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (2 * x + i) * (1 - i) = y) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2412_241251


namespace NUMINAMATH_CALUDE_investment_rate_problem_l2412_241261

theorem investment_rate_problem (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (desired_income : ℝ) :
  total_investment = 12000 →
  first_investment = 5000 →
  second_investment = 4000 →
  first_rate = 0.03 →
  second_rate = 0.035 →
  desired_income = 430 →
  let remainder := total_investment - first_investment - second_investment
  let income_from_first := first_investment * first_rate
  let income_from_second := second_investment * second_rate
  let required_additional_income := desired_income - income_from_first - income_from_second
  let required_rate := required_additional_income / remainder
  required_rate = 0.047 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l2412_241261


namespace NUMINAMATH_CALUDE_matrix_identity_sum_l2412_241290

open Matrix

variable {n : Type*} [DecidableEq n] [Fintype n]

theorem matrix_identity_sum (B : Matrix n n ℝ) :
  Invertible B →
  (B - 3 • 1) * (B - 5 • 1) = 0 →
  B + 15 • B⁻¹ = 8 • 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_identity_sum_l2412_241290


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l2412_241238

theorem complex_modulus_squared (w : ℂ) (h : w + 3 * Complex.abs w = -1 + 12 * Complex.I) :
  Complex.abs w ^ 2 = 2545 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l2412_241238


namespace NUMINAMATH_CALUDE_complex_subtraction_l2412_241283

theorem complex_subtraction (i : ℂ) (h : i^2 = -1) :
  (5 - 3*i) - (7 - 7*i) = -2 + 4*i :=
sorry

end NUMINAMATH_CALUDE_complex_subtraction_l2412_241283


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2412_241263

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2412_241263


namespace NUMINAMATH_CALUDE_max_surrounding_squares_l2412_241202

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Represents an arrangement of squares around a central square -/
structure SquareArrangement where
  centralSquare : Square
  surroundingSquare : Square
  numSurroundingSquares : ℕ

/-- The condition that the surrounding squares fit perfectly around the central square -/
def perfectFit (arrangement : SquareArrangement) : Prop :=
  arrangement.centralSquare.sideLength = arrangement.surroundingSquare.sideLength * (arrangement.numSurroundingSquares / 4 : ℝ)

/-- The theorem stating the maximum number of surrounding squares -/
theorem max_surrounding_squares (centralSquare : Square) (surroundingSquare : Square) 
    (h_central : centralSquare.sideLength = 4)
    (h_surrounding : surroundingSquare.sideLength = 1) :
    ∃ (arrangement : SquareArrangement), 
      arrangement.centralSquare = centralSquare ∧ 
      arrangement.surroundingSquare = surroundingSquare ∧
      arrangement.numSurroundingSquares = 16 ∧
      perfectFit arrangement ∧
      ∀ (otherArrangement : SquareArrangement), 
        otherArrangement.centralSquare = centralSquare → 
        otherArrangement.surroundingSquare = surroundingSquare → 
        perfectFit otherArrangement → 
        otherArrangement.numSurroundingSquares ≤ 16 :=
  sorry

end NUMINAMATH_CALUDE_max_surrounding_squares_l2412_241202


namespace NUMINAMATH_CALUDE_fencing_cost_is_5300_l2412_241224

/-- The cost of fencing per meter -/
def fencing_cost_per_meter : ℝ := 26.50

/-- The length of the rectangular plot in meters -/
def plot_length : ℝ := 57

/-- Calculate the breadth of the plot given the length -/
def plot_breadth : ℝ := plot_length - 14

/-- Calculate the perimeter of the rectangular plot -/
def plot_perimeter : ℝ := 2 * (plot_length + plot_breadth)

/-- Calculate the total cost of fencing the plot -/
def total_fencing_cost : ℝ := plot_perimeter * fencing_cost_per_meter

/-- Theorem stating that the total cost of fencing is 5300 currency units -/
theorem fencing_cost_is_5300 : total_fencing_cost = 5300 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_is_5300_l2412_241224


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2412_241264

/-- A square inscribed in a semicircle with radius 1 -/
structure InscribedSquare where
  /-- The center of the semicircle -/
  center : ℝ × ℝ
  /-- The vertices of the square -/
  vertices : Fin 4 → ℝ × ℝ
  /-- Two vertices are on the semicircle -/
  on_semicircle : ∃ (i j : Fin 4), i ≠ j ∧
    (vertices i).1^2 + (vertices i).2^2 = 1 ∧
    (vertices j).1^2 + (vertices j).2^2 = 1 ∧
    (vertices i).2 ≥ 0 ∧ (vertices j).2 ≥ 0
  /-- Two vertices are on the diameter -/
  on_diameter : ∃ (i j : Fin 4), i ≠ j ∧
    (vertices i).2 = 0 ∧ (vertices j).2 = 0 ∧
    abs ((vertices i).1 - (vertices j).1) = 2
  /-- The vertices form a square -/
  is_square : ∀ (i j : Fin 4), i ≠ j →
    (vertices i).1^2 + (vertices i).2^2 =
    (vertices j).1^2 + (vertices j).2^2

/-- The area of an inscribed square is 4/5 -/
theorem inscribed_square_area (s : InscribedSquare) :
  let side_length := abs ((s.vertices 0).1 - (s.vertices 1).1)
  side_length^2 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2412_241264


namespace NUMINAMATH_CALUDE_correct_ranking_l2412_241294

-- Define the set of friends
inductive Friend : Type
| Amy : Friend
| Bill : Friend
| Celine : Friend

-- Define the age relation
def older_than : Friend → Friend → Prop := sorry

-- Define the statements
def statement_I : Prop := older_than Friend.Bill Friend.Amy ∧ older_than Friend.Bill Friend.Celine
def statement_II : Prop := ¬(older_than Friend.Amy Friend.Bill ∧ older_than Friend.Amy Friend.Celine)
def statement_III : Prop := ¬(older_than Friend.Amy Friend.Celine ∧ older_than Friend.Bill Friend.Celine)

-- Define the theorem
theorem correct_ranking :
  -- Conditions
  (∀ (x y : Friend), x ≠ y → (older_than x y ∨ older_than y x)) →
  (∀ (x y z : Friend), older_than x y → older_than y z → older_than x z) →
  (statement_I ∨ statement_II ∨ statement_III) →
  (¬statement_I ∨ ¬statement_II) →
  (¬statement_I ∨ ¬statement_III) →
  (¬statement_II ∨ ¬statement_III) →
  -- Conclusion
  older_than Friend.Amy Friend.Celine ∧ older_than Friend.Celine Friend.Bill :=
by sorry

end NUMINAMATH_CALUDE_correct_ranking_l2412_241294


namespace NUMINAMATH_CALUDE_art_exhibition_tickets_l2412_241285

/-- Calculates the total number of tickets sold given the conditions -/
def totalTicketsSold (advancedPrice : ℕ) (doorPrice : ℕ) (totalCollected : ℕ) (advancedSold : ℕ) : ℕ :=
  let doorSold := (totalCollected - advancedPrice * advancedSold) / doorPrice
  advancedSold + doorSold

/-- Theorem stating that under the given conditions, 165 tickets were sold in total -/
theorem art_exhibition_tickets :
  totalTicketsSold 8 14 1720 100 = 165 := by
  sorry

#eval totalTicketsSold 8 14 1720 100

end NUMINAMATH_CALUDE_art_exhibition_tickets_l2412_241285


namespace NUMINAMATH_CALUDE_factors_of_1320_l2412_241292

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem factors_of_1320 : number_of_factors 1320 = 32 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1320_l2412_241292


namespace NUMINAMATH_CALUDE_kelly_carrot_harvest_l2412_241214

/-- Calculates the total weight of carrots harvested given the number of carrots in each bed and the weight ratio --/
def total_carrot_weight (bed1 bed2 bed3 carrots_per_pound : ℕ) : ℕ :=
  ((bed1 + bed2 + bed3) / carrots_per_pound : ℕ)

/-- Theorem stating that Kelly harvested 39 pounds of carrots --/
theorem kelly_carrot_harvest :
  total_carrot_weight 55 101 78 6 = 39 := by
  sorry

#eval total_carrot_weight 55 101 78 6

end NUMINAMATH_CALUDE_kelly_carrot_harvest_l2412_241214


namespace NUMINAMATH_CALUDE_delta_value_l2412_241209

theorem delta_value : ∀ Δ : ℤ, 5 * (-3) = Δ - 3 → Δ = -12 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l2412_241209


namespace NUMINAMATH_CALUDE_problem_1_l2412_241254

theorem problem_1 (a : ℝ) : 
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) ↔ a ∈ Set.Iic (-1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_problem_1_l2412_241254


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2412_241289

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 13*x + 4 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = 13 ∧ r₁ * r₂ = 4 ∧ r₁^2 + r₂^2 = 161 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2412_241289


namespace NUMINAMATH_CALUDE_flour_to_add_l2412_241208

theorem flour_to_add (recipe_amount : ℕ) (already_added : ℕ) (h1 : recipe_amount = 8) (h2 : already_added = 2) :
  recipe_amount - already_added = 6 := by
  sorry

end NUMINAMATH_CALUDE_flour_to_add_l2412_241208


namespace NUMINAMATH_CALUDE_stratified_sampling_teachers_l2412_241207

theorem stratified_sampling_teachers (total : ℕ) (sample_size : ℕ) (students_in_sample : ℕ) 
  (h1 : total = 4000)
  (h2 : sample_size = 200)
  (h3 : students_in_sample = 190) :
  (sample_size : ℚ) / total * (sample_size - students_in_sample) = 200 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_teachers_l2412_241207


namespace NUMINAMATH_CALUDE_geometric_progression_formula_l2412_241295

/-- A geometric progression with positive terms, where a₁ = 1 and a₂ + a₃ = 6 -/
def GeometricProgression (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q) ∧
  a 1 = 1 ∧
  a 2 + a 3 = 6

/-- The general term of the geometric progression is 2^(n-1) -/
theorem geometric_progression_formula (a : ℕ → ℝ) (h : GeometricProgression a) :
  ∀ n : ℕ, a n = 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_formula_l2412_241295


namespace NUMINAMATH_CALUDE_probability_prime_product_l2412_241280

/-- A standard 6-sided die -/
def Die : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of prime numbers on a 6-sided die -/
def PrimesOnDie : Finset ℕ := {2, 3, 5}

/-- The number of favorable outcomes -/
def FavorableOutcomes : ℕ := 9

/-- The total number of possible outcomes when rolling three dice -/
def TotalOutcomes : ℕ := 216

/-- The probability of rolling three 6-sided dice and getting a prime number as the product of their face values -/
theorem probability_prime_product (d : Finset ℕ) (p : Finset ℕ) (f : ℕ) (t : ℕ) 
  (h1 : d = Die) 
  (h2 : p = PrimesOnDie) 
  (h3 : f = FavorableOutcomes) 
  (h4 : t = TotalOutcomes) :
  (f : ℚ) / t = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_probability_prime_product_l2412_241280


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2412_241222

/-- Given a set of 50 numbers with arithmetic mean 38, prove that removing 45 and 55
    results in a new set with arithmetic mean 37.5 -/
theorem arithmetic_mean_after_removal (S : Finset ℝ) (sum_S : ℝ) : 
  S.card = 50 →
  sum_S = S.sum id →
  sum_S / 50 = 38 →
  45 ∈ S →
  55 ∈ S →
  let S' := S.erase 45 |>.erase 55
  let sum_S' := sum_S - 45 - 55
  sum_S' / S'.card = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2412_241222


namespace NUMINAMATH_CALUDE_log_product_equals_ten_l2412_241234

theorem log_product_equals_ten (n : ℕ) (h : n = 2) : 
  7.63 * (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_ten_l2412_241234


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l2412_241243

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 2) :
  ∃ (m : ℝ), m = -5/2 ∧ ∀ x, 8 * x^2 + 6 * x + 5 = 2 → 3 * x + 2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l2412_241243


namespace NUMINAMATH_CALUDE_circle_max_area_l2412_241205

/-- Given a circle with equation x^2 + y^2 + kx + 2y + k^2 = 0, 
    prove that its area is maximized when its center is at (0, -1) -/
theorem circle_max_area (k : ℝ) : 
  let circle_eq (x y : ℝ) := x^2 + y^2 + k*x + 2*y + k^2 = 0
  let center := (0, -1)
  let radius_squared (k : ℝ) := 1 - (3/4) * k^2
  ∀ x y : ℝ, circle_eq x y → 
    radius_squared k ≤ radius_squared 0 ∧ 
    circle_eq (center.1) (center.2) := by
  sorry

end NUMINAMATH_CALUDE_circle_max_area_l2412_241205


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2412_241217

theorem sum_of_reciprocals (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / a + 1 / b = (a + b) / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2412_241217


namespace NUMINAMATH_CALUDE_subtract_fractions_l2412_241262

theorem subtract_fractions (p q : ℚ) (h1 : 3 / p = 4) (h2 : 3 / q = 18) : p - q = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l2412_241262


namespace NUMINAMATH_CALUDE_second_class_average_l2412_241260

/-- Proves that given two classes with specified student counts and averages,
    the average mark of the second class is 90. -/
theorem second_class_average (students1 students2 : ℕ) (avg1 avg_combined : ℚ) :
  students1 = 30 →
  students2 = 50 →
  avg1 = 40 →
  avg_combined = 71.25 →
  (students1 * avg1 + students2 * (90 : ℚ)) / (students1 + students2 : ℚ) = avg_combined :=
by sorry

end NUMINAMATH_CALUDE_second_class_average_l2412_241260


namespace NUMINAMATH_CALUDE_f_inequality_solution_f_bounded_by_mn_l2412_241204

def f (x : ℝ) : ℝ := 2 * |x| + |x - 1|

theorem f_inequality_solution (x : ℝ) :
  f x > 4 ↔ x < -1 ∨ x > 5/3 := by sorry

theorem f_bounded_by_mn (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  {x : ℝ | f x ≤ 1/m^2 + 1/n^2 + 2*n*m} = {x : ℝ | -1 ≤ x ∧ x ≤ 5/3} := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_f_bounded_by_mn_l2412_241204


namespace NUMINAMATH_CALUDE_total_posters_proof_l2412_241286

def mario_posters : ℕ := 18
def samantha_extra_posters : ℕ := 15

theorem total_posters_proof :
  mario_posters + (mario_posters + samantha_extra_posters) = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_posters_proof_l2412_241286


namespace NUMINAMATH_CALUDE_largest_base5_5digit_in_base10_l2412_241211

/-- The largest five-digit number in base 5 -/
def largest_base5_5digit : ℕ := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_5digit_in_base10 : largest_base5_5digit = 3124 := by
  sorry

end NUMINAMATH_CALUDE_largest_base5_5digit_in_base10_l2412_241211


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l2412_241241

/-- Proves that the length of a rectangular plot is 62 meters given the specified conditions -/
theorem rectangular_plot_length : ∀ (breadth length perimeter : ℝ),
  length = breadth + 24 →
  perimeter = 2 * (length + breadth) →
  perimeter * 26.5 = 5300 →
  length = 62 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l2412_241241


namespace NUMINAMATH_CALUDE_x_lt_neg_one_necessary_not_sufficient_l2412_241267

theorem x_lt_neg_one_necessary_not_sufficient :
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by sorry

end NUMINAMATH_CALUDE_x_lt_neg_one_necessary_not_sufficient_l2412_241267


namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l2412_241226

theorem polynomial_sum_theorem (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l2412_241226


namespace NUMINAMATH_CALUDE_boat_speed_is_18_l2412_241230

/-- The speed of the boat in still water -/
def boat_speed : ℝ := 18

/-- The speed of the stream -/
def stream_speed : ℝ := 6

/-- Theorem stating that the boat speed in still water is 18 kmph -/
theorem boat_speed_is_18 :
  (∀ t : ℝ, t > 0 → 1 / (boat_speed - stream_speed) = 2 / (boat_speed + stream_speed)) →
  boat_speed = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_is_18_l2412_241230


namespace NUMINAMATH_CALUDE_total_distance_driven_l2412_241257

theorem total_distance_driven (renaldo_distance : ℝ) (ernesto_extra : ℝ) (marcos_percentage : ℝ) : 
  renaldo_distance = 15 →
  ernesto_extra = 7 →
  marcos_percentage = 0.2 →
  let ernesto_distance := renaldo_distance / 3 + ernesto_extra
  let marcos_distance := (renaldo_distance + ernesto_distance) * (1 + marcos_percentage)
  renaldo_distance + ernesto_distance + marcos_distance = 59.4 := by
sorry

end NUMINAMATH_CALUDE_total_distance_driven_l2412_241257


namespace NUMINAMATH_CALUDE_root_equation_value_l2412_241249

theorem root_equation_value (m : ℝ) (h : m^2 - 3*m + 1 = 0) : 
  (m - 3)^2 + (m + 2)*(m - 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l2412_241249


namespace NUMINAMATH_CALUDE_new_year_markup_l2412_241232

theorem new_year_markup (initial_markup : ℝ) (discount : ℝ) (final_profit : ℝ) :
  initial_markup = 0.20 →
  discount = 0.07 →
  final_profit = 0.395 →
  ∃ (new_year_markup : ℝ),
    (1 + initial_markup) * (1 + new_year_markup) * (1 - discount) = 1 + final_profit ∧
    new_year_markup = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_new_year_markup_l2412_241232


namespace NUMINAMATH_CALUDE_g_zeros_count_l2412_241265

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x + a)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (x - a) - x^2

theorem g_zeros_count (a : ℝ) :
  (∀ x, g a x ≠ 0) ∨
  (∃! x, g a x = 0) ∨
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧ ∀ x, g a x = 0 → x = x₁ ∨ x = x₂) ∨
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧ g a x₃ = 0 ∧
    ∀ x, g a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) :=
by sorry

end NUMINAMATH_CALUDE_g_zeros_count_l2412_241265


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_equals_one_l2412_241200

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a

theorem tangent_line_implies_a_equals_one (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ = x₀ ∧ (deriv (f a)) x₀ = 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_equals_one_l2412_241200


namespace NUMINAMATH_CALUDE_polar_coordinate_transformation_l2412_241206

theorem polar_coordinate_transformation (x y r θ : ℝ) :
  x = 8 ∧ y = 6 ∧ r = Real.sqrt (x^2 + y^2) ∧ 
  x = r * Real.cos θ ∧ y = r * Real.sin θ →
  ∃ (x' y' : ℝ), 
    x' = 2 * Real.sqrt 2 ∧ 
    y' = 14 * Real.sqrt 2 ∧
    x' = (2 * r) * Real.cos (θ + π/4) ∧ 
    y' = (2 * r) * Real.sin (θ + π/4) := by
  sorry

end NUMINAMATH_CALUDE_polar_coordinate_transformation_l2412_241206


namespace NUMINAMATH_CALUDE_min_votes_to_win_l2412_241277

-- Define the voting structure
def total_voters : ℕ := 135
def num_districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3

-- Define winning conditions
def win_precinct (votes : ℕ) : Prop := votes > voters_per_precinct / 2
def win_district (precincts_won : ℕ) : Prop := precincts_won > precincts_per_district / 2
def win_final (districts_won : ℕ) : Prop := districts_won > num_districts / 2

-- Theorem statement
theorem min_votes_to_win (min_votes : ℕ) : 
  (∃ (districts_won precincts_won votes_per_precinct : ℕ),
    win_final districts_won ∧
    win_district precincts_won ∧
    win_precinct votes_per_precinct ∧
    min_votes = districts_won * precincts_won * votes_per_precinct) →
  min_votes = 30 := by sorry

end NUMINAMATH_CALUDE_min_votes_to_win_l2412_241277


namespace NUMINAMATH_CALUDE_computer_literate_female_employees_l2412_241282

/-- Proves the number of computer literate female employees in an office -/
theorem computer_literate_female_employees
  (total_employees : ℕ)
  (female_percentage : ℚ)
  (male_computer_literate_percentage : ℚ)
  (total_computer_literate_percentage : ℚ)
  (h_total : total_employees = 1200)
  (h_female : female_percentage = 60 / 100)
  (h_male_cl : male_computer_literate_percentage = 50 / 100)
  (h_total_cl : total_computer_literate_percentage = 62 / 100) :
  ↑(total_employees : ℚ) * female_percentage * total_computer_literate_percentage -
  (↑(total_employees : ℚ) * (1 - female_percentage) * male_computer_literate_percentage) = 504 :=
sorry

end NUMINAMATH_CALUDE_computer_literate_female_employees_l2412_241282


namespace NUMINAMATH_CALUDE_average_weight_increase_l2412_241242

/-- Proves that replacing a person weighing 58 kg with a person weighing 106 kg
    in a group of 12 people increases the average weight by 4 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total_weight := 12 * initial_average
  let new_total_weight := initial_total_weight - 58 + 106
  let new_average := new_total_weight / 12
  new_average - initial_average = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2412_241242


namespace NUMINAMATH_CALUDE_water_reservoir_ratio_l2412_241250

/-- The ratio of the amount of water in the reservoir at the end of the month to the normal level -/
theorem water_reservoir_ratio : 
  ∀ (total_capacity normal_level end_month_level : ℝ),
  end_month_level = 30 →
  end_month_level = 0.75 * total_capacity →
  normal_level = total_capacity - 20 →
  end_month_level / normal_level = 1.5 := by
sorry

end NUMINAMATH_CALUDE_water_reservoir_ratio_l2412_241250


namespace NUMINAMATH_CALUDE_eggs_remaining_l2412_241259

theorem eggs_remaining (initial_eggs : ℕ) (eggs_removed : ℕ) (eggs_left : ℕ) : 
  initial_eggs = 47 → eggs_removed = 5 → eggs_left = initial_eggs - eggs_removed → eggs_left = 42 := by
  sorry

end NUMINAMATH_CALUDE_eggs_remaining_l2412_241259


namespace NUMINAMATH_CALUDE_max_vertex_product_sum_l2412_241275

/-- Represents the assignment of numbers to the faces of a cube -/
structure CubeAssignment where
  faces : Fin 6 → ℕ
  valid : ∀ i, faces i ∈ ({3, 4, 5, 6, 7, 8} : Set ℕ)
  distinct : ∀ i j, i ≠ j → faces i ≠ faces j

/-- Calculates the sum of products at vertices for a given cube assignment -/
def vertexProductSum (assignment : CubeAssignment) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem: The maximum sum of vertex products is 1331 -/
theorem max_vertex_product_sum :
  ∃ (assignment : CubeAssignment), 
    vertexProductSum assignment = 1331 ∧
    ∀ (other : CubeAssignment), vertexProductSum other ≤ 1331 :=
  sorry

end NUMINAMATH_CALUDE_max_vertex_product_sum_l2412_241275


namespace NUMINAMATH_CALUDE_greatest_integer_prime_quadratic_l2412_241201

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def abs_quadratic (x : ℤ) : ℕ := Int.natAbs (8 * x^2 - 53 * x + 21)

theorem greatest_integer_prime_quadratic :
  ∀ x : ℤ, x > 1 → ¬(is_prime (abs_quadratic x)) ∧
  (is_prime (abs_quadratic 1)) ∧
  (∀ y : ℤ, y ≤ 1 → is_prime (abs_quadratic y) → y = 1) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_prime_quadratic_l2412_241201


namespace NUMINAMATH_CALUDE_firewood_collection_l2412_241276

theorem firewood_collection (total kimberley houston : ℕ) (h1 : total = 35) (h2 : kimberley = 10) (h3 : houston = 12) :
  ∃ ela : ℕ, total = kimberley + houston + ela ∧ ela = 13 := by
  sorry

end NUMINAMATH_CALUDE_firewood_collection_l2412_241276


namespace NUMINAMATH_CALUDE_third_generation_tail_length_l2412_241239

/-- The tail length growth factor between generations -/
def growth_factor : ℝ := 1.25

/-- The initial tail length of the first generation in centimeters -/
def initial_length : ℝ := 16

/-- The tail length of the nth generation -/
def tail_length (n : ℕ) : ℝ := initial_length * growth_factor ^ n

theorem third_generation_tail_length :
  tail_length 2 = 25 := by sorry

end NUMINAMATH_CALUDE_third_generation_tail_length_l2412_241239


namespace NUMINAMATH_CALUDE_cubic_roots_geometric_progression_l2412_241293

/-- 
A cubic polynomial with coefficients a, b, and c has roots that form 
a geometric progression if and only if a^3 * c = b^3.
-/
theorem cubic_roots_geometric_progression 
  (a b c : ℝ) : 
  (∃ x y z : ℝ, (x^3 + a*x^2 + b*x + c = 0) ∧ 
                (y^3 + a*y^2 + b*y + c = 0) ∧ 
                (z^3 + a*z^2 + b*z + c = 0) ∧ 
                (y^2 = x*z)) ↔ 
  (a^3 * c = b^3) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_geometric_progression_l2412_241293


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2412_241240

theorem arithmetic_calculation : 5 + 15 / 3 - 2^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2412_241240


namespace NUMINAMATH_CALUDE_graph_is_parabola_l2412_241247

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * (x - 2)^2 + 6

-- Theorem stating that the graph of f is a parabola
theorem graph_is_parabola : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c :=
sorry

end NUMINAMATH_CALUDE_graph_is_parabola_l2412_241247


namespace NUMINAMATH_CALUDE_pencil_distribution_l2412_241266

theorem pencil_distribution (total : ℕ) (h1 : total = 8 * 6 + 4) : 
  total / 4 = 13 := by
sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2412_241266


namespace NUMINAMATH_CALUDE_twenty_sixth_digit_of_N_l2412_241215

def N (d : ℕ) : ℕ := 
  (10^49 - 1) / 9 + d * 10^24 - 10^25 + 1

theorem twenty_sixth_digit_of_N (d : ℕ) : 
  d < 10 → N d % 13 = 0 → d = 9 := by
  sorry

end NUMINAMATH_CALUDE_twenty_sixth_digit_of_N_l2412_241215


namespace NUMINAMATH_CALUDE_ned_games_problem_l2412_241291

theorem ned_games_problem (initial_games : ℕ) : 
  (3/4 : ℚ) * (2/3 : ℚ) * initial_games = 6 → initial_games = 12 := by
  sorry

end NUMINAMATH_CALUDE_ned_games_problem_l2412_241291


namespace NUMINAMATH_CALUDE_tree_initial_height_l2412_241253

/-- Given a tree with constant yearly growth for 6 years, prove its initial height. -/
theorem tree_initial_height (growth_rate : ℝ) (h1 : growth_rate = 0.4) : ∃ (initial_height : ℝ),
  initial_height + 6 * growth_rate = (initial_height + 4 * growth_rate) * (1 + 1/7) ∧
  initial_height = 4 := by
  sorry

end NUMINAMATH_CALUDE_tree_initial_height_l2412_241253


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l2412_241270

-- Define the function f(x) = x³ - x² + x + 1
def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l2412_241270


namespace NUMINAMATH_CALUDE_solution_set_inequality_proof_l2412_241272

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 5|

-- Part 1: Solution set of f(x) < 10
theorem solution_set (x : ℝ) : f x < 10 ↔ x ∈ Set.Ioo (-19/3) (-1) := by sorry

-- Part 2: Prove |a+b| + |a-b| < f(x) given |a| < 3 and |b| < 3
theorem inequality_proof (x a b : ℝ) (ha : |a| < 3) (hb : |b| < 3) :
  |a + b| + |a - b| < f x := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_proof_l2412_241272


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2412_241233

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_seq (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_seq a →
  geometric_seq (a 1) (a 2) (a 4) →
  a 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2412_241233


namespace NUMINAMATH_CALUDE_copper_zinc_mass_ranges_l2412_241299

/-- Represents the properties of a copper-zinc mixture -/
structure CopperZincMixture where
  total_mass : ℝ
  total_volume : ℝ
  copper_density_min : ℝ
  copper_density_max : ℝ
  zinc_density_min : ℝ
  zinc_density_max : ℝ

/-- Theorem stating the mass ranges of copper and zinc in the mixture -/
theorem copper_zinc_mass_ranges (mixture : CopperZincMixture)
  (h_total_mass : mixture.total_mass = 400)
  (h_total_volume : mixture.total_volume = 50)
  (h_copper_density : mixture.copper_density_min = 8.8 ∧ mixture.copper_density_max = 9)
  (h_zinc_density : mixture.zinc_density_min = 7.1 ∧ mixture.zinc_density_max = 7.2) :
  ∃ (copper_mass zinc_mass : ℝ),
    200 ≤ copper_mass ∧ copper_mass ≤ 233 ∧
    167 ≤ zinc_mass ∧ zinc_mass ≤ 200 ∧
    copper_mass + zinc_mass = mixture.total_mass ∧
    copper_mass / mixture.copper_density_max + zinc_mass / mixture.zinc_density_min = mixture.total_volume :=
by sorry

end NUMINAMATH_CALUDE_copper_zinc_mass_ranges_l2412_241299


namespace NUMINAMATH_CALUDE_solve_for_k_l2412_241248

theorem solve_for_k : ∃ k : ℚ, (4 * k - 3 * (-1) = 2) ∧ (k = -1/4) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l2412_241248


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2412_241297

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ - m = 0 ∧ x₂^2 - 2*x₂ - m = 0) → m ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2412_241297


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2412_241219

theorem ellipse_k_range (k : ℝ) :
  (∃ (x y : ℝ), 2 * x^2 + k * y^2 = 1 ∧ 
   ∃ (c : ℝ), c > 0 ∧ c^2 = 2 * x^2 + k * y^2 - k * (x^2 + y^2)) ↔ 
  (0 < k ∧ k < 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2412_241219


namespace NUMINAMATH_CALUDE_derek_remaining_money_l2412_241288

theorem derek_remaining_money (initial_amount : ℕ) : 
  initial_amount = 960 →
  let textbook_expense := initial_amount / 2
  let remaining_after_textbooks := initial_amount - textbook_expense
  let supply_expense := remaining_after_textbooks / 4
  let final_remaining := remaining_after_textbooks - supply_expense
  final_remaining = 360 := by
sorry

end NUMINAMATH_CALUDE_derek_remaining_money_l2412_241288


namespace NUMINAMATH_CALUDE_factorial_120_121_is_perfect_square_l2412_241296

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Definition of perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- Theorem: 120! · 121! is a perfect square -/
theorem factorial_120_121_is_perfect_square :
  is_perfect_square (factorial 120 * factorial 121) := by
  sorry

end NUMINAMATH_CALUDE_factorial_120_121_is_perfect_square_l2412_241296


namespace NUMINAMATH_CALUDE_outfit_combinations_l2412_241237

theorem outfit_combinations (s p h : ℕ) (hs : s = 5) (hp : p = 6) (hh : h = 3) :
  s * p * h = 90 := by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2412_241237


namespace NUMINAMATH_CALUDE_fixed_distance_vector_l2412_241281

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem fixed_distance_vector (a b : E) :
  ∃ t u : ℝ, ∀ p : E,
    (‖p - b‖ = 3 * ‖p - a‖) →
    (∃ c : ℝ, ∀ q : E, (‖p - b‖ = 3 * ‖p - a‖) → ‖q - (t • a + u • b)‖ = c) →
    t = 9/8 ∧ u = -1/8 :=
by sorry

end NUMINAMATH_CALUDE_fixed_distance_vector_l2412_241281


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l2412_241274

theorem gcd_special_numbers :
  let m : ℕ := 55555555
  let n : ℕ := 111111111
  Nat.gcd m n = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l2412_241274


namespace NUMINAMATH_CALUDE_min_value_of_f_min_value_of_sum_squares_l2412_241225

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x - 1) + abs (2 * x + 1)

-- Theorem 1: The minimum value of f(x) is 3
theorem min_value_of_f : ∃ k : ℝ, k = 3 ∧ ∀ x : ℝ, f x ≥ k :=
sorry

-- Theorem 2: Minimum value of a² + b² + c² given the conditions
theorem min_value_of_sum_squares :
  ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  3 * a + 2 * b + c = 3 →
  a^2 + b^2 + c^2 ≥ 9/14 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_min_value_of_sum_squares_l2412_241225


namespace NUMINAMATH_CALUDE_binomial_probability_problem_l2412_241279

theorem binomial_probability_problem (p : ℝ) (X : ℕ → ℝ) :
  (∀ k, X k = Nat.choose 4 k * p^k * (1 - p)^(4 - k)) →
  X 2 = 8/27 →
  p = 1/3 ∨ p = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_binomial_probability_problem_l2412_241279


namespace NUMINAMATH_CALUDE_not_closed_sequence_3_pow_arithmetic_closed_sequence_iff_l2412_241229

/-- Definition of a closed sequence -/
def is_closed_sequence (a : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, ∃ k : ℕ, a m + a n = a k

/-- The sequence a_n = 3^n is not a closed sequence -/
theorem not_closed_sequence_3_pow : ¬ is_closed_sequence (λ n => 3^n) := by sorry

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- Necessary and sufficient condition for an arithmetic sequence to be a closed sequence -/
theorem arithmetic_closed_sequence_iff (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d →
  (is_closed_sequence a ↔ ∃ m : ℤ, m ≥ -1 ∧ a 1 = m * d) := by sorry

end NUMINAMATH_CALUDE_not_closed_sequence_3_pow_arithmetic_closed_sequence_iff_l2412_241229


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2412_241210

-- Define the equation
def equation (k : ℝ) (x : ℝ) : Prop :=
  (x - 3) / (k * x + 2) = x

-- Define the condition for exactly one solution
def has_exactly_one_solution (k : ℝ) : Prop :=
  ∃! x : ℝ, equation k x

-- Theorem statement
theorem unique_solution_condition :
  ∀ k : ℝ, has_exactly_one_solution k ↔ k = -1/12 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2412_241210
