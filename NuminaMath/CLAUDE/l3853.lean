import Mathlib

namespace digit_for_divisibility_by_6_l3853_385373

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem digit_for_divisibility_by_6 :
  ∃ B : ℕ, B < 10 ∧ is_divisible_by_6 (5170 + B) ∧ (B = 2 ∨ B = 8) :=
by sorry

end digit_for_divisibility_by_6_l3853_385373


namespace variance_transformation_l3853_385350

-- Define a type for our dataset
def Dataset := Fin 10 → ℝ

-- Define the variance of a dataset
noncomputable def variance (data : Dataset) : ℝ := sorry

-- State the theorem
theorem variance_transformation (data : Dataset) :
  variance data = 3 →
  variance (fun i => 2 * (data i) + 3) = 12 := by sorry

end variance_transformation_l3853_385350


namespace teds_chocolates_l3853_385339

theorem teds_chocolates : ∃ (x : ℚ), 
  x > 0 ∧ 
  (3/16 * x - 3/4 - 5 = 10) ∧ 
  x = 84 := by
  sorry

end teds_chocolates_l3853_385339


namespace max_gold_coins_l3853_385330

theorem max_gold_coins (n : ℕ) (h1 : n < 150) 
  (h2 : ∃ (k : ℕ), n = 13 * k + 3) : n ≤ 143 :=
by
  sorry

end max_gold_coins_l3853_385330


namespace chickens_and_rabbits_l3853_385321

theorem chickens_and_rabbits (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 40) 
  (h2 : total_legs = 108) : 
  ∃ (chickens rabbits : ℕ), 
    chickens + rabbits = total_animals ∧ 
    2 * chickens + 4 * rabbits = total_legs ∧ 
    chickens = 26 ∧ 
    rabbits = 14 := by
  sorry

end chickens_and_rabbits_l3853_385321


namespace fixed_point_on_line_l3853_385342

/-- The line passing through a fixed point for all real values of a -/
def line (a x y : ℝ) : Prop := (a - 1) * x + a * y + 3 = 0

/-- The fixed point through which the line passes -/
def fixed_point : ℝ × ℝ := (3, -3)

/-- Theorem stating that the fixed point lies on the line for all real a -/
theorem fixed_point_on_line :
  ∀ a : ℝ, line a (fixed_point.1) (fixed_point.2) := by
sorry

end fixed_point_on_line_l3853_385342


namespace mashas_dolls_l3853_385389

theorem mashas_dolls (n : ℕ) : 
  (n / 2 : ℚ) * 1 + (n / 4 : ℚ) * 2 + (n / 4 : ℚ) * 4 = 24 → n = 12 := by
  sorry

end mashas_dolls_l3853_385389


namespace quadratic_integer_roots_l3853_385381

theorem quadratic_integer_roots (p q : ℤ) :
  ∀ n : ℕ, n ≤ 9 →
  ∃ x y : ℤ, x^2 + (p + n) * x + (q + n) = 0 ∧
             y^2 + (p + n) * y + (q + n) = 0 ∧
             x ≠ y :=
by sorry

end quadratic_integer_roots_l3853_385381


namespace bottle_cap_probability_l3853_385377

theorem bottle_cap_probability (p_convex : ℝ) (h1 : p_convex = 0.44) :
  1 - p_convex = 0.56 := by
  sorry

end bottle_cap_probability_l3853_385377


namespace sin_shift_l3853_385378

theorem sin_shift (x : ℝ) : Real.sin x = Real.sin (2 * (x - π / 4)) := by
  sorry

end sin_shift_l3853_385378


namespace quadratic_inequality_l3853_385306

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) (x₀ y₁ y₂ y₃ : ℝ) :
  a ≠ 0 →
  f a b c (-2) = 0 →
  x₀ > 1 →
  f a b c x₀ = 0 →
  (a + b + c) * (4 * a + 2 * b + c) < 0 →
  ∃ y, y < 0 ∧ f a b c 0 = y →
  f a b c (-1) = y₁ →
  f a b c (-Real.sqrt 2 / 2) = y₂ →
  f a b c 1 = y₃ →
  y₃ > y₁ ∧ y₁ > y₂ :=
by sorry

end quadratic_inequality_l3853_385306


namespace greatest_integer_fraction_l3853_385399

theorem greatest_integer_fraction (x : ℤ) : 
  x ≠ 3 → 
  (∀ y : ℤ, y > 28 → ¬(∃ k : ℤ, (y^2 + 2*y + 10) = k * (y - 3))) → 
  (∃ k : ℤ, (28^2 + 2*28 + 10) = k * (28 - 3)) :=
sorry

end greatest_integer_fraction_l3853_385399


namespace derivative_at_two_l3853_385374

/-- Given a function f with the property that f(x) = 2xf'(2) + x^3 for all x,
    prove that f'(2) = -12 -/
theorem derivative_at_two (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x * (deriv f 2) + x^3) :
  deriv f 2 = -12 := by sorry

end derivative_at_two_l3853_385374


namespace minimum_h_22_l3853_385333

def IsTenuous (h : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, h x + h y > (y : ℤ)^2

def SumUpTo30 (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 30).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

theorem minimum_h_22 (h : ℕ+ → ℤ) (h_tenuous : IsTenuous h) 
    (h_min : ∀ g : ℕ+ → ℤ, IsTenuous g → SumUpTo30 h ≤ SumUpTo30 g) :
    h ⟨22, by norm_num⟩ ≥ 357 := by
  sorry

end minimum_h_22_l3853_385333


namespace polygon_sides_when_interior_triple_exterior_l3853_385328

theorem polygon_sides_when_interior_triple_exterior : ∃ n : ℕ,
  (n ≥ 3) ∧
  ((n - 2) * 180 = 3 * 360) ∧
  (∀ m : ℕ, m ≥ 3 → (m - 2) * 180 = 3 * 360 → m = n) :=
by sorry

end polygon_sides_when_interior_triple_exterior_l3853_385328


namespace tractor_oil_theorem_l3853_385326

/-- Represents the remaining oil in liters after t hours of work -/
def remaining_oil (initial_oil : ℝ) (consumption_rate : ℝ) (t : ℝ) : ℝ :=
  initial_oil - consumption_rate * t

theorem tractor_oil_theorem (initial_oil : ℝ) (consumption_rate : ℝ) (t : ℝ) :
  initial_oil = 50 → consumption_rate = 8 →
  (∀ t, remaining_oil initial_oil consumption_rate t = 50 - 8 * t) ∧
  (remaining_oil initial_oil consumption_rate 4 = 18) := by
  sorry


end tractor_oil_theorem_l3853_385326


namespace hex_to_binary_digits_l3853_385386

theorem hex_to_binary_digits : ∃ (n : ℕ), n = 20 ∧ 
  (∀ (m : ℕ), 2^m ≤ (11 * 16^4 + 1 * 16^3 + 2 * 16^2 + 3 * 16^1 + 4 * 16^0) → m ≤ n) ∧
  (2^n > 11 * 16^4 + 1 * 16^3 + 2 * 16^2 + 3 * 16^1 + 4 * 16^0) :=
by sorry

end hex_to_binary_digits_l3853_385386


namespace divisibility_by_thirteen_l3853_385343

theorem divisibility_by_thirteen (a b c : ℤ) (h : 13 ∣ (a + b + c)) :
  13 ∣ (a^2007 + b^2007 + c^2007 + 2 * 2007 * a * b * c) := by
  sorry

end divisibility_by_thirteen_l3853_385343


namespace correct_dial_probability_l3853_385383

/-- The probability of correctly dialing a phone number with a missing last digit -/
def dial_probability : ℚ := 3 / 10

/-- The number of possible digits for a phone number -/
def num_digits : ℕ := 10

/-- The maximum number of attempts allowed -/
def max_attempts : ℕ := 3

theorem correct_dial_probability :
  (∀ n : ℕ, n ≤ max_attempts → (1 : ℚ) / num_digits = 1 / 10) →
  (∀ n : ℕ, n < max_attempts → (num_digits - n : ℚ) / num_digits * (1 : ℚ) / (num_digits - n) = 1 / 10) →
  dial_probability = 3 / 10 := by
  sorry

end correct_dial_probability_l3853_385383


namespace work_completion_time_l3853_385338

/-- Given that A can do a work in 6 days and A and B together can finish the work in 4 days,
    prove that B can do the work alone in 12 days. -/
theorem work_completion_time (a b : ℝ) 
  (ha : a = 6)  -- A can do the work in 6 days
  (hab : 1 / a + 1 / b = 1 / 4)  -- A and B together can finish the work in 4 days
  : b = 12 := by  -- B can do the work alone in 12 days
sorry

end work_completion_time_l3853_385338


namespace cos_2017pi_minus_2alpha_l3853_385325

theorem cos_2017pi_minus_2alpha (α : Real) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 2) :
  Real.cos (2017 * π - 2 * α) = 1/2 := by sorry

end cos_2017pi_minus_2alpha_l3853_385325


namespace solve_equation_l3853_385340

theorem solve_equation (x : ℝ) : 0.3 * x = 45 → (10 / 3) * (0.3 * x) = 150 := by
  sorry

end solve_equation_l3853_385340


namespace clock_angle_at_9_l3853_385310

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle : ℕ := 360

/-- The number of degrees each hour represents on a clock face -/
def degrees_per_hour : ℕ := full_circle / clock_hours

/-- The position of the minute hand at 9:00 in degrees -/
def minute_hand_position : ℕ := 0

/-- The position of the hour hand at 9:00 in degrees -/
def hour_hand_position : ℕ := 9 * degrees_per_hour

/-- The smaller angle between the hour hand and minute hand at 9:00 -/
def smaller_angle : ℕ := min (hour_hand_position - minute_hand_position) (full_circle - (hour_hand_position - minute_hand_position))

theorem clock_angle_at_9 : smaller_angle = 90 := by sorry

end clock_angle_at_9_l3853_385310


namespace fibonacci_rectangle_division_l3853_385334

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- A rectangle that can be divided into n squares -/
structure DivisibleRectangle (n : ℕ) :=
  (width : ℕ)
  (height : ℕ)
  (divides_into_squares : ∃ (squares : Finset (ℕ × ℕ)), 
    squares.card = n ∧ 
    (∀ (s : ℕ × ℕ), s ∈ squares → s.1 * s.2 ≤ width * height) ∧
    (∀ (s1 s2 s3 : ℕ × ℕ), s1 ∈ squares → s2 ∈ squares → s3 ∈ squares → 
      s1 = s2 ∧ s2 = s3 → s1 = s2))

/-- Theorem: For each natural number n, there exists a rectangle that can be 
    divided into n squares with no more than two squares being the same size -/
theorem fibonacci_rectangle_division (n : ℕ) : 
  ∃ (rect : DivisibleRectangle n), rect.width = fib n ∧ rect.height = fib (n + 1) := by
  sorry

end fibonacci_rectangle_division_l3853_385334


namespace f_monotone_implies_a_range_l3853_385302

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x - 1 else Real.log x / Real.log a

-- Define the property of being monotonically increasing
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem f_monotone_implies_a_range (a : ℝ) :
  MonotonicallyIncreasing (f a) → 3 < a ∧ a ≤ 4 := by
  sorry

end f_monotone_implies_a_range_l3853_385302


namespace triangle_count_is_68_l3853_385368

/-- Represents a grid-divided rectangle with diagonals -/
structure GridRectangle where
  width : ℕ
  height : ℕ
  vertical_divisions : ℕ
  horizontal_divisions : ℕ
  has_corner_diagonals : Bool
  has_midpoint_diagonals : Bool
  has_full_diagonal : Bool

/-- Counts the number of triangles in a GridRectangle -/
def count_triangles (rect : GridRectangle) : ℕ :=
  sorry

/-- The specific rectangle from the problem -/
def problem_rectangle : GridRectangle :=
  { width := 40
  , height := 30
  , vertical_divisions := 3
  , horizontal_divisions := 2
  , has_corner_diagonals := true
  , has_midpoint_diagonals := true
  , has_full_diagonal := true }

theorem triangle_count_is_68 : count_triangles problem_rectangle = 68 := by
  sorry

end triangle_count_is_68_l3853_385368


namespace ufo_convention_attendees_l3853_385379

theorem ufo_convention_attendees (total : ℕ) (difference : ℕ) : 
  total = 120 → difference = 4 → 
  ∃ (male female : ℕ), 
    male + female = total ∧ 
    male = female + difference ∧ 
    male = 62 := by
  sorry

end ufo_convention_attendees_l3853_385379


namespace imaginary_part_i_2015_l3853_385384

theorem imaginary_part_i_2015 : Complex.im (Complex.I ^ 2015) = -1 := by
  sorry

end imaginary_part_i_2015_l3853_385384


namespace faye_coloring_books_l3853_385344

/-- The number of coloring books Faye gave away -/
def books_given_away : ℕ := sorry

/-- The initial number of coloring books Faye had -/
def initial_books : ℕ := 34

/-- The number of coloring books Faye bought -/
def books_bought : ℕ := 48

/-- The final number of coloring books Faye has -/
def final_books : ℕ := 79

theorem faye_coloring_books : 
  initial_books - books_given_away + books_bought = final_books ∧ 
  books_given_away = 3 := by sorry

end faye_coloring_books_l3853_385344


namespace total_rent_is_7800_l3853_385305

/-- Represents the rent shares of four people renting a house -/
structure RentShares where
  purity : ℝ
  sheila : ℝ
  rose : ℝ
  john : ℝ

/-- Calculates the total rent based on the given rent shares -/
def totalRent (shares : RentShares) : ℝ :=
  shares.purity + shares.sheila + shares.rose + shares.john

/-- Theorem stating that the total rent is $7,800 given the conditions -/
theorem total_rent_is_7800 :
  ∀ (shares : RentShares),
    shares.sheila = 5 * shares.purity →
    shares.rose = 3 * shares.purity →
    shares.john = 4 * shares.purity →
    shares.rose = 1800 →
    totalRent shares = 7800 := by
  sorry

#check total_rent_is_7800

end total_rent_is_7800_l3853_385305


namespace prob_S7_eq_3_l3853_385303

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the outcome of a single draw -/
def drawOutcome (c : BallColor) : Int :=
  match c with
  | BallColor.Red => -1
  | BallColor.White => 1

/-- The probability of drawing a red ball -/
def probRed : ℚ := 2/3

/-- The probability of drawing a white ball -/
def probWhite : ℚ := 1/3

/-- The number of draws -/
def n : ℕ := 7

/-- The sum we're interested in -/
def targetSum : Int := 3

/-- The probability of getting the target sum after n draws -/
def probTargetSum (n : ℕ) (targetSum : Int) : ℚ :=
  sorry

theorem prob_S7_eq_3 :
  probTargetSum n targetSum = 28 / 3^6 :=
sorry

end prob_S7_eq_3_l3853_385303


namespace inequality_solution_set_l3853_385314

-- Define the inequality
def inequality (x : ℝ) : Prop := -x^2 + 5*x > 6

-- Define the solution set
def solution_set : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set :=
sorry

end inequality_solution_set_l3853_385314


namespace scaling_property_l3853_385348

theorem scaling_property (x y z : ℝ) (h : 2994 * x * 14.5 = 173) : 29.94 * x * 1.45 = 1.73 := by
  sorry

end scaling_property_l3853_385348


namespace unit_vector_parallel_to_d_l3853_385315

def vector_d : Fin 2 → ℝ := ![12, 5]

theorem unit_vector_parallel_to_d :
  let magnitude : ℝ := Real.sqrt (12^2 + 5^2)
  let unit_vector_positive : Fin 2 → ℝ := ![12 / magnitude, 5 / magnitude]
  let unit_vector_negative : Fin 2 → ℝ := ![-12 / magnitude, -5 / magnitude]
  (∀ i, vector_d i = magnitude * unit_vector_positive i) ∧
  (∀ i, vector_d i = magnitude * unit_vector_negative i) ∧
  (∀ i, unit_vector_positive i * unit_vector_positive i + 
        unit_vector_negative i * unit_vector_negative i = 2) :=
by sorry

end unit_vector_parallel_to_d_l3853_385315


namespace factorizable_polynomial_l3853_385398

theorem factorizable_polynomial (x y a b : ℝ) : 
  ∃ (p q : ℝ), x^2 - x + (1/4) = (p - q)^2 ∧ 
  (∀ (r s : ℝ), 4*x^2 + 1 ≠ (r - s)^2) ∧
  (∀ (r s : ℝ), 9*a^2*b^2 - 3*a*b + 1 ≠ (r - s)^2) ∧
  (∀ (r s : ℝ), -x^2 - y^2 ≠ (r - s)^2) :=
by
  sorry

end factorizable_polynomial_l3853_385398


namespace rationalize_denominator_l3853_385346

theorem rationalize_denominator (x : ℝ) :
  x > 0 → (45 * Real.sqrt 3) / Real.sqrt x = 3 * Real.sqrt 15 ↔ x = 45 := by
  sorry

end rationalize_denominator_l3853_385346


namespace student_claim_incorrect_l3853_385319

theorem student_claim_incorrect :
  ¬ ∃ (m n : ℤ), 
    n > 0 ∧ 
    n ≤ 100 ∧ 
    ∃ (a : ℕ → ℕ), (m : ℚ) / n = 0.167 + ∑' i, (a i : ℚ) / 10^(i+3) :=
by sorry

end student_claim_incorrect_l3853_385319


namespace imaginary_part_of_1_plus_2i_l3853_385358

theorem imaginary_part_of_1_plus_2i :
  Complex.im (1 + 2*I) = 2 := by
  sorry

end imaginary_part_of_1_plus_2i_l3853_385358


namespace number_of_days_function_l3853_385336

/-- The "number of days function" for given day points and a point on its graph -/
theorem number_of_days_function (k : ℝ) :
  (∀ x : ℝ, ∃ y₁ y₂ : ℝ, y₁ = k * x + 4 ∧ y₂ = 2 * x) →
  (∃ y : ℝ → ℝ, y 2 = 3 ∧ ∀ x : ℝ, y x = (k * x + 4) - (2 * x)) →
  (∃ y : ℝ → ℝ, ∀ x : ℝ, y x = -1/2 * x + 4) :=
by sorry

end number_of_days_function_l3853_385336


namespace non_pine_trees_l3853_385322

theorem non_pine_trees (total : ℕ) (pine_percentage : ℚ) : 
  total = 350 → pine_percentage = 70 / 100 → 
  total - (total * pine_percentage).floor = 105 := by
  sorry

end non_pine_trees_l3853_385322


namespace x_less_than_y_l3853_385366

theorem x_less_than_y (a : ℝ) : (a + 3) * (a - 5) < (a + 2) * (a - 4) := by
  sorry

end x_less_than_y_l3853_385366


namespace special_sequence_sum_l3853_385355

/-- A sequence with specific initial conditions -/
def special_sequence : ℕ → ℚ := sorry

/-- The sum of the first n terms of the special sequence -/
def sum_n (n : ℕ) : ℚ := sorry

theorem special_sequence_sum :
  (special_sequence 1 = 2) →
  (sum_n 2 = 8) →
  (sum_n 3 = 20) →
  ∀ n : ℕ, sum_n n = n * (n + 1) * (2 * n + 4) / 3 := by sorry

end special_sequence_sum_l3853_385355


namespace tax_reduction_problem_l3853_385309

theorem tax_reduction_problem (T C : ℝ) (X : ℝ) 
  (h1 : X > 0 ∧ X < 100) -- Ensure X is a valid percentage
  (h2 : T > 0 ∧ C > 0)   -- Ensure initial tax and consumption are positive
  (h3 : T * (1 - X / 100) * C * 1.25 = 0.75 * T * C) -- Revenue equation
  : X = 40 := by
  sorry

end tax_reduction_problem_l3853_385309


namespace tom_car_washing_earnings_l3853_385301

/-- Represents the amount of money Tom had last week in dollars -/
def initial_amount : ℕ := 74

/-- Represents the amount of money Tom has now in dollars -/
def current_amount : ℕ := 86

/-- Represents the amount of money Tom made washing cars in dollars -/
def money_made : ℕ := current_amount - initial_amount

theorem tom_car_washing_earnings :
  money_made = current_amount - initial_amount :=
by sorry

end tom_car_washing_earnings_l3853_385301


namespace todd_total_gum_l3853_385362

/-- The number of gum pieces Todd has now, given his initial amount and the amount he received. -/
def total_gum (initial : ℕ) (received : ℕ) : ℕ := initial + received

/-- Todd's initial number of gum pieces -/
def todd_initial : ℕ := 38

/-- Number of gum pieces Todd received from Steve -/
def steve_gave : ℕ := 16

/-- Theorem stating that Todd's total gum pieces is 54 -/
theorem todd_total_gum : total_gum todd_initial steve_gave = 54 := by
  sorry

end todd_total_gum_l3853_385362


namespace jade_handled_83_transactions_l3853_385392

-- Define the number of transactions for each person
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := mabel_transactions + mabel_transactions / 10
def cal_transactions : ℕ := anthony_transactions * 2 / 3
def jade_transactions : ℕ := cal_transactions + 17

-- Theorem to prove
theorem jade_handled_83_transactions : jade_transactions = 83 := by
  sorry

end jade_handled_83_transactions_l3853_385392


namespace expression_change_l3853_385316

/-- The change in the expression x³ - 5x + 1 when x changes by b -/
def expressionChange (x b : ℝ) : ℝ :=
  let f := fun t => t^3 - 5*t + 1
  f (x + b) - f x

theorem expression_change (x b : ℝ) (h : b > 0) :
  expressionChange x b = 3*b*x^2 + 3*b^2*x + b^3 - 5*b ∨
  expressionChange x (-b) = -3*b*x^2 + 3*b^2*x - b^3 + 5*b :=
sorry

end expression_change_l3853_385316


namespace min_sum_given_log_condition_l3853_385382

theorem min_sum_given_log_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : Real.log a / Real.log 4 + Real.log b / Real.log 4 ≥ 5) : 
  a + b ≥ 64 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    Real.log a₀ / Real.log 4 + Real.log b₀ / Real.log 4 ≥ 5 ∧ a₀ + b₀ = 64 := by
  sorry

end min_sum_given_log_condition_l3853_385382


namespace minimum_point_of_translated_graph_l3853_385345

def f (x : ℝ) : ℝ := |x - 4| - 2 + 5

theorem minimum_point_of_translated_graph :
  ∀ x : ℝ, f x ≥ f 4 ∧ f 4 = 3 :=
sorry

end minimum_point_of_translated_graph_l3853_385345


namespace at_least_one_half_l3853_385394

theorem at_least_one_half (x y z : ℝ) 
  (h : x + y + z - 2*(x*y + y*z + x*z) + 4*x*y*z = 1/2) :
  x = 1/2 ∨ y = 1/2 ∨ z = 1/2 := by
sorry

end at_least_one_half_l3853_385394


namespace slope_dividing_area_l3853_385324

-- Define the vertices of the L-shaped region
def vertices : List (ℝ × ℝ) := [(0, 0), (0, 4), (4, 4), (4, 2), (7, 2), (7, 0)]

-- Define the L-shaped region
def l_shape (x y : ℝ) : Prop :=
  (0 ≤ x ∧ x ≤ 7 ∧ 0 ≤ y ∧ y ≤ 4) ∧
  (x ≤ 4 ∨ y ≤ 2)

-- Define the area of the L-shaped region
def area_l_shape : ℝ := 22

-- Define a line through the origin
def line_through_origin (m : ℝ) (x y : ℝ) : Prop :=
  y = m * x

-- Define the area above the line
def area_above_line (m : ℝ) : ℝ := 11

-- Theorem: The slope of the line that divides the area in half is -0.375
theorem slope_dividing_area :
  ∃ (m : ℝ), m = -0.375 ∧
    area_above_line m = area_l_shape / 2 ∧
    ∀ (x y : ℝ), l_shape x y → line_through_origin m x y →
      (y ≥ m * x → area_above_line m ≥ area_l_shape / 2) ∧
      (y ≤ m * x → area_above_line m ≤ area_l_shape / 2) :=
by sorry

end slope_dividing_area_l3853_385324


namespace can_cross_all_rivers_l3853_385363

def river1_width : ℕ := 487
def river2_width : ℕ := 621
def river3_width : ℕ := 376
def existing_bridge : ℕ := 295
def additional_material : ℕ := 1020

def extra_needed (river_width : ℕ) : ℕ :=
  if river_width > existing_bridge then river_width - existing_bridge else 0

theorem can_cross_all_rivers :
  extra_needed river1_width + extra_needed river2_width + extra_needed river3_width ≤ additional_material :=
by sorry

end can_cross_all_rivers_l3853_385363


namespace chocolate_bars_in_large_box_l3853_385327

theorem chocolate_bars_in_large_box :
  let small_boxes : ℕ := 16
  let bars_per_small_box : ℕ := 25
  let total_bars : ℕ := small_boxes * bars_per_small_box
  total_bars = 400 :=
by
  sorry

end chocolate_bars_in_large_box_l3853_385327


namespace exactly_one_absent_probability_l3853_385313

/-- The probability of a student being absent on a given day -/
def p_absent : ℚ := 1 / 15

/-- The probability of a student being present on a given day -/
def p_present : ℚ := 1 - p_absent

/-- The number of students chosen -/
def n : ℕ := 3

/-- The number of students that should be absent -/
def k : ℕ := 1

theorem exactly_one_absent_probability :
  (n.choose k : ℚ) * p_absent^k * p_present^(n - k) = 588 / 3375 := by
  sorry

end exactly_one_absent_probability_l3853_385313


namespace fraction_transformation_l3853_385331

theorem fraction_transformation (x : ℚ) : 
  x = 437 → (537 - x) / (463 + x) = 1 / 9 := by
  sorry

end fraction_transformation_l3853_385331


namespace lcm_gcf_problem_l3853_385300

theorem lcm_gcf_problem (n : ℕ+) :
  (Nat.lcm n 12 = 48) → (Nat.gcd n 12 = 8) → n = 32 := by
  sorry

end lcm_gcf_problem_l3853_385300


namespace min_pressure_cyclic_process_l3853_385357

-- Define the constants and variables
variable (V₀ T₀ a b c R : ℝ)
variable (V T P : ℝ → ℝ)

-- Define the cyclic process equation
def cyclic_process (t : ℝ) : Prop :=
  ((V t) / V₀ - a)^2 + ((T t) / T₀ - b)^2 = c^2

-- Define the ideal gas law
def ideal_gas_law (t : ℝ) : Prop :=
  (P t) * (V t) = R * (T t)

-- State the theorem
theorem min_pressure_cyclic_process
  (h1 : ∀ t, cyclic_process V₀ T₀ a b c V T t)
  (h2 : ∀ t, ideal_gas_law R V T P t)
  (h3 : c^2 < a^2 + b^2) :
  ∃ P_min : ℝ, ∀ t, P t ≥ P_min ∧ 
    P_min = (R * T₀ / V₀) * (a * Real.sqrt (a^2 + b^2 - c^2) - b * c) / 
      (b * Real.sqrt (a^2 + b^2 - c^2) + a * c) :=
sorry

end min_pressure_cyclic_process_l3853_385357


namespace max_value_d_l3853_385397

theorem max_value_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10) 
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) : 
  d ≤ (5 + Real.sqrt 105) / 2 := by
sorry

end max_value_d_l3853_385397


namespace min_value_expression_min_value_achievable_l3853_385341

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (9 * z) / (3 * x + 2 * y) + (9 * x) / (2 * y + 3 * z) + (4 * y) / (2 * x + z) ≥ 9 / 2 :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    (9 * z) / (3 * x + 2 * y) + (9 * x) / (2 * y + 3 * z) + (4 * y) / (2 * x + z) = 9 / 2 :=
by sorry

end min_value_expression_min_value_achievable_l3853_385341


namespace xiao_wang_total_score_l3853_385393

/-- Xiao Wang's jump rope scores -/
def score1 : ℕ := 23
def score2 : ℕ := 34
def score3 : ℕ := 29

/-- Theorem: The sum of Xiao Wang's three jump rope scores equals 86 -/
theorem xiao_wang_total_score : score1 + score2 + score3 = 86 := by
  sorry

end xiao_wang_total_score_l3853_385393


namespace unfair_die_theorem_l3853_385390

def unfair_die_expected_value (p1to6 p7 p8 : ℚ) : ℚ :=
  (1 * p1to6 + 2 * p1to6 + 3 * p1to6 + 4 * p1to6 + 5 * p1to6 + 6 * p1to6) +
  (7 * p7) + (8 * p8)

theorem unfair_die_theorem :
  let p1to6 : ℚ := 1 / 15
  let p7 : ℚ := 1 / 6
  let p8 : ℚ := 1 / 3
  unfair_die_expected_value p1to6 p7 p8 = 157 / 30 :=
by
  sorry

#eval unfair_die_expected_value (1/15) (1/6) (1/3)

end unfair_die_theorem_l3853_385390


namespace square_base_exponent_l3853_385388

theorem square_base_exponent (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (a^2)^(2*b) = a^b * y^b → y = a^3 := by sorry

end square_base_exponent_l3853_385388


namespace last_number_is_2802_l3853_385351

/-- Represents a piece of paper with a given width and height in characters. -/
structure Paper where
  width : Nat
  height : Nat

/-- Represents the space required to write a number, including the following space. -/
def spaceRequired (n : Nat) : Nat :=
  if n < 10 then 2
  else if n < 100 then 3
  else if n < 1000 then 4
  else 5

/-- The last number that can be fully written on the paper. -/
def lastNumberWritten (p : Paper) : Nat :=
  2802

/-- Theorem stating that 2802 is the last number that can be fully written on a 100x100 character paper. -/
theorem last_number_is_2802 (p : Paper) (h1 : p.width = 100) (h2 : p.height = 100) :
  lastNumberWritten p = 2802 := by
  sorry

end last_number_is_2802_l3853_385351


namespace ninas_age_l3853_385385

theorem ninas_age (lisa mike nina : ℝ) 
  (h1 : (lisa + mike + nina) / 3 = 12)
  (h2 : nina - 5 = 2 * lisa)
  (h3 : mike + 2 = (lisa + 2) / 2) :
  nina = 34.6 := by
  sorry

end ninas_age_l3853_385385


namespace sector_area_l3853_385396

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 16) (h2 : central_angle = 2) : 
  let radius := perimeter / (2 + central_angle)
  (1/2) * central_angle * radius^2 = 16 := by
  sorry

end sector_area_l3853_385396


namespace floor_area_closest_to_160000_l3853_385312

def hand_length : ℝ := 20

def floor_width (hl : ℝ) : ℝ := 18 * hl
def floor_length (hl : ℝ) : ℝ := 22 * hl

def floor_area (w l : ℝ) : ℝ := w * l

def closest_area : ℝ := 160000

theorem floor_area_closest_to_160000 :
  ∀ (ε : ℝ), ε > 0 →
  |floor_area (floor_width hand_length) (floor_length hand_length) - closest_area| < ε →
  ∀ (other_area : ℝ), other_area ≠ closest_area →
  |floor_area (floor_width hand_length) (floor_length hand_length) - other_area| ≥ ε :=
by sorry

end floor_area_closest_to_160000_l3853_385312


namespace intersection_line_of_given_circles_l3853_385356

/-- Circle with center and radius --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line equation of the form ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The intersection line of two circles --/
def intersection_line (c1 c2 : Circle) : Line :=
  sorry

theorem intersection_line_of_given_circles :
  let c1 : Circle := { center := (1, 5), radius := 7 }
  let c2 : Circle := { center := (-2, -1), radius := 5 * Real.sqrt 2 }
  let l : Line := intersection_line c1 c2
  l.a = 1 ∧ l.b = 1 ∧ l.c = 3 :=
sorry

end intersection_line_of_given_circles_l3853_385356


namespace max_min_value_sqrt_three_l3853_385367

theorem max_min_value_sqrt_three : 
  ∃ (M : ℝ), M > 0 ∧ 
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 
    min (min (min (1/a) (1/(b^2))) (1/(c^3))) (a + b^2 + c^3) ≤ M) ∧
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    min (min (min (1/a) (1/(b^2))) (1/(c^3))) (a + b^2 + c^3) = M) ∧
  M = Real.sqrt 3 :=
by
  sorry


end max_min_value_sqrt_three_l3853_385367


namespace total_money_together_l3853_385349

def henry_initial_money : ℕ := 5
def henry_earned_money : ℕ := 2
def friend_money : ℕ := 13

theorem total_money_together : 
  henry_initial_money + henry_earned_money + friend_money = 20 := by
  sorry

end total_money_together_l3853_385349


namespace election_votes_calculation_l3853_385395

theorem election_votes_calculation (total_votes : ℕ) : 
  (∃ (candidate_votes rival_votes : ℕ),
    candidate_votes = (40 * total_votes) / 100 ∧
    rival_votes = candidate_votes + 5000 ∧
    rival_votes + candidate_votes = total_votes) →
  total_votes = 25000 := by
sorry

end election_votes_calculation_l3853_385395


namespace all_statements_false_l3853_385337

theorem all_statements_false : ∀ (a b c d : ℝ),
  (¬((a ≠ b ∧ c ≠ d) → (a + c ≠ b + d))) ∧
  (¬((a + c ≠ b + d) → (a ≠ b ∧ c ≠ d))) ∧
  (¬(a = b ∧ c = d ∧ a + c ≠ b + d)) ∧
  (¬((a + c = b + d) → (a = b ∨ c = d))) :=
by sorry

end all_statements_false_l3853_385337


namespace remainder_of_2468135792_div_101_l3853_385359

theorem remainder_of_2468135792_div_101 :
  (2468135792 : ℕ) % 101 = 52 := by sorry

end remainder_of_2468135792_div_101_l3853_385359


namespace book_sale_price_l3853_385332

-- Define the total number of books
def total_books : ℕ := 150

-- Define the number of unsold books
def unsold_books : ℕ := 50

-- Define the total amount received
def total_amount : ℕ := 500

-- Define the fraction of books sold
def fraction_sold : ℚ := 2/3

-- Theorem to prove
theorem book_sale_price :
  let sold_books := total_books - unsold_books
  let price_per_book := total_amount / sold_books
  fraction_sold * total_books = sold_books ∧
  price_per_book = 5 := by
sorry

end book_sale_price_l3853_385332


namespace arithmetic_sequence_sum_l3853_385369

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3)^2 - 3*(a 3) - 5 = 0 →
  (a 10)^2 - 3*(a 10) - 5 = 0 →
  a 5 + a 8 = 3 :=
by
  sorry

end arithmetic_sequence_sum_l3853_385369


namespace percentage_loss_calculation_l3853_385364

def cost_price : ℝ := 800
def selling_price : ℝ := 680

theorem percentage_loss_calculation : 
  (cost_price - selling_price) / cost_price * 100 = 15 := by sorry

end percentage_loss_calculation_l3853_385364


namespace period_is_24_hours_period_in_hours_is_24_l3853_385387

/-- Represents the period in seconds --/
def period (birth_rate : ℚ) (death_rate : ℚ) (net_increase : ℕ) : ℚ :=
  net_increase / (birth_rate / 2 - death_rate / 2)

/-- Theorem stating that the period is 24 hours given the problem conditions --/
theorem period_is_24_hours :
  let birth_rate : ℚ := 10
  let death_rate : ℚ := 2
  let net_increase : ℕ := 345600
  period birth_rate death_rate net_increase = 86400 := by
  sorry

/-- Converts seconds to hours --/
def seconds_to_hours (seconds : ℚ) : ℚ :=
  seconds / 3600

/-- Theorem stating that 86400 seconds is equal to 24 hours --/
theorem period_in_hours_is_24 :
  seconds_to_hours 86400 = 24 := by
  sorry

end period_is_24_hours_period_in_hours_is_24_l3853_385387


namespace number_added_to_multiples_of_three_l3853_385308

theorem number_added_to_multiples_of_three : ∃ x : ℕ, 
  x + (3 * 14 + 3 * 15 + 3 * 18) = 152 ∧ x = 11 := by
  sorry

end number_added_to_multiples_of_three_l3853_385308


namespace geometric_sequence_common_ratio_l3853_385360

/-- Given a geometric sequence {a_n} with a_1 = 1/2 and a_4 = -4, prove that the common ratio q is -2. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) 
  (h_a1 : a 1 = 1/2) 
  (h_a4 : a 4 = -4) 
  : q = -2 := by
  sorry

end geometric_sequence_common_ratio_l3853_385360


namespace absolute_value_inequality_solution_set_l3853_385370

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| ≤ 1} = Set.Icc 0 2 := by sorry

end absolute_value_inequality_solution_set_l3853_385370


namespace semicircle_in_quarter_circle_l3853_385304

theorem semicircle_in_quarter_circle (r : ℝ) (hr : r > 0) :
  let s := r * Real.sqrt 3
  let quarter_circle_area := π * s^2 / 4
  let semicircle_area := π * r^2 / 2
  semicircle_area / quarter_circle_area = 2 / 3 :=
by sorry

end semicircle_in_quarter_circle_l3853_385304


namespace triple_hash_90_l3853_385320

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.3 * N + 2

-- State the theorem
theorem triple_hash_90 : hash (hash (hash 90)) = 5.21 := by
  sorry

end triple_hash_90_l3853_385320


namespace grade_assignment_count_l3853_385371

theorem grade_assignment_count (num_students : ℕ) (num_grades : ℕ) :
  num_students = 12 → num_grades = 4 →
  (num_grades : ℕ) ^ num_students = 16777216 :=
by
  sorry

end grade_assignment_count_l3853_385371


namespace square_side_length_l3853_385365

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) (square_side : ℝ) :
  rectangle_length = 9 →
  rectangle_width = 16 →
  rectangle_length * rectangle_width = square_side * square_side →
  square_side = 12 := by
sorry

end square_side_length_l3853_385365


namespace allocation_methods_l3853_385311

/-- The number of warriors in the class -/
def total_warriors : ℕ := 6

/-- The number of tasks to be completed -/
def num_tasks : ℕ := 4

/-- The number of leadership positions (captain and vice-captain) -/
def leadership_positions : ℕ := 2

/-- The number of participating warriors -/
def participating_warriors : ℕ := 4

theorem allocation_methods :
  (leadership_positions.choose 1) *
  ((total_warriors - leadership_positions).choose (participating_warriors - 1)) *
  (participating_warriors.factorial) = 192 :=
sorry

end allocation_methods_l3853_385311


namespace double_inequality_solution_l3853_385361

theorem double_inequality_solution (x : ℝ) : 
  (4 * x + 2 > (x - 1)^2 ∧ (x - 1)^2 > 3 * x + 6) ↔ 
  (x > 3 + 2 * Real.sqrt 10 ∧ x < (5 + 3 * Real.sqrt 5) / 2) :=
sorry

end double_inequality_solution_l3853_385361


namespace fundraising_shortfall_l3853_385329

def goal : ℕ := 10000

def ken_raised : ℕ := 800

theorem fundraising_shortfall (mary_raised scott_raised amy_raised : ℕ) 
  (h1 : mary_raised = 5 * ken_raised)
  (h2 : mary_raised = 3 * scott_raised)
  (h3 : amy_raised = 2 * ken_raised)
  (h4 : amy_raised = scott_raised / 2)
  : ken_raised + mary_raised + scott_raised + amy_raised = goal - 400 := by
  sorry

end fundraising_shortfall_l3853_385329


namespace right_triangle_area_l3853_385354

/-- A right-angled triangle with an altitude from the right angle -/
structure RightTriangleWithAltitude where
  /-- The length of one leg of the triangle -/
  a : ℝ
  /-- The length of the other leg of the triangle -/
  b : ℝ
  /-- The radius of the inscribed circle in one of the smaller triangles -/
  r₁ : ℝ
  /-- The radius of the inscribed circle in the other smaller triangle -/
  r₂ : ℝ
  /-- Ensure the radii are positive -/
  h_positive_r₁ : r₁ > 0
  h_positive_r₂ : r₂ > 0
  /-- The ratio of the legs is equal to the ratio of the radii -/
  h_ratio : a / b = r₁ / r₂

/-- The theorem stating the area of the right-angled triangle -/
theorem right_triangle_area (t : RightTriangleWithAltitude) (h_r₁ : t.r₁ = 3) (h_r₂ : t.r₂ = 4) : 
  (1/2) * t.a * t.b = 150 := by
  sorry


end right_triangle_area_l3853_385354


namespace percentage_difference_l3853_385307

theorem percentage_difference (A B C : ℝ) 
  (hB_C : B = 0.63 * C) 
  (hB_A : B = 0.90 * A) : 
  A = 0.70 * C := by
  sorry

end percentage_difference_l3853_385307


namespace min_value_of_f_l3853_385318

/-- The quadratic function f(x) = 3(x+2)^2 - 5 -/
def f (x : ℝ) : ℝ := 3 * (x + 2)^2 - 5

/-- The minimum value of f(x) is -5 -/
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ -5 ∧ ∃ x₀ : ℝ, f x₀ = -5 :=
by sorry

end min_value_of_f_l3853_385318


namespace investment_interest_l3853_385391

/-- Calculate simple interest --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_interest (y : ℝ) : 
  simple_interest 3000 (y / 100) 2 = 60 * y := by
  sorry

end investment_interest_l3853_385391


namespace day_284_is_saturday_l3853_385352

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the day of the week for a given day number -/
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem day_284_is_saturday (h : dayOfWeek 25 = DayOfWeek.Saturday) :
  dayOfWeek 284 = DayOfWeek.Saturday := by
  sorry

end day_284_is_saturday_l3853_385352


namespace shell_collection_l3853_385372

theorem shell_collection (laurie_shells : ℕ) (h1 : laurie_shells = 36) :
  ∃ (ben_shells alan_shells : ℕ),
    ben_shells = laurie_shells / 3 ∧
    alan_shells = ben_shells * 4 ∧
    alan_shells = 48 := by
  sorry

end shell_collection_l3853_385372


namespace right_isosceles_triangle_circle_segment_area_l3853_385323

theorem right_isosceles_triangle_circle_segment_area :
  let hypotenuse : ℝ := 10
  let radius : ℝ := hypotenuse / 2
  let sector_angle : ℝ := 45 -- in degrees
  let sector_area : ℝ := (sector_angle / 360) * π * radius^2
  let triangle_area : ℝ := (1 / 2) * radius^2
  let shaded_area : ℝ := sector_area - triangle_area
  let a : ℝ := 25
  let b : ℝ := 50
  let c : ℝ := 1
  (shaded_area = a * π - b * Real.sqrt c) ∧ (a + b + c = 76) := by
  sorry

end right_isosceles_triangle_circle_segment_area_l3853_385323


namespace quadratic_equation_properties_l3853_385380

theorem quadratic_equation_properties (m : ℝ) (hm : m ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 - (m^2 + 2)*x + m^2 + 1
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - 2*x₁ - 1 = m^2 - 2 :=
by
  sorry

end quadratic_equation_properties_l3853_385380


namespace balanced_polynomial_existence_balanced_polynomial_equality_l3853_385335

-- Define what it means for an integer to be balanced
def IsBalanced (n : ℤ) : Prop :=
  n = 1 ∨ ∃ (k : ℕ) (p : List ℤ), k % 2 = 0 ∧ n = p.prod ∧ ∀ x ∈ p, Nat.Prime x.natAbs

-- Define the polynomial P(x) = (x+a)(x+b)
def P (a b : ℤ) (x : ℤ) : ℤ := (x + a) * (x + b)

theorem balanced_polynomial_existence :
  ∃ (a b : ℤ), a ≠ b ∧ a > 0 ∧ b > 0 ∧ ∀ n : ℤ, 1 ≤ n ∧ n ≤ 50 → IsBalanced (P a b n) :=
sorry

theorem balanced_polynomial_equality (a b : ℤ) (h : ∀ n : ℤ, IsBalanced (P a b n)) :
  a = b :=
sorry

end balanced_polynomial_existence_balanced_polynomial_equality_l3853_385335


namespace polynomial_symmetry_representation_l3853_385353

theorem polynomial_symmetry_representation
  (p : ℝ → ℝ) (a : ℝ)
  (h_symmetry : ∀ x, p x = p (a - x)) :
  ∃ h : ℝ → ℝ, ∀ x, p x = h ((x - a / 2) ^ 2) :=
sorry

end polynomial_symmetry_representation_l3853_385353


namespace max_jogs_is_seven_l3853_385375

/-- Represents the number of items Bill buys -/
structure BillsPurchase where
  jags : Nat
  jigs : Nat
  jogs : Nat
  jugs : Nat

/-- Calculates the total cost of Bill's purchase -/
def totalCost (p : BillsPurchase) : Nat :=
  2 * p.jags + 3 * p.jigs + 8 * p.jogs + 5 * p.jugs

/-- Represents a valid purchase satisfying all conditions -/
def isValidPurchase (p : BillsPurchase) : Prop :=
  p.jags ≥ 1 ∧ p.jigs ≥ 1 ∧ p.jogs ≥ 1 ∧ p.jugs ≥ 1 ∧ totalCost p = 72

theorem max_jogs_is_seven :
  ∀ p : BillsPurchase, isValidPurchase p → p.jogs ≤ 7 :=
by sorry

end max_jogs_is_seven_l3853_385375


namespace perpendicular_lines_a_equals_one_l3853_385317

/-- Given two lines l₁: ax + (3-a)y + 1 = 0 and l₂: 2x - y = 0,
    if l₁ is perpendicular to l₂, then a = 1 -/
theorem perpendicular_lines_a_equals_one (a : ℝ) :
  (∀ x y : ℝ, a * x + (3 - a) * y + 1 = 0 → 2 * x - y = 0 → 
    (a * 2 + (-1) * (3 - a) = 0)) → a = 1 := by
  sorry

end perpendicular_lines_a_equals_one_l3853_385317


namespace carol_college_distance_l3853_385347

/-- The distance between Carol's college and home -/
def college_distance (fuel_efficiency : ℝ) (tank_capacity : ℝ) (remaining_distance : ℝ) : ℝ :=
  fuel_efficiency * tank_capacity + remaining_distance

/-- Theorem stating the distance between Carol's college and home -/
theorem carol_college_distance :
  college_distance 20 16 100 = 420 := by
  sorry

end carol_college_distance_l3853_385347


namespace f_properties_l3853_385376

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (h2 : ∀ x : ℝ, x > 0 → f x < 0)

-- Theorem statement
theorem f_properties : (∀ x : ℝ, f (-x) = -f x) ∧
                       (∀ x y : ℝ, x < y → f y < f x) := by
  sorry

end f_properties_l3853_385376
