import Mathlib

namespace NUMINAMATH_CALUDE_probability_is_one_third_l2598_259827

/-- The set of digits used to form the number -/
def digits : Finset Nat := {2, 4, 6, 7}

/-- A function to check if a number is odd -/
def isOdd (n : Nat) : Bool := n % 2 = 1

/-- A function to check if a number is not a multiple of 3 -/
def notMultipleOf3 (n : Nat) : Bool := n % 3 ≠ 0

/-- The set of all four-digit numbers that can be formed using the given digits -/
def allNumbers : Finset Nat := sorry

/-- The set of favorable numbers (odd with hundreds digit not multiple of 3) -/
def favorableNumbers : Finset Nat := sorry

/-- The probability of forming a favorable number -/
def probability : Rat := (Finset.card favorableNumbers : Rat) / (Finset.card allNumbers : Rat)

theorem probability_is_one_third :
  probability = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l2598_259827


namespace NUMINAMATH_CALUDE_pythagorean_triple_for_eleven_l2598_259880

theorem pythagorean_triple_for_eleven : ∃ b c : ℕ, 11^2 + b^2 = c^2 ∧ c = 61 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_for_eleven_l2598_259880


namespace NUMINAMATH_CALUDE_min_investment_amount_l2598_259862

/-- Represents an investment plan with two interest rates -/
structure InvestmentPlan where
  amount_at_7_percent : ℝ
  amount_at_12_percent : ℝ

/-- Calculates the total interest earned from an investment plan -/
def total_interest (plan : InvestmentPlan) : ℝ :=
  0.07 * plan.amount_at_7_percent + 0.12 * plan.amount_at_12_percent

/-- Calculates the total investment amount -/
def total_investment (plan : InvestmentPlan) : ℝ :=
  plan.amount_at_7_percent + plan.amount_at_12_percent

/-- Theorem: The minimum total investment amount is $25,000 -/
theorem min_investment_amount :
  ∀ (plan : InvestmentPlan),
    plan.amount_at_7_percent ≤ 11000 →
    total_interest plan ≥ 2450 →
    total_investment plan ≥ 25000 :=
by sorry

end NUMINAMATH_CALUDE_min_investment_amount_l2598_259862


namespace NUMINAMATH_CALUDE_second_smallest_number_l2598_259885

def digits : List Nat := [1, 5, 6, 9]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n / 10 = 5) ∧ (n % 10 ∈ digits) ∧ (n / 10 ∈ digits)

def count_smaller (n : Nat) : Nat :=
  (digits.filter (λ d => d < n % 10)).length

theorem second_smallest_number :
  ∃ n : Nat, is_valid_number n ∧ count_smaller n = 1 ∧ n = 56 := by
  sorry

end NUMINAMATH_CALUDE_second_smallest_number_l2598_259885


namespace NUMINAMATH_CALUDE_accuracy_of_0_598_l2598_259836

/-- Represents the place value of a digit in a decimal number. -/
inductive PlaceValue
  | Ones
  | Tenths
  | Hundredths
  | Thousandths
  | TenThousandths
  deriving Repr

/-- Determines the place value of accuracy for a given decimal number. -/
def placeOfAccuracy (n : Float) : PlaceValue :=
  match n.toString.split (· = '.') with
  | [_, fractional] =>
    match fractional.length with
    | 1 => PlaceValue.Tenths
    | 2 => PlaceValue.Hundredths
    | 3 => PlaceValue.Thousandths
    | _ => PlaceValue.TenThousandths
  | _ => PlaceValue.Ones

/-- Theorem: The approximate number 0.598 is accurate to the thousandths place. -/
theorem accuracy_of_0_598 :
  placeOfAccuracy 0.598 = PlaceValue.Thousandths := by
  sorry

end NUMINAMATH_CALUDE_accuracy_of_0_598_l2598_259836


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2598_259894

theorem arithmetic_mean_problem : 
  let a := 9/16
  let b := 3/4
  let c := 5/8
  c = (a + b) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2598_259894


namespace NUMINAMATH_CALUDE_tortoise_wins_l2598_259871

-- Define the race distance
def race_distance : ℝ := 100

-- Define the animals
inductive Animal
| tortoise
| hare

-- Define the speed function for each animal
def speed (a : Animal) (t : ℝ) : ℝ :=
  match a with
  | Animal.tortoise => sorry -- Increasing speed function
  | Animal.hare => sorry -- Piecewise function for hare's speed

-- Define the position function for each animal
def position (a : Animal) (t : ℝ) : ℝ :=
  sorry -- Integral of speed function

-- Define the finish time for each animal
def finish_time (a : Animal) : ℝ :=
  sorry -- Time when position equals race_distance

-- Theorem stating the tortoise wins
theorem tortoise_wins :
  finish_time Animal.tortoise < finish_time Animal.hare :=
sorry


end NUMINAMATH_CALUDE_tortoise_wins_l2598_259871


namespace NUMINAMATH_CALUDE_rectangle_arrangement_perimeter_bounds_l2598_259824

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents an arrangement of rectangles -/
structure Arrangement where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of an arrangement -/
def perimeter (a : Arrangement) : ℝ :=
  2 * (a.length + a.width)

/-- The set of all possible arrangements of four 7x5 rectangles -/
def possible_arrangements : Set Arrangement :=
  sorry

theorem rectangle_arrangement_perimeter_bounds :
  let r : Rectangle := { length := 7, width := 5 }
  let arrangements := possible_arrangements
  ∃ (max_arr min_arr : Arrangement),
    max_arr ∈ arrangements ∧
    min_arr ∈ arrangements ∧
    (∀ a ∈ arrangements, perimeter a ≤ perimeter max_arr) ∧
    (∀ a ∈ arrangements, perimeter a ≥ perimeter min_arr) ∧
    perimeter max_arr = 66 ∧
    perimeter min_arr = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_perimeter_bounds_l2598_259824


namespace NUMINAMATH_CALUDE_profit_and_pricing_analysis_l2598_259809

/-- Represents the daily sales quantity as a function of selling price -/
def sales_quantity (x : ℝ) : ℝ := -2 * x + 200

/-- Represents the profit as a function of selling price -/
def profit (x : ℝ) : ℝ := (x - 50) * (sales_quantity x)

/-- Represents the new profit function after cost price increase -/
def new_profit (x a : ℝ) : ℝ := (x - 50 - a) * (sales_quantity x)

theorem profit_and_pricing_analysis 
  (cost_price : ℝ) 
  (a : ℝ) 
  (h1 : cost_price = 50) 
  (h2 : a > 0) :
  (∃ x₁ x₂, profit x₁ = 800 ∧ profit x₂ = 800 ∧ x₁ ≠ x₂) ∧ 
  (∃ x_max, ∀ x, profit x ≤ profit x_max) ∧
  (∃ x, 50 + a ≤ x ∧ x ≤ 70 ∧ new_profit x a = 960 ∧ a = 4) := by
  sorry


end NUMINAMATH_CALUDE_profit_and_pricing_analysis_l2598_259809


namespace NUMINAMATH_CALUDE_nautical_mile_conversion_l2598_259803

/-- Proves that under given conditions, one nautical mile equals 1.15 land miles -/
theorem nautical_mile_conversion (speed_one_sail : ℝ) (speed_two_sails : ℝ) 
  (time_one_sail : ℝ) (time_two_sails : ℝ) (total_distance : ℝ) :
  speed_one_sail = 25 →
  speed_two_sails = 50 →
  time_one_sail = 4 →
  time_two_sails = 4 →
  total_distance = 345 →
  speed_one_sail * time_one_sail + speed_two_sails * time_two_sails = total_distance →
  (1 : ℝ) * (345 / 300) = 1.15 := by
  sorry

#check nautical_mile_conversion

end NUMINAMATH_CALUDE_nautical_mile_conversion_l2598_259803


namespace NUMINAMATH_CALUDE_jack_king_ace_probability_l2598_259882

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the event of drawing three specific cards in order -/
def draw_three_cards (d : Deck) (first second third : Fin 52) : ℚ :=
  (4 : ℚ) / 52 * (4 : ℚ) / 51 * (4 : ℚ) / 50

/-- The probability of drawing a Jack, then a King, then an Ace from a standard deck without replacement -/
theorem jack_king_ace_probability (d : Deck) :
  ∃ (j k a : Fin 52), draw_three_cards d j k a = 16 / 33150 :=
sorry

end NUMINAMATH_CALUDE_jack_king_ace_probability_l2598_259882


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l2598_259860

theorem value_of_a_minus_b (a b : ℚ) 
  (eq1 : 2025 * a + 2030 * b = 2035)
  (eq2 : 2027 * a + 2032 * b = 2037) : 
  a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l2598_259860


namespace NUMINAMATH_CALUDE_olivia_chocolate_sales_l2598_259843

/-- The amount of money Olivia made selling chocolate bars -/
def olivia_money (total_bars : ℕ) (unsold_bars : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

theorem olivia_chocolate_sales : olivia_money 7 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_olivia_chocolate_sales_l2598_259843


namespace NUMINAMATH_CALUDE_work_completion_time_l2598_259852

/-- Given that Ravi can do a piece of work in 15 days and Prakash can do it in 30 days,
    prove that they will finish it together in 10 days. -/
theorem work_completion_time (ravi_time prakash_time : ℝ) (h1 : ravi_time = 15) (h2 : prakash_time = 30) :
  1 / (1 / ravi_time + 1 / prakash_time) = 10 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2598_259852


namespace NUMINAMATH_CALUDE_central_angle_doubles_when_radius_halves_l2598_259823

theorem central_angle_doubles_when_radius_halves (r l α β : ℝ) (h1 : r > 0) (h2 : l > 0) (h3 : α > 0) :
  α = l / r →
  β = l / (r / 2) →
  β = 2 * α := by
sorry

end NUMINAMATH_CALUDE_central_angle_doubles_when_radius_halves_l2598_259823


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2598_259884

/-- A function f: ℝ → ℝ is an "H function" if for any two distinct real numbers x₁ and x₂,
    the condition x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁ holds. -/
def is_H_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function f: ℝ → ℝ is strictly increasing if for any two real numbers x₁ and x₂,
    x₁ < x₂ implies f x₁ < f x₂. -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

/-- Theorem: A function is an "H function" if and only if it is strictly increasing. -/
theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_H_function f ↔ strictly_increasing f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2598_259884


namespace NUMINAMATH_CALUDE_tree_growth_relation_l2598_259869

/-- The height of a tree after a number of months -/
def tree_height (initial_height growth_rate : ℝ) (months : ℝ) : ℝ :=
  initial_height + growth_rate * months

/-- Theorem: The height of the tree after x months is 80 + 2x -/
theorem tree_growth_relation (x : ℝ) :
  tree_height 80 2 x = 80 + 2 * x := by sorry

end NUMINAMATH_CALUDE_tree_growth_relation_l2598_259869


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l2598_259875

theorem trigonometric_expression_value :
  Real.sin (315 * π / 180) * Real.sin (-1260 * π / 180) + 
  Real.cos (390 * π / 180) * Real.sin (-1020 * π / 180) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l2598_259875


namespace NUMINAMATH_CALUDE_max_min_y_over_x_l2598_259887

theorem max_min_y_over_x :
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 3 →
  (∀ (z w : ℝ), (z - 2)^2 + w^2 = 3 → w / z ≤ Real.sqrt 3) ∧
  (∀ (z w : ℝ), (z - 2)^2 + w^2 = 3 → w / z ≥ -Real.sqrt 3) ∧
  (∃ (x₁ y₁ : ℝ), (x₁ - 2)^2 + y₁^2 = 3 ∧ y₁ / x₁ = Real.sqrt 3) ∧
  (∃ (x₂ y₂ : ℝ), (x₂ - 2)^2 + y₂^2 = 3 ∧ y₂ / x₂ = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_max_min_y_over_x_l2598_259887


namespace NUMINAMATH_CALUDE_friendly_seq_uniqueness_l2598_259861

/-- A sequence of strictly increasing natural numbers -/
def IncreasingSeq := ℕ → ℕ

/-- Two sequences are friendly if every natural number is represented exactly once as their product -/
def Friendly (a b : IncreasingSeq) : Prop :=
  ∀ n : ℕ, ∃! (i j : ℕ), n = a i * b j

/-- The theorem stating that one friendly sequence uniquely determines the other -/
theorem friendly_seq_uniqueness (a b c : IncreasingSeq) :
  Friendly a b → Friendly a c → b = c := by sorry

end NUMINAMATH_CALUDE_friendly_seq_uniqueness_l2598_259861


namespace NUMINAMATH_CALUDE_chime_2500_date_l2598_259878

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a time with hour and minute -/
structure Time where
  hour : ℕ
  minute : ℕ

/-- Calculates the number of chimes from a given start time to midnight -/
def chimesToMidnight (startTime : Time) : ℕ :=
  sorry

/-- Calculates the number of chimes in a full day -/
def chimesPerDay : ℕ :=
  sorry

/-- Calculates the date of the nth chime given a start date and time -/
def dateOfNthChime (n : ℕ) (startDate : Date) (startTime : Time) : Date :=
  sorry

/-- Theorem stating that the 2500th chime occurs on January 21, 2023 -/
theorem chime_2500_date :
  let startDate := Date.mk 2023 1 1
  let startTime := Time.mk 14 30
  dateOfNthChime 2500 startDate startTime = Date.mk 2023 1 21 :=
sorry

end NUMINAMATH_CALUDE_chime_2500_date_l2598_259878


namespace NUMINAMATH_CALUDE_min_value_implies_a_eq_one_exp_log_sin_positive_l2598_259867

noncomputable section

variable (x : ℝ)
variable (a : ℝ)

def f (x : ℝ) : ℝ := Real.log x + a / x - 1

theorem min_value_implies_a_eq_one (h : ∀ x > 0, f x ≥ 0) (h' : ∃ x > 0, f x = 0) : a = 1 :=
sorry

theorem exp_log_sin_positive : ∀ x > 0, Real.exp x + (Real.log x - 1) * Real.sin x > 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_eq_one_exp_log_sin_positive_l2598_259867


namespace NUMINAMATH_CALUDE_point_2_4_is_D_l2598_259874

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the diagram points
def F : Point2D := ⟨5, 5⟩
def D : Point2D := ⟨2, 4⟩

-- Theorem statement
theorem point_2_4_is_D : 
  ∃ (p : Point2D), p.x = 2 ∧ p.y = 4 ∧ p = D :=
sorry

end NUMINAMATH_CALUDE_point_2_4_is_D_l2598_259874


namespace NUMINAMATH_CALUDE_tangent_line_properties_l2598_259838

open Real

theorem tangent_line_properties (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₁ ≠ 1) :
  (∃ (k b : ℝ), 
    (∀ x, k * x + b = (1 / x₁) * x - 1 + log x₁) ∧
    (∀ x, k * x + b = exp x₂ * x + exp x₂ * (1 - x₂))) →
  (x₁ * exp x₂ = 1 ∧ (x₁ + 1) / (x₁ - 1) + x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l2598_259838


namespace NUMINAMATH_CALUDE_relay_team_count_l2598_259831

/-- The number of sprinters --/
def total_sprinters : ℕ := 6

/-- The number of sprinters to be selected --/
def selected_sprinters : ℕ := 4

/-- The number of ways to form the relay team --/
def relay_team_formations : ℕ := 252

/-- Theorem stating the number of ways to form the relay team --/
theorem relay_team_count :
  (total_sprinters = 6) →
  (selected_sprinters = 4) →
  (∃ A B : ℕ, A ≠ B ∧ A ≤ total_sprinters ∧ B ≤ total_sprinters) →
  relay_team_formations = 252 :=
by sorry

end NUMINAMATH_CALUDE_relay_team_count_l2598_259831


namespace NUMINAMATH_CALUDE_score_not_above_average_l2598_259846

structure ClassData where
  participants : ℕ
  mean : ℝ
  median : ℝ
  mode : ℝ
  variance : ℝ
  excellenceRate : ℝ

def class901 : ClassData :=
  { participants := 40
  , mean := 75
  , median := 78
  , mode := 77
  , variance := 158
  , excellenceRate := 0.2 }

def class902 : ClassData :=
  { participants := 45
  , mean := 75
  , median := 76
  , mode := 74
  , variance := 122
  , excellenceRate := 0.2 }

theorem score_not_above_average (score : ℝ) :
  score = 77 → ¬(score > class902.mean) := by
  sorry

end NUMINAMATH_CALUDE_score_not_above_average_l2598_259846


namespace NUMINAMATH_CALUDE_probability_is_three_fourths_l2598_259850

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The probability that a point (x, y) satisfies x + 2y < 4 when randomly and uniformly chosen from the given square --/
def probabilityLessThan4 (s : Square) : ℝ :=
  sorry

/-- The square with vertices (0, 0), (0, 3), (3, 3), and (3, 0) --/
def givenSquare : Square :=
  { bottomLeft := (0, 0), topRight := (3, 3) }

theorem probability_is_three_fourths :
  probabilityLessThan4 givenSquare = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_three_fourths_l2598_259850


namespace NUMINAMATH_CALUDE_worker_efficiency_l2598_259895

/-- Given two workers A and B, where A is half as efficient as B,
    this theorem proves that if they together complete a job in 13 days,
    then B alone can complete the job in 19.5 days. -/
theorem worker_efficiency (A B : ℝ) (h1 : A = (1/2) * B) (h2 : (A + B) * 13 = 1) :
  (1 / B) = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_worker_efficiency_l2598_259895


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l2598_259856

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l2598_259856


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2598_259814

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : B ∩ (U \ A) = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2598_259814


namespace NUMINAMATH_CALUDE_painted_cubes_equality_l2598_259806

theorem painted_cubes_equality (n : ℕ) (h : n > 2) :
  2 * ((n - 2) * (n - 1) + (n - 2) * n + (n - 1) * n) = (n - 2) * (n - 1) * n ↔ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_equality_l2598_259806


namespace NUMINAMATH_CALUDE_ellipse_properties_l2598_259834

/-- Ellipse C: x^2/4 + y^2 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Circle: x^2 + y^2 = 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Line passing through F2 -/
def line_through_F2 (x y : ℝ) (m : ℝ) : Prop := y = m * x

/-- Line 2mx - 2y - 2m + 1 = 0 -/
def intersecting_line (x y : ℝ) (m : ℝ) : Prop := 2*m*x - 2*y - 2*m + 1 = 0

/-- Theorem stating the properties of the ellipse -/
theorem ellipse_properties :
  ∃ (F1 F2 : ℝ × ℝ),
    (∀ x y : ℝ, ellipse_C x y →
      (∃ A B : ℝ × ℝ, 
        line_through_F2 A.1 A.2 (F2.2 / F2.1) ∧
        line_through_F2 B.1 B.2 (F2.2 / F2.1) ∧
        ellipse_C A.1 A.2 ∧
        ellipse_C B.1 B.2 ∧
        (Real.sqrt ((A.1 - F1.1)^2 + (A.2 - F1.2)^2) +
         Real.sqrt ((B.1 - F1.1)^2 + (B.2 - F1.2)^2) +
         Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8))) ∧
    (∀ m : ℝ, ∃ x y : ℝ, ellipse_C x y ∧ intersecting_line x y m) ∧
    (∀ P Q : ℝ × ℝ, 
      ellipse_C P.1 P.2 →
      unit_circle Q.1 Q.2 →
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ 3) ∧
    (∃ P Q : ℝ × ℝ,
      ellipse_C P.1 P.2 ∧
      unit_circle Q.1 Q.2 ∧
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2598_259834


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2598_259842

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log x}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x^2)}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2598_259842


namespace NUMINAMATH_CALUDE_goshawk_eurasian_nature_reserve_birds_l2598_259821

theorem goshawk_eurasian_nature_reserve_birds (B : ℝ) (h : B > 0) :
  let hawks := 0.30 * B
  let non_hawks := B - hawks
  let paddyfield_warblers := 0.40 * non_hawks
  let other_birds := 0.35 * B
  let kingfishers := B - hawks - paddyfield_warblers - other_birds
  kingfishers / paddyfield_warblers = 0.25
:= by sorry

end NUMINAMATH_CALUDE_goshawk_eurasian_nature_reserve_birds_l2598_259821


namespace NUMINAMATH_CALUDE_field_length_width_ratio_l2598_259844

theorem field_length_width_ratio :
  ∀ (w : ℝ),
    w > 0 →
    24 > 0 →
    ∃ (k : ℕ), 24 = k * w →
    36 = (1/8) * (24 * w) →
    24 / w = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_length_width_ratio_l2598_259844


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_powers_l2598_259877

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Definition of x -/
noncomputable def x : ℂ := (-1 + i * Real.sqrt 3) / 2

/-- Definition of y -/
noncomputable def y : ℂ := (-1 - i * Real.sqrt 3) / 2

/-- Main theorem -/
theorem cube_roots_of_unity_powers :
  (x ^ 5 + y ^ 5 = -2) ∧
  (x ^ 7 + y ^ 7 = 2) ∧
  (x ^ 9 + y ^ 9 = -2) ∧
  (x ^ 11 + y ^ 11 = 2) ∧
  (x ^ 13 + y ^ 13 = -2) :=
by sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_powers_l2598_259877


namespace NUMINAMATH_CALUDE_rectangle_not_always_similar_l2598_259891

-- Define the shapes
structure Square :=
  (side : ℝ)

structure IsoscelesRightTriangle :=
  (leg : ℝ)

structure Rectangle :=
  (length width : ℝ)

structure EquilateralTriangle :=
  (side : ℝ)

-- Define similarity for each shape
def similar_squares (s1 s2 : Square) : Prop :=
  true

def similar_isosceles_right_triangles (t1 t2 : IsoscelesRightTriangle) : Prop :=
  true

def similar_rectangles (r1 r2 : Rectangle) : Prop :=
  r1.length / r1.width = r2.length / r2.width

def similar_equilateral_triangles (e1 e2 : EquilateralTriangle) : Prop :=
  true

-- Theorem statement
theorem rectangle_not_always_similar :
  ∃ r1 r2 : Rectangle, ¬(similar_rectangles r1 r2) ∧
  (∀ s1 s2 : Square, similar_squares s1 s2) ∧
  (∀ t1 t2 : IsoscelesRightTriangle, similar_isosceles_right_triangles t1 t2) ∧
  (∀ e1 e2 : EquilateralTriangle, similar_equilateral_triangles e1 e2) :=
sorry

end NUMINAMATH_CALUDE_rectangle_not_always_similar_l2598_259891


namespace NUMINAMATH_CALUDE_birds_on_fence_l2598_259868

theorem birds_on_fence (initial_birds : ℝ) (birds_flown_away : ℝ) :
  initial_birds = 12.0 →
  birds_flown_away = 8.0 →
  initial_birds - birds_flown_away = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l2598_259868


namespace NUMINAMATH_CALUDE_no_articles_in_general_context_l2598_259879

/-- Represents the possible article choices for a noun in a sentence -/
inductive Article
  | Definite   -- represents "the"
  | Indefinite -- represents "a" or "an"
  | None       -- represents no article

/-- Represents the context of a sentence -/
inductive Context
  | General
  | Specific

/-- Represents a noun in the sentence -/
inductive Noun
  | College
  | Prison

/-- Determines the correct article for a noun given the context -/
def correctArticle (context : Context) (noun : Noun) : Article :=
  match context, noun with
  | Context.General, _ => Article.None
  | Context.Specific, _ => Article.Definite

/-- The main theorem stating that in a general context, 
    both "college" and "prison" should have no article -/
theorem no_articles_in_general_context : 
  ∀ (context : Context),
    context = Context.General →
    correctArticle context Noun.College = Article.None ∧
    correctArticle context Noun.Prison = Article.None :=
by sorry

end NUMINAMATH_CALUDE_no_articles_in_general_context_l2598_259879


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2598_259812

theorem polynomial_simplification (x : ℝ) :
  (x^3 + 4*x^2 - 7*x + 11) + (-4*x^4 - x^3 + x^2 + 7*x + 3) + (3*x^4 - 2*x^3 + 5*x - 1) =
  -x^4 - 2*x^3 + 5*x^2 + 5*x + 13 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2598_259812


namespace NUMINAMATH_CALUDE_debate_team_boys_l2598_259847

theorem debate_team_boys (total : ℕ) (girls : ℕ) (groups : ℕ) :
  total % 9 = 0 →
  total / 9 = groups →
  girls = 46 →
  groups = 8 →
  total - girls = 26 :=
by sorry

end NUMINAMATH_CALUDE_debate_team_boys_l2598_259847


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2598_259820

theorem inequality_and_equality_condition 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) 
  (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) 
  (hb₁ : b₁ > 0) (hb₂ : b₂ > 0) (hb₃ : b₃ > 0) : 
  (a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂)^2 ≥ 
    4 * (a₁ * a₂ + a₂ * a₃ + a₃ * a₁) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁) ∧ 
  ((a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂)^2 = 
    4 * (a₁ * a₂ + a₂ * a₃ + a₃ * a₁) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁) ↔ 
    a₁ * b₂ = a₂ * b₁ ∧ a₂ * b₃ = a₃ * b₂ ∧ a₃ * b₁ = a₁ * b₃) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2598_259820


namespace NUMINAMATH_CALUDE_trout_catch_total_l2598_259810

theorem trout_catch_total (people : ℕ) (individual_share : ℕ) (h1 : people = 2) (h2 : individual_share = 9) :
  people * individual_share = 18 := by
  sorry

end NUMINAMATH_CALUDE_trout_catch_total_l2598_259810


namespace NUMINAMATH_CALUDE_empty_container_mass_l2598_259888

/-- The mass of an empty container, given its mass when filled with kerosene and water, and the densities of kerosene and water. -/
theorem empty_container_mass
  (mass_with_kerosene : ℝ)
  (mass_with_water : ℝ)
  (density_water : ℝ)
  (density_kerosene : ℝ)
  (h1 : mass_with_kerosene = 20)
  (h2 : mass_with_water = 24)
  (h3 : density_water = 1000)
  (h4 : density_kerosene = 800) :
  ∃ (empty_mass : ℝ), empty_mass = 4 ∧
  mass_with_kerosene = empty_mass + density_kerosene * ((mass_with_water - mass_with_kerosene) / (density_water - density_kerosene)) ∧
  mass_with_water = empty_mass + density_water * ((mass_with_water - mass_with_kerosene) / (density_water - density_kerosene)) :=
by
  sorry


end NUMINAMATH_CALUDE_empty_container_mass_l2598_259888


namespace NUMINAMATH_CALUDE_rowing_time_ratio_l2598_259896

/-- Proves that the ratio of time taken to row upstream to downstream is 2:1 
    given the man's rowing speed in still water and the current speed. -/
theorem rowing_time_ratio 
  (man_speed : ℝ) 
  (current_speed : ℝ) 
  (h1 : man_speed = 3.9)
  (h2 : current_speed = 1.3) :
  (man_speed - current_speed) / (man_speed + current_speed) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rowing_time_ratio_l2598_259896


namespace NUMINAMATH_CALUDE_survey_result_l2598_259892

def survey (total : ℕ) (neither : ℕ) (enjoyed : ℕ) (understood : ℕ) : Prop :=
  total = 600 ∧ 
  neither = 150 ∧
  enjoyed = understood ∧
  enjoyed + neither = total

theorem survey_result (total neither enjoyed understood : ℕ) 
  (h : survey total neither enjoyed understood) : 
  (enjoyed : ℚ) / total = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_survey_result_l2598_259892


namespace NUMINAMATH_CALUDE_stratified_sampling_survey_l2598_259815

theorem stratified_sampling_survey (teachers : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (sample_female : ℕ) (sample_size : ℕ) : 
  teachers = 200 → 
  male_students = 1200 → 
  female_students = 1000 → 
  sample_female = 80 → 
  sample_size * (female_students / (teachers + male_students + female_students)) = sample_female → 
  sample_size = 192 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_survey_l2598_259815


namespace NUMINAMATH_CALUDE_strawberry_picking_l2598_259816

theorem strawberry_picking (basket_capacity : ℕ) (picked_ratio : ℚ) : 
  basket_capacity = 60 → 
  picked_ratio = 4/5 → 
  (basket_capacity / picked_ratio : ℚ) * 5 = 75 := by
sorry

end NUMINAMATH_CALUDE_strawberry_picking_l2598_259816


namespace NUMINAMATH_CALUDE_adam_quarters_l2598_259819

/-- The number of quarters Adam spent at the arcade -/
def quarters_spent : ℕ := 9

/-- The number of quarters Adam had left over -/
def quarters_left : ℕ := 79

/-- The initial number of quarters Adam had -/
def initial_quarters : ℕ := quarters_spent + quarters_left

theorem adam_quarters : initial_quarters = 88 := by
  sorry

end NUMINAMATH_CALUDE_adam_quarters_l2598_259819


namespace NUMINAMATH_CALUDE_third_fourth_product_l2598_259858

def arithmetic_sequence (a : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => arithmetic_sequence a d n + d

theorem third_fourth_product (a : ℝ) (d : ℝ) :
  arithmetic_sequence a d 5 = 17 ∧ d = 2 →
  (arithmetic_sequence a d 2) * (arithmetic_sequence a d 3) = 143 := by
sorry

end NUMINAMATH_CALUDE_third_fourth_product_l2598_259858


namespace NUMINAMATH_CALUDE_nisos_population_estimate_l2598_259876

/-- The initial population of Nisos in the year 2000 -/
def initial_population : ℕ := 400

/-- The number of years between 2000 and 2030 -/
def years_passed : ℕ := 30

/-- The number of years it takes for the population to double -/
def doubling_period : ℕ := 20

/-- The estimated population of Nisos in 2030 -/
def estimated_population_2030 : ℕ := 1131

/-- Theorem stating that the estimated population of Nisos in 2030 is approximately 1131 -/
theorem nisos_population_estimate :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  (initial_population : ℝ) * (2 : ℝ) ^ (years_passed / doubling_period : ℝ) ∈ 
  Set.Icc (estimated_population_2030 - ε) (estimated_population_2030 + ε) :=
sorry

end NUMINAMATH_CALUDE_nisos_population_estimate_l2598_259876


namespace NUMINAMATH_CALUDE_ceiling_abs_negative_l2598_259839

theorem ceiling_abs_negative : ⌈|(-52.7 : ℝ)|⌉ = 53 := by sorry

end NUMINAMATH_CALUDE_ceiling_abs_negative_l2598_259839


namespace NUMINAMATH_CALUDE_bridge_length_l2598_259830

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 170 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 205 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l2598_259830


namespace NUMINAMATH_CALUDE_no_perfect_square_exists_l2598_259855

theorem no_perfect_square_exists (a : ℕ) : 
  (∃ k : ℕ, ((a^2 - 3)^3 + 1)^a - 1 = k^2) → False ∧
  (∃ k : ℕ, ((a^2 - 3)^3 + 1)^(a+1) - 1 = k^2) → False :=
by sorry

end NUMINAMATH_CALUDE_no_perfect_square_exists_l2598_259855


namespace NUMINAMATH_CALUDE_practice_schedule_l2598_259849

theorem practice_schedule (trumpet flute piano : ℕ) 
  (h_trumpet : trumpet = 11)
  (h_flute : flute = 3)
  (h_piano : piano = 7) :
  Nat.lcm trumpet (Nat.lcm flute piano) = 231 := by
  sorry

end NUMINAMATH_CALUDE_practice_schedule_l2598_259849


namespace NUMINAMATH_CALUDE_rectangle_to_square_l2598_259808

-- Define the rectangle dimensions
def rectangle_length : ℕ := 9
def rectangle_width : ℕ := 4

-- Define the number of parts
def num_parts : ℕ := 3

-- Define the square side length
def square_side : ℕ := 6

-- Theorem statement
theorem rectangle_to_square :
  ∃ (part1 part2 part3 : ℕ × ℕ),
    -- The parts fit within the original rectangle
    part1.1 ≤ rectangle_length ∧ part1.2 ≤ rectangle_width ∧
    part2.1 ≤ rectangle_length ∧ part2.2 ≤ rectangle_width ∧
    part3.1 ≤ rectangle_length ∧ part3.2 ≤ rectangle_width ∧
    -- The total area of the parts equals the area of the original rectangle
    part1.1 * part1.2 + part2.1 * part2.2 + part3.1 * part3.2 = rectangle_length * rectangle_width ∧
    -- The parts can form a square
    (part1.1 = square_side ∨ part1.2 = square_side) ∧
    (part2.1 + part3.1 = square_side ∨ part2.2 + part3.2 = square_side) :=
by sorry

#check rectangle_to_square

end NUMINAMATH_CALUDE_rectangle_to_square_l2598_259808


namespace NUMINAMATH_CALUDE_parallel_iff_perpendicular_iff_l2598_259859

-- Define the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * m * y + 2 * m = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := 
  ∀ x y z w, l1 m x y ∧ l2 m z w → (x - z) * (m - 2) = m * (y - w)

-- Define perpendicular lines
def perpendicular (m : ℝ) : Prop := 
  ∀ x y z w, l1 m x y ∧ l2 m z w → (x - z) * (z - x) + m * (y - w) * (w - y) = 0

-- Theorem for parallel lines
theorem parallel_iff : 
  ∀ m : ℝ, parallel m ↔ m = 0 ∨ m = 5 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_iff : 
  ∀ m : ℝ, perpendicular m ↔ m = -1 ∨ m = 2/3 :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_perpendicular_iff_l2598_259859


namespace NUMINAMATH_CALUDE_niles_collection_l2598_259841

/-- The total amount collected by Niles from the book club -/
def total_collected (num_members : ℕ) (snack_fee : ℕ) (num_hardcover : ℕ) (hardcover_price : ℕ) (num_paperback : ℕ) (paperback_price : ℕ) : ℕ :=
  num_members * (snack_fee + num_hardcover * hardcover_price + num_paperback * paperback_price)

/-- Theorem stating the total amount collected by Niles -/
theorem niles_collection : total_collected 6 150 6 30 6 12 = 2412 := by
  sorry

end NUMINAMATH_CALUDE_niles_collection_l2598_259841


namespace NUMINAMATH_CALUDE_birthday_money_ratio_l2598_259890

theorem birthday_money_ratio : 
  let aunt_money : ℚ := 75
  let grandfather_money : ℚ := 150
  let bank_money : ℚ := 45
  let total_money := aunt_money + grandfather_money
  (bank_money / total_money) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_birthday_money_ratio_l2598_259890


namespace NUMINAMATH_CALUDE_city_partition_theorem_l2598_259805

/-- A directed graph where each vertex has outdegree 2 -/
structure CityGraph (V : Type) :=
  (edges : V → V → Prop)
  (outdegree_two : ∀ v : V, ∃ u w : V, u ≠ w ∧ edges v u ∧ edges v w ∧ ∀ x : V, edges v x → (x = u ∨ x = w))

/-- A partition of the vertices into 1014 sets -/
def ValidPartition (V : Type) (G : CityGraph V) :=
  ∃ (f : V → Fin 1014),
    (∀ v w : V, G.edges v w → f v ≠ f w) ∧
    (∀ i j : Fin 1014, i ≠ j →
      (∀ v w : V, f v = i ∧ f w = j → G.edges v w) ∨
      (∀ v w : V, f v = i ∧ f w = j → G.edges w v))

/-- The main theorem: every CityGraph has a ValidPartition -/
theorem city_partition_theorem (V : Type) (G : CityGraph V) :
  ValidPartition V G :=
sorry

end NUMINAMATH_CALUDE_city_partition_theorem_l2598_259805


namespace NUMINAMATH_CALUDE_box_number_problem_l2598_259864

theorem box_number_problem (a b c d e : ℕ) 
  (sum_all : a + b + c + d + e = 35)
  (sum_first_three : a + b + c = 22)
  (sum_last_three : c + d + e = 25)
  (first_box : a = 3)
  (last_box : e = 4) :
  b * d = 63 := by
  sorry

end NUMINAMATH_CALUDE_box_number_problem_l2598_259864


namespace NUMINAMATH_CALUDE_ducks_joined_l2598_259835

theorem ducks_joined (initial_ducks final_ducks : ℕ) (h : final_ducks ≥ initial_ducks) :
  final_ducks - initial_ducks = final_ducks - initial_ducks :=
by sorry

end NUMINAMATH_CALUDE_ducks_joined_l2598_259835


namespace NUMINAMATH_CALUDE_line_y_intercept_l2598_259845

/-- Given a line passing through points (3,2), (1,k), and (-4,1), 
    prove that its y-intercept is 11/7 -/
theorem line_y_intercept (k : ℚ) : 
  (∃ m b : ℚ, (3 : ℚ) * m + b = 2 ∧ 1 * m + b = k ∧ (-4 : ℚ) * m + b = 1) → 
  (∃ m b : ℚ, (3 : ℚ) * m + b = 2 ∧ 1 * m + b = k ∧ (-4 : ℚ) * m + b = 1 ∧ b = 11/7) :=
by sorry

end NUMINAMATH_CALUDE_line_y_intercept_l2598_259845


namespace NUMINAMATH_CALUDE_students_taking_physics_or_chemistry_but_not_both_l2598_259832

theorem students_taking_physics_or_chemistry_but_not_both 
  (both : ℕ) 
  (physics : ℕ) 
  (only_chemistry : ℕ) 
  (h1 : both = 12) 
  (h2 : physics = 22) 
  (h3 : only_chemistry = 9) : 
  (physics - both) + only_chemistry = 19 := by
sorry

end NUMINAMATH_CALUDE_students_taking_physics_or_chemistry_but_not_both_l2598_259832


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2598_259802

/-- Given a circle with area 225π cm², its diameter is 30 cm. -/
theorem circle_diameter_from_area : 
  ∀ (r : ℝ), r > 0 → π * r^2 = 225 * π → 2 * r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2598_259802


namespace NUMINAMATH_CALUDE_product_and_multiply_l2598_259829

theorem product_and_multiply : (3.6 * 0.25) * 0.4 = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_product_and_multiply_l2598_259829


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2598_259897

theorem quadratic_inequality (x : ℝ) : x^2 + 5*x + 6 > 0 ↔ x < -3 ∨ x > -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2598_259897


namespace NUMINAMATH_CALUDE_q_at_4_equals_6_l2598_259807

-- Define the function q(x)
def q (x : ℝ) : ℝ := |x - 3|^(1/3) + 3*|x - 3|^(1/5) + 2

-- Theorem statement
theorem q_at_4_equals_6 : q 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_q_at_4_equals_6_l2598_259807


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2598_259837

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a_7 · a_19 = 8, then a_3 · a_23 = 8 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geom : geometric_sequence a) 
    (h_prod : a 7 * a 19 = 8) : 
  a 3 * a 23 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2598_259837


namespace NUMINAMATH_CALUDE_max_leftover_stickers_l2598_259811

theorem max_leftover_stickers (y : ℕ+) : 
  ∃ (q r : ℕ), y = 12 * q + r ∧ r < 12 ∧ r ≤ 11 ∧ 
  ∀ (q' r' : ℕ), y = 12 * q' + r' ∧ r' < 12 → r' ≤ r :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_stickers_l2598_259811


namespace NUMINAMATH_CALUDE_infinite_series_solution_l2598_259893

theorem infinite_series_solution (x : ℝ) : 
  (∑' n, (2*n + 1) * x^n) = 16 → x = 5/8 := by sorry

end NUMINAMATH_CALUDE_infinite_series_solution_l2598_259893


namespace NUMINAMATH_CALUDE_circle_tangent_y_axis_a_value_l2598_259889

/-- A circle is tangent to the y-axis if and only if the absolute value of its center's x-coordinate equals its radius -/
axiom circle_tangent_y_axis {a r : ℝ} (h : ∀ x y : ℝ, (x - a)^2 + (y + 4)^2 = r^2) :
  (∃ y : ℝ, (0 - a)^2 + (y + 4)^2 = r^2) ↔ |a| = r

/-- If a circle with equation (x-a)^2+(y+4)^2=9 is tangent to the y-axis, then a = 3 or a = -3 -/
theorem circle_tangent_y_axis_a_value (h : ∀ x y : ℝ, (x - a)^2 + (y + 4)^2 = 9) 
  (tangent : ∃ y : ℝ, (0 - a)^2 + (y + 4)^2 = 9) : 
  a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_y_axis_a_value_l2598_259889


namespace NUMINAMATH_CALUDE_stability_comparison_l2598_259873

/-- Represents an athlete's performance in a series of tests -/
structure AthletePerformance where
  average_score : ℝ
  variance : ℝ

/-- Defines stability of performance based on variance -/
def more_stable (a b : AthletePerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two athletes with the same average score,
    the one with lower variance has more stable performance -/
theorem stability_comparison 
  (athlete_A athlete_B : AthletePerformance)
  (h_same_average : athlete_A.average_score = athlete_B.average_score)
  (h_A_variance : athlete_A.variance = 1.2)
  (h_B_variance : athlete_B.variance = 1) :
  more_stable athlete_B athlete_A :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l2598_259873


namespace NUMINAMATH_CALUDE_equal_ratios_sum_ratio_l2598_259813

theorem equal_ratios_sum_ratio (x y z : ℚ) : 
  x / 2 = y / 3 ∧ y / 3 = z / 4 → (x + y + z) / (2 * z) = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_sum_ratio_l2598_259813


namespace NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l2598_259853

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 221 → (∀ a b : ℕ, a^2 - b^2 = 221 → x^2 + y^2 ≤ a^2 + b^2) → 
  x^2 + y^2 = 24421 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l2598_259853


namespace NUMINAMATH_CALUDE_triangle_perimeter_with_inscribed_circles_l2598_259817

/-- The perimeter of an equilateral triangle inscribing three circles -/
theorem triangle_perimeter_with_inscribed_circles (r : ℝ) :
  r > 0 →
  let side_length := 4 * r + 4 * r * Real.sqrt 3
  3 * side_length = 12 * r * Real.sqrt 3 + 48 * r :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_with_inscribed_circles_l2598_259817


namespace NUMINAMATH_CALUDE_boy_age_problem_l2598_259800

theorem boy_age_problem (present_age : ℕ) (h : present_age = 16) : 
  ∃ (years_ago : ℕ), 
    (present_age + 4 = 2 * (present_age - years_ago)) ∧ 
    (present_age - years_ago = (present_age + 4) / 2) ∧
    years_ago = 6 := by
  sorry

end NUMINAMATH_CALUDE_boy_age_problem_l2598_259800


namespace NUMINAMATH_CALUDE_complete_square_factorize_l2598_259857

-- Problem 1: Complete the square
theorem complete_square (x p : ℝ) : x^2 + 2*p*x + 1 = (x + p)^2 + (1 - p^2) := by sorry

-- Problem 2: Factorization
theorem factorize (a b : ℝ) : a^2 - b^2 + 4*a + 2*b + 3 = (a + b + 1)*(a - b + 3) := by sorry

end NUMINAMATH_CALUDE_complete_square_factorize_l2598_259857


namespace NUMINAMATH_CALUDE_inscribed_squares_inequality_l2598_259818

/-- Given a triangle ABC with semiperimeter s and area F, and squares with side lengths x, y, and z
    inscribed such that:
    - Square with side x has two vertices on BC
    - Square with side y has two vertices on AC
    - Square with side z has two vertices on AB
    The sum of the reciprocals of their side lengths is less than or equal to s(2+√3)/(2F) -/
theorem inscribed_squares_inequality (s F x y z : ℝ) (h_pos : s > 0 ∧ F > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0) :
  1/x + 1/y + 1/z ≤ s * (2 + Real.sqrt 3) / (2 * F) := by
  sorry


end NUMINAMATH_CALUDE_inscribed_squares_inequality_l2598_259818


namespace NUMINAMATH_CALUDE_a_explicit_formula_l2598_259840

def a : ℕ → ℤ
  | 0 => -1
  | 1 => -3
  | 2 => -5
  | 3 => 5
  | (n + 4) => 8 * a (n + 3) - 22 * a (n + 2) + 24 * a (n + 1) - 9 * a n

theorem a_explicit_formula (n : ℕ) :
  a n = 2 + n - 3^(n + 1) + n * 3^n :=
by sorry

end NUMINAMATH_CALUDE_a_explicit_formula_l2598_259840


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2598_259886

theorem inequality_solution_set (x : ℝ) : 3 * x + 2 > 5 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2598_259886


namespace NUMINAMATH_CALUDE_distance_difference_l2598_259881

-- Define the distances
def mart_to_home : ℕ := 800
def home_to_academy : ℕ := 1300  -- 1 km + 300 m = 1000 m + 300 m = 1300 m
def academy_to_restaurant : ℕ := 1700

-- Theorem to prove
theorem distance_difference :
  (mart_to_home + home_to_academy) - academy_to_restaurant = 400 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l2598_259881


namespace NUMINAMATH_CALUDE_inverse_mod_89_l2598_259863

theorem inverse_mod_89 (h : (16⁻¹ : ZMod 89) = 28) : (256⁻¹ : ZMod 89) = 56 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_89_l2598_259863


namespace NUMINAMATH_CALUDE_square_EFGH_product_l2598_259822

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A square on the grid -/
structure GridSquare where
  E : GridPoint
  F : GridPoint
  G : GridPoint
  H : GridPoint

/-- The side length of a square given two of its corners -/
def sideLength (p1 p2 : GridPoint) : ℤ :=
  max (abs (p1.x - p2.x)) (abs (p1.y - p2.y))

/-- The area of a square -/
def area (s : GridSquare) : ℤ :=
  (sideLength s.E s.F) ^ 2

/-- The perimeter of a square -/
def perimeter (s : GridSquare) : ℤ :=
  4 * (sideLength s.E s.F)

theorem square_EFGH_product :
  ∃ (s : GridSquare),
    s.E = ⟨1, 5⟩ ∧
    s.F = ⟨5, 5⟩ ∧
    s.G = ⟨5, 1⟩ ∧
    s.H = ⟨1, 1⟩ ∧
    (area s * perimeter s = 256) := by
  sorry

end NUMINAMATH_CALUDE_square_EFGH_product_l2598_259822


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2598_259828

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (2 * a + I) / (1 - 2 * I) = b * I) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2598_259828


namespace NUMINAMATH_CALUDE_pizzas_bought_l2598_259865

def total_slices : ℕ := 32
def slices_left : ℕ := 7
def slices_per_pizza : ℕ := 8

theorem pizzas_bought : (total_slices - slices_left) / slices_per_pizza = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizzas_bought_l2598_259865


namespace NUMINAMATH_CALUDE_total_full_price_tickets_is_16525_l2598_259851

/-- Represents the ticket sales data for a play over three weeks -/
structure PlayTicketSales where
  total_tickets : ℕ
  week1_tickets : ℕ
  week2_tickets : ℕ
  week3_tickets : ℕ
  week2_full_price_ratio : ℕ
  week3_full_price_ratio : ℕ

/-- Calculates the total number of full-price tickets sold during the play's run -/
def total_full_price_tickets (sales : PlayTicketSales) : ℕ :=
  let week2_full_price := sales.week2_tickets * sales.week2_full_price_ratio / (sales.week2_full_price_ratio + 1)
  let week3_full_price := sales.week3_tickets * sales.week3_full_price_ratio / (sales.week3_full_price_ratio + 1)
  week2_full_price + week3_full_price

/-- Theorem stating that given the specific ticket sales data, the total number of full-price tickets is 16525 -/
theorem total_full_price_tickets_is_16525 (sales : PlayTicketSales) 
  (h1 : sales.total_tickets = 25200)
  (h2 : sales.week1_tickets = 5400)
  (h3 : sales.week2_tickets = 7200)
  (h4 : sales.week3_tickets = 13400)
  (h5 : sales.week2_full_price_ratio = 2)
  (h6 : sales.week3_full_price_ratio = 7) :
  total_full_price_tickets sales = 16525 := by
  sorry


end NUMINAMATH_CALUDE_total_full_price_tickets_is_16525_l2598_259851


namespace NUMINAMATH_CALUDE_prob_tails_at_least_twice_eq_half_l2598_259872

/-- Probability of getting tails k times in n flips of a fair coin -/
def binomialProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

/-- The number of coin flips -/
def numFlips : ℕ := 3

/-- Probability of getting tails at least twice but not more than 3 times in 3 flips -/
def probTailsAtLeastTwice : ℚ :=
  binomialProbability numFlips 2 + binomialProbability numFlips 3

theorem prob_tails_at_least_twice_eq_half :
  probTailsAtLeastTwice = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_tails_at_least_twice_eq_half_l2598_259872


namespace NUMINAMATH_CALUDE_workshop_inspection_problem_l2598_259898

-- Define the number of products produced each day
variable (n : ℕ)

-- Define the probability of passing inspection on the first day
def prob_pass_first_day : ℚ := 3/5

-- Define the probability of passing inspection on the second day
def prob_pass_second_day (n : ℕ) : ℚ := (n - 2).choose 4 / n.choose 4

-- Define the probability of passing inspection on both days
def prob_pass_both_days (n : ℕ) : ℚ := prob_pass_first_day * prob_pass_second_day n

-- Define the probability of passing inspection on at least one day
def prob_pass_at_least_one_day (n : ℕ) : ℚ := 1 - (1 - prob_pass_first_day) * (1 - prob_pass_second_day n)

-- Theorem statement
theorem workshop_inspection_problem (n : ℕ) :
  (prob_pass_first_day = (n - 1).choose 4 / n.choose 4) →
  (n = 10) ∧
  (prob_pass_both_days n = 1/5) ∧
  (prob_pass_at_least_one_day n = 11/15) := by
  sorry


end NUMINAMATH_CALUDE_workshop_inspection_problem_l2598_259898


namespace NUMINAMATH_CALUDE_walk_legs_count_l2598_259866

/-- The number of legs of a human -/
def human_legs : ℕ := 2

/-- The number of legs of a dog -/
def dog_legs : ℕ := 4

/-- The number of humans on the walk -/
def num_humans : ℕ := 2

/-- The number of dogs on the walk -/
def num_dogs : ℕ := 2

/-- The total number of legs of all organisms on the walk -/
def total_legs : ℕ := human_legs * num_humans + dog_legs * num_dogs

theorem walk_legs_count : total_legs = 12 := by
  sorry

end NUMINAMATH_CALUDE_walk_legs_count_l2598_259866


namespace NUMINAMATH_CALUDE_wall_bricks_l2598_259833

/-- Represents the number of bricks in the wall -/
def num_bricks : ℕ := 288

/-- Time taken by the first bricklayer to build the wall alone -/
def time1 : ℕ := 8

/-- Time taken by the second bricklayer to build the wall alone -/
def time2 : ℕ := 12

/-- Reduction in combined output when working together -/
def output_reduction : ℕ := 12

/-- Time taken by both bricklayers working together -/
def combined_time : ℕ := 6

theorem wall_bricks :
  (combined_time : ℚ) * ((num_bricks / time1 : ℚ) + (num_bricks / time2 : ℚ) - output_reduction) = num_bricks := by
  sorry

#check wall_bricks

end NUMINAMATH_CALUDE_wall_bricks_l2598_259833


namespace NUMINAMATH_CALUDE_cable_length_l2598_259883

/-- The length of a curve defined by the intersection of a plane and a sphere -/
theorem cable_length (x y z : ℝ) : 
  x + y + z = 10 → 
  x * y + y * z + x * z = -22 → 
  (4 * Real.pi * Real.sqrt (83 / 3)) = 
    (2 * Real.pi * Real.sqrt (144 - (10 ^ 2) / 3)) := by
  sorry

end NUMINAMATH_CALUDE_cable_length_l2598_259883


namespace NUMINAMATH_CALUDE_factor_expression_l2598_259801

theorem factor_expression (b : ℝ) : 275 * b^2 + 55 * b = 55 * b * (5 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2598_259801


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2598_259848

/-- Two 2D vectors are parallel if the cross product of their components is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (m, -1)
  let b : ℝ × ℝ := (1, m + 2)
  parallel a b → m = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2598_259848


namespace NUMINAMATH_CALUDE_lizzy_candy_spending_l2598_259854

/-- The amount of money Lizzy spent on candy --/
def candy_spent : ℕ := sorry

/-- The amount of money Lizzy received from her mother --/
def mother_gave : ℕ := 80

/-- The amount of money Lizzy received from her father --/
def father_gave : ℕ := 40

/-- The amount of money Lizzy received from her uncle --/
def uncle_gave : ℕ := 70

/-- The total amount of money Lizzy has now --/
def current_total : ℕ := 140

theorem lizzy_candy_spending :
  candy_spent = 50 ∧
  current_total = mother_gave + father_gave - candy_spent + uncle_gave :=
sorry

end NUMINAMATH_CALUDE_lizzy_candy_spending_l2598_259854


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2598_259825

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (n : ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a (n + 3) = 39 →
  a (n + 1) + a (n + 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2598_259825


namespace NUMINAMATH_CALUDE_boyden_family_children_l2598_259804

theorem boyden_family_children (adult_ticket_cost child_ticket_cost total_cost : ℕ) 
  (num_adults : ℕ) (h1 : adult_ticket_cost = child_ticket_cost + 6)
  (h2 : total_cost = 77) (h3 : adult_ticket_cost = 19) (h4 : num_adults = 2) :
  ∃ (num_children : ℕ), 
    num_children * child_ticket_cost + num_adults * adult_ticket_cost = total_cost ∧ 
    num_children = 3 := by
  sorry

end NUMINAMATH_CALUDE_boyden_family_children_l2598_259804


namespace NUMINAMATH_CALUDE_smallest_a_value_l2598_259870

theorem smallest_a_value (a b : ℕ) (r₁ r₂ r₃ : ℕ+) : 
  (∀ x : ℝ, x^3 - a*x^2 + b*x - 2310 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  r₁ * r₂ * r₃ = 2310 →
  a = r₁ + r₂ + r₃ →
  28 ≤ a :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2598_259870


namespace NUMINAMATH_CALUDE_max_value_under_constraints_l2598_259899

theorem max_value_under_constraints (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 12) : 
  2 * x + y ≤ 39 / 11 := by
  sorry

end NUMINAMATH_CALUDE_max_value_under_constraints_l2598_259899


namespace NUMINAMATH_CALUDE_vector_collinearity_l2598_259826

theorem vector_collinearity (k : ℝ) : 
  let a : Fin 2 → ℝ := ![Real.sqrt 3, 1]
  let b : Fin 2 → ℝ := ![0, -1]
  let c : Fin 2 → ℝ := ![k, Real.sqrt 3]
  (∃ (t : ℝ), a + 2 • b = t • c) → k = -3 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2598_259826
