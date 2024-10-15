import Mathlib

namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2121_212170

-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define permutation
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem problem_1 : (binomial 100 2 + binomial 100 97) / (permutation 101 3) = 1 / 6 := by
  sorry

theorem problem_2 : (Finset.sum (Finset.range 8) (λ i => binomial (i + 3) 3)) = 330 := by
  sorry

theorem problem_3 (n m : ℕ) (h : m ≤ n) : 
  (binomial (n + 1) m / binomial n m) - (binomial n (n - m + 1) / binomial n (n - m)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2121_212170


namespace NUMINAMATH_CALUDE_equation_solution_l2121_212124

theorem equation_solution (x y c : ℝ) : 
  7^(3*x - 1) * 3^(4*y - 3) = c^x * 27^y ∧ x + y = 4 → c = 49 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2121_212124


namespace NUMINAMATH_CALUDE_smallest_multiple_360_l2121_212190

theorem smallest_multiple_360 : ∀ n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 6 ∣ n ∧ 5 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n → n ≥ 360 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_360_l2121_212190


namespace NUMINAMATH_CALUDE_science_fiction_total_pages_l2121_212186

/-- The number of books in the science fiction section -/
def num_books : ℕ := 8

/-- The number of pages in each book -/
def pages_per_book : ℕ := 478

/-- The total number of pages in the science fiction section -/
def total_pages : ℕ := num_books * pages_per_book

theorem science_fiction_total_pages :
  total_pages = 3824 := by sorry

end NUMINAMATH_CALUDE_science_fiction_total_pages_l2121_212186


namespace NUMINAMATH_CALUDE_oligarch_wealth_comparison_l2121_212134

/-- Represents the wealth of an oligarch at a given time -/
structure OligarchWealth where
  amount : ℝ
  year : ℕ
  name : String

/-- Represents the national wealth of the country -/
def NationalWealth : Type := ℝ

/-- The problem statement -/
theorem oligarch_wealth_comparison 
  (maximilian_2011 maximilian_2012 alejandro_2011 alejandro_2012 : OligarchWealth)
  (national_wealth : NationalWealth) :
  (alejandro_2012.amount = 2 * maximilian_2011.amount) →
  (maximilian_2012.amount < alejandro_2011.amount) →
  (national_wealth = alejandro_2012.amount + maximilian_2012.amount - alejandro_2011.amount - maximilian_2011.amount) →
  (maximilian_2011.amount > national_wealth) := by
  sorry

end NUMINAMATH_CALUDE_oligarch_wealth_comparison_l2121_212134


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l2121_212105

/-- A positive geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_min_value (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (a 6 = a 5 + 2 * a 4) →
  (Real.sqrt (a m * a n) = 4 * a 1) →
  (∃ min_value : ℝ, min_value = (3 + 2 * Real.sqrt 2) / 6 ∧
    ∀ x y : ℕ, (Real.sqrt (a x * a y) = 4 * a 1) → (1 / x + 2 / y) ≥ min_value) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_value_l2121_212105


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l2121_212139

def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem permutation_equation_solution (x : ℕ) : 
  (3 * A 8 x = 4 * A 9 (x - 1)) → x ≤ 8 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l2121_212139


namespace NUMINAMATH_CALUDE_weeks_to_buy_bike_l2121_212114

def bike_cost : ℕ := 650
def birthday_money : ℕ := 60 + 45 + 25
def weekly_earnings : ℕ := 20

theorem weeks_to_buy_bike :
  ∃ (weeks : ℕ), birthday_money + weeks * weekly_earnings = bike_cost ∧ weeks = 26 :=
by sorry

end NUMINAMATH_CALUDE_weeks_to_buy_bike_l2121_212114


namespace NUMINAMATH_CALUDE_solve_equation1_solve_equation2_l2121_212192

-- Define the equations
def equation1 (x : ℚ) : Prop := 5 * x - 2 * (x - 1) = 3
def equation2 (x : ℚ) : Prop := (x + 3) / 2 - 1 = (2 * x - 1) / 3

-- Theorem statements
theorem solve_equation1 : ∃ x : ℚ, equation1 x ∧ x = 1/3 := by sorry

theorem solve_equation2 : ∃ x : ℚ, equation2 x ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_solve_equation1_solve_equation2_l2121_212192


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2121_212135

theorem arithmetic_computation : 7^2 - 4*5 + 4^3 = 93 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2121_212135


namespace NUMINAMATH_CALUDE_root_implies_a_value_l2121_212145

theorem root_implies_a_value (a : ℝ) : 
  ((-2 : ℝ)^2 + 3*(-2) + a = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_value_l2121_212145


namespace NUMINAMATH_CALUDE_stadium_attendance_l2121_212125

/-- The number of people in a stadium at the start and end of a game -/
theorem stadium_attendance (boys_start girls_start boys_end girls_end : ℕ) :
  girls_start = 240 →
  boys_end = boys_start - boys_start / 4 →
  girls_end = girls_start - girls_start / 8 →
  boys_end + girls_end = 480 →
  boys_start + girls_start = 600 := by
sorry

end NUMINAMATH_CALUDE_stadium_attendance_l2121_212125


namespace NUMINAMATH_CALUDE_blue_marbles_after_replacement_l2121_212195

/-- Represents the distribution of marbles in a jar -/
structure MarbleDistribution where
  total : ℕ
  red : ℕ
  yellow : ℕ
  blue : ℕ
  purple : ℕ
  orange : ℕ
  green : ℕ

/-- Calculates the number of blue marbles after replacement -/
def blueMarbleCount (dist : MarbleDistribution) : ℕ :=
  dist.blue + dist.red / 3

theorem blue_marbles_after_replacement (dist : MarbleDistribution) 
  (h1 : dist.total = 160)
  (h2 : dist.red = 40)
  (h3 : dist.yellow = 32)
  (h4 : dist.blue = 16)
  (h5 : dist.purple = 24)
  (h6 : dist.orange = 8)
  (h7 : dist.green = 40)
  (h8 : dist.total = dist.red + dist.yellow + dist.blue + dist.purple + dist.orange + dist.green) :
  blueMarbleCount dist = 29 := by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_after_replacement_l2121_212195


namespace NUMINAMATH_CALUDE_johns_total_hours_l2121_212119

/-- Represents John's volunteering schedule for the year -/
structure VolunteerSchedule where
  jan_to_mar : Nat  -- Hours for January to March
  apr_to_jun : Nat  -- Hours for April to June
  jul_to_aug : Nat  -- Hours for July and August
  sep_to_oct : Nat  -- Hours for September and October
  november : Nat    -- Hours for November
  december : Nat    -- Hours for December
  bonus_days : Nat  -- Hours for bonus days (third Saturday of every month except May and June)
  charity_run : Nat -- Hours for annual charity run in June

/-- Calculates the total volunteering hours for the year -/
def total_hours (schedule : VolunteerSchedule) : Nat :=
  schedule.jan_to_mar +
  schedule.apr_to_jun +
  schedule.jul_to_aug +
  schedule.sep_to_oct +
  schedule.november +
  schedule.december +
  schedule.bonus_days +
  schedule.charity_run

/-- John's actual volunteering schedule for the year -/
def johns_schedule : VolunteerSchedule :=
  { jan_to_mar := 18
  , apr_to_jun := 24
  , jul_to_aug := 64
  , sep_to_oct := 24
  , november := 6
  , december := 6
  , bonus_days := 40
  , charity_run := 8
  }

/-- Theorem stating that John's total volunteering hours for the year is 190 -/
theorem johns_total_hours : total_hours johns_schedule = 190 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_hours_l2121_212119


namespace NUMINAMATH_CALUDE_points_form_circle_l2121_212152

theorem points_form_circle :
  ∀ (x y : ℝ), (∃ t : ℝ, x = Real.cos t ∧ y = Real.sin t) → x^2 + y^2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_points_form_circle_l2121_212152


namespace NUMINAMATH_CALUDE_total_candy_cases_is_80_l2121_212183

/-- The Sweet Shop gets a new candy shipment every 35 days. -/
def shipment_interval : ℕ := 35

/-- The number of cases of chocolate bars. -/
def chocolate_cases : ℕ := 25

/-- The number of cases of lollipops. -/
def lollipop_cases : ℕ := 55

/-- The total number of candy cases. -/
def total_candy_cases : ℕ := chocolate_cases + lollipop_cases

/-- Theorem stating that the total number of candy cases is 80. -/
theorem total_candy_cases_is_80 : total_candy_cases = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_candy_cases_is_80_l2121_212183


namespace NUMINAMATH_CALUDE_tan_70_cos_10_expression_l2121_212174

theorem tan_70_cos_10_expression : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_70_cos_10_expression_l2121_212174


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2121_212148

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2 * y - 5 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (-2, 1)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x - y + 5 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  ∃ (m b : ℝ),
    (∀ (x y : ℝ), perpendicular_line x y ↔ y = m * x + b) ∧
    (perpendicular_line point.1 point.2) ∧
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      given_line x₁ y₁ → given_line x₂ y₂ → x₁ ≠ x₂ →
      (y₂ - y₁) / (x₂ - x₁) * m = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2121_212148


namespace NUMINAMATH_CALUDE_circles_intersecting_parallel_lines_l2121_212141

-- Define the types for our objects
variable (Point Circle Line : Type)

-- Define the necessary relations and properties
variable (onCircle : Point → Circle → Prop)
variable (intersectsAt : Circle → Circle → Point → Prop)
variable (passesThrough : Line → Point → Prop)
variable (intersectsCircleAt : Line → Circle → Point → Prop)
variable (parallel : Line → Line → Prop)
variable (lineThroughPoints : Point → Point → Line)

-- State the theorem
theorem circles_intersecting_parallel_lines
  (Γ₁ Γ₂ : Circle)
  (P Q A A' B B' : Point) :
  intersectsAt Γ₁ Γ₂ P →
  intersectsAt Γ₁ Γ₂ Q →
  (∃ l : Line, passesThrough l P ∧ intersectsCircleAt l Γ₁ A ∧ intersectsCircleAt l Γ₂ A') →
  (∃ m : Line, passesThrough m Q ∧ intersectsCircleAt m Γ₁ B ∧ intersectsCircleAt m Γ₂ B') →
  A ≠ P →
  A' ≠ P →
  B ≠ Q →
  B' ≠ Q →
  parallel (lineThroughPoints A B) (lineThroughPoints A' B') :=
sorry

end NUMINAMATH_CALUDE_circles_intersecting_parallel_lines_l2121_212141


namespace NUMINAMATH_CALUDE_stella_profit_l2121_212104

def dolls : ℕ := 3
def clocks : ℕ := 2
def glasses : ℕ := 5

def doll_price : ℕ := 5
def clock_price : ℕ := 15
def glass_price : ℕ := 4

def total_cost : ℕ := 40

def total_sales : ℕ := dolls * doll_price + clocks * clock_price + glasses * glass_price

def profit : ℕ := total_sales - total_cost

theorem stella_profit : profit = 25 := by
  sorry

end NUMINAMATH_CALUDE_stella_profit_l2121_212104


namespace NUMINAMATH_CALUDE_complex_number_on_line_l2121_212136

theorem complex_number_on_line (a : ℝ) : 
  (Complex.I * (Complex.I⁻¹ * a + (1 - Complex.I) / 2)).re + 
  (Complex.I * (Complex.I⁻¹ * a + (1 - Complex.I) / 2)).im = 0 → 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_on_line_l2121_212136


namespace NUMINAMATH_CALUDE_guy_speed_increase_point_l2121_212162

/-- Represents the problem of finding the point where Guy increases his speed --/
theorem guy_speed_increase_point
  (total_distance : ℝ)
  (average_speed : ℝ)
  (first_half_speed : ℝ)
  (speed_increase : ℝ)
  (h1 : total_distance = 60)
  (h2 : average_speed = 30)
  (h3 : first_half_speed = 24)
  (h4 : speed_increase = 16) :
  let second_half_speed := first_half_speed + speed_increase
  let increase_point := (total_distance * first_half_speed) / (first_half_speed + second_half_speed)
  increase_point = 30 := by sorry

end NUMINAMATH_CALUDE_guy_speed_increase_point_l2121_212162


namespace NUMINAMATH_CALUDE_solve_equation_l2121_212106

theorem solve_equation (x : ℝ) : 3 * x + 20 = (1/3) * (7 * x + 45) → x = -7.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2121_212106


namespace NUMINAMATH_CALUDE_average_price_is_45_cents_l2121_212132

/-- Represents the fruit selection and pricing problem -/
structure FruitProblem where
  apple_price : ℕ
  orange_price : ℕ
  total_fruits : ℕ
  initial_avg_price : ℕ
  oranges_removed : ℕ

/-- Calculates the average price of remaining fruits -/
def average_price_after_removal (fp : FruitProblem) : ℚ :=
  sorry

/-- Theorem stating that the average price of remaining fruits is 45 cents -/
theorem average_price_is_45_cents (fp : FruitProblem) 
  (h1 : fp.apple_price = 40)
  (h2 : fp.orange_price = 60)
  (h3 : fp.total_fruits = 10)
  (h4 : fp.initial_avg_price = 54)
  (h5 : fp.oranges_removed = 6) :
  average_price_after_removal fp = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_price_is_45_cents_l2121_212132


namespace NUMINAMATH_CALUDE_katy_brownies_l2121_212199

theorem katy_brownies (x : ℕ) : 
  x + 2 * x = 15 → x = 5 := by sorry

end NUMINAMATH_CALUDE_katy_brownies_l2121_212199


namespace NUMINAMATH_CALUDE_burger_cost_is_100_l2121_212154

/-- The cost of a burger in cents -/
def burger_cost : ℕ := 100

/-- The cost of a soda in cents -/
def soda_cost : ℕ := 50

/-- Charles' purchase -/
def charles_purchase (b s : ℕ) : Prop := 4 * b + 3 * s = 550

/-- Alice's purchase -/
def alice_purchase (b s : ℕ) : Prop := 3 * b + 2 * s = 400

/-- Bill's purchase -/
def bill_purchase (b s : ℕ) : Prop := 2 * b + s = 250

theorem burger_cost_is_100 :
  charles_purchase burger_cost soda_cost ∧
  alice_purchase burger_cost soda_cost ∧
  bill_purchase burger_cost soda_cost ∧
  burger_cost = 100 :=
sorry

end NUMINAMATH_CALUDE_burger_cost_is_100_l2121_212154


namespace NUMINAMATH_CALUDE_base_subtraction_proof_l2121_212110

def to_base_10 (digits : List ℕ) (base : ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

theorem base_subtraction_proof :
  let base_7_num := to_base_10 [5, 2, 3] 7
  let base_5_num := to_base_10 [4, 6, 1] 5
  base_7_num - base_5_num = 107 := by sorry

end NUMINAMATH_CALUDE_base_subtraction_proof_l2121_212110


namespace NUMINAMATH_CALUDE_expression_equality_l2121_212171

theorem expression_equality : 
  (1 / 3) ^ 2000 * 27 ^ 669 + Real.sin (60 * π / 180) * Real.tan (60 * π / 180) + (2009 + Real.sin (25 * π / 180)) ^ 0 = 2 + 29 / 54 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2121_212171


namespace NUMINAMATH_CALUDE_complex_not_in_first_quadrant_l2121_212102

theorem complex_not_in_first_quadrant (a : ℝ) : 
  let z : ℂ := (a - Complex.I) / (1 + Complex.I)
  ¬ (z.re > 0 ∧ z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_not_in_first_quadrant_l2121_212102


namespace NUMINAMATH_CALUDE_product_equals_32_l2121_212128

theorem product_equals_32 : 
  (1 / 4 : ℚ) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_32_l2121_212128


namespace NUMINAMATH_CALUDE_derek_car_increase_l2121_212149

/-- Represents the number of dogs and cars Derek owns at a given time --/
structure DereksPets where
  dogs : ℕ
  cars : ℕ

/-- The change in Derek's pet ownership over 10 years --/
def petsChange (initial final : DereksPets) : ℕ := final.cars - initial.cars

/-- Theorem stating the increase in cars Derek owns over 10 years --/
theorem derek_car_increase :
  ∀ (initial final : DereksPets),
  initial.dogs = 90 →
  initial.dogs = 3 * initial.cars →
  final.dogs = 120 →
  final.cars = 2 * final.dogs →
  petsChange initial final = 210 := by
  sorry

end NUMINAMATH_CALUDE_derek_car_increase_l2121_212149


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l2121_212133

def U : Finset Nat := {1,2,3,4,5,6}
def M : Finset Nat := {1,3,4}

theorem complement_of_M_in_U : 
  (U \ M) = {2,5,6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l2121_212133


namespace NUMINAMATH_CALUDE_min_value_of_4x2_plus_y2_l2121_212198

theorem min_value_of_4x2_plus_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 6) :
  ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = 6 → 4 * x^2 + y^2 ≤ 4 * a^2 + b^2 ∧ 4 * x^2 + y^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_4x2_plus_y2_l2121_212198


namespace NUMINAMATH_CALUDE_tangent_slope_at_4_l2121_212143

def f (x : ℝ) : ℝ := x^3 - 7*x^2 + 1

theorem tangent_slope_at_4 : 
  (deriv f) 4 = -8 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_4_l2121_212143


namespace NUMINAMATH_CALUDE_red_marbles_after_replacement_l2121_212127

theorem red_marbles_after_replacement (total : ℕ) (blue green red : ℕ) : 
  total > 0 →
  blue = (40 * total + 99) / 100 →
  green = 20 →
  red = (10 * total + 99) / 100 →
  (15 * total + 99) / 100 + (5 * total + 99) / 100 + blue + green + red = total →
  (16 : ℕ) = red + blue / 3 := by
  sorry

end NUMINAMATH_CALUDE_red_marbles_after_replacement_l2121_212127


namespace NUMINAMATH_CALUDE_complex_square_equality_l2121_212115

theorem complex_square_equality (x y : ℕ+) : 
  (x + y * Complex.I) ^ 2 = 7 + 24 * Complex.I → x + y * Complex.I = 4 + 3 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_square_equality_l2121_212115


namespace NUMINAMATH_CALUDE_sequence_problem_l2121_212120

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n ≠ 0) →
  a 3 - (a 7)^2 / 2 + a 11 = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 1 * b 13 = 16 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l2121_212120


namespace NUMINAMATH_CALUDE_gear_teeth_problem_l2121_212109

theorem gear_teeth_problem (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 60) (h4 : 4 * x - 20 = 5 * y) (h5 : 5 * y = 10 * z) : x = 30 ∧ y = 20 ∧ z = 10 := by
  sorry

end NUMINAMATH_CALUDE_gear_teeth_problem_l2121_212109


namespace NUMINAMATH_CALUDE_opposite_of_four_l2121_212118

-- Define the concept of opposite (additive inverse)
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_four : opposite 4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_four_l2121_212118


namespace NUMINAMATH_CALUDE_boat_journey_time_l2121_212111

/-- Calculates the total journey time for a boat traveling upstream and downstream -/
theorem boat_journey_time 
  (distance : ℝ) 
  (initial_current_speed : ℝ) 
  (upstream_current_speed : ℝ) 
  (boat_still_speed : ℝ) 
  (headwind_speed_reduction : ℝ) : 
  let upstream_time := distance / (boat_still_speed - upstream_current_speed)
  let downstream_speed := (boat_still_speed - headwind_speed_reduction) + initial_current_speed
  let downstream_time := distance / downstream_speed
  upstream_time + downstream_time = 26.67 :=
by
  sorry

#check boat_journey_time 56 2 3 6 1

end NUMINAMATH_CALUDE_boat_journey_time_l2121_212111


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l2121_212181

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l2121_212181


namespace NUMINAMATH_CALUDE_pool_depth_relationship_l2121_212178

/-- The depth of Sarah's pool in feet -/
def sarahs_pool_depth : ℝ := 5

/-- The depth of John's pool in feet -/
def johns_pool_depth : ℝ := 15

/-- Theorem stating the relationship between John's and Sarah's pool depths -/
theorem pool_depth_relationship : 
  johns_pool_depth = 2 * sarahs_pool_depth + 5 ∧ sarahs_pool_depth = 5 := by
  sorry

end NUMINAMATH_CALUDE_pool_depth_relationship_l2121_212178


namespace NUMINAMATH_CALUDE_largest_number_l2121_212180

-- Define the numbers as real numbers
def a : ℝ := 9.12445
def b : ℝ := 9.124555555555555555555555555555555555555555555555555
def c : ℝ := 9.124545454545454545454545454545454545454545454545454
def d : ℝ := 9.124524524524524524524524524524524524524524524524524
def e : ℝ := 9.124512451245124512451245124512451245124512451245124

-- Theorem statement
theorem largest_number : b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l2121_212180


namespace NUMINAMATH_CALUDE_geometric_series_product_l2121_212157

theorem geometric_series_product (x : ℝ) : 
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/x)^n → x = 9 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_product_l2121_212157


namespace NUMINAMATH_CALUDE_expression_evaluation_l2121_212155

theorem expression_evaluation : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2121_212155


namespace NUMINAMATH_CALUDE_percentage_of_singles_is_70_percent_l2121_212158

def total_hits : ℕ := 50
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 10

def singles : ℕ := total_hits - (home_runs + triples + doubles)

theorem percentage_of_singles_is_70_percent :
  (singles : ℚ) / total_hits * 100 = 70 := by sorry

end NUMINAMATH_CALUDE_percentage_of_singles_is_70_percent_l2121_212158


namespace NUMINAMATH_CALUDE_mixed_fraction_decimal_calculation_l2121_212107

theorem mixed_fraction_decimal_calculation :
  let a : ℚ := 84 + 4 / 19
  let b : ℚ := 105 + 5 / 19
  let c : ℚ := 1.375
  let d : ℚ := 0.8
  a * c + b * d = 200 := by sorry

end NUMINAMATH_CALUDE_mixed_fraction_decimal_calculation_l2121_212107


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2121_212113

theorem ratio_x_to_y (x y : ℝ) (h : (8*x - 5*y) / (11*x - 3*y) = 2/7) : 
  x/y = 29/34 := by sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2121_212113


namespace NUMINAMATH_CALUDE_equation_represents_three_lines_l2121_212172

/-- The equation x²(x+y+1) = y²(x+y+1) represents three lines that do not all pass through a common point -/
theorem equation_represents_three_lines (x y : ℝ) : 
  (x^2 * (x + y + 1) = y^2 * (x + y + 1)) ↔ 
  ((y = -x) ∨ (y = x) ∨ (y = -x - 1)) ∧ 
  ¬(∃ p : ℝ × ℝ, (p.1 = p.2 ∧ p.2 = -p.1) ∧ 
                 (p.1 = -p.2 - 1 ∧ p.2 = p.1) ∧ 
                 (p.1 = p.2 ∧ p.2 = -p.1 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_three_lines_l2121_212172


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l2121_212184

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log (1/2)

def domain (x : ℝ) : Prop := x^2 - 2*x - 3 > 0

theorem f_increasing_on_interval :
  ∀ x y, x < y → x < -1 → y < -1 → domain x → domain y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l2121_212184


namespace NUMINAMATH_CALUDE_calculation_proof_l2121_212129

theorem calculation_proof : (120 : ℝ) / ((6 : ℝ) / 2) * 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2121_212129


namespace NUMINAMATH_CALUDE_inequalities_theorem_l2121_212137

theorem inequalities_theorem (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : 
  (b / a > c / a) ∧ ((b - a) / c > 0) ∧ ((a - c) / (a * c) < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l2121_212137


namespace NUMINAMATH_CALUDE_jungkook_has_larger_number_l2121_212173

theorem jungkook_has_larger_number (yoongi_number jungkook_number : ℕ) : 
  yoongi_number = 4 → jungkook_number = 6 * 3 → jungkook_number > yoongi_number := by
  sorry

end NUMINAMATH_CALUDE_jungkook_has_larger_number_l2121_212173


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l2121_212163

theorem unique_solution_cube_equation :
  ∃! (y : ℝ), y ≠ 0 ∧ (3 * y)^6 = (9 * y)^5 :=
by
  use 81
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l2121_212163


namespace NUMINAMATH_CALUDE_olympic_production_l2121_212159

/-- The number of sets of Olympic logo and mascots that can be produced -/
theorem olympic_production : ∃ (x y : ℕ), 
  4 * x + 5 * y = 20000 ∧ 
  3 * x + 10 * y = 30000 ∧ 
  x = 2000 ∧ 
  y = 2400 := by
  sorry

end NUMINAMATH_CALUDE_olympic_production_l2121_212159


namespace NUMINAMATH_CALUDE_trapezoid_area_l2121_212164

/-- Given an outer equilateral triangle with area 36, an inner equilateral triangle
    with area 4, and three congruent trapezoids between them, the area of one
    trapezoid is 32/3. -/
theorem trapezoid_area
  (outer_triangle_area : ℝ)
  (inner_triangle_area : ℝ)
  (num_trapezoids : ℕ)
  (h1 : outer_triangle_area = 36)
  (h2 : inner_triangle_area = 4)
  (h3 : num_trapezoids = 3) :
  (outer_triangle_area - inner_triangle_area) / num_trapezoids = 32 / 3 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2121_212164


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l2121_212138

theorem arctan_sum_special_case : 
  ∀ (a b : ℝ), 
    a = -1/3 → 
    (2*a + 1)*(2*b + 1) = 1 → 
    Real.arctan a + Real.arctan b = π/4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l2121_212138


namespace NUMINAMATH_CALUDE_dad_caught_more_trouts_l2121_212176

-- Define the number of trouts Caleb caught
def caleb_trouts : ℕ := 2

-- Define the number of trouts Caleb's dad caught
def dad_trouts : ℕ := 3 * caleb_trouts

-- Theorem to prove
theorem dad_caught_more_trouts : dad_trouts - caleb_trouts = 4 := by
  sorry

end NUMINAMATH_CALUDE_dad_caught_more_trouts_l2121_212176


namespace NUMINAMATH_CALUDE_dice_probability_l2121_212188

/-- The number of possible outcomes for a single die roll -/
def die_outcomes : ℕ := 6

/-- The number of favorable outcomes for a single die roll (not equal to 2) -/
def favorable_outcomes : ℕ := 5

/-- The probability that (a-2)(b-2)(c-2) ≠ 0 when three standard dice are tossed -/
theorem dice_probability : 
  (favorable_outcomes ^ 3 : ℚ) / (die_outcomes ^ 3 : ℚ) = 125 / 216 := by
sorry

end NUMINAMATH_CALUDE_dice_probability_l2121_212188


namespace NUMINAMATH_CALUDE_reporter_earnings_l2121_212103

/-- Reporter's earnings calculation --/
theorem reporter_earnings
  (words_per_minute : ℕ)
  (pay_per_word : ℚ)
  (pay_per_article : ℕ)
  (num_articles : ℕ)
  (hours_available : ℕ)
  (h1 : words_per_minute = 10)
  (h2 : pay_per_word = 1/10)
  (h3 : pay_per_article = 60)
  (h4 : num_articles = 3)
  (h5 : hours_available = 4)
  : (((hours_available * 60 * words_per_minute) * pay_per_word + num_articles * pay_per_article) / hours_available : ℚ) = 105 :=
by sorry

end NUMINAMATH_CALUDE_reporter_earnings_l2121_212103


namespace NUMINAMATH_CALUDE_good_numbers_l2121_212179

def isGoodNumber (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), ∀ k : Fin n, ∃ m : ℕ, k.val + 1 + a k = m^2

theorem good_numbers :
  isGoodNumber 13 ∧
  isGoodNumber 15 ∧
  isGoodNumber 17 ∧
  isGoodNumber 19 ∧
  ¬isGoodNumber 11 := by sorry

end NUMINAMATH_CALUDE_good_numbers_l2121_212179


namespace NUMINAMATH_CALUDE_quadratic_properties_l2121_212122

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 4

-- Theorem stating the properties of the function
theorem quadratic_properties :
  (∃ (max_value : ℝ), ∀ (x : ℝ), f x ≤ max_value ∧ max_value = 10.75) ∧
  (∃ (max_point : ℝ), max_point > 0 ∧ max_point = 1.5 ∧ ∀ (x : ℝ), f x ≤ f max_point) ∧
  (∀ (x y : ℝ), x > 1.5 → y > x → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2121_212122


namespace NUMINAMATH_CALUDE_smallest_a_for_96a_squared_equals_b_cubed_l2121_212165

theorem smallest_a_for_96a_squared_equals_b_cubed :
  ∀ a : ℕ+, a < 12 → ¬∃ b : ℕ+, 96 * a^2 = b^3 ∧ 
  ∃ b : ℕ+, 96 * 12^2 = b^3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_96a_squared_equals_b_cubed_l2121_212165


namespace NUMINAMATH_CALUDE_expression_equivalence_l2121_212191

theorem expression_equivalence (a b c : ℝ) : a - (2*b - 3*c) = a + (-2*b + 3*c) := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l2121_212191


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l2121_212185

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l2121_212185


namespace NUMINAMATH_CALUDE_eighth_term_matchsticks_l2121_212151

/-- The number of matchsticks in the nth term of the sequence -/
def matchsticks (n : ℕ) : ℕ := (n + 1) * 3

/-- Theorem: The number of matchsticks in the eighth term is 27 -/
theorem eighth_term_matchsticks : matchsticks 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_matchsticks_l2121_212151


namespace NUMINAMATH_CALUDE_parkway_fifth_grade_count_l2121_212160

/-- The number of students in the fifth grade at Parkway Elementary School -/
def total_students : ℕ := sorry

/-- The number of boys in the fifth grade -/
def boys : ℕ := 312

/-- The number of students playing soccer -/
def soccer_players : ℕ := 250

/-- The percentage of soccer players who are boys -/
def boys_soccer_percentage : ℚ := 82 / 100

/-- The number of girls not playing soccer -/
def girls_not_soccer : ℕ := 63

theorem parkway_fifth_grade_count :
  total_students = 420 :=
by sorry

end NUMINAMATH_CALUDE_parkway_fifth_grade_count_l2121_212160


namespace NUMINAMATH_CALUDE_gervais_mileage_proof_l2121_212116

/-- Gervais' average daily mileage --/
def gervais_average_mileage : ℝ := 315

/-- Number of days Gervais drove --/
def gervais_days : ℕ := 3

/-- Total miles Henri drove in a week --/
def henri_total_miles : ℝ := 1250

/-- Difference in miles between Henri and Gervais --/
def miles_difference : ℝ := 305

theorem gervais_mileage_proof :
  gervais_average_mileage * gervais_days = henri_total_miles - miles_difference :=
by sorry

end NUMINAMATH_CALUDE_gervais_mileage_proof_l2121_212116


namespace NUMINAMATH_CALUDE_infinitely_many_composite_f_increasing_l2121_212144

/-- The number of positive divisors of a natural number -/
def tau (a : ℕ) : ℕ := (Nat.divisors a).card

/-- The function f(n) as defined in the problem -/
def f (n : ℕ) : ℕ := tau (Nat.factorial n) - tau (Nat.factorial (n - 1))

/-- A composite number -/
def Composite (n : ℕ) : Prop := ¬Nat.Prime n ∧ n > 1

/-- The main theorem to be proved -/
theorem infinitely_many_composite_f_increasing :
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
  (∀ n ∈ S, Composite n ∧ 
    (∀ m : ℕ, m < n → f m < f n)) := by sorry

end NUMINAMATH_CALUDE_infinitely_many_composite_f_increasing_l2121_212144


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2121_212101

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 7) % 25 = 0 ∧ (n + 7) % 49 = 0 ∧ (n + 7) % 15 = 0 ∧ (n + 7) % 21 = 0

theorem smallest_number_divisible_by_all : 
  is_divisible_by_all 3668 ∧ ∀ m : ℕ, m < 3668 → ¬is_divisible_by_all m := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2121_212101


namespace NUMINAMATH_CALUDE_larger_number_proof_l2121_212167

theorem larger_number_proof (x y : ℝ) (h1 : 4 * y = 7 * x) (h2 : y - x = 12) : y = 28 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2121_212167


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2121_212187

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → 2^x > x^2) ↔ (∃ x : ℝ, x ≥ 0 ∧ 2^x ≤ x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2121_212187


namespace NUMINAMATH_CALUDE_book_cost_problem_l2121_212197

theorem book_cost_problem : ∃ (s b c : ℕ+), 
  s > 18 ∧ 
  b > 1 ∧ 
  c > b ∧ 
  s * b * c = 3203 ∧ 
  c = 11 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l2121_212197


namespace NUMINAMATH_CALUDE_system_has_solution_l2121_212100

/-- The system of equations has a solution for the given range of b -/
theorem system_has_solution (b : ℝ) 
  (h : b ∈ Set.Iic (-7/12) ∪ Set.Ioi 0) : 
  ∃ (a x y : ℝ), x = 7/b - |y + b| ∧ 
                 x^2 + y^2 + 96 = -a*(2*y + a) - 20*x := by
  sorry


end NUMINAMATH_CALUDE_system_has_solution_l2121_212100


namespace NUMINAMATH_CALUDE_girls_fraction_l2121_212112

theorem girls_fraction (T G B : ℚ) 
  (h1 : G > 0) 
  (h2 : T > 0) 
  (h3 : ∃ X : ℚ, X * G = (1/5) * T) 
  (h4 : B / G = 7/3) 
  (h5 : T = B + G) : 
  ∃ X : ℚ, X * G = (1/5) * T ∧ X = 2/3 := by
sorry

end NUMINAMATH_CALUDE_girls_fraction_l2121_212112


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l2121_212182

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 2)
  (hy : y / z = 5 / 3)
  (hz : z / x = 1 / 6) :
  w / y = 9 / 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l2121_212182


namespace NUMINAMATH_CALUDE_cherry_soda_count_l2121_212194

theorem cherry_soda_count (total : ℕ) (cherry : ℕ) (orange : ℕ) 
  (h1 : total = 24)
  (h2 : orange = 2 * cherry)
  (h3 : total = cherry + orange) : cherry = 8 := by
  sorry

end NUMINAMATH_CALUDE_cherry_soda_count_l2121_212194


namespace NUMINAMATH_CALUDE_coin_toss_probability_l2121_212123

theorem coin_toss_probability (p : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : p ^ 5 = 0.0625) :
  p = 0.5 := by
sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l2121_212123


namespace NUMINAMATH_CALUDE_money_distribution_l2121_212166

theorem money_distribution (total : ℝ) (a b c : ℝ) : 
  total = 1080 →
  a = (1/3) * (b + c) →
  b = (2/7) * (a + c) →
  a + b + c = total →
  a > b →
  a - b = 30 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l2121_212166


namespace NUMINAMATH_CALUDE_second_discount_percentage_l2121_212196

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 350 →
  first_discount = 20 →
  final_price = 266 →
  (original_price * (1 - first_discount / 100) * (1 - (original_price * (1 - first_discount / 100) - final_price) / (original_price * (1 - first_discount / 100))) = final_price) →
  (original_price * (1 - first_discount / 100) - final_price) / (original_price * (1 - first_discount / 100)) * 100 = 5 :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l2121_212196


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l2121_212121

/-- Represents a die with 6 faces -/
structure Die :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7)

/-- Represents a 4x4x4 cube made of dice -/
def GiantCube := Fin 4 → Fin 4 → Fin 4 → Die

/-- The sum of visible values on the 6 faces of the giant cube -/
def visible_sum (cube : GiantCube) : ℕ :=
  sorry

/-- The theorem stating the smallest possible sum of visible values -/
theorem smallest_visible_sum (cube : GiantCube) :
  visible_sum cube ≥ 144 :=
sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l2121_212121


namespace NUMINAMATH_CALUDE_sqrt_of_three_minus_negative_one_equals_two_l2121_212126

theorem sqrt_of_three_minus_negative_one_equals_two :
  Real.sqrt (3 - (-1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_three_minus_negative_one_equals_two_l2121_212126


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_zero_l2121_212175

theorem sum_mod_thirteen_zero : (9023 + 9024 + 9025 + 9026) % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_zero_l2121_212175


namespace NUMINAMATH_CALUDE_supplement_of_forty_degrees_l2121_212142

/-- Given a system of parallel lines where an angle of 40° is formed, 
    prove that its supplement measures 140°. -/
theorem supplement_of_forty_degrees (α : Real) (h1 : α = 40) : 180 - α = 140 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_forty_degrees_l2121_212142


namespace NUMINAMATH_CALUDE_problem_statement_l2121_212131

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2121_212131


namespace NUMINAMATH_CALUDE_square_floor_area_l2121_212146

theorem square_floor_area (rug_length : ℝ) (rug_width : ℝ) (uncovered_fraction : ℝ) :
  rug_length = 2 →
  rug_width = 7 →
  uncovered_fraction = 0.78125 →
  ∃ (floor_side : ℝ),
    floor_side ^ 2 = 64 ∧
    rug_length * rug_width = (1 - uncovered_fraction) * floor_side ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_square_floor_area_l2121_212146


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2121_212168

def f (x : ℝ) : ℝ := x^4 - 3*x^3 + 10*x^2 - 16*x + 5

def g (x k : ℝ) : ℝ := x^2 - x + k

theorem polynomial_division_remainder (k a : ℝ) :
  (∃ q : ℝ → ℝ, ∀ x, f x = g x k * q x + (2*x + a)) ↔ k = 8.5 ∧ a = 9.25 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2121_212168


namespace NUMINAMATH_CALUDE_unique_rectangle_l2121_212117

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area_eq : length * width = 100
  perimeter_eq : length + width = 24

/-- Two rectangles are considered distinct if they are not congruent -/
def distinct (r1 r2 : Rectangle) : Prop :=
  (r1.length ≠ r2.length ∧ r1.length ≠ r2.width) ∨
  (r1.width ≠ r2.length ∧ r1.width ≠ r2.width)

/-- There is exactly one distinct rectangle with area 100 and perimeter 24 -/
theorem unique_rectangle : ∃! r : Rectangle, ∀ s : Rectangle, ¬(distinct r s) :=
  sorry

end NUMINAMATH_CALUDE_unique_rectangle_l2121_212117


namespace NUMINAMATH_CALUDE_base8_divisibility_by_13_l2121_212147

/-- Converts a base-8 number of the form 3dd7₈ to base 10 --/
def base8_to_base10 (d : ℕ) : ℕ := 3 * 512 + d * 64 + d * 8 + 7

/-- Checks if a natural number is divisible by 13 --/
def divisible_by_13 (n : ℕ) : Prop := ∃ k : ℕ, n = 13 * k

/-- A base-8 digit is between 0 and 7 inclusive --/
def is_base8_digit (d : ℕ) : Prop := 0 ≤ d ∧ d ≤ 7

theorem base8_divisibility_by_13 (d : ℕ) (h : is_base8_digit d) : 
  divisible_by_13 (base8_to_base10 d) ↔ (d = 1 ∨ d = 2) :=
sorry

end NUMINAMATH_CALUDE_base8_divisibility_by_13_l2121_212147


namespace NUMINAMATH_CALUDE_min_product_positive_reals_l2121_212140

theorem min_product_positive_reals (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  x ≤ 2 * (y + z) →
  y ≤ 2 * (x + z) →
  z ≤ 2 * (x + y) →
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 1 → 
    a ≤ 2 * (b + c) → b ≤ 2 * (a + c) → c ≤ 2 * (a + b) →
    x * y * z ≤ a * b * c →
  x * y * z = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_min_product_positive_reals_l2121_212140


namespace NUMINAMATH_CALUDE_quadratic_solution_values_second_quadratic_solution_set_l2121_212177

-- Definition for the quadratic inequality
def quadratic_inequality (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 + 3 * x - 2 > 0

-- Definition for the solution set
def solution_set (n : ℝ) (x : ℝ) : Prop :=
  n < x ∧ x < 2

-- Theorem for the first part of the problem
theorem quadratic_solution_values :
  (∀ x, quadratic_inequality m x ↔ solution_set n x) →
  m = -1 ∧ n = 1 :=
sorry

-- Definition for the second quadratic inequality
def second_quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  x^2 + (a - 1) * x - a > 0

-- Theorem for the second part of the problem
theorem second_quadratic_solution_set (a : ℝ) :
  (a < -1 → ∀ x, second_quadratic_inequality a x ↔ (x > 1 ∨ x < -a)) ∧
  (a = -1 → ∀ x, second_quadratic_inequality a x ↔ x ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_values_second_quadratic_solution_set_l2121_212177


namespace NUMINAMATH_CALUDE_positive_real_properties_l2121_212150

theorem positive_real_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 - b^2 = 1 → a - b < 1) ∧
  (a > b + 1 → a^2 > b^2 + 1) := by
sorry

end NUMINAMATH_CALUDE_positive_real_properties_l2121_212150


namespace NUMINAMATH_CALUDE_two_roots_implies_a_greater_than_e_l2121_212156

-- Define the function f(x) = x / ln(x)
noncomputable def f (x : ℝ) : ℝ := x / Real.log x

-- State the theorem
theorem two_roots_implies_a_greater_than_e (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * Real.log x = x ∧ a * Real.log y = y) → a > Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_two_roots_implies_a_greater_than_e_l2121_212156


namespace NUMINAMATH_CALUDE_literary_readers_count_l2121_212108

theorem literary_readers_count (total : ℕ) (science_fiction : ℕ) (both : ℕ) 
  (h1 : total = 650) 
  (h2 : science_fiction = 250) 
  (h3 : both = 150) : 
  total = science_fiction + (550 : ℕ) - both :=
by sorry

end NUMINAMATH_CALUDE_literary_readers_count_l2121_212108


namespace NUMINAMATH_CALUDE_rectangle_area_21_implies_y_7_l2121_212153

/-- Represents a rectangle EFGH with vertices E(0, 0), F(0, 3), G(y, 3), and H(y, 0) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := 3 * r.y

theorem rectangle_area_21_implies_y_7 (r : Rectangle) (h : area r = 21) : r.y = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_21_implies_y_7_l2121_212153


namespace NUMINAMATH_CALUDE_tea_milk_problem_l2121_212189

/-- Represents the amount of liquid in a mug -/
structure Mug where
  tea : ℚ
  milk : ℚ

/-- Calculates the fraction of milk in a mug -/
def milkFraction (m : Mug) : ℚ :=
  m.milk / (m.tea + m.milk)

theorem tea_milk_problem : 
  let mug1_initial := Mug.mk 5 0
  let mug2_initial := Mug.mk 0 3
  let mug1_after_first_transfer := Mug.mk (mug1_initial.tea - 2) 0
  let mug2_after_first_transfer := Mug.mk 2 3
  let tea_fraction_in_mug2 := mug2_after_first_transfer.tea / 
    (mug2_after_first_transfer.tea + mug2_after_first_transfer.milk)
  let milk_fraction_in_mug2 := mug2_after_first_transfer.milk / 
    (mug2_after_first_transfer.tea + mug2_after_first_transfer.milk)
  let tea_returned := 3 * tea_fraction_in_mug2
  let milk_returned := 3 * milk_fraction_in_mug2
  let mug1_final := Mug.mk (mug1_after_first_transfer.tea + tea_returned) milk_returned
  milkFraction mug1_final = 3/10 := by
sorry

end NUMINAMATH_CALUDE_tea_milk_problem_l2121_212189


namespace NUMINAMATH_CALUDE_calculate_otimes_expression_l2121_212161

-- Define the ⊗ operation
def otimes (a b : ℚ) : ℚ := (a + b) / (a - b)

-- The main theorem to prove
theorem calculate_otimes_expression :
  (otimes (otimes 8 6) (otimes 2 1)) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_otimes_expression_l2121_212161


namespace NUMINAMATH_CALUDE_least_k_divisible_by_480_l2121_212169

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem least_k_divisible_by_480 :
  ∃ k : ℕ+, (k : ℕ) = 101250 ∧
    is_divisible (k^4) 480 ∧
    ∀ m : ℕ+, m < k → ¬is_divisible (m^4) 480 := by
  sorry

end NUMINAMATH_CALUDE_least_k_divisible_by_480_l2121_212169


namespace NUMINAMATH_CALUDE_company_women_workers_l2121_212130

theorem company_women_workers 
  (total_workers : ℕ) 
  (h1 : total_workers / 3 = total_workers - total_workers * 2 / 3) -- A third of workers do not have a retirement plan
  (h2 : (total_workers / 3) / 2 = total_workers / 6) -- 50% of workers without a retirement plan are women
  (h3 : (total_workers * 2 / 3) * 2 / 5 = total_workers * 8 / 30) -- 40% of workers with a retirement plan are men
  (h4 : total_workers * 8 / 30 = 120) -- 120 workers are men
  : total_workers - 120 = 330 := by
sorry

end NUMINAMATH_CALUDE_company_women_workers_l2121_212130


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2121_212193

/-- An arithmetic sequence starting with 2 and ending with 2006 has 502 terms. -/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ → ℕ), 
    a 0 = 2 → 
    (∃ n : ℕ, a n = 2006) → 
    (∀ i j : ℕ, a (i + 1) - a i = a (j + 1) - a j) → 
    (∃ n : ℕ, a n = 2006 ∧ n + 1 = 502) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2121_212193
