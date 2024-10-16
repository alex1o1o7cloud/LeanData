import Mathlib

namespace NUMINAMATH_CALUDE_mary_has_ten_more_than_marco_l2361_236163

/-- Calculates the difference in money between Mary and Marco after transactions. -/
def moneyDifference (marco_initial : ℕ) (mary_initial : ℕ) (mary_spent : ℕ) : ℕ :=
  let marco_gives := marco_initial / 2
  let marco_final := marco_initial - marco_gives
  let mary_final := mary_initial + marco_gives - mary_spent
  mary_final - marco_final

/-- Proves that Mary has $10 more than Marco after the described transactions. -/
theorem mary_has_ten_more_than_marco :
  moneyDifference 24 15 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_ten_more_than_marco_l2361_236163


namespace NUMINAMATH_CALUDE_prime_square_mod_twelve_l2361_236129

theorem prime_square_mod_twelve (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) :
  p^2 % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_twelve_l2361_236129


namespace NUMINAMATH_CALUDE_specimen_expiration_time_l2361_236114

def seconds_in_day : ℕ := 24 * 60 * 60

def expiration_time (submission_time : Nat) (expiration_seconds : Nat) : Nat :=
  (submission_time + expiration_seconds) % seconds_in_day

theorem specimen_expiration_time :
  let submission_time : Nat := 15 * 60 * 60  -- 3 PM in seconds
  let expiration_seconds : Nat := 7 * 6 * 5 * 4 * 3 * 2 * 1  -- 7!
  expiration_time submission_time expiration_seconds = 16 * 60 * 60 + 24 * 60  -- 4:24 PM in seconds
  := by sorry

end NUMINAMATH_CALUDE_specimen_expiration_time_l2361_236114


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l2361_236149

/-- Parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- Point of intersection between a line and a parabola -/
def intersection (p : Parabola) (l : Line) : ℝ × ℝ := sorry

/-- Foot of the perpendicular from a point to a line -/
def perpendicularFoot (point : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem parabola_triangle_area 
  (p : Parabola) 
  (l : Line) 
  (h1 : p.equation = fun x y => y^2 = 4*x)
  (h2 : p.focus = (1, 0))
  (h3 : p.directrix = fun x y => x = -1)
  (h4 : l.point = (1, 0))
  (h5 : l.slope = Real.sqrt 3)
  (h6 : (intersection p l).2 > 0) :
  let A := intersection p l
  let K := perpendicularFoot A p.directrix
  triangleArea A K p.focus = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l2361_236149


namespace NUMINAMATH_CALUDE_prime_sum_of_squares_l2361_236133

theorem prime_sum_of_squares (p : ℕ) (hp : Nat.Prime p) :
  (∃ k : ℕ, p = 4 * k + 1) → (∃ a b : ℤ, p = a^2 + b^2) ∧
  (∃ k : ℕ, p = 8 * k + 3) → (∃ a b c : ℤ, p = a^2 + b^2 + c^2) :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_of_squares_l2361_236133


namespace NUMINAMATH_CALUDE_stephanie_remaining_payment_l2361_236131

/-- Represents the bills and payments in Stephanie's household budget --/
structure BudgetInfo where
  electricity_bill : ℝ
  gas_bill : ℝ
  water_bill : ℝ
  internet_bill : ℝ
  gas_initial_payment_fraction : ℝ
  gas_additional_payment : ℝ
  water_payment_fraction : ℝ
  internet_payment_count : ℕ
  internet_payment_amount : ℝ

/-- Calculates the remaining amount to pay given the budget information --/
def remaining_payment (budget : BudgetInfo) : ℝ :=
  let total_bills := budget.electricity_bill + budget.gas_bill + budget.water_bill + budget.internet_bill
  let total_paid := budget.electricity_bill +
                    (budget.gas_bill * budget.gas_initial_payment_fraction + budget.gas_additional_payment) +
                    (budget.water_bill * budget.water_payment_fraction) +
                    (budget.internet_payment_count : ℝ) * budget.internet_payment_amount
  total_bills - total_paid

/-- Theorem stating that the remaining payment for Stephanie's bills is $30 --/
theorem stephanie_remaining_payment :
  let budget : BudgetInfo := {
    electricity_bill := 60,
    gas_bill := 40,
    water_bill := 40,
    internet_bill := 25,
    gas_initial_payment_fraction := 0.75,
    gas_additional_payment := 5,
    water_payment_fraction := 0.5,
    internet_payment_count := 4,
    internet_payment_amount := 5
  }
  remaining_payment budget = 30 := by sorry

end NUMINAMATH_CALUDE_stephanie_remaining_payment_l2361_236131


namespace NUMINAMATH_CALUDE_repeated_two_digit_divisible_by_101_l2361_236186

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n < 100 }

/-- Constructs a four-digit number by repeating a two-digit number -/
def repeat_two_digit (n : TwoDigitNumber) : ℕ :=
  100 * n.val + n.val

theorem repeated_two_digit_divisible_by_101 (n : TwoDigitNumber) :
  (repeat_two_digit n) % 101 = 0 := by
  sorry

end NUMINAMATH_CALUDE_repeated_two_digit_divisible_by_101_l2361_236186


namespace NUMINAMATH_CALUDE_tan_beta_value_l2361_236199

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := by sorry

end NUMINAMATH_CALUDE_tan_beta_value_l2361_236199


namespace NUMINAMATH_CALUDE_min_omega_value_l2361_236178

theorem min_omega_value (f : ℝ → ℝ) (ω φ T : ℝ) :
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < π →
  (∀ t > 0, f (t + T) = f t) →
  (∀ t > T, ∃ s ∈ Set.Ioo 0 T, f t = f s) →
  f T = Real.sqrt 3 / 2 →
  f (π / 9) = 0 →
  3 ≤ ω ∧ ∀ ω' > 0, (∀ x, Real.cos (ω' * x + φ) = f x) → ω ≤ ω' :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l2361_236178


namespace NUMINAMATH_CALUDE_power_equality_l2361_236173

theorem power_equality (p : ℕ) : 16^5 = 4^p → p = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2361_236173


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2361_236141

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 < 3}
def B : Set ℝ := {x | 2*x - 3 < 3*x - 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2361_236141


namespace NUMINAMATH_CALUDE_ice_skating_falls_ratio_l2361_236143

/-- Given the number of falls for Steven, Stephanie, and Sonya while ice skating,
    prove that the ratio of Sonya's falls to half of Stephanie's falls is 3:4. -/
theorem ice_skating_falls_ratio 
  (steven_falls : ℕ) 
  (stephanie_falls : ℕ) 
  (sonya_falls : ℕ) 
  (h1 : steven_falls = 3)
  (h2 : stephanie_falls = steven_falls + 13)
  (h3 : sonya_falls = 6) :
  (sonya_falls : ℚ) / ((stephanie_falls : ℚ) / 2) = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_ice_skating_falls_ratio_l2361_236143


namespace NUMINAMATH_CALUDE_system_solution_l2361_236183

theorem system_solution (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y z : ℝ),
    (x + a*y + a^2*z + a^3 = 0) ∧
    (x + b*y + b^2*z + b^3 = 0) ∧
    (x + c*y + c^2*z + c^3 = 0) ∧
    (x = -a*b*c) ∧
    (y = a*b + b*c + c*a) ∧
    (z = -(a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2361_236183


namespace NUMINAMATH_CALUDE_solve_cab_driver_problem_l2361_236105

def cab_driver_problem (day1 day2 day4 day5 average : ℕ) : Prop :=
  let total := 5 * average
  let known_sum := day1 + day2 + day4 + day5
  let day3 := total - known_sum
  (day1 = 300) ∧ (day2 = 150) ∧ (day4 = 400) ∧ (day5 = 500) ∧ (average = 420) → day3 = 750

theorem solve_cab_driver_problem :
  cab_driver_problem 300 150 400 500 420 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_cab_driver_problem_l2361_236105


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2361_236164

theorem quadratic_minimum : 
  (∃ (x : ℝ), x^2 + 12*x + 9 = -27) ∧ 
  (∀ (x : ℝ), x^2 + 12*x + 9 ≥ -27) := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2361_236164


namespace NUMINAMATH_CALUDE_total_bones_equals_twelve_l2361_236188

/-- The number of bones carried by each dog in a pack of 5 dogs. -/
def DogBones : Fin 5 → ℕ
  | 0 => 3  -- First dog
  | 1 => DogBones 0 - 1  -- Second dog
  | 2 => 2 * DogBones 1  -- Third dog
  | 3 => 1  -- Fourth dog
  | 4 => 2 * DogBones 3  -- Fifth dog

/-- The theorem states that the sum of bones carried by all 5 dogs equals 12. -/
theorem total_bones_equals_twelve :
  (Finset.sum Finset.univ DogBones) = 12 := by
  sorry


end NUMINAMATH_CALUDE_total_bones_equals_twelve_l2361_236188


namespace NUMINAMATH_CALUDE_same_figure_l2361_236194

noncomputable section

open Complex

/-- Two equations describe the same figure in the complex plane -/
theorem same_figure (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  {z : ℂ | abs (z + n * I) + abs (z - m * I) = n} =
  {z : ℂ | abs (z + n * I) - abs (z - m * I) = -m} :=
sorry

end

end NUMINAMATH_CALUDE_same_figure_l2361_236194


namespace NUMINAMATH_CALUDE_hurdle_race_calculations_l2361_236145

/-- Calculates the distance between adjacent hurdles and the theoretical best time for a 110m hurdle race --/
theorem hurdle_race_calculations 
  (total_distance : ℝ) 
  (num_hurdles : ℕ) 
  (start_to_first : ℝ) 
  (last_to_finish : ℝ) 
  (time_to_first : ℝ) 
  (time_after_last : ℝ) 
  (fastest_cycle : ℝ) 
  (h1 : total_distance = 110) 
  (h2 : num_hurdles = 10) 
  (h3 : start_to_first = 13.72) 
  (h4 : last_to_finish = 14.02) 
  (h5 : time_to_first = 2.5) 
  (h6 : time_after_last = 1.4) 
  (h7 : fastest_cycle = 0.96) :
  let inter_hurdle_distance := (total_distance - start_to_first - last_to_finish) / num_hurdles
  let theoretical_best_time := time_to_first + (num_hurdles : ℝ) * fastest_cycle + time_after_last
  inter_hurdle_distance = 8.28 ∧ theoretical_best_time = 12.1 := by
  sorry


end NUMINAMATH_CALUDE_hurdle_race_calculations_l2361_236145


namespace NUMINAMATH_CALUDE_correct_equation_l2361_236184

theorem correct_equation : 4 - 4 / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l2361_236184


namespace NUMINAMATH_CALUDE_shape_area_is_94_l2361_236168

/-- A shape composed of three rectangles with given dimensions -/
structure Shape where
  rect1_width : ℕ
  rect1_height : ℕ
  rect2_width : ℕ
  rect2_height : ℕ
  rect3_width : ℕ
  rect3_height : ℕ

/-- Calculate the area of a rectangle -/
def rectangle_area (width height : ℕ) : ℕ := width * height

/-- Calculate the total area of the shape -/
def total_area (s : Shape) : ℕ :=
  rectangle_area s.rect1_width s.rect1_height +
  rectangle_area s.rect2_width s.rect2_height +
  rectangle_area s.rect3_width s.rect3_height

/-- The shape described in the problem -/
def problem_shape : Shape :=
  { rect1_width := 7
  , rect1_height := 7
  , rect2_width := 3
  , rect2_height := 5
  , rect3_width := 5
  , rect3_height := 6 }

theorem shape_area_is_94 : total_area problem_shape = 94 := by
  sorry


end NUMINAMATH_CALUDE_shape_area_is_94_l2361_236168


namespace NUMINAMATH_CALUDE_exponential_function_property_l2361_236147

theorem exponential_function_property (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  ∀ x y : ℝ, (fun x => a^x) (x + y) = (fun x => a^x) x * (fun x => a^x) y :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l2361_236147


namespace NUMINAMATH_CALUDE_loan_sum_proof_l2361_236104

theorem loan_sum_proof (x y : ℝ) : 
  x * (3 / 100) * 5 = y * (5 / 100) * 3 →
  y = 1332.5 →
  x + y = 2665 := by
sorry

end NUMINAMATH_CALUDE_loan_sum_proof_l2361_236104


namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_one_l2361_236144

/-- Given a polynomial g(x) = 3x^4 + 2x^3 - x^2 - 4x + s,
    prove that if g(-1) = 0, then s = -4 -/
theorem polynomial_value_at_negative_one (s : ℝ) :
  let g : ℝ → ℝ := λ x ↦ 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s
  g (-1) = 0 → s = -4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_negative_one_l2361_236144


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2361_236116

theorem arithmetic_sequence_sum (a₁ a₂ a₃ a₆ : ℕ) (h₁ : a₁ = 5) (h₂ : a₂ = 12) (h₃ : a₃ = 19) (h₆ : a₆ = 40) :
  let d := a₂ - a₁
  let a₄ := a₃ + d
  let a₅ := a₄ + d
  a₄ + a₅ = 59 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2361_236116


namespace NUMINAMATH_CALUDE_razorback_tshirt_shop_profit_l2361_236154

/-- Calculate the net profit for the Razorback T-shirt Shop on game day -/
theorem razorback_tshirt_shop_profit :
  let regular_price : ℚ := 15
  let production_cost : ℚ := 4
  let first_event_quantity : ℕ := 150
  let second_event_quantity : ℕ := 175
  let first_event_discount : ℚ := 0.1
  let second_event_discount : ℚ := 0.15
  let overhead_expense : ℚ := 200
  let sales_tax_rate : ℚ := 0.05

  let first_event_revenue := (regular_price * (1 - first_event_discount)) * first_event_quantity
  let second_event_revenue := (regular_price * (1 - second_event_discount)) * second_event_quantity
  let total_revenue := first_event_revenue + second_event_revenue
  let total_quantity := first_event_quantity + second_event_quantity
  let total_production_cost := production_cost * total_quantity
  let sales_tax := sales_tax_rate * total_revenue
  let total_expenses := total_production_cost + overhead_expense + sales_tax
  let net_profit := total_revenue - total_expenses

  net_profit = 2543.4375 := by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_shop_profit_l2361_236154


namespace NUMINAMATH_CALUDE_square_root_problem_l2361_236134

theorem square_root_problem (m a b c n : ℝ) (hm : m > 0) :
  (Real.sqrt m = 2*n + 1 ∧ Real.sqrt m = 4 - 3*n) →
  (|a - 1| + Real.sqrt b + (c - n)^2 = 0) →
  (m = 121 ∨ m = 121/25) ∧ Real.sqrt (a + b + c) = Real.sqrt 6 ∨ Real.sqrt (a + b + c) = -Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l2361_236134


namespace NUMINAMATH_CALUDE_distance_from_two_is_six_l2361_236162

theorem distance_from_two_is_six (x : ℝ) : |x - 2| = 6 → x = 8 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_two_is_six_l2361_236162


namespace NUMINAMATH_CALUDE_a_plus_b_value_l2361_236152

theorem a_plus_b_value (a b : ℝ) (h1 : a^2 = 4) (h2 : b^2 = 9) (h3 : a * b < 0) :
  a + b = 1 ∨ a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l2361_236152


namespace NUMINAMATH_CALUDE_m_range_l2361_236159

theorem m_range : ∀ m : ℝ, m = 3 * Real.sqrt 2 - 1 → 3 < m ∧ m < 4 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2361_236159


namespace NUMINAMATH_CALUDE_time_to_finish_book_l2361_236191

/-- Calculates the time needed to finish reading a book given the specified conditions -/
theorem time_to_finish_book 
  (total_chapters : ℕ) 
  (chapters_read : ℕ) 
  (time_for_read_chapters : ℝ) 
  (break_time : ℝ) 
  (h1 : total_chapters = 14) 
  (h2 : chapters_read = 4) 
  (h3 : time_for_read_chapters = 6) 
  (h4 : break_time = 1/6) : 
  let remaining_chapters := total_chapters - chapters_read
  let time_per_chapter := time_for_read_chapters / chapters_read
  let reading_time := time_per_chapter * remaining_chapters
  let total_breaks := remaining_chapters - 1
  let total_break_time := total_breaks * break_time
  reading_time + total_break_time = 33/2 := by
sorry

end NUMINAMATH_CALUDE_time_to_finish_book_l2361_236191


namespace NUMINAMATH_CALUDE_ab_plus_cd_value_l2361_236158

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 8)
  (eq3 : a + c + d = 20)
  (eq4 : b + c + d = 15) :
  a * b + c * d = 84 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_value_l2361_236158


namespace NUMINAMATH_CALUDE_average_fuel_efficiency_l2361_236171

/-- Calculate the average fuel efficiency for a round trip with two different vehicles -/
theorem average_fuel_efficiency
  (total_distance : ℝ)
  (distance_first_leg : ℝ)
  (efficiency_first_vehicle : ℝ)
  (efficiency_second_vehicle : ℝ)
  (h1 : total_distance = 300)
  (h2 : distance_first_leg = total_distance / 2)
  (h3 : efficiency_first_vehicle = 50)
  (h4 : efficiency_second_vehicle = 25) :
  (total_distance) / ((distance_first_leg / efficiency_first_vehicle) + 
  (distance_first_leg / efficiency_second_vehicle)) = 33 := by
  sorry

#check average_fuel_efficiency

end NUMINAMATH_CALUDE_average_fuel_efficiency_l2361_236171


namespace NUMINAMATH_CALUDE_fishing_competition_duration_l2361_236128

theorem fishing_competition_duration 
  (jackson_daily : ℕ) 
  (jonah_daily : ℕ) 
  (george_daily : ℕ) 
  (total_catch : ℕ) 
  (h1 : jackson_daily = 6)
  (h2 : jonah_daily = 4)
  (h3 : george_daily = 8)
  (h4 : total_catch = 90) :
  ∃ (days : ℕ), days * (jackson_daily + jonah_daily + george_daily) = total_catch ∧ days = 5 := by
  sorry

end NUMINAMATH_CALUDE_fishing_competition_duration_l2361_236128


namespace NUMINAMATH_CALUDE_basic_astrophysics_degrees_l2361_236139

def microphotonics : ℝ := 13
def home_electronics : ℝ := 24
def food_additives : ℝ := 15
def genetically_modified_microorganisms : ℝ := 29
def industrial_lubricants : ℝ := 8
def total_circle_degrees : ℝ := 360

def other_sectors_sum : ℝ := 
  microphotonics + home_electronics + food_additives + 
  genetically_modified_microorganisms + industrial_lubricants

def basic_astrophysics_percentage : ℝ := 100 - other_sectors_sum

theorem basic_astrophysics_degrees : 
  (basic_astrophysics_percentage / 100) * total_circle_degrees = 39.6 := by
  sorry

end NUMINAMATH_CALUDE_basic_astrophysics_degrees_l2361_236139


namespace NUMINAMATH_CALUDE_stephanie_oranges_l2361_236120

theorem stephanie_oranges (store_visits : ℕ) (oranges_per_visit : ℕ) : 
  store_visits = 8 → oranges_per_visit = 2 → store_visits * oranges_per_visit = 16 := by
sorry

end NUMINAMATH_CALUDE_stephanie_oranges_l2361_236120


namespace NUMINAMATH_CALUDE_total_sheets_required_l2361_236146

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of times each letter needs to be written -/
def writing_times : ℕ := 3

/-- The number of sheets needed for one writing of a letter -/
def sheets_per_writing : ℕ := 1

/-- Theorem: The total number of sheets required to write each letter of the English alphabet
    three times (uppercase, lowercase, and cursive script) is 78. -/
theorem total_sheets_required :
  alphabet_size * writing_times * sheets_per_writing = 78 := by sorry

end NUMINAMATH_CALUDE_total_sheets_required_l2361_236146


namespace NUMINAMATH_CALUDE_square_to_rectangle_ratio_l2361_236180

/-- The number of rectangles formed by 10 horizontal and 10 vertical lines on a 9x9 chessboard -/
def num_rectangles : ℕ := 2025

/-- The number of squares formed by 10 horizontal and 10 vertical lines on a 9x9 chessboard -/
def num_squares : ℕ := 285

/-- The ratio of squares to rectangles on a 9x9 chessboard with 10 horizontal and 10 vertical lines -/
theorem square_to_rectangle_ratio : 
  (num_squares : ℚ) / num_rectangles = 19 / 135 := by sorry

end NUMINAMATH_CALUDE_square_to_rectangle_ratio_l2361_236180


namespace NUMINAMATH_CALUDE_pipe_speed_ratio_l2361_236177

-- Define the rates of pipes A, B, and C
def rate_A : ℚ := 1 / 28
def rate_B : ℚ := 1 / 14
def rate_C : ℚ := 1 / 7

-- Theorem statement
theorem pipe_speed_ratio :
  -- Given conditions
  (rate_A + rate_B + rate_C = 1 / 4) →  -- All pipes fill the tank in 4 hours
  (rate_C = 2 * rate_B) →               -- Pipe C is twice as fast as B
  (rate_A = 1 / 28) →                   -- Pipe A alone takes 28 hours
  -- Conclusion
  (rate_B / rate_A = 2) :=
by sorry


end NUMINAMATH_CALUDE_pipe_speed_ratio_l2361_236177


namespace NUMINAMATH_CALUDE_double_inequality_solution_l2361_236108

theorem double_inequality_solution (x : ℝ) : 
  (0 < (x^2 - 8*x + 13) / (x^2 - 4*x + 7) ∧ (x^2 - 8*x + 13) / (x^2 - 4*x + 7) < 2) ↔ 
  (4 - Real.sqrt 17 < x ∧ x < 4 - Real.sqrt 3) ∨ (4 + Real.sqrt 3 < x ∧ x < 4 + Real.sqrt 17) :=
sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l2361_236108


namespace NUMINAMATH_CALUDE_line_properties_l2361_236153

-- Define the line equation
def line_equation (a x y : ℝ) : Prop :=
  (a - 1) * x + y - a - 5 = 0

-- Define the fixed point
def fixed_point (A : ℝ × ℝ) : Prop :=
  ∀ a : ℝ, line_equation a A.1 A.2

-- Define the condition for not passing through the second quadrant
def not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation a x y → (x ≤ 0 ∧ y > 0 → False)

-- Theorem statement
theorem line_properties :
  (∃ A : ℝ × ℝ, fixed_point A ∧ A = (1, 6)) ∧
  (∀ a : ℝ, not_in_second_quadrant a ↔ a ≤ -5) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l2361_236153


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_concentric_circles_l2361_236137

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a circle with center and radius
structure Circle where
  center : Point2D
  radius : ℝ

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point2D
  b : Point2D
  c : Point2D

-- Function to check if a point lies on a circle
def pointOnCircle (p : Point2D) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Theorem statement
theorem equilateral_triangle_on_concentric_circles 
  (center : Point2D) (r₁ r₂ r₃ : ℝ) 
  (h₁ : 0 < r₁) (h₂ : r₁ < r₂) (h₃ : r₂ < r₃) :
  ∃ (t : EquilateralTriangle),
    pointOnCircle t.a (Circle.mk center r₂) ∧
    pointOnCircle t.b (Circle.mk center r₁) ∧
    pointOnCircle t.c (Circle.mk center r₃) :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_concentric_circles_l2361_236137


namespace NUMINAMATH_CALUDE_cats_remaining_l2361_236165

/-- The number of cats remaining after a sale in a pet store -/
theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 19 → house = 45 → sold = 56 → siamese + house - sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_l2361_236165


namespace NUMINAMATH_CALUDE_seven_lines_regions_l2361_236136

/-- The number of regions formed by n lines in a plane, where no two are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1)) / 2

/-- The property that no two lines are parallel and no three are concurrent -/
def general_position (n : ℕ) : Prop := n > 0

theorem seven_lines_regions :
  general_position 7 → num_regions 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_seven_lines_regions_l2361_236136


namespace NUMINAMATH_CALUDE_binomial_variance_4_half_l2361_236167

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ :=
  ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: The variance of a binomial distribution B(4, 1/2) is 1 -/
theorem binomial_variance_4_half :
  ∀ ξ : BinomialDistribution, ξ.n = 4 ∧ ξ.p = 1/2 → variance ξ = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_4_half_l2361_236167


namespace NUMINAMATH_CALUDE_juniper_bones_l2361_236160

theorem juniper_bones (b : ℕ) : 2 * b - 2 = (b + b) - 2 := by sorry

end NUMINAMATH_CALUDE_juniper_bones_l2361_236160


namespace NUMINAMATH_CALUDE_total_savings_is_150_l2361_236196

/-- Calculates the total savings for the year based on the given savings pattern. -/
def total_savings (savings_jan_to_jul : ℕ) (savings_aug_to_nov : ℕ) (savings_dec : ℕ) : ℕ :=
  7 * savings_jan_to_jul + 4 * savings_aug_to_nov + savings_dec

/-- Proves that the total savings for the year is $150 given the specified savings pattern. -/
theorem total_savings_is_150 :
  total_savings 10 15 20 = 150 := by sorry

end NUMINAMATH_CALUDE_total_savings_is_150_l2361_236196


namespace NUMINAMATH_CALUDE_circle_symmetry_l2361_236121

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - Real.sqrt 3)^2 = 4

-- Theorem statement
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    original_circle x₁ y₁ →
    symmetric_circle x₂ y₂ →
    ∃ (x_mid y_mid : ℝ),
      symmetry_line x_mid y_mid ∧
      x_mid = (x₁ + x₂) / 2 ∧
      y_mid = (y₁ + y₂) / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2361_236121


namespace NUMINAMATH_CALUDE_pepperoni_coverage_fraction_l2361_236189

-- Define the pizza and pepperoni characteristics
def pizza_diameter : ℝ := 16
def pepperoni_count : ℕ := 32
def pepperoni_across_diameter : ℕ := 8
def pepperoni_overlap_fraction : ℝ := 0.25

-- Theorem statement
theorem pepperoni_coverage_fraction :
  let pepperoni_diameter : ℝ := pizza_diameter / pepperoni_across_diameter
  let pepperoni_radius : ℝ := pepperoni_diameter / 2
  let pepperoni_area : ℝ := π * pepperoni_radius^2
  let effective_pepperoni_area : ℝ := pepperoni_area * (1 - pepperoni_overlap_fraction)
  let total_pepperoni_area : ℝ := pepperoni_count * effective_pepperoni_area
  let pizza_area : ℝ := π * (pizza_diameter / 2)^2
  total_pepperoni_area / pizza_area = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_pepperoni_coverage_fraction_l2361_236189


namespace NUMINAMATH_CALUDE_complex_on_line_l2361_236119

theorem complex_on_line (a : ℝ) : 
  (∃ z : ℂ, z = (a - Complex.I) / (1 + Complex.I) ∧ 
   z.re - z.im + 1 = 0) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_line_l2361_236119


namespace NUMINAMATH_CALUDE_path_count_theorem_l2361_236193

def grid_path (right up : ℕ) : ℕ := Nat.choose (right + up) up

theorem path_count_theorem :
  let right : ℕ := 6
  let up : ℕ := 4
  let total_path_length : ℕ := right + up
  grid_path right up = 210 := by
  sorry

end NUMINAMATH_CALUDE_path_count_theorem_l2361_236193


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l2361_236122

theorem cubic_equation_solutions (x y z n : ℕ+) :
  x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l2361_236122


namespace NUMINAMATH_CALUDE_intersection_empty_implies_m_range_l2361_236157

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x^2 + x + m + 2 = 0}
def B : Set ℝ := {x | x > 0}

-- Theorem statement
theorem intersection_empty_implies_m_range (m : ℝ) :
  A m ∩ B = ∅ → m ≤ -2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_empty_implies_m_range_l2361_236157


namespace NUMINAMATH_CALUDE_no_common_root_l2361_236132

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬ ∃ x₀ : ℝ, x₀^2 + b*x₀ + c = 0 ∧ x₀^2 + a*x₀ + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_root_l2361_236132


namespace NUMINAMATH_CALUDE_coefficient_x_squared_proof_l2361_236156

/-- The coefficient of x² in the expansion of (2x³ + 5x² - 3x)(3x² - 5x + 1) -/
def coefficient_x_squared : ℤ := 20

/-- The first polynomial in the product -/
def poly1 (x : ℚ) : ℚ := 2 * x^3 + 5 * x^2 - 3 * x

/-- The second polynomial in the product -/
def poly2 (x : ℚ) : ℚ := 3 * x^2 - 5 * x + 1

theorem coefficient_x_squared_proof :
  ∃ (a b c d e f : ℚ),
    poly1 x * poly2 x = a * x^5 + b * x^4 + c * x^3 + coefficient_x_squared * x^2 + e * x + f :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_proof_l2361_236156


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2361_236138

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x = 1) → ¬(x^2 = 1)) ↔ (¬(x = 1) → (x ≠ 1 ∧ x ≠ -1)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2361_236138


namespace NUMINAMATH_CALUDE_log_y_equality_l2361_236103

theorem log_y_equality (y : ℝ) (h : y = (Real.log 3 / Real.log 4) ^ (Real.log 9 / Real.log 3)) :
  Real.log y / Real.log 2 = 2 * Real.log (Real.log 3 / Real.log 2) / Real.log 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_log_y_equality_l2361_236103


namespace NUMINAMATH_CALUDE_markup_constant_l2361_236109

theorem markup_constant (C S : ℝ) (k : ℝ) (hk : k > 0) (hC : C > 0) (hS : S > 0) : 
  (S = C + k * S) → (k * S = 0.25 * C) → k = 1/5 := by
sorry

end NUMINAMATH_CALUDE_markup_constant_l2361_236109


namespace NUMINAMATH_CALUDE_system_solution_l2361_236166

theorem system_solution (x y : ℝ) : 
  (3 * x^2 + 9 * x + 3 * y + 2 = 0 ∧ 3 * x + y + 4 = 0) ↔ 
  (y = -4 + Real.sqrt 30 ∨ y = -4 - Real.sqrt 30) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2361_236166


namespace NUMINAMATH_CALUDE_vector_expression_not_equal_PQ_l2361_236170

variable {V : Type*} [AddCommGroup V]
variable (A B P Q : V)

theorem vector_expression_not_equal_PQ :
  A - B + B - P - (A - Q) ≠ P - Q :=
sorry

end NUMINAMATH_CALUDE_vector_expression_not_equal_PQ_l2361_236170


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_seven_halves_l2361_236197

theorem sqrt_expression_equals_seven_halves :
  Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 / Real.sqrt 24 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_seven_halves_l2361_236197


namespace NUMINAMATH_CALUDE_trapezoid_intersection_distances_l2361_236124

/-- Given a trapezoid ABCD with legs AB and CD, and bases AD and BC where AD > BC,
    this theorem proves the distances from the intersection point M of the extended legs
    to the vertices of the trapezoid. -/
theorem trapezoid_intersection_distances
  (AB CD AD BC : ℝ) -- Lengths of sides
  (h_AD_gt_BC : AD > BC) -- Condition: AD > BC
  : ∃ (BM AM CM DM : ℝ),
    BM = (AB * BC) / (AD - BC) ∧
    AM = (AB * AD) / (AD - BC) ∧
    CM = (CD * BC) / (AD - BC) ∧
    DM = (CD * AD) / (AD - BC) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_intersection_distances_l2361_236124


namespace NUMINAMATH_CALUDE_xy_zero_necessary_not_sufficient_l2361_236126

theorem xy_zero_necessary_not_sufficient (x y : ℝ) :
  (∀ x, x = 0 → x * y = 0) ∧ 
  ¬(∀ x y, x * y = 0 → x = 0) :=
sorry

end NUMINAMATH_CALUDE_xy_zero_necessary_not_sufficient_l2361_236126


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l2361_236118

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  (b - 2*a) * Real.cos C + c * Real.cos B = 0 →
  c = 2 →
  S = Real.sqrt 3 →
  S = 1/2 * a * b * Real.sin C →
  a^2 + b^2 - c^2 = 2*a*b * Real.cos C →
  C = π/3 ∧ a = 2 ∧ b = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l2361_236118


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l2361_236192

/-- The probability of drawing two white balls consecutively without replacement -/
theorem two_white_balls_probability 
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (red_balls : ℕ) 
  (h1 : total_balls = white_balls + red_balls)
  (h2 : white_balls = 5)
  (h3 : red_balls = 3) : 
  (white_balls : ℚ) / total_balls * ((white_balls - 1) : ℚ) / (total_balls - 1) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l2361_236192


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l2361_236111

/-- The magnitude of the sum of vectors (1, √3) and (-2, 0) is 2 -/
theorem magnitude_of_vector_sum : 
  let a : Fin 2 → ℝ := ![1, Real.sqrt 3]
  let b : Fin 2 → ℝ := ![-2, 0]
  Real.sqrt ((a 0 + b 0)^2 + (a 1 + b 1)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l2361_236111


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l2361_236187

theorem arithmetic_sequence_count (a₁ : ℝ) (aₙ : ℝ) (d : ℝ) (n : ℕ) :
  a₁ = 2.5 ∧ aₙ = 68.5 ∧ d = 6 →
  aₙ = a₁ + (n - 1) * d →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l2361_236187


namespace NUMINAMATH_CALUDE_binary_of_89_l2361_236150

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Theorem: The binary representation of 89 is [true, false, true, true, false, false, true] -/
theorem binary_of_89 :
  toBinary 89 = [true, false, true, true, false, false, true] := by
  sorry

#eval toBinary 89

end NUMINAMATH_CALUDE_binary_of_89_l2361_236150


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l2361_236135

/-- The first repeating decimal 0.030303... -/
def decimal1 : ℚ := 1 / 33

/-- The second repeating decimal 0.363636... -/
def decimal2 : ℚ := 4 / 11

/-- Theorem stating that the product of the two repeating decimals is 4/363 -/
theorem product_of_repeating_decimals :
  decimal1 * decimal2 = 4 / 363 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l2361_236135


namespace NUMINAMATH_CALUDE_scientific_notation_8350_l2361_236151

theorem scientific_notation_8350 : 
  8350 = 8.35 * (10 : ℝ)^3 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_8350_l2361_236151


namespace NUMINAMATH_CALUDE_sum_xy_equals_negative_two_l2361_236100

theorem sum_xy_equals_negative_two (x y : ℝ) :
  (x + y + 2)^2 + |2*x - 3*y - 1| = 0 → x + y = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_xy_equals_negative_two_l2361_236100


namespace NUMINAMATH_CALUDE_triangle_inequality_l2361_236130

theorem triangle_inequality (a b c : ℝ) (C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hC : 0 < C ∧ C < π) :
  c ≥ (a + b) * Real.sin (C / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2361_236130


namespace NUMINAMATH_CALUDE_sum_of_base8_digits_878_l2361_236107

/-- Converts a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The sum of the digits in the base 8 representation of 878 is 17 -/
theorem sum_of_base8_digits_878 :
  sumDigits (toBase8 878) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_base8_digits_878_l2361_236107


namespace NUMINAMATH_CALUDE_lactate_bicarbonate_reaction_in_extracellular_fluid_l2361_236123

-- Define the extracellular fluid
structure ExtracellularFluid where
  is_liquid_environment : Bool

-- Define a biochemical reaction
structure BiochemicalReaction where
  occurs_in_extracellular_fluid : Bool

-- Define the specific reaction
def lactate_bicarbonate_reaction : BiochemicalReaction where
  occurs_in_extracellular_fluid := true

-- Theorem statement
theorem lactate_bicarbonate_reaction_in_extracellular_fluid 
  (ecf : ExtracellularFluid) 
  (h : ecf.is_liquid_environment = true) : 
  lactate_bicarbonate_reaction.occurs_in_extracellular_fluid = true := by
  sorry

end NUMINAMATH_CALUDE_lactate_bicarbonate_reaction_in_extracellular_fluid_l2361_236123


namespace NUMINAMATH_CALUDE_sum_of_divisors_2i3j_l2361_236161

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the sum of divisors of 2^i * 3^j is 960, then i + j = 5 -/
theorem sum_of_divisors_2i3j (i j : ℕ) :
  sum_of_divisors (2^i * 3^j) = 960 → i + j = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_2i3j_l2361_236161


namespace NUMINAMATH_CALUDE_probability_at_least_one_head_three_coins_l2361_236179

theorem probability_at_least_one_head_three_coins :
  let p_head : ℝ := 1 / 2
  let p_tail : ℝ := 1 - p_head
  let p_three_tails : ℝ := p_tail ^ 3
  let p_at_least_one_head : ℝ := 1 - p_three_tails
  p_at_least_one_head = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_head_three_coins_l2361_236179


namespace NUMINAMATH_CALUDE_parabola_c_value_l2361_236190

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  x = p.a * y^2 + p.b * y + p.c

/-- Checks if (h, k) is the vertex of the parabola -/
def Parabola.hasVertex (p : Parabola) (h k : ℝ) : Prop :=
  h = p.a * k^2 + p.b * k + p.c ∧
  ∀ y, p.a * y^2 + p.b * y + p.c ≤ h

/-- States that the parabola opens downwards -/
def Parabola.opensDownwards (p : Parabola) : Prop :=
  p.a < 0

theorem parabola_c_value
  (p : Parabola)
  (vertex : p.hasVertex 5 3)
  (point : p.contains 7 6)
  (down : p.opensDownwards) :
  p.c = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2361_236190


namespace NUMINAMATH_CALUDE_knight_probability_after_2023_moves_l2361_236148

/-- Knight's move on an infinite chessboard -/
def KnightMove (a b : ℤ) : Set (ℤ × ℤ) :=
  {(a+1, b+2), (a+1, b-2), (a-1, b+2), (a-1, b-2),
   (a+2, b+1), (a+2, b-1), (a-2, b+1), (a-2, b-1)}

/-- Probability space for knight's moves -/
def KnightProbSpace : Type := ℤ × ℤ

/-- Probability measure for knight's moves -/
noncomputable def KnightProb : KnightProbSpace → ℝ := sorry

/-- The set of positions (a, b) where a ≡ 4 (mod 8) and b ≡ 5 (mod 8) -/
def TargetPositions : Set (ℤ × ℤ) :=
  {(a, b) | a % 8 = 4 ∧ b % 8 = 5}

/-- The probability of the knight being at a target position after n moves -/
noncomputable def ProbAtTargetAfterMoves (n : ℕ) : ℝ := sorry

theorem knight_probability_after_2023_moves :
  ProbAtTargetAfterMoves 2023 = 1/32 - 1/2^2027 := by sorry

end NUMINAMATH_CALUDE_knight_probability_after_2023_moves_l2361_236148


namespace NUMINAMATH_CALUDE_set_intersection_problem_l2361_236172

theorem set_intersection_problem :
  let A : Set ℤ := {-2, -1, 0, 1}
  let B : Set ℤ := {-1, 0, 1, 2}
  A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l2361_236172


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_a_less_than_one_l2361_236198

theorem quadratic_roots_imply_a_less_than_one (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_a_less_than_one_l2361_236198


namespace NUMINAMATH_CALUDE_no_proper_divisor_sum_set_equality_l2361_236155

theorem no_proper_divisor_sum_set_equality (n : ℕ) : 
  (∃ (d₁ d₂ d₃ : ℕ), 1 < d₁ ∧ d₁ < n ∧ d₁ ∣ n ∧
                     1 < d₂ ∧ d₂ < n ∧ d₂ ∣ n ∧
                     1 < d₃ ∧ d₃ < n ∧ d₃ ∣ n ∧
                     d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃) →
  ¬∃ (m : ℕ), {x : ℕ | ∃ (a b : ℕ), 1 < a ∧ a < n ∧ a ∣ n ∧
                                   1 < b ∧ b < n ∧ b ∣ n ∧
                                   x = a + b} =
              {y : ℕ | 1 < y ∧ y < m ∧ y ∣ m} :=
by sorry

end NUMINAMATH_CALUDE_no_proper_divisor_sum_set_equality_l2361_236155


namespace NUMINAMATH_CALUDE_power_calculation_l2361_236101

theorem power_calculation : (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2361_236101


namespace NUMINAMATH_CALUDE_duplicated_chromosome_configuration_l2361_236169

/-- Represents a duplicated chromosome -/
structure DuplicatedChromosome where
  centromeres : ℕ
  chromatids : ℕ
  dna_molecules : ℕ

/-- The correct configuration of a duplicated chromosome -/
def correct_configuration : DuplicatedChromosome :=
  { centromeres := 1
  , chromatids := 2
  , dna_molecules := 2 }

/-- Theorem stating that a duplicated chromosome has the correct configuration -/
theorem duplicated_chromosome_configuration :
  ∀ (dc : DuplicatedChromosome), dc = correct_configuration :=
by sorry

end NUMINAMATH_CALUDE_duplicated_chromosome_configuration_l2361_236169


namespace NUMINAMATH_CALUDE_f_of_5_eq_2515_l2361_236182

/-- The polynomial function f(x) -/
def f (x : ℝ) : ℝ := 3*x^5 - 15*x^4 + 27*x^3 - 20*x^2 - 72*x + 40

/-- Theorem: f(5) equals 2515 -/
theorem f_of_5_eq_2515 : f 5 = 2515 := by sorry

end NUMINAMATH_CALUDE_f_of_5_eq_2515_l2361_236182


namespace NUMINAMATH_CALUDE_alternate_angle_measure_l2361_236181

-- Define the angle measures as real numbers
def angle_A : ℝ := 0
def angle_B : ℝ := 0
def angle_C : ℝ := 0

-- State the theorem
theorem alternate_angle_measure :
  -- Conditions
  (angle_A = (1/4) * angle_B) →  -- ∠A is 1/4 of ∠B
  (angle_C = angle_A) →          -- ∠C and ∠A are alternate angles (due to parallel lines)
  (angle_B + angle_C = 180) →    -- ∠B and ∠C form a straight line
  -- Conclusion
  (angle_C = 36) := by
  sorry

end NUMINAMATH_CALUDE_alternate_angle_measure_l2361_236181


namespace NUMINAMATH_CALUDE_binomial_coefficient_7_4_l2361_236176

theorem binomial_coefficient_7_4 : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_7_4_l2361_236176


namespace NUMINAMATH_CALUDE_exchange_problem_l2361_236125

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Represents the exchange problem and proves the sum of digits -/
theorem exchange_problem (d : ℕ) : 
  (11 * d : ℚ) / 8 - 70 = d → sumOfDigits d = 16 := by
  sorry

#eval sumOfDigits 187  -- Expected output: 16

end NUMINAMATH_CALUDE_exchange_problem_l2361_236125


namespace NUMINAMATH_CALUDE_student_score_problem_l2361_236117

theorem student_score_problem (total_questions : ℕ) (student_score : ℤ) 
  (h1 : total_questions = 100)
  (h2 : student_score = 61) : 
  ∃ (correct_answers : ℕ), 
    correct_answers = 87 ∧ 
    student_score = correct_answers - 2 * (total_questions - correct_answers) :=
by
  sorry

end NUMINAMATH_CALUDE_student_score_problem_l2361_236117


namespace NUMINAMATH_CALUDE_inequality_proof_l2361_236140

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) : a < a * b^2 ∧ a * b^2 < a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2361_236140


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2361_236195

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p (1, 0) = |p.1 + 1|}

-- Define the property of line l intersecting C at M and N
def intersects_at_MN (l : Set (ℝ × ℝ)) (M N : ℝ × ℝ) : Prop :=
  M ∈ C ∧ N ∈ C ∧ M ∈ l ∧ N ∈ l ∧ M ≠ N ∧ M ≠ (0, 0) ∧ N ≠ (0, 0)

-- Define the perpendicularity of OM and ON
def OM_perp_ON (M N : ℝ × ℝ) : Prop :=
  M.1 * N.1 + M.2 * N.2 = 0

-- Theorem statement
theorem line_passes_through_fixed_point :
  ∀ l : Set (ℝ × ℝ), ∀ M N : ℝ × ℝ,
  intersects_at_MN l M N → OM_perp_ON M N →
  (4, 0) ∈ l :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2361_236195


namespace NUMINAMATH_CALUDE_complex_division_simplification_l2361_236185

theorem complex_division_simplification :
  let i : ℂ := Complex.I
  let z : ℂ := 5 / (2 - i)
  z = 2 + i := by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l2361_236185


namespace NUMINAMATH_CALUDE_point_order_l2361_236115

def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 12

theorem point_order (y₁ y₂ y₃ : ℝ) 
  (h₁ : f (-1) = y₁)
  (h₂ : f (-3) = y₂)
  (h₃ : f 2 = y₃) :
  y₃ > y₂ ∧ y₂ > y₁ := by
  sorry

end NUMINAMATH_CALUDE_point_order_l2361_236115


namespace NUMINAMATH_CALUDE_cookie_sheet_width_l2361_236175

/-- Represents a rectangle with given width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.length)

/-- Theorem: A rectangle with length 2 and perimeter 24 has width 10 -/
theorem cookie_sheet_width : 
  ∀ (r : Rectangle), r.length = 2 → r.perimeter = 24 → r.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sheet_width_l2361_236175


namespace NUMINAMATH_CALUDE_det_dilation_matrix_5_l2361_236142

/-- A dilation matrix with scale factor k -/
def dilationMatrix (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ => k)

/-- Theorem: The determinant of a 3x3 dilation matrix with scale factor 5 is 125 -/
theorem det_dilation_matrix_5 :
  let D := dilationMatrix 5
  Matrix.det D = 125 := by sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_5_l2361_236142


namespace NUMINAMATH_CALUDE_min_sum_is_two_l2361_236106

/-- Represents a sequence of five digits -/
def DigitSequence := Fin 5 → Nat

/-- Ensures all digits in the sequence are between 1 and 9 -/
def valid_sequence (s : DigitSequence) : Prop :=
  ∀ i, 1 ≤ s i ∧ s i ≤ 9

/-- Computes the sum of the last four digits in the sequence -/
def sum_last_four (s : DigitSequence) : Nat :=
  (s 1) + (s 2) + (s 3) + (s 4)

/-- Represents the evolution rule for the sequence -/
def evolve (s : DigitSequence) : DigitSequence :=
  fun i => match i with
    | 0 => s 1
    | 1 => s 2
    | 2 => s 3
    | 3 => s 4
    | 4 => sum_last_four s % 10

/-- Represents the sum of all digits in the sequence -/
def sequence_sum (s : DigitSequence) : Nat :=
  (s 0) + (s 1) + (s 2) + (s 3) + (s 4)

/-- The main theorem stating that the minimum sum is 2 -/
theorem min_sum_is_two :
  ∃ (s : DigitSequence), valid_sequence s ∧
  ∀ (n : Nat), sequence_sum (Nat.iterate evolve n s) ≥ 2 ∧
  ∃ (m : Nat), sequence_sum (Nat.iterate evolve m s) = 2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_is_two_l2361_236106


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l2361_236112

theorem fifteenth_student_age
  (n : ℕ)
  (total_students : n = 15)
  (avg_age : ℝ)
  (total_avg : avg_age = 15)
  (group1_size group2_size : ℕ)
  (group1_avg group2_avg : ℝ)
  (group_sizes : group1_size = 7 ∧ group2_size = 7)
  (group_avgs : group1_avg = 14 ∧ group2_avg = 16)
  : ∃ (fifteenth_age : ℝ), fifteenth_age = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l2361_236112


namespace NUMINAMATH_CALUDE_lottery_winning_probability_l2361_236174

/-- The number of balls for MegaBall selection -/
def megaBallCount : ℕ := 30

/-- The number of balls for WinnerBall selection -/
def winnerBallCount : ℕ := 45

/-- The number of WinnerBalls to be drawn -/
def winnerBallDrawCount : ℕ := 6

/-- The probability of winning the lottery -/
def winningProbability : ℚ := 1 / 244351800

/-- Theorem stating the probability of winning the lottery -/
theorem lottery_winning_probability :
  (1 / megaBallCount) * (1 / (winnerBallCount.choose winnerBallDrawCount)) = winningProbability := by
  sorry

end NUMINAMATH_CALUDE_lottery_winning_probability_l2361_236174


namespace NUMINAMATH_CALUDE_network_connections_l2361_236102

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l2361_236102


namespace NUMINAMATH_CALUDE_perpendicular_slope_l2361_236113

/-- Given a line with equation 4x - 5y = 10, the slope of the perpendicular line is -5/4 -/
theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 5 * y = 10) → (slope_of_perpendicular_line = -5/4) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l2361_236113


namespace NUMINAMATH_CALUDE_x_percent_of_x_equals_nine_l2361_236127

theorem x_percent_of_x_equals_nine (x : ℝ) : 
  x > 0 → (x / 100) * x = 9 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_x_percent_of_x_equals_nine_l2361_236127


namespace NUMINAMATH_CALUDE_parabola_one_x_intercept_l2361_236110

/-- The parabola defined by x = -3y^2 + 2y + 3 has exactly one x-intercept. -/
theorem parabola_one_x_intercept : 
  ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 3 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_one_x_intercept_l2361_236110
