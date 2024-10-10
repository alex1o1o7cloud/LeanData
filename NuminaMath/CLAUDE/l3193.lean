import Mathlib

namespace sum_four_consecutive_odd_divisible_by_two_l3193_319361

theorem sum_four_consecutive_odd_divisible_by_two (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1) + (2*n + 3) + (2*n + 5) + (2*n + 7) = 2 * k := by
  sorry

end sum_four_consecutive_odd_divisible_by_two_l3193_319361


namespace equation_solution_l3193_319318

theorem equation_solution : ∃! x : ℝ, (128 : ℝ)^(x - 1) / (16 : ℝ)^(x - 1) = (64 : ℝ)^(3 * x) ∧ x = -1/5 := by
  sorry

end equation_solution_l3193_319318


namespace can_make_all_white_l3193_319382

/-- Represents the color of a number -/
inductive Color
| Black
| White

/-- Represents a move in the repainting process -/
structure Move where
  number : Nat
  deriving Repr

/-- The state of all numbers from 1 to 1,000,000 -/
def State := Fin 1000000 → Color

/-- Apply a move to a state -/
def applyMove (s : State) (m : Move) : State :=
  sorry

/-- Check if all numbers in the state are white -/
def allWhite (s : State) : Prop :=
  sorry

/-- The initial state where all numbers are black -/
def initialState : State :=
  sorry

/-- Theorem stating that it's possible to make all numbers white -/
theorem can_make_all_white : ∃ (moves : List Move), allWhite (moves.foldl applyMove initialState) := by
  sorry

end can_make_all_white_l3193_319382


namespace digit_change_sum_inequality_l3193_319373

/-- Changes each digit of a positive integer by 1 (either up or down) -/
def change_digits (n : ℕ) : ℕ :=
  sorry

theorem digit_change_sum_inequality (a b : ℕ) :
  let c := a + b
  let a' := change_digits a
  let b' := change_digits b
  let c' := change_digits c
  a' + b' ≠ c' :=
by sorry

end digit_change_sum_inequality_l3193_319373


namespace log_inequality_implies_range_l3193_319336

theorem log_inequality_implies_range (x : ℝ) (hx : x > 0) :
  (Real.log x) ^ 2015 < (Real.log x) ^ 2014 ∧ 
  (Real.log x) ^ 2014 < (Real.log x) ^ 2016 →
  0 < x ∧ x < (1/10 : ℝ) := by sorry

end log_inequality_implies_range_l3193_319336


namespace total_pages_in_textbooks_l3193_319398

/-- Represents the number of pages in each textbook and calculates the total --/
def textbook_pages : ℕ → ℕ → ℕ → ℕ → ℕ := fun history geography math science =>
  history + geography + math + science

/-- Theorem stating the total number of pages in Suzanna's textbooks --/
theorem total_pages_in_textbooks : ∃ (history geography math science : ℕ),
  history = 160 ∧
  geography = history + 70 ∧
  math = (history + geography) / 2 ∧
  science = 2 * history ∧
  textbook_pages history geography math science = 905 := by
  sorry

#eval textbook_pages 160 230 195 320

end total_pages_in_textbooks_l3193_319398


namespace geometric_series_r_value_l3193_319352

/-- Given a geometric series with first term a and common ratio r,
    S is the sum of the entire series,
    S_odd is the sum of terms with odd powers of r -/
def geometric_series (a r : ℝ) (S S_odd : ℝ) : Prop :=
  ∃ (n : ℕ), S = a * (1 - r^n) / (1 - r) ∧
             S_odd = a * r * (1 - r^(2*n)) / (1 - r^2)

theorem geometric_series_r_value (a r : ℝ) :
  geometric_series a r 20 8 → r = 2/3 := by
  sorry

end geometric_series_r_value_l3193_319352


namespace chocolate_cost_l3193_319323

theorem chocolate_cost (candies_per_box : ℕ) (cost_per_box : ℕ) (total_candies : ℕ) :
  candies_per_box = 25 →
  cost_per_box = 6 →
  total_candies = 600 →
  (total_candies / candies_per_box) * cost_per_box = 144 :=
by sorry

end chocolate_cost_l3193_319323


namespace investment_calculation_l3193_319341

/-- Calculates the total investment in shares given the following conditions:
  * Face value of shares is 100 rupees
  * Shares are bought at a 20% premium
  * Company declares a 6% dividend
  * Total dividend received is 720 rupees
-/
def calculate_investment (face_value : ℕ) (premium_percent : ℕ) (dividend_percent : ℕ) (total_dividend : ℕ) : ℕ :=
  let premium_price := face_value + face_value * premium_percent / 100
  let dividend_per_share := face_value * dividend_percent / 100
  let num_shares := total_dividend / dividend_per_share
  num_shares * premium_price

/-- Theorem stating that under the given conditions, the total investment is 14400 rupees -/
theorem investment_calculation :
  calculate_investment 100 20 6 720 = 14400 := by
  sorry

end investment_calculation_l3193_319341


namespace statement_II_must_be_true_l3193_319391

-- Define the possible contents of the card
inductive CardContent
| Number : Nat → CardContent
| Symbol : Char → CardContent

-- Define the statements
def statementI (c : CardContent) : Prop :=
  match c with
  | CardContent.Symbol _ => True
  | CardContent.Number _ => False

def statementII (c : CardContent) : Prop :=
  match c with
  | CardContent.Symbol '%' => False
  | _ => True

def statementIII (c : CardContent) : Prop :=
  c = CardContent.Number 3

def statementIV (c : CardContent) : Prop :=
  c ≠ CardContent.Number 4

-- Theorem statement
theorem statement_II_must_be_true :
  ∃ (c : CardContent),
    (statementI c ∧ statementII c ∧ statementIII c) ∨
    (statementI c ∧ statementII c ∧ statementIV c) ∨
    (statementI c ∧ statementIII c ∧ statementIV c) ∨
    (statementII c ∧ statementIII c ∧ statementIV c) :=
  sorry

end statement_II_must_be_true_l3193_319391


namespace rain_probability_theorem_l3193_319343

/-- Given probabilities for rain events in counties -/
theorem rain_probability_theorem 
  (p_monday : ℝ) 
  (p_neither : ℝ) 
  (p_both : ℝ) 
  (h1 : p_monday = 0.7) 
  (h2 : p_neither = 0.35) 
  (h3 : p_both = 0.6) :
  ∃ (p_tuesday : ℝ), p_tuesday = 0.55 := by
sorry


end rain_probability_theorem_l3193_319343


namespace greatest_c_for_non_range_greatest_integer_c_l3193_319365

theorem greatest_c_for_non_range (c : ℤ) : 
  (∀ x : ℝ, x^2 + c*x + 20 ≠ 5) ↔ c^2 < 60 :=
sorry

theorem greatest_integer_c : 
  ∃ c : ℤ, c = 7 ∧ (∀ x : ℝ, x^2 + c*x + 20 ≠ 5) ∧ 
  (∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 20 = 5) :=
sorry

end greatest_c_for_non_range_greatest_integer_c_l3193_319365


namespace annual_turbans_count_l3193_319313

/-- Represents the annual salary structure and partial payment details --/
structure SalaryInfo where
  annual_cash : ℕ  -- Annual cash component in Rupees
  turban_price : ℕ  -- Price of one turban in Rupees
  partial_months : ℕ  -- Number of months worked
  partial_cash : ℕ  -- Cash received for partial work in Rupees
  partial_turbans : ℕ  -- Number of turbans received for partial work

/-- Calculates the number of turbans in the annual salary --/
def calculate_annual_turbans (info : SalaryInfo) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the number of turbans in the annual salary is 1 --/
theorem annual_turbans_count (info : SalaryInfo) 
  (h1 : info.annual_cash = 90)
  (h2 : info.turban_price = 50)
  (h3 : info.partial_months = 9)
  (h4 : info.partial_cash = 55)
  (h5 : info.partial_turbans = 1) :
  calculate_annual_turbans info = 1 := by
  sorry

end annual_turbans_count_l3193_319313


namespace lyssa_incorrect_percentage_is_12_l3193_319374

def exam_items : ℕ := 75
def precious_mistakes : ℕ := 12
def lyssa_additional_correct : ℕ := 3

def lyssa_incorrect_percentage : ℚ :=
  (exam_items - (exam_items - precious_mistakes + lyssa_additional_correct)) / exam_items * 100

theorem lyssa_incorrect_percentage_is_12 :
  lyssa_incorrect_percentage = 12 := by sorry

end lyssa_incorrect_percentage_is_12_l3193_319374


namespace segment_length_is_52_l3193_319337

/-- A right triangle with sides 10, 24, and 26 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_a : a = 10
  side_b : b = 24
  side_c : c = 26

/-- Three identical circles inscribed in the triangle -/
structure InscribedCircles where
  radius : ℝ
  radius_value : radius = 2
  touches_sides : Bool
  touches_other_circles : Bool

/-- The total length of segments from vertices to tangency points -/
def total_segment_length (t : RightTriangle) (circles : InscribedCircles) : ℝ :=
  (t.a - circles.radius) + (t.b - circles.radius) + (t.c - 2 * circles.radius)

theorem segment_length_is_52 (t : RightTriangle) (circles : InscribedCircles) :
  total_segment_length t circles = 52 :=
sorry

end segment_length_is_52_l3193_319337


namespace fall_semester_duration_l3193_319350

/-- The duration of the fall semester in weeks -/
def semester_length : ℕ := 15

/-- The number of hours Paris studies during weekdays -/
def weekday_hours : ℕ := 3

/-- The number of hours Paris studies on Saturday -/
def saturday_hours : ℕ := 4

/-- The number of hours Paris studies on Sunday -/
def sunday_hours : ℕ := 5

/-- The total number of hours Paris studies during the semester -/
def total_study_hours : ℕ := 360

theorem fall_semester_duration :
  semester_length * (5 * weekday_hours + saturday_hours + sunday_hours) = total_study_hours := by
  sorry

end fall_semester_duration_l3193_319350


namespace fraction_undefined_at_two_l3193_319342

theorem fraction_undefined_at_two (x : ℝ) : 
  x / (2 - x) = x / (2 - x) → x ≠ 2 :=
by
  sorry

end fraction_undefined_at_two_l3193_319342


namespace cube_root_of_two_solves_equation_l3193_319321

theorem cube_root_of_two_solves_equation :
  ∃ x : ℝ, x^3 = 2 ∧ x = Real.rpow 2 (1/3) :=
sorry

end cube_root_of_two_solves_equation_l3193_319321


namespace man_rowing_speed_l3193_319384

/-- Proves that given a man's speed in still water and downstream speed, his upstream speed can be calculated. -/
theorem man_rowing_speed (v_still : ℝ) (v_downstream : ℝ) (h1 : v_still = 50) (h2 : v_downstream = 80) :
  v_still - (v_downstream - v_still) = 20 := by
  sorry

#check man_rowing_speed

end man_rowing_speed_l3193_319384


namespace coordinate_axis_angles_characterization_l3193_319364

-- Define the set of angles whose terminal sides lie on the coordinate axes
def CoordinateAxisAngles : Set ℝ :=
  {α | ∃ n : ℤ, α = n * Real.pi / 2}

-- Theorem stating that the set of angles whose terminal sides lie on the coordinate axes
-- is equal to the set {α | α = nπ/2, n ∈ ℤ}
theorem coordinate_axis_angles_characterization :
  CoordinateAxisAngles = {α | ∃ n : ℤ, α = n * Real.pi / 2} := by
  sorry

end coordinate_axis_angles_characterization_l3193_319364


namespace composite_sum_product_l3193_319387

theorem composite_sum_product (a b c d e : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧
  a^4 + b^4 = c^4 + d^4 ∧
  a^4 + b^4 = e^5 →
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ x * y = a * c + b * d :=
by sorry

end composite_sum_product_l3193_319387


namespace pizza_cost_per_slice_l3193_319302

-- Define the pizza and topping costs
def large_pizza_cost : ℚ := 10
def first_topping_cost : ℚ := 2
def next_two_toppings_cost : ℚ := 1
def remaining_toppings_cost : ℚ := 0.5

-- Define the number of slices and toppings
def num_slices : ℕ := 8
def num_toppings : ℕ := 7

-- Calculate the total cost of toppings
def total_toppings_cost : ℚ :=
  first_topping_cost +
  2 * next_two_toppings_cost +
  (num_toppings - 3) * remaining_toppings_cost

-- Calculate the total cost of the pizza
def total_pizza_cost : ℚ := large_pizza_cost + total_toppings_cost

-- Theorem to prove
theorem pizza_cost_per_slice :
  total_pizza_cost / num_slices = 2 := by sorry

end pizza_cost_per_slice_l3193_319302


namespace shopkeeper_mango_profit_l3193_319378

/-- Calculates the profit percentage given the cost price and selling price -/
def profit_percent (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: A shopkeeper who buys mangoes at 6 for 1 rupee and sells them at 3 for 1 rupee makes a 100% profit -/
theorem shopkeeper_mango_profit :
  let cost_price : ℚ := 1 / 6  -- Cost price per mango
  let selling_price : ℚ := 1 / 3  -- Selling price per mango
  profit_percent cost_price selling_price = 100 := by
sorry

end shopkeeper_mango_profit_l3193_319378


namespace younger_brother_age_l3193_319392

/-- Represents the age of Viggo's younger brother -/
def brother_age : ℕ := sorry

/-- Represents Viggo's age -/
def viggo_age : ℕ := sorry

/-- The age difference between Viggo and his brother remains constant -/
axiom age_difference : viggo_age - brother_age = 12

/-- Viggo's age was 10 years more than twice his younger brother's age when his brother was 2 -/
axiom initial_age_relation : viggo_age - brother_age = 2 * 2 + 10 - 2

/-- The sum of their current ages is 32 -/
axiom current_age_sum : brother_age + viggo_age = 32

theorem younger_brother_age : brother_age = 10 := by sorry

end younger_brother_age_l3193_319392


namespace bus_meeting_time_l3193_319346

structure BusJourney where
  totalDistance : ℝ
  distanceToCountyTown : ℝ
  bus1DepartureTime : ℝ
  bus1ArrivalCountyTown : ℝ
  bus1StopTime : ℝ
  bus1ArrivalProvincialCapital : ℝ
  bus2DepartureTime : ℝ
  bus2Speed : ℝ

def meetingTime (j : BusJourney) : ℝ := sorry

theorem bus_meeting_time (j : BusJourney) 
  (h1 : j.totalDistance = 189)
  (h2 : j.distanceToCountyTown = 54)
  (h3 : j.bus1DepartureTime = 8.5)
  (h4 : j.bus1ArrivalCountyTown = 9.25)
  (h5 : j.bus1StopTime = 0.25)
  (h6 : j.bus1ArrivalProvincialCapital = 11)
  (h7 : j.bus2DepartureTime = 9)
  (h8 : j.bus2Speed = 60) :
  meetingTime j = 72 / 60 := by sorry

end bus_meeting_time_l3193_319346


namespace calculation_proof_l3193_319327

theorem calculation_proof : 
  47 * ((4 + 3/7) - (5 + 1/3)) / ((3 + 1/2) + (2 + 1/5)) = -(7 + 119/171) := by
  sorry

end calculation_proof_l3193_319327


namespace min_vertical_distance_l3193_319315

/-- The minimum vertical distance between y = |x-1| and y = -x^2 - 5x - 6 is 4 -/
theorem min_vertical_distance : ∃ (d : ℝ), d = 4 ∧ 
  ∀ (x : ℝ), d ≤ |x - 1| - (-x^2 - 5*x - 6) :=
by sorry

end min_vertical_distance_l3193_319315


namespace no_natural_solution_l3193_319311

theorem no_natural_solution :
  ¬ ∃ (x y z t : ℕ), 16^x + 21^y + 26^z = t^2 := by
sorry

end no_natural_solution_l3193_319311


namespace complex_magnitude_two_thirds_plus_three_i_l3193_319325

theorem complex_magnitude_two_thirds_plus_three_i :
  Complex.abs (2/3 + 3*Complex.I) = Real.sqrt 85 / 3 := by
  sorry

end complex_magnitude_two_thirds_plus_three_i_l3193_319325


namespace louise_cakes_proof_l3193_319385

/-- The number of cakes Louise needs for the gathering -/
def total_cakes : ℕ := 60

/-- The number of cakes Louise has already baked -/
def baked_cakes : ℕ := total_cakes / 2

/-- The number of cakes Louise bakes on the second day -/
def second_day_bakes : ℕ := (total_cakes - baked_cakes) / 2

/-- The number of cakes Louise bakes on the third day -/
def third_day_bakes : ℕ := (total_cakes - baked_cakes - second_day_bakes) / 3

/-- The number of cakes left to bake after the third day -/
def remaining_cakes : ℕ := total_cakes - baked_cakes - second_day_bakes - third_day_bakes

theorem louise_cakes_proof : remaining_cakes = 10 := by
  sorry

#eval total_cakes
#eval remaining_cakes

end louise_cakes_proof_l3193_319385


namespace largest_y_in_special_right_triangle_l3193_319314

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_y_in_special_right_triangle (x y z : ℕ) 
  (h1 : is_prime x ∧ is_prime y ∧ is_prime z)
  (h2 : x + y + z = 90)
  (h3 : y < x)
  (h4 : y > z) :
  y ≤ 47 ∧ ∃ (x' z' : ℕ), is_prime x' ∧ is_prime z' ∧ x' + 47 + z' = 90 ∧ 47 < x' ∧ 47 > z' :=
sorry

end largest_y_in_special_right_triangle_l3193_319314


namespace cut_depth_proof_l3193_319396

theorem cut_depth_proof (sheet_width sheet_height : ℕ) 
  (cut_width_1 cut_width_2 cut_width_3 : ℕ → ℕ) 
  (remaining_area : ℕ) : 
  sheet_width = 80 → 
  sheet_height = 15 → 
  (∀ d : ℕ, cut_width_1 d = 5 * d) →
  (∀ d : ℕ, cut_width_2 d = 15 * d) →
  (∀ d : ℕ, cut_width_3 d = 10 * d) →
  remaining_area = 990 →
  ∃ d : ℕ, d = 7 ∧ 
    sheet_width * sheet_height - (cut_width_1 d + cut_width_2 d + cut_width_3 d) = remaining_area :=
by sorry

end cut_depth_proof_l3193_319396


namespace gerald_expense_l3193_319317

/-- Represents Gerald's baseball supplies expense situation -/
structure BaseballExpenses where
  season_length : ℕ
  saving_months : ℕ
  chore_price : ℕ
  chores_per_month : ℕ

/-- Calculates the monthly expense for baseball supplies -/
def monthly_expense (e : BaseballExpenses) : ℕ :=
  (e.saving_months * e.chores_per_month * e.chore_price) / e.season_length

/-- Theorem: Given Gerald's specific situation, his monthly expense is $100 -/
theorem gerald_expense :
  let e : BaseballExpenses := {
    season_length := 4,
    saving_months := 8,
    chore_price := 10,
    chores_per_month := 5
  }
  monthly_expense e = 100 := by sorry

end gerald_expense_l3193_319317


namespace min_value_expression_l3193_319351

theorem min_value_expression (a b c : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 4) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (4/c - 1)^2 ≥ 12 - 8 * Real.sqrt 2 := by
  sorry

end min_value_expression_l3193_319351


namespace neg_two_oplus_three_solve_equation_find_expression_value_l3193_319383

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := 2 * a - a * b

-- Theorem 1
theorem neg_two_oplus_three : oplus (-2) 3 = 2 := by sorry

-- Theorem 2
theorem solve_equation (x : ℝ) : oplus (-3) x = oplus (x + 1) 5 → x = 1/2 := by sorry

-- Theorem 3
theorem find_expression_value (x y : ℝ) : oplus x 1 = 2 * (oplus 1 y) → (1/2) * x + y + 1 = 3 := by sorry

end neg_two_oplus_three_solve_equation_find_expression_value_l3193_319383


namespace pencils_in_drawer_l3193_319301

/-- The total number of pencils after adding more -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of pencils is 71 -/
theorem pencils_in_drawer : total_pencils 41 30 = 71 := by
  sorry

end pencils_in_drawer_l3193_319301


namespace tangent_slope_at_two_l3193_319338

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem tangent_slope_at_two
  (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = x / (x - 1))
  : deriv f 2 = 1 / 9 :=
sorry

end tangent_slope_at_two_l3193_319338


namespace functional_equation_l3193_319303

-- Define the function f
def f (x : ℝ) : ℝ := 1 - x^2

-- State the theorem
theorem functional_equation (x : ℝ) : x^2 * f x + f (1 - x) = 2*x - x^4 := by
  sorry

end functional_equation_l3193_319303


namespace fenced_area_with_cutouts_l3193_319309

/-- The area of a fenced region with cutouts -/
theorem fenced_area_with_cutouts :
  let rectangle_length : ℝ := 20
  let rectangle_width : ℝ := 18
  let square_side : ℝ := 4
  let triangle_leg : ℝ := 3
  let rectangle_area := rectangle_length * rectangle_width
  let square_cutout_area := square_side * square_side
  let triangle_cutout_area := (1 / 2) * triangle_leg * triangle_leg
  rectangle_area - square_cutout_area - triangle_cutout_area = 339.5 := by
sorry

end fenced_area_with_cutouts_l3193_319309


namespace alligator_growth_in_year_l3193_319363

def initial_population : ℝ := 4
def growth_factor : ℝ := 1.5
def months : ℕ := 12

def alligator_population (t : ℕ) : ℝ :=
  initial_population * growth_factor ^ t

theorem alligator_growth_in_year :
  alligator_population months = 518.9853515625 :=
sorry

end alligator_growth_in_year_l3193_319363


namespace product_remainder_by_ten_l3193_319367

theorem product_remainder_by_ten : 
  (2468 * 7531 * 92045) % 10 = 0 := by
  sorry

end product_remainder_by_ten_l3193_319367


namespace expression_evaluation_l3193_319307

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -1) :
  -2*a - b^2 + 2*a*b = -17 := by sorry

end expression_evaluation_l3193_319307


namespace reflection_line_sum_l3193_319349

/-- Given a point and its image under reflection across a line, prove the sum of the line's slope and y-intercept. -/
theorem reflection_line_sum (x₁ y₁ x₂ y₂ : ℝ) (m b : ℝ) 
  (h₁ : (x₁, y₁) = (2, 3))  -- Original point
  (h₂ : (x₂, y₂) = (10, 7))  -- Image point
  (h₃ : ∀ x y, y = m * x + b →  -- Reflection line equation
              (x - x₁) * (x - x₂) + (y - y₁) * (y - y₂) = 0) :
  m + b = 15 := by sorry

end reflection_line_sum_l3193_319349


namespace fifteenth_row_seats_l3193_319386

/-- Represents the number of seats in a row of the stadium -/
def seats (n : ℕ) : ℕ := 5 + 2 * (n - 1)

/-- Theorem stating that the 15th row has 33 seats -/
theorem fifteenth_row_seats :
  seats 15 = 33 := by
  sorry

end fifteenth_row_seats_l3193_319386


namespace planes_and_perpendicular_lines_l3193_319330

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem planes_and_perpendicular_lines 
  (α β : Plane) (m n : Line) :
  parallel α β → 
  perpendicular n α → 
  perpendicular m β → 
  line_parallel m n :=
by sorry

end planes_and_perpendicular_lines_l3193_319330


namespace base6_divisibility_by_13_l3193_319300

/-- Converts a base-6 number of the form 3dd4₆ to base 10 --/
def base6ToBase10 (d : ℕ) : ℕ := 3 * 6^3 + d * 6^2 + d * 6^1 + 4 * 6^0

/-- Checks if a natural number is a valid base-6 digit --/
def isBase6Digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 5

theorem base6_divisibility_by_13 :
  ∀ d : ℕ, isBase6Digit d → (base6ToBase10 d % 13 = 0 ↔ d = 4) := by sorry

end base6_divisibility_by_13_l3193_319300


namespace no_square_divisibility_l3193_319372

theorem no_square_divisibility (a b : ℕ) (α : ℕ) (ha : a > 1) (hb : b > 1) 
  (hodd_a : Odd a) (hodd_b : Odd b) (hsum : a + b = 2^α) (hα : α ≥ 1) :
  ¬∃ (k : ℕ), k > 1 ∧ (k^2 ∣ a^k + b^k) := by
  sorry

end no_square_divisibility_l3193_319372


namespace number_difference_l3193_319394

theorem number_difference (x y : ℕ) : 
  x + y = 50 → 
  y = 31 → 
  x < 2 * y → 
  2 * y - x = 43 := by
  sorry

end number_difference_l3193_319394


namespace rhombus_area_with_diagonals_6_and_8_l3193_319320

/-- The area of a rhombus with diagonals of lengths 6 and 8 is 24. -/
theorem rhombus_area_with_diagonals_6_and_8 : 
  ∀ (r : ℝ × ℝ → ℝ), 
  (∀ d₁ d₂, r (d₁, d₂) = (1/2) * d₁ * d₂) →
  r (6, 8) = 24 := by
sorry

end rhombus_area_with_diagonals_6_and_8_l3193_319320


namespace perfect_square_condition_l3193_319380

theorem perfect_square_condition (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + k*x + 25 = y^2) → (k = 10 ∨ k = -10) := by
  sorry

end perfect_square_condition_l3193_319380


namespace ruels_usable_stamps_l3193_319379

/-- The number of usable stamps Ruel has -/
def usable_stamps : ℕ :=
  let books_10 := 4
  let stamps_per_book_10 := 10
  let books_15 := 6
  let stamps_per_book_15 := 15
  let books_25 := 3
  let stamps_per_book_25 := 25
  let books_30 := 2
  let stamps_per_book_30 := 30
  let damaged_25 := 5
  let damaged_30 := 3
  let total_stamps := books_10 * stamps_per_book_10 +
                      books_15 * stamps_per_book_15 +
                      books_25 * stamps_per_book_25 +
                      books_30 * stamps_per_book_30
  let total_damaged := damaged_25 + damaged_30
  total_stamps - total_damaged

theorem ruels_usable_stamps :
  usable_stamps = 257 := by
  sorry

end ruels_usable_stamps_l3193_319379


namespace sum_of_roots_equals_eighteen_l3193_319312

theorem sum_of_roots_equals_eighteen : 
  let f (x : ℝ) := (3 * x^3 + 2 * x^2 - 9 * x + 15) - (4 * x^3 - 16 * x^2 + 27)
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃)) ∧ r₁ + r₂ + r₃ = 18 := by
  sorry

end sum_of_roots_equals_eighteen_l3193_319312


namespace scientific_notation_of_2410000_l3193_319381

theorem scientific_notation_of_2410000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 2410000 = a * (10 : ℝ) ^ n ∧ a = 2.41 ∧ n = 6 := by
  sorry

end scientific_notation_of_2410000_l3193_319381


namespace division_remainder_proof_l3193_319316

theorem division_remainder_proof (a b : ℕ) 
  (h1 : a - b = 2415)
  (h2 : a = 2520)
  (h3 : a / b = 21) : 
  a % b = 315 := by
sorry

end division_remainder_proof_l3193_319316


namespace circle_c_equation_l3193_319358

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  passes_through : ℝ × ℝ

-- Define the equation of a circle
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = 
    (c.passes_through.1 - c.center.1)^2 + (c.passes_through.2 - c.center.2)^2

-- Theorem statement
theorem circle_c_equation :
  let c : Circle := { center := (1, 1), passes_through := (0, 0) }
  ∀ x y : ℝ, circle_equation c x y ↔ (x - 1)^2 + (y - 1)^2 = 2 :=
by sorry

end circle_c_equation_l3193_319358


namespace water_for_bread_dough_l3193_319395

/-- The amount of water (in mL) needed for a given amount of flour (in mL),
    given a water-to-flour ratio. -/
def water_needed (water_ratio : ℚ) (flour_amount : ℚ) : ℚ :=
  (water_ratio * flour_amount)

/-- Theorem stating that for 1000 mL of flour, given the ratio of 80 mL water
    to 200 mL flour, the amount of water needed is 400 mL. -/
theorem water_for_bread_dough : water_needed (80 / 200) 1000 = 400 := by
  sorry

end water_for_bread_dough_l3193_319395


namespace trader_profit_loss_percentage_trader_overall_loss_l3193_319371

/-- Calculates the overall profit or loss percentage for a trader selling two cars -/
theorem trader_profit_loss_percentage 
  (selling_price : ℝ) 
  (gain_percentage : ℝ) 
  (loss_percentage : ℝ) : ℝ :=
  let cost_price1 := selling_price / (1 + gain_percentage / 100)
  let cost_price2 := selling_price / (1 - loss_percentage / 100)
  let total_cost := cost_price1 + cost_price2
  let total_selling := 2 * selling_price
  let profit_loss := total_selling - total_cost
  (profit_loss / total_cost) * 100

/-- Proof that the trader's overall loss is approximately 1.44% -/
theorem trader_overall_loss :
  ∃ ε > 0, abs (trader_profit_loss_percentage 325475 12 12 + 1.44) < ε :=
sorry

end trader_profit_loss_percentage_trader_overall_loss_l3193_319371


namespace parabola_b_value_l3193_319377

/-- Prove that for a parabola y = 2x^2 + bx + 3 passing through (1, 2) and (-2, -1), b = 11/2 -/
theorem parabola_b_value (b : ℝ) : 
  (2 * (1 : ℝ)^2 + b * 1 + 3 = 2) ∧ 
  (2 * (-2 : ℝ)^2 + b * (-2) + 3 = -1) → 
  b = 11/2 := by
sorry

end parabola_b_value_l3193_319377


namespace brownies_problem_l3193_319332

theorem brownies_problem (total_brownies : ℕ) (tina_per_day : ℕ) (husband_per_day : ℕ) 
  (shared : ℕ) (left : ℕ) :
  total_brownies = 24 →
  tina_per_day = 2 →
  husband_per_day = 1 →
  shared = 4 →
  left = 5 →
  ∃ (days : ℕ), days = 5 ∧ 
    total_brownies = days * (tina_per_day + husband_per_day) + shared + left :=
by sorry

end brownies_problem_l3193_319332


namespace smallest_a_for_minimum_l3193_319366

noncomputable def f (a x : ℝ) : ℝ := -Real.log x / x + Real.exp (a * x - 1)

theorem smallest_a_for_minimum (a : ℝ) : 
  (∀ x > 0, f a x ≥ a) ∧ (∃ x > 0, f a x = a) ↔ a = -Real.exp (-2) :=
sorry

end smallest_a_for_minimum_l3193_319366


namespace simplify_expression_l3193_319353

theorem simplify_expression : 4 * (15 / 7) * (21 / (-45)) = -4 := by
  sorry

end simplify_expression_l3193_319353


namespace pie_to_bar_representation_l3193_319348

-- Define the structure of a pie chart
structure PieChart :=
  (section1 : ℝ)
  (section2 : ℝ)
  (section3 : ℝ)

-- Define the structure of a bar graph
structure BarGraph :=
  (bar1 : ℝ)
  (bar2 : ℝ)
  (bar3 : ℝ)

-- Define the conditions of the pie chart
def validPieChart (p : PieChart) : Prop :=
  p.section1 = p.section2 ∧ p.section3 = p.section1 + p.section2

-- Define the correct bar graph representation
def correctBarGraph (p : PieChart) (b : BarGraph) : Prop :=
  b.bar1 = b.bar2 ∧ b.bar3 = b.bar1 + b.bar2

-- Theorem: For a valid pie chart, there exists a correct bar graph representation
theorem pie_to_bar_representation (p : PieChart) (h : validPieChart p) :
  ∃ b : BarGraph, correctBarGraph p b :=
sorry

end pie_to_bar_representation_l3193_319348


namespace regular_polygon_diagonals_l3193_319329

/-- A regular polygon with exterior angles measuring 60° has 9 diagonals -/
theorem regular_polygon_diagonals :
  ∀ (n : ℕ),
  (360 / n = 60) →  -- Each exterior angle measures 60°
  (n * (n - 3)) / 2 = 9  -- Number of diagonals
  := by sorry

end regular_polygon_diagonals_l3193_319329


namespace tim_total_score_l3193_319324

/-- The score for a single line in Tetris -/
def single_line_score : ℕ := 1000

/-- The score for a Tetris (four lines cleared at once) -/
def tetris_score : ℕ := 8 * single_line_score

/-- Tim's number of single lines cleared -/
def tim_singles : ℕ := 6

/-- Tim's number of Tetrises -/
def tim_tetrises : ℕ := 4

/-- Theorem stating Tim's total score -/
theorem tim_total_score : tim_singles * single_line_score + tim_tetrises * tetris_score = 38000 := by
  sorry

end tim_total_score_l3193_319324


namespace number_of_scooters_l3193_319335

/-- Represents the number of wheels on a vehicle -/
def wheels (vehicle : String) : ℕ :=
  match vehicle with
  | "bicycle" => 2
  | "tricycle" => 3
  | "scooter" => 2
  | _ => 0

/-- The total number of vehicles -/
def total_vehicles : ℕ := 10

/-- The total number of wheels -/
def total_wheels : ℕ := 26

/-- Proves that the number of scooters is 2 -/
theorem number_of_scooters :
  ∃ (b t s : ℕ),
    b + t + s = total_vehicles ∧
    b * wheels "bicycle" + t * wheels "tricycle" + s * wheels "scooter" = total_wheels ∧
    s = 2 := by
  sorry

end number_of_scooters_l3193_319335


namespace pebble_collection_l3193_319376

theorem pebble_collection (n : ℕ) (a : ℕ) (d : ℕ) : 
  n = 15 → a = 1 → d = 1 → (n * (2 * a + (n - 1) * d)) / 2 = 120 := by
  sorry

end pebble_collection_l3193_319376


namespace max_area_rectangle_perimeter_24_l3193_319362

/-- The maximum area of a rectangle with perimeter 24 is 36 -/
theorem max_area_rectangle_perimeter_24 :
  ∀ (length width : ℝ), length > 0 → width > 0 →
  2 * (length + width) = 24 →
  length * width ≤ 36 := by
sorry

end max_area_rectangle_perimeter_24_l3193_319362


namespace task_completion_condition_l3193_319354

/-- Represents the completion of a task given the number of people working in two phases -/
def task_completion (x : ℝ) : Prop :=
  let total_time : ℝ := 40
  let phase1_time : ℝ := 4
  let phase2_time : ℝ := 8
  let phase1_people : ℝ := x
  let phase2_people : ℝ := x + 2
  (phase1_time * phase1_people) / total_time + (phase2_time * phase2_people) / total_time = 1

/-- Theorem stating the condition for task completion -/
theorem task_completion_condition (x : ℝ) :
  task_completion x ↔ 4 * x / 40 + 8 * (x + 2) / 40 = 1 :=
by sorry

end task_completion_condition_l3193_319354


namespace series_sum_minus_eight_l3193_319328

theorem series_sum_minus_eight : 
  (5/3 + 13/9 + 41/27 + 125/81 + 379/243 + 1145/729) - 8 = 950/729 := by
  sorry

end series_sum_minus_eight_l3193_319328


namespace negation_of_proposition_l3193_319344

theorem negation_of_proposition (p : Prop) : 
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 < 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≥ 0) :=
by sorry

end negation_of_proposition_l3193_319344


namespace multiple_of_seven_proposition_l3193_319370

theorem multiple_of_seven_proposition : 
  (∃ k : ℤ, 47 = 7 * k) ∨ (∃ m : ℤ, 49 = 7 * m) := by sorry

end multiple_of_seven_proposition_l3193_319370


namespace bowling_team_weight_l3193_319355

theorem bowling_team_weight (initial_players : ℕ) (initial_avg : ℝ) 
  (new_player1_weight : ℝ) (new_avg : ℝ) :
  initial_players = 7 →
  initial_avg = 103 →
  new_player1_weight = 110 →
  new_avg = 99 →
  ∃ (new_player2_weight : ℝ),
    (initial_players * initial_avg + new_player1_weight + new_player2_weight) / 
    (initial_players + 2) = new_avg ∧
    new_player2_weight = 60 := by
  sorry

end bowling_team_weight_l3193_319355


namespace quadratic_equation_solution_l3193_319319

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x - 1
  ∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end quadratic_equation_solution_l3193_319319


namespace sin_plus_cos_equals_one_l3193_319393

theorem sin_plus_cos_equals_one (x : ℝ) :
  0 ≤ x ∧ x < 2 * Real.pi ∧ Real.sin x + Real.cos x = 1 → x = 0 ∨ x = Real.pi / 2 := by
  sorry

end sin_plus_cos_equals_one_l3193_319393


namespace cube_five_minus_thirteen_equals_square_six_plus_seventysix_l3193_319359

theorem cube_five_minus_thirteen_equals_square_six_plus_seventysix :
  5^3 - 13 = 6^2 + 76 := by
  sorry

end cube_five_minus_thirteen_equals_square_six_plus_seventysix_l3193_319359


namespace tree_height_after_two_years_l3193_319333

/-- Given a tree that triples its height every year and reaches 81 feet after 4 years,
    this function calculates its height after a given number of years. -/
def tree_height (years : ℕ) : ℚ :=
  81 / (3 ^ (4 - years))

/-- Theorem stating that the height of the tree after 2 years is 9 feet. -/
theorem tree_height_after_two_years :
  tree_height 2 = 9 := by sorry

end tree_height_after_two_years_l3193_319333


namespace zephyr_island_population_reaches_capacity_l3193_319357

/-- Represents the population growth on Zephyr Island -/
def zephyr_island_population (initial_year : ℕ) (initial_population : ℕ) (years_passed : ℕ) : ℕ :=
  initial_population * (4 ^ (years_passed / 20))

/-- Represents the maximum capacity of Zephyr Island -/
def zephyr_island_capacity (total_acres : ℕ) (acres_per_person : ℕ) : ℕ :=
  total_acres / acres_per_person

/-- Theorem stating that the population will reach or exceed the maximum capacity in 40 years -/
theorem zephyr_island_population_reaches_capacity :
  let initial_year := 2023
  let initial_population := 500
  let total_acres := 30000
  let acres_per_person := 2
  let years_to_capacity := 40
  zephyr_island_population initial_year initial_population years_to_capacity ≥ 
    zephyr_island_capacity total_acres acres_per_person ∧
  zephyr_island_population initial_year initial_population (years_to_capacity - 20) < 
    zephyr_island_capacity total_acres acres_per_person :=
by
  sorry


end zephyr_island_population_reaches_capacity_l3193_319357


namespace tan_2x_geq_1_solution_set_l3193_319305

theorem tan_2x_geq_1_solution_set :
  {x : ℝ | Real.tan (2 * x) ≥ 1} = {x : ℝ | ∃ k : ℤ, k * Real.pi / 2 + Real.pi / 8 ≤ x ∧ x < k * Real.pi / 2 + Real.pi / 4} :=
by sorry

end tan_2x_geq_1_solution_set_l3193_319305


namespace candy_sampling_percentage_l3193_319331

/-- The percentage of customers caught sampling candy -/
def caught_percent : ℝ := 22

/-- The percentage of candy samplers who are not caught -/
def not_caught_percent : ℝ := 20

/-- The total percentage of customers who sample candy -/
def total_sample_percent : ℝ := 28

/-- Theorem stating that the total percentage of customers who sample candy is 28% -/
theorem candy_sampling_percentage :
  total_sample_percent = caught_percent / (1 - not_caught_percent / 100) :=
by sorry

end candy_sampling_percentage_l3193_319331


namespace volume_complete_octagonal_pyramid_l3193_319347

/-- The volume of a complete pyramid with a regular octagonal base, given the dimensions of its truncated version. -/
theorem volume_complete_octagonal_pyramid 
  (lower_base_side : ℝ) 
  (upper_base_side : ℝ) 
  (truncated_height : ℝ) 
  (h_lower : lower_base_side = 0.4) 
  (h_upper : upper_base_side = 0.3) 
  (h_height : truncated_height = 0.5) : 
  ∃ (volume : ℝ), volume = (16/75) * (Real.sqrt 2 + 1) := by
  sorry

#check volume_complete_octagonal_pyramid

end volume_complete_octagonal_pyramid_l3193_319347


namespace grass_field_width_l3193_319375

/-- The width of a rectangular grass field with specific conditions -/
theorem grass_field_width : ∃ (w : ℝ), w = 40 ∧ w > 0 := by
  -- Define the length of the grass field
  let length : ℝ := 75

  -- Define the width of the path
  let path_width : ℝ := 2.5

  -- Define the cost per square meter of the path
  let cost_per_sqm : ℝ := 2

  -- Define the total cost of the path
  let total_cost : ℝ := 1200

  -- The width w satisfies the equation:
  -- 2 * (80 * (w + 5) - 75 * w) = 1200
  -- where 80 = length + 2 * path_width
  -- and 75 = length

  sorry

end grass_field_width_l3193_319375


namespace geometric_sequence_minimum_l3193_319304

def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_minimum (a : ℕ → ℝ) (h1 : isGeometric a)
  (h2 : ∀ n : ℕ, a n > 0)
  (h3 : ∃ m n : ℕ, Real.sqrt (a m * a n) = 8 * a 1)
  (h4 : a 9 = a 8 + 2 * a 7) :
  (∃ m n : ℕ, 1 / m + 4 / n = 17 / 15) ∧
  (∀ m n : ℕ, 1 / m + 4 / n ≥ 17 / 15) := by
  sorry

end geometric_sequence_minimum_l3193_319304


namespace triangle_side_value_l3193_319306

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem triangle_side_value (A B C : ℝ) (a b c : ℝ) :
  f A = 2 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  a = Real.sqrt 13 :=
by sorry

end triangle_side_value_l3193_319306


namespace problem_solution_l3193_319368

theorem problem_solution (x : ℝ) :
  x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10 →
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 + Real.sqrt (x^4 - 4)) = 841 / 100 :=
by sorry

end problem_solution_l3193_319368


namespace polygon_diagonals_sides_l3193_319360

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A polygon has 33 more diagonals than sides if and only if it has 11 sides -/
theorem polygon_diagonals_sides (n : ℕ) : diagonals n = n + 33 ↔ n = 11 := by
  sorry

end polygon_diagonals_sides_l3193_319360


namespace quadratic_real_root_condition_l3193_319345

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end quadratic_real_root_condition_l3193_319345


namespace extreme_points_range_l3193_319389

/-- The function f(x) = x^2 + a*ln(1+x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * Real.log (1 + x)

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := (2 * x^2 + 2 * x + a) / (1 + x)

theorem extreme_points_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
    f_derivative a x = 0 ∧ 
    f_derivative a y = 0 ∧ 
    (∀ z : ℝ, f_derivative a z = 0 → z = x ∨ z = y)) →
  (0 < a ∧ a < 1/2) :=
sorry

end extreme_points_range_l3193_319389


namespace chef_michel_pies_l3193_319326

/-- Represents the number of pieces a shepherd's pie is cut into -/
def shepherds_pie_pieces : ℕ := 4

/-- Represents the number of pieces a chicken pot pie is cut into -/
def chicken_pot_pie_pieces : ℕ := 5

/-- Represents the number of customers who ordered shepherd's pie slices -/
def shepherds_pie_customers : ℕ := 52

/-- Represents the number of customers who ordered chicken pot pie slices -/
def chicken_pot_pie_customers : ℕ := 80

/-- Calculates the total number of pies sold by Chef Michel -/
def total_pies_sold : ℕ := 
  (shepherds_pie_customers / shepherds_pie_pieces) + 
  (chicken_pot_pie_customers / chicken_pot_pie_pieces)

theorem chef_michel_pies : total_pies_sold = 29 := by
  sorry

end chef_michel_pies_l3193_319326


namespace range_of_a_l3193_319369

open Set Real

theorem range_of_a (p q : Prop) (h : ¬(p ∧ q)) : 
  ∀ a : ℝ, (∀ x ∈ Icc 0 1, a ≥ exp x) = p → 
  (∃ x₀ : ℝ, x₀^2 + 4*x₀ + a = 0) = q → 
  a ∈ Ioi 4 ∪ Iic (exp 1) :=
sorry

end range_of_a_l3193_319369


namespace jane_reading_period_l3193_319339

/-- Represents Jane's reading habits and total pages read --/
structure ReadingHabit where
  morning_pages : ℕ
  evening_pages : ℕ
  total_pages : ℕ

/-- Calculates the number of days Jane reads based on her reading habit --/
def calculate_reading_days (habit : ReadingHabit) : ℚ :=
  habit.total_pages / (habit.morning_pages + habit.evening_pages)

/-- Theorem stating that Jane reads for 7 days --/
theorem jane_reading_period (habit : ReadingHabit) 
  (h1 : habit.morning_pages = 5)
  (h2 : habit.evening_pages = 10)
  (h3 : habit.total_pages = 105) :
  calculate_reading_days habit = 7 := by
  sorry


end jane_reading_period_l3193_319339


namespace drivers_distance_comparison_l3193_319390

/-- Conversion factor from miles to kilometers -/
def mile_to_km : ℝ := 1.60934

/-- Gervais's distance in miles per day -/
def gervais_miles_per_day : ℝ := 315

/-- Number of days Gervais drove -/
def gervais_days : ℕ := 3

/-- Gervais's speed in miles per hour -/
def gervais_speed : ℝ := 60

/-- Henri's total distance in miles -/
def henri_miles : ℝ := 1250

/-- Henri's speed in miles per hour -/
def henri_speed : ℝ := 50

/-- Madeleine's distance in miles per day -/
def madeleine_miles_per_day : ℝ := 100

/-- Number of days Madeleine drove -/
def madeleine_days : ℕ := 5

/-- Madeleine's speed in miles per hour -/
def madeleine_speed : ℝ := 40

/-- Calculate total distance driven by all three drivers in kilometers -/
def total_distance : ℝ :=
  (gervais_miles_per_day * gervais_days * mile_to_km) +
  (henri_miles * mile_to_km) +
  (madeleine_miles_per_day * madeleine_days * mile_to_km)

/-- Calculate Henri's distance in kilometers -/
def henri_distance : ℝ := henri_miles * mile_to_km

theorem drivers_distance_comparison :
  total_distance = 4337.16905 ∧
  henri_distance = 2011.675 ∧
  henri_distance > gervais_miles_per_day * gervais_days * mile_to_km ∧
  henri_distance > madeleine_miles_per_day * madeleine_days * mile_to_km :=
by sorry

end drivers_distance_comparison_l3193_319390


namespace negation_equivalence_l3193_319356

-- Define the original proposition
def original_proposition : Prop := ∃ x : ℝ, Real.exp x - x - 2 ≤ 0

-- Define the negation of the proposition
def negation_proposition : Prop := ∀ x : ℝ, Real.exp x - x - 2 > 0

-- Theorem stating the equivalence between the negation of the original proposition
-- and the negation_proposition
theorem negation_equivalence : 
  (¬ original_proposition) ↔ negation_proposition :=
sorry

end negation_equivalence_l3193_319356


namespace equality_of_exponential_equation_l3193_319399

theorem equality_of_exponential_equation (a b : ℝ) : 
  0 < a → 0 < b → a < 1 → a^b = b^a → a = b := by sorry

end equality_of_exponential_equation_l3193_319399


namespace paperback_copies_sold_l3193_319397

theorem paperback_copies_sold (hardback_copies : ℕ) (total_copies : ℕ) : 
  hardback_copies = 36000 →
  total_copies = 440000 →
  ∃ paperback_copies : ℕ, 
    paperback_copies = 9 * hardback_copies ∧
    hardback_copies + paperback_copies = total_copies ∧
    paperback_copies = 360000 := by
  sorry

end paperback_copies_sold_l3193_319397


namespace largest_prime_factor_of_3136_l3193_319334

theorem largest_prime_factor_of_3136 (p : Nat) : 
  Nat.Prime p ∧ p ∣ 3136 → p ≤ 7 :=
by sorry

end largest_prime_factor_of_3136_l3193_319334


namespace smallest_k_remainder_l3193_319388

theorem smallest_k_remainder (k : ℕ) : 
  k > 0 ∧ 
  k % 5 = 2 ∧ 
  k % 6 = 5 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 5 = 2 ∧ m % 6 = 5 → k ≤ m) → 
  k % 7 = 3 := by
sorry

end smallest_k_remainder_l3193_319388


namespace percentage_spent_is_80_percent_l3193_319308

-- Define the costs and money amounts
def cheeseburger_cost : ℚ := 3
def milkshake_cost : ℚ := 5
def cheese_fries_cost : ℚ := 8
def jim_money : ℚ := 20
def cousin_money : ℚ := 10

-- Define the total cost of the meal
def total_cost : ℚ := 2 * cheeseburger_cost + 2 * milkshake_cost + cheese_fries_cost

-- Define the combined money
def combined_money : ℚ := jim_money + cousin_money

-- Theorem to prove
theorem percentage_spent_is_80_percent :
  (total_cost / combined_money) * 100 = 80 := by
  sorry

end percentage_spent_is_80_percent_l3193_319308


namespace annual_interest_income_l3193_319340

/-- Calculates the annual interest income from two municipal bonds -/
theorem annual_interest_income
  (total_investment : ℝ)
  (rate1 rate2 : ℝ)
  (investment1 : ℝ)
  (h1 : total_investment = 32000)
  (h2 : rate1 = 0.0575)
  (h3 : rate2 = 0.0625)
  (h4 : investment1 = 20000)
  (h5 : investment1 < total_investment) :
  investment1 * rate1 + (total_investment - investment1) * rate2 = 1900 := by
  sorry

end annual_interest_income_l3193_319340


namespace chen_pushups_l3193_319310

/-- The number of push-ups done by Chen -/
def chen : ℕ := sorry

/-- The number of push-ups done by Ruan -/
def ruan : ℕ := sorry

/-- The number of push-ups done by Lu -/
def lu : ℕ := sorry

/-- The number of push-ups done by Tao -/
def tao : ℕ := sorry

/-- The number of push-ups done by Yang -/
def yang : ℕ := sorry

/-- Chen, Lu, and Yang together averaged 40 push-ups per person -/
axiom condition1 : chen + lu + yang = 40 * 3

/-- Ruan, Tao, and Chen together averaged 28 push-ups per person -/
axiom condition2 : ruan + tao + chen = 28 * 3

/-- Ruan, Lu, Tao, and Yang together averaged 33 push-ups per person -/
axiom condition3 : ruan + lu + tao + yang = 33 * 4

theorem chen_pushups : chen = 36 := by
  sorry

end chen_pushups_l3193_319310


namespace second_derivative_parametric_function_l3193_319322

/-- The second-order derivative of a parametrically defined function -/
theorem second_derivative_parametric_function (t : ℝ) (h : t ≠ 0) :
  let x := 1 / t
  let y := 1 / (1 + t^2)
  let y''_xx := (2 * (t^2 - 3) * t^4) / ((1 + t^2)^3)
  ∃ (d2y_dx2 : ℝ), d2y_dx2 = y''_xx := by
  sorry

end second_derivative_parametric_function_l3193_319322
