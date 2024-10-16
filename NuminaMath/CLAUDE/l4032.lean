import Mathlib

namespace NUMINAMATH_CALUDE_area_difference_l4032_403267

/-- The difference in area between a square and a rectangle -/
theorem area_difference (square_side : ℝ) (rect_length rect_width : ℝ) : 
  square_side = 5 → rect_length = 3 → rect_width = 6 → 
  square_side * square_side - rect_length * rect_width = 7 := by
  sorry

#check area_difference

end NUMINAMATH_CALUDE_area_difference_l4032_403267


namespace NUMINAMATH_CALUDE_money_left_is_five_l4032_403208

/-- The cost of the gift in dollars -/
def gift_cost : ℕ := 250

/-- Erika's savings in dollars -/
def erika_savings : ℕ := 155

/-- The cost of the cake in dollars -/
def cake_cost : ℕ := 25

/-- Rick's savings in dollars, defined as half of the gift cost -/
def rick_savings : ℕ := gift_cost / 2

/-- The total savings of Erika and Rick -/
def total_savings : ℕ := erika_savings + rick_savings

/-- The total cost of the gift and cake -/
def total_cost : ℕ := gift_cost + cake_cost

/-- The amount of money left after buying the gift and cake -/
def money_left : ℕ := total_savings - total_cost

theorem money_left_is_five : money_left = 5 := by
  sorry

end NUMINAMATH_CALUDE_money_left_is_five_l4032_403208


namespace NUMINAMATH_CALUDE_first_term_of_ap_l4032_403288

def arithmetic_progression (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_progression (a₁ d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ) * d)

theorem first_term_of_ap (a₁ d : ℚ) :
  sum_arithmetic_progression a₁ d 22 = 1045 ∧
  sum_arithmetic_progression (arithmetic_progression a₁ d 23) d 22 = 2013 →
  a₁ = 53 / 2 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_ap_l4032_403288


namespace NUMINAMATH_CALUDE_noah_age_in_ten_years_l4032_403221

/-- Calculates Noah's age after a given number of years -/
def noah_age_after (joe_age : ℕ) (years_passed : ℕ) : ℕ :=
  2 * joe_age + years_passed

/-- Proves that Noah will be 22 years old after 10 years, given the initial conditions -/
theorem noah_age_in_ten_years (joe_age : ℕ) (h : joe_age = 6) :
  noah_age_after joe_age 10 = 22 := by
  sorry

end NUMINAMATH_CALUDE_noah_age_in_ten_years_l4032_403221


namespace NUMINAMATH_CALUDE_floor_power_equality_l4032_403276

theorem floor_power_equality (a b : ℝ) (h : a > 0) (h' : b > 0)
  (h_infinite : ∃ᶠ k : ℕ in atTop, ⌊a^k⌋ + ⌊b^k⌋ = ⌊a⌋^k + ⌊b⌋^k) :
  ⌊a^2014⌋ + ⌊b^2014⌋ = ⌊a⌋^2014 + ⌊b⌋^2014 := by
sorry

end NUMINAMATH_CALUDE_floor_power_equality_l4032_403276


namespace NUMINAMATH_CALUDE_geometric_series_product_l4032_403212

theorem geometric_series_product (y : ℝ) : y = 9 ↔ 
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/y)^n := by sorry

end NUMINAMATH_CALUDE_geometric_series_product_l4032_403212


namespace NUMINAMATH_CALUDE_b_initial_investment_l4032_403264

/-- Represents the business investment problem --/
structure BusinessInvestment where
  a_initial : ℕ  -- A's initial investment
  total_profit : ℕ  -- Total profit at the end of the year
  a_profit : ℕ  -- A's share of the profit
  a_withdraw : ℕ  -- Amount A withdraws after 8 months
  b_advance : ℕ  -- Amount B advances after 8 months

/-- Calculates B's initial investment given the business conditions --/
def calculate_b_initial (bi : BusinessInvestment) : ℕ :=
  sorry

/-- Theorem stating that B's initial investment is 4000 given the problem conditions --/
theorem b_initial_investment (bi : BusinessInvestment) 
  (h1 : bi.a_initial = 3000)
  (h2 : bi.total_profit = 630)
  (h3 : bi.a_profit = 240)
  (h4 : bi.a_withdraw = 1000)
  (h5 : bi.b_advance = 1000) :
  calculate_b_initial bi = 4000 := by
  sorry

end NUMINAMATH_CALUDE_b_initial_investment_l4032_403264


namespace NUMINAMATH_CALUDE_total_investment_equals_eight_thousand_l4032_403287

/-- Represents an investment account with a given balance and interest rate. -/
structure Account where
  balance : ℝ
  interestRate : ℝ

/-- Calculates the total investment given two accounts. -/
def totalInvestment (account1 account2 : Account) : ℝ :=
  account1.balance + account2.balance

/-- Theorem: The total investment in two accounts with $4,000 each is $8,000. -/
theorem total_investment_equals_eight_thousand 
  (account1 account2 : Account)
  (h1 : account1.balance = 4000)
  (h2 : account2.balance = 4000) :
  totalInvestment account1 account2 = 8000 := by
  sorry

#check total_investment_equals_eight_thousand

end NUMINAMATH_CALUDE_total_investment_equals_eight_thousand_l4032_403287


namespace NUMINAMATH_CALUDE_math_team_selection_ways_l4032_403219

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem math_team_selection_ways :
  let boys := 7
  let girls := 9
  let team_boys := 3
  let team_girls := 3
  (choose boys team_boys) * (choose girls team_girls) = 2940 := by
sorry

end NUMINAMATH_CALUDE_math_team_selection_ways_l4032_403219


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_term_l4032_403236

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the exponent of x in the general term
def exponent (r : ℕ) : ℚ := 9 - (3 / 2) * r

theorem coefficient_of_x_cubed_term :
  ∃ (r : ℕ), exponent r = 3 ∧ binomial 9 r = 126 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_term_l4032_403236


namespace NUMINAMATH_CALUDE_lenny_pens_percentage_l4032_403205

theorem lenny_pens_percentage (total_boxes : ℕ) (pens_per_box : ℕ) (remaining_pens : ℕ) : 
  total_boxes = 20 →
  pens_per_box = 5 →
  remaining_pens = 45 →
  ∃ (percentage : ℚ),
    percentage = 40 ∧
    (3/4 : ℚ) * ((total_boxes * pens_per_box : ℚ) - percentage) / 100 * (total_boxes * pens_per_box) = remaining_pens :=
by sorry

end NUMINAMATH_CALUDE_lenny_pens_percentage_l4032_403205


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4032_403299

/-- Given a quadratic function f(x) = x^2 + bx + c, 
    prove that c < 0 is sufficient but not necessary for f(x) < 0 to have a real solution -/
theorem sufficient_not_necessary_condition (b c : ℝ) :
  (∀ x, (x : ℝ)^2 + b*x + c < 0 → c < 0) ∧
  ¬(∀ b c : ℝ, (∃ x, (x : ℝ)^2 + b*x + c < 0) → c < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4032_403299


namespace NUMINAMATH_CALUDE_set_equation_solution_l4032_403289

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 1 = 0}
def B (a b : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*a*x + b = 0}

-- State the theorem
theorem set_equation_solution (a b : ℝ) : 
  B a b ≠ ∅ ∧ B a b ⊆ A → 
  ((a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) := by
  sorry

end NUMINAMATH_CALUDE_set_equation_solution_l4032_403289


namespace NUMINAMATH_CALUDE_car_trip_duration_l4032_403291

theorem car_trip_duration : ∀ (x : ℝ),
  let d1 : ℝ := 70 * 4  -- distance covered in first segment
  let d2 : ℝ := 60 * 5  -- distance covered in second segment
  let d3 : ℝ := 50 * x  -- distance covered in third segment
  let total_distance : ℝ := d1 + d2 + d3
  let total_time : ℝ := 4 + 5 + x
  let average_speed : ℝ := 58
  average_speed = total_distance / total_time →
  total_time = 16.25 :=
by sorry


end NUMINAMATH_CALUDE_car_trip_duration_l4032_403291


namespace NUMINAMATH_CALUDE_company_survey_l4032_403278

theorem company_survey (total employees_with_tool employees_with_training employees_with_both : ℕ)
  (h_total : total = 150)
  (h_tool : employees_with_tool = 90)
  (h_training : employees_with_training = 60)
  (h_both : employees_with_both = 30) :
  (↑(total - (employees_with_tool + employees_with_training - employees_with_both)) / ↑total) * 100 = 20 :=
by sorry

end NUMINAMATH_CALUDE_company_survey_l4032_403278


namespace NUMINAMATH_CALUDE_remainder_zero_l4032_403295

theorem remainder_zero : (88144 * 88145 + 88146 + 88147 + 88148 + 88149 + 88150) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_zero_l4032_403295


namespace NUMINAMATH_CALUDE_marks_fruit_consumption_l4032_403241

/-- Given the conditions of Mark's fruit consumption, prove that he ate 5 pieces in the first four days --/
theorem marks_fruit_consumption
  (total : ℕ)
  (kept_for_next_week : ℕ)
  (brought_on_friday : ℕ)
  (h1 : total = 10)
  (h2 : kept_for_next_week = 2)
  (h3 : brought_on_friday = 3) :
  total - kept_for_next_week - brought_on_friday = 5 := by
  sorry

end NUMINAMATH_CALUDE_marks_fruit_consumption_l4032_403241


namespace NUMINAMATH_CALUDE_chemical_mixture_problem_l4032_403283

theorem chemical_mixture_problem (original_conc : ℝ) (final_conc : ℝ) (replaced_portion : ℝ) 
  (h1 : original_conc = 0.9)
  (h2 : final_conc = 0.4)
  (h3 : replaced_portion = 0.7142857142857143) :
  let replacement_conc := (final_conc - original_conc * (1 - replaced_portion)) / replaced_portion
  replacement_conc = 0.2 := by
sorry

end NUMINAMATH_CALUDE_chemical_mixture_problem_l4032_403283


namespace NUMINAMATH_CALUDE_problem_statement_l4032_403270

theorem problem_statement (x y : ℝ) : 
  (|x - y| > x) → (x + y > 0) → (x > 0 ∧ y > 0) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l4032_403270


namespace NUMINAMATH_CALUDE_natural_solutions_3x_plus_4y_eq_12_l4032_403269

theorem natural_solutions_3x_plus_4y_eq_12 :
  {(x, y) : ℕ × ℕ | 3 * x + 4 * y = 12} = {(4, 0), (0, 3)} := by
  sorry

end NUMINAMATH_CALUDE_natural_solutions_3x_plus_4y_eq_12_l4032_403269


namespace NUMINAMATH_CALUDE_shoe_size_for_given_length_xiao_gang_shoe_size_l4032_403237

/-- A linear function representing the relationship between shoe size and foot length. -/
def shoe_size_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

/-- Theorem stating the shoe size for a given foot length based on the given conditions. -/
theorem shoe_size_for_given_length (k b : ℝ) :
  shoe_size_function k b 23 = 36 →
  shoe_size_function k b 26 = 42 →
  shoe_size_function k b 24.5 = 39 := by
  sorry

/-- Corollary: Xiao Gang's shoe size -/
theorem xiao_gang_shoe_size (k b : ℝ) :
  shoe_size_function k b 23 = 36 →
  shoe_size_function k b 26 = 42 →
  ∃ y : ℝ, y = shoe_size_function k b 24.5 ∧ y = 39 := by
  sorry

end NUMINAMATH_CALUDE_shoe_size_for_given_length_xiao_gang_shoe_size_l4032_403237


namespace NUMINAMATH_CALUDE_expression_undefined_l4032_403229

theorem expression_undefined (a : ℝ) : 
  ¬∃x, x = (a + 3) / (a^2 - 9*a + 20) ↔ a = 4 ∨ a = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_undefined_l4032_403229


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l4032_403285

-- Equation 1
theorem solve_equation_one : 
  ∃ x : ℝ, (3 * x - 4 = -2 * (x - 1)) ∧ (x = 1.2) := by sorry

-- Equation 2
theorem solve_equation_two :
  ∃ x : ℝ, (1 + (2 * x + 1) / 3 = (3 * x - 2) / 2) ∧ (x = 14 / 5) := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l4032_403285


namespace NUMINAMATH_CALUDE_solution_set_f_positive_range_of_m_l4032_403247

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for part I
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = Set.Iio (-1/3) ∪ Set.Ioi 3 := by sorry

-- Theorem for part II
theorem range_of_m (h : ∃ x₀ : ℝ, f x₀ + 2*m^2 < 4*m) :
  m ∈ Set.Ioo (-1/2) (5/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_range_of_m_l4032_403247


namespace NUMINAMATH_CALUDE_lunch_with_tip_theorem_l4032_403222

/-- Calculates the total amount spent on a meal including tip -/
def total_amount_spent (lunch_cost : ℚ) (tip_rate : ℚ) : ℚ :=
  lunch_cost * (1 + tip_rate)

/-- Theorem stating that for a $50.20 lunch with 20% tip, the total spent is $60.24 -/
theorem lunch_with_tip_theorem :
  total_amount_spent 50.20 0.20 = 60.24 := by
  sorry

end NUMINAMATH_CALUDE_lunch_with_tip_theorem_l4032_403222


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_two_distinct_real_roots_l4032_403265

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  a * x^2 + b * x + c = 0 ↔ 
    (discriminant > 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) ∧
    (discriminant = 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0) ∧
    (discriminant < 0 → ¬∃ x : ℝ, a * x^2 + b * x + c = 0) :=
by sorry

theorem two_distinct_real_roots :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 2*x1 - 1 = 0 ∧ x2^2 - 2*x2 - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_two_distinct_real_roots_l4032_403265


namespace NUMINAMATH_CALUDE_prob_same_length_is_17_35_l4032_403280

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of segments (sides and diagonals) in a regular hexagon -/
def total_segments : ℕ := num_sides + num_diagonals

/-- The number of pairs of segments with the same length -/
def same_length_pairs : ℕ := (num_sides.choose 2) + (num_diagonals.choose 2)

/-- The total number of possible pairs of segments -/
def total_pairs : ℕ := total_segments.choose 2

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ := same_length_pairs / total_pairs

theorem prob_same_length_is_17_35 : prob_same_length = 17 / 35 := by sorry

end NUMINAMATH_CALUDE_prob_same_length_is_17_35_l4032_403280


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l4032_403297

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 4620 → 
  Nat.gcd a b = 21 → 
  a = 210 → 
  b = 462 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l4032_403297


namespace NUMINAMATH_CALUDE_sandwich_cost_is_181_l4032_403272

/-- The cost in cents for Joe to make a deluxe ham and cheese sandwich -/
def sandwich_cost : ℕ :=
  let bread_cost : ℕ := 15 -- Cost of one slice of bread in cents
  let ham_cost : ℕ := 25 -- Cost of one slice of ham in cents
  let cheese_cost : ℕ := 35 -- Cost of one slice of cheese in cents
  let mayo_cost : ℕ := 10 -- Cost of one tablespoon of mayonnaise in cents
  let lettuce_cost : ℕ := 5 -- Cost of one lettuce leaf in cents
  let tomato_cost : ℕ := 8 -- Cost of one tomato slice in cents
  
  let bread_slices : ℕ := 2 -- Number of bread slices used
  let ham_slices : ℕ := 2 -- Number of ham slices used
  let cheese_slices : ℕ := 2 -- Number of cheese slices used
  let mayo_tbsp : ℕ := 1 -- Number of tablespoons of mayonnaise used
  let lettuce_leaves : ℕ := 1 -- Number of lettuce leaves used
  let tomato_slices : ℕ := 2 -- Number of tomato slices used
  
  bread_cost * bread_slices +
  ham_cost * ham_slices +
  cheese_cost * cheese_slices +
  mayo_cost * mayo_tbsp +
  lettuce_cost * lettuce_leaves +
  tomato_cost * tomato_slices

theorem sandwich_cost_is_181 : sandwich_cost = 181 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_is_181_l4032_403272


namespace NUMINAMATH_CALUDE_initial_amount_is_100_l4032_403293

/-- The amount of money Jasmine spent on fruits -/
def spent_on_fruits : ℝ := 15

/-- The amount of money Jasmine had left to spend -/
def money_left : ℝ := 85

/-- The initial amount of money Jasmine's mom gave her -/
def initial_amount : ℝ := spent_on_fruits + money_left

/-- Theorem stating that the initial amount of money Jasmine's mom gave her is $100.00 -/
theorem initial_amount_is_100 : initial_amount = 100 := by sorry

end NUMINAMATH_CALUDE_initial_amount_is_100_l4032_403293


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l4032_403277

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    (n ≥ 100 ∧ n < 1000) ∧ 
    (70 * n) % 350 = 210 ∧
    (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (70 * m) % 350 = 210 → n ≤ m) ∧
    n = 103 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l4032_403277


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l4032_403203

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem f_strictly_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < exp 1 → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l4032_403203


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_parallel_line_l4032_403259

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the axioms
variable (different_lines : ∀ a b l : Line, a ≠ b ∧ b ≠ l ∧ a ≠ l)
variable (non_coincident_planes : ∀ α β : Plane, α ≠ β)

-- State the theorem
theorem line_perpendicular_to_plane_and_parallel_line 
  (a b l : Line) (α : Plane) :
  parallel a b → perpendicular_line_plane l α → perpendicular_line_line l b :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_parallel_line_l4032_403259


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l4032_403201

theorem fraction_sum_equality : (3 : ℚ) / 8 - 5 / 6 + 9 / 4 = 43 / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l4032_403201


namespace NUMINAMATH_CALUDE_book_price_theorem_l4032_403248

/-- The price of a book on Monday when prices are 10% more than normal -/
def monday_price : ℚ := 5.50

/-- The normal price increase factor on Monday -/
def monday_factor : ℚ := 1.10

/-- The normal price decrease factor on Friday -/
def friday_factor : ℚ := 0.90

/-- The price of the book on Friday -/
def friday_price : ℚ := monday_price / monday_factor * friday_factor

theorem book_price_theorem :
  friday_price = 4.50 := by sorry

end NUMINAMATH_CALUDE_book_price_theorem_l4032_403248


namespace NUMINAMATH_CALUDE_total_amount_paid_l4032_403249

def apple_quantity : ℕ := 8
def apple_rate : ℕ := 70
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 70

theorem total_amount_paid : 
  apple_quantity * apple_rate + mango_quantity * mango_rate = 1190 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l4032_403249


namespace NUMINAMATH_CALUDE_country_club_members_l4032_403282

def initial_fee : ℕ := 4000
def monthly_cost : ℕ := 1000
def john_payment : ℕ := 32000

theorem country_club_members : 
  ∀ (F : ℕ), 
    (F + 1) * (initial_fee + 12 * monthly_cost) / 2 = john_payment → 
    F = 3 :=
by sorry

end NUMINAMATH_CALUDE_country_club_members_l4032_403282


namespace NUMINAMATH_CALUDE_k_value_l4032_403214

def length (k : ℕ) : ℕ := sorry

theorem k_value (k : ℕ) (h1 : k > 1) (h2 : length k = 4) (h3 : k = 2 * 2 * 2 * 3) : k = 24 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l4032_403214


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l4032_403253

def circle_radius : ℝ := 6

theorem rectangle_longer_side (rectangle_area rectangle_shorter_side rectangle_longer_side : ℝ) : 
  rectangle_area = 3 * (π * circle_radius^2) →
  rectangle_shorter_side = 2 * circle_radius →
  rectangle_area = rectangle_shorter_side * rectangle_longer_side →
  rectangle_longer_side = 9 * π := by
sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l4032_403253


namespace NUMINAMATH_CALUDE_phantom_needs_126_more_l4032_403242

/-- The amount of additional money Phantom needs to buy printer inks -/
def additional_money_needed (initial_money : ℕ) 
  (black_price red_price yellow_price blue_price : ℕ) 
  (black_quantity red_quantity yellow_quantity blue_quantity : ℕ) : ℕ :=
  let total_cost := black_price * black_quantity + 
                    red_price * red_quantity + 
                    yellow_price * yellow_quantity + 
                    blue_price * blue_quantity
  total_cost - initial_money

/-- Theorem stating that Phantom needs $126 more to buy the printer inks -/
theorem phantom_needs_126_more : 
  additional_money_needed 50 12 16 14 17 3 4 3 2 = 126 := by
  sorry

end NUMINAMATH_CALUDE_phantom_needs_126_more_l4032_403242


namespace NUMINAMATH_CALUDE_integral_equation_solution_l4032_403243

theorem integral_equation_solution (k : ℝ) : 
  (∫ x in (0:ℝ)..2, (3 * x^2 + k)) = 16 → k = 4 := by
sorry

end NUMINAMATH_CALUDE_integral_equation_solution_l4032_403243


namespace NUMINAMATH_CALUDE_simplify_polynomial_l4032_403254

theorem simplify_polynomial (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l4032_403254


namespace NUMINAMATH_CALUDE_sqrt_equation_solvability_l4032_403262

theorem sqrt_equation_solvability (a : ℝ) :
  (∃ x : ℝ, Real.sqrt x - Real.sqrt (x - a) = 2) ↔ a ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solvability_l4032_403262


namespace NUMINAMATH_CALUDE_smallest_maximizer_of_g_l4032_403233

/-- Sum of all positive divisors of n -/
def σ (n : ℕ) : ℕ := sorry

/-- Function g(n) = σ(n) / n -/
def g (n : ℕ) : ℚ := (σ n : ℚ) / n

/-- Theorem stating that 6 is the smallest N maximizing g(n) for 1 ≤ n ≤ 100 -/
theorem smallest_maximizer_of_g :
  ∃ (N : ℕ), N = 6 ∧ 
  (∀ n : ℕ, 1 ≤ n → n ≤ 100 → n ≠ N → g n < g N) ∧
  (∀ m : ℕ, 1 ≤ m → m < N → ∃ k : ℕ, 1 ≤ k ∧ k ≤ 100 ∧ k ≠ m ∧ g m ≤ g k) :=
sorry

end NUMINAMATH_CALUDE_smallest_maximizer_of_g_l4032_403233


namespace NUMINAMATH_CALUDE_ellipse_equation_l4032_403290

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given an ellipse E with specific properties, prove its equation -/
theorem ellipse_equation (E : Ellipse) (F A B M : Point) :
  E.a > E.b ∧ E.b > 0 ∧  -- a > b > 0
  F = ⟨3, 0⟩ ∧  -- Right focus at F(3,0)
  (A.y - F.y) / (A.x - F.x) = 1/2 ∧  -- Line through F with slope 1/2
  (B.y - F.y) / (B.x - F.x) = 1/2 ∧  -- intersects E at A and B
  M = ⟨1, -1⟩ ∧  -- Midpoint of AB is (1,-1)
  M.x = (A.x + B.x) / 2 ∧
  M.y = (A.y + B.y) / 2 ∧
  (A.x^2 / E.a^2) + (A.y^2 / E.b^2) = 1 ∧  -- A and B lie on the ellipse
  (B.x^2 / E.a^2) + (B.y^2 / E.b^2) = 1 →
  E.a^2 = 18 ∧ E.b^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4032_403290


namespace NUMINAMATH_CALUDE_car_cleaning_time_l4032_403210

theorem car_cleaning_time (outside_time : ℕ) (inside_time : ℕ) : 
  outside_time = 80 →
  inside_time = outside_time / 4 →
  outside_time + inside_time = 100 :=
by sorry

end NUMINAMATH_CALUDE_car_cleaning_time_l4032_403210


namespace NUMINAMATH_CALUDE_factorization_equality_l4032_403260

theorem factorization_equality (a b : ℝ) : a * b^2 - 4 * a = a * (b + 2) * (b - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4032_403260


namespace NUMINAMATH_CALUDE_octal_to_decimal_conversion_coral_age_conversion_l4032_403263

-- Define the octal age
def octal_age : ℕ := 753

-- Define the decimal age
def decimal_age : ℕ := 491

-- Theorem to prove the equivalence
theorem octal_to_decimal_conversion :
  (3 * 8^0 + 5 * 8^1 + 7 * 8^2 : ℕ) = decimal_age :=
by sorry

-- Theorem to prove that octal_age in decimal is equal to decimal_age
theorem coral_age_conversion :
  octal_age.digits 8 = [3, 5, 7] ∧
  (3 * 8^0 + 5 * 8^1 + 7 * 8^2 : ℕ) = decimal_age :=
by sorry

end NUMINAMATH_CALUDE_octal_to_decimal_conversion_coral_age_conversion_l4032_403263


namespace NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l4032_403275

theorem circle_radius_from_longest_chord (c : ℝ) (h : c > 0) :
  (∃ (r : ℝ), r > 0 ∧ c = 24 ∧ c = 2 * r) → c / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l4032_403275


namespace NUMINAMATH_CALUDE_num_teachers_at_king_middle_school_l4032_403239

/-- The number of students at King Middle School -/
def num_students : ℕ := 1500

/-- The number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- The number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- The number of students in each class -/
def students_per_class : ℕ := 25

/-- The number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- Theorem: The number of teachers at King Middle School is 72 -/
theorem num_teachers_at_king_middle_school : 
  (num_students * classes_per_student) / (students_per_class * classes_per_teacher) = 72 :=
by sorry

end NUMINAMATH_CALUDE_num_teachers_at_king_middle_school_l4032_403239


namespace NUMINAMATH_CALUDE_largest_fraction_l4032_403234

theorem largest_fraction :
  let fractions := [2/5, 3/7, 4/9, 5/11, 6/13]
  ∀ x ∈ fractions, (6/13 : ℚ) ≥ x := by
sorry

end NUMINAMATH_CALUDE_largest_fraction_l4032_403234


namespace NUMINAMATH_CALUDE_equation_classification_l4032_403245

def equation (m : ℝ) (x : ℝ) : ℝ := (m^2 - 1) * x^2 + (m + 1) * x + (m - 2)

theorem equation_classification (m : ℝ) :
  (∀ x, equation m x = 0 → (m^2 - 1 ≠ 0 ↔ m ≠ 1 ∧ m ≠ -1)) ∧
  (∀ x, equation m x = 0 → (m^2 - 1 = 0 ∧ m + 1 ≠ 0 ↔ m = 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_classification_l4032_403245


namespace NUMINAMATH_CALUDE_two_letter_language_max_words_l4032_403204

/-- A language with two letters and specific word formation rules -/
structure TwoLetterLanguage where
  alphabet : Finset Char
  max_word_length : ℕ
  is_valid_word : List Char → Prop
  no_concatenation : ∀ (w1 w2 : List Char), is_valid_word w1 → is_valid_word w2 → ¬is_valid_word (w1 ++ w2)

/-- The maximum number of words in the specific two-letter language -/
def max_word_count (L : TwoLetterLanguage) : ℕ := 16056

/-- Theorem stating the maximum number of words in the specific two-letter language -/
theorem two_letter_language_max_words (L : TwoLetterLanguage) 
  (h1 : L.alphabet.card = 2)
  (h2 : L.max_word_length = 13)
  : max_word_count L = 16056 := by
  sorry

end NUMINAMATH_CALUDE_two_letter_language_max_words_l4032_403204


namespace NUMINAMATH_CALUDE_kim_average_increase_l4032_403296

def kim_scores : List ℝ := [92, 85, 90, 95]

theorem kim_average_increase :
  let initial_avg := (kim_scores.take 3).sum / 3
  let new_avg := kim_scores.sum / 4
  new_avg - initial_avg = 1.5 := by sorry

end NUMINAMATH_CALUDE_kim_average_increase_l4032_403296


namespace NUMINAMATH_CALUDE_triangle_side_e_theorem_l4032_403231

noncomputable def triangle_side_e (D E : ℝ) (d f : ℝ) : ℝ :=
  let sinD := Real.sqrt ((8 + Real.sqrt (32 / 3)) / 8)
  let sin5D := 5 * sinD - 20 * sinD^3 + 16 * sinD^5
  (36 * sin5D) / sinD

theorem triangle_side_e_theorem (D E : ℝ) (d f : ℝ) :
  E = 4 * D → d = 36 → f = 60 →
  triangle_side_e D E d f = (36 * Real.sin (5 * D)) / Real.sqrt (Real.sin D)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_e_theorem_l4032_403231


namespace NUMINAMATH_CALUDE_non_congruent_triangles_count_l4032_403213

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a set of points -/
def PointSet : Type := List Point

/-- The set of nine points as described in the problem -/
def ninePoints : PointSet :=
  [
    {x := 0, y := 0}, {x := 1, y := 0}, {x := 2, y := 0},
    {x := 0, y := 0}, {x := 1, y := 1}, {x := 2, y := 2},
    {x := 0, y := 0}, {x := 0.5, y := 1}, {x := 1, y := 2}
  ]

/-- Checks if three points form a non-congruent triangle with respect to a set of triangles -/
def isNonCongruentTriangle (p1 p2 p3 : Point) (triangles : List (Point × Point × Point)) : Bool :=
  sorry

/-- Counts the number of non-congruent triangles that can be formed from a set of points -/
def countNonCongruentTriangles (points : PointSet) : Nat :=
  sorry

/-- The main theorem stating that the number of non-congruent triangles is 5 -/
theorem non_congruent_triangles_count :
  countNonCongruentTriangles ninePoints = 5 :=
sorry

end NUMINAMATH_CALUDE_non_congruent_triangles_count_l4032_403213


namespace NUMINAMATH_CALUDE_angle_value_in_triangle_l4032_403216

/-- Given a triangle ABC where ∠ABC = 120°, and two angles are 3y° and y°, prove that y = 30 -/
theorem angle_value_in_triangle (y : ℝ) : 
  (3 * y + y = 120) → y = 30 := by sorry

end NUMINAMATH_CALUDE_angle_value_in_triangle_l4032_403216


namespace NUMINAMATH_CALUDE_second_test_score_proof_l4032_403258

def first_test_score : ℝ := 78
def new_average : ℝ := 81

theorem second_test_score_proof :
  ∃ (second_score : ℝ), (first_test_score + second_score) / 2 = new_average ∧ second_score = 84 :=
by sorry

end NUMINAMATH_CALUDE_second_test_score_proof_l4032_403258


namespace NUMINAMATH_CALUDE_quadratic_with_irrational_root_l4032_403200

theorem quadratic_with_irrational_root :
  ∃ (a b c : ℚ), a ≠ 0 ∧
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3) ∧
  a = 1 ∧ b = 6 ∧ c = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_with_irrational_root_l4032_403200


namespace NUMINAMATH_CALUDE_system_solvability_l4032_403220

-- Define the system of equations
def system (x y p : ℝ) : Prop :=
  (x - p)^2 = 16 * (y - 3 + p) ∧
  y^2 + ((x - 3) / (|x| - 3))^2 = 1 ∧
  |x| ≠ 3

-- Define the set of valid p values
def valid_p_set : Set ℝ :=
  {p | (3 < p ∧ p ≤ 4) ∨ (12 ≤ p ∧ p < 19) ∨ (p > 19)}

-- Theorem statement
theorem system_solvability (p : ℝ) :
  (∃ x y, system x y p) ↔ p ∈ valid_p_set :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l4032_403220


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4032_403215

theorem quadratic_inequality (x : ℝ) : 9 * x^2 + 6 * x - 8 > 0 ↔ x < -4/3 ∨ x > 2/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4032_403215


namespace NUMINAMATH_CALUDE_janet_pages_per_day_l4032_403227

/-- Prove that Janet reads 80 pages a day given the conditions -/
theorem janet_pages_per_day :
  let belinda_pages_per_day : ℕ := 30
  let weeks : ℕ := 6
  let days_per_week : ℕ := 7
  let extra_pages : ℕ := 2100
  ∀ (janet_pages_per_day : ℕ),
    janet_pages_per_day * (weeks * days_per_week) = 
      belinda_pages_per_day * (weeks * days_per_week) + extra_pages →
    janet_pages_per_day = 80 := by
  sorry

end NUMINAMATH_CALUDE_janet_pages_per_day_l4032_403227


namespace NUMINAMATH_CALUDE_f_composition_theorem_l4032_403294

-- Define the function f
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the condition that |f(x)| ≤ 1/2 for all x in [2, 4]
def f_condition (p q : ℝ) : Prop :=
  ∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → |f p q x| ≤ 1/2

-- Define the n-fold composition of f
def f_compose (p q : ℝ) : ℕ → ℝ → ℝ
  | 0, x => x
  | n+1, x => f p q (f_compose p q n x)

-- The theorem to prove
theorem f_composition_theorem (p q : ℝ) (h : f_condition p q) :
  f_compose p q 2017 ((5 - Real.sqrt 11) / 2) = (5 + Real.sqrt 11) / 2 :=
sorry

end NUMINAMATH_CALUDE_f_composition_theorem_l4032_403294


namespace NUMINAMATH_CALUDE_combined_collection_size_l4032_403244

/-- The number of books in Tim's collection -/
def tim_books : ℕ := 44

/-- The number of books in Sam's collection -/
def sam_books : ℕ := 52

/-- The number of books in Alex's collection -/
def alex_books : ℕ := 65

/-- The number of books in Katie's collection -/
def katie_books : ℕ := 37

/-- The total number of books in the combined collections -/
def total_books : ℕ := tim_books + sam_books + alex_books + katie_books

theorem combined_collection_size : total_books = 198 := by
  sorry

end NUMINAMATH_CALUDE_combined_collection_size_l4032_403244


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l4032_403226

/-- Given vectors AB, BC, and AD, prove that if A, C, and D are collinear, then m = -2/3 -/
theorem collinear_points_m_value 
  (AB BC AD : ℝ × ℝ)
  (h1 : AB = (7, 6))
  (h2 : BC = (-3, m))
  (h3 : AD = (-1, 2*m))
  (h4 : ∃ k : ℝ, k ≠ 0 ∧ AB + BC = k • AD) :
  m = -2/3 :=
sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l4032_403226


namespace NUMINAMATH_CALUDE_even_sum_necessary_not_sufficient_l4032_403246

theorem even_sum_necessary_not_sufficient :
  (∀ a b : ℤ, (Even a ∧ Even b) → Even (a + b)) ∧
  (∃ a b : ℤ, Even (a + b) ∧ ¬(Even a ∧ Even b)) :=
by sorry

end NUMINAMATH_CALUDE_even_sum_necessary_not_sufficient_l4032_403246


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l4032_403257

/-- The volume of a cylinder minus the volumes of two cones -/
theorem cylinder_minus_cones_volume (r h₁ h₂ h : ℝ) (hr : r = 10) (hh₁ : h₁ = 10) (hh₂ : h₂ = 16) (hh : h = 26) :
  π * r^2 * h - (1/3 * π * r^2 * h₁ + 1/3 * π * r^2 * h₂) = 2600/3 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l4032_403257


namespace NUMINAMATH_CALUDE_ellipse_focus_circle_radius_l4032_403223

/-- The radius of a circle centered at a focus of an ellipse and tangent to it -/
theorem ellipse_focus_circle_radius 
  (a b : ℝ) 
  (ha : a = 6) 
  (hb : b = 5) 
  (h_ellipse : a > b) 
  (h_positive : a > 0 ∧ b > 0) :
  let c := Real.sqrt (a^2 - b^2)
  let r := Real.sqrt ((a + c)^2 - a^2)
  r = Real.sqrt 705 / 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_circle_radius_l4032_403223


namespace NUMINAMATH_CALUDE_tile_difference_is_88_l4032_403298

/-- The number of tiles in the n-th square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- The difference in the number of tiles between the 7th and 5th squares -/
def tile_difference : ℕ := tiles_in_square 7 - tiles_in_square 5

theorem tile_difference_is_88 : tile_difference = 88 := by
  sorry

end NUMINAMATH_CALUDE_tile_difference_is_88_l4032_403298


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4032_403266

theorem complex_fraction_simplification :
  (5 - 3*I) / (2 - 3*I) = -19/5 - 9/5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4032_403266


namespace NUMINAMATH_CALUDE_exam_max_marks_l4032_403225

theorem exam_max_marks (percentage : ℝ) (scored_marks : ℝ) (max_marks : ℝ) : 
  percentage = 0.95 → scored_marks = 285 → percentage * max_marks = scored_marks → max_marks = 300 := by
  sorry

end NUMINAMATH_CALUDE_exam_max_marks_l4032_403225


namespace NUMINAMATH_CALUDE_diver_min_trips_l4032_403206

/-- The minimum number of trips required to carry all objects to the surface -/
def min_trips (capacity : ℕ) (total_objects : ℕ) : ℕ :=
  (total_objects + capacity - 1) / capacity

/-- Theorem: Given a diver who can carry 3 objects at a time and has found 17 objects,
    the minimum number of trips required to carry all objects to the surface is 6 -/
theorem diver_min_trips :
  min_trips 3 17 = 6 := by
  sorry

end NUMINAMATH_CALUDE_diver_min_trips_l4032_403206


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4032_403261

theorem expression_simplification_and_evaluation :
  let x : ℝ := 3
  let expr := (1 / (x - 2) - 1 / (x + 1)) / (3 / (x^2 - 1))
  expr = 2 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4032_403261


namespace NUMINAMATH_CALUDE_planet_combinations_count_l4032_403252

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 5

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := 6

/-- Represents the colonization units required for an Earth-like planet -/
def earth_like_units : ℕ := 2

/-- Represents the colonization units required for a Mars-like planet -/
def mars_like_units : ℕ := 1

/-- Represents the total available colonization units -/
def total_units : ℕ := 12

/-- Calculates the number of ways to choose planets given the constraints -/
def count_planet_combinations : ℕ :=
  (Nat.choose earth_like_planets 3 * Nat.choose mars_like_planets 6) +
  (Nat.choose earth_like_planets 4 * Nat.choose mars_like_planets 4) +
  (Nat.choose earth_like_planets 5 * Nat.choose mars_like_planets 2)

/-- Theorem stating that the number of planet combinations is 100 -/
theorem planet_combinations_count :
  count_planet_combinations = 100 := by sorry

end NUMINAMATH_CALUDE_planet_combinations_count_l4032_403252


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l4032_403273

theorem infinitely_many_solutions (c : ℚ) :
  (∀ y : ℚ, 3 * (2 + 2 * c * y) = 15 * y + 6) ↔ c = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l4032_403273


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l4032_403255

theorem scientific_notation_equality : 3422000 = 3.422 * (10 ^ 6) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l4032_403255


namespace NUMINAMATH_CALUDE_smallest_period_of_sine_function_l4032_403207

/-- Given a function f(x) = √3 * sin(πx/k) whose adjacent maximum and minimum points
    lie on the circle x^2 + y^2 = k^2, prove that its smallest positive period is 4. -/
theorem smallest_period_of_sine_function (k : ℝ) (h : k > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sqrt 3 * Real.sin (π * x / k)
  (∃ x y : ℝ, x^2 + y^2 = k^2 ∧ 
    (f x = Real.sqrt 3 ∧ f ((x + k/2) % (2*k)) = -Real.sqrt 3)) →
  2 * k = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_period_of_sine_function_l4032_403207


namespace NUMINAMATH_CALUDE_intersecting_lines_slope_product_l4032_403268

/-- Given two lines in the xy-plane that intersect at a 30° angle, 
    where the slope of one line is 3 times the slope of the other, 
    the product of their slopes is 1. -/
theorem intersecting_lines_slope_product (m₁ m₂ : ℝ) : 
  m₂ = 3 * m₁ → 
  (|((m₂ - m₁) / (1 + m₁ * m₂))|) = Real.tan (30 * π / 180) → 
  m₁ * m₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_intersecting_lines_slope_product_l4032_403268


namespace NUMINAMATH_CALUDE_school_girls_count_l4032_403232

theorem school_girls_count (total_students sample_size : ℕ) 
  (h_total : total_students = 1600)
  (h_sample : sample_size = 200)
  (h_stratified_sample : ∃ (girls_sampled boys_sampled : ℕ), 
    girls_sampled + boys_sampled = sample_size ∧ 
    girls_sampled + 20 = boys_sampled) :
  ∃ (school_girls : ℕ), 
    school_girls * sample_size = 720 * total_students ∧
    school_girls ≤ total_students :=
by sorry

end NUMINAMATH_CALUDE_school_girls_count_l4032_403232


namespace NUMINAMATH_CALUDE_martha_crayons_count_l4032_403256

/-- Calculate the final number of crayons Martha has after losing half and buying new ones. -/
def final_crayons (initial : ℕ) (new_set : ℕ) : ℕ :=
  initial / 2 + new_set

/-- Theorem stating that Martha's final crayon count is correct. -/
theorem martha_crayons_count : final_crayons 18 20 = 29 := by
  sorry

end NUMINAMATH_CALUDE_martha_crayons_count_l4032_403256


namespace NUMINAMATH_CALUDE_initial_ducks_l4032_403284

theorem initial_ducks (initial final additional : ℕ) 
  (h1 : final = initial + additional)
  (h2 : final = 33)
  (h3 : additional = 20) : 
  initial = 13 := by
sorry

end NUMINAMATH_CALUDE_initial_ducks_l4032_403284


namespace NUMINAMATH_CALUDE_largest_inscribed_square_l4032_403279

theorem largest_inscribed_square (outer_square_side : ℝ) (triangle_side : ℝ) 
  (h1 : outer_square_side = 8)
  (h2 : triangle_side = outer_square_side)
  (h3 : 0 < outer_square_side) :
  let triangle_height : ℝ := triangle_side * (Real.sqrt 3) / 2
  let center_to_midpoint : ℝ := triangle_height / 2
  let inscribed_square_side : ℝ := 2 * center_to_midpoint
  inscribed_square_side = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_square_l4032_403279


namespace NUMINAMATH_CALUDE_circle_area_ratio_l4032_403250

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 * (2 * Real.pi * r₁)) = (48 / 360 * (2 * Real.pi * r₂)) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 16 / 25 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l4032_403250


namespace NUMINAMATH_CALUDE_magnitude_of_vector_difference_l4032_403218

def vector_a : Fin 2 → ℝ := ![2, 1]
def vector_b : Fin 2 → ℝ := ![-2, 4]

theorem magnitude_of_vector_difference :
  Real.sqrt ((vector_a 0 - vector_b 0)^2 + (vector_a 1 - vector_b 1)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_difference_l4032_403218


namespace NUMINAMATH_CALUDE_product_of_roots_l4032_403224

theorem product_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 6*x₁ + 8 = 0 → x₂^2 - 6*x₂ + 8 = 0 → x₁ * x₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l4032_403224


namespace NUMINAMATH_CALUDE_gasoline_spending_increase_l4032_403281

theorem gasoline_spending_increase (P Q : ℝ) (P_new Q_new : ℝ) : 
  P_new = 1.20 * P →
  Q_new = 0.90 * Q →
  P_new * Q_new = 1.08 * (P * Q) :=
sorry

end NUMINAMATH_CALUDE_gasoline_spending_increase_l4032_403281


namespace NUMINAMATH_CALUDE_inequality_proof_l4032_403235

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + a / b) * (1 + b / c) * (1 + c / a) ≥ 2 * (1 + (a + b + c) / Real.rpow (a * b * c) (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4032_403235


namespace NUMINAMATH_CALUDE_investment_solution_l4032_403251

def compound_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate)

def investment_problem (principal : ℝ) : Prop :=
  let year1_amount := compound_interest principal 0.05
  let year2_amount := compound_interest year1_amount 0.07
  let year3_amount := compound_interest year2_amount 0.04
  year3_amount = 1232

theorem investment_solution :
  ∃ (principal : ℝ), investment_problem principal ∧ 
    (principal ≥ 1054.75 ∧ principal ≤ 1054.77) :=
by
  sorry

#check investment_solution

end NUMINAMATH_CALUDE_investment_solution_l4032_403251


namespace NUMINAMATH_CALUDE_existence_of_a_l4032_403274

theorem existence_of_a : ∃ a : ℝ, a ≥ 1 ∧ 
  (∀ x : ℝ, |x - 1| > a → Real.log (x^2 - 3*x + 3) > 0) ∧
  (∃ x : ℝ, Real.log (x^2 - 3*x + 3) > 0 ∧ |x - 1| ≤ a) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_a_l4032_403274


namespace NUMINAMATH_CALUDE_factorial_2007_properties_l4032_403202

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def trailingZeros (n : ℕ) : ℕ :=
  (List.range 4).foldl (λ acc i => acc + n / (5 ^ (i + 1))) 0

def lastNonZeroDigit (n : ℕ) : ℕ := n % 10

theorem factorial_2007_properties :
  trailingZeros (factorial 2007) = 500 ∧
  lastNonZeroDigit (factorial 2007 / (10 ^ trailingZeros (factorial 2007))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_2007_properties_l4032_403202


namespace NUMINAMATH_CALUDE_sqrt_588_simplification_l4032_403271

theorem sqrt_588_simplification : Real.sqrt 588 = 14 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_588_simplification_l4032_403271


namespace NUMINAMATH_CALUDE_sqrt_seven_minus_a_l4032_403240

theorem sqrt_seven_minus_a (a : ℝ) : a = -1 → Real.sqrt (7 - a) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_minus_a_l4032_403240


namespace NUMINAMATH_CALUDE_difference_of_squares_l4032_403238

theorem difference_of_squares (x : ℝ) : 9 - 4 * x^2 = (3 - 2*x) * (3 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4032_403238


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l4032_403286

/-- Given a geometric sequence {a_n} with sum of first n terms S_n, 
    if S_1, S_3, and S_2 form an arithmetic sequence, and a_1 - a_3 = 3, 
    then the common ratio q = -1/2 and a_1 = 4 -/
theorem geometric_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * (a 2 / a 1))
  (h_sum : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1)))
  (h_arith : S 3 - S 2 = S 2 - S 1)
  (h_diff : a 1 - a 3 = 3) :
  a 2 / a 1 = -1/2 ∧ a 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l4032_403286


namespace NUMINAMATH_CALUDE_product_with_floor_l4032_403292

theorem product_with_floor (x : ℝ) : 
  x > 0 → x * ⌊x⌋ = 48 → x = 8 := by sorry

end NUMINAMATH_CALUDE_product_with_floor_l4032_403292


namespace NUMINAMATH_CALUDE_similar_polygons_perimeter_l4032_403209

theorem similar_polygons_perimeter (A₁ A₂ P₁ P₂ : ℝ) : 
  A₁ / A₂ = 1 / 16 →  -- ratio of areas
  P₂ - P₁ = 9 →       -- difference in perimeters
  P₁ = 3 :=           -- perimeter of smaller polygon
by sorry

end NUMINAMATH_CALUDE_similar_polygons_perimeter_l4032_403209


namespace NUMINAMATH_CALUDE_volleyball_teams_l4032_403230

theorem volleyball_teams (total_people : ℕ) (people_per_team : ℕ) (h1 : total_people = 6) (h2 : people_per_team = 2) :
  total_people / people_per_team = 3 := by
sorry

end NUMINAMATH_CALUDE_volleyball_teams_l4032_403230


namespace NUMINAMATH_CALUDE_magnitude_of_mn_l4032_403211

/-- Given vectors and conditions, prove the magnitude of MN --/
theorem magnitude_of_mn (a b c : ℝ × ℝ) (x y : ℝ) : 
  a = (2, -1) →
  b = (x, -2) →
  c = (3, y) →
  ∃ (k : ℝ), a = k • b →  -- a is parallel to b
  (a + b) • (b - c) = 0 →  -- (a + b) is perpendicular to (b - c)
  ‖(y - x, x - y)‖ = 8 * Real.sqrt 2 := by
  sorry

#check magnitude_of_mn

end NUMINAMATH_CALUDE_magnitude_of_mn_l4032_403211


namespace NUMINAMATH_CALUDE_percentage_subtraction_equivalence_l4032_403228

theorem percentage_subtraction_equivalence :
  ∀ (a : ℝ), a - (0.07 * a) = 0.93 * a :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_subtraction_equivalence_l4032_403228


namespace NUMINAMATH_CALUDE_calculator_time_saved_l4032_403217

/-- Proves that using a calculator saves 60 minutes for a 20-problem assignment -/
theorem calculator_time_saved 
  (time_with_calculator : ℕ) 
  (time_without_calculator : ℕ) 
  (num_problems : ℕ) 
  (h1 : time_with_calculator = 2)
  (h2 : time_without_calculator = 5)
  (h3 : num_problems = 20) :
  (time_without_calculator - time_with_calculator) * num_problems = 60 :=
by sorry

end NUMINAMATH_CALUDE_calculator_time_saved_l4032_403217
