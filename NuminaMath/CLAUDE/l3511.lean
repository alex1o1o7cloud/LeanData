import Mathlib

namespace NUMINAMATH_CALUDE_line_point_distance_constraint_l3511_351181

/-- Given a line l: x + y + a = 0 and a point A(2,0), if there exists a point M on line l
    such that |MA| = 2|MO|, then a is in the interval [($2-4\sqrt{2})/3$, ($2+4\sqrt{2})/3$] -/
theorem line_point_distance_constraint (a : ℝ) :
  (∃ x y : ℝ, x + y + a = 0 ∧
    (x - 2)^2 + y^2 = 4 * (x^2 + y^2)) →
  a ∈ Set.Icc ((2 - 4 * Real.sqrt 2) / 3) ((2 + 4 * Real.sqrt 2) / 3) :=
by sorry


end NUMINAMATH_CALUDE_line_point_distance_constraint_l3511_351181


namespace NUMINAMATH_CALUDE_condition_implies_right_triangle_l3511_351104

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition given in the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a + t.b)^2 = t.c^2 + 2*t.a*t.b

-- Define what it means for a triangle to be a right triangle
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

-- State the theorem
theorem condition_implies_right_triangle (t : Triangle) :
  satisfiesCondition t → isRightTriangle t := by
  sorry

end NUMINAMATH_CALUDE_condition_implies_right_triangle_l3511_351104


namespace NUMINAMATH_CALUDE_giannas_savings_l3511_351130

/-- Gianna's savings calculation --/
theorem giannas_savings (daily_savings : ℕ) (days_in_year : ℕ) (total_savings : ℕ) :
  daily_savings = 39 →
  days_in_year = 365 →
  total_savings = daily_savings * days_in_year →
  total_savings = 14235 := by
  sorry

end NUMINAMATH_CALUDE_giannas_savings_l3511_351130


namespace NUMINAMATH_CALUDE_theater_revenue_calculation_l3511_351122

/-- Calculates the total revenue of a movie theater for a day --/
def theater_revenue (
  matinee_ticket_price evening_ticket_price opening_night_ticket_price : ℕ)
  (matinee_popcorn_price evening_popcorn_price opening_night_popcorn_price : ℕ)
  (matinee_drink_price evening_drink_price opening_night_drink_price : ℕ)
  (matinee_customers evening_customers opening_night_customers : ℕ)
  (popcorn_ratio drink_ratio : ℚ)
  (discount_groups : ℕ)
  (discount_group_size : ℕ)
  (discount_percentage : ℚ) : ℕ :=
  sorry

theorem theater_revenue_calculation :
  theater_revenue 5 7 10 8 10 12 3 4 5 32 40 58 (1/2) (1/4) 4 5 (1/10) = 1778 := by
  sorry

end NUMINAMATH_CALUDE_theater_revenue_calculation_l3511_351122


namespace NUMINAMATH_CALUDE_moon_arrangements_l3511_351183

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- Theorem: The number of distinct arrangements of the letters in a word with 4 letters,
    where one letter appears twice and the other two letters appear once each, is 12 -/
theorem moon_arrangements :
  distinctArrangements 4 [2, 1, 1] = 12 := by
  sorry

end NUMINAMATH_CALUDE_moon_arrangements_l3511_351183


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3511_351106

theorem min_value_quadratic :
  ∃ (min_z : ℝ), min_z = -44 ∧ ∀ (x : ℝ), x^2 + 16*x + 20 ≥ min_z :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3511_351106


namespace NUMINAMATH_CALUDE_angle_value_for_point_l3511_351159

theorem angle_value_for_point (θ : Real) (P : Real × Real) :
  P.1 = Real.sin (3 * Real.pi / 4) →
  P.2 = Real.cos (3 * Real.pi / 4) →
  0 ≤ θ →
  θ < 2 * Real.pi →
  (Real.cos θ, Real.sin θ) = (P.1 / Real.sqrt (P.1^2 + P.2^2), P.2 / Real.sqrt (P.1^2 + P.2^2)) →
  θ = 7 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_value_for_point_l3511_351159


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l3511_351179

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  hundreds_lt_10 : hundreds < 10
  tens_lt_10 : tens < 10
  units_lt_10 : units < 10

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

theorem three_digit_number_problem (n : ThreeDigitNumber) 
  (sum_18 : n.hundreds + n.tens + n.units = 18)
  (hundreds_tens_relation : n.hundreds = n.tens + 1)
  (units_tens_relation : n.units = n.tens + 2) :
  n.toNat = 657 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l3511_351179


namespace NUMINAMATH_CALUDE_volume_of_specific_pyramid_l3511_351108

/-- A rectangular base pyramid PABCD with given dimensions --/
structure RectangularBasePyramid where
  AB : ℝ
  BC : ℝ
  PB : ℝ
  PA_perpendicular_to_AD_and_AB : Prop

/-- The volume of a rectangular base pyramid --/
def volume (p : RectangularBasePyramid) : ℝ := sorry

/-- Theorem stating the volume of the specific pyramid --/
theorem volume_of_specific_pyramid :
  ∀ (p : RectangularBasePyramid),
    p.AB = 10 →
    p.BC = 6 →
    p.PB = 20 →
    p.PA_perpendicular_to_AD_and_AB →
    volume p = 200 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_pyramid_l3511_351108


namespace NUMINAMATH_CALUDE_root_equivalence_l3511_351111

theorem root_equivalence (α : ℂ) : 
  α^2 - 2*α - 2 = 0 → α^5 - 44*α^3 - 32*α^2 - 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_root_equivalence_l3511_351111


namespace NUMINAMATH_CALUDE_waiter_tips_ratio_l3511_351178

theorem waiter_tips_ratio (salary tips : ℝ) 
  (h : tips / (salary + tips) = 0.7142857142857143) : 
  tips / salary = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_ratio_l3511_351178


namespace NUMINAMATH_CALUDE_sum_of_integers_between_1_and_10_l3511_351175

theorem sum_of_integers_between_1_and_10 : 
  (Finset.range 8).sum (fun i => i + 2) = 44 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_between_1_and_10_l3511_351175


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l3511_351163

theorem quadratic_root_condition (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
   p * x₁^2 + (p - 1) * x₁ + p + 1 = 0 ∧
   p * x₂^2 + (p - 1) * x₂ + p + 1 = 0 ∧
   x₂ > 2 * x₁) →
  (0 < p ∧ p < 1/7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l3511_351163


namespace NUMINAMATH_CALUDE_pizza_slices_l3511_351112

theorem pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_pizzas = 17) 
  (h2 : slices_per_pizza = 4) : 
  num_pizzas * slices_per_pizza = 68 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l3511_351112


namespace NUMINAMATH_CALUDE_bill_processing_error_l3511_351118

theorem bill_processing_error (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 →
  100 * y + x - (100 * x + y) = 2970 →
  y = x + 30 ∧ 10 ≤ x ∧ x ≤ 69 :=
by sorry

end NUMINAMATH_CALUDE_bill_processing_error_l3511_351118


namespace NUMINAMATH_CALUDE_probability_odd_even_function_selection_l3511_351169

theorem probability_odd_even_function_selection :
  let total_functions : ℕ := 7
  let odd_functions : ℕ := 3
  let even_functions : ℕ := 3
  let neither_odd_nor_even : ℕ := 1
  let total_selections : ℕ := total_functions.choose 2
  let favorable_selections : ℕ := odd_functions * even_functions
  favorable_selections / total_selections = 3 / 7 := by
sorry

end NUMINAMATH_CALUDE_probability_odd_even_function_selection_l3511_351169


namespace NUMINAMATH_CALUDE_jack_and_jill_speed_l3511_351182

/-- Jack and Jill's walking speed problem -/
theorem jack_and_jill_speed :
  ∀ x : ℝ,
  let jack_speed := x^2 - 7*x - 18
  let jill_distance := x^2 + x - 72
  let jill_time := x + 8
  let jill_speed := jill_distance / jill_time
  (jack_speed = jill_speed) →
  (x = 10 → jack_speed = 2) :=
by sorry

end NUMINAMATH_CALUDE_jack_and_jill_speed_l3511_351182


namespace NUMINAMATH_CALUDE_work_problem_underdetermined_l3511_351102

-- Define the work rate of one man and one woman
variable (m w : ℝ)

-- Define the unknown number of men
variable (x : ℝ)

-- Condition 1: x men or 12 women can do the work in 20 days
def condition1 : Prop := x * m * 20 = 12 * w * 20

-- Condition 2: 6 men and 11 women can do the work in 12 days
def condition2 : Prop := (6 * m + 11 * w) * 12 = 1

-- Theorem: The conditions are insufficient to uniquely determine x
theorem work_problem_underdetermined :
  ∃ (m1 w1 x1 : ℝ) (m2 w2 x2 : ℝ),
    condition1 m1 w1 x1 ∧ condition2 m1 w1 ∧
    condition1 m2 w2 x2 ∧ condition2 m2 w2 ∧
    x1 ≠ x2 :=
sorry

end NUMINAMATH_CALUDE_work_problem_underdetermined_l3511_351102


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l3511_351100

theorem quadratic_equation_root (m : ℝ) (α : ℝ) :
  (∃ x : ℂ, x^2 + (1 - 2*I)*x + 3*m - I = 0) →
  (α^2 + (1 - 2*I)*α + 3*m - I = 0) →
  (∃ β : ℂ, β^2 + (1 - 2*I)*β + 3*m - I = 0 ∧ β ≠ α) →
  (β = -1/2 + 2*I) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l3511_351100


namespace NUMINAMATH_CALUDE_bert_tax_percentage_l3511_351136

/-- Represents the tax percentage as a real number between 0 and 1 -/
def tax_percentage : ℝ := sorry

/-- The amount by which Bert increases the price when selling -/
def price_increase : ℝ := 10

/-- The selling price of the barrel -/
def selling_price : ℝ := 90

/-- Bert's profit on the sale -/
def profit : ℝ := 1

theorem bert_tax_percentage :
  tax_percentage = 0.1 ∧
  selling_price = (selling_price - price_increase) + price_increase ∧
  profit = selling_price - (selling_price - price_increase) - (tax_percentage * selling_price) :=
by sorry

end NUMINAMATH_CALUDE_bert_tax_percentage_l3511_351136


namespace NUMINAMATH_CALUDE_prob_draw_queen_l3511_351142

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- Represents the number of a specific card in the deck -/
def cardsOfType (d : Deck) : Nat := d.suits

/-- A standard deck of cards -/
def standardDeck : Deck :=
  { cards := 52
  , ranks := 13
  , suits := 4 }

/-- The probability of drawing a specific card from the deck -/
def probDraw (d : Deck) : ℚ := (cardsOfType d : ℚ) / (d.cards : ℚ)

theorem prob_draw_queen (d : Deck := standardDeck) :
  probDraw d = 1 / 13 := by
  sorry

#eval probDraw standardDeck

end NUMINAMATH_CALUDE_prob_draw_queen_l3511_351142


namespace NUMINAMATH_CALUDE_f_properties_l3511_351114

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 - 18 * x + 5

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 12 * x^2 - 6 * x - 18

theorem f_properties :
  (f' (-1) = 0) ∧
  (f' (3/2) = 0) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) (3/2), f' x < 0) ∧
  (∀ x ∈ Set.Iic (-1 : ℝ), f' x > 0) ∧
  (∀ x ∈ Set.Ioi (3/2 : ℝ), f' x > 0) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≤ 16) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≥ -61/4) ∧
  (f (-1) = 16) ∧
  (f (3/2) = -61/4) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3511_351114


namespace NUMINAMATH_CALUDE_cherry_tree_leaves_l3511_351121

/-- The number of cherry trees originally planned to be planted -/
def originalTreeCount : ℕ := 7

/-- The actual number of cherry trees planted -/
def actualTreeCount : ℕ := 2 * originalTreeCount

/-- The number of leaves dropped by each tree during fall -/
def leavesPerTree : ℕ := 100

/-- The total number of leaves falling from all cherry trees -/
def totalLeaves : ℕ := actualTreeCount * leavesPerTree

theorem cherry_tree_leaves :
  totalLeaves = 1400 := by sorry

end NUMINAMATH_CALUDE_cherry_tree_leaves_l3511_351121


namespace NUMINAMATH_CALUDE_proposition_equivalence_l3511_351155

theorem proposition_equivalence (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x - 4*a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l3511_351155


namespace NUMINAMATH_CALUDE_production_time_theorem_l3511_351154

-- Define the time ratios for parts A, B, and C
def time_ratio_A : ℝ := 1
def time_ratio_B : ℝ := 2
def time_ratio_C : ℝ := 3

-- Define the number of parts produced in 10 hours
def parts_A_10h : ℕ := 2
def parts_B_10h : ℕ := 3
def parts_C_10h : ℕ := 4

-- Define the number of parts to be produced
def parts_A_target : ℕ := 14
def parts_B_target : ℕ := 10
def parts_C_target : ℕ := 2

-- Theorem to prove
theorem production_time_theorem :
  ∃ (x : ℝ),
    x > 0 ∧
    x * time_ratio_A * parts_A_10h + x * time_ratio_B * parts_B_10h + x * time_ratio_C * parts_C_10h = 10 ∧
    x * time_ratio_A * parts_A_target + x * time_ratio_B * parts_B_target + x * time_ratio_C * parts_C_target = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_production_time_theorem_l3511_351154


namespace NUMINAMATH_CALUDE_student_guinea_pig_difference_l3511_351172

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 24

/-- The number of guinea pigs in each fourth-grade classroom -/
def guinea_pigs_per_classroom : ℕ := 3

/-- The number of fourth-grade classrooms -/
def number_of_classrooms : ℕ := 6

/-- The theorem stating the difference between total students and guinea pigs -/
theorem student_guinea_pig_difference :
  students_per_classroom * number_of_classrooms - 
  guinea_pigs_per_classroom * number_of_classrooms = 126 := by
  sorry


end NUMINAMATH_CALUDE_student_guinea_pig_difference_l3511_351172


namespace NUMINAMATH_CALUDE_darwin_food_expense_l3511_351162

theorem darwin_food_expense (initial_amount : ℚ) (gas_fraction : ℚ) (remaining : ℚ) : 
  initial_amount = 600 →
  gas_fraction = 1/3 →
  remaining = 300 →
  (initial_amount - gas_fraction * initial_amount - remaining) / (initial_amount - gas_fraction * initial_amount) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_darwin_food_expense_l3511_351162


namespace NUMINAMATH_CALUDE_twenty_paise_coins_l3511_351184

theorem twenty_paise_coins (total_coins : ℕ) (total_value : ℚ) : 
  total_coins = 324 →
  total_value = 71 →
  ∃ (coins_20 : ℕ) (coins_25 : ℕ),
    coins_20 + coins_25 = total_coins ∧
    (20 * coins_20 + 25 * coins_25 : ℚ) / 100 = total_value ∧
    coins_20 = 200 := by
  sorry

end NUMINAMATH_CALUDE_twenty_paise_coins_l3511_351184


namespace NUMINAMATH_CALUDE_chess_grandmaster_learning_time_l3511_351192

theorem chess_grandmaster_learning_time 
  (total_time : ℕ) 
  (proficiency_multiplier : ℕ) 
  (mastery_multiplier : ℕ) 
  (h1 : total_time = 10100)
  (h2 : proficiency_multiplier = 49)
  (h3 : mastery_multiplier = 100) : 
  ∃ (rule_learning_time : ℕ), 
    rule_learning_time = 2 ∧ 
    total_time = rule_learning_time + 
                 proficiency_multiplier * rule_learning_time + 
                 mastery_multiplier * (rule_learning_time + proficiency_multiplier * rule_learning_time) :=
by sorry

end NUMINAMATH_CALUDE_chess_grandmaster_learning_time_l3511_351192


namespace NUMINAMATH_CALUDE_select_president_and_vice_president_l3511_351109

/-- The number of students in the classroom --/
def num_students : ℕ := 4

/-- The number of positions to be filled (president and vice president) --/
def num_positions : ℕ := 2

/-- Theorem stating the number of ways to select a class president and vice president --/
theorem select_president_and_vice_president :
  (num_students * (num_students - 1)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_select_president_and_vice_president_l3511_351109


namespace NUMINAMATH_CALUDE_average_salary_theorem_l3511_351139

def salary_A : ℕ := 9000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

theorem average_salary_theorem :
  (total_salary : ℚ) / num_people = 8200 := by sorry

end NUMINAMATH_CALUDE_average_salary_theorem_l3511_351139


namespace NUMINAMATH_CALUDE_double_inequality_solution_l3511_351127

theorem double_inequality_solution (x : ℝ) :
  (-2 < (x^2 - 16*x + 15) / (x^2 - 2*x + 5) ∧ (x^2 - 16*x + 15) / (x^2 - 2*x + 5) < 1) ↔
  (5/7 < x ∧ x < 5/3) ∨ (5 < x) :=
by sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l3511_351127


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l3511_351116

/-- Given a line segment with midpoint (-1, 4) and one endpoint (3, -1), 
    the other endpoint is (-5, 9). -/
theorem other_endpoint_of_line_segment (m x₁ y₁ x₂ y₂ : ℝ) : 
  m = (-1 : ℝ) ∧ 
  (4 : ℝ) = (y₁ + y₂) / 2 ∧ 
  x₁ = (3 : ℝ) ∧ 
  y₁ = (-1 : ℝ) ∧ 
  m = (x₁ + x₂) / 2 → 
  x₂ = (-5 : ℝ) ∧ y₂ = (9 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l3511_351116


namespace NUMINAMATH_CALUDE_rectangle_placement_l3511_351180

theorem rectangle_placement (a b c d : ℝ) 
  (h1 : a < c) (h2 : c ≤ d) (h3 : d < b) (h4 : a * b < c * d) :
  (∃ (α : ℝ), a * (Real.cos α) + b * (Real.sin α) ≤ c ∧ 
              a * (Real.sin α) + b * (Real.cos α) ≤ d) ↔ 
  (b^2 - a^2)^2 ≤ (b*d - a*c)^2 + (b*c - a*d)^2 := by sorry

end NUMINAMATH_CALUDE_rectangle_placement_l3511_351180


namespace NUMINAMATH_CALUDE_clock_time_sum_l3511_351166

/-- Represents time on a 12-hour digital clock -/
structure ClockTime where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

def addTime (start : ClockTime) (hours minutes seconds : Nat) : ClockTime :=
  let totalSeconds := start.hours * 3600 + start.minutes * 60 + start.seconds +
                      hours * 3600 + minutes * 60 + seconds
  let newSeconds := totalSeconds % 86400  -- 24 hours in seconds
  { hours := (newSeconds / 3600) % 12,
    minutes := (newSeconds % 3600) / 60,
    seconds := newSeconds % 60 }

def sumDigits (time : ClockTime) : Nat :=
  time.hours + time.minutes + time.seconds

theorem clock_time_sum (startTime : ClockTime) :
  let endTime := addTime startTime 189 58 52
  sumDigits endTime = 122 := by
  sorry

end NUMINAMATH_CALUDE_clock_time_sum_l3511_351166


namespace NUMINAMATH_CALUDE_smallest_n_for_floor_equation_l3511_351119

theorem smallest_n_for_floor_equation : 
  ∀ n : ℕ, n < 7 → ¬∃ x : ℤ, ⌊(10 : ℝ)^n / x⌋ = 2006 ∧ 
  ∃ x : ℤ, ⌊(10 : ℝ)^7 / x⌋ = 2006 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_floor_equation_l3511_351119


namespace NUMINAMATH_CALUDE_rose_additional_money_l3511_351134

def paintbrush_cost : ℚ := 2.4
def paints_cost : ℚ := 9.2
def easel_cost : ℚ := 6.5
def rose_money : ℚ := 7.1

theorem rose_additional_money :
  paintbrush_cost + paints_cost + easel_cost - rose_money = 11 := by sorry

end NUMINAMATH_CALUDE_rose_additional_money_l3511_351134


namespace NUMINAMATH_CALUDE_soda_difference_is_21_l3511_351191

/-- The number of regular soda bottles -/
def regular_soda : ℕ := 81

/-- The number of diet soda bottles -/
def diet_soda : ℕ := 60

/-- The difference between regular and diet soda bottles -/
def soda_difference : ℕ := regular_soda - diet_soda

theorem soda_difference_is_21 : soda_difference = 21 := by
  sorry

end NUMINAMATH_CALUDE_soda_difference_is_21_l3511_351191


namespace NUMINAMATH_CALUDE_jar_water_problem_l3511_351197

theorem jar_water_problem (small_capacity large_capacity water_amount : ℝ) 
  (h1 : water_amount = (1/6) * small_capacity)
  (h2 : water_amount = (1/3) * large_capacity)
  (h3 : small_capacity > 0)
  (h4 : large_capacity > 0) :
  water_amount / large_capacity = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_jar_water_problem_l3511_351197


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_ratio_l3511_351125

theorem arithmetic_sequence_cosine_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 9 - a 8) →  -- arithmetic sequence condition
  a 8 = 8 →                            -- given condition
  a 9 = 8 + π / 3 →                    -- given condition
  (Real.cos (a 5) + Real.cos (a 7)) / Real.cos (a 6) = 1 := by
    sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_ratio_l3511_351125


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l3511_351194

theorem multiply_mixed_number : 7 * (12 + 1/4) = 85 + 3/4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l3511_351194


namespace NUMINAMATH_CALUDE_locus_of_midpoints_correct_l3511_351132

/-- Given a square ABCD with center at the origin, rotating around its center,
    and a fixed line l with equation y = a, this function represents the locus of
    the midpoints of segments PQ, where P is the foot of the perpendicular from D to l,
    and Q is the midpoint of AB. -/
def locusOfMidpoints (a : ℝ) (x y : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 
    p.1 = t ∧ 
    p.2 = -t + a/2 ∧ 
    t = (x - y)/4 ∧ 
    x - y ∈ Set.Icc (-Real.sqrt (2 * (x^2 + y^2))) (Real.sqrt (2 * (x^2 + y^2)))}

/-- Theorem stating that the locus of midpoints is correct for any rotating square ABCD
    with center at the origin and any fixed line y = a. -/
theorem locus_of_midpoints_correct (a : ℝ) : 
  ∀ x y : ℝ, locusOfMidpoints a x y = 
    {p : ℝ × ℝ | ∃ t : ℝ, 
      p.1 = t ∧ 
      p.2 = -t + a/2 ∧ 
      t = (x - y)/4 ∧ 
      x - y ∈ Set.Icc (-Real.sqrt (2 * (x^2 + y^2))) (Real.sqrt (2 * (x^2 + y^2)))} :=
by sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_correct_l3511_351132


namespace NUMINAMATH_CALUDE_city_distance_proof_l3511_351101

theorem city_distance_proof : 
  ∃ S : ℕ+, 
    (∀ x : ℕ, x ≤ S → (Nat.gcd x (S - x) = 1 ∨ Nat.gcd x (S - x) = 3 ∨ Nat.gcd x (S - x) = 13)) ∧ 
    (∀ T : ℕ+, T < S → ∃ y : ℕ, y ≤ T ∧ Nat.gcd y (T - y) ≠ 1 ∧ Nat.gcd y (T - y) ≠ 3 ∧ Nat.gcd y (T - y) ≠ 13) ∧
    S = 39 :=
by sorry

end NUMINAMATH_CALUDE_city_distance_proof_l3511_351101


namespace NUMINAMATH_CALUDE_average_of_combined_sets_l3511_351158

theorem average_of_combined_sets (M N : ℕ) (X Y : ℝ) :
  let sum_M := M * X
  let sum_N := N * Y
  let total_sum := sum_M + sum_N
  let total_count := M + N
  (sum_M / M = X) → (sum_N / N = Y) → (total_sum / total_count = (M * X + N * Y) / (M + N)) :=
by sorry

end NUMINAMATH_CALUDE_average_of_combined_sets_l3511_351158


namespace NUMINAMATH_CALUDE_equation_solution_l3511_351157

theorem equation_solution : ∃ x : ℚ, (1/7 : ℚ) + 7/x = 15/x + (1/15 : ℚ) ∧ x = 105 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3511_351157


namespace NUMINAMATH_CALUDE_two_books_different_genres_count_l3511_351171

/-- Represents the number of books in each genre -/
def booksPerGenre : ℕ := 3

/-- Represents the number of genres -/
def numberOfGenres : ℕ := 4

/-- Represents the number of genres to choose -/
def genresToChoose : ℕ := 2

/-- Calculates the number of ways to choose two books of different genres -/
def chooseTwoBooksOfDifferentGenres : ℕ :=
  Nat.choose numberOfGenres genresToChoose * booksPerGenre * booksPerGenre

theorem two_books_different_genres_count :
  chooseTwoBooksOfDifferentGenres = 54 := by
  sorry

end NUMINAMATH_CALUDE_two_books_different_genres_count_l3511_351171


namespace NUMINAMATH_CALUDE_quadratic_roots_and_sum_l3511_351196

theorem quadratic_roots_and_sum : ∃ (m n p : ℕ), 
  (∀ x : ℝ, 2 * x * (5 * x - 11) = -5 ↔ x = (m + Real.sqrt n : ℝ) / p ∨ x = (m - Real.sqrt n : ℝ) / p) ∧ 
  Nat.gcd m (Nat.gcd n p) = 1 ∧
  m + n + p = 92 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_sum_l3511_351196


namespace NUMINAMATH_CALUDE_complex_absolute_value_l3511_351131

theorem complex_absolute_value (ω : ℂ) (h : ω = 7 + 4 * Complex.I) :
  Complex.abs (ω^2 + 10*ω + 88) = Real.sqrt 313 * 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l3511_351131


namespace NUMINAMATH_CALUDE_negation_of_inequality_implication_is_true_l3511_351177

theorem negation_of_inequality_implication_is_true :
  ∀ (a b c : ℝ), (a ≤ b → a * c^2 ≤ b * c^2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_inequality_implication_is_true_l3511_351177


namespace NUMINAMATH_CALUDE_professors_age_l3511_351146

/-- Represents a four-digit number abac --/
def FourDigitNumber (a b c : Nat) : Nat :=
  1000 * a + 100 * b + 10 * a + c

/-- Represents a two-digit number ab --/
def TwoDigitNumber (a b : Nat) : Nat :=
  10 * a + b

theorem professors_age (a b c : Nat) (x : Nat) 
  (h1 : x^2 = FourDigitNumber a b c)
  (h2 : x = TwoDigitNumber a b + TwoDigitNumber a c) :
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_professors_age_l3511_351146


namespace NUMINAMATH_CALUDE_starting_number_is_24_l3511_351187

/-- Given that there are 35 even integers between a starting number and 95,
    prove that the starting number is 24. -/
theorem starting_number_is_24 (start : ℤ) : 
  (start < 95) →
  (∃ (evens : Finset ℤ), evens.card = 35 ∧ 
    (∀ n ∈ evens, start < n ∧ n < 95 ∧ Even n) ∧
    (∀ n, start < n ∧ n < 95 ∧ Even n → n ∈ evens)) →
  start = 24 := by
sorry

end NUMINAMATH_CALUDE_starting_number_is_24_l3511_351187


namespace NUMINAMATH_CALUDE_cyclic_power_inequality_l3511_351135

theorem cyclic_power_inequality (a b c r s : ℝ) 
  (hr : r > s) (hs : s > 0) (hab : a > b) (hbc : b > c) :
  a^r * b^s + b^r * c^s + c^r * a^s ≥ a^s * b^r + b^s * c^r + c^s * a^r :=
by sorry

end NUMINAMATH_CALUDE_cyclic_power_inequality_l3511_351135


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l3511_351126

theorem sqrt_product_simplification : Real.sqrt 5 * Real.sqrt (4/5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l3511_351126


namespace NUMINAMATH_CALUDE_basketball_only_count_l3511_351113

theorem basketball_only_count (total : ℕ) (basketball : ℕ) (table_tennis : ℕ) (neither : ℕ)
  (h_total : total = 30)
  (h_basketball : basketball = 15)
  (h_table_tennis : table_tennis = 10)
  (h_neither : neither = 8)
  (h_sum : total = basketball + table_tennis - (basketball + table_tennis - total + neither) + neither) :
  basketball - (basketball + table_tennis - total + neither) = 12 := by
  sorry

end NUMINAMATH_CALUDE_basketball_only_count_l3511_351113


namespace NUMINAMATH_CALUDE_pool_capacity_l3511_351141

theorem pool_capacity (C : ℝ) 
  (h1 : 0.55 * C + 300 = 0.85 * C) : C = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l3511_351141


namespace NUMINAMATH_CALUDE_sequence_identity_l3511_351161

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n ≥ 1 ∧ a (a n) + a n = 2 * n

theorem sequence_identity (a : ℕ → ℕ) (h : is_valid_sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = n :=
sorry

end NUMINAMATH_CALUDE_sequence_identity_l3511_351161


namespace NUMINAMATH_CALUDE_extreme_values_when_a_neg_one_max_value_when_a_positive_l3511_351149

noncomputable section

-- Define the function f(x) = (ax^2 + x + a)e^x
def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x + a) * Real.exp x

-- Theorem for part (1)
theorem extreme_values_when_a_neg_one :
  let f := f (-1)
  (∃ x, ∀ y, f y ≥ f x) ∧ (f 0 = -1) ∧
  (∃ x, ∀ y, f y ≤ f x) ∧ (f (-1) = -3 * Real.exp (-1)) := by sorry

-- Theorem for part (2)
theorem max_value_when_a_positive (a : ℝ) (h : a > 0) :
  let f := f a
  let max_value := if a > 1 then (2*a + 1) * Real.exp (-1 - 1/a)
                   else (5*a - 2) * Real.exp (-2)
  ∀ x ∈ Set.Icc (-2) (-1), f x ≤ max_value := by sorry

end

end NUMINAMATH_CALUDE_extreme_values_when_a_neg_one_max_value_when_a_positive_l3511_351149


namespace NUMINAMATH_CALUDE_a_upper_bound_l3511_351107

def f (x a : ℝ) := |x - 1| + |x - 2| - a

theorem a_upper_bound (h : ∀ x : ℝ, f x a > 0) : a < 1 := by
  sorry

end NUMINAMATH_CALUDE_a_upper_bound_l3511_351107


namespace NUMINAMATH_CALUDE_not_all_primes_in_arithmetic_progression_l3511_351144

def arithmetic_progression (a d : ℤ) (n : ℕ) : ℤ := a + d * n

theorem not_all_primes_in_arithmetic_progression (a d : ℤ) (h : d ≥ 1) :
  ∃ n : ℕ, ¬ Prime (arithmetic_progression a d n) :=
sorry

end NUMINAMATH_CALUDE_not_all_primes_in_arithmetic_progression_l3511_351144


namespace NUMINAMATH_CALUDE_smallest_three_digit_number_l3511_351103

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h1 : tens ≥ 1 ∧ tens ≤ 9
  h2 : ones ≥ 0 ∧ ones ≤ 9

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h1 : hundreds ≥ 1 ∧ hundreds ≤ 9
  h2 : tens ≥ 0 ∧ tens ≤ 9
  h3 : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to its numerical value -/
def twoDigitToNat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

/-- Converts a ThreeDigitNumber to its numerical value -/
def threeDigitToNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem smallest_three_digit_number 
  (ab : TwoDigitNumber) 
  (aab : ThreeDigitNumber) 
  (h1 : ab.tens = aab.hundreds ∧ ab.tens = aab.tens)
  (h2 : ab.ones = aab.ones)
  (h3 : ab.tens ≠ ab.ones)
  (h4 : twoDigitToNat ab = (threeDigitToNat aab) / 9) :
  225 ≤ threeDigitToNat aab :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_number_l3511_351103


namespace NUMINAMATH_CALUDE_sum_in_range_l3511_351165

theorem sum_in_range : ∃ (s : ℚ), 
  s = 3 + 1/8 + 4 + 1/3 + 6 + 1/21 ∧ 13 < s ∧ s < 14.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_range_l3511_351165


namespace NUMINAMATH_CALUDE_sophie_savings_l3511_351128

/-- Represents the amount of money saved in a year by not buying dryer sheets -/
def money_saved (loads_per_week : ℕ) (sheets_per_load : ℕ) (sheets_per_box : ℕ) (cost_per_box : ℚ) (weeks_per_year : ℕ) : ℚ :=
  let sheets_per_year := loads_per_week * sheets_per_load * weeks_per_year
  let boxes_per_year := (sheets_per_year + sheets_per_box - 1) / sheets_per_box
  boxes_per_year * cost_per_box

/-- The amount of money Sophie saves in a year by not buying dryer sheets is $11.00 -/
theorem sophie_savings : 
  money_saved 4 1 104 (11/2) 52 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sophie_savings_l3511_351128


namespace NUMINAMATH_CALUDE_product_of_roots_l3511_351124

theorem product_of_roots (x : ℝ) : (x + 2) * (x - 3) = 14 → ∃ y : ℝ, (x + 2) * (x - 3) = 14 ∧ (x * y = -20) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3511_351124


namespace NUMINAMATH_CALUDE_largest_number_l3511_351140

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The value of 85 in base 9 --/
def num1 : Nat := to_base_10 [5, 8] 9

/-- The value of 210 in base 6 --/
def num2 : Nat := to_base_10 [0, 1, 2] 6

/-- The value of 1000 in base 4 --/
def num3 : Nat := to_base_10 [0, 0, 0, 1] 4

/-- The value of 111111 in base 2 --/
def num4 : Nat := to_base_10 [1, 1, 1, 1, 1, 1] 2

theorem largest_number : num2 > num1 ∧ num2 > num3 ∧ num2 > num4 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3511_351140


namespace NUMINAMATH_CALUDE_total_shark_teeth_l3511_351190

/-- The number of teeth a tiger shark has -/
def tiger_teeth : ℕ := 180

/-- The number of teeth a hammerhead shark has -/
def hammerhead_teeth : ℕ := tiger_teeth / 6

/-- The number of teeth a great white shark has -/
def great_white_teeth : ℕ := 2 * (tiger_teeth + hammerhead_teeth)

/-- The number of teeth a mako shark has -/
def mako_teeth : ℕ := (5 * hammerhead_teeth) / 3

/-- The total number of teeth for all four sharks -/
def total_teeth : ℕ := tiger_teeth + hammerhead_teeth + great_white_teeth + mako_teeth

theorem total_shark_teeth : total_teeth = 680 := by
  sorry

end NUMINAMATH_CALUDE_total_shark_teeth_l3511_351190


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l3511_351160

theorem trig_expression_simplification (α : ℝ) :
  (Real.sin (α + Real.pi))^2 * Real.cos (Real.pi + α) * Real.cos (-α - 2*Real.pi) /
  (Real.tan (Real.pi + α) * (Real.sin (Real.pi/2 + α))^3 * Real.sin (-α - 2*Real.pi)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l3511_351160


namespace NUMINAMATH_CALUDE_weight_of_N2O3_l3511_351167

/-- The molar mass of nitrogen in g/mol -/
def molar_mass_N : ℝ := 14.01

/-- The molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- The number of moles of N2O3 -/
def moles_N2O3 : ℝ := 7

/-- The molar mass of N2O3 in g/mol -/
def molar_mass_N2O3 : ℝ := 2 * molar_mass_N + 3 * molar_mass_O

/-- The weight of N2O3 in grams -/
def weight_N2O3 : ℝ := moles_N2O3 * molar_mass_N2O3

theorem weight_of_N2O3 : weight_N2O3 = 532.14 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_N2O3_l3511_351167


namespace NUMINAMATH_CALUDE_max_value_constraint_l3511_351150

theorem max_value_constraint (x y z : ℝ) (h : x^2 + 4*y^2 + 9*z^2 = 3) :
  ∃ (M : ℝ), M = 3 ∧ x + 2*y + 3*z ≤ M ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀^2 + 4*y₀^2 + 9*z₀^2 = 3 ∧ x₀ + 2*y₀ + 3*z₀ = M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3511_351150


namespace NUMINAMATH_CALUDE_simplify_fraction_l3511_351151

theorem simplify_fraction (b c : ℚ) (hb : b = 2) (hc : c = 3) :
  15 * b^4 * c^2 / (45 * b^3 * c) = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3511_351151


namespace NUMINAMATH_CALUDE_sum_of_ages_l3511_351198

/-- Given that Ann is 5 years older than Susan and Ann is 16 years old, 
    prove that the sum of their ages is 27 years. -/
theorem sum_of_ages (ann_age susan_age : ℕ) : 
  ann_age = 16 → 
  ann_age = susan_age + 5 → 
  ann_age + susan_age = 27 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3511_351198


namespace NUMINAMATH_CALUDE_sin_150_degrees_l3511_351105

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l3511_351105


namespace NUMINAMATH_CALUDE_jim_age_proof_l3511_351186

/-- Calculates Jim's age X years from now -/
def jim_future_age (x : ℕ) : ℕ :=
  let tom_age_5_years_ago : ℕ := 32
  let years_since_tom_32 : ℕ := 5
  let years_to_past_reference : ℕ := 7
  let jim_age_difference : ℕ := 5
  27 + x

/-- Proves that Jim's age X years from now is (27 + X) -/
theorem jim_age_proof (x : ℕ) :
  jim_future_age x = 27 + x := by
  sorry

end NUMINAMATH_CALUDE_jim_age_proof_l3511_351186


namespace NUMINAMATH_CALUDE_student_line_arrangements_l3511_351123

theorem student_line_arrangements (n : ℕ) (h : n = 5) :
  (n.factorial : ℕ) - (((n - 1).factorial : ℕ) * 2) = 72 := by
  sorry

end NUMINAMATH_CALUDE_student_line_arrangements_l3511_351123


namespace NUMINAMATH_CALUDE_natural_pairs_with_sum_and_gcd_l3511_351193

theorem natural_pairs_with_sum_and_gcd (a b : ℕ) : 
  a + b = 288 → Nat.gcd a b = 36 → 
  ((a = 36 ∧ b = 252) ∨ (a = 252 ∧ b = 36) ∨ (a = 108 ∧ b = 180) ∨ (a = 180 ∧ b = 108)) :=
by sorry

end NUMINAMATH_CALUDE_natural_pairs_with_sum_and_gcd_l3511_351193


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_two_l3511_351189

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio_two 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) 
  (h_ratio : ∀ n : ℕ, a (n + 1) = 2 * a n) : 
  (2 * a 1 + a 2) / (2 * a 3 + a 4) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_two_l3511_351189


namespace NUMINAMATH_CALUDE_meet_once_l3511_351110

/-- Represents the meeting scenario between Michael and the garbage truck --/
structure MeetingScenario where
  michael_speed : ℝ
  pail_distance : ℝ
  truck_speed : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def number_of_meetings (scenario : MeetingScenario) : ℕ :=
  sorry

/-- The theorem stating that Michael and the truck meet exactly once --/
theorem meet_once (scenario : MeetingScenario) 
  (h1 : scenario.michael_speed = 4)
  (h2 : scenario.pail_distance = 300)
  (h3 : scenario.truck_speed = 6)
  (h4 : scenario.truck_stop_time = 20)
  (h5 : scenario.initial_distance = 300) :
  number_of_meetings scenario = 1 :=
sorry

end NUMINAMATH_CALUDE_meet_once_l3511_351110


namespace NUMINAMATH_CALUDE_log_comparison_l3511_351115

theorem log_comparison : Real.log 7 / Real.log 5 > Real.log 17 / Real.log 13 := by
  sorry

end NUMINAMATH_CALUDE_log_comparison_l3511_351115


namespace NUMINAMATH_CALUDE_cubic_expansion_sum_l3511_351174

theorem cubic_expansion_sum (a a₁ a₂ a₃ : ℝ) :
  (∀ x, (2*x + 1)^3 = a + a₁*x + a₂*x^2 + a₃*x^3) →
  -a + a₁ - a₂ + a₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_sum_l3511_351174


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3511_351188

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - x -/
def f (x : ℝ) : ℝ := x^2 - x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3511_351188


namespace NUMINAMATH_CALUDE_square_rectangle_contradiction_l3511_351185

-- Define the square and rectangle
structure Square where
  side : ℝ
  area : ℝ := side ^ 2

structure Rectangle where
  length : ℝ
  width : ℝ
  area : ℝ := length * width

-- Define the theorem
theorem square_rectangle_contradiction 
  (s : Square) 
  (r : Rectangle) 
  (h1 : r.area = 0.25 * s.area) 
  (h2 : s.area = 0.5 * r.area) : 
  False := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_contradiction_l3511_351185


namespace NUMINAMATH_CALUDE_total_earrings_l3511_351152

theorem total_earrings (bella_earrings : ℕ) (monica_earrings : ℕ) (rachel_earrings : ℕ) 
  (h1 : bella_earrings = 10)
  (h2 : bella_earrings = monica_earrings / 4)
  (h3 : monica_earrings = 2 * rachel_earrings) : 
  bella_earrings + monica_earrings + rachel_earrings = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_earrings_l3511_351152


namespace NUMINAMATH_CALUDE_nested_root_equality_l3511_351120

theorem nested_root_equality (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) :
  (∀ N : ℝ, N ≠ 1 → (N^(1/a + 1/(a*b) + 1/(a*b*c)) = N^(15/24))) →
  c = 8 := by
sorry

end NUMINAMATH_CALUDE_nested_root_equality_l3511_351120


namespace NUMINAMATH_CALUDE_chicken_duck_difference_l3511_351143

theorem chicken_duck_difference (total_birds ducks : ℕ) 
  (h1 : total_birds = 95) 
  (h2 : ducks = 32) : 
  total_birds - ducks - ducks = 31 := by
  sorry

end NUMINAMATH_CALUDE_chicken_duck_difference_l3511_351143


namespace NUMINAMATH_CALUDE_taxi_ride_distance_l3511_351137

/-- Calculates the distance of a taxi ride given the fare structure and total fare -/
theorem taxi_ride_distance
  (initial_fare : ℚ)
  (initial_distance : ℚ)
  (additional_fare : ℚ)
  (additional_distance : ℚ)
  (total_fare : ℚ)
  (h1 : initial_fare = 8)
  (h2 : initial_distance = 1/5)
  (h3 : additional_fare = 4/5)
  (h4 : additional_distance = 1/5)
  (h5 : total_fare = 39.2) :
  ∃ (distance : ℚ), distance = 8 ∧ 
    total_fare = initial_fare + (distance - initial_distance) / additional_distance * additional_fare :=
by sorry

end NUMINAMATH_CALUDE_taxi_ride_distance_l3511_351137


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l3511_351138

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Define the theorem
theorem lines_perpendicular_to_plane_are_parallel
  (l m : Line) (α : Plane) :
  l ≠ m →
  perpendicular l α →
  perpendicular m α →
  parallel l m :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l3511_351138


namespace NUMINAMATH_CALUDE_min_discriminant_l3511_351156

/-- A quadratic trinomial that satisfies the problem conditions -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ
  nonnegative : ∀ x, a * x^2 + b * x + c ≥ 0
  below_curve : ∀ x, abs x < 1 → a * x^2 + b * x + c ≤ 1 / Real.sqrt (1 - x^2)

/-- The discriminant of a quadratic trinomial -/
def discriminant (q : QuadraticTrinomial) : ℝ := q.b^2 - 4 * q.a * q.c

/-- The theorem stating the minimum value of the discriminant -/
theorem min_discriminant :
  (∀ q : QuadraticTrinomial, discriminant q ≥ -4) ∧
  (∃ q : QuadraticTrinomial, discriminant q = -4) := by sorry

end NUMINAMATH_CALUDE_min_discriminant_l3511_351156


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l3511_351176

/-- Hyperbola C: x²/a² - y²/b² = 1 -/
def Hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Line l: y = kx + m -/
def Line (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m

/-- Point on a line -/
def PointOnLine (k m x y : ℝ) : Prop :=
  Line k m x y

/-- Point on the hyperbola -/
def PointOnHyperbola (a b x y : ℝ) : Prop :=
  Hyperbola a b x y

/-- Midpoint of two points -/
def Midpoint (x1 y1 x2 y2 xm ym : ℝ) : Prop :=
  xm = (x1 + x2) / 2 ∧ ym = (y1 + y2) / 2

/-- kAB · kOM = 3/4 condition -/
def SlopeProduct (xa ya xb yb xm ym : ℝ) : Prop :=
  ((yb - ya) / (xb - xa)) * (ym / xm) = 3/4

/-- Circle passing through three points -/
def CircleThroughPoints (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  ∃ (xc yc r : ℝ), (x1 - xc)^2 + (y1 - yc)^2 = r^2 ∧
                   (x2 - xc)^2 + (y2 - yc)^2 = r^2 ∧
                   (x3 - xc)^2 + (y3 - yc)^2 = r^2

theorem hyperbola_line_intersection
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a^2 / b^2 = 4/3)  -- Derived from eccentricity √7/2
  (k m : ℝ)
  (xa ya xb yb : ℝ)
  (h4 : PointOnHyperbola a b xa ya)
  (h5 : PointOnHyperbola a b xb yb)
  (h6 : PointOnLine k m xa ya)
  (h7 : PointOnLine k m xb yb)
  (xm ym : ℝ)
  (h8 : Midpoint xa ya xb yb xm ym)
  (h9 : SlopeProduct xa ya xb yb xm ym)
  (h10 : ¬(PointOnLine k m 2 0))
  (h11 : CircleThroughPoints xa ya xb yb 2 0) :
  PointOnLine k m 14 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l3511_351176


namespace NUMINAMATH_CALUDE_arithmetic_sum_modulo_15_l3511_351145

/-- The sum of an arithmetic sequence modulo m -/
def arithmetic_sum_mod (a₁ aₙ d n m : ℕ) : ℕ :=
  ((n * (a₁ + aₙ)) / 2) % m

/-- The number of terms in an arithmetic sequence -/
def arithmetic_terms (a₁ aₙ d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

theorem arithmetic_sum_modulo_15 :
  let a₁ := 2  -- First term
  let aₙ := 102  -- Last term
  let d := 5   -- Common difference
  let m := 15  -- Modulus
  let n := arithmetic_terms a₁ aₙ d
  arithmetic_sum_mod a₁ aₙ d n m = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sum_modulo_15_l3511_351145


namespace NUMINAMATH_CALUDE_f_at_5_l3511_351164

def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 524

theorem f_at_5 : f 5 = 2176 := by
  sorry

end NUMINAMATH_CALUDE_f_at_5_l3511_351164


namespace NUMINAMATH_CALUDE_budget_is_seven_seventy_l3511_351199

/-- The budget for bulbs given the number of crocus bulbs and their cost -/
def budget_for_bulbs (num_crocus : ℕ) (cost_per_crocus : ℚ) : ℚ :=
  num_crocus * cost_per_crocus

/-- Theorem stating that the budget for bulbs is $7.70 -/
theorem budget_is_seven_seventy :
  budget_for_bulbs 22 (35/100) = 77/10 := by
  sorry

end NUMINAMATH_CALUDE_budget_is_seven_seventy_l3511_351199


namespace NUMINAMATH_CALUDE_dave_tickets_l3511_351129

/-- The number of tickets Dave spent on the stuffed tiger -/
def tickets_spent : ℕ := 43

/-- The number of tickets Dave had left after the purchase -/
def tickets_left : ℕ := 55

/-- The initial number of tickets Dave had -/
def initial_tickets : ℕ := tickets_spent + tickets_left

theorem dave_tickets : initial_tickets = 98 := by sorry

end NUMINAMATH_CALUDE_dave_tickets_l3511_351129


namespace NUMINAMATH_CALUDE_correct_scientific_notation_l3511_351173

/-- Scientific notation representation of a positive real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10
  normalize : coefficient * (10 : ℝ) ^ exponent = number

/-- The number we want to represent in scientific notation -/
def number : ℕ := 37600

/-- The scientific notation of the number -/
def scientific_notation : ScientificNotation where
  coefficient := 3.76
  exponent := 4
  coeff_range := by sorry
  normalize := by sorry

/-- Theorem stating that the given scientific notation is correct for the number -/
theorem correct_scientific_notation :
  scientific_notation.coefficient * (10 : ℝ) ^ scientific_notation.exponent = number := by sorry

end NUMINAMATH_CALUDE_correct_scientific_notation_l3511_351173


namespace NUMINAMATH_CALUDE_sign_of_product_l3511_351153

theorem sign_of_product (h1 : 0 < 1 ∧ 1 < Real.pi / 2) 
  (h2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → Real.sin x < Real.sin y)
  (h3 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → Real.cos y < Real.cos x) :
  (Real.cos (Real.cos 1) - Real.cos 1) * (Real.sin (Real.sin 1) - Real.sin 1) < 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_of_product_l3511_351153


namespace NUMINAMATH_CALUDE_dot_product_on_trajectory_l3511_351117

/-- The trajectory E in the xy-plane -/
def TrajectoryE (x y : ℝ) : Prop :=
  |((x + 2)^2 + y^2).sqrt - ((x - 2)^2 + y^2).sqrt| = 2

/-- Point A -/
def A : ℝ × ℝ := (-2, 0)

/-- Point B -/
def B : ℝ × ℝ := (2, 0)

/-- Theorem stating that for any point C on trajectory E with BC perpendicular to x-axis,
    the dot product of AC and BC equals 9 -/
theorem dot_product_on_trajectory (C : ℝ × ℝ) (hC : TrajectoryE C.1 C.2)
  (hPerp : C.1 = B.1) : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_on_trajectory_l3511_351117


namespace NUMINAMATH_CALUDE_solve_for_x_l3511_351195

-- Define the € operation
def euro (x y : ℝ) : ℝ := 3 * x * y

-- State the theorem
theorem solve_for_x (y : ℝ) (h1 : y = 3) (h2 : euro y (euro 4 x) = 540) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3511_351195


namespace NUMINAMATH_CALUDE_max_visible_time_l3511_351168

/-- The maximum time two people can see each other on a circular track with an obstacle -/
theorem max_visible_time (track_radius : ℝ) (obstacle_radius : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : track_radius = 60)
  (h2 : obstacle_radius = 30)
  (h3 : speed1 = 0.4)
  (h4 : speed2 = 0.2) :
  (track_radius * (2 * Real.pi / 3)) / (speed1 - speed2) = 200 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_max_visible_time_l3511_351168


namespace NUMINAMATH_CALUDE_product_of_roots_quartic_l3511_351133

theorem product_of_roots_quartic (p q r s : ℂ) : 
  (3 * p^4 - 8 * p^3 + p^2 - 10 * p - 24 = 0) →
  (3 * q^4 - 8 * q^3 + q^2 - 10 * q - 24 = 0) →
  (3 * r^4 - 8 * r^3 + r^2 - 10 * r - 24 = 0) →
  (3 * s^4 - 8 * s^3 + s^2 - 10 * s - 24 = 0) →
  p * q * r * s = -8 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_quartic_l3511_351133


namespace NUMINAMATH_CALUDE_expression_evaluation_l3511_351148

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = -6 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3511_351148


namespace NUMINAMATH_CALUDE_order_of_numbers_l3511_351147

theorem order_of_numbers : 
  20.3 > 1 → 
  0 < 0.32 ∧ 0.32 < 1 → 
  Real.log 0.32 < 0 → 
  Real.log 0.32 < 0.32 ∧ 0.32 < 20.3 := by
sorry

end NUMINAMATH_CALUDE_order_of_numbers_l3511_351147


namespace NUMINAMATH_CALUDE_b_21_equals_861_l3511_351170

def a (n : ℕ) : ℕ := n * (n + 1) / 2

def b (n : ℕ) : ℕ := a (2 * n - 1)

theorem b_21_equals_861 : b 21 = 861 := by sorry

end NUMINAMATH_CALUDE_b_21_equals_861_l3511_351170
