import Mathlib

namespace NUMINAMATH_CALUDE_joe_running_speed_l3728_372829

/-- Proves that Joe's running speed is 16 km/h given the problem conditions --/
theorem joe_running_speed : 
  ∀ (joe_speed pete_speed : ℝ),
  joe_speed = 2 * pete_speed →  -- Joe runs twice as fast as Pete
  (joe_speed + pete_speed) * (40 / 60) = 16 →  -- Total distance after 40 minutes
  joe_speed = 16 := by
  sorry

end NUMINAMATH_CALUDE_joe_running_speed_l3728_372829


namespace NUMINAMATH_CALUDE_steve_pages_written_l3728_372830

/-- Calculates the total number of pages Steve writes in a month -/
def total_pages_written (days_in_month : ℕ) (letter_frequency : ℕ) (regular_letter_time : ℕ) 
  (time_per_page : ℕ) (long_letter_time : ℕ) : ℕ :=
  let regular_letters := days_in_month / letter_frequency
  let pages_per_regular_letter := regular_letter_time / time_per_page
  let regular_letter_pages := regular_letters * pages_per_regular_letter
  let long_letter_pages := long_letter_time / (2 * time_per_page)
  regular_letter_pages + long_letter_pages

theorem steve_pages_written :
  total_pages_written 30 3 20 10 80 = 24 := by
  sorry

end NUMINAMATH_CALUDE_steve_pages_written_l3728_372830


namespace NUMINAMATH_CALUDE_circle_tangent_to_y_axis_l3728_372873

-- Define the circle's equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 5)^2 = 1

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 5)

-- Define what it means for a circle to be tangent to the y-axis
def tangent_to_y_axis (equation : (ℝ → ℝ → Prop)) : Prop :=
  ∃ y : ℝ, equation 0 y ∧ ∀ x ≠ 0, ¬equation x y

-- Theorem statement
theorem circle_tangent_to_y_axis :
  tangent_to_y_axis circle_equation ∧
  ∀ x y : ℝ, circle_equation x y → (x - circle_center.1)^2 + (y - circle_center.2)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_y_axis_l3728_372873


namespace NUMINAMATH_CALUDE_smallest_number_l3728_372894

-- Define a function to convert a number from base b to decimal
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the given numbers
def num1 : Nat := to_decimal [8, 5] 9
def num2 : Nat := to_decimal [2, 1, 0] 6
def num3 : Nat := to_decimal [1, 0, 0, 0] 4
def num4 : Nat := to_decimal [1, 1, 1, 1, 1, 1] 2

-- Theorem statement
theorem smallest_number : num4 < num1 ∧ num4 < num2 ∧ num4 < num3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3728_372894


namespace NUMINAMATH_CALUDE_difference_of_squares_401_399_l3728_372804

theorem difference_of_squares_401_399 : 401^2 - 399^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_401_399_l3728_372804


namespace NUMINAMATH_CALUDE_smarties_remainder_l3728_372841

theorem smarties_remainder (n : ℕ) (h : n % 11 = 8) : (2 * n) % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_smarties_remainder_l3728_372841


namespace NUMINAMATH_CALUDE_odell_kershaw_passing_l3728_372843

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- radius of the lane in meters
  direction : ℤ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def passingCount (runner1 runner2 : Runner) (totalTime : ℝ) : ℕ :=
  sorry

theorem odell_kershaw_passing :
  let odell : Runner := { speed := 240, radius := 40, direction := 1 }
  let kershaw : Runner := { speed := 320, radius := 55, direction := -1 }
  let totalTime : ℝ := 40
  passingCount odell kershaw totalTime = 75 := by
  sorry

end NUMINAMATH_CALUDE_odell_kershaw_passing_l3728_372843


namespace NUMINAMATH_CALUDE_zoo_ratio_l3728_372892

theorem zoo_ratio (sea_lions penguins : ℕ) : 
  sea_lions = 48 →
  penguins = sea_lions + 84 →
  (sea_lions : ℚ) / penguins = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_zoo_ratio_l3728_372892


namespace NUMINAMATH_CALUDE_lingonberry_price_theorem_l3728_372826

/-- The price per pound of lingonberries picked -/
def price_per_pound : ℚ := 2

/-- The total amount Steve wants to make -/
def total_amount : ℚ := 100

/-- The amount of lingonberries picked on Monday -/
def monday_picked : ℚ := 8

/-- The amount of lingonberries picked on Tuesday -/
def tuesday_picked : ℚ := 3 * monday_picked

/-- The amount of lingonberries picked on Wednesday -/
def wednesday_picked : ℚ := 0

/-- The amount of lingonberries picked on Thursday -/
def thursday_picked : ℚ := 18

/-- The total amount of lingonberries picked over four days -/
def total_picked : ℚ := monday_picked + tuesday_picked + wednesday_picked + thursday_picked

theorem lingonberry_price_theorem : 
  price_per_pound * total_picked = total_amount :=
by sorry

end NUMINAMATH_CALUDE_lingonberry_price_theorem_l3728_372826


namespace NUMINAMATH_CALUDE_customers_left_l3728_372828

theorem customers_left (initial : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 33 → new = 26 → final = 28 → initial - (initial - new + final) = 31 := by
sorry

end NUMINAMATH_CALUDE_customers_left_l3728_372828


namespace NUMINAMATH_CALUDE_grace_exchange_result_l3728_372856

/-- The value of a dime in pennies -/
def dime_value : ℕ := 10

/-- The value of a nickel in pennies -/
def nickel_value : ℕ := 5

/-- The number of dimes Grace has -/
def grace_dimes : ℕ := 10

/-- The number of nickels Grace has -/
def grace_nickels : ℕ := 10

/-- Theorem stating that exchanging Grace's dimes and nickels results in 150 pennies -/
theorem grace_exchange_result : 
  grace_dimes * dime_value + grace_nickels * nickel_value = 150 := by
  sorry

end NUMINAMATH_CALUDE_grace_exchange_result_l3728_372856


namespace NUMINAMATH_CALUDE_last_number_crossed_out_l3728_372814

/-- Represents the circular arrangement of numbers from 1 to 2016 -/
def CircularArrangement := Fin 2016

/-- The deletion process function -/
def deletionProcess (n : ℕ) : ℕ :=
  (n + 2) * (n - 1) / 2

/-- Theorem stating that 2015 is the last number to be crossed out -/
theorem last_number_crossed_out :
  ∃ (n : ℕ), deletionProcess n = 2015 ∧ 
  ∀ (m : ℕ), m > n → deletionProcess m > 2015 :=
sorry

end NUMINAMATH_CALUDE_last_number_crossed_out_l3728_372814


namespace NUMINAMATH_CALUDE_outstanding_student_allocation_schemes_l3728_372878

theorem outstanding_student_allocation_schemes :
  let total_slots : ℕ := 7
  let num_schools : ℕ := 5
  let min_slots_for_two_schools : ℕ := 2
  let remaining_slots : ℕ := total_slots - 2 * min_slots_for_two_schools
  Nat.choose (remaining_slots + num_schools - 1) (num_schools - 1) = Nat.choose total_slots (total_slots - remaining_slots) := by
  sorry

end NUMINAMATH_CALUDE_outstanding_student_allocation_schemes_l3728_372878


namespace NUMINAMATH_CALUDE_research_group_sampling_l3728_372882

/-- Proves that given the research group sizes and selection conditions, m = 24 and n = 9 -/
theorem research_group_sampling (m n : ℕ) : 
  (3 : ℝ) = (n : ℝ) / (30 + m : ℝ) * 18 →  -- Condition for group B selection
  (4 : ℝ) = (n : ℝ) / (30 + m : ℝ) * m →   -- Condition for group C selection
  m = 24 ∧ n = 9 := by
  sorry


end NUMINAMATH_CALUDE_research_group_sampling_l3728_372882


namespace NUMINAMATH_CALUDE_unique_number_property_l3728_372898

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l3728_372898


namespace NUMINAMATH_CALUDE_parry_prob_secretary_or_treasurer_l3728_372879

-- Define the number of club members
def total_members : ℕ := 10

-- Define the probability of being chosen as secretary
def prob_secretary : ℚ := 1 / 9

-- Define the probability of being chosen as treasurer
def prob_treasurer : ℚ := 1 / 10

-- Theorem statement
theorem parry_prob_secretary_or_treasurer :
  let prob_either := prob_secretary + prob_treasurer
  prob_either = 19 / 90 := by
  sorry

end NUMINAMATH_CALUDE_parry_prob_secretary_or_treasurer_l3728_372879


namespace NUMINAMATH_CALUDE_parabola_c_value_l3728_372833

theorem parabola_c_value (b c : ℝ) : 
  (2^2 + 2*b + c = 10) → 
  (4^2 + 4*b + c = 31) → 
  c = -3 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3728_372833


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l3728_372831

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l3728_372831


namespace NUMINAMATH_CALUDE_part_a_part_b_part_c_l3728_372851

/-- Rachel's jump length in cm -/
def rachel_jump : ℕ := 168

/-- Joel's jump length in cm -/
def joel_jump : ℕ := 120

/-- Mark's jump length in cm -/
def mark_jump : ℕ := 72

/-- Theorem for part (a) -/
theorem part_a (n : ℕ) : 
  n > 0 → 5 * rachel_jump = n * joel_jump → n = 7 := by sorry

/-- Theorem for part (b) -/
theorem part_b (r t : ℕ) : 
  r > 0 → t > 0 → 11 ≤ t → t ≤ 19 → r * joel_jump = t * mark_jump → r = 9 ∧ t = 15 := by sorry

/-- Theorem for part (c) -/
theorem part_c (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 → 
  a * rachel_jump = b * joel_jump → 
  b * joel_jump = c * mark_jump → 
  (∀ c' : ℕ, c' > 0 → c' * mark_jump = a * rachel_jump → c ≤ c') → 
  c = 35 := by sorry

end NUMINAMATH_CALUDE_part_a_part_b_part_c_l3728_372851


namespace NUMINAMATH_CALUDE_books_together_l3728_372821

/-- The number of books Tim and Mike have together -/
def total_books (tim_books mike_books : ℕ) : ℕ := tim_books + mike_books

/-- Theorem: Tim and Mike have 42 books together -/
theorem books_together : total_books 22 20 = 42 := by
  sorry

end NUMINAMATH_CALUDE_books_together_l3728_372821


namespace NUMINAMATH_CALUDE_chocolate_eggs_problem_l3728_372876

theorem chocolate_eggs_problem (egg_weight : ℕ) (num_boxes : ℕ) (remaining_weight : ℕ) : 
  egg_weight = 10 →
  num_boxes = 4 →
  remaining_weight = 90 →
  ∃ (total_eggs : ℕ), 
    total_eggs = num_boxes * (remaining_weight / (egg_weight * (num_boxes - 1))) ∧
    total_eggs = 12 := by
sorry

end NUMINAMATH_CALUDE_chocolate_eggs_problem_l3728_372876


namespace NUMINAMATH_CALUDE_grunters_win_probability_l3728_372839

/-- The number of games played -/
def n : ℕ := 6

/-- The number of games to be won -/
def k : ℕ := 5

/-- The probability of winning a single game -/
def p : ℚ := 4/5

/-- The probability of winning exactly k out of n games -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem grunters_win_probability :
  binomial_probability n k p = 6144/15625 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l3728_372839


namespace NUMINAMATH_CALUDE_radish_basket_difference_l3728_372838

theorem radish_basket_difference (total : ℕ) (first_basket : ℕ) : 
  total = 88 → first_basket = 37 → total - first_basket - first_basket = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_radish_basket_difference_l3728_372838


namespace NUMINAMATH_CALUDE_f_properties_l3728_372846

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_periodic : ∀ x : ℝ, f (x + 6) = f x + f 3

-- Theorem to prove
theorem f_properties :
  (f 3 = 0) ∧
  (f (-3) = 0) ∧
  (∀ x : ℝ, f (6 + x) = f (6 - x)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3728_372846


namespace NUMINAMATH_CALUDE_divisor_counts_of_N_l3728_372823

def N : ℕ := 10^40

/-- The number of natural divisors of N that are neither perfect squares nor perfect cubes -/
def count_non_square_non_cube_divisors (n : ℕ) : ℕ := sorry

/-- The number of natural divisors of N that cannot be represented as m^n where m and n are natural numbers and n > 1 -/
def count_non_power_divisors (n : ℕ) : ℕ := sorry

theorem divisor_counts_of_N :
  (count_non_square_non_cube_divisors N = 1093) ∧
  (count_non_power_divisors N = 981) := by sorry

end NUMINAMATH_CALUDE_divisor_counts_of_N_l3728_372823


namespace NUMINAMATH_CALUDE_captain_bonus_calculation_l3728_372865

/-- The number of students in the team -/
def team_size : ℕ := 10

/-- The number of team members (excluding the captain) -/
def team_members : ℕ := 9

/-- The bonus amount for each team member -/
def member_bonus : ℕ := 200

/-- The additional amount the captain receives above the average -/
def captain_extra : ℕ := 90

/-- The bonus amount for the captain -/
def captain_bonus : ℕ := 300

theorem captain_bonus_calculation :
  captain_bonus = 
    (team_members * member_bonus + captain_bonus) / team_size + captain_extra := by
  sorry

#check captain_bonus_calculation

end NUMINAMATH_CALUDE_captain_bonus_calculation_l3728_372865


namespace NUMINAMATH_CALUDE_option_b_more_favorable_example_option_b_more_favorable_l3728_372884

/-- Represents the financial data for a business --/
structure FinancialData where
  planned_revenue : ℕ
  advances_received : ℕ
  monthly_expenses : ℕ

/-- Calculates the tax payable under option (a) --/
def tax_option_a (data : FinancialData) : ℕ :=
  let total_income := data.planned_revenue + data.advances_received
  let tax := total_income * 6 / 100
  let insurance_contributions := data.monthly_expenses * 12
  let deduction := min (tax / 2) insurance_contributions
  tax - deduction

/-- Calculates the tax payable under option (b) --/
def tax_option_b (data : FinancialData) : ℕ :=
  let total_income := data.planned_revenue + data.advances_received
  let annual_expenses := data.monthly_expenses * 12
  let tax_base := max 0 (total_income - annual_expenses)
  let tax := max (total_income / 100) (tax_base * 15 / 100)
  tax

/-- Theorem stating that option (b) results in lower tax --/
theorem option_b_more_favorable (data : FinancialData) :
  tax_option_b data < tax_option_a data :=
by sorry

/-- Example financial data --/
def example_data : FinancialData :=
  { planned_revenue := 120000000
  , advances_received := 30000000
  , monthly_expenses := 11790000 }

/-- Proof that option (b) is more favorable for the example data --/
theorem example_option_b_more_favorable :
  tax_option_b example_data < tax_option_a example_data :=
by sorry

end NUMINAMATH_CALUDE_option_b_more_favorable_example_option_b_more_favorable_l3728_372884


namespace NUMINAMATH_CALUDE_range_of_m_l3728_372855

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def q (x m : ℝ) : Prop := (x - 1 - m) * (x - 1 + m) ≤ 0

-- Define the sufficient condition relationship
def sufficient_condition (m : ℝ) : Prop :=
  ∀ x, q x m → p x

-- Define the not necessary condition relationship
def not_necessary_condition (m : ℝ) : Prop :=
  ∃ x, p x ∧ ¬(q x m)

-- Main theorem
theorem range_of_m :
  ∀ m : ℝ, m > 0 ∧ sufficient_condition m ∧ not_necessary_condition m
  → m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3728_372855


namespace NUMINAMATH_CALUDE_parabola_translation_l3728_372874

/-- The equation of a parabola after vertical translation -/
def translated_parabola (original : ℝ → ℝ) (translation : ℝ) : ℝ → ℝ :=
  fun x => original x + translation

/-- Theorem: Moving y = x^2 up 3 units results in y = x^2 + 3 -/
theorem parabola_translation :
  let original := fun x : ℝ => x^2
  translated_parabola original 3 = fun x => x^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l3728_372874


namespace NUMINAMATH_CALUDE_factorization_of_M_l3728_372880

theorem factorization_of_M (a b c d : ℝ) :
  ((a - c)^2 + (b - d)^2) * (a^2 + b^2) - (a * d - b * c)^2 = (a * c + b * d - a^2 - b^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_M_l3728_372880


namespace NUMINAMATH_CALUDE_fraction_power_five_l3728_372811

theorem fraction_power_five : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_five_l3728_372811


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_count_l3728_372871

/-- Represents a club with members having different characteristics -/
structure Club where
  total : Nat
  leftHanded : Nat
  jazzLovers : Nat
  rightHandedNonJazz : Nat

/-- The number of left-handed jazz lovers in the club -/
def leftHandedJazzLovers (c : Club) : Nat :=
  c.leftHanded + c.jazzLovers - (c.total - c.rightHandedNonJazz)

/-- Theorem stating the number of left-handed jazz lovers in the specific club -/
theorem left_handed_jazz_lovers_count (c : Club) 
  (h1 : c.total = 30)
  (h2 : c.leftHanded = 12)
  (h3 : c.jazzLovers = 20)
  (h4 : c.rightHandedNonJazz = 3) :
  leftHandedJazzLovers c = 5 := by
  sorry

#check left_handed_jazz_lovers_count

end NUMINAMATH_CALUDE_left_handed_jazz_lovers_count_l3728_372871


namespace NUMINAMATH_CALUDE_transformation_correctness_l3728_372842

theorem transformation_correctness (a b : ℝ) (h : a > b) : 1 + 2*a > 1 + 2*b := by
  sorry

end NUMINAMATH_CALUDE_transformation_correctness_l3728_372842


namespace NUMINAMATH_CALUDE_impossibleTransformation_l3728_372889

/-- Represents the three possible colors of the sides of the 99-gon -/
inductive Color
| Red
| Blue
| Yellow

/-- Represents the coloring of the 99-gon -/
def Coloring := Fin 99 → Color

/-- The initial coloring of the 99-gon -/
def initialColoring : Coloring :=
  fun i => match i.val % 3 with
    | 0 => Color.Red
    | 1 => Color.Blue
    | _ => Color.Yellow

/-- The target coloring of the 99-gon -/
def targetColoring : Coloring :=
  fun i => match i.val % 3 with
    | 0 => Color.Blue
    | 1 => Color.Red
    | _ => if i.val == 98 then Color.Blue else Color.Yellow

/-- Checks if a coloring is valid (no adjacent sides have the same color) -/
def isValidColoring (c : Coloring) : Prop :=
  ∀ i : Fin 98, c i ≠ c (i.succ)

/-- Represents a single color change operation -/
def colorChange (c : Coloring) (i : Fin 99) (newColor : Color) : Coloring :=
  fun j => if j = i then newColor else c j

/-- Theorem stating the impossibility of transforming the initial coloring to the target coloring -/
theorem impossibleTransformation :
  ¬∃ (steps : List (Fin 99 × Color)),
    (steps.foldl (fun acc (i, col) => colorChange acc i col) initialColoring = targetColoring) ∧
    (∀ step ∈ steps, isValidColoring (colorChange (steps.foldl (fun acc (i, col) => colorChange acc i col) initialColoring) step.fst step.snd)) :=
sorry


end NUMINAMATH_CALUDE_impossibleTransformation_l3728_372889


namespace NUMINAMATH_CALUDE_min_marked_cells_l3728_372805

/-- Represents a board with dimensions m × n -/
structure Board (m n : ℕ) where
  cells : Fin m → Fin n → Bool

/-- Represents an L-shaped piece on the board -/
inductive LShape
  | makeL : Fin 2 → Fin 2 → LShape

/-- Checks if an L-shape touches a marked cell on the board -/
def touchesMarked (b : Board m n) (l : LShape) : Bool :=
  sorry

/-- Checks if a marking strategy satisfies the condition for all L-shape placements -/
def validMarking (b : Board m n) : Prop :=
  ∀ l : LShape, touchesMarked b l = true

/-- Counts the number of marked cells on the board -/
def countMarked (b : Board m n) : ℕ :=
  sorry

/-- The main theorem stating that 50 is the smallest number of cells to be marked -/
theorem min_marked_cells :
  ∃ (b : Board 10 11), validMarking b ∧ countMarked b = 50 ∧
  ∀ (b' : Board 10 11), validMarking b' → countMarked b' ≥ 50 :=
sorry

end NUMINAMATH_CALUDE_min_marked_cells_l3728_372805


namespace NUMINAMATH_CALUDE_two_correct_probability_l3728_372888

/-- The number of packages and houses -/
def n : ℕ := 5

/-- The probability of exactly 2 out of n packages being delivered correctly -/
def prob_two_correct (n : ℕ) : ℚ :=
  if n ≥ 2 then
    (n.choose 2 : ℚ) / n.factorial
  else 0

theorem two_correct_probability :
  prob_two_correct n = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_two_correct_probability_l3728_372888


namespace NUMINAMATH_CALUDE_twelfth_odd_multiple_of_5_l3728_372809

/-- The nth positive odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- Predicate for a number being odd and a multiple of 5 -/
def isOddMultipleOf5 (x : ℕ) : Prop :=
  x % 2 = 1 ∧ x % 5 = 0

theorem twelfth_odd_multiple_of_5 :
  nthOddMultipleOf5 12 = 115 ∧
  isOddMultipleOf5 (nthOddMultipleOf5 12) :=
sorry

end NUMINAMATH_CALUDE_twelfth_odd_multiple_of_5_l3728_372809


namespace NUMINAMATH_CALUDE_postcard_area_l3728_372861

/-- Represents a rectangular postcard -/
structure Postcard where
  vertical_length : ℝ
  horizontal_length : ℝ

/-- Calculates the area of a postcard -/
def area (p : Postcard) : ℝ := p.vertical_length * p.horizontal_length

/-- Calculates the perimeter of two attached postcards -/
def attached_perimeter (p : Postcard) : ℝ := 2 * p.vertical_length + 4 * p.horizontal_length

theorem postcard_area (p : Postcard) 
  (h1 : p.vertical_length = 15)
  (h2 : attached_perimeter p = 70) : 
  area p = 150 := by
  sorry

#check postcard_area

end NUMINAMATH_CALUDE_postcard_area_l3728_372861


namespace NUMINAMATH_CALUDE_ambers_age_l3728_372801

theorem ambers_age :
  ∀ (a g : ℕ),
  g = 15 * a →
  g - a = 70 →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_ambers_age_l3728_372801


namespace NUMINAMATH_CALUDE_friends_seinfeld_relationship_l3728_372872

-- Define the variables
variable (x y z : ℚ)

-- Define the conditions
def friends_episodes : ℚ := 50
def seinfeld_episodes : ℚ := 75

-- State the theorem
theorem friends_seinfeld_relationship 
  (h1 : x * z = friends_episodes) 
  (h2 : y * z = seinfeld_episodes) :
  y = 1.5 * x := by
  sorry

end NUMINAMATH_CALUDE_friends_seinfeld_relationship_l3728_372872


namespace NUMINAMATH_CALUDE_stratified_sampling_population_size_l3728_372854

theorem stratified_sampling_population_size 
  (total_male : ℕ) 
  (sample_size : ℕ) 
  (female_in_sample : ℕ) 
  (h1 : total_male = 570) 
  (h2 : sample_size = 110) 
  (h3 : female_in_sample = 53) :
  let male_in_sample := sample_size - female_in_sample
  let total_population := (total_male * sample_size) / male_in_sample
  total_population = 1100 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_population_size_l3728_372854


namespace NUMINAMATH_CALUDE_selling_price_increase_for_3360_profit_max_profit_at_10_yuan_increase_l3728_372867

/-- Represents the profit function for T-shirt sales -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 3000

/-- Represents the constraint for a specific profit -/
def profit_constraint (x : ℝ) : Prop := profit_function x = 3360

/-- Theorem: The selling price increase that results in a profit of 3360 yuan is 2 yuan -/
theorem selling_price_increase_for_3360_profit :
  ∃ x : ℝ, profit_constraint x ∧ x = 2 := by sorry

/-- Theorem: The maximum profit occurs when the selling price is increased by 10 yuan, resulting in a profit of 4000 yuan -/
theorem max_profit_at_10_yuan_increase :
  ∃ x : ℝ, x = 10 ∧ profit_function x = 4000 ∧ 
  ∀ y : ℝ, profit_function y ≤ profit_function x := by sorry

end NUMINAMATH_CALUDE_selling_price_increase_for_3360_profit_max_profit_at_10_yuan_increase_l3728_372867


namespace NUMINAMATH_CALUDE_pentagon_area_condition_l3728_372850

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculates the area of a pentagon given its vertices -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Checks if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop := sorry

theorem pentagon_area_condition (y : ℝ) : 
  let p := Pentagon.mk (0, 0) (0, 5) (3, y) (6, 5) (6, 0)
  hasVerticalSymmetry p ∧ pentagonArea p = 50 → y = 35/3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_condition_l3728_372850


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3728_372890

theorem complex_magnitude_problem (z : ℂ) : z = (3 + I) / (2 - I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3728_372890


namespace NUMINAMATH_CALUDE_christen_peeled_24_potatoes_l3728_372807

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  totalPotatoes : ℕ
  homerRate : ℕ
  christenRate : ℕ
  timeBeforeChristenJoins : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christenPeeledPotatoes (scenario : PotatoPeeling) : ℕ :=
  let potatoesLeftAfterHomer := scenario.totalPotatoes - scenario.homerRate * scenario.timeBeforeChristenJoins
  let combinedRate := scenario.homerRate + scenario.christenRate
  let timeForRemaining := potatoesLeftAfterHomer / combinedRate
  scenario.christenRate * timeForRemaining

/-- Theorem stating that Christen peeled 24 potatoes -/
theorem christen_peeled_24_potatoes (scenario : PotatoPeeling) 
  (h1 : scenario.totalPotatoes = 60)
  (h2 : scenario.homerRate = 3)
  (h3 : scenario.christenRate = 4)
  (h4 : scenario.timeBeforeChristenJoins = 6) :
  christenPeeledPotatoes scenario = 24 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_24_potatoes_l3728_372807


namespace NUMINAMATH_CALUDE_negation_of_forall_leq_zero_l3728_372860

theorem negation_of_forall_leq_zero :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_leq_zero_l3728_372860


namespace NUMINAMATH_CALUDE_coefficient_x3_equals_negative_30_l3728_372896

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (1-2x)(1-x)^5
def coefficient_x3 : ℤ :=
  -1 * (-1) * binomial 5 3 + (-2) * binomial 5 2

-- Theorem statement
theorem coefficient_x3_equals_negative_30 : coefficient_x3 = -30 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3_equals_negative_30_l3728_372896


namespace NUMINAMATH_CALUDE_smallest_shift_for_scaled_function_l3728_372819

/-- A function with period 15 -/
def isPeriodic15 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 15) = f x

/-- The property we want to prove -/
def hasShiftProperty (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f ((x - b) / 3) = f (x / 3)

theorem smallest_shift_for_scaled_function 
  (f : ℝ → ℝ) (h : isPeriodic15 f) :
  (∃ b > 0, hasShiftProperty f b) ∧ 
  (∀ b > 0, hasShiftProperty f b → b ≥ 45) :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_for_scaled_function_l3728_372819


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l3728_372859

theorem largest_prime_divisor_to_test (n : ℕ) (h : 1900 ≤ n ∧ n ≤ 1950) :
  (∀ p : ℕ, p.Prime → p ≤ 43 → ¬(p ∣ n)) → n.Prime :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l3728_372859


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3728_372836

theorem quadratic_root_difference (b : ℝ) : 
  (∃ (x y : ℝ), 2 * x^2 + b * x = 12 ∧ 
                 2 * y^2 + b * y = 12 ∧ 
                 y - x = 5.5 ∧ 
                 (∀ z : ℝ, 2 * z^2 + b * z = 12 → (z = x ∨ z = y))) →
  b = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3728_372836


namespace NUMINAMATH_CALUDE_cone_base_radius_l3728_372887

/-- Given a cone formed from a sector with arc length 8π, its base radius is 4. -/
theorem cone_base_radius (cone : Real) (sector : Real) :
  (sector = 8 * Real.pi) →    -- arc length of sector
  (sector = 2 * Real.pi * cone) →    -- arc length equals circumference of base
  (cone = 4) :=    -- radius of base
by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3728_372887


namespace NUMINAMATH_CALUDE_unique_intersection_point_l3728_372883

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

-- State the theorem
theorem unique_intersection_point :
  ∃! c : ℝ, g c = c ∧ c = -3 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l3728_372883


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3728_372845

theorem polynomial_division_remainder : ∃ q r : Polynomial ℤ,
  (3 * X^4 + 14 * X^3 - 35 * X^2 - 80 * X + 56) = 
  (X^2 + 8 * X - 6) * q + r ∧ 
  r.degree < 2 ∧ 
  r = 364 * X - 322 :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3728_372845


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l3728_372824

theorem distance_between_complex_points :
  let z₁ : ℂ := 2 + 3*I
  let z₂ : ℂ := -2 + 2*I
  Complex.abs (z₁ - z₂) = Real.sqrt 17 := by
sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l3728_372824


namespace NUMINAMATH_CALUDE_polynomial_not_equal_77_l3728_372864

theorem polynomial_not_equal_77 (x y : ℤ) : 
  x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
sorry

end NUMINAMATH_CALUDE_polynomial_not_equal_77_l3728_372864


namespace NUMINAMATH_CALUDE_parabola_p_value_l3728_372834

/-- The latus rectum of a parabola y^2 = 2px --/
def latus_rectum (p : ℝ) : ℝ := 4 * p

/-- Theorem: For a parabola y^2 = 2px with latus rectum equal to 4, p equals 2 --/
theorem parabola_p_value : ∀ p : ℝ, latus_rectum p = 4 → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_p_value_l3728_372834


namespace NUMINAMATH_CALUDE_right_triangle_area_l3728_372881

theorem right_triangle_area (h : ℝ) (θ : ℝ) (area : ℝ) : 
  h = 12 →  -- hypotenuse is 12 inches
  θ = 30 * π / 180 →  -- one angle is 30 degrees (converted to radians)
  area = 18 * Real.sqrt 3 →  -- area is 18√3 square inches
  area = (h * h * Real.sin θ * Real.cos θ) / 2 :=
by sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_right_triangle_area_l3728_372881


namespace NUMINAMATH_CALUDE_total_marks_proof_l3728_372837

theorem total_marks_proof (keith_score larry_score danny_score : ℕ) : 
  keith_score = 3 →
  larry_score = 3 * keith_score →
  danny_score = larry_score + 5 →
  keith_score + larry_score + danny_score = 26 := by
sorry

end NUMINAMATH_CALUDE_total_marks_proof_l3728_372837


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3728_372849

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ 
    (x = -7 ∧ y = -99) ∨ 
    (x = -1 ∧ y = -9) ∨ 
    (x = 1 ∧ y = 5) ∨ 
    (x = 7 ∧ y = -97) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3728_372849


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l3728_372800

theorem parallelogram_side_sum (x y : ℝ) : 
  (5 : ℝ) = 10 * y - 3 ∧ (11 : ℝ) = 4 * x + 1 → x + y = 3.3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l3728_372800


namespace NUMINAMATH_CALUDE_expression_value_l3728_372825

theorem expression_value : 
  let a := 2020
  (a^3 - 3*a^2*(a+1) + 4*a*(a+1)^2 - (a+1)^3 + 1) / (a*(a+1)) = 2021 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3728_372825


namespace NUMINAMATH_CALUDE_existence_of_equal_segments_l3728_372815

/-- An acute-angled triangle is a triangle where all angles are less than 90 degrees -/
def AcuteTriangle (A B C : Point) : Prop := sorry

/-- Point X is on line segment AB -/
def OnSegment (X A B : Point) : Prop := sorry

/-- AX = XY = YC -/
def EqualSegments (A X Y C : Point) : Prop := sorry

/-- Theorem: In any acute-angled triangle, there exist points X and Y on its sides
    such that AX = XY = YC -/
theorem existence_of_equal_segments (A B C : Point) 
  (h : AcuteTriangle A B C) : 
  ∃ X Y, OnSegment X A B ∧ OnSegment Y B C ∧ EqualSegments A X Y C := by
  sorry

end NUMINAMATH_CALUDE_existence_of_equal_segments_l3728_372815


namespace NUMINAMATH_CALUDE_diaz_future_age_l3728_372885

/-- Proves Diaz's age 20 years from now given the conditions in the problem -/
theorem diaz_future_age (sierra_age : ℕ) (diaz_age : ℕ) : 
  sierra_age = 30 →
  10 * diaz_age - 40 = 10 * sierra_age + 20 →
  diaz_age + 20 = 56 := by
  sorry

end NUMINAMATH_CALUDE_diaz_future_age_l3728_372885


namespace NUMINAMATH_CALUDE_problem_solution_l3728_372866

theorem problem_solution : 
  (Real.sqrt 12 - 3 * Real.sqrt (1/3) + Real.sqrt 8 = Real.sqrt 3 + 2 * Real.sqrt 2) ∧ 
  ((Real.sqrt 5 - 1)^2 + Real.sqrt 5 * (Real.sqrt 5 + 2) = 11) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3728_372866


namespace NUMINAMATH_CALUDE_girls_from_valley_l3728_372816

theorem girls_from_valley (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (highland_students : ℕ) (valley_students : ℕ) (highland_boys : ℕ)
  (h1 : total_students = 120)
  (h2 : total_boys = 70)
  (h3 : total_girls = 50)
  (h4 : highland_students = 45)
  (h5 : valley_students = 75)
  (h6 : highland_boys = 30)
  (h7 : total_students = total_boys + total_girls)
  (h8 : total_students = highland_students + valley_students)
  (h9 : total_boys ≥ highland_boys) :
  valley_students - (total_boys - highland_boys) = 35 := by
sorry

end NUMINAMATH_CALUDE_girls_from_valley_l3728_372816


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3728_372844

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - k*x + 9 = (x - a)^2) → (k = 6 ∨ k = -6) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3728_372844


namespace NUMINAMATH_CALUDE_max_price_reduction_l3728_372803

/-- The maximum price reduction for a product while maintaining a minimum profit margin -/
theorem max_price_reduction (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 1000 →
  selling_price = 1500 →
  min_profit_margin = 0.05 →
  ∃ (max_reduction : ℝ),
    max_reduction = 450 ∧
    selling_price - max_reduction = cost_price * (1 + min_profit_margin) :=
by sorry

end NUMINAMATH_CALUDE_max_price_reduction_l3728_372803


namespace NUMINAMATH_CALUDE_concert_tickets_l3728_372832

theorem concert_tickets (section_a_price section_b_price : ℝ)
  (total_tickets : ℕ) (total_revenue : ℝ) :
  section_a_price = 8 →
  section_b_price = 4.25 →
  total_tickets = 4500 →
  total_revenue = 30000 →
  ∃ (section_a_sold section_b_sold : ℕ),
    section_a_sold + section_b_sold = total_tickets ∧
    section_a_price * (section_a_sold : ℝ) + section_b_price * (section_b_sold : ℝ) = total_revenue ∧
    section_b_sold = 1600 :=
by sorry

end NUMINAMATH_CALUDE_concert_tickets_l3728_372832


namespace NUMINAMATH_CALUDE_second_number_value_l3728_372822

theorem second_number_value (A B C : ℚ) : 
  A + B + C = 98 → 
  A / B = 2 / 3 → 
  B / C = 5 / 8 → 
  B = 30 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l3728_372822


namespace NUMINAMATH_CALUDE_salesman_pear_sales_l3728_372847

/-- A salesman's pear sales problem -/
theorem salesman_pear_sales (morning_sales afternoon_sales total_sales : ℕ) : 
  morning_sales = 120 →
  afternoon_sales = 240 →
  afternoon_sales = 2 * morning_sales →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 360 := by
  sorry

#check salesman_pear_sales

end NUMINAMATH_CALUDE_salesman_pear_sales_l3728_372847


namespace NUMINAMATH_CALUDE_sum_of_even_integers_l3728_372863

theorem sum_of_even_integers (first last : ℕ) (n : ℕ) (sum : ℕ) : 
  first = 202 →
  last = 300 →
  n = 50 →
  sum = 12550 →
  (last - first) / 2 + 1 = n →
  sum = n / 2 * (first + last) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_l3728_372863


namespace NUMINAMATH_CALUDE_students_without_pens_l3728_372870

theorem students_without_pens (total students_with_blue students_with_red students_with_both : ℕ) 
  (h1 : total = 40)
  (h2 : students_with_blue = 18)
  (h3 : students_with_red = 26)
  (h4 : students_with_both = 10) :
  total - (students_with_blue + students_with_red - students_with_both) = 6 :=
by sorry

end NUMINAMATH_CALUDE_students_without_pens_l3728_372870


namespace NUMINAMATH_CALUDE_special_function_properties_l3728_372862

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) - f y = (x + 2*y + 2) * x

theorem special_function_properties
  (f : ℝ → ℝ) (h : special_function f) (h2 : f 2 = 12) :
  (f 0 = 4) ∧
  (∀ a : ℝ, (∃ x₀ : ℝ, 1 < x₀ ∧ x₀ < 4 ∧ f x₀ - 8 = a * x₀) ↔ -1 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l3728_372862


namespace NUMINAMATH_CALUDE_palmer_photos_l3728_372818

theorem palmer_photos (initial_photos : ℕ) (first_week : ℕ) (final_total : ℕ) :
  initial_photos = 100 →
  first_week = 50 →
  final_total = 380 →
  final_total - initial_photos - first_week - 2 * first_week = 130 := by
sorry

end NUMINAMATH_CALUDE_palmer_photos_l3728_372818


namespace NUMINAMATH_CALUDE_function_equality_l3728_372827

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 7
def g (k : ℝ) (x : ℝ) : ℝ := x^2 - k * x + 5

-- State the theorem
theorem function_equality (k : ℝ) : f 5 - g k 5 = 0 → k = -92 / 5 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l3728_372827


namespace NUMINAMATH_CALUDE_equation_solution_l3728_372899

theorem equation_solution (a b : ℝ) 
  (h1 : 2*a + b = -3) 
  (h2 : 2*a - b = 2) : 
  4*a^2 - b^2 = -6 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3728_372899


namespace NUMINAMATH_CALUDE_mother_twice_bob_age_year_l3728_372840

def bob_age_2010 : ℕ := 10
def mother_age_2010 : ℕ := 5 * bob_age_2010

def year_mother_twice_bob_age : ℕ :=
  2010 + (mother_age_2010 - 2 * bob_age_2010)

theorem mother_twice_bob_age_year :
  year_mother_twice_bob_age = 2040 := by
  sorry

end NUMINAMATH_CALUDE_mother_twice_bob_age_year_l3728_372840


namespace NUMINAMATH_CALUDE_complex_coordinate_to_z_l3728_372812

theorem complex_coordinate_to_z (z : ℂ) :
  (z / Complex.I).re = 3 ∧ (z / Complex.I).im = -1 → z = 1 + 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_coordinate_to_z_l3728_372812


namespace NUMINAMATH_CALUDE_tractor_circuits_l3728_372869

theorem tractor_circuits (r₁ r₂ : ℝ) (n₁ : ℕ) (h₁ : r₁ = 30) (h₂ : r₂ = 10) (h₃ : n₁ = 20) :
  ∃ n₂ : ℕ, n₂ = 60 ∧ r₁ * n₁ = r₂ * n₂ := by
  sorry

end NUMINAMATH_CALUDE_tractor_circuits_l3728_372869


namespace NUMINAMATH_CALUDE_fair_number_exists_l3728_372808

/-- Represents a digit as a natural number between 0 and 9 -/
def Digit : Type := { n : ℕ // n < 10 }

/-- Represents a number as a list of digits -/
def Number := List Digit

/-- Checks if a digit is even -/
def isEven (d : Digit) : Bool :=
  d.val % 2 = 0

/-- Counts the number of even digits at odd positions and even positions -/
def countEvenDigits (n : Number) : ℕ × ℕ :=
  let rec count (digits : List Digit) (isOddPosition : Bool) (evenOdd evenEven : ℕ) : ℕ × ℕ :=
    match digits with
    | [] => (evenOdd, evenEven)
    | d :: ds =>
      if isEven d then
        if isOddPosition then
          count ds (not isOddPosition) (evenOdd + 1) evenEven
        else
          count ds (not isOddPosition) evenOdd (evenEven + 1)
      else
        count ds (not isOddPosition) evenOdd evenEven
  count n true 0 0

/-- Checks if a number is fair (equal number of even digits at odd and even positions) -/
def isFair (n : Number) : Bool :=
  let (evenOdd, evenEven) := countEvenDigits n
  evenOdd = evenEven

/-- Main theorem: For any number with an odd number of digits, 
    there exists a way to remove one digit to make it fair -/
theorem fair_number_exists (n : Number) (h : n.length % 2 = 1) :
  ∃ (i : Fin n.length), isFair (n.removeNth i) := by
  sorry

end NUMINAMATH_CALUDE_fair_number_exists_l3728_372808


namespace NUMINAMATH_CALUDE_prime_relations_l3728_372848

theorem prime_relations (p : ℕ) : 
  (Prime p ∧ Prime (8*p - 1)) → (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = 8*p + 1) ∧
  (Prime p ∧ Prime (8*p^2 + 1)) → Prime (8*p^2 - 1) :=
sorry

end NUMINAMATH_CALUDE_prime_relations_l3728_372848


namespace NUMINAMATH_CALUDE_stratified_random_most_appropriate_l3728_372852

/-- Represents a laboratory with a certain number of mice -/
structure Laboratory where
  mice : ℕ

/-- Represents a sampling method -/
inductive SamplingMethod
  | EqualFromEach
  | FullyRandom
  | ArbitraryStratified
  | StratifiedRandom

/-- The problem setup -/
def biochemistryLabs : List Laboratory := [
  { mice := 18 },
  { mice := 24 },
  { mice := 54 },
  { mice := 48 }
]

/-- The total number of mice to be selected -/
def selectionSize : ℕ := 24

/-- Function to determine the most appropriate sampling method -/
def mostAppropriateSamplingMethod (labs : List Laboratory) (selectionSize : ℕ) : SamplingMethod :=
  SamplingMethod.StratifiedRandom

/-- Theorem stating that StratifiedRandom is the most appropriate method -/
theorem stratified_random_most_appropriate :
  mostAppropriateSamplingMethod biochemistryLabs selectionSize = SamplingMethod.StratifiedRandom := by
  sorry


end NUMINAMATH_CALUDE_stratified_random_most_appropriate_l3728_372852


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3728_372817

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  A = π / 3 →
  Real.cos C = 1 / 3 →
  c = 4 * Real.sqrt 2 →
  (1 / 2) * a * c * Real.sin B = 4 * Real.sqrt 3 + 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_proof_l3728_372817


namespace NUMINAMATH_CALUDE_sin_cos_15_product_eq_neg_sqrt3_div_2_l3728_372813

theorem sin_cos_15_product_eq_neg_sqrt3_div_2 :
  (Real.sin (15 * π / 180) + Real.cos (15 * π / 180)) *
  (Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_15_product_eq_neg_sqrt3_div_2_l3728_372813


namespace NUMINAMATH_CALUDE_probability_one_defective_l3728_372875

def total_items : ℕ := 6
def good_items : ℕ := 4
def defective_items : ℕ := 2
def selected_items : ℕ := 3

theorem probability_one_defective :
  (Nat.choose good_items (selected_items - 1) * Nat.choose defective_items 1) /
  Nat.choose total_items selected_items = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_one_defective_l3728_372875


namespace NUMINAMATH_CALUDE_problem_solution_l3728_372891

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -5) :
  x + x^3 / y^2 + y^3 / x^2 + y = 285 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3728_372891


namespace NUMINAMATH_CALUDE_quadratic_polynomial_root_l3728_372877

theorem quadratic_polynomial_root (x : ℂ) : 
  let p : ℂ → ℂ := λ z => 3 * z^2 - 24 * z + 51
  (p (4 + I) = 0) ∧ (∀ z : ℂ, p z = 3 * z^2 + ((-24) * z + 51)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_root_l3728_372877


namespace NUMINAMATH_CALUDE_average_equation_solution_l3728_372868

theorem average_equation_solution (y : ℚ) : 
  (1 / 3 : ℚ) * ((y + 10) + (5 * y + 4) + (3 * y + 12)) = 6 * y - 8 → y = 50 / 9 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l3728_372868


namespace NUMINAMATH_CALUDE_largest_number_with_sum_18_l3728_372810

def is_valid_number (n : ℕ) : Prop :=
  (n.digits 10).sum = 18 ∧ (n.digits 10).Nodup

theorem largest_number_with_sum_18 :
  ∀ n : ℕ, is_valid_number n → n ≤ 843210 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_sum_18_l3728_372810


namespace NUMINAMATH_CALUDE_exponent_rules_l3728_372858

theorem exponent_rules (a : ℝ) : (a^3 * a^2 = a^5) ∧ (a^6 / a^2 = a^4) := by
  sorry

end NUMINAMATH_CALUDE_exponent_rules_l3728_372858


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3728_372893

theorem negative_fraction_comparison : -3/4 > -4/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3728_372893


namespace NUMINAMATH_CALUDE_homework_situations_l3728_372895

/-- The number of teachers who have assigned homework -/
def num_teachers : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of possible homework situations for all students -/
def total_situations : ℕ := num_teachers ^ num_students

theorem homework_situations :
  total_situations = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_homework_situations_l3728_372895


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3728_372857

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b) / (a + b) + (b * c) / (b + c) + (c * a) / (c + a) ≤ 
  (3 * (a * b + b * c + c * a)) / (2 * (a + b + c)) := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3728_372857


namespace NUMINAMATH_CALUDE_negative_x_squared_times_x_cubed_l3728_372802

theorem negative_x_squared_times_x_cubed (x : ℝ) : (-x)^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_squared_times_x_cubed_l3728_372802


namespace NUMINAMATH_CALUDE_pants_cost_rita_pants_cost_l3728_372853

/-- Calculates the cost of each pair of pants given Rita's shopping information -/
theorem pants_cost (initial_money : ℕ) (remaining_money : ℕ) (num_dresses : ℕ) (dress_cost : ℕ) 
  (num_pants : ℕ) (num_jackets : ℕ) (jacket_cost : ℕ) (transportation_cost : ℕ) : ℕ :=
  let total_spent := initial_money - remaining_money
  let dress_total := num_dresses * dress_cost
  let jacket_total := num_jackets * jacket_cost
  let pants_total := total_spent - dress_total - jacket_total - transportation_cost
  pants_total / num_pants

/-- Proves that each pair of pants costs $12 given Rita's shopping information -/
theorem rita_pants_cost : pants_cost 400 139 5 20 3 4 30 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_rita_pants_cost_l3728_372853


namespace NUMINAMATH_CALUDE_sum_coordinates_of_D_l3728_372835

/-- Given that N(3,5) is the midpoint of line segment CD and C has coordinates (1,10),
    prove that the sum of the coordinates of point D is 5. -/
theorem sum_coordinates_of_D (C D N : ℝ × ℝ) : 
  C = (1, 10) →
  N = (3, 5) →
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_D_l3728_372835


namespace NUMINAMATH_CALUDE_fraction_equality_l3728_372806

theorem fraction_equality (a b : ℚ) (h : (a - 2*b) / b = 3/5) : a / b = 13/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3728_372806


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3728_372897

-- Define the inverse proportionality relationship
def inverse_proportional (y x : ℝ) := ∃ k : ℝ, y = k / (x + 2)

-- Define the theorem
theorem inverse_proportion_problem (y x : ℝ) 
  (h1 : inverse_proportional y x) 
  (h2 : y = 3 ∧ x = -1) :
  (∀ x, y = 3 / (x + 2)) ∧ 
  (x = 0 → y = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3728_372897


namespace NUMINAMATH_CALUDE_division_remainder_l3728_372820

theorem division_remainder : ∃ r : ℕ, 
  12401 = 163 * 76 + r ∧ r < 163 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3728_372820


namespace NUMINAMATH_CALUDE_cube_sum_of_three_numbers_l3728_372886

theorem cube_sum_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_prod_eq : x * y + x * z + y * z = 1)
  (prod_eq : x * y * z = 1) :
  x^3 + y^3 + z^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_of_three_numbers_l3728_372886
