import Mathlib

namespace stratified_sample_theorem_l3090_309057

/-- Represents the number of female students in a stratified sample -/
def female_in_sample (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ) : ℚ :=
  (female_students : ℚ) * (sample_size : ℚ) / (total_students : ℚ)

/-- Theorem: In a school with 2100 total students (900 female),
    a stratified sample of 70 students will contain 30 female students -/
theorem stratified_sample_theorem :
  female_in_sample 2100 900 70 = 30 := by
  sorry

end stratified_sample_theorem_l3090_309057


namespace equation_solutions_l3090_309078

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 = 169 ↔ x = 15 ∨ x = -11) ∧
  (∀ x : ℝ, 3*(x - 3)^3 - 24 = 0 ↔ x = 5) := by
sorry

end equation_solutions_l3090_309078


namespace inequality_solution_set_l3090_309062

theorem inequality_solution_set (x : ℝ) : 
  1 / (x^2 + 1) < 5 / x + 21 / 10 ↔ x ∈ Set.Ioi (-1/2) ∪ Set.Ioi 0 \ {-1/2} :=
by sorry

end inequality_solution_set_l3090_309062


namespace quartic_equation_sum_l3090_309040

theorem quartic_equation_sum (a b c : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℕ+, 
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    (∀ x : ℝ, x^4 - 10*x^3 + a*x^2 + b*x + c = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) →
  a + b + c = 109 := by
sorry

end quartic_equation_sum_l3090_309040


namespace nested_square_root_value_l3090_309034

theorem nested_square_root_value :
  ∃ y : ℝ, y = Real.sqrt (2 - Real.sqrt (2 + y)) → y = (1 + Real.sqrt 5) / 2 := by
  sorry

end nested_square_root_value_l3090_309034


namespace trenton_fixed_earnings_l3090_309009

/-- Trenton's weekly earnings structure -/
structure WeeklyEarnings where
  fixed : ℝ
  commissionRate : ℝ
  salesGoal : ℝ
  totalEarningsGoal : ℝ

/-- Trenton's actual weekly earnings -/
def actualEarnings (w : WeeklyEarnings) : ℝ :=
  w.fixed + w.commissionRate * w.salesGoal

/-- Theorem: Trenton's fixed weekly earnings are $190 -/
theorem trenton_fixed_earnings :
  ∀ w : WeeklyEarnings,
  w.commissionRate = 0.04 →
  w.salesGoal = 7750 →
  w.totalEarningsGoal = 500 →
  actualEarnings w ≥ w.totalEarningsGoal →
  w.fixed = 190 := by
sorry

end trenton_fixed_earnings_l3090_309009


namespace decimal_sum_l3090_309027

theorem decimal_sum : 5.467 + 2.349 + 3.785 = 11.751 := by
  sorry

end decimal_sum_l3090_309027


namespace diagram_scale_l3090_309020

/-- Represents the scale of a diagram as a ratio of two natural numbers -/
structure Scale where
  numerator : ℕ
  denominator : ℕ

/-- Converts centimeters to millimeters -/
def cm_to_mm (cm : ℕ) : ℕ := cm * 10

theorem diagram_scale (actual_length_mm : ℕ) (diagram_length_cm : ℕ) :
  actual_length_mm = 4 →
  diagram_length_cm = 8 →
  ∃ (s : Scale), s.numerator = 20 ∧ s.denominator = 1 ∧
    cm_to_mm diagram_length_cm * s.denominator = actual_length_mm * s.numerator :=
by sorry

end diagram_scale_l3090_309020


namespace factorial_not_ending_1976_zeros_l3090_309054

theorem factorial_not_ending_1976_zeros (n : ℕ) : ∃ k : ℕ, n! % (10^k) ≠ 1976 * (10^k) :=
sorry

end factorial_not_ending_1976_zeros_l3090_309054


namespace b_value_proof_l3090_309037

theorem b_value_proof (a b c m : ℝ) (h : m = (c * a * b) / (a - b)) : 
  b = (m * a) / (m + c * a) := by
  sorry

end b_value_proof_l3090_309037


namespace smallest_of_three_consecutive_sum_90_l3090_309007

theorem smallest_of_three_consecutive_sum_90 (x y z : ℤ) :
  y = x + 1 ∧ z = y + 1 ∧ x + y + z = 90 → x = 29 := by
  sorry

end smallest_of_three_consecutive_sum_90_l3090_309007


namespace adam_ate_three_more_than_bill_l3090_309042

-- Define the number of pies eaten by each person
def sierra_pies : ℕ := 12
def total_pies : ℕ := 27

-- Define the relationships between the number of pies eaten
def bill_pies : ℕ := sierra_pies / 2
def adam_pies : ℕ := total_pies - sierra_pies - bill_pies

-- Theorem to prove
theorem adam_ate_three_more_than_bill :
  adam_pies = bill_pies + 3 := by
  sorry

end adam_ate_three_more_than_bill_l3090_309042


namespace jenny_research_time_l3090_309008

/-- Represents the time allocation for Jenny's school project -/
structure ProjectTime where
  total : ℕ
  proposal : ℕ
  report : ℕ

/-- Calculates the time spent on research given the project time allocation -/
def researchTime (pt : ProjectTime) : ℕ :=
  pt.total - pt.proposal - pt.report

/-- Theorem stating that Jenny spent 10 hours on research -/
theorem jenny_research_time :
  ∀ (pt : ProjectTime),
  pt.total = 20 ∧ pt.proposal = 2 ∧ pt.report = 8 →
  researchTime pt = 10 := by
  sorry

end jenny_research_time_l3090_309008


namespace wire_cutting_problem_l3090_309044

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) 
  (h1 : total_length = 28)
  (h2 : ratio = 2.00001 / 5) : 
  ∃ (shorter_piece : ℝ), 
    shorter_piece + ratio * shorter_piece = total_length ∧ 
    shorter_piece = 20 := by
sorry

end wire_cutting_problem_l3090_309044


namespace triangle_vector_relation_l3090_309066

-- Define the triangle ABC and vectors a and b
variable (A B C : EuclideanSpace ℝ (Fin 2))
variable (a b : EuclideanSpace ℝ (Fin 2))

-- Define points P and Q
variable (P : EuclideanSpace ℝ (Fin 2))
variable (Q : EuclideanSpace ℝ (Fin 2))

-- State the theorem
theorem triangle_vector_relation
  (h1 : B - A = a)
  (h2 : C - A = b)
  (h3 : P - A = (1/3) • (B - A))
  (h4 : Q - B = (1/3) • (C - B)) :
  Q - P = (1/3) • a + (1/3) • b := by sorry

end triangle_vector_relation_l3090_309066


namespace prob_both_red_given_one_red_l3090_309074

/-- Represents a card with two sides -/
structure Card where
  side1 : Bool  -- True for red, False for black
  side2 : Bool

/-- Represents the box of cards -/
def box : List Card := [
  {side1 := false, side2 := false},
  {side1 := false, side2 := false},
  {side1 := false, side2 := false},
  {side1 := false, side2 := false},
  {side1 := true,  side2 := false},
  {side1 := true,  side2 := false},
  {side1 := true,  side2 := true},
  {side1 := true,  side2 := true},
  {side1 := true,  side2 := true}
]

/-- The probability of drawing a card with a red side -/
def probRedSide : Rat := 8 / 18

/-- The probability that both sides are red, given that one side is red -/
theorem prob_both_red_given_one_red :
  (3 : Rat) / 4 = (List.filter (fun c => c.side1 ∧ c.side2) box).length / 
                  (List.filter (fun c => c.side1 ∨ c.side2) box).length :=
by sorry

end prob_both_red_given_one_red_l3090_309074


namespace sum_abcd_equals_negative_twenty_thirds_l3090_309083

theorem sum_abcd_equals_negative_twenty_thirds 
  (y a b c d : ℚ) 
  (h1 : y = a + 2)
  (h2 : y = b + 4)
  (h3 : y = c + 6)
  (h4 : y = d + 8)
  (h5 : y = a + b + c + d + 10) :
  a + b + c + d = -20 / 3 := by
sorry

end sum_abcd_equals_negative_twenty_thirds_l3090_309083


namespace twelfth_day_is_monday_l3090_309011

/-- Represents days of the week --/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with specific properties --/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  numberOfFridays : Nat
  numberOfDays : Nat
  firstDayNotFriday : firstDay ≠ DayOfWeek.Friday
  lastDayNotFriday : lastDay ≠ DayOfWeek.Friday
  exactlyFiveFridays : numberOfFridays = 5

/-- Function to determine the day of the week for a given day number --/
def dayOfWeekForDay (m : Month) (day : Nat) : DayOfWeek :=
  sorry

theorem twelfth_day_is_monday (m : Month) : 
  dayOfWeekForDay m 12 = DayOfWeek.Monday :=
sorry

end twelfth_day_is_monday_l3090_309011


namespace homework_problems_per_page_l3090_309000

theorem homework_problems_per_page 
  (total_problems : ℕ) 
  (finished_problems : ℕ) 
  (remaining_pages : ℕ) 
  (h1 : total_problems = 101)
  (h2 : finished_problems = 47)
  (h3 : remaining_pages = 6)
  (h4 : remaining_pages > 0)
  : (total_problems - finished_problems) / remaining_pages = 9 := by
  sorry

end homework_problems_per_page_l3090_309000


namespace sum_five_consecutive_odd_numbers_l3090_309033

theorem sum_five_consecutive_odd_numbers (n : ℤ) :
  let middle := 2 * n + 1
  let sum := (2 * n - 3) + (2 * n - 1) + (2 * n + 1) + (2 * n + 3) + (2 * n + 5)
  sum = 5 * middle :=
by sorry

end sum_five_consecutive_odd_numbers_l3090_309033


namespace apples_given_correct_l3090_309070

/-- The number of apples the farmer originally had -/
def original_apples : ℕ := 127

/-- The number of apples the farmer now has -/
def current_apples : ℕ := 39

/-- The number of apples given to the neighbor -/
def apples_given : ℕ := original_apples - current_apples

theorem apples_given_correct : apples_given = 88 := by sorry

end apples_given_correct_l3090_309070


namespace upstream_travel_time_l3090_309097

theorem upstream_travel_time
  (distance : ℝ)
  (downstream_time : ℝ)
  (current_speed : ℝ)
  (h1 : distance = 126)
  (h2 : downstream_time = 7)
  (h3 : current_speed = 2)
  : (distance / (distance / downstream_time - 2 * current_speed)) = 9 := by
  sorry

end upstream_travel_time_l3090_309097


namespace arrange_programs_count_l3090_309071

/-- The number of ways to arrange n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 5 programs with 2 consecutive -/
def arrange_programs : ℕ :=
  2 * permutations 4

theorem arrange_programs_count : arrange_programs = 48 := by
  sorry

end arrange_programs_count_l3090_309071


namespace reflected_ray_equation_l3090_309073

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 4*y + 12 = 0

-- Define the point A
def point_A : ℝ × ℝ := (-2, 3)

-- Define a line in slope-intercept form
def line (k b : ℝ) (x y : ℝ) : Prop :=
  y = k * x + b

-- Define the property of being tangent to the circle
def is_tangent_to_circle (k b : ℝ) : Prop :=
  ∃ (x y : ℝ), line k b x y ∧ circle_C x y

-- Define the property of passing through the reflection of A
def passes_through_reflection (k b : ℝ) : Prop :=
  line k b (-2) (-3)

-- State the theorem
theorem reflected_ray_equation :
  ∃ (k b : ℝ), 
    is_tangent_to_circle k b ∧
    passes_through_reflection k b ∧
    ((k = 4/3 ∧ b = -1/3) ∨ (k = 3/4 ∧ b = -3/2)) :=
sorry

end reflected_ray_equation_l3090_309073


namespace mack_journal_pages_l3090_309036

/-- The number of pages Mack writes on Monday -/
def monday_pages : ℕ := 60 / 30

/-- The number of pages Mack writes on Tuesday -/
def tuesday_pages : ℕ := 45 / 15

/-- The number of pages Mack writes on Wednesday -/
def wednesday_pages : ℕ := 5

/-- The total number of pages Mack writes from Monday to Wednesday -/
def total_pages : ℕ := monday_pages + tuesday_pages + wednesday_pages

theorem mack_journal_pages : total_pages = 10 := by sorry

end mack_journal_pages_l3090_309036


namespace trig_sum_problem_l3090_309045

theorem trig_sum_problem (α : Real) (h1 : 0 < α) (h2 : α < Real.pi) 
  (h3 : Real.sin α * Real.cos α = -1/2) : 
  1/(1 + Real.sin α) + 1/(1 + Real.cos α) = 4 := by
  sorry

end trig_sum_problem_l3090_309045


namespace lcm_of_12_and_18_l3090_309077

theorem lcm_of_12_and_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_of_12_and_18_l3090_309077


namespace mirror_area_is_2016_l3090_309089

/-- Calculates the area of a rectangular mirror inside a frame with rounded corners. -/
def mirror_area (frame_width : ℝ) (frame_height : ℝ) (frame_side_width : ℝ) : ℝ :=
  (frame_width - 2 * frame_side_width) * (frame_height - 2 * frame_side_width)

/-- Proves that the area of the mirror is 2016 cm² given the frame dimensions. -/
theorem mirror_area_is_2016 :
  mirror_area 50 70 7 = 2016 := by
  sorry

end mirror_area_is_2016_l3090_309089


namespace problem_solution_l3090_309068

theorem problem_solution (x y z : ℚ) (w : ℚ) : 
  x = (1 / 3) * y → 
  y = (1 / 4) * z → 
  z = 80 → 
  x = 20 / 3 ∧ w = x + y + z ∧ w = 320 / 3 := by
  sorry

end problem_solution_l3090_309068


namespace kendra_initial_money_l3090_309002

def wooden_toy_price : ℕ := 20
def hat_price : ℕ := 10
def wooden_toys_bought : ℕ := 2
def hats_bought : ℕ := 3
def change_received : ℕ := 30

theorem kendra_initial_money :
  wooden_toy_price * wooden_toys_bought + hat_price * hats_bought + change_received = 100 :=
by sorry

end kendra_initial_money_l3090_309002


namespace equation_solution_l3090_309010

def solution_set : Set (ℤ × ℤ) :=
  {(1, 12), (1, -12), (-9, 12), (-9, -12), (-4, 12), (-4, -12), (0, 0), (-8, 0), (-1, 0), (-7, 0)}

def satisfies_equation (p : ℤ × ℤ) : Prop :=
  let x := p.1
  let y := p.2
  x * (x + 1) * (x + 7) * (x + 8) = y^2

theorem equation_solution :
  ∀ p : ℤ × ℤ, satisfies_equation p ↔ p ∈ solution_set := by sorry

end equation_solution_l3090_309010


namespace coloring_books_total_l3090_309098

theorem coloring_books_total (initial : ℕ) (given_away : ℕ) (bought : ℕ) : 
  initial = 45 → given_away = 6 → bought = 20 → 
  initial - given_away + bought = 59 := by
sorry

end coloring_books_total_l3090_309098


namespace fraction_simplification_l3090_309056

theorem fraction_simplification (a b d : ℝ) (h : a^2 + d^2 - b^2 + 2*a*d ≠ 0) :
  (a^2 + b^2 + d^2 + 2*b*d) / (a^2 + d^2 - b^2 + 2*a*d) = 
  (a^2 + (b+d)^2) / ((a+d)^2 + a^2 - b^2) := by
sorry

end fraction_simplification_l3090_309056


namespace polynomial_factorization_l3090_309047

theorem polynomial_factorization (x : ℝ) :
  x^12 - 3*x^9 + 3*x^3 + 1 = (x+1)^4 * (x^2-x+1)^4 := by
  sorry

end polynomial_factorization_l3090_309047


namespace clapping_groups_l3090_309055

def number_of_people : ℕ := 4043
def claps_per_hand : ℕ := 2021

def valid_groups (n k : ℕ) : ℕ := Nat.choose n k

def invalid_groups (n m : ℕ) : ℕ := n * Nat.choose m 2

theorem clapping_groups :
  valid_groups number_of_people 3 - invalid_groups number_of_people claps_per_hand =
  valid_groups number_of_people 3 - number_of_people * valid_groups claps_per_hand 2 :=
by sorry

end clapping_groups_l3090_309055


namespace parabola_point_relationship_l3090_309076

/-- A parabola defined by the equation y = -x² + 2x + m --/
def parabola (x y m : ℝ) : Prop := y = -x^2 + 2*x + m

/-- Point A on the parabola --/
def point_A (y₁ m : ℝ) : Prop := parabola (-1) y₁ m

/-- Point B on the parabola --/
def point_B (y₂ m : ℝ) : Prop := parabola 1 y₂ m

/-- Point C on the parabola --/
def point_C (y₃ m : ℝ) : Prop := parabola 2 y₃ m

theorem parabola_point_relationship (y₁ y₂ y₃ m : ℝ) 
  (hA : point_A y₁ m) (hB : point_B y₂ m) (hC : point_C y₃ m) : 
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end parabola_point_relationship_l3090_309076


namespace vacuum_cleaner_price_difference_l3090_309032

/-- The in-store price of the vacuum cleaner in dollars -/
def in_store_price : ℚ := 150

/-- The cost of each online payment in dollars -/
def online_payment : ℚ := 35

/-- The number of online payments -/
def num_payments : ℕ := 4

/-- The one-time processing fee for online purchase in dollars -/
def processing_fee : ℚ := 12

/-- The difference in cents between online and in-store purchase -/
def price_difference_cents : ℤ := 200

theorem vacuum_cleaner_price_difference :
  (num_payments * online_payment + processing_fee - in_store_price) * 100 = price_difference_cents := by
  sorry

end vacuum_cleaner_price_difference_l3090_309032


namespace one_point_of_contact_condition_l3090_309030

/-- Two equations have exactly one point of contact -/
def has_one_point_of_contact (f g : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, f x = g x

/-- The parabola y = x^2 + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The line y = 4x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := 4*x + c

/-- The theorem stating the condition for one point of contact -/
theorem one_point_of_contact_condition :
  ∀ c : ℝ, has_one_point_of_contact f (g c) ↔ c = -3 := by sorry

end one_point_of_contact_condition_l3090_309030


namespace sixtieth_number_is_sixteen_l3090_309067

/-- Defines the number of elements in each row of the sequence -/
def elementsInRow (n : ℕ) : ℕ := 2 * n

/-- Defines the value of elements in each row of the sequence -/
def valueInRow (n : ℕ) : ℕ := 2 * n

/-- Calculates the cumulative sum of elements up to and including row n -/
def cumulativeSum (n : ℕ) : ℕ :=
  (List.range n).map elementsInRow |>.sum

/-- Finds the row number for a given position in the sequence -/
def findRow (position : ℕ) : ℕ :=
  (List.range position).find? (fun n => cumulativeSum (n + 1) ≥ position)
    |>.getD 0

/-- The main theorem stating that the 60th number in the sequence is 16 -/
theorem sixtieth_number_is_sixteen :
  valueInRow (findRow 60 + 1) = 16 := by
  sorry

#eval valueInRow (findRow 60 + 1)

end sixtieth_number_is_sixteen_l3090_309067


namespace building_floors_l3090_309021

-- Define the number of floors in each building
def alexie_floors : ℕ := sorry
def baptiste_floors : ℕ := sorry

-- Define the total number of bathrooms and bedrooms
def total_bathrooms : ℕ := 25
def total_bedrooms : ℕ := 18

-- State the theorem
theorem building_floors :
  (3 * alexie_floors + 4 * baptiste_floors = total_bathrooms) ∧
  (2 * alexie_floors + 3 * baptiste_floors = total_bedrooms) →
  alexie_floors = 3 ∧ baptiste_floors = 4 := by
  sorry

end building_floors_l3090_309021


namespace range_of_a_l3090_309016

theorem range_of_a (a : ℝ) : 
  ((∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
   (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0)) ↔ 
  (a ≤ -2 ∨ a = 1) := by
sorry

end range_of_a_l3090_309016


namespace sqrt_product_simplification_l3090_309006

theorem sqrt_product_simplification : Real.sqrt 12 * Real.sqrt 75 = 30 := by
  sorry

end sqrt_product_simplification_l3090_309006


namespace claire_cooking_time_l3090_309013

/-- Represents Claire's daily schedule -/
structure DailySchedule where
  total_hours : ℕ
  sleep_hours : ℕ
  clean_hours : ℕ
  craft_hours : ℕ
  tailor_hours : ℕ
  cook_hours : ℕ

/-- Claire's schedule satisfies the given conditions -/
def is_valid_schedule (s : DailySchedule) : Prop :=
  s.total_hours = 24 ∧
  s.sleep_hours = 8 ∧
  s.clean_hours = 4 ∧
  s.craft_hours = 5 ∧
  s.tailor_hours = s.craft_hours ∧
  s.total_hours = s.sleep_hours + s.clean_hours + s.craft_hours + s.tailor_hours + s.cook_hours

theorem claire_cooking_time (s : DailySchedule) (h : is_valid_schedule s) : s.cook_hours = 2 := by
  sorry

end claire_cooking_time_l3090_309013


namespace max_time_proof_l3090_309059

/-- The number of digits in the lock combination -/
def num_digits : ℕ := 3

/-- The number of possible values for each digit (0 to 8, inclusive) -/
def digits_range : ℕ := 9

/-- The time in seconds required for each trial -/
def time_per_trial : ℕ := 3

/-- Calculates the maximum time in seconds required to try all combinations -/
def max_time_seconds : ℕ := digits_range ^ num_digits * time_per_trial

/-- Theorem: The maximum time required to try all combinations is 2187 seconds -/
theorem max_time_proof : max_time_seconds = 2187 := by
  sorry

end max_time_proof_l3090_309059


namespace max_tiles_on_floor_l3090_309080

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit in one dimension -/
def maxTilesInDimension (floorDim tileADim tileBDim : ℕ) : ℕ :=
  max (floorDim / tileADim) (floorDim / tileBDim)

/-- Calculates the total number of tiles for a given orientation -/
def totalTiles (floor tile : Dimensions) : ℕ :=
  (maxTilesInDimension floor.length tile.length tile.width) *
  (maxTilesInDimension floor.width tile.length tile.width)

/-- The main theorem stating the maximum number of tiles that can be accommodated -/
theorem max_tiles_on_floor :
  let floor := Dimensions.mk 180 120
  let tile := Dimensions.mk 25 16
  (max (totalTiles floor tile) (totalTiles floor (Dimensions.mk tile.width tile.length))) = 49 := by
  sorry

end max_tiles_on_floor_l3090_309080


namespace fifth_term_sum_l3090_309094

def sequence_a (n : ℕ) : ℕ := 2 * n - 1

def sequence_b (n : ℕ) : ℕ := 2^(n-1)

def sequence_c (n : ℕ) : ℕ := sequence_a n * sequence_b n

theorem fifth_term_sum :
  sequence_a 5 + sequence_b 5 + sequence_c 5 = 169 := by
sorry

end fifth_term_sum_l3090_309094


namespace original_number_is_429_l3090_309095

/-- Given a three-digit number abc, this function returns the sum of all its permutations -/
def sum_of_permutations (a b c : Nat) : Nat :=
  100 * a + 10 * b + c +
  100 * a + 10 * c + b +
  100 * b + 10 * a + c +
  100 * b + 10 * c + a +
  100 * c + 10 * a + b +
  100 * c + 10 * b + a

/-- The sum of all permutations of the three-digit number we're looking for -/
def S : Nat := 4239

/-- Theorem stating that the original three-digit number is 429 -/
theorem original_number_is_429 :
  ∃ (a b c : Nat), a < 10 ∧ b < 10 ∧ c < 10 ∧ a ≠ 0 ∧ sum_of_permutations a b c = S ∧ a = 4 ∧ b = 2 ∧ c = 9 := by
  sorry


end original_number_is_429_l3090_309095


namespace student_number_problem_l3090_309051

theorem student_number_problem (x : ℝ) : 4 * x - 142 = 110 → x = 63 := by
  sorry

end student_number_problem_l3090_309051


namespace sum_of_solutions_equals_zero_l3090_309093

theorem sum_of_solutions_equals_zero : 
  ∃ (S : Finset Int), 
    (∀ x ∈ S, x^4 - 13*x^2 + 36 = 0) ∧ 
    (∀ x : Int, x^4 - 13*x^2 + 36 = 0 → x ∈ S) ∧ 
    (S.sum id = 0) := by
  sorry

end sum_of_solutions_equals_zero_l3090_309093


namespace right_triangle_side_lengths_l3090_309087

/-- Represents a right-angled triangle with side lengths a, b, and c (hypotenuse) -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2

theorem right_triangle_side_lengths
  (t : RightTriangle)
  (leg_a : t.a = 10)
  (sum_squares : t.a^2 + t.b^2 + t.c^2 = 2050) :
  t.b = Real.sqrt 925 ∧ t.c = Real.sqrt 1025 := by
  sorry


end right_triangle_side_lengths_l3090_309087


namespace number_puzzle_l3090_309026

theorem number_puzzle : 
  ∀ x : ℚ, (x / 7 - x / 11 = 100) → x = 1925 := by
  sorry

end number_puzzle_l3090_309026


namespace complex_modulus_range_l3090_309060

theorem complex_modulus_range (a : ℝ) (z : ℂ) (h1 : 0 < a) (h2 : a < 2) (h3 : z.re = a) (h4 : z.im = 1) :
  1 < Complex.abs z ∧ Complex.abs z < Real.sqrt 5 := by
  sorry

end complex_modulus_range_l3090_309060


namespace basketball_game_scores_l3090_309025

/-- Represents the quarterly scores of a team -/
structure QuarterlyScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if the scores form an increasing geometric sequence -/
def is_increasing_geometric (s : QuarterlyScores) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if the scores form an increasing arithmetic sequence -/
def is_increasing_arithmetic (s : QuarterlyScores) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- Calculates the total score for a team -/
def total_score (s : QuarterlyScores) : ℕ :=
  s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the first half score for a team -/
def first_half_score (s : QuarterlyScores) : ℕ :=
  s.q1 + s.q2

theorem basketball_game_scores :
  ∀ (raiders wildcats : QuarterlyScores),
    is_increasing_geometric raiders →
    is_increasing_arithmetic wildcats →
    raiders.q1 = wildcats.q1 + 1 →
    total_score raiders = total_score wildcats + 2 →
    total_score raiders ≤ 100 →
    total_score wildcats ≤ 100 →
    first_half_score raiders + first_half_score wildcats = 25 := by
  sorry

end basketball_game_scores_l3090_309025


namespace video_game_lives_calculation_l3090_309053

/-- Calculate the total number of lives for remaining players in a video game --/
theorem video_game_lives_calculation (initial_players : ℕ) (initial_lives : ℕ) 
  (quit_players : ℕ) (powerup_players : ℕ) (penalty_players : ℕ) 
  (powerup_lives : ℕ) (penalty_lives : ℕ) : 
  initial_players = 15 →
  initial_lives = 10 →
  quit_players = 5 →
  powerup_players = 4 →
  penalty_players = 6 →
  powerup_lives = 3 →
  penalty_lives = 2 →
  (initial_players - quit_players) * initial_lives + 
    powerup_players * powerup_lives - penalty_players * penalty_lives = 100 := by
  sorry


end video_game_lives_calculation_l3090_309053


namespace park_shape_l3090_309049

theorem park_shape (total_cost : ℕ) (cost_per_side : ℕ) (h1 : total_cost = 224) (h2 : cost_per_side = 56) :
  total_cost / cost_per_side = 4 :=
by sorry

end park_shape_l3090_309049


namespace sum_equation_implies_n_value_l3090_309050

theorem sum_equation_implies_n_value : 
  990 + 992 + 994 + 996 + 998 = 5000 - N → N = 30 := by
  sorry

end sum_equation_implies_n_value_l3090_309050


namespace quadratic_max_iff_a_neg_l3090_309084

/-- A quadratic function -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Definition of having a maximum value for a quadratic function -/
def has_maximum (f : QuadraticFunction) : Prop :=
  ∃ x₀ : ℝ, ∀ x : ℝ, f.a * x^2 + f.b * x + f.c ≤ f.a * x₀^2 + f.b * x₀ + f.c

/-- Theorem: A quadratic function has a maximum value if and only if a < 0 -/
theorem quadratic_max_iff_a_neg (f : QuadraticFunction) :
  has_maximum f ↔ f.a < 0 :=
sorry

end quadratic_max_iff_a_neg_l3090_309084


namespace initial_chips_count_l3090_309004

/-- The number of tortilla chips Nancy initially had in her bag -/
def initial_chips : ℕ := sorry

/-- The number of tortilla chips Nancy gave to her brother -/
def chips_to_brother : ℕ := 7

/-- The number of tortilla chips Nancy gave to her sister -/
def chips_to_sister : ℕ := 5

/-- The number of tortilla chips Nancy kept for herself -/
def chips_kept : ℕ := 10

/-- Theorem stating that the initial number of chips is 22 -/
theorem initial_chips_count : initial_chips = 22 := by sorry

end initial_chips_count_l3090_309004


namespace integral_equals_ten_l3090_309090

theorem integral_equals_ten (k : ℝ) : 
  (∫ x in (0 : ℝ)..2, 3 * x^2 + k) = 10 → k = 1 := by
  sorry

end integral_equals_ten_l3090_309090


namespace two_face_painted_count_l3090_309065

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  is_painted : Bool

/-- Represents a cube that has been cut into unit cubes -/
structure CutCube (n : ℕ) extends PaintedCube n where
  unit_cubes : Fin n → Fin n → Fin n → PaintedCube 1

/-- Returns the number of unit cubes with exactly two painted faces -/
def count_two_face_painted (c : CutCube 4) : ℕ := sorry

/-- Theorem stating that a 4-inch painted cube cut into 1-inch cubes has 24 cubes with exactly two painted faces -/
theorem two_face_painted_count (c : CutCube 4) : count_two_face_painted c = 24 := by sorry

end two_face_painted_count_l3090_309065


namespace floor_times_self_162_l3090_309082

theorem floor_times_self_162 (x : ℝ) : ⌊x⌋ * x = 162 → x = 13.5 := by
  sorry

end floor_times_self_162_l3090_309082


namespace arithmetic_expression_equality_l3090_309075

theorem arithmetic_expression_equality : (24 / (8 + 2 - 6)) * 4 = 24 := by
  sorry

end arithmetic_expression_equality_l3090_309075


namespace circular_arrangement_size_l3090_309052

/-- A circular arrangement of people with the property that the 7th person is directly opposite the 18th person -/
structure CircularArrangement where
  n : ℕ  -- Total number of people
  seventh_opposite_eighteenth : n ≥ 18 ∧ (18 - 7) * 2 + 2 = n

/-- The theorem stating that in a circular arrangement where the 7th person is directly opposite the 18th person, the total number of people is 24 -/
theorem circular_arrangement_size (c : CircularArrangement) : c.n = 24 := by
  sorry

end circular_arrangement_size_l3090_309052


namespace greatest_integer_x_cubed_le_27_l3090_309079

theorem greatest_integer_x_cubed_le_27 :
  ∃ (x : ℕ), x > 0 ∧ (x^6 / x^3 : ℚ) ≤ 27 ∧ ∀ (y : ℕ), y > x → (y^6 / y^3 : ℚ) > 27 :=
by
  sorry

end greatest_integer_x_cubed_le_27_l3090_309079


namespace selection_with_at_least_one_boy_l3090_309092

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of boys in the group -/
def num_boys : ℕ := 8

/-- The number of girls in the group -/
def num_girls : ℕ := 6

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of people to be selected -/
def selection_size : ℕ := 3

theorem selection_with_at_least_one_boy :
  choose total_people selection_size - choose num_girls selection_size = 344 := by
  sorry

end selection_with_at_least_one_boy_l3090_309092


namespace sin_cos_pi_12_l3090_309035

theorem sin_cos_pi_12 : 2 * Real.sin (π / 12) * Real.cos (π / 12) = 1 / 2 := by
  sorry

end sin_cos_pi_12_l3090_309035


namespace hexagon_rearrangement_l3090_309005

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with given side length -/
structure Square where
  side : ℝ

/-- Represents the problem setup -/
structure HexagonProblem where
  original_rectangle : Rectangle
  resulting_square : Square
  is_valid : Prop

/-- The theorem stating the relationship between the original rectangle and the resulting square -/
theorem hexagon_rearrangement (p : HexagonProblem) 
  (h1 : p.original_rectangle.length = 9)
  (h2 : p.original_rectangle.width = 16)
  (h3 : p.is_valid)
  (h4 : p.original_rectangle.length * p.original_rectangle.width = p.resulting_square.side ^ 2) :
  p.resulting_square.side / 2 = 6 := by sorry

end hexagon_rearrangement_l3090_309005


namespace geometric_mean_of_sqrt2_plus_minus_one_l3090_309014

theorem geometric_mean_of_sqrt2_plus_minus_one :
  let a := Real.sqrt 2 + 1
  let b := Real.sqrt 2 - 1
  let geometric_mean := Real.sqrt (a * b)
  geometric_mean = 1 ∨ geometric_mean = -1 := by
sorry

end geometric_mean_of_sqrt2_plus_minus_one_l3090_309014


namespace guaranteed_babysitting_hours_is_eight_l3090_309039

/-- Calculates the number of guaranteed babysitting hours on Saturday given Donna's work schedule and earnings. -/
def guaranteed_babysitting_hours (
  dog_walking_hours : ℕ)
  (dog_walking_rate : ℚ)
  (dog_walking_days : ℕ)
  (card_shop_hours : ℕ)
  (card_shop_rate : ℚ)
  (card_shop_days : ℕ)
  (babysitting_rate : ℚ)
  (total_earnings : ℚ) : ℚ :=
  let dog_walking_earnings := ↑dog_walking_hours * dog_walking_rate * ↑dog_walking_days
  let card_shop_earnings := ↑card_shop_hours * card_shop_rate * ↑card_shop_days
  let other_earnings := dog_walking_earnings + card_shop_earnings
  let babysitting_earnings := total_earnings - other_earnings
  babysitting_earnings / babysitting_rate

theorem guaranteed_babysitting_hours_is_eight :
  guaranteed_babysitting_hours 2 10 5 2 (25/2) 5 10 305 = 8 := by
  sorry

end guaranteed_babysitting_hours_is_eight_l3090_309039


namespace rhombus_area_in_square_l3090_309088

/-- The area of a rhombus formed by the intersection of two equilateral triangles in a square -/
theorem rhombus_area_in_square (square_side : ℝ) (h_square_side : square_side = 4) :
  let triangle_side : ℝ := square_side
  let triangle_height : ℝ := (Real.sqrt 3 / 2) * triangle_side
  let rhombus_diagonal1 : ℝ := square_side
  let rhombus_diagonal2 : ℝ := triangle_height
  let rhombus_area : ℝ := (1 / 2) * rhombus_diagonal1 * rhombus_diagonal2
  rhombus_area = 4 * Real.sqrt 3 := by
  sorry

end rhombus_area_in_square_l3090_309088


namespace smallest_representable_number_l3090_309038

/-- Sum of decimal digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number can be represented as the sum of k positive integers
    with the same sum of decimal digits -/
def representable (n k : ℕ) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ n = k * d ∧ sum_of_digits d = sum_of_digits n / k

theorem smallest_representable_number :
  (∀ m : ℕ, m < 10010 → ¬(representable m 2002 ∧ representable m 2003)) ∧
  (representable 10010 2002 ∧ representable 10010 2003) := by sorry

end smallest_representable_number_l3090_309038


namespace optimal_bicycle_dropoff_l3090_309029

/-- Represents the problem of finding the optimal bicycle drop-off point --/
theorem optimal_bicycle_dropoff
  (total_distance : ℝ)
  (walking_speed : ℝ)
  (biking_speed : ℝ)
  (h_total_distance : total_distance = 30)
  (h_walking_speed : walking_speed = 5)
  (h_biking_speed : biking_speed = 20)
  (h_speeds_positive : 0 < walking_speed ∧ 0 < biking_speed)
  (h_speeds_order : walking_speed < biking_speed) :
  ∃ (x : ℝ),
    x = 5 ∧
    (∀ (y : ℝ),
      0 ≤ y ∧ y ≤ total_distance →
      max
        ((total_distance - y) / biking_speed + y / walking_speed)
        ((total_distance / 2 - y) / walking_speed + y / biking_speed)
      ≥
      max
        ((total_distance - x) / biking_speed + x / walking_speed)
        ((total_distance / 2 - x) / walking_speed + x / biking_speed)) :=
by
  sorry

end optimal_bicycle_dropoff_l3090_309029


namespace parallel_lines_b_value_l3090_309041

theorem parallel_lines_b_value (b : ℝ) : 
  (∀ x y : ℝ, 4 * y - 3 * x - 2 = 0 ↔ y = (3/4) * x + 1/2) →
  (∀ x y : ℝ, 6 * y + b * x + 1 = 0 ↔ y = (-b/6) * x - 1/6) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (4 * y₁ - 3 * x₁ - 2 = 0 ∧ 6 * y₂ + b * x₂ + 1 = 0) → 
    (y₂ - y₁) / (x₂ - x₁) = (3/4)) →
  b = -4.5 := by
sorry

end parallel_lines_b_value_l3090_309041


namespace average_fee_is_4_6_l3090_309072

/-- Represents the delivery statistics for a delivery person in December -/
structure DeliveryStats where
  short_distance_percent : ℝ  -- Percentage of deliveries ≤ 3 km
  long_distance_percent : ℝ   -- Percentage of deliveries > 3 km
  short_distance_fee : ℝ      -- Fee for deliveries ≤ 3 km
  long_distance_fee : ℝ       -- Fee for deliveries > 3 km

/-- Calculates the average delivery fee per order -/
def average_delivery_fee (stats : DeliveryStats) : ℝ :=
  stats.short_distance_percent * stats.short_distance_fee +
  stats.long_distance_percent * stats.long_distance_fee

/-- Theorem stating that the average delivery fee is 4.6 yuan for the given statistics -/
theorem average_fee_is_4_6 (stats : DeliveryStats) 
  (h1 : stats.short_distance_percent = 0.7)
  (h2 : stats.long_distance_percent = 0.3)
  (h3 : stats.short_distance_fee = 4)
  (h4 : stats.long_distance_fee = 6) :
  average_delivery_fee stats = 4.6 := by
  sorry

end average_fee_is_4_6_l3090_309072


namespace group_four_frequency_and_relative_frequency_l3090_309023

/-- Given a sample with capacity 50 and frequencies for groups 1, 2, 3, and 5,
    prove the frequency and relative frequency of group 4 -/
theorem group_four_frequency_and_relative_frequency 
  (total_capacity : ℕ) 
  (freq_1 freq_2 freq_3 freq_5 : ℕ) 
  (h1 : total_capacity = 50)
  (h2 : freq_1 = 8)
  (h3 : freq_2 = 11)
  (h4 : freq_3 = 10)
  (h5 : freq_5 = 9) :
  ∃ (freq_4 : ℕ) (rel_freq_4 : ℚ),
    freq_4 = total_capacity - (freq_1 + freq_2 + freq_3 + freq_5) ∧
    rel_freq_4 = freq_4 / total_capacity ∧
    freq_4 = 12 ∧
    rel_freq_4 = 0.24 := by
  sorry

end group_four_frequency_and_relative_frequency_l3090_309023


namespace johns_out_of_pocket_expense_l3090_309043

/-- Calculate the amount John paid out of pocket for his new computer setup --/
theorem johns_out_of_pocket_expense :
  let computer_cost : ℚ := 1200
  let computer_discount : ℚ := 0.15
  let chair_cost : ℚ := 300
  let chair_discount : ℚ := 0.10
  let accessories_cost : ℚ := 350
  let sales_tax_rate : ℚ := 0.08
  let playstation_value : ℚ := 500
  let playstation_discount : ℚ := 0.30
  let bicycle_sale : ℚ := 100

  let discounted_computer := computer_cost * (1 - computer_discount)
  let discounted_chair := chair_cost * (1 - chair_discount)
  let total_before_tax := discounted_computer + discounted_chair + accessories_cost
  let total_with_tax := total_before_tax * (1 + sales_tax_rate)
  let sold_items := playstation_value * (1 - playstation_discount) + bicycle_sale
  let out_of_pocket := total_with_tax - sold_items

  out_of_pocket = 1321.20
  := by sorry

end johns_out_of_pocket_expense_l3090_309043


namespace zoo_treats_problem_l3090_309022

/-- The percentage of pieces of bread Jane brings compared to treats -/
def jane_bread_percentage (jane_treats : ℕ) (jane_bread : ℕ) (wanda_treats : ℕ) (wanda_bread : ℕ) : ℚ :=
  (jane_bread : ℚ) / (jane_treats : ℚ) * 100

/-- The problem statement -/
theorem zoo_treats_problem (jane_treats : ℕ) (jane_bread : ℕ) (wanda_treats : ℕ) (wanda_bread : ℕ) :
  wanda_treats = jane_treats / 2 →
  wanda_bread = 3 * wanda_treats →
  wanda_bread = 90 →
  jane_treats + jane_bread + wanda_treats + wanda_bread = 225 →
  jane_bread_percentage jane_treats jane_bread wanda_treats wanda_bread = 75 := by
  sorry

end zoo_treats_problem_l3090_309022


namespace perpendicular_sufficient_not_necessary_l3090_309001

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the "lies in" relation between a line and a plane
variable (lies_in : Line → Plane → Prop)

-- Define our specific objects
variable (l m n : Line) (α : Plane)

-- State the theorem
theorem perpendicular_sufficient_not_necessary :
  (lies_in m α) → 
  (lies_in n α) → 
  (∀ x y : Line, lies_in x α → lies_in y α → perp_line_plane l α → perp_line_line l x ∧ perp_line_line l y) ∧ 
  (∃ x y : Line, lies_in x α → lies_in y α → perp_line_line l x ∧ perp_line_line l y ∧ ¬perp_line_plane l α) :=
by sorry

end perpendicular_sufficient_not_necessary_l3090_309001


namespace businessmen_neither_coffee_nor_tea_l3090_309024

theorem businessmen_neither_coffee_nor_tea 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (both : ℕ) 
  (h1 : total = 25) 
  (h2 : coffee = 12) 
  (h3 : tea = 10) 
  (h4 : both = 5) : 
  total - (coffee + tea - both) = 8 := by
  sorry

end businessmen_neither_coffee_nor_tea_l3090_309024


namespace unique_zero_implies_m_equals_one_l3090_309085

/-- A quadratic function with coefficient 1 for x^2, 2 for x, and m as the constant term -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

/-- The discriminant of the quadratic function -/
def discriminant (m : ℝ) : ℝ := 4 - 4*m

theorem unique_zero_implies_m_equals_one (m : ℝ) :
  (∃! x, quadratic m x = 0) → m = 1 := by
  sorry

end unique_zero_implies_m_equals_one_l3090_309085


namespace prime_product_sum_relation_l3090_309015

theorem prime_product_sum_relation (a b c d : ℕ) :
  (Prime a ∧ Prime b ∧ Prime c ∧ Prime d) →
  (a * b * c * d = 11 * (a + b + c + d)) →
  (a + b + c + d = 20) := by
sorry

end prime_product_sum_relation_l3090_309015


namespace triangle_sum_l3090_309096

theorem triangle_sum (AC BC : ℝ) (HE HD : ℝ) (a b : ℝ) :
  AC = 16.25 →
  BC = 13.75 →
  HE = 6 →
  HD = 3 →
  b - a = 5 →
  BC * (HD + b) = AC * (HE + a) →
  a + b = 15 := by
  sorry

end triangle_sum_l3090_309096


namespace b_work_alone_days_l3090_309091

/-- The number of days A takes to finish the work alone -/
def A_days : ℝ := 5

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B takes to finish the remaining work after A leaves -/
def B_remaining_days : ℝ := 7

/-- The number of days B takes to finish the work alone -/
def B_days : ℝ := 15

/-- Theorem stating that given the conditions, B can finish the work alone in 15 days -/
theorem b_work_alone_days :
  (together_days * (1 / A_days + 1 / B_days) + B_remaining_days * (1 / B_days) = 1) :=
sorry

end b_work_alone_days_l3090_309091


namespace prime_power_sum_l3090_309017

theorem prime_power_sum (p q r : ℕ) : 
  p.Prime → q.Prime → r.Prime → p^q + q^p = r → 
  ((p = 2 ∧ q = 3 ∧ r = 17) ∨ (p = 3 ∧ q = 2 ∧ r = 17)) :=
sorry

end prime_power_sum_l3090_309017


namespace f_uniqueness_and_fixed_points_l3090_309019

def is_prime (p : ℕ) : Prop := Nat.Prime p

def f_conditions (f : ℕ → ℕ) : Prop :=
  (∀ p, is_prime p → f p = 1) ∧
  (∀ a b, f (a * b) = a * f b + f a * b)

theorem f_uniqueness_and_fixed_points (f : ℕ → ℕ) (h : f_conditions f) :
  (∀ g, f_conditions g → f = g) ∧
  (∀ n, n = f n ↔ ∃ p, is_prime p ∧ n = p^p) :=
sorry

end f_uniqueness_and_fixed_points_l3090_309019


namespace dice_edge_length_l3090_309048

/-- The volume of the dice in cubic centimeters -/
def dice_volume : ℝ := 8

/-- The conversion factor from centimeters to millimeters -/
def cm_to_mm : ℝ := 10

/-- The length of one edge of the dice in millimeters -/
def edge_length_mm : ℝ := 20

theorem dice_edge_length :
  edge_length_mm = (dice_volume ^ (1/3 : ℝ)) * cm_to_mm :=
by sorry

end dice_edge_length_l3090_309048


namespace square_area_ratio_l3090_309028

theorem square_area_ratio : 
  ∀ (a b : ℝ), 
  (4 * a = 16 * b) →  -- Perimeter relation
  (a = 2 * b + 5) →   -- Side length relation
  (a^2 / b^2 = 16) := by  -- Area ratio
sorry

end square_area_ratio_l3090_309028


namespace nine_digit_square_impossibility_l3090_309012

theorem nine_digit_square_impossibility (n : ℕ) : 
  (100000000 ≤ n ∧ n < 1000000000) →  -- n is a nine-digit number
  (∃ (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ), 
    n = 100000000 * d1 + 10000000 * d2 + 1000000 * d3 + 100000 * d4 + 
        10000 * d5 + 1000 * d6 + 100 * d7 + 10 * d8 + 5 ∧
    ({d1, d2, d3, d4, d5, d6, d7, d8, 5} : Finset ℕ) = Finset.range 9) →  -- n uses all digits from 1 to 9 and ends in 5
  ¬∃ (m : ℕ), n = m^2 :=  -- n is not a perfect square
by
  sorry

end nine_digit_square_impossibility_l3090_309012


namespace movie_tickets_bought_l3090_309099

def computer_game_cost : ℕ := 66
def movie_ticket_cost : ℕ := 12
def total_spent : ℕ := 102

theorem movie_tickets_bought : 
  ∃ (x : ℕ), x * movie_ticket_cost + computer_game_cost = total_spent ∧ x = 3 := by
  sorry

end movie_tickets_bought_l3090_309099


namespace arithmetic_sequence_common_difference_l3090_309061

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmeticSequence a) 
  (h_a1 : a 1 = 3) 
  (h_a3 : a 3 = 7) : 
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end arithmetic_sequence_common_difference_l3090_309061


namespace erik_money_left_l3090_309058

/-- The amount of money Erik started with -/
def initial_money : ℕ := 86

/-- The number of loaves of bread Erik bought -/
def bread_quantity : ℕ := 3

/-- The cost of each loaf of bread -/
def bread_cost : ℕ := 3

/-- The number of cartons of orange juice Erik bought -/
def juice_quantity : ℕ := 3

/-- The cost of each carton of orange juice -/
def juice_cost : ℕ := 6

/-- The theorem stating how much money Erik has left -/
theorem erik_money_left : 
  initial_money - (bread_quantity * bread_cost + juice_quantity * juice_cost) = 59 := by
  sorry

end erik_money_left_l3090_309058


namespace coin_bag_total_l3090_309081

theorem coin_bag_total (p : ℕ) : ∃ (p : ℕ), 
  (0.01 * p + 0.05 * (3 * p) + 0.10 * (4 * 3 * p) : ℚ) = 408 := by
  sorry

end coin_bag_total_l3090_309081


namespace platform_length_l3090_309063

/-- Given a train that passes a pole and a platform, prove the length of the platform. -/
theorem platform_length
  (train_length : ℝ)
  (pole_time : ℝ)
  (platform_time : ℝ)
  (h1 : train_length = 120)
  (h2 : pole_time = 11)
  (h3 : platform_time = 22) :
  (train_length * platform_time / pole_time) - train_length = 120 :=
by sorry

end platform_length_l3090_309063


namespace map_distance_l3090_309064

/-- Given a map scale where 0.6 cm represents 6.6 km, and an actual distance of 885.5 km
    between two points, the distance between these points on the map is 80.5 cm. -/
theorem map_distance (scale_map : Real) (scale_actual : Real) (actual_distance : Real) :
  scale_map = 0.6 ∧ scale_actual = 6.6 ∧ actual_distance = 885.5 →
  (actual_distance / (scale_actual / scale_map)) = 80.5 := by
  sorry

#check map_distance

end map_distance_l3090_309064


namespace train_speed_is_25_l3090_309069

-- Define the train and its properties
structure Train :=
  (speed : ℝ)
  (length : ℝ)

-- Define the tunnels
def tunnel1_length : ℝ := 85
def tunnel2_length : ℝ := 160
def tunnel1_time : ℝ := 5
def tunnel2_time : ℝ := 8

-- Theorem statement
theorem train_speed_is_25 (t : Train) :
  (tunnel1_length + t.length) / tunnel1_time = t.speed →
  (tunnel2_length + t.length) / tunnel2_time = t.speed →
  t.speed = 25 := by
  sorry


end train_speed_is_25_l3090_309069


namespace arithmetic_square_root_of_9_l3090_309046

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l3090_309046


namespace typists_productivity_l3090_309018

/-- Given that 10 typists can type 20 letters in 20 minutes, 
    prove that 40 typists working at the same rate for 1 hour will complete 240 letters. -/
theorem typists_productivity 
  (base_typists : ℕ) 
  (base_letters : ℕ) 
  (base_minutes : ℕ) 
  (new_typists : ℕ) 
  (new_minutes : ℕ)
  (h1 : base_typists = 10)
  (h2 : base_letters = 20)
  (h3 : base_minutes = 20)
  (h4 : new_typists = 40)
  (h5 : new_minutes = 60) :
  (new_typists * new_minutes * base_letters) / (base_typists * base_minutes) = 240 :=
sorry

end typists_productivity_l3090_309018


namespace min_stamps_proof_l3090_309003

/-- The minimum number of stamps needed to make 60 cents using only 5 cent and 6 cent stamps -/
def min_stamps : ℕ := 10

/-- The value of stamps in cents -/
def total_value : ℕ := 60

/-- Proves that the minimum number of stamps needed to make 60 cents using only 5 cent and 6 cent stamps is 10 -/
theorem min_stamps_proof :
  ∀ c f : ℕ, 5 * c + 6 * f = total_value → c + f ≥ min_stamps :=
sorry

end min_stamps_proof_l3090_309003


namespace vector_equation_result_l3090_309086

def a : ℝ × ℝ := (2, -3)
def b : ℝ × ℝ := (1, 2)
def c : ℝ × ℝ := (9, 4)

theorem vector_equation_result (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : c = (m * a.1 + n * b.1, m * a.2 + n * b.2)) : 
  1/m + 1/n = 7/10 := by
  sorry

end vector_equation_result_l3090_309086


namespace triangle_angle_theorem_l3090_309031

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h₃ : ℝ
  angleA : ℝ
  angleC : ℝ

-- Define the theorem
theorem triangle_angle_theorem (t : Triangle) 
  (h : 1 / t.h₃^2 = 1 / t.a^2 + 1 / t.b^2) : 
  t.angleC = 90 ∨ |t.angleA - t.angleC| = 90 := by
  sorry

end triangle_angle_theorem_l3090_309031
