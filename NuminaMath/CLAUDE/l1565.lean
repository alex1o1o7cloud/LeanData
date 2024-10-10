import Mathlib

namespace tangent_relation_l1565_156538

theorem tangent_relation (α β : Real) 
  (h : Real.tan (α - β) = Real.sin (2 * β) / (5 - Real.cos (2 * β))) :
  2 * Real.tan α = 3 * Real.tan β := by
  sorry

end tangent_relation_l1565_156538


namespace max_fridays_12th_non_leap_year_max_fridays_12th_leap_year_l1565_156549

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

/-- Represents a month of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Returns the number of days in a given month for a given year -/
def daysInMonth (m : Month) (isLeapYear : Bool) : Nat :=
  match m with
  | .January => 31
  | .February => if isLeapYear then 29 else 28
  | .March => 31
  | .April => 30
  | .May => 31
  | .June => 30
  | .July => 31
  | .August => 31
  | .September => 30
  | .October => 31
  | .November => 30
  | .December => 31

/-- Returns the day of the week for the 12th of a given month, 
    given the day of the week of January 1st -/
def dayOfWeekOn12th (m : Month) (jan1 : DayOfWeek) (isLeapYear : Bool) : DayOfWeek :=
  sorry

/-- Counts the number of Fridays that fall on the 12th in a year -/
def countFridays12th (jan1 : DayOfWeek) (isLeapYear : Bool) : Nat :=
  sorry

/-- Theorem: In a non-leap year, there can be at most 3 Fridays 
    that fall on the 12th of a month -/
theorem max_fridays_12th_non_leap_year :
  ∀ (jan1 : DayOfWeek), countFridays12th jan1 false ≤ 3 :=
  sorry

/-- Theorem: In a leap year, there can be at most 4 Fridays 
    that fall on the 12th of a month -/
theorem max_fridays_12th_leap_year :
  ∀ (jan1 : DayOfWeek), countFridays12th jan1 true ≤ 4 :=
  sorry

end max_fridays_12th_non_leap_year_max_fridays_12th_leap_year_l1565_156549


namespace ball_max_height_l1565_156540

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

-- Theorem statement
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 40 :=
sorry

end ball_max_height_l1565_156540


namespace complex_equation_solution_l1565_156509

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 1 - Real.sqrt 5 * Complex.I → z = Real.sqrt 5 - Complex.I := by
  sorry

end complex_equation_solution_l1565_156509


namespace valid_numbers_l1565_156571

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000 ∧ n ≤ 999999) ∧  -- six-digit number
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = a * 100000 + 2014 * 10 + b) ∧  -- formed by adding digits to 2014
  n % 36 = 0  -- divisible by 36

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {220140, 720144, 320148} :=
sorry

end valid_numbers_l1565_156571


namespace sufficient_not_necessary_condition_l1565_156548

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (((x - y) * x^4 < 0 → x < y) ∧
   ∃ a b : ℝ, a < b ∧ (a - b) * a^4 ≥ 0) :=
by sorry

end sufficient_not_necessary_condition_l1565_156548


namespace direct_proportion_problem_l1565_156589

theorem direct_proportion_problem (α β : ℝ) (k : ℝ) (h1 : α = k * β) (h2 : 6 = k * 18) (h3 : α = 15) : β = 45 := by
  sorry

end direct_proportion_problem_l1565_156589


namespace cos_product_pi_ninths_l1565_156541

theorem cos_product_pi_ninths : 
  Real.cos (π / 9) * Real.cos (2 * π / 9) * Real.cos (4 * π / 9) = 1 / 8 := by
  sorry

end cos_product_pi_ninths_l1565_156541


namespace jonah_calories_per_hour_l1565_156526

/-- The number of calories Jonah burns per hour while running -/
def calories_per_hour : ℝ := 30

/-- The number of hours Jonah actually ran -/
def actual_hours : ℝ := 2

/-- The hypothetical number of hours Jonah could have run -/
def hypothetical_hours : ℝ := 5

/-- The additional calories Jonah would have burned if he ran for the hypothetical hours -/
def additional_calories : ℝ := 90

theorem jonah_calories_per_hour :
  calories_per_hour * hypothetical_hours = 
  calories_per_hour * actual_hours + additional_calories :=
sorry

end jonah_calories_per_hour_l1565_156526


namespace condition_2_is_sufficient_for_condition_1_l1565_156527

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationship between conditions
def condition_relationship (A B C D : Prop) : Prop :=
  (C < D → A > B)

-- Define sufficient condition
def is_sufficient_condition (P Q : Prop) : Prop :=
  P → Q

-- Theorem statement
theorem condition_2_is_sufficient_for_condition_1 
  (h : condition_relationship A B C D) :
  is_sufficient_condition (C < D) (A > B) :=
sorry

end condition_2_is_sufficient_for_condition_1_l1565_156527


namespace dance_to_electropop_ratio_l1565_156551

def total_requests : ℕ := 30
def electropop_requests : ℕ := total_requests / 2
def rock_requests : ℕ := 5
def oldies_requests : ℕ := rock_requests - 3
def dj_choice_requests : ℕ := oldies_requests / 2
def rap_requests : ℕ := 2

def non_electropop_requests : ℕ := rock_requests + oldies_requests + dj_choice_requests + rap_requests

def dance_music_requests : ℕ := total_requests - non_electropop_requests

theorem dance_to_electropop_ratio :
  dance_music_requests = electropop_requests :=
sorry

end dance_to_electropop_ratio_l1565_156551


namespace family_composition_l1565_156552

/-- Represents a family with boys and girls -/
structure Family where
  boys : Nat
  girls : Nat

/-- A boy in the family has equal number of brothers and sisters -/
def equal_siblings (f : Family) : Prop :=
  f.boys - 1 = f.girls

/-- A girl in the family has twice as many brothers as sisters -/
def double_brothers (f : Family) : Prop :=
  f.boys = 2 * (f.girls - 1)

/-- The family satisfies both conditions and has 4 boys and 3 girls -/
theorem family_composition :
  ∃ (f : Family), equal_siblings f ∧ double_brothers f ∧ f.boys = 4 ∧ f.girls = 3 :=
by
  sorry


end family_composition_l1565_156552


namespace range_of_t_l1565_156506

-- Define the solution set
def solution_set : Set ℤ := {1, 2, 3}

-- Define the inequality condition
def inequality_condition (t : ℝ) (x : ℤ) : Prop :=
  |3 * (x : ℝ) + t| < 4

-- Define the main theorem
theorem range_of_t :
  ∀ t : ℝ,
  (∀ x : ℤ, x ∈ solution_set ↔ inequality_condition t x) →
  -7 < t ∧ t < -5 :=
sorry

end range_of_t_l1565_156506


namespace adelkas_numbers_l1565_156529

theorem adelkas_numbers : ∃ (a b : ℕ), 
  0 < a ∧ a < b ∧ b < 100 ∧
  (Nat.gcd a b) < a ∧ a < b ∧ b < (Nat.lcm a b) ∧ (Nat.lcm a b) < 100 ∧
  (Nat.lcm a b) / (Nat.gcd a b) = Nat.gcd (Nat.gcd a b) (Nat.gcd a (Nat.gcd b (Nat.lcm a b))) ∧
  a = 12 ∧ b = 18 := by
sorry

end adelkas_numbers_l1565_156529


namespace lesser_number_l1565_156579

theorem lesser_number (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 6) : y = 22 := by
  sorry

end lesser_number_l1565_156579


namespace vector_equation_solution_l1565_156576

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b : V) (x y : ℝ) 
  (h_not_collinear : ¬ ∃ (k : ℝ), b = k • a) 
  (h_eq : (3 * x - 4 * y) • a + (2 * x - 3 * y) • b = 6 • a + 3 • b) :
  x - y = 3 := by
sorry

end vector_equation_solution_l1565_156576


namespace simple_interest_duration_l1565_156581

/-- Simple interest calculation -/
theorem simple_interest_duration (P R SI : ℝ) (h1 : P = 10000) (h2 : R = 9) (h3 : SI = 900) :
  (SI * 100) / (P * R) * 12 = 12 := by
  sorry

end simple_interest_duration_l1565_156581


namespace absolute_value_inequality_l1565_156596

theorem absolute_value_inequality (x : ℝ) : |2*x - 1| ≤ 1 ↔ 0 ≤ x ∧ x ≤ 1 := by
  sorry

end absolute_value_inequality_l1565_156596


namespace words_with_b_count_l1565_156572

/-- The number of letters in the alphabet we're using -/
def alphabet_size : ℕ := 5

/-- The length of the words we're forming -/
def word_length : ℕ := 4

/-- The number of letters in the alphabet excluding B -/
def alphabet_size_without_b : ℕ := 4

/-- The total number of possible words -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words without B -/
def words_without_b : ℕ := alphabet_size_without_b ^ word_length

/-- The number of words with at least one B -/
def words_with_b : ℕ := total_words - words_without_b

theorem words_with_b_count : words_with_b = 369 := by
  sorry

end words_with_b_count_l1565_156572


namespace solve_for_a_l1565_156582

-- Define the equation
def equation (a b c : ℝ) : Prop :=
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

-- Define the theorem
theorem solve_for_a :
  ∀ a : ℝ, equation a 15 7 → a * 15 * 7 = 1.5 → a = 6 := by
  sorry

end solve_for_a_l1565_156582


namespace twentieth_term_of_combined_sequence_l1565_156563

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

def geometric_sequence (g₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := g₁ * r ^ (n - 1)

def combined_sequence (a₁ g₁ d r : ℝ) (n : ℕ) : ℝ :=
  arithmetic_sequence a₁ d n + geometric_sequence g₁ r n

theorem twentieth_term_of_combined_sequence :
  combined_sequence 3 2 4 2 20 = 1048655 := by sorry

end twentieth_term_of_combined_sequence_l1565_156563


namespace problem_solution_l1565_156584

def A (a : ℝ) : Set ℝ := {x | 2 * a - 3 < x ∧ x < a + 1}

def B : Set ℝ := {x | (x + 1) * (x - 3) < 0}

theorem problem_solution :
  (∀ x : ℝ, x ∈ A 0 ∩ B ↔ -1 < x ∧ x < 1) ∧
  (∀ a : ℝ, A a ∩ (Set.univ \ B) = A a ↔ a ≤ -2 ∨ a ≥ 3) :=
sorry

end problem_solution_l1565_156584


namespace seven_eighths_of_64_l1565_156566

theorem seven_eighths_of_64 : (7 / 8 : ℚ) * 64 = 56 := by
  sorry

end seven_eighths_of_64_l1565_156566


namespace arithmetic_sequence_problem_l1565_156577

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : isArithmeticSequence a)
  (h_sum : a 6 + a 9 = 16)
  (h_a4 : a 4 = 1) :
  a 11 = 15 := by
  sorry

end arithmetic_sequence_problem_l1565_156577


namespace statement_true_except_two_and_five_l1565_156598

theorem statement_true_except_two_and_five (x : ℝ) :
  (x - 2) * (x - 5) ≠ 0 ↔ x ≠ 2 ∧ x ≠ 5 := by
  sorry

end statement_true_except_two_and_five_l1565_156598


namespace grid_state_theorem_l1565_156565

/-- Represents the number of times a 2x2 square was picked -/
structure SquarePicks where
  topLeft : ℕ
  topRight : ℕ
  bottomLeft : ℕ
  bottomRight : ℕ

/-- Represents the state of the 3x3 grid -/
def GridState (p : SquarePicks) : Matrix (Fin 3) (Fin 3) ℕ :=
  fun i j =>
    match i, j with
    | 0, 0 => p.topLeft
    | 0, 2 => p.topRight
    | 2, 0 => p.bottomLeft
    | 2, 2 => p.bottomRight
    | 0, 1 => p.topLeft + p.topRight
    | 1, 0 => p.topLeft + p.bottomLeft
    | 1, 2 => p.topRight + p.bottomRight
    | 2, 1 => p.bottomLeft + p.bottomRight
    | 1, 1 => p.topLeft + p.topRight + p.bottomLeft + p.bottomRight

theorem grid_state_theorem (p : SquarePicks) :
  (GridState p 2 0 = 13) →
  (GridState p 0 1 = 18) →
  (GridState p 1 1 = 47) →
  (GridState p 2 2 = 16) := by
    sorry

end grid_state_theorem_l1565_156565


namespace zoo_count_l1565_156510

/-- Represents the number of peacocks in the zoo -/
def num_peacocks : ℕ := 7

/-- Represents the number of tortoises in the zoo -/
def num_tortoises : ℕ := 17 - num_peacocks

/-- The total number of legs in the zoo -/
def total_legs : ℕ := 54

/-- The total number of heads in the zoo -/
def total_heads : ℕ := 17

/-- Each peacock has 2 legs -/
def peacock_legs : ℕ := 2

/-- Each peacock has 1 head -/
def peacock_head : ℕ := 1

/-- Each tortoise has 4 legs -/
def tortoise_legs : ℕ := 4

/-- Each tortoise has 1 head -/
def tortoise_head : ℕ := 1

theorem zoo_count :
  num_peacocks * peacock_legs + num_tortoises * tortoise_legs = total_legs ∧
  num_peacocks * peacock_head + num_tortoises * tortoise_head = total_heads :=
by sorry

end zoo_count_l1565_156510


namespace salary_reduction_percentage_l1565_156535

theorem salary_reduction_percentage (S : ℝ) (R : ℝ) :
  S > 0 →
  (S - (R / 100 * S)) * (1 + 1 / 3) = S →
  R = 25 := by
sorry

end salary_reduction_percentage_l1565_156535


namespace total_peanuts_l1565_156518

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := 10

/-- The number of peanuts Mary adds to the box -/
def added_peanuts : ℕ := 8

/-- Theorem stating the total number of peanuts in the box -/
theorem total_peanuts : initial_peanuts + added_peanuts = 18 := by
  sorry

end total_peanuts_l1565_156518


namespace tablet_savings_l1565_156550

/-- Proves that buying a tablet in cash saves $70 compared to an installment plan -/
theorem tablet_savings : 
  let cash_price : ℕ := 450
  let down_payment : ℕ := 100
  let first_four_months : ℕ := 4 * 40
  let next_four_months : ℕ := 4 * 35
  let last_four_months : ℕ := 4 * 30
  let total_installment : ℕ := down_payment + first_four_months + next_four_months + last_four_months
  total_installment - cash_price = 70 := by
  sorry

end tablet_savings_l1565_156550


namespace rectangular_shape_perimeter_and_area_l1565_156523

/-- A rectangular shape composed of 5 cm segments -/
structure RectangularShape where
  segmentLength : ℝ
  length : ℝ
  height : ℝ

/-- Calculate the perimeter of the rectangular shape -/
def perimeter (shape : RectangularShape) : ℝ :=
  2 * (shape.length + shape.height)

/-- Calculate the area of the rectangular shape -/
def area (shape : RectangularShape) : ℝ :=
  shape.length * shape.height

theorem rectangular_shape_perimeter_and_area 
  (shape : RectangularShape)
  (h1 : shape.segmentLength = 5)
  (h2 : shape.length = 45)
  (h3 : shape.height = 30) :
  perimeter shape = 200 ∧ area shape = 725 := by
  sorry

#check rectangular_shape_perimeter_and_area

end rectangular_shape_perimeter_and_area_l1565_156523


namespace car_profit_percent_l1565_156564

/-- Calculate the profit percent from buying, repairing, and selling a car -/
theorem car_profit_percent 
  (purchase_price : ℝ) 
  (repair_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : purchase_price = 42000) 
  (h2 : repair_cost = 12000) 
  (h3 : selling_price = 64900) : 
  ∃ (profit_percent : ℝ), abs (profit_percent - 20.19) < 0.01 := by
  sorry

end car_profit_percent_l1565_156564


namespace sams_seashells_l1565_156586

theorem sams_seashells (mary_seashells : ℕ) (total_seashells : ℕ) 
  (h1 : mary_seashells = 47)
  (h2 : total_seashells = 65) :
  total_seashells - mary_seashells = 18 := by
  sorry

end sams_seashells_l1565_156586


namespace larger_cube_volume_l1565_156573

theorem larger_cube_volume (v : ℝ) (k : ℝ) : 
  v = 216 → k = 2.5 → (k * (v ^ (1/3 : ℝ)))^3 = 3375 := by
  sorry

end larger_cube_volume_l1565_156573


namespace interest_rate_calculation_l1565_156514

theorem interest_rate_calculation (P t D : ℝ) (h1 : P = 500) (h2 : t = 2) (h3 : D = 20) : 
  ∃ r : ℝ, r = 20 ∧ 
    P * ((1 + r / 100) ^ t - 1) - P * r * t / 100 = D :=
sorry

end interest_rate_calculation_l1565_156514


namespace existence_of_suitable_set_l1565_156560

theorem existence_of_suitable_set (ε : Real) (h_ε : 0 < ε ∧ ε < 1) :
  ∃ N₀ : ℕ, ∀ N ≥ N₀, ∃ S : Finset ℕ,
    (S.card : ℝ) ≥ ε * N ∧
    (∀ x ∈ S, x ≤ N) ∧
    (∀ x ∈ S, Nat.gcd x (S.sum id) > 1) :=
by sorry

end existence_of_suitable_set_l1565_156560


namespace ordered_pair_satisfies_equation_l1565_156530

theorem ordered_pair_satisfies_equation :
  let a : ℝ := 9
  let b : ℝ := -4
  (Real.sqrt (25 - 16 * Real.cos (π / 3)) = a - b * (1 / Real.cos (π / 3))) := by
  sorry

end ordered_pair_satisfies_equation_l1565_156530


namespace pony_discount_rate_l1565_156517

-- Define the regular prices
def fox_price : ℝ := 15
def pony_price : ℝ := 18

-- Define the number of jeans purchased
def fox_quantity : ℕ := 3
def pony_quantity : ℕ := 2

-- Define the total savings
def total_savings : ℝ := 9

-- Define the sum of discount rates
def total_discount_rate : ℝ := 22

-- Theorem statement
theorem pony_discount_rate :
  ∃ (fox_discount pony_discount : ℝ),
    fox_discount + pony_discount = total_discount_rate ∧
    fox_quantity * fox_price * (fox_discount / 100) +
    pony_quantity * pony_price * (pony_discount / 100) = total_savings ∧
    pony_discount = 10 := by
  sorry

end pony_discount_rate_l1565_156517


namespace negation_of_proposition_l1565_156512

def proposition (x : Real) : Prop := x ∈ Set.Icc 0 (2 * Real.pi) → |Real.sin x| ≤ 1

theorem negation_of_proposition :
  (¬ ∀ x, proposition x) ↔ (∃ x, x ∈ Set.Icc 0 (2 * Real.pi) ∧ |Real.sin x| > 1) :=
by sorry

end negation_of_proposition_l1565_156512


namespace problem_1_l1565_156515

theorem problem_1 : (1/3)⁻¹ + Real.sqrt 18 - 4 * Real.cos (π/4) = 3 + Real.sqrt 2 := by
  sorry

end problem_1_l1565_156515


namespace largest_number_l1565_156599

theorem largest_number (a b c d : ℝ) (h1 : a = -1) (h2 : b = 0) (h3 : c = 1) (h4 : d = 2) :
  d ≥ a ∧ d ≥ b ∧ d ≥ c := by
  sorry

end largest_number_l1565_156599


namespace ac_values_l1565_156545

theorem ac_values (a c : ℝ) (h : ∀ x, 2 * Real.sin (3 * x) = a * Real.cos (3 * x + c)) :
  ∃ k : ℤ, a * c = (4 * k - 1) * Real.pi :=
sorry

end ac_values_l1565_156545


namespace player_pay_is_23000_l1565_156574

/-- Represents the player's performance in a single game -/
structure GamePerformance :=
  (points : ℕ)
  (assists : ℕ)
  (rebounds : ℕ)
  (steals : ℕ)

/-- Calculates the base pay based on average points per game -/
def basePay (games : List GamePerformance) : ℕ :=
  if (games.map GamePerformance.points).sum / games.length ≥ 30 then 10000 else 8000

/-- Calculates the assists bonus based on total assists -/
def assistsBonus (games : List GamePerformance) : ℕ :=
  let totalAssists := (games.map GamePerformance.assists).sum
  if totalAssists ≥ 20 then 5000
  else if totalAssists ≥ 10 then 3000
  else 1000

/-- Calculates the rebounds bonus based on total rebounds -/
def reboundsBonus (games : List GamePerformance) : ℕ :=
  let totalRebounds := (games.map GamePerformance.rebounds).sum
  if totalRebounds ≥ 40 then 5000
  else if totalRebounds ≥ 20 then 3000
  else 1000

/-- Calculates the steals bonus based on total steals -/
def stealsBonus (games : List GamePerformance) : ℕ :=
  let totalSteals := (games.map GamePerformance.steals).sum
  if totalSteals ≥ 15 then 5000
  else if totalSteals ≥ 5 then 3000
  else 1000

/-- Calculates the total pay for the week -/
def totalPay (games : List GamePerformance) : ℕ :=
  basePay games + assistsBonus games + reboundsBonus games + stealsBonus games

/-- Theorem: Given the player's performance, the total pay for the week is $23,000 -/
theorem player_pay_is_23000 (games : List GamePerformance) 
  (h1 : games = [
    ⟨30, 5, 7, 3⟩, 
    ⟨28, 6, 5, 2⟩, 
    ⟨32, 4, 9, 1⟩, 
    ⟨34, 3, 11, 2⟩, 
    ⟨26, 2, 8, 3⟩
  ]) : 
  totalPay games = 23000 := by
  sorry


end player_pay_is_23000_l1565_156574


namespace inscribed_triangle_ratio_l1565_156501

-- Define the ellipse
def ellipse (p q : ℝ) (x y : ℝ) : Prop :=
  x^2 / p^2 + y^2 / q^2 = 1

-- Define an equilateral triangle
def equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C A)

-- Define that a point is on a line segment
def on_segment (P A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = ((1 - t) • A.1 + t • B.1, (1 - t) • A.2 + t • B.2)

theorem inscribed_triangle_ratio (p q : ℝ) (A B C F₁ F₂ : ℝ × ℝ) :
  ellipse p q A.1 A.2 →
  ellipse p q B.1 B.2 →
  ellipse p q C.1 C.2 →
  B = (0, q) →
  A.2 = C.2 →
  equilateral_triangle A B C →
  on_segment F₁ B C →
  on_segment F₂ A B →
  dist F₁ F₂ = 2 →
  dist A B / dist F₁ F₂ = 8/5 :=
sorry

end inscribed_triangle_ratio_l1565_156501


namespace remainder_problem_l1565_156595

theorem remainder_problem (m n : ℕ) (hm : m > 0) (hn : n > 0) (hmn : m > n) 
  (hm_mod : m % 6 = 2) (hdiff_mod : (m - n) % 6 = 5) : n % 6 = 4 := by
  sorry

end remainder_problem_l1565_156595


namespace response_change_difference_l1565_156543

/-- Represents the percentages of student responses --/
structure ResponsePercentages where
  yes : ℝ
  no : ℝ
  undecided : ℝ

/-- The problem statement --/
theorem response_change_difference
  (initial : ResponsePercentages)
  (final : ResponsePercentages)
  (h_initial_sum : initial.yes + initial.no + initial.undecided = 100)
  (h_final_sum : final.yes + final.no + final.undecided = 100)
  (h_initial_yes : initial.yes = 40)
  (h_initial_no : initial.no = 30)
  (h_initial_undecided : initial.undecided = 30)
  (h_final_yes : final.yes = 60)
  (h_final_no : final.no = 10)
  (h_final_undecided : final.undecided = 30) :
  ∃ (min_change max_change : ℝ),
    (∀ (change : ℝ), min_change ≤ change ∧ change ≤ max_change) ∧
    max_change - min_change = 20 :=
sorry

end response_change_difference_l1565_156543


namespace sqrt_equation_solution_l1565_156522

theorem sqrt_equation_solution :
  ∃! (y : ℝ), y > 0 ∧ 3 * Real.sqrt (4 + y) + 3 * Real.sqrt (4 - y) = 6 * Real.sqrt 3 :=
by
  use 2 * Real.sqrt 3
  sorry

end sqrt_equation_solution_l1565_156522


namespace fraction_sum_equality_l1565_156597

theorem fraction_sum_equality : 
  (4 : ℚ) / 3 + 13 / 9 + 40 / 27 + 121 / 81 - 8 / 3 = 171 / 81 := by
  sorry

end fraction_sum_equality_l1565_156597


namespace parabola_coef_sum_zero_l1565_156536

/-- A parabola with equation y = ax^2 + bx + c, vertex (3, 4), and passing through (1, 0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := 4
  point_x : ℝ := 1
  point_y : ℝ := 0
  eq_at_vertex : vertex_y = a * vertex_x^2 + b * vertex_x + c
  eq_at_point : point_y = a * point_x^2 + b * point_x + c

/-- The sum of coefficients a, b, and c for the specified parabola is 0 -/
theorem parabola_coef_sum_zero (p : Parabola) : p.a + p.b + p.c = 0 := by
  sorry


end parabola_coef_sum_zero_l1565_156536


namespace fraction_equality_l1565_156525

theorem fraction_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b * (a + b) = 1) :
  a / (a^3 + a + 1) = b / (b^3 + b + 1) := by
  sorry

end fraction_equality_l1565_156525


namespace circle_circumference_ratio_l1565_156544

theorem circle_circumference_ratio (r_large r_small : ℝ) (h : r_large / r_small = 3 / 2) :
  (2 * Real.pi * r_large) / (2 * Real.pi * r_small) = 3 / 2 := by
  sorry

end circle_circumference_ratio_l1565_156544


namespace barry_fifth_game_yards_l1565_156556

theorem barry_fifth_game_yards (game1 game2 game3 game4 game6 : ℕ) 
  (h1 : game1 = 98)
  (h2 : game2 = 107)
  (h3 : game3 = 85)
  (h4 : game4 = 89)
  (h5 : game6 ≥ 130)
  (h6 : (game1 + game2 + game3 + game4 + game6 : ℚ) / 6 > 100) :
  ∃ game5 : ℕ, game5 = 91 ∧ (game1 + game2 + game3 + game4 + game5 + game6 : ℚ) / 6 > 100 := by
sorry

end barry_fifth_game_yards_l1565_156556


namespace train_speed_problem_l1565_156516

/-- Proves that given the conditions of the train problem, the speeds of the regular and high-speed trains are 100 km/h and 250 km/h respectively. -/
theorem train_speed_problem (regular_speed : ℝ) (bullet_speed : ℝ) (high_speed : ℝ) (express_speed : ℝ)
  (h1 : bullet_speed = 2 * regular_speed)
  (h2 : high_speed = bullet_speed * 1.25)
  (h3 : (high_speed + regular_speed) / 2 = express_speed + 15)
  (h4 : (bullet_speed + regular_speed) / 2 = express_speed - 10) :
  regular_speed = 100 ∧ high_speed = 250 := by
  sorry

end train_speed_problem_l1565_156516


namespace eggs_per_tray_calculation_l1565_156562

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 45

/-- The number of trays bought weekly -/
def trays_per_week : ℕ := 2

/-- The number of children -/
def num_children : ℕ := 2

/-- The number of eggs eaten by each child daily -/
def child_eggs_per_day : ℕ := 2

/-- The number of adults -/
def num_adults : ℕ := 2

/-- The number of eggs eaten by each adult daily -/
def adult_eggs_per_day : ℕ := 4

/-- The number of eggs left uneaten weekly -/
def uneaten_eggs_per_week : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem eggs_per_tray_calculation :
  eggs_per_tray * trays_per_week = 
    num_children * child_eggs_per_day * days_per_week +
    num_adults * adult_eggs_per_day * days_per_week +
    uneaten_eggs_per_week :=
by sorry

end eggs_per_tray_calculation_l1565_156562


namespace largest_prime_square_root_l1565_156507

theorem largest_prime_square_root (p : ℕ) (a b : ℕ+) (h_prime : Nat.Prime p) 
  (h_eq : (p : ℝ) = (b.val : ℝ) / 2 * Real.sqrt ((a.val - b.val : ℝ) / (a.val + b.val))) :
  p ≤ 5 :=
sorry

end largest_prime_square_root_l1565_156507


namespace john_total_spent_l1565_156537

/-- The total amount John spends on t-shirts and pants -/
def total_spent (num_tshirts : ℕ) (price_per_tshirt : ℕ) (pants_cost : ℕ) : ℕ :=
  num_tshirts * price_per_tshirt + pants_cost

/-- Theorem: John spends $110 in total -/
theorem john_total_spent :
  total_spent 3 20 50 = 110 := by
  sorry

end john_total_spent_l1565_156537


namespace cost_per_bag_first_is_24_l1565_156533

/-- The cost per bag of zongzi in the first batch -/
def cost_per_bag_first : ℝ := 24

/-- The total cost of the first batch of zongzi -/
def total_cost_first : ℝ := 3000

/-- The total cost of the second batch of zongzi -/
def total_cost_second : ℝ := 7500

/-- The number of bags in the second batch is three times the number in the first batch -/
def batch_ratio : ℝ := 3

/-- The cost difference per bag between the first and second batch -/
def cost_difference : ℝ := 4

theorem cost_per_bag_first_is_24 :
  cost_per_bag_first = 24 ∧
  total_cost_first = 3000 ∧
  total_cost_second = 7500 ∧
  batch_ratio = 3 ∧
  cost_difference = 4 →
  cost_per_bag_first = 24 :=
by sorry

end cost_per_bag_first_is_24_l1565_156533


namespace slower_walking_speed_l1565_156554

theorem slower_walking_speed (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 40 → delay = 10 → 
  (usual_time / (usual_time + delay)) = (4 : ℝ) / 5 := by
  sorry

end slower_walking_speed_l1565_156554


namespace bakers_pastries_l1565_156521

/-- Baker's pastry problem -/
theorem bakers_pastries 
  (initial_cakes : ℕ)
  (sold_cakes : ℕ)
  (sold_pastries : ℕ)
  (remaining_pastries : ℕ)
  (h1 : initial_cakes = 7)
  (h2 : sold_cakes = 15)
  (h3 : sold_pastries = 103)
  (h4 : remaining_pastries = 45) :
  sold_pastries + remaining_pastries = 148 :=
sorry

end bakers_pastries_l1565_156521


namespace price_reduction_theorem_l1565_156557

-- Define the reduction factors
def first_reduction : ℝ := 0.85  -- 1 - 0.15
def second_reduction : ℝ := 0.90 -- 1 - 0.10

-- Theorem statement
theorem price_reduction_theorem :
  first_reduction * second_reduction * 100 = 76.5 := by
  sorry

#eval first_reduction * second_reduction * 100

end price_reduction_theorem_l1565_156557


namespace jelly_overlap_l1565_156508

/-- The number of jellies -/
def num_jellies : ℕ := 12

/-- The length of each jelly in centimeters -/
def jelly_length : ℝ := 18

/-- The circumference of the ring in centimeters -/
def ring_circumference : ℝ := 210

/-- The overlapping portion of each jelly in millimeters -/
def overlap_mm : ℝ := 5

theorem jelly_overlap :
  (num_jellies : ℝ) * jelly_length - ring_circumference = num_jellies * overlap_mm / 10 := by
  sorry

end jelly_overlap_l1565_156508


namespace batsman_average_after_12_innings_l1565_156568

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat

/-- Calculates the average score of a batsman -/
def calculateAverage (b : Batsman) : Nat :=
  b.totalRuns / b.innings

/-- Theorem: A batsman's average after 12 innings is 70 runs -/
theorem batsman_average_after_12_innings (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.totalRuns = calculateAverage b * 11 + 92)
  (h3 : b.averageIncrease = 2)
  : calculateAverage b = 70 := by
  sorry


end batsman_average_after_12_innings_l1565_156568


namespace average_score_of_group_specific_group_average_l1565_156593

theorem average_score_of_group (total_people : ℕ) (group1_size : ℕ) (group2_size : ℕ) 
  (group1_avg : ℝ) (group2_avg : ℝ) :
  total_people = group1_size + group2_size →
  (group1_size : ℝ) * group1_avg + (group2_size : ℝ) * group2_avg = 
    (total_people : ℝ) * ((group1_size : ℝ) * group1_avg + (group2_size : ℝ) * group2_avg) / (total_people : ℝ) :=
by
  sorry

-- The specific problem instance
theorem specific_group_average :
  let total_people : ℕ := 10
  let group1_size : ℕ := 6
  let group2_size : ℕ := 4
  let group1_avg : ℝ := 90
  let group2_avg : ℝ := 80
  ((group1_size : ℝ) * group1_avg + (group2_size : ℝ) * group2_avg) / (total_people : ℝ) = 86 :=
by
  sorry

end average_score_of_group_specific_group_average_l1565_156593


namespace basketball_score_proof_l1565_156539

theorem basketball_score_proof :
  ∀ (two_pointers three_pointers free_throws : ℕ),
    2 * two_pointers = 3 * three_pointers →
    free_throws = 2 * two_pointers →
    2 * two_pointers + 3 * three_pointers + free_throws = 78 →
    free_throws = 26 :=
by
  sorry

end basketball_score_proof_l1565_156539


namespace matthew_crackers_l1565_156580

def crackers_problem (initial_crackers : ℕ) (friends : ℕ) (crackers_per_friend : ℕ) : Prop :=
  initial_crackers - (friends * crackers_per_friend) = 3

theorem matthew_crackers : crackers_problem 24 3 7 := by
  sorry

end matthew_crackers_l1565_156580


namespace larger_number_problem_l1565_156583

theorem larger_number_problem (x y : ℝ) (h1 : x - y = 8) (h2 : (1/4) * (x + y) = 6) : max x y = 16 := by
  sorry

end larger_number_problem_l1565_156583


namespace lining_fabric_cost_l1565_156524

/-- The cost of lining fabric per yard -/
def lining_cost : ℝ := 30.69

theorem lining_fabric_cost :
  let velvet_cost : ℝ := 24
  let pattern_cost : ℝ := 15
  let thread_cost : ℝ := 3 * 2
  let buttons_cost : ℝ := 14
  let trim_cost : ℝ := 19 * 3
  let velvet_yards : ℝ := 5
  let lining_yards : ℝ := 4
  let discount_rate : ℝ := 0.1
  let total_cost : ℝ := 310.50
  
  total_cost = (1 - discount_rate) * (velvet_cost * velvet_yards + lining_cost * lining_yards) +
               pattern_cost + thread_cost + buttons_cost + trim_cost :=
by sorry


end lining_fabric_cost_l1565_156524


namespace area_max_opposite_angles_sum_pi_l1565_156513

/-- A quadrilateral with sides a, b, c, d and angles α, β, γ, δ. -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  δ : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d
  angle_sum : α + β + γ + δ = 2 * Real.pi

/-- The area of a quadrilateral. -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Theorem: The area of a quadrilateral is maximized when the sum of its opposite angles is π (180°). -/
theorem area_max_opposite_angles_sum_pi (q : Quadrilateral) :
  ∀ q' : Quadrilateral, q'.a = q.a ∧ q'.b = q.b ∧ q'.c = q.c ∧ q'.d = q.d →
  area q' ≤ area q ↔ q.α + q.γ = Real.pi ∧ q.β + q.δ = Real.pi :=
sorry

end area_max_opposite_angles_sum_pi_l1565_156513


namespace currency_denomination_proof_l1565_156519

theorem currency_denomination_proof :
  let press_F_rate : ℚ := 1000 / 60  -- bills per second
  let press_F_value : ℚ := 5 * press_F_rate  -- dollars per second
  let press_T_rate : ℚ := 200 / 60  -- bills per second
  let time : ℚ := 3  -- seconds
  let extra_value : ℚ := 50  -- dollars
  ∃ x : ℚ, 
    (time * press_F_value = time * (x * press_T_rate) + extra_value) ∧ 
    x = 20 :=
by
  sorry

end currency_denomination_proof_l1565_156519


namespace third_term_of_specific_arithmetic_sequence_l1565_156553

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- State the theorem
theorem third_term_of_specific_arithmetic_sequence 
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 2) :
  a 3 = 3 :=
by sorry

end third_term_of_specific_arithmetic_sequence_l1565_156553


namespace runner_journey_time_l1565_156588

/-- Represents the runner's journey --/
structure RunnerJourney where
  totalDistance : ℝ
  firstHalfSpeed : ℝ
  secondHalfSpeed : ℝ
  firstHalfTime : ℝ
  secondHalfTime : ℝ

/-- Theorem stating the conditions and the result to be proved --/
theorem runner_journey_time (j : RunnerJourney) 
  (h1 : j.totalDistance = 40)
  (h2 : j.secondHalfSpeed = j.firstHalfSpeed / 2)
  (h3 : j.secondHalfTime = j.firstHalfTime + 5)
  (h4 : j.firstHalfTime = (j.totalDistance / 2) / j.firstHalfSpeed)
  (h5 : j.secondHalfTime = (j.totalDistance / 2) / j.secondHalfSpeed) :
  j.secondHalfTime = 10 := by
  sorry

end runner_journey_time_l1565_156588


namespace f_monotone_decreasing_no_minimum_l1565_156587

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem f_monotone_decreasing_no_minimum :
  (∀ x y : ℝ, x < y → f x > f y) ∧ 
  (∀ ε > 0, ∃ x : ℝ, f x < ε) :=
by sorry

end f_monotone_decreasing_no_minimum_l1565_156587


namespace jose_painting_time_l1565_156503

/-- The time it takes for Alex to paint a car alone -/
def alex_time : ℝ := 5

/-- The time it takes for Jose and Alex to paint a car together -/
def combined_time : ℝ := 2.91666666667

/-- The time it takes for Jose to paint a car alone -/
def jose_time : ℝ := 7

/-- Theorem stating that given Alex's time and the combined time, Jose's time is 7 days -/
theorem jose_painting_time : 
  1 / alex_time + 1 / jose_time = 1 / combined_time :=
sorry

end jose_painting_time_l1565_156503


namespace a_10_equals_133_l1565_156528

/-- The number of subsets of {1,2,...,n} with at least two elements and 
    the absolute difference between any two elements greater than 1 -/
def a (n : ℕ) : ℕ :=
  if n ≤ 2 then 0
  else if n = 3 then 1
  else if n = 4 then 3
  else a (n-1) + a (n-2) + (n-2)

/-- The main theorem to prove -/
theorem a_10_equals_133 : a 10 = 133 := by
  sorry

end a_10_equals_133_l1565_156528


namespace binary_octal_equivalence_l1565_156570

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 8^i) 0

/-- The binary number 1001101₂ is equal to the octal number 115₈ -/
theorem binary_octal_equivalence : 
  binary_to_decimal [1, 0, 1, 1, 0, 0, 1] = octal_to_decimal [5, 1, 1] := by
  sorry

end binary_octal_equivalence_l1565_156570


namespace division_calculation_l1565_156558

theorem division_calculation : 250 / (5 + 15 * 3^2) = 25 / 14 := by
  sorry

end division_calculation_l1565_156558


namespace theater_seat_interpretation_l1565_156559

/-- Represents a theater seat as an ordered pair of natural numbers -/
structure TheaterSeat :=
  (row : ℕ)
  (seat : ℕ)

/-- Interprets a TheaterSeat as a description -/
def interpret (s : TheaterSeat) : String :=
  s!"seat {s.seat} in row {s.row}"

theorem theater_seat_interpretation :
  interpret ⟨6, 2⟩ = "seat 2 in row 6" := by
  sorry

end theater_seat_interpretation_l1565_156559


namespace pencil_count_l1565_156569

/-- Given an initial number of pencils and a number of pencils added,
    the total number of pencils is equal to the sum of these two numbers. -/
theorem pencil_count (initial : ℕ) (added : ℕ) :
  initial + added = initial + added :=
by sorry

end pencil_count_l1565_156569


namespace photo_arrangement_count_l1565_156591

/-- Represents the number of people in the arrangement -/
def total_people : ℕ := 6

/-- Represents the number of students in the arrangement -/
def num_students : ℕ := 4

/-- Represents the number of teachers in the arrangement -/
def num_teachers : ℕ := 2

/-- Represents the number of students that must stand together -/
def students_together : ℕ := 2

/-- Calculates the number of ways to arrange the people given the constraints -/
def arrangement_count : ℕ :=
  (num_teachers.factorial) *    -- Ways to arrange teachers in the middle
  2 *                           -- Ways to place students A and B (left or right of teachers)
  (students_together.factorial) * -- Ways to arrange A and B within their unit
  ((num_students - students_together).factorial) -- Ways to arrange remaining students

theorem photo_arrangement_count :
  arrangement_count = 8 := by sorry

end photo_arrangement_count_l1565_156591


namespace prize_prices_and_min_cost_l1565_156520

/- Define the unit prices of prizes A and B -/
def price_A : ℝ := 20
def price_B : ℝ := 10

/- Define the total number of prizes and minimum number of prize A -/
def total_prizes : ℕ := 60
def min_prize_A : ℕ := 20

/- Define the cost function -/
def cost (m : ℕ) : ℝ := price_A * m + price_B * (total_prizes - m)

theorem prize_prices_and_min_cost :
  /- Condition 1: 1 A and 2 B cost $40 -/
  price_A + 2 * price_B = 40 ∧
  /- Condition 2: 2 A and 3 B cost $70 -/
  2 * price_A + 3 * price_B = 70 ∧
  /- The minimum cost occurs when m = min_prize_A -/
  (∀ m : ℕ, min_prize_A ≤ m → m ≤ total_prizes → cost min_prize_A ≤ cost m) ∧
  /- The minimum cost is $800 -/
  cost min_prize_A = 800 := by
  sorry

#check prize_prices_and_min_cost

end prize_prices_and_min_cost_l1565_156520


namespace circle_radius_from_inscribed_rectangle_l1565_156594

theorem circle_radius_from_inscribed_rectangle (r : ℝ) : 
  (∃ (s : ℝ), s^2 = 72 ∧ s^2 = 2 * r^2) → r = 6 := by
  sorry

end circle_radius_from_inscribed_rectangle_l1565_156594


namespace sin_beta_value_l1565_156590

theorem sin_beta_value (α β : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos (α - β) = -5/13)
  (h4 : Real.sin α = 4/5) :
  Real.sin β = -56/65 := by
  sorry

end sin_beta_value_l1565_156590


namespace chicken_difference_l1565_156592

/-- The number of Rhode Island Reds Susie has -/
def susie_rir : ℕ := 11

/-- The number of Golden Comets Susie has -/
def susie_gc : ℕ := 6

/-- The number of Rhode Island Reds Britney has -/
def britney_rir : ℕ := 2 * susie_rir

/-- The number of Golden Comets Britney has -/
def britney_gc : ℕ := susie_gc / 2

/-- The total number of chickens Susie has -/
def susie_total : ℕ := susie_rir + susie_gc

/-- The total number of chickens Britney has -/
def britney_total : ℕ := britney_rir + britney_gc

theorem chicken_difference : britney_total - susie_total = 8 := by
  sorry

end chicken_difference_l1565_156592


namespace solve_equation_l1565_156500

theorem solve_equation (x : ℝ) (h : 7 * (x - 1) = 21) : x = 4 := by
  sorry

end solve_equation_l1565_156500


namespace unique_prime_pair_solution_l1565_156575

theorem unique_prime_pair_solution : 
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (3 * p^(q-1) + 1) ∣ (11^p + 17^p) → 
    p = 3 ∧ q = 3 :=
by sorry

end unique_prime_pair_solution_l1565_156575


namespace tangent_line_and_max_value_l1565_156555

noncomputable section

-- Define the function f
def f (e a b x : ℝ) : ℝ := Real.exp (-x) * (a * x^2 + b * x + 1)

-- Define the derivative of f
def f' (e a b x : ℝ) : ℝ := 
  -Real.exp (-x) * (a * x^2 + b * x + 1) + Real.exp (-x) * (2 * a * x + b)

theorem tangent_line_and_max_value 
  (e : ℝ) (h_e : e > 0) :
  ∀ a b : ℝ, 
  (a > 0) → 
  (f' e a b (-1) = 0) →
  (
    -- Part I
    (a = 1) → 
    (∃ m c : ℝ, m = 1 ∧ c = 1 ∧ 
      ∀ x : ℝ, f e a b x = m * x + c → x = 0
    ) ∧
    -- Part II
    (a > 1/5) → 
    (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f e a b x ≤ 4 * e) →
    (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f e a b x = 4 * e) →
    (a = (8 * e^2 - 3) / 5 ∧ b = (12 * e^2 - 2) / 5)
  ) := by sorry

end tangent_line_and_max_value_l1565_156555


namespace eleven_remainders_l1565_156511

theorem eleven_remainders (A : Fin 100 → ℕ) 
  (h_perm : Function.Bijective A) 
  (h_range : ∀ i : Fin 100, A i ∈ Finset.range 101 \ {0}) : 
  let B : Fin 100 → ℕ := λ i => (Finset.range i.succ).sum (λ j => A j)
  Finset.card (Finset.image (λ i => B i % 100) Finset.univ) ≥ 11 := by
sorry

end eleven_remainders_l1565_156511


namespace hyperbola_eccentricity_l1565_156585

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let parabola := fun (x y : ℝ) ↦ y^2 = 4 * x
  let directrix := fun (x : ℝ) ↦ x = -1
  let asymptote1 := fun (x y : ℝ) ↦ y = (b / a) * x
  let asymptote2 := fun (x y : ℝ) ↦ y = -(b / a) * x
  let triangle_area := 2 * Real.sqrt 3
  let eccentricity := Real.sqrt ((a^2 + b^2) / a^2)
  (∃ A B : ℝ × ℝ, 
    directrix A.1 ∧ asymptote1 A.1 A.2 ∧
    directrix B.1 ∧ asymptote2 B.1 B.2 ∧
    (1/2) * (A.2 - B.2) = triangle_area) →
  eccentricity = Real.sqrt 13 :=
by sorry

end hyperbola_eccentricity_l1565_156585


namespace power_product_equality_l1565_156578

theorem power_product_equality : 2^4 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end power_product_equality_l1565_156578


namespace mutually_exclusive_not_contradictory_l1565_156561

/-- A pocket containing balls of two colors -/
structure Pocket where
  red : ℕ
  white : ℕ

/-- The possible outcomes when drawing two balls -/
inductive Outcome
  | TwoRed
  | OneRedOneWhite
  | TwoWhite

/-- Define the events -/
def ExactlyOneWhite (o : Outcome) : Prop :=
  o = Outcome.OneRedOneWhite

def ExactlyTwoWhite (o : Outcome) : Prop :=
  o = Outcome.TwoWhite

/-- The probability of an outcome given a pocket -/
def probability (p : Pocket) (o : Outcome) : ℚ :=
  match o with
  | Outcome.TwoRed => (p.red * (p.red - 1)) / ((p.red + p.white) * (p.red + p.white - 1))
  | Outcome.OneRedOneWhite => (2 * p.red * p.white) / ((p.red + p.white) * (p.red + p.white - 1))
  | Outcome.TwoWhite => (p.white * (p.white - 1)) / ((p.red + p.white) * (p.red + p.white - 1))

theorem mutually_exclusive_not_contradictory (p : Pocket) (h : p.red = 2 ∧ p.white = 2) :
  (∀ o : Outcome, ¬(ExactlyOneWhite o ∧ ExactlyTwoWhite o)) ∧ 
  (probability p Outcome.OneRedOneWhite + probability p Outcome.TwoWhite < 1) :=
sorry

end mutually_exclusive_not_contradictory_l1565_156561


namespace sum_xyz_equals_twenty_ninths_l1565_156531

theorem sum_xyz_equals_twenty_ninths 
  (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : x^2 + 4*y^2 + 9*z^2 = 4)
  (eq2 : 2*x + 4*y + 3*z = 6) :
  x + y + z = 20/9 := by
sorry

end sum_xyz_equals_twenty_ninths_l1565_156531


namespace two_equal_roots_sum_l1565_156532

theorem two_equal_roots_sum (a : ℝ) (α β : ℝ) :
  (∃! (x : ℝ), x ∈ Set.Ioo 0 (2 * Real.pi) ∧ 3 * Real.sin x + 4 * Real.cos x = a) →
  (α ∈ Set.Ioo 0 (2 * Real.pi) ∧ 3 * Real.sin α + 4 * Real.cos α = a) →
  (β ∈ Set.Ioo 0 (2 * Real.pi) ∧ 3 * Real.sin β + 4 * Real.cos β = a) →
  (α + β = Real.pi - 2 * Real.arcsin (4/5) ∨ α + β = 3 * Real.pi - 2 * Real.arcsin (4/5)) :=
by sorry


end two_equal_roots_sum_l1565_156532


namespace arithmetic_sequence_formula_l1565_156542

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem stating the general formula for the n-th term of the sequence -/
theorem arithmetic_sequence_formula (seq : ArithmeticSequence) 
  (h3 : (S seq 3)^2 = 9 * (S seq 2))
  (h4 : S seq 4 = 4 * (S seq 2)) :
  ∀ n : ℕ, seq.a n = (4 : ℚ) / 9 * (2 * n - 1) := by
  sorry

end arithmetic_sequence_formula_l1565_156542


namespace simplify_expression_evaluate_expression_l1565_156504

-- Problem 1
theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((1) * (2 * a^(1/2) * b^(1/3)) * (a^(2/3) * b^(1/2))) / ((1/3) * a^(1/6) * b^(5/6)) = 6 * a :=
sorry

-- Problem 2
theorem evaluate_expression :
  (2 * (9/16)^(1/2) + 10^(Real.log 9 / Real.log 10 - 2 * Real.log 2 / Real.log 10) + 
   Real.log (4 * Real.exp 3) - Real.log 8 / Real.log 9 * Real.log 33 / Real.log 4) = 7/2 :=
sorry

end simplify_expression_evaluate_expression_l1565_156504


namespace coefficient_x_squared_in_expansion_l1565_156546

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (x - 1)^4 = x^4 - 4*x^3 + 6*x^2 - 4*x + 1 := by
  sorry

#check coefficient_x_squared_in_expansion

end coefficient_x_squared_in_expansion_l1565_156546


namespace total_books_stu_and_albert_l1565_156534

/-- Given that Stu has 9 books and Albert has 4 times as many books as Stu,
    prove that the total number of books Stu and Albert have is 45. -/
theorem total_books_stu_and_albert :
  let stu_books : ℕ := 9
  let albert_books : ℕ := 4 * stu_books
  stu_books + albert_books = 45 := by
sorry

end total_books_stu_and_albert_l1565_156534


namespace printers_finish_time_l1565_156547

-- Define the start time of the first printer
def printer1_start : Real := 9

-- Define the time when half the tasks are completed
def half_tasks_time : Real := 12.5

-- Define the start time of the second printer
def printer2_start : Real := 13

-- Define the time taken by the second printer to complete its set amount
def printer2_duration : Real := 2

-- Theorem to prove
theorem printers_finish_time :
  let printer1_duration := 2 * (half_tasks_time - printer1_start)
  let printer1_finish := printer1_start + printer1_duration
  let printer2_finish := printer2_start + printer2_duration
  max printer1_finish printer2_finish = 16 := by
  sorry

end printers_finish_time_l1565_156547


namespace well_capacity_1200_gallons_l1565_156502

/-- The capacity of a well filled by two pipes -/
def well_capacity (rate1 rate2 : ℝ) (time : ℝ) : ℝ :=
  (rate1 + rate2) * time

/-- Theorem stating the capacity of the well -/
theorem well_capacity_1200_gallons (rate1 rate2 time : ℝ) 
  (h1 : rate1 = 48)
  (h2 : rate2 = 192)
  (h3 : time = 5) :
  well_capacity rate1 rate2 time = 1200 := by
  sorry

end well_capacity_1200_gallons_l1565_156502


namespace two_trains_problem_l1565_156505

/-- The problem of two trains approaching each other -/
theorem two_trains_problem (length1 length2 speed1 clear_time : ℝ) 
  (h1 : length1 = 120)
  (h2 : length2 = 300)
  (h3 : speed1 = 42)
  (h4 : clear_time = 20.99832013438925) : 
  ∃ speed2 : ℝ, speed2 = 30 := by
  sorry

end two_trains_problem_l1565_156505


namespace inscribed_angles_sum_l1565_156567

theorem inscribed_angles_sum (circle : Real) (x y : Real) : 
  (circle > 0) →  -- circle has positive circumference
  (x = (2 / 12) * circle) →  -- x subtends 2/12 of the circle
  (y = (4 / 12) * circle) →  -- y subtends 4/12 of the circle
  (∃ (central_x central_y : Real), 
    central_x = 2 * x ∧ 
    central_y = 2 * y ∧ 
    central_x + central_y = circle) →  -- inscribed angle theorem
  x + y = 90 := by
  sorry

end inscribed_angles_sum_l1565_156567
