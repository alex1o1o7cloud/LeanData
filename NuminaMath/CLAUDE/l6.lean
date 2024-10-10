import Mathlib

namespace last_page_cards_l6_678

/-- Calculates the number of cards on the last page after reorganization --/
def cards_on_last_page (initial_albums : ℕ) (initial_pages_per_album : ℕ) 
  (initial_cards_per_page : ℕ) (new_cards_per_page : ℕ) (full_albums : ℕ) 
  (extra_full_pages : ℕ) : ℕ :=
  let total_cards := initial_albums * initial_pages_per_album * initial_cards_per_page
  let cards_in_full_albums := full_albums * initial_pages_per_album * new_cards_per_page
  let cards_in_extra_pages := extra_full_pages * new_cards_per_page
  let remaining_cards := total_cards - (cards_in_full_albums + cards_in_extra_pages)
  remaining_cards - (extra_full_pages * new_cards_per_page)

/-- Theorem stating that given the problem conditions, the last page contains 40 cards --/
theorem last_page_cards : 
  cards_on_last_page 10 50 8 12 5 40 = 40 := by
  sorry

end last_page_cards_l6_678


namespace arithmetic_square_root_l6_650

theorem arithmetic_square_root (a : ℝ) (h : a > 0) : Real.sqrt a = (a ^ (1/2 : ℝ)) := by
  sorry

end arithmetic_square_root_l6_650


namespace miss_two_consecutive_probability_l6_619

/-- The probability of hitting a target in one shot. -/
def hit_probability : ℝ := 0.8

/-- The probability of missing a target in one shot. -/
def miss_probability : ℝ := 1 - hit_probability

/-- The probability of missing a target in two consecutive shots. -/
def miss_two_consecutive : ℝ := miss_probability * miss_probability

theorem miss_two_consecutive_probability :
  miss_two_consecutive = 0.04 := by sorry

end miss_two_consecutive_probability_l6_619


namespace initial_average_age_l6_629

theorem initial_average_age (n : ℕ) (new_person_age : ℕ) (new_average : ℚ) :
  n = 17 ∧ new_person_age = 32 ∧ new_average = 15 →
  ∃ initial_average : ℚ, 
    initial_average * n + new_person_age = new_average * (n + 1) ∧
    initial_average = 14 :=
by sorry

end initial_average_age_l6_629


namespace integer_sum_problem_l6_620

theorem integer_sum_problem : ∃ (a b : ℕ+), 
  (a * b + a + b = 167) ∧ 
  (Nat.gcd a.val b.val = 1) ∧ 
  (a < 30) ∧ (b < 30) ∧ 
  (a + b = 24) := by
  sorry

end integer_sum_problem_l6_620


namespace train_speed_l6_645

/-- The speed of a train traveling between two points, given the conditions of the problem -/
theorem train_speed (distance : ℝ) (return_speed : ℝ) (time_difference : ℝ) :
  distance = 480 ∧ 
  return_speed = 120 ∧ 
  time_difference = 1 →
  ∃ speed : ℝ, 
    speed = 160 ∧ 
    distance / speed + time_difference = distance / return_speed :=
by
  sorry

end train_speed_l6_645


namespace count_valid_digits_l6_604

def is_valid_digit (n : ℕ) : Prop :=
  n < 10

def appended_number (n : ℕ) : ℕ :=
  7580 + n

theorem count_valid_digits :
  ∃ (valid_digits : Finset ℕ),
    (∀ d ∈ valid_digits, is_valid_digit d ∧ (appended_number d).mod 4 = 0) ∧
    (∀ d, is_valid_digit d ∧ (appended_number d).mod 4 = 0 → d ∈ valid_digits) ∧
    valid_digits.card = 3 :=
by sorry

end count_valid_digits_l6_604


namespace incorrect_expression_for_repeating_decimal_l6_626

/-- Represents a repeating decimal number -/
structure RepeatingDecimal where
  nonRepeating : ℕ → ℕ  -- X: non-repeating part
  repeating : ℕ → ℕ     -- Y: repeating part
  a : ℕ                 -- length of non-repeating part
  b : ℕ                 -- length of repeating part

/-- Converts a RepeatingDecimal to a real number -/
def toReal (v : RepeatingDecimal) : ℝ :=
  sorry

/-- Theorem stating that the given expression is incorrect for repeating decimals -/
theorem incorrect_expression_for_repeating_decimal (v : RepeatingDecimal) :
  ∃ (x y : ℕ), 10^v.a * (10^v.b - 1) * (toReal v) ≠ x * (y - 1) := by
  sorry

end incorrect_expression_for_repeating_decimal_l6_626


namespace survey_result_l6_682

theorem survey_result (X : ℝ) (total : ℕ) (h_total : total ≥ 100) : ℝ :=
  let liked_A := X
  let liked_both := 23
  let liked_neither := 23
  let liked_B := 100 - X
  sorry

end survey_result_l6_682


namespace jacoby_trip_savings_l6_654

/-- The amount Jacoby needs for his trip to Brickville --/
def trip_cost : ℝ := 8000

/-- Jacoby's hourly wage --/
def hourly_wage : ℝ := 25

/-- Hours Jacoby worked --/
def hours_worked : ℝ := 15

/-- Tax rate on Jacoby's salary --/
def tax_rate : ℝ := 0.1

/-- Price of each cookie --/
def cookie_price : ℝ := 5

/-- Number of cookies sold --/
def cookies_sold : ℕ := 30

/-- Weekly tutoring earnings --/
def tutoring_weekly : ℝ := 100

/-- Weeks of tutoring --/
def tutoring_weeks : ℕ := 4

/-- Cost of lottery ticket --/
def lottery_ticket_cost : ℝ := 20

/-- Lottery winnings --/
def lottery_winnings : ℝ := 700

/-- Percentage of lottery winnings given to friend --/
def lottery_share : ℝ := 0.3

/-- Gift amount from each sister --/
def sister_gift : ℝ := 700

/-- Number of sisters --/
def number_of_sisters : ℕ := 2

/-- Cost of keychain --/
def keychain_cost : ℝ := 3

/-- Cost of backpack --/
def backpack_cost : ℝ := 47

/-- The amount Jacoby still needs for his trip --/
def amount_needed : ℝ := 5286.50

theorem jacoby_trip_savings : 
  trip_cost - (
    (hourly_wage * hours_worked * (1 - tax_rate)) +
    (cookie_price * cookies_sold) +
    (tutoring_weekly * tutoring_weeks) +
    ((lottery_winnings - lottery_ticket_cost) * (1 - lottery_share)) +
    (sister_gift * number_of_sisters) -
    (keychain_cost + backpack_cost)
  ) = amount_needed := by sorry

end jacoby_trip_savings_l6_654


namespace absolute_value_inequality_solution_set_l6_613

theorem absolute_value_inequality_solution_set :
  {x : ℝ | 1 ≤ |x + 2| ∧ |x + 2| ≤ 5} = {x : ℝ | (-7 ≤ x ∧ x ≤ -3) ∨ (-1 ≤ x ∧ x ≤ 3)} := by
  sorry

end absolute_value_inequality_solution_set_l6_613


namespace tournament_games_count_l6_653

/-- Calculates the total number of games played in a tournament given the ratio of outcomes and the number of games won. -/
def totalGamesPlayed (ratioWon ratioLost ratioTied : ℕ) (gamesWon : ℕ) : ℕ :=
  let partValue := gamesWon / ratioWon
  let gamesLost := ratioLost * partValue
  let gamesTied := ratioTied * partValue
  gamesWon + gamesLost + gamesTied

/-- Theorem stating that given the specified ratio and number of games won, the total games played is 96. -/
theorem tournament_games_count :
  totalGamesPlayed 7 4 5 42 = 96 := by
  sorry

#eval totalGamesPlayed 7 4 5 42

end tournament_games_count_l6_653


namespace max_product_of_three_numbers_l6_612

theorem max_product_of_three_numbers (n : ℕ) :
  let S := Finset.range (3 * n + 1) \ {0}
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    a < b ∧ b < c ∧
    a + b + c = 3 * n ∧
    ∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S →
      x < y → y < z →
      x + y + z = 3 * n →
      x * y * z ≤ a * b * c ∧
    a * b * c = n^3 - n :=
by sorry

end max_product_of_three_numbers_l6_612


namespace situp_difference_l6_610

/-- The number of sit-ups Ken can do -/
def ken_situps : ℕ := 20

/-- The number of sit-ups Nathan can do -/
def nathan_situps : ℕ := 2 * ken_situps

/-- The number of sit-ups Bob can do -/
def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

/-- The number of sit-ups Emma can do -/
def emma_situps : ℕ := bob_situps / 3

/-- The theorem stating the difference in sit-ups between the group (Nathan, Bob, Emma) and Ken -/
theorem situp_difference : nathan_situps + bob_situps + emma_situps - ken_situps = 60 := by
  sorry

end situp_difference_l6_610


namespace parabolas_intersection_sum_l6_658

/-- The parabolas y = x^2 + 15x + 32 and x = y^2 + 49y + 593 meet at one point (x₀, y₀). -/
theorem parabolas_intersection_sum (x₀ y₀ : ℝ) :
  y₀ = x₀^2 + 15*x₀ + 32 ∧ 
  x₀ = y₀^2 + 49*y₀ + 593 →
  x₀ + y₀ = -33 :=
by sorry

end parabolas_intersection_sum_l6_658


namespace chord_max_surface_area_l6_635

/-- 
Given a circle with radius R, the chord of length R√2 maximizes the surface area 
of the cylindrical shell formed when rotating the chord around the diameter parallel to it.
-/
theorem chord_max_surface_area (R : ℝ) (R_pos : R > 0) : 
  let chord_length (x : ℝ) := 2 * x
  let surface_area (x : ℝ) := 4 * Real.pi * x * Real.sqrt (R^2 - x^2)
  ∃ (x : ℝ), x > 0 ∧ x < R ∧ 
    chord_length x = R * Real.sqrt 2 ∧
    ∀ (y : ℝ), y > 0 → y < R → surface_area x ≥ surface_area y :=
by sorry

end chord_max_surface_area_l6_635


namespace product_increase_l6_643

theorem product_increase : ∃ n : ℤ, 
  53 * n = 1585 ∧ 
  53 * n - 35 * n = 535 :=
by sorry

end product_increase_l6_643


namespace max_regions_five_lines_l6_607

/-- The number of regions created by n intersecting lines in a plane --/
def num_regions (n : ℕ) : ℕ := sorry

/-- The maximum number of regions created by n intersecting lines in a rectangle --/
def max_regions_rectangle (n : ℕ) : ℕ := num_regions n

theorem max_regions_five_lines : 
  max_regions_rectangle 5 = 16 := by sorry

end max_regions_five_lines_l6_607


namespace function_properties_l6_663

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1)

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := a^(2*x) - 4*a^x + 8

-- State the theorem
theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 3 = 1/9) :
  a = 1/3 ∧ Set.Icc 4 53 = Set.image (g (1/3)) (Set.Icc (-2) 1) := by sorry

end

end function_properties_l6_663


namespace order_of_logarithmic_expressions_l6_617

noncomputable section

def a : ℝ := Real.log 2 / 2
def b : ℝ := Real.log 3 / 3
def c : ℝ := Real.log Real.pi / Real.pi
def d : ℝ := Real.log 2.72 / 2.72
def f : ℝ := (Real.sqrt 10 * Real.log 10) / 20

theorem order_of_logarithmic_expressions :
  a < f ∧ f < c ∧ c < b ∧ b < d := by sorry

end order_of_logarithmic_expressions_l6_617


namespace B_equals_interval_A_union_C_equals_A_l6_667

-- Define sets A, B, and C
def A : Set ℝ := {x | 2 * x^2 - 9 * x + 4 > 0}
def B : Set ℝ := {y | ∃ x ∈ (Set.univ \ A), y = -x^2 + 2 * x}
def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x ≤ 2 * m - 1}

-- Theorem 1: B is equal to the closed interval [-8, 1]
theorem B_equals_interval : B = Set.Icc (-8) 1 := by sorry

-- Theorem 2: A ∪ C = A if and only if m ≤ 2 or m ≥ 3
theorem A_union_C_equals_A (m : ℝ) : A ∪ C m = A ↔ m ≤ 2 ∨ m ≥ 3 := by sorry

end B_equals_interval_A_union_C_equals_A_l6_667


namespace part_I_part_II_l6_628

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}

-- Part I
theorem part_I (a : ℝ) (h : a = 3) :
  (A ∪ B a = {x | 1 ≤ x ∧ x ≤ 5}) ∧
  (B a ∩ (Set.univ \ A) = {x | 4 < x ∧ x ≤ 5}) := by
  sorry

-- Part II
theorem part_II (a : ℝ) :
  B a ⊆ A ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

end part_I_part_II_l6_628


namespace correct_missile_sampling_l6_602

/-- Represents a systematic sampling of missiles -/
structure MissileSampling where
  total : ℕ
  sample_size : ℕ
  first : ℕ
  interval : ℕ

/-- Generates the sequence of sampled missile numbers -/
def generate_sequence (ms : MissileSampling) : List ℕ :=
  List.range ms.sample_size |>.map (λ i => ms.first + i * ms.interval)

/-- Checks if all elements in the list are within the valid range -/
def valid_range (l : List ℕ) (max : ℕ) : Prop :=
  l.all (λ x => x > 0 ∧ x ≤ max)

theorem correct_missile_sampling :
  let ms : MissileSampling := {
    total := 60,
    sample_size := 6,
    first := 3,
    interval := 10
  }
  let sequence := generate_sequence ms
  (sequence = [3, 13, 23, 33, 43, 53]) ∧
  (ms.interval = ms.total / ms.sample_size) ∧
  (valid_range sequence ms.total) :=
by sorry

end correct_missile_sampling_l6_602


namespace borrowing_ten_sheets_avg_49_l6_625

/-- Represents a notebook with a given number of sheets --/
structure Notebook where
  sheets : ℕ
  pages : ℕ
  h_pages : pages = 2 * sheets

/-- Represents a borrowing of consecutive sheets from a notebook --/
structure Borrowing where
  notebook : Notebook
  borrowed_sheets : ℕ
  start_sheet : ℕ
  h_consecutive : start_sheet + borrowed_sheets ≤ notebook.sheets

/-- Calculates the average page number of remaining sheets --/
def average_remaining_pages (b : Borrowing) : ℚ :=
  let total_pages := b.notebook.pages
  let borrowed_pages := 2 * b.borrowed_sheets
  let remaining_pages := total_pages - borrowed_pages
  let sum_before := b.start_sheet * (2 * b.start_sheet + 1)
  let sum_after := ((total_pages - 2 * (b.start_sheet + b.borrowed_sheets) + 1) * 
                    (2 * (b.start_sheet + b.borrowed_sheets) + total_pages)) / 2
  (sum_before + sum_after) / remaining_pages

/-- Theorem stating that borrowing 10 sheets results in an average of 49 for remaining pages --/
theorem borrowing_ten_sheets_avg_49 (n : Notebook) (h_100_pages : n.pages = 100) :
  ∃ b : Borrowing, b.notebook = n ∧ b.borrowed_sheets = 10 ∧ average_remaining_pages b = 49 := by
  sorry


end borrowing_ten_sheets_avg_49_l6_625


namespace andrew_work_hours_l6_601

theorem andrew_work_hours : 
  let day1 : ℝ := 1.5
  let day2 : ℝ := 2.75
  let day3 : ℝ := 3.25
  day1 + day2 + day3 = 7.5 := by
sorry

end andrew_work_hours_l6_601


namespace valid_stacks_count_l6_669

/-- Represents a card with a color and number -/
structure Card where
  color : Nat
  number : Nat

/-- Represents a stack of cards -/
def Stack := List Card

/-- Checks if a stack is valid according to the rules -/
def isValidStack (stack : Stack) : Bool :=
  sorry

/-- Generates all possible stacks -/
def generateStacks : List Stack :=
  sorry

/-- Counts the number of valid stacking sequences -/
def countValidStacks : Nat :=
  (generateStacks.filter isValidStack).length

/-- The main theorem stating that the number of valid stacking sequences is 6 -/
theorem valid_stacks_count :
  let redCards := [1, 2, 3, 4]
  let blueCards := [2, 3, 4]
  let greenCards := [5, 6, 7]
  countValidStacks = 6 := by
  sorry

end valid_stacks_count_l6_669


namespace sequence_characterization_l6_623

theorem sequence_characterization (a : ℕ → ℕ) :
  (∀ n : ℕ, n ≥ 1 → a (n + 2) * (a (n + 1) - 1) = a n * (a (n + 1) + 1)) →
  ∃ k : ℕ, ∀ n : ℕ, n ≥ 1 → a n = k + n :=
sorry

end sequence_characterization_l6_623


namespace complex_fraction_equality_l6_661

theorem complex_fraction_equality : Complex.I * Complex.I = -1 → (3 : ℂ) / (1 - Complex.I)^2 = (3 / 2 : ℂ) * Complex.I := by
  sorry

end complex_fraction_equality_l6_661


namespace solve_equation_l6_624

theorem solve_equation (n m q : ℚ) : 
  (7 / 8 : ℚ) = n / 96 ∧ 
  (7 / 8 : ℚ) = (m + n) / 112 ∧ 
  (7 / 8 : ℚ) = (q - m) / 144 → 
  q = 140 := by
  sorry

end solve_equation_l6_624


namespace triangle_properties_l6_694

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.b * Real.sin abc.A = Real.sqrt 3 * abc.a * Real.cos abc.B)
  (h2 : abc.b = 3)
  (h3 : Real.sin abc.C = 2 * Real.sin abc.A) :
  abc.B = π / 3 ∧ 
  abc.a = Real.sqrt 3 ∧ 
  abc.c = 2 * Real.sqrt 3 ∧ 
  (1 / 2 : ℝ) * abc.a * abc.c * Real.sin abc.B = 3 * Real.sqrt 3 / 2 := by
  sorry


end triangle_properties_l6_694


namespace yahs_to_bahs_1500_l6_697

/-- Conversion rates between bahs, rahs, and yahs -/
structure ConversionRates where
  bahs_to_rahs : ℚ
  rahs_to_yahs : ℚ

/-- Calculate the number of bahs equivalent to a given number of yahs -/
def yahs_to_bahs (rates : ConversionRates) (yahs : ℚ) : ℚ :=
  yahs * rates.rahs_to_yahs * rates.bahs_to_rahs

/-- Theorem stating that 1500 yahs are equivalent to 600 bahs given the conversion rates -/
theorem yahs_to_bahs_1500 (rates : ConversionRates) 
    (h1 : rates.bahs_to_rahs = 30 / 20)
    (h2 : rates.rahs_to_yahs = 12 / 20) : 
  yahs_to_bahs rates 1500 = 600 := by
  sorry

end yahs_to_bahs_1500_l6_697


namespace arithmetic_sequence_cos_sum_l6_675

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cos_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 5 + a 9 = 5 * Real.pi →
  Real.cos (a 2 + a 8) = -1/2 := by
  sorry

end arithmetic_sequence_cos_sum_l6_675


namespace fraction_equivalence_l6_691

theorem fraction_equivalence : (8 : ℚ) / (7 * 67) = (0.8 : ℚ) / (0.7 * 67) := by
  sorry

end fraction_equivalence_l6_691


namespace solution_satisfies_conditions_l6_672

/-- Represents the number of bicycles, cars, and carts passing in front of a house. -/
structure VehicleCount where
  bicycles : ℕ
  cars : ℕ
  carts : ℕ

/-- Checks if the given vehicle count satisfies the problem conditions. -/
def satisfiesConditions (vc : VehicleCount) : Prop :=
  (vc.bicycles + vc.cars = 3 * vc.carts) ∧
  (2 * vc.bicycles + 4 * vc.cars + vc.bicycles + vc.cars = 100)

/-- The solution to the vehicle counting problem. -/
def solution : VehicleCount :=
  { bicycles := 10, cars := 14, carts := 8 }

/-- Theorem stating that the given solution satisfies the problem conditions. -/
theorem solution_satisfies_conditions : satisfiesConditions solution := by
  sorry


end solution_satisfies_conditions_l6_672


namespace f_is_periodic_l6_686

/-- Given two functions f and g on ℝ satisfying certain conditions, 
    prove that f is periodic -/
theorem f_is_periodic 
  (f g : ℝ → ℝ)
  (h₁ : f 0 = 1)
  (h₂ : ∃ a : ℝ, a ≠ 0 ∧ g a = 1)
  (h₃ : ∀ x, g (-x) = -g x)
  (h₄ : ∀ x y, f (x - y) = f x * f y + g x * g y) :
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x :=
sorry

end f_is_periodic_l6_686


namespace inequality_not_always_hold_l6_662

theorem inequality_not_always_hold (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ Real.sqrt a + Real.sqrt b > Real.sqrt 2 := by
  sorry

end inequality_not_always_hold_l6_662


namespace norris_remaining_money_l6_649

/-- Calculates the remaining money for Norris after savings and spending --/
def remaining_money (september_savings october_savings november_savings spent : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - spent

/-- Theorem stating that Norris has $10 left after his savings and spending --/
theorem norris_remaining_money :
  remaining_money 29 25 31 75 = 10 := by
  sorry

end norris_remaining_money_l6_649


namespace circle_point_x_coordinate_l6_681

theorem circle_point_x_coordinate :
  ∀ (x : ℝ),
  let center_x : ℝ := (-3 + 21) / 2
  let center_y : ℝ := 0
  let radius : ℝ := (21 - (-3)) / 2
  (x - center_x)^2 + (12 - center_y)^2 = radius^2 →
  x = 9 :=
by
  sorry

end circle_point_x_coordinate_l6_681


namespace range_of_a_l6_648

open Real

theorem range_of_a (a : ℝ) : 
  let P := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
  let Q := ∀ x y : ℝ, x < y → -(5-2*a)^x > -(5-2*a)^y
  (P ∧ ¬Q) ∨ (¬P ∧ Q) → a ≤ -2 :=
by sorry

end range_of_a_l6_648


namespace spirangle_length_is_10301_l6_646

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def spirangle_length (a₁ : ℕ) (d : ℕ) (last_seq : ℕ) (final_seg : ℕ) : ℕ :=
  let n := (last_seq - a₁) / d + 1
  arithmetic_sequence_sum a₁ d n + final_seg

theorem spirangle_length_is_10301 :
  spirangle_length 2 2 200 201 = 10301 :=
by sorry

end spirangle_length_is_10301_l6_646


namespace greatest_integer_radius_l6_657

theorem greatest_integer_radius (r : ℝ) : r > 0 → r * r * Real.pi < 75 * Real.pi → ∃ n : ℕ, n = 8 ∧ (∀ m : ℕ, m * m * Real.pi < 75 * Real.pi → m ≤ n) := by
  sorry

end greatest_integer_radius_l6_657


namespace correct_average_after_error_correction_l6_676

theorem correct_average_after_error_correction (numbers : List ℝ) 
  (h1 : numbers.length = 15)
  (h2 : numbers.sum / numbers.length = 20)
  (h3 : ∃ a b c, a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ 
               a = 35 ∧ b = 60 ∧ c = 25) :
  let corrected_numbers := numbers.map (fun x => 
    if x = 35 then 45 else if x = 60 then 58 else if x = 25 then 30 else x)
  corrected_numbers.sum / corrected_numbers.length = 20.8666666667 := by
sorry

end correct_average_after_error_correction_l6_676


namespace initial_amount_proof_l6_673

/-- 
Proves that if an amount increases by 1/8th of itself each year for two years 
and becomes 72900, then the initial amount was 57600.
-/
theorem initial_amount_proof (P : ℝ) : 
  (((P + P / 8) + (P + P / 8) / 8) = 72900) → P = 57600 :=
by sorry

end initial_amount_proof_l6_673


namespace intersection_of_M_and_N_l6_659

def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {0, 2, 3, 5}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end intersection_of_M_and_N_l6_659


namespace three_digit_perfect_cube_divisible_by_16_l6_699

theorem three_digit_perfect_cube_divisible_by_16 :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^3 ∧ n % 16 = 0 :=
by sorry

end three_digit_perfect_cube_divisible_by_16_l6_699


namespace lawn_mowing_theorem_l6_637

/-- Represents the time (in hours) it takes to mow the entire lawn -/
def MaryTime : ℚ := 4
def TomTime : ℚ := 5

/-- Represents the fraction of the lawn mowed per hour -/
def MaryRate : ℚ := 1 / MaryTime
def TomRate : ℚ := 1 / TomTime

/-- Represents the time Tom works alone -/
def TomAloneTime : ℚ := 3

/-- Represents the time Mary and Tom work together -/
def TogetherTime : ℚ := 1

/-- The fraction of lawn remaining to be mowed -/
def RemainingFraction : ℚ := 1 / 20

theorem lawn_mowing_theorem :
  1 - (TomRate * TomAloneTime + (MaryRate + TomRate) * TogetherTime) = RemainingFraction := by
  sorry

end lawn_mowing_theorem_l6_637


namespace consecutive_numbers_product_l6_698

theorem consecutive_numbers_product (n : ℕ) : 
  (n + (n + 1) = 11) → (n * (n + 1) * (n + 2) = 210) := by
  sorry

end consecutive_numbers_product_l6_698


namespace larger_number_problem_l6_652

theorem larger_number_problem (x y : ℕ) 
  (h1 : y - x = 1365)
  (h2 : y = 6 * x + 15) : 
  y = 1635 := by sorry

end larger_number_problem_l6_652


namespace arithmetic_sequence_fifth_term_l6_618

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  sum_property : a 1 + a 3 = 8
  geometric_mean : a 4 ^ 2 = a 2 * a 9
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term (seq : ArithmeticSequence) : seq.a 5 = 13 := by
  sorry

end arithmetic_sequence_fifth_term_l6_618


namespace complex_fraction_equality_l6_642

theorem complex_fraction_equality (z : ℂ) (h : z = 1 + I) :
  (3 * I) / (z + 1) = 3/5 + 6/5 * I := by
  sorry

end complex_fraction_equality_l6_642


namespace arithmetic_proof_l6_641

theorem arithmetic_proof : (100 + 20 / 90) * 90 = 120 := by
  sorry

end arithmetic_proof_l6_641


namespace log_absolute_equality_l6_639

/-- Given a function f(x) = |log x|, prove that if 0 < a < b and f(a) = f(b), then ab = 1 -/
theorem log_absolute_equality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) 
  (h3 : |Real.log a| = |Real.log b|) : a * b = 1 := by
  sorry

end log_absolute_equality_l6_639


namespace boxes_sold_saturday_l6_660

theorem boxes_sold_saturday (saturday_sales : ℕ) (sunday_sales : ℕ) : 
  sunday_sales = saturday_sales + saturday_sales / 2 →
  saturday_sales + sunday_sales = 150 →
  saturday_sales = 60 := by
sorry

end boxes_sold_saturday_l6_660


namespace margos_walking_distance_l6_609

/-- Margo's Walking Problem -/
theorem margos_walking_distance
  (outbound_time : Real) (return_time : Real)
  (outbound_speed : Real) (return_speed : Real)
  (average_speed : Real)
  (h1 : outbound_time = 15 / 60)
  (h2 : return_time = 30 / 60)
  (h3 : outbound_speed = 5)
  (h4 : return_speed = 3)
  (h5 : average_speed = 3.6)
  (h6 : average_speed = (outbound_time + return_time) / 
        ((outbound_time / outbound_speed) + (return_time / return_speed))) :
  outbound_speed * outbound_time + return_speed * return_time = 2.75 := by
  sorry

#check margos_walking_distance

end margos_walking_distance_l6_609


namespace garden_area_l6_621

theorem garden_area (width length perimeter area : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  area = width * length →
  area = 243 := by
sorry

end garden_area_l6_621


namespace fraction_multiplication_l6_603

theorem fraction_multiplication (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (5*x + 5*y) / ((5*x) * (5*y)) = (1 / 5) * ((x + y) / (x * y)) :=
by sorry

end fraction_multiplication_l6_603


namespace point_transformation_l6_614

/-- Given a point B(5, -1) moved 3 units upwards to point A(a+1, 1-b), prove that a = 4 and b = -1 -/
theorem point_transformation (a b : ℝ) : 
  (5 : ℝ) = a + 1 ∧ 
  (1 : ℝ) - b = -1 + 3 → 
  a = 4 ∧ b = -1 := by sorry

end point_transformation_l6_614


namespace approximate_root_of_f_l6_692

def f (x : ℝ) := x^3 + x^2 - 2*x - 2

theorem approximate_root_of_f :
  f 1 = -2 →
  f 1.5 = 0.625 →
  f 1.25 = -0.984 →
  f 1.375 = -0.260 →
  f 1.438 = 0.165 →
  f 1.4065 = -0.052 →
  ∃ (root : ℝ), f root = 0 ∧ |root - 1.43| < 0.1 :=
by sorry

end approximate_root_of_f_l6_692


namespace eulers_formula_modulus_l6_630

theorem eulers_formula_modulus (i : ℂ) (π : ℝ) : 
  Complex.abs (Complex.exp (i * π / 3)) = 1 :=
by
  sorry

end eulers_formula_modulus_l6_630


namespace right_triangle_area_l6_638

theorem right_triangle_area (DF EF : ℝ) (angle_DEF : ℝ) :
  DF = 4 →
  angle_DEF = π / 4 →
  DF = EF →
  (1 / 2) * DF * EF = 8 :=
by sorry

end right_triangle_area_l6_638


namespace history_paper_pages_l6_627

/-- Given a paper due in 6 days with a required writing pace of 11 pages per day,
    the total number of pages in the paper is 66. -/
theorem history_paper_pages (days : ℕ) (pages_per_day : ℕ) (h1 : days = 6) (h2 : pages_per_day = 11) :
  days * pages_per_day = 66 := by
  sorry

end history_paper_pages_l6_627


namespace no_fixed_points_iff_a_in_open_interval_l6_693

theorem no_fixed_points_iff_a_in_open_interval
  (f : ℝ → ℝ)
  (h : ∀ x, f x = x^2 + a*x + 1) :
  (∀ x, f x ≠ x) ↔ a ∈ Set.Ioo (-1 : ℝ) 3 := by
  sorry

end no_fixed_points_iff_a_in_open_interval_l6_693


namespace min_surface_area_height_l6_655

/-- The height that minimizes the surface area of an open-top rectangular box with square base and volume 4 -/
theorem min_surface_area_height : ∃ (h : ℝ), h > 0 ∧ 
  (∀ (x : ℝ), x > 0 → x^2 * h = 4 → 
    ∀ (h' : ℝ), h' > 0 → x^2 * h' = 4 → 
      x^2 + 4*x*h ≤ x^2 + 4*x*h') ∧ 
  h = 1 := by
  sorry

end min_surface_area_height_l6_655


namespace abs_func_differentiable_l6_615

-- Define the absolute value function
def abs_func (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_func_differentiable :
  ∀ x : ℝ, x ≠ 0 →
    (DifferentiableAt ℝ abs_func x) ∧
    (deriv abs_func x = if x > 0 then 1 else -1) :=
by sorry

end abs_func_differentiable_l6_615


namespace least_addition_for_divisibility_l6_668

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), n = 3 ∧ 
  (∀ (m : ℕ), (1101 + m) % 24 = 0 → m ≥ n) ∧
  (1101 + n) % 24 = 0 := by
sorry

end least_addition_for_divisibility_l6_668


namespace sum_x_y_equals_seven_a_l6_608

theorem sum_x_y_equals_seven_a (a x y : ℝ) (h1 : a / x = 1 / 3) (h2 : a / y = 1 / 4) :
  x + y = 7 * a := by
  sorry

end sum_x_y_equals_seven_a_l6_608


namespace parallel_iff_abs_x_eq_two_l6_696

def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (x^2, 4*x)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_iff_abs_x_eq_two (x : ℝ) :
  (vector_a x ≠ (0, 0)) → (vector_b x ≠ (0, 0)) →
  (parallel (vector_a x) (vector_b x) ↔ |x| = 2) :=
sorry

end parallel_iff_abs_x_eq_two_l6_696


namespace x_one_minus_f_equals_four_power_500_l6_600

/-- Given x = (3 + √5)^500, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 4^500 -/
theorem x_one_minus_f_equals_four_power_500 :
  let x : ℝ := (3 + Real.sqrt 5) ^ 500
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 4 ^ 500 := by
  sorry

end x_one_minus_f_equals_four_power_500_l6_600


namespace quadratic_function_minimum_l6_687

/-- A quadratic function that takes specific values for consecutive integers -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 13 ∧ f (n + 1) = 13 ∧ f (n + 2) = 35

/-- The theorem stating the minimum value of the quadratic function -/
theorem quadratic_function_minimum (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = 41 / 4 := by
  sorry

end quadratic_function_minimum_l6_687


namespace remainder_equality_l6_606

theorem remainder_equality (A B C S T s t : ℕ) 
  (h1 : A > B)
  (h2 : A^2 % C = S)
  (h3 : B^2 % C = T)
  (h4 : (A^2 * B^2) % C = s)
  (h5 : (S * T) % C = t) :
  s = t := by
  sorry

end remainder_equality_l6_606


namespace step_increase_proof_l6_656

def daily_steps (x : ℕ) (week : ℕ) : ℕ :=
  1000 + (week - 1) * x

def weekly_steps (x : ℕ) (week : ℕ) : ℕ :=
  7 * daily_steps x week

def total_steps (x : ℕ) : ℕ :=
  weekly_steps x 1 + weekly_steps x 2 + weekly_steps x 3 + weekly_steps x 4

theorem step_increase_proof :
  ∃ x : ℕ, total_steps x = 70000 ∧ x = 1000 :=
by sorry

end step_increase_proof_l6_656


namespace rice_qualification_condition_l6_636

/-- The maximum number of chaff grains allowed in a qualified rice sample -/
def max_chaff_grains : ℕ := 7

/-- The total number of grains in the rice sample -/
def total_grains : ℕ := 235

/-- The maximum allowed percentage of chaff for qualified rice -/
def max_chaff_percentage : ℚ := 3 / 100

/-- Theorem stating the condition for qualified rice -/
theorem rice_qualification_condition (n : ℕ) :
  (n : ℚ) / total_grains ≤ max_chaff_percentage ↔ n ≤ max_chaff_grains :=
by sorry

end rice_qualification_condition_l6_636


namespace number_of_values_l6_631

theorem number_of_values (initial_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) (correct_mean : ℚ) : 
  initial_mean = 250 →
  incorrect_value = 135 →
  correct_value = 165 →
  correct_mean = 251 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * initial_mean + correct_value - incorrect_value = (n : ℚ) * correct_mean ∧
    n = 30 :=
by sorry

end number_of_values_l6_631


namespace parabola_hyperbola_intersection_l6_665

/-- Represents a parabola with vertex at the origin -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 4 * p * x

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : ℝ → ℝ → Prop := fun x y => x^2 / a^2 - y^2 / b^2 = 1

/-- The problem statement -/
theorem parabola_hyperbola_intersection (p : Parabola) (h : Hyperbola) : 
  (h.a > 0) →
  (h.b > 0) →
  (p.p = 2 * h.a) →  -- directrix passes through one focus of hyperbola
  (p.eq (3/2) (Real.sqrt 6)) →  -- intersection point
  (h.eq (3/2) (Real.sqrt 6)) →  -- intersection point
  (p.p = 1) ∧ (h.a^2 = 1/4) ∧ (h.b^2 = 3/4) := by
  sorry

#check parabola_hyperbola_intersection

end parabola_hyperbola_intersection_l6_665


namespace sample_survey_suitability_l6_632

-- Define the set of all surveys
def Surveys : Set Nat := {1, 2, 3, 4}

-- Define the characteristics of each survey
def is_destructive_testing (s : Nat) : Prop :=
  s = 1 ∨ s = 4

def has_large_scope (s : Nat) : Prop :=
  s = 2

def has_small_scope (s : Nat) : Prop :=
  s = 3

-- Define what makes a survey suitable for sampling
def suitable_for_sampling (s : Nat) : Prop :=
  is_destructive_testing s ∨ has_large_scope s

-- Theorem to prove
theorem sample_survey_suitability :
  {s ∈ Surveys | suitable_for_sampling s} = {1, 2, 4} := by
  sorry


end sample_survey_suitability_l6_632


namespace trajectory_of_tangent_circles_l6_616

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 25
def C2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop := y^2 / 9 + x^2 / 5 = 1 ∧ y ≠ 3

-- Theorem statement
theorem trajectory_of_tangent_circles :
  ∀ x y : ℝ, 
  (∃ r : ℝ, (x - 0)^2 + (y - (-1))^2 = (5 - r)^2 ∧ (x - 0)^2 + (y - 2)^2 = (r + 1)^2) →
  trajectory_equation x y :=
sorry

end trajectory_of_tangent_circles_l6_616


namespace complex_square_root_l6_647

theorem complex_square_root (z : ℂ) : 
  z^2 = -100 - 48*I ↔ z = 2 - 12*I ∨ z = -2 + 12*I :=
by sorry

end complex_square_root_l6_647


namespace nine_sequences_exist_l6_679

/-- An arithmetic sequence of natural numbers. -/
structure ArithSeq where
  first : ℕ
  diff : ℕ

/-- The nth term of an arithmetic sequence. -/
def ArithSeq.nthTerm (seq : ArithSeq) (n : ℕ) : ℕ :=
  seq.first + (n - 1) * seq.diff

/-- The sum of the first n terms of an arithmetic sequence. -/
def ArithSeq.sumFirstN (seq : ArithSeq) (n : ℕ) : ℕ :=
  n * (2 * seq.first + (n - 1) * seq.diff) / 2

/-- The property that the ratio of sum of first 2n terms to sum of first n terms is constant. -/
def ArithSeq.hasConstantRatio (seq : ArithSeq) : Prop :=
  ∀ n : ℕ, n > 0 → (seq.sumFirstN (2*n)) / (seq.sumFirstN n) = 4

/-- The property that 1971 is a term in the sequence. -/
def ArithSeq.contains1971 (seq : ArithSeq) : Prop :=
  ∃ k : ℕ, seq.nthTerm k = 1971

/-- The main theorem stating that there are exactly 9 sequences satisfying both properties. -/
theorem nine_sequences_exist : 
  ∃! (s : Finset ArithSeq), 
    s.card = 9 ∧ 
    (∀ seq ∈ s, seq.hasConstantRatio ∧ seq.contains1971) ∧
    (∀ seq : ArithSeq, seq.hasConstantRatio ∧ seq.contains1971 → seq ∈ s) :=
sorry

end nine_sequences_exist_l6_679


namespace mascs_age_l6_644

/-- Given that Masc is 7 years older than Sam and the sum of their ages is 27,
    prove that Masc's age is 17 years old. -/
theorem mascs_age (sam : ℕ) (masc : ℕ) 
    (h1 : masc = sam + 7)
    (h2 : sam + masc = 27) : 
  masc = 17 := by
  sorry

end mascs_age_l6_644


namespace proposition_b_l6_666

theorem proposition_b (a : ℝ) : 0 < a → a < 1 → a^3 < a := by sorry

end proposition_b_l6_666


namespace smallest_number_with_2020_divisors_l6_690

def number_of_divisors (n : ℕ) : ℕ := sorry

def is_prime_factorization (n : ℕ) (factors : List (ℕ × ℕ)) : Prop := sorry

theorem smallest_number_with_2020_divisors :
  ∃ (n : ℕ) (factors : List (ℕ × ℕ)),
    is_prime_factorization n factors ∧
    number_of_divisors n = 2020 ∧
    (∀ m : ℕ, m < n → number_of_divisors m ≠ 2020) ∧
    factors = [(2, 100), (3, 4), (5, 1), (7, 1)] :=
  sorry

end smallest_number_with_2020_divisors_l6_690


namespace exterior_angle_smaller_implies_obtuse_l6_685

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Predicate to check if a triangle is obtuse -/
def is_obtuse_triangle (t : Triangle) : Prop := sorry

/-- Predicate to check if an exterior angle is smaller than its adjacent interior angle -/
def exterior_angle_smaller_than_interior (t : Triangle) : Prop := sorry

/-- Theorem: If an exterior angle of a triangle is smaller than its adjacent interior angle, 
    then the triangle is obtuse -/
theorem exterior_angle_smaller_implies_obtuse (t : Triangle) :
  exterior_angle_smaller_than_interior t → is_obtuse_triangle t := by
  sorry

end exterior_angle_smaller_implies_obtuse_l6_685


namespace two_correct_probability_l6_684

/-- The number of houses and packages -/
def n : ℕ := 4

/-- The probability of exactly two packages being delivered to the correct houses -/
def prob_two_correct : ℚ := 1/4

/-- Theorem stating that the probability of exactly two packages being delivered 
    to the correct houses out of four is 1/4 -/
theorem two_correct_probability : 
  (n.choose 2 : ℚ) * (1/n) * (1/(n-1)) * (1/2) = prob_two_correct := by
  sorry

end two_correct_probability_l6_684


namespace equation_solution_range_l6_674

theorem equation_solution_range (a : ℝ) : 
  (∃ x : ℝ, (Real.exp (2 * x) + a * Real.exp x + 1 = 0)) ↔ a ≤ -2 :=
by sorry

end equation_solution_range_l6_674


namespace find_divisor_l6_680

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 17698) (h2 : quotient = 89) (h3 : remainder = 14) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 198 := by
  sorry

end find_divisor_l6_680


namespace factorization_cubic_l6_640

theorem factorization_cubic (a : ℝ) : a^3 - 10*a^2 + 25*a = a*(a-5)^2 := by
  sorry

end factorization_cubic_l6_640


namespace function_transformation_l6_611

theorem function_transformation (g : ℝ → ℝ) (h : ∀ x, g (x + 2) = 2 * x + 3) :
  ∀ x, g x = 2 * x - 1 := by
sorry

end function_transformation_l6_611


namespace myrtle_final_eggs_l6_683

/-- Calculate the number of eggs Myrtle has after her trip --/
def myrtle_eggs (num_hens : ℕ) (eggs_per_hen_per_day : ℕ) (days_away : ℕ) 
                (eggs_taken_by_neighbor : ℕ) (eggs_dropped : ℕ) : ℕ :=
  num_hens * eggs_per_hen_per_day * days_away - eggs_taken_by_neighbor - eggs_dropped

/-- Theorem stating the number of eggs Myrtle has --/
theorem myrtle_final_eggs : 
  myrtle_eggs 3 3 7 12 5 = 46 := by
  sorry

end myrtle_final_eggs_l6_683


namespace no_hexagon_with_special_point_l6_670

/-- A convex hexagon is represented by its vertices -/
def ConvexHexagon := Fin 6 → ℝ × ℝ

/-- Check if a hexagon is convex -/
def is_convex (h : ConvexHexagon) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Check if a point is inside a hexagon -/
def is_inside (p : ℝ × ℝ) (h : ConvexHexagon) : Prop := sorry

/-- The main theorem -/
theorem no_hexagon_with_special_point :
  ¬ ∃ (h : ConvexHexagon) (m : ℝ × ℝ),
    is_convex h ∧
    (∀ i : Fin 5, distance (h i) (h (i.succ)) > 1) ∧
    distance (h 5) (h 0) > 1 ∧
    is_inside m h ∧
    (∀ i : Fin 6, distance m (h i) < 1) :=
sorry

end no_hexagon_with_special_point_l6_670


namespace exercise_minutes_proof_l6_671

/-- The number of minutes Javier exercised daily -/
def javier_daily_minutes : ℕ := 50

/-- The number of days Javier exercised in a week -/
def javier_days : ℕ := 7

/-- The number of minutes Sanda exercised on each day she exercised -/
def sanda_daily_minutes : ℕ := 90

/-- The number of days Sanda exercised -/
def sanda_days : ℕ := 3

/-- The total number of minutes Javier and Sanda exercised -/
def total_exercise_minutes : ℕ := javier_daily_minutes * javier_days + sanda_daily_minutes * sanda_days

theorem exercise_minutes_proof : total_exercise_minutes = 620 := by
  sorry

end exercise_minutes_proof_l6_671


namespace fox_max_berries_l6_633

/-- The number of bear cubs -/
def num_cubs : ℕ := 100

/-- The total number of berries initially -/
def total_berries : ℕ := 2^num_cubs - 1

/-- The maximum number of berries the fox can eat -/
def max_fox_berries : ℕ := 1

/-- Theorem stating the maximum number of berries the fox can eat -/
theorem fox_max_berries :
  max_fox_berries = (total_berries % num_cubs) :=
by sorry

end fox_max_berries_l6_633


namespace problem_statement_l6_688

theorem problem_statement : (-4 : ℝ)^2007 * (-0.25 : ℝ)^2008 = -0.25 := by
  sorry

end problem_statement_l6_688


namespace Michael_birth_year_l6_634

def IMO_start_year : ℕ := 1959

def Michael_age_at_10th_IMO : ℕ := 15

def IMO_held_annually : Prop := ∀ n : ℕ, n ≥ IMO_start_year → ∃ m : ℕ, m = n - IMO_start_year + 1

theorem Michael_birth_year :
  IMO_held_annually →
  ∃ year : ℕ, year = IMO_start_year + 9 - Michael_age_at_10th_IMO ∧ year = 1953 :=
by sorry

end Michael_birth_year_l6_634


namespace friends_weight_loss_l6_622

/-- The combined weight loss of two friends over different periods -/
theorem friends_weight_loss (aleesia_weekly_loss : ℝ) (aleesia_weeks : ℕ)
                             (alexei_weekly_loss : ℝ) (alexei_weeks : ℕ) :
  aleesia_weekly_loss = 1.5 ∧ 
  aleesia_weeks = 10 ∧
  alexei_weekly_loss = 2.5 ∧ 
  alexei_weeks = 8 →
  aleesia_weekly_loss * aleesia_weeks + alexei_weekly_loss * alexei_weeks = 35 := by
  sorry

end friends_weight_loss_l6_622


namespace exponential_inequality_minimum_value_of_f_l6_664

-- Proposition 2
theorem exponential_inequality (x₁ x₂ : ℝ) :
  Real.exp ((x₁ + x₂) / 2) ≤ (Real.exp x₁ + Real.exp x₂) / 2 := by
  sorry

-- Proposition 4
def f (x : ℝ) : ℝ := (x - 2014)^2 - 2

theorem minimum_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = -2 := by
  sorry

end exponential_inequality_minimum_value_of_f_l6_664


namespace racetrack_probability_l6_651

/-- Represents a circular racetrack -/
structure Racetrack where
  length : ℝ
  isCircular : Bool

/-- Represents a car on the racetrack -/
structure Car where
  position : ℝ
  travelDistance : ℝ

/-- Calculates the probability of the car ending within the specified range -/
def probabilityOfEndingInRange (track : Racetrack) (car : Car) (targetPosition : ℝ) (range : ℝ) : ℝ :=
  sorry

theorem racetrack_probability (track : Racetrack) (car : Car) : 
  track.length = 3 →
  track.isCircular = true →
  car.travelDistance = 0.5 →
  probabilityOfEndingInRange track car 2.5 0.5 = 1/3 := by
  sorry

end racetrack_probability_l6_651


namespace intersection_implies_k_zero_l6_695

/-- Line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem intersection_implies_k_zero (m n : Line) (h1 : m.slope = 3) (h2 : m.intercept = 5)
    (h3 : n.intercept = -7) (h4 : m.equation (-4) (-7)) (h5 : n.equation (-4) (-7)) :
    n.slope = 0 := by
  sorry

end intersection_implies_k_zero_l6_695


namespace seven_people_arrangement_count_l6_677

def total_arrangements (n : ℕ) : ℕ := n.factorial

def adjacent_pair_arrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

def front_restricted_arrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

def end_restricted_arrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

def front_and_end_restricted_arrangements (n : ℕ) : ℕ := (n - 2).factorial * 2

theorem seven_people_arrangement_count : 
  total_arrangements 6 - 
  front_restricted_arrangements 6 - 
  end_restricted_arrangements 6 + 
  front_and_end_restricted_arrangements 6 = 1008 :=
by sorry

end seven_people_arrangement_count_l6_677


namespace inequality_implies_m_range_l6_689

theorem inequality_implies_m_range (m : ℝ) :
  (∀ x : ℝ, 4^x - m * 2^x + 1 > 0) → -2 < m ∧ m < 2 := by
  sorry

end inequality_implies_m_range_l6_689


namespace legislation_approval_probability_l6_605

/-- The probability of a voter approving the legislation -/
def p_approve : ℝ := 0.6

/-- The number of voters surveyed -/
def n : ℕ := 4

/-- The number of approving voters we're interested in -/
def k : ℕ := 2

/-- The probability of exactly k out of n voters approving the legislation -/
def prob_k_approve (p : ℝ) (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem legislation_approval_probability :
  prob_k_approve p_approve n k = 0.3456 := by
  sorry

end legislation_approval_probability_l6_605
