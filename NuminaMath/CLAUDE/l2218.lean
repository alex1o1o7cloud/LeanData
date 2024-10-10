import Mathlib

namespace jackie_working_hours_l2218_221887

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents the number of hours Jackie spends exercising -/
def exercise_hours : ℕ := 3

/-- Represents the number of hours Jackie spends sleeping -/
def sleep_hours : ℕ := 8

/-- Represents the number of hours Jackie has as free time -/
def free_time_hours : ℕ := 5

/-- Calculates the number of hours Jackie spends working -/
def working_hours : ℕ := hours_in_day - (sleep_hours + exercise_hours + free_time_hours)

theorem jackie_working_hours :
  working_hours = 8 := by sorry

end jackie_working_hours_l2218_221887


namespace first_thousand_diff_l2218_221895

/-- The sum of the first n even numbers starting from 0 -/
def sumEven (n : ℕ) : ℕ := n * (n - 1)

/-- The sum of the first n odd numbers starting from 1 -/
def sumOdd (n : ℕ) : ℕ := n^2

/-- The difference between the sum of the first n even numbers (including 0) 
    and the sum of the first n odd numbers -/
def diffEvenOdd (n : ℕ) : ℤ := (sumEven n : ℤ) - (sumOdd n : ℤ)

theorem first_thousand_diff : diffEvenOdd 1000 = -1000 := by
  sorry

end first_thousand_diff_l2218_221895


namespace cubic_root_theorem_l2218_221840

theorem cubic_root_theorem :
  ∃ (a b c : ℕ+) (x : ℝ),
    a = 1 ∧ b = 9 ∧ c = 1 ∧
    x = (Real.rpow a (1/3 : ℝ) + Real.rpow b (1/3 : ℝ) + 1) / c ∧
    27 * x^3 - 9 * x^2 - 9 * x - 3 = 0 := by
  sorry

end cubic_root_theorem_l2218_221840


namespace quadratic_function_inequality_theorem_l2218_221878

theorem quadratic_function_inequality_theorem :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = 0 → x = -1) ∧
    (∀ x : ℝ, x ≤ a * x^2 + b * x + c) ∧
    (∀ x : ℝ, a * x^2 + b * x + c ≤ (1 + x^2) / 2) ∧
    a = 1/4 ∧ b = 1/2 ∧ c = 1/4 :=
by sorry

end quadratic_function_inequality_theorem_l2218_221878


namespace hyperbola_range_l2218_221891

/-- The equation represents a hyperbola with parameter m -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) + y^2 / (m - 2) = 1

/-- The range of m for which the equation represents a hyperbola -/
theorem hyperbola_range :
  ∀ m : ℝ, is_hyperbola m ↔ m > -2 ∧ m < 2 := by sorry

end hyperbola_range_l2218_221891


namespace ceiling_sum_sqrt_l2218_221850

theorem ceiling_sum_sqrt : ⌈Real.sqrt 8⌉ + ⌈Real.sqrt 48⌉ + ⌈Real.sqrt 288⌉ = 27 := by
  sorry

end ceiling_sum_sqrt_l2218_221850


namespace ball_drawing_theorem_l2218_221824

/-- Represents the three bags of balls -/
inductive Bag
  | A
  | B
  | C

/-- The number of balls in each bag -/
def ballCount (bag : Bag) : Nat :=
  match bag with
  | Bag.A => 1
  | Bag.B => 2
  | Bag.C => 3

/-- The color of balls in each bag -/
def ballColor (bag : Bag) : String :=
  match bag with
  | Bag.A => "red"
  | Bag.B => "white"
  | Bag.C => "yellow"

/-- The number of ways to draw two balls of different colors -/
def differentColorDraws : Nat := sorry

/-- The number of ways to draw two balls of the same color -/
def sameColorDraws : Nat := sorry

theorem ball_drawing_theorem :
  differentColorDraws = 11 ∧ sameColorDraws = 4 := by sorry

end ball_drawing_theorem_l2218_221824


namespace student_average_greater_than_true_average_l2218_221889

theorem student_average_greater_than_true_average 
  (x y w z : ℝ) (h : x < y ∧ y < w ∧ w < z) : 
  (x + y + 2*w + 2*z) / 6 > (x + y + w + z) / 4 := by
  sorry

end student_average_greater_than_true_average_l2218_221889


namespace possible_values_of_a_l2218_221802

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 + b^3 = 35*x^3) 
  (h3 : a^2 - b^2 = 4*x^2) : 
  a = 2*x ∨ a = -2*x := by
sorry

end possible_values_of_a_l2218_221802


namespace unique_solution_E_l2218_221800

/-- Definition of the function E --/
def E (a b c : ℝ) : ℝ := a * b^2 + c

/-- Theorem stating that -1/16 is the unique solution to E(a, 3, 2) = E(a, 5, 3) --/
theorem unique_solution_E :
  ∃! a : ℝ, E a 3 2 = E a 5 3 ∧ a = -1/16 := by
  sorry

end unique_solution_E_l2218_221800


namespace actions_probability_is_one_four_hundredth_l2218_221805

/-- The probability of selecting specific letters from given words -/
def select_probability (total : ℕ) (choose : ℕ) (specific : ℕ) : ℚ :=
  (specific : ℚ) / (Nat.choose total choose : ℚ)

/-- The probability of selecting all letters from ACTIONS -/
def actions_probability : ℚ :=
  (select_probability 5 3 1) * (select_probability 5 2 1) * (select_probability 4 1 1)

/-- Theorem stating the probability of selecting all letters from ACTIONS -/
theorem actions_probability_is_one_four_hundredth :
  actions_probability = 1 / 400 := by sorry

end actions_probability_is_one_four_hundredth_l2218_221805


namespace min_value_reciprocal_sum_l2218_221898

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (3 : ℝ)^a * (3 : ℝ)^b = 3 → 
  ∀ x y : ℝ, x > 0 → y > 0 → (3 : ℝ)^x * (3 : ℝ)^y = 3 → 
  1/a + 1/b ≤ 1/x + 1/y := by sorry

end min_value_reciprocal_sum_l2218_221898


namespace events_mutually_exclusive_not_complementary_l2218_221845

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card
| White : Card

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "Person A gets the red card"
def event_A_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Person D gets the red card"
def event_D_red (d : Distribution) : Prop := d Person.D = Card.Red

-- Statement: The events are mutually exclusive but not complementary
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(event_A_red d ∧ event_D_red d)) ∧
  (∃ d : Distribution, ¬event_A_red d ∧ ¬event_D_red d) :=
sorry

end events_mutually_exclusive_not_complementary_l2218_221845


namespace percentage_of_sikh_boys_l2218_221827

/-- Given a school with the following student composition:
    - Total number of boys: 650
    - 44% are Muslims
    - 28% are Hindus
    - 117 boys are from other communities
    This theorem proves that 10% of the boys are Sikhs. -/
theorem percentage_of_sikh_boys (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (other : ℕ) :
  total = 650 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  other = 117 →
  (total - (muslim_percent * total + hindu_percent * total + other)) / total = 1 / 10 := by
  sorry


end percentage_of_sikh_boys_l2218_221827


namespace beta_max_success_ratio_l2218_221881

/-- Represents a participant's score in a day of the competition -/
structure DayScore where
  scored : ℕ
  attempted : ℕ

/-- Represents a participant's scores over the three-day competition -/
structure CompetitionScore where
  day1 : DayScore
  day2 : DayScore
  day3 : DayScore

/-- Alpha's competition score -/
def alpha_score : CompetitionScore := {
  day1 := { scored := 200, attempted := 300 }
  day2 := { scored := 150, attempted := 200 }
  day3 := { scored := 100, attempted := 100 }
}

/-- Calculates the success ratio for a DayScore -/
def success_ratio (score : DayScore) : ℚ :=
  score.scored / score.attempted

/-- Calculates the total success ratio for a CompetitionScore -/
def total_success_ratio (score : CompetitionScore) : ℚ :=
  (score.day1.scored + score.day2.scored + score.day3.scored) /
  (score.day1.attempted + score.day2.attempted + score.day3.attempted)

theorem beta_max_success_ratio :
  ∀ beta_score : CompetitionScore,
    (beta_score.day1.attempted + beta_score.day2.attempted + beta_score.day3.attempted = 600) →
    (beta_score.day1.attempted ≠ 300) →
    (beta_score.day2.attempted ≠ 200) →
    (beta_score.day1.scored > 0) →
    (beta_score.day2.scored > 0) →
    (beta_score.day3.scored > 0) →
    (success_ratio beta_score.day1 < success_ratio alpha_score.day1) →
    (success_ratio beta_score.day2 < success_ratio alpha_score.day2) →
    (success_ratio beta_score.day3 < success_ratio alpha_score.day3) →
    total_success_ratio beta_score ≤ 358 / 600 :=
by sorry

end beta_max_success_ratio_l2218_221881


namespace expansion_property_p_value_l2218_221844

/-- The value of p in the expansion of (x+y)^10 -/
def p : ℚ :=
  8/11

/-- The value of q in the expansion of (x+y)^10 -/
def q : ℚ :=
  3/11

/-- The third term in the expansion of (x+y)^10 -/
def third_term (x y : ℚ) : ℚ :=
  45 * x^8 * y^2

/-- The fourth term in the expansion of (x+y)^10 -/
def fourth_term (x y : ℚ) : ℚ :=
  120 * x^7 * y^3

theorem expansion_property : 
  p + q = 1 ∧ third_term p q = fourth_term p q :=
sorry

theorem p_value : p = 8/11 :=
sorry

end expansion_property_p_value_l2218_221844


namespace candy_distribution_l2218_221865

theorem candy_distribution (total_candy : ℕ) (num_friends : ℕ) : 
  total_candy = 30 → num_friends = 4 → 
  ∃ (removed : ℕ) (equal_share : ℕ), 
    removed ≤ 2 ∧ 
    (total_candy - removed) % num_friends = 0 ∧ 
    (total_candy - removed) / num_friends = equal_share ∧
    ∀ (r : ℕ), r < removed → (total_candy - r) % num_friends ≠ 0 :=
by sorry

end candy_distribution_l2218_221865


namespace equation_solutions_l2218_221871

def equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ -2 ∧ -x^2 = (5*x - 2)/(x - 2) - (x + 4)/(x + 2)

theorem equation_solutions :
  {x : ℝ | equation x} = {3, -1, -1 + Real.sqrt 5, -1 - Real.sqrt 5} := by
  sorry

end equation_solutions_l2218_221871


namespace greatest_integer_b_for_quadratic_range_l2218_221883

theorem greatest_integer_b_for_quadratic_range : 
  ∃ (b : ℤ), b = 9 ∧ 
  (∀ x : ℝ, x^2 + b*x + 20 ≠ -4) ∧
  (∀ c : ℤ, c > b → ∃ x : ℝ, x^2 + c*x + 20 = -4) := by
  sorry

end greatest_integer_b_for_quadratic_range_l2218_221883


namespace total_mangoes_l2218_221821

/-- The number of mangoes each person has -/
structure MangoDistribution where
  alexis : ℝ
  dilan : ℝ
  ashley : ℝ
  ben : ℝ

/-- The conditions of the mango distribution problem -/
def mango_conditions (m : MangoDistribution) : Prop :=
  m.alexis = 4 * (m.dilan + m.ashley) ∧
  m.ashley = 2 * m.dilan ∧
  m.alexis = 60 ∧
  m.ben = (m.ashley + m.dilan) / 2

/-- The theorem stating the total number of mangoes -/
theorem total_mangoes (m : MangoDistribution) 
  (h : mango_conditions m) : 
  m.alexis + m.dilan + m.ashley + m.ben = 82.5 :=
by sorry

end total_mangoes_l2218_221821


namespace minimum_width_for_garden_l2218_221842

theorem minimum_width_for_garden (w : ℝ) : w > 0 → w * (w + 10) ≥ 150 → 
  ∀ x > 0, x * (x + 10) ≥ 150 → 2 * (w + w + 10) ≤ 2 * (x + x + 10) → w = 10 := by
  sorry

end minimum_width_for_garden_l2218_221842


namespace concentric_circles_area_ratio_l2218_221817

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 1  -- diameter of smaller circle
  let d₂ : ℝ := 3  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  let area_small : ℝ := π * r₁^2  -- area of smaller circle
  let area_large : ℝ := π * r₂^2  -- area of larger circle
  let area_between : ℝ := area_large - area_small  -- area between circles
  (area_between / area_small) = 8 :=
by
  sorry

end concentric_circles_area_ratio_l2218_221817


namespace log_function_passes_through_point_l2218_221804

-- Define the logarithm function for any base a > 0 and a ≠ 1
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = log_a(x-1) + 2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x - 1) + 2

-- Theorem statement
theorem log_function_passes_through_point (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  f a 2 = 2 := by
  sorry

end log_function_passes_through_point_l2218_221804


namespace diamond_calculation_l2218_221863

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation : 
  let x := diamond (diamond 2 3) 4
  let y := diamond 2 (diamond 3 4)
  x - y = -29 / 132 := by sorry

end diamond_calculation_l2218_221863


namespace books_sold_l2218_221859

theorem books_sold (initial : ℕ) (added : ℕ) (final : ℕ) : 
  initial = 41 → added = 2 → final = 10 → initial - (initial - final + added) = 33 :=
by sorry

end books_sold_l2218_221859


namespace volleyball_ticket_sales_l2218_221815

theorem volleyball_ticket_sales (total_tickets : ℕ) (tickets_left : ℕ) : 
  total_tickets = 100 →
  tickets_left = 40 →
  ∃ (jude_tickets : ℕ),
    (jude_tickets : ℚ) + 2 * (jude_tickets : ℚ) + ((1/2 : ℚ) * (jude_tickets : ℚ) + 4) = (total_tickets - tickets_left : ℚ) ∧
    jude_tickets = 16 := by
  sorry

end volleyball_ticket_sales_l2218_221815


namespace tangent_slope_at_one_l2218_221808

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 1 :=
sorry

end tangent_slope_at_one_l2218_221808


namespace integral_over_pyramidal_region_l2218_221801

/-- The pyramidal region V -/
def V : Set (Fin 3 → ℝ) :=
  {v | ∀ i, v i ≥ 0 ∧ v 0 + v 1 + v 2 ≤ 1}

/-- The integrand function -/
def f (v : Fin 3 → ℝ) : ℝ :=
  v 0 * v 1^9 * v 2^8 * (1 - v 0 - v 1 - v 2)^4

/-- The theorem statement -/
theorem integral_over_pyramidal_region :
  ∫ v in V, f v = (Nat.factorial 9 * Nat.factorial 8 * Nat.factorial 4) / Nat.factorial 25 := by
  sorry

end integral_over_pyramidal_region_l2218_221801


namespace percentage_error_calculation_l2218_221819

theorem percentage_error_calculation : 
  let correct_multiplier : ℚ := 5/3
  let incorrect_multiplier : ℚ := 3/5
  let percentage_error := ((correct_multiplier - incorrect_multiplier) / correct_multiplier) * 100
  percentage_error = 64
  := by sorry

end percentage_error_calculation_l2218_221819


namespace collinear_points_k_value_l2218_221851

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The theorem states that if the points (5, 10), (-3, k), and (-11, 6) are collinear, then k = 8 -/
theorem collinear_points_k_value :
  collinear 5 10 (-3) k (-11) 6 → k = 8 := by
  sorry


end collinear_points_k_value_l2218_221851


namespace exists_zero_sequence_l2218_221834

-- Define the operations
def add_one (x : ℚ) : ℚ := x + 1
def neg_reciprocal (x : ℚ) : ℚ := -1 / x

-- Define a sequence of operations
inductive Operation
| AddOne
| NegReciprocal

def apply_operation (op : Operation) (x : ℚ) : ℚ :=
  match op with
  | Operation.AddOne => add_one x
  | Operation.NegReciprocal => neg_reciprocal x

-- Theorem statement
theorem exists_zero_sequence : ∃ (seq : List Operation), 
  let final_value := seq.foldl (λ acc op => apply_operation op acc) 0
  final_value = 0 ∧ seq.length > 0 :=
sorry

end exists_zero_sequence_l2218_221834


namespace exists_function_with_properties_l2218_221820

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the properties of the function
def HasFunctionalEquation (f : RealFunction) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂

def HasNegativeDerivative (f : RealFunction) : Prop :=
  ∀ x : ℝ, x > 0 → (deriv f x) < 0

-- State the theorem
theorem exists_function_with_properties :
  ∃ f : RealFunction,
    HasFunctionalEquation f ∧ HasNegativeDerivative f := by
  sorry

end exists_function_with_properties_l2218_221820


namespace triangle_height_l2218_221833

theorem triangle_height (base area height : ℝ) : 
  base = 4 ∧ area = 16 ∧ area = (base * height) / 2 → height = 8 := by
  sorry

end triangle_height_l2218_221833


namespace circle_not_proportional_line_directly_proportional_hyperbola_inversely_proportional_line_through_origin_directly_proportional_another_line_through_origin_directly_proportional_l2218_221885

/-- A relation between x and y is directly proportional if it can be expressed as y = kx for some constant k ≠ 0 -/
def DirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- A relation between x and y is inversely proportional if it can be expressed as xy = k for some constant k ≠ 0 -/
def InverselyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, x ≠ 0 → f x * x = k

/-- The main theorem stating that x^2 + y^2 = 16 is neither directly nor inversely proportional -/
theorem circle_not_proportional :
  ¬ (DirectlyProportional (fun x => Real.sqrt (16 - x^2)) ∨
     InverselyProportional (fun x => Real.sqrt (16 - x^2))) :=
sorry

/-- 2x + 3y = 6 describes y as directly proportional to x -/
theorem line_directly_proportional :
  DirectlyProportional (fun x => (6 - 2*x) / 3) ∨
  InverselyProportional (fun x => (6 - 2*x) / 3) :=
sorry

/-- xy = 5 describes y as inversely proportional to x -/
theorem hyperbola_inversely_proportional :
  DirectlyProportional (fun x => 5 / x) ∨
  InverselyProportional (fun x => 5 / x) :=
sorry

/-- x = 7y describes y as directly proportional to x -/
theorem line_through_origin_directly_proportional :
  DirectlyProportional (fun x => x / 7) ∨
  InverselyProportional (fun x => x / 7) :=
sorry

/-- x/y = 2 describes y as directly proportional to x -/
theorem another_line_through_origin_directly_proportional :
  DirectlyProportional (fun x => x / 2) ∨
  InverselyProportional (fun x => x / 2) :=
sorry

end circle_not_proportional_line_directly_proportional_hyperbola_inversely_proportional_line_through_origin_directly_proportional_another_line_through_origin_directly_proportional_l2218_221885


namespace largest_divisible_by_9_after_erasure_l2218_221830

def original_number : ℕ := 321321321321

def erase_digits (n : ℕ) (positions : List ℕ) : ℕ :=
  sorry

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem largest_divisible_by_9_after_erasure :
  ∃ (positions : List ℕ),
    let result := erase_digits original_number positions
    is_divisible_by_9 result ∧
    ∀ (other_positions : List ℕ),
      let other_result := erase_digits original_number other_positions
      is_divisible_by_9 other_result →
      other_result ≤ result ∧
      result = 32132132121 :=
sorry

end largest_divisible_by_9_after_erasure_l2218_221830


namespace right_triangle_integer_sides_l2218_221858

theorem right_triangle_integer_sides (a b c : ℕ) : 
  a^2 + b^2 = c^2 → -- Pythagorean theorem (right-angled triangle)
  Nat.gcd a (Nat.gcd b c) = 1 → -- GCD of sides is 1
  ∃ m n : ℕ, 
    (a = 2*m*n ∧ b = m^2 - n^2 ∧ c = m^2 + n^2) ∨ 
    (b = 2*m*n ∧ a = m^2 - n^2 ∧ c = m^2 + n^2) :=
by sorry

end right_triangle_integer_sides_l2218_221858


namespace decimal_13_equals_binary_1101_l2218_221852

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

def binary_to_decimal (l : List Bool) : ℕ :=
  l.enum.foldl (λ sum (i, b) => sum + if b then 2^i else 0) 0

def decimal_13 : ℕ := 13

def binary_1101 : List Bool := [true, false, true, true]

theorem decimal_13_equals_binary_1101 : 
  binary_to_decimal binary_1101 = decimal_13 :=
sorry

end decimal_13_equals_binary_1101_l2218_221852


namespace matching_socks_probability_l2218_221813

/-- The number of pairs of socks -/
def num_pairs : ℕ := 5

/-- The total number of socks -/
def total_socks : ℕ := 2 * num_pairs

/-- The number of socks selected each day -/
def socks_per_day : ℕ := 2

/-- The probability of selecting matching socks for the first time on Wednesday -/
def prob_match_wednesday : ℚ :=
  26 / 315

theorem matching_socks_probability :
  let monday_selections := Nat.choose total_socks socks_per_day
  let tuesday_selections := Nat.choose (total_socks - socks_per_day) socks_per_day
  let wednesday_selections := Nat.choose (total_socks - 2 * socks_per_day) socks_per_day
  prob_match_wednesday =
    (monday_selections - num_pairs) / monday_selections *
    ((1 / tuesday_selections * 1 / 5) +
     (12 / tuesday_selections * 2 / 15) +
     (12 / tuesday_selections * 1 / 15)) :=
by sorry

#eval prob_match_wednesday

end matching_socks_probability_l2218_221813


namespace ship_supplies_problem_l2218_221809

theorem ship_supplies_problem (initial_supply : ℝ) 
  (remaining_supply : ℝ) (h1 : initial_supply = 400) 
  (h2 : remaining_supply = 96) : 
  ∃ x : ℝ, x = 2/5 ∧ 
    remaining_supply = (2/5) * (1 - x) * initial_supply :=
by sorry

end ship_supplies_problem_l2218_221809


namespace anne_drawings_per_marker_l2218_221818

/-- Given:
  * Anne has 12 markers
  * She has already made 8 drawings
  * She can make 10 more drawings before running out of markers
  Prove that Anne can make 1.5 drawings with one marker -/
theorem anne_drawings_per_marker (markers : ℕ) (made_drawings : ℕ) (remaining_drawings : ℕ) 
  (h1 : markers = 12)
  (h2 : made_drawings = 8)
  (h3 : remaining_drawings = 10) :
  (made_drawings + remaining_drawings : ℚ) / markers = 1.5 := by
  sorry

end anne_drawings_per_marker_l2218_221818


namespace min_subset_size_l2218_221869

def is_valid_subset (s : Finset ℕ) : Prop :=
  s ⊆ Finset.range 11 ∧
  ∀ n : ℕ, n ∈ Finset.range 21 →
    (n ∈ s ∨ ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a + b = n)

theorem min_subset_size :
  ∃ (s : Finset ℕ), is_valid_subset s ∧ s.card = 6 ∧
  ∀ (t : Finset ℕ), is_valid_subset t → t.card ≥ 6 :=
sorry

end min_subset_size_l2218_221869


namespace min_value_z_l2218_221811

theorem min_value_z (x y : ℝ) : 
  x^2 + 2*y^2 + y^3 + 6*x - 4*y + 30 ≥ 20 ∧ 
  ∃ x y : ℝ, x^2 + 2*y^2 + y^3 + 6*x - 4*y + 30 = 20 :=
by sorry

end min_value_z_l2218_221811


namespace series_sum_l2218_221854

theorem series_sum : 1000 + 20 + 1000 + 30 + 1000 + 40 + 1000 + 10 = 4100 := by
  sorry

end series_sum_l2218_221854


namespace bird_migration_distance_l2218_221892

/-- Calculates the total distance traveled by migrating birds over two seasons -/
theorem bird_migration_distance (num_birds : ℕ) (dist_jim_disney : ℝ) (dist_disney_london : ℝ) :
  num_birds = 20 →
  dist_jim_disney = 50 →
  dist_disney_london = 60 →
  num_birds * (dist_jim_disney + dist_disney_london) = 2200 := by
  sorry

end bird_migration_distance_l2218_221892


namespace min_binomial_ratio_five_seven_l2218_221877

theorem min_binomial_ratio_five_seven (n : ℕ) : n > 0 → (
  (∃ r : ℕ, r < n ∧ (n.choose r : ℚ) / (n.choose (r + 1)) = 5 / 7) ↔ n ≥ 11
) := by sorry

end min_binomial_ratio_five_seven_l2218_221877


namespace sum_of_bases_l2218_221814

-- Define the fractions F₁ and F₂
def F₁ (R : ℕ) : ℚ := (4 * R + 5) / (R^2 - 1)
def F₂ (R : ℕ) : ℚ := (5 * R + 4) / (R^2 - 1)

-- Define the conditions
def conditions (R₁ R₂ : ℕ) : Prop :=
  F₁ R₁ = F₁ R₂ ∧ F₂ R₁ = F₂ R₂ ∧
  R₁ ≥ 2 ∧ R₂ ≥ 2 ∧ -- Ensure bases are valid
  (∃ k : ℕ, F₁ R₁ = k / 11) ∧ -- Represents the repeating decimal 0.454545...
  (∃ k : ℕ, F₂ R₁ = k / 11) ∧ -- Represents the repeating decimal 0.545454...
  (∃ k : ℕ, F₁ R₂ = k / 11) ∧ -- Represents the repeating decimal 0.363636...
  (∃ k : ℕ, F₂ R₂ = k / 11)   -- Represents the repeating decimal 0.636363...

-- State the theorem
theorem sum_of_bases (R₁ R₂ : ℕ) : 
  conditions R₁ R₂ → R₁ + R₂ = 19 :=
by sorry

end sum_of_bases_l2218_221814


namespace min_abs_z_l2218_221886

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 5*I) + Complex.abs (z - 2) = 10) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 10 / Real.sqrt 29 := by
  sorry

end min_abs_z_l2218_221886


namespace chord_length_on_unit_circle_l2218_221848

/-- The length of the chord intercepted by the line x-y=0 on the circle x^2 + y^2 = 1 is equal to 2 -/
theorem chord_length_on_unit_circle : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | x = y}
  let chord := circle ∩ line
  ∃ (a b : ℝ × ℝ), a ∈ chord ∧ b ∈ chord ∧ a ≠ b ∧ Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 2 :=
sorry

end chord_length_on_unit_circle_l2218_221848


namespace fraction_of_As_l2218_221876

theorem fraction_of_As (total_students : ℕ) (fraction_Bs fraction_Cs : ℚ) (num_Ds : ℕ) :
  total_students = 100 →
  fraction_Bs = 1/4 →
  fraction_Cs = 1/2 →
  num_Ds = 5 →
  (total_students - (fraction_Bs * total_students + fraction_Cs * total_students + num_Ds)) / total_students = 1/5 := by
  sorry

end fraction_of_As_l2218_221876


namespace negation_of_forall_positive_negation_of_quadratic_inequality_l2218_221867

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ ∃ x > 0, ¬ P x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∀ x > 0, x^2 + 3*x - 2 > 0) ↔ (∃ x > 0, x^2 + 3*x - 2 ≤ 0) := by sorry

end negation_of_forall_positive_negation_of_quadratic_inequality_l2218_221867


namespace tammy_mountain_climb_l2218_221839

/-- Tammy's mountain climbing problem -/
theorem tammy_mountain_climb 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (speed_diff : ℝ) 
  (time_diff : ℝ) 
  (h_total_time : total_time = 14) 
  (h_total_distance : total_distance = 52) 
  (h_speed_diff : speed_diff = 0.5) 
  (h_time_diff : time_diff = 2) :
  ∃ (v : ℝ), 
    v > 0 ∧ 
    v + speed_diff > 0 ∧ 
    (∃ (t : ℝ), 
      t > 0 ∧ 
      t - time_diff > 0 ∧ 
      t + (t - time_diff) = total_time ∧ 
      v * t + (v + speed_diff) * (t - time_diff) = total_distance) ∧ 
    v + speed_diff = 4 := by
  sorry

end tammy_mountain_climb_l2218_221839


namespace not_monotone_condition_l2218_221841

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the property of being not monotone on an interval
def not_monotone_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a < x ∧ x < y ∧ y < b ∧ (f x < f y ∧ ∃ z, x < z ∧ z < y ∧ f z < f x) ∨
                                 (f x > f y ∧ ∃ z, x < z ∧ z < y ∧ f z > f x)

-- State the theorem
theorem not_monotone_condition (k : ℝ) :
  not_monotone_on f k (k + 2) ↔ (-4 < k ∧ k < -2) ∨ (0 < k ∧ k < 2) :=
sorry

end not_monotone_condition_l2218_221841


namespace no_solution_exists_l2218_221860

theorem no_solution_exists : ¬∃ (a b c : ℕ+), 
  (a * b + b * c = a * c) ∧ (a * b * c = Nat.factorial 10) := by
  sorry

end no_solution_exists_l2218_221860


namespace perpendicular_vectors_l2218_221836

def a : ℝ × ℝ := (1, -2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)
def c : ℝ × ℝ := (1, 2)

theorem perpendicular_vectors (x : ℝ) : 
  (a.1 + (b x).1) * c.1 + (a.2 + (b x).2) * c.2 = 0 → x = 1 := by
  sorry

end perpendicular_vectors_l2218_221836


namespace tree_planting_ratio_l2218_221894

/-- 
Given a forest with an initial number of trees, and a forester who plants trees over two days,
this theorem proves that the ratio of trees planted on the second day to the first day is 1/3,
given specific conditions about the planting process.
-/
theorem tree_planting_ratio 
  (initial_trees : ℕ) 
  (trees_after_monday : ℕ) 
  (total_planted : ℕ) 
  (h1 : initial_trees = 30)
  (h2 : trees_after_monday = initial_trees * 3)
  (h3 : total_planted = 80) :
  (total_planted - (trees_after_monday - initial_trees)) / (trees_after_monday - initial_trees) = 1 / 3 := by
  sorry

#check tree_planting_ratio

end tree_planting_ratio_l2218_221894


namespace composite_expression_l2218_221899

/-- A positive integer is composite if it can be expressed as a product of two integers
    greater than 1. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- Every composite positive integer can be expressed as xy+xz+yz+1,
    where x, y, and z are positive integers. -/
theorem composite_expression (n : ℕ) (h : IsComposite n) :
    ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ n = x * y + x * z + y * z + 1 := by
  sorry

end composite_expression_l2218_221899


namespace exists_negative_value_implies_a_greater_than_nine_halves_l2218_221873

theorem exists_negative_value_implies_a_greater_than_nine_halves
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = x^3 - a*x^2 + 10)
  (a : ℝ)
  (h_exists : ∃ x ∈ Set.Icc 1 2, f x < 0) :
  a > 9/2 := by
sorry

end exists_negative_value_implies_a_greater_than_nine_halves_l2218_221873


namespace divisibility_and_finiteness_l2218_221872

theorem divisibility_and_finiteness :
  (∀ x : ℕ+, ∃ y : ℕ+, (x + y + 1) ∣ (x^3 + y^3 + 1)) ∧
  (∀ x : ℕ+, Set.Finite {y : ℕ+ | (x + y + 1) ∣ (x^3 + y^3 + 1)}) := by
  sorry

end divisibility_and_finiteness_l2218_221872


namespace horner_rule_v4_l2218_221807

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  v3 * x + 240

theorem horner_rule_v4 :
  horner_v4 2 = 80 :=
by sorry

#eval horner_v4 2
#eval horner_polynomial 2

end horner_rule_v4_l2218_221807


namespace complex_modulus_problem_l2218_221825

theorem complex_modulus_problem (z : ℂ) (h : (2 - Complex.I) * z = 4 + 3 * Complex.I) :
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l2218_221825


namespace invisible_square_exists_l2218_221890

/-- A lattice point is invisible if the segment from the origin to that point contains another lattice point. -/
def invisible (x y : ℤ) : Prop :=
  ∃ k : ℤ, 1 < k ∧ k < max x.natAbs y.natAbs ∧ (k ∣ x) ∧ (k ∣ y)

/-- For any positive integer L, there exists a square with side length L where all lattice points are invisible. -/
theorem invisible_square_exists (L : ℕ) (hL : 0 < L) :
  ∃ x y : ℤ, ∀ i j : ℕ, i ≤ L → j ≤ L → invisible (x + i) (y + j) := by
  sorry

end invisible_square_exists_l2218_221890


namespace xiaotong_pe_score_l2218_221835

/-- Calculates the physical education score based on extracurricular activities and final exam scores -/
def physical_education_score (extracurricular_score : ℝ) (final_exam_score : ℝ) : ℝ :=
  0.3 * extracurricular_score + 0.7 * final_exam_score

/-- Xiaotong's physical education score theorem -/
theorem xiaotong_pe_score :
  let max_score : ℝ := 100
  let extracurricular_weight : ℝ := 0.3
  let final_exam_weight : ℝ := 0.7
  let xiaotong_extracurricular_score : ℝ := 90
  let xiaotong_final_exam_score : ℝ := 80
  physical_education_score xiaotong_extracurricular_score xiaotong_final_exam_score = 83 :=
by
  sorry

#eval physical_education_score 90 80

end xiaotong_pe_score_l2218_221835


namespace distance_relation_l2218_221857

/-- Given four points on a directed line satisfying a certain condition, 
    prove a relationship between their distances. -/
theorem distance_relation (A B C D : ℝ) 
    (h : (C - A) / (B - C) + (D - A) / (B - D) = 0) : 
    1 / (C - A) + 1 / (D - A) = 2 / (B - A) := by
  sorry

end distance_relation_l2218_221857


namespace min_value_fraction_l2218_221843

theorem min_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∃ (m : ℝ), m = (x + y) / x ∧ ∀ (z w : ℝ), (-6 ≤ z ∧ z ≤ -3) → (3 ≤ w ∧ w ≤ 6) → m ≤ (z + w) / z :=
by
  sorry

end min_value_fraction_l2218_221843


namespace cylinder_height_problem_l2218_221879

/-- The height of cylinder B given the conditions of the problem -/
def height_cylinder_B : ℝ := 75

/-- The base radius of cylinder A in cm -/
def radius_A : ℝ := 10

/-- The height of cylinder A in cm -/
def height_A : ℝ := 8

/-- The base radius of cylinder B in cm -/
def radius_B : ℝ := 4

/-- The volume ratio of cylinder B to cylinder A -/
def volume_ratio : ℝ := 1.5

theorem cylinder_height_problem :
  volume_ratio * (Real.pi * radius_A^2 * height_A) = Real.pi * radius_B^2 * height_cylinder_B :=
by sorry

end cylinder_height_problem_l2218_221879


namespace ellipse_ratio_l2218_221829

theorem ellipse_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a / b = b / c) : a^2 / b^2 = 2 / (-1 + Real.sqrt 5) := by
  sorry

end ellipse_ratio_l2218_221829


namespace arithmetic_sequence_common_difference_l2218_221882

theorem arithmetic_sequence_common_difference 
  (a : Fin 4 → ℚ) 
  (h_arithmetic : ∀ i j k, i < j → j < k → a j - a i = a k - a j) 
  (h_first : a 0 = 1) 
  (h_last : a 3 = 2) : 
  ∀ i j, i < j → a j - a i = 1/3 := by
sorry

end arithmetic_sequence_common_difference_l2218_221882


namespace quadratic_transformation_l2218_221847

theorem quadratic_transformation (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 3 * (x - 5)^2 + 9) →
  ∃ m k, ∀ x, 2 * (a * x^2 + b * x + c) = m * (x - 5)^2 + k :=
by sorry

end quadratic_transformation_l2218_221847


namespace dried_grapes_weight_l2218_221864

/-- Calculates the weight of dried grapes from fresh grapes -/
theorem dried_grapes_weight
  (fresh_weight : ℝ)
  (fresh_water_content : ℝ)
  (dried_water_content : ℝ)
  (h1 : fresh_weight = 40)
  (h2 : fresh_water_content = 0.9)
  (h3 : dried_water_content = 0.2) :
  (fresh_weight * (1 - fresh_water_content)) / (1 - dried_water_content) = 5 := by
  sorry

end dried_grapes_weight_l2218_221864


namespace ferris_wheel_capacity_l2218_221870

theorem ferris_wheel_capacity (num_seats : ℕ) (people_per_seat : ℕ) 
  (h1 : num_seats = 14) (h2 : people_per_seat = 6) : 
  num_seats * people_per_seat = 84 := by
  sorry

end ferris_wheel_capacity_l2218_221870


namespace half_square_area_l2218_221862

theorem half_square_area (square_area : Real) (h1 : square_area = 100) :
  square_area / 2 = 50 := by
  sorry

end half_square_area_l2218_221862


namespace cos_alpha_value_l2218_221855

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.sin (α - Real.pi / 3) = 1 / 3) :
  Real.cos α = (2 * Real.sqrt 2 - Real.sqrt 3) / 6 := by
  sorry

end cos_alpha_value_l2218_221855


namespace cubic_root_function_l2218_221861

/-- Given a function y = kx^(1/3) where y = 3√2 when x = 64, prove that y = 3 when x = 8 -/
theorem cubic_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * 64^(1/3) ∧ y = 3 * Real.sqrt 2) →
  k * 8^(1/3) = 3 := by
  sorry

end cubic_root_function_l2218_221861


namespace yard_length_24_trees_18m_spacing_l2218_221853

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_spacing : ℝ) : ℝ :=
  (num_trees - 1) * tree_spacing

/-- Theorem: The length of a yard with 24 trees planted at equal distances,
    with one tree at each end and 18 meters between consecutive trees, is 414 meters. -/
theorem yard_length_24_trees_18m_spacing :
  yard_length 24 18 = 414 := by
  sorry

end yard_length_24_trees_18m_spacing_l2218_221853


namespace divisibility_problem_l2218_221822

theorem divisibility_problem (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 30)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 60)
  (h4 : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) :
  15 ∣ p := by
  sorry

end divisibility_problem_l2218_221822


namespace town_distance_approx_l2218_221810

/-- Represents the map scale as a fraction of inches per mile -/
def map_scale : ℚ := 7 / (15 * 19)

/-- Represents the distance between two points on the map in inches -/
def map_distance : ℚ := 37 / 8

/-- Calculates the actual distance in miles given the map scale and map distance -/
def actual_distance (scale : ℚ) (distance : ℚ) : ℚ := distance / scale

/-- Theorem stating that the actual distance between the towns is approximately 41.0083 miles -/
theorem town_distance_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/10000 ∧ 
  |actual_distance map_scale map_distance - 41.0083| < ε :=
sorry

end town_distance_approx_l2218_221810


namespace odd_polyhedron_sum_not_nine_l2218_221823

/-- Represents a convex polyhedron with odd-sided faces and odd-valence vertices -/
structure OddPolyhedron where
  -- Number of edges
  e : ℕ
  -- Number of faces with i sides (i is odd)
  ℓ : ℕ → ℕ
  -- Number of vertices where i edges meet (i is odd)
  c : ℕ → ℕ
  -- Each face has an odd number of sides
  face_odd : ∀ i, ℓ i > 0 → Odd i
  -- Each vertex has an odd number of edges meeting at it
  vertex_odd : ∀ i, c i > 0 → Odd i
  -- Edge-face relation
  edge_face : 2 * e = ∑' i, i * ℓ i
  -- Edge-vertex relation
  edge_vertex : 2 * e = ∑' i, i * c i
  -- Euler's formula
  euler : e + 2 = (∑' i, ℓ i) + (∑' i, c i)

/-- The sum of triangular faces and vertices where three edges meet cannot be 9 -/
theorem odd_polyhedron_sum_not_nine (P : OddPolyhedron) : ¬(P.ℓ 3 + P.c 3 = 9) := by
  sorry

end odd_polyhedron_sum_not_nine_l2218_221823


namespace valid_assignment_y_equals_x_plus_1_l2218_221849

/-- Represents a variable name in a programming language --/
def Variable : Type := String

/-- Represents an expression in a programming language --/
inductive Expression
| Var : Variable → Expression
| Num : Int → Expression
| Add : Expression → Expression → Expression

/-- Represents an assignment statement in a programming language --/
structure Assignment :=
  (lhs : Variable)
  (rhs : Expression)

/-- Checks if an assignment statement is valid --/
def is_valid_assignment (a : Assignment) : Prop :=
  ∃ (x : Variable), a.rhs = Expression.Add (Expression.Var x) (Expression.Num 1)

/-- The statement "y = x + 1" is a valid assignment --/
theorem valid_assignment_y_equals_x_plus_1 :
  is_valid_assignment { lhs := "y", rhs := Expression.Add (Expression.Var "x") (Expression.Num 1) } :=
by sorry

end valid_assignment_y_equals_x_plus_1_l2218_221849


namespace largest_even_integer_l2218_221897

theorem largest_even_integer (n : ℕ) : 
  (2 * (List.range 20).sum) = (4 * n - 12) → n = 108 :=
by
  sorry

end largest_even_integer_l2218_221897


namespace regression_unit_increase_l2218_221846

/-- Represents a simple linear regression model -/
structure LinearRegression where
  intercept : ℝ
  slope : ℝ

/-- The predicted value for a given x in a linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.intercept + model.slope * x

/-- Theorem: In the given linear regression model, when x increases by 1, y increases by 3 -/
theorem regression_unit_increase (model : LinearRegression) (x : ℝ) 
    (h : model = { intercept := 2, slope := 3 }) :
    predict model (x + 1) - predict model x = 3 := by
  sorry

end regression_unit_increase_l2218_221846


namespace opposite_of_negative_three_l2218_221832

theorem opposite_of_negative_three : -((-3 : ℤ)) = 3 := by sorry

end opposite_of_negative_three_l2218_221832


namespace minimize_segment_expression_l2218_221884

/-- Given a line segment AB of length a, the point C that minimizes AC^2 + 3CB^2 is at 3a/4 from A -/
theorem minimize_segment_expression (a : ℝ) (h : a > 0) :
  ∃ c : ℝ, c = 3*a/4 ∧ 
    ∀ x : ℝ, 0 ≤ x ∧ x ≤ a → 
      x^2 + 3*(a-x)^2 ≥ c^2 + 3*(a-c)^2 :=
by sorry


end minimize_segment_expression_l2218_221884


namespace max_rectangle_area_l2218_221806

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def isComposite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def rectangle_area (length width : ℕ) : ℕ := length * width

theorem max_rectangle_area (length width : ℕ) :
  length + width = 25 →
  isPrime length →
  isComposite width →
  ∀ l w : ℕ, l + w = 25 → isPrime l → isComposite w → 
    rectangle_area length width ≥ rectangle_area l w →
  rectangle_area length width = 156 :=
sorry

end max_rectangle_area_l2218_221806


namespace missing_number_is_eight_l2218_221893

/-- Given the equation |9 - x(3 - 12)| - |5 - 11| = 75, prove that x = 8 is the solution. -/
theorem missing_number_is_eight : ∃ x : ℝ, 
  (|9 - x * (3 - 12)| - |5 - 11| = 75) ∧ (x = 8) := by
  sorry

end missing_number_is_eight_l2218_221893


namespace min_value_of_a_l2218_221826

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | x > a}

theorem min_value_of_a (a : ℝ) (h : A ∩ B a = ∅) : a ≥ 1 := by
  sorry

end min_value_of_a_l2218_221826


namespace sphere_sum_l2218_221874

theorem sphere_sum (x y z : ℝ) : 
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0 → x + y + z = 2 := by
  sorry

end sphere_sum_l2218_221874


namespace books_per_shelf_l2218_221828

theorem books_per_shelf 
  (total_shelves : ℕ) 
  (total_books : ℕ) 
  (h1 : total_shelves = 150) 
  (h2 : total_books = 2250) : 
  total_books / total_shelves = 15 := by
  sorry

end books_per_shelf_l2218_221828


namespace f_properties_l2218_221856

/-- The function f(x) defined as (2^x - a) / (2^x + a) where a > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x - a) / (2^x + a)

/-- Theorem stating properties of the function f -/
theorem f_properties (a : ℝ) (h : a > 0) 
  (h_odd : ∀ x, f a (-x) = -(f a x)) :
  (a = 1) ∧ 
  (∀ x y, x < y → f a x < f a y) ∧
  (∀ x, x ≤ 1 → f a x ≤ 1/3) ∧
  (f a 1 = 1/3) := by
sorry

end f_properties_l2218_221856


namespace expected_socks_removed_theorem_l2218_221838

/-- The expected number of socks removed to get both favorite socks -/
def expected_socks_removed (n : ℕ) : ℚ :=
  2 * (n + 1) / 3

/-- Theorem stating the expected number of socks removed to get both favorite socks -/
theorem expected_socks_removed_theorem (n : ℕ) (h : n ≥ 2) :
  expected_socks_removed n = 2 * (n + 1) / 3 := by
  sorry

#check expected_socks_removed_theorem

end expected_socks_removed_theorem_l2218_221838


namespace unfoldable_cone_ratio_l2218_221816

/-- A cone with lateral surface that forms a semicircle when unfolded -/
structure UnfoldableCone where
  /-- Radius of the base of the cone -/
  base_radius : ℝ
  /-- Length of the generatrix of the cone -/
  generatrix_length : ℝ
  /-- The lateral surface forms a semicircle when unfolded -/
  unfolded_is_semicircle : π * generatrix_length = 2 * π * base_radius

/-- 
If the lateral surface of a cone forms a semicircle when unfolded, 
then the ratio of the length of the cone's generatrix to the radius of its base is 2:1
-/
theorem unfoldable_cone_ratio (cone : UnfoldableCone) : 
  cone.generatrix_length / cone.base_radius = 2 := by
  sorry

end unfoldable_cone_ratio_l2218_221816


namespace triangle_theorem_l2218_221868

/-- Triangle ABC with sides a, b, c corresponding to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- The given condition a^2 + c^2 = b^2 - ac -/
def triangleCondition (t : Triangle) : Prop :=
  t.a^2 + t.c^2 = t.b^2 - t.a * t.c

/-- The angle bisector condition -/
def angleBisectorCondition (t : Triangle) (AD BD : ℝ) : Prop :=
  AD = 2 * Real.sqrt 3 ∧ BD = 1

theorem triangle_theorem (t : Triangle) 
  (h1 : triangleCondition t)
  (h2 : angleBisectorCondition t (2 * Real.sqrt 3) 1) :
  t.B = 2 * Real.pi / 3 ∧ Real.sin t.A = Real.sqrt 15 / 8 := by
  sorry

end triangle_theorem_l2218_221868


namespace num_correct_propositions_is_one_l2218_221875

/-- Represents a geometric proposition -/
inductive GeometricProposition
  | ThreePointsCircle
  | EqualArcEqualAngle
  | RightTrianglesSimilar
  | RhombusesSimilar

/-- Determines if a geometric proposition is correct -/
def is_correct (prop : GeometricProposition) : Bool :=
  match prop with
  | GeometricProposition.ThreePointsCircle => false
  | GeometricProposition.EqualArcEqualAngle => true
  | GeometricProposition.RightTrianglesSimilar => false
  | GeometricProposition.RhombusesSimilar => false

/-- The list of all propositions to be evaluated -/
def all_propositions : List GeometricProposition :=
  [GeometricProposition.ThreePointsCircle,
   GeometricProposition.EqualArcEqualAngle,
   GeometricProposition.RightTrianglesSimilar,
   GeometricProposition.RhombusesSimilar]

/-- Theorem stating that the number of correct propositions is 1 -/
theorem num_correct_propositions_is_one :
  (all_propositions.filter is_correct).length = 1 := by
  sorry

end num_correct_propositions_is_one_l2218_221875


namespace product_in_A_l2218_221880

-- Define the set A
def A : Set ℤ := {x | ∃ a b : ℤ, x = a^2 + b^2}

-- State the theorem
theorem product_in_A (x₁ x₂ : ℤ) (h₁ : x₁ ∈ A) (h₂ : x₂ ∈ A) : 
  x₁ * x₂ ∈ A := by
  sorry

end product_in_A_l2218_221880


namespace parallel_necessary_not_sufficient_l2218_221831

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b ∨ b = k • a

theorem parallel_necessary_not_sufficient :
  (∀ a b : V, a = b → parallel a b) ∧
  (∃ a b : V, parallel a b ∧ a ≠ b) := by sorry

end parallel_necessary_not_sufficient_l2218_221831


namespace B_coords_when_A_on_y_axis_a_value_when_AB_parallel_x_axis_l2218_221803

-- Define points A and B in the Cartesian coordinate system
def A (a : ℝ) : ℝ × ℝ := (a + 1, -3)
def B (a : ℝ) : ℝ × ℝ := (3, 2 * a + 1)

-- Theorem 1: When A lies on the y-axis, B has coordinates (3, -1)
theorem B_coords_when_A_on_y_axis (a : ℝ) :
  A a = (0, -3) → B a = (3, -1) := by sorry

-- Theorem 2: When AB is parallel to x-axis, a = -2
theorem a_value_when_AB_parallel_x_axis (a : ℝ) :
  (A a).2 = (B a).2 → a = -2 := by sorry

end B_coords_when_A_on_y_axis_a_value_when_AB_parallel_x_axis_l2218_221803


namespace shooting_probabilities_l2218_221837

/-- The probability of A hitting the target -/
def prob_A_hit : ℚ := 2/3

/-- The probability of B hitting the target -/
def prob_B_hit : ℚ := 3/4

/-- The probability that A shoots 3 times and misses at least once -/
def prob_A_miss_at_least_once : ℚ := 19/27

/-- The probability that A shoots twice and hits both times, while B shoots twice and hits exactly once -/
def prob_A_hit_twice_B_hit_once : ℚ := 1/6

theorem shooting_probabilities 
  (hA : prob_A_hit = 2/3)
  (hB : prob_B_hit = 3/4)
  (indep : ∀ (n : ℕ) (m : ℕ), (prob_A_hit ^ n) * ((1 - prob_A_hit) ^ (m - n)) = 
    (2/3 ^ n) * ((1/3) ^ (m - n))) :
  prob_A_miss_at_least_once = 19/27 ∧ 
  prob_A_hit_twice_B_hit_once = 1/6 := by
  sorry

end shooting_probabilities_l2218_221837


namespace log_expression_equality_l2218_221812

theorem log_expression_equality : Real.log 4 / Real.log 10 + 2 * Real.log 5 / Real.log 10 - (Real.sqrt 3 + 1) ^ 0 = 1 := by
  sorry

end log_expression_equality_l2218_221812


namespace woodworker_job_days_l2218_221888

/-- Represents the woodworker's job details -/
structure WoodworkerJob where
  normal_days : ℕ            -- Normal number of days to complete the job
  normal_parts : ℕ           -- Normal number of parts produced
  productivity_increase : ℕ  -- Increase in parts produced per day
  extra_parts : ℕ            -- Extra parts produced with increased productivity

/-- Calculates the number of days required to finish the job with increased productivity -/
def days_with_increased_productivity (job : WoodworkerJob) : ℕ :=
  let normal_rate := job.normal_parts / job.normal_days
  let new_rate := normal_rate + job.productivity_increase
  let total_parts := job.normal_parts + job.extra_parts
  total_parts / new_rate

/-- Theorem stating that for the given conditions, the job takes 22 days with increased productivity -/
theorem woodworker_job_days (job : WoodworkerJob)
  (h1 : job.normal_days = 24)
  (h2 : job.normal_parts = 360)
  (h3 : job.productivity_increase = 5)
  (h4 : job.extra_parts = 80) :
  days_with_increased_productivity job = 22 := by
  sorry

end woodworker_job_days_l2218_221888


namespace rect_to_spherical_l2218_221866

/-- Conversion from rectangular to spherical coordinates -/
theorem rect_to_spherical (x y z : ℝ) :
  x = 1 ∧ y = Real.sqrt 3 ∧ z = 2 →
  ∃ (ρ θ φ : ℝ),
    ρ > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    0 ≤ φ ∧ φ ≤ Real.pi ∧
    ρ = 3 ∧
    θ = Real.pi / 3 ∧
    φ = Real.arccos (2/3) ∧
    x = ρ * Real.sin φ * Real.cos θ ∧
    y = ρ * Real.sin φ * Real.sin θ ∧
    z = ρ * Real.cos φ :=
by sorry

end rect_to_spherical_l2218_221866


namespace expression_simplification_l2218_221896

theorem expression_simplification 
  (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  ∃ k : ℝ, ∀ x : ℝ, 
    (x + a)^4 / ((a - b) * (a - c) * (a - d)) +
    (x + b)^4 / ((b - a) * (b - c) * (b - d)) +
    (x + c)^4 / ((c - a) * (c - b) * (c - d)) +
    (x + d)^4 / ((d - a) * (d - b) * (d - c)) =
    k * (x + a) * (x + b) * (x + c) * (x + d) := by
  sorry

end expression_simplification_l2218_221896
