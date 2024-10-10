import Mathlib

namespace ashton_sheets_l4107_410709

theorem ashton_sheets (jimmy_sheets : ℕ) (tommy_sheets : ℕ) (ashton_sheets : ℕ) : 
  jimmy_sheets = 32 →
  tommy_sheets = jimmy_sheets + 10 →
  jimmy_sheets + ashton_sheets = tommy_sheets + 30 →
  ashton_sheets = 40 :=
by
  sorry

end ashton_sheets_l4107_410709


namespace change_calculation_l4107_410710

/-- Given the cost of milk and water, and the amount paid, calculate the change received. -/
theorem change_calculation (milk_cost water_cost paid : ℕ) 
  (h_milk : milk_cost = 350)
  (h_water : water_cost = 500)
  (h_paid : paid = 1000) :
  paid - (milk_cost + water_cost) = 150 := by
  sorry

end change_calculation_l4107_410710


namespace arithmetic_sequence_solution_l4107_410793

theorem arithmetic_sequence_solution (x : ℝ) (h1 : x ≠ 0) :
  (x - Int.floor x) + (Int.floor x + 1) + x = 3 * ((Int.floor x + 1)) →
  x = -2 ∨ x = -1/2 := by
  sorry

end arithmetic_sequence_solution_l4107_410793


namespace increase_by_percentage_l4107_410769

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 90 ∧ percentage = 50 ∧ final = initial * (1 + percentage / 100) →
  final = 135 := by
  sorry

end increase_by_percentage_l4107_410769


namespace intersection_A_B_l4107_410799

def A : Set ℝ := {x | 1 / x < 1}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {-1, 2} := by
  sorry

end intersection_A_B_l4107_410799


namespace encyclopedia_cost_l4107_410797

/-- Proves that the cost of encyclopedias is approximately $1002.86 given the specified conditions --/
theorem encyclopedia_cost (down_payment : ℝ) (monthly_payment : ℝ) (num_monthly_payments : ℕ)
  (final_payment : ℝ) (interest_rate : ℝ) :
  down_payment = 300 →
  monthly_payment = 57 →
  num_monthly_payments = 9 →
  final_payment = 21 →
  interest_rate = 18.666666666666668 / 100 →
  ∃ (cost : ℝ), abs (cost - 1002.86) < 0.01 :=
by
  sorry


end encyclopedia_cost_l4107_410797


namespace total_turnips_l4107_410712

theorem total_turnips (melanie_turnips benny_turnips : ℕ) 
  (h1 : melanie_turnips = 139) 
  (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := by
  sorry

end total_turnips_l4107_410712


namespace sum_in_range_l4107_410774

theorem sum_in_range : ∃ (s : ℚ), 
  s = (1 + 3/8) + (4 + 1/3) + (6 + 2/21) ∧ 11 < s ∧ s < 12 := by
  sorry

end sum_in_range_l4107_410774


namespace range_of_a_l4107_410715

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x : ℝ | 2*a - 1 < x ∧ x < 4*a}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 2}

-- State the theorem
theorem range_of_a (h : N ⊆ M a) : a ∈ Set.Icc (1/2 : ℝ) 2 := by
  sorry

-- Note: Set.Icc represents a closed interval [1/2, 2]

end range_of_a_l4107_410715


namespace diborane_combustion_heat_correct_l4107_410726

/-- Represents the heat of vaporization of water in kJ/mol -/
def water_vaporization_heat : ℝ := 44

/-- Represents the amount of diborane in moles -/
def diborane_amount : ℝ := 0.3

/-- Represents the heat released during combustion in kJ -/
def heat_released : ℝ := 609.9

/-- Represents the heat of combustion of diborane in kJ/mol -/
def diborane_combustion_heat : ℝ := -2165

/-- Theorem stating that the given heat of combustion of diborane is correct -/
theorem diborane_combustion_heat_correct : 
  diborane_combustion_heat = -heat_released / diborane_amount - 3 * water_vaporization_heat :=
sorry

end diborane_combustion_heat_correct_l4107_410726


namespace original_profit_percentage_l4107_410745

theorem original_profit_percentage (cost_price selling_price : ℝ) :
  cost_price > 0 →
  selling_price > cost_price →
  (2 * selling_price - cost_price) / cost_price = 3.2 →
  (selling_price - cost_price) / cost_price = 1.1 := by
  sorry

end original_profit_percentage_l4107_410745


namespace brazil_nut_price_is_five_l4107_410767

/-- Represents the price of Brazil nuts per pound -/
def brazil_nut_price : ℝ := 5

/-- Represents the price of cashews per pound -/
def cashew_price : ℝ := 6.75

/-- Represents the total weight of the mixture in pounds -/
def total_mixture_weight : ℝ := 50

/-- Represents the selling price of the mixture per pound -/
def mixture_selling_price : ℝ := 5.70

/-- Represents the weight of cashews used in the mixture in pounds -/
def cashew_weight : ℝ := 20

/-- Theorem stating that the price of Brazil nuts is $5 per pound given the conditions -/
theorem brazil_nut_price_is_five :
  brazil_nut_price = 5 ∧
  cashew_price = 6.75 ∧
  total_mixture_weight = 50 ∧
  mixture_selling_price = 5.70 ∧
  cashew_weight = 20 →
  brazil_nut_price = 5 := by
  sorry

end brazil_nut_price_is_five_l4107_410767


namespace smallest_distance_complex_circles_l4107_410792

theorem smallest_distance_complex_circles (z w : ℂ) 
  (hz : Complex.abs (z + 1 + 3*I) = 1)
  (hw : Complex.abs (w - 7 - 8*I) = 3) :
  ∃ (min_dist : ℝ), 
    (∀ (z' w' : ℂ), 
      Complex.abs (z' + 1 + 3*I) = 1 → 
      Complex.abs (w' - 7 - 8*I) = 3 → 
      Complex.abs (z' - w') ≥ min_dist) ∧
    (∃ (z₀ w₀ : ℂ), 
      Complex.abs (z₀ + 1 + 3*I) = 1 ∧ 
      Complex.abs (w₀ - 7 - 8*I) = 3 ∧ 
      Complex.abs (z₀ - w₀) = min_dist) ∧
    min_dist = Real.sqrt 185 - 4 :=
by sorry

end smallest_distance_complex_circles_l4107_410792


namespace sufficient_not_necessary_condition_negation_equivalence_necessary_sufficient_condition_l4107_410743

-- Statement 1
theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  ¬(∀ x : ℝ, |x| > 1 → x > 1) :=
sorry

-- Statement 2
theorem negation_equivalence :
  ¬(∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) :=
sorry

-- Statement 3
theorem necessary_sufficient_condition (a b c : ℝ) :
  (a + b + c = 0) ↔ (a * 1^2 + b * 1 + c = 0) :=
sorry

end sufficient_not_necessary_condition_negation_equivalence_necessary_sufficient_condition_l4107_410743


namespace sum_of_roots_cubic_equation_l4107_410739

theorem sum_of_roots_cubic_equation : 
  let p (x : ℝ) := 3 * x^3 - 15 * x^2 - 36 * x + 7
  ∃ r s t : ℝ, (p r = 0 ∧ p s = 0 ∧ p t = 0) ∧ (r + s + t = 5) :=
by sorry

end sum_of_roots_cubic_equation_l4107_410739


namespace x_zero_sufficient_not_necessary_l4107_410711

theorem x_zero_sufficient_not_necessary :
  (∃ x : ℝ, x = 0 → x^2 - 2*x = 0) ∧
  (∃ x : ℝ, x^2 - 2*x = 0 ∧ x ≠ 0) := by
  sorry

end x_zero_sufficient_not_necessary_l4107_410711


namespace hexagon_side_length_l4107_410765

/-- The length of one side of a regular hexagon with perimeter 43.56 -/
theorem hexagon_side_length : ∃ (s : ℝ), s > 0 ∧ s * 6 = 43.56 ∧ s = 7.26 := by
  sorry

end hexagon_side_length_l4107_410765


namespace walking_speed_l4107_410796

/-- 
Given that:
- Jack's speed is (x^2 - 13x - 30) miles per hour
- Jill covers (x^2 - 6x - 91) miles in (x + 7) hours
- Jack and Jill walk at the same rate

Prove that their speed is 4 miles per hour
-/
theorem walking_speed (x : ℝ) 
  (h1 : x ≠ -7)  -- Assumption to avoid division by zero
  (h2 : x > 0)   -- Assumption for positive speed
  (h3 : (x^2 - 6*x - 91) / (x + 7) = x^2 - 13*x - 30) :  -- Jack and Jill walk at the same rate
  x^2 - 13*x - 30 = 4 := by sorry

end walking_speed_l4107_410796


namespace divisibility_criterion_1207_l4107_410705

-- Define a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define the sum of cubes of digits
def sum_of_cubes_of_digits (n : ℕ) : ℕ :=
  (n / 10) ^ 3 + (n % 10) ^ 3

-- Theorem statement
theorem divisibility_criterion_1207 (x : ℕ) :
  is_two_digit x →
  sum_of_cubes_of_digits x = 344 →
  (1207 % x = 0 ↔ (x = 17 ∨ x = 71)) :=
by sorry

end divisibility_criterion_1207_l4107_410705


namespace three_five_two_takes_five_steps_l4107_410795

/-- Reverses a natural number -/
def reverseNum (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Bool := sorry

/-- Performs one step of the process: reverse, add 3, then add to original -/
def step (n : ℕ) : ℕ := n + (reverseNum n + 3)

/-- Counts the number of steps to reach a palindrome -/
def stepsToBecomePalindrome (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem three_five_two_takes_five_steps :
  352 ≥ 100 ∧ 352 ≤ 400 ∧ 
  ¬isPalindrome 352 ∧
  stepsToBecomePalindrome 352 = 5 := by sorry

end three_five_two_takes_five_steps_l4107_410795


namespace triangle_rotation_l4107_410788

theorem triangle_rotation (α β γ : ℝ) (k m : ℤ) (h1 : 15 * α = 360 * k)
    (h2 : 6 * β = 360 * m) (h3 : α + β + γ = 180) :
  ∃ (n : ℕ+), n * γ = 360 * (n / 5 : ℤ) ∧ ∀ (n' : ℕ+), n' < n → ¬(∃ (l : ℤ), n' * γ = 360 * l) := by
  sorry

end triangle_rotation_l4107_410788


namespace quadratic_root_constant_l4107_410771

/-- 
Given a quadratic equation 5x^2 + 6x + k = 0 with roots (-3 ± √69) / 10,
prove that k = -1.65
-/
theorem quadratic_root_constant (k : ℝ) : 
  (∀ x : ℝ, 5 * x^2 + 6 * x + k = 0 ↔ x = (-3 - Real.sqrt 69) / 10 ∨ x = (-3 + Real.sqrt 69) / 10) →
  k = -1.65 :=
by sorry

end quadratic_root_constant_l4107_410771


namespace bible_length_l4107_410719

/-- The number of pages in John's bible --/
def bible_pages : ℕ := sorry

/-- The number of hours John reads per day --/
def hours_per_day : ℕ := 2

/-- The number of pages John reads per hour --/
def pages_per_hour : ℕ := 50

/-- The number of weeks it takes John to read the entire bible --/
def weeks_to_read : ℕ := 4

/-- The number of days in a week --/
def days_per_week : ℕ := 7

theorem bible_length : bible_pages = 2800 := by sorry

end bible_length_l4107_410719


namespace mp3_song_count_l4107_410744

theorem mp3_song_count (x y : ℕ) : 
  (15 : ℕ) - x + y = 2 * 15 → y = x + 15 := by
sorry

end mp3_song_count_l4107_410744


namespace rectangle_area_divisible_by_12_l4107_410779

theorem rectangle_area_divisible_by_12 (a b c : ℕ) 
  (h1 : a * a + b * b = c * c) : 
  12 ∣ (a * b) :=
by sorry

end rectangle_area_divisible_by_12_l4107_410779


namespace square_perimeter_from_p_shape_l4107_410731

/-- Given a square cut into four equal rectangles arranged to form a 'P' shape with a perimeter of 56,
    the perimeter of the original square is 74 2/3. -/
theorem square_perimeter_from_p_shape (x : ℚ) : 
  (2 * (4 * x) + 4 * x = 56) →  -- Perimeter of 'P' shape
  (4 * (4 * x) = 74 + 2/3) -- Perimeter of original square
  := by sorry

end square_perimeter_from_p_shape_l4107_410731


namespace imaginary_town_population_l4107_410725

theorem imaginary_town_population (n m p : ℕ) 
  (h1 : n^2 + 150 = m^2 + 1) 
  (h2 : n^2 + 300 = p^2) : 
  4 ∣ n := by
  sorry

end imaginary_town_population_l4107_410725


namespace constant_term_expansion_l4107_410736

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = (1/x^2 - 2*x)^6) ∧ 
  (∃ c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) ∧
  (∃! c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) ∧
  c = 240 :=
sorry

end constant_term_expansion_l4107_410736


namespace trig_identity_l4107_410723

theorem trig_identity (α β : ℝ) : 
  (Real.cos α - Real.cos β)^2 - (Real.sin α - Real.sin β)^2 = 
  -4 * (Real.sin ((α - β)/2))^2 * Real.cos (α + β) := by sorry

end trig_identity_l4107_410723


namespace terms_before_four_l4107_410790

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

theorem terms_before_four (a₁ : ℝ) (d : ℝ) (n : ℕ) :
  a₁ = 100 ∧ d = -6 ∧ arithmetic_sequence a₁ d n = 4 → n - 1 = 16 := by
  sorry

end terms_before_four_l4107_410790


namespace solution_set_when_a_is_one_a_value_when_f_always_nonpositive_l4107_410734

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - 2 * |x - a|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | 2/3 < x ∧ x < 2} :=
sorry

-- Part II
theorem a_value_when_f_always_nonpositive :
  (∀ x : ℝ, f a x ≤ 0) → a = -1 :=
sorry

end solution_set_when_a_is_one_a_value_when_f_always_nonpositive_l4107_410734


namespace lucy_groceries_l4107_410783

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 2

/-- The number of packs of cake Lucy bought -/
def cake : ℕ := 12

/-- The total number of grocery packs Lucy bought -/
def total_groceries : ℕ := cookies + cake

theorem lucy_groceries : total_groceries = 14 := by
  sorry

end lucy_groceries_l4107_410783


namespace figure_perimeter_l4107_410791

theorem figure_perimeter (total_area : ℝ) (square_area : ℝ) (rect_width rect_length : ℝ) :
  total_area = 130 →
  3 * square_area + rect_width * rect_length = total_area →
  rect_length = 2 * rect_width →
  square_area = rect_width ^ 2 →
  (3 * square_area.sqrt + rect_width + rect_length) * 2 = 11 * Real.sqrt 26 := by
  sorry

end figure_perimeter_l4107_410791


namespace tan_15_plus_3sin_15_l4107_410775

theorem tan_15_plus_3sin_15 : 
  Real.tan (15 * π / 180) + 3 * Real.sin (15 * π / 180) = 
    (Real.sqrt 6 - Real.sqrt 2 + 3) / (Real.sqrt 6 + Real.sqrt 2) := by
  sorry

end tan_15_plus_3sin_15_l4107_410775


namespace triangle_max_area_l4107_410714

open Real

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a + c = 6 →
  (3 - cos A) * tan (B / 2) = sin A →
  ∃ (S : ℝ), S ≤ 2 * sqrt 2 ∧
    ∀ (S' : ℝ), S' = (1 / 2) * a * c * sin B → S' ≤ S :=
sorry

end triangle_max_area_l4107_410714


namespace M_subset_N_l4107_410747

def M : Set ℝ := {-1, 1}

def N : Set ℝ := {x | (1 : ℝ) / x < 2}

theorem M_subset_N : M ⊆ N := by sorry

end M_subset_N_l4107_410747


namespace average_shift_l4107_410732

theorem average_shift (x₁ x₂ x₃ : ℝ) (h : (x₁ + x₂ + x₃) / 3 = 40) :
  ((x₁ + 40) + (x₂ + 40) + (x₃ + 40)) / 3 = 80 := by
  sorry

end average_shift_l4107_410732


namespace negation_and_range_of_a_l4107_410751

def proposition_p (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0

theorem negation_and_range_of_a :
  (∀ a : ℝ, ¬(proposition_p a) ↔ ∀ x : ℝ, x^2 + 2*a*x + a > 0) ∧
  (∀ a : ℝ, (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1)) :=
sorry

end negation_and_range_of_a_l4107_410751


namespace linear_inequalities_solution_sets_l4107_410702

theorem linear_inequalities_solution_sets :
  (∀ x : ℝ, (4 * (x + 1) ≤ 7 * x + 10 ∧ x - 5 < (x - 8) / 3) ↔ (-2 ≤ x ∧ x < 7 / 2)) ∧
  (∀ x : ℝ, (x - 3 * (x - 2) ≥ 4 ∧ (2 * x - 1) / 5 ≥ (x + 1) / 2) ↔ x ≤ -7) :=
by sorry

end linear_inequalities_solution_sets_l4107_410702


namespace dogwood_tree_count_l4107_410713

/-- The number of dogwood trees in the park after planting -/
def total_trees (initial_trees new_trees : ℕ) : ℕ :=
  initial_trees + new_trees

/-- Theorem stating that the total number of dogwood trees after planting is 83 -/
theorem dogwood_tree_count : total_trees 34 49 = 83 := by
  sorry

end dogwood_tree_count_l4107_410713


namespace eight_people_twentyeight_handshakes_l4107_410782

/-- The number of handshakes in a function where every person shakes hands with every other person exactly once -/
def total_handshakes : ℕ := 28

/-- Calculates the number of handshakes given the number of people -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Proves that 8 people results in 28 handshakes -/
theorem eight_people_twentyeight_handshakes :
  ∃ (n : ℕ), n > 0 ∧ handshakes n = total_handshakes ∧ n = 8 :=
sorry

end eight_people_twentyeight_handshakes_l4107_410782


namespace franks_earnings_l4107_410727

/-- Represents Frank's work schedule and pay rates -/
structure WorkSchedule where
  totalHours : ℕ
  days : ℕ
  regularRate : ℚ
  overtimeRate : ℚ
  day1Hours : ℕ
  day2Hours : ℕ
  day3Hours : ℕ
  day4Hours : ℕ

/-- Calculates the total earnings based on the work schedule -/
def calculateEarnings (schedule : WorkSchedule) : ℚ :=
  let regularHours := min schedule.totalHours (schedule.days * 8)
  let overtimeHours := schedule.totalHours - regularHours
  regularHours * schedule.regularRate + overtimeHours * schedule.overtimeRate

/-- Frank's work schedule for the week -/
def franksSchedule : WorkSchedule :=
  { totalHours := 32
  , days := 4
  , regularRate := 15
  , overtimeRate := 22.5
  , day1Hours := 12
  , day2Hours := 8
  , day3Hours := 8
  , day4Hours := 12
  }

/-- Theorem stating that Frank's total earnings for the week are $660 -/
theorem franks_earnings : calculateEarnings franksSchedule = 660 := by
  sorry

end franks_earnings_l4107_410727


namespace sum_of_quadratic_solutions_l4107_410766

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - 5*x - 20 = 4*x + 25

-- Define a function to represent the sum of solutions
def sum_of_solutions : ℝ := 9

-- Theorem statement
theorem sum_of_quadratic_solutions :
  ∃ (x₁ x₂ : ℝ), 
    quadratic_equation x₁ ∧ 
    quadratic_equation x₂ ∧ 
    x₁ ≠ x₂ ∧
    x₁ + x₂ = sum_of_solutions :=
sorry

end sum_of_quadratic_solutions_l4107_410766


namespace smallest_number_l4107_410708

theorem smallest_number (a b c d : ℚ) (ha : a = 0) (hb : b = 5) (hc : c = -0.3) (hd : d = -1/3) :
  min a (min b (min c d)) = d := by sorry

end smallest_number_l4107_410708


namespace book_profit_rate_l4107_410704

/-- Given a cost price and a selling price, calculate the rate of profit -/
def rate_of_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The rate of profit for a book bought at 50 Rs and sold at 70 Rs is 40% -/
theorem book_profit_rate : rate_of_profit 50 70 = 40 := by
  sorry

end book_profit_rate_l4107_410704


namespace conference_handshakes_l4107_410756

/-- Represents a conference with specific group dynamics -/
structure Conference where
  total_people : Nat
  group_a_size : Nat
  group_b_size : Nat
  exceptions : Nat
  unknown_per_exception : Nat

/-- Calculates the number of handshakes in the conference -/
def handshakes (c : Conference) : Nat :=
  let group_a_b_handshakes := c.group_a_size * c.group_b_size
  let group_b_internal_handshakes := c.group_b_size * (c.group_b_size - 1) / 2
  let exception_handshakes := c.exceptions * c.unknown_per_exception
  group_a_b_handshakes + group_b_internal_handshakes + exception_handshakes

/-- The theorem to be proved -/
theorem conference_handshakes :
  let c := Conference.mk 40 25 15 5 3
  handshakes c = 495 := by
  sorry

#eval handshakes (Conference.mk 40 25 15 5 3)

end conference_handshakes_l4107_410756


namespace set_union_problem_l4107_410748

theorem set_union_problem (A B : Set ℕ) (a : ℕ) :
  A = {1, 2, 3} →
  B = {2, a} →
  A ∪ B = {0, 1, 2, 3} →
  a = 0 := by
sorry

end set_union_problem_l4107_410748


namespace square_root_difference_l4107_410773

theorem square_root_difference (a b : ℝ) (ha : a = 7 + 4 * Real.sqrt 3) (hb : b = 7 - 4 * Real.sqrt 3) :
  Real.sqrt a - Real.sqrt b = 2 * Real.sqrt 3 := by
  sorry

end square_root_difference_l4107_410773


namespace obtuse_triangles_in_100gon_l4107_410758

/-- The number of vertices in the regular polygon -/
def n : ℕ := 100

/-- A function that determines if three vertices form an obtuse triangle in a regular n-gon -/
def is_obtuse (k l m : Fin n) : Prop :=
  (m - k : ℕ) % n > n / 4

/-- The number of ways to choose three vertices forming an obtuse triangle in a regular n-gon -/
def num_obtuse_triangles : ℕ := n * (n / 2 - 1).choose 2

/-- Theorem stating the number of obtuse triangles in a regular 100-gon -/
theorem obtuse_triangles_in_100gon :
  num_obtuse_triangles = 117600 :=
sorry

end obtuse_triangles_in_100gon_l4107_410758


namespace cooking_time_for_remaining_potatoes_l4107_410753

/-- Given a chef cooking potatoes with the following conditions:
  - The total number of potatoes to cook is 15
  - The number of potatoes already cooked is 6
  - Each potato takes 8 minutes to cook
  This theorem proves that the time required to cook the remaining potatoes is 72 minutes. -/
theorem cooking_time_for_remaining_potatoes :
  let total_potatoes : ℕ := 15
  let cooked_potatoes : ℕ := 6
  let cooking_time_per_potato : ℕ := 8
  let remaining_potatoes : ℕ := total_potatoes - cooked_potatoes
  remaining_potatoes * cooking_time_per_potato = 72 := by
  sorry

end cooking_time_for_remaining_potatoes_l4107_410753


namespace same_color_probability_is_correct_l4107_410728

def white_balls : ℕ := 7
def black_balls : ℕ := 6
def red_balls : ℕ := 2

def total_balls : ℕ := white_balls + black_balls + red_balls

def same_color_probability : ℚ :=
  (Nat.choose white_balls 2 + Nat.choose black_balls 2 + Nat.choose red_balls 2) /
  Nat.choose total_balls 2

theorem same_color_probability_is_correct :
  same_color_probability = 37 / 105 := by
  sorry

end same_color_probability_is_correct_l4107_410728


namespace least_number_of_trees_sixty_divisible_by_four_five_six_least_number_of_trees_is_sixty_l4107_410754

theorem least_number_of_trees (n : ℕ) : n > 0 ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n → n ≥ 60 := by
  sorry

theorem sixty_divisible_by_four_five_six : 4 ∣ 60 ∧ 5 ∣ 60 ∧ 6 ∣ 60 := by
  sorry

theorem least_number_of_trees_is_sixty :
  ∃ n : ℕ, n > 0 ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 4 ∣ m ∧ 5 ∣ m ∧ 6 ∣ m) → n ≤ m := by
  sorry

end least_number_of_trees_sixty_divisible_by_four_five_six_least_number_of_trees_is_sixty_l4107_410754


namespace cheezits_calorie_count_l4107_410750

/-- The number of calories in an ounce of Cheezits -/
def calories_per_ounce : ℕ := sorry

/-- The number of bags of Cheezits James ate -/
def bags_eaten : ℕ := 3

/-- The number of ounces per bag of Cheezits -/
def ounces_per_bag : ℕ := 2

/-- The number of minutes James ran -/
def minutes_run : ℕ := 40

/-- The number of calories James burned per minute of running -/
def calories_burned_per_minute : ℕ := 12

/-- The number of excess calories James had after eating and running -/
def excess_calories : ℕ := 420

theorem cheezits_calorie_count :
  calories_per_ounce = 150 ∧
  bags_eaten * ounces_per_bag * calories_per_ounce - minutes_run * calories_burned_per_minute = excess_calories :=
by sorry

end cheezits_calorie_count_l4107_410750


namespace yellow_score_mixture_l4107_410735

theorem yellow_score_mixture (white_ratio : ℕ) (black_ratio : ℕ) (total_yellow : ℕ) 
  (h1 : white_ratio = 7)
  (h2 : black_ratio = 6)
  (h3 : total_yellow = 78) :
  (white_ratio * (total_yellow / (white_ratio + black_ratio)) - 
   black_ratio * (total_yellow / (white_ratio + black_ratio))) / total_yellow = 1 / 13 := by
  sorry

end yellow_score_mixture_l4107_410735


namespace complex_real_condition_l4107_410776

theorem complex_real_condition (m : ℝ) :
  (Complex.I * (m^2 - 2*m - 15) : ℂ).im = 0 → m = 5 ∨ m = -3 := by
  sorry

end complex_real_condition_l4107_410776


namespace not_juggling_sequence_l4107_410794

/-- Definition of the juggling sequence -/
def j : ℕ → ℕ
| 0 => 5
| 1 => 7
| 2 => 2
| n + 3 => j n

/-- Function f that calculates the time when a ball will be caught -/
def f (t : ℕ) : ℕ := t + j (t % 3)

/-- Theorem stating that 572 is not a juggling sequence -/
theorem not_juggling_sequence : ¬ (∀ n m : ℕ, n < 3 → m < 3 → n ≠ m → f n ≠ f m) := by
  sorry

end not_juggling_sequence_l4107_410794


namespace base_conversion_and_arithmetic_l4107_410738

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

def decimal_division (a b : Nat) : Nat :=
  a / b

theorem base_conversion_and_arithmetic :
  let n1 := base_to_decimal [3, 6, 4, 1] 7
  let n2 := base_to_decimal [1, 2, 1] 5
  let n3 := base_to_decimal [4, 5, 7, 1] 6
  let n4 := base_to_decimal [6, 5, 4, 3] 7
  decimal_division n1 n2 - n3 * 2 + n4 = 278 := by sorry

end base_conversion_and_arithmetic_l4107_410738


namespace composition_ratio_l4107_410742

def f (x : ℝ) : ℝ := 3 * x + 4
def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio : (f (g (f 2))) / (g (f (g 2))) = 115 / 73 := by
  sorry

end composition_ratio_l4107_410742


namespace last_interval_correct_l4107_410720

/-- Represents a clock with specific ringing behavior -/
structure Clock where
  n : ℕ  -- number of rings per day
  x : ℝ  -- time between first two rings (in hours)
  y : ℝ  -- increase in time between subsequent rings (in hours)

/-- The time between the last two rings of the clock -/
def lastInterval (c : Clock) : ℝ :=
  c.x + (c.n - 3 : ℝ) * c.y

theorem last_interval_correct (c : Clock) (h : c.n ≥ 2) :
  lastInterval c = c.x + (c.n - 3 : ℝ) * c.y :=
sorry

end last_interval_correct_l4107_410720


namespace slope_implies_y_coordinate_l4107_410781

/-- Given two points A and B in a coordinate plane, if the slope of the line through A and B is 1/3, then the y-coordinate of B is 12. -/
theorem slope_implies_y_coordinate
  (xA yA xB : ℝ)
  (h1 : xA = -3)
  (h2 : yA = 9)
  (h3 : xB = 6) :
  (yB - yA) / (xB - xA) = 1/3 → yB = 12 :=
by sorry

end slope_implies_y_coordinate_l4107_410781


namespace cost_of_one_each_l4107_410707

/-- Represents the cost of goods A, B, and C -/
structure GoodsCost where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The total cost of a combination of goods -/
def total_cost (g : GoodsCost) (a b c : ℝ) : ℝ :=
  a * g.A + b * g.B + c * g.C

theorem cost_of_one_each (g : GoodsCost) 
  (h1 : total_cost g 3 7 1 = 315)
  (h2 : total_cost g 4 10 1 = 420) :
  total_cost g 1 1 1 = 105 := by
  sorry

end cost_of_one_each_l4107_410707


namespace conditional_probability_suitable_joint_structure_l4107_410789

/-- The probability of a child having a suitable joint structure given that they have a suitable physique -/
theorem conditional_probability_suitable_joint_structure 
  (total : ℕ) 
  (physique : ℕ) 
  (joint : ℕ) 
  (both : ℕ) 
  (h_total : total = 20)
  (h_physique : physique = 4)
  (h_joint : joint = 5)
  (h_both : both = 2) :
  (both : ℚ) / physique = 1 / 2 := by
sorry

end conditional_probability_suitable_joint_structure_l4107_410789


namespace polar_to_cartesian_conversion_l4107_410755

/-- Polar to Cartesian Coordinate Conversion Theorem -/
theorem polar_to_cartesian_conversion (x y ρ θ : ℝ) :
  (ρ = 4 * Real.sin θ) ∧
  (x = ρ * Real.cos θ) ∧
  (y = ρ * Real.sin θ) ∧
  (ρ^2 = x^2 + y^2) →
  (x^2 + (y - 2)^2 = 4) :=
by sorry

end polar_to_cartesian_conversion_l4107_410755


namespace parabola_shift_l4107_410798

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_shift :
  let original := Parabola.mk 2 0 0
  let shifted := shift_parabola original 2 1
  shifted = Parabola.mk 2 (-8) 7 := by sorry

end parabola_shift_l4107_410798


namespace smallest_k_for_tangent_circle_l4107_410737

theorem smallest_k_for_tangent_circle : ∃ (h : ℕ+), 
  (1 - h.val)^2 + (1000 + 58 - h.val)^2 = h.val^2 ∧
  ∀ (k : ℕ), k < 58 → ¬∃ (h : ℕ+), (1 - h.val)^2 + (1000 + k - h.val)^2 = h.val^2 := by
  sorry

end smallest_k_for_tangent_circle_l4107_410737


namespace abs_inequality_equivalence_l4107_410780

theorem abs_inequality_equivalence (x : ℝ) : 
  |2*x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 := by sorry

end abs_inequality_equivalence_l4107_410780


namespace remainder_sum_l4107_410760

theorem remainder_sum (n : ℤ) : n % 18 = 11 → (n % 3 + n % 6 = 7) := by
  sorry

end remainder_sum_l4107_410760


namespace g_at_negative_three_l4107_410752

theorem g_at_negative_three (g : ℝ → ℝ) :
  (∀ x, g x = 10 * x^3 - 7 * x^2 - 5 * x + 6) →
  g (-3) = -312 := by
  sorry

end g_at_negative_three_l4107_410752


namespace min_value_fraction_l4107_410733

theorem min_value_fraction (x : ℝ) (h : x > 10) :
  x^2 / (x - 10) ≥ 40 ∧ ∃ y > 10, y^2 / (y - 10) = 40 := by
  sorry

end min_value_fraction_l4107_410733


namespace largest_n_implies_x_l4107_410721

/-- Binary operation @ defined as n - (n * x) -/
def binary_op (n : ℤ) (x : ℝ) : ℝ := n - (n * x)

/-- Theorem stating that if 5 is the largest positive integer n such that n @ x < 21, then x = -3 -/
theorem largest_n_implies_x (x : ℝ) :
  (∀ n : ℤ, n > 0 → binary_op n x < 21 → n ≤ 5) ∧
  (binary_op 5 x < 21) →
  x = -3 := by
  sorry

end largest_n_implies_x_l4107_410721


namespace midpoints_form_equilateral_triangle_l4107_410764

/-- A hexagon inscribed in a unit circle with alternate sides of length 1 -/
structure InscribedHexagon where
  /-- The vertices of the hexagon -/
  vertices : Fin 6 → ℝ × ℝ
  /-- The hexagon is inscribed in a unit circle -/
  inscribed : ∀ i, (vertices i).1^2 + (vertices i).2^2 = 1
  /-- Alternate sides have length 1 -/
  alt_sides_length : ∀ i, dist (vertices i) (vertices ((i + 1) % 6)) = 1 ∨ 
                           dist (vertices ((i + 1) % 6)) (vertices ((i + 2) % 6)) = 1

/-- The midpoints of the three sides that don't have length 1 -/
def midpoints (h : InscribedHexagon) : Fin 3 → ℝ × ℝ := sorry

/-- The theorem statement -/
theorem midpoints_form_equilateral_triangle (h : InscribedHexagon) : 
  ∀ i j, dist (midpoints h i) (midpoints h j) = dist (midpoints h 0) (midpoints h 1) :=
sorry

end midpoints_form_equilateral_triangle_l4107_410764


namespace cheryl_material_usage_l4107_410730

/-- The amount of material Cheryl used for her project -/
def material_used (material1 material2 leftover : ℚ) : ℚ :=
  material1 + material2 - leftover

/-- Theorem stating the total amount of material Cheryl used -/
theorem cheryl_material_usage :
  let material1 : ℚ := 5 / 11
  let material2 : ℚ := 2 / 3
  let leftover : ℚ := 25 / 55
  material_used material1 material2 leftover = 22 / 33 := by
sorry

#eval material_used (5/11) (2/3) (25/55)

end cheryl_material_usage_l4107_410730


namespace two_numbers_sum_and_ratio_l4107_410724

theorem two_numbers_sum_and_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 900) (h4 : y = 19 * x) : x = 45 ∧ y = 855 := by
  sorry

end two_numbers_sum_and_ratio_l4107_410724


namespace min_value_reciprocal_sum_l4107_410741

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  (∀ x y : ℝ, 2 * x + y = 2 → x * y > 0 → 1 / m + 2 / n ≤ 1 / x + 2 / y) ∧
  (∃ x y : ℝ, 2 * x + y = 2 ∧ x * y > 0 ∧ 1 / x + 2 / y = 4) :=
sorry

end min_value_reciprocal_sum_l4107_410741


namespace count_is_58_l4107_410762

/-- A function that generates all permutations of a list -/
def permutations (l : List ℕ) : List (List ℕ) :=
  sorry

/-- A function that converts a list of digits to a number -/
def list_to_number (l : List ℕ) : ℕ :=
  sorry

/-- The set of digits we're working with -/
def digits : List ℕ := [1, 2, 3, 4, 5]

/-- All possible five-digit numbers from the given digits -/
def all_numbers : List ℕ :=
  (permutations digits).map list_to_number

/-- The count of numbers satisfying our conditions -/
def count_numbers : ℕ :=
  (all_numbers.filter (λ n => n > 23145 ∧ n < 43521)).length

theorem count_is_58 : count_numbers = 58 :=
  sorry

end count_is_58_l4107_410762


namespace quadratic_sum_zero_l4107_410784

theorem quadratic_sum_zero (a b c x₁ x₂ : ℝ) (ha : a ≠ 0)
  (hx₁ : a * x₁^2 + b * x₁ + c = 0)
  (hx₂ : a * x₂^2 + b * x₂ + c = 0)
  (s₁ : ℝ := x₁^2005 + x₂^2005)
  (s₂ : ℝ := x₁^2004 + x₂^2004)
  (s₃ : ℝ := x₁^2003 + x₂^2003) :
  a * s₁ + b * s₂ + c * s₃ = 0 := by
sorry

end quadratic_sum_zero_l4107_410784


namespace art_fair_sales_l4107_410761

theorem art_fair_sales (total_visitors : ℕ) (two_painting_buyers : ℕ) (one_painting_buyers : ℕ) (total_paintings_sold : ℕ) :
  total_visitors = 20 →
  two_painting_buyers = 4 →
  one_painting_buyers = 12 →
  total_paintings_sold = 36 →
  ∃ (four_painting_buyers : ℕ),
    four_painting_buyers * 4 + two_painting_buyers * 2 + one_painting_buyers = total_paintings_sold ∧
    four_painting_buyers + two_painting_buyers + one_painting_buyers ≤ total_visitors ∧
    four_painting_buyers = 4 :=
by sorry

end art_fair_sales_l4107_410761


namespace talent_show_gender_difference_l4107_410700

theorem talent_show_gender_difference (total : ℕ) (girls : ℕ) :
  total = 34 →
  girls = 28 →
  girls > total - girls →
  girls - (total - girls) = 22 :=
by
  sorry

end talent_show_gender_difference_l4107_410700


namespace inscribed_circle_radius_in_third_sector_l4107_410703

/-- The radius of an inscribed circle in a sector that is one-third of a circle -/
theorem inscribed_circle_radius_in_third_sector (R : ℝ) (h : R = 5) :
  let r := (R * Real.sqrt 3 - R) / 2
  r * (1 + Real.sqrt 3) = R :=
by sorry

end inscribed_circle_radius_in_third_sector_l4107_410703


namespace joan_apple_count_l4107_410786

/-- Theorem: Given Joan picked 43 apples initially and Melanie gave her 27 more apples, Joan now has 70 apples. -/
theorem joan_apple_count (initial_apples : ℕ) (given_apples : ℕ) (total_apples : ℕ) : 
  initial_apples = 43 → given_apples = 27 → total_apples = initial_apples + given_apples → total_apples = 70 := by
  sorry

end joan_apple_count_l4107_410786


namespace equation_solution_exists_l4107_410701

theorem equation_solution_exists : ∃ x : ℝ, 
  (x^3 - (0.1)^3) / (x^2 + 0.066 + (0.1)^2) = 0.5599999999999999 ∧ 
  abs (x - 0.8) < 0.0001 := by
sorry

end equation_solution_exists_l4107_410701


namespace seven_lines_intersections_l4107_410717

/-- The maximum number of intersection points for n lines in a plane -/
def max_intersections (n : ℕ) : ℕ := n.choose 2

/-- The set of possible numbers of intersection points for 7 lines in a plane -/
def possible_intersections : Set ℕ :=
  {0, 1} ∪ Set.Icc 6 21

theorem seven_lines_intersections :
  (max_intersections 7 = 21) ∧
  (possible_intersections = {0, 1} ∪ Set.Icc 6 21) :=
sorry

end seven_lines_intersections_l4107_410717


namespace three_tetrominoes_with_symmetry_l4107_410772

-- Define the set of tetrominoes
inductive Tetromino
| I -- Line
| O -- Square
| T
| S
| Z

-- Define a function to check if a tetromino has reflectional symmetry
def has_reflectional_symmetry : Tetromino → Bool
| Tetromino.I => true
| Tetromino.O => true
| Tetromino.T => true
| Tetromino.S => false
| Tetromino.Z => false

-- Define the set of all tetrominoes
def all_tetrominoes : List Tetromino :=
  [Tetromino.I, Tetromino.O, Tetromino.T, Tetromino.S, Tetromino.Z]

-- Theorem: Exactly 3 tetrominoes have reflectional symmetry
theorem three_tetrominoes_with_symmetry :
  (all_tetrominoes.filter has_reflectional_symmetry).length = 3 := by
  sorry

end three_tetrominoes_with_symmetry_l4107_410772


namespace exists_integer_divisible_by_15_with_sqrt_between_25_and_26_l4107_410759

theorem exists_integer_divisible_by_15_with_sqrt_between_25_and_26 :
  ∃ n : ℕ+, 15 ∣ n ∧ (25 : ℝ) < Real.sqrt n ∧ Real.sqrt n < 26 := by
  sorry

end exists_integer_divisible_by_15_with_sqrt_between_25_and_26_l4107_410759


namespace arithmetic_calculation_l4107_410718

theorem arithmetic_calculation : 8 / 2 - 3 - 12 + 3 * (5^2 - 4) = 52 := by
  sorry

end arithmetic_calculation_l4107_410718


namespace kate_museum_visits_cost_l4107_410777

/-- Calculates the total amount spent on museum visits over 3 years -/
def total_spent (initial_fee : ℕ) (increased_fee : ℕ) (visits_first_year : ℕ) (visits_per_year_after : ℕ) : ℕ :=
  initial_fee * visits_first_year + increased_fee * visits_per_year_after * 2

/-- Theorem stating the total amount Kate spent on museum visits over 3 years -/
theorem kate_museum_visits_cost :
  let initial_fee := 5
  let increased_fee := 7
  let visits_first_year := 12
  let visits_per_year_after := 4
  total_spent initial_fee increased_fee visits_first_year visits_per_year_after = 116 := by
  sorry

#eval total_spent 5 7 12 4

end kate_museum_visits_cost_l4107_410777


namespace quadratic_equation_real_root_l4107_410746

theorem quadratic_equation_real_root (p : ℝ) : 
  ((-p)^2 - 4 * (3*(p+2)) * (-(4*p+7))) ≥ 0 := by
  sorry

end quadratic_equation_real_root_l4107_410746


namespace extreme_value_conditions_max_min_values_l4107_410778

/-- The function f(x) = x^3 + 3ax^2 + bx -/
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x

/-- The derivative of f(x) -/
def f_deriv (a b x : ℝ) : ℝ := 3*x^2 + 6*a*x + b

theorem extreme_value_conditions (a b : ℝ) :
  f a b (-1) = 0 ∧ f_deriv a b (-1) = 0 →
  a = 2/3 ∧ b = 1 :=
sorry

theorem max_min_values (a b : ℝ) :
  a = 2/3 ∧ b = 1 →
  (∀ x ∈ Set.Icc (-2 : ℝ) (-1/4), f a b x ≤ 0) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) (-1/4), f a b x = 0) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) (-1/4), f a b x ≥ -2) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) (-1/4), f a b x = -2) :=
sorry

end extreme_value_conditions_max_min_values_l4107_410778


namespace kite_cost_l4107_410740

theorem kite_cost (initial_amount : ℕ) (frisbee_cost : ℕ) (remaining_amount : ℕ) (kite_cost : ℕ) : 
  initial_amount = 78 →
  frisbee_cost = 9 →
  remaining_amount = 61 →
  initial_amount = kite_cost + frisbee_cost + remaining_amount →
  kite_cost = 8 := by
sorry

end kite_cost_l4107_410740


namespace club_selection_theorem_l4107_410770

/-- The number of ways to choose a president, vice-president, and secretary from a club -/
def club_selection_ways (total_members boys girls : ℕ) : ℕ :=
  let president_vp_ways := boys * girls + girls * boys
  let secretary_ways := boys * (boys - 1) + girls * (girls - 1)
  president_vp_ways * secretary_ways

/-- Theorem stating the number of ways to choose club positions under specific conditions -/
theorem club_selection_theorem :
  club_selection_ways 25 15 10 = 90000 :=
by sorry

end club_selection_theorem_l4107_410770


namespace jacket_price_reduction_l4107_410757

theorem jacket_price_reduction (x : ℝ) : 
  (1 - x) * (1 - 0.3) * (1 + 0.9047619047619048) = 1 → x = 0.25 := by
sorry

end jacket_price_reduction_l4107_410757


namespace sqrt_18_times_sqrt_32_l4107_410749

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by sorry

end sqrt_18_times_sqrt_32_l4107_410749


namespace sqrt_sum_inequality_l4107_410706

theorem sqrt_sum_inequality (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) 
  (sum_eq : a + b + c = 9) : 
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := by
sorry

end sqrt_sum_inequality_l4107_410706


namespace f_properties_l4107_410729

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - 2*a| + |x - a|

theorem f_properties (a : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, (a = 1 ∧ f 1 x > 3) ↔ (x < 0 ∨ x > 3)) ∧
  (∀ b : ℝ, b ≠ 0 → f a b ≥ f a a) ∧
  (∀ b : ℝ, b ≠ 0 → (f a b = f a a ↔ (2*a - b) * (b - a) ≥ 0)) :=
by sorry

end f_properties_l4107_410729


namespace probability_one_person_two_days_l4107_410787

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of days
def num_days : ℕ := 2

-- Define the number of students required each day
def students_per_day : ℕ := 2

-- Define the function to calculate combinations
def C (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the total number of ways to select students for two days
def total_ways : ℕ := (C num_students students_per_day) * (C num_students students_per_day)

-- Define the number of ways exactly 1 person participates for two consecutive days
def favorable_ways : ℕ := (C num_students 1) * (C (num_students - 1) 1) * (C (num_students - 2) 1)

-- State the theorem
theorem probability_one_person_two_days :
  (favorable_ways : ℚ) / total_ways = 3 / 5 := by sorry

end probability_one_person_two_days_l4107_410787


namespace cube_root_sum_equals_two_sqrt_five_l4107_410768

theorem cube_root_sum_equals_two_sqrt_five :
  (((17 * Real.sqrt 5 + 38) ^ (1/3 : ℝ)) + ((17 * Real.sqrt 5 - 38) ^ (1/3 : ℝ))) = 2 * Real.sqrt 5 :=
by sorry

end cube_root_sum_equals_two_sqrt_five_l4107_410768


namespace lamp_arrangement_probability_l4107_410722

/-- The total number of lamps -/
def total_lamps : ℕ := 8

/-- The number of red lamps -/
def red_lamps : ℕ := 4

/-- The number of blue lamps -/
def blue_lamps : ℕ := 4

/-- The number of lamps to be turned on -/
def lamps_on : ℕ := 4

/-- The probability of the specific arrangement -/
def target_probability : ℚ := 4 / 49

/-- Theorem stating the probability of the specific arrangement -/
theorem lamp_arrangement_probability :
  let total_arrangements := Nat.choose total_lamps red_lamps * Nat.choose total_lamps lamps_on
  let favorable_arrangements := Nat.choose (total_lamps - 2) (red_lamps - 1) * Nat.choose (total_lamps - 2) (lamps_on - 1)
  (favorable_arrangements : ℚ) / total_arrangements = target_probability := by
  sorry


end lamp_arrangement_probability_l4107_410722


namespace no_common_points_l4107_410716

theorem no_common_points : ¬∃ (x y : ℝ), 
  (x^2 + 4*y^2 = 4) ∧ (4*x^2 + y^2 = 4) ∧ (x^2 + y^2 = 1) := by
  sorry

end no_common_points_l4107_410716


namespace trajectory_is_hyperbola_l4107_410763

-- Define the two fixed circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define the moving circle
def movingCircle (cx cy r : ℝ) : Prop := ∀ (x y : ℝ), (x - cx)^2 + (y - cy)^2 = r^2

-- Define the tangency condition
def isTangent (cx cy r : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), circle x y ∧ movingCircle cx cy r ∧ (x - cx)^2 + (y - cy)^2 = r^2

-- Define the trajectory of the center of the moving circle
def trajectory (x y : ℝ) : Prop :=
  ∃ (r : ℝ), isTangent x y r circle1 ∧ isTangent x y r circle2

-- Theorem statement
theorem trajectory_is_hyperbola :
  ∃ (a b : ℝ), ∀ (x y : ℝ), trajectory x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1 :=
sorry

end trajectory_is_hyperbola_l4107_410763


namespace pascal_triangle_30_rows_sum_l4107_410785

/-- The number of elements in the n-th row of Pascal's Triangle -/
def pascal_row_elements (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_sum (n : ℕ) : ℕ :=
  (n * (pascal_row_elements 0 + pascal_row_elements (n - 1))) / 2

theorem pascal_triangle_30_rows_sum :
  pascal_triangle_sum 30 = 465 := by
  sorry

end pascal_triangle_30_rows_sum_l4107_410785
