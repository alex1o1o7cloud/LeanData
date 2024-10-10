import Mathlib

namespace find_x_value_l1399_139955

theorem find_x_value (x : ℝ) : 3639 + 11.95 - x = 3054 → x = 596.95 := by
  sorry

end find_x_value_l1399_139955


namespace jills_income_ratio_l1399_139938

/-- Proves that the ratio of Jill's discretionary income to her net monthly salary is 1/5 -/
theorem jills_income_ratio :
  let net_salary : ℚ := 3500
  let discretionary_income : ℚ := 105 / (15/100)
  discretionary_income / net_salary = 1/5 := by
  sorry

end jills_income_ratio_l1399_139938


namespace additional_bottles_l1399_139963

theorem additional_bottles (initial_bottles : ℕ) (capacity_per_bottle : ℕ) (total_stars : ℕ) : 
  initial_bottles = 2 → capacity_per_bottle = 15 → total_stars = 75 →
  (total_stars - initial_bottles * capacity_per_bottle) / capacity_per_bottle = 3 := by
sorry

end additional_bottles_l1399_139963


namespace largest_integer_quadratic_negative_l1399_139988

theorem largest_integer_quadratic_negative : 
  (∀ m : ℤ, m > 7 → m^2 - 11*m + 24 ≥ 0) ∧ 
  (7^2 - 11*7 + 24 < 0) := by
sorry

end largest_integer_quadratic_negative_l1399_139988


namespace trigonometric_problem_l1399_139951

theorem trigonometric_problem (x y : ℝ) (h_nonzero : x ≠ 0 ∧ y ≠ 0) 
  (h_eq : (x * Real.sin (π/5) + y * Real.cos (π/5)) / (x * Real.cos (π/5) - y * Real.sin (π/5)) = Real.tan (9*π/20)) :
  (y / x = 1) ∧
  (∀ A B : ℝ, 0 < A ∧ 0 < B ∧ A + B = 3*π/4 → 
    Real.sin (2*A) + 2 * Real.cos B ≤ 3/2) ∧
  (∃ A B : ℝ, 0 < A ∧ 0 < B ∧ A + B = 3*π/4 ∧ 
    Real.sin (2*A) + 2 * Real.cos B = 3/2) :=
by sorry

end trigonometric_problem_l1399_139951


namespace arithmetic_sequence_general_term_l1399_139961

/-- An arithmetic sequence with first term 1 and sum of first three terms 9 has general term 2n - 1 -/
theorem arithmetic_sequence_general_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) → -- arithmetic sequence condition
    a 1 = 1 → -- first term condition
    a 1 + a 2 + a 3 = 9 → -- sum of first three terms condition
    ∀ n, a n = 2 * n - 1 := by
  sorry

end arithmetic_sequence_general_term_l1399_139961


namespace binary_multiplication_theorem_l1399_139940

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n |>.reverse

theorem binary_multiplication_theorem :
  let a := [true, true, false, true, false, true]  -- 110101₂
  let b := [true, true, true, false, true]  -- 11101₂
  let c := [true, false, true, false, true, true, true, false, true, false, true]  -- 10101110101₂
  binary_to_nat a * binary_to_nat b = binary_to_nat c := by
  sorry

end binary_multiplication_theorem_l1399_139940


namespace volume_of_region_l1399_139959

-- Define the region in space
def Region : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   (|x + y + 2*z| + |x + y - 2*z| ≤ 12) ∧
                   (x ≥ 0) ∧ (y ≥ 0) ∧ (z ≥ 0)}

-- State the theorem
theorem volume_of_region : MeasureTheory.volume Region = 54 := by
  sorry

end volume_of_region_l1399_139959


namespace ellipse_condition_l1399_139943

def is_ellipse (k : ℝ) : Prop :=
  1 < k ∧ k < 5 ∧ k ≠ 3

theorem ellipse_condition (k : ℝ) :
  (is_ellipse k → (1 < k ∧ k < 5)) ∧
  ¬(1 < k ∧ k < 5 → is_ellipse k) :=
sorry

end ellipse_condition_l1399_139943


namespace range_of_a_l1399_139972

/-- The range of non-negative real number a that satisfies the given conditions -/
theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → ∃ (an bn : ℝ), bn = an^3) →  -- Points are on y = x^3
  (∃ (a1 : ℝ), a1 = a ∧ a ≥ 0) →  -- a1 = a and a ≥ 0
  (∀ n : ℕ, n ≥ 1 → ∃ (cn : ℝ), cn = an + an+1) →  -- cn = an + an+1
  (∀ n : ℕ, n ≥ 1 → ∃ (cn an : ℝ), cn = 1/2 * an + 3/2) →  -- cn = 1/2*an + 3/2
  (∀ n : ℕ, an ≠ 1) →  -- All terms of {an} are not equal to 1
  (∀ n : ℕ, n ≥ 1 → ∃ (kn : ℝ), kn = (bn - 1) / (an - 1)) →  -- kn = (bn - 1) / (an - 1)
  (∃ (k0 : ℝ), ∀ n : ℕ, n ≥ 1 → (kn - k0) * (kn+1 - k0) < 0) →  -- Existence of k0
  (0 ≤ a ∧ a < 7 ∧ a ≠ 1) :=
by
  sorry


end range_of_a_l1399_139972


namespace abs_equation_roots_properties_l1399_139998

def abs_equation (x : ℝ) : Prop := |x|^2 + 2*|x| - 8 = 0

theorem abs_equation_roots_properties :
  ∃ (root1 root2 : ℝ),
    (abs_equation root1 ∧ abs_equation root2) ∧
    (root1 = 2 ∧ root2 = -2) ∧
    (root1 + root2 = 0) ∧
    (root1 * root2 = -4) := by sorry

end abs_equation_roots_properties_l1399_139998


namespace system_solution_l1399_139902

theorem system_solution : ∃! (x y : ℝ), x - y = 2 ∧ 2*x + y = 7 := by
  sorry

end system_solution_l1399_139902


namespace jessica_quarters_l1399_139932

theorem jessica_quarters (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 8 → received = 3 → total = initial + received → total = 11 :=
by sorry

end jessica_quarters_l1399_139932


namespace sister_bars_count_l1399_139944

/-- Represents the number of granola bars in a pack --/
def pack_size : ℕ := 20

/-- Represents the number of days in a week --/
def days_in_week : ℕ := 7

/-- Represents the number of bars traded to Pete --/
def bars_traded : ℕ := 3

/-- Represents the number of sisters --/
def num_sisters : ℕ := 2

/-- Calculates the number of granola bars each sister receives --/
def bars_per_sister : ℕ := (pack_size - days_in_week - bars_traded) / num_sisters

/-- Proves that each sister receives 5 granola bars --/
theorem sister_bars_count : bars_per_sister = 5 := by
  sorry

#eval bars_per_sister  -- This will evaluate to 5

end sister_bars_count_l1399_139944


namespace geometric_sequence_fourth_term_l1399_139953

/-- A geometric sequence of positive integers with first term 5 and fifth term 3125 has its fourth term equal to 625. -/
theorem geometric_sequence_fourth_term : ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 5 →                            -- First term is 5
  a 5 = 3125 →                         -- Fifth term is 3125
  a 4 = 625 := by
sorry

end geometric_sequence_fourth_term_l1399_139953


namespace special_function_properties_l1399_139923

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (f 1 = 1) ∧
  (∀ x y : ℝ, f (x + y) = f x + f y + f x * f y) ∧
  (∀ x : ℝ, x > 0 → f x > 0)

theorem special_function_properties (f : ℝ → ℝ) (hf : SpecialFunction f) :
  (f 0 = 0) ∧
  (∀ n : ℕ, f (n + 1) + 1 = 2 * (f n + 1)) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → f (x + y) > f x) :=
by
  sorry

end special_function_properties_l1399_139923


namespace recreational_space_perimeter_l1399_139920

-- Define the playground and sandbox dimensions
def playground_width : ℕ := 20
def playground_height : ℕ := 16
def sandbox_width : ℕ := 4
def sandbox_height : ℕ := 3

-- Define the sandbox position
def sandbox_top_distance : ℕ := 6
def sandbox_left_distance : ℕ := 8

-- Define the perimeter calculation function
def calculate_perimeter (playground_width playground_height sandbox_width sandbox_height sandbox_top_distance sandbox_left_distance : ℕ) : ℕ :=
  let right_width := playground_width - sandbox_left_distance - sandbox_width
  let bottom_height := playground_height - sandbox_top_distance - sandbox_height
  let right_perimeter := 2 * (playground_height + right_width)
  let bottom_perimeter := 2 * (bottom_height + sandbox_left_distance)
  let left_perimeter := 2 * (sandbox_top_distance + sandbox_left_distance)
  let overlap := 4 * sandbox_left_distance
  right_perimeter + bottom_perimeter + left_perimeter - overlap

-- Theorem statement
theorem recreational_space_perimeter :
  calculate_perimeter playground_width playground_height sandbox_width sandbox_height sandbox_top_distance sandbox_left_distance = 74 := by
  sorry

end recreational_space_perimeter_l1399_139920


namespace ellipse_eccentricity_l1399_139909

theorem ellipse_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, x^2/2 + y^2/m = 1) →  -- ellipse equation
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (x^2/a^2 + y^2/b^2 = 1 ↔ x^2/2 + y^2/m = 1) ∧ 
    c^2 = a^2 - b^2 ∧ c/a = 1/2) →  -- eccentricity condition
  m = 3/2 ∨ m = 8/3 :=
by sorry

end ellipse_eccentricity_l1399_139909


namespace initial_volume_calculation_l1399_139992

/-- The initial volume of a solution in liters -/
def initial_volume : ℝ := 6

/-- The percentage of alcohol in the initial solution -/
def initial_alcohol_percentage : ℝ := 0.30

/-- The volume of pure alcohol added in liters -/
def added_alcohol : ℝ := 2.4

/-- The percentage of alcohol in the final solution -/
def final_alcohol_percentage : ℝ := 0.50

theorem initial_volume_calculation :
  initial_volume * initial_alcohol_percentage + added_alcohol =
  final_alcohol_percentage * (initial_volume + added_alcohol) :=
by sorry

end initial_volume_calculation_l1399_139992


namespace quadratic_solution_sum_l1399_139958

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 5 * x^2 + 4 * x + 20 = 0 ↔ x = a + b * I ∨ x = a - b * I) → 
  a + b^2 = 86/25 := by
sorry

end quadratic_solution_sum_l1399_139958


namespace a_equals_2_sufficient_not_necessary_l1399_139971

def A : Set ℝ := {0, 4}
def B (a : ℝ) : Set ℝ := {2, a^2}

theorem a_equals_2_sufficient_not_necessary :
  (∀ a : ℝ, a = 2 → A ∩ B a = {4}) ∧
  (∃ a : ℝ, a ≠ 2 ∧ A ∩ B a = {4}) :=
by sorry

end a_equals_2_sufficient_not_necessary_l1399_139971


namespace max_ballpoint_pens_l1399_139933

/-- Represents the number of pens of each type -/
structure PenCounts where
  ballpoint : ℕ
  gel : ℕ
  fountain : ℕ

/-- Checks if the given pen counts satisfy all conditions -/
def satisfiesConditions (counts : PenCounts) : Prop :=
  counts.ballpoint + counts.gel + counts.fountain = 15 ∧
  counts.ballpoint ≥ 1 ∧ counts.gel ≥ 1 ∧ counts.fountain ≥ 1 ∧
  10 * counts.ballpoint + 40 * counts.gel + 60 * counts.fountain = 500

/-- Theorem stating that the maximum number of ballpoint pens is 6 -/
theorem max_ballpoint_pens : 
  (∃ counts : PenCounts, satisfiesConditions counts) →
  (∀ counts : PenCounts, satisfiesConditions counts → counts.ballpoint ≤ 6) ∧
  (∃ counts : PenCounts, satisfiesConditions counts ∧ counts.ballpoint = 6) :=
by sorry

end max_ballpoint_pens_l1399_139933


namespace sum_234_78_base5_l1399_139976

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_234_78_base5 : 
  toBase5 (234 + 78) = [2, 2, 2, 2] := by sorry

end sum_234_78_base5_l1399_139976


namespace log_equation_solution_l1399_139952

theorem log_equation_solution (x : ℝ) :
  (Real.log x / Real.log 8 + Real.log (x^3) / Real.log 4 = 9) ↔ (x = 2^(54/11)) :=
by sorry

end log_equation_solution_l1399_139952


namespace total_donation_theorem_l1399_139935

def initial_donation : ℝ := 1707

def percentage_increases : List ℝ := [0.03, 0.05, 0.08, 0.02, 0.10, 0.04, 0.06, 0.09, 0.07, 0.03, 0.05]

def calculate_monthly_donation (prev_donation : ℝ) (percentage_increase : ℝ) : ℝ :=
  prev_donation * (1 + percentage_increase)

def calculate_total_donation (initial : ℝ) (increases : List ℝ) : ℝ :=
  let monthly_donations := increases.scanl calculate_monthly_donation initial
  initial + monthly_donations.sum

theorem total_donation_theorem :
  calculate_total_donation initial_donation percentage_increases = 29906.10 := by
  sorry

end total_donation_theorem_l1399_139935


namespace university_box_cost_l1399_139912

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dim : BoxDimensions) : ℕ :=
  dim.length * dim.width * dim.height

/-- Calculates the number of boxes needed given the total volume and box volume -/
def boxesNeeded (totalVolume boxVolume : ℕ) : ℕ :=
  (totalVolume + boxVolume - 1) / boxVolume

/-- Calculates the total cost given the number of boxes and cost per box -/
def totalCost (numBoxes : ℕ) (costPerBox : ℚ) : ℚ :=
  (numBoxes : ℚ) * costPerBox

/-- Theorem stating the minimum amount the university must spend on boxes -/
theorem university_box_cost
  (boxDim : BoxDimensions)
  (costPerBox : ℚ)
  (totalVolume : ℕ)
  (h1 : boxDim = ⟨20, 20, 15⟩)
  (h2 : costPerBox = 6/5)
  (h3 : totalVolume = 3060000) :
  totalCost (boxesNeeded totalVolume (boxVolume boxDim)) costPerBox = 612 := by
  sorry


end university_box_cost_l1399_139912


namespace star_equation_solution_l1399_139985

-- Define the star operation
noncomputable def star (a b : ℝ) : ℝ :=
  a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- State the theorem
theorem star_equation_solution :
  ∃ y : ℝ, star 3 y = 18 ∧ y = 30 := by
sorry

end star_equation_solution_l1399_139985


namespace computer_price_increase_l1399_139948

theorem computer_price_increase (d : ℝ) (h1 : 2 * d = 540) : 
  d * (1 + 0.3) = 351 := by sorry

end computer_price_increase_l1399_139948


namespace hotel_expenditure_l1399_139970

/-- The total expenditure of 9 persons in a hotel, given specific spending conditions. -/
theorem hotel_expenditure (n : ℕ) (individual_cost : ℕ) (extra_cost : ℕ) : 
  n = 9 → 
  individual_cost = 12 → 
  extra_cost = 8 → 
  (n - 1) * individual_cost + 
  (individual_cost + ((n - 1) * individual_cost + (individual_cost + extra_cost)) / n) = 117 :=
by sorry

end hotel_expenditure_l1399_139970


namespace rotation_270_of_8_minus_4i_l1399_139989

-- Define the rotation function
def rotate270 (z : ℂ) : ℂ := -z.im + z.re * Complex.I

-- State the theorem
theorem rotation_270_of_8_minus_4i :
  rotate270 (8 - 4 * Complex.I) = -4 - 8 * Complex.I := by
  sorry

end rotation_270_of_8_minus_4i_l1399_139989


namespace pirate_count_correct_l1399_139946

/-- The number of pirates on the schooner satisfying the given conditions -/
def pirate_count : ℕ := 60

/-- The fraction of pirates who lost a leg -/
def leg_loss_fraction : ℚ := 2/3

/-- The fraction of fight participants who lost an arm -/
def arm_loss_fraction : ℚ := 54/100

/-- The fraction of fight participants who lost both an arm and a leg -/
def both_loss_fraction : ℚ := 34/100

/-- The number of pirates who did not participate in the fight -/
def non_participants : ℕ := 10

theorem pirate_count_correct : 
  ∃ (p : ℕ), p = pirate_count ∧ 
  (leg_loss_fraction : ℚ) * p = (p - non_participants) * both_loss_fraction + 
    ((p - non_participants) * arm_loss_fraction - (p - non_participants) * both_loss_fraction) +
    (leg_loss_fraction * p - (p - non_participants) * both_loss_fraction) :=
sorry

end pirate_count_correct_l1399_139946


namespace course_selection_theorem_l1399_139930

/-- The number of ways to select 4 courses out of 9, where 3 specific courses are mutually exclusive -/
def course_selection_schemes (total_courses : ℕ) (exclusive_courses : ℕ) (courses_to_choose : ℕ) : ℕ :=
  (exclusive_courses * Nat.choose (total_courses - exclusive_courses) (courses_to_choose - 1)) +
  Nat.choose (total_courses - exclusive_courses) courses_to_choose

/-- Theorem stating that the number of course selection schemes is 75 -/
theorem course_selection_theorem : course_selection_schemes 9 3 4 = 75 := by
  sorry

end course_selection_theorem_l1399_139930


namespace sum_with_reverse_has_even_digit_l1399_139901

/-- A function that reverses a five-digit integer -/
def reverse_digits (n : ℕ) : ℕ :=
  let a := n / 10000
  let b := (n / 1000) % 10
  let c := (n / 100) % 10
  let d := (n / 10) % 10
  let e := n % 10
  e * 10000 + d * 1000 + c * 100 + b * 10 + a

/-- Predicate to check if a natural number has at least one even digit -/
def has_even_digit (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ 2 ∣ d ∧ ∃ k : ℕ, n / (10^k) % 10 = d

theorem sum_with_reverse_has_even_digit (n : ℕ) 
  (h : 10000 ≤ n ∧ n < 100000) : 
  has_even_digit (n + reverse_digits n) :=
sorry

end sum_with_reverse_has_even_digit_l1399_139901


namespace repeating_decimal_division_l1399_139979

theorem repeating_decimal_division :
  let a : ℚ := 54 / 99
  let b : ℚ := 18 / 99
  a / b = 3 := by sorry

end repeating_decimal_division_l1399_139979


namespace complement_of_A_l1399_139903

def U : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}
def A : Set ℕ := {1, 2, 3, 5, 8}

theorem complement_of_A : (Aᶜ : Set ℕ) = {4, 6, 7, 9, 10} := by sorry

end complement_of_A_l1399_139903


namespace jason_attended_11_games_this_month_l1399_139999

/-- Represents the number of football games Jason attended or plans to attend -/
structure FootballGames where
  lastMonth : Nat
  thisMonth : Nat
  nextMonth : Nat
  total : Nat

/-- Given information about Jason's football game attendance -/
def jasonGames : FootballGames where
  lastMonth := 17
  thisMonth := 11 -- This is what we want to prove
  nextMonth := 16
  total := 44

/-- Theorem stating that Jason attended 11 games this month -/
theorem jason_attended_11_games_this_month :
  jasonGames.thisMonth = 11 ∧
  jasonGames.total = jasonGames.lastMonth + jasonGames.thisMonth + jasonGames.nextMonth :=
by sorry

end jason_attended_11_games_this_month_l1399_139999


namespace function_inequality_l1399_139957

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end function_inequality_l1399_139957


namespace sin_double_angle_problem_l1399_139922

theorem sin_double_angle_problem (x : ℝ) (h : Real.sin (x - π/4) = 3/5) : 
  Real.sin (2*x) = 7/25 := by
sorry

end sin_double_angle_problem_l1399_139922


namespace inequality_solution_implies_a_range_l1399_139918

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x, (1 - a) * x > 3 ↔ x < 3 / (1 - a)) → a > 1 := by
  sorry

end inequality_solution_implies_a_range_l1399_139918


namespace fixed_point_on_line_l1399_139956

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

-- Main theorem
theorem fixed_point_on_line (x₁ y₁ x₂ y₂ : ℝ) :
  parabola x₁ y₁ →
  parabola x₂ y₂ →
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  perpendicular x₁ y₁ x₂ y₂ →
  ∃ m : ℝ, x₁ = m*y₁ + 8 ∧ x₂ = m*y₂ + 8 :=
sorry

end fixed_point_on_line_l1399_139956


namespace tan_addition_special_case_l1399_139904

theorem tan_addition_special_case (x : Real) (h : Real.tan x = 3) :
  Real.tan (x + π/3) = (12 * Real.sqrt 3 + 3) / 26 := by
  sorry

end tan_addition_special_case_l1399_139904


namespace equation_solution_l1399_139906

theorem equation_solution :
  ∃ x : ℚ, (5 * x - 3) / (6 * x - 6) = 4 / 3 ∧ x = 5 / 3 := by
  sorry

end equation_solution_l1399_139906


namespace cooking_mode_and_median_l1399_139941

def cooking_data : List Nat := [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8]

def mode (data : List Nat) : Nat :=
  sorry

def median (data : List Nat) : Nat :=
  sorry

theorem cooking_mode_and_median :
  mode cooking_data = 6 ∧ median cooking_data = 6 := by
  sorry

end cooking_mode_and_median_l1399_139941


namespace zero_of_function_l1399_139921

/-- Given a function f(x) = m + (1/3)^x where f(-2) = 0, prove that m = -9 -/
theorem zero_of_function (m : ℝ) : 
  (let f : ℝ → ℝ := λ x ↦ m + (1/3)^x
   f (-2) = 0) → 
  m = -9 := by sorry

end zero_of_function_l1399_139921


namespace same_color_probability_l1399_139924

theorem same_color_probability (total_balls : ℕ) (green_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = green_balls + white_balls)
  (h2 : green_balls = 5)
  (h3 : white_balls = 8) :
  (green_balls * (green_balls - 1) + white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1)) = 19 / 39 := by
  sorry

end same_color_probability_l1399_139924


namespace susan_jen_time_difference_l1399_139942

/-- A relay race with 4 runners -/
structure RelayRace where
  mary_time : ℝ
  susan_time : ℝ
  jen_time : ℝ
  tiffany_time : ℝ

/-- The conditions of the relay race -/
def race_conditions (race : RelayRace) : Prop :=
  race.mary_time = 2 * race.susan_time ∧
  race.susan_time > race.jen_time ∧
  race.jen_time = 30 ∧
  race.tiffany_time = race.mary_time - 7 ∧
  race.mary_time + race.susan_time + race.jen_time + race.tiffany_time = 223

/-- The theorem stating that Susan's time is 10 seconds longer than Jen's time -/
theorem susan_jen_time_difference (race : RelayRace) 
  (h : race_conditions race) : race.susan_time - race.jen_time = 10 := by
  sorry

end susan_jen_time_difference_l1399_139942


namespace molecular_weight_CaI2_l1399_139929

/-- Given that the molecular weight of 3 moles of CaI2 is 882 g/mol,
    prove that the molecular weight of 1 mole of CaI2 is 294 g/mol. -/
theorem molecular_weight_CaI2 (weight_3_moles : ℝ) (h : weight_3_moles = 882) :
  weight_3_moles / 3 = 294 := by
  sorry

end molecular_weight_CaI2_l1399_139929


namespace unique_desk_arrangement_l1399_139993

theorem unique_desk_arrangement (total_desks : ℕ) (h_total : total_desks = 49) :
  ∃! (rows columns : ℕ),
    rows * columns = total_desks ∧
    rows ≥ 2 ∧
    columns ≥ 2 ∧
    (∀ r c : ℕ, r * c = total_desks → r ≥ 2 → c ≥ 2 → r = rows ∧ c = columns) :=
by sorry

end unique_desk_arrangement_l1399_139993


namespace division_problem_l1399_139969

theorem division_problem (N x : ℕ) : 
  (N / x = 500) → 
  (N % x = 20) → 
  (4 * 500 + 20 = 2020) → 
  x = 4 := by
  sorry

end division_problem_l1399_139969


namespace cosine_cube_sum_l1399_139996

theorem cosine_cube_sum (α : ℝ) :
  (Real.cos α)^3 + (Real.cos (α + 2 * Real.pi / 3))^3 + (Real.cos (α - 2 * Real.pi / 3))^3 = 
  3/4 * Real.cos (3 * α) := by
  sorry

end cosine_cube_sum_l1399_139996


namespace factor_tree_X_value_l1399_139900

def factor_tree (X Y Z Q R : ℕ) : Prop :=
  Y = 2 * Q ∧
  Z = 7 * R ∧
  Q = 5 * 3 ∧
  R = 11 * 2 ∧
  X = Y * Z

theorem factor_tree_X_value :
  ∀ X Y Z Q R : ℕ, factor_tree X Y Z Q R → X = 4620 :=
by
  sorry

end factor_tree_X_value_l1399_139900


namespace hyperbola_ellipse_shared_foci_l1399_139964

/-- Given a hyperbola and an ellipse that share the same foci, 
    prove that the parameter m in the hyperbola equation is 7. -/
theorem hyperbola_ellipse_shared_foci (m : ℝ) : 
  (∃ (c : ℝ), c^2 = 8 ∧ c^2 = m + 1) → m = 7 := by
  sorry

end hyperbola_ellipse_shared_foci_l1399_139964


namespace ellipse_intersection_slope_sum_l1399_139939

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Define the line that intersects the ellipse
def intersecting_line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 3

-- Define the condition for the sum of slopes
def slope_sum_condition (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    intersecting_line k x₁ y₁ ∧ intersecting_line k x₂ y₂ ∧
    (y₁ - 1) / x₁ + (y₂ - 1) / x₂ = 1

theorem ellipse_intersection_slope_sum (k : ℝ) :
  slope_sum_condition k → k = 2 :=
sorry

end ellipse_intersection_slope_sum_l1399_139939


namespace joan_has_five_apples_l1399_139949

/-- The number of apples Joan has after giving some away -/
def apples_remaining (initial : ℕ) (given_to_melanie : ℕ) (given_to_sarah : ℕ) : ℕ :=
  initial - given_to_melanie - given_to_sarah

/-- Proof that Joan has 5 apples remaining -/
theorem joan_has_five_apples :
  apples_remaining 43 27 11 = 5 := by
  sorry

end joan_has_five_apples_l1399_139949


namespace right_triangle_area_l1399_139987

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) : 
  hypotenuse = 10 * Real.sqrt 2 →
  angle = 45 * (π / 180) →
  (1 / 2) * (hypotenuse / Real.sqrt 2) * (hypotenuse / Real.sqrt 2) = 50 :=
by sorry

end right_triangle_area_l1399_139987


namespace parallelogram_sides_l1399_139982

/-- Represents a parallelogram with side lengths a and b -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  a_positive : 0 < a
  b_positive : 0 < b

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := 2 * (p.a + p.b)

/-- The difference between perimeters of adjacent triangles formed by diagonals -/
def triangle_perimeter_difference (p : Parallelogram) : ℝ := abs (p.b - p.a)

theorem parallelogram_sides (p : Parallelogram) 
  (h_perimeter : perimeter p = 44)
  (h_diff : triangle_perimeter_difference p = 6) :
  p.a = 8 ∧ p.b = 14 := by
  sorry

#check parallelogram_sides

end parallelogram_sides_l1399_139982


namespace unique_triple_sum_l1399_139910

/-- Two-digit positive integer -/
def TwoDigitPositiveInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem unique_triple_sum (a b c : ℕ) 
  (ha : TwoDigitPositiveInt a) 
  (hb : TwoDigitPositiveInt b) 
  (hc : TwoDigitPositiveInt c) 
  (h : a^3 + 3*b^3 + 9*c^3 = 9*a*b*c + 1) : 
  a + b + c = 9 := by
sorry

end unique_triple_sum_l1399_139910


namespace last_two_digits_of_sum_l1399_139995

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a number encoded as ZARAZA -/
structure Zaraza where
  z : Digit
  a : Digit
  r : Digit
  ne_za : z ≠ a
  ne_zr : z ≠ r
  ne_ar : a ≠ r

/-- Represents a number encoded as ALMAZ -/
structure Almaz where
  a : Digit
  l : Digit
  m : Digit
  z : Digit
  ne_al : a ≠ l
  ne_am : a ≠ m
  ne_az : a ≠ z
  ne_lm : l ≠ m
  ne_lz : l ≠ z
  ne_mz : m ≠ z

/-- Convert Zaraza to a natural number -/
def zarazaToNat (x : Zaraza) : ℕ :=
  x.z.val * 100000 + x.a.val * 10000 + x.r.val * 1000 + x.a.val * 100 + x.z.val * 10 + x.a.val

/-- Convert Almaz to a natural number -/
def almazToNat (x : Almaz) : ℕ :=
  x.a.val * 10000 + x.l.val * 1000 + x.m.val * 100 + x.a.val * 10 + x.z.val

/-- The main theorem -/
theorem last_two_digits_of_sum (zar : Zaraza) (alm : Almaz) 
    (h1 : zarazaToNat zar % 4 = 0)
    (h2 : almazToNat alm % 28 = 0)
    (h3 : zar.z = alm.z ∧ zar.a = alm.a) :
    (zarazaToNat zar + almazToNat alm) % 100 = 32 := by
  sorry


end last_two_digits_of_sum_l1399_139995


namespace odd_integers_between_fractions_l1399_139917

theorem odd_integers_between_fractions : 
  let lower_bound : ℚ := 17 / 4
  let upper_bound : ℚ := 35 / 2
  ∃ (S : Finset ℤ), (∀ n ∈ S, (n : ℚ) > lower_bound ∧ (n : ℚ) < upper_bound ∧ Odd n) ∧ 
                    (∀ n : ℤ, (n : ℚ) > lower_bound ∧ (n : ℚ) < upper_bound ∧ Odd n → n ∈ S) ∧
                    Finset.card S = 7 :=
by sorry

end odd_integers_between_fractions_l1399_139917


namespace inverse_composition_result_l1399_139919

-- Define the function f
def f : Fin 5 → Fin 5
| 1 => 3
| 2 => 5
| 3 => 1
| 4 => 2
| 5 => 4

-- Define the inverse function f⁻¹
def f_inv : Fin 5 → Fin 5
| 1 => 3
| 2 => 4
| 3 => 1
| 4 => 5
| 5 => 2

-- State the theorem
theorem inverse_composition_result :
  f_inv (f_inv (f_inv 5)) = 5 := by sorry

end inverse_composition_result_l1399_139919


namespace constants_are_like_terms_l1399_139916

/-- Two terms are considered like terms if they have the same variables raised to the same powers, or if they are both constants. -/
def like_terms (t1 t2 : ℝ) : Prop :=
  (∃ (c1 c2 : ℝ), t1 = c1 ∧ t2 = c2) ∨ 
  (∃ (c1 c2 : ℝ) (f : ℝ → ℝ), t1 = c1 * f 0 ∧ t2 = c2 * f 0)

/-- Constants 0 and π are like terms. -/
theorem constants_are_like_terms : like_terms 0 π := by
  sorry

end constants_are_like_terms_l1399_139916


namespace exists_unique_function_satisfying_equation_l1399_139947

/-- A functional equation that uniquely determines a function f: ℝ → ℤ --/
def functional_equation (f : ℝ → ℤ) (x₁ x₂ : ℝ) : Prop :=
  0 = (f (-x₁^2 - (x₁ * x₂ - 1)^2))^2 +
      ((f (-x₁^2 - (x₁ * x₂ - 1)^2 + 1) - 1/2)^2 - 1/4)^2 +
      (f (x₁^2 + 2) - 2 * f (x₁^2) + f (x₁^2 - 2))^2 +
      ((f (x₁^2) - f (x₁^2 - 2))^2 - 1)^2 +
      ((f (x₁^2) + f (x₁^2 + 1) - 1/2)^2 - 1/4)^2

/-- The theorem stating the existence of a unique function satisfying the functional equation --/
theorem exists_unique_function_satisfying_equation :
  ∃! f : ℝ → ℤ, (∀ x₁ x₂ : ℝ, functional_equation f x₁ x₂) ∧ Set.range f = Set.univ :=
sorry

end exists_unique_function_satisfying_equation_l1399_139947


namespace tracy_art_fair_sales_l1399_139945

theorem tracy_art_fair_sales : 
  let total_customers : ℕ := 20
  let first_group_size : ℕ := 4
  let second_group_size : ℕ := 12
  let third_group_size : ℕ := 4
  let first_group_paintings_per_customer : ℕ := 2
  let second_group_paintings_per_customer : ℕ := 1
  let third_group_paintings_per_customer : ℕ := 4
  first_group_size + second_group_size + third_group_size = total_customers →
  first_group_size * first_group_paintings_per_customer + 
  second_group_size * second_group_paintings_per_customer + 
  third_group_size * third_group_paintings_per_customer = 36 := by
sorry


end tracy_art_fair_sales_l1399_139945


namespace tv_price_change_l1399_139978

theorem tv_price_change (initial_price : ℝ) (x : ℝ) 
  (h1 : initial_price > 0) 
  (h2 : x > 0) : 
  (initial_price * 0.8 * (1 + x / 100) = initial_price * 1.12) → x = 40 := by
  sorry

end tv_price_change_l1399_139978


namespace square_sum_equals_29_l1399_139986

theorem square_sum_equals_29 (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) : 
  a^2 + b^2 = 29 := by
sorry

end square_sum_equals_29_l1399_139986


namespace exactly_one_greater_than_one_l1399_139936

theorem exactly_one_greater_than_one 
  (x y z : ℝ) 
  (pos_x : x > 0) 
  (pos_y : y > 0) 
  (pos_z : z > 0) 
  (product_one : x * y * z = 1) 
  (sum_inequality : x + y + z > 1/x + 1/y + 1/z) : 
  (x > 1 ∧ y ≤ 1 ∧ z ≤ 1) ∨ 
  (x ≤ 1 ∧ y > 1 ∧ z ≤ 1) ∨ 
  (x ≤ 1 ∧ y ≤ 1 ∧ z > 1) :=
sorry

end exactly_one_greater_than_one_l1399_139936


namespace unqualified_weight_l1399_139968

def flour_label_center : ℝ := 25
def flour_label_tolerance : ℝ := 0.25

def is_qualified (weight : ℝ) : Prop :=
  flour_label_center - flour_label_tolerance ≤ weight ∧ 
  weight ≤ flour_label_center + flour_label_tolerance

theorem unqualified_weight : ¬ (is_qualified 25.26) := by
  sorry

end unqualified_weight_l1399_139968


namespace system_solution_l1399_139980

theorem system_solution (x y z t : ℝ) : 
  (x^2 - 9*y^2 = 0 ∧ x + y + z = 0) ↔ 
  ((x = 3*t ∧ y = t ∧ z = -4*t) ∨ (x = -3*t ∧ y = t ∧ z = 2*t)) :=
by sorry

end system_solution_l1399_139980


namespace fraction_equality_l1399_139907

theorem fraction_equality : (4 * 5) / 10 = 2 := by
  sorry

end fraction_equality_l1399_139907


namespace salt_addition_problem_l1399_139973

theorem salt_addition_problem (x : ℝ) (salt_added : ℝ) : 
  x = 104.99999999999997 →
  let initial_salt := 0.2 * x
  let water_after_evaporation := 0.75 * x
  let volume_after_evaporation := water_after_evaporation + initial_salt
  let final_volume := volume_after_evaporation + 7 + salt_added
  let final_salt := initial_salt + salt_added
  (final_salt / final_volume = 1/3) →
  salt_added = 11.375 := by
sorry

#eval (11.375 : Float)

end salt_addition_problem_l1399_139973


namespace bacteria_growth_time_l1399_139965

/-- The growth factor of bacteria population in one tripling period -/
def tripling_factor : ℕ := 3

/-- The duration in hours of one tripling period -/
def hours_per_tripling : ℕ := 3

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 300

/-- The final number of bacteria -/
def final_bacteria : ℕ := 72900

/-- The time in hours for bacteria to grow from initial to final count -/
def growth_time : ℕ := 15

theorem bacteria_growth_time :
  (tripling_factor ^ (growth_time / hours_per_tripling)) * initial_bacteria = final_bacteria :=
sorry

end bacteria_growth_time_l1399_139965


namespace resort_tips_fraction_l1399_139934

theorem resort_tips_fraction (avg_tips : ℝ) (h : avg_tips > 0) :
  let other_months_tips := 6 * avg_tips
  let august_tips := 6 * avg_tips
  let total_tips := other_months_tips + august_tips
  august_tips / total_tips = 1 / 2 := by
sorry

end resort_tips_fraction_l1399_139934


namespace solve_stick_problem_l1399_139967

def stick_problem (dave_sticks amy_sticks ben_sticks total_sticks : ℕ) : Prop :=
  let total_picked := dave_sticks + amy_sticks + ben_sticks
  let sticks_left := total_sticks - total_picked
  total_picked - sticks_left = 5

theorem solve_stick_problem :
  stick_problem 14 9 12 65 := by
  sorry

end solve_stick_problem_l1399_139967


namespace base7_divisibility_l1399_139975

/-- Converts a base 7 number of the form 25y3₇ to base 10 -/
def base7ToBase10 (y : ℕ) : ℕ := 2 * 7^3 + 5 * 7^2 + y * 7 + 3

/-- Checks if a number is divisible by 19 -/
def isDivisibleBy19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

theorem base7_divisibility (y : ℕ) : 
  y < 7 → (isDivisibleBy19 (base7ToBase10 y) ↔ y = 3) := by sorry

end base7_divisibility_l1399_139975


namespace tan_negative_405_degrees_l1399_139966

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by sorry

end tan_negative_405_degrees_l1399_139966


namespace vessel_mixture_problem_l1399_139960

theorem vessel_mixture_problem (x : ℝ) : 
  (0 < x) ∧ (x < 8) →
  (((8 * 0.16 - (8 * 0.16) * (x / 8)) - ((8 * 0.16 - (8 * 0.16) * (x / 8)) * (x / 8))) / 8 = 0.09) →
  x = 2 := by sorry

end vessel_mixture_problem_l1399_139960


namespace complex_division_simplification_l1399_139926

theorem complex_division_simplification : 
  let i : ℂ := Complex.I
  (2 * i) / (1 + i) = 1 + i := by sorry

end complex_division_simplification_l1399_139926


namespace largest_divisor_n4_minus_n_l1399_139962

/-- A positive integer greater than 1 is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ∃ k, 1 < k ∧ k < n ∧ k ∣ n

/-- The largest integer that always divides n^4 - n for all composite n is 6 -/
theorem largest_divisor_n4_minus_n (n : ℕ) (h : IsComposite n) :
  (∀ m : ℕ, m > 6 → ¬(m ∣ (n^4 - n))) ∧ (6 ∣ (n^4 - n)) := by
  sorry

end largest_divisor_n4_minus_n_l1399_139962


namespace simplify_and_evaluate_l1399_139977

theorem simplify_and_evaluate (x y : ℚ) (hx : x = 1/2) (hy : y = -1) :
  (x - 3*y)^2 - (x - y)*(x + 2*y) = 29/2 := by
  sorry

end simplify_and_evaluate_l1399_139977


namespace license_plate_palindrome_probability_l1399_139925

/-- The probability of a license plate containing at least one palindrome -/
theorem license_plate_palindrome_probability :
  let num_letters : ℕ := 26
  let num_digits : ℕ := 10
  let total_arrangements : ℕ := num_letters^4 * num_digits^4
  let letter_palindromes : ℕ := num_letters^2
  let digit_palindromes : ℕ := num_digits^2
  let prob_letter_palindrome : ℚ := letter_palindromes / (num_letters^4 : ℚ)
  let prob_digit_palindrome : ℚ := digit_palindromes / (num_digits^4 : ℚ)
  let prob_both_palindromes : ℚ := (letter_palindromes * digit_palindromes) / (total_arrangements : ℚ)
  let prob_at_least_one_palindrome : ℚ := prob_letter_palindrome + prob_digit_palindrome - prob_both_palindromes
  prob_at_least_one_palindrome = 131 / 1142 :=
by sorry

end license_plate_palindrome_probability_l1399_139925


namespace product_expansion_l1399_139950

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * ((8 / x) + 12 * x^3 - (2 / x^2)) = (6 / x) + 9 * x^3 - (3 / (2 * x^2)) := by
  sorry

end product_expansion_l1399_139950


namespace A_inter_B_empty_A_union_B_complement_A_inter_complement_B_empty_complement_A_union_complement_B_eq_U_l1399_139911

-- Define the universal set U
def U : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}

-- Define set A
def A : Set ℝ := {x | -5 ≤ x ∧ x < -1}

-- Define set B
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Theorem for the intersection of A and B
theorem A_inter_B_empty : A ∩ B = ∅ := by sorry

-- Theorem for the union of A and B
theorem A_union_B : A ∪ B = {x | -5 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for the intersection of complements of A and B
theorem complement_A_inter_complement_B_empty : (U \ A) ∩ (U \ B) = ∅ := by sorry

-- Theorem for the union of complements of A and B
theorem complement_A_union_complement_B_eq_U : (U \ A) ∪ (U \ B) = U := by sorry

end A_inter_B_empty_A_union_B_complement_A_inter_complement_B_empty_complement_A_union_complement_B_eq_U_l1399_139911


namespace sum_of_solutions_quadratic_l1399_139928

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 10*x + 16) → (∃ y : ℝ, y^2 = 10*y + 16 ∧ x + y = 10) :=
by sorry

end sum_of_solutions_quadratic_l1399_139928


namespace extraneous_root_value_l1399_139983

theorem extraneous_root_value (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (1 / (x^2 - x) + (k - 5) / (x^2 + x) = (k - 1) / (x^2 - 1))) ∧
  (1 / (1^2 - 1) + (k - 5) / (1^2 + 1) ≠ (k - 1) / (1^2 - 1)) →
  k = 3 := by sorry

end extraneous_root_value_l1399_139983


namespace unique_k_for_prime_roots_l1399_139915

theorem unique_k_for_prime_roots : ∃! k : ℕ, 
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (p + q = 73) ∧ (p * q = k) ∧ 
  ∀ x : ℝ, x^2 - 73*x + k = 0 ↔ (x = p ∨ x = q) := by
  sorry

end unique_k_for_prime_roots_l1399_139915


namespace max_area_rectangle_l1399_139997

/-- The maximum area of a rectangle with perimeter 40 cm is 100 square centimeters. -/
theorem max_area_rectangle (x y : ℝ) (h : x + y = 20) : 
  x * y ≤ 100 :=
sorry

end max_area_rectangle_l1399_139997


namespace arithmetic_calculation_l1399_139974

theorem arithmetic_calculation : (-0.5) - (-3.2) + 2.8 - 6.5 = -1 := by
  sorry

end arithmetic_calculation_l1399_139974


namespace arithmetic_sequence_problem_l1399_139927

theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  (a 2)^2 + 12*(a 2) - 8 = 0 →  -- a₂ is a root
  (a 10)^2 + 12*(a 10) - 8 = 0 →  -- a₁₀ is a root
  a 2 ≠ a 10 →  -- a₂ and a₁₀ are distinct roots
  a 6 = -6 := by sorry

end arithmetic_sequence_problem_l1399_139927


namespace biscuit_count_l1399_139937

-- Define the dimensions of the dough sheet
def dough_side : ℕ := 12

-- Define the dimensions of each biscuit
def biscuit_side : ℕ := 3

-- Theorem to prove
theorem biscuit_count : (dough_side * dough_side) / (biscuit_side * biscuit_side) = 16 := by
  sorry

end biscuit_count_l1399_139937


namespace smallest_sticker_collection_l1399_139913

theorem smallest_sticker_collection (S : ℕ) : 
  S > 2 →
  S % 5 = 2 →
  S % 11 = 2 →
  S % 13 = 2 →
  (∀ T : ℕ, T > 2 ∧ T % 5 = 2 ∧ T % 11 = 2 ∧ T % 13 = 2 → S ≤ T) →
  S = 717 :=
by sorry

end smallest_sticker_collection_l1399_139913


namespace geometric_sequence_product_l1399_139981

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 * a 4 = 16 →
  a 1 * a 3 * a 5 = 64 ∨ a 1 * a 3 * a 5 = -64 :=
by
  sorry

end geometric_sequence_product_l1399_139981


namespace lucy_fish_purchase_l1399_139954

/-- The number of fish Lucy needs to buy to reach her desired total -/
def fish_to_buy (initial : ℕ) (desired : ℕ) : ℕ := desired - initial

/-- Theorem: Lucy needs to buy 68 fish -/
theorem lucy_fish_purchase : fish_to_buy 212 280 = 68 := by
  sorry

end lucy_fish_purchase_l1399_139954


namespace negation_of_proposition_ln_negation_l1399_139931

open Real

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x, p x) ↔ (∃ x, ¬ p x) :=
by sorry

theorem ln_negation :
  (¬ ∀ x : ℝ, log x > 1) ↔ (∃ x : ℝ, log x ≤ 1) :=
by sorry

end negation_of_proposition_ln_negation_l1399_139931


namespace sum_of_fractions_equals_two_ninths_l1399_139994

theorem sum_of_fractions_equals_two_ninths :
  (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) +
  (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) + (1 / (8 * 9 : ℚ)) = 2 / 9 := by
  sorry

end sum_of_fractions_equals_two_ninths_l1399_139994


namespace pink_highlighters_count_l1399_139908

theorem pink_highlighters_count (total yellow blue : ℕ) (h1 : total = 22) (h2 : yellow = 8) (h3 : blue = 5) :
  ∃ pink : ℕ, pink + yellow + blue = total ∧ pink = 9 := by
  sorry

end pink_highlighters_count_l1399_139908


namespace binomial_20_2_l1399_139905

theorem binomial_20_2 : Nat.choose 20 2 = 190 := by
  sorry

end binomial_20_2_l1399_139905


namespace third_month_sale_calculation_l1399_139914

/-- Calculates the sale in the third month given the sales of other months and the average -/
def third_month_sale (first_month : ℕ) (second_month : ℕ) (fourth_month : ℕ) (average : ℕ) : ℕ :=
  4 * average - (first_month + second_month + fourth_month)

/-- Theorem stating the sale in the third month given the problem conditions -/
theorem third_month_sale_calculation :
  third_month_sale 2500 4000 1520 2890 = 3540 := by
  sorry

end third_month_sale_calculation_l1399_139914


namespace stratified_sample_size_l1399_139990

/-- Represents the ratio of quantities for three product models -/
structure ProductRatio :=
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)

/-- Calculates the total sample size given the number of items from the smallest group -/
def calculateSampleSize (ratio : ProductRatio) (smallestGroupSample : ℕ) : ℕ :=
  smallestGroupSample * (ratio.a + ratio.b + ratio.c) / ratio.a

/-- Theorem: For a stratified sample with ratio 3:4:7, if the smallest group has 9 items, the total sample size is 42 -/
theorem stratified_sample_size (ratio : ProductRatio) (h1 : ratio.a = 3) (h2 : ratio.b = 4) (h3 : ratio.c = 7) :
  calculateSampleSize ratio 9 = 42 := by
  sorry

#eval calculateSampleSize ⟨3, 4, 7⟩ 9

end stratified_sample_size_l1399_139990


namespace qin_jiushao_v3_l1399_139991

/-- Qin Jiushao algorithm for polynomial evaluation -/
def qin_jiushao (f : ℤ → ℤ) (x : ℤ) : ℕ → ℤ
| 0 => 1
| 1 => qin_jiushao f x 0 * x + 47
| 2 => qin_jiushao f x 1 * x + 0
| 3 => qin_jiushao f x 2 * x - 37
| _ => 0

/-- The polynomial f(x) = x^5 + 47x^4 - 37x^2 + 1 -/
def f (x : ℤ) : ℤ := x^5 + 47*x^4 - 37*x^2 + 1

theorem qin_jiushao_v3 : qin_jiushao f (-1) 3 = 9 := by sorry

end qin_jiushao_v3_l1399_139991


namespace toms_reading_speed_l1399_139984

/-- Tom's reading speed problem -/
theorem toms_reading_speed (normal_speed : ℕ) : 
  (2 * (3 * normal_speed) = 72) → normal_speed = 12 := by
  sorry

#check toms_reading_speed

end toms_reading_speed_l1399_139984
