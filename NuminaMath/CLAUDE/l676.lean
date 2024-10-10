import Mathlib

namespace intersection_of_M_and_N_l676_67664

def M : Set ℝ := {-2, 0, 1}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end intersection_of_M_and_N_l676_67664


namespace square_difference_pattern_l676_67688

theorem square_difference_pattern (n : ℕ+) : (2*n + 1)^2 - (2*n - 1)^2 = 8*n := by
  sorry

end square_difference_pattern_l676_67688


namespace yellow_apples_probability_l676_67695

/-- The probability of choosing 2 yellow apples out of 10 apples, where 4 are yellow -/
theorem yellow_apples_probability (total_apples : ℕ) (yellow_apples : ℕ) (chosen_apples : ℕ)
  (h1 : total_apples = 10)
  (h2 : yellow_apples = 4)
  (h3 : chosen_apples = 2) :
  (yellow_apples.choose chosen_apples : ℚ) / (total_apples.choose chosen_apples) = 2 / 15 :=
by sorry

end yellow_apples_probability_l676_67695


namespace max_x_value_l676_67617

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 6) (prod_sum_eq : x*y + x*z + y*z = 9) :
  x ≤ 4 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ + y₀ + z₀ = 6 ∧ x₀*y₀ + x₀*z₀ + y₀*z₀ = 9 ∧ x₀ = 4 :=
by sorry

end max_x_value_l676_67617


namespace imaginary_part_of_2_minus_i_l676_67666

theorem imaginary_part_of_2_minus_i :
  Complex.im (2 - Complex.I) = -1 := by
  sorry

end imaginary_part_of_2_minus_i_l676_67666


namespace fruit_basket_problem_l676_67644

theorem fruit_basket_problem (total_fruits : ℕ) (mango_count : ℕ) (pear_count : ℕ) (pawpaw_count : ℕ) 
  (h1 : total_fruits = 58)
  (h2 : mango_count = 18)
  (h3 : pear_count = 10)
  (h4 : pawpaw_count = 12) :
  ∃ (lemon_count : ℕ), 
    lemon_count = (total_fruits - (mango_count + pear_count + pawpaw_count)) / 2 ∧ 
    lemon_count = 9 := by
  sorry

end fruit_basket_problem_l676_67644


namespace darwin_gas_expense_l676_67652

def initial_amount : ℝ := 600
def final_amount : ℝ := 300

theorem darwin_gas_expense (x : ℝ) 
  (h1 : 0 < x ∧ x < 1) 
  (h2 : final_amount = initial_amount - x * initial_amount - (1/4) * (initial_amount - x * initial_amount)) :
  x = 1/3 := by
sorry

end darwin_gas_expense_l676_67652


namespace min_workers_theorem_l676_67618

/-- Represents the company's keychain production and sales model -/
structure KeychainCompany where
  maintenance_fee : ℝ  -- Daily maintenance fee
  worker_wage : ℝ      -- Hourly wage per worker
  keychains_per_hour : ℝ  -- Keychains produced per worker per hour
  keychain_price : ℝ   -- Price of each keychain
  work_hours : ℝ       -- Hours in a workday

/-- Calculates the minimum number of workers needed for profit -/
def min_workers_for_profit (company : KeychainCompany) : ℕ :=
  sorry

/-- Theorem stating the minimum number of workers needed for profit -/
theorem min_workers_theorem (company : KeychainCompany) 
  (h1 : company.maintenance_fee = 500)
  (h2 : company.worker_wage = 15)
  (h3 : company.keychains_per_hour = 5)
  (h4 : company.keychain_price = 3.10)
  (h5 : company.work_hours = 8) :
  min_workers_for_profit company = 126 :=
sorry

end min_workers_theorem_l676_67618


namespace volunteer_distribution_count_l676_67687

/-- The number of volunteers --/
def num_volunteers : ℕ := 7

/-- The number of positions --/
def num_positions : ℕ := 4

/-- The number of ways to choose 2 people from 5 --/
def choose_two_from_five : ℕ := (5 * 4) / (2 * 1)

/-- The number of ways to permute 4 items --/
def permute_four : ℕ := 4 * 3 * 2 * 1

/-- The total number of ways to distribute volunteers when A and B can be in the same group --/
def total_ways : ℕ := choose_two_from_five * permute_four

/-- The number of ways where A and B are in the same position --/
def same_position_ways : ℕ := permute_four

/-- The number of ways for A and B not to serve at the same position --/
def different_position_ways : ℕ := total_ways - same_position_ways

theorem volunteer_distribution_count :
  different_position_ways = 216 :=
sorry

end volunteer_distribution_count_l676_67687


namespace function_equality_implies_sum_l676_67616

theorem function_equality_implies_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 2) = 2 * x^2 + 5 * x + 3) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 0 := by
  sorry

end function_equality_implies_sum_l676_67616


namespace distance_2_neg5_distance_neg2_neg5_distance_1_neg3_solutions_abs_x_plus_1_eq_2_min_value_range_l676_67673

-- Define distance between two points on a number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Theorem 1: Distance between 2 and -5
theorem distance_2_neg5 : distance 2 (-5) = 7 := by sorry

-- Theorem 2: Distance between -2 and -5
theorem distance_neg2_neg5 : distance (-2) (-5) = 3 := by sorry

-- Theorem 3: Distance between 1 and -3
theorem distance_1_neg3 : distance 1 (-3) = 4 := by sorry

-- Theorem 4: Solutions for |x + 1| = 2
theorem solutions_abs_x_plus_1_eq_2 : 
  ∀ x : ℝ, |x + 1| = 2 ↔ x = 1 ∨ x = -3 := by sorry

-- Theorem 5: Range of x for minimum value of |x+1| + |x-2|
theorem min_value_range : 
  ∀ x : ℝ, (∀ y : ℝ, |x+1| + |x-2| ≤ |y+1| + |y-2|) → -1 ≤ x ∧ x ≤ 2 := by sorry

end distance_2_neg5_distance_neg2_neg5_distance_1_neg3_solutions_abs_x_plus_1_eq_2_min_value_range_l676_67673


namespace problem_solution_l676_67694

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a / x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + a * x - 6 * log x

def h (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x + 4

theorem problem_solution :
  -- Part I
  (∀ a x, x > 0 → 
    (a ≥ 0 → (deriv (f a)) x > 0) ∧ 
    (a < 0 → ((0 < x ∧ x < -a) → (deriv (f a)) x < 0) ∧ 
             (x > -a → (deriv (f a)) x > 0))) ∧
  -- Part II
  (∀ a, (∀ x, x > 0 → (deriv (g a)) x ≥ 0) → a ≥ 5/2) ∧
  -- Part III
  (∀ m, (∃ x₁, 0 < x₁ ∧ x₁ < 1 ∧ 
        ∀ x₂, 1 ≤ x₂ ∧ x₂ ≤ 2 → g 2 x₁ ≥ h m x₂) → 
        m ≥ 8 - 5 * log 2) := by
  sorry

end problem_solution_l676_67694


namespace oranges_and_cookies_donation_l676_67627

theorem oranges_and_cookies_donation (total_oranges : ℕ) (total_cookies : ℕ) (num_children : ℕ) 
  (h_oranges : total_oranges = 81)
  (h_cookies : total_cookies = 65)
  (h_children : num_children = 7) :
  (total_oranges % num_children = 4) ∧ (total_cookies % num_children = 2) :=
by sorry

end oranges_and_cookies_donation_l676_67627


namespace molecular_weight_CCl4_proof_l676_67680

/-- The molecular weight of CCl4 in g/mol -/
def molecular_weight_CCl4 : ℝ := 152

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 7

/-- The total weight of the given moles of CCl4 in g/mol -/
def given_total_weight : ℝ := 1064

/-- Theorem stating that the molecular weight of CCl4 is correct given the condition -/
theorem molecular_weight_CCl4_proof :
  molecular_weight_CCl4 * given_moles = given_total_weight :=
by sorry

end molecular_weight_CCl4_proof_l676_67680


namespace amanda_remaining_budget_l676_67682

/- Define the budgets -/
def samuel_budget : ℚ := 25
def kevin_budget : ℚ := 20
def laura_budget : ℚ := 18
def amanda_budget : ℚ := 15

/- Define the regular ticket prices -/
def samuel_ticket_price : ℚ := 14
def kevin_ticket_price : ℚ := 10
def laura_ticket_price : ℚ := 10
def amanda_ticket_price : ℚ := 8

/- Define the discount rates -/
def general_discount : ℚ := 0.1
def student_discount : ℚ := 0.1

/- Define Samuel's additional expenses -/
def samuel_drink : ℚ := 6
def samuel_popcorn : ℚ := 3
def samuel_candy : ℚ := 1

/- Define Kevin's additional expense -/
def kevin_combo : ℚ := 7

/- Define Laura's additional expenses -/
def laura_popcorn : ℚ := 4
def laura_drink : ℚ := 2

/- Calculate discounted ticket prices -/
def samuel_discounted_ticket : ℚ := samuel_ticket_price * (1 - general_discount)
def kevin_discounted_ticket : ℚ := kevin_ticket_price * (1 - general_discount)
def laura_discounted_ticket : ℚ := laura_ticket_price * (1 - general_discount)
def amanda_discounted_ticket : ℚ := amanda_ticket_price * (1 - general_discount) * (1 - student_discount)

/- Define the theorem -/
theorem amanda_remaining_budget :
  amanda_budget - amanda_discounted_ticket = 8.52 := by sorry

end amanda_remaining_budget_l676_67682


namespace race_time_difference_l676_67619

def malcolm_speed : ℝ := 5.5
def joshua_speed : ℝ := 7.5
def race_distance : ℝ := 12

theorem race_time_difference : 
  (joshua_speed * race_distance) - (malcolm_speed * race_distance) = 24 := by
  sorry

end race_time_difference_l676_67619


namespace horner_method_equals_f_at_2_l676_67651

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + x + 1

-- Define Horner's method for this specific polynomial
def horner_method (x : ℝ) : ℝ := x * (x * (3 * x + 2) + 1) + 1

-- Theorem statement
theorem horner_method_equals_f_at_2 : 
  horner_method 2 = f 2 ∧ f 2 = 35 := by sorry

end horner_method_equals_f_at_2_l676_67651


namespace recurring_decimal_sum_l676_67601

/-- The sum of 0.3̄, 0.04̄, and 0.005̄ is equal to 112386/296703 -/
theorem recurring_decimal_sum : 
  (1 : ℚ) / 3 + 4 / 99 + 5 / 999 = 112386 / 296703 := by sorry

end recurring_decimal_sum_l676_67601


namespace f_of_two_equals_five_l676_67628

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 2 * (x - 1) + 3

-- State the theorem
theorem f_of_two_equals_five : f 2 = 5 := by
  sorry

end f_of_two_equals_five_l676_67628


namespace sphere_volume_from_surface_area_l676_67610

theorem sphere_volume_from_surface_area (O : Set ℝ) (surface_area : ℝ) (volume : ℝ) :
  (∃ (r : ℝ), surface_area = 4 * Real.pi * r^2) →
  surface_area = 4 * Real.pi →
  (∃ (r : ℝ), volume = (4 / 3) * Real.pi * r^3) →
  volume = (4 / 3) * Real.pi :=
by sorry

end sphere_volume_from_surface_area_l676_67610


namespace range_of_f_l676_67681

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem range_of_f :
  (∀ x : ℝ, (1 + x > 0 ∧ 1 - x > 0) → (3/4 ≤ f x ∧ f x ≤ 57)) →
  Set.range f = Set.Ici 0 := by
  sorry

end range_of_f_l676_67681


namespace black_car_overtake_time_l676_67670

/-- Proves that the time for the black car to overtake the red car is 3 hours. -/
theorem black_car_overtake_time (red_speed black_speed initial_distance : ℝ) 
  (h1 : red_speed = 40)
  (h2 : black_speed = 50)
  (h3 : initial_distance = 30)
  (h4 : red_speed > 0)
  (h5 : black_speed > red_speed) :
  (initial_distance / (black_speed - red_speed)) = 3 := by
  sorry

#check black_car_overtake_time

end black_car_overtake_time_l676_67670


namespace quadratic_root_difference_l676_67660

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ |x - y| = 1) →
  p = Real.sqrt (4*q + 1) :=
by sorry

end quadratic_root_difference_l676_67660


namespace proportion_problem_l676_67634

theorem proportion_problem (x y : ℚ) : 
  (3/4 : ℚ) / x = 7/8 → x / y = 5/6 → x = 6/7 ∧ y = 36/35 := by
  sorry

end proportion_problem_l676_67634


namespace least_frood_number_l676_67631

def drop_score (n : ℕ) : ℕ := n * (n + 1) / 2

def eat_score (n : ℕ) : ℕ := 10 * n

theorem least_frood_number : ∀ k : ℕ, k < 20 → drop_score k ≤ eat_score k ∧ drop_score 20 > eat_score 20 := by
  sorry

end least_frood_number_l676_67631


namespace line_tangent_to_circle_l676_67643

/-- The line l is tangent to the circle C -/
theorem line_tangent_to_circle :
  ∀ (x y : ℝ),
  (x - y + 4 = 0) →
  ((x - 2)^2 + (y - 2)^2 = 8) →
  ∃! (p : ℝ × ℝ), p.1 - p.2 + 4 = 0 ∧ (p.1 - 2)^2 + (p.2 - 2)^2 = 8 :=
sorry

end line_tangent_to_circle_l676_67643


namespace courtyard_length_l676_67661

/-- Calculates the length of a rectangular courtyard given its width, number of bricks, and brick dimensions --/
theorem courtyard_length 
  (width : ℝ) 
  (num_bricks : ℕ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (h1 : width = 16) 
  (h2 : num_bricks = 20000) 
  (h3 : brick_length = 0.2) 
  (h4 : brick_width = 0.1) : 
  (num_bricks : ℝ) * brick_length * brick_width / width = 25 := by
  sorry

end courtyard_length_l676_67661


namespace minimal_tile_placement_l676_67697

/-- Represents a tile placement on a grid -/
structure TilePlacement where
  tiles : ℕ
  grid_size : ℕ
  is_valid : Bool

/-- Checks if a tile placement is valid -/
def is_valid_placement (p : TilePlacement) : Prop :=
  p.is_valid ∧ 
  p.grid_size = 8 ∧ 
  p.tiles > 0 ∧ 
  p.tiles ≤ 32 ∧
  ∀ (t : TilePlacement), t.tiles < p.tiles → ¬t.is_valid

theorem minimal_tile_placement : 
  ∃ (p : TilePlacement), is_valid_placement p ∧ p.tiles = 28 := by
  sorry

end minimal_tile_placement_l676_67697


namespace stratified_sampling_most_appropriate_l676_67640

/-- Represents the different employee categories in the company -/
inductive EmployeeCategory
  | Senior
  | Intermediate
  | General

/-- Represents the company's employee distribution -/
structure CompanyDistribution where
  total : Nat
  senior : Nat
  intermediate : Nat
  general : Nat
  senior_count : senior ≤ total
  intermediate_count : intermediate ≤ total
  general_count : general ≤ total
  total_sum : senior + intermediate + general = total

/-- Represents a sampling method -/
inductive SamplingMethod
  | Simple
  | Stratified
  | Cluster
  | Systematic

/-- Determines the most appropriate sampling method given a company distribution and sample size -/
def mostAppropriateSamplingMethod (dist : CompanyDistribution) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is the most appropriate method for the given scenario -/
theorem stratified_sampling_most_appropriate (dist : CompanyDistribution) (sampleSize : Nat) :
  dist.total = 150 ∧ dist.senior = 15 ∧ dist.intermediate = 45 ∧ dist.general = 90 ∧ sampleSize = 30 →
  mostAppropriateSamplingMethod dist sampleSize = SamplingMethod.Stratified :=
  sorry

end stratified_sampling_most_appropriate_l676_67640


namespace chocolate_probability_l676_67656

/-- Represents a chocolate bar with dark and white segments -/
structure ChocolateBar :=
  (segments : List (Float × Bool))  -- List of (length, isDark) pairs

/-- The process of cutting and switching chocolate bars -/
def cutAndSwitch (p : Float) (bar1 bar2 : ChocolateBar) : ChocolateBar × ChocolateBar :=
  sorry

/-- Checks if the chocolate at 1/3 and 2/3 are the same type -/
def sameTypeAt13And23 (bar : ChocolateBar) : Bool :=
  sorry

/-- Performs the cutting and switching process for n steps -/
def processSteps (n : Nat) (bar1 bar2 : ChocolateBar) : ChocolateBar × ChocolateBar :=
  sorry

/-- The probability of getting the same type at 1/3 and 2/3 after n steps -/
def probabilitySameType (n : Nat) : Float :=
  sorry

theorem chocolate_probability :
  probabilitySameType 100 = 1/2 * (1 + (1/3)^100) :=
sorry

end chocolate_probability_l676_67656


namespace stratified_sample_correct_l676_67614

/-- Represents the count of households in each income category -/
structure Population :=
  (high : ℕ)
  (middle : ℕ)
  (low : ℕ)

/-- Represents the sample sizes for each income category -/
structure Sample :=
  (high : ℕ)
  (middle : ℕ)
  (low : ℕ)

/-- Calculates the total population size -/
def totalPopulation (p : Population) : ℕ :=
  p.high + p.middle + p.low

/-- Checks if the sample sizes are proportional to the population sizes -/
def isProportionalSample (p : Population) (s : Sample) (sampleSize : ℕ) : Prop :=
  s.high * (totalPopulation p) = sampleSize * p.high ∧
  s.middle * (totalPopulation p) = sampleSize * p.middle ∧
  s.low * (totalPopulation p) = sampleSize * p.low

/-- The main theorem stating that the given sample is proportional for the given population -/
theorem stratified_sample_correct 
  (pop : Population) 
  (sample : Sample) : 
  pop.high = 150 → 
  pop.middle = 360 → 
  pop.low = 90 → 
  sample.high = 25 → 
  sample.middle = 60 → 
  sample.low = 15 → 
  isProportionalSample pop sample 100 := by
  sorry


end stratified_sample_correct_l676_67614


namespace symmetry_problem_l676_67636

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_wrt_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

/-- Given that z₁ = 1 - 2i and z₁ and z₂ are symmetric with respect to the imaginary axis,
    prove that z₂ = -1 - 2i. -/
theorem symmetry_problem (z₁ z₂ : ℂ) 
    (h₁ : z₁ = 1 - 2*I) 
    (h₂ : symmetric_wrt_imaginary_axis z₁ z₂) : 
  z₂ = -1 - 2*I :=
sorry

end symmetry_problem_l676_67636


namespace contestant_speaking_orders_l676_67637

theorem contestant_speaking_orders :
  let total_contestants : ℕ := 6
  let restricted_contestant : ℕ := 1
  let available_positions : ℕ := total_contestants - 2

  available_positions * Nat.factorial (total_contestants - restricted_contestant) = 480 :=
by sorry

end contestant_speaking_orders_l676_67637


namespace gingerbread_problem_l676_67605

theorem gingerbread_problem (total : ℕ) (red_hats blue_boots both : ℕ) : 
  red_hats = 6 →
  blue_boots = 9 →
  2 * red_hats = total →
  both = red_hats + blue_boots - total →
  both = 3 := by
sorry

end gingerbread_problem_l676_67605


namespace pyramid_volume_l676_67686

/-- The volume of a triangular pyramid with an equilateral base of side length 6√3 and height 9 is 81√3 -/
theorem pyramid_volume : 
  let s : ℝ := 6 * Real.sqrt 3
  let base_area : ℝ := (Real.sqrt 3 / 4) * s^2
  let height : ℝ := 9
  let volume : ℝ := (1/3) * base_area * height
  volume = 81 * Real.sqrt 3 := by sorry

end pyramid_volume_l676_67686


namespace largest_r_same_range_l676_67690

/-- A quadratic polynomial function -/
def f (r : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 3 * x + r

/-- The theorem stating the largest value of r for which f and f ∘ f have the same range -/
theorem largest_r_same_range :
  ∃ (r_max : ℝ), r_max = 15/8 ∧
  ∀ (r : ℝ), Set.range (f r) = Set.range (f r ∘ f r) ↔ r ≤ r_max :=
sorry

end largest_r_same_range_l676_67690


namespace points_separated_by_line_l676_67645

/-- Definition of a line in 2D space --/
def Line (a b c : ℝ) : ℝ × ℝ → ℝ :=
  fun p => a * p.1 + b * p.2 + c

/-- Definition of η for two points with respect to a line --/
def eta (l : ℝ × ℝ → ℝ) (p1 p2 : ℝ × ℝ) : ℝ :=
  (l p1) * (l p2)

/-- Definition of two points being separated by a line --/
def separatedByLine (l : ℝ × ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  eta l p1 p2 < 0

/-- Theorem: Points A(1,2) and B(-1,0) are separated by the line x+y-1=0 --/
theorem points_separated_by_line :
  let l := Line 1 1 (-1)
  let A := (1, 2)
  let B := (-1, 0)
  separatedByLine l A B := by
  sorry


end points_separated_by_line_l676_67645


namespace simplify_expression_l676_67685

theorem simplify_expression :
  2 - (1 / (2 + Real.sqrt 5)) + (1 / (2 - Real.sqrt 5)) = 2 * (1 - Real.sqrt 5) :=
by sorry

end simplify_expression_l676_67685


namespace distance_to_origin_of_complex_number_l676_67668

theorem distance_to_origin_of_complex_number : ∃ (z : ℂ), 
  z = (2 * Complex.I) / (1 - Complex.I) ∧ 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end distance_to_origin_of_complex_number_l676_67668


namespace curve_expression_bound_l676_67676

theorem curve_expression_bound (x y : ℝ) : 
  4 * x^2 + y^2 = 16 → -4 ≤ Real.sqrt 3 * x + (1/2) * y ∧ Real.sqrt 3 * x + (1/2) * y ≤ 4 := by
  sorry

end curve_expression_bound_l676_67676


namespace rhombus_side_length_l676_67659

/-- The length of a side of a rhombus given one diagonal and its area -/
theorem rhombus_side_length 
  (d1 : ℝ) 
  (area : ℝ) 
  (h1 : d1 = 16) 
  (h2 : area = 327.90242451070714) : 
  ∃ (side : ℝ), abs (side - 37.73592452822641) < 1e-10 := by
  sorry

end rhombus_side_length_l676_67659


namespace dads_dimes_count_l676_67611

/-- The number of dimes Tom's dad gave him -/
def dimes_from_dad (initial_dimes final_dimes : ℕ) : ℕ :=
  final_dimes - initial_dimes

/-- Proof that Tom's dad gave him 33 dimes -/
theorem dads_dimes_count : dimes_from_dad 15 48 = 33 := by
  sorry

end dads_dimes_count_l676_67611


namespace additional_cats_needed_l676_67630

def current_cats : ℕ := 11
def target_cats : ℕ := 43

theorem additional_cats_needed : target_cats - current_cats = 32 := by
  sorry

end additional_cats_needed_l676_67630


namespace m_range_theorem_l676_67654

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem m_range_theorem (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ Set.Icc (-2) 2, f x ≠ 0 → True)  -- f is defined on [-2, 2]
  (h2 : is_even f)
  (h3 : monotone_decreasing_on f 0 2)
  (h4 : ∀ m, f (1 - m) < f m) :
  ∀ m, -2 ≤ m ∧ m < (1/2) := by
sorry

end m_range_theorem_l676_67654


namespace halves_in_two_sevenths_l676_67633

theorem halves_in_two_sevenths : (2 : ℚ) / 7 / (1 : ℚ) / 2 = 4 / 7 := by
  sorry

end halves_in_two_sevenths_l676_67633


namespace hanoi_moves_correct_l676_67612

/-- The minimum number of moves required to solve the Tower of Hanoi problem with n disks -/
def hanoi_moves (n : ℕ) : ℕ := 2^n - 1

/-- Theorem: The minimum number of moves required to solve the Tower of Hanoi problem with n disks is 2^n - 1 -/
theorem hanoi_moves_correct (n : ℕ) : hanoi_moves n = 2^n - 1 := by
  sorry

end hanoi_moves_correct_l676_67612


namespace ellipse_axes_l676_67642

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 - 12 = 2 * x + 4 * y

-- Define the standard form of the ellipse
def standard_form (a b h k : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

-- Theorem stating the semi-major and semi-minor axes of the ellipse
theorem ellipse_axes :
  ∃ h k : ℝ, 
    (∀ x y : ℝ, ellipse_equation x y ↔ standard_form 17 8.5 h k x y) ∧
    (17 > 8.5) :=
sorry

end ellipse_axes_l676_67642


namespace inequality_implies_a_positive_l676_67650

theorem inequality_implies_a_positive (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-2) (-1) → x^2 + x + a > 0) →
  a > 0 := by
  sorry

end inequality_implies_a_positive_l676_67650


namespace money_lasts_9_weeks_l676_67689

def lawn_mowing_earnings : ℕ := 9
def weed_eating_earnings : ℕ := 18
def weekly_spending : ℕ := 3

theorem money_lasts_9_weeks : 
  (lawn_mowing_earnings + weed_eating_earnings) / weekly_spending = 9 := by
  sorry

end money_lasts_9_weeks_l676_67689


namespace number_of_factors_of_n_l676_67648

def n : ℕ := 2^2 * 3^2 * 7^2

theorem number_of_factors_of_n : (Finset.card (Nat.divisors n)) = 27 := by
  sorry

end number_of_factors_of_n_l676_67648


namespace coefficient_x4_in_expansion_l676_67672

theorem coefficient_x4_in_expansion (x : ℝ) : 
  (Finset.range 9).sum (λ k => (Nat.choose 8 k : ℝ) * x^k * (2 * Real.sqrt 3)^(8 - k)) = 
  10080 * x^4 + (Finset.range 9).sum (λ k => if k ≠ 4 then (Nat.choose 8 k : ℝ) * x^k * (2 * Real.sqrt 3)^(8 - k) else 0) := by
sorry

end coefficient_x4_in_expansion_l676_67672


namespace probability_all_odd_is_correct_l676_67662

def total_slips : ℕ := 10
def odd_slips : ℕ := 5
def drawn_slips : ℕ := 4

def probability_all_odd : ℚ := (odd_slips.choose drawn_slips) / (total_slips.choose drawn_slips)

theorem probability_all_odd_is_correct : 
  probability_all_odd = 1 / 42 := by sorry

end probability_all_odd_is_correct_l676_67662


namespace quadratic_monotonic_condition_l676_67635

/-- A quadratic function f(x) = x^2 - 2mx + 3 is monotonic on [2, 3] if and only if m ∈ (-∞, 2] ∪ [3, +∞) -/
theorem quadratic_monotonic_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 2 3, Monotone (fun x => x^2 - 2*m*x + 3)) ↔ 
  (m ≤ 2 ∨ m ≥ 3) :=
sorry

end quadratic_monotonic_condition_l676_67635


namespace six_tangent_circles_l676_67624

-- Define the circles C₁ and C₂
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of being tangent
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

-- Define the problem setup
def problem_setup (C₁ C₂ : Circle) : Prop :=
  C₁.radius = 2 ∧
  C₂.radius = 2 ∧
  are_tangent C₁ C₂

-- Define a function to count tangent circles
def count_tangent_circles (C₁ C₂ : Circle) : ℕ :=
  sorry -- The actual counting logic would go here

-- The main theorem
theorem six_tangent_circles (C₁ C₂ : Circle) :
  problem_setup C₁ C₂ → count_tangent_circles C₁ C₂ = 6 :=
by sorry


end six_tangent_circles_l676_67624


namespace min_even_integers_l676_67674

theorem min_even_integers (x y z a b c m n o : ℤ) : 
  x + y + z = 30 →
  x + y + z + a + b + c = 55 →
  x + y + z + a + b + c + m + n + o = 88 →
  ∃ (count : ℕ), count = (if Even x then 1 else 0) + 
                         (if Even y then 1 else 0) + 
                         (if Even z then 1 else 0) + 
                         (if Even a then 1 else 0) + 
                         (if Even b then 1 else 0) + 
                         (if Even c then 1 else 0) + 
                         (if Even m then 1 else 0) + 
                         (if Even n then 1 else 0) + 
                         (if Even o then 1 else 0) ∧
  count ≥ 1 ∧
  ∀ (other_count : ℕ), other_count ≥ count →
    ∃ (x' y' z' a' b' c' m' n' o' : ℤ),
      x' + y' + z' = 30 ∧
      x' + y' + z' + a' + b' + c' = 55 ∧
      x' + y' + z' + a' + b' + c' + m' + n' + o' = 88 ∧
      other_count = (if Even x' then 1 else 0) + 
                    (if Even y' then 1 else 0) + 
                    (if Even z' then 1 else 0) + 
                    (if Even a' then 1 else 0) + 
                    (if Even b' then 1 else 0) + 
                    (if Even c' then 1 else 0) + 
                    (if Even m' then 1 else 0) + 
                    (if Even n' then 1 else 0) + 
                    (if Even o' then 1 else 0) :=
by
  sorry

end min_even_integers_l676_67674


namespace books_bought_l676_67653

theorem books_bought (initial_amount : ℕ) (cost_per_book : ℕ) (remaining_amount : ℕ) :
  initial_amount = 79 →
  cost_per_book = 7 →
  remaining_amount = 16 →
  (initial_amount - remaining_amount) / cost_per_book = 9 :=
by sorry

end books_bought_l676_67653


namespace total_class_time_l676_67699

def hours_per_day : ℝ := 8
def days_per_week : ℝ := 5
def math_percentage : ℝ := 0.25
def language_percentage : ℝ := 0.30
def science_percentage : ℝ := 0.20
def history_percentage : ℝ := 0.10

theorem total_class_time :
  let total_hours := hours_per_day * days_per_week
  let math_hours := total_hours * math_percentage
  let language_hours := total_hours * language_percentage
  let science_hours := total_hours * science_percentage
  let history_hours := total_hours * history_percentage
  math_hours + language_hours + science_hours + history_hours = 34 := by
  sorry

end total_class_time_l676_67699


namespace sum_of_fractions_simplest_form_l676_67683

theorem sum_of_fractions : 
  7 / 12 + 11 / 15 = 79 / 60 :=
by sorry

theorem simplest_form : 
  ∀ n m : ℕ, n ≠ 0 → m ≠ 0 → Nat.gcd n m = 1 → (n : ℚ) / m = 79 / 60 → n = 79 ∧ m = 60 :=
by sorry

end sum_of_fractions_simplest_form_l676_67683


namespace trigonometric_equation_solution_l676_67698

theorem trigonometric_equation_solution (z : ℂ) : 
  (Complex.sin z + Complex.sin (2 * z) + Complex.sin (3 * z) = 
   Complex.cos z + Complex.cos (2 * z) + Complex.cos (3 * z)) ↔ 
  (∃ (k : ℤ), z = (2 / 3 : ℂ) * π * (3 * k + 1) ∨ z = (2 / 3 : ℂ) * π * (3 * k - 1)) ∨
  (∃ (n : ℤ), z = (π / 8 : ℂ) * (4 * n + 1)) :=
sorry

end trigonometric_equation_solution_l676_67698


namespace parabola_vertex_and_a_range_l676_67669

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a^2 + 2*a

-- Define the line
def line (x : ℝ) : ℝ := 2*x - 2

-- Define the length of PQ
def PQ_length (a : ℝ) (m : ℝ) : ℝ := (m - (a + 1))^2 + 1

theorem parabola_vertex_and_a_range :
  (∀ x : ℝ, parabola 1 x ≥ 2) ∧
  (parabola 1 1 = 2) ∧
  (∀ a : ℝ, a > 0 → 
    (∀ m : ℝ, m < 3 → 
      (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < h → h < δ → 
        PQ_length a (m + h) < PQ_length a m
      ) → a ≥ 2
    )
  ) := by sorry

end parabola_vertex_and_a_range_l676_67669


namespace more_triangles_2003_l676_67607

/-- A triangle with integer sides -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of triangles with integer sides and perimeter 2000 -/
def Triangles2000 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 2000}

/-- The set of triangles with integer sides and perimeter 2003 -/
def Triangles2003 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 2003}

/-- Function that maps a triangle with perimeter 2000 to a triangle with perimeter 2003 -/
def f (t : IntTriangle) : IntTriangle :=
  ⟨t.a + 1, t.b + 1, t.c + 1, sorry⟩

theorem more_triangles_2003 :
  ∃ (g : Triangles2000 → Triangles2003), Function.Injective g ∧
  ∃ (t : Triangles2003), t ∉ Set.range g :=
sorry

end more_triangles_2003_l676_67607


namespace inequality_proof_l676_67649

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end inequality_proof_l676_67649


namespace interest_groups_participation_l676_67604

theorem interest_groups_participation (total_students : ℕ) (total_participants : ℕ) 
  (sports_and_literature : ℕ) (sports_and_math : ℕ) (literature_and_math : ℕ) (all_three : ℕ) :
  total_students = 120 →
  total_participants = 135 →
  sports_and_literature = 15 →
  sports_and_math = 10 →
  literature_and_math = 8 →
  all_three = 4 →
  total_students - (total_participants - sports_and_literature - sports_and_math - literature_and_math + all_three) = 14 :=
by sorry

end interest_groups_participation_l676_67604


namespace arithmetic_number_difference_l676_67620

-- Define a function to check if a number is a valid 3-digit arithmetic number
def isArithmeticNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧
                  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                  b - a = c - b

-- Define the largest and smallest arithmetic numbers
def largestArithmeticNumber : ℕ := 759
def smallestArithmeticNumber : ℕ := 123

-- State the theorem
theorem arithmetic_number_difference :
  isArithmeticNumber largestArithmeticNumber ∧
  isArithmeticNumber smallestArithmeticNumber ∧
  (∀ n : ℕ, isArithmeticNumber n → smallestArithmeticNumber ≤ n ∧ n ≤ largestArithmeticNumber) ∧
  largestArithmeticNumber - smallestArithmeticNumber = 636 := by
  sorry


end arithmetic_number_difference_l676_67620


namespace min_cos_sum_acute_angles_l676_67641

theorem min_cos_sum_acute_angles (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α + Real.tan β = 4 * Real.sin (α + β)) :
  Real.cos (α + β) ≥ -1/2 := by
  sorry

end min_cos_sum_acute_angles_l676_67641


namespace remainder_sum_of_powers_l676_67693

theorem remainder_sum_of_powers (n : ℕ) : (8^6 + 7^7 + 6^8) % 5 = 3 := by
  sorry

end remainder_sum_of_powers_l676_67693


namespace negation_equivalence_l676_67600

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, 2 * x^2 + x + m ≤ 0) ↔ (∀ x : ℤ, 2 * x^2 + x + m > 0) :=
by sorry

end negation_equivalence_l676_67600


namespace cube_diff_prime_mod_six_l676_67602

theorem cube_diff_prime_mod_six (a b p : ℕ) : 
  0 < a → 0 < b → Prime p → a^3 - b^3 = p → p % 6 = 1 := by
  sorry

end cube_diff_prime_mod_six_l676_67602


namespace solution_set_equivalence_l676_67603

open Real

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State the theorem
theorem solution_set_equivalence 
  (h1 : ∀ x, 3 * f x + f' x < 0)
  (h2 : f (log 2) = 1) :
  ∀ x, f x > 8 * exp (-3 * x) ↔ x < log 2 :=
by sorry

end solution_set_equivalence_l676_67603


namespace stating_parking_arrangement_count_l676_67626

/-- Represents the number of parking spaces -/
def num_spaces : ℕ := 8

/-- Represents the number of trucks -/
def num_trucks : ℕ := 2

/-- Represents the number of cars -/
def num_cars : ℕ := 2

/-- Represents the total number of vehicles -/
def total_vehicles : ℕ := num_trucks + num_cars

/-- 
Represents the number of ways to arrange trucks and cars in a row of parking spaces,
where vehicles of the same type must be adjacent.
-/
def parking_arrangements (spaces : ℕ) (trucks : ℕ) (cars : ℕ) : ℕ :=
  sorry

/-- 
Theorem stating that the number of ways to arrange 2 trucks and 2 cars
in a row of 8 parking spaces, where vehicles of the same type must be adjacent,
is equal to 120.
-/
theorem parking_arrangement_count :
  parking_arrangements num_spaces num_trucks num_cars = 120 := by
  sorry

end stating_parking_arrangement_count_l676_67626


namespace skittles_distribution_l676_67613

theorem skittles_distribution (num_friends : ℕ) (total_skittles : ℕ) 
  (h1 : num_friends = 5) 
  (h2 : total_skittles = 200) : 
  total_skittles / num_friends = 40 := by
  sorry

end skittles_distribution_l676_67613


namespace correct_average_after_error_correction_l676_67609

theorem correct_average_after_error_correction 
  (n : ℕ) 
  (incorrect_avg : ℚ) 
  (incorrect_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 16 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 55 →
  (n * incorrect_avg - incorrect_num + correct_num) / n = 19 := by
  sorry

end correct_average_after_error_correction_l676_67609


namespace warehouse_boxes_theorem_l676_67658

/-- The number of boxes in two warehouses -/
def total_boxes (first_warehouse : ℕ) (second_warehouse : ℕ) : ℕ :=
  first_warehouse + second_warehouse

theorem warehouse_boxes_theorem (first_warehouse second_warehouse : ℕ) 
  (h1 : first_warehouse = 400)
  (h2 : first_warehouse = 2 * second_warehouse) : 
  total_boxes first_warehouse second_warehouse = 600 := by
  sorry

end warehouse_boxes_theorem_l676_67658


namespace extended_triangle_PQ_length_l676_67608

/-- Triangle ABC with extended sides and intersection points -/
structure ExtendedTriangle where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Extended segments
  DA : ℝ
  BE : ℝ
  -- Intersection points with circumcircle of CDE
  PQ : ℝ

/-- Theorem stating the length of PQ in the given configuration -/
theorem extended_triangle_PQ_length 
  (triangle : ExtendedTriangle)
  (h1 : triangle.AB = 15)
  (h2 : triangle.BC = 18)
  (h3 : triangle.CA = 20)
  (h4 : triangle.DA = triangle.AB)
  (h5 : triangle.BE = triangle.AB)
  : triangle.PQ = 37 := by
  sorry

#check extended_triangle_PQ_length

end extended_triangle_PQ_length_l676_67608


namespace unique_base_eight_l676_67646

/-- Converts a list of digits in base b to its decimal representation -/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Checks if the equation 243₍ᵦ₎ + 152₍ᵦ₎ = 415₍ᵦ₎ holds for a given base b -/
def equationHolds (b : Nat) : Prop :=
  toDecimal [2, 4, 3] b + toDecimal [1, 5, 2] b = toDecimal [4, 1, 5] b

theorem unique_base_eight :
  ∃! b, b > 5 ∧ equationHolds b :=
sorry

end unique_base_eight_l676_67646


namespace choose_four_from_seven_l676_67638

theorem choose_four_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 4 → Nat.choose n k = 35 := by
  sorry

end choose_four_from_seven_l676_67638


namespace carbon_neutral_olympics_emissions_l676_67623

theorem carbon_neutral_olympics_emissions (emissions : ℝ) : 
  emissions = 320000 → emissions = 3.2 * (10 ^ 5) := by
  sorry

end carbon_neutral_olympics_emissions_l676_67623


namespace complex_square_root_of_negative_one_l676_67622

theorem complex_square_root_of_negative_one (z : ℂ) : 
  (z - 1)^2 = -1 → z = 1 + I ∨ z = 1 - I :=
by sorry

end complex_square_root_of_negative_one_l676_67622


namespace polynomial_equality_l676_67679

-- Define the polynomials
variable (x : ℝ)
def f (x : ℝ) : ℝ := x^3 - 3*x - 1
def h (x : ℝ) : ℝ := -x^3 + 5*x^2 + 3*x

-- State the theorem
theorem polynomial_equality :
  (∀ x, f x + h x = 5*x^2 - 1) ∧ 
  (∀ x, f x = x^3 - 3*x - 1) →
  (∀ x, h x = -x^3 + 5*x^2 + 3*x) :=
by
  sorry

end polynomial_equality_l676_67679


namespace same_terminal_side_angles_l676_67621

open Set Real

def isObtuse (α : ℝ) : Prop := π / 2 < α ∧ α < π

theorem same_terminal_side_angles
  (α : ℝ)
  (h_obtuse : isObtuse α)
  (h_sin : sin α = 1 / 2) :
  {β | ∃ k : ℤ, β = 5 * π / 6 + 2 * π * k} =
  {β | ∃ k : ℤ, β = α + 2 * π * k} :=
by sorry

end same_terminal_side_angles_l676_67621


namespace scooter_distance_l676_67696

/-- Proves that a scooter traveling 5/8 as fast as a motorcycle going 96 miles per hour will cover 40 miles in 40 minutes. -/
theorem scooter_distance (motorcycle_speed : ℝ) (scooter_ratio : ℝ) (travel_time : ℝ) :
  motorcycle_speed = 96 →
  scooter_ratio = 5/8 →
  travel_time = 40/60 →
  scooter_ratio * motorcycle_speed * travel_time = 40 :=
by sorry

end scooter_distance_l676_67696


namespace r_amount_l676_67632

theorem r_amount (total : ℝ) (p_q_amount : ℝ) (r_amount : ℝ) : 
  total = 6000 →
  r_amount = (2/3) * p_q_amount →
  total = p_q_amount + r_amount →
  r_amount = 2400 := by
sorry

end r_amount_l676_67632


namespace infinite_sqrt_twelve_l676_67691

theorem infinite_sqrt_twelve (x : ℝ) : x = Real.sqrt (12 + x) → x = 4 := by
  sorry

end infinite_sqrt_twelve_l676_67691


namespace tangent_lines_count_l676_67663

/-- The function f(x) = -x³ + 6x² - 9x + 8 -/
def f (x : ℝ) : ℝ := -x^3 + 6*x^2 - 9*x + 8

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -3*x^2 + 12*x - 9

/-- Condition for a point (x₀, f(x₀)) to be on a tangent line passing through (0, 0) -/
def is_tangent_point (x₀ : ℝ) : Prop :=
  f x₀ = (f' x₀) * x₀

theorem tangent_lines_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, is_tangent_point x) ∧ S.card = 2 :=
sorry

end tangent_lines_count_l676_67663


namespace root_transformation_l676_67667

theorem root_transformation {a₁ a₂ a₃ b c₁ c₂ c₃ : ℝ} 
  (h_distinct : c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃)
  (h_roots : ∀ x : ℝ, (x - a₁) * (x - a₂) * (x - a₃) = b ↔ x = c₁ ∨ x = c₂ ∨ x = c₃) :
  ∀ x : ℝ, (x + c₁) * (x + c₂) * (x + c₃) = b ↔ x = -a₁ ∨ x = -a₂ ∨ x = -a₃ := by
sorry

end root_transformation_l676_67667


namespace function_identically_zero_l676_67692

/-- A function satisfying the given conditions is identically zero. -/
theorem function_identically_zero (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_zero : f 0 = 0)
  (h_bound : ∀ x : ℝ, 0 < |f x| → |f x| < (1/2) → 
    |deriv f x| ≤ |f x * Real.log (|f x|)|) : 
  ∀ x : ℝ, f x = 0 := by
sorry

end function_identically_zero_l676_67692


namespace lemon_heads_distribution_l676_67684

theorem lemon_heads_distribution (total : Nat) (friends : Nat) (each : Nat) : 
  total = 72 → friends = 6 → total / friends = each → each = 12 := by sorry

end lemon_heads_distribution_l676_67684


namespace otimes_properties_l676_67629

def otimes (a b : ℝ) : ℝ := a * b + a + b

theorem otimes_properties :
  (∀ a b : ℝ, otimes a b = otimes b a) ∧
  (∃ a b c : ℝ, otimes (otimes a b) c ≠ otimes a (otimes b c)) ∧
  (∃ a b c : ℝ, otimes (a + b) c ≠ otimes a c + otimes b c) := by
  sorry

end otimes_properties_l676_67629


namespace percentage_men_science_majors_l676_67678

/-- Given a college class, proves that 28% of men are science majors -/
theorem percentage_men_science_majors 
  (women_science_major_ratio : Real) 
  (non_science_ratio : Real) 
  (men_ratio : Real) 
  (h1 : women_science_major_ratio = 0.2)
  (h2 : non_science_ratio = 0.6)
  (h3 : men_ratio = 0.4) :
  (1 - non_science_ratio - women_science_major_ratio * (1 - men_ratio)) / men_ratio = 0.28 := by
  sorry

end percentage_men_science_majors_l676_67678


namespace factorization_equality_l676_67639

/-- For all real numbers a and b, ab² - 2ab + a = a(b-1)² --/
theorem factorization_equality (a b : ℝ) : a * b^2 - 2 * a * b + a = a * (b - 1)^2 := by
  sorry

end factorization_equality_l676_67639


namespace tetrahedron_volume_is_ten_l676_67615

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  pq : ℝ
  pr : ℝ
  ps : ℝ
  qr : ℝ
  qs : ℝ
  rs : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem: The volume of a tetrahedron PQRS with given edge lengths is 10 -/
theorem tetrahedron_volume_is_ten :
  let t : Tetrahedron := {
    pq := 3,
    pr := 5,
    ps := 6,
    qr := 4,
    qs := Real.sqrt 26,
    rs := 5
  }
  tetrahedronVolume t = 10 := by
  sorry

end tetrahedron_volume_is_ten_l676_67615


namespace product_from_lcm_gcd_l676_67625

theorem product_from_lcm_gcd (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 48) 
  (h_gcd : Nat.gcd a b = 8) : 
  a * b = 384 := by
sorry

end product_from_lcm_gcd_l676_67625


namespace circle_area_ratio_l676_67606

/-- Given two circles R and S, if the diameter of R is 80% of the diameter of S,
    then the area of R is 64% of the area of S. -/
theorem circle_area_ratio (R S : Real) (hdiameter : R = 0.8 * S) :
  (π * (R / 2)^2) / (π * (S / 2)^2) = 0.64 := by
  sorry

end circle_area_ratio_l676_67606


namespace opposite_of_neg_five_l676_67647

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- Theorem: The opposite of -5 is 5. -/
theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l676_67647


namespace power_equality_natural_numbers_l676_67655

theorem power_equality_natural_numbers (a b : ℕ) :
  a ^ b = b ^ a ↔ (a = b) ∨ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2) := by
sorry

end power_equality_natural_numbers_l676_67655


namespace factorial_fraction_simplification_l676_67657

theorem factorial_fraction_simplification :
  (4 * Nat.factorial 6 + 20 * Nat.factorial 5 + 48 * Nat.factorial 4) / Nat.factorial 7 = 134 / 105 := by
  sorry

end factorial_fraction_simplification_l676_67657


namespace cosine_symmetry_l676_67675

/-- A function f is symmetric about the origin if f(-x) = -f(x) for all x -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem cosine_symmetry (φ : ℝ) :
  SymmetricAboutOrigin (fun x ↦ Real.cos (3 * x + φ)) →
  ¬ ∃ k : ℤ, φ = k * Real.pi := by
  sorry

end cosine_symmetry_l676_67675


namespace omega_properties_l676_67677

/-- The weight function ω(n) that returns the sum of binary digits of n -/
def ω (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 2 + ω (n / 2))

/-- Theorem stating the properties of the ω function -/
theorem omega_properties :
  ∀ n : ℕ,
  (ω (2 * n) = ω n) ∧
  (ω (8 * n + 5) = ω (4 * n + 3)) ∧
  (ω ((2 ^ n) - 1) = n) :=
by sorry

end omega_properties_l676_67677


namespace stating_children_count_l676_67671

/-- Represents the number of children in the problem -/
def num_children : ℕ := 6

/-- Represents the age of the youngest child -/
def youngest_age : ℕ := 7

/-- Represents the interval between children's ages -/
def age_interval : ℕ := 3

/-- Represents the sum of all children's ages -/
def total_age : ℕ := 65

/-- 
  Theorem stating that given the conditions of the problem,
  the number of children is 6
-/
theorem children_count : 
  (∃ (n : ℕ), 
    n * (2 * youngest_age + (n - 1) * age_interval) = 2 * total_age ∧
    n = num_children) :=
by sorry

end stating_children_count_l676_67671


namespace apec_photo_arrangements_l676_67665

def arrangement_count (n : ℕ) (k : ℕ) : ℕ := n.factorial

theorem apec_photo_arrangements :
  let total_leaders : ℕ := 21
  let front_row : ℕ := 11
  let back_row : ℕ := 10
  let fixed_positions : ℕ := 3
  let remaining_leaders : ℕ := total_leaders - fixed_positions
  let us_russia_arrangements : ℕ := arrangement_count 2 2
  let other_arrangements : ℕ := arrangement_count remaining_leaders remaining_leaders
  
  (us_russia_arrangements * other_arrangements : ℕ) = 
    arrangement_count 2 2 * arrangement_count 18 18 :=
by sorry

end apec_photo_arrangements_l676_67665
