import Mathlib

namespace cylinder_lateral_area_l3429_342905

/-- The lateral surface area of a cylinder with given circumference and height -/
def lateral_surface_area (circumference : ℝ) (height : ℝ) : ℝ :=
  circumference * height

/-- Theorem: The lateral surface area of a cylinder with circumference 5cm and height 2cm is 10 cm² -/
theorem cylinder_lateral_area :
  lateral_surface_area 5 2 = 10 := by
  sorry

end cylinder_lateral_area_l3429_342905


namespace point_symmetry_l3429_342991

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = p.y

theorem point_symmetry (M N P : Point) (hM : M = Point.mk (-4) 3)
    (hN : symmetricOrigin M N) (hP : symmetricYAxis N P) :
    P = Point.mk 4 3 := by
  sorry

end point_symmetry_l3429_342991


namespace city_distance_ratio_l3429_342968

/-- Prove that the ratio of distances between cities is 2:1 --/
theorem city_distance_ratio :
  ∀ (AB BC CD AD : ℝ),
  AB = 100 →
  BC = AB + 50 →
  AD = 550 →
  AD = AB + BC + CD →
  ∃ (k : ℝ), CD = k * BC →
  CD / BC = 2 := by
sorry

end city_distance_ratio_l3429_342968


namespace towels_per_load_l3429_342961

theorem towels_per_load (total_towels : ℕ) (num_loads : ℕ) (h1 : total_towels = 42) (h2 : num_loads = 6) :
  total_towels / num_loads = 7 := by
  sorry

end towels_per_load_l3429_342961


namespace diophantine_equation_solution_l3429_342927

def is_solution (x y z : ℕ+) : Prop :=
  x^4 + y^4 + z^4 = 2*x^2*y^2 + 2*y^2*z^2 + 2*z^2*x^2 - 63

def solution_set : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(1, 4, 4), (4, 1, 4), (4, 4, 1), (2, 2, 3), (2, 3, 2), (3, 2, 2)}

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+, is_solution x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end diophantine_equation_solution_l3429_342927


namespace basketball_game_scores_l3429_342994

/-- Represents the quarterly scores of a team --/
structure QuarterlyScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Check if a sequence of four numbers is arithmetic --/
def isArithmetic (s : QuarterlyScores) : Prop :=
  s.q2 - s.q1 = s.q3 - s.q2 ∧ s.q3 - s.q2 = s.q4 - s.q3

/-- Check if a sequence of four numbers is geometric --/
def isGeometric (s : QuarterlyScores) : Prop :=
  s.q1 > 0 ∧ s.q2 % s.q1 = 0 ∧ s.q3 % s.q2 = 0 ∧ s.q4 % s.q3 = 0 ∧
  s.q2 / s.q1 = s.q3 / s.q2 ∧ s.q3 / s.q2 = s.q4 / s.q3

/-- Sum of all quarterly scores --/
def totalScore (s : QuarterlyScores) : ℕ :=
  s.q1 + s.q2 + s.q3 + s.q4

theorem basketball_game_scores :
  ∃ (raiders wildcats : QuarterlyScores),
    -- Tied at halftime
    raiders.q1 + raiders.q2 = wildcats.q1 + wildcats.q2 ∧
    -- Raiders' scores form an arithmetic sequence
    isArithmetic raiders ∧
    -- Wildcats' scores form a geometric sequence
    isGeometric wildcats ∧
    -- Fourth quarter combined score is half of total combined score
    raiders.q4 + wildcats.q4 = (totalScore raiders + totalScore wildcats) / 2 ∧
    -- Neither team scored more than 100 points
    totalScore raiders ≤ 100 ∧ totalScore wildcats ≤ 100 ∧
    -- First quarter total is one of the given options
    (raiders.q1 + wildcats.q1 = 10 ∨
     raiders.q1 + wildcats.q1 = 15 ∨
     raiders.q1 + wildcats.q1 = 20 ∨
     raiders.q1 + wildcats.q1 = 9 ∨
     raiders.q1 + wildcats.q1 = 12) :=
by
  sorry

end basketball_game_scores_l3429_342994


namespace no_solution_for_diophantine_equation_l3429_342912

theorem no_solution_for_diophantine_equation :
  ¬ ∃ (m n : ℕ+), 5 * m.val^2 - 6 * m.val * n.val + 7 * n.val^2 = 2006 := by
  sorry

end no_solution_for_diophantine_equation_l3429_342912


namespace stock_price_change_l3429_342958

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_decrease := initial_price * (1 - 0.05)
  let final_price := price_after_decrease * (1 + 0.10)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = 4.5 := by
  sorry

end stock_price_change_l3429_342958


namespace triangle_inequality_l3429_342980

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) :
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7/3 := by
  sorry

end triangle_inequality_l3429_342980


namespace final_balance_approx_l3429_342931

/-- Calculates the final amount in Steve's bank account after 5 years --/
def bank_account_balance : ℝ := 
  let initial_deposit : ℝ := 100
  let interest_rate_1 : ℝ := 0.1  -- 10% for first 3 years
  let interest_rate_2 : ℝ := 0.08 -- 8% for next 2 years
  let deposit_1 : ℝ := 10 -- annual deposit for first 2 years
  let deposit_2 : ℝ := 15 -- annual deposit for remaining 3 years
  let year_1 : ℝ := initial_deposit * (1 + interest_rate_1) + deposit_1
  let year_2 : ℝ := year_1 * (1 + interest_rate_1) + deposit_1
  let year_3 : ℝ := year_2 * (1 + interest_rate_1) + deposit_2
  let year_4 : ℝ := year_3 * (1 + interest_rate_2) + deposit_2
  let year_5 : ℝ := year_4 * (1 + interest_rate_2) + deposit_2
  year_5

/-- The final balance in Steve's bank account after 5 years is approximately $230.89 --/
theorem final_balance_approx : 
  ∃ ε > 0, |bank_account_balance - 230.89| < ε :=
by
  sorry

end final_balance_approx_l3429_342931


namespace expression_simplification_l3429_342915

theorem expression_simplification (x y : ℝ) (h : |x - 2| + (y + 1)^2 = 0) :
  3*x - 2*(x^2 - 1/2*y^2) + (x - 1/2*y^2) = 1/2 := by
  sorry

end expression_simplification_l3429_342915


namespace value_at_2023_l3429_342959

/-- An even function satisfying the given functional equation -/
def EvenFunctionWithProperty (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f (x + 2) = 3 - Real.sqrt (6 * f x - f x ^ 2))

/-- The main theorem stating the value of f(2023) -/
theorem value_at_2023 (f : ℝ → ℝ) (h : EvenFunctionWithProperty f) : 
  f 2023 = 3 - (3 / 2) * Real.sqrt 2 := by
  sorry

end value_at_2023_l3429_342959


namespace expression_factorization_l3429_342901

theorem expression_factorization (x : ℝ) :
  (20 * x^3 + 100 * x - 10) - (-5 * x^3 + 5 * x - 10) = 5 * x * (5 * x^2 + 19) := by
  sorry

end expression_factorization_l3429_342901


namespace unique_factorial_sum_power_l3429_342917

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def factorial_sum (m : ℕ) : ℕ := (List.range m).map factorial |>.sum

theorem unique_factorial_sum_power (m n k : ℕ) :
  m > 1 ∧ n^k > 1 ∧ factorial_sum m = n^k → m = 3 ∧ n = 3 ∧ k = 2 := by sorry

end unique_factorial_sum_power_l3429_342917


namespace area_scaled_and_shifted_l3429_342998

-- Define a function g: ℝ → ℝ
variable (g : ℝ → ℝ)

-- Define the area between a function and the x-axis
def area_between_curve_and_axis (f : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem area_scaled_and_shifted (h : area_between_curve_and_axis g = 15) :
  area_between_curve_and_axis (fun x ↦ 4 * g (x + 3)) = 60 := by
  sorry

end area_scaled_and_shifted_l3429_342998


namespace matching_allocation_theorem_l3429_342999

/-- Represents the allocation of workers to produce parts A and B -/
structure WorkerAllocation where
  partA : ℕ
  partB : ℕ

/-- Checks if the given allocation produces matching sets of parts A and B -/
def isMatchingAllocation (totalWorkers : ℕ) (prodRateA : ℕ) (prodRateB : ℕ) (allocation : WorkerAllocation) : Prop :=
  allocation.partA + allocation.partB = totalWorkers ∧
  prodRateB * allocation.partB = 2 * (prodRateA * allocation.partA)

/-- Theorem stating that the given allocation produces matching sets -/
theorem matching_allocation_theorem :
  let totalWorkers : ℕ := 50
  let prodRateA : ℕ := 40
  let prodRateB : ℕ := 120
  let allocation : WorkerAllocation := ⟨30, 20⟩
  isMatchingAllocation totalWorkers prodRateA prodRateB allocation := by
  sorry

#check matching_allocation_theorem

end matching_allocation_theorem_l3429_342999


namespace snow_probability_first_week_february_l3429_342945

theorem snow_probability_first_week_february : 
  let prob_snow_first_three_days : ℚ := 1/4
  let prob_snow_next_four_days : ℚ := 1/3
  let days_in_week : ℕ := 7
  let first_period : ℕ := 3
  let second_period : ℕ := 4
  
  first_period + second_period = days_in_week →
  
  (1 - (1 - prob_snow_first_three_days)^first_period * 
       (1 - prob_snow_next_four_days)^second_period) = 11/12 := by
  sorry

end snow_probability_first_week_february_l3429_342945


namespace trig_inequality_l3429_342911

theorem trig_inequality : 2 * Real.sin (160 * π / 180) < Real.tan (50 * π / 180) ∧
                          Real.tan (50 * π / 180) < 1 + Real.cos (20 * π / 180) := by
  sorry

end trig_inequality_l3429_342911


namespace x_range_l3429_342972

theorem x_range (x : ℝ) (h1 : 1 / x < 4) (h2 : 1 / x > -2) : x > -1/2 := by
  sorry

end x_range_l3429_342972


namespace binomial_plus_ten_l3429_342938

theorem binomial_plus_ten : (Nat.choose 15 12) + 10 = 465 := by
  sorry

end binomial_plus_ten_l3429_342938


namespace calculate_overall_profit_specific_profit_l3429_342941

/-- Calculate the overall profit or loss from selling a refrigerator and a mobile phone -/
theorem calculate_overall_profit (refrigerator_cost mobile_cost : ℝ) 
  (refrigerator_loss_percent mobile_profit_percent : ℝ) : ℝ :=
  let refrigerator_loss := refrigerator_cost * (refrigerator_loss_percent / 100)
  let refrigerator_sell := refrigerator_cost - refrigerator_loss
  let mobile_profit := mobile_cost * (mobile_profit_percent / 100)
  let mobile_sell := mobile_cost + mobile_profit
  let total_cost := refrigerator_cost + mobile_cost
  let total_sell := refrigerator_sell + mobile_sell
  total_sell - total_cost

/-- Prove that the overall profit is 120 Rs given the specific conditions -/
theorem specific_profit : calculate_overall_profit 15000 8000 4 9 = 120 := by
  sorry

end calculate_overall_profit_specific_profit_l3429_342941


namespace necessary_not_sufficient_condition_for_purely_imaginary_l3429_342985

-- Define a complex number
def complex (a b : ℝ) := a + b * Complex.I

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem necessary_not_sufficient_condition_for_purely_imaginary (a b : ℝ) :
  (isPurelyImaginary (complex a b) → a = 0) ∧
  ¬(a = 0 → isPurelyImaginary (complex a b)) := by
  sorry

end necessary_not_sufficient_condition_for_purely_imaginary_l3429_342985


namespace probability_at_least_one_female_l3429_342974

def total_students : ℕ := 5
def male_students : ℕ := 3
def female_students : ℕ := 2
def selected_students : ℕ := 3

theorem probability_at_least_one_female :
  let total_combinations := Nat.choose total_students selected_students
  let all_male_combinations := Nat.choose male_students selected_students
  (1 : ℚ) - (all_male_combinations : ℚ) / (total_combinations : ℚ) = 9 / 10 := by
  sorry

end probability_at_least_one_female_l3429_342974


namespace new_boarders_count_l3429_342922

/-- The number of new boarders that joined the school -/
def new_boarders : ℕ := 30

/-- The initial number of boarders -/
def initial_boarders : ℕ := 150

/-- The initial ratio of boarders to day students -/
def initial_ratio : ℚ := 5 / 12

/-- The final ratio of boarders to day students -/
def final_ratio : ℚ := 1 / 2

theorem new_boarders_count :
  ∃ (initial_day_students : ℕ),
    (initial_boarders : ℚ) / initial_day_students = initial_ratio ∧
    (initial_boarders + new_boarders : ℚ) / initial_day_students = final_ratio :=
by sorry

end new_boarders_count_l3429_342922


namespace teds_fruit_purchase_cost_l3429_342900

/-- The total cost of purchasing fruits given their quantities and unit prices -/
def total_cost (banana_qty : ℕ) (orange_qty : ℕ) (apple_qty : ℕ) (grape_qty : ℕ)
                (banana_price : ℚ) (orange_price : ℚ) (apple_price : ℚ) (grape_price : ℚ) : ℚ :=
  banana_qty * banana_price + orange_qty * orange_price + 
  apple_qty * apple_price + grape_qty * grape_price

/-- Theorem stating that the total cost of Ted's fruit purchase is $47 -/
theorem teds_fruit_purchase_cost : 
  total_cost 7 15 6 4 2 1.5 1.25 0.75 = 47 := by
  sorry

end teds_fruit_purchase_cost_l3429_342900


namespace polynomial_division_theorem_l3429_342989

theorem polynomial_division_theorem :
  let p (z : ℝ) := 4 * z^3 - 8 * z^2 + 9 * z - 7
  let d (z : ℝ) := 4 * z + 2
  let q (z : ℝ) := z^2 - 2.5 * z + 3.5
  let r : ℝ := -14
  ∀ z : ℝ, p z = d z * q z + r := by
sorry

end polynomial_division_theorem_l3429_342989


namespace dress_difference_l3429_342979

theorem dress_difference (total_dresses : ℕ) (ana_dresses : ℕ) 
  (h1 : total_dresses = 48) 
  (h2 : ana_dresses = 15) : 
  total_dresses - ana_dresses - ana_dresses = 18 := by
  sorry

end dress_difference_l3429_342979


namespace more_girls_than_boys_l3429_342935

theorem more_girls_than_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 35 → 
  3 * girls = 4 * boys → 
  total = boys + girls → 
  girls - boys = 5 :=
by
  sorry

end more_girls_than_boys_l3429_342935


namespace prime_iff_factorial_congruence_l3429_342977

theorem prime_iff_factorial_congruence (p : ℕ) (hp : p > 1) : 
  Nat.Prime p ↔ (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end prime_iff_factorial_congruence_l3429_342977


namespace john_park_distance_l3429_342923

/-- John's journey to the park -/
theorem john_park_distance (speed : ℝ) (time_minutes : ℝ) (h1 : speed = 9) (h2 : time_minutes = 2) :
  speed * (time_minutes / 60) = 0.3 := by
  sorry

end john_park_distance_l3429_342923


namespace divisibility_by_six_l3429_342940

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_875a (a : ℕ) : ℕ := 8750 + a

theorem divisibility_by_six (a : ℕ) (h : is_single_digit a) : 
  (number_875a a) % 6 = 0 ↔ a = 4 := by
sorry

end divisibility_by_six_l3429_342940


namespace smallest_positive_integer_satisfying_inequality_l3429_342993

theorem smallest_positive_integer_satisfying_inequality :
  ∀ x : ℕ, x > 0 → (x + 3 < 2 * x - 7) → x ≥ 11 ∧
  (11 + 3 < 2 * 11 - 7) :=
sorry

end smallest_positive_integer_satisfying_inequality_l3429_342993


namespace ramanujan_number_l3429_342918

/-- Given Hardy's complex number and the product of Hardy's and Ramanujan's numbers,
    prove that Ramanujan's number is 144/25 + 8/25i. -/
theorem ramanujan_number (h r : ℂ) : 
  h = 3 + 4*I ∧ r * h = 16 + 24*I → r = 144/25 + 8/25*I := by
  sorry

end ramanujan_number_l3429_342918


namespace quadratic_function_determination_l3429_342949

theorem quadratic_function_determination (a b c : ℝ) (h_a : a > 0) : 
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, a * x + b ≤ 2) →
  (∃ x : ℝ, a * x + b = 2) →
  (a * x^2 + b * x + c = 2 * x^2 - 1) :=
by sorry

end quadratic_function_determination_l3429_342949


namespace largest_of_four_consecutive_evens_l3429_342952

theorem largest_of_four_consecutive_evens (a b c d : ℤ) : 
  (∀ n : ℤ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6) →
  a + b + c + d = 140 →
  d = 38 := by
sorry

end largest_of_four_consecutive_evens_l3429_342952


namespace min_guests_at_banquet_l3429_342992

/-- The minimum number of guests at a football banquet given the total food consumed and maximum consumption per guest -/
theorem min_guests_at_banquet (total_food : ℝ) (max_per_guest : ℝ) (h1 : total_food = 319) (h2 : max_per_guest = 2.0) : ℕ := by
  sorry

end min_guests_at_banquet_l3429_342992


namespace unique_line_intersection_l3429_342964

theorem unique_line_intersection (m b : ℝ) : 
  (∃! k, ∃ y₁ y₂, y₁ = k^2 + 4*k + 4 ∧ y₂ = m*k + b ∧ |y₁ - y₂| = 4) ∧
  (m * 2 + b = 8) ∧
  (b ≠ 0) →
  m = 12 ∧ b = -16 := by sorry

end unique_line_intersection_l3429_342964


namespace parabola_vertex_l3429_342920

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = (x - 2)^2 + 1

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 1)

/-- Theorem: The vertex of the parabola y = (x-2)^2 + 1 is (2, 1) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
by sorry

end parabola_vertex_l3429_342920


namespace x_geq_1_necessary_not_sufficient_for_lg_x_geq_1_l3429_342910

-- Define the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem x_geq_1_necessary_not_sufficient_for_lg_x_geq_1 :
  (∀ x : ℝ, lg x ≥ 1 → x ≥ 1) ∧
  ¬(∀ x : ℝ, x ≥ 1 → lg x ≥ 1) :=
by sorry

end x_geq_1_necessary_not_sufficient_for_lg_x_geq_1_l3429_342910


namespace rectangular_plot_length_l3429_342936

/-- The length of a rectangular plot in meters -/
def length : ℝ := 55

/-- The breadth of a rectangular plot in meters -/
def breadth : ℝ := 45

/-- The cost of fencing per meter in rupees -/
def cost_per_meter : ℝ := 26.50

/-- The total cost of fencing in rupees -/
def total_cost : ℝ := 5300

/-- Theorem stating the length of the rectangular plot -/
theorem rectangular_plot_length :
  (length = breadth + 10) ∧
  (total_cost = cost_per_meter * (2 * (length + breadth))) →
  length = 55 := by
  sorry

end rectangular_plot_length_l3429_342936


namespace gus_egg_consumption_l3429_342973

/-- The number of eggs Gus ate for breakfast -/
def breakfast_eggs : ℕ := 2

/-- The number of eggs Gus ate for lunch -/
def lunch_eggs : ℕ := 3

/-- The number of eggs Gus ate for dinner -/
def dinner_eggs : ℕ := 1

/-- The total number of eggs Gus ate -/
def total_eggs : ℕ := breakfast_eggs + lunch_eggs + dinner_eggs

theorem gus_egg_consumption : total_eggs = 6 := by
  sorry

end gus_egg_consumption_l3429_342973


namespace sin_4theta_from_exp_itheta_l3429_342951

theorem sin_4theta_from_exp_itheta (θ : ℝ) :
  Complex.exp (Complex.I * θ) = (4 + Complex.I * Real.sqrt 7) / 5 →
  Real.sin (4 * θ) = 144 * Real.sqrt 7 / 625 := by
  sorry

end sin_4theta_from_exp_itheta_l3429_342951


namespace arithmetic_progression_difference_l3429_342984

/-- An arithmetic progression with first term a₁, last term aₙ, common difference d, and sum Sₙ. -/
structure ArithmeticProgression (α : Type*) [Field α] where
  a₁ : α
  aₙ : α
  d : α
  n : ℕ
  Sₙ : α
  h₁ : aₙ = a₁ + (n - 1) * d
  h₂ : Sₙ = n / 2 * (a₁ + aₙ)

/-- The common difference of an arithmetic progression can be expressed in terms of its first term, 
last term, and sum. -/
theorem arithmetic_progression_difference (α : Type*) [Field α] (ap : ArithmeticProgression α) :
  ap.d = (ap.aₙ^2 - ap.a₁^2) / (2 * ap.Sₙ - (ap.a₁ + ap.aₙ)) := by
  sorry

end arithmetic_progression_difference_l3429_342984


namespace rectangles_on_specific_grid_l3429_342921

/-- Represents a grid with specified dimensions and properties. -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (unit_distance : ℕ)
  (allow_diagonals : Bool)

/-- Counts the number of rectangles that can be formed on the grid. -/
def count_rectangles (g : Grid) : ℕ := sorry

/-- The specific 3x3 grid with 2-unit spacing and allowed diagonals. -/
def specific_grid : Grid :=
  { rows := 3
  , cols := 3
  , unit_distance := 2
  , allow_diagonals := true }

/-- Theorem stating that the number of rectangles on the specific grid is 60. -/
theorem rectangles_on_specific_grid :
  count_rectangles specific_grid = 60 := by sorry

end rectangles_on_specific_grid_l3429_342921


namespace morning_routine_duration_l3429_342930

def coffee_bagel_time : ℕ := 15

def paper_eating_time : ℕ := 2 * coffee_bagel_time

def total_routine_time : ℕ := coffee_bagel_time + paper_eating_time

theorem morning_routine_duration :
  total_routine_time = 45 :=
by sorry

end morning_routine_duration_l3429_342930


namespace parallelogram_vertex_sum_l3429_342967

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The sum of coordinates of a point -/
def sumCoordinates (p : Point) : ℝ := p.x + p.y

/-- Theorem: Sum of coordinates of vertex C in the given parallelogram is 7 -/
theorem parallelogram_vertex_sum : 
  ∀ (ABCD : Parallelogram),
    ABCD.A = ⟨2, 3⟩ →
    ABCD.B = ⟨-1, 0⟩ →
    ABCD.D = ⟨5, -4⟩ →
    sumCoordinates ABCD.C = 7 := by
  sorry

end parallelogram_vertex_sum_l3429_342967


namespace num_distinct_representations_eq_six_l3429_342933

/-- Represents a digit configuration using matchsticks -/
def DigitConfig := Nat

/-- The maximum number of matchsticks in the original configuration -/
def max_sticks : Nat := 7

/-- The set of all possible digit configurations -/
def all_configs : Finset DigitConfig := sorry

/-- The number of distinct digit representations -/
def num_distinct_representations : Nat := Finset.card all_configs

/-- Theorem stating that the number of distinct representations is 6 -/
theorem num_distinct_representations_eq_six :
  num_distinct_representations = 6 := by sorry

end num_distinct_representations_eq_six_l3429_342933


namespace characterization_theorem_l3429_342987

/-- A function that checks if a number satisfies the given conditions -/
def satisfies_condition (n : ℕ) : Prop :=
  ∃ a b : ℕ, 
    n ≥ 2 ∧
    n = a^2 + b^2 ∧
    a > 1 ∧
    a ∣ n ∧
    b ∣ n ∧
    ∀ d : ℕ, d > 1 → d ∣ n → d ≥ a

/-- The main theorem stating the characterization of numbers satisfying the condition -/
theorem characterization_theorem :
  ∀ n : ℕ, satisfies_condition n ↔ 
    (n = 4) ∨ 
    (∃ k j : ℕ, k ≥ 2 ∧ j ≥ 1 ∧ j ≤ k ∧ n = 2^k * (2^(k*(j-1)) + 1)) :=
by sorry

end characterization_theorem_l3429_342987


namespace square_root_of_nine_l3429_342926

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end square_root_of_nine_l3429_342926


namespace distance_between_centers_l3429_342956

-- Define a circle in the first quadrant tangent to both axes
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = center.2
  in_first_quadrant : center.1 > 0 ∧ center.2 > 0

-- Define the property of passing through (4,1)
def passes_through_point (c : Circle) : Prop :=
  (c.center.1 - 4)^2 + (c.center.2 - 1)^2 = c.radius^2

-- Theorem statement
theorem distance_between_centers (c1 c2 : Circle)
  (h1 : passes_through_point c1)
  (h2 : passes_through_point c2)
  (h3 : c1 ≠ c2) :
  Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = 8 :=
sorry

end distance_between_centers_l3429_342956


namespace solution_set_of_inequality_l3429_342983

theorem solution_set_of_inequality (x : ℝ) : 
  (x-1)/(x^2-x-6) ≥ 0 ↔ x ∈ Set.Ioc (-2) 1 ∪ Set.Ioi 3 :=
sorry

end solution_set_of_inequality_l3429_342983


namespace balloon_difference_l3429_342981

theorem balloon_difference (your_balloons friend_balloons : ℕ) 
  (h1 : your_balloons = 7) 
  (h2 : friend_balloons = 5) : 
  your_balloons - friend_balloons = 2 := by
  sorry

end balloon_difference_l3429_342981


namespace triangle_perimeter_l3429_342995

theorem triangle_perimeter (AB AC : ℝ) (h_right_angle : AB ^ 2 + AC ^ 2 = (AB + AC + Real.sqrt (AB ^ 2 + AC ^ 2)) ^ 2 - 2 * AB * AC) (h_AB : AB = 8) (h_AC : AC = 15) :
  AB + AC + Real.sqrt (AB ^ 2 + AC ^ 2) = 40 := by
  sorry

end triangle_perimeter_l3429_342995


namespace hexagonal_pyramid_surface_area_l3429_342908

/-- The total surface area of a right pyramid with a regular hexagonal base -/
theorem hexagonal_pyramid_surface_area 
  (base_edge : ℝ) 
  (slant_height : ℝ) 
  (h : base_edge = 8) 
  (k : slant_height = 10) : 
  ∃ (area : ℝ), area = 48 * Real.sqrt 21 := by
  sorry

end hexagonal_pyramid_surface_area_l3429_342908


namespace problem_statement_l3429_342939

theorem problem_statement (x y : ℝ) (h1 : x + 3*y = 5) (h2 : 2*x - y = 2) :
  2*x^2 + 5*x*y - 3*y^2 = 10 := by
sorry

end problem_statement_l3429_342939


namespace syllogism_arrangement_correct_l3429_342907

-- Define the statements
def statement1 : Prop := 2012 % 2 = 0
def statement2 : Prop := ∀ n : ℕ, Even n → n % 2 = 0
def statement3 : Prop := Even 2012

-- Define the syllogism structure
inductive SyllogismStep
| MajorPremise
| MinorPremise
| Conclusion

-- Define a function to represent the correct arrangement
def correctArrangement : List (SyllogismStep × Prop) :=
  [(SyllogismStep.MajorPremise, statement2),
   (SyllogismStep.MinorPremise, statement3),
   (SyllogismStep.Conclusion, statement1)]

-- Theorem to prove
theorem syllogism_arrangement_correct :
  correctArrangement = 
    [(SyllogismStep.MajorPremise, statement2),
     (SyllogismStep.MinorPremise, statement3),
     (SyllogismStep.Conclusion, statement1)] :=
by sorry

end syllogism_arrangement_correct_l3429_342907


namespace original_equals_scientific_l3429_342916

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 75500000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation := {
  coefficient := 7.55
  exponent := 7
  property := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end original_equals_scientific_l3429_342916


namespace boards_per_package_not_integer_l3429_342942

theorem boards_per_package_not_integer (total_boards : ℕ) (num_packages : ℕ) 
  (h1 : total_boards = 154) (h2 : num_packages = 52) : 
  ¬ ∃ (n : ℕ), (total_boards : ℚ) / (num_packages : ℚ) = n := by
  sorry

end boards_per_package_not_integer_l3429_342942


namespace distance_travelled_l3429_342953

theorem distance_travelled (normal_speed : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) :
  normal_speed = 10 →
  faster_speed = 14 →
  additional_distance = 20 →
  (∃ (actual_distance : ℝ), 
    actual_distance / normal_speed = (actual_distance + additional_distance) / faster_speed ∧
    actual_distance = 50) :=
by sorry

end distance_travelled_l3429_342953


namespace bob_homework_time_l3429_342934

theorem bob_homework_time (alice_time bob_time : ℕ) : 
  alice_time = 40 → bob_time = (3 * alice_time) / 8 → bob_time = 15 := by
  sorry

end bob_homework_time_l3429_342934


namespace orchard_problem_l3429_342914

theorem orchard_problem (total_trees : ℕ) (pure_fuji : ℕ) (pure_gala : ℕ) :
  (pure_fuji : ℚ) = 3 / 4 * total_trees →
  (pure_fuji : ℚ) + 1 / 10 * total_trees = 221 →
  pure_gala = 39 := by
  sorry

end orchard_problem_l3429_342914


namespace automotive_test_distance_l3429_342902

/-- Calculates the total distance driven in an automotive test -/
theorem automotive_test_distance (d : ℝ) (t : ℝ) : 
  t = d / 4 + d / 5 + d / 6 ∧ t = 37 → 3 * d = 180 := by
  sorry

#check automotive_test_distance

end automotive_test_distance_l3429_342902


namespace double_seven_eighths_of_48_l3429_342978

theorem double_seven_eighths_of_48 : 2 * (7 / 8 * 48) = 84 := by
  sorry

end double_seven_eighths_of_48_l3429_342978


namespace perpendicular_lines_m_values_l3429_342976

theorem perpendicular_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, x + 2 * m * y - 1 = 0 ∧ (3 * m - 1) * x - m * y - 1 = 0 →
    ((-1 / (2 * m)) * ((3 * m - 1) / m) = -1)) →
  m = 1 ∨ m = 1 / 2 :=
by sorry

end perpendicular_lines_m_values_l3429_342976


namespace additive_inverse_of_zero_l3429_342919

theorem additive_inverse_of_zero : 
  (∀ x : ℝ, x + 0 = x) → 
  (∀ x : ℝ, x + (-x) = 0) → 
  (0 : ℝ) + 0 = 0 := by
  sorry

end additive_inverse_of_zero_l3429_342919


namespace polygon_internal_angle_sum_l3429_342924

theorem polygon_internal_angle_sum (n : ℕ) (h : n > 2) :
  let external_angle : ℚ := 40
  let internal_angle_sum : ℚ := (n - 2) * 180
  external_angle * n = 360 → internal_angle_sum = 1260 := by
  sorry

end polygon_internal_angle_sum_l3429_342924


namespace largest_integer_with_remainder_l3429_342966

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n → n = 93 :=
by sorry

end largest_integer_with_remainder_l3429_342966


namespace max_value_polynomial_l3429_342971

/-- Given real numbers x and y such that x + y = 5, 
    the maximum value of x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 is 30625/44 -/
theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (z w : ℝ), z + w = 5 ∧ 
    ∀ (a b : ℝ), a + b = 5 → 
      z^5*w + z^4*w^2 + z^3*w^3 + z^2*w^4 + z*w^5 ≥ a^5*b + a^4*b^2 + a^3*b^3 + a^2*b^4 + a*b^5) ∧
  (∀ (a b : ℝ), a + b = 5 → 
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 ≤ 30625/44) :=
by sorry

end max_value_polynomial_l3429_342971


namespace flower_shop_expenses_l3429_342950

/-- Calculates the weekly expenses for running a flower shop --/
theorem flower_shop_expenses 
  (rent : ℝ) 
  (utility_rate : ℝ) 
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (employees_per_shift : ℕ) 
  (hourly_wage : ℝ) 
  (h_rent : rent = 1200) 
  (h_utility : utility_rate = 0.2) 
  (h_hours : hours_per_day = 16) 
  (h_days : days_per_week = 5) 
  (h_employees : employees_per_shift = 2) 
  (h_wage : hourly_wage = 12.5) : 
  rent + rent * utility_rate + 
  (↑hours_per_day * ↑days_per_week * ↑employees_per_shift * hourly_wage) = 3440 := by
  sorry

#check flower_shop_expenses

end flower_shop_expenses_l3429_342950


namespace geometric_sequence_property_l3429_342929

/-- A sequence a, b, c forms a geometric sequence if there exists a non-zero real number r such that b = ar and c = br -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r

theorem geometric_sequence_property (a b c : ℝ) :
  (IsGeometricSequence a b c → b^2 = a * c) ∧
  ¬(b^2 = a * c → IsGeometricSequence a b c) :=
sorry

end geometric_sequence_property_l3429_342929


namespace midpoint_coordinate_sum_l3429_342913

/-- Given a triangle in the Cartesian plane with vertices (a, d), (b, e), and (c, f),
    if the sum of x-coordinates (a + b + c) is 15 and the sum of y-coordinates (d + e + f) is 12,
    then the sum of x-coordinates of the midpoints of its sides is 15 and
    the sum of y-coordinates of the midpoints of its sides is 12. -/
theorem midpoint_coordinate_sum (a b c d e f : ℝ) 
  (h1 : a + b + c = 15) (h2 : d + e + f = 12) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 ∧ 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 12 := by
  sorry


end midpoint_coordinate_sum_l3429_342913


namespace parallel_vectors_x_value_l3429_342944

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  are_parallel a b → x = -2 :=
by
  sorry

end parallel_vectors_x_value_l3429_342944


namespace circle_division_theorem_l3429_342928

/-- The number of regions a circle is divided into by radii and concentric circles -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

theorem circle_division_theorem :
  num_regions 16 10 = 176 := by
  sorry

end circle_division_theorem_l3429_342928


namespace wall_width_l3429_342990

theorem wall_width (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  l = 7 * h →
  volume = w * h * l →
  volume = 6804 →
  w = 3 := by
sorry

end wall_width_l3429_342990


namespace find_a_l3429_342909

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {3, 7, a^2 - 2*a - 3}

-- Define set A
def A (a : ℝ) : Set ℝ := {7, |a - 7|}

-- Define the complement of A in U
def complement_A (a : ℝ) : Set ℝ := U a \ A a

-- Theorem statement
theorem find_a : ∃ (a : ℝ), 
  (U a = {3, 7, a^2 - 2*a - 3}) ∧ 
  (A a = {7, |a - 7|}) ∧ 
  (complement_A a = {5}) → 
  a = 4 := by
sorry

end find_a_l3429_342909


namespace course_selection_theorem_l3429_342969

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def total_courses : ℕ := physical_education_courses + art_courses

def choose (n k : ℕ) : ℕ := Nat.choose n k

def two_course_selections : ℕ := choose physical_education_courses 1 * choose art_courses 1

def three_course_selections : ℕ := 
  choose physical_education_courses 2 * choose art_courses 1 + 
  choose physical_education_courses 1 * choose art_courses 2

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_theorem : total_selections = 64 := by
  sorry

end course_selection_theorem_l3429_342969


namespace intersection_and_union_when_a_is_two_union_with_complement_equals_reals_iff_l3429_342963

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x < a}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_two :
  (A 2 ∩ B = {x : ℝ | 1 < x ∧ x < 2}) ∧
  (A 2 ∪ B = {x : ℝ | x < 3}) := by
sorry

-- Theorem for part (2)
theorem union_with_complement_equals_reals_iff (a : ℝ) :
  (A a ∪ (Set.univ \ B) = Set.univ) ↔ a ≥ 3 := by
sorry

end intersection_and_union_when_a_is_two_union_with_complement_equals_reals_iff_l3429_342963


namespace max_area_rectangular_play_area_l3429_342997

/-- 
Given a rectangular area with perimeter P (excluding one side) and length l and width w,
prove that the maximum area A is achieved when l = P/2 and w = P/6, resulting in A = (P^2)/48.
-/
theorem max_area_rectangular_play_area (P : ℝ) (h : P > 0) :
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ l + 2*w = P ∧
  ∀ (l' w' : ℝ), l' > 0 → w' > 0 → l' + 2*w' = P →
  l * w ≥ l' * w' ∧
  l = P/2 ∧ w = P/6 ∧ l * w = (P^2)/48 :=
by sorry

end max_area_rectangular_play_area_l3429_342997


namespace maple_trees_remaining_l3429_342957

theorem maple_trees_remaining (initial_maples : Real) (cut_maples : Real) (remaining_maples : Real) : 
  initial_maples = 9.0 → cut_maples = 2.0 → remaining_maples = initial_maples - cut_maples → remaining_maples = 7.0 := by
  sorry

end maple_trees_remaining_l3429_342957


namespace geometric_seq_increasing_condition_l3429_342937

/-- A sequence is geometric if there exists a constant r such that aₙ₊₁ = r * aₙ for all n. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence is increasing if aₙ₊₁ > aₙ for all n. -/
def IsIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem geometric_seq_increasing_condition (a : ℕ → ℝ) (h : IsGeometric a) :
  (IsIncreasing a → a 2 > a 1) ∧ ¬(a 2 > a 1 → IsIncreasing a) :=
by sorry

end geometric_seq_increasing_condition_l3429_342937


namespace monotonic_decreasing_interval_l3429_342986

def f (x : ℝ) := x^3 - x^2 - x

def f_derivative (x : ℝ) := 3*x^2 - 2*x - 1

theorem monotonic_decreasing_interval :
  {x : ℝ | f_derivative x < 0} = {x : ℝ | -1/3 < x ∧ x < 1} := by sorry

end monotonic_decreasing_interval_l3429_342986


namespace shortest_distance_between_circles_l3429_342954

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : 
  let circle1 := {(x, y) : ℝ × ℝ | x^2 - 6*x + y^2 + 2*y - 11 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + 10*x + y^2 - 8*y + 25 = 0}
  (shortest_distance : ℝ) →
  shortest_distance = Real.sqrt 89 - Real.sqrt 21 - 4 ∧
  ∀ (p1 : ℝ × ℝ) (p2 : ℝ × ℝ), 
    p1 ∈ circle1 → p2 ∈ circle2 → 
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) ≥ shortest_distance :=
by
  sorry


end shortest_distance_between_circles_l3429_342954


namespace prime_factorization_problem_l3429_342962

theorem prime_factorization_problem :
  2006^2 * 2262 - 669^2 * 3599 + 1593^2 * 1337 = 2 * 3 * 7 * 13 * 29 * 59 * 61 * 191 := by
  sorry

end prime_factorization_problem_l3429_342962


namespace a_range_theorem_l3429_342925

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- Define the range of a
def a_range (a : ℝ) : Prop := a ≥ 1

-- State the theorem
theorem a_range_theorem :
  (∀ x a : ℝ, (¬(q x a) → ¬(p x)) ∧ (∃ x : ℝ, ¬(p x) ∧ q x a)) →
  (∀ a : ℝ, a_range a ↔ ∀ x : ℝ, p x → q x a) :=
sorry

end a_range_theorem_l3429_342925


namespace next_door_neighbor_subscriptions_l3429_342982

/-- Represents the number of subscriptions sold to the next-door neighbor -/
def next_door_subscriptions : ℕ := sorry

/-- Represents the total number of subscriptions sold -/
def total_subscriptions : ℕ := sorry

/-- The amount earned per subscription -/
def amount_per_subscription : ℕ := 5

/-- The total amount earned -/
def total_amount_earned : ℕ := 55

/-- Subscriptions sold to parents -/
def parent_subscriptions : ℕ := 4

/-- Subscriptions sold to grandfather -/
def grandfather_subscriptions : ℕ := 1

theorem next_door_neighbor_subscriptions :
  (next_door_subscriptions * amount_per_subscription +
   2 * next_door_subscriptions * amount_per_subscription +
   parent_subscriptions * amount_per_subscription +
   grandfather_subscriptions * amount_per_subscription = total_amount_earned) →
  (total_subscriptions = total_amount_earned / amount_per_subscription) →
  (next_door_subscriptions = 2) := by
  sorry

end next_door_neighbor_subscriptions_l3429_342982


namespace right_triangle_side_ratio_l3429_342975

theorem right_triangle_side_ratio (a d : ℝ) (ha : a > 0) (hd : d > 0) :
  (a^2 + (a + d)^2 = (a + 2*d)^2) → (a = 3*d) := by
  sorry

end right_triangle_side_ratio_l3429_342975


namespace inequality_holds_for_n_2_and_8_l3429_342970

theorem inequality_holds_for_n_2_and_8 (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  ((Real.exp (2 * x)) / (Real.log y)^2 > (x / y)^2) ∧
  ((Real.exp (2 * x)) / (Real.log y)^2 > (x / y)^8) :=
by sorry

end inequality_holds_for_n_2_and_8_l3429_342970


namespace intersection_in_fourth_quadrant_l3429_342903

/-- The intersection point of two lines is in the fourth quadrant if and only if k is within a specific range -/
theorem intersection_in_fourth_quadrant (k : ℝ) :
  (∃ x y : ℝ, y = -2 * x + 3 * k + 14 ∧ x - 4 * y = -3 * k - 2 ∧ x > 0 ∧ y < 0) ↔ 
  -6 < k ∧ k < -2 := by
sorry

end intersection_in_fourth_quadrant_l3429_342903


namespace triangle_side_length_l3429_342948

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  -- Sides form an arithmetic sequence
  2 * b = a + c →
  -- Angle B is 30°
  B = π / 6 →
  -- Area of triangle is 3/2
  (1 / 2) * a * c * Real.sin B = 3 / 2 →
  -- Side b has length √3 + 1
  b = Real.sqrt 3 + 1 := by
sorry

end triangle_side_length_l3429_342948


namespace m_range_l3429_342947

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem m_range (m : ℝ) : (¬(p m ∨ q m)) → m ≥ 2 := by sorry

end m_range_l3429_342947


namespace square_diagonal_ratio_l3429_342996

theorem square_diagonal_ratio (a b : ℝ) (h : a > 0) (k : b > 0) :
  (4 * a) / (4 * b) = 3 / 2 → (a * Real.sqrt 2) / (b * Real.sqrt 2) = 3 / 2 := by
  sorry

end square_diagonal_ratio_l3429_342996


namespace proposition_truth_l3429_342906

theorem proposition_truth : 
  (¬ (∀ x : ℝ, x + 1/x ≥ 2)) ∧ 
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi/2) ∧ Real.sin x + Real.cos x = Real.sqrt 2) := by
  sorry

end proposition_truth_l3429_342906


namespace ava_activities_duration_l3429_342960

/-- Converts hours to minutes -/
def hours_to_minutes (h : ℕ) : ℕ := h * 60

/-- Represents a duration in hours and minutes -/
structure Duration :=
  (hours : ℕ)
  (minutes : ℕ)

/-- Converts a Duration to total minutes -/
def duration_to_minutes (d : Duration) : ℕ :=
  hours_to_minutes d.hours + d.minutes

/-- The total duration of Ava's activities in minutes -/
def total_duration : ℕ :=
  hours_to_minutes 4 +  -- TV watching
  duration_to_minutes { hours := 2, minutes := 30 } +  -- Video game playing
  duration_to_minutes { hours := 1, minutes := 45 }  -- Walking

theorem ava_activities_duration :
  total_duration = 495 := by sorry

end ava_activities_duration_l3429_342960


namespace equation_real_root_implies_a_range_l3429_342904

theorem equation_real_root_implies_a_range (a : ℝ) :
  (∃ x : ℝ, 2^(2*x) + 2^x * a + a + 1 = 0) →
  a ∈ Set.Iic (2 - 2 * Real.sqrt 2) ∪ Set.Ici (2 + 2 * Real.sqrt 2) :=
by sorry

end equation_real_root_implies_a_range_l3429_342904


namespace infinite_solutions_infinitely_many_solutions_l3429_342943

/-- The type of solutions to the equation x^3 + y^3 = z^4 - t^2 -/
def Solution := ℤ × ℤ × ℤ × ℤ

/-- Predicate to check if a tuple (x, y, z, t) is a solution to the equation -/
def is_solution (s : Solution) : Prop :=
  let (x, y, z, t) := s
  x^3 + y^3 = z^4 - t^2

/-- Function to transform a solution using an integer k -/
def transform (k : ℤ) (s : Solution) : Solution :=
  let (x, y, z, t) := s
  (k^4 * x, k^4 * y, k^3 * z, k^6 * t)

/-- Theorem stating that if (x, y, z, t) is a solution, then (k^4*x, k^4*y, k^3*z, k^6*t) is also a solution for any integer k -/
theorem infinite_solutions (s : Solution) (k : ℤ) :
  is_solution s → is_solution (transform k s) := by
  sorry

/-- Corollary: There are infinitely many solutions to the equation -/
theorem infinitely_many_solutions :
  ∃ f : ℕ → Solution, ∀ n : ℕ, is_solution (f n) ∧ ∀ m : ℕ, m ≠ n → f m ≠ f n := by
  sorry

end infinite_solutions_infinitely_many_solutions_l3429_342943


namespace sum_remainder_l3429_342946

theorem sum_remainder (a b c d e : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  a % 13 = 3 → b % 13 = 5 → c % 13 = 7 → d % 13 = 9 → e % 13 = 12 →
  (a + b + c + d + e) % 13 = 10 := by
  sorry

end sum_remainder_l3429_342946


namespace abhays_speed_l3429_342965

/-- Proves that Abhay's speed is 10.5 km/h given the problem conditions --/
theorem abhays_speed (distance : ℝ) (abhay_speed sameer_speed : ℝ) 
  (h1 : distance = 42)
  (h2 : distance / abhay_speed = distance / sameer_speed + 2)
  (h3 : distance / (2 * abhay_speed) = distance / sameer_speed - 1) :
  abhay_speed = 10.5 := by
  sorry

end abhays_speed_l3429_342965


namespace solid_identification_l3429_342932

-- Define the structure of a solid
structure Solid :=
  (faces : Nat)
  (hasParallelCongruentHexagons : Bool)
  (hasRectangularFaces : Bool)
  (hasSquareFace : Bool)
  (hasCongruentTriangles : Bool)
  (hasCommonVertex : Bool)

-- Define the types of solids
inductive SolidType
  | RegularHexagonalPrism
  | RegularSquarePyramid
  | Other

-- Function to determine the type of solid based on its structure
def identifySolid (s : Solid) : SolidType :=
  if s.faces == 8 && s.hasParallelCongruentHexagons && s.hasRectangularFaces then
    SolidType.RegularHexagonalPrism
  else if s.faces == 5 && s.hasSquareFace && s.hasCongruentTriangles && s.hasCommonVertex then
    SolidType.RegularSquarePyramid
  else
    SolidType.Other

-- Theorem stating that the given descriptions correspond to the correct solid types
theorem solid_identification :
  (∀ s : Solid, s.faces = 8 ∧ s.hasParallelCongruentHexagons ∧ s.hasRectangularFaces →
    identifySolid s = SolidType.RegularHexagonalPrism) ∧
  (∀ s : Solid, s.faces = 5 ∧ s.hasSquareFace ∧ s.hasCongruentTriangles ∧ s.hasCommonVertex →
    identifySolid s = SolidType.RegularSquarePyramid) :=
by sorry


end solid_identification_l3429_342932


namespace other_asymptote_equation_l3429_342955

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One of the asymptotes of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- x-coordinate of the foci -/
  foci_x : ℝ

/-- Theorem: Given a hyperbola with one asymptote y = 4x - 3 and foci with x-coordinate 3,
    the equation of the other asymptote is y = -4x + 21 -/
theorem other_asymptote_equation (h : Hyperbola) 
    (h1 : h.asymptote1 = fun x ↦ 4 * x - 3) 
    (h2 : h.foci_x = 3) : 
    ∃ asymptote2 : ℝ → ℝ, asymptote2 = fun x ↦ -4 * x + 21 := by
  sorry

end other_asymptote_equation_l3429_342955


namespace ashley_interest_earned_l3429_342988

/-- Calculates the total interest earned in one year given the investment conditions --/
def total_interest (contest_winnings : ℝ) (investment1 : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let investment2 := 2 * investment1 - 400
  let interest1 := investment1 * rate1
  let interest2 := investment2 * rate2
  interest1 + interest2

/-- Theorem stating that the total interest earned is $298 --/
theorem ashley_interest_earned :
  total_interest 5000 1800 0.05 0.065 = 298 := by
  sorry

end ashley_interest_earned_l3429_342988
