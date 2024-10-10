import Mathlib

namespace fraction_equality_l3625_362580

theorem fraction_equality (a b : ℝ) (h : a ≠ 0) : b / a = (a * b) / (a^2) := by
  sorry

end fraction_equality_l3625_362580


namespace resale_value_drops_below_target_in_four_years_l3625_362569

def initial_price : ℝ := 625000
def first_year_depreciation : ℝ := 0.20
def subsequent_depreciation : ℝ := 0.08
def target_value : ℝ := 400000

def resale_value (n : ℕ) : ℝ :=
  if n = 0 then initial_price
  else if n = 1 then initial_price * (1 - first_year_depreciation)
  else initial_price * (1 - first_year_depreciation) * (1 - subsequent_depreciation) ^ (n - 1)

theorem resale_value_drops_below_target_in_four_years :
  resale_value 4 < target_value ∧ ∀ k : ℕ, k < 4 → resale_value k ≥ target_value :=
sorry

end resale_value_drops_below_target_in_four_years_l3625_362569


namespace sin_cos_identity_l3625_362524

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.sin (10 * π / 180) - 
  Real.cos (10 * π / 180) * Real.sin (70 * π / 180) = 
  -(Real.sqrt 3 / 2) := by sorry

end sin_cos_identity_l3625_362524


namespace function_inequality_l3625_362523

def f (x : ℝ) := x^2 - 2*x

theorem function_inequality (a : ℝ) : 
  (∃ x ∈ Set.Icc 2 4, f x ≤ a^2 + 2*a) → a ∈ Set.Iic (-2) ∪ Set.Ici 0 := by
  sorry

end function_inequality_l3625_362523


namespace line_symmetry_l3625_362562

/-- Given two lines in the plane and a point, this theorem states that
    the lines are symmetric about the point. -/
theorem line_symmetry (x y : ℝ) : 
  (2 * x + 3 * y - 6 = 0) → 
  (2 * ((2 : ℝ) - x) + 3 * ((2 : ℝ) - y) - 6 = 0) →
  (2 * x + 3 * y - 4 = 0) := by
  sorry

end line_symmetry_l3625_362562


namespace chord_equation_l3625_362549

/-- The equation of a line containing a chord of a circle, given specific conditions -/
theorem chord_equation (O : ℝ × ℝ) (r : ℝ) (P : ℝ × ℝ) :
  O = (-1, 0) →
  r = 5 →
  P = (2, -3) →
  (∃ A B : ℝ × ℝ, 
    (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2 ∧
    (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2 ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) →
  ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ x - y - 5 = 0 :=
by sorry

end chord_equation_l3625_362549


namespace parabola_shift_l3625_362506

/-- Given a parabola with equation y = 3x², prove that after shifting 2 units right and 5 units up, the new equation is y = 3(x-2)² + 5 -/
theorem parabola_shift (x y : ℝ) : 
  (y = 3 * x^2) → 
  (∃ (new_y : ℝ), new_y = 3 * (x - 2)^2 + 5 ∧ 
    new_y = y + 5 ∧ 
    ∀ (new_x : ℝ), new_x = x - 2 → 3 * new_x^2 = 3 * (x - 2)^2) := by
  sorry

end parabola_shift_l3625_362506


namespace log_ratio_squared_l3625_362528

theorem log_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1)
  (h1 : Real.log a / Real.log 3 = Real.log 81 / Real.log b) (h2 : a * b = 243) :
  (Real.log (a / b) / Real.log 3)^2 = 9 := by
sorry

end log_ratio_squared_l3625_362528


namespace math_competition_problem_solving_l3625_362511

theorem math_competition_problem_solving (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.85) 
  (h2 : p2 = 0.80) 
  (h3 : p3 = 0.75) : 
  (p1 + p2 + p3 - 2) ≥ 0.40 := by
sorry

end math_competition_problem_solving_l3625_362511


namespace round_robin_tournament_sessions_l3625_362547

/-- The number of matches in a round-robin tournament with n players -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The minimum number of sessions required for a tournament -/
def min_sessions (total_matches : ℕ) (max_per_session : ℕ) : ℕ :=
  (total_matches + max_per_session - 1) / max_per_session

theorem round_robin_tournament_sessions :
  let n : ℕ := 10  -- number of players
  let max_per_session : ℕ := 15  -- maximum matches per session
  min_sessions (num_matches n) max_per_session = 3 := by
  sorry

end round_robin_tournament_sessions_l3625_362547


namespace permutation_ratio_l3625_362530

/-- The number of permutations of m elements chosen from n elements -/
def A (n m : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial (n - m))

/-- Theorem stating that the ratio of A(n,m) to A(n-1,m-1) equals n -/
theorem permutation_ratio (n m : ℕ) (h : n ≥ m) :
  A n m / A (n - 1) (m - 1) = n := by sorry

end permutation_ratio_l3625_362530


namespace rectangular_field_dimensions_l3625_362543

theorem rectangular_field_dimensions (m : ℝ) : 
  (3 * m + 8 > 0) →
  (m - 3 > 0) →
  (3 * m + 8) * (m - 3) = 85 →
  m = (1 + Real.sqrt 1309) / 6 := by sorry

end rectangular_field_dimensions_l3625_362543


namespace largest_fraction_l3625_362507

theorem largest_fraction :
  let f1 := 397 / 101
  let f2 := 487 / 121
  let f3 := 596 / 153
  let f4 := 678 / 173
  let f5 := 796 / 203
  f2 > f1 ∧ f2 > f3 ∧ f2 > f4 ∧ f2 > f5 := by
  sorry

end largest_fraction_l3625_362507


namespace exchange_to_hundred_bills_l3625_362519

def twenty_bills : ℕ := 10
def ten_bills : ℕ := 8
def five_bills : ℕ := 4

def total_amount : ℕ := twenty_bills * 20 + ten_bills * 10 + five_bills * 5

theorem exchange_to_hundred_bills :
  (total_amount / 100 : ℕ) = 3 := by sorry

end exchange_to_hundred_bills_l3625_362519


namespace sqrt_x_minus_3_undefined_l3625_362517

theorem sqrt_x_minus_3_undefined (x : ℕ+) : 
  ¬ (∃ (y : ℝ), y^2 = (x : ℝ) - 3) ↔ x = 1 ∨ x = 2 := by
  sorry

end sqrt_x_minus_3_undefined_l3625_362517


namespace pizza_ratio_proof_l3625_362560

theorem pizza_ratio_proof (total_slices : ℕ) (calories_per_slice : ℕ) (calories_eaten : ℕ) : 
  total_slices = 8 → 
  calories_per_slice = 300 → 
  calories_eaten = 1200 → 
  (calories_eaten / calories_per_slice : ℚ) / total_slices = 1 / 2 := by
  sorry

end pizza_ratio_proof_l3625_362560


namespace radian_measure_60_degrees_l3625_362501

/-- The radian measure of a 60° angle is π/3. -/
theorem radian_measure_60_degrees :
  (60 * Real.pi / 180 : ℝ) = Real.pi / 3 := by
  sorry

end radian_measure_60_degrees_l3625_362501


namespace quadratic_inequality_solution_set_l3625_362518

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 < x + 6} = {x : ℝ | -2 < x ∧ x < 3} :=
by sorry

end quadratic_inequality_solution_set_l3625_362518


namespace palm_tree_count_l3625_362561

theorem palm_tree_count (desert forest : ℕ) 
  (h1 : desert = (2 : ℚ) / 5 * forest)  -- Desert has 2/5 the trees of the forest
  (h2 : desert + forest = 7000)         -- Total trees in both locations
  : forest = 5000 := by
  sorry

end palm_tree_count_l3625_362561


namespace no_solution_for_equation_l3625_362534

theorem no_solution_for_equation :
  ¬ ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (2 / a + 2 / b = 1 / (a + b)) := by
  sorry

end no_solution_for_equation_l3625_362534


namespace increasing_linear_function_k_range_l3625_362540

theorem increasing_linear_function_k_range (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((k + 2) * x₁ + 1) < ((k + 2) * x₂ + 1)) →
  k > -2 :=
by sorry

end increasing_linear_function_k_range_l3625_362540


namespace students_behind_yoongi_l3625_362584

/-- Given a line of students, prove the number standing behind a specific student. -/
theorem students_behind_yoongi (total_students : ℕ) (jungkook_position : ℕ) (yoongi_position : ℕ) :
  total_students = 20 →
  jungkook_position = 3 →
  yoongi_position = jungkook_position - 1 →
  total_students - yoongi_position = 18 := by
  sorry

end students_behind_yoongi_l3625_362584


namespace total_rounded_to_nearest_dollar_l3625_362590

def purchase1 : ℚ := 245/100
def purchase2 : ℚ := 358/100
def purchase3 : ℚ := 796/100

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - x.floor < 1/2 then x.floor else x.ceil

theorem total_rounded_to_nearest_dollar :
  round_to_nearest_dollar (purchase1 + purchase2 + purchase3) = 14 := by
  sorry

end total_rounded_to_nearest_dollar_l3625_362590


namespace cake_cost_is_correct_l3625_362555

/-- The cost of a piece of cake in dollars -/
def cake_cost : ℚ := 7

/-- The cost of a cup of coffee in dollars -/
def coffee_cost : ℚ := 4

/-- The cost of a bowl of ice cream in dollars -/
def ice_cream_cost : ℚ := 3

/-- The total cost for Mell and her two friends in dollars -/
def total_cost : ℚ := 51

/-- Theorem stating that the cake cost is correct given the conditions -/
theorem cake_cost_is_correct :
  cake_cost = 7 ∧
  coffee_cost = 4 ∧
  ice_cream_cost = 3 ∧
  total_cost = 51 ∧
  (2 * coffee_cost + cake_cost) + 2 * (2 * coffee_cost + cake_cost + ice_cream_cost) = total_cost :=
by sorry

end cake_cost_is_correct_l3625_362555


namespace fraction_problem_l3625_362566

theorem fraction_problem (N : ℝ) (f : ℝ) 
  (h1 : (1 / 3) * f * N = 15) 
  (h2 : (3 / 10) * N = 54) : 
  f = 1 / 4 := by
sorry

end fraction_problem_l3625_362566


namespace sum_of_solutions_quadratic_l3625_362532

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ p q : ℝ, p + q = -27 ∧ 81 - 27*x - x^2 = 0 → x = p ∨ x = q) :=
by sorry

end sum_of_solutions_quadratic_l3625_362532


namespace half_equals_fifty_percent_l3625_362504

theorem half_equals_fifty_percent (muffin : ℝ) (h : muffin > 0) :
  (1 / 2 : ℝ) * muffin = (50 / 100 : ℝ) * muffin := by sorry

end half_equals_fifty_percent_l3625_362504


namespace books_returned_thursday_l3625_362553

/-- The number of books returned on Thursday given the initial conditions and final count. -/
theorem books_returned_thursday 
  (initial_wednesday : ℕ) 
  (checkout_wednesday : ℕ) 
  (checkout_thursday : ℕ) 
  (returned_friday : ℕ) 
  (final_friday : ℕ) 
  (h1 : initial_wednesday = 98) 
  (h2 : checkout_wednesday = 43) 
  (h3 : checkout_thursday = 5) 
  (h4 : returned_friday = 7) 
  (h5 : final_friday = 80) : 
  final_friday = initial_wednesday - checkout_wednesday - checkout_thursday + returned_friday + 23 := by
  sorry

#check books_returned_thursday

end books_returned_thursday_l3625_362553


namespace fourth_root_equation_solution_l3625_362556

theorem fourth_root_equation_solution (x : ℝ) : 
  (x * (x^4)^(1/2))^(1/4) = 4 → x = 2^(8/3) := by sorry

end fourth_root_equation_solution_l3625_362556


namespace bacteria_growth_l3625_362542

/-- The factor by which the bacteria population increases each minute -/
def growth_factor : ℕ := 2

/-- The number of minutes that pass -/
def time : ℕ := 4

/-- The function that calculates the population after n minutes -/
def population (n : ℕ) : ℕ := growth_factor ^ n

/-- Theorem stating that after 4 minutes, the population is 16 times the original -/
theorem bacteria_growth :
  population time = 16 := by
  sorry

end bacteria_growth_l3625_362542


namespace pear_juice_percentage_approx_19_23_l3625_362576

/-- Represents the juice yield from fruits -/
structure JuiceYield where
  pears : ℕ
  pearJuice : ℚ
  oranges : ℕ
  orangeJuice : ℚ

/-- Represents the blend composition -/
structure Blend where
  pears : ℕ
  oranges : ℕ

/-- Calculates the percentage of pear juice in a blend -/
def pear_juice_percentage (yield : JuiceYield) (blend : Blend) : ℚ :=
  let pear_juice := (blend.pears : ℚ) * yield.pearJuice / yield.pears
  let orange_juice := (blend.oranges : ℚ) * yield.orangeJuice / yield.oranges
  let total_juice := pear_juice + orange_juice
  pear_juice / total_juice * 100

theorem pear_juice_percentage_approx_19_23 (yield : JuiceYield) (blend : Blend) :
  yield.pears = 4 ∧ 
  yield.pearJuice = 10 ∧ 
  yield.oranges = 1 ∧ 
  yield.orangeJuice = 7 ∧
  blend.pears = 8 ∧
  blend.oranges = 12 →
  abs (pear_juice_percentage yield blend - 19.23) < 0.01 := by
  sorry

end pear_juice_percentage_approx_19_23_l3625_362576


namespace dans_remaining_marbles_l3625_362597

-- Define the initial number of green marbles Dan has
def initial_green_marbles : ℝ := 32.0

-- Define the number of green marbles Mike took
def marbles_taken : ℝ := 23.0

-- Define the number of green marbles Dan has now
def remaining_green_marbles : ℝ := initial_green_marbles - marbles_taken

-- Theorem to prove
theorem dans_remaining_marbles :
  remaining_green_marbles = 9.0 := by sorry

end dans_remaining_marbles_l3625_362597


namespace arithmetic_sequence_product_l3625_362593

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n : ℕ, b (n + 1) > b n) →  -- increasing sequence
  (∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d) →  -- arithmetic sequence
  (b 5 * b 6 = 21) →  -- given condition
  (b 4 * b 7 = -779 ∨ b 4 * b 7 = 21) := by
sorry

end arithmetic_sequence_product_l3625_362593


namespace smallest_number_is_negative_one_l3625_362582

theorem smallest_number_is_negative_one :
  let numbers : Finset ℝ := {0, 1/3, -1, Real.sqrt 2}
  ∀ x ∈ numbers, -1 ≤ x :=
by
  sorry

end smallest_number_is_negative_one_l3625_362582


namespace dimes_borrowed_l3625_362585

/-- Represents the number of dimes Sam had initially -/
def initial_dimes : ℕ := 8

/-- Represents the number of dimes Sam has now -/
def remaining_dimes : ℕ := 4

/-- Represents the number of dimes Sam's sister borrowed -/
def borrowed_dimes : ℕ := initial_dimes - remaining_dimes

theorem dimes_borrowed :
  borrowed_dimes = initial_dimes - remaining_dimes :=
by sorry

end dimes_borrowed_l3625_362585


namespace total_calculators_l3625_362548

/-- Represents the number of calculators assembled by a person in a unit of time -/
structure AssemblyRate where
  calculators : ℕ
  time_units : ℕ

/-- The problem setup -/
def calculator_problem (erika nick sam : AssemblyRate) : Prop :=
  -- Erika assembles 3 calculators in the same time Nick assembles 2
  erika.calculators * nick.time_units = 3 * nick.calculators * erika.time_units ∧
  -- Nick assembles 1 calculator in the same time Sam assembles 3
  nick.calculators * sam.time_units = sam.calculators * nick.time_units ∧
  -- Erika's rate is 3 calculators per time unit
  erika.calculators = 3 ∧ erika.time_units = 1

/-- The theorem to prove -/
theorem total_calculators (erika nick sam : AssemblyRate) 
  (h : calculator_problem erika nick sam) : 
  9 * erika.time_units / erika.calculators * 
  (erika.calculators + nick.calculators * erika.time_units / nick.time_units + 
   sam.calculators * erika.time_units / sam.time_units) = 33 := by
  sorry

end total_calculators_l3625_362548


namespace triangle_property_l3625_362592

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2a*sin(B) = √3*b, a = 6, and b = 2√3, then angle A = π/3 and the area is 6√3 --/
theorem triangle_property (a b c A B C : Real) : 
  0 < A ∧ A < π/2 →  -- Acute triangle condition
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  2 * a * Real.sin B = Real.sqrt 3 * b →  -- Given condition
  a = 6 →  -- Given condition
  b = 2 * Real.sqrt 3 →  -- Given condition
  A = π/3 ∧ (1/2 * b * c * Real.sin A = 6 * Real.sqrt 3) := by
  sorry

end triangle_property_l3625_362592


namespace carpenter_rate_proof_l3625_362525

def carpenter_hourly_rate (total_estimate : ℚ) (material_cost : ℚ) (job_duration : ℚ) : ℚ :=
  (total_estimate - material_cost) / job_duration

theorem carpenter_rate_proof (total_estimate : ℚ) (material_cost : ℚ) (job_duration : ℚ)
  (h1 : total_estimate = 980)
  (h2 : material_cost = 560)
  (h3 : job_duration = 15) :
  carpenter_hourly_rate total_estimate material_cost job_duration = 28 := by
  sorry

end carpenter_rate_proof_l3625_362525


namespace floor_sum_equals_129_l3625_362538

theorem floor_sum_equals_129 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + 2*b^2 = 2016)
  (h2 : c^2 + 2*d^2 = 2016)
  (h3 : a*c = 1024)
  (h4 : b*d = 1024) :
  ⌊a + b + c + d⌋ = 129 := by
sorry

end floor_sum_equals_129_l3625_362538


namespace smallest_integer_satisfying_inequality_eight_satisfies_inequality_eight_is_smallest_integer_l3625_362594

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, y < 3*y - 15 → y ≥ 8 :=
by
  sorry

theorem eight_satisfies_inequality : 
  (8 : ℤ) < 3*8 - 15 :=
by
  sorry

theorem eight_is_smallest_integer :
  ∃ y : ℤ, y < 3*y - 15 ∧ ∀ z : ℤ, z < 3*z - 15 → z ≥ y :=
by
  sorry

end smallest_integer_satisfying_inequality_eight_satisfies_inequality_eight_is_smallest_integer_l3625_362594


namespace function_properties_l3625_362545

-- Define the function f(x) = x^3 + ax^2 + b
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 3*x + y - 3 = 0

-- Theorem statement
theorem function_properties (a b : ℝ) :
  (tangent_line 1 (f a b 1)) →
  (∃ (f' : ℝ → ℝ), 
    (∀ x, f' x = 3*x^2 - 6*x) ∧
    (∀ x, x < 0 → (f' x > 0)) ∧
    (∀ x, 0 < x ∧ x < 2 → (f' x < 0)) ∧
    (∀ x, x > 2 → (f' x > 0))) ∧
  (∀ t, t > 0 →
    (t ≤ 2 → 
      (∀ x, x ∈ Set.Icc 0 t → f (-3) 2 x ≤ 2 ∧ f (-3) 2 t ≤ f (-3) 2 x) ∧
      f (-3) 2 t = t^3 - 3*t^2 + 2) ∧
    (2 < t ∧ t ≤ 3 →
      (∀ x, x ∈ Set.Icc 0 t → -2 ≤ f (-3) 2 x ∧ f (-3) 2 x ≤ 2)) ∧
    (t > 3 →
      (∀ x, x ∈ Set.Icc 0 t → -2 ≤ f (-3) 2 x ∧ f (-3) 2 x ≤ f (-3) 2 t) ∧
      f (-3) 2 t = t^3 - 3*t^2 + 2)) := by
  sorry

end function_properties_l3625_362545


namespace bread_cost_calculation_l3625_362515

/-- Calculates the total cost of bread for a committee luncheon --/
def calculate_bread_cost (committee_size : ℕ) (sandwiches_per_person : ℕ) 
  (bread_types : ℕ) (croissant_pack_size : ℕ) (croissant_pack_price : ℚ)
  (ciabatta_pack_size : ℕ) (ciabatta_pack_price : ℚ)
  (multigrain_pack_size : ℕ) (multigrain_pack_price : ℚ)
  (discount_threshold : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  sorry

/-- The total cost of bread for the committee luncheon is $51.36 --/
theorem bread_cost_calculation :
  calculate_bread_cost 24 2 3 12 8 10 9 20 7 50 0.1 0.07 = 51.36 := by
  sorry

end bread_cost_calculation_l3625_362515


namespace water_depth_calculation_l3625_362567

/-- The depth of water given Dean's height and a multiplier -/
def water_depth (dean_height : ℝ) (depth_multiplier : ℝ) : ℝ :=
  dean_height * depth_multiplier

/-- Theorem: The water depth is 60 feet when Dean's height is 6 feet
    and the depth is 10 times his height -/
theorem water_depth_calculation :
  water_depth 6 10 = 60 := by
  sorry

end water_depth_calculation_l3625_362567


namespace simple_interest_problem_l3625_362529

/-- Simple interest calculation -/
theorem simple_interest_problem (interest_rate : ℚ) (time_period : ℚ) (earned_interest : ℕ) :
  interest_rate = 50 / 3 →
  time_period = 3 / 4 →
  earned_interest = 8625 →
  ∃ (principal : ℕ), 
    principal = 6900000 ∧
    earned_interest = (principal * interest_rate * time_period : ℚ).num / 100 := by
  sorry

end simple_interest_problem_l3625_362529


namespace solve_for_y_l3625_362520

theorem solve_for_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 25) (h4 : x / y = 36) : y = 5 / 6 := by
  sorry

end solve_for_y_l3625_362520


namespace parallelogram_area_example_l3625_362572

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_example : parallelogram_area 12 6 = 72 := by
  sorry

end parallelogram_area_example_l3625_362572


namespace quadratic_equation_solution_l3625_362512

theorem quadratic_equation_solution : 
  ∀ y : ℝ, y^2 - 2*y + 1 = -(y - 1)*(y - 3) → y = 1 ∨ y = 2 :=
by
  sorry

end quadratic_equation_solution_l3625_362512


namespace rebeccas_income_l3625_362577

/-- Rebecca's annual income problem -/
theorem rebeccas_income (R : ℚ) : 
  (∃ (J : ℚ), J = 18000 ∧ R + 3000 = 0.5 * (R + 3000 + J)) → R = 15000 := by
  sorry

end rebeccas_income_l3625_362577


namespace wedding_catering_budget_l3625_362536

/-- Calculates the total catering budget for a wedding given the specified conditions. -/
theorem wedding_catering_budget 
  (total_guests : ℕ) 
  (steak_to_chicken_ratio : ℕ) 
  (steak_cost chicken_cost : ℕ) : 
  total_guests = 80 → 
  steak_to_chicken_ratio = 3 → 
  steak_cost = 25 → 
  chicken_cost = 18 → 
  (total_guests * steak_cost * steak_to_chicken_ratio + total_guests * chicken_cost) / (steak_to_chicken_ratio + 1) = 1860 := by
  sorry

#eval (80 * 25 * 3 + 80 * 18) / (3 + 1)

end wedding_catering_budget_l3625_362536


namespace meet_time_opposite_directions_l3625_362587

/-- Represents an athlete running on a track -/
structure Athlete where
  lap_time : ℝ
  speed : ℝ

/-- Represents a closed track -/
structure Track where
  length : ℝ

/-- The scenario of two athletes running on a track -/
def running_scenario (t : Track) (a1 a2 : Athlete) : Prop :=
  a1.speed = t.length / a1.lap_time ∧
  a2.speed = t.length / a2.lap_time ∧
  a2.lap_time = a1.lap_time + 5 ∧
  30 * a1.speed - 30 * a2.speed = t.length

theorem meet_time_opposite_directions 
  (t : Track) (a1 a2 : Athlete) 
  (h : running_scenario t a1 a2) : 
  t.length / (a1.speed + a2.speed) = 6 := by
  sorry


end meet_time_opposite_directions_l3625_362587


namespace berry_average_temperature_l3625_362551

def berry_temperatures : List (List Float) := [
  [37.3, 37.2, 36.9],  -- Sunday
  [36.6, 36.9, 37.1],  -- Monday
  [37.1, 37.3, 37.2],  -- Tuesday
  [36.8, 37.3, 37.5],  -- Wednesday
  [37.1, 37.7, 37.3],  -- Thursday
  [37.5, 37.4, 36.9],  -- Friday
  [36.9, 37.0, 37.1]   -- Saturday
]

def average_temperature (temperatures : List (List Float)) : Float :=
  let total_sum := temperatures.map (·.sum) |>.sum
  let total_count := temperatures.length * temperatures.head!.length
  total_sum / total_count.toFloat

theorem berry_average_temperature :
  (average_temperature berry_temperatures).floor = 37 ∧
  (average_temperature berry_temperatures - (average_temperature berry_temperatures).floor) * 100 ≥ 62 :=
by sorry

end berry_average_temperature_l3625_362551


namespace base_9_101_to_decimal_l3625_362527

def base_9_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ (digits.length - 1 - i))) 0

theorem base_9_101_to_decimal :
  base_9_to_decimal [1, 0, 1] = 82 := by
  sorry

end base_9_101_to_decimal_l3625_362527


namespace line_always_intersects_hyperbola_iff_k_in_range_l3625_362514

/-- A line intersects a hyperbola if their equations have a common solution -/
def intersects (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ x^2 - 2 * y^2 = 1

/-- The main theorem: if a line always intersects the hyperbola, then k is in the open interval (-√2/2, √2/2) -/
theorem line_always_intersects_hyperbola_iff_k_in_range (k : ℝ) :
  (∀ b : ℝ, intersects k b) ↔ -Real.sqrt 2 / 2 < k ∧ k < Real.sqrt 2 / 2 := by
  sorry

end line_always_intersects_hyperbola_iff_k_in_range_l3625_362514


namespace geometric_sequence_first_term_l3625_362598

/-- A monotonically decreasing geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 0 < q ∧ q < 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_a3 : a 3 = 1)
  (h_sum : a 2 + a 4 = 5/2) :
  a 1 = 4 := by
sorry

end geometric_sequence_first_term_l3625_362598


namespace pond_radius_l3625_362573

/-- The radius of a circular pond with a diameter of 14 meters is 7 meters. -/
theorem pond_radius (diameter : ℝ) (h : diameter = 14) : diameter / 2 = 7 := by
  sorry

end pond_radius_l3625_362573


namespace coefficient_of_b_l3625_362578

theorem coefficient_of_b (a b : ℝ) (h1 : 7 * a = b) (h2 : b = 15) (h3 : 42 * a * b = 675) :
  42 * a = 45 := by
sorry

end coefficient_of_b_l3625_362578


namespace quadratic_has_real_roots_k_values_l3625_362564

-- Define the quadratic equation
def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := (x - 1)^2 + k*(x - 1)

-- Theorem 1: The quadratic equation always has real roots
theorem quadratic_has_real_roots (k : ℝ) : 
  ∃ x : ℝ, quadratic_equation k x = 0 :=
sorry

-- Theorem 2: If the roots satisfy the given condition, k is either 4 or -1
theorem k_values (k : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    quadratic_equation k x₁ = 0 ∧ 
    quadratic_equation k x₂ = 0 ∧ 
    x₁^2 + x₂^2 = 7 - x₁*x₂) →
  (k = 4 ∨ k = -1) :=
sorry

end quadratic_has_real_roots_k_values_l3625_362564


namespace sufficient_not_necessary_contrapositive_correct_negation_incorrect_disjunction_false_implication_l3625_362505

-- Definition for the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

-- Proposition 1
theorem sufficient_not_necessary : 
  (∃ x : ℝ, x ≠ 1 ∧ quadratic_eq x) ∧ 
  (∀ x : ℝ, x = 1 → quadratic_eq x) := by sorry

-- Proposition 2
theorem contrapositive_correct :
  (∀ x : ℝ, quadratic_eq x → x = 1) ↔ 
  (∀ x : ℝ, x ≠ 1 → ¬(quadratic_eq x)) := by sorry

-- Proposition 3
theorem negation_incorrect :
  ¬(∃ x : ℝ, x > 0 ∧ x^2 + x + 1 < 0) ≠ 
  (∀ x : ℝ, x ≤ 0 → x^2 + x + 1 ≥ 0) := by sorry

-- Proposition 4
theorem disjunction_false_implication :
  ¬(∀ p q : Prop, ¬(p ∨ q) → (¬p ∧ ¬q)) := by sorry

end sufficient_not_necessary_contrapositive_correct_negation_incorrect_disjunction_false_implication_l3625_362505


namespace james_paper_usage_l3625_362571

/-- The number of books James prints -/
def num_books : ℕ := 2

/-- The number of pages in each book -/
def pages_per_book : ℕ := 600

/-- The number of pages printed on each side of a sheet -/
def pages_per_side : ℕ := 4

/-- Whether the printing is double-sided (true) or single-sided (false) -/
def is_double_sided : Bool := true

/-- Calculates the total number of sheets of paper James uses -/
def sheets_used : ℕ :=
  let total_pages := num_books * pages_per_book
  let pages_per_sheet := pages_per_side * (if is_double_sided then 2 else 1)
  total_pages / pages_per_sheet

theorem james_paper_usage :
  sheets_used = 150 := by sorry

end james_paper_usage_l3625_362571


namespace sum_product_range_l3625_362502

theorem sum_product_range (x y z : ℝ) (h : x + y + z = 3) :
  ∃ S : Set ℝ, S = Set.Iic (9/4) ∧
  ∀ t : ℝ, (∃ a b c : ℝ, a + b + c = 3 ∧ t = a*b + a*c + b*c) ↔ t ∈ S :=
sorry

end sum_product_range_l3625_362502


namespace scientific_notation_equivalence_l3625_362570

theorem scientific_notation_equivalence : 
  ∃ (x : ℝ) (n : ℤ), 11580000 = x * (10 : ℝ) ^ n ∧ 1 ≤ x ∧ x < 10 := by
  sorry

end scientific_notation_equivalence_l3625_362570


namespace frog_jump_coordinates_l3625_362559

def initial_point : ℝ × ℝ := (-1, 0)
def right_jump : ℝ := 2
def up_jump : ℝ := 2

def final_point (p : ℝ × ℝ) (right : ℝ) (up : ℝ) : ℝ × ℝ :=
  (p.1 + right, p.2 + up)

theorem frog_jump_coordinates :
  final_point initial_point right_jump up_jump = (1, 2) := by sorry

end frog_jump_coordinates_l3625_362559


namespace inequality_solution_set_l3625_362591

theorem inequality_solution_set (x : ℝ) : x^2 + 3 < 4*x ↔ 1 < x ∧ x < 3 := by
  sorry

end inequality_solution_set_l3625_362591


namespace lavinias_son_older_than_daughter_l3625_362513

def katies_daughter_age : ℕ := 12

def lavinias_daughter_age (k : ℕ) : ℕ :=
  k / 3

def lavinias_son_age (k : ℕ) : ℕ :=
  2 * k

theorem lavinias_son_older_than_daughter :
  lavinias_son_age katies_daughter_age - lavinias_daughter_age katies_daughter_age = 20 := by
  sorry

end lavinias_son_older_than_daughter_l3625_362513


namespace john_skateboard_distance_l3625_362508

/-- Represents John's journey with skateboarding distances -/
structure JourneyDistances where
  to_park_skateboard : ℕ
  to_park_walk : ℕ
  to_park_bike : ℕ
  park_jog : ℕ
  from_park_bike : ℕ
  from_park_swim : ℕ
  from_park_skateboard : ℕ

/-- Calculates the total skateboarding distance for John's journey -/
def total_skateboard_distance (j : JourneyDistances) : ℕ :=
  j.to_park_skateboard + j.from_park_skateboard

/-- Theorem: John's total skateboarding distance is 25 miles -/
theorem john_skateboard_distance (j : JourneyDistances)
  (h1 : j.to_park_skateboard = 16)
  (h2 : j.to_park_walk = 8)
  (h3 : j.to_park_bike = 6)
  (h4 : j.park_jog = 3)
  (h5 : j.from_park_bike = 5)
  (h6 : j.from_park_swim = 1)
  (h7 : j.from_park_skateboard = 9) :
  total_skateboard_distance j = 25 := by
  sorry

end john_skateboard_distance_l3625_362508


namespace quadratic_equation_roots_l3625_362522

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 3 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 + m*y + 3 = 0 ∧ y = 3 ∧ m = -4) :=
by sorry

end quadratic_equation_roots_l3625_362522


namespace hotel_rate_proof_l3625_362579

/-- The flat rate for the first night in a hotel. -/
def flat_rate : ℝ := 80

/-- The additional fee for each subsequent night. -/
def additional_fee : ℝ := 40

/-- The cost for a stay of n nights. -/
def cost (n : ℕ) : ℝ := flat_rate + additional_fee * (n - 1)

theorem hotel_rate_proof :
  (cost 4 = 200) ∧ (cost 7 = 320) → flat_rate = 80 := by
  sorry

end hotel_rate_proof_l3625_362579


namespace max_value_of_f_l3625_362558

-- Define the function f(x) = x³ - 3x²
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Theorem statement
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 4 ∧
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 4 → f x ≤ f c) ∧
  f c = 16 := by
sorry


end max_value_of_f_l3625_362558


namespace larger_number_problem_l3625_362500

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 45) (h2 : x - y = 5) : x = 25 := by
  sorry

end larger_number_problem_l3625_362500


namespace sum_of_three_numbers_l3625_362557

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 99) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 19 := by
sorry

end sum_of_three_numbers_l3625_362557


namespace multiples_of_six_ending_in_four_l3625_362596

theorem multiples_of_six_ending_in_four (n : ℕ) : 
  (∃ k, k ∈ Finset.range 1000 ∧ k % 10 = 4 ∧ k % 6 = 0) ↔ n = 17 :=
by sorry

end multiples_of_six_ending_in_four_l3625_362596


namespace solve_equation_l3625_362552

theorem solve_equation (x y : ℝ) :
  3 * x - 5 * y = 7 → y = (3 * x - 7) / 5 := by
sorry

end solve_equation_l3625_362552


namespace wild_ducks_geese_meeting_l3625_362509

/-- The number of days it takes wild ducks to fly from South Sea to North Sea -/
def wild_ducks_days : ℕ := 7

/-- The number of days it takes geese to fly from North Sea to South Sea -/
def geese_days : ℕ := 9

/-- The equation representing the meeting of wild ducks and geese -/
def meeting_equation (x : ℝ) : Prop :=
  (1 / wild_ducks_days : ℝ) * x + (1 / geese_days : ℝ) * x = 1

/-- Theorem stating that the solution to the meeting equation represents
    the number of days it takes for wild ducks and geese to meet -/
theorem wild_ducks_geese_meeting :
  ∃ x : ℝ, x > 0 ∧ meeting_equation x ∧
    ∀ y : ℝ, y > 0 ∧ meeting_equation y → x = y :=
sorry

end wild_ducks_geese_meeting_l3625_362509


namespace egg_count_l3625_362521

theorem egg_count (initial_eggs : ℕ) (added_eggs : ℕ) : 
  initial_eggs = 7 → added_eggs = 4 → initial_eggs + added_eggs = 11 := by
  sorry

#check egg_count

end egg_count_l3625_362521


namespace eighth_term_of_sequence_l3625_362550

theorem eighth_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, S n = n^2) → a 8 = 15 := by
  sorry

end eighth_term_of_sequence_l3625_362550


namespace longest_piece_length_l3625_362568

theorem longest_piece_length (a b c : ℕ) (ha : a = 45) (hb : b = 75) (hc : c = 90) :
  Nat.gcd a (Nat.gcd b c) = 15 := by
  sorry

end longest_piece_length_l3625_362568


namespace sum_of_reciprocals_of_factors_of_30_l3625_362554

def factors_of_30 : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

theorem sum_of_reciprocals_of_factors_of_30 :
  (factors_of_30.map (λ x => (1 : ℚ) / x)).sum = 12 / 5 := by
  sorry

end sum_of_reciprocals_of_factors_of_30_l3625_362554


namespace largest_number_of_three_l3625_362586

theorem largest_number_of_three (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (sum_prod_eq : a * b + a * c + b * c = -10)
  (prod_eq : a * b * c = -18) :
  max a (max b c) = -1 + Real.sqrt 7 := by
  sorry

end largest_number_of_three_l3625_362586


namespace symmetric_center_phi_l3625_362546

theorem symmetric_center_phi (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (-2 * x + φ)) →
  0 < φ →
  φ < π →
  (∃ k : ℤ, -2 * (π / 3) + φ = k * π) →
  φ = 2 * π / 3 := by
sorry

end symmetric_center_phi_l3625_362546


namespace stratified_sampling_second_group_l3625_362575

theorem stratified_sampling_second_group (total_sample : ℕ) 
  (ratio_first ratio_second ratio_third : ℕ) :
  ratio_first > 0 ∧ ratio_second > 0 ∧ ratio_third > 0 →
  total_sample = 240 →
  ratio_first = 5 ∧ ratio_second = 4 ∧ ratio_third = 3 →
  (ratio_second : ℚ) / (ratio_first + ratio_second + ratio_third : ℚ) * total_sample = 80 :=
by sorry

end stratified_sampling_second_group_l3625_362575


namespace reciprocal_of_negative_fraction_l3625_362544

theorem reciprocal_of_negative_fraction (n : ℤ) (h : n ≠ 0) :
  (-(1 : ℚ) / n)⁻¹ = -n := by sorry

end reciprocal_of_negative_fraction_l3625_362544


namespace movie_ticket_price_l3625_362526

/-- The cost of a movie date, given ticket price, combo meal price, and candy price -/
def movie_date_cost (ticket_price : ℚ) (combo_price : ℚ) (candy_price : ℚ) : ℚ :=
  2 * ticket_price + combo_price + 2 * candy_price

/-- Theorem stating that the movie ticket price is $10.00 given the conditions of Connor's date -/
theorem movie_ticket_price :
  ∃ (ticket_price : ℚ),
    movie_date_cost ticket_price 11 2.5 = 36 ∧
    ticket_price = 10 := by
  sorry

end movie_ticket_price_l3625_362526


namespace lowest_common_multiple_even_14_to_21_l3625_362535

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem lowest_common_multiple_even_14_to_21 :
  ∀ n : ℕ, n > 0 →
  (∀ k : ℕ, 14 ≤ k → k ≤ 21 → is_even k → divides k n) →
  n ≥ 5040 :=
sorry

end lowest_common_multiple_even_14_to_21_l3625_362535


namespace physics_class_size_l3625_362595

theorem physics_class_size (total_students : ℕ) (both_classes : ℕ) :
  total_students = 75 →
  both_classes = 9 →
  ∃ (math_only : ℕ) (phys_only : ℕ),
    total_students = math_only + phys_only + both_classes ∧
    phys_only + both_classes = 2 * (math_only + both_classes) →
  phys_only + both_classes = 56 := by
  sorry

end physics_class_size_l3625_362595


namespace equation_solution_system_solution_l3625_362531

-- Equation 1
theorem equation_solution (x : ℚ) : 
  (3 * x + 1) / 5 = 1 - (4 * x + 3) / 2 ↔ x = -7 / 26 := by sorry

-- System of equations
theorem system_solution (x y : ℚ) : 
  (3 * x - 4 * y = 14 ∧ 5 * x + 4 * y = 2) ↔ (x = 2 ∧ y = -2) := by sorry

end equation_solution_system_solution_l3625_362531


namespace unique_digit_product_solution_l3625_362565

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem unique_digit_product_solution :
  ∃! n : ℕ, digit_product n = n^2 - 10*n - 22 :=
sorry

end unique_digit_product_solution_l3625_362565


namespace consecutive_cubes_to_consecutive_squares_l3625_362510

theorem consecutive_cubes_to_consecutive_squares (A : ℕ) :
  (∃ k : ℕ, A^2 = (k + 1)^3 - k^3) →
  (∃ m : ℕ, A = m^2 + (m + 1)^2) :=
by sorry

end consecutive_cubes_to_consecutive_squares_l3625_362510


namespace six_digit_numbers_with_three_even_three_odd_count_six_digit_numbers_with_three_even_three_odd_l3625_362541

theorem six_digit_numbers_with_three_even_three_odd : ℕ :=
  let first_digit_choices := 9
  let position_choices := Nat.choose 5 2
  let same_parity_fill := 5^2
  let opposite_parity_fill := 2^3
  first_digit_choices * position_choices * same_parity_fill * opposite_parity_fill

theorem count_six_digit_numbers_with_three_even_three_odd :
  six_digit_numbers_with_three_even_three_odd = 90000 := by
  sorry

end six_digit_numbers_with_three_even_three_odd_count_six_digit_numbers_with_three_even_three_odd_l3625_362541


namespace largest_n_value_l3625_362503

/-- A function that checks if for any group of at least 145 candies,
    there is a type of candy which appears exactly 10 times -/
def has_type_with_10_occurrences (candies : List Nat) : Prop :=
  ∀ (group : List Nat), group.length ≥ 145 → group ⊆ candies →
    ∃ (type : Nat), (group.filter (· = type)).length = 10

/-- The theorem stating the largest possible value of n -/
theorem largest_n_value :
  ∀ (n : Nat),
    n > 145 →
    (∀ (candies : List Nat), candies.length = n →
      has_type_with_10_occurrences candies) →
    n ≤ 160 :=
by sorry

end largest_n_value_l3625_362503


namespace hidden_block_surface_area_l3625_362574

/-- Represents a block with a surface area -/
structure Block where
  surfaceArea : ℝ

/-- Represents a set of blocks created by cutting a larger block -/
structure CutBlocks where
  blocks : List Block
  numCuts : ℕ

/-- The proposition that the surface area of the hidden block is correct -/
def hiddenBlockSurfaceAreaIsCorrect (cb : CutBlocks) (hiddenSurfaceArea : ℝ) : Prop :=
  cb.numCuts = 3 ∧ 
  cb.blocks.length = 7 ∧ 
  (cb.blocks.map Block.surfaceArea).sum = 566 ∧
  hiddenSurfaceArea = 22

/-- Theorem stating that given the conditions, the hidden block's surface area is 22 -/
theorem hidden_block_surface_area 
  (cb : CutBlocks) (hiddenSurfaceArea : ℝ) : 
  hiddenBlockSurfaceAreaIsCorrect cb hiddenSurfaceArea := by
  sorry

#check hidden_block_surface_area

end hidden_block_surface_area_l3625_362574


namespace shared_vertex_angle_measure_l3625_362583

/-- The measure of the angle at the common vertex formed by a side of an equilateral triangle
    and a side of a regular pentagon, both inscribed in a circle. -/
def common_vertex_angle : ℝ := 24

/-- A regular pentagon inscribed in a circle -/
structure RegularPentagonInCircle where
  -- Add necessary fields

/-- An equilateral triangle inscribed in a circle -/
structure EquilateralTriangleInCircle where
  -- Add necessary fields

/-- Configuration of a regular pentagon and an equilateral triangle inscribed in a circle
    with a shared vertex -/
structure SharedVertexConfiguration where
  pentagon : RegularPentagonInCircle
  triangle : EquilateralTriangleInCircle
  -- Add field to represent the shared vertex

theorem shared_vertex_angle_measure (config : SharedVertexConfiguration) :
  common_vertex_angle = 24 := by
  sorry

end shared_vertex_angle_measure_l3625_362583


namespace min_value_expression_l3625_362516

theorem min_value_expression (a b m n : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_sum : a + b = 1) (h_prod : m * n = 2) :
  2 ≤ (a * m + b * n) * (b * m + a * n) := by
  sorry

end min_value_expression_l3625_362516


namespace constant_term_expansion_l3625_362599

theorem constant_term_expansion (n : ℕ) : 
  (∃ k : ℕ, k = n / 3 ∧ Nat.choose n k = 15) ↔ n = 6 := by
  sorry

end constant_term_expansion_l3625_362599


namespace bryans_milk_volume_l3625_362537

/-- The volume of milk in the first bottle, given the conditions of Bryan's milk purchase --/
theorem bryans_milk_volume (total_volume : ℚ) (second_bottle : ℚ) (third_bottle : ℚ) 
  (h1 : total_volume = 3)
  (h2 : second_bottle = 750 / 1000)
  (h3 : third_bottle = 250 / 1000) :
  total_volume - second_bottle - third_bottle = 2 := by
  sorry

end bryans_milk_volume_l3625_362537


namespace max_additional_plates_l3625_362581

/-- Represents the sets of letters for license plates --/
structure LicensePlateSets :=
  (first : Finset Char)
  (second : Finset Char)
  (third : Finset Char)

/-- The initial configuration of license plate sets --/
def initialSets : LicensePlateSets :=
  { first := {'B', 'F', 'G', 'T', 'Y'},
    second := {'E', 'U'},
    third := {'K', 'S', 'W'} }

/-- Calculates the number of possible license plates --/
def numPlates (sets : LicensePlateSets) : Nat :=
  sets.first.card * sets.second.card * sets.third.card

/-- Represents a new configuration after adding letters --/
structure NewConfig :=
  (sets : LicensePlateSets)
  (totalAdded : Nat)

/-- The theorem to be proved --/
theorem max_additional_plates :
  ∃ (newConfig : NewConfig),
    newConfig.totalAdded = 3 ∧
    ∀ (otherConfig : NewConfig),
      otherConfig.totalAdded = 3 →
      numPlates newConfig.sets - numPlates initialSets ≥
      numPlates otherConfig.sets - numPlates initialSets ∧
    numPlates newConfig.sets - numPlates initialSets = 50 :=
  sorry

end max_additional_plates_l3625_362581


namespace julia_math_contest_julia_math_contest_proof_l3625_362563

theorem julia_math_contest (total_problems : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) 
  (julia_score : ℤ) (julia_correct : ℕ) : Prop :=
  total_problems = 12 →
  correct_points = 6 →
  incorrect_points = -3 →
  julia_score = 27 →
  julia_correct = 7 →
  (julia_correct : ℤ) * correct_points + (total_problems - julia_correct : ℤ) * incorrect_points = julia_score

theorem julia_math_contest_proof : 
  ∃ (total_problems : ℕ) (correct_points incorrect_points julia_score : ℤ) (julia_correct : ℕ),
    julia_math_contest total_problems correct_points incorrect_points julia_score julia_correct :=
by
  sorry

end julia_math_contest_julia_math_contest_proof_l3625_362563


namespace initial_marble_difference_l3625_362539

/-- The number of marbles Ed and Doug initially had, and the number Ed currently has -/
structure MarbleCount where
  ed_initial : ℕ
  doug_initial : ℕ
  ed_current : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : MarbleCount) : Prop :=
  m.ed_current = 45 ∧
  m.ed_initial > m.doug_initial ∧
  m.ed_current = m.doug_initial - 11 + 21

/-- The theorem stating the initial difference in marbles -/
theorem initial_marble_difference (m : MarbleCount) 
  (h : marble_problem m) : m.ed_initial - m.doug_initial = 10 := by
  sorry


end initial_marble_difference_l3625_362539


namespace school_students_count_l3625_362533

theorem school_students_count (football cricket both neither : ℕ) 
  (h1 : football = 325)
  (h2 : cricket = 175)
  (h3 : neither = 50)
  (h4 : both = 90) :
  football + cricket - both + neither = 460 :=
by sorry

end school_students_count_l3625_362533


namespace triangle_problem_l3625_362589

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.b * Real.sin t.A = t.a * Real.cos t.B)
  (h2 : t.b = 3)
  (h3 : Real.sin t.C = Real.sqrt 3 * Real.sin t.A) :
  t.B = π / 6 ∧ t.a = 3 ∧ t.c = 3 * Real.sqrt 3 := by
  sorry

end triangle_problem_l3625_362589


namespace simplify_trig_expression_l3625_362588

theorem simplify_trig_expression :
  let cos45 := Real.sqrt 2 / 2
  let sin45 := Real.sqrt 2 / 2
  (cos45^3 + sin45^3) / (cos45 + sin45) = 1/2 := by
sorry

end simplify_trig_expression_l3625_362588
