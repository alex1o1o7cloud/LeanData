import Mathlib

namespace marbles_gcd_l3723_372312

theorem marbles_gcd (blue : Nat) (white : Nat) (red : Nat) (green : Nat) (yellow : Nat)
  (h_blue : blue = 24)
  (h_white : white = 17)
  (h_red : red = 13)
  (h_green : green = 7)
  (h_yellow : yellow = 5) :
  Nat.gcd blue (Nat.gcd white (Nat.gcd red (Nat.gcd green yellow))) = 1 := by
  sorry

end marbles_gcd_l3723_372312


namespace right_triangle_hypotenuse_l3723_372391

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 12 →
  (1/2) * a * b = 6 →
  a^2 + b^2 = c^2 →
  c = 5 := by
sorry

end right_triangle_hypotenuse_l3723_372391


namespace variance_binomial_4_half_l3723_372397

/-- The variance of a binomial distribution with n trials and probability p -/
def binomialVariance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- Theorem: The variance of a binomial distribution B(4, 1/2) is 1 -/
theorem variance_binomial_4_half :
  binomialVariance 4 (1/2 : ℝ) = 1 := by
  sorry

end variance_binomial_4_half_l3723_372397


namespace numbers_with_2019_divisors_l3723_372330

def has_2019_divisors (n : ℕ) : Prop :=
  (Finset.card (Nat.divisors n) = 2019)

theorem numbers_with_2019_divisors :
  {n : ℕ | n < 128^97 ∧ has_2019_divisors n} =
  {2^672 * 3^2, 2^672 * 5^2, 2^672 * 7^2, 2^672 * 11^2} :=
by sorry

end numbers_with_2019_divisors_l3723_372330


namespace train_length_train_length_example_l3723_372357

/-- The length of a train given its speed and time to cross a pole --/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ := by
  sorry

/-- Proof that a train with speed 60 km/hr crossing a pole in 4 seconds has a length of approximately 66.68 meters --/
theorem train_length_example : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |train_length 60 4 - 66.68| < ε := by
  sorry

end train_length_train_length_example_l3723_372357


namespace exponential_inequality_l3723_372311

theorem exponential_inequality (x y a b : ℝ) (h1 : x > y) (h2 : y > a) (h3 : a > b) (h4 : b > 1) :
  a^x > b^y := by sorry

end exponential_inequality_l3723_372311


namespace power_of_product_with_negative_l3723_372365

theorem power_of_product_with_negative (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end power_of_product_with_negative_l3723_372365


namespace arithmetic_sequence_general_term_l3723_372319

/-- An increasing arithmetic sequence with specific initial conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) > a n) ∧  -- Increasing
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧  -- Arithmetic
  (a 1 = 1) ∧  -- Initial condition
  (a 3 = (a 2)^2 - 4)  -- Given relation

/-- The theorem stating the general term of the sequence -/
theorem arithmetic_sequence_general_term (a : ℕ → ℝ) 
  (h : ArithmeticSequence a) : 
  ∀ n : ℕ, a n = 2 * n - 1 :=
sorry

end arithmetic_sequence_general_term_l3723_372319


namespace orangeade_price_day2_l3723_372378

/-- Represents the price and volume of orangeade on a given day -/
structure OrangeadeDay where
  orange_juice : ℝ
  water : ℝ
  price : ℝ

/-- The orangeade scenario over two days -/
def OrangeadeScenario (day1 day2 : OrangeadeDay) : Prop :=
  day1.orange_juice > 0 ∧
  day1.water = day1.orange_juice ∧
  day2.orange_juice = day1.orange_juice ∧
  day2.water = 2 * day1.water ∧
  day1.price = 0.5 ∧
  (day1.orange_juice + day1.water) * day1.price = (day2.orange_juice + day2.water) * day2.price

theorem orangeade_price_day2 (day1 day2 : OrangeadeDay) 
  (h : OrangeadeScenario day1 day2) : day2.price = 1/3 := by
  sorry

end orangeade_price_day2_l3723_372378


namespace abs_diff_ge_one_l3723_372303

theorem abs_diff_ge_one (a b c : ℝ) 
  (sum_eq : a + b + c = 2) 
  (sum_sq_eq : a^2 + b^2 + c^2 = 2) : 
  max (|a - b|) (max (|b - c|) (|c - a|)) ≥ 1 := by
  sorry

end abs_diff_ge_one_l3723_372303


namespace smallest_integer_in_consecutive_set_l3723_372356

theorem smallest_integer_in_consecutive_set (n : ℤ) : 
  (n + 6 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7)) →
  n = 0 :=
sorry

end smallest_integer_in_consecutive_set_l3723_372356


namespace unique_n_solution_l3723_372352

theorem unique_n_solution : ∃! (n : ℕ+), 
  Real.cos (π / (2 * n.val)) - Real.sin (π / (2 * n.val)) = Real.sqrt n.val / 2 :=
by
  -- The unique solution is n = 4
  use 4
  constructor
  -- Proof that n = 4 satisfies the equation
  sorry
  -- Proof of uniqueness
  sorry

end unique_n_solution_l3723_372352


namespace sequence_general_term_l3723_372335

/-- Given a sequence {a_n} where S_n is the sum of its first n terms, 
    prove that if S_n + a_n = (n-1) / (n(n+1)) for n ≥ 1, 
    then a_n = 1/(2^n) - 1/(n(n+1)) for all n ≥ 1 -/
theorem sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ, n ≥ 1 → S n + a n = (n - 1 : ℚ) / (n * (n + 1))) →
  ∀ n : ℕ, n ≥ 1 → a n = 1 / (2 ^ n) - 1 / (n * (n + 1)) :=
by sorry

end sequence_general_term_l3723_372335


namespace quadratic_roots_ratio_l3723_372359

theorem quadratic_roots_ratio (a b c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    (∀ x, a * x^2 - b * x + c = 0 ↔ x = x₁ ∨ x = x₂) ∧
    (∀ y, c * y^2 - a * y + b = 0 ↔ y = y₁ ∨ y = y₂) ∧
    (b / a ≥ 0) ∧
    (c / a = 9 * (a / c))) →
  (b / a) / (b / c) = -3 := by
sorry

end quadratic_roots_ratio_l3723_372359


namespace min_values_problem_l3723_372349

theorem min_values_problem (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a + b = 1) :
  (∀ x y : ℝ, x > y ∧ y > 0 ∧ x + y = 1 → a^2 + 2*b^2 ≤ x^2 + 2*y^2) ∧
  (∀ x y : ℝ, x > y ∧ y > 0 ∧ x + y = 1 → 4 / (a - b) + 1 / (2*b) ≤ 4 / (x - y) + 1 / (2*y)) ∧
  a^2 + 2*b^2 = 2/3 ∧
  4 / (a - b) + 1 / (2*b) = 9 :=
by sorry

end min_values_problem_l3723_372349


namespace avery_donation_l3723_372350

/-- The number of clothes Avery is donating -/
def total_clothes (shirts pants shorts : ℕ) : ℕ := shirts + pants + shorts

/-- Theorem stating the total number of clothes Avery is donating -/
theorem avery_donation :
  ∀ (shirts pants shorts : ℕ),
    shirts = 4 →
    pants = 2 * shirts →
    shorts = pants / 2 →
    total_clothes shirts pants shorts = 16 := by
  sorry

end avery_donation_l3723_372350


namespace train_length_l3723_372333

/-- The length of a train given its speed, bridge length, and crossing time -/
theorem train_length (speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  speed = 45 * 1000 / 3600 →
  bridge_length = 205 →
  crossing_time = 30 →
  speed * crossing_time - bridge_length = 170 := by
  sorry


end train_length_l3723_372333


namespace fifth_stair_area_fifth_stair_perimeter_twelfth_stair_area_twentyfifth_stair_perimeter_l3723_372358

-- Define the stair structure
structure Stair :=
  (n : ℕ)

-- Define the area function
def area (s : Stair) : ℕ :=
  s.n * (s.n + 1) / 2

-- Define the perimeter function
def perimeter (s : Stair) : ℕ :=
  4 * s.n

-- Theorem statements
theorem fifth_stair_area :
  area { n := 5 } = 15 := by sorry

theorem fifth_stair_perimeter :
  perimeter { n := 5 } = 20 := by sorry

theorem twelfth_stair_area :
  area { n := 12 } = 78 := by sorry

theorem twentyfifth_stair_perimeter :
  perimeter { n := 25 } = 100 := by sorry

end fifth_stair_area_fifth_stair_perimeter_twelfth_stair_area_twentyfifth_stair_perimeter_l3723_372358


namespace inequality_solution_range_l3723_372331

theorem inequality_solution_range (a : ℝ) : 
  (∃ x ∈ Set.Ioo 1 4, x^2 - 4*x - 2 - a > 0) → a ∈ Set.Ioi (-2) := by
  sorry

end inequality_solution_range_l3723_372331


namespace solve_fraction_equation_l3723_372361

theorem solve_fraction_equation :
  ∃ x : ℚ, (1 / 4 : ℚ) - (1 / 5 : ℚ) = 1 / x ∧ x = 20 := by
sorry

end solve_fraction_equation_l3723_372361


namespace junior_has_sixteen_rabbits_l3723_372376

/-- The number of toys bought on Monday -/
def monday_toys : ℕ := 6

/-- The number of toys bought on Wednesday -/
def wednesday_toys : ℕ := 2 * monday_toys

/-- The number of toys bought on Friday -/
def friday_toys : ℕ := 4 * monday_toys

/-- The number of toys bought on Saturday -/
def saturday_toys : ℕ := wednesday_toys / 2

/-- The total number of toys -/
def total_toys : ℕ := monday_toys + wednesday_toys + friday_toys + saturday_toys

/-- The number of toys each rabbit receives -/
def toys_per_rabbit : ℕ := 3

/-- The number of rabbits Junior has -/
def num_rabbits : ℕ := total_toys / toys_per_rabbit

theorem junior_has_sixteen_rabbits : num_rabbits = 16 := by
  sorry

end junior_has_sixteen_rabbits_l3723_372376


namespace product_expansion_l3723_372394

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * (7 / x^2 + 7*x - 7/x) = 3 / x^2 + 3*x - 3/x := by
  sorry

end product_expansion_l3723_372394


namespace function_properties_l3723_372327

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_deriv : ∀ x ∈ Set.Ioo 0 (π/2), f' x * Real.sin x - f x * Real.cos x > 0) :
  f (π/4) > -Real.sqrt 2 * f (-π/6) ∧ f (π/3) > Real.sqrt 3 * f (π/6) := by
  sorry

end function_properties_l3723_372327


namespace intersection_of_lines_l3723_372390

theorem intersection_of_lines :
  ∃! p : ℝ × ℝ, 
    5 * p.1 - 3 * p.2 = 15 ∧ 
    6 * p.1 + 2 * p.2 = 14 :=
by
  -- The proof goes here
  sorry

end intersection_of_lines_l3723_372390


namespace manoj_lending_amount_l3723_372315

/-- The amount Manoj borrowed from Anwar -/
def borrowed_amount : ℝ := 3900

/-- The interest rate for borrowing (as a decimal) -/
def borrowing_rate : ℝ := 0.06

/-- The interest rate for lending (as a decimal) -/
def lending_rate : ℝ := 0.09

/-- The duration of both borrowing and lending in years -/
def duration : ℝ := 3

/-- Manoj's total gain from the transaction -/
def total_gain : ℝ := 824.85

/-- Calculate simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The sum lent by Manoj to Ramu -/
def lent_amount : ℝ := 4355

theorem manoj_lending_amount :
  simple_interest lent_amount lending_rate duration -
  simple_interest borrowed_amount borrowing_rate duration =
  total_gain := by sorry

end manoj_lending_amount_l3723_372315


namespace same_function_constant_one_and_x_power_zero_l3723_372326

theorem same_function_constant_one_and_x_power_zero :
  ∀ x : ℝ, (1 : ℝ) = x^(0 : ℕ) :=
by sorry

end same_function_constant_one_and_x_power_zero_l3723_372326


namespace problem_solution_l3723_372318

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem problem_solution :
  let x := sum_integers 20 30
  let y := count_even_integers 20 30
  x + y = 281 → y = 6 := by
sorry

end problem_solution_l3723_372318


namespace factor_x8_minus_625_l3723_372377

theorem factor_x8_minus_625 (x : ℝ) : 
  x^8 - 625 = (x^4 + 25) * (x^2 + 5) * (x + Real.sqrt 5) * (x - Real.sqrt 5) := by
  sorry

end factor_x8_minus_625_l3723_372377


namespace tan_theta_range_l3723_372336

-- Define the condition
def condition (θ : ℝ) : Prop := (Real.sin θ) / (Real.sqrt 3 * Real.cos θ + 1) > 1

-- Define the range of tan θ
def tan_range (x : ℝ) : Prop := x ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 3 / 3) (Real.sqrt 2)

-- Theorem statement
theorem tan_theta_range (θ : ℝ) : condition θ → tan_range (Real.tan θ) := by sorry

end tan_theta_range_l3723_372336


namespace arrangements_with_restriction_l3723_372387

def num_actors : ℕ := 6

-- Define a function to calculate the number of arrangements
def num_arrangements (n : ℕ) (restricted_positions : ℕ) : ℕ :=
  (n - restricted_positions) * (n - 1).factorial

-- Theorem statement
theorem arrangements_with_restriction :
  num_arrangements num_actors 2 = 480 := by
  sorry

end arrangements_with_restriction_l3723_372387


namespace man_mass_from_boat_displacement_l3723_372347

/-- Calculates the mass of a man based on the displacement of a boat -/
theorem man_mass_from_boat_displacement (boat_length boat_breadth boat_sink_height water_density : Real) 
  (h1 : boat_length = 3)
  (h2 : boat_breadth = 2)
  (h3 : boat_sink_height = 0.01)
  (h4 : water_density = 1000) : 
  boat_length * boat_breadth * boat_sink_height * water_density = 60 := by
  sorry

#check man_mass_from_boat_displacement

end man_mass_from_boat_displacement_l3723_372347


namespace final_price_after_discounts_l3723_372338

/-- Given an original price p and two consecutive 10% discounts,
    the final selling price is 0.81p -/
theorem final_price_after_discounts (p : ℝ) : 
  let discount := 0.1
  let first_discount := p * (1 - discount)
  let second_discount := first_discount * (1 - discount)
  second_discount = 0.81 * p := by
sorry

end final_price_after_discounts_l3723_372338


namespace parabola_focus_vertex_distance_l3723_372332

theorem parabola_focus_vertex_distance (p : ℝ) (h_p : p > 0) : 
  let C : Set (ℝ × ℝ) := {(x, y) | y^2 = 2*p*x}
  let F : ℝ × ℝ := (p/2, 0)
  let l : Set (ℝ × ℝ) := {(x, y) | y = x - p/2}
  let chord_length : ℝ := 4
  let angle_with_axis : ℝ := π/4
  (∀ (x y : ℝ), (x, y) ∈ l → (x - F.1)^2 + (y - F.2)^2 = (x + F.1)^2 + (y - F.2)^2) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ C ∩ l ∧ (x₂, y₂) ∈ C ∩ l ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2) →
  (∀ (x y : ℝ), (x, y) ∈ l → y / x = Real.tan angle_with_axis) →
  F.1 = 1/2 := by
sorry

end parabola_focus_vertex_distance_l3723_372332


namespace inequality_requires_conditional_structure_l3723_372392

-- Define the types of algorithms
inductive Algorithm
  | SolveInequality
  | CalculateAverage
  | CalculateCircleArea
  | FindRoots

-- Define a function to check if an algorithm requires a conditional structure
def requiresConditionalStructure (alg : Algorithm) : Prop :=
  match alg with
  | Algorithm.SolveInequality => true
  | _ => false

-- Theorem statement
theorem inequality_requires_conditional_structure :
  requiresConditionalStructure Algorithm.SolveInequality ∧
  ¬requiresConditionalStructure Algorithm.CalculateAverage ∧
  ¬requiresConditionalStructure Algorithm.CalculateCircleArea ∧
  ¬requiresConditionalStructure Algorithm.FindRoots :=
sorry

end inequality_requires_conditional_structure_l3723_372392


namespace circle_plus_five_two_l3723_372320

-- Define the ⊕ operation
def circle_plus (a b : ℝ) : ℝ := 5 * a + 2 * b

-- State the theorem
theorem circle_plus_five_two : circle_plus 5 2 = 29 := by
  sorry

end circle_plus_five_two_l3723_372320


namespace mikes_total_payment_l3723_372353

/-- Calculates the amount Mike needs to pay after insurance coverage for his medical tests. -/
def mikes_payment (xray_cost : ℚ) (blood_test_cost : ℚ) : ℚ :=
  let mri_cost := 3 * xray_cost
  let ct_scan_cost := 2 * mri_cost
  let xray_payment := xray_cost * (1 - 0.8)
  let mri_payment := mri_cost * (1 - 0.8)
  let ct_scan_payment := ct_scan_cost * (1 - 0.7)
  let blood_test_payment := blood_test_cost * (1 - 0.5)
  xray_payment + mri_payment + ct_scan_payment + blood_test_payment

/-- Theorem stating that Mike's payment after insurance coverage is $750. -/
theorem mikes_total_payment :
  mikes_payment 250 200 = 750 := by
  sorry

end mikes_total_payment_l3723_372353


namespace fraction_ordering_l3723_372395

theorem fraction_ordering : 16/13 < 21/17 ∧ 21/17 < 20/15 := by sorry

end fraction_ordering_l3723_372395


namespace diophantine_equation_solution_l3723_372329

theorem diophantine_equation_solution (x y : ℤ) :
  x^2 - 5*y^2 = 1 →
  ∃ n : ℕ, (x + y * Real.sqrt 5 = (9 + 4 * Real.sqrt 5)^n) ∨
           (x + y * Real.sqrt 5 = -(9 + 4 * Real.sqrt 5)^n) := by
  sorry

end diophantine_equation_solution_l3723_372329


namespace sum_parity_from_cube_sum_parity_l3723_372360

theorem sum_parity_from_cube_sum_parity (n m : ℤ) (h : Even (n^3 + m^3)) : Even (n + m) := by
  sorry

end sum_parity_from_cube_sum_parity_l3723_372360


namespace project_completion_time_l3723_372374

theorem project_completion_time 
  (initial_team : ℕ) 
  (initial_work : ℚ) 
  (initial_time : ℕ) 
  (additional_members : ℕ) 
  (total_team : ℕ) :
  initial_team = 8 →
  initial_work = 1/3 →
  initial_time = 30 →
  additional_members = 4 →
  total_team = initial_team + additional_members →
  let work_efficiency := initial_work / (initial_team * initial_time)
  let remaining_work := 1 - initial_work
  let remaining_time := remaining_work / (total_team * work_efficiency)
  initial_time + remaining_time = 70 := by
sorry

end project_completion_time_l3723_372374


namespace range_of_m_solution_set_l3723_372306

-- Define the functions f and g
def f (x : ℝ) : ℝ := -abs (x - 2)
def g (x m : ℝ) : ℝ := -abs (x - 3) + m

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, f x > g x m) ↔ m < 1 :=
sorry

-- Theorem for the solution set of f(x) + a - 1 > 0
theorem solution_set (a : ℝ) :
  (∀ x : ℝ, f x + a - 1 > 0) ↔
    (a = 1 ∧ (∀ x : ℝ, x ≠ 2 → x ∈ Set.univ)) ∨
    (a > 1 ∧ (∀ x : ℝ, x ∈ Set.univ)) ∨
    (a < 1 ∧ (∀ x : ℝ, x < 1 + a ∨ x > 3 - a)) :=
sorry

end range_of_m_solution_set_l3723_372306


namespace problem_I_problem_II_l3723_372368

theorem problem_I (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin (π - α) - 2 * Real.cos (-α)) / (3 * Real.cos (π/2 - α) - 5 * Real.cos (π + α)) = 5/7 := by
  sorry

theorem problem_II (x : Real) (h1 : Real.sin x + Real.cos x = 1/5) (h2 : 0 < x) (h3 : x < π) :
  Real.sin x = 4/5 ∧ Real.cos x = -3/5 := by
  sorry

end problem_I_problem_II_l3723_372368


namespace board_block_system_l3723_372339

/-- A proof problem about forces and acceleration on a board and block system. -/
theorem board_block_system 
  (M : Real) (m : Real) (μ : Real) (g : Real) (a : Real)
  (hM : M = 4)
  (hm : m = 1)
  (hμ : μ = 0.2)
  (hg : g = 10)
  (ha : a = g / 5) :
  let T := m * (a + μ * g)
  let F := μ * g * (M + 2 * m) + M * a + T
  T = 4 ∧ F = 24 := by
  sorry


end board_block_system_l3723_372339


namespace sum_of_solutions_quadratic_l3723_372317

theorem sum_of_solutions_quadratic (x : ℝ) :
  let a : ℝ := -3
  let b : ℝ := -18
  let c : ℝ := 81
  let sum_of_roots := -b / a
  (a * x^2 + b * x + c = 0) → sum_of_roots = -6 := by
  sorry

end sum_of_solutions_quadratic_l3723_372317


namespace dumbbell_weight_l3723_372324

theorem dumbbell_weight (total_dumbbells : ℕ) (total_weight : ℕ) 
  (h1 : total_dumbbells = 6)
  (h2 : total_weight = 120) :
  total_weight / total_dumbbells = 20 := by
sorry

end dumbbell_weight_l3723_372324


namespace flour_salt_difference_l3723_372321

theorem flour_salt_difference (total_flour sugar total_salt flour_added : ℕ) : 
  total_flour = 12 → 
  sugar = 14 →
  total_salt = 7 → 
  flour_added = 2 → 
  (total_flour - flour_added) - total_salt = 3 := by
sorry

end flour_salt_difference_l3723_372321


namespace pineapple_problem_l3723_372369

/-- Calculates the number of rotten pineapples given the initial count, sold count, and remaining fresh count. -/
def rottenPineapples (initial sold fresh : ℕ) : ℕ :=
  initial - sold - fresh

/-- Theorem stating that given the specific conditions from the problem, 
    the number of rotten pineapples thrown away is 9. -/
theorem pineapple_problem : rottenPineapples 86 48 29 = 9 := by
  sorry

end pineapple_problem_l3723_372369


namespace austin_starting_amount_l3723_372385

def robot_cost : ℚ := 875 / 100
def discount_rate : ℚ := 1 / 10
def coupon_discount : ℚ := 5
def tax_rate : ℚ := 2 / 25
def total_tax : ℚ := 722 / 100
def shipping_fee : ℚ := 499 / 100
def gift_card : ℚ := 25
def change : ℚ := 1153 / 100

def total_robots : ℕ := 2 * 1 + 3 * 2 + 2 * 3

theorem austin_starting_amount (initial_amount : ℚ) :
  (∃ (discounted_price : ℚ),
    discounted_price = total_robots * robot_cost * (1 - discount_rate) - coupon_discount ∧
    total_tax = discounted_price * tax_rate ∧
    initial_amount = discounted_price + total_tax + shipping_fee - gift_card + change) →
  initial_amount = 7746 / 100 :=
by sorry

end austin_starting_amount_l3723_372385


namespace perpendicular_condition_l3723_372314

def is_perpendicular (a : ℝ) : Prop :=
  (a ≠ -1 ∧ a ≠ 0 ∧ -(a + 1) / (3 * a) * ((1 - a) / (a + 1)) = -1) ∨
  (a = -1)

theorem perpendicular_condition (a : ℝ) :
  (a = 1/4 → is_perpendicular a) ∧
  ¬(is_perpendicular a → a = 1/4) :=
sorry

end perpendicular_condition_l3723_372314


namespace gabled_cuboid_theorem_l3723_372348

/-- Represents a cuboid with gable-shaped figures on each face -/
structure GabledCuboid where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_bc : b > c

/-- Properties of the gabled cuboid -/
def GabledCuboidProperties (g : GabledCuboid) : Prop :=
  ∃ (num_faces num_edges num_vertices : ℕ) (volume : ℝ),
    num_faces = 12 ∧
    num_edges = 30 ∧
    num_vertices = 20 ∧
    volume = g.a * g.b * g.c + (1/2) * (g.a * g.b^2 + g.a * g.c^2 + g.b * g.c^2) - g.b^3/6 - g.c^3/3

theorem gabled_cuboid_theorem (g : GabledCuboid) : GabledCuboidProperties g := by
  sorry

end gabled_cuboid_theorem_l3723_372348


namespace product_of_powers_equals_sum_l3723_372366

theorem product_of_powers_equals_sum (w x y z k : ℕ) : 
  2^w * 3^x * 5^y * 7^z * 11^k = 900 → 2*w + 3*x + 5*y + 7*z + 11*k = 20 := by
  sorry

end product_of_powers_equals_sum_l3723_372366


namespace unique_solution_mod_30_l3723_372355

theorem unique_solution_mod_30 : 
  ∃! x : ℕ, x < 30 ∧ (x^4 + 2*x^3 + 3*x^2 - x + 1) % 30 = 0 :=
by sorry

end unique_solution_mod_30_l3723_372355


namespace sum_of_consecutive_primes_has_three_prime_factors_l3723_372351

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def consecutive_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p < q ∧ ∀ n, p < n → n < q → ¬(is_prime n)

theorem sum_of_consecutive_primes_has_three_prime_factors (p q : ℕ) :
  p > 2 → q > 2 → consecutive_primes p q →
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ p + q = a * b * c :=
sorry

end sum_of_consecutive_primes_has_three_prime_factors_l3723_372351


namespace sarahs_initial_trucks_l3723_372300

/-- Given that Sarah gave away 13 trucks and has 38 trucks remaining,
    prove that she initially had 51 trucks. -/
theorem sarahs_initial_trucks :
  ∀ (initial_trucks given_trucks remaining_trucks : ℕ),
    given_trucks = 13 →
    remaining_trucks = 38 →
    initial_trucks = given_trucks + remaining_trucks →
    initial_trucks = 51 :=
by sorry

end sarahs_initial_trucks_l3723_372300


namespace locus_of_points_l3723_372370

/-- Given two parallel lines e₁ and e₂ in the plane, separated by a distance 2g,
    and a perpendicular line f intersecting them at O₁ and O₂ respectively,
    this theorem characterizes the locus of points P(x, y) such that a line through P
    intersects e₁ at P₁ and e₂ at P₂ with O₁P₁ · O₂P₂ = k. -/
theorem locus_of_points (g : ℝ) (k : ℝ) (x y : ℝ) :
  k = 1 → (y^2 / g^2 ≥ 1 - x^2) ∧
  k = -1 → (y^2 / g^2 ≤ 1 + x^2) := by
  sorry


end locus_of_points_l3723_372370


namespace range_of_m_l3723_372384

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x^2 / (2 - m) + y^2 / (m - 1) = 1 → m > 2) →
  (∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≠ 0) →
  ((m > 2) ∨ (1 < m ∧ m < 3)) →
  ¬(1 < m ∧ m < 3) →
  m ≥ 3 :=
by sorry

end range_of_m_l3723_372384


namespace smallest_class_size_l3723_372340

/-- Represents a class of students and their test scores. -/
structure TestClass where
  n : ℕ              -- Total number of students
  scores : Fin n → ℕ -- Scores of each student

/-- Conditions for the test class. -/
def validTestClass (c : TestClass) : Prop :=
  (∀ i, c.scores i ≥ 70 ∧ c.scores i ≤ 120) ∧
  (∃ s : Finset (Fin c.n), s.card = 7 ∧ ∀ i ∈ s, c.scores i = 120) ∧
  (Finset.sum (Finset.univ : Finset (Fin c.n)) c.scores / c.n = 85)

/-- The theorem stating the smallest possible number of students. -/
theorem smallest_class_size :
  ∀ c : TestClass, validTestClass c → c.n ≥ 24 :=
sorry

end smallest_class_size_l3723_372340


namespace orange_cost_l3723_372308

/-- Given that 4 dozen oranges cost $28.80, prove that 5 dozen oranges at the same rate cost $36.00 -/
theorem orange_cost (cost_four_dozen : ℝ) (h1 : cost_four_dozen = 28.80) :
  let cost_per_dozen : ℝ := cost_four_dozen / 4
  let cost_five_dozen : ℝ := 5 * cost_per_dozen
  cost_five_dozen = 36 :=
by sorry

end orange_cost_l3723_372308


namespace difference_of_squares_l3723_372399

theorem difference_of_squares (x y : ℝ) : 
  x > 0 → y > 0 → x < y → 
  Real.sqrt x + Real.sqrt y = 1 → 
  Real.sqrt (x / y) + Real.sqrt (y / x) = 10 / 3 → 
  y - x = 1 / 2 := by
sorry

end difference_of_squares_l3723_372399


namespace polynomial_division_l3723_372367

-- Define the dividend polynomial
def dividend (z : ℚ) : ℚ := 4*z^5 - 9*z^4 + 7*z^3 - 12*z^2 + 8*z - 3

-- Define the divisor polynomial
def divisor (z : ℚ) : ℚ := 2*z + 3

-- Define the quotient polynomial
def quotient (z : ℚ) : ℚ := 2*z^4 - 5*z^3 + 4*z^2 - (5/2)*z + 3/4

-- State the theorem
theorem polynomial_division :
  ∀ z : ℚ, dividend z / divisor z = quotient z :=
by sorry

end polynomial_division_l3723_372367


namespace sparklers_burn_time_l3723_372382

/-- The number of sparklers -/
def num_sparklers : ℕ := 10

/-- The time it takes for one sparkler to burn down completely (in minutes) -/
def burn_time : ℚ := 2

/-- The fraction of time left when the next sparkler is lit -/
def fraction_left : ℚ := 1/10

/-- The time each sparkler burns before the next one is lit -/
def individual_burn_time : ℚ := burn_time * (1 - fraction_left)

/-- The total time for all sparklers to burn down (in minutes) -/
def total_burn_time : ℚ := (num_sparklers - 1) * individual_burn_time + burn_time

/-- Conversion function from minutes to minutes and seconds -/
def to_minutes_and_seconds (time : ℚ) : ℕ × ℕ :=
  let minutes := time.floor
  let seconds := ((time - minutes) * 60).floor
  (minutes.toNat, seconds.toNat)

theorem sparklers_burn_time :
  to_minutes_and_seconds total_burn_time = (18, 12) :=
sorry

end sparklers_burn_time_l3723_372382


namespace shaded_region_characterization_l3723_372383

def shaded_region (z : ℂ) : Prop :=
  Complex.abs z ≤ 1 ∧ Complex.im z ≥ (1/2 : ℝ)

theorem shaded_region_characterization :
  ∀ z : ℂ, z ∈ {z | shaded_region z} ↔ 
    (Complex.abs z ≤ 1 ∧ Complex.im z ≥ (1/2 : ℝ)) :=
by sorry

end shaded_region_characterization_l3723_372383


namespace a_plus_b_squared_l3723_372386

theorem a_plus_b_squared (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 2) (h3 : a < b) :
  (a + b)^2 = 1 ∨ (a + b)^2 = 25 := by
  sorry

end a_plus_b_squared_l3723_372386


namespace divisibility_problem_l3723_372323

theorem divisibility_problem (p : ℕ) (hp : p.Prime) (hp_gt_7 : p > 7) 
  (hp_mod_6 : p % 6 = 1) (m : ℕ) (hm : m = 2^p - 1) : 
  (127 * m) ∣ (2^(m-1) - 1) := by
sorry

end divisibility_problem_l3723_372323


namespace complex_equation_solution_l3723_372381

theorem complex_equation_solution (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a - 2 * i) * i = b - i) : a + b * i = -1 + 2 * i :=
sorry

end complex_equation_solution_l3723_372381


namespace fraction_simplification_l3723_372341

theorem fraction_simplification : (2020 : ℚ) / (20 * 20) = 5.05 := by sorry

end fraction_simplification_l3723_372341


namespace sum_of_variables_l3723_372396

theorem sum_of_variables (x y z : ℝ) 
  (eq1 : y + z = 18 - 4*x)
  (eq2 : x + z = 22 - 4*y)
  (eq3 : x + y = 15 - 4*z) :
  3*x + 3*y + 3*z = 55/2 := by
sorry

end sum_of_variables_l3723_372396


namespace equality_implications_l3723_372362

theorem equality_implications (a b x y : ℝ) (h : a = b) : 
  (a - 3 = b - 3) ∧ 
  (3 * a = 3 * b) ∧ 
  ((a + 3) / 4 = (b + 3) / 4) ∧
  (∃ x y, a * x ≠ b * y) := by
sorry

end equality_implications_l3723_372362


namespace min_value_a_over_b_l3723_372343

theorem min_value_a_over_b (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (h : 2 * Real.sqrt a + b = 1) :
  ∃ (x : ℝ), x = a / b ∧ ∀ (y : ℝ), y = a / b → x ≤ y ∧ x = 0 :=
by sorry

end min_value_a_over_b_l3723_372343


namespace quadratic_equation_solution_l3723_372354

theorem quadratic_equation_solution : 
  ∀ x : ℝ, (x - 2)^2 = 2*x - 4 ↔ x = 2 ∨ x = 4 := by sorry

end quadratic_equation_solution_l3723_372354


namespace s_5_value_l3723_372307

/-- s(n) is a number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ :=
  sorry

/-- The theorem states that s(5) equals 1491625 -/
theorem s_5_value : s 5 = 1491625 :=
  sorry

end s_5_value_l3723_372307


namespace median_of_consecutive_integers_l3723_372398

theorem median_of_consecutive_integers (n : ℕ) (sum : ℕ) (h1 : n = 36) (h2 : sum = 1296) :
  sum / n = 36 := by
  sorry

end median_of_consecutive_integers_l3723_372398


namespace only_group_D_forms_triangle_l3723_372316

/-- Triangle inequality theorem for a set of three lengths -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Groups of line segments -/
def group_A : (ℝ × ℝ × ℝ) := (3, 8, 5)
def group_B : (ℝ × ℝ × ℝ) := (12, 5, 6)
def group_C : (ℝ × ℝ × ℝ) := (5, 5, 10)
def group_D : (ℝ × ℝ × ℝ) := (15, 10, 7)

/-- Theorem: Only group D can form a triangle -/
theorem only_group_D_forms_triangle :
  ¬(triangle_inequality group_A.1 group_A.2.1 group_A.2.2) ∧
  ¬(triangle_inequality group_B.1 group_B.2.1 group_B.2.2) ∧
  ¬(triangle_inequality group_C.1 group_C.2.1 group_C.2.2) ∧
  (triangle_inequality group_D.1 group_D.2.1 group_D.2.2) :=
by sorry

end only_group_D_forms_triangle_l3723_372316


namespace solve_cab_driver_income_l3723_372302

def cab_driver_income_problem (day1 day2 day3 day5 average : ℕ) : Prop :=
  let total := 5 * average
  let known_sum := day1 + day2 + day3 + day5
  let day4 := total - known_sum
  (day1 = 45 ∧ day2 = 50 ∧ day3 = 60 ∧ day5 = 70 ∧ average = 58) →
  day4 = 65

theorem solve_cab_driver_income :
  cab_driver_income_problem 45 50 60 70 58 :=
sorry

end solve_cab_driver_income_l3723_372302


namespace new_student_weight_l3723_372364

/-- Calculates the weight of a new student given the initial and final conditions of a group of students. -/
theorem new_student_weight
  (initial_count : ℕ)
  (initial_avg : ℝ)
  (final_count : ℕ)
  (final_avg : ℝ)
  (h1 : initial_count = 19)
  (h2 : initial_avg = 15)
  (h3 : final_count = initial_count + 1)
  (h4 : final_avg = 14.6) :
  final_count * final_avg - initial_count * initial_avg = 7 :=
by sorry

end new_student_weight_l3723_372364


namespace historical_fiction_new_releases_l3723_372363

theorem historical_fiction_new_releases 
  (total_inventory : ℝ)
  (historical_fiction_ratio : ℝ)
  (historical_fiction_new_release_ratio : ℝ)
  (other_new_release_ratio : ℝ)
  (h1 : historical_fiction_ratio = 0.3)
  (h2 : historical_fiction_new_release_ratio = 0.3)
  (h3 : other_new_release_ratio = 0.4)
  (h4 : total_inventory > 0) :
  let historical_fiction := total_inventory * historical_fiction_ratio
  let other_books := total_inventory * (1 - historical_fiction_ratio)
  let historical_fiction_new_releases := historical_fiction * historical_fiction_new_release_ratio
  let other_new_releases := other_books * other_new_release_ratio
  let total_new_releases := historical_fiction_new_releases + other_new_releases
  historical_fiction_new_releases / total_new_releases = 9 / 37 := by
    sorry

end historical_fiction_new_releases_l3723_372363


namespace gel_pen_price_l3723_372313

theorem gel_pen_price (x y b g : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : b > 0) (h4 : g > 0) : 
  ((x + y) * g = 4 * (x * b + y * g)) → 
  ((x + y) * b = (1/2) * (x * b + y * g)) → 
  g = 8 * b :=
by sorry

end gel_pen_price_l3723_372313


namespace terminal_side_point_y_value_l3723_372334

theorem terminal_side_point_y_value (α : Real) (y : Real) :
  let P : Real × Real := (-Real.sqrt 3, y)
  (P.1^2 + P.2^2 ≠ 0) →  -- Ensure the point is not at the origin
  (Real.sin α = Real.sqrt 13 / 13) →
  y = 1 / 2 := by
sorry

end terminal_side_point_y_value_l3723_372334


namespace g_formula_and_domain_intersection_points_l3723_372342

noncomputable section

-- Define the original function f
def f (x : ℝ) : ℝ := x + 1/x

-- Define the domain of f
def f_domain : Set ℝ := {x | x < 0 ∨ x > 0}

-- Define the symmetric function g
def g (x : ℝ) : ℝ := x - 2 + 1/(x-4)

-- Define the domain of g
def g_domain : Set ℝ := {x | x < 4 ∨ x > 4}

-- Define the symmetry point
def A : ℝ × ℝ := (2, 1)

-- Theorem for the correct formula and domain of g
theorem g_formula_and_domain :
  (∀ x ∈ g_domain, g x = x - 2 + 1/(x-4)) ∧
  (∀ x, x ∈ g_domain ↔ x < 4 ∨ x > 4) :=
sorry

-- Theorem for the intersection points
theorem intersection_points :
  (∀ b : ℝ, (∃! x, g x = b) ↔ b = 4 ∨ b = 0) ∧
  (g 5 = 4 ∧ g 3 = 0) :=
sorry

end

end g_formula_and_domain_intersection_points_l3723_372342


namespace regression_line_equation_l3723_372322

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear equation in the form y = mx + b -/
structure LinearEquation where
  slope : ℝ
  intercept : ℝ

/-- Check if a point lies on a given linear equation -/
def pointOnLine (p : Point) (eq : LinearEquation) : Prop :=
  p.y = eq.slope * p.x + eq.intercept

/-- The theorem to be proved -/
theorem regression_line_equation 
  (slope : ℝ) 
  (center : Point) 
  (h_slope : slope = 1.23)
  (h_center : center = ⟨4, 5⟩) :
  ∃ (eq : LinearEquation), 
    eq.slope = slope ∧ 
    pointOnLine center eq ∧ 
    eq = ⟨1.23, 0.08⟩ := by
  sorry

end regression_line_equation_l3723_372322


namespace line_point_x_coordinate_l3723_372301

/-- Given a line passing through (10, 3) with x-intercept 4,
    the x-coordinate of the point on this line with y-coordinate -3 is -2. -/
theorem line_point_x_coordinate 
  (line : ℝ → ℝ) 
  (passes_through_10_3 : line 10 = 3)
  (x_intercept_4 : line 4 = 0) :
  ∃ x : ℝ, line x = -3 ∧ x = -2 := by
  sorry

end line_point_x_coordinate_l3723_372301


namespace cricket_team_throwers_l3723_372372

theorem cricket_team_throwers (total_players : ℕ) (right_handed : ℕ) : 
  total_players = 61 → right_handed = 53 → ∃ (throwers : ℕ), 
    throwers = 37 ∧ 
    throwers ≤ right_handed ∧
    throwers ≤ total_players ∧
    3 * (right_handed - throwers) = 2 * (total_players - throwers) := by
  sorry

end cricket_team_throwers_l3723_372372


namespace complex_in_first_quadrant_l3723_372393

theorem complex_in_first_quadrant (a : ℝ) : 
  (((1 : ℂ) + a * Complex.I) / ((2 : ℂ) - Complex.I)).re > 0 ∧
  (((1 : ℂ) + a * Complex.I) / ((2 : ℂ) - Complex.I)).im > 0 →
  -1/2 < a ∧ a < 2 := by
sorry

end complex_in_first_quadrant_l3723_372393


namespace max_value_of_symmetric_f_l3723_372379

/-- A function f that is symmetric about x = -2 and has the form (1-x^2)(x^2+ax+b) -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

/-- The symmetry condition for f about x = -2 -/
def symmetric_about_neg_two (a b : ℝ) : Prop :=
  ∀ t, f a b (-2 + t) = f a b (-2 - t)

/-- The theorem stating that if f is symmetric about x = -2, its maximum value is 16 -/
theorem max_value_of_symmetric_f (a b : ℝ) 
  (h : symmetric_about_neg_two a b) : 
  ∃ x, f a b x = 16 ∧ ∀ y, f a b y ≤ 16 := by
  sorry

end max_value_of_symmetric_f_l3723_372379


namespace at_least_13_blondes_identifiable_l3723_372310

/-- Represents a woman in the factory -/
inductive Woman
| Blonde
| Brunette

/-- The total number of women in the factory -/
def total_women : ℕ := 217

/-- The number of brunettes in the factory -/
def num_brunettes : ℕ := 17

/-- The number of blondes in the factory -/
def num_blondes : ℕ := 200

/-- The number of women each woman lists as blonde -/
def list_size : ℕ := 200

/-- A function representing a woman's list of supposed blondes -/
def list_blondes (w : Woman) : Finset Woman := sorry

theorem at_least_13_blondes_identifiable :
  ∃ (identified_blondes : Finset Woman),
    (∀ w ∈ identified_blondes, w = Woman.Blonde) ∧
    identified_blondes.card ≥ 13 := by sorry

end at_least_13_blondes_identifiable_l3723_372310


namespace min_distance_sum_parabola_line_l3723_372344

/-- The minimum distance sum from a point on the parabola y² = -4x to the y-axis and the line 2x + y - 4 = 0 -/
theorem min_distance_sum_parabola_line : 
  ∃ (min_sum : ℝ), 
    min_sum = (6 * Real.sqrt 5) / 5 - 1 ∧
    ∀ (x y : ℝ),
      y^2 = -4*x →  -- point (x,y) is on the parabola
      ∃ (m n : ℝ),
        m = |x| ∧   -- distance to y-axis
        n = |2*x + y - 4| / Real.sqrt 5 ∧  -- distance to line
        m + n ≥ min_sum :=
by sorry

end min_distance_sum_parabola_line_l3723_372344


namespace quadrilateral_area_inequality_l3723_372325

-- Define a quadrilateral structure
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  area : ℝ
  a_positive : 0 < a
  b_positive : 0 < b
  c_positive : 0 < c
  d_positive : 0 < d
  area_positive : 0 < area

-- State the theorem
theorem quadrilateral_area_inequality (q : Quadrilateral) : 2 * q.area ≤ q.a * q.c + q.b * q.d := by
  sorry

end quadrilateral_area_inequality_l3723_372325


namespace maddie_spent_95_l3723_372388

/-- Calculates the total amount spent on T-shirts with a bulk discount -/
def total_spent (white_packs blue_packs : ℕ) 
                (white_per_pack blue_per_pack : ℕ) 
                (white_price blue_price : ℚ) 
                (discount_percent : ℚ) : ℚ :=
  let white_total := white_packs * white_per_pack * white_price
  let blue_total := blue_packs * blue_per_pack * blue_price
  let subtotal := white_total + blue_total
  let discount := subtotal * (discount_percent / 100)
  subtotal - discount

/-- Proves that Maddie spent $95 on T-shirts -/
theorem maddie_spent_95 : 
  total_spent 2 4 5 3 4 5 5 = 95 := by
  sorry

end maddie_spent_95_l3723_372388


namespace no_extreme_points_iff_l3723_372345

/-- The function f(x) defined as ax³ + ax² + 7x -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + a * x^2 + 7 * x

/-- A function has no extreme points if its derivative is always non-negative or always non-positive -/
def has_no_extreme_points (g : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, (deriv g) x ≥ 0) ∨ (∀ x : ℝ, (deriv g) x ≤ 0)

/-- The main theorem: f(x) has no extreme points if and only if 0 ≤ a ≤ 21 -/
theorem no_extreme_points_iff (a : ℝ) :
  has_no_extreme_points (f a) ↔ 0 ≤ a ∧ a ≤ 21 := by sorry

end no_extreme_points_iff_l3723_372345


namespace parking_duration_for_5_5_yuan_l3723_372389

/-- Calculates the parking duration given the total fee paid -/
def parking_duration (total_fee : ℚ) : ℚ :=
  (total_fee - 0.5) / (0.5 + 0.5) + 1

/-- Theorem stating that given the specific fee paid, the parking duration is 6 hours -/
theorem parking_duration_for_5_5_yuan :
  parking_duration 5.5 = 6 := by sorry

end parking_duration_for_5_5_yuan_l3723_372389


namespace angle_D_measure_l3723_372373

structure CyclicQuadrilateral where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_360 : A + B + C + D = 360
  ratio_ABC : ∃ (x : ℝ), A = 3*x ∧ B = 4*x ∧ C = 6*x

theorem angle_D_measure (q : CyclicQuadrilateral) : q.D = 100 := by
  sorry

end angle_D_measure_l3723_372373


namespace arithmetic_sequence_sum_l3723_372309

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 5 + a 7 = ∫ x in (0 : ℝ)..2, |1 - x^2|) →
  a 4 + a 6 + a 8 = 3 := by
  sorry

end arithmetic_sequence_sum_l3723_372309


namespace bus_car_speed_equation_l3723_372337

theorem bus_car_speed_equation (x : ℝ) (h1 : x > 0) : 
  (20 / x - 20 / (1.5 * x) = 1 / 6) ↔ 
  (20 / x = 20 / (1.5 * x) + 1 / 6) := by sorry

end bus_car_speed_equation_l3723_372337


namespace male_athletes_to_sample_l3723_372305

def total_athletes : ℕ := 98
def female_athletes : ℕ := 42
def selection_probability : ℚ := 2/7

def male_athletes : ℕ := total_athletes - female_athletes

theorem male_athletes_to_sample :
  ⌊(male_athletes : ℚ) * selection_probability⌋ = 16 := by
  sorry

end male_athletes_to_sample_l3723_372305


namespace take_home_pay_calculation_l3723_372328

/-- Calculate take-home pay after deductions -/
theorem take_home_pay_calculation (total_pay : ℝ) 
  (tax_rate insurance_rate pension_rate union_rate : ℝ) :
  total_pay = 500 →
  tax_rate = 0.10 →
  insurance_rate = 0.05 →
  pension_rate = 0.03 →
  union_rate = 0.02 →
  total_pay * (1 - (tax_rate + insurance_rate + pension_rate + union_rate)) = 400 := by
  sorry

end take_home_pay_calculation_l3723_372328


namespace group_size_l3723_372304

theorem group_size (n : ℕ) 
  (avg_increase : ℝ) 
  (old_weight new_weight : ℝ) 
  (h1 : avg_increase = 1.5)
  (h2 : old_weight = 65)
  (h3 : new_weight = 74)
  (h4 : n * avg_increase = new_weight - old_weight) : 
  n = 6 := by
sorry

end group_size_l3723_372304


namespace identity_function_divisibility_l3723_372375

theorem identity_function_divisibility (f : ℕ+ → ℕ+) :
  (∀ (a b : ℕ+), (a.val ^ 2 + (f a).val * (f b).val) % ((f a).val + b.val) = 0) →
  (∀ (n : ℕ+), f n = n) :=
by sorry

end identity_function_divisibility_l3723_372375


namespace kerosene_cost_friday_l3723_372371

/-- The cost of a liter of kerosene on Friday given the market conditions --/
theorem kerosene_cost_friday (rice_cost_monday : ℝ) 
  (h1 : rice_cost_monday = 0.36)
  (h2 : ∀ x, x > 0 → x * 12 * rice_cost_monday = x * 8 * (0.5 * rice_cost_monday))
  (h3 : ∀ x, x > 0 → 1.2 * x * rice_cost_monday = x * 1.2 * rice_cost_monday) :
  ∃ (kerosene_cost_friday : ℝ), kerosene_cost_friday = 0.576 :=
by sorry

end kerosene_cost_friday_l3723_372371


namespace power_of_power_equals_ten_l3723_372380

theorem power_of_power_equals_ten (m : ℝ) : (m^2)^5 = m^10 := by
  sorry

end power_of_power_equals_ten_l3723_372380


namespace line_through_P_with_opposite_sign_intercepts_l3723_372346

-- Define the point P
def P : ℝ × ℝ := (3, -2)

-- Define the line equation types
inductive LineEquation
| Standard (a b c : ℝ) : LineEquation  -- ax + by + c = 0
| SlopeIntercept (m b : ℝ) : LineEquation  -- y = mx + b

-- Define a predicate for a line passing through a point
def passesThrough (eq : LineEquation) (p : ℝ × ℝ) : Prop :=
  match eq with
  | LineEquation.Standard a b c => a * p.1 + b * p.2 + c = 0
  | LineEquation.SlopeIntercept m b => p.2 = m * p.1 + b

-- Define a predicate for a line having intercepts of opposite signs
def hasOppositeSignIntercepts (eq : LineEquation) : Prop :=
  match eq with
  | LineEquation.Standard a b c =>
    (c / a) * (c / b) < 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
  | LineEquation.SlopeIntercept m b =>
    b * (b / m) < 0 ∧ m ≠ 0 ∧ b ≠ 0

-- The main theorem
theorem line_through_P_with_opposite_sign_intercepts :
  ∃ (eq : LineEquation),
    (eq = LineEquation.Standard 1 (-1) (-5) ∨ eq = LineEquation.SlopeIntercept (-2/3) 0) ∧
    passesThrough eq P ∧
    hasOppositeSignIntercepts eq :=
  sorry

end line_through_P_with_opposite_sign_intercepts_l3723_372346
