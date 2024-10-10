import Mathlib

namespace boys_insects_count_l362_36258

/-- The number of groups in the class -/
def num_groups : ℕ := 4

/-- The number of insects each group receives -/
def insects_per_group : ℕ := 125

/-- The number of insects collected by the girls -/
def girls_insects : ℕ := 300

/-- The number of insects collected by the boys -/
def boys_insects : ℕ := num_groups * insects_per_group - girls_insects

theorem boys_insects_count :
  boys_insects = 200 := by sorry

end boys_insects_count_l362_36258


namespace mean_equality_problem_l362_36281

theorem mean_equality_problem (x y : ℚ) : 
  (((7 : ℚ) + 11 + 19 + 23) / 4 = (14 + x + y) / 3) →
  x = 2 * y →
  x = 62 / 3 ∧ y = 31 / 3 := by
  sorry

end mean_equality_problem_l362_36281


namespace sphere_cylinder_surface_area_difference_l362_36220

/-- The difference between the surface area of a sphere and the lateral surface area of its inscribed cylinder is zero. -/
theorem sphere_cylinder_surface_area_difference (R : ℝ) (R_pos : R > 0) : 
  4 * Real.pi * R^2 - (2 * Real.pi * R * (2 * R)) = 0 := by
  sorry

end sphere_cylinder_surface_area_difference_l362_36220


namespace quadratic_minimum_l362_36254

/-- The function f(x) = x^2 + 6x + 13 has a minimum value of 4 -/
theorem quadratic_minimum (x : ℝ) : ∀ y : ℝ, x^2 + 6*x + 13 ≥ 4 := by
  sorry

end quadratic_minimum_l362_36254


namespace termite_ridden_homes_l362_36286

theorem termite_ridden_homes (total_homes : ℝ) (termite_ridden : ℝ) 
  (h1 : termite_ridden > 0) 
  (h2 : termite_ridden / total_homes ≤ 1) 
  (h3 : (3/4) * (termite_ridden / total_homes) = 1/4) : 
  termite_ridden / total_homes = 1/3 := by
sorry

end termite_ridden_homes_l362_36286


namespace expression_value_at_three_l362_36227

theorem expression_value_at_three : 
  let x : ℝ := 3
  (x^8 + 24*x^4 + 144) / (x^4 + 12) = 93 := by
sorry

end expression_value_at_three_l362_36227


namespace impossibility_of_identical_remainders_l362_36260

theorem impossibility_of_identical_remainders :
  ¬ ∃ (a : Fin 100 → ℕ) (r : ℕ),
    r ≠ 0 ∧
    ∀ i : Fin 100, a i % a (i.succ) = r :=
sorry

end impossibility_of_identical_remainders_l362_36260


namespace geometric_seq_arithmetic_property_l362_36294

/-- Given a geometric sequence with common ratio q, prove that if the sums S_m, S_n, and S_l
    form an arithmetic sequence, then for any natural number k, the terms a_{m+k}, a_{n+k},
    and a_{l+k} also form an arithmetic sequence. -/
theorem geometric_seq_arithmetic_property
  (a : ℝ) (q : ℝ) (m n l k : ℕ) :
  let a_seq : ℕ → ℝ := λ i => a * q ^ (i - 1)
  let S : ℕ → ℝ := λ i => if q = 1 then a * i else a * (1 - q^i) / (1 - q)
  (2 * S n = S m + S l) →
  2 * a_seq (n + k) = a_seq (m + k) + a_seq (l + k) :=
by sorry

end geometric_seq_arithmetic_property_l362_36294


namespace partial_fraction_decomposition_product_l362_36275

theorem partial_fraction_decomposition_product (M₁ M₂ : ℝ) :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ 3 → 
    (42 * x - 53) / (x^2 - 4*x + 3) = M₁ / (x - 1) + M₂ / (x - 3)) →
  M₁ * M₂ = 200.75 := by
sorry

end partial_fraction_decomposition_product_l362_36275


namespace max_d_value_l362_36250

def is_valid_number (d e : ℕ) : Prop :=
  d < 10 ∧ e < 10 ∧ e % 2 = 0

def number_value (d e : ℕ) : ℕ :=
  505220 + d * 1000 + e

theorem max_d_value :
  ∃ (d : ℕ), d = 8 ∧
  ∀ (d' e : ℕ), is_valid_number d' e →
  number_value d' e % 22 = 0 →
  d' ≤ d :=
sorry

end max_d_value_l362_36250


namespace incorrect_statement_about_real_square_roots_l362_36276

theorem incorrect_statement_about_real_square_roots :
  ¬ (∀ a b : ℝ, a < b ∧ b < 0 → ¬∃ x y : ℝ, x^2 = a ∧ y^2 = b) :=
sorry

end incorrect_statement_about_real_square_roots_l362_36276


namespace thirteen_people_in_line_l362_36261

/-- The number of people in line at an amusement park ride -/
def people_in_line (people_in_front : ℕ) (position_from_back : ℕ) : ℕ :=
  people_in_front + 1 + (position_from_back - 1)

/-- Theorem stating that there are 13 people in line given the conditions -/
theorem thirteen_people_in_line :
  people_in_line 7 6 = 13 := by
  sorry

end thirteen_people_in_line_l362_36261


namespace min_value_of_fraction_l362_36274

theorem min_value_of_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (2 / a + 1 / b) ≥ 9 := by
  sorry

end min_value_of_fraction_l362_36274


namespace megan_popsicle_consumption_l362_36209

/-- The number of popsicles Megan eats in a given time period -/
def popsicles_eaten (minutes_per_popsicle : ℕ) (total_minutes : ℕ) : ℕ :=
  (total_minutes / minutes_per_popsicle : ℕ)

/-- Converts hours and minutes to total minutes -/
def to_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * 60 + minutes

theorem megan_popsicle_consumption :
  popsicles_eaten 12 (to_minutes 6 45) = 33 := by
  sorry

end megan_popsicle_consumption_l362_36209


namespace star_composition_l362_36297

-- Define the * operation
def star (A B : Set α) : Set α := {x | x ∈ A ∧ x ∉ B}

-- Theorem statement
theorem star_composition (A B : Set α) : star A (star A B) = A ∩ B := by
  sorry

end star_composition_l362_36297


namespace original_houses_l362_36272

/-- The number of houses built during the housing boom in Lincoln County. -/
def houses_built : ℕ := 97741

/-- The current total number of houses in Lincoln County. -/
def current_total : ℕ := 118558

/-- Theorem stating that the original number of houses in Lincoln County is 20817. -/
theorem original_houses : current_total - houses_built = 20817 := by
  sorry

end original_houses_l362_36272


namespace solution_set_when_a_is_1_range_of_a_for_inequality_l362_36238

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 2|

-- Part 1
theorem solution_set_when_a_is_1 :
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≤ -3 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_for_inequality :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 1 → f a x ≤ |x + 4|) ↔ a ∈ Set.Icc (-1) 2 := by sorry

end solution_set_when_a_is_1_range_of_a_for_inequality_l362_36238


namespace max_value_sqrt_sum_l362_36285

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 7)
  (x_ge_2 : x ≥ 2)
  (y_ge_2 : y ≥ 2)
  (z_ge_2 : z ≥ 2) :
  ∃ (max : ℝ), max = Real.sqrt 69 ∧ 
    ∀ a b c : ℝ, a + b + c = 7 → a ≥ 2 → b ≥ 2 → c ≥ 2 →
      Real.sqrt (2 * a + 3) + Real.sqrt (2 * b + 3) + Real.sqrt (2 * c + 3) ≤ max :=
by sorry

end max_value_sqrt_sum_l362_36285


namespace trains_clearing_time_l362_36255

/-- Calculates the time for two trains to clear each other -/
theorem trains_clearing_time (length1 length2 speed1 speed2 : ℝ) : 
  length1 = 160 ∧ 
  length2 = 320 ∧ 
  speed1 = 42 ∧ 
  speed2 = 30 → 
  (length1 + length2) / ((speed1 + speed2) * (1000 / 3600)) = 24 := by
  sorry

#check trains_clearing_time

end trains_clearing_time_l362_36255


namespace cone_slant_height_l362_36251

/-- Given a cone with base radius 1 and lateral surface that unfolds into a semicircle,
    prove that its slant height is 2. -/
theorem cone_slant_height (r : ℝ) (s : ℝ) (h1 : r = 1) (h2 : π * s = 2 * π * r) : s = 2 := by
  sorry

end cone_slant_height_l362_36251


namespace cycle_gain_percentage_l362_36213

def cycleA_cp : ℚ := 1000
def cycleB_cp : ℚ := 3000
def cycleC_cp : ℚ := 6000

def cycleB_discount_rate : ℚ := 10 / 100
def cycleC_tax_rate : ℚ := 5 / 100

def cycleA_sp : ℚ := 2000
def cycleB_sp : ℚ := 4500
def cycleC_sp : ℚ := 8000

def cycleA_sales_tax_rate : ℚ := 5 / 100
def cycleB_selling_discount_rate : ℚ := 8 / 100

def total_cp : ℚ := cycleA_cp + cycleB_cp * (1 - cycleB_discount_rate) + cycleC_cp * (1 + cycleC_tax_rate)
def total_sp : ℚ := cycleA_sp * (1 + cycleA_sales_tax_rate) + cycleB_sp * (1 - cycleB_selling_discount_rate) + cycleC_sp

def overall_gain : ℚ := total_sp - total_cp
def gain_percentage : ℚ := (overall_gain / total_cp) * 100

theorem cycle_gain_percentage :
  gain_percentage = 42.4 := by sorry

end cycle_gain_percentage_l362_36213


namespace number_of_teachers_l362_36233

/-- Represents the total number of people (teachers and students) in the school -/
def total : ℕ := 2400

/-- Represents the size of the stratified sample -/
def sample_size : ℕ := 160

/-- Represents the number of students in the sample -/
def students_in_sample : ℕ := 150

/-- Calculates the number of teachers in the school based on the given information -/
def teachers : ℕ := total - (total * students_in_sample / sample_size)

/-- Theorem stating that the number of teachers in the school is 150 -/
theorem number_of_teachers : teachers = 150 := by sorry

end number_of_teachers_l362_36233


namespace product_of_935421_and_625_l362_36277

theorem product_of_935421_and_625 : 935421 * 625 = 584638125 := by
  sorry

end product_of_935421_and_625_l362_36277


namespace cow_ratio_l362_36287

theorem cow_ratio (total : ℕ) (females males : ℕ) : 
  total = 300 →
  females + males = total →
  females = 2 * (females / 2) →
  males = 2 * (males / 2) →
  females / 2 = males / 2 + 50 →
  females = 2 * males :=
by
  sorry

end cow_ratio_l362_36287


namespace right_triangle_sum_squares_l362_36225

theorem right_triangle_sum_squares (A B C : ℝ × ℝ) : 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 →  -- Right triangle condition
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = ((A.1 - C.1)^2 + (A.2 - C.2)^2) →  -- BC is hypotenuse
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4 →  -- BC = 2
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.1 - C.1)^2 + (A.2 - C.2)^2 + (C.1 - B.1)^2 + (C.2 - B.2)^2 = 8 := by
sorry

end right_triangle_sum_squares_l362_36225


namespace coffee_beans_cost_l362_36242

/-- Proves the amount spent on coffee beans given initial amount, cost of tumbler, and remaining amount -/
theorem coffee_beans_cost (initial_amount : ℕ) (tumbler_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 50 →
  tumbler_cost = 30 →
  remaining_amount = 10 →
  initial_amount - tumbler_cost - remaining_amount = 10 := by
  sorry

end coffee_beans_cost_l362_36242


namespace gcd_lcm_sum_product_l362_36245

theorem gcd_lcm_sum_product (a b : ℕ) (ha : a = 8) (hb : b = 12) :
  (Nat.gcd a b + Nat.lcm a b) * Nat.gcd a b = 112 := by
  sorry

end gcd_lcm_sum_product_l362_36245


namespace product_of_exponents_l362_36292

theorem product_of_exponents (p r s : ℕ) : 
  4^p + 4^2 = 280 → 
  3^r + 29 = 56 → 
  7^s + 6^3 + 7^2 = 728 → 
  p * r * s = 27 := by
  sorry

end product_of_exponents_l362_36292


namespace tangent_line_problem_l362_36259

theorem tangent_line_problem (a : ℝ) :
  (∃ l : Set (ℝ × ℝ),
    -- l is a line
    (∃ m k : ℝ, l = {(x, y) | y = m*x + k}) ∧
    -- l passes through (1,0)
    (1, 0) ∈ l ∧
    -- l is tangent to y = x^3
    (∃ x₀ y₀ : ℝ, (x₀, y₀) ∈ l ∧ y₀ = x₀^3 ∧ m = 3*x₀^2) ∧
    -- l is tangent to y = ax^2 + (15/4)x - 9
    (∃ x₁ y₁ : ℝ, (x₁, y₁) ∈ l ∧ y₁ = a*x₁^2 + (15/4)*x₁ - 9 ∧ m = 2*a*x₁ + 15/4)) →
  a = -25/64 ∨ a = -1 :=
by sorry

end tangent_line_problem_l362_36259


namespace last_digit_of_2_to_2024_l362_36264

def last_digit (n : ℕ) : ℕ := n % 10

def power_of_two_last_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 0 => 6
  | _ => 0  -- This case should never occur, but Lean requires it for completeness

theorem last_digit_of_2_to_2024 :
  last_digit (2^2024) = power_of_two_last_digit 2024 :=
by sorry

end last_digit_of_2_to_2024_l362_36264


namespace fifth_score_calculation_l362_36217

theorem fifth_score_calculation (score1 score2 score3 score4 score5 : ℝ) :
  score1 = 85 ∧ score2 = 90 ∧ score3 = 87 ∧ score4 = 92 →
  (score1 + score2 + score3 + score4 + score5) / 5 = 89 →
  score5 = 91 := by
sorry

end fifth_score_calculation_l362_36217


namespace equation_graph_is_axes_l362_36248

/-- The set of points (x, y) satisfying (x-y)^2 = x^2 + y^2 is equivalent to the union of the x-axis and y-axis -/
theorem equation_graph_is_axes (x y : ℝ) : 
  (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := by
  sorry

end equation_graph_is_axes_l362_36248


namespace intersection_M_N_l362_36288

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}
def N : Set ℝ := {y | ∃ x ∈ (Set.Ioo 0 2), y = Real.log (2*x - x^2)}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 1 2 := by sorry

end intersection_M_N_l362_36288


namespace watch_payment_in_dimes_l362_36212

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of cents in a dime -/
def cents_per_dime : ℕ := 10

/-- The cost of the watch in dollars -/
def watch_cost : ℕ := 9

/-- Calculates the number of dimes needed to pay for an item given its cost in dollars -/
def dimes_needed (cost : ℕ) : ℕ :=
  (cost * cents_per_dollar) / cents_per_dime

theorem watch_payment_in_dimes :
  dimes_needed watch_cost = 90 := by
  sorry

end watch_payment_in_dimes_l362_36212


namespace no_zero_root_for_equations_l362_36244

theorem no_zero_root_for_equations :
  (∀ x : ℝ, 3 * x^2 - 5 = 50 → x ≠ 0) ∧
  (∀ x : ℝ, (3*x - 2)^2 = (x - 2)^2 → x ≠ 0) ∧
  (∀ x : ℝ, x^2 - 15 = x + 2 → x ≠ 0) := by
  sorry

end no_zero_root_for_equations_l362_36244


namespace smallest_inexpressible_is_eleven_l362_36205

def expressible (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, n = (2^a - 2^b) / (2^c - 2^d)

theorem smallest_inexpressible_is_eleven :
  (∀ m < 11, expressible m) ∧ ¬expressible 11 :=
sorry

end smallest_inexpressible_is_eleven_l362_36205


namespace periodic_odd_quadratic_function_properties_l362_36296

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_odd_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f (-x) = -f x

def is_quadratic_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ A B C : ℝ, ∀ x, a ≤ x ∧ x ≤ b → f x = A * x^2 + B * x + C

theorem periodic_odd_quadratic_function_properties
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 5)
  (h_odd : is_odd_on_interval f (-1) 1)
  (h_quadratic : is_quadratic_on_interval f 1 4)
  (h_min : f 2 = -5 ∧ ∀ x, f x ≥ -5) :
  (f 1 + f 4 = 0) ∧
  (∀ x, 1 ≤ x ∧ x ≤ 4 → f x = 2 * (x - 2)^2 - 5) :=
by sorry

end periodic_odd_quadratic_function_properties_l362_36296


namespace range_of_a_l362_36263

/-- A function f : ℝ → ℝ is even -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f : ℝ → ℝ is increasing on [0, +∞) -/
def IsIncreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

/-- The condition that f(ax+1) ≤ f(x-2) holds for all x in [1/2, 1] -/
def Condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, 1/2 ≤ x → x ≤ 1 → f (a*x + 1) ≤ f (x - 2)

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
    (h1 : IsEven f)
    (h2 : IsIncreasingOnNonnegative f)
    (h3 : Condition f a) :
  -2 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l362_36263


namespace balloon_permutations_l362_36203

def balloon_arrangements : ℕ := 1260

theorem balloon_permutations :
  let total_letters : ℕ := 7
  let repeated_l : ℕ := 2
  let repeated_o : ℕ := 2
  let unique_letters : ℕ := 3
  (total_letters = repeated_l + repeated_o + unique_letters) →
  (balloon_arrangements = (Nat.factorial total_letters) / ((Nat.factorial repeated_l) * (Nat.factorial repeated_o))) :=
by sorry

end balloon_permutations_l362_36203


namespace travel_ways_count_l362_36211

/-- The number of highways from A to B -/
def num_highways : ℕ := 3

/-- The number of railways from A to B -/
def num_railways : ℕ := 2

/-- The total number of ways to travel from A to B -/
def total_ways : ℕ := num_highways + num_railways

theorem travel_ways_count : total_ways = 5 := by
  sorry

end travel_ways_count_l362_36211


namespace h_value_l362_36200

theorem h_value (h : ℝ) : 
  (∃ x y : ℝ, x^2 - 4*h*x = 5 ∧ y^2 - 4*h*y = 5 ∧ x^2 + y^2 = 34) → 
  |h| = Real.sqrt (3/2) := by
sorry

end h_value_l362_36200


namespace volleyball_prob_l362_36228

-- Define the set of sports
inductive Sport
| Soccer
| Basketball
| Volleyball

-- Define the probability space
def sportProbabilitySpace : Type := Sport

-- Define the probability measure
axiom prob : sportProbabilitySpace → ℝ

-- Axioms for probability measure
axiom prob_nonneg : ∀ s : sportProbabilitySpace, 0 ≤ prob s
axiom prob_sum_one : (prob Sport.Soccer) + (prob Sport.Basketball) + (prob Sport.Volleyball) = 1

-- Axiom for equal probability of each sport
axiom equal_prob : prob Sport.Soccer = prob Sport.Basketball ∧ 
                   prob Sport.Basketball = prob Sport.Volleyball

-- Theorem: The probability of choosing volleyball is 1/3
theorem volleyball_prob : prob Sport.Volleyball = 1/3 := by
  sorry

end volleyball_prob_l362_36228


namespace special_square_area_l362_36282

/-- A square with two vertices on a parabola and one side on a line -/
structure SpecialSquare where
  /-- The parabola on which two vertices of the square lie -/
  parabola : ℝ → ℝ
  /-- The line on which one side of the square lies -/
  line : ℝ → ℝ
  /-- Condition that the parabola is y = x^2 -/
  parabola_eq : parabola = fun x ↦ x^2
  /-- Condition that the line is y = 2x - 17 -/
  line_eq : line = fun x ↦ 2*x - 17

/-- The area of the special square is either 80 or 1280 -/
theorem special_square_area (s : SpecialSquare) :
  ∃ (area : ℝ), (area = 80 ∨ area = 1280) ∧ 
  (∃ (side : ℝ), side^2 = area ∧
   ∃ (x₁ y₁ x₂ y₂ : ℝ),
     y₁ = s.parabola x₁ ∧
     y₂ = s.parabola x₂ ∧
     (∃ (x₃ y₃ : ℝ), y₃ = s.line x₃ ∧
      side = ((x₃ - x₁)^2 + (y₃ - y₁)^2)^(1/2))) :=
by sorry

end special_square_area_l362_36282


namespace factorization_problems_l362_36219

theorem factorization_problems (m x n : ℝ) : 
  (m * x^2 - 2 * m^2 * x + m^3 = m * (x - m)^2) ∧ 
  (8 * m^2 * n + 2 * m * n = 2 * m * n * (4 * m + 1)) := by
  sorry

end factorization_problems_l362_36219


namespace absolute_value_of_squared_negative_l362_36269

theorem absolute_value_of_squared_negative : |(-2)^2| = 2 := by
  sorry

end absolute_value_of_squared_negative_l362_36269


namespace perimeter_of_circular_sector_problem_perimeter_l362_36268

/-- The perimeter of a region formed by two radii and an arc of a circle -/
theorem perimeter_of_circular_sector (r : ℝ) (arc_fraction : ℝ) : 
  r > 0 → 
  0 < arc_fraction → 
  arc_fraction ≤ 1 → 
  2 * r + arc_fraction * (2 * π * r) = 2 * r + 2 * arc_fraction * π * r :=
by sorry

/-- The perimeter of the specific region in the problem -/
theorem problem_perimeter : 
  let r : ℝ := 8
  let arc_fraction : ℝ := 5/6
  2 * r + arc_fraction * (2 * π * r) = 16 + (40/3) * π :=
by sorry

end perimeter_of_circular_sector_problem_perimeter_l362_36268


namespace series_sum_equals_closed_form_l362_36230

/-- The sum of the series Σ(n=1 to ∞) (-1)^(n+1)/(3n-2) -/
noncomputable def seriesSum : ℝ := ∑' n, ((-1 : ℝ)^(n+1)) / (3*n - 2)

/-- The closed form of the series sum -/
noncomputable def closedForm : ℝ := (1/3) * (Real.log 2 + 2 * Real.pi / Real.sqrt 3)

/-- Theorem stating that the series sum equals the closed form -/
theorem series_sum_equals_closed_form : seriesSum = closedForm := by sorry

end series_sum_equals_closed_form_l362_36230


namespace distance_covered_l362_36208

theorem distance_covered (time_minutes : ℝ) (speed_km_per_hour : ℝ) : 
  time_minutes = 42 → speed_km_per_hour = 10 → 
  (time_minutes / 60) * speed_km_per_hour = 7 := by
  sorry

end distance_covered_l362_36208


namespace fraction_denominator_l362_36224

theorem fraction_denominator (y a : ℝ) (h1 : y > 0) (h2 : (2 * y) / a + (3 * y) / a = 0.5 * y) : a = 10 := by
  sorry

end fraction_denominator_l362_36224


namespace square_difference_formula_l362_36283

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8 / 15) (h2 : x - y = 2 / 15) : x^2 - y^2 = 16 / 225 := by
  sorry

end square_difference_formula_l362_36283


namespace tan_symmetry_cos_squared_plus_sin_min_value_l362_36262

-- Define the tangent function
noncomputable def tan (x : ℝ) := Real.tan x

-- Define the cosine function
noncomputable def cos (x : ℝ) := Real.cos x

-- Define the sine function
noncomputable def sin (x : ℝ) := Real.sin x

-- Proposition ①
theorem tan_symmetry (k : ℤ) :
  ∀ x : ℝ, tan (k * π / 2 + x) = -tan (k * π / 2 - x) :=
sorry

-- Proposition ④
theorem cos_squared_plus_sin_min_value :
  ∃ x : ℝ, ∀ y : ℝ, cos y ^ 2 + sin y ≥ cos x ^ 2 + sin x ∧ cos x ^ 2 + sin x = -1 :=
sorry

end tan_symmetry_cos_squared_plus_sin_min_value_l362_36262


namespace specific_pyramid_perimeter_l362_36214

/-- A square pyramid with specific dimensions and properties -/
structure SquarePyramid where
  base_edge : ℝ
  lateral_edge : ℝ
  front_view_isosceles : Prop
  side_view_isosceles : Prop
  views_congruent : Prop

/-- The perimeter of the front view of a square pyramid -/
def front_view_perimeter (p : SquarePyramid) : ℝ := sorry

/-- Theorem stating the perimeter of the front view for a specific square pyramid -/
theorem specific_pyramid_perimeter :
  ∀ (p : SquarePyramid),
    p.base_edge = 2 ∧
    p.lateral_edge = Real.sqrt 3 ∧
    p.front_view_isosceles ∧
    p.side_view_isosceles ∧
    p.views_congruent →
    front_view_perimeter p = 2 + 2 * Real.sqrt 2 := by sorry

end specific_pyramid_perimeter_l362_36214


namespace successive_integers_product_l362_36271

theorem successive_integers_product (n : ℤ) : n * (n + 1) = 9506 → n = 97 := by
  sorry

end successive_integers_product_l362_36271


namespace michaels_art_show_earnings_l362_36226

/-- Calculates Michael's earnings from an art show -/
def michaels_earnings (
  extra_large_price : ℝ)
  (large_price : ℝ)
  (medium_price : ℝ)
  (small_price : ℝ)
  (extra_large_sold : ℕ)
  (large_sold : ℕ)
  (medium_sold : ℕ)
  (small_sold : ℕ)
  (large_discount_rate : ℝ)
  (sales_tax_rate : ℝ)
  (material_cost : ℝ)
  (commission_rate : ℝ) : ℝ :=
  let extra_large_revenue := extra_large_price * extra_large_sold
  let large_revenue := large_price * large_sold * (1 - large_discount_rate)
  let medium_revenue := medium_price * medium_sold
  let small_revenue := small_price * small_sold
  let total_revenue := extra_large_revenue + large_revenue + medium_revenue + small_revenue
  let sales_tax := total_revenue * sales_tax_rate
  let total_collected := total_revenue + sales_tax
  let commission := total_revenue * commission_rate
  let total_deductions := material_cost + commission
  total_collected - total_deductions

/-- Theorem stating Michael's earnings from the art show -/
theorem michaels_art_show_earnings :
  michaels_earnings 150 100 80 60 3 5 8 10 0.1 0.05 300 0.1 = 1733 := by
  sorry

end michaels_art_show_earnings_l362_36226


namespace b_in_terms_of_a_l362_36295

theorem b_in_terms_of_a (k : ℝ) (a b : ℝ) 
  (ha : a = 3 + 3^k) 
  (hb : b = 3 + 3^(-k)) : 
  b = (3*a - 8) / (a - 3) := by
  sorry

end b_in_terms_of_a_l362_36295


namespace unique_positive_solution_l362_36279

theorem unique_positive_solution (n : ℕ) (hn : n > 1) :
  ∀ x : ℝ, x > 0 → (x^n - n*x + n - 1 = 0) → x = 1 := by
  sorry

end unique_positive_solution_l362_36279


namespace path_length_is_twelve_l362_36206

/-- A right triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  side_a : ℝ
  side_b : ℝ
  hypotenuse : ℝ
  is_right : side_a^2 + side_b^2 = hypotenuse^2
  side_values : side_a = 9 ∧ side_b = 12 ∧ hypotenuse = 15

/-- A circle rolling inside the triangle -/
structure RollingCircle where
  radius : ℝ
  radius_value : radius = 2

/-- The path traced by the center of the rolling circle -/
def path_length (t : RightTriangle) (c : RollingCircle) : ℝ := 
  t.side_a + t.side_b + t.hypotenuse - 2 * (t.side_a + t.side_b + t.hypotenuse - 6 * c.radius)

/-- Theorem stating that the path length is 12 -/
theorem path_length_is_twelve (t : RightTriangle) (c : RollingCircle) : 
  path_length t c = 12 := by sorry

end path_length_is_twelve_l362_36206


namespace game_end_not_one_l362_36243

/-- Represents the state of the board with the number of ones and twos -/
structure BoardState where
  ones : Nat
  twos : Nat

/-- Represents a move in the game -/
inductive Move
  | SameDigits : Move
  | DifferentDigits : Move

/-- Applies a move to the board state -/
def applyMove (state : BoardState) (move : Move) : BoardState :=
  match move with
  | Move.SameDigits => 
    if state.ones ≥ 2 then BoardState.mk (state.ones - 2) (state.twos + 1)
    else BoardState.mk state.ones (state.twos - 1)
  | Move.DifferentDigits => 
    if state.ones > 0 && state.twos > 0 
    then BoardState.mk state.ones state.twos
    else state -- This case should not occur in a valid game

/-- The theorem stating that if we start with an even number of ones, 
    the game cannot end with a single one -/
theorem game_end_not_one (initialOnes : Nat) (initialTwos : Nat) :
  initialOnes % 2 = 0 → 
  ∀ (moves : List Move), 
    let finalState := moves.foldl applyMove (BoardState.mk initialOnes initialTwos)
    finalState.ones + finalState.twos = 1 → finalState.ones ≠ 1 :=
by sorry

end game_end_not_one_l362_36243


namespace largest_number_game_l362_36232

theorem largest_number_game (a b c d : ℤ) : 
  (let game := λ (x y z w : ℤ) => (x + y + z) / 3 + w
   ({game a b c d, game a b d c, game a c d b, game b c d a} : Set ℤ) = {17, 21, 23, 29}) →
  (max a (max b (max c d)) = 21) := by
  sorry

end largest_number_game_l362_36232


namespace basketball_shooting_probability_unique_shot_probability_l362_36298

/-- The probability of passing a basketball shooting test -/
def pass_probability : ℝ := 0.784

/-- The probability of making a single shot -/
def shot_probability : ℝ := 0.4

/-- The number of shooting opportunities -/
def max_attempts : ℕ := 3

/-- Theorem stating that the given shot probability results in the specified pass probability -/
theorem basketball_shooting_probability :
  shot_probability + (1 - shot_probability) * shot_probability + 
  (1 - shot_probability)^2 * shot_probability = pass_probability := by
  sorry

/-- Theorem stating that the shot probability is the unique solution -/
theorem unique_shot_probability :
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 →
  (p + (1 - p) * p + (1 - p)^2 * p = pass_probability) →
  p = shot_probability := by
  sorry

end basketball_shooting_probability_unique_shot_probability_l362_36298


namespace min_value_expression_l362_36280

theorem min_value_expression (a : ℚ) : 
  |2*a + 1| + 1 ≥ 1 ∧ ∃ a : ℚ, |2*a + 1| + 1 = 1 := by
  sorry

end min_value_expression_l362_36280


namespace downstream_speed_calculation_l362_36252

/-- The speed downstream of a boat, given its speed in still water and the speed of the current. -/
def speed_downstream (speed_still_water speed_current : ℝ) : ℝ :=
  speed_still_water + speed_current

/-- Theorem stating that the speed downstream is 77 kmph when the boat's speed in still water is 60 kmph and the current speed is 17 kmph. -/
theorem downstream_speed_calculation :
  speed_downstream 60 17 = 77 := by
  sorry

end downstream_speed_calculation_l362_36252


namespace max_b_in_box_l362_36207

theorem max_b_in_box (a b c : ℕ) : 
  a * b * c = 360 →
  1 < c →
  c < b →
  b < a →
  b ≤ 12 :=
by sorry

end max_b_in_box_l362_36207


namespace average_median_relation_l362_36229

theorem average_median_relation (a b c : ℤ) : 
  (a + b + c) / 3 = 4 * b →
  a < b →
  b < c →
  a = 0 →
  c / b = 11 := by
  sorry

end average_median_relation_l362_36229


namespace addie_stamp_ratio_l362_36284

theorem addie_stamp_ratio (parker_initial stamps : ℕ) (parker_final : ℕ) (addie_total : ℕ) : 
  parker_initial = 18 → 
  parker_final = 36 → 
  addie_total = 72 → 
  (parker_final - parker_initial) * 4 = addie_total := by
sorry

end addie_stamp_ratio_l362_36284


namespace cistern_filling_time_l362_36293

-- Define the time to fill 1/11 of the cistern
def time_for_one_eleventh : ℝ := 3

-- Define the function to calculate the time to fill the entire cistern
def time_for_full_cistern : ℝ := time_for_one_eleventh * 11

-- Theorem statement
theorem cistern_filling_time : time_for_full_cistern = 33 := by
  sorry

end cistern_filling_time_l362_36293


namespace quadratic_inequality_l362_36204

theorem quadratic_inequality (y : ℝ) : 
  y^2 + 3*y - 54 > 0 ↔ y < -9 ∨ y > 6 := by sorry

end quadratic_inequality_l362_36204


namespace arctan_equation_solution_l362_36239

theorem arctan_equation_solution (x : ℝ) :
  2 * Real.arctan (1/3) + Real.arctan (1/10) + Real.arctan (1/x) = π/4 → x = 37/3 := by
  sorry

end arctan_equation_solution_l362_36239


namespace quadratic_rational_root_even_coefficients_l362_36215

theorem quadratic_rational_root_even_coefficients 
  (a b c : ℤ) (x : ℚ) : 
  (a * x^2 + b * x + c = 0) → (Even a ∧ Even b ∧ Even c) :=
sorry

end quadratic_rational_root_even_coefficients_l362_36215


namespace smallest_y_in_arithmetic_series_l362_36223

theorem smallest_y_in_arithmetic_series (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →  -- all terms are positive
  ∃ d : ℝ, x = y - d ∧ z = y + d →  -- arithmetic series
  x * y * z = 216 →  -- product is 216
  y ≥ 6 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    (∃ d₀ : ℝ, x₀ = y₀ - d₀ ∧ z₀ = y₀ + d₀) ∧ 
    x₀ * y₀ * z₀ = 216 ∧ y₀ = 6 :=
by sorry

end smallest_y_in_arithmetic_series_l362_36223


namespace total_consumption_theorem_l362_36237

/-- Represents the amount of liquid consumed by each person --/
structure Consumption where
  elijah : Float
  emilio : Float
  isabella : Float
  xavier_soda : Float
  xavier_fruit_punch : Float

/-- Converts pints to cups --/
def pints_to_cups (pints : Float) : Float := pints * 2

/-- Converts liters to cups --/
def liters_to_cups (liters : Float) : Float := liters * 4.22675

/-- Converts gallons to cups --/
def gallons_to_cups (gallons : Float) : Float := gallons * 16

/-- Calculates the total cups consumed based on the given consumption --/
def total_cups (c : Consumption) : Float :=
  c.elijah + c.emilio + c.isabella + c.xavier_soda + c.xavier_fruit_punch

/-- Theorem stating that the total cups consumed is equal to 80.68025 --/
theorem total_consumption_theorem (c : Consumption)
  (h1 : c.elijah = pints_to_cups 8.5)
  (h2 : c.emilio = pints_to_cups 9.5)
  (h3 : c.isabella = liters_to_cups 3)
  (h4 : c.xavier_soda = gallons_to_cups 2 * 0.6)
  (h5 : c.xavier_fruit_punch = gallons_to_cups 2 * 0.4) :
  total_cups c = 80.68025 := by
  sorry


end total_consumption_theorem_l362_36237


namespace logarithm_inequality_l362_36256

theorem logarithm_inequality (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : 0 < c) 
  (h4 : c < 1) : 
  a * Real.log c / Real.log b < b * Real.log c / Real.log a := by
  sorry

end logarithm_inequality_l362_36256


namespace problem_statement_l362_36273

theorem problem_statement (a b c : ℝ) : 
  (¬(∀ x y : ℝ, x > y → x^2 > y^2) ∧ ¬(∀ x y : ℝ, x^2 > y^2 → x > y)) ∧
  ((∀ x y z : ℝ, x*z^2 > y*z^2 → x > y) ∧ ¬(∀ x y z : ℝ, x > y → x*z^2 > y*z^2)) := by
  sorry

end problem_statement_l362_36273


namespace factorization_difference_of_squares_l362_36222

theorem factorization_difference_of_squares (a : ℝ) : 4 * a^2 - 1 = (2*a + 1) * (2*a - 1) := by
  sorry

end factorization_difference_of_squares_l362_36222


namespace min_sum_squares_roots_l362_36221

theorem min_sum_squares_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 4 * m * x₁ + 2 * m^2 + 3 * m - 2 = 0) →
  (2 * x₂^2 - 4 * m * x₂ + 2 * m^2 + 3 * m - 2 = 0) →
  (∀ m' : ℝ, ∃ x₁' x₂' : ℝ, 2 * x₁'^2 - 4 * m' * x₁' + 2 * m'^2 + 3 * m' - 2 = 0 ∧
                             2 * x₂'^2 - 4 * m' * x₂' + 2 * m'^2 + 3 * m' - 2 = 0) →
  x₁^2 + x₂^2 ≥ 8/9 ∧ (x₁^2 + x₂^2 = 8/9 ↔ m = 2/3) := by
sorry

end min_sum_squares_roots_l362_36221


namespace sufficient_not_necessary_l362_36253

theorem sufficient_not_necessary (x a : ℝ) (h : x > 0) :
  (a = 4 → ∀ x > 0, x + a / x ≥ 4) ∧
  ¬(∀ x > 0, x + a / x ≥ 4 → a = 4) :=
by
  sorry

end sufficient_not_necessary_l362_36253


namespace vector_operation_result_l362_36246

theorem vector_operation_result :
  let a : ℝ × ℝ × ℝ := (-3, 5, 2)
  let b : ℝ × ℝ × ℝ := (1, -1, 3)
  let c : ℝ × ℝ × ℝ := (2, 0, -4)
  a - 4 • b + c = (-5, 9, -14) :=
by sorry

end vector_operation_result_l362_36246


namespace bob_gardening_project_cost_l362_36235

/-- Calculates the total cost of a gardening project -/
def gardening_project_cost (num_rose_bushes : ℕ) (cost_per_rose_bush : ℕ) 
  (gardener_hourly_rate : ℕ) (hours_per_day : ℕ) (num_days : ℕ)
  (soil_volume : ℕ) (soil_cost_per_unit : ℕ) : ℕ :=
  num_rose_bushes * cost_per_rose_bush +
  gardener_hourly_rate * hours_per_day * num_days +
  soil_volume * soil_cost_per_unit

/-- The total cost of Bob's gardening project is $4100 -/
theorem bob_gardening_project_cost :
  gardening_project_cost 20 150 30 5 4 100 5 = 4100 := by
  sorry

end bob_gardening_project_cost_l362_36235


namespace g_range_l362_36236

/-- The function f(x) = 2x^2 + 3x - 2 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 2

/-- The function g(x) = f(f(x)) -/
def g (x : ℝ) : ℝ := f (f x)

/-- The domain of g -/
def g_domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

theorem g_range :
  ∀ y ∈ g '' g_domain, -2 ≤ y ∧ y ≤ 424 :=
sorry

end g_range_l362_36236


namespace ratio_problem_l362_36266

theorem ratio_problem (p q r s : ℚ) 
  (h1 : p / q = 8)
  (h2 : r / q = 5)
  (h3 : r / s = 3 / 4) :
  s^2 / p^2 = 25 / 36 := by
sorry

end ratio_problem_l362_36266


namespace area_preserved_l362_36290

-- Define the transformation
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - 1, p.2 + 2)

-- Define a quadrilateral as a set of four points in ℝ²
def Quadrilateral := Fin 4 → ℝ × ℝ

-- Define the area of a quadrilateral
noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Define F and F'
def F : Quadrilateral := sorry
def F' : Quadrilateral := fun i => f (F i)

-- Theorem statement
theorem area_preserved (h : area F = 6) : area F' = area F := by sorry

end area_preserved_l362_36290


namespace prob_at_least_one_is_correct_l362_36291

/-- The probability that person A tells the truth -/
def prob_A : ℝ := 0.8

/-- The probability that person B tells the truth -/
def prob_B : ℝ := 0.6

/-- The probability that person C tells the truth -/
def prob_C : ℝ := 0.75

/-- The probability that at least one person tells the truth -/
def prob_at_least_one : ℝ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

theorem prob_at_least_one_is_correct : prob_at_least_one = 0.98 := by
  sorry

end prob_at_least_one_is_correct_l362_36291


namespace infinitely_many_benelux_couples_l362_36249

/-- Definition of a Benelux couple -/
def is_benelux_couple (m n : ℕ) : Prop :=
  1 < m ∧ m < n ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ m ↔ p ∣ n)) ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ (m + 1) ↔ p ∣ (n + 1)))

/-- Theorem: There are infinitely many Benelux couples -/
theorem infinitely_many_benelux_couples :
  ∀ N : ℕ, ∃ m n : ℕ, N < m ∧ is_benelux_couple m n :=
sorry

end infinitely_many_benelux_couples_l362_36249


namespace sequence_elements_l362_36241

theorem sequence_elements : ∃ (n₁ n₂ : ℕ+), 
  (n₁.val^2 + n₁.val = 12) ∧ 
  (n₂.val^2 + n₂.val = 30) ∧ 
  (∀ n : ℕ+, n.val^2 + n.val ≠ 18) ∧ 
  (∀ n : ℕ+, n.val^2 + n.val ≠ 25) := by
  sorry

end sequence_elements_l362_36241


namespace complex_product_quadrant_l362_36216

theorem complex_product_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0 ∧ z.im > 0) := by sorry

end complex_product_quadrant_l362_36216


namespace logarithm_equation_solution_l362_36231

theorem logarithm_equation_solution (x : ℝ) (h1 : Real.log x + Real.log (x - 3) = 1) (h2 : x > 0) : x = 5 := by
  sorry

end logarithm_equation_solution_l362_36231


namespace number_calculation_l362_36278

theorem number_calculation (x : ℝ) : (0.1 * 0.3 * 0.5 * x = 90) → x = 6000 := by
  sorry

end number_calculation_l362_36278


namespace floor_area_approx_l362_36240

/-- The length of the floor in feet -/
def floor_length_ft : ℝ := 15

/-- The width of the floor in feet -/
def floor_width_ft : ℝ := 10

/-- The conversion factor from feet to meters -/
def ft_to_m : ℝ := 0.3048

/-- The area of the floor in square meters -/
def floor_area_m2 : ℝ := floor_length_ft * ft_to_m * floor_width_ft * ft_to_m

theorem floor_area_approx :
  ∃ ε > 0, abs (floor_area_m2 - 13.93) < ε :=
sorry

end floor_area_approx_l362_36240


namespace third_batch_size_l362_36289

/-- Proves that given the conditions of the problem, the number of students in the third batch is 60 -/
theorem third_batch_size :
  let batch1_size : ℕ := 40
  let batch2_size : ℕ := 50
  let batch1_avg : ℚ := 45
  let batch2_avg : ℚ := 55
  let batch3_avg : ℚ := 65
  let total_avg : ℚ := 56333333333333336 / 1000000000000000
  let batch3_size : ℕ := 60
  (batch1_size * batch1_avg + batch2_size * batch2_avg + batch3_size * batch3_avg) / 
    (batch1_size + batch2_size + batch3_size) = total_avg :=
by sorry


end third_batch_size_l362_36289


namespace bicycle_weight_is_12_l362_36247

-- Define the weight of a bicycle and a car
def bicycle_weight : ℝ := sorry
def car_weight : ℝ := sorry

-- State the theorem
theorem bicycle_weight_is_12 :
  (10 * bicycle_weight = 4 * car_weight) →
  (3 * car_weight = 90) →
  bicycle_weight = 12 :=
by sorry

end bicycle_weight_is_12_l362_36247


namespace books_in_box_l362_36299

/-- The number of books in a box given the total weight and weight per book -/
def number_of_books (total_weight weight_per_book : ℚ) : ℚ :=
  total_weight / weight_per_book

/-- Theorem stating that a box weighing 42 pounds with books weighing 3 pounds each contains 14 books -/
theorem books_in_box : number_of_books 42 3 = 14 := by
  sorry

end books_in_box_l362_36299


namespace max_value_fraction_l362_36202

theorem max_value_fraction (a b : ℝ) (h : a^2 + 2*b^2 = 6) :
  ∃ (M : ℝ), M = 1 ∧ ∀ x, x = b/(a-3) → x ≤ M :=
sorry

end max_value_fraction_l362_36202


namespace shirt_cost_l362_36270

theorem shirt_cost (initial_amount : ℕ) (num_shirts : ℕ) (pants_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 109 →
  num_shirts = 2 →
  pants_cost = 13 →
  remaining_amount = 74 →
  (initial_amount - remaining_amount - pants_cost) / num_shirts = 11 := by
  sorry

end shirt_cost_l362_36270


namespace divisors_product_18_l362_36201

def divisors_product (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem divisors_product_18 : divisors_product 18 = 5832 := by
  sorry

end divisors_product_18_l362_36201


namespace complex_fraction_equality_l362_36234

theorem complex_fraction_equality : Complex.I / (1 + Complex.I) = (1 / 2 : ℂ) + (1 / 2 : ℂ) * Complex.I := by
  sorry

end complex_fraction_equality_l362_36234


namespace simplify_complex_root_expression_l362_36210

theorem simplify_complex_root_expression (x : ℝ) (h : x ≥ 0) :
  (6 * x * (5 + 2 * Real.sqrt 6)) ^ (1/4) * Real.sqrt (3 * Real.sqrt (2 * x) - 2 * Real.sqrt (3 * x)) = Real.sqrt (6 * x) := by
  sorry

end simplify_complex_root_expression_l362_36210


namespace cyclist_problem_solution_l362_36257

/-- Represents the cyclist's journey to the bus stop -/
structure CyclistJourney where
  usual_speed : ℝ
  usual_time : ℝ
  reduced_speed_ratio : ℝ
  miss_time : ℝ
  bus_cover_ratio : ℝ

/-- Theorem stating the solution to the cyclist problem -/
theorem cyclist_problem_solution (journey : CyclistJourney) 
  (h1 : journey.reduced_speed_ratio = 4/5)
  (h2 : journey.miss_time = 5)
  (h3 : journey.bus_cover_ratio = 1/3)
  (h4 : journey.usual_time * journey.reduced_speed_ratio = journey.usual_time + journey.miss_time)
  (h5 : journey.usual_time > 0) :
  journey.usual_time = 20 ∧ 
  (journey.usual_time * journey.bus_cover_ratio = journey.usual_time * (1 - journey.bus_cover_ratio)) := by
  sorry

#check cyclist_problem_solution

end cyclist_problem_solution_l362_36257


namespace quadratic_roots_relation_l362_36218

theorem quadratic_roots_relation (m n p : ℝ) : 
  (∀ r : ℝ, (r^2 + p*r + m = 0) → ((3*r)^2 + m*(3*r) + n = 0)) →
  n / p = 27 := by
  sorry

end quadratic_roots_relation_l362_36218


namespace emily_beads_count_l362_36265

/-- The number of necklaces Emily made -/
def necklaces : ℕ := 26

/-- The number of beads required for each necklace -/
def beads_per_necklace : ℕ := 2

/-- The total number of beads Emily had -/
def total_beads : ℕ := necklaces * beads_per_necklace

theorem emily_beads_count : total_beads = 52 := by
  sorry

end emily_beads_count_l362_36265


namespace empire_state_building_height_l362_36267

/-- The height of the Empire State Building to the top floor, in feet -/
def height_to_top_floor : ℕ := 1250

/-- The height of the antenna spire, in feet -/
def antenna_height : ℕ := 204

/-- The total height of the Empire State Building, in feet -/
def total_height : ℕ := height_to_top_floor + antenna_height

/-- Theorem stating that the total height of the Empire State Building is 1454 feet -/
theorem empire_state_building_height : total_height = 1454 := by
  sorry

end empire_state_building_height_l362_36267
