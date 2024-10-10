import Mathlib

namespace wire_length_ratio_l3952_395234

-- Define the given conditions
def bonnie_wire_length : ℝ := 12 * 8
def roark_cube_volume : ℝ := 2
def roark_edge_length : ℝ := 1.5
def bonnie_cube_volume : ℝ := 8^3

-- Define the theorem
theorem wire_length_ratio :
  let roark_cubes_count : ℝ := bonnie_cube_volume / roark_cube_volume
  let roark_total_wire_length : ℝ := roark_cubes_count * (12 * roark_edge_length)
  bonnie_wire_length / roark_total_wire_length = 1 / 48 := by
sorry

end wire_length_ratio_l3952_395234


namespace gcd_lcm_product_l3952_395259

theorem gcd_lcm_product (a b : ℕ) (ha : a = 100) (hb : b = 120) :
  (Nat.gcd a b) * (Nat.lcm a b) = 12000 := by
  sorry

end gcd_lcm_product_l3952_395259


namespace sum_of_possible_k_is_95_l3952_395271

/-- Given a quadratic equation x^2 + 10x + k = 0 with two distinct negative integer solutions,
    this function returns the sum of all possible values of k. -/
def sumOfPossibleK : ℤ := by
  sorry

/-- The theorem states that the sum of all possible values of k is 95. -/
theorem sum_of_possible_k_is_95 : sumOfPossibleK = 95 := by
  sorry

end sum_of_possible_k_is_95_l3952_395271


namespace largest_n_divisibility_l3952_395240

theorem largest_n_divisibility : ∃ (n : ℕ), n = 890 ∧ 
  (∀ m : ℕ, m > n → ¬(m + 10 ∣ m^3 + 100)) ∧ 
  (n + 10 ∣ n^3 + 100) := by
  sorry

end largest_n_divisibility_l3952_395240


namespace grady_blue_cubes_l3952_395208

theorem grady_blue_cubes (grady_red : ℕ) (gage_initial_red gage_initial_blue : ℕ) (gage_total : ℕ) :
  grady_red = 20 →
  gage_initial_red = 10 →
  gage_initial_blue = 12 →
  gage_total = 35 →
  ∃ (grady_blue : ℕ),
    (2 * grady_red / 5 + grady_blue / 3 + gage_initial_red + gage_initial_blue = gage_total) ∧
    grady_blue = 15 :=
by sorry

end grady_blue_cubes_l3952_395208


namespace lucas_50th_mod5_lucas_50th_remainder_l3952_395290

/-- Lucas sequence -/
def lucas : ℕ → ℤ
  | 0 => 2
  | 1 => 1
  | n + 2 => lucas (n + 1) + lucas n

/-- Lucas sequence modulo 5 -/
def lucas_mod5 (n : ℕ) : ℤ := lucas n % 5

/-- The Lucas sequence modulo 5 has a period of 4 -/
axiom lucas_mod5_period : ∀ n, lucas_mod5 (n + 4) = lucas_mod5 n

/-- The 50th term of the Lucas sequence modulo 5 equals the 2nd term modulo 5 -/
theorem lucas_50th_mod5 : lucas_mod5 50 = lucas_mod5 2 := by sorry

/-- The remainder when the 50th term of the Lucas sequence is divided by 5 is 1 -/
theorem lucas_50th_remainder : lucas 50 % 5 = 1 := by sorry

end lucas_50th_mod5_lucas_50th_remainder_l3952_395290


namespace five_eighths_decimal_l3952_395246

theorem five_eighths_decimal : (5 : ℚ) / 8 = 0.625 := by sorry

end five_eighths_decimal_l3952_395246


namespace ab_dot_bc_equals_two_l3952_395233

/-- Given two vectors AB and AC in R², and the magnitude of BC is 1, 
    prove that the dot product of AB and BC is 2. -/
theorem ab_dot_bc_equals_two 
  (AB : ℝ × ℝ) 
  (AC : ℝ × ℝ) 
  (h1 : AB = (2, 3)) 
  (h2 : ∃ t, AC = (3, t)) 
  (h3 : ‖AC - AB‖ = 1) : 
  AB • (AC - AB) = 2 := by
  sorry

end ab_dot_bc_equals_two_l3952_395233


namespace book_original_price_l3952_395253

/-- Proves that given a book sold for Rs 60 with a 20% profit rate, the original price of the book was Rs 50. -/
theorem book_original_price (selling_price : ℝ) (profit_rate : ℝ) : 
  selling_price = 60 → profit_rate = 0.20 → 
  ∃ (original_price : ℝ), original_price = 50 ∧ selling_price = original_price * (1 + profit_rate) :=
by
  sorry

end book_original_price_l3952_395253


namespace one_true_statement_l3952_395287

theorem one_true_statement (a b c : ℝ) : 
  (∃! n : Nat, n = 1 ∧ 
    (((a ≤ b → a * c^2 ≤ b * c^2) ∨ 
      (a > b → a * c^2 > b * c^2) ∨ 
      (a * c^2 ≤ b * c^2 → a ≤ b)))) := by sorry

end one_true_statement_l3952_395287


namespace woojin_harvest_weight_l3952_395225

-- Define the weights harvested by each family member
def younger_brother_weight : Real := 3.8
def older_sister_extra : Real := 8.4
def woojin_extra_grams : Real := 3720

-- Define the conversion factor from kg to g
def kg_to_g : Real := 1000

-- Theorem statement
theorem woojin_harvest_weight :
  let older_sister_weight := younger_brother_weight + older_sister_extra
  let woojin_weight_g := (older_sister_weight / 10) * kg_to_g + woojin_extra_grams
  woojin_weight_g / kg_to_g = 4.94 := by
sorry


end woojin_harvest_weight_l3952_395225


namespace max_pairs_with_distinct_sums_l3952_395242

theorem max_pairs_with_distinct_sums (n : ℕ) (hn : n = 2009) :
  let S := Finset.range n
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 803 ∧
    pairs.card = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p1 p2 : ℕ × ℕ), p1 ∈ pairs → p2 ∈ pairs → p1 ≠ p2 →
      p1.1 ≠ p2.1 ∧ p1.1 ≠ p2.2 ∧ p1.2 ≠ p2.1 ∧ p1.2 ≠ p2.2) ∧
    (∀ (p1 p2 : ℕ × ℕ), p1 ∈ pairs → p2 ∈ pairs → p1 ≠ p2 →
      p1.1 + p1.2 ≠ p2.1 + p2.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) ∧
    (∀ (pairs' : Finset (ℕ × ℕ)),
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) →
      (∀ (p1 p2 : ℕ × ℕ), p1 ∈ pairs' → p2 ∈ pairs' → p1 ≠ p2 →
        p1.1 ≠ p2.1 ∧ p1.1 ≠ p2.2 ∧ p1.2 ≠ p2.1 ∧ p1.2 ≠ p2.2) →
      (∀ (p1 p2 : ℕ × ℕ), p1 ∈ pairs' → p2 ∈ pairs' → p1 ≠ p2 →
        p1.1 + p1.2 ≠ p2.1 + p2.2) →
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 + p.2 ≤ n) →
      pairs'.card ≤ k) :=
by sorry

end max_pairs_with_distinct_sums_l3952_395242


namespace min_value_of_2x_plus_y_min_value_is_one_min_value_exists_l3952_395214

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x + y + 2 * x * y = 5/4) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 2 * a + b + 2 * a * b = 5/4 → 2 * x + y ≤ 2 * a + b :=
by sorry

theorem min_value_is_one (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x + y + 2 * x * y = 5/4) : 
  2 * x + y ≥ 1 :=
by sorry

theorem min_value_exists (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x + y + 2 * x * y = 5/4) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a + b + 2 * a * b = 5/4 ∧ 2 * a + b = 1 :=
by sorry

end min_value_of_2x_plus_y_min_value_is_one_min_value_exists_l3952_395214


namespace waiter_theorem_l3952_395243

def waiter_problem (total_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) : Prop :=
  let remaining_customers := total_customers - left_customers
  remaining_customers / people_per_table = 3

theorem waiter_theorem : waiter_problem 21 12 3 := by
  sorry

end waiter_theorem_l3952_395243


namespace shyam_weight_increase_l3952_395237

-- Define the ratio of Ram's weight to Shyam's weight
def weight_ratio : ℚ := 2 / 5

-- Define Ram's weight increase percentage
def ram_increase : ℚ := 10 / 100

-- Define the total new weight
def total_new_weight : ℚ := 828 / 10

-- Define the total weight increase percentage
def total_increase : ℚ := 15 / 100

-- Function to calculate Shyam's weight increase percentage
def shyam_increase_percentage : ℚ := sorry

-- Theorem statement
theorem shyam_weight_increase :
  abs (shyam_increase_percentage - 1709 / 10000) < 1 / 1000 := by sorry

end shyam_weight_increase_l3952_395237


namespace number_division_problem_l3952_395275

theorem number_division_problem (x : ℝ) : x / 3 = 50 + x / 4 ↔ x = 600 := by
  sorry

end number_division_problem_l3952_395275


namespace continued_fraction_sqrt_15_l3952_395248

theorem continued_fraction_sqrt_15 (y : ℝ) : y = 3 + 5 / (2 + 5 / y) → y = Real.sqrt 15 := by
  sorry

end continued_fraction_sqrt_15_l3952_395248


namespace smallest_n_for_terminating_decimal_l3952_395206

theorem smallest_n_for_terminating_decimal : 
  ∃ (n : ℕ+), n = 24 ∧ 
  (∀ (m : ℕ+), m < n → ¬(∃ (a b : ℕ), (m : ℚ) / (m + 101 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0 ∧ 
  (∀ (p : ℕ), Prime p → p ∣ b → p = 2 ∨ p = 5))) ∧
  (∃ (a b : ℕ), (n : ℚ) / (n + 101 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0 ∧ 
  (∀ (p : ℕ), Prime p → p ∣ b → p = 2 ∨ p = 5)) :=
by sorry


end smallest_n_for_terminating_decimal_l3952_395206


namespace negation_exists_product_zero_l3952_395250

open Real

theorem negation_exists_product_zero (f g : ℝ → ℝ) :
  (¬ ∃ x₀ : ℝ, f x₀ * g x₀ = 0) ↔ (∀ x : ℝ, f x ≠ 0 ∧ g x ≠ 0) :=
by sorry

end negation_exists_product_zero_l3952_395250


namespace largest_constant_inequality_l3952_395299

theorem largest_constant_inequality :
  ∃ (C : ℝ), (C = 2 / Real.sqrt 3) ∧
  (∀ (x y z : ℝ), x^2 + y^2 + 2*z^2 + 1 ≥ C*(x + y + z)) ∧
  (∀ (C' : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + 2*z^2 + 1 ≥ C'*(x + y + z)) → C' ≤ C) :=
by sorry

end largest_constant_inequality_l3952_395299


namespace linear_function_quadrants_l3952_395283

/-- A linear function y = mx - 1 passing through the second, third, and fourth quadrants implies m < 0 -/
theorem linear_function_quadrants (m : ℝ) : 
  (∀ x y : ℝ, y = m * x - 1 →
    ((x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0))) →
  m < 0 :=
by sorry

end linear_function_quadrants_l3952_395283


namespace square_not_always_positive_l3952_395230

theorem square_not_always_positive : ¬ ∀ a : ℝ, a^2 > 0 := by sorry

end square_not_always_positive_l3952_395230


namespace magic_square_sum_l3952_395207

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e f g h i : ℝ)
  (row_sum : a + b + c = d + e + f ∧ d + e + f = g + h + i)
  (col_sum : a + d + g = b + e + h ∧ b + e + h = c + f + i)
  (diag_sum : a + e + i = c + e + g)

/-- The sum of all elements in a magic square -/
def total_sum (m : MagicSquare) : ℝ := m.a + m.b + m.c + m.d + m.e + m.f + m.g + m.h + m.i

/-- Theorem: Sum of remaining squares in a specific magic square -/
theorem magic_square_sum :
  ∀ (m : MagicSquare),
    m.b = 7 ∧ m.c = 2018 ∧ m.g = 4 →
    (total_sum m) - (m.b + m.c + m.g) = -11042.5 := by
  sorry


end magic_square_sum_l3952_395207


namespace apple_juice_fraction_l3952_395229

theorem apple_juice_fraction (pitcher1_capacity pitcher2_capacity : ℚ)
  (pitcher1_fullness pitcher2_fullness : ℚ) :
  pitcher1_capacity = 800 →
  pitcher2_capacity = 500 →
  pitcher1_fullness = 1/4 →
  pitcher2_fullness = 3/8 →
  (pitcher1_capacity * pitcher1_fullness + pitcher2_capacity * pitcher2_fullness) /
  (pitcher1_capacity + pitcher2_capacity) = 31/104 := by
  sorry

end apple_juice_fraction_l3952_395229


namespace fifty_three_is_perfect_sum_x_y_is_one_k_equals_36_max_x_minus_2y_is_two_l3952_395265

-- Definition of a perfect number
def isPerfectNumber (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

-- Statement 1
theorem fifty_three_is_perfect : isPerfectNumber 53 := by sorry

-- Statement 2
theorem sum_x_y_is_one (x y : ℝ) (h : x^2 + y^2 - 4*x + 2*y + 5 = 0) : 
  x + y = 1 := by sorry

-- Statement 3
theorem k_equals_36 (k : ℤ) : 
  (∀ x y : ℤ, isPerfectNumber (2*x^2 + y^2 + 2*x*y + 12*x + k)) → k = 36 := by sorry

-- Statement 4
theorem max_x_minus_2y_is_two (x y : ℝ) (h : -x^2 + (7/2)*x + y - 3 = 0) :
  x - 2*y ≤ 2 := by sorry

end fifty_three_is_perfect_sum_x_y_is_one_k_equals_36_max_x_minus_2y_is_two_l3952_395265


namespace projection_of_a_onto_b_l3952_395291

noncomputable section

/-- Given two non-zero vectors a and b in ℝ², prove that under certain conditions,
    the projection of a onto b is (1/4) * b. -/
theorem projection_of_a_onto_b (a b : ℝ × ℝ) : 
  a ≠ (0, 0) → 
  b = (Real.sqrt 3, 1) → 
  a.1 * b.1 + a.2 * b.2 = π / 3 → 
  (a.1 - b.1) * a.1 + (a.2 - b.2) * a.2 = 0 → 
  ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b = (1/4) • b := by
  sorry

end

end projection_of_a_onto_b_l3952_395291


namespace hyperbola_foci_distance_l3952_395215

/-- Given a hyperbola with equation x²/32 - y²/4 = 1, the distance between its foci is 12 -/
theorem hyperbola_foci_distance (x y : ℝ) :
  x^2 / 32 - y^2 / 4 = 1 → ∃ c : ℝ, c = 6 ∧ 2 * c = 12 :=
by sorry

end hyperbola_foci_distance_l3952_395215


namespace cubic_equation_solution_l3952_395252

theorem cubic_equation_solution (b : ℝ) : 
  let x := b
  let c := 0
  x^3 + c^2 = (b - x)^2 := by sorry

end cubic_equation_solution_l3952_395252


namespace min_value_problem_l3952_395224

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = x * y) :
  3 * x + 4 * y ≥ 28 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 28 :=
sorry

end min_value_problem_l3952_395224


namespace one_true_proposition_l3952_395256

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x > 0 → x^2 > 0

-- Define the converse
def converse (x : ℝ) : Prop := x^2 > 0 → x > 0

-- Define the negation
def negation (x : ℝ) : Prop := x > 0 → x^2 ≤ 0

-- Define the inverse negation
def inverse_negation (x : ℝ) : Prop := x^2 ≤ 0 → x ≤ 0

-- Theorem stating that exactly one of these is true
theorem one_true_proposition :
  ∃! p : (ℝ → Prop), p = converse ∨ p = negation ∨ p = inverse_negation ∧ ∀ x, p x :=
sorry

end one_true_proposition_l3952_395256


namespace inequality_proof_l3952_395284

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end inequality_proof_l3952_395284


namespace student_616_selected_l3952_395218

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  population : ℕ
  sample_size : ℕ
  first_selected : ℕ

/-- Checks if a student number is selected in the systematic sampling -/
def is_selected (s : SystematicSampling) (student : ℕ) : Prop :=
  ∃ k : ℕ, student = s.first_selected + k * (s.population / s.sample_size)

theorem student_616_selected (s : SystematicSampling)
  (h_pop : s.population = 1000)
  (h_sample : s.sample_size = 100)
  (h_46_selected : is_selected s 46) :
  is_selected s 616 := by
sorry

end student_616_selected_l3952_395218


namespace max_profit_week_is_5_l3952_395262

/-- Price function based on week number -/
def price (x : ℕ) : ℚ :=
  if x ≤ 4 then 10 + 2 * x
  else if x ≤ 10 then 20
  else 20 - 2 * (x - 10)

/-- Cost function based on week number -/
def cost (x : ℕ) : ℚ :=
  -0.125 * (x - 8)^2 + 12

/-- Profit function based on week number -/
def profit (x : ℕ) : ℚ :=
  price x - cost x

/-- The week with maximum profit is the 5th week -/
theorem max_profit_week_is_5 :
  ∀ x : ℕ, x ≤ 16 → profit 5 ≥ profit x :=
sorry

end max_profit_week_is_5_l3952_395262


namespace arithmetic_expressions_evaluation_l3952_395217

theorem arithmetic_expressions_evaluation :
  ((-12) - 5 + (-14) - (-39) = 8) ∧
  (-2^2 * 5 - (-12) / 4 - 4 = -21) :=
by sorry

end arithmetic_expressions_evaluation_l3952_395217


namespace jayas_rank_from_bottom_l3952_395264

/-- Given a class of students, calculate the rank from the bottom based on the rank from the top. -/
def rankFromBottom (totalStudents : ℕ) (rankFromTop : ℕ) : ℕ :=
  totalStudents - rankFromTop + 1

/-- Jaya's rank from the bottom in a class of 53 students where she ranks 5th from the top is 50th. -/
theorem jayas_rank_from_bottom :
  rankFromBottom 53 5 = 50 := by
  sorry

end jayas_rank_from_bottom_l3952_395264


namespace inequality_system_solution_l3952_395270

theorem inequality_system_solution (x : ℝ) :
  (1 - x > 3) ∧ (2 * x + 5 ≥ 0) → -2.5 ≤ x ∧ x < -2 := by
  sorry

end inequality_system_solution_l3952_395270


namespace stock_exchange_problem_l3952_395200

theorem stock_exchange_problem (total_stocks : ℕ) 
  (h_total : total_stocks = 1980) 
  (H L : ℕ) 
  (h_relation : H = L + L / 5) 
  (h_sum : H + L = total_stocks) : 
  H = 1080 := by
sorry

end stock_exchange_problem_l3952_395200


namespace trig_identity_proof_l3952_395292

theorem trig_identity_proof : 
  Real.cos (70 * π / 180) * Real.sin (80 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end trig_identity_proof_l3952_395292


namespace cubic_fraction_equals_four_l3952_395276

theorem cubic_fraction_equals_four (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : a + b + 2*c = 0) : (a^3 + b^3 - c^3) / (a*b*c) = 4 := by
  sorry

end cubic_fraction_equals_four_l3952_395276


namespace function_and_tangent_line_l3952_395278

/-- Given a function f(x) = (ax-6) / (x^2 + b) and its tangent line at (-1, f(-1)) 
    with equation x + 2y + 5 = 0, prove that f(x) = (2x-6) / (x^2 + 3) -/
theorem function_and_tangent_line (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => (a * x - 6) / (x^2 + b)
  let tangent_line : ℝ → ℝ := λ x => -(1/2) * x - 5/2
  (f (-1) = tangent_line (-1)) ∧ 
  (deriv f (-1) = deriv tangent_line (-1)) →
  f = λ x => (2 * x - 6) / (x^2 + 3) := by
sorry

end function_and_tangent_line_l3952_395278


namespace visible_yellow_bus_length_l3952_395220

/-- Proves that the visible length of the yellow bus is 18 feet --/
theorem visible_yellow_bus_length (red_bus_length green_truck_length yellow_bus_length orange_car_length : ℝ) :
  red_bus_length = 48 →
  red_bus_length = 4 * orange_car_length →
  yellow_bus_length = 3.5 * orange_car_length →
  green_truck_length = 2 * orange_car_length →
  yellow_bus_length - green_truck_length = 18 := by
  sorry

end visible_yellow_bus_length_l3952_395220


namespace factor_polynomial_l3952_395251

theorem factor_polynomial (x : ℝ) : 45 * x^3 + 135 * x^2 = 45 * x^2 * (x + 3) := by
  sorry

end factor_polynomial_l3952_395251


namespace arithmetic_mean_of_specific_numbers_l3952_395231

theorem arithmetic_mean_of_specific_numbers :
  let numbers := [17, 29, 45, 64]
  (numbers.sum / numbers.length : ℚ) = 38.75 := by
sorry

end arithmetic_mean_of_specific_numbers_l3952_395231


namespace palindrome_count_is_60_l3952_395288

/-- Represents a time on a 24-hour digital clock --/
structure DigitalTime where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Checks if a given DigitalTime is a palindrome --/
def is_palindrome (t : DigitalTime) : Bool :=
  sorry

/-- Counts the number of palindromes on a 24-hour digital clock --/
def count_palindromes : Nat :=
  sorry

/-- Theorem stating that the number of palindromes on a 24-hour digital clock is 60 --/
theorem palindrome_count_is_60 : count_palindromes = 60 := by
  sorry

end palindrome_count_is_60_l3952_395288


namespace quadratic_root_expression_l3952_395261

theorem quadratic_root_expression (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ = 0 → 
  x₂^2 - 2*x₂ = 0 → 
  (x₁ * x₂) / (x₁^2 + x₂^2) = 0 := by
  sorry

end quadratic_root_expression_l3952_395261


namespace complex_square_i_positive_l3952_395293

theorem complex_square_i_positive (a : ℝ) :
  (((a : ℂ) + Complex.I)^2 * Complex.I).re > 0 → a = -1 := by sorry

end complex_square_i_positive_l3952_395293


namespace class_ratios_l3952_395298

theorem class_ratios (male_students female_students : ℕ) 
  (h1 : male_students = 30) 
  (h2 : female_students = 24) : 
  (female_students : ℚ) / male_students = 4/5 ∧ 
  (male_students : ℚ) / (male_students + female_students) = 5/9 := by
  sorry

end class_ratios_l3952_395298


namespace stone_piles_problem_l3952_395213

theorem stone_piles_problem (x y : ℕ) : 
  (y + 100 = 2 * (x - 100)) → 
  (∃ z : ℕ, x + z = 5 * (y - z)) → 
  x ≥ 170 → 
  (x = 170 ∧ y = 40) ∨ x > 170 :=
sorry

end stone_piles_problem_l3952_395213


namespace min_value_fraction_min_value_achievable_l3952_395282

theorem min_value_fraction (x : ℝ) (h : x > 0) :
  (x^2 + x + 3) / (x + 1) ≥ 2 * Real.sqrt 3 - 1 :=
by sorry

theorem min_value_achievable :
  ∃ x > 0, (x^2 + x + 3) / (x + 1) = 2 * Real.sqrt 3 - 1 :=
by sorry

end min_value_fraction_min_value_achievable_l3952_395282


namespace fraction_of_fifteen_l3952_395277

theorem fraction_of_fifteen (x : ℚ) : 
  (x * 15 = 0.8 * 40 - 20) → x = 4/5 := by
sorry

end fraction_of_fifteen_l3952_395277


namespace solve_exponential_equation_l3952_395223

theorem solve_exponential_equation :
  ∃ y : ℝ, (9 : ℝ) ^ y = (3 : ℝ) ^ 12 ∧ y = 6 := by
  sorry

end solve_exponential_equation_l3952_395223


namespace polynomial_factorization_l3952_395260

theorem polynomial_factorization (x y : ℝ) : x * y^2 - 36 * x = x * (y + 6) * (y - 6) := by
  sorry

end polynomial_factorization_l3952_395260


namespace regular_polygon_interior_angle_l3952_395221

theorem regular_polygon_interior_angle (n : ℕ) (n_ge_3 : n ≥ 3) :
  (180 * (n - 2) : ℚ) / n = 150 → n = 12 := by
  sorry

end regular_polygon_interior_angle_l3952_395221


namespace sin_240_degrees_l3952_395266

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l3952_395266


namespace fraction_problem_l3952_395201

theorem fraction_problem (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 4 / 15 := by
sorry

end fraction_problem_l3952_395201


namespace sequence_c_increasing_l3952_395203

theorem sequence_c_increasing (n : ℕ) : 
  let a : ℕ → ℤ := λ n => 2 * n^2 - 5 * n + 1
  a (n + 1) > a n :=
by
  sorry

end sequence_c_increasing_l3952_395203


namespace problem_solution_l3952_395273

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := by
  sorry

end problem_solution_l3952_395273


namespace inequality_proof_l3952_395255

theorem inequality_proof (x : ℝ) :
  x > 0 →
  (x * Real.sqrt (12 - x) + Real.sqrt (12 * x - x^3) ≥ 12) ↔ x = 3 := by
  sorry

end inequality_proof_l3952_395255


namespace inequality_for_all_reals_l3952_395279

theorem inequality_for_all_reals (a : ℝ) : a + a^3 - a^4 - a^6 < 1 := by
  sorry

end inequality_for_all_reals_l3952_395279


namespace factorization_x_squared_minus_4x_l3952_395239

theorem factorization_x_squared_minus_4x (x : ℝ) : x^2 - 4*x = x*(x - 4) := by
  sorry

end factorization_x_squared_minus_4x_l3952_395239


namespace isosceles_if_neg_one_root_side_c_value_l3952_395257

/-- Triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation b(x²-1) + 2ax + c(x²+1) = 0 -/
def equation (t : Triangle) (x : ℝ) : Prop :=
  t.b * (x^2 - 1) + 2 * t.a * x + t.c * (x^2 + 1) = 0

theorem isosceles_if_neg_one_root (t : Triangle) :
  equation t (-1) → t.a = t.c :=
sorry

theorem side_c_value (t : Triangle) :
  (∃ x : ℝ, ∀ y : ℝ, equation t x ↔ y = x) →
  t.a = 5 →
  t.b = 12 →
  t.c = 13 :=
sorry

end isosceles_if_neg_one_root_side_c_value_l3952_395257


namespace john_game_period_duration_l3952_395216

/-- Calculates the duration of each period in John's game --/
def period_duration (points_per_interval : ℕ) (total_points : ℕ) (num_periods : ℕ) : ℕ :=
  (total_points / points_per_interval * 4) / num_periods

/-- Proves that each period lasts 12 minutes given the game conditions --/
theorem john_game_period_duration :
  period_duration 7 42 2 = 12 := by
  sorry

end john_game_period_duration_l3952_395216


namespace Q_equals_sum_l3952_395280

/-- Binomial coefficient -/
def binomial (a b : ℕ) : ℕ :=
  if a ≥ b then
    Nat.factorial a / (Nat.factorial b * Nat.factorial (a - b))
  else
    0

/-- Coefficient of x^k in (1+x+x^2+x^3)^n -/
def Q (n k : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (fun j => binomial n j * binomial n (k - 2 * j))

/-- The main theorem -/
theorem Q_equals_sum (n k : ℕ) :
    Q n k = (Finset.range (n + 1)).sum (fun j => binomial n j * binomial n (k - 2 * j)) := by
  sorry

end Q_equals_sum_l3952_395280


namespace triangle_inequality_and_equality_condition_l3952_395244

theorem triangle_inequality_and_equality_condition (a b c : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end triangle_inequality_and_equality_condition_l3952_395244


namespace intersection_equals_positive_l3952_395202

-- Define sets A and B
def A : Set ℝ := {x | 2 * x^2 + x > 0}
def B : Set ℝ := {x | 2 * x + 1 > 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_positive : A_intersect_B = {x : ℝ | x > 0} := by
  sorry

end intersection_equals_positive_l3952_395202


namespace square_root_divided_by_19_equals_4_l3952_395289

theorem square_root_divided_by_19_equals_4 : 
  ∃ (x : ℝ), x > 0 ∧ (Real.sqrt x) / 19 = 4 ∧ x = 5776 := by
  sorry

end square_root_divided_by_19_equals_4_l3952_395289


namespace initial_concentrated_kola_percentage_l3952_395222

/-- Proves that the initial percentage of concentrated kola in a solution is 9% -/
theorem initial_concentrated_kola_percentage
  (initial_volume : ℝ)
  (initial_water_percentage : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_concentrated_kola : ℝ)
  (final_sugar_percentage : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percentage = 64)
  (h3 : added_sugar = 3.2)
  (h4 : added_water = 8)
  (h5 : added_concentrated_kola = 6.8)
  (h6 : final_sugar_percentage = 26.536312849162012)
  (h7 : (((100 - initial_water_percentage - 9) * initial_volume / 100 + added_sugar) /
         (initial_volume + added_sugar + added_water + added_concentrated_kola)) * 100 = final_sugar_percentage) :
  9 = 100 - initial_water_percentage - ((initial_volume * initial_water_percentage / 100 + added_water) /
    (initial_volume + added_sugar + added_water + added_concentrated_kola) * 100) :=
by sorry

end initial_concentrated_kola_percentage_l3952_395222


namespace xiaolis_estimate_l3952_395296

theorem xiaolis_estimate (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (1.1 * x) / (0.9 * y) > x / y := by
  sorry

end xiaolis_estimate_l3952_395296


namespace garden_perimeter_proof_l3952_395212

/-- The perimeter of a rectangular garden with given length and breadth. -/
def garden_perimeter (length breadth : ℝ) : ℝ :=
  2 * length + 2 * breadth

/-- Theorem: The perimeter of a rectangular garden with length 375 m and breadth 100 m is 950 m. -/
theorem garden_perimeter_proof :
  garden_perimeter 375 100 = 950 := by
  sorry

end garden_perimeter_proof_l3952_395212


namespace no_solution_to_exponential_equation_l3952_395269

theorem no_solution_to_exponential_equation :
  ¬∃ (x y : ℝ), (9 : ℝ) ^ (x^3 + y) + (9 : ℝ) ^ (x + y^3) = 1 := by
  sorry

end no_solution_to_exponential_equation_l3952_395269


namespace digit_sum_in_base_d_l3952_395285

/-- A function to represent a two-digit number in base d -/
def two_digit_number (d a b : ℕ) : ℕ := a * d + b

/-- The problem statement -/
theorem digit_sum_in_base_d (d A B : ℕ) : 
  d > 8 →
  A < d →
  B < d →
  two_digit_number d A B + two_digit_number d A A - two_digit_number d B A = 180 →
  A + B = 10 := by
  sorry

end digit_sum_in_base_d_l3952_395285


namespace deposit_calculation_l3952_395247

theorem deposit_calculation (remaining_amount : ℝ) (deposit_percentage : ℝ) :
  remaining_amount = 950 ∧ deposit_percentage = 0.05 →
  (remaining_amount / (1 - deposit_percentage)) * deposit_percentage = 50 :=
by sorry

end deposit_calculation_l3952_395247


namespace last_digit_of_3_power_2023_l3952_395235

/-- The last digit of 3^n for n ≥ 1 -/
def lastDigitOf3Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | 0 => 1
  | _ => 0  -- This case should never occur

theorem last_digit_of_3_power_2023 :
  lastDigitOf3Power 2023 = 7 := by
  sorry

end last_digit_of_3_power_2023_l3952_395235


namespace prob_at_most_one_even_is_three_fourths_l3952_395286

/-- A die is fair if each number has an equal probability of 1/6 -/
def FairDie (d : Fin 6 → ℝ) : Prop :=
  ∀ n : Fin 6, d n = 1 / 6

/-- The probability of getting an even number on a fair die -/
def ProbEven (d : Fin 6 → ℝ) : ℝ :=
  d 1 + d 3 + d 5

/-- The probability of getting an odd number on a fair die -/
def ProbOdd (d : Fin 6 → ℝ) : ℝ :=
  d 0 + d 2 + d 4

/-- The probability of at most one die showing an even number when throwing two fair dice -/
def ProbAtMostOneEven (d1 d2 : Fin 6 → ℝ) : ℝ :=
  ProbOdd d1 * ProbOdd d2 + ProbOdd d1 * ProbEven d2 + ProbEven d1 * ProbOdd d2

theorem prob_at_most_one_even_is_three_fourths 
  (red blue : Fin 6 → ℝ) 
  (hred : FairDie red) 
  (hblue : FairDie blue) : 
  ProbAtMostOneEven red blue = 3/4 := by
  sorry

end prob_at_most_one_even_is_three_fourths_l3952_395286


namespace panda_bamboo_consumption_l3952_395210

/-- The amount of bamboo eaten by bigger pandas each day -/
def bigger_panda_bamboo : ℝ := 275

/-- The number of small pandas -/
def small_pandas : ℕ := 4

/-- The number of bigger pandas -/
def bigger_pandas : ℕ := 5

/-- The amount of bamboo eaten by small pandas each day -/
def small_panda_bamboo : ℝ := 25

/-- The total amount of bamboo eaten by all pandas in a week -/
def total_weekly_bamboo : ℝ := 2100

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem panda_bamboo_consumption :
  bigger_panda_bamboo * bigger_pandas * days_in_week +
  small_panda_bamboo * small_pandas * days_in_week =
  total_weekly_bamboo :=
sorry

end panda_bamboo_consumption_l3952_395210


namespace ellipse_eccentricity_l3952_395238

/-- Represents an ellipse with semi-major axis a and eccentricity e -/
structure Ellipse where
  a : ℝ
  e : ℝ

/-- The equation of the ellipse in terms of m -/
def ellipse_equation (m : ℝ) : Prop :=
  m > 1 ∧ ∃ x y : ℝ, x^2 / m^2 + y^2 / (m^2 - 1) = 1

/-- The distances from a point on the ellipse to its foci -/
def focus_distances (left right : ℝ) : Prop :=
  left = 3 ∧ right = 1

/-- The theorem stating the eccentricity of the ellipse -/
theorem ellipse_eccentricity (m : ℝ) :
  ellipse_equation m →
  (∃ left right : ℝ, focus_distances left right) →
  ∃ e : Ellipse, e.e = 1/2 :=
sorry

end ellipse_eccentricity_l3952_395238


namespace bob_has_31_pennies_l3952_395267

/-- The number of pennies Alex currently has -/
def alexPennies : ℕ := sorry

/-- The number of pennies Bob currently has -/
def bobPennies : ℕ := sorry

/-- If Alex gives Bob a penny, Bob will have four times as many pennies as Alex has -/
axiom condition1 : bobPennies + 1 = 4 * (alexPennies - 1)

/-- If Bob gives Alex a penny, Bob will have three times as many pennies as Alex has -/
axiom condition2 : bobPennies - 1 = 3 * (alexPennies + 1)

/-- Bob currently has 31 pennies -/
theorem bob_has_31_pennies : bobPennies = 31 := by sorry

end bob_has_31_pennies_l3952_395267


namespace no_simultaneous_integer_quotients_l3952_395294

theorem no_simultaneous_integer_quotients : ¬ ∃ (n : ℤ), (∃ (k : ℤ), n - 5 = 6 * k) ∧ (∃ (m : ℤ), n - 1 = 21 * m) := by
  sorry

end no_simultaneous_integer_quotients_l3952_395294


namespace a_2007_equals_4_l3952_395211

def f : ℕ → ℕ
  | 1 => 4
  | 2 => 1
  | 3 => 3
  | 4 => 5
  | 5 => 2
  | _ => 0

def a : ℕ → ℕ
  | 0 => 5
  | n + 1 => f (a n)

theorem a_2007_equals_4 : a 2007 = 4 := by
  sorry

end a_2007_equals_4_l3952_395211


namespace polynomial_coefficient_l3952_395272

theorem polynomial_coefficient (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^3 + x^10 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                         a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + 
                         a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₉ = -10 := by
sorry

end polynomial_coefficient_l3952_395272


namespace negation_of_universal_is_existential_l3952_395263

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Boy : U → Prop)
variable (LovesFootball : U → Prop)

-- State the theorem
theorem negation_of_universal_is_existential :
  (¬ ∀ x, Boy x → LovesFootball x) ↔ (∃ x, Boy x ∧ ¬ LovesFootball x) :=
by sorry

end negation_of_universal_is_existential_l3952_395263


namespace inequality_implies_upper_bound_l3952_395226

theorem inequality_implies_upper_bound (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ Real.exp x * (x^2 - x + 1) * (a*x + 3*a - 1) < 1) →
  a < 2/3 := by
sorry

end inequality_implies_upper_bound_l3952_395226


namespace largest_number_divisible_by_89_l3952_395232

def largest_n_digit_number (n : ℕ) : ℕ := 10^n - 1

def is_divisible_by_89 (x : ℕ) : Prop := x % 89 = 0

theorem largest_number_divisible_by_89 :
  ∃ (n : ℕ), 
    (n % 2 = 1) ∧ 
    (3 ≤ n) ∧ 
    (n ≤ 7) ∧ 
    (is_divisible_by_89 (largest_n_digit_number n)) ∧
    (∀ (m : ℕ), 
      (m % 2 = 1) → 
      (3 ≤ m) → 
      (m ≤ 7) → 
      (is_divisible_by_89 (largest_n_digit_number m)) → 
      (largest_n_digit_number m ≤ largest_n_digit_number n)) ∧
    (largest_n_digit_number n = 9999951) := by
  sorry

end largest_number_divisible_by_89_l3952_395232


namespace acid_concentration_percentage_l3952_395245

/-- 
Given a solution with a certain volume of pure acid and a total volume,
calculate the percentage concentration of pure acid in the solution.
-/
theorem acid_concentration_percentage 
  (pure_acid_volume : ℝ) 
  (total_solution_volume : ℝ) 
  (h1 : pure_acid_volume = 4.800000000000001)
  (h2 : total_solution_volume = 12) :
  (pure_acid_volume / total_solution_volume) * 100 = 40 := by
sorry

end acid_concentration_percentage_l3952_395245


namespace max_points_scored_l3952_395205

-- Define the variables
def total_shots : ℕ := 50
def three_point_success_rate : ℚ := 3 / 10
def two_point_success_rate : ℚ := 4 / 10

-- Define the function to calculate points
def calculate_points (three_point_shots : ℕ) : ℚ :=
  let two_point_shots : ℕ := total_shots - three_point_shots
  (three_point_success_rate * 3 * three_point_shots) + (two_point_success_rate * 2 * two_point_shots)

-- Theorem statement
theorem max_points_scored :
  ∃ (max_points : ℚ), max_points = 45 ∧
  ∀ (x : ℕ), x ≤ total_shots → calculate_points x ≤ max_points :=
sorry

end max_points_scored_l3952_395205


namespace derivative_at_two_l3952_395228

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2*x + 1

theorem derivative_at_two :
  (deriv f) 2 = 2 := by sorry

end derivative_at_two_l3952_395228


namespace log_x2y2_l3952_395204

theorem log_x2y2 (x y : ℝ) (h1 : Real.log (x^2 * y^5) = 2) (h2 : Real.log (x^3 * y^2) = 2) :
  Real.log (x^2 * y^2) = 16/11 := by
sorry

end log_x2y2_l3952_395204


namespace increase_in_position_for_given_slope_l3952_395274

/-- The increase in position for a person moving along a slope --/
def increase_in_position (slope_ratio : ℚ) (total_distance : ℝ) : ℝ :=
  sorry

/-- The theorem stating the increase in position for the given problem --/
theorem increase_in_position_for_given_slope : 
  increase_in_position (1/2) (100 * Real.sqrt 5) = 100 := by
  sorry

end increase_in_position_for_given_slope_l3952_395274


namespace expansion_unique_solution_l3952_395297

/-- The number of terms in the expansion of (a+b+c+d+1)^n that include all four variables
    a, b, c, and d, each to some positive power. -/
def num_terms (n : ℕ) : ℕ := Nat.choose n 4

/-- The proposition that n is the unique positive integer such that the expansion of (a+b+c+d+1)^n
    contains exactly 715 terms with all four variables a, b, c, and d each to some positive power. -/
def is_unique_solution (n : ℕ) : Prop :=
  n > 0 ∧ num_terms n = 715 ∧ ∀ m : ℕ, m ≠ n → num_terms m ≠ 715

theorem expansion_unique_solution :
  is_unique_solution 13 :=
sorry

end expansion_unique_solution_l3952_395297


namespace isosceles_triangle_l3952_395254

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- State the theorem
theorem isosceles_triangle (t : Triangle) 
  (h : 2 * Real.sin t.A * Real.cos t.B = Real.sin t.C) : 
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A :=
sorry

end isosceles_triangle_l3952_395254


namespace arithmetic_progression_x_value_l3952_395281

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The first three terms of the arithmetic progression -/
def first_three_terms (x : ℝ) : ℕ → ℝ
| 0 => x - 2
| 1 => x + 2
| 2 => 3*x + 4
| _ => 0  -- This is just a placeholder for other terms

theorem arithmetic_progression_x_value :
  ∀ x : ℝ, is_arithmetic_progression (first_three_terms x) → x = 1 :=
by sorry

end arithmetic_progression_x_value_l3952_395281


namespace two_digit_times_99_l3952_395258

theorem two_digit_times_99 (A B : ℕ) (h1 : A ≤ 9) (h2 : B ≤ 9) (h3 : A ≠ 0) :
  (10 * A + B) * 99 = 100 * (10 * A + B - 1) + (100 - (10 * A + B)) := by
  sorry

end two_digit_times_99_l3952_395258


namespace vectors_orthogonal_l3952_395219

def v1 : ℝ × ℝ := (3, -4)
def v2 : ℝ × ℝ := (8, 6)

theorem vectors_orthogonal : v1.1 * v2.1 + v1.2 * v2.2 = 0 := by
  sorry

end vectors_orthogonal_l3952_395219


namespace fold_reflection_l3952_395295

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given three points A, B, and C on a coordinate grid, if A coincides with B after folding,
    then C will coincide with the reflection of C across the perpendicular bisector of AB. -/
theorem fold_reflection (A B C : Point) (h : A.x = 10 ∧ A.y = 0 ∧ B.x = -6 ∧ B.y = 8 ∧ C.x = -4 ∧ C.y = 2) :
  ∃ (P : Point), P.x = 4 ∧ P.y = -2 ∧ 
  (2 * (C.x + P.x) = 2 * ((A.x + B.x) / 2)) ∧
  (C.y + P.y = 2 * ((A.y + B.y) / 2)) := by
  sorry


end fold_reflection_l3952_395295


namespace x_value_l3952_395268

theorem x_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end x_value_l3952_395268


namespace point_below_left_of_line_l3952_395249

-- Define the dice outcomes
def dice_outcomes : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the lines
def l1 (a b : ℕ) (x y : ℝ) : Prop := a * x + b * y = 2
def l2 (x y : ℝ) : Prop := x + 2 * y = 2

-- Define the probabilities
def p1 : ℚ := 1 / 18
def p2 : ℚ := 11 / 12

-- Define the point P
def P : ℝ × ℝ := (p1, p2)

-- Theorem statement
theorem point_below_left_of_line :
  (P.1 : ℝ) + 2 * (P.2 : ℝ) < 2 := by sorry

end point_below_left_of_line_l3952_395249


namespace simplify_expression_l3952_395236

theorem simplify_expression (x y : ℝ) :
  (15 * x + 35 * y) + (20 * x + 45 * y) - (8 * x + 40 * y) = 27 * x + 40 * y := by
  sorry

end simplify_expression_l3952_395236


namespace equation_solutions_l3952_395241

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 - 9 = 0 ↔ x = -4 ∨ x = 2) ∧
  (∀ x : ℝ, x^2 - 12*x - 4 = 0 ↔ x = 6 + 2*Real.sqrt 10 ∨ x = 6 - 2*Real.sqrt 10) ∧
  (∀ x : ℝ, 3*(x - 2)^2 = x*(x - 2) ↔ x = 2 ∨ x = 3) := by
  sorry

end equation_solutions_l3952_395241


namespace power_function_theorem_l3952_395209

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, ∀ x : ℝ, f x = x ^ n

-- Define the theorem
theorem power_function_theorem (f : ℝ → ℝ) (h : isPowerFunction f) :
  f 2 = 1/4 → f (1/2) = 4 := by
  sorry

end power_function_theorem_l3952_395209


namespace mrs_lim_revenue_l3952_395227

/-- Calculates the revenue from milk sales given the milk production and sales data --/
def milk_revenue (yesterday_morning : ℕ) (yesterday_evening : ℕ) (morning_decrease : ℕ) (remaining : ℕ) (price_per_gallon : ℚ) : ℚ :=
  let total_yesterday := yesterday_morning + yesterday_evening
  let this_morning := yesterday_morning - morning_decrease
  let total_milk := total_yesterday + this_morning
  let sold_milk := total_milk - remaining
  sold_milk * price_per_gallon

/-- Theorem stating that Mrs. Lim's revenue is $616 given the specified conditions --/
theorem mrs_lim_revenue :
  milk_revenue 68 82 18 24 (350/100) = 616 := by
  sorry

end mrs_lim_revenue_l3952_395227
