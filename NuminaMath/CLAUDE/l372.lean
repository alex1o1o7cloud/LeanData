import Mathlib

namespace solution_in_quadrant_I_l372_37286

theorem solution_in_quadrant_I (c : ℝ) : 
  (∃ x y : ℝ, x - 2*y = 4 ∧ 2*c*x + y = 5 ∧ x > 0 ∧ y > 0) ↔ 
  (-1/4 < c ∧ c < 5/8) :=
sorry

end solution_in_quadrant_I_l372_37286


namespace distinct_paintings_l372_37235

/-- The number of disks --/
def n : ℕ := 7

/-- The number of blue disks --/
def blue : ℕ := 4

/-- The number of red disks --/
def red : ℕ := 2

/-- The number of green disks --/
def green : ℕ := 1

/-- The number of symmetry operations (identity and reflection) --/
def symmetries : ℕ := 2

/-- The total number of colorings --/
def total_colorings : ℕ := (Nat.choose n blue) * (Nat.choose (n - blue) red) * (Nat.choose (n - blue - red) green)

/-- The number of colorings fixed by reflection --/
def fixed_colorings : ℕ := 3

/-- The theorem stating the number of distinct paintings --/
theorem distinct_paintings : (total_colorings + fixed_colorings) / symmetries = 54 := by
  sorry

end distinct_paintings_l372_37235


namespace batsman_average_increase_l372_37228

theorem batsman_average_increase (total_runs_before : ℕ) : 
  let innings_before : ℕ := 10
  let new_score : ℕ := 80
  let new_average : ℝ := 30
  let old_average : ℝ := total_runs_before / innings_before
  let increase : ℝ := new_average - old_average
  (total_runs_before + new_score) / (innings_before + 1) = new_average →
  increase = 5 := by
sorry

end batsman_average_increase_l372_37228


namespace greatest_integer_difference_l372_37223

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8) :
  ∃ (n : ℕ), n = 2 ∧ ∀ (m : ℕ), (∃ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 8 ∧ m = ⌊b - a⌋) → m ≤ n :=
sorry

end greatest_integer_difference_l372_37223


namespace bike_shop_profit_is_3000_l372_37262

/-- Calculates the profit of a bike shop given various parameters. -/
def bike_shop_profit (tire_repair_charge : ℕ) (tire_repair_cost : ℕ) (tire_repairs : ℕ)
                     (complex_repair_charge : ℕ) (complex_repair_cost : ℕ) (complex_repairs : ℕ)
                     (retail_profit : ℕ) (fixed_expenses : ℕ) : ℕ :=
  (tire_repairs * (tire_repair_charge - tire_repair_cost)) +
  (complex_repairs * (complex_repair_charge - complex_repair_cost)) +
  retail_profit - fixed_expenses

/-- Theorem stating that the bike shop's profit is $3000 under given conditions. -/
theorem bike_shop_profit_is_3000 :
  bike_shop_profit 20 5 300 300 50 2 2000 4000 = 3000 := by
  sorry

end bike_shop_profit_is_3000_l372_37262


namespace trip_time_difference_l372_37230

theorem trip_time_difference (total_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  average_speed = 40 →
  let first_half_time := (total_distance / 2) / first_half_speed
  let total_time := total_distance / average_speed
  let second_half_time := total_time - first_half_time
  (second_half_time - first_half_time) / first_half_time * 100 = 200 := by
  sorry

end trip_time_difference_l372_37230


namespace sum_of_cubes_l372_37276

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := by
sorry

end sum_of_cubes_l372_37276


namespace sum_of_three_numbers_l372_37233

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : b = 7)
  (h4 : (a + b + c) / 3 = a + 15)
  (h5 : (a + b + c) / 3 = c - 10) :
  a + b + c = 6 := by
  sorry

end sum_of_three_numbers_l372_37233


namespace power_division_equals_729_l372_37209

theorem power_division_equals_729 : 3^12 / 27^2 = 729 := by
  sorry

end power_division_equals_729_l372_37209


namespace half_day_percentage_l372_37207

def total_students : ℕ := 80
def full_day_students : ℕ := 60

theorem half_day_percentage :
  (total_students - full_day_students) / total_students * 100 = 25 := by
  sorry

end half_day_percentage_l372_37207


namespace complex_square_sum_l372_37212

theorem complex_square_sum (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (1 + 2 / Complex.I)^2 = a + b * Complex.I →
  a + b = -7 := by
  sorry

end complex_square_sum_l372_37212


namespace polynomial_simplification_l372_37221

theorem polynomial_simplification (x : ℝ) :
  (2 * x^3 - 5 * x^2 + 8 * x - 9) + (3 * x^4 - 2 * x^3 + x^2 - 8 * x + 6) = 3 * x^4 - 4 * x^2 - 3 := by
  sorry

end polynomial_simplification_l372_37221


namespace clubsuit_equality_theorem_l372_37293

-- Define the clubsuit operation
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the set of points (x, y) where x ⋆ y = y ⋆ x
def equality_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | clubsuit p.1 p.2 = clubsuit p.2 p.1}

-- Define the set of points on x-axis, y-axis, y = x, and y = -x
def target_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2}

theorem clubsuit_equality_theorem : equality_set = target_set := by
  sorry


end clubsuit_equality_theorem_l372_37293


namespace triangle_third_side_length_l372_37285

theorem triangle_third_side_length 
  (a b x : ℕ) 
  (ha : a = 1) 
  (hb : b = 5) 
  (hx : x > 0) :
  (a + b > x ∧ a + x > b ∧ b + x > a) → x = 5 :=
by sorry

end triangle_third_side_length_l372_37285


namespace function_expression_l372_37214

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (2 * x + 1) = x + 1) :
  ∀ x, f x = (1/2) * (x + 1) := by
sorry

end function_expression_l372_37214


namespace invertible_function_problem_l372_37246

theorem invertible_function_problem (g : ℝ → ℝ) (c : ℝ) 
  (h_invertible : Function.Bijective g)
  (h_gc : g c = 3)
  (h_g3 : g 3 = 5) :
  c - 3 = -3 := by
  sorry

end invertible_function_problem_l372_37246


namespace part_one_part_two_l372_37288

-- Part 1
theorem part_one : Real.sqrt 9 + 2 * Real.sin (30 * π / 180) - 1 = 3 := by sorry

-- Part 2
theorem part_two : 
  ∀ x : ℝ, (2*x - 3)^2 = 2*(2*x - 3) ↔ x = 3/2 ∨ x = 5/2 := by sorry

end part_one_part_two_l372_37288


namespace total_profit_is_60000_l372_37260

/-- Calculates the total profit of a partnership given the investments and one partner's share of the profit -/
def calculate_total_profit (a_investment b_investment c_investment : ℕ) (c_profit : ℕ) : ℕ :=
  let total_parts := a_investment / 9000 + b_investment / 9000 + c_investment / 9000
  let c_parts := c_investment / 9000
  let profit_per_part := c_profit / c_parts
  total_parts * profit_per_part

/-- Proves that the total profit is $60,000 given the specific investments and c's profit share -/
theorem total_profit_is_60000 :
  calculate_total_profit 45000 63000 72000 24000 = 60000 := by
  sorry

end total_profit_is_60000_l372_37260


namespace total_path_length_is_5pi_sqrt34_l372_37284

/-- Rectangle ABCD with given dimensions -/
structure Rectangle where
  AB : ℝ
  BC : ℝ

/-- Rotation parameters -/
structure RotationParams where
  firstAngle : ℝ  -- in radians
  secondAngle : ℝ  -- in radians

/-- Calculate the total path length of point A during rotations -/
def totalPathLength (rect : Rectangle) (rotParams : RotationParams) : ℝ :=
  sorry

/-- Theorem: The total path length of point A is 5π × √34 -/
theorem total_path_length_is_5pi_sqrt34 (rect : Rectangle) (rotParams : RotationParams) :
  rect.AB = 3 → rect.BC = 5 → rotParams.firstAngle = π → rotParams.secondAngle = 3 * π / 2 →
  totalPathLength rect rotParams = 5 * π * Real.sqrt 34 :=
sorry

end total_path_length_is_5pi_sqrt34_l372_37284


namespace walters_pocket_percentage_l372_37242

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of pennies Walter has -/
def num_pennies : ℕ := 2

/-- The number of nickels Walter has -/
def num_nickels : ℕ := 3

/-- The number of dimes Walter has -/
def num_dimes : ℕ := 2

/-- The total value of coins in Walter's pocket in cents -/
def total_value : ℕ := 
  num_pennies * penny_value + num_nickels * nickel_value + num_dimes * dime_value

/-- The percentage of one dollar that Walter has in his pocket -/
theorem walters_pocket_percentage :
  (total_value : ℚ) / 100 * 100 = 37 := by sorry

end walters_pocket_percentage_l372_37242


namespace sum_first_100_base6_l372_37264

/-- Represents a number in base 6 --/
def Base6 := Nat

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : Nat) : Base6 := sorry

/-- Adds two numbers in base 6 --/
def addBase6 (a b : Base6) : Base6 := sorry

/-- Multiplies two numbers in base 6 --/
def mulBase6 (a b : Base6) : Base6 := sorry

/-- Divides two numbers in base 6 --/
def divBase6 (a b : Base6) : Base6 := sorry

/-- Computes the sum of the first n (in base 6) natural numbers in base 6 --/
def sumFirstNBase6 (n : Base6) : Base6 := sorry

theorem sum_first_100_base6 :
  sumFirstNBase6 (toBase6 100) = toBase6 7222 := by sorry

end sum_first_100_base6_l372_37264


namespace mean_temperature_is_87_l372_37272

def temperatures : List ℝ := [84, 86, 85, 87, 89, 90, 88]

theorem mean_temperature_is_87 :
  (temperatures.sum / temperatures.length : ℝ) = 87 := by
  sorry

end mean_temperature_is_87_l372_37272


namespace abs_diff_eq_sum_abs_iff_product_nonpositive_l372_37205

theorem abs_diff_eq_sum_abs_iff_product_nonpositive (a b : ℝ) :
  |a - b| = |a| + |b| ↔ a * b ≤ 0 := by sorry

end abs_diff_eq_sum_abs_iff_product_nonpositive_l372_37205


namespace sixth_term_is_three_l372_37234

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_first_three : a 0 + a 1 + a 2 = 168
  specific_diff : a 1 - a 4 = 42

/-- The 6th term of the arithmetic progression is 3 -/
theorem sixth_term_is_three (ap : ArithmeticProgression) : ap.a 5 = 3 := by
  sorry

end sixth_term_is_three_l372_37234


namespace unique_solution_l372_37275

theorem unique_solution : ∃! (n : ℕ), n > 0 ∧ 5^29 * 4^15 = 2 * n^29 :=
by
  use 10
  constructor
  · sorry -- Proof that 10 satisfies the equation
  · sorry -- Proof of uniqueness

#check unique_solution

end unique_solution_l372_37275


namespace eighteen_tons_equals_18000kg_l372_37244

-- Define the conversion factor between tons and kilograms
def tons_to_kg (t : ℝ) : ℝ := 1000 * t

-- Theorem statement
theorem eighteen_tons_equals_18000kg : tons_to_kg 18 = 18000 := by
  sorry

end eighteen_tons_equals_18000kg_l372_37244


namespace reciprocal_of_negative_2023_l372_37254

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_negative_2023 :
  reciprocal (-2023) = -1 / 2023 := by
  sorry

end reciprocal_of_negative_2023_l372_37254


namespace division_problem_l372_37219

theorem division_problem (n : ℕ) : 
  n / 15 = 9 ∧ n % 15 = 1 → n = 136 := by
  sorry

end division_problem_l372_37219


namespace audrey_dream_fraction_l372_37289

theorem audrey_dream_fraction (total_sleep : ℝ) (not_dreaming : ℝ) 
  (h1 : total_sleep = 10)
  (h2 : not_dreaming = 6) :
  (total_sleep - not_dreaming) / total_sleep = 2 / 5 := by
  sorry

end audrey_dream_fraction_l372_37289


namespace original_equals_scientific_l372_37208

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 280000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 2.8
    exponent := 5
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end original_equals_scientific_l372_37208


namespace counterexample_existence_l372_37239

theorem counterexample_existence : ∃ (n : ℕ), ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n + 2)) ∧ n = 8 := by
  sorry

end counterexample_existence_l372_37239


namespace modulus_of_z_l372_37265

theorem modulus_of_z (z : ℂ) (h : z * (Complex.I + 1) = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by sorry

end modulus_of_z_l372_37265


namespace projections_of_P_l372_37283

def P : ℝ × ℝ × ℝ := (2, 3, 4)

def projection_planes : List (ℝ × ℝ × ℝ) := [(2, 3, 0), (0, 3, 4), (2, 0, 4)]
def projection_axes : List (ℝ × ℝ × ℝ) := [(2, 0, 0), (0, 3, 0), (0, 0, 4)]

theorem projections_of_P :
  (projection_planes = [(2, 3, 0), (0, 3, 4), (2, 0, 4)]) ∧
  (projection_axes = [(2, 0, 0), (0, 3, 0), (0, 0, 4)]) := by
  sorry

end projections_of_P_l372_37283


namespace paint_used_approx_253_33_l372_37210

/-- Calculate the amount of paint used over five weeks given an initial amount and weekly usage fractions. -/
def paintUsed (initialPaint : ℝ) (week1Fraction week2Fraction week3Fraction week4Fraction week5Fraction : ℝ) : ℝ :=
  let remainingAfterWeek1 := initialPaint * (1 - week1Fraction)
  let remainingAfterWeek2 := remainingAfterWeek1 * (1 - week2Fraction)
  let remainingAfterWeek3 := remainingAfterWeek2 * (1 - week3Fraction)
  let remainingAfterWeek4 := remainingAfterWeek3 * (1 - week4Fraction)
  let usedInWeek5 := remainingAfterWeek4 * week5Fraction
  initialPaint - remainingAfterWeek4 + usedInWeek5

/-- Theorem stating that given the initial paint amount and weekly usage fractions, 
    the total paint used after five weeks is approximately 253.33 gallons. -/
theorem paint_used_approx_253_33 :
  ∃ ε > 0, ε < 0.01 ∧ 
  |paintUsed 360 (1/9) (1/5) (1/3) (1/4) (1/6) - 253.33| < ε :=
sorry

end paint_used_approx_253_33_l372_37210


namespace magical_stack_with_89_fixed_has_266_cards_l372_37269

/-- Represents a stack of cards -/
structure CardStack :=
  (n : ℕ)
  (is_magical : Bool)
  (card_89_position : ℕ)

/-- Checks if a card stack is magical and card 89 retains its position -/
def is_magical_with_89_fixed (stack : CardStack) : Prop :=
  stack.is_magical ∧ stack.card_89_position = 89

/-- Theorem: A magical stack where card 89 retains its position has 266 cards -/
theorem magical_stack_with_89_fixed_has_266_cards (stack : CardStack) :
  is_magical_with_89_fixed stack → 2 * stack.n = 266 := by
  sorry

#check magical_stack_with_89_fixed_has_266_cards

end magical_stack_with_89_fixed_has_266_cards_l372_37269


namespace quadratic_equation_problem_l372_37216

theorem quadratic_equation_problem (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*(a-1)*x + a^2 - 7*a - 4 = 0 ↔ (x = x₁ ∨ x = x₂)) →
  x₁*x₂ - 3*x₁ - 3*x₂ - 2 = 0 →
  (1 + 4/(a^2 - 4)) * (a + 2)/a = 2 := by
sorry

end quadratic_equation_problem_l372_37216


namespace smallest_sum_of_reciprocals_l372_37251

theorem smallest_sum_of_reciprocals (a b : ℕ+) : 
  a ≠ b → 
  (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 → 
  (∀ c d : ℕ+, c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12 → (a : ℕ) + (b : ℕ) ≤ (c : ℕ) + (d : ℕ)) →
  (a : ℕ) + (b : ℕ) = 49 := by
sorry

end smallest_sum_of_reciprocals_l372_37251


namespace tricolor_triangles_odd_l372_37243

/-- Represents the color of a point -/
inductive Color
| Red
| Yellow
| Blue

/-- Represents a point in the triangle -/
structure Point where
  color : Color

/-- Represents a triangle ABC with m interior points -/
structure ColoredTriangle where
  m : ℕ
  A : Point
  B : Point
  C : Point
  interior_points : Fin m → Point

/-- A function that counts the number of triangles with vertices of all different colors -/
def count_tricolor_triangles (t : ColoredTriangle) : ℕ := sorry

/-- The main theorem stating that the number of triangles with vertices of all different colors is odd -/
theorem tricolor_triangles_odd (t : ColoredTriangle) 
  (h1 : t.A.color = Color.Red)
  (h2 : t.B.color = Color.Yellow)
  (h3 : t.C.color = Color.Blue) :
  Odd (count_tricolor_triangles t) := by sorry

end tricolor_triangles_odd_l372_37243


namespace arithmetic_calculations_l372_37245

theorem arithmetic_calculations :
  (2 / 19 * 8 / 25 + 17 / 25 / (19 / 2) = 2 / 19) ∧
  (1 / 4 * 125 * 1 / 25 * 8 = 10) ∧
  ((1 / 3 + 1 / 4) / (1 / 2 - 1 / 3) = 7 / 2) ∧
  ((1 / 6 + 1 / 8) * 24 * 1 / 9 = 7 / 9) :=
by sorry

end arithmetic_calculations_l372_37245


namespace tree_planting_variance_l372_37218

def group_data : List (Nat × Nat) := [(5, 3), (6, 4), (7, 3)]

def total_groups : Nat := (group_data.map Prod.snd).sum

theorem tree_planting_variance :
  let mean : Rat := (group_data.map (λ (x, y) => x * y)).sum / total_groups
  let variance : Rat := (group_data.map (λ (x, y) => y * ((x : Rat) - mean)^2)).sum / total_groups
  variance = 6/10 := by sorry

end tree_planting_variance_l372_37218


namespace triangle_inradius_l372_37256

theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h1 : p = 40) 
  (h2 : A = 50) 
  (h3 : A = r * p / 2) : r = 2.5 := by
  sorry

end triangle_inradius_l372_37256


namespace arithmetic_sequence_properties_l372_37282

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties 
  (a : ℕ → ℤ) (d : ℤ) (h : arithmetic_sequence a d) :
  (a 5 = -1 ∧ a 8 = 2 → a 1 = -5 ∧ d = 1) ∧
  (a 1 + a 6 = 12 ∧ a 4 = 7 → a 9 = 17) :=
by sorry

end arithmetic_sequence_properties_l372_37282


namespace right_triangle_hypotenuse_l372_37213

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  (a > 0) → (b > 0) → (c > 0) →
  (a^2 + b^2 = c^2) →  -- right-angled triangle condition
  (a^2 + b^2 + c^2 = 4500) →  -- sum of squares condition
  (a = 3*b) →  -- one leg is three times the other
  c = 15 * Real.sqrt 10 := by
sorry

end right_triangle_hypotenuse_l372_37213


namespace project_completion_theorem_l372_37240

/-- The number of days it takes to complete the project -/
def project_completion_time (a_time b_time : ℝ) (a_quit_before : ℝ) : ℝ :=
  let a_rate := 1 / a_time
  let b_rate := 1 / b_time
  let combined_rate := a_rate + b_rate
  15

/-- Theorem stating that the project will be completed in 15 days -/
theorem project_completion_theorem :
  project_completion_time 10 30 10 = 15 := by
  sorry

#eval project_completion_time 10 30 10

end project_completion_theorem_l372_37240


namespace largest_seven_digit_divisible_by_337_l372_37231

theorem largest_seven_digit_divisible_by_337 :
  ∀ n : ℕ, n ≤ 9999999 → n % 337 = 0 → n ≤ 9999829 :=
by
  sorry

end largest_seven_digit_divisible_by_337_l372_37231


namespace min_value_sum_product_l372_37270

theorem min_value_sum_product (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 12 := by
  sorry

end min_value_sum_product_l372_37270


namespace cube_sum_positive_l372_37261

theorem cube_sum_positive (x y z : ℝ) (h1 : x < y) (h2 : y < z) :
  (x - y)^3 + (y - z)^3 + (z - x)^3 > 0 := by
  sorry

end cube_sum_positive_l372_37261


namespace min_value_of_g_l372_37266

theorem min_value_of_g (φ : Real) (h1 : 0 < φ) (h2 : φ < π) : 
  let f := fun x => Real.sqrt 3 * Real.sin (2 * x + φ) + Real.cos (2 * x + φ)
  let g := fun x => f (x - 3 * π / 4)
  (∀ y, f (π / 12 - y) = f (π / 12 + y)) →
  (∃ x ∈ Set.Icc (-π / 4) (π / 6), g x = -1) ∧
  (∀ x ∈ Set.Icc (-π / 4) (π / 6), g x ≥ -1) :=
by sorry

end min_value_of_g_l372_37266


namespace triangle_property_l372_37201

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (b² + c² - a²) / cos A = 2 and (a cos B - b cos A) / (a cos B + b cos A) - b/c = 1,
    then bc = 1 and the area of triangle ABC is √3/4 -/
theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = Real.pi) →
  ((b^2 + c^2 - a^2) / Real.cos A = 2) →
  ((a * Real.cos B - b * Real.cos A) / (a * Real.cos B + b * Real.cos A) - b / c = 1) →
  (b * c = 1 ∧ (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4) := by
  sorry

end triangle_property_l372_37201


namespace set_operations_l372_37202

def U : Set ℕ := {x | x ≤ 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 2, 4, 6}

theorem set_operations :
  (A ∩ B = {2, 4}) ∧
  (U \ (A ∪ B) = {0, 3, 7}) := by
  sorry

end set_operations_l372_37202


namespace existence_of_ones_divisible_by_2019_l372_37258

theorem existence_of_ones_divisible_by_2019 :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, k > 0 ∧ (10^n - 1) / 9 = k * 2019) :=
sorry

end existence_of_ones_divisible_by_2019_l372_37258


namespace quadratic_equation_roots_l372_37277

theorem quadratic_equation_roots (k : ℝ) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 3 * x₁^2 + k * x₁ = 5 ∧ 3 * x₂^2 + k * x₂ = 5 :=
sorry

end quadratic_equation_roots_l372_37277


namespace work_completion_time_l372_37257

-- Define the work rates for each person
def amit_rate : ℚ := 1 / 15
def ananthu_rate : ℚ := 1 / 90
def chandra_rate : ℚ := 1 / 45

-- Define the number of days each person worked alone
def amit_solo_days : ℕ := 3
def ananthu_solo_days : ℕ := 6

-- Define the combined work rate of all three people
def combined_rate : ℚ := amit_rate + ananthu_rate + chandra_rate

-- Theorem statement
theorem work_completion_time : 
  let work_done_solo := amit_rate * amit_solo_days + ananthu_rate * ananthu_solo_days
  let remaining_work := 1 - work_done_solo
  let days_together := (remaining_work / combined_rate).ceil
  amit_solo_days + ananthu_solo_days + days_together = 17 := by
  sorry

end work_completion_time_l372_37257


namespace no_solution_absolute_value_plus_constant_l372_37295

theorem no_solution_absolute_value_plus_constant :
  ∀ x : ℝ, ¬(|5*x| + 7 = 0) :=
sorry

end no_solution_absolute_value_plus_constant_l372_37295


namespace pool_filling_time_l372_37267

theorem pool_filling_time (t1 t2 t_combined : ℝ) : 
  t1 = 8 → t_combined = 4.8 → 1/t1 + 1/t2 = 1/t_combined → t2 = 12 := by
sorry

end pool_filling_time_l372_37267


namespace min_value_a_l372_37238

theorem min_value_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ (x y : ℝ), x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 25) : 
  a ≥ 16 ∧ ∀ (ε : ℝ), ε > 0 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y) * (1/x + (16 - ε)/y) < 25 :=
sorry

end min_value_a_l372_37238


namespace prank_combinations_l372_37299

theorem prank_combinations (monday tuesday wednesday thursday friday saturday sunday : ℕ) :
  monday = 1 →
  tuesday = 2 →
  wednesday = 6 →
  thursday = 5 →
  friday = 0 →
  saturday = 2 →
  sunday = 1 →
  monday * tuesday * wednesday * thursday * friday * saturday * sunday = 0 := by
sorry

end prank_combinations_l372_37299


namespace incorrect_statement_l372_37224

theorem incorrect_statement : ¬(∀ m : ℝ, (∃ x : ℝ, x^2 + x - m = 0) → m > 0) := by
  sorry

end incorrect_statement_l372_37224


namespace flag_distance_not_nine_l372_37227

theorem flag_distance_not_nine (track_length : ℝ) (num_flags : ℕ) : 
  track_length = 90 → 
  num_flags = 10 → 
  (track_length / (num_flags - 1) ≠ 9) :=
by sorry

end flag_distance_not_nine_l372_37227


namespace egg_difference_l372_37268

/-- Given that Megan bought 2 dozen eggs, 3 eggs broke, and twice as many cracked,
    prove that the difference between the eggs in perfect condition and those that are cracked is 9. -/
theorem egg_difference (total : ℕ) (broken : ℕ) (cracked : ℕ) :
  total = 2 * 12 →
  broken = 3 →
  cracked = 2 * broken →
  total - broken - cracked - cracked = 9 :=
by sorry

end egg_difference_l372_37268


namespace min_hexagon_area_l372_37287

/-- Represents a configuration of two intersecting triangles -/
structure IntersectingTriangles where
  /-- The number of finite disjoint regions formed -/
  regions : Nat
  /-- The number of triangular regions -/
  triangular_regions : Nat
  /-- The area of each triangular region -/
  triangle_area : ℝ
  /-- The area of the hexagonal region -/
  hexagon_area : ℝ

/-- Theorem stating the minimum possible area of the hexagonal region -/
theorem min_hexagon_area (config : IntersectingTriangles) :
  config.regions = 7 →
  config.triangular_regions = 6 →
  config.triangle_area = 1 →
  config.hexagon_area ≥ 6 := by
  sorry

#check min_hexagon_area

end min_hexagon_area_l372_37287


namespace petyas_friends_l372_37247

theorem petyas_friends (total_stickers : ℕ) : 
  (∃ (friends : ℕ), total_stickers = 5 * friends + 8 ∧ total_stickers = 6 * friends - 11) →
  (∃ (friends : ℕ), friends = 19 ∧ total_stickers = 5 * friends + 8 ∧ total_stickers = 6 * friends - 11) :=
by sorry

end petyas_friends_l372_37247


namespace smallest_n_for_inequality_l372_37252

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧ (∀ m : ℕ, m < n → ∃ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) :=
sorry

end smallest_n_for_inequality_l372_37252


namespace gnomon_magic_diagonal_sums_equal_l372_37297

/-- Represents a 3x3 square --/
def Square := Matrix (Fin 3) (Fin 3) ℝ

/-- Checks if a 3x3 square is gnomon-magic --/
def is_gnomon_magic (s : Square) : Prop :=
  let sum1 := s 1 1 + s 1 2 + s 2 1 + s 2 2
  let sum2 := s 1 2 + s 1 3 + s 2 2 + s 2 3
  let sum3 := s 2 1 + s 2 2 + s 3 1 + s 3 2
  let sum4 := s 2 2 + s 2 3 + s 3 2 + s 3 3
  sum1 = sum2 ∧ sum2 = sum3 ∧ sum3 = sum4

/-- Calculates the sum of the main diagonal --/
def main_diagonal_sum (s : Square) : ℝ :=
  s 1 1 + s 2 2 + s 3 3

/-- Calculates the sum of the anti-diagonal --/
def anti_diagonal_sum (s : Square) : ℝ :=
  s 1 3 + s 2 2 + s 3 1

/-- Theorem: In a 3x3 gnomon-magic square, the sums of numbers along the two diagonals are equal --/
theorem gnomon_magic_diagonal_sums_equal (s : Square) (h : is_gnomon_magic s) :
  main_diagonal_sum s = anti_diagonal_sum s := by
  sorry

end gnomon_magic_diagonal_sums_equal_l372_37297


namespace highlight_film_average_time_l372_37296

/-- Given the footage times for 5 players, prove that the average time each player gets in the highlight film is 2 minutes -/
theorem highlight_film_average_time (point_guard shooting_guard small_forward power_forward center : ℕ) 
  (h1 : point_guard = 130)
  (h2 : shooting_guard = 145)
  (h3 : small_forward = 85)
  (h4 : power_forward = 60)
  (h5 : center = 180) :
  (point_guard + shooting_guard + small_forward + power_forward + center) / (5 * 60) = 2 := by
  sorry

end highlight_film_average_time_l372_37296


namespace power_product_simplification_l372_37290

theorem power_product_simplification :
  (-4/5 : ℚ)^2022 * (5/4 : ℚ)^2021 = 4/5 := by
  sorry

end power_product_simplification_l372_37290


namespace armands_guessing_game_l372_37250

theorem armands_guessing_game (x : ℕ) : x = 33 ↔ 3 * x = 2 * 51 - 3 := by
  sorry

end armands_guessing_game_l372_37250


namespace ab_value_l372_37222

-- Define the sets A and B
def A : Set ℝ := {-1.3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

-- State the theorem
theorem ab_value (a b : ℝ) (h : A = B a b) : a * b = 0.104 := by
  sorry

end ab_value_l372_37222


namespace max_value_of_expression_l372_37206

theorem max_value_of_expression (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 3) :
  (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) ≤ 1 ∧
  ∃ a b c, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 3 ∧
    (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 1 := by
  sorry

end max_value_of_expression_l372_37206


namespace reseating_problem_l372_37226

/-- Number of ways n people can be reseated according to the rules -/
def S : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | n + 3 => S (n + 2) + S (n + 1)

/-- The reseating problem for 12 people -/
theorem reseating_problem : S 12 = 89 := by
  sorry

end reseating_problem_l372_37226


namespace one_is_hilbert_number_h_hilbert_formula_larger_h_hilbert_number_l372_37279

-- Definition of a Hilbert number
def is_hilbert_number (p : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ p = x^2 + y^2 - x*y

-- Definition of an H Hilbert number
def is_h_hilbert_number (p : ℕ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ p = (2*n - 1)^2 + (2*n + 1)^2 - (2*n - 1)*(2*n + 1)

-- Theorem statements
theorem one_is_hilbert_number : is_hilbert_number 1 := by sorry

theorem h_hilbert_formula (n : ℕ) (h : n > 0) : 
  is_h_hilbert_number (4*n^2 + 3) := by sorry

theorem larger_h_hilbert_number (m n : ℕ) (hm : m > 0) (hn : n > 0) (h_diff : 4*n^2 + 3 - (4*m^2 + 3) = 48) :
  4*n^2 + 3 = 67 := by sorry

end one_is_hilbert_number_h_hilbert_formula_larger_h_hilbert_number_l372_37279


namespace car_discount_proof_l372_37249

/-- Given a car's original price and trading conditions, prove the discount percentage. -/
theorem car_discount_proof (P : ℝ) (P_b P_s : ℝ) (h1 : P > 0) (h2 : P_s = 1.60 * P_b) (h3 : P_s = 1.52 * P) : 
  ∃ D : ℝ, D = 0.05 ∧ P_b = P * (1 - D) := by
sorry

end car_discount_proof_l372_37249


namespace matrix_multiplication_result_l372_37255

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -1; 5, 2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![2, 6; -1, 3]

theorem matrix_multiplication_result :
  A * B = !![7, 15; 8, 36] := by sorry

end matrix_multiplication_result_l372_37255


namespace divisible_by_3_or_5_count_l372_37294

def count_divisible (n : Nat) : Nat :=
  (n / 3) + (n / 5) - (n / 15)

theorem divisible_by_3_or_5_count : count_divisible 46 = 21 := by
  sorry

end divisible_by_3_or_5_count_l372_37294


namespace joan_payment_l372_37225

/-- The amount Joan paid for her purchases, given the costs and change received -/
def amount_paid (cat_toy_cost cage_cost change : ℚ) : ℚ :=
  cat_toy_cost + cage_cost - change

/-- Theorem stating that Joan paid $19.48 for her purchases -/
theorem joan_payment : amount_paid 8.77 10.97 0.26 = 19.48 := by
  sorry

end joan_payment_l372_37225


namespace cos_495_degrees_l372_37259

theorem cos_495_degrees : Real.cos (495 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end cos_495_degrees_l372_37259


namespace f_range_l372_37271

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem f_range : ∀ x : ℝ, f x = -3 * π / 4 ∨ f x = π / 4 := by
  sorry

end f_range_l372_37271


namespace three_digit_difference_times_second_largest_l372_37291

def largest_three_digit (a b c : Nat) : Nat :=
  100 * max a (max b c) + 10 * max (min (max a b) (max b c)) (min a (min b c)) + min a (min b c)

def smallest_three_digit (a b c : Nat) : Nat :=
  100 * min a (min b c) + 10 * min (max (min a b) (min b c)) (max a (max b c)) + max a (max b c)

def second_largest_three_digit (a b c : Nat) : Nat :=
  let max_digit := max a (max b c)
  let min_digit := min a (min b c)
  let mid_digit := a + b + c - max_digit - min_digit
  100 * max_digit + 10 * mid_digit + min_digit

theorem three_digit_difference_times_second_largest (a b c : Nat) 
  (ha : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (hd : a ∈ [2, 5, 8] ∧ b ∈ [2, 5, 8] ∧ c ∈ [2, 5, 8]) : 
  (largest_three_digit a b c - smallest_three_digit a b c) * second_largest_three_digit a b c = 490050 := by
  sorry

end three_digit_difference_times_second_largest_l372_37291


namespace fisherman_multiple_is_three_l372_37241

/-- The multiple of fish caught by the fisherman compared to the pelican and kingfisher combined -/
def fisherman_multiple (pelican_fish kingfisher_fish fisherman_fish : ℕ) : ℚ :=
  fisherman_fish / (pelican_fish + kingfisher_fish)

/-- Theorem stating the multiple of fish caught by the fisherman -/
theorem fisherman_multiple_is_three :
  ∀ (pelican_fish kingfisher_fish fisherman_fish : ℕ),
    pelican_fish = 13 →
    kingfisher_fish = pelican_fish + 7 →
    fisherman_fish = pelican_fish + 86 →
    fisherman_multiple pelican_fish kingfisher_fish fisherman_fish = 3 := by
  sorry

#eval fisherman_multiple 13 20 99

end fisherman_multiple_is_three_l372_37241


namespace income_comparison_l372_37280

theorem income_comparison (A B : ℝ) (h : B = A * (1 + 1/3)) : 
  A = B * (1 - 1/4) := by
sorry

end income_comparison_l372_37280


namespace max_overlapping_squares_theorem_l372_37203

/-- Represents a square on the checkerboard -/
structure CheckerboardSquare where
  sideLength : Real
  (side_positive : sideLength > 0)

/-- Represents the square card -/
structure Card where
  sideLength : Real
  (side_positive : sideLength > 0)

/-- Calculates the maximum number of squares a card can overlap -/
def maxOverlappingSquares (square : CheckerboardSquare) (card : Card) (minOverlap : Real) : Nat :=
  sorry

theorem max_overlapping_squares_theorem (square : CheckerboardSquare) (card : Card) :
  square.sideLength = 0.75 →
  card.sideLength = 2 →
  maxOverlappingSquares square card 0.25 = 9 := by
  sorry

end max_overlapping_squares_theorem_l372_37203


namespace even_function_implies_k_equals_one_l372_37273

/-- A function f is even if f(-x) = f(x) for all x in its domain. -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = kx^2 + (k-1)x + 2. -/
def f (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + (k - 1) * x + 2

/-- If f(x) = kx^2 + (k-1)x + 2 is an even function, then k = 1. -/
theorem even_function_implies_k_equals_one :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
sorry

end even_function_implies_k_equals_one_l372_37273


namespace similar_triangle_scaling_l372_37253

theorem similar_triangle_scaling (base1 height1 base2 : ℝ) (height2 : ℝ) : 
  base1 = 12 → height1 = 6 → base2 = 8 → 
  (base1 / height1 = base2 / height2) → 
  height2 = 4 := by sorry

end similar_triangle_scaling_l372_37253


namespace veena_payment_fraction_l372_37204

/-- Represents the payment amounts of 6 friends at a restaurant -/
structure DinnerPayment where
  akshitha : ℚ
  veena : ℚ
  lasya : ℚ
  sandhya : ℚ
  ramesh : ℚ
  kavya : ℚ

/-- Theorem stating that Veena paid 1/8 of the total bill -/
theorem veena_payment_fraction (p : DinnerPayment) 
  (h1 : p.akshitha = 3/4 * p.veena)
  (h2 : p.veena = 1/2 * p.lasya)
  (h3 : p.lasya = 5/6 * p.sandhya)
  (h4 : p.sandhya = 4/8 * p.ramesh)
  (h5 : p.ramesh = 3/5 * p.kavya)
  : p.veena = 1/8 * (p.akshitha + p.veena + p.lasya + p.sandhya + p.ramesh + p.kavya) := by
  sorry


end veena_payment_fraction_l372_37204


namespace point_on_y_axis_implies_a_equals_two_l372_37217

/-- A point lies on the y-axis if and only if its x-coordinate is 0 -/
axiom point_on_y_axis (x y : ℝ) : (x, y) ∈ {p : ℝ × ℝ | p.1 = 0} ↔ x = 0

/-- The theorem states that if the point A(a-2, 2a+8) lies on the y-axis, then a = 2 -/
theorem point_on_y_axis_implies_a_equals_two (a : ℝ) :
  (a - 2, 2 * a + 8) ∈ {p : ℝ × ℝ | p.1 = 0} → a = 2 := by
  sorry

end point_on_y_axis_implies_a_equals_two_l372_37217


namespace six_point_configuration_exists_l372_37215

/-- A configuration of six points in 3D space -/
def Configuration := Fin 6 → ℝ × ℝ × ℝ

/-- Predicate to check if two line segments intersect only at their endpoints -/
def valid_intersection (a b c d : ℝ × ℝ × ℝ) : Prop :=
  (a = c ∧ b ≠ d) ∨ (a = d ∧ b ≠ c) ∨ (b = c ∧ a ≠ d) ∨ (b = d ∧ a ≠ c) ∨ (a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d)

/-- Predicate to check if a configuration is valid -/
def valid_configuration (config : Configuration) : Prop :=
  ∀ i j k l : Fin 6, i ≠ j → k ≠ l → i ≠ k ∨ i ≠ l → j ≠ k ∨ j ≠ l →
    valid_intersection (config i) (config j) (config k) (config l)

theorem six_point_configuration_exists : ∃ config : Configuration, valid_configuration config :=
sorry

end six_point_configuration_exists_l372_37215


namespace work_comparison_l372_37220

/-- Represents the amount of work that can be done by a group of people in a given time -/
structure WorkCapacity where
  people : ℕ
  work : ℝ
  days : ℕ

/-- Given two work capacities, proves that the first group did twice the initially considered work -/
theorem work_comparison (w1 w2 : WorkCapacity) : 
  w1.people = 3 ∧ 
  w1.days = 3 ∧ 
  w2.people = 6 ∧ 
  w2.days = 3 ∧ 
  w2.work = 6 * w1.work → 
  w1.work = 2 * w1.work := by
sorry

end work_comparison_l372_37220


namespace perfume_usage_fraction_l372_37298

/-- The fraction of perfume used in a cylindrical bottle -/
theorem perfume_usage_fraction 
  (r : ℝ) -- radius of the cylinder base
  (h : ℝ) -- height of the cylinder
  (v_remaining : ℝ) -- volume of remaining perfume in liters
  (hr : r = 7) -- given radius
  (hh : h = 10) -- given height
  (hv : v_remaining = 0.45) -- given remaining volume
  : (π * r^2 * h / 1000 - v_remaining) / (π * r^2 * h / 1000) = (49 * π - 45) / (49 * π) :=
by sorry

end perfume_usage_fraction_l372_37298


namespace log_base_2_negative_range_l372_37236

-- Define the function f(x) = lg x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log_base_2_negative_range :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  {x : ℝ | f x < 0} = Set.Ioo 0 1 := by sorry

end log_base_2_negative_range_l372_37236


namespace incorrect_operation_l372_37248

theorem incorrect_operation (a b : ℝ) : (-a^3)^2 * (-b^2)^3 = -a^6 * b^6 := by
  sorry

end incorrect_operation_l372_37248


namespace final_water_level_change_l372_37200

def water_level_change (initial_change : ℝ) (subsequent_change : ℝ) : ℝ :=
  initial_change + subsequent_change

theorem final_water_level_change :
  water_level_change (-3) 4 = 1 := by
  sorry

end final_water_level_change_l372_37200


namespace apple_ratio_problem_l372_37278

theorem apple_ratio_problem (green_apples red_apples : ℕ) : 
  (green_apples : ℚ) / red_apples = 5 / 3 → 
  green_apples = 15 → 
  red_apples = 9 := by
sorry

end apple_ratio_problem_l372_37278


namespace parallel_tangent_implies_a_le_one_l372_37292

open Real

/-- The function f(x) = ln x + (1/2)x^2 + ax has a tangent line parallel to 3x - y = 0 for some x > 0 -/
def has_parallel_tangent (a : ℝ) : Prop :=
  ∃ x > 0, (1 / x) + x + a = 3

/-- Theorem: If f(x) has a tangent line parallel to 3x - y = 0, then a ≤ 1 -/
theorem parallel_tangent_implies_a_le_one (a : ℝ) (h : has_parallel_tangent a) : a ≤ 1 := by
  sorry

end parallel_tangent_implies_a_le_one_l372_37292


namespace functional_eq_solution_l372_37263

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = max (f (x + y)) (f x * f y)

/-- The main theorem stating that any function satisfying the functional equation
    must be constant with values between 0 and 1, inclusive -/
theorem functional_eq_solution (f : ℝ → ℝ) (h : SatisfiesFunctionalEq f) :
    ∃ c : ℝ, (0 ≤ c ∧ c ≤ 1) ∧ (∀ x : ℝ, f x = c) :=
  sorry

end functional_eq_solution_l372_37263


namespace carolyn_sticker_count_l372_37211

/-- Given that Belle collected 97 stickers and Carolyn collected 18 fewer stickers than Belle,
    prove that Carolyn collected 79 stickers. -/
theorem carolyn_sticker_count :
  ∀ (belle_stickers carolyn_stickers : ℕ),
    belle_stickers = 97 →
    carolyn_stickers = belle_stickers - 18 →
    carolyn_stickers = 79 := by
  sorry

end carolyn_sticker_count_l372_37211


namespace fish_sample_count_l372_37229

/-- Given a population of fish and a stratified sampling method, 
    calculate the number of black carp and common carp in the sample. -/
theorem fish_sample_count 
  (total_fish : ℕ) 
  (black_carp : ℕ) 
  (common_carp : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_fish = 200) 
  (h2 : black_carp = 20) 
  (h3 : common_carp = 40) 
  (h4 : sample_size = 20) : 
  (black_carp * sample_size / total_fish + common_carp * sample_size / total_fish : ℕ) = 6 := by
  sorry

#check fish_sample_count

end fish_sample_count_l372_37229


namespace undeclared_majors_fraction_l372_37232

/-- Represents the distribution of students across different years -/
structure StudentDistribution where
  firstYear : Rat
  secondYear : Rat
  thirdYear : Rat
  fourthYear : Rat
  postgraduate : Rat

/-- Represents the proportion of students who have not declared a major in each year -/
structure UndeclaredMajors where
  firstYear : Rat
  secondYear : Rat
  thirdYear : Rat
  fourthYear : Rat
  postgraduate : Rat

/-- Calculates the fraction of all students who have not declared a major -/
def fractionUndeclaredMajors (dist : StudentDistribution) (undeclared : UndeclaredMajors) : Rat :=
  dist.firstYear * undeclared.firstYear +
  dist.secondYear * undeclared.secondYear +
  dist.thirdYear * undeclared.thirdYear +
  dist.fourthYear * undeclared.fourthYear +
  dist.postgraduate * undeclared.postgraduate

theorem undeclared_majors_fraction 
  (dist : StudentDistribution)
  (undeclared : UndeclaredMajors)
  (h1 : dist.firstYear = 1/5)
  (h2 : dist.secondYear = 2/5)
  (h3 : dist.thirdYear = 1/5)
  (h4 : dist.fourthYear = 1/10)
  (h5 : dist.postgraduate = 1/10)
  (h6 : undeclared.firstYear = 4/5)
  (h7 : undeclared.secondYear = 3/4)
  (h8 : undeclared.thirdYear = 1/3)
  (h9 : undeclared.fourthYear = 1/6)
  (h10 : undeclared.postgraduate = 1/12) :
  fractionUndeclaredMajors dist undeclared = 14/25 := by
  sorry


end undeclared_majors_fraction_l372_37232


namespace battery_mass_problem_l372_37237

theorem battery_mass_problem (x y : ℝ) 
  (eq1 : 2 * x + 2 * y = 72)
  (eq2 : 3 * x + 2 * y = 96) :
  x = 24 := by
sorry

end battery_mass_problem_l372_37237


namespace berry_temperature_theorem_l372_37274

theorem berry_temperature_theorem (temps : List Float) (avg : Float) : 
  temps.length = 6 ∧ 
  temps = [99.1, 98.2, 98.7, 99.3, 99, 98.9] ∧ 
  avg = 99 →
  (temps.sum + 99.8) / 7 = avg :=
by sorry

end berry_temperature_theorem_l372_37274


namespace sports_club_members_l372_37281

theorem sports_club_members (B T Both Neither : ℕ) 
  (hB : B = 17)
  (hT : T = 19)
  (hBoth : Both = 11)
  (hNeither : Neither = 2) :
  B + T - Both + Neither = 27 := by
  sorry

end sports_club_members_l372_37281
