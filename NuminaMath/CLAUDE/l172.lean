import Mathlib

namespace polygon_sides_l172_17244

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 = 3 * 360 + 180) →
  n = 9 := by
sorry

end polygon_sides_l172_17244


namespace bookshelf_discount_percentage_l172_17259

theorem bookshelf_discount_percentage (discount : ℝ) (final_price : ℝ) (tax_rate : ℝ) : 
  discount = 4.50 →
  final_price = 49.50 →
  tax_rate = 0.10 →
  (discount / (final_price / (1 + tax_rate) + discount)) * 100 = 9 := by
sorry

end bookshelf_discount_percentage_l172_17259


namespace probability_both_bins_contain_items_l172_17215

theorem probability_both_bins_contain_items (p : ℝ) (h1 : 0.5 < p) (h2 : p ≤ 1) :
  let prob_both := 1 - 2 * p^5 + p^10
  prob_both = (1 - p^5)^2 + p^10 := by
  sorry

end probability_both_bins_contain_items_l172_17215


namespace right_triangle_acute_angles_l172_17284

theorem right_triangle_acute_angles (θ : ℝ) : 
  θ = 27 → 
  90 + θ + (90 - θ) = 180 :=
by sorry

end right_triangle_acute_angles_l172_17284


namespace computer_peripherals_cost_fraction_l172_17261

theorem computer_peripherals_cost_fraction :
  let computer_cost : ℚ := 1500
  let base_video_card_cost : ℚ := 300
  let upgraded_video_card_cost : ℚ := 2 * base_video_card_cost
  let total_spent : ℚ := 2100
  let computer_with_upgrade_cost : ℚ := computer_cost + upgraded_video_card_cost - base_video_card_cost
  let peripherals_cost : ℚ := total_spent - computer_with_upgrade_cost
  peripherals_cost / computer_cost = 1 / 5 := by
sorry

end computer_peripherals_cost_fraction_l172_17261


namespace sliced_meat_variety_pack_l172_17276

theorem sliced_meat_variety_pack :
  let base_cost : ℚ := 40
  let rush_delivery_rate : ℚ := 0.3
  let cost_per_type : ℚ := 13
  let total_cost : ℚ := base_cost * (1 + rush_delivery_rate)
  (total_cost / cost_per_type : ℚ) = 4 := by
  sorry

end sliced_meat_variety_pack_l172_17276


namespace richard_twice_scott_age_l172_17243

/-- Represents the ages of the three brothers -/
structure BrothersAges where
  david : ℕ
  richard : ℕ
  scott : ℕ

/-- The current ages of the brothers -/
def currentAges : BrothersAges :=
  { david := 14
    richard := 20
    scott := 6 }

/-- The conditions given in the problem -/
axiom age_difference_richard_david : currentAges.richard = currentAges.david + 6
axiom age_difference_david_scott : currentAges.david = currentAges.scott + 8
axiom david_age_three_years_ago : currentAges.david = 11 + 3

/-- The theorem to be proved -/
theorem richard_twice_scott_age (x : ℕ) :
  x = 8 ↔ currentAges.richard + x = 2 * (currentAges.scott + x) :=
sorry

end richard_twice_scott_age_l172_17243


namespace system_solutions_l172_17295

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  x + y = (5 * x * y) / (1 + x * y) ∧
  y + z = (6 * y * z) / (1 + y * z) ∧
  z + x = (7 * z * x) / (1 + z * x)

-- Define the set of solutions
def solutions : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0),
   ((3 + Real.sqrt 5) / 2, 1, 2 + Real.sqrt 3),
   ((3 + Real.sqrt 5) / 2, 1, 2 - Real.sqrt 3),
   ((3 - Real.sqrt 5) / 2, 1, 2 + Real.sqrt 3),
   ((3 - Real.sqrt 5) / 2, 1, 2 - Real.sqrt 3)}

-- Theorem statement
theorem system_solutions :
  ∀ x y z, system x y z ↔ (x, y, z) ∈ solutions := by
  sorry

end system_solutions_l172_17295


namespace gloria_tickets_count_l172_17207

/-- Given that Gloria has 9 boxes of tickets and each box contains 5 tickets,
    prove that the total number of tickets is 45. -/
theorem gloria_tickets_count :
  let num_boxes : ℕ := 9
  let tickets_per_box : ℕ := 5
  num_boxes * tickets_per_box = 45 := by
  sorry

end gloria_tickets_count_l172_17207


namespace cannot_make_105_with_5_coins_l172_17287

def coin_denominations : List ℕ := [1, 5, 10, 25, 50]

def is_valid_sum (sum : ℕ) (n : ℕ) : Prop :=
  ∃ (coins : List ℕ), 
    coins.all (λ c => c ∈ coin_denominations) ∧ 
    coins.length = n ∧
    coins.sum = sum

theorem cannot_make_105_with_5_coins : 
  ¬ (is_valid_sum 105 5) :=
sorry

end cannot_make_105_with_5_coins_l172_17287


namespace total_rulers_problem_solution_l172_17292

/-- Given an initial number of rulers and a number of rulers added, 
    the total number of rulers is equal to the sum of the initial number and the added number. -/
theorem total_rulers (initial_rulers added_rulers : ℕ) :
  initial_rulers + added_rulers = initial_rulers + added_rulers := by sorry

/-- The specific case mentioned in the problem -/
theorem problem_solution :
  let initial_rulers : ℕ := 11
  let added_rulers : ℕ := 14
  initial_rulers + added_rulers = 25 := by sorry

end total_rulers_problem_solution_l172_17292


namespace cost_of_25_pencils_20_notebooks_l172_17282

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := sorry

/-- The cost of a notebook in dollars -/
def notebook_cost : ℝ := sorry

/-- The pricing conditions for pencils and notebooks -/
axiom pricing_condition_1 : 9 * pencil_cost + 10 * notebook_cost = 5.45
axiom pricing_condition_2 : 7 * pencil_cost + 6 * notebook_cost = 3.67
axiom pricing_condition_3 : 20 * pencil_cost + 15 * notebook_cost = 10.00

/-- The theorem stating the cost of 25 pencils and 20 notebooks -/
theorem cost_of_25_pencils_20_notebooks : 
  25 * pencil_cost + 20 * notebook_cost = 12.89 := by sorry

end cost_of_25_pencils_20_notebooks_l172_17282


namespace preimage_of_three_l172_17270

def A : Set ℝ := Set.univ
def B : Set ℝ := Set.univ

def f : ℝ → ℝ := fun x ↦ 2 * x - 1

theorem preimage_of_three (h : f 2 = 3) : 
  ∃ x ∈ A, f x = 3 ∧ x = 2 :=
sorry

end preimage_of_three_l172_17270


namespace solution_approximation_l172_17290

theorem solution_approximation : ∃ x : ℝ, 
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * x * 0.5)) = 2800.0000000000005 ∧ 
  abs (x - 0.225) < 0.0001 := by
  sorry

end solution_approximation_l172_17290


namespace selection_ways_eq_six_l172_17248

/-- The number of types of pencils -/
def num_pencil_types : ℕ := 3

/-- The number of types of erasers -/
def num_eraser_types : ℕ := 2

/-- The number of ways to select one pencil and one eraser -/
def num_selection_ways : ℕ := num_pencil_types * num_eraser_types

/-- Theorem stating that the number of ways to select one pencil and one eraser is 6 -/
theorem selection_ways_eq_six : num_selection_ways = 6 := by
  sorry

end selection_ways_eq_six_l172_17248


namespace ellipse_and_trajectory_l172_17228

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Point A is on the ellipse C -/
def point_A_on_C (a b : ℝ) : Prop := ellipse_C 1 (3/2) a b

/-- Sum of distances from A to foci equals 4 -/
def sum_distances_4 (a : ℝ) : Prop := 2 * a = 4

/-- Conditions on a and b -/
def a_b_conditions (a b : ℝ) : Prop := a > b ∧ b > 0

/-- Theorem stating the equation of ellipse C and the trajectory of midpoint M -/
theorem ellipse_and_trajectory (a b : ℝ) 
  (h1 : a_b_conditions a b) 
  (h2 : point_A_on_C a b) 
  (h3 : sum_distances_4 a) : 
  (∀ x y, ellipse_C x y a b ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ F1 : ℝ × ℝ, ∀ x y, 
    (∃ x1 y1, ellipse_C x1 y1 a b ∧ x = (F1.1 + x1) / 2 ∧ y = (F1.2 + y1) / 2) ↔ 
    (x + 1/2)^2 + 4*y^2 / 3 = 1) :=
sorry

end ellipse_and_trajectory_l172_17228


namespace domain_relationship_l172_17252

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2x+1)
def domain_f_2x_plus_1 : Set ℝ := Set.Icc (-3) 3

-- Theorem stating the relationship between the domains
theorem domain_relationship :
  (∀ y ∈ domain_f_2x_plus_1, ∃ x, y = 2*x + 1) →
  {x : ℝ | f x ≠ 0} = Set.Icc (-5) 7 :=
sorry

end domain_relationship_l172_17252


namespace building_entry_exit_ways_l172_17241

/-- The number of ways to enter and exit a building with 4 doors, entering and exiting through different doors -/
def number_of_ways (num_doors : ℕ) : ℕ :=
  num_doors * (num_doors - 1)

/-- Theorem stating that for a building with 4 doors, there are 12 ways to enter and exit through different doors -/
theorem building_entry_exit_ways :
  number_of_ways 4 = 12 := by
  sorry

end building_entry_exit_ways_l172_17241


namespace remaining_credit_l172_17250

/-- Calculates the remaining credit to be paid given a credit limit and two payments -/
theorem remaining_credit (credit_limit : ℕ) (payment1 : ℕ) (payment2 : ℕ) :
  credit_limit = 100 →
  payment1 = 15 →
  payment2 = 23 →
  credit_limit - (payment1 + payment2) = 62 := by
  sorry

#check remaining_credit

end remaining_credit_l172_17250


namespace skew_iff_b_neq_4_l172_17229

def line1 (b t : ℝ) : ℝ × ℝ × ℝ := (2 + 3*t, 3 + 4*t, b + 5*t)
def line2 (u : ℝ) : ℝ × ℝ × ℝ := (5 + 6*u, 2 + 3*u, 1 + 2*u)

def are_skew (b : ℝ) : Prop :=
  ∀ t u : ℝ, line1 b t ≠ line2 u

theorem skew_iff_b_neq_4 (b : ℝ) :
  are_skew b ↔ b ≠ 4 :=
sorry

end skew_iff_b_neq_4_l172_17229


namespace expression_evaluation_l172_17281

theorem expression_evaluation (a b : ℤ) (h1 : a = -1) (h2 : b = 2) :
  (2*a + b) - 2*(3*a - 2*b) = 14 := by
  sorry

end expression_evaluation_l172_17281


namespace logarithmic_function_fixed_point_l172_17238

theorem logarithmic_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log a + 1
  f 1 = 1 := by sorry

end logarithmic_function_fixed_point_l172_17238


namespace dave_chocolate_boxes_l172_17253

theorem dave_chocolate_boxes (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) : 
  total_boxes = 12 → pieces_per_box = 3 → pieces_left = 21 →
  (total_boxes * pieces_per_box - pieces_left) / pieces_per_box = 5 := by
sorry

end dave_chocolate_boxes_l172_17253


namespace probability_all_white_balls_l172_17231

/-- The probability of drawing 7 white balls from a box containing 7 white and 8 black balls -/
theorem probability_all_white_balls (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) :
  total_balls = white_balls + black_balls →
  total_balls = 15 →
  white_balls = 7 →
  black_balls = 8 →
  drawn_balls = 7 →
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 6435 :=
by sorry

end probability_all_white_balls_l172_17231


namespace min_value_inequality_l172_17271

theorem min_value_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 12 :=
by sorry

end min_value_inequality_l172_17271


namespace cereal_spending_l172_17223

theorem cereal_spending (total : ℝ) (snap crackle pop : ℝ) : 
  total = 150 ∧ 
  snap = 2 * crackle ∧ 
  crackle = 3 * pop ∧ 
  total = snap + crackle + pop → 
  pop = 15 := by
  sorry

end cereal_spending_l172_17223


namespace sam_total_pennies_l172_17279

def initial_pennies : ℕ := 98
def found_pennies : ℕ := 93

theorem sam_total_pennies :
  initial_pennies + found_pennies = 191 := by
  sorry

end sam_total_pennies_l172_17279


namespace special_polynomial_max_value_l172_17274

/-- A polynomial with integer coefficients satisfying certain conditions -/
def SpecialPolynomial (p : ℤ → ℤ) : Prop :=
  (∀ m n : ℤ, (p m - p n) ∣ (m^2 - n^2)) ∧ 
  p 0 = 1 ∧ 
  p 1 = 2

/-- The maximum value of p(100) for a SpecialPolynomial p is 10001 -/
theorem special_polynomial_max_value : 
  ∀ p : ℤ → ℤ, SpecialPolynomial p → p 100 ≤ 10001 :=
by sorry

end special_polynomial_max_value_l172_17274


namespace pure_imaginary_modulus_l172_17262

theorem pure_imaginary_modulus (b : ℝ) : 
  (Complex.I : ℂ).re * ((2 + b * Complex.I) * (2 - Complex.I)).re = 0 ∧ 
  (Complex.I : ℂ).im * ((2 + b * Complex.I) * (2 - Complex.I)).im ≠ 0 → 
  Complex.abs (1 + b * Complex.I) = Real.sqrt 17 := by
  sorry

end pure_imaginary_modulus_l172_17262


namespace games_required_equals_participants_minus_one_l172_17232

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  participants : ℕ
  games_required : ℕ

/-- The number of games required in a single-elimination tournament is one less than the number of participants -/
theorem games_required_equals_participants_minus_one 
  (tournament : SingleEliminationTournament) 
  (h : tournament.participants = 512) : 
  tournament.games_required = 511 := by
  sorry

end games_required_equals_participants_minus_one_l172_17232


namespace absolute_value_inequality_find_a_value_l172_17257

-- Part 1
theorem absolute_value_inequality (x : ℝ) :
  (|x - 1| + |x + 2| ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) := by sorry

-- Part 2
theorem find_a_value (a : ℝ) :
  (∀ x, |a*x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3) → a = -3 := by sorry

end absolute_value_inequality_find_a_value_l172_17257


namespace cage_cost_proof_l172_17280

def snake_toy_cost : ℝ := 11.76
def dollar_found : ℝ := 1
def total_cost : ℝ := 26.3

theorem cage_cost_proof :
  total_cost - (snake_toy_cost + dollar_found) = 13.54 := by
  sorry

end cage_cost_proof_l172_17280


namespace ellipse_focus_k_l172_17202

-- Define the ellipse
def ellipse (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 2) + y^2 / 9 = 1

-- Define the focus
def focus : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem ellipse_focus_k (k : ℝ) :
  (∀ x y, ellipse k x y) → (focus.1 = 0 ∧ focus.2 = 2) → k = 3 :=
by sorry

end ellipse_focus_k_l172_17202


namespace nail_painting_problem_l172_17263

theorem nail_painting_problem (total_nails purple_nails blue_nails : ℕ) 
  (h1 : total_nails = 20)
  (h2 : purple_nails = 6)
  (h3 : blue_nails = 8)
  (h4 : (blue_nails : ℚ) / total_nails - (striped_nails : ℚ) / total_nails = 1/10) :
  striped_nails = 6 :=
by
  sorry

#check nail_painting_problem

end nail_painting_problem_l172_17263


namespace isosceles_triangle_perimeter_l172_17278

/-- A triangle with sides a, b, and c is isosceles if at least two sides are equal -/
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

/-- The triangle inequality theorem -/
def SatisfiesTriangleInequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- The perimeter of a triangle -/
def Perimeter (a b c : ℝ) : ℝ :=
  a + b + c

/-- The unique isosceles triangle with sides 10 and 22 has perimeter 54 -/
theorem isosceles_triangle_perimeter :
  ∃! (a b c : ℝ),
    a = 10 ∧ (b = 22 ∨ c = 22) ∧
    IsIsosceles a b c ∧
    SatisfiesTriangleInequality a b c ∧
    Perimeter a b c = 54 := by
  sorry

end isosceles_triangle_perimeter_l172_17278


namespace four_digit_perfect_square_same_digits_l172_17260

theorem four_digit_perfect_square_same_digits : ∃ n : ℕ,
  (1000 ≤ n) ∧ (n < 10000) ∧  -- four-digit number
  (∃ m : ℕ, n = m^2) ∧  -- perfect square
  (∃ a b : ℕ, n = 1100 * a + 11 * b) ∧  -- first two digits same, last two digits same
  (n = 7744) :=
by sorry

end four_digit_perfect_square_same_digits_l172_17260


namespace fraction_irreducible_l172_17213

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end fraction_irreducible_l172_17213


namespace art_supply_sales_percentage_l172_17273

theorem art_supply_sales_percentage (total_percentage brush_percentage paint_percentage : ℝ) :
  total_percentage = 100 ∧
  brush_percentage = 45 ∧
  paint_percentage = 22 →
  total_percentage - brush_percentage - paint_percentage = 33 :=
by
  sorry

end art_supply_sales_percentage_l172_17273


namespace prob_red_two_cans_l172_17299

/-- Represents a can containing red and white balls -/
structure Can where
  red : ℕ
  white : ℕ

/-- The probability of drawing a red ball from a can -/
def probRed (c : Can) : ℚ :=
  c.red / (c.red + c.white)

/-- The probability of drawing a white ball from a can -/
def probWhite (c : Can) : ℚ :=
  c.white / (c.red + c.white)

/-- The probability of drawing a red ball from can B after transferring a ball from can A -/
def probRedAfterTransfer (a b : Can) : ℚ :=
  probRed a * probRed (Can.mk (b.red + 1) b.white) +
  probWhite a * probRed (Can.mk b.red (b.white + 1))

theorem prob_red_two_cans (a b : Can) (ha : a.red = 2 ∧ a.white = 3) (hb : b.red = 4 ∧ b.white = 1) :
  probRedAfterTransfer a b = 11 / 15 := by
  sorry

end prob_red_two_cans_l172_17299


namespace inequality_solution_range_l172_17205

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) ↔ a ∈ Set.Ioi 3 ∪ Set.Iio 1 :=
sorry

end inequality_solution_range_l172_17205


namespace min_benches_for_equal_occupancy_l172_17249

/-- Represents the capacity of a bench for adults and children -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Finds the minimum number of benches required for equal and full occupancy -/
def minBenchesRequired (capacity : BenchCapacity) : Nat :=
  Nat.lcm capacity.adults capacity.children / capacity.adults

/-- Theorem stating the minimum number of benches required -/
theorem min_benches_for_equal_occupancy (capacity : BenchCapacity) 
  (h1 : capacity.adults = 8) 
  (h2 : capacity.children = 12) : 
  minBenchesRequired capacity = 3 := by
  sorry

#eval minBenchesRequired ⟨8, 12⟩

end min_benches_for_equal_occupancy_l172_17249


namespace quadratic_other_root_l172_17291

theorem quadratic_other_root 
  (a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : a * 2^2 = b) : 
  a * (-2)^2 = b := by
sorry

end quadratic_other_root_l172_17291


namespace integral_equals_three_implies_k_equals_four_l172_17268

theorem integral_equals_three_implies_k_equals_four (k : ℝ) : 
  (∫ x in (0:ℝ)..(1:ℝ), 3 * x^2 + k * x) = 3 → k = 4 := by
sorry

end integral_equals_three_implies_k_equals_four_l172_17268


namespace isosceles_triangle_l172_17200

/-- Given a triangle ABC where sin A = 2 sin C cos B, prove that B = C -/
theorem isosceles_triangle (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_sin : Real.sin A = 2 * Real.sin C * Real.cos B) : B = C := by
  sorry

end isosceles_triangle_l172_17200


namespace tax_calculation_l172_17239

/-- Calculates the tax paid given gross pay and net pay -/
def tax_paid (gross_pay : ℕ) (net_pay : ℕ) : ℕ :=
  gross_pay - net_pay

/-- Theorem stating that the tax paid is 135 dollars given the conditions -/
theorem tax_calculation (gross_pay net_pay : ℕ) 
  (h1 : gross_pay = 450)
  (h2 : net_pay = 315)
  (h3 : tax_paid gross_pay net_pay = gross_pay - net_pay) :
  tax_paid gross_pay net_pay = 135 := by
  sorry

end tax_calculation_l172_17239


namespace days_to_pay_for_register_is_8_l172_17233

/-- The number of days required to pay for a cash register given daily sales and costs -/
def days_to_pay_for_register (register_cost : ℕ) (bread_sold : ℕ) (bread_price : ℕ) 
  (cakes_sold : ℕ) (cake_price : ℕ) (rent : ℕ) (electricity : ℕ) : ℕ :=
  let daily_revenue := bread_sold * bread_price + cakes_sold * cake_price
  let daily_expenses := rent + electricity
  let daily_profit := daily_revenue - daily_expenses
  (register_cost + daily_profit - 1) / daily_profit

theorem days_to_pay_for_register_is_8 :
  days_to_pay_for_register 1040 40 2 6 12 20 2 = 8 := by sorry

end days_to_pay_for_register_is_8_l172_17233


namespace students_playing_cricket_l172_17204

theorem students_playing_cricket 
  (total_students : ℕ) 
  (football_players : ℕ) 
  (neither_players : ℕ) 
  (both_players : ℕ) 
  (h1 : total_students = 450)
  (h2 : football_players = 325)
  (h3 : neither_players = 50)
  (h4 : both_players = 100)
  : ∃ cricket_players : ℕ, cricket_players = 175 :=
by
  sorry

end students_playing_cricket_l172_17204


namespace contractor_payment_l172_17209

/-- A contractor's payment problem -/
theorem contractor_payment
  (total_days : ℕ)
  (payment_per_day : ℚ)
  (fine_per_day : ℚ)
  (absent_days : ℕ)
  (h1 : total_days = 30)
  (h2 : payment_per_day = 25)
  (h3 : fine_per_day = 7.5)
  (h4 : absent_days = 10)
  (h5 : absent_days ≤ total_days) :
  (total_days - absent_days) * payment_per_day - absent_days * fine_per_day = 425 :=
by sorry

end contractor_payment_l172_17209


namespace fraction_equality_implies_values_l172_17297

theorem fraction_equality_implies_values (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 4 →
    (B * x - 17) / (x^2 - 7*x + 12) = A / (x - 3) + 4 / (x - 4)) →
  A = 5/4 ∧ B = 21/4 ∧ A + B = 13/2 := by
  sorry

end fraction_equality_implies_values_l172_17297


namespace product_97_103_l172_17269

theorem product_97_103 : 97 * 103 = 9991 := by
  sorry

end product_97_103_l172_17269


namespace smallest_c_inequality_l172_17210

theorem smallest_c_inequality (c : ℝ) : 
  (∀ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 → 
    (x * y * z) ^ (1/3 : ℝ) + c * |x - y + z| ≥ (x + y + z) / 3) ↔ 
  c ≥ 1/3 :=
sorry

end smallest_c_inequality_l172_17210


namespace geometric_sequence_second_term_l172_17255

theorem geometric_sequence_second_term 
  (a : ℕ → ℚ) -- a is the sequence
  (h1 : a 3 = 12) -- third term is 12
  (h2 : a 4 = 18) -- fourth term is 18
  (h3 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * (a 4 / a 3)) -- definition of geometric sequence
  : a 2 = 8 := by
sorry

end geometric_sequence_second_term_l172_17255


namespace five_solutions_l172_17203

/-- The number of integer solution pairs (x, y) to the equation √x + √y = √336 -/
def num_solutions : ℕ := 5

/-- A predicate that checks if a pair of natural numbers satisfies the equation -/
def is_solution (x y : ℕ) : Prop :=
  Real.sqrt (x : ℝ) + Real.sqrt (y : ℝ) = Real.sqrt 336

/-- The theorem stating that there are exactly 5 solution pairs -/
theorem five_solutions :
  ∃! (s : Finset (ℕ × ℕ)), s.card = num_solutions ∧ ∀ (x y : ℕ), (x, y) ∈ s ↔ is_solution x y :=
sorry

end five_solutions_l172_17203


namespace complex_real_condition_l172_17267

theorem complex_real_condition (m : ℝ) : 
  (((m : ℂ) + Complex.I) / (1 - Complex.I)).im = 0 → m = -1 :=
by sorry

end complex_real_condition_l172_17267


namespace line_relationship_l172_17288

-- Define a type for lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (intersecting : Line → Line → Prop)

-- State the theorem
theorem line_relationship (a b c : Line) 
  (h1 : skew a b) 
  (h2 : parallel c a) : 
  skew c b ∨ intersecting c b :=
sorry

end line_relationship_l172_17288


namespace tan_eleven_pi_over_four_l172_17206

theorem tan_eleven_pi_over_four : Real.tan (11 * π / 4) = -1 := by
  sorry

end tan_eleven_pi_over_four_l172_17206


namespace smallest_w_l172_17264

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w : 
  let w := 2571912
  ∀ x : ℕ, x > 0 →
    (is_factor (2^5) (3692 * x) ∧
     is_factor (3^4) (3692 * x) ∧
     is_factor (7^3) (3692 * x) ∧
     is_factor (17^2) (3692 * x)) →
    x ≥ w :=
by sorry

end smallest_w_l172_17264


namespace sum_of_roots_quadratic_l172_17236

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 3*x - 4 = 0 ↔ x = x₁ ∨ x = x₂) → x₁ + x₂ = 3 := by
  sorry

end sum_of_roots_quadratic_l172_17236


namespace amy_connor_score_difference_l172_17234

theorem amy_connor_score_difference
  (connor_score : ℕ)
  (amy_score : ℕ)
  (jason_score : ℕ)
  (connor_scored_two : connor_score = 2)
  (amy_scored_more : amy_score > connor_score)
  (jason_scored_twice_amy : jason_score = 2 * amy_score)
  (team_total_score : connor_score + amy_score + jason_score = 20) :
  amy_score - connor_score = 4 := by
sorry

end amy_connor_score_difference_l172_17234


namespace expression_simplification_and_evaluation_expression_evaluation_at_3_l172_17242

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) :
  ((x + 1) / (x - 2) + 1) / ((x^2 - 2*x) / (x^2 - 4*x + 4)) = (2*x - 1) / x :=
by sorry

theorem expression_evaluation_at_3 :
  let x : ℝ := 3
  ((x + 1) / (x - 2) + 1) / ((x^2 - 2*x) / (x^2 - 4*x + 4)) = 5/3 :=
by sorry

end expression_simplification_and_evaluation_expression_evaluation_at_3_l172_17242


namespace probability_select_four_or_five_l172_17221

/-- The probability of selecting a product with a number not less than 4 from 5 products -/
theorem probability_select_four_or_five (n : ℕ) (h : n = 5) :
  (Finset.filter (λ i => i ≥ 4) (Finset.range n)).card / n = 2 / 5 := by
  sorry

end probability_select_four_or_five_l172_17221


namespace quadratic_polynomials_sum_l172_17220

/-- Two distinct quadratic polynomials with the given properties have a + c = -600 -/
theorem quadratic_polynomials_sum (a b c d : ℝ) : 
  let f (x : ℝ) := x^2 + a*x + b
  let g (x : ℝ) := x^2 + c*x + d
  ∀ x y : ℝ, 
  (f ≠ g) →  -- f and g are distinct
  (g (-a/2) = 0) →  -- x-coordinate of vertex of f is a root of g
  (f (-c/2) = 0) →  -- x-coordinate of vertex of g is a root of f
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ g x ≥ m) →  -- f and g yield the same minimum value
  (f 150 = -200 ∧ g 150 = -200) →  -- f and g intersect at (150, -200)
  a + c = -600 := by
  sorry

end quadratic_polynomials_sum_l172_17220


namespace cube_through_cube_l172_17285

theorem cube_through_cube (a : ℝ) (h : a > 0) : ∃ (s : ℝ), s > a ∧ s = (2 * a * Real.sqrt 2) / 3 := by
  sorry

end cube_through_cube_l172_17285


namespace mango_rate_calculation_l172_17277

/-- The rate per kg of mangoes given the purchase details --/
def mango_rate (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (total_payment : ℕ) : ℕ :=
  (total_payment - grape_quantity * grape_rate) / mango_quantity

theorem mango_rate_calculation :
  mango_rate 9 70 9 1125 = 55 := by
  sorry

end mango_rate_calculation_l172_17277


namespace dumplings_remaining_l172_17296

theorem dumplings_remaining (cooked : ℕ) (eaten : ℕ) (h1 : cooked = 14) (h2 : eaten = 7) :
  cooked - eaten = 7 := by
  sorry

end dumplings_remaining_l172_17296


namespace quadratic_two_distinct_roots_l172_17283

theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + a = 0 ∧ x₂^2 - 2*x₂ + a = 0) ↔ a < 1 := by
  sorry

end quadratic_two_distinct_roots_l172_17283


namespace student_number_problem_l172_17225

theorem student_number_problem (x : ℝ) : 6 * x - 138 = 102 → x = 40 := by
  sorry

end student_number_problem_l172_17225


namespace solution_value_l172_17254

theorem solution_value (a : ℝ) : (∃ x : ℝ, x = -2 ∧ a * x - 6 = a + 3) → a = -3 := by
  sorry

end solution_value_l172_17254


namespace expected_carrot_yield_l172_17258

def garden_length_steps : ℕ := 25
def garden_width_steps : ℕ := 35
def step_length_feet : ℕ := 3
def yield_per_sqft : ℚ := 3/4

theorem expected_carrot_yield :
  let garden_length_feet : ℕ := garden_length_steps * step_length_feet
  let garden_width_feet : ℕ := garden_width_steps * step_length_feet
  let garden_area_sqft : ℕ := garden_length_feet * garden_width_feet
  garden_area_sqft * yield_per_sqft = 5906.25 := by
  sorry

end expected_carrot_yield_l172_17258


namespace avg_speed_BC_l172_17212

/-- Represents the journey of a motorcyclist --/
structure Journey where
  distanceAB : ℝ
  distanceBC : ℝ
  timeAB : ℝ
  timeBC : ℝ
  avgSpeedTotal : ℝ

/-- Theorem stating the average speed from B to C given the journey conditions --/
theorem avg_speed_BC (j : Journey)
  (h1 : j.distanceAB = 120)
  (h2 : j.distanceBC = j.distanceAB / 2)
  (h3 : j.timeAB = 3 * j.timeBC)
  (h4 : j.avgSpeedTotal = 20)
  (h5 : j.avgSpeedTotal = (j.distanceAB + j.distanceBC) / (j.timeAB + j.timeBC)) :
  j.distanceBC / j.timeBC = 80 / 3 := by
  sorry

end avg_speed_BC_l172_17212


namespace sum_of_digits_divisible_by_11_l172_17211

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among any 39 consecutive natural numbers, there is always one whose sum of digits is divisible by 11 -/
theorem sum_of_digits_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k ≤ 38 ∧ (sum_of_digits (N + k)) % 11 = 0 := by sorry

end sum_of_digits_divisible_by_11_l172_17211


namespace seven_fifth_sum_minus_two_fifth_l172_17266

theorem seven_fifth_sum_minus_two_fifth (n : ℕ) : 
  (7^5 : ℕ) + (7^5 : ℕ) + (7^5 : ℕ) + (7^5 : ℕ) + (7^5 : ℕ) + (7^5 : ℕ) - (2^5 : ℕ) = 6 * (7^5 : ℕ) - 32 := by
sorry

end seven_fifth_sum_minus_two_fifth_l172_17266


namespace inverse_proportion_k_value_l172_17289

theorem inverse_proportion_k_value (k : ℝ) (y x : ℝ → ℝ) :
  (∀ x, y x = k / x) →  -- y is an inverse proportion function of x
  k < 0 →  -- k is negative
  (∀ x, 1 ≤ x → x ≤ 3 → y x ≤ y 1 ∧ y x ≥ y 3) →  -- y is decreasing on [1, 3]
  y 1 - y 3 = 4 →  -- difference between max and min values is 4
  k = -6 := by
sorry

end inverse_proportion_k_value_l172_17289


namespace semicircle_area_l172_17219

theorem semicircle_area (rectangle_width : Real) (rectangle_length : Real)
  (triangle_leg1 : Real) (triangle_leg2 : Real) :
  rectangle_width = 1 →
  rectangle_length = 3 →
  triangle_leg1 = 1 →
  triangle_leg2 = 2 →
  triangle_leg1^2 + triangle_leg2^2 = rectangle_length^2 →
  (π * rectangle_length^2) / 8 = 9 * π / 8 := by
  sorry

end semicircle_area_l172_17219


namespace infinite_pairs_exist_l172_17201

/-- C(n) is the number of distinct prime divisors of n -/
def C (n : ℕ) : ℕ := sorry

/-- There exist infinitely many pairs of natural numbers (a,b) satisfying the given conditions -/
theorem infinite_pairs_exist : ∀ k : ℕ, ∃ a b : ℕ, a ≠ b ∧ a > k ∧ b > k ∧ C (a + b) = C a + C b := by
  sorry

end infinite_pairs_exist_l172_17201


namespace matthias_basketballs_l172_17240

/-- Given information about Matthias' balls, prove the total number of basketballs --/
theorem matthias_basketballs 
  (total_soccer : ℕ)
  (soccer_with_holes : ℕ)
  (basketball_with_holes : ℕ)
  (total_without_holes : ℕ)
  (h1 : total_soccer = 40)
  (h2 : soccer_with_holes = 30)
  (h3 : basketball_with_holes = 7)
  (h4 : total_without_holes = 18)
  (h5 : total_without_holes = total_soccer - soccer_with_holes + (total_basketballs - basketball_with_holes)) :
  total_basketballs = 15 :=
by
  sorry


end matthias_basketballs_l172_17240


namespace circle_tangent_to_line_l172_17227

/-- A circle with equation x^2 + y^2 = 4m is tangent to a line with equation x - y = 2√m if and only if m = 0 -/
theorem circle_tangent_to_line (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = 4*m ∧ x - y = 2*Real.sqrt m) →
  (∀ (x y : ℝ), x^2 + y^2 = 4*m → (x - y ≠ 2*Real.sqrt m ∨ (x - y = 2*Real.sqrt m ∧ 
    ∀ ε > 0, ∃ x' y', (x' - x)^2 + (y' - y)^2 < ε^2 ∧ x'^2 + y'^2 > 4*m))) →
  m = 0 := by
sorry

end circle_tangent_to_line_l172_17227


namespace arithmetic_sequence_properties_l172_17222

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Define the equation
def equation (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the theorem
theorem arithmetic_sequence_properties
  (a : ℝ) (d : ℝ)
  (h1 : equation a 1 = 0)
  (h2 : equation a d = 0) :
  ∃ (S_n : ℕ → ℝ) (T_n : ℕ → ℝ),
    (∀ n : ℕ, arithmetic_sequence a d n = n + 1) ∧
    (∀ n : ℕ, S_n n = (n^2 + 3*n) / 2) ∧
    (∀ n : ℕ, T_n n = 1 + (n - 1) * 3^n + (3^n - 1) / 2) :=
by sorry


end arithmetic_sequence_properties_l172_17222


namespace man_son_age_difference_l172_17251

/-- Represents the age difference between a man and his son -/
def AgeDifference (sonAge manAge : ℕ) : ℕ := manAge - sonAge

theorem man_son_age_difference :
  ∀ (sonAge manAge : ℕ),
  sonAge = 22 →
  manAge + 2 = 2 * (sonAge + 2) →
  AgeDifference sonAge manAge = 24 := by
  sorry

end man_son_age_difference_l172_17251


namespace distance_to_charlie_l172_17286

/-- The vertical distance Annie and Barbara walk together to reach Charlie -/
theorem distance_to_charlie 
  (annie_x annie_y barbara_x barbara_y charlie_x charlie_y : ℚ) : 
  annie_x = 6 → 
  annie_y = -20 → 
  barbara_x = 1 → 
  barbara_y = 14 → 
  charlie_x = 7/2 → 
  charlie_y = 2 → 
  charlie_y - (annie_y + barbara_y) / 2 = 5 := by
sorry

end distance_to_charlie_l172_17286


namespace bouncy_balls_per_package_l172_17237

/-- The number of bouncy balls in each package -/
def balls_per_package : ℝ := 10

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℝ := 8.0

/-- The number of packs of green bouncy balls Maggie gave away -/
def green_packs_given : ℝ := 4.0

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs_bought : ℝ := 4.0

/-- The total number of bouncy balls Maggie kept -/
def total_balls : ℕ := 80

theorem bouncy_balls_per_package :
  yellow_packs * balls_per_package = total_balls := by sorry

end bouncy_balls_per_package_l172_17237


namespace toy_car_gift_difference_l172_17218

/-- Proves the difference between Mum's and Dad's toy car gifts --/
theorem toy_car_gift_difference :
  ∀ (initial final dad uncle auntie grandpa : ℕ),
  initial = 150 →
  final = 196 →
  dad = 10 →
  auntie = uncle + 1 →
  auntie = 6 →
  grandpa = 2 * uncle →
  ∃ (mum : ℕ),
    final = initial + dad + uncle + auntie + grandpa + mum ∧
    mum - dad = 5 := by
  sorry

end toy_car_gift_difference_l172_17218


namespace geese_percentage_among_non_swans_l172_17293

theorem geese_percentage_among_non_swans :
  ∀ (total : ℝ) (geese swan heron duck : ℝ),
    geese / total = 0.35 →
    swan / total = 0.20 →
    heron / total = 0.15 →
    duck / total = 0.30 →
    total > 0 →
    geese / (total - swan) = 0.4375 :=
by
  sorry

end geese_percentage_among_non_swans_l172_17293


namespace parallel_iff_a_eq_two_l172_17247

/-- Two lines in the plane -/
structure TwoLines where
  line1 : ℝ → ℝ → ℝ  -- represents ax + 2y = 0
  line2 : ℝ → ℝ → ℝ  -- represents x + y = 1

/-- The condition for parallelism -/
def parallel (l : TwoLines) (a : ℝ) : Prop :=
  ∀ x y, l.line1 x y = 0 ∧ l.line2 x y = 1 → 
    ∃ k : ℝ, k ≠ 0 ∧ (a = k ∧ 2 = k)

/-- The theorem stating that a=2 is necessary and sufficient for parallelism -/
theorem parallel_iff_a_eq_two (l : TwoLines) : 
  (∀ a, parallel l a ↔ a = 2) := by sorry

end parallel_iff_a_eq_two_l172_17247


namespace book_pages_count_l172_17294

theorem book_pages_count : 
  let total_chapters : ℕ := 31
  let first_ten_pages : ℕ := 61
  let middle_ten_pages : ℕ := 59
  let last_eleven_pages : List ℕ := [58, 65, 62, 63, 64, 57, 66, 60, 59, 67]
  
  (10 * first_ten_pages) + 
  (10 * middle_ten_pages) + 
  (last_eleven_pages.sum) = 1821 := by
  sorry

end book_pages_count_l172_17294


namespace work_completion_time_l172_17214

theorem work_completion_time (a b : ℝ) (h1 : a = 2 * b) (h2 : 1 / a + 1 / b = 3 / 10) : 1 / b = 1 / 10 := by
  sorry

end work_completion_time_l172_17214


namespace parabola_point_coordinates_l172_17208

/-- A point on a parabola with a specific distance to its focus -/
def PointOnParabola (x y : ℝ) : Prop :=
  x^2 = 4*y ∧ (x - 0)^2 + (y - 1/4)^2 = 10^2

/-- The coordinates of the point satisfy the given conditions -/
theorem parabola_point_coordinates :
  ∀ x y : ℝ, PointOnParabola x y → (x = 6 ∨ x = -6) ∧ y = 9 :=
by sorry

end parabola_point_coordinates_l172_17208


namespace part_one_part_two_l172_17226

-- Define sets A and B
def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Part 1: Prove that (C_R B) ∩ A = {x | 3 ≤ x ≤ 5} when m = 3
theorem part_one : (Set.compl (B 3) ∩ A) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

-- Part 2: Prove that if A ∩ B = {x | -1 < x < 4}, then m = 8
theorem part_two : (∃ m : ℝ, A ∩ B m = {x | -1 < x ∧ x < 4}) → (∃ m : ℝ, m = 8) := by sorry

end part_one_part_two_l172_17226


namespace complement_union_theorem_l172_17256

def U : Set ℤ := {-1, 1, 2, 3, 4}
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {2, 4}

theorem complement_union_theorem :
  (Set.compl A ∩ U) ∪ B = {-1, 2, 4} := by sorry

end complement_union_theorem_l172_17256


namespace collinear_points_k_value_l172_17230

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The theorem states that if the points (1,-2), (3,4), and (6,k/3) are collinear, then k = 39. -/
theorem collinear_points_k_value :
  ∀ k : ℝ, collinear 1 (-2) 3 4 6 (k/3) → k = 39 := by
  sorry

#check collinear_points_k_value

end collinear_points_k_value_l172_17230


namespace min_value_expression_l172_17235

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 5*x + 2) * (y^2 + 5*y + 2) * (z^2 + 5*z + 2) / (x*y*z) ≥ 512 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a*b*c) = 512 :=
by sorry

end min_value_expression_l172_17235


namespace prob_even_sum_is_one_third_l172_17245

/-- Probability of an even outcome for the first wheel -/
def p_even_1 : ℚ := 1/2

/-- Probability of an even outcome for the second wheel -/
def p_even_2 : ℚ := 1/3

/-- Probability of an even outcome for the third wheel -/
def p_even_3 : ℚ := 3/4

/-- The probability of getting an even sum from three independent events -/
def prob_even_sum (p1 p2 p3 : ℚ) : ℚ :=
  p1 * p2 * p3 +
  (1 - p1) * p2 * p3 +
  p1 * (1 - p2) * p3 +
  p1 * p2 * (1 - p3)

/-- Theorem stating that the probability of an even sum is 1/3 given the specific probabilities -/
theorem prob_even_sum_is_one_third :
  prob_even_sum p_even_1 p_even_2 p_even_3 = 1/3 := by
  sorry

end prob_even_sum_is_one_third_l172_17245


namespace median_and_midpoint_lengths_l172_17224

/-- A right triangle with specific side lengths and a median -/
structure RightTriangleWithMedian where
  -- The length of side XY
  xy : ℝ
  -- The length of side YZ
  yz : ℝ
  -- The point W on side YZ
  w : ℝ
  -- Condition: XY = 6
  xy_eq : xy = 6
  -- Condition: YZ = 8
  yz_eq : yz = 8
  -- Condition: W is the midpoint of YZ
  w_midpoint : w = yz / 2

/-- The length of XW is 5 and the length of WZ is 4 in the given right triangle -/
theorem median_and_midpoint_lengths (t : RightTriangleWithMedian) : 
  Real.sqrt (t.xy^2 + (t.yz - t.w)^2) = 5 ∧ t.w = 4 := by
  sorry


end median_and_midpoint_lengths_l172_17224


namespace perpendicular_line_equation_l172_17265

/-- A line passing through a point and perpendicular to another line --/
structure PerpendicularLine where
  -- The point that the line passes through
  point : ℝ × ℝ
  -- The line that our line is perpendicular to, represented by its coefficients (a, b, c) in ax + by + c = 0
  perp_line : ℝ × ℝ × ℝ

/-- The equation of a line, represented by its coefficients (a, b, c) in ax + by + c = 0 --/
def LineEquation := ℝ × ℝ × ℝ

/-- Check if a point lies on a line given by its equation --/
def point_on_line (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  let (x, y) := p
  let (a, b, c) := l
  a * x + b * y + c = 0

/-- Check if two lines are perpendicular --/
def perpendicular (l1 l2 : LineEquation) : Prop :=
  let (a1, b1, _) := l1
  let (a2, b2, _) := l2
  a1 * a2 + b1 * b2 = 0

/-- The main theorem --/
theorem perpendicular_line_equation (l : PerpendicularLine) :
  let given_line : LineEquation := (1, -2, -3)
  let result_line : LineEquation := (2, 1, -1)
  l.point = (-1, 3) ∧ perpendicular given_line (result_line) →
  point_on_line l.point result_line ∧ perpendicular given_line result_line :=
by sorry

end perpendicular_line_equation_l172_17265


namespace great_great_grandmother_age_calculation_l172_17246

-- Define the ages of family members
def darcie_age : ℚ := 4
def mother_age : ℚ := darcie_age * 6
def grandmother_age : ℚ := mother_age * (5/4)
def great_grandfather_age : ℚ := grandmother_age * (4/3)
def great_great_grandmother_age : ℚ := great_grandfather_age * (10/7)

-- Theorem statement
theorem great_great_grandmother_age_calculation :
  great_great_grandmother_age = 400/7 := by
  sorry

end great_great_grandmother_age_calculation_l172_17246


namespace periodic_odd_function_at_one_l172_17272

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_at_one (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_odd : is_odd f) : 
  f 1 = 0 := by
sorry

end periodic_odd_function_at_one_l172_17272


namespace smallest_positive_integer_with_remainders_l172_17217

theorem smallest_positive_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧
  (x % 5 = 4) ∧ 
  (x % 7 = 6) ∧ 
  (x % 9 = 8) ∧
  (∀ y : ℕ, y > 0 → y % 5 = 4 → y % 7 = 6 → y % 9 = 8 → x ≤ y) ∧
  (x = 314) :=
by sorry

end smallest_positive_integer_with_remainders_l172_17217


namespace function_value_comparison_l172_17216

def f (x : ℝ) : ℝ := 3 * (x - 2)^2 + 5

theorem function_value_comparison (x₁ x₂ : ℝ) 
  (h : |x₁ - 2| > |x₂ - 2|) : f x₁ > f x₂ := by
  sorry

end function_value_comparison_l172_17216


namespace quilt_shaded_fraction_l172_17298

/-- Represents a square quilt block -/
structure QuiltBlock where
  total_squares : Nat
  divided_squares : Nat
  shaded_triangles : Nat

/-- The fraction of a square covered by a shaded triangle -/
def triangle_coverage : Rat := 1/2

/-- Calculates the fraction of the quilt block that is shaded -/
def shaded_fraction (quilt : QuiltBlock) : Rat :=
  (quilt.shaded_triangles : Rat) * triangle_coverage / (quilt.total_squares : Rat)

/-- Theorem stating that the shaded fraction of the quilt block is 1/8 -/
theorem quilt_shaded_fraction :
  ∀ (quilt : QuiltBlock),
    quilt.total_squares = 16 ∧
    quilt.divided_squares = 4 ∧
    quilt.shaded_triangles = 4 →
    shaded_fraction quilt = 1/8 := by
  sorry

end quilt_shaded_fraction_l172_17298


namespace constant_value_proof_l172_17275

theorem constant_value_proof (f : ℝ → ℝ) (c : ℝ) 
  (h1 : ∀ x, f x + 3 * f (c - x) = x) 
  (h2 : f 2 = 2) : 
  c = 8 := by
  sorry

end constant_value_proof_l172_17275
