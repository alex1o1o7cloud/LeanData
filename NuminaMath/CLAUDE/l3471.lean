import Mathlib

namespace total_adding_schemes_l3471_347142

/-- Represents the number of available raw materials -/
def total_materials : ℕ := 5

/-- Represents the number of materials to be added sequentially -/
def materials_to_add : ℕ := 2

/-- Represents the number of ways to add material A first -/
def ways_with_A_first : ℕ := 3

/-- Represents the number of ways to add material B first -/
def ways_with_B_first : ℕ := 6

/-- Represents the number of ways to add materials without A or B -/
def ways_without_A_or_B : ℕ := 6

/-- Theorem stating the total number of different adding schemes -/
theorem total_adding_schemes :
  ways_with_A_first + ways_with_B_first + ways_without_A_or_B = 15 :=
by sorry

end total_adding_schemes_l3471_347142


namespace extremum_of_f_and_range_of_a_l3471_347129

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1 - a * Real.log x

noncomputable def g (x : ℝ) : ℝ := Real.exp x / x

theorem extremum_of_f_and_range_of_a :
  (∃ (x : ℝ), x > 0 ∧ f (1 / Real.exp 1) x = 1 / Real.exp 1 ∧
    ∀ (y : ℝ), y > 0 → f (1 / Real.exp 1) y ≥ f (1 / Real.exp 1) x) ∧
  (∀ (a : ℝ), a < 0 →
    (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 4 5 → x₂ ∈ Set.Icc 4 5 → x₁ ≠ x₂ →
      |f a x₁ - f a x₂| < |g x₁ - g x₂|) →
    4 - 3 / 4 * Real.exp 4 ≤ a) := by
  sorry

end extremum_of_f_and_range_of_a_l3471_347129


namespace no_solution_a_no_solution_b_no_solution_c_no_solution_d_no_solution_e_no_solution_f_no_solution_g_no_solution_h_l3471_347126

-- a) √(x+2) = -2
theorem no_solution_a : ¬∃ x : ℝ, Real.sqrt (x + 2) = -2 := by sorry

-- b) √(2x+3) + √(x+3) = 0
theorem no_solution_b : ¬∃ x : ℝ, Real.sqrt (2*x + 3) + Real.sqrt (x + 3) = 0 := by sorry

-- c) √(4-x) - √(x-6) = 2
theorem no_solution_c : ¬∃ x : ℝ, Real.sqrt (4 - x) - Real.sqrt (x - 6) = 2 := by sorry

-- d) √(-1-x) = ∛(x-5)
theorem no_solution_d : ¬∃ x : ℝ, Real.sqrt (-1 - x) = (x - 5) ^ (1/3 : ℝ) := by sorry

-- e) 5√x - 3√(-x) + 17/x = 4
theorem no_solution_e : ¬∃ x : ℝ, 5 * Real.sqrt x - 3 * Real.sqrt (-x) + 17 / x = 4 := by sorry

-- f) √(x-3) - √(x+9) = √(x-2)
theorem no_solution_f : ¬∃ x : ℝ, Real.sqrt (x - 3) - Real.sqrt (x + 9) = Real.sqrt (x - 2) := by sorry

-- g) √x + √(x+9) = 2
theorem no_solution_g : ¬∃ x : ℝ, Real.sqrt x + Real.sqrt (x + 9) = 2 := by sorry

-- h) ∛(x + 1/x) = √(-x) - 1
theorem no_solution_h : ¬∃ x : ℝ, (x + 1/x) ^ (1/3 : ℝ) = Real.sqrt (-x) - 1 := by sorry

end no_solution_a_no_solution_b_no_solution_c_no_solution_d_no_solution_e_no_solution_f_no_solution_g_no_solution_h_l3471_347126


namespace problem_statement_l3471_347193

theorem problem_statement (a b : ℝ) : 
  (a^2 + 4*a + 6) * (2*b^2 - 4*b + 7) ≤ 10 → a + 2*b = 0 :=
by sorry

end problem_statement_l3471_347193


namespace quadratic_coefficient_l3471_347125

/-- A quadratic function with vertex (3, 5) passing through (-2, -20) has a = -1 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Condition 1
  (3, 5) = (- b / (2 * a), a * (- b / (2 * a))^2 + b * (- b / (2 * a)) + c) →  -- Condition 2
  a * (-2)^2 + b * (-2) + c = -20 →  -- Condition 3
  a = -1 := by
sorry

end quadratic_coefficient_l3471_347125


namespace custom_operation_equality_l3471_347141

/-- Custom binary operation ⊗ -/
def otimes (a b : ℚ) : ℚ := a^2 / b

/-- Theorem statement -/
theorem custom_operation_equality : 
  (otimes (otimes 3 4) 6) - (otimes 3 (otimes 4 6)) - 1 = -113/32 := by
  sorry

end custom_operation_equality_l3471_347141


namespace cyclic_quadrilateral_theorem_l3471_347110

-- Define the basic structures
structure Point := (x y : ℝ)

structure Line := (a b c : ℝ)

-- Define the quadrilateral ABCD
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry
def D : Point := sorry

-- Define points E and F on CD
def E : Point := sorry
def F : Point := sorry

-- Define the circumcenters G and H
def G : Point := sorry
def H : Point := sorry

-- Define the lines AB, CD, and GH
def AB : Line := sorry
def CD : Line := sorry
def GH : Line := sorry

-- Define the property of being cyclic
def is_cyclic (p q r s : Point) : Prop := sorry

-- Define the property of lines being concurrent or parallel
def lines_concurrent_or_parallel (l m n : Line) : Prop := sorry

-- Define the property of a point lying on a line
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Define the property of being a circumcenter
def is_circumcenter (p : Point) (a b c : Point) : Prop := sorry

-- Main theorem
theorem cyclic_quadrilateral_theorem :
  (∀ (X : Point), is_cyclic A B C D) →  -- ABCD is cyclic
  (¬ (AB.a * CD.b = AB.b * CD.a)) →  -- AD is not parallel to BC
  point_on_line E CD →  -- E lies on CD
  point_on_line F CD →  -- F lies on CD
  is_circumcenter G B C E →  -- G is circumcenter of BCE
  is_circumcenter H A D F →  -- H is circumcenter of ADF
  (lines_concurrent_or_parallel AB CD GH ↔ is_cyclic A B E F) := by
  sorry

end cyclic_quadrilateral_theorem_l3471_347110


namespace largest_four_digit_divisible_by_24_l3471_347184

theorem largest_four_digit_divisible_by_24 : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 24 = 0 → n ≤ 9984 :=
by sorry

end largest_four_digit_divisible_by_24_l3471_347184


namespace daily_production_is_2170_l3471_347160

/-- The number of toys produced per week -/
def weekly_production : ℕ := 4340

/-- The number of working days per week -/
def working_days : ℕ := 2

/-- The number of toys produced each day -/
def daily_production : ℕ := weekly_production / working_days

/-- Theorem stating that the daily production is 2170 toys -/
theorem daily_production_is_2170 : daily_production = 2170 := by
  sorry

end daily_production_is_2170_l3471_347160


namespace cubic_root_sum_l3471_347137

theorem cubic_root_sum (a b m n p : ℝ) 
  (hm : m^3 + a*m + b = 0)
  (hn : n^3 + a*n + b = 0)
  (hp : p^3 + a*p + b = 0)
  (hmn : m ≠ n)
  (hnp : n ≠ p)
  (hmp : m ≠ p) :
  m + n + p = 0 := by
  sorry

end cubic_root_sum_l3471_347137


namespace complex_magnitude_l3471_347151

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 1) : Complex.abs z = 1 := by
  sorry

end complex_magnitude_l3471_347151


namespace rationalize_denominator_sqrt5_l3471_347195

theorem rationalize_denominator_sqrt5 :
  ∃ (A B C : ℤ),
    (A = -9 ∧ B = -4 ∧ C = 5) ∧
    (A * B * C = 180) ∧
    ∃ (x : ℝ),
      x = (2 + Real.sqrt 5) / (2 - Real.sqrt 5) ∧
      x = A + B * Real.sqrt C := by
  sorry

end rationalize_denominator_sqrt5_l3471_347195


namespace isosceles_top_angle_l3471_347177

-- Define an isosceles triangle
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

-- Define the sum of angles in a triangle
axiom angle_sum : ∀ (x y z : ℝ), x + y + z = 180

-- Theorem statement
theorem isosceles_top_angle (a b c : ℝ) 
  (h1 : IsIsosceles a b c) (h2 : a = 40 ∨ b = 40 ∨ c = 40) : 
  a = 40 ∨ b = 40 ∨ c = 40 ∨ a = 100 ∨ b = 100 ∨ c = 100 := by
  sorry


end isosceles_top_angle_l3471_347177


namespace retirement_total_is_70_l3471_347176

/-- Represents the retirement eligibility rule for a company -/
structure RetirementRule :=
  (hire_year : ℕ)
  (hire_age : ℕ)
  (retirement_year : ℕ)

/-- Calculates the required total of age and years of employment for retirement -/
def retirement_total (rule : RetirementRule) : ℕ :=
  (rule.retirement_year - rule.hire_year) + rule.hire_age + (rule.retirement_year - rule.hire_year)

/-- Theorem stating the required total for retirement is 70 -/
theorem retirement_total_is_70 (rule : RetirementRule) 
  (h1 : rule.hire_year = 1986)
  (h2 : rule.hire_age = 30)
  (h3 : rule.retirement_year = 2006) :
  retirement_total rule = 70 := by
  sorry

end retirement_total_is_70_l3471_347176


namespace correct_factorization_l3471_347181

theorem correct_factorization (x : ℝ) : x^2 - 0.01 = (x + 0.1) * (x - 0.1) := by
  sorry

end correct_factorization_l3471_347181


namespace animal_feet_count_animal_feet_theorem_l3471_347180

theorem animal_feet_count (total_heads : Nat) (hen_count : Nat) : Nat :=
  let cow_count := total_heads - hen_count
  let hen_feet := hen_count * 2
  let cow_feet := cow_count * 4
  hen_feet + cow_feet

theorem animal_feet_theorem (total_heads : Nat) (hen_count : Nat) 
  (h1 : total_heads = 44) (h2 : hen_count = 18) : 
  animal_feet_count total_heads hen_count = 140 := by
  sorry

end animal_feet_count_animal_feet_theorem_l3471_347180


namespace solution_set_and_range_l3471_347101

def f (x : ℝ) : ℝ := |2*x + 1| + 2*|x - 3|

theorem solution_set_and_range :
  (∃ (S : Set ℝ), S = {x : ℝ | f x ≤ 7*x} ∧ S = {x : ℝ | x ≥ 1}) ∧
  (∃ (M : Set ℝ), M = {m : ℝ | ∃ x : ℝ, f x = |m|} ∧ M = {m : ℝ | m ≥ 7 ∨ m ≤ -7}) :=
by sorry

end solution_set_and_range_l3471_347101


namespace milk_tea_sales_distribution_l3471_347107

/-- Represents the sales distribution of milk tea flavors -/
structure MilkTeaSales where
  total : ℕ
  winterMelon : ℕ
  okinawa : ℕ
  chocolate : ℕ
  thai : ℕ
  taro : ℕ

/-- Conditions for the milk tea sales problem -/
def salesConditions (s : MilkTeaSales) : Prop :=
  s.total = 100 ∧
  s.winterMelon = (35 * s.total) / 100 ∧
  s.okinawa = s.total / 4 ∧
  s.taro = 12 ∧
  3 * s.chocolate = 7 * s.thai ∧
  s.chocolate + s.thai = s.total - s.winterMelon - s.okinawa - s.taro

/-- Theorem stating the correct distribution of milk tea sales -/
theorem milk_tea_sales_distribution :
  ∃ (s : MilkTeaSales),
    salesConditions s ∧
    s.winterMelon = 35 ∧
    s.okinawa = 25 ∧
    s.chocolate = 8 ∧
    s.thai = 20 ∧
    s.taro = 12 ∧
    s.winterMelon + s.okinawa + s.chocolate + s.thai + s.taro = s.total :=
by
  sorry

end milk_tea_sales_distribution_l3471_347107


namespace smallest_number_is_negative_sqrt_5_l3471_347136

theorem smallest_number_is_negative_sqrt_5 :
  let a := (-5 : ℝ)^0
  let b := -Real.sqrt 5
  let c := -(1 / 5 : ℝ)
  let d := |(-5 : ℝ)|
  b < a ∧ b < c ∧ b < d := by sorry

end smallest_number_is_negative_sqrt_5_l3471_347136


namespace gcd_consecutive_terms_unbounded_l3471_347175

def a (n : ℕ) : ℤ := n.factorial - n

theorem gcd_consecutive_terms_unbounded :
  ∀ M : ℕ, ∃ n : ℕ, Int.gcd (a n) (a (n + 1)) > M :=
sorry

end gcd_consecutive_terms_unbounded_l3471_347175


namespace x_cube_plus_reciprocal_l3471_347114

theorem x_cube_plus_reciprocal (x : ℝ) (h : 11 = x^6 + 1/x^6) : x^3 + 1/x^3 = Real.sqrt 13 := by
  sorry

end x_cube_plus_reciprocal_l3471_347114


namespace function_properties_l3471_347186

noncomputable def f (a b x : ℝ) : ℝ := (a * x) / (Real.exp x + 1) + b * Real.exp (-x)

theorem function_properties (a b k : ℝ) :
  (f a b 0 = 1) →
  (HasDerivAt (f a b) (-1/2) 0) →
  (∀ x ≠ 0, f a b x > x / (Real.exp x - 1) + k * Real.exp (-x)) →
  (a = 1 ∧ b = 1 ∧ k ≤ 0) := by
  sorry

end function_properties_l3471_347186


namespace largest_square_tile_size_l3471_347134

/-- The length of the courtyard in centimeters -/
def courtyard_length : ℕ := 378

/-- The width of the courtyard in centimeters -/
def courtyard_width : ℕ := 525

/-- The size of the largest square tile in centimeters -/
def largest_tile_size : ℕ := 21

theorem largest_square_tile_size :
  (largest_tile_size ∣ courtyard_length) ∧
  (largest_tile_size ∣ courtyard_width) ∧
  ∀ n : ℕ, n > largest_tile_size →
    ¬(n ∣ courtyard_length) ∨ ¬(n ∣ courtyard_width) :=
by sorry

end largest_square_tile_size_l3471_347134


namespace correct_equation_only_E_is_true_l3471_347153

theorem correct_equation : 15618 = 1 + 5^6 - 1 * 8 := by
  sorry

-- The following definitions represent the conditions from the original problem
def equation_A : Prop := 15614 = 1 + 5^6 - 1 * 4
def equation_B : Prop := 15615 = 1 + 5^6 - 1 * 5
def equation_C : Prop := 15616 = 1 + 5^6 - 1 * 6
def equation_D : Prop := 15617 = 1 + 5^6 - 1 * 7
def equation_E : Prop := 15618 = 1 + 5^6 - 1 * 8

-- This theorem states that equation_E is the only true equation among the given options
theorem only_E_is_true : 
  ¬equation_A ∧ ¬equation_B ∧ ¬equation_C ∧ ¬equation_D ∧ equation_E := by
  sorry

end correct_equation_only_E_is_true_l3471_347153


namespace sequence_properties_l3471_347113

def sequence_sum (n : ℕ) : ℝ := n^2

def sequence_term (n : ℕ+) : ℝ := 2 * n.val - 1

def is_geometric_triple (a b c : ℝ) : Prop :=
  a * c = b^2

theorem sequence_properties :
  (∀ n : ℕ+, n > 1 →
    1 / Real.sqrt (sequence_sum (n.val - 1)) -
    1 / Real.sqrt (sequence_sum n.val) -
    1 / Real.sqrt (sequence_sum n.val * sequence_sum (n.val - 1)) = 0) →
  sequence_term 1 = 1 →
  (∀ n : ℕ+, sequence_term n = 2 * n.val - 1) ∧
  (∀ m t : ℕ+, 1 < m → m < t → t ≤ 100 →
    is_geometric_triple (1 / sequence_term 2) (1 / sequence_term m) (1 / sequence_term t) ↔
    (m = 5 ∧ t = 14) ∨ (m = 8 ∧ t = 38) ∨ (m = 11 ∧ t = 74)) :=
by sorry

end sequence_properties_l3471_347113


namespace consecutive_integers_product_sum_l3471_347172

theorem consecutive_integers_product_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) * (n + 2) * (n + 3) = 358800 → 
  n + (n + 1) + (n + 2) + (n + 3) = 98 := by
sorry

end consecutive_integers_product_sum_l3471_347172


namespace cubic_equation_integer_solutions_l3471_347197

theorem cubic_equation_integer_solutions :
  (∀ k : ℤ, ∃! n : ℕ, ∀ x : ℤ, x^3 - 24*x + k = 0 → (x.natAbs ≤ n)) ∧
  (∃! x : ℤ, x^3 + 24*x - 2016 = 0) ∧
  (12^3 + 24*12 - 2016 = 0) :=
by sorry

end cubic_equation_integer_solutions_l3471_347197


namespace jeff_scores_mean_l3471_347164

theorem jeff_scores_mean : 
  let scores : List ℕ := [89, 92, 88, 95, 91, 93]
  (scores.sum : ℚ) / scores.length = 548 / 6 := by sorry

end jeff_scores_mean_l3471_347164


namespace most_likely_red_balls_l3471_347168

theorem most_likely_red_balls
  (total_balls : ℕ)
  (red_probability : ℚ)
  (h_total : total_balls = 20)
  (h_prob : red_probability = 1/5) :
  (red_probability * total_balls : ℚ) = 4 := by
sorry

end most_likely_red_balls_l3471_347168


namespace circle_radius_with_tangent_l3471_347106

/-- The radius of a circle with equation x^2 + y^2 = 25 and a tangent at y = 5 is 5 -/
theorem circle_radius_with_tangent (x y : ℝ) :
  x^2 + y^2 = 25 → ∃ (x₀ : ℝ), x₀^2 + 5^2 = 25 → 
  Real.sqrt ((0 - x₀)^2 + (5 - 0)^2) = 5 := by
sorry

end circle_radius_with_tangent_l3471_347106


namespace sally_peach_cost_l3471_347165

-- Define the given amounts
def total_spent : ℚ := 23.86
def cherry_cost : ℚ := 11.54

-- Define the amount spent on peaches after coupon
def peach_cost : ℚ := total_spent - cherry_cost

-- Theorem to prove
theorem sally_peach_cost : peach_cost = 12.32 := by
  sorry

end sally_peach_cost_l3471_347165


namespace brian_tape_problem_l3471_347190

/-- The amount of tape needed for a rectangular box -/
def tape_needed (length width : ℕ) : ℕ := length + 2 * width

/-- The total amount of tape needed for multiple boxes of the same size -/
def total_tape_for_boxes (length width count : ℕ) : ℕ :=
  count * tape_needed length width

/-- The problem statement -/
theorem brian_tape_problem :
  let tape_for_small_boxes := total_tape_for_boxes 30 15 5
  let tape_for_large_boxes := total_tape_for_boxes 40 40 2
  tape_for_small_boxes + tape_for_large_boxes = 540 := by
sorry


end brian_tape_problem_l3471_347190


namespace inverse_function_intersection_implies_root_l3471_347131

theorem inverse_function_intersection_implies_root (f : ℝ → ℝ) (f_inv : ℝ → ℝ) :
  (∀ x, f_inv (f x) = x) →  -- f_inv is the inverse of f
  (∀ x, -f_inv x = f_inv (-x)) →  -- given condition about -f^(-1)(x)
  f_inv 0 = 2 →  -- intersection point (0, 2)
  f 2 = 0 :=  -- conclusion: 2 is a root of f(x) = 0
by
  sorry


end inverse_function_intersection_implies_root_l3471_347131


namespace max_value_constrained_l3471_347149

theorem max_value_constrained (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (max : ℝ), max = 14 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 4 → x'^2 + 6*y' + 2 ≤ max :=
sorry

end max_value_constrained_l3471_347149


namespace num_buses_is_ten_l3471_347144

-- Define the given conditions
def total_people : ℕ := 342
def num_vans : ℕ := 9
def people_per_van : ℕ := 8
def people_per_bus : ℕ := 27

-- Define the function to calculate the number of buses
def calculate_buses : ℕ :=
  (total_people - num_vans * people_per_van) / people_per_bus

-- Theorem statement
theorem num_buses_is_ten : calculate_buses = 10 := by
  sorry

end num_buses_is_ten_l3471_347144


namespace no_solution_lcm_gcd_equation_l3471_347198

theorem no_solution_lcm_gcd_equation : 
  ∀ n : ℕ+, n.lcm 200 ≠ n.gcd 200 + 1000 := by
sorry

end no_solution_lcm_gcd_equation_l3471_347198


namespace probability_not_snowing_l3471_347130

theorem probability_not_snowing (p_snow : ℚ) (h : p_snow = 2 / 5) : 
  1 - p_snow = 3 / 5 := by
  sorry

end probability_not_snowing_l3471_347130


namespace mod_congruence_solution_l3471_347163

theorem mod_congruence_solution : ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 10 ∧ n ≡ -2154 [ZMOD 7] ∧ n = 2 := by
  sorry

end mod_congruence_solution_l3471_347163


namespace pizza_toppings_combinations_l3471_347100

theorem pizza_toppings_combinations (n : ℕ) (h : n = 8) : 
  n + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end pizza_toppings_combinations_l3471_347100


namespace prob_at_least_two_diff_fruits_l3471_347103

/-- Represents the types of fruit Joe can choose from -/
inductive Fruit
  | apple
  | orange
  | banana
  | grape

/-- The probability of choosing a specific fruit -/
def fruit_prob (f : Fruit) : ℝ :=
  match f with
  | Fruit.apple => 0.4
  | Fruit.orange => 0.3
  | Fruit.banana => 0.2
  | Fruit.grape => 0.1

/-- The probability of choosing the same fruit for all three meals -/
def same_fruit_prob : ℝ :=
  (fruit_prob Fruit.apple) ^ 3 +
  (fruit_prob Fruit.orange) ^ 3 +
  (fruit_prob Fruit.banana) ^ 3 +
  (fruit_prob Fruit.grape) ^ 3

/-- Theorem: The probability of eating at least two different kinds of fruit in a day is 0.9 -/
theorem prob_at_least_two_diff_fruits :
  1 - same_fruit_prob = 0.9 := by
  sorry

end prob_at_least_two_diff_fruits_l3471_347103


namespace problem_solution_l3471_347179

theorem problem_solution (a m n : ℚ) : 
  (∀ x, (a * x - 3) * (2 * x + 1) - 4 * x^2 + m = (a - 6) * x) → 
  a * n + m * n = 1 → 
  2 * n^3 - 9 * n^2 + 8 * n = 157 / 125 := by
sorry

end problem_solution_l3471_347179


namespace cylinder_height_relationship_l3471_347102

theorem cylinder_height_relationship (r1 h1 r2 h2 : ℝ) :
  r1 > 0 ∧ h1 > 0 ∧ r2 > 0 ∧ h2 > 0 →
  r2 = 1.2 * r1 →
  π * r1^2 * h1 = π * r2^2 * h2 →
  h1 = 1.44 * h2 :=
by sorry

end cylinder_height_relationship_l3471_347102


namespace candy_cost_l3471_347159

theorem candy_cost (total_cents : ℕ) (num_gumdrops : ℕ) (h1 : total_cents = 224) (h2 : num_gumdrops = 28) :
  total_cents / num_gumdrops = 8 := by
  sorry

end candy_cost_l3471_347159


namespace chess_players_never_lost_to_ai_l3471_347155

theorem chess_players_never_lost_to_ai (total_players : ℕ) (players_lost : ℕ) :
  total_players = 40 →
  players_lost = 30 →
  (total_players - players_lost : ℚ) / total_players = 1/4 := by
sorry

end chess_players_never_lost_to_ai_l3471_347155


namespace total_cost_is_43_l3471_347143

-- Define the prices
def sandwich_price : ℚ := 4
def soda_price : ℚ := 3

-- Define the discount threshold and rate
def discount_threshold : ℚ := 50
def discount_rate : ℚ := 0.1

-- Define the quantities
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 5

-- Calculate the total cost before discount
def total_cost : ℚ := sandwich_price * num_sandwiches + soda_price * num_sodas

-- Function to apply discount if applicable
def apply_discount (cost : ℚ) : ℚ :=
  if cost > discount_threshold then cost * (1 - discount_rate) else cost

-- Theorem to prove
theorem total_cost_is_43 : apply_discount total_cost = 43 := by
  sorry

end total_cost_is_43_l3471_347143


namespace sqrt_27_simplification_l3471_347178

theorem sqrt_27_simplification : Real.sqrt 27 = 3 * Real.sqrt 3 := by
  sorry

end sqrt_27_simplification_l3471_347178


namespace imaginary_part_of_z_l3471_347187

theorem imaginary_part_of_z (z : ℂ) (h : z / (1 + Complex.I) = -1 / (2 * Complex.I)) :
  z.im = 1 / 2 := by
  sorry

end imaginary_part_of_z_l3471_347187


namespace x_plus_2y_equals_10_l3471_347157

theorem x_plus_2y_equals_10 (x y : ℝ) (hx : x = 4) (hy : y = 3) : x + 2*y = 10 := by
  sorry

end x_plus_2y_equals_10_l3471_347157


namespace park_birds_difference_l3471_347189

/-- The number of geese and ducks remaining at a park after some changes. -/
theorem park_birds_difference (initial_ducks : ℕ) (geese_leave : ℕ) : 
  let initial_geese := 2 * initial_ducks - 10
  let final_ducks := initial_ducks + 4
  let final_geese := initial_geese - (15 - 5)
  final_geese - final_ducks = 1 :=
by sorry

end park_birds_difference_l3471_347189


namespace min_value_of_f_l3471_347120

def f (x : ℕ) : ℤ := 3 * x^2 - 12 * x + 800

theorem min_value_of_f :
  ∀ x : ℕ, f x ≥ 788 ∧ ∃ x₀ : ℕ, f x₀ = 788 :=
by sorry

end min_value_of_f_l3471_347120


namespace tree_subgraph_existence_l3471_347119

-- Define a tree
def is_tree (T : SimpleGraph V) : Prop := sorry

-- Define the order of a graph
def graph_order (G : SimpleGraph V) : ℕ := sorry

-- Define the minimum degree of a graph
def min_degree (G : SimpleGraph V) : ℕ := sorry

-- Define graph isomorphism
def is_isomorphic_subgraph (T G : SimpleGraph V) : Prop := sorry

theorem tree_subgraph_existence 
  {V : Type*} (T G : SimpleGraph V) :
  is_tree T →
  min_degree G ≥ graph_order T - 1 →
  is_isomorphic_subgraph T G :=
by sorry

end tree_subgraph_existence_l3471_347119


namespace tv_show_payment_l3471_347154

theorem tv_show_payment (main_characters minor_characters : ℕ)
  (major_pay_ratio : ℕ) (total_payment : ℕ) :
  main_characters = 5 →
  minor_characters = 4 →
  major_pay_ratio = 3 →
  total_payment = 285000 →
  ∃ (minor_pay : ℕ),
    minor_pay = 15000 ∧
    minor_pay * (minor_characters + main_characters * major_pay_ratio) = total_payment :=
by sorry

end tv_show_payment_l3471_347154


namespace expression_simplification_l3471_347109

theorem expression_simplification (a b : ℝ) :
  (3 * a^5 * b^3 + a^4 * b^2) / ((-a^2 * b)^2) - (2 + a) * (2 - a) - a * (a - 5 * b) = 8 * a * b - 3 := by
  sorry

end expression_simplification_l3471_347109


namespace expand_triple_product_l3471_347122

theorem expand_triple_product (x y z : ℝ) :
  (x + 8) * (3 * y + 12) * (2 * z + 4) =
  6 * x * y * z + 12 * x * z + 24 * y * z + 12 * x * y + 48 * x + 96 * y + 96 * z + 384 := by
  sorry

end expand_triple_product_l3471_347122


namespace negative_four_squared_equals_sixteen_l3471_347139

theorem negative_four_squared_equals_sixteen :
  (-4 : ℤ) ^ 2 = 16 := by
  sorry

end negative_four_squared_equals_sixteen_l3471_347139


namespace symmetric_sequence_sum_is_n_squared_l3471_347166

/-- The sum of the symmetric sequence 1+2+3+...+(n-1)+n+(n+1)+n+...+3+2+1 -/
def symmetricSequenceSum (n : ℕ) : ℕ :=
  (List.range n).sum + n + (n + 1) + (List.range n).sum

/-- Theorem: The sum of the symmetric sequence is equal to n^2 -/
theorem symmetric_sequence_sum_is_n_squared (n : ℕ) :
  symmetricSequenceSum n = n^2 := by
  sorry

end symmetric_sequence_sum_is_n_squared_l3471_347166


namespace train_stop_time_l3471_347138

/-- Calculates the time a train stops per hour given its speeds with and without stoppages -/
theorem train_stop_time (speed_without_stoppages speed_with_stoppages : ℝ) : 
  speed_without_stoppages = 45 →
  speed_with_stoppages = 42 →
  (speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages * 60 = 4 := by
  sorry

#check train_stop_time

end train_stop_time_l3471_347138


namespace quadratic_function_properties_l3471_347115

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 6*a*x + 9

-- Define the theorem
theorem quadratic_function_properties (a : ℝ) :
  -- Part (1)
  (f a 2 = 7 → 
    a = 1/2 ∧ 
    ∃ (x y : ℝ), x = 3/2 ∧ y = 27/4 ∧ ∀ (t : ℝ), f a t ≥ f a x) ∧
  -- Part (2)
  (f a 2 = 7 → 
    ∀ (x : ℝ), -1 ≤ x ∧ x < 3 → 27/4 ≤ f a x ∧ f a x ≤ 13) ∧
  -- Part (3)
  (∀ (x : ℝ), x ≥ 3 → ∀ (y : ℝ), y > x → f a y > f a x) →
  (∀ (x₁ x₂ : ℝ), 3*a - 2 ≤ x₁ ∧ x₁ ≤ 5 ∧ 3*a - 2 ≤ x₂ ∧ x₂ ≤ 5 → 
    f a x₁ - f a x₂ ≤ 9*a^2 + 20) →
  1/6 ≤ a ∧ a ≤ 1 :=
by sorry


end quadratic_function_properties_l3471_347115


namespace merchant_profit_percentage_l3471_347123

theorem merchant_profit_percentage (C S : ℝ) (h : C > 0) :
  20 * C = 15 * S →
  (S - C) / C * 100 = 100/3 :=
by
  sorry

end merchant_profit_percentage_l3471_347123


namespace book_sale_problem_l3471_347152

/-- Proves that the total cost of two books is 600, given the specified conditions --/
theorem book_sale_problem (cost_loss : ℝ) (selling_price : ℝ) :
  cost_loss = 350 →
  selling_price = cost_loss * (1 - 0.15) →
  ∃ (cost_gain : ℝ), 
    selling_price = cost_gain * (1 + 0.19) ∧
    cost_loss + cost_gain = 600 := by
  sorry

end book_sale_problem_l3471_347152


namespace real_part_of_i_times_one_minus_i_l3471_347188

theorem real_part_of_i_times_one_minus_i (i : ℂ) : 
  i * i = -1 → Complex.re (i * (1 - i)) = 1 := by sorry

end real_part_of_i_times_one_minus_i_l3471_347188


namespace triangle_abc_is_right_l3471_347127

/-- Given three points in a 2D plane, determines if they form a right triangle --/
def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let ab_squared := (B.1 - A.1)^2 + (B.2 - A.2)^2
  let bc_squared := (C.1 - B.1)^2 + (C.2 - B.2)^2
  let ca_squared := (A.1 - C.1)^2 + (A.2 - C.2)^2
  (ab_squared = bc_squared + ca_squared) ∨
  (bc_squared = ab_squared + ca_squared) ∨
  (ca_squared = ab_squared + bc_squared)

/-- The triangle formed by points A(5, -2), B(1, 5), and C(-1, 2) is a right triangle --/
theorem triangle_abc_is_right :
  is_right_triangle (5, -2) (1, 5) (-1, 2) := by
  sorry

end triangle_abc_is_right_l3471_347127


namespace mathborough_rainfall_2004_l3471_347169

/-- The total rainfall in Mathborough for the year 2004 -/
def total_rainfall_2004 (avg_rainfall_2003 : ℕ) (rainfall_increase : ℕ) : ℕ := 
  let avg_rainfall_2004 := avg_rainfall_2003 + rainfall_increase
  let feb_rainfall := avg_rainfall_2004 * 29
  let other_months_rainfall := avg_rainfall_2004 * 30 * 11
  feb_rainfall + other_months_rainfall

/-- Theorem stating the total rainfall in Mathborough for 2004 -/
theorem mathborough_rainfall_2004 : 
  total_rainfall_2004 50 3 = 19027 := by
sorry

end mathborough_rainfall_2004_l3471_347169


namespace sum_of_squares_values_l3471_347140

theorem sum_of_squares_values (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
  sorry

end sum_of_squares_values_l3471_347140


namespace books_on_shelves_l3471_347135

theorem books_on_shelves (total : ℕ) (bottom middle top : ℕ) : 
  bottom = (total - bottom) / 2 →
  middle = (total - middle) / 3 →
  top = 30 →
  total = bottom + middle + top →
  total = 72 := by
sorry

end books_on_shelves_l3471_347135


namespace arithmetic_sequence_common_difference_l3471_347170

/-- 
Given an arithmetic sequence with n + 2 terms, where the first term is x and the last term is y,
prove that the common difference is (y - x) / (n + 1).
-/
theorem arithmetic_sequence_common_difference 
  (n : ℕ) (x y : ℝ) : 
  let d := (y - x) / (n + 1)
  ∀ (a : Fin (n + 2) → ℝ), 
    (a 0 = x) → 
    (a (Fin.last (n + 1)) = y) → 
    (∀ i : Fin (n + 1), a i.succ - a i = d) → 
    d = (y - x) / (n + 1) := by
  sorry

end arithmetic_sequence_common_difference_l3471_347170


namespace system_solution_l3471_347183

theorem system_solution :
  let solutions : List (ℝ × ℝ × ℝ) := [
    (1, 2, 3), (1, 5, -3), (3, -2, 5),
    (3, 3, -5), (6, -5, 2), (6, -3, -2)
  ]
  ∀ (x y z : ℝ),
    (3*x + 2*y + z = 10 ∧
     3*x^2 + 4*x*y + 2*x*z + y^2 + y*z = 27 ∧
     x^3 + 2*x^2*y + x^2*z + x*y^2 + x*y*z = 18) ↔
    (x, y, z) ∈ solutions := by
  sorry

end system_solution_l3471_347183


namespace symmetric_points_sum_power_l3471_347194

/-- 
Given two points A(m,3) and B(4,n) that are symmetric about the y-axis,
prove that (m+n)^2015 = -1
-/
theorem symmetric_points_sum_power (m n : ℝ) : 
  (∃ (A B : ℝ × ℝ), A = (m, 3) ∧ B = (4, n) ∧ 
   A.1 = -B.1 ∧ A.2 = B.2) → 
  (m + n)^2015 = -1 := by
  sorry

end symmetric_points_sum_power_l3471_347194


namespace equilateral_triangle_on_parabola_l3471_347148

/-- Given two points A(1, 0) and B(b, 0), if there exists a point C on the parabola y^2 = 4x
    such that triangle ABC is equilateral, then b = 5 or b = -1/3 -/
theorem equilateral_triangle_on_parabola (b : ℝ) :
  (∃ (x y : ℝ), y^2 = 4*x ∧ 
    ((x - 1)^2 + y^2 = (x - b)^2 + y^2) ∧
    ((x - 1)^2 + y^2 = (b - 1)^2) ∧
    ((x - b)^2 + y^2 = (b - 1)^2)) →
  b = 5 ∨ b = -1/3 :=
by sorry

end equilateral_triangle_on_parabola_l3471_347148


namespace sad_girls_count_l3471_347104

theorem sad_girls_count (total_children happy_children sad_children neutral_children
                         boys girls happy_boys neutral_boys : ℕ) : 
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  boys = 19 →
  girls = 41 →
  happy_boys = 6 →
  neutral_boys = 7 →
  total_children = happy_children + sad_children + neutral_children →
  total_children = boys + girls →
  ∃ (sad_girls : ℕ), sad_girls = 4 ∧ sad_children = sad_girls + (boys - happy_boys - neutral_boys) :=
by sorry

end sad_girls_count_l3471_347104


namespace radio_contest_winner_l3471_347118

theorem radio_contest_winner (n : ℕ) : 
  n > 1 ∧ 
  n < 35 ∧ 
  35 % n = 0 ∧ 
  35 % 7 = 0 ∧ 
  n ≠ 7 → 
  n = 5 := by sorry

end radio_contest_winner_l3471_347118


namespace five_n_plus_three_composite_l3471_347111

theorem five_n_plus_three_composite (n : ℕ+) 
  (h1 : ∃ k : ℕ+, 2 * n + 1 = k^2) 
  (h2 : ∃ m : ℕ+, 3 * n + 1 = m^2) : 
  ¬(Nat.Prime (5 * n + 3)) :=
by sorry

end five_n_plus_three_composite_l3471_347111


namespace equation_solution_exists_l3471_347108

theorem equation_solution_exists : ∃ (x y z t : ℕ+), x + y + z + t = 10 ∧ z = 7 := by
  sorry

end equation_solution_exists_l3471_347108


namespace sum_of_four_numbers_l3471_347132

theorem sum_of_four_numbers (a b c d : ℕ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : a > d)
  (h2 : a * b = c * d)
  (h3 : a + b + c + d = a * c) :
  a + b + c + d = 12 := by
sorry

end sum_of_four_numbers_l3471_347132


namespace solve_linear_equation_l3471_347174

theorem solve_linear_equation (x : ℝ) (h : 7 - x = 12) : x = -5 := by
  sorry

end solve_linear_equation_l3471_347174


namespace fourth_roots_of_unity_solution_l3471_347162

theorem fourth_roots_of_unity_solution (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (eq1 : a * k^3 + b * k^2 + c * k + d = 0)
  (eq2 : b * k^3 + c * k^2 + d * k + a = 0) :
  k = 1 ∨ k = -1 ∨ k = I ∨ k = -I :=
sorry

end fourth_roots_of_unity_solution_l3471_347162


namespace problem_solution_l3471_347112

/-- The function f(x) as defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + 8*x

/-- The function g(x) as defined in the problem -/
def g (a : ℝ) (x : ℝ) : ℝ := f a x - 7*x - a^2 + 3

theorem problem_solution :
  (∀ x > -2, ∀ a > 0,
    (a = 1 → {x | f a x ≥ 2*x + 1} = {x | x ≥ 0}) ∧
    ({a | ∀ x > -2, g a x ≥ 0} = Set.Ioo 0 2)) := by
  sorry

end problem_solution_l3471_347112


namespace rectangle_with_hole_area_formula_l3471_347128

/-- The area of a rectangle with a hole -/
def rectangle_with_hole_area (x : ℝ) : ℝ :=
  (2*x + 8) * (x + 6) - (3*x - 4) * (x + 1)

/-- Theorem: The area of the rectangle with a hole is equal to -x^2 + 21x + 52 -/
theorem rectangle_with_hole_area_formula (x : ℝ) :
  rectangle_with_hole_area x = -x^2 + 21*x + 52 := by
  sorry

end rectangle_with_hole_area_formula_l3471_347128


namespace total_production_theorem_l3471_347146

def week1_production : ℕ := 320
def week2_production : ℕ := 400
def week3_production : ℕ := 300

def total_3_weeks : ℕ := week1_production + week2_production + week3_production
def average_3_weeks : ℕ := total_3_weeks / 3
def total_4_weeks : ℕ := total_3_weeks + average_3_weeks

theorem total_production_theorem : total_4_weeks = 1360 := by
  sorry

end total_production_theorem_l3471_347146


namespace five_digit_divisible_count_l3471_347185

theorem five_digit_divisible_count : 
  let lcm := Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))
  let lower_bound := ((10000 + lcm - 1) / lcm) * lcm
  let upper_bound := (99999 / lcm) * lcm
  (upper_bound - lower_bound) / lcm + 1 = 179 := by
  sorry

end five_digit_divisible_count_l3471_347185


namespace side_length_relationship_l3471_347150

/-- Side length of an inscribed regular n-gon in a circle with radius r -/
def a (n : ℕ) (r : ℝ) : ℝ := sorry

/-- Side length of a circumscribed regular n-gon around a circle with radius r -/
def A (n : ℕ) (r : ℝ) : ℝ := sorry

/-- Theorem stating the relationship between side lengths of regular polygons -/
theorem side_length_relationship (n : ℕ) (r : ℝ) (h : 0 < r) :
  1 / A (2 * n) r = 1 / A n r + 1 / a n r := by sorry

end side_length_relationship_l3471_347150


namespace fraction_order_l3471_347117

theorem fraction_order : (24 : ℚ) / 19 < 23 / 17 ∧ 23 / 17 < 11 / 8 := by
  sorry

end fraction_order_l3471_347117


namespace base_seven_43210_equals_10738_l3471_347173

def base_seven_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_43210_equals_10738 :
  base_seven_to_decimal [0, 1, 2, 3, 4] = 10738 := by
  sorry

end base_seven_43210_equals_10738_l3471_347173


namespace masha_wins_l3471_347191

/-- Represents a pile of candies -/
structure Pile :=
  (size : ℕ)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)

/-- Calculates the number of moves required to split a pile into single candies -/
def movesForPile (p : Pile) : ℕ :=
  p.size - 1

/-- Calculates the total number of moves for all piles -/
def totalMoves (gs : GameState) : ℕ :=
  gs.piles.map movesForPile |>.sum

/-- Determines if the first player wins given a game state -/
def firstPlayerWins (gs : GameState) : Prop :=
  Odd (totalMoves gs)

/-- Theorem: Masha (first player) wins the candy splitting game -/
theorem masha_wins :
  let initialState : GameState := ⟨[⟨10⟩, ⟨20⟩, ⟨30⟩]⟩
  firstPlayerWins initialState := by
  sorry


end masha_wins_l3471_347191


namespace parallel_vectors_x_value_l3471_347199

/-- Given two vectors a and b in ℝ², if a is parallel to b and a = (1, -2) and b = (x, 1), then x = -1/2 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (1, -2))
  (h2 : b = (x, 1))
  (h_parallel : ∃ (k : ℝ), a = k • b) :
  x = -1/2 := by
  sorry

end parallel_vectors_x_value_l3471_347199


namespace hormone_related_phenomena_l3471_347192

-- Define the set of all phenomena
def Phenomena : Set String :=
  {"Fruit ripening", "Leaves turning yellow", "Fruit shedding", "CO2 fixation",
   "Topping cotton plants", "Absorption of mineral elements"}

-- Define the set of phenomena related to plant hormones
def HormoneRelatedPhenomena : Set String :=
  {"Fruit ripening", "Fruit shedding", "Topping cotton plants"}

-- Define a predicate for phenomena related to plant hormones
def isHormoneRelated (p : String) : Prop :=
  p ∈ HormoneRelatedPhenomena

-- Theorem statement
theorem hormone_related_phenomena :
  ∀ p ∈ Phenomena, isHormoneRelated p ↔
    (p = "Fruit ripening" ∨ p = "Fruit shedding" ∨ p = "Topping cotton plants") :=
by sorry

end hormone_related_phenomena_l3471_347192


namespace profit_1200_optimal_price_reduction_l3471_347116

/-- Represents the shirt sales scenario --/
structure ShirtSales where
  baseSales : ℕ := 20
  baseProfit : ℕ := 40
  salesIncrease : ℕ := 2
  priceReduction : ℚ

/-- Calculates the daily profit for a given price reduction --/
def dailyProfit (s : ShirtSales) : ℚ :=
  (s.baseProfit - s.priceReduction) * (s.baseSales + s.salesIncrease * s.priceReduction)

/-- Theorem for the price reductions that result in a daily profit of 1200 yuan --/
theorem profit_1200 (s : ShirtSales) :
  dailyProfit s = 1200 ↔ s.priceReduction = 10 ∨ s.priceReduction = 20 := by sorry

/-- Theorem for the optimal price reduction and maximum profit --/
theorem optimal_price_reduction (s : ShirtSales) :
  (∀ x, dailyProfit { s with priceReduction := x } ≤ dailyProfit { s with priceReduction := 15 }) ∧
  dailyProfit { s with priceReduction := 15 } = 1250 := by sorry

end profit_1200_optimal_price_reduction_l3471_347116


namespace glass_volume_l3471_347105

/-- The volume of a glass given pessimist and optimist perspectives --/
theorem glass_volume (V : ℝ) (h1 : V > 0) : 
  let pessimist_empty_percent : ℝ := 0.6
  let optimist_full_percent : ℝ := 0.6
  let water_difference : ℝ := 46
  (optimist_full_percent * V) - ((1 - pessimist_empty_percent) * V) = water_difference →
  V = 230 := by
  sorry

end glass_volume_l3471_347105


namespace cos_x_plus_2y_equals_one_l3471_347147

theorem cos_x_plus_2y_equals_one 
  (x y : ℝ) 
  (h_x : x ∈ Set.Icc (-π/4) (π/4))
  (h_y : y ∈ Set.Icc (-π/4) (π/4))
  (h_eq1 : ∃ a : ℝ, x^3 + Real.sin x - 2*a = 0)
  (h_eq2 : ∃ a : ℝ, 4*y^3 + (1/2) * Real.sin (2*y) + a = 0) :
  Real.cos (x + 2*y) = 1 :=
by sorry

end cos_x_plus_2y_equals_one_l3471_347147


namespace min_value_of_f_l3471_347121

def f (x : ℝ) := -2 * x + 5

theorem min_value_of_f :
  ∀ x ∈ Set.Icc 2 4, f x ≥ f 4 ∧ f 4 = -3 :=
by sorry

end min_value_of_f_l3471_347121


namespace train_speed_calculation_l3471_347156

theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) : 
  train_length = 120 ∧ crossing_time = 8 → 
  ∃ (speed : ℝ), speed = 54 ∧ 
  (2 * train_length) / crossing_time * 3.6 = 2 * speed := by
sorry

end train_speed_calculation_l3471_347156


namespace sin_negative_seventeen_pi_thirds_l3471_347167

theorem sin_negative_seventeen_pi_thirds :
  Real.sin (-17 * π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_negative_seventeen_pi_thirds_l3471_347167


namespace sara_initial_peaches_l3471_347158

/-- The number of peaches Sara picked at the orchard -/
def peaches_picked : ℕ := 37

/-- The total number of peaches Sara has now -/
def total_peaches_now : ℕ := 61

/-- The initial number of peaches Sara had -/
def initial_peaches : ℕ := total_peaches_now - peaches_picked

theorem sara_initial_peaches : initial_peaches = 24 := by
  sorry

end sara_initial_peaches_l3471_347158


namespace fifth_term_of_sequence_l3471_347145

def geometric_sequence (a : ℕ → ℝ) (x : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * (3 * x)

theorem fifth_term_of_sequence (a : ℕ → ℝ) (x : ℝ) :
  geometric_sequence a x →
  a 0 = 3 →
  a 1 = 9 * x →
  a 2 = 27 * x^2 →
  a 3 = 81 * x^3 →
  a 4 = 243 * x^4 := by
sorry

end fifth_term_of_sequence_l3471_347145


namespace jellybean_probability_l3471_347161

theorem jellybean_probability : 
  let total_jellybeans : ℕ := 12
  let red_jellybeans : ℕ := 5
  let blue_jellybeans : ℕ := 3
  let white_jellybeans : ℕ := 4
  let picked_jellybeans : ℕ := 3
  
  total_jellybeans = red_jellybeans + blue_jellybeans + white_jellybeans →
  
  (Nat.choose blue_jellybeans 2 * Nat.choose white_jellybeans 1) / 
  Nat.choose total_jellybeans picked_jellybeans = 3 / 55 := by
  sorry

end jellybean_probability_l3471_347161


namespace composite_function_properties_l3471_347171

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_domain : ∀ x, x ∈ Set.univ)
variable (h_increasing : ∀ x y, 2 < x ∧ x < y ∧ y < 6 → f x < f y)

-- Define the composite function g
def g (x : ℝ) := f (2 - x)

-- Theorem statement
theorem composite_function_properties :
  (∀ x y, 4 < x ∧ x < y ∧ y < 8 → g x < g y) ∧
  (∀ x, g x = g (4 - x)) :=
sorry

end composite_function_properties_l3471_347171


namespace parallelepipeds_from_four_points_l3471_347182

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A predicate that checks if four points are coplanar -/
def coplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧
    a * p1.x + b * p1.y + c * p1.z + d = 0 ∧
    a * p2.x + b * p2.y + c * p2.z + d = 0 ∧
    a * p3.x + b * p3.y + c * p3.z + d = 0 ∧
    a * p4.x + b * p4.y + c * p4.z + d = 0

/-- A function that counts the number of distinct parallelepipeds -/
def count_parallelepipeds (p1 p2 p3 p4 : Point3D) : ℕ :=
  sorry -- The actual implementation is not needed for the theorem statement

/-- Theorem stating that 4 non-coplanar points form 29 distinct parallelepipeds -/
theorem parallelepipeds_from_four_points (p1 p2 p3 p4 : Point3D) 
  (h : ¬coplanar p1 p2 p3 p4) : 
  count_parallelepipeds p1 p2 p3 p4 = 29 := by
  sorry

end parallelepipeds_from_four_points_l3471_347182


namespace system_solution_l3471_347196

theorem system_solution (a b x y : ℝ) : 
  (2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9 ∧ a = 8.3 ∧ b = 1.2) →
  (2 * (x + 2) - 3 * (y - 1) = 13 ∧ 3 * (x + 2) + 5 * (y - 1) = 30.9) →
  (x = 6.3 ∧ y = 2.2) :=
by sorry

end system_solution_l3471_347196


namespace bike_ride_time_l3471_347124

/-- Given a constant speed where 2 miles are covered in 8 minutes, 
    prove that the time required to cover 5 miles is 20 minutes. -/
theorem bike_ride_time (speed : ℝ) (distance_to_julia : ℝ) (time_to_julia : ℝ) 
  (distance_to_bernard : ℝ) : 
  distance_to_julia = 2 →
  time_to_julia = 8 →
  distance_to_bernard = 5 →
  speed = distance_to_julia / time_to_julia →
  distance_to_bernard / speed = 20 := by
  sorry

#check bike_ride_time

end bike_ride_time_l3471_347124


namespace fourth_root_cube_problem_l3471_347133

theorem fourth_root_cube_problem : 
  (((2 * Real.sqrt 2) ^ 3) ^ (1/4)) ^ 3 = 16 * Real.sqrt 2 := by sorry

end fourth_root_cube_problem_l3471_347133
