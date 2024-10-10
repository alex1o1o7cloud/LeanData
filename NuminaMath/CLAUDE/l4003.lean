import Mathlib

namespace triangle_angle_calculation_l4003_400343

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a = Real.sqrt 2 ∧
  b = 2 ∧
  Real.sin B + Real.cos B = Real.sqrt 2 →
  A = π / 6 := by
sorry

end triangle_angle_calculation_l4003_400343


namespace fraction_equality_l4003_400307

theorem fraction_equality (x y : ℝ) (h : x / y = 2) : (x - y) / y = 1 := by
  sorry

end fraction_equality_l4003_400307


namespace no_real_solutions_l4003_400308

theorem no_real_solutions :
  ∀ x : ℝ, (2*x - 3*x + 7)^2 + 4 ≠ -|2*x| :=
by sorry

end no_real_solutions_l4003_400308


namespace cos_180_degrees_l4003_400351

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end cos_180_degrees_l4003_400351


namespace cats_count_l4003_400344

/-- Represents the number of animals in a wildlife refuge --/
structure WildlifeRefuge where
  total_animals : ℕ
  birds : ℕ
  mammals : ℕ
  cats : ℕ
  dogs : ℕ

/-- The conditions of the wildlife refuge problem --/
def wildlife_refuge_conditions (w : WildlifeRefuge) : Prop :=
  w.total_animals = 1200 ∧
  w.birds = w.mammals + 145 ∧
  w.cats = w.dogs + 75 ∧
  w.mammals = w.cats + w.dogs ∧
  w.total_animals = w.birds + w.mammals

/-- The theorem stating that under the given conditions, the number of cats is 301 --/
theorem cats_count (w : WildlifeRefuge) :
  wildlife_refuge_conditions w → w.cats = 301 := by
  sorry


end cats_count_l4003_400344


namespace ammunition_depot_explosion_probability_l4003_400305

theorem ammunition_depot_explosion_probability 
  (p_first : ℝ) 
  (p_others : ℝ) 
  (h1 : p_first = 0.025) 
  (h2 : p_others = 0.1) : 
  1 - (1 - p_first) * (1 - p_others) * (1 - p_others) = 0.21025 := by
  sorry

end ammunition_depot_explosion_probability_l4003_400305


namespace tangent_line_of_odd_cubic_l4003_400361

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = x^3 + (a-1)x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a-1)*x^2 + a*x

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*(a-1)*x + a

/-- The tangent line equation at (0,0) is y = mx, where m is the slope at x = 0 -/
def TangentLineAt0 (f : ℝ → ℝ) (f' : ℝ → ℝ) : ℝ → ℝ := fun x ↦ (f' 0) * x

theorem tangent_line_of_odd_cubic (a : ℝ) :
  IsOdd (f a) → TangentLineAt0 (f a) (f' a) = fun x ↦ x := by sorry

end tangent_line_of_odd_cubic_l4003_400361


namespace no_division_between_valid_numbers_l4003_400366

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ),
    d₁ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₂ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₃ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₄ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₅ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₆ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₇ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₁ ≠ d₆ ∧ d₁ ≠ d₇ ∧
    d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₂ ≠ d₆ ∧ d₂ ≠ d₇ ∧
    d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₃ ≠ d₆ ∧ d₃ ≠ d₇ ∧
    d₄ ≠ d₅ ∧ d₄ ≠ d₆ ∧ d₄ ≠ d₇ ∧
    d₅ ≠ d₆ ∧ d₅ ≠ d₇ ∧
    d₆ ≠ d₇ ∧
    n = d₁ * 1000000 + d₂ * 100000 + d₃ * 10000 + d₄ * 1000 + d₅ * 100 + d₆ * 10 + d₇

theorem no_division_between_valid_numbers :
  ∀ a b : ℕ, is_valid_number a → is_valid_number b → a ≠ b → ¬(a ∣ b) := by
  sorry

end no_division_between_valid_numbers_l4003_400366


namespace digit_equation_solutions_l4003_400347

theorem digit_equation_solutions (n : ℕ) (x y z : ℕ) :
  n ≥ 2 →
  let a : ℚ := x * (10^n - 1) / 9
  let b : ℚ := y * (10^n - 1) / 9
  let c : ℚ := z * (10^(2*n) - 1) / 9
  a^2 + b = c →
  ((x = 3 ∧ y = 2 ∧ z = 1) ∨
   (x = 6 ∧ y = 8 ∧ z = 4) ∨
   (x = 8 ∧ y = 3 ∧ z = 7 ∧ n = 2)) :=
by sorry

end digit_equation_solutions_l4003_400347


namespace solution_of_equation_l4003_400300

theorem solution_of_equation :
  ∃! y : ℚ, 6 * (4 * y - 1) - 3 = 3 * (2 - 5 * y) ∧ y = 5 / 13 := by
  sorry

end solution_of_equation_l4003_400300


namespace min_sum_positive_reals_l4003_400302

theorem min_sum_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (2 * b) + b / (4 * c) + c / (8 * a)) ≥ (3 / 4) :=
by sorry

end min_sum_positive_reals_l4003_400302


namespace relationship_between_exponents_l4003_400353

theorem relationship_between_exponents (a c e f : ℝ) (x y q z : ℝ) 
  (h1 : a^(3*x) = c^(4*q))
  (h2 : a^(3*x) = e)
  (h3 : c^(4*q) = e)
  (h4 : c^(2*y) = a^(5*z))
  (h5 : c^(2*y) = f)
  (h6 : a^(5*z) = f)
  (h7 : a ≠ 0)
  (h8 : c ≠ 0)
  (h9 : e > 0)
  (h10 : f > 0) :
  3*y = 10*q := by
sorry

end relationship_between_exponents_l4003_400353


namespace propositions_truth_values_l4003_400365

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def is_solution (x : ℚ) : Prop := x^2 + x - 2 = 0

theorem propositions_truth_values :
  (is_prime 3 ∨ is_even 3) ∧
  ¬(is_prime 3 ∧ is_even 3) ∧
  ¬(¬is_prime 3) ∧
  (is_solution (-2) ∨ is_solution 1) ∧
  (is_solution (-2) ∧ is_solution 1) ∧
  ¬(¬is_solution (-2)) := by
  sorry

end propositions_truth_values_l4003_400365


namespace problem_solution_l4003_400359

theorem problem_solution (a b : ℕ) (h1 : a > b) (h2 : (a + b) + (3 * a + a * b - b) + 4 * a / b = 64) :
  a = 8 ∧ b = 2 := by
sorry

end problem_solution_l4003_400359


namespace smallest_gcd_multiple_l4003_400309

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 8) :
  ∃ (k : ℕ+), k ≥ 32 ∧ Nat.gcd (8 * m) (12 * n) = k ∧
  ∀ (l : ℕ+), Nat.gcd (8 * m) (12 * n) = l → l ≥ 32 :=
by sorry

end smallest_gcd_multiple_l4003_400309


namespace fitted_bowling_ball_volume_l4003_400317

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let sphere_diameter : ℝ := 30
  let hole_depth : ℝ := 8
  let small_hole_diameter : ℝ := 2
  let large_hole_diameter : ℝ := 3
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2) ^ 3
  let small_hole_volume := π * (small_hole_diameter / 2) ^ 2 * hole_depth
  let large_hole_volume := π * (large_hole_diameter / 2) ^ 2 * hole_depth
  sphere_volume - 2 * small_hole_volume - large_hole_volume = 4466 * π :=
by sorry

end fitted_bowling_ball_volume_l4003_400317


namespace marquita_garden_length_l4003_400342

-- Define the number of gardens for each person
def mancino_gardens : ℕ := 3
def marquita_gardens : ℕ := 2

-- Define the dimensions of Mancino's gardens
def mancino_garden_length : ℕ := 16
def mancino_garden_width : ℕ := 5

-- Define the width of Marquita's gardens
def marquita_garden_width : ℕ := 4

-- Define the total area of all gardens
def total_area : ℕ := 304

-- Theorem to prove
theorem marquita_garden_length :
  ∃ (l : ℕ), 
    mancino_gardens * mancino_garden_length * mancino_garden_width +
    marquita_gardens * l * marquita_garden_width = total_area ∧
    l = 8 := by
  sorry

end marquita_garden_length_l4003_400342


namespace max_integer_value_of_expression_l4003_400320

theorem max_integer_value_of_expression (x : ℝ) : 
  (4 * x^2 + 8 * x + 21) / (4 * x^2 + 8 * x + 9) ≤ 7/3 ∧ 
  ∃ (y : ℝ), (4 * y^2 + 8 * y + 21) / (4 * y^2 + 8 * y + 9) > 2 ∧
  ∀ (z : ℝ), (4 * z^2 + 8 * z + 21) / (4 * z^2 + 8 * z + 9) < 3 := by
  sorry

end max_integer_value_of_expression_l4003_400320


namespace prob_at_least_one_contract_l4003_400372

/-- The probability of getting at least one contract given specific probabilities for hardware and software contracts -/
theorem prob_at_least_one_contract 
  (p_hardware : ℝ) 
  (p_not_software : ℝ) 
  (p_both : ℝ) 
  (h1 : p_hardware = 4/5)
  (h2 : p_not_software = 3/5)
  (h3 : p_both = 0.3) :
  p_hardware + (1 - p_not_software) - p_both = 0.9 := by
  sorry

end prob_at_least_one_contract_l4003_400372


namespace problem1_solution_problem2_solution_l4003_400316

-- Problem 1
def problem1 (x y : ℤ) : ℤ :=
  (2 * x^2 * y - 4 * x * y^2) - (-3 * x * y^2 + x^2 * y)

theorem problem1_solution :
  problem1 (-1) 2 = 6 := by sorry

-- Problem 2
def A (x y : ℤ) : ℤ := x^2 - x*y + y^2
def B (x y : ℤ) : ℤ := -x^2 + 2*x*y + y^2

theorem problem2_solution :
  A 2010 (-1) + B 2010 (-1) = -2008 := by sorry

end problem1_solution_problem2_solution_l4003_400316


namespace smallest_x_value_l4003_400362

theorem smallest_x_value (x : ℝ) : 
  (((15 * x^2 - 40 * x + 20) / (4 * x - 3)) + 7 * x = 8 * x - 3) →
  x ≥ (25 - Real.sqrt 141) / 22 ∧
  ∃ y, y = (25 - Real.sqrt 141) / 22 ∧ 
     ((15 * y^2 - 40 * y + 20) / (4 * y - 3)) + 7 * y = 8 * y - 3 :=
by sorry

end smallest_x_value_l4003_400362


namespace sqrt_equation_solution_l4003_400389

theorem sqrt_equation_solution (y : ℝ) :
  y > 2 → (Real.sqrt (7 * y) / Real.sqrt (4 * (y - 2)) = 3) → y = 72 / 29 := by
  sorry

end sqrt_equation_solution_l4003_400389


namespace negation_equivalence_l4003_400328

theorem negation_equivalence :
  (¬ ∃ x : ℝ, Real.exp x > x) ↔ (∀ x : ℝ, Real.exp x ≤ x) :=
by sorry

end negation_equivalence_l4003_400328


namespace equation_a_l4003_400357

theorem equation_a (a : ℝ) (x : ℝ) : 
  (x + a) * (x + 2*a) * (x + 3*a) * (x + 4*a) = 3*a^4 ↔ 
  x = (-5*a + a*Real.sqrt 37)/2 ∨ x = (-5*a - a*Real.sqrt 37)/2 :=
sorry


end equation_a_l4003_400357


namespace circle_properties_l4003_400301

-- Define the circle C
def C : Set (ℝ × ℝ) := {(x, y) | x^2 + (y - 1)^2 = 9}

-- Define the line intercepting the chord
def L : Set (ℝ × ℝ) := {(x, y) | 12*x - 5*y - 8 = 0}

-- Define a general line through the origin
def l (k : ℝ) : Set (ℝ × ℝ) := {(x, y) | y = k*x}

-- Define point Q
def Q : ℝ × ℝ := (1, 2)

theorem circle_properties :
  -- Part 1: Length of the chord
  ∃ (A B : ℝ × ℝ), A ∈ C ∩ L ∧ B ∈ C ∩ L ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = 32 ∧
  -- Part 2: Sum of reciprocals of y-coordinates is constant
  ∀ (k : ℝ) (A B : ℝ × ℝ), k ≠ 0 → A ∈ C ∩ l k → B ∈ C ∩ l k → A ≠ B →
    1 / A.2 + 1 / B.2 = -1/4 ∧
  -- Part 3: Slope of line l when sum of squared distances is 22
  ∃ (k : ℝ) (A B : ℝ × ℝ), k ≠ 0 ∧ A ∈ C ∩ l k ∧ B ∈ C ∩ l k ∧ A ≠ B ∧
    (A.1 - Q.1)^2 + (A.2 - Q.2)^2 + (B.1 - Q.1)^2 + (B.2 - Q.2)^2 = 22 ∧ k = 1 :=
sorry

end circle_properties_l4003_400301


namespace f_solutions_when_a_neg_one_f_monotonic_increasing_iff_f_max_min_when_a_one_l4003_400392

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (x - 1) * |x - a|

-- Part 1
theorem f_solutions_when_a_neg_one :
  ∀ x : ℝ, f (-1) x = 1 ↔ (x ≤ -1 ∨ x = 1) :=
sorry

-- Part 2
theorem f_monotonic_increasing_iff :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≥ 1/3 :=
sorry

-- Part 3
theorem f_max_min_when_a_one :
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f 1 x ≤ 1) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f 1 x = 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f 1 x ≥ -1) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f 1 x = -1) :=
sorry

end f_solutions_when_a_neg_one_f_monotonic_increasing_iff_f_max_min_when_a_one_l4003_400392


namespace expand_expression_l4003_400381

theorem expand_expression (x : ℝ) : (17 * x + 21) * (3 * x) = 51 * x^2 + 63 * x := by
  sorry

end expand_expression_l4003_400381


namespace max_ab_value_l4003_400371

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * (x - 1)

theorem max_ab_value (a : ℝ) (h : a > 0) :
  (∃ b : ℝ, ∀ x : ℝ, f a x ≥ b) →
  (∃ M : ℝ, M = (exp 3) / 2 ∧ ∀ b : ℝ, (∀ x : ℝ, f a x ≥ b) → a * b ≤ M) :=
by sorry

end max_ab_value_l4003_400371


namespace ice_cream_scoops_l4003_400336

/-- Proves that Aaron and Carson each bought 8 scoops of ice cream given the problem conditions --/
theorem ice_cream_scoops (aaron_savings : ℚ) (carson_savings : ℚ) 
  (restaurant_bill_fraction : ℚ) (service_charge : ℚ) (ice_cream_cost : ℚ) 
  (leftover_money : ℚ) :
  aaron_savings = 150 →
  carson_savings = 150 →
  restaurant_bill_fraction = 3/4 →
  service_charge = 15/100 →
  ice_cream_cost = 4 →
  leftover_money = 4 →
  ∃ (scoops : ℕ), scoops = 8 ∧ 
    (aaron_savings + carson_savings) * restaurant_bill_fraction + 
    2 * scoops * ice_cream_cost + 2 * leftover_money = 
    aaron_savings + carson_savings :=
by sorry

end ice_cream_scoops_l4003_400336


namespace placemat_length_l4003_400397

theorem placemat_length (r : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) : 
  r = 5 → n = 8 → w = 1 → 
  y = 2 * r * Real.sin ((π / n) / 2) →
  y = 10 * Real.sin (5 * π / 16) :=
by sorry

end placemat_length_l4003_400397


namespace book_pages_calculation_l4003_400390

theorem book_pages_calculation (pages_remaining : ℕ) (percentage_read : ℚ) : 
  pages_remaining = 320 ∧ percentage_read = 1/5 → 
  pages_remaining / (1 - percentage_read) = 400 := by
sorry

end book_pages_calculation_l4003_400390


namespace max_sum_x_y_is_seven_l4003_400337

theorem max_sum_x_y_is_seven (x y : ℕ+) (h : x.val^4 = (x.val - 1) * (y.val^3 - 23) - 1) :
  x.val + y.val ≤ 7 ∧ ∃ (x₀ y₀ : ℕ+), x₀.val^4 = (x₀.val - 1) * (y₀.val^3 - 23) - 1 ∧ x₀.val + y₀.val = 7 := by
  sorry

end max_sum_x_y_is_seven_l4003_400337


namespace two_numbers_sum_and_difference_l4003_400311

theorem two_numbers_sum_and_difference (x y : ℝ) : 
  x + y = 18 ∧ x - y = 6 → x = 12 ∧ y = 6 := by
sorry

end two_numbers_sum_and_difference_l4003_400311


namespace emily_chocolate_sales_l4003_400323

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (bars_left : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - bars_left) * price_per_bar

/-- Proves that Emily makes $20 from selling chocolate bars -/
theorem emily_chocolate_sales : money_made 8 3 4 = 20 := by
  sorry

end emily_chocolate_sales_l4003_400323


namespace city_C_highest_growth_l4003_400333

structure City where
  name : String
  pop1970 : ℕ
  pop1980 : ℕ

def cities : List City := [
  { name := "A", pop1970 := 40, pop1980 := 50 },
  { name := "B", pop1970 := 50, pop1980 := 70 },
  { name := "C", pop1970 := 70, pop1980 := 100 },
  { name := "D", pop1970 := 100, pop1980 := 130 },
  { name := "E", pop1970 := 120, pop1980 := 160 }
]

def growthRatio (city : City) : ℚ :=
  city.pop1980 / city.pop1970

theorem city_C_highest_growth :
  ∃ c ∈ cities, c.name = "C" ∧
  ∀ other ∈ cities, growthRatio c ≥ growthRatio other :=
by sorry

end city_C_highest_growth_l4003_400333


namespace geometric_sequence_a5_l4003_400349

/-- A geometric sequence with a_1 = 1 and a_9 = 3 has a_5 = √3 -/
theorem geometric_sequence_a5 (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →
  a 9 = 3 →
  a 5 = Real.sqrt 3 := by
sorry

end geometric_sequence_a5_l4003_400349


namespace determine_x_value_l4003_400306

theorem determine_x_value (w y z x : ℤ) 
  (hw : w = 65)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 7) : 
  x = 112 := by
  sorry

end determine_x_value_l4003_400306


namespace union_of_M_and_N_l4003_400358

-- Define the sets M and N
def M : Set Nat := {1, 2}
def N : Set Nat := {2, 3}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by sorry

end union_of_M_and_N_l4003_400358


namespace equal_areas_imply_equal_dimensions_l4003_400338

theorem equal_areas_imply_equal_dimensions (square_side : ℝ) (rect_width : ℝ) (tri_base : ℝ) 
  (h1 : square_side = 4)
  (h2 : rect_width = 4)
  (h3 : tri_base = 8)
  (h4 : square_side ^ 2 = rect_width * (square_side ^ 2 / rect_width))
  (h5 : square_side ^ 2 = (tri_base * (2 * square_side ^ 2 / tri_base)) / 2) :
  square_side ^ 2 / rect_width = 4 ∧ 2 * square_side ^ 2 / tri_base = 4 :=
by sorry

end equal_areas_imply_equal_dimensions_l4003_400338


namespace gcd_75_360_l4003_400360

theorem gcd_75_360 : Nat.gcd 75 360 = 15 := by
  sorry

end gcd_75_360_l4003_400360


namespace complex_modulus_proof_l4003_400352

theorem complex_modulus_proof (z z₁ z₂ : ℂ) 
  (h₁ : z₁ ≠ z₂)
  (h₂ : z₁^2 = -2 - 2 * Complex.I * Real.sqrt 3)
  (h₃ : z₂^2 = -2 - 2 * Complex.I * Real.sqrt 3)
  (h₄ : Complex.abs (z - z₁) = 4)
  (h₅ : Complex.abs (z - z₂) = 4) :
  Complex.abs z = 2 * Real.sqrt 3 := by
sorry

end complex_modulus_proof_l4003_400352


namespace daves_age_ratio_l4003_400321

theorem daves_age_ratio (D N : ℚ) : 
  (D > 0) → 
  (N > 0) → 
  (∃ (a b c d : ℚ), a + b + c + d = D) → -- Combined ages of four children equal D
  (D - N = 3 * (D - 4 * N)) → -- N years ago, Dave's age was thrice the sum of children's ages
  D / N = 11 / 2 := by
sorry

end daves_age_ratio_l4003_400321


namespace oil_leaked_during_fix_correct_l4003_400394

/-- The amount of oil leaked while engineers were fixing the pipe -/
def oil_leaked_during_fix (total_leaked : ℕ) (leaked_before : ℕ) : ℕ :=
  total_leaked - leaked_before

/-- Theorem: The amount of oil leaked during fix is correct -/
theorem oil_leaked_during_fix_correct 
  (total_leaked : ℕ) 
  (leaked_before : ℕ) 
  (h1 : total_leaked = 6206)
  (h2 : leaked_before = 2475) :
  oil_leaked_during_fix total_leaked leaked_before = 3731 :=
by sorry

end oil_leaked_during_fix_correct_l4003_400394


namespace smallest_winning_number_l4003_400325

/-- Represents the state of the game -/
inductive GameState
  | WinningPosition
  | LosingPosition

/-- Determines if a move is valid according to the game rules -/
def validMove (n : ℕ) (k : ℕ) : Prop :=
  k ≥ 1 ∧ 
  ((n % 2 = 0 ∧ k % 2 = 0 ∧ k ≤ n / 2) ∨ 
   (n % 2 = 1 ∧ k % 2 = 1 ∧ n / 2 ≤ k ∧ k ≤ n))

/-- Determines the game state for a given number of marbles -/
def gameState (n : ℕ) : GameState :=
  if n = 2^17 - 2 then GameState.LosingPosition else GameState.WinningPosition

/-- The main theorem to prove -/
theorem smallest_winning_number : 
  (∀ n, 100000 ≤ n ∧ n < 131070 → gameState n = GameState.WinningPosition) ∧
  gameState 131070 = GameState.LosingPosition :=
sorry

end smallest_winning_number_l4003_400325


namespace expected_value_specific_coin_l4003_400329

/-- A three-sided coin with probabilities and winnings for each outcome -/
structure ThreeSidedCoin where
  prob_heads : ℚ
  prob_tails : ℚ
  prob_edge : ℚ
  win_heads : ℚ
  win_tails : ℚ
  win_edge : ℚ

/-- The expected value of winnings for a three-sided coin flip -/
def expectedValue (coin : ThreeSidedCoin) : ℚ :=
  coin.prob_heads * coin.win_heads +
  coin.prob_tails * coin.win_tails +
  coin.prob_edge * coin.win_edge

/-- Theorem stating the expected value of winnings for a specific three-sided coin -/
theorem expected_value_specific_coin :
  ∃ (coin : ThreeSidedCoin),
    coin.prob_heads = 1/4 ∧
    coin.prob_tails = 3/4 - 1/20 ∧
    coin.prob_edge = 1/20 ∧
    coin.win_heads = 4 ∧
    coin.win_tails = -3 ∧
    coin.win_edge = -1 ∧
    coin.prob_heads + coin.prob_tails + coin.prob_edge = 1 ∧
    expectedValue coin = -23/20 := by
  sorry

end expected_value_specific_coin_l4003_400329


namespace max_ab_value_l4003_400324

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a * log x + (a + 1) * x - (1/2) * x^2

theorem max_ab_value (a : ℝ) (h_a : a > 0) :
  (∀ x > 0, f a x ≥ -(1/2) * x^2 + a * x + b) →
  (∃ c : ℝ, c = Real.exp 1 / 2 ∧ ∀ b : ℝ, a * b ≤ c) :=
sorry

end max_ab_value_l4003_400324


namespace quadratic_function_properties_l4003_400374

/-- A quadratic function with vertex (1,16) and roots 8 units apart -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 15

/-- The function g(x) defined in terms of f(x) and a parameter a -/
def g (a : ℝ) (x : ℝ) : ℝ := (2 - 2*a)*x - f x

theorem quadratic_function_properties :
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ |x₁ - x₂| = 8) ∧
  (∀ x : ℝ, f x ≤ f 1) ∧
  f 1 = 16 ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc 0 2, Monotone (g a)) ↔ a ≤ 0) := by
  sorry

end quadratic_function_properties_l4003_400374


namespace decagon_diagonals_l4003_400395

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: A regular decagon has 35 diagonals -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l4003_400395


namespace sunglasses_hat_probability_l4003_400380

theorem sunglasses_hat_probability (total_sunglasses : ℕ) (total_hats : ℕ) 
  (prob_hat_and_sunglasses : ℚ) :
  total_sunglasses = 80 →
  total_hats = 60 →
  prob_hat_and_sunglasses = 1/3 →
  (prob_hat_and_sunglasses * total_hats) / total_sunglasses = 1/4 :=
by sorry

end sunglasses_hat_probability_l4003_400380


namespace remaining_slices_for_phill_l4003_400363

/-- Represents the number of slices in a pizza -/
def total_slices : ℕ := 8

/-- Represents the number of slices given to friends -/
def given_slices : ℕ := 7

/-- Theorem stating that the remaining slices for Phill is 1 -/
theorem remaining_slices_for_phill : total_slices - given_slices = 1 := by
  sorry

end remaining_slices_for_phill_l4003_400363


namespace solve_apples_problem_l4003_400398

def apples_problem (initial_apples : ℕ) (given_to_father : ℕ) (apples_per_person : ℕ) : Prop :=
  let remaining_apples := initial_apples - given_to_father
  let friends := (remaining_apples - apples_per_person) / apples_per_person
  friends = 4

theorem solve_apples_problem :
  apples_problem 55 10 9 :=
by sorry

end solve_apples_problem_l4003_400398


namespace highest_divisible_digit_l4003_400346

theorem highest_divisible_digit : 
  ∃ (a : ℕ), a ≤ 9 ∧ 
  (∀ (b : ℕ), b ≤ 9 → 365 * 100 * b + 16 ≡ 0 [MOD 8] → b ≤ a) ∧
  (365 * 100 * a + 16 ≡ 0 [MOD 8]) :=
by sorry

end highest_divisible_digit_l4003_400346


namespace sum_of_gcd_and_lcm_l4003_400312

def A : ℕ := Nat.gcd 18 (Nat.gcd 24 36)
def B : ℕ := Nat.lcm 18 (Nat.lcm 24 36)

theorem sum_of_gcd_and_lcm : A + B = 78 := by
  sorry

end sum_of_gcd_and_lcm_l4003_400312


namespace probability_at_least_one_white_l4003_400369

/-- The probability of drawing at least one white ball from a box -/
theorem probability_at_least_one_white (total : ℕ) (white : ℕ) (red : ℕ) (draw : ℕ) :
  total = white + red →
  white = 8 →
  red = 2 →
  draw = 2 →
  (Nat.choose white 1 * Nat.choose red 1 + Nat.choose white 2 * Nat.choose red 0) / Nat.choose total draw = 44 / 45 :=
by sorry

end probability_at_least_one_white_l4003_400369


namespace inequality_solution_l4003_400334

theorem inequality_solution (m : ℝ) : 
  (∃ (a : ℝ), a = 5 ∧ 
   ∃ (x : ℝ), |x - 1| - |x + m| ≥ a ∧ 
   ∀ (b : ℝ), (∃ (y : ℝ), |y - 1| - |y + m| ≥ b) → b ≤ a) → 
  m = 4 ∨ m = -6 :=
by sorry

end inequality_solution_l4003_400334


namespace red_balls_count_l4003_400313

theorem red_balls_count (total : ℕ) (prob : ℚ) : 
  total = 15 →
  prob = 1 / 21 →
  ∃ (r : ℕ), r ≤ total ∧ 
    (r : ℚ) / total * ((r : ℚ) - 1) / (total - 1 : ℚ) = prob ∧
    r = 5 :=
by sorry

end red_balls_count_l4003_400313


namespace f_range_l4003_400348

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then |x| - 1 else Real.sin x ^ 2

theorem f_range :
  Set.range f = Set.Ioi (-1) := by sorry

end f_range_l4003_400348


namespace equation_solution_l4003_400370

theorem equation_solution (x : ℝ) (h : x ≠ 0) : 4 / x^2 = x / 16 → x = 4 := by
  sorry

end equation_solution_l4003_400370


namespace expression_evaluation_l4003_400399

theorem expression_evaluation : 
  (-2/3)^2023 * (3/2)^2022 = -2/3 := by
  sorry

end expression_evaluation_l4003_400399


namespace range_of_m_l4003_400367

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | -1 < x ∧ x < m + 1}

-- State the theorem
theorem range_of_m (m : ℝ) : B m ⊂ A → -2 < m ∧ m < 2 := by
  sorry

end range_of_m_l4003_400367


namespace ends_with_k_zeros_l4003_400384

/-- A p-adic integer with a nonzero last digit -/
def NonZeroLastDigitPAdicInteger (p : ℕ) (a : ℕ) : Prop :=
  Nat.Prime p ∧ a % p ≠ 0

theorem ends_with_k_zeros (p k : ℕ) (a : ℕ) 
  (h_p : Nat.Prime p) 
  (h_a : NonZeroLastDigitPAdicInteger p a) 
  (h_k : k > 0) :
  (a^(p^(k-1) * (p-1)) - 1) % p^k = 0 := by
sorry

end ends_with_k_zeros_l4003_400384


namespace a_21_value_l4003_400377

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The property of being a geometric sequence -/
def IsGeometric (b : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem a_21_value
  (a b : Sequence)
  (h1 : a 1 = 1)
  (h2 : IsGeometric b)
  (h3 : ∀ n : ℕ, b n = a (n + 1) / a n)
  (h4 : b 10 * b 11 = 52) :
  a 21 = 4 := by
sorry

end a_21_value_l4003_400377


namespace jacob_calorie_limit_l4003_400322

/-- Jacob's calorie intake and limit problem -/
theorem jacob_calorie_limit :
  ∀ (breakfast lunch dinner total_eaten planned_limit : ℕ),
    breakfast = 400 →
    lunch = 900 →
    dinner = 1100 →
    total_eaten = breakfast + lunch + dinner →
    total_eaten = planned_limit + 600 →
    planned_limit = 1800 := by
  sorry

end jacob_calorie_limit_l4003_400322


namespace max_profit_theorem_l4003_400387

/-- Represents the profit function for a mobile phone store -/
def profit_function (x : ℝ) : ℝ := -200 * x + 140000

/-- Represents the constraint on the number of type B phones -/
def constraint (x : ℝ) : Prop := 100 - x ≤ 3 * x

/-- Theorem stating the maximum profit and optimal purchase strategy -/
theorem max_profit_theorem :
  ∃ (x : ℝ),
    x ≥ 0 ∧
    x ≤ 100 ∧
    constraint x ∧
    profit_function x = 135000 ∧
    (∀ y : ℝ, y ≥ 0 → y ≤ 100 → constraint y → profit_function y ≤ profit_function x) ∧
    x = 25 :=
  sorry

end max_profit_theorem_l4003_400387


namespace toy_store_shelves_l4003_400373

/-- The number of shelves needed to display bears in a toy store. -/
def shelves_needed (initial_stock new_shipment bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

/-- Theorem stating that the toy store used 2 shelves to display the bears. -/
theorem toy_store_shelves :
  shelves_needed 5 7 6 = 2 := by
  sorry

end toy_store_shelves_l4003_400373


namespace painting_efficiency_theorem_l4003_400330

/-- Represents the efficiency of painting classrooms -/
structure PaintingEfficiency where
  groups : ℕ
  workers_per_group : ℕ
  hours_per_day : ℕ
  classrooms : ℕ
  days : ℚ

/-- The initial painting scenario -/
def initial_scenario : PaintingEfficiency :=
  { groups := 6
  , workers_per_group := 6
  , hours_per_day := 6
  , classrooms := 6
  , days := 6 }

/-- The new painting scenario -/
def new_scenario : PaintingEfficiency :=
  { groups := 8
  , workers_per_group := 8
  , hours_per_day := 8
  , classrooms := 8
  , days := 27/8 }

/-- Calculates the painting rate (classrooms per worker-hour) -/
def painting_rate (p : PaintingEfficiency) : ℚ :=
  p.classrooms / (p.groups * p.workers_per_group * p.hours_per_day * p.days)

theorem painting_efficiency_theorem :
  painting_rate initial_scenario = painting_rate new_scenario := by
  sorry

#check painting_efficiency_theorem

end painting_efficiency_theorem_l4003_400330


namespace retirement_plan_ratio_l4003_400382

/-- Represents the number of workers in each category -/
structure WorkerCounts where
  men : ℕ
  women : ℕ
  withPlan : ℕ
  withoutPlan : ℕ

/-- Represents the percentages of workers in different categories -/
structure WorkerPercentages where
  womenWithoutPlan : ℚ
  menWithPlan : ℚ

/-- The main theorem about the ratio of workers without a retirement plan -/
theorem retirement_plan_ratio
  (counts : WorkerCounts)
  (percentages : WorkerPercentages)
  (h1 : counts.men = 120)
  (h2 : counts.women = 180)
  (h3 : percentages.womenWithoutPlan = 3/5)
  (h4 : percentages.menWithPlan = 2/5)
  (h5 : counts.men + counts.women = counts.withPlan + counts.withoutPlan)
  (h6 : percentages.womenWithoutPlan * counts.withoutPlan = counts.women - percentages.menWithPlan * counts.withPlan)
  (h7 : (1 - percentages.womenWithoutPlan) * counts.withoutPlan = counts.men - percentages.menWithPlan * counts.withPlan) :
  counts.withoutPlan * 13 = (counts.withPlan + counts.withoutPlan) * 9 :=
sorry

end retirement_plan_ratio_l4003_400382


namespace journey_time_change_l4003_400356

/-- Proves that for a journey of 40 km, if increasing the speed by 3 kmph reduces
    the time by 40 minutes, then decreasing the speed by 2 kmph from the original
    speed increases the time by 40 minutes. -/
theorem journey_time_change (v : ℝ) (h1 : v > 0) : 
  (40 / v - 40 / (v + 3) = 2 / 3) → 
  (40 / (v - 2) - 40 / v = 2 / 3) := by
  sorry

end journey_time_change_l4003_400356


namespace triangle_formation_l4003_400332

/-- Check if three lengths can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given two stick lengths -/
def stick1 : ℝ := 3
def stick2 : ℝ := 5

theorem triangle_formation :
  ¬(canFormTriangle stick1 stick2 2) ∧
  (canFormTriangle stick1 stick2 3) ∧
  (canFormTriangle stick1 stick2 4) ∧
  (canFormTriangle stick1 stick2 6) := by
  sorry

end triangle_formation_l4003_400332


namespace classroom_notebooks_l4003_400331

theorem classroom_notebooks (total_students : ℕ) 
  (h1 : total_students = 28)
  (group1_notebooks : ℕ) (h2 : group1_notebooks = 5)
  (group2_notebooks : ℕ) (h3 : group2_notebooks = 3)
  (group3_notebooks : ℕ) (h4 : group3_notebooks = 7) :
  (total_students / 3) * group1_notebooks +
  (total_students / 3) * group2_notebooks +
  (total_students - 2 * (total_students / 3)) * group3_notebooks = 142 :=
by sorry

end classroom_notebooks_l4003_400331


namespace expression_evaluation_l4003_400354

theorem expression_evaluation : (50 - (3050 - 501))^2 + (3050 - (501 - 50)) = 6251600 := by
  sorry

end expression_evaluation_l4003_400354


namespace cindys_calculation_l4003_400341

theorem cindys_calculation (x : ℤ) : (x - 7) / 5 = 37 → (x - 5) / 7 = 26 := by
  sorry

end cindys_calculation_l4003_400341


namespace log2_derivative_l4003_400319

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
sorry

end log2_derivative_l4003_400319


namespace tom_swim_time_l4003_400318

/-- Proves that Tom swam for 2 hours given the conditions of the problem -/
theorem tom_swim_time (swim_speed : ℝ) (run_speed_multiplier : ℝ) (total_distance : ℝ) :
  swim_speed = 2 →
  run_speed_multiplier = 4 →
  total_distance = 12 →
  ∃ (swim_time : ℝ),
    swim_time * swim_speed + (swim_time / 2) * (run_speed_multiplier * swim_speed) = total_distance ∧
    swim_time = 2 := by
  sorry

end tom_swim_time_l4003_400318


namespace kids_difference_l4003_400385

theorem kids_difference (monday tuesday : ℕ) 
  (h1 : monday = 18) 
  (h2 : tuesday = 10) : 
  monday - tuesday = 8 := by
sorry

end kids_difference_l4003_400385


namespace consecutive_color_draw_probability_l4003_400378

def numTan : ℕ := 4
def numPink : ℕ := 3
def numViolet : ℕ := 5
def totalChips : ℕ := numTan + numPink + numViolet

theorem consecutive_color_draw_probability :
  (numTan.factorial * numPink.factorial * numViolet.factorial) / totalChips.factorial = 1 / 27720 :=
by sorry

end consecutive_color_draw_probability_l4003_400378


namespace elevator_weight_problem_l4003_400355

/-- Given 10 people in an elevator with an average weight of 165 lbs, 
    prove that if an 11th person enters and increases the average weight to 170 lbs, 
    then the weight of the 11th person is 220 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
  (new_avg_weight : ℝ) (new_person_weight : ℝ) :
  initial_people = 10 →
  initial_avg_weight = 165 →
  new_avg_weight = 170 →
  (initial_people * initial_avg_weight + new_person_weight) / (initial_people + 1) = new_avg_weight →
  new_person_weight = 220 :=
by sorry

end elevator_weight_problem_l4003_400355


namespace count_valid_numbers_l4003_400304

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 10 % 10 : ℚ) = ((n / 100) + (n % 10)) / 2 ∧
  n % 10 = 2 * (n / 100)

theorem count_valid_numbers : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid_number n) ∧ S.card = 2 :=
sorry

end count_valid_numbers_l4003_400304


namespace factor_sum_l4003_400350

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 - 4*X + 8) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 64 :=
by sorry

end factor_sum_l4003_400350


namespace inverse_g_sum_l4003_400364

-- Define the function g
def g (x : ℝ) : ℝ := x * |x| + 3 * x

-- State the theorem
theorem inverse_g_sum : 
  ∃ (a b : ℝ), g a = 9 ∧ g b = -121 ∧ a + b = (3 * Real.sqrt 5 - 23) / 2 :=
sorry

end inverse_g_sum_l4003_400364


namespace product_remainder_mod_seven_l4003_400345

theorem product_remainder_mod_seven : (1233 * 1984 * 2006 * 2021) % 7 = 2 := by
  sorry

end product_remainder_mod_seven_l4003_400345


namespace system_equations_solution_system_inequalities_solution_l4003_400375

-- Part 1: System of equations
theorem system_equations_solution (x y : ℝ) : 
  (x = 4*y + 1 ∧ 2*x - 5*y = 8) → (x = 9 ∧ y = 2) := by sorry

-- Part 2: System of inequalities
theorem system_inequalities_solution (x : ℝ) :
  (4*x - 5 ≤ 3 ∧ (x - 1) / 3 < (2*x + 1) / 5) ↔ (-8 < x ∧ x ≤ 2) := by sorry

end system_equations_solution_system_inequalities_solution_l4003_400375


namespace range_of_a_l4003_400314

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(5-2*a))^x > (-(5-2*a))^y

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Iic (-2) :=
by sorry

end range_of_a_l4003_400314


namespace solve_equation_l4003_400335

theorem solve_equation (y : ℚ) (h : (1 : ℚ) / 3 - (1 : ℚ) / 4 = 1 / y) : y = 12 := by
  sorry

end solve_equation_l4003_400335


namespace thermometer_distribution_methods_l4003_400326

/-- The number of ways to distribute thermometers among classes. -/
def distribute_thermometers (total_thermometers : ℕ) (num_classes : ℕ) (min_per_class : ℕ) : ℕ :=
  Nat.choose num_classes 1 + 
  2 * Nat.choose num_classes 2 + 
  Nat.choose num_classes 3

/-- Theorem stating the number of distribution methods for the given problem. -/
theorem thermometer_distribution_methods : 
  distribute_thermometers 23 10 2 = 220 := by
  sorry

#eval distribute_thermometers 23 10 2

end thermometer_distribution_methods_l4003_400326


namespace third_root_of_polynomial_l4003_400396

/-- Given a polynomial ax^3 + (a + 3b)x^2 + (b - 4a)x + (10 - a) with roots -3 and 4,
    prove that the third root is -17/10 -/
theorem third_root_of_polynomial (a b : ℝ) :
  (∀ x : ℝ, a * x^3 + (a + 3*b) * x^2 + (b - 4*a) * x + (10 - a) = 0 ↔ x = -3 ∨ x = 4 ∨ x = -17/10) :=
by sorry

end third_root_of_polynomial_l4003_400396


namespace initial_breads_count_l4003_400303

/-- The number of thieves -/
def num_thieves : ℕ := 5

/-- The number of breads remaining after all thieves -/
def remaining_breads : ℕ := 3

/-- Function to calculate the number of breads after a thief takes their share -/
def breads_after_thief (x : ℚ) : ℚ := x / 2 - 1 / 2

/-- Function to calculate the number of breads after n thieves -/
def breads_after_n_thieves : ℕ → ℚ → ℚ
  | 0, x => x
  | n + 1, x => breads_after_n_thieves n (breads_after_thief x)

/-- Theorem stating that the initial number of breads was 127 -/
theorem initial_breads_count : 
  breads_after_n_thieves num_thieves 127 = remaining_breads := by sorry

end initial_breads_count_l4003_400303


namespace isosceles_right_triangle_hypotenuse_squared_l4003_400340

theorem isosceles_right_triangle_hypotenuse_squared 
  (u v w : ℂ) (s t : ℝ) (k : ℝ) : 
  (∀ z : ℂ, z^3 + 2*z^2 + s*z + t = 0 ↔ z = u ∨ z = v ∨ z = w) → 
  Complex.abs u^2 + Complex.abs v^2 + Complex.abs w^2 = 350 →
  ∃ (x y : ℝ), 
    (Complex.abs (u - v))^2 = x^2 + y^2 ∧ 
    (Complex.abs (v - w))^2 = x^2 + y^2 ∧
    (Complex.abs (w - u))^2 = x^2 + y^2 ∧
    k^2 = (Complex.abs (w - u))^2 →
  k^2 = 525 := by sorry

end isosceles_right_triangle_hypotenuse_squared_l4003_400340


namespace function_with_same_length_image_l4003_400386

-- Define the property for f
def HasSameLengthImage (f : ℝ → ℝ) : Prop :=
  ∀ (a b : ℝ), a < b → ∃ (c d : ℝ), c < d ∧ 
    (Set.Ioo c d = f '' Set.Ioo a b) ∧ 
    (d - c = b - a)

-- State the theorem
theorem function_with_same_length_image (f : ℝ → ℝ) 
  (h : HasSameLengthImage f) : 
  ∃ (C : ℝ), (∀ x, f x = x + C) ∨ (∀ x, f x = -x + C) := by
  sorry

end function_with_same_length_image_l4003_400386


namespace frank_final_position_l4003_400315

/-- Represents Frank's position relative to his starting point -/
def dance_position (back1 forward1 back2 : ℤ) : ℤ :=
  -back1 + forward1 - back2 + 2 * back2

/-- Proves that Frank's final position is 7 steps forward from his starting point -/
theorem frank_final_position :
  dance_position 5 10 2 = 7 := by sorry

end frank_final_position_l4003_400315


namespace thirty_percent_less_than_ninety_equals_one_fourth_more_than_fifty_l4003_400391

theorem thirty_percent_less_than_ninety_equals_one_fourth_more_than_fifty : 
  (90 : ℝ) * (1 - 0.3) = 50 * (1 + 0.25) := by sorry

end thirty_percent_less_than_ninety_equals_one_fourth_more_than_fifty_l4003_400391


namespace largest_four_digit_sum_25_l4003_400376

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Checks if a FourDigitNumber is valid (each digit is less than 10) -/
def isValidFourDigitNumber (n : FourDigitNumber) : Prop :=
  n.1 < 10 ∧ n.2.1 < 10 ∧ n.2.2.1 < 10 ∧ n.2.2.2 < 10

/-- Calculates the sum of digits of a FourDigitNumber -/
def digitSum (n : FourDigitNumber) : Nat :=
  n.1 + n.2.1 + n.2.2.1 + n.2.2.2

/-- Converts a FourDigitNumber to its numerical value -/
def toNumber (n : FourDigitNumber) : Nat :=
  1000 * n.1 + 100 * n.2.1 + 10 * n.2.2.1 + n.2.2.2

/-- The main theorem stating that 9970 is the largest four-digit number with digit sum 25 -/
theorem largest_four_digit_sum_25 :
  ∀ n : FourDigitNumber,
    isValidFourDigitNumber n →
    digitSum n = 25 →
    toNumber n ≤ 9970 := by
  sorry

end largest_four_digit_sum_25_l4003_400376


namespace sandbox_volume_l4003_400368

def sandbox_length : ℝ := 312
def sandbox_width : ℝ := 146
def sandbox_depth : ℝ := 56

theorem sandbox_volume :
  sandbox_length * sandbox_width * sandbox_depth = 2555520 := by
  sorry

end sandbox_volume_l4003_400368


namespace company_workforce_l4003_400388

theorem company_workforce (initial_total : ℕ) : 
  (initial_total * 3 / 5 : ℚ) = initial_total * 0.6 →
  (initial_total * 3 / 5 : ℚ) / (initial_total + 28) = 0.55 →
  initial_total + 28 = 336 := by
sorry

end company_workforce_l4003_400388


namespace nicholas_crackers_l4003_400327

theorem nicholas_crackers (marcus_crackers : ℕ) (mona_crackers : ℕ) (nicholas_crackers : ℕ)
  (h1 : marcus_crackers = 27)
  (h2 : marcus_crackers = 3 * mona_crackers)
  (h3 : nicholas_crackers = mona_crackers + 6) :
  nicholas_crackers = 15 := by
sorry

end nicholas_crackers_l4003_400327


namespace seventh_term_value_l4003_400310

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first five terms
  sum_first_five : ℚ
  -- The sixth term
  sixth_term : ℚ
  -- Property: The sum of the first five terms is 15
  sum_property : sum_first_five = 15
  -- Property: The sixth term is 7
  sixth_property : sixth_term = 7

/-- The seventh term of the arithmetic sequence -/
def seventh_term (seq : ArithmeticSequence) : ℚ := 25/3

/-- Theorem: The seventh term of the arithmetic sequence is 25/3 -/
theorem seventh_term_value (seq : ArithmeticSequence) :
  seventh_term seq = 25/3 := by
  sorry

end seventh_term_value_l4003_400310


namespace gardener_path_tiles_l4003_400393

def park_width : ℕ := 13
def park_length : ℕ := 19

theorem gardener_path_tiles :
  ∀ (avoid : ℕ), avoid = 1 →
  (park_width + park_length - Nat.gcd park_width park_length) - avoid = 30 := by
sorry

end gardener_path_tiles_l4003_400393


namespace same_monotonicity_implies_phi_value_l4003_400379

open Real

theorem same_monotonicity_implies_phi_value (φ : Real) :
  (∀ x ∈ Set.Icc 0 (π / 2), 
    (∀ y ∈ Set.Icc 0 (π / 2), x < y → cos (2 * x) > cos (2 * y)) ↔ 
    (∀ y ∈ Set.Icc 0 (π / 2), x < y → sin (x + φ) > sin (y + φ))) →
  φ = π / 2 := by
sorry

end same_monotonicity_implies_phi_value_l4003_400379


namespace count_theorem_l4003_400339

/-- Count of numbers between 100 and 799 with digits in strictly increasing order -/
def strictlyIncreasingCount : Nat := Nat.choose 7 3

/-- Count of numbers between 100 and 799 with last two digits equal -/
def lastTwoEqualCount : Nat := Nat.choose 7 2

/-- Total count of numbers between 100 and 799 with digits in strictly increasing order or equal to the last digit -/
def totalCount : Nat := strictlyIncreasingCount + lastTwoEqualCount

theorem count_theorem : totalCount = 56 := by sorry

end count_theorem_l4003_400339


namespace parabola_b_value_l4003_400383

/-- Given a parabola y = x^2 + ax + b passing through (2, 5) and (-2, -11), prove b = -7 -/
theorem parabola_b_value (a b : ℝ) : 
  (5 = 2^2 + 2*a + b) ∧ (-11 = (-2)^2 + (-2)*a + b) → b = -7 := by
  sorry

end parabola_b_value_l4003_400383
