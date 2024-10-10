import Mathlib

namespace sqrt_four_plus_abs_sqrt_three_minus_two_l2421_242114

theorem sqrt_four_plus_abs_sqrt_three_minus_two :
  Real.sqrt 4 + |Real.sqrt 3 - 2| = 4 - Real.sqrt 3 := by
  sorry

end sqrt_four_plus_abs_sqrt_three_minus_two_l2421_242114


namespace initial_profit_percentage_l2421_242136

/-- Proves that given an article with a cost price of Rs. 50, if reducing the cost price by 20% 
    and the selling price by Rs. 10.50 results in a 30% profit, then the initial profit percentage is 25%. -/
theorem initial_profit_percentage 
  (cost : ℝ) 
  (reduced_cost_percentage : ℝ) 
  (reduced_selling_price : ℝ) 
  (new_profit_percentage : ℝ) :
  cost = 50 →
  reduced_cost_percentage = 0.8 →
  reduced_selling_price = 10.5 →
  new_profit_percentage = 0.3 →
  (reduced_cost_percentage * cost * (1 + new_profit_percentage) - reduced_selling_price) / cost * 100 = 25 := by
sorry

end initial_profit_percentage_l2421_242136


namespace cube_sum_theorem_l2421_242195

theorem cube_sum_theorem (x y k c : ℝ) (h1 : x^3 * y^3 = k) (h2 : 1 / x^3 + 1 / y^3 = c) :
  ∃ m : ℝ, m = x + y ∧ (x + y)^3 = c * k + 3 * (k^(1/3)) * m :=
sorry

end cube_sum_theorem_l2421_242195


namespace arithmetic_operations_l2421_242173

theorem arithmetic_operations :
  ((-16) + (-29) = -45) ∧
  ((-10) - 7 = -17) ∧
  (5 * (-2) = -10) ∧
  ((-16) / (-2) = 8) := by
  sorry

end arithmetic_operations_l2421_242173


namespace unique_prime_factorization_and_sum_l2421_242101

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem unique_prime_factorization_and_sum (q r s p1 p2 p3 : ℕ) : 
  (q * r * s = 2206 ∧ 
   is_prime q ∧ is_prime r ∧ is_prime s ∧ 
   q ≠ r ∧ q ≠ s ∧ r ≠ s) →
  (p1 + p2 + p3 = q + r + s + 1 ∧ 
   is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ 
   p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3) →
  ((q = 2 ∧ r = 3 ∧ s = 367) ∨ (q = 2 ∧ r = 367 ∧ s = 3) ∨ (q = 3 ∧ r = 2 ∧ s = 367) ∨ 
   (q = 3 ∧ r = 367 ∧ s = 2) ∧ (q = 367 ∧ r = 2 ∧ s = 3) ∨ (q = 367 ∧ r = 3 ∧ s = 2)) ∧
  (p1 = 2 ∧ p2 = 3 ∧ p3 = 367) :=
by sorry

end unique_prime_factorization_and_sum_l2421_242101


namespace prism_volume_l2421_242120

/-- The volume of a right rectangular prism with face areas 30, 40, and 60 is 120√5 -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 40) (h3 : b * c = 60) :
  a * b * c = 120 * Real.sqrt 5 := by
  sorry

end prism_volume_l2421_242120


namespace hyperbola_equation_l2421_242194

/-- Given a hyperbola with eccentricity √5 and one vertex at (1, 0), 
    prove that its equation is x^2 - y^2/4 = 1 -/
theorem hyperbola_equation (e : ℝ) (v : ℝ × ℝ) :
  e = Real.sqrt 5 →
  v = (1, 0) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 ↔ x^2 - y^2/4 = 1) :=
sorry

end hyperbola_equation_l2421_242194


namespace complex_equation_result_l2421_242183

theorem complex_equation_result (a b : ℝ) (h : (a + 4 * Complex.I) * Complex.I = b + Complex.I) : a - b = -5 := by
  sorry

end complex_equation_result_l2421_242183


namespace system_solution_l2421_242145

theorem system_solution :
  let S := {(x, y, z, t) : ℕ × ℕ × ℕ × ℕ | 
    x + y + z + t = 5 ∧ 
    x + 2*y + 5*z + 10*t = 17}
  S = {(1, 3, 0, 1), (2, 0, 3, 0)} := by
  sorry

end system_solution_l2421_242145


namespace sum_of_roots_quadratic_l2421_242142

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 3 = 0) → (x₂^2 - 2*x₂ - 3 = 0) → (x₁ + x₂ = 2) := by
  sorry

end sum_of_roots_quadratic_l2421_242142


namespace vegetable_options_count_l2421_242123

/-- The number of cheese options available -/
def cheese_options : ℕ := 3

/-- The number of meat options available -/
def meat_options : ℕ := 4

/-- The total number of topping combinations -/
def total_combinations : ℕ := 57

/-- Calculates the number of topping combinations given the number of vegetable options -/
def calculate_combinations (veg_options : ℕ) : ℕ :=
  cheese_options * meat_options * veg_options - 
  cheese_options * (veg_options - 1) + 
  cheese_options

/-- Theorem stating that there are 5 vegetable options -/
theorem vegetable_options_count : 
  ∃ (veg_options : ℕ), veg_options = 5 ∧ calculate_combinations veg_options = total_combinations :=
sorry

end vegetable_options_count_l2421_242123


namespace power_equality_l2421_242158

theorem power_equality : (4 : ℝ) ^ 10 = 16 ^ 5 := by sorry

end power_equality_l2421_242158


namespace hyperbola_symmetric_points_parabola_midpoint_l2421_242164

/-- Given a hyperbola, two symmetric points on it, and their midpoint on a parabola, prove the possible values of m -/
theorem hyperbola_symmetric_points_parabola_midpoint (m : ℝ) : 
  (∃ (M N : ℝ × ℝ),
    -- M and N are on the hyperbola
    (M.1^2 - M.2^2/3 = 1) ∧ (N.1^2 - N.2^2/3 = 1) ∧
    -- M and N are symmetric about y = x + m
    (M.2 + N.2 = M.1 + N.1 + 2*m) ∧
    -- The midpoint of MN is on the parabola y^2 = 18x
    (((M.2 + N.2)/2)^2 = 18 * ((M.1 + N.1)/2))) →
  (m = 0 ∨ m = -8) :=
sorry

end hyperbola_symmetric_points_parabola_midpoint_l2421_242164


namespace bridge_toll_base_cost_l2421_242150

/-- Represents the toll calculation for a bridge -/
structure BridgeToll where
  base_cost : ℝ
  axle_cost : ℝ

/-- Calculates the toll for a given number of axles -/
def calc_toll (bt : BridgeToll) (axles : ℕ) : ℝ :=
  bt.base_cost + bt.axle_cost * (axles - 2)

/-- Represents a truck with a specific number of wheels and axles -/
structure Truck where
  total_wheels : ℕ
  front_axle_wheels : ℕ
  other_axle_wheels : ℕ

/-- Calculates the number of axles for a truck -/
def calc_axles (t : Truck) : ℕ :=
  1 + (t.total_wheels - t.front_axle_wheels) / t.other_axle_wheels

theorem bridge_toll_base_cost :
  ∃ (bt : BridgeToll),
    bt.axle_cost = 0.5 ∧
    let truck := Truck.mk 18 2 4
    let axles := calc_axles truck
    calc_toll bt axles = 5 ∧
    bt.base_cost = 3.5 := by
  sorry

end bridge_toll_base_cost_l2421_242150


namespace fifth_element_row_20_l2421_242172

-- Define Pascal's triangle
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.choose n k

-- Theorem statement
theorem fifth_element_row_20 : pascal 20 4 = 4845 := by
  sorry

end fifth_element_row_20_l2421_242172


namespace alcohol_solution_percentage_l2421_242122

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_alcohol : ℝ) : 
  initial_volume = 6 → 
  initial_percentage = 0.3 → 
  added_alcohol = 2.4 → 
  let final_volume := initial_volume + added_alcohol
  let final_alcohol := initial_volume * initial_percentage + added_alcohol
  final_alcohol / final_volume = 0.5 := by
sorry

end alcohol_solution_percentage_l2421_242122


namespace inscribed_rectangle_area_coefficient_l2421_242103

/-- Triangle XYZ with side lengths -/
structure Triangle where
  xy : ℝ
  yz : ℝ
  xz : ℝ

/-- Rectangle MNPQ inscribed in Triangle XYZ -/
structure InscribedRectangle where
  triangle : Triangle
  ω : ℝ  -- side length MN

/-- Area of rectangle MNPQ as a function of ω -/
def rectangleArea (rect : InscribedRectangle) : ℝ → ℝ :=
  fun ω => a * ω - b * ω^2
  where
    a : ℝ := sorry
    b : ℝ := sorry

/-- Theorem statement -/
theorem inscribed_rectangle_area_coefficient
  (t : Triangle)
  (h1 : t.xy = 15)
  (h2 : t.yz = 20)
  (h3 : t.xz = 13) :
  ∃ (rect : InscribedRectangle),
    rect.triangle = t ∧
    ∃ (a b : ℝ),
      (∀ ω, rectangleArea rect ω = a * ω - b * ω^2) ∧
      b = 9 / 25 :=
sorry

end inscribed_rectangle_area_coefficient_l2421_242103


namespace yoki_cans_collected_l2421_242106

/-- Given the conditions of the can collection problem, prove that Yoki picked up 9 cans. -/
theorem yoki_cans_collected (total_cans ladonna_cans prikya_cans avi_cans yoki_cans : ℕ) : 
  total_cans = 85 →
  ladonna_cans = 25 →
  prikya_cans = 2 * ladonna_cans - 3 →
  avi_cans = 8 / 2 →
  yoki_cans = total_cans - (ladonna_cans + prikya_cans + avi_cans) →
  yoki_cans = 9 := by
sorry

end yoki_cans_collected_l2421_242106


namespace total_cost_ratio_l2421_242137

-- Define the cost of shorts
variable (x : ℝ)

-- Define the costs of other items based on the given conditions
def cost_tshirt : ℝ := x
def cost_boots : ℝ := 4 * x
def cost_shinguards : ℝ := 2 * x

-- State the theorem
theorem total_cost_ratio : 
  (x + cost_tshirt x + cost_boots x + cost_shinguards x) / x = 8 := by
  sorry

end total_cost_ratio_l2421_242137


namespace gcd_of_136_and_1275_l2421_242168

theorem gcd_of_136_and_1275 : Nat.gcd 136 1275 = 17 := by
  sorry

end gcd_of_136_and_1275_l2421_242168


namespace infinitely_many_divisible_l2421_242119

/-- The prime counting function -/
def prime_counting (n : ℕ) : ℕ := sorry

/-- π(n) is non-decreasing -/
axiom prime_counting_nondecreasing : ∀ m n : ℕ, m ≤ n → prime_counting m ≤ prime_counting n

/-- The set of integers n such that π(n) divides n -/
def divisible_set : Set ℕ := {n : ℕ | prime_counting n ∣ n}

/-- There are infinitely many integers n such that π(n) divides n -/
theorem infinitely_many_divisible : Set.Infinite divisible_set := by sorry

end infinitely_many_divisible_l2421_242119


namespace task_completion_probability_l2421_242104

theorem task_completion_probability (p1 p2 p3 : ℚ) 
  (h1 : p1 = 2/3) (h2 : p2 = 3/5) (h3 : p3 = 4/7) :
  p1 * (1 - p2) * p3 = 16/105 := by
  sorry

end task_completion_probability_l2421_242104


namespace taylor_series_expansion_of_f_l2421_242127

def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 2

def taylor_expansion (x : ℝ) : ℝ := -12 + 16*(x + 1) - 7*(x + 1)^2 + (x + 1)^3

theorem taylor_series_expansion_of_f :
  ∀ x : ℝ, f x = taylor_expansion x := by
  sorry

end taylor_series_expansion_of_f_l2421_242127


namespace system_solution_l2421_242140

theorem system_solution : 
  ∃! (x y : ℚ), (2010 * x - 2011 * y = 2009) ∧ (2009 * x - 2008 * y = 2010) ∧ x = 2 ∧ y = 1 := by
  sorry

end system_solution_l2421_242140


namespace triangle_problem_l2421_242100

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  a = 3 * Real.sqrt 2 →
  c = Real.sqrt 3 →
  Real.cos C = 2 * Real.sqrt 2 / 3 →
  b < a →
  -- Conclusion
  Real.sin A = Real.sqrt 6 / 3 ∧ b = 3 := by
sorry

end triangle_problem_l2421_242100


namespace g_sum_property_l2421_242198

def g (x : ℝ) : ℝ := 2 * x^8 + 3 * x^6 - 4 * x^4 + 5

theorem g_sum_property : g 5 = 7 → g 5 + g (-5) = 14 := by sorry

end g_sum_property_l2421_242198


namespace total_profit_calculation_l2421_242116

theorem total_profit_calculation (x_investment y_investment z_investment : ℕ)
  (x_months y_months z_months : ℕ) (z_profit : ℕ) :
  x_investment = 36000 →
  y_investment = 42000 →
  z_investment = 48000 →
  x_months = 12 →
  y_months = 12 →
  z_months = 8 →
  z_profit = 4096 →
  (z_investment * z_months * 14080 = z_profit * (x_investment * x_months + y_investment * y_months + z_investment * z_months)) :=
by
  sorry

#check total_profit_calculation

end total_profit_calculation_l2421_242116


namespace pushup_progression_l2421_242193

/-- 
Given a person who does push-ups 3 times a week, increasing by 5 each time,
prove that if the total for the week is 45, then the number of push-ups on the first day is 10.
-/
theorem pushup_progression (first_day : ℕ) : 
  first_day + (first_day + 5) + (first_day + 10) = 45 → first_day = 10 := by
  sorry

end pushup_progression_l2421_242193


namespace preimage_of_neg_three_two_l2421_242187

def f (x y : ℝ) : ℝ × ℝ := (x * y, x + y)

theorem preimage_of_neg_three_two :
  {p : ℝ × ℝ | f p.1 p.2 = (-3, 2)} = {(3, -1), (-1, 3)} := by
  sorry

end preimage_of_neg_three_two_l2421_242187


namespace proportion_fourth_term_l2421_242121

theorem proportion_fourth_term (x y : ℝ) : 
  (0.6 : ℝ) / x = 5 / y → x = 0.96 → y = 8 := by
  sorry

end proportion_fourth_term_l2421_242121


namespace incorrect_combination_not_equivalent_l2421_242197

-- Define the polynomial
def original_polynomial (a b : ℚ) : ℚ := 2 * a * b - 4 * a^2 - 5 * a * b + 9 * a^2

-- Define the incorrect combination
def incorrect_combination (a b : ℚ) : ℚ := (2 * a * b - 5 * a * b) - (4 * a^2 + 9 * a^2)

-- Theorem stating that the incorrect combination is not equivalent to the original polynomial
theorem incorrect_combination_not_equivalent :
  ∃ a b : ℚ, original_polynomial a b ≠ incorrect_combination a b :=
sorry

end incorrect_combination_not_equivalent_l2421_242197


namespace smallest_N_with_301_l2421_242131

/-- The function that generates the concatenated string for a given N -/
def generateString (N : ℕ) : String := sorry

/-- The predicate that checks if "301" appears in a string -/
def contains301 (s : String) : Prop := sorry

/-- The theorem stating that 38 is the smallest N that satisfies the condition -/
theorem smallest_N_with_301 : 
  (∀ n < 38, ¬ contains301 (generateString n)) ∧ 
  contains301 (generateString 38) := by sorry

end smallest_N_with_301_l2421_242131


namespace product_of_three_numbers_l2421_242167

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 210)
  (rel_b : 5 * a = b - 11)
  (rel_c : 5 * a = c + 11) :
  a * b * c = 168504 := by
sorry

end product_of_three_numbers_l2421_242167


namespace gcd_360_150_l2421_242118

theorem gcd_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end gcd_360_150_l2421_242118


namespace money_distribution_l2421_242124

theorem money_distribution (a b c : ℕ) 
  (total : a + b + c = 450)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 350) :
  c = 100 := by
  sorry

end money_distribution_l2421_242124


namespace sqrt_expression_equality_l2421_242134

theorem sqrt_expression_equality (t : ℝ) : 
  Real.sqrt (t^6 + t^4 + t^2) = |t| * Real.sqrt (t^4 + t^2 + 1) := by
  sorry

end sqrt_expression_equality_l2421_242134


namespace complex_magnitude_product_l2421_242161

theorem complex_magnitude_product : 
  Complex.abs ((5 * Real.sqrt 2 - 5 * Complex.I) * (2 * Real.sqrt 3 + 6 * Complex.I)) = 60 := by
  sorry

end complex_magnitude_product_l2421_242161


namespace expression_evaluation_l2421_242176

theorem expression_evaluation : (980^2 : ℚ) / (210^2 - 206^2) = 577.5 := by
  sorry

end expression_evaluation_l2421_242176


namespace gina_textbooks_l2421_242138

/-- Calculates the number of textbooks Gina needs to buy given her college expenses. -/
def calculate_textbooks (credits : ℕ) (credit_cost : ℕ) (facilities_fee : ℕ) (textbook_cost : ℕ) (total_spending : ℕ) : ℕ :=
  let credit_total := credits * credit_cost
  let non_textbook_cost := credit_total + facilities_fee
  let textbook_budget := total_spending - non_textbook_cost
  textbook_budget / textbook_cost

theorem gina_textbooks :
  calculate_textbooks 14 450 200 120 7100 = 5 := by
  sorry

end gina_textbooks_l2421_242138


namespace hyperbola_asymptotes_l2421_242151

/-- The asymptotic lines of a hyperbola with equation x^2 - y^2/9 = 1 are y = ±3x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 - y^2/9 = 1) → (∃ k : ℝ, k = 3 ∨ k = -3) → (y = k*x) := by
  sorry

end hyperbola_asymptotes_l2421_242151


namespace bacteria_growth_calculation_l2421_242130

/-- Given an original bacteria count and a current bacteria count, 
    calculate the increase in bacteria. -/
def bacteria_increase (original current : ℕ) : ℕ :=
  current - original

/-- Theorem stating that the increase in bacteria from 600 to 8917 is 8317. -/
theorem bacteria_growth_calculation :
  bacteria_increase 600 8917 = 8317 := by
  sorry

end bacteria_growth_calculation_l2421_242130


namespace pairwise_ratio_sum_bound_l2421_242148

theorem pairwise_ratio_sum_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 : ℝ) / 2 := by
  sorry

end pairwise_ratio_sum_bound_l2421_242148


namespace complex_number_quadrant_l2421_242181

theorem complex_number_quadrant : 
  let z : ℂ := (2 + 3*I) / (1 + 2*I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end complex_number_quadrant_l2421_242181


namespace max_pairs_sum_l2421_242128

theorem max_pairs_sum (n : ℕ) (h : n = 3011) : 
  (∃ (k : ℕ) (pairs : List (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    pairs.length = k) ∧
  (∀ (m : ℕ) (pairs : List (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
    pairs.length = m →
    m ≤ k) →
  k = 1204 := by
sorry

end max_pairs_sum_l2421_242128


namespace xy_sum_l2421_242154

theorem xy_sum (x y : ℤ) (h : 2*x*y + x + y = 83) : x + y = 83 ∨ x + y = -85 := by
  sorry

end xy_sum_l2421_242154


namespace problem_solution_l2421_242117

def p (m : ℝ) : Prop := ∀ x, 2*x - 5 > 0 → x > m

def q (m : ℝ) : Prop := ∃ a b, a ≠ 0 ∧ b ≠ 0 ∧ 
  ∀ x y, x^2/(m-1) + y^2/(2-m) = 1 ↔ (x/a)^2 - (y/b)^2 = 1

theorem problem_solution (m : ℝ) : 
  (p m ∧ q m → m < 1 ∨ (2 < m ∧ m ≤ 5/2)) ∧
  (¬(p m ∧ q m) ∧ (p m ∨ q m) → (1 ≤ m ∧ m ≤ 2) ∨ m > 5/2) :=
sorry

end problem_solution_l2421_242117


namespace salt_mixture_concentration_l2421_242163

/-- Given two salt solutions and their volumes, calculate the salt concentration of the mixture -/
theorem salt_mixture_concentration 
  (vol1 : ℝ) (conc1 : ℝ) (vol2 : ℝ) (conc2 : ℝ) 
  (h1 : vol1 = 600) 
  (h2 : conc1 = 0.03) 
  (h3 : vol2 = 400) 
  (h4 : conc2 = 0.12) 
  (h5 : vol1 + vol2 = 1000) :
  (vol1 * conc1 + vol2 * conc2) / (vol1 + vol2) = 0.066 := by
sorry

end salt_mixture_concentration_l2421_242163


namespace completing_square_correct_l2421_242110

-- Define the original equation
def original_equation (x : ℝ) : Prop := x^2 - 4*x - 22 = 0

-- Define the result of completing the square
def completed_square_result (x : ℝ) : Prop := (x - 2)^2 = 26

-- Theorem statement
theorem completing_square_correct :
  ∀ x : ℝ, original_equation x ↔ completed_square_result x :=
by sorry

end completing_square_correct_l2421_242110


namespace tony_between_paul_and_rochelle_l2421_242149

-- Define the set of people
inductive Person : Type
  | Paul : Person
  | Quincy : Person
  | Rochelle : Person
  | Surinder : Person
  | Tony : Person

-- Define the seating arrangement as a function from Person to ℕ
def SeatingArrangement := Person → ℕ

-- Define the conditions of the seating arrangement
def ValidSeatingArrangement (s : SeatingArrangement) : Prop :=
  -- Condition 1: All seats are distinct
  (∀ p q : Person, p ≠ q → s p ≠ s q) ∧
  -- Condition 2: Seats are consecutive around a circular table
  (∀ p : Person, s p < 5) ∧
  -- Condition 3: Quincy sits between Paul and Surinder
  ((s Person.Quincy = (s Person.Paul + 1) % 5 ∧ s Person.Quincy = (s Person.Surinder + 4) % 5) ∨
   (s Person.Quincy = (s Person.Paul + 4) % 5 ∧ s Person.Quincy = (s Person.Surinder + 1) % 5)) ∧
  -- Condition 4: Tony is not beside Surinder
  (s Person.Tony ≠ (s Person.Surinder + 1) % 5 ∧ s Person.Tony ≠ (s Person.Surinder + 4) % 5)

-- Theorem: In any valid seating arrangement, Paul and Rochelle must be sitting on either side of Tony
theorem tony_between_paul_and_rochelle (s : SeatingArrangement) 
  (h : ValidSeatingArrangement s) : 
  (s Person.Tony = (s Person.Paul + 1) % 5 ∧ s Person.Tony = (s Person.Rochelle + 4) % 5) ∨
  (s Person.Tony = (s Person.Paul + 4) % 5 ∧ s Person.Tony = (s Person.Rochelle + 1) % 5) :=
sorry

end tony_between_paul_and_rochelle_l2421_242149


namespace wedding_ring_cost_l2421_242111

/-- Proves that the cost of the first wedding ring is $10,000 given the problem conditions --/
theorem wedding_ring_cost (first_ring_cost : ℝ) : 
  (3 * first_ring_cost - first_ring_cost / 2 = 25000) → 
  first_ring_cost = 10000 := by
  sorry

#check wedding_ring_cost

end wedding_ring_cost_l2421_242111


namespace parabola_sum_l2421_242190

/-- A parabola with equation y = px^2 + qx + r, vertex (3, 7), vertical axis of symmetry, and containing the point (0, 4) -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := 7
  point_x : ℝ := 0
  point_y : ℝ := 4
  eq_at_point : p * point_x^2 + q * point_x + r = point_y
  vertex_form : ∀ x y, y = p * (x - vertex_x)^2 + vertex_y

theorem parabola_sum (par : Parabola) : par.p + par.q + par.r = 13/3 := by
  sorry

end parabola_sum_l2421_242190


namespace correct_number_probability_l2421_242139

-- Define the number of options for the first three digits
def first_three_options : ℕ := 3

-- Define the number of digits used in the last five digits
def last_five_digits : ℕ := 5

-- Theorem statement
theorem correct_number_probability :
  (1 : ℚ) / (first_three_options * Nat.factorial last_five_digits) = (1 : ℚ) / 360 :=
by sorry

end correct_number_probability_l2421_242139


namespace bank_deposit_calculation_l2421_242186

/-- Calculates the total amount of principal and interest for a fixed deposit -/
def totalAmount (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate * years)

/-- Calculates the amount left after paying interest tax -/
def amountAfterTax (totalAmount : ℝ) (principal : ℝ) (taxRate : ℝ) : ℝ :=
  totalAmount - (totalAmount - principal) * taxRate

theorem bank_deposit_calculation :
  let principal : ℝ := 1000
  let rate : ℝ := 0.0225
  let years : ℝ := 1
  let taxRate : ℝ := 0.20
  let total := totalAmount principal rate years
  let afterTax := amountAfterTax total principal taxRate
  total = 1022.5 ∧ afterTax = 1018 := by sorry

end bank_deposit_calculation_l2421_242186


namespace simplify_fraction_l2421_242174

theorem simplify_fraction (a : ℝ) (ha : a ≠ 0) (ha2 : a ≠ 2) :
  (a^2 - 6*a + 9) / (a^2 - 2*a) / (1 - 1/(a - 2)) = (a - 3) / a :=
by sorry

end simplify_fraction_l2421_242174


namespace function_composition_result_l2421_242179

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem function_composition_result : f (g (Real.sqrt 3)) = 1 := by
  sorry

end function_composition_result_l2421_242179


namespace function_inequality_implies_t_bound_l2421_242171

theorem function_inequality_implies_t_bound (t : ℝ) : 
  (∀ x : ℝ, (Real.exp (2 * x) - t) ≥ (t * Real.exp x - 1)) → 
  t ≤ 2 * Real.sqrt 2 - 2 := by
  sorry

end function_inequality_implies_t_bound_l2421_242171


namespace oranges_per_box_l2421_242184

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) 
  (h1 : total_oranges = 42) (h2 : num_boxes = 7) :
  total_oranges / num_boxes = 6 := by
  sorry

end oranges_per_box_l2421_242184


namespace binomial_12_6_l2421_242102

theorem binomial_12_6 : Nat.choose 12 6 = 924 := by sorry

end binomial_12_6_l2421_242102


namespace flatrate_calculation_l2421_242105

/-- Represents the tutoring session details and pricing -/
structure TutoringSession where
  flatRate : ℕ
  perMinuteRate : ℕ
  durationMinutes : ℕ
  totalAmount : ℕ

/-- Theorem stating the flat rate for the given tutoring session -/
theorem flatrate_calculation (session : TutoringSession)
  (h1 : session.perMinuteRate = 7)
  (h2 : session.durationMinutes = 18)
  (h3 : session.totalAmount = 146)
  (h4 : session.totalAmount = session.flatRate + session.perMinuteRate * session.durationMinutes) :
  session.flatRate = 20 := by
  sorry

#check flatrate_calculation

end flatrate_calculation_l2421_242105


namespace correct_stratified_sample_l2421_242177

/-- Represents the number of students in each grade --/
structure GradePopulation where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Represents the number of students to be sampled from each grade --/
structure SampleSize where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Calculates the stratified sample size for each grade --/
def stratifiedSample (pop : GradePopulation) (totalSample : ℕ) : SampleSize :=
  let totalPop := pop.grade10 + pop.grade11 + pop.grade12
  { grade10 := totalSample * pop.grade10 / totalPop,
    grade11 := totalSample * pop.grade11 / totalPop,
    grade12 := totalSample * pop.grade12 / totalPop }

/-- Theorem: The stratified sample for the given population and sample size is correct --/
theorem correct_stratified_sample :
  let pop := GradePopulation.mk 600 800 400
  let sample := stratifiedSample pop 18
  sample = SampleSize.mk 6 8 4 := by sorry

end correct_stratified_sample_l2421_242177


namespace largest_increase_2018_2019_l2421_242178

def students : Fin 6 → ℕ
  | 0 => 110  -- 2015
  | 1 => 125  -- 2016
  | 2 => 130  -- 2017
  | 3 => 140  -- 2018
  | 4 => 160  -- 2019
  | 5 => 165  -- 2020

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreaseYears : Fin 5 := sorry

theorem largest_increase_2018_2019 :
  largestIncreaseYears = 3 ∧
  ∀ i : Fin 5, percentageIncrease (students i) (students (i + 1)) ≤
    percentageIncrease (students 3) (students 4) :=
by sorry

end largest_increase_2018_2019_l2421_242178


namespace product_and_multiple_l2421_242175

theorem product_and_multiple : ∃ x : ℕ, x = 320 * 6 ∧ x * 7 = 420 → x = 1920 := by
  sorry

end product_and_multiple_l2421_242175


namespace square_circle_radius_l2421_242199

/-- Given a square with a circumscribed circle, if the sum of the lengths of all sides
    of the square equals the area of the circumscribed circle, then the radius of the
    circle is 4√2/π. -/
theorem square_circle_radius (s : ℝ) (r : ℝ) (h : s > 0) (h' : r > 0) :
  4 * s = π * r^2 → r = 4 * Real.sqrt 2 / π :=
by sorry

end square_circle_radius_l2421_242199


namespace original_number_is_27_l2421_242153

theorem original_number_is_27 :
  ∃ (n : ℕ), 
    (Odd (3 * n)) ∧ 
    (∃ (k : ℕ), k > 1 ∧ (3 * n) % k = 0) ∧ 
    (4 * n = 108) ∧
    n = 27 := by
  sorry

end original_number_is_27_l2421_242153


namespace prob_different_grades_is_four_fifths_l2421_242162

/-- Represents the number of students in each grade --/
structure GradeDistribution where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Calculates the probability of selecting two students from different grades --/
def probabilityDifferentGrades (dist : GradeDistribution) : ℚ :=
  4/5

/-- Theorem stating that the probability of selecting two students from different grades is 4/5 --/
theorem prob_different_grades_is_four_fifths (dist : GradeDistribution) 
  (h1 : dist.grade10 = 180)
  (h2 : dist.grade11 = 180)
  (h3 : dist.grade12 = 90) :
  probabilityDifferentGrades dist = 4/5 := by
  sorry

#check prob_different_grades_is_four_fifths

end prob_different_grades_is_four_fifths_l2421_242162


namespace exists_non_prime_power_plus_a_l2421_242188

theorem exists_non_prime_power_plus_a (a : ℕ) (ha : a > 1) :
  ∃ n : ℕ, ¬ Nat.Prime (2^(2^n) + a) := by
  sorry

end exists_non_prime_power_plus_a_l2421_242188


namespace two_integers_sum_l2421_242115

theorem two_integers_sum (x y : ℕ+) : 
  x - y = 4 → x * y = 192 → x + y = 28 := by sorry

end two_integers_sum_l2421_242115


namespace opera_house_earnings_correct_l2421_242108

/-- Calculates the earnings of an opera house for a single show. -/
def opera_house_earnings (rows : ℕ) (seats_per_row : ℕ) (ticket_price : ℕ) (percent_empty : ℕ) : ℕ :=
  let total_seats := rows * seats_per_row
  let occupied_seats := total_seats - (total_seats * percent_empty / 100)
  occupied_seats * ticket_price

/-- Theorem stating that the opera house earnings for the given conditions equal $12000. -/
theorem opera_house_earnings_correct : opera_house_earnings 150 10 10 20 = 12000 := by
  sorry

end opera_house_earnings_correct_l2421_242108


namespace selling_price_is_200_l2421_242146

/-- Calculates the selling price per acre given the initial purchase details and profit --/
def selling_price_per_acre (total_acres : ℕ) (purchase_price_per_acre : ℕ) (profit : ℕ) : ℕ :=
  let total_cost := total_acres * purchase_price_per_acre
  let acres_sold := total_acres / 2
  let total_revenue := total_cost + profit
  total_revenue / acres_sold

/-- Proves that the selling price per acre is $200 given the problem conditions --/
theorem selling_price_is_200 :
  selling_price_per_acre 200 70 6000 = 200 := by
  sorry

end selling_price_is_200_l2421_242146


namespace goods_train_passing_time_l2421_242189

/-- The time taken for a goods train to pass a man in an opposite moving train -/
theorem goods_train_passing_time
  (man_train_speed : ℝ)
  (goods_train_speed : ℝ)
  (goods_train_length : ℝ)
  (h1 : man_train_speed = 55)
  (h2 : goods_train_speed = 60.2)
  (h3 : goods_train_length = 320) :
  (goods_train_length / ((man_train_speed + goods_train_speed) * (1000 / 3600))) = 10 := by
  sorry


end goods_train_passing_time_l2421_242189


namespace functional_equation_solution_l2421_242152

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → f 1 = 2 → f (-2) = 2 := by
  sorry

end functional_equation_solution_l2421_242152


namespace sum_of_numbers_l2421_242109

theorem sum_of_numbers : 1324 + 2431 + 3142 + 4213 + 1234 = 12344 := by
  sorry

end sum_of_numbers_l2421_242109


namespace max_perfect_squares_l2421_242159

/-- The sequence (a_n) defined recursively -/
def a : ℕ → ℕ → ℕ
  | m, 0 => m
  | m, n + 1 => (a m n)^5 + 487

/-- Proposition: m = 9 is the unique positive integer that maximizes perfect squares in the sequence -/
theorem max_perfect_squares (m : ℕ) : m > 0 → (∀ k : ℕ, k > 0 → (∀ n : ℕ, ∃ i : ℕ, i ≤ n ∧ ∃ j : ℕ, a k i = j^2) → 
  (∀ n : ℕ, ∃ i : ℕ, i ≤ n ∧ ∃ j : ℕ, a m i = j^2)) → m = 9 :=
by sorry

end max_perfect_squares_l2421_242159


namespace tan_zero_degrees_l2421_242125

theorem tan_zero_degrees : Real.tan 0 = 0 := by
  sorry

end tan_zero_degrees_l2421_242125


namespace reader_group_size_l2421_242133

theorem reader_group_size (S L B : ℕ) (hS : S = 180) (hL : L = 88) (hB : B = 18) :
  S + L - B = 250 := by
  sorry

end reader_group_size_l2421_242133


namespace weekend_classes_count_l2421_242165

/-- The number of beginning diving classes offered on each day of the weekend -/
def weekend_classes : ℕ := 4

/-- The number of beginning diving classes offered on weekdays -/
def weekday_classes : ℕ := 2

/-- The number of people that can be accommodated in each class -/
def class_capacity : ℕ := 5

/-- The total number of people that can take classes in 3 weeks -/
def total_people : ℕ := 270

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

/-- The number of weeks considered -/
def weeks : ℕ := 3

theorem weekend_classes_count :
  weekend_classes * class_capacity * weekend_days_per_week * weeks +
  weekday_classes * class_capacity * weekdays_per_week * weeks = total_people :=
by sorry

end weekend_classes_count_l2421_242165


namespace oatmeal_cookies_divisible_by_containers_l2421_242166

/-- The number of chocolate chip cookies Kiara baked -/
def chocolate_chip_cookies : ℕ := 48

/-- The number of containers Kiara wants to use -/
def num_containers : ℕ := 6

/-- The number of oatmeal cookies Kiara baked -/
def oatmeal_cookies : ℕ := sorry

/-- Theorem stating that the number of oatmeal cookies must be divisible by the number of containers -/
theorem oatmeal_cookies_divisible_by_containers :
  oatmeal_cookies % num_containers = 0 :=
sorry

end oatmeal_cookies_divisible_by_containers_l2421_242166


namespace train_length_is_600_l2421_242160

/-- The length of the train in meters -/
def train_length : ℝ := 600

/-- The time it takes for the train to cross a tree, in seconds -/
def time_to_cross_tree : ℝ := 60

/-- The time it takes for the train to pass a platform, in seconds -/
def time_to_pass_platform : ℝ := 105

/-- The length of the platform, in meters -/
def platform_length : ℝ := 450

/-- Theorem stating that the train length is 600 meters -/
theorem train_length_is_600 :
  train_length = (time_to_pass_platform * platform_length) / (time_to_pass_platform - time_to_cross_tree) :=
by sorry

end train_length_is_600_l2421_242160


namespace center_coordinate_sum_l2421_242147

/-- Given two points that are endpoints of a diameter of a circle,
    prove that the sum of the coordinates of the center is -3. -/
theorem center_coordinate_sum (p1 p2 : ℝ × ℝ) : 
  p1 = (5, -7) → p2 = (-7, 3) → 
  (∃ (c : ℝ × ℝ), c.1 + c.2 = -3 ∧ 
    c = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)) := by
  sorry

end center_coordinate_sum_l2421_242147


namespace line_equation_proof_l2421_242191

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The projection of one point onto a line -/
def Point.projection (p : Point) (l : Line) : Point :=
  sorry

theorem line_equation_proof (A : Point) (P : Point) (l : Line) :
  A.x = 1 ∧ A.y = 2 ∧ P.x = -1 ∧ P.y = 4 ∧ P = A.projection l →
  l.a = 1 ∧ l.b = -1 ∧ l.c = 5 :=
sorry

end line_equation_proof_l2421_242191


namespace problem_solution_l2421_242156

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : ((x * y) / 7)^3 = x^2) (h2 : ((x * y) / 7)^3 = y^3) : 
  x = 7 ∧ y = 7^(2/3) := by
  sorry

end problem_solution_l2421_242156


namespace fraction_subtraction_l2421_242107

theorem fraction_subtraction : 
  (3 + 6 + 9) / (2 + 5 + 8) - 4 * (1 + 2 + 3) / (5 + 10 + 15) = 2 / 5 := by
  sorry

end fraction_subtraction_l2421_242107


namespace jim_gave_eight_sets_to_brother_l2421_242129

/-- The number of trading cards in one set -/
def cards_per_set : ℕ := 13

/-- The number of sets Jim gave to his sister -/
def sets_to_sister : ℕ := 5

/-- The number of sets Jim gave to his friend -/
def sets_to_friend : ℕ := 2

/-- The total number of trading cards Jim had initially -/
def initial_cards : ℕ := 365

/-- The total number of trading cards Jim gave away -/
def total_given_away : ℕ := 195

/-- The number of sets Jim gave to his brother -/
def sets_to_brother : ℕ := (total_given_away - (sets_to_sister + sets_to_friend) * cards_per_set) / cards_per_set

theorem jim_gave_eight_sets_to_brother : sets_to_brother = 8 := by
  sorry

end jim_gave_eight_sets_to_brother_l2421_242129


namespace logarithmic_equation_solution_l2421_242126

theorem logarithmic_equation_solution :
  ∃ x : ℝ, x > 0 ∧ (Real.log x / Real.log 8 + 3 * Real.log (x^2) / Real.log 2 - Real.log x / Real.log 4 = 14) ∧
  x = 2^(12/5) := by
sorry

end logarithmic_equation_solution_l2421_242126


namespace max_sin_a_given_condition_l2421_242113

open Real

theorem max_sin_a_given_condition (a b : ℝ) :
  cos (a + b) + sin (a - b) = cos a + cos b →
  ∃ (max_sin_a : ℝ), (∀ x, sin x ≤ max_sin_a) ∧ (max_sin_a = 1) :=
sorry

end max_sin_a_given_condition_l2421_242113


namespace sum_of_fractions_geq_six_l2421_242144

theorem sum_of_fractions_geq_six (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / y + y / z + z / x + x / z + z / y + y / x ≥ 6 := by
  sorry

end sum_of_fractions_geq_six_l2421_242144


namespace cubic_equation_solutions_l2421_242192

theorem cubic_equation_solutions :
  let f : ℝ → ℝ := λ x => x^3 - 13*x - 12
  (f 4 = 0) ∧ (f (-1) = 0) ∧ (f (-3) = 0) ∧
  (∀ x : ℝ, f x = 0 → (x = 4 ∨ x = -1 ∨ x = -3)) :=
by sorry

end cubic_equation_solutions_l2421_242192


namespace larger_number_proof_l2421_242180

theorem larger_number_proof (a b : ℝ) : 
  a + b = 104 → 
  a^2 - b^2 = 208 → 
  max a b = 53 := by
sorry

end larger_number_proof_l2421_242180


namespace peter_has_25_candies_l2421_242196

/-- The number of candies each person has after sharing equally -/
def shared_candies : ℕ := 30

/-- The number of candies Mark has -/
def mark_candies : ℕ := 30

/-- The number of candies John has -/
def john_candies : ℕ := 35

/-- The number of candies Peter has -/
def peter_candies : ℕ := shared_candies * 3 - mark_candies - john_candies

theorem peter_has_25_candies : peter_candies = 25 := by
  sorry

end peter_has_25_candies_l2421_242196


namespace tangent_circle_radius_l2421_242155

/-- A circle with two parallel tangents and a third connecting tangent -/
structure TangentCircle where
  -- Radius of the circle
  r : ℝ
  -- Length of the first parallel tangent
  ab : ℝ
  -- Length of the second parallel tangent
  cd : ℝ
  -- Length of the connecting tangent
  ef : ℝ
  -- Condition that ab and cd are parallel tangents
  h_parallel : ab < cd
  -- Condition that ef is a tangent connecting ab and cd
  h_connecting : ef > ab ∧ ef < cd

/-- The theorem stating that for the given configuration, the radius is 2.5 -/
theorem tangent_circle_radius (c : TangentCircle)
    (h_ab : c.ab = 5)
    (h_cd : c.cd = 11)
    (h_ef : c.ef = 15) :
    c.r = 2.5 := by
  sorry

end tangent_circle_radius_l2421_242155


namespace arithmetic_sequence_length_l2421_242169

theorem arithmetic_sequence_length :
  ∀ (a₁ d n : ℤ),
    a₁ = -48 →
    d = 8 →
    a₁ + (n - 1) * d = 80 →
    n = 17 :=
by
  sorry

end arithmetic_sequence_length_l2421_242169


namespace tan_45_degrees_l2421_242182

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_l2421_242182


namespace initial_group_size_l2421_242170

/-- The number of initial persons in a group, given specific average age conditions. -/
theorem initial_group_size (initial_avg : ℝ) (new_persons : ℕ) (new_avg : ℝ) (final_avg : ℝ) :
  initial_avg = 16 →
  new_persons = 12 →
  new_avg = 15 →
  final_avg = 15.5 →
  ∃ n : ℕ, n * initial_avg + new_persons * new_avg = (n + new_persons) * final_avg ∧ n = 12 :=
by sorry

end initial_group_size_l2421_242170


namespace calculate_expression_l2421_242132

theorem calculate_expression : 150 * (150 - 5) - (150 * 150 - 7) = -743 := by
  sorry

end calculate_expression_l2421_242132


namespace john_text_messages_l2421_242157

theorem john_text_messages 
  (total_messages_per_day : ℕ) 
  (unintended_messages_per_week : ℕ) 
  (days_per_week : ℕ) 
  (h1 : total_messages_per_day = 55) 
  (h2 : unintended_messages_per_week = 245) 
  (h3 : days_per_week = 7) : 
  total_messages_per_day - (unintended_messages_per_week / days_per_week) = 20 := by
sorry

end john_text_messages_l2421_242157


namespace b_share_is_3000_l2421_242135

/-- Proves that B's share is 3000 when money is distributed in the proportion 6:3:5:4 and C gets 1000 more than D -/
theorem b_share_is_3000 (total : ℕ) (a b c d : ℕ) : 
  a + b + c + d = total →  -- Sum of all shares equals the total
  6 * b = 3 * a →          -- A:B proportion is 6:3
  5 * b = 5 * a →          -- B:C proportion is 3:5
  4 * b = 3 * d →          -- B:D proportion is 3:4
  c = d + 1000 →           -- C gets 1000 more than D
  b = 3000 := by
sorry

end b_share_is_3000_l2421_242135


namespace bank_account_difference_l2421_242185

theorem bank_account_difference (bob_amount jenna_amount phil_amount : ℝ) : 
  bob_amount = 60 →
  phil_amount = (1/3) * bob_amount →
  jenna_amount = 2 * phil_amount →
  bob_amount - jenna_amount = 20 := by
sorry

end bank_account_difference_l2421_242185


namespace rectangle_area_function_l2421_242112

/-- For a rectangle with area 10 and adjacent sides x and y, prove that y = 10/x --/
theorem rectangle_area_function (x y : ℝ) (h : x * y = 10) : y = 10 / x := by
  sorry

end rectangle_area_function_l2421_242112


namespace A_when_half_in_A_B_values_l2421_242141

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + 1 = 0}

theorem A_when_half_in_A (a : ℝ) (h : (1/2 : ℝ) ∈ A a) : 
  A a = {-(1/4), 1/2} := by sorry

def B : Set ℝ := {a : ℝ | ∃! x, x ∈ A a}

theorem B_values : B = {0, 1} := by sorry

end A_when_half_in_A_B_values_l2421_242141


namespace current_speed_l2421_242143

/-- Proves that the speed of the current is approximately 3 km/hr given the conditions -/
theorem current_speed (rowing_speed : ℝ) (distance : ℝ) (time : ℝ) :
  rowing_speed = 6 →
  distance = 80 →
  time = 31.99744020478362 →
  ∃ (current_speed : ℝ), 
    (abs (current_speed - 3) < 0.001) ∧ 
    (distance / time = rowing_speed / 3.6 + current_speed / 3.6) := by
  sorry


end current_speed_l2421_242143
