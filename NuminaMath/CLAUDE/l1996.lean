import Mathlib

namespace unique_element_in_S_l1996_199664

-- Define the set
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.log (p.1^3 + (1/3) * p.2^3 + 1/9) = Real.log p.1 + Real.log p.2}

-- Theorem statement
theorem unique_element_in_S : ∃! p : ℝ × ℝ, p ∈ S := by sorry

end unique_element_in_S_l1996_199664


namespace f_composition_value_l1996_199619

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then (1/2)^x
  else if 0 < x ∧ x < 1 then Real.log x / Real.log 4
  else 0  -- This case is added to make the function total

-- State the theorem
theorem f_composition_value : f (f 2) = -1 := by
  sorry

end f_composition_value_l1996_199619


namespace binomial_1300_2_l1996_199687

theorem binomial_1300_2 : Nat.choose 1300 2 = 844350 := by
  sorry

end binomial_1300_2_l1996_199687


namespace proposition_false_iff_a_less_than_neg_thirteen_half_l1996_199622

theorem proposition_false_iff_a_less_than_neg_thirteen_half :
  (∀ x ∈ Set.Icc 1 2, x^2 + a*x + 9 ≥ 0) = false ↔ a < -13/2 :=
sorry

end proposition_false_iff_a_less_than_neg_thirteen_half_l1996_199622


namespace smartphone_sale_price_l1996_199669

theorem smartphone_sale_price (initial_cost : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : 
  initial_cost = 300 →
  loss_percentage = 15 →
  selling_price = initial_cost - (loss_percentage / 100) * initial_cost →
  selling_price = 255 := by
sorry

end smartphone_sale_price_l1996_199669


namespace polynomial_equality_l1996_199698

theorem polynomial_equality (a b c : ℤ) :
  (∀ x : ℝ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) →
  (a = 3 ∨ a = 7) :=
by sorry

end polynomial_equality_l1996_199698


namespace selection_methods_count_l1996_199623

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of male athletes. -/
def num_males : ℕ := 4

/-- The number of female athletes. -/
def num_females : ℕ := 5

/-- The total number of athletes to be selected. -/
def num_selected : ℕ := 3

theorem selection_methods_count :
  (choose num_males 2 * choose num_females 1) + (choose num_males 1 * choose num_females 2) = 70 := by
  sorry

end selection_methods_count_l1996_199623


namespace simplify_expression_l1996_199678

theorem simplify_expression (a b : ℝ) :
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 := by sorry

end simplify_expression_l1996_199678


namespace gas_cost_per_gallon_l1996_199630

/-- Proves that the cost of gas per gallon is $4, given the conditions of Dan's car fuel efficiency and travel distance. -/
theorem gas_cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) : 
  miles_per_gallon = 32 → total_miles = 464 → total_cost = 58 → 
  (total_cost / (total_miles / miles_per_gallon)) = 4 := by
  sorry

/-- The actual cost of gas per gallon based on the given conditions. -/
def actual_gas_cost : ℝ := 4

#check gas_cost_per_gallon
#check actual_gas_cost

end gas_cost_per_gallon_l1996_199630


namespace exam_pass_percentage_l1996_199668

theorem exam_pass_percentage 
  (failed_hindi : Real) 
  (failed_english : Real) 
  (failed_both : Real) 
  (h1 : failed_hindi = 34)
  (h2 : failed_english = 44)
  (h3 : failed_both = 22) :
  100 - (failed_hindi + failed_english - failed_both) = 44 := by
  sorry

end exam_pass_percentage_l1996_199668


namespace intersection_M_N_l1996_199627

def M : Set ℝ := {-2, -1, 0, 1, 2}

def N : Set ℝ := {x | x < 0 ∨ x > 3}

theorem intersection_M_N : M ∩ N = {-2, -1} := by sorry

end intersection_M_N_l1996_199627


namespace eight_digit_divisibility_l1996_199615

theorem eight_digit_divisibility (A B : ℕ) : 
  A < 10 → B < 10 → (757 * 10^5 + A * 10^4 + B * 10^3 + 384) % 357 = 0 → A = 5 := by
sorry

end eight_digit_divisibility_l1996_199615


namespace set_B_equality_l1996_199696

def A : Set ℤ := {-1, 0, 1, 2}

def B : Set ℤ := {y | ∃ x ∈ A, y = x^2 - 2*x}

theorem set_B_equality : B = {-1, 0, 3} := by sorry

end set_B_equality_l1996_199696


namespace min_value_theorem_l1996_199612

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2*m + n = 1) :
  (1/m + 2/n) ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2*m₀ + n₀ = 1 ∧ 1/m₀ + 2/n₀ = 8 :=
sorry

end min_value_theorem_l1996_199612


namespace external_tangent_same_color_l1996_199617

/-- A point on a line --/
structure Point where
  x : ℝ

/-- A circle with diameter endpoints --/
structure Circle where
  p1 : Point
  p2 : Point

/-- A color represented as a natural number --/
def Color := ℕ

/-- The set of all circles formed by pairs of points --/
def allCircles (points : List Point) : List Circle :=
  sorry

/-- Checks if two circles are externally tangent --/
def areExternallyTangent (c1 c2 : Circle) : Prop :=
  sorry

/-- Assigns a color to each circle --/
def colorAssignment (circles : List Circle) (n : ℕ) : Circle → Color :=
  sorry

/-- Main theorem --/
theorem external_tangent_same_color 
  (k n : ℕ) (points : List Point) (h : k > 2^n) (h2 : points.length = k) :
  ∃ (c1 c2 : Circle), c1 ∈ allCircles points ∧ c2 ∈ allCircles points ∧ 
    c1 ≠ c2 ∧
    areExternallyTangent c1 c2 ∧
    colorAssignment (allCircles points) n c1 = colorAssignment (allCircles points) n c2 :=
  sorry

end external_tangent_same_color_l1996_199617


namespace square_difference_minus_product_l1996_199682

theorem square_difference_minus_product (a b : ℝ) : (a - b)^2 - b * (b - 2*a) = a^2 := by
  sorry

end square_difference_minus_product_l1996_199682


namespace largest_n_for_factorization_l1996_199640

theorem largest_n_for_factorization : 
  let P (n : ℤ) := ∃ (A B : ℤ), 5 * X^2 + n * X + 120 = (5 * X + A) * (X + B)
  ∀ (m : ℤ), P m → m ≤ 601 ∧ P 601 := by sorry

end largest_n_for_factorization_l1996_199640


namespace profit_percentage_previous_year_l1996_199611

/-- Given the following conditions for a company's finances over two years:
    1. In the previous year, profits were a percentage of revenues
    2. In 2009, revenues fell by 20%
    3. In 2009, profits were 20% of revenues
    4. Profits in 2009 were 160% of profits in the previous year
    
    This theorem proves that the percentage of profits to revenues in the previous year was 10%. -/
theorem profit_percentage_previous_year 
  (R : ℝ) -- Revenues in the previous year
  (P : ℝ) -- Profits in the previous year
  (h1 : P > 0) -- Ensure profits are positive
  (h2 : R > 0) -- Ensure revenues are positive
  (h3 : 0.8 * R * 0.2 = 1.6 * P) -- Condition relating 2009 profits to previous year
  : P / R = 0.1 := by
  sorry

#check profit_percentage_previous_year

end profit_percentage_previous_year_l1996_199611


namespace smallest_base_for_150_l1996_199670

theorem smallest_base_for_150 :
  ∃ b : ℕ, b = 6 ∧ b^2 ≤ 150 ∧ 150 < b^3 ∧ ∀ n : ℕ, n < b → (n^2 > 150 ∨ 150 ≥ n^3) :=
by sorry

end smallest_base_for_150_l1996_199670


namespace max_a_value_l1996_199677

theorem max_a_value (a : ℝ) : 
  a > 0 →
  (∀ x ∈ Set.Icc 1 2, ∃ y, y = x - a / x) →
  (∀ M : ℝ × ℝ, M.1 ∈ Set.Icc 1 2 → M.2 = M.1 - a / M.1 → 
    ∀ N : ℝ × ℝ, N.1 = M.1 ∧ 
    N.2 = (1 + a / 2) * (M.1 - 1) + (1 - a) → 
    (M.2 - N.2)^2 ≤ 1) →
  a ≤ 6 + 4 * Real.sqrt 2 :=
sorry

end max_a_value_l1996_199677


namespace domain_of_g_l1996_199695

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f (x : ℝ) : Prop := 0 ≤ x + 1 ∧ x + 1 ≤ 2

-- Define the function g
def g (x : ℝ) : ℝ := f (x + 3)

-- Theorem statement
theorem domain_of_g :
  (∀ x, domain_f x ↔ 0 ≤ x + 1 ∧ x + 1 ≤ 2) →
  (∀ x, g x = f (x + 3)) →
  (∀ x, g x ≠ 0 ↔ -3 ≤ x ∧ x ≤ -1) :=
sorry

end domain_of_g_l1996_199695


namespace profit_loss_recording_l1996_199674

/-- Represents the financial record of a store. -/
inductive FinancialRecord
  | profit (amount : ℤ)
  | loss (amount : ℤ)

/-- Records a financial transaction. -/
def recordTransaction (transaction : FinancialRecord) : ℤ :=
  match transaction with
  | FinancialRecord.profit amount => amount
  | FinancialRecord.loss amount => -amount

/-- The theorem stating how profits and losses should be recorded. -/
theorem profit_loss_recording (profitAmount lossAmount : ℤ) 
  (h : profitAmount = 20 ∧ lossAmount = 10) : 
  recordTransaction (FinancialRecord.profit profitAmount) = 20 ∧
  recordTransaction (FinancialRecord.loss lossAmount) = -10 := by
  sorry

end profit_loss_recording_l1996_199674


namespace inscribed_rectangle_area_l1996_199642

/-- Given a triangle with base 12 and altitude 8, and an inscribed rectangle with height 4,
    the area of the rectangle is 48. -/
theorem inscribed_rectangle_area (b h x : ℝ) : 
  b = 12 → h = 8 → x = h / 2 → x = 4 → 
  ∃ (w : ℝ), w * x = 48 := by sorry

end inscribed_rectangle_area_l1996_199642


namespace min_value_reciprocal_sum_l1996_199609

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  1 / x + 2 / y ≥ 8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 1 / x + 2 / y = 8 :=
sorry

end min_value_reciprocal_sum_l1996_199609


namespace diet_soda_bottles_l1996_199660

/-- Given a grocery store inventory, prove the number of diet soda bottles -/
theorem diet_soda_bottles (regular_soda : ℕ) (total_regular_and_diet : ℕ) 
  (h1 : regular_soda = 49)
  (h2 : total_regular_and_diet = 89) :
  total_regular_and_diet - regular_soda = 40 := by
  sorry

#check diet_soda_bottles

end diet_soda_bottles_l1996_199660


namespace elenas_bread_recipe_l1996_199605

/-- Given Elena's bread recipe, prove the amount of butter needed for the original recipe -/
theorem elenas_bread_recipe (original_flour : ℝ) (scale_factor : ℝ) (new_butter : ℝ) (new_flour : ℝ) :
  original_flour = 14 →
  scale_factor = 4 →
  new_butter = 12 →
  new_flour = 56 →
  (new_butter / new_flour) * original_flour = 3 := by
  sorry

end elenas_bread_recipe_l1996_199605


namespace bisector_triangle_area_l1996_199608

/-- Given a tetrahedron ABCD with face areas P (ABC) and Q (ADC), and dihedral angle α between these faces,
    the area S of the triangle formed by the plane bisecting α is (2PQ cos(α/2)) / (P + Q) -/
theorem bisector_triangle_area (P Q α : ℝ) (hP : P > 0) (hQ : Q > 0) (hα : 0 < α ∧ α < π) :
  ∃ S : ℝ, S = (2 * P * Q * Real.cos (α / 2)) / (P + Q) ∧ S > 0 :=
sorry

end bisector_triangle_area_l1996_199608


namespace student_bicycle_speed_l1996_199690

/-- Given two students A and B traveling 12 km, where A's speed is 1.2 times B's,
    and A arrives 1/6 hour earlier, B's speed is 12 km/h. -/
theorem student_bicycle_speed (distance : ℝ) (speed_ratio : ℝ) (time_diff : ℝ) :
  distance = 12 →
  speed_ratio = 1.2 →
  time_diff = 1/6 →
  ∃ (speed_B : ℝ),
    distance / speed_B - distance / (speed_ratio * speed_B) = time_diff ∧
    speed_B = 12 := by
  sorry

end student_bicycle_speed_l1996_199690


namespace smallest_four_digit_congruence_l1996_199650

theorem smallest_four_digit_congruence :
  ∃ (n : ℕ), 
    (n ≥ 1000 ∧ n < 10000) ∧ 
    (75 * n) % 345 = 225 ∧
    (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ (75 * m) % 345 = 225 → m ≥ n) ∧
    n = 1015 := by
  sorry

end smallest_four_digit_congruence_l1996_199650


namespace problem_solution_l1996_199635

theorem problem_solution : 
  ∀ N : ℝ, (2 + 3 + 4) / 3 = (1990 + 1991 + 1992) / N → N = 1991 := by
sorry

end problem_solution_l1996_199635


namespace probability_of_two_mismatches_l1996_199656

/-- Represents a set of pens and caps -/
structure PenSet :=
  (pens : Finset (Fin 3))
  (caps : Finset (Fin 3))

/-- Represents a pairing of pens and caps -/
def Pairing := Fin 3 → Fin 3

/-- The set of all possible pairings -/
def allPairings : Finset Pairing := sorry

/-- Predicate for a pairing that mismatches two pairs -/
def mismatchesTwoPairs (p : Pairing) : Prop := sorry

/-- The number of pairings that mismatch two pairs -/
def numMismatchedPairings : Nat := sorry

theorem probability_of_two_mismatches (ps : PenSet) :
  (numMismatchedPairings : ℚ) / (Finset.card allPairings : ℚ) = 1 / 2 := by sorry

end probability_of_two_mismatches_l1996_199656


namespace sqrt_product_equality_l1996_199639

theorem sqrt_product_equality : Real.sqrt 72 * Real.sqrt 27 * Real.sqrt 8 = 72 * Real.sqrt 3 := by
  sorry

end sqrt_product_equality_l1996_199639


namespace max_value_theorem_l1996_199620

theorem max_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ (x : ℝ), x = Real.rpow (a * b * c) (1/3) ∧
  (∀ (y : ℝ), (∃ (p q r : ℝ), 0 < p ∧ 0 < q ∧ 0 < r ∧ p + q + r = 1 ∧
    y ≤ a * p / q ∧ y ≤ b * q / r ∧ y ≤ c * r / p) → y ≤ x) :=
by sorry

end max_value_theorem_l1996_199620


namespace complex_magnitude_range_l1996_199683

theorem complex_magnitude_range (θ : ℝ) :
  let z : ℂ := Complex.mk (Real.sqrt 3 * Real.sin θ) (Real.cos θ)
  Complex.abs z < Real.sqrt 2 ↔ ∃ k : ℤ, -π/4 + k*π < θ ∧ θ < π/4 + k*π :=
by sorry

end complex_magnitude_range_l1996_199683


namespace concert_ticket_revenue_l1996_199614

theorem concert_ticket_revenue 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (full_price_tickets : ℕ) 
  (student_price_tickets : ℕ) 
  (full_price : ℕ) 
  (h1 : total_tickets = 150)
  (h2 : total_revenue = 2450)
  (h3 : student_price_tickets = total_tickets - full_price_tickets)
  (h4 : total_revenue = full_price_tickets * full_price + student_price_tickets * (full_price / 2))
  : full_price_tickets * full_price = 1150 := by
  sorry

#check concert_ticket_revenue

end concert_ticket_revenue_l1996_199614


namespace price_ratio_theorem_l1996_199662

theorem price_ratio_theorem (cost_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) 
  (profit_price : ℝ) (loss_price : ℝ) :
  profit_percent = 26 →
  loss_percent = 16 →
  profit_price = cost_price * (1 + profit_percent / 100) →
  loss_price = cost_price * (1 - loss_percent / 100) →
  loss_price / profit_price = 2 / 3 :=
by sorry

end price_ratio_theorem_l1996_199662


namespace final_nickel_count_is_45_l1996_199649

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  half_dollars : ℕ

/-- Represents a transaction of coins -/
structure CoinTransaction where
  nickels : ℤ
  dimes : ℤ
  quarters : ℤ
  half_dollars : ℤ

def initial_coins : CoinCount := {
  pennies := 45,
  nickels := 29,
  dimes := 16,
  quarters := 8,
  half_dollars := 4
}

def dad_gives : CoinTransaction := {
  nickels := 24,
  dimes := 15,
  quarters := 12,
  half_dollars := 6
}

def dad_takes : CoinTransaction := {
  nickels := -13,
  dimes := -9,
  quarters := -5,
  half_dollars := 0
}

def additional_percentage : ℚ := 20 / 100

/-- Applies a transaction to the coin count -/
def apply_transaction (coins : CoinCount) (transaction : CoinTransaction) : CoinCount :=
  { coins with
    nickels := (coins.nickels : ℤ) + transaction.nickels |>.toNat,
    dimes := (coins.dimes : ℤ) + transaction.dimes |>.toNat,
    quarters := (coins.quarters : ℤ) + transaction.quarters |>.toNat,
    half_dollars := (coins.half_dollars : ℤ) + transaction.half_dollars |>.toNat
  }

/-- Calculates the final number of nickels Sam has -/
def final_nickel_count : ℕ :=
  let after_first_transaction := apply_transaction initial_coins dad_gives
  let after_second_transaction := apply_transaction after_first_transaction dad_takes
  let additional_nickels := (dad_gives.nickels : ℚ) * additional_percentage |>.ceil.toNat
  after_second_transaction.nickels + additional_nickels

theorem final_nickel_count_is_45 : final_nickel_count = 45 := by
  sorry

end final_nickel_count_is_45_l1996_199649


namespace elevator_scenarios_count_l1996_199654

/-- Represents the number of floors in the building -/
def num_floors : ℕ := 7

/-- Represents the number of people entering the elevator -/
def num_people : ℕ := 3

/-- Calculates the number of scenarios where exactly one person goes to the top floor
    and person A does not get off on the second floor -/
def elevator_scenarios : ℕ :=
  let a_to_top := (num_floors - 2)^(num_people - 1)
  let others_to_top := (num_people - 1) * (num_floors - 3) * (num_floors - 2)
  a_to_top + others_to_top

/-- The main theorem stating that the number of scenarios is 65 -/
theorem elevator_scenarios_count :
  elevator_scenarios = 65 := by sorry

end elevator_scenarios_count_l1996_199654


namespace inscribed_circle_radius_l1996_199652

/-- Given a right triangle with area 24 cm² and hypotenuse 10 cm, 
    prove that the radius of its inscribed circle is 2 cm. -/
theorem inscribed_circle_radius 
  (S : ℝ) 
  (c : ℝ) 
  (h1 : S = 24) 
  (h2 : c = 10) : 
  let a := Real.sqrt ((c^2 / 2) + Real.sqrt ((c^4 / 4) - S^2))
  let b := Real.sqrt ((c^2 / 2) - Real.sqrt ((c^4 / 4) - S^2))
  (a + b - c) / 2 = 2 := by
  sorry

end inscribed_circle_radius_l1996_199652


namespace right_triangle_properties_l1996_199686

theorem right_triangle_properties (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9) :
  let variance := (a^2 + b^2 + 9) / 3 - ((a + b + 3) / 3)^2
  let std_dev := Real.sqrt variance
  let min_std_dev := Real.sqrt 2 - 1
  let optimal_leg := 3 * Real.sqrt 2 / 2
  (variance < 5) ∧
  (std_dev ≥ min_std_dev) ∧
  (std_dev = min_std_dev ↔ a = optimal_leg ∧ b = optimal_leg) :=
by sorry

end right_triangle_properties_l1996_199686


namespace quadratic_root_existence_l1996_199621

/-- Given a quadratic function f(x) = x^2 + x + m where m is positive,
    if f(t) < 0 for some real t, then f has a root in the interval (t, t+1) -/
theorem quadratic_root_existence (m : ℝ) (t : ℝ) (h_m : m > 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 + x + m
  f t < 0 → ∃ x : ℝ, t < x ∧ x < t + 1 ∧ f x = 0 := by
  sorry

end quadratic_root_existence_l1996_199621


namespace oil_leaked_before_is_6522_l1996_199663

/-- The amount of oil leaked before engineers started to fix the pipe -/
def oil_leaked_before : ℕ := 11687 - 5165

/-- Theorem stating that the amount of oil leaked before engineers started to fix the pipe is 6522 liters -/
theorem oil_leaked_before_is_6522 : oil_leaked_before = 6522 := by
  sorry

end oil_leaked_before_is_6522_l1996_199663


namespace square_greater_not_sufficient_nor_necessary_l1996_199659

theorem square_greater_not_sufficient_nor_necessary :
  ∃ (a b : ℝ), a^2 > b^2 ∧ ¬(a > b) ∧
  ∃ (c d : ℝ), c > d ∧ ¬(c^2 > d^2) := by
  sorry

end square_greater_not_sufficient_nor_necessary_l1996_199659


namespace inequality_proof_l1996_199691

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) ≥ (a*b + b*c + c*a)^3 := by
  sorry

end inequality_proof_l1996_199691


namespace trigonometric_equation_solutions_l1996_199631

theorem trigonometric_equation_solutions :
  ∀ x : ℝ, x ∈ Set.Icc 0 (2 * Real.pi) →
    (3 * Real.sin x = 1 + Real.cos (2 * x)) ↔ (x = Real.pi / 6 ∨ x = 5 * Real.pi / 6) := by
  sorry

end trigonometric_equation_solutions_l1996_199631


namespace empirical_regression_equation_l1996_199600

/-- Data for 10 years of resident income and goods sales -/
def income : List Float := [32.2, 31.1, 32.9, 35.7, 37.1, 38.0, 39.0, 43.0, 44.6, 46.0]
def sales : List Float := [25.0, 30.0, 34.0, 37.0, 39.0, 41.0, 42.0, 44.0, 48.0, 51.0]

/-- Given statistics -/
def sum_x : Float := 379.6
def sum_y : Float := 391.0
def sum_x_squared : Float := 246.904
def sum_y_squared : Float := 568.9
def correlation_coefficient : Float := 0.95

/-- Mean values -/
def mean_x : Float := sum_x / 10
def mean_y : Float := sum_y / 10

/-- Regression coefficients -/
def b_hat : Float := correlation_coefficient * (sum_y_squared.sqrt / sum_x_squared.sqrt)
def a_hat : Float := mean_y - b_hat * mean_x

theorem empirical_regression_equation :
  (b_hat * 100).round / 100 = 1.44 ∧ 
  (a_hat * 100).round / 100 = -15.56 := by
  sorry

#check empirical_regression_equation

end empirical_regression_equation_l1996_199600


namespace average_age_increase_l1996_199637

theorem average_age_increase 
  (n : Nat) 
  (initial_avg : ℝ) 
  (man1_age man2_age : ℝ) 
  (women_avg : ℝ) : 
  n = 8 → 
  man1_age = 20 → 
  man2_age = 22 → 
  women_avg = 29 → 
  ((n * initial_avg - man1_age - man2_age + 2 * women_avg) / n) - initial_avg = 2 :=
by
  sorry

end average_age_increase_l1996_199637


namespace secret_spreading_day_l1996_199680

/-- The number of new students who learn the secret on day n -/
def new_students (n : ℕ) : ℕ := 3^n

/-- The total number of students who know the secret after n days -/
def total_students (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day when 3280 students know the secret -/
theorem secret_spreading_day : 
  ∃ n : ℕ, total_students n = 3280 ∧ n = 7 :=
sorry

end secret_spreading_day_l1996_199680


namespace sum_of_powers_of_i_l1996_199676

theorem sum_of_powers_of_i (i : ℂ) (h : i^2 = -1) :
  i^14761 + i^14762 + i^14763 + i^14764 = 0 := by
  sorry

end sum_of_powers_of_i_l1996_199676


namespace problem_solution_l1996_199610

-- Define the solution set for |x-m| < |x|
def solution_set (m : ℝ) : Set ℝ := {x : ℝ | 1 < x}

-- Define the inequality condition
def inequality_condition (a m : ℝ) (x : ℝ) : Prop :=
  (a - 5) / x < |1 + 1/x| - |1 - m/x| ∧ |1 + 1/x| - |1 - m/x| < (a + 2) / x

theorem problem_solution :
  (∀ x : ℝ, x ∈ solution_set m ↔ |x - m| < |x|) →
  m = 2 ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → inequality_condition a m x) ↔ 1 < a ∧ a ≤ 4) :=
sorry

end problem_solution_l1996_199610


namespace parabolas_intersection_circle_l1996_199689

/-- The parabolas y = (x - 2)^2 and x + 1 = (y + 2)^2 intersect at four points that lie on a circle with radius squared equal to 3/2 -/
theorem parabolas_intersection_circle (x y : ℝ) : 
  (y = (x - 2)^2 ∧ x + 1 = (y + 2)^2) → 
  (x - 5/2)^2 + (y + 3/2)^2 = 3/2 := by sorry

end parabolas_intersection_circle_l1996_199689


namespace percentage_problem_l1996_199601

theorem percentage_problem (p : ℝ) (h1 : 0.25 * 660 = (p/100) * 1500 - 15) : p = 12 := by
  sorry

end percentage_problem_l1996_199601


namespace complex_number_magnitude_l1996_199667

theorem complex_number_magnitude (a : ℝ) (i : ℂ) (z : ℂ) : 
  a < 0 → 
  i * i = -1 → 
  z = a * i / (1 - 2 * i) → 
  Complex.abs z = Real.sqrt 5 → 
  a = -5 := by sorry

end complex_number_magnitude_l1996_199667


namespace mango_count_proof_l1996_199684

/-- Given a ratio of mangoes to apples and the number of apples, 
    calculate the number of mangoes -/
def calculate_mangoes (mango_ratio : ℕ) (apple_ratio : ℕ) (apple_count : ℕ) : ℕ :=
  (mango_ratio * apple_count) / apple_ratio

/-- Theorem: Given the ratio of mangoes to apples is 10:3 and there are 36 apples,
    prove that the number of mangoes is 120 -/
theorem mango_count_proof :
  calculate_mangoes 10 3 36 = 120 := by
  sorry

end mango_count_proof_l1996_199684


namespace vector_computation_l1996_199648

def a : Fin 2 → ℝ := ![2, 4]
def b : Fin 2 → ℝ := ![-1, 1]

theorem vector_computation : 
  (2 • a - b) = ![5, 7] := by sorry

end vector_computation_l1996_199648


namespace program_output_is_twenty_l1996_199607

/-- The result of evaluating the arithmetic expression (3+2)*4 -/
def program_result : ℕ := (3 + 2) * 4

/-- Theorem stating that the result of the program is 20 -/
theorem program_output_is_twenty : program_result = 20 := by
  sorry

end program_output_is_twenty_l1996_199607


namespace compound_has_three_oxygen_atoms_l1996_199673

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  aluminium : Nat
  oxygen : Nat
  hydrogen : Nat

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) : Nat :=
  27 * c.aluminium + 16 * c.oxygen + c.hydrogen

/-- Theorem stating that the compound with 3 oxygen atoms satisfies the given conditions -/
theorem compound_has_three_oxygen_atoms :
  ∃ (c : Compound), c.aluminium = 1 ∧ c.hydrogen = 3 ∧ molecularWeight c = 78 ∧ c.oxygen = 3 := by
  sorry

#check compound_has_three_oxygen_atoms

end compound_has_three_oxygen_atoms_l1996_199673


namespace y_is_odd_square_l1996_199688

def x : ℕ → ℤ
| 0 => 0
| 1 => 1
| (n + 2) => 3 * x (n + 1) - 2 * x n

def y (n : ℕ) : ℤ := x n ^ 2 + 2 ^ (n + 2)

theorem y_is_odd_square (n : ℕ) (h : n > 0) :
  ∃ k : ℤ, Odd k ∧ y n = k ^ 2 := by sorry

end y_is_odd_square_l1996_199688


namespace odd_integers_sum_169_l1996_199636

/-- Sum of consecutive odd integers from 1 to n -/
def sumOddIntegers (n : ℕ) : ℕ :=
  (n * n + n) / 2

/-- The problem statement -/
theorem odd_integers_sum_169 :
  ∃ n : ℕ, n % 2 = 1 ∧ sumOddIntegers n = 169 ∧ n = 25 := by
  sorry

end odd_integers_sum_169_l1996_199636


namespace y_value_l1996_199625

theorem y_value (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -5) : y = 18 := by
  sorry

end y_value_l1996_199625


namespace gcf_of_36_60_90_l1996_199629

theorem gcf_of_36_60_90 : Nat.gcd 36 (Nat.gcd 60 90) = 6 := by
  sorry

end gcf_of_36_60_90_l1996_199629


namespace tangent_slope_at_one_l1996_199616

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x^2 + 1

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 5 := by sorry

end tangent_slope_at_one_l1996_199616


namespace shipping_scenario_l1996_199661

/-- The number of broken artworks in a shipping scenario -/
def broken_artworks : ℕ :=
  -- We'll define this later in the theorem
  4

/-- The total number of artworks -/
def total_artworks : ℕ := 2000

/-- The shipping cost per artwork in yuan -/
def shipping_cost : ℚ := 0.2

/-- The compensation cost for a broken artwork in yuan -/
def compensation_cost : ℚ := 2.3

/-- The total profit in yuan -/
def total_profit : ℚ := 390

theorem shipping_scenario :
  shipping_cost * (total_artworks - broken_artworks : ℚ) - compensation_cost * broken_artworks = total_profit :=
by sorry

end shipping_scenario_l1996_199661


namespace max_abs_sum_l1996_199643

theorem max_abs_sum (a b c : ℝ) : 
  (∀ x : ℝ, |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) → 
  |a| + |b| + |c| ≤ 3 :=
by sorry

end max_abs_sum_l1996_199643


namespace combined_solid_volume_l1996_199638

/-- The volume of the combined solid with a square base and triangular prism on top -/
theorem combined_solid_volume (s : ℝ) (h : s = 8 * Real.sqrt 2) :
  let original_volume := (Real.sqrt 2 * (2 * s)^3) / 24
  let prism_volume := (s^3 * Real.sqrt 15) / 4
  original_volume + prism_volume = 2048 + 576 * Real.sqrt 30 := by
sorry

end combined_solid_volume_l1996_199638


namespace z1_over_z2_value_l1996_199644

def complex_symmetric_about_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem z1_over_z2_value (z₁ z₂ : ℂ) :
  complex_symmetric_about_imaginary_axis z₁ z₂ →
  z₁ = 3 - I →
  z₁ / z₂ = -4/5 + 3/5 * I :=
by sorry

end z1_over_z2_value_l1996_199644


namespace smallest_number_proof_smallest_number_l1996_199681

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_proof (N : ℕ) (x : ℕ) : Prop :=
  N - x = 746 ∧
  is_divisible_by (N - x) 8 ∧
  is_divisible_by (N - x) 14 ∧
  is_divisible_by (N - x) 26 ∧
  is_divisible_by (N - x) 28 ∧
  ∀ M : ℕ, M < N → ¬(∃ y : ℕ, smallest_number_proof M y)

theorem smallest_number : smallest_number_proof 1474 728 := by
  sorry

end smallest_number_proof_smallest_number_l1996_199681


namespace problem_statement_l1996_199645

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 6)
  (h2 : x * y = 128)
  (h3 : x = 2 * y^2) :
  (x + y) / 2 = 18 := by
  sorry

end problem_statement_l1996_199645


namespace class_size_proof_l1996_199634

theorem class_size_proof (total : ℕ) 
  (h1 : (3 : ℚ) / 5 * total + (1 : ℚ) / 5 * total + 10 = total) : total = 50 := by
  sorry

end class_size_proof_l1996_199634


namespace circular_film_radius_l1996_199606

/-- The radius of a circular film formed by a non-mixing liquid on water -/
theorem circular_film_radius 
  (volume : ℝ) 
  (thickness : ℝ) 
  (radius : ℝ) 
  (h1 : volume = 400) 
  (h2 : thickness = 0.2) 
  (h3 : π * radius^2 * thickness = volume) : 
  radius = Real.sqrt (2000 / π) := by
sorry

end circular_film_radius_l1996_199606


namespace equation_solutions_l1996_199693

theorem equation_solutions :
  (∃ x : ℝ, 6 * x - 7 = 4 * x - 5 ∧ x = 1) ∧
  (∃ x : ℝ, (1/2) * x - 6 = (3/4) * x ∧ x = -24) := by
  sorry

end equation_solutions_l1996_199693


namespace max_n_value_l1996_199657

theorem max_n_value (a b c : ℝ) (n : ℕ) 
  (h1 : a > b) (h2 : b > c)
  (h3 : ∀ a b c, a > b → b > c → 1 / (a - b) + 1 / (b - c) ≥ n / (a - c)) :
  n ≤ 4 ∧ ∃ a b c, a > b ∧ b > c ∧ 1 / (a - b) + 1 / (b - c) = 4 / (a - c) :=
sorry

end max_n_value_l1996_199657


namespace distance_P_to_x_axis_l1996_199671

/-- The distance from a point to the x-axis is the absolute value of its y-coordinate -/
def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

/-- Point P with coordinates (3, -5) -/
def P : ℝ × ℝ := (3, -5)

/-- Theorem stating that the distance from P to the x-axis is 5 -/
theorem distance_P_to_x_axis :
  distance_to_x_axis P = 5 := by sorry

end distance_P_to_x_axis_l1996_199671


namespace distance_to_hole_is_250_l1996_199632

/-- The distance from the starting tee to the hole in a golf game --/
def distance_to_hole (first_hit second_hit beyond_hole : ℕ) : ℕ :=
  first_hit + second_hit - beyond_hole

/-- Theorem stating the distance to the hole given the conditions in the problem --/
theorem distance_to_hole_is_250 :
  let first_hit := 180
  let second_hit := first_hit / 2
  let beyond_hole := 20
  distance_to_hole first_hit second_hit beyond_hole = 250 := by
  sorry

#eval distance_to_hole 180 90 20

end distance_to_hole_is_250_l1996_199632


namespace infinite_geometric_series_ratio_l1996_199685

/-- An infinite geometric series with first term a and common ratio r has sum S if and only if |r| < 1 and S = a / (1 - r) -/
def is_infinite_geometric_series_sum (a : ℝ) (r : ℝ) (S : ℝ) : Prop :=
  |r| < 1 ∧ S = a / (1 - r)

/-- The positive common ratio of an infinite geometric series with first term 500 and sum 4000 is 7/8 -/
theorem infinite_geometric_series_ratio : 
  ∃ (r : ℝ), r > 0 ∧ is_infinite_geometric_series_sum 500 r 4000 ∧ r = 7/8 := by
sorry

end infinite_geometric_series_ratio_l1996_199685


namespace existence_condition_equiv_range_l1996_199624

theorem existence_condition_equiv_range (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 3 ∧ |x₀^2 - a*x₀ + 4| ≤ 3*x₀) ↔ 
  (2 ≤ a ∧ a ≤ 7 + 1/3) := by
sorry

end existence_condition_equiv_range_l1996_199624


namespace tulip_petals_l1996_199697

/-- Proves that each tulip has 3 petals given the conditions in Elena's garden --/
theorem tulip_petals (num_lilies : ℕ) (num_tulips : ℕ) (lily_petals : ℕ) (total_petals : ℕ)
  (h1 : num_lilies = 8)
  (h2 : num_tulips = 5)
  (h3 : lily_petals = 6)
  (h4 : total_petals = 63)
  (h5 : total_petals = num_lilies * lily_petals + num_tulips * (total_petals - num_lilies * lily_petals) / num_tulips) :
  (total_petals - num_lilies * lily_petals) / num_tulips = 3 := by
  sorry

#eval (63 - 8 * 6) / 5  -- This should output 3

end tulip_petals_l1996_199697


namespace helga_work_days_l1996_199633

/-- Represents Helga's work schedule and output --/
structure HelgaWork where
  articles_per_half_hour : ℕ := 5
  usual_hours_per_day : ℕ := 4
  extra_hours_thursday : ℕ := 2
  extra_hours_friday : ℕ := 3
  total_articles_week : ℕ := 250

/-- Calculates the number of days Helga usually works in a week --/
def usual_work_days (hw : HelgaWork) : ℕ :=
  let articles_per_hour := hw.articles_per_half_hour * 2
  let articles_per_day := articles_per_hour * hw.usual_hours_per_day
  let extra_articles := articles_per_hour * (hw.extra_hours_thursday + hw.extra_hours_friday)
  let usual_articles := hw.total_articles_week - extra_articles
  usual_articles / articles_per_day

theorem helga_work_days (hw : HelgaWork) : usual_work_days hw = 5 := by
  sorry

end helga_work_days_l1996_199633


namespace rhombus_perimeter_l1996_199641

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * s = 4 * Real.sqrt 241 :=
by sorry

end rhombus_perimeter_l1996_199641


namespace library_shelf_capacity_l1996_199694

/-- Given a library with a total number of books and shelves, 
    calculate the number of books per shelf. -/
def books_per_shelf (total_books : ℕ) (total_shelves : ℕ) : ℕ :=
  total_books / total_shelves

/-- Theorem stating that in a library with 14240 books and 1780 shelves,
    each shelf holds 8 books. -/
theorem library_shelf_capacity : books_per_shelf 14240 1780 = 8 := by
  sorry

end library_shelf_capacity_l1996_199694


namespace max_students_before_third_wave_l1996_199651

/-- The total number of students in the class -/
def total_students : ℕ := 35

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- The theorem to be proved -/
theorem max_students_before_third_wave :
  ∃ (a b c : ℕ),
    is_prime a ∧ is_prime b ∧ is_prime c ∧
    a + b + c = total_students ∧
    ∀ (x y z : ℕ),
      is_prime x ∧ is_prime y ∧ is_prime z ∧
      x + y + z = total_students →
      total_students - (a + b) ≥ total_students - (x + y) :=
sorry

end max_students_before_third_wave_l1996_199651


namespace parallel_vectors_x_value_l1996_199613

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -1)
  parallel a b → x = -3 := by
sorry

end parallel_vectors_x_value_l1996_199613


namespace hyperbola_parabola_symmetry_l1996_199618

/-- Given a hyperbola and points on a parabola, prove the value of m -/
theorem hyperbola_parabola_symmetry (a b : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (m : ℝ) : 
  a > 0 → b > 0 → 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ |2*x| = 4) →  -- Condition for hyperbola
  y₁ = a*x₁^2 → y₂ = a*x₂^2 →  -- Points on parabola
  (x₁ + x₂)/2 + m = (y₁ + y₂)/2 →  -- Midpoint on symmetry line
  (y₂ - y₁)/(x₂ - x₁) = -1 →  -- Perpendicular to symmetry line
  x₁*x₂ = -1/2 → 
  m = 3/2 := by
sorry

end hyperbola_parabola_symmetry_l1996_199618


namespace first_year_after_2020_with_sum_4_l1996_199658

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Check if a year is after 2020 and has sum of digits equal to 4 -/
def isValidYear (year : ℕ) : Prop :=
  year > 2020 ∧ sumOfDigits year = 4

/-- 2030 is the first year after 2020 with sum of digits equal to 4 -/
theorem first_year_after_2020_with_sum_4 :
  (∀ y : ℕ, y > 2020 ∧ y < 2030 → sumOfDigits y ≠ 4) ∧
  sumOfDigits 2030 = 4 :=
sorry

end first_year_after_2020_with_sum_4_l1996_199658


namespace malcolm_total_followers_l1996_199647

/-- The total number of followers Malcolm has on all social media platforms --/
def total_followers (instagram facebook : ℕ) : ℕ :=
  let twitter := (instagram + facebook) / 2
  let tiktok := 3 * twitter
  let youtube := tiktok + 510
  instagram + facebook + twitter + tiktok + youtube

/-- Theorem stating that Malcolm's total followers across all platforms is 3840 --/
theorem malcolm_total_followers :
  total_followers 240 500 = 3840 := by
  sorry

#eval total_followers 240 500

end malcolm_total_followers_l1996_199647


namespace weighted_average_salary_l1996_199666

/-- Represents the categories of employees in the departmental store -/
inductive EmployeeCategory
  | Manager
  | Associate
  | LeadCashier
  | SalesRepresentative

/-- Returns the number of employees for a given category -/
def employeeCount (category : EmployeeCategory) : Nat :=
  match category with
  | .Manager => 9
  | .Associate => 18
  | .LeadCashier => 6
  | .SalesRepresentative => 45

/-- Returns the average salary for a given category -/
def averageSalary (category : EmployeeCategory) : Nat :=
  match category with
  | .Manager => 4500
  | .Associate => 3500
  | .LeadCashier => 3000
  | .SalesRepresentative => 2500

/-- Calculates the total salary for all employees -/
def totalSalary : Nat :=
  (employeeCount .Manager * averageSalary .Manager) +
  (employeeCount .Associate * averageSalary .Associate) +
  (employeeCount .LeadCashier * averageSalary .LeadCashier) +
  (employeeCount .SalesRepresentative * averageSalary .SalesRepresentative)

/-- Calculates the total number of employees -/
def totalEmployees : Nat :=
  employeeCount .Manager +
  employeeCount .Associate +
  employeeCount .LeadCashier +
  employeeCount .SalesRepresentative

/-- Theorem stating that the weighted average salary is $3000 -/
theorem weighted_average_salary :
  totalSalary / totalEmployees = 3000 := by
  sorry


end weighted_average_salary_l1996_199666


namespace grid_solution_l1996_199675

/-- Represents a 3x3 grid of integers -/
structure Grid :=
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : Int)

/-- Checks if the middle number in each row is the sum of the numbers at its ends -/
def rowSumsValid (g : Grid) : Prop :=
  g.a12 = g.a11 + g.a13 ∧ g.a22 = g.a21 + g.a23 ∧ g.a32 = g.a31 + g.a33

/-- Checks if the sums of the numbers on both diagonals are equal -/
def diagonalSumsEqual (g : Grid) : Prop :=
  g.a11 + g.a22 + g.a33 = g.a13 + g.a22 + g.a31

/-- The theorem stating the solution to the grid problem -/
theorem grid_solution :
  ∀ (g : Grid),
    g.a11 = 4 ∧ g.a12 = 12 ∧ g.a13 = 8 ∧ g.a21 = 10 →
    rowSumsValid g →
    diagonalSumsEqual g →
    g.a22 = 3 ∧ g.a23 = 9 ∧ g.a31 = -3 ∧ g.a32 = -2 ∧ g.a33 = 1 :=
by sorry

end grid_solution_l1996_199675


namespace intersection_union_equality_l1996_199604

def M : Set Nat := {0, 1, 2, 4, 5, 7}
def N : Set Nat := {1, 4, 6, 8, 9}
def P : Set Nat := {4, 7, 9}

theorem intersection_union_equality : (M ∩ N) ∪ (M ∩ P) = {1, 4, 7} := by
  sorry

end intersection_union_equality_l1996_199604


namespace inequality_solution_set_l1996_199628

theorem inequality_solution_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (x - a) * (x - a^2) < 0} = {x : ℝ | a^2 < x ∧ x < a} := by
  sorry

end inequality_solution_set_l1996_199628


namespace eugene_toothpick_boxes_l1996_199653

def toothpicks_per_card : ℕ := 64
def total_cards : ℕ := 52
def unused_cards : ℕ := 23
def toothpicks_per_box : ℕ := 550

theorem eugene_toothpick_boxes : 
  ∃ (boxes : ℕ), 
    boxes = (((total_cards - unused_cards) * toothpicks_per_card + toothpicks_per_box - 1) / toothpicks_per_box : ℕ) ∧ 
    boxes = 4 := by
  sorry

end eugene_toothpick_boxes_l1996_199653


namespace min_value_x_plus_2y_l1996_199679

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = x*y) :
  x + 2*y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = x₀*y₀ ∧ x₀ + 2*y₀ = 8 := by
  sorry

end min_value_x_plus_2y_l1996_199679


namespace computer_device_properties_l1996_199665

theorem computer_device_properties :
  ∃ f : ℕ → ℕ → ℕ,
    (f 1 1 = 1) ∧
    (∀ m n : ℕ, f m (n + 1) = f m n + 2) ∧
    (∀ m : ℕ, f (m + 1) 1 = 2 * f m 1) ∧
    (∀ n : ℕ, f 1 n = 2 * n - 1) ∧
    (∀ m : ℕ, f m 1 = 2^(m - 1)) :=
by sorry

end computer_device_properties_l1996_199665


namespace students_opted_both_math_and_science_l1996_199626

theorem students_opted_both_math_and_science 
  (total_students : ℕ) 
  (not_math : ℕ) 
  (not_science : ℕ) 
  (not_either : ℕ) 
  (h1 : total_students = 40)
  (h2 : not_math = 10)
  (h3 : not_science = 15)
  (h4 : not_either = 2) :
  total_students - (not_math + not_science - not_either) = 17 := by
  sorry

#check students_opted_both_math_and_science

end students_opted_both_math_and_science_l1996_199626


namespace derivative_at_one_l1996_199692

-- Define the function f(x) = (x-2)²
def f (x : ℝ) : ℝ := (x - 2)^2

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = -2 := by sorry

end derivative_at_one_l1996_199692


namespace cubic_function_unique_form_l1996_199672

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b

theorem cubic_function_unique_form 
  (a b : ℝ) 
  (h_a : a > 0)
  (h_max : ∃ x₁, ∀ x, f x a b ≤ f x₁ a b ∧ f x₁ a b = 5)
  (h_min : ∃ x₂, ∀ x, f x a b ≥ f x₂ a b ∧ f x₂ a b = 1) :
  ∀ x, f x a b = x^3 + 3*x^2 + 1 :=
sorry

end cubic_function_unique_form_l1996_199672


namespace least_clock_equivalent_is_nine_l1996_199699

/-- A number is clock equivalent to its square if their difference is divisible by 12 -/
def ClockEquivalent (n : ℕ) : Prop :=
  (n ^ 2 - n) % 12 = 0

/-- The least whole number greater than 4 that is clock equivalent to its square -/
def LeastClockEquivalent : ℕ := 9

theorem least_clock_equivalent_is_nine :
  (LeastClockEquivalent > 4) ∧
  ClockEquivalent LeastClockEquivalent ∧
  ∀ n : ℕ, (n > 4 ∧ n < LeastClockEquivalent) → ¬ClockEquivalent n :=
by sorry

end least_clock_equivalent_is_nine_l1996_199699


namespace abc_def_ratio_l1996_199603

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) :
  a * b * c / (d * e * f) = 1 / 16 := by
  sorry

end abc_def_ratio_l1996_199603


namespace journey_duration_first_part_l1996_199602

/-- Proves the duration of the first part of a journey given specific conditions -/
theorem journey_duration_first_part 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (speed_first_part : ℝ) 
  (speed_second_part : ℝ) 
  (h1 : total_distance = 240)
  (h2 : total_time = 5)
  (h3 : speed_first_part = 40)
  (h4 : speed_second_part = 60) :
  ∃ (t1 : ℝ), t1 = 3 ∧ 
    speed_first_part * t1 + speed_second_part * (total_time - t1) = total_distance :=
by sorry


end journey_duration_first_part_l1996_199602


namespace plane_perpendicular_criterion_l1996_199646

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_criterion 
  (m : Line) (α β : Plane) :
  contains β m → perp m α → perpPlanes α β :=
sorry

end plane_perpendicular_criterion_l1996_199646


namespace combinatorial_identity_l1996_199655

theorem combinatorial_identity : Nat.choose 98 97 + 2 * Nat.choose 98 96 + Nat.choose 98 95 = Nat.choose 100 97 := by
  sorry

end combinatorial_identity_l1996_199655
