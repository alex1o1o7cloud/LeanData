import Mathlib

namespace complex_equation_sum_l1657_165784

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (a + 2*i)/i = b + i) : a + b = 1 := by
  sorry

end complex_equation_sum_l1657_165784


namespace square_construction_l1657_165792

noncomputable section

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define the line
def Line (a b c : ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define the square
structure Square (P Q V U : ℝ × ℝ) : Prop where
  side_equal : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (Q.1 - V.1)^2 + (Q.2 - V.2)^2
             ∧ (Q.1 - V.1)^2 + (Q.2 - V.2)^2 = (V.1 - U.1)^2 + (V.2 - U.2)^2
             ∧ (V.1 - U.1)^2 + (V.2 - U.2)^2 = (U.1 - P.1)^2 + (U.2 - P.2)^2
  right_angles : (P.1 - Q.1) * (Q.1 - V.1) + (P.2 - Q.2) * (Q.2 - V.2) = 0
                ∧ (Q.1 - V.1) * (V.1 - U.1) + (Q.2 - V.2) * (V.2 - U.2) = 0
                ∧ (V.1 - U.1) * (U.1 - P.1) + (V.2 - U.2) * (U.2 - P.2) = 0
                ∧ (U.1 - P.1) * (P.1 - Q.1) + (U.2 - P.2) * (P.2 - Q.2) = 0

theorem square_construction (O : ℝ × ℝ) (r : ℝ) (a b c : ℝ) 
  (h : ∀ p ∈ Line a b c, p ∉ Circle O r) :
  ∃ P Q V U : ℝ × ℝ, Square P Q V U ∧ 
    P ∈ Line a b c ∧ Q ∈ Line a b c ∧
    V ∈ Circle O r ∧ U ∈ Circle O r :=
sorry

end square_construction_l1657_165792


namespace alice_original_seat_was_six_l1657_165722

/-- Represents the number of seats -/
def num_seats : Nat := 7

/-- Represents the seat Alice ends up in -/
def alice_final_seat : Nat := 4

/-- Represents the net movement of all other friends -/
def net_movement : Int := 2

/-- Calculates Alice's original seat given her final seat and the net movement of others -/
def alice_original_seat (final_seat : Nat) (net_move : Int) : Nat :=
  final_seat + net_move.toNat

/-- Theorem stating Alice's original seat was 6 -/
theorem alice_original_seat_was_six :
  alice_original_seat alice_final_seat net_movement = 6 := by
  sorry

#eval alice_original_seat alice_final_seat net_movement

end alice_original_seat_was_six_l1657_165722


namespace manufacturer_central_tendencies_l1657_165717

def manufacturer_A : List ℝ := [3, 4, 5, 6, 8, 8, 8, 10]
def manufacturer_B : List ℝ := [4, 6, 6, 6, 8, 9, 12, 13]
def manufacturer_C : List ℝ := [3, 3, 4, 7, 9, 10, 11, 12]

def mode (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry
def median (l : List ℝ) : ℝ := sorry

theorem manufacturer_central_tendencies :
  (mode manufacturer_A = 8) ∧
  (mean manufacturer_B = 8) ∧
  (median manufacturer_C = 8) := by sorry

end manufacturer_central_tendencies_l1657_165717


namespace inequality_proof_l1657_165715

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) : 
  (x^2 + y*z) / Real.sqrt (2*x^2*(y+z)) + 
  (y^2 + z*x) / Real.sqrt (2*y^2*(z+x)) + 
  (z^2 + x*y) / Real.sqrt (2*z^2*(x+y)) ≥ 1 := by
  sorry

end inequality_proof_l1657_165715


namespace josies_initial_money_l1657_165777

/-- The amount of money Josie's mom gave her initially --/
def initial_money (milk_price bread_price detergent_price banana_price_per_pound : ℚ)
  (milk_discount detergent_discount : ℚ) (banana_pounds leftover_money : ℚ) : ℚ :=
  (milk_price * (1 - milk_discount) + bread_price + 
   (detergent_price - detergent_discount) + 
   (banana_price_per_pound * banana_pounds) + leftover_money)

/-- Theorem stating that Josie's mom gave her $20.00 initially --/
theorem josies_initial_money :
  initial_money 4 3.5 10.25 0.75 0.5 1.25 2 4 = 20 := by
  sorry

end josies_initial_money_l1657_165777


namespace softball_team_ratio_l1657_165775

theorem softball_team_ratio (n : ℕ) (men women : ℕ → ℕ) : 
  n = 20 →
  (∀ k, k ≤ n → k ≥ 3 → women k = men k + k / 3) →
  men n + women n = n →
  (men n : ℚ) / (women n : ℚ) = 7 / 13 :=
sorry

end softball_team_ratio_l1657_165775


namespace problem_statement_l1657_165794

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 3) :
  (9/5 ≤ a^2 + b^2 ∧ a^2 + b^2 < 9) ∧ a^3*b + 4*a*b^3 ≤ 81/16 := by
  sorry

end problem_statement_l1657_165794


namespace even_sum_probability_l1657_165705

/-- Represents a wheel with even and odd sections -/
structure Wheel where
  total : ℕ
  even : ℕ
  odd : ℕ
  sum_sections : total = even + odd

/-- The probability of getting an even sum when spinning two wheels -/
def prob_even_sum (w1 w2 : Wheel) : ℚ :=
  let p1_even := w1.even / w1.total
  let p1_odd := w1.odd / w1.total
  let p2_even := w2.even / w2.total
  let p2_odd := w2.odd / w2.total
  p1_even * p2_even + p1_odd * p2_odd

theorem even_sum_probability :
  let w1 : Wheel := { total := 6, even := 2, odd := 4, sum_sections := by rfl }
  let w2 : Wheel := { total := 8, even := 3, odd := 5, sum_sections := by rfl }
  prob_even_sum w1 w2 = 13 / 24 := by
  sorry

end even_sum_probability_l1657_165705


namespace subsets_count_l1657_165759

theorem subsets_count : ∃ (n : ℕ), n = (Finset.filter (fun X => {1, 2} ⊆ X ∧ X ⊆ {1, 2, 3, 4, 5}) (Finset.powerset {1, 2, 3, 4, 5})).card ∧ n = 8 := by
  sorry

end subsets_count_l1657_165759


namespace parallel_vectors_x_value_l1657_165714

/-- Given two vectors a and b, if (a + xb) is parallel to (a - b), then x = -1 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (3, 4))
  (h2 : b = (2, 1))
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ (a.1 + x * b.1, a.2 + x * b.2) = k • (a.1 - b.1, a.2 - b.2)) :
  x = -1 := by
  sorry

end parallel_vectors_x_value_l1657_165714


namespace equation_solutions_l1657_165727

def equation (x : ℝ) : Prop :=
  x ≠ 1 ∧ x ≠ -6 → (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 3 ∨ x = -6) :=
by sorry

end equation_solutions_l1657_165727


namespace desmond_bought_240_toys_l1657_165799

/-- The number of toys Mr. Desmond bought for his elder son -/
def elder_son_toys : ℕ := 60

/-- The number of toys Mr. Desmond bought for his younger son -/
def younger_son_toys : ℕ := 3 * elder_son_toys

/-- The total number of toys Mr. Desmond bought -/
def total_toys : ℕ := elder_son_toys + younger_son_toys

theorem desmond_bought_240_toys : total_toys = 240 := by
  sorry

end desmond_bought_240_toys_l1657_165799


namespace expression_value_l1657_165730

theorem expression_value : 
  let a := 2020
  (a^3 - 3*a^2*(a+1) + 4*a*(a+1)^2 - (a+1)^3 + 1) / (a*(a+1)) = 2021 := by
  sorry

end expression_value_l1657_165730


namespace inequality_proof_l1657_165719

theorem inequality_proof (a b : ℝ) (h : a - |b| > 0) : b + a > 0 := by
  sorry

end inequality_proof_l1657_165719


namespace problem_2004_l1657_165757

theorem problem_2004 (a : ℝ) : 
  (|2004 - a| + Real.sqrt (a - 2005) = a) → (a - 2004^2 = 2005) := by
  sorry

end problem_2004_l1657_165757


namespace orange_bucket_theorem_l1657_165712

/-- Represents the number of oranges in each bucket -/
structure OrangeBuckets where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of oranges in all buckets -/
def total_oranges (buckets : OrangeBuckets) : ℕ :=
  buckets.first + buckets.second + buckets.third

/-- Theorem stating the total number of oranges in the given conditions -/
theorem orange_bucket_theorem (buckets : OrangeBuckets) 
  (h1 : buckets.first = 22)
  (h2 : buckets.second = buckets.first + 17)
  (h3 : buckets.third = buckets.second - 11) :
  total_oranges buckets = 89 := by
  sorry

#check orange_bucket_theorem

end orange_bucket_theorem_l1657_165712


namespace least_number_for_divisibility_l1657_165770

theorem least_number_for_divisibility (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ (1101 + k) % 24 = 0) → 
  (∃ m : ℕ, m ≥ 0 ∧ (1101 + m) % 24 = 0 ∧ ∀ l : ℕ, l < m → (1101 + l) % 24 ≠ 0) →
  n = 3 := by
sorry

#eval (1101 + 3) % 24  -- This should evaluate to 0

end least_number_for_divisibility_l1657_165770


namespace function_equality_l1657_165732

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 7
def g (k : ℝ) (x : ℝ) : ℝ := x^2 - k * x + 5

-- State the theorem
theorem function_equality (k : ℝ) : f 5 - g k 5 = 0 → k = -92 / 5 := by
  sorry

end function_equality_l1657_165732


namespace definite_integral_3x_plus_sin_x_l1657_165737

theorem definite_integral_3x_plus_sin_x : 
  ∫ x in (0)..(π/2), (3*x + Real.sin x) = (3*π^2)/8 + 1 := by
  sorry

end definite_integral_3x_plus_sin_x_l1657_165737


namespace range_of_m_l1657_165739

theorem range_of_m (m : ℝ) : 
  (∀ x, (1 ≤ x ∧ x ≤ 3) → (m + 1 ≤ x ∧ x ≤ 2*m + 7)) ∧ 
  (∃ x, m + 1 ≤ x ∧ x ≤ 2*m + 7 ∧ ¬(1 ≤ x ∧ x ≤ 3)) → 
  -2 ≤ m ∧ m ≤ 0 :=
sorry

end range_of_m_l1657_165739


namespace power_equation_l1657_165773

theorem power_equation (a m n : ℝ) (hm : a^m = 3) (hn : a^n = 4) : a^(2*m + 3*n) = 576 := by
  sorry

end power_equation_l1657_165773


namespace total_apples_formula_l1657_165703

/-- Represents the number of apples each person has -/
structure AppleCount where
  sarah : ℕ
  jackie : ℕ
  adam : ℕ

/-- Calculates the total number of apples -/
def totalApples (count : AppleCount) : ℕ :=
  count.sarah + count.jackie + count.adam

/-- Theorem: The total number of apples is 5X + 5, where X is Sarah's apple count -/
theorem total_apples_formula (X : ℕ) : 
  ∀ (count : AppleCount), 
    count.sarah = X → 
    count.jackie = 2 * X → 
    count.adam = count.jackie + 5 → 
    totalApples count = 5 * X + 5 := by
  sorry

end total_apples_formula_l1657_165703


namespace sphere_radii_ratio_l1657_165707

theorem sphere_radii_ratio (V1 V2 r1 r2 : ℝ) :
  V1 = 450 * Real.pi →
  V2 = 36 * Real.pi →
  V2 / V1 = (r2 / r1) ^ 3 →
  r2 / r1 = Real.rpow 2 (1/3) / 5 := by
sorry

end sphere_radii_ratio_l1657_165707


namespace arithmetic_sequence_k_value_l1657_165790

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_k_value
  (a : ℕ → ℝ) (d : ℝ) (k : ℕ) 
  (h1 : d ≠ 0)
  (h2 : arithmetic_sequence a d)
  (h3 : a 1 = 4 * d)
  (h4 : a k ^ 2 = a 1 * a (2 * k)) :
  k = 3 := by
  sorry

end arithmetic_sequence_k_value_l1657_165790


namespace expression_equality_l1657_165733

theorem expression_equality : 2 * Real.sin (π / 3) + Real.sqrt 12 + abs (-5) - (π - Real.sqrt 2) ^ 0 = 3 * Real.sqrt 3 + 4 := by
  sorry

end expression_equality_l1657_165733


namespace shop_profit_days_l1657_165753

theorem shop_profit_days (mean_profit : ℝ) (first_15_mean : ℝ) (last_15_mean : ℝ)
  (h1 : mean_profit = 350)
  (h2 : first_15_mean = 255)
  (h3 : last_15_mean = 445) :
  ∃ (total_days : ℕ), 
    total_days = 30 ∧ 
    (first_15_mean * 15 + last_15_mean * 15 : ℝ) = mean_profit * total_days :=
by
  sorry

end shop_profit_days_l1657_165753


namespace carrots_picked_next_day_l1657_165740

theorem carrots_picked_next_day (initial_carrots : ℕ) (thrown_out : ℕ) (total_carrots : ℕ) : 
  initial_carrots = 48 → thrown_out = 11 → total_carrots = 52 →
  total_carrots - (initial_carrots - thrown_out) = 15 := by
  sorry

end carrots_picked_next_day_l1657_165740


namespace girls_grades_l1657_165736

theorem girls_grades (M L S : ℕ) 
  (h1 : M + L = 23)
  (h2 : S + M = 18)
  (h3 : S + L = 15) :
  M = 13 ∧ L = 10 ∧ S = 5 := by
  sorry

end girls_grades_l1657_165736


namespace karl_garden_larger_l1657_165762

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (g : GardenDimensions) : ℝ :=
  g.length * g.width

/-- Theorem: Karl's garden is larger than Makenna's usable garden area by 150 square feet -/
theorem karl_garden_larger (karl : GardenDimensions) (makenna : GardenDimensions) 
  (h1 : karl.length = 30 ∧ karl.width = 50)
  (h2 : makenna.length = 35 ∧ makenna.width = 45)
  (path_width : ℝ) (h3 : path_width = 5) : 
  gardenArea karl - (gardenArea makenna - path_width * makenna.length) = 150 := by
  sorry

#check karl_garden_larger

end karl_garden_larger_l1657_165762


namespace max_value_of_expression_l1657_165742

theorem max_value_of_expression (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d ≤ 4) :
  (2 * a^2 + a^2 * b)^(1/4) + (2 * b^2 + b^2 * c)^(1/4) + 
  (2 * c^2 + c^2 * d)^(1/4) + (2 * d^2 + d^2 * a)^(1/4) ≤ 4 * (3^(1/4)) :=
by sorry

end max_value_of_expression_l1657_165742


namespace mean_of_class_scores_l1657_165780

def class_scores : List ℕ := [50, 57, 49, 57, 32, 46, 65, 28]

theorem mean_of_class_scores : 
  (List.sum class_scores) / (List.length class_scores) = 48 := by
  sorry

end mean_of_class_scores_l1657_165780


namespace effective_discount_l1657_165765

theorem effective_discount (initial_discount coupon_discount : ℝ) : 
  initial_discount = 0.6 →
  coupon_discount = 0.3 →
  let sale_price := 1 - initial_discount
  let final_price := sale_price * (1 - coupon_discount)
  1 - final_price = 0.72 :=
by sorry

end effective_discount_l1657_165765


namespace trigonometric_identity_l1657_165713

theorem trigonometric_identity : 4 * Real.sin (20 * π / 180) + Real.tan (20 * π / 180) = Real.sqrt 3 := by sorry

end trigonometric_identity_l1657_165713


namespace total_ingredients_l1657_165749

def strawberries : ℚ := 0.2
def yogurt : ℚ := 0.1
def orange_juice : ℚ := 0.2

theorem total_ingredients : strawberries + yogurt + orange_juice = 0.5 := by
  sorry

end total_ingredients_l1657_165749


namespace simple_interest_problem_l1657_165706

/-- Proves that for a principal of 1000 Rs., if increasing the interest rate by 3%
    results in 90 Rs. more interest, then the time period for which the sum was invested is 3 years. -/
theorem simple_interest_problem (R : ℝ) (T : ℝ) :
  (1000 * R * T / 100 + 90 = 1000 * (R + 3) * T / 100) →
  T = 3 := by
  sorry

end simple_interest_problem_l1657_165706


namespace geometric_sequence_general_term_l1657_165791

theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = 4 * a n) 
  (h_sum : a 1 + a 2 + a 3 = 21) :
  ∀ n : ℕ, a n = 4^(n - 1) := by
sorry

end geometric_sequence_general_term_l1657_165791


namespace investment_plans_count_l1657_165748

/-- The number of ways to distribute 3 distinct objects into 4 distinct containers,
    with no container holding more than 2 objects. -/
def investmentPlans : ℕ :=
  let numProjects : ℕ := 3
  let numCities : ℕ := 4
  let maxProjectsPerCity : ℕ := 2
  -- The actual calculation is not implemented here
  60

/-- Theorem stating that the number of investment plans is 60 -/
theorem investment_plans_count : investmentPlans = 60 := by
  sorry

end investment_plans_count_l1657_165748


namespace vault_code_thickness_l1657_165723

/-- Thickness of an Alpha card in millimeters -/
def alpha_thickness : ℚ := 1.65

/-- Thickness of a Beta card in millimeters -/
def beta_thickness : ℚ := 2.05

/-- Thickness of a Gamma card in millimeters -/
def gamma_thickness : ℚ := 1.25

/-- Thickness of a Delta card in millimeters -/
def delta_thickness : ℚ := 1.85

/-- Total thickness of the stack in millimeters -/
def total_thickness : ℚ := 15.6

/-- The number of cards in the stack -/
def num_cards : ℕ := 8

theorem vault_code_thickness :
  num_cards * delta_thickness = total_thickness ∧
  ∀ (a b c d : ℕ), 
    a * alpha_thickness + b * beta_thickness + c * gamma_thickness + d * delta_thickness = total_thickness →
    a = 0 ∧ b = 0 ∧ c = 0 ∧ d = num_cards :=
by sorry

end vault_code_thickness_l1657_165723


namespace divisor_problem_l1657_165768

theorem divisor_problem (n m : ℕ) (h1 : n = 987654) (h2 : m = 42) : 
  (n + m) % m = 0 := by
sorry

end divisor_problem_l1657_165768


namespace sufficient_not_necessary_condition_l1657_165797

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) := by
  sorry

end sufficient_not_necessary_condition_l1657_165797


namespace min_containers_to_fill_jumbo_l1657_165772

/-- The volume of a regular size container in milliliters -/
def regular_container_volume : ℕ := 75

/-- The volume of a jumbo container in milliliters -/
def jumbo_container_volume : ℕ := 1800

/-- The minimum number of regular size containers needed to fill a jumbo container -/
def min_containers : ℕ := (jumbo_container_volume + regular_container_volume - 1) / regular_container_volume

theorem min_containers_to_fill_jumbo : min_containers = 24 := by
  sorry

end min_containers_to_fill_jumbo_l1657_165772


namespace solution_of_equations_l1657_165752

theorem solution_of_equations (x : ℝ) : 
  (|x|^2 - 5*|x| + 6 = 0 ∧ x^2 - 4 = 0) ↔ (x = 2 ∨ x = -2) :=
by sorry

end solution_of_equations_l1657_165752


namespace largest_a_value_l1657_165729

theorem largest_a_value (a : ℝ) :
  (5 * Real.sqrt ((3 * a)^2 + 2^2) - 5 * a^2 - 2) / (Real.sqrt (2 + 3 * a^2) + 2) = 1 →
  ∃ y : ℝ, y^2 - (5 * Real.sqrt 3 - 1) * y + 5 = 0 ∧
           y ≥ (5 * Real.sqrt 3 - 1 + Real.sqrt ((5 * Real.sqrt 3 - 1)^2 - 20)) / 2 ∧
           a = Real.sqrt ((y^2 - 2) / 3) ∧
           ∀ a' : ℝ, (5 * Real.sqrt ((3 * a')^2 + 2^2) - 5 * a'^2 - 2) / (Real.sqrt (2 + 3 * a'^2) + 2) = 1 →
                     a' ≤ a :=
by sorry

end largest_a_value_l1657_165729


namespace p_arithmetic_fibonacci_property_l1657_165725

/-- Definition of p-arithmetic Fibonacci sequence -/
def PArithmeticFibonacci (p : ℕ) (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + p) = v n + v (n + 1)

/-- Theorem: For any p-arithmetic Fibonacci sequence, vₖ + vₖ₊ₚ = vₖ₊₂ₚ holds for all k -/
theorem p_arithmetic_fibonacci_property {p : ℕ} {v : ℕ → ℝ} 
  (hv : PArithmeticFibonacci p v) :
  ∀ k, v k + v (k + p) = v (k + 2 * p) := by
  sorry

end p_arithmetic_fibonacci_property_l1657_165725


namespace sqrt_inequality_l1657_165750

theorem sqrt_inequality (x : ℝ) (h : x ≥ 4) :
  Real.sqrt (x - 3) - Real.sqrt (x - 1) > Real.sqrt (x - 4) - Real.sqrt (x - 2) := by
  sorry

end sqrt_inequality_l1657_165750


namespace sqrt_expressions_l1657_165771

theorem sqrt_expressions (a b : ℝ) (h1 : a = Real.sqrt 5 + Real.sqrt 3) (h2 : b = Real.sqrt 5 - Real.sqrt 3) :
  (a + b = 2 * Real.sqrt 5) ∧ (a * b = 2) ∧ (a^2 + a*b + b^2 = 18) := by
  sorry

end sqrt_expressions_l1657_165771


namespace new_shoes_average_speed_l1657_165760

/-- Calculate the average speed of a hiker using new high-tech shoes over a 4-hour hike -/
theorem new_shoes_average_speed
  (old_speed : ℝ)
  (new_speed_multiplier : ℝ)
  (hike_duration : ℝ)
  (blister_interval : ℝ)
  (speed_reduction_per_blister : ℝ)
  (h_old_speed : old_speed = 6)
  (h_new_speed_multiplier : new_speed_multiplier = 2)
  (h_hike_duration : hike_duration = 4)
  (h_blister_interval : blister_interval = 2)
  (h_speed_reduction : speed_reduction_per_blister = 2)
  : (old_speed * new_speed_multiplier + 
     (old_speed * new_speed_multiplier - speed_reduction_per_blister)) / 2 = 11 := by
  sorry

end new_shoes_average_speed_l1657_165760


namespace problem_statement_l1657_165796

noncomputable section

def f (x : ℝ) := Real.exp x * Real.sin x - Real.cos x
def g (x : ℝ) := x * Real.cos x - Real.sqrt 2 * Real.exp x

theorem problem_statement :
  (∀ m : ℝ, (∀ x₁ ∈ Set.Icc 0 (Real.pi / 2), ∃ x₂ ∈ Set.Icc 0 (Real.pi / 2), f x₁ + g x₂ ≥ m) → m ≤ -1 - Real.sqrt 2) ∧
  (∀ x > -1, f x - g x > 0) :=
by sorry

end

end problem_statement_l1657_165796


namespace translated_min_point_l1657_165702

/-- The original function before translation -/
def f (x : ℝ) : ℝ := |x + 1| - 4

/-- The translated function -/
def g (x : ℝ) : ℝ := f (x - 3) - 4

/-- The minimum point of the translated function -/
def min_point : ℝ × ℝ := (2, -8)

theorem translated_min_point :
  (∀ x : ℝ, g x ≥ g (min_point.1)) ∧
  g (min_point.1) = min_point.2 :=
sorry

end translated_min_point_l1657_165702


namespace three_odd_factors_is_nine_l1657_165720

theorem three_odd_factors_is_nine :
  ∃! n : ℕ, n > 1 ∧ (∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧
    Odd a ∧ Odd b ∧ Odd c ∧
    {d : ℕ | d > 1 ∧ d ∣ n ∧ Odd d} = {a, b, c}) :=
by
  sorry

end three_odd_factors_is_nine_l1657_165720


namespace binomial_10_5_l1657_165754

theorem binomial_10_5 : Nat.choose 10 5 = 252 := by
  sorry

end binomial_10_5_l1657_165754


namespace overall_profit_percentage_l1657_165782

def apples : ℝ := 280
def oranges : ℝ := 150
def bananas : ℝ := 100

def apples_high_profit_ratio : ℝ := 0.4
def oranges_high_profit_ratio : ℝ := 0.45
def bananas_high_profit_ratio : ℝ := 0.5

def apples_high_profit_percentage : ℝ := 0.2
def oranges_high_profit_percentage : ℝ := 0.25
def bananas_high_profit_percentage : ℝ := 0.3

def low_profit_percentage : ℝ := 0.15

def total_fruits : ℝ := apples + oranges + bananas

theorem overall_profit_percentage (ε : ℝ) (h : ε > 0) :
  ∃ (profit_percentage : ℝ),
    abs (profit_percentage - 0.1875) < ε ∧
    profit_percentage = 
      (apples_high_profit_ratio * apples * apples_high_profit_percentage +
       oranges_high_profit_ratio * oranges * oranges_high_profit_percentage +
       bananas_high_profit_ratio * bananas * bananas_high_profit_percentage +
       (1 - apples_high_profit_ratio) * apples * low_profit_percentage +
       (1 - oranges_high_profit_ratio) * oranges * low_profit_percentage +
       (1 - bananas_high_profit_ratio) * bananas * low_profit_percentage) /
      total_fruits :=
by sorry

end overall_profit_percentage_l1657_165782


namespace line_slope_intercept_product_l1657_165779

theorem line_slope_intercept_product (m b : ℚ) : 
  m = 3/5 → b = -3/2 → -1 < m * b ∧ m * b < 0 := by sorry

end line_slope_intercept_product_l1657_165779


namespace min_m_value_l1657_165728

theorem min_m_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a > b) (hbc : b > c) (hcd : c > d) :
  ∃ (m : ℝ), m = 9 ∧ 
  (∀ (k : ℝ), k < 9 → 
    ∃ (x y z w : ℝ), x > y ∧ y > z ∧ z > w ∧ w > 0 ∧
      Real.log (2004 : ℝ) / Real.log (y / x) + 
      Real.log (2004 : ℝ) / Real.log (z / y) + 
      Real.log (2004 : ℝ) / Real.log (w / z) < 
      k * (Real.log (2004 : ℝ) / Real.log (w / x))) ∧
  (∀ (a' b' c' d' : ℝ), a' > b' ∧ b' > c' ∧ c' > d' ∧ d' > 0 →
    Real.log (2004 : ℝ) / Real.log (b' / a') + 
    Real.log (2004 : ℝ) / Real.log (c' / b') + 
    Real.log (2004 : ℝ) / Real.log (d' / c') ≥ 
    9 * (Real.log (2004 : ℝ) / Real.log (d' / a'))) := by
  sorry

end min_m_value_l1657_165728


namespace hyperbola_focus_l1657_165763

/-- Given a hyperbola with equation ((x-5)^2)/7^2 - ((y-20)^2)/15^2 = 1,
    the focus with the larger x-coordinate has coordinates (5 + √274, 20) -/
theorem hyperbola_focus (x y : ℝ) :
  ((x - 5)^2 / 7^2) - ((y - 20)^2 / 15^2) = 1 →
  ∃ (f_x f_y : ℝ), f_x > 5 ∧ f_y = 20 ∧ f_x = 5 + Real.sqrt 274 ∧
  ∀ (x' y' : ℝ), ((x' - 5)^2 / 7^2) - ((y' - 20)^2 / 15^2) = 1 →
  (x' - 5)^2 / 7^2 + (y' - 20)^2 / 15^2 = (x' - f_x)^2 / 7^2 + (y' - f_y)^2 / 15^2 :=
by sorry

end hyperbola_focus_l1657_165763


namespace no_solution_l1657_165769

/-- Q(n) denotes the greatest prime factor of n -/
def Q (n : ℕ) : ℕ := sorry

/-- The theorem states that there are no positive integers n > 1 satisfying
    both Q(n) = √n and Q(3n + 16) = √(3n + 16) -/
theorem no_solution :
  ¬ ∃ (n : ℕ), n > 1 ∧ 
    Q n = Nat.sqrt n ∧ 
    Q (3 * n + 16) = Nat.sqrt (3 * n + 16) := by
  sorry

end no_solution_l1657_165769


namespace min_value_inequality_sum_squared_ratio_inequality_l1657_165789

-- Part 1
theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 4 / b ≥ 9 := by sorry

-- Part 2
theorem sum_squared_ratio_inequality (a b c m : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (habc : a + b + c = m) :
  a^2 / b + b^2 / c + c^2 / a ≥ m := by sorry

end min_value_inequality_sum_squared_ratio_inequality_l1657_165789


namespace arithmetic_sequence_middle_term_l1657_165795

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_first : a 0 = 23)
  (h_last : a 4 = 53) :
  a 2 = 38 := by
sorry

end arithmetic_sequence_middle_term_l1657_165795


namespace sqrt_16_equals_4_l1657_165776

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_16_equals_4_l1657_165776


namespace square_perimeter_ratio_l1657_165758

theorem square_perimeter_ratio (s1 s2 : ℝ) (h : s1^2 / s2^2 = 16 / 81) :
  (4 * s1) / (4 * s2) = 4 / 9 := by
sorry

end square_perimeter_ratio_l1657_165758


namespace abs_z_eq_one_l1657_165761

-- Define the complex number z
variable (z : ℂ)

-- Define the real number a
variable (a : ℝ)

-- Define the condition on a
axiom a_lt_one : a < 1

-- Define the equation that z satisfies
axiom z_equation : (a - 2) * z^2018 + a * z^2017 * Complex.I + a * z * Complex.I + 2 - a = 0

-- Theorem to prove
theorem abs_z_eq_one : Complex.abs z = 1 := by sorry

end abs_z_eq_one_l1657_165761


namespace difference_proof_l1657_165716

/-- Given a total number of students and the number of first graders,
    calculate the difference between second graders and first graders. -/
def difference_between_grades (total : ℕ) (first_graders : ℕ) : ℕ :=
  (total - first_graders) - first_graders

theorem difference_proof :
  difference_between_grades 95 32 = 31 :=
by sorry

end difference_proof_l1657_165716


namespace right_triangle_increase_sides_acute_l1657_165788

theorem right_triangle_increase_sides_acute (a b c x : ℝ) :
  a > 0 → b > 0 → c > 0 → x > 0 →
  c^2 = a^2 + b^2 →  -- right-angled triangle condition
  (a + x)^2 + (b + x)^2 > (c + x)^2  -- acute triangle condition
  := by sorry

end right_triangle_increase_sides_acute_l1657_165788


namespace cos_double_angle_specific_l1657_165798

theorem cos_double_angle_specific (α : Real) :
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →
  Real.cos (2 * α) = -(Real.sqrt 5 / 3) :=
by sorry

end cos_double_angle_specific_l1657_165798


namespace tourist_guide_distribution_l1657_165711

/-- The number of ways to distribute n tourists among k guides, 
    where each guide must have at least one tourist -/
def validDistributions (n k : ℕ) : ℕ :=
  k^n - (k.choose 1) * (k-1)^n + (k.choose 2) * (k-2)^n

theorem tourist_guide_distribution :
  validDistributions 8 3 = 5796 := by
  sorry

end tourist_guide_distribution_l1657_165711


namespace captain_bonus_calculation_l1657_165721

/-- The number of students in the team -/
def team_size : ℕ := 10

/-- The number of team members (excluding the captain) -/
def team_members : ℕ := 9

/-- The bonus amount for each team member -/
def member_bonus : ℕ := 200

/-- The additional amount the captain receives above the average -/
def captain_extra : ℕ := 90

/-- The bonus amount for the captain -/
def captain_bonus : ℕ := 300

theorem captain_bonus_calculation :
  captain_bonus = 
    (team_members * member_bonus + captain_bonus) / team_size + captain_extra := by
  sorry

#check captain_bonus_calculation

end captain_bonus_calculation_l1657_165721


namespace distance_between_homes_l1657_165700

/-- Proves that the distance between Maxwell's and Brad's homes is 36 km given the problem conditions -/
theorem distance_between_homes : 
  ∀ (maxwell_speed brad_speed maxwell_distance : ℝ),
    maxwell_speed = 2 →
    brad_speed = 4 →
    maxwell_distance = 12 →
    maxwell_distance + maxwell_distance * (brad_speed / maxwell_speed) = 36 :=
by
  sorry


end distance_between_homes_l1657_165700


namespace inequality_system_solution_range_l1657_165793

theorem inequality_system_solution_range (m : ℝ) : 
  (∀ x : ℝ, ((x - 1) / 2 ≥ (x - 2) / 3 ∧ 2 * x - m ≥ x) ↔ x ≥ m) → 
  m ≥ -1 := by
  sorry

end inequality_system_solution_range_l1657_165793


namespace hexagon_arrangement_count_l1657_165786

/-- Represents a valid arrangement of digits on a regular hexagon with center -/
structure HexagonArrangement where
  vertices : Fin 6 → Fin 7
  center : Fin 7
  all_different : ∀ i j : Fin 6, i ≠ j → vertices i ≠ vertices j
  center_different : ∀ i : Fin 6, center ≠ vertices i
  sum_equal : ∀ i : Fin 3, 
    (vertices i).val + center.val + (vertices (i + 3)).val = 
    (vertices (i + 1)).val + center.val + (vertices (i + 4)).val

/-- The number of valid hexagon arrangements -/
def count_arrangements : ℕ := sorry

/-- Theorem stating the correct number of arrangements -/
theorem hexagon_arrangement_count : count_arrangements = 144 := by sorry

end hexagon_arrangement_count_l1657_165786


namespace min_value_quadratic_with_linear_constraint_l1657_165738

theorem min_value_quadratic_with_linear_constraint :
  ∃ (min_u : ℝ), min_u = -66/13 ∧
  ∀ (x y : ℝ), 3*x + 2*y - 1 ≥ 0 →
    x^2 + y^2 + 6*x - 2*y ≥ min_u :=
by sorry

end min_value_quadratic_with_linear_constraint_l1657_165738


namespace coeff_x6_q_squared_is_16_l1657_165726

/-- The polynomial q(x) -/
def q (x : ℝ) : ℝ := x^5 - 4*x^3 + 3

/-- The coefficient of x^6 in (q(x))^2 -/
def coeff_x6_q_squared : ℝ := 16

/-- Theorem: The coefficient of x^6 in (q(x))^2 is 16 -/
theorem coeff_x6_q_squared_is_16 : coeff_x6_q_squared = 16 := by
  sorry

end coeff_x6_q_squared_is_16_l1657_165726


namespace average_age_of_four_students_l1657_165745

theorem average_age_of_four_students
  (total_students : ℕ)
  (average_age_all : ℝ)
  (num_group1 : ℕ)
  (average_age_group1 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : num_group1 = 9)
  (h4 : average_age_group1 = 16)
  (h5 : age_last_student = 25)
  : (total_students * average_age_all - num_group1 * average_age_group1 - age_last_student) / (total_students - num_group1 - 1) = 14 :=
by
  sorry

#check average_age_of_four_students

end average_age_of_four_students_l1657_165745


namespace elle_piano_practice_l1657_165781

/-- The number of minutes Elle practices piano on a weekday -/
def weekday_practice : ℕ := 30

/-- The number of minutes Elle practices piano on Saturday -/
def saturday_practice : ℕ := 3 * weekday_practice

/-- The total number of minutes Elle practices piano in a week -/
def total_practice : ℕ := 5 * weekday_practice + saturday_practice

theorem elle_piano_practice :
  weekday_practice = 30 ∧
  saturday_practice = 3 * weekday_practice ∧
  total_practice = 5 * weekday_practice + saturday_practice ∧
  total_practice = 4 * 60 := by
  sorry

end elle_piano_practice_l1657_165781


namespace max_min_sum_theorem_l1657_165764

/-- A function satisfying the given property -/
def FunctionProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2014

/-- The function g defined in terms of f -/
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2014 * x^2013

theorem max_min_sum_theorem (f : ℝ → ℝ) (hf : FunctionProperty f)
  (hM : ∃ M : ℝ, ∀ x : ℝ, g f x ≤ M)
  (hm : ∃ m : ℝ, ∀ x : ℝ, m ≤ g f x)
  (hMm : ∃ M m : ℝ, (∀ x : ℝ, g f x ≤ M ∧ m ≤ g f x) ∧ 
    (∃ x1 x2 : ℝ, g f x1 = M ∧ g f x2 = m)) :
  ∃ M m : ℝ, (∀ x : ℝ, g f x ≤ M ∧ m ≤ g f x) ∧ 
    (∃ x1 x2 : ℝ, g f x1 = M ∧ g f x2 = m) ∧ M + m = -4028 :=
sorry

end max_min_sum_theorem_l1657_165764


namespace distance_traveled_l1657_165743

/-- Calculates the total distance traveled given two speeds and two durations -/
def total_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Theorem: The total distance traveled is 255 miles -/
theorem distance_traveled : total_distance 45 2 55 3 = 255 := by
  sorry

end distance_traveled_l1657_165743


namespace positive_integer_triplets_l1657_165747

theorem positive_integer_triplets (a b c : ℕ+) :
  (a ^ b.val ∣ b ^ c.val - 1) ∧ (a ^ c.val ∣ c ^ b.val - 1) →
  (a = 1) ∨ (b = 1 ∧ c = 1) := by
  sorry

end positive_integer_triplets_l1657_165747


namespace actual_tissue_diameter_l1657_165785

/-- The actual diameter of a circular tissue given its magnification and magnified image diameter -/
theorem actual_tissue_diameter 
  (magnification : ℝ) 
  (magnified_diameter : ℝ) 
  (h_magnification : magnification = 1000) 
  (h_magnified_diameter : magnified_diameter = 1) : 
  magnified_diameter / magnification = 0.001 := by
  sorry

end actual_tissue_diameter_l1657_165785


namespace specific_trapezoid_height_l1657_165744

/-- A trapezoid with given dimensions -/
structure Trapezoid where
  leg1 : ℝ
  leg2 : ℝ
  base1 : ℝ
  base2 : ℝ

/-- The height of a trapezoid -/
def trapezoidHeight (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating the height of the specific trapezoid -/
theorem specific_trapezoid_height :
  let t : Trapezoid := { leg1 := 6, leg2 := 8, base1 := 4, base2 := 14 }
  trapezoidHeight t = 4.8 := by
  sorry

end specific_trapezoid_height_l1657_165744


namespace opposite_of_three_l1657_165751

theorem opposite_of_three : ∃ x : ℤ, x + 3 = 0 ∧ x = -3 := by
  sorry

end opposite_of_three_l1657_165751


namespace probability_of_white_coverage_l1657_165704

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of the central white rectangle -/
structure CentralRectangle where
  length : ℝ
  width : ℝ

/-- Represents the circular sheet used to cover the field -/
structure CircularSheet where
  diameter : ℝ

/-- Calculates the probability of covering white area -/
def probability_of_covering_white (field : FieldDimensions) (central : CentralRectangle) (sheet : CircularSheet) (num_circles : ℕ) (circle_radius : ℝ) : ℝ :=
  sorry

/-- The main theorem stating the probability -/
theorem probability_of_white_coverage :
  let field := FieldDimensions.mk 12 10
  let central := CentralRectangle.mk 4 2
  let sheet := CircularSheet.mk 1.5
  let num_circles := 5
  let circle_radius := 1
  abs (probability_of_covering_white field central sheet num_circles circle_radius - 0.647) < 0.001 := by
  sorry

end probability_of_white_coverage_l1657_165704


namespace smart_number_characterization_smart_number_2015_l1657_165746

/-- A positive integer is a smart number if it can be expressed as the difference of squares of two positive integers. -/
def is_smart_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > b ∧ n = a ^ 2 - b ^ 2

/-- Theorem stating the characterization of smart numbers -/
theorem smart_number_characterization (n : ℕ) :
  is_smart_number n ↔ (n > 1 ∧ n % 2 = 1) ∨ (n ≥ 8 ∧ n % 4 = 0) :=
sorry

/-- Function to get the nth smart number -/
def nth_smart_number (n : ℕ) : ℕ :=
sorry

/-- Theorem stating that the 2015th smart number is 2689 -/
theorem smart_number_2015 : nth_smart_number 2015 = 2689 :=
sorry

end smart_number_characterization_smart_number_2015_l1657_165746


namespace tripod_new_height_l1657_165709

/-- Represents a tripod with given parameters -/
structure Tripod where
  leg_length : ℝ
  initial_height : ℝ
  sink_depth : ℝ

/-- Calculates the new height of a tripod after one leg sinks -/
noncomputable def new_height (t : Tripod) : ℝ :=
  144 / Real.sqrt 262.2

/-- Theorem stating the new height of the tripod after one leg sinks -/
theorem tripod_new_height (t : Tripod) 
  (h_leg : t.leg_length = 8)
  (h_init : t.initial_height = 6)
  (h_sink : t.sink_depth = 2) :
  new_height t = 144 / Real.sqrt 262.2 := by
  sorry

end tripod_new_height_l1657_165709


namespace negation_of_universal_proposition_l1657_165710

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l1657_165710


namespace quadratic_has_real_root_l1657_165741

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end quadratic_has_real_root_l1657_165741


namespace log_sum_equals_one_l1657_165774

theorem log_sum_equals_one : Real.log 2 + 2 * Real.log (Real.sqrt 5) = Real.log 10 := by
  sorry

end log_sum_equals_one_l1657_165774


namespace negation_of_proposition_sin_reciprocal_inequality_l1657_165735

open Real

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x ∈ (Set.Ioo 0 π), p x) ↔ (∃ x ∈ (Set.Ioo 0 π), ¬ p x) := by sorry

theorem sin_reciprocal_inequality :
  (¬ ∀ x ∈ (Set.Ioo 0 π), sin x + (1 / sin x) > 2) ↔
  (∃ x ∈ (Set.Ioo 0 π), sin x + (1 / sin x) ≤ 2) := by sorry

end negation_of_proposition_sin_reciprocal_inequality_l1657_165735


namespace lingonberry_price_theorem_l1657_165731

/-- The price per pound of lingonberries picked -/
def price_per_pound : ℚ := 2

/-- The total amount Steve wants to make -/
def total_amount : ℚ := 100

/-- The amount of lingonberries picked on Monday -/
def monday_picked : ℚ := 8

/-- The amount of lingonberries picked on Tuesday -/
def tuesday_picked : ℚ := 3 * monday_picked

/-- The amount of lingonberries picked on Wednesday -/
def wednesday_picked : ℚ := 0

/-- The amount of lingonberries picked on Thursday -/
def thursday_picked : ℚ := 18

/-- The total amount of lingonberries picked over four days -/
def total_picked : ℚ := monday_picked + tuesday_picked + wednesday_picked + thursday_picked

theorem lingonberry_price_theorem : 
  price_per_pound * total_picked = total_amount :=
by sorry

end lingonberry_price_theorem_l1657_165731


namespace marble_boxes_theorem_l1657_165766

/-- Given a number of marbles per box and a total number of marbles,
    calculate the number of boxes. -/
def number_of_boxes (marbles_per_box : ℕ) (total_marbles : ℕ) : ℕ :=
  total_marbles / marbles_per_box

/-- Theorem stating that with 6 marbles per box and 18 total marbles,
    the number of boxes is 3. -/
theorem marble_boxes_theorem :
  number_of_boxes 6 18 = 3 := by
  sorry

end marble_boxes_theorem_l1657_165766


namespace quadratic_roots_max_value_l1657_165783

theorem quadratic_roots_max_value (s p r₁ : ℝ) (h1 : r₁ ≠ 0) : 
  (r₁ + (-r₁) = 0) → 
  (r₁ * (-r₁) = p) → 
  (∀ (n : ℕ), n ≤ 2005 → r₁^(2*n) + (-r₁)^(2*n) = 2 * r₁^(2*n)) →
  (∃ (x : ℝ), x^2 - s*x + p = 0) →
  (∀ (y : ℝ), (1 / r₁^2006) + (1 / (-r₁)^2006) ≤ y) →
  y = 2 := by sorry

end quadratic_roots_max_value_l1657_165783


namespace color_coded_figure_areas_l1657_165708

/-- A figure composed of squares with color-coded parts -/
structure ColorCodedFigure where
  /-- The side length of each square in the figure -/
  square_side : ℝ
  /-- The total number of squares in the figure -/
  total_squares : ℕ
  /-- The area of the black part of the figure -/
  black_area : ℝ

/-- Theorem stating the areas of the remaining parts in the figure -/
theorem color_coded_figure_areas (fig : ColorCodedFigure)
  (h1 : fig.total_squares = 8)
  (h2 : fig.square_side ^ 2 * fig.total_squares = 24)
  (h3 : fig.black_area = 7.5) :
  ∃ (white dark_gray light_gray shaded : ℝ),
    white = 1.5 ∧
    dark_gray = 6 ∧
    light_gray = 5.25 ∧
    shaded = 3.75 ∧
    white + dark_gray + light_gray + shaded + fig.black_area = fig.square_side ^ 2 * fig.total_squares :=
by sorry

end color_coded_figure_areas_l1657_165708


namespace perimeter_of_figure_C_l1657_165787

/-- Represents the dimensions of a rectangle in terms of small rectangles -/
structure RectangleDimension where
  width : ℕ
  height : ℕ

/-- Calculates the perimeter of a rectangle given its dimensions and the size of small rectangles -/
def calculatePerimeter (dim : RectangleDimension) (x y : ℝ) : ℝ :=
  2 * (dim.width * x + dim.height * y)

theorem perimeter_of_figure_C (x y : ℝ) : 
  calculatePerimeter ⟨6, 1⟩ x y = 56 →
  calculatePerimeter ⟨4, 3⟩ x y = 56 →
  calculatePerimeter ⟨2, 3⟩ x y = 40 := by
  sorry

end perimeter_of_figure_C_l1657_165787


namespace denominator_of_0_27_repeating_l1657_165756

def repeating_decimal_to_fraction (a b : ℕ) : ℚ :=
  (a : ℚ) / (99 : ℚ)

theorem denominator_of_0_27_repeating :
  (repeating_decimal_to_fraction 27 2).den = 11 := by
  sorry

end denominator_of_0_27_repeating_l1657_165756


namespace driver_net_pay_rate_l1657_165724

/-- Calculates the net rate of pay for a driver given travel conditions and expenses. -/
theorem driver_net_pay_rate
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (earnings_rate : ℝ)
  (gasoline_cost : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 60)
  (h3 : fuel_efficiency = 30)
  (h4 : earnings_rate = 0.75)
  (h5 : gasoline_cost = 3) :
  let distance := travel_time * speed
  let fuel_used := distance / fuel_efficiency
  let earnings := distance * earnings_rate
  let fuel_expense := fuel_used * gasoline_cost
  let net_earnings := earnings - fuel_expense
  net_earnings / travel_time = 39 := by
sorry

end driver_net_pay_rate_l1657_165724


namespace quadratic_solution_fractional_no_solution_l1657_165718

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 4*x - 4 = 0

-- Define the fractional equation
def fractional_equation (x : ℝ) : Prop := x / (x - 2) + 3 = (x - 4) / (2 - x)

-- Theorem for the quadratic equation
theorem quadratic_solution :
  ∃ x₁ x₂ : ℝ, 
    (x₁ = 2 * Real.sqrt 2 - 2 ∧ quadratic_equation x₁) ∧
    (x₂ = -2 * Real.sqrt 2 - 2 ∧ quadratic_equation x₂) ∧
    (∀ x : ℝ, quadratic_equation x → x = x₁ ∨ x = x₂) :=
sorry

-- Theorem for the fractional equation
theorem fractional_no_solution :
  ¬∃ x : ℝ, fractional_equation x :=
sorry

end quadratic_solution_fractional_no_solution_l1657_165718


namespace remainder_problem_l1657_165701

theorem remainder_problem (x : ℤ) (h : x % 7 = 5) : (4 * x) % 7 = 6 := by
  sorry

end remainder_problem_l1657_165701


namespace subtraction_of_one_and_two_l1657_165734

theorem subtraction_of_one_and_two : 1 - 2 = -1 := by
  sorry

end subtraction_of_one_and_two_l1657_165734


namespace only_100_not_sum_of_four_consecutive_odds_l1657_165755

def is_sum_of_four_consecutive_odds (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 4 * k + 12 ∧ k % 2 = 1

theorem only_100_not_sum_of_four_consecutive_odds :
  ¬ is_sum_of_four_consecutive_odds 100 ∧
  (is_sum_of_four_consecutive_odds 16 ∧
   is_sum_of_four_consecutive_odds 40 ∧
   is_sum_of_four_consecutive_odds 72 ∧
   is_sum_of_four_consecutive_odds 200) :=
by sorry

end only_100_not_sum_of_four_consecutive_odds_l1657_165755


namespace ellipse_and_line_property_l1657_165767

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a line -/
structure Line where
  k : ℝ
  b : ℝ
  h : k ≠ 0 ∧ b ≠ 0

/-- Given conditions of the problem -/
axiom ellipse_condition (C : Ellipse) : 
  C.a^2 - C.b^2 = 4 ∧ 2/C.a^2 + 3/C.b^2 = 1

/-- The theorem to be proved -/
theorem ellipse_and_line_property (C : Ellipse) (l : Line) :
  (∀ x y, x^2/C.a^2 + y^2/C.b^2 = 1 ↔ x^2/8 + y^2/4 = 1) ∧
  (∃ x₁ y₁ x₂ y₂, 
    x₁^2/8 + y₁^2/4 = 1 ∧
    x₂^2/8 + y₂^2/4 = 1 ∧
    y₁ = l.k * x₁ + l.b ∧
    y₂ = l.k * x₂ + l.b ∧
    x₁ ≠ x₂ ∧
    let xₘ := (x₁ + x₂)/2
    let yₘ := (y₁ + y₂)/2
    (yₘ / xₘ) * l.k = -1/2) := by sorry

end ellipse_and_line_property_l1657_165767


namespace subtract_fractions_l1657_165778

theorem subtract_fractions : (5 : ℚ) / 9 - (1 : ℚ) / 6 = (7 : ℚ) / 18 := by
  sorry

end subtract_fractions_l1657_165778
