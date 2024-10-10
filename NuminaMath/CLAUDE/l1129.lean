import Mathlib

namespace tan_difference_of_angles_l1129_112993

theorem tan_difference_of_angles (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α - Real.sin β = -1/2) (h4 : Real.cos α - Real.cos β = 1/2) :
  Real.tan (α - β) = -Real.sqrt 7 / 3 := by
  sorry

end tan_difference_of_angles_l1129_112993


namespace marcy_makeup_count_l1129_112980

/-- The number of people Marcy can paint with one tube of lip gloss -/
def people_per_tube : ℕ := 3

/-- The number of tubs of lip gloss Marcy brings -/
def tubs : ℕ := 6

/-- The number of tubes of lip gloss in each tub -/
def tubes_per_tub : ℕ := 2

/-- The total number of people Marcy is painting with makeup -/
def total_people : ℕ := tubs * tubes_per_tub * people_per_tube

theorem marcy_makeup_count : total_people = 36 := by
  sorry

end marcy_makeup_count_l1129_112980


namespace cube_volume_from_lateral_area_l1129_112948

/-- 
Given a cube with lateral surface area of 100 square units, 
prove that its volume is 125 cubic units.
-/
theorem cube_volume_from_lateral_area : 
  ∀ s : ℝ, 
  (4 * s^2 = 100) → 
  (s^3 = 125) :=
by
  sorry

end cube_volume_from_lateral_area_l1129_112948


namespace q_plus_r_at_one_eq_neg_47_l1129_112996

/-- The polynomial f(x) = 3x^5 + 4x^4 - 5x^3 + 2x^2 + x + 6 -/
def f (x : ℝ) : ℝ := 3*x^5 + 4*x^4 - 5*x^3 + 2*x^2 + x + 6

/-- The polynomial d(x) = x^3 + 2x^2 - x - 3 -/
def d (x : ℝ) : ℝ := x^3 + 2*x^2 - x - 3

/-- The existence of polynomials q and r satisfying the division algorithm -/
axiom exists_q_r : ∃ (q r : ℝ → ℝ), ∀ x, f x = q x * d x + r x

/-- The degree of r is less than the degree of d -/
axiom deg_r_lt_deg_d : sorry -- We can't easily express polynomial degrees in this simple setup

theorem q_plus_r_at_one_eq_neg_47 : 
  ∃ (q r : ℝ → ℝ), (∀ x, f x = q x * d x + r x) ∧ q 1 + r 1 = -47 := by
  sorry

end q_plus_r_at_one_eq_neg_47_l1129_112996


namespace contrapositive_equivalence_l1129_112983

theorem contrapositive_equivalence (x y : ℝ) :
  (((x - 1) * (y + 2) ≠ 0 → x ≠ 1 ∧ y ≠ -2) ↔
   (x = 1 ∨ y = -2 → (x - 1) * (y + 2) = 0)) :=
by sorry

end contrapositive_equivalence_l1129_112983


namespace drug_effectiveness_max_effective_hours_l1129_112979

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x ≤ 4 then -1/2 * x^2 + 2*x + 8
  else if 4 < x ∧ x ≤ 16 then -x/2 - Real.log x / Real.log 2 + 12
  else 0

def is_effective (m : ℝ) (x : ℝ) : Prop := m * f x ≥ 12

theorem drug_effectiveness (m : ℝ) (h : m > 0) :
  (∀ x, 0 < x ∧ x ≤ 8 → is_effective m x) ↔ m ≥ 12/5 :=
sorry

theorem max_effective_hours :
  ∃ k : ℕ, k = 6 ∧ 
  (∀ x, 0 < x ∧ x ≤ ↑k → is_effective 2 x) ∧
  (∀ k' : ℕ, k' > k → ∃ x, 0 < x ∧ x ≤ ↑k' ∧ ¬is_effective 2 x) :=
sorry

end drug_effectiveness_max_effective_hours_l1129_112979


namespace multiplication_problem_l1129_112926

theorem multiplication_problem : 7 * (1 / 11) * 33 = 21 := by
  sorry

end multiplication_problem_l1129_112926


namespace head_start_calculation_l1129_112913

/-- Prove that given A runs 1 ¾ times as fast as B, and A and B reach a winning post 196 m away at the same time, the head start A gives B is 84 meters. -/
theorem head_start_calculation (speed_a speed_b head_start : ℝ) 
  (h1 : speed_a = (7/4) * speed_b)
  (h2 : (196 - head_start) / speed_b = 196 / speed_a) :
  head_start = 84 := by
  sorry

end head_start_calculation_l1129_112913


namespace trig_functions_right_triangle_l1129_112985

/-- Define trigonometric functions for a right-angled triangle --/
theorem trig_functions_right_triangle 
  (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ (A : ℝ), 
    Real.sin A = a / c ∧ 
    Real.cos A = b / c ∧ 
    Real.tan A = a / b :=
sorry

end trig_functions_right_triangle_l1129_112985


namespace inequality_solution_l1129_112969

def inequality (x : ℝ) : Prop :=
  (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 2

def solution_set (x : ℝ) : Prop :=
  (0 < x ∧ x ≤ 0.5) ∨ (x ≥ 6)

theorem inequality_solution : ∀ x : ℝ, inequality x ↔ solution_set x :=
sorry

end inequality_solution_l1129_112969


namespace five_n_plus_three_composite_l1129_112934

theorem five_n_plus_three_composite (n : ℕ) 
  (h1 : ∃ k : ℕ, 2 * n + 1 = k^2) 
  (h2 : ∃ m : ℕ, 3 * n + 1 = m^2) : 
  ¬(Nat.Prime (5 * n + 3)) :=
sorry

end five_n_plus_three_composite_l1129_112934


namespace bead_necklaces_sold_l1129_112974

/-- Proves that the number of bead necklaces sold is 4, given the conditions of the problem -/
theorem bead_necklaces_sold (gem_necklaces : ℕ) (price_per_necklace : ℕ) (total_earnings : ℕ) 
  (h1 : gem_necklaces = 3)
  (h2 : price_per_necklace = 3)
  (h3 : total_earnings = 21) :
  total_earnings - gem_necklaces * price_per_necklace = 4 * price_per_necklace :=
by sorry

end bead_necklaces_sold_l1129_112974


namespace xuzhou_metro_scientific_notation_l1129_112955

theorem xuzhou_metro_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 31900 = a * (10 : ℝ) ^ n ∧ a = 3.19 ∧ n = 4 := by
  sorry

end xuzhou_metro_scientific_notation_l1129_112955


namespace luke_candy_purchase_luke_candy_purchase_result_l1129_112920

/-- The number of candy pieces Luke can buy given his tickets and candy cost -/
theorem luke_candy_purchase (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) : ℕ :=
  by
  have h1 : whack_a_mole_tickets = 2 := by sorry
  have h2 : skee_ball_tickets = 13 := by sorry
  have h3 : candy_cost = 3 := by sorry
  
  have total_tickets : ℕ := whack_a_mole_tickets + skee_ball_tickets
  
  exact total_tickets / candy_cost

/-- Proof that Luke can buy 5 pieces of candy -/
theorem luke_candy_purchase_result : luke_candy_purchase 2 13 3 = 5 := by sorry

end luke_candy_purchase_luke_candy_purchase_result_l1129_112920


namespace multiples_of_4_or_6_in_100_l1129_112971

theorem multiples_of_4_or_6_in_100 :
  let S := Finset.range 100
  (S.filter (fun n => n % 4 = 0 ∨ n % 6 = 0)).card = 33 := by
  sorry

end multiples_of_4_or_6_in_100_l1129_112971


namespace max_a_value_l1129_112951

/-- A lattice point in an xy-coordinate system -/
def LatticePoint (x y : ℤ) : Prop := True

/-- The line equation y = mx + 5 -/
def LineEquation (m : ℚ) (x y : ℤ) : Prop := y = m * x + 5

/-- The condition for x -/
def XCondition (x : ℤ) : Prop := 0 < x ∧ x ≤ 150

/-- The condition for m -/
def MCondition (m a : ℚ) : Prop := 1/3 < m ∧ m < a

/-- The main theorem -/
theorem max_a_value : 
  ∃ (a : ℚ), a = 52/151 ∧ 
  (∀ (m : ℚ), MCondition m a → 
    ∀ (x y : ℤ), XCondition x → LatticePoint x y → ¬LineEquation m x y) ∧
  (∀ (a' : ℚ), a' > a → 
    ∃ (m : ℚ), MCondition m a' ∧
    ∃ (x y : ℤ), XCondition x ∧ LatticePoint x y ∧ LineEquation m x y) :=
sorry

end max_a_value_l1129_112951


namespace circle_trajectory_intersection_l1129_112961

-- Define the point F
def F : ℝ × ℝ := (1, 0)

-- Define the trajectory curve E
def E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line that intersects E and C₁
def line (x y b : ℝ) : Prop := y = (1/2)*x + b

-- Define the condition for complementary angles
def complementary_angles (B D : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := B
  let (x₂, y₂) := D
  (y₁ / (x₁ - 1)) + (y₂ / (x₂ - 1)) = 0

theorem circle_trajectory_intersection :
  ∀ (A B C D : ℝ × ℝ) (b : ℝ),
  E B.1 B.2 → E D.1 D.2 →
  C₁ A.1 A.2 → C₁ C.1 C.2 →
  line A.1 A.2 b → line B.1 B.2 b → line C.1 C.2 b → line D.1 D.2 b →
  complementary_angles B D →
  ∃ (AB CD : ℝ), AB + CD = (36 * Real.sqrt 5) / 5 :=
sorry

end circle_trajectory_intersection_l1129_112961


namespace equal_roots_quadratic_l1129_112997

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 4 = 0 ∧ 
   ∀ y : ℝ, y^2 + k*y + 4 = 0 → y = x) → 
  k = 4 ∨ k = -4 := by
sorry

end equal_roots_quadratic_l1129_112997


namespace max_goats_l1129_112950

/-- Represents the number of coconuts Max can trade for one crab -/
def coconuts_per_crab : ℕ := 3

/-- Represents the number of crabs Max can trade for one goat -/
def crabs_per_goat : ℕ := 6

/-- Represents the initial number of coconuts Max has -/
def initial_coconuts : ℕ := 342

/-- Calculates the number of goats Max will have after trading all his coconuts -/
def goats_from_coconuts (coconuts : ℕ) (coconuts_per_crab : ℕ) (crabs_per_goat : ℕ) : ℕ :=
  (coconuts / coconuts_per_crab) / crabs_per_goat

/-- Theorem stating that Max will end up with 19 goats -/
theorem max_goats : 
  goats_from_coconuts initial_coconuts coconuts_per_crab crabs_per_goat = 19 := by
  sorry

end max_goats_l1129_112950


namespace not_prime_n4_plus_n2_plus_1_l1129_112982

theorem not_prime_n4_plus_n2_plus_1 (n : ℕ) (h : n > 1) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + n^2 + 1 = a * b :=
by
  sorry

end not_prime_n4_plus_n2_plus_1_l1129_112982


namespace product_difference_bound_l1129_112933

theorem product_difference_bound (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h1 : x * y - z = x * z - y) (h2 : x * z - y = y * z - x) : 
  x * y - z ≥ -1/4 := by
sorry

end product_difference_bound_l1129_112933


namespace meaningful_range_l1129_112987

def is_meaningful (x : ℝ) : Prop :=
  x ≥ 3 ∧ x ≠ 4

theorem meaningful_range (x : ℝ) :
  (∃ y : ℝ, y = Real.sqrt (x - 3) / (x - 4)) ↔ is_meaningful x :=
sorry

end meaningful_range_l1129_112987


namespace eighth_week_hours_l1129_112907

def hours_worked : List ℕ := [9, 13, 8, 14, 12, 10, 11]
def total_weeks : ℕ := 8
def target_average : ℕ := 12

theorem eighth_week_hours : 
  ∃ x : ℕ, 
    (List.sum hours_worked + x) / total_weeks = target_average ∧ 
    x = 19 := by
  sorry

end eighth_week_hours_l1129_112907


namespace intersection_of_A_and_B_l1129_112999

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo 0 3 := by sorry

end intersection_of_A_and_B_l1129_112999


namespace cubic_sum_l1129_112917

theorem cubic_sum (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 8) / a = (b^3 + 8) / b ∧ (b^3 + 8) / b = (c^3 + 8) / c) : 
  a^3 + b^3 + c^3 = -24 := by
  sorry

end cubic_sum_l1129_112917


namespace sozopolian_inequality_sozopolian_equality_l1129_112978

/-- Definition of a Sozopolian set -/
def is_sozopolian (p a b c : ℕ) : Prop :=
  Nat.Prime p ∧ p % 2 = 1 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a * b + 1) % p = 0 ∧
  (b * c + 1) % p = 0 ∧
  (c * a + 1) % p = 0

theorem sozopolian_inequality (p a b c : ℕ) :
  is_sozopolian p a b c → p + 2 ≤ (a + b + c) / 3 :=
sorry

theorem sozopolian_equality (p : ℕ) :
  (∃ a b c : ℕ, is_sozopolian p a b c ∧ p + 2 = (a + b + c) / 3) ↔ p = 5 :=
sorry

end sozopolian_inequality_sozopolian_equality_l1129_112978


namespace least_product_of_primes_over_50_l1129_112941

theorem least_product_of_primes_over_50 :
  ∃ p q : ℕ,
    p.Prime ∧ q.Prime ∧
    p > 50 ∧ q > 50 ∧
    p ≠ q ∧
    p * q = 3127 ∧
    ∀ r s : ℕ,
      r.Prime → s.Prime →
      r > 50 → s > 50 →
      r ≠ s →
      r * s ≥ 3127 :=
by sorry

end least_product_of_primes_over_50_l1129_112941


namespace matrix_operation_result_l1129_112909

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 6, 4]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-1, 8; -7, 3]

theorem matrix_operation_result :
  (2 : ℤ) • (A + B) = !![8, 10; -2, 14] := by sorry

end matrix_operation_result_l1129_112909


namespace mom_tshirt_packages_l1129_112903

/-- The number of t-shirts in each package -/
def package_size : ℕ := 13

/-- The total number of t-shirts mom buys -/
def total_tshirts : ℕ := 39

/-- The number of packages mom will have -/
def num_packages : ℕ := total_tshirts / package_size

theorem mom_tshirt_packages : num_packages = 3 := by
  sorry

end mom_tshirt_packages_l1129_112903


namespace tens_digit_of_23_to_2045_l1129_112936

theorem tens_digit_of_23_to_2045 : 23^2045 ≡ 43 [ZMOD 100] := by
  sorry

end tens_digit_of_23_to_2045_l1129_112936


namespace largest_prime_factor_of_1581_l1129_112929

def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_1581 : 
  largest_prime_factor 1581 = 113 := by sorry

end largest_prime_factor_of_1581_l1129_112929


namespace perception_arrangements_l1129_112989

def word_length : ℕ := 10

def repeating_letters : List (Char × ℕ) := [('E', 2), ('P', 2), ('I', 2)]

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem perception_arrangements : 
  (factorial word_length) / ((repeating_letters.map (λ (_, count) => factorial count)).prod) = 453600 := by
  sorry

end perception_arrangements_l1129_112989


namespace riza_age_proof_l1129_112998

/-- Represents Riza's age when her son was born -/
def riza_age_at_birth : ℕ := 25

/-- Represents the current age of Riza's son -/
def son_current_age : ℕ := 40

/-- Represents the sum of Riza's and her son's current ages -/
def sum_of_ages : ℕ := 105

theorem riza_age_proof : 
  riza_age_at_birth + son_current_age + son_current_age = sum_of_ages := by
  sorry

end riza_age_proof_l1129_112998


namespace expression_simplification_l1129_112958

theorem expression_simplification :
  Real.sqrt 12 - 2 * Real.cos (30 * π / 180) - (1/3)⁻¹ = Real.sqrt 3 - 3 := by
  sorry

end expression_simplification_l1129_112958


namespace pendant_sales_theorem_l1129_112939

/-- Parameters for the Asian Games mascot pendant sales problem -/
structure PendantSales where
  cost : ℝ             -- Cost price of each pendant
  initial_price : ℝ    -- Initial selling price
  initial_sales : ℝ    -- Initial monthly sales
  price_sensitivity : ℝ -- Daily sales decrease per 1 yuan price increase

/-- Calculate profit based on price increase -/
def profit (p : PendantSales) (x : ℝ) : ℝ :=
  (p.initial_price + x - p.cost) * (p.initial_sales - 30 * p.price_sensitivity * x)

/-- Theorem for the Asian Games mascot pendant sales problem -/
theorem pendant_sales_theorem (p : PendantSales) 
  (h1 : p.cost = 13)
  (h2 : p.initial_price = 20)
  (h3 : p.initial_sales = 200)
  (h4 : p.price_sensitivity = 10) :
  (∃ x : ℝ, x^2 - 13*x + 22 = 0 ∧ profit p x = 1620) ∧
  (∃ x : ℝ, x = 53/2 ∧ ∀ y : ℝ, profit p y ≤ profit p x) ∧
  profit p (13/2) = 3645/2 := by
  sorry


end pendant_sales_theorem_l1129_112939


namespace omega_range_l1129_112916

open Real

theorem omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∃! (r₁ r₂ r₃ : ℝ), 0 < r₁ ∧ r₁ < r₂ ∧ r₂ < r₃ ∧ r₃ < π ∧
    (∀ x, sin (ω * x) - Real.sqrt 3 * cos (ω * x) = -1 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃)) →
  13/6 < ω ∧ ω ≤ 7/2 :=
by sorry

end omega_range_l1129_112916


namespace half_angle_quadrant_l1129_112910

-- Define the second quadrant
def second_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 2 * Real.pi + Real.pi / 2 < α ∧ α < k * 2 * Real.pi + Real.pi

-- Define the first quadrant
def first_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 2 * Real.pi < α ∧ α < n * 2 * Real.pi + Real.pi / 2

-- Define the third quadrant
def third_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 2 * Real.pi + Real.pi < α ∧ α < n * 2 * Real.pi + 3 * Real.pi / 2

-- Theorem statement
theorem half_angle_quadrant (α : Real) :
  second_quadrant α → first_quadrant (α / 2) ∨ third_quadrant (α / 2) := by
  sorry

end half_angle_quadrant_l1129_112910


namespace rhombus_perimeter_l1129_112954

/-- Given a rhombus with diagonals of 10 inches and 24 inches, its perimeter is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by
  sorry

end rhombus_perimeter_l1129_112954


namespace sum_of_divisors_154_l1129_112977

/-- The sum of all positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all positive divisors of 154 is 288 -/
theorem sum_of_divisors_154 : sum_of_divisors 154 = 288 := by sorry

end sum_of_divisors_154_l1129_112977


namespace isosceles_triangle_base_angle_l1129_112994

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We'll use degrees for simplicity
  base_angle : ℝ
  vertex_angle : ℝ
  is_isosceles : base_angle * 2 + vertex_angle = 180

-- Define our specific isosceles triangle
def our_triangle (t : IsoscelesTriangle) : Prop :=
  t.base_angle = 50 ∧ t.vertex_angle = 80 ∨
  t.base_angle = 80 ∧ t.vertex_angle = 20

-- Theorem statement
theorem isosceles_triangle_base_angle :
  ∀ t : IsoscelesTriangle, (t.base_angle = 50 ∨ t.base_angle = 80) ↔ 
  (t.base_angle = 50 ∧ t.vertex_angle = 80 ∨ t.base_angle = 80 ∧ t.vertex_angle = 20) :=
by sorry

end isosceles_triangle_base_angle_l1129_112994


namespace unique_solution_system_l1129_112962

theorem unique_solution_system : 
  ∃! (x y z : ℕ+), 
    (2 * x * z = y^2) ∧ 
    (x + z = 1987) ∧
    (x = 1458) ∧ 
    (y = 1242) ∧ 
    (z = 529) := by
  sorry

end unique_solution_system_l1129_112962


namespace sara_height_l1129_112959

/-- Given the heights of Mark, Roy, Joe, and Sara, prove Sara's height is 45 inches -/
theorem sara_height (mark_height joe_height roy_height sara_height : ℕ) 
  (h1 : mark_height = 34)
  (h2 : roy_height = mark_height + 2)
  (h3 : joe_height = roy_height + 3)
  (h4 : sara_height = joe_height + 6) :
  sara_height = 45 := by
  sorry

end sara_height_l1129_112959


namespace gcd_repeating_even_three_digit_l1129_112900

theorem gcd_repeating_even_three_digit : 
  ∃ g : ℕ, ∀ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ Even n → 
    g = Nat.gcd (1001 * n) (Nat.gcd (1001 * (n + 2)) (1001 * (n + 4))) ∧ 
    g = 2002 := by
  sorry

end gcd_repeating_even_three_digit_l1129_112900


namespace square_sum_value_l1129_112953

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 6) : a^2 + b^2 = 21 := by
  sorry

end square_sum_value_l1129_112953


namespace price_restoration_l1129_112924

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) 
  (h1 : reduced_price = 0.9 * original_price) : 
  (11 + 1/9) / 100 * reduced_price = original_price := by
sorry

end price_restoration_l1129_112924


namespace vasil_can_win_more_l1129_112975

/-- Represents the possible objects on a coin side -/
inductive Object
| Scissors
| Paper
| Rock

/-- Represents a coin with two sides -/
structure Coin where
  side1 : Object
  side2 : Object

/-- The set of available coins -/
def coins : List Coin := [
  ⟨Object.Scissors, Object.Paper⟩,
  ⟨Object.Rock, Object.Scissors⟩,
  ⟨Object.Paper, Object.Rock⟩
]

/-- Determines if object1 beats object2 -/
def beats (object1 object2 : Object) : Bool :=
  match object1, object2 with
  | Object.Scissors, Object.Paper => true
  | Object.Paper, Object.Rock => true
  | Object.Rock, Object.Scissors => true
  | _, _ => false

/-- Calculates the probability of Vasil winning against Asya -/
def winProbability (asyaCoin vasilCoin : Coin) : Rat :=
  sorry

/-- Theorem stating that Vasil can choose a coin to have a higher winning probability -/
theorem vasil_can_win_more : ∃ (strategy : Coin → Coin),
  ∀ (asyaChoice : Coin),
    asyaChoice ∈ coins →
    winProbability asyaChoice (strategy asyaChoice) > 1/2 :=
  sorry

end vasil_can_win_more_l1129_112975


namespace polygon_interior_angles_l1129_112991

theorem polygon_interior_angles (n : ℕ) : 
  (n ≥ 3) → 
  (2005 + 180 = (n - 2) * 180) → 
  n = 14 :=
by sorry

end polygon_interior_angles_l1129_112991


namespace ninety_ninth_digit_sum_l1129_112956

/-- The decimal expansion of 2/9 -/
def decimal_expansion_2_9 : ℚ := 2/9

/-- The decimal expansion of 3/11 -/
def decimal_expansion_3_11 : ℚ := 3/11

/-- The 99th digit after the decimal point in a rational number -/
def digit_99 (q : ℚ) : ℕ :=
  sorry

/-- Theorem: The 99th digit after the decimal point in the decimal expansion of 2/9 + 3/11 is 4 -/
theorem ninety_ninth_digit_sum :
  digit_99 (decimal_expansion_2_9 + decimal_expansion_3_11) = 4 := by
  sorry

end ninety_ninth_digit_sum_l1129_112956


namespace g_satisfies_conditions_g_value_at_neg_two_l1129_112968

/-- A cubic polynomial -/
def CubicPolynomial (α : Type*) [Field α] := α → α

/-- The polynomial f(x) = x^3 - 2x^2 + 5 -/
def f : CubicPolynomial ℝ := λ x ↦ x^3 - 2*x^2 + 5

/-- The polynomial g, which is defined by the problem conditions -/
noncomputable def g : CubicPolynomial ℝ := sorry

/-- The roots of f -/
noncomputable def roots_f : Finset ℝ := sorry

theorem g_satisfies_conditions :
  (g 0 = 1) ∧ 
  (∀ r ∈ roots_f, ∃ s, g s = 0 ∧ s = r^2) ∧
  (∀ s, g s = 0 → ∃ r ∈ roots_f, s = r^2) := sorry

theorem g_value_at_neg_two : g (-2) = 24.2 := sorry

end g_satisfies_conditions_g_value_at_neg_two_l1129_112968


namespace alyssa_ate_25_limes_l1129_112964

/-- The number of limes Mike picked -/
def mike_limes : ℝ := 32.0

/-- The number of limes left -/
def limes_left : ℝ := 7

/-- The number of limes Alyssa ate -/
def alyssa_limes : ℝ := mike_limes - limes_left

/-- Proof that Alyssa ate 25.0 limes -/
theorem alyssa_ate_25_limes : alyssa_limes = 25.0 := by
  sorry

end alyssa_ate_25_limes_l1129_112964


namespace smallest_x_value_l1129_112927

theorem smallest_x_value (x y : ℕ+) (h : (9 : ℚ) / 10 = y / (275 + x)) : 
  x ≥ 5 ∧ ∃ (y' : ℕ+), (9 : ℚ) / 10 = y' / (275 + 5) :=
sorry

end smallest_x_value_l1129_112927


namespace rice_bags_weight_analysis_l1129_112919

def standard_weight : ℝ := 50
def num_bags : ℕ := 10
def weight_deviations : List ℝ := [0.5, 0.3, 0, -0.2, -0.3, 1.1, -0.7, -0.2, 0.6, 0.7]

theorem rice_bags_weight_analysis :
  let total_deviation : ℝ := weight_deviations.sum
  let total_weight : ℝ := (standard_weight * num_bags) + total_deviation
  let average_weight : ℝ := total_weight / num_bags
  (total_deviation = 1.7) ∧ 
  (total_weight = 501.7) ∧ 
  (average_weight = 50.17) := by
sorry

end rice_bags_weight_analysis_l1129_112919


namespace difference_closure_l1129_112995

def is_closed_set (A : Set Int) : Prop :=
  (∃ (a b : Int), a ∈ A ∧ a > 0 ∧ b ∈ A ∧ b < 0) ∧
  (∀ a b : Int, a ∈ A → b ∈ A → (2 * a) ∈ A ∧ (a + b) ∈ A)

theorem difference_closure (A : Set Int) (h : is_closed_set A) :
  ∀ x y : Int, x ∈ A → y ∈ A → (x - y) ∈ A :=
by sorry

end difference_closure_l1129_112995


namespace f_satisfies_conditions_l1129_112967

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end f_satisfies_conditions_l1129_112967


namespace probability_of_vowels_in_all_sets_l1129_112923

-- Define the sets
def set1 : Finset Char := {'a', 'b', 'o', 'd', 'e', 'f', 'g'}
def set2 : Finset Char := {'k', 'l', 'm', 'n', 'u', 'p', 'r', 's'}
def set3 : Finset Char := {'t', 'v', 'w', 'i', 'x', 'y', 'z'}
def set4 : Finset Char := {'a', 'c', 'e', 'u', 'g', 'h', 'j'}

-- Define vowels
def vowels : Finset Char := {'a', 'e', 'i', 'o', 'u'}

-- Function to count vowels in a set
def countVowels (s : Finset Char) : Nat :=
  (s ∩ vowels).card

-- Theorem statement
theorem probability_of_vowels_in_all_sets :
  let p1 := (countVowels set1 : ℚ) / set1.card
  let p2 := (countVowels set2 : ℚ) / set2.card
  let p3 := (countVowels set3 : ℚ) / set3.card
  let p4 := (countVowels set4 : ℚ) / set4.card
  p1 * p2 * p3 * p4 = 9 / 2744 := by
  sorry

end probability_of_vowels_in_all_sets_l1129_112923


namespace eighth_group_student_number_l1129_112966

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  students_per_group : ℕ
  selected_number : ℕ
  selected_group : ℕ

/-- Calculates the number of the student in a given group -/
def student_number (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.selected_number + (group - s.selected_group) * s.students_per_group

/-- Theorem: In the given systematic sampling, the student from the 8th group has number 37 -/
theorem eighth_group_student_number (s : SystematicSampling) 
    (h1 : s.total_students = 50)
    (h2 : s.num_groups = 10)
    (h3 : s.students_per_group = 5)
    (h4 : s.selected_number = 12)
    (h5 : s.selected_group = 3) :
    student_number s 8 = 37 := by
  sorry


end eighth_group_student_number_l1129_112966


namespace base_10_423_equals_base_5_3143_l1129_112914

def base_10_to_base_5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base_10_423_equals_base_5_3143 :
  base_10_to_base_5 423 = [3, 1, 4, 3] := by
  sorry

end base_10_423_equals_base_5_3143_l1129_112914


namespace simplify_expression_l1129_112918

theorem simplify_expression : (18 * (10^10)) / (6 * (10^4)) * 2 = 6 * (10^6) := by
  sorry

end simplify_expression_l1129_112918


namespace roots_sum_cubic_l1129_112965

theorem roots_sum_cubic (a b c : ℂ) : 
  a^3 + 2*a^2 + 3*a + 4 = 0 →
  b^3 + 2*b^2 + 3*b + 4 = 0 →
  c^3 + 2*c^2 + 3*c + 4 = 0 →
  (a^3 - b^3) / (a - b) + (b^3 - c^3) / (b - c) + (c^3 - a^3) / (c - a) = -1 := by
  sorry

end roots_sum_cubic_l1129_112965


namespace intersection_of_A_and_B_l1129_112908

def A : Set ℕ := {2, 4}
def B : Set ℕ := {3, 4}

theorem intersection_of_A_and_B : A ∩ B = {4} := by sorry

end intersection_of_A_and_B_l1129_112908


namespace unique_x_intercept_l1129_112932

theorem unique_x_intercept (x : ℝ) : 
  ∃! x, (x - 4) * (x^2 + 4*x + 13) = 0 :=
by sorry

end unique_x_intercept_l1129_112932


namespace chord_line_equation_l1129_112957

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a circle and a point that is the midpoint of a chord, 
    return the line containing that chord -/
def chordLine (c : Circle) (p : ℝ × ℝ) : Line :=
  sorry

theorem chord_line_equation (c : Circle) (p : ℝ × ℝ) :
  let circle : Circle := { center := (3, 0), radius := 3 }
  let midpoint : ℝ × ℝ := (4, 2)
  let line := chordLine circle midpoint
  line.a = 1 ∧ line.b = 2 ∧ line.c = -8 := by sorry

end chord_line_equation_l1129_112957


namespace sachins_age_l1129_112921

theorem sachins_age (sachin_age rahul_age : ℝ) 
  (h1 : rahul_age = sachin_age + 9)
  (h2 : sachin_age / rahul_age = 7 / 9) :
  sachin_age = 31.5 := by
  sorry

end sachins_age_l1129_112921


namespace candy_distribution_l1129_112949

theorem candy_distribution (N : ℕ) : N > 1 ∧ 
  N % 2 = 1 ∧ 
  N % 3 = 1 ∧ 
  N % 5 = 1 ∧ 
  (∀ m : ℕ, m > 1 → m % 2 = 1 → m % 3 = 1 → m % 5 = 1 → m ≥ N) → 
  N = 31 := by
sorry

end candy_distribution_l1129_112949


namespace unique_abc_solution_l1129_112986

/-- Represents a base-7 number with two digits -/
def Base7TwoDigit (a b : Nat) : Nat := 7 * a + b

/-- Represents a base-7 number with one digit -/
def Base7OneDigit (c : Nat) : Nat := c

/-- Represents a base-7 number with two digits, where the first digit is 'c' and the second is 0 -/
def Base7TwoDigitWithZero (c : Nat) : Nat := 7 * c

theorem unique_abc_solution :
  ∀ (A B C : Nat),
    A ≠ 0 → B ≠ 0 → C ≠ 0 →
    A < 7 → B < 7 → C < 7 →
    A ≠ B → B ≠ C → A ≠ C →
    Base7TwoDigit A B + Base7OneDigit C = Base7TwoDigitWithZero C →
    Base7TwoDigit A B + Base7TwoDigit B A = Base7TwoDigit C C →
    A = 3 ∧ B = 2 ∧ C = 5 := by
  sorry

#check unique_abc_solution

end unique_abc_solution_l1129_112986


namespace f_solution_sets_l1129_112901

/-- The function f(x) = ax^2 - (a+c)x + c -/
def f (a c x : ℝ) : ℝ := a * x^2 - (a + c) * x + c

theorem f_solution_sets :
  /- Part 1 -/
  (∀ a c : ℝ, a > 0 → (∀ x : ℝ, f a c x = f a c (-2 - x)) →
    {x : ℝ | f a c x > 0} = {x : ℝ | x < -3 ∨ x > 1}) ∧
  /- Part 2 -/
  (∀ a : ℝ, a ≥ 0 → f a 1 0 = 1 →
    {x : ℝ | f a 1 x > 0} =
      if a = 0 then {x : ℝ | x > 1}
      else if 0 < a ∧ a < 1 then {x : ℝ | 1 < x ∧ x < 1/a}
      else if a > 1 then {x : ℝ | 1/a < x ∧ x < 1}
      else ∅) :=
by sorry

end f_solution_sets_l1129_112901


namespace aunts_gift_l1129_112911

theorem aunts_gift (grandfather_gift : ℕ) (bank_deposit : ℕ) (total_gift : ℕ) :
  grandfather_gift = 150 →
  bank_deposit = 45 →
  bank_deposit * 5 = total_gift →
  total_gift - grandfather_gift = 75 :=
by
  sorry

end aunts_gift_l1129_112911


namespace eighth_diagram_fully_shaded_l1129_112943

/-- The number of shaded squares in the n-th diagram -/
def shaded_squares (n : ℕ) : ℕ := n^2

/-- The total number of squares in the n-th diagram -/
def total_squares (n : ℕ) : ℕ := n^2

/-- The fraction of shaded squares in the n-th diagram -/
def shaded_fraction (n : ℕ) : ℚ :=
  (shaded_squares n : ℚ) / (total_squares n : ℚ)

theorem eighth_diagram_fully_shaded :
  shaded_fraction 8 = 1 := by sorry

end eighth_diagram_fully_shaded_l1129_112943


namespace cells_intersected_303x202_l1129_112942

/-- Represents a grid rectangle with diagonals --/
structure GridRectangle where
  length : Nat
  width : Nat

/-- Calculates the number of cells intersected by diagonals in a grid rectangle --/
def cells_intersected_by_diagonals (grid : GridRectangle) : Nat :=
  let small_rectangles := (grid.length / 3) * (grid.width / 2)
  let cells_per_diagonal := small_rectangles * 4
  let total_cells := cells_per_diagonal * 2
  total_cells - 2

/-- Theorem stating that in a 303 x 202 grid rectangle, 806 cells are intersected by diagonals --/
theorem cells_intersected_303x202 :
  cells_intersected_by_diagonals ⟨303, 202⟩ = 806 := by
  sorry

end cells_intersected_303x202_l1129_112942


namespace fishbowl_count_l1129_112988

theorem fishbowl_count (total_fish : ℕ) (fish_per_bowl : ℕ) (h1 : total_fish = 6003) (h2 : fish_per_bowl = 23) :
  total_fish / fish_per_bowl = 261 :=
by sorry

end fishbowl_count_l1129_112988


namespace average_letters_per_day_l1129_112935

def letters_monday : ℕ := 7
def letters_tuesday : ℕ := 10
def letters_wednesday : ℕ := 3
def letters_thursday : ℕ := 5
def letters_friday : ℕ := 12
def total_days : ℕ := 5

theorem average_letters_per_day :
  (letters_monday + letters_tuesday + letters_wednesday + letters_thursday + letters_friday : ℚ) / total_days = 37 / 5 := by
  sorry

end average_letters_per_day_l1129_112935


namespace all_statements_correct_l1129_112915

-- Define chemical elements and their atomic masses
def H : ℝ := 1
def O : ℝ := 16
def S : ℝ := 32
def N : ℝ := 14
def C : ℝ := 12

-- Define molecules and their molar masses
def H2SO4_mass : ℝ := 2 * H + S + 4 * O
def NO_mass : ℝ := N + O
def NO2_mass : ℝ := N + 2 * O
def O2_mass : ℝ := 2 * O
def O3_mass : ℝ := 3 * O
def CO_mass : ℝ := C + O
def CO2_mass : ℝ := C + 2 * O

-- Define the number of atoms in 2 mol of NO and NO2
def NO_atoms : ℕ := 2
def NO2_atoms : ℕ := 3

-- Theorem stating all given statements are correct
theorem all_statements_correct :
  (H2SO4_mass = 98) ∧
  (2 * NO_atoms ≠ 2 * NO2_atoms) ∧
  (∀ m : ℝ, m > 0 → m / O2_mass * 2 = m / O3_mass * 3) ∧
  (∀ n : ℝ, n > 0 → n * (CO_mass / C) = n * (CO2_mass / C)) :=
by sorry

end all_statements_correct_l1129_112915


namespace balance_proof_l1129_112928

def initial_balance : ℕ := 27004
def transferred_amount : ℕ := 69
def remaining_balance : ℕ := 26935

theorem balance_proof : initial_balance = transferred_amount + remaining_balance := by
  sorry

end balance_proof_l1129_112928


namespace experienced_sailors_monthly_earnings_l1129_112947

/-- Calculate the total combined monthly earnings of experienced sailors --/
theorem experienced_sailors_monthly_earnings
  (total_sailors : ℕ)
  (inexperienced_sailors : ℕ)
  (inexperienced_hourly_wage : ℚ)
  (weekly_hours : ℕ)
  (weeks_per_month : ℕ)
  (h_total : total_sailors = 17)
  (h_inexperienced : inexperienced_sailors = 5)
  (h_wage : inexperienced_hourly_wage = 10)
  (h_hours : weekly_hours = 60)
  (h_weeks : weeks_per_month = 4) :
  let experienced_sailors := total_sailors - inexperienced_sailors
  let experienced_hourly_wage := inexperienced_hourly_wage * (1 + 1/5)
  let weekly_earnings := experienced_hourly_wage * weekly_hours
  let total_monthly_earnings := weekly_earnings * experienced_sailors * weeks_per_month
  total_monthly_earnings = 34560 :=
by sorry

end experienced_sailors_monthly_earnings_l1129_112947


namespace corrected_mean_l1129_112972

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 →
  original_mean = 40 →
  incorrect_value = 15 →
  correct_value = 45 →
  let original_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := original_sum + difference
  corrected_sum / n = 40.6 := by
  sorry

end corrected_mean_l1129_112972


namespace min_value_problem_l1129_112937

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2/y = 3) :
  ∃ (m : ℝ), m = 8/3 ∧ ∀ z, z = 2/x + y → z ≥ m := by
  sorry

end min_value_problem_l1129_112937


namespace expression_value_l1129_112952

theorem expression_value : 3^(2+4+6) - (3^2 + 3^4 + 3^6) + (3^2 * 3^4 * 3^6) = 1062242 := by
  sorry

end expression_value_l1129_112952


namespace squirrel_nut_difference_l1129_112945

theorem squirrel_nut_difference :
  let num_squirrels : ℕ := 4
  let num_nuts : ℕ := 2
  num_squirrels - num_nuts = 2 :=
by sorry

end squirrel_nut_difference_l1129_112945


namespace consecutive_natural_even_product_inequality_l1129_112963

theorem consecutive_natural_even_product_inequality (n : ℕ) (m : ℕ) (h : Even m) (h_pos : m > 0) :
  n * (n + 1) ≠ m * (m + 2) := by
  sorry

end consecutive_natural_even_product_inequality_l1129_112963


namespace comic_arrangement_count_l1129_112984

/-- The number of different Batman comic books --/
def batman_comics : ℕ := 8

/-- The number of different Superman comic books --/
def superman_comics : ℕ := 7

/-- The number of different Wonder Woman comic books --/
def wonder_woman_comics : ℕ := 5

/-- The total number of comic books --/
def total_comics : ℕ := batman_comics + superman_comics + wonder_woman_comics

/-- The number of different comic book types --/
def comic_types : ℕ := 3

theorem comic_arrangement_count :
  (Nat.factorial batman_comics) * (Nat.factorial superman_comics) * 
  (Nat.factorial wonder_woman_comics) * (Nat.factorial comic_types) = 12203212800 := by
  sorry

end comic_arrangement_count_l1129_112984


namespace fraction_exists_l1129_112906

theorem fraction_exists : ∃ n : ℕ, (n : ℚ) / 22 = 9545 / 10000 := by
  sorry

end fraction_exists_l1129_112906


namespace product_ratio_l1129_112946

def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def first_three_primes : List ℕ := [2, 3, 5]
def next_three_composites : List ℕ := [14, 15, 16]

theorem product_ratio :
  (first_six_composites.prod) / ((first_three_primes ++ next_three_composites).prod) = 24 / 7 := by
  sorry

end product_ratio_l1129_112946


namespace grape_yield_after_change_l1129_112931

/-- Represents the number of jars that can be made from one can of juice -/
structure JuiceYield where
  apple : ℚ
  grape : ℚ

/-- Represents the recipe for the beverage -/
structure Recipe where
  apple : ℚ
  grape : ℚ

/-- The initial recipe yield -/
def initial_yield : JuiceYield :=
  { apple := 6,
    grape := 10 }

/-- The changed recipe yield for apple juice -/
def changed_apple_yield : ℚ := 5

/-- Theorem stating that after the recipe change, one can of grape juice makes 15 jars -/
theorem grape_yield_after_change
  (initial : JuiceYield)
  (changed_apple : ℚ)
  (h_initial : initial = initial_yield)
  (h_changed_apple : changed_apple = changed_apple_yield)
  : ∃ (changed : JuiceYield), changed.grape = 15 :=
sorry

end grape_yield_after_change_l1129_112931


namespace intersection_distance_implies_a_value_l1129_112938

-- Define the curve C
def curve_C (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x ∧ a > 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the intersection points
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ curve_C a x y ∧ line_l x y}

-- Theorem statement
theorem intersection_distance_implies_a_value (a : ℝ) :
  (∃ (A B : ℝ × ℝ), A ∈ intersection_points a ∧ B ∈ intersection_points a ∧ 
   A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 10) →
  a = 1 := by sorry

end intersection_distance_implies_a_value_l1129_112938


namespace xyz_value_l1129_112990

theorem xyz_value (a b c x y z : ℂ)
  (eq1 : a = (b + c) / (x - 2))
  (eq2 : b = (c + a) / (y - 2))
  (eq3 : c = (a + b) / (z - 2))
  (sum_prod : x * y + y * z + x * z = 67)
  (sum : x + y + z = 2010) :
  x * y * z = -5892 := by
sorry

end xyz_value_l1129_112990


namespace baker_pastries_l1129_112992

/-- Given that Baker made 43 cakes, sold 154 pastries and 78 cakes,
    and sold 76 more pastries than cakes, prove that Baker made 154 pastries. -/
theorem baker_pastries :
  let cakes_made : ℕ := 43
  let pastries_sold : ℕ := 154
  let cakes_sold : ℕ := 78
  let difference : ℕ := 76
  pastries_sold = cakes_sold + difference →
  pastries_sold = 154
:= by sorry

end baker_pastries_l1129_112992


namespace boxes_needed_proof_l1129_112925

/-- The number of chocolate bars Tom needs to sell -/
def total_bars : ℕ := 849

/-- The number of chocolate bars in each box -/
def bars_per_box : ℕ := 5

/-- The minimum number of boxes needed to contain all the bars -/
def min_boxes_needed : ℕ := (total_bars + bars_per_box - 1) / bars_per_box

theorem boxes_needed_proof : min_boxes_needed = 170 := by
  sorry

end boxes_needed_proof_l1129_112925


namespace quadratic_root_sum_l1129_112970

theorem quadratic_root_sum (a b c : ℤ) (ha : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = a ∨ x = b) →
  a + b + c = 18 := by
sorry

end quadratic_root_sum_l1129_112970


namespace books_read_together_l1129_112905

theorem books_read_together (tony_books dean_books breanna_books tony_dean_overlap total_different : ℕ)
  (h1 : tony_books = 23)
  (h2 : dean_books = 12)
  (h3 : breanna_books = 17)
  (h4 : tony_dean_overlap = 3)
  (h5 : total_different = 47) :
  tony_books + dean_books + breanna_books - tony_dean_overlap - total_different = 2 :=
by sorry

end books_read_together_l1129_112905


namespace pentagon_cannot_tile_floor_l1129_112960

-- Define a function to calculate the interior angle of a regular polygon
def interior_angle (n : ℕ) : ℚ :=
  180 - 360 / n

-- Define a function to check if an angle can divide 360° evenly
def divides_360 (angle : ℚ) : Prop :=
  ∃ k : ℕ, k * angle = 360

-- Theorem statement
theorem pentagon_cannot_tile_floor :
  divides_360 (interior_angle 6) ∧
  divides_360 90 ∧
  divides_360 60 ∧
  ¬ divides_360 (interior_angle 5) := by
  sorry

end pentagon_cannot_tile_floor_l1129_112960


namespace population_growth_two_periods_l1129_112902

/-- Theorem: Population growth over two periods --/
theorem population_growth_two_periods (P : ℝ) (h : P > 0) :
  let first_half := P * 3
  let second_half := first_half * 4
  (second_half - P) / P * 100 = 1100 := by
  sorry

end population_growth_two_periods_l1129_112902


namespace theo_homework_assignments_l1129_112944

/-- Calculates the number of assignments for a given set number -/
def assignmentsPerSet (setNumber : Nat) : Nat :=
  2^(setNumber - 1)

/-- Calculates the total assignments for a given number of sets -/
def totalAssignments (sets : Nat) : Nat :=
  (List.range sets).map (fun i => 6 * assignmentsPerSet (i + 1)) |>.sum

theorem theo_homework_assignments :
  totalAssignments 5 = 186 := by
  sorry

#eval totalAssignments 5

end theo_homework_assignments_l1129_112944


namespace find_m_l1129_112976

def A (m : ℕ) : Set ℝ := {x : ℝ | (m * x - 1) / x < 0}

def B : Set ℝ := {x : ℝ | 2 * x^2 - x < 0}

def is_necessary_not_sufficient (A B : Set ℝ) : Prop :=
  B ⊆ A ∧ A ≠ B

theorem find_m :
  ∃ (m : ℕ), m > 0 ∧ m < 6 ∧ is_necessary_not_sufficient (A m) B ∧ m = 1 :=
by sorry

end find_m_l1129_112976


namespace negative_abs_opposite_double_negative_l1129_112973

-- Define the property of being opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- State the theorem
theorem negative_abs_opposite_double_negative :
  are_opposite (-|(-3 : ℝ)|) (-(-3)) :=
sorry

end negative_abs_opposite_double_negative_l1129_112973


namespace smallest_n_for_sqrt_63n_l1129_112981

theorem smallest_n_for_sqrt_63n (n : ℕ) : n > 0 ∧ ∃ k : ℕ, k > 0 ∧ k^2 = 63 * n → n ≥ 7 :=
sorry

end smallest_n_for_sqrt_63n_l1129_112981


namespace infinitely_many_good_numbers_good_not_divisible_by_seven_l1129_112930

/-- A natural number n is good if there exist natural numbers a and b
    such that a + b = n and ab | n^2 + n + 1 -/
def is_good (n : ℕ) : Prop :=
  ∃ a b : ℕ, a + b = n ∧ (n^2 + n + 1) % (a * b) = 0

theorem infinitely_many_good_numbers :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ is_good n :=
sorry

theorem good_not_divisible_by_seven :
  ∀ n : ℕ, is_good n → ¬(7 ∣ n) :=
sorry

end infinitely_many_good_numbers_good_not_divisible_by_seven_l1129_112930


namespace remaining_trees_correct_l1129_112904

/-- The number of oak trees remaining in the park after cutting down damaged trees -/
def remaining_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that the number of remaining trees is correct -/
theorem remaining_trees_correct (initial : ℕ) (cut_down : ℕ) 
  (h : cut_down ≤ initial) : 
  remaining_trees initial cut_down = initial - cut_down :=
by sorry

end remaining_trees_correct_l1129_112904


namespace least_product_of_primes_above_30_l1129_112922

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ,
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p > 30 ∧ 
    q > 30 ∧ 
    p ≠ q ∧
    p * q = 1147 ∧
    ∀ a b : ℕ, 
      Nat.Prime a → 
      Nat.Prime b → 
      a > 30 → 
      b > 30 → 
      a ≠ b → 
      a * b ≥ 1147 :=
by
  sorry

end least_product_of_primes_above_30_l1129_112922


namespace inequality_holds_l1129_112940

theorem inequality_holds (x y : ℝ) (h : 2 * y + 5 * x = 10) : 3 * x * y - x^2 - y^2 < 7 := by
  sorry

end inequality_holds_l1129_112940


namespace bob_remaining_corn_l1129_112912

theorem bob_remaining_corn (initial_bushels : ℕ) (ears_per_bushel : ℕ) 
  (terry_bushels jerry_bushels linda_bushels : ℕ) (stacy_ears : ℕ) : 
  initial_bushels = 50 →
  ears_per_bushel = 14 →
  terry_bushels = 8 →
  jerry_bushels = 3 →
  linda_bushels = 12 →
  stacy_ears = 21 →
  initial_bushels * ears_per_bushel - 
  (terry_bushels * ears_per_bushel + jerry_bushels * ears_per_bushel + 
   linda_bushels * ears_per_bushel + stacy_ears) = 357 :=
by sorry

end bob_remaining_corn_l1129_112912
