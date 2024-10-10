import Mathlib

namespace min_sections_problem_l300_30050

theorem min_sections_problem (num_boys num_girls max_per_section : ℕ) 
  (h_boys : num_boys = 408)
  (h_girls : num_girls = 240)
  (h_max : max_per_section = 24)
  (h_ratio : ∃ (x : ℕ), x > 0 ∧ num_boys ≤ 3 * x * max_per_section ∧ num_girls ≤ 2 * x * max_per_section) :
  ∃ (total_sections : ℕ), 
    total_sections = 30 ∧
    ∃ (boys_sections girls_sections : ℕ),
      boys_sections + girls_sections = total_sections ∧
      3 * girls_sections = 2 * boys_sections ∧
      num_boys ≤ boys_sections * max_per_section ∧
      num_girls ≤ girls_sections * max_per_section ∧
      ∀ (other_total : ℕ),
        (∃ (other_boys other_girls : ℕ),
          other_boys + other_girls = other_total ∧
          3 * other_girls = 2 * other_boys ∧
          num_boys ≤ other_boys * max_per_section ∧
          num_girls ≤ other_girls * max_per_section) →
        other_total ≥ total_sections :=
by sorry

end min_sections_problem_l300_30050


namespace simplified_quadratic_radical_example_l300_30010

def is_simplified_quadratic_radical (x : ℝ) : Prop :=
  ∃ n : ℕ, x = Real.sqrt n ∧ ¬∃ m : ℕ, m > 1 ∧ ∃ k : ℕ, n = m^2 * k

theorem simplified_quadratic_radical_example :
  is_simplified_quadratic_radical (Real.sqrt 6) ∧
  ¬is_simplified_quadratic_radical (Real.sqrt 12) ∧
  ¬is_simplified_quadratic_radical (Real.sqrt 20) ∧
  ¬is_simplified_quadratic_radical (Real.sqrt 32) :=
by sorry

end simplified_quadratic_radical_example_l300_30010


namespace mp_eq_nq_l300_30055

/-- Two circles in a plane -/
structure TwoCircles where
  c1 : Set (ℝ × ℝ)
  c2 : Set (ℝ × ℝ)

/-- Points on the circles -/
structure CirclePoints (tc : TwoCircles) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_A_intersect : A ∈ tc.c1 ∩ tc.c2
  h_B_intersect : B ∈ tc.c1 ∩ tc.c2
  h_M_on_c1 : M ∈ tc.c1
  h_N_on_c2 : N ∈ tc.c2
  h_P_on_c1 : P ∈ tc.c1
  h_Q_on_c2 : Q ∈ tc.c2

/-- AM is tangent to c2 at A -/
def is_tangent_AM (tc : TwoCircles) (cp : CirclePoints tc) : Prop := sorry

/-- AN is tangent to c1 at A -/
def is_tangent_AN (tc : TwoCircles) (cp : CirclePoints tc) : Prop := sorry

/-- B, M, and P are collinear -/
def collinear_BMP (cp : CirclePoints tc) : Prop := sorry

/-- B, N, and Q are collinear -/
def collinear_BNQ (cp : CirclePoints tc) : Prop := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem mp_eq_nq (tc : TwoCircles) (cp : CirclePoints tc)
    (h_AM_tangent : is_tangent_AM tc cp)
    (h_AN_tangent : is_tangent_AN tc cp)
    (h_BMP_collinear : collinear_BMP cp)
    (h_BNQ_collinear : collinear_BNQ cp) :
    distance cp.M cp.P = distance cp.N cp.Q := by sorry

end mp_eq_nq_l300_30055


namespace coin_value_difference_l300_30082

/-- Represents the total number of coins Alice has -/
def total_coins : ℕ := 3030

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Calculates the total value in cents given the number of dimes -/
def total_value (dimes : ℕ) : ℕ :=
  dime_value * dimes + nickel_value * (total_coins - dimes)

/-- Represents the constraint that Alice has at least three times as many nickels as dimes -/
def nickel_constraint (dimes : ℕ) : Prop :=
  3 * dimes ≤ total_coins - dimes

theorem coin_value_difference :
  ∃ (max_dimes min_dimes : ℕ),
    nickel_constraint max_dimes ∧
    nickel_constraint min_dimes ∧
    (∀ d, nickel_constraint d → total_value d ≤ total_value max_dimes) ∧
    (∀ d, nickel_constraint d → total_value min_dimes ≤ total_value d) ∧
    total_value max_dimes - total_value min_dimes = 3780 := by
  sorry

end coin_value_difference_l300_30082


namespace simplify_complex_fraction_l300_30039

theorem simplify_complex_fraction (m : ℝ) (h1 : m ≠ 2) (h2 : m ≠ -3) :
  (m - (4*m - 9) / (m - 2)) / ((m^2 - 9) / (m - 2)) = (m - 3) / (m + 3) := by
  sorry

end simplify_complex_fraction_l300_30039


namespace polynomial_simplification_l300_30096

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + x^3 - 6 * x^2 + 9 * x - 5) + 
  (-x^4 + 2 * x^3 - 3 * x^2 + 4 * x - 2) + 
  (3 * x^4 - 3 * x^3 + x^2 - x + 1) = 
  4 * x^4 - 8 * x^2 + 12 * x - 6 := by
  sorry

end polynomial_simplification_l300_30096


namespace ratio_equality_l300_30063

theorem ratio_equality (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) :
  (a / 3) / (b / 2) = 1 := by
  sorry

end ratio_equality_l300_30063


namespace no_solutions_power_equation_l300_30069

theorem no_solutions_power_equation (x n r : ℕ) (hx : x > 1) :
  x^(2*n + 1) ≠ 2^r + 1 ∧ x^(2*n + 1) ≠ 2^r - 1 := by
  sorry

end no_solutions_power_equation_l300_30069


namespace acute_triangle_properties_l300_30030

open Real

-- Define the triangle
def Triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / sin A = b / sin B ∧ b / sin B = c / sin C

theorem acute_triangle_properties
  (A B C a b c : ℝ)
  (h_triangle : Triangle A B C a b c)
  (h_acute : A < π/2 ∧ B < π/2 ∧ C < π/2)
  (h_relation : c - b = 2 * b * cos A) :
  A = 2 * B ∧
  π/6 < B ∧ B < π/4 ∧
  sqrt 2 < a/b ∧ a/b < sqrt 3 ∧
  5 * sqrt 3 / 3 < 1 / tan B - 1 / tan A + 2 * sin A ∧
  1 / tan B - 1 / tan A + 2 * sin A < 3 :=
by sorry

end acute_triangle_properties_l300_30030


namespace chess_board_pawn_arrangements_l300_30094

theorem chess_board_pawn_arrangements (n : ℕ) (h : n = 5) : 
  (Finset.range n).card.factorial = 120 := by sorry

end chess_board_pawn_arrangements_l300_30094


namespace quadrilateral_exists_l300_30006

/-- A quadrilateral with side lengths and a diagonal -/
structure Quadrilateral :=
  (AB BC CD DA AC : ℝ)

/-- The triangle inequality theorem -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b

/-- Theorem: There exists a quadrilateral ABCD with diagonal AC, where AB = 10, BC = 9, CD = 19, DA = 5, and AC = 15 -/
theorem quadrilateral_exists : ∃ (q : Quadrilateral), 
  q.AB = 10 ∧ 
  q.BC = 9 ∧ 
  q.CD = 19 ∧ 
  q.DA = 5 ∧ 
  q.AC = 15 ∧
  triangle_inequality q.AB q.BC q.AC ∧
  triangle_inequality q.AC q.CD q.DA :=
sorry

end quadrilateral_exists_l300_30006


namespace polynomial_division_theorem_l300_30013

theorem polynomial_division_theorem (x : ℝ) : 
  (x - 3) * (x^4 + 3*x^3 - 7*x^2 - 10*x - 39) + (-47) = 
  x^5 - 16*x^3 + 11*x^2 - 9*x + 10 := by
  sorry

end polynomial_division_theorem_l300_30013


namespace price_restoration_l300_30004

theorem price_restoration (original_price : ℝ) (original_price_positive : original_price > 0) :
  let reduced_price := original_price * (1 - 0.15)
  let restoration_factor := (1 + 0.1765)
  (reduced_price * restoration_factor - original_price) / original_price < 0.0001 := by
sorry

end price_restoration_l300_30004


namespace outfit_count_l300_30056

/-- The number of different outfits that can be made with shirts, pants, and hats of different colors. -/
def num_outfits (red_shirts green_shirts blue_shirts : ℕ) 
                (pants : ℕ) 
                (red_hats green_hats blue_hats : ℕ) : ℕ :=
  (red_shirts * pants * (green_hats + blue_hats)) +
  (green_shirts * pants * (red_hats + blue_hats)) +
  (blue_shirts * pants * (red_hats + green_hats))

/-- Theorem stating the number of outfits under given conditions. -/
theorem outfit_count : 
  num_outfits 4 4 4 10 6 6 4 = 1280 :=
by sorry

end outfit_count_l300_30056


namespace largest_n_satisfying_conditions_l300_30045

theorem largest_n_satisfying_conditions : 
  ∃ (m : ℤ), (313 : ℤ)^2 = (m + 1)^3 - m^3 ∧ 
  ∃ (k : ℤ), (2 * 313 + 103 : ℤ) = k^2 ∧
  ∀ (n : ℤ), n > 313 → 
    (∃ (m : ℤ), n^2 = (m + 1)^3 - m^3 ∧ 
    ∃ (k : ℤ), (2 * n + 103 : ℤ) = k^2) → False :=
sorry

end largest_n_satisfying_conditions_l300_30045


namespace arithmetic_sequence_sum_specific_l300_30077

def arithmetic_sequence_sum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℤ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_specific :
  arithmetic_sequence_sum (-41) 3 2 = -437 := by
  sorry

end arithmetic_sequence_sum_specific_l300_30077


namespace linear_function_theorem_l300_30062

/-- A linear function satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The inverse function of f -/
noncomputable def f_inv (x : ℝ) : ℝ := sorry

theorem linear_function_theorem :
  (∀ x y t : ℝ, f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y) → -- f is linear
  (∀ x : ℝ, f x = 3 * f_inv x + 9) → -- f(x) = 3f^(-1)(x) + 9
  f 0 = 3 → -- f(0) = 3
  f_inv 3 = 0 → -- f^(-1)(3) = 0
  f 3 = 6 * Real.sqrt 3 := by -- f(3) = 6√3
sorry

end linear_function_theorem_l300_30062


namespace number_of_larger_planes_l300_30087

/-- Represents the number of airplanes --/
def total_planes : ℕ := 4

/-- Represents the capacity of smaller tanks in liters --/
def smaller_tank_capacity : ℕ := 60

/-- Represents the fuel cost per liter in cents --/
def fuel_cost_per_liter : ℕ := 50

/-- Represents the service charge per plane in cents --/
def service_charge : ℕ := 10000

/-- Represents the total cost to fill all planes in cents --/
def total_cost : ℕ := 55000

/-- Calculates the capacity of larger tanks --/
def larger_tank_capacity : ℕ := smaller_tank_capacity + smaller_tank_capacity / 2

/-- Calculates the fuel cost for a smaller plane in cents --/
def smaller_plane_fuel_cost : ℕ := smaller_tank_capacity * fuel_cost_per_liter

/-- Calculates the fuel cost for a larger plane in cents --/
def larger_plane_fuel_cost : ℕ := larger_tank_capacity * fuel_cost_per_liter

/-- Calculates the total cost for a smaller plane in cents --/
def smaller_plane_total_cost : ℕ := smaller_plane_fuel_cost + service_charge

/-- Calculates the total cost for a larger plane in cents --/
def larger_plane_total_cost : ℕ := larger_plane_fuel_cost + service_charge

/-- Proves that the number of larger planes is 2 --/
theorem number_of_larger_planes : 
  ∃ (n : ℕ), n + (total_planes - n) = total_planes ∧ 
             n * larger_plane_total_cost + (total_planes - n) * smaller_plane_total_cost = total_cost ∧
             n = 2 := by
  sorry

end number_of_larger_planes_l300_30087


namespace walters_hourly_wage_l300_30060

/-- Walter's work schedule and earnings allocation --/
structure WorkSchedule where
  days_per_week : ℕ
  hours_per_day : ℕ
  school_allocation_ratio : ℚ
  school_allocation_amount : ℚ

/-- Calculate Walter's hourly wage --/
def hourly_wage (w : WorkSchedule) : ℚ :=
  w.school_allocation_amount / w.school_allocation_ratio / (w.days_per_week * w.hours_per_day)

/-- Theorem: Walter's hourly wage is $5 --/
theorem walters_hourly_wage (w : WorkSchedule)
  (h1 : w.days_per_week = 5)
  (h2 : w.hours_per_day = 4)
  (h3 : w.school_allocation_ratio = 3/4)
  (h4 : w.school_allocation_amount = 75) :
  hourly_wage w = 5 := by
  sorry

end walters_hourly_wage_l300_30060


namespace snack_cost_l300_30052

/-- Given the following conditions:
    - There are 4 people
    - Each ticket costs $18
    - The total cost for tickets and snacks for all 4 people is $92
    Prove that the cost of a set of snacks is $5. -/
theorem snack_cost (num_people : ℕ) (ticket_price : ℕ) (total_cost : ℕ) :
  num_people = 4 →
  ticket_price = 18 →
  total_cost = 92 →
  (total_cost - num_people * ticket_price) / num_people = 5 := by
  sorry

end snack_cost_l300_30052


namespace fruit_store_problem_l300_30092

-- Define the types of fruits
inductive FruitType
| A
| B

-- Define the purchase data
structure PurchaseData where
  typeA : ℕ
  typeB : ℕ
  totalCost : ℕ

-- Define the problem parameters
def firstPurchase : PurchaseData := ⟨60, 40, 1520⟩
def secondPurchase : PurchaseData := ⟨30, 50, 1360⟩
def thirdPurchaseTotal : ℕ := 200
def thirdPurchaseMaxCost : ℕ := 3360
def typeASellingPrice : ℕ := 17
def typeBSellingPrice : ℕ := 30
def minProfit : ℕ := 800

-- Define the theorem
theorem fruit_store_problem :
  ∃ (priceA priceB : ℕ) (m : ℕ),
    -- Conditions for the first two purchases
    priceA * firstPurchase.typeA + priceB * firstPurchase.typeB = firstPurchase.totalCost ∧
    priceA * secondPurchase.typeA + priceB * secondPurchase.typeB = secondPurchase.totalCost ∧
    -- Conditions for the third purchase
    ∀ (x : ℕ),
      x ≤ thirdPurchaseTotal →
      priceA * x + priceB * (thirdPurchaseTotal - x) ≤ thirdPurchaseMaxCost →
      (typeASellingPrice - priceA) * (x - m) + (typeBSellingPrice - priceB) * (thirdPurchaseTotal - x - 3 * m) ≥ minProfit →
      -- Conclusion
      priceA = 12 ∧ priceB = 20 ∧ m ≤ 22 ∧
      ∀ (m' : ℕ), m' > m → 
        ¬(∃ (x : ℕ),
          x ≤ thirdPurchaseTotal ∧
          priceA * x + priceB * (thirdPurchaseTotal - x) ≤ thirdPurchaseMaxCost ∧
          (typeASellingPrice - priceA) * (x - m') + (typeBSellingPrice - priceB) * (thirdPurchaseTotal - x - 3 * m') ≥ minProfit) :=
by sorry

end fruit_store_problem_l300_30092


namespace a_representation_l300_30046

theorem a_representation (a : ℤ) (x y : ℤ) (h : 3 * a = x^2 + 2 * y^2) :
  ∃ (u v : ℤ), a = u^2 + 2 * v^2 := by
sorry

end a_representation_l300_30046


namespace product_of_roots_l300_30076

theorem product_of_roots (x : ℝ) : 
  (∃ α β : ℝ, α * β = -10 ∧ -20 = -2 * x^2 - 6 * x ↔ (x = α ∨ x = β)) :=
by sorry

end product_of_roots_l300_30076


namespace fraction_sum_from_hcf_lcm_and_sum_l300_30049

theorem fraction_sum_from_hcf_lcm_and_sum (m n : ℕ+) 
  (hcf : Nat.gcd m n = 6)
  (lcm : Nat.lcm m n = 210)
  (sum : m + n = 80) :
  (1 : ℚ) / m + (1 : ℚ) / n = 2 / 31.5 := by
  sorry

end fraction_sum_from_hcf_lcm_and_sum_l300_30049


namespace green_blue_difference_after_borders_l300_30036

/-- Represents the number of tiles in a hexagonal figure -/
structure HexagonalFigure where
  blue : ℕ
  green : ℕ

/-- Calculates the number of green tiles added by one border -/
def greenTilesPerBorder : ℕ := 6 * 3

/-- Theorem: The difference between green and blue tiles after adding two borders -/
theorem green_blue_difference_after_borders (initial : HexagonalFigure) :
  let newFigure := HexagonalFigure.mk
    initial.blue
    (initial.green + 2 * greenTilesPerBorder)
  newFigure.green - newFigure.blue = 27 :=
by
  sorry

end green_blue_difference_after_borders_l300_30036


namespace wizard_elixir_combinations_l300_30066

-- Define the number of flowers
def num_flowers : ℕ := 4

-- Define the number of gems
def num_gems : ℕ := 6

-- Define the number of invalid combinations
def num_invalid : ℕ := 3

-- Theorem statement
theorem wizard_elixir_combinations :
  (num_flowers * num_gems) - num_invalid = 21 := by
  sorry

end wizard_elixir_combinations_l300_30066


namespace max_value_of_expression_l300_30095

theorem max_value_of_expression (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a ∈ ({2, 3, 6} : Set ℕ) →
  b ∈ ({2, 3, 6} : Set ℕ) →
  c ∈ ({2, 3, 6} : Set ℕ) →
  (a : ℚ) / ((b : ℚ) / (c : ℚ)) ≤ 9 →
  (∃ a' b' c' : ℕ, 
    a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c' ∧
    a' ∈ ({2, 3, 6} : Set ℕ) ∧
    b' ∈ ({2, 3, 6} : Set ℕ) ∧
    c' ∈ ({2, 3, 6} : Set ℕ) ∧
    (a' : ℚ) / ((b' : ℚ) / (c' : ℚ)) = 9) →
  a = 3 :=
by sorry

end max_value_of_expression_l300_30095


namespace log_stack_sum_l300_30084

/-- Given an arithmetic sequence with 11 terms, starting at 5 and ending at 15,
    prove that the sum of all terms is 110. -/
theorem log_stack_sum : 
  let n : ℕ := 11  -- number of terms
  let a : ℕ := 5   -- first term
  let l : ℕ := 15  -- last term
  n * (a + l) / 2 = 110 := by
  sorry

end log_stack_sum_l300_30084


namespace shark_sightings_l300_30003

theorem shark_sightings (cape_may daytona_beach : ℕ) : 
  cape_may + daytona_beach = 40 →
  cape_may = 2 * daytona_beach - 8 →
  cape_may = 24 :=
by sorry

end shark_sightings_l300_30003


namespace count_convex_cyclic_quads_l300_30014

/-- A convex cyclic quadrilateral with integer sides --/
structure ConvexCyclicQuad where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  sum_eq_40 : a + b + c + d = 40
  convex : a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c
  ordered : a ≥ b ∧ b ≥ c ∧ c ≥ d
  has_odd_side : Odd a ∨ Odd b ∨ Odd c ∨ Odd d

/-- The count of valid quadrilaterals --/
def count_valid_quads : ℕ := sorry

theorem count_convex_cyclic_quads : count_valid_quads = 760 := by
  sorry

end count_convex_cyclic_quads_l300_30014


namespace solution_set_quadratic_inequality_l300_30040

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end solution_set_quadratic_inequality_l300_30040


namespace exponent_division_l300_30029

theorem exponent_division (a : ℝ) : a^6 / a^2 = a^4 := by
  sorry

end exponent_division_l300_30029


namespace endpoint_coordinate_sum_l300_30098

/-- Given a line segment with one endpoint at (-3, -15) and midpoint at (2, -5),
    the sum of coordinates of the other endpoint is 12 -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (2 : ℝ) = (-3 + x) / 2 → 
    (-5 : ℝ) = (-15 + y) / 2 → 
    x + y = 12 := by
  sorry

end endpoint_coordinate_sum_l300_30098


namespace notebook_difference_l300_30031

theorem notebook_difference (tara_spent lea_spent : ℚ) 
  (h1 : tara_spent = 5.20)
  (h2 : lea_spent = 7.80)
  (h3 : ∃ (price : ℚ), price > 1 ∧ 
    ∃ (tara_count lea_count : ℕ), 
      tara_count * price = tara_spent ∧ 
      lea_count * price = lea_spent) :
  ∃ (price : ℚ) (tara_count lea_count : ℕ), 
    price > 1 ∧
    tara_count * price = tara_spent ∧
    lea_count * price = lea_spent ∧
    lea_count = tara_count + 2 :=
by sorry

end notebook_difference_l300_30031


namespace three_real_roots_l300_30000

/-- The polynomial f(x) = x^3 - 6x^2 + 9x - 2 -/
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - 2

/-- Theorem: The equation f(x) = 0 has exactly three real roots -/
theorem three_real_roots : ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f x = 0 := by sorry

end three_real_roots_l300_30000


namespace binary_111011_equals_59_l300_30027

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_111011_equals_59 :
  binary_to_decimal [true, true, false, true, true, true] = 59 := by
  sorry

end binary_111011_equals_59_l300_30027


namespace square_difference_65_35_l300_30067

theorem square_difference_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end square_difference_65_35_l300_30067


namespace simplify_expression_l300_30016

theorem simplify_expression (x : ℝ) (h : x ≠ 1) :
  (x^2 / (x + 1) - x + 1) / ((x^2 - 1) / (x^2 + 2*x + 1)) = 1 / (x - 1) :=
by sorry

end simplify_expression_l300_30016


namespace bart_earnings_l300_30019

/-- The amount of money Bart earns per question answered -/
def money_per_question : ℚ := 0.2

/-- The number of questions in each survey -/
def questions_per_survey : ℕ := 10

/-- The number of surveys Bart completed on Monday -/
def surveys_monday : ℕ := 3

/-- The number of surveys Bart completed on Tuesday -/
def surveys_tuesday : ℕ := 4

/-- Theorem stating the total money Bart earned over two days -/
theorem bart_earnings : 
  (surveys_monday + surveys_tuesday) * questions_per_survey * money_per_question = 14 := by
  sorry

end bart_earnings_l300_30019


namespace solve_m_n_l300_30073

def A : Set ℝ := {3, 5}
def B (m n : ℝ) : Set ℝ := {x | x^2 + m*x + n = 0}

theorem solve_m_n :
  ∃ (m n : ℝ),
    (A ∪ B m n = A) ∧
    (A ∩ B m n = {5}) ∧
    m = -10 ∧
    n = 25 := by
  sorry

end solve_m_n_l300_30073


namespace fraction_multiplication_l300_30065

theorem fraction_multiplication :
  (3 : ℚ) / 4 * 4 / 5 * 5 / 6 * 6 / 7 = 3 / 7 := by
  sorry

end fraction_multiplication_l300_30065


namespace chef_cakes_l300_30083

/-- Given a total number of eggs, eggs put in the fridge, and eggs needed per cake,
    calculate the number of cakes that can be made. -/
def cakes_made (total_eggs : ℕ) (fridge_eggs : ℕ) (eggs_per_cake : ℕ) : ℕ :=
  (total_eggs - fridge_eggs) / eggs_per_cake

/-- Prove that given 60 total eggs, with 10 eggs put in the fridge,
    and 5 eggs needed for one cake, the number of cakes the chef can make is 10. -/
theorem chef_cakes :
  cakes_made 60 10 5 = 10 := by
sorry

end chef_cakes_l300_30083


namespace simplify_polynomial_l300_30021

theorem simplify_polynomial (x : ℝ) : 
  x * (4 * x^3 - 3) - 6 * (x^2 - 3*x + 9) = 4 * x^4 - 6 * x^2 + 15 * x - 54 := by
  sorry

end simplify_polynomial_l300_30021


namespace kramer_packing_theorem_l300_30053

/-- Kramer's packing rate in cases per hour -/
def packing_rate : ℝ := 120

/-- Number of boxes Kramer packs per minute -/
def boxes_per_minute : ℕ := 10

/-- Number of boxes in one case -/
def boxes_per_case : ℕ := 5

/-- Number of cases Kramer packs in 2 hours -/
def cases_in_two_hours : ℕ := 240

/-- The number of cases Kramer can pack in x hours -/
def cases_packed (x : ℝ) : ℝ := packing_rate * x

theorem kramer_packing_theorem (x : ℝ) : 
  cases_packed x = packing_rate * x ∧
  (boxes_per_minute : ℝ) * 60 / boxes_per_case = packing_rate ∧
  cases_in_two_hours = packing_rate * 2 :=
sorry

end kramer_packing_theorem_l300_30053


namespace coefficient_not_fifty_l300_30032

theorem coefficient_not_fifty :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 →
  (Nat.choose 5 k) * (2^(5-k)) ≠ 50 := by
sorry

end coefficient_not_fifty_l300_30032


namespace probability_of_six_or_less_l300_30002

def num_red_balls : ℕ := 4
def num_black_balls : ℕ := 3
def total_balls : ℕ := num_red_balls + num_black_balls
def num_drawn : ℕ := 4
def red_points : ℕ := 1
def black_points : ℕ := 3

def score (red_drawn : ℕ) : ℕ :=
  red_drawn * red_points + (num_drawn - red_drawn) * black_points

def probability_of_score (s : ℕ) : ℚ :=
  (Nat.choose num_red_balls s * Nat.choose num_black_balls (num_drawn - s)) /
  Nat.choose total_balls num_drawn

theorem probability_of_six_or_less :
  probability_of_score 4 + probability_of_score 3 = 13 / 35 := by
  sorry

end probability_of_six_or_less_l300_30002


namespace sqrt_2x_minus_1_real_l300_30088

theorem sqrt_2x_minus_1_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 2 * x - 1) ↔ x ≥ 1 / 2 := by sorry

end sqrt_2x_minus_1_real_l300_30088


namespace solution_set_reciprocal_gt_one_l300_30023

theorem solution_set_reciprocal_gt_one :
  {x : ℝ | (1 : ℝ) / x > 1} = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

end solution_set_reciprocal_gt_one_l300_30023


namespace root_product_theorem_l300_30038

theorem root_product_theorem (n r : ℝ) (c d : ℝ) : 
  (c^2 - n*c + 3 = 0) →
  (d^2 - n*d + 3 = 0) →
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) →
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) →
  s = 16/3 := by
sorry

end root_product_theorem_l300_30038


namespace largest_g_is_correct_l300_30044

/-- The largest positive integer g for which there exists exactly one pair of positive integers (a, b) satisfying 5a + gb = 70 -/
def largest_g : ℕ := 65

/-- The unique pair of positive integers (a, b) satisfying 5a + (largest_g)b = 70 -/
def unique_pair : ℕ × ℕ := (1, 1)

theorem largest_g_is_correct :
  (∀ g : ℕ, g > largest_g →
    ¬(∃! p : ℕ × ℕ, p.1 > 0 ∧ p.2 > 0 ∧ 5 * p.1 + g * p.2 = 70)) ∧
  (∃! p : ℕ × ℕ, p.1 > 0 ∧ p.2 > 0 ∧ 5 * p.1 + largest_g * p.2 = 70) ∧
  (unique_pair.1 > 0 ∧ unique_pair.2 > 0 ∧ 5 * unique_pair.1 + largest_g * unique_pair.2 = 70) :=
by sorry

#check largest_g_is_correct

end largest_g_is_correct_l300_30044


namespace total_amount_received_l300_30093

def lottery_winnings : ℚ := 555850
def num_students : ℕ := 500
def fraction : ℚ := 3 / 10000

theorem total_amount_received :
  (lottery_winnings * fraction * num_students : ℚ) = 833775 := by
  sorry

end total_amount_received_l300_30093


namespace greatest_whole_number_satisfying_inequality_l300_30075

theorem greatest_whole_number_satisfying_inequality :
  ∀ n : ℤ, (∀ x : ℤ, x ≤ n → 4 * x - 3 < 2 - x) → n ≤ 0 :=
by sorry

end greatest_whole_number_satisfying_inequality_l300_30075


namespace product_base5_digit_sum_l300_30017

/-- Converts a base-5 number represented as a list of digits to base-10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base-10 number to base-5, returning a list of digits --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec toDigits (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else toDigits (m / 5) ((m % 5) :: acc)
    toDigits n []

/-- Sums the digits of a number represented as a list --/
def sumDigits (digits : List Nat) : Nat :=
  digits.sum

theorem product_base5_digit_sum (n1 n2 : List Nat) :
  sumDigits (base10ToBase5 (base5ToBase10 n1 * base5ToBase10 n2)) = 8 :=
sorry

end product_base5_digit_sum_l300_30017


namespace complex_arithmetic_expression_l300_30001

theorem complex_arithmetic_expression : 
  10 - 10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.3 := by
  sorry

end complex_arithmetic_expression_l300_30001


namespace composite_ratio_l300_30057

def first_seven_composites : List Nat := [4, 6, 8, 9, 10, 12, 14]
def next_seven_composites : List Nat := [15, 16, 18, 20, 21, 22, 24]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

theorem composite_ratio :
  (product_of_list first_seven_composites) / 
  (product_of_list next_seven_composites) = 1 / 176 := by
  sorry

end composite_ratio_l300_30057


namespace quadratic_symmetry_axis_l300_30015

/-- A quadratic function of the form y = x^2 - bx + 2c with axis of symmetry x = 3 has b = 6 -/
theorem quadratic_symmetry_axis (b c : ℝ) : 
  (∀ x y : ℝ, y = x^2 - b*x + 2*c → (∀ y1 y2 : ℝ, (3 - x)^2 = (3 + x)^2 → y1 = y2)) → 
  b = 6 :=
sorry

end quadratic_symmetry_axis_l300_30015


namespace sandy_turnips_count_undetermined_l300_30080

/-- Represents the number of vegetables grown by a person -/
structure VegetableCount where
  carrots : ℕ
  turnips : ℕ

/-- The given information about Sandy and Mary's vegetable growth -/
def given : Prop :=
  ∃ (sandy : VegetableCount) (mary : VegetableCount),
    sandy.carrots = 8 ∧
    mary.carrots = 6 ∧
    sandy.carrots + mary.carrots = 14

/-- The statement that Sandy's turnip count cannot be determined -/
def sandy_turnips_undetermined : Prop :=
  ∀ (n : ℕ),
    (∃ (sandy : VegetableCount) (mary : VegetableCount),
      sandy.carrots = 8 ∧
      mary.carrots = 6 ∧
      sandy.carrots + mary.carrots = 14 ∧
      sandy.turnips = n) →
    (∃ (sandy : VegetableCount) (mary : VegetableCount),
      sandy.carrots = 8 ∧
      mary.carrots = 6 ∧
      sandy.carrots + mary.carrots = 14 ∧
      sandy.turnips ≠ n)

theorem sandy_turnips_count_undetermined :
  given → sandy_turnips_undetermined :=
by
  sorry

end sandy_turnips_count_undetermined_l300_30080


namespace trigonometric_equation_solutions_l300_30091

theorem trigonometric_equation_solutions (x : ℝ) :
  1 + Real.sin x - Real.cos (5 * x) - Real.sin (7 * x) = 2 * (Real.cos (3 * x / 2))^2 ↔
  (∃ k : ℤ, x = π / 8 * (2 * k + 1)) ∨ (∃ n : ℤ, x = π / 4 * (4 * n - 1)) :=
by sorry

end trigonometric_equation_solutions_l300_30091


namespace triangle_inequality_triangle_inequality_certain_event_l300_30068

/-- Definition of a triangle -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b

/-- The triangle inequality theorem -/
theorem triangle_inequality (t : Triangle) : 
  (t.a + t.b > t.c) ∧ (t.b + t.c > t.a) ∧ (t.c + t.a > t.b) :=
sorry

/-- Proof that the triangle inequality is a certain event -/
theorem triangle_inequality_certain_event : 
  ∀ (t : Triangle), (t.a + t.b > t.c) ∧ (t.b + t.c > t.a) ∧ (t.c + t.a > t.b) :=
sorry

end triangle_inequality_triangle_inequality_certain_event_l300_30068


namespace orange_cells_theorem_l300_30070

/-- Represents the possible outcomes of orange cells on the board -/
inductive OrangeCellsOutcome
  | lower  : OrangeCellsOutcome  -- represents 2021 * 2020
  | higher : OrangeCellsOutcome  -- represents 2022 * 2020

/-- The size of one side of the square board -/
def boardSize : Nat := 2022

/-- The size of one side of the paintable square -/
def squareSize : Nat := 2

/-- Represents the game rules and outcomes -/
structure GameBoard where
  size : Nat
  squareSize : Nat
  possibleOutcomes : List OrangeCellsOutcome

/-- The main theorem to prove -/
theorem orange_cells_theorem (board : GameBoard) 
  (h1 : board.size = boardSize) 
  (h2 : board.squareSize = squareSize) 
  (h3 : board.possibleOutcomes = [OrangeCellsOutcome.lower, OrangeCellsOutcome.higher]) : 
  ∃ (n : Nat), (n = 2021 * 2020 ∨ n = 2022 * 2020) ∧ 
  (∀ (m : Nat), m = 2021 * 2020 ∨ m = 2022 * 2020 → 
    ∃ (outcome : OrangeCellsOutcome), outcome ∈ board.possibleOutcomes ∧
    (outcome = OrangeCellsOutcome.lower → m = 2021 * 2020) ∧
    (outcome = OrangeCellsOutcome.higher → m = 2022 * 2020)) :=
  sorry


end orange_cells_theorem_l300_30070


namespace geometric_sequence_angle_l300_30085

theorem geometric_sequence_angle (a : ℕ → ℝ) (α : ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  (a 1 * a 8 = -Real.sqrt 3 * Real.sin α) →  -- root product condition
  (a 1 + a 8 = 2 * Real.sin α) →  -- root sum condition
  ((a 1 + a 8)^2 = 2 * a 3 * a 6 + 6) →  -- given equation
  (0 < α ∧ α < π / 2) →  -- acute angle condition
  α = π / 3 := by
sorry

end geometric_sequence_angle_l300_30085


namespace triangle_problem_l300_30033

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a > t.c ∧
  t.a * t.c * (1/3) = 2 ∧  -- Vector BA · Vector BC = 2 and cos B = 1/3
  t.b = 3

-- Theorem statement
theorem triangle_problem (t : Triangle) 
  (h : triangle_conditions t) : 
  t.a = 3 ∧ t.c = 2 ∧ Real.cos (t.B - t.C) = 23/27 := by
  sorry


end triangle_problem_l300_30033


namespace subtracted_number_l300_30047

theorem subtracted_number (x : ℝ) : 3889 + 12.808 - x = 3854.002 → x = 47.806 := by
  sorry

end subtracted_number_l300_30047


namespace average_of_first_45_results_l300_30020

theorem average_of_first_45_results
  (n₁ : ℕ)
  (n₂ : ℕ)
  (a₂ : ℝ)
  (total_avg : ℝ)
  (h₁ : n₁ = 45)
  (h₂ : n₂ = 25)
  (h₃ : a₂ = 45)
  (h₄ : total_avg = 32.142857142857146)
  (h₅ : (n₁ : ℝ) * a₁ + (n₂ : ℝ) * a₂ = (n₁ + n₂ : ℝ) * total_avg) :
  a₁ = 25 :=
by sorry

end average_of_first_45_results_l300_30020


namespace apricot_tea_calories_l300_30048

/-- Represents the composition of the apricot tea -/
structure ApricotTea where
  apricot_juice : ℝ
  honey : ℝ
  water : ℝ

/-- Calculates the total calories in the apricot tea mixture -/
def total_calories (tea : ApricotTea) : ℝ :=
  tea.apricot_juice * 0.3 + tea.honey * 3.04

/-- Calculates the total weight of the apricot tea mixture -/
def total_weight (tea : ApricotTea) : ℝ :=
  tea.apricot_juice + tea.honey + tea.water

/-- Theorem: 250g of Nathan's apricot tea contains 98.5 calories -/
theorem apricot_tea_calories :
  let tea : ApricotTea := { apricot_juice := 150, honey := 50, water := 300 }
  let caloric_density : ℝ := total_calories tea / total_weight tea
  250 * caloric_density = 98.5 := by
  sorry

#check apricot_tea_calories

end apricot_tea_calories_l300_30048


namespace bakery_stop_difference_l300_30018

/-- Represents the distances between locations in Kona's trip -/
structure TripDistances where
  apartment_to_bakery : ℕ
  bakery_to_grandma : ℕ
  grandma_to_apartment : ℕ

/-- Calculates the additional miles driven with a bakery stop -/
def additional_miles (d : TripDistances) : ℕ :=
  (d.apartment_to_bakery + d.bakery_to_grandma + d.grandma_to_apartment) -
  (2 * d.grandma_to_apartment)

/-- Theorem stating that the additional miles driven with a bakery stop is 6 -/
theorem bakery_stop_difference (d : TripDistances)
  (h1 : d.apartment_to_bakery = 9)
  (h2 : d.bakery_to_grandma = 24)
  (h3 : d.grandma_to_apartment = 27) :
  additional_miles d = 6 := by
  sorry


end bakery_stop_difference_l300_30018


namespace dice_faces_theorem_l300_30064

theorem dice_faces_theorem (n m : ℕ) : 
  (n ≥ 1) → 
  (m ≥ 1) → 
  (∀ i ∈ Finset.range n, ∀ j ∈ Finset.range m, 
    (Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 8) (Finset.product (Finset.range n) (Finset.range m))).card = 
    (1/2 : ℚ) * (Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 11) (Finset.product (Finset.range n) (Finset.range m))).card) →
  ((Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 13) (Finset.product (Finset.range n) (Finset.range m))).card : ℚ) / (n * m) = 1/15 →
  (∃ k : ℕ, n + m = 5 * k) →
  (∀ n' m' : ℕ, n' + m' < n + m → 
    ¬((n' ≥ 1) ∧ 
      (m' ≥ 1) ∧ 
      (∀ i ∈ Finset.range n', ∀ j ∈ Finset.range m', 
        (Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 8) (Finset.product (Finset.range n') (Finset.range m'))).card = 
        (1/2 : ℚ) * (Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 11) (Finset.product (Finset.range n') (Finset.range m'))).card) ∧
      ((Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 13) (Finset.product (Finset.range n') (Finset.range m'))).card : ℚ) / (n' * m') = 1/15 ∧
      (∃ k : ℕ, n' + m' = 5 * k))) →
  n + m = 25 := by
sorry

end dice_faces_theorem_l300_30064


namespace hexagonal_diagram_impossible_l300_30043

/-- Represents a hexagonal diagram filled with numbers -/
structure HexagonalDiagram :=
  (first_row : Fin 6 → ℕ)
  (is_valid : ∀ i : Fin 6, first_row i ∈ Finset.range 22)

/-- Calculates the sum of all numbers in the hexagonal diagram -/
def hexagon_sum (h : HexagonalDiagram) : ℕ :=
  6 * h.first_row 0 + 20 * h.first_row 1 + 34 * h.first_row 2 +
  34 * h.first_row 3 + 20 * h.first_row 4 + 6 * h.first_row 5

/-- The sum of numbers from 1 to 21 -/
def sum_1_to_21 : ℕ := (21 * 22) / 2

/-- Theorem stating the impossibility of filling the hexagonal diagram -/
theorem hexagonal_diagram_impossible :
  ¬ ∃ (h : HexagonalDiagram), hexagon_sum h = sum_1_to_21 :=
sorry

end hexagonal_diagram_impossible_l300_30043


namespace pyramid_x_value_l300_30086

/-- Pyramid represents a numerical pyramid where each number below the top row
    is the product of the number to the right and the number to the left in the row immediately above it. -/
structure Pyramid where
  top_left : ℕ
  middle : ℕ
  bottom_left : ℕ
  x : ℕ

/-- Given a Pyramid, this theorem proves that x must be 4 -/
theorem pyramid_x_value (p : Pyramid) (h1 : p.top_left = 35) (h2 : p.middle = 700) (h3 : p.bottom_left = 5)
  (h4 : p.middle = p.top_left * (p.middle / p.top_left))
  (h5 : p.middle / p.top_left = p.bottom_left * p.x) :
  p.x = 4 := by
  sorry

end pyramid_x_value_l300_30086


namespace not_divisible_by_2006_l300_30081

theorem not_divisible_by_2006 (k : ℤ) : ¬(2006 ∣ (k^2 + k + 1)) := by
  sorry

end not_divisible_by_2006_l300_30081


namespace no_square_prime_ratio_in_triangular_sequence_l300_30007

theorem no_square_prime_ratio_in_triangular_sequence (p : ℕ) (hp : Prime p) :
  ∀ (x y l : ℕ), l ≥ 1 →
    (x * (x + 1)) / (y * (y + 1)) ≠ p^(2 * l) := by
  sorry

end no_square_prime_ratio_in_triangular_sequence_l300_30007


namespace two_primes_not_congruent_to_one_l300_30037

theorem two_primes_not_congruent_to_one (p : Nat) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ (q r : Nat), q ≠ r ∧ q.Prime ∧ r.Prime ∧ 2 ≤ q ∧ q ≤ p - 2 ∧ 2 ≤ r ∧ r ≤ p - 2 ∧
  ¬(q^(p-1) ≡ 1 [MOD p^2]) ∧ ¬(r^(p-1) ≡ 1 [MOD p^2]) := by
  sorry

end two_primes_not_congruent_to_one_l300_30037


namespace fraction_numerator_l300_30022

theorem fraction_numerator (y : ℝ) (x : ℤ) (h1 : y > 0) :
  (x : ℝ) / y + 3 * y / 10 = (35 : ℝ) / 100 * y → x = 32 := by
  sorry

end fraction_numerator_l300_30022


namespace number_equation_solution_l300_30058

theorem number_equation_solution :
  ∀ B : ℝ, (4 * B + 4 = 33) → B = 7.25 := by
  sorry

end number_equation_solution_l300_30058


namespace trapezium_area_l300_30024

/-- The area of a trapezium with given dimensions -/
theorem trapezium_area (a b h : ℝ) (ha : a = 4) (hb : b = 5) (hh : h = 6) :
  (1/2 : ℝ) * (a + b) * h = 27 :=
by sorry

end trapezium_area_l300_30024


namespace units_digit_of_m_squared_plus_3_to_m_l300_30042

def m : ℕ := 2010^2 + 2^2010

theorem units_digit_of_m_squared_plus_3_to_m (m : ℕ) : (m^2 + 3^m) % 10 = 7 := by
  sorry

end units_digit_of_m_squared_plus_3_to_m_l300_30042


namespace two_std_dev_below_mean_l300_30012

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  stdDev : ℝ

/-- The value that is exactly n standard deviations less than the mean -/
def valueNStdDevBelow (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.stdDev

/-- Theorem: For a normal distribution with mean 17.5 and standard deviation 2.5,
    the value that is exactly 2 standard deviations less than the mean is 12.5 -/
theorem two_std_dev_below_mean :
  let d : NormalDistribution := { mean := 17.5, stdDev := 2.5 }
  valueNStdDevBelow d 2 = 12.5 := by
  sorry

end two_std_dev_below_mean_l300_30012


namespace sum_of_powers_of_three_l300_30089

theorem sum_of_powers_of_three : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end sum_of_powers_of_three_l300_30089


namespace train_length_l300_30054

/-- The length of a train given specific passing times -/
theorem train_length : ∃ (L : ℝ), 
  (L / 24 = (L + 650) / 89) ∧ L = 240 := by sorry

end train_length_l300_30054


namespace bond_face_value_l300_30025

/-- Proves that the face value of a bond is 5000 given specific conditions --/
theorem bond_face_value (F : ℝ) : 
  (0.10 * F = 0.065 * 7692.307692307692) → F = 5000 := by
  sorry

end bond_face_value_l300_30025


namespace solution_set_implies_a_equals_one_l300_30009

theorem solution_set_implies_a_equals_one (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x < 0 ↔ 0 < x ∧ x < 1) → 
  a = 1 := by
sorry

end solution_set_implies_a_equals_one_l300_30009


namespace five_vents_per_zone_l300_30061

/-- Represents an HVAC system -/
structure HVACSystem where
  totalCost : ℕ
  numZones : ℕ
  costPerVent : ℕ

/-- Calculates the number of vents in each zone of an HVAC system -/
def ventsPerZone (system : HVACSystem) : ℕ :=
  (system.totalCost / system.costPerVent) / system.numZones

/-- Theorem: For the given HVAC system, there are 5 vents in each zone -/
theorem five_vents_per_zone (system : HVACSystem)
    (h1 : system.totalCost = 20000)
    (h2 : system.numZones = 2)
    (h3 : system.costPerVent = 2000) :
    ventsPerZone system = 5 := by
  sorry

#eval ventsPerZone { totalCost := 20000, numZones := 2, costPerVent := 2000 }

end five_vents_per_zone_l300_30061


namespace largest_constant_inequality_largest_constant_is_three_equality_condition_l300_30072

theorem largest_constant_inequality (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) :=
sorry

theorem largest_constant_is_three :
  ∀ C > 3, ∃ x₁ x₂ x₃ x₄ x₅ x₆ : ℝ,
    (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 < C * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) :=
sorry

theorem equality_condition (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 = 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) ↔
  (x₁ + x₄ = x₂ + x₅) ∧ (x₂ + x₅ = x₃ + x₆) :=
sorry

end largest_constant_inequality_largest_constant_is_three_equality_condition_l300_30072


namespace usual_time_is_36_l300_30008

-- Define the usual time T as a positive real number
variable (T : ℝ) (hT : T > 0)

-- Define the relationship between normal speed and reduced speed
def reduced_speed_time : ℝ := T + 12

-- Theorem stating that the usual time T is 36 minutes
theorem usual_time_is_36 : T = 36 := by
  sorry

end usual_time_is_36_l300_30008


namespace distance_between_points_l300_30011

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (4, -6)
  let p2 : ℝ × ℝ := (-8, 5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 265 := by
  sorry

end distance_between_points_l300_30011


namespace cube_of_99999_l300_30079

theorem cube_of_99999 : 
  let N : ℕ := 99999
  N^3 = 999970000299999 := by
  sorry

end cube_of_99999_l300_30079


namespace binomial_coefficient_n_plus_one_choose_n_l300_30035

theorem binomial_coefficient_n_plus_one_choose_n (n : ℕ+) : 
  Nat.choose (n + 1) n = n + 1 := by
  sorry

end binomial_coefficient_n_plus_one_choose_n_l300_30035


namespace horner_method_example_l300_30041

def f (x : ℝ) : ℝ := 3 * x^3 + x - 3

theorem horner_method_example : f 3 = 28 := by
  sorry

end horner_method_example_l300_30041


namespace cubes_fill_box_l300_30097

def box_length : ℕ := 8
def box_width : ℕ := 4
def box_height : ℕ := 12
def cube_size : ℕ := 2

theorem cubes_fill_box : 
  (box_length / cube_size) * (box_width / cube_size) * (box_height / cube_size) * (cube_size^3) = 
  box_length * box_width * box_height := by
  sorry

end cubes_fill_box_l300_30097


namespace complex_modulus_equality_l300_30028

theorem complex_modulus_equality (n : ℝ) :
  n > 0 → Complex.abs (4 + n * Complex.I) = 4 * Real.sqrt 13 → n = 8 * Real.sqrt 3 := by
  sorry

end complex_modulus_equality_l300_30028


namespace atBatsAgainstLeft_is_180_l300_30059

/-- Represents the batting statistics of a baseball player -/
structure BattingStats where
  totalAtBats : ℕ
  totalHits : ℕ
  avgAgainstLeft : ℚ
  avgAgainstRight : ℚ

/-- Calculates the number of at-bats against left-handed pitchers -/
def atBatsAgainstLeft (stats : BattingStats) : ℕ :=
  sorry

/-- Theorem stating that the number of at-bats against left-handed pitchers is 180 -/
theorem atBatsAgainstLeft_is_180 (stats : BattingStats) 
  (h1 : stats.totalAtBats = 600)
  (h2 : stats.totalHits = 192)
  (h3 : stats.avgAgainstLeft = 1/4)
  (h4 : stats.avgAgainstRight = 7/20)
  (h5 : (stats.totalHits : ℚ) / stats.totalAtBats = 8/25) :
  atBatsAgainstLeft stats = 180 :=
by
  sorry

end atBatsAgainstLeft_is_180_l300_30059


namespace equilateral_triangle_area_l300_30051

/-- Family of lines parameterized by t -/
def C (t : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x * Real.cos t + (y + 1) * Real.sin t = 2

/-- Predicate for three lines from C forming an equilateral triangle -/
def forms_equilateral_triangle (t₁ t₂ t₃ : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ x₃ y₃,
    C t₁ x₁ y₁ ∧ C t₂ x₂ y₂ ∧ C t₃ x₃ y₃ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₂ - x₃)^2 + (y₂ - y₃)^2 ∧
    (x₂ - x₃)^2 + (y₂ - y₃)^2 = (x₃ - x₁)^2 + (y₃ - y₁)^2

/-- The area of the equilateral triangle formed by three lines from C -/
def triangle_area (t₁ t₂ t₃ : ℝ) : ℝ := sorry

theorem equilateral_triangle_area :
  ∀ t₁ t₂ t₃, forms_equilateral_triangle t₁ t₂ t₃ →
  triangle_area t₁ t₂ t₃ = 4 * Real.sqrt 3 :=
sorry

end equilateral_triangle_area_l300_30051


namespace area_of_triangle_AFK_l300_30071

/-- Parabola with equation y² = 8x, focus F(2, 0), and directrix intersecting x-axis at K(-2, 0) -/
structure Parabola where
  F : ℝ × ℝ := (2, 0)
  K : ℝ × ℝ := (-2, 0)

/-- Point on the parabola -/
structure PointOnParabola (p : Parabola) where
  A : ℝ × ℝ
  on_parabola : A.2^2 = 8 * A.1
  distance_condition : (A.1 + 2)^2 + A.2^2 = 2 * ((A.1 - 2)^2 + A.2^2)

/-- The area of triangle AFK is 8 -/
theorem area_of_triangle_AFK (p : Parabola) (point : PointOnParabola p) :
  (1 / 2 : ℝ) * 4 * |point.A.2| = 8 := by sorry

end area_of_triangle_AFK_l300_30071


namespace sequences_1992_values_l300_30090

/-- Two sequences of integer numbers satisfying given conditions -/
def Sequences (a b : ℕ → ℤ) : Prop :=
  (a 0 = 0) ∧ (b 0 = 8) ∧
  (∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n + 2) ∧
  (∀ n : ℕ, b (n + 2) = 2 * b (n + 1) - b n) ∧
  (∀ n : ℕ, ∃ k : ℤ, (a n)^2 + (b n)^2 = k^2)

/-- The theorem to be proved -/
theorem sequences_1992_values (a b : ℕ → ℤ) (h : Sequences a b) :
  ((a 1992 = 31872 ∧ b 1992 = 31880) ∨ (a 1992 = -31872 ∧ b 1992 = -31864)) :=
sorry

end sequences_1992_values_l300_30090


namespace speaking_brother_is_tryalya_l300_30074

-- Define the brothers
inductive Brother
| Tralyalya
| Tryalya

-- Define the card suits
inductive Suit
| Black
| Red

-- Define the statement about having a black suit card
def claims_black_suit (b : Brother) : Prop :=
  match b with
  | Brother.Tralyalya => true
  | Brother.Tryalya => true

-- Define the rule that the brother with the black suit card cannot tell the truth
axiom black_suit_rule : ∀ (b : Brother), 
  (∃ (s : Suit), s = Suit.Black ∧ claims_black_suit b) → ¬(claims_black_suit b)

-- Theorem: The speaking brother must be Tryalya and he must have the black suit card
theorem speaking_brother_is_tryalya : 
  ∃ (b : Brother) (s : Suit), 
    b = Brother.Tryalya ∧ 
    s = Suit.Black ∧ 
    claims_black_suit b :=
sorry

end speaking_brother_is_tryalya_l300_30074


namespace range_of_t_squared_minus_one_l300_30005

theorem range_of_t_squared_minus_one :
  ∀ z : ℝ, ∃ x y : ℝ, x ≠ 0 ∧ (y / x)^2 - 1 = z :=
by sorry

end range_of_t_squared_minus_one_l300_30005


namespace point_B_in_third_quadrant_l300_30099

/-- Given that point A (-x, y-1) is in the fourth quadrant, 
    prove that point B (y-1, x) is in the third quadrant. -/
theorem point_B_in_third_quadrant 
  (x y : ℝ) 
  (h_fourth : -x > 0 ∧ y - 1 < 0) : 
  y - 1 < 0 ∧ x < 0 := by
sorry

end point_B_in_third_quadrant_l300_30099


namespace fraction_evaluation_l300_30078

theorem fraction_evaluation : 
  (20-18+16-14+12-10+8-6+4-2) / (2-4+6-8+10-12+14-16+18) = 1 := by
  sorry

end fraction_evaluation_l300_30078


namespace marble_distribution_l300_30034

theorem marble_distribution (a : ℕ) : 
  let angela := a
  let brian := 2 * a
  let caden := angela + brian
  let daryl := 2 * caden
  angela + brian + caden + daryl = 144 → a = 12 := by
sorry

end marble_distribution_l300_30034


namespace calculation_proof_l300_30026

theorem calculation_proof : 211 * 555 + 445 * 789 + 555 * 789 + 211 * 445 = 10^6 := by
  sorry

end calculation_proof_l300_30026
