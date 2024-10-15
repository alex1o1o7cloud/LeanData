import Mathlib

namespace NUMINAMATH_CALUDE_job_completion_time_l2500_250025

theorem job_completion_time (y : ℝ) 
  (h1 : (1 : ℝ) / (y + 8) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) : y = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2500_250025


namespace NUMINAMATH_CALUDE_money_exchange_equations_l2500_250070

/-- Represents the money exchange problem from "Nine Chapters on the Mathematical Art" --/
theorem money_exchange_equations (x y : ℝ) : 
  (x + 1/2 * y = 50 ∧ y + 2/3 * x = 50) ↔ 
  (∃ (a b : ℝ), 
    a = x ∧ 
    b = y ∧ 
    a + 1/2 * b = 50 ∧ 
    b + 2/3 * a = 50) :=
by sorry

end NUMINAMATH_CALUDE_money_exchange_equations_l2500_250070


namespace NUMINAMATH_CALUDE_range_of_a_l2500_250020

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y + 4 = 2*x*y → 
    x^2 + 2*x*y + y^2 - a*x - a*y + 1 ≥ 0) → 
  a ≤ 17/4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2500_250020


namespace NUMINAMATH_CALUDE_modular_inverse_89_mod_90_l2500_250062

theorem modular_inverse_89_mod_90 : ∃ x : ℕ, 0 ≤ x ∧ x < 90 ∧ (89 * x) % 90 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_89_mod_90_l2500_250062


namespace NUMINAMATH_CALUDE_k_range_l2500_250069

-- Define the hyperbola equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (k - 4) + y^2 / (k - 6) = 1

-- Define the ellipse equation
def ellipse (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / 5 + y^2 / k = 1

-- Define the condition for the line through M(2,1) intersecting the ellipse
def line_intersects_ellipse (k : ℝ) : Prop :=
  ∀ m b : ℝ, ∃ x y : ℝ, y = m * x + b ∧ ellipse k x y ∧ (2 * m + b = 1)

-- Main theorem
theorem k_range :
  (∀ k : ℝ, is_hyperbola k → line_intersects_ellipse k → k > 5 ∧ k < 6) ∧
  (∀ k : ℝ, k > 5 ∧ k < 6 → is_hyperbola k ∧ line_intersects_ellipse k) :=
sorry

end NUMINAMATH_CALUDE_k_range_l2500_250069


namespace NUMINAMATH_CALUDE_total_money_proof_l2500_250087

/-- The amount of money Beth currently has -/
def beth_money : ℕ := 70

/-- The amount of money Jan currently has -/
def jan_money : ℕ := 80

/-- The amount of money Tom currently has -/
def tom_money : ℕ := 210

theorem total_money_proof :
  (beth_money + 35 = 105) ∧
  (jan_money - 10 = beth_money) ∧
  (tom_money = 3 * (jan_money - 10)) →
  beth_money + jan_money + tom_money = 360 := by
sorry

end NUMINAMATH_CALUDE_total_money_proof_l2500_250087


namespace NUMINAMATH_CALUDE_profit_achieved_l2500_250084

/-- Calculates the number of pens needed to be sold to achieve a specific profit --/
def pens_to_sell (num_purchased : ℕ) (purchase_price : ℚ) (sell_price : ℚ) (desired_profit : ℚ) : ℕ :=
  let total_cost := num_purchased * purchase_price
  let revenue_needed := total_cost + desired_profit
  (revenue_needed / sell_price).ceil.toNat

/-- Theorem stating that selling 1500 pens achieves the desired profit --/
theorem profit_achieved (num_purchased : ℕ) (purchase_price sell_price desired_profit : ℚ) :
  num_purchased = 2000 →
  purchase_price = 15/100 →
  sell_price = 30/100 →
  desired_profit = 150 →
  pens_to_sell num_purchased purchase_price sell_price desired_profit = 1500 := by
  sorry

end NUMINAMATH_CALUDE_profit_achieved_l2500_250084


namespace NUMINAMATH_CALUDE_linear_function_composition_l2500_250064

/-- A linear function f: ℝ → ℝ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b ∧ a ≠ 0

theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x : ℝ, f (f x) = x - 2) → ∀ x : ℝ, f x = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_composition_l2500_250064


namespace NUMINAMATH_CALUDE_distance_proof_l2500_250098

theorem distance_proof (v1 v2 : ℝ) : 
  (5 * v1 + 5 * v2 = 30) →
  (3 * (v1 + 2) + 3 * (v2 + 2) = 30) →
  30 = 30 := by
  sorry

end NUMINAMATH_CALUDE_distance_proof_l2500_250098


namespace NUMINAMATH_CALUDE_base_five_of_156_l2500_250068

def base_five_equiv (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base_five_of_156 :
  base_five_equiv 156 = [1, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_five_of_156_l2500_250068


namespace NUMINAMATH_CALUDE_F_value_at_2_l2500_250031

/-- F is a polynomial function of degree 7 -/
def F (a b c d : ℝ) (x : ℝ) : ℝ := a*x^7 + b*x^5 + c*x^3 + d*x - 6

/-- Theorem: Given F(x) = ax^7 + bx^5 + cx^3 + dx - 6 and F(-2) = 10, prove that F(2) = -22 -/
theorem F_value_at_2 (a b c d : ℝ) (h : F a b c d (-2) = 10) : F a b c d 2 = -22 := by
  sorry

end NUMINAMATH_CALUDE_F_value_at_2_l2500_250031


namespace NUMINAMATH_CALUDE_unique_all_ones_polynomial_l2500_250039

def is_all_ones (n : ℕ) : Prop :=
  ∃ k : ℕ+, n = (10^k.val - 1) / 9

def polynomial_all_ones (P : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, is_all_ones n → is_all_ones (P n)

theorem unique_all_ones_polynomial :
  ∀ P : ℕ → ℕ, polynomial_all_ones P → P = id := by sorry

end NUMINAMATH_CALUDE_unique_all_ones_polynomial_l2500_250039


namespace NUMINAMATH_CALUDE_sum_of_mixed_numbers_l2500_250000

theorem sum_of_mixed_numbers : 
  (481 + 1/6 : ℚ) + (265 + 1/12 : ℚ) + (904 + 1/20 : ℚ) - 
  (184 + 29/30 : ℚ) - (160 + 41/42 : ℚ) - (703 + 55/56 : ℚ) = 
  603 + 3/8 := by sorry

end NUMINAMATH_CALUDE_sum_of_mixed_numbers_l2500_250000


namespace NUMINAMATH_CALUDE_max_quotient_four_digit_number_l2500_250055

def is_digit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

theorem max_quotient_four_digit_number (a b c d : ℕ) 
  (ha : is_digit a) (hb : is_digit b) (hc : is_digit c) (hd : is_digit d)
  (hdiff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  (1000 * a + 100 * b + 10 * c + d : ℚ) / (a + b + c + d) ≤ 329.2 := by
  sorry

end NUMINAMATH_CALUDE_max_quotient_four_digit_number_l2500_250055


namespace NUMINAMATH_CALUDE_ticket_cost_proof_l2500_250065

def initial_amount : ℕ := 760
def remaining_amount : ℕ := 310
def ticket_cost : ℕ := 300

theorem ticket_cost_proof :
  (initial_amount - remaining_amount = ticket_cost + ticket_cost / 2) →
  ticket_cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_proof_l2500_250065


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l2500_250042

theorem polar_to_rectangular (r θ : ℝ) :
  r = -3 ∧ θ = 5 * π / 6 →
  (r * Real.cos θ = 3 * Real.sqrt 3 / 2) ∧ (r * Real.sin θ = -3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l2500_250042


namespace NUMINAMATH_CALUDE_proposition_equivalence_l2500_250004

theorem proposition_equivalence (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l2500_250004


namespace NUMINAMATH_CALUDE_function_value_at_one_l2500_250049

theorem function_value_at_one (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x - x^2| ≤ 1/4)
  (h2 : ∀ x, |f x + 1 - x^2| ≤ 3/4) : 
  f 1 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_one_l2500_250049


namespace NUMINAMATH_CALUDE_smallest_sum_with_conditions_l2500_250095

theorem smallest_sum_with_conditions (a b : ℕ+) 
  (h1 : Nat.gcd (a + b) 330 = 1)
  (h2 : (a : ℕ)^(a : ℕ) % (b : ℕ)^(b : ℕ) = 0)
  (h3 : ¬(∃k : ℕ, b = k * a)) :
  (∀ c d : ℕ+, 
    Nat.gcd (c + d) 330 = 1 → 
    (c : ℕ)^(c : ℕ) % (d : ℕ)^(d : ℕ) = 0 → 
    ¬(∃k : ℕ, d = k * c) → 
    a + b ≤ c + d) ∧ 
  a + b = 147 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_with_conditions_l2500_250095


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2500_250099

theorem sum_of_solutions (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 7 * x₁ - 9 = 0) → 
  (2 * x₂^2 - 7 * x₂ - 9 = 0) → 
  (x₁ ≠ x₂) →
  (x₁ + x₂ = 7/2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2500_250099


namespace NUMINAMATH_CALUDE_johns_piggy_bank_l2500_250030

theorem johns_piggy_bank (quarters dimes nickels : ℕ) : 
  quarters + dimes + nickels = 63 →
  dimes = quarters + 3 →
  nickels = quarters - 6 →
  quarters = 22 := by
sorry

end NUMINAMATH_CALUDE_johns_piggy_bank_l2500_250030


namespace NUMINAMATH_CALUDE_paint_stones_l2500_250081

def canPaintAllBlack (k : Nat) : Prop :=
  1 ≤ k ∧ k ≤ 50 ∧ Nat.gcd 100 (k - 1) = 1

theorem paint_stones (k : Nat) :
  canPaintAllBlack k ↔ ¬∃m : Nat, m ∈ Finset.range 13 ∧ k = 4 * m + 1 :=
by sorry

end NUMINAMATH_CALUDE_paint_stones_l2500_250081


namespace NUMINAMATH_CALUDE_shaded_to_white_ratio_is_five_thirds_l2500_250016

/-- A nested square figure where vertices of inner squares are at the midpoints of the sides of the outer squares. -/
structure NestedSquareFigure where
  /-- The number of nested squares in the figure -/
  num_squares : ℕ
  /-- The side length of the outermost square -/
  outer_side_length : ℝ
  /-- Assumption that the figure is constructed with vertices at midpoints -/
  vertices_at_midpoints : Bool

/-- The ratio of the shaded area to the white area in the nested square figure -/
def shaded_to_white_ratio (figure : NestedSquareFigure) : ℚ :=
  5 / 3

/-- Theorem stating that the ratio of shaded to white area is 5/3 -/
theorem shaded_to_white_ratio_is_five_thirds (figure : NestedSquareFigure) 
  (h : figure.vertices_at_midpoints = true) : 
  shaded_to_white_ratio figure = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_white_ratio_is_five_thirds_l2500_250016


namespace NUMINAMATH_CALUDE_complementary_of_same_angle_are_equal_l2500_250059

/-- Two angles are complementary if their sum is equal to a right angle (90°) -/
def Complementary (α β : Real) : Prop := α + β = Real.pi / 2

/-- An angle is complementary to itself if it is half of a right angle -/
def SelfComplementary (α : Real) : Prop := α = Real.pi / 4

theorem complementary_of_same_angle_are_equal (α : Real) (h : SelfComplementary α) :
  ∃ β, Complementary α β ∧ α = β := by
  sorry

end NUMINAMATH_CALUDE_complementary_of_same_angle_are_equal_l2500_250059


namespace NUMINAMATH_CALUDE_volunteer_assignment_l2500_250047

def number_of_volunteers : ℕ := 6
def number_for_training : ℕ := 4
def number_per_location : ℕ := 2

def select_and_assign (n m k : ℕ) : Prop :=
  ∃ (total : ℕ),
    total = Nat.choose (n - 1) k * Nat.choose (n - k - 1) k +
            Nat.choose (n - 1) 1 * Nat.choose (n - 2) k ∧
    total = 60

theorem volunteer_assignment :
  select_and_assign number_of_volunteers number_for_training number_per_location :=
sorry

end NUMINAMATH_CALUDE_volunteer_assignment_l2500_250047


namespace NUMINAMATH_CALUDE_sum_a_d_equals_two_l2500_250050

theorem sum_a_d_equals_two (a b c d : ℝ) 
  (h1 : a + b = 4)
  (h2 : b + c = 7)
  (h3 : c + d = 5) :
  a + d = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_two_l2500_250050


namespace NUMINAMATH_CALUDE_sequence_max_value_l2500_250082

def x (n : ℕ) : ℚ := (n - 1 : ℚ) / ((n : ℚ)^2 + 1)

theorem sequence_max_value :
  (∀ n : ℕ, x n ≤ (1 : ℚ) / 5) ∧
  x 2 = (1 : ℚ) / 5 ∧
  x 3 = (1 : ℚ) / 5 :=
sorry

end NUMINAMATH_CALUDE_sequence_max_value_l2500_250082


namespace NUMINAMATH_CALUDE_sum_three_numbers_l2500_250038

theorem sum_three_numbers (a b c N : ℝ) 
  (sum_eq : a + b + c = 105)
  (a_eq : a - 5 = N)
  (b_eq : b + 10 = N)
  (c_eq : 5 * c = N) : 
  N = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l2500_250038


namespace NUMINAMATH_CALUDE_max_area_rectangle_perimeter_40_l2500_250008

/-- The maximum area of a rectangle with perimeter 40 is 100 -/
theorem max_area_rectangle_perimeter_40 :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * x + 2 * y = 40 →
  x * y ≤ 100 := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_perimeter_40_l2500_250008


namespace NUMINAMATH_CALUDE_hamburger_combinations_l2500_250023

/-- The number of available condiments -/
def num_condiments : ℕ := 9

/-- The number of patty options -/
def num_patty_options : ℕ := 4

/-- The number of possible combinations for condiments -/
def condiment_combinations : ℕ := 2^num_condiments

/-- The total number of different hamburger combinations -/
def total_combinations : ℕ := num_patty_options * condiment_combinations

/-- Theorem stating that the total number of different hamburger combinations is 2048 -/
theorem hamburger_combinations : total_combinations = 2048 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l2500_250023


namespace NUMINAMATH_CALUDE_number_of_friends_l2500_250066

/-- Given that Sam, Dan, Tom, and Keith each have 14 Pokemon cards, 
    prove that the number of friends is 4. -/
theorem number_of_friends : ℕ := by
  sorry

end NUMINAMATH_CALUDE_number_of_friends_l2500_250066


namespace NUMINAMATH_CALUDE_dog_bunny_ratio_l2500_250046

/-- Given a total of 375 dogs and bunnies, with 75 dogs, prove that the ratio of dogs to bunnies is 1:4 -/
theorem dog_bunny_ratio (total : ℕ) (dogs : ℕ) (h1 : total = 375) (h2 : dogs = 75) :
  (dogs : ℚ) / (total - dogs : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_dog_bunny_ratio_l2500_250046


namespace NUMINAMATH_CALUDE_intersection_distance_l2500_250045

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_distance (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l2500_250045


namespace NUMINAMATH_CALUDE_tangent_product_special_angles_l2500_250051

theorem tangent_product_special_angles :
  let A : Real := 30 * π / 180
  let B : Real := 60 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = 2 + 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_special_angles_l2500_250051


namespace NUMINAMATH_CALUDE_union_P_S_when_m_2_S_subset_P_iff_m_in_zero_one_l2500_250077

-- Define the sets P and S
def P : Set ℝ := {x | 4 / (x + 2) ≥ 1}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Part 1: P ∪ S when m = 2
theorem union_P_S_when_m_2 : 
  P ∪ S 2 = {x | -2 < x ∧ x ≤ 3} := by sorry

-- Part 2: S ⊆ P iff m ∈ [0, 1]
theorem S_subset_P_iff_m_in_zero_one (m : ℝ) : 
  S m ⊆ P ↔ 0 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_union_P_S_when_m_2_S_subset_P_iff_m_in_zero_one_l2500_250077


namespace NUMINAMATH_CALUDE_f_properties_l2500_250034

/-- Represents a natural number in base 3 notation -/
structure Base3 where
  digits : List Nat
  first_nonzero : digits.head? ≠ some 0
  all_less_than_3 : ∀ d ∈ digits, d < 3

/-- Converts a natural number to its Base3 representation -/
noncomputable def toBase3 (n : ℕ) : Base3 := sorry

/-- Converts a Base3 representation back to a natural number -/
noncomputable def fromBase3 (b : Base3) : ℕ := sorry

/-- The function f as described in the problem -/
noncomputable def f (n : ℕ) : ℕ :=
  let b := toBase3 n
  match b.digits with
  | 1 :: rest => fromBase3 ⟨2 :: rest, sorry, sorry⟩
  | 2 :: rest => fromBase3 ⟨1 :: (rest ++ [0]), sorry, sorry⟩
  | _ => n  -- This case should not occur for valid Base3 numbers

/-- The main theorem to be proved -/
theorem f_properties :
  (∀ m n, m < n → f m < f n) ∧  -- Strictly monotone
  (∀ n, f (f n) = 3 * n) := by
  sorry


end NUMINAMATH_CALUDE_f_properties_l2500_250034


namespace NUMINAMATH_CALUDE_calculate_principal_amount_l2500_250060

/-- Given simple interest, time period, and interest rate, calculate the principal amount -/
theorem calculate_principal_amount (simple_interest rate : ℚ) (time : ℕ) : 
  simple_interest = 200 → 
  time = 4 → 
  rate = 3125 / 100000 → 
  simple_interest = (1600 : ℚ) * rate * (time : ℚ) := by
  sorry

#check calculate_principal_amount

end NUMINAMATH_CALUDE_calculate_principal_amount_l2500_250060


namespace NUMINAMATH_CALUDE_expression_simplification_l2500_250018

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3 * x^2 + 2 * x) / ((x - 1) * (x + 3)) - (5 * x + 3) / ((x - 1) * (x + 3)) =
  3 * (x^2 - x - 1) / ((x - 1) * (x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2500_250018


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2500_250035

theorem trigonometric_simplification (x : ℝ) :
  (2 + 3 * Real.sin x - 4 * Real.cos x) / (2 + 3 * Real.sin x + 4 * Real.cos x) = Real.tan (x / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2500_250035


namespace NUMINAMATH_CALUDE_equation_solution_l2500_250021

theorem equation_solution (p m z : ℤ) : 
  Prime p ∧ m > 0 ∧ z < 0 ∧ p^3 + p*m + 2*z*m = m^2 + p*z + z^2 ↔ 
  (p = 2 ∧ m = 4 + z ∧ (z = -1 ∨ z = -2 ∨ z = -3)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2500_250021


namespace NUMINAMATH_CALUDE_trapezoid_area_sum_l2500_250006

/-- Represents a trapezoid with four side lengths -/
structure Trapezoid :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

/-- Calculates the area of a trapezoid using Heron's formula -/
def area (t : Trapezoid) : ℝ := sorry

/-- Checks if a number is a square-free positive integer -/
def isSquareFree (n : ℕ) : Prop := sorry

/-- Theorem: The sum of areas of all possible trapezoids with sides 4, 6, 8, and 10
    can be expressed as r₁√n₁ + r₂√n₂ + r₃, where r₁, r₂, r₃ are rational,
    n₁, n₂ are distinct square-free positive integers, and r₁ + r₂ + r₃ + n₁ + n₂ = 80 -/
theorem trapezoid_area_sum :
  ∃ (r₁ r₂ r₃ : ℚ) (n₁ n₂ : ℕ),
    let t₁ : Trapezoid := ⟨4, 6, 8, 10⟩
    let t₂ : Trapezoid := ⟨6, 10, 4, 8⟩
    isSquareFree n₁ ∧ isSquareFree n₂ ∧ n₁ ≠ n₂ ∧
    area t₁ + area t₂ = r₁ * Real.sqrt n₁ + r₂ * Real.sqrt n₂ + r₃ ∧
    r₁ + r₂ + r₃ + n₁ + n₂ = 80 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_sum_l2500_250006


namespace NUMINAMATH_CALUDE_initial_gathering_size_l2500_250014

theorem initial_gathering_size (initial_snackers : ℕ)
  (h1 : initial_snackers = 100)
  (h2 : ∃ (a b c d e : ℕ),
    a = initial_snackers + 20 ∧
    b = a / 2 + 10 ∧
    c = b - 30 ∧
    d = c / 2 ∧
    d = 20) :
  initial_snackers = 100 := by
sorry

end NUMINAMATH_CALUDE_initial_gathering_size_l2500_250014


namespace NUMINAMATH_CALUDE_range_of_a_l2500_250097

-- Define propositions p and q
def p (x : ℝ) : Prop := (4*x - 3)^2 ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the set A
def A : Set ℝ := {x | p x}

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | q x a}

-- Define the condition that ¬p is necessary but not sufficient for ¬q
def condition (a : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a))

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, condition a ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2500_250097


namespace NUMINAMATH_CALUDE_alexanders_apples_l2500_250027

/-- Prove that Alexander bought 5 apples given the conditions of his shopping trip -/
theorem alexanders_apples : 
  ∀ (apple_price orange_price total_spent num_oranges : ℕ),
    apple_price = 1 →
    orange_price = 2 →
    num_oranges = 2 →
    total_spent = 9 →
    ∃ (num_apples : ℕ), 
      num_apples * apple_price + num_oranges * orange_price = total_spent ∧
      num_apples = 5 := by
  sorry

end NUMINAMATH_CALUDE_alexanders_apples_l2500_250027


namespace NUMINAMATH_CALUDE_power_function_m_value_l2500_250093

/-- A function y = (m^2 + 2m - 2)x^m is a power function and increasing in the first quadrant -/
def is_power_and_increasing (m : ℝ) : Prop :=
  (m^2 + 2*m - 2 = 1) ∧ (m > 0)

/-- If y = (m^2 + 2m - 2)x^m is a power function and increasing in the first quadrant, then m = 1 -/
theorem power_function_m_value :
  ∀ m : ℝ, is_power_and_increasing m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_m_value_l2500_250093


namespace NUMINAMATH_CALUDE_min_value_inequality_l2500_250096

theorem min_value_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a + 2*b = 1) (h2 : c + 2*d = 1) : 
  1/a + 1/(b*c*d) > 25 := by
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2500_250096


namespace NUMINAMATH_CALUDE_a_minus_b_value_l2500_250002

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 8) (h2 : |b| = 5) (h3 : a + b > 0) :
  a - b = 3 ∨ a - b = 13 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l2500_250002


namespace NUMINAMATH_CALUDE_negation_of_exists_negation_of_proposition_l2500_250075

theorem negation_of_exists (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x > 0, 2 * x + 3 ≤ 0) ↔ (∀ x > 0, 2 * x + 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_negation_of_proposition_l2500_250075


namespace NUMINAMATH_CALUDE_inequality_proof_l2500_250089

theorem inequality_proof (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) :
  a + b < b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2500_250089


namespace NUMINAMATH_CALUDE_bouncy_balls_cost_l2500_250054

def red_packs : ℕ := 5
def yellow_packs : ℕ := 4
def blue_packs : ℕ := 3

def red_balls_per_pack : ℕ := 18
def yellow_balls_per_pack : ℕ := 15
def blue_balls_per_pack : ℕ := 12

def red_price : ℚ := 3/2
def yellow_price : ℚ := 5/4
def blue_price : ℚ := 1

def red_discount : ℚ := 1/10
def blue_discount : ℚ := 1/20

def total_cost (packs : ℕ) (balls_per_pack : ℕ) (price : ℚ) : ℚ :=
  (packs * balls_per_pack : ℚ) * price

def discounted_cost (cost : ℚ) (discount : ℚ) : ℚ :=
  cost * (1 - discount)

theorem bouncy_balls_cost :
  discounted_cost (total_cost red_packs red_balls_per_pack red_price) red_discount = 243/2 ∧
  total_cost yellow_packs yellow_balls_per_pack yellow_price = 75 ∧
  discounted_cost (total_cost blue_packs blue_balls_per_pack blue_price) blue_discount = 342/10 :=
by sorry

end NUMINAMATH_CALUDE_bouncy_balls_cost_l2500_250054


namespace NUMINAMATH_CALUDE_unique_b_value_l2500_250026

theorem unique_b_value : ∃! b : ℝ, ∃ x : ℝ, x^2 + b*x + 1 = 0 ∧ x^2 + x + b = 0 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_b_value_l2500_250026


namespace NUMINAMATH_CALUDE_no_141_cents_combination_l2500_250033

/-- Represents the different types of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | HalfDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.HalfDollar => 50

/-- Represents a selection of three coins --/
structure CoinSelection :=
  (coin1 : Coin)
  (coin2 : Coin)
  (coin3 : Coin)

/-- Calculates the total value of a coin selection in cents --/
def totalValue (selection : CoinSelection) : Nat :=
  coinValue selection.coin1 + coinValue selection.coin2 + coinValue selection.coin3

/-- Theorem stating that no combination of three coins can sum to 141 cents --/
theorem no_141_cents_combination :
  ∀ (selection : CoinSelection), totalValue selection ≠ 141 := by
  sorry

end NUMINAMATH_CALUDE_no_141_cents_combination_l2500_250033


namespace NUMINAMATH_CALUDE_potential_parallel_necessary_not_sufficient_l2500_250056

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate for parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The condition for potential parallelism -/
def potential_parallel_condition (l1 l2 : Line) : Prop :=
  l1.a * l2.b - l2.a * l1.b = 0

/-- Theorem stating that the condition is necessary but not sufficient for parallelism -/
theorem potential_parallel_necessary_not_sufficient :
  (∀ l1 l2 : Line, parallel l1 l2 → potential_parallel_condition l1 l2) ∧
  ¬(∀ l1 l2 : Line, potential_parallel_condition l1 l2 → parallel l1 l2) :=
sorry

end NUMINAMATH_CALUDE_potential_parallel_necessary_not_sufficient_l2500_250056


namespace NUMINAMATH_CALUDE_square_sum_halving_l2500_250094

theorem square_sum_halving (a b : ℕ) (h : a^2 + b^2 = 18728) :
  ∃ (n m : ℕ), n^2 + m^2 = 9364 ∧ ((n = 30 ∧ m = 92) ∨ (n = 92 ∧ m = 30)) :=
by
  sorry

end NUMINAMATH_CALUDE_square_sum_halving_l2500_250094


namespace NUMINAMATH_CALUDE_tiles_required_for_room_l2500_250041

theorem tiles_required_for_room (room_length room_width tile_length tile_width : ℚ) :
  room_length = 10 →
  room_width = 15 →
  tile_length = 5 / 12 →
  tile_width = 2 / 3 →
  (room_length * room_width) / (tile_length * tile_width) = 540 :=
by
  sorry

end NUMINAMATH_CALUDE_tiles_required_for_room_l2500_250041


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_l2500_250083

theorem quadratic_root_sum_product (x₁ x₂ : ℝ) : 
  x₁^2 + 3*x₁ - 2023 = 0 →
  x₂^2 + 3*x₂ - 2023 = 0 →
  x₁^2 * x₂ + x₁ * x₂^2 = 6069 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_l2500_250083


namespace NUMINAMATH_CALUDE_math_class_size_l2500_250090

theorem math_class_size (total : ℕ) (both : ℕ) :
  total = 75 →
  both = 10 →
  ∃ (math physics : ℕ),
    total = math + physics - both ∧
    math = 2 * physics →
    math = 56 := by
  sorry

end NUMINAMATH_CALUDE_math_class_size_l2500_250090


namespace NUMINAMATH_CALUDE_cos_210_degrees_l2500_250028

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l2500_250028


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2500_250058

/-- Given a hyperbola with equation x²/a² - y²/4 = 1 (a > 0), 
    if a right triangle is formed by its left and right foci and the point (2, 1),
    then the length of its real axis is 2. -/
theorem hyperbola_real_axis_length (a : ℝ) (h1 : a > 0) : 
  let f (x y : ℝ) := x^2 / a^2 - y^2 / 4
  ∃ (c : ℝ), (2 - c) * (2 + c) + 1 * 1 = 0 ∧ 2 * a = 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2500_250058


namespace NUMINAMATH_CALUDE_sqrt5_diamond_sqrt5_equals_20_l2500_250022

-- Define the custom operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt5_diamond_sqrt5_equals_20 : diamond (Real.sqrt 5) (Real.sqrt 5) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sqrt5_diamond_sqrt5_equals_20_l2500_250022


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2500_250044

theorem cube_root_equation_solution (x : ℝ) :
  (7 * x * (x^2)^(1/2))^(1/3) = 5 → x = 5 * (35^(1/2)) / 7 ∨ x = -5 * (35^(1/2)) / 7 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2500_250044


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2500_250071

theorem quadratic_factorization (a x y : ℝ) : a * x^2 + 2*a*x*y + a * y^2 = a * (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2500_250071


namespace NUMINAMATH_CALUDE_total_amount_correct_l2500_250078

/-- The rate for painting fences in dollars per meter -/
def painting_rate : ℚ := 0.20

/-- The number of fences to be painted -/
def number_of_fences : ℕ := 50

/-- The length of each fence in meters -/
def fence_length : ℕ := 500

/-- The total amount earned from painting all fences -/
def total_amount : ℚ := 5000

/-- Theorem stating that the total amount earned is correct given the conditions -/
theorem total_amount_correct : 
  painting_rate * (number_of_fences * fence_length : ℚ) = total_amount := by
  sorry

end NUMINAMATH_CALUDE_total_amount_correct_l2500_250078


namespace NUMINAMATH_CALUDE_rounding_and_scientific_notation_l2500_250088

-- Define rounding to significant figures
def roundToSignificantFigures (x : ℝ) (n : ℕ) : ℝ := sorry

-- Define rounding to decimal places
def roundToDecimalPlaces (x : ℝ) (n : ℕ) : ℝ := sorry

-- Define scientific notation
def scientificNotation (x : ℝ) : ℝ × ℤ := sorry

theorem rounding_and_scientific_notation :
  (roundToSignificantFigures 12.349 2 = 12) ∧
  (roundToDecimalPlaces 0.12349 3 = 0.123) ∧
  (scientificNotation 201200 = (2.012, 5)) ∧
  (scientificNotation 0.0002012 = (2.012, -4)) := by sorry

end NUMINAMATH_CALUDE_rounding_and_scientific_notation_l2500_250088


namespace NUMINAMATH_CALUDE_union_C_R_A_B_eq_expected_result_l2500_250057

-- Define the sets A and B
def A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }
def B : Set ℝ := { x | 1 < x ∧ x ≤ 3 }

-- Define the complement of A with respect to ℝ
def C_R_A : Set ℝ := { x | x ∉ A }

-- Define the union of C_R A and B
def union_C_R_A_B : Set ℝ := C_R_A ∪ B

-- Define the expected result
def expected_result : Set ℝ := { x | x < -1 ∨ 1 < x }

-- Theorem statement
theorem union_C_R_A_B_eq_expected_result : union_C_R_A_B = expected_result := by
  sorry

end NUMINAMATH_CALUDE_union_C_R_A_B_eq_expected_result_l2500_250057


namespace NUMINAMATH_CALUDE_junior_score_l2500_250013

theorem junior_score (n : ℝ) (h_n_pos : n > 0) : 
  let junior_percent : ℝ := 0.2
  let senior_percent : ℝ := 0.8
  let overall_avg : ℝ := 86
  let senior_avg : ℝ := 85
  let junior_count : ℝ := junior_percent * n
  let senior_count : ℝ := senior_percent * n
  let total_score : ℝ := overall_avg * n
  let senior_total_score : ℝ := senior_avg * senior_count
  let junior_total_score : ℝ := total_score - senior_total_score
  junior_total_score / junior_count = 90 :=
by sorry

end NUMINAMATH_CALUDE_junior_score_l2500_250013


namespace NUMINAMATH_CALUDE_two_axes_implies_center_symmetry_l2500_250076

/-- A geometric figure in a 2D plane. -/
structure Figure where
  -- The implementation details of the figure are abstracted away

/-- An axis of symmetry for a figure. -/
structure AxisOfSymmetry where
  -- The implementation details of the axis are abstracted away

/-- A center of symmetry for a figure. -/
structure CenterOfSymmetry where
  -- The implementation details of the center are abstracted away

/-- Predicate to check if a figure has an axis of symmetry. -/
def hasAxisOfSymmetry (f : Figure) (a : AxisOfSymmetry) : Prop :=
  sorry

/-- Predicate to check if a figure has a center of symmetry. -/
def hasCenterOfSymmetry (f : Figure) (c : CenterOfSymmetry) : Prop :=
  sorry

/-- Theorem: If a figure has exactly two axes of symmetry, it must have a center of symmetry. -/
theorem two_axes_implies_center_symmetry (f : Figure) (a1 a2 : AxisOfSymmetry) :
  (hasAxisOfSymmetry f a1) ∧ 
  (hasAxisOfSymmetry f a2) ∧ 
  (a1 ≠ a2) ∧
  (∀ a : AxisOfSymmetry, hasAxisOfSymmetry f a → (a = a1 ∨ a = a2)) →
  ∃ c : CenterOfSymmetry, hasCenterOfSymmetry f c :=
sorry

end NUMINAMATH_CALUDE_two_axes_implies_center_symmetry_l2500_250076


namespace NUMINAMATH_CALUDE_average_and_difference_l2500_250043

theorem average_and_difference (x : ℝ) : 
  (30 + x) / 2 = 34 → |x - 30| = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l2500_250043


namespace NUMINAMATH_CALUDE_aquarium_visitors_not_ill_l2500_250015

theorem aquarium_visitors_not_ill (total_visitors : ℕ) (ill_percentage : ℚ) : 
  total_visitors = 500 → 
  ill_percentage = 40 / 100 → 
  total_visitors - (total_visitors * ill_percentage).floor = 300 := by
sorry

end NUMINAMATH_CALUDE_aquarium_visitors_not_ill_l2500_250015


namespace NUMINAMATH_CALUDE_work_efficiency_l2500_250010

/-- Given a person who takes x days to complete a task, and Tanya who is 25% more efficient
    and takes 12 days to complete the same task, prove that x is equal to 15 days. -/
theorem work_efficiency (x : ℝ) : 
  (∃ (person : ℝ → ℝ) (tanya : ℝ → ℝ), 
    (∀ t, tanya t = 0.75 * person t) ∧ 
    (tanya 12 = person x)) → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_work_efficiency_l2500_250010


namespace NUMINAMATH_CALUDE_age_difference_l2500_250024

theorem age_difference (matt_age john_age : ℕ) 
  (h1 : matt_age + john_age = 52)
  (h2 : ∃ k : ℕ, matt_age + k = 4 * john_age) : 
  4 * john_age - matt_age = 3 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2500_250024


namespace NUMINAMATH_CALUDE_original_price_calculation_l2500_250072

/-- Given an item sold at a 20% loss with a selling price of 480, prove that the original price was 600. -/
theorem original_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) : 
  selling_price = 480 → 
  loss_percentage = 20 → 
  ∃ original_price : ℝ, 
    selling_price = original_price * (1 - loss_percentage / 100) ∧ 
    original_price = 600 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2500_250072


namespace NUMINAMATH_CALUDE_find_set_C_l2500_250007

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

theorem find_set_C : 
  ∃ C : Set ℝ, 
    (C = {0, 1, 2}) ∧ 
    (∀ a : ℝ, a ∈ C ↔ A ∪ B a = A) :=
by sorry

end NUMINAMATH_CALUDE_find_set_C_l2500_250007


namespace NUMINAMATH_CALUDE_number_equation_solution_l2500_250061

theorem number_equation_solution : ∃ n : ℝ, 7 * n = 3 * n + 12 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2500_250061


namespace NUMINAMATH_CALUDE_gunther_free_time_l2500_250048

/-- Represents the time in minutes for each cleaning task and the total free time --/
structure CleaningTime where
  vacuuming : ℕ
  dusting : ℕ
  mopping : ℕ
  brushing_per_cat : ℕ
  num_cats : ℕ
  free_time : ℕ

/-- Calculates the remaining free time after cleaning --/
def remaining_free_time (ct : CleaningTime) : ℕ :=
  ct.free_time - (ct.vacuuming + ct.dusting + ct.mopping + ct.brushing_per_cat * ct.num_cats)

/-- Theorem stating that Gunther will have 30 minutes of free time left --/
theorem gunther_free_time :
  let ct : CleaningTime := {
    vacuuming := 45,
    dusting := 60,
    mopping := 30,
    brushing_per_cat := 5,
    num_cats := 3,
    free_time := 180
  }
  remaining_free_time ct = 30 := by
  sorry


end NUMINAMATH_CALUDE_gunther_free_time_l2500_250048


namespace NUMINAMATH_CALUDE_potato_price_is_one_l2500_250011

def initial_money : ℚ := 60
def celery_price : ℚ := 5
def cereal_price : ℚ := 12
def cereal_discount : ℚ := 0.5
def bread_price : ℚ := 8
def milk_price : ℚ := 10
def milk_discount : ℚ := 0.1
def num_potatoes : ℕ := 6
def money_left : ℚ := 26

def discounted_price (price : ℚ) (discount : ℚ) : ℚ :=
  price * (1 - discount)

theorem potato_price_is_one :
  let celery_cost := celery_price
  let cereal_cost := discounted_price cereal_price cereal_discount
  let bread_cost := bread_price
  let milk_cost := discounted_price milk_price milk_discount
  let total_cost := celery_cost + cereal_cost + bread_cost + milk_cost
  let potato_coffee_cost := initial_money - money_left
  let potato_cost := potato_coffee_cost - total_cost
  potato_cost / num_potatoes = 1 := by sorry

end NUMINAMATH_CALUDE_potato_price_is_one_l2500_250011


namespace NUMINAMATH_CALUDE_p_and_q_implies_p_or_q_l2500_250040

theorem p_and_q_implies_p_or_q (p q : Prop) : (p ∧ q) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_implies_p_or_q_l2500_250040


namespace NUMINAMATH_CALUDE_equality_for_all_n_l2500_250079

theorem equality_for_all_n (x y a b : ℝ) 
  (h1 : x + y = a + b) 
  (h2 : x^2 + y^2 = a^2 + b^2) : 
  ∀ n : ℤ, x^n + y^n = a^n + b^n := by sorry

end NUMINAMATH_CALUDE_equality_for_all_n_l2500_250079


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l2500_250085

/-- The cost per pound of the first candy -/
def first_candy_cost : ℝ := 8

/-- The weight of the first candy in pounds -/
def first_candy_weight : ℝ := 30

/-- The cost per pound of the second candy -/
def second_candy_cost : ℝ := 5

/-- The weight of the second candy in pounds -/
def second_candy_weight : ℝ := 60

/-- The cost per pound of the mixture -/
def mixture_cost : ℝ := 6

/-- The total weight of the mixture in pounds -/
def total_weight : ℝ := first_candy_weight + second_candy_weight

theorem candy_mixture_cost :
  first_candy_cost * first_candy_weight + second_candy_cost * second_candy_weight =
  mixture_cost * total_weight :=
by sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l2500_250085


namespace NUMINAMATH_CALUDE_train_distance_in_three_hours_l2500_250086

-- Define the train's speed
def train_speed : ℚ := 1 / 2

-- Define the duration in hours
def duration : ℚ := 3

-- Define the number of minutes in an hour
def minutes_per_hour : ℚ := 60

-- Theorem statement
theorem train_distance_in_three_hours :
  train_speed * minutes_per_hour * duration = 90 := by
  sorry


end NUMINAMATH_CALUDE_train_distance_in_three_hours_l2500_250086


namespace NUMINAMATH_CALUDE_sum_of_tens_for_hundred_to_ten_l2500_250012

theorem sum_of_tens_for_hundred_to_ten (n : ℕ) : (100 ^ 10) = n * 10 → n = 10 ^ 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tens_for_hundred_to_ten_l2500_250012


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l2500_250053

theorem simplify_nested_roots (a : ℝ) (ha : a > 0) :
  (((a^16)^(1/8))^(1/4))^3 * (((a^16)^(1/4))^(1/8))^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l2500_250053


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2500_250003

theorem quadratic_inequality (x : ℝ) : -3 * x^2 - 9 * x - 6 ≥ -12 ↔ -2 ≤ x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2500_250003


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_two_l2500_250063

theorem cube_root_sum_equals_two :
  ∃ n : ℤ, (Real.rpow (2 + 10/9 * Real.sqrt 3) (1/3 : ℝ) + Real.rpow (2 - 10/9 * Real.sqrt 3) (1/3 : ℝ) = n) →
  Real.rpow (2 + 10/9 * Real.sqrt 3) (1/3 : ℝ) + Real.rpow (2 - 10/9 * Real.sqrt 3) (1/3 : ℝ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_two_l2500_250063


namespace NUMINAMATH_CALUDE_x_equals_ten_l2500_250032

/-- A structure representing the number pyramid --/
structure NumberPyramid where
  row1_left : ℕ
  row1_right : ℕ
  row2_left : ℕ
  row2_middle : ℕ → ℕ
  row2_right : ℕ → ℕ
  row3_left : ℕ → ℕ
  row3_right : ℕ → ℕ
  row4 : ℕ → ℕ

/-- The theorem stating that x must be 10 given the conditions --/
theorem x_equals_ten (pyramid : NumberPyramid) 
  (h1 : pyramid.row1_left = 11)
  (h2 : pyramid.row1_right = 49)
  (h3 : pyramid.row2_left = 11)
  (h4 : ∀ x, pyramid.row2_middle x = 6 + x)
  (h5 : ∀ x, pyramid.row2_right x = x + 7)
  (h6 : ∀ x, pyramid.row3_left x = pyramid.row2_left + pyramid.row2_middle x)
  (h7 : ∀ x, pyramid.row3_right x = pyramid.row2_middle x + pyramid.row2_right x)
  (h8 : ∀ x, pyramid.row4 x = pyramid.row3_left x + pyramid.row3_right x)
  (h9 : pyramid.row4 10 = 60) :
  ∃ x, x = 10 ∧ pyramid.row4 x = 60 :=
sorry

end NUMINAMATH_CALUDE_x_equals_ten_l2500_250032


namespace NUMINAMATH_CALUDE_fraction_of_cats_l2500_250073

theorem fraction_of_cats (total_animals : ℕ) (total_dog_legs : ℕ) : 
  total_animals = 300 →
  total_dog_legs = 400 →
  (2 : ℚ) / 3 = (total_animals - (total_dog_legs / 4)) / total_animals :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_cats_l2500_250073


namespace NUMINAMATH_CALUDE_total_movies_is_nineteen_l2500_250009

/-- The number of movies shown on each screen in a movie theater --/
def movies_per_screen : List Nat := [3, 4, 2, 3, 5, 2]

/-- The total number of movies shown in the theater --/
def total_movies : Nat := movies_per_screen.sum

/-- Theorem stating that the total number of movies shown is 19 --/
theorem total_movies_is_nineteen : total_movies = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_movies_is_nineteen_l2500_250009


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_pi_eighth_l2500_250029

theorem cos_squared_minus_sin_squared_pi_eighth :
  Real.cos (π / 8) ^ 2 - Real.sin (π / 8) ^ 2 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_pi_eighth_l2500_250029


namespace NUMINAMATH_CALUDE_convergence_bound_l2500_250067

def v : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => (3/2) * v n - (3/2) * (v n)^2

def M : ℚ := 1/2

theorem convergence_bound (k : ℕ) : k = 5 ↔ (∀ j < k, |v j - M| > 1/2^20) ∧ |v k - M| ≤ 1/2^20 := by
  sorry

end NUMINAMATH_CALUDE_convergence_bound_l2500_250067


namespace NUMINAMATH_CALUDE_jeans_savings_theorem_l2500_250091

/-- Calculates the amount saved on a pair of jeans given the original price and discounts -/
def calculate_savings (original_price : ℝ) (sale_discount_percent : ℝ) (coupon_discount : ℝ) (credit_card_discount_percent : ℝ) : ℝ :=
  let price_after_sale := original_price * (1 - sale_discount_percent)
  let price_after_coupon := price_after_sale - coupon_discount
  let final_price := price_after_coupon * (1 - credit_card_discount_percent)
  original_price - final_price

/-- Theorem stating that the savings on the jeans is $44 -/
theorem jeans_savings_theorem :
  calculate_savings 125 0.20 10 0.10 = 44 := by
  sorry

#eval calculate_savings 125 0.20 10 0.10

end NUMINAMATH_CALUDE_jeans_savings_theorem_l2500_250091


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l2500_250017

/-- Represents a square quilt block -/
structure QuiltBlock where
  totalSquares : ℕ
  dividedSquares : ℕ
  shadePerDividedSquare : ℚ

/-- The fraction of the quilt block that is shaded -/
def shadedFraction (q : QuiltBlock) : ℚ :=
  (q.dividedSquares : ℚ) * q.shadePerDividedSquare / q.totalSquares

/-- Theorem stating that for a quilt block with 16 total squares, 
    4 divided squares, and half of each divided square shaded,
    the shaded fraction is 1/8 -/
theorem quilt_shaded_fraction :
  ∀ (q : QuiltBlock), 
    q.totalSquares = 16 → 
    q.dividedSquares = 4 → 
    q.shadePerDividedSquare = 1/2 →
    shadedFraction q = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l2500_250017


namespace NUMINAMATH_CALUDE_simone_finish_time_l2500_250036

-- Define the start time
def start_time : Nat := 8 * 60  -- 8:00 AM in minutes since midnight

-- Define the duration of the first two tasks
def first_two_tasks_duration : Nat := 2 * 45

-- Define the break duration
def break_duration : Nat := 15

-- Define the duration of the third task
def third_task_duration : Nat := 2 * 45

-- Define the total duration
def total_duration : Nat := first_two_tasks_duration + break_duration + third_task_duration

-- Define the finish time in minutes since midnight
def finish_time : Nat := start_time + total_duration

-- Theorem to prove
theorem simone_finish_time : 
  finish_time = 11 * 60 + 15  -- 11:15 AM in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_simone_finish_time_l2500_250036


namespace NUMINAMATH_CALUDE_simons_age_is_45_l2500_250080

/-- Simon's age in 2010, given Jorge's age in 2005 and the age difference between Simon and Jorge -/
def simons_age_2010 (jorges_age_2005 : ℕ) (age_difference : ℕ) : ℕ :=
  jorges_age_2005 + (2010 - 2005) + age_difference

/-- Theorem stating that Simon's age in 2010 is 45 years old -/
theorem simons_age_is_45 :
  simons_age_2010 16 24 = 45 := by
  sorry

end NUMINAMATH_CALUDE_simons_age_is_45_l2500_250080


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2500_250052

/-- A geometric sequence with first term 1 and common ratio q -/
def geometric_sequence (q : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => q * geometric_sequence q n

theorem geometric_sequence_product (q : ℝ) (h : q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1) :
  ∃ m : ℕ, geometric_sequence q (m - 1) = (geometric_sequence q 0) *
    (geometric_sequence q 1) * (geometric_sequence q 2) *
    (geometric_sequence q 3) * (geometric_sequence q 4) ∧ m = 11 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2500_250052


namespace NUMINAMATH_CALUDE_average_monthly_balance_l2500_250005

def monthly_balances : List ℝ := [100, 200, 250, 50, 300, 300]
def num_months : ℕ := 6

theorem average_monthly_balance :
  (monthly_balances.sum / num_months) = 200 := by sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l2500_250005


namespace NUMINAMATH_CALUDE_range_of_m_for_cubic_equation_l2500_250037

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem range_of_m_for_cubic_equation (m : ℝ) :
  (∃ x ∈ Set.Icc 0 2, f x + m = 0) → m ∈ Set.Icc (-2) 2 := by
  sorry


end NUMINAMATH_CALUDE_range_of_m_for_cubic_equation_l2500_250037


namespace NUMINAMATH_CALUDE_tan_beta_minus_2alpha_l2500_250074

theorem tan_beta_minus_2alpha (α β : ℝ) 
  (h1 : Real.tan α = 1 / 2) 
  (h2 : Real.tan (α - β) = -1 / 3) : 
  Real.tan (β - 2 * α) = -1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_minus_2alpha_l2500_250074


namespace NUMINAMATH_CALUDE_larry_substitution_l2500_250019

theorem larry_substitution (a b c d f : ℚ) : 
  a = 12 → b = 4 → c = 3 → d = 5 →
  (a / (b / (c * (d - f))) = 12 / 4 / 3 * 5 - f) → f = 5 := by
  sorry

end NUMINAMATH_CALUDE_larry_substitution_l2500_250019


namespace NUMINAMATH_CALUDE_circle_inequality_l2500_250001

theorem circle_inequality (r s d : ℝ) (h1 : r > s) (h2 : r > 0) (h3 : s > 0) (h4 : d > 0) :
  r - s ≤ d :=
sorry

end NUMINAMATH_CALUDE_circle_inequality_l2500_250001


namespace NUMINAMATH_CALUDE_abs_sine_period_l2500_250092

-- Define the sine function and its period
noncomputable def sine_period : ℝ := 2 * Real.pi

-- Define the property that sine has this period
axiom sine_periodic (x : ℝ) : Real.sin (x + sine_period) = Real.sin x

-- Define the property that taking absolute value halves the period
axiom abs_halves_period {f : ℝ → ℝ} {p : ℝ} (h : ∀ x, f (x + p) = f x) :
  ∀ x, |f (x + p/2)| = |f x|

-- State the theorem
theorem abs_sine_period : 
  ∃ p : ℝ, p > 0 ∧ p = Real.pi ∧ ∀ x, |Real.sin (x + p)| = |Real.sin x| ∧
  ∀ q, q > 0 → (∀ x, |Real.sin (x + q)| = |Real.sin x|) → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_abs_sine_period_l2500_250092
