import Mathlib

namespace NUMINAMATH_CALUDE_equivalence_proof_l2198_219830

variable (P Q : Prop)

theorem equivalence_proof :
  (P → Q) ↔ (¬Q → ¬P) ∧ (¬P ∨ Q) ∧ ¬((Q → P) ↔ (P → Q)) :=
sorry

end NUMINAMATH_CALUDE_equivalence_proof_l2198_219830


namespace NUMINAMATH_CALUDE_inequality_proof_l2198_219825

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ 0) :
  (x^2 * y) / z + (y^2 * z) / x + (z^2 * x) / y ≥ x^2 + y^2 + z^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2198_219825


namespace NUMINAMATH_CALUDE_smallest_number_divisible_after_subtraction_l2198_219865

theorem smallest_number_divisible_after_subtraction : ∃ (n : ℕ), 
  (∀ (m : ℕ), m > 0 → (m - 8 : ℤ) % 9 = 0 ∧ (m - 8 : ℤ) % 6 = 0 ∧ 
   (m - 8 : ℤ) % 12 = 0 ∧ (m - 8 : ℤ) % 18 = 0 → m ≥ n) ∧
  (n - 8 : ℤ) % 9 = 0 ∧ (n - 8 : ℤ) % 6 = 0 ∧ 
  (n - 8 : ℤ) % 12 = 0 ∧ (n - 8 : ℤ) % 18 = 0 ∧
  n = 44 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_after_subtraction_l2198_219865


namespace NUMINAMATH_CALUDE_tangent_line_at_P_l2198_219893

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem tangent_line_at_P :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a*x + b*y + c = 0 ↔ 
      (y - f P.1 = (deriv f P.1) * (x - P.1) ∧ 
       (x, y) ≠ P)) ∧
    a = 3 ∧ b = -1 ∧ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_P_l2198_219893


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l2198_219876

theorem sqrt_sum_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) > 2 ∧
  ∀ n : ℝ, (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
    Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
    Real.sqrt (e / (a + b + c + d)) > n) →
  n ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l2198_219876


namespace NUMINAMATH_CALUDE_quadratic_max_value_l2198_219827

theorem quadratic_max_value (m : ℝ) (h_m : m ≠ 0) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, m * x^2 - 2 * m * x + 2 ≤ 4) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, m * x^2 - 2 * m * x + 2 = 4) →
  m = 2/3 ∨ m = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l2198_219827


namespace NUMINAMATH_CALUDE_blue_hat_cost_l2198_219834

theorem blue_hat_cost (total_hats : ℕ) (green_hat_cost : ℕ) (total_price : ℕ) (green_hats : ℕ) :
  total_hats = 85 →
  green_hat_cost = 7 →
  total_price = 548 →
  green_hats = 38 →
  (total_price - green_hats * green_hat_cost) / (total_hats - green_hats) = 6 :=
by sorry

end NUMINAMATH_CALUDE_blue_hat_cost_l2198_219834


namespace NUMINAMATH_CALUDE_range_of_m_l2198_219886

-- Define the equations
def equation1 (m x : ℝ) := x^2 + m*x + 1 = 0
def equation2 (m x : ℝ) := 4*x^2 + 4*(m-2)*x + 1 = 0

-- Define the conditions
def condition_p (m : ℝ) := ∃ x y, x < 0 ∧ y < 0 ∧ x ≠ y ∧ equation1 m x ∧ equation1 m y
def condition_q (m : ℝ) := ∀ x, ¬(equation2 m x)

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  ((condition_p m ∨ condition_q m) ∧ ¬(condition_p m ∧ condition_q m)) →
  (m ∈ Set.Ioo 1 2 ∪ Set.Ici 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2198_219886


namespace NUMINAMATH_CALUDE_inequality_solution_l2198_219842

theorem inequality_solution (x : ℝ) : 
  x + |2*x + 3| ≥ 2 ↔ x ≤ -5 ∨ x ≥ -1/3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2198_219842


namespace NUMINAMATH_CALUDE_multiplier_problem_l2198_219862

theorem multiplier_problem (n : ℝ) (h : n = 15) : 
  ∃ m : ℝ, 2 * n = (26 - n) + 19 ∧ n * m = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_multiplier_problem_l2198_219862


namespace NUMINAMATH_CALUDE_cd_equals_three_plus_b_l2198_219888

theorem cd_equals_three_plus_b 
  (a b c d : ℝ) 
  (h1 : a + b = 11) 
  (h2 : b + c = 9) 
  (h3 : a + d = 5) : 
  c + d = 3 + b := by
sorry

end NUMINAMATH_CALUDE_cd_equals_three_plus_b_l2198_219888


namespace NUMINAMATH_CALUDE_area_of_PQRS_l2198_219824

/-- A circle with an inscribed square ABCD and another inscribed square PQRS -/
structure InscribedSquares where
  /-- The radius of the circle -/
  r : ℝ
  /-- The side length of square ABCD -/
  s : ℝ
  /-- Half the side length of square PQRS -/
  t : ℝ
  /-- The area of square ABCD is 4 -/
  h_area : s^2 = 4
  /-- The radius of the circle is related to the side of ABCD -/
  h_radius : r^2 = 2 * s^2
  /-- Relationship between r, s, and t based on the Pythagorean theorem -/
  h_pythagorean : (s/2 + t)^2 + t^2 = r^2

/-- The area of square PQRS in the configuration of InscribedSquares -/
def areaOfPQRS (cfg : InscribedSquares) : ℝ := (2 * cfg.t)^2

/-- Theorem stating that the area of PQRS is 2 - √3 -/
theorem area_of_PQRS (cfg : InscribedSquares) : areaOfPQRS cfg = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_PQRS_l2198_219824


namespace NUMINAMATH_CALUDE_equation_A_is_circle_l2198_219894

/-- A polar equation represents a circle if and only if it describes all points at a constant distance from the origin. -/
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ ρ θ : ℝ, f ρ θ ↔ ρ = r

/-- The polar equation ρ = 1 -/
def equation_A (ρ θ : ℝ) : Prop := ρ = 1

theorem equation_A_is_circle : is_circle equation_A :=
sorry

end NUMINAMATH_CALUDE_equation_A_is_circle_l2198_219894


namespace NUMINAMATH_CALUDE_jordan_rectangle_length_l2198_219872

/-- Given two rectangles with equal area, where one rectangle measures 5 inches by 24 inches
    and the other has a width of 15 inches, prove that the length of the second rectangle is 8 inches. -/
theorem jordan_rectangle_length (carol_length carol_width jordan_width : ℕ) 
  (h1 : carol_length = 5)
  (h2 : carol_width = 24)
  (h3 : jordan_width = 15)
  (h4 : carol_length * carol_width = jordan_width * (carol_length * carol_width / jordan_width)) :
  carol_length * carol_width / jordan_width = 8 := by
  sorry

#check jordan_rectangle_length

end NUMINAMATH_CALUDE_jordan_rectangle_length_l2198_219872


namespace NUMINAMATH_CALUDE_article_price_proof_l2198_219810

-- Define the original price
def original_price : ℝ := 2500

-- Define the profit percentage
def profit_percentage : ℝ := 0.25

-- Define the profit amount
def profit_amount : ℝ := 625

-- Theorem statement
theorem article_price_proof :
  profit_amount = original_price * profit_percentage :=
by
  sorry

#check article_price_proof

end NUMINAMATH_CALUDE_article_price_proof_l2198_219810


namespace NUMINAMATH_CALUDE_C14_not_allotrope_C60_l2198_219879

/-- Represents an atom -/
structure Atom where
  name : String

/-- Represents a molecule -/
structure Molecule where
  name : String

/-- Defines the concept of allotrope -/
def is_allotrope (a b : Atom) : Prop :=
  ∃ (element : String), a.name = element ∧ b.name = element

/-- C14 is an atom -/
def C14 : Atom := ⟨"C14"⟩

/-- C60 is a molecule -/
def C60 : Molecule := ⟨"C60"⟩

/-- Theorem stating that C14 is not an allotrope of C60 -/
theorem C14_not_allotrope_C60 : ¬∃ (a : Atom), is_allotrope C14 a ∧ a.name = C60.name := by
  sorry

end NUMINAMATH_CALUDE_C14_not_allotrope_C60_l2198_219879


namespace NUMINAMATH_CALUDE_white_balls_count_l2198_219878

theorem white_balls_count (red_balls : ℕ) (total_balls : ℕ) (white_balls : ℕ) : 
  red_balls = 3 →
  (red_balls : ℚ) / total_balls = 1 / 4 →
  total_balls = red_balls + white_balls →
  white_balls = 9 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l2198_219878


namespace NUMINAMATH_CALUDE_janet_action_figures_l2198_219873

theorem janet_action_figures (x : ℕ) : 
  (x - 2 : ℤ) + 2 * (x - 2 : ℤ) = 24 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_janet_action_figures_l2198_219873


namespace NUMINAMATH_CALUDE_equation_solution_l2198_219870

theorem equation_solution (x y : ℝ) :
  3 * x^2 - 12 * y^2 = 0 ↔ (x = 2*y ∨ x = -2*y) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2198_219870


namespace NUMINAMATH_CALUDE_third_packing_number_l2198_219864

theorem third_packing_number (N : ℕ) (h1 : N = 301) (h2 : N % 3 = 1) (h3 : N % 4 = 1) (h4 : N % 7 = 0) :
  ∃ x : ℕ, x ≠ 3 ∧ x ≠ 4 ∧ x > 4 ∧ N % x = 1 ∧ (∀ y : ℕ, y ≠ 3 ∧ y ≠ 4 ∧ y < x ∧ y > 4 → N % y ≠ 1) ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_third_packing_number_l2198_219864


namespace NUMINAMATH_CALUDE_round_table_seating_l2198_219848

theorem round_table_seating (W M : ℕ) : 
  W = 19 → 
  M = 16 → 
  (7 : ℕ) + 12 = W → 
  (3 : ℕ) * 12 = 3 * W - 3 * M → 
  W + M = 35 := by
sorry

end NUMINAMATH_CALUDE_round_table_seating_l2198_219848


namespace NUMINAMATH_CALUDE_change_ways_50_cents_l2198_219899

/-- Represents the number of ways to make change for a given amount using pennies, nickels, and dimes. -/
def changeWays (amount : ℕ) : ℕ := sorry

/-- The value of a penny in cents -/
def pennyValue : ℕ := 1

/-- The value of a nickel in cents -/
def nickelValue : ℕ := 5

/-- The value of a dime in cents -/
def dimeValue : ℕ := 10

/-- The total amount we want to make change for, in cents -/
def totalAmount : ℕ := 50

theorem change_ways_50_cents :
  changeWays totalAmount = 35 := by sorry

end NUMINAMATH_CALUDE_change_ways_50_cents_l2198_219899


namespace NUMINAMATH_CALUDE_polynomial_one_root_product_l2198_219880

theorem polynomial_one_root_product (d e : ℝ) : 
  (∃! x : ℝ, x^2 + d*x + e = 0) → 
  d = 2*e - 3 → 
  ∃ e₁ e₂ : ℝ, (∀ e' : ℝ, (∃ x : ℝ, x^2 + d*x + e' = 0) → (e' = e₁ ∨ e' = e₂)) ∧ 
              e₁ * e₂ = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_one_root_product_l2198_219880


namespace NUMINAMATH_CALUDE_senior_japanese_fraction_l2198_219849

theorem senior_japanese_fraction (j : ℝ) (s : ℝ) (x : ℝ) :
  s = 2 * j →                     -- Senior class is twice the size of junior class
  (1 / 3) * (j + s) = (3 / 4) * j + x * s →  -- 1/3 of all students equals 3/4 of juniors plus x fraction of seniors
  x = 1 / 8 :=                    -- Fraction of seniors studying Japanese
by sorry

end NUMINAMATH_CALUDE_senior_japanese_fraction_l2198_219849


namespace NUMINAMATH_CALUDE_final_result_l2198_219837

theorem final_result (chosen_number : ℕ) (h : chosen_number = 1152) : 
  (chosen_number / 6 : ℚ) - 189 = 3 := by
  sorry

end NUMINAMATH_CALUDE_final_result_l2198_219837


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2198_219860

theorem pure_imaginary_complex_number (m : ℝ) : 
  (∃ (z : ℂ), z = (m^2 - 1) + (m - 1) * I ∧ z.re = 0 ∧ z.im ≠ 0) → m = -1 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2198_219860


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l2198_219846

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the xy-plane of the form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Predicate to check if a circle is symmetric with respect to a line -/
def is_symmetric (c : Circle) (l : Line) : Prop :=
  c.center.1 + c.center.2 = l.slope * c.center.1 + l.intercept + c.center.2

theorem circle_symmetry_line (b : ℝ) : 
  let c : Circle := { center := (1, 2), radius := 1 }
  let l : Line := { slope := 1, intercept := b }
  is_symmetric c l → b = 1 := by
  sorry

#check circle_symmetry_line

end NUMINAMATH_CALUDE_circle_symmetry_line_l2198_219846


namespace NUMINAMATH_CALUDE_can_identify_counterfeit_coins_l2198_219898

/-- Represents the result of checking a pair of coins -/
inductive CheckResult
  | Zero
  | One
  | Two

/-- Represents a coin -/
inductive Coin
  | One
  | Two
  | Three
  | Four
  | Five

/-- A function that checks a pair of coins and returns the number of counterfeit coins -/
def checkPair (c1 c2 : Coin) : CheckResult := sorry

/-- The set of all coins -/
def allCoins : Finset Coin := sorry

/-- The set of counterfeit coins -/
def counterfeitCoins : Finset Coin := sorry

/-- The four pairs of coins to be checked -/
def pairsToCheck : List (Coin × Coin) := sorry

theorem can_identify_counterfeit_coins :
  (Finset.card allCoins = 5) →
  (Finset.card counterfeitCoins = 2) →
  (List.length pairsToCheck = 4) →
  ∃ (f : List CheckResult → Finset Coin),
    ∀ (results : List CheckResult),
      List.length results = 4 →
      results = List.map (fun (p : Coin × Coin) => checkPair p.1 p.2) pairsToCheck →
      f results = counterfeitCoins :=
sorry

end NUMINAMATH_CALUDE_can_identify_counterfeit_coins_l2198_219898


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l2198_219836

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l2198_219836


namespace NUMINAMATH_CALUDE_fourth_day_temperature_l2198_219882

def temperature_problem (t1 t2 t3 avg : ℚ) : Prop :=
  let sum3 := t1 + t2 + t3
  let sum4 := 4 * avg
  sum4 - sum3 = -36

theorem fourth_day_temperature :
  temperature_problem 13 (-15) (-10) (-12) := by sorry

end NUMINAMATH_CALUDE_fourth_day_temperature_l2198_219882


namespace NUMINAMATH_CALUDE_value_of_a_l2198_219874

theorem value_of_a (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) 
  (h : ∀ x : ℝ, x^2 + 2*x^10 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                 a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10) : 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l2198_219874


namespace NUMINAMATH_CALUDE_expression_evaluation_l2198_219808

theorem expression_evaluation (x : ℤ) (h1 : -1 ≤ x) (h2 : x ≤ 2) 
  (h3 : x ≠ 1) (h4 : x ≠ 0) (h5 : x ≠ 2) : 
  (x^2 - 1) / (x^2 - 2*x + 1) + (x^2 - 2*x) / (x - 2) / x = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2198_219808


namespace NUMINAMATH_CALUDE_min_distance_sum_l2198_219896

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 1

/-- Definition of the locus of point P -/
def P_locus (a b : ℝ) : Prop := b = -(1/2) * a + 5/2

/-- The main theorem -/
theorem min_distance_sum (a b : ℝ) : 
  C₁ a b → C₂ a b → P_locus a b → 
  Real.sqrt (a^2 + b^2) + Real.sqrt ((a - 5)^2 + (b + 1)^2) ≥ Real.sqrt 34 :=
sorry

end NUMINAMATH_CALUDE_min_distance_sum_l2198_219896


namespace NUMINAMATH_CALUDE_ivan_commute_l2198_219887

theorem ivan_commute (T : ℝ) (D : ℝ) (h1 : T > 0) (h2 : D > 0) : 
  let v := D / T
  let new_time := T - 65
  (D / (1.6 * v) = new_time) → 
  (D / (1.3 * v) = T - 40) :=
by sorry

end NUMINAMATH_CALUDE_ivan_commute_l2198_219887


namespace NUMINAMATH_CALUDE_g_of_x_plus_3_l2198_219835

def g (x : ℝ) : ℝ := x^2 - x

theorem g_of_x_plus_3 : g (x + 3) = x^2 + 5*x + 6 := by sorry

end NUMINAMATH_CALUDE_g_of_x_plus_3_l2198_219835


namespace NUMINAMATH_CALUDE_unique_n_l2198_219812

theorem unique_n : ∃! n : ℤ, 
  50 ≤ n ∧ n ≤ 150 ∧ 
  7 ∣ n ∧ 
  n % 9 = 3 ∧ 
  n % 4 = 3 ∧
  n = 147 := by
sorry

end NUMINAMATH_CALUDE_unique_n_l2198_219812


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2198_219890

theorem inequality_solution_set (x : ℝ) :
  (x + 5) * (3 - 2*x) ≤ 6 ↔ x ≤ -9/2 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2198_219890


namespace NUMINAMATH_CALUDE_smallest_k_for_distinct_roots_l2198_219852

theorem smallest_k_for_distinct_roots (k : ℤ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 3 * x - 9/4 = 0 ∧ k * y^2 - 3 * y - 9/4 = 0) →
  k ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_distinct_roots_l2198_219852


namespace NUMINAMATH_CALUDE_eel_length_problem_l2198_219839

theorem eel_length_problem (jenna_eel : ℝ) (bill_eel : ℝ) : 
  jenna_eel = (1/3 : ℝ) * bill_eel → 
  jenna_eel + bill_eel = 64 → 
  jenna_eel = 16 := by
  sorry

end NUMINAMATH_CALUDE_eel_length_problem_l2198_219839


namespace NUMINAMATH_CALUDE_mothers_day_bouquet_l2198_219829

/-- Represents the flower shop problem --/
structure FlowerShop where
  carnation_price : ℚ
  rose_price : ℚ
  processing_fee : ℚ
  total_budget : ℚ
  total_flowers : ℕ

/-- Represents a bouquet composition --/
structure Bouquet where
  carnations : ℕ
  roses : ℕ

/-- Checks if a bouquet satisfies the conditions of the flower shop problem --/
def is_valid_bouquet (shop : FlowerShop) (bouquet : Bouquet) : Prop :=
  let total_cost := shop.carnation_price * bouquet.carnations + shop.rose_price * bouquet.roses + shop.processing_fee
  bouquet.carnations + bouquet.roses = shop.total_flowers ∧
  total_cost = shop.total_budget

/-- The main theorem to prove --/
theorem mothers_day_bouquet : 
  let shop := FlowerShop.mk 1.5 2 2 21 10
  let bouquet := Bouquet.mk 2 8
  is_valid_bouquet shop bouquet := by
  sorry

end NUMINAMATH_CALUDE_mothers_day_bouquet_l2198_219829


namespace NUMINAMATH_CALUDE_lars_bakeshop_production_l2198_219871

/-- Lars' bakeshop productivity calculation -/
theorem lars_bakeshop_production :
  let loaves_per_hour : ℕ := 10
  let baguettes_per_two_hours : ℕ := 30
  let working_hours_per_day : ℕ := 6
  
  let loaves_per_day : ℕ := loaves_per_hour * working_hours_per_day
  let baguette_intervals : ℕ := working_hours_per_day / 2
  let baguettes_per_day : ℕ := baguettes_per_two_hours * baguette_intervals
  
  loaves_per_day + baguettes_per_day = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_lars_bakeshop_production_l2198_219871


namespace NUMINAMATH_CALUDE_AQ_length_l2198_219877

/-- Square ABCD with side length 10 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (10, 0) ∧ C = (10, 10) ∧ D = (0, 10))

/-- Points P, Q, R, X, Y -/
structure SpecialPoints (ABCD : Square) :=
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (R : ℝ × ℝ)
  (X : ℝ × ℝ)
  (Y : ℝ × ℝ)
  (P_on_CD : P.2 = 10)
  (Q_on_AD : Q.1 = 0)
  (R_on_CD : R.2 = 10)
  (BQ_perp_AP : (Q.2 / 10) * ((10 - Q.2) / P.1) = -1)
  (RQ_parallel_PA : (Q.2 - 10) / (-P.1) = (10 - Q.2) / P.1)
  (X_on_BC_AP : X.1 = 10 ∧ X.2 = (10 - Q.2) * (X.1 / P.1) + Q.2)
  (Y_on_circumcircle : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (Y.1 - center.1)^2 + (Y.2 - center.2)^2 = radius^2 ∧
    (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius^2 ∧
    (Q.1 - center.1)^2 + (Q.2 - center.2)^2 = radius^2 ∧
    (0 - center.1)^2 + (10 - center.2)^2 = radius^2)
  (angle_PYR : Real.cos (105 * π / 180) = 
    ((Y.1 - P.1) * (R.1 - Y.1) + (Y.2 - P.2) * (R.2 - Y.2)) /
    (Real.sqrt ((Y.1 - P.1)^2 + (Y.2 - P.2)^2) * Real.sqrt ((R.1 - Y.1)^2 + (R.2 - Y.2)^2)))

/-- The main theorem -/
theorem AQ_length (ABCD : Square) (points : SpecialPoints ABCD) :
  Real.sqrt ((points.Q.1 - ABCD.A.1)^2 + (points.Q.2 - ABCD.A.2)^2) = 10 * Real.sqrt 3 - 10 := by
  sorry

end NUMINAMATH_CALUDE_AQ_length_l2198_219877


namespace NUMINAMATH_CALUDE_log_sum_equals_three_l2198_219809

theorem log_sum_equals_three : 
  Real.log 0.125 / Real.log 0.5 + Real.log (Real.log (Real.log 64 / Real.log 4) / Real.log 3) / Real.log 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_three_l2198_219809


namespace NUMINAMATH_CALUDE_travel_time_calculation_l2198_219861

theorem travel_time_calculation (total_distance : ℝ) (foot_speed : ℝ) (bicycle_speed : ℝ) (foot_distance : ℝ)
  (h1 : total_distance = 61)
  (h2 : foot_speed = 4)
  (h3 : bicycle_speed = 9)
  (h4 : foot_distance = 16) :
  (foot_distance / foot_speed) + ((total_distance - foot_distance) / bicycle_speed) = 9 :=
by sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l2198_219861


namespace NUMINAMATH_CALUDE_unique_non_right_triangle_l2198_219841

/-- A function that checks if three numbers can form a right-angled triangle -/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The theorem stating that among the given sets, only (7, 24, 26) cannot form a right-angled triangle -/
theorem unique_non_right_triangle :
  is_right_triangle 3 4 5 ∧
  is_right_triangle 5 12 13 ∧
  is_right_triangle 8 15 17 ∧
  ¬ is_right_triangle 7 24 26 :=
sorry

end NUMINAMATH_CALUDE_unique_non_right_triangle_l2198_219841


namespace NUMINAMATH_CALUDE_sara_quarters_l2198_219838

/-- The total number of quarters after receiving additional quarters -/
def total_quarters (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Sara's total quarters is the sum of her initial quarters and additional quarters -/
theorem sara_quarters : total_quarters 21 49 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l2198_219838


namespace NUMINAMATH_CALUDE_sqrt_x_plus_sqrt_y_equals_two_l2198_219863

theorem sqrt_x_plus_sqrt_y_equals_two (θ : ℝ) (x y : ℝ) 
  (h1 : x + y = 3 - Real.cos (4 * θ)) 
  (h2 : x - y = 4 * Real.sin (2 * θ)) : 
  Real.sqrt x + Real.sqrt y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_sqrt_y_equals_two_l2198_219863


namespace NUMINAMATH_CALUDE_distance_between_points_l2198_219859

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-5, 2)
  let p2 : ℝ × ℝ := (7, 7)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 13 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l2198_219859


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l2198_219828

-- Define the geometric sequence
def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ :=
  b * x^2 + c * x + a

-- State the theorem
theorem unique_root_quadratic (a b c : ℝ) :
  is_geometric_sequence a b c →
  a ≤ b →
  b ≤ c →
  c ≤ 1 →
  (∃! x : ℝ, quadratic a b c x = 0) →
  (∃ x : ℝ, quadratic a b c x = 0 ∧ x = -(Real.rpow 4 (1/3) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l2198_219828


namespace NUMINAMATH_CALUDE_place_value_sum_place_value_sum_holds_l2198_219818

theorem place_value_sum : Real → Prop :=
  fun x => 
    let ten_thousands : Real := 4
    let thousands : Real := 3
    let hundreds : Real := 7
    let tens : Real := 5
    let ones : Real := 2
    let tenths : Real := 8
    let hundredths : Real := 4
    x = ten_thousands * 10000 + thousands * 1000 + hundreds * 100 + 
        tens * 10 + ones + tenths / 10 + hundredths / 100 ∧ 
    x = 43752.84

theorem place_value_sum_holds : ∃ x, place_value_sum x := by
  sorry

end NUMINAMATH_CALUDE_place_value_sum_place_value_sum_holds_l2198_219818


namespace NUMINAMATH_CALUDE_square_of_trinomial_13_5_3_l2198_219884

theorem square_of_trinomial_13_5_3 : (13 + 5 + 3)^2 = 441 := by
  sorry

end NUMINAMATH_CALUDE_square_of_trinomial_13_5_3_l2198_219884


namespace NUMINAMATH_CALUDE_tangent_line_to_ellipse_l2198_219869

/-- Given a line y = mx + 3 tangent to the ellipse 4x^2 + y^2 = 4, m^2 = 5 -/
theorem tangent_line_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 3 → 4 * x^2 + y^2 = 4) → 
  (∃! x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + y^2 = 4) → 
  m^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ellipse_l2198_219869


namespace NUMINAMATH_CALUDE_coat_price_calculation_l2198_219821

/-- The total selling price of a coat after discount and tax -/
def totalSellingPrice (originalPrice discount taxRate : ℝ) : ℝ :=
  let salePrice := originalPrice * (1 - discount)
  salePrice * (1 + taxRate)

/-- Theorem: The total selling price of a $120 coat with 30% discount and 8% tax is $90.72 -/
theorem coat_price_calculation :
  totalSellingPrice 120 0.3 0.08 = 90.72 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_calculation_l2198_219821


namespace NUMINAMATH_CALUDE_triangle_side_length_l2198_219800

theorem triangle_side_length (a c b : ℝ) (B : ℝ) : 
  a = 3 * Real.sqrt 3 → 
  c = 2 → 
  B = 150 * π / 180 → 
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B → 
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2198_219800


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l2198_219858

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := fun x => x^2 - 2*x - 8
  ∃ (x₁ x₂ : ℝ), x₁ = 4 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l2198_219858


namespace NUMINAMATH_CALUDE_exponential_comparison_l2198_219868

theorem exponential_comparison : 0.3^2.1 < 2.1^0.3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_comparison_l2198_219868


namespace NUMINAMATH_CALUDE_odd_function_property_l2198_219820

/-- A function f(x) = ax^5 - bx^3 + cx is odd and f(-3) = 7 implies f(3) = -7 -/
theorem odd_function_property (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^5 - b * x^3 + c * x)
  (h2 : f (-3) = 7) : 
  f 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l2198_219820


namespace NUMINAMATH_CALUDE_decimal_to_base5_conversion_l2198_219844

-- Define a function to convert a base-5 number to base-10
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define the base-5 representation of the number we want to prove
def base5Representation : List Nat := [0, 0, 2, 1]

-- State the theorem
theorem decimal_to_base5_conversion :
  base5ToBase10 base5Representation = 175 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_base5_conversion_l2198_219844


namespace NUMINAMATH_CALUDE_repeating_decimal_prime_l2198_219817

/-- A function that determines if a rational number has a repeating decimal representation with a given period length. -/
def has_repeating_decimal_period (q : ℚ) (period : ℕ) : Prop :=
  ∃ (k : ℕ) (r : ℚ), q = k + r ∧ r < 1 ∧ (10 ^ period : ℚ) * r = r

theorem repeating_decimal_prime (n : ℕ) (h1 : n > 1) 
  (h2 : has_repeating_decimal_period (1 / n : ℚ) (n - 1)) : 
  Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_prime_l2198_219817


namespace NUMINAMATH_CALUDE_basketball_team_cutoff_l2198_219853

theorem basketball_team_cutoff (girls boys callback : ℕ) 
  (h1 : girls = 17) 
  (h2 : boys = 32) 
  (h3 : callback = 10) : 
  girls + boys - callback = 39 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_cutoff_l2198_219853


namespace NUMINAMATH_CALUDE_smallest_m_divisibility_l2198_219867

def a : ℕ → ℕ
  | 0 => 3
  | n + 1 => (n + 2) * a n - (n + 1)

def divisible (x y : ℕ) : Prop := ∃ k, y = k * x

theorem smallest_m_divisibility :
  ∀ m : ℕ, m ≥ 2005 →
    (divisible (a (m + 1) - 1) (a m ^ 2 - 1) ∧
     ∀ k : ℕ, 2005 ≤ k ∧ k < m →
       ¬divisible (a (k + 1) - 1) (a k ^ 2 - 1)) ↔
    m = 2010 := by sorry

end NUMINAMATH_CALUDE_smallest_m_divisibility_l2198_219867


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l2198_219885

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point on a circle. -/
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

/-- A coloring function that assigns either red or blue to each point on the circle. -/
def Coloring (c : Circle) := PointOnCircle c → Bool

/-- Predicate to check if three points form a right-angled triangle. -/
def IsRightAngledTriangle (c : Circle) (p1 p2 p3 : PointOnCircle c) : Prop :=
  ∃ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    let points := [p1, p2, p3]
    let a := points[i].point
    let b := points[j].point
    let c := points[k].point
    (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0

/-- The main theorem: there exists a coloring such that no inscribed right-angled triangle
    has all vertices of the same color. -/
theorem exists_valid_coloring (c : Circle) :
  ∃ (coloring : Coloring c),
    ∀ (p1 p2 p3 : PointOnCircle c),
      IsRightAngledTriangle c p1 p2 p3 →
        coloring p1 ≠ coloring p2 ∨ coloring p2 ≠ coloring p3 ∨ coloring p1 ≠ coloring p3 :=
by sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l2198_219885


namespace NUMINAMATH_CALUDE_solve_equation_l2198_219823

theorem solve_equation (x : ℝ) : 3 + 2 * (x - 3) = 24.16 → x = 13.58 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2198_219823


namespace NUMINAMATH_CALUDE_floor_equality_iff_range_l2198_219816

theorem floor_equality_iff_range (x : ℝ) : 
  ⌊⌊3 * x⌋ - 1⌋ = ⌊x + 1⌋ ↔ 2/3 ≤ x ∧ x < 4/3 :=
by sorry

end NUMINAMATH_CALUDE_floor_equality_iff_range_l2198_219816


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2198_219831

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt ((6 / (x + 1)) - 1)}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 2*x + 3)}

-- Statement to prove
theorem intersection_A_complement_B : 
  A ∩ (Set.univ \ B) = {x : ℝ | 3 ≤ x ∧ x ≤ 5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2198_219831


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2198_219843

-- Define set A
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2198_219843


namespace NUMINAMATH_CALUDE_spade_heart_eval_l2198_219845

/-- Operation ♠ for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- Operation ♥ for real numbers -/
def heart (x y : ℝ) : ℝ := x^2 - y^2

/-- Theorem stating that 5 ♠ (3 ♥ 2) = 0 -/
theorem spade_heart_eval : spade 5 (heart 3 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_spade_heart_eval_l2198_219845


namespace NUMINAMATH_CALUDE_min_colors_is_thirteen_l2198_219819

/-- A coloring function for a 25x25 chessboard. -/
def Coloring := Fin 25 → Fin 25 → ℕ

/-- Predicate to check if a coloring satisfies the given condition. -/
def ValidColoring (c : Coloring) : Prop :=
  ∀ i j s t, 1 ≤ i ∧ i < j ∧ j ≤ 25 ∧ 1 ≤ s ∧ s < t ∧ t ≤ 25 →
    (c i s ≠ c j s ∨ c i s ≠ c j t ∨ c j s ≠ c j t)

/-- The minimum number of colors needed for a valid coloring. -/
def MinColors : ℕ := 13

/-- Theorem stating that 13 is the smallest number of colors needed for a valid coloring. -/
theorem min_colors_is_thirteen :
  (∃ c : Coloring, ValidColoring c ∧ (∀ i j, c i j < MinColors)) ∧
  (∀ n : ℕ, n < MinColors →
    ¬∃ c : Coloring, ValidColoring c ∧ (∀ i j, c i j < n)) := by
  sorry

end NUMINAMATH_CALUDE_min_colors_is_thirteen_l2198_219819


namespace NUMINAMATH_CALUDE_multiple_of_119_l2198_219807

theorem multiple_of_119 : ∃ k : ℤ, 119 = 7 * k ∧ 
  (∀ m : ℤ, 119 ≠ 2 * m) ∧ 
  (∀ n : ℤ, 119 ≠ 3 * n) ∧ 
  (∀ p : ℤ, 119 ≠ 5 * p) ∧ 
  (∀ q : ℤ, 119 ≠ 11 * q) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_119_l2198_219807


namespace NUMINAMATH_CALUDE_road_repaving_l2198_219801

theorem road_repaving (total_repaved : ℕ) (previously_repaved : ℕ) :
  total_repaved = 4938 ∧ previously_repaved = 4133 →
  total_repaved - previously_repaved = 805 := by
  sorry

end NUMINAMATH_CALUDE_road_repaving_l2198_219801


namespace NUMINAMATH_CALUDE_product_of_roots_l2198_219856

theorem product_of_roots (x : ℝ) : (x - 4) * (x + 5) = -24 → ∃ y : ℝ, (x * y = 4 ∧ (y - 4) * (y + 5) = -24) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2198_219856


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2198_219850

/-- Given an inequality (ax-1)(x+1) < 0 with respect to x, where the solution set
    is (-∞, 1/a) ∪ (-1, +∞), prove that the range of the real number a is -1 ≤ a < 0. -/
theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, (a * x - 1) * (x + 1) < 0 ↔ x ∈ ({y : ℝ | y < (1 : ℝ) / a} ∪ {y : ℝ | y > -1})) →
  -1 ≤ a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2198_219850


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l2198_219897

theorem fraction_sum_inequality (a b c : ℝ) :
  a / (a + 2*b + c) + b / (a + b + 2*c) + c / (2*a + b + c) ≥ 3/4 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l2198_219897


namespace NUMINAMATH_CALUDE_unique_prime_pair_square_sum_l2198_219892

theorem unique_prime_pair_square_sum : 
  ∀ p q : ℕ, 
    Prime p → Prime q → p > 0 → q > 0 →
    (∃ n : ℕ, p^(q-1) + q^(p-1) = n^2) →
    p = 2 ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_pair_square_sum_l2198_219892


namespace NUMINAMATH_CALUDE_shaded_perimeter_equals_48_l2198_219826

/-- Represents a circle in the arrangement -/
structure Circle where
  circumference : ℝ

/-- Represents the arrangement of four circles -/
structure CircleArrangement where
  circles : Fin 4 → Circle
  symmetric : Bool
  touching : Bool

/-- Calculates the perimeter of the shaded region -/
def shadedPerimeter (arrangement : CircleArrangement) : ℝ :=
  sorry

theorem shaded_perimeter_equals_48 (arrangement : CircleArrangement) 
    (h1 : ∀ i, (arrangement.circles i).circumference = 48) 
    (h2 : arrangement.symmetric = true) 
    (h3 : arrangement.touching = true) : 
  shadedPerimeter arrangement = 48 := by
  sorry

end NUMINAMATH_CALUDE_shaded_perimeter_equals_48_l2198_219826


namespace NUMINAMATH_CALUDE_height_percentage_difference_l2198_219822

theorem height_percentage_difference (A B : ℝ) (h : B = A * (1 + 1/3)) :
  (B - A) / B = 1/4 := by sorry

end NUMINAMATH_CALUDE_height_percentage_difference_l2198_219822


namespace NUMINAMATH_CALUDE_smallest_integer_divisibility_l2198_219854

theorem smallest_integer_divisibility (n : ℕ) : 
  ∃ (a_n : ℤ), 
    (a_n > (Real.sqrt 3 + 1)^(2*n)) ∧ 
    (∀ (x : ℤ), x > (Real.sqrt 3 + 1)^(2*n) → a_n ≤ x) ∧ 
    (∃ (k : ℤ), a_n = 2^(n+1) * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_divisibility_l2198_219854


namespace NUMINAMATH_CALUDE_fraction_zero_at_zero_l2198_219802

theorem fraction_zero_at_zero (x : ℝ) : 
  (2 * x) / (x + 3) = 0 ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_at_zero_l2198_219802


namespace NUMINAMATH_CALUDE_find_divisor_l2198_219803

theorem find_divisor (x n : ℕ) (h1 : ∃ k : ℕ, x = k * n + 27)
                     (h2 : ∃ m : ℕ, x = 8 * m + 3)
                     (h3 : n > 27) :
  n = 32 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2198_219803


namespace NUMINAMATH_CALUDE_number_of_pupils_l2198_219847

theorem number_of_pupils (total_people : ℕ) (parents : ℕ) (pupils : ℕ) : 
  total_people = 676 → parents = 22 → pupils = total_people - parents → pupils = 654 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_l2198_219847


namespace NUMINAMATH_CALUDE_repeating_decimal_56_eq_fraction_l2198_219813

/-- The decimal representation of a number with infinitely repeating digits 56 after the decimal point -/
def repeating_decimal_56 : ℚ :=
  56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to 56/99 -/
theorem repeating_decimal_56_eq_fraction : repeating_decimal_56 = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_56_eq_fraction_l2198_219813


namespace NUMINAMATH_CALUDE_point_to_line_distance_l2198_219805

theorem point_to_line_distance (a : ℝ) : 
  (∃ d : ℝ, d = 4 ∧ d = |3*a - 4*6 - 2| / Real.sqrt (3^2 + (-4)^2)) →
  (a = 2 ∨ a = 46/3) :=
sorry

end NUMINAMATH_CALUDE_point_to_line_distance_l2198_219805


namespace NUMINAMATH_CALUDE_insert_two_digits_into_five_digit_number_l2198_219889

/-- The number of ways to insert two indistinguishable digits into a 5-digit number to form a 7-digit number -/
def insert_two_digits (n : ℕ) : ℕ :=
  let total_positions := n + 1
  let total_arrangements := total_positions * total_positions
  let arrangements_together := total_positions
  total_arrangements - arrangements_together

/-- The theorem stating that inserting two indistinguishable digits into a 5-digit number results in 30 different 7-digit numbers -/
theorem insert_two_digits_into_five_digit_number :
  insert_two_digits 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_insert_two_digits_into_five_digit_number_l2198_219889


namespace NUMINAMATH_CALUDE_arrangements_equal_l2198_219857

/-- The number of arrangements when adding 2 books to 3 existing books while keeping their relative order --/
def arrangements_books : ℕ := 20

/-- The number of arrangements for 7 people with height constraints --/
def arrangements_people : ℕ := 20

/-- Theorem stating that both arrangement problems result in 20 different arrangements --/
theorem arrangements_equal : arrangements_books = arrangements_people := by
  sorry

end NUMINAMATH_CALUDE_arrangements_equal_l2198_219857


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_18_l2198_219806

theorem factorization_of_2x_squared_minus_18 (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_18_l2198_219806


namespace NUMINAMATH_CALUDE_no_valid_triples_l2198_219814

theorem no_valid_triples :
  ¬∃ (x y z : ℕ+), 
    (Nat.lcm x.val y.val = 180) ∧
    (Nat.lcm x.val z.val = 450) ∧
    (Nat.lcm y.val z.val = 600) ∧
    (x.val + y.val + z.val = 120) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_triples_l2198_219814


namespace NUMINAMATH_CALUDE_marker_cost_l2198_219883

theorem marker_cost (total_students : ℕ) (total_cost : ℕ) 
  (h_total_students : total_students = 40)
  (h_total_cost : total_cost = 3388) :
  ∃ (s n c : ℕ),
    s > total_students / 2 ∧
    s ≤ total_students ∧
    n > 1 ∧
    c > n ∧
    s * n * c = total_cost ∧
    c = 11 := by
  sorry

end NUMINAMATH_CALUDE_marker_cost_l2198_219883


namespace NUMINAMATH_CALUDE_sin_pi_sixth_plus_tan_pi_third_l2198_219866

theorem sin_pi_sixth_plus_tan_pi_third :
  Real.sin (π / 6) + Real.tan (π / 3) = 1 / 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_sixth_plus_tan_pi_third_l2198_219866


namespace NUMINAMATH_CALUDE_probability_red_green_white_l2198_219851

def red_marbles : ℕ := 4
def green_marbles : ℕ := 5
def white_marbles : ℕ := 11

def total_marbles : ℕ := red_marbles + green_marbles + white_marbles

theorem probability_red_green_white :
  (red_marbles : ℚ) / total_marbles *
  green_marbles / (total_marbles - 1) *
  white_marbles / (total_marbles - 2) = 11 / 342 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_green_white_l2198_219851


namespace NUMINAMATH_CALUDE_fanfan_distance_is_120_l2198_219804

/-- Represents the cost and distance information for a shared car journey -/
structure JourneyInfo where
  ningning_cost : ℝ
  leilei_cost : ℝ
  fanfan_cost : ℝ
  ningning_distance : ℝ

/-- Calculates the distance to Fanfan's home given the journey information -/
def calculate_fanfan_distance (info : JourneyInfo) : ℝ :=
  sorry

/-- Theorem stating that given the journey information, Fanfan's home is 120 km from school -/
theorem fanfan_distance_is_120 (info : JourneyInfo) 
  (h1 : info.ningning_cost = 10)
  (h2 : info.leilei_cost = 25)
  (h3 : info.fanfan_cost = 85)
  (h4 : info.ningning_distance = 12) :
  calculate_fanfan_distance info = 120 := by
  sorry

end NUMINAMATH_CALUDE_fanfan_distance_is_120_l2198_219804


namespace NUMINAMATH_CALUDE_concrete_amount_l2198_219895

/-- The amount of bricks ordered in tons -/
def bricks : ℝ := 0.17

/-- The amount of stone ordered in tons -/
def stone : ℝ := 0.5

/-- The total amount of material ordered in tons -/
def total_material : ℝ := 0.83

/-- The amount of concrete ordered in tons -/
def concrete : ℝ := total_material - (bricks + stone)

theorem concrete_amount : concrete = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_concrete_amount_l2198_219895


namespace NUMINAMATH_CALUDE_solve_for_a_l2198_219881

theorem solve_for_a : ∃ a : ℝ, (3 * 3 - 2 * a = 5) ∧ a = 2 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l2198_219881


namespace NUMINAMATH_CALUDE_double_price_profit_percentage_l2198_219815

theorem double_price_profit_percentage (cost : ℝ) (initial_profit_rate : ℝ) 
  (initial_selling_price : ℝ) (new_selling_price : ℝ) (new_profit_rate : ℝ) :
  initial_profit_rate = 0.20 →
  initial_selling_price = cost * (1 + initial_profit_rate) →
  new_selling_price = 2 * initial_selling_price →
  new_profit_rate = (new_selling_price - cost) / cost →
  new_profit_rate = 1.40 :=
by sorry

end NUMINAMATH_CALUDE_double_price_profit_percentage_l2198_219815


namespace NUMINAMATH_CALUDE_porter_painting_sale_l2198_219811

/-- The sale price of Porter's previous painting in dollars -/
def previous_sale : ℕ := 9000

/-- The sale price of Porter's most recent painting in dollars -/
def recent_sale : ℕ := 5 * previous_sale - 1000

theorem porter_painting_sale : recent_sale = 44000 := by
  sorry

end NUMINAMATH_CALUDE_porter_painting_sale_l2198_219811


namespace NUMINAMATH_CALUDE_square_area_ratio_when_doubled_l2198_219875

theorem square_area_ratio_when_doubled (s : ℝ) (h : s > 0) :
  (s^2) / ((2*s)^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_when_doubled_l2198_219875


namespace NUMINAMATH_CALUDE_tan_angle_equality_l2198_219891

theorem tan_angle_equality (n : ℤ) : 
  -150 < n ∧ n < 150 ∧ Real.tan (n * π / 180) = Real.tan (1600 * π / 180) → n = -20 :=
by sorry

end NUMINAMATH_CALUDE_tan_angle_equality_l2198_219891


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l2198_219840

theorem min_value_and_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (∃ (y : ℝ), y = (a + 1/a) * (b + 1/b) ∧ 
    (∀ (z : ℝ), z = (a + 1/a) * (b + 1/b) → y ≤ z) ∧ 
    y = 25/4) ∧ 
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l2198_219840


namespace NUMINAMATH_CALUDE_simplify_expression_l2198_219832

theorem simplify_expression (b : ℝ) : ((3 * b + 6) - 6 * b) / 3 = -b + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2198_219832


namespace NUMINAMATH_CALUDE_leakage_empty_time_l2198_219833

/-- Given a pipe that fills a tank in 'a' hours without leakage,
    and takes 7 times longer with leakage, the time taken by the leakage
    alone to empty the tank is (7a/6) hours. -/
theorem leakage_empty_time (a : ℝ) (h : a > 0) :
  let fill_time_with_leakage := 7 * a
  let fill_rate := 1 / a
  let combined_fill_rate := 1 / fill_time_with_leakage
  let leakage_rate := fill_rate - combined_fill_rate
  leakage_rate⁻¹ = 7 * a / 6 :=
by sorry

end NUMINAMATH_CALUDE_leakage_empty_time_l2198_219833


namespace NUMINAMATH_CALUDE_intersection_determines_a_l2198_219855

theorem intersection_determines_a (a : ℝ) : 
  let A : Set ℝ := {1, 2}
  let B : Set ℝ := {a, a^2 + 3}
  A ∩ B = {1} → a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_determines_a_l2198_219855
