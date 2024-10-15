import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_angles_l877_87773

-- Define a right triangle
structure RightTriangle where
  a : ℝ  -- length of one leg
  b : ℝ  -- length of the other leg
  h : ℝ  -- length of the hypotenuse
  right_angle : a^2 + b^2 = h^2  -- Pythagorean theorem

-- Define the quadrilateral formed by the perpendicular bisector
structure Quadrilateral where
  d1 : ℝ  -- length of one diagonal
  d2 : ℝ  -- length of the other diagonal

-- Main theorem
theorem right_triangle_angles (triangle : RightTriangle) (quad : Quadrilateral) :
  quad.d1 / quad.d2 = (1 + Real.sqrt 3) / (2 * Real.sqrt 2) →
  ∃ (angle1 angle2 : ℝ),
    angle1 = 15 * π / 180 ∧
    angle2 = 75 * π / 180 ∧
    angle1 + angle2 = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_angles_l877_87773


namespace NUMINAMATH_CALUDE_cube_roots_sum_power_nine_l877_87766

/-- Given complex numbers x and y as defined, prove that x⁹ + y⁹ ≠ -1 --/
theorem cube_roots_sum_power_nine (x y : ℂ) : 
  x = (-1 + Complex.I * Real.sqrt 3) / 2 →
  y = (-1 - Complex.I * Real.sqrt 3) / 2 →
  x^9 + y^9 ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_cube_roots_sum_power_nine_l877_87766


namespace NUMINAMATH_CALUDE_square_difference_emily_calculation_l877_87779

theorem square_difference (a b : ℕ) : (a - b)^2 = a^2 - 2*a*b + b^2 := by sorry

theorem emily_calculation : 50^2 - 46^2 = 384 := by sorry

end NUMINAMATH_CALUDE_square_difference_emily_calculation_l877_87779


namespace NUMINAMATH_CALUDE_savings_for_engagement_ring_l877_87730

/-- Calculates the monthly savings required to accumulate two months' salary in a given time period. -/
def monthly_savings (annual_salary : ℚ) (months_to_save : ℕ) : ℚ :=
  (2 * annual_salary) / (12 * months_to_save)

/-- Proves that given an annual salary of $60,000 and 10 months to save,
    the amount to save per month to accumulate two months' salary is $1,000. -/
theorem savings_for_engagement_ring :
  monthly_savings 60000 10 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_savings_for_engagement_ring_l877_87730


namespace NUMINAMATH_CALUDE_sum_of_n_values_l877_87733

theorem sum_of_n_values (n : ℤ) : 
  (∃ (S : Finset ℤ), (∀ m ∈ S, (∃ k : ℤ, 24 = k * (2 * m - 1))) ∧ 
   (∀ m : ℤ, (∃ k : ℤ, 24 = k * (2 * m - 1)) → m ∈ S) ∧ 
   (Finset.sum S id = 3)) := by
sorry

end NUMINAMATH_CALUDE_sum_of_n_values_l877_87733


namespace NUMINAMATH_CALUDE_min_income_2020_l877_87728

/-- Represents the per capita income growth over a 40-year period -/
def income_growth (initial : ℝ) (mid : ℝ) (final : ℝ) : Prop :=
  ∃ (x : ℝ), 
    initial * (1 + x)^20 ≥ mid ∧
    initial * (1 + x)^40 ≥ final

/-- Theorem stating the minimum per capita income in 2020 based on 1980 and 2000 data -/
theorem min_income_2020 : income_growth 250 800 2560 := by
  sorry

end NUMINAMATH_CALUDE_min_income_2020_l877_87728


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_special_case_l877_87764

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a circle with radius r -/
structure Circle where
  r : ℝ
  h_positive : 0 < r

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Eccentricity of ellipse under specific conditions -/
theorem ellipse_eccentricity_special_case 
  (E : Ellipse) 
  (O : Circle)
  (B : Point)
  (A : Point)
  (h_circle : O.r = E.a)
  (h_B_on_y : B.x = 0 ∧ B.y = E.a)
  (h_B_on_ellipse : B.x^2 / E.a^2 + B.y^2 / E.b^2 = 1)
  (h_B_on_circle : B.x^2 + B.y^2 = O.r^2)
  (h_A_on_circle : A.x^2 + A.y^2 = O.r^2)
  (h_tangent : ∃ (m : ℝ), (A.y - B.y) = m * (A.x - B.x) ∧ 
               ∀ (x y : ℝ), y = m * (x - B.x) + B.y → x^2 / E.a^2 + y^2 / E.b^2 ≥ 1)
  (h_angle : Real.cos (60 * π / 180) = (A.x * B.x + A.y * B.y) / (O.r^2)) :
  let e := Real.sqrt (E.a^2 - E.b^2) / E.a
  e = Real.sqrt 3 / 3 := by
    sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_special_case_l877_87764


namespace NUMINAMATH_CALUDE_stick_swap_triangle_formation_l877_87770

/-- Represents a set of three stick lengths -/
structure StickSet where
  s₁ : ℝ
  s₂ : ℝ
  s₃ : ℝ
  sum_is_one : s₁ + s₂ + s₃ = 1
  all_positive : 0 < s₁ ∧ 0 < s₂ ∧ 0 < s₃

/-- Checks if a triangle can be formed from the given stick lengths -/
def can_form_triangle (s : StickSet) : Prop :=
  s.s₁ < s.s₂ + s.s₃ ∧ s.s₂ < s.s₁ + s.s₃ ∧ s.s₃ < s.s₁ + s.s₂

theorem stick_swap_triangle_formation 
  (v_initial w_initial : StickSet)
  (v_can_form_initial : can_form_triangle v_initial)
  (w_can_form_initial : can_form_triangle w_initial)
  (v_final w_final : StickSet)
  (swap_occurred : ∃ (i j : Fin 3), 
    v_final.s₁ + v_final.s₂ + v_final.s₃ + w_final.s₁ + w_final.s₂ + w_final.s₃ = 
    v_initial.s₁ + v_initial.s₂ + v_initial.s₃ + w_initial.s₁ + w_initial.s₂ + w_initial.s₃)
  (v_cannot_form_final : ¬can_form_triangle v_final) :
  can_form_triangle w_final :=
sorry

end NUMINAMATH_CALUDE_stick_swap_triangle_formation_l877_87770


namespace NUMINAMATH_CALUDE_factorization_equality_l877_87714

theorem factorization_equality (x : ℝ) : 8*x - 2*x^2 = 2*x*(4 - x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l877_87714


namespace NUMINAMATH_CALUDE_existence_of_xy_l877_87798

theorem existence_of_xy (a b c : ℝ) 
  (h1 : |a| > 2) 
  (h2 : a^2 + b^2 + c^2 = a*b*c + 4) : 
  ∃ x y : ℝ, 
    a = x + 1/x ∧ 
    b = y + 1/y ∧ 
    c = x*y + 1/(x*y) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_xy_l877_87798


namespace NUMINAMATH_CALUDE_cos_30_minus_cos_60_l877_87787

theorem cos_30_minus_cos_60 : Real.cos (π / 6) - Real.cos (π / 3) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_30_minus_cos_60_l877_87787


namespace NUMINAMATH_CALUDE_regular_octagon_area_l877_87722

/-- The area of a regular octagon with side length 2√2 is 16 + 16√2 -/
theorem regular_octagon_area : 
  let s : ℝ := 2 * Real.sqrt 2
  8 * (s^2 / (4 * Real.tan (π/8))) = 16 + 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_area_l877_87722


namespace NUMINAMATH_CALUDE_coin_weighing_possible_l877_87797

/-- Represents a coin with a weight -/
structure Coin where
  weight : ℕ

/-- Represents the result of a weighing operation -/
inductive WeighResult
  | Equal
  | LeftHeavier
  | RightHeavier

/-- Represents a 2-pan scale that can compare two sets of coins -/
def Scale : (List Coin) → (List Coin) → WeighResult := sorry

/-- The main theorem to be proven -/
theorem coin_weighing_possible (k : ℕ) : 
  ∀ (coins : List Coin), 
    coins.length = 2^k → 
    (∀ a b c : Coin, a ∈ coins → b ∈ coins → c ∈ coins → 
      (a.weight ≠ b.weight ∧ b.weight ≠ c.weight) → a.weight = c.weight) →
    ∃ (measurements : List (List Coin × List Coin)),
      measurements.length ≤ k ∧
      (∀ m ∈ measurements, m.1.length + m.2.length ≤ coins.length) ∧
      (∃ (heavy light : Coin), heavy ∈ coins ∧ light ∈ coins ∧ heavy.weight > light.weight) ∨
      (∀ c1 c2 : Coin, c1 ∈ coins → c2 ∈ coins → c1.weight = c2.weight) := by
  sorry

end NUMINAMATH_CALUDE_coin_weighing_possible_l877_87797


namespace NUMINAMATH_CALUDE_negation_of_exists_ellipse_eccentricity_lt_one_l877_87721

/-- An ellipse is a geometric shape with an eccentricity. -/
structure Ellipse where
  eccentricity : ℝ

/-- The negation of "There exists an ellipse with an eccentricity e < 1" 
    is equivalent to "The eccentricity e ≥ 1 for any ellipse". -/
theorem negation_of_exists_ellipse_eccentricity_lt_one :
  (¬ ∃ (e : Ellipse), e.eccentricity < 1) ↔ (∀ (e : Ellipse), e.eccentricity ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_negation_of_exists_ellipse_eccentricity_lt_one_l877_87721


namespace NUMINAMATH_CALUDE_exists_in_set_A_l877_87723

/-- Sum of digits of a positive integer -/
def digit_sum (n : ℕ+) : ℕ := sorry

/-- Predicate for a number having all non-zero digits -/
def all_digits_nonzero (n : ℕ+) : Prop := sorry

/-- Number of digits in a positive integer -/
def num_digits (n : ℕ+) : ℕ := sorry

/-- The main theorem -/
theorem exists_in_set_A (k : ℕ+) : 
  ∃ x : ℕ+, num_digits x = k ∧ all_digits_nonzero x ∧ (digit_sum x ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_exists_in_set_A_l877_87723


namespace NUMINAMATH_CALUDE_committee_formation_count_l877_87768

/-- Represents a department in the science division -/
inductive Department : Type
| Biology : Department
| Physics : Department
| Chemistry : Department
| Mathematics : Department

/-- Represents the gender of a professor -/
inductive Gender : Type
| Male : Gender
| Female : Gender

/-- Represents a professor with their department and gender -/
structure Professor :=
  (dept : Department)
  (gender : Gender)

/-- The total number of departments -/
def total_departments : Nat := 4

/-- The number of male professors in each department -/
def male_professors_per_dept : Nat := 3

/-- The number of female professors in each department -/
def female_professors_per_dept : Nat := 3

/-- The total number of professors in the committee -/
def committee_size : Nat := 8

/-- The number of male professors required in the committee -/
def required_males : Nat := 4

/-- The number of female professors required in the committee -/
def required_females : Nat := 4

/-- The number of departments that should contribute exactly 2 professors -/
def depts_with_two_profs : Nat := 2

/-- The minimum number of professors required from the Mathematics department -/
def min_math_profs : Nat := 2

/-- Calculates the number of ways to form the committee -/
def count_committee_formations : Nat :=
  sorry

/-- Theorem stating that the number of ways to form the committee is 1944 -/
theorem committee_formation_count :
  count_committee_formations = 1944 :=
sorry

end NUMINAMATH_CALUDE_committee_formation_count_l877_87768


namespace NUMINAMATH_CALUDE_intersection_M_N_l877_87716

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {-2, -1, 1, 2}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l877_87716


namespace NUMINAMATH_CALUDE_find_x_l877_87704

def binary_op (n : ℤ) (x : ℚ) : ℚ := n - (n * x)

theorem find_x : 
  (∀ n : ℤ, n > 3 → binary_op n x ≥ 14) ∧
  (binary_op 3 x < 14) →
  x = -3 := by sorry

end NUMINAMATH_CALUDE_find_x_l877_87704


namespace NUMINAMATH_CALUDE_can_capacity_proof_l877_87788

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- The capacity of the can in liters -/
def canCapacity : ℝ := 36

/-- The amount of milk added in liters -/
def milkAdded : ℝ := 8

theorem can_capacity_proof (initial : CanContents) (final : CanContents) :
  -- Initial ratio of milk to water is 4:3
  initial.milk / initial.water = 4 / 3 →
  -- Final contents after adding milk
  final.milk = initial.milk + milkAdded ∧
  final.water = initial.water →
  -- Can is full after adding milk
  final.milk + final.water = canCapacity →
  -- Final ratio of milk to water is 2:1
  final.milk / final.water = 2 / 1 →
  -- Prove that the capacity of the can is 36 liters
  canCapacity = 36 := by
  sorry


end NUMINAMATH_CALUDE_can_capacity_proof_l877_87788


namespace NUMINAMATH_CALUDE_peanuts_in_box_l877_87793

/-- Calculate the final number of peanuts in a box after removing and adding some. -/
def final_peanuts (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Theorem stating that for the given values, the final number of peanuts is 13. -/
theorem peanuts_in_box : final_peanuts 4 3 12 = 13 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l877_87793


namespace NUMINAMATH_CALUDE_minimum_implies_a_range_l877_87700

/-- A function f with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + a

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

/-- Theorem stating that if f has a minimum in (0,2), then a is in (0,4) -/
theorem minimum_implies_a_range (a : ℝ) :
  (∃ x₀ ∈ Set.Ioo 0 2, ∀ x ∈ Set.Ioo 0 2, f a x₀ ≤ f a x) →
  a ∈ Set.Ioo 0 4 :=
sorry

end NUMINAMATH_CALUDE_minimum_implies_a_range_l877_87700


namespace NUMINAMATH_CALUDE_investment_quoted_price_l877_87783

/-- Calculates the quoted price of shares given investment details -/
def quoted_price (total_investment : ℚ) (nominal_value : ℚ) (dividend_rate : ℚ) (annual_income : ℚ) : ℚ :=
  let dividend_per_share := (dividend_rate / 100) * nominal_value
  let number_of_shares := annual_income / dividend_per_share
  total_investment / number_of_shares

/-- Theorem stating that given the investment details, the quoted price is 9.5 -/
theorem investment_quoted_price :
  quoted_price 4940 10 14 728 = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_investment_quoted_price_l877_87783


namespace NUMINAMATH_CALUDE_triangle_translation_inconsistency_l877_87756

structure Point where
  x : ℝ
  y : ℝ

def Triangle (A B C : Point) := Unit

def isTranslation (A B C A' B' C' : Point) : Prop :=
  ∃ dx dy : ℝ, 
    A'.x = A.x + dx ∧ A'.y = A.y + dy ∧
    B'.x = B.x + dx ∧ B'.y = B.y + dy ∧
    C'.x = C.x + dx ∧ C'.y = C.y + dy

def correctYCoordinates (A B C A' B' C' : Point) : Prop :=
  A'.y = A.y - 3 ∧ B'.y = B.y - 3 ∧ C'.y = C.y - 3

def oneCorrectXCoordinate (A B C A' B' C' : Point) : Prop :=
  (A'.x = A.x ∧ B'.x ≠ B.x - 1 ∧ C'.x ≠ C.x + 1) ∨
  (A'.x ≠ A.x ∧ B'.x = B.x - 1 ∧ C'.x ≠ C.x + 1) ∨
  (A'.x ≠ A.x ∧ B'.x ≠ B.x - 1 ∧ C'.x = C.x + 1)

theorem triangle_translation_inconsistency 
  (A B C A' B' C' : Point)
  (h1 : A = ⟨0, 3⟩)
  (h2 : B = ⟨-1, 0⟩)
  (h3 : C = ⟨1, 0⟩)
  (h4 : A' = ⟨0, 0⟩)
  (h5 : B' = ⟨-2, -3⟩)
  (h6 : C' = ⟨2, -3⟩)
  (h7 : correctYCoordinates A B C A' B' C')
  (h8 : oneCorrectXCoordinate A B C A' B' C') :
  ¬(isTranslation A B C A' B' C') ∧
  ((A' = ⟨0, 0⟩ ∧ B' = ⟨-1, -3⟩ ∧ C' = ⟨1, -3⟩) ∨
   (A' = ⟨-1, 0⟩ ∧ B' = ⟨-2, -3⟩ ∧ C' = ⟨0, -3⟩) ∨
   (A' = ⟨1, 0⟩ ∧ B' = ⟨0, -3⟩ ∧ C' = ⟨2, -3⟩)) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_translation_inconsistency_l877_87756


namespace NUMINAMATH_CALUDE_inequality_proofs_l877_87753

def M : Set ℝ := {x | x ≥ 2}

theorem inequality_proofs :
  (∀ (a b c d : ℝ), a ∈ M → b ∈ M → c ∈ M → d ∈ M →
    Real.sqrt (a * b) + Real.sqrt (c * d) ≥ Real.sqrt (a + b) + Real.sqrt (c + d)) ∧
  (∀ (a b c d : ℝ), a ∈ M → b ∈ M → c ∈ M → d ∈ M →
    Real.sqrt (a * b) + Real.sqrt (c * d) ≥ Real.sqrt (a + c) + Real.sqrt (b + d)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l877_87753


namespace NUMINAMATH_CALUDE_matt_card_trade_profit_l877_87750

def matt_card_value : ℕ := 6
def jane_card1_value : ℕ := 2
def jane_card2_value : ℕ := 9
def matt_cards_traded : ℕ := 2
def jane_cards1_received : ℕ := 3
def jane_cards2_received : ℕ := 1
def profit : ℕ := 3

theorem matt_card_trade_profit :
  (jane_cards1_received * jane_card1_value + jane_cards2_received * jane_card2_value) -
  (matt_cards_traded * matt_card_value) = profit := by
  sorry

end NUMINAMATH_CALUDE_matt_card_trade_profit_l877_87750


namespace NUMINAMATH_CALUDE_principle_countable_noun_meaning_l877_87748

/-- Define a type for English words -/
def EnglishWord : Type := String

/-- Define a type for word meanings -/
def WordMeaning : Type := String

/-- Function to get the meaning of a word when used as a countable noun -/
def countableNounMeaning (word : EnglishWord) : WordMeaning :=
  sorry

/-- Theorem stating that "principle" as a countable noun means "principle, criterion" -/
theorem principle_countable_noun_meaning :
  countableNounMeaning "principle" = "principle, criterion" :=
sorry

end NUMINAMATH_CALUDE_principle_countable_noun_meaning_l877_87748


namespace NUMINAMATH_CALUDE_bake_sale_group_l877_87765

theorem bake_sale_group (p : ℕ) : 
  p > 0 → 
  (p : ℚ) / 2 - 2 = (2 * p : ℚ) / 5 → 
  (p : ℚ) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_group_l877_87765


namespace NUMINAMATH_CALUDE_scientific_notation_eleven_million_l877_87796

theorem scientific_notation_eleven_million :
  (11000000 : ℝ) = 1.1 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_eleven_million_l877_87796


namespace NUMINAMATH_CALUDE_condition_for_cubic_equation_l877_87703

theorem condition_for_cubic_equation (a b : ℝ) (h : a * b ≠ 0) :
  (a - b = 1) ↔ (a^3 - b^3 - a*b - a^2 - b^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_for_cubic_equation_l877_87703


namespace NUMINAMATH_CALUDE_smallest_pencil_collection_l877_87791

theorem smallest_pencil_collection (P : ℕ) : 
  P > 2 ∧ 
  P % 5 = 2 ∧ 
  P % 9 = 2 ∧ 
  P % 11 = 2 → 
  (∀ Q : ℕ, Q > 2 ∧ Q % 5 = 2 ∧ Q % 9 = 2 ∧ Q % 11 = 2 → P ≤ Q) →
  P = 497 := by
sorry

end NUMINAMATH_CALUDE_smallest_pencil_collection_l877_87791


namespace NUMINAMATH_CALUDE_ellipse_k_range_l877_87705

/-- Given an ellipse with equation x^2 / (3-k) + y^2 / (1+k) = 1 and foci on the x-axis,
    the range of k values is (-1, 1) -/
theorem ellipse_k_range (k : ℝ) :
  (∀ x y : ℝ, x^2 / (3-k) + y^2 / (1+k) = 1) →
  (∃ c : ℝ, c > 0 ∧ c^2 = (3-k) - (1+k)) →
  -1 < k ∧ k < 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l877_87705


namespace NUMINAMATH_CALUDE_tan_15_degrees_l877_87707

theorem tan_15_degrees : Real.tan (15 * π / 180) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_15_degrees_l877_87707


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l877_87799

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

-- Define the roots of the quadratic equation
def roots (a b : ℝ) : Prop := quadratic_eq a ∧ quadratic_eq b ∧ a ≠ b

-- Define the linear function
def linear_function (x : ℝ) (a b : ℝ) : ℝ := (a*b - 1)*x + a + b

-- Theorem: The linear function does not pass through the third quadrant
theorem linear_function_not_in_third_quadrant (a b : ℝ) :
  roots a b →
  ∀ x y : ℝ, y = linear_function x a b →
  ¬(x < 0 ∧ y < 0) :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l877_87799


namespace NUMINAMATH_CALUDE_not_perfect_square_l877_87718

theorem not_perfect_square (t : ℤ) : ¬ ∃ k : ℤ, 7 * t + 3 = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l877_87718


namespace NUMINAMATH_CALUDE_problem_statement_l877_87776

theorem problem_statement (n : ℕ+) : 
  3 * (Nat.choose (n - 1) (n - 5)) = 5 * (Nat.factorial (n - 2) / Nat.factorial (n - 4)) → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l877_87776


namespace NUMINAMATH_CALUDE_set_operations_l877_87715

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {1, 3, 5, 7}
def B : Set Nat := {3, 5}

theorem set_operations :
  (A ∪ B = {1, 3, 5, 7}) ∧
  ((U \ A) ∪ B = {2, 3, 4, 5, 6}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l877_87715


namespace NUMINAMATH_CALUDE_tire_usage_proof_l877_87769

-- Define the number of tires
def total_tires : ℕ := 6

-- Define the total miles traveled by the car
def total_miles : ℕ := 45000

-- Define the number of tires used at any given time
def tires_in_use : ℕ := 4

-- Define the function to calculate miles per tire
def miles_per_tire (total_tires : ℕ) (total_miles : ℕ) (tires_in_use : ℕ) : ℕ :=
  (total_miles * tires_in_use) / total_tires

-- Theorem statement
theorem tire_usage_proof :
  miles_per_tire total_tires total_miles tires_in_use = 30000 := by
  sorry

end NUMINAMATH_CALUDE_tire_usage_proof_l877_87769


namespace NUMINAMATH_CALUDE_complementary_event_correct_l877_87763

/-- A bag containing red and white balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- An event in the sample space of drawing balls -/
inductive Event
  | AtLeastOneWhite
  | AllRed

/-- The complementary event function -/
def complementary_event : Event → Event
  | Event.AtLeastOneWhite => Event.AllRed
  | Event.AllRed => Event.AtLeastOneWhite

theorem complementary_event_correct (bag : Bag) (h1 : bag.red = 3) (h2 : bag.white = 2) :
  complementary_event Event.AtLeastOneWhite = Event.AllRed :=
by sorry

end NUMINAMATH_CALUDE_complementary_event_correct_l877_87763


namespace NUMINAMATH_CALUDE_cosine_sum_theorem_l877_87702

theorem cosine_sum_theorem : 
  Real.cos 0 ^ 4 + Real.cos (π / 6) ^ 4 + Real.cos (π / 3) ^ 4 + Real.cos (π / 2) ^ 4 + 
  Real.cos (2 * π / 3) ^ 4 + Real.cos (5 * π / 6) ^ 4 + Real.cos π ^ 4 = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_theorem_l877_87702


namespace NUMINAMATH_CALUDE_mean_score_all_students_l877_87780

theorem mean_score_all_students
  (score_first : ℝ)
  (score_second : ℝ)
  (ratio_first_to_second : ℚ)
  (h1 : score_first = 90)
  (h2 : score_second = 75)
  (h3 : ratio_first_to_second = 2 / 3) :
  let total_students := (ratio_first_to_second + 1) * students_second
  let total_score := score_first * ratio_first_to_second * students_second + score_second * students_second
  total_score / total_students = 81 :=
by sorry

end NUMINAMATH_CALUDE_mean_score_all_students_l877_87780


namespace NUMINAMATH_CALUDE_triangle_side_range_l877_87754

theorem triangle_side_range (A B C : ℝ) (AB BC : ℝ) :
  AB = Real.sqrt 3 →
  C = π / 3 →
  (∃ (A₁ A₂ : ℝ), A₁ ≠ A₂ ∧ 
    Real.sin A₁ = BC / 2 ∧ 
    Real.sin A₂ = BC / 2 ∧ 
    A₁ ∈ Set.Ioo (π / 3) (2 * π / 3) ∧ 
    A₂ ∈ Set.Ioo (π / 3) (2 * π / 3)) →
  BC > Real.sqrt 3 ∧ BC < 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_range_l877_87754


namespace NUMINAMATH_CALUDE_brown_eyes_fraction_l877_87774

theorem brown_eyes_fraction (total_students : ℕ) 
  (brown_eyes_black_hair : ℕ) 
  (h1 : total_students = 18) 
  (h2 : brown_eyes_black_hair = 6) 
  (h3 : brown_eyes_black_hair * 2 = brown_eyes_black_hair + brown_eyes_black_hair) :
  (brown_eyes_black_hair * 2 : ℚ) / total_students = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_brown_eyes_fraction_l877_87774


namespace NUMINAMATH_CALUDE_p_false_q_true_l877_87790

theorem p_false_q_true (h1 : p ∨ q) (h2 : ¬p) : ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_p_false_q_true_l877_87790


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l877_87741

theorem geometric_progression_fourth_term :
  ∀ x : ℝ,
  (∃ r : ℝ, r ≠ 0 ∧ (2*x + 2) = r * x ∧ (3*x + 3) = r * (2*x + 2)) →
  ∃ fourth_term : ℝ, fourth_term = -13/2 ∧ (3*x + 3) * r = fourth_term :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l877_87741


namespace NUMINAMATH_CALUDE_odd_sum_probability_redesigned_board_l877_87757

/-- Represents the redesigned dartboard -/
structure Dartboard where
  outer_radius : ℝ
  inner_radius : ℝ
  inner_points : Fin 3 → ℕ
  outer_points : Fin 3 → ℕ

/-- The probability of getting an odd sum when throwing two darts -/
def odd_sum_probability (d : Dartboard) : ℝ :=
  sorry

/-- The redesigned dartboard as described in the problem -/
def redesigned_board : Dartboard :=
  { outer_radius := 8
    inner_radius := 4
    inner_points := λ i => if i = 0 then 3 else 1
    outer_points := λ i => if i = 0 then 2 else 3 }

/-- Theorem stating the probability of an odd sum on the redesigned board -/
theorem odd_sum_probability_redesigned_board :
    odd_sum_probability redesigned_board = 4 / 9 :=
  sorry

end NUMINAMATH_CALUDE_odd_sum_probability_redesigned_board_l877_87757


namespace NUMINAMATH_CALUDE_base_seven_528_l877_87755

def base_seven_representation (n : ℕ) : List ℕ :=
  sorry

theorem base_seven_528 :
  base_seven_representation 528 = [1, 3, 5, 3] := by
  sorry

end NUMINAMATH_CALUDE_base_seven_528_l877_87755


namespace NUMINAMATH_CALUDE_lila_tulips_l877_87747

/-- Calculates the number of tulips after maintaining the ratio --/
def final_tulips (initial_orchids : ℕ) (added_orchids : ℕ) (tulip_ratio : ℕ) (orchid_ratio : ℕ) : ℕ :=
  let final_orchids := initial_orchids + added_orchids
  let groups := final_orchids / orchid_ratio
  tulip_ratio * groups

/-- Proves that Lila will have 21 tulips after maintaining the ratio --/
theorem lila_tulips : 
  final_tulips 16 12 3 4 = 21 := by
  sorry

#eval final_tulips 16 12 3 4

end NUMINAMATH_CALUDE_lila_tulips_l877_87747


namespace NUMINAMATH_CALUDE_length_of_A_l877_87719

def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 7)

def on_line_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

def intersect_at (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    (p.1 + t * (r.1 - p.1) = q.1) ∧
    (p.2 + t * (r.2 - p.2) = q.2)

theorem length_of_A'B' :
  ∃ A' B' : ℝ × ℝ, 
    on_line_y_eq_x A' ∧
    on_line_y_eq_x B' ∧
    intersect_at A A' C ∧
    intersect_at B B' C ∧
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 10 * Real.sqrt 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_length_of_A_l877_87719


namespace NUMINAMATH_CALUDE_planes_parallel_if_line_perpendicular_to_both_l877_87717

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_line_perpendicular_to_both 
  (a : Line) (α β : Plane) (h1 : α ≠ β) :
  perp a α → perp a β → para α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_line_perpendicular_to_both_l877_87717


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_l877_87710

def is_divisible_by_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0 → n % d = 0

theorem largest_three_digit_divisible_by_digits :
  ∀ n : ℕ, 800 ≤ n → n ≤ 899 → is_divisible_by_digits n →
  n ≤ 864 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_l877_87710


namespace NUMINAMATH_CALUDE_computer_store_discount_rate_l877_87781

/-- Proves that the discount rate of the second store is approximately 0.87% given the conditions of the problem -/
theorem computer_store_discount_rate (price1 : ℝ) (discount1 : ℝ) (price2 : ℝ) (price_diff : ℝ) :
  price1 = 950 →
  discount1 = 0.06 →
  price2 = 920 →
  price_diff = 19 →
  let discounted_price1 := price1 * (1 - discount1)
  let discounted_price2 := discounted_price1 + price_diff
  let discount2 := (price2 - discounted_price2) / price2
  ∃ ε > 0, |discount2 - 0.0087| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_computer_store_discount_rate_l877_87781


namespace NUMINAMATH_CALUDE_arccos_one_half_l877_87743

theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := by sorry

end NUMINAMATH_CALUDE_arccos_one_half_l877_87743


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_390_l877_87771

theorem sin_n_equals_cos_390 (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) :
  Real.sin (n * Real.pi / 180) = Real.cos (390 * Real.pi / 180) → n = 60 := by
sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_390_l877_87771


namespace NUMINAMATH_CALUDE_amusement_park_cost_per_trip_l877_87782

/-- Calculates the cost per trip to an amusement park given the number of sons,
    cost per pass, and number of trips by each son. -/
def cost_per_trip (num_sons : ℕ) (cost_per_pass : ℚ) (trips_oldest : ℕ) (trips_youngest : ℕ) : ℚ :=
  (num_sons * cost_per_pass) / (trips_oldest + trips_youngest)

/-- Theorem stating that for the given inputs, the cost per trip is $4.00 -/
theorem amusement_park_cost_per_trip :
  cost_per_trip 2 100 35 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_cost_per_trip_l877_87782


namespace NUMINAMATH_CALUDE_odd_power_sum_is_prime_power_l877_87726

theorem odd_power_sum_is_prime_power (n p x y k : ℕ) :
  Odd n →
  n > 1 →
  Prime p →
  Odd p →
  x^n + y^n = p^k →
  ∃ m : ℕ, n = p^m :=
by sorry

end NUMINAMATH_CALUDE_odd_power_sum_is_prime_power_l877_87726


namespace NUMINAMATH_CALUDE_negation_equivalence_l877_87760

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2*x = 0) ↔ (∀ x : ℝ, x^2 - 2*x ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l877_87760


namespace NUMINAMATH_CALUDE_set_difference_equals_open_interval_l877_87736

/-- The set A of real numbers x such that |4x - 1| > 9 -/
def A : Set ℝ := {x | |4*x - 1| > 9}

/-- The set B of non-negative real numbers -/
def B : Set ℝ := {x | x ≥ 0}

/-- The open interval (5/2, +∞) -/
def openInterval : Set ℝ := {x | x > 5/2}

/-- Theorem stating that the set difference A - B is equal to the open interval (5/2, +∞) -/
theorem set_difference_equals_open_interval : A \ B = openInterval := by
  sorry

end NUMINAMATH_CALUDE_set_difference_equals_open_interval_l877_87736


namespace NUMINAMATH_CALUDE_sum_three_digit_even_integers_eq_247050_l877_87712

/-- The sum of all three-digit positive even integers -/
def sum_three_digit_even_integers : ℕ :=
  let first : ℕ := 100  -- First three-digit even integer
  let last : ℕ := 998   -- Last three-digit even integer
  let count : ℕ := (last - first) / 2 + 1  -- Number of terms
  count * (first + last) / 2

/-- Theorem stating that the sum of all three-digit positive even integers is 247050 -/
theorem sum_three_digit_even_integers_eq_247050 :
  sum_three_digit_even_integers = 247050 := by
  sorry

#eval sum_three_digit_even_integers

end NUMINAMATH_CALUDE_sum_three_digit_even_integers_eq_247050_l877_87712


namespace NUMINAMATH_CALUDE_six_cows_satisfy_condition_unique_cow_count_l877_87739

/-- Represents the farm with cows and chickens -/
structure Farm where
  cows : ℕ
  chickens : ℕ

/-- The total number of legs on the farm -/
def totalLegs (f : Farm) : ℕ := 5 * f.cows + 2 * f.chickens

/-- The total number of heads on the farm -/
def totalHeads (f : Farm) : ℕ := f.cows + f.chickens

/-- The farm satisfies the given condition -/
def satisfiesCondition (f : Farm) : Prop :=
  totalLegs f = 20 + 2 * totalHeads f

/-- Theorem stating that the farm with 6 cows satisfies the condition -/
theorem six_cows_satisfy_condition :
  ∃ (f : Farm), f.cows = 6 ∧ satisfiesCondition f :=
sorry

/-- Theorem stating that 6 is the only number of cows that satisfies the condition -/
theorem unique_cow_count :
  ∀ (f : Farm), satisfiesCondition f → f.cows = 6 :=
sorry

end NUMINAMATH_CALUDE_six_cows_satisfy_condition_unique_cow_count_l877_87739


namespace NUMINAMATH_CALUDE_student_count_is_35_l877_87746

/-- The number of different Roman numerals -/
def num_roman_numerals : ℕ := 7

/-- The number of sketches for each Roman numeral -/
def sketches_per_numeral : ℕ := 5

/-- The total number of students in the class -/
def num_students : ℕ := num_roman_numerals * sketches_per_numeral

theorem student_count_is_35 : num_students = 35 := by
  sorry

end NUMINAMATH_CALUDE_student_count_is_35_l877_87746


namespace NUMINAMATH_CALUDE_first_term_of_constant_ratio_l877_87731

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a + (n - 1 : ℚ) * d) / 2

theorem first_term_of_constant_ratio (d : ℚ) (c : ℚ) :
  d = 5 →
  (∀ n : ℕ, n > 0 → 
    (arithmetic_sum a d (2*n)) / (arithmetic_sum a d n) = c) →
  a = 5/2 :=
sorry

end NUMINAMATH_CALUDE_first_term_of_constant_ratio_l877_87731


namespace NUMINAMATH_CALUDE_northern_car_speed_l877_87749

/-- Proves that given the initial conditions of two cars and their movement,
    the speed of the northern car must be 80 mph. -/
theorem northern_car_speed 
  (initial_distance : ℝ) 
  (southern_speed : ℝ) 
  (time : ℝ) 
  (final_distance : ℝ) 
  (h1 : initial_distance = 300) 
  (h2 : southern_speed = 60) 
  (h3 : time = 5) 
  (h4 : final_distance = 500) : 
  ∃ v : ℝ, v = 80 ∧ 
  final_distance^2 = initial_distance^2 + (time * v)^2 + (time * southern_speed)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_northern_car_speed_l877_87749


namespace NUMINAMATH_CALUDE_intersecting_digit_is_three_l877_87720

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def powers_of_three : Set ℕ := {n | ∃ m : ℕ, n = 3^m ∧ is_three_digit n}
def powers_of_seven : Set ℕ := {n | ∃ m : ℕ, n = 7^m ∧ is_three_digit n}

theorem intersecting_digit_is_three :
  ∃! d : ℕ, d < 10 ∧ 
  (∃ n ∈ powers_of_three, ∃ i : ℕ, n / 10^i % 10 = d) ∧
  (∃ n ∈ powers_of_seven, ∃ i : ℕ, n / 10^i % 10 = d) :=
by sorry

end NUMINAMATH_CALUDE_intersecting_digit_is_three_l877_87720


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l877_87744

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 51,
    the 8th term is 79. -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a) 
    (h_4th : a 4 = 23)
    (h_6th : a 6 = 51) : 
    a 8 = 79 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l877_87744


namespace NUMINAMATH_CALUDE_group_dinner_cost_l877_87794

/-- Calculate the total cost for a group dinner including service charge -/
theorem group_dinner_cost (num_people : ℕ) (meal_cost drink_cost dessert_cost : ℚ) 
  (service_charge_rate : ℚ) : 
  num_people = 5 →
  meal_cost = 12 →
  drink_cost = 3 →
  dessert_cost = 5 →
  service_charge_rate = 1/10 →
  (num_people * (meal_cost + drink_cost + dessert_cost)) * (1 + service_charge_rate) = 110 := by
  sorry

end NUMINAMATH_CALUDE_group_dinner_cost_l877_87794


namespace NUMINAMATH_CALUDE_circle_diameter_l877_87727

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l877_87727


namespace NUMINAMATH_CALUDE_grace_weeding_hours_l877_87785

/-- Represents Grace's landscaping business earnings in September --/
def graces_earnings (mowing_rate : ℕ) (weeding_rate : ℕ) (mulching_rate : ℕ)
                    (mowing_hours : ℕ) (weeding_hours : ℕ) (mulching_hours : ℕ) : ℕ :=
  mowing_rate * mowing_hours + weeding_rate * weeding_hours + mulching_rate * mulching_hours

/-- Theorem stating that Grace spent 9 hours pulling weeds in September --/
theorem grace_weeding_hours :
  ∀ (mowing_rate weeding_rate mulching_rate mowing_hours mulching_hours total_earnings : ℕ),
    mowing_rate = 6 →
    weeding_rate = 11 →
    mulching_rate = 9 →
    mowing_hours = 63 →
    mulching_hours = 10 →
    total_earnings = 567 →
    ∃ (weeding_hours : ℕ),
      graces_earnings mowing_rate weeding_rate mulching_rate mowing_hours weeding_hours mulching_hours = total_earnings ∧
      weeding_hours = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_grace_weeding_hours_l877_87785


namespace NUMINAMATH_CALUDE_part_one_part_two_l877_87795

-- Define the conditions
def p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0
def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 5

-- Part 1
theorem part_one (x : ℝ) : 
  p x 1 → q x → x ∈ Set.Ioo 2 4 :=
sorry

-- Part 2
theorem part_two (a : ℝ) : 
  a > 0 → 
  (Set.Ioo 2 5 ⊂ {x | p x a}) → 
  ({x | p x a} ≠ Set.Ioo 2 5) → 
  a ∈ Set.Ioc (5/4) 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l877_87795


namespace NUMINAMATH_CALUDE_even_digits_base7_528_l877_87775

/-- Converts a natural number from base 10 to base 7 --/
def toBase7 (n : ℕ) : List ℕ := sorry

/-- Counts the number of even digits in a list of natural numbers --/
def countEvenDigits (digits : List ℕ) : ℕ := sorry

/-- The number of even digits in the base-7 representation of 528 is 0 --/
theorem even_digits_base7_528 :
  countEvenDigits (toBase7 528) = 0 := by sorry

end NUMINAMATH_CALUDE_even_digits_base7_528_l877_87775


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l877_87706

/-- 
Given:
- The selling price of 13 balls is 720 Rs.
- The loss incurred is equal to the cost price of 5 balls.

Prove that the cost price of one ball is 90 Rs.
-/
theorem cost_price_of_ball (selling_price : ℕ) (num_balls : ℕ) (loss_balls : ℕ) :
  selling_price = 720 →
  num_balls = 13 →
  loss_balls = 5 →
  ∃ (cost_price : ℕ), 
    cost_price * num_balls - selling_price = cost_price * loss_balls ∧
    cost_price = 90 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l877_87706


namespace NUMINAMATH_CALUDE_shm_first_return_time_l877_87777

/-- Time for a particle in Simple Harmonic Motion to first return to origin -/
theorem shm_first_return_time (m k : ℝ) (hm : m > 0) (hk : k > 0) :
  ∃ (t : ℝ), t = π * Real.sqrt (m / k) ∧ t > 0 := by
  sorry

end NUMINAMATH_CALUDE_shm_first_return_time_l877_87777


namespace NUMINAMATH_CALUDE_same_terminal_side_l877_87735

theorem same_terminal_side (θ : ℝ) : 
  ∃ k : ℤ, θ + 360 * k = 330 → θ = -30 := by sorry

end NUMINAMATH_CALUDE_same_terminal_side_l877_87735


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l877_87740

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

theorem tenth_term_of_specific_sequence :
  geometric_sequence 5 (3/4) 10 = 98415/262144 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l877_87740


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l877_87786

/-- Given a quadratic equation (m-3)x^2 + 4x + 1 = 0 with real solutions,
    the range of values for m is m ≤ 7 and m ≠ 3 -/
theorem quadratic_equation_range (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l877_87786


namespace NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l877_87789

theorem sqrt_product_equals_sqrt_of_product :
  Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l877_87789


namespace NUMINAMATH_CALUDE_fourth_root_81_times_cube_root_27_times_sqrt_16_l877_87732

theorem fourth_root_81_times_cube_root_27_times_sqrt_16 : 
  (81 : ℝ) ^ (1/4 : ℝ) * (27 : ℝ) ^ (1/3 : ℝ) * (16 : ℝ) ^ (1/2 : ℝ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_81_times_cube_root_27_times_sqrt_16_l877_87732


namespace NUMINAMATH_CALUDE_f_2015_value_l877_87737

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2015_value (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_periodic : ∀ x, f (x + 4) = f x)
  (h_f_1 : f 1 = 2) : 
  f 2015 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_2015_value_l877_87737


namespace NUMINAMATH_CALUDE_parabola_properties_l877_87792

/-- Parabola properties -/
structure Parabola where
  a : ℝ
  x₁ : ℝ
  x₂ : ℝ
  y : ℝ → ℝ
  h₁ : a ≠ 0
  h₂ : x₁ ≠ x₂
  h₃ : ∀ x, y x = x^2 + (1 - 2*a)*x + a^2
  h₄ : y x₁ = 0
  h₅ : y x₂ = 0

/-- Main theorem about the parabola -/
theorem parabola_properties (p : Parabola) :
  (0 < p.a ∧ p.a < 1/4 ∧ p.x₁ < 0 ∧ p.x₂ < 0) ∧
  (p.y 0 - 2 = -p.x₁ - p.x₂ → p.a = -3) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l877_87792


namespace NUMINAMATH_CALUDE_unique_color_for_X_l877_87734

/-- Represents the four colors used in the grid --/
inductive Color
| Red
| Blue
| Yellow
| Green

/-- Represents a position in the grid --/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Represents the grid --/
def Grid := Position → Option Color

/-- Checks if two positions are adjacent (share a vertex) --/
def adjacent (p1 p2 : Position) : Prop :=
  (p1.x = p2.x ∧ (p1.y = p2.y + 1 ∨ p1.y = p2.y - 1)) ∨
  (p1.y = p2.y ∧ (p1.x = p2.x + 1 ∨ p1.x = p2.x - 1)) ∨
  (p1.x = p2.x + 1 ∧ p1.y = p2.y + 1) ∨
  (p1.x = p2.x - 1 ∧ p1.y = p2.y - 1) ∨
  (p1.x = p2.x + 1 ∧ p1.y = p2.y - 1) ∨
  (p1.x = p2.x - 1 ∧ p1.y = p2.y + 1)

/-- Checks if the grid coloring is valid --/
def valid_coloring (g : Grid) : Prop :=
  ∀ p1 p2 : Position, adjacent p1 p2 →
    (g p1).isSome ∧ (g p2).isSome →
    (g p1 ≠ g p2)

/-- The position of cell X --/
def X : Position := ⟨5, 5⟩

/-- Theorem: There exists a unique color for cell X in a valid 4-color grid --/
theorem unique_color_for_X (g : Grid) (h : valid_coloring g) :
  ∃! c : Color, g X = some c :=
sorry

end NUMINAMATH_CALUDE_unique_color_for_X_l877_87734


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_64_l877_87751

theorem arithmetic_square_root_of_64 : Real.sqrt 64 = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_64_l877_87751


namespace NUMINAMATH_CALUDE_ratio_equality_l877_87713

theorem ratio_equality (x y z : ℝ) (h : x / 3 = y / 4 ∧ y / 4 = z / 5) :
  (2 * x + y - z) / (3 * x - 2 * y + z) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l877_87713


namespace NUMINAMATH_CALUDE_xyz_equality_l877_87778

theorem xyz_equality (x y z : ℝ) (h : x * y * z = x + y + z) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z :=
by sorry

end NUMINAMATH_CALUDE_xyz_equality_l877_87778


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_46_l877_87767

/-- The coefficient of x^2 in the expansion of (2x^3 + 4x^2 - 3x + 5)(3x^2 - 9x + 1) -/
def coefficient_x_squared : ℤ :=
  let p1 := [2, 4, -3, 5]  -- Coefficients of 2x^3 + 4x^2 - 3x + 5
  let p2 := [3, -9, 1]     -- Coefficients of 3x^2 - 9x + 1
  46

/-- Proof that the coefficient of x^2 in the expansion of (2x^3 + 4x^2 - 3x + 5)(3x^2 - 9x + 1) is 46 -/
theorem coefficient_x_squared_is_46 : coefficient_x_squared = 46 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_46_l877_87767


namespace NUMINAMATH_CALUDE_race_probability_l877_87725

theorem race_probability (total_cars : ℕ) (prob_x prob_y prob_z : ℚ) :
  total_cars = 15 →
  prob_x = 1 / 4 →
  prob_y = 1 / 8 →
  prob_z = 1 / 12 →
  (prob_x + prob_y + prob_z : ℚ) = 11 / 24 :=
by sorry

end NUMINAMATH_CALUDE_race_probability_l877_87725


namespace NUMINAMATH_CALUDE_prob_same_color_is_49_128_l877_87761

-- Define the number of balls of each color
def green_balls : ℕ := 8
def red_balls : ℕ := 5
def blue_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := green_balls + red_balls + blue_balls

-- Define the probability of drawing two balls of the same color
def prob_same_color : ℚ := 
  (green_balls * green_balls + red_balls * red_balls + blue_balls * blue_balls) / 
  (total_balls * total_balls)

-- Theorem statement
theorem prob_same_color_is_49_128 : prob_same_color = 49 / 128 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_49_128_l877_87761


namespace NUMINAMATH_CALUDE_book_collection_problem_l877_87701

theorem book_collection_problem (shared_books books_alice books_bob_unique : ℕ) 
  (h1 : shared_books = 12)
  (h2 : books_alice = 26)
  (h3 : books_bob_unique = 8) :
  books_alice - shared_books + books_bob_unique = 22 := by
  sorry

end NUMINAMATH_CALUDE_book_collection_problem_l877_87701


namespace NUMINAMATH_CALUDE_v_2002_equals_1_l877_87762

/-- The function g as defined in the problem --/
def g : ℕ → ℕ
| 1 => 2
| 2 => 3
| 3 => 1
| 4 => 4
| 5 => 5
| _ => 0  -- default case for inputs not in the table

/-- The sequence v defined recursively --/
def v : ℕ → ℕ
| 0 => 2
| (n + 1) => g (v n)

/-- Theorem stating that the 2002nd term of the sequence is 1 --/
theorem v_2002_equals_1 : v 2002 = 1 := by
  sorry


end NUMINAMATH_CALUDE_v_2002_equals_1_l877_87762


namespace NUMINAMATH_CALUDE_circles_centers_form_rectangle_l877_87752

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

def inside (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 < (c2.radius - c1.radius)^2

def rectangle (a b c d : ℝ × ℝ) : Prop :=
  let ab := (b.1 - a.1, b.2 - a.2)
  let bc := (c.1 - b.1, c.2 - b.2)
  let cd := (d.1 - c.1, d.2 - c.2)
  let da := (a.1 - d.1, a.2 - d.2)
  ab.1 * bc.1 + ab.2 * bc.2 = 0 ∧
  bc.1 * cd.1 + bc.2 * cd.2 = 0 ∧
  cd.1 * da.1 + cd.2 * da.2 = 0 ∧
  da.1 * ab.1 + da.2 * ab.2 = 0 ∧
  ab.1^2 + ab.2^2 = cd.1^2 + cd.2^2 ∧
  bc.1^2 + bc.2^2 = da.1^2 + da.2^2

theorem circles_centers_form_rectangle 
  (C C1 C2 C3 C4 : Circle)
  (h1 : C.radius = 2)
  (h2 : C1.radius = 1)
  (h3 : C2.radius = 1)
  (h4 : tangent C1 C2)
  (h5 : inside C1 C)
  (h6 : inside C2 C)
  (h7 : inside C3 C)
  (h8 : inside C4 C)
  (h9 : tangent C3 C)
  (h10 : tangent C3 C1)
  (h11 : tangent C3 C2)
  (h12 : tangent C4 C)
  (h13 : tangent C4 C1)
  (h14 : tangent C4 C3)
  : rectangle C.center C1.center C3.center C4.center :=
sorry

end NUMINAMATH_CALUDE_circles_centers_form_rectangle_l877_87752


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_10_l877_87711

theorem factorization_of_2x_squared_minus_10 :
  ∀ x : ℝ, 2 * x^2 - 10 = 2 * (x + Real.sqrt 5) * (x - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_10_l877_87711


namespace NUMINAMATH_CALUDE_lena_time_to_counter_l877_87709

/-- The time it takes Lena to reach the counter given her initial movement and remaining distance -/
theorem lena_time_to_counter (initial_distance : ℝ) (initial_time : ℝ) (remaining_distance_meters : ℝ) :
  initial_distance = 40 →
  initial_time = 20 →
  remaining_distance_meters = 100 →
  (remaining_distance_meters * 3.28084) / (initial_distance / initial_time) = 164.042 := by
sorry

end NUMINAMATH_CALUDE_lena_time_to_counter_l877_87709


namespace NUMINAMATH_CALUDE_min_copy_paste_actions_l877_87784

theorem min_copy_paste_actions (n : ℕ) : (2^n - 1 ≥ 1000) ∧ (∀ m : ℕ, m < n → 2^m - 1 < 1000) ↔ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_copy_paste_actions_l877_87784


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l877_87745

theorem rectangle_area_difference (x : ℝ) : 
  (2 * (x + 7)) * (2 * (x + 5)) - (3 * (2 * x - 3)) * (3 * (x - 2)) = -14 * x^2 + 111 * x + 86 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l877_87745


namespace NUMINAMATH_CALUDE_coin_combinations_theorem_l877_87759

/-- Represents the denominations of coins available in kopecks -/
def coin_denominations : List Nat := [1, 2, 5, 10, 20, 50]

/-- Represents the total amount to be made in kopecks -/
def total_amount : Nat := 100

/-- 
  Calculates the number of ways to make the total amount using the given coin denominations
  
  @param coins The list of available coin denominations
  @param amount The total amount to be made
  @return The number of ways to make the total amount
-/
def count_ways (coins : List Nat) (amount : Nat) : Nat :=
  sorry

theorem coin_combinations_theorem : 
  count_ways coin_denominations total_amount = 4562 := by
  sorry

end NUMINAMATH_CALUDE_coin_combinations_theorem_l877_87759


namespace NUMINAMATH_CALUDE_valid_spy_placement_exists_l877_87708

/-- Represents a position on the board -/
structure Position where
  x : Fin 6
  y : Fin 6

/-- Represents the vision of a spy -/
inductive Vision where
  | ahead : Position → Vision
  | right : Position → Vision
  | left : Position → Vision

/-- Checks if a spy at the given position can see the target position -/
def canSee (spyPos : Position) (targetPos : Position) : Prop :=
  ∃ (v : Vision),
    match v with
    | Vision.ahead p => p.x = spyPos.x ∧ p.y = spyPos.y + 1 ∨ p.y = spyPos.y + 2
    | Vision.right p => p.x = spyPos.x + 1 ∧ p.y = spyPos.y
    | Vision.left p => p.x = spyPos.x - 1 ∧ p.y = spyPos.y

/-- A valid spy placement is a list of 18 positions where no spy can see another -/
def ValidSpyPlacement (placement : List Position) : Prop :=
  placement.length = 18 ∧
  ∀ (spy1 spy2 : Position),
    spy1 ∈ placement → spy2 ∈ placement → spy1 ≠ spy2 →
    ¬(canSee spy1 spy2 ∨ canSee spy2 spy1)

/-- Theorem stating that a valid spy placement exists -/
theorem valid_spy_placement_exists : ∃ (placement : List Position), ValidSpyPlacement placement :=
  sorry

end NUMINAMATH_CALUDE_valid_spy_placement_exists_l877_87708


namespace NUMINAMATH_CALUDE_integral_symmetric_function_l877_87729

theorem integral_symmetric_function (a : ℝ) (h : a > 0) :
  ∫ x in -a..a, (x^2 * Real.cos x + Real.exp x) / (Real.exp x + 1) = a := by sorry

end NUMINAMATH_CALUDE_integral_symmetric_function_l877_87729


namespace NUMINAMATH_CALUDE_max_value_theorem_l877_87758

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 - x*y + y^2 = 8) :
  (∃ (z : ℝ), z = x^2 + x*y + y^2 ∧ z ≤ 24) ∧
  (∃ (a b c d : ℕ+), 24 = (a + b * Real.sqrt c) / d ∧ a + b + c + d = 26) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l877_87758


namespace NUMINAMATH_CALUDE_photo_ratio_proof_l877_87772

def claire_photos : ℕ := 6
def robert_photos (claire : ℕ) : ℕ := claire + 12

theorem photo_ratio_proof (lisa : ℕ) (h1 : lisa = robert_photos claire_photos) :
  lisa / claire_photos = 3 := by
  sorry

end NUMINAMATH_CALUDE_photo_ratio_proof_l877_87772


namespace NUMINAMATH_CALUDE_percentage_equality_l877_87738

theorem percentage_equality (x : ℝ) (h : 0.3 * 0.4 * x = 24) : 0.4 * 0.3 * x = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l877_87738


namespace NUMINAMATH_CALUDE_other_factors_of_twenty_l877_87724

theorem other_factors_of_twenty (y : ℕ) : 
  y = 20 ∧ y % 5 = 0 ∧ y % 8 ≠ 0 → 
  (∀ x : ℕ, x ≠ 1 ∧ x ≠ 5 ∧ y % x = 0 → x = 2 ∨ x = 4 ∨ x = 10) :=
by sorry

end NUMINAMATH_CALUDE_other_factors_of_twenty_l877_87724


namespace NUMINAMATH_CALUDE_square_product_inequality_l877_87742

theorem square_product_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 > a*b ∧ a*b > b^2 := by sorry

end NUMINAMATH_CALUDE_square_product_inequality_l877_87742
