import Mathlib

namespace NUMINAMATH_CALUDE_nested_sqrt_simplification_l84_8426

theorem nested_sqrt_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x ^ 9) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_simplification_l84_8426


namespace NUMINAMATH_CALUDE_remaining_expenses_l84_8492

def base_8_to_10 (n : ℕ) : ℕ := 
  5 * 8^3 + 4 * 8^2 + 3 * 8^1 + 2 * 8^0

def savings : ℕ := base_8_to_10 5432
def ticket_cost : ℕ := 1200

theorem remaining_expenses : savings - ticket_cost = 1642 := by
  sorry

end NUMINAMATH_CALUDE_remaining_expenses_l84_8492


namespace NUMINAMATH_CALUDE_sixth_power_of_complex_number_l84_8453

theorem sixth_power_of_complex_number :
  let z : ℂ := (Real.sqrt 3 + Complex.I) / 2
  z^6 = -1 := by sorry

end NUMINAMATH_CALUDE_sixth_power_of_complex_number_l84_8453


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l84_8460

-- Define set A
def A : Set ℝ := {x | x^2 + x - 12 < 0}

-- Define set B
def B : Set ℝ := {x | Real.sqrt (x + 2) < 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l84_8460


namespace NUMINAMATH_CALUDE_oranges_per_box_l84_8452

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (h1 : total_oranges = 24) (h2 : num_boxes = 3) :
  total_oranges / num_boxes = 8 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l84_8452


namespace NUMINAMATH_CALUDE_range_of_m_l84_8494

theorem range_of_m (x : ℝ) (m : ℝ) : 
  (∃ x ∈ Set.Icc (-1) 1, x^2 - x - (m + 1) = 0) →
  m ∈ Set.Icc (-5/4) 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l84_8494


namespace NUMINAMATH_CALUDE_sqrt_calculations_l84_8420

theorem sqrt_calculations :
  (2 * Real.sqrt 2 + Real.sqrt 27 - Real.sqrt 8 = 3 * Real.sqrt 3) ∧
  ((2 * Real.sqrt 12 - 3 * Real.sqrt (1/3)) * Real.sqrt 6 = 9 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l84_8420


namespace NUMINAMATH_CALUDE_part_one_part_two_l84_8414

-- Define the quadratic functions p and q
def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0
def q (m x : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0

-- Theorem for part (1)
theorem part_one (m : ℝ) (h : m = 4) :
  (∀ x, p x ∧ q m x ↔ 4 < x ∧ x < 5) :=
sorry

-- Theorem for part (2)
theorem part_two :
  (∀ m, (∀ x, ¬(q m x) → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ q m x)) ↔ (5/3 ≤ m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l84_8414


namespace NUMINAMATH_CALUDE_equation_solution_l84_8499

theorem equation_solution :
  let f (x : ℝ) := x^2 + 2*x + 1
  let g (x : ℝ) := |3*x - 2|
  let sol₁ := (-7 + Real.sqrt 37) / 2
  let sol₂ := (-7 - Real.sqrt 37) / 2
  (∀ x : ℝ, f x = g x ↔ x = sol₁ ∨ x = sol₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l84_8499


namespace NUMINAMATH_CALUDE_hexagon_ring_area_l84_8480

/-- The area of the ring between the inscribed and circumscribed circles of a regular hexagon -/
theorem hexagon_ring_area (a : ℝ) (h : a > 0) : 
  let r_inscribed := (Real.sqrt 3 / 2) * a
  let r_circumscribed := a
  let area_inscribed := π * r_inscribed ^ 2
  let area_circumscribed := π * r_circumscribed ^ 2
  area_circumscribed - area_inscribed = π * a^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_ring_area_l84_8480


namespace NUMINAMATH_CALUDE_jose_fowls_count_l84_8439

/-- The number of fowls Jose has is the sum of his chickens and ducks -/
theorem jose_fowls_count :
  let chickens : ℕ := 28
  let ducks : ℕ := 18
  let fowls : ℕ := chickens + ducks
  fowls = 46 := by sorry

end NUMINAMATH_CALUDE_jose_fowls_count_l84_8439


namespace NUMINAMATH_CALUDE_complex_fraction_difference_l84_8472

theorem complex_fraction_difference : 
  (Complex.mk 3 2) / (Complex.mk 2 (-3)) - (Complex.mk 3 (-2)) / (Complex.mk 2 3) = Complex.I * 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_difference_l84_8472


namespace NUMINAMATH_CALUDE_max_value_on_line_l84_8409

/-- Given a point A(3,1) on the line mx + ny + 1 = 0, where mn > 0, 
    the maximum value of 3/m + 1/n is -16. -/
theorem max_value_on_line (m n : ℝ) : 
  (3 * m + n = -1) → 
  (m * n > 0) → 
  (∀ k l : ℝ, (3 * k + l = -1) → (k * l > 0) → (3 / m + 1 / n ≥ 3 / k + 1 / l)) →
  3 / m + 1 / n = -16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_line_l84_8409


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_l84_8491

def f (x : ℝ) : ℝ := x^3 + 2*x

theorem f_extrema_on_interval :
  let a := -1
  let b := 1
  ∃ (x_min x_max : ℝ),
    x_min ∈ [a, b] ∧
    x_max ∈ [a, b] ∧
    (∀ x ∈ [a, b], f x ≥ f x_min) ∧
    (∀ x ∈ [a, b], f x ≤ f x_max) ∧
    f x_min = -3 ∧
    f x_max = 3 :=
sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_l84_8491


namespace NUMINAMATH_CALUDE_rachel_essay_pages_l84_8490

/-- Rachel's essay writing problem -/
theorem rachel_essay_pages :
  let pages_per_30_min : ℕ := 1
  let research_time : ℕ := 45
  let editing_time : ℕ := 75
  let total_time : ℕ := 300
  let writing_time : ℕ := total_time - (research_time + editing_time)
  let pages_written : ℕ := writing_time / 30
  pages_written = 6 := by sorry

end NUMINAMATH_CALUDE_rachel_essay_pages_l84_8490


namespace NUMINAMATH_CALUDE_difference_is_negative_1200_l84_8435

def A : ℕ → ℕ
| 0 => 2 * 49
| n + 1 => (2 * n + 1) * (2 * n + 2) + A n

def B : ℕ → ℕ
| 0 => 48 * 49
| n + 1 => (2 * n) * (2 * n + 1) + B n

def difference : ℤ := A 24 - B 24

theorem difference_is_negative_1200 : difference = -1200 := by
  sorry

end NUMINAMATH_CALUDE_difference_is_negative_1200_l84_8435


namespace NUMINAMATH_CALUDE_negation_of_proposition_l84_8425

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, (x ≠ 3 ∧ x ≠ 2) → x^2 - 5*x + 6 ≠ 0)) ↔
  (∀ x : ℝ, (x = 3 ∨ x = 2) → x^2 - 5*x + 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l84_8425


namespace NUMINAMATH_CALUDE_circle_intersection_and_distance_four_points_distance_l84_8417

/-- The circle equation parameters -/
structure CircleParams where
  m : ℝ
  h : m < 5

/-- The line equation parameters -/
structure LineParams where
  c : ℝ

/-- The theorem statement -/
theorem circle_intersection_and_distance (p : CircleParams) :
  (∃ M N : ℝ × ℝ, 
    (M.1^2 + M.2^2 - 2*M.1 - 4*M.2 + p.m = 0) ∧
    (N.1^2 + N.2^2 - 2*N.1 - 4*N.2 + p.m = 0) ∧
    (M.1 + 2*M.2 - 4 = 0) ∧
    (N.1 + 2*N.2 - 4 = 0) ∧
    ((M.1 - N.1)^2 + (M.2 - N.2)^2 = (4*Real.sqrt 5/5)^2)) →
  p.m = 4 :=
sorry

/-- The theorem statement for the second part -/
theorem four_points_distance (p : CircleParams) (l : LineParams) :
  (p.m = 4) →
  (∃ A B C D : ℝ × ℝ,
    (A.1^2 + A.2^2 - 2*A.1 - 4*A.2 + p.m = 0) ∧
    (B.1^2 + B.2^2 - 2*B.1 - 4*B.2 + p.m = 0) ∧
    (C.1^2 + C.2^2 - 2*C.1 - 4*C.2 + p.m = 0) ∧
    (D.1^2 + D.2^2 - 2*D.1 - 4*D.2 + p.m = 0) ∧
    ((A.1 - 2*A.2 + l.c)^2 / 5 = (Real.sqrt 5/5)^2) ∧
    ((B.1 - 2*B.2 + l.c)^2 / 5 = (Real.sqrt 5/5)^2) ∧
    ((C.1 - 2*C.2 + l.c)^2 / 5 = (Real.sqrt 5/5)^2) ∧
    ((D.1 - 2*D.2 + l.c)^2 / 5 = (Real.sqrt 5/5)^2)) ↔
  (4 - Real.sqrt 5 < l.c ∧ l.c < 2 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_and_distance_four_points_distance_l84_8417


namespace NUMINAMATH_CALUDE_find_a_min_g_l84_8436

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- Define the function g
def g (x : ℝ) : ℝ := f 2 x - |x + 1|

-- Theorem for part (I)
theorem find_a : 
  (∀ x : ℝ, f 2 x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) → 
  ∃ a : ℝ, ∀ x : ℝ, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1 :=
sorry

-- Theorem for part (II)
theorem min_g : 
  ∃ m : ℝ, m = -1/2 ∧ ∀ x : ℝ, g x ≥ m :=
sorry

end NUMINAMATH_CALUDE_find_a_min_g_l84_8436


namespace NUMINAMATH_CALUDE_currency_conversion_l84_8497

/-- Conversion rates and constants --/
def paise_per_rupee : ℚ := 100
def usd_per_inr : ℚ := 12 / 1000
def eur_per_inr : ℚ := 10 / 1000
def gbp_per_inr : ℚ := 9 / 1000

/-- The value 'a' in paise --/
def a_paise : ℚ := 15000

/-- Theorem stating the correct values of 'a' in different currencies --/
theorem currency_conversion (a : ℚ) 
  (h1 : a * (1/2) / 100 = 75) : 
  a / paise_per_rupee = 150 ∧ 
  a / paise_per_rupee * usd_per_inr = 9/5 ∧ 
  a / paise_per_rupee * eur_per_inr = 3/2 ∧ 
  a / paise_per_rupee * gbp_per_inr = 27/20 := by
  sorry

#check currency_conversion

end NUMINAMATH_CALUDE_currency_conversion_l84_8497


namespace NUMINAMATH_CALUDE_max_candies_consumed_max_candies_is_1225_l84_8400

/-- The number of initial ones on the board -/
def initial_ones : ℕ := 50

/-- The number of minutes the process continues -/
def total_minutes : ℕ := 50

/-- The number of candies consumed is equal to the number of edges in a complete graph -/
theorem max_candies_consumed (n : ℕ) (h : n = initial_ones) :
  (n * (n - 1)) / 2 = total_minutes * (total_minutes - 1) / 2 := by sorry

/-- The maximum number of candies consumed after the process -/
def max_candies : ℕ := (initial_ones * (initial_ones - 1)) / 2

/-- Proof that the maximum number of candies consumed is 1225 -/
theorem max_candies_is_1225 : max_candies = 1225 := by sorry

end NUMINAMATH_CALUDE_max_candies_consumed_max_candies_is_1225_l84_8400


namespace NUMINAMATH_CALUDE_distribution_recurrence_l84_8454

/-- The number of ways to distribute n distinct items to k people,
    such that each person receives at least one item -/
def g (n k : ℕ) : ℕ := sorry

theorem distribution_recurrence (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 2) :
  g (n + 1) k = k * g n (k - 1) + k * g n k :=
sorry

end NUMINAMATH_CALUDE_distribution_recurrence_l84_8454


namespace NUMINAMATH_CALUDE_parabola_line_slope_l84_8456

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2

-- Define a point in the first quadrant
def in_first_quadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 > 0

-- Define the vector relationship
def vector_relation (P Q : ℝ × ℝ) : Prop :=
  3 * (focus.1 - P.1) = Q.1 - focus.1 ∧
  3 * (focus.2 - P.2) = Q.2 - focus.2

-- Main theorem
theorem parabola_line_slope (P Q : ℝ × ℝ) :
  on_parabola P →
  on_parabola Q →
  in_first_quadrant Q →
  vector_relation P Q →
  (Q.2 - P.2) / (Q.1 - P.1) = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_slope_l84_8456


namespace NUMINAMATH_CALUDE_black_population_west_percentage_l84_8481

def black_population_ne : ℕ := 6
def black_population_mw : ℕ := 7
def black_population_south : ℕ := 18
def black_population_west : ℕ := 4

def total_black_population : ℕ := black_population_ne + black_population_mw + black_population_south + black_population_west

def percentage_in_west : ℚ := black_population_west / total_black_population

theorem black_population_west_percentage :
  ∃ (p : ℚ), abs (percentage_in_west - p) < 1/100 ∧ p = 11/100 := by
  sorry

end NUMINAMATH_CALUDE_black_population_west_percentage_l84_8481


namespace NUMINAMATH_CALUDE_complex_modulus_l84_8474

/-- If z is a complex number satisfying (2+i)z = 5, then |z| = √5 -/
theorem complex_modulus (z : ℂ) (h : (2 + Complex.I) * z = 5) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l84_8474


namespace NUMINAMATH_CALUDE_factorization_equality_l84_8448

theorem factorization_equality (a x y : ℝ) :
  5 * a * x^2 - 5 * a * y^2 = 5 * a * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l84_8448


namespace NUMINAMATH_CALUDE_range_of_a_l84_8418

/-- Given two predicates p and q on real numbers, where p is a sufficient but not necessary condition for q, 
    this theorem states that the range of the parameter a in q is [0, 1/2]. -/
theorem range_of_a (p q : ℝ → Prop) (a : ℝ) : 
  (∀ x, p x ↔ |4*x - 3| ≤ 1) →
  (∀ x, q x ↔ x^2 - (2*a + 1)*x + a^2 + a ≤ 0) →
  (∀ x, p x → q x) →
  (∃ x, q x ∧ ¬p x) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l84_8418


namespace NUMINAMATH_CALUDE_rectangle_max_area_l84_8423

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) → 
  l * w = 100 := by
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l84_8423


namespace NUMINAMATH_CALUDE_remainder_sum_factorials_60_l84_8408

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem remainder_sum_factorials_60 (h : ∀ k ≥ 5, 15 ∣ factorial k) :
  sum_factorials 60 % 15 = (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_factorials_60_l84_8408


namespace NUMINAMATH_CALUDE_compare_fractions_compare_specific_fractions_l84_8438

theorem compare_fractions (a b : ℝ) (h1 : 3 * a > b) (h2 : b > 0) : a / b > (a + 1) / (b + 3) := by
  sorry

theorem compare_specific_fractions : (23 : ℝ) / 68 < 22 / 65 := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_compare_specific_fractions_l84_8438


namespace NUMINAMATH_CALUDE_foreign_language_score_foreign_language_score_is_98_l84_8421

theorem foreign_language_score (average_three : ℝ) (average_two : ℝ) 
  (h1 : average_three = 94) (h2 : average_two = 92) : ℝ :=
  3 * average_three - 2 * average_two

theorem foreign_language_score_is_98 (average_three : ℝ) (average_two : ℝ) 
  (h1 : average_three = 94) (h2 : average_two = 92) : 
  foreign_language_score average_three average_two h1 h2 = 98 := by
  sorry

end NUMINAMATH_CALUDE_foreign_language_score_foreign_language_score_is_98_l84_8421


namespace NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l84_8447

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

def sum_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem max_sum_arithmetic_sequence 
  (a d : ℤ) 
  (h1 : a + 16 * d = 52) 
  (h2 : a + 29 * d = 13) :
  ∃ n : ℕ, 
    (arithmetic_sequence a d n > 0) ∧ 
    (arithmetic_sequence a d (n + 1) ≤ 0) ∧
    (∀ m : ℕ, m > n → arithmetic_sequence a d m ≤ 0) ∧
    (sum_arithmetic_sequence a d n = 1717) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l84_8447


namespace NUMINAMATH_CALUDE_coin_value_ratio_l84_8440

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of nickels -/
def num_nickels : ℕ := 4

/-- The number of dimes -/
def num_dimes : ℕ := 6

/-- The number of quarters -/
def num_quarters : ℕ := 2

theorem coin_value_ratio :
  ∃ (k : ℕ), k > 0 ∧
    num_nickels * nickel_value = 2 * k ∧
    num_dimes * dime_value = 6 * k ∧
    num_quarters * quarter_value = 5 * k :=
sorry

end NUMINAMATH_CALUDE_coin_value_ratio_l84_8440


namespace NUMINAMATH_CALUDE_sequence_formula_l84_8449

theorem sequence_formula (a : ℕ+ → ℚ) 
  (h1 : a 1 = 1/2)
  (h2 : ∀ n : ℕ+, a n * a (n + 1) = n / (n + 2)) :
  ∀ n : ℕ+, a n = n / (n + 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l84_8449


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_500_l84_8412

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 = 495 ∧ 
  495 < 500 ∧ 
  ∀ m : ℕ, m * 15 < 500 → m * 15 ≤ 495 := by
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_500_l84_8412


namespace NUMINAMATH_CALUDE_f_properties_imply_b_range_l84_8431

-- Define the function f
noncomputable def f (b : ℝ) : ℝ → ℝ := fun x =>
  if 0 < x ∧ x < 2 then Real.log (x^2 - x + b) else 0  -- placeholder for other x values

-- State the theorem
theorem f_properties_imply_b_range :
  ∀ b : ℝ,
  (∀ x : ℝ, f b (-x) = -(f b x)) →  -- f is odd
  (∀ x : ℝ, f b (x + 4) = f b x) →  -- f has period 4
  (∀ x : ℝ, 0 < x → x < 2 → f b x = Real.log (x^2 - x + b)) →  -- f definition for x ∈ (0, 2)
  (∃ x₁ x₂ x₃ x₄ x₅ : ℝ, -2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧ x₅ ≤ 2 ∧
    f b x₁ = 0 ∧ f b x₂ = 0 ∧ f b x₃ = 0 ∧ f b x₄ = 0 ∧ f b x₅ = 0) →  -- 5 zero points in [-2, 2]
  ((1/4 < b ∧ b ≤ 1) ∨ b = 5/4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_imply_b_range_l84_8431


namespace NUMINAMATH_CALUDE_x_value_proof_l84_8410

theorem x_value_proof (x y : ℝ) 
  (eq1 : x^2 - 4*x + y = 0) 
  (eq2 : y = 4) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l84_8410


namespace NUMINAMATH_CALUDE_lcm_gcd_product_12_75_l84_8465

theorem lcm_gcd_product_12_75 : Nat.lcm 12 75 * Nat.gcd 12 75 = 900 := by sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_12_75_l84_8465


namespace NUMINAMATH_CALUDE_total_barks_after_duration_l84_8473

/-- Represents the number of barks per minute for a single dog -/
def barks_per_minute : ℕ := 30

/-- Represents the number of dogs -/
def num_dogs : ℕ := 2

/-- Represents the duration in minutes -/
def duration : ℕ := 10

/-- Theorem stating that the total number of barks after the given duration is 600 -/
theorem total_barks_after_duration :
  num_dogs * barks_per_minute * duration = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_barks_after_duration_l84_8473


namespace NUMINAMATH_CALUDE_function_equation_solution_l84_8475

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 4 = x + y + 3) : 
  ∀ x : ℝ, f x = x + 4 := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l84_8475


namespace NUMINAMATH_CALUDE_constant_term_value_l84_8407

/-- The constant term in the expansion of (3x + 2/x)^8 -/
def constant_term : ℕ :=
  let binomial_coeff := (8 : ℕ).choose 4
  let x_power_term := 3^4 * 2^4
  binomial_coeff * x_power_term

/-- The constant term in the expansion of (3x + 2/x)^8 is 90720 -/
theorem constant_term_value : constant_term = 90720 := by
  sorry

#eval constant_term -- This will evaluate the constant term

end NUMINAMATH_CALUDE_constant_term_value_l84_8407


namespace NUMINAMATH_CALUDE_limes_remaining_l84_8479

/-- The number of limes Mike picked -/
def mike_limes : ℝ := 32.0

/-- The number of limes Alyssa ate -/
def alyssa_limes : ℝ := 25.0

/-- The number of limes left -/
def limes_left : ℝ := mike_limes - alyssa_limes

theorem limes_remaining : limes_left = 7.0 := by sorry

end NUMINAMATH_CALUDE_limes_remaining_l84_8479


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l84_8495

/-- 
Given a rectangular solid with dimensions x, y, and z,
if the total surface area is 34 cm² and the total length of all edges is 28 cm,
then the length of any interior diagonal is √15 cm.
-/
theorem rectangular_solid_diagonal 
  (x y z : ℝ) 
  (h_surface_area : 2 * (x * y + y * z + z * x) = 34)
  (h_edge_length : 4 * (x + y + z) = 28) :
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l84_8495


namespace NUMINAMATH_CALUDE_transaction_error_l84_8450

theorem transaction_error (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 →
  (100 * y + x) - (100 * x + y) = 5616 →
  y = x + 56 := by
sorry

end NUMINAMATH_CALUDE_transaction_error_l84_8450


namespace NUMINAMATH_CALUDE_edmund_computer_savings_l84_8496

/-- Represents the savings problem for Edmund's computer purchase. -/
def computer_savings_problem (computer_cost starting_balance monthly_gift : ℚ)
  (part_time_daily_wage part_time_days_per_week : ℚ)
  (extra_chore_wage chores_per_day regular_chores_per_week : ℚ)
  (car_wash_wage car_washes_per_week : ℚ)
  (lawn_mowing_wage lawns_per_week : ℚ) : Prop :=
  let weekly_earnings := 
    part_time_daily_wage * part_time_days_per_week +
    extra_chore_wage * (chores_per_day * 7 - regular_chores_per_week) +
    car_wash_wage * car_washes_per_week +
    lawn_mowing_wage * lawns_per_week
  let weekly_savings := weekly_earnings + monthly_gift / 4
  let days_to_save := 
    (↑(Nat.ceil ((computer_cost - starting_balance) / weekly_savings)) * 7 : ℚ)
  days_to_save = 49

/-- Theorem stating that Edmund will save enough for the computer in 49 days. -/
theorem edmund_computer_savings : 
  computer_savings_problem 750 200 50 10 3 2 4 12 3 2 5 1 := by
  sorry


end NUMINAMATH_CALUDE_edmund_computer_savings_l84_8496


namespace NUMINAMATH_CALUDE_equation_solution_l84_8416

theorem equation_solution :
  ∃ (t₁ t₂ : ℝ), t₁ > t₂ ∧
  (∀ t : ℝ, t ≠ 10 → (t^2 - 3*t - 70) / (t - 10) = 7 / (t + 4) ↔ t = t₁ ∨ t = t₂) ∧
  t₁ = -3 ∧ t₂ = -7 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l84_8416


namespace NUMINAMATH_CALUDE_inequality_solutions_l84_8489

theorem inequality_solutions :
  (∀ x : ℝ, 3 + 2*x > -x - 6 ↔ x > -3) ∧
  (∀ x : ℝ, (2*x + 1 ≤ x + 3 ∧ (2*x + 1) / 3 > 1) ↔ 1 < x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_l84_8489


namespace NUMINAMATH_CALUDE_order_of_a_l84_8471

theorem order_of_a (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^2 ∧ -a^2 > a := by
  sorry

end NUMINAMATH_CALUDE_order_of_a_l84_8471


namespace NUMINAMATH_CALUDE_jasmine_remaining_money_l84_8430

/-- Calculates the remaining amount after spending on fruits --/
def remaining_amount (initial : ℝ) (spent : ℝ) : ℝ :=
  initial - spent

/-- Theorem: The remaining amount after spending $15.00 from an initial $100.00 is $85.00 --/
theorem jasmine_remaining_money :
  remaining_amount 100 15 = 85 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_remaining_money_l84_8430


namespace NUMINAMATH_CALUDE_gecko_eating_pattern_l84_8429

/-- Represents the gecko's eating pattern over three days -/
structure GeckoEating where
  total_crickets : ℕ
  third_day_crickets : ℕ
  second_day_difference : ℕ

/-- Calculates the percentage of crickets eaten on the first day -/
def first_day_percentage (g : GeckoEating) : ℚ :=
  let first_two_days := g.total_crickets - g.third_day_crickets
  let x := (2 * first_two_days + g.second_day_difference) / (2 * g.total_crickets)
  x * 100

/-- Theorem stating that under the given conditions, the gecko eats 30% of crickets on the first day -/
theorem gecko_eating_pattern :
  let g : GeckoEating := {
    total_crickets := 70,
    third_day_crickets := 34,
    second_day_difference := 6
  }
  first_day_percentage g = 30 := by sorry

end NUMINAMATH_CALUDE_gecko_eating_pattern_l84_8429


namespace NUMINAMATH_CALUDE_marble_jar_problem_l84_8487

theorem marble_jar_problem :
  ∀ (total_marbles : ℕ) (blue1 green1 blue2 green2 : ℕ),
    -- Jar 1 ratio condition
    7 * green1 = 2 * blue1 →
    -- Jar 2 ratio condition
    8 * green2 = blue2 →
    -- Equal total marbles in each jar
    blue1 + green1 = blue2 + green2 →
    -- Total green marbles
    green1 + green2 = 135 →
    -- Difference in blue marbles
    blue2 - blue1 = 45 :=
by sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l84_8487


namespace NUMINAMATH_CALUDE_dogwood_trees_tomorrow_l84_8468

/-- The number of dogwood trees to be planted tomorrow in the park --/
def trees_planted_tomorrow (initial_trees : ℕ) (planted_today : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial_trees + planted_today)

/-- Theorem: Given the initial number of trees, the number planted today, and the final total,
    prove that 20 trees will be planted tomorrow --/
theorem dogwood_trees_tomorrow :
  trees_planted_tomorrow 39 41 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_tomorrow_l84_8468


namespace NUMINAMATH_CALUDE_fraction_of_girls_at_dance_l84_8422

theorem fraction_of_girls_at_dance (
  dalton_total : ℕ) (dalton_ratio_boys : ℕ) (dalton_ratio_girls : ℕ)
  (berkeley_total : ℕ) (berkeley_ratio_boys : ℕ) (berkeley_ratio_girls : ℕ)
  (kingston_total : ℕ) (kingston_ratio_boys : ℕ) (kingston_ratio_girls : ℕ)
  (h1 : dalton_total = 300)
  (h2 : dalton_ratio_boys = 3)
  (h3 : dalton_ratio_girls = 2)
  (h4 : berkeley_total = 210)
  (h5 : berkeley_ratio_boys = 3)
  (h6 : berkeley_ratio_girls = 4)
  (h7 : kingston_total = 240)
  (h8 : kingston_ratio_boys = 5)
  (h9 : kingston_ratio_girls = 7)
  : (dalton_total * dalton_ratio_girls / (dalton_ratio_boys + dalton_ratio_girls) +
     berkeley_total * berkeley_ratio_girls / (berkeley_ratio_boys + berkeley_ratio_girls) +
     kingston_total * kingston_ratio_girls / (kingston_ratio_boys + kingston_ratio_girls)) /
    (dalton_total + berkeley_total + kingston_total) = 38 / 75 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_girls_at_dance_l84_8422


namespace NUMINAMATH_CALUDE_rectangle_side_length_l84_8459

/-- Given a square with side length 5 and a rectangle with one side 4,
    if they have the same area, then the other side of the rectangle is 6.25 -/
theorem rectangle_side_length (square_side : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  square_side = 5 →
  rectangle_width = 4 →
  square_side * square_side = rectangle_width * rectangle_length →
  rectangle_length = 6.25 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l84_8459


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l84_8424

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if c = 2√3, sin B = 2 sin A, and C = π/3, then a = 2, b = 4, and the area is 2√3 -/
theorem triangle_abc_properties (a b c A B C : ℝ) : 
  c = 2 * Real.sqrt 3 →
  Real.sin B = 2 * Real.sin A →
  C = π / 3 →
  (a = 2 ∧ b = 4 ∧ (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l84_8424


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_is_18000_l84_8402

/-- 
Represents an integer consisting of n repetitions of a digit d.
For example, repeat_digit 3 2000 represents 333...333 (2000 threes).
-/
def repeat_digit (d : ℕ) (n : ℕ) : ℕ :=
  (d * (10^n - 1)) / 9

/-- 
Calculates the sum of digits of a natural number in base 10.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

theorem sum_of_digits_9ab_is_18000 : 
  let a := repeat_digit 3 2000
  let b := repeat_digit 7 2000
  sum_of_digits (9 * a * b) = 18000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_is_18000_l84_8402


namespace NUMINAMATH_CALUDE_inequality_proof_l84_8415

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l84_8415


namespace NUMINAMATH_CALUDE_negation_of_implication_l84_8413

theorem negation_of_implication (a b : ℝ) : 
  ¬(ab = 0 → a = 0 ∨ b = 0) ↔ (ab ≠ 0 → a ≠ 0 ∧ b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l84_8413


namespace NUMINAMATH_CALUDE_christen_peeled_22_l84_8457

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initial_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  time_before_christen : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def potatoes_peeled_by_christen (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- Theorem stating that Christen peeled 22 potatoes -/
theorem christen_peeled_22 (scenario : PotatoPeeling) 
  (h1 : scenario.initial_potatoes = 60)
  (h2 : scenario.homer_rate = 4)
  (h3 : scenario.christen_rate = 6)
  (h4 : scenario.time_before_christen = 6) :
  potatoes_peeled_by_christen scenario = 22 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_22_l84_8457


namespace NUMINAMATH_CALUDE_complex_equation_solution_l84_8466

theorem complex_equation_solution (z : ℂ) : 
  (1 : ℂ) + Complex.I * Real.sqrt 3 = z * ((1 : ℂ) - Complex.I * Real.sqrt 3) → 
  z = -1/2 + Complex.I * (Real.sqrt 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l84_8466


namespace NUMINAMATH_CALUDE_even_function_iff_a_eq_zero_l84_8404

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

theorem even_function_iff_a_eq_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_iff_a_eq_zero_l84_8404


namespace NUMINAMATH_CALUDE_age_multiplier_proof_l84_8434

theorem age_multiplier_proof (matt_age john_age : ℕ) (h1 : matt_age = 41) (h2 : john_age = 11) :
  ∃ x : ℚ, matt_age = x * john_age - 3 :=
by
  sorry

end NUMINAMATH_CALUDE_age_multiplier_proof_l84_8434


namespace NUMINAMATH_CALUDE_bird_legs_count_l84_8455

theorem bird_legs_count (num_birds : ℕ) (legs_per_bird : ℕ) (h1 : num_birds = 5) (h2 : legs_per_bird = 2) :
  num_birds * legs_per_bird = 10 := by
  sorry

end NUMINAMATH_CALUDE_bird_legs_count_l84_8455


namespace NUMINAMATH_CALUDE_sum_of_squares_l84_8432

theorem sum_of_squares (a b c : ℕ+) (h1 : a < b) (h2 : b < c)
  (h3 : (b.val * c.val - 1) % a.val = 0)
  (h4 : (a.val * c.val - 1) % b.val = 0)
  (h5 : (a.val * b.val - 1) % c.val = 0) :
  a^2 + b^2 + c^2 = 38 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l84_8432


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l84_8445

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 7 ↔ (x : ℚ) / 4 + 3 / 7 < 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l84_8445


namespace NUMINAMATH_CALUDE_balcony_orchestra_difference_is_40_l84_8428

/-- Represents the ticket sales for a theater performance --/
structure TheaterSales where
  orchestra_price : ℕ
  balcony_price : ℕ
  total_tickets : ℕ
  total_revenue : ℕ

/-- Calculates the difference between balcony and orchestra ticket sales --/
def balcony_orchestra_difference (sales : TheaterSales) : ℕ :=
  sales.total_tickets - 2 * (sales.total_revenue - sales.balcony_price * sales.total_tickets) / (sales.orchestra_price - sales.balcony_price)

/-- Theorem stating the difference between balcony and orchestra ticket sales --/
theorem balcony_orchestra_difference_is_40 (sales : TheaterSales) 
  (h1 : sales.orchestra_price = 12)
  (h2 : sales.balcony_price = 8)
  (h3 : sales.total_tickets = 340)
  (h4 : sales.total_revenue = 3320) :
  balcony_orchestra_difference sales = 40 := by
  sorry

#eval balcony_orchestra_difference ⟨12, 8, 340, 3320⟩

end NUMINAMATH_CALUDE_balcony_orchestra_difference_is_40_l84_8428


namespace NUMINAMATH_CALUDE_product_125_sum_31_l84_8484

theorem product_125_sum_31 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → a * b * c = 125 → a + b + c = 31 := by
  sorry

end NUMINAMATH_CALUDE_product_125_sum_31_l84_8484


namespace NUMINAMATH_CALUDE_union_complement_equality_l84_8411

def U : Finset Nat := {0, 1, 2, 4, 6, 8}
def M : Finset Nat := {0, 4, 6}
def N : Finset Nat := {0, 1, 6}

theorem union_complement_equality : M ∪ (U \ N) = {0, 2, 4, 6, 8} := by sorry

end NUMINAMATH_CALUDE_union_complement_equality_l84_8411


namespace NUMINAMATH_CALUDE_fraction_comparison_l84_8446

theorem fraction_comparison : -3/4 > -4/5 := by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l84_8446


namespace NUMINAMATH_CALUDE_foci_distance_of_problem_ellipse_l84_8464

-- Define the ellipse
structure Ellipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

-- Define the conditions of the problem
def problem_ellipse : Ellipse := {
  center := (5, 2)
  semi_major_axis := 5
  semi_minor_axis := 2
}

-- Theorem statement
theorem foci_distance_of_problem_ellipse :
  let e := problem_ellipse
  let c := Real.sqrt (e.semi_major_axis ^ 2 - e.semi_minor_axis ^ 2)
  c = Real.sqrt 21 := by sorry

end NUMINAMATH_CALUDE_foci_distance_of_problem_ellipse_l84_8464


namespace NUMINAMATH_CALUDE_dice_probability_l84_8451

/-- The probability of all dice showing the same number -/
def probability : ℝ := 0.0007716049382716049

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The number of dice thrown -/
def num_dice : ℕ := 5

theorem dice_probability :
  (1 / faces : ℝ) ^ (num_dice - 1) = probability := by sorry

end NUMINAMATH_CALUDE_dice_probability_l84_8451


namespace NUMINAMATH_CALUDE_sum_of_products_l84_8469

theorem sum_of_products (x : Fin 150 → ℝ) : 
  (∀ i, x i = Real.sqrt 2 + 1 ∨ x i = Real.sqrt 2 - 1) →
  (∃ x : Fin 150 → ℝ, (∀ i, x i = Real.sqrt 2 + 1 ∨ x i = Real.sqrt 2 - 1) ∧ 
    (Finset.sum (Finset.range 75) (λ i => x (2*i) * x (2*i+1)) = 111)) ∧
  (¬ ∃ x : Fin 150 → ℝ, (∀ i, x i = Real.sqrt 2 + 1 ∨ x i = Real.sqrt 2 - 1) ∧ 
    (Finset.sum (Finset.range 75) (λ i => x (2*i) * x (2*i+1)) = 121)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_products_l84_8469


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l84_8488

theorem perfect_square_binomial : ∃ a b : ℝ, ∀ x : ℝ, x^2 - 20*x + 100 = (a*x + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l84_8488


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l84_8467

theorem profit_percent_calculation (selling_price cost_price profit : ℝ) :
  cost_price = 0.75 * selling_price →
  profit = selling_price - cost_price →
  (profit / cost_price) * 100 = 33.33333333333333 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l84_8467


namespace NUMINAMATH_CALUDE_counterfeit_coin_weighings_l84_8461

/-- Represents a weighing operation on a balance scale -/
def Weighing := List Nat → List Nat → Bool

/-- Represents a strategy for finding the counterfeit coin -/
def Strategy := List Nat → List (List Nat × List Nat)

/-- The number of coins -/
def n : Nat := 15

/-- The maximum number of weighings needed -/
def max_weighings : Nat := 3

theorem counterfeit_coin_weighings :
  ∃ (s : Strategy),
    ∀ (counterfeit : Fin n),
      ∀ (w : Weighing),
        (∀ i j : Fin n, i ≠ j → w [i.val] [j.val] = true) →
        (∀ i : Fin n, i ≠ counterfeit → w [i.val] [counterfeit.val] = false) →
        (s (List.range n)).length ≤ max_weighings ∧
        ∃ (result : Fin n), result = counterfeit := by sorry

end NUMINAMATH_CALUDE_counterfeit_coin_weighings_l84_8461


namespace NUMINAMATH_CALUDE_a_values_l84_8463

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (2 * x + a) ^ 3

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 6 * (2 * x + a) ^ 2

-- Theorem statement
theorem a_values (a : ℝ) : f_derivative a 1 = 6 → a = -1 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_a_values_l84_8463


namespace NUMINAMATH_CALUDE_seven_power_plus_one_prime_divisors_l84_8478

theorem seven_power_plus_one_prime_divisors (n : ℕ) :
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ 
    (∀ p ∈ S, p ∣ (7^(7^n) + 1)) ∧ 
    (Finset.card S ≥ 2*n + 3) :=
by sorry

end NUMINAMATH_CALUDE_seven_power_plus_one_prime_divisors_l84_8478


namespace NUMINAMATH_CALUDE_f_max_at_zero_l84_8458

-- Define the function f and its derivative
def f (x : ℝ) : ℝ := x^4 - 2*x^2 - 5

def f_deriv (x : ℝ) : ℝ := 4*x^3 - 4*x

-- State the theorem
theorem f_max_at_zero :
  (f 0 = -5) →
  (∀ x : ℝ, f_deriv x = 4*x^3 - 4*x) →
  ∀ x : ℝ, f x ≤ f 0 :=
by sorry

end NUMINAMATH_CALUDE_f_max_at_zero_l84_8458


namespace NUMINAMATH_CALUDE_wam_gm_difference_bound_l84_8437

theorem wam_gm_difference_bound (k b : ℝ) (h1 : 0 < k) (h2 : k < 1) (h3 : b > 0) : 
  let a := k * b
  let c := (a + b) / 2
  let wam := (2 * a + 3 * b + 4 * c) / 9
  let gm := (a * b * c) ^ (1/3 : ℝ)
  (wam - gm = b * ((5 * k + 5) / 9 - ((k * (k + 1) * b^2) / 2) ^ (1/3 : ℝ))) ∧
  (wam - gm < ((1 - k)^2 * b) / (8 * k)) := by sorry

end NUMINAMATH_CALUDE_wam_gm_difference_bound_l84_8437


namespace NUMINAMATH_CALUDE_number_difference_l84_8427

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 23405)
  (b_div_5 : ∃ k : ℕ, b = 5 * k)
  (b_div_10_eq_5a : b / 10 = 5 * a) :
  b - a = 21600 :=
by sorry

end NUMINAMATH_CALUDE_number_difference_l84_8427


namespace NUMINAMATH_CALUDE_convex_nonagon_diagonals_l84_8470

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem convex_nonagon_diagonals : 
  ∀ (n : ℕ), n = 9 → nonagon_diagonals = n * (n - 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_convex_nonagon_diagonals_l84_8470


namespace NUMINAMATH_CALUDE_max_golf_rounds_is_eight_l84_8498

/-- Calculates the maximum number of golf rounds that can be played given the specified conditions. -/
def maxGolfRounds (initialCost : ℚ) (membershipFee : ℚ) (budget : ℚ) 
  (discount2nd : ℚ) (discount3rd : ℚ) (discountSubsequent : ℚ) : ℕ :=
  let totalBudget := budget + membershipFee
  let cost1st := initialCost
  let cost2nd := initialCost * (1 - discount2nd)
  let cost3rd := initialCost * (1 - discount3rd)
  let costSubsequent := initialCost * (1 - discountSubsequent)
  let remainingAfter3 := totalBudget - cost1st - cost2nd - cost3rd
  let additionalRounds := (remainingAfter3 / costSubsequent).floor
  3 + additionalRounds.toNat

/-- Theorem stating that the maximum number of golf rounds is 8 under the given conditions. -/
theorem max_golf_rounds_is_eight :
  maxGolfRounds 80 100 400 (1/10) (1/5) (3/10) = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_golf_rounds_is_eight_l84_8498


namespace NUMINAMATH_CALUDE_decimal_2011_equals_base7_5602_l84_8462

/-- Converts a base 10 number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: toBase7 (n / 7)

/-- Converts a list of digits in base 7 to a natural number in base 10 -/
def fromBase7 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 7 * acc + d) 0

theorem decimal_2011_equals_base7_5602 :
  fromBase7 [2, 0, 6, 5] = 2011 :=
by sorry

end NUMINAMATH_CALUDE_decimal_2011_equals_base7_5602_l84_8462


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l84_8444

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l84_8444


namespace NUMINAMATH_CALUDE_triangle_perimeter_l84_8477

/-- Given a triangle with inradius 1.5 cm and area 29.25 cm², its perimeter is 39 cm. -/
theorem triangle_perimeter (inradius : ℝ) (area : ℝ) (perimeter : ℝ) : 
  inradius = 1.5 → area = 29.25 → perimeter = area / inradius * 2 → perimeter = 39 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l84_8477


namespace NUMINAMATH_CALUDE_female_red_ants_percentage_l84_8406

/-- Given an ant colony where 85% of the population is red and 46.75% of the total population
    are male red ants, prove that 45% of the red ants are females. -/
theorem female_red_ants_percentage
  (total_population : ℝ)
  (red_ants_percentage : ℝ)
  (male_red_ants_percentage : ℝ)
  (h1 : red_ants_percentage = 85)
  (h2 : male_red_ants_percentage = 46.75)
  (h3 : total_population > 0) :
  let total_red_ants := red_ants_percentage * total_population / 100
  let male_red_ants := male_red_ants_percentage * total_population / 100
  let female_red_ants := total_red_ants - male_red_ants
  female_red_ants / total_red_ants * 100 = 45 :=
by
  sorry


end NUMINAMATH_CALUDE_female_red_ants_percentage_l84_8406


namespace NUMINAMATH_CALUDE_ruler_cost_l84_8493

theorem ruler_cost (total_students : ℕ) (total_expense : ℕ) :
  total_students = 42 →
  total_expense = 2310 →
  ∃ (num_buyers : ℕ) (rulers_per_student : ℕ) (cost_per_ruler : ℕ),
    num_buyers > total_students / 2 ∧
    cost_per_ruler > rulers_per_student ∧
    num_buyers * rulers_per_student * cost_per_ruler = total_expense ∧
    cost_per_ruler = 11 :=
by sorry


end NUMINAMATH_CALUDE_ruler_cost_l84_8493


namespace NUMINAMATH_CALUDE_smallest_n_cookies_l84_8485

theorem smallest_n_cookies (n : ℕ) : (∀ m : ℕ, m > 0 → (15 * m - 3) % 7 ≠ 0) ∨ 
  ((15 * n - 3) % 7 = 0 ∧ n > 0 ∧ ∀ m : ℕ, 0 < m ∧ m < n → (15 * m - 3) % 7 ≠ 0) ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_cookies_l84_8485


namespace NUMINAMATH_CALUDE_cubic_polynomials_with_constant_difference_l84_8401

/-- Two monic cubic polynomials with specific roots and a constant difference -/
theorem cubic_polynomials_with_constant_difference 
  (f g : ℝ → ℝ) 
  (r : ℝ) 
  (hf : ∃ a : ℝ, ∀ x, f x = (x - (r + 2)) * (x - (r + 8)) * (x - a))
  (hg : ∃ b : ℝ, ∀ x, g x = (x - (r + 4)) * (x - (r + 10)) * (x - b))
  (h_diff : ∀ x, f x - g x = r) :
  r = 32 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_with_constant_difference_l84_8401


namespace NUMINAMATH_CALUDE_f_sum_l84_8441

/-- A function satisfying the given properties -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- f(t) = f(1-t) for all t ∈ ℝ -/
axiom f_symmetry (t : ℝ) : f t = f (1 - t)

/-- f(x) = -x² for x ∈ [0, 1/2] -/
axiom f_def (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1/2) : f x = -x^2

/-- The main theorem to prove -/
theorem f_sum : f 3 + f (-3/2) = -1/4 := by sorry

end NUMINAMATH_CALUDE_f_sum_l84_8441


namespace NUMINAMATH_CALUDE_difference_of_squares_l84_8419

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l84_8419


namespace NUMINAMATH_CALUDE_min_cos_C_in_special_triangle_l84_8403

theorem min_cos_C_in_special_triangle (A B C : ℝ) (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π)
  (h5 : ∃ k : ℝ, (1 / Real.tan A) + k = 2 / Real.tan C ∧
                 2 / Real.tan C + k = 1 / Real.tan B) :
  ∃ (cosC : ℝ), cosC = Real.cos C ∧ cosC ≥ 1/3 ∧
  ∀ (cosC' : ℝ), cosC' = Real.cos C → cosC' ≥ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_min_cos_C_in_special_triangle_l84_8403


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l84_8482

theorem magical_red_knights_fraction (total : ℕ) (total_pos : 0 < total) :
  let red := (2 : ℚ) / 7 * total
  let blue := total - red
  let magical := (1 : ℚ) / 6 * total
  let red_magical_fraction := magical / red
  let blue_magical_fraction := magical / blue
  red_magical_fraction = 2 * blue_magical_fraction →
  red_magical_fraction = 7 / 27 := by
sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l84_8482


namespace NUMINAMATH_CALUDE_xy_commutativity_l84_8442

theorem xy_commutativity (x y : ℝ) : 10 * x * y - 10 * y * x = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_commutativity_l84_8442


namespace NUMINAMATH_CALUDE_nina_taller_than_lena_probability_l84_8405

-- Define the set of friends
inductive Friend
| Masha
| Nina
| Lena
| Olya

-- Define the height relation
def taller_than (a b : Friend) : Prop := sorry

-- Define the conditions
axiom different_heights :
  ∀ (a b : Friend), a ≠ b → (taller_than a b ∨ taller_than b a)

axiom nina_shorter_than_masha :
  taller_than Friend.Masha Friend.Nina

axiom lena_taller_than_olya :
  taller_than Friend.Lena Friend.Olya

-- Define the probability function
noncomputable def probability (event : Prop) : ℝ := sorry

-- Theorem to prove
theorem nina_taller_than_lena_probability :
  probability (taller_than Friend.Nina Friend.Lena) = 0 := by sorry

end NUMINAMATH_CALUDE_nina_taller_than_lena_probability_l84_8405


namespace NUMINAMATH_CALUDE_problem_solution_l84_8433

theorem problem_solution (x y : ℚ) (hx : x = 2/3) (hy : y = 3/2) :
  (3/4 : ℚ) * x^4 * y^5 = 9/8 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l84_8433


namespace NUMINAMATH_CALUDE_line_points_k_value_l84_8443

/-- Given a line with equation x = 2y + 5 and two points (m, n) and (m + 3, n + k) on this line, k = 3/2. -/
theorem line_points_k_value (m n k : ℝ) : 
  (m = 2 * n + 5) → 
  (m + 3 = 2 * (n + k) + 5) → 
  k = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_line_points_k_value_l84_8443


namespace NUMINAMATH_CALUDE_book_pages_calculation_l84_8483

theorem book_pages_calculation (first_day_percent : ℝ) (second_day_percent : ℝ) 
  (third_day_pages : ℕ) :
  first_day_percent = 0.1 →
  second_day_percent = 0.25 →
  (first_day_percent + second_day_percent + (third_day_pages : ℝ) / (240 : ℝ)) = 0.5 →
  third_day_pages = 30 →
  (240 : ℕ) = 240 :=
by sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l84_8483


namespace NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l84_8486

/-- The number of diagonals from a single vertex in a decagon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- Theorem: In a decagon, the number of diagonals from a single vertex is 7 -/
theorem decagon_diagonals_from_vertex : 
  diagonals_from_vertex 10 = 7 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l84_8486


namespace NUMINAMATH_CALUDE_perimeter_equals_127_32_l84_8476

/-- The perimeter of a figure constructed with 6 equilateral triangles, where the first triangle
    has a side length of 1 cm and each subsequent triangle has sides equal to half the length
    of the previous triangle. -/
def perimeter_of_triangles : ℚ :=
  let side_lengths : List ℚ := [1, 1/2, 1/4, 1/8, 1/16, 1/32]
  let unique_segments : List ℚ := [1, 1, 1/2, 1/2, 1/4, 1/4, 1/8, 1/8, 1/16, 1/16, 1/32, 1/32, 1/32]
  unique_segments.sum

theorem perimeter_equals_127_32 : perimeter_of_triangles = 127 / 32 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_equals_127_32_l84_8476
