import Mathlib

namespace NUMINAMATH_CALUDE_tan_theta_value_l2431_243160

theorem tan_theta_value (θ : Real) 
  (h : (Real.sin (π - θ) + Real.cos (θ - 2*π)) / (Real.sin θ + Real.cos (π + θ)) = 1/2) : 
  Real.tan θ = -3 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l2431_243160


namespace NUMINAMATH_CALUDE_snowflake_area_ratio_l2431_243118

/-- Represents the snowflake shape after n iterations --/
def Snowflake (n : ℕ) : Type := Unit

/-- The area of the snowflake shape after n iterations --/
def area (s : Snowflake n) : ℚ := sorry

/-- The initial equilateral triangle --/
def initial_triangle : Snowflake 0 := sorry

/-- The snowflake shape after one iteration --/
def first_iteration : Snowflake 1 := sorry

/-- The snowflake shape after two iterations --/
def second_iteration : Snowflake 2 := sorry

theorem snowflake_area_ratio :
  area second_iteration / area initial_triangle = 40 / 27 := by sorry

end NUMINAMATH_CALUDE_snowflake_area_ratio_l2431_243118


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l2431_243145

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 2 / 3 → 
  Nat.gcd a b = 6 → 
  Nat.lcm a b = 36 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l2431_243145


namespace NUMINAMATH_CALUDE_proposition_correctness_l2431_243150

-- Define the propositions
def prop1 (a b c : ℝ) : Prop := a > b → a * c^2 > b * c^2
def prop2 (a b : ℝ) : Prop := a > |b| → a^2 > b^2
def prop3 (a b : ℝ) : Prop := |a| > b → a^2 > b^2
def prop4 (a b : ℝ) : Prop := a > b → a^3 > b^3

-- Theorem stating the correctness of propositions
theorem proposition_correctness :
  (∃ a b c : ℝ, ¬(prop1 a b c)) ∧
  (∀ a b : ℝ, prop2 a b) ∧
  (∃ a b : ℝ, ¬(prop3 a b)) ∧
  (∀ a b : ℝ, prop4 a b) :=
sorry

end NUMINAMATH_CALUDE_proposition_correctness_l2431_243150


namespace NUMINAMATH_CALUDE_f_satisfies_equation_l2431_243189

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0.5 then 1 / (0.5 - x) else 0.5

theorem f_satisfies_equation :
  ∀ x : ℝ, f x + (0.5 + x) * f (1 - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_equation_l2431_243189


namespace NUMINAMATH_CALUDE_negation_equivalence_l2431_243147

-- Define a triangle
def Triangle : Type := Unit

-- Define an angle in a triangle
def Angle (t : Triangle) : Type := Unit

-- Define the property of being obtuse for an angle
def IsObtuse (t : Triangle) (a : Angle t) : Prop := sorry

-- Define the statement "at most one angle is obtuse"
def AtMostOneObtuse (t : Triangle) : Prop :=
  ∃ (a : Angle t), IsObtuse t a ∧ ∀ (b : Angle t), IsObtuse t b → b = a

-- Define the statement "at least two angles are obtuse"
def AtLeastTwoObtuse (t : Triangle) : Prop :=
  ∃ (a b : Angle t), a ≠ b ∧ IsObtuse t a ∧ IsObtuse t b

-- The theorem stating the negation equivalence
theorem negation_equivalence (t : Triangle) :
  ¬(AtMostOneObtuse t) ↔ AtLeastTwoObtuse t :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2431_243147


namespace NUMINAMATH_CALUDE_correct_calculation_l2431_243116

theorem correct_calculation (x y : ℝ) : 6 * x * y^2 - 3 * y^2 * x = 3 * x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2431_243116


namespace NUMINAMATH_CALUDE_first_complete_shading_l2431_243177

def board_width : ℕ := 10

def shaded_square (n : ℕ) : ℕ := n * n

def is_shaded (square : ℕ) : Prop :=
  ∃ n : ℕ, shaded_square n = square

def column_of_square (square : ℕ) : ℕ :=
  (square - 1) % board_width + 1

theorem first_complete_shading :
  (∀ col : ℕ, col ≤ board_width → 
    ∃ square : ℕ, is_shaded square ∧ column_of_square square = col) ∧
  (∀ smaller : ℕ, smaller < 100 → 
    ¬(∀ col : ℕ, col ≤ board_width → 
      ∃ square : ℕ, square ≤ smaller ∧ is_shaded square ∧ column_of_square square = col)) :=
by sorry

end NUMINAMATH_CALUDE_first_complete_shading_l2431_243177


namespace NUMINAMATH_CALUDE_cos_angle_relation_l2431_243117

theorem cos_angle_relation (α : ℝ) (h : Real.cos (α + π/3) = 4/5) :
  Real.cos (π/3 - 2*α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_relation_l2431_243117


namespace NUMINAMATH_CALUDE_output_for_three_l2431_243172

/-- Represents the output of the program based on the input x -/
def program_output (x : ℤ) : ℤ :=
  if x < 0 then -1
  else if x = 0 then 0
  else 1

/-- Theorem stating that when x = 3, the program outputs 1 -/
theorem output_for_three : program_output 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_output_for_three_l2431_243172


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2431_243169

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 + 3 / (a + b + c)^2 ≥ 2 :=
sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 + 3 / (a + b + c)^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2431_243169


namespace NUMINAMATH_CALUDE_sheila_attend_picnic_l2431_243187

/-- The probability of rain tomorrow -/
def rain_prob : ℝ := 0.5

/-- The probability Sheila decides to go if it rains -/
def go_if_rain : ℝ := 0.4

/-- The probability Sheila decides to go if it's sunny -/
def go_if_sunny : ℝ := 0.9

/-- The probability Sheila finishes her homework -/
def finish_homework : ℝ := 0.7

/-- The overall probability that Sheila attends the picnic -/
def attend_prob : ℝ := rain_prob * go_if_rain * finish_homework + 
                       (1 - rain_prob) * go_if_sunny * finish_homework

theorem sheila_attend_picnic : attend_prob = 0.455 := by
  sorry

end NUMINAMATH_CALUDE_sheila_attend_picnic_l2431_243187


namespace NUMINAMATH_CALUDE_union_covers_reals_l2431_243193

open Set Real

theorem union_covers_reals (A B : Set ℝ) (a : ℝ) :
  A = Iic 0 ∧ B = Ioi a ∧ A ∪ B = univ ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_l2431_243193


namespace NUMINAMATH_CALUDE_min_number_after_operations_l2431_243110

def board_operation (S : Finset ℕ) : Finset ℕ :=
  sorry

def min_after_operations (n : ℕ) : ℕ :=
  sorry

theorem min_number_after_operations :
  (min_after_operations 111 = 0) ∧ (min_after_operations 110 = 1) :=
sorry

end NUMINAMATH_CALUDE_min_number_after_operations_l2431_243110


namespace NUMINAMATH_CALUDE_least_positive_integer_for_zero_sums_l2431_243158

theorem least_positive_integer_for_zero_sums (x₁ x₂ x₃ x₄ x₅ : ℝ) : 
  (∃ (S : Finset (Fin 5 × Fin 5 × Fin 5)), 
    S.card = 7 ∧ 
    (∀ (p q r : Fin 5), (p, q, r) ∈ S → p < q ∧ q < r) ∧
    (∀ (p q r : Fin 5), (p, q, r) ∈ S → x₁ * p.val + x₂ * q.val + x₃ * r.val = 0) →
    x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∧
  (∀ n : ℕ, n < 7 → 
    ∃ (x₁' x₂' x₃' x₄' x₅' : ℝ), 
      ∃ (S : Finset (Fin 5 × Fin 5 × Fin 5)),
        S.card = n ∧
        (∀ (p q r : Fin 5), (p, q, r) ∈ S → p < q ∧ q < r) ∧
        (∀ (p q r : Fin 5), (p, q, r) ∈ S → 
          x₁' * p.val + x₂' * q.val + x₃' * r.val = 0) ∧
        ¬(x₁' = 0 ∧ x₂' = 0 ∧ x₃' = 0 ∧ x₄' = 0 ∧ x₅' = 0)) := by
  sorry


end NUMINAMATH_CALUDE_least_positive_integer_for_zero_sums_l2431_243158


namespace NUMINAMATH_CALUDE_max_value_parabola_l2431_243139

/-- The maximum value of y = -3x^2 + 6, where x is a real number, is 6. -/
theorem max_value_parabola :
  ∃ (M : ℝ), M = 6 ∧ ∀ (x : ℝ), -3 * x^2 + 6 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_parabola_l2431_243139


namespace NUMINAMATH_CALUDE_computer_table_markup_l2431_243183

/-- The percentage markup on a product's cost price, given its selling price and cost price. -/
def percentageMarkup (sellingPrice costPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

/-- Theorem stating that the percentage markup on a computer table with a selling price of 8215 
    and a cost price of 6625 is 24%. -/
theorem computer_table_markup :
  percentageMarkup 8215 6625 = 24 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_markup_l2431_243183


namespace NUMINAMATH_CALUDE_equation_roots_imply_sum_l2431_243181

/-- Given two equations with constants a and b, prove that 100a + b = 156 -/
theorem equation_roots_imply_sum (a b : ℝ) : 
  (∃! x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (x + a) * (x + b) * (x + 12) = 0 ∧
    (y + a) * (y + b) * (y + 12) = 0 ∧
    (z + a) * (z + b) * (z + 12) = 0 ∧
    x ≠ -3 ∧ y ≠ -3 ∧ z ≠ -3) →
  (∃! w, (w + 2*a) * (w + 3) * (w + 6) = 0 ∧ 
    w + b ≠ 0 ∧ w + 12 ≠ 0) →
  100 * a + b = 156 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_imply_sum_l2431_243181


namespace NUMINAMATH_CALUDE_largest_multiple_of_7_less_than_neg_95_l2431_243114

theorem largest_multiple_of_7_less_than_neg_95 :
  ∀ n : ℤ, n * 7 < -95 → n * 7 ≤ -98 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_7_less_than_neg_95_l2431_243114


namespace NUMINAMATH_CALUDE_owen_daily_chores_hours_l2431_243186

/-- 
Given that:
- There are 24 hours in a day
- Owen spends 6 hours at work
- Owen sleeps for 11 hours

Prove that Owen spends 7 hours on other daily chores.
-/
theorem owen_daily_chores_hours : 
  let total_hours : ℕ := 24
  let work_hours : ℕ := 6
  let sleep_hours : ℕ := 11
  total_hours - work_hours - sleep_hours = 7 := by sorry

end NUMINAMATH_CALUDE_owen_daily_chores_hours_l2431_243186


namespace NUMINAMATH_CALUDE_segment_count_is_21_l2431_243112

/-- A configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  triple_intersection : Bool
  triple_intersection_count : ℕ

/-- Calculate the number of non-overlapping line segments in a given configuration -/
def count_segments (config : LineConfiguration) : ℕ :=
  config.num_lines * 4 - config.triple_intersection_count

/-- The specific configuration given in the problem -/
def problem_config : LineConfiguration :=
  { num_lines := 6
  , triple_intersection := true
  , triple_intersection_count := 3 }

/-- Theorem stating that the number of non-overlapping line segments in the given configuration is 21 -/
theorem segment_count_is_21 : count_segments problem_config = 21 := by
  sorry

end NUMINAMATH_CALUDE_segment_count_is_21_l2431_243112


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2431_243113

/-- Given a hyperbola with the standard form equation, prove that under certain conditions, 
    it has a specific equation. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (c : ℝ), c - a = 1 ∧ b = Real.sqrt 3) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2431_243113


namespace NUMINAMATH_CALUDE_cryptarithm_solutions_l2431_243164

def is_valid_solution (tuk : ℕ) (ctuk : ℕ) : Prop :=
  tuk ≥ 100 ∧ tuk < 1000 ∧ ctuk ≥ 1000 ∧ ctuk < 10000 ∧
  5 * tuk = ctuk ∧
  (tuk.digits 10).card = 3 ∧ (ctuk.digits 10).card = 4

theorem cryptarithm_solutions :
  (∀ tuk ctuk : ℕ, is_valid_solution tuk ctuk → (tuk = 250 ∧ ctuk = 1250) ∨ (tuk = 750 ∧ ctuk = 3750)) ∧
  is_valid_solution 250 1250 ∧
  is_valid_solution 750 3750 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solutions_l2431_243164


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l2431_243198

-- Define the triangles and their properties
structure Triangle :=
  (X Y Z : ℝ × ℝ)

def XYZ : Triangle := sorry
def PQR : Triangle := sorry

-- Define the lengths of the sides
def XY : ℝ := 9
def YZ : ℝ := 21
def XZ : ℝ := 15
def PQ : ℝ := 3
def QR : ℝ := 7

-- Define the angles
def angle_XYZ : ℝ := sorry
def angle_PQR : ℝ := sorry

-- State the theorem
theorem similar_triangles_side_length :
  angle_XYZ = angle_PQR →
  XY = 9 →
  XZ = 15 →
  PQ = 3 →
  ∃ (PR : ℝ), PR = 5 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l2431_243198


namespace NUMINAMATH_CALUDE_total_cost_is_14000_l2431_243199

/-- Represents the dimensions and costs of the roads on a rectangular lawn. -/
structure LawnRoads where
  lawn_length : ℝ
  lawn_width : ℝ
  road1_width : ℝ
  road1_cost_per_sqm : ℝ
  road2_width : ℝ
  road2_cost_per_sqm : ℝ
  hill_length : ℝ
  hill_cost_increase : ℝ

/-- Calculates the total cost of traveling both roads on the lawn. -/
def total_cost (lr : LawnRoads) : ℝ :=
  let road1_area := lr.lawn_length * lr.road1_width
  let road1_cost := road1_area * lr.road1_cost_per_sqm
  let hill_area := lr.hill_length * lr.road1_width
  let hill_additional_cost := hill_area * (lr.road1_cost_per_sqm * lr.hill_cost_increase)
  let road2_area := lr.lawn_width * lr.road2_width
  let road2_cost := road2_area * lr.road2_cost_per_sqm
  road1_cost + hill_additional_cost + road2_cost

/-- Theorem stating that the total cost of traveling both roads is 14000. -/
theorem total_cost_is_14000 (lr : LawnRoads) 
    (h1 : lr.lawn_length = 150)
    (h2 : lr.lawn_width = 80)
    (h3 : lr.road1_width = 12)
    (h4 : lr.road1_cost_per_sqm = 4)
    (h5 : lr.road2_width = 8)
    (h6 : lr.road2_cost_per_sqm = 5)
    (h7 : lr.hill_length = 60)
    (h8 : lr.hill_cost_increase = 0.25) :
    total_cost lr = 14000 := by
  sorry


end NUMINAMATH_CALUDE_total_cost_is_14000_l2431_243199


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l2431_243173

/-- The nature of roots of a quadratic equation based on parameters a and b -/
theorem quadratic_roots_nature (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f : ℝ → ℝ := λ x => (a^2 + b^2) * x^2 + 4 * a * b * x + 2 * a * b
  (a = b → (∃! x, f x = 0)) ∧
  (a ≠ b → a * b > 0 → ∀ x, f x ≠ 0) ∧
  (a ≠ b → a * b < 0 → ∃ x y, x ≠ y ∧ f x = 0 ∧ f y = 0) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_nature_l2431_243173


namespace NUMINAMATH_CALUDE_fifth_month_sale_l2431_243153

def sales_first_four : List ℕ := [6435, 6927, 6855, 7230]
def sale_sixth : ℕ := 6191
def average_sale : ℕ := 6700
def num_months : ℕ := 6

theorem fifth_month_sale :
  let total_sales := average_sale * num_months
  let sum_known_sales := sales_first_four.sum + sale_sixth
  total_sales - sum_known_sales = 6562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l2431_243153


namespace NUMINAMATH_CALUDE_original_denominator_proof_l2431_243196

theorem original_denominator_proof (d : ℕ) : 
  (4 : ℚ) / d ≠ 0 →
  (4 + 3 : ℚ) / (d + 3) = 1 / 3 →
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l2431_243196


namespace NUMINAMATH_CALUDE_negation_proofs_l2431_243197

-- Define a multi-digit number
def MultiDigitNumber (n : ℕ) : Prop := n ≥ 10

-- Define the last digit of a number
def LastDigit (n : ℕ) : ℕ := n % 10

-- Define divisibility
def Divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem negation_proofs :
  (∃ n : ℕ, MultiDigitNumber n ∧ LastDigit n ≠ 0 ∧ ¬(Divides 5 n)) = False ∧
  (∃ n : ℕ, Even n ∧ ¬(Divides 2 n)) = False :=
by sorry

end NUMINAMATH_CALUDE_negation_proofs_l2431_243197


namespace NUMINAMATH_CALUDE_unknown_number_solution_l2431_243188

theorem unknown_number_solution : 
  ∃ x : ℚ, (x + 23 / 89) * 89 = 4028 ∧ x = 45 := by sorry

end NUMINAMATH_CALUDE_unknown_number_solution_l2431_243188


namespace NUMINAMATH_CALUDE_tiling_8x1_board_remainder_l2431_243146

/-- Represents a tiling of an 8x1 board -/
structure Tiling :=
  (num_1x1 : ℕ)
  (num_2x1 : ℕ)
  (h_sum : num_1x1 + 2 * num_2x1 = 8)

/-- Calculates the number of valid colorings for a given tiling -/
def validColorings (t : Tiling) : ℕ :=
  3^(t.num_1x1 + t.num_2x1) - 3 * 2^(t.num_1x1 + t.num_2x1) + 3

/-- The set of all possible tilings -/
def allTilings : Finset Tiling :=
  sorry

theorem tiling_8x1_board_remainder (M : ℕ) (h_M : M = (allTilings.sum validColorings)) :
  M % 1000 = 328 :=
sorry

end NUMINAMATH_CALUDE_tiling_8x1_board_remainder_l2431_243146


namespace NUMINAMATH_CALUDE_equation_solution_l2431_243127

theorem equation_solution : 
  ∃ x : ℚ, (24 - 4 = 3 * (1 + x)) ∧ (x = 17 / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2431_243127


namespace NUMINAMATH_CALUDE_mrs_hilt_shortage_l2431_243134

def initial_amount : ℚ := 375 / 100
def pencil_cost : ℚ := 115 / 100
def eraser_cost : ℚ := 85 / 100
def notebook_cost : ℚ := 225 / 100

theorem mrs_hilt_shortage :
  initial_amount - (pencil_cost + eraser_cost + notebook_cost) = -50 / 100 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_shortage_l2431_243134


namespace NUMINAMATH_CALUDE_fraction_problem_l2431_243108

theorem fraction_problem (n : ℚ) (f : ℚ) (h1 : n = 120) (h2 : (1/2) * f * n = 36) : f = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2431_243108


namespace NUMINAMATH_CALUDE_tina_pens_count_l2431_243155

/-- Calculates the total number of pens Tina has given the number of pink pens and the relationships between different colored pens. -/
def total_pens (pink : ℕ) (green_diff : ℕ) (blue_diff : ℕ) : ℕ :=
  pink + (pink - green_diff) + ((pink - green_diff) + blue_diff)

/-- Proves that given the conditions, Tina has 21 pens in total. -/
theorem tina_pens_count : total_pens 12 9 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_tina_pens_count_l2431_243155


namespace NUMINAMATH_CALUDE_satisfying_polynomial_iff_quadratic_l2431_243120

/-- A polynomial that satisfies the given functional equation -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, P (a + b - 2*c) + P (b + c - 2*a) + P (a + c - 2*b) = 
               3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

/-- The theorem stating the equivalence between the functional equation and the quadratic form -/
theorem satisfying_polynomial_iff_quadratic :
  ∀ P : ℝ → ℝ, SatisfyingPolynomial P ↔ 
    ∃ a b : ℝ, ∀ x : ℝ, P x = a * x^2 + b * x :=
by sorry

end NUMINAMATH_CALUDE_satisfying_polynomial_iff_quadratic_l2431_243120


namespace NUMINAMATH_CALUDE_rectangle_circle_square_area_l2431_243166

theorem rectangle_circle_square_area : 
  ∀ (r : ℝ) (l w : ℝ),
    r = 7 →  -- Circle radius
    l = 3 * w →  -- Rectangle length to width ratio
    2 * r = w →  -- Circle diameter equals rectangle width
    l * w + 2 * r^2 = 686 :=  -- Total area of rectangle and square
by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_square_area_l2431_243166


namespace NUMINAMATH_CALUDE_locus_of_equidistant_points_l2431_243184

-- Define the oblique coordinate system
structure ObliqueCoordSystem where
  angle : ℝ
  e₁ : ℝ × ℝ
  e₂ : ℝ × ℝ

-- Define a point in the oblique coordinate system
structure ObliquePoint where
  x : ℝ
  y : ℝ

-- Define the locus equation
def locusEquation (p : ObliquePoint) : Prop :=
  Real.sqrt 2 * p.x + p.y = 0

-- State the theorem
theorem locus_of_equidistant_points
  (sys : ObliqueCoordSystem)
  (F₁ F₂ M : ObliquePoint)
  (h_angle : sys.angle = Real.pi / 4)
  (h_F₁ : F₁ = ⟨-1, 0⟩)
  (h_F₂ : F₂ = ⟨1, 0⟩)
  (h_equidistant : ‖(M.x - F₁.x, M.y - F₁.y)‖ = ‖(M.x - F₂.x, M.y - F₂.y)‖) :
  locusEquation M :=
sorry

end NUMINAMATH_CALUDE_locus_of_equidistant_points_l2431_243184


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2431_243195

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_a1 : a 1 = -1) 
  (h_sum : a 2 + a 3 = -2) :
  ∃ q : ℝ, (q = -2 ∨ q = 1) ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2431_243195


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_average_l2431_243111

theorem consecutive_odd_integers_average (n : ℕ) (first : ℤ) :
  n = 10 →
  first = 145 →
  first % 2 = 1 →
  let sequence := List.range n |>.map (λ i => first + 2 * i)
  (sequence.sum / n : ℚ) = 154 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_average_l2431_243111


namespace NUMINAMATH_CALUDE_robins_count_l2431_243174

theorem robins_count (total : ℕ) (robins penguins pigeons : ℕ) : 
  robins = 2 * total / 3 →
  penguins = total / 8 →
  pigeons = 5 →
  total = robins + penguins + pigeons →
  robins = 16 := by
sorry

end NUMINAMATH_CALUDE_robins_count_l2431_243174


namespace NUMINAMATH_CALUDE_integer_divisibility_problem_l2431_243107

theorem integer_divisibility_problem (n : ℤ) :
  (∃ k : ℤ, n - 4 = 6 * k) ∧ (∃ m : ℤ, n - 8 = 10 * m) →
  n ≡ 28 [ZMOD 30] := by
  sorry

end NUMINAMATH_CALUDE_integer_divisibility_problem_l2431_243107


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l2431_243142

theorem tank_capacity_proof (tank_capacity : ℕ) (large_bucket_count : ℕ) (large_bucket_capacity : ℕ) (small_bucket_count : ℕ) :
  tank_capacity = large_bucket_count * large_bucket_capacity →
  tank_capacity = small_bucket_count * (tank_capacity / small_bucket_count) →
  large_bucket_count = 12 →
  large_bucket_capacity = 81 →
  small_bucket_count = 108 →
  tank_capacity / small_bucket_count = 9 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l2431_243142


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2431_243103

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2431_243103


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2431_243144

theorem perfect_square_condition (n : ℕ) : 
  ∃ m : ℕ, n^5 - n^4 - 2*n^3 + 2*n^2 + n - 1 = m^2 ↔ ∃ k : ℕ, n = k^2 + 1 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2431_243144


namespace NUMINAMATH_CALUDE_cubic_polynomial_properties_l2431_243129

/-- The cubic polynomial f(x) = x³ + px + q -/
noncomputable def f (p q x : ℝ) : ℝ := x^3 + p*x + q

theorem cubic_polynomial_properties (p q : ℝ) :
  (p ≥ 0 → ∀ x y : ℝ, x < y → f p q x < f p q y) ∧ 
  (p < 0 → ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f p q x = 0 ∧ f p q y = 0 ∧ f p q z = 0) ∧
  (p < 0 → ∃! x y : ℝ, x ≠ y ∧ (∀ z : ℝ, f p q x ≤ f p q z) ∧ (∀ z : ℝ, f p q y ≥ f p q z)) ∧
  (p < 0 → ∃ x y : ℝ, x ≠ y ∧ (∀ z : ℝ, f p q x ≤ f p q z) ∧ (∀ z : ℝ, f p q y ≥ f p q z) ∧ x = -y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_properties_l2431_243129


namespace NUMINAMATH_CALUDE_emily_buys_12_cucumbers_l2431_243152

/-- The cost of one apple -/
def apple_cost : ℝ := sorry

/-- The cost of one banana -/
def banana_cost : ℝ := sorry

/-- The cost of one cucumber -/
def cucumber_cost : ℝ := sorry

/-- Six apples cost the same as three bananas -/
axiom six_apples_eq_three_bananas : 6 * apple_cost = 3 * banana_cost

/-- Three bananas cost the same as four cucumbers -/
axiom three_bananas_eq_four_cucumbers : 3 * banana_cost = 4 * cucumber_cost

/-- The number of cucumbers Emily can buy for the price of 18 apples -/
def cucumbers_for_18_apples : ℕ := sorry

/-- Proof that Emily can buy 12 cucumbers for the price of 18 apples -/
theorem emily_buys_12_cucumbers : cucumbers_for_18_apples = 12 := by
  sorry

end NUMINAMATH_CALUDE_emily_buys_12_cucumbers_l2431_243152


namespace NUMINAMATH_CALUDE_computer_price_increase_l2431_243178

theorem computer_price_increase (d : ℝ) : 
  2 * d = 560 →
  ((364 - d) / d) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_computer_price_increase_l2431_243178


namespace NUMINAMATH_CALUDE_cubic_tangent_perpendicular_l2431_243132

/-- Given a cubic function f(x) = ax³ + x + 1, if its tangent line at x = 1 is
    perpendicular to the line x + 4y = 0, then a = 1. -/
theorem cubic_tangent_perpendicular (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + x + 1
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 1
  (f' 1) * (-1/4) = -1 →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_cubic_tangent_perpendicular_l2431_243132


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_seven_sqrt_two_over_six_l2431_243162

theorem sqrt_difference_equals_seven_sqrt_two_over_six :
  Real.sqrt (9 / 2) - Real.sqrt (2 / 9) = 7 * Real.sqrt 2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_seven_sqrt_two_over_six_l2431_243162


namespace NUMINAMATH_CALUDE_negative_of_negative_five_greater_than_negative_five_l2431_243179

theorem negative_of_negative_five_greater_than_negative_five : -(-5) > -5 := by
  sorry

end NUMINAMATH_CALUDE_negative_of_negative_five_greater_than_negative_five_l2431_243179


namespace NUMINAMATH_CALUDE_square_of_119_l2431_243121

theorem square_of_119 : 119^2 = 14161 := by
  sorry

end NUMINAMATH_CALUDE_square_of_119_l2431_243121


namespace NUMINAMATH_CALUDE_box_length_proof_l2431_243194

/-- Proves that a rectangular box with given dimensions and fill rate has a specific length -/
theorem box_length_proof (fill_rate : ℝ) (width depth time : ℝ) (h1 : fill_rate = 4)
    (h2 : width = 6) (h3 : depth = 2) (h4 : time = 21) :
  (fill_rate * time) / (width * depth) = 7 := by
  sorry

end NUMINAMATH_CALUDE_box_length_proof_l2431_243194


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2431_243175

theorem division_remainder_proof (dividend : Nat) (divisor : Nat) (quotient : Nat) 
    (h1 : dividend = 131)
    (h2 : divisor = 14)
    (h3 : quotient = 9)
    (h4 : dividend = divisor * quotient + (dividend % divisor)) :
  dividend % divisor = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2431_243175


namespace NUMINAMATH_CALUDE_binomial_coefficient_60_2_l2431_243105

theorem binomial_coefficient_60_2 : Nat.choose 60 2 = 1770 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_60_2_l2431_243105


namespace NUMINAMATH_CALUDE_inequality_proof_l2431_243192

theorem inequality_proof (x : ℝ) :
  x ≥ Real.rpow 7 (1/3) / Real.rpow 2 (1/3) ∧
  x < Real.rpow 373 (1/3) / Real.rpow 72 (1/3) →
  Real.sqrt (2*x + 7/x^2) + Real.sqrt (2*x - 7/x^2) < 6/x :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2431_243192


namespace NUMINAMATH_CALUDE_total_score_three_probability_l2431_243122

def yellow_balls : ℕ := 2
def white_balls : ℕ := 3
def total_balls : ℕ := yellow_balls + white_balls

def yellow_score : ℕ := 1
def white_score : ℕ := 2

def prob_yellow (balls_left : ℕ) : ℚ := yellow_balls / balls_left
def prob_white (balls_left : ℕ) : ℚ := white_balls / balls_left

theorem total_score_three_probability :
  (prob_yellow total_balls * prob_white (total_balls - 1) +
   prob_white total_balls * prob_yellow (total_balls - 1)) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_total_score_three_probability_l2431_243122


namespace NUMINAMATH_CALUDE_vector_operation_l2431_243136

/-- Given plane vectors a and b, prove that -2a - b equals (-3, -1) --/
theorem vector_operation (a b : ℝ × ℝ) : 
  a = (1, 1) → b = (1, -1) → -2 • a - b = (-3, -1) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l2431_243136


namespace NUMINAMATH_CALUDE_min_value_theorem_range_theorem_l2431_243182

-- Define the variables and conditions
variable (a b : ℝ) (hsum : a + b = 1) (ha : a > 0) (hb : b > 0)

-- Part I: Minimum value theorem
theorem min_value_theorem : 
  ∃ (min : ℝ), min = 9 ∧ ∀ (a b : ℝ), a + b = 1 → a > 0 → b > 0 → 1/a + 4/b ≥ min :=
sorry

-- Part II: Range theorem
theorem range_theorem :
  ∀ (x : ℝ), (∀ (a b : ℝ), a + b = 1 → a > 0 → b > 0 → 1/a + 4/b ≥ |2*x - 1| - |x + 1|) ↔ x ∈ Set.Icc (-7) 11 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_range_theorem_l2431_243182


namespace NUMINAMATH_CALUDE_pants_and_belt_cost_l2431_243126

/-- The total cost of a pair of pants and a belt, given their prices -/
def total_cost (pants_price belt_price : ℝ) : ℝ := pants_price + belt_price

theorem pants_and_belt_cost :
  let pants_price : ℝ := 34.0
  let belt_price : ℝ := pants_price + 2.93
  total_cost pants_price belt_price = 70.93 := by
sorry

end NUMINAMATH_CALUDE_pants_and_belt_cost_l2431_243126


namespace NUMINAMATH_CALUDE_circle_configuration_theorem_l2431_243180

/-- Represents a configuration of three circles tangent to each other and a line -/
structure CircleConfiguration where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : 0 < c
  h2 : c < b
  h3 : b < a

/-- The relation between radii of mutually tangent circles according to Descartes' theorem -/
def descartes_relation (config : CircleConfiguration) : Prop :=
  ((1 / config.a + 1 / config.b + 1 / config.c) ^ 2 : ℝ) = 
  2 * ((1 / config.a ^ 2 + 1 / config.b ^ 2 + 1 / config.c ^ 2) : ℝ)

/-- A configuration is nice if all radii are integers -/
def is_nice (config : CircleConfiguration) : Prop :=
  ∃ (i j k : ℕ), (config.a = i) ∧ (config.b = j) ∧ (config.c = k)

theorem circle_configuration_theorem :
  ∀ (config : CircleConfiguration),
  descartes_relation config →
  (∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
    (config.a = 16 ∧ config.b = 4 → |config.c - 40| < ε)) ∧
  (∀ (nice_config : CircleConfiguration),
    is_nice nice_config → descartes_relation nice_config → 
    nice_config.c ≥ 2) ∧
  (∃ (nice_config : CircleConfiguration),
    is_nice nice_config ∧ descartes_relation nice_config ∧ nice_config.c = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_configuration_theorem_l2431_243180


namespace NUMINAMATH_CALUDE_meeting_point_distance_l2431_243185

/-- 
Given two people walking towards each other from a distance of 50 km, 
with one person walking at 4 km/h and the other at 6 km/h, 
the distance traveled by the slower person when they meet is 20 km.
-/
theorem meeting_point_distance 
  (total_distance : ℝ) 
  (speed_a : ℝ) 
  (speed_b : ℝ) 
  (h1 : total_distance = 50) 
  (h2 : speed_a = 4) 
  (h3 : speed_b = 6) : 
  (total_distance * speed_a) / (speed_a + speed_b) = 20 := by
sorry

end NUMINAMATH_CALUDE_meeting_point_distance_l2431_243185


namespace NUMINAMATH_CALUDE_no_solution_iff_m_geq_two_l2431_243176

theorem no_solution_iff_m_geq_two (m : ℝ) :
  (∀ x : ℝ, ¬(x < m + 1 ∧ x > 2*m - 1)) ↔ m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_geq_two_l2431_243176


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_2pi_3_l2431_243140

theorem cos_2alpha_minus_2pi_3 (α : Real) 
  (h : Real.sin (α + π / 6) = 1 / 3) : 
  Real.cos (2 * α - 2 * π / 3) = - 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_2pi_3_l2431_243140


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2431_243101

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2431_243101


namespace NUMINAMATH_CALUDE_evaluate_expression_l2431_243109

theorem evaluate_expression : (4^4 - 4*(4-1)^4)^4 = 21381376 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2431_243109


namespace NUMINAMATH_CALUDE_c_profit_is_400_l2431_243100

/-- Represents the investment and profit distribution for three individuals --/
structure BusinessInvestment where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ

/-- Calculates C's share of the profit based on the given investments and total profit --/
def c_profit_share (investment : BusinessInvestment) : ℕ :=
  (investment.c_investment * investment.total_profit) / (investment.a_investment + investment.b_investment + investment.c_investment)

/-- Theorem stating that C's share of the profit is 400 given the specific investments and total profit --/
theorem c_profit_is_400 (investment : BusinessInvestment)
  (h1 : investment.a_investment = 800)
  (h2 : investment.b_investment = 1000)
  (h3 : investment.c_investment = 1200)
  (h4 : investment.total_profit = 1000) :
  c_profit_share investment = 400 := by
  sorry

#eval c_profit_share ⟨800, 1000, 1200, 1000⟩

end NUMINAMATH_CALUDE_c_profit_is_400_l2431_243100


namespace NUMINAMATH_CALUDE_vector_parallel_sum_l2431_243167

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

theorem vector_parallel_sum (m : ℝ) : 
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (3, m)
  parallel a (a.1 + b.1, a.2 + b.2) → m = 3 := by
sorry

end NUMINAMATH_CALUDE_vector_parallel_sum_l2431_243167


namespace NUMINAMATH_CALUDE_count_valid_arrangements_l2431_243157

/-- Represents a valid arrangement of multiples of 2013 in a table -/
def ValidArrangement : Type :=
  { arr : Fin 11 → Fin 11 // Function.Injective arr ∧ 
    ∀ i : Fin 11, (2013 * (arr i + 1)) % (i + 1) = 0 }

/-- The number of valid arrangements -/
def numValidArrangements : ℕ := sorry

theorem count_valid_arrangements : numValidArrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_arrangements_l2431_243157


namespace NUMINAMATH_CALUDE_percent_to_decimal_five_percent_to_decimal_l2431_243161

theorem percent_to_decimal (p : ℝ) : p / 100 = p * 0.01 := by sorry

theorem five_percent_to_decimal : (5 : ℝ) / 100 = 0.05 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_five_percent_to_decimal_l2431_243161


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2431_243138

/-- Calculates the length of a bridge given the parameters of an elephant train passing through it. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_cm : ℝ) (time_to_pass : ℝ) : 
  train_length = 15 →
  train_speed_cm = 275 →
  time_to_pass = 48 →
  (train_speed_cm / 100 * time_to_pass) - train_length = 117 := by
  sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l2431_243138


namespace NUMINAMATH_CALUDE_range_of_g_l2431_243191

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - 3

-- Define the function g as a composition of f five times
def g (x : ℝ) : ℝ := f (f (f (f (f x))))

-- State the theorem
theorem range_of_g :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 →
  ∃ y : ℝ, g x = y ∧ -1023 ≤ y ∧ y ≤ 2049 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l2431_243191


namespace NUMINAMATH_CALUDE_original_orange_price_l2431_243148

theorem original_orange_price 
  (price_increase : ℝ) 
  (original_mango_price : ℝ) 
  (new_total_cost : ℝ) 
  (h1 : price_increase = 0.15)
  (h2 : original_mango_price = 50)
  (h3 : new_total_cost = 1035) :
  ∃ (original_orange_price : ℝ),
    original_orange_price = 40 ∧
    new_total_cost = 10 * (original_orange_price * (1 + price_increase)) + 
                     10 * (original_mango_price * (1 + price_increase)) :=
by sorry

end NUMINAMATH_CALUDE_original_orange_price_l2431_243148


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2431_243168

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ  -- coefficient of x

def is_on_parabola (p : Point) (par : Parabola) : Prop :=
  p.y^2 = 4 * par.a * p.x

def is_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

def is_focus (f : Point) (par : Parabola) : Prop :=
  f.x = par.a ∧ f.y = 0

def is_on_circle_diameter (a : Point) (p : Point) (q : Point) : Prop :=
  (p.x - a.x) * (q.x - a.x) + (p.y - a.y) * (q.y - a.y) = 0

theorem parabola_line_intersection 
  (par : Parabola) (l : Line) (f p q : Point) (h_focus : is_focus f par)
  (h_line_through_focus : is_on_line f l)
  (h_p_on_parabola : is_on_parabola p par) (h_p_on_line : is_on_line p l)
  (h_q_on_parabola : is_on_parabola q par) (h_q_on_line : is_on_line q l)
  (h_circle : is_on_circle_diameter ⟨-1, 1⟩ p q) :
  l.m = 1/2 ∧ l.b = -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2431_243168


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2431_243124

theorem complex_equation_sum (a b : ℝ) :
  (a - 2 * Complex.I^3) / (b + Complex.I) = Complex.I → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2431_243124


namespace NUMINAMATH_CALUDE_max_tan_B_in_triangle_l2431_243133

/-- In a triangle ABC, given that 3a*cos(C) + b = 0, the maximum value of tan(B) is 3/4 -/
theorem max_tan_B_in_triangle (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  3 * a * Real.cos C + b = 0 →
  ∀ (a' b' c' : ℝ) (A' B' C' : ℝ),
    a' > 0 → b' > 0 → c' > 0 →
    A' > 0 → B' > 0 → C' > 0 →
    A' + B' + C' = Real.pi →
    3 * a' * Real.cos C' + b' = 0 →
    Real.tan B ≤ Real.tan B' →
  Real.tan B ≤ 3/4 :=
by sorry

end NUMINAMATH_CALUDE_max_tan_B_in_triangle_l2431_243133


namespace NUMINAMATH_CALUDE_managers_salary_managers_salary_proof_l2431_243170

/-- The manager's salary problem -/
theorem managers_salary (num_employees : ℕ) (initial_avg_salary : ℕ) (salary_increase : ℕ) : ℕ :=
  let total_initial_salary := num_employees * initial_avg_salary
  let new_avg_salary := initial_avg_salary + salary_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_initial_salary

/-- Proof of the manager's salary -/
theorem managers_salary_proof :
  managers_salary 50 2500 1500 = 79000 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_managers_salary_proof_l2431_243170


namespace NUMINAMATH_CALUDE_adoption_cost_calculation_l2431_243106

/-- Calculates the total cost of preparing animals for adoption --/
def total_adoption_cost (cat_prep_cost adult_dog_prep_cost puppy_prep_cost : ℕ) 
                        (num_cats num_adult_dogs num_puppies : ℕ)
                        (additional_costs : List ℝ) : ℝ :=
  (cat_prep_cost * num_cats + 
   adult_dog_prep_cost * num_adult_dogs + 
   puppy_prep_cost * num_puppies : ℝ) + 
  additional_costs.sum

/-- Theorem stating the total cost for the given scenario --/
theorem adoption_cost_calculation 
  (cat_prep_cost : ℕ) (adult_dog_prep_cost : ℕ) (puppy_prep_cost : ℕ)
  (x1 x2 x3 x4 x5 x6 x7 : ℝ) :
  cat_prep_cost = 50 →
  adult_dog_prep_cost = 100 →
  puppy_prep_cost = 150 →
  total_adoption_cost cat_prep_cost adult_dog_prep_cost puppy_prep_cost 2 3 2 [x1, x2, x3, x4, x5, x6, x7] = 
    700 + x1 + x2 + x3 + x4 + x5 + x6 + x7 :=
by
  sorry

end NUMINAMATH_CALUDE_adoption_cost_calculation_l2431_243106


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2431_243130

theorem complex_equation_solution :
  ∀ (a b : ℝ), (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2431_243130


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l2431_243171

/-- The fraction of homes on Gotham Street that are termite-ridden -/
def termite_ridden_fraction : ℚ := 1/3

/-- The fraction of termite-ridden homes that are collapsing -/
def collapsing_fraction : ℚ := 7/10

/-- Theorem: The fraction of homes that are termite-ridden but not collapsing is 1/10 -/
theorem termite_ridden_not_collapsing : 
  termite_ridden_fraction - (termite_ridden_fraction * collapsing_fraction) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l2431_243171


namespace NUMINAMATH_CALUDE_value_of_a_l2431_243163

theorem value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 15 - 7 * a) : a = 15 / 11 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2431_243163


namespace NUMINAMATH_CALUDE_third_grade_students_l2431_243137

theorem third_grade_students (total_students : ℕ) (sample_size : ℕ) (first_grade_sample : ℕ) (second_grade_sample : ℕ) :
  total_students = 2000 →
  sample_size = 100 →
  first_grade_sample = 30 →
  second_grade_sample = 30 →
  (∃ third_grade_students : ℕ,
    third_grade_students = 800 ∧
    third_grade_students = (total_students * (sample_size - first_grade_sample - second_grade_sample)) / sample_size) :=
by
  sorry

end NUMINAMATH_CALUDE_third_grade_students_l2431_243137


namespace NUMINAMATH_CALUDE_arithmetic_sequence_iff_c_eq_neg_one_l2431_243159

/-- Definition of the sum of the first n terms of the sequence -/
def S (n : ℕ) (c : ℝ) : ℝ := (n + 1)^2 + c

/-- Definition of the nth term of the sequence -/
def a (n : ℕ) (c : ℝ) : ℝ := S n c - S (n - 1) c

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Theorem: The sequence is arithmetic if and only if c = -1 -/
theorem arithmetic_sequence_iff_c_eq_neg_one (c : ℝ) :
  is_arithmetic_sequence (a · c) ↔ c = -1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_iff_c_eq_neg_one_l2431_243159


namespace NUMINAMATH_CALUDE_lecture_theorem_l2431_243125

def lecture_problem (total_duration : ℝ) (total_audience : ℕ) 
  (full_lecture_percent : ℝ) (slept_percent : ℝ) 
  (half_lecture_percent : ℝ) (quarter_lecture_percent : ℝ) : Prop :=
  let full_lecture := (full_lecture_percent / 100) * total_audience
  let slept := (slept_percent / 100) * total_audience
  let remaining := total_audience - full_lecture - slept
  let half_lecture := (half_lecture_percent / 100) * remaining
  let quarter_lecture := remaining - half_lecture
  let total_minutes := full_lecture * total_duration + 
                       half_lecture * (total_duration / 2) + 
                       quarter_lecture * (total_duration / 4)
  let average_minutes := total_minutes / total_audience
  average_minutes = 47.5

theorem lecture_theorem : 
  lecture_problem 90 200 30 5 40 60 :=
sorry

end NUMINAMATH_CALUDE_lecture_theorem_l2431_243125


namespace NUMINAMATH_CALUDE_fraction_modification_l2431_243190

theorem fraction_modification (a : ℕ) : (29 - a : ℚ) / (43 + a) = 3/5 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_modification_l2431_243190


namespace NUMINAMATH_CALUDE_man_walking_running_time_l2431_243156

/-- Given a man who walks at 5 km/h for 5 hours, prove that the time taken to cover the same distance when running at 15 km/h is 1.6667 hours. -/
theorem man_walking_running_time (walking_speed : ℝ) (walking_time : ℝ) (running_speed : ℝ) :
  walking_speed = 5 →
  walking_time = 5 →
  running_speed = 15 →
  (walking_speed * walking_time) / running_speed = 1.6667 := by
  sorry

#eval (5 * 5) / 15

end NUMINAMATH_CALUDE_man_walking_running_time_l2431_243156


namespace NUMINAMATH_CALUDE_isosceles_triangle_third_side_l2431_243165

theorem isosceles_triangle_third_side 
  (a b c : ℝ) 
  (h_isosceles : (a = b ∧ c = 5) ∨ (a = c ∧ b = 5) ∨ (b = c ∧ a = 5)) 
  (h_side : a = 2 ∨ b = 2 ∨ c = 2) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a = 5 ∨ b = 5 ∨ c = 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_third_side_l2431_243165


namespace NUMINAMATH_CALUDE_remainder_problem_l2431_243143

theorem remainder_problem (N : ℕ) 
  (h1 : ∃ (R : ℕ), N = 68 * 269 + R ∧ R < 68) 
  (h2 : ∃ (Q : ℕ), N = 67 * Q + 1) : 
  N % 68 = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2431_243143


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2431_243102

/-- A geometric sequence with positive terms -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsPositiveGeometricSequence a →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2431_243102


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2431_243151

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 4 < 0) → (a < -4 ∨ a > 4) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2431_243151


namespace NUMINAMATH_CALUDE_jims_out_of_pocket_l2431_243135

/-- The cost of Jim's first wedding ring in dollars -/
def first_ring_cost : ℕ := 10000

/-- The cost of Jim's wife's ring in dollars -/
def second_ring_cost : ℕ := 2 * first_ring_cost

/-- The selling price of Jim's first ring in dollars -/
def first_ring_selling_price : ℕ := first_ring_cost / 2

/-- Jim's total out-of-pocket expense in dollars -/
def total_out_of_pocket : ℕ := second_ring_cost + (first_ring_cost - first_ring_selling_price)

/-- Theorem stating Jim's total out-of-pocket expense -/
theorem jims_out_of_pocket : total_out_of_pocket = 25000 := by
  sorry

end NUMINAMATH_CALUDE_jims_out_of_pocket_l2431_243135


namespace NUMINAMATH_CALUDE_circle_points_equidistant_l2431_243141

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point is on the circle if its distance from the center equals the radius -/
def IsOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem circle_points_equidistant (c : Circle) (p : ℝ × ℝ) :
  IsOnCircle c p → (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_points_equidistant_l2431_243141


namespace NUMINAMATH_CALUDE_log_216_equals_3_log_2_plus_3_log_3_l2431_243154

theorem log_216_equals_3_log_2_plus_3_log_3 :
  Real.log 216 = 3 * (Real.log 2 + Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_log_216_equals_3_log_2_plus_3_log_3_l2431_243154


namespace NUMINAMATH_CALUDE_g_neg_one_equals_neg_one_l2431_243104

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of y being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) + (-x)^2 = -(f x + x^2)

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- Theorem statement
theorem g_neg_one_equals_neg_one
  (h1 : is_odd_function f)
  (h2 : f 1 = 1) :
  g f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_one_equals_neg_one_l2431_243104


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2431_243131

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p₁ p₂ p₃ : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), p₃ = (1 - t₁ - t₂) • p₁ + t₁ • p₂ + t₂ • p₃

/-- If the points (2, a, b), (a, 3, b), and (a, b, 4) are collinear, then a + b = 6. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l2431_243131


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l2431_243128

/-- Represents a character in the message -/
inductive Character
| Letter (c : Char)
| ExclamationMark

/-- Represents the state of the game board -/
def Board := List Character

/-- Represents a valid move in the game -/
inductive Move
| EraseSingle (c : Character)
| EraseMultiple (c : Char) (n : Nat)

/-- Applies a move to the board -/
def applyMove (board : Board) (move : Move) : Board :=
  sorry

/-- Checks if the game is over (no more characters to erase) -/
def isGameOver (board : Board) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := Board → Move

/-- Checks if a strategy is winning for the current player -/
def isWinningStrategy (strategy : Strategy) (board : Board) : Bool :=
  sorry

theorem second_player_winning_strategy 
  (initialBoard : Board) : 
  ∃ (strategy : Strategy), isWinningStrategy strategy initialBoard :=
sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l2431_243128


namespace NUMINAMATH_CALUDE_unique_tangent_line_l2431_243119

/-- The function f(x) = x^4 + 4x^3 - 26x^2 -/
def f (x : ℝ) : ℝ := x^4 + 4*x^3 - 26*x^2

/-- The line L(x) = 60x - 225 -/
def L (x : ℝ) : ℝ := 60*x - 225

theorem unique_tangent_line :
  ∃! (a b : ℝ), 
    (∀ x : ℝ, f x ≥ a*x + b ∨ f x ≤ a*x + b) ∧ 
    (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a*x₁ + b ∧ f x₂ = a*x₂ + b) ∧
    a = 60 ∧ b = -225 :=
sorry

end NUMINAMATH_CALUDE_unique_tangent_line_l2431_243119


namespace NUMINAMATH_CALUDE_tshirt_cost_calculation_l2431_243149

def sweatshirt_cost : ℕ := 15
def num_sweatshirts : ℕ := 3
def num_tshirts : ℕ := 2
def total_spent : ℕ := 65

theorem tshirt_cost_calculation :
  ∃ (tshirt_cost : ℕ), 
    num_sweatshirts * sweatshirt_cost + num_tshirts * tshirt_cost = total_spent ∧
    tshirt_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_cost_calculation_l2431_243149


namespace NUMINAMATH_CALUDE_largest_square_from_wire_l2431_243123

/-- Given a wire of length 28 centimeters forming the largest possible square,
    the length of one side of the square is 7 centimeters. -/
theorem largest_square_from_wire (wire_length : ℝ) (side_length : ℝ) :
  wire_length = 28 →
  side_length * 4 = wire_length →
  side_length = 7 := by sorry

end NUMINAMATH_CALUDE_largest_square_from_wire_l2431_243123


namespace NUMINAMATH_CALUDE_equations_not_intersecting_at_roots_l2431_243115

theorem equations_not_intersecting_at_roots : ∀ (x : ℝ),
  (x = 0 ∨ x = 3) →
  (x = x - 3) →
  False :=
by sorry

#check equations_not_intersecting_at_roots

end NUMINAMATH_CALUDE_equations_not_intersecting_at_roots_l2431_243115
