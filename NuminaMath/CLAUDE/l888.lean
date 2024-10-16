import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l888_88883

/-- Given a geometric sequence {a_n} with common ratio q < 0,
    if a_2 = 1 - a_1 and a_4 = 4 - a_3, then a_4 + a_5 = 16 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : q < 0)
  (h2 : ∀ n, a (n + 1) = q * a n)  -- Definition of geometric sequence
  (h3 : a 2 = 1 - a 1)
  (h4 : a 4 = 4 - a 3) :
  a 4 + a 5 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l888_88883


namespace NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l888_88862

/-- Given a line with slope 4 and x-intercept 2, its equation is 4x - y - 8 = 0 -/
theorem line_equation_from_slope_and_intercept :
  ∀ (f : ℝ → ℝ), 
    (∀ x y, f y = 4 * (x - 2)) →  -- slope is 4, x-intercept is 2
    (f 0 = -8) →                  -- y-intercept is -8
    ∀ x, 4 * x - f x - 8 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l888_88862


namespace NUMINAMATH_CALUDE_crypto_encoding_theorem_l888_88838

/-- Represents the digits in the cryptographic encoding -/
inductive CryptoDigit
| V
| W
| X
| Y
| Z

/-- Represents a number in the cryptographic encoding -/
def CryptoNumber := List CryptoDigit

/-- Converts a CryptoNumber to its base 5 representation -/
def toBase5 : CryptoNumber → Nat := sorry

/-- Converts a base 5 number to base 10 -/
def base5ToBase10 : Nat → Nat := sorry

/-- The theorem to be proved -/
theorem crypto_encoding_theorem 
  (encode : Nat → CryptoNumber) 
  (n : Nat) :
  encode n = [CryptoDigit.V, CryptoDigit.Y, CryptoDigit.Z] ∧
  encode (n + 1) = [CryptoDigit.V, CryptoDigit.Y, CryptoDigit.X] ∧
  encode (n + 2) = [CryptoDigit.V, CryptoDigit.V, CryptoDigit.W] →
  base5ToBase10 (toBase5 [CryptoDigit.X, CryptoDigit.Y, CryptoDigit.Z]) = 108 := by
  sorry

end NUMINAMATH_CALUDE_crypto_encoding_theorem_l888_88838


namespace NUMINAMATH_CALUDE_apple_ratio_l888_88874

/-- The number of apples Billy ate in a week -/
def total_apples : ℕ := 20

/-- The number of apples Billy ate on Monday -/
def monday_apples : ℕ := 2

/-- The number of apples Billy ate on Tuesday -/
def tuesday_apples : ℕ := 2 * monday_apples

/-- The number of apples Billy ate on Wednesday -/
def wednesday_apples : ℕ := 9

/-- The number of apples Billy ate on Friday -/
def friday_apples : ℕ := monday_apples / 2

/-- The number of apples Billy ate on Thursday -/
def thursday_apples : ℕ := total_apples - (monday_apples + tuesday_apples + wednesday_apples + friday_apples)

theorem apple_ratio : thursday_apples = 4 * friday_apples := by sorry

end NUMINAMATH_CALUDE_apple_ratio_l888_88874


namespace NUMINAMATH_CALUDE_whitney_bought_two_posters_l888_88831

/-- Represents the purchase at the school book fair -/
structure BookFairPurchase where
  initialAmount : ℕ
  posterCost : ℕ
  notebookCost : ℕ
  bookmarkCost : ℕ
  numNotebooks : ℕ
  numBookmarks : ℕ
  amountLeft : ℕ

/-- Theorem stating that Whitney bought 2 posters -/
theorem whitney_bought_two_posters (purchase : BookFairPurchase)
  (h1 : purchase.initialAmount = 40)
  (h2 : purchase.posterCost = 5)
  (h3 : purchase.notebookCost = 4)
  (h4 : purchase.bookmarkCost = 2)
  (h5 : purchase.numNotebooks = 3)
  (h6 : purchase.numBookmarks = 2)
  (h7 : purchase.amountLeft = 14) :
  ∃ (numPosters : ℕ), numPosters = 2 ∧
    purchase.initialAmount = 
      numPosters * purchase.posterCost +
      purchase.numNotebooks * purchase.notebookCost +
      purchase.numBookmarks * purchase.bookmarkCost +
      purchase.amountLeft :=
by sorry

end NUMINAMATH_CALUDE_whitney_bought_two_posters_l888_88831


namespace NUMINAMATH_CALUDE_product_remainder_zero_l888_88865

theorem product_remainder_zero : (4251 * 7396 * 4625) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_zero_l888_88865


namespace NUMINAMATH_CALUDE_right_triangles_AF_length_l888_88849

theorem right_triangles_AF_length 
  (AB DE CD EF BC : ℝ)
  (h1 : AB = 12)
  (h2 : DE = 12)
  (h3 : CD = 8)
  (h4 : EF = 8)
  (h5 : BC = 5)
  (h6 : AB^2 + BC^2 = AC^2)  -- ABC is a right triangle
  (h7 : AC^2 + CD^2 = AD^2)  -- ACD is a right triangle
  (h8 : AD^2 + DE^2 = AE^2)  -- ADE is a right triangle
  (h9 : AE^2 + EF^2 = AF^2)  -- AEF is a right triangle
  : AF = 21 := by
    sorry

end NUMINAMATH_CALUDE_right_triangles_AF_length_l888_88849


namespace NUMINAMATH_CALUDE_purchase_price_satisfies_conditions_l888_88888

/-- The purchase price of a pen that satisfies the given conditions -/
def purchase_price : ℝ := 5

/-- The selling price of one pen -/
def selling_price_one : ℝ := 10

/-- The selling price of three pens -/
def selling_price_three : ℝ := 20

/-- Theorem stating that the purchase price satisfies the given conditions -/
theorem purchase_price_satisfies_conditions :
  (selling_price_one - purchase_price = selling_price_three - 3 * purchase_price) ∧
  (purchase_price > 0) ∧
  (purchase_price < selling_price_one) ∧
  (3 * purchase_price < selling_price_three) := by
  sorry

end NUMINAMATH_CALUDE_purchase_price_satisfies_conditions_l888_88888


namespace NUMINAMATH_CALUDE_power_three_mod_eleven_l888_88820

theorem power_three_mod_eleven : 3^2040 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_eleven_l888_88820


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l888_88813

theorem sum_of_roots_cubic (x₁ x₂ x₃ k m : ℝ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)
  (h_root₁ : 2 * x₁^3 - k * x₁ = m)
  (h_root₂ : 2 * x₂^3 - k * x₂ = m)
  (h_root₃ : 2 * x₃^3 - k * x₃ = m) :
  x₁ + x₂ + x₃ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l888_88813


namespace NUMINAMATH_CALUDE_sum_of_factors_l888_88866

theorem sum_of_factors (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 0 →
  a + b + c + d + e = 35 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l888_88866


namespace NUMINAMATH_CALUDE_xray_cost_correct_l888_88823

/-- The cost of an x-ray, given the conditions of the problem -/
def xray_cost : ℝ := 250

/-- The cost of an MRI, given that it's triple the x-ray cost -/
def mri_cost : ℝ := 3 * xray_cost

/-- The total cost of both procedures -/
def total_cost : ℝ := xray_cost + mri_cost

/-- The insurance coverage percentage -/
def insurance_coverage : ℝ := 0.8

/-- The amount paid by the patient -/
def patient_payment : ℝ := 200

/-- Theorem stating that the x-ray cost is correct given the problem conditions -/
theorem xray_cost_correct : 
  mri_cost = 3 * xray_cost ∧ 
  (1 - insurance_coverage) * total_cost = patient_payment ∧
  xray_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_xray_cost_correct_l888_88823


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l888_88829

/-- Represents the dimensions of a cistern with an elevated platform --/
structure CisternDimensions where
  length : Real
  width : Real
  waterDepth : Real
  platformLength : Real
  platformWidth : Real
  platformHeight : Real

/-- Calculates the total wet surface area of the cistern --/
def totalWetSurfaceArea (d : CisternDimensions) : Real :=
  let wallArea := 2 * (d.length * d.waterDepth) + 2 * (d.width * d.waterDepth)
  let bottomArea := d.length * d.width
  let submergedHeight := d.waterDepth - d.platformHeight
  let platformSideArea := 2 * (d.platformLength * submergedHeight) + 2 * (d.platformWidth * submergedHeight)
  wallArea + bottomArea + platformSideArea

/-- Theorem stating that the total wet surface area of the given cistern is 63.5 square meters --/
theorem cistern_wet_surface_area :
  let d : CisternDimensions := {
    length := 8,
    width := 4,
    waterDepth := 1.25,
    platformLength := 1,
    platformWidth := 0.5,
    platformHeight := 0.75
  }
  totalWetSurfaceArea d = 63.5 := by
  sorry


end NUMINAMATH_CALUDE_cistern_wet_surface_area_l888_88829


namespace NUMINAMATH_CALUDE_total_peaches_l888_88832

theorem total_peaches (red : ℕ) (yellow : ℕ) (green : ℕ) 
  (h_red : red = 7) (h_yellow : yellow = 15) (h_green : green = 8) :
  red + yellow + green = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l888_88832


namespace NUMINAMATH_CALUDE_max_x_on_3x3_grid_l888_88873

/-- Represents a 3x3 grid where X's can be placed. -/
def Grid := Fin 3 → Fin 3 → Bool

/-- Checks if three X's are aligned in any direction on the grid. -/
def hasThreeAligned (g : Grid) : Bool :=
  sorry

/-- Counts the number of X's placed on the grid. -/
def countX (g : Grid) : Nat :=
  sorry

/-- Theorem stating the maximum number of X's that can be placed on a 3x3 grid
    without three X's aligning vertically, horizontally, or diagonally is 4. -/
theorem max_x_on_3x3_grid :
  (∃ g : Grid, ¬hasThreeAligned g ∧ countX g = 4) ∧
  (∀ g : Grid, ¬hasThreeAligned g → countX g ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_x_on_3x3_grid_l888_88873


namespace NUMINAMATH_CALUDE_excess_students_equals_35_l888_88859

/-- Represents a kindergarten classroom at Maple Ridge School -/
structure Classroom where
  students : Nat
  rabbits : Nat
  guinea_pigs : Nat

/-- The number of classrooms at Maple Ridge School -/
def num_classrooms : Nat := 5

/-- A standard classroom at Maple Ridge School -/
def standard_classroom : Classroom := {
  students := 15,
  rabbits := 3,
  guinea_pigs := 5
}

/-- The total number of students in all classrooms -/
def total_students : Nat := num_classrooms * standard_classroom.students

/-- The total number of rabbits in all classrooms -/
def total_rabbits : Nat := num_classrooms * standard_classroom.rabbits

/-- The total number of guinea pigs in all classrooms -/
def total_guinea_pigs : Nat := num_classrooms * standard_classroom.guinea_pigs

/-- 
Theorem: The sum of the number of students in excess of the number of pet rabbits 
and the number of guinea pigs in all 5 classrooms is equal to 35.
-/
theorem excess_students_equals_35 : 
  total_students - (total_rabbits + total_guinea_pigs) = 35 := by
  sorry

end NUMINAMATH_CALUDE_excess_students_equals_35_l888_88859


namespace NUMINAMATH_CALUDE_class_size_proof_l888_88843

def class_composition (total : ℕ) (girls : ℕ) (boys : ℕ) : Prop :=
  girls + boys = total ∧ girls = (60 * total) / 100

def absent_composition (total : ℕ) (girls : ℕ) (boys : ℕ) : Prop :=
  (girls - 1) = (625 * (total - 3)) / 1000

theorem class_size_proof (total : ℕ) (girls : ℕ) (boys : ℕ) :
  class_composition total girls boys ∧ 
  absent_composition total girls boys →
  girls = 21 ∧ boys = 14 :=
by sorry

end NUMINAMATH_CALUDE_class_size_proof_l888_88843


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_zero_l888_88821

theorem negation_of_forall_geq_zero :
  (¬ ∀ x : ℝ, x^2 - x ≥ 0) ↔ (∃ x : ℝ, x^2 - x < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_zero_l888_88821


namespace NUMINAMATH_CALUDE_inequality_implies_bounds_l888_88816

/-- Custom operation ⊗ defined on ℝ -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the relationship between the inequality and the bounds on a -/
theorem inequality_implies_bounds (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_bounds_l888_88816


namespace NUMINAMATH_CALUDE_triangle_side_length_l888_88884

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C,
    if a = √5, c = 2, and cos(A) = 2/3, then b = 3 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2/3 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  b = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l888_88884


namespace NUMINAMATH_CALUDE_clive_olive_money_l888_88885

/-- Proves that Clive has $10.00 to spend on olives given the problem conditions -/
theorem clive_olive_money : 
  -- Define the given conditions
  let olives_needed : ℕ := 80
  let olives_per_jar : ℕ := 20
  let cost_per_jar : ℚ := 3/2  -- $1.50 represented as a rational number
  let change : ℚ := 4  -- $4.00 change

  -- Calculate the number of jars needed
  let jars_needed : ℕ := olives_needed / olives_per_jar

  -- Calculate the total cost of olives
  let total_cost : ℚ := jars_needed * cost_per_jar

  -- Define Clive's total money as the sum of total cost and change
  let clive_money : ℚ := total_cost + change

  -- Prove that Clive's total money is $10.00
  clive_money = 10 := by sorry

end NUMINAMATH_CALUDE_clive_olive_money_l888_88885


namespace NUMINAMATH_CALUDE_app_cost_calculation_l888_88830

/-- Calculates the total cost of an app with online access -/
def total_cost (app_price : ℕ) (monthly_fee : ℕ) (months : ℕ) : ℕ :=
  app_price + monthly_fee * months

/-- Theorem: The total cost for an app with an initial price of $5 and 
    a monthly online access fee of $8, used for 2 months, is $21 -/
theorem app_cost_calculation : total_cost 5 8 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_app_cost_calculation_l888_88830


namespace NUMINAMATH_CALUDE_small_boxes_count_l888_88875

theorem small_boxes_count (total_chocolates : ℕ) (chocolates_per_box : ℕ) 
  (h1 : total_chocolates = 525) 
  (h2 : chocolates_per_box = 25) : 
  total_chocolates / chocolates_per_box = 21 := by
  sorry

#check small_boxes_count

end NUMINAMATH_CALUDE_small_boxes_count_l888_88875


namespace NUMINAMATH_CALUDE_women_average_age_l888_88872

theorem women_average_age (n : ℕ) (A : ℝ) :
  n = 12 ∧
  (n * (A + 3) = n * A + 3 * 42 - (25 + 30 + 35)) →
  42 = (3 * 42) / 3 :=
by sorry

end NUMINAMATH_CALUDE_women_average_age_l888_88872


namespace NUMINAMATH_CALUDE_donovans_test_score_l888_88854

theorem donovans_test_score (incorrect_answers : ℕ) (correct_percentage : ℚ) 
  (h1 : incorrect_answers = 13)
  (h2 : correct_percentage = 7292 / 10000) : 
  ∃ (correct_answers : ℕ), 
    (correct_answers : ℚ) / ((correct_answers : ℚ) + (incorrect_answers : ℚ)) = correct_percentage ∧ 
    correct_answers = 35 := by
  sorry

end NUMINAMATH_CALUDE_donovans_test_score_l888_88854


namespace NUMINAMATH_CALUDE_system_implies_quadratic_l888_88867

theorem system_implies_quadratic (x y : ℝ) :
  (3 * x^2 + 9 * x + 4 * y - 2 = 0) ∧ (3 * x + 2 * y - 6 = 0) →
  y^2 - 13 * y + 26 = 0 := by
  sorry

end NUMINAMATH_CALUDE_system_implies_quadratic_l888_88867


namespace NUMINAMATH_CALUDE_jerry_birthday_games_l888_88889

/-- The number of games Jerry received for his birthday -/
def games_received (initial_games total_games : ℕ) : ℕ :=
  total_games - initial_games

/-- Theorem stating that Jerry received 2 games for his birthday -/
theorem jerry_birthday_games :
  games_received 7 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_birthday_games_l888_88889


namespace NUMINAMATH_CALUDE_real_roots_condition_l888_88833

theorem real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, m * x^2 - 4 * x + 3 = 0) ↔ m ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_real_roots_condition_l888_88833


namespace NUMINAMATH_CALUDE_simplify_expressions_l888_88895

theorem simplify_expressions :
  (2 * Real.sqrt 12 - 6 * Real.sqrt (1/3) + Real.sqrt 48 = -2 * Real.sqrt 3) ∧
  ((Real.sqrt 3 - Real.pi)^0 - (Real.sqrt 20 - Real.sqrt 15) / Real.sqrt 5 + (-1)^2017 = Real.sqrt 3 - 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l888_88895


namespace NUMINAMATH_CALUDE_interesting_numbers_l888_88899

def is_interesting (n : ℕ) : Prop :=
  20 ≤ n ∧ n ≤ 90 ∧
  ∃ (p : ℕ) (k : ℕ), 
    Nat.Prime p ∧ 
    k ≥ 2 ∧ 
    n = p^k

theorem interesting_numbers : 
  {n : ℕ | is_interesting n} = {25, 27, 32, 49, 64, 81} :=
by sorry

end NUMINAMATH_CALUDE_interesting_numbers_l888_88899


namespace NUMINAMATH_CALUDE_vegan_soy_free_fraction_l888_88858

theorem vegan_soy_free_fraction (total_dishes : ℕ) (vegan_dishes : ℕ) (soy_vegan_dishes : ℕ) :
  vegan_dishes = total_dishes / 4 →
  vegan_dishes = 6 →
  soy_vegan_dishes = 5 →
  (vegan_dishes - soy_vegan_dishes : ℚ) / total_dishes = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_vegan_soy_free_fraction_l888_88858


namespace NUMINAMATH_CALUDE_train_speed_l888_88890

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 240)
  (h2 : bridge_length = 750)
  (h3 : crossing_time = 80) :
  (train_length + bridge_length) / crossing_time = 12.375 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l888_88890


namespace NUMINAMATH_CALUDE_triangle_properties_l888_88864

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Theorem statement
theorem triangle_properties (t : AcuteTriangle) 
  (h1 : t.c = 2) 
  (h2 : t.A = π/3) : 
  t.a * Real.sin t.C = Real.sqrt 3 ∧ 
  1 + Real.sqrt 3 < t.a + t.b ∧ 
  t.a + t.b < 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l888_88864


namespace NUMINAMATH_CALUDE_distance_inequality_l888_88824

-- Define the types for planes, lines, and points
variable (Plane Line Point : Type)

-- Define the distance function
variable (distance : Point → Point → ℝ)
variable (distance_point_line : Point → Line → ℝ)
variable (distance_line_line : Line → Line → ℝ)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the containment relations
variable (line_in_plane : Line → Plane → Prop)
variable (point_on_line : Point → Line → Prop)

-- Define the specific objects in the problem
variable (α β : Plane) (m n : Line) (A B : Point)

-- Define the distances
variable (a b c : ℝ)

theorem distance_inequality 
  (h_parallel : parallel α β)
  (h_m_in_α : line_in_plane m α)
  (h_n_in_β : line_in_plane n β)
  (h_A_on_m : point_on_line A m)
  (h_B_on_n : point_on_line B n)
  (h_a_def : a = distance A B)
  (h_b_def : b = distance_point_line A n)
  (h_c_def : c = distance_line_line m n) :
  c ≤ b ∧ b ≤ a :=
by sorry

end NUMINAMATH_CALUDE_distance_inequality_l888_88824


namespace NUMINAMATH_CALUDE_henrys_initial_games_l888_88828

/-- The number of games Henry had initially -/
def initial_games_henry : ℕ := 33

/-- The number of games Neil had initially -/
def initial_games_neil : ℕ := 2

/-- The number of games Henry gave to Neil -/
def games_given : ℕ := 5

theorem henrys_initial_games :
  initial_games_henry = 33 ∧
  initial_games_neil = 2 ∧
  games_given = 5 ∧
  initial_games_henry - games_given = 4 * (initial_games_neil + games_given) :=
by sorry

end NUMINAMATH_CALUDE_henrys_initial_games_l888_88828


namespace NUMINAMATH_CALUDE_toothpicks_stage_15_l888_88897

/-- Calculates the number of toothpicks at a given stage -/
def toothpicks (stage : ℕ) : ℕ :=
  let initial := 3
  let baseIncrease := 2
  let extraIncreaseInterval := 3
  let extraIncrease := (stage - 1) / extraIncreaseInterval

  initial + (stage - 1) * baseIncrease + 
    ((stage - 1) / extraIncreaseInterval) * (stage - 1) * (stage - 2) / 2

theorem toothpicks_stage_15 : toothpicks 15 = 61 := by
  sorry

#eval toothpicks 15

end NUMINAMATH_CALUDE_toothpicks_stage_15_l888_88897


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l888_88842

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x, f x = a * x^2 + b * x + c) ∧
    (∀ x, f x ≥ f (-1)) ∧
    f (-1) = -4 ∧
    f (-2) = 5

theorem quadratic_function_properties (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  (∀ x, f x = 9 * (x + 1)^2 - 4) ∧
  f 0 = 5 ∧
  f (-5/3) = 0 ∧
  f (-1/3) = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l888_88842


namespace NUMINAMATH_CALUDE_ellipse_focus_l888_88860

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  majorAxisEnd1 : Point
  majorAxisEnd2 : Point
  minorAxisEnd1 : Point
  minorAxisEnd2 : Point

/-- Theorem: The focus with greater x-coordinate of the given ellipse -/
theorem ellipse_focus (e : Ellipse) 
  (h1 : e.center = ⟨4, -1⟩)
  (h2 : e.majorAxisEnd1 = ⟨0, -1⟩)
  (h3 : e.majorAxisEnd2 = ⟨8, -1⟩)
  (h4 : e.minorAxisEnd1 = ⟨4, 2⟩)
  (h5 : e.minorAxisEnd2 = ⟨4, -4⟩) :
  ∃ (focus : Point), focus.x = 4 + Real.sqrt 7 ∧ focus.y = -1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_l888_88860


namespace NUMINAMATH_CALUDE_odd_function_property_l888_88802

-- Define an odd function on an interval
def odd_function_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ Set.Icc a b, f (-x) = -f x) ∧ (a ≤ b)

-- Theorem statement
theorem odd_function_property (f : ℝ → ℝ) (a b : ℝ) 
  (h : odd_function_on_interval f a b) : f (2 * (a + b)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l888_88802


namespace NUMINAMATH_CALUDE_max_a_value_l888_88870

theorem max_a_value (a b : ℕ) (ha : 1 < a) (hb : a < b) :
  (∃! x : ℝ, -2 * x + 4033 = |x - 1| + |x + a| + |x - b|) →
  (∀ a' : ℕ, 1 < a' ∧ a' < b ∧ (∃! x : ℝ, -2 * x + 4033 = |x - 1| + |x + a'| + |x - b|) → a' ≤ a) ∧
  a = 4031 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l888_88870


namespace NUMINAMATH_CALUDE_division_remainder_l888_88827

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = divisor * quotient + remainder →
  dividend = 199 →
  divisor = 18 →
  quotient = 11 →
  remainder = 1 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l888_88827


namespace NUMINAMATH_CALUDE_circle_to_octagon_area_ratio_l888_88808

/-- The ratio of the area of a circle inscribed in a regular octagon to the area of the octagon,
    where the circle's radius reaches the midpoint of each octagon side. -/
theorem circle_to_octagon_area_ratio : ∃ (a b : ℕ), 
  (a : ℝ).sqrt / b * π = π / (4 * (1 + Real.sqrt 2)) ∧ a * b = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_to_octagon_area_ratio_l888_88808


namespace NUMINAMATH_CALUDE_inequality_solution_set_l888_88857

theorem inequality_solution_set (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  let S := {x : ℝ | (x - a) * (x + a - 1) < 0}
  (0 ≤ a ∧ a < 1/2 → S = Set.Ioo a (1 - a)) ∧
  (a = 1/2 → S = ∅) ∧
  (1/2 < a ∧ a ≤ 1 → S = Set.Ioo (1 - a) a) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l888_88857


namespace NUMINAMATH_CALUDE_negative_abs_not_equal_five_l888_88894

theorem negative_abs_not_equal_five : -|(-5 : ℤ)| ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_not_equal_five_l888_88894


namespace NUMINAMATH_CALUDE_participation_and_optimality_l888_88887

/-- Represents a company in country A --/
structure Company where
  investmentCost : ℝ
  successProbability : ℝ
  potentialRevenue : ℝ

/-- Conditions for the problem --/
axiom probability_bounds {α : ℝ} : 0 < α ∧ α < 1

/-- Expected income when both companies participate --/
def expectedIncomeBoth (c : Company) : ℝ :=
  c.successProbability * (1 - c.successProbability) * c.potentialRevenue +
  0.5 * c.successProbability^2 * c.potentialRevenue

/-- Expected income when only one company participates --/
def expectedIncomeOne (c : Company) : ℝ :=
  c.successProbability * c.potentialRevenue

/-- Condition for a company to participate --/
def willParticipate (c : Company) : Prop :=
  expectedIncomeBoth c - c.investmentCost ≥ 0

/-- Social welfare as total profit of both companies --/
def socialWelfare (c1 c2 : Company) : ℝ :=
  2 * (expectedIncomeBoth c1 - c1.investmentCost)

/-- Theorem stating both companies will participate and it's not socially optimal --/
theorem participation_and_optimality (c1 c2 : Company)
  (h1 : c1.potentialRevenue = 24 ∧ c1.successProbability = 0.5 ∧ c1.investmentCost = 7)
  (h2 : c2.potentialRevenue = 24 ∧ c2.successProbability = 0.5 ∧ c2.investmentCost = 7) :
  willParticipate c1 ∧ willParticipate c2 ∧
  socialWelfare c1 c2 < expectedIncomeOne c1 - c1.investmentCost := by
  sorry

end NUMINAMATH_CALUDE_participation_and_optimality_l888_88887


namespace NUMINAMATH_CALUDE_cosine_equality_l888_88877

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (942 * π / 180) → n = 138 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l888_88877


namespace NUMINAMATH_CALUDE_regression_lines_intersection_l888_88882

/-- A regression line in 2D space -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point where a regression line passes through -/
def RegressionLine.point_on_line (l : RegressionLine) (x : ℝ) : ℝ × ℝ :=
  (x, l.slope * x + l.intercept)

theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : (s, t) = l₁.point_on_line s)
  (h₂ : (s, t) = l₂.point_on_line s) :
  ∃ (x y : ℝ), l₁.point_on_line x = (x, y) ∧ l₂.point_on_line x = (x, y) ∧ x = s ∧ y = t :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersection_l888_88882


namespace NUMINAMATH_CALUDE_sum_difference_equals_three_l888_88836

theorem sum_difference_equals_three : (2 + 4 + 6) - (1 + 3 + 5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equals_three_l888_88836


namespace NUMINAMATH_CALUDE_parabola_properties_l888_88846

/-- Represents a quadratic function of the form f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Properties of a parabola -/
structure ParabolaProperties where
  opensDownward : Bool
  axisOfSymmetry : ℝ
  vertexX : ℝ
  vertexY : ℝ

/-- Compute the properties of a parabola given its quadratic function -/
def computeParabolaProperties (f : QuadraticFunction) : ParabolaProperties :=
  sorry

theorem parabola_properties (f : QuadraticFunction) 
  (h : f = QuadraticFunction.mk (-2) 4 8) :
  computeParabolaProperties f = 
    ParabolaProperties.mk true 1 1 10 := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l888_88846


namespace NUMINAMATH_CALUDE_debose_family_mean_age_l888_88834

theorem debose_family_mean_age : 
  let ages : List ℕ := [8, 8, 16, 18]
  let num_children := ages.length
  let sum_ages := ages.sum
  (sum_ages : ℚ) / num_children = 25/2 := by sorry

end NUMINAMATH_CALUDE_debose_family_mean_age_l888_88834


namespace NUMINAMATH_CALUDE_odd_function_has_zero_point_l888_88898

theorem odd_function_has_zero_point (f : ℝ → ℝ) (h : ∀ x, f (-x) = -f x) :
  ∃ x, f x = 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_has_zero_point_l888_88898


namespace NUMINAMATH_CALUDE_f_has_three_zeros_l888_88815

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2017^x + Real.log x / Real.log 2017
  else if x < 0 then -(2017^(-x) + Real.log (-x) / Real.log 2017)
  else 0

theorem f_has_three_zeros :
  (∃! a b c : ℝ, a < 0 ∧ b = 0 ∧ c > 0 ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) ∧
  (∀ x : ℝ, f x = 0 → x = a ∨ x = b ∨ x = c) :=
sorry

end NUMINAMATH_CALUDE_f_has_three_zeros_l888_88815


namespace NUMINAMATH_CALUDE_total_sums_attempted_l888_88896

theorem total_sums_attempted (right_sums wrong_sums total_sums : ℕ) : 
  wrong_sums = 2 * right_sums →
  right_sums = 8 →
  total_sums = right_sums + wrong_sums →
  total_sums = 24 := by
sorry

end NUMINAMATH_CALUDE_total_sums_attempted_l888_88896


namespace NUMINAMATH_CALUDE_section_b_students_l888_88893

/-- Proves that given the conditions of the class sections, the number of students in section B is 20 -/
theorem section_b_students (students_a : ℕ) (avg_weight_a : ℚ) (avg_weight_b : ℚ) (avg_weight_total : ℚ) :
  students_a = 40 →
  avg_weight_a = 50 →
  avg_weight_b = 40 →
  avg_weight_total = 467/10 →
  ∃ (students_b : ℕ), students_b = 20 ∧
    (students_a * avg_weight_a + students_b * avg_weight_b) / (students_a + students_b) = avg_weight_total :=
by sorry


end NUMINAMATH_CALUDE_section_b_students_l888_88893


namespace NUMINAMATH_CALUDE_lynne_book_purchase_l888_88841

/-- The number of books about the solar system Lynne bought -/
def solar_system_books : ℕ := 2

/-- The total amount Lynne spent -/
def total_spent : ℕ := 75

/-- The number of books about cats Lynne bought -/
def cat_books : ℕ := 7

/-- The number of magazines Lynne bought -/
def magazines : ℕ := 3

/-- The cost of each book -/
def book_cost : ℕ := 7

/-- The cost of each magazine -/
def magazine_cost : ℕ := 4

theorem lynne_book_purchase :
  cat_books * book_cost + solar_system_books * book_cost + magazines * magazine_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_lynne_book_purchase_l888_88841


namespace NUMINAMATH_CALUDE_larger_number_proof_l888_88863

theorem larger_number_proof (a b : ℕ) (h1 : Nat.gcd a b = 23) (h2 : Nat.lcm a b = 23 * 15 * 16) :
  max a b = 368 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l888_88863


namespace NUMINAMATH_CALUDE_coupon_a_best_discount_correct_prices_l888_88861

def coupon_a_discount (x : ℝ) : ℝ := 0.15 * x

def coupon_b_discount : ℝ := 30

def coupon_c_discount (x : ℝ) : ℝ := 0.22 * (x - 150)

theorem coupon_a_best_discount (x : ℝ) 
  (h1 : 200 < x) (h2 : x < 471.43) : 
  coupon_a_discount x > coupon_b_discount ∧ 
  coupon_a_discount x > coupon_c_discount x := by
  sorry

def price_list : List ℝ := [179.95, 199.95, 249.95, 299.95, 349.95]

theorem correct_prices (p : ℝ) (h : p ∈ price_list) :
  (200 < p ∧ p < 471.43) ↔ (p = 249.95 ∨ p = 299.95 ∨ p = 349.95) := by
  sorry

end NUMINAMATH_CALUDE_coupon_a_best_discount_correct_prices_l888_88861


namespace NUMINAMATH_CALUDE_inequality_proof_l888_88852

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) + Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2)) ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l888_88852


namespace NUMINAMATH_CALUDE_tricycle_wheels_l888_88891

theorem tricycle_wheels (num_bicycles : ℕ) (num_tricycles : ℕ) (total_wheels : ℕ) :
  num_bicycles = 24 →
  num_tricycles = 14 →
  total_wheels = 90 →
  ∃ (tricycle_wheels : ℕ),
    tricycle_wheels = 3 ∧
    total_wheels = num_bicycles * 2 + num_tricycles * tricycle_wheels :=
by
  sorry

end NUMINAMATH_CALUDE_tricycle_wheels_l888_88891


namespace NUMINAMATH_CALUDE_min_t_value_l888_88812

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-3, 2]
def interval : Set ℝ := Set.Icc (-3) 2

-- Statement: The minimum value of t that satisfies |f(x₁) - f(x₂)| ≤ t for all x₁, x₂ in the interval is 20
theorem min_t_value : 
  (∃ t : ℝ, ∀ x₁ x₂ : ℝ, x₁ ∈ interval → x₂ ∈ interval → |f x₁ - f x₂| ≤ t) ∧ 
  (∀ t : ℝ, (∀ x₁ x₂ : ℝ, x₁ ∈ interval → x₂ ∈ interval → |f x₁ - f x₂| ≤ t) → t ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_min_t_value_l888_88812


namespace NUMINAMATH_CALUDE_sin_negative_three_pi_halves_l888_88878

theorem sin_negative_three_pi_halves : Real.sin (-3 * π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_three_pi_halves_l888_88878


namespace NUMINAMATH_CALUDE_sofia_survey_l888_88880

theorem sofia_survey (mashed_potatoes bacon : ℕ) 
  (h1 : mashed_potatoes = 185) 
  (h2 : bacon = 125) : 
  mashed_potatoes + bacon = 310 := by
sorry

end NUMINAMATH_CALUDE_sofia_survey_l888_88880


namespace NUMINAMATH_CALUDE_salary_change_l888_88839

theorem salary_change (initial_salary : ℝ) (h : initial_salary > 0) :
  let decreased_salary := initial_salary * (1 - 0.4)
  let final_salary := decreased_salary * (1 + 0.4)
  (initial_salary - final_salary) / initial_salary = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_l888_88839


namespace NUMINAMATH_CALUDE_original_network_engineers_l888_88856

/-- The number of new network engineers hired from University A -/
def new_hires : ℕ := 8

/-- The fraction of network engineers from University A after hiring -/
def fraction_after : ℚ := 3/4

/-- The fraction of original network engineers from University A -/
def fraction_original : ℚ := 13/20

/-- The original number of network engineers -/
def original_count : ℕ := 20

theorem original_network_engineers :
  ∃ (o : ℕ), 
    (o : ℚ) * fraction_original + new_hires = 
    ((o : ℚ) + new_hires) * fraction_after ∧
    o = original_count :=
by sorry

end NUMINAMATH_CALUDE_original_network_engineers_l888_88856


namespace NUMINAMATH_CALUDE_prob_truth_or_lie_classroom_l888_88850

/-- Represents the characteristics of a student population -/
structure StudentPopulation where
  total : ℕ
  truth_tellers : ℕ
  liars : ℕ
  both : ℕ
  avoiders : ℕ
  serious_liars_ratio : ℚ

/-- Calculates the probability of a student speaking truth or lying in a serious situation -/
def prob_truth_or_lie (pop : StudentPopulation) : ℚ :=
  let serious_liars := (pop.both : ℚ) * pop.serious_liars_ratio
  (pop.truth_tellers + pop.liars + serious_liars) / pop.total

/-- Theorem stating the probability of a student speaking truth or lying in a serious situation -/
theorem prob_truth_or_lie_classroom (pop : StudentPopulation) 
  (h1 : pop.total = 100)
  (h2 : pop.truth_tellers = 40)
  (h3 : pop.liars = 25)
  (h4 : pop.both = 15)
  (h5 : pop.avoiders = 20)
  (h6 : pop.serious_liars_ratio = 70 / 100)
  (h7 : pop.truth_tellers + pop.liars + pop.both + pop.avoiders = pop.total) :
  prob_truth_or_lie pop = 76 / 100 := by
  sorry

end NUMINAMATH_CALUDE_prob_truth_or_lie_classroom_l888_88850


namespace NUMINAMATH_CALUDE_absolute_value_equation_simplification_l888_88817

theorem absolute_value_equation_simplification (a b c : ℝ) :
  (∀ x : ℝ, |5 * x - 4| + a ≠ 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |4 * x₁ - 3| + b = 0 ∧ |4 * x₂ - 3| + b = 0) →
  (∃! x : ℝ, |3 * x - 2| + c = 0) →
  |a - c| + |c - b| - |a - b| = 0 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_simplification_l888_88817


namespace NUMINAMATH_CALUDE_special_cone_volume_l888_88837

/-- A cone with base area π and lateral surface in the shape of a semicircle -/
structure SpecialCone where
  /-- The radius of the base of the cone -/
  r : ℝ
  /-- The height of the cone -/
  h : ℝ
  /-- The slant height of the cone -/
  l : ℝ
  /-- The base area is π -/
  base_area : π * r^2 = π
  /-- The lateral surface is a semicircle -/
  lateral_surface : π * l = 2 * π * r

/-- The volume of the special cone is (√3/3)π -/
theorem special_cone_volume (c : SpecialCone) : 
  (1/3) * π * c.r^2 * c.h = (Real.sqrt 3 / 3) * π := by
  sorry


end NUMINAMATH_CALUDE_special_cone_volume_l888_88837


namespace NUMINAMATH_CALUDE_blue_lipstick_count_l888_88822

theorem blue_lipstick_count (total_students : ℕ) 
  (h_total : total_students = 360)
  (h_half_lipstick : ∃ lipstick_wearers : ℕ, 2 * lipstick_wearers = total_students)
  (h_red : ∃ red_wearers : ℕ, 4 * red_wearers = lipstick_wearers)
  (h_pink : ∃ pink_wearers : ℕ, 3 * pink_wearers = lipstick_wearers)
  (h_purple : ∃ purple_wearers : ℕ, 6 * purple_wearers = lipstick_wearers)
  (h_green : ∃ green_wearers : ℕ, 12 * green_wearers = lipstick_wearers)
  (h_blue : ∃ blue_wearers : ℕ, blue_wearers = lipstick_wearers - (red_wearers + pink_wearers + purple_wearers + green_wearers)) :
  blue_wearers = 30 := by
  sorry


end NUMINAMATH_CALUDE_blue_lipstick_count_l888_88822


namespace NUMINAMATH_CALUDE_exponent_subtraction_l888_88853

theorem exponent_subtraction : (-2)^3 - (-3)^2 = -17 := by
  sorry

end NUMINAMATH_CALUDE_exponent_subtraction_l888_88853


namespace NUMINAMATH_CALUDE_matrix_determinant_l888_88855

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; -3, 6]

theorem matrix_determinant :
  Matrix.det matrix = 36 := by sorry

end NUMINAMATH_CALUDE_matrix_determinant_l888_88855


namespace NUMINAMATH_CALUDE_apple_sales_theorem_l888_88844

/-- Calculate the total money earned from selling apples from a rectangular plot of trees -/
def apple_sales_revenue (rows : ℕ) (cols : ℕ) (apples_per_tree : ℕ) (price_per_apple : ℚ) : ℚ :=
  (rows * cols * apples_per_tree : ℕ) * price_per_apple

/-- Theorem: The total money earned from selling apples from a 3x4 plot of trees,
    where each tree produces 5 apples and each apple is sold for $0.5, is equal to $30 -/
theorem apple_sales_theorem :
  apple_sales_revenue 3 4 5 (1/2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_apple_sales_theorem_l888_88844


namespace NUMINAMATH_CALUDE_unique_number_with_special_divisor_property_l888_88869

theorem unique_number_with_special_divisor_property :
  ∃! (N : ℕ), 
    N > 0 ∧
    (∃ (m : ℕ), 
      m > 0 ∧ 
      m < N ∧
      N % m = 0 ∧
      (∀ (d : ℕ), d > 0 → d < N → N % d = 0 → d ≤ m) ∧
      (∃ (k : ℕ), N + m = 10^k)) ∧
    N = 75 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_special_divisor_property_l888_88869


namespace NUMINAMATH_CALUDE_partnership_profit_share_l888_88810

/-- Partnership profit sharing problem -/
theorem partnership_profit_share
  (x : ℝ)  -- A's investment amount
  (annual_gain : ℝ)  -- Total annual gain
  (h1 : annual_gain = 18900)  -- Given annual gain
  (h2 : x > 0)  -- Assumption that A's investment is positive
  : x * 12 / (x * 12 + 2 * x * 6 + 3 * x * 4) * annual_gain = 6300 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_share_l888_88810


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l888_88871

theorem quadratic_roots_properties (p : ℝ) (x₁ x₂ : ℂ) :
  x₁^2 + p * x₁ + 2 = 0 →
  x₂^2 + p * x₂ + 2 = 0 →
  x₁ = 1 + I →
  (x₂ = 1 - I ∧ x₁ / x₂ = I) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l888_88871


namespace NUMINAMATH_CALUDE_larger_number_problem_l888_88826

theorem larger_number_problem (x y : ℝ) : 
  5 * y = 6 * x → y - x = 10 → y = 60 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l888_88826


namespace NUMINAMATH_CALUDE_hash_problem_l888_88800

-- Define the # operation
def hash (a b : ℚ) : ℚ := a + (a^2 / b)

-- Theorem statement
theorem hash_problem : (hash 4 3) - 10 = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_hash_problem_l888_88800


namespace NUMINAMATH_CALUDE_parabola_c_value_l888_88814

/-- A parabola passing through two points with equal y-coordinates -/
structure Parabola where
  b : ℝ
  c : ℝ
  pass_through_minus_one : 2 = 1 + (-b) + c
  pass_through_three : 2 = 9 + 3*b + c

/-- The value of c for a parabola passing through (-1, 2) and (3, 2) is -1 -/
theorem parabola_c_value (p : Parabola) : p.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l888_88814


namespace NUMINAMATH_CALUDE_third_term_is_five_l888_88835

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℝ  -- second term
  d : ℝ  -- common difference

/-- The sum of the second and fourth terms is 10 -/
def sum_second_fourth (seq : ArithmeticSequence) : Prop :=
  seq.a + (seq.a + 2 * seq.d) = 10

/-- The third term of the sequence -/
def third_term (seq : ArithmeticSequence) : ℝ :=
  seq.a + seq.d

/-- Theorem: If the sum of the second and fourth terms of an arithmetic sequence is 10,
    then the third term is 5 -/
theorem third_term_is_five (seq : ArithmeticSequence) 
    (h : sum_second_fourth seq) : third_term seq = 5 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_five_l888_88835


namespace NUMINAMATH_CALUDE_sqrt_15_has_two_roots_l888_88811

-- Define √15 as a real number
noncomputable def sqrt15 : ℝ := Real.sqrt 15

-- State the theorem
theorem sqrt_15_has_two_roots :
  ∃ (x : ℝ), x ≠ sqrt15 ∧ x * x = 15 :=
by
  -- The proof would go here, but we're using sorry as instructed
  sorry

end NUMINAMATH_CALUDE_sqrt_15_has_two_roots_l888_88811


namespace NUMINAMATH_CALUDE_complex_number_range_l888_88892

theorem complex_number_range (z : ℂ) (h : Complex.abs (z - (3 - 4*I)) = 1) :
  4 ≤ Complex.abs z ∧ Complex.abs z ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_range_l888_88892


namespace NUMINAMATH_CALUDE_bucket_capacity_first_case_l888_88876

/-- The capacity of a bucket in the first case, given the following conditions:
  - 22 buckets of water fill a tank in the first case
  - 33 buckets of water fill the same tank in the second case
  - In the second case, each bucket has a capacity of 9 litres
-/
theorem bucket_capacity_first_case : 
  ∀ (capacity_first : ℝ) (tank_volume : ℝ),
  22 * capacity_first = tank_volume →
  33 * 9 = tank_volume →
  capacity_first = 13.5 := by
sorry

end NUMINAMATH_CALUDE_bucket_capacity_first_case_l888_88876


namespace NUMINAMATH_CALUDE_quadratic_complex_roots_l888_88840

theorem quadratic_complex_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = -1 + 2*I ∧ 
  z₂ = -3 - 2*I ∧ 
  (z₁^2 + 2*z₁ = -3 + 4*I) ∧ 
  (z₂^2 + 2*z₂ = -3 + 4*I) := by
sorry

end NUMINAMATH_CALUDE_quadratic_complex_roots_l888_88840


namespace NUMINAMATH_CALUDE_no_square_divisible_by_six_between_55_and_120_l888_88801

theorem no_square_divisible_by_six_between_55_and_120 : ¬ ∃ x : ℕ, 
  (∃ n : ℕ, x = n ^ 2) ∧ 
  (x % 6 = 0) ∧ 
  (55 < x) ∧ 
  (x < 120) := by
sorry

end NUMINAMATH_CALUDE_no_square_divisible_by_six_between_55_and_120_l888_88801


namespace NUMINAMATH_CALUDE_queenie_earnings_l888_88851

/-- Calculates the total earnings for a worker given their daily rate, overtime rate, 
    number of days worked, and number of overtime hours. -/
def total_earnings (daily_rate : ℕ) (overtime_rate : ℕ) (days_worked : ℕ) (overtime_hours : ℕ) : ℕ :=
  daily_rate * days_worked + overtime_rate * overtime_hours

/-- Proves that Queenie's total earnings for 5 days of work with 4 hours overtime
    are equal to $770, given her daily rate of $150 and overtime rate of $5 per hour. -/
theorem queenie_earnings : total_earnings 150 5 5 4 = 770 := by
  sorry

end NUMINAMATH_CALUDE_queenie_earnings_l888_88851


namespace NUMINAMATH_CALUDE_martha_points_l888_88804

/-- Represents the point system and shopping details for Martha --/
structure PointSystem where
  points_per_ten_dollars : ℕ
  bonus_points : ℕ
  bonus_threshold : ℕ
  beef_price : ℕ
  beef_quantity : ℕ
  fruit_veg_price : ℕ
  fruit_veg_quantity : ℕ
  spice_price : ℕ
  spice_quantity : ℕ
  other_groceries : ℕ

/-- Calculates the total points Martha earns based on her shopping --/
def calculate_points (ps : PointSystem) : ℕ :=
  let total_spent := ps.beef_price * ps.beef_quantity +
                     ps.fruit_veg_price * ps.fruit_veg_quantity +
                     ps.spice_price * ps.spice_quantity +
                     ps.other_groceries
  let base_points := (total_spent * ps.points_per_ten_dollars) / 10
  let bonus := if total_spent > ps.bonus_threshold then ps.bonus_points else 0
  base_points + bonus

/-- Theorem stating that Martha earns 850 points based on her shopping --/
theorem martha_points : 
  ∀ (ps : PointSystem), 
    ps.points_per_ten_dollars = 50 ∧ 
    ps.bonus_points = 250 ∧ 
    ps.bonus_threshold = 100 ∧
    ps.beef_price = 11 ∧ 
    ps.beef_quantity = 3 ∧
    ps.fruit_veg_price = 4 ∧ 
    ps.fruit_veg_quantity = 8 ∧
    ps.spice_price = 6 ∧ 
    ps.spice_quantity = 3 ∧
    ps.other_groceries = 37 →
    calculate_points ps = 850 := by
  sorry

end NUMINAMATH_CALUDE_martha_points_l888_88804


namespace NUMINAMATH_CALUDE_hoseok_friends_left_l888_88848

theorem hoseok_friends_left (total : ℕ) (right : ℕ) (h1 : total = 16) (h2 : right = 8) :
  total - (right + 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_friends_left_l888_88848


namespace NUMINAMATH_CALUDE_sum_and_product_reciprocal_sum_cube_surface_area_probability_white_ball_equilateral_triangle_area_l888_88886

-- Problem 1
theorem sum_and_product_reciprocal_sum (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 20) :
  1 / x + 1 / y = 2 := by sorry

-- Problem 2
theorem cube_surface_area (a : ℝ) :
  6 * (a + 1)^2 = 54 := by sorry

-- Problem 3
theorem probability_white_ball (b : ℝ) (c : ℝ) :
  (b - 4) / (2 * b + 42) = c / 6 := by sorry

-- Problem 4
theorem equilateral_triangle_area (c d : ℝ) :
  d * Real.sqrt 3 = (Real.sqrt 3 / 4) * c^2 := by sorry

end NUMINAMATH_CALUDE_sum_and_product_reciprocal_sum_cube_surface_area_probability_white_ball_equilateral_triangle_area_l888_88886


namespace NUMINAMATH_CALUDE_inverse_equals_one_implies_a_equals_one_l888_88879

theorem inverse_equals_one_implies_a_equals_one (a : ℝ) (h : a ≠ 0) :
  a⁻¹ = (-1)^0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_equals_one_implies_a_equals_one_l888_88879


namespace NUMINAMATH_CALUDE_function_inequality_implies_non_negative_l888_88819

theorem function_inequality_implies_non_negative 
  (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), f (x * y) + f (y - x) ≥ f (y + x)) : 
  ∀ (x : ℝ), f x ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_non_negative_l888_88819


namespace NUMINAMATH_CALUDE_inequality_solution_range_l888_88847

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) → 
  (a < 1 ∨ a > 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l888_88847


namespace NUMINAMATH_CALUDE_negation_equivalence_l888_88845

/-- A number is even if it's divisible by 2 -/
def IsEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The negation of "There is an even number that is a prime number" is equivalent to "No even number is a prime number" -/
theorem negation_equivalence : 
  (¬ ∃ n : ℕ, IsEven n ∧ IsPrime n) ↔ (∀ n : ℕ, IsEven n → ¬ IsPrime n) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l888_88845


namespace NUMINAMATH_CALUDE_calculate_expression_l888_88809

theorem calculate_expression : 
  |-Real.sqrt 3| - (4 - Real.pi)^0 - 2 * Real.sin (60 * π / 180) + (1/5)⁻¹ = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l888_88809


namespace NUMINAMATH_CALUDE_marys_tickets_l888_88807

theorem marys_tickets (total_tickets : ℕ) (probability : ℚ) (marys_tickets : ℕ) : 
  total_tickets = 120 →
  probability = 1 / 15 →
  (marys_tickets : ℚ) / total_tickets = probability →
  marys_tickets = 8 := by
  sorry

end NUMINAMATH_CALUDE_marys_tickets_l888_88807


namespace NUMINAMATH_CALUDE_conditional_probability_of_longevity_l888_88868

theorem conditional_probability_of_longevity 
  (p_20 : ℝ) 
  (p_25 : ℝ) 
  (h1 : p_20 = 0.8) 
  (h2 : p_25 = 0.4) : 
  p_25 / p_20 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_of_longevity_l888_88868


namespace NUMINAMATH_CALUDE_max_car_loan_payment_l888_88806

def total_income : ℝ := 84600
def total_expenses : ℝ := 49800
def emergency_fund_rate : ℝ := 0.1

theorem max_car_loan_payment :
  let remaining_after_expenses := total_income - total_expenses
  let emergency_fund := emergency_fund_rate * remaining_after_expenses
  let max_payment := remaining_after_expenses - emergency_fund
  max_payment = 31320 := by sorry

end NUMINAMATH_CALUDE_max_car_loan_payment_l888_88806


namespace NUMINAMATH_CALUDE_probability_non_black_ball_l888_88805

/-- Given a box with white, black, and red balls, calculate the probability of drawing a non-black ball -/
theorem probability_non_black_ball (white black red : ℕ) (h : white = 7 ∧ black = 6 ∧ red = 4) :
  (white + red) / (white + black + red : ℚ) = 11 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_non_black_ball_l888_88805


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l888_88803

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n * (a 1 + a n)) / 2

/-- Theorem: For an arithmetic sequence satisfying given conditions, a₈ = -26 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence)
    (h1 : seq.S 6 = 8 * seq.S 3)
    (h2 : seq.a 3 - seq.a 5 = 8) :
  seq.a 8 = -26 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l888_88803


namespace NUMINAMATH_CALUDE_max_ratio_abcd_l888_88825

theorem max_ratio_abcd (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d ≥ 0)
  (h5 : (a^2 + b^2 + c^2 + d^2) / (a + b + c + d)^2 = 3/8) :
  (∀ x y z w, x ≥ y ∧ y ≥ z ∧ z ≥ w ∧ w ≥ 0 ∧ 
    (x^2 + y^2 + z^2 + w^2) / (x + y + z + w)^2 = 3/8 →
    (x + z) / (y + w) ≤ (a + c) / (b + d)) ∧
  (a + c) / (b + d) ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_ratio_abcd_l888_88825


namespace NUMINAMATH_CALUDE_polynomial_roots_and_product_l888_88881

/-- Given a polynomial p(x) = x³ + (3/2)(1-a)x² - 3ax + b where a and b are real numbers,
    and |p(x)| ≤ 1 for all x in [0, √3], prove that p(x) = 0 has three real roots
    and calculate a specific product of these roots. -/
theorem polynomial_roots_and_product (a b : ℝ) 
    (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 3 → 
      |x^3 + (3/2)*(1-a)*x^2 - 3*a*x + b| ≤ 1) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ x : ℝ, x^3 + (3/2)*(1-a)*x^2 - 3*a*x + b = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    (x₁^2 - 2 - x₂) * (x₂^2 - 2 - x₃) * (x₃^2 - 2 - x₁) = -9 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_and_product_l888_88881


namespace NUMINAMATH_CALUDE_complex_equation_solution_l888_88818

theorem complex_equation_solution (x y : ℝ) :
  (x + y - 3 : ℂ) + (x - 4 : ℂ) * I = 0 → x = 4 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l888_88818
