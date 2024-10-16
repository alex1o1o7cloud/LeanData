import Mathlib

namespace NUMINAMATH_CALUDE_sine_unit_implies_on_y_axis_l2602_260271

-- Define the type for angles
def Angle : Type := ℝ

-- Define the sine function
noncomputable def sine (α : Angle) : ℝ := Real.sin α

-- Define a predicate for a directed line segment of unit length
def is_unit_directed_segment (x : ℝ) : Prop := x = 1 ∨ x = -1

-- Define a predicate for a point being on the y-axis
def on_y_axis (x y : ℝ) : Prop := x = 0

-- Theorem statement
theorem sine_unit_implies_on_y_axis (α : Angle) :
  is_unit_directed_segment (sine α) →
  ∃ (y : ℝ), on_y_axis 0 y ∧ (0, y) = (Real.cos α, Real.sin α) :=
sorry

end NUMINAMATH_CALUDE_sine_unit_implies_on_y_axis_l2602_260271


namespace NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l2602_260224

/-- Given that 5 pages cost 10 cents, prove that $15 can copy 750 pages. -/
theorem pages_copied_for_fifteen_dollars : 
  let cost_per_five_pages : ℚ := 10 / 100  -- 10 cents in dollars
  let total_amount : ℚ := 15  -- $15
  let pages_per_dollar : ℚ := 5 / cost_per_five_pages
  ⌊total_amount * pages_per_dollar⌋ = 750 :=
by sorry

end NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l2602_260224


namespace NUMINAMATH_CALUDE_sum_of_max_min_values_l2602_260269

/-- Given real numbers a and b satisfying the condition,
    prove that the sum of max and min values of a^2 + 2b^2 is 16/7 -/
theorem sum_of_max_min_values (a b : ℝ) 
  (h : (a - b/2)^2 = 1 - (7/4)*b^2) : 
  ∃ (t_max t_min : ℝ), 
    (∀ t, t = a^2 + 2*b^2 → t ≤ t_max ∧ t ≥ t_min) ∧
    t_max + t_min = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_values_l2602_260269


namespace NUMINAMATH_CALUDE_set_operation_equality_l2602_260237

def U : Finset Nat := {1,2,3,4,5}
def M : Finset Nat := {3,4,5}
def N : Finset Nat := {1,2,5}

theorem set_operation_equality : 
  (U \ M) ∩ N = {1,2} := by sorry

end NUMINAMATH_CALUDE_set_operation_equality_l2602_260237


namespace NUMINAMATH_CALUDE_ratio_comparison_is_three_l2602_260223

/-- Represents the ratio of flavoring to corn syrup to water in the standard formulation -/
def standard_ratio : Fin 3 → ℚ
| 0 => 1
| 1 => 12
| 2 => 30

/-- The ratio of flavoring to water in the sport formulation is half that of the standard formulation -/
def sport_water_ratio : ℚ := standard_ratio 2 * 2

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 5

/-- Amount of water in the sport formulation (in ounces) -/
def sport_water : ℚ := 75

/-- The ratio of flavoring to corn syrup in the sport formulation compared to the standard formulation -/
def ratio_comparison : ℚ :=
  (sport_water / sport_water_ratio) / sport_corn_syrup /
  (standard_ratio 0 / standard_ratio 1)

theorem ratio_comparison_is_three : ratio_comparison = 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_comparison_is_three_l2602_260223


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2602_260251

theorem smallest_integer_satisfying_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 13*m + 22 ≤ 0 → n ≤ m) ∧ n^2 - 13*n + 22 ≤ 0 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2602_260251


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l2602_260220

/-- Given a person's walking speeds and additional distance covered at higher speed,
    prove the actual distance traveled. -/
theorem actual_distance_traveled
  (original_speed : ℝ)
  (faster_speed : ℝ)
  (additional_distance : ℝ)
  (h1 : original_speed = 10)
  (h2 : faster_speed = 15)
  (h3 : additional_distance = 20)
  (h4 : faster_speed * (additional_distance / (faster_speed - original_speed)) =
        original_speed * (additional_distance / (faster_speed - original_speed)) + additional_distance) :
  original_speed * (additional_distance / (faster_speed - original_speed)) = 40 := by
sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l2602_260220


namespace NUMINAMATH_CALUDE_tangent_sqrt_two_implications_l2602_260239

theorem tangent_sqrt_two_implications (θ : Real) (h : Real.tan θ = Real.sqrt 2) :
  ((Real.cos θ + Real.sin θ) / (Real.cos θ - Real.sin θ) = -3 - 2 * Real.sqrt 2) ∧
  (Real.sin θ ^ 2 - Real.sin θ * Real.cos θ + 2 * Real.cos θ ^ 2 = (4 - Real.sqrt 2) / 3) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sqrt_two_implications_l2602_260239


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2602_260233

/-- The sum of two polynomials is equal to the simplified polynomial. -/
theorem polynomial_simplification (x : ℝ) :
  (12 * x^10 + 6 * x^9 + 3 * x^8) + (2 * x^11 + x^10 + 4 * x^9 + x^7 + 4 * x^4 + 7 * x + 9) =
  2 * x^11 + 13 * x^10 + 10 * x^9 + 3 * x^8 + x^7 + 4 * x^4 + 7 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2602_260233


namespace NUMINAMATH_CALUDE_chemistry_class_size_l2602_260209

theorem chemistry_class_size :
  ∀ (x y : ℕ),
  -- Total number of students
  x + y + 5 = 43 →
  -- Chemistry class is three times as large as biology class
  y + 5 = 3 * (x + 5) →
  -- Number of students in chemistry class
  y + 5 = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_chemistry_class_size_l2602_260209


namespace NUMINAMATH_CALUDE_A_intersect_B_l2602_260261

def A : Set ℝ := {-3, -2, 0, 2}
def B : Set ℝ := {x : ℝ | |x - 1| < 2}

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2602_260261


namespace NUMINAMATH_CALUDE_jessica_current_age_l2602_260284

/-- Proves that Jessica's current age is 40 given the problem conditions -/
theorem jessica_current_age :
  ∀ (jessica_age_at_death : ℕ) (mother_age_at_death : ℕ),
    jessica_age_at_death = mother_age_at_death / 2 →
    mother_age_at_death + 10 = 70 →
    jessica_age_at_death + 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_jessica_current_age_l2602_260284


namespace NUMINAMATH_CALUDE_four_is_eight_percent_of_fifty_l2602_260266

theorem four_is_eight_percent_of_fifty :
  (4 : ℝ) / 50 * 100 = 8 := by sorry

end NUMINAMATH_CALUDE_four_is_eight_percent_of_fifty_l2602_260266


namespace NUMINAMATH_CALUDE_chimpanzee_arrangements_l2602_260229

theorem chimpanzee_arrangements : 
  let word := "chimpanzee"
  let total_letters := word.length
  let unique_letters := word.toList.eraseDups.length
  let repeat_letter := 'e'
  let repeat_count := word.toList.filter (· == repeat_letter) |>.length
  (total_letters.factorial / repeat_count.factorial : ℕ) = 1814400 := by
  sorry

end NUMINAMATH_CALUDE_chimpanzee_arrangements_l2602_260229


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_24_l2602_260249

theorem smallest_divisible_by_18_and_24 : ∃ n : ℕ, n > 0 ∧ n % 18 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, m > 0 → m % 18 = 0 → m % 24 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_24_l2602_260249


namespace NUMINAMATH_CALUDE_tree_planting_group_size_l2602_260241

theorem tree_planting_group_size :
  ∀ x : ℕ,
  (7 * x + 9 > 9 * (x - 1)) →
  (7 * x + 9 < 9 * (x - 1) + 3) →
  x = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_planting_group_size_l2602_260241


namespace NUMINAMATH_CALUDE_min_product_reciprocal_sum_l2602_260238

theorem min_product_reciprocal_sum (a b : ℕ+) (h : (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = 1/9) : 
  (∀ c d : ℕ+, (c : ℚ)⁻¹ + (3 * d : ℚ)⁻¹ = 1/9 → c * d ≥ a * b) ∧ a * b = 108 := by
  sorry

end NUMINAMATH_CALUDE_min_product_reciprocal_sum_l2602_260238


namespace NUMINAMATH_CALUDE_carpet_area_proof_l2602_260254

/-- Calculates the total carpet area required for three rooms -/
def totalCarpetArea (w1 l1 w2 l2 w3 l3 : ℝ) : ℝ :=
  w1 * l1 + w2 * l2 + w3 * l3

/-- Proves that the total carpet area for the given room dimensions is 353 square feet -/
theorem carpet_area_proof :
  totalCarpetArea 12 15 7 9 10 11 = 353 := by
  sorry

#eval totalCarpetArea 12 15 7 9 10 11

end NUMINAMATH_CALUDE_carpet_area_proof_l2602_260254


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2602_260235

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 24| + |x - 30| = |3*x - 72| :=
by
  -- The unique solution is x = 26
  use 26
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2602_260235


namespace NUMINAMATH_CALUDE_face_value_is_75_l2602_260253

/-- Given banker's discount (BD) and true discount (TD), calculate the face value (FV) -/
def calculate_face_value (BD TD : ℚ) : ℚ :=
  (TD^2) / (BD - TD)

/-- Theorem stating that given BD = 18 and TD = 15, the face value is 75 -/
theorem face_value_is_75 :
  calculate_face_value 18 15 = 75 := by
  sorry

#eval calculate_face_value 18 15

end NUMINAMATH_CALUDE_face_value_is_75_l2602_260253


namespace NUMINAMATH_CALUDE_power_difference_l2602_260299

theorem power_difference (m n : ℕ) (h1 : 2^m = 32) (h2 : 3^n = 81) : 5^(m-n) = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l2602_260299


namespace NUMINAMATH_CALUDE_smallest_S_value_l2602_260280

def is_valid_arrangement (a b c d : Fin 4 → ℕ) : Prop :=
  ∀ i : Fin 16, ∃! j : Fin 4, ∃! k : Fin 4,
    i.val + 1 = a j ∨ i.val + 1 = b j ∨ i.val + 1 = c j ∨ i.val + 1 = d j

def S (a b c d : Fin 4 → ℕ) : ℕ :=
  (a 0) * (a 1) * (a 2) * (a 3) +
  (b 0) * (b 1) * (b 2) * (b 3) +
  (c 0) * (c 1) * (c 2) * (c 3) +
  (d 0) * (d 1) * (d 2) * (d 3)

theorem smallest_S_value :
  ∀ a b c d : Fin 4 → ℕ, is_valid_arrangement a b c d → S a b c d ≥ 2074 :=
by sorry

end NUMINAMATH_CALUDE_smallest_S_value_l2602_260280


namespace NUMINAMATH_CALUDE_intersection_point_determines_m_l2602_260236

-- Define the two lines
def line1 (x y m : ℝ) : Prop := 3 * x - 2 * y = m
def line2 (x y : ℝ) : Prop := -x - 2 * y = -10

-- Define the intersection point
def intersection (x y : ℝ) : Prop := line1 x y 6 ∧ line2 x y

-- Theorem statement
theorem intersection_point_determines_m :
  ∃ y : ℝ, intersection 4 y → (∀ m : ℝ, line1 4 y m → m = 6) := by sorry

end NUMINAMATH_CALUDE_intersection_point_determines_m_l2602_260236


namespace NUMINAMATH_CALUDE_star_properties_l2602_260245

-- Define the new operation "*"
def star (a b : ℚ) : ℚ := (2 + a) / b

-- Theorem statement
theorem star_properties :
  (star 4 (-3) = -2) ∧ (star 8 (star 4 3) = 5) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l2602_260245


namespace NUMINAMATH_CALUDE_expression_evaluation_l2602_260268

theorem expression_evaluation : 
  let x : ℤ := -3
  7 * x^2 - 3 * (2 * x^2 - 1) - 4 = 8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2602_260268


namespace NUMINAMATH_CALUDE_different_amounts_eq_127_l2602_260225

/-- Represents the number of coins of each denomination --/
structure CoinCounts where
  jiao_1 : Nat
  jiao_5 : Nat
  yuan_1 : Nat
  yuan_5 : Nat

/-- Calculates the number of different non-zero amounts that can be paid with the given coins --/
def differentAmounts (coins : CoinCounts) : Nat :=
  sorry

/-- The specific coin counts given in the problem --/
def problemCoins : CoinCounts :=
  { jiao_1 := 1
  , jiao_5 := 2
  , yuan_1 := 5
  , yuan_5 := 2 }

/-- Theorem stating that the number of different non-zero amounts is 127 --/
theorem different_amounts_eq_127 : differentAmounts problemCoins = 127 :=
  sorry

end NUMINAMATH_CALUDE_different_amounts_eq_127_l2602_260225


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l2602_260275

theorem least_n_satisfying_inequality : 
  (∀ k : ℕ, k > 0 → k < 4 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ 1 / 12) ∧ 
  ((1 : ℚ) / 4 - (1 : ℚ) / 5 < 1 / 12) := by
  sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l2602_260275


namespace NUMINAMATH_CALUDE_polygon_sides_l2602_260272

/-- The number of diagonals that can be drawn from one vertex of an n-sided polygon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- Theorem: If 2018 diagonals can be drawn from one vertex of an n-sided polygon, then n = 2021 -/
theorem polygon_sides (n : ℕ) (h : diagonals_from_vertex n = 2018) : n = 2021 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2602_260272


namespace NUMINAMATH_CALUDE_total_animal_sightings_l2602_260212

def week1_sightings : List Nat := [8, 7, 8, 11, 8, 7, 13]
def week2_sightings : List Nat := [7, 9, 10, 21, 11, 7, 17]

theorem total_animal_sightings :
  (week1_sightings.sum + week2_sightings.sum) = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_animal_sightings_l2602_260212


namespace NUMINAMATH_CALUDE_min_triangle_count_l2602_260231

structure Graph (n : ℕ) :=
  (m : ℕ)
  (edges : Finset (Fin n × Fin n))
  (edge_count : edges.card = m)
  (edge_distinct : ∀ (e : Fin n × Fin n), e ∈ edges → e.1 ≠ e.2)

def triangle_count (n : ℕ) (G : Graph n) : ℕ := sorry

theorem min_triangle_count (n : ℕ) (G : Graph n) :
  triangle_count n G ≥ (4 * G.m : ℚ) / (3 * n) * (G.m - n^2 / 4) :=
sorry

end NUMINAMATH_CALUDE_min_triangle_count_l2602_260231


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l2602_260203

theorem divisibility_equivalence (m n : ℕ) : 
  (((2^m : ℕ) - 1)^2 ∣ ((2^n : ℕ) - 1)) ↔ (m * ((2^m : ℕ) - 1) ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l2602_260203


namespace NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l2602_260230

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64 / 9)⌋ = -3 := by sorry

end NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l2602_260230


namespace NUMINAMATH_CALUDE_integral_sin_cubed_over_cos_fifth_l2602_260246

theorem integral_sin_cubed_over_cos_fifth (x : Real) :
  let f := fun (x : Real) => (1 / (4 * (Real.cos x)^4)) - (1 / (2 * (Real.cos x)^2))
  deriv f x = (Real.sin x)^3 / (Real.cos x)^5 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin_cubed_over_cos_fifth_l2602_260246


namespace NUMINAMATH_CALUDE_action_figures_removed_l2602_260264

/-- 
Given:
- initial_figures: The initial number of action figures
- added_figures: The number of action figures added
- final_figures: The final number of action figures on the shelf

Prove that the number of removed figures is 1.
-/
theorem action_figures_removed 
  (initial_figures : ℕ) 
  (added_figures : ℕ) 
  (final_figures : ℕ) 
  (h1 : initial_figures = 3)
  (h2 : added_figures = 4)
  (h3 : final_figures = 6) :
  initial_figures + added_figures - final_figures = 1 := by
  sorry

end NUMINAMATH_CALUDE_action_figures_removed_l2602_260264


namespace NUMINAMATH_CALUDE_sum_opposite_and_sqrt_81_l2602_260270

-- Define the function for the sum
def sum_opposite_and_sqrt : ℝ → Set ℝ :=
  λ x => {2 + Real.sqrt 2, Real.sqrt 2 - 4}

-- State the theorem
theorem sum_opposite_and_sqrt_81 :
  sum_opposite_and_sqrt (Real.sqrt 81) = {2 + Real.sqrt 2, Real.sqrt 2 - 4} :=
by sorry

end NUMINAMATH_CALUDE_sum_opposite_and_sqrt_81_l2602_260270


namespace NUMINAMATH_CALUDE_attic_boxes_count_l2602_260234

/-- Represents the problem of arranging teacups in an attic --/
def TeacupArrangement (B : ℕ) : Prop :=
  let boxes_without_pans := B - 6
  let boxes_with_teacups := boxes_without_pans / 2
  let cups_per_box := 5 * 4
  let broken_cups := 2 * boxes_with_teacups
  let original_cups := cups_per_box * boxes_with_teacups
  original_cups = 180 + broken_cups

/-- Theorem stating that there are 26 boxes in the attic --/
theorem attic_boxes_count : ∃ B : ℕ, TeacupArrangement B ∧ B = 26 := by
  sorry

end NUMINAMATH_CALUDE_attic_boxes_count_l2602_260234


namespace NUMINAMATH_CALUDE_midpoint_sum_coordinates_l2602_260219

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (2, 3) and (8, 15) is 14. -/
theorem midpoint_sum_coordinates : 
  let x₁ : ℝ := 2
  let y₁ : ℝ := 3
  let x₂ : ℝ := 8
  let y₂ : ℝ := 15
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 14 := by
  sorry


end NUMINAMATH_CALUDE_midpoint_sum_coordinates_l2602_260219


namespace NUMINAMATH_CALUDE_sin_double_angle_for_line_l2602_260221

/-- Given a line with equation 2x-4y+5=0 and angle of inclination α, prove that sin2α = 4/5 -/
theorem sin_double_angle_for_line (x y : ℝ) (α : ℝ) 
  (h : 2 * x - 4 * y + 5 = 0) 
  (h_incline : α = Real.arctan (1 / 2)) : 
  Real.sin (2 * α) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_for_line_l2602_260221


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l2602_260204

theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 4 →                     -- first term condition
  q ≠ 1 →                       -- common ratio condition
  2 * a 5 = 4 * a 1 - 2 * a 3 → -- arithmetic sequence condition
  q = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l2602_260204


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2602_260260

def sachin_age : ℚ := 38.5
def age_difference : ℕ := 7

def rahul_age : ℚ := sachin_age - age_difference

theorem age_ratio_proof :
  (sachin_age * 2 / rahul_age * 2).num = 11 ∧
  (sachin_age * 2 / rahul_age * 2).den = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l2602_260260


namespace NUMINAMATH_CALUDE_iris_mall_spending_l2602_260285

/-- Represents the total amount spent by Iris at the mall -/
def total_spent (jacket_price shorts_price pants_price : ℕ) 
                (jacket_count shorts_count pants_count : ℕ) : ℕ :=
  jacket_price * jacket_count + shorts_price * shorts_count + pants_price * pants_count

/-- Proves that Iris spent $90 at the mall -/
theorem iris_mall_spending : 
  total_spent 10 6 12 3 2 4 = 90 := by
  sorry

end NUMINAMATH_CALUDE_iris_mall_spending_l2602_260285


namespace NUMINAMATH_CALUDE_train_crossing_time_specific_train_crossing_time_l2602_260201

/-- The time (in seconds) it takes for a train to cross a man walking in the same direction --/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let train_speed_ms := train_speed * 1000 / 3600
  let man_speed_ms := man_speed * 1000 / 3600
  let relative_speed := train_speed_ms - man_speed_ms
  train_length / relative_speed

/-- The specific problem instance --/
theorem specific_train_crossing_time :
  train_crossing_time 900 63 3 = 54 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_specific_train_crossing_time_l2602_260201


namespace NUMINAMATH_CALUDE_total_birds_is_168_l2602_260296

/-- Represents the number of birds of each species -/
structure BirdCounts where
  bluebirds : ℕ
  cardinals : ℕ
  goldfinches : ℕ
  sparrows : ℕ
  swallows : ℕ

/-- Conditions for the bird counts -/
def validBirdCounts (b : BirdCounts) : Prop :=
  b.cardinals = 2 * b.bluebirds ∧
  b.goldfinches = 4 * b.bluebirds ∧
  b.sparrows = (b.cardinals + b.goldfinches) / 2 ∧
  b.swallows = 8 ∧
  b.bluebirds = 2 * b.swallows

/-- The total number of birds -/
def totalBirds (b : BirdCounts) : ℕ :=
  b.bluebirds + b.cardinals + b.goldfinches + b.sparrows + b.swallows

/-- Theorem: The total number of birds is 168 -/
theorem total_birds_is_168 :
  ∀ b : BirdCounts, validBirdCounts b → totalBirds b = 168 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_is_168_l2602_260296


namespace NUMINAMATH_CALUDE_ratio_of_a_to_c_l2602_260277

theorem ratio_of_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 7 / 3)
  (hdb : d / b = 5 / 4) :
  a / c = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_c_l2602_260277


namespace NUMINAMATH_CALUDE_journey_duration_l2602_260243

/-- Given a journey with two parts, prove that the duration of the first part is 3 hours. -/
theorem journey_duration (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : total_distance = 240)
  (h2 : total_time = 5)
  (h3 : speed1 = 40)
  (h4 : speed2 = 60)
  (h5 : ∃ (distance1 : ℝ), 
    distance1 / speed1 + (total_distance - distance1) / speed2 = total_time) :
  ∃ (duration1 : ℝ), duration1 = 3 ∧ duration1 * speed1 + (total_time - duration1) * speed2 = total_distance :=
by sorry

end NUMINAMATH_CALUDE_journey_duration_l2602_260243


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2602_260290

/-- Given a right triangle, if rotating it about one leg produces a cone of volume 972π cm³
    and rotating it about the other leg produces a cone of volume 1458π cm³,
    then the length of the hypotenuse is 12√5 cm. -/
theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  (1/3) * π * a * b^2 = 972 * π →
  (1/3) * π * b * a^2 = 1458 * π →
  c = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2602_260290


namespace NUMINAMATH_CALUDE_min_value_fraction_l2602_260205

theorem min_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) :
  (x + y) / (x^2) ≥ -1/12 := by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2602_260205


namespace NUMINAMATH_CALUDE_percentage_of_boys_from_school_A_l2602_260200

theorem percentage_of_boys_from_school_A (total_boys : ℕ) (boys_A_not_science : ℕ) 
  (h1 : total_boys = 450)
  (h2 : boys_A_not_science = 63)
  (h3 : (30 : ℚ) / 100 = 1 - (boys_A_not_science : ℚ) / ((20 : ℚ) / 100 * total_boys)) :
  (20 : ℚ) / 100 = (boys_A_not_science : ℚ) / (0.7 * total_boys) :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_boys_from_school_A_l2602_260200


namespace NUMINAMATH_CALUDE_square_diagonal_half_l2602_260226

/-- Given a square with side length 6 cm and AE = 8 cm, prove that OB = 4.5 cm -/
theorem square_diagonal_half (side_length : ℝ) (AE : ℝ) (OB : ℝ) :
  side_length = 6 →
  AE = 8 →
  OB = side_length * Real.sqrt 2 / 2 →
  OB = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_half_l2602_260226


namespace NUMINAMATH_CALUDE_discriminant_zero_iff_unique_solution_unique_solution_iff_m_eq_three_l2602_260295

/-- A quadratic equation ax^2 + bx + c = 0 has exactly one solution if and only if its discriminant is zero -/
theorem discriminant_zero_iff_unique_solution (a b c : ℝ) (ha : a ≠ 0) :
  (b^2 - 4*a*c = 0) ↔ (∃! x, a*x^2 + b*x + c = 0) :=
sorry

/-- The quadratic equation 3x^2 - 6x + m = 0 has exactly one solution if and only if m = 3 -/
theorem unique_solution_iff_m_eq_three :
  (∃! x, 3*x^2 - 6*x + m = 0) ↔ m = 3 :=
sorry

end NUMINAMATH_CALUDE_discriminant_zero_iff_unique_solution_unique_solution_iff_m_eq_three_l2602_260295


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2602_260262

def p (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2602_260262


namespace NUMINAMATH_CALUDE_parallelogram_segment_sum_l2602_260293

/-- A grid of equilateral triangles -/
structure TriangularGrid where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A parallelogram on the triangular grid -/
structure Parallelogram (grid : TriangularGrid) where
  vertices : Fin 4 → ℕ × ℕ  -- Grid coordinates of the vertices
  area : ℝ

/-- The possible sums of lengths of grid segments inside the parallelogram -/
def possible_segment_sums (grid : TriangularGrid) (p : Parallelogram grid) : Set ℝ :=
  {3, 4, 5, 6}

theorem parallelogram_segment_sum 
  (grid : TriangularGrid) 
  (p : Parallelogram grid) 
  (h_side_length : grid.side_length = 1) 
  (h_area : p.area = Real.sqrt 3) :
  ∃ (sum : ℝ), sum ∈ possible_segment_sums grid p :=
sorry

end NUMINAMATH_CALUDE_parallelogram_segment_sum_l2602_260293


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l2602_260282

theorem min_value_trig_expression (A : Real) (h : 0 < A ∧ A < Real.pi / 2) :
  Real.sqrt (Real.sin A ^ 4 + 1) + Real.sqrt (Real.cos A ^ 4 + 4) ≥ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l2602_260282


namespace NUMINAMATH_CALUDE_solve_for_a_l2602_260210

-- Define the function f
def f (a c x : ℝ) : ℝ := a * x^2 + c

-- Define the derivative of f
def f_derivative (a : ℝ) : ℝ → ℝ := λ x ↦ 2 * a * x

-- Theorem statement
theorem solve_for_a (a c : ℝ) : f_derivative a 1 = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2602_260210


namespace NUMINAMATH_CALUDE_largest_band_formation_l2602_260289

/-- Represents a rectangular band formation -/
structure BandFormation where
  m : ℕ  -- Total number of band members
  r : ℕ  -- Number of rows
  x : ℕ  -- Number of members in each row

/-- Checks if a band formation is valid according to the problem conditions -/
def isValidFormation (f : BandFormation) : Prop :=
  f.r * f.x + 5 = f.m ∧
  (f.r - 3) * (f.x + 2) = f.m ∧
  f.m < 100

/-- The theorem stating the largest possible number of band members -/
theorem largest_band_formation :
  ∃ (f : BandFormation), isValidFormation f ∧
    ∀ (g : BandFormation), isValidFormation g → g.m ≤ f.m :=
by sorry

end NUMINAMATH_CALUDE_largest_band_formation_l2602_260289


namespace NUMINAMATH_CALUDE_expression_bounds_l2602_260279

theorem expression_bounds (a b c d e : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) : 
  2 * Real.sqrt 2 ≤ Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + 
    Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ∧
  Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + 
    Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ≤ 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l2602_260279


namespace NUMINAMATH_CALUDE_parade_vehicles_l2602_260250

theorem parade_vehicles (b t q : ℕ) : 
  b + t + q = 12 →
  2*b + 3*t + 4*q = 35 →
  q = 5 :=
by sorry

end NUMINAMATH_CALUDE_parade_vehicles_l2602_260250


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l2602_260211

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0. The least significant bit is at the head of the list. -/
def BinaryNum := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNum) : ℕ :=
  b.enum.foldr (λ (i, bit) acc => acc + if bit then 2^i else 0) 0

/-- Converts a decimal number to its binary representation -/
def decimal_to_binary (n : ℕ) : BinaryNum :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : ℕ) : BinaryNum :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNum) : BinaryNum :=
  decimal_to_binary (binary_to_decimal a * binary_to_decimal b)

theorem binary_multiplication_theorem :
  let a : BinaryNum := [true, true, false, true, true]  -- 11011₂
  let b : BinaryNum := [true, false, true]              -- 101₂
  let result : BinaryNum := [true, true, true, false, true, true, false, false, true]  -- 100110111₂
  binary_multiply a b = result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l2602_260211


namespace NUMINAMATH_CALUDE_composite_number_impossibility_l2602_260297

theorem composite_number_impossibility (n a q : ℕ) (h_n : n > 1) (h_q_prime : Nat.Prime q) 
  (h_q_div : q ∣ (n - 1)) (h_q_sqrt : q > Nat.sqrt n - 1) (h_n_div : n ∣ (a^(n-1) - 1)) 
  (h_gcd : Nat.gcd (a^((n-1)/q) - 1) n = 1) : 
  Nat.Prime n := by
sorry

end NUMINAMATH_CALUDE_composite_number_impossibility_l2602_260297


namespace NUMINAMATH_CALUDE_school_population_l2602_260202

/-- Given a school with 42 boys and a boy-to-girl ratio of 7:1, 
    prove that the total number of students is 48. -/
theorem school_population (num_boys : ℕ) (ratio : ℚ) : 
  num_boys = 42 → ratio = 7/1 → num_boys + (num_boys / ratio.num) = 48 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l2602_260202


namespace NUMINAMATH_CALUDE_envelope_game_properties_l2602_260265

/-- A game with envelopes and two evenly matched teams -/
structure EnvelopeGame where
  num_envelopes : ℕ
  win_points : ℕ
  win_probability : ℝ

/-- Calculate the expected number of points for one team in a single game -/
noncomputable def expected_points (game : EnvelopeGame) : ℝ :=
  sorry

/-- Calculate the probability of a specific envelope being chosen in a game -/
noncomputable def envelope_probability (game : EnvelopeGame) : ℝ :=
  sorry

/-- Theorem about the expected points and envelope probability in the specific game -/
theorem envelope_game_properties :
  let game : EnvelopeGame := ⟨13, 6, 1/2⟩
  (100 * expected_points game = 465) ∧
  (envelope_probability game = 12/13) := by
  sorry

end NUMINAMATH_CALUDE_envelope_game_properties_l2602_260265


namespace NUMINAMATH_CALUDE_percentage_increase_problem_l2602_260278

theorem percentage_increase_problem (a b x m : ℝ) (k : ℝ) (h1 : a > 0) (h2 : b > 0) :
  a = 4 * k ∧ b = 5 * k ∧ k > 0 →
  (∃ p : ℝ, x = a * (1 + p / 100)) →
  m = b * 0.4 →
  m / x = 0.4 →
  ∃ p : ℝ, x = a * (1 + p / 100) ∧ p = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_problem_l2602_260278


namespace NUMINAMATH_CALUDE_algae_coverage_day_l2602_260228

/-- Represents the coverage of algae in the lake on a given day -/
def algaeCoverage (day : ℕ) : ℚ :=
  1 / 2^(30 - day)

/-- The problem statement -/
theorem algae_coverage_day : ∃ d : ℕ, d ≤ 30 ∧ algaeCoverage d < (1/10) ∧ (1/10) ≤ algaeCoverage (d+1) :=
  sorry

end NUMINAMATH_CALUDE_algae_coverage_day_l2602_260228


namespace NUMINAMATH_CALUDE_octal_123_equals_decimal_83_l2602_260227

/-- Converts an octal digit to its decimal equivalent -/
def octal_to_decimal (digit : ℕ) : ℕ := digit

/-- Represents an octal number as a list of natural numbers -/
def octal_number : List ℕ := [1, 2, 3]

/-- Converts an octal number to its decimal equivalent -/
def octal_to_decimal_conversion (octal : List ℕ) : ℕ :=
  octal.enum.foldr (fun (i, digit) acc => acc + octal_to_decimal digit * 8^i) 0

theorem octal_123_equals_decimal_83 :
  octal_to_decimal_conversion octal_number = 83 := by
  sorry

end NUMINAMATH_CALUDE_octal_123_equals_decimal_83_l2602_260227


namespace NUMINAMATH_CALUDE_faucet_leak_approx_l2602_260274

/-- The volume of water leaked by an untightened faucet in 4 hours -/
def faucet_leak_volume : ℝ :=
  let drops_per_second : ℝ := 2
  let milliliters_per_drop : ℝ := 0.05
  let hours : ℝ := 4
  let seconds_per_hour : ℝ := 3600
  drops_per_second * milliliters_per_drop * hours * seconds_per_hour

/-- Assertion that the faucet leak volume is approximately 1.4 × 10^3 milliliters -/
theorem faucet_leak_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 10 ∧ |faucet_leak_volume - 1.4e3| < ε :=
sorry

end NUMINAMATH_CALUDE_faucet_leak_approx_l2602_260274


namespace NUMINAMATH_CALUDE_log_product_equality_l2602_260255

theorem log_product_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.log x^2 / Real.log y^5 * 
  Real.log y^3 / Real.log x^4 * 
  Real.log x^4 / Real.log y^3 * 
  Real.log y^5 / Real.log x^3 * 
  Real.log x^3 / Real.log y^4 = 
  (1 / 6) * (Real.log x / Real.log y) :=
sorry

end NUMINAMATH_CALUDE_log_product_equality_l2602_260255


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l2602_260216

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l2602_260216


namespace NUMINAMATH_CALUDE_no_rational_roots_for_three_digit_prime_quadratic_l2602_260208

def is_three_digit_prime (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ Nat.Prime n

def digits_of_three_digit_number (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (a, b, c)

theorem no_rational_roots_for_three_digit_prime_quadratic :
  ∀ A : ℕ, is_three_digit_prime A →
    let (a, b, c) := digits_of_three_digit_number A
    ∀ x : ℚ, a * x^2 + b * x + c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_rational_roots_for_three_digit_prime_quadratic_l2602_260208


namespace NUMINAMATH_CALUDE_y_plus_z_negative_l2602_260291

theorem y_plus_z_negative (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hy : 0 < y ∧ y < 1) 
  (hz : -2 < z ∧ z < -1) : 
  y + z < 0 := by
  sorry

end NUMINAMATH_CALUDE_y_plus_z_negative_l2602_260291


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2602_260276

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := 2 * x^2 - y^2 = 8

-- Define the length of the real axis
def real_axis_length : ℝ := 4

-- Theorem statement
theorem hyperbola_real_axis_length :
  ∀ x y : ℝ, hyperbola_equation x y → real_axis_length = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2602_260276


namespace NUMINAMATH_CALUDE_exponent_division_l2602_260257

theorem exponent_division (a : ℝ) : a^6 / a^4 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2602_260257


namespace NUMINAMATH_CALUDE_square_pyramid_sphere_ratio_l2602_260281

/-- Represents a pyramid with a square base -/
structure SquarePyramid where
  -- Length of the edge of the square base
  baseEdge : ℝ
  -- Height of the pyramid (perpendicular distance from apex to base)
  height : ℝ

/-- Calculates the ratio of surface areas of circumscribed to inscribed spheres for a square pyramid -/
def sphereAreaRatio (p : SquarePyramid) : ℝ :=
  -- This function would contain the actual calculation
  sorry

theorem square_pyramid_sphere_ratio :
  let p := SquarePyramid.mk 8 6
  sphereAreaRatio p = 41 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_sphere_ratio_l2602_260281


namespace NUMINAMATH_CALUDE_backpacking_roles_l2602_260256

theorem backpacking_roles (n : ℕ) (h : n = 10) : 
  (n.choose 2) * ((n - 2).choose 1) = 360 := by
  sorry

end NUMINAMATH_CALUDE_backpacking_roles_l2602_260256


namespace NUMINAMATH_CALUDE_double_force_quadruple_power_l2602_260247

/-- Represents the scenario of tugboats pushing a barge -/
structure TugboatScenario where
  k : ℝ  -- Constant of proportionality for water resistance
  F : ℝ  -- Initial force applied by one tugboat
  v : ℝ  -- Initial speed of the barge

/-- Calculates the power expended given force and velocity -/
def power (force velocity : ℝ) : ℝ := force * velocity

/-- Theorem stating that doubling the force quadruples the power when water resistance is proportional to speed -/
theorem double_force_quadruple_power (scenario : TugboatScenario) :
  let initial_power := power scenario.F scenario.v
  let final_power := power (2 * scenario.F) ((2 * scenario.F) / scenario.k)
  final_power = 4 * initial_power := by
  sorry


end NUMINAMATH_CALUDE_double_force_quadruple_power_l2602_260247


namespace NUMINAMATH_CALUDE_correct_parentheses_removal_l2602_260273

theorem correct_parentheses_removal (a b c d : ℝ) : 
  (a^2 - (1 - 2*a) ≠ a^2 - 1 - 2*a) ∧ 
  (a^2 + (-1 - 2*a) ≠ a^2 - 1 + 2*a) ∧ 
  (a - (5*b - (2*c - 1)) = a - 5*b + 2*c - 1) ∧ 
  (-(a + b) + (c - d) ≠ -a - b + c + d) :=
by sorry

end NUMINAMATH_CALUDE_correct_parentheses_removal_l2602_260273


namespace NUMINAMATH_CALUDE_two_numbers_product_sum_l2602_260292

theorem two_numbers_product_sum (n : Nat) : n = 45 →
  ∃ x y : Nat, x ∈ Finset.range (n + 1) ∧ 
             y ∈ Finset.range (n + 1) ∧ 
             x < y ∧
             (Finset.sum (Finset.range (n + 1)) id - x - y = x * y) ∧
             y - x = 9 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_product_sum_l2602_260292


namespace NUMINAMATH_CALUDE_bank_line_time_l2602_260244

/-- Given a constant speed calculated from moving 20 meters in 40 minutes,
    prove that the time required to move an additional 100 meters is 200 minutes. -/
theorem bank_line_time (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ)
    (h1 : initial_distance = 20)
    (h2 : initial_time = 40)
    (h3 : additional_distance = 100) :
    (additional_distance / (initial_distance / initial_time)) = 200 := by
  sorry

end NUMINAMATH_CALUDE_bank_line_time_l2602_260244


namespace NUMINAMATH_CALUDE_f_properties_l2602_260217

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x * x^2

theorem f_properties :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 →
  (∀ x ∈ Set.Ioo 0 (Real.exp (-1/2)), f x ≥ f (Real.exp (-1/2))) ∧
  (∀ x ∈ Set.Ioi (Real.exp (-1/2)), f x ≥ f (Real.exp (-1/2))) ∧
  f (Real.exp (-1/2)) = -1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2602_260217


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_l2602_260288

/-- A quadratic function with vertex (h, k) and y-intercept (0, y0) -/
def QuadraticFunction (a b c h k y0 : ℝ) : Prop :=
  ∀ x, a * x^2 + b * x + c = a * (x - h)^2 + k ∧
  c = y0 ∧
  -b / (2 * a) = h ∧
  a * h^2 + b * h + c = k

theorem quadratic_function_uniqueness (a b c : ℝ) : 
  QuadraticFunction a b c 2 (-1) 11 → a = 3 ∧ b = -12 ∧ c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_uniqueness_l2602_260288


namespace NUMINAMATH_CALUDE_escalator_travel_time_l2602_260263

/-- Proves that a person walking on a moving escalator takes 8 seconds to cover its length --/
theorem escalator_travel_time 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (person_speed : ℝ) 
  (h1 : escalator_speed = 10) 
  (h2 : escalator_length = 112) 
  (h3 : person_speed = 4) : 
  escalator_length / (escalator_speed + person_speed) = 8 := by
  sorry

end NUMINAMATH_CALUDE_escalator_travel_time_l2602_260263


namespace NUMINAMATH_CALUDE_smallest_multiple_of_30_and_40_not_16_l2602_260258

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_multiple_of_30_and_40_not_16 : 
  (∀ n : ℕ, n > 0 ∧ is_multiple n 30 ∧ is_multiple n 40 ∧ ¬is_multiple n 16 → n ≥ 120) ∧ 
  (is_multiple 120 30 ∧ is_multiple 120 40 ∧ ¬is_multiple 120 16) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_30_and_40_not_16_l2602_260258


namespace NUMINAMATH_CALUDE_flu_infection_equation_l2602_260248

theorem flu_infection_equation (x : ℝ) : 
  (∃ (initial_infected : ℕ) (rounds : ℕ) (total_infected : ℕ),
    initial_infected = 1 ∧ 
    rounds = 2 ∧ 
    total_infected = 64 ∧ 
    (∀ r : ℕ, r ≤ rounds → 
      (initial_infected * (1 + x)^r = initial_infected * (total_infected / initial_infected)^(r/rounds))))
  → (1 + x)^2 = 64 :=
by sorry

end NUMINAMATH_CALUDE_flu_infection_equation_l2602_260248


namespace NUMINAMATH_CALUDE_chocolate_bars_per_box_l2602_260259

theorem chocolate_bars_per_box 
  (total_bars : ℕ) 
  (total_boxes : ℕ) 
  (h1 : total_bars = 710) 
  (h2 : total_boxes = 142) : 
  total_bars / total_boxes = 5 := by
sorry

end NUMINAMATH_CALUDE_chocolate_bars_per_box_l2602_260259


namespace NUMINAMATH_CALUDE_solve_equation_l2602_260232

theorem solve_equation : 
  ∃ y : ℚ, (y^2 - 9*y + 8)/(y - 1) + (3*y^2 + 16*y - 12)/(3*y - 2) = -3 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2602_260232


namespace NUMINAMATH_CALUDE_uber_cost_is_22_l2602_260207

/-- The cost of a taxi ride --/
def taxi_cost : ℝ := 15

/-- The cost of a Lyft ride --/
def lyft_cost : ℝ := taxi_cost + 4

/-- The cost of an Uber ride --/
def uber_cost : ℝ := lyft_cost + 3

/-- The total cost of a taxi ride including a 20% tip --/
def taxi_total_cost : ℝ := taxi_cost * 1.2

theorem uber_cost_is_22 :
  (taxi_total_cost = 18) →
  (uber_cost = 22) :=
by
  sorry

#eval uber_cost

end NUMINAMATH_CALUDE_uber_cost_is_22_l2602_260207


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l2602_260294

/-- The line x + y = c is a perpendicular bisector of the line segment from (2,4) to (6,8) -/
def is_perpendicular_bisector (c : ℝ) : Prop :=
  let midpoint := ((2 + 6) / 2, (4 + 8) / 2)
  (midpoint.1 + midpoint.2 = c) ∧
  (∀ (x y : ℝ), x + y = c → (x - 2)^2 + (y - 4)^2 = (x - 6)^2 + (y - 8)^2)

/-- If the line x + y = c is a perpendicular bisector of the line segment from (2,4) to (6,8), then c = 10 -/
theorem perpendicular_bisector_c_value :
  ∃ c, is_perpendicular_bisector c → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l2602_260294


namespace NUMINAMATH_CALUDE_razorback_tshirt_profit_l2602_260218

/-- The Razorback T-shirt Shop problem -/
theorem razorback_tshirt_profit :
  let total_shirts : ℕ := 245
  let total_revenue : ℚ := 2205
  let profit_per_shirt : ℚ := total_revenue / total_shirts
  profit_per_shirt = 9 := by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_profit_l2602_260218


namespace NUMINAMATH_CALUDE_triangle_min_perimeter_l2602_260267

theorem triangle_min_perimeter (a b c : ℕ) : 
  a = 24 → b = 37 → c > 0 → 
  (a + b > c ∧ a + c > b ∧ b + c > a) →
  (∀ x : ℕ, x > 0 → x + b > a ∧ a + b > x ∧ a + x > b → a + b + x ≥ a + b + c) →
  a + b + c = 75 := by sorry

end NUMINAMATH_CALUDE_triangle_min_perimeter_l2602_260267


namespace NUMINAMATH_CALUDE_lcm_of_numbers_in_ratio_l2602_260252

def are_in_ratio (a b c : ℕ) (x y z : ℕ) : Prop :=
  ∃ (k : ℕ), a = k * x ∧ b = k * y ∧ c = k * z

theorem lcm_of_numbers_in_ratio (a b c : ℕ) 
  (h_ratio : are_in_ratio a b c 5 7 9)
  (h_hcf : Nat.gcd a (Nat.gcd b c) = 11) :
  Nat.lcm a (Nat.lcm b c) = 99 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_numbers_in_ratio_l2602_260252


namespace NUMINAMATH_CALUDE_tree_count_after_planting_l2602_260222

theorem tree_count_after_planting (road_length : ℕ) (original_spacing : ℕ) (additional_trees : ℕ) : 
  road_length = 7200 → 
  original_spacing = 120 → 
  additional_trees = 5 → 
  (road_length / original_spacing * (additional_trees + 1) + 1) * 2 = 722 := by
sorry

end NUMINAMATH_CALUDE_tree_count_after_planting_l2602_260222


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2602_260287

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (l : ℕ) (d : ℕ) : ℕ :=
  let n : ℕ := (l - a) / d + 1
  n * (a + l) / 2

/-- Theorem: The sum of the arithmetic sequence with first term 2, last term 102, and common difference 5 is 1092 -/
theorem arithmetic_sequence_sum :
  arithmetic_sum 2 102 5 = 1092 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2602_260287


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l2602_260215

/-- Given two books with a total cost of 600, where one is sold at a loss and the other at a 19% gain,
    both sold at the same price, and the cost of the book sold at a loss is 350,
    prove that the loss percentage on the first book is 15%. -/
theorem book_sale_loss_percentage : 
  ∀ (total_cost cost_book1 cost_book2 selling_price gain_percentage : ℚ),
  total_cost = 600 →
  cost_book1 = 350 →
  cost_book2 = total_cost - cost_book1 →
  gain_percentage = 19 →
  selling_price = cost_book2 * (1 + gain_percentage / 100) →
  (cost_book1 - selling_price) / cost_book1 * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l2602_260215


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_for_1000_l2602_260242

def is_sum_of_distinct_powers_of_two (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (λ e => 2^e)).sum ∧ exponents.Nodup

theorem least_sum_of_exponents_for_1000 :
  ∀ exponents : List ℕ,
    is_sum_of_distinct_powers_of_two 1000 exponents →
    exponents.length ≥ 3 →
    exponents.sum ≥ 38 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_for_1000_l2602_260242


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2602_260283

def vector_a : Fin 2 → ℝ := ![1, 1]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![2, x]

theorem parallel_vectors_x_value (x : ℝ) :
  (∃ (k : ℝ), k ≠ 0 ∧ (vector_a + vector_b x) = k • (vector_a - vector_b x)) →
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2602_260283


namespace NUMINAMATH_CALUDE_line_equation_l2602_260213

/-- A line parameterized by (x,y) = (3t + 6, 5t - 7) where t is a real number -/
def parameterized_line (t : ℝ) : ℝ × ℝ := (3 * t + 6, 5 * t - 7)

/-- The slope-intercept form of a line -/
def slope_intercept_form (m b : ℝ) (x : ℝ) : ℝ := m * x + b

theorem line_equation :
  ∀ (t x y : ℝ), parameterized_line t = (x, y) →
  y = slope_intercept_form (5/3) (-17) x := by
sorry

end NUMINAMATH_CALUDE_line_equation_l2602_260213


namespace NUMINAMATH_CALUDE_expression_simplification_l2602_260298

theorem expression_simplification (y : ℝ) : 3*y + 4*y^2 + 2 - (8 - 3*y - 4*y^2) = 8*y^2 + 6*y - 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2602_260298


namespace NUMINAMATH_CALUDE_circle_area_difference_l2602_260214

theorem circle_area_difference : 
  let r1 : ℝ := 20
  let d2 : ℝ := 20
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 300 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l2602_260214


namespace NUMINAMATH_CALUDE_select_three_from_five_l2602_260206

theorem select_three_from_five (n : ℕ) (h : n = 5) : 
  n * (n - 1) * (n - 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_select_three_from_five_l2602_260206


namespace NUMINAMATH_CALUDE_prob_miss_at_least_once_prob_A_twice_B_once_l2602_260240

-- Define the probabilities of hitting the target
def prob_hit_A : ℚ := 2/3
def prob_hit_B : ℚ := 3/4

-- Define the number of shots for each part
def shots_part1 : ℕ := 3
def shots_part2 : ℕ := 2

-- Assume independence of shots
axiom independence : ∀ (n : ℕ), (prob_hit_A ^ n) = prob_hit_A * (prob_hit_A ^ (n - 1))

-- Part 1: Probability that Person A misses at least once in 3 shots
theorem prob_miss_at_least_once : 
  1 - (prob_hit_A ^ shots_part1) = 19/27 := by sorry

-- Part 2: Probability that A hits exactly twice and B hits exactly once in 2 shots each
theorem prob_A_twice_B_once :
  (prob_hit_A ^ 2) * (2 * prob_hit_B * (1 - prob_hit_B)) = 1/6 := by sorry

end NUMINAMATH_CALUDE_prob_miss_at_least_once_prob_A_twice_B_once_l2602_260240


namespace NUMINAMATH_CALUDE_votes_against_percentage_l2602_260286

theorem votes_against_percentage (total_votes : ℕ) (difference : ℕ) :
  total_votes = 330 →
  difference = 66 →
  let votes_against := (total_votes - difference) / 2
  let percentage_against := (votes_against : ℚ) / total_votes * 100
  percentage_against = 40 := by
  sorry

end NUMINAMATH_CALUDE_votes_against_percentage_l2602_260286
