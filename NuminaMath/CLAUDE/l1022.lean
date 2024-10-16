import Mathlib

namespace NUMINAMATH_CALUDE_final_savings_is_105_l1022_102219

/-- Calculates the final savings amount after a series of bets and savings --/
def finalSavings (initialWinnings : ℝ) : ℝ :=
  let firstSavings := initialWinnings * 0.5
  let secondBetAmount := initialWinnings * 0.5
  let secondBetProfit := secondBetAmount * 0.6
  let secondBetTotal := secondBetAmount + secondBetProfit
  let secondSavings := secondBetTotal * 0.5
  let remainingAfterSecond := secondBetTotal
  let thirdBetAmount := remainingAfterSecond * 0.3
  let thirdBetProfit := thirdBetAmount * 0.25
  let thirdBetTotal := thirdBetAmount + thirdBetProfit
  let thirdSavings := thirdBetTotal * 0.5
  firstSavings + secondSavings + thirdSavings

/-- The theorem stating that the final savings amount is $105.00 --/
theorem final_savings_is_105 :
  finalSavings 100 = 105 := by
  sorry

end NUMINAMATH_CALUDE_final_savings_is_105_l1022_102219


namespace NUMINAMATH_CALUDE_trailing_zeros_bound_l1022_102200

theorem trailing_zeros_bound (n : ℕ) : ∃ (k : ℕ), k ≤ 2 ∧ (1^n + 2^n + 3^n + 4^n) % 10^(k+1) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_bound_l1022_102200


namespace NUMINAMATH_CALUDE_tree_leaf_drop_l1022_102269

theorem tree_leaf_drop (initial_leaves : ℕ) (final_drop : ℕ) : 
  initial_leaves = 340 → 
  final_drop = 204 → 
  ∃ (n : ℕ), n = 4 ∧ 
    initial_leaves * (9/10)^n = final_drop ∧ 
    ∀ (k : ℕ), k < n → initial_leaves * (9/10)^k > final_drop :=
by sorry

end NUMINAMATH_CALUDE_tree_leaf_drop_l1022_102269


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1022_102252

theorem inequality_solution_range (a b x : ℝ) : 
  (a > 0 ∧ b > 0) → 
  (∀ a b, a > 0 → b > 0 → x^2 + 2*x < a/b + 16*b/a) ↔ 
  (-4 < x ∧ x < 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1022_102252


namespace NUMINAMATH_CALUDE_cos_negative_thirty_degrees_l1022_102229

theorem cos_negative_thirty_degrees : Real.cos (-(30 * π / 180)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_thirty_degrees_l1022_102229


namespace NUMINAMATH_CALUDE_snow_probability_l1022_102297

theorem snow_probability (p : ℝ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l1022_102297


namespace NUMINAMATH_CALUDE_system_solution_l1022_102247

theorem system_solution :
  ∃ (x y : ℚ), 
    (4 * x - 3 * y = -7) ∧ 
    (5 * x + 4 * y = -6) ∧ 
    (x = -46 / 31) ∧ 
    (y = 11 / 31) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1022_102247


namespace NUMINAMATH_CALUDE_not_divisible_by_2n_plus_65_l1022_102277

theorem not_divisible_by_2n_plus_65 (n : ℕ+) : ¬(2^n.val + 65 ∣ 5^n.val - 3^n.val) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_2n_plus_65_l1022_102277


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1022_102287

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 4) : 
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1022_102287


namespace NUMINAMATH_CALUDE_fourth_root_cube_problem_l1022_102203

theorem fourth_root_cube_problem : 
  (((2 * Real.sqrt 2) ^ 3) ^ (1/4)) ^ 3 = 16 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_fourth_root_cube_problem_l1022_102203


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l1022_102266

/-- Given a rectangle with area 450 square centimeters, if its length is increased by 20% and its width by 30%, the new area will be 702 square centimeters. -/
theorem rectangle_area_increase (L W : ℝ) (h : L * W = 450) :
  (1.2 * L) * (1.3 * W) = 702 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l1022_102266


namespace NUMINAMATH_CALUDE_inequality_proof_l1022_102201

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + y * z + z * x = 1) : 
  (27 / 4) * (x + y) * (y + z) * (z + x) ≥ (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x))^2 ∧ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x))^2 ≥ 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1022_102201


namespace NUMINAMATH_CALUDE_quadratic_roots_l1022_102274

/-- A quadratic function f(x) = ax² - 12ax + 36a - 5 has roots at x = 4 and x = 8 -/
theorem quadratic_roots (a : ℝ) : 
  (∀ x ∈ Set.Ioo 4 5, a * x^2 - 12 * a * x + 36 * a - 5 < 0) →
  (∀ x ∈ Set.Ioo 8 9, a * x^2 - 12 * a * x + 36 * a - 5 > 0) →
  a = 5/4 := by
sorry


end NUMINAMATH_CALUDE_quadratic_roots_l1022_102274


namespace NUMINAMATH_CALUDE_probability_of_snow_in_three_days_l1022_102298

theorem probability_of_snow_in_three_days 
  (p1 : ℚ) (p2 : ℚ) (p3 : ℚ)
  (h1 : p1 = 1/2) 
  (h2 : p2 = 2/3) 
  (h3 : p3 = 3/4) : 
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 23/24 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_snow_in_three_days_l1022_102298


namespace NUMINAMATH_CALUDE_prob_xi_equals_two_l1022_102222

/-- A random variable following a binomial distribution with n = 3 and p = 1/3 -/
def ξ : ℕ → ℝ := sorry

/-- The probability mass function for the binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1-p)^(n-k)

/-- Theorem: The probability that ξ equals 2 is 2/9 -/
theorem prob_xi_equals_two :
  binomial_pmf 3 (1/3) 2 = 2/9 := by sorry

end NUMINAMATH_CALUDE_prob_xi_equals_two_l1022_102222


namespace NUMINAMATH_CALUDE_aladdin_theorem_l1022_102299

/-- A continuous function that takes all values in [0, 1) -/
def AllValuesContinuousFunction (φ : ℝ → ℝ) : Prop :=
  Continuous φ ∧ ∀ y ∈ Set.Iio 1, ∃ t, φ t = y

/-- The difference between max and min of an AllValuesContinuousFunction is at least 1 -/
theorem aladdin_theorem (φ : ℝ → ℝ) (h : AllValuesContinuousFunction φ) :
    ⨆ t, φ t - ⨅ t, φ t ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_aladdin_theorem_l1022_102299


namespace NUMINAMATH_CALUDE_sawz_logging_total_cost_l1022_102253

/-- The total cost of trees for Sawz Logging Co. -/
theorem sawz_logging_total_cost :
  let total_trees : ℕ := 850
  let douglas_fir_trees : ℕ := 350
  let ponderosa_pine_trees : ℕ := total_trees - douglas_fir_trees
  let douglas_fir_cost : ℕ := 300
  let ponderosa_pine_cost : ℕ := 225
  let total_cost : ℕ := douglas_fir_trees * douglas_fir_cost + ponderosa_pine_trees * ponderosa_pine_cost
  total_cost = 217500 := by
  sorry

#check sawz_logging_total_cost

end NUMINAMATH_CALUDE_sawz_logging_total_cost_l1022_102253


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1022_102218

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → e = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1022_102218


namespace NUMINAMATH_CALUDE_x_in_terms_of_z_l1022_102290

/-- Given that x is 30% less than y and y = z + 50, prove that x = 0.70z + 35 -/
theorem x_in_terms_of_z (z : ℝ) :
  let y := z + 50
  let x := y - 0.30 * y
  x = 0.70 * z + 35 := by
  sorry

end NUMINAMATH_CALUDE_x_in_terms_of_z_l1022_102290


namespace NUMINAMATH_CALUDE_sum_equals_60_l1022_102295

/-- An arithmetic sequence with specific terms. -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  a3_eq_4 : a 3 = 4
  a101_eq_36 : a 101 = 36

/-- The sum of specific terms in the arithmetic sequence equals 60. -/
theorem sum_equals_60 (seq : ArithmeticSequence) : seq.a 9 + seq.a 52 + seq.a 95 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_60_l1022_102295


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1022_102272

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The main theorem -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 * a 11 = 3 →
  a 3 + a 13 = 4 →
  (a 15 / a 5 = 1/3 ∨ a 15 / a 5 = 3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1022_102272


namespace NUMINAMATH_CALUDE_count_right_triangles_with_leg_15_l1022_102270

/-- The number of right triangles with integer side lengths and one leg equal to 15 -/
def rightTrianglesWithLeg15 : ℕ :=
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    let (a, b, c) := t
    a = 15 ∧ a^2 + b^2 = c^2 ∧ a < b ∧ b < c) (Finset.product (Finset.range 1000) (Finset.product (Finset.range 1000) (Finset.range 1000)))).card

/-- Theorem stating that there are exactly 4 right triangles with integer side lengths and one leg equal to 15 -/
theorem count_right_triangles_with_leg_15 : rightTrianglesWithLeg15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_right_triangles_with_leg_15_l1022_102270


namespace NUMINAMATH_CALUDE_jeff_took_six_cans_l1022_102220

/-- Represents the number of soda cans in various stages --/
structure SodaCans where
  initial : ℕ
  taken : ℕ
  final : ℕ

/-- Calculates the number of cans Jeff took from Tim --/
def cans_taken (s : SodaCans) : Prop :=
  s.initial - s.taken + (s.initial - s.taken) / 2 = s.final

/-- The main theorem to prove --/
theorem jeff_took_six_cans : ∃ (s : SodaCans), s.initial = 22 ∧ s.final = 24 ∧ s.taken = 6 ∧ cans_taken s := by
  sorry


end NUMINAMATH_CALUDE_jeff_took_six_cans_l1022_102220


namespace NUMINAMATH_CALUDE_factor_theorem_application_l1022_102236

-- Define the polynomial P(x)
def P (c : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + c*x + 10

-- Theorem statement
theorem factor_theorem_application (c : ℝ) :
  (∀ x, P c x = 0 ↔ x = 5) → c = -37 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_application_l1022_102236


namespace NUMINAMATH_CALUDE_dragon_invincible_l1022_102273

-- Define the possible head-cutting operations
inductive CutOperation
| cut13 : CutOperation
| cut17 : CutOperation
| cut6 : CutOperation

-- Define the state of the dragon
structure DragonState :=
  (heads : ℕ)

-- Define the rules for head regeneration
def regenerate (s : DragonState) : DragonState :=
  match s.heads with
  | 1 => ⟨8⟩  -- 1 + 7 regenerated
  | 2 => ⟨13⟩ -- 2 + 11 regenerated
  | 3 => ⟨12⟩ -- 3 + 9 regenerated
  | n => s

-- Define a single step of the process (cutting and potential regeneration)
def step (s : DragonState) (op : CutOperation) : DragonState :=
  let s' := match op with
    | CutOperation.cut13 => ⟨s.heads - min s.heads 13⟩
    | CutOperation.cut17 => ⟨s.heads - min s.heads 17⟩
    | CutOperation.cut6 => ⟨s.heads - min s.heads 6⟩
  regenerate s'

-- Define the theorem
theorem dragon_invincible :
  ∀ (ops : List CutOperation),
    let final_state := ops.foldl step ⟨100⟩
    final_state.heads > 0 ∨ final_state.heads ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_dragon_invincible_l1022_102273


namespace NUMINAMATH_CALUDE_tetrahedron_projection_areas_l1022_102207

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the area of the orthogonal projection of a triangle on the xOy plane -/
def projectionAreaXOY (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the area of the orthogonal projection of a triangle on the yOz plane -/
def projectionAreaYOZ (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the area of the orthogonal projection of a triangle on the zOx plane -/
def projectionAreaZOX (p1 p2 p3 : Point3D) : ℝ := sorry

theorem tetrahedron_projection_areas :
  let A : Point3D := ⟨2, 0, 0⟩
  let B : Point3D := ⟨2, 2, 0⟩
  let C : Point3D := ⟨0, 2, 0⟩
  let D : Point3D := ⟨1, 1, Real.sqrt 2⟩
  let S₁ := projectionAreaXOY A B C + projectionAreaXOY A B D + projectionAreaXOY A C D + projectionAreaXOY B C D
  let S₂ := projectionAreaYOZ A B C + projectionAreaYOZ A B D + projectionAreaYOZ A C D + projectionAreaYOZ B C D
  let S₃ := projectionAreaZOX A B C + projectionAreaZOX A B D + projectionAreaZOX A C D + projectionAreaZOX B C D
  S₃ = S₂ ∧ S₃ ≠ S₁ := by sorry

end NUMINAMATH_CALUDE_tetrahedron_projection_areas_l1022_102207


namespace NUMINAMATH_CALUDE_max_mn_value_l1022_102286

theorem max_mn_value (m n : ℝ) : 
  m > 0 → n > 0 → m * 2 - 1 + n = 0 → m * n ≤ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_max_mn_value_l1022_102286


namespace NUMINAMATH_CALUDE_tank_fill_time_l1022_102278

/-- Given three pipes with specified fill/empty rates, prove that the tank will be filled in 3 hours when all pipes are opened together. -/
theorem tank_fill_time (rate_A rate_B rate_C : ℚ)
  (h_A : rate_A = 1 / 6)
  (h_B : rate_B = 1 / 4)
  (h_C : rate_C = -1 / 12)
  : 1 / (rate_A + rate_B + rate_C) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l1022_102278


namespace NUMINAMATH_CALUDE_rectangle_sides_l1022_102217

theorem rectangle_sides (x y : ℚ) (h1 : 4 * x = 3 * y) (h2 : x * y = 2 * (x + y)) :
  x = 7 / 2 ∧ y = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_sides_l1022_102217


namespace NUMINAMATH_CALUDE_limit_implies_a_and_b_l1022_102238

/-- Given that the limit of (ln(2-x))^2 / (x^2 + ax + b) as x approaches 1 is equal to 1,
    prove that a = -2 and b = 1. -/
theorem limit_implies_a_and_b (a b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → 
    |((Real.log (2 - x))^2) / (x^2 + a*x + b) - 1| < ε) →
  a = -2 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_limit_implies_a_and_b_l1022_102238


namespace NUMINAMATH_CALUDE_fourth_game_shots_l1022_102279

/-- Given a basketball player's performance over four games, calculate the number of successful shots in the fourth game. -/
theorem fourth_game_shots (initial_shots initial_made fourth_game_shots : ℕ) 
  (h1 : initial_shots = 30)
  (h2 : initial_made = 12)
  (h3 : fourth_game_shots = 10)
  (h4 : (initial_made : ℚ) / initial_shots = 2/5)
  (h5 : ((initial_made + x) : ℚ) / (initial_shots + fourth_game_shots) = 1/2) :
  x = 8 :=
sorry

end NUMINAMATH_CALUDE_fourth_game_shots_l1022_102279


namespace NUMINAMATH_CALUDE_complement_A_in_U_l1022_102244

def U : Set ℝ := {x | x < 3}
def A : Set ℝ := {x | x < 1}

theorem complement_A_in_U : 
  U \ A = {x | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l1022_102244


namespace NUMINAMATH_CALUDE_alberts_number_l1022_102292

theorem alberts_number (n : ℕ) : 
  (1 : ℚ) / n + (1 : ℚ) / 2 = (1 : ℚ) / 3 + (2 : ℚ) / (n + 1) ↔ n = 2 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_alberts_number_l1022_102292


namespace NUMINAMATH_CALUDE_smallest_square_from_smaller_squares_l1022_102263

theorem smallest_square_from_smaller_squares :
  ∀ n : ℕ,
  (∃ a : ℕ, a * a = n * (1 * 1 + 2 * 2 + 3 * 3)) →
  (∀ m : ℕ, m < n → ¬∃ b : ℕ, b * b = m * (1 * 1 + 2 * 2 + 3 * 3)) →
  n = 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_from_smaller_squares_l1022_102263


namespace NUMINAMATH_CALUDE_typing_difference_is_856800_l1022_102234

/-- The number of minutes in a week -/
def minutes_per_week : ℕ := 60 * 24 * 7

/-- Micah's typing speed in words per minute -/
def micah_speed : ℕ := 35

/-- Isaiah's typing speed in words per minute -/
def isaiah_speed : ℕ := 120

/-- The difference in words typed between Isaiah and Micah in a week -/
def typing_difference : ℕ := isaiah_speed * minutes_per_week - micah_speed * minutes_per_week

theorem typing_difference_is_856800 : typing_difference = 856800 := by
  sorry

end NUMINAMATH_CALUDE_typing_difference_is_856800_l1022_102234


namespace NUMINAMATH_CALUDE_negation_of_all_divisible_by_five_are_odd_l1022_102276

theorem negation_of_all_divisible_by_five_are_odd :
  ¬(∀ n : ℤ, 5 ∣ n → Odd n) ↔ ∃ n : ℤ, 5 ∣ n ∧ ¬(Odd n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_divisible_by_five_are_odd_l1022_102276


namespace NUMINAMATH_CALUDE_roses_kept_l1022_102242

theorem roses_kept (initial : ℕ) (mother grandmother sister : ℕ) 
  (h1 : initial = 20)
  (h2 : mother = 6)
  (h3 : grandmother = 9)
  (h4 : sister = 4) :
  initial - (mother + grandmother + sister) = 1 := by
  sorry

end NUMINAMATH_CALUDE_roses_kept_l1022_102242


namespace NUMINAMATH_CALUDE_max_value_theorem_l1022_102291

theorem max_value_theorem (x y : ℝ) :
  (2 * x + 3 * y + 2) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 17 ∧
  ∃ (x₀ y₀ : ℝ), (2 * x₀ + 3 * y₀ + 2) / Real.sqrt (x₀^2 + y₀^2 + 1) = Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1022_102291


namespace NUMINAMATH_CALUDE_root_difference_implies_h_value_l1022_102281

theorem root_difference_implies_h_value (h : ℝ) : 
  (∃ p q : ℝ, p^2 + h*p + 8 = 0 ∧ q^2 + h*q + 8 = 0 ∧
   (p+6)^2 - h*(p+6) + 8 = 0 ∧ (q+6)^2 - h*(q+6) + 8 = 0) →
  h = 6 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_implies_h_value_l1022_102281


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l1022_102285

/-- Given a rectangle with length x^2 and width x + 5, prove that if its area
    equals three times its perimeter, then x = 3. -/
theorem rectangle_area_perimeter_relation (x : ℝ) : 
  (x^2 * (x + 5) = 3 * (2 * x^2 + 2 * (x + 5))) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l1022_102285


namespace NUMINAMATH_CALUDE_cos_half_plus_sin_max_value_l1022_102265

theorem cos_half_plus_sin_max_value (θ : Real) (h : 0 < θ ∧ θ < Real.pi) :
  (∀ φ, 0 < φ ∧ φ < Real.pi → Real.cos (φ/2) * (1 + Real.sin φ) ≤ Real.cos (θ/2) * (1 + Real.sin θ)) →
  Real.cos (θ/2) * (1 + Real.sin θ) = 4 * Real.sqrt 3 / 9 :=
by sorry

end NUMINAMATH_CALUDE_cos_half_plus_sin_max_value_l1022_102265


namespace NUMINAMATH_CALUDE_unique_triple_lcm_gcd_l1022_102246

theorem unique_triple_lcm_gcd : 
  ∃! (x y z : ℕ+), 
    Nat.lcm x y = 100 ∧ 
    Nat.lcm x z = 450 ∧ 
    Nat.lcm y z = 1100 ∧ 
    Nat.gcd (Nat.gcd x y) z = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_lcm_gcd_l1022_102246


namespace NUMINAMATH_CALUDE_current_speed_is_4_l1022_102282

/-- Represents the speed of a boat in a river with a current -/
structure RiverBoat where
  boatSpeed : ℝ  -- Speed of the boat in still water
  currentSpeed : ℝ  -- Speed of the current

/-- Calculates the effective downstream speed -/
def downstreamSpeed (rb : RiverBoat) : ℝ :=
  rb.boatSpeed + rb.currentSpeed

/-- Calculates the effective upstream speed -/
def upstreamSpeed (rb : RiverBoat) : ℝ :=
  rb.boatSpeed - rb.currentSpeed

/-- Theorem stating the speed of the current given the problem conditions -/
theorem current_speed_is_4 (rb : RiverBoat) 
  (h1 : downstreamSpeed rb * 8 = 96)
  (h2 : upstreamSpeed rb * 2 = 8) :
  rb.currentSpeed = 4 := by
  sorry


end NUMINAMATH_CALUDE_current_speed_is_4_l1022_102282


namespace NUMINAMATH_CALUDE_cube_root_and_square_root_l1022_102259

theorem cube_root_and_square_root (a b : ℝ) : 
  (b - 4)^(1/3) = -2 → 
  b = -4 ∧ 
  Real.sqrt (5 * a - b) = 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_and_square_root_l1022_102259


namespace NUMINAMATH_CALUDE_sock_purchase_theorem_l1022_102215

/-- Represents the number of pairs of socks at each price point -/
structure SockPurchase where
  two_dollar : ℕ
  four_dollar : ℕ
  five_dollar : ℕ

/-- Checks if the SockPurchase satisfies the given conditions -/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.two_dollar + p.four_dollar + p.five_dollar = 15 ∧
  2 * p.two_dollar + 4 * p.four_dollar + 5 * p.five_dollar = 38 ∧
  p.two_dollar ≥ 1 ∧ p.four_dollar ≥ 1 ∧ p.five_dollar ≥ 1

theorem sock_purchase_theorem :
  ∃ (p : SockPurchase), is_valid_purchase p ∧ p.two_dollar = 12 :=
by sorry

end NUMINAMATH_CALUDE_sock_purchase_theorem_l1022_102215


namespace NUMINAMATH_CALUDE_fib_sum_squares_l1022_102211

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: Sum of squares of consecutive Fibonacci numbers -/
theorem fib_sum_squares (n : ℕ) : (fib n)^2 + (fib (n + 1))^2 = fib (2 * n + 2) := by
  sorry

end NUMINAMATH_CALUDE_fib_sum_squares_l1022_102211


namespace NUMINAMATH_CALUDE_logan_snowfall_total_l1022_102205

/-- Represents the snowfall recorded over three days during a snowstorm -/
structure SnowfallRecord where
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- Calculates the total snowfall from a three-day record -/
def totalSnowfall (record : SnowfallRecord) : ℝ :=
  record.wednesday + record.thursday + record.friday

/-- Theorem stating that Logan's recorded snowfall totals 0.88 cm -/
theorem logan_snowfall_total :
  let record : SnowfallRecord := {
    wednesday := 0.33,
    thursday := 0.33,
    friday := 0.22
  }
  totalSnowfall record = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_logan_snowfall_total_l1022_102205


namespace NUMINAMATH_CALUDE_project_completion_time_l1022_102208

theorem project_completion_time
  (days_A : ℝ)
  (days_B : ℝ)
  (break_days : ℝ)
  (h1 : days_A = 18)
  (h2 : days_B = 15)
  (h3 : break_days = 4) :
  let efficiency_A := 1 / days_A
  let efficiency_B := 1 / days_B
  let combined_efficiency := efficiency_A + efficiency_B
  let work_during_break := efficiency_B * break_days
  (1 - work_during_break) / combined_efficiency + break_days = 10 :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l1022_102208


namespace NUMINAMATH_CALUDE_pyramid_with_14_edges_has_8_vertices_l1022_102241

/-- A pyramid is a polyhedron with a polygonal base and triangular faces meeting at a common point (apex) --/
structure Pyramid where
  num_edges : ℕ

/-- The number of vertices in a pyramid --/
def num_vertices (p : Pyramid) : ℕ :=
  (p.num_edges / 2) + 2

theorem pyramid_with_14_edges_has_8_vertices (p : Pyramid) (h : p.num_edges = 14) : 
  num_vertices p = 8 := by
  sorry

#check pyramid_with_14_edges_has_8_vertices

end NUMINAMATH_CALUDE_pyramid_with_14_edges_has_8_vertices_l1022_102241


namespace NUMINAMATH_CALUDE_two_power_and_factorial_l1022_102202

theorem two_power_and_factorial (n : ℕ) :
  (¬ (2^n ∣ n!)) ∧ (∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, 2^(n-1) ∣ n!) := by
  sorry

end NUMINAMATH_CALUDE_two_power_and_factorial_l1022_102202


namespace NUMINAMATH_CALUDE_not_divisible_by_2019_l1022_102214

theorem not_divisible_by_2019 (n : ℕ) : ¬(2019 ∣ (n^2 + n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_2019_l1022_102214


namespace NUMINAMATH_CALUDE_area_code_letters_l1022_102288

/-- The number of letters in each area code. -/
def n : ℕ := 2

/-- The total number of signs available. -/
def total_signs : ℕ := 224

/-- The number of signs fully used. -/
def used_signs : ℕ := 222

/-- The number of unused signs. -/
def unused_signs : ℕ := 2

/-- The number of additional area codes created by using all signs. -/
def additional_codes : ℕ := 888

theorem area_code_letters :
  n = 2 ∧
  total_signs = 224 ∧
  used_signs = 222 ∧
  unused_signs = 2 ∧
  additional_codes = 888 ∧
  total_signs ^ n - used_signs ^ n - n * unused_signs = additional_codes :=
sorry

end NUMINAMATH_CALUDE_area_code_letters_l1022_102288


namespace NUMINAMATH_CALUDE_phi_value_l1022_102243

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem phi_value (φ : ℝ) :
  -π ≤ φ ∧ φ < π →
  (∀ x, f (x - π/2) φ = Real.sin x * Real.cos x + (Real.sqrt 3 / 2) * Real.cos x) →
  |φ| = 5*π/6 := by
  sorry

end NUMINAMATH_CALUDE_phi_value_l1022_102243


namespace NUMINAMATH_CALUDE_red_balls_count_l1022_102226

/-- Given a bag with red and blue balls, if the total number of balls is 12
    and the probability of drawing two red balls at the same time is 1/18,
    then the number of red balls is 3. -/
theorem red_balls_count (total : ℕ) (red : ℕ) (prob : ℚ) :
  total = 12 →
  prob = 1 / 18 →
  prob = (red / total) * ((red - 1) / (total - 1)) →
  red = 3 :=
sorry

end NUMINAMATH_CALUDE_red_balls_count_l1022_102226


namespace NUMINAMATH_CALUDE_product_even_if_sum_odd_l1022_102245

theorem product_even_if_sum_odd (a b : ℤ) : 
  Odd (a + b) → Even (a * b) := by
  sorry

end NUMINAMATH_CALUDE_product_even_if_sum_odd_l1022_102245


namespace NUMINAMATH_CALUDE_bankers_gain_l1022_102264

/-- Calculate the banker's gain given present worth, interest rate, and time period -/
theorem bankers_gain (present_worth : ℝ) (interest_rate : ℝ) (time_period : ℕ) : 
  present_worth = 600 → 
  interest_rate = 0.1 → 
  time_period = 2 → 
  present_worth * (1 + interest_rate) ^ time_period - present_worth = 126 := by
sorry

end NUMINAMATH_CALUDE_bankers_gain_l1022_102264


namespace NUMINAMATH_CALUDE_first_number_value_l1022_102235

theorem first_number_value (x y : ℝ) 
  (sum_condition : x + y = 33)
  (double_condition : y = 2 * x)
  (second_number_value : y = 22) :
  x = 11 := by
  sorry

end NUMINAMATH_CALUDE_first_number_value_l1022_102235


namespace NUMINAMATH_CALUDE_combined_area_of_tracts_l1022_102228

/-- The combined area of two rectangular tracts of land -/
theorem combined_area_of_tracts (length1 width1 length2 width2 : ℕ) 
  (h1 : length1 = 300)
  (h2 : width1 = 500)
  (h3 : length2 = 250)
  (h4 : width2 = 630) :
  length1 * width1 + length2 * width2 = 307500 := by
  sorry

end NUMINAMATH_CALUDE_combined_area_of_tracts_l1022_102228


namespace NUMINAMATH_CALUDE_equal_areas_of_inscribed_polygons_with_same_side_lengths_l1022_102283

-- Define a type for polygons
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

-- Define a function to calculate the side lengths of a polygon
def sideLengths (n : ℕ) (p : Polygon n) : Multiset ℝ :=
  sorry

-- Define a function to calculate the area of a polygon
def area (n : ℕ) (p : Polygon n) : ℝ :=
  sorry

-- Define a predicate to check if a polygon is inscribed in a circle
def isInscribed (n : ℕ) (p : Polygon n) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  sorry

-- Theorem statement
theorem equal_areas_of_inscribed_polygons_with_same_side_lengths
  (n : ℕ) (p1 p2 : Polygon n) (center : ℝ × ℝ) (radius : ℝ) :
  isInscribed n p1 center radius →
  isInscribed n p2 center radius →
  sideLengths n p1 = sideLengths n p2 →
  area n p1 = area n p2 :=
sorry

end NUMINAMATH_CALUDE_equal_areas_of_inscribed_polygons_with_same_side_lengths_l1022_102283


namespace NUMINAMATH_CALUDE_cube_construction_count_l1022_102216

/-- Represents the group of rotations for a 3x3x3 cube -/
def CubeRotations : Type := Unit

/-- The number of elements in the group of rotations for a 3x3x3 cube -/
def rotationGroupSize : ℕ := 27

/-- The total number of ways to arrange 13 white cubes in a 3x3x3 cube -/
def totalArrangements : ℕ := 10400600

/-- The estimated number of fixed points for non-identity rotations -/
def fixedPointsNonIdentity : ℕ := 1000

/-- The total number of fixed points across all rotations -/
def totalFixedPoints : ℕ := totalArrangements + fixedPointsNonIdentity

/-- The number of distinct ways to construct the 3x3x3 cube -/
def distinctConstructions : ℕ := totalFixedPoints / rotationGroupSize

theorem cube_construction_count :
  distinctConstructions = 385244 := by sorry

end NUMINAMATH_CALUDE_cube_construction_count_l1022_102216


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1022_102237

theorem inequality_solution_set (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  {x : ℝ | -b < 1/x ∧ 1/x < a} = {x : ℝ | x < -1/b ∨ x > 1/a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1022_102237


namespace NUMINAMATH_CALUDE_triangles_in_hexagon_count_l1022_102230

/-- The number of vertices in a hexagon -/
def hexagon_vertices : ℕ := 6

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of different triangles that can be formed using the vertices of a hexagon -/
def triangles_in_hexagon : ℕ := Nat.choose hexagon_vertices triangle_vertices

theorem triangles_in_hexagon_count :
  triangles_in_hexagon = 20 := by sorry

end NUMINAMATH_CALUDE_triangles_in_hexagon_count_l1022_102230


namespace NUMINAMATH_CALUDE_dad_steps_l1022_102232

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- Defines the relationship between Dad's and Masha's steps -/
def dad_masha_ratio (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha

/-- Defines the relationship between Masha's and Yasha's steps -/
def masha_yasha_ratio (s : Steps) : Prop :=
  5 * s.masha = 3 * s.yasha

/-- States that Masha and Yasha together took 400 steps -/
def total_masha_yasha (s : Steps) : Prop :=
  s.masha + s.yasha = 400

theorem dad_steps (s : Steps) 
  (h1 : dad_masha_ratio s)
  (h2 : masha_yasha_ratio s)
  (h3 : total_masha_yasha s) :
  s.dad = 90 := by
  sorry

end NUMINAMATH_CALUDE_dad_steps_l1022_102232


namespace NUMINAMATH_CALUDE_coordinate_system_change_l1022_102260

/-- Represents a point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a coordinate system in a 2D plane -/
structure CoordinateSystem where
  origin : Point2D

/-- Returns the coordinates of a point in a given coordinate system -/
def getCoordinates (p : Point2D) (cs : CoordinateSystem) : Point2D :=
  { x := p.x - cs.origin.x, y := p.y - cs.origin.y }

theorem coordinate_system_change 
  (A B : Point2D) 
  (csA csB : CoordinateSystem) 
  (h1 : csA.origin = A) 
  (h2 : csB.origin = B) 
  (h3 : getCoordinates B csA = { x := a, y := b }) :
  getCoordinates A csB = { x := -a, y := -b } := by
  sorry


end NUMINAMATH_CALUDE_coordinate_system_change_l1022_102260


namespace NUMINAMATH_CALUDE_quotient_zeros_l1022_102258

theorem quotient_zeros (x : ℚ) (h : x = 4227/1000) : 
  ¬ (∃ a b c : ℕ, x / 3 = (a : ℚ) + 1/10 + c/1000) :=
sorry

end NUMINAMATH_CALUDE_quotient_zeros_l1022_102258


namespace NUMINAMATH_CALUDE_problem_solution_l1022_102240

def f (m : ℝ) (x : ℝ) : ℝ := |x - m| - |x + 3*m|

theorem problem_solution (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, f 1 x ≥ 1 ↔ x ≤ -3/2) ∧
  ((∀ x t : ℝ, f m x < |2 + t| + |t - 1|) → 0 < m ∧ m < 3/4) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1022_102240


namespace NUMINAMATH_CALUDE_table_tennis_tournament_l1022_102221

theorem table_tennis_tournament (n : ℕ) : 
  (∃ r : ℕ, r ≤ 3 ∧ (n^2 - 7*n - 76 + 2*r = 0) ∧ 
   (n - 3).choose 2 + 6 + r = 50) → 
  (∃! r : ℕ, r = 1 ∧ r ≤ 3 ∧ (n^2 - 7*n - 76 + 2*r = 0) ∧ 
   (n - 3).choose 2 + 6 + r = 50) :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_tournament_l1022_102221


namespace NUMINAMATH_CALUDE_lead_is_29_points_l1022_102293

/-- The lead in points between two teams -/
def lead (our_score green_score : ℕ) : ℕ :=
  our_score - green_score

/-- Theorem: Given the final scores, prove the lead is 29 points -/
theorem lead_is_29_points : lead 68 39 = 29 := by
  sorry

end NUMINAMATH_CALUDE_lead_is_29_points_l1022_102293


namespace NUMINAMATH_CALUDE_red_balls_count_l1022_102249

theorem red_balls_count (total : ℕ) (red : ℕ) (h1 : total = 15) 
  (h2 : red ≤ total) 
  (h3 : (red * (red - 1)) / (total * (total - 1)) = 1 / 21) : 
  red = 4 := by
sorry

end NUMINAMATH_CALUDE_red_balls_count_l1022_102249


namespace NUMINAMATH_CALUDE_wang_liang_age_l1022_102239

def is_valid_age (age : ℕ) : Prop :=
  ∃ (birth_year : ℕ),
    (2012 - birth_year = age) ∧
    (age = (birth_year / 1000) + ((birth_year / 100) % 10) + ((birth_year / 10) % 10) + (birth_year % 10))

theorem wang_liang_age :
  (is_valid_age 7 ∨ is_valid_age 25) ∧
  ∀ (age : ℕ), is_valid_age age → (age = 7 ∨ age = 25) :=
sorry

end NUMINAMATH_CALUDE_wang_liang_age_l1022_102239


namespace NUMINAMATH_CALUDE_first_issue_pages_l1022_102227

/-- Represents the number of pages Trevor drew in a month -/
structure MonthlyPages where
  regular : ℕ  -- Regular pages
  bonus : ℕ    -- Bonus pages

/-- Represents Trevor's comic book production over three months -/
structure ComicProduction where
  month1 : MonthlyPages
  month2 : MonthlyPages
  month3 : MonthlyPages
  total_pages : ℕ
  pages_per_day_month1 : ℕ
  pages_per_day_month23 : ℕ

/-- The conditions of Trevor's comic book production -/
def comic_conditions (prod : ComicProduction) : Prop :=
  prod.total_pages = 220 ∧
  prod.pages_per_day_month1 = 5 ∧
  prod.pages_per_day_month23 = 4 ∧
  prod.month1.regular = prod.month2.regular ∧
  prod.month3.regular = prod.month1.regular + 4 ∧
  prod.month1.bonus = 3 ∧
  prod.month2.bonus = 3 ∧
  prod.month3.bonus = 3

theorem first_issue_pages (prod : ComicProduction) 
  (h : comic_conditions prod) : prod.month1.regular = 69 := by
  sorry

end NUMINAMATH_CALUDE_first_issue_pages_l1022_102227


namespace NUMINAMATH_CALUDE_area_of_sine_curve_l1022_102213

theorem area_of_sine_curve (f : ℝ → ℝ) (a b : ℝ) : 
  (f = λ x => Real.sin x) →
  (a = -π/2) →
  (b = 5*π/4) →
  (∫ x in a..b, |f x| ) = 4 - Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_area_of_sine_curve_l1022_102213


namespace NUMINAMATH_CALUDE_handshakes_count_l1022_102209

/-- Represents a social gathering with specific group interactions -/
structure SocialGathering where
  total_people : ℕ
  group_a : ℕ  -- People who all know each other
  group_b : ℕ  -- People who know no one
  group_c : ℕ  -- People who know exactly 15 from group_a
  h_total : total_people = group_a + group_b + group_c
  h_group_a : group_a = 25
  h_group_b : group_b = 10
  h_group_c : group_c = 5

/-- Calculates the number of handshakes in the social gathering -/
def handshakes (sg : SocialGathering) : ℕ :=
  let ab_handshakes := sg.group_b * (sg.group_a + sg.group_c)
  let b_internal_handshakes := sg.group_b * (sg.group_b - 1) / 2
  let c_handshakes := sg.group_c * (sg.group_a - 15 + sg.group_c)
  ab_handshakes + b_internal_handshakes + c_handshakes

/-- Theorem stating that the number of handshakes in the given social gathering is 420 -/
theorem handshakes_count (sg : SocialGathering) : handshakes sg = 420 := by
  sorry

#eval handshakes { total_people := 40, group_a := 25, group_b := 10, group_c := 5,
                   h_total := rfl, h_group_a := rfl, h_group_b := rfl, h_group_c := rfl }

end NUMINAMATH_CALUDE_handshakes_count_l1022_102209


namespace NUMINAMATH_CALUDE_f_geq_one_iff_a_nonneg_l1022_102254

/-- The quadratic function f(x) = x^2 + 2ax + 2a + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2*a + 1

/-- The theorem stating the range of a for which f(x) ≥ 1 for all x in [-1, 1] -/
theorem f_geq_one_iff_a_nonneg (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 1) ↔ a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_f_geq_one_iff_a_nonneg_l1022_102254


namespace NUMINAMATH_CALUDE_min_sum_squares_l1022_102267

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 3 ∧ (∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1022_102267


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l1022_102296

theorem convex_polygon_sides (n : ℕ) (a₁ : ℝ) (d : ℝ) : 
  n > 2 →
  a₁ = 120 →
  d = 5 →
  (n - 2) * 180 = (2 * a₁ + (n - 1) * d) * n / 2 →
  (∀ k : ℕ, k ≤ n → a₁ + (k - 1) * d < 180) →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l1022_102296


namespace NUMINAMATH_CALUDE_range_of_m_l1022_102225

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + m - 1 = 0}

theorem range_of_m : ∀ m : ℝ, (A ∪ B m = A) → m = 3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1022_102225


namespace NUMINAMATH_CALUDE_max_coins_ali_baba_l1022_102280

/-- Represents the coin distribution game --/
structure CoinGame where
  totalPiles : Nat
  initialCoinsPerPile : Nat
  totalCoins : Nat
  selectablePiles : Nat
  takablePiles : Nat

/-- Defines the specific game instance --/
def aliBabaGame : CoinGame :=
  { totalPiles := 10
  , initialCoinsPerPile := 10
  , totalCoins := 100
  , selectablePiles := 4
  , takablePiles := 3 
  }

/-- Theorem stating the maximum number of coins Ali Baba can take --/
theorem max_coins_ali_baba (game : CoinGame) (h1 : game = aliBabaGame) : 
  ∃ (maxCoins : Nat), maxCoins = 72 ∧ 
  (∀ (strategy : CoinGame → Nat), strategy game ≤ maxCoins) := by
  sorry

end NUMINAMATH_CALUDE_max_coins_ali_baba_l1022_102280


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1022_102224

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1022_102224


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l1022_102289

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two :
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l1022_102289


namespace NUMINAMATH_CALUDE_sum_of_exponents_l1022_102251

theorem sum_of_exponents (a b : ℕ) : 
  2^4 + 2^4 = 2^a → 3^5 + 3^5 + 3^5 = 3^b → a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_l1022_102251


namespace NUMINAMATH_CALUDE_tan_three_expression_equals_zero_l1022_102256

theorem tan_three_expression_equals_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_expression_equals_zero_l1022_102256


namespace NUMINAMATH_CALUDE_cost_solution_l1022_102271

/-- The cost of electronic whiteboards and projectors -/
def CostProblem (projector_cost : ℕ) (whiteboard_cost : ℕ) : Prop :=
  (whiteboard_cost = projector_cost + 4000) ∧
  (4 * whiteboard_cost + 3 * projector_cost = 44000)

/-- Theorem stating the correct costs for the whiteboard and projector -/
theorem cost_solution :
  ∃ (projector_cost whiteboard_cost : ℕ),
    CostProblem projector_cost whiteboard_cost ∧
    projector_cost = 4000 ∧
    whiteboard_cost = 8000 := by
  sorry

end NUMINAMATH_CALUDE_cost_solution_l1022_102271


namespace NUMINAMATH_CALUDE_fifth_runner_speed_doubling_l1022_102268

-- Define the total time and individual runner times
variable (T : ℝ) -- Total time
variable (T1 T2 T3 T4 T5 : ℝ) -- Individual runner times

-- Define the conditions from the problem
axiom total_time : T1 + T2 + T3 + T4 + T5 = T
axiom first_runner : T1 / 2 + T2 + T3 + T4 + T5 = 0.95 * T
axiom second_runner : T1 + T2 / 2 + T3 + T4 + T5 = 0.9 * T
axiom third_runner : T1 + T2 + T3 / 2 + T4 + T5 = 0.88 * T
axiom fourth_runner : T1 + T2 + T3 + T4 / 2 + T5 = 0.85 * T

-- The theorem to prove
theorem fifth_runner_speed_doubling (h1 : T > 0) :
  T1 + T2 + T3 + T4 + T5 / 2 = 0.92 * T := by sorry

end NUMINAMATH_CALUDE_fifth_runner_speed_doubling_l1022_102268


namespace NUMINAMATH_CALUDE_overall_loss_percentage_l1022_102206

/-- Calculate the overall loss percentage for three items given their cost and selling prices -/
theorem overall_loss_percentage 
  (cp_radio cp_tv cp_blender : ℝ) 
  (sp_radio sp_tv sp_blender : ℝ) : 
  let total_cp := cp_radio + cp_tv + cp_blender
  let total_sp := sp_radio + sp_tv + sp_blender
  ((total_cp - total_sp) / total_cp) * 100 = 
    ((4500 + 8000 + 1300) - (3200 + 7500 + 1000)) / (4500 + 8000 + 1300) * 100 := by
  sorry

#eval ((4500 + 8000 + 1300) - (3200 + 7500 + 1000)) / (4500 + 8000 + 1300) * 100

end NUMINAMATH_CALUDE_overall_loss_percentage_l1022_102206


namespace NUMINAMATH_CALUDE_inequality_three_intervals_l1022_102233

theorem inequality_three_intervals (a : ℝ) (h : a > 1) :
  ∃ (I₁ I₂ I₃ : Set ℝ), 
    (∀ x : ℝ, (x^2 + (a+1)*x + a) / (x^2 + 5*x + 4) ≥ 0 ↔ x ∈ I₁ ∪ I₂ ∪ I₃) ∧
    (I₁.Nonempty ∧ I₂.Nonempty ∧ I₃.Nonempty) ∧
    (I₁ ∩ I₂ = ∅ ∧ I₁ ∩ I₃ = ∅ ∧ I₂ ∩ I₃ = ∅) :=
by sorry

end NUMINAMATH_CALUDE_inequality_three_intervals_l1022_102233


namespace NUMINAMATH_CALUDE_sin_m_theta_bound_l1022_102261

theorem sin_m_theta_bound (θ : ℝ) (m : ℕ) : 
  |Real.sin (m * θ)| ≤ m * |Real.sin θ| := by
  sorry

end NUMINAMATH_CALUDE_sin_m_theta_bound_l1022_102261


namespace NUMINAMATH_CALUDE_factorization_equality_l1022_102294

theorem factorization_equality (a : ℝ) : 2 * a^2 - 2 * a + (1/2 : ℝ) = 2 * (a - 1/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1022_102294


namespace NUMINAMATH_CALUDE_largest_square_tile_size_l1022_102204

/-- The length of the courtyard in centimeters -/
def courtyard_length : ℕ := 378

/-- The width of the courtyard in centimeters -/
def courtyard_width : ℕ := 525

/-- The size of the largest square tile in centimeters -/
def largest_tile_size : ℕ := 21

theorem largest_square_tile_size :
  (largest_tile_size ∣ courtyard_length) ∧
  (largest_tile_size ∣ courtyard_width) ∧
  ∀ n : ℕ, n > largest_tile_size →
    ¬(n ∣ courtyard_length) ∨ ¬(n ∣ courtyard_width) :=
by sorry

end NUMINAMATH_CALUDE_largest_square_tile_size_l1022_102204


namespace NUMINAMATH_CALUDE_interior_lattice_points_collinear_l1022_102250

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle in the plane -/
structure Triangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Predicate to check if a point is inside a triangle -/
def isInside (p : LatticePoint) (t : Triangle) : Prop :=
  sorry

/-- Predicate to check if a point is on the boundary of a triangle -/
def isOnBoundary (p : LatticePoint) (t : Triangle) : Prop :=
  sorry

/-- Predicate to check if points are collinear -/
def areCollinear (points : List LatticePoint) : Prop :=
  sorry

/-- The main theorem -/
theorem interior_lattice_points_collinear (t : Triangle) :
  (∀ p : LatticePoint, isOnBoundary p t → (p = t.A ∨ p = t.B ∨ p = t.C)) →
  (∃ (p1 p2 p3 p4 : LatticePoint),
    isInside p1 t ∧ isInside p2 t ∧ isInside p3 t ∧ isInside p4 t ∧
    (∀ q : LatticePoint, isInside q t → (q = p1 ∨ q = p2 ∨ q = p3 ∨ q = p4))) →
  ∃ (p1 p2 p3 p4 : LatticePoint),
    isInside p1 t ∧ isInside p2 t ∧ isInside p3 t ∧ isInside p4 t ∧
    areCollinear [p1, p2, p3, p4] :=
by
  sorry


end NUMINAMATH_CALUDE_interior_lattice_points_collinear_l1022_102250


namespace NUMINAMATH_CALUDE_expand_product_l1022_102255

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1022_102255


namespace NUMINAMATH_CALUDE_circle_equation_AB_l1022_102275

/-- Given two points A and B, this function returns the equation of the circle
    with AB as its diameter in the form (x - h)² + (y - k)² = r², where
    (h, k) is the center of the circle and r is its radius. -/
def circle_equation_with_diameter (A B : ℝ × ℝ) : (ℝ → ℝ → Prop) :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let h := (x₁ + x₂) / 2
  let k := (y₁ + y₂) / 2
  let r := ((x₁ - x₂)^2 + (y₁ - y₂)^2).sqrt / 2
  fun x y => (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that for points A(3, -2) and B(-5, 4), the equation of the circle
    with AB as its diameter is (x + 1)² + (y - 1)² = 25. -/
theorem circle_equation_AB : 
  circle_equation_with_diameter (3, -2) (-5, 4) = fun x y => (x + 1)^2 + (y - 1)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_AB_l1022_102275


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1022_102284

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def N : Set ℝ := {x | -2 < x ∧ x ≤ 4}

-- Theorem statement
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1022_102284


namespace NUMINAMATH_CALUDE_expression_evaluation_l1022_102223

theorem expression_evaluation : 
  (Real.sqrt 3) / (Real.cos (10 * π / 180)) - 1 / (Real.sin (170 * π / 180)) = -4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1022_102223


namespace NUMINAMATH_CALUDE_geometric_progression_exists_l1022_102257

theorem geometric_progression_exists : ∃ (a b c d : ℚ) (e f g h : ℚ),
  (b = a * (-4) ∧ c = b * (-4) ∧ d = c * (-4) ∧
   b = a - 35 ∧ c = d + 560) ∧
  (f = e * 4 ∧ g = f * 4 ∧ h = g * 4 ∧
   f = e - 35 ∧ g = h + 560) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_exists_l1022_102257


namespace NUMINAMATH_CALUDE_james_off_road_vehicles_l1022_102248

/-- The number of off-road vehicles James bought -/
def num_off_road_vehicles : ℕ := 4

/-- The cost of each dirt bike -/
def dirt_bike_cost : ℕ := 150

/-- The cost of each off-road vehicle -/
def off_road_vehicle_cost : ℕ := 300

/-- The registration cost for each vehicle -/
def registration_cost : ℕ := 25

/-- The number of dirt bikes James bought -/
def num_dirt_bikes : ℕ := 3

/-- The total amount James paid -/
def total_amount : ℕ := 1825

theorem james_off_road_vehicles :
  num_off_road_vehicles * (off_road_vehicle_cost + registration_cost) +
  num_dirt_bikes * (dirt_bike_cost + registration_cost) =
  total_amount :=
by sorry

end NUMINAMATH_CALUDE_james_off_road_vehicles_l1022_102248


namespace NUMINAMATH_CALUDE_base_three_to_decimal_l1022_102262

/-- Converts a digit in base 3 to its decimal value -/
def toDecimal (d : Nat) : Nat :=
  if d < 3 then d else 0

/-- Calculates the value of a base-3 number given its digits -/
def baseThreeToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + toDecimal d * 3^i) 0

/-- The decimal representation of 10212 in base 3 -/
def baseThreeNumber : Nat :=
  baseThreeToDecimal [2, 1, 2, 0, 1]

theorem base_three_to_decimal :
  baseThreeNumber = 104 := by sorry

end NUMINAMATH_CALUDE_base_three_to_decimal_l1022_102262


namespace NUMINAMATH_CALUDE_total_cost_is_1340_l1022_102231

def number_of_vaccines : ℕ := 10
def vaccine_cost : ℚ := 45
def doctors_visit_cost : ℚ := 250
def insurance_coverage_rate : ℚ := 0.8
def trip_cost : ℚ := 1200

def total_cost : ℚ :=
  trip_cost + (1 - insurance_coverage_rate) * (number_of_vaccines * vaccine_cost + doctors_visit_cost)

theorem total_cost_is_1340 : total_cost = 1340 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_1340_l1022_102231


namespace NUMINAMATH_CALUDE_circle_symmetry_l1022_102212

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 5 = 0

-- Define circle D
def circle_D (x y : ℝ) : Prop := (x + 2)^2 + (y - 6)^2 = 1

-- Define circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 1

-- Define symmetry with respect to a line
def symmetric_wrt_line (c₁ c₂ : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (p : ℝ × ℝ), l p.1 p.2 ∧ 
    (c₁.1 + c₂.1) / 2 = p.1 ∧ 
    (c₁.2 + c₂.2) / 2 = p.2 ∧
    (c₂.1 - c₁.1) * (p.2 - c₁.2) = (c₂.2 - c₁.2) * (p.1 - c₁.1)

theorem circle_symmetry :
  symmetric_wrt_line (-2, 6) (1, 3) line_l →
  (∀ x y : ℝ, circle_D x y ↔ circle_C (x + 3) (y - 3)) :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1022_102212


namespace NUMINAMATH_CALUDE_sum_100_from_neg_49_l1022_102210

/-- Sum of consecutive integers -/
def sum_consecutive_integers (start : Int) (count : Nat) : Int :=
  count * (2 * start + count.pred) / 2

/-- Theorem: Sum of 100 consecutive integers from -49 is 50 -/
theorem sum_100_from_neg_49 : sum_consecutive_integers (-49) 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_100_from_neg_49_l1022_102210
