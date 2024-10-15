import Mathlib

namespace NUMINAMATH_CALUDE_trapezoid_area_sum_l2252_225216

/-- Given a trapezoid with side lengths 5, 6, 8, and 9, the sum of all possible areas is 28√3 + 42√2. -/
theorem trapezoid_area_sum :
  ∀ (s₁ s₂ s₃ s₄ : ℝ),
  s₁ = 5 ∧ s₂ = 6 ∧ s₃ = 8 ∧ s₄ = 9 →
  ∃ (A₁ A₂ : ℝ),
  (A₁ = (s₁ + s₄) * Real.sqrt 3 ∧
   A₂ = (s₂ + s₃) * Real.sqrt 2) →
  A₁ + A₂ = 28 * Real.sqrt 3 + 42 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_sum_l2252_225216


namespace NUMINAMATH_CALUDE_lauras_garden_tulips_l2252_225276

/-- Represents a garden with tulips and lilies -/
structure Garden where
  tulips : ℕ
  lilies : ℕ

/-- Calculates the number of tulips needed to maintain a 3:4 ratio with the given number of lilies -/
def tulipsForRatio (lilies : ℕ) : ℕ :=
  (3 * lilies) / 4

/-- Represents Laura's garden before and after adding flowers -/
def lauras_garden : Garden × Garden :=
  let initial := Garden.mk (tulipsForRatio 32) 32
  let final := Garden.mk (tulipsForRatio (32 + 24)) (32 + 24)
  (initial, final)

/-- Theorem stating that after adding 24 lilies and maintaining the 3:4 ratio, 
    Laura will have 42 tulips in total -/
theorem lauras_garden_tulips : 
  (lauras_garden.2).tulips = 42 := by sorry

end NUMINAMATH_CALUDE_lauras_garden_tulips_l2252_225276


namespace NUMINAMATH_CALUDE_fraction_equals_244_375_l2252_225252

/-- The fraction in the original problem -/
def original_fraction : ℚ :=
  (12^4+400)*(24^4+400)*(36^4+400)*(48^4+400)*(60^4+400) /
  ((6^4+400)*(18^4+400)*(30^4+400)*(42^4+400)*(54^4+400))

/-- The theorem stating that the original fraction equals 244.375 -/
theorem fraction_equals_244_375 : original_fraction = 244.375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_244_375_l2252_225252


namespace NUMINAMATH_CALUDE_smallest_divisible_by_14_15_16_l2252_225228

theorem smallest_divisible_by_14_15_16 : ∃ n : ℕ+, 
  (∀ m : ℕ+, 14 ∣ m ∧ 15 ∣ m ∧ 16 ∣ m → n ≤ m) ∧
  14 ∣ n ∧ 15 ∣ n ∧ 16 ∣ n :=
by
  use 1680
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_14_15_16_l2252_225228


namespace NUMINAMATH_CALUDE_missed_number_sum_l2252_225215

theorem missed_number_sum (n : ℕ) (missing : ℕ) : 
  n = 63 → 
  missing ≤ n →
  (n * (n + 1)) / 2 - missing = 1991 →
  missing = 25 := by
sorry

end NUMINAMATH_CALUDE_missed_number_sum_l2252_225215


namespace NUMINAMATH_CALUDE_area_of_S₃_l2252_225250

/-- Given a square S₁ with area 25, S₂ is constructed by bisecting the sides of S₁,
    and S₃ is constructed by bisecting the sides of S₂. -/
def square_construction (S₁ S₂ S₃ : Real → Real → Prop) : Prop :=
  (∀ x y, S₁ x y ↔ x^2 + y^2 = 25) ∧
  (∀ x y, S₂ x y ↔ ∃ a b, S₁ a b ∧ x = a/2 ∧ y = b/2) ∧
  (∀ x y, S₃ x y ↔ ∃ a b, S₂ a b ∧ x = a/2 ∧ y = b/2)

/-- The area of S₃ is 6.25 -/
theorem area_of_S₃ (S₁ S₂ S₃ : Real → Real → Prop) :
  square_construction S₁ S₂ S₃ →
  (∃ x y, S₃ x y ∧ x^2 + y^2 = 6.25) :=
sorry

end NUMINAMATH_CALUDE_area_of_S₃_l2252_225250


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l2252_225236

def geometric_series (n : ℕ) : ℚ :=
  match n with
  | 0 => 4/7
  | 1 => 36/49
  | 2 => 324/343
  | _ => 0  -- We only need the first three terms for this problem

theorem common_ratio_of_geometric_series :
  (geometric_series 1) / (geometric_series 0) = 9/7 :=
by sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l2252_225236


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l2252_225272

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : (x^3 * x^(1/2))^(1/4) = x^(7/8) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l2252_225272


namespace NUMINAMATH_CALUDE_characterization_of_M_inequality_for_M_elements_l2252_225234

-- Define the set M
def M : Set ℝ := {x | |2*x - 1| < 1}

-- Theorem 1: Characterization of set M
theorem characterization_of_M : M = {x | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: Inequality for elements in M
theorem inequality_for_M_elements (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 
  a * b + 1 > a + b := by sorry

end NUMINAMATH_CALUDE_characterization_of_M_inequality_for_M_elements_l2252_225234


namespace NUMINAMATH_CALUDE_largest_number_less_than_threshold_l2252_225260

def given_numbers : List ℚ := [4, 9/10, 6/5, 1/2, 13/10]
def threshold : ℚ := 111/100

theorem largest_number_less_than_threshold :
  (given_numbers.filter (· < threshold)).maximum? = some (9/10) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_less_than_threshold_l2252_225260


namespace NUMINAMATH_CALUDE_fraction_powers_sum_l2252_225249

theorem fraction_powers_sum : 
  (8/9 : ℚ)^3 * (3/4 : ℚ)^3 + (1/2 : ℚ)^3 = 91/216 := by
  sorry

end NUMINAMATH_CALUDE_fraction_powers_sum_l2252_225249


namespace NUMINAMATH_CALUDE_line_inclination_angle_l2252_225271

/-- The inclination angle of a line with equation √3x - y + 1 = 0 is 60° -/
theorem line_inclination_angle (x y : ℝ) :
  (Real.sqrt 3 * x - y + 1 = 0) →
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < Real.pi ∧ θ = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l2252_225271


namespace NUMINAMATH_CALUDE_goose_eggs_count_l2252_225202

theorem goose_eggs_count (total_eggs : ℕ) : 
  (total_eggs : ℚ) * (1/2) * (3/4) * (2/5) = 120 →
  total_eggs = 400 := by
  sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l2252_225202


namespace NUMINAMATH_CALUDE_square_root_of_polynomial_l2252_225270

theorem square_root_of_polynomial (a b c : ℝ) :
  (2*a - 3*b + 4*c)^2 = 16*a*c + 4*a^2 - 12*a*b + 9*b^2 - 24*b*c + 16*c^2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_polynomial_l2252_225270


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l2252_225274

theorem isosceles_triangle_vertex_angle (α : ℝ) :
  α > 0 ∧ α < 180 →  -- Angle is positive and less than 180°
  50 > 0 ∧ 50 < 180 →  -- 50° is a valid angle
  α + 50 + 50 = 180 →  -- Sum of angles in a triangle is 180°
  α = 80 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l2252_225274


namespace NUMINAMATH_CALUDE_explanatory_variable_is_fertilizer_amount_l2252_225201

/-- A study on crop yield prediction -/
structure CropStudy where
  fertilizer_amount : ℝ
  crop_yield : ℝ

/-- The explanatory variable in a regression analysis -/
inductive ExplanatoryVariable
  | CropYield
  | FertilizerAmount
  | Experimenter
  | OtherFactors

/-- The study aims to predict crop yield based on fertilizer amount -/
def study_aim (s : CropStudy) : Prop :=
  ∃ f : ℝ → ℝ, s.crop_yield = f s.fertilizer_amount

/-- The correct explanatory variable for the given study -/
def correct_explanatory_variable : ExplanatoryVariable :=
  ExplanatoryVariable.FertilizerAmount

/-- Theorem: The explanatory variable in the crop yield study is the fertilizer amount -/
theorem explanatory_variable_is_fertilizer_amount 
  (s : CropStudy) (aim : study_aim s) :
  correct_explanatory_variable = ExplanatoryVariable.FertilizerAmount :=
sorry

end NUMINAMATH_CALUDE_explanatory_variable_is_fertilizer_amount_l2252_225201


namespace NUMINAMATH_CALUDE_train_distance_l2252_225291

theorem train_distance (v1 v2 t : ℝ) (h1 : v1 = 11) (h2 : v2 = 31) (h3 : t = 8) :
  (v2 * t) - (v1 * t) = 160 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_l2252_225291


namespace NUMINAMATH_CALUDE_consecutive_integers_square_difference_l2252_225233

theorem consecutive_integers_square_difference (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 2720 → (n + 1)^2 - n^2 = 103 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_difference_l2252_225233


namespace NUMINAMATH_CALUDE_two_hundred_thousand_squared_l2252_225256

theorem two_hundred_thousand_squared : 200000 * 200000 = 40000000000 := by
  sorry

end NUMINAMATH_CALUDE_two_hundred_thousand_squared_l2252_225256


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l2252_225244

theorem triangle_side_lengths 
  (α : Real) 
  (r R : Real) 
  (hr : r > 0) 
  (hR : R > 0) 
  (ha : ∃ a, a = Real.sqrt (r * R)) :
  ∃ b c : Real,
    b^2 - (Real.sqrt (r * R) * (5 + 4 * Real.cos α)) * b + 4 * r * R * (3 + 2 * Real.cos α) = 0 ∧
    c^2 - (Real.sqrt (r * R) * (5 + 4 * Real.cos α)) * c + 4 * r * R * (3 + 2 * Real.cos α) = 0 ∧
    b ≠ c :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l2252_225244


namespace NUMINAMATH_CALUDE_jordans_sister_jars_l2252_225243

def total_plums : ℕ := 240
def ripe_ratio : ℚ := 1/4
def unripe_ratio : ℚ := 3/4
def kept_unripe : ℕ := 46
def plums_per_mango : ℕ := 7
def mangoes_per_jar : ℕ := 5

theorem jordans_sister_jars : 
  ⌊(total_plums * unripe_ratio - kept_unripe + total_plums * ripe_ratio) / plums_per_mango / mangoes_per_jar⌋ = 5 := by
  sorry

end NUMINAMATH_CALUDE_jordans_sister_jars_l2252_225243


namespace NUMINAMATH_CALUDE_one_volleyball_outside_range_l2252_225206

def volleyball_weights : List ℝ := [275, 263, 278, 270, 261, 277, 282, 269]
def standard_weight : ℝ := 270
def tolerance : ℝ := 10

theorem one_volleyball_outside_range : 
  (volleyball_weights.filter (λ w => w < standard_weight - tolerance ∨ 
                                     w > standard_weight + tolerance)).length = 1 :=
by sorry

end NUMINAMATH_CALUDE_one_volleyball_outside_range_l2252_225206


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2252_225237

def A : Set ℝ := {x | ∃ k : ℤ, x = 2 * k + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2252_225237


namespace NUMINAMATH_CALUDE_max_gcd_thirteen_numbers_sum_1988_l2252_225266

theorem max_gcd_thirteen_numbers_sum_1988 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ : ℕ) 
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ + a₁₂ + a₁₃ = 1988) :
  Nat.gcd a₁ (Nat.gcd a₂ (Nat.gcd a₃ (Nat.gcd a₄ (Nat.gcd a₅ (Nat.gcd a₆ (Nat.gcd a₇ (Nat.gcd a₈ (Nat.gcd a₉ (Nat.gcd a₁₀ (Nat.gcd a₁₁ (Nat.gcd a₁₂ a₁₃))))))))))) ≤ 142 :=
by
  sorry

end NUMINAMATH_CALUDE_max_gcd_thirteen_numbers_sum_1988_l2252_225266


namespace NUMINAMATH_CALUDE_boys_total_toys_l2252_225259

/-- The number of toys Bill has -/
def bill_toys : ℕ := 60

/-- The number of toys Hash has -/
def hash_toys : ℕ := bill_toys / 2 + 9

/-- The total number of toys both boys have -/
def total_toys : ℕ := bill_toys + hash_toys

theorem boys_total_toys : total_toys = 99 := by
  sorry

end NUMINAMATH_CALUDE_boys_total_toys_l2252_225259


namespace NUMINAMATH_CALUDE_blueberry_picking_relationships_l2252_225264

/-- Represents the relationship between y₁ and x for blueberry picking -/
def y₁ (x : ℝ) : ℝ := 60 + 18 * x

/-- Represents the relationship between y₂ and x for blueberry picking -/
def y₂ (x : ℝ) : ℝ := 150 + 15 * x

/-- Theorem stating the relationships between y₁, y₂, and x when blueberry picking amount exceeds 10 kg -/
theorem blueberry_picking_relationships (x : ℝ) (h : x > 10) :
  y₁ x = 60 + 18 * x ∧ y₂ x = 150 + 15 * x := by
  sorry

end NUMINAMATH_CALUDE_blueberry_picking_relationships_l2252_225264


namespace NUMINAMATH_CALUDE_power_two_mod_seven_l2252_225263

theorem power_two_mod_seven : 2^2010 ≡ 1 [MOD 7] := by sorry

end NUMINAMATH_CALUDE_power_two_mod_seven_l2252_225263


namespace NUMINAMATH_CALUDE_tangent_product_simplification_l2252_225230

theorem tangent_product_simplification :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_simplification_l2252_225230


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l2252_225218

theorem fraction_sum_simplification :
  1 / 462 + 17 / 42 = 94 / 231 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l2252_225218


namespace NUMINAMATH_CALUDE_derivative_at_one_l2252_225242

/-- Given a function f(x) = 2k*ln(x) - x, where k is a constant, prove that f'(1) = 1 -/
theorem derivative_at_one (k : ℝ) : 
  let f := fun (x : ℝ) => 2 * k * Real.log x - x
  deriv f 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2252_225242


namespace NUMINAMATH_CALUDE_brian_pencils_given_to_friend_l2252_225273

/-- 
Given that Brian initially had 39 pencils, bought 22 more, and ended up with 43 pencils,
this theorem proves that Brian gave 18 pencils to his friend.
-/
theorem brian_pencils_given_to_friend : 
  ∀ (initial_pencils bought_pencils final_pencils pencils_given : ℕ),
    initial_pencils = 39 →
    bought_pencils = 22 →
    final_pencils = 43 →
    final_pencils = initial_pencils - pencils_given + bought_pencils →
    pencils_given = 18 := by
  sorry

end NUMINAMATH_CALUDE_brian_pencils_given_to_friend_l2252_225273


namespace NUMINAMATH_CALUDE_remaining_black_cards_after_removal_l2252_225245

/-- Represents a deck of cards -/
structure Deck :=
  (black_cards : ℕ)

/-- Calculates the number of remaining black cards after removing some -/
def remaining_black_cards (d : Deck) (removed : ℕ) : ℕ :=
  d.black_cards - removed

/-- Theorem stating that removing 5 black cards from a deck with 26 black cards leaves 21 black cards -/
theorem remaining_black_cards_after_removal :
  ∀ (d : Deck), d.black_cards = 26 → remaining_black_cards d 5 = 21 := by
  sorry


end NUMINAMATH_CALUDE_remaining_black_cards_after_removal_l2252_225245


namespace NUMINAMATH_CALUDE_certain_number_proof_l2252_225235

theorem certain_number_proof (x y C : ℝ) : 
  (2 * x - y = C) → (6 * x - 3 * y = 12) → C = 4 := by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2252_225235


namespace NUMINAMATH_CALUDE_ratio_of_vectors_l2252_225287

/-- Given points O, A, B, C in a Cartesian coordinate system where O is the origin,
    prove that if OC = 2/3 * OA + 1/3 * OB, then |AC| / |AB| = 1/3 -/
theorem ratio_of_vectors (O A B C : ℝ × ℝ × ℝ) (h : C = (2/3 : ℝ) • A + (1/3 : ℝ) • B) :
  ‖C - A‖ / ‖B - A‖ = 1/3 := by sorry

end NUMINAMATH_CALUDE_ratio_of_vectors_l2252_225287


namespace NUMINAMATH_CALUDE_fixed_fee_is_ten_l2252_225255

/-- Represents the billing structure for an online service provider -/
structure BillingStructure where
  fixed_fee : ℝ
  hourly_charge : ℝ

/-- Represents the monthly usage and bill -/
structure MonthlyBill where
  connect_time : ℝ
  total_bill : ℝ

/-- The billing problem with given conditions -/
def billing_problem (b : BillingStructure) : Prop :=
  ∃ (feb_time : ℝ),
    let feb : MonthlyBill := ⟨feb_time, 20⟩
    let mar : MonthlyBill := ⟨2 * feb_time, 30⟩
    let apr : MonthlyBill := ⟨3 * feb_time, 40⟩
    (b.fixed_fee + b.hourly_charge * feb.connect_time = feb.total_bill) ∧
    (b.fixed_fee + b.hourly_charge * mar.connect_time = mar.total_bill) ∧
    (b.fixed_fee + b.hourly_charge * apr.connect_time = apr.total_bill)

/-- The theorem stating that the fixed monthly fee is $10.00 -/
theorem fixed_fee_is_ten :
  ∀ b : BillingStructure, billing_problem b → b.fixed_fee = 10 := by
  sorry


end NUMINAMATH_CALUDE_fixed_fee_is_ten_l2252_225255


namespace NUMINAMATH_CALUDE_min_value_expression_l2252_225231

theorem min_value_expression (x y : ℝ) :
  Real.sqrt (x^2 + y^2 - 2*x - 2*y + 2) + 
  Real.sqrt (x^2 + y^2 - 2*x + 4*y + 2*Real.sqrt 3*y + 8 + 4*Real.sqrt 3) + 
  Real.sqrt (x^2 + y^2 + 8*x + 4*Real.sqrt 3*x - 4*y + 32 + 16*Real.sqrt 3) ≥ 
  3 * Real.sqrt 6 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2252_225231


namespace NUMINAMATH_CALUDE_inequality_proof_l2252_225227

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / Real.sqrt y + y / Real.sqrt x ≥ Real.sqrt x + Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2252_225227


namespace NUMINAMATH_CALUDE_parallel_segments_between_parallel_planes_are_equal_l2252_225282

/-- Two planes are parallel if they do not intersect -/
def ParallelPlanes (p q : Plane) : Prop := sorry

/-- A line segment between two planes -/
def LineSegmentBetweenPlanes (p q : Plane) (s : Segment) : Prop := sorry

/-- Two line segments are parallel -/
def ParallelSegments (s₁ s₂ : Segment) : Prop := sorry

/-- Two line segments are equal (have the same length) -/
def EqualSegments (s₁ s₂ : Segment) : Prop := sorry

/-- Theorem: Parallel line segments between two parallel planes are equal -/
theorem parallel_segments_between_parallel_planes_are_equal 
  (p q : Plane) (s₁ s₂ : Segment) :
  ParallelPlanes p q →
  LineSegmentBetweenPlanes p q s₁ →
  LineSegmentBetweenPlanes p q s₂ →
  ParallelSegments s₁ s₂ →
  EqualSegments s₁ s₂ := by
  sorry

end NUMINAMATH_CALUDE_parallel_segments_between_parallel_planes_are_equal_l2252_225282


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l2252_225208

theorem arccos_equation_solution :
  ∃! x : ℝ, 
    x = 1 / (2 * Real.sqrt (19 - 8 * Real.sqrt 2)) ∧
    Real.arccos (4 * x) - Real.arccos (2 * x) = π / 4 ∧
    0 ≤ 4 * x ∧ 4 * x ≤ 1 ∧
    0 ≤ 2 * x ∧ 2 * x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l2252_225208


namespace NUMINAMATH_CALUDE_two_ducks_in_garden_l2252_225257

/-- The number of ducks in a garden with dogs and ducks -/
def number_of_ducks (num_dogs : ℕ) (total_feet : ℕ) : ℕ :=
  (total_feet - 4 * num_dogs) / 2

/-- Theorem: There are 2 ducks in the garden -/
theorem two_ducks_in_garden : number_of_ducks 6 28 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_ducks_in_garden_l2252_225257


namespace NUMINAMATH_CALUDE_inequality_proof_l2252_225281

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y ≥ 1) :
  x^3 + y^3 + 4*x*y ≥ x^2 + y^2 + x + y + 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2252_225281


namespace NUMINAMATH_CALUDE_complex_sum_equality_l2252_225285

theorem complex_sum_equality : ∃ (r θ : ℝ), 
  5 * Complex.exp (2 * π * Complex.I / 13) + 5 * Complex.exp (17 * π * Complex.I / 26) = 
  r * Complex.exp (θ * Complex.I) ∧ 
  r = 5 * Real.sqrt 2 ∧ 
  θ = 21 * π / 52 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l2252_225285


namespace NUMINAMATH_CALUDE_sin_145_cos_35_l2252_225232

theorem sin_145_cos_35 :
  Real.sin (145 * π / 180) * Real.cos (35 * π / 180) = (1/2) * Real.sin (70 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_sin_145_cos_35_l2252_225232


namespace NUMINAMATH_CALUDE_rotate_vector_90_degrees_l2252_225220

/-- Given points O and P in a 2D Cartesian coordinate system, and Q obtained by rotating OP counterclockwise by π/2, prove that Q has coordinates (-2, 1) -/
theorem rotate_vector_90_degrees (O P Q : ℝ × ℝ) : 
  O = (0, 0) → 
  P = (1, 2) → 
  (Q.1 - O.1, Q.2 - O.2) = (-(P.2 - O.2), P.1 - O.1) → 
  Q = (-2, 1) := by
  sorry

end NUMINAMATH_CALUDE_rotate_vector_90_degrees_l2252_225220


namespace NUMINAMATH_CALUDE_lcm_problem_l2252_225251

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 6) (h2 : a * b = 432) :
  Nat.lcm a b = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2252_225251


namespace NUMINAMATH_CALUDE_girls_percentage_after_boy_added_l2252_225224

theorem girls_percentage_after_boy_added (initial_boys initial_girls added_boys : ℕ) 
  (h1 : initial_boys = 11)
  (h2 : initial_girls = 13)
  (h3 : added_boys = 1) :
  (initial_girls : ℚ) / ((initial_boys + added_boys + initial_girls) : ℚ) = 52 / 100 := by
sorry

end NUMINAMATH_CALUDE_girls_percentage_after_boy_added_l2252_225224


namespace NUMINAMATH_CALUDE_prime_divides_all_f_l2252_225226

def f (n x : ℕ) : ℕ := Nat.choose n x

theorem prime_divides_all_f (p : ℕ) (hp : Prime p) (n : ℕ) (hn : n > 1) :
  (∀ x : ℕ, 1 ≤ x ∧ x < n → p ∣ f n x) ↔ ∃ m : ℕ, n = p ^ m :=
sorry

end NUMINAMATH_CALUDE_prime_divides_all_f_l2252_225226


namespace NUMINAMATH_CALUDE_cube_string_length_l2252_225275

theorem cube_string_length (volume : ℝ) (edge_length : ℝ) (string_length : ℝ) : 
  volume = 3375 → 
  edge_length ^ 3 = volume →
  string_length = 12 * edge_length →
  string_length = 180 := by sorry

end NUMINAMATH_CALUDE_cube_string_length_l2252_225275


namespace NUMINAMATH_CALUDE_pool_filling_solution_l2252_225225

/-- Represents the time taken to fill a pool given two pumps with specific properties -/
def pool_filling_time (pool_volume : ℝ) : Prop :=
  ∃ (t1 t2 : ℝ),
    t1 > 0 ∧ t2 > 0 ∧
    -- First pump fills 8 hours faster than second pump
    t2 - t1 = 8 ∧
    -- Second pump initially runs for twice the time of both pumps together
    2 * (1 / (1/t1 + 1/t2)) * (1/t2) +
    -- Then both pumps run for 1.5 hours
    1.5 * (1/t1 + 1/t2) = 1 ∧
    -- Times for each pump to fill separately
    t1 = 4 ∧ t2 = 12

/-- Theorem stating the existence of a solution for the pool filling problem -/
theorem pool_filling_solution (pool_volume : ℝ) (h : pool_volume > 0) :
  pool_filling_time pool_volume :=
sorry

end NUMINAMATH_CALUDE_pool_filling_solution_l2252_225225


namespace NUMINAMATH_CALUDE_nth_equation_l2252_225212

theorem nth_equation (n : ℕ) : ((n + 1)^2 - n^2 - 1) / 2 = n := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l2252_225212


namespace NUMINAMATH_CALUDE_isosceles_triangle_solution_l2252_225211

def isosceles_triangle_sides (perimeter : ℝ) (height_ratio : ℝ) : Prop :=
  let base := 130
  let leg := 169
  perimeter = base + 2 * leg ∧
  height_ratio = 10 / 13 ∧
  base * (13 : ℝ) = leg * (10 : ℝ)

theorem isosceles_triangle_solution :
  isosceles_triangle_sides 468 (10 / 13) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_solution_l2252_225211


namespace NUMINAMATH_CALUDE_condo_cats_count_l2252_225238

theorem condo_cats_count :
  ∀ (x y z : ℕ),
    x + y + z = 29 →
    x = z →
    87 = x * 1 + y * 3 + z * 5 :=
by
  sorry

end NUMINAMATH_CALUDE_condo_cats_count_l2252_225238


namespace NUMINAMATH_CALUDE_bottle_cap_count_l2252_225293

/-- Represents the number of bottle caps in one ounce -/
def caps_per_ounce : ℕ := 7

/-- Represents the weight of the bottle cap collection in pounds -/
def collection_weight_pounds : ℕ := 18

/-- Represents the number of ounces in one pound -/
def ounces_per_pound : ℕ := 16

/-- Calculates the total number of bottle caps in the collection -/
def total_caps : ℕ := collection_weight_pounds * ounces_per_pound * caps_per_ounce

theorem bottle_cap_count : total_caps = 2016 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_count_l2252_225293


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2252_225209

-- Define the complex number i
def i : ℂ := Complex.I

-- Define z as given in the problem
def z : ℂ := (1 + i) * i

-- Theorem statement
theorem imaginary_part_of_z :
  Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2252_225209


namespace NUMINAMATH_CALUDE_triangle_properties_l2252_225222

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  1 + Real.tan t.C / Real.tan t.B = 2 * t.a / t.b

def condition2 (t : Triangle) : Prop :=
  (t.a + t.b)^2 - t.c^2 = 4

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) : 
  t.C = Real.pi / 3 ∧ 
  ∃ (min : ℝ), min = -4 ∧ ∀ (x : ℝ), x ≥ min → 1 / t.b^2 - 3 * t.a ≥ x :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2252_225222


namespace NUMINAMATH_CALUDE_parabola_directrix_tangent_to_circle_l2252_225221

/-- The value of p for a parabola y^2 = 2px (p > 0) whose directrix is tangent to the circle (x-3)^2 + y^2 = 16 -/
theorem parabola_directrix_tangent_to_circle : 
  ∃ (p : ℝ), p > 0 ∧ 
  (∃ (x y : ℝ), y^2 = 2*p*x) ∧ 
  (∃ (x y : ℝ), (x-3)^2 + y^2 = 16) ∧
  (∃ (x : ℝ), x = -p/2 ∧ (x-3)^2 = 16) →
  p = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_tangent_to_circle_l2252_225221


namespace NUMINAMATH_CALUDE_inequality_proof_l2252_225205

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  Real.sqrt (a^(1 - a) * b^(1 - b) * c^(1 - c)) ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2252_225205


namespace NUMINAMATH_CALUDE_cubes_occupy_two_thirds_l2252_225284

/-- The dimensions of the rectangular box in inches -/
def box_dimensions : Fin 3 → ℕ
| 0 => 8
| 1 => 6
| 2 => 12
| _ => 0

/-- The side length of a cube in inches -/
def cube_side_length : ℕ := 4

/-- The volume of the rectangular box -/
def box_volume : ℕ := (box_dimensions 0) * (box_dimensions 1) * (box_dimensions 2)

/-- The volume occupied by cubes -/
def cubes_volume : ℕ := 
  ((box_dimensions 0) / cube_side_length) * 
  ((box_dimensions 1) / cube_side_length) * 
  ((box_dimensions 2) / cube_side_length) * 
  (cube_side_length ^ 3)

/-- The percentage of the box volume occupied by cubes -/
def volume_percentage : ℚ := (cubes_volume : ℚ) / (box_volume : ℚ) * 100

theorem cubes_occupy_two_thirds : volume_percentage = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cubes_occupy_two_thirds_l2252_225284


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l2252_225298

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -1 / x else 2 * Real.sqrt x

theorem f_composition_negative_two : f (f (-2)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l2252_225298


namespace NUMINAMATH_CALUDE_pool_length_l2252_225204

theorem pool_length (width : ℝ) (depth : ℝ) (capacity : ℝ) (drain_rate : ℝ) (drain_time : ℝ) :
  width = 50 →
  depth = 10 →
  capacity = 0.8 →
  drain_rate = 60 →
  drain_time = 1000 →
  ∃ (length : ℝ), length = 150 ∧ capacity * width * length * depth = drain_rate * drain_time :=
by sorry

end NUMINAMATH_CALUDE_pool_length_l2252_225204


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2252_225279

theorem max_value_of_expression (a b : ℝ) (h1 : 300 ≤ a ∧ a ≤ 500) 
  (h2 : 500 ≤ b ∧ b ≤ 1500) : 
  let c : ℝ := 100
  ∀ x ∈ Set.Icc 300 500, ∀ y ∈ Set.Icc 500 1500, 
    (b + c) / (a - c) ≤ 8 ∧ (y + c) / (x - c) ≤ 8 :=
by
  sorry

#check max_value_of_expression

end NUMINAMATH_CALUDE_max_value_of_expression_l2252_225279


namespace NUMINAMATH_CALUDE_problem_solution_l2252_225268

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 / x) * (2 / y) = 1 / 3) : x * y = 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2252_225268


namespace NUMINAMATH_CALUDE_max_value_abc_l2252_225265

theorem max_value_abc (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (sum_eq_3 : a + b + c = 3) :
  ∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 → a + b^2 + c^4 ≤ x + y^2 + z^4 ∧ a + b^2 + c^4 ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l2252_225265


namespace NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l2252_225240

theorem inscribed_rectangle_circle_circumference :
  ∀ (rectangle_width rectangle_height : ℝ) (circle_circumference : ℝ),
    rectangle_width = 9 →
    rectangle_height = 12 →
    (rectangle_width ^ 2 + rectangle_height ^ 2).sqrt * π = circle_circumference →
    circle_circumference = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l2252_225240


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2252_225277

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 5| ≤ 7} = {x : ℝ | -1 ≤ x ∧ x ≤ 6} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2252_225277


namespace NUMINAMATH_CALUDE_correct_both_problems_l2252_225283

theorem correct_both_problems (total : ℕ) (correct_sets : ℕ) (correct_functions : ℕ) (wrong_both : ℕ)
  (h1 : total = 50)
  (h2 : correct_sets = 40)
  (h3 : correct_functions = 31)
  (h4 : wrong_both = 4) :
  correct_sets + correct_functions - (total - wrong_both) = 25 := by
sorry

end NUMINAMATH_CALUDE_correct_both_problems_l2252_225283


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2252_225258

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) :
  (1 / x + 1 / y) ≥ 1 / 3 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 12 ∧ 1 / x + 1 / y = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2252_225258


namespace NUMINAMATH_CALUDE_absolute_value_five_l2252_225207

theorem absolute_value_five (x : ℝ) : |x| = 5 → x = 5 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_five_l2252_225207


namespace NUMINAMATH_CALUDE_solve_for_x_l2252_225253

theorem solve_for_x (x y : ℝ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2252_225253


namespace NUMINAMATH_CALUDE_mixed_number_comparison_l2252_225210

theorem mixed_number_comparison : (-2 - 1/3 : ℚ) < -2.3 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_comparison_l2252_225210


namespace NUMINAMATH_CALUDE_cube_sum_over_product_l2252_225286

theorem cube_sum_over_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h_sum : x + y + z = 3) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_l2252_225286


namespace NUMINAMATH_CALUDE_time_spent_playing_games_l2252_225299

/-- Calculates the time spent playing games during a flight -/
theorem time_spent_playing_games 
  (total_flight_time : ℕ) 
  (reading_time : ℕ) 
  (movie_time : ℕ) 
  (dinner_time : ℕ) 
  (radio_time : ℕ) 
  (nap_time : ℕ) : 
  total_flight_time = 11 * 60 + 20 → 
  reading_time = 2 * 60 → 
  movie_time = 4 * 60 → 
  dinner_time = 30 → 
  radio_time = 40 → 
  nap_time = 3 * 60 → 
  total_flight_time - (reading_time + movie_time + dinner_time + radio_time + nap_time) = 70 := by
sorry

end NUMINAMATH_CALUDE_time_spent_playing_games_l2252_225299


namespace NUMINAMATH_CALUDE_initial_gasoline_percentage_l2252_225262

/-- Proves that the initial gasoline percentage is 95% given the problem conditions --/
theorem initial_gasoline_percentage
  (initial_volume : ℝ)
  (initial_ethanol_percentage : ℝ)
  (desired_ethanol_percentage : ℝ)
  (added_ethanol : ℝ)
  (h1 : initial_volume = 54)
  (h2 : initial_ethanol_percentage = 0.05)
  (h3 : desired_ethanol_percentage = 0.10)
  (h4 : added_ethanol = 3)
  (h5 : initial_volume * initial_ethanol_percentage + added_ethanol = 
        (initial_volume + added_ethanol) * desired_ethanol_percentage) :
  1 - initial_ethanol_percentage = 0.95 := by
  sorry

end NUMINAMATH_CALUDE_initial_gasoline_percentage_l2252_225262


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2252_225296

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq_two : x + y + z = 2) :
  1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) ≥ 27 / 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2252_225296


namespace NUMINAMATH_CALUDE_cookie_sale_total_l2252_225246

theorem cookie_sale_total (raisin_cookies : ℕ) (ratio : ℚ) : 
  raisin_cookies = 42 → ratio = 6/1 → raisin_cookies + (raisin_cookies / ratio.num) = 49 :=
by sorry

end NUMINAMATH_CALUDE_cookie_sale_total_l2252_225246


namespace NUMINAMATH_CALUDE_hannahs_peppers_total_l2252_225223

theorem hannahs_peppers_total :
  let green_peppers : ℝ := 0.3333333333333333
  let red_peppers : ℝ := 0.4444444444444444
  let yellow_peppers : ℝ := 0.2222222222222222
  let orange_peppers : ℝ := 0.7777777777777778
  green_peppers + red_peppers + yellow_peppers + orange_peppers = 1.7777777777777777 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_peppers_total_l2252_225223


namespace NUMINAMATH_CALUDE_high_school_elite_season_games_l2252_225289

/-- The number of teams in the "High School Elite" basketball league -/
def num_teams : ℕ := 8

/-- The number of times each team plays every other team -/
def games_per_pairing : ℕ := 3

/-- The number of games each team plays against non-conference opponents -/
def non_conference_games : ℕ := 5

/-- The total number of games in a season for the "High School Elite" league -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pairing) + (num_teams * non_conference_games)

theorem high_school_elite_season_games :
  total_games = 124 := by sorry

end NUMINAMATH_CALUDE_high_school_elite_season_games_l2252_225289


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l2252_225248

/-- The perpendicular bisector of a line segment AB is the line that passes through 
    the midpoint of AB and is perpendicular to AB. This theorem proves that 
    y = -2x + 3 is the equation of the perpendicular bisector of the line segment 
    connecting points A(-1, 0) and B(3, 2). -/
theorem perpendicular_bisector_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (3, 2)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let slope_AB : ℝ := (B.2 - A.2) / (B.1 - A.1)
  let slope_perp : ℝ := -1 / slope_AB
  y = -2 * x + 3 ↔ 
    (x, y) ∈ {p : ℝ × ℝ | (p.1 - midpoint.1) * slope_AB = (midpoint.2 - p.2)} ∧
    (y - midpoint.2) = slope_perp * (x - midpoint.1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l2252_225248


namespace NUMINAMATH_CALUDE_library_book_selection_l2252_225267

theorem library_book_selection (math_books : Nat) (literature_books : Nat) (english_books : Nat) :
  math_books = 3 →
  literature_books = 5 →
  english_books = 8 →
  math_books + literature_books + english_books = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_library_book_selection_l2252_225267


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2252_225229

def U : Set ℕ := {x | 1 < x ∧ x < 5}

def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : 
  (U \ A) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2252_225229


namespace NUMINAMATH_CALUDE_calculate_expression_l2252_225247

theorem calculate_expression : |1 - Real.sqrt 2| - Real.sqrt 8 + (Real.sqrt 2 - 1)^0 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2252_225247


namespace NUMINAMATH_CALUDE_pascal_ratio_in_row_98_l2252_225290

/-- Pascal's Triangle entry -/
def pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Check if three consecutive entries in a row are in ratio 4:5:6 -/
def hasRatio456 (n : ℕ) : Prop :=
  ∃ r : ℕ, 
    (pascal n r : ℚ) / (pascal n (r + 1)) = 4 / 5 ∧
    (pascal n (r + 1) : ℚ) / (pascal n (r + 2)) = 5 / 6

theorem pascal_ratio_in_row_98 : hasRatio456 98 := by
  sorry

end NUMINAMATH_CALUDE_pascal_ratio_in_row_98_l2252_225290


namespace NUMINAMATH_CALUDE_gecko_eggs_calcification_fraction_l2252_225288

def total_eggs : ℕ := 30
def infertile_percentage : ℚ := 1/5
def hatched_eggs : ℕ := 16

theorem gecko_eggs_calcification_fraction :
  (total_eggs * (1 - infertile_percentage) - hatched_eggs) / (total_eggs * (1 - infertile_percentage)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_gecko_eggs_calcification_fraction_l2252_225288


namespace NUMINAMATH_CALUDE_smallest_voltage_l2252_225241

theorem smallest_voltage (a b c : ℕ) : 
  a ≤ 10 ∧ b ≤ 10 ∧ c ≤ 10 ∧
  (a + b + c : ℚ) / 3 = 4 ∧
  (a + b + c) % 5 = 0 →
  min a (min b c) = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_voltage_l2252_225241


namespace NUMINAMATH_CALUDE_new_average_salary_l2252_225219

theorem new_average_salary
  (initial_average : ℚ)
  (old_supervisor_salary : ℚ)
  (new_supervisor_salary : ℚ)
  (num_people : ℕ)
  (h1 : initial_average = 430)
  (h2 : old_supervisor_salary = 870)
  (h3 : new_supervisor_salary = 690)
  (h4 : num_people = 9)
  : (num_people - 1 : ℚ) * initial_average + new_supervisor_salary - old_supervisor_salary = 410 * num_people :=
by
  sorry

#eval (9 - 1 : ℚ) * 430 + 690 - 870
#eval 410 * 9

end NUMINAMATH_CALUDE_new_average_salary_l2252_225219


namespace NUMINAMATH_CALUDE_pentagon_angle_sum_l2252_225295

theorem pentagon_angle_sum (a b c d : ℝ) (h1 : a = 130) (h2 : b = 95) (h3 : c = 110) (h4 : d = 104) :
  ∃ q : ℝ, a + b + c + d + q = 540 ∧ q = 101 := by sorry

end NUMINAMATH_CALUDE_pentagon_angle_sum_l2252_225295


namespace NUMINAMATH_CALUDE_seventh_root_ratio_l2252_225294

theorem seventh_root_ratio (x : ℝ) (hx : x > 0) :
  (x ^ (1/2)) / (x ^ (1/4)) = x ^ (1/4) :=
sorry

end NUMINAMATH_CALUDE_seventh_root_ratio_l2252_225294


namespace NUMINAMATH_CALUDE_trajectory_equation_l2252_225297

/-- The trajectory of a point whose sum of distances to the coordinate axes is 6 -/
theorem trajectory_equation (x y : ℝ) : 
  (dist x 0 + dist y 0 = 6) → (|x| + |y| = 6) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2252_225297


namespace NUMINAMATH_CALUDE_arrangement_count_is_70_l2252_225239

/-- The number of ways to arrange 6 indistinguishable objects of type A
    and 4 indistinguishable objects of type B in a row of 10 positions,
    where type A objects must occupy the first and last positions. -/
def arrangement_count : ℕ := sorry

/-- Theorem stating that the number of arrangements is 70 -/
theorem arrangement_count_is_70 : arrangement_count = 70 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_70_l2252_225239


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l2252_225292

/-- The equation of a conic section -/
def conic_equation (x y : ℝ) : Prop :=
  (3*x - 2)^2 - 2*(5*y + 1)^2 = 288

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
    ∀ x y, f x y ↔ a * x^2 + b * y^2 + c * x + d * y + e = 0

/-- Theorem: The given conic equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l2252_225292


namespace NUMINAMATH_CALUDE_conference_handshakes_l2252_225254

/-- The number of handshakes in a conference of n people where each person
    shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := Nat.choose n 2

/-- Theorem stating that in a conference of 10 people, where each person
    shakes hands with every other person exactly once, there are 45 handshakes. -/
theorem conference_handshakes :
  handshakes 10 = 45 := by sorry

end NUMINAMATH_CALUDE_conference_handshakes_l2252_225254


namespace NUMINAMATH_CALUDE_decimal_to_base5_l2252_225200

theorem decimal_to_base5 : 
  (3 * 5^2 + 2 * 5^1 + 3 * 5^0 : ℕ) = 88 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_base5_l2252_225200


namespace NUMINAMATH_CALUDE_solve_probability_problem_l2252_225214

/-- Given three independent events A, B, and C with their respective probabilities -/
def probability_problem (P_A P_B P_C : ℝ) : Prop :=
  0 ≤ P_A ∧ P_A ≤ 1 ∧
  0 ≤ P_B ∧ P_B ≤ 1 ∧
  0 ≤ P_C ∧ P_C ≤ 1 →
  -- All three events occur simultaneously
  P_A * P_B * P_C = 0.612 ∧
  -- At least two events do not occur
  (1 - P_A) * (1 - P_B) * P_C +
  (1 - P_A) * P_B * (1 - P_C) +
  P_A * (1 - P_B) * (1 - P_C) +
  (1 - P_A) * (1 - P_B) * (1 - P_C) = 0.059

/-- The theorem stating the solution to the probability problem -/
theorem solve_probability_problem :
  probability_problem 0.9 0.8 0.85 := by
  sorry


end NUMINAMATH_CALUDE_solve_probability_problem_l2252_225214


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_abs_value_l2252_225217

/-- A fourth-degree polynomial with real coefficients -/
def fourth_degree_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d e : ℝ, ∀ x, f x = a * x^4 + b * x^3 + c * x^2 + d * x + e

/-- The absolute value of f at specific points is 16 -/
def abs_value_16 (f : ℝ → ℝ) : Prop :=
  |f 1| = 16 ∧ |f 3| = 16 ∧ |f 4| = 16 ∧ |f 5| = 16 ∧ |f 7| = 16

theorem fourth_degree_polynomial_abs_value (f : ℝ → ℝ) :
  fourth_degree_polynomial f → abs_value_16 f → |f 0| = 436 := by
  sorry

end NUMINAMATH_CALUDE_fourth_degree_polynomial_abs_value_l2252_225217


namespace NUMINAMATH_CALUDE_expression_equality_l2252_225213

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 = 1 / y^2) :
  (x^2 - 1/x^2) * (y^2 + 1/y^2) = x^4 - y^4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2252_225213


namespace NUMINAMATH_CALUDE_basketball_price_proof_l2252_225269

/-- The price of a basketball in yuan -/
def basketball_price : ℕ := 124

/-- The price of a soccer ball in yuan -/
def soccer_ball_price : ℕ := 62

/-- The total cost of a basketball and a soccer ball in yuan -/
def total_cost : ℕ := 186

theorem basketball_price_proof :
  (basketball_price = 124) ∧
  (basketball_price + soccer_ball_price = total_cost) ∧
  (basketball_price = 2 * soccer_ball_price) :=
sorry

end NUMINAMATH_CALUDE_basketball_price_proof_l2252_225269


namespace NUMINAMATH_CALUDE_cycle_sale_theorem_l2252_225278

/-- Calculates the net total amount received after selling three cycles and paying tax -/
def net_total_amount (price1 price2 price3 : ℚ) (profit1 loss2 profit3 tax_rate : ℚ) : ℚ :=
  let sell1 := price1 * (1 + profit1)
  let sell2 := price2 * (1 - loss2)
  let sell3 := price3 * (1 + profit3)
  let total_sell := sell1 + sell2 + sell3
  let tax := total_sell * tax_rate
  total_sell - tax

theorem cycle_sale_theorem :
  net_total_amount 3600 4800 6000 (20/100) (15/100) (10/100) (5/100) = 14250 := by
  sorry

#eval net_total_amount 3600 4800 6000 (20/100) (15/100) (10/100) (5/100)

end NUMINAMATH_CALUDE_cycle_sale_theorem_l2252_225278


namespace NUMINAMATH_CALUDE_division_with_remainder_l2252_225280

theorem division_with_remainder (A : ℕ) : 
  (A / 7 = 5) ∧ (A % 7 = 3) → A = 38 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l2252_225280


namespace NUMINAMATH_CALUDE_circle_transformation_l2252_225261

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

theorem circle_transformation :
  let initial_point : ℝ × ℝ := (3, -4)
  let reflected_point := reflect_x initial_point
  let final_point := translate_right reflected_point 10
  final_point = (13, 4) := by sorry

end NUMINAMATH_CALUDE_circle_transformation_l2252_225261


namespace NUMINAMATH_CALUDE_pyramid_triangular_faces_area_l2252_225203

/-- The area of triangular faces of a right square-based pyramid -/
theorem pyramid_triangular_faces_area
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  4 * (1/2 * base_edge * Real.sqrt (lateral_edge^2 - (base_edge/2)^2)) = 16 * Real.sqrt 33 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_triangular_faces_area_l2252_225203
