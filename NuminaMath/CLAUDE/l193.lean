import Mathlib

namespace NUMINAMATH_CALUDE_probability_three_heads_seven_tosses_prove_probability_three_heads_seven_tosses_l193_19321

/-- The probability of getting exactly 3 heads in 7 fair coin tosses -/
theorem probability_three_heads_seven_tosses : ℚ :=
  35 / 128

/-- Prove that the probability of getting exactly 3 heads in 7 fair coin tosses is 35/128 -/
theorem prove_probability_three_heads_seven_tosses :
  probability_three_heads_seven_tosses = 35 / 128 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_heads_seven_tosses_prove_probability_three_heads_seven_tosses_l193_19321


namespace NUMINAMATH_CALUDE_tan_neg_alpha_problem_l193_19359

theorem tan_neg_alpha_problem (α : Real) (h : Real.tan (-α) = -2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 ∧ Real.sin (2 * α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_alpha_problem_l193_19359


namespace NUMINAMATH_CALUDE_sixth_angle_measure_l193_19385

/-- The sum of interior angles in a hexagon -/
def hexagon_angle_sum : ℝ := 720

/-- The measures of the five known angles in the hexagon -/
def known_angles : List ℝ := [130, 95, 115, 120, 110]

/-- The theorem stating that the sixth angle in the hexagon measures 150° -/
theorem sixth_angle_measure :
  hexagon_angle_sum - (known_angles.sum) = 150 := by sorry

end NUMINAMATH_CALUDE_sixth_angle_measure_l193_19385


namespace NUMINAMATH_CALUDE_smallest_value_l193_19380

theorem smallest_value (x : ℝ) (h : 1 < x ∧ x < 2) : 
  (1 / x^2 < x) ∧ 
  (1 / x^2 < x^2) ∧ 
  (1 / x^2 < 2*x^2) ∧ 
  (1 / x^2 < 3*x) ∧ 
  (1 / x^2 < Real.sqrt x) ∧ 
  (1 / x^2 < 1 / x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_l193_19380


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l193_19334

/-- A rectangle in a 2D coordinate system --/
structure Rectangle where
  x1 : ℝ
  x2 : ℝ
  y1 : ℝ
  y2 : ℝ

/-- The area of a rectangle --/
def Rectangle.area (r : Rectangle) : ℝ :=
  |r.x2 - r.x1| * |r.y2 - r.y1|

/-- Theorem: If a rectangle with vertices (-8, y), (1, y), (1, -7), and (-8, -7) has an area of 72, then y = 1 --/
theorem rectangle_area_theorem (y : ℝ) :
  let r := Rectangle.mk (-8) 1 y (-7)
  r.area = 72 → y = 1 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_theorem_l193_19334


namespace NUMINAMATH_CALUDE_luke_laundry_loads_l193_19374

def total_clothing : ℕ := 47
def first_load : ℕ := 17
def pieces_per_load : ℕ := 6

theorem luke_laundry_loads : 
  (total_clothing - first_load) / pieces_per_load = 5 :=
by sorry

end NUMINAMATH_CALUDE_luke_laundry_loads_l193_19374


namespace NUMINAMATH_CALUDE_count_nonzero_terms_l193_19300

/-- The number of nonzero terms in the simplified expression of (x+y+z+w)^2008 + (x-y-z-w)^2008 -/
def nonzero_terms : ℕ := 56883810

/-- The exponent used in the expression -/
def exponent : ℕ := 2008

theorem count_nonzero_terms (a b c : ℤ) :
  nonzero_terms = (exponent + 3).choose 3 := by sorry

end NUMINAMATH_CALUDE_count_nonzero_terms_l193_19300


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l193_19320

/-- The eccentricity of a hyperbola with the given conditions is between 1 and 2 -/
theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ (x y e : ℝ),
    x^2 / a^2 - y^2 / b^2 = 1 ∧
    x ≥ a ∧
    ∃ (f1 f2 : ℝ × ℝ),
      (∀ (p : ℝ × ℝ), p.1^2 / a^2 - p.2^2 / b^2 = 1 →
        |p.1 - f1.1| - |p.1 - f2.1| = 2 * a * e) ∧
      |x - f1.1| = 3 * |x - f2.1| →
    1 < e ∧ e ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l193_19320


namespace NUMINAMATH_CALUDE_fraction_simplification_l193_19344

theorem fraction_simplification (x : ℝ) (h : x ≠ -2 ∧ x ≠ 2) :
  (x^2 - 4) / (x^2 - 4*x + 4) / ((x^2 + 4*x + 4) / (2*x - x^2)) = -x / (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l193_19344


namespace NUMINAMATH_CALUDE_sum_xyz_equals_four_l193_19363

theorem sum_xyz_equals_four (X Y Z : ℕ+) 
  (h_gcd : Nat.gcd X.val (Nat.gcd Y.val Z.val) = 1)
  (h_eq : (X : ℝ) * (Real.log 3 / Real.log 100) + (Y : ℝ) * (Real.log 4 / Real.log 100) = Z) :
  X + Y + Z = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_xyz_equals_four_l193_19363


namespace NUMINAMATH_CALUDE_speed_ratio_l193_19324

/-- The speed of Person A -/
def speed_A : ℝ := sorry

/-- The speed of Person B -/
def speed_B : ℝ := sorry

/-- The distance covered by Person A in a given time -/
def distance_A : ℝ := sorry

/-- The distance covered by Person B in the same time -/
def distance_B : ℝ := sorry

/-- The time taken for both persons to cover their respective distances -/
def time : ℝ := sorry

theorem speed_ratio :
  (speed_A / speed_B = 3 / 2) ∧
  (distance_A = 3) ∧
  (distance_B = 2) ∧
  (speed_A = distance_A / time) ∧
  (speed_B = distance_B / time) :=
by sorry

end NUMINAMATH_CALUDE_speed_ratio_l193_19324


namespace NUMINAMATH_CALUDE_solve_for_b_l193_19382

theorem solve_for_b (a b : ℝ) (h1 : 2 * a + 1 = 1) (h2 : 2 * b - 3 * a = 2) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l193_19382


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l193_19313

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a population with two equal-sized subgroups -/
structure Population where
  total : ℕ
  group1 : ℕ
  group2 : ℕ
  h1 : group1 + group2 = total
  h2 : group1 = group2

/-- Represents a sample drawn from a population -/
structure Sample where
  size : ℕ
  population : Population
  method : SamplingMethod

/-- Predicate to determine if a sampling method is appropriate for comparing subgroups -/
def is_appropriate_for_subgroup_comparison (s : Sample) : Prop :=
  s.method = SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the most appropriate method
    for comparing characteristics between two equal-sized subgroups -/
theorem stratified_sampling_most_appropriate
  (pop : Population)
  (sample_size : ℕ)
  (h_sample_size : sample_size > 0 ∧ sample_size < pop.total) :
  ∀ (s : Sample),
    s.population = pop →
    s.size = sample_size →
    is_appropriate_for_subgroup_comparison s ↔ s.method = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l193_19313


namespace NUMINAMATH_CALUDE_product_abcde_l193_19373

theorem product_abcde (a b c d e : ℚ) : 
  3 * a + 4 * b + 6 * c + 8 * d + 10 * e = 55 →
  4 * (d + c + e) = b →
  4 * b + 2 * c = a →
  c - 2 = d →
  d + 1 = e →
  a * b * c * d * e = -1912397372 / 78364164096 := by
sorry

end NUMINAMATH_CALUDE_product_abcde_l193_19373


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l193_19357

theorem nested_fraction_evaluation :
  2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l193_19357


namespace NUMINAMATH_CALUDE_specific_mountain_depth_l193_19386

/-- Represents a cone-shaped mountain partially submerged in water -/
structure Mountain where
  totalHeight : ℝ
  baseRadius : ℝ
  aboveWaterVolumeFraction : ℝ

/-- Calculates the depth of the ocean at the base of the mountain -/
def oceanDepth (m : Mountain) : ℝ :=
  m.totalHeight * (1 - (m.aboveWaterVolumeFraction) ^ (1/3))

/-- Theorem stating the ocean depth for the specific mountain described in the problem -/
theorem specific_mountain_depth :
  let m : Mountain := {
    totalHeight := 10000,
    baseRadius := 3000,
    aboveWaterVolumeFraction := 1/10
  }
  oceanDepth m = 5360 := by
  sorry


end NUMINAMATH_CALUDE_specific_mountain_depth_l193_19386


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l193_19342

theorem vector_difference_magnitude (x : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 4)
  (a.1 * b.1 + a.2 * b.2 = 10) → 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l193_19342


namespace NUMINAMATH_CALUDE_negative_two_two_two_two_mod_thirteen_l193_19340

theorem negative_two_two_two_two_mod_thirteen : ∃! n : ℤ, 0 ≤ n ∧ n < 13 ∧ -2222 ≡ n [ZMOD 13] ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_two_two_two_mod_thirteen_l193_19340


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l193_19312

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 12 * x + c = 0) →  -- exactly one solution
  (a + c = 14) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 7 - Real.sqrt 13 ∧ c = 7 + Real.sqrt 13) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l193_19312


namespace NUMINAMATH_CALUDE_max_students_planting_trees_l193_19392

theorem max_students_planting_trees :
  ∃ (a b : ℕ), 3 * a + 5 * b = 115 ∧
  ∀ (x y : ℕ), 3 * x + 5 * y = 115 → x + y ≤ a + b ∧
  a + b = 37 := by
sorry

end NUMINAMATH_CALUDE_max_students_planting_trees_l193_19392


namespace NUMINAMATH_CALUDE_dice_probability_l193_19343

/-- A fair 10-sided die -/
def ten_sided_die : Finset ℕ := Finset.range 10

/-- A fair 6-sided die -/
def six_sided_die : Finset ℕ := Finset.range 6

/-- The event that the number on the 10-sided die is less than or equal to the number on the 6-sided die -/
def favorable_event : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 ≤ p.2) (ten_sided_die.product six_sided_die)

/-- The probability of the event -/
def probability : ℚ :=
  (favorable_event.card : ℚ) / ((ten_sided_die.card * six_sided_die.card) : ℚ)

theorem dice_probability : probability = 7 / 20 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l193_19343


namespace NUMINAMATH_CALUDE_perpendicular_implies_m_eq_neg_one_parallel_implies_m_eq_neg_one_l193_19346

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0

-- Define perpendicularity of lines
def perpendicular (m : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, l₁ m x₁ y₁ → l₂ m x₂ y₂ → (m - 2) / 3 * m = -1

-- Define parallelism of lines
def parallel (m : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, l₁ m x₁ y₁ → l₂ m x₂ y₂ → (m - 2) / 3 = m

-- Theorem for perpendicular case
theorem perpendicular_implies_m_eq_neg_one :
  ∀ m : ℝ, perpendicular m → m = -1 := by sorry

-- Theorem for parallel case
theorem parallel_implies_m_eq_neg_one :
  ∀ m : ℝ, parallel m → m = -1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_implies_m_eq_neg_one_parallel_implies_m_eq_neg_one_l193_19346


namespace NUMINAMATH_CALUDE_triangle_inequality_ratio_123_l193_19368

theorem triangle_inequality_ratio_123 :
  ∀ (x : ℝ), x > 0 → ¬(x + 2*x > 3*x) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_ratio_123_l193_19368


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l193_19387

theorem condition_neither_sufficient_nor_necessary :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b ≥ ((a + b) / 2)^2) ∧
  (∃ a b : ℝ, a * b < ((a + b) / 2)^2 ∧ (a ≤ 0 ∨ b ≤ 0)) := by sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l193_19387


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l193_19318

theorem cubic_polynomial_satisfies_conditions :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = -2/3 * x^3 + 3 * x^2 - 35/3 * x - 2) ∧
    q 0 = -2 ∧
    q 1 = -8 ∧
    q 3 = -18 ∧
    q 5 = -52 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l193_19318


namespace NUMINAMATH_CALUDE_store_pricing_theorem_l193_19325

/-- Represents the store's pricing and discount options -/
structure Store where
  suit_price : ℕ
  tie_price : ℕ
  option1 : ℕ → ℕ → ℕ  -- number of suits, number of ties → total price
  option2 : ℕ → ℕ → ℕ  -- number of suits, number of ties → total price

/-- The specific store in the problem -/
def problem_store : Store :=
  { suit_price := 1000
  , tie_price := 200
  , option1 := λ s t => s * 1000 + (t - s) * 200
  , option2 := λ s t => (s * 1000 + t * 200) * 9 / 10 }

theorem store_pricing_theorem (x : ℕ) (h : x > 10) :
  let s := problem_store
  (∀ x, s.option1 10 x = 200 * x + 8000) ∧ 
  (∀ x, s.option2 10 x = 180 * x + 9000) ∧
  (s.option1 10 30 < s.option2 10 30) ∧
  (s.option1 10 10 + s.option2 0 20 < min (s.option1 10 30) (s.option2 10 30)) := by
  sorry

end NUMINAMATH_CALUDE_store_pricing_theorem_l193_19325


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l193_19369

theorem cylinder_volume_equality (x : ℝ) (hx : x > 0) : 
  (π * (7 + x)^2 * 5 = π * 7^2 * (5 + 2*x)) → x = 28/5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l193_19369


namespace NUMINAMATH_CALUDE_combined_distance_is_91261_136_l193_19397

/-- The combined distance traveled by friends in feet -/
def combined_distance : ℝ :=
  let mile_to_feet : ℝ := 5280
  let yard_to_feet : ℝ := 3
  let km_to_meter : ℝ := 1000
  let meter_to_feet : ℝ := 3.28084
  let lionel_miles : ℝ := 4
  let esther_yards : ℝ := 975
  let niklaus_feet : ℝ := 1287
  let isabella_km : ℝ := 18
  let sebastian_meters : ℝ := 2400
  lionel_miles * mile_to_feet +
  esther_yards * yard_to_feet +
  niklaus_feet +
  isabella_km * km_to_meter * meter_to_feet +
  sebastian_meters * meter_to_feet

/-- Theorem stating that the combined distance traveled by friends is 91261.136 feet -/
theorem combined_distance_is_91261_136 : combined_distance = 91261.136 := by
  sorry

end NUMINAMATH_CALUDE_combined_distance_is_91261_136_l193_19397


namespace NUMINAMATH_CALUDE_unique_intersection_l193_19339

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|2 * x - 1|

-- State the theorem
theorem unique_intersection :
  ∃! p : ℝ × ℝ, 
    f p.1 = g p.1 ∧ 
    p.1 = -1 ∧ 
    p.2 = -3 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_l193_19339


namespace NUMINAMATH_CALUDE_yard_length_with_32_trees_l193_19360

/-- The length of a yard with equally spaced trees -/
def yardLength (numTrees : ℕ) (distanceBetweenTrees : ℕ) : ℕ :=
  (numTrees - 1) * distanceBetweenTrees

/-- Theorem: The length of a yard with 32 equally spaced trees and 14 meters between consecutive trees is 434 meters -/
theorem yard_length_with_32_trees : yardLength 32 14 = 434 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_with_32_trees_l193_19360


namespace NUMINAMATH_CALUDE_hat_guessing_strategy_exists_l193_19331

/-- Represents a strategy for guessing hat numbers -/
def Strategy := (ι : Fin 2023 → ℕ) → Fin 2023 → ℕ

/-- Theorem stating that there exists a winning strategy for the hat guessing game -/
theorem hat_guessing_strategy_exists :
  ∃ (s : Strategy),
    ∀ (ι : Fin 2023 → ℕ),
      (∀ i, 1 ≤ ι i ∧ ι i ≤ 2023) →
      ∃ i, s ι i = ι i :=
sorry

end NUMINAMATH_CALUDE_hat_guessing_strategy_exists_l193_19331


namespace NUMINAMATH_CALUDE_expression_value_l193_19308

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (2 * x - 3 * y) - f (x + y) = -2 * x + 8 * y

/-- The main theorem stating that the given expression is always equal to 4 -/
theorem expression_value (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ t, f 4*t ≠ f 3*t → (f 5*t - f t) / (f 4*t - f 3*t) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l193_19308


namespace NUMINAMATH_CALUDE_min_value_quadratic_l193_19356

/-- The function f(x) = 3x^2 - 15x + 7 attains its minimum value when x = 5/2. -/
theorem min_value_quadratic (x : ℝ) :
  ∀ y : ℝ, 3 * x^2 - 15 * x + 7 ≤ 3 * y^2 - 15 * y + 7 ↔ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l193_19356


namespace NUMINAMATH_CALUDE_tangent_line_equation_l193_19338

-- Define the function f(x) = x³ + 1
def f (x : ℝ) : ℝ := x^3 + 1

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 3 * x - y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l193_19338


namespace NUMINAMATH_CALUDE_book_count_l193_19333

theorem book_count : ∃ (B : ℕ), 
  B > 0 ∧
  (2 * B) % 5 = 0 ∧  -- Two-fifths of books are reading books
  (3 * B) % 10 = 0 ∧ -- Three-tenths of books are math books
  (B * 3) / 10 - 1 = (B * 3) / 10 - (B * 3) % 10 / 10 - 1 ∧ -- Science books are one fewer than math books
  ((2 * B) / 5 + (3 * B) / 10 + ((3 * B) / 10 - 1) + 1 = B) ∧ -- Sum of all book types equals total
  B = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_count_l193_19333


namespace NUMINAMATH_CALUDE_christinas_total_distance_l193_19311

/-- The total distance Christina walks in a week given her routine -/
def christinas_weekly_distance (school_distance : ℕ) (friend_distance : ℕ) : ℕ :=
  (school_distance * 2 * 4) + (school_distance * 2 + friend_distance * 2)

/-- Theorem stating that Christina's total distance for the week is 74km -/
theorem christinas_total_distance :
  christinas_weekly_distance 7 2 = 74 := by
  sorry

end NUMINAMATH_CALUDE_christinas_total_distance_l193_19311


namespace NUMINAMATH_CALUDE_q_coordinates_l193_19330

/-- Triangle ABC with points G on AC and H on AB -/
structure Triangle (A B C G H : ℝ × ℝ) : Prop where
  g_on_ac : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ G = t • C + (1 - t) • A
  h_on_ab : ∃ s : ℝ, 0 < s ∧ s < 1 ∧ H = s • B + (1 - s) • A
  ag_gc_ratio : (G.1 - A.1) / (C.1 - G.1) = 3 / 2 ∧ (G.2 - A.2) / (C.2 - G.2) = 3 / 2
  ah_hb_ratio : (H.1 - A.1) / (B.1 - H.1) = 2 / 3 ∧ (H.2 - A.2) / (B.2 - H.2) = 2 / 3

/-- Q is the intersection of BG and CH -/
def Q (A B C G H : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: Coordinates of Q in terms of A, B, and C -/
theorem q_coordinates (A B C G H : ℝ × ℝ) (tri : Triangle A B C G H) :
  ∃ (u v w : ℝ), u + v + w = 1 ∧ 
    Q A B C G H = (u • A.1 + v • B.1 + w • C.1, u • A.2 + v • B.2 + w • C.2) ∧
    u = 5/13 ∧ v = 11/26 ∧ w = 3/13 :=
  sorry

end NUMINAMATH_CALUDE_q_coordinates_l193_19330


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l193_19372

theorem fraction_equals_zero (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l193_19372


namespace NUMINAMATH_CALUDE_opposite_pairs_l193_19326

theorem opposite_pairs : 
  (2^2 = -(-(2^2))) ∧ 
  (2^2 ≠ -((-2)^2)) ∧ 
  (-(-2) ≠ -|(-2)|) ∧ 
  (-2^3 ≠ -((-2)^3)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_pairs_l193_19326


namespace NUMINAMATH_CALUDE_max_roses_for_680_l193_19317

/-- Represents the pricing structure for roses -/
structure RosePricing where
  individual_price : ℚ
  dozen_price : ℚ
  two_dozen_price : ℚ
  five_dozen_price : ℚ
  discount_rate : ℚ
  discount_threshold : ℕ

/-- Calculates the maximum number of roses that can be purchased given a budget -/
def max_roses_purchased (pricing : RosePricing) (budget : ℚ) : ℕ :=
  sorry

/-- The specific pricing structure given in the problem -/
def problem_pricing : RosePricing :=
  { individual_price := 9/2,
    dozen_price := 36,
    two_dozen_price := 50,
    five_dozen_price := 110,
    discount_rate := 1/10,
    discount_threshold := 36 }

theorem max_roses_for_680 :
  max_roses_purchased problem_pricing 680 = 364 :=
sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l193_19317


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l193_19352

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l193_19352


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_four_satisfies_conditions_ninety_four_is_largest_l193_19361

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 100 ∧ n % 6 = 4 → n ≤ 94 :=
by
  sorry

theorem ninety_four_satisfies_conditions : 
  94 < 100 ∧ 94 % 6 = 4 :=
by
  sorry

theorem ninety_four_is_largest : 
  ∀ m : ℕ, m < 100 ∧ m % 6 = 4 → m ≤ 94 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_four_satisfies_conditions_ninety_four_is_largest_l193_19361


namespace NUMINAMATH_CALUDE_cat_and_mouse_positions_after_359_moves_l193_19366

/-- Represents the possible positions of the cat -/
inductive CatPosition
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

/-- Represents the possible positions of the mouse -/
inductive MousePosition
  | TopMiddle
  | TopRight
  | RightMiddle
  | BottomRight
  | BottomMiddle
  | BottomLeft
  | LeftMiddle
  | TopLeft

/-- Calculate the position of the cat after a given number of moves -/
def catPositionAfterMoves (moves : ℕ) : CatPosition :=
  match moves % 4 with
  | 0 => CatPosition.BottomLeft
  | 1 => CatPosition.TopLeft
  | 2 => CatPosition.TopRight
  | _ => CatPosition.BottomRight

/-- Calculate the position of the mouse after a given number of moves -/
def mousePositionAfterMoves (moves : ℕ) : MousePosition :=
  match moves % 8 with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.TopMiddle
  | 2 => MousePosition.TopRight
  | 3 => MousePosition.RightMiddle
  | 4 => MousePosition.BottomRight
  | 5 => MousePosition.BottomMiddle
  | 6 => MousePosition.BottomLeft
  | _ => MousePosition.LeftMiddle

theorem cat_and_mouse_positions_after_359_moves :
  (catPositionAfterMoves 359 = CatPosition.BottomRight) ∧
  (mousePositionAfterMoves 359 = MousePosition.LeftMiddle) := by
  sorry


end NUMINAMATH_CALUDE_cat_and_mouse_positions_after_359_moves_l193_19366


namespace NUMINAMATH_CALUDE_ellipse_and_line_equation_l193_19341

-- Define the ellipse C₁
def C₁ (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the point M
def M (x y : ℝ) : Prop := C₁ 2 (Real.sqrt 3) x y ∧ C₂ x y ∧ x > 0 ∧ y > 0

-- Define the distance between M and F₂
def MF₂_distance (x y : ℝ) : Prop := (x - 1)^2 + y^2 = (5/3)^2

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := y = Real.sqrt 6 * (x - m)

-- Define the perpendicularity condition
def perpendicular_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem ellipse_and_line_equation :
  ∃ (x y : ℝ),
    M x y ∧ MF₂_distance x y ∧
    (∀ (m : ℝ),
      (∃ (x₁ y₁ x₂ y₂ : ℝ),
        C₁ 2 (Real.sqrt 3) x₁ y₁ ∧ C₁ 2 (Real.sqrt 3) x₂ y₂ ∧
        line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧
        perpendicular_condition x₁ y₁ x₂ y₂) →
      m = Real.sqrt 2 ∨ m = -Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_equation_l193_19341


namespace NUMINAMATH_CALUDE_function_identity_l193_19319

def is_positive_integer (n : ℕ) : Prop := n > 0

theorem function_identity 
  (f : ℕ → ℕ) 
  (h1 : ∀ n, is_positive_integer n → is_positive_integer (f n))
  (h2 : ∀ n, is_positive_integer n → f (n + 1) > f (f n)) :
  ∀ n, is_positive_integer n → f n = n :=
sorry

end NUMINAMATH_CALUDE_function_identity_l193_19319


namespace NUMINAMATH_CALUDE_quadratic_factorization_l193_19329

theorem quadratic_factorization :
  ∀ x : ℝ, 12 * x^2 + 16 * x - 20 = 4 * (x - 1) * (3 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l193_19329


namespace NUMINAMATH_CALUDE_prime_power_sum_l193_19381

theorem prime_power_sum (w x y z : ℕ) :
  2^w * 3^x * 5^y * 7^z = 1260 →
  w + 2*x + 3*y + 4*z = 13 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l193_19381


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l193_19394

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l193_19394


namespace NUMINAMATH_CALUDE_customer_difference_l193_19304

theorem customer_difference (initial : ℕ) (remaining : ℕ) 
  (h1 : initial = 19) (h2 : remaining = 4) : 
  initial - remaining = 15 := by
  sorry

end NUMINAMATH_CALUDE_customer_difference_l193_19304


namespace NUMINAMATH_CALUDE_additional_beads_needed_bella_needs_twelve_more_beads_l193_19367

/-- Given the number of friends, beads per bracelet, and beads Bella has,
    calculate the number of additional beads needed. -/
theorem additional_beads_needed 
  (num_friends : ℕ) 
  (beads_per_bracelet : ℕ) 
  (beads_bella_has : ℕ) : ℕ :=
  (num_friends * beads_per_bracelet) - beads_bella_has

/-- Prove that Bella needs 12 more beads to make bracelets for her friends. -/
theorem bella_needs_twelve_more_beads : 
  additional_beads_needed 6 8 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_beads_needed_bella_needs_twelve_more_beads_l193_19367


namespace NUMINAMATH_CALUDE_unique_real_root_l193_19362

/-- A quadratic polynomial P(x) = x^2 - 2ax + b -/
def P (a b x : ℝ) : ℝ := x^2 - 2*a*x + b

/-- The condition that P(0), P(1), and P(2) form a geometric progression -/
def geometric_progression (a b : ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ P a b 1 = (P a b 0) * r ∧ P a b 2 = (P a b 0) * r^2

/-- The theorem stating that under given conditions, a = 1 is the only value for which P(x) = 0 has real roots -/
theorem unique_real_root (a b : ℝ) :
  geometric_progression a b ∧ P a b 0 * P a b 1 * P a b 2 ≠ 0 →
  (∃ x : ℝ, P a b x = 0) ↔ a = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_real_root_l193_19362


namespace NUMINAMATH_CALUDE_final_concentration_l193_19365

-- Define the volumes and concentrations
def volume1 : ℝ := 2
def concentration1 : ℝ := 0.4
def volume2 : ℝ := 3
def concentration2 : ℝ := 0.6

-- Define the total volume
def totalVolume : ℝ := volume1 + volume2

-- Define the total amount of acid
def totalAcid : ℝ := volume1 * concentration1 + volume2 * concentration2

-- Theorem: The final concentration is 52%
theorem final_concentration :
  totalAcid / totalVolume = 0.52 := by sorry

end NUMINAMATH_CALUDE_final_concentration_l193_19365


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l193_19348

/-- Given a triangle with two sides of lengths that are roots of x^2 - 5x + 6 = 0,
    the perimeter l of the triangle satisfies 6 < l < 10 -/
theorem triangle_perimeter_range : ∀ a b c : ℝ,
  (a^2 - 5*a + 6 = 0) →
  (b^2 - 5*b + 6 = 0) →
  (a ≠ b) →
  (a + b > c) →
  (b + c > a) →
  (c + a > b) →
  let l := a + b + c
  6 < l ∧ l < 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l193_19348


namespace NUMINAMATH_CALUDE_james_height_fraction_l193_19345

/-- Proves that James was 2/3 as tall as his uncle before the growth spurt -/
theorem james_height_fraction (uncle_height : ℝ) (james_growth : ℝ) (height_difference : ℝ) :
  uncle_height = 72 →
  james_growth = 10 →
  height_difference = 14 →
  (uncle_height - (james_growth + height_difference)) / uncle_height = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_james_height_fraction_l193_19345


namespace NUMINAMATH_CALUDE_f_has_one_or_two_zeros_l193_19332

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x - m^2

-- State the theorem
theorem f_has_one_or_two_zeros (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), f m x₁ = 0 ∧ f m x₂ = 0 ∧ (x₁ = x₂ ∨ x₁ ≠ x₂) :=
sorry

end NUMINAMATH_CALUDE_f_has_one_or_two_zeros_l193_19332


namespace NUMINAMATH_CALUDE_sum_of_function_values_positive_l193_19379

def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem sum_of_function_values_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h_monotone : is_monotone_increasing f)
  (h_odd : is_odd_function f)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_a3_positive : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_function_values_positive_l193_19379


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l193_19390

/-- Given that f(x) = e^(2x) - ae^x + 2x is an increasing function on ℝ, 
    prove that the range of a is (-∞, 4]. -/
theorem range_of_a_for_increasing_f (a : ℝ) : 
  (∀ x : ℝ, Monotone (fun x => Real.exp (2 * x) - a * Real.exp x + 2 * x)) →
  a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l193_19390


namespace NUMINAMATH_CALUDE_picnic_attendance_difference_picnic_attendance_difference_proof_l193_19398

/-- Proves that there are 80 more adults than children at a picnic -/
theorem picnic_attendance_difference : ℕ → Prop :=
  fun total_persons : ℕ =>
    ∀ (men women children adults : ℕ),
      total_persons = 240 →
      men = 120 →
      men = women + 80 →
      adults = men + women →
      total_persons = men + women + children →
      adults - children = 80

-- The proof is omitted
theorem picnic_attendance_difference_proof : picnic_attendance_difference 240 := by
  sorry

end NUMINAMATH_CALUDE_picnic_attendance_difference_picnic_attendance_difference_proof_l193_19398


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l193_19302

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 9 / 3.6) -- Convert 9 km/hr to m/s
  (h2 : train_speed = 45 / 3.6) -- Convert 45 km/hr to m/s
  (h3 : train_length = 100)
  (h4 : initial_distance = 240) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 34 := by
sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l193_19302


namespace NUMINAMATH_CALUDE_difference_calculation_l193_19376

theorem difference_calculation : 
  (1 / 10 : ℚ) * 8000 - (1 / 20 : ℚ) / 100 * 8000 = 796 := by
  sorry

end NUMINAMATH_CALUDE_difference_calculation_l193_19376


namespace NUMINAMATH_CALUDE_equation_solutions_l193_19391

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (1/4 + Real.sqrt 17 / 4) ∧ x₂ = (1/4 - Real.sqrt 17 / 4) ∧
    2 * x₁^2 - 2 = x₁ ∧ 2 * x₂^2 - 2 = x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 2 ∧
    x₁ * (x₁ - 2) + x₁ - 2 = 0 ∧ x₂ * (x₂ - 2) + x₂ - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l193_19391


namespace NUMINAMATH_CALUDE_min_MN_length_l193_19322

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the circle containing point P -/
def circle_P (x y : ℝ) : Prop :=
  x^2 + y^2 = 16

/-- Theorem statement -/
theorem min_MN_length (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ellipse_C 0 1 a b) -- vertex at (0,1)
  (h4 : (a^2 - b^2) / a^2 = 3/4) -- eccentricity is √3/2
  : ∃ (x_p y_p x_m y_n : ℝ),
    circle_P x_p y_p ∧
    (∀ (x_a y_a x_b y_b : ℝ),
      ellipse_C x_a y_a a b →
      ellipse_C x_b y_b a b →
      (y_n - y_a) * (x_p - x_a) = (y_p - y_a) * (x_m - x_a) →
      (y_n - y_b) * (x_p - x_b) = (y_p - y_b) * (x_m - x_b) →
      x_m = 0 ∧ y_n = 0) →
    (x_m - 0)^2 + (0 - y_n)^2 ≥ (5/4)^2 :=
by sorry

end NUMINAMATH_CALUDE_min_MN_length_l193_19322


namespace NUMINAMATH_CALUDE_student_weight_l193_19307

theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight - 5 = 2 * sister_weight)
  (h2 : student_weight + sister_weight = 110) :
  student_weight = 75 := by
sorry

end NUMINAMATH_CALUDE_student_weight_l193_19307


namespace NUMINAMATH_CALUDE_rachel_homework_l193_19377

theorem rachel_homework (math_pages reading_pages : ℕ) : 
  math_pages = 3 → reading_pages = math_pages + 1 → reading_pages = 4 := by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_l193_19377


namespace NUMINAMATH_CALUDE_millet_exceeds_half_on_wednesday_l193_19355

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  millet : Real
  otherSeeds : Real

/-- Calculates the next day's feeder state based on the current state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    millet := 0.7 * state.millet + 0.2,
    otherSeeds := 0.1 * state.otherSeeds + 0.3 }

/-- Initial state of the feeder on Monday -/
def initialState : FeederState :=
  { day := 1, millet := 0.2, otherSeeds := 0.3 }

/-- Theorem stating that on Wednesday, the proportion of millet exceeds half of the total seeds -/
theorem millet_exceeds_half_on_wednesday :
  let wednesdayState := nextDay (nextDay initialState)
  wednesdayState.millet > (wednesdayState.millet + wednesdayState.otherSeeds) / 2 :=
by sorry


end NUMINAMATH_CALUDE_millet_exceeds_half_on_wednesday_l193_19355


namespace NUMINAMATH_CALUDE_multi_painted_cubes_count_l193_19383

/-- Represents a cube with a given side length -/
structure Cube where
  side_length : ℕ

/-- Represents a painted cube with a given number of painted faces -/
structure PaintedCube extends Cube where
  painted_faces : ℕ

/-- Counts the number of small cubes with paint on at least two faces
    when a larger cube is cut into unit cubes -/
def count_multi_painted_cubes (c : PaintedCube) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem multi_painted_cubes_count :
  let large_cube : PaintedCube := { side_length := 4, painted_faces := 3 }
  count_multi_painted_cubes large_cube = 16 :=
by sorry

end NUMINAMATH_CALUDE_multi_painted_cubes_count_l193_19383


namespace NUMINAMATH_CALUDE_liters_to_pints_conversion_l193_19316

/-- Given that 0.33 liters is approximately 0.7 pints, prove that one liter is approximately 2.1 pints. -/
theorem liters_to_pints_conversion (ε : ℝ) (h_ε : ε > 0) :
  ∃ (δ : ℝ), δ > 0 ∧ 
  ∀ (x y : ℝ), 
    (abs (x - 0.33) < δ ∧ abs (y - 0.7) < δ) → 
    abs ((1 / x * y) - 2.1) < ε :=
sorry

end NUMINAMATH_CALUDE_liters_to_pints_conversion_l193_19316


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l193_19305

theorem absolute_value_inequality_solution :
  {x : ℝ | |2 - x| ≤ 1} = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l193_19305


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l193_19371

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 = 20) →
  (a 3 + a 4 = 40) →
  (a 5 + a 6 = 80) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l193_19371


namespace NUMINAMATH_CALUDE_exponential_function_property_l193_19303

theorem exponential_function_property (a b : ℝ) :
  let f : ℝ → ℝ := fun x ↦ Real.exp x
  f (a + b) = 2 → f (2 * a) * f (2 * b) = 4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_property_l193_19303


namespace NUMINAMATH_CALUDE_water_percentage_in_fresh_mushrooms_l193_19395

theorem water_percentage_in_fresh_mushrooms 
  (fresh_mass : ℝ) 
  (dried_mass : ℝ) 
  (dried_water_percentage : ℝ) 
  (h1 : fresh_mass = 22) 
  (h2 : dried_mass = 2.5) 
  (h3 : dried_water_percentage = 12) : 
  (fresh_mass - dried_mass * (1 - dried_water_percentage / 100)) / fresh_mass * 100 = 90 := by
sorry

end NUMINAMATH_CALUDE_water_percentage_in_fresh_mushrooms_l193_19395


namespace NUMINAMATH_CALUDE_range_of_x_l193_19336

def is_meaningful (x : ℝ) : Prop := x ≠ 5

theorem range_of_x : ∀ x : ℝ, is_meaningful x ↔ x ≠ 5 := by sorry

end NUMINAMATH_CALUDE_range_of_x_l193_19336


namespace NUMINAMATH_CALUDE_intersection_y_intercept_l193_19393

/-- Given two lines that intersect at a specific x-coordinate, 
    prove that the y-intercept of the first line has a specific value. -/
theorem intersection_y_intercept (k : ℝ) : 
  (∃ y : ℝ, -3 * (-6.8) + y = k ∧ 0.25 * (-6.8) + y = 10) → k = 32.1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_l193_19393


namespace NUMINAMATH_CALUDE_interest_rate_for_doubling_l193_19335

/-- Represents the number of years required for the principal to double. -/
def years_to_double : ℝ := 10

/-- Theorem stating that if a principal doubles in 10 years due to simple interest,
    then the rate of interest is 10% per annum. -/
theorem interest_rate_for_doubling (P : ℝ) (P_pos : P > 0) :
  ∃ R : ℝ, R > 0 ∧ P + (P * R * years_to_double / 100) = 2 * P ∧ R = 10 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_for_doubling_l193_19335


namespace NUMINAMATH_CALUDE_count_negative_numbers_l193_19399

theorem count_negative_numbers : let numbers := [-3^2, (-1)^2006, 0, |(-2)|, -(-2), -3 * 2^2]
  (numbers.filter (· < 0)).length = 2 := by sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l193_19399


namespace NUMINAMATH_CALUDE_infinite_parallel_lines_l193_19323

/-- A plane in 3D space -/
structure Plane3D where
  -- Define the plane (implementation details omitted)

/-- A point in 3D space -/
structure Point3D where
  -- Define the point (implementation details omitted)

/-- A line in 3D space -/
structure Line3D where
  -- Define the line (implementation details omitted)

/-- Predicate to check if a point is not on a plane -/
def notOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line is parallel to a plane -/
def isParallelToPlane (l : Line3D) (plane : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line passes through a point -/
def passesThroughPoint (l : Line3D) (p : Point3D) : Prop :=
  sorry

/-- The main theorem -/
theorem infinite_parallel_lines
  (plane : Plane3D) (p : Point3D) (h : notOnPlane p plane) :
  ∃ (s : Set Line3D), (∀ l ∈ s, isParallelToPlane l plane ∧ passesThroughPoint l p) ∧ Set.Infinite s :=
sorry

end NUMINAMATH_CALUDE_infinite_parallel_lines_l193_19323


namespace NUMINAMATH_CALUDE_basketball_win_calculation_l193_19384

/-- Given a basketball team's performance, calculate the number of games they need to win to achieve a specific win rate -/
theorem basketball_win_calculation 
  (total_games : ℕ) 
  (first_segment_games : ℕ) 
  (games_won_first_segment : ℕ) 
  (target_win_rate : ℚ) :
  total_games = 130 →
  first_segment_games = 75 →
  games_won_first_segment = 60 →
  target_win_rate = 4/5 →
  ∃ (games_to_win : ℕ), 
    games_to_win = 44 ∧ 
    (games_won_first_segment + games_to_win : ℚ) / total_games = target_win_rate :=
by sorry

end NUMINAMATH_CALUDE_basketball_win_calculation_l193_19384


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l193_19396

theorem book_sale_loss_percentage 
  (total_cost : ℝ) 
  (cost_book1 : ℝ) 
  (gain_percentage : ℝ) :
  total_cost = 300 →
  cost_book1 = 175 →
  gain_percentage = 19 →
  let cost_book2 := total_cost - cost_book1
  let selling_price := cost_book2 * (1 + gain_percentage / 100)
  let loss_amount := cost_book1 - selling_price
  let loss_percentage := (loss_amount / cost_book1) * 100
  loss_percentage = 15 := by sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l193_19396


namespace NUMINAMATH_CALUDE_train_vs_airplanes_capacity_difference_l193_19370

/-- The number of passengers a single train car can carry -/
def train_car_capacity : ℕ := 60

/-- The number of passengers a 747 airplane can carry -/
def airplane_capacity : ℕ := 366

/-- The number of cars in the train -/
def train_cars : ℕ := 16

/-- The number of airplanes being compared -/
def num_airplanes : ℕ := 2

/-- Theorem stating the difference in passenger capacity between the train and the airplanes -/
theorem train_vs_airplanes_capacity_difference :
  train_cars * train_car_capacity - num_airplanes * airplane_capacity = 228 := by
  sorry

end NUMINAMATH_CALUDE_train_vs_airplanes_capacity_difference_l193_19370


namespace NUMINAMATH_CALUDE_triangle_probability_l193_19364

def stickLengths : List ℕ := [1, 2, 4, 6, 9, 10, 14, 15, 18]

def canFormTriangle (a b c : ℕ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

def validTriangleCombinations : List (ℕ × ℕ × ℕ) := 
  [(4, 6, 9), (4, 9, 10), (4, 9, 14), (4, 10, 14), (4, 14, 15),
   (6, 9, 10), (6, 9, 14), (6, 10, 14), (6, 14, 15), (6, 9, 15), (6, 10, 15),
   (9, 10, 14), (9, 14, 15), (9, 10, 15),
   (10, 14, 15)]

def totalCombinations : ℕ := Nat.choose 9 3

theorem triangle_probability : 
  (validTriangleCombinations.length : ℚ) / totalCombinations = 4 / 21 := by
  sorry

end NUMINAMATH_CALUDE_triangle_probability_l193_19364


namespace NUMINAMATH_CALUDE_license_plate_ratio_l193_19388

/-- The number of possible letters in a license plate -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate -/
def num_digits : ℕ := 10

/-- The number of letters in an old license plate -/
def old_letters : ℕ := 2

/-- The number of digits in an old license plate -/
def old_digits : ℕ := 3

/-- The number of letters in a new license plate -/
def new_letters : ℕ := 3

/-- The number of digits in a new license plate -/
def new_digits : ℕ := 4

/-- The ratio of new license plates to old license plates -/
theorem license_plate_ratio :
  (num_letters ^ new_letters * num_digits ^ new_digits) /
  (num_letters ^ old_letters * num_digits ^ old_digits) = 260 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_ratio_l193_19388


namespace NUMINAMATH_CALUDE_star_five_three_l193_19328

-- Define the ※ operation
def star (a b : ℝ) : ℝ := b^2 + 1

-- Theorem statement
theorem star_five_three : star 5 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_star_five_three_l193_19328


namespace NUMINAMATH_CALUDE_alcohol_quantity_l193_19347

/-- Proves that the quantity of alcohol is 16 liters given the initial and final ratios -/
theorem alcohol_quantity (initial_alcohol : ℚ) (initial_water : ℚ) (final_water : ℚ) :
  initial_alcohol / initial_water = 4 / 3 →
  initial_alcohol / (initial_water + 8) = 4 / 5 →
  initial_alcohol = 16 := by
sorry


end NUMINAMATH_CALUDE_alcohol_quantity_l193_19347


namespace NUMINAMATH_CALUDE_aras_height_l193_19327

theorem aras_height (original_height : ℝ) (sheas_growth_rate : ℝ) (aras_growth_fraction : ℝ) (sheas_final_height : ℝ) :
  original_height > 0 ∧
  sheas_growth_rate = 0.25 ∧
  aras_growth_fraction = 1/3 ∧
  sheas_final_height = 70 ∧
  sheas_final_height = original_height * (1 + sheas_growth_rate) →
  original_height + (sheas_final_height - original_height) * aras_growth_fraction = 60.67 := by
  sorry

#check aras_height

end NUMINAMATH_CALUDE_aras_height_l193_19327


namespace NUMINAMATH_CALUDE_smallest_sum_ABAb_l193_19389

/-- Represents a digit in base 4 -/
def Base4Digit := Fin 4

theorem smallest_sum_ABAb (A B : Base4Digit) (b : ℕ) : 
  A ≠ B →
  b > 5 →
  16 * A.val + 4 * B.val + A.val = 3 * b + 3 →
  ∀ (A' B' : Base4Digit) (b' : ℕ),
    A' ≠ B' →
    b' > 5 →
    16 * A'.val + 4 * B'.val + A'.val = 3 * b' + 3 →
    A.val + B.val + b ≤ A'.val + B'.val + b' →
  A.val + B.val + b = 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_ABAb_l193_19389


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l193_19353

theorem solution_set_quadratic_inequality (x : ℝ) :
  x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l193_19353


namespace NUMINAMATH_CALUDE_divisibility_statement_l193_19314

theorem divisibility_statement (a : ℤ) :
  (∃! n : Fin 4, ¬ (
    (n = 0 → a % 2 = 0) ∧
    (n = 1 → a % 4 = 0) ∧
    (n = 2 → a % 12 = 0) ∧
    (n = 3 → a % 24 = 0)
  )) →
  ¬(a % 24 = 0) :=
by sorry


end NUMINAMATH_CALUDE_divisibility_statement_l193_19314


namespace NUMINAMATH_CALUDE_spheres_fit_in_box_l193_19349

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the maximum number of spheres that can fit in a box using a specific packing method -/
noncomputable def maxSpheres (box : BoxDimensions) (sphereDiameter : ℝ) : ℕ :=
  sorry

/-- Theorem stating that 100,000 spheres of 4 cm diameter can fit in the given box -/
theorem spheres_fit_in_box :
  let box : BoxDimensions := ⟨200, 164, 146⟩
  let sphereDiameter : ℝ := 4
  maxSpheres box sphereDiameter ≥ 100000 := by
  sorry

end NUMINAMATH_CALUDE_spheres_fit_in_box_l193_19349


namespace NUMINAMATH_CALUDE_five_students_three_communities_l193_19350

/-- The number of ways to assign students to communities -/
def assign_students (n : ℕ) (k : ℕ) : ℕ :=
  -- Number of ways to assign n students to k communities
  -- with at least 1 student in each community
  sorry

/-- Theorem: 5 students assigned to 3 communities results in 150 ways -/
theorem five_students_three_communities :
  assign_students 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_students_three_communities_l193_19350


namespace NUMINAMATH_CALUDE_smallest_six_digit_multiple_of_1379_l193_19358

theorem smallest_six_digit_multiple_of_1379 : 
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 1379 = 0 → n ≥ 100657 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_six_digit_multiple_of_1379_l193_19358


namespace NUMINAMATH_CALUDE_line_points_determine_m_l193_19306

-- Define the points on the line
def point1 : ℝ × ℝ := (7, 10)
def point2 : ℝ → ℝ × ℝ := λ m ↦ (-3, m)
def point3 : ℝ × ℝ := (-11, 5)

-- Define the condition that the points are collinear
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1)

-- Theorem statement
theorem line_points_determine_m :
  collinear point1 (point2 m) point3 → m = 65 / 9 := by
  sorry

end NUMINAMATH_CALUDE_line_points_determine_m_l193_19306


namespace NUMINAMATH_CALUDE_min_value_theorem_l193_19301

theorem min_value_theorem (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + 2*n = 1) :
  (1 / (2*m)) + (1 / n) ≥ 9/2 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ m₀ + 2*n₀ = 1 ∧ (1 / (2*m₀)) + (1 / n₀) = 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l193_19301


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_product_l193_19309

theorem root_sum_reciprocal_product (p q r s : ℂ) : 
  (p^4 + 6*p^3 + 13*p^2 + 7*p + 3 = 0) →
  (q^4 + 6*q^3 + 13*q^2 + 7*q + 3 = 0) →
  (r^4 + 6*r^3 + 13*r^2 + 7*r + 3 = 0) →
  (s^4 + 6*s^3 + 13*s^2 + 7*s + 3 = 0) →
  (1 / (p*q*r) + 1 / (p*q*s) + 1 / (p*r*s) + 1 / (q*r*s) = -2) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_product_l193_19309


namespace NUMINAMATH_CALUDE_factorization_equality_l193_19310

theorem factorization_equality (a b : ℝ) : a * b^2 - 2 * a * b + a = a * (b - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l193_19310


namespace NUMINAMATH_CALUDE_sandy_comic_books_l193_19378

theorem sandy_comic_books (x : ℕ) : 
  (x / 2 : ℚ) - 3 + 6 = 13 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_sandy_comic_books_l193_19378


namespace NUMINAMATH_CALUDE_vector_subtraction_l193_19354

/-- Given two vectors AB and AC in a plane, prove that vector BC is their difference. -/
theorem vector_subtraction (AB AC : ℝ × ℝ) (h1 : AB = (3, 4)) (h2 : AC = (1, 3)) :
  AC - AB = (-2, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l193_19354


namespace NUMINAMATH_CALUDE_annie_cookie_ratio_l193_19375

-- Define the number of cookies eaten on each day
def monday_cookies : ℕ := 5
def tuesday_cookies : ℕ := 10  -- We know this from the solution, but it's not given in the problem
def wednesday_cookies : ℕ := (tuesday_cookies * 140) / 100

-- Define the total number of cookies eaten
def total_cookies : ℕ := 29

-- State the theorem
theorem annie_cookie_ratio :
  monday_cookies + tuesday_cookies + wednesday_cookies = total_cookies ∧
  wednesday_cookies = (tuesday_cookies * 140) / 100 ∧
  tuesday_cookies / monday_cookies = 2 := by
sorry

end NUMINAMATH_CALUDE_annie_cookie_ratio_l193_19375


namespace NUMINAMATH_CALUDE_circumscribed_circle_radius_of_specific_trapezoid_l193_19315

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ
  isIsosceles : True

/-- The radius of the circumscribed circle of an isosceles trapezoid -/
def circumscribedCircleRadius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- The theorem stating that the radius of the circumscribed circle of the given isosceles trapezoid is 10.625 -/
theorem circumscribed_circle_radius_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := {
    base1 := 9,
    base2 := 21,
    height := 8,
    isIsosceles := True.intro
  }
  circumscribedCircleRadius t = 10.625 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_radius_of_specific_trapezoid_l193_19315


namespace NUMINAMATH_CALUDE_chemical_plant_max_profit_l193_19351

/-- Represents the annual profit function for a chemical plant. -/
def L (x a : ℝ) : ℝ := (x - 3 - a) * (11 - x)^2

/-- Proves the maximum annual profit for the chemical plant under given conditions. -/
theorem chemical_plant_max_profit :
  ∀ (a : ℝ), 1 ≤ a → a ≤ 3 →
    (∀ (x : ℝ), 7 ≤ x → x ≤ 10 →
      (1 ≤ a ∧ a ≤ 2 →
        L x a ≤ 16 * (4 - a) ∧
        L 7 a = 16 * (4 - a)) ∧
      (2 < a →
        L x a ≤ (8 - a)^3 ∧
        L ((17 + 2*a)/3) a = (8 - a)^3)) :=
by sorry

end NUMINAMATH_CALUDE_chemical_plant_max_profit_l193_19351


namespace NUMINAMATH_CALUDE_original_number_proof_l193_19337

theorem original_number_proof : ∃ N : ℕ, 
  (N > 30) ∧ 
  (N - 30) % 87 = 0 ∧ 
  (∀ M : ℕ, M > 30 ∧ (M - 30) % 87 = 0 → M ≥ N) :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l193_19337
