import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_range_l2713_271377

theorem quadratic_inequality_implies_range (x : ℝ) :
  x^2 - 5*x + 4 < 0 → 10 < x^2 + 4*x + 5 ∧ x^2 + 4*x + 5 < 37 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_range_l2713_271377


namespace NUMINAMATH_CALUDE_unique_fraction_decomposition_l2713_271367

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (n m : ℕ), n ≠ m ∧ 2 / p = 1 / n + 1 / m ∧
  ((n = (p + 1) / 2 ∧ m = p * (p + 1) / 2) ∨
   (m = (p + 1) / 2 ∧ n = p * (p + 1) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_unique_fraction_decomposition_l2713_271367


namespace NUMINAMATH_CALUDE_gary_book_multiple_l2713_271311

/-- Proves that Gary's books are 5 times the combined number of Darla's and Katie's books -/
theorem gary_book_multiple (darla_books katie_books gary_books : ℕ) : 
  darla_books = 6 →
  katie_books = darla_books / 2 →
  gary_books = (darla_books + katie_books) * (gary_books / (darla_books + katie_books)) →
  darla_books + katie_books + gary_books = 54 →
  gary_books / (darla_books + katie_books) = 5 := by
sorry

end NUMINAMATH_CALUDE_gary_book_multiple_l2713_271311


namespace NUMINAMATH_CALUDE_det_A_equals_two_l2713_271387

open Matrix

theorem det_A_equals_two (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A + 2 * A⁻¹ = 0) : 
  det A = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_A_equals_two_l2713_271387


namespace NUMINAMATH_CALUDE_opposite_sign_pairs_l2713_271381

theorem opposite_sign_pairs : 
  ¬((-2^3) * ((-2)^3) < 0) ∧
  ¬((|-4|) * (-(-4)) < 0) ∧
  ((-3^4) * ((-3)^4) < 0) ∧
  ¬((10^2) * (2^10) < 0) := by
sorry

end NUMINAMATH_CALUDE_opposite_sign_pairs_l2713_271381


namespace NUMINAMATH_CALUDE_no_real_solution_condition_l2713_271391

theorem no_real_solution_condition (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, a^x ≠ x) ↔ a > Real.exp (1 / Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_condition_l2713_271391


namespace NUMINAMATH_CALUDE_max_m_inequality_l2713_271309

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ m : ℝ, ∀ a b : ℝ, a > 0 → b > 0 → 4/a + 1/b ≥ m/(a+4*b)) ∧
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → 4/a + 1/b ≥ m/(a+4*b)) → m ≤ 16) :=
sorry

end NUMINAMATH_CALUDE_max_m_inequality_l2713_271309


namespace NUMINAMATH_CALUDE_intersection_nonempty_intersection_equals_A_l2713_271373

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x > 5}

-- Theorem 1
theorem intersection_nonempty (a : ℝ) : 
  (A a ∩ B).Nonempty ↔ a < -1 ∨ a > 2 := by sorry

-- Theorem 2
theorem intersection_equals_A (a : ℝ) : 
  A a ∩ B = A a ↔ a < -4 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_intersection_equals_A_l2713_271373


namespace NUMINAMATH_CALUDE_unique_solution_for_k_l2713_271384

/-- The equation has exactly one solution when k = -3/4 -/
theorem unique_solution_for_k (k : ℝ) : 
  (∃! x, (x + 3) / (k * x - 2) = x) ↔ k = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_k_l2713_271384


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l2713_271305

/-- Calculates the profit percentage for a merchant who marks up goods by 50%
    and then offers a 10% discount on the marked price. -/
theorem merchant_profit_percentage (cost_price : ℝ) (cost_price_pos : 0 < cost_price) :
  let markup_percentage : ℝ := 0.5
  let discount_percentage : ℝ := 0.1
  let marked_price : ℝ := cost_price * (1 + markup_percentage)
  let selling_price : ℝ := marked_price * (1 - discount_percentage)
  let profit : ℝ := selling_price - cost_price
  let profit_percentage : ℝ := profit / cost_price * 100
  profit_percentage = 35 := by
  sorry


end NUMINAMATH_CALUDE_merchant_profit_percentage_l2713_271305


namespace NUMINAMATH_CALUDE_journey_distance_is_25_l2713_271300

/-- Represents a segment of the journey with speed and duration -/
structure Segment where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance covered in a segment -/
def distance_covered (s : Segment) : ℝ := s.speed * s.duration

/-- The journey segments -/
def journey_segments : List Segment := [
  ⟨4, 1⟩,
  ⟨5, 0.5⟩,
  ⟨3, 0.75⟩,
  ⟨2, 0.5⟩,
  ⟨6, 0.5⟩,
  ⟨7, 0.25⟩,
  ⟨4, 1.5⟩,
  ⟨6, 0.75⟩
]

/-- The total distance covered during the journey -/
def total_distance : ℝ := (journey_segments.map distance_covered).sum

theorem journey_distance_is_25 : total_distance = 25 := by sorry

end NUMINAMATH_CALUDE_journey_distance_is_25_l2713_271300


namespace NUMINAMATH_CALUDE_min_value_of_y_l2713_271332

theorem min_value_of_y (x : ℝ) (h : x > 3) : x + 1 / (x - 3) ≥ 5 ∧ ∃ x₀ > 3, x₀ + 1 / (x₀ - 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_y_l2713_271332


namespace NUMINAMATH_CALUDE_reciprocal_counterexample_l2713_271360

theorem reciprocal_counterexample : ∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x > y ∧ (1 / x) > (1 / y) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_counterexample_l2713_271360


namespace NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l2713_271312

-- Part 1
theorem simplify_fraction_1 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (3 * a^2 * b) / (6 * a * b^2 * c) = a / (2 * b * c) := by sorry

-- Part 2
theorem simplify_fraction_2 (x y : ℝ) (h : x ≠ y) :
  (2 * (x - y)^3) / (y - x) = -2 * (x - y)^2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l2713_271312


namespace NUMINAMATH_CALUDE_car_speed_problem_l2713_271355

/-- Proves that the speed of Car A is 70 km/h given the conditions of the problem -/
theorem car_speed_problem (time : ℝ) (speed_B : ℝ) (ratio : ℝ) :
  time = 10 →
  speed_B = 35 →
  ratio = 2 →
  let distance_A := time * (ratio * speed_B)
  let distance_B := time * speed_B
  (distance_A / distance_B = ratio) →
  (ratio * speed_B = 70) :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2713_271355


namespace NUMINAMATH_CALUDE_garland_theorem_l2713_271320

/-- The number of ways to arrange light bulbs in a garland -/
def garland_arrangements (blue red white : ℕ) : ℕ :=
  Nat.choose (blue + red + 1) white * Nat.choose (blue + red) blue

/-- Theorem: The number of ways to arrange 9 blue, 7 red, and 14 white light bulbs
    in a garland, such that no two white light bulbs are adjacent, is 7,779,200 -/
theorem garland_theorem :
  garland_arrangements 9 7 14 = 7779200 := by
  sorry

#eval garland_arrangements 9 7 14

end NUMINAMATH_CALUDE_garland_theorem_l2713_271320


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2713_271318

def A : Set ℝ := {x | x ≥ -4}
def B : Set ℝ := {x | x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {x | -4 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2713_271318


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1023_l2713_271325

theorem largest_prime_factor_of_1023 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1023 ∧ ∀ q, Nat.Prime q → q ∣ 1023 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1023_l2713_271325


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l2713_271337

theorem magnitude_of_complex_power : 
  Complex.abs ((5 : ℂ) - (2 * Real.sqrt 3) * Complex.I) ^ 4 = 1369 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l2713_271337


namespace NUMINAMATH_CALUDE_least_sum_of_exponential_equality_l2713_271371

theorem least_sum_of_exponential_equality (x y z : ℕ+) 
  (h : (2 : ℕ)^(x : ℕ) = (5 : ℕ)^(y : ℕ) ∧ (5 : ℕ)^(y : ℕ) = (8 : ℕ)^(z : ℕ)) : 
  (∀ a b c : ℕ+, (2 : ℕ)^(a : ℕ) = (5 : ℕ)^(b : ℕ) ∧ (5 : ℕ)^(b : ℕ) = (8 : ℕ)^(c : ℕ) → 
    (x : ℕ) + (y : ℕ) + (z : ℕ) ≤ (a : ℕ) + (b : ℕ) + (c : ℕ)) ∧
  (x : ℕ) + (y : ℕ) + (z : ℕ) = 33 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_exponential_equality_l2713_271371


namespace NUMINAMATH_CALUDE_max_true_statements_l2713_271327

theorem max_true_statements (x : ℝ) : 
  let statements := [
    (0 < x^2 ∧ x^2 < 1),
    (x^2 > 1),
    (-1 < x ∧ x < 0),
    (0 < x ∧ x < 1),
    (0 < x - Real.sqrt x ∧ x - Real.sqrt x < 1)
  ]
  ¬∃ (s : Finset (Fin 5)), s.card > 3 ∧ (∀ i ∈ s, statements[i.val]) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l2713_271327


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l2713_271319

/-- Given two concentric circles with radii R and r, where the area of the ring between them is 18π,
    the length of a chord of the larger circle that is tangent to the smaller circle is 6√2. -/
theorem chord_length_concentric_circles (R r : ℝ) (h : R > r) :
  (π * R^2 - π * r^2 = 18 * π) →
  ∃ c : ℝ, c = 6 * Real.sqrt 2 ∧ c^2 = 4 * (R^2 - r^2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l2713_271319


namespace NUMINAMATH_CALUDE_mario_expected_doors_l2713_271374

/-- The expected number of doors Mario will pass before reaching Bowser's level -/
def expected_doors (d r : ℕ) : ℚ :=
  (d * (d^r - 1)) / (d - 1)

/-- Theorem stating the expected number of doors Mario will pass -/
theorem mario_expected_doors (d r : ℕ) (hd : d > 1) (hr : r > 0) :
  let E := expected_doors d r
  ∀ k : ℕ, k ≤ r → 
    (∃ Ek : ℚ, Ek = E ∧ 
      Ek = 1 + (d - 1) / d * E + 1 / d * expected_doors d (r - k)) :=
by sorry

end NUMINAMATH_CALUDE_mario_expected_doors_l2713_271374


namespace NUMINAMATH_CALUDE_kids_joined_in_l2713_271350

theorem kids_joined_in (initial_kids final_kids : ℕ) (h : initial_kids = 14 ∧ final_kids = 36) :
  final_kids - initial_kids = 22 := by
  sorry

end NUMINAMATH_CALUDE_kids_joined_in_l2713_271350


namespace NUMINAMATH_CALUDE_inequality_solution_l2713_271370

/-- The numerator of the inequality -/
def numerator (x : ℝ) : ℝ := |3*x^2 + 8*x - 3| + |3*x^4 + 2*x^3 - 10*x^2 + 30*x - 9|

/-- The denominator of the inequality -/
def denominator (x : ℝ) : ℝ := |x-2| - 2*x - 1

/-- The inequality function -/
def inequality (x : ℝ) : Prop := numerator x / denominator x ≤ 0

/-- The solution set of the inequality -/
def solution_set : Set ℝ := {x | x < 1/3 ∨ x > 1/3}

theorem inequality_solution : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2713_271370


namespace NUMINAMATH_CALUDE_smallest_possible_b_l2713_271351

theorem smallest_possible_b : ∃ (b : ℝ), b = 2 ∧ 
  (∀ (a : ℝ), (2 < a ∧ a < b) → 
    (2 + a ≤ b ∧ 1/a + 1/b ≤ 1)) ∧
  (∀ (b' : ℝ), 2 < b' → 
    (∃ (a : ℝ), (2 < a ∧ a < b') ∧ 
      (2 + a > b' ∨ 1/a + 1/b' > 1))) :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l2713_271351


namespace NUMINAMATH_CALUDE_unique_integer_sum_l2713_271303

theorem unique_integer_sum (y : ℝ) : 
  y = Real.sqrt ((Real.sqrt 77) / 2 + 5 / 2) →
  ∃! (d e f : ℕ+), 
    y^100 = 2*y^98 + 18*y^96 + 15*y^94 - y^50 + (d:ℝ)*y^46 + (e:ℝ)*y^44 + (f:ℝ)*y^40 ∧
    d + e + f = 242 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_sum_l2713_271303


namespace NUMINAMATH_CALUDE_house_cost_l2713_271389

theorem house_cost (total : ℝ) (interest_rate1 : ℝ) (interest_rate2 : ℝ) (total_interest : ℝ) 
  (h1 : total = 120000)
  (h2 : interest_rate1 = 0.04)
  (h3 : interest_rate2 = 0.05)
  (h4 : total_interest = 3920) :
  ∃ (house_cost : ℝ),
    house_cost = 36000 ∧
    (1/3 * (total - house_cost) * interest_rate1 + 2/3 * (total - house_cost) * interest_rate2 = total_interest) :=
by
  sorry

end NUMINAMATH_CALUDE_house_cost_l2713_271389


namespace NUMINAMATH_CALUDE_base_conversion_equality_l2713_271358

theorem base_conversion_equality (b : ℝ) : b > 0 → (4 * 5 + 3 = b^2 + 2) → b = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l2713_271358


namespace NUMINAMATH_CALUDE_min_value_fraction_l2713_271335

theorem min_value_fraction (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2713_271335


namespace NUMINAMATH_CALUDE_triangular_grid_4_has_17_triangles_l2713_271395

/-- Represents a triangular grid with n rows -/
structure TriangularGrid (n : ℕ) where
  rows : Fin n → ℕ
  row_content : ∀ i : Fin n, rows i = i.val + 1

/-- Counts the number of triangles in a triangular grid -/
def count_triangles (grid : TriangularGrid n) : ℕ :=
  sorry

theorem triangular_grid_4_has_17_triangles :
  ∃ (grid : TriangularGrid 4), count_triangles grid = 17 :=
sorry

end NUMINAMATH_CALUDE_triangular_grid_4_has_17_triangles_l2713_271395


namespace NUMINAMATH_CALUDE_soap_survey_households_l2713_271326

theorem soap_survey_households (total : ℕ) (neither : ℕ) (only_A : ℕ) (both : ℕ) :
  total = 160 ∧
  neither = 80 ∧
  only_A = 60 ∧
  both = 5 →
  total = neither + only_A + both + 3 * both :=
by sorry

end NUMINAMATH_CALUDE_soap_survey_households_l2713_271326


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2713_271339

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, 2 * x^2 - 8 * x + c < 0) ↔ (0 < c ∧ c < 8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2713_271339


namespace NUMINAMATH_CALUDE_max_min_product_l2713_271331

theorem max_min_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (sum_eq : a + b + c = 12) (sum_prod_eq : a * b + b * c + c * a = 32) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 4 ∧
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_max_min_product_l2713_271331


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2713_271385

theorem rectangle_perimeter (w : ℝ) (h1 : w > 0) :
  let l := 3 * w
  let d := 8 * Real.sqrt 10
  d^2 = l^2 + w^2 →
  2 * l + 2 * w = 64 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2713_271385


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l2713_271321

-- Define the total number of votes
def total_votes : ℕ := 7600

-- Define the difference in votes between the winner and loser
def vote_difference : ℕ := 2280

-- Define the percentage of votes received by the losing candidate
def losing_candidate_percentage : ℚ := 35

-- Theorem statement
theorem candidate_vote_percentage :
  (2 * losing_candidate_percentage * total_votes : ℚ) = 
  (100 * (total_votes - vote_difference) : ℚ) := by sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l2713_271321


namespace NUMINAMATH_CALUDE_double_counted_page_number_l2713_271348

theorem double_counted_page_number :
  ∃! (n : ℕ) (x : ℕ), 
    1 ≤ x ∧ 
    x ≤ n ∧ 
    n * (n + 1) / 2 + x = 2550 ∧ 
    x = 65 := by
  sorry

end NUMINAMATH_CALUDE_double_counted_page_number_l2713_271348


namespace NUMINAMATH_CALUDE_olympic_medal_scenario_l2713_271342

/-- The number of ways to award medals in the Olympic 100-meter sprint -/
def olympic_medal_ways (total_athletes : ℕ) (european_athletes : ℕ) (asian_athletes : ℕ) (max_european_medals : ℕ) : ℕ :=
  -- Define the function here
  sorry

/-- Theorem: The number of ways to award medals in the given Olympic scenario is 588 -/
theorem olympic_medal_scenario : olympic_medal_ways 10 4 6 2 = 588 := by
  sorry

end NUMINAMATH_CALUDE_olympic_medal_scenario_l2713_271342


namespace NUMINAMATH_CALUDE_infinite_product_of_a_l2713_271376

noncomputable def a : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 1 + (a n - 1)^3

theorem infinite_product_of_a : ∏' n, a n = 3/5 := by sorry

end NUMINAMATH_CALUDE_infinite_product_of_a_l2713_271376


namespace NUMINAMATH_CALUDE_triangle_inequality_l2713_271369

-- Define a structure for a triangle
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z
  triangle_ineq : x + y > z ∧ y + z > x ∧ z + x > y

-- State the theorem
theorem triangle_inequality (t : Triangle) :
  |t.x^2 * (t.y - t.z) + t.y^2 * (t.z - t.x) + t.z^2 * (t.x - t.y)| < t.x * t.y * t.z :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2713_271369


namespace NUMINAMATH_CALUDE_calculation_proofs_l2713_271383

theorem calculation_proofs :
  (∃ (x : ℝ), x = (1/2 * Real.sqrt 24 - 2 * Real.sqrt 2 * Real.sqrt 3) ∧ x = -Real.sqrt 6) ∧
  (∃ (y : ℝ), y = ((Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) + Real.sqrt 8 - Real.sqrt (9/2)) ∧ y = -1 + Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proofs_l2713_271383


namespace NUMINAMATH_CALUDE_circle_radius_is_five_thirds_l2713_271380

/-- An isosceles triangle with a circle constructed on its base -/
structure IsoscelesTriangleWithCircle where
  /-- The base length of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the circle constructed on the base -/
  radius : ℝ

/-- The radius of the circle in an isosceles triangle with given base and height -/
def circleRadius (triangle : IsoscelesTriangleWithCircle) : ℝ :=
  triangle.radius

/-- Theorem: The radius of the circle is 5/3 given the specified conditions -/
theorem circle_radius_is_five_thirds (triangle : IsoscelesTriangleWithCircle)
    (h1 : triangle.base = 8)
    (h2 : triangle.height = 3) :
    circleRadius triangle = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_five_thirds_l2713_271380


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l2713_271353

theorem unique_integer_divisible_by_14_with_sqrt_between_25_and_25_3 :
  ∃! n : ℕ+, 14 ∣ n ∧ 25 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 25.3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l2713_271353


namespace NUMINAMATH_CALUDE_smallest_abc_cba_divisible_by_11_l2713_271364

/-- Represents a six-digit number in the form ABC,CBA -/
def AbcCba (a b c : Nat) : Nat :=
  100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a

theorem smallest_abc_cba_divisible_by_11 :
  ∀ a b c : Nat,
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    0 < a ∧ a < 10 ∧ b < 10 ∧ c < 10 →
    AbcCba a b c ≥ 123321 ∨ ¬(AbcCba a b c % 11 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_abc_cba_divisible_by_11_l2713_271364


namespace NUMINAMATH_CALUDE_profit_calculation_l2713_271328

/-- Profit calculation for a product with variable price reduction --/
theorem profit_calculation 
  (price_tag : ℕ) 
  (discount : ℚ) 
  (initial_profit : ℕ) 
  (initial_sales : ℕ) 
  (sales_increase : ℕ) 
  (x : ℕ) 
  (h1 : price_tag = 80)
  (h2 : discount = 1/5)
  (h3 : initial_profit = 24)
  (h4 : initial_sales = 220)
  (h5 : sales_increase = 20) :
  ∃ y : ℤ, y = (24 - x) * (initial_sales + sales_increase * x) :=
by sorry

end NUMINAMATH_CALUDE_profit_calculation_l2713_271328


namespace NUMINAMATH_CALUDE_total_amount_calculation_l2713_271356

theorem total_amount_calculation (two_won_bills : ℕ) (one_won_bills : ℕ) : 
  two_won_bills = 8 → one_won_bills = 2 → two_won_bills * 2 + one_won_bills * 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l2713_271356


namespace NUMINAMATH_CALUDE_base_three_20201_equals_181_l2713_271304

def base_three_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

theorem base_three_20201_equals_181 :
  base_three_to_ten [1, 0, 2, 0, 2] = 181 := by
  sorry

end NUMINAMATH_CALUDE_base_three_20201_equals_181_l2713_271304


namespace NUMINAMATH_CALUDE_probability_of_sum_15_is_correct_l2713_271323

/-- Represents a standard 52-card deck -/
def standardDeck : Nat := 52

/-- Represents the number of cards for each value in a standard deck -/
def cardsPerValue : Nat := 4

/-- Represents the probability of drawing two number cards (2 through 10) 
    from a standard 52-card deck that total 15 -/
def probabilityOfSum15 : ℚ := 28 / 221

theorem probability_of_sum_15_is_correct : 
  probabilityOfSum15 = (
    -- Probability of drawing a 5, 6, 7, 8, or 9 first, then completing the pair
    (5 * cardsPerValue * 4 * cardsPerValue) / (standardDeck * (standardDeck - 1)) +
    -- Probability of drawing a 10 first, then a 5
    (cardsPerValue * cardsPerValue) / (standardDeck * (standardDeck - 1))
  ) := by sorry

end NUMINAMATH_CALUDE_probability_of_sum_15_is_correct_l2713_271323


namespace NUMINAMATH_CALUDE_extreme_value_condition_monotonicity_intervals_min_value_on_interval_l2713_271357

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 1

-- Theorem 1: f(x) has an extreme value at x = 1 if and only if a = -1
theorem extreme_value_condition (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a x ≤ f a 1) ↔ a = -1 :=
sorry

-- Theorem 2: Monotonicity intervals depend on the value of a
theorem monotonicity_intervals (a : ℝ) :
  (a = 0 → ∀ (x y : ℝ), x < y → f a x < f a y) ∧
  (a > 0 → ∀ (x y : ℝ), (x < y ∧ y < -a) ∨ (x > 0 ∧ y > x) → f a x < f a y) ∧
  (a > 0 → ∀ (x y : ℝ), -a < x ∧ x < y ∧ y < 0 → f a x > f a y) ∧
  (a < 0 → ∀ (x y : ℝ), (x < y ∧ y < 0) ∨ (x > -a ∧ y > x) → f a x < f a y) ∧
  (a < 0 → ∀ (x y : ℝ), 0 < x ∧ x < y ∧ y < -a → f a x > f a y) :=
sorry

-- Theorem 3: Minimum value on [0, 2] depends on the value of a
theorem min_value_on_interval (a : ℝ) :
  (a ≥ 0 → ∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ f a 0) ∧
  (-2 < a ∧ a < 0 → ∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ f a (-a)) ∧
  (a ≤ -2 → ∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ f a 2) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_monotonicity_intervals_min_value_on_interval_l2713_271357


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2713_271352

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (z - 3) = -1 + 3 * Complex.I) : 
  z.im = 1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2713_271352


namespace NUMINAMATH_CALUDE_solution_set_of_f_neg_x_l2713_271313

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * x - 1) * (x - b)

-- State the theorem
theorem solution_set_of_f_neg_x (a b : ℝ) :
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, f a b (-x) < 0 ↔ x < -3 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_neg_x_l2713_271313


namespace NUMINAMATH_CALUDE_roots_sum_squared_plus_double_plus_other_l2713_271375

theorem roots_sum_squared_plus_double_plus_other (a b : ℝ) : 
  a^2 + a - 2023 = 0 → b^2 + b - 2023 = 0 → a^2 + 2*a + b = 2022 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_squared_plus_double_plus_other_l2713_271375


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l2713_271322

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - 10*x + 3

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3*x^2 - 10

-- Theorem statement
theorem tangent_point_coordinates :
  ∀ (x y : ℝ),
    x < 0 →
    y = curve x →
    curve_derivative x = 2 →
    (x = -2 ∧ y = 15) :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l2713_271322


namespace NUMINAMATH_CALUDE_negative_a_fifth_squared_l2713_271314

theorem negative_a_fifth_squared (a : ℝ) : (-a^5)^2 = a^10 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_fifth_squared_l2713_271314


namespace NUMINAMATH_CALUDE_donut_combinations_l2713_271306

theorem donut_combinations : 
  let total_donuts : ℕ := 8
  let donut_types : ℕ := 5
  let remaining_donuts : ℕ := total_donuts - donut_types
  Nat.choose (remaining_donuts + donut_types - 1) (donut_types - 1) = 35 :=
by sorry

end NUMINAMATH_CALUDE_donut_combinations_l2713_271306


namespace NUMINAMATH_CALUDE_factor_implication_l2713_271307

theorem factor_implication (m n : ℝ) : 
  (∃ a b : ℝ, 3 * X^3 - m * X + n = a * (X - 3) * (X + 1) * X) →
  |3 * m + n| = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_factor_implication_l2713_271307


namespace NUMINAMATH_CALUDE_odd_even_intersection_empty_l2713_271361

def odd_integers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}
def even_integers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem odd_even_intersection_empty : odd_integers ∩ even_integers = ∅ := by
  sorry

end NUMINAMATH_CALUDE_odd_even_intersection_empty_l2713_271361


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2713_271363

/-- Calculate the profit percentage given the selling price and cost price -/
theorem profit_percentage_calculation (selling_price cost_price : ℚ) :
  selling_price = 1800 ∧ cost_price = 1500 →
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2713_271363


namespace NUMINAMATH_CALUDE_theater_queue_arrangements_l2713_271366

theorem theater_queue_arrangements :
  let total_people : ℕ := 7
  let pair_size : ℕ := 2
  let units : ℕ := total_people - pair_size + 1
  units.factorial * pair_size.factorial = 1440 :=
by sorry

end NUMINAMATH_CALUDE_theater_queue_arrangements_l2713_271366


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2713_271338

/-- Given a quadratic function f(x) = x^2 + bx + c, 
    prove that c < 0 is sufficient but not necessary for f(x) < 0 to have a real solution -/
theorem sufficient_not_necessary_condition (b c : ℝ) :
  (∀ x, (x : ℝ)^2 + b*x + c < 0 → c < 0) ∧
  ¬(∀ b c : ℝ, (∃ x, (x : ℝ)^2 + b*x + c < 0) → c < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2713_271338


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2713_271394

/-- Given a cube with surface area 150 cm², prove its volume is 125 cm³ -/
theorem cube_volume_from_surface_area :
  ∀ (side_length : ℝ),
  (6 : ℝ) * side_length^2 = 150 →
  side_length^3 = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2713_271394


namespace NUMINAMATH_CALUDE_intersecting_lines_regions_l2713_271336

/-- The number of regions created by n intersecting lines in a plane -/
def total_regions (n : ℕ) : ℚ :=
  (n^2 + n + 2) / 2

/-- The number of bounded regions (polygons) created by n intersecting lines in a plane -/
def bounded_regions (n : ℕ) : ℚ :=
  (n^2 - 3*n + 2) / 2

/-- Theorem stating the number of regions and bounded regions created by n intersecting lines -/
theorem intersecting_lines_regions (n : ℕ) :
  (total_regions n = (n^2 + n + 2) / 2) ∧
  (bounded_regions n = (n^2 - 3*n + 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_regions_l2713_271336


namespace NUMINAMATH_CALUDE_probability_of_cooking_sum_of_probabilities_l2713_271329

/-- Represents the set of available courses. -/
inductive Course
| Planting
| Cooking
| Pottery
| Carpentry

/-- The probability of selecting a specific course from the available courses. -/
def probability_of_course (course : Course) : ℚ :=
  1 / 4

/-- Theorem stating that the probability of selecting "cooking" is 1/4. -/
theorem probability_of_cooking :
  probability_of_course Course.Cooking = 1 / 4 := by
  sorry

/-- Theorem stating that the sum of probabilities for all courses is 1. -/
theorem sum_of_probabilities :
  (probability_of_course Course.Planting) +
  (probability_of_course Course.Cooking) +
  (probability_of_course Course.Pottery) +
  (probability_of_course Course.Carpentry) = 1 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_cooking_sum_of_probabilities_l2713_271329


namespace NUMINAMATH_CALUDE_factor_sum_l2713_271346

theorem factor_sum (x y : ℝ) (a b c d e f g : ℤ) :
  16 * x^8 - 256 * y^4 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x^2 + g*y^2) →
  a + b + c + d + e + f + g = 7 := by
  sorry

end NUMINAMATH_CALUDE_factor_sum_l2713_271346


namespace NUMINAMATH_CALUDE_expression_simplification_l2713_271368

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ -1) :
  (a - (2*a - 1) / a) + (1 - a^2) / (a^2 + a) = (a^2 - 3*a + 2) / a := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2713_271368


namespace NUMINAMATH_CALUDE_car_travel_time_ratio_l2713_271359

theorem car_travel_time_ratio : 
  let distance : ℝ := 504
  let original_time : ℝ := 6
  let new_speed : ℝ := 56
  let new_time := distance / new_speed
  new_time / original_time = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_ratio_l2713_271359


namespace NUMINAMATH_CALUDE_coefficient_of_z_in_equation1_l2713_271345

-- Define the system of equations
def equation1 (x y z : ℚ) : Prop := 6 * x - 5 * y + z = 22 / 3
def equation2 (x y z : ℚ) : Prop := 4 * x + 8 * y - 11 * z = 7
def equation3 (x y z : ℚ) : Prop := 5 * x - 6 * y + 2 * z = 12

-- Define the sum condition
def sum_condition (x y z : ℚ) : Prop := x + y + z = 10

-- Theorem statement
theorem coefficient_of_z_in_equation1 (x y z : ℚ) 
  (eq1 : equation1 x y z) (eq2 : equation2 x y z) (eq3 : equation3 x y z) 
  (sum : sum_condition x y z) : 
  ∃ (a b c : ℚ), equation1 x y z ↔ a * x + b * y + 1 * z = 22 / 3 :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_z_in_equation1_l2713_271345


namespace NUMINAMATH_CALUDE_gcf_of_26_and_16_l2713_271317

theorem gcf_of_26_and_16 :
  let n : ℕ := 26
  let m : ℕ := 16
  let lcm_nm : ℕ := 52
  Nat.lcm n m = lcm_nm →
  Nat.gcd n m = 8 := by
sorry

end NUMINAMATH_CALUDE_gcf_of_26_and_16_l2713_271317


namespace NUMINAMATH_CALUDE_sqrt_x_minus_8_range_l2713_271343

-- Define the condition for a meaningful square root
def meaningful_sqrt (x : ℝ) : Prop := x - 8 ≥ 0

-- Theorem stating the range of x for which √(x-8) is meaningful
theorem sqrt_x_minus_8_range (x : ℝ) : 
  meaningful_sqrt x ↔ x ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_8_range_l2713_271343


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2713_271349

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2713_271349


namespace NUMINAMATH_CALUDE_water_height_in_aquarium_l2713_271378

/-- Proves that 10 litres of water in an aquarium with dimensions 50 cm length
and 20 cm breadth will rise to a height of 10 cm. -/
theorem water_height_in_aquarium (length : ℝ) (breadth : ℝ) (volume_litres : ℝ) :
  length = 50 →
  breadth = 20 →
  volume_litres = 10 →
  (volume_litres * 1000) / (length * breadth) = 10 := by
  sorry

end NUMINAMATH_CALUDE_water_height_in_aquarium_l2713_271378


namespace NUMINAMATH_CALUDE_min_value_expression_l2713_271333

theorem min_value_expression (a b : ℝ) (h : a ≠ -1) :
  |a + b| + |1 / (a + 1) - b| ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2713_271333


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l2713_271334

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- The area of overlap between a rectangle and a square -/
def overlap_area (r : Rectangle) (s : Square) : ℝ := sorry

/-- The theorem stating the ratio of rectangle's width to height -/
theorem rectangle_square_overlap_ratio 
  (r : Rectangle) 
  (s : Square) 
  (h1 : overlap_area r s = 0.6 * r.width * r.height) 
  (h2 : overlap_area r s = 0.3 * s.side * s.side) : 
  r.width / r.height = 12.5 := by sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l2713_271334


namespace NUMINAMATH_CALUDE_tangent_line_and_zeros_l2713_271344

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 6*x + 1

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 6

-- Define the function g
def g (a : ℝ) (m : ℝ) (x : ℝ) : ℝ := f a x - m

theorem tangent_line_and_zeros (a : ℝ) :
  f' a 1 = -6 →
  (∃ b c : ℝ, ∀ x y : ℝ, 12*x + 2*y - 1 = 0 ↔ y = (f a 1) + f' a 1 * (x - 1)) ∧
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ∈ Set.Icc (-2) 4 ∧ x₂ ∈ Set.Icc (-2) 4 ∧ x₃ ∈ Set.Icc (-2) 4 ∧
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    g a m x₁ = 0 ∧ g a m x₂ = 0 ∧ g a m x₃ = 0) →
    m ∈ Set.Icc (-1) (9/2)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_zeros_l2713_271344


namespace NUMINAMATH_CALUDE_cubic_equation_value_l2713_271390

theorem cubic_equation_value (a : ℝ) (h : a^2 + a - 1 = 0) : a^3 + 2*a^2 + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l2713_271390


namespace NUMINAMATH_CALUDE_one_minus_repeating_third_equals_two_thirds_l2713_271398

def repeating_decimal_one_third : ℚ := 1/3

theorem one_minus_repeating_third_equals_two_thirds :
  1 - repeating_decimal_one_third = 2/3 := by sorry

end NUMINAMATH_CALUDE_one_minus_repeating_third_equals_two_thirds_l2713_271398


namespace NUMINAMATH_CALUDE_solve_average_salary_l2713_271324

def average_salary_problem (num_employees : ℕ) (manager_salary : ℕ) (avg_increase : ℕ) : Prop :=
  let total_salary := num_employees * (manager_salary / (num_employees + 1) - avg_increase)
  let new_total_salary := total_salary + manager_salary
  let new_average := new_total_salary / (num_employees + 1)
  (manager_salary / (num_employees + 1) - avg_increase) = 2400 ∧
  new_average = (manager_salary / (num_employees + 1) - avg_increase) + avg_increase

theorem solve_average_salary :
  average_salary_problem 24 4900 100 := by
  sorry

end NUMINAMATH_CALUDE_solve_average_salary_l2713_271324


namespace NUMINAMATH_CALUDE_eliminate_cycles_in_complete_digraph_l2713_271392

/-- A complete directed graph with 32 vertices -/
def CompleteDigraph : Type := Fin 32 → Fin 32 → Prop

/-- The property that a graph contains no directed cycles -/
def NoCycles (g : CompleteDigraph) : Prop := sorry

/-- A step that changes the direction of a single edge -/
def Step (g₁ g₂ : CompleteDigraph) : Prop := sorry

/-- The theorem stating that it's possible to eliminate all cycles in at most 208 steps -/
theorem eliminate_cycles_in_complete_digraph :
  ∃ (sequence : Fin 209 → CompleteDigraph),
    (∀ i : Fin 208, Step (sequence i) (sequence (i + 1))) ∧
    NoCycles (sequence 208) :=
  sorry

end NUMINAMATH_CALUDE_eliminate_cycles_in_complete_digraph_l2713_271392


namespace NUMINAMATH_CALUDE_trapezoid_wings_area_l2713_271347

/-- A trapezoid divided into four triangles -/
structure Trapezoid :=
  (A₁ : ℝ) -- Area of first triangle
  (A₂ : ℝ) -- Area of second triangle
  (A₃ : ℝ) -- Area of third triangle
  (A₄ : ℝ) -- Area of fourth triangle

/-- The theorem stating that if two triangles in the trapezoid have areas 4 and 9,
    then the sum of the areas of the other two triangles is 12 -/
theorem trapezoid_wings_area (T : Trapezoid) 
  (h₁ : T.A₁ = 4) 
  (h₂ : T.A₂ = 9) : 
  T.A₃ + T.A₄ = 12 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_wings_area_l2713_271347


namespace NUMINAMATH_CALUDE_two_digit_number_subtraction_l2713_271365

theorem two_digit_number_subtraction (b : ℕ) (h1 : b < 9) : 
  (11 * b + 10) - (11 * b + 1) = 9 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_subtraction_l2713_271365


namespace NUMINAMATH_CALUDE_cubic_inequality_l2713_271379

theorem cubic_inequality (x : ℝ) : x^3 - 4*x^2 + 4*x < 0 ↔ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2713_271379


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2713_271382

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x^2 - 2*m*x + 4 = 0 ∧ y^2 - 2*m*y + 4 = 0 ∧ x ≠ y ∧ x > 2 ∧ y < 2) → 
  m > 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2713_271382


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l2713_271397

/-- The length of the diagonal of a rectangular solid with edges of length 2, 3, and 4 is √29. -/
theorem rectangular_solid_diagonal : 
  let a : ℝ := 2
  let b : ℝ := 3
  let c : ℝ := 4
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l2713_271397


namespace NUMINAMATH_CALUDE_toms_fruit_purchase_cost_l2713_271302

/-- Calculates the total cost of a fruit purchase with a quantity-based discount --/
def fruitPurchaseCost (lemonPrice papayaPrice mangoPrice : ℕ) 
                      (lemonQty papayaQty mangoQty : ℕ) 
                      (fruitPerDiscount : ℕ) (discountAmount : ℕ) : ℕ :=
  let totalCost := lemonPrice * lemonQty + papayaPrice * papayaQty + mangoPrice * mangoQty
  let totalFruits := lemonQty + papayaQty + mangoQty
  let discountQty := totalFruits / fruitPerDiscount
  totalCost - discountQty * discountAmount

/-- Theorem: Tom's fruit purchase costs $21 --/
theorem toms_fruit_purchase_cost : 
  fruitPurchaseCost 2 1 4 6 4 2 4 1 = 21 := by
  sorry

#eval fruitPurchaseCost 2 1 4 6 4 2 4 1

end NUMINAMATH_CALUDE_toms_fruit_purchase_cost_l2713_271302


namespace NUMINAMATH_CALUDE_tiling_uniqueness_l2713_271316

/-- A rectangular grid --/
structure RectangularGrid where
  rows : ℕ
  cols : ℕ

/-- A cell in the grid --/
structure Cell where
  row : ℕ
  col : ℕ

/-- A tiling of the grid --/
def Tiling (grid : RectangularGrid) := Set (Set Cell)

/-- The set of central cells for a given tiling --/
def CentralCells (grid : RectangularGrid) (tiling : Tiling grid) : Set Cell :=
  sorry

/-- Theorem: The set of central cells uniquely determines the tiling --/
theorem tiling_uniqueness (grid : RectangularGrid) 
  (tiling1 tiling2 : Tiling grid) :
  CentralCells grid tiling1 = CentralCells grid tiling2 → tiling1 = tiling2 :=
sorry

end NUMINAMATH_CALUDE_tiling_uniqueness_l2713_271316


namespace NUMINAMATH_CALUDE_log_range_incorrect_l2713_271388

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_range_incorrect (b : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : b > 1) 
  (h2 : y = log b x) 
  (h3 : Real.sqrt b < x) 
  (h4 : x < b) : 
  ¬ (0.5 < y ∧ y < 1.5) :=
sorry

end NUMINAMATH_CALUDE_log_range_incorrect_l2713_271388


namespace NUMINAMATH_CALUDE_non_negative_xy_l2713_271372

theorem non_negative_xy (x y : ℝ) :
  |x^2 + y^2 - 4*x - 4*y + 5| = |2*x + 2*y - 4| → x ≥ 0 ∧ y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_non_negative_xy_l2713_271372


namespace NUMINAMATH_CALUDE_horner_method_f_3_l2713_271396

def f (x : ℝ) : ℝ := x^5 - 2*x^3 + 3*x^2 - x + 1

def horner_v3 (x : ℝ) : ℝ := ((((x + 0)*x - 2)*x + 3)*x - 1)*x + 1

theorem horner_method_f_3 : horner_v3 3 = 24 := by sorry

end NUMINAMATH_CALUDE_horner_method_f_3_l2713_271396


namespace NUMINAMATH_CALUDE_eric_marbles_l2713_271301

/-- The number of marbles Eric has -/
def total_marbles (white blue green : ℕ) : ℕ := white + blue + green

/-- Proof that Eric has 20 marbles in total -/
theorem eric_marbles : total_marbles 12 6 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_eric_marbles_l2713_271301


namespace NUMINAMATH_CALUDE_ab_max_and_inverse_sum_min_l2713_271399

theorem ab_max_and_inverse_sum_min (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + 4*b = 4) : 
  (∀ x y, x > 0 → y > 0 → x + 4*y = 4 → a*b ≥ x*y) ∧ 
  (∀ x y, x > 0 → y > 0 → x + 4*y = 4 → 1/a + 4/b ≤ 1/x + 4/y) ∧
  (a*b = 1) ∧ (1/a + 4/b = 25/4) :=
sorry

end NUMINAMATH_CALUDE_ab_max_and_inverse_sum_min_l2713_271399


namespace NUMINAMATH_CALUDE_sticker_problem_l2713_271310

theorem sticker_problem (bob tom dan : ℕ) 
  (h1 : dan = 2 * tom) 
  (h2 : tom = 3 * bob) 
  (h3 : dan = 72) : 
  bob = 12 := by
  sorry

end NUMINAMATH_CALUDE_sticker_problem_l2713_271310


namespace NUMINAMATH_CALUDE_backpack_pencilcase_combinations_l2713_271340

/-- The number of combinations formed by selecting one item from each of two sets -/
def combinations (set1 : ℕ) (set2 : ℕ) : ℕ := set1 * set2

/-- Theorem: The number of combinations formed by selecting one backpack from 2 styles
    and one pencil case from 2 styles is equal to 4 -/
theorem backpack_pencilcase_combinations :
  let backpacks : ℕ := 2
  let pencilcases : ℕ := 2
  combinations backpacks pencilcases = 4 := by
  sorry

end NUMINAMATH_CALUDE_backpack_pencilcase_combinations_l2713_271340


namespace NUMINAMATH_CALUDE_set_operations_l2713_271354

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 * x - 4 ≥ 0}

-- Define the theorem
theorem set_operations :
  (Set.univ \ (A ∩ B) = {x | x < 2 ∨ x ≥ 3}) ∧
  ((Set.univ \ A) ∩ (Set.univ \ B) = {x | x < -1}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l2713_271354


namespace NUMINAMATH_CALUDE_nine_b_value_l2713_271308

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : b - 3 = a) : 9 * b = 216 / 11 := by
  sorry

end NUMINAMATH_CALUDE_nine_b_value_l2713_271308


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_solution_l2713_271393

theorem cryptarithmetic_puzzle_solution :
  ∃ (A B C D E F : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    A * B + C * D = 10 * E + F ∧
    B + C + D ≠ A ∧
    A = 2 * D ∧
    F = 8 :=
by sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_solution_l2713_271393


namespace NUMINAMATH_CALUDE_min_value_on_circle_l2713_271386

theorem min_value_on_circle (x y : ℝ) :
  x^2 + y^2 - 4*x - 6*y + 12 = 0 →
  ∃ (min_val : ℝ), min_val = 14 - 2 * Real.sqrt 13 ∧
    ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 6*y' + 12 = 0 →
      x'^2 + y'^2 ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l2713_271386


namespace NUMINAMATH_CALUDE_problem_1_l2713_271315

theorem problem_1 : (Real.sqrt 5 + Real.sqrt 3) * (Real.sqrt 5 - Real.sqrt 3) + (Real.sqrt 3 - 2)^2 = 9 - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2713_271315


namespace NUMINAMATH_CALUDE_coefficient_a7_equals_negative_eight_l2713_271362

theorem coefficient_a7_equals_negative_eight :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ),
  (∀ x : ℝ, (x - 2)^8 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
                        a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8) →
  a₇ = -8 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a7_equals_negative_eight_l2713_271362


namespace NUMINAMATH_CALUDE_arithmetic_operations_equal_reciprocal_2016_l2713_271330

theorem arithmetic_operations_equal_reciprocal_2016 :
  (1 / 8 * 1 / 9 * 1 / 28 : ℚ) = 1 / 2016 ∧ 
  ((1 / 8 - 1 / 9) * 1 / 28 : ℚ) = 1 / 2016 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_equal_reciprocal_2016_l2713_271330


namespace NUMINAMATH_CALUDE_sheet_width_correct_l2713_271341

/-- The width of a rectangular metallic sheet -/
def sheet_width : ℝ := 36

/-- The length of the rectangular metallic sheet -/
def sheet_length : ℝ := 48

/-- The side length of the square cut from each corner -/
def cut_square_side : ℝ := 8

/-- The volume of the resulting open box -/
def box_volume : ℝ := 5120

/-- Theorem stating that the given dimensions result in the correct volume -/
theorem sheet_width_correct : 
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = box_volume :=
by sorry

end NUMINAMATH_CALUDE_sheet_width_correct_l2713_271341
