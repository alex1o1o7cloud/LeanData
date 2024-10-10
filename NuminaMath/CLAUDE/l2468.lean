import Mathlib

namespace stratified_sample_composition_l2468_246899

theorem stratified_sample_composition 
  (total_athletes : ℕ) 
  (male_athletes : ℕ) 
  (female_athletes : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = male_athletes + female_athletes)
  (h2 : total_athletes = 98)
  (h3 : male_athletes = 56)
  (h4 : female_athletes = 42)
  (h5 : sample_size = 14) :
  (male_athletes * sample_size / total_athletes : ℚ) = 8 ∧ 
  (female_athletes * sample_size / total_athletes : ℚ) = 6 :=
by sorry

end stratified_sample_composition_l2468_246899


namespace max_value_theorem_l2468_246846

theorem max_value_theorem (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  3 * x * z * Real.sqrt 2 + 5 * x * y ≤ Real.sqrt 43 :=
by sorry

end max_value_theorem_l2468_246846


namespace n_equals_four_l2468_246868

theorem n_equals_four (n : ℝ) (h : 3 * n = 6 * 2) : n = 4 := by
  sorry

end n_equals_four_l2468_246868


namespace reciprocal_equality_l2468_246896

theorem reciprocal_equality (a b : ℝ) 
  (ha : a⁻¹ = -8) 
  (hb : (-b)⁻¹ = 8) : 
  a = b := by sorry

end reciprocal_equality_l2468_246896


namespace order_of_powers_l2468_246807

theorem order_of_powers : 
  let a : ℝ := (2/5: ℝ)^(3/5: ℝ)
  let b : ℝ := (2/5: ℝ)^(2/5: ℝ)
  let c : ℝ := (3/5: ℝ)^(2/5: ℝ)
  a < b ∧ b < c := by sorry

end order_of_powers_l2468_246807


namespace pens_given_to_sharon_problem_l2468_246844

/-- Calculates the number of pens given to Sharon -/
def pens_given_to_sharon (initial_pens : ℕ) (mike_pens : ℕ) (final_pens : ℕ) : ℕ :=
  2 * (initial_pens + mike_pens) - final_pens

theorem pens_given_to_sharon_problem :
  let initial_pens : ℕ := 5
  let mike_pens : ℕ := 20
  let final_pens : ℕ := 31
  pens_given_to_sharon initial_pens mike_pens final_pens = 19 := by
  sorry

#eval pens_given_to_sharon 5 20 31

end pens_given_to_sharon_problem_l2468_246844


namespace power_of_power_l2468_246829

theorem power_of_power (a : ℝ) : (a^5)^3 = a^15 := by
  sorry

end power_of_power_l2468_246829


namespace isosceles_trapezoid_ABCD_l2468_246849

/-- Given points A, B, C, and D in a 2D coordinate system, prove that ABCD is an isosceles trapezoid with AB parallel to CD -/
theorem isosceles_trapezoid_ABCD :
  let A : ℝ × ℝ := (-6, -1)
  let B : ℝ × ℝ := (2, 3)
  let C : ℝ × ℝ := (-1, 4)
  let D : ℝ × ℝ := (-5, 2)
  
  -- AB is parallel to CD
  (B.2 - A.2) / (B.1 - A.1) = (D.2 - C.2) / (D.1 - C.1) ∧
  
  -- AD = BC (isosceles condition)
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
  
  -- AB ≠ CD (trapezoid condition)
  (B.1 - A.1)^2 + (B.2 - A.2)^2 ≠ (D.1 - C.1)^2 + (D.2 - C.2)^2 :=
by
  sorry


end isosceles_trapezoid_ABCD_l2468_246849


namespace paint_area_calculation_l2468_246898

/-- The height of the wall in feet -/
def wall_height : ℝ := 10

/-- The length of the wall in feet -/
def wall_length : ℝ := 15

/-- The height of the door in feet -/
def door_height : ℝ := 3

/-- The width of the door in feet -/
def door_width : ℝ := 5

/-- The area to paint in square feet -/
def area_to_paint : ℝ := wall_height * wall_length - door_height * door_width

theorem paint_area_calculation :
  area_to_paint = 135 := by sorry

end paint_area_calculation_l2468_246898


namespace convex_quad_interior_point_inequality_l2468_246823

/-- A convex quadrilateral with an interior point and parallel lines -/
structure ConvexQuadWithInteriorPoint where
  /-- The area of the convex quadrilateral ABCD -/
  T : ℝ
  /-- The area of quadrilateral AEPH -/
  t₁ : ℝ
  /-- The area of quadrilateral PFCG -/
  t₂ : ℝ
  /-- The areas are non-negative -/
  h₁ : 0 ≤ T
  h₂ : 0 ≤ t₁
  h₃ : 0 ≤ t₂

/-- The inequality holds for any convex quadrilateral with an interior point -/
theorem convex_quad_interior_point_inequality (q : ConvexQuadWithInteriorPoint) :
  Real.sqrt q.t₁ + Real.sqrt q.t₂ ≤ Real.sqrt q.T :=
sorry

end convex_quad_interior_point_inequality_l2468_246823


namespace systematic_sampling_interval_l2468_246828

/-- Calculates the sampling interval for systematic sampling -/
def samplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (populationSize - populationSize % sampleSize) / sampleSize

theorem systematic_sampling_interval :
  samplingInterval 1003 50 = 20 := by
  sorry

end systematic_sampling_interval_l2468_246828


namespace abs_difference_opposite_signs_l2468_246835

theorem abs_difference_opposite_signs (a b : ℝ) 
  (ha : |a| = 4) 
  (hb : |b| = 2) 
  (hab : a * b < 0) : 
  |a - b| = 6 := by
sorry

end abs_difference_opposite_signs_l2468_246835


namespace expression_simplification_l2468_246869

theorem expression_simplification (a : ℝ) (h : a = 2) :
  (a^2 / (a - 1) - a) / ((a + a^2) / (1 - 2*a + a^2)) = 1/3 := by
  sorry

end expression_simplification_l2468_246869


namespace reading_time_comparison_l2468_246852

/-- Given two people A and B, where A reads 5 times faster than B,
    prove that if B takes 3 hours to read a book,
    then A will take 36 minutes to read the same book. -/
theorem reading_time_comparison (reading_speed_ratio : ℝ) (person_b_time : ℝ) :
  reading_speed_ratio = 5 →
  person_b_time = 3 →
  (person_b_time * 60) / reading_speed_ratio = 36 :=
by sorry

end reading_time_comparison_l2468_246852


namespace square_root_of_product_plus_one_l2468_246878

theorem square_root_of_product_plus_one : 
  Real.sqrt ((34 : ℝ) * 32 * 28 * 26 + 1) = 170 := by sorry

end square_root_of_product_plus_one_l2468_246878


namespace unique_solution_system_l2468_246891

theorem unique_solution_system (x y u v : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ u > 0 ∧ v > 0)
  (h1 : x + y = u)
  (h2 : v * x * y = u + v)
  (h3 : x * y * u * v = 16) :
  x = 2 ∧ y = 2 ∧ u = 2 ∧ v = 2 := by
sorry

end unique_solution_system_l2468_246891


namespace highest_uniquely_identifiable_score_l2468_246885

/-- The AHSME scoring system -/
def score (c w : ℕ) : ℕ := 30 + 4 * c - w

/-- The maximum number of questions in AHSME -/
def max_questions : ℕ := 30

/-- Predicate to check if a score is uniquely identifiable -/
def is_uniquely_identifiable (s : ℕ) : Prop :=
  ∃! (c w : ℕ), c ≤ max_questions ∧ w ≤ max_questions ∧ s = score c w

/-- Theorem stating that 130 is the highest possible uniquely identifiable score over 100 -/
theorem highest_uniquely_identifiable_score :
  (∀ s : ℕ, s > 130 → ¬(is_uniquely_identifiable s)) ∧
  (is_uniquely_identifiable 130) ∧
  (130 > 100) :=
sorry

end highest_uniquely_identifiable_score_l2468_246885


namespace graduating_class_male_percentage_l2468_246827

theorem graduating_class_male_percentage :
  ∀ (M F : ℝ),
  M + F = 100 →
  0.5 * M + 0.7 * F = 62 →
  M = 40 :=
by
  sorry

end graduating_class_male_percentage_l2468_246827


namespace power_of_product_l2468_246834

theorem power_of_product (a : ℝ) : (-4 * a^3)^2 = 16 * a^6 := by
  sorry

end power_of_product_l2468_246834


namespace triangle_cosine_theorem_l2468_246821

theorem triangle_cosine_theorem (A : ℝ) (a m : ℝ) (θ : ℝ) 
  (h1 : A = 40) 
  (h2 : a = 12) 
  (h3 : m = 10) 
  (h4 : A = 1/2 * a * m * Real.sin θ) 
  (h5 : 0 < θ) 
  (h6 : θ < π/2) : 
  Real.cos θ = Real.sqrt 5 / 3 := by
sorry

end triangle_cosine_theorem_l2468_246821


namespace jackson_chairs_count_l2468_246802

/-- The number of chairs Jackson needs to buy for his restaurant -/
def total_chairs (four_seat_tables six_seat_tables : ℕ) (seats_per_four_seat_table seats_per_six_seat_table : ℕ) : ℕ :=
  four_seat_tables * seats_per_four_seat_table + six_seat_tables * seats_per_six_seat_table

/-- Proof that Jackson needs to buy 96 chairs for his restaurant -/
theorem jackson_chairs_count :
  total_chairs 6 12 4 6 = 96 := by
  sorry

end jackson_chairs_count_l2468_246802


namespace largest_integer_l2468_246800

theorem largest_integer (a b c d : ℤ) 
  (sum_abc : a + b + c = 160)
  (sum_abd : a + b + d = 185)
  (sum_acd : a + c + d = 205)
  (sum_bcd : b + c + d = 230) :
  max a (max b (max c d)) = 100 := by
sorry

end largest_integer_l2468_246800


namespace probability_at_least_one_from_A_l2468_246809

/-- Represents the number of factories in each district -/
structure DistrictFactories where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Represents the number of factories sampled from each district -/
structure SampledFactories where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Calculates the probability of selecting at least one factory from district A 
    when randomly choosing 2 out of 7 stratified sampled factories -/
def probabilityAtLeastOneFromA (df : DistrictFactories) (sf : SampledFactories) : ℚ :=
  sorry

/-- Theorem stating the probability of selecting at least one factory from district A 
    is 11/21 given the specific conditions -/
theorem probability_at_least_one_from_A : 
  let df : DistrictFactories := { A := 18, B := 27, C := 18 }
  let sf : SampledFactories := { A := 2, B := 3, C := 2 }
  probabilityAtLeastOneFromA df sf = 11/21 := by
  sorry

end probability_at_least_one_from_A_l2468_246809


namespace common_chord_equation_l2468_246837

/-- Two circles in a plane -/
structure TwoCircles where
  circle1 : (ℝ × ℝ) → Prop
  circle2 : (ℝ × ℝ) → Prop

/-- The equation of a line in a plane -/
structure Line where
  equation : (ℝ × ℝ) → Prop

/-- The common chord of two intersecting circles -/
def commonChord (circles : TwoCircles) : Line :=
  sorry

/-- Combining equations of two circles -/
def combineEquations (circles : TwoCircles) : (ℝ × ℝ) → Prop :=
  sorry

/-- Eliminating quadratic terms from an equation -/
def eliminateQuadraticTerms (eq : (ℝ × ℝ) → Prop) : (ℝ × ℝ) → Prop :=
  sorry

/-- Theorem: The equation of the common chord of two intersecting circles
    is obtained by eliminating the quadratic terms after combining
    the equations of the two circles -/
theorem common_chord_equation (circles : TwoCircles) :
  (commonChord circles).equation =
  eliminateQuadraticTerms (combineEquations circles) :=
sorry

end common_chord_equation_l2468_246837


namespace permutation_sum_l2468_246848

theorem permutation_sum (n : ℕ) : 
  n + 3 ≤ 2 * n ∧ n + 1 ≤ 4 → 
  (Nat.factorial (2 * n)) / (Nat.factorial (2 * n - (n + 3))) + 
  (Nat.factorial 4) / (Nat.factorial (4 - (n + 1))) = 744 := by
  sorry

end permutation_sum_l2468_246848


namespace monotonic_quadratic_constraint_l2468_246890

/-- A function f is monotonic on an interval [a, b] if and only if
    its derivative f' does not change sign on (a, b) -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ (Set.Icc a b), (∀ y ∈ (Set.Icc a b), x ≤ y → f x ≤ f y) ∨
                       (∀ y ∈ (Set.Icc a b), x ≤ y → f y ≤ f x)

/-- The quadratic function f(x) = 4x² - kx - 8 -/
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

theorem monotonic_quadratic_constraint (k : ℝ) :
  IsMonotonic (f k) 5 8 ↔ k ∈ Set.Iic 40 ∪ Set.Ici 64 := by
  sorry

#check monotonic_quadratic_constraint

end monotonic_quadratic_constraint_l2468_246890


namespace fifth_term_of_sequence_l2468_246831

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ n

theorem fifth_term_of_sequence (x : ℝ) :
  let a₁ : ℝ := 4
  let a₂ : ℝ := 12 * x^2
  let a₃ : ℝ := 36 * x^4
  let a₄ : ℝ := 108 * x^6
  let r : ℝ := 3 * x^2
  geometric_sequence a₁ r 4 = 324 * x^8 :=
by sorry

end fifth_term_of_sequence_l2468_246831


namespace circle_division_theorem_l2468_246894

/-- Represents a region in the circle --/
structure Region where
  value : Nat
  deriving Repr

/-- Represents a line dividing the circle --/
structure DividingLine where
  left_regions : List Region
  right_regions : List Region
  deriving Repr

/-- The configuration of regions in the circle --/
def CircleConfiguration := List Region

/-- Checks if the sums on both sides of a line are equal --/
def is_line_balanced (line : DividingLine) : Bool :=
  (line.left_regions.map Region.value).sum = (line.right_regions.map Region.value).sum

/-- Checks if all lines in the configuration are balanced --/
def is_configuration_valid (config : CircleConfiguration) (lines : List DividingLine) : Bool :=
  lines.all is_line_balanced

/-- Theorem: There exists a valid configuration for distributing numbers 1 to 7 in a circle divided by 3 lines --/
theorem circle_division_theorem :
  ∃ (config : CircleConfiguration) (lines : List DividingLine),
    config.length = 7 ∧
    (∀ n, n ∈ config.map Region.value → n ∈ List.range 7) ∧
    lines.length = 3 ∧
    is_configuration_valid config lines :=
sorry

end circle_division_theorem_l2468_246894


namespace A_3_2_l2468_246859

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2 : A 3 2 = 29 := by sorry

end A_3_2_l2468_246859


namespace smallest_solution_for_floor_equation_l2468_246813

theorem smallest_solution_for_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^2⌋ : ℝ) - x * (⌊x⌋ : ℝ) = 10 ∧
  ∀ (y : ℝ), y > 0 → (⌊y^2⌋ : ℝ) - y * (⌊y⌋ : ℝ) = 10 → y ≥ x :=
by
  use 131 / 11
  sorry

end smallest_solution_for_floor_equation_l2468_246813


namespace negation_of_existence_negation_of_quadratic_inequality_l2468_246845

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x < 0, f x ≥ 0) ↔ (∀ x < 0, f x < 0) :=
by
  sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x < 0, x^2 - 3*x + 1 ≥ 0) ↔ (∀ x < 0, x^2 - 3*x + 1 < 0) :=
by
  sorry

end negation_of_existence_negation_of_quadratic_inequality_l2468_246845


namespace intercepted_arc_measure_l2468_246808

/-- An equilateral triangle with a circle rolling along its side -/
structure TriangleWithCircle where
  /-- Side length of the equilateral triangle -/
  side : ℝ
  /-- Radius of the circle (equal to the height of the triangle) -/
  radius : ℝ
  /-- The radius is equal to the height of the equilateral triangle -/
  height_eq_radius : radius = side * Real.sqrt 3 / 2

/-- The theorem stating that the intercepted arc measure is 60° -/
theorem intercepted_arc_measure (tc : TriangleWithCircle) :
  let arc_measure := Real.pi / 3  -- 60° in radians
  ∃ (center : ℝ × ℝ) (point_on_side : ℝ × ℝ),
    arc_measure = Real.arccos ((point_on_side.1 - center.1) / tc.radius) :=
sorry

end intercepted_arc_measure_l2468_246808


namespace min_value_of_f_l2468_246875

/-- The function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 3 := by
  sorry

end min_value_of_f_l2468_246875


namespace manuscript_cost_theorem_l2468_246841

/-- The total cost to copy and bind multiple manuscripts. -/
def total_cost (num_copies : ℕ) (pages_per_copy : ℕ) (copy_cost_per_page : ℚ) (binding_cost_per_copy : ℚ) : ℚ :=
  num_copies * (pages_per_copy * copy_cost_per_page + binding_cost_per_copy)

/-- Theorem stating the total cost for the given manuscript copying and binding scenario. -/
theorem manuscript_cost_theorem :
  total_cost 10 400 (5 / 100) 5 = 250 := by
  sorry

end manuscript_cost_theorem_l2468_246841


namespace inverse_proportion_inequality_l2468_246820

/-- Given an inverse proportion function f(x) = -6/x, prove that y₁ < y₂ 
    where (2, y₁) and (-1, y₂) lie on the graph of f. -/
theorem inverse_proportion_inequality (y₁ y₂ : ℝ) : 
  y₁ = -6/2 → y₂ = -6/(-1) → y₁ < y₂ := by
  sorry

end inverse_proportion_inequality_l2468_246820


namespace arithmetic_geometric_sequence_l2468_246840

/-- Given an arithmetic sequence with common difference 2 where a₁, a₂, a₅ form a geometric sequence, prove a₂ = 3 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 1 * a 5 = a 2 * a 2) →     -- a₁, a₂, a₅ form a geometric sequence
  a 2 = 3 := by
sorry

end arithmetic_geometric_sequence_l2468_246840


namespace football_progress_l2468_246888

def round1 : Int := -5
def round2 : Int := 9
def round3 : Int := -12
def round4 : Int := 17
def round5 : Int := -15
def round6 : Int := 24
def round7 : Int := -7

def overall_progress : Int := round1 + round2 + round3 + round4 + round5 + round6 + round7

theorem football_progress : overall_progress = 11 := by
  sorry

end football_progress_l2468_246888


namespace waiter_tables_problem_l2468_246833

theorem waiter_tables_problem (initial_tables : ℝ) : 
  (initial_tables - 12.0) * 8.0 = 256 → initial_tables = 44.0 := by
  sorry

end waiter_tables_problem_l2468_246833


namespace fifth_term_of_geometric_sequence_l2468_246872

/-- Given a geometric sequence with first term a₁ and common ratio r,
    a_n represents the nth term of the sequence. -/
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem fifth_term_of_geometric_sequence :
  let a₁ : ℝ := 5
  let r : ℝ := -2
  geometric_sequence a₁ r 5 = 80 := by sorry

end fifth_term_of_geometric_sequence_l2468_246872


namespace divisibility_by_thirty_l2468_246843

theorem divisibility_by_thirty (n : ℕ) (h_prime : Nat.Prime n) (h_geq_7 : n ≥ 7) :
  30 ∣ (n^2 - 1) := by
  sorry

end divisibility_by_thirty_l2468_246843


namespace max_sum_of_product_3003_l2468_246879

theorem max_sum_of_product_3003 :
  ∀ A B C : ℕ+,
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 3003 →
  A + B + C ≤ 105 :=
by sorry

end max_sum_of_product_3003_l2468_246879


namespace shirt_price_is_correct_l2468_246832

/-- The price of a shirt and sweater with given conditions -/
def shirt_price (total_cost sweater_price : ℝ) : ℝ :=
  let shirt_price := sweater_price - 7.43
  let discounted_sweater_price := sweater_price * 0.9
  shirt_price

theorem shirt_price_is_correct (total_cost sweater_price : ℝ) :
  total_cost = 80.34 ∧ 
  shirt_price total_cost sweater_price + sweater_price * 0.9 = total_cost →
  shirt_price total_cost sweater_price = 38.76 :=
by
  sorry

#eval shirt_price 80.34 46.19

end shirt_price_is_correct_l2468_246832


namespace lindas_family_women_without_daughters_l2468_246877

/-- Represents the family structure of Linda and her descendants -/
structure Family where
  total_daughters_and_granddaughters : Nat
  lindas_daughters : Nat
  daughters_with_five_children : Nat

/-- The number of women (daughters and granddaughters) who have no daughters in Linda's family -/
def women_without_daughters (f : Family) : Nat :=
  f.total_daughters_and_granddaughters - f.daughters_with_five_children

/-- Theorem stating the number of women without daughters in Linda's specific family situation -/
theorem lindas_family_women_without_daughters :
  ∀ f : Family,
  f.total_daughters_and_granddaughters = 43 →
  f.lindas_daughters = 8 →
  f.daughters_with_five_children * 5 = f.total_daughters_and_granddaughters - f.lindas_daughters →
  women_without_daughters f = 36 := by
  sorry


end lindas_family_women_without_daughters_l2468_246877


namespace solution_set_implies_m_value_l2468_246817

theorem solution_set_implies_m_value (m : ℝ) 
  (h : ∀ x : ℝ, x - m > 5 ↔ x > 2) : 
  m = -3 := by
  sorry

end solution_set_implies_m_value_l2468_246817


namespace hyperbola_eccentricity_l2468_246895

/-- Represents a hyperbola with semi-major axis a, semi-minor axis b, and focal length c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0
  h_equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  h_focal_length : c = 2 * c
  h_foci_to_asymptotes : b = c / 2

/-- The eccentricity of a hyperbola is 2√3/3 given the conditions -/
theorem hyperbola_eccentricity (C : Hyperbola) : 
  Real.sqrt ((C.c^2) / (C.a^2)) = 2 * Real.sqrt 3 / 3 := by
  sorry

end hyperbola_eccentricity_l2468_246895


namespace precious_stones_count_l2468_246855

theorem precious_stones_count (N : ℕ) (W : ℝ) : 
  (N > 0) →
  (W > 0) →
  (0.35 * W = 3 * (W / N)) →
  (5/13 * (0.65 * W) = 3 * ((0.65 * W) / (N - 3))) →
  N = 10 := by
sorry

end precious_stones_count_l2468_246855


namespace triangle_problem_l2468_246856

open Real

noncomputable def f (x : ℝ) := 2 * (cos x)^2 + sin (2*x - π/6)

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  f A = 3/2 →
  b + c = 2 →
  (∀ x, f x ≤ 2) ∧
  A = π/3 ∧
  1 ≤ a ∧ a < 2 :=
by sorry

end triangle_problem_l2468_246856


namespace first_shift_participation_is_twenty_percent_l2468_246861

/-- Represents a company with three shifts of employees and a pension program. -/
structure Company where
  first_shift : ℕ
  second_shift : ℕ
  third_shift : ℕ
  second_shift_participation : ℚ
  third_shift_participation : ℚ
  total_participation : ℚ

/-- The percentage of first shift employees participating in the pension program. -/
def first_shift_participation (c : Company) : ℚ :=
  let total_employees := c.first_shift + c.second_shift + c.third_shift
  let total_participants := (c.total_participation * total_employees) / 100
  let second_shift_participants := (c.second_shift_participation * c.second_shift) / 100
  let third_shift_participants := (c.third_shift_participation * c.third_shift) / 100
  let first_shift_participants := total_participants - second_shift_participants - third_shift_participants
  (first_shift_participants * 100) / c.first_shift

theorem first_shift_participation_is_twenty_percent (c : Company) 
  (h1 : c.first_shift = 60)
  (h2 : c.second_shift = 50)
  (h3 : c.third_shift = 40)
  (h4 : c.second_shift_participation = 40)
  (h5 : c.third_shift_participation = 10)
  (h6 : c.total_participation = 24) :
  first_shift_participation c = 20 := by
  sorry

end first_shift_participation_is_twenty_percent_l2468_246861


namespace system_equation_ratio_l2468_246889

theorem system_equation_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (eq1 : 8 * x - 6 * y = c) (eq2 : 12 * y - 18 * x = d) : c / d = -4 / 9 := by
  sorry

end system_equation_ratio_l2468_246889


namespace circle_equation_is_correct_l2468_246858

/-- A circle with center on the y-axis, radius 1, passing through (1, 3) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_y_axis : center.1 = 0
  radius_is_one : radius = 1
  point_on_circle : (passes_through.1 - center.1)^2 + (passes_through.2 - center.2)^2 = radius^2

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  x^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_equation_is_correct (c : Circle) (h : c.passes_through = (1, 3)) :
  ∀ x y : ℝ, circle_equation c x y ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
sorry

end circle_equation_is_correct_l2468_246858


namespace cubic_expression_factorization_l2468_246824

theorem cubic_expression_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (-(x*y + x*z + y*z)) := by
  sorry

end cubic_expression_factorization_l2468_246824


namespace unique_solution_for_rational_equation_l2468_246805

theorem unique_solution_for_rational_equation :
  ∃! x : ℝ, x ≠ 3 ∧ (x^2 - 9) / (x - 3) = 3 * x :=
by
  -- The unique solution is x = 3/2
  use 3/2
  -- Proof goes here
  sorry

end unique_solution_for_rational_equation_l2468_246805


namespace tangent_point_x_coordinate_l2468_246854

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a / Real.exp x

theorem tangent_point_x_coordinate 
  (a : ℝ) 
  (h_even : ∀ x, f a x = f a (-x)) 
  (h_slope : ∃ x, (deriv (f a)) x = 3/2) :
  ∃ x, (deriv (f a)) x = 3/2 ∧ x = Real.log 2 :=
sorry

end tangent_point_x_coordinate_l2468_246854


namespace roses_cut_l2468_246881

theorem roses_cut (initial_roses final_roses : ℕ) : 
  initial_roses = 6 → final_roses = 16 → final_roses - initial_roses = 10 := by
  sorry

end roses_cut_l2468_246881


namespace geometric_series_problem_l2468_246815

theorem geometric_series_problem (b₁ q : ℝ) (h_decrease : |q| < 1) : 
  (b₁ / (1 - q^2) = 2 + b₁ * q / (1 - q^2)) →
  (b₁^2 / (1 - q^4) - b₁^2 * q^2 / (1 - q^4) = 36/5) →
  (b₁ = 3 ∧ q = 1/2) := by
sorry

end geometric_series_problem_l2468_246815


namespace carousel_horse_ratio_l2468_246873

/-- The carousel problem -/
theorem carousel_horse_ratio :
  let blue_horses : ℕ := 3
  let purple_horses : ℕ := 3 * blue_horses
  let green_horses : ℕ := 2 * purple_horses
  let total_horses : ℕ := 33
  let gold_horses : ℕ := total_horses - (blue_horses + purple_horses + green_horses)
  (gold_horses : ℚ) / green_horses = 1 / 6 := by
  sorry

end carousel_horse_ratio_l2468_246873


namespace applicant_a_wins_l2468_246811

/-- Represents an applicant with their test scores -/
structure Applicant where
  education : ℝ
  experience : ℝ
  work_attitude : ℝ

/-- Calculates the final score of an applicant given the weights -/
def final_score (a : Applicant) (w_edu w_exp w_att : ℝ) : ℝ :=
  a.education * w_edu + a.experience * w_exp + a.work_attitude * w_att

/-- Theorem stating that Applicant A's final score is higher than Applicant B's -/
theorem applicant_a_wins (applicant_a applicant_b : Applicant)
    (h_a_edu : applicant_a.education = 7)
    (h_a_exp : applicant_a.experience = 8)
    (h_a_att : applicant_a.work_attitude = 9)
    (h_b_edu : applicant_b.education = 10)
    (h_b_exp : applicant_b.experience = 7)
    (h_b_att : applicant_b.work_attitude = 8) :
    final_score applicant_a (1/6) (1/3) (1/2) > final_score applicant_b (1/6) (1/3) (1/2) := by
  sorry

end applicant_a_wins_l2468_246811


namespace two_zeros_cubic_l2468_246865

def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

theorem two_zeros_cubic (c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ f c x = 0 ∧ f c y = 0 ∧ ∀ z : ℝ, f c z = 0 → z = x ∨ z = y) → 
  c = -2 ∨ c = 2 := by
sorry

end two_zeros_cubic_l2468_246865


namespace stratified_sampling_theorem_l2468_246893

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  sample_size : ℕ
  first_grade_sample : ℕ
  second_grade_sample : ℕ
  third_grade_sample : ℕ

/-- The stratified sampling theorem -/
theorem stratified_sampling_theorem (s : School) 
  (h1 : s.sample_size = 45)
  (h2 : s.first_grade_sample = 20)
  (h3 : s.third_grade_sample = 10)
  (h4 : s.second_grade = 300)
  (h5 : s.sample_size = s.first_grade_sample + s.second_grade_sample + s.third_grade_sample)
  (h6 : s.total_students = s.first_grade + s.second_grade + s.third_grade)
  (h7 : s.second_grade_sample / s.second_grade = s.sample_size / s.total_students) :
  s.total_students = 900 := by
  sorry


end stratified_sampling_theorem_l2468_246893


namespace range_of_a_l2468_246887

theorem range_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x + a * (y - 2 * Real.exp 1 * x) * (Real.log y - Real.log x) = 0) : 
  a < 0 ∨ a ≥ 2 / Real.exp 1 := by
sorry

end range_of_a_l2468_246887


namespace min_cans_for_drinks_l2468_246801

/-- Represents the available can sizes in liters -/
inductive CanSize
  | half
  | one
  | two

/-- Calculates the number of cans needed for a given volume and can size -/
def cansNeeded (volume : ℕ) (size : CanSize) : ℕ :=
  match size with
  | CanSize.half => volume * 2
  | CanSize.one => volume
  | CanSize.two => volume / 2

/-- Finds the minimum number of cans needed for a given volume -/
def minCansForVolume (volume : ℕ) : ℕ :=
  min (cansNeeded volume CanSize.half)
    (min (cansNeeded volume CanSize.one)
      (cansNeeded volume CanSize.two))

/-- The main theorem stating the minimum number of cans required -/
theorem min_cans_for_drinks :
  minCansForVolume 60 +
  minCansForVolume 220 +
  minCansForVolume 500 +
  minCansForVolume 315 +
  minCansForVolume 125 = 830 := by
  sorry


end min_cans_for_drinks_l2468_246801


namespace fault_line_movement_l2468_246886

/-- Fault line movement problem -/
theorem fault_line_movement 
  (total_movement : ℝ) 
  (past_year_movement : ℝ) 
  (h1 : total_movement = 6.5)
  (h2 : past_year_movement = 1.25) :
  total_movement - past_year_movement = 5.25 := by
  sorry

end fault_line_movement_l2468_246886


namespace expected_value_is_three_l2468_246866

/-- Represents the outcome of rolling a six-sided dice -/
inductive DiceOutcome
  | Two
  | Five
  | Other

/-- The probability of each dice outcome -/
def probability (outcome : DiceOutcome) : ℚ :=
  match outcome with
  | DiceOutcome.Two => 1/4
  | DiceOutcome.Five => 1/2
  | DiceOutcome.Other => 1/12

/-- The payoff for each dice outcome in dollars -/
def payoff (outcome : DiceOutcome) : ℚ :=
  match outcome with
  | DiceOutcome.Two => 4
  | DiceOutcome.Five => 6
  | DiceOutcome.Other => -3

/-- The expected value of rolling the dice once -/
def expectedValue : ℚ :=
  (probability DiceOutcome.Two * payoff DiceOutcome.Two) +
  (probability DiceOutcome.Five * payoff DiceOutcome.Five) +
  (4 * probability DiceOutcome.Other * payoff DiceOutcome.Other)

theorem expected_value_is_three :
  expectedValue = 3 := by
  sorry

end expected_value_is_three_l2468_246866


namespace jacks_initial_yen_l2468_246864

/-- Represents Jack's currency holdings and exchange rates -/
structure CurrencyHolding where
  pounds : ℕ
  euros : ℕ
  total_yen : ℕ
  pounds_per_euro : ℕ
  yen_per_pound : ℕ

/-- Calculates the initial yen amount given Jack's currency holding -/
def initial_yen (holding : CurrencyHolding) : ℕ :=
  holding.total_yen - (holding.pounds * holding.yen_per_pound + holding.euros * holding.pounds_per_euro * holding.yen_per_pound)

/-- Theorem stating that Jack's initial yen is 3000 given the problem conditions -/
theorem jacks_initial_yen :
  let jack : CurrencyHolding := {
    pounds := 42,
    euros := 11,
    total_yen := 9400,
    pounds_per_euro := 2,
    yen_per_pound := 100
  }
  initial_yen jack = 3000 := by sorry

end jacks_initial_yen_l2468_246864


namespace prob_even_odd_is_one_fourth_l2468_246871

/-- Represents a six-sided die -/
def Die := Fin 6

/-- The probability of rolling an even number on a six-sided die -/
def prob_even (d : Die) : ℚ := 1/2

/-- The probability of rolling an odd number on a six-sided die -/
def prob_odd (d : Die) : ℚ := 1/2

/-- The probability of rolling an even number on the first die and an odd number on the second die -/
def prob_even_odd (d1 d2 : Die) : ℚ := prob_even d1 * prob_odd d2

theorem prob_even_odd_is_one_fourth (d1 d2 : Die) :
  prob_even_odd d1 d2 = 1/4 := by
  sorry

end prob_even_odd_is_one_fourth_l2468_246871


namespace quadratic_inequality_solution_l2468_246842

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, (-x^2 + c*x - 9 < -4) ↔ (x < 2 ∨ x > 7)) → c = 9 := by
sorry

end quadratic_inequality_solution_l2468_246842


namespace fantasy_ball_handshakes_l2468_246876

/-- The number of goblins attending the Fantasy Creatures Ball -/
def num_goblins : ℕ := 30

/-- The number of pixies attending the Fantasy Creatures Ball -/
def num_pixies : ℕ := 10

/-- Represents whether pixies can shake hands with a given number of goblins -/
def pixie_can_shake (n : ℕ) : Prop := Even n

/-- Calculates the number of handshakes between goblins -/
def goblin_handshakes (n : ℕ) : ℕ := n.choose 2

/-- Calculates the number of handshakes between goblins and pixies -/
def goblin_pixie_handshakes (g p : ℕ) : ℕ := g * p

/-- The total number of handshakes at the Fantasy Creatures Ball -/
def total_handshakes : ℕ := goblin_handshakes num_goblins + goblin_pixie_handshakes num_goblins num_pixies

theorem fantasy_ball_handshakes :
  pixie_can_shake num_goblins →
  total_handshakes = 735 := by
  sorry

end fantasy_ball_handshakes_l2468_246876


namespace average_time_per_flower_l2468_246838

/-- Proves that the average time to find a flower is 10 minutes -/
theorem average_time_per_flower 
  (total_time : ℕ) 
  (total_flowers : ℕ) 
  (h1 : total_time = 330) 
  (h2 : total_flowers = 33) 
  (h3 : total_time % total_flowers = 0) :
  total_time / total_flowers = 10 := by
  sorry

#check average_time_per_flower

end average_time_per_flower_l2468_246838


namespace least_multiplier_for_72_l2468_246857

theorem least_multiplier_for_72 (n : ℕ) : n = 62087668 ↔ 
  n > 0 ∧
  (∀ m : ℕ, m > 0 → m < n →
    (¬(112 ∣ (72 * m)) ∨
     ¬(199 ∣ (72 * m)) ∨
     ¬∃ k : ℕ, 72 * m = k * k)) ∧
  (112 ∣ (72 * n)) ∧
  (199 ∣ (72 * n)) ∧
  ∃ k : ℕ, 72 * n = k * k :=
sorry

end least_multiplier_for_72_l2468_246857


namespace long_division_puzzle_l2468_246822

theorem long_division_puzzle :
  (631938 : ℚ) / 625 = 1011.1008 := by
  sorry

end long_division_puzzle_l2468_246822


namespace linear_function_theorem_l2468_246892

/-- A linear function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem linear_function_theorem (x : ℝ) :
  (∀ x y : ℝ, ∃ a b : ℝ, f x = a * x + b) →  -- f is linear
  (∀ x : ℝ, f x = 3 * (f⁻¹ x) + 5) →        -- f(x) = 3f^(-1)(x) + 5
  f 0 = 3 →                                 -- f(0) = 3
  f 3 = 3 * Real.sqrt 3 + 3 :=               -- f(3) = 3√3 + 3
by sorry

end linear_function_theorem_l2468_246892


namespace rhombus_area_l2468_246853

/-- The area of a rhombus with side length 25 and one diagonal of 30 is 600 -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) :
  side = 25 →
  diagonal1 = 30 →
  diagonal2 = 2 * Real.sqrt (side^2 - (diagonal1 / 2)^2) →
  (diagonal1 * diagonal2) / 2 = 600 := by
  sorry

end rhombus_area_l2468_246853


namespace nonCubeSequence_250th_term_l2468_246880

/-- Function that determines if a positive integer is a perfect cube --/
def isPerfectCube (n : ℕ+) : Prop :=
  ∃ m : ℕ+, n = m^3

/-- The sequence of positive integers omitting perfect cubes --/
def nonCubeSequence : ℕ+ → ℕ+ :=
  sorry

/-- The 250th term of the sequence is 256 --/
theorem nonCubeSequence_250th_term :
  nonCubeSequence 250 = 256 := by
  sorry

end nonCubeSequence_250th_term_l2468_246880


namespace product_of_trig_expressions_l2468_246851

theorem product_of_trig_expressions :
  (1 - Real.sin (π / 8)) * (1 - Real.sin (3 * π / 8)) *
  (1 + Real.sin (π / 8)) * (1 + Real.sin (3 * π / 8)) = 1 / 8 := by
  sorry

end product_of_trig_expressions_l2468_246851


namespace quadratic_function_properties_l2468_246863

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * x - 3

-- State the theorem
theorem quadratic_function_properties :
  ∀ m : ℝ,
  (m > 0) →
  (∀ x : ℝ, f m x < 0 ↔ -1 < x ∧ x < 3) →
  (m = 1) ∧
  (∀ x : ℝ, 2 * x^2 - 4 * x + 3 > 2 * x - 1 ↔ x < 1 ∨ x > 2) ∧
  (∃ a : ℝ, 0 < a ∧ a < 1 ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f m (a^x) - 4 * a^(x+1) ≥ -4) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f m (a^x) - 4 * a^(x+1) = -4) ∧
    a = 1/3) :=
by sorry

end quadratic_function_properties_l2468_246863


namespace equation_pattern_find_a_b_l2468_246860

theorem equation_pattern (n : ℕ) (hn : n ≥ 2) :
  Real.sqrt (n + n / (n^2 - 1)) = n * Real.sqrt (n / (n^2 - 1)) := by sorry

theorem find_a_b (a b : ℝ) (h : Real.sqrt (6 + a / b) = 6 * Real.sqrt (a / b)) :
  a = 6 ∧ b = 35 := by sorry

end equation_pattern_find_a_b_l2468_246860


namespace complex_modulus_l2468_246874

theorem complex_modulus (z : ℂ) (h : (2 - Complex.I) * z = 4 + 3 * Complex.I) : 
  Complex.abs (z - Complex.I) = Real.sqrt 2 := by
  sorry

end complex_modulus_l2468_246874


namespace square_difference_of_solutions_l2468_246804

theorem square_difference_of_solutions (α β : ℝ) : 
  α ≠ β ∧ α^2 = 3*α + 1 ∧ β^2 = 3*β + 1 → (α - β)^2 = 13 := by
  sorry

end square_difference_of_solutions_l2468_246804


namespace roots_quadratic_equation_l2468_246847

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 + 2*a - 5 = 0) → (b^2 + 2*b - 5 = 0) → (a^2 + a*b + 2*a = 0) := by
  sorry

end roots_quadratic_equation_l2468_246847


namespace B_subset_A_l2468_246870

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 + 2*x = 0}

theorem B_subset_A : B ⊆ A := by sorry

end B_subset_A_l2468_246870


namespace set_operations_and_range_l2468_246806

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

-- State the theorem
theorem set_operations_and_range :
  (∃ (a : ℝ),
    (A ∩ B = {x | 2 ≤ x ∧ x < 3}) ∧
    ((U \ A) ∪ B = {x | x < -1 ∨ x ≥ 2}) ∧
    (B ∪ C a = C a → a > 4)) := by sorry

end set_operations_and_range_l2468_246806


namespace symmetry_implies_periodicity_l2468_246862

/-- A function is symmetric about a line x = a -/
def SymmetricAboutLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

/-- A function is symmetric about a point (m, n) -/
def SymmetricAboutPoint (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  ∀ x, 2 * n - f x = f (2 * m - x)

/-- A function is periodic with period p -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem symmetry_implies_periodicity
  (f : ℝ → ℝ) (a m n : ℝ) (ha : a ≠ 0) (hm : m ≠ a)
  (h_line : SymmetricAboutLine f a)
  (h_point : SymmetricAboutPoint f m n) :
  IsPeriodic f (4 * (m - a)) :=
sorry

end symmetry_implies_periodicity_l2468_246862


namespace lara_flowers_in_vase_l2468_246814

def flowers_in_vase (total_flowers mom_flowers grandma_extra : ℕ) : ℕ :=
  total_flowers - (mom_flowers + (mom_flowers + grandma_extra))

theorem lara_flowers_in_vase :
  flowers_in_vase 52 15 6 = 16 := by
  sorry

end lara_flowers_in_vase_l2468_246814


namespace saree_stripe_ratio_l2468_246897

theorem saree_stripe_ratio (brown_stripes : ℕ) (blue_stripes : ℕ) (gold_stripes : ℕ) :
  brown_stripes = 4 →
  gold_stripes = 3 * brown_stripes →
  blue_stripes = 60 →
  blue_stripes = gold_stripes →
  blue_stripes / gold_stripes = 5 / 1 :=
by
  sorry

end saree_stripe_ratio_l2468_246897


namespace arithmetic_sequence_a6_l2468_246819

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 4)^2 - 6*(a 4) + 5 = 0 →
  (a 8)^2 - 6*(a 8) + 5 = 0 →
  a 6 = 3 := by
sorry

end arithmetic_sequence_a6_l2468_246819


namespace monday_rainfall_calculation_l2468_246839

/-- The rainfall on Monday in inches -/
def monday_rainfall : ℝ := sorry

/-- The rainfall on Tuesday in inches -/
def tuesday_rainfall : ℝ := 0.2

/-- The difference in rainfall between Monday and Tuesday in inches -/
def rainfall_difference : ℝ := 0.7

theorem monday_rainfall_calculation : monday_rainfall = 0.9 := by
  sorry

end monday_rainfall_calculation_l2468_246839


namespace subtraction_relation_l2468_246826

theorem subtraction_relation (minuend subtrahend difference : ℝ) 
  (h : subtrahend + difference = minuend) : 
  (minuend + subtrahend + difference) / minuend = 2 := by
sorry

end subtraction_relation_l2468_246826


namespace hall_reunion_attendance_l2468_246825

/-- The number of people attending the Hall reunion -/
def hall_attendees (total_guests oates_attendees both_attendees : ℕ) : ℕ :=
  total_guests - (oates_attendees - both_attendees)

/-- Theorem stating the number of people attending the Hall reunion -/
theorem hall_reunion_attendance 
  (total_guests : ℕ) 
  (oates_attendees : ℕ) 
  (both_attendees : ℕ) 
  (h1 : total_guests = 100) 
  (h2 : oates_attendees = 40) 
  (h3 : both_attendees = 10) 
  (h4 : total_guests ≥ oates_attendees) 
  (h5 : oates_attendees ≥ both_attendees) : 
  hall_attendees total_guests oates_attendees both_attendees = 70 := by
  sorry

end hall_reunion_attendance_l2468_246825


namespace solution_set_of_inequality_l2468_246884

theorem solution_set_of_inequality (x : ℝ) :
  (3 * x^2 - 7 * x > 6) ↔ (x < -2/3 ∨ x > 3) := by sorry

end solution_set_of_inequality_l2468_246884


namespace max_distance_AB_l2468_246816

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the point M
def M : ℝ × ℝ := (2, 0)

-- Define the line passing through M and intersecting C at A and B
def line_through_M (k : ℝ) (x : ℝ) : ℝ := k * (x - 2)

-- Define the condition for A and B being on both the line and the ellipse
def A_B_on_line_and_C (k x y : ℝ) : Prop :=
  C x y ∧ y = line_through_M k x

-- Define the vector addition condition
def vector_addition_condition (xA yA xB yB xP yP t : ℝ) : Prop :=
  xA + xB = t * xP ∧ yA + yB = t * yP

-- Main theorem
theorem max_distance_AB :
  ∀ (k xA yA xB yB xP yP t : ℝ),
    A_B_on_line_and_C k xA yA →
    A_B_on_line_and_C k xB yB →
    C xP yP →
    vector_addition_condition xA yA xB yB xP yP t →
    2 * Real.sqrt 6 / 3 < t →
    t < 2 →
    ∃ (max_dist : ℝ), max_dist = 2 * Real.sqrt 5 / 3 ∧
      ((xA - xB)^2 + (yA - yB)^2)^(1/2 : ℝ) ≤ max_dist :=
by sorry

end max_distance_AB_l2468_246816


namespace bryan_books_per_continent_l2468_246883

/-- The number of continents Bryan visited -/
def num_continents : ℕ := 4

/-- The total number of books Bryan collected -/
def total_books : ℕ := 488

/-- The number of books Bryan collected per continent -/
def books_per_continent : ℕ := total_books / num_continents

/-- Theorem stating that Bryan collected 122 books per continent -/
theorem bryan_books_per_continent : books_per_continent = 122 := by
  sorry

end bryan_books_per_continent_l2468_246883


namespace hyperbola_equation_from_properties_l2468_246803

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The asymptote of a hyperbola -/
def asymptote (h : Hyperbola) (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x

/-- The focus of the parabola y^2 = 16x -/
def parabola_focus : ℝ × ℝ := (4, 0)

/-- Theorem: If a hyperbola has the given properties, its equation is x^2/4 - y^2/12 = 1 -/
theorem hyperbola_equation_from_properties (h : Hyperbola) :
  (∃ x y : ℝ, asymptote h x y) →
  (∃ x y : ℝ, hyperbola_equation h x y ∧ (x, y) = parabola_focus) →
  (∀ x y : ℝ, hyperbola_equation h x y ↔ x^2/4 - y^2/12 = 1) :=
by sorry

end hyperbola_equation_from_properties_l2468_246803


namespace ratio_to_percent_l2468_246882

theorem ratio_to_percent (a b : ℕ) (h : a = 6 ∧ b = 3) :
  (a : ℚ) / b * 100 = 200 := by
  sorry

end ratio_to_percent_l2468_246882


namespace armband_break_even_l2468_246836

/-- The cost of an individual ticket in dollars -/
def individual_ticket_cost : ℚ := 3/4

/-- The cost of an armband in dollars -/
def armband_cost : ℚ := 15

/-- The number of rides at which the armband cost equals the cost of individual tickets -/
def break_even_rides : ℕ := 20

theorem armband_break_even :
  (individual_ticket_cost * break_even_rides : ℚ) = armband_cost :=
by sorry

end armband_break_even_l2468_246836


namespace function_minimum_value_equality_condition_l2468_246810

theorem function_minimum_value (x : ℝ) (h : x > 0) : x^2 + 2/x ≥ 3 := by sorry

theorem equality_condition (x : ℝ) (h : x > 0) : x^2 + 2/x = 3 ↔ x = 1 := by sorry

end function_minimum_value_equality_condition_l2468_246810


namespace three_digit_numbers_property_l2468_246850

theorem three_digit_numbers_property : 
  (∃! (l : List Nat), 
    (∀ n ∈ l, 100 ≤ n ∧ n < 1000) ∧ 
    (∀ n ∈ l, let a := n / 100
              let b := (n / 10) % 10
              let c := n % 10
              10 * a + c = (100 * a + 10 * b + c) / 9) ∧
    l.length = 4) := by sorry

end three_digit_numbers_property_l2468_246850


namespace set_operations_l2468_246830

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set Nat := {4, 5, 6, 7, 8, 9}
def B : Set Nat := {1, 2, 3, 4, 5, 6}

theorem set_operations :
  (A ∪ B = U) ∧
  (A ∩ B = {4, 5, 6}) ∧
  (U \ (A ∩ B) = {1, 2, 3, 7, 8, 9}) := by
  sorry

end set_operations_l2468_246830


namespace road_repaving_l2468_246867

/-- Given that a construction company repaved 4133 inches of road before today
    and 805 inches today, prove that the total length of road repaved is 4938 inches. -/
theorem road_repaving (inches_before : ℕ) (inches_today : ℕ) 
  (h1 : inches_before = 4133) (h2 : inches_today = 805) :
  inches_before + inches_today = 4938 := by
  sorry

end road_repaving_l2468_246867


namespace sequence_partition_sequence_partition_general_l2468_246812

-- Define the type for our sequence
def Sequence := ℕ → Set ℝ

-- Define what it means for a sequence to be in [0, 1)
def InUnitInterval (s : Sequence) : Prop :=
  ∀ n, ∀ x ∈ s n, 0 ≤ x ∧ x < 1

-- Define what it means for a set to contain infinitely many elements of a sequence
def ContainsInfinitelyMany (A : Set ℝ) (s : Sequence) : Prop :=
  ∀ N, ∃ n ≥ N, ∃ x ∈ s n, x ∈ A

theorem sequence_partition (s : Sequence) (h : InUnitInterval s) :
  ContainsInfinitelyMany (Set.Icc 0 (1/2)) s ∨ ContainsInfinitelyMany (Set.Ico (1/2) 1) s :=
sorry

theorem sequence_partition_general (s : Sequence) (h : InUnitInterval s) :
  ∀ n : ℕ, n ≥ 1 →
    ∃ k : ℕ, k < 2^n ∧
      ContainsInfinitelyMany (Set.Ico (k / 2^n) ((k + 1) / 2^n)) s :=
sorry

end sequence_partition_sequence_partition_general_l2468_246812


namespace system_solution_l2468_246818

theorem system_solution (x y : ℝ) : 
  0 < x + y → 
  x + y ≠ 1 → 
  2*x - y ≠ 0 → 
  (x + y) * (2 ^ (y - 2*x)) = 6.25 → 
  (x + y) * (1 / (2*x - y)) = 5 → 
  x = 9 ∧ y = 16 := by
sorry

end system_solution_l2468_246818
