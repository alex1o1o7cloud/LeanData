import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_numbers_l1252_125292

theorem sum_of_numbers (x y : ℝ) (h1 : x - y = 10) (h2 : x * y = 200) : x + y = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1252_125292


namespace NUMINAMATH_CALUDE_m_range_l1252_125221

theorem m_range : ∀ m : ℝ, m = 5 * Real.sqrt (1/5) - Real.sqrt 45 → -5 < m ∧ m < -4 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l1252_125221


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1252_125207

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) / (1 - Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1252_125207


namespace NUMINAMATH_CALUDE_no_double_by_digit_move_l1252_125244

theorem no_double_by_digit_move :
  ¬ ∃ (x : ℕ) (n : ℕ), n ≥ 1 ∧
    (∃ (a : ℕ) (N : ℕ),
      x = a * 10^n + N ∧
      0 < a ∧ a < 10 ∧
      N < 10^n ∧
      10 * N + a = 2 * x) :=
sorry

end NUMINAMATH_CALUDE_no_double_by_digit_move_l1252_125244


namespace NUMINAMATH_CALUDE_liquid_volume_range_l1252_125287

-- Define the cube
def cube_volume : ℝ := 6

-- Define the liquid volume as a real number between 0 and the cube volume
def liquid_volume : ℝ := sorry

-- Define the condition that the liquid surface is not a triangle
def not_triangle_surface : Prop := sorry

-- Theorem statement
theorem liquid_volume_range (h : not_triangle_surface) : 
  1 < liquid_volume ∧ liquid_volume < 5 := by sorry

end NUMINAMATH_CALUDE_liquid_volume_range_l1252_125287


namespace NUMINAMATH_CALUDE_comic_book_problem_l1252_125218

theorem comic_book_problem (x y : ℕ) : 
  (y + 7 = 5 * (x - 7)) →
  (y - 9 = 3 * (x + 9)) →
  (x = 39 ∧ y = 153) := by
sorry

end NUMINAMATH_CALUDE_comic_book_problem_l1252_125218


namespace NUMINAMATH_CALUDE_inequality_proof_l1252_125279

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  1 / (a + b + c) + 1 / (b + c + d) + 1 / (c + d + a) + 1 / (a + b + d) ≥ 4 / (3 * (a + b + c + d)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1252_125279


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_count_l1252_125238

theorem systematic_sampling_interval_count :
  let total_employees : ℕ := 840
  let sample_size : ℕ := 42
  let interval_start : ℕ := 481
  let interval_end : ℕ := 720
  let interval_size : ℕ := interval_end - interval_start + 1
  let sampling_interval : ℕ := total_employees / sample_size
  (interval_size : ℚ) / (total_employees : ℚ) * (sample_size : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_count_l1252_125238


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l1252_125249

theorem increasing_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_sum_positive : a + b > 0) : 
  f a + f b > f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l1252_125249


namespace NUMINAMATH_CALUDE_farmer_brown_animals_legs_l1252_125278

/-- The number of legs for each animal type -/
def chicken_legs : ℕ := 2
def sheep_legs : ℕ := 4
def grasshopper_legs : ℕ := 6
def spider_legs : ℕ := 8

/-- The number of each animal type -/
def num_chickens : ℕ := 7
def num_sheep : ℕ := 5
def num_grasshoppers : ℕ := 10
def num_spiders : ℕ := 3

/-- The total number of legs -/
def total_legs : ℕ := 
  num_chickens * chicken_legs + 
  num_sheep * sheep_legs + 
  num_grasshoppers * grasshopper_legs + 
  num_spiders * spider_legs

theorem farmer_brown_animals_legs : total_legs = 118 := by
  sorry

end NUMINAMATH_CALUDE_farmer_brown_animals_legs_l1252_125278


namespace NUMINAMATH_CALUDE_boat_upstream_speed_l1252_125280

/-- Calculates the upstream speed of a boat given its still water speed and downstream speed. -/
def upstream_speed (still_water_speed downstream_speed : ℝ) : ℝ :=
  2 * still_water_speed - downstream_speed

/-- Proves that a boat with a still water speed of 11 km/hr and a downstream speed of 15 km/hr 
    has an upstream speed of 7 km/hr. -/
theorem boat_upstream_speed :
  upstream_speed 11 15 = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_speed_l1252_125280


namespace NUMINAMATH_CALUDE_peter_large_glasses_bought_l1252_125286

def small_glass_cost : ℕ := 3
def large_glass_cost : ℕ := 5
def initial_amount : ℕ := 50
def small_glasses_bought : ℕ := 8
def change : ℕ := 1

def large_glasses_bought : ℕ := (initial_amount - change - small_glass_cost * small_glasses_bought) / large_glass_cost

theorem peter_large_glasses_bought :
  large_glasses_bought = 5 :=
by sorry

end NUMINAMATH_CALUDE_peter_large_glasses_bought_l1252_125286


namespace NUMINAMATH_CALUDE_annual_income_proof_l1252_125213

/-- Calculates the yearly simple interest income given principal and rate -/
def simple_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

theorem annual_income_proof (total_amount : ℝ) (part1 : ℝ) (rate1 : ℝ) (rate2 : ℝ) 
  (h1 : total_amount = 2500)
  (h2 : part1 = 1000)
  (h3 : rate1 = 0.05)
  (h4 : rate2 = 0.06) :
  simple_interest part1 rate1 + simple_interest (total_amount - part1) rate2 = 140 := by
  sorry

end NUMINAMATH_CALUDE_annual_income_proof_l1252_125213


namespace NUMINAMATH_CALUDE_eighth_term_and_half_l1252_125267

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r ^ (n - 1)

theorem eighth_term_and_half (a : ℚ) (r : ℚ) :
  a = 12 → r = 1/2 →
  geometric_sequence a r 8 = 3/32 ∧
  (1/2 * geometric_sequence a r 8) = 3/64 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_and_half_l1252_125267


namespace NUMINAMATH_CALUDE_irrationality_classification_l1252_125297

-- Define rational numbers
def isRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

-- Define irrational numbers
def isIrrational (x : ℝ) : Prop := ¬ (isRational x)

theorem irrationality_classification :
  isRational (-2) ∧ 
  isRational (1/2) ∧ 
  isIrrational (Real.sqrt 3) ∧ 
  isRational 2 :=
sorry

end NUMINAMATH_CALUDE_irrationality_classification_l1252_125297


namespace NUMINAMATH_CALUDE_charlie_max_success_ratio_l1252_125269

theorem charlie_max_success_ratio 
  (alpha_first_two : ℚ)
  (alpha_last_two : ℚ)
  (charlie_daily : ℕ → ℚ)
  (charlie_attempted : ℕ → ℕ)
  (h1 : alpha_first_two = 120 / 200)
  (h2 : alpha_last_two = 80 / 200)
  (h3 : ∀ i ∈ Finset.range 4, 0 < charlie_daily i)
  (h4 : ∀ i ∈ Finset.range 4, charlie_daily i < 1)
  (h5 : ∀ i ∈ Finset.range 2, charlie_daily i < alpha_first_two)
  (h6 : ∀ i ∈ Finset.range 2, charlie_daily (i + 2) < alpha_last_two)
  (h7 : ∀ i ∈ Finset.range 4, charlie_attempted i > 0)
  (h8 : charlie_attempted 0 + charlie_attempted 1 < 200)
  (h9 : (charlie_attempted 0 + charlie_attempted 1 + charlie_attempted 2 + charlie_attempted 3) = 400)
  : (charlie_daily 0 * charlie_attempted 0 + charlie_daily 1 * charlie_attempted 1 + 
     charlie_daily 2 * charlie_attempted 2 + charlie_daily 3 * charlie_attempted 3) / 400 ≤ 239 / 400 :=
sorry

end NUMINAMATH_CALUDE_charlie_max_success_ratio_l1252_125269


namespace NUMINAMATH_CALUDE_parallel_vectors_l1252_125239

/-- Given vectors in R² -/
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (1, 6)

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 * w.2 = t * v.2 * w.1

/-- The main theorem -/
theorem parallel_vectors (k : ℝ) :
  are_parallel (a.1 + k * c.1, a.2 + k * c.2) (a.1 + b.1, a.2 + b.2) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l1252_125239


namespace NUMINAMATH_CALUDE_expression_evaluation_l1252_125237

theorem expression_evaluation (a : ℝ) (h : a^2 + 2*a - 1 = 0) :
  (((a - 2) / (a^2 + 2*a) - (a - 1) / (a^2 + 4*a + 4)) / ((a - 4) / (a + 2))) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1252_125237


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_negative_four_l1252_125230

/-- Represents a parabola and its transformations -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Applies transformations to a parabola -/
def transform (p : Parabola) : Parabola :=
  { a := -p.a,  -- 180-degree rotation
    h := p.h - 4,  -- 4-unit left shift
    k := p.k - 3 } -- 3-unit downward shift

/-- Calculates the sum of zeros for a parabola -/
def sumOfZeros (p : Parabola) : ℝ := -2 * p.h

/-- Theorem: The sum of zeros of the transformed parabola is -4 -/
theorem sum_of_zeros_is_negative_four :
  let original := Parabola.mk 1 2 3
  let transformed := transform original
  sumOfZeros transformed = -4 := by sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_negative_four_l1252_125230


namespace NUMINAMATH_CALUDE_geometric_sequence_product_property_geometric_sequence_specific_product_l1252_125294

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- In a geometric sequence, the product of terms equidistant from any center term is constant. -/
theorem geometric_sequence_product_property {a : ℕ → ℝ} (h : IsGeometricSequence a) :
    ∀ n k : ℕ, a (n - k) * a (n + k) = (a n) ^ 2 :=
  sorry

/-- Given a geometric sequence {a_n} where a_4 = 4, prove that a_2 * a_6 = 16 -/
theorem geometric_sequence_specific_product (a : ℕ → ℝ) 
    (h1 : IsGeometricSequence a) (h2 : a 4 = 4) : a 2 * a 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_property_geometric_sequence_specific_product_l1252_125294


namespace NUMINAMATH_CALUDE_minimum_fourth_exam_score_l1252_125252

def exam1 : ℕ := 86
def exam2 : ℕ := 82
def exam3 : ℕ := 89
def required_increase : ℚ := 2

def average (a b c d : ℕ) : ℚ := (a + b + c + d : ℚ) / 4

theorem minimum_fourth_exam_score :
  ∀ x : ℕ,
    (average exam1 exam2 exam3 x ≥ (exam1 + exam2 + exam3 : ℚ) / 3 + required_increase) ↔
    x ≥ 94 :=
by sorry

end NUMINAMATH_CALUDE_minimum_fourth_exam_score_l1252_125252


namespace NUMINAMATH_CALUDE_smallest_n_for_divisible_by_20_l1252_125240

theorem smallest_n_for_divisible_by_20 :
  ∃ (n : ℕ), n = 9 ∧ n ≥ 4 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < 9 → m ≥ 4 →
    ∃ (T : Finset ℤ), T.card = m ∧
    ∀ (a b c d : ℤ), a ∈ T → b ∈ T → c ∈ T → d ∈ T →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    ¬(20 ∣ (a + b - c - d))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisible_by_20_l1252_125240


namespace NUMINAMATH_CALUDE_factorial_simplification_l1252_125289

theorem factorial_simplification : (13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / 
  ((10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) + 3 * (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l1252_125289


namespace NUMINAMATH_CALUDE_fraction_value_l1252_125257

theorem fraction_value (a b : ℝ) (h : 1/a - 1/(2*b) = 4) : 
  4*a*b / (a - 2*b) = -1/2 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l1252_125257


namespace NUMINAMATH_CALUDE_smaller_factor_of_4851_l1252_125243

theorem smaller_factor_of_4851 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4851 → 
  min a b = 53 := by
sorry

end NUMINAMATH_CALUDE_smaller_factor_of_4851_l1252_125243


namespace NUMINAMATH_CALUDE_pants_cost_is_250_l1252_125203

/-- The cost of each pair of pants given the total cost of t-shirts and pants, 
    the number of t-shirts and pants, and the cost of each t-shirt. -/
def cost_of_pants (total_cost : ℕ) (num_tshirts : ℕ) (num_pants : ℕ) (tshirt_cost : ℕ) : ℕ :=
  (total_cost - num_tshirts * tshirt_cost) / num_pants

/-- Theorem stating that the cost of each pair of pants is 250 
    given the conditions in the problem. -/
theorem pants_cost_is_250 :
  cost_of_pants 1500 5 4 100 = 250 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_is_250_l1252_125203


namespace NUMINAMATH_CALUDE_three_similar_points_l1252_125248

-- Define the trapezoid ABCD
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define a point P on AB
def PointOnAB (t : Trapezoid) (x : ℝ) : ℝ × ℝ :=
  (x * t.B.1 + (1 - x) * t.A.1, x * t.B.2 + (1 - x) * t.A.2)

-- Define the similarity condition
def IsSimilar (t : Trapezoid) (x : ℝ) : Prop :=
  let P := PointOnAB t x
  ∃ k : ℝ, k > 0 ∧
    (P.1 - t.A.1)^2 + (P.2 - t.A.2)^2 = k * ((t.C.1 - P.1)^2 + (t.C.2 - P.2)^2) ∧
    (t.D.1 - t.A.1)^2 + (t.D.2 - t.A.2)^2 = k * ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)

-- Theorem statement
theorem three_similar_points (t : Trapezoid) 
  (h1 : t.B.1 - t.A.1 = 7) 
  (h2 : t.D.2 - t.A.2 = 2) 
  (h3 : t.C.1 - t.B.1 = 3) 
  (h4 : t.A.2 = t.B.2) 
  (h5 : t.C.2 = t.D.2) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, 0 ≤ x ∧ x ≤ 1 ∧ IsSimilar t x :=
sorry

end NUMINAMATH_CALUDE_three_similar_points_l1252_125248


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1252_125296

theorem rationalize_denominator (x : ℝ) : 
  x > 0 → (7 / Real.sqrt (98 : ℝ)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1252_125296


namespace NUMINAMATH_CALUDE_max_people_satisfying_conditions_l1252_125284

/-- Represents a group of people and their relationships -/
structure PeopleGroup where
  n : ℕ
  knows : Fin n → Fin n → Prop
  knows_sym : ∀ i j, knows i j ↔ knows j i

/-- Any 3 people have at least 2 who know each other -/
def condition_a (g : PeopleGroup) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    g.knows i j ∨ g.knows j k ∨ g.knows i k

/-- Any 4 people have at least 2 who don't know each other -/
def condition_b (g : PeopleGroup) : Prop :=
  ∀ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l →
    ¬g.knows i j ∨ ¬g.knows i k ∨ ¬g.knows i l ∨
    ¬g.knows j k ∨ ¬g.knows j l ∨ ¬g.knows k l

/-- The maximum number of people satisfying both conditions is 8 -/
theorem max_people_satisfying_conditions :
  (∃ g : PeopleGroup, g.n = 8 ∧ condition_a g ∧ condition_b g) ∧
  (∀ g : PeopleGroup, condition_a g ∧ condition_b g → g.n ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_max_people_satisfying_conditions_l1252_125284


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1252_125215

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (1 - x)}

-- Theorem to prove
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1252_125215


namespace NUMINAMATH_CALUDE_workshop_participation_l1252_125247

theorem workshop_participation (total : ℕ) (A B C : ℕ) (at_least_two : ℕ) 
  (h_total : total = 25)
  (h_A : A = 15)
  (h_B : B = 14)
  (h_C : C = 11)
  (h_at_least_two : at_least_two = 12)
  (h_sum : A + B + C ≥ total + at_least_two) :
  ∃ (x y z a b c : ℕ), 
    x + y + z + a + b + c = total ∧
    a + b + c = at_least_two ∧
    x + a + c = A ∧
    y + a + b = B ∧
    z + b + c = C ∧
    0 = total - (x + y + z + a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_workshop_participation_l1252_125247


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l1252_125201

theorem rectangular_field_dimensions (m : ℕ) : 
  (3 * m + 10) * (m - 5) = 72 → m = 7 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l1252_125201


namespace NUMINAMATH_CALUDE_inequality_chain_l1252_125299

theorem inequality_chain (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / a + 1 / b + 1 / c ≥ 2 / (a + b) + 2 / (b + c) + 2 / (c + a) ∧
  2 / (a + b) + 2 / (b + c) + 2 / (c + a) ≥ 9 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l1252_125299


namespace NUMINAMATH_CALUDE_equation_solution_l1252_125255

theorem equation_solution (x y z k : ℝ) :
  (5 / (x - z) = k / (y + z)) ∧ (k / (y + z) = 12 / (x + y)) → k = 17 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1252_125255


namespace NUMINAMATH_CALUDE_min_value_theorem_l1252_125245

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2 * f m x + x^2

noncomputable def h (c b : ℝ) (x : ℝ) : ℝ := Real.log x - c * x^2 - b * x

theorem min_value_theorem (m c b : ℝ) (h_m : m ≥ 3 * Real.sqrt 2 / 2) :
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
  (∀ x, x > 0 → g m x ≤ g m x₁ ∧ g m x ≤ g m x₂) ∧
  h c b x₁ = 0 ∧ h c b x₂ = 0 ∧
  (∀ y, y = (x₁ - x₂) * h c b ((x₁ + x₂) / 2) → y ≥ -2/3 + Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1252_125245


namespace NUMINAMATH_CALUDE_geometric_means_equality_l1252_125228

/-- Represents a quadrilateral with sides a, b, c, and d -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents the geometric means of sides in the quadrilateral -/
structure GeometricMeans (q : Quadrilateral) where
  k : ℝ
  l : ℝ
  m : ℝ
  n : ℝ
  hk : k^2 = q.a * q.d
  hl : l^2 = q.a * q.d
  hm : m^2 = q.b * q.c
  hn : n^2 = q.b * q.c

/-- The main theorem stating the condition for KL = MN -/
theorem geometric_means_equality (q : Quadrilateral) (g : GeometricMeans q) :
  (g.k - g.l)^2 = (g.m - g.n)^2 ↔ (q.a + q.b = q.c + q.d ∨ q.a + q.c = q.b + q.d) :=
by sorry


end NUMINAMATH_CALUDE_geometric_means_equality_l1252_125228


namespace NUMINAMATH_CALUDE_increasing_function_iff_a_in_range_l1252_125209

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem increasing_function_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3/2 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_iff_a_in_range_l1252_125209


namespace NUMINAMATH_CALUDE_dad_eyes_l1252_125259

theorem dad_eyes (mom_eyes : ℕ) (num_kids : ℕ) (kid_eyes : ℕ) (total_eyes : ℕ) : 
  mom_eyes = 1 → 
  num_kids = 3 → 
  kid_eyes = 4 → 
  total_eyes = 16 → 
  total_eyes = mom_eyes + num_kids * kid_eyes + (total_eyes - (mom_eyes + num_kids * kid_eyes)) →
  (total_eyes - (mom_eyes + num_kids * kid_eyes)) = 3 := by
sorry

end NUMINAMATH_CALUDE_dad_eyes_l1252_125259


namespace NUMINAMATH_CALUDE_sports_club_purchase_l1252_125208

/-- The price difference between a basketball and a soccer ball -/
def price_difference : ℕ := 30

/-- The budget for soccer balls -/
def soccer_budget : ℕ := 1500

/-- The budget for basketballs -/
def basketball_budget : ℕ := 2400

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 100

/-- The minimum discount on basketballs -/
def min_discount : ℕ := 25

/-- The maximum discount on basketballs -/
def max_discount : ℕ := 35

/-- The price of a soccer ball -/
def soccer_price : ℕ := 50

/-- The price of a basketball -/
def basketball_price : ℕ := 80

theorem sports_club_purchase :
  ∀ (m : ℕ), min_discount ≤ m → m ≤ max_discount →
  (∃ (y : ℕ), y ≤ total_balls ∧ 3 * (total_balls - y) ≤ y ∧
    (∀ (z : ℕ), z ≤ total_balls → 3 * (total_balls - z) ≤ z →
      (if 30 < m then
        (basketball_price - m) * y + soccer_price * (total_balls - y) ≤ (basketball_price - m) * z + soccer_price * (total_balls - z)
      else if m < 30 then
        (basketball_price - m) * y + soccer_price * (total_balls - y) ≤ (basketball_price - m) * z + soccer_price * (total_balls - z)
      else
        (basketball_price - m) * y + soccer_price * (total_balls - y) = (basketball_price - m) * z + soccer_price * (total_balls - z)))) ∧
  basketball_price = soccer_price + price_difference ∧
  soccer_budget / soccer_price = basketball_budget / basketball_price := by
  sorry

end NUMINAMATH_CALUDE_sports_club_purchase_l1252_125208


namespace NUMINAMATH_CALUDE_expression_value_at_x_2_l1252_125211

theorem expression_value_at_x_2 :
  let x : ℝ := 2
  (3 * x + 4)^2 - 10 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_x_2_l1252_125211


namespace NUMINAMATH_CALUDE_min_sum_squares_l1252_125290

theorem min_sum_squares (x y z : ℝ) (h : x^2 + y^2 + z^2 - 3*x*y*z = 1) :
  ∀ a b c : ℝ, a^2 + b^2 + c^2 - 3*a*b*c = 1 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1252_125290


namespace NUMINAMATH_CALUDE_remainder_98_power_50_mod_100_l1252_125222

theorem remainder_98_power_50_mod_100 : 98^50 % 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_power_50_mod_100_l1252_125222


namespace NUMINAMATH_CALUDE_railway_length_scientific_notation_l1252_125233

theorem railway_length_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    139000 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ a ∧ 
    a < 10 ∧ 
    a = 1.39 ∧ 
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_railway_length_scientific_notation_l1252_125233


namespace NUMINAMATH_CALUDE_davids_age_twice_daughters_l1252_125246

/-- 
Given:
- David is currently 40 years old
- David's daughter is currently 12 years old

Prove that in 16 years, David's age will be twice his daughter's age.
-/
theorem davids_age_twice_daughters (david_age : ℕ) (daughter_age : ℕ) (years_passed : ℕ) : 
  david_age = 40 → daughter_age = 12 → years_passed = 16 → 
  david_age + years_passed = 2 * (daughter_age + years_passed) :=
by sorry

end NUMINAMATH_CALUDE_davids_age_twice_daughters_l1252_125246


namespace NUMINAMATH_CALUDE_minimum_area_is_14_l1252_125270

-- Define the variation ranges
def normal_variation : ℝ := 0.5
def approximate_variation : ℝ := 1.0

-- Define the reported dimensions
def reported_length : ℝ := 4.0
def reported_width : ℝ := 5.0

-- Define the actual minimum dimensions
def min_length : ℝ := reported_length - normal_variation
def min_width : ℝ := reported_width - approximate_variation

-- Define the minimum area
def min_area : ℝ := min_length * min_width

-- Theorem statement
theorem minimum_area_is_14 : min_area = 14 := by
  sorry

end NUMINAMATH_CALUDE_minimum_area_is_14_l1252_125270


namespace NUMINAMATH_CALUDE_volume_of_solid_l1252_125229

-- Define the region S
def S : Set (ℝ × ℝ) :=
  {(x, y) | |9 - x| + y ≤ 12 ∧ 3 * y - x ≥ 18}

-- Define the line of revolution
def revolution_line (x y : ℝ) : Prop :=
  3 * y - x = 18

-- Define the volume of the solid
def solid_volume (S : Set (ℝ × ℝ)) (line : ℝ → ℝ → Prop) : ℝ :=
  -- This is a placeholder for the actual volume calculation
  sorry

-- Theorem statement
theorem volume_of_solid :
  solid_volume S revolution_line = 135 * Real.pi / (8 * Real.sqrt 10) :=
by
  sorry

end NUMINAMATH_CALUDE_volume_of_solid_l1252_125229


namespace NUMINAMATH_CALUDE_max_distance_with_turns_l1252_125277

theorem max_distance_with_turns (total_distance : ℕ) (num_turns : ℕ) 
  (h1 : total_distance = 500) (h2 : num_turns = 300) :
  ∃ (d : ℝ), d ≤ Real.sqrt 145000 ∧ 
  (∀ (a b : ℕ), a + b = total_distance → a ≥ num_turns / 2 → b ≥ num_turns / 2 → 
    Real.sqrt (a^2 + b^2 : ℝ) ≤ d) ∧
  ⌊d⌋ = 380 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_with_turns_l1252_125277


namespace NUMINAMATH_CALUDE_missing_angle_in_polygon_l1252_125250

theorem missing_angle_in_polygon (n : ℕ) (sum_angles : ℝ) (common_angle : ℝ) : 
  sum_angles = 3420 →
  common_angle = 150 →
  n > 2 →
  (n - 1) * common_angle + (sum_angles - (n - 1) * common_angle) = sum_angles →
  sum_angles - (n - 1) * common_angle = 420 :=
by sorry

end NUMINAMATH_CALUDE_missing_angle_in_polygon_l1252_125250


namespace NUMINAMATH_CALUDE_intersection_sum_l1252_125236

theorem intersection_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + 2 → y = -2 * x + b) → 
  (12 : ℝ) = m * 4 + 2 → 
  (12 : ℝ) = -2 * 4 + b → 
  b + m = 22.5 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l1252_125236


namespace NUMINAMATH_CALUDE_rent_increase_proof_l1252_125234

/-- Given a group of 4 friends with an initial average rent and a new average rent after
    one friend's rent is increased, proves that the original rent of the friend whose rent
    was increased is equal to a specific value. -/
theorem rent_increase_proof (initial_avg : ℝ) (new_avg : ℝ) (increase_rate : ℝ) :
  initial_avg = 800 →
  new_avg = 850 →
  increase_rate = 0.25 →
  (4 : ℝ) * new_avg - (4 : ℝ) * initial_avg = increase_rate * ((4 : ℝ) * new_avg - (4 : ℝ) * initial_avg) / increase_rate :=
by sorry

#check rent_increase_proof

end NUMINAMATH_CALUDE_rent_increase_proof_l1252_125234


namespace NUMINAMATH_CALUDE_monkey_ladder_min_steps_l1252_125202

/-- The minimum number of steps for the monkey's ladder. -/
def min_steps : ℕ := 26

/-- Represents the possible movements of the monkey. -/
inductive Movement
| up : Movement
| down : Movement

/-- The number of steps the monkey moves in each direction. -/
def step_count (m : Movement) : ℤ :=
  match m with
  | Movement.up => 18
  | Movement.down => -10

/-- A sequence of movements that allows the monkey to reach the top and return to the ground. -/
def valid_sequence : List Movement := 
  [Movement.up, Movement.down, Movement.up, Movement.down, Movement.down, Movement.up, 
   Movement.down, Movement.down, Movement.up, Movement.down, Movement.down, Movement.up, 
   Movement.down, Movement.down]

theorem monkey_ladder_min_steps :
  (∀ (seq : List Movement), 
    (seq.foldl (λ acc m => acc + step_count m) 0 = 0) →
    (seq.foldl (λ acc m => max acc (acc + step_count m)) 0 ≥ min_steps)) ∧
  (valid_sequence.foldl (λ acc m => acc + step_count m) 0 = 0) ∧
  (valid_sequence.foldl (λ acc m => max acc (acc + step_count m)) 0 = min_steps) := by
  sorry

#check monkey_ladder_min_steps

end NUMINAMATH_CALUDE_monkey_ladder_min_steps_l1252_125202


namespace NUMINAMATH_CALUDE_fourth_selected_id_is_16_l1252_125253

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  totalStudents : Nat
  numTickets : Nat
  selectedIDs : Fin 3 → Nat

/-- Calculates the sampling interval for a given systematic sampling -/
def samplingInterval (s : SystematicSampling) : Nat :=
  s.totalStudents / s.numTickets

/-- Checks if a given ID is part of the systematic sampling -/
def isSelectedID (s : SystematicSampling) (id : Nat) : Prop :=
  ∃ k : Fin s.numTickets, id = (s.selectedIDs 0) + k * samplingInterval s

/-- Theorem: Given the conditions, the fourth selected ID is 16 -/
theorem fourth_selected_id_is_16 (s : SystematicSampling) 
  (h1 : s.totalStudents = 54)
  (h2 : s.numTickets = 4)
  (h3 : s.selectedIDs 0 = 3)
  (h4 : s.selectedIDs 1 = 29)
  (h5 : s.selectedIDs 2 = 42)
  : ∃ id : Nat, id = 16 ∧ isSelectedID s id :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_selected_id_is_16_l1252_125253


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l1252_125204

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (γ : ℝ) 
  (ha : a = 7) 
  (hb : b = 8) 
  (hγ : γ = 2 * π / 3) -- 120° in radians
  (hc : c^2 = a^2 + b^2 - 2*a*b*Real.cos γ) : -- Law of Cosines
  c = 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l1252_125204


namespace NUMINAMATH_CALUDE_sphere_surface_area_containing_cuboid_l1252_125217

theorem sphere_surface_area_containing_cuboid (a b c : ℝ) (S : ℝ) :
  a = 3 → b = 4 → c = 5 →
  S = 4 * Real.pi * ((a^2 + b^2 + c^2) / 4) →
  S = 50 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_containing_cuboid_l1252_125217


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l1252_125212

theorem roots_quadratic_equation (α β : ℝ) : 
  (α^2 - 2*α - 1 = 0) → 
  (β^2 - 2*β - 1 = 0) → 
  (4 * α^3 + 5 * β^4 = -40*α + 153) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l1252_125212


namespace NUMINAMATH_CALUDE_seeds_in_big_garden_l1252_125283

/-- Given Nancy's gardening scenario, prove the number of seeds in the big garden. -/
theorem seeds_in_big_garden 
  (total_seeds : ℕ) 
  (num_small_gardens : ℕ) 
  (seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 52)
  (h2 : num_small_gardens = 6)
  (h3 : seeds_per_small_garden = 4) : 
  total_seeds - (num_small_gardens * seeds_per_small_garden) = 28 := by
  sorry

end NUMINAMATH_CALUDE_seeds_in_big_garden_l1252_125283


namespace NUMINAMATH_CALUDE_max_hangers_buyable_l1252_125264

def total_budget : ℝ := 60
def tissue_cost : ℝ := 34.8
def hanger_cost : ℝ := 1.6

theorem max_hangers_buyable : 
  ⌊(total_budget - tissue_cost) / hanger_cost⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_max_hangers_buyable_l1252_125264


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1252_125206

theorem roots_of_quadratic_equation :
  let f : ℂ → ℂ := λ x => x^2 + 4
  ∀ x : ℂ, f x = 0 ↔ x = 2*I ∨ x = -2*I :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1252_125206


namespace NUMINAMATH_CALUDE_increase_by_percentage_l1252_125273

/-- Proves that increasing 90 by 50% results in 135 -/
theorem increase_by_percentage (initial : ℕ) (percentage : ℚ) (result : ℕ) : 
  initial = 90 → percentage = 50 / 100 → result = initial + (initial * percentage) → result = 135 :=
by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l1252_125273


namespace NUMINAMATH_CALUDE_parabola_properties_l1252_125272

/-- Parabola function -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

/-- Theorem about properties of the parabola y = x^2 + 2x - 3 -/
theorem parabola_properties :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (0, -3)
  ∀ x ∈ Set.Icc (-3 : ℝ) 2,
  (f A.1 = A.2 ∧ f B.1 = B.2 ∧ f C.1 = C.2) ∧
  (A.1 < B.1) ∧
  (∃ (y_max y_min : ℝ), 
    (∀ x' ∈ Set.Icc (-3 : ℝ) 2, f x' ≤ y_max ∧ f x' ≥ y_min) ∧
    y_max - y_min = 9) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1252_125272


namespace NUMINAMATH_CALUDE_visible_cubes_12_cube_l1252_125263

/-- Represents a cube with side length n --/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Calculates the number of visible unit cubes from a corner of a cube --/
def visible_unit_cubes (c : Cube n) : ℕ :=
  3 * n^2 - 3 * (n - 1) + 1

/-- Theorem stating that for a 12×12×12 cube, the number of visible unit cubes from a corner is 400 --/
theorem visible_cubes_12_cube :
  ∃ (c : Cube 12), visible_unit_cubes c = 400 :=
sorry

end NUMINAMATH_CALUDE_visible_cubes_12_cube_l1252_125263


namespace NUMINAMATH_CALUDE_weekly_jog_distance_l1252_125216

/-- The total distance jogged throughout the week in kilometers -/
def total_distance (mon tue wed thu fri_miles : ℝ) (mile_to_km : ℝ) : ℝ :=
  mon + tue + wed + thu + (fri_miles * mile_to_km)

/-- Theorem stating the total distance jogged throughout the week -/
theorem weekly_jog_distance :
  let mon := 3
  let tue := 5.5
  let wed := 9.7
  let thu := 10.8
  let fri_miles := 2
  let mile_to_km := 1.60934
  total_distance mon tue wed thu fri_miles mile_to_km = 32.21868 := by
  sorry

end NUMINAMATH_CALUDE_weekly_jog_distance_l1252_125216


namespace NUMINAMATH_CALUDE_sequence_general_term_l1252_125232

theorem sequence_general_term (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) 
  (h2 : ∀ n, 2 * a n = 3 * a (n + 1)) (h3 : a 2 * a 5 = 8 / 27) :
  ∀ n, a n = (2 / 3) ^ (n - 2) := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1252_125232


namespace NUMINAMATH_CALUDE_abs_sum_problem_l1252_125271

theorem abs_sum_problem (x y : ℝ) 
  (h1 : |x| + x + y = 8) 
  (h2 : x + |y| - y = 14) : 
  x + y = 2 := by sorry

end NUMINAMATH_CALUDE_abs_sum_problem_l1252_125271


namespace NUMINAMATH_CALUDE_union_of_sets_l1252_125214

theorem union_of_sets (M N : Set ℕ) : 
  M = {0, 2, 3} → N = {1, 3} → M ∪ N = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1252_125214


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l1252_125225

/-- If three lines x = 2, x - y - 1 = 0, and x + ky = 0 intersect at a single point, then k = -2 -/
theorem intersection_of_three_lines (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.1 = 2 ∧ p.1 - p.2 - 1 = 0 ∧ p.1 + k * p.2 = 0) → k = -2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l1252_125225


namespace NUMINAMATH_CALUDE_first_grade_sample_size_l1252_125224

/-- Represents the ratio of students in each grade -/
structure GradeRatio :=
  (first second third fourth : ℕ)

/-- Calculates the number of students to be sampled from the first grade
    given the total sample size and the grade ratio -/
def sampleFirstGrade (totalSample : ℕ) (ratio : GradeRatio) : ℕ :=
  (totalSample * ratio.first) / (ratio.first + ratio.second + ratio.third + ratio.fourth)

/-- Theorem stating that for a sample size of 300 and a grade ratio of 4:5:5:6,
    the number of students to be sampled from the first grade is 60 -/
theorem first_grade_sample_size :
  let totalSample : ℕ := 300
  let ratio : GradeRatio := { first := 4, second := 5, third := 5, fourth := 6 }
  sampleFirstGrade totalSample ratio = 60 := by
  sorry


end NUMINAMATH_CALUDE_first_grade_sample_size_l1252_125224


namespace NUMINAMATH_CALUDE_f_derivative_l1252_125276

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

theorem f_derivative :
  deriv f = fun x => x * Real.cos x := by sorry

end NUMINAMATH_CALUDE_f_derivative_l1252_125276


namespace NUMINAMATH_CALUDE_square_perimeter_l1252_125210

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 900) (h2 : side * side = area) : 
  4 * side = 120 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1252_125210


namespace NUMINAMATH_CALUDE_cylinder_plane_intersection_l1252_125226

/-- The equation of the curve formed by intersecting a cylinder with a plane -/
theorem cylinder_plane_intersection
  (r h : ℝ) -- radius and height of the cylinder
  (α : ℝ) -- angle between cutting plane and base plane
  (hr : r > 0)
  (hh : h > 0)
  (hα : 0 < α ∧ α < π/2) :
  ∃ f : ℝ → ℝ,
    (∀ x, 0 < x → x < 2*π*r →
      f x = r * Real.tan α * Real.sin (x/r - π/2)) ∧
    (∀ x, f x = 0 → (x = 0 ∨ x = 2*π*r)) :=
sorry

end NUMINAMATH_CALUDE_cylinder_plane_intersection_l1252_125226


namespace NUMINAMATH_CALUDE_worker_completion_time_l1252_125200

/-- Given workers A and B, where A can complete a job in 15 days,
    A works for 5 days, and B finishes the remaining work in 12 days,
    prove that B alone can complete the entire job in 18 days. -/
theorem worker_completion_time (a_total_days b_remaining_days : ℕ) 
    (h1 : a_total_days = 15)
    (h2 : b_remaining_days = 12) : 
  (18 : ℚ) = (b_remaining_days : ℚ) / ((a_total_days - 5 : ℚ) / a_total_days) := by
  sorry

end NUMINAMATH_CALUDE_worker_completion_time_l1252_125200


namespace NUMINAMATH_CALUDE_dance_troupe_arrangement_l1252_125285

theorem dance_troupe_arrangement (n : ℕ) : n > 0 ∧ 
  6 ∣ n ∧ 9 ∣ n ∧ 12 ∣ n ∧ 5 ∣ n → n ≥ 180 :=
by sorry

end NUMINAMATH_CALUDE_dance_troupe_arrangement_l1252_125285


namespace NUMINAMATH_CALUDE_product_of_eight_consecutive_integers_divisible_by_ten_l1252_125258

theorem product_of_eight_consecutive_integers_divisible_by_ten (n : ℕ) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7)) = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_product_of_eight_consecutive_integers_divisible_by_ten_l1252_125258


namespace NUMINAMATH_CALUDE_task_assignments_and_arrangements_l1252_125298

def num_volunteers : ℕ := 5
def num_tasks : ℕ := 4

theorem task_assignments_and_arrangements :
  let assign_all_tasks := (num_volunteers.choose 2) * num_tasks.factorial
  let assign_one_task_to_two := (num_volunteers.choose 2) * (num_tasks - 1).factorial
  let photo_arrangement := (num_volunteers.factorial / (num_volunteers - 2).factorial) * 2
  (assign_all_tasks = 240) ∧
  (assign_one_task_to_two = 60) ∧
  (photo_arrangement = 40) := by sorry

end NUMINAMATH_CALUDE_task_assignments_and_arrangements_l1252_125298


namespace NUMINAMATH_CALUDE_max_value_implies_m_eq_two_l1252_125288

/-- The function f(x) = x^3 - 3x^2 + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + m

/-- Theorem: If the maximum value of f(x) in [-1, 1] is 2, then m = 2 -/
theorem max_value_implies_m_eq_two (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f m x ≤ 2) ∧ (∃ x ∈ Set.Icc (-1) 1, f m x = 2) → m = 2 := by
  sorry

#check max_value_implies_m_eq_two

end NUMINAMATH_CALUDE_max_value_implies_m_eq_two_l1252_125288


namespace NUMINAMATH_CALUDE_min_balls_needed_l1252_125231

/-- Represents the number of balls of each color -/
structure BallCounts where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Conditions for drawing balls -/
def satisfiesConditions (counts : BallCounts) : Prop :=
  counts.red ≥ 4 ∧
  counts.white ≥ 1 ∧
  counts.blue ≥ 1 ∧
  counts.green ≥ 1 ∧
  (counts.red.choose 4 : ℚ) = 
    (counts.red.choose 3 * counts.white : ℚ) ∧
  (counts.red.choose 3 * counts.white : ℚ) = 
    (counts.red.choose 2 * counts.white * counts.blue : ℚ) ∧
  (counts.red.choose 2 * counts.white * counts.blue : ℚ) = 
    (counts.red * counts.white * counts.blue * counts.green : ℚ)

/-- The theorem to be proved -/
theorem min_balls_needed : 
  ∃ (counts : BallCounts), 
    satisfiesConditions counts ∧ 
    (∀ (other : BallCounts), satisfiesConditions other → 
      counts.red + counts.white + counts.blue + counts.green ≤ 
      other.red + other.white + other.blue + other.green) ∧
    counts.red + counts.white + counts.blue + counts.green = 21 :=
sorry

end NUMINAMATH_CALUDE_min_balls_needed_l1252_125231


namespace NUMINAMATH_CALUDE_fruit_shop_costs_and_profit_l1252_125265

/-- Represents the fruit shop's purchases and sales --/
structure FruitShop where
  first_purchase_cost : ℝ
  first_purchase_price : ℝ
  second_purchase_cost : ℝ
  second_purchase_quantity_increase : ℝ
  second_sale_price : ℝ
  second_sale_quantity : ℝ
  second_sale_discount : ℝ

/-- Calculates the cost per kg and profit for the fruit shop --/
def calculate_costs_and_profit (shop : FruitShop) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating the correct cost per kg and profit given the shop's conditions --/
theorem fruit_shop_costs_and_profit (shop : FruitShop) 
  (h1 : shop.first_purchase_cost = 1200)
  (h2 : shop.first_purchase_price = 8)
  (h3 : shop.second_purchase_cost = 1452)
  (h4 : shop.second_purchase_quantity_increase = 20)
  (h5 : shop.second_sale_price = 9)
  (h6 : shop.second_sale_quantity = 100)
  (h7 : shop.second_sale_discount = 0.5) :
  let (first_cost, second_cost, profit) := calculate_costs_and_profit shop
  first_cost = 6 ∧ second_cost = 6.6 ∧ profit = 388 := by sorry

end NUMINAMATH_CALUDE_fruit_shop_costs_and_profit_l1252_125265


namespace NUMINAMATH_CALUDE_number_divided_by_3000_l1252_125242

theorem number_divided_by_3000 : 
  ∃ x : ℝ, x / 3000 = 0.008416666666666666 ∧ x = 25.25 :=
by sorry

end NUMINAMATH_CALUDE_number_divided_by_3000_l1252_125242


namespace NUMINAMATH_CALUDE_oranges_per_bag_l1252_125220

theorem oranges_per_bag (total_bags : ℕ) (rotten_oranges : ℕ) (juice_oranges : ℕ) (sold_oranges : ℕ)
  (h1 : total_bags = 10)
  (h2 : rotten_oranges = 50)
  (h3 : juice_oranges = 30)
  (h4 : sold_oranges = 220) :
  (rotten_oranges + juice_oranges + sold_oranges) / total_bags = 30 :=
by sorry

end NUMINAMATH_CALUDE_oranges_per_bag_l1252_125220


namespace NUMINAMATH_CALUDE_interest_rate_is_five_percent_l1252_125293

/-- Calculates the interest rate given the principal, time, and interest amount -/
def calculate_interest_rate (principal : ℚ) (time : ℚ) (interest : ℚ) : ℚ :=
  interest / (principal * time)

theorem interest_rate_is_five_percent 
  (principal : ℚ) 
  (time : ℚ) 
  (interest : ℚ) 
  (h1 : principal = 6200)
  (h2 : time = 10)
  (h3 : interest = principal - 3100) :
  calculate_interest_rate principal time interest = 1/20 := by
  sorry

#eval (1/20 : ℚ) * 100 -- To show the result as a percentage

end NUMINAMATH_CALUDE_interest_rate_is_five_percent_l1252_125293


namespace NUMINAMATH_CALUDE_a_equals_one_necessary_not_sufficient_l1252_125205

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a}

-- Statement of the theorem
theorem a_equals_one_necessary_not_sufficient :
  (∃ a : ℝ, a ≠ 1 ∧ A ∪ B a = Set.univ) ∧
  (∀ a : ℝ, a = 1 → A ∪ B a = Set.univ) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_necessary_not_sufficient_l1252_125205


namespace NUMINAMATH_CALUDE_salt_solution_mixture_salt_solution_mixture_proof_l1252_125282

/-- Proves that adding 50 ounces of 10% salt solution to 50 ounces of 40% salt solution results in a 25% salt solution -/
theorem salt_solution_mixture : ℝ → Prop :=
  λ x : ℝ =>
    let initial_volume : ℝ := 50
    let initial_concentration : ℝ := 0.4
    let added_concentration : ℝ := 0.1
    let final_concentration : ℝ := 0.25
    let final_volume : ℝ := initial_volume + x
    let final_salt : ℝ := initial_volume * initial_concentration + x * added_concentration
    (x = 50) →
    (final_salt / final_volume = final_concentration)

/-- The proof of the theorem -/
theorem salt_solution_mixture_proof : salt_solution_mixture 50 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_salt_solution_mixture_proof_l1252_125282


namespace NUMINAMATH_CALUDE_certain_value_proof_l1252_125295

theorem certain_value_proof (number : ℤ) (certain_value : ℤ) 
  (h1 : number = 109)
  (h2 : 150 - number = number + certain_value) : 
  certain_value = -68 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_proof_l1252_125295


namespace NUMINAMATH_CALUDE_laptop_purchase_solution_l1252_125291

/-- Represents the laptop purchase problem for a unit -/
structure LaptopPurchase where
  totalStaff : ℕ
  totalBudget : ℕ
  costA1B2 : ℕ
  costA2B1 : ℕ

/-- The cost of one A laptop -/
def costA (lp : LaptopPurchase) : ℕ := 4080

/-- The cost of one B laptop -/
def costB (lp : LaptopPurchase) : ℕ := 3880

/-- The maximum number of A laptops that can be bought -/
def maxA (lp : LaptopPurchase) : ℕ := 26

/-- Theorem stating the correctness of the laptop purchase solution -/
theorem laptop_purchase_solution (lp : LaptopPurchase) 
  (h1 : lp.totalStaff = 36)
  (h2 : lp.totalBudget = 145000)
  (h3 : lp.costA1B2 = 11840)
  (h4 : lp.costA2B1 = 12040) :
  (costA lp + 2 * costB lp = lp.costA1B2) ∧ 
  (2 * costA lp + costB lp = lp.costA2B1) ∧
  (maxA lp * costA lp + (lp.totalStaff - maxA lp) * costB lp ≤ lp.totalBudget) ∧
  (∀ n : ℕ, n > maxA lp → n * costA lp + (lp.totalStaff - n) * costB lp > lp.totalBudget) := by
  sorry

#check laptop_purchase_solution

end NUMINAMATH_CALUDE_laptop_purchase_solution_l1252_125291


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l1252_125266

/-- Given two linear functions f and g, prove that A + B = 0 under certain conditions -/
theorem sum_of_coefficients_is_zero
  (A B : ℝ)
  (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = A * x + B)
  (h₂ : ∀ x, g x = B * x + A)
  (h₃ : A ≠ B)
  (h₄ : ∀ x, f (g x) - g (f x) = B - A) :
  A + B = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l1252_125266


namespace NUMINAMATH_CALUDE_incorrect_equation_property_l1252_125262

theorem incorrect_equation_property : ¬ (∀ a b : ℝ, a * b = a → b = 1) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_equation_property_l1252_125262


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1252_125256

theorem partial_fraction_decomposition :
  ∀ (x : ℝ) (P Q R : ℚ),
    P = -8/15 ∧ Q = -7/6 ∧ R = 27/10 →
    x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
    (x^2 - 9) / ((x - 1)*(x - 4)*(x - 6)) = P / (x - 1) + Q / (x - 4) + R / (x - 6) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1252_125256


namespace NUMINAMATH_CALUDE_sum_of_primes_floor_condition_l1252_125235

theorem sum_of_primes_floor_condition : 
  (∃ p₁ p₂ : ℕ, 
    p₁.Prime ∧ p₂.Prime ∧ p₁ ≠ p₂ ∧
    (∃ n₁ : ℕ+, 5 * p₁ = ⌊(n₁.val ^ 2 : ℚ) / 5⌋) ∧
    (∃ n₂ : ℕ+, 5 * p₂ = ⌊(n₂.val ^ 2 : ℚ) / 5⌋) ∧
    (∀ p : ℕ, p.Prime → 
      (∃ n : ℕ+, 5 * p = ⌊(n.val ^ 2 : ℚ) / 5⌋) → 
      p = p₁ ∨ p = p₂) ∧
    p₁ + p₂ = 52) :=
  sorry

end NUMINAMATH_CALUDE_sum_of_primes_floor_condition_l1252_125235


namespace NUMINAMATH_CALUDE_four_solutions_l1252_125223

/-- The number of integer solutions to x^4 + y^2 = 2y + 1 -/
def solution_count : ℕ := 4

/-- A function that checks if a pair of integers satisfies the equation -/
def satisfies_equation (x y : ℤ) : Prop :=
  x^4 + y^2 = 2*y + 1

/-- The theorem stating that there are exactly 4 integer solutions -/
theorem four_solutions :
  ∃! (solutions : Finset (ℤ × ℤ)), 
    solutions.card = solution_count ∧ 
    ∀ (x y : ℤ), (x, y) ∈ solutions ↔ satisfies_equation x y :=
sorry

end NUMINAMATH_CALUDE_four_solutions_l1252_125223


namespace NUMINAMATH_CALUDE_max_handshakes_25_20_l1252_125261

/-- Represents a meeting with a given number of people and a maximum number of handshakes per person. -/
structure Meeting where
  num_people : ℕ
  max_handshakes_per_person : ℕ

/-- Calculates the maximum number of handshakes in a meeting. -/
def max_handshakes (m : Meeting) : ℕ :=
  (m.num_people * m.max_handshakes_per_person) / 2

/-- Theorem stating that in a meeting of 25 people where each person shakes hands with at most 20 others,
    the maximum number of handshakes is 250. -/
theorem max_handshakes_25_20 :
  let m : Meeting := ⟨25, 20⟩
  max_handshakes m = 250 := by
  sorry

#eval max_handshakes ⟨25, 20⟩

end NUMINAMATH_CALUDE_max_handshakes_25_20_l1252_125261


namespace NUMINAMATH_CALUDE_deepak_age_l1252_125281

theorem deepak_age (rahul deepak rohan : ℕ) : 
  rahul = 5 * (rahul / 5) → 
  deepak = 2 * (rahul / 5) → 
  rohan = 3 * (rahul / 5) → 
  rahul + 8 = 28 → 
  deepak = 8 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l1252_125281


namespace NUMINAMATH_CALUDE_cannot_transform_fraction_l1252_125260

/-- Represents a fraction with integer numerator and denominator -/
structure Fraction where
  numerator : Int
  denominator : Int
  denominator_nonzero : denominator ≠ 0

/-- Represents the allowed operations on fractions -/
inductive Operation
  | Add (k : Nat)
  | Multiply (n : Nat)

/-- Applies an operation to a fraction -/
def applyOperation (f : Fraction) (op : Operation) : Fraction :=
  match op with
  | Operation.Add k => 
      { numerator := f.numerator + k, 
        denominator := f.denominator + k,
        denominator_nonzero := sorry }
  | Operation.Multiply n => 
      { numerator := f.numerator * n, 
        denominator := f.denominator * n,
        denominator_nonzero := sorry }

/-- Checks if two fractions are equal -/
def fractionEqual (f1 f2 : Fraction) : Prop :=
  f1.numerator * f2.denominator = f2.numerator * f1.denominator

/-- The main theorem stating that it's impossible to transform 5/8 into 3/5 -/
theorem cannot_transform_fraction :
  ∀ (ops : List Operation),
  let start := Fraction.mk 5 8 (by norm_num)
  let target := Fraction.mk 3 5 (by norm_num)
  let result := ops.foldl applyOperation start
  ¬(fractionEqual result target) := by
  sorry


end NUMINAMATH_CALUDE_cannot_transform_fraction_l1252_125260


namespace NUMINAMATH_CALUDE_f_range_l1252_125219

def f (x : ℝ) : ℝ := -x^2 + 2*x + 4

theorem f_range : ∀ x ∈ Set.Icc 0 3, 1 ≤ f x ∧ f x ≤ 5 := by sorry

end NUMINAMATH_CALUDE_f_range_l1252_125219


namespace NUMINAMATH_CALUDE_fortune_telling_app_probability_l1252_125275

theorem fortune_telling_app_probability :
  let n : ℕ := 7  -- Total number of trials
  let k : ℕ := 3  -- Number of successful trials
  let p : ℚ := 1/3  -- Probability of success in each trial
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 := by
  sorry

end NUMINAMATH_CALUDE_fortune_telling_app_probability_l1252_125275


namespace NUMINAMATH_CALUDE_dividend_calculation_l1252_125268

theorem dividend_calculation (dividend divisor : ℕ) : 
  dividend / divisor = 15 →
  dividend % divisor = 5 →
  dividend + divisor + 15 + 5 = 2169 →
  dividend = 2015 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1252_125268


namespace NUMINAMATH_CALUDE_tan_two_implies_fraction_l1252_125254

theorem tan_two_implies_fraction (θ : Real) (h : Real.tan θ = 2) :
  (Real.cos θ - Real.sin θ) / (Real.cos θ + Real.sin θ) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_fraction_l1252_125254


namespace NUMINAMATH_CALUDE_qr_length_l1252_125241

/-- Given points P, Q, R on a line segment where Q is between P and R -/
structure LineSegment where
  P : ℝ
  Q : ℝ
  R : ℝ
  Q_between : P ≤ Q ∧ Q ≤ R

/-- The length of a line segment -/
def length (a b : ℝ) : ℝ := |b - a|

theorem qr_length (seg : LineSegment) 
  (h1 : length seg.P seg.R = 12)
  (h2 : length seg.P seg.Q = 3) : 
  length seg.Q seg.R = 9 := by
sorry

end NUMINAMATH_CALUDE_qr_length_l1252_125241


namespace NUMINAMATH_CALUDE_no_periodic_difference_with_periods_3_and_pi_l1252_125274

-- Define a periodic function
def IsPeriodic (f : ℝ → ℝ) :=
  (∃ x y, f x ≠ f y) ∧
  (∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x)

-- Define the period of a function
def IsPeriodOf (p : ℝ) (f : ℝ → ℝ) :=
  p > 0 ∧ ∀ x, f (x + p) = f x

-- Theorem statement
theorem no_periodic_difference_with_periods_3_and_pi :
  ¬ ∃ (g h : ℝ → ℝ),
    IsPeriodic g ∧ IsPeriodic h ∧
    IsPeriodOf 3 g ∧ IsPeriodOf π h ∧
    IsPeriodic (g - h) :=
sorry

end NUMINAMATH_CALUDE_no_periodic_difference_with_periods_3_and_pi_l1252_125274


namespace NUMINAMATH_CALUDE_expression_value_l1252_125251

theorem expression_value (a : ℝ) (h1 : 1 < a) (h2 : a < 2) :
  Real.sqrt ((a - 2)^2) + |1 - a| = 1 := by sorry

end NUMINAMATH_CALUDE_expression_value_l1252_125251


namespace NUMINAMATH_CALUDE_symmetry_about_y_axis_l1252_125227

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x
def g (x : ℝ) : ℝ := x^2 + 2*x

-- State the theorem
theorem symmetry_about_y_axis (x : ℝ) : 
  (∀ (y : ℝ), f x = y ↔ g (-x) = y) → g x = x^2 + 2*x :=
by sorry

end NUMINAMATH_CALUDE_symmetry_about_y_axis_l1252_125227
