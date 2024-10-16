import Mathlib

namespace NUMINAMATH_CALUDE_prob_at_least_one_3_or_5_correct_l3932_393220

/-- The probability of at least one die showing either a 3 or a 5 when rolling two fair 6-sided dice -/
def prob_at_least_one_3_or_5 : ℚ :=
  5 / 9

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The set of outcomes for a single die roll -/
def die_outcomes : Finset ℕ := Finset.range num_sides

/-- The set of favorable outcomes for a single die (3 or 5) -/
def favorable_single : Finset ℕ := {3, 5}

/-- The sample space for rolling two dice -/
def sample_space : Finset (ℕ × ℕ) := die_outcomes.product die_outcomes

/-- The event where at least one die shows 3 or 5 -/
def event : Finset (ℕ × ℕ) :=
  sample_space.filter (fun p => p.1 ∈ favorable_single ∨ p.2 ∈ favorable_single)

theorem prob_at_least_one_3_or_5_correct :
  (event.card : ℚ) / sample_space.card = prob_at_least_one_3_or_5 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_3_or_5_correct_l3932_393220


namespace NUMINAMATH_CALUDE_age_sum_problem_l3932_393266

theorem age_sum_problem (a b c : ℕ+) : 
  b = c → b < a → a * b * c = 144 → a + b + c = 22 := by sorry

end NUMINAMATH_CALUDE_age_sum_problem_l3932_393266


namespace NUMINAMATH_CALUDE_only_drunk_drivers_traffic_accidents_correlated_l3932_393210

-- Define the types of quantities
inductive Quantity
  | Time
  | Displacement
  | StudentGrade
  | Weight
  | DrunkDrivers
  | TrafficAccidents
  | Volume

-- Define the relationship between quantities
inductive Relationship
  | NoRelation
  | Correlation
  | FunctionalRelation

-- Define a function to determine the relationship between two quantities
def relationshipBetween (q1 q2 : Quantity) : Relationship :=
  match q1, q2 with
  | Quantity.Time, Quantity.Displacement => Relationship.FunctionalRelation
  | Quantity.StudentGrade, Quantity.Weight => Relationship.NoRelation
  | Quantity.DrunkDrivers, Quantity.TrafficAccidents => Relationship.Correlation
  | Quantity.Volume, Quantity.Weight => Relationship.FunctionalRelation
  | _, _ => Relationship.NoRelation

-- Theorem to prove
theorem only_drunk_drivers_traffic_accidents_correlated :
  ∀ q1 q2 : Quantity,
    relationshipBetween q1 q2 = Relationship.Correlation →
    (q1 = Quantity.DrunkDrivers ∧ q2 = Quantity.TrafficAccidents) ∨
    (q1 = Quantity.TrafficAccidents ∧ q2 = Quantity.DrunkDrivers) :=
by
  sorry


end NUMINAMATH_CALUDE_only_drunk_drivers_traffic_accidents_correlated_l3932_393210


namespace NUMINAMATH_CALUDE_max_segment_length_squared_l3932_393219

/-- Circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- Line defined by two points -/
structure Line where
  P : ℝ × ℝ
  Q : ℝ × ℝ

/-- Point on a circle -/
def PointOnCircle (ω : Circle) (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  let (cx, cy) := ω.O
  (x - cx)^2 + (y - cy)^2 = ω.r^2

/-- Tangent line to a circle at a point -/
def TangentLine (ω : Circle) (T : ℝ × ℝ) (l : Line) : Prop :=
  PointOnCircle ω T ∧ 
  ∃ (P : ℝ × ℝ), P ≠ T ∧ PointOnCircle ω P ∧ 
    ((P.1 - T.1) * (l.Q.1 - l.P.1) + (P.2 - T.2) * (l.Q.2 - l.P.2) = 0)

/-- Perpendicular foot from a point to a line -/
def PerpendicularFoot (A : ℝ × ℝ) (l : Line) (P : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (l.Q.1 - l.P.1) + (P.2 - A.2) * (l.Q.2 - l.P.2) = 0 ∧
  ∃ (t : ℝ), P = (l.P.1 + t * (l.Q.1 - l.P.1), l.P.2 + t * (l.Q.2 - l.P.2))

/-- The main theorem -/
theorem max_segment_length_squared 
  (ω : Circle) 
  (A B C T : ℝ × ℝ) 
  (l : Line) 
  (P : ℝ × ℝ) :
  PointOnCircle ω A ∧ 
  PointOnCircle ω B ∧
  ω.r = 12 ∧
  (A.1 - ω.O.1)^2 + (A.2 - ω.O.2)^2 = (B.1 - ω.O.1)^2 + (B.2 - ω.O.2)^2 ∧
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) < 0 ∧
  TangentLine ω T l ∧
  PerpendicularFoot A l P →
  ∃ (m : ℝ), m^2 = 612 ∧ 
    ∀ (X : ℝ × ℝ), PointOnCircle ω X → 
      (X.1 - B.1)^2 + (X.2 - B.2)^2 ≤ m^2 := by
  sorry

end NUMINAMATH_CALUDE_max_segment_length_squared_l3932_393219


namespace NUMINAMATH_CALUDE_sum_remainder_mod_17_l3932_393239

theorem sum_remainder_mod_17 : (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_17_l3932_393239


namespace NUMINAMATH_CALUDE_final_result_l3932_393260

/-- The number of different five-digit even numbers that can be formed using the digits 0, 1, 2, 3, and 4 -/
def even_numbers : ℕ := 60

/-- The number of different five-digit numbers that can be formed using the digits 1, 2, 3, 4, and 5 such that 2 and 3 are not adjacent -/
def non_adjacent_23 : ℕ := 72

/-- The number of different five-digit numbers that can be formed using the digits 1, 2, 3, 4, and 5 such that the digits 1, 2, and 3 must be arranged in descending order -/
def descending_123 : ℕ := 20

/-- The final result is the sum of the three subproblems -/
theorem final_result : even_numbers + non_adjacent_23 + descending_123 = 152 := by
  sorry

end NUMINAMATH_CALUDE_final_result_l3932_393260


namespace NUMINAMATH_CALUDE_dante_recipe_eggs_l3932_393280

theorem dante_recipe_eggs :
  ∀ (eggs flour : ℕ),
  flour = eggs / 2 →
  flour + eggs = 90 →
  eggs = 60 := by
sorry

end NUMINAMATH_CALUDE_dante_recipe_eggs_l3932_393280


namespace NUMINAMATH_CALUDE_polynomial_product_equality_l3932_393205

theorem polynomial_product_equality (x a : ℝ) : (x - a) * (x^2 + a*x + a^2) = x^3 - a^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_equality_l3932_393205


namespace NUMINAMATH_CALUDE_four_intersection_points_iff_c_gt_one_l3932_393215

-- Define the ellipse equation
def ellipse (x y c : ℝ) : Prop :=
  x^2 + y^2/4 = c^2

-- Define the parabola equation
def parabola (x y c : ℝ) : Prop :=
  y = x^2 - 2*c

-- Define the intersection points
def intersection_points (c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse p.1 p.2 c ∧ parabola p.1 p.2 c}

-- Theorem statement
theorem four_intersection_points_iff_c_gt_one (c : ℝ) :
  (∃ (p₁ p₂ p₃ p₄ : ℝ × ℝ), p₁ ∈ intersection_points c ∧
                            p₂ ∈ intersection_points c ∧
                            p₃ ∈ intersection_points c ∧
                            p₄ ∈ intersection_points c ∧
                            p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧
                            p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧
                            p₃ ≠ p₄) ↔
  c > 1 := by
  sorry

end NUMINAMATH_CALUDE_four_intersection_points_iff_c_gt_one_l3932_393215


namespace NUMINAMATH_CALUDE_solve_equation_l3932_393271

theorem solve_equation (y : ℝ) (h : Real.sqrt (3 / y + 3) = 5 / 3) : y = -27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3932_393271


namespace NUMINAMATH_CALUDE_slope_angle_range_l3932_393243

noncomputable def slope_angle (α : Real) : Prop :=
  ∃ (x : Real), x ≠ 0 ∧ Real.tan α = (1/2) * (x + 1/x)

theorem slope_angle_range :
  ∀ α, slope_angle α → 
    (α ∈ Set.Icc (π/4) (π/2) ∪ Set.Ioc (π/2) (3*π/4)) := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_range_l3932_393243


namespace NUMINAMATH_CALUDE_not_right_triangle_l3932_393298

theorem not_right_triangle (a b c : ℕ) (h1 : a = 5) (h2 : b = 11) (h3 : c = 12) :
  ¬(a^2 + b^2 = c^2) := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_l3932_393298


namespace NUMINAMATH_CALUDE_unique_toy_value_l3932_393249

theorem unique_toy_value (total_toys : ℕ) (total_worth : ℕ) (common_value : ℕ) (common_count : ℕ) :
  total_toys = common_count + 1 →
  total_worth = common_value * common_count + (total_worth - common_value * common_count) →
  common_count = 8 →
  total_toys = 9 →
  total_worth = 52 →
  common_value = 5 →
  total_worth - common_value * common_count = 12 :=
by sorry

end NUMINAMATH_CALUDE_unique_toy_value_l3932_393249


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_parallel_lines_l3932_393232

/-- A line in a plane --/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Three parallel lines --/
def parallel_lines : Vector Line 3 :=
  sorry

/-- Definition of an equilateral triangle --/
def is_equilateral_triangle (a b c : Point) : Prop :=
  sorry

/-- Theorem: There exists an equilateral triangle with vertices on three parallel lines --/
theorem equilateral_triangle_on_parallel_lines :
  ∃ (a b c : Point),
    (∀ i : Fin 3, ∃ j : Fin 3, a.y = parallel_lines[i].slope * a.x + parallel_lines[i].intercept ∨
                               b.y = parallel_lines[i].slope * b.x + parallel_lines[i].intercept ∨
                               c.y = parallel_lines[i].slope * c.x + parallel_lines[i].intercept) ∧
    is_equilateral_triangle a b c :=
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_parallel_lines_l3932_393232


namespace NUMINAMATH_CALUDE_base_conversion_435_to_3_l3932_393229

-- Define a function to convert a list of digits in base 3 to base 10
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

-- Theorem statement
theorem base_conversion_435_to_3 :
  base3ToBase10 [1, 1, 1, 0, 2, 1] = 435 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_435_to_3_l3932_393229


namespace NUMINAMATH_CALUDE_rachel_math_homework_l3932_393268

theorem rachel_math_homework (total_math_bio : ℕ) (bio_pages : ℕ) (h1 : total_math_bio = 11) (h2 : bio_pages = 3) :
  total_math_bio - bio_pages = 8 :=
by sorry

end NUMINAMATH_CALUDE_rachel_math_homework_l3932_393268


namespace NUMINAMATH_CALUDE_eight_integer_pairs_satisfy_equation_l3932_393237

theorem eight_integer_pairs_satisfy_equation :
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ s ↔ x > 0 ∧ y > 0 ∧ Real.sqrt (x * y) - 71 * Real.sqrt x + 30 = 0) ∧
    s.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_integer_pairs_satisfy_equation_l3932_393237


namespace NUMINAMATH_CALUDE_square_remainder_l3932_393213

theorem square_remainder (N : ℤ) : 
  (N % 5 = 3) → (N^2 % 5 = 4) := by
sorry

end NUMINAMATH_CALUDE_square_remainder_l3932_393213


namespace NUMINAMATH_CALUDE_find_fifth_month_sale_l3932_393286

def sales_problem (sales : Fin 6 → ℕ) (average : ℕ) : Prop :=
  sales 0 = 800 ∧
  sales 1 = 900 ∧
  sales 2 = 1000 ∧
  sales 3 = 700 ∧
  sales 5 = 900 ∧
  (sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = average

theorem find_fifth_month_sale (sales : Fin 6 → ℕ) (average : ℕ) 
  (h : sales_problem sales average) : sales 4 = 800 := by
  sorry

end NUMINAMATH_CALUDE_find_fifth_month_sale_l3932_393286


namespace NUMINAMATH_CALUDE_square_elements_iff_odd_order_l3932_393211

theorem square_elements_iff_odd_order (G : Type*) [Group G] [Fintype G] :
  (∀ g : G, ∃ h : G, h ^ 2 = g) ↔ Odd (Fintype.card G) :=
sorry

end NUMINAMATH_CALUDE_square_elements_iff_odd_order_l3932_393211


namespace NUMINAMATH_CALUDE_range_of_a_l3932_393223

-- Define the conditions
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x a : ℝ) : Prop := x > a

-- State the theorem
theorem range_of_a (x a : ℝ) :
  (∀ x, p x → (x < -3 ∨ x > 1)) →
  (∀ x, ¬(p x) ↔ (-3 ≤ x ∧ x ≤ 1)) →
  (∀ x, ¬(q x a) ↔ x ≤ a) →
  (∀ x, ¬(p x) → ¬(q x a)) →
  (∃ x, ¬(p x) ∧ q x a) →
  a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3932_393223


namespace NUMINAMATH_CALUDE_count_distinct_keys_l3932_393228

/-- Represents a rotational stencil cipher key of size n × n -/
structure StencilKey (n : ℕ) where
  size : n % 2 = 0  -- n is even

/-- The number of distinct rotational stencil cipher keys for a given even size n -/
def num_distinct_keys (n : ℕ) : ℕ := 4^(n^2/4)

/-- Theorem stating the number of distinct rotational stencil cipher keys -/
theorem count_distinct_keys (n : ℕ) (key : StencilKey n) :
  num_distinct_keys n = 4^(n^2/4) := by
  sorry

#check count_distinct_keys

end NUMINAMATH_CALUDE_count_distinct_keys_l3932_393228


namespace NUMINAMATH_CALUDE_condition_for_inequality_l3932_393274

theorem condition_for_inequality (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_condition_for_inequality_l3932_393274


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3932_393257

/-- Given an arithmetic sequence {a_n} where a_5 = 8 and a_9 = 24, prove that a_4 = 4 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_a5 : a 5 = 8)
  (h_a9 : a 9 = 24) : 
  a 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3932_393257


namespace NUMINAMATH_CALUDE_special_polynomial_inequality_l3932_393247

/-- A polynomial with real coefficients that has three positive real roots and a negative value at x = 0 -/
structure SpecialPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  has_three_positive_roots : ∃ (x₁ x₂ x₃ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧
    ∀ x, a * x^3 + b * x^2 + c * x + d = a * (x - x₁) * (x - x₂) * (x - x₃)
  negative_at_zero : d < 0

/-- The inequality holds for special polynomials -/
theorem special_polynomial_inequality (φ : SpecialPolynomial) :
  2 * φ.b^3 + 9 * φ.a^2 * φ.d - 7 * φ.a * φ.b * φ.c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_inequality_l3932_393247


namespace NUMINAMATH_CALUDE_freshmen_sophomores_without_pets_l3932_393284

theorem freshmen_sophomores_without_pets (total_students : ℕ) 
  (freshmen_sophomore_ratio : ℚ) (pet_owner_ratio : ℚ) : ℕ :=
  by
  sorry

#check freshmen_sophomores_without_pets 400 (1/2) (1/5) = 160

end NUMINAMATH_CALUDE_freshmen_sophomores_without_pets_l3932_393284


namespace NUMINAMATH_CALUDE_even_number_decomposition_theorem_l3932_393254

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m ∧ n > 0

def even_number_decomposition (k : ℤ) : Prop :=
  (∃ a b : ℤ, 2 * k = a + b ∧ is_perfect_square (a * b)) ∨
  (∃ c d : ℤ, 2 * k = c - d ∧ is_perfect_square (c * d))

theorem even_number_decomposition_theorem :
  ∃ S : Set ℤ, S.Finite ∧ ∀ k : ℤ, k ∉ S → even_number_decomposition k :=
sorry

end NUMINAMATH_CALUDE_even_number_decomposition_theorem_l3932_393254


namespace NUMINAMATH_CALUDE_max_value_theorem_l3932_393290

-- Define the constraint function
def constraint (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the objective function
def objective (x y : ℝ) : ℝ := 3*x + 4*y

-- Theorem statement
theorem max_value_theorem :
  ∃ (max : ℝ), max = 5 * Real.sqrt 2 ∧
  (∀ x y : ℝ, constraint x y → objective x y ≤ max) ∧
  (∃ x y : ℝ, constraint x y ∧ objective x y = max) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3932_393290


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3932_393234

theorem unique_triple_solution :
  ∃! (x y z : ℕ), (x + 1)^(y + 1) + 1 = (x + 2)^(z + 1) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3932_393234


namespace NUMINAMATH_CALUDE_series_sum_l3932_393256

theorem series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series_term (n : ℕ) := 1 / (((n - 1) * a - (n - 2) * b) * (n * a - (n - 1) * b))
  let series_sum := ∑' n, series_term n
  series_sum = 1 / ((a - b) * b) := by sorry

end NUMINAMATH_CALUDE_series_sum_l3932_393256


namespace NUMINAMATH_CALUDE_matthew_cooks_30_hotdogs_l3932_393203

/-- The number of hotdogs Matthew needs to cook for his family dinner -/
def total_hotdogs : ℝ :=
  let ella_emma := 2.5 * 2
  let luke := 2 * ella_emma
  let michael := 7
  let hunter := 1.5 * ella_emma
  let zoe := 0.5
  ella_emma + luke + michael + hunter + zoe

/-- Theorem stating that Matthew needs to cook 30 hotdogs -/
theorem matthew_cooks_30_hotdogs : total_hotdogs = 30 := by
  sorry

end NUMINAMATH_CALUDE_matthew_cooks_30_hotdogs_l3932_393203


namespace NUMINAMATH_CALUDE_expected_votes_for_candidate_a_l3932_393259

theorem expected_votes_for_candidate_a (total_voters : ℕ) 
  (democrat_percentage : ℚ) (republican_percentage : ℚ)
  (democrat_support : ℚ) (republican_support : ℚ) :
  democrat_percentage + republican_percentage = 1 →
  democrat_percentage = 3/5 →
  democrat_support = 3/4 →
  republican_support = 1/5 →
  (democrat_percentage * democrat_support + 
   republican_percentage * republican_support) * total_voters = 
  (53/100) * total_voters :=
by sorry

end NUMINAMATH_CALUDE_expected_votes_for_candidate_a_l3932_393259


namespace NUMINAMATH_CALUDE_vacation_miles_theorem_l3932_393258

/-- Calculates the total miles driven during a vacation -/
def total_miles_driven (days : ℕ) (miles_per_day : ℕ) : ℕ :=
  days * miles_per_day

/-- Proves that a 5-day vacation driving 250 miles per day results in 1250 total miles -/
theorem vacation_miles_theorem : 
  total_miles_driven 5 250 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_vacation_miles_theorem_l3932_393258


namespace NUMINAMATH_CALUDE_inequality_proof_l3932_393287

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c + 3 * a * b * c ≥ (a * b)^2 + (b * c)^2 + (c * a)^2 + 3) :
  (a^3 + b^3 + c^3) / 3 ≥ (a * b * c + 2021) / 2022 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3932_393287


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3932_393273

/-- For all a > 0 and a ≠ 1, the function f(x) = a^(x-1) + 4 passes through (1, 5) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 4
  f 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3932_393273


namespace NUMINAMATH_CALUDE_rectangle_y_value_l3932_393275

/-- Given a rectangle with vertices at (-2, y), (10, y), (-2, 4), and (10, 4),
    where y is positive and the area is 108 square units, prove that y = 13. -/
theorem rectangle_y_value (y : ℝ) (h1 : y > 0) (h2 : (10 - (-2)) * (y - 4) = 108) : y = 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l3932_393275


namespace NUMINAMATH_CALUDE_autumn_grain_purchase_l3932_393246

/-- 
Theorem: If the total purchase of autumn grain nationwide exceeded 180 million tons, 
and x represents the amount of autumn grain purchased in China this year in billion tons, 
then x > 1.8.
-/
theorem autumn_grain_purchase (x : ℝ) 
  (h : x * 1000 > 180) : x > 1.8 := by
  sorry

end NUMINAMATH_CALUDE_autumn_grain_purchase_l3932_393246


namespace NUMINAMATH_CALUDE_volume_of_specific_cuboid_l3932_393216

/-- The volume of a cuboid with given edge lengths. -/
def cuboid_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a cuboid with edges 2 cm, 5 cm, and 3 cm is 30 cubic centimeters. -/
theorem volume_of_specific_cuboid : 
  cuboid_volume 2 5 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_cuboid_l3932_393216


namespace NUMINAMATH_CALUDE_expression_equality_l3932_393217

theorem expression_equality : 
  (2^3 ≠ 3^2) ∧ 
  (-2^3 = (-2)^3) ∧ 
  ((-2 * 3)^2 ≠ -2 * 3^2) ∧ 
  ((-5)^2 ≠ -5^2) :=
by sorry

end NUMINAMATH_CALUDE_expression_equality_l3932_393217


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3932_393295

theorem simplify_fraction_product : (240 / 24) * (7 / 140) * (6 / 4) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3932_393295


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3932_393206

theorem cubic_equation_roots : ∃ (x₁ x₂ x₃ : ℚ),
  (x₁ = 3/2 ∧ x₂ = 1/2 ∧ x₃ = -5/2) ∧
  (8 * x₁^3 + 4 * x₁^2 - 34 * x₁ + 15 = 0) ∧
  (8 * x₂^3 + 4 * x₂^2 - 34 * x₂ + 15 = 0) ∧
  (8 * x₃^3 + 4 * x₃^2 - 34 * x₃ + 15 = 0) ∧
  (2 * x₁ - 4 * x₂ = 1) := by
  sorry

#check cubic_equation_roots

end NUMINAMATH_CALUDE_cubic_equation_roots_l3932_393206


namespace NUMINAMATH_CALUDE_bricklayer_wage_is_44_l3932_393255

/-- Represents the hourly wage of a worker -/
structure HourlyWage where
  amount : ℝ
  nonneg : amount ≥ 0

/-- Represents the total hours worked by both workers -/
def total_hours : ℝ := 90

/-- Represents the hourly wage of the electrician -/
def electrician_wage : HourlyWage := ⟨16, by norm_num⟩

/-- Represents the total payment for both workers -/
def total_payment : ℝ := 1350

/-- Represents the hours worked by each worker -/
def individual_hours : ℝ := 22.5

/-- Theorem stating that the bricklayer's hourly wage is $44 -/
theorem bricklayer_wage_is_44 :
  ∃ (bricklayer_wage : HourlyWage),
    bricklayer_wage.amount = 44 ∧
    individual_hours * (bricklayer_wage.amount + electrician_wage.amount) = total_payment ∧
    2 * individual_hours = total_hours :=
by sorry


end NUMINAMATH_CALUDE_bricklayer_wage_is_44_l3932_393255


namespace NUMINAMATH_CALUDE_fireflies_win_by_five_l3932_393289

/-- Represents a basketball team's score -/
structure TeamScore where
  initial : ℕ
  final_baskets : ℕ
  basket_value : ℕ

/-- Calculates the final score of a team -/
def final_score (team : TeamScore) : ℕ :=
  team.initial + team.final_baskets * team.basket_value

/-- Represents the scores of both teams in the basketball game -/
structure GameScore where
  hornets : TeamScore
  fireflies : TeamScore

/-- The theorem stating the final score difference between Fireflies and Hornets -/
theorem fireflies_win_by_five (game : GameScore)
  (h1 : game.hornets = ⟨86, 2, 2⟩)
  (h2 : game.fireflies = ⟨74, 7, 3⟩) :
  final_score game.fireflies - final_score game.hornets = 5 := by
  sorry

#check fireflies_win_by_five

end NUMINAMATH_CALUDE_fireflies_win_by_five_l3932_393289


namespace NUMINAMATH_CALUDE_fraction_equality_l3932_393272

theorem fraction_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^2 / (a^2 + 2*a*b + b^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3932_393272


namespace NUMINAMATH_CALUDE_ahmed_age_l3932_393221

theorem ahmed_age (fouad_age : ℕ) (ahmed_age : ℕ) (h : fouad_age = 26) :
  (fouad_age + 4 = 2 * ahmed_age) → ahmed_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_ahmed_age_l3932_393221


namespace NUMINAMATH_CALUDE_contractor_daily_wage_l3932_393269

/-- Calculates the daily wage for a contractor given the contract terms and outcomes. -/
def calculate_daily_wage (total_days : ℕ) (fine_per_day : ℚ) (total_received : ℚ) (absent_days : ℕ) : ℚ :=
  let worked_days := total_days - absent_days
  let total_fine := fine_per_day * absent_days
  (total_received + total_fine) / worked_days

/-- Proves that the daily wage is 25 given the specific contract terms. -/
theorem contractor_daily_wage :
  calculate_daily_wage 30 (7.5 : ℚ) 360 12 = 25 := by
  sorry

end NUMINAMATH_CALUDE_contractor_daily_wage_l3932_393269


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3932_393231

theorem point_in_second_quadrant (m : ℝ) : 
  let p : ℝ × ℝ := (-1, m^2 + 1)
  p.1 < 0 ∧ p.2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3932_393231


namespace NUMINAMATH_CALUDE_box_volume_increase_l3932_393209

/-- Theorem about the volume of a rectangular box after increasing its dimensions --/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + l * h + w * h) = 1950)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7198 := by sorry

end NUMINAMATH_CALUDE_box_volume_increase_l3932_393209


namespace NUMINAMATH_CALUDE_omega_squared_plus_four_omega_plus_forty_modulus_l3932_393201

theorem omega_squared_plus_four_omega_plus_forty_modulus (ω : ℂ) (h : ω = 5 + 3*I) : 
  Complex.abs (ω^2 + 4*ω + 40) = 2 * Real.sqrt 1885 := by sorry

end NUMINAMATH_CALUDE_omega_squared_plus_four_omega_plus_forty_modulus_l3932_393201


namespace NUMINAMATH_CALUDE_no_equal_perimeter_area_volume_cuboid_l3932_393291

theorem no_equal_perimeter_area_volume_cuboid :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (4 * (a + b + c) = 2 * (a * b + b * c + c * a)) ∧
    (4 * (a + b + c) = a * b * c) :=
by sorry

end NUMINAMATH_CALUDE_no_equal_perimeter_area_volume_cuboid_l3932_393291


namespace NUMINAMATH_CALUDE_max_xy_value_l3932_393282

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x/3 + y/4 = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → a/3 + b/4 = 1 → x*y ≥ a*b ∧ x*y ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l3932_393282


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3932_393285

theorem min_value_of_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) / (c + d) + (a + c) / (b + d) + (a + d) / (b + c) +
  (b + c) / (a + d) + (b + d) / (a + c) + (c + d) / (a + b) ≥ 6 ∧
  ((a + b) / (c + d) + (a + c) / (b + d) + (a + d) / (b + c) +
   (b + c) / (a + d) + (b + d) / (a + c) + (c + d) / (a + b) = 6 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3932_393285


namespace NUMINAMATH_CALUDE_solve_system_l3932_393224

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 14) 
  (eq2 : 6 * p + 5 * q = 17) : 
  q = -1/11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l3932_393224


namespace NUMINAMATH_CALUDE_interval_relation_l3932_393265

theorem interval_relation : 
  (∀ x : ℝ, 3 < x ∧ x < 4 → 2 < x ∧ x < 5) ∧ 
  (∃ x : ℝ, 2 < x ∧ x < 5 ∧ ¬(3 < x ∧ x < 4)) :=
by sorry

end NUMINAMATH_CALUDE_interval_relation_l3932_393265


namespace NUMINAMATH_CALUDE_find_number_to_multiply_l3932_393250

theorem find_number_to_multiply : ∃ x : ℤ, 43 * x - 34 * x = 1215 :=
by sorry

end NUMINAMATH_CALUDE_find_number_to_multiply_l3932_393250


namespace NUMINAMATH_CALUDE_f_composition_value_l3932_393283

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (-x) else Real.log x

theorem f_composition_value : f (f (1/3)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3932_393283


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3932_393245

theorem fraction_sum_equals_decimal : (1 : ℚ) / 10 + 2 / 20 - 3 / 60 = (15 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3932_393245


namespace NUMINAMATH_CALUDE_plant_branches_l3932_393288

theorem plant_branches : ∃ (x : ℕ), x > 0 ∧ 1 + x + x * x = 57 := by
  sorry

end NUMINAMATH_CALUDE_plant_branches_l3932_393288


namespace NUMINAMATH_CALUDE_oshea_large_planters_l3932_393200

/-- The number of large planters Oshea has -/
def num_large_planters (total_seeds small_planter_capacity large_planter_capacity num_small_planters : ℕ) : ℕ :=
  (total_seeds - small_planter_capacity * num_small_planters) / large_planter_capacity

/-- Proof that Oshea has 4 large planters -/
theorem oshea_large_planters :
  num_large_planters 200 4 20 30 = 4 := by
  sorry

end NUMINAMATH_CALUDE_oshea_large_planters_l3932_393200


namespace NUMINAMATH_CALUDE_gcd_nine_digit_repeats_l3932_393297

/-- The set of all nine-digit integers formed by repeating a three-digit integer three times -/
def NineDigitRepeats : Set ℕ :=
  {n | ∃ k : ℕ, 100 ≤ k ∧ k ≤ 999 ∧ n = 1001001 * k}

/-- The greatest common divisor of all numbers in NineDigitRepeats is 1001001 -/
theorem gcd_nine_digit_repeats :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ NineDigitRepeats, d ∣ n) ∧
  (∀ m : ℕ, m > 0 → (∀ n ∈ NineDigitRepeats, m ∣ n) → m ≤ d) ∧
  d = 1001001 := by
  sorry


end NUMINAMATH_CALUDE_gcd_nine_digit_repeats_l3932_393297


namespace NUMINAMATH_CALUDE_sara_marbles_l3932_393252

/-- The number of black marbles Sara has after Fred takes some -/
def remaining_marbles (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Theorem stating that Sara's remaining black marbles is the difference between initial and taken -/
theorem sara_marbles (initial : ℕ) (taken : ℕ) (h : taken ≤ initial) :
  remaining_marbles initial taken = initial - taken :=
by sorry

end NUMINAMATH_CALUDE_sara_marbles_l3932_393252


namespace NUMINAMATH_CALUDE_equation_solution_l3932_393248

theorem equation_solution (x : ℝ) : 
  Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 2 ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3932_393248


namespace NUMINAMATH_CALUDE_helmet_sales_theorem_l3932_393276

/-- Represents the helmet sales scenario -/
structure HelmetSales where
  originalPrice : ℝ
  originalSales : ℝ
  costPrice : ℝ
  salesIncrease : ℝ

/-- Calculates the monthly profit given a price reduction -/
def monthlyProfit (hs : HelmetSales) (priceReduction : ℝ) : ℝ :=
  (hs.originalPrice - priceReduction - hs.costPrice) * (hs.originalSales + hs.salesIncrease * priceReduction)

/-- The main theorem about helmet sales -/
theorem helmet_sales_theorem (hs : HelmetSales) 
    (h1 : hs.originalPrice = 80)
    (h2 : hs.originalSales = 200)
    (h3 : hs.costPrice = 50)
    (h4 : hs.salesIncrease = 10) :
  (∃ (x : ℝ), x ≥ 10 ∧ monthlyProfit hs x = 5250 ∧ hs.originalPrice - x = 65) ∧
  (∃ (maxProfit : ℝ), maxProfit = 6000 ∧ 
    ∀ (y : ℝ), y ≥ 10 → monthlyProfit hs y ≤ maxProfit ∧
    monthlyProfit hs 10 = maxProfit) := by
  sorry

end NUMINAMATH_CALUDE_helmet_sales_theorem_l3932_393276


namespace NUMINAMATH_CALUDE_camping_trip_items_l3932_393242

theorem camping_trip_items (total_items : ℕ) 
  (tent_stakes : ℕ) (drink_mix : ℕ) (water_bottles : ℕ) : 
  total_items = 22 → 
  drink_mix = 3 * tent_stakes → 
  water_bottles = tent_stakes + 2 → 
  total_items = tent_stakes + drink_mix + water_bottles → 
  tent_stakes = 4 := by
sorry

end NUMINAMATH_CALUDE_camping_trip_items_l3932_393242


namespace NUMINAMATH_CALUDE_no_call_days_l3932_393240

theorem no_call_days (total_days : ℕ) (call_period1 call_period2 call_period3 : ℕ) : 
  total_days = 365 ∧ call_period1 = 2 ∧ call_period2 = 5 ∧ call_period3 = 7 →
  total_days - (
    (total_days / call_period1 + total_days / call_period2 + total_days / call_period3) -
    (total_days / (Nat.lcm call_period1 call_period2) + 
     total_days / (Nat.lcm call_period1 call_period3) + 
     total_days / (Nat.lcm call_period2 call_period3)) +
    total_days / (Nat.lcm call_period1 (Nat.lcm call_period2 call_period3))
  ) = 125 := by
  sorry

end NUMINAMATH_CALUDE_no_call_days_l3932_393240


namespace NUMINAMATH_CALUDE_spherical_coordinates_reflection_l3932_393202

theorem spherical_coordinates_reflection :
  ∀ (x y z : ℝ),
  (∃ (ρ θ φ : ℝ),
    ρ = 4 ∧ θ = 5 * π / 6 ∧ φ = π / 4 ∧
    x = ρ * Real.sin φ * Real.cos θ ∧
    y = ρ * Real.sin φ * Real.sin θ ∧
    z = ρ * Real.cos φ) →
  (∃ (ρ' θ' φ' : ℝ),
    ρ' = 2 * Real.sqrt 10 ∧ θ' = 5 * π / 6 ∧ φ' = 3 * π / 4 ∧
    x = ρ' * Real.sin φ' * Real.cos θ' ∧
    y = ρ' * Real.sin φ' * Real.sin θ' ∧
    -z = ρ' * Real.cos φ' ∧
    ρ' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * π ∧ 0 ≤ φ' ∧ φ' ≤ π) :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinates_reflection_l3932_393202


namespace NUMINAMATH_CALUDE_days_from_friday_l3932_393267

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def addDays (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | Nat.succ m => nextDay (addDays start m)

theorem days_from_friday :
  addDays DayOfWeek.Friday 72 = DayOfWeek.Sunday :=
by
  sorry

end NUMINAMATH_CALUDE_days_from_friday_l3932_393267


namespace NUMINAMATH_CALUDE_equilateral_triangle_l3932_393233

theorem equilateral_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ a + c > b)
  (side_relation : a^2 + 2*b^2 + c^2 - 2*b*(a + c) = 0) : 
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_l3932_393233


namespace NUMINAMATH_CALUDE_multiply_divide_equality_l3932_393293

theorem multiply_divide_equality : (3.241 * 14) / 100 = 0.45374 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_equality_l3932_393293


namespace NUMINAMATH_CALUDE_two_rolls_give_target_prob_l3932_393218

-- Define a type for a six-sided die
def Die := Fin 6

-- Define the number of sides on the die
def numSides : ℕ := 6

-- Define the target sum
def targetSum : ℕ := 9

-- Define the target probability
def targetProb : ℚ := 1 / 9

-- Function to calculate the number of ways to get a sum of 9 with n rolls
def waysToGetSum (n : ℕ) : ℕ := sorry

-- Function to calculate the total number of possible outcomes with n rolls
def totalOutcomes (n : ℕ) : ℕ := numSides ^ n

-- Theorem stating that rolling the die twice gives the target probability
theorem two_rolls_give_target_prob :
  ∃ (n : ℕ), (waysToGetSum n : ℚ) / (totalOutcomes n) = targetProb ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_two_rolls_give_target_prob_l3932_393218


namespace NUMINAMATH_CALUDE_equation_solutions_count_l3932_393262

theorem equation_solutions_count :
  ∃! (S : Set ℝ), (∀ x ∈ S, (x^2 - 7)^2 = 25) ∧ S.Finite ∧ S.ncard = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l3932_393262


namespace NUMINAMATH_CALUDE_queen_middle_teachers_l3932_393212

/-- Represents a school with students, teachers, and classes. -/
structure School where
  num_students : ℕ
  classes_per_student : ℕ
  classes_per_teacher : ℕ
  students_per_class : ℕ

/-- Calculates the number of teachers in a school. -/
def num_teachers (s : School) : ℕ :=
  (s.num_students * s.classes_per_student) / (s.students_per_class * s.classes_per_teacher)

/-- Queen Middle School -/
def queen_middle : School :=
  { num_students := 1500
  , classes_per_student := 6
  , classes_per_teacher := 5
  , students_per_class := 25
  }

/-- Theorem stating that Queen Middle School has 72 teachers -/
theorem queen_middle_teachers : num_teachers queen_middle = 72 := by
  sorry

end NUMINAMATH_CALUDE_queen_middle_teachers_l3932_393212


namespace NUMINAMATH_CALUDE_parabola_smallest_a_l3932_393292

theorem parabola_smallest_a (a b c : ℝ) : 
  a > 0 ∧ 
  b^2 - 4*a*c = 7 ∧
  (∀ x y : ℝ, y = a*x^2 + b*x + c ↔ y + 5/9 = a*(x - 1/3)^2) →
  a ≥ 63/20 ∧ ∃ b c : ℝ, b^2 - 4*a*c = 7 ∧ (∀ x y : ℝ, y = a*x^2 + b*x + c ↔ y + 5/9 = 63/20*(x - 1/3)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_smallest_a_l3932_393292


namespace NUMINAMATH_CALUDE_sneakers_discount_proof_l3932_393235

/-- Calculates the membership discount percentage given the original price,
    coupon discount, and final price after both discounts are applied. -/
def membership_discount_percentage (original_price coupon_discount final_price : ℚ) : ℚ :=
  let price_after_coupon := original_price - coupon_discount
  let discount_amount := price_after_coupon - final_price
  (discount_amount / price_after_coupon) * 100

/-- Proves that the membership discount percentage is 10% for the given scenario. -/
theorem sneakers_discount_proof :
  membership_discount_percentage 120 10 99 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sneakers_discount_proof_l3932_393235


namespace NUMINAMATH_CALUDE_customers_in_other_countries_l3932_393294

/-- Given a cell phone company with a total of 7422 customers,
    of which 723 live in the United States,
    prove that 6699 customers live in other countries. -/
theorem customers_in_other_countries
  (total : ℕ)
  (usa : ℕ)
  (h1 : total = 7422)
  (h2 : usa = 723) :
  total - usa = 6699 := by
  sorry

end NUMINAMATH_CALUDE_customers_in_other_countries_l3932_393294


namespace NUMINAMATH_CALUDE_sequence_difference_proof_l3932_393208

def arithmetic_sequence_sum (a1 n d : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem sequence_difference_proof : 
  let n1 := (2298 - 2204) / 2 + 1
  let n2 := (400 - 306) / 2 + 1
  arithmetic_sequence_sum 2204 n1 2 - arithmetic_sequence_sum 306 n2 2 = 91056 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_proof_l3932_393208


namespace NUMINAMATH_CALUDE_coin_distribution_l3932_393207

theorem coin_distribution (a d : ℤ) : 
  (a - 3*d) + (a - 2*d) = 58 ∧ 
  (a + d) + (a + 2*d) + (a + 3*d) = 60 →
  (a - 2*d = 28 ∧ a = 24) := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_l3932_393207


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3932_393236

theorem arithmetic_equality : 8 / 2 - 5 + 3^2 * 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3932_393236


namespace NUMINAMATH_CALUDE_f_properties_l3932_393251

def f (x : ℝ) : ℝ := 1 - |x - x^2|

theorem f_properties :
  (∀ x, f x ≤ 1) ∧
  (f 0 = 1 ∧ f 1 = 1) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 1 - x + x^2) ∧
  (∀ x, (x < 0 ∨ x > 1) → f x = 1 + x - x^2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3932_393251


namespace NUMINAMATH_CALUDE_percent_cat_owners_l3932_393296

def total_students : ℕ := 500
def cat_owners : ℕ := 90

theorem percent_cat_owners : (cat_owners : ℚ) / total_students * 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_percent_cat_owners_l3932_393296


namespace NUMINAMATH_CALUDE_sports_club_size_l3932_393270

/-- The number of members in a sports club -/
def sports_club_members (badminton tennis both neither : ℕ) : ℕ :=
  badminton + tennis - both + neither

/-- Theorem: The sports club has 28 members -/
theorem sports_club_size :
  ∃ (badminton tennis both neither : ℕ),
    badminton = 17 ∧
    tennis = 19 ∧
    both = 10 ∧
    neither = 2 ∧
    sports_club_members badminton tennis both neither = 28 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_size_l3932_393270


namespace NUMINAMATH_CALUDE_square_root_of_nine_l3932_393241

-- Define the square root function
def square_root (x : ℝ) : Set ℝ := {y : ℝ | y * y = x}

-- State the theorem
theorem square_root_of_nine : square_root 9 = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l3932_393241


namespace NUMINAMATH_CALUDE_concert_group_discount_l3932_393261

theorem concert_group_discount (P : ℝ) (h : P > 0) :
  ∃ (x : ℕ), 3 * P = (3 + x) * (0.75 * P) ∧ 3 + x = 4 := by
  sorry

end NUMINAMATH_CALUDE_concert_group_discount_l3932_393261


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_l3932_393222

/-- Represents the daily rainfall probabilities and amounts -/
structure DailyRainfall where
  no_rain_prob : Real
  light_rain_prob : Real
  heavy_rain_prob : Real
  light_rain_amount : Real
  heavy_rain_amount : Real

/-- Calculates the expected rainfall for a single day -/
def expected_daily_rainfall (d : DailyRainfall) : Real :=
  d.no_rain_prob * 0 + d.light_rain_prob * d.light_rain_amount + d.heavy_rain_prob * d.heavy_rain_amount

/-- Theorem: Expected total rainfall for a week -/
theorem expected_weekly_rainfall (d : DailyRainfall)
  (h1 : d.no_rain_prob = 0.2)
  (h2 : d.light_rain_prob = 0.3)
  (h3 : d.heavy_rain_prob = 0.5)
  (h4 : d.light_rain_amount = 2)
  (h5 : d.heavy_rain_amount = 8)
  (h6 : d.no_rain_prob + d.light_rain_prob + d.heavy_rain_prob = 1) :
  7 * (expected_daily_rainfall d) = 32.2 := by
  sorry

#eval 7 * (0.2 * 0 + 0.3 * 2 + 0.5 * 8)

end NUMINAMATH_CALUDE_expected_weekly_rainfall_l3932_393222


namespace NUMINAMATH_CALUDE_triangle_side_length_l3932_393238

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  a = 1 ∧
  A = π / 6 ∧
  B = π / 3 →
  b = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3932_393238


namespace NUMINAMATH_CALUDE_marbles_remainder_l3932_393226

theorem marbles_remainder (a b c : ℤ) 
  (ha : a % 8 = 5)
  (hb : b % 8 = 7)
  (hc : c % 8 = 2) : 
  (a + b + c) % 8 = 6 := by
sorry

end NUMINAMATH_CALUDE_marbles_remainder_l3932_393226


namespace NUMINAMATH_CALUDE_product_of_numbers_l3932_393264

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 20) (sum_squares_eq : x^2 + y^2 = 200) :
  x * y = 100 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3932_393264


namespace NUMINAMATH_CALUDE_derivative_equals_one_l3932_393244

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2
  else 2^x

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ :=
  if x > 0 then 1 / (x * Real.log 2)
  else 2^x * Real.log 2

-- Theorem statement
theorem derivative_equals_one (a : ℝ) :
  f_derivative a = 1 ↔ a = 1 / Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_derivative_equals_one_l3932_393244


namespace NUMINAMATH_CALUDE_unique_solution_sin_equation_l3932_393263

theorem unique_solution_sin_equation :
  ∃! x : ℝ, x = Real.sin x + 1993 := by sorry

end NUMINAMATH_CALUDE_unique_solution_sin_equation_l3932_393263


namespace NUMINAMATH_CALUDE_unique_grid_with_star_one_l3932_393278

/-- Represents a 5x5 grid of integers -/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Checks if a given row in the grid contains 1, 2, 3, 4, 5 without repetition -/
def valid_row (g : Grid) (row : Fin 5) : Prop :=
  ∀ n : Fin 5, ∃! col : Fin 5, g row col = n

/-- Checks if a given column in the grid contains 1, 2, 3, 4, 5 without repetition -/
def valid_column (g : Grid) (col : Fin 5) : Prop :=
  ∀ n : Fin 5, ∃! row : Fin 5, g row col = n

/-- Checks if a given 3x3 box in the grid contains 1, 2, 3, 4, 5 without repetition -/
def valid_box (g : Grid) (box_row box_col : Fin 2) : Prop :=
  ∀ n : Fin 5, ∃! (row col : Fin 3), g (3 * box_row + row) (3 * box_col + col) = n

/-- Checks if the entire grid is valid according to the problem constraints -/
def valid_grid (g : Grid) : Prop :=
  (∀ row : Fin 5, valid_row g row) ∧
  (∀ col : Fin 5, valid_column g col) ∧
  (∀ box_row box_col : Fin 2, valid_box g box_row box_col)

/-- The position of the cell marked with a star -/
def star_position : Fin 5 × Fin 5 := ⟨2, 4⟩

/-- The main theorem: There exists a unique valid grid where the star cell contains 1 -/
theorem unique_grid_with_star_one :
  ∃! g : Grid, valid_grid g ∧ g star_position.1 star_position.2 = 1 := by sorry

end NUMINAMATH_CALUDE_unique_grid_with_star_one_l3932_393278


namespace NUMINAMATH_CALUDE_sequence_properties_l3932_393281

/-- Given a sequence a with sum S, prove specific properties -/
theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ n, S (n + 1) = S n + a (n + 1))
  (h2 : a 2 = 3)
  (h3 : ∀ n, S (n + 1) = 2 * S n + n) :
  (∀ n, a (n + 1) > S n) ∧ 
  (∀ n ≥ 2, S (n + 1) / (2^(n + 1)) > S n / (2^n)) := by
sorry

end NUMINAMATH_CALUDE_sequence_properties_l3932_393281


namespace NUMINAMATH_CALUDE_trash_cans_veterans_park_l3932_393299

/-- The number of trash cans in Veteran's Park after the transfer -/
def final_trash_cans_veterans_park (initial_veterans_park : ℕ) (initial_central_park : ℕ) : ℕ :=
  initial_veterans_park + initial_central_park / 2

/-- Theorem stating the final number of trash cans in Veteran's Park -/
theorem trash_cans_veterans_park :
  ∃ (initial_central_park : ℕ),
    (initial_central_park = 24 / 2 + 8) ∧
    (final_trash_cans_veterans_park 24 initial_central_park = 34) := by
  sorry

#check trash_cans_veterans_park

end NUMINAMATH_CALUDE_trash_cans_veterans_park_l3932_393299


namespace NUMINAMATH_CALUDE_marble_box_capacity_l3932_393230

theorem marble_box_capacity (jack_capacity : ℕ) (lucy_scale : ℕ) : 
  jack_capacity = 50 → lucy_scale = 3 → 
  (lucy_scale ^ 3) * jack_capacity = 1350 := by
  sorry

end NUMINAMATH_CALUDE_marble_box_capacity_l3932_393230


namespace NUMINAMATH_CALUDE_sqrt_difference_of_squares_l3932_393204

theorem sqrt_difference_of_squares : (Real.sqrt 5 + Real.sqrt 3) * (Real.sqrt 5 - Real.sqrt 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_of_squares_l3932_393204


namespace NUMINAMATH_CALUDE_smallest_k_value_l3932_393225

theorem smallest_k_value (a b c d x y z t : ℝ) :
  ∃ k : ℝ, k = 1 ∧ 
  (∀ k' : ℝ, (a * d - b * c + y * z - x * t + (a + c) * (y + t) - (b + d) * (x + z) ≤ 
    k' * (Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (x^2 + y^2) + Real.sqrt (z^2 + t^2))^2) → 
  k ≤ k') ∧
  (a * d - b * c + y * z - x * t + (a + c) * (y + t) - (b + d) * (x + z) ≤ 
    k * (Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (x^2 + y^2) + Real.sqrt (z^2 + t^2))^2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_value_l3932_393225


namespace NUMINAMATH_CALUDE_root_difference_implies_k_l3932_393279

theorem root_difference_implies_k (k : ℝ) : 
  (∀ r s : ℝ, r^2 + k*r + 6 = 0 ∧ s^2 + k*s + 6 = 0 → 
    ∃ r' s' : ℝ, r'^2 - k*r' + 6 = 0 ∧ s'^2 - k*s' + 6 = 0 ∧ 
    r' = r + 5 ∧ s' = s + 5) → 
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_l3932_393279


namespace NUMINAMATH_CALUDE_ellipse_dot_product_bounds_l3932_393214

/-- The ellipse with equation x²/9 + y²/8 = 1 -/
def Ellipse (x y : ℝ) : Prop := x^2/9 + y^2/8 = 1

/-- The left focus of the ellipse -/
def F1 : ℝ × ℝ := (-1, 0)

/-- The right focus of the ellipse -/
def F2 : ℝ × ℝ := (1, 0)

/-- The dot product of vectors EF₁ and EF₂ -/
def dotProduct (E : ℝ × ℝ) : ℝ :=
  let (x, y) := E
  (-1-x)*(1-x) + (-y)*(-y)

theorem ellipse_dot_product_bounds :
  ∀ E : ℝ × ℝ, Ellipse E.1 E.2 → 7 ≤ dotProduct E ∧ dotProduct E ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_bounds_l3932_393214


namespace NUMINAMATH_CALUDE_polynomial_real_root_iff_b_negative_l3932_393253

/-- The polynomial x^3 + bx^2 - x + b = 0 has at least one real root if and only if b < 0 -/
theorem polynomial_real_root_iff_b_negative :
  ∀ b : ℝ, (∃ x : ℝ, x^3 + b*x^2 - x + b = 0) ↔ b < 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_iff_b_negative_l3932_393253


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_seven_l3932_393227

theorem smallest_positive_multiple_of_seven (x : ℕ) : 
  (∃ k : ℕ, x = 7 * k) → -- x is a positive multiple of 7
  x^2 > 144 →            -- x^2 > 144
  x < 25 →               -- x < 25
  x = 14 :=              -- x = 14 is the smallest value satisfying all conditions
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_seven_l3932_393227


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3932_393277

/-- The vertex of the parabola y = x^2 - 4x + a lies on the line y = -4x - 1 -/
def vertex_on_line (a : ℝ) : Prop :=
  ∃ x y : ℝ, y = x^2 - 4*x + a ∧ y = -4*x - 1

/-- The coordinates of the vertex of the parabola y = x^2 - 4x + a -/
def vertex_coordinates (a : ℝ) : ℝ × ℝ := (2, -9)

theorem parabola_vertex_coordinates (a : ℝ) :
  vertex_on_line a → vertex_coordinates a = (2, -9) := by
  sorry


end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3932_393277
