import Mathlib

namespace NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l931_93146

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℝ × ℝ := (-36, 26)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℝ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: -4y = 3x + 4 -/
def line2 (x y : ℝ) : Prop := -4 * y = 3 * x + 4

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_both_lines :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℝ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l931_93146


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l931_93186

theorem triangle_abc_properties (a b : ℝ) (cosB : ℝ) (S : ℝ) :
  a = 5 →
  b = 6 →
  cosB = -4/5 →
  S = 15 * Real.sqrt 7 / 4 →
  ∃ (A R c : ℝ),
    (A = π/6 ∧ R = 5) ∧
    (c = 4 ∨ c = Real.sqrt 106) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l931_93186


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l931_93101

theorem geometric_sequence_first_term
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r^2 = 3) -- third term is 3
  (h2 : a * r^4 = 27) -- fifth term is 27
  : a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l931_93101


namespace NUMINAMATH_CALUDE_max_tan_B_in_triangle_l931_93182

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

end NUMINAMATH_CALUDE_max_tan_B_in_triangle_l931_93182


namespace NUMINAMATH_CALUDE_negative_double_inequality_l931_93167

theorem negative_double_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_double_inequality_l931_93167


namespace NUMINAMATH_CALUDE_canal_bottom_width_l931_93159

/-- Given a trapezoidal canal cross-section with the following properties:
  - Top width: 12 meters
  - Depth: 84 meters
  - Area: 840 square meters
  Prove that the bottom width is 8 meters. -/
theorem canal_bottom_width (top_width : ℝ) (depth : ℝ) (area : ℝ) (bottom_width : ℝ) :
  top_width = 12 →
  depth = 84 →
  area = 840 →
  area = (1/2) * (top_width + bottom_width) * depth →
  bottom_width = 8 := by
sorry

end NUMINAMATH_CALUDE_canal_bottom_width_l931_93159


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l931_93113

theorem simplify_trig_expression :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) =
  Real.tan (60 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l931_93113


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l931_93168

theorem smallest_integer_with_remainders : ∃! M : ℕ,
  (M > 0) ∧
  (M % 3 = 2) ∧
  (M % 4 = 3) ∧
  (M % 5 = 4) ∧
  (M % 6 = 5) ∧
  (M % 7 = 6) ∧
  (M % 11 = 10) ∧
  (∀ n : ℕ, n > 0 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 ∧ n % 11 = 10 → n ≥ M) :=
by
  sorry

#eval Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 11)))) - 1

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l931_93168


namespace NUMINAMATH_CALUDE_transport_percentage_l931_93177

/-- Calculate the percentage of income spent on public transport -/
theorem transport_percentage (income : ℝ) (remaining : ℝ) 
  (h1 : income = 2000)
  (h2 : remaining = 1900) :
  (income - remaining) / income * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_transport_percentage_l931_93177


namespace NUMINAMATH_CALUDE_jack_evening_emails_l931_93171

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 9

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 10

/-- The difference between morning and evening emails -/
def morning_evening_difference : ℕ := 2

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := morning_emails - morning_evening_difference

theorem jack_evening_emails :
  evening_emails = 7 :=
sorry

end NUMINAMATH_CALUDE_jack_evening_emails_l931_93171


namespace NUMINAMATH_CALUDE_deepak_age_l931_93173

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age -/
theorem deepak_age (rahul_ratio : ℕ) (deepak_ratio : ℕ) (rahul_future_age : ℕ) 
    (h1 : rahul_ratio = 4)
    (h2 : deepak_ratio = 3)
    (h3 : rahul_future_age = 26)
    (h4 : rahul_ratio * (rahul_future_age - 10) = deepak_ratio * deepak_present_age) :
  deepak_present_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l931_93173


namespace NUMINAMATH_CALUDE_game_cost_l931_93149

/-- The cost of a new game given initial money, birthday gift, and remaining money -/
theorem game_cost (initial : ℕ) (gift : ℕ) (remaining : ℕ) : 
  initial = 16 → gift = 28 → remaining = 19 → initial + gift - remaining = 25 := by
  sorry

end NUMINAMATH_CALUDE_game_cost_l931_93149


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l931_93163

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l931_93163


namespace NUMINAMATH_CALUDE_find_transmitter_probability_l931_93193

/-- The number of possible government vehicle license plates starting with 79 -/
def total_vehicles : ℕ := 900

/-- The number of vehicles police can inspect per hour -/
def inspection_rate : ℕ := 6

/-- The search time in hours -/
def search_time : ℕ := 3

/-- The probability of finding the transmitter within the given search time -/
theorem find_transmitter_probability :
  (inspection_rate * search_time : ℚ) / total_vehicles = 1 / 50 := by
  sorry

end NUMINAMATH_CALUDE_find_transmitter_probability_l931_93193


namespace NUMINAMATH_CALUDE_prob_at_least_two_fruits_l931_93196

/-- The number of fruit types available -/
def num_fruit_types : ℕ := 4

/-- The number of meals in a day -/
def num_meals : ℕ := 3

/-- The probability of choosing a specific fruit for one meal -/
def prob_one_fruit : ℚ := 1 / num_fruit_types

/-- The probability of eating the same fruit for all meals -/
def prob_same_fruit : ℚ := prob_one_fruit ^ num_meals

/-- The probability of eating at least two different kinds of fruit in a day -/
theorem prob_at_least_two_fruits : 
  1 - (num_fruit_types : ℚ) * prob_same_fruit = 15 / 16 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_fruits_l931_93196


namespace NUMINAMATH_CALUDE_moon_speed_km_per_hour_l931_93147

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_sec : ℚ := 103/100

/-- Converts speed from km/s to km/h -/
def convert_km_per_sec_to_km_per_hour (speed_km_per_sec : ℚ) : ℚ :=
  speed_km_per_sec * seconds_per_hour

theorem moon_speed_km_per_hour :
  convert_km_per_sec_to_km_per_hour moon_speed_km_per_sec = 3708 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_km_per_hour_l931_93147


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l931_93148

theorem polynomial_divisibility (W : ℕ → ℤ) :
  (∀ n : ℕ, (2^n - 1) % W n = 0) →
  (∀ n : ℕ, W n = 1 ∨ W n = -1 ∨ W n = 2*n - 1 ∨ W n = -2*n + 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l931_93148


namespace NUMINAMATH_CALUDE_shaded_area_l931_93129

/-- The shaded area in a geometric configuration --/
theorem shaded_area (AB BC : ℝ) (h1 : AB = Real.sqrt ((8 + Real.sqrt (64 - π^2)) / π))
  (h2 : BC = Real.sqrt ((8 - Real.sqrt (64 - π^2)) / π)) :
  (π / 4) * (AB^2 + BC^2) - AB * BC = 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_l931_93129


namespace NUMINAMATH_CALUDE_minimum_distance_problem_l931_93145

open Real

theorem minimum_distance_problem (a : ℝ) : 
  (∃ x₀ : ℝ, (x₀ - a)^2 + (log (3 * x₀) - 3 * a)^2 ≤ 1/10) → a = 1/30 := by
  sorry

end NUMINAMATH_CALUDE_minimum_distance_problem_l931_93145


namespace NUMINAMATH_CALUDE_tangent_line_and_extreme_values_l931_93139

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - 4

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Theorem statement
theorem tangent_line_and_extreme_values :
  ∃ (a b : ℝ),
  (f a b 2 = -2) ∧
  (f' a b 2 = 1) ∧
  (a = -4 ∧ b = 5) ∧
  (∀ x : ℝ, f (-4) 5 x ≤ -2) ∧
  (f (-4) 5 1 = -2) ∧
  (∀ x : ℝ, f (-4) 5 x ≥ -58/27) ∧
  (f (-4) 5 (5/3) = -58/27) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_and_extreme_values_l931_93139


namespace NUMINAMATH_CALUDE_radius_of_circle_Q_l931_93125

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB : ℝ)
  (AC : ℝ)
  (BC : ℝ)

/-- Circle with center and radius -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Tangency between a circle and a line segment -/
def IsTangent (c : Circle) (p q : ℝ × ℝ) : Prop := sorry

/-- External tangency between two circles -/
def IsExternallyTangent (c1 c2 : Circle) : Prop := sorry

/-- Circle lies inside a triangle -/
def CircleInsideTriangle (c : Circle) (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem radius_of_circle_Q (t : Triangle) (p q : Circle) :
  t.AB = 144 ∧ t.AC = 144 ∧ t.BC = 80 ∧
  p.radius = 24 ∧
  IsTangent p (0, 0) (t.AC, 0) ∧
  IsTangent p (t.BC, 0) (0, 0) ∧
  IsExternallyTangent p q ∧
  IsTangent q (0, 0) (t.AB, 0) ∧
  IsTangent q (t.BC, 0) (0, 0) ∧
  CircleInsideTriangle q t →
  q.radius = 64 - 12 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_circle_Q_l931_93125


namespace NUMINAMATH_CALUDE_correct_statements_l931_93108

-- Define the types of relationships
inductive Relationship
| Function
| Correlation

-- Define the types of analysis methods
inductive AnalysisMethod
| Regression

-- Define the properties of relationships
def isDeterministic (r : Relationship) : Prop :=
  match r with
  | Relationship.Function => True
  | Relationship.Correlation => False

-- Define the properties of analysis methods
def isCommonlyUsedFor (m : AnalysisMethod) (r : Relationship) : Prop :=
  match m, r with
  | AnalysisMethod.Regression, Relationship.Correlation => True
  | _, _ => False

-- Theorem to prove
theorem correct_statements :
  isDeterministic Relationship.Function ∧
  ¬isDeterministic Relationship.Correlation ∧
  isCommonlyUsedFor AnalysisMethod.Regression Relationship.Correlation :=
by sorry


end NUMINAMATH_CALUDE_correct_statements_l931_93108


namespace NUMINAMATH_CALUDE_lucas_purchase_problem_l931_93110

theorem lucas_purchase_problem :
  ∀ (a b c : ℕ),
    a + b + c = 50 →
    50 * a + 400 * b + 500 * c = 10000 →
    a = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_lucas_purchase_problem_l931_93110


namespace NUMINAMATH_CALUDE_george_sticker_count_l931_93158

/-- Given the following sticker counts:
  * Dan has 2 times as many stickers as Tom
  * Tom has 3 times as many stickers as Bob
  * George has 5 times as many stickers as Dan
  * Bob has 12 stickers
  Prove that George has 360 stickers -/
theorem george_sticker_count :
  ∀ (bob tom dan george : ℕ),
    dan = 2 * tom →
    tom = 3 * bob →
    george = 5 * dan →
    bob = 12 →
    george = 360 := by
  sorry

end NUMINAMATH_CALUDE_george_sticker_count_l931_93158


namespace NUMINAMATH_CALUDE_opposite_minus_six_l931_93117

theorem opposite_minus_six (a : ℤ) : a = -(-6) → 1 - a = -5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_minus_six_l931_93117


namespace NUMINAMATH_CALUDE_sarah_copies_pages_l931_93194

theorem sarah_copies_pages (people meeting_size copies_per_person contract_pages : ℕ) 
  (h1 : meeting_size = 15)
  (h2 : copies_per_person = 5)
  (h3 : contract_pages = 35) :
  people = meeting_size * copies_per_person * contract_pages :=
by sorry

end NUMINAMATH_CALUDE_sarah_copies_pages_l931_93194


namespace NUMINAMATH_CALUDE_worm_length_difference_l931_93197

theorem worm_length_difference : 
  let longer_worm : ℝ := 0.8
  let shorter_worm : ℝ := 0.1
  longer_worm - shorter_worm = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_worm_length_difference_l931_93197


namespace NUMINAMATH_CALUDE_even_function_derivative_zero_l931_93174

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- Theorem: If f is an even function and its derivative exists, then f'(0) = 0 -/
theorem even_function_derivative_zero (f : ℝ → ℝ) (hf : IsEven f) (hf' : Differentiable ℝ f) :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_derivative_zero_l931_93174


namespace NUMINAMATH_CALUDE_candy_cost_l931_93179

theorem candy_cost (packs : ℕ) (paid : ℕ) (change : ℕ) (h1 : packs = 3) (h2 : paid = 20) (h3 : change = 11) :
  (paid - change) / packs = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_l931_93179


namespace NUMINAMATH_CALUDE_value_of_expression_l931_93191

theorem value_of_expression (a b : ℝ) (h : 2 * a - b = -1) : 
  4 * a - 2 * b + 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l931_93191


namespace NUMINAMATH_CALUDE_point_on_y_axis_l931_93106

/-- A point on the y-axis has an x-coordinate of 0 -/
axiom y_axis_x_zero (x y : ℝ) : (x, y) ∈ Set.range (λ t : ℝ => (0, t)) ↔ x = 0

/-- The point A with coordinates (2-a, -3a+1) lies on the y-axis -/
def A_on_y_axis (a : ℝ) : Prop := (2 - a, -3 * a + 1) ∈ Set.range (λ t : ℝ => (0, t))

theorem point_on_y_axis (a : ℝ) (h : A_on_y_axis a) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l931_93106


namespace NUMINAMATH_CALUDE_kids_stayed_home_l931_93130

theorem kids_stayed_home (camp_kids : ℕ) (additional_home_kids : ℕ) 
  (h1 : camp_kids = 202958)
  (h2 : additional_home_kids = 574664) :
  camp_kids + additional_home_kids = 777622 := by
  sorry

end NUMINAMATH_CALUDE_kids_stayed_home_l931_93130


namespace NUMINAMATH_CALUDE_median_sum_ge_four_times_circumradius_l931_93102

/-- A triangle in a 2D Euclidean space --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The radius of the circumscribed circle of a triangle --/
def circumradius (t : Triangle) : ℝ := sorry

/-- The length of a median of a triangle --/
def median_length (t : Triangle) (vertex : Fin 3) : ℝ := sorry

/-- Predicate to check if a triangle is not obtuse --/
def is_not_obtuse (t : Triangle) : Prop := sorry

/-- Theorem: For any non-obtuse triangle, the sum of its three medians
    is greater than or equal to four times the radius of its circumscribed circle --/
theorem median_sum_ge_four_times_circumradius (t : Triangle) :
  is_not_obtuse t →
  (median_length t 0) + (median_length t 1) + (median_length t 2) ≥ 4 * (circumradius t) := by
  sorry

end NUMINAMATH_CALUDE_median_sum_ge_four_times_circumradius_l931_93102


namespace NUMINAMATH_CALUDE_special_shape_perimeter_l931_93195

/-- A shape with right angles, a base of 12 feet, 10 congruent sides of 2 feet each, and an area of 132 square feet -/
structure SpecialShape where
  base : ℝ
  congruent_side : ℝ
  num_congruent_sides : ℕ
  area : ℝ
  base_eq : base = 12
  congruent_side_eq : congruent_side = 2
  num_congruent_sides_eq : num_congruent_sides = 10
  area_eq : area = 132

/-- The perimeter of the SpecialShape is 54 feet -/
theorem special_shape_perimeter (s : SpecialShape) : 
  s.base + s.congruent_side * s.num_congruent_sides + 4 + 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_special_shape_perimeter_l931_93195


namespace NUMINAMATH_CALUDE_completing_square_transformation_l931_93141

theorem completing_square_transformation :
  ∃ (m n : ℝ), (∀ x : ℝ, x^2 - 4*x - 4 = 0 ↔ (x + m)^2 = n) ∧ m = -2 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l931_93141


namespace NUMINAMATH_CALUDE_fraction_power_equality_l931_93122

theorem fraction_power_equality : (81000 ^ 5) / (27000 ^ 5) = 243 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l931_93122


namespace NUMINAMATH_CALUDE_wades_total_spend_l931_93190

/-- Wade's purchase at a rest stop -/
def wades_purchase : ℕ → ℕ → ℕ → ℕ → ℕ := fun num_sandwiches sandwich_price num_drinks drink_price =>
  num_sandwiches * sandwich_price + num_drinks * drink_price

theorem wades_total_spend :
  wades_purchase 3 6 2 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_wades_total_spend_l931_93190


namespace NUMINAMATH_CALUDE_fraction_sum_product_equality_l931_93189

theorem fraction_sum_product_equality (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : a / b + c / d = (a / b) * (c / d)) : 
  b / a + d / c = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_product_equality_l931_93189


namespace NUMINAMATH_CALUDE_factoring_expression_l931_93156

theorem factoring_expression (x : ℝ) :
  (12 * x^6 + 40 * x^4 - 6) - (2 * x^6 - 6 * x^4 - 6) = 2 * x^4 * (5 * x^2 + 23) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l931_93156


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l931_93143

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l931_93143


namespace NUMINAMATH_CALUDE_similar_triangles_problem_l931_93192

/-- Triangle similarity -/
structure SimilarTriangles (G H I J K L : ℝ × ℝ) : Prop where
  similar : True  -- Placeholder for similarity condition

/-- Angle measure in degrees -/
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

theorem similar_triangles_problem 
  (G H I J K L : ℝ × ℝ) 
  (sim : SimilarTriangles G H I J K L)
  (gh_length : dist G H = 8)
  (hi_length : dist H I = 16)
  (kl_length : dist K L = 24)
  (ghi_angle : angle_measure G H I = 30) :
  dist J K = 12 ∧ angle_measure J K L = 30 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_problem_l931_93192


namespace NUMINAMATH_CALUDE_power_function_through_point_is_odd_l931_93188

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem power_function_through_point_is_odd
  (f : ℝ → ℝ)
  (h1 : is_power_function f)
  (h2 : f (Real.sqrt 3 / 3) = Real.sqrt 3) :
  is_odd_function f :=
sorry

end NUMINAMATH_CALUDE_power_function_through_point_is_odd_l931_93188


namespace NUMINAMATH_CALUDE_ordering_abc_l931_93157

-- Define the logarithm base 2
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- Define a, b, and c
noncomputable def a : ℝ := log2 9 - log2 (Real.sqrt 3)
noncomputable def b : ℝ := 1 + log2 (Real.sqrt 7)
noncomputable def c : ℝ := 1/2 + log2 (Real.sqrt 13)

-- Theorem statement
theorem ordering_abc : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l931_93157


namespace NUMINAMATH_CALUDE_cos_150_degrees_l931_93119

theorem cos_150_degrees : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l931_93119


namespace NUMINAMATH_CALUDE_inequality_proof_l931_93164

theorem inequality_proof (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : c * a < c * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l931_93164


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l931_93162

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 9 to base 10 -/
def base9_to_base10 (n : ℕ) : ℕ := sorry

theorem base_conversion_subtraction :
  base8_to_base10 52143 - base9_to_base10 3456 = 19041 := by sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l931_93162


namespace NUMINAMATH_CALUDE_division_remainder_problem_l931_93111

theorem division_remainder_problem : ∃ r, 0 ≤ r ∧ r < 9 ∧ 83 = 9 * 9 + r := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l931_93111


namespace NUMINAMATH_CALUDE_highest_frequency_last_3_groups_l931_93138

/-- Represents the frequency distribution of a sample -/
structure FrequencyDistribution where
  total_sample : ℕ
  num_groups : ℕ
  cumulative_freq_7 : ℚ
  last_3_geometric : Bool
  common_ratio_gt_2 : Bool

/-- Theorem stating the highest frequency in the last 3 groups -/
theorem highest_frequency_last_3_groups
  (fd : FrequencyDistribution)
  (h1 : fd.total_sample = 100)
  (h2 : fd.num_groups = 10)
  (h3 : fd.cumulative_freq_7 = 79/100)
  (h4 : fd.last_3_geometric)
  (h5 : fd.common_ratio_gt_2) :
  ∃ (a r : ℕ),
    r > 2 ∧
    a + a * r + a * r^2 = 21 ∧
    (∀ x : ℕ, x ∈ [a, a * r, a * r^2] → x ≤ 16) ∧
    16 ∈ [a, a * r, a * r^2] :=
sorry

end NUMINAMATH_CALUDE_highest_frequency_last_3_groups_l931_93138


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l931_93123

/-- Proves that the speed of a boat in still water is 24 km/hr, given the conditions -/
theorem boat_speed_in_still_water :
  let stream_speed : ℝ := 4
  let downstream_distance : ℝ := 84
  let downstream_time : ℝ := 3
  let boat_speed : ℝ := (downstream_distance / downstream_time) - stream_speed
  boat_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l931_93123


namespace NUMINAMATH_CALUDE_fixed_points_of_specific_quadratic_min_value_of_ratio_sum_min_value_achieved_l931_93176

-- Define the quadratic function
def quadratic (m n t : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + t

-- Define what it means to be a fixed point
def is_fixed_point (m n t : ℝ) (x : ℝ) : Prop :=
  quadratic m n t x = x

-- Part 1: Fixed points of y = x^2 - x - 3
theorem fixed_points_of_specific_quadratic :
  {x : ℝ | is_fixed_point 1 (-1) (-3) x} = {-1, 3} := by sorry

-- Part 2: Minimum value of x1/x2 + x2/x1
theorem min_value_of_ratio_sum :
  ∀ a x₁ x₂ : ℝ,
    x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
    is_fixed_point 2 (-(2+a)) (a-1) x₁ →
    is_fixed_point 2 (-(2+a)) (a-1) x₂ →
    (x₁ / x₂ + x₂ / x₁) ≥ 6 := by sorry

-- The minimum value is achieved when a = 5
theorem min_value_achieved :
  ∃ a x₁ x₂ : ℝ,
    x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    is_fixed_point 2 (-(2+a)) (a-1) x₁ ∧
    is_fixed_point 2 (-(2+a)) (a-1) x₂ ∧
    x₁ / x₂ + x₂ / x₁ = 6 := by sorry

end NUMINAMATH_CALUDE_fixed_points_of_specific_quadratic_min_value_of_ratio_sum_min_value_achieved_l931_93176


namespace NUMINAMATH_CALUDE_pages_per_day_l931_93115

theorem pages_per_day (total_pages : ℕ) (weeks : ℕ) (days_per_week : ℕ) 
  (h1 : total_pages = 2100)
  (h2 : weeks = 7)
  (h3 : days_per_week = 3) :
  total_pages / (weeks * days_per_week) = 100 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_day_l931_93115


namespace NUMINAMATH_CALUDE_complex_equation_solution_l931_93100

/-- Given a real number a such that (2+ai)/(1+i) = 3+i, prove that a = 4 -/
theorem complex_equation_solution (a : ℝ) : (2 + a * Complex.I) / (1 + Complex.I) = 3 + Complex.I → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l931_93100


namespace NUMINAMATH_CALUDE_functional_equation_solution_l931_93150

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l931_93150


namespace NUMINAMATH_CALUDE_equality_from_cubic_equation_l931_93107

theorem equality_from_cubic_equation (a b : ℕ) 
  (h : a^3 + a + 4*b^2 = 4*a*b + b + b*a^2) : a = b := by
  sorry

end NUMINAMATH_CALUDE_equality_from_cubic_equation_l931_93107


namespace NUMINAMATH_CALUDE_height_difference_l931_93144

theorem height_difference (amy_height helen_height angela_height : ℕ) : 
  helen_height = amy_height + 3 →
  amy_height = 150 →
  angela_height = 157 →
  angela_height - helen_height = 4 := by
sorry

end NUMINAMATH_CALUDE_height_difference_l931_93144


namespace NUMINAMATH_CALUDE_min_value_expression_l931_93169

theorem min_value_expression (x y : ℝ) (hx : x ≥ 4) (hy : y ≥ -3) :
  x^2 + y^2 - 8*x + 6*y + 20 ≥ -5 ∧
  ∃ (x₀ y₀ : ℝ), x₀ ≥ 4 ∧ y₀ ≥ -3 ∧ x₀^2 + y₀^2 - 8*x₀ + 6*y₀ + 20 = -5 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l931_93169


namespace NUMINAMATH_CALUDE_third_divisor_l931_93120

theorem third_divisor (x : ℕ) 
  (h1 : x - 16 = 136)
  (h2 : 4 ∣ x)
  (h3 : 6 ∣ x)
  (h4 : 10 ∣ x)
  (h5 : ∀ y, y - 16 = 136 ∧ 4 ∣ y ∧ 6 ∣ y ∧ 10 ∣ y → x ≤ y) :
  19 ∣ x ∧ 19 ≠ 4 ∧ 19 ≠ 6 ∧ 19 ≠ 10 :=
sorry

end NUMINAMATH_CALUDE_third_divisor_l931_93120


namespace NUMINAMATH_CALUDE_cindy_added_pens_l931_93118

/-- Proves the number of pens Cindy added given the initial conditions and final result --/
theorem cindy_added_pens (initial_pens : ℕ) (mike_gives : ℕ) (sharon_receives : ℕ) (final_pens : ℕ)
  (h1 : initial_pens = 7)
  (h2 : mike_gives = 22)
  (h3 : sharon_receives = 19)
  (h4 : final_pens = 39) :
  final_pens = initial_pens + mike_gives - sharon_receives + 29 := by
  sorry

#check cindy_added_pens

end NUMINAMATH_CALUDE_cindy_added_pens_l931_93118


namespace NUMINAMATH_CALUDE_square_area_l931_93128

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x^2 + 5*x + 6

-- Define the horizontal line
def horizontal_line : ℝ := 10

-- Theorem statement
theorem square_area : ∃ (x₁ x₂ : ℝ), 
  parabola x₁ = horizontal_line ∧ 
  parabola x₂ = horizontal_line ∧ 
  (x₂ - x₁)^2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l931_93128


namespace NUMINAMATH_CALUDE_childrens_ticket_cost_l931_93133

theorem childrens_ticket_cost :
  let adult_ticket_cost : ℚ := 25
  let total_receipts : ℚ := 7200
  let total_attendance : ℕ := 400
  let adult_attendance : ℕ := 280
  let child_attendance : ℕ := 120
  let child_ticket_cost : ℚ := (total_receipts - (adult_ticket_cost * adult_attendance)) / child_attendance
  child_ticket_cost = 5/3 := by sorry

end NUMINAMATH_CALUDE_childrens_ticket_cost_l931_93133


namespace NUMINAMATH_CALUDE_fifth_month_sales_l931_93154

def sales_1 : ℕ := 4000
def sales_2 : ℕ := 6524
def sales_3 : ℕ := 5689
def sales_4 : ℕ := 7230
def sales_6 : ℕ := 12557
def average_sale : ℕ := 7000
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sale ∧
    sales_5 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sales_l931_93154


namespace NUMINAMATH_CALUDE_function_evaluation_l931_93103

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x, f x = 4*x - 2) : f (-3) = -14 := by
  sorry

end NUMINAMATH_CALUDE_function_evaluation_l931_93103


namespace NUMINAMATH_CALUDE_clara_age_multiple_of_anna_l931_93142

theorem clara_age_multiple_of_anna (anna_current_age clara_current_age : ℕ) 
  (h1 : anna_current_age = 54)
  (h2 : clara_current_age = 80) :
  ∃ (years_ago : ℕ), 
    clara_current_age - years_ago = 3 * (anna_current_age - years_ago) ∧ 
    years_ago = 41 := by
  sorry

end NUMINAMATH_CALUDE_clara_age_multiple_of_anna_l931_93142


namespace NUMINAMATH_CALUDE_b_initial_investment_l931_93160

/-- Represents the business investment problem --/
structure BusinessInvestment where
  a_initial : ℕ  -- A's initial investment
  total_profit : ℕ  -- Total profit at the end of the year
  a_profit : ℕ  -- A's share of the profit
  a_withdraw : ℕ  -- Amount A withdraws after 8 months
  b_advance : ℕ  -- Amount B advances after 8 months

/-- Calculates B's initial investment given the business conditions --/
def calculate_b_initial (bi : BusinessInvestment) : ℕ :=
  sorry

/-- Theorem stating that B's initial investment is 4000 given the problem conditions --/
theorem b_initial_investment (bi : BusinessInvestment) 
  (h1 : bi.a_initial = 3000)
  (h2 : bi.total_profit = 630)
  (h3 : bi.a_profit = 240)
  (h4 : bi.a_withdraw = 1000)
  (h5 : bi.b_advance = 1000) :
  calculate_b_initial bi = 4000 := by
  sorry

end NUMINAMATH_CALUDE_b_initial_investment_l931_93160


namespace NUMINAMATH_CALUDE_sams_nickels_l931_93180

/-- Given Sam's initial nickels and his dad's gift of nickels, calculate Sam's total nickels -/
theorem sams_nickels (initial_nickels dad_gift_nickels : ℕ) :
  initial_nickels = 24 → dad_gift_nickels = 39 →
  initial_nickels + dad_gift_nickels = 63 := by sorry

end NUMINAMATH_CALUDE_sams_nickels_l931_93180


namespace NUMINAMATH_CALUDE_wine_consumption_equations_l931_93112

/-- Represents the wine consumption and intoxication scenario from the Ming Dynasty poem --/
theorem wine_consumption_equations :
  ∃ (x y : ℚ),
    (x + y = 19) ∧
    (3 * x + (1/3) * y = 33) ∧
    (x ≥ 0) ∧ (y ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_wine_consumption_equations_l931_93112


namespace NUMINAMATH_CALUDE_specific_l_shape_perimeter_l931_93183

/-- Represents an L-shaped region formed by congruent squares -/
structure LShapedRegion where
  squareCount : Nat
  topRowCount : Nat
  bottomRowCount : Nat
  totalArea : ℝ

/-- Calculates the perimeter of an L-shaped region -/
def calculatePerimeter (region : LShapedRegion) : ℝ :=
  sorry

/-- Theorem: The perimeter of the specific L-shaped region is 91 cm -/
theorem specific_l_shape_perimeter :
  let region : LShapedRegion := {
    squareCount := 8,
    topRowCount := 3,
    bottomRowCount := 5,
    totalArea := 392
  }
  calculatePerimeter region = 91 := by
  sorry

end NUMINAMATH_CALUDE_specific_l_shape_perimeter_l931_93183


namespace NUMINAMATH_CALUDE_brianas_yield_percentage_l931_93136

theorem brianas_yield_percentage (emma_investment briana_investment : ℝ)
                                 (emma_yield : ℝ)
                                 (investment_period : ℕ)
                                 (roi_difference : ℝ) :
  emma_investment = 300 →
  briana_investment = 500 →
  emma_yield = 0.15 →
  investment_period = 2 →
  roi_difference = 10 →
  briana_investment * (investment_period : ℝ) * (briana_yield / 100) -
  emma_investment * (investment_period : ℝ) * emma_yield = roi_difference →
  briana_yield = 10 :=
by
  sorry

#check brianas_yield_percentage

end NUMINAMATH_CALUDE_brianas_yield_percentage_l931_93136


namespace NUMINAMATH_CALUDE_money_distribution_l931_93124

/-- Given a distribution of money in the ratio 3:5:7 among three people,
    where the second person's share is 1500,
    prove that the difference between the first and third person's shares is 1200. -/
theorem money_distribution (total : ℕ) (share1 share2 share3 : ℕ) :
  share1 + share2 + share3 = total →
  3 * share2 = 5 * share1 →
  7 * share1 = 3 * share3 →
  share2 = 1500 →
  share3 - share1 = 1200 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l931_93124


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_two_distinct_real_roots_l931_93161

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  a * x^2 + b * x + c = 0 ↔ 
    (discriminant > 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) ∧
    (discriminant = 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0) ∧
    (discriminant < 0 → ¬∃ x : ℝ, a * x^2 + b * x + c = 0) :=
by sorry

theorem two_distinct_real_roots :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 2*x1 - 1 = 0 ∧ x2^2 - 2*x2 - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_two_distinct_real_roots_l931_93161


namespace NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_32767_l931_93187

def greatest_prime_divisor (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_greatest_prime_divisor_32767 :
  sum_of_digits (greatest_prime_divisor 32767) = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_32767_l931_93187


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l931_93134

theorem imaginary_part_of_complex_fraction (m : ℝ) : 
  (Complex.im ((2 - Complex.I) * (m + Complex.I)) = 0) → 
  (Complex.im (m * Complex.I / (1 - Complex.I)) = 1) := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l931_93134


namespace NUMINAMATH_CALUDE_ourNumber_decimal_l931_93140

/-- Represents a number in millions, thousands, and ones -/
structure LargeNumber where
  millions : Nat
  thousands : Nat
  ones : Nat

/-- Converts a LargeNumber to its decimal representation -/
def toDecimal (n : LargeNumber) : Nat :=
  n.millions * 1000000 + n.thousands * 1000 + n.ones

/-- The specific large number we're working with -/
def ourNumber : LargeNumber :=
  { millions := 10
  , thousands := 300
  , ones := 50 }

theorem ourNumber_decimal : toDecimal ourNumber = 10300050 := by
  sorry

end NUMINAMATH_CALUDE_ourNumber_decimal_l931_93140


namespace NUMINAMATH_CALUDE_quadratic_even_iff_b_eq_zero_l931_93131

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ :=
  x^2 + b*x + c

/-- Theorem: f(x) = x^2 + bx + c is an even function if and only if b = 0 -/
theorem quadratic_even_iff_b_eq_zero (b c : ℝ) :
  IsEven (f b c) ↔ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_even_iff_b_eq_zero_l931_93131


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l931_93121

def f (x : ℝ) := x^3 + x + 1

theorem root_exists_in_interval :
  ∃ r ∈ Set.Ioo (-1 : ℝ) 0, f r = 0 :=
sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l931_93121


namespace NUMINAMATH_CALUDE_binary_101011_eq_43_l931_93155

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101011_eq_43 : 
  binary_to_decimal [true, true, false, true, false, true] = 43 := by
sorry

end NUMINAMATH_CALUDE_binary_101011_eq_43_l931_93155


namespace NUMINAMATH_CALUDE_triangle_area_change_l931_93114

theorem triangle_area_change (h b : ℝ) (h_pos : h > 0) (b_pos : b > 0) :
  let new_height := 0.6 * h
  let new_base := b * (1 + 40 / 100)
  let original_area := (1 / 2) * b * h
  let new_area := (1 / 2) * new_base * new_height
  new_area = 0.84 * original_area :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_change_l931_93114


namespace NUMINAMATH_CALUDE_square_area_decrease_l931_93181

theorem square_area_decrease (initial_area : ℝ) (side_decrease_percent : ℝ) 
  (h1 : initial_area = 50)
  (h2 : side_decrease_percent = 25) : 
  let new_side := (1 - side_decrease_percent / 100) * Real.sqrt initial_area
  let new_area := new_side * Real.sqrt initial_area
  let percent_decrease := (initial_area - new_area) / initial_area * 100
  percent_decrease = 43.75 := by
sorry

end NUMINAMATH_CALUDE_square_area_decrease_l931_93181


namespace NUMINAMATH_CALUDE_athletes_arrangement_l931_93185

/-- The number of ways to arrange athletes from three teams in a row -/
def arrange_athletes (team_a : ℕ) (team_b : ℕ) (team_c : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial team_a) * (Nat.factorial team_b) * (Nat.factorial team_c)

/-- Theorem: The number of ways to arrange 10 athletes from 3 teams (with 4, 3, and 3 athletes respectively) in a row, where athletes from the same team must sit together, is 5184 -/
theorem athletes_arrangement :
  arrange_athletes 4 3 3 = 5184 :=
by sorry

end NUMINAMATH_CALUDE_athletes_arrangement_l931_93185


namespace NUMINAMATH_CALUDE_ticket_price_increase_l931_93153

theorem ticket_price_increase (original_price : ℝ) (increase_percentage : ℝ) : 
  original_price = 85 → 
  increase_percentage = 20 → 
  original_price * (1 + increase_percentage / 100) = 102 := by
sorry

end NUMINAMATH_CALUDE_ticket_price_increase_l931_93153


namespace NUMINAMATH_CALUDE_fill_tank_times_l931_93172

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboid_volume (length width height : ℝ) : ℝ := length * width * height

/-- Represents the dimensions of the tank -/
def tank_length : ℝ := 30
def tank_width : ℝ := 20
def tank_height : ℝ := 5

/-- Represents the dimensions of the bowl -/
def bowl_length : ℝ := 6
def bowl_width : ℝ := 4
def bowl_height : ℝ := 1

/-- Theorem stating the number of times needed to fill the tank -/
theorem fill_tank_times : 
  (cuboid_volume tank_length tank_width tank_height) / 
  (cuboid_volume bowl_length bowl_width bowl_height) = 125 := by
  sorry

end NUMINAMATH_CALUDE_fill_tank_times_l931_93172


namespace NUMINAMATH_CALUDE_smallest_n_for_unique_k_l931_93199

theorem smallest_n_for_unique_k : ∃ (n : ℕ),
  n > 0 ∧
  (∃! (k : ℤ), (9 : ℚ)/17 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 10/19) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬∃! (k : ℤ), (9 : ℚ)/17 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 10/19) ∧
  n = 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_unique_k_l931_93199


namespace NUMINAMATH_CALUDE_sum_even_10_mod_6_l931_93137

/-- The sum of the first n even numbers starting from 2 -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- The theorem stating that the remainder of the sum of the first 10 even numbers divided by 6 is 2 -/
theorem sum_even_10_mod_6 : sum_even 10 % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_10_mod_6_l931_93137


namespace NUMINAMATH_CALUDE_sum_of_squares_of_solutions_l931_93166

theorem sum_of_squares_of_solutions : ∃ (a b c d : ℝ),
  (|a^2 - 2*a + 1/1004| = 1/502) ∧
  (|b^2 - 2*b + 1/1004| = 1/502) ∧
  (|c^2 - 2*c + 1/1004| = 1/502) ∧
  (|d^2 - 2*d + 1/1004| = 1/502) ∧
  (a^2 + b^2 + c^2 + d^2 = 8050/1008) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_solutions_l931_93166


namespace NUMINAMATH_CALUDE_different_color_probability_l931_93170

theorem different_color_probability (blue_chips yellow_chips : ℕ) 
  (h_blue : blue_chips = 5) (h_yellow : yellow_chips = 7) :
  let total_chips := blue_chips + yellow_chips
  let p_blue := blue_chips / total_chips
  let p_yellow := yellow_chips / total_chips
  p_blue * p_yellow + p_yellow * p_blue = 35 / 72 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l931_93170


namespace NUMINAMATH_CALUDE_complex_coordinate_proof_l931_93127

theorem complex_coordinate_proof (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) :
  z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinate_proof_l931_93127


namespace NUMINAMATH_CALUDE_car_sales_profit_loss_percentage_l931_93178

/-- Calculates the overall profit or loss percentage for two car sales --/
theorem car_sales_profit_loss_percentage 
  (selling_price : ℝ) 
  (gain_percentage : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : selling_price > 0) 
  (h2 : gain_percentage > 0) 
  (h3 : loss_percentage > 0) 
  (h4 : gain_percentage = loss_percentage) : 
  ∃ (loss_percent : ℝ), 
    loss_percent > 0 ∧ 
    loss_percent < gain_percentage ∧
    loss_percent = (2 * selling_price - (selling_price / (1 + gain_percentage / 100) + selling_price / (1 - loss_percentage / 100))) / 
                   (selling_price / (1 + gain_percentage / 100) + selling_price / (1 - loss_percentage / 100)) * 100 := by
  sorry

end NUMINAMATH_CALUDE_car_sales_profit_loss_percentage_l931_93178


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l931_93135

theorem unique_congruence_in_range : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l931_93135


namespace NUMINAMATH_CALUDE_zero_smallest_natural_l931_93109

theorem zero_smallest_natural : ∀ n : ℕ, 0 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_zero_smallest_natural_l931_93109


namespace NUMINAMATH_CALUDE_number_of_students_l931_93105

theorem number_of_students (n : ℕ) : 
  (n : ℝ) * 15 = 7 * 14 + 7 * 16 + 15 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l931_93105


namespace NUMINAMATH_CALUDE_sunglasses_and_hats_probability_l931_93175

theorem sunglasses_and_hats_probability 
  (total_sunglasses : ℕ) 
  (total_hats : ℕ) 
  (prob_sunglasses_given_hat : ℚ) :
  total_sunglasses = 75 →
  total_hats = 50 →
  prob_sunglasses_given_hat = 1 / 5 →
  (total_hats * prob_sunglasses_given_hat : ℚ) / total_sunglasses = 2 / 15 :=
by sorry

end NUMINAMATH_CALUDE_sunglasses_and_hats_probability_l931_93175


namespace NUMINAMATH_CALUDE_cubic_factorization_l931_93104

theorem cubic_factorization (x : ℝ) : x^3 - 6*x^2 + 9*x = x*(x-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l931_93104


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l931_93151

theorem quadratic_inequality_solution (b c : ℝ) : 
  (∀ x, x^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) → c + b = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l931_93151


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l931_93198

/-- Represents the side lengths of an isosceles triangle -/
structure IsoscelesTriangle where
  a : ℝ
  is_positive : 0 < a

/-- Checks if the given side lengths can form a valid triangle -/
def is_valid_triangle (t : IsoscelesTriangle) : Prop :=
  3 + t.a > 6 ∧ 3 + 6 > t.a ∧ t.a + 6 > 3

/-- Calculates the perimeter of the triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  3 + t.a + 6

/-- Theorem: If a valid isosceles triangle can be formed with side lengths 3, a, and 6,
    then its perimeter is 15 -/
theorem isosceles_triangle_perimeter
  (t : IsoscelesTriangle)
  (h : is_valid_triangle t) :
  perimeter t = 15 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l931_93198


namespace NUMINAMATH_CALUDE_floor_2a_eq_floor_a_plus_floor_a_half_l931_93152

theorem floor_2a_eq_floor_a_plus_floor_a_half (a : ℝ) (h : a > 0) :
  ⌊2 * a⌋ = ⌊a⌋ + ⌊a + 1/2⌋ := by sorry

end NUMINAMATH_CALUDE_floor_2a_eq_floor_a_plus_floor_a_half_l931_93152


namespace NUMINAMATH_CALUDE_rectangular_field_length_l931_93126

/-- Represents a rectangular field with a given width and area. -/
structure RectangularField where
  width : ℝ
  area : ℝ

/-- The length of a rectangular field is 10 meters more than its width. -/
def length (field : RectangularField) : ℝ := field.width + 10

/-- The theorem stating that a rectangular field with an area of 171 square meters
    and length 10 meters more than its width has a length of 19 meters. -/
theorem rectangular_field_length (field : RectangularField) 
  (h1 : field.area = 171)
  (h2 : field.area = field.width * (field.width + 10)) :
  length field = 19 := by
  sorry

#check rectangular_field_length

end NUMINAMATH_CALUDE_rectangular_field_length_l931_93126


namespace NUMINAMATH_CALUDE_circle_sum_l931_93132

theorem circle_sum (x y : ℝ) (h : x^2 + y^2 = 8*x - 10*y + 5) : x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_l931_93132


namespace NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l931_93165

theorem polar_to_cartesian_equivalence :
  ∀ (x y ρ θ : ℝ),
    ρ = 2 * Real.sin θ + 4 * Real.cos θ →
    x = ρ * Real.cos θ →
    y = ρ * Real.sin θ →
    (x - 8)^2 + (y - 2)^2 = 68 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l931_93165


namespace NUMINAMATH_CALUDE_fraction_simplification_l931_93116

theorem fraction_simplification : (180 : ℚ) / 16200 = 1 / 90 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l931_93116


namespace NUMINAMATH_CALUDE_crofton_orchestra_max_members_l931_93184

theorem crofton_orchestra_max_members :
  ∀ n : ℕ,
  (25 * n < 1000) →
  (25 * n % 24 = 5) →
  (∀ m : ℕ, (25 * m < 1000) ∧ (25 * m % 24 = 5) → m ≤ n) →
  25 * n = 725 :=
by
  sorry

end NUMINAMATH_CALUDE_crofton_orchestra_max_members_l931_93184
