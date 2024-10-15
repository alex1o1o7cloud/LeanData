import Mathlib

namespace NUMINAMATH_CALUDE_function_value_order_l191_19173

/-- A quadratic function with symmetry about x = 5 -/
structure SymmetricQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  symmetric : ∀ x, a * (5 - x)^2 + b * (5 - x) + c = a * (5 + x)^2 + b * (5 + x) + c

/-- The quadratic function -/
def f (q : SymmetricQuadratic) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- Theorem stating the order of function values -/
theorem function_value_order (q : SymmetricQuadratic) :
  f q (2 * Real.pi) < f q (Real.sqrt 40) ∧ f q (Real.sqrt 40) < f q (5 * Real.sin (π / 4)) := by
  sorry

end NUMINAMATH_CALUDE_function_value_order_l191_19173


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_over_2_l191_19141

noncomputable def f (x : ℝ) : ℝ := Real.sin x / (Real.sin x + Real.cos x)

theorem derivative_f_at_pi_over_2 :
  deriv f (π / 2) = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_over_2_l191_19141


namespace NUMINAMATH_CALUDE_a_value_l191_19196

/-- Custom operation @ for positive integers -/
def custom_op (k j : ℕ+) : ℕ+ :=
  sorry

/-- The value of b -/
def b : ℕ := 2120

/-- The ratio q -/
def q : ℚ := 1/2

/-- The value of a -/
def a : ℕ := 1060

/-- Theorem stating that a = 1060 given the conditions -/
theorem a_value : a = 1060 :=
  sorry

end NUMINAMATH_CALUDE_a_value_l191_19196


namespace NUMINAMATH_CALUDE_constant_term_product_l191_19122

variables (p q r : ℝ[X])

theorem constant_term_product (hp : p.coeff 0 = 5) (hr : r.coeff 0 = -15) (h_prod : r = p * q) :
  q.coeff 0 = -3 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_product_l191_19122


namespace NUMINAMATH_CALUDE_unique_solution_for_digit_equation_l191_19156

theorem unique_solution_for_digit_equation :
  ∃! (A B D E : ℕ),
    (A < 10 ∧ B < 10 ∧ D < 10 ∧ E < 10) ∧  -- Base 10 digits
    (A ≠ B ∧ A ≠ D ∧ A ≠ E ∧ B ≠ D ∧ B ≠ E ∧ D ≠ E) ∧  -- Different digits
    (A^(10*A + A) + 10*A + A = 
      B * 10^15 + B * 10^14 + 9 * 10^13 +
      D * 10^12 + E * 10^11 + D * 10^10 +
      B * 10^9 + E * 10^8 + E * 10^7 +
      B * 10^6 + B * 10^5 + B * 10^4 +
      B * 10^3 + B * 10^2 + E * 10^1 + E * 10^0) ∧
    (A = 3 ∧ B = 5 ∧ D = 0 ∧ E = 6) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_digit_equation_l191_19156


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l191_19120

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The bouncing ball theorem -/
theorem bouncing_ball_distance :
  let initialHeight : ℝ := 120
  let reboundFactor : ℝ := 0.75
  let bounces : ℕ := 5
  totalDistance initialHeight reboundFactor bounces = 612.1875 := by
  sorry

end NUMINAMATH_CALUDE_bouncing_ball_distance_l191_19120


namespace NUMINAMATH_CALUDE_inequality_proof_l191_19121

theorem inequality_proof (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1)
  (h : a / (1 - x) + b / (1 - y) = 1) : 
  (a * y) ^ (1/3 : ℝ) + (b * x) ^ (1/3 : ℝ) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l191_19121


namespace NUMINAMATH_CALUDE_sin_225_degrees_l191_19182

theorem sin_225_degrees : Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_degrees_l191_19182


namespace NUMINAMATH_CALUDE_two_true_propositions_l191_19127

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the parallel and perpendicular relations
def parallel : Plane → Plane → Prop := sorry
def perpendicular : Plane → Plane → Prop := sorry
def parallel_line_plane : Line → Plane → Prop := sorry
def perpendicular_line_plane : Line → Plane → Prop := sorry
def parallel_lines : Line → Line → Prop := sorry
def perpendicular_lines : Line → Line → Prop := sorry

-- Define the original proposition for planes
def original_proposition (α β γ : Plane) : Prop :=
  parallel α β ∧ perpendicular α γ → perpendicular β γ

-- Define the propositions with two planes replaced by lines
def prop_αβ_lines (a b : Line) (γ : Plane) : Prop :=
  parallel_lines a b ∧ perpendicular_line_plane a γ → perpendicular_line_plane b γ

def prop_αγ_lines (a : Line) (β : Plane) (b : Line) : Prop :=
  parallel_line_plane a β ∧ perpendicular_lines a b → perpendicular_line_plane b β

def prop_βγ_lines (α : Plane) (a b : Line) : Prop :=
  parallel_line_plane a α ∧ perpendicular_line_plane α b → perpendicular_lines a b

-- The main theorem
theorem two_true_propositions :
  ∃ (α β γ : Plane),
    original_proposition α β γ = true ∧
    (∀ (a b : Line),
      (prop_αβ_lines a b γ = true ∧ prop_αγ_lines a β b = false ∧ prop_βγ_lines α a b = true) ∨
      (prop_αβ_lines a b γ = true ∧ prop_αγ_lines a β b = true ∧ prop_βγ_lines α a b = false) ∨
      (prop_αβ_lines a b γ = false ∧ prop_αγ_lines a β b = true ∧ prop_βγ_lines α a b = true)) :=
sorry

end NUMINAMATH_CALUDE_two_true_propositions_l191_19127


namespace NUMINAMATH_CALUDE_min_selling_price_theorem_l191_19166

/-- Represents the fruit shop scenario with two batches of fruits. -/
structure FruitShop where
  batch1_price : ℝ  -- Price per kg of first batch
  batch1_quantity : ℝ  -- Quantity of first batch in kg
  batch2_price : ℝ  -- Price per kg of second batch
  batch2_quantity : ℝ  -- Quantity of second batch in kg

/-- Calculates the minimum selling price for remaining fruits to achieve the target profit. -/
def min_selling_price (shop : FruitShop) (target_profit : ℝ) : ℝ :=
  sorry

/-- Theorem stating the minimum selling price for the given scenario. -/
theorem min_selling_price_theorem (shop : FruitShop) :
  shop.batch1_price = 50 ∧
  shop.batch2_price = 55 ∧
  shop.batch1_quantity * shop.batch1_price = 1100 ∧
  shop.batch2_quantity * shop.batch2_price = 1100 ∧
  shop.batch1_quantity = shop.batch2_quantity + 2 ∧
  shop.batch2_price = 1.1 * shop.batch1_price →
  min_selling_price shop 1000 = 60 :=
by sorry

end NUMINAMATH_CALUDE_min_selling_price_theorem_l191_19166


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l191_19152

theorem same_terminal_side_angle : ∃ (θ : ℝ), 
  θ ∈ Set.Icc (-2 * Real.pi) 0 ∧ 
  ∃ (k : ℤ), θ = (52 / 7 : ℝ) * Real.pi + 2 * k * Real.pi ∧
  θ = -(4 / 7 : ℝ) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l191_19152


namespace NUMINAMATH_CALUDE_intersecting_lines_determine_plane_l191_19147

-- Define the concepts of point, line, and plane
variable (Point Line Plane : Type)

-- Define the concept of intersection for lines
variable (intersect : Line → Line → Prop)

-- Define the concept of a line lying on a plane
variable (lieOn : Line → Plane → Prop)

-- Define the concept of a plane containing two lines
variable (contains : Plane → Line → Line → Prop)

-- Theorem: Two intersecting lines determine a unique plane
theorem intersecting_lines_determine_plane
  (l1 l2 : Line)
  (h_intersect : intersect l1 l2)
  : ∃! p : Plane, contains p l1 l2 :=
sorry

end NUMINAMATH_CALUDE_intersecting_lines_determine_plane_l191_19147


namespace NUMINAMATH_CALUDE_student_marks_l191_19181

theorem student_marks (total_marks : ℕ) (passing_percentage : ℚ) (failed_by : ℕ) (marks_obtained : ℕ) : 
  total_marks = 800 →
  passing_percentage = 33 / 100 →
  failed_by = 89 →
  marks_obtained = total_marks * passing_percentage - failed_by →
  marks_obtained = 175 := by
sorry

#eval (800 : ℕ) * (33 : ℚ) / 100 - 89  -- Expected output: 175

end NUMINAMATH_CALUDE_student_marks_l191_19181


namespace NUMINAMATH_CALUDE_probability_two_defective_shipment_l191_19189

/-- The probability of selecting two defective smartphones from a shipment -/
def probability_two_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / (total : ℚ) * ((defective - 1) : ℚ) / ((total - 1) : ℚ)

/-- Theorem stating the probability of selecting two defective smartphones -/
theorem probability_two_defective_shipment :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000 ∧ 
  |probability_two_defective 240 84 - 1216/10000| < ε :=
sorry

end NUMINAMATH_CALUDE_probability_two_defective_shipment_l191_19189


namespace NUMINAMATH_CALUDE_zoe_average_speed_l191_19106

/-- Represents the hiking scenario with Chantal and Zoe -/
structure HikingScenario where
  d : ℝ  -- Represents one-third of the total distance
  chantal_speed1 : ℝ  -- Chantal's speed for the first third
  chantal_speed2 : ℝ  -- Chantal's speed for the rocky part
  chantal_speed3 : ℝ  -- Chantal's speed for descent on rocky part

/-- The theorem stating Zoe's average speed -/
theorem zoe_average_speed (h : HikingScenario) 
  (h_chantal_speed1 : h.chantal_speed1 = 5)
  (h_chantal_speed2 : h.chantal_speed2 = 3)
  (h_chantal_speed3 : h.chantal_speed3 = 4) :
  let total_time := h.d / h.chantal_speed1 + h.d / h.chantal_speed2 + h.d / h.chantal_speed2 + h.d / h.chantal_speed3
  (h.d / total_time) = 60 / 47 := by
  sorry

#check zoe_average_speed

end NUMINAMATH_CALUDE_zoe_average_speed_l191_19106


namespace NUMINAMATH_CALUDE_cubes_in_figure_100_l191_19157

/-- Represents the number of cubes in a figure at position n -/
def num_cubes (n : ℕ) : ℕ := 2 * n^3 + n^2 + 3 * n + 1

/-- The sequence of cubes follows the given pattern for the first four figures -/
axiom pattern_holds : num_cubes 0 = 1 ∧ num_cubes 1 = 7 ∧ num_cubes 2 = 25 ∧ num_cubes 3 = 63

/-- The number of cubes in figure 100 is 2010301 -/
theorem cubes_in_figure_100 : num_cubes 100 = 2010301 := by
  sorry

end NUMINAMATH_CALUDE_cubes_in_figure_100_l191_19157


namespace NUMINAMATH_CALUDE_f_properties_l191_19105

def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

theorem f_properties :
  (∀ x, f 0 x = f 0 (-x)) ∧
  (∀ a, a > 1/2 → ∀ x, f a x ≥ a + 3/4) ∧
  (∀ a, a ≤ -1/2 → ∀ x, f a x ≥ -a + 3/4) ∧
  (∀ a, -1/2 < a ∧ a ≤ 1/2 → ∀ x, f a x ≥ a^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l191_19105


namespace NUMINAMATH_CALUDE_sequence_inequality_l191_19159

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) = 2 * b n

theorem sequence_inequality (a b : ℕ → ℝ) 
  (ha : arithmetic_sequence a) 
  (hb : geometric_sequence b)
  (h1 : a 1 + b 1 > 0)
  (h2 : a 2 + b 2 < 0) :
  let m := a 4 + b 3
  m < 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l191_19159


namespace NUMINAMATH_CALUDE_polygon_with_40_degree_exterior_angles_has_9_sides_l191_19128

/-- The number of sides in a polygon where each exterior angle measures 40 degrees -/
def polygon_sides : ℕ :=
  (360 : ℕ) / 40

/-- Theorem: A polygon with exterior angles of 40° has 9 sides -/
theorem polygon_with_40_degree_exterior_angles_has_9_sides :
  polygon_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_40_degree_exterior_angles_has_9_sides_l191_19128


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l191_19165

-- Define an isosceles triangle with side lengths 3 and 5
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 3 ∧ b = 5 ∧ (a = c ∨ b = c)) ∨ (a = 5 ∧ b = 3 ∧ (a = c ∨ b = c))

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → (Perimeter a b c = 11 ∨ Perimeter a b c = 13) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l191_19165


namespace NUMINAMATH_CALUDE_shortest_side_length_l191_19184

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Radius of inscribed circle
  r : ℝ
  -- Segments of side 'a' divided by tangent point
  a1 : ℝ
  a2 : ℝ
  -- Conditions
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < r ∧ 0 < a1 ∧ 0 < a2
  tangent_point : a = a1 + a2
  radius : r = 5
  side_sum : b + c = 36
  segments : a1 = 7 ∧ a2 = 9

/-- The length of the shortest side in the triangle is 14 units -/
theorem shortest_side_length (t : TriangleWithInscribedCircle) : 
  min t.a (min t.b t.c) = 14 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l191_19184


namespace NUMINAMATH_CALUDE_average_distance_is_600_l191_19123

/-- The length of one lap around the block in meters -/
def block_length : ℕ := 200

/-- The number of times Johnny runs around the block -/
def johnny_laps : ℕ := 4

/-- The number of times Mickey runs around the block -/
def mickey_laps : ℕ := johnny_laps / 2

/-- The total distance run by Johnny in meters -/
def johnny_distance : ℕ := johnny_laps * block_length

/-- The total distance run by Mickey in meters -/
def mickey_distance : ℕ := mickey_laps * block_length

/-- The average distance run by Johnny and Mickey in meters -/
def average_distance : ℕ := (johnny_distance + mickey_distance) / 2

theorem average_distance_is_600 : average_distance = 600 := by
  sorry

end NUMINAMATH_CALUDE_average_distance_is_600_l191_19123


namespace NUMINAMATH_CALUDE_mark_collection_l191_19178

/-- The amount Mark collects for the homeless -/
theorem mark_collection (households_per_day : ℕ) (days : ℕ) (giving_ratio : ℚ) (donation_amount : ℕ) : 
  households_per_day = 20 →
  days = 5 →
  giving_ratio = 1/2 →
  donation_amount = 40 →
  (households_per_day * days : ℚ) * giving_ratio * donation_amount = 2000 := by
  sorry

#check mark_collection

end NUMINAMATH_CALUDE_mark_collection_l191_19178


namespace NUMINAMATH_CALUDE_villages_with_more_knights_count_l191_19117

/-- The number of villages on the island -/
def total_villages : ℕ := 1000

/-- The number of inhabitants in each village -/
def inhabitants_per_village : ℕ := 99

/-- The total number of knights on the island -/
def total_knights : ℕ := 54054

/-- The number of people in each village who answered there are more knights -/
def more_knights_answers : ℕ := 66

/-- The number of people in each village who answered there are more liars -/
def more_liars_answers : ℕ := 33

/-- The number of villages with more knights than liars -/
def villages_with_more_knights : ℕ := 638

theorem villages_with_more_knights_count :
  villages_with_more_knights = 
    (total_knights - more_liars_answers * total_villages) / 
    (more_knights_answers - more_liars_answers) :=
by sorry

end NUMINAMATH_CALUDE_villages_with_more_knights_count_l191_19117


namespace NUMINAMATH_CALUDE_trig_expression_range_l191_19118

theorem trig_expression_range (C : ℝ) (h : 0 < C ∧ C < π) :
  ∃ (lower upper : ℝ), lower = -1 ∧ upper = Real.sqrt 2 ∧
  -1 < (2 * Real.cos (2 * C) / Real.tan C) + 1 ∧
  (2 * Real.cos (2 * C) / Real.tan C) + 1 ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_range_l191_19118


namespace NUMINAMATH_CALUDE_unique_valid_sequence_l191_19183

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ i j k, a i + a j ≠ a k) ∧
  (∀ m, ∃ k > m, a k = 2 * k - 1)

theorem unique_valid_sequence :
  ∀ a : ℕ → ℕ, is_valid_sequence a ↔ (∀ n, a n = 2 * n - 1) :=
sorry

end NUMINAMATH_CALUDE_unique_valid_sequence_l191_19183


namespace NUMINAMATH_CALUDE_product_without_x_terms_l191_19133

theorem product_without_x_terms (m n : ℝ) : 
  (∀ x : ℝ, (x + 2*m) * (x^2 - x + 1/2*n) = x^3 + 2*m*n) → 
  m^2023 * n^2022 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_product_without_x_terms_l191_19133


namespace NUMINAMATH_CALUDE_indeterminate_b_value_l191_19135

theorem indeterminate_b_value (a b c d : ℝ) : 
  a > b ∧ b > c ∧ c > d → 
  (a + b + c + d) / 4 = 12.345 → 
  ¬(∀ x : ℝ, x = b → (x > 12.345 ∨ x < 12.345 ∨ x = 12.345)) :=
by sorry

end NUMINAMATH_CALUDE_indeterminate_b_value_l191_19135


namespace NUMINAMATH_CALUDE_c_highest_prob_exactly_two_passing_l191_19186

-- Define the probabilities of passing each exam for A, B, and C
def probATheory : ℚ := 4/5
def probAPractical : ℚ := 1/2
def probBTheory : ℚ := 3/4
def probBPractical : ℚ := 2/3
def probCTheory : ℚ := 2/3
def probCPractical : ℚ := 5/6

-- Define the probabilities of obtaining the "certificate of passing" for A, B, and C
def probAPassing : ℚ := probATheory * probAPractical
def probBPassing : ℚ := probBTheory * probBPractical
def probCPassing : ℚ := probCTheory * probCPractical

-- Theorem 1: C has the highest probability of obtaining the "certificate of passing"
theorem c_highest_prob : 
  probCPassing > probAPassing ∧ probCPassing > probBPassing :=
sorry

-- Theorem 2: The probability that exactly two out of A, B, and C obtain the "certificate of passing" is 11/30
theorem exactly_two_passing :
  probAPassing * probBPassing * (1 - probCPassing) +
  probAPassing * (1 - probBPassing) * probCPassing +
  (1 - probAPassing) * probBPassing * probCPassing = 11/30 :=
sorry

end NUMINAMATH_CALUDE_c_highest_prob_exactly_two_passing_l191_19186


namespace NUMINAMATH_CALUDE_trigonometric_identities_l191_19153

theorem trigonometric_identities (α : ℝ) 
  (h : (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7) :
  (Real.tan (π / 2 - α) = 1 / 2) ∧
  (3 * Real.cos α * Real.sin (α + π) + 2 * (Real.cos (α + π / 2))^2 = 2 / 5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l191_19153


namespace NUMINAMATH_CALUDE_angle_trig_sum_l191_19142

theorem angle_trig_sum (a : ℝ) (ha : a ≠ 0) :
  let α := Real.arctan (3*a / (-4*a))
  if a > 0 then
    Real.sin α + Real.cos α - Real.tan α = 11/20
  else
    Real.sin α + Real.cos α - Real.tan α = 19/20 := by
  sorry

end NUMINAMATH_CALUDE_angle_trig_sum_l191_19142


namespace NUMINAMATH_CALUDE_well_volume_approximation_l191_19194

/-- The volume of a cylinder with diameter 6 meters and height 24 meters is approximately 678.58464 cubic meters. -/
theorem well_volume_approximation :
  let diameter : ℝ := 6
  let height : ℝ := 24
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * height
  ∃ ε > 0, |volume - 678.58464| < ε :=
by sorry

end NUMINAMATH_CALUDE_well_volume_approximation_l191_19194


namespace NUMINAMATH_CALUDE_odd_decreasing_function_range_l191_19136

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

-- State the theorem
theorem odd_decreasing_function_range (a : ℝ) 
  (h_odd : is_odd f) 
  (h_decreasing : is_decreasing f) 
  (h_condition : f (2 - a) + f (4 - a) < 0) : 
  a < 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_decreasing_function_range_l191_19136


namespace NUMINAMATH_CALUDE_toms_out_of_pocket_cost_l191_19185

theorem toms_out_of_pocket_cost 
  (visit_cost : ℝ) 
  (cast_cost : ℝ) 
  (insurance_coverage_percentage : ℝ) 
  (h1 : visit_cost = 300)
  (h2 : cast_cost = 200)
  (h3 : insurance_coverage_percentage = 60) :
  let total_cost := visit_cost + cast_cost
  let insurance_coverage := (insurance_coverage_percentage / 100) * total_cost
  let out_of_pocket_cost := total_cost - insurance_coverage
  out_of_pocket_cost = 200 := by
sorry

end NUMINAMATH_CALUDE_toms_out_of_pocket_cost_l191_19185


namespace NUMINAMATH_CALUDE_existence_of_strictly_decreasing_function_with_inequality_l191_19109

/-- A strictly decreasing function from (0, +∞) to (0, +∞) -/
def StrictlyDecreasingPositiveFunction :=
  {g : ℝ → ℝ | ∀ x y, 0 < x → 0 < y → x < y → g y < g x}

theorem existence_of_strictly_decreasing_function_with_inequality
  (k : ℝ) (h_k : 0 < k) :
  (∃ g : ℝ → ℝ, g ∈ StrictlyDecreasingPositiveFunction ∧
    ∀ x, 0 < x → 0 < g x ∧ g x ≥ k * g (x + g x)) ↔ k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_strictly_decreasing_function_with_inequality_l191_19109


namespace NUMINAMATH_CALUDE_matrix_fourth_power_l191_19130

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_fourth_power :
  A ^ 4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_fourth_power_l191_19130


namespace NUMINAMATH_CALUDE_complex_sum_example_l191_19172

theorem complex_sum_example (z₁ z₂ : ℂ) : 
  z₁ = 2 + 3*I ∧ z₂ = -4 - 5*I → z₁ + z₂ = -2 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_example_l191_19172


namespace NUMINAMATH_CALUDE_more_larger_boxes_l191_19197

/-- Represents the number of glasses in a small box -/
def small_box : ℕ := 12

/-- Represents the number of glasses in a large box -/
def large_box : ℕ := 16

/-- Represents the average number of glasses per box -/
def average_glasses : ℕ := 15

/-- Represents the total number of glasses -/
def total_glasses : ℕ := 480

theorem more_larger_boxes (s l : ℕ) : 
  s * small_box + l * large_box = total_glasses →
  (s + l : ℚ) = (total_glasses : ℚ) / average_glasses →
  l > s →
  l - s = 16 := by
  sorry

end NUMINAMATH_CALUDE_more_larger_boxes_l191_19197


namespace NUMINAMATH_CALUDE_mothers_age_l191_19134

/-- Given a person and their mother, with the following conditions:
  1. The person's present age is two-fifths of the age of his mother.
  2. After 10 years, the person will be one-half of the age of his mother.
  This theorem proves that the mother's present age is 50 years. -/
theorem mothers_age (person_age mother_age : ℕ) 
  (h1 : person_age = (2 * mother_age) / 5)
  (h2 : person_age + 10 = (mother_age + 10) / 2) : 
  mother_age = 50 := by
  sorry

end NUMINAMATH_CALUDE_mothers_age_l191_19134


namespace NUMINAMATH_CALUDE_tan_beta_value_l191_19177

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := by sorry

end NUMINAMATH_CALUDE_tan_beta_value_l191_19177


namespace NUMINAMATH_CALUDE_a_range_l191_19161

theorem a_range (a b c : ℝ) 
  (sum_eq : a + b + c = 2) 
  (sum_sq_eq : a^2 + b^2 + c^2 = 4) 
  (order : a > b ∧ b > c) : 
  2/3 < a ∧ a < 2 := by
sorry

end NUMINAMATH_CALUDE_a_range_l191_19161


namespace NUMINAMATH_CALUDE_cuboid_volume_l191_19180

/-- Given a cuboid with three side faces sharing a common vertex having areas 3, 5, and 15,
    prove that its volume is 15. -/
theorem cuboid_volume (a b c : ℝ) 
  (h1 : a * b = 3)
  (h2 : b * c = 5)
  (h3 : a * c = 15) : 
  a * b * c = 15 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_l191_19180


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l191_19146

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x^2 - x - 3 > 0) ↔ (x > 3/2 ∨ x < -1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l191_19146


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_l191_19143

theorem product_of_roots_cubic (a b c : ℝ) : 
  (3 * a^3 - 9 * a^2 + 4 * a - 12 = 0) ∧
  (3 * b^3 - 9 * b^2 + 4 * b - 12 = 0) ∧
  (3 * c^3 - 9 * c^2 + 4 * c - 12 = 0) →
  a * b * c = 4 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_l191_19143


namespace NUMINAMATH_CALUDE_factory_workers_count_l191_19167

/-- Proves the number of workers in a factory given certain salary information --/
theorem factory_workers_count :
  let initial_average : ℚ := 430
  let initial_supervisor_salary : ℚ := 870
  let new_average : ℚ := 390
  let new_supervisor_salary : ℚ := 510
  let total_people : ℕ := 9
  ∃ (workers : ℕ),
    (workers : ℚ) + 1 = (total_people : ℚ) ∧
    (workers + 1) * initial_average = workers * initial_average + initial_supervisor_salary ∧
    total_people * new_average = workers * initial_average + new_supervisor_salary ∧
    workers = 8 := by
  sorry

end NUMINAMATH_CALUDE_factory_workers_count_l191_19167


namespace NUMINAMATH_CALUDE_bank_line_theorem_l191_19199

/-- Represents a bank line with fast and slow customers. -/
structure BankLine where
  total_customers : Nat
  fast_customers : Nat
  slow_customers : Nat
  fast_operation_time : Nat
  slow_operation_time : Nat

/-- Calculates the minimum total wasted person-minutes. -/
def minimum_wasted_time (line : BankLine) : Nat :=
  sorry

/-- Calculates the maximum total wasted person-minutes. -/
def maximum_wasted_time (line : BankLine) : Nat :=
  sorry

/-- Calculates the expected number of wasted person-minutes. -/
def expected_wasted_time (line : BankLine) : Nat :=
  sorry

/-- Theorem stating the results for the specific bank line scenario. -/
theorem bank_line_theorem (line : BankLine) 
    (h1 : line.total_customers = 8)
    (h2 : line.fast_customers = 5)
    (h3 : line.slow_customers = 3)
    (h4 : line.fast_operation_time = 1)
    (h5 : line.slow_operation_time = 5) :
  minimum_wasted_time line = 40 ∧
  maximum_wasted_time line = 100 ∧
  expected_wasted_time line = 70 :=
  sorry

end NUMINAMATH_CALUDE_bank_line_theorem_l191_19199


namespace NUMINAMATH_CALUDE_trig_expression_equals_negative_two_l191_19158

theorem trig_expression_equals_negative_two :
  5 * Real.sin (π / 2) + 2 * Real.cos 0 - 3 * Real.sin (3 * π / 2) + 10 * Real.cos π = -2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_negative_two_l191_19158


namespace NUMINAMATH_CALUDE_range_of_x₀_l191_19188

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point M
def point_M (x₀ : ℝ) : ℝ × ℝ := (x₀, 2 - x₀)

-- Define the angle OMN
def angle_OMN (O M N : ℝ × ℝ) : ℝ := sorry

-- Define the existence of point N on circle O
def exists_N (x₀ : ℝ) : Prop :=
  ∃ N : ℝ × ℝ, circle_O N.1 N.2 ∧ angle_OMN (0, 0) (point_M x₀) N = 30

-- Theorem statement
theorem range_of_x₀ (x₀ : ℝ) :
  exists_N x₀ → 0 ≤ x₀ ∧ x₀ ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x₀_l191_19188


namespace NUMINAMATH_CALUDE_simplify_polynomial_l191_19111

theorem simplify_polynomial (x : ℝ) :
  2 * x * (5 * x^2 - 3 * x + 1) + 4 * (x^2 - 3 * x + 6) =
  10 * x^3 - 2 * x^2 - 10 * x + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l191_19111


namespace NUMINAMATH_CALUDE_ian_lottery_winnings_l191_19171

theorem ian_lottery_winnings :
  ∀ (lottery_winnings : ℕ) (colin_payment helen_payment benedict_payment remaining : ℕ),
  colin_payment = 20 →
  helen_payment = 2 * colin_payment →
  benedict_payment = helen_payment / 2 →
  remaining = 20 →
  lottery_winnings = colin_payment + helen_payment + benedict_payment + remaining →
  lottery_winnings = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ian_lottery_winnings_l191_19171


namespace NUMINAMATH_CALUDE_extreme_values_when_a_is_4_a_range_when_f_geq_4_on_interval_l191_19151

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x

-- Part I
theorem extreme_values_when_a_is_4 :
  let f := f 4
  (∃ x, ∀ y, f y ≤ f x) ∧ (∃ x, ∀ y, f y ≥ f x) ∧
  (∀ x, f x ≤ 1) ∧ (∀ x, f x ≥ -1) :=
sorry

-- Part II
theorem a_range_when_f_geq_4_on_interval :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x ≥ 4) → a ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_when_a_is_4_a_range_when_f_geq_4_on_interval_l191_19151


namespace NUMINAMATH_CALUDE_divisibility_by_66_l191_19132

theorem divisibility_by_66 : ∃ k : ℤ, 43^23 + 23^43 = 66 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_66_l191_19132


namespace NUMINAMATH_CALUDE_increasing_symmetric_function_inequality_l191_19179

theorem increasing_symmetric_function_inequality 
  (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x ≥ 2 → y ≥ 2 → x < y → f x < f y) 
  (h_symmetric : ∀ x, f (2 + x) = f (2 - x)) 
  (h_inequality : f (1 - 2 * x^2) < f (1 + 2*x - x^2)) :
  -2 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_increasing_symmetric_function_inequality_l191_19179


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_four_l191_19155

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of 4 is -4 -/
theorem opposite_of_four : opposite 4 = -4 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_four_l191_19155


namespace NUMINAMATH_CALUDE_tetrakaidecagon_area_approx_l191_19198

/-- A tetrakaidecagon inscribed in a square -/
structure InscribedTetrakaidecagon where
  /-- The side length of the square -/
  square_side : ℝ
  /-- The number of segments each side of the square is divided into -/
  num_segments : ℕ
  /-- The perimeter of the square -/
  square_perimeter : ℝ
  /-- The perimeter of the square is 56 meters -/
  perimeter_constraint : square_perimeter = 56
  /-- Each side of the square is divided into equal segments -/
  side_division : square_side = square_perimeter / 4
  /-- The number of segments is 7 -/
  segment_count : num_segments = 7

/-- The area of the inscribed tetrakaidecagon -/
noncomputable def tetrakaidecagon_area (t : InscribedTetrakaidecagon) : ℝ :=
  t.square_side ^ 2 - 16 * (1 / 2 * (t.square_side / t.num_segments) ^ 2)

/-- Theorem stating the area of the inscribed tetrakaidecagon -/
theorem tetrakaidecagon_area_approx (t : InscribedTetrakaidecagon) :
  abs (tetrakaidecagon_area t - 21.92) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_tetrakaidecagon_area_approx_l191_19198


namespace NUMINAMATH_CALUDE_circle_C_tangent_line_l_line_AB_passes_through_intersection_l191_19164

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Define the line l
def line_l (x : ℝ) : Prop := x = 3

-- Define circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define line AB
def line_AB (x y : ℝ) : Prop := 2*x + y - 4 = 0

-- Theorem 1: Circle C is tangent to line l
theorem circle_C_tangent_line_l : ∃ (x y : ℝ), circle_C x y ∧ line_l x := by sorry

-- Theorem 2: Line AB passes through the intersection of circles C and O
theorem line_AB_passes_through_intersection :
  ∀ (x y : ℝ), (circle_C x y ∧ circle_O x y) → line_AB x y := by sorry

end NUMINAMATH_CALUDE_circle_C_tangent_line_l_line_AB_passes_through_intersection_l191_19164


namespace NUMINAMATH_CALUDE_bridge_length_l191_19112

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_sec : ℝ) :
  train_length = 145 →
  train_speed_kmh = 45 →
  crossing_time_sec = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 230 ∧
    bridge_length = (train_speed_kmh * 1000 / 3600 * crossing_time_sec) - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l191_19112


namespace NUMINAMATH_CALUDE_marbles_exceed_200_l191_19103

theorem marbles_exceed_200 : ∃ k : ℕ, (∀ j : ℕ, j < k → 5 * 2^j ≤ 200) ∧ 5 * 2^k > 200 ∧ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_marbles_exceed_200_l191_19103


namespace NUMINAMATH_CALUDE_class_grade_point_average_l191_19125

/-- Calculate the grade point average of a class given the distribution of grades --/
theorem class_grade_point_average 
  (total_students : ℕ) 
  (gpa_60_percent : ℚ) 
  (gpa_65_percent : ℚ) 
  (gpa_70_percent : ℚ) 
  (gpa_80_percent : ℚ) 
  (h1 : total_students = 120)
  (h2 : gpa_60_percent = 25 / 100)
  (h3 : gpa_65_percent = 35 / 100)
  (h4 : gpa_70_percent = 15 / 100)
  (h5 : gpa_80_percent = 1 - (gpa_60_percent + gpa_65_percent + gpa_70_percent))
  (h6 : gpa_60_percent + gpa_65_percent + gpa_70_percent + gpa_80_percent = 1) :
  let weighted_average := 
    (gpa_60_percent * 60 + gpa_65_percent * 65 + gpa_70_percent * 70 + gpa_80_percent * 80)
  weighted_average = 68.25 := by
  sorry

end NUMINAMATH_CALUDE_class_grade_point_average_l191_19125


namespace NUMINAMATH_CALUDE_impossible_to_reach_in_time_l191_19139

/-- Represents the problem of traveling to the train station -/
structure TravelProblem where
  totalTime : ℝ
  totalDistance : ℝ
  firstKilometerSpeed : ℝ

/-- Defines the given travel problem -/
def givenProblem : TravelProblem where
  totalTime := 2  -- 2 minutes
  totalDistance := 2  -- 2 km
  firstKilometerSpeed := 30  -- 30 km/h

/-- Theorem stating that it's impossible to reach the destination in time -/
theorem impossible_to_reach_in_time (p : TravelProblem) 
  (h1 : p.totalTime = 2)
  (h2 : p.totalDistance = 2)
  (h3 : p.firstKilometerSpeed = 30) : 
  ¬ ∃ (secondKilometerSpeed : ℝ), 
    (1 / (p.firstKilometerSpeed / 60)) + (1 / secondKilometerSpeed) ≤ p.totalTime :=
by sorry

#check impossible_to_reach_in_time givenProblem rfl rfl rfl

end NUMINAMATH_CALUDE_impossible_to_reach_in_time_l191_19139


namespace NUMINAMATH_CALUDE_calculator_game_sum_l191_19145

def iterate_calculator (n : ℕ) (initial : ℤ) (f : ℤ → ℤ) : ℤ :=
  match n with
  | 0 => initial
  | m + 1 => f (iterate_calculator m initial f)

theorem calculator_game_sum (n : ℕ) : 
  iterate_calculator n 1 (λ x => x^3) + 
  iterate_calculator n 0 (λ x => x^2) + 
  iterate_calculator n (-1) (λ x => -x) = 0 :=
by sorry

end NUMINAMATH_CALUDE_calculator_game_sum_l191_19145


namespace NUMINAMATH_CALUDE_millionaire_hat_sale_l191_19192

/-- Proves that the fraction of hats sold is 2/3 given the conditions of the problem -/
theorem millionaire_hat_sale (H : ℝ) (h1 : H > 0) : 
  let brown_hats := (1/4 : ℝ) * H
  let sold_brown_hats := (4/5 : ℝ) * brown_hats
  let remaining_hats := H - sold_brown_hats - ((3/4 : ℝ) * H - (1/5 : ℝ) * brown_hats)
  let remaining_brown_hats := brown_hats - sold_brown_hats
  (remaining_brown_hats / remaining_hats) = (15/100 : ℝ) →
  (sold_brown_hats + ((3/4 : ℝ) * H - (1/5 : ℝ) * brown_hats)) / H = (2/3 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_millionaire_hat_sale_l191_19192


namespace NUMINAMATH_CALUDE_origin_on_circle_l191_19150

theorem origin_on_circle (center_x center_y radius : ℝ) 
  (h1 : center_x = 5)
  (h2 : center_y = 12)
  (h3 : radius = 13) :
  (center_x^2 + center_y^2).sqrt = radius :=
sorry

end NUMINAMATH_CALUDE_origin_on_circle_l191_19150


namespace NUMINAMATH_CALUDE_cosine_sine_inequality_l191_19119

theorem cosine_sine_inequality (a b : ℝ) 
  (h : ∀ x : ℝ, Real.cos (a * Real.sin x) > Real.sin (b * Real.cos x)) : 
  a^2 + b^2 < (Real.pi^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_inequality_l191_19119


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l191_19144

/-- Given that Virginia, Adrienne, and Dennis have taught history for a combined total of 93 years,
    Virginia has taught for 9 more years than Adrienne, and Virginia has taught for 9 fewer years than Dennis,
    prove that Dennis has taught for 40 years. -/
theorem dennis_teaching_years (v a d : ℕ) 
  (total : v + a + d = 93)
  (v_more_than_a : v = a + 9)
  (v_less_than_d : v = d - 9) :
  d = 40 := by
  sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l191_19144


namespace NUMINAMATH_CALUDE_amoeba_growth_after_week_l191_19108

def amoeba_population (initial_population : ℕ) (days : ℕ) : ℕ :=
  if days = 0 then
    initial_population
  else if days % 2 = 1 then
    2 * amoeba_population initial_population (days - 1)
  else
    3 * 2 * amoeba_population initial_population (days - 1)

theorem amoeba_growth_after_week :
  amoeba_population 4 7 = 13824 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_growth_after_week_l191_19108


namespace NUMINAMATH_CALUDE_highway_distance_theorem_l191_19174

/-- The distance between two points A and B on a highway -/
def distance_AB : ℝ := 198

/-- The speed of vehicles traveling from A to B -/
def speed_AB : ℝ := 50

/-- The speed of vehicles traveling from B to A -/
def speed_BA : ℝ := 60

/-- The distance from point B where car X breaks down -/
def breakdown_distance : ℝ := 30

/-- The delay in the second meeting due to the breakdown -/
def delay_time : ℝ := 1.2

theorem highway_distance_theorem :
  distance_AB = 198 :=
sorry

end NUMINAMATH_CALUDE_highway_distance_theorem_l191_19174


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l191_19193

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I * z) = 1) : 
  Complex.abs (2 * z - 3) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l191_19193


namespace NUMINAMATH_CALUDE_half_radius_circle_y_l191_19113

-- Define the circles
def circle_x : Real → Prop := λ r => r > 0
def circle_y : Real → Prop := λ r => r > 0

-- Define the theorem
theorem half_radius_circle_y 
  (h_area : ∀ (rx ry : Real), circle_x rx → circle_y ry → π * rx^2 = π * ry^2)
  (h_circum : ∀ (rx : Real), circle_x rx → 2 * π * rx = 10 * π) :
  ∃ (ry : Real), circle_y ry ∧ ry / 2 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_half_radius_circle_y_l191_19113


namespace NUMINAMATH_CALUDE_quadrilateral_ABCD_area_l191_19115

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (A B C D : Point) : ℝ := sorry

theorem quadrilateral_ABCD_area :
  let A : Point := ⟨0, 1⟩
  let B : Point := ⟨1, 3⟩
  let C : Point := ⟨5, 2⟩
  let D : Point := ⟨4, 0⟩
  quadrilateralArea A B C D = 9 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_ABCD_area_l191_19115


namespace NUMINAMATH_CALUDE_exist_integers_product_minus_third_l191_19170

theorem exist_integers_product_minus_third : ∃ (a b c : ℤ), 
  (a * b - c = 2018) ∧ (b * c - a = 2018) ∧ (c * a - b = 2018) := by
sorry

end NUMINAMATH_CALUDE_exist_integers_product_minus_third_l191_19170


namespace NUMINAMATH_CALUDE_square_difference_l191_19138

theorem square_difference (a b : ℝ) :
  let A : ℝ := (5*a + 3*b)^2 - (5*a - 3*b)^2
  A = 60*a*b := by sorry

end NUMINAMATH_CALUDE_square_difference_l191_19138


namespace NUMINAMATH_CALUDE_restaurant_group_size_l191_19104

theorem restaurant_group_size :
  let adult_meal_cost : ℕ := 3
  let kids_eat_free : Bool := true
  let num_kids : ℕ := 7
  let total_cost : ℕ := 15
  let num_adults : ℕ := total_cost / adult_meal_cost
  let total_people : ℕ := num_adults + num_kids
  total_people = 12 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_size_l191_19104


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l191_19149

theorem largest_x_sqrt_3x_eq_5x :
  ∃ (x_max : ℚ), x_max = 3/25 ∧
  (∀ x : ℚ, x ≥ 0 → Real.sqrt (3 * x) = 5 * x → x ≤ x_max) ∧
  Real.sqrt (3 * x_max) = 5 * x_max := by
  sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l191_19149


namespace NUMINAMATH_CALUDE_amoeba_count_after_ten_days_l191_19169

/-- The number of amoebas in the puddle after n days -/
def amoeba_count (n : ℕ) : ℕ :=
  3^n

/-- The number of days the amoeba growth process continues -/
def days : ℕ := 10

/-- Theorem: The number of amoebas after 10 days is 59049 -/
theorem amoeba_count_after_ten_days :
  amoeba_count days = 59049 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_after_ten_days_l191_19169


namespace NUMINAMATH_CALUDE_gcf_252_96_l191_19195

theorem gcf_252_96 : Nat.gcd 252 96 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_252_96_l191_19195


namespace NUMINAMATH_CALUDE_discount_order_matters_l191_19140

/-- Proves that applying a percentage discount followed by a fixed discount
    results in a lower final price than the reverse order. -/
theorem discount_order_matters (initial_price percent_off fixed_off : ℝ) 
  (h_initial : initial_price = 50)
  (h_percent : percent_off = 0.15)
  (h_fixed : fixed_off = 6) : 
  (1 - percent_off) * initial_price - fixed_off < 
  (1 - percent_off) * (initial_price - fixed_off) := by
  sorry

end NUMINAMATH_CALUDE_discount_order_matters_l191_19140


namespace NUMINAMATH_CALUDE_range_of_m_l191_19116

theorem range_of_m (m : ℝ) : (∃ x : ℝ, |x - 3| + |x - 4| < m) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l191_19116


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l191_19176

/-- Given a geometric sequence of positive numbers where the fourth term is 16
    and the ninth term is 8, the sixth term is equal to 16 * (4^(1/5)) -/
theorem geometric_sequence_sixth_term
  (a : ℝ → ℝ)  -- The sequence
  (r : ℝ)      -- Common ratio
  (h_positive : ∀ n, a n > 0)  -- All terms are positive
  (h_geometric : ∀ n, a (n + 1) = a n * r)  -- It's a geometric sequence
  (h_fourth : a 4 = 16)  -- The fourth term is 16
  (h_ninth : a 9 = 8)  -- The ninth term is 8
  : a 6 = 16 * (4^(1/5)) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l191_19176


namespace NUMINAMATH_CALUDE_library_visitors_l191_19110

/-- Calculates the average number of visitors on non-Sunday days in a library -/
theorem library_visitors (total_days : Nat) (sunday_visitors : Nat) (avg_visitors : Nat) :
  total_days = 30 ∧ 
  sunday_visitors = 510 ∧ 
  avg_visitors = 285 →
  (total_days * avg_visitors - 5 * sunday_visitors) / 25 = 240 := by
sorry

end NUMINAMATH_CALUDE_library_visitors_l191_19110


namespace NUMINAMATH_CALUDE_no_equal_volume_increase_l191_19102

theorem no_equal_volume_increase (x : ℝ) : ¬ (
  let R : ℝ := 10
  let H : ℝ := 5
  let V (r h : ℝ) := Real.pi * r^2 * h
  V (R + x) H - V R H = V R (H + x) - V R H
) := by sorry

end NUMINAMATH_CALUDE_no_equal_volume_increase_l191_19102


namespace NUMINAMATH_CALUDE_extraneous_roots_equation_l191_19114

theorem extraneous_roots_equation :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧
  (∀ x : ℝ, Real.sqrt (x + 15) - 8 / Real.sqrt (x + 15) = 6 →
    (x = r₁ ∨ x = r₂) ∧
    Real.sqrt (r₁ + 15) - 8 / Real.sqrt (r₁ + 15) ≠ 6 ∧
    Real.sqrt (r₂ + 15) - 8 / Real.sqrt (r₂ + 15) ≠ 6) :=
by
  sorry

end NUMINAMATH_CALUDE_extraneous_roots_equation_l191_19114


namespace NUMINAMATH_CALUDE_necessary_implies_sufficient_l191_19162

-- Define what it means for q to be a necessary condition for p
def necessary_condition (p q : Prop) : Prop :=
  p → q

-- Define what it means for p to be a sufficient condition for q
def sufficient_condition (p q : Prop) : Prop :=
  p → q

-- Theorem statement
theorem necessary_implies_sufficient (p q : Prop) 
  (h : necessary_condition p q) : sufficient_condition p q :=
by
  sorry


end NUMINAMATH_CALUDE_necessary_implies_sufficient_l191_19162


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l191_19163

theorem roots_of_quadratic_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁^2 - 4 = 0 ∧ x₂^2 - 4 = 0) ∧ x₁ = 2 ∧ x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l191_19163


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l191_19131

theorem complex_modulus_equation (a : ℝ) : 
  Complex.abs ((5 : ℂ) / (2 + Complex.I) + a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l191_19131


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l191_19160

theorem inequality_not_always_true (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x^2 > y^2) (hz : z ≠ 0) :
  ¬ ∀ (x y z : ℝ), x > 0 → y > 0 → x^2 > y^2 → z ≠ 0 → x * z^3 > y * z^3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l191_19160


namespace NUMINAMATH_CALUDE_find_b_value_l191_19100

theorem find_b_value (b : ℝ) : (5 : ℝ)^2 + b * 5 - 35 = 0 → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l191_19100


namespace NUMINAMATH_CALUDE_sin_sum_alpha_beta_l191_19175

theorem sin_sum_alpha_beta (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1) 
  (h2 : Real.cos α + Real.sin β = 0) : 
  Real.sin (α + β) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_alpha_beta_l191_19175


namespace NUMINAMATH_CALUDE_perpendicular_lines_l191_19101

/-- Two lines y = ax - 2 and y = x + 1 are perpendicular if and only if a = -1 -/
theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, y = a * x - 2 ∧ y = x + 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l191_19101


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l191_19126

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 4*x + 3 = 0) ∧ 
  (∃ x : ℝ, x*(x-2) = 2*(2-x)) ∧
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 ↔ (x = 3 ∨ x = 1)) ∧
  (∀ x : ℝ, x*(x-2) = 2*(2-x) ↔ (x = 2 ∨ x = -2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l191_19126


namespace NUMINAMATH_CALUDE_two_digit_congruent_to_three_mod_four_count_l191_19154

theorem two_digit_congruent_to_three_mod_four_count : 
  (Finset.filter (fun n => n ≥ 10 ∧ n ≤ 99 ∧ n % 4 = 3) (Finset.range 100)).card = 23 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_congruent_to_three_mod_four_count_l191_19154


namespace NUMINAMATH_CALUDE_ipod_problem_l191_19191

def problem (emmy_initial : ℕ) (emmy_lost : ℕ) (rosa_given_away : ℕ) : Prop :=
  let emmy_current := emmy_initial - emmy_lost
  let rosa_current := emmy_current / 3
  let rosa_initial := rosa_current + rosa_given_away
  emmy_current + rosa_current = 21

theorem ipod_problem : problem 25 9 4 := by
  sorry

end NUMINAMATH_CALUDE_ipod_problem_l191_19191


namespace NUMINAMATH_CALUDE_cards_after_exchange_and_giveaway_l191_19168

/-- Represents the number of cards in a box for each sport --/
structure CardCounts where
  basketball : ℕ
  baseball : ℕ
  football : ℕ
  hockey : ℕ
  soccer : ℕ

/-- Represents the number of boxes for each sport --/
structure BoxCounts where
  basketball : ℕ
  baseball : ℕ
  football : ℕ
  hockey : ℕ
  soccer : ℕ

/-- Calculate the total number of cards --/
def totalCards (cards : CardCounts) (boxes : BoxCounts) : ℕ :=
  cards.basketball * boxes.basketball +
  cards.baseball * boxes.baseball +
  cards.football * boxes.football +
  cards.hockey * boxes.hockey +
  cards.soccer * boxes.soccer

/-- The number of cards exchanged between Ben and Alex --/
def exchangedCards (cards : CardCounts) (boxes : BoxCounts) : ℕ :=
  (cards.basketball / 2) * boxes.basketball +
  (cards.baseball / 2) * boxes.baseball

theorem cards_after_exchange_and_giveaway 
  (ben_cards : CardCounts)
  (ben_boxes : BoxCounts)
  (alex_cards : CardCounts)
  (alex_boxes : BoxCounts)
  (h1 : ben_cards.basketball = 20)
  (h2 : ben_cards.baseball = 15)
  (h3 : ben_cards.football = 12)
  (h4 : ben_boxes.basketball = 8)
  (h5 : ben_boxes.baseball = 10)
  (h6 : ben_boxes.football = 12)
  (h7 : alex_cards.hockey = 15)
  (h8 : alex_cards.soccer = 18)
  (h9 : alex_boxes.hockey = 6)
  (h10 : alex_boxes.soccer = 9)
  (cards_given_away : ℕ)
  (h11 : cards_given_away = 175) :
  totalCards ben_cards ben_boxes + totalCards alex_cards alex_boxes - cards_given_away = 531 := by
  sorry


end NUMINAMATH_CALUDE_cards_after_exchange_and_giveaway_l191_19168


namespace NUMINAMATH_CALUDE_no_intersection_and_in_circle_l191_19124

theorem no_intersection_and_in_circle : ¬∃ (a b : ℝ),
  (∃ (n m : ℤ), n = m ∧ n * a + b = 3 * m^2 + 15) ∧
  (a^2 + b^2 ≤ 144) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_and_in_circle_l191_19124


namespace NUMINAMATH_CALUDE_min_value_theorem_l191_19129

theorem min_value_theorem (m n p x y z : ℝ) 
  (hm : m > 0) (hn : n > 0) (hp : p > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_mnp : m * n * p = 8) (h_xyz : x * y * z = 8) :
  let f := x^2 + y^2 + z^2 + m*x*y + n*x*z + p*y*z
  (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x' * y' * z' = 8 → f ≤ x'^2 + y'^2 + z'^2 + m*x'*y' + n*x'*z' + p*y'*z') ∧
  (m = 2 ∧ n = 2 ∧ p = 2 → ∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x' * y' * z' = 8 → 
    36 ≤ x'^2 + y'^2 + z'^2 + m*x'*y' + n*x'*z' + p*y'*z') ∧
  (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x' * y' * z' = 8 → 
    6 * (2^(1/3 : ℝ)) * (m^(2/3 : ℝ) + n^(2/3 : ℝ) + p^(2/3 : ℝ)) ≤ x'^2 + y'^2 + z'^2 + m*x'*y' + n*x'*z' + p*y'*z') :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l191_19129


namespace NUMINAMATH_CALUDE_sector_radius_l191_19148

/-- Given a sector with a central angle of 150° and an arc length of 5π/2 cm, its radius is 3 cm. -/
theorem sector_radius (θ : ℝ) (arc_length : ℝ) (radius : ℝ) : 
  θ = 150 → 
  arc_length = (5/2) * Real.pi → 
  arc_length = (θ / 360) * 2 * Real.pi * radius → 
  radius = 3 := by
sorry

end NUMINAMATH_CALUDE_sector_radius_l191_19148


namespace NUMINAMATH_CALUDE_problem_solution_l191_19190

theorem problem_solution (x : ℝ) :
  x - Real.sqrt (x^2 + 1) + 1 / (x + Real.sqrt (x^2 + 1)) = 28 →
  x^2 - Real.sqrt (x^4 + 1) + 1 / (x^2 - Real.sqrt (x^4 + 1)) = -2 * Real.sqrt 38026 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l191_19190


namespace NUMINAMATH_CALUDE_power_of_negative_product_l191_19187

theorem power_of_negative_product (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l191_19187


namespace NUMINAMATH_CALUDE_medal_distribution_proof_l191_19107

def total_sprinters : Nat := 10
def american_sprinters : Nat := 4
def medals : Nat := 3

def ways_to_distribute_medals : Nat :=
  let non_american_sprinters := total_sprinters - american_sprinters
  let no_american_medalists := non_american_sprinters * (non_american_sprinters - 1) * (non_american_sprinters - 2)
  let one_american_medalist := american_sprinters * medals * (non_american_sprinters * (non_american_sprinters - 1))
  no_american_medalists + one_american_medalist

theorem medal_distribution_proof : 
  ways_to_distribute_medals = 480 := by sorry

end NUMINAMATH_CALUDE_medal_distribution_proof_l191_19107


namespace NUMINAMATH_CALUDE_factor_implies_b_value_l191_19137

theorem factor_implies_b_value (b : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, 9*x^2 + b*x + 44 = (3*x + 4) * k) → b = 45 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_b_value_l191_19137
