import Mathlib

namespace NUMINAMATH_CALUDE_central_angle_unchanged_l778_77869

theorem central_angle_unchanged (r₁ r₂ arc_length₁ arc_length₂ angle₁ angle₂ : ℝ) :
  r₁ > 0 →
  r₂ = 2 * r₁ →
  arc_length₂ = 2 * arc_length₁ →
  angle₁ = arc_length₁ / r₁ →
  angle₂ = arc_length₂ / r₂ →
  angle₂ = angle₁ :=
by sorry

end NUMINAMATH_CALUDE_central_angle_unchanged_l778_77869


namespace NUMINAMATH_CALUDE_largest_c_for_g_range_two_l778_77835

/-- The function g(x) defined as x^2 - 5x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 - 5*x + c

/-- Theorem stating that the largest value of c such that 2 is in the range of g(x) is 33/4 -/
theorem largest_c_for_g_range_two :
  ∃ (c_max : ℝ), c_max = 33/4 ∧
  (∀ c : ℝ, (∃ x : ℝ, g c x = 2) → c ≤ c_max) ∧
  (∃ x : ℝ, g c_max x = 2) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_g_range_two_l778_77835


namespace NUMINAMATH_CALUDE_baseball_soccer_difference_l778_77870

def total_balls : ℕ := 145
def soccer_balls : ℕ := 20
def volleyball_balls : ℕ := 30

def basketball_balls : ℕ := soccer_balls + 5
def tennis_balls : ℕ := 2 * soccer_balls

def baseball_balls : ℕ := total_balls - (soccer_balls + basketball_balls + tennis_balls + volleyball_balls)

theorem baseball_soccer_difference :
  baseball_balls - soccer_balls = 10 :=
by sorry

end NUMINAMATH_CALUDE_baseball_soccer_difference_l778_77870


namespace NUMINAMATH_CALUDE_four_point_equal_inradii_congruent_triangles_l778_77834

-- Define a type for points in a plane
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a type for triangles
structure Triangle :=
  (a b c : Point)

-- Define a function to check if three points are collinear
def collinear (p q r : Point) : Prop := sorry

-- Define a function to calculate the inradius of a triangle
def inradius (t : Triangle) : ℝ := sorry

-- Define a function to check if two triangles are congruent
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Main theorem
theorem four_point_equal_inradii_congruent_triangles 
  (A B C D : Point) : 
  (¬collinear A B C ∧ ¬collinear A B D ∧ ¬collinear A C D ∧ ¬collinear B C D) →
  (inradius ⟨A, B, C⟩ = inradius ⟨A, B, D⟩) →
  (inradius ⟨A, B, C⟩ = inradius ⟨A, C, D⟩) →
  (inradius ⟨A, B, C⟩ = inradius ⟨B, C, D⟩) →
  (congruent ⟨A, B, C⟩ ⟨A, B, D⟩) ∧ 
  (congruent ⟨A, B, C⟩ ⟨A, C, D⟩) ∧ 
  (congruent ⟨A, B, C⟩ ⟨B, C, D⟩) :=
by sorry

end NUMINAMATH_CALUDE_four_point_equal_inradii_congruent_triangles_l778_77834


namespace NUMINAMATH_CALUDE_vector_sum_l778_77876

/-- Given two plane vectors a and b, prove their sum is (0, 1) -/
theorem vector_sum (a b : ℝ × ℝ) (h1 : a = (1, -1)) (h2 : b = (-1, 2)) :
  a + b = (0, 1) := by sorry

end NUMINAMATH_CALUDE_vector_sum_l778_77876


namespace NUMINAMATH_CALUDE_only_two_digit_divisor_with_remainder_four_l778_77851

theorem only_two_digit_divisor_with_remainder_four (d : ℕ) : 
  d > 0 ∧ d ≥ 10 ∧ d ≤ 99 ∧ 143 % d = 4 → d = 139 :=
by sorry

end NUMINAMATH_CALUDE_only_two_digit_divisor_with_remainder_four_l778_77851


namespace NUMINAMATH_CALUDE_factorization_of_x2y_plus_xy2_l778_77845

theorem factorization_of_x2y_plus_xy2 (x y : ℝ) : x^2*y + x*y^2 = x*y*(x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x2y_plus_xy2_l778_77845


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l778_77881

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h1 : a 3 + a 8 = 10) : 3 * a 5 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l778_77881


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l778_77836

/-- A parabola with its focus on the line x-2y+2=0 has a standard equation of either x^2 = 4y or y^2 = -8x -/
theorem parabola_standard_equation (F : ℝ × ℝ) :
  (F.1 - 2 * F.2 + 2 = 0) →
  (∃ (x y : ℝ → ℝ), (∀ t, x t ^ 2 = 4 * y t) ∨ (∀ t, y t ^ 2 = -8 * x t)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l778_77836


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l778_77825

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the cost of insulating a rectangular tank -/
def insulationCost (l w h costPerSqFt : ℝ) : ℝ :=
  surfaceArea l w h * costPerSqFt

theorem tank_insulation_cost :
  let l : ℝ := 6
  let w : ℝ := 3
  let h : ℝ := 2
  let costPerSqFt : ℝ := 20
  insulationCost l w h costPerSqFt = 1440 := by
  sorry

#eval insulationCost 6 3 2 20

end NUMINAMATH_CALUDE_tank_insulation_cost_l778_77825


namespace NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l778_77824

-- Define the circles and lines
def circle1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 10
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 11 = 0
def line1 (x y : ℝ) : Prop := y = 2*x - 3
def line2 (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0
def circle3 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle4 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Theorem for the first circle
theorem circle1_properties :
  (∀ x y : ℝ, circle1 x y → ((x = 5 ∧ y = 2) ∨ (x = 3 ∧ y = 2))) ∧
  (∃ x y : ℝ, circle1 x y ∧ line1 x y) :=
sorry

-- Theorem for the second circle
theorem circle2_properties :
  (∀ x y : ℝ, (circle3 x y ∧ circle4 x y) → circle2 x y) ∧
  (∃ x y : ℝ, circle2 x y ∧ line2 x y) :=
sorry

end NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l778_77824


namespace NUMINAMATH_CALUDE_census_contradiction_l778_77887

/-- Represents a family in the house -/
structure Family where
  boys : ℕ
  girls : ℕ

/-- The census data for the house -/
structure CensusData where
  families : List Family

/-- Conditions from the problem -/
def ValidCensus (data : CensusData) : Prop :=
  ∀ f ∈ data.families,
    (f.boys > 0 → f.girls > 0) ∧  -- Every boy has a sister
    (f.boys + f.girls > 0)  -- No families without children

/-- Total number of boys in the house -/
def TotalBoys (data : CensusData) : ℕ :=
  (data.families.map (λ f => f.boys)).sum

/-- Total number of girls in the house -/
def TotalGirls (data : CensusData) : ℕ :=
  (data.families.map (λ f => f.girls)).sum

/-- Total number of children in the house -/
def TotalChildren (data : CensusData) : ℕ :=
  TotalBoys data + TotalGirls data

/-- Total number of adults in the house -/
def TotalAdults (data : CensusData) : ℕ :=
  2 * data.families.length

/-- The main theorem to prove -/
theorem census_contradiction (data : CensusData) 
  (h_valid : ValidCensus data)
  (h_more_boys : TotalBoys data > TotalGirls data) :
  TotalChildren data > TotalAdults data :=
sorry

end NUMINAMATH_CALUDE_census_contradiction_l778_77887


namespace NUMINAMATH_CALUDE_ratio_calculation_l778_77838

theorem ratio_calculation (X Y Z : ℚ) (h : X / Y = 3 / 2 ∧ Y / Z = 1 / 3) :
  (4 * X + 3 * Y) / (5 * Z - 2 * X) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l778_77838


namespace NUMINAMATH_CALUDE_complex_cubic_sum_ratio_l778_77809

theorem complex_cubic_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10)
  (h_eq : 2 * ((x - y)^2 + (x - z)^2 + (y - z)^2) = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_cubic_sum_ratio_l778_77809


namespace NUMINAMATH_CALUDE_biathlon_distance_l778_77859

/-- Biathlon problem -/
theorem biathlon_distance (total_distance : ℝ) (run_velocity : ℝ) (bike_velocity : ℝ) (total_time : ℝ)
  (h1 : total_distance = 155)
  (h2 : run_velocity = 10)
  (h3 : bike_velocity = 29)
  (h4 : total_time = 6) :
  ∃ (bike_distance : ℝ), 
    bike_distance + (total_distance - bike_distance) = total_distance ∧
    bike_distance / bike_velocity + (total_distance - bike_distance) / run_velocity = total_time ∧
    bike_distance = 145 := by
  sorry

end NUMINAMATH_CALUDE_biathlon_distance_l778_77859


namespace NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_65_l778_77898

theorem right_triangle_with_hypotenuse_65 :
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25 :=          -- Shorter leg length
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_65_l778_77898


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l778_77837

theorem cubic_inequality_solution (x : ℝ) :
  x^3 + x^2 - 7*x + 6 < 0 ↔ -3 < x ∧ x < 1 ∨ 1 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l778_77837


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_product_l778_77841

/-- Given four consecutive natural numbers x-1, x, x+1, and x+2, 
    if the product of their sum and the sum of their squares 
    equals three times the sum of their cubes, then x = 5. -/
theorem consecutive_numbers_sum_product (x : ℕ) : 
  (x - 1 + x + (x + 1) + (x + 2)) * 
  ((x - 1)^2 + x^2 + (x + 1)^2 + (x + 2)^2) = 
  3 * ((x - 1)^3 + x^3 + (x + 1)^3 + (x + 2)^3) → 
  x = 5 := by
  sorry

#check consecutive_numbers_sum_product

end NUMINAMATH_CALUDE_consecutive_numbers_sum_product_l778_77841


namespace NUMINAMATH_CALUDE_river_crossing_l778_77860

/-- Represents a river with islands -/
structure River :=
  (width : ℝ)
  (islandsPerimeter : ℝ)

/-- Theorem stating that it's possible to cross the river in less than 3 meters -/
theorem river_crossing (r : River) 
  (h_width : r.width = 1)
  (h_perimeter : r.islandsPerimeter = 8) : 
  ∃ (path : ℝ), path < 3 ∧ path ≥ r.width :=
sorry

end NUMINAMATH_CALUDE_river_crossing_l778_77860


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l778_77801

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l778_77801


namespace NUMINAMATH_CALUDE_am_gm_inequality_and_equality_condition_l778_77863

theorem am_gm_inequality_and_equality_condition (x : ℝ) (h : x > 0) :
  x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by sorry

end NUMINAMATH_CALUDE_am_gm_inequality_and_equality_condition_l778_77863


namespace NUMINAMATH_CALUDE_yellow_beads_proof_l778_77823

theorem yellow_beads_proof (green_beads : ℕ) (yellow_fraction : ℚ) : 
  green_beads = 4 → 
  yellow_fraction = 4/5 → 
  (yellow_fraction * (green_beads + 16 : ℚ)).num = 16 := by
  sorry

end NUMINAMATH_CALUDE_yellow_beads_proof_l778_77823


namespace NUMINAMATH_CALUDE_expression_equality_l778_77886

theorem expression_equality : 12 + 5*(4-9)^2 - 3 = 134 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l778_77886


namespace NUMINAMATH_CALUDE_charcoal_drawings_count_l778_77858

/-- The total number of drawings Thomas has -/
def total_drawings : ℕ := 120

/-- The number of drawings made with colored pencils -/
def colored_pencil_drawings : ℕ := 35

/-- The number of drawings made with blending markers -/
def blending_marker_drawings : ℕ := 22

/-- The number of drawings made with pastels -/
def pastel_drawings : ℕ := 15

/-- The number of drawings made with watercolors -/
def watercolor_drawings : ℕ := 12

/-- The number of charcoal drawings -/
def charcoal_drawings : ℕ := total_drawings - (colored_pencil_drawings + blending_marker_drawings + pastel_drawings + watercolor_drawings)

theorem charcoal_drawings_count : charcoal_drawings = 36 := by
  sorry

end NUMINAMATH_CALUDE_charcoal_drawings_count_l778_77858


namespace NUMINAMATH_CALUDE_xiaotong_message_forwarding_l778_77810

theorem xiaotong_message_forwarding :
  ∃ (x : ℕ), x > 0 ∧ 1 + x + x^2 = 91 :=
by sorry

end NUMINAMATH_CALUDE_xiaotong_message_forwarding_l778_77810


namespace NUMINAMATH_CALUDE_BRICS_is_set_closeToZero_is_not_l778_77885

-- Define a type for countries
structure Country where
  name : String

-- Define the BRICS summit participants
def BRICS2016Participants : Set Country := sorry

-- Define a property for real numbers "close to 0"
def closeToZero (x : ℝ) : Prop := sorry

theorem BRICS_is_set_closeToZero_is_not :
  (∃ (S : Set Country), S = BRICS2016Participants) ∧
  (¬ ∃ (T : Set ℝ), ∀ x, x ∈ T ↔ closeToZero x) :=
sorry

end NUMINAMATH_CALUDE_BRICS_is_set_closeToZero_is_not_l778_77885


namespace NUMINAMATH_CALUDE_ac_over_b_squared_range_l778_77822

/-- Given an obtuse triangle ABC with sides a, b, c satisfying a < b < c
    and internal angles forming an arithmetic sequence,
    the value of ac/b^2 is strictly between 0 and 2/3. -/
theorem ac_over_b_squared_range (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a < b → b < c →
  0 < A → A < π → 0 < B → B < π → 0 < C → C < π →
  A + B + C = π →
  C > π / 2 →
  ∃ (k : ℝ), B - A = C - B ∧ B = k * A ∧ C = (k + 1) * A →
  0 < a * c / (b * b) ∧ a * c / (b * b) < 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ac_over_b_squared_range_l778_77822


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l778_77897

theorem integer_roots_of_cubic (a : ℤ) : 
  (∃ x : ℤ, x^3 + 3*x^2 + a*x + 8 = 0) ↔ 
  a ∈ ({-89, -39, -30, -14, -12, -6, -2, 10} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l778_77897


namespace NUMINAMATH_CALUDE_class_size_proof_l778_77850

theorem class_size_proof (total : ℕ) : 
  (1 / 4 : ℚ) * total = total - ((3 / 4 : ℚ) * total) →
  (1 / 3 : ℚ) * ((3 / 4 : ℚ) * total) = ((3 / 4 : ℚ) * total) - 10 →
  10 = (2 / 3 : ℚ) * ((3 / 4 : ℚ) * total) →
  total = 20 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l778_77850


namespace NUMINAMATH_CALUDE_correct_allocation_count_l778_77899

def num_volunteers : ℕ := 4
def num_events : ℕ := 3

def allocation_schemes (n_volunteers : ℕ) (n_events : ℕ) : ℕ :=
  if n_volunteers < n_events then 0
  else (n_events.factorial * n_events^(n_volunteers - n_events))

theorem correct_allocation_count :
  allocation_schemes num_volunteers num_events = 18 :=
sorry

end NUMINAMATH_CALUDE_correct_allocation_count_l778_77899


namespace NUMINAMATH_CALUDE_garden_area_l778_77868

/-- Proves that the area of a rectangular garden with length three times its width
    and width of 14 meters is 588 square meters. -/
theorem garden_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 14 →
  length = 3 * width →
  area = length * width →
  area = 588 :=
by sorry

end NUMINAMATH_CALUDE_garden_area_l778_77868


namespace NUMINAMATH_CALUDE_inverse_73_mod_74_l778_77865

theorem inverse_73_mod_74 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 73 ∧ (73 * x) % 74 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_73_mod_74_l778_77865


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l778_77864

theorem min_value_of_sum_of_squares (x y : ℝ) (h : 4 * x^2 + 4 * x * y + 7 * y^2 = 3) :
  ∃ (m : ℝ), m = 3/8 ∧ x^2 + y^2 ≥ m ∧ ∃ (x₀ y₀ : ℝ), 4 * x₀^2 + 4 * x₀ * y₀ + 7 * y₀^2 = 3 ∧ x₀^2 + y₀^2 = m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l778_77864


namespace NUMINAMATH_CALUDE_simplify_expression_l778_77853

theorem simplify_expression : (7^5 + 2^7) * (2^3 - (-1)^3)^8 = 729000080835 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l778_77853


namespace NUMINAMATH_CALUDE_part_one_part_two_l778_77811

-- Define the sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | x < 2*m - 3}

-- Statement for part 1
theorem part_one (m : ℝ) (h : m = 5) : 
  A ∩ B m = A ∧ (Aᶜ ∪ B m) = Set.univ := by sorry

-- Statement for part 2
theorem part_two (m : ℝ) : 
  A ⊆ B m ↔ m > 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l778_77811


namespace NUMINAMATH_CALUDE_conference_handshakes_l778_77821

/-- Conference attendees are divided into three groups -/
structure ConferenceGroups where
  total : ℕ
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ

/-- Calculate the number of handshakes in the conference -/
def calculate_handshakes (groups : ConferenceGroups) : ℕ :=
  let group3_handshakes := groups.group3 * (groups.total - groups.group3)
  let group2_handshakes := groups.group2 * (groups.group1 + groups.group3)
  (group3_handshakes + group2_handshakes) / 2

/-- Theorem stating that the number of handshakes is 237 -/
theorem conference_handshakes :
  ∃ (groups : ConferenceGroups),
    groups.total = 40 ∧
    groups.group1 = 25 ∧
    groups.group2 = 10 ∧
    groups.group3 = 5 ∧
    calculate_handshakes groups = 237 := by
  sorry


end NUMINAMATH_CALUDE_conference_handshakes_l778_77821


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l778_77816

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 4 * a 8 = 4 →
  a 5 * a 6 * a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l778_77816


namespace NUMINAMATH_CALUDE_square_preserves_order_l778_77892

theorem square_preserves_order (a b : ℝ) : a > b ∧ b > 0 → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_preserves_order_l778_77892


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_ab_l778_77888

theorem factorization_a_squared_minus_ab (a b : ℝ) : a^2 - a*b = a*(a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_ab_l778_77888


namespace NUMINAMATH_CALUDE_power_division_l778_77815

theorem power_division : 2^12 / 8^3 = 8 := by sorry

end NUMINAMATH_CALUDE_power_division_l778_77815


namespace NUMINAMATH_CALUDE_fourth_pentagon_dots_l778_77817

/-- Calculates the number of dots in a pentagon given its layer number -/
def dots_in_pentagon (n : ℕ) : ℕ :=
  if n = 0 then 1
  else dots_in_pentagon (n - 1) + 5 * n

theorem fourth_pentagon_dots :
  dots_in_pentagon 3 = 31 := by
  sorry

#eval dots_in_pentagon 3

end NUMINAMATH_CALUDE_fourth_pentagon_dots_l778_77817


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l778_77866

/-- Given a hyperbola mx^2 + 5y^2 = 5m with eccentricity e = 2, prove that m = -15 -/
theorem hyperbola_eccentricity (m : ℝ) : 
  (∃ (x y : ℝ), m*x^2 + 5*y^2 = 5*m) → -- Hyperbola equation
  (∃ (e : ℝ), e = 2 ∧ e^2 = 1 - m/5) → -- Eccentricity definition
  m = -15 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l778_77866


namespace NUMINAMATH_CALUDE_unique_solution_l778_77844

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x, x > 0 → f x > 0}

/-- The functional equation that f must satisfy -/
def SatisfiesEquation (f : PositiveRealFunction) : Prop :=
  ∀ x y, x > 0 → y > 0 → f.val (x + f.val y) = f.val (x + y) + f.val y

/-- The theorem stating that the only solution is f(x) = 2x -/
theorem unique_solution (f : PositiveRealFunction) (h : SatisfiesEquation f) :
  ∀ x, x > 0 → f.val x = 2 * x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l778_77844


namespace NUMINAMATH_CALUDE_absolute_value_equation_roots_l778_77883

theorem absolute_value_equation_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ ≠ x₂) ∧ 
  (|x₁|^2 + |x₁| - 12 = 0) ∧ 
  (|x₂|^2 + |x₂| - 12 = 0) ∧
  (x₁ + x₂ = 0) ∧
  (∀ x : ℝ, |x|^2 + |x| - 12 = 0 → (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_roots_l778_77883


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l778_77874

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  ((x - 2)^2 + 3^2)^(1/2) = 6 → 
  x = 2 + 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l778_77874


namespace NUMINAMATH_CALUDE_abc_inequality_l778_77895

theorem abc_inequality : 
  let a : ℝ := (2/5)^(3/5)
  let b : ℝ := (2/5)^(2/5)
  let c : ℝ := (3/5)^(2/5)
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l778_77895


namespace NUMINAMATH_CALUDE_eddys_climbing_rate_l778_77803

/-- Proves that Eddy's climbing rate is 500 ft/hr given the conditions of the problem --/
theorem eddys_climbing_rate (hillary_climb_rate : ℝ) (hillary_descent_rate : ℝ) 
  (start_time : ℝ) (pass_time : ℝ) (base_camp_distance : ℝ) (hillary_stop_distance : ℝ) :
  hillary_climb_rate = 800 →
  hillary_descent_rate = 1000 →
  start_time = 6 →
  pass_time = 12 →
  base_camp_distance = 5000 →
  hillary_stop_distance = 1000 →
  ∃ (eddy_climb_rate : ℝ), eddy_climb_rate = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_eddys_climbing_rate_l778_77803


namespace NUMINAMATH_CALUDE_girls_in_school_l778_77867

theorem girls_in_school (total_people boys teachers : ℕ)
  (h1 : total_people = 1396)
  (h2 : boys = 309)
  (h3 : teachers = 772) :
  total_people - boys - teachers = 315 := by
sorry

end NUMINAMATH_CALUDE_girls_in_school_l778_77867


namespace NUMINAMATH_CALUDE_angle_Q_measure_l778_77849

-- Define a scalene triangle PQR
structure ScaleneTriangle where
  P : ℝ
  Q : ℝ
  R : ℝ
  scalene : P ≠ Q ∧ Q ≠ R ∧ R ≠ P
  sum_180 : P + Q + R = 180

-- Theorem statement
theorem angle_Q_measure (t : ScaleneTriangle) 
  (h1 : t.Q = 2 * t.P) 
  (h2 : t.R = 3 * t.P) : 
  t.Q = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_Q_measure_l778_77849


namespace NUMINAMATH_CALUDE_largest_class_size_l778_77827

/-- Represents the number of students in the largest class of a school -/
def largest_class (n : ℕ) : ℕ := n

/-- Represents the total number of students in the school -/
def total_students (n : ℕ) : ℕ := 
  (largest_class n) + 
  (largest_class n - 2) + 
  (largest_class n - 4) + 
  (largest_class n - 6) + 
  (largest_class n - 8)

/-- Theorem stating that the largest class has 25 students -/
theorem largest_class_size : 
  (total_students 25 = 105) ∧ (largest_class 25 = 25) := by
  sorry

#check largest_class_size

end NUMINAMATH_CALUDE_largest_class_size_l778_77827


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_product_l778_77839

/-- The coefficient of x^3 in the product of two specific polynomials -/
theorem coefficient_x_cubed_in_product : ∃ (p q : Polynomial ℤ),
  p = 3 * X^3 + 2 * X^2 + 4 * X + 5 ∧
  q = 4 * X^3 + 6 * X^2 + 5 * X + 2 ∧
  (p * q).coeff 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_product_l778_77839


namespace NUMINAMATH_CALUDE_expression_evaluation_l778_77856

theorem expression_evaluation :
  let a : ℚ := -1/3
  (3*a - 1)^2 + 3*a*(3*a + 2) = 3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l778_77856


namespace NUMINAMATH_CALUDE_expression_evaluation_l778_77880

theorem expression_evaluation : 
  let c : ℕ := 4
  (c^c - 2*c*(c-2)^c + c^2)^c = 431441456 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l778_77880


namespace NUMINAMATH_CALUDE_rainfall_increase_l778_77861

/-- Given the rainfall data for Rainville in 2010 and 2011, prove the increase in average monthly rainfall. -/
theorem rainfall_increase (average_2010 total_2011 : ℝ) (h1 : average_2010 = 35) 
  (h2 : total_2011 = 504) : ∃ x : ℝ, 
  12 * (average_2010 + x) = total_2011 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_increase_l778_77861


namespace NUMINAMATH_CALUDE_all_statements_false_l778_77855

def sharp (n : ℕ) : ℚ := 1 / (n + 1)

theorem all_statements_false :
  (sharp 4 + sharp 8 ≠ sharp 12) ∧
  (sharp 9 - sharp 3 ≠ sharp 6) ∧
  (sharp 5 * sharp 7 ≠ sharp 35) ∧
  (sharp 15 / sharp 3 ≠ sharp 5) := by
sorry

end NUMINAMATH_CALUDE_all_statements_false_l778_77855


namespace NUMINAMATH_CALUDE_product_evaluation_l778_77846

theorem product_evaluation (a : ℕ) (h : a = 7) : 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l778_77846


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l778_77832

theorem square_area_from_diagonal (d : ℝ) (h : d = 40) : 
  (d^2 / 2) = 800 := by sorry

#check square_area_from_diagonal

end NUMINAMATH_CALUDE_square_area_from_diagonal_l778_77832


namespace NUMINAMATH_CALUDE_walk_distance_proof_l778_77820

/-- Calculates the total distance walked given a constant speed and two walking durations -/
def total_distance (speed : ℝ) (duration1 : ℝ) (duration2 : ℝ) : ℝ :=
  speed * (duration1 + duration2)

/-- Proves that walking at 4 miles per hour for 2 hours and then 0.5 hours results in 10 miles -/
theorem walk_distance_proof :
  let speed := 4
  let duration1 := 2
  let duration2 := 0.5
  total_distance speed duration1 duration2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_walk_distance_proof_l778_77820


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l778_77875

/-- Given that the average of a and b is 40, and the average of b and c is 60,
    prove that the difference between c and a is 40. -/
theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 40)
  (h2 : (b + c) / 2 = 60) :
  c - a = 40 := by
  sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l778_77875


namespace NUMINAMATH_CALUDE_prime_relation_l778_77826

theorem prime_relation (p q : ℕ) : 
  Nat.Prime p ∧ 
  p = Nat.minFac (Nat.minFac 2) ∧ 
  q = 13 * p + 3 ∧ 
  Nat.Prime q → 
  q = 29 := by sorry

end NUMINAMATH_CALUDE_prime_relation_l778_77826


namespace NUMINAMATH_CALUDE_four_digit_multiple_of_65_l778_77854

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_number (n : ℕ) : ℕ :=
  let d := n % 10
  let c := (n / 10) % 10
  let b := (n / 100) % 10
  let a := n / 1000
  1000 * d + 100 * c + 10 * b + a

theorem four_digit_multiple_of_65 :
  ∃! n : ℕ, is_four_digit n ∧ 
            65 ∣ n ∧ 
            65 ∣ (reverse_number n) ∧
            n = 5005 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_multiple_of_65_l778_77854


namespace NUMINAMATH_CALUDE_twelve_point_zero_six_million_scientific_notation_l778_77806

-- Define 12.06 million
def twelve_point_zero_six_million : ℝ := 12.06 * 1000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.206 * 10^7

-- Theorem statement
theorem twelve_point_zero_six_million_scientific_notation :
  twelve_point_zero_six_million = scientific_notation :=
by sorry

end NUMINAMATH_CALUDE_twelve_point_zero_six_million_scientific_notation_l778_77806


namespace NUMINAMATH_CALUDE_coefficient_of_x8_in_expansion_l778_77896

/-- The coefficient of x^8 in the expansion of (1 + 3x - 2x^2)^5 is -720 -/
theorem coefficient_of_x8_in_expansion : 
  let p : Polynomial ℤ := 1 + 3 * X - 2 * X^2
  let coeff := (p^5).coeff 8
  coeff = -720 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x8_in_expansion_l778_77896


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l778_77879

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/5 + 1/6, 1/5 + 1/7, 1/5 + 1/3, 1/5 + 1/8, 1/5 + 1/9]
  (∀ x ∈ sums, x ≤ (1/5 + 1/3)) ∧ (1/5 + 1/3 = 8/15) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l778_77879


namespace NUMINAMATH_CALUDE_area_of_large_rectangle_l778_77878

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the area of a square -/
def Square.area (s : Square) : ℝ := s.side * s.side

/-- The theorem to be proved -/
theorem area_of_large_rectangle (shaded_square : Square) 
  (bottom_rect left_rect : Rectangle) :
  shaded_square.area = 4 →
  bottom_rect.width = shaded_square.side →
  bottom_rect.height + left_rect.height = shaded_square.side →
  left_rect.width + bottom_rect.width = shaded_square.side →
  (shaded_square.area + bottom_rect.area + left_rect.area = 12) := by
  sorry

end NUMINAMATH_CALUDE_area_of_large_rectangle_l778_77878


namespace NUMINAMATH_CALUDE_simplify_fraction_l778_77804

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l778_77804


namespace NUMINAMATH_CALUDE_max_value_on_circle_l778_77872

theorem max_value_on_circle (x y : ℝ) (h : x^2 + y^2 = 14*x + 6*y + 6) :
  3*x + 4*y ≤ 73 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l778_77872


namespace NUMINAMATH_CALUDE_function_extrema_l778_77807

theorem function_extrema (x : ℝ) (hx : x ∈ Set.Icc (-π/3) (π/4)) :
  let y := (1 / (Real.cos x)^2) + 2 * Real.tan x + 1
  ∃ (min_y max_y : ℝ),
    (∀ z ∈ Set.Icc (-π/3) (π/4), y ≤ max_y ∧ min_y ≤ ((1 / (Real.cos z)^2) + 2 * Real.tan z + 1)) ∧
    y = min_y ↔ x = -π/4 ∧
    y = max_y ↔ x = π/4 ∧
    min_y = 1 ∧
    max_y = 5 :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_l778_77807


namespace NUMINAMATH_CALUDE_binomial_coeff_x8_eq_10_l778_77819

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := sorry

-- Define the function to get the exponent of x in the general term
def x_exponent (r : ℕ) : ℚ := 15 - (7 * r) / 2

-- Define the function to find the binomial coefficient for x^8
def binomial_coeff_x8 (n : ℕ) : ℕ :=
  let r := 2 -- r is 2 when x_exponent(r) = 8
  binomial_coeff n r

-- Theorem statement
theorem binomial_coeff_x8_eq_10 :
  binomial_coeff_x8 5 = 10 := by sorry

end NUMINAMATH_CALUDE_binomial_coeff_x8_eq_10_l778_77819


namespace NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l778_77830

/-- Given that 8 oranges weigh as much as 6 apples, prove that 32 oranges weigh as much as 24 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℝ),
  orange_weight > 0 →
  apple_weight > 0 →
  8 * orange_weight = 6 * apple_weight →
  32 * orange_weight = 24 * apple_weight :=
by
  sorry

end NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l778_77830


namespace NUMINAMATH_CALUDE_angle_D_measure_l778_77848

-- Define a convex hexagon
structure ConvexHexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ
  sum_of_angles : A + B + C + D + E + F = 720
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧ 0 < E ∧ 0 < F

-- Define the specific conditions of the hexagon
def SpecialHexagon (h : ConvexHexagon) : Prop :=
  h.A = h.B ∧ h.B = h.C ∧   -- Angles A, B, and C are congruent
  h.D = h.E ∧ h.E = h.F ∧   -- Angles D, E, and F are congruent
  h.A + 30 = h.D            -- Angle A is 30° less than angle D

-- State the theorem
theorem angle_D_measure (h : ConvexHexagon) (special : SpecialHexagon h) : h.D = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l778_77848


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l778_77833

theorem vector_difference_magnitude (a b : ℝ × ℝ × ℝ) :
  a = (2, 3, -1) → b = (-2, 1, 3) → ‖a - b‖ = 6 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l778_77833


namespace NUMINAMATH_CALUDE_prime_power_minus_cube_eq_one_l778_77889

theorem prime_power_minus_cube_eq_one (p : ℕ) (hp : Prime p) :
  ∀ x y : ℕ, x > 0 → y > 0 → p^x - y^3 = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_prime_power_minus_cube_eq_one_l778_77889


namespace NUMINAMATH_CALUDE_average_sum_difference_l778_77829

theorem average_sum_difference (x₁ x₂ x₃ y₁ y₂ y₃ z₁ z₂ z₃ a b c : ℝ) 
  (hx : (x₁ + x₂ + x₃) / 3 = a)
  (hy : (y₁ + y₂ + y₃) / 3 = b)
  (hz : (z₁ + z₂ + z₃) / 3 = c) :
  ((x₁ + y₁ - z₁) + (x₂ + y₂ - z₂) + (x₃ + y₃ - z₃)) / 3 = a + b - c := by
  sorry

end NUMINAMATH_CALUDE_average_sum_difference_l778_77829


namespace NUMINAMATH_CALUDE_gcd_problem_l778_77802

theorem gcd_problem (h : Nat.Prime 97) : Nat.gcd (97^9 + 1) (97^9 + 97^2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l778_77802


namespace NUMINAMATH_CALUDE_gcd_property_l778_77891

theorem gcd_property (a b : ℤ) : Int.gcd a b = 1 → Int.gcd (2*a + b) (a*(a + b)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_property_l778_77891


namespace NUMINAMATH_CALUDE_player1_can_achieve_6_player2_can_prevent_above_6_max_achievable_sum_is_6_l778_77884

/-- Represents a cell on the 5x5 board -/
inductive Cell
| mk (row : Fin 5) (col : Fin 5)

/-- Represents the state of a cell (Empty, Marked by Player 1, or Marked by Player 2) -/
inductive CellState
| Empty
| Player1
| Player2

/-- Represents the game board -/
def Board := Cell → CellState

/-- Checks if a given 3x3 sub-square is valid on the 5x5 board -/
def isValid3x3Square (topLeft : Cell) : Prop :=
  ∃ (r c : Fin 3), topLeft = Cell.mk r c

/-- Computes the sum of a 3x3 sub-square -/
def subSquareSum (b : Board) (topLeft : Cell) : ℕ :=
  sorry

/-- The maximum sum of any 3x3 sub-square on the board -/
def maxSubSquareSum (b : Board) : ℕ :=
  sorry

/-- A strategy for Player 1 -/
def Player1Strategy := Board → Cell

/-- A strategy for Player 2 -/
def Player2Strategy := Board → Cell

/-- Simulates a game given strategies for both players -/
def playGame (s1 : Player1Strategy) (s2 : Player2Strategy) : Board :=
  sorry

/-- Theorem stating that Player 1 can always achieve a maximum 3x3 sub-square sum of at least 6 -/
theorem player1_can_achieve_6 :
  ∃ (s1 : Player1Strategy), ∀ (s2 : Player2Strategy),
    maxSubSquareSum (playGame s1 s2) ≥ 6 :=
  sorry

/-- Theorem stating that Player 2 can always prevent the maximum 3x3 sub-square sum from exceeding 6 -/
theorem player2_can_prevent_above_6 :
  ∃ (s2 : Player2Strategy), ∀ (s1 : Player1Strategy),
    maxSubSquareSum (playGame s1 s2) ≤ 6 :=
  sorry

/-- Main theorem combining the above results -/
theorem max_achievable_sum_is_6 :
  (∃ (s1 : Player1Strategy), ∀ (s2 : Player2Strategy),
    maxSubSquareSum (playGame s1 s2) ≥ 6) ∧
  (∃ (s2 : Player2Strategy), ∀ (s1 : Player1Strategy),
    maxSubSquareSum (playGame s1 s2) ≤ 6) :=
  sorry

end NUMINAMATH_CALUDE_player1_can_achieve_6_player2_can_prevent_above_6_max_achievable_sum_is_6_l778_77884


namespace NUMINAMATH_CALUDE_percentage_increase_l778_77877

theorem percentage_increase (original : ℝ) (new : ℝ) : 
  original = 50 → new = 80 → (new - original) / original * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l778_77877


namespace NUMINAMATH_CALUDE_problem_solution_l778_77852

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - x + 2

def tangent_line (x y : ℝ) : Prop := x + y - 2 = 0

theorem problem_solution (m : ℝ) (h_m : m > 0) :
  (∀ x : ℝ, x > 0 → x < 1/Real.exp 1 → (deriv f) x < 0) ∧
  (m > 1/Real.exp 1 → ∀ x : ℝ, 1/Real.exp 1 < x → x < m → (deriv f) x > 0) ∧
  (∀ x : ℝ, x > 0 → 2 * f x ≤ (deriv (g (-2))) x + 2) ∧
  (∀ a : ℝ, a ≥ -2 → ∀ x : ℝ, x > 0 → 2 * f x ≤ (deriv (g a)) x + 2) ∧
  tangent_line (0 : ℝ) (g 1 0) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l778_77852


namespace NUMINAMATH_CALUDE_roberts_journey_distance_l778_77893

/-- Represents the time in hours for each leg of Robert's journey -/
structure JourneyTimes where
  ab : ℝ
  bc : ℝ
  ca : ℝ

/-- Calculates the total distance of Robert's journey -/
def totalDistance (times : JourneyTimes) : ℝ :=
  let adjustedTime := times.ab + times.bc + times.ca - 1.5
  90 * adjustedTime

/-- Theorem stating that the total distance of Robert's journey is 1305 miles -/
theorem roberts_journey_distance (times : JourneyTimes) 
  (h1 : times.ab = 6)
  (h2 : times.bc = 5.5)
  (h3 : times.ca = 4.5) : 
  totalDistance times = 1305 := by
  sorry

#eval totalDistance { ab := 6, bc := 5.5, ca := 4.5 }

end NUMINAMATH_CALUDE_roberts_journey_distance_l778_77893


namespace NUMINAMATH_CALUDE_seans_soda_purchase_l778_77873

/-- The number of cans of soda Sean bought -/
def num_sodas : ℕ := sorry

/-- The cost of one soup in dollars -/
def cost_soup : ℚ := sorry

/-- The cost of the sandwich in dollars -/
def cost_sandwich : ℚ := sorry

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 18

theorem seans_soda_purchase :
  (num_sodas : ℚ) = cost_soup ∧
  cost_sandwich = 3 * cost_soup ∧
  (num_sodas : ℚ) * 1 + 2 * cost_soup + cost_sandwich = total_cost ∧
  num_sodas = 3 := by sorry

end NUMINAMATH_CALUDE_seans_soda_purchase_l778_77873


namespace NUMINAMATH_CALUDE_polynomial_value_equivalence_l778_77894

theorem polynomial_value_equivalence (x y : ℝ) :
  3 * x^2 + 4 * y + 9 = 8 → 9 * x^2 + 12 * y + 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_equivalence_l778_77894


namespace NUMINAMATH_CALUDE_power_relation_l778_77857

theorem power_relation (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 9) : x^(3*a - b) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l778_77857


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_2015029_l778_77818

/-- The area of a quadrilateral with vertices at (2, 4), (2, 2), (3, 2), and (2010, 2011) -/
def quadrilateralArea : ℝ := 2015029

/-- The vertices of the quadrilateral -/
def vertices : List (ℝ × ℝ) := [(2, 4), (2, 2), (3, 2), (2010, 2011)]

/-- Theorem stating that the area of the quadrilateral with the given vertices is 2015029 square units -/
theorem quadrilateral_area_is_2015029 :
  let computeArea : List (ℝ × ℝ) → ℝ := sorry -- Function to compute area from vertices
  computeArea vertices = quadrilateralArea := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_2015029_l778_77818


namespace NUMINAMATH_CALUDE_hiking_speeds_l778_77808

-- Define the hiking speeds and relationships
def lucas_speed : ℚ := 5
def mia_speed_ratio : ℚ := 3/4
def grace_speed_ratio : ℚ := 6/7
def liam_speed_ratio : ℚ := 4/3

-- Define the hiking speeds of Mia, Grace, and Liam
def mia_speed : ℚ := lucas_speed * mia_speed_ratio
def grace_speed : ℚ := mia_speed * grace_speed_ratio
def liam_speed : ℚ := grace_speed * liam_speed_ratio

-- Theorem to prove Grace's and Liam's hiking speeds
theorem hiking_speeds :
  grace_speed = 45/14 ∧ liam_speed = 30/7 := by
  sorry

end NUMINAMATH_CALUDE_hiking_speeds_l778_77808


namespace NUMINAMATH_CALUDE_harmonio_theorem_l778_77862

/-- Represents the student population at Harmonio Middle School -/
structure School where
  total : ℝ
  enjoy_singing : ℝ
  admit_liking : ℝ
  dislike_consistent : ℝ

/-- Conditions for Harmonio Middle School -/
def harmonio_conditions (s : School) : Prop :=
  s.total > 0 ∧
  s.enjoy_singing = 0.7 * s.total ∧
  s.admit_liking = 0.75 * s.enjoy_singing ∧
  s.dislike_consistent = 0.8 * (s.total - s.enjoy_singing)

/-- Theorem statement for the problem -/
theorem harmonio_theorem (s : School) (h : harmonio_conditions s) :
  let claim_dislike := s.dislike_consistent + (s.enjoy_singing - s.admit_liking)
  (s.enjoy_singing - s.admit_liking) / claim_dislike = 0.4217 := by
  sorry


end NUMINAMATH_CALUDE_harmonio_theorem_l778_77862


namespace NUMINAMATH_CALUDE_shopping_ratio_l778_77813

theorem shopping_ratio (emma_spent elsa_spent elizabeth_spent total_spent : ℚ) : 
  emma_spent = 58 →
  elizabeth_spent = 4 * elsa_spent →
  total_spent = 638 →
  emma_spent + elsa_spent + elizabeth_spent = total_spent →
  elsa_spent / emma_spent = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_shopping_ratio_l778_77813


namespace NUMINAMATH_CALUDE_adult_meals_calculation_l778_77847

/-- Given a ratio of kids meals to adult meals and the number of kids meals sold,
    calculate the number of adult meals sold. -/
def adult_meals_sold (kids_ratio : ℕ) (adult_ratio : ℕ) (kids_meals : ℕ) : ℕ :=
  (adult_ratio * kids_meals) / kids_ratio

/-- Theorem stating that given the specific ratio and number of kids meals,
    the number of adult meals sold is 49. -/
theorem adult_meals_calculation :
  adult_meals_sold 10 7 70 = 49 := by
  sorry

#eval adult_meals_sold 10 7 70

end NUMINAMATH_CALUDE_adult_meals_calculation_l778_77847


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l778_77805

theorem shaded_area_ratio : 
  ∀ (r₁ r₂ r₃ r₄ : ℝ), 
    r₁ = 1 → r₂ = 2 → r₃ = 3 → r₄ = 4 →
    (π * r₁^2 + π * r₃^2 - π * r₂^2) / (π * r₄^2) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l778_77805


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l778_77831

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 4 + 2 * a 8 * a 5 + a 9 * a 7 = 36 →
  a 5 + a 8 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l778_77831


namespace NUMINAMATH_CALUDE_triangle_special_case_triangle_inequality_l778_77840

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ
  hpos : 0 < a ∧ 0 < b ∧ 0 < c
  htri : a + b > c ∧ b + c > a ∧ c + a > b

-- Part (a)
theorem triangle_special_case (t : Triangle) 
  (h : 6 * t.area = 2 * t.a^2 + t.b * t.c) : 
  t.b = t.c ∧ t.b = Real.sqrt (5/2) * t.a :=
sorry

-- Part (b)
theorem triangle_inequality (t : Triangle) :
  3 * t.a^2 + 3 * t.b^2 - t.c^2 ≥ 4 * Real.sqrt 3 * t.area :=
sorry

end NUMINAMATH_CALUDE_triangle_special_case_triangle_inequality_l778_77840


namespace NUMINAMATH_CALUDE_number_pairs_sum_800_or_400_l778_77842

theorem number_pairs_sum_800_or_400 (x y : ℤ) (A : ℤ) (h1 : x ≥ y) (h2 : A = 800 ∨ A = 400) 
  (h3 : (x + y) + (x - y) + x * y + x / y = A) :
  (A = 800 ∧ ((x = 38 ∧ y = 19) ∨ (x = -42 ∧ y = -21) ∨ (x = 36 ∧ y = 9) ∨ 
              (x = -44 ∧ y = -11) ∨ (x = 40 ∧ y = 4) ∨ (x = -60 ∧ y = -6) ∨ 
              (x = 20 ∧ y = 1) ∨ (x = -60 ∧ y = -3))) ∨
  (A = 400 ∧ ((x = 19 ∧ y = 19) ∨ (x = -21 ∧ y = -21) ∨ (x = 36 ∧ y = 9) ∨ 
              (x = -44 ∧ y = -11) ∨ (x = 64 ∧ y = 4) ∨ (x = -96 ∧ y = -6) ∨ 
              (x = 75 ∧ y = 3) ∨ (x = -125 ∧ y = -5) ∨ (x = 100 ∧ y = 1) ∨ 
              (x = -300 ∧ y = -3))) :=
by sorry

end NUMINAMATH_CALUDE_number_pairs_sum_800_or_400_l778_77842


namespace NUMINAMATH_CALUDE_hapok_guarantee_l778_77843

/-- Represents the coin division game between Hapok and Glazok -/
structure CoinGame where
  totalCoins : Nat
  maxHandfuls : Nat

/-- Represents a strategy for Hapok -/
structure Strategy where
  coinsPerHandful : Nat

/-- Calculates the minimum number of coins Hapok can guarantee with a given strategy -/
def guaranteedCoins (game : CoinGame) (strategy : Strategy) : Nat :=
  let fullHandfuls := game.totalCoins / strategy.coinsPerHandful
  let remainingCoins := game.totalCoins % strategy.coinsPerHandful
  if fullHandfuls ≥ 2 * game.maxHandfuls - 1 then
    (game.maxHandfuls - 1) * strategy.coinsPerHandful + remainingCoins
  else
    (fullHandfuls - game.maxHandfuls) * strategy.coinsPerHandful

/-- Theorem stating that Hapok can guarantee at least 46 coins -/
theorem hapok_guarantee (game : CoinGame) (strategy : Strategy) :
  game.totalCoins = 100 →
  game.maxHandfuls = 9 →
  strategy.coinsPerHandful = 6 →
  guaranteedCoins game strategy ≥ 46 := by
  sorry

#eval guaranteedCoins { totalCoins := 100, maxHandfuls := 9 } { coinsPerHandful := 6 }

end NUMINAMATH_CALUDE_hapok_guarantee_l778_77843


namespace NUMINAMATH_CALUDE_dorokhov_vacation_cost_l778_77890

/-- Represents a travel agency with its pricing structure -/
structure TravelAgency where
  name : String
  under_age_price : ℕ
  over_age_price : ℕ
  age_threshold : ℕ
  discount_or_commission : ℚ
  is_discount : Bool

/-- Calculates the total cost for a family's vacation package -/
def calculate_cost (agency : TravelAgency) (num_adults num_children : ℕ) (child_age : ℕ) : ℚ :=
  let base_cost := 
    if child_age < agency.age_threshold
    then agency.under_age_price * num_children + agency.over_age_price * num_adults
    else agency.over_age_price * (num_adults + num_children)
  let adjustment := base_cost * agency.discount_or_commission
  if agency.is_discount
  then base_cost - adjustment
  else base_cost + adjustment

/-- The Dorokhov family vacation problem -/
theorem dorokhov_vacation_cost : 
  let globus : TravelAgency := {
    name := "Globus",
    under_age_price := 11200,
    over_age_price := 25400,
    age_threshold := 5,
    discount_or_commission := 2 / 100,
    is_discount := true
  }
  let around_world : TravelAgency := {
    name := "Around the World",
    under_age_price := 11400,
    over_age_price := 23500,
    age_threshold := 6,
    discount_or_commission := 1 / 100,
    is_discount := false
  }
  let globus_cost := calculate_cost globus 2 1 5
  let around_world_cost := calculate_cost around_world 2 1 5
  min globus_cost around_world_cost = 58984 := by sorry

end NUMINAMATH_CALUDE_dorokhov_vacation_cost_l778_77890


namespace NUMINAMATH_CALUDE_sourball_theorem_l778_77814

def sourball_problem (nellie jacob lana bucket_total : ℕ) : Prop :=
  nellie = 12 ∧
  jacob = nellie / 2 ∧
  lana = jacob - 3 ∧
  bucket_total = 30 ∧
  let total_eaten := nellie + jacob + lana
  let remaining := bucket_total - total_eaten
  remaining / 3 = 3

theorem sourball_theorem :
  ∃ (nellie jacob lana bucket_total : ℕ),
    sourball_problem nellie jacob lana bucket_total :=
by
  sorry

end NUMINAMATH_CALUDE_sourball_theorem_l778_77814


namespace NUMINAMATH_CALUDE_calculation_result_l778_77800

theorem calculation_result : 12.05 * 5.4 + 0.6 = 65.67 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l778_77800


namespace NUMINAMATH_CALUDE_john_quiz_goal_l778_77882

theorem john_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) (completed_quizzes : ℕ) (current_as : ℕ) :
  total_quizzes = 60 →
  goal_percentage = 70 / 100 →
  completed_quizzes = 40 →
  current_as = 25 →
  ∃ (max_non_as : ℕ),
    max_non_as = 3 ∧
    (total_quizzes - completed_quizzes - max_non_as) + current_as ≥ ⌈(goal_percentage * total_quizzes : ℚ)⌉ ∧
    ∀ (n : ℕ), n > max_non_as →
      (total_quizzes - completed_quizzes - n) + current_as < ⌈(goal_percentage * total_quizzes : ℚ)⌉ :=
by sorry

end NUMINAMATH_CALUDE_john_quiz_goal_l778_77882


namespace NUMINAMATH_CALUDE_sum_coordinates_of_B_l778_77871

/-- Given that M(6,8) is the midpoint of AB and A has coordinates (10,8), 
    prove that the sum of the coordinates of B is 10. -/
theorem sum_coordinates_of_B (A B M : ℝ × ℝ) : 
  M = (6, 8) → 
  A = (10, 8) → 
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  B.1 + B.2 = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_B_l778_77871


namespace NUMINAMATH_CALUDE_infimum_of_function_over_D_l778_77812

-- Define the set D
def D : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≠ p.2 ∧ p.1 ^ p.2 = p.2 ^ p.1}

-- State the theorem
theorem infimum_of_function_over_D (α β : ℝ) (hα : α > 0) (hβ : β > 0) (hαβ : α ≤ β) :
  ∃ (inf : ℝ), inf = Real.exp (α + β) ∧
    ∀ (x y : ℝ), (x, y) ∈ D → inf ≤ x^α * y^β :=
sorry

end NUMINAMATH_CALUDE_infimum_of_function_over_D_l778_77812


namespace NUMINAMATH_CALUDE_chocolate_milk_total_ounces_l778_77828

-- Define the ingredients per glass
def milk_per_glass : ℚ := 6
def syrup_per_glass : ℚ := 1.5
def cream_per_glass : ℚ := 0.5

-- Define the total available ingredients
def total_milk : ℚ := 130
def total_syrup : ℚ := 60
def total_cream : ℚ := 25

-- Define the size of each glass
def glass_size : ℚ := 8

-- Theorem to prove
theorem chocolate_milk_total_ounces :
  let max_glasses := min (total_milk / milk_per_glass) 
                         (min (total_syrup / syrup_per_glass) (total_cream / cream_per_glass))
  let full_glasses := ⌊max_glasses⌋
  full_glasses * glass_size = 168 := by
sorry

end NUMINAMATH_CALUDE_chocolate_milk_total_ounces_l778_77828
