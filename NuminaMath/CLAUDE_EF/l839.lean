import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_in_ratio_l839_83946

def is_ratio (a b : ℕ) (r s : ℕ) : Prop :=
  r * b = s * a

theorem first_number_in_ratio (a b : ℕ) :
  is_ratio a b 5 4 →
  Nat.lcm a b = 80 →
  a = 20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_in_ratio_l839_83946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_property_l839_83982

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- Function f(x) = x^2 + 2x - 2/x -/
noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 2/x

/-- Theorem statement -/
theorem hyperbola_eccentricity_property (h : Hyperbola) : 
  ∃ (M N : ℝ × ℝ),
    let e := eccentricity h
    M.2 = (h.b / h.a) * M.1 ∧ 
    N.1^2 / h.a^2 - N.2^2 / h.b^2 = 1 ∧
    M.1 > 0 ∧ M.2 > 0 ∧ N.1 > 0 ∧ N.2 > 0 ∧
    M.2 / (M.1 + h.a * e) = N.2 / N.1 →
    f e = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_property_l839_83982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_arrangements_three_objects_one_empty_l839_83973

theorem distinct_arrangements_three_objects_one_empty :
  (Finset.range 4).card = 24 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_arrangements_three_objects_one_empty_l839_83973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_property_l839_83939

def is_valid_number (N : ℕ) : Prop :=
  ∃ (c : ℕ) (k : ℕ),
    c ∈ ({1, 2, 3} : Set ℕ) ∧
    k = 197 ∧
    N = 132 * c * (10 ^ k)

def has_200_digits (N : ℕ) : Prop :=
  10^199 ≤ N ∧ N < 10^200

def remove_digits (N : ℕ) : ℕ :=
  let a := N / 10^199
  let c := (N / 10^197) % 10
  N - a * 10^199 - c * 10^197

theorem number_property (N : ℕ) :
  has_200_digits N →
  remove_digits N = N / 44 →
  is_valid_number N :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_property_l839_83939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_correct_l839_83904

/-- Calculate the total cost of Teresa's purchase at the local shop --/
def calculate_total_cost : ℚ :=
  let sandwich_price : ℚ := 7.75
  let sandwich_quantity : ℕ := 2
  let salami_price : ℚ := 4
  let brie_price : ℚ := 3 * salami_price
  let olive_price_per_pound : ℚ := 10
  let olive_quantity : ℚ := 1/4
  let feta_price_per_pound : ℚ := 8
  let feta_quantity : ℚ := 1/2
  let bread_price : ℚ := 2
  let popcorn_price : ℚ := 3.5
  let brie_discount : ℚ := 0.1
  let sandwich_discount : ℚ := 0.15
  let sales_tax : ℚ := 0.05

  let sandwich_cost : ℚ := sandwich_price * sandwich_quantity * (1 - sandwich_discount)
  let brie_cost : ℚ := brie_price * (1 - brie_discount)
  let olive_cost : ℚ := olive_price_per_pound * olive_quantity
  let feta_cost : ℚ := feta_price_per_pound * feta_quantity
  let popcorn_cost : ℚ := popcorn_price -- Buy-one-get-one-free applied

  let subtotal : ℚ := sandwich_cost + salami_price + brie_cost + olive_cost + feta_cost + bread_price + popcorn_cost
  let taxable_subtotal : ℚ := subtotal - popcorn_cost
  let tax_amount : ℚ := taxable_subtotal * sales_tax

  subtotal + tax_amount

theorem total_cost_is_correct : 
  (calculate_total_cost : ℚ) = 4185/100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_correct_l839_83904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_diligence_l839_83964

/-- The number of students in Diligence before the transfer -/
def D : ℕ := sorry

/-- The number of students in Industry before the transfer -/
def I : ℕ := sorry

/-- The number of students in Progress before the transfer -/
def P : ℕ := sorry

/-- The total number of students in all sections -/
def total : ℕ := 75

/-- Theorem stating the number of students in Diligence before the transfer -/
theorem students_in_diligence : 
  (D + 2 = I - 2 + 3) ∧ 
  (D + 2 = P - 3) ∧ 
  (D + I + P = total) → 
  D = 23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_diligence_l839_83964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_running_distance_l839_83999

/-- Calculates the total distance John travels given his running conditions --/
theorem john_running_distance
  (speed_alone : ℝ)
  (speed_reduction_alone : ℝ)
  (speed_with_dog : ℝ)
  (speed_reduction_with_dog : ℝ)
  (time_with_dog : ℝ)
  (elevation_gain_with_dog : ℝ)
  (time_alone : ℝ)
  (elevation_gain_alone : ℝ)
  (h1 : speed_alone = 4)
  (h2 : speed_reduction_alone = 0.5)
  (h3 : speed_with_dog = 6)
  (h4 : speed_reduction_with_dog = 0.75)
  (h5 : time_with_dog = 0.5)
  (h6 : elevation_gain_with_dog = 1000)
  (h7 : time_alone = 0.5)
  (h8 : elevation_gain_alone = 500) :
  (speed_with_dog - speed_reduction_with_dog * (elevation_gain_with_dog / 500)) * time_with_dog +
  (speed_alone - speed_reduction_alone * (elevation_gain_alone / 500)) * time_alone = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_running_distance_l839_83999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l839_83918

/-- Curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := x^2 / 12 + y^2 / 8 = 1

/-- Curve C₂ in polar coordinates -/
def C₂ (ρ θ : ℝ) : Prop := ρ^2 = 2 * ρ * Real.cos θ + 1

/-- Distance between two points in 2D space -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Theorem stating the minimum distance between points on C₁ and C₂ -/
theorem min_distance_C₁_C₂ : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    C₁ x₁ y₁ ∧ 
    (∃ ρ θ, C₂ ρ θ ∧ x₂ = ρ * Real.cos θ ∧ y₂ = ρ * Real.sin θ) ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 6 - Real.sqrt 2 ∧
    ∀ (x₁' y₁' x₂' y₂' : ℝ), 
      C₁ x₁' y₁' → 
      (∃ ρ' θ', C₂ ρ' θ' ∧ x₂' = ρ' * Real.cos θ' ∧ y₂' = ρ' * Real.sin θ') → 
      distance x₁' y₁' x₂' y₂' ≥ Real.sqrt 6 - Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l839_83918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_inequality_l839_83926

theorem cosine_sine_inequality (α β : ℝ) (h1 : 0 < α) (h2 : α < π/2) (h3 : 0 < β) (h4 : β < π/2) :
  Real.cos α + Real.cos β + Real.sqrt 2 * Real.sin α * Real.sin β ≤ 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_inequality_l839_83926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curved_triangle_area_l839_83914

/-- The area enclosed by three circular arcs of radius 3 units, each with a central angle of 90 degrees
    and intersecting at points of tangency, is equal to (27/4)π - 9 square units. -/
theorem curved_triangle_area (r : ℝ) (h : r = 3) : 
  (3 * ((1/4) * π * r^2) - (1/2) * (r * Real.sqrt 2) * (r * Real.sqrt 2)) = (27/4) * π - 9 := by
  -- Substitute r = 3
  have h1 : r = 3 := h
  
  -- Calculate sector area
  have sector_area : ℝ := (1/4) * π * r^2
  
  -- Calculate triangle area
  have triangle_area : ℝ := (1/2) * (r * Real.sqrt 2) * (r * Real.sqrt 2)
  
  -- Calculate enclosed area
  have enclosed_area : ℝ := 3 * sector_area - triangle_area
  
  -- Prove the equality
  sorry  -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curved_triangle_area_l839_83914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_angle_measure_l839_83948

-- Define the ratio of the angles
def angle_ratio : ℚ := 4 / 3

-- Define the sum of complementary angles in degrees
def complementary_sum : ℝ := 90

-- Define the smaller angle
noncomputable def smaller_angle : ℝ := complementary_sum / (1 + angle_ratio) * (angle_ratio.den : ℝ) / (angle_ratio.num : ℝ)

-- Theorem statement
theorem smaller_angle_measure :
  ∃ ε > 0, |smaller_angle - 38.571| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_angle_measure_l839_83948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_value_l839_83998

theorem square_value (p : ℤ) (square : ℤ) (h1 : square + p = 75) (h2 : (square + p) + p = 143) : square = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_value_l839_83998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l839_83976

/-- Given points A, B, and C in 3D space, prove that A is equidistant from B and C -/
theorem equidistant_point (A B C : ℝ × ℝ × ℝ) : 
  A = (0, 17, 0) ∧ B = (-2, 4, -6) ∧ C = (8, 5, 1) →
  (A.1 - B.1)^2 + (A.2.1 - B.2.1)^2 + (A.2.2 - B.2.2)^2 = 
  (A.1 - C.1)^2 + (A.2.1 - C.2.1)^2 + (A.2.2 - C.2.2)^2 := by
  sorry

#check equidistant_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l839_83976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l839_83919

theorem expression_evaluation : 
  (0.027 : ℝ) ^ (-(1/3 : ℝ)) - (-(1/7 : ℝ)) ^ (-2 : ℝ) + 256 ^ (3/4 : ℝ) - 3 ^ (-1 : ℝ) + (Real.sqrt 2 - 1) ^ (0 : ℝ) = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l839_83919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ticket_cost_l839_83978

noncomputable def deal_cost : ℝ := 20
noncomputable def savings : ℝ := 2

noncomputable def normal_cost (ticket : ℝ) : ℝ :=
  ticket + (ticket - 3) + (ticket - 2) + (ticket - 2) / 2

theorem movie_ticket_cost :
  ∃ (ticket : ℝ),
    normal_cost ticket = deal_cost + savings ∧
    ticket = 8 := by
  use 8
  constructor
  · simp [normal_cost, deal_cost, savings]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ticket_cost_l839_83978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l839_83979

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the asymptote
def is_asymptote (f : ℝ → ℝ) (h : ℝ → ℝ → Prop) : Prop :=
  ∀ ε > 0, ∃ M > 0, ∀ x y, h x y → abs x > M → abs (y - f x) < ε

-- Theorem statement
theorem hyperbola_asymptote :
  ∃ f : ℝ → ℝ, (f = (λ x ↦ 2*x) ∨ f = (λ x ↦ -2*x)) ∧ is_asymptote f hyperbola :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l839_83979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_oil_price_l839_83907

/-- Represents the price reduction percentage -/
noncomputable def price_reduction : ℝ := 35

/-- Represents the additional amount of oil that can be bought after the price reduction -/
noncomputable def additional_oil : ℝ := 7.5

/-- Represents the fixed amount of money spent -/
noncomputable def fixed_cost : ℝ := 965

/-- Calculates the reduced price per kg of oil -/
noncomputable def reduced_price (original_price : ℝ) : ℝ :=
  original_price * (100 - price_reduction) / 100

/-- Theorem stating the reduced price of oil given the conditions -/
theorem reduced_oil_price :
  ∃ (original_price : ℝ),
    fixed_cost / (reduced_price original_price) = fixed_cost / original_price + additional_oil ∧
    abs (reduced_price original_price - 45.03) < 0.01 := by
  sorry

#check reduced_oil_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_oil_price_l839_83907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_value_l839_83938

/-- A convex polyhedron with triangular and hexagonal faces -/
structure Polyhedron where
  faces : ℕ
  triangles : ℕ
  hexagons : ℕ
  vertices : ℕ
  edges : ℕ
  faces_sum : faces = triangles + hexagons
  euler : vertices - edges + faces = 2
  edge_count : edges = (3 * triangles + 6 * hexagons) / 2
  vertex_config : vertices * 3 = 2 * triangles + hexagons

/-- The theorem to be proved -/
theorem polyhedron_value (P : Polyhedron) 
  (h1 : P.faces = 20) : 
  100 + 20 + P.vertices = 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_value_l839_83938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l839_83945

theorem quadratic_max_value (m : ℝ) (a b : ℝ) (H : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*m*x + m
  (f a = 0 ∧ f b = 0) →
  (4 ≤ a + b ∧ a + b ≤ 6) →
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f x ≤ H) →
  H = 3*m + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l839_83945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_chapter_pages_l839_83905

theorem second_chapter_pages (total_chapters first_chapter_pages page_difference second_chapter_pages : ℕ) :
  total_chapters = 2 →
  first_chapter_pages = 48 →
  first_chapter_pages = page_difference + second_chapter_pages →
  page_difference = 37 →
  second_chapter_pages = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_chapter_pages_l839_83905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laurent_expansion_f_l839_83937

noncomputable def f (z : ℂ) : ℂ := (z + 2) / (z^2 - 2*z - 3)

noncomputable def laurent_series_inner (z : ℂ) : ℂ :=
  (1/4) * ∑' n, ((-1:ℂ)^(n+1) - 5/(3:ℂ)^(n+1)) * z^n

noncomputable def laurent_series_middle (z : ℂ) : ℂ :=
  ∑' n, ((-1:ℂ)^n / (4 * z^n)) - ∑' n, (5 * z^n / (4 * (3:ℂ)^(n+1)))

noncomputable def laurent_series_outer (z : ℂ) : ℂ :=
  ∑' n, (((-1:ℂ)^n + 5 * (3:ℂ)^(n-1)) / (4 * z^n))

theorem laurent_expansion_f :
  ∀ z : ℂ,
    (Complex.abs z < 1 → f z = laurent_series_inner z) ∧
    (1 < Complex.abs z ∧ Complex.abs z < 3 → f z = laurent_series_middle z) ∧
    (3 < Complex.abs z → f z = laurent_series_outer z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laurent_expansion_f_l839_83937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_property_l839_83993

theorem infinite_pairs_property (m : ℕ+) :
  ∀ x y : ℤ, 
    (Int.gcd x y = 1) →
    (∃ k : ℤ, y * k = x^2 + m) →
    (∃ l : ℤ, x * l = y^2 + m) →
    ∃ x₁ : ℤ,
      x₁ = (x^2 + m) / y ∧
      x₁ > y ∧
      (Int.gcd x₁ y = 1) ∧
      (∃ k₁ : ℤ, x₁ * k₁ = y^2 + m) ∧
      (∃ l₁ : ℤ, y * l₁ = x₁^2 + m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_property_l839_83993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expressions_evaluation_l839_83959

theorem complex_expressions_evaluation :
  (1 - (-1)^2018 - |Real.sqrt 3 - 2| + Real.sqrt 81 + ((-27) ^ (1/3 : ℝ)) = 3 + Real.sqrt 3) ∧
  (Real.sqrt 2 * (Real.sqrt 2 + 2) - 3 * Real.sqrt 2 + Real.sqrt 3 * (1 / Real.sqrt 3) = 3 - Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expressions_evaluation_l839_83959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_is_600_l839_83947

/-- Calculates the monthly rent for a rectangular plot of farmland -/
noncomputable def monthly_rent (length width rent_per_acre_per_month : ℚ) : ℚ :=
  let area_sqft := length * width
  let area_acres := area_sqft / 43560
  area_acres * rent_per_acre_per_month

/-- Proves that the monthly rent for the given plot is $600 -/
theorem rent_is_600 :
  monthly_rent 360 1210 60 = 600 := by
  -- Unfold the definition of monthly_rent
  unfold monthly_rent
  -- Simplify the arithmetic expressions
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_is_600_l839_83947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l839_83989

theorem train_average_speed (x : ℝ) (h : x > 0) :
  (6 * x) / ((x / 40) + (x / 10) + (x / 20)) = 240 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l839_83989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l839_83971

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def Perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line y = mx + c is m -/
def slope_of_line (m c : ℝ) : ℝ := m

/-- The slope of the line ay + bx = c is -b/a -/
noncomputable def slope_of_general_line (a b c : ℝ) : ℝ := -b / a

theorem perpendicular_lines_b_value :
  let line1 := slope_of_line 3 7
  let line2 := slope_of_general_line 4 b 12
  Perpendicular line1 line2 → b = 4/3 := by
  sorry

#check perpendicular_lines_b_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l839_83971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_increasing_sum_l839_83912

/-- Definition of a geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a 0
  else a 0 * (1 - q^n) / (1 - q)

/-- The sequence of sums is increasing -/
def increasing_sum_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, geometric_sum a q (n + 1) > geometric_sum a q n

theorem geometric_sequence_increasing_sum (a : ℕ → ℝ) (q : ℝ) :
  (increasing_sum_sequence a q → q > 0) ∧
  (∃ a q, q > 0 ∧ ¬increasing_sum_sequence a q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_increasing_sum_l839_83912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_height_is_correct_l839_83910

/-- A trapezoid with specific properties -/
structure Trapezoid where
  /-- The length of the segment connecting the midpoints of the bases -/
  midline_length : ℝ
  /-- The angle at one end of the larger base -/
  angle1 : ℝ
  /-- The angle at the other end of the larger base -/
  angle2 : ℝ
  /-- The midline length is positive -/
  midline_pos : midline_length > 0
  /-- The first angle is 30 degrees -/
  angle1_is_30 : angle1 = 30 * Real.pi / 180
  /-- The second angle is 60 degrees -/
  angle2_is_60 : angle2 = 60 * Real.pi / 180

/-- The height of a trapezoid with the given properties -/
noncomputable def trapezoid_height (t : Trapezoid) : ℝ := 3 * Real.sqrt 3 / 2

/-- Theorem stating that the height of the trapezoid is (3√3)/2 -/
theorem trapezoid_height_is_correct (t : Trapezoid) (h : t.midline_length = 3) :
  trapezoid_height t = 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_height_is_correct_l839_83910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_train_length_eval_l839_83935

/-- Calculates the length of a train given its speed in km/h and time in seconds to cross an electric pole. -/
noncomputable def trainLength (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time

/-- Theorem stating that a train with a speed of 72 km/h, which takes 20 seconds to cross an electric pole, has a length of 400 meters. -/
theorem train_length_calculation (speed : ℝ) (time : ℝ) 
  (h1 : speed = 72) 
  (h2 : time = 20) : 
  trainLength speed time = 400 := by
  sorry

/-- Evaluates the train length for the given speed and time -/
theorem train_length_eval : 
  ∃ (x : ℚ), (x : ℝ) = trainLength 72 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_train_length_eval_l839_83935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l839_83917

theorem range_of_a (a : ℝ) : 
  (∀ x ≥ 3, Monotone (λ x : ℝ ↦ 2*x^2 + a*x + 4)) ∧ 
  (∀ x ∈ Set.Icc 0 1, a ≤ Real.exp x) → 
  a ∈ Set.Icc (-12) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l839_83917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l839_83966

/-- The surface area of a cube in cm² -/
def b : ℝ := sorry

/-- The volume increase constant in cm³ -/
def a : ℝ := sorry

/-- The side length of the original cube in cm -/
def x : ℝ := sorry

/-- The surface area of a cube is b cm² -/
axiom surface_area : b = 6 * x^2

/-- The volume increase when each side is increased by 3 cm is (2b - a) cm³ -/
axiom volume_increase : 9 * x^2 + 27 * x + 27 = 2 * b - a

/-- The theorem to prove -/
theorem cube_surface_area : b = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l839_83966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_crates_pigeonhole_l839_83934

theorem apple_crates_pigeonhole :
  ∀ (f : Fin 150 → ℕ),
  (∀ i, 110 ≤ f i ∧ f i ≤ 145) →
  ∃ (n : ℕ) (S : Finset (Fin 150)),
    S.card = 5 ∧ (∀ i j, i ∈ S → j ∈ S → f i = f j) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_crates_pigeonhole_l839_83934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_between_racers_is_50_minutes_time_between_racers_example_l839_83951

/-- The time in minutes between the start of the first racer and the third racer --/
noncomputable def time_between_racers (a : ℝ) : ℝ :=
  let t := (5 : ℝ) / 3
  let t1 := 1.5 * t
  (t1 - t) * 60

/-- The theorem stating that the time between racers is 50 minutes --/
theorem time_between_racers_is_50_minutes (a : ℝ) (h : a > 0) :
  time_between_racers a = 50 := by
  unfold time_between_racers
  -- Expand the definition and simplify
  simp [mul_sub, sub_mul]
  -- Perform the arithmetic
  norm_num
  -- QED

-- We can't use #eval with noncomputable definitions
-- Instead, we can state a theorem about the result
theorem time_between_racers_example :
  ∃ (a : ℝ), a > 0 ∧ time_between_racers a = 50 := by
  use 1
  constructor
  · -- Prove 1 > 0
    norm_num
  · -- Prove time_between_racers 1 = 50
    exact time_between_racers_is_50_minutes 1 (by norm_num)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_between_racers_is_50_minutes_time_between_racers_example_l839_83951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_gt_2x_plus_4_l839_83932

noncomputable section

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Domain of f is ℝ (implicitly defined by the type signature above)

-- f(-1) = 2
axiom f_at_neg_one : f (-1) = 2

-- For all x in ℝ, f'(x) > 2
axiom f_derivative_gt_two : ∀ x : ℝ, deriv f x > 2

-- Theorem stating the solution set
theorem solution_set_of_f_gt_2x_plus_4 :
  {x : ℝ | f x > 2 * x + 4} = {x : ℝ | x > -1} := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_gt_2x_plus_4_l839_83932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_schedules_l839_83958

/-- Represents a lesson type -/
inductive Lesson
| Chinese
| Mathematics
| English
| Music
| PhysicalEducation

/-- Represents a schedule of 5 lessons -/
def Schedule := Fin 5 → Lesson

/-- Checks if Chinese and English are adjacent in a schedule -/
def chineseEnglishAdjacent (s : Schedule) : Prop := sorry

/-- Checks if Music and Physical Education are not adjacent in a schedule -/
def musicPENotAdjacent (s : Schedule) : Prop := sorry

/-- Checks if Mathematics is not the first lesson in a schedule -/
def mathNotFirst (s : Schedule) : Prop := s 0 ≠ Lesson.Mathematics

/-- The set of all valid schedules -/
def ValidSchedules : Set Schedule :=
  {s | chineseEnglishAdjacent s ∧ musicPENotAdjacent s ∧ mathNotFirst s}

/-- Ensure ValidSchedules has a finite number of elements -/
instance : Fintype ValidSchedules := sorry

theorem number_of_valid_schedules : Fintype.card ValidSchedules = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_schedules_l839_83958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l839_83913

/-- The length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
noncomputable def train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) : ℝ :=
  (train_speed + person_speed) * (5/18) * passing_time

theorem train_length_calculation :
  let train_speed : ℝ := 60
  let person_speed : ℝ := 6
  let passing_time : ℝ := 11.999040076793857
  abs (train_length train_speed person_speed passing_time - 220) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l839_83913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_eq_one_solutions_l839_83974

theorem sin_plus_cos_eq_one_solutions (x : ℝ) :
  0 ≤ x ∧ x < 2 * Real.pi →
  (Real.sin x + Real.cos x = 1 ↔ x = 0 ∨ x = Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_eq_one_solutions_l839_83974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increase_interval_power_function_increase_to_infinity_l839_83950

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x^a

-- Define the property of passing through (2, 4)
def passesThroughTwoFour (f : ℝ → ℝ) : Prop :=
  f 2 = 4

-- Define the interval of increase
def intervalOfIncrease (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- The main theorem
theorem power_function_increase_interval
  (f : ℝ → ℝ)
  (h1 : isPowerFunction f)
  (h2 : passesThroughTwoFour f) :
  intervalOfIncrease f 0 (Real.sqrt Real.pi) := by
  sorry

-- Additional theorem to show that the interval extends to positive infinity
theorem power_function_increase_to_infinity
  (f : ℝ → ℝ)
  (h1 : isPowerFunction f)
  (h2 : passesThroughTwoFour f) :
  ∀ x y : ℝ, 0 ≤ x ∧ x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increase_interval_power_function_increase_to_infinity_l839_83950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_blocks_needed_l839_83977

/-- Represents the dimensions of a rectangular block -/
structure Block where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a cylindrical sculpture -/
structure Cylinder where
  height : ℝ
  diameter : ℝ

/-- Calculates the volume of a rectangular block -/
def blockVolume (b : Block) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a cylindrical sculpture -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ :=
  Real.pi * (c.diameter / 2)^2 * c.height

/-- The main theorem stating the minimum number of blocks needed -/
theorem min_blocks_needed (block : Block) (sculpture : Cylinder) :
  block.length = 6 ∧ block.width = 2 ∧ block.height = 1 ∧
  sculpture.height = 7 ∧ sculpture.diameter = 4 →
  (∃ n : ℕ, n * blockVolume block ≥ cylinderVolume sculpture ∧
             ∀ m : ℕ, m < n → m * blockVolume block < cylinderVolume sculpture) →
  (∃ n : ℕ, n = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_blocks_needed_l839_83977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_f_implies_a_range_l839_83957

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 2 then (a - 1) * x - 2 * a else Real.log x / Real.log a

theorem monotone_decreasing_f_implies_a_range (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x > f a y) :
  Real.sqrt 2 / 2 ≤ a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_f_implies_a_range_l839_83957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_problem_l839_83970

/-- Given two points A and B in the plane, and a point P, this theorem proves:
    1. The equation of the perpendicular bisector of AB
    2. The equation of the line parallel to AB passing through P
    3. The equation of the reflected ray from B to A after hitting the line in (2) -/
theorem geometry_problem (A B P : ℝ × ℝ) (h1 : A = (8, -6)) (h2 : B = (2, 2)) (h3 : P = (2, -3)) :
  ∃ (perpendicular_bisector parallel_line reflected_ray : ℝ → ℝ → Prop),
    (∀ x y, perpendicular_bisector x y ↔ 
      3 * x - 4 * y - 23 = 0) ∧
    (∀ x y, parallel_line x y ↔ 
      4 * x + 3 * y + 1 = 0) ∧
    (∀ x y, reflected_ray x y ↔
      11 * x + 27 * y + 74 = 0) ∧
    (∀ x y, perpendicular_bisector x y ↔ 
      (x - (A.1 + B.1) / 2) * (B.2 - A.2) = (y - (A.2 + B.2) / 2) * (B.1 - A.1)) ∧
    (∀ x y, parallel_line x y ↔ 
      (y - P.2) * (B.1 - A.1) = (x - P.1) * (B.2 - A.2)) ∧
    (∃ Q : ℝ × ℝ, 
      parallel_line Q.1 Q.2 ∧
      (Q.1 - B.1) * (A.1 - Q.1) + (Q.2 - B.2) * (A.2 - Q.2) = 0 ∧
      ∀ x y, reflected_ray x y ↔ (y - A.2) * (Q.1 - B.1) = (x - A.1) * (Q.2 - B.2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_problem_l839_83970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_area_of_specific_prism_l839_83925

/-- Represents a triangular prism -/
structure TriangularPrism where
  -- Base side length
  base : ℝ
  -- Height of the prism
  height : ℝ
  -- Assertion that lateral edges are perpendicular to the base
  lateral_perpendicular : True

/-- The surface area of the circumscribed sphere of a triangular prism -/
noncomputable def circumscribed_sphere_area (prism : TriangularPrism) : ℝ :=
  4 * Real.pi * (prism.base^2 / 3 + prism.height^2 / 4)

/-- Theorem stating the surface area of the circumscribed sphere for the given prism -/
theorem circumscribed_sphere_area_of_specific_prism :
  let prism : TriangularPrism := {
    base := Real.sqrt 3
    height := 2 * Real.sqrt 3
    lateral_perpendicular := True.intro
  }
  circumscribed_sphere_area prism = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_area_of_specific_prism_l839_83925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_negative_half_l839_83988

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + Real.arctan x * Real.sqrt (x^2 + 1)

-- Define the equation
def equation (x : ℝ) : Prop :=
  2*x + 1 + Real.arctan x * Real.sqrt (x^2 + 1) + Real.arctan (x+1) * Real.sqrt (x^2 + 2*x + 2) = 0

-- Theorem statement
theorem solution_is_negative_half :
  equation (-1/2) ∧ ∀ x : ℝ, equation x → x = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_negative_half_l839_83988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_proof_l839_83969

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
theorem triangle_abc_proof 
  (a b c : ℝ) 
  (A B C : ℝ)
  (h1 : Real.cos B / b = -(Real.cos C / (2 * a + c)))
  (h2 : b = Real.sqrt 13)
  (h3 : a + c = 4) :
  B = 2 * Real.pi / 3 ∧ 
  (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_proof_l839_83969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_subset_not_perfect_power_sum_l839_83962

/-- A sequence of natural numbers -/
def q : ℕ → ℕ := sorry

/-- The set A constructed from the sequence q -/
def A : Set ℕ := {n : ℕ | ∃ i, q i = n}

/-- Predicate to check if a natural number is a perfect power (a^b where a, b ≥ 2) -/
def is_perfect_power (n : ℕ) : Prop := ∃ a b : ℕ, a ≥ 2 ∧ b ≥ 2 ∧ a^b = n

theorem infinite_subset_not_perfect_power_sum :
  Set.Infinite A ∧
  ∀ (S : Finset ℕ), (↑S : Set ℕ) ⊆ A → ¬ is_perfect_power (S.sum id) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_subset_not_perfect_power_sum_l839_83962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l839_83986

-- Define the hyperbola
noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 100 - y^2 / 64 = 1

-- Define the asymptote slope
noncomputable def asymptote_slope : ℝ := 4 / 5

-- Theorem statement
theorem hyperbola_asymptote_slope :
  ∀ x y : ℝ, hyperbola x y →
  ∃ ε > 0, ∀ t : ℝ, t > 1/ε →
    (|y - asymptote_slope * x| < ε * |x| ∨ |y + asymptote_slope * x| < ε * |x|) := by
  sorry

#check hyperbola_asymptote_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l839_83986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_last_segment_speed_l839_83920

/-- The average speed during the last 45 minutes of John's trip -/
noncomputable def last_segment_speed (total_distance : ℝ) (total_time : ℝ) (first_segment_speed : ℝ) (second_segment_speed : ℝ) : ℝ :=
  3 * (total_distance / total_time) - first_segment_speed - second_segment_speed

/-- Theorem stating that John's average speed during the last 45 minutes was 45 mph -/
theorem johns_last_segment_speed :
  last_segment_speed 150 (135 / 60) 75 80 = 45 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval last_segment_speed 150 (135 / 60) 75 80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_last_segment_speed_l839_83920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_difference_l839_83963

/-- Calculates the stream speed given boat speed and time ratio -/
noncomputable def streamSpeed (boatSpeed : ℝ) (timeRatio : ℝ) : ℝ :=
  (timeRatio * boatSpeed - boatSpeed) / (timeRatio + 1)

theorem stream_speed_difference : 
  let boat1Speed : ℝ := 36
  let boat2Speed : ℝ := 54
  let timeRatio1 : ℝ := 2
  let timeRatio2 : ℝ := 3
  streamSpeed boat2Speed timeRatio2 - streamSpeed boat1Speed timeRatio1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_difference_l839_83963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_squared_l839_83968

-- Define the quadrilateral EFGH and the inscribed circle
structure InscribedCircle where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  O : ℝ × ℝ
  r : ℝ

-- Define the properties of the inscribed circle
def is_valid_inscribed_circle (c : InscribedCircle) : Prop :=
  -- The circle is tangent to EF at R and to GH at S
  ((c.R.1 - c.O.1)^2 + (c.R.2 - c.O.2)^2 = c.r^2) ∧
  ((c.S.1 - c.O.1)^2 + (c.S.2 - c.O.2)^2 = c.r^2) ∧
  -- Given lengths
  ((c.E.1 - c.R.1)^2 + (c.E.2 - c.R.2)^2 = 24^2) ∧
  ((c.R.1 - c.F.1)^2 + (c.R.2 - c.F.2)^2 = 31^2) ∧
  ((c.G.1 - c.S.1)^2 + (c.G.2 - c.S.2)^2 = 40^2) ∧
  ((c.S.1 - c.H.1)^2 + (c.S.2 - c.H.2)^2 = 29^2)

theorem inscribed_circle_radius_squared (c : InscribedCircle) 
  (h : is_valid_inscribed_circle c) : c.r^2 = 945 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_squared_l839_83968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trays_to_fill_equals_eight_l839_83984

-- Define the given conditions
def ice_cubes_per_soda : ℕ := 8
def ice_cubes_per_lemonade : ℕ := 2 * ice_cubes_per_soda
def num_guests : ℕ := 5
def spaces_per_tray : ℕ := 14
def used_percentage : ℚ := 3/4

-- Define the function to calculate the number of trays needed
noncomputable def trays_needed : ℕ :=
  let total_used := ice_cubes_per_lemonade * num_guests
  let total_ice_cubes := (total_used : ℚ) / used_percentage
  (Int.ceil (total_ice_cubes / spaces_per_tray)).toNat

-- Theorem statement
theorem trays_to_fill_equals_eight :
  trays_needed = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trays_to_fill_equals_eight_l839_83984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_distances_count_l839_83985

noncomputable section

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Represents the rectangular prism with all points -/
structure Prism where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E : Point3D
  F : Point3D
  G : Point3D
  H : Point3D
  I : Point3D
  J : Point3D
  K : Point3D
  L : Point3D
  M : Point3D
  N : Point3D
  O : Point3D
  P : Point3D

/-- Check if a prism satisfies the given conditions -/
def isValidPrism (p : Prism) : Prop :=
  distance p.A p.E = 3 ∧
  distance p.B p.F = 3 ∧
  distance p.C p.G = 3 ∧
  distance p.D p.H = 3 ∧
  distance p.A p.I = 1 ∧
  distance p.I p.J = 1 ∧
  distance p.J p.E = 1 ∧
  distance p.B p.K = 1 ∧
  distance p.K p.L = 1 ∧
  distance p.L p.F = 1 ∧
  distance p.C p.M = 1 ∧
  distance p.M p.N = 1 ∧
  distance p.N p.G = 1 ∧
  distance p.D p.O = 1 ∧
  distance p.O p.P = 1 ∧
  distance p.P p.H = 1

/-- Count the number of point pairs with distance √2 -/
noncomputable def countSqrt2Distances (p : Prism) : ℕ :=
  let points := [p.A, p.B, p.C, p.D, p.E, p.F, p.G, p.H, p.I, p.J, p.K, p.L, p.M, p.N, p.O, p.P]
  (points.foldl (λ acc x => 
    acc + (points.filter (λ y => distance x y = Real.sqrt 2)).length
  ) 0) / 2

theorem sqrt2_distances_count (p : Prism) : 
  isValidPrism p → countSqrt2Distances p = 32 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_distances_count_l839_83985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_eq_neg_pi_fourth_l839_83975

noncomputable section

open Real

theorem alpha_plus_beta_eq_neg_pi_fourth 
  (h1 : Real.cos (2 * α) = -Real.sqrt 10 / 10)
  (h2 : Real.sin (α - β) = Real.sqrt 5 / 5)
  (h3 : α ∈ Set.Ioo (π / 4) (π / 2))
  (h4 : β ∈ Set.Ioo (-π) (-π / 2)) :
  α + β = -π / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_eq_neg_pi_fourth_l839_83975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l839_83906

theorem parallel_vectors_lambda (a b : ℝ × ℝ) (lambda : ℝ) :
  a = (2, 5) →
  b = (lambda, 4) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  lambda = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l839_83906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_german_team_goals_l839_83992

def journalist1 (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2 (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3 (x : ℕ) : Prop := x % 2 = 1

def exactly_two_correct (x : ℕ) : Prop :=
  (journalist1 x ∧ journalist2 x ∧ ¬journalist3 x) ∨
  (journalist1 x ∧ ¬journalist2 x ∧ journalist3 x) ∨
  (¬journalist1 x ∧ journalist2 x ∧ journalist3 x)

theorem german_team_goals :
  ∀ x : ℕ, exactly_two_correct x ↔ x ∈ ({11, 12, 14, 16, 17} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_german_team_goals_l839_83992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l839_83930

theorem power_equation (a b : ℝ) (h1 : (30 : ℝ)^a = 2) (h2 : (30 : ℝ)^b = 3) :
  (15 : ℝ)^((1 - a - b) / (2 * (1 - b))) = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l839_83930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_classes_l839_83903

/-- Represents the number of students in each class -/
def s : ℕ+ := 1

/-- Represents the number of classes in the school -/
def num_classes : ℕ := 1

/-- The number of books a student reads in a year -/
def student_yearly_books : ℕ := 5 * 12

/-- The total number of books read by all students in a year -/
def total_yearly_books : ℕ := 60

/-- The number of students in a class -/
def class_size : ℕ+ := s

/-- Theorem stating that the number of classes in the school is 1 -/
theorem school_classes : 
  (student_yearly_books = 60) →
  (total_yearly_books = 60) →
  (class_size = s) →
  num_classes = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_classes_l839_83903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_average_score_l839_83902

/-- Represents the scores of three students in a quiz. -/
structure QuizScores where
  dorothy : ℚ
  ivanna : ℚ
  tatuya : ℚ

/-- Calculates the average score of the three students. -/
def average_score (scores : QuizScores) : ℚ :=
  (scores.dorothy + scores.ivanna + scores.tatuya) / 3

/-- Theorem stating the average score of the three students given the conditions. -/
theorem quiz_average_score :
  ∀ (scores : QuizScores),
    scores.dorothy = 90 →
    scores.ivanna = 3/5 * scores.dorothy →
    scores.tatuya = 2 * scores.ivanna →
    average_score scores = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_average_score_l839_83902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l839_83995

-- Define the hyperbola C
def hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

-- Define the line
def line (x y m : ℝ) : Prop :=
  x - y + m = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 5

-- Main theorem
theorem hyperbola_intersection_theorem :
  ∀ (a b : ℝ) (m : ℝ),
  a > 0 ∧ b > 0 ∧
  eccentricity a b = Real.sqrt 3 ∧
  a = 1 ∧
  (∃ (x1 y1 x2 y2 : ℝ),
    hyperbola x1 y1 a b ∧
    hyperbola x2 y2 a b ∧
    line x1 y1 m ∧
    line x2 y2 m ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    circle_eq ((x1 + x2) / 2) ((y1 + y2) / 2)) →
  m = 1 ∨ m = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l839_83995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_ray_sine_cosine_sum_l839_83943

theorem angle_on_ray_sine_cosine_sum (α : ℝ) :
  (∃ (x y : ℝ), x > 0 ∧ y = -3/4 * x ∧ 
   x * Real.cos α = x ∧ y * Real.cos α = -x * Real.sin α) →
  2 * Real.sin α + Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_ray_sine_cosine_sum_l839_83943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_theorem_l839_83941

-- Define complex numbers
variable (z₁ z₂ : ℂ)
variable (m : ℝ)

-- Theorem statement
theorem complex_number_theorem
  (h₁ : z₁ = -2 + I)
  (h₂ : z₁ * z₂ = -5 + 5 * I) :
  (z₂ = 3 - I) ∧
  (∀ m, (-(m - 1) > 0 ∧ m^2 - 2*m - 3 < 0) ↔ -1 < m ∧ m < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_theorem_l839_83941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_decrease_l839_83915

/-- Represents the annual decrease in arable land -/
def annual_decrease : ℕ := sorry

/-- The current amount of arable land in acres -/
def initial_arable_land : ℕ := 100000

/-- The increase in grain yield per unit area over 10 years -/
def yield_increase : ℚ := 22 / 100

/-- The increase in per capita grain possession -/
def per_capita_increase : ℚ := 10 / 100

/-- The annual population growth rate -/
def population_growth_rate : ℚ := 1 / 100

/-- The number of years in the planning period -/
def years : ℕ := 10

theorem max_annual_decrease :
  ∃ (max_decrease : ℕ),
    (max_decrease = 4) ∧
    (∀ d : ℕ, d ≤ max_decrease →
      let final_arable_land := initial_arable_land - d * years
      let final_population := initial_arable_land * (1 + population_growth_rate) ^ years
      let final_yield := initial_arable_land * (1 + yield_increase)
      let final_per_capita := (final_yield * final_arable_land) / final_population
      final_per_capita ≥ initial_arable_land * (1 + per_capita_increase)) ∧
    (∀ d : ℕ, d > max_decrease →
      ∃ final_arable_land final_population final_yield final_per_capita,
        final_arable_land = initial_arable_land - d * years ∧
        final_population = initial_arable_land * (1 + population_growth_rate) ^ years ∧
        final_yield = initial_arable_land * (1 + yield_increase) ∧
        final_per_capita = (final_yield * final_arable_land) / final_population ∧
        final_per_capita < initial_arable_land * (1 + per_capita_increase)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_decrease_l839_83915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clicks_approximate_speed_l839_83994

/-- Represents the length of a rail in feet -/
noncomputable def rail_length : ℝ := 40

/-- Converts miles per hour to feet per minute -/
noncomputable def mph_to_fpm (speed : ℝ) : ℝ := speed * 5280 / 60

/-- Calculates the number of clicks per minute given a speed in miles per hour -/
noncomputable def clicks_per_minute (speed : ℝ) : ℝ := mph_to_fpm speed / rail_length

/-- Theorem stating that the number of clicks in 30 seconds approximates the speed in mph -/
theorem clicks_approximate_speed (speed : ℝ) (h : speed > 0) :
  ∃ ε > 0, |clicks_per_minute speed * 0.5 - speed| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clicks_approximate_speed_l839_83994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_f_50_eq_18_l839_83972

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- f₁(n) is three times the number of positive integer divisors of n -/
def f₁ (n : ℕ) : ℕ := 3 * num_divisors n

/-- For j ≥ 1, fⱼ(n) = f₁(fⱼ₋₁(n)) when j > 1, and f₁(n) when j = 1 -/
def f : ℕ → ℕ → ℕ
  | 0, n => f₁ n
  | 1, n => f₁ n
  | (j+2), n => f₁ (f (j+1) n)

/-- The number of positive integers n ≤ 60 for which f₅₀(n) = 18 is equal to 13 -/
theorem count_f_50_eq_18 : 
  Finset.card (Finset.filter (fun n => n ≤ 60 ∧ f 50 n = 18) (Finset.range 61)) = 13 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_f_50_eq_18_l839_83972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clay_volume_constant_l839_83996

/-- The volume of a piece of clay remains constant when reshaped -/
theorem clay_volume_constant (m : ℝ) (V₁ V₂ : ℝ) (ρ₁ ρ₂ : ℝ) 
  (h₁ : m = ρ₁ * V₁) (h₂ : m = ρ₂ * V₂) (h₃ : ρ₁ > 0) (h₄ : ρ₂ > 0) :
  V₁ = V₂ := by
  sorry

#check clay_volume_constant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clay_volume_constant_l839_83996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_2sin_x_max_value_l839_83983

theorem cos_2x_plus_2sin_x_max_value (x : ℝ) : Real.cos (2 * x) + 2 * Real.sin x ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_2sin_x_max_value_l839_83983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_rounded_to_tenth_l839_83949

-- Define the ratio
def ratio : ℚ := 10 / 15

-- Define the rounding function to the nearest tenth
def roundToTenth (x : ℚ) : ℚ := 
  (x * 10).floor / 10 + if (x * 10 - (x * 10).floor ≥ 1/2) then 1/10 else 0

-- Theorem statement
theorem ratio_rounded_to_tenth : roundToTenth ratio = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_rounded_to_tenth_l839_83949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_proof_l839_83901

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem total_distance_proof (start_x start_y end_x end_y stop1_x stop1_y stop2_x stop2_y : ℝ) 
  (h1 : start_x = 2 ∧ start_y = -3)
  (h2 : end_x = -3 ∧ end_y = 2)
  (h3 : stop1_x = 0 ∧ stop1_y = 0)
  (h4 : stop2_x = 1 ∧ stop2_y = 1) :
  distance start_x start_y stop1_x stop1_y + 
  distance stop1_x stop1_y stop2_x stop2_y + 
  distance stop2_x stop2_y end_x end_y = 
  Real.sqrt 13 + Real.sqrt 2 + Real.sqrt 17 := by
  sorry

#eval "Theorem statement type-checks correctly"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_proof_l839_83901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_minus_2b_l839_83900

/-- Given two planar vectors a and b with an angle of π/3 between them,
    |a| = 1, and |b| = 1/2, prove that |a - 2b| = 1 -/
theorem magnitude_a_minus_2b (a b : ℝ × ℝ) 
  (angle : Real.cos (π / 3) = (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (norm_a : Real.sqrt (a.1^2 + a.2^2) = 1)
  (norm_b : Real.sqrt (b.1^2 + b.2^2) = 1 / 2) : 
  Real.sqrt ((a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_minus_2b_l839_83900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_butterfly_collection_l839_83924

/-- Represents Martha's butterfly collection -/
structure ButterflyCollection where
  total : ℕ
  blue : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ
  black : ℕ

/-- Theorem about Martha's butterfly collection -/
theorem martha_butterfly_collection (c : ButterflyCollection) : 
  c.total = 120 ∧
  c.blue = 25 ∧
  c.red = 15 ∧
  c.blue = (5 * c.yellow / 2) ∧
  c.red = (3 * c.yellow / 2) ∧
  c.green = 3 * c.yellow ∧
  c.total = c.blue + c.red + c.yellow + c.green + c.black →
  c.black = 40 := by
  sorry

#check martha_butterfly_collection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_butterfly_collection_l839_83924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosC_range_max_area_l839_83981

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the inequality condition
def inequalityHolds (C : Real) : Prop :=
  ∀ x : Real, x^2 * Real.cos C + 2*x * Real.sin C + 3/2 ≥ 0

-- Define the perimeter condition
def perimeterIs9 (t : Triangle) : Prop :=
  t.a + t.b + t.c = 9

-- Theorem 1: Range of cos(C)
theorem cosC_range (t : Triangle) (h : inequalityHolds t.C) :
  1/2 ≤ Real.cos t.C ∧ Real.cos t.C < 1 := by
  sorry

-- Theorem 2: Maximum area
theorem max_area (t : Triangle) (h1 : inequalityHolds t.C) (h2 : perimeterIs9 t) :
  t.C = Real.pi / 3 → t.a * t.b * Real.sin t.C / 2 ≤ 9 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosC_range_max_area_l839_83981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_resilience_maximizes_probability_l839_83931

noncomputable section

/-- The probability of observing s new heads growing, given dragon's resilience x -/
def prob (x : ℝ) (s : ℕ) : ℝ :=
  x^s / (1 + x + x^2)

/-- The vector K representing the number of heads that grew back in 10 minutes -/
def K : List ℕ := [1, 2, 2, 1, 0, 2, 1, 0, 1, 2]

/-- The probability of observing the vector K given dragon's resilience x -/
noncomputable def prob_K (x : ℝ) : ℝ :=
  (List.map (prob x) K).prod

/-- The theorem stating that the dragon's resilience maximizes the probability -/
theorem dragon_resilience_maximizes_probability (x : ℝ) (h : x > 0) :
  x = (1 + Real.sqrt 97) / 8 ↔ IsLocalMax prob_K x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_resilience_maximizes_probability_l839_83931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_after_translation_l839_83960

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Apply a translation to a point -/
def translate (p : Point2D) (v : Vector2D) : Point2D :=
  ⟨p.x + v.x, p.y + v.y⟩

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem square_side_length_after_translation :
  let original_vertices : List Point2D := [⟨5, -3⟩, ⟨9, 1⟩, ⟨5, 5⟩, ⟨1, 1⟩]
  let translation_vector : Vector2D := ⟨-2, 3⟩
  let translated_vertices := original_vertices.map (λ v => translate v translation_vector)
  ∀ i : Fin 4, distance (translated_vertices[i]) (translated_vertices[(i + 1) % 4]) = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_after_translation_l839_83960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_center_l839_83997

/-- The function f representing a rotation in the complex plane -/
noncomputable def f (z : ℂ) : ℂ := ((2 + Complex.I * Real.sqrt 2) * z + (4 * Real.sqrt 2 + 10 * Complex.I)) / 3

/-- The center of rotation for the function f -/
noncomputable def c : ℂ := (14 * Real.sqrt 2) / 3 - (2 * Complex.I) / 3

/-- Theorem stating that c is indeed the center of rotation for f -/
theorem rotation_center : f c = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_center_l839_83997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_identity_l839_83909

theorem binomial_sum_identity (m n : ℕ) (h : n ≥ m) :
  (Finset.range (n - m + 1)).sum (λ i => Nat.choose (m + i + 1) m * Nat.choose (m + i + 1) (m + i)) =
  Nat.choose (m + 1) (m + 2) * Nat.choose (n + 2) (m + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_identity_l839_83909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_plus_one_fixed_point_l839_83923

/-- Given a function f: ℝ → ℝ such that f(0) = 1 and f has an inverse function,
    the graph of g(x) = f⁻¹(x) + 1 passes through the point (1,1). -/
theorem inverse_function_plus_one_fixed_point (f : ℝ → ℝ) (hf : Function.Bijective f) 
    (h0 : f 0 = 1) : 
    let f_inv := Function.invFun f
    let g := λ x => f_inv x + 1
    g 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_plus_one_fixed_point_l839_83923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l839_83936

/-- A parabola with given properties -/
noncomputable def Parabola : ℝ → ℝ := λ x ↦ (1/3) * x^2 - 2*x + 1

theorem parabola_properties :
  (∀ x, Parabola x = (1/3) * x^2 - 2*x + 1) ∧
  (Parabola 3 = -2) ∧
  (∀ h ≠ 0, Parabola (3 + h) = Parabola (3 - h)) ∧
  (Parabola 0 = 1) := by
  sorry

#check parabola_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l839_83936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surface_area_inscribed_cuboid_l839_83961

/-- Given a sphere of radius R, the maximum surface area of an inscribed cuboid is 8R^2 -/
theorem max_surface_area_inscribed_cuboid (R : ℝ) (R_pos : R > 0) :
  ∃ (cuboid : ℝ × ℝ × ℝ),
    (let ⟨a, b, c⟩ := cuboid; a^2 + b^2 + c^2 = 4 * R^2) ∧
    (∀ (other_cuboid : ℝ × ℝ × ℝ),
      (let ⟨x, y, z⟩ := other_cuboid; x^2 + y^2 + z^2 = 4 * R^2) →
      (let ⟨a, b, c⟩ := cuboid; 2*(a*b + b*c + a*c)) ≥
      (let ⟨x, y, z⟩ := other_cuboid; 2*(x*y + y*z + x*z))) ∧
    (let ⟨a, b, c⟩ := cuboid; 2*(a*b + b*c + a*c) = 8 * R^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surface_area_inscribed_cuboid_l839_83961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_divisor_15_factorial_l839_83928

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def divisors (n : ℕ) : Finset ℕ := Finset.filter (λ d => n % d = 0) (Finset.range n)

def oddDivisors (n : ℕ) : Finset ℕ := Finset.filter (λ d => d % 2 ≠ 0) (divisors n)

theorem probability_odd_divisor_15_factorial :
  (oddDivisors (factorial 15)).card / (divisors (factorial 15)).card = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_divisor_15_factorial_l839_83928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_negative_six_sevenths_l839_83956

/-- The slope of a line passing through two points (x₁, y₁) and (x₂, y₂) -/
def our_slope (x₁ y₁ x₂ y₂ : ℚ) : ℚ :=
  (y₂ - y₁) / (x₂ - x₁)

/-- Theorem: The slope of a straight line passing through (4, -2) and (-3, 4) is -6/7 -/
theorem line_slope_is_negative_six_sevenths :
  our_slope 4 (-2) (-3) 4 = -6/7 := by
  -- Unfold the definition of our_slope
  unfold our_slope
  -- Simplify the expression
  simp
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_negative_six_sevenths_l839_83956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l839_83967

-- Define the function f(x) = |sin²x - 4sinx - a|
noncomputable def f (x a : ℝ) : ℝ := |Real.sin x ^ 2 - 4 * Real.sin x - a|

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∀ x, f x a ≤ 4) ∧ (∃ x, f x a = 4) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l839_83967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l839_83921

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Represents a pyramid in 3D space -/
structure Pyramid where
  apex : Point3D
  base1 : Point3D
  base2 : Point3D
  base3 : Point3D

/-- Calculates the edge length of a cube -/
noncomputable def edge_length (c : Cube) : ℝ :=
  sorry

/-- Checks if a point is the midpoint of two other points -/
def is_midpoint (p : Point3D) (p1 p2 : Point3D) : Prop :=
  sorry

/-- Calculates the volume of a pyramid -/
noncomputable def volume (p : Pyramid) : ℝ :=
  sorry

/-- Theorem: The volume of the specific pyramid in the cube is 2/3 -/
theorem volume_of_specific_pyramid (c : Cube) (E : Point3D) :
  edge_length c = 2 →
  is_midpoint E c.C c.D →
  volume { apex := c.D, base1 := c.B, base2 := c.D₁, base3 := E } = 2/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l839_83921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l839_83916

theorem quadratic_root_form (m n p : ℤ) : 
  (∃ x : ℝ, 3 * x^2 - 7 * x + 4 = 0 ∧ 
   ∃ s : ℤ, s = 1 ∨ s = -1 ∧ x = (m + s * Real.sqrt (n : ℝ)) / (p : ℝ)) →
  Nat.gcd (Nat.gcd m.natAbs n.natAbs) p.natAbs = 1 →
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l839_83916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_store_time_is_56_minutes_l839_83922

-- Define the constants
noncomputable def walking_speed : ℝ := 2
noncomputable def running_speed : ℝ := 10
noncomputable def store_distance : ℝ := 4

-- Define the function to calculate time in minutes
noncomputable def time_to_store (speed : ℝ) : ℝ :=
  store_distance / speed * 60

-- Define the function to calculate average time
noncomputable def average_time (sunday_time tuesday_time thursday_time : ℝ) : ℝ :=
  (sunday_time + tuesday_time + thursday_time) / 3

-- Theorem statement
theorem average_store_time_is_56_minutes :
  average_time (time_to_store walking_speed) (time_to_store running_speed) (time_to_store running_speed) = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_store_time_is_56_minutes_l839_83922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_24_l839_83953

/-- A function that satisfies the symmetry property g(4+x) = g(4-x) for all real x -/
def SymmetricFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (4 + x) = g (4 - x)

/-- The set of roots of a function g -/
def RootSet (g : ℝ → ℝ) : Set ℝ :=
  {x | g x = 0}

/-- The theorem stating that if g is symmetric about x = 4 and has exactly 6 distinct roots,
    then the sum of these roots is 24 -/
theorem sum_of_roots_is_24 (g : ℝ → ℝ) 
    (h_sym : SymmetricFunction g) 
    (h_roots : ∃ s : Finset ℝ, s.card = 6 ∧ ∀ x, x ∈ s ↔ g x = 0) : 
    ∃ s : Finset ℝ, (∀ x, x ∈ s ↔ g x = 0) ∧ s.sum id = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_24_l839_83953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisection_light_reflection_l839_83929

-- Define the lines and points
def l₁ (x y : ℝ) : Prop := 2 * x + y - 8 = 0
def l₂ (x y : ℝ) : Prop := x - 3 * y + 10 = 0
def l (x y : ℝ) : Prop := 3 * x - 2 * y + 7 = 0
def P : ℝ × ℝ := (0, 1)

-- Define the bisection property
def is_bisected (l : (ℝ → ℝ → Prop)) (P : ℝ × ℝ) (l₁ l₂ : ℝ → ℝ → Prop) : Prop :=
  ∃ (A B : ℝ × ℝ), l A.1 A.2 ∧ l B.1 B.2 ∧ l₁ A.1 A.2 ∧ l₂ B.1 B.2 ∧
    P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define the reflection property
def reflects (l_incident l_reflect l_surface : ℝ → ℝ → Prop) : Prop :=
  ∃ (M : ℝ × ℝ), l_incident M.1 M.2 ∧ l_surface M.1 M.2 ∧ l_reflect M.1 M.2 ∧
    ∀ (x y : ℝ), l_incident x y → l_surface x y → l_reflect x y

-- State the theorems
theorem line_bisection :
  is_bisected (λ x y ↦ x + 4 * y - 4 = 0) P l₁ l₂ :=
sorry

theorem light_reflection :
  reflects (λ x y ↦ x - 2 * y + 5 = 0) (λ x y ↦ 29 * x - 2 * y + 33 = 0) l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisection_light_reflection_l839_83929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_and_increasing_functions_l839_83980

noncomputable section

-- Define the functions
def f (x : ℝ) := 2 - x
def g (x : ℝ) := x^2 + 2
def h (x : ℝ) := -1/x
def k (x : ℝ) := |x| + 1

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define what it means for a function to be increasing on (0, +∞)
def is_increasing_on_positive (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x ∧ x < y → f x < f y

theorem even_and_increasing_functions :
  (is_even g ∧ is_increasing_on_positive g) ∧
  (is_even k ∧ is_increasing_on_positive k) ∧
  ¬(is_even f ∧ is_increasing_on_positive f) ∧
  ¬(is_even h ∧ is_increasing_on_positive h) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_and_increasing_functions_l839_83980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_solution_l839_83908

/-- The sum of the infinite series 1 + 2x + 3x^2 + ... -/
noncomputable def S (x : ℝ) : ℝ := 1 / (1 - x)^2

theorem infinite_series_solution (x : ℝ) (hx : x < 1) :
  S x = 9 → x = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_solution_l839_83908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l839_83954

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

-- Theorem statement
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l839_83954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_fractions_l839_83987

variable (x : ℚ)

def fraction1 (x : ℚ) : ℚ := 1 / (4 * x)
def fraction2 (x : ℚ) : ℚ := 1 / (6 * x)
def fraction3 (x : ℚ) : ℚ := 1 / (9 * x)

theorem lcm_of_fractions (hx : x ≠ 0) :
  Nat.lcm (Nat.lcm (fraction1 x).den (fraction2 x).den) (fraction3 x).den = 36 * x.den :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_fractions_l839_83987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_initial_set_l839_83955

/-- A function that represents the polynomial operations that can be performed on the blackboard numbers -/
noncomputable def polynomial_operations (numbers : Set ℤ) : Set ℤ := sorry

/-- The set of integers between -2016 and 2016, inclusive -/
def target_set : Set ℤ := {x : ℤ | -2016 ≤ x ∧ x ≤ 2016}

/-- The theorem stating that 2 is the smallest number of initial numbers needed -/
theorem smallest_initial_set :
  ∀ (S : Finset ℤ), 0 ∈ S → S.card = 2 →
  (∃ (T : Set ℤ), ↑S ⊆ T ∧ target_set ⊆ T ∧ T ⊆ polynomial_operations ↑S) ∧
  (∀ (R : Finset ℤ), 0 ∈ R → R.card < 2 →
    ¬∃ (U : Set ℤ), ↑R ⊆ U ∧ target_set ⊆ U ∧ U ⊆ polynomial_operations ↑R) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_initial_set_l839_83955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_t_values_l839_83911

-- Define the ellipse equation
def ellipse_equation (x y t : ℝ) : Prop :=
  x^2 / t^2 + y^2 / (5*t) = 1

-- Define the focal length
noncomputable def focal_length : ℝ := 2 * Real.sqrt 6

-- Theorem statement
theorem ellipse_t_values :
  ∃ (t : ℝ), t > 0 ∧ 
  (∀ x y : ℝ, ellipse_equation x y t) ∧
  (t = 2 ∨ t = 3 ∨ t = 6) ∧
  (let c := focal_length / 2;
   (t > 5 ∧ c^2 = t^2 - 5*t) ∨
   (0 < t ∧ t < 5 ∧ c^2 = 5*t - t^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_t_values_l839_83911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_circle_l839_83942

theorem shaded_area_in_circle (r : ℝ) (h : r = 5) :
  2 * (π * r^2 / 4) + 2 * (r^2 / 2) = 25 + 12.5 * π := by
  have h1 : r^2 = 25 := by
    rw [h]
    ring
  calc
    2 * (π * r^2 / 4) + 2 * (r^2 / 2)
      = π * r^2 / 2 + r^2 := by ring
    _ = π * 25 / 2 + 25 := by rw [h1]
    _ = 12.5 * π + 25 := by ring
    _ = 25 + 12.5 * π := by ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_circle_l839_83942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l839_83944

theorem problem_statement :
  let A : Finset Nat := {0, 1}
  (Finset.card (Finset.powerset A) = 4) ∧
  (∃ (m : ℝ), m ≠ 0 ∧ ∃ (a b : ℝ), a < b ∧ ¬(a * m^2 < b * m^2)) ∧
  (∀ (p q : Prop), (p ∨ q → p ∧ q) → (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) ∧
  (¬(∀ (x : ℝ), x^2 - 3*x - 2 ≥ 0) ↔ (∃ (x : ℝ), x^2 - 3*x - 2 < 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l839_83944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l839_83927

/-- A polynomial that satisfies the given functional equation for a fixed k ≥ 1 -/
structure SolutionPolynomial (k : ℕ) where
  P : ℝ → ℝ
  is_polynomial : Polynomial ℝ
  satisfies_equation : ∀ x : ℝ, P (x^k) - P (k * x) = x^k * P x
  k_geq_one : k ≥ 1

/-- The set of all solution polynomials for a given k ≥ 1 -/
def SolutionSet (k : ℕ) : Set (SolutionPolynomial k) :=
  {p : SolutionPolynomial k | True}

theorem solution_characterization (k : ℕ) (h : k ≥ 1) :
  (SolutionSet k = {p : SolutionPolynomial k | p.P = fun _ ↦ 0}) ∨
  (k = 2 ∧ SolutionSet k = {p : SolutionPolynomial k | ∃ a : ℝ, p.P = fun x ↦ a * (x^2 - 4)}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l839_83927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_345_not_right_l839_83965

/-- A triangle with interior angles in the ratio 3:4:5 is not a right triangle -/
theorem triangle_345_not_right : 
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 
  (a + b + c = 180) → (a / b : ℝ) = 3 / 4 → (b / c : ℝ) = 4 / 5 → 
  ¬(a = 90 ∨ b = 90 ∨ c = 90) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_345_not_right_l839_83965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l839_83952

theorem vector_angle_problem (α β : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi/2) 
  (h3 : -Real.pi/2 < β) (h4 : β < 0)
  (h5 : (Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2 = (2*Real.sqrt 5/5)^2)
  (h6 : Real.cos β = 12/13) :
  (Real.cos (α - β) = 3/5) ∧ (Real.cos α = 56/65) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l839_83952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_walking_rate_l839_83990

/-- Given a distance in miles and a time in hours, calculates the rate in miles per hour -/
noncomputable def calculate_rate (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem jack_walking_rate :
  let distance : ℝ := 9
  let time : ℝ := 1.25
  calculate_rate distance time = 7.2 := by
  -- Unfold the definition of calculate_rate
  unfold calculate_rate
  -- Perform the division
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_walking_rate_l839_83990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ranking_l839_83933

-- Define the set of students
inductive Student : Type
  | xiaojun : Student
  | xiaomin : Student
  | xiaole : Student

-- Define a ranking as a function from Student to ℕ
def Ranking := Student → ℕ

-- Define the teacher's guesses
def teacherGuess1 (r : Ranking) : Prop := r Student.xiaojun = 1
def teacherGuess2 (r : Ranking) : Prop := r Student.xiaomin ≠ 1
def teacherGuess3 (r : Ranking) : Prop := r Student.xiaole ≠ 3

-- Define a valid ranking
def validRanking (r : Ranking) : Prop :=
  (r Student.xiaojun ∈ ({1, 2, 3} : Set ℕ)) ∧
  (r Student.xiaomin ∈ ({1, 2, 3} : Set ℕ)) ∧
  (r Student.xiaole ∈ ({1, 2, 3} : Set ℕ)) ∧
  (r Student.xiaojun ≠ r Student.xiaomin) ∧
  (r Student.xiaojun ≠ r Student.xiaole) ∧
  (r Student.xiaomin ≠ r Student.xiaole)

-- Define the condition that only one guess is correct
def onlyOneGuessCorrect (r : Ranking) : Prop :=
  (teacherGuess1 r ∧ ¬teacherGuess2 r ∧ ¬teacherGuess3 r) ∨
  (¬teacherGuess1 r ∧ teacherGuess2 r ∧ ¬teacherGuess3 r) ∨
  (¬teacherGuess1 r ∧ ¬teacherGuess2 r ∧ teacherGuess3 r)

-- The theorem to prove
theorem correct_ranking :
  ∃! r : Ranking, validRanking r ∧ onlyOneGuessCorrect r ∧
    r Student.xiaomin = 1 ∧ r Student.xiaole = 2 ∧ r Student.xiaojun = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ranking_l839_83933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_workers_l839_83940

/-- The total number of workers in the workshop -/
def W : ℕ := sorry

/-- The number of Type C technicians -/
def C : ℕ := sorry

/-- The average salary of all workers -/
def avg_salary : ℕ := 750

/-- The average salary of Type A technicians -/
def avg_salary_A : ℕ := 900

/-- The average salary of Type B technicians -/
def avg_salary_B : ℕ := 800

/-- The average salary of Type C technicians -/
def avg_salary_C : ℕ := 700

/-- The number of Type A technicians -/
def num_A : ℕ := 5

/-- The number of Type B technicians -/
def num_B : ℕ := 4

theorem workshop_workers :
  (W * avg_salary = num_A * avg_salary_A + num_B * avg_salary_B + C * avg_salary_C) →
  (W = num_A + num_B + C) →
  W = 28 := by
  sorry

#check workshop_workers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_workers_l839_83940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_range_l839_83991

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  point : Point
  angle : ℝ

/-- Checks if a line intersects a line segment -/
def intersectsSegment (l : Line) (a b : Point) : Prop :=
  sorry

/-- The range of angles for a line -/
structure AngleRange where
  min : ℝ
  max : ℝ

theorem line_angle_range :
  let p := Point.mk 1 0
  let a := Point.mk 2 1
  let b := Point.mk 0 (Real.sqrt 3)
  let l := Line.mk p 0  -- The angle doesn't matter for the statement
  intersectsSegment l a b →
  ∃ r : AngleRange, r.min = π/4 ∧ r.max = 2*π/3 ∧
    ∀ θ : ℝ, (θ ≥ r.min ∧ θ ≤ r.max) ↔ intersectsSegment (Line.mk p θ) a b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_range_l839_83991
