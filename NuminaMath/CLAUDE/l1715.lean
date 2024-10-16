import Mathlib

namespace NUMINAMATH_CALUDE_jakes_birdhouse_height_l1715_171528

/-- Represents the dimensions of a birdhouse in inches -/
structure BirdhouseDimensions where
  width : ℕ
  height : ℕ
  depth : ℕ

/-- Calculates the volume of a birdhouse given its dimensions -/
def birdhouse_volume (d : BirdhouseDimensions) : ℕ :=
  d.width * d.height * d.depth

theorem jakes_birdhouse_height :
  let sara_birdhouse : BirdhouseDimensions := {
    width := 12,  -- 1 foot = 12 inches
    height := 24, -- 2 feet = 24 inches
    depth := 24   -- 2 feet = 24 inches
  }
  let jake_birdhouse : BirdhouseDimensions := {
    width := 16,
    height := 20, -- We'll prove this is correct
    depth := 18
  }
  birdhouse_volume sara_birdhouse - birdhouse_volume jake_birdhouse = 1152 :=
by sorry


end NUMINAMATH_CALUDE_jakes_birdhouse_height_l1715_171528


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1715_171523

theorem gcd_of_specific_numbers (p : Nat) (h : Prime p) :
  Nat.gcd (p^10 + 1) (p^10 + p^3 + 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1715_171523


namespace NUMINAMATH_CALUDE_expression_evaluation_l1715_171593

theorem expression_evaluation : (4 + 5 + 6) / 3 * 2 - 2 / 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1715_171593


namespace NUMINAMATH_CALUDE_apple_distribution_l1715_171572

/-- The number of ways to distribute n apples among k people, with each person receiving at least m apples -/
def distribution_ways (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- The problem statement -/
theorem apple_distribution :
  distribution_ways 26 3 3 = 171 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1715_171572


namespace NUMINAMATH_CALUDE_min_garden_cost_l1715_171536

/-- Represents a rectangular region in the garden -/
structure Region where
  length : ℝ
  width : ℝ

/-- Represents a type of flower with its price -/
structure Flower where
  price : ℝ

/-- The garden layout -/
def garden : List Region := [
  ⟨5, 2⟩, -- Region 1
  ⟨3, 5⟩, -- Region 2
  ⟨2, 4⟩, -- Region 3
  ⟨5, 4⟩, -- Region 4
  ⟨5, 3⟩  -- Region 5
]

/-- Available flowers with their prices -/
def flowers : List Flower := [
  ⟨1.20⟩, -- Asters
  ⟨1.70⟩, -- Begonias
  ⟨2.20⟩, -- Cannas
  ⟨2.70⟩, -- Dahlias
  ⟨3.20⟩  -- Freesias
]

/-- Calculate the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Calculate the total area of the garden -/
def totalArea : ℝ := List.sum (List.map area garden)

/-- The main theorem: prove that the minimum cost is $152.60 -/
theorem min_garden_cost :
  ∃ (assignment : List (Region × Flower)),
    (List.length assignment = List.length garden) ∧
    (∀ r ∈ garden, ∃ f ∈ flowers, (r, f) ∈ assignment) ∧
    (List.sum (List.map (λ (r, f) => area r * f.price) assignment) = 152.60) ∧
    (∀ other_assignment : List (Region × Flower),
      (List.length other_assignment = List.length garden) →
      (∀ r ∈ garden, ∃ f ∈ flowers, (r, f) ∈ other_assignment) →
      List.sum (List.map (λ (r, f) => area r * f.price) other_assignment) ≥ 152.60) :=
by sorry

end NUMINAMATH_CALUDE_min_garden_cost_l1715_171536


namespace NUMINAMATH_CALUDE_triangle_area_l1715_171563

/-- The area of a triangle with side lengths 15, 36, and 39 is 270 -/
theorem triangle_area (a b c : ℝ) (ha : a = 15) (hb : b = 36) (hc : c = 39) :
  (1 / 2 : ℝ) * a * b = 270 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1715_171563


namespace NUMINAMATH_CALUDE_cubic_values_quadratic_polynomial_l1715_171526

theorem cubic_values_quadratic_polynomial 
  (a b c : ℕ) (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c) :
  ∃ (p q r : ℤ) (x₁ x₂ x₃ : ℤ), 
    p > 0 ∧ 
    (p * x₁^2 + q * x₁ + r = a^3) ∧
    (p * x₂^2 + q * x₂ + r = b^3) ∧
    (p * x₃^2 + q * x₃ + r = c^3) :=
sorry

end NUMINAMATH_CALUDE_cubic_values_quadratic_polynomial_l1715_171526


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1715_171504

/-- Given a distance of 8640 meters and a time of 36 minutes, 
    the average speed is 4 meters per second. -/
theorem average_speed_calculation (distance : ℝ) (time_minutes : ℝ) :
  distance = 8640 ∧ time_minutes = 36 →
  (distance / (time_minutes * 60)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1715_171504


namespace NUMINAMATH_CALUDE_pascals_theorem_l1715_171534

-- Define a circle
def Circle : Type := {p : ℝ × ℝ // (p.1^2 + p.2^2 = 1)}

-- Define a line
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

-- Define the intersection of two lines
def intersect (l1 l2 : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  l1 ∩ l2

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, q - p = t₁ • (r - p) ∧ r - p = t₂ • (q - p)

-- State Pascal's Theorem
theorem pascals_theorem 
  (A B C D E F : Circle) 
  (P : intersect (Line A.val B.val) (Line D.val E.val))
  (Q : intersect (Line B.val C.val) (Line E.val F.val))
  (R : intersect (Line C.val D.val) (Line F.val A.val)) :
  collinear P Q R :=
sorry

end NUMINAMATH_CALUDE_pascals_theorem_l1715_171534


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1715_171539

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function

/-- The common difference of the arithmetic sequence is 4 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h1 : seq.S 5 = -15)
  (h2 : seq.a 2 + seq.a 5 = -2) :
  seq.d = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1715_171539


namespace NUMINAMATH_CALUDE_ball_box_arrangement_l1715_171533

/-- The number of ways to arrange n balls in n boxes with exactly k balls in their corresponding boxes. -/
def arrangeWithExactMatches (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The number of derangements of n objects. -/
def derangement (n : ℕ) : ℕ :=
  sorry

theorem ball_box_arrangement :
  arrangeWithExactMatches 5 2 = 20 :=
sorry

end NUMINAMATH_CALUDE_ball_box_arrangement_l1715_171533


namespace NUMINAMATH_CALUDE_sum_of_sides_for_15cm_pentagon_l1715_171541

/-- A pentagon with all sides of equal length -/
structure RegularPentagon where
  side_length : ℝ
  side_length_positive : side_length > 0

/-- The sum of all side lengths of a regular pentagon -/
def sum_of_sides (p : RegularPentagon) : ℝ := 5 * p.side_length

/-- Theorem: If one side of a regular pentagon is 15 cm long, 
    then the sum of all side lengths is 75 cm -/
theorem sum_of_sides_for_15cm_pentagon : 
  ∀ (p : RegularPentagon), p.side_length = 15 → sum_of_sides p = 75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sides_for_15cm_pentagon_l1715_171541


namespace NUMINAMATH_CALUDE_horner_method_v3_l1715_171595

def f (x : ℝ) : ℝ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def horner_v3 (a₆ a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((a₆ * x + a₅) * x + a₄) * x + a₃

theorem horner_method_v3 :
  horner_v3 2 5 6 23 (-8) 10 (-3) 2 = 71 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l1715_171595


namespace NUMINAMATH_CALUDE_walking_distance_l1715_171514

-- Define the walking speeds in miles per minute
def jay_speed : ℚ := 1 / 20
def paul_speed : ℚ := 3 / 40
def sam_speed : ℚ := 1.5 / 30

-- Define the time period in minutes
def time_period : ℚ := 120

-- Define the theorem
theorem walking_distance :
  let jay_distance := jay_speed * time_period
  let paul_distance := paul_speed * time_period
  let sam_distance := sam_speed * time_period
  let max_north_distance := max jay_distance sam_distance
  paul_distance + max_north_distance = 15 := by
sorry

end NUMINAMATH_CALUDE_walking_distance_l1715_171514


namespace NUMINAMATH_CALUDE_area_of_ABC_l1715_171553

-- Define the triangle ABC and point P
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def P : ℝ × ℝ := sorry

-- Define the conditions
def is_scalene_right_triangle (A B C : ℝ × ℝ) : Prop := sorry
def point_on_hypotenuse (A C P : ℝ × ℝ) : Prop := sorry
def angle_ABP_45 (A B P : ℝ × ℝ) : Prop := sorry
def AP_equals_1 (A P : ℝ × ℝ) : Prop := sorry
def CP_equals_2 (C P : ℝ × ℝ) : Prop := sorry

-- Define the area function
def area (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_ABC :
  is_scalene_right_triangle A B C →
  point_on_hypotenuse A C P →
  angle_ABP_45 A B P →
  AP_equals_1 A P →
  CP_equals_2 C P →
  area A B C = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_ABC_l1715_171553


namespace NUMINAMATH_CALUDE_parabola_intersection_l1715_171564

/-- First parabola equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 5

/-- Second parabola equation -/
def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

/-- Theorem stating that (-0.5, 0.25) and (6, 49) are the only intersection points -/
theorem parabola_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = -0.5 ∧ y = 0.25) ∨ (x = 6 ∧ y = 49)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1715_171564


namespace NUMINAMATH_CALUDE_product_104_96_l1715_171555

theorem product_104_96 : 104 * 96 = 9984 := by
  sorry

end NUMINAMATH_CALUDE_product_104_96_l1715_171555


namespace NUMINAMATH_CALUDE_equal_area_division_ratio_l1715_171521

/-- Represents a T-shaped figure composed of unit squares -/
structure TShape :=
  (squares : ℕ)
  (is_t_shaped : squares = 22)

/-- Represents a line passing through a point in the T-shaped figure -/
structure DividingLine :=
  (t_shape : TShape)
  (divides_equally : Bool)

/-- Represents a segment in the T-shaped figure -/
structure Segment :=
  (t_shape : TShape)
  (length : ℚ)

/-- Theorem stating that if a line divides a T-shaped figure into equal areas,
    it divides a certain segment in the ratio 3:1 -/
theorem equal_area_division_ratio 
  (t : TShape) 
  (l : DividingLine) 
  (s : Segment) 
  (h1 : l.t_shape = t) 
  (h2 : s.t_shape = t) 
  (h3 : l.divides_equally = true) :
  s.length * (1/4) = 1 ∧ s.length * (3/4) = 3 :=
sorry

end NUMINAMATH_CALUDE_equal_area_division_ratio_l1715_171521


namespace NUMINAMATH_CALUDE_password_decryption_probability_l1715_171562

theorem password_decryption_probability :
  let p_A : ℚ := 1/5  -- Probability of A decrypting the password
  let p_B : ℚ := 1/4  -- Probability of B decrypting the password
  let p_either : ℚ := 1 - (1 - p_A) * (1 - p_B)  -- Probability of either A or B (or both) decrypting the password
  p_either = 2/5 := by sorry

end NUMINAMATH_CALUDE_password_decryption_probability_l1715_171562


namespace NUMINAMATH_CALUDE_discount_restoration_l1715_171568

theorem discount_restoration (original_price : ℝ) (discount_rate : ℝ) (restoration_rate : ℝ) : 
  discount_rate = 0.2 ∧ restoration_rate = 0.25 →
  original_price * (1 - discount_rate) * (1 + restoration_rate) = original_price :=
by sorry

end NUMINAMATH_CALUDE_discount_restoration_l1715_171568


namespace NUMINAMATH_CALUDE_rectangle_inscribed_circle_circumference_l1715_171586

theorem rectangle_inscribed_circle_circumference 
  (width : ℝ) (height : ℝ) (circumference : ℝ) :
  width = 7 ∧ height = 24 →
  circumference = Real.pi * Real.sqrt (width^2 + height^2) →
  circumference = 25 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_rectangle_inscribed_circle_circumference_l1715_171586


namespace NUMINAMATH_CALUDE_probability_different_colors_l1715_171540

def num_red_balls : ℕ := 2
def num_black_balls : ℕ := 3
def total_balls : ℕ := num_red_balls + num_black_balls

def different_color_combinations : ℕ := num_red_balls * num_black_balls

def total_combinations : ℕ := (total_balls * (total_balls - 1)) / 2

theorem probability_different_colors :
  (different_color_combinations : ℚ) / total_combinations = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_different_colors_l1715_171540


namespace NUMINAMATH_CALUDE_decreasing_cubic_condition_l1715_171545

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x

-- State the theorem
theorem decreasing_cubic_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_cubic_condition_l1715_171545


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1715_171584

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a < 1 → ∃ x : ℝ, x^2 - 2*x + a = 0) ∧
  ¬(∃ x : ℝ, x^2 - 2*x + a = 0 → a < 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1715_171584


namespace NUMINAMATH_CALUDE_economics_test_absentees_l1715_171506

theorem economics_test_absentees (total_students : ℕ) (q1_correct : ℕ) (q2_correct : ℕ) (both_correct : ℕ)
  (h1 : total_students = 40)
  (h2 : q1_correct = 30)
  (h3 : q2_correct = 29)
  (h4 : both_correct = 29)
  : total_students - (q1_correct + q2_correct - both_correct) = 10 := by
  sorry

#check economics_test_absentees

end NUMINAMATH_CALUDE_economics_test_absentees_l1715_171506


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l1715_171587

/-- A line is tangent to an ellipse if and only if it intersects the ellipse at exactly one point. -/
axiom tangent_iff_single_intersection {m : ℝ} :
  (∃! x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6) ↔
  (∃ x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6 ∧
    ∀ x' y' : ℝ, y' = m * x' + 2 ∧ 3 * x'^2 + 6 * y'^2 = 6 → x' = x ∧ y' = y)

/-- The theorem stating that if the line y = mx + 2 is tangent to the ellipse 3x^2 + 6y^2 = 6,
    then m^2 = 3/2. -/
theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6) → m^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l1715_171587


namespace NUMINAMATH_CALUDE_algebraic_expressions_equality_l1715_171557

theorem algebraic_expressions_equality (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  (((a^2 * b) / (-c))^3 * ((c^2) / (-a * b))^2) / ((b * c / a)^4) = -a^10 / (b^3 * c^7) ∧
  ((2 / (a^2 - b^2) - 1 / (a^2 - a * b)) / (a / (a + b))) = 1 / a^2 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expressions_equality_l1715_171557


namespace NUMINAMATH_CALUDE_cruise_group_selection_l1715_171576

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem cruise_group_selection :
  choose 9 4 = 126 := by
  sorry

end NUMINAMATH_CALUDE_cruise_group_selection_l1715_171576


namespace NUMINAMATH_CALUDE_quadrilateral_vector_proof_l1715_171571

-- Define the space
variable (V : Type*) [AddCommGroup V] [Module ℝ V]

-- Define the points and vectors
variable (O A B C D M N : V)
variable (a b c : V)

-- State the theorem
theorem quadrilateral_vector_proof 
  (h1 : O + a = A) 
  (h2 : O + b = B) 
  (h3 : O + c = C) 
  (h4 : ∃ t : ℝ, M = O + t • a) 
  (h5 : M - O = 2 • (A - M)) 
  (h6 : N = (1/2) • B + (1/2) • C) :
  M - N = -(2/3) • a + (1/2) • b + (1/2) • c := by sorry

end NUMINAMATH_CALUDE_quadrilateral_vector_proof_l1715_171571


namespace NUMINAMATH_CALUDE_roller_coaster_rides_l1715_171527

/-- Given a roller coaster that costs 5 tickets per ride and a person with 10 tickets,
    prove that the number of possible rides is 2. -/
theorem roller_coaster_rides (total_tickets : ℕ) (cost_per_ride : ℕ) (h1 : total_tickets = 10) (h2 : cost_per_ride = 5) :
  total_tickets / cost_per_ride = 2 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_rides_l1715_171527


namespace NUMINAMATH_CALUDE_max_element_of_A_l1715_171567

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) := Real.log x / Real.log 10

-- Define the set A
def A (x y : ℝ) : Set ℝ := {log x, log y, log (x + y/x)}

-- Define the theorem
theorem max_element_of_A :
  ∀ x y : ℝ, x > 0 → y > 0 → {0, 1} ⊆ A x y →
  ∃ z ∈ A x y, ∀ w ∈ A x y, w ≤ z ∧ z = log 11 :=
sorry

end NUMINAMATH_CALUDE_max_element_of_A_l1715_171567


namespace NUMINAMATH_CALUDE_all_diagonal_triangles_count_l1715_171589

/-- Represents a convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : ℕ
  is_convex : sides = n

/-- Represents a diagonal division of a polygon -/
structure DiagonalDivision (p : ConvexPolygon n) where
  diagonals : ℕ
  triangles : ℕ
  non_intersecting : Bool
  vertex_diagonals : ℕ → ℕ
  valid_division : diagonals = n - 3 ∧ triangles = n - 2
  valid_vertex_diagonals : ∀ v, vertex_diagonals v = 3 ∨ vertex_diagonals v = 0

/-- Counts the number of triangles with all sides as diagonals -/
def count_all_diagonal_triangles (p : ConvexPolygon 102) (d : DiagonalDivision p) : ℕ :=
  sorry

theorem all_diagonal_triangles_count 
  (p : ConvexPolygon 102) 
  (d : DiagonalDivision p) : 
  count_all_diagonal_triangles p d = 34 :=
sorry

end NUMINAMATH_CALUDE_all_diagonal_triangles_count_l1715_171589


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l1715_171574

def double_factorial (n : ℕ) : ℕ := 
  if n ≤ 1 then 1 else n * double_factorial (n - 2)

theorem greatest_prime_factor_of_sum (n : ℕ) : 
  ∃ p : ℕ, Nat.Prime p ∧ 
    p = Nat.gcd (double_factorial 22 + double_factorial 20) p ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (double_factorial 22 + double_factorial 20) → q ≤ p :=
by
  -- The proof goes here
  sorry

#check greatest_prime_factor_of_sum

end NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l1715_171574


namespace NUMINAMATH_CALUDE_a_9_value_l1715_171544

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_9_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 11) / 2 = 15 →
  a 1 + a 2 + a 3 = 9 →
  a 9 = 24 := by
sorry

end NUMINAMATH_CALUDE_a_9_value_l1715_171544


namespace NUMINAMATH_CALUDE_penelope_candy_count_l1715_171501

/-- The ratio of M&M candies to Starbursts candies -/
def candy_ratio : ℚ := 5 / 3

/-- The number of M&M candies Penelope has -/
def mm_count : ℕ := 25

/-- The number of Starbursts candies Penelope has -/
def starburst_count : ℕ := 15

/-- Theorem stating the relationship between M&M and Starbursts candies -/
theorem penelope_candy_count : 
  (mm_count : ℚ) / candy_ratio = starburst_count := by sorry

end NUMINAMATH_CALUDE_penelope_candy_count_l1715_171501


namespace NUMINAMATH_CALUDE_triangle_properties_l1715_171519

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)
  (angle_sum : A + B + C = π)

theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt 3 * Real.cos t.C + Real.sin t.C = (Real.sqrt 3 * t.a) / t.b)
  (h2 : t.a + t.c = 5 * Real.sqrt 7)
  (h3 : t.b = 7) :
  t.B = π / 3 ∧ t.a * t.c * Real.cos t.B = -21 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1715_171519


namespace NUMINAMATH_CALUDE_trailing_zeros_theorem_l1715_171590

/-- Count trailing zeros in factorial -/
def trailingZeros (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125)

/-- Check if n satisfies the condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  ∃ k : ℕ, trailingZeros (n + 3) = k ∧ trailingZeros (2 * n + 6) = 4 * k

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem -/
theorem trailing_zeros_theorem :
  ∃ t : ℕ,
    (∃ a b c d : ℕ,
      a > 6 ∧ b > 6 ∧ c > 6 ∧ d > 6 ∧
      a < b ∧ b < c ∧ c < d ∧
      satisfiesCondition a ∧ satisfiesCondition b ∧ satisfiesCondition c ∧ satisfiesCondition d ∧
      t = a + b + c + d ∧
      ∀ n : ℕ, n > 6 ∧ satisfiesCondition n → n ≥ a) ∧
    sumOfDigits t = 4 :=
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_theorem_l1715_171590


namespace NUMINAMATH_CALUDE_total_games_played_l1715_171566

-- Define the structure for the team's season statistics
structure SeasonStats where
  first100WinPercentage : ℝ
  homeWinPercentageAfter100 : ℝ
  awayWinPercentageAfter100 : ℝ
  overallWinPercentage : ℝ
  consecutiveWinStreak : ℕ

-- Define the theorem
theorem total_games_played (stats : SeasonStats) 
  (h1 : stats.first100WinPercentage = 0.85)
  (h2 : stats.homeWinPercentageAfter100 = 0.60)
  (h3 : stats.awayWinPercentageAfter100 = 0.45)
  (h4 : stats.overallWinPercentage = 0.70)
  (h5 : stats.consecutiveWinStreak = 15) :
  ∃ (totalGames : ℕ), totalGames = 186 ∧ 
  (∃ (remainingGames : ℕ), 
    remainingGames % 2 = 0 ∧
    totalGames = 100 + remainingGames ∧
    (85 + (stats.homeWinPercentageAfter100 + stats.awayWinPercentageAfter100) / 2 * remainingGames) / totalGames = stats.overallWinPercentage) :=
by
  sorry

end NUMINAMATH_CALUDE_total_games_played_l1715_171566


namespace NUMINAMATH_CALUDE_complex_square_simplification_l1715_171529

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (5 - 3 * i)^2 = 16 - 30 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l1715_171529


namespace NUMINAMATH_CALUDE_function_positive_l1715_171503

theorem function_positive (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, (x + 1) * f x + x * (deriv^[2] f x) > 0) : 
  ∀ x : ℝ, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_positive_l1715_171503


namespace NUMINAMATH_CALUDE_base_10_to_base_8_l1715_171554

theorem base_10_to_base_8 : 
  ∃ (a b c d : ℕ), 
    947 = a * 8^3 + b * 8^2 + c * 8^1 + d * 8^0 ∧ 
    a = 1 ∧ b = 6 ∧ c = 6 ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_8_l1715_171554


namespace NUMINAMATH_CALUDE_johns_numbers_l1715_171530

/-- Given a natural number, returns the number with its digits reversed -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is between 96 and 98 inclusive -/
def isBetween96And98 (n : ℕ) : Prop :=
  96 ≤ n ∧ n ≤ 98

/-- Represents the operation John performed on his number -/
def johnOperation (x : ℕ) : ℕ :=
  reverseDigits (4 * x + 17)

/-- A two-digit number satisfies John's conditions -/
def satisfiesConditions (x : ℕ) : Prop :=
  10 ≤ x ∧ x ≤ 99 ∧ isBetween96And98 (johnOperation x)

theorem johns_numbers :
  ∃ x y : ℕ, x ≠ y ∧ satisfiesConditions x ∧ satisfiesConditions y ∧
  (∀ z : ℕ, satisfiesConditions z → z = x ∨ z = y) ∧
  x = 13 ∧ y = 18 := by sorry

end NUMINAMATH_CALUDE_johns_numbers_l1715_171530


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l1715_171558

/-- Proves that the initial volume of a mixture is 425 litres given the conditions -/
theorem initial_mixture_volume :
  ∀ (V : ℝ),
  (V > 0) →
  (0.10 * V = V * 0.10) →
  (0.10 * V + 25 = 0.15 * (V + 25)) →
  V = 425 :=
λ V hV_pos hWater_ratio hNew_ratio =>
  sorry

#check initial_mixture_volume

end NUMINAMATH_CALUDE_initial_mixture_volume_l1715_171558


namespace NUMINAMATH_CALUDE_pizza_theorem_l1715_171556

/-- Calculates the number of pizza slices remaining after a series of consumption events. -/
def remainingSlices (initialSlices : ℕ) : ℕ :=
  let afterLunch := initialSlices / 2
  let afterDinner := afterLunch - (afterLunch / 3)
  let afterSharing := afterDinner - (afterDinner / 4)
  afterSharing - (afterSharing / 5)

/-- Theorem stating that given 12 initial slices, 3 slices remain after the described events. -/
theorem pizza_theorem : remainingSlices 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l1715_171556


namespace NUMINAMATH_CALUDE_jeans_cost_l1715_171579

theorem jeans_cost (shirt_cost hat_cost total_cost : ℚ) 
  (h1 : shirt_cost = 5)
  (h2 : hat_cost = 4)
  (h3 : total_cost = 51)
  (h4 : 3 * shirt_cost + 2 * (total_cost - 3 * shirt_cost - 4 * hat_cost) / 2 + 4 * hat_cost = total_cost) :
  (total_cost - 3 * shirt_cost - 4 * hat_cost) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jeans_cost_l1715_171579


namespace NUMINAMATH_CALUDE_towels_given_to_mother_l1715_171509

theorem towels_given_to_mother (green_towels white_towels remaining_towels : ℕ) : 
  green_towels = 35 → 
  white_towels = 21 → 
  remaining_towels = 22 → 
  green_towels + white_towels - remaining_towels = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_towels_given_to_mother_l1715_171509


namespace NUMINAMATH_CALUDE_cranberry_juice_can_ounces_l1715_171575

/-- Given a can of cranberry juice that sells for 84 cents with a unit cost of 7.0 cents per ounce,
    prove that the can contains 12 ounces of juice. -/
theorem cranberry_juice_can_ounces :
  ∀ (total_cost unit_cost : ℚ),
    total_cost = 84 →
    unit_cost = 7 →
    total_cost / unit_cost = 12 := by
sorry

end NUMINAMATH_CALUDE_cranberry_juice_can_ounces_l1715_171575


namespace NUMINAMATH_CALUDE_expression_evaluation_l1715_171513

theorem expression_evaluation : 
  let f (x : ℚ) := (2*x - 2) / (x + 2)
  let g (x : ℚ) := (2 * f x - 2) / (f x + 2)
  g 2 = -2/5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1715_171513


namespace NUMINAMATH_CALUDE_distance_AC_l1715_171502

/-- Given three points A, B, and C on a line, with AB = 5 and BC = 4, 
    the distance AC is either 1 or 9. -/
theorem distance_AC (A B C : ℝ) : 
  (A < B ∧ B < C) ∨ (C < B ∧ B < A) →  -- Points are on the same line
  |B - A| = 5 →                        -- AB = 5
  |C - B| = 4 →                        -- BC = 4
  |C - A| = 1 ∨ |C - A| = 9 :=         -- AC is either 1 or 9
by sorry


end NUMINAMATH_CALUDE_distance_AC_l1715_171502


namespace NUMINAMATH_CALUDE_rook_placement_on_colored_board_l1715_171552

theorem rook_placement_on_colored_board :
  let board_size : ℕ := 64
  let num_rooks : ℕ := 8
  let num_colors : ℕ := 32
  let cells_per_color : ℕ := 2

  let total_placements : ℕ := num_rooks.factorial
  let same_color_placements : ℕ := num_colors * (num_rooks - 2).factorial

  total_placements > same_color_placements :=
by sorry

end NUMINAMATH_CALUDE_rook_placement_on_colored_board_l1715_171552


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1715_171546

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = b) → ¬(a^2 - b^2 = 0)) ↔ (a^2 - b^2 = 0 → a = b) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1715_171546


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l1715_171588

theorem sum_of_squares_theorem (x y z t : ℤ) (h : x + y = z + t) :
  x^2 + y^2 + z^2 + t^2 = (x + y)^2 + (x - z)^2 + (x - t)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l1715_171588


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l1715_171580

theorem lcm_gcd_product (a b : ℕ) (ha : a = 15) (hb : b = 10) :
  Nat.lcm a b * Nat.gcd a b = 150 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l1715_171580


namespace NUMINAMATH_CALUDE_cube_sum_difference_l1715_171591

theorem cube_sum_difference (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 210 → a + b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_difference_l1715_171591


namespace NUMINAMATH_CALUDE_function_parameters_l1715_171559

/-- Given a function f(x) = 2sin(ωx + φ) with the specified properties, prove that ω = 2 and φ = π/3 -/
theorem function_parameters (ω φ : ℝ) (f : ℝ → ℝ) : 
  ω > 0 →
  |φ| < π/2 →
  (∀ x, f x = 2 * Real.sin (ω * x + φ)) →
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ π) →
  f 0 = Real.sqrt 3 →
  ω = 2 ∧ φ = π/3 := by
sorry

end NUMINAMATH_CALUDE_function_parameters_l1715_171559


namespace NUMINAMATH_CALUDE_switch_strategy_wins_l1715_171512

/-- Represents the three boxes in the game -/
inductive Box
| A
| B
| C

/-- Represents the possible states of a box -/
inductive BoxState
| Prize
| Empty

/-- Represents the game state -/
structure GameState where
  boxes : Box → BoxState
  initialChoice : Box
  hostOpened : Box
  finalChoice : Box

/-- The probability of winning by switching in the three-box game -/
def winProbabilityBySwitch (game : GameState) : ℚ :=
  2/3

/-- Theorem stating that the probability of winning by switching is greater than 1/2 -/
theorem switch_strategy_wins (game : GameState) :
  winProbabilityBySwitch game > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_switch_strategy_wins_l1715_171512


namespace NUMINAMATH_CALUDE_troy_computer_savings_l1715_171524

/-- The amount Troy needs to save to buy a new computer -/
theorem troy_computer_savings (new_computer_cost initial_savings old_computer_value : ℕ) 
  (h1 : new_computer_cost = 1800)
  (h2 : initial_savings = 350)
  (h3 : old_computer_value = 100) :
  new_computer_cost - (initial_savings + old_computer_value) = 1350 := by
  sorry

end NUMINAMATH_CALUDE_troy_computer_savings_l1715_171524


namespace NUMINAMATH_CALUDE_parallel_lines_alternate_angles_l1715_171543

/-- Two lines in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Angle between two lines -/
def angle (l1 l2 : Line) : ℝ := sorry

/-- Predicate for parallel lines -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Predicate for a line intersecting two other lines -/
def intersects (l : Line) (l1 l2 : Line) : Prop := sorry

/-- Predicate for alternate interior angles -/
def alternate_interior_angles (l : Line) (l1 l2 : Line) (α β : ℝ) : Prop := sorry

/-- Theorem: If two parallel lines are intersected by a third line, 
    then the alternate interior angles are equal -/
theorem parallel_lines_alternate_angles 
  (l1 l2 l : Line) (α β : ℝ) : 
  parallel l1 l2 → 
  intersects l l1 l2 → 
  alternate_interior_angles l l1 l2 α β →
  α = β := by sorry

end NUMINAMATH_CALUDE_parallel_lines_alternate_angles_l1715_171543


namespace NUMINAMATH_CALUDE_equivalent_discount_l1715_171517

theorem equivalent_discount (p : ℝ) (k : ℝ) : 
  (1 - k) * p = (1 - 0.05) * (1 - 0.10) * (1 - 0.15) * p ↔ k = 0.27325 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_discount_l1715_171517


namespace NUMINAMATH_CALUDE_two_apples_per_slice_l1715_171516

/-- The number of apples in each slice of pie -/
def apples_per_slice (total_apples : ℕ) (num_pies : ℕ) (slices_per_pie : ℕ) : ℚ :=
  (total_apples : ℚ) / (num_pies * slices_per_pie)

/-- Theorem stating that there are 2 apples per slice given the problem conditions -/
theorem two_apples_per_slice :
  apples_per_slice (4 * 12) 4 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_apples_per_slice_l1715_171516


namespace NUMINAMATH_CALUDE_complex_division_problem_l1715_171599

theorem complex_division_problem : (2 - I) / (2 + I) = 3/5 - 4/5 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_division_problem_l1715_171599


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_attainable_l1715_171598

theorem min_value_quadratic (a b : ℝ) : a^2 + a*b + b^2 - a - 2*b ≥ -1 := by sorry

theorem min_value_attainable : ∃ (a b : ℝ), a^2 + a*b + b^2 - a - 2*b = -1 := by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_attainable_l1715_171598


namespace NUMINAMATH_CALUDE_continuity_at_four_l1715_171538

/-- Continuity of f(x) = -2x^2 + 9 at x₀ = 4 -/
theorem continuity_at_four :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 4| < δ → |(-2 * x^2 + 9) - (-2 * 4^2 + 9)| < ε := by
sorry

end NUMINAMATH_CALUDE_continuity_at_four_l1715_171538


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l1715_171547

theorem retailer_profit_percentage 
  (cost : ℝ) 
  (discounted_price : ℝ) 
  (discount_rate : ℝ) : 
  cost = 80 → 
  discounted_price = 130 → 
  discount_rate = 0.2 → 
  ((discounted_price / (1 - discount_rate) - cost) / cost) * 100 = 103.125 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_l1715_171547


namespace NUMINAMATH_CALUDE_price_restoration_l1715_171542

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) (h : reduced_price = 0.8 * original_price) :
  reduced_price * 1.25 = original_price := by
  sorry

end NUMINAMATH_CALUDE_price_restoration_l1715_171542


namespace NUMINAMATH_CALUDE_parabola_focus_l1715_171500

/-- The parabola is defined by the equation x = (1/4)y^2 -/
def parabola (x y : ℝ) : Prop := x = (1/4) * y^2

/-- The focus of a parabola is a point (f, 0) such that for any point (x, y) on the parabola,
    the distance from (x, y) to (f, 0) equals the distance from (x, y) to the directrix x = d,
    where d = f + 1 -/
def is_focus (f : ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, parabola x y →
    (x - f)^2 + y^2 = (x - (f + 1))^2

/-- The focus of the parabola x = (1/4)y^2 is at the point (-1, 0) -/
theorem parabola_focus :
  is_focus (-1) parabola := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1715_171500


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_condition_l1715_171597

/-- If the ratios of sine and cosine of angles to their opposite sides are equal in a triangle, then it is an isosceles right triangle. -/
theorem isosceles_right_triangle_condition (A B C : Real) (a b c : Real) :
  (A + B + C = Real.pi) →
  (a > 0) →
  (b > 0) →
  (c > 0) →
  ((Real.sin A) / a = (Real.cos B) / b) →
  ((Real.sin A) / a = (Real.cos C) / c) →
  ((Real.cos B) / b = (Real.cos C) / c) →
  (A = Real.pi / 2 ∧ B = Real.pi / 4 ∧ C = Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_condition_l1715_171597


namespace NUMINAMATH_CALUDE_gcd_consecutive_b_terms_l1715_171565

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem gcd_consecutive_b_terms (n : ℕ) (hn : n ≥ 1) : Nat.gcd (b n) (b (n + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_consecutive_b_terms_l1715_171565


namespace NUMINAMATH_CALUDE_trinomial_square_equality_l1715_171582

theorem trinomial_square_equality : 
  15^2 + 3^2 + 1^2 + 2*(15*3) + 2*(15*1) + 2*(3*1) = (15 + 3 + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_trinomial_square_equality_l1715_171582


namespace NUMINAMATH_CALUDE_perfect_cube_units_digits_l1715_171581

theorem perfect_cube_units_digits : 
  ∃! (s : Finset Nat), 
    (∀ d ∈ s, d < 10) ∧ 
    (∀ n : ℤ, ∃ d ∈ s, (n ^ 3) % 10 = d) ∧
    s.card = 10 :=
by sorry

end NUMINAMATH_CALUDE_perfect_cube_units_digits_l1715_171581


namespace NUMINAMATH_CALUDE_f_deriv_inequality_l1715_171515

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * log x - (1/2) * a * x^2 + (2-a) * x

-- Define the derivative of f
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2 / x - a * x + (2-a)

-- State the theorem
theorem f_deriv_inequality (a : ℝ) (x₁ x₂ x₀ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ < x₂) (h₄ : 0 < x₀)
  (h₅ : f a x₂ - f a x₁ = f_deriv a x₀ * (x₂ - x₁)) :
  f_deriv a ((x₁ + x₂) / 2) < f_deriv a x₀ := by
  sorry

end NUMINAMATH_CALUDE_f_deriv_inequality_l1715_171515


namespace NUMINAMATH_CALUDE_binomial_sum_l1715_171531

theorem binomial_sum (n : ℕ) (h : n > 0) : 
  Nat.choose n 1 + Nat.choose n (n - 2) = (n^2 + n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l1715_171531


namespace NUMINAMATH_CALUDE_f_max_value_l1715_171537

/-- The quadratic function f(x) = -5x^2 + 25x - 1 -/
def f (x : ℝ) : ℝ := -5 * x^2 + 25 * x - 1

/-- The maximum value of f(x) is 129/4 -/
theorem f_max_value : ∃ (M : ℝ), M = 129 / 4 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l1715_171537


namespace NUMINAMATH_CALUDE_linear_dependence_condition_l1715_171548

def vector1 : Fin 2 → ℝ := ![2, 3]
def vector2 (k : ℝ) : Fin 2 → ℝ := ![4, k]

def is_linearly_dependent (v1 v2 : Fin 2 → ℝ) : Prop :=
  ∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ c1 • v1 + c2 • v2 = 0

theorem linear_dependence_condition (k : ℝ) :
  is_linearly_dependent vector1 (vector2 k) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_linear_dependence_condition_l1715_171548


namespace NUMINAMATH_CALUDE_inclined_plane_angle_theorem_l1715_171550

/-- 
Given a system with two blocks connected by a cord over a frictionless pulley,
where one block of mass m is on a frictionless inclined plane and the other block
of mass M is hanging vertically, this theorem proves that if M = 1.5 * m and the
acceleration of the system is g/3, then the angle θ of the inclined plane
satisfies sin θ = 2/3.
-/
theorem inclined_plane_angle_theorem 
  (m : ℝ) 
  (M : ℝ) 
  (g : ℝ) 
  (θ : ℝ) 
  (h_mass : M = 1.5 * m) 
  (h_accel : m * g / 3 = m * g * (1 - Real.sin θ)) : 
  Real.sin θ = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inclined_plane_angle_theorem_l1715_171550


namespace NUMINAMATH_CALUDE_vector_inequality_iff_positive_dot_product_l1715_171535

variable (n : ℕ)
variable (a b : Fin n → ℝ)

theorem vector_inequality_iff_positive_dot_product :
  ‖a + b‖ > ‖a - b‖ ↔ a • b > 0 := by sorry

end NUMINAMATH_CALUDE_vector_inequality_iff_positive_dot_product_l1715_171535


namespace NUMINAMATH_CALUDE_equation_solution_l1715_171520

theorem equation_solution :
  ∃ x : ℝ, x > 0 ∧ 6 * x^(1/3) - 3 * (x / x^(2/3)) = 10 + 2 * x^(1/3) ∧ x = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1715_171520


namespace NUMINAMATH_CALUDE_polygon_properties_l1715_171518

/-- Proves that a polygon with n sides, where the sum of interior angles is 5 times
    the sum of exterior angles, has 12 sides and 54 diagonals. -/
theorem polygon_properties (n : ℕ) : 
  (n - 2) * 180 = 5 * 360 → 
  n = 12 ∧ 
  n * (n - 3) / 2 = 54 := by
sorry

end NUMINAMATH_CALUDE_polygon_properties_l1715_171518


namespace NUMINAMATH_CALUDE_food_percentage_is_ten_percent_l1715_171573

-- Define the total amount spent
variable (T : ℝ)

-- Define the percentage spent on food
variable (F : ℝ)

-- Define the conditions
axiom clothing_percentage : 0.60 * T = T * 0.60
axiom other_items_percentage : 0.30 * T = T * 0.30
axiom food_percentage : F * T = T - (0.60 * T + 0.30 * T)

axiom tax_clothing : 0.04 * (0.60 * T) = 0.024 * T
axiom tax_other_items : 0.08 * (0.30 * T) = 0.024 * T
axiom total_tax : 0.048 * T = 0.024 * T + 0.024 * T

-- Theorem to prove
theorem food_percentage_is_ten_percent : F = 0.10 := by
  sorry

end NUMINAMATH_CALUDE_food_percentage_is_ten_percent_l1715_171573


namespace NUMINAMATH_CALUDE_exactly_two_referees_match_l1715_171596

-- Define the number of referees/seats
def n : ℕ := 5

-- Define the number of referees that should match their seat number
def k : ℕ := 2

-- Define the function to calculate the number of permutations
-- where exactly k out of n elements are in their original positions
def permutations_with_k_fixed (n k : ℕ) : ℕ :=
  (n.choose k) * 2

-- State the theorem
theorem exactly_two_referees_match : permutations_with_k_fixed n k = 20 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_referees_match_l1715_171596


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_sum_of_roots_when_k_is_nine_l1715_171578

theorem quadratic_equation_properties (k : ℝ) :
  let equation := fun x => k * x^2 - 6 * x + 1
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ equation x₁ = 0 ∧ equation x₂ = 0) ↔ (0 < k ∧ k ≤ 9) :=
by sorry

theorem sum_of_roots_when_k_is_nine :
  let equation := fun x => 9 * x^2 - 6 * x + 1
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ equation x₁ = 0 ∧ equation x₂ = 0 ∧ x₁ + x₂ = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_sum_of_roots_when_k_is_nine_l1715_171578


namespace NUMINAMATH_CALUDE_sin_cos_inequality_l1715_171585

theorem sin_cos_inequality (x : ℝ) : 
  2 - Real.sqrt 2 ≤ Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2 ∧ 
  Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2 ≤ 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_inequality_l1715_171585


namespace NUMINAMATH_CALUDE_sequence_bounds_l1715_171511

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => (2 * a n + 1) / (a n + 2)

theorem sequence_bounds : ∀ n : ℕ, n ≥ 1 → 1 < a n ∧ a n < 1 + 1 / 3^n := by
  sorry

end NUMINAMATH_CALUDE_sequence_bounds_l1715_171511


namespace NUMINAMATH_CALUDE_sequence_range_l1715_171551

theorem sequence_range (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, 12 * S n = 4 * a (n + 1) + 5^n - 13) →
  (∀ n : ℕ, S n ≤ S 4) →
  (∀ n : ℕ, S (n + 1) = S n + a (n + 1)) →
  13/48 ≤ a 1 ∧ a 1 ≤ 59/64 := by
  sorry

end NUMINAMATH_CALUDE_sequence_range_l1715_171551


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_six_l1715_171508

theorem largest_three_digit_multiple_of_six :
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 6 = 0 → n ≤ 996 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_six_l1715_171508


namespace NUMINAMATH_CALUDE_first_box_contacts_l1715_171561

/-- Given two boxes of contacts, prove that the first box contains 75 contacts. -/
theorem first_box_contacts (price1 : ℚ) (quantity2 : ℕ) (price2 : ℚ) 
  (chosen_price : ℚ) (chosen_quantity : ℕ) :
  price1 = 25 →
  quantity2 = 99 →
  price2 = 33 →
  chosen_price = 1 →
  chosen_quantity = 3 →
  ∃ quantity1 : ℕ, quantity1 = 75 ∧ 
    price1 / quantity1 = min (price1 / quantity1) (price2 / quantity2) ∧
    price1 / quantity1 = chosen_price / chosen_quantity :=
by sorry


end NUMINAMATH_CALUDE_first_box_contacts_l1715_171561


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_in_binomial_expansion_l1715_171577

theorem coefficient_of_x_cubed_in_binomial_expansion (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ) :
  (∀ x : ℚ, (x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₃ = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_in_binomial_expansion_l1715_171577


namespace NUMINAMATH_CALUDE_mothers_age_l1715_171525

theorem mothers_age (D M : ℕ) 
  (h1 : 2 * D + M = 70)
  (h2 : D + 2 * M = 95) : 
  M = 40 := by
sorry

end NUMINAMATH_CALUDE_mothers_age_l1715_171525


namespace NUMINAMATH_CALUDE_pencil_boxes_filled_l1715_171594

/-- Given 648 pencils and 4 pencils per box, prove that the number of filled boxes is 162. -/
theorem pencil_boxes_filled (total_pencils : ℕ) (pencils_per_box : ℕ) (h1 : total_pencils = 648) (h2 : pencils_per_box = 4) :
  total_pencils / pencils_per_box = 162 := by
  sorry

end NUMINAMATH_CALUDE_pencil_boxes_filled_l1715_171594


namespace NUMINAMATH_CALUDE_xyz_product_l1715_171522

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 3 * y = -9)
  (eq2 : y * z + 3 * z = -9)
  (eq3 : z * x + 3 * x = -9) : 
  x * y * z = -27 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l1715_171522


namespace NUMINAMATH_CALUDE_expression_evaluation_l1715_171570

theorem expression_evaluation (a b c d : ℤ) 
  (ha : a = 10) (hb : b = 15) (hc : c = 3) (hd : d = 2) : 
  (a - (b - c + d)) - ((a - b + d) - c) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1715_171570


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l1715_171569

/-- The perimeter of the shaded region formed by three identical touching circles --/
theorem shaded_region_perimeter (c : ℝ) (n : ℕ) (α : ℝ) : 
  c = 48 → n = 3 → α = 90 → c * (α / 360) * n = 36 := by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l1715_171569


namespace NUMINAMATH_CALUDE_bmw_sales_count_l1715_171549

def total_cars : ℕ := 250
def mercedes_percent : ℚ := 18 / 100
def toyota_percent : ℚ := 25 / 100
def acura_percent : ℚ := 15 / 100

theorem bmw_sales_count :
  (total_cars : ℚ) * (1 - (mercedes_percent + toyota_percent + acura_percent)) = 105 := by
  sorry

end NUMINAMATH_CALUDE_bmw_sales_count_l1715_171549


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1715_171560

/-- Two vectors in R² -/
def Vector2 := Fin 2 → ℝ

/-- Dot product of two vectors in R² -/
def dot_product (v w : Vector2) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

/-- First direction vector -/
def v1 : Vector2 := ![- 6, 2]

/-- Second direction vector -/
def v2 (b : ℝ) : Vector2 := ![b, 3]

/-- Theorem: The value of b that makes the vectors perpendicular is 1 -/
theorem perpendicular_vectors : 
  ∃ b : ℝ, dot_product v1 (v2 b) = 0 ∧ b = 1 := by
sorry


end NUMINAMATH_CALUDE_perpendicular_vectors_l1715_171560


namespace NUMINAMATH_CALUDE_division_problem_l1715_171505

theorem division_problem (n : ℕ) : 
  n / 22 = 12 ∧ n % 22 = 1 → n = 265 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1715_171505


namespace NUMINAMATH_CALUDE_fifth_day_distance_l1715_171507

/-- Represents the daily walking distance sequence -/
def walkingSequence (d : ℕ) : ℕ → ℕ := fun n => 100 + (n - 1) * d

/-- The sum of the first n terms of the walking sequence -/
def walkingSum (d : ℕ) (n : ℕ) : ℕ := n * 100 + n * (n - 1) / 2 * d

theorem fifth_day_distance (d : ℕ) :
  walkingSum d 9 = 1260 → walkingSequence d 5 = 660 :=
by sorry

end NUMINAMATH_CALUDE_fifth_day_distance_l1715_171507


namespace NUMINAMATH_CALUDE_inverse_proportion_points_relation_l1715_171510

theorem inverse_proportion_points_relation :
  ∀ x₁ x₂ x₃ : ℝ,
  (2 = 8 / x₁) →
  (-1 = 8 / x₂) →
  (4 = 8 / x₃) →
  (x₁ > x₃ ∧ x₃ > x₂) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_relation_l1715_171510


namespace NUMINAMATH_CALUDE_circle_equation_l1715_171583

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

-- Define the line x - y = 0
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = p.2}

-- State the theorem
theorem circle_equation :
  -- C passes through the origin
  (0, 0) ∈ C ∧
  -- The center of C is on the positive x-axis
  (∃ a : ℝ, a > 0 ∧ (a, 0) ∈ C) ∧
  -- The chord intercepted by the line x-y=0 on C has a length of 2√2
  (∃ p q : ℝ × ℝ, p ∈ C ∧ q ∈ C ∧ p ∈ L ∧ q ∈ L ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 2) →
  -- Then the equation of C is (x-2)^2 + y^2 = 4
  C = {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4} :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1715_171583


namespace NUMINAMATH_CALUDE_election_winner_percentage_l1715_171532

theorem election_winner_percentage (total_votes : ℕ) (vote_majority : ℕ) : 
  total_votes = 600 → vote_majority = 240 → 
  (70 : ℚ) / 100 * total_votes = (total_votes + vote_majority) / 2 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l1715_171532


namespace NUMINAMATH_CALUDE_q_value_l1715_171592

theorem q_value (t R m q : ℝ) (h : R = t / ((2 + m) ^ q)) :
  q = Real.log (t / R) / Real.log (2 + m) := by
  sorry

end NUMINAMATH_CALUDE_q_value_l1715_171592
