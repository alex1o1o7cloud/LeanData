import Mathlib

namespace NUMINAMATH_CALUDE_vector_subtraction_l3987_398757

/-- Given two vectors OA and OB in ℝ², prove that the vector AB is their difference. -/
theorem vector_subtraction (OA OB : ℝ × ℝ) (h1 : OA = (1, -2)) (h2 : OB = (-3, 1)) :
  OB - OA = (-4, 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3987_398757


namespace NUMINAMATH_CALUDE_museum_travel_distance_l3987_398799

/-- Calculates the total distance traveled to visit two museums on separate days -/
def totalDistanceTraveled (distance1 : ℕ) (distance2 : ℕ) : ℕ :=
  2 * distance1 + 2 * distance2

/-- Proves that visiting museums at 5 and 15 miles results in a total travel of 40 miles -/
theorem museum_travel_distance :
  totalDistanceTraveled 5 15 = 40 := by
  sorry

#eval totalDistanceTraveled 5 15

end NUMINAMATH_CALUDE_museum_travel_distance_l3987_398799


namespace NUMINAMATH_CALUDE_intersection_point_l3987_398738

-- Define the system of equations
def system_solution (a b : ℝ) : ℝ × ℝ := (-1, 3)

-- Define the condition that the system solution satisfies the equations
def system_satisfies (a b : ℝ) : Prop :=
  let (x, y) := system_solution a b
  2 * x + y = b ∧ x - y = a

-- Define the lines
def line1 (x : ℝ) (b : ℝ) : ℝ := -2 * x + b
def line2 (x : ℝ) (a : ℝ) : ℝ := x - a

-- State the theorem
theorem intersection_point (a b : ℝ) (h : system_satisfies a b) :
  let (x, y) := system_solution a b
  line1 x b = y ∧ line2 x a = y := by sorry

end NUMINAMATH_CALUDE_intersection_point_l3987_398738


namespace NUMINAMATH_CALUDE_dot_product_range_l3987_398719

theorem dot_product_range (a b : ℝ × ℝ) : 
  let norm_a := Real.sqrt (a.1^2 + a.2^2)
  let angle := Real.arccos ((b.1 * (a.1 - b.1) + b.2 * (a.2 - b.2)) / 
    (Real.sqrt (b.1^2 + b.2^2) * Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)))
  norm_a = 2 ∧ angle = 2 * Real.pi / 3 →
  2 - 4 * Real.sqrt 3 / 3 ≤ a.1 * b.1 + a.2 * b.2 ∧ 
  a.1 * b.1 + a.2 * b.2 ≤ 2 + 4 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l3987_398719


namespace NUMINAMATH_CALUDE_find_a_min_value_g_l3987_398772

-- Define the function f(x) = |x - a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem 1: Prove that a = 3
theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a x ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) → a = 3 := by sorry

-- Define g(x) = f(2x) + f(x + 2) where f(x) = |x - 3|
def g (x : ℝ) : ℝ := |2*x - 3| + |x + 2 - 3|

-- Theorem 2: Prove that the minimum value of g(x) is 1/2
theorem min_value_g : 
  ∀ x : ℝ, g x ≥ 1/2 ∧ ∃ y : ℝ, g y = 1/2 := by sorry

end NUMINAMATH_CALUDE_find_a_min_value_g_l3987_398772


namespace NUMINAMATH_CALUDE_product_inequality_l3987_398779

theorem product_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  (1 + x) * (1 + y) * (1 + z) ≥ 2 * (1 + (y / x)^(1/3) + (z / y)^(1/3) + (x / z)^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3987_398779


namespace NUMINAMATH_CALUDE_percentage_problem_l3987_398730

theorem percentage_problem (x : ℝ) (h : x = 942.8571428571427) :
  ∃ P : ℝ, (P / 100) * x = (1 / 3) * x + 110 ∧ P = 45 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3987_398730


namespace NUMINAMATH_CALUDE_michaels_brothers_age_multiple_prove_michaels_brothers_age_multiple_l3987_398752

theorem michaels_brothers_age_multiple : ℕ → ℕ → ℕ → Prop :=
  fun michael_age older_brother_age younger_brother_age =>
    let k : ℚ := (older_brother_age - 1 : ℚ) / (michael_age - 1 : ℚ)
    younger_brother_age = 5 ∧
    older_brother_age = 3 * younger_brother_age ∧
    michael_age + older_brother_age + younger_brother_age = 28 ∧
    older_brother_age = k * (michael_age - 1) + 1 →
    k = 2

theorem prove_michaels_brothers_age_multiple :
  ∃ (michael_age older_brother_age younger_brother_age : ℕ),
    michaels_brothers_age_multiple michael_age older_brother_age younger_brother_age :=
by
  sorry

end NUMINAMATH_CALUDE_michaels_brothers_age_multiple_prove_michaels_brothers_age_multiple_l3987_398752


namespace NUMINAMATH_CALUDE_baker_bought_two_boxes_of_baking_soda_l3987_398766

/-- The number of boxes of baking soda bought by the baker -/
def baking_soda_boxes : ℕ :=
  let flour_cost : ℕ := 3 * 3
  let eggs_cost : ℕ := 3 * 10
  let milk_cost : ℕ := 7 * 5
  let total_cost : ℕ := 80
  let baking_soda_unit_cost : ℕ := 3
  (total_cost - (flour_cost + eggs_cost + milk_cost)) / baking_soda_unit_cost

theorem baker_bought_two_boxes_of_baking_soda :
  baking_soda_boxes = 2 := by sorry

end NUMINAMATH_CALUDE_baker_bought_two_boxes_of_baking_soda_l3987_398766


namespace NUMINAMATH_CALUDE_ellipse_properties_l3987_398754

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_minor_axis : b = Real.sqrt 3
  h_eccentricity : Real.sqrt (a^2 - b^2) / a = 1/2

/-- The standard form of the ellipse -/
def standard_form (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2/4 + y^2/3 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1

/-- The maximum area of triangle F₁AB -/
def max_triangle_area (e : Ellipse) : Prop :=
  ∃ (max_area : ℝ),
    max_area = 3 ∧
    ∀ (A B : ℝ × ℝ),
      A ≠ B →
      (∃ (m : ℝ), (A.1 = m * A.2 + 1 ∧ A.1^2/e.a^2 + A.2^2/e.b^2 = 1) ∧
                  (B.1 = m * B.2 + 1 ∧ B.1^2/e.a^2 + B.2^2/e.b^2 = 1)) →
      abs (A.2 - B.2) ≤ max_area

theorem ellipse_properties (e : Ellipse) :
  standard_form e ∧ max_triangle_area e := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3987_398754


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l3987_398704

/-- The sampling interval for systematic sampling -/
def sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The sampling interval for a population of 1200 and sample size of 40 is 30 -/
theorem systematic_sampling_interval :
  sampling_interval 1200 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l3987_398704


namespace NUMINAMATH_CALUDE_wall_volume_is_12_8_l3987_398703

/-- Calculates the volume of a wall given its dimensions --/
def wall_volume (breadth : ℝ) : ℝ :=
  let height := 5 * breadth
  let length := 8 * height
  breadth * height * length

/-- Theorem stating that the volume of the wall with given dimensions is 12.8 cubic meters --/
theorem wall_volume_is_12_8 :
  wall_volume (40 / 100) = 12.8 := by sorry

end NUMINAMATH_CALUDE_wall_volume_is_12_8_l3987_398703


namespace NUMINAMATH_CALUDE_charge_account_interest_l3987_398774

/-- Calculates the total amount owed after one year given an initial charge and simple annual interest rate -/
def total_amount_owed (initial_charge : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_charge * (1 + interest_rate)

/-- Proves that the total amount owed after one year for a $60 charge at 6% simple annual interest is $63.60 -/
theorem charge_account_interest :
  total_amount_owed 60 0.06 = 63.60 := by
  sorry

end NUMINAMATH_CALUDE_charge_account_interest_l3987_398774


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3987_398712

/-- Returns the units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Given two natural numbers m and n, returns true if m has a units digit of 4 -/
def hasUnitsDigitFour (m : ℕ) : Prop := unitsDigit m = 4

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : hasUnitsDigitFour m) :
  unitsDigit n = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3987_398712


namespace NUMINAMATH_CALUDE_distance_to_origin_l3987_398773

theorem distance_to_origin (x y : ℝ) (h1 : y = 16) (h2 : x > 3) 
  (h3 : Real.sqrt ((x - 3)^2 + (y - 6)^2) = 14) : 
  Real.sqrt (x^2 + y^2) = 19 + 12 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l3987_398773


namespace NUMINAMATH_CALUDE_sector_area_l3987_398723

/-- Given a circular sector with central angle 2 radians and arc length 4, the area of the sector is 4. -/
theorem sector_area (θ : Real) (l : Real) (h1 : θ = 2) (h2 : l = 4) :
  (1/2) * (l/θ)^2 * θ = 4 := by sorry

end NUMINAMATH_CALUDE_sector_area_l3987_398723


namespace NUMINAMATH_CALUDE_car_speed_proof_l3987_398750

theorem car_speed_proof (reduced_speed : ℝ) (distance : ℝ) (time : ℝ) (actual_speed : ℝ) : 
  reduced_speed = 5 / 7 * actual_speed →
  distance = 42 →
  time = 42 / 25 →
  reduced_speed = distance / time →
  actual_speed = 35 := by
sorry


end NUMINAMATH_CALUDE_car_speed_proof_l3987_398750


namespace NUMINAMATH_CALUDE_solution_to_equation_l3987_398798

theorem solution_to_equation (x : ℝ) (h : (9 : ℝ) / x^2 = x / 81) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3987_398798


namespace NUMINAMATH_CALUDE_company_layoff_payment_l3987_398732

theorem company_layoff_payment (total_employees : ℕ) (salary : ℕ) (layoff_fraction : ℚ) : 
  total_employees = 450 →
  salary = 2000 →
  layoff_fraction = 1/3 →
  (total_employees : ℚ) * (1 - layoff_fraction) * salary = 600000 := by
sorry

end NUMINAMATH_CALUDE_company_layoff_payment_l3987_398732


namespace NUMINAMATH_CALUDE_range_of_a_l3987_398745

def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem range_of_a : 
  ∃ a : ℝ, p a ∧ ¬(q a) ∧ -1 ≤ a ∧ a < 0 ∧
  ∀ b : ℝ, p b ∧ ¬(q b) → -1 ≤ b ∧ b < 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3987_398745


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_sum_zero_l3987_398709

theorem sum_and_reciprocal_sum_zero (a b c d : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d)
  (h4 : a + b + c + d = 0)
  (h5 : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_sum_zero_l3987_398709


namespace NUMINAMATH_CALUDE_hole_filling_problem_l3987_398740

/-- The amount of additional water needed to fill a hole -/
def additional_water_needed (total_water : ℕ) (initial_water : ℕ) : ℕ :=
  total_water - initial_water

/-- Theorem stating the additional water needed to fill the hole -/
theorem hole_filling_problem (total_water : ℕ) (initial_water : ℕ)
    (h1 : total_water = 823)
    (h2 : initial_water = 676) :
    additional_water_needed total_water initial_water = 147 := by
  sorry

end NUMINAMATH_CALUDE_hole_filling_problem_l3987_398740


namespace NUMINAMATH_CALUDE_real_part_of_z_l3987_398710

theorem real_part_of_z (i : ℂ) (h : i^2 = -1) : Complex.re ((1 + 2*i)^2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3987_398710


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3987_398758

/-- Given a geometric sequence {a_n} with common ratio q, if a₃ = 2S₂ + 1 and a₄ = 2S₃ + 1, then q = 3 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- Definition of sum of geometric sequence
  a 3 = 2 * S 2 + 1 →
  a 4 = 2 * S 3 + 1 →
  q = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3987_398758


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l3987_398770

/-- A circle with equation x^2 + y^2 = m is tangent to the line x - y = √m if and only if m = 0 -/
theorem circle_tangent_to_line (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = m ∧ x - y = Real.sqrt m) ↔ m = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l3987_398770


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l3987_398700

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l3987_398700


namespace NUMINAMATH_CALUDE_root_product_is_root_l3987_398742

/-- Given that a and b are two of the four roots of x^4 + x^3 - 1,
    prove that ab is a root of x^6 + x^4 + x^3 - x^2 - 1 -/
theorem root_product_is_root (a b : ℂ) : 
  (a^4 + a^3 - 1 = 0) → 
  (b^4 + b^3 - 1 = 0) → 
  ((a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_root_product_is_root_l3987_398742


namespace NUMINAMATH_CALUDE_room_width_calculation_l3987_398760

theorem room_width_calculation (length area : ℝ) (h1 : length = 12) (h2 : area = 96) :
  area / length = 8 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l3987_398760


namespace NUMINAMATH_CALUDE_smallest_angle_trig_equation_l3987_398753

theorem smallest_angle_trig_equation :
  let θ := Real.pi / 14
  (∀ φ > 0, φ < θ → Real.sin (3 * φ) * Real.sin (4 * φ) ≠ Real.cos (3 * φ) * Real.cos (4 * φ)) ∧
  Real.sin (3 * θ) * Real.sin (4 * θ) = Real.cos (3 * θ) * Real.cos (4 * θ) := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_trig_equation_l3987_398753


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_five_l3987_398755

theorem sum_of_cubes_of_five : 5^3 + 5^3 + 5^3 + 5^3 = 625 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_five_l3987_398755


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_270_l3987_398744

theorem rectangle_area : ℕ → Prop :=
  fun area =>
    ∃ (square_side : ℕ) (length breadth : ℕ),
      square_side * square_side = 2025 ∧
      length = (2 * square_side) / 5 ∧
      breadth = length / 2 + 5 ∧
      (length + breadth) % 3 = 0 ∧
      length * breadth = area

theorem rectangle_area_is_270 : rectangle_area 270 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_270_l3987_398744


namespace NUMINAMATH_CALUDE_sqrt_two_difference_product_l3987_398715

theorem sqrt_two_difference_product : (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_difference_product_l3987_398715


namespace NUMINAMATH_CALUDE_camel_count_theorem_l3987_398763

/-- Represents the number of humps on a camel -/
inductive CamelType
  | dromedary : CamelType  -- one hump
  | bactrian : CamelType   -- two humps

/-- Calculate the number of humps for a given camel type -/
def humps (c : CamelType) : Nat :=
  match c with
  | .dromedary => 1
  | .bactrian => 2

/-- A group of camels -/
structure CamelGroup where
  dromedaryCount : Nat
  bactrianCount : Nat

/-- Calculate the total number of humps in a camel group -/
def totalHumps (g : CamelGroup) : Nat :=
  g.dromedaryCount * humps CamelType.dromedary + g.bactrianCount * humps CamelType.bactrian

/-- Calculate the total number of feet in a camel group -/
def totalFeet (g : CamelGroup) : Nat :=
  (g.dromedaryCount + g.bactrianCount) * 4

/-- Calculate the total number of camels in a group -/
def totalCamels (g : CamelGroup) : Nat :=
  g.dromedaryCount + g.bactrianCount

theorem camel_count_theorem (g : CamelGroup) :
  totalHumps g = 23 → totalFeet g = 60 → totalCamels g = 15 := by
  sorry

end NUMINAMATH_CALUDE_camel_count_theorem_l3987_398763


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l3987_398706

/-- Represents a triangle with an area and side length -/
structure Triangle where
  area : ℝ
  side : ℝ

/-- Given two similar triangles, proves that the corresponding side of the larger triangle is 15 feet -/
theorem similar_triangles_side_length 
  (t1 t2 : Triangle) 
  (h_area_diff : t1.area - t2.area = 50)
  (h_area_ratio : t1.area / t2.area = 9)
  (h_t2_area_int : ∃ n : ℕ, t2.area = n)
  (h_t2_side : t2.side = 5) :
  t1.side = 15 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l3987_398706


namespace NUMINAMATH_CALUDE_water_speed_calculation_l3987_398747

def swim_speed : ℝ := 4
def distance : ℝ := 8
def time : ℝ := 4

theorem water_speed_calculation (v : ℝ) : 
  (swim_speed - v) * time = distance → v = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l3987_398747


namespace NUMINAMATH_CALUDE_face_card_then_heart_probability_l3987_398786

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of face cards in a standard deck -/
def FaceCards : ℕ := 12

/-- Number of hearts in a standard deck -/
def Hearts : ℕ := 13

/-- Number of face cards that are hearts -/
def FaceHearts : ℕ := 3

/-- Probability of drawing a face card followed by a heart from a standard deck -/
theorem face_card_then_heart_probability :
  (FaceCards / StandardDeck) * (Hearts / (StandardDeck - 1)) = 19 / 210 :=
sorry

end NUMINAMATH_CALUDE_face_card_then_heart_probability_l3987_398786


namespace NUMINAMATH_CALUDE_senate_committee_seating_l3987_398702

/-- The number of unique circular arrangements of n distinguishable objects -/
def circularArrangements (n : ℕ) : ℕ := (n - 1).factorial

theorem senate_committee_seating :
  circularArrangements 10 = 362880 := by
  sorry

end NUMINAMATH_CALUDE_senate_committee_seating_l3987_398702


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3987_398711

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (x - 3) = 5 → x = 28 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3987_398711


namespace NUMINAMATH_CALUDE_inequality_proof_l3987_398780

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a + b + c ≥ (a * (b * c + c + 1)) / (c * a + a + 1) +
              (b * (c * a + a + 1)) / (a * b + b + 1) +
              (c * (a * b + b + 1)) / (b * c + c + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3987_398780


namespace NUMINAMATH_CALUDE_line_y_coordinate_at_x_10_l3987_398764

/-- Given a line passing through points (4, 0) and (-4, -4), 
    prove that the y-coordinate of the point on this line with x-coordinate 10 is 3. -/
theorem line_y_coordinate_at_x_10 (L : Set (ℝ × ℝ)) :
  ((4 : ℝ), 0) ∈ L →
  ((-4 : ℝ), -4) ∈ L →
  ∃ m b : ℝ, ∀ x y : ℝ, (x, y) ∈ L ↔ y = m * x + b →
  (10, 3) ∈ L := by
sorry

end NUMINAMATH_CALUDE_line_y_coordinate_at_x_10_l3987_398764


namespace NUMINAMATH_CALUDE_plane_equation_correct_l3987_398725

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parametric equation of a plane -/
structure ParametricPlane where
  origin : Point3D
  direction1 : Point3D
  direction2 : Point3D

/-- Represents the equation of a plane in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point satisfies a plane equation -/
def satisfiesPlaneEquation (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- The given parametric equation of the plane -/
def givenPlane : ParametricPlane :=
  { origin := { x := 2, y := 4, z := 1 }
  , direction1 := { x := 2, y := 1, z := -3 }
  , direction2 := { x := -3, y := 0, z := 1 }
  }

/-- The equation of the plane to be proven -/
def planeEquation : PlaneEquation :=
  { A := 1, B := 8, C := 3, D := -37 }

theorem plane_equation_correct :
  (∀ s t : ℝ, satisfiesPlaneEquation
    { x := 2 + 2*s - 3*t
    , y := 4 + s
    , z := 1 - 3*s + t
    } planeEquation) ∧
  planeEquation.A > 0 ∧
  Nat.gcd (Nat.gcd (Int.natAbs planeEquation.A) (Int.natAbs planeEquation.B))
          (Nat.gcd (Int.natAbs planeEquation.C) (Int.natAbs planeEquation.D)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l3987_398725


namespace NUMINAMATH_CALUDE_inequality_solution_set_range_of_a_l3987_398761

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2|

-- Theorem for the first part
theorem inequality_solution_set :
  {x : ℝ | 2 * f x < 4 - |x - 1|} = {x : ℝ | -7/3 < x ∧ x < -1} := by sorry

-- Theorem for the second part
theorem range_of_a (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) ↔ -6 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_range_of_a_l3987_398761


namespace NUMINAMATH_CALUDE_science_club_team_selection_l3987_398785

theorem science_club_team_selection (n : ℕ) (k : ℕ) :
  n = 22 → k = 8 → Nat.choose n k = 319770 := by
  sorry

end NUMINAMATH_CALUDE_science_club_team_selection_l3987_398785


namespace NUMINAMATH_CALUDE_vasyas_numbers_l3987_398716

theorem vasyas_numbers (x y : ℝ) :
  x + y = x * y ∧ x + y = x / y → x = 1/2 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l3987_398716


namespace NUMINAMATH_CALUDE_milk_cartons_consumption_l3987_398749

theorem milk_cartons_consumption (total_cartons : ℕ) 
  (younger_sister_fraction : ℚ) (older_sister_fraction : ℚ) :
  total_cartons = 24 →
  younger_sister_fraction = 1 / 8 →
  older_sister_fraction = 3 / 8 →
  (younger_sister_fraction * total_cartons : ℚ) = 3 ∧
  (older_sister_fraction * total_cartons : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_milk_cartons_consumption_l3987_398749


namespace NUMINAMATH_CALUDE_square_side_length_l3987_398720

/-- The perimeter of an equilateral triangle with side length s -/
def triangle_perimeter (s : ℝ) : ℝ := 3 * s

/-- The perimeter of a square with side length s -/
def square_perimeter (s : ℝ) : ℝ := 4 * s

/-- The side length of the equilateral triangle -/
def triangle_side : ℝ := 12

theorem square_side_length :
  ∃ (s : ℝ), s = 9 ∧ square_perimeter s = triangle_perimeter triangle_side :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l3987_398720


namespace NUMINAMATH_CALUDE_equal_probabilities_decreasing_probabilities_l3987_398735

/-- Represents the probability of finding a specific item -/
def item_probability : ℝ := 0.1

/-- Represents the total number of items in the collection -/
def total_items : ℕ := 10

/-- Represents the probability that the second collection is missing exactly k items when the first collection is completed -/
noncomputable def p (k : ℕ) : ℝ := sorry

/-- The probability of missing 1 item equals the probability of missing 2 items -/
theorem equal_probabilities : p 1 = p 2 := by sorry

/-- The probabilities form a strictly decreasing sequence for k from 2 to 10 -/
theorem decreasing_probabilities : ∀ k ∈ Finset.range 9, p (k + 2) > p (k + 3) := by sorry

end NUMINAMATH_CALUDE_equal_probabilities_decreasing_probabilities_l3987_398735


namespace NUMINAMATH_CALUDE_unique_p_q_l3987_398729

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B (p q : ℝ) : Set ℝ := {x | x^2 + p*x + q ≤ 0}

-- State the theorem
theorem unique_p_q : 
  ∃! (p q : ℝ), 
    (A ∪ B p q = Set.univ) ∧ 
    (A ∩ B p q = Set.Icc (-2) (-1)) ∧
    p = -1 ∧ 
    q = -6 := by sorry

end NUMINAMATH_CALUDE_unique_p_q_l3987_398729


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3987_398783

theorem smallest_integer_with_remainders (n : ℕ) : 
  n > 1 ∧ 
  n % 5 = 1 ∧ 
  n % 7 = 1 ∧ 
  n % 8 = 1 ∧ 
  (∀ m : ℕ, m > 1 → m % 5 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) →
  n = 281 ∧ 240 < n ∧ n < 359 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3987_398783


namespace NUMINAMATH_CALUDE_jackson_vacation_savings_l3987_398708

/-- Calculates the total savings for a vacation given the number of months,
    paychecks per month, and amount saved per paycheck. -/
def vacation_savings (months : ℕ) (paychecks_per_month : ℕ) (savings_per_paycheck : ℕ) : ℕ :=
  months * paychecks_per_month * savings_per_paycheck

/-- Proves that Jackson's vacation savings equal $3000 given the problem conditions. -/
theorem jackson_vacation_savings :
  vacation_savings 15 2 100 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_jackson_vacation_savings_l3987_398708


namespace NUMINAMATH_CALUDE_backpack_cost_l3987_398789

/-- Calculates the total cost of personalized backpacks for grandchildren --/
def totalCost (originalPrice taxRates : List ℝ) (discount monogrammingCost coupon : ℝ) : ℝ :=
  let discountedPrice := originalPrice.map (λ p => p * (1 - discount))
  let priceWithMonogram := discountedPrice.map (λ p => p + monogrammingCost)
  let priceWithTax := List.zipWith (λ p r => p * (1 + r)) priceWithMonogram taxRates
  priceWithTax.sum - coupon

/-- Theorem stating the total cost of backpacks for grandchildren --/
theorem backpack_cost :
  let originalPrice := [20, 20, 20, 20, 20]
  let taxRates := [0.06, 0.08, 0.055, 0.0725, 0.04]
  let discount := 0.2
  let monogrammingCost := 12
  let coupon := 5
  totalCost originalPrice taxRates discount monogrammingCost coupon = 143.61 := by
  sorry

#eval totalCost [20, 20, 20, 20, 20] [0.06, 0.08, 0.055, 0.0725, 0.04] 0.2 12 5

end NUMINAMATH_CALUDE_backpack_cost_l3987_398789


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3987_398759

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 - 4*I) / (1 - 2*I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3987_398759


namespace NUMINAMATH_CALUDE_brendans_hourly_wage_l3987_398776

-- Define Brendan's work schedule
def hours_per_week : ℕ := 2 * 8 + 1 * 12

-- Define Brendan's hourly tip rate
def hourly_tips : ℚ := 12

-- Define the fraction of tips reported to IRS
def reported_tips_fraction : ℚ := 1 / 3

-- Define the tax rate
def tax_rate : ℚ := 1 / 5

-- Define the weekly tax amount
def weekly_tax : ℚ := 56

-- Theorem to prove Brendan's hourly wage
theorem brendans_hourly_wage :
  ∃ (hourly_wage : ℚ),
    hourly_wage * hours_per_week +
    reported_tips_fraction * (hourly_tips * hours_per_week) =
    weekly_tax / tax_rate ∧
    hourly_wage = 6 := by
  sorry

end NUMINAMATH_CALUDE_brendans_hourly_wage_l3987_398776


namespace NUMINAMATH_CALUDE_paper_folding_l3987_398795

theorem paper_folding (n : ℕ) : 2^n = 128 → n = 7 := by sorry

end NUMINAMATH_CALUDE_paper_folding_l3987_398795


namespace NUMINAMATH_CALUDE_prob_diff_colors_our_deck_l3987_398765

/-- A deck of cards -/
structure Deck :=
  (red : ℕ)
  (black : ℕ)

/-- The probability of drawing two cards of different colors -/
def prob_diff_colors (d : Deck) : ℚ :=
  if d.red + d.black < 2 then 0
  else (d.red * d.black) / ((d.red + d.black) * (d.red + d.black - 1) / 2)

/-- The deck in our problem -/
def our_deck : Deck := ⟨2, 2⟩

theorem prob_diff_colors_our_deck :
  prob_diff_colors our_deck = 2/3 := by
  sorry

#eval prob_diff_colors our_deck

end NUMINAMATH_CALUDE_prob_diff_colors_our_deck_l3987_398765


namespace NUMINAMATH_CALUDE_difference_from_averages_l3987_398726

theorem difference_from_averages (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) : 
  c - a = 90 := by
sorry

end NUMINAMATH_CALUDE_difference_from_averages_l3987_398726


namespace NUMINAMATH_CALUDE_x_minus_y_equals_eight_l3987_398792

theorem x_minus_y_equals_eight (x y : ℝ) (h1 : 4 = 0.25 * x) (h2 : 4 = 0.50 * y) : x - y = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_eight_l3987_398792


namespace NUMINAMATH_CALUDE_scientific_notation_2310000_l3987_398777

theorem scientific_notation_2310000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 2310000 = a * (10 : ℝ) ^ n ∧ a = 2.31 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_2310000_l3987_398777


namespace NUMINAMATH_CALUDE_garden_plants_l3987_398768

/-- The total number of plants in a rectangular garden -/
def total_plants (rows : ℕ) (columns : ℕ) : ℕ := rows * columns

/-- Theorem: A garden with 52 rows and 15 columns has 780 plants in total -/
theorem garden_plants : total_plants 52 15 = 780 := by
  sorry

end NUMINAMATH_CALUDE_garden_plants_l3987_398768


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l3987_398701

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Part I
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
sorry

-- Part II
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x + g x ≥ 3) → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l3987_398701


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l3987_398713

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_increasing_neg : ∀ x y, x ≤ y → y ≤ 0 → f x ≤ f y

-- Define the solution set
def solution_set := {x : ℝ | f (3 - 2*x) > f 1}

-- State the theorem
theorem solution_set_is_open_interval :
  solution_set = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l3987_398713


namespace NUMINAMATH_CALUDE_tournament_committee_count_l3987_398794

/-- The number of teams in the league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members the host team contributes to the committee -/
def host_contribution : ℕ := 3

/-- The number of members each non-host team contributes to the committee -/
def non_host_contribution : ℕ := 2

/-- The total number of members in the tournament committee -/
def committee_size : ℕ := 11

/-- The number of possible tournament committees -/
def num_committees : ℕ := 172043520

theorem tournament_committee_count :
  (num_teams * (Nat.choose team_size host_contribution) * 
   (Nat.choose team_size non_host_contribution)^(num_teams - 1)) = num_committees := by
  sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l3987_398794


namespace NUMINAMATH_CALUDE_cube_root_function_l3987_398762

/-- Given a function y = kx^(1/3) where y = 4√3 when x = 64, prove that y = 2√3 when x = 8 -/
theorem cube_root_function (k : ℝ) (y : ℝ → ℝ) :
  (∀ x, y x = k * x^(1/3)) →
  y 64 = 4 * Real.sqrt 3 →
  y 8 = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cube_root_function_l3987_398762


namespace NUMINAMATH_CALUDE_complex_distance_sum_l3987_398748

theorem complex_distance_sum (z : ℂ) (h : Complex.abs (z - (3 - 2*I)) = 4) :
  Complex.abs (z + (1 - I))^2 + Complex.abs (z - (7 - 3*I))^2 = 94 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_sum_l3987_398748


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3987_398734

theorem closest_integer_to_cube_root : ∃ (n : ℤ), 
  n = 10 ∧ ∀ (m : ℤ), |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3987_398734


namespace NUMINAMATH_CALUDE_binomial_10_3_l3987_398771

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l3987_398771


namespace NUMINAMATH_CALUDE_trapezoid_area_is_2198_l3987_398707

-- Define the trapezoid
structure Trapezoid where
  leg : ℝ
  diagonal : ℝ
  longer_base : ℝ

-- Define the properties of our specific trapezoid
def my_trapezoid : Trapezoid := {
  leg := 40
  diagonal := 50
  longer_base := 60
}

-- Function to calculate the area of the trapezoid
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

-- Theorem statement
theorem trapezoid_area_is_2198 : 
  trapezoid_area my_trapezoid = 2198 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_2198_l3987_398707


namespace NUMINAMATH_CALUDE_min_value_of_exponential_expression_l3987_398727

theorem min_value_of_exponential_expression :
  ∀ x : ℝ, 16^x - 4^x + 1 ≥ (3:ℝ)/4 ∧ 
  (16^(-(1:ℝ)/2) - 4^(-(1:ℝ)/2) + 1 = (3:ℝ)/4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_expression_l3987_398727


namespace NUMINAMATH_CALUDE_line_through_A_equal_intercepts_line_BC_equation_l3987_398751

-- Define the point A
def A : ℝ × ℝ := (2, 1)

-- Part 1: Line through A with equal intercepts
theorem line_through_A_equal_intercepts :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (a * A.1 + b * A.2 + c = 0) ∧
  (a + b + c = 0) ∧
  (a = 1 ∧ b = 1 ∧ c = -3) := by sorry

-- Part 2: Triangle ABC
theorem line_BC_equation (B C : ℝ × ℝ) :
  -- Given conditions
  (B.1 - B.2 = 0) →  -- B is on the line x - y = 0
  (2 * ((A.1 + B.1) / 2) + ((A.2 + B.2) / 2) - 1 = 0) →  -- CM is on 2x + y - 1 = 0
  (C.1 + C.2 - 3 = 0) →  -- C is on x + y - 3 = 0
  (2 * C.1 + C.2 - 1 = 0) →  -- C is on 2x + y - 1 = 0
  -- Conclusion
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (a * B.1 + b * B.2 + c = 0) ∧
  (a * C.1 + b * C.2 + c = 0) ∧
  (a = 6 ∧ b = 1 ∧ c = 7) := by sorry

end NUMINAMATH_CALUDE_line_through_A_equal_intercepts_line_BC_equation_l3987_398751


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l3987_398778

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : (m^2 + 2^m) % 10 = 3 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l3987_398778


namespace NUMINAMATH_CALUDE_prob_not_all_same_dice_l3987_398746

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability that not all dice show the same number when rolled -/
def prob_not_all_same : ℚ := 1295 / 1296

/-- Theorem stating that the probability of not all dice showing the same number is 1295/1296 -/
theorem prob_not_all_same_dice (h : sides = 6 ∧ num_dice = 5) : 
  prob_not_all_same = 1295 / 1296 := by sorry

end NUMINAMATH_CALUDE_prob_not_all_same_dice_l3987_398746


namespace NUMINAMATH_CALUDE_bottom_right_not_divisible_by_2011_l3987_398722

/-- Represents a cell on the board -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the board configuration -/
structure Board where
  size : Nat
  markedCells : List Cell

/-- Checks if a cell is on the main diagonal -/
def isOnMainDiagonal (c : Cell) : Prop := c.row + c.col = 2011

/-- Checks if a cell is in a corner -/
def isCorner (c : Cell) (n : Nat) : Prop :=
  (c.row = 0 ∧ c.col = 0) ∨ (c.row = 0 ∧ c.col = n - 1) ∨
  (c.row = n - 1 ∧ c.col = 0) ∨ (c.row = n - 1 ∧ c.col = n - 1)

/-- The value in the bottom-right corner of the board -/
def bottomRightValue (b : Board) : Nat :=
  sorry  -- Implementation not required for the statement

theorem bottom_right_not_divisible_by_2011 (b : Board) :
  b.size = 2012 →
  (∀ c ∈ b.markedCells, isOnMainDiagonal c ∧ ¬isCorner c b.size) →
  bottomRightValue b % 2011 = 2 :=
sorry

end NUMINAMATH_CALUDE_bottom_right_not_divisible_by_2011_l3987_398722


namespace NUMINAMATH_CALUDE_prob_eight_rolls_prime_odd_l3987_398796

/-- A function representing the probability of rolling either 3 or 5 on a standard die -/
def prob_prime_odd_roll : ℚ := 1 / 3

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 8

/-- The probability of getting a product of all rolls that is odd and consists only of prime numbers -/
def prob_all_prime_odd : ℚ := (prob_prime_odd_roll) ^ num_rolls

theorem prob_eight_rolls_prime_odd :
  prob_all_prime_odd = 1 / 6561 := by sorry

end NUMINAMATH_CALUDE_prob_eight_rolls_prime_odd_l3987_398796


namespace NUMINAMATH_CALUDE_smallest_result_l3987_398721

def S : Finset Nat := {2, 3, 4, 6, 8, 9}

def process (a b c : Nat) : Nat :=
  max (max ((a + b) * c) ((a + c) * b)) ((b + c) * a)

theorem smallest_result :
  ∃ (a b c : Nat), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  process a b c = 14 ∧
  ∀ (x y z : Nat), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  process x y z ≥ 14 :=
sorry

end NUMINAMATH_CALUDE_smallest_result_l3987_398721


namespace NUMINAMATH_CALUDE_complex_subtraction_simplify_complex_expression_l3987_398743

theorem complex_subtraction (z₁ z₂ : ℂ) : z₁ - z₂ = (z₁.re - z₂.re) + (z₁.im - z₂.im) * I := by sorry

theorem simplify_complex_expression : (3 - 2 * I) - (5 - 2 * I) = -2 := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplify_complex_expression_l3987_398743


namespace NUMINAMATH_CALUDE_odd_fraction_in_multiplication_table_l3987_398717

def table_size : Nat := 16

theorem odd_fraction_in_multiplication_table :
  let total_products := table_size * table_size
  let odd_products := (table_size / 2) * (table_size / 2)
  (odd_products : ℚ) / total_products = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_fraction_in_multiplication_table_l3987_398717


namespace NUMINAMATH_CALUDE_nilpotent_matrix_cube_zero_l3987_398741

theorem nilpotent_matrix_cube_zero
  (A : Matrix (Fin 3) (Fin 3) ℝ)
  (h : A ^ 4 = 0) :
  A ^ 3 = 0 := by
sorry

end NUMINAMATH_CALUDE_nilpotent_matrix_cube_zero_l3987_398741


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3987_398756

/-- The original polynomial expression -/
def original_expression (x : ℝ) : ℝ := 3 * (x^3 - 4*x^2 + 3*x - 1) - 5 * (2*x^3 - x^2 + x + 2)

/-- The simplified polynomial expression -/
def simplified_expression (x : ℝ) : ℝ := -7*x^3 - 7*x^2 + 4*x - 13

/-- Coefficients of the simplified expression -/
def coefficients : List ℝ := [-7, -7, 4, -13]

theorem sum_of_squared_coefficients :
  (original_expression = simplified_expression) →
  (List.sum (List.map (λ x => x^2) coefficients) = 283) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3987_398756


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l3987_398797

theorem shaded_area_fraction (n : ℕ) (h : n = 18) :
  let total_rectangles := n
  let shaded_rectangles := n / 2
  (shaded_rectangles : ℚ) / total_rectangles = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l3987_398797


namespace NUMINAMATH_CALUDE_quadratic_root_square_relation_l3987_398718

theorem quadratic_root_square_relation (c : ℝ) : 
  (c > 0) →
  (∃ x₁ x₂ : ℝ, (8 * x₁^2 - 6 * x₁ + 9 * c^2 = 0) ∧ 
                (8 * x₂^2 - 6 * x₂ + 9 * c^2 = 0) ∧ 
                (x₂ = x₁^2)) →
  (c = 1/3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_square_relation_l3987_398718


namespace NUMINAMATH_CALUDE_mini_van_tank_capacity_l3987_398705

/-- Proves that the capacity of a mini-van's tank is 65 liters given the specified conditions -/
theorem mini_van_tank_capacity :
  let service_cost : ℝ := 2.20
  let fuel_cost_per_liter : ℝ := 0.70
  let num_mini_vans : ℕ := 4
  let num_trucks : ℕ := 2
  let total_cost : ℝ := 395.4
  let truck_tank_ratio : ℝ := 2.2  -- 120% bigger means 2.2 times the size

  ∃ (mini_van_capacity : ℝ),
    mini_van_capacity > 0 ∧
    (service_cost * (num_mini_vans + num_trucks) +
     fuel_cost_per_liter * (num_mini_vans * mini_van_capacity + num_trucks * (truck_tank_ratio * mini_van_capacity)) = total_cost) ∧
    mini_van_capacity = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_mini_van_tank_capacity_l3987_398705


namespace NUMINAMATH_CALUDE_sum_of_cubes_equals_cube_l3987_398733

theorem sum_of_cubes_equals_cube : 57^6 + 95^6 + 109^6 = 228^6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equals_cube_l3987_398733


namespace NUMINAMATH_CALUDE_remaining_doughnuts_theorem_l3987_398737

/-- Represents the types of doughnuts -/
inductive DoughnutType
  | Glazed
  | Chocolate
  | RaspberryFilled

/-- Represents a person who ate doughnuts -/
structure Person where
  glazed : Nat
  chocolate : Nat
  raspberryFilled : Nat

/-- Calculates the remaining doughnuts after consumption -/
def remainingDoughnuts (initial : DoughnutType → Nat) (people : List Person) : DoughnutType → Nat :=
  fun type =>
    initial type - (people.map fun p =>
      match type with
      | DoughnutType.Glazed => p.glazed
      | DoughnutType.Chocolate => p.chocolate
      | DoughnutType.RaspberryFilled => p.raspberryFilled
    ).sum

/-- The main theorem stating the remaining quantities of doughnuts -/
theorem remaining_doughnuts_theorem (initial : DoughnutType → Nat) (people : List Person)
  (h_initial_glazed : initial DoughnutType.Glazed = 10)
  (h_initial_chocolate : initial DoughnutType.Chocolate = 8)
  (h_initial_raspberry : initial DoughnutType.RaspberryFilled = 6)
  (h_people : people = [
    ⟨2, 1, 0⟩, -- Person A
    ⟨1, 0, 0⟩, -- Person B
    ⟨0, 3, 0⟩, -- Person C
    ⟨1, 0, 1⟩, -- Person D
    ⟨0, 0, 1⟩, -- Person E
    ⟨0, 0, 2⟩  -- Person F
  ]) :
  (remainingDoughnuts initial people DoughnutType.Glazed = 6) ∧
  (remainingDoughnuts initial people DoughnutType.Chocolate = 4) ∧
  (remainingDoughnuts initial people DoughnutType.RaspberryFilled = 2) :=
by sorry


end NUMINAMATH_CALUDE_remaining_doughnuts_theorem_l3987_398737


namespace NUMINAMATH_CALUDE_lowest_fifth_score_for_target_average_l3987_398790

def number_of_tests : ℕ := 5
def max_score : ℕ := 100
def target_average : ℕ := 85

def first_three_scores : List ℕ := [76, 94, 87]
def fourth_score : ℕ := 92

def total_needed_score : ℕ := number_of_tests * target_average

theorem lowest_fifth_score_for_target_average :
  ∃ (fifth_score : ℕ),
    fifth_score = total_needed_score - (first_three_scores.sum + fourth_score) ∧
    fifth_score = 76 ∧
    (∀ (x : ℕ), x < fifth_score →
      (first_three_scores.sum + fourth_score + x) / number_of_tests < target_average) :=
by sorry

end NUMINAMATH_CALUDE_lowest_fifth_score_for_target_average_l3987_398790


namespace NUMINAMATH_CALUDE_lindas_outfits_l3987_398731

/-- The number of different outfits that can be created from a given number of skirts, blouses, and shoes. -/
def number_of_outfits (skirts blouses shoes : ℕ) : ℕ :=
  skirts * blouses * shoes

/-- Theorem stating that with 5 skirts, 8 blouses, and 2 pairs of shoes, 80 different outfits can be created. -/
theorem lindas_outfits :
  number_of_outfits 5 8 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_lindas_outfits_l3987_398731


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_points_l3987_398788

theorem smallest_distance_between_complex_points (z w : ℂ) :
  Complex.abs (z - (2 + 4*I)) = 2 →
  Complex.abs (w - (5 + 5*I)) = 4 →
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 10 + 6 ∧
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 + 4*I)) = 2 →
                   Complex.abs (w' - (5 + 5*I)) = 4 →
                   Complex.abs (z' - w') ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_points_l3987_398788


namespace NUMINAMATH_CALUDE_probability_of_non_intersection_l3987_398767

-- Define the circles and their properties
def CircleA : Type := Unit
def CircleB : Type := Unit

-- Define the probability space
def Ω : Type := CircleA × CircleB

-- Define the center distributions
def centerA_distribution : Set ℝ := Set.Icc 0 2
def centerB_distribution : Set ℝ := Set.Icc 0 3

-- Define the radius of each circle
def radiusA : ℝ := 2
def radiusB : ℝ := 1

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the event of non-intersection
def non_intersection : Set Ω := sorry

-- Theorem statement
theorem probability_of_non_intersection :
  P non_intersection = (4 * Real.sqrt 5 - 5) / 3 := by sorry

end NUMINAMATH_CALUDE_probability_of_non_intersection_l3987_398767


namespace NUMINAMATH_CALUDE_triangle_existence_l3987_398784

/-- Given a square area t, segment length 2s, and angle α, 
    this theorem states the existence condition for a triangle 
    with area t, perimeter 2s, and one angle α. -/
theorem triangle_existence 
  (t s : ℝ) (α : Real) 
  (h_t : t > 0) (h_s : s > 0) (h_α : 0 < α ∧ α < π) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = 2 * s ∧
    1/2 * a * b * Real.sin α = t ∧
    ∃ (β γ : Real), 
      β > 0 ∧ γ > 0 ∧
      α + β + γ = π ∧
      a / Real.sin α = b / Real.sin β ∧
      b / Real.sin β = c / Real.sin γ :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l3987_398784


namespace NUMINAMATH_CALUDE_find_first_group_men_l3987_398793

/-- Represents the work rate of one person -/
structure WorkRate where
  rate : ℝ

/-- Represents a group of workers -/
structure WorkerGroup where
  men : ℕ
  women : ℕ

/-- Calculates the total work done by a group -/
def totalWork (m w : WorkRate) (g : WorkerGroup) : ℝ :=
  (g.men : ℝ) * m.rate + (g.women : ℝ) * w.rate

theorem find_first_group_men (m w : WorkRate) : ∃ x : ℕ, 
  totalWork m w ⟨x, 8⟩ = totalWork m w ⟨6, 2⟩ ∧
  2 * totalWork m w ⟨2, 3⟩ = totalWork m w ⟨x, 8⟩ ∧
  x = 3 := by
  sorry

#check find_first_group_men

end NUMINAMATH_CALUDE_find_first_group_men_l3987_398793


namespace NUMINAMATH_CALUDE_proposition_equivalence_l3987_398775

theorem proposition_equivalence (p q : Prop) :
  (¬p → ¬q) ↔ (p → q) := by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l3987_398775


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l3987_398782

/-- Calculates the number of people to be selected from a stratum in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (stratum_size : ℕ) (total_sample_size : ℕ) : ℕ :=
  (total_sample_size * stratum_size) / total_population

/-- The problem statement -/
theorem stratified_sampling_problem (total_population : ℕ) (stratum_size : ℕ) (total_sample_size : ℕ) 
  (h1 : total_population = 360) 
  (h2 : stratum_size = 108) 
  (h3 : total_sample_size = 20) :
  stratified_sample_size total_population stratum_size total_sample_size = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l3987_398782


namespace NUMINAMATH_CALUDE_divisibility_property_l3987_398769

theorem divisibility_property (p m n : ℕ) : 
  Nat.Prime p → 
  p % 2 = 1 →
  m > 1 → 
  n > 0 → 
  Nat.Prime ((m^(p*n) - 1) / (m^n - 1)) → 
  (p * n) ∣ ((p - 1)^n + 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l3987_398769


namespace NUMINAMATH_CALUDE_power_function_property_l3987_398739

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 9 / f 3 = 2) : 
  f (1/9) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_power_function_property_l3987_398739


namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l3987_398736

/-- The number of candy pieces -/
def total_candy : ℕ := 108

/-- Predicate to check if a number divides the total candy evenly -/
def divides_candy (n : ℕ) : Prop := total_candy % n = 0

/-- Predicate to check if a number is a valid student count -/
def valid_student_count (n : ℕ) : Prop :=
  n > 1 ∧ divides_candy n

/-- The set of possible student counts -/
def possible_student_counts : Set ℕ := {12, 36, 54}

/-- Theorem stating that the possible student counts are correct -/
theorem candy_distribution_theorem :
  ∀ n : ℕ, n ∈ possible_student_counts ↔ valid_student_count n :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l3987_398736


namespace NUMINAMATH_CALUDE_Q_equals_N_l3987_398791

-- Define the sets Q and N
def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {x : ℝ | x ≥ 1}

-- Theorem statement
theorem Q_equals_N : Q = N := by sorry

end NUMINAMATH_CALUDE_Q_equals_N_l3987_398791


namespace NUMINAMATH_CALUDE_probability_between_lines_l3987_398781

-- Define the lines
def line_l (x : ℝ) : ℝ := -2 * x + 8
def line_m (x : ℝ) : ℝ := -3 * x + 9

-- Define the region of interest
def region_of_interest (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ y ≤ line_l x ∧ y ≥ line_m x

-- Define the area calculation function
def area_between_lines : ℝ := 2.5

-- Define the total area under line l in the first quadrant
def total_area : ℝ := 16

-- Theorem statement
theorem probability_between_lines :
  (area_between_lines / total_area) = 0.15625 :=
sorry

end NUMINAMATH_CALUDE_probability_between_lines_l3987_398781


namespace NUMINAMATH_CALUDE_cylinder_not_identical_views_l3987_398787

-- Define the basic shapes
structure Shape :=
  (name : String)

-- Define the views
inductive View
  | Top
  | Front
  | Side

-- Define a function to get the shape of a view
def getViewShape (object : Shape) (view : View) : Shape :=
  sorry

-- Define the property of having identical views
def hasIdenticalViews (object : Shape) : Prop :=
  ∀ v1 v2 : View, getViewShape object v1 = getViewShape object v2

-- Define specific shapes
def cylinder : Shape :=
  { name := "Cylinder" }

def cube : Shape :=
  { name := "Cube" }

-- State the theorem
theorem cylinder_not_identical_views :
  ¬(hasIdenticalViews cylinder) ∧ hasIdenticalViews cube :=
sorry

end NUMINAMATH_CALUDE_cylinder_not_identical_views_l3987_398787


namespace NUMINAMATH_CALUDE_sqrt_diff_positive_implies_square_diff_positive_l3987_398728

theorem sqrt_diff_positive_implies_square_diff_positive (a b : ℝ) :
  (∀ (a b : ℝ), Real.sqrt a - Real.sqrt b > 0 → a^2 - b^2 > 0) ∧
  (∃ (a b : ℝ), a^2 - b^2 > 0 ∧ ¬(Real.sqrt a - Real.sqrt b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_diff_positive_implies_square_diff_positive_l3987_398728


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l3987_398724

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define a point on the parabola
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

-- Define the y-axis intersection point
def y_axis_intersection (N : ℝ × ℝ) : Prop :=
  N.1 = 0

-- Define the midpoint condition
def is_midpoint (F M N : ℝ × ℝ) : Prop :=
  M.1 = (F.1 + N.1) / 2 ∧ M.2 = (F.2 + N.2) / 2

-- Main theorem
theorem parabola_focus_distance (M N : ℝ × ℝ) :
  point_on_parabola M →
  y_axis_intersection N →
  is_midpoint focus M N →
  (focus.1 - N.1)^2 + (focus.2 - N.2)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l3987_398724


namespace NUMINAMATH_CALUDE_tangent_product_equals_two_l3987_398714

theorem tangent_product_equals_two :
  (1 + Real.tan (20 * π / 180)) * (1 + Real.tan (25 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_equals_two_l3987_398714
