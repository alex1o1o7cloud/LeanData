import Mathlib

namespace NUMINAMATH_CALUDE_action_figure_fraction_l2506_250675

theorem action_figure_fraction (total_toys dolls : ℕ) : 
  total_toys = 24 → 
  dolls = 18 → 
  (total_toys - dolls : ℚ) / total_toys = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_action_figure_fraction_l2506_250675


namespace NUMINAMATH_CALUDE_angle_value_l2506_250618

theorem angle_value (θ : Real) (h : Real.tan θ = 2) : Real.sin (2 * θ + Real.pi / 2) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_l2506_250618


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l2506_250686

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 309400) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) * 100 = 65 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l2506_250686


namespace NUMINAMATH_CALUDE_integral_f_equals_344_over_15_l2506_250695

-- Define the function to be integrated
def f (x : ℝ) : ℝ := (x^2 + 2*x - 3) * (4*x^2 - x + 1)

-- State the theorem
theorem integral_f_equals_344_over_15 : 
  ∫ x in (0)..(2), f x = 344 / 15 := by sorry

end NUMINAMATH_CALUDE_integral_f_equals_344_over_15_l2506_250695


namespace NUMINAMATH_CALUDE_even_digits_in_base_7_of_789_l2506_250689

def base_7_representation (n : ℕ) : List ℕ :=
  sorry

def count_even_digits (digits : List ℕ) : ℕ :=
  sorry

theorem even_digits_in_base_7_of_789 :
  count_even_digits (base_7_representation 789) = 3 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_in_base_7_of_789_l2506_250689


namespace NUMINAMATH_CALUDE_blocks_per_group_l2506_250673

theorem blocks_per_group (total_blocks : ℕ) (num_groups : ℕ) (blocks_per_group : ℕ) :
  total_blocks = 820 →
  num_groups = 82 →
  total_blocks = num_groups * blocks_per_group →
  blocks_per_group = 10 := by
  sorry

end NUMINAMATH_CALUDE_blocks_per_group_l2506_250673


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l2506_250619

-- Define the triangle's angles
def angle1 : ℝ := 40
def angle2 : ℝ := 70
def angle3 : ℝ := 180 - angle1 - angle2

-- Theorem statement
theorem largest_angle_in_triangle : 
  max angle1 (max angle2 angle3) = 70 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l2506_250619


namespace NUMINAMATH_CALUDE_danny_initial_caps_l2506_250633

/-- The number of bottle caps Danny found at the park -/
def found_caps : ℕ := 7

/-- The total number of bottle caps Danny has after adding the found ones -/
def total_caps : ℕ := 32

/-- The number of bottle caps Danny had before finding the ones at the park -/
def initial_caps : ℕ := total_caps - found_caps

theorem danny_initial_caps : initial_caps = 25 := by
  sorry

end NUMINAMATH_CALUDE_danny_initial_caps_l2506_250633


namespace NUMINAMATH_CALUDE_greatest_integer_absolute_value_l2506_250626

theorem greatest_integer_absolute_value (y : ℤ) : (∀ z : ℤ, |3*z - 4| ≤ 21 → z ≤ y) ↔ y = 8 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_absolute_value_l2506_250626


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2506_250667

theorem polynomial_evaluation (x : ℝ) (h : x = 3) : x^6 - 3*x = 720 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2506_250667


namespace NUMINAMATH_CALUDE_right_triangle_dot_product_l2506_250637

/-- Given a right triangle ABC with ∠ABC = 90°, AB = 4, and BC = 3, 
    prove that the dot product of AC and BC is 9. -/
theorem right_triangle_dot_product (A B C : ℝ × ℝ) : 
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 4^2 →  -- AB = 4
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 3^2 →  -- BC = 3
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0 →  -- ∠ABC = 90°
  ((C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2)) = 9 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_dot_product_l2506_250637


namespace NUMINAMATH_CALUDE_goals_scored_theorem_l2506_250688

/-- The number of goals scored by Bruce and Michael -/
def total_goals (bruce_goals : ℕ) (michael_multiplier : ℕ) : ℕ :=
  bruce_goals + michael_multiplier * bruce_goals

/-- Theorem stating that Bruce and Michael scored 16 goals in total -/
theorem goals_scored_theorem :
  total_goals 4 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_goals_scored_theorem_l2506_250688


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2506_250617

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = (1 : ℝ) / 3 ∧ x₂ = (3 : ℝ) / 2 ∧ 
  (∀ x : ℝ, -6 * x^2 + 11 * x - 3 = 0 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2506_250617


namespace NUMINAMATH_CALUDE_first_solution_percentage_l2506_250669

-- Define the volumes and percentages
def volume_first : ℝ := 40
def volume_second : ℝ := 60
def percent_second : ℝ := 0.7
def percent_final : ℝ := 0.5
def total_volume : ℝ := 100

-- Define the theorem
theorem first_solution_percentage :
  ∃ (percent_first : ℝ),
    volume_first * percent_first + volume_second * percent_second = total_volume * percent_final ∧
    percent_first = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_first_solution_percentage_l2506_250669


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2506_250622

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - 3*x + 2 < 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2506_250622


namespace NUMINAMATH_CALUDE_inverse_of_A_l2506_250691

def A : Matrix (Fin 2) (Fin 2) ℚ := ![![5, -3], ![2, 1]]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ := ![![1/11, 3/11], ![-2/11, 5/11]]

theorem inverse_of_A : A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l2506_250691


namespace NUMINAMATH_CALUDE_cubic_factorization_l2506_250693

theorem cubic_factorization (x : ℝ) : 4 * x^3 - 4 * x^2 + x = x * (2 * x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2506_250693


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2506_250631

theorem contrapositive_equivalence (M : Set α) (a b : α) :
  (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2506_250631


namespace NUMINAMATH_CALUDE_convex_polygon_four_equal_areas_l2506_250656

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  -- We don't need to define the internal structure of the polygon
  -- for this theorem statement

/-- Represents a line in 2D space -/
structure Line where
  -- We don't need to define the internal structure of the line
  -- for this theorem statement

/-- Represents an area measurement -/
def Area : Type := ℝ

/-- Function to calculate the area of a region of a polygon -/
def areaOfRegion (p : ConvexPolygon) (region : Set (ℝ × ℝ)) : Area :=
  sorry -- Implementation not needed for the theorem statement

/-- Two lines are perpendicular -/
def arePerpendicular (l1 l2 : Line) : Prop :=
  sorry -- Definition not needed for the theorem statement

/-- A line divides a polygon into two regions -/
def dividePolygon (p : ConvexPolygon) (l : Line) : (Set (ℝ × ℝ)) × (Set (ℝ × ℝ)) :=
  sorry -- Implementation not needed for the theorem statement

/-- Theorem: Any convex polygon can be divided into four equal areas by two perpendicular lines -/
theorem convex_polygon_four_equal_areas (p : ConvexPolygon) :
  ∃ (l1 l2 : Line),
    arePerpendicular l1 l2 ∧
    let (r1, r2) := dividePolygon p l1
    let (r11, r12) := dividePolygon p l2
    let a1 := areaOfRegion p (r1 ∩ r11)
    let a2 := areaOfRegion p (r1 ∩ r12)
    let a3 := areaOfRegion p (r2 ∩ r11)
    let a4 := areaOfRegion p (r2 ∩ r12)
    a1 = a2 ∧ a2 = a3 ∧ a3 = a4 :=
  sorry

end NUMINAMATH_CALUDE_convex_polygon_four_equal_areas_l2506_250656


namespace NUMINAMATH_CALUDE_point_symmetry_l2506_250636

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other. -/
def symmetric_wrt_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

/-- The point (3,4) -/
def point1 : ℝ × ℝ := (3, 4)

/-- The point (3,-4) -/
def point2 : ℝ × ℝ := (3, -4)

/-- Theorem stating that point1 and point2 are symmetric with respect to the x-axis -/
theorem point_symmetry : symmetric_wrt_x_axis point1 point2 := by sorry

end NUMINAMATH_CALUDE_point_symmetry_l2506_250636


namespace NUMINAMATH_CALUDE_max_volume_container_l2506_250643

/-- Represents the dimensions of a rectangular container --/
structure ContainerDimensions where
  length : Real
  width : Real
  height : Real

/-- Calculates the volume of a rectangular container --/
def volume (d : ContainerDimensions) : Real :=
  d.length * d.width * d.height

/-- Represents the constraints of the problem --/
def containerConstraints (d : ContainerDimensions) : Prop :=
  d.length + d.width + d.height = 7.4 ∧  -- Half of the total bar length
  d.length = d.width + 0.5

/-- The main theorem to prove --/
theorem max_volume_container :
  ∃ (d : ContainerDimensions),
    containerConstraints d ∧
    d.height = 1.2 ∧
    volume d = 1.8 ∧
    (∀ (d' : ContainerDimensions), containerConstraints d' → volume d' ≤ volume d) :=
sorry

end NUMINAMATH_CALUDE_max_volume_container_l2506_250643


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_symmetric_line_example_l2506_250690

/-- Given a line with equation y = mx + b, the line symmetric to it
    with respect to the y-axis has equation y = -mx + b -/
theorem symmetric_line_wrt_y_axis (m b : ℝ) :
  let original_line := fun (x : ℝ) => m * x + b
  let symmetric_line := fun (x : ℝ) => -m * x + b
  ∀ x y : ℝ, symmetric_line x = y ↔ original_line (-x) = y := by sorry

/-- The equation of the line symmetric to y = 2x + 1 with respect to the y-axis is y = -2x + 1 -/
theorem symmetric_line_example :
  let original_line := fun (x : ℝ) => 2 * x + 1
  let symmetric_line := fun (x : ℝ) => -2 * x + 1
  ∀ x y : ℝ, symmetric_line x = y ↔ original_line (-x) = y := by sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_symmetric_line_example_l2506_250690


namespace NUMINAMATH_CALUDE_roll_three_probability_l2506_250699

/-- A fair six-sided die -/
structure FairDie :=
  (sides : Finset Nat)
  (fair : sides = {1, 2, 3, 4, 5, 6})

/-- The event of rolling a 3 -/
def rollThree (d : FairDie) : Finset Nat :=
  {3}

/-- The probability of an event for a fair die -/
def probability (d : FairDie) (event : Finset Nat) : Rat :=
  (event ∩ d.sides).card / d.sides.card

theorem roll_three_probability (d : FairDie) :
  probability d (rollThree d) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_roll_three_probability_l2506_250699


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2506_250644

/-- An isosceles triangle with side lengths 5 and 6 has a perimeter of either 16 or 17 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 5 ∧ b = 6 ∧
  ((a = b ∧ c ≤ a + b) ∨ (a = c ∧ b ≤ a + c) ∨ (b = c ∧ a ≤ b + c)) →
  a + b + c = 16 ∨ a + b + c = 17 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2506_250644


namespace NUMINAMATH_CALUDE_abs_negative_2010_l2506_250614

theorem abs_negative_2010 : |(-2010 : ℤ)| = 2010 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2010_l2506_250614


namespace NUMINAMATH_CALUDE_kelly_weight_percentage_l2506_250658

/-- Proves that Kelly weighs 15% less than Megan given the bridge and children's weight conditions -/
theorem kelly_weight_percentage (bridge_limit : ℝ) (kelly_weight : ℝ) (excess_weight : ℝ) :
  bridge_limit = 100 →
  kelly_weight = 34 →
  excess_weight = 19 →
  ∃ (megan_weight : ℝ),
    megan_weight + kelly_weight + (megan_weight + 5) = bridge_limit + excess_weight ∧
    kelly_weight = megan_weight * (1 - 0.15) :=
by sorry

end NUMINAMATH_CALUDE_kelly_weight_percentage_l2506_250658


namespace NUMINAMATH_CALUDE_sum_of_distances_to_intersection_points_l2506_250623

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y = 3
def C₂ (x y : ℝ) : Prop := y^2 = 2*x

-- Define point P
def P : ℝ × ℝ := (1, 2)

-- Define the distance function
def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Theorem statement
theorem sum_of_distances_to_intersection_points :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₂ A.1 A.2 ∧
    C₁ B.1 B.2 ∧ C₂ B.1 B.2 ∧
    distance P A + distance P B = 6 * Real.sqrt 2 :=
sorry

end

end NUMINAMATH_CALUDE_sum_of_distances_to_intersection_points_l2506_250623


namespace NUMINAMATH_CALUDE_triangle_side_length_l2506_250615

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that when a = 2, b = 3, and angle C = 60°, the length of side c (AB) is √7. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  b = 3 →
  C = Real.pi / 3 →  -- 60° in radians
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2506_250615


namespace NUMINAMATH_CALUDE_columbus_discovery_year_l2506_250678

def is_valid_year (year : ℕ) : Prop :=
  1000 ≤ year ∧ year < 2000 ∧
  (year / 1000 = 1) ∧
  (year / 100 % 10 ≠ year / 10 % 10) ∧
  (year / 100 % 10 ≠ year % 10) ∧
  (year / 10 % 10 ≠ year % 10) ∧
  (year / 1000 + year / 100 % 10 + year / 10 % 10 + year % 10 = 16) ∧
  (year / 10 % 10 + 1 = 5 * (year % 10))

theorem columbus_discovery_year :
  ∀ year : ℕ, is_valid_year year ↔ year = 1492 :=
by sorry

end NUMINAMATH_CALUDE_columbus_discovery_year_l2506_250678


namespace NUMINAMATH_CALUDE_parabola_vertex_l2506_250650

/-- A parabola defined by the equation y^2 + 6y + 2x + 5 = 0 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 + 6*p.2 + 2*p.1 + 5 = 0}

/-- The vertex of a parabola -/
def vertex (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Theorem stating that the vertex of the given parabola is (2, -3) -/
theorem parabola_vertex : vertex Parabola = (2, -3) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2506_250650


namespace NUMINAMATH_CALUDE_kyunghwan_spent_most_l2506_250645

def initial_amount : ℕ := 20000

def seunga_remaining : ℕ := initial_amount / 4
def kyunghwan_remaining : ℕ := initial_amount / 8
def doyun_remaining : ℕ := initial_amount / 5

def seunga_spent : ℕ := initial_amount - seunga_remaining
def kyunghwan_spent : ℕ := initial_amount - kyunghwan_remaining
def doyun_spent : ℕ := initial_amount - doyun_remaining

theorem kyunghwan_spent_most : 
  kyunghwan_spent > seunga_spent ∧ kyunghwan_spent > doyun_spent :=
by sorry

end NUMINAMATH_CALUDE_kyunghwan_spent_most_l2506_250645


namespace NUMINAMATH_CALUDE_comm_add_comm_mul_distrib_l2506_250642

-- Commutative law of addition
theorem comm_add (a b : ℝ) : a + b = b + a := by sorry

-- Commutative law of multiplication
theorem comm_mul (a b : ℝ) : a * b = b * a := by sorry

-- Distributive law of multiplication over addition
theorem distrib (a b c : ℝ) : (a + b) * c = a * c + b * c := by sorry

end NUMINAMATH_CALUDE_comm_add_comm_mul_distrib_l2506_250642


namespace NUMINAMATH_CALUDE_distance_to_sons_house_l2506_250651

/-- The distance to Jennie's son's house -/
def distance : ℝ := 200

/-- The travel time during heavy traffic (in hours) -/
def heavy_traffic_time : ℝ := 5

/-- The travel time with no traffic (in hours) -/
def no_traffic_time : ℝ := 4

/-- The difference in average speed between no traffic and heavy traffic conditions (in mph) -/
def speed_difference : ℝ := 10

/-- Theorem stating that the distance to Jennie's son's house is 200 miles -/
theorem distance_to_sons_house :
  distance = heavy_traffic_time * (distance / heavy_traffic_time) ∧
  distance = no_traffic_time * (distance / no_traffic_time) ∧
  distance / no_traffic_time = distance / heavy_traffic_time + speed_difference :=
by sorry

end NUMINAMATH_CALUDE_distance_to_sons_house_l2506_250651


namespace NUMINAMATH_CALUDE_triangle_properties_l2506_250654

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    this theorem proves properties about sin 2B and the perimeter under specific conditions. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  b = 6 →
  (1/2) * a * c * Real.sin B = 15 →
  (b / Real.sin B) / 2 = 5 →
  (∃ (R : ℝ), R = 5 ∧ b / Real.sin B = 2 * R) →
  Real.sin (2 * B) = 24/25 ∧
  a + b + c = 6 + 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2506_250654


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2506_250666

theorem polynomial_divisibility (C D : ℂ) : 
  (∀ x : ℂ, x^2 - x + 1 = 0 → x^103 + C*x^2 + D*x + 1 = 0) →
  C = -1 ∧ D = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2506_250666


namespace NUMINAMATH_CALUDE_no_two_right_angles_l2506_250696

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180

-- Define a right angle
def is_right_angle (angle : ℝ) : Prop := angle = 90

-- Theorem statement
theorem no_two_right_angles (t : Triangle) : 
  ¬(is_right_angle t.A ∧ is_right_angle t.B) ∧ 
  ¬(is_right_angle t.B ∧ is_right_angle t.C) ∧ 
  ¬(is_right_angle t.A ∧ is_right_angle t.C) :=
sorry

end NUMINAMATH_CALUDE_no_two_right_angles_l2506_250696


namespace NUMINAMATH_CALUDE_compound_interest_principal_l2506_250609

/-- Given a sum of 8820 after 2 years with an interest rate of 5% per annum compounded yearly,
    prove that the initial principal amount was 8000. -/
theorem compound_interest_principal (sum : ℝ) (years : ℕ) (rate : ℝ) (principal : ℝ) 
    (h1 : sum = 8820)
    (h2 : years = 2)
    (h3 : rate = 0.05)
    (h4 : sum = principal * (1 + rate) ^ years) :
  principal = 8000 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_principal_l2506_250609


namespace NUMINAMATH_CALUDE_manicure_total_cost_l2506_250605

theorem manicure_total_cost (manicure_cost : ℝ) (tip_percentage : ℝ) : 
  manicure_cost = 30 →
  tip_percentage = 30 →
  manicure_cost + (tip_percentage / 100) * manicure_cost = 39 := by
  sorry

end NUMINAMATH_CALUDE_manicure_total_cost_l2506_250605


namespace NUMINAMATH_CALUDE_excircle_radius_eq_semiperimeter_implies_right_angle_l2506_250639

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The excircle of a triangle -/
structure Excircle (T : Triangle) where
  center : Point
  radius : ℝ

/-- The semiperimeter of a triangle -/
def semiperimeter (T : Triangle) : ℝ := sorry

/-- A triangle is right-angled -/
def is_right_angled (T : Triangle) : Prop := sorry

/-- Main theorem: If the radius of the excircle equals the semiperimeter, 
    then the triangle is right-angled -/
theorem excircle_radius_eq_semiperimeter_implies_right_angle 
  (T : Triangle) (E : Excircle T) : 
  E.radius = semiperimeter T → is_right_angled T := by sorry

end NUMINAMATH_CALUDE_excircle_radius_eq_semiperimeter_implies_right_angle_l2506_250639


namespace NUMINAMATH_CALUDE_probability_to_reach_3_3_l2506_250604

/-- Represents a point in a 2D grid --/
structure Point where
  x : Int
  y : Int

/-- Represents a direction of movement --/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- Calculates the probability of reaching the target point from the start point
    in the given number of steps or fewer --/
def probability_to_reach (start : Point) (target : Point) (max_steps : Nat) : Rat :=
  sorry

/-- The main theorem stating the probability of reaching (3,3) from (0,0) in 8 or fewer steps --/
theorem probability_to_reach_3_3 :
  probability_to_reach ⟨0, 0⟩ ⟨3, 3⟩ 8 = 55 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_probability_to_reach_3_3_l2506_250604


namespace NUMINAMATH_CALUDE_keystone_arch_angle_theorem_l2506_250665

/-- Represents a keystone arch made of congruent isosceles trapezoids -/
structure KeystoneArch where
  num_trapezoids : ℕ
  trapezoids_congruent : Bool
  trapezoids_isosceles : Bool
  end_trapezoids_horizontal : Bool

/-- Calculate the larger interior angle of a trapezoid in a keystone arch -/
def larger_interior_angle (arch : KeystoneArch) : ℝ :=
  if arch.num_trapezoids = 9 ∧ 
     arch.trapezoids_congruent ∧ 
     arch.trapezoids_isosceles ∧ 
     arch.end_trapezoids_horizontal
  then 100
  else 0

/-- Theorem: The larger interior angle of each trapezoid in a keystone arch 
    with 9 congruent isosceles trapezoids is 100 degrees -/
theorem keystone_arch_angle_theorem (arch : KeystoneArch) :
  arch.num_trapezoids = 9 ∧ 
  arch.trapezoids_congruent ∧ 
  arch.trapezoids_isosceles ∧ 
  arch.end_trapezoids_horizontal →
  larger_interior_angle arch = 100 := by
  sorry

end NUMINAMATH_CALUDE_keystone_arch_angle_theorem_l2506_250665


namespace NUMINAMATH_CALUDE_cost_doubling_l2506_250659

theorem cost_doubling (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) :
  let original_cost := t * b^4
  let new_cost := t * (2*b)^4
  (new_cost / original_cost) * 100 = 1600 := by
sorry

end NUMINAMATH_CALUDE_cost_doubling_l2506_250659


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2506_250613

theorem root_sum_reciprocal (p q r : ℂ) : 
  p^3 - p + 1 = 0 → q^3 - q + 1 = 0 → r^3 - r + 1 = 0 →
  (p ≠ q) → (q ≠ r) → (p ≠ r) →
  1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) = -10 / 13 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2506_250613


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l2506_250632

theorem quadratic_root_sum (b c : ℝ) (h : c ≠ 0) : 
  (c^2 + 2*b*c - 5*c = 0) → (2*b + c = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l2506_250632


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2506_250634

theorem gcd_lcm_product (a b : ℕ) (h : a = 90 ∧ b = 135) : 
  (Nat.gcd a b) * (Nat.lcm a b) = 12150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2506_250634


namespace NUMINAMATH_CALUDE_mean_median_difference_l2506_250668

/-- Represents the frequency distribution of days missed --/
def frequency_distribution : List (Nat × Nat) :=
  [(0, 4), (1, 2), (2, 5), (3, 2), (4, 3), (5, 4)]

/-- Total number of students --/
def total_students : Nat := 20

/-- Calculates the median of the dataset --/
def median (data : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

/-- Calculates the mean of the dataset --/
def mean (data : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

/-- The main theorem to prove --/
theorem mean_median_difference :
  (mean frequency_distribution total_students) - 
  (median frequency_distribution total_students) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l2506_250668


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2506_250692

theorem sphere_volume_equals_surface_area (r : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2506_250692


namespace NUMINAMATH_CALUDE_problem_solution_l2506_250698

def A (a : ℝ) : ℝ := a + 2
def B (a : ℝ) : ℝ := 2 * a^2 - 3 * a + 10
def C (a : ℝ) : ℝ := a^2 + 5 * a - 3

theorem problem_solution :
  (∀ a : ℝ, A a < B a) ∧
  (∀ a : ℝ, (a < -5 ∨ a > 1) → C a > A a) ∧
  (∀ a : ℝ, (a = -5 ∨ a = 1) → C a = A a) ∧
  (∀ a : ℝ, (-5 < a ∧ a < 1) → C a < A a) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2506_250698


namespace NUMINAMATH_CALUDE_line_segment_ratio_l2506_250601

/-- Given seven points O, A, B, C, D, E, F on a straight line, with P on CD,
    prove that OP = (3a + 2d) / 5 when AP:PD = 2:3 and BP:PC = 3:4 -/
theorem line_segment_ratio (a b c d e f : ℝ) :
  let O : ℝ := 0
  let A : ℝ := a
  let B : ℝ := b
  let C : ℝ := c
  let D : ℝ := d
  let E : ℝ := e
  let F : ℝ := f
  ∀ P : ℝ,
    c ≤ P ∧ P ≤ d →
    (A - P) / (P - D) = 2 / 3 →
    (B - P) / (P - C) = 3 / 4 →
    P = (3 * a + 2 * d) / 5 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_ratio_l2506_250601


namespace NUMINAMATH_CALUDE_game_probabilities_and_earnings_l2506_250687

/-- Represents the outcome of drawing balls -/
inductive DrawOutcome
  | AllSameColor
  | DifferentColors

/-- Represents the game setup -/
structure GameSetup :=
  (total_balls : Nat)
  (white_balls : Nat)
  (yellow_balls : Nat)
  (same_color_payout : Int)
  (diff_color_payment : Int)
  (draws_per_day : Nat)
  (days_per_month : Nat)

/-- Calculates the probability of drawing 3 white balls -/
def prob_three_white (setup : GameSetup) : Rat :=
  sorry

/-- Calculates the probability of drawing 2 yellow and 1 white ball -/
def prob_two_yellow_one_white (setup : GameSetup) : Rat :=
  sorry

/-- Calculates the expected monthly earnings -/
def expected_monthly_earnings (setup : GameSetup) : Int :=
  sorry

/-- Main theorem stating the probabilities and expected earnings -/
theorem game_probabilities_and_earnings (setup : GameSetup)
  (h1 : setup.total_balls = 6)
  (h2 : setup.white_balls = 3)
  (h3 : setup.yellow_balls = 3)
  (h4 : setup.same_color_payout = -5)
  (h5 : setup.diff_color_payment = 1)
  (h6 : setup.draws_per_day = 100)
  (h7 : setup.days_per_month = 30) :
  prob_three_white setup = 1/20 ∧
  prob_two_yellow_one_white setup = 1/10 ∧
  expected_monthly_earnings setup = 1200 :=
sorry

end NUMINAMATH_CALUDE_game_probabilities_and_earnings_l2506_250687


namespace NUMINAMATH_CALUDE_james_hats_per_yard_l2506_250641

/-- The number of yards of velvet needed to make one cloak -/
def yards_per_cloak : ℕ := 3

/-- The total number of yards of velvet needed for 6 cloaks and 12 hats -/
def total_yards : ℕ := 21

/-- The number of cloaks made -/
def num_cloaks : ℕ := 6

/-- The number of hats made -/
def num_hats : ℕ := 12

/-- The number of hats James can make out of one yard of velvet -/
def hats_per_yard : ℕ := 4

theorem james_hats_per_yard :
  (total_yards - num_cloaks * yards_per_cloak) * hats_per_yard = num_hats := by
  sorry

end NUMINAMATH_CALUDE_james_hats_per_yard_l2506_250641


namespace NUMINAMATH_CALUDE_y_coordinate_range_of_C_l2506_250630

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = x + 4

-- Define point A
def A : ℝ × ℝ := (0, 2)

-- Define perpendicularity of line segments
def perpendicular (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem y_coordinate_range_of_C 
  (B C : ℝ × ℝ) 
  (hB : parabola B.1 B.2)
  (hC : parabola C.1 C.2)
  (h_perp : perpendicular A B C) :
  C.2 ≤ 0 ∨ C.2 ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_y_coordinate_range_of_C_l2506_250630


namespace NUMINAMATH_CALUDE_infinitely_many_special_numbers_l2506_250628

/-- The false derived function -/
noncomputable def false_derived (n : ℕ) : ℕ :=
  sorry

/-- The set of natural numbers n > 1 such that f(n) = f(n-1) + 1 -/
def special_set : Set ℕ :=
  {n : ℕ | n > 1 ∧ false_derived n = false_derived (n - 1) + 1}

/-- Theorem: There are infinitely many natural numbers n such that f(n) = f(n-1) + 1 -/
theorem infinitely_many_special_numbers : Set.Infinite special_set := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_special_numbers_l2506_250628


namespace NUMINAMATH_CALUDE_expression_value_l2506_250607

theorem expression_value : ∀ a b : ℝ, 
  (a * (1 : ℝ)^4 + b * (1 : ℝ)^2 + 2 = -3) → 
  (a * (-1 : ℝ)^4 + b * (-1 : ℝ)^2 - 2 = -7) := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2506_250607


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l2506_250653

theorem unique_solution_absolute_value_equation :
  ∃! y : ℝ, |y - 25| + |y - 23| = |2*y - 46| :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l2506_250653


namespace NUMINAMATH_CALUDE_complement_B_union_A_when_a_is_1_A_subset_B_iff_a_in_range_l2506_250602

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2*x + a ∧ 2*x + a ≤ 3}
def B : Set ℝ := {x | -1/2 < x ∧ x < 2}

-- Theorem for part (1)
theorem complement_B_union_A_when_a_is_1 :
  (Set.univ \ B) ∪ A 1 = {x | x ≤ 1 ∨ x ≥ 2} := by sorry

-- Theorem for part (2)
theorem A_subset_B_iff_a_in_range (a : ℝ) :
  A a ⊆ B ↔ -1 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_complement_B_union_A_when_a_is_1_A_subset_B_iff_a_in_range_l2506_250602


namespace NUMINAMATH_CALUDE_linear_search_average_comparisons_linear_search_most_efficient_l2506_250646

/-- Represents an array with a specific size and a search function. -/
structure SearchArray (α : Type) where
  size : Nat
  elements : Fin size → α
  search : α → Option (Fin size)

/-- Calculates the average number of comparisons for a linear search. -/
def averageLinearSearchComparisons (n : Nat) : ℚ :=
  (1 + n) / 2

/-- Theorem: The average number of comparisons for a linear search
    on an array of 10,000 elements is 5,000.5 when the element is not present. -/
theorem linear_search_average_comparisons :
  averageLinearSearchComparisons 10000 = 5000.5 := by
  sorry

/-- Theorem: Linear search is the most efficient algorithm for an array
    with partial ordering that doesn't allow for more efficient searches. -/
theorem linear_search_most_efficient (α : Type) (arr : SearchArray α) :
  arr.size = 10000 →
  (∃ (p : α → Prop), ∀ (i j : Fin arr.size), i < j → p (arr.elements i) → p (arr.elements j)) →
  (∀ (search : α → Option (Fin arr.size)), 
    (∀ x, search x = arr.search x) →
    ∃ c, ∀ x, (search x).isNone → c ≥ averageLinearSearchComparisons arr.size) := by
  sorry

end NUMINAMATH_CALUDE_linear_search_average_comparisons_linear_search_most_efficient_l2506_250646


namespace NUMINAMATH_CALUDE_division_problem_l2506_250647

theorem division_problem (L S Q : ℕ) : 
  L - S = 1515 →
  L = 1600 →
  L = Q * S + 15 →
  Q = 18 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2506_250647


namespace NUMINAMATH_CALUDE_dice_roll_probability_l2506_250629

theorem dice_roll_probability (m : ℝ) : 
  (∀ x y : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 → (x^2 : ℝ) + y^2 ≤ m) ↔ 
  72 ≤ m :=
by sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l2506_250629


namespace NUMINAMATH_CALUDE_prize_money_calculation_l2506_250640

theorem prize_money_calculation (total : ℚ) (rica_share : ℚ) (rica_spent : ℚ) (rica_left : ℚ) : 
  rica_share = 3 / 8 * total →
  rica_spent = 1 / 5 * rica_share →
  rica_left = rica_share - rica_spent →
  rica_left = 300 →
  total = 1000 := by
sorry

end NUMINAMATH_CALUDE_prize_money_calculation_l2506_250640


namespace NUMINAMATH_CALUDE_parabola_and_hyperbola_equations_l2506_250624

-- Define the parabola and hyperbola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the conditions
axiom parabola_vertex_origin : ∃ p > 0, parabola p 0 0
axiom axis_of_symmetry : ∃ p > 0, ∀ x, parabola p x 0 → x = 1
axiom intersection_point : ∃ p a b, parabola p (3/2) (Real.sqrt 6) ∧ hyperbola a b (3/2) (Real.sqrt 6)

-- Theorem to prove
theorem parabola_and_hyperbola_equations :
  ∃ p a b, (∀ x y, parabola p x y ↔ y^2 = 4*x) ∧
           (∀ x y, hyperbola a b x y ↔ 4*x^2 - (4/3)*y^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_hyperbola_equations_l2506_250624


namespace NUMINAMATH_CALUDE_sin_540_plus_alpha_implies_cos_alpha_minus_270_l2506_250682

theorem sin_540_plus_alpha_implies_cos_alpha_minus_270
  (α : Real)
  (h : Real.sin (540 * Real.pi / 180 + α) = -4/5) :
  Real.cos (α - 270 * Real.pi / 180) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_540_plus_alpha_implies_cos_alpha_minus_270_l2506_250682


namespace NUMINAMATH_CALUDE_orange_preference_percentage_l2506_250683

/-- The color preferences survey results -/
def color_frequencies : List (String × ℕ) :=
  [("Red", 75), ("Blue", 80), ("Green", 50), ("Yellow", 45), ("Purple", 60), ("Orange", 55)]

/-- The total number of responses in the survey -/
def total_responses : ℕ := (color_frequencies.map (·.2)).sum

/-- Calculate the percentage of respondents who preferred a given color -/
def color_percentage (color : String) : ℚ :=
  match color_frequencies.find? (·.1 = color) with
  | some (_, freq) => (freq : ℚ) / (total_responses : ℚ) * 100
  | none => 0

/-- The theorem stating that the percentage who preferred orange is 15% -/
theorem orange_preference_percentage :
  ⌊color_percentage "Orange"⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_orange_preference_percentage_l2506_250683


namespace NUMINAMATH_CALUDE_dress_cost_calculation_l2506_250611

/-- The cost of a dress in dinars -/
def dress_cost : ℚ := 10/9

/-- The monthly pay in dinars (excluding the dress) -/
def monthly_pay : ℚ := 10

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The number of days worked to earn a dress -/
def days_worked : ℕ := 3

/-- Theorem stating the cost of the dress -/
theorem dress_cost_calculation :
  dress_cost = (monthly_pay + dress_cost) * days_worked / days_in_month :=
by sorry

end NUMINAMATH_CALUDE_dress_cost_calculation_l2506_250611


namespace NUMINAMATH_CALUDE_green_room_fraction_l2506_250625

theorem green_room_fraction (total_rooms : ℕ) (walls_per_room : ℕ) (purple_walls : ℕ) :
  total_rooms = 10 →
  walls_per_room = 8 →
  purple_walls = 32 →
  (total_rooms : ℚ) - (purple_walls / walls_per_room : ℚ) = 3/5 * total_rooms :=
by sorry

end NUMINAMATH_CALUDE_green_room_fraction_l2506_250625


namespace NUMINAMATH_CALUDE_sample_size_calculation_l2506_250621

/-- Given a population of 1000 people and a simple random sampling method where
    the probability of each person being selected is 0.2, prove that the sample size is 200. -/
theorem sample_size_calculation (population : ℕ) (prob : ℝ) (sample_size : ℕ) :
  population = 1000 →
  prob = 0.2 →
  sample_size = population * prob →
  sample_size = 200 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l2506_250621


namespace NUMINAMATH_CALUDE_binders_for_1600_books_20_days_l2506_250684

/-- The number of binders required to bind a certain number of books in a given number of days -/
def binders_required (books : ℕ) (days : ℕ) : ℚ :=
  books / (days * (1400 / (30 * 21)))

theorem binders_for_1600_books_20_days :
  binders_required 1600 20 = 36 :=
sorry

end NUMINAMATH_CALUDE_binders_for_1600_books_20_days_l2506_250684


namespace NUMINAMATH_CALUDE_present_ages_sum_l2506_250677

theorem present_ages_sum (A B S : ℕ) : 
  A + B = S →
  A = 2 * B →
  (A + 3) + (B + 3) = 66 →
  S = 60 := by
sorry

end NUMINAMATH_CALUDE_present_ages_sum_l2506_250677


namespace NUMINAMATH_CALUDE_flag_arrangements_l2506_250635

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def N : ℕ := 858

/-- The number of red flags -/
def red_flags : ℕ := 12

/-- The number of yellow flags -/
def yellow_flags : ℕ := 11

/-- The total number of flags -/
def total_flags : ℕ := red_flags + yellow_flags

/-- Theorem stating that N is the correct number of distinguishable arrangements -/
theorem flag_arrangements :
  N = (red_flags - 1) * (Nat.choose (red_flags + 1) yellow_flags) :=
by sorry

end NUMINAMATH_CALUDE_flag_arrangements_l2506_250635


namespace NUMINAMATH_CALUDE_nancy_football_games_l2506_250685

/-- Nancy's football game attendance problem -/
theorem nancy_football_games 
  (total_games : ℕ) 
  (this_month_games : ℕ) 
  (next_month_games : ℕ) 
  (h1 : total_games = 24)
  (h2 : this_month_games = 9)
  (h3 : next_month_games = 7) :
  total_games - this_month_games - next_month_games = 8 := by
  sorry

end NUMINAMATH_CALUDE_nancy_football_games_l2506_250685


namespace NUMINAMATH_CALUDE_m_range_l2506_250612

theorem m_range (x : ℝ) (h1 : x ∈ Set.Icc 2 4) 
  (h2 : ∃ x ∈ Set.Icc 2 4, x^2 - 2*x + 5 - m < 0) : 
  m ∈ Set.Ioi 5 := by
sorry

end NUMINAMATH_CALUDE_m_range_l2506_250612


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2506_250661

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (3 - 4 * z) = 7 :=
by
  use -23/2
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2506_250661


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2506_250627

/-- Given a hyperbola with the following properties:
    1) Standard form equation: x²/a² - y²/b² = 1
    2) a > 0 and b > 0
    3) Focal length is 2√5
    4) One asymptote is perpendicular to the line 2x + y = 0
    Prove that the equation of the hyperbola is x²/4 - y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_focal : (2 * Real.sqrt 5 : ℝ) = 2 * Real.sqrt (a^2 + b^2))
  (h_asymptote : b / a = 1 / 2) :
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2506_250627


namespace NUMINAMATH_CALUDE_pokemon_card_difference_l2506_250648

theorem pokemon_card_difference : ∀ (orlando_cards : ℕ),
  orlando_cards > 6 →
  6 + orlando_cards + 3 * orlando_cards = 38 →
  orlando_cards - 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_pokemon_card_difference_l2506_250648


namespace NUMINAMATH_CALUDE_smallest_fraction_above_five_sevenths_l2506_250638

theorem smallest_fraction_above_five_sevenths :
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →  -- a is a two-digit number
  10 ≤ b ∧ b ≤ 99 →  -- b is a two-digit number
  (5 : ℚ) / 7 < (a : ℚ) / b →  -- fraction is greater than 5/7
  (68 : ℚ) / 95 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_above_five_sevenths_l2506_250638


namespace NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l2506_250662

theorem largest_whole_number_satisfying_inequality :
  ∀ x : ℤ, (1/4 : ℚ) + (x : ℚ)/8 < 1 → x ≤ 5 ∧
  ((1/4 : ℚ) + (5 : ℚ)/8 < 1 ∧ ∀ y : ℤ, y > 5 → (1/4 : ℚ) + (y : ℚ)/8 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l2506_250662


namespace NUMINAMATH_CALUDE_smallest_n_dividing_m_pow_n_minus_one_l2506_250681

theorem smallest_n_dividing_m_pow_n_minus_one (m : ℕ) (h_m_odd : Odd m) (h_m_gt_1 : m > 1) :
  (∀ n : ℕ, n > 0 → (2^1989 ∣ m^n - 1)) ↔ n ≥ 2^1987 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_dividing_m_pow_n_minus_one_l2506_250681


namespace NUMINAMATH_CALUDE_product_of_numbers_l2506_250672

theorem product_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 22) 
  (sum_squares_eq : x^2 + y^2 = 404) : 
  x * y = 40 := by sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2506_250672


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2506_250610

theorem trigonometric_identity (α : Real) : 
  (2 * Real.tan (π / 4 - α)) / (1 - Real.tan (π / 4 - α)^2) * 
  (Real.sin α * Real.cos α) / (Real.cos α^2 - Real.sin α^2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2506_250610


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2506_250664

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 5) - (x^6 + 2 * x^5 + x^4 - x^3 + 7) =
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2506_250664


namespace NUMINAMATH_CALUDE_computer_price_ratio_l2506_250620

theorem computer_price_ratio (d : ℝ) : 
  d + 0.3 * d = 377 → (d + 377) / d = 2.3 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_ratio_l2506_250620


namespace NUMINAMATH_CALUDE_work_efficiency_ratio_l2506_250660

-- Define the work efficiencies of A and B
def work_efficiency_A : ℚ := 1 / 45
def work_efficiency_B : ℚ := 1 / 22.5

-- Define the combined work time
def combined_work_time : ℚ := 15

-- Define B's individual work time
def B_work_time : ℚ := 22.5

-- Theorem statement
theorem work_efficiency_ratio :
  (work_efficiency_A / work_efficiency_B) = 45 / 2 := by
  sorry

end NUMINAMATH_CALUDE_work_efficiency_ratio_l2506_250660


namespace NUMINAMATH_CALUDE_sqrt_144_divided_by_6_l2506_250603

theorem sqrt_144_divided_by_6 : Real.sqrt 144 / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_144_divided_by_6_l2506_250603


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l2506_250674

/-- Proves that a boat with given characteristics travels 500 km downstream in 5 hours -/
theorem boat_downstream_distance
  (boat_speed : ℝ)
  (upstream_distance : ℝ)
  (upstream_time : ℝ)
  (downstream_time : ℝ)
  (h_boat_speed : boat_speed = 70)
  (h_upstream_distance : upstream_distance = 240)
  (h_upstream_time : upstream_time = 6)
  (h_downstream_time : downstream_time = 5)
  : ∃ (stream_speed : ℝ),
    stream_speed > 0 ∧
    upstream_distance / upstream_time = boat_speed - stream_speed ∧
    downstream_time * (boat_speed + stream_speed) = 500 :=
by sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l2506_250674


namespace NUMINAMATH_CALUDE_last_digit_of_2_to_20_l2506_250679

theorem last_digit_of_2_to_20 (n : ℕ) :
  n ≥ 1 → (2^n : ℕ) % 10 = ((2^(n % 4)) : ℕ) % 10 →
  (2^20 : ℕ) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_2_to_20_l2506_250679


namespace NUMINAMATH_CALUDE_prob_two_twos_in_five_rolls_l2506_250652

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll : ℚ := 1 / 6

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of times we want to roll the specific number -/
def target_rolls : ℕ := 2

/-- The probability of rolling a specific number exactly k times in n rolls of a fair six-sided die -/
def prob_specific_rolls (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (prob_single_roll ^ k) * ((1 - prob_single_roll) ^ (n - k))

theorem prob_two_twos_in_five_rolls :
  prob_specific_rolls num_rolls target_rolls = 625 / 3888 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_twos_in_five_rolls_l2506_250652


namespace NUMINAMATH_CALUDE_employed_males_percentage_proof_l2506_250655

/-- The percentage of the population that is employed -/
def employed_percentage : ℝ := 72

/-- The percentage of employed people who are female -/
def female_employed_percentage : ℝ := 50

/-- The percentage of the population who are employed males -/
def employed_males_percentage : ℝ := 36

theorem employed_males_percentage_proof :
  employed_males_percentage = employed_percentage * (100 - female_employed_percentage) / 100 :=
by sorry

end NUMINAMATH_CALUDE_employed_males_percentage_proof_l2506_250655


namespace NUMINAMATH_CALUDE_chicken_count_l2506_250670

/-- The number of chickens Colten has -/
def colten_chickens : ℕ := 37

/-- The number of chickens Skylar has -/
def skylar_chickens : ℕ := 3 * colten_chickens - 4

/-- The number of chickens Quentin has -/
def quentin_chickens : ℕ := 2 * skylar_chickens + 25

/-- The total number of chickens -/
def total_chickens : ℕ := quentin_chickens + skylar_chickens + colten_chickens

theorem chicken_count : total_chickens = 383 := by
  sorry

end NUMINAMATH_CALUDE_chicken_count_l2506_250670


namespace NUMINAMATH_CALUDE_dilation_result_l2506_250600

/-- Dilation of a complex number -/
def dilation (c k z : ℂ) : ℂ := c + k * (z - c)

theorem dilation_result :
  let c : ℂ := 1 - 3*I
  let k : ℂ := 3
  let z : ℂ := -2 + I
  dilation c k z = -8 + 9*I := by sorry

end NUMINAMATH_CALUDE_dilation_result_l2506_250600


namespace NUMINAMATH_CALUDE_airplane_average_speed_l2506_250608

/-- The average speed of an airplane -/
theorem airplane_average_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 1584) 
  (h2 : time = 24) 
  (h3 : speed = distance / time) : speed = 66 := by
  sorry

end NUMINAMATH_CALUDE_airplane_average_speed_l2506_250608


namespace NUMINAMATH_CALUDE_min_obtuse_triangle_l2506_250671

-- Define the initial angles of the triangle
def α₀ : Real := 60.001
def β₀ : Real := 60
def γ₀ : Real := 59.999

-- Define a function to calculate the nth angle
def angle (n : Nat) (initial : Real) : Real :=
  (-2)^n * (initial - 60) + 60

-- Define a predicate for an obtuse triangle
def is_obtuse (α β γ : Real) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

-- State the theorem
theorem min_obtuse_triangle :
  ∃ (n : Nat), (∀ k < n, ¬is_obtuse (angle k α₀) (angle k β₀) (angle k γ₀)) ∧
               is_obtuse (angle n α₀) (angle n β₀) (angle n γ₀) ∧
               n = 15 := by
  sorry


end NUMINAMATH_CALUDE_min_obtuse_triangle_l2506_250671


namespace NUMINAMATH_CALUDE_total_pencils_donna_marcia_l2506_250694

/-- The number of pencils Cindi bought -/
def cindi_pencils : ℕ := 60

/-- The number of pencils Marcia bought -/
def marcia_pencils : ℕ := 2 * cindi_pencils

/-- The number of pencils Donna bought -/
def donna_pencils : ℕ := 3 * marcia_pencils

/-- Theorem: The total number of pencils bought by Donna and Marcia is 480 -/
theorem total_pencils_donna_marcia : donna_pencils + marcia_pencils = 480 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_donna_marcia_l2506_250694


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2506_250680

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (c x y : ℝ),
    (x^2 / a^2 - y^2 / b^2 = 1) ∧  -- P is on the hyperbola
    (x = c) ∧  -- PF is perpendicular to x-axis
    ((c - b) / (c + b) = 1/3) →  -- ratio of distances to asymptotes
    c^2 / a^2 = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2506_250680


namespace NUMINAMATH_CALUDE_money_sharing_problem_l2506_250676

theorem money_sharing_problem (amanda_ratio ben_ratio carlos_ratio : ℕ) 
  (ben_share : ℕ) (total : ℕ) : 
  amanda_ratio = 3 → 
  ben_ratio = 5 → 
  carlos_ratio = 8 → 
  ben_share = 25 → 
  total = amanda_ratio * (ben_share / ben_ratio) + 
          ben_share + 
          carlos_ratio * (ben_share / ben_ratio) → 
  total = 80 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_problem_l2506_250676


namespace NUMINAMATH_CALUDE_integer_equation_proof_l2506_250657

theorem integer_equation_proof (m n : ℤ) (h : 3 * m * n + 3 * m = n + 2) : 
  3 * m + n = -2 := by
sorry

end NUMINAMATH_CALUDE_integer_equation_proof_l2506_250657


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2506_250697

/-- The sum of roots of two quadratic equations given specific conditions -/
theorem sum_of_roots_quadratic (a b c d p q : ℝ) : a ≠ 0 →
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2020*a*x + c = 0 ∧ y^2 + 2020*a*y + c = 0) →
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + d = 0 ∧ a*y^2 + b*y + d = 0) →
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + p*x + q = 0 ∧ a*y^2 + p*y + q = 0) →
  (∃ w x y z : ℝ, a*w^2 + b*w + d = 0 ∧ a*x^2 + b*x + d = 0 ∧
                  a*y^2 + p*y + q = 0 ∧ a*z^2 + p*z + q = 0 ∧
                  w + x + y + z = 2020) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2506_250697


namespace NUMINAMATH_CALUDE_stocking_price_calculation_l2506_250663

/-- The original price of a stocking before discount -/
def original_price : ℝ := 122.22

/-- The number of stockings ordered -/
def num_stockings : ℕ := 9

/-- The discount rate applied to the stockings -/
def discount_rate : ℝ := 0.1

/-- The cost of monogramming per stocking -/
def monogram_cost : ℝ := 5

/-- The total cost after discount and including monogramming -/
def total_cost : ℝ := 1035

/-- Theorem stating that the calculated original price satisfies the given conditions -/
theorem stocking_price_calculation :
  total_cost = num_stockings * (original_price * (1 - discount_rate) + monogram_cost) :=
by sorry

end NUMINAMATH_CALUDE_stocking_price_calculation_l2506_250663


namespace NUMINAMATH_CALUDE_fraction_decomposition_l2506_250606

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 7) (h2 : x ≠ -2) :
  let f := (2 * x + 4) / (x^2 - 5*x - 14)
  let g := 2 / (x - 7) + 0 / (x + 2)
  (x^2 - 5*x - 14 = (x - 7) * (x + 2)) → f = g :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l2506_250606


namespace NUMINAMATH_CALUDE_greene_nursery_flower_count_l2506_250649

/-- The number of red roses at Greene Nursery -/
def red_roses : ℕ := 1491

/-- The number of yellow carnations at Greene Nursery -/
def yellow_carnations : ℕ := 3025

/-- The number of white roses at Greene Nursery -/
def white_roses : ℕ := 1768

/-- The number of purple tulips at Greene Nursery -/
def purple_tulips : ℕ := 2150

/-- The number of pink daisies at Greene Nursery -/
def pink_daisies : ℕ := 3500

/-- The number of blue irises at Greene Nursery -/
def blue_irises : ℕ := 2973

/-- The number of orange marigolds at Greene Nursery -/
def orange_marigolds : ℕ := 4234

/-- The total number of flowers at Greene Nursery -/
def total_flowers : ℕ := red_roses + yellow_carnations + white_roses + purple_tulips + 
                          pink_daisies + blue_irises + orange_marigolds

theorem greene_nursery_flower_count : total_flowers = 19141 := by
  sorry

end NUMINAMATH_CALUDE_greene_nursery_flower_count_l2506_250649


namespace NUMINAMATH_CALUDE_person_age_puzzle_l2506_250616

theorem person_age_puzzle : ∃ (A : ℕ), 4 * (A + 3) - 4 * (A - 3) = A ∧ A = 24 := by
  sorry

end NUMINAMATH_CALUDE_person_age_puzzle_l2506_250616
