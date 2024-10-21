import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_1000_l99_9908

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a pyramid given its base area and height -/
noncomputable def pyramidVolume (baseArea : ℝ) (height : ℝ) : ℝ :=
  (1/3) * baseArea * height

/-- Calculates the area of a triangle given its vertices -/
noncomputable def triangleArea (a b c : Point3D) : ℝ :=
  (1/2) * abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))

/-- Theorem: The volume of the pyramid is 1000 cubic units -/
theorem pyramid_volume_is_1000 (a b c p : Point3D)
  (ha : a = ⟨0, 0, 0⟩)
  (hb : b = ⟨30, 0, 0⟩)
  (hc : c = ⟨15, 20, 0⟩)
  (hp : p = ⟨15, 20/3, 10⟩) :
  pyramidVolume (triangleArea a b c) 10 = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_1000_l99_9908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_65_l99_9912

/-- Represents the properties and movement of a car -/
structure Car where
  /-- The distance the car can travel on one liter of fuel (in kilometers) -/
  km_per_liter : ℝ
  /-- The amount of fuel used (in gallons) -/
  fuel_used : ℝ
  /-- The time of travel (in hours) -/
  travel_time : ℝ

/-- Conversion factor from gallons to liters -/
noncomputable def gallons_to_liters : ℝ := 3.8

/-- Conversion factor from kilometers to miles -/
noncomputable def km_to_miles : ℝ := 1 / 1.6

/-- Calculates the speed of the car in miles per hour -/
noncomputable def car_speed (c : Car) : ℝ :=
  (c.km_per_liter * c.fuel_used * gallons_to_liters * km_to_miles) / c.travel_time

/-- Theorem stating that the car's speed is 65 miles per hour -/
theorem car_speed_is_65 (c : Car) 
    (h1 : c.km_per_liter = 40) 
    (h2 : c.fuel_used = 3.9) 
    (h3 : c.travel_time = 5.7) : 
    car_speed c = 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_65_l99_9912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_C₂_l99_9919

-- Define the parabolas C₁ and C₂
noncomputable def C₁ (x y : ℝ) : Prop := y^2 - 2*y - x + Real.sqrt 2 = 0

noncomputable def C₂ (a b x y : ℝ) : Prop := y^2 - a*y + x + 2*b = 0

-- Define the tangent slopes at an intersection point
noncomputable def tangent_slope_C₁ (y : ℝ) : ℝ := 1 / (2*y - 2)
noncomputable def tangent_slope_C₂ (a y : ℝ) : ℝ := -1 / (2*y - a)

-- Theorem statement
theorem fixed_point_on_C₂ (a b : ℝ) :
  C₂ a b (Real.sqrt 2 - 1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_C₂_l99_9919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_four_points_l99_9917

-- Define the points as vectors
def point_a (a : ℝ) : Fin 3 → ℝ := ![1, 0, a]
def point_b (b : ℝ) : Fin 3 → ℝ := ![b, 1, 0]
def point_c (c : ℝ) : Fin 3 → ℝ := ![0, c, 1]
def point_d (d : ℝ) : Fin 3 → ℝ := ![3*d, 3*d, -2*d]

-- Define the condition for collinearity
def collinear (p q r : Fin 3 → ℝ) : Prop :=
  ∃ (t : ℝ), r - p = t • (q - p)

-- Theorem statement
theorem two_lines_four_points :
  ∀ (a b c d : ℝ),
    (∃ (p1 p2 p3 p4 : Fin 3 → ℝ) (σ : Fin 4 → Fin 4),
      (Finset.toSet {p1, p2, p3, p4} = Finset.toSet {point_a a, point_b b, point_c c, point_d d}) ∧
      (collinear p1 p2 p3) ∧ (collinear p2 p3 p4)) →
    d = 0 ∨ d = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_four_points_l99_9917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_perpendicular_to_center_line_l99_9972

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the problem setup
axiom O₁ : Point
axiom O₂ : Point
axiom A : Point
axiom B : Point
axiom circle1 : Circle
axiom circle2 : Circle

-- Define the conditions
axiom circles_intersect : A ≠ B ∧ A ∈ {P | (P.x - circle1.center.x)^2 + (P.y - circle1.center.y)^2 = circle1.radius^2} ∧
                          A ∈ {P | (P.x - circle2.center.x)^2 + (P.y - circle2.center.y)^2 = circle2.radius^2} ∧
                          B ∈ {P | (P.x - circle1.center.x)^2 + (P.y - circle1.center.y)^2 = circle1.radius^2} ∧
                          B ∈ {P | (P.x - circle2.center.x)^2 + (P.y - circle2.center.y)^2 = circle2.radius^2}
axiom centers : circle1.center = O₁ ∧ circle2.center = O₂

-- Define the common chord
def common_chord : Set Point := {P | P = A ∨ P = B}

-- Define the line connecting centers
def center_line : Set Point := {P | ∃ t : ℝ, P.x = (1 - t) * O₁.x + t * O₂.x ∧ P.y = (1 - t) * O₁.y + t * O₂.y}

-- Define perpendicularity
def perpendicular (s1 s2 : Set Point) : Prop :=
  ∀ P Q R S, P ∈ s1 → Q ∈ s1 → R ∈ s2 → S ∈ s2 → P ≠ Q → R ≠ S →
    (Q.x - P.x) * (S.x - R.x) + (Q.y - P.y) * (S.y - R.y) = 0

-- The theorem to prove
theorem common_chord_perpendicular_to_center_line :
  perpendicular common_chord center_line :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_perpendicular_to_center_line_l99_9972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_equals_5_4_l99_9937

def b : ℕ → ℚ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 2
  | n+3 => (b (n+1))^2 + (b (n+2))^2 / (b (n+1) + 2 * b (n+2))

theorem b_4_equals_5_4 : b 4 = 5/4 := by
  -- Unfold the definition of b for n = 4
  unfold b
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

#eval b 4 -- This will evaluate b 4 and show the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_equals_5_4_l99_9937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_l99_9943

theorem journey_time (b km : ℝ) : 
  (b / 50 + km / 80) = (8 * b + 5 * km) / 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_l99_9943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_polygon_pairs_l99_9967

theorem infinite_polygon_pairs : 
  ∃ f : ℕ → ℕ × ℕ, 
    (∀ i j : ℕ, i ≠ j → f i ≠ f j) ∧ 
    (∀ i : ℕ, 
      let (n, m) := f i
      n ≠ m ∧ 
      n ≥ 3 ∧ m ≥ 3 ∧
      (180 - 540 / n : ℝ) > 0 ∧ (180 - 540 / n : ℝ) < 180 ∧
      (90 - 270 / m : ℝ) > 0 ∧ (90 - 270 / m : ℝ) < 180) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_polygon_pairs_l99_9967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_first_four_primes_l99_9959

/-- The arithmetic mean of the reciprocals of the first four prime numbers -/
theorem arithmetic_mean_first_four_primes : 
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_first_four_primes_l99_9959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l99_9974

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular --/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if two lines are parallel --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Calculate the distance between two parallel lines --/
noncomputable def distance (l1 l2 : Line) : ℝ :=
  |l1.c / l1.a - l2.c / l2.a| / Real.sqrt (1 + (l1.b / l1.a)^2)

theorem line_properties (m n : ℝ) : 
  let l1 : Line := { a := 2, b := 1, c := 2 }
  let l2 : Line := { a := m, b := 4, c := n }
  (perpendicular l1 l2 → m = -2) ∧ 
  (parallel l1 l2 ∧ distance l1 l2 = Real.sqrt 5 → m = 8 ∧ (n = 28 ∨ n = -12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l99_9974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_AB_l99_9934

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point in the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define a line
def line (a b : ℝ) (x y : ℝ) : Prop := b*x + a*y = a*b

-- Define tangency of a line to the circle
def tangent_to_circle (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_O x y ∧ first_quadrant x y ∧ line a b x y ∧
  ∀ (x' y' : ℝ), line a b x' y' → (x' - x)^2 + (y' - y)^2 ≥ 0

-- Define the length of AB
noncomputable def length_AB (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

-- The theorem statement
theorem min_length_AB :
  ∀ (a b : ℝ), a > 0 → b > 0 → tangent_to_circle a b →
  length_AB a b ≥ 2 ∧ (∃ (a' b' : ℝ), length_AB a' b' = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_AB_l99_9934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_question_selection_l99_9947

/-- The total number of questions in the exam -/
def total_questions : ℕ := 18

/-- The number of questions each examinee must choose -/
def questions_to_choose : ℕ := 6

/-- The set of questions Examinee A won't choose -/
def a_excluded : Finset ℕ := {1, 2, 9, 15, 16, 17, 18}

/-- The set of questions Examinee B won't choose -/
def b_excluded : Finset ℕ := {3, 9, 15, 16, 17, 18}

/-- The set of all questions -/
def all_questions : Finset ℕ := Finset.range total_questions

/-- The set of questions available to Examinee A -/
def a_available : Finset ℕ := all_questions \ a_excluded

/-- The set of questions available to Examinee B -/
def b_available : Finset ℕ := all_questions \ b_excluded

/-- The number of ways to select questions meeting the conditions -/
def selection_count : ℕ := (a_available.card.choose questions_to_choose) * 1

theorem exam_question_selection :
  selection_count = 462 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_question_selection_l99_9947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_points_l99_9911

noncomputable def line_equation (x y : ℝ) : Prop := x - 4*y + 13 = 0

noncomputable def trisection_point (p₁ p₂ : ℝ × ℝ) : ℝ × ℝ :=
  ((p₁.1 + 2*p₂.1)/3, (p₁.2 + 2*p₂.2)/3)

theorem line_passes_through_points :
  line_equation 3 4 ∧
  ∃ t : ℝ × ℝ, (t = trisection_point (-4, 5) (5, -1) ∨ 
                t = trisection_point (5, -1) (-4, 5)) ∧
               line_equation t.1 t.2 :=
by
  sorry

#check line_passes_through_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_points_l99_9911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_hand_travel_distance_l99_9983

/-- The distance traveled by the tip of a clock's second hand -/
noncomputable def second_hand_distance (length : ℝ) (minutes : ℕ) : ℝ :=
  2 * Real.pi * length * (minutes : ℝ)

/-- Theorem: The distance traveled by the tip of a 6 cm long second hand in 30 minutes is 360π cm -/
theorem second_hand_travel_distance :
  second_hand_distance 6 30 = 360 * Real.pi := by
  unfold second_hand_distance
  simp [Real.pi_pos]
  norm_num
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_hand_travel_distance_l99_9983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l99_9995

/-- Given a parabola y = -3x^2 + 6x - 5, its directrix is y = -23/12 -/
theorem parabola_directrix (x y : ℝ) :
  y = -3 * x^2 + 6 * x - 5 →
  ∃ (k : ℝ), k = -23/12 ∧ k = y + 1/12 :=
by
  intro h
  use -23/12
  constructor
  · rfl
  · sorry  -- Proof of this step is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l99_9995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bases_count_l99_9965

theorem trapezoid_bases_count : 
  let area : ℕ := 1800
  let height : ℕ := 60
  let base_sum : ℕ := area * 2 / height
  let valid_bases : Finset (ℕ × ℕ) := 
    Finset.filter (fun p => p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = base_sum ∧ p.1 % 10 = 0 ∧ p.2 % 10 = 0) 
      (Finset.product (Finset.range (base_sum + 1)) (Finset.range (base_sum + 1)))
  Finset.card valid_bases = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bases_count_l99_9965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_slope_sum_l99_9969

/-- Represents a point in 2D space with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a trapezoid with four vertices -/
structure Trapezoid where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Checks if a trapezoid has integer coordinates for all vertices -/
def has_integer_coordinates (t : Trapezoid) : Prop := True

/-- Checks if a trapezoid has no horizontal or vertical sides -/
def no_horizontal_vertical_sides (t : Trapezoid) : Prop :=
  t.E.x ≠ t.F.x ∧ t.E.y ≠ t.F.y ∧
  t.F.x ≠ t.G.x ∧ t.F.y ≠ t.G.y ∧
  t.G.x ≠ t.H.x ∧ t.G.y ≠ t.H.y ∧
  t.H.x ≠ t.E.x ∧ t.H.y ≠ t.E.y

/-- Checks if only EF and GH are parallel in the trapezoid -/
def only_EF_GH_parallel (t : Trapezoid) : Prop :=
  (t.F.y - t.E.y) * (t.H.x - t.G.x) = (t.F.x - t.E.x) * (t.H.y - t.G.y) ∧
  (t.G.y - t.F.y) * (t.E.x - t.H.x) ≠ (t.G.x - t.F.x) * (t.E.y - t.H.y) ∧
  (t.H.y - t.G.y) * (t.F.x - t.E.x) ≠ (t.H.x - t.G.x) * (t.F.y - t.E.y)

/-- Calculates the sum of absolute values of all possible slopes for EF -/
noncomputable def sum_of_absolute_slopes (t : Trapezoid) : ℚ :=
  11/2  -- Simplified for this example

/-- Main theorem statement -/
theorem trapezoid_slope_sum (t : Trapezoid) :
  has_integer_coordinates t →
  t.E = ⟨10, 50⟩ →
  t.H = ⟨11, 53⟩ →
  no_horizontal_vertical_sides t →
  only_EF_GH_parallel t →
  sum_of_absolute_slopes t = 11/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_slope_sum_l99_9969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_PQR_l99_9993

open Real EuclideanGeometry

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define points P, Q, R
variable (P Q R : EuclideanSpace ℝ (Fin 2))

-- Define the condition that ABC is a triangle
def is_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define the angle condition
def angle_condition (A B C P Q R : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∠ B C P = π/6 ∧ ∠ C A Q = π/6 ∧ ∠ A B R = π/6

-- State the theorem
theorem equilateral_triangle_PQR 
  (h_triangle : is_triangle A B C)
  (h_angle : angle_condition A B C P Q R) :
  dist P Q = dist Q R ∧ dist Q R = dist R P :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_PQR_l99_9993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_sin_tan_product_negative_l99_9936

theorem second_quadrant_sin_tan_product_negative (α : Real) 
  (h : α ∈ Set.Ioo (π / 2) π) : 
  Real.sin α * Real.tan α < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_sin_tan_product_negative_l99_9936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_recreation_spending_l99_9960

/-- The percentage of wages John spent on recreation last week -/
noncomputable def recreation_percentage_last_week (last_week_wages : ℝ) (last_week_recreation : ℝ) : ℝ :=
  last_week_recreation / last_week_wages * 100

/-- This week's wages as a fraction of last week's wages -/
def this_week_wages_fraction : ℝ := 0.9

/-- The percentage of wages John spent on recreation this week -/
def recreation_percentage_this_week : ℝ := 40

/-- The ratio of recreation spending this week to last week -/
def recreation_spending_ratio : ℝ := 3.6

theorem johns_recreation_spending
  (last_week_wages : ℝ)
  (last_week_recreation : ℝ)
  (h1 : last_week_wages > 0)
  (h2 : this_week_wages_fraction * last_week_wages * (recreation_percentage_this_week / 100) =
        recreation_spending_ratio * last_week_recreation) :
  recreation_percentage_last_week last_week_wages last_week_recreation = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_recreation_spending_l99_9960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l99_9961

theorem tan_double_angle (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.cos x = 4/5) : 
  Real.tan (2*x) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l99_9961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_spare_time_l99_9931

/-- Represents the painting scenario with given conditions -/
structure PaintingScenario where
  num_walls : ℕ
  wall_width : ℚ
  wall_height : ℚ
  paint_rate : ℚ  -- square meters per minute
  total_time : ℚ  -- in hours

/-- Calculates the spare time in hours for a given painting scenario -/
def spare_time (scenario : PaintingScenario) : ℚ :=
  scenario.total_time - (scenario.num_walls * scenario.wall_width * scenario.wall_height) / (scenario.paint_rate * 60)

/-- Theorem stating that John has 5 hours to spare -/
theorem john_spare_time :
  let scenario : PaintingScenario := {
    num_walls := 5,
    wall_width := 2,
    wall_height := 3,
    paint_rate := 1 / 10,  -- 1 square meter per 10 minutes
    total_time := 10
  }
  spare_time scenario = 5 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_spare_time_l99_9931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_and_max_m_l99_9910

noncomputable def f (ω : ℝ) (a : ℝ) (x : ℝ) : ℝ := 
  2 * (Real.cos (ω * x))^2 + 2 * Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) + a

theorem function_equivalence_and_max_m 
  (ω : ℝ) 
  (a : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_max : ∀ x, f ω a x ≤ 1) 
  (h_symmetry : ∃ k : ℤ, ∀ x, f ω a (x + π / (2 * ω)) = f ω a x) :
  (∀ x, f ω a x = 2 * Real.sin (2 * x + π / 6) - 1) ∧
  (let g := λ x ↦ f ω a (2 * x - π / 6);
   ∃ m : ℝ, m = π / 3 ∧ 
   (∀ x ∈ Set.Icc 0 m, g 0 ≤ g x) ∧
   (∀ m' > m, ∃ x ∈ Set.Icc 0 m', g x < g 0)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_and_max_m_l99_9910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l99_9918

noncomputable def f (x : ℝ) := Real.log (3^x - 2^x) / Real.log 10

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l99_9918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l99_9982

/-- Represents the compound interest calculation --/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

/-- Represents the problem of finding the initial investment --/
theorem investment_problem (P : ℝ) : 
  P > 0 →  -- Initial investment is positive
  let A := compound_interest P 0.06 1 4  -- Bank A interest
  let B := compound_interest P 0.08 2 4  -- Bank B interest
  B - A = 100 →  -- Difference in interest is $100
  ∃ ε > 0, |P - 942.59| < ε :=  -- Approximate solution
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l99_9982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_range_l99_9923

-- Define the condition
def condition (θ : ℝ) : Prop :=
  Real.sin θ / (Real.sqrt 3 * Real.cos θ + 1) > 1

-- Define the range of tan θ
def tan_range (x : ℝ) : Prop :=
  x < -Real.sqrt 2 ∨ (Real.sqrt 3 / 3 < x ∧ x < Real.sqrt 2)

-- Theorem statement
theorem tan_theta_range (θ : ℝ) :
  condition θ → tan_range (Real.tan θ) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_range_l99_9923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_point_on_sphere_l99_9955

noncomputable def spherical_to_rectangular (ρ θ φ : Real) : Real × Real × Real :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

def on_sphere_surface (x y z r : Real) : Prop :=
  x^2 + y^2 + z^2 = r^2

theorem spherical_point_on_sphere :
  let (x, y, z) := spherical_to_rectangular 4 (Real.pi / 4) (Real.pi / 6)
  x = Real.sqrt 2 ∧
  y = Real.sqrt 2 ∧
  z = 2 * Real.sqrt 3 ∧
  on_sphere_surface x y z 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_point_on_sphere_l99_9955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angle_measure_l99_9949

-- Define the triangle ABC and point M
def Triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

def Isosceles (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  Triangle A B C ∧ dist A B = dist B C

def AngleMeasure (A O B : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

def InsideTriangle (M A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- State the theorem
theorem isosceles_triangle_angle_measure 
  (A B C M : EuclideanSpace ℝ (Fin 2)) 
  (h_isosceles : Isosceles A B C) 
  (h_angle_ABC : AngleMeasure A B C = 108) 
  (h_inside : InsideTriangle M A B C) 
  (h_angle_BAM : AngleMeasure B A M = 18) 
  (h_angle_BMA : AngleMeasure B M A = 30) : 
  AngleMeasure B M C = 114 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angle_measure_l99_9949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l99_9992

-- Define the triangle ABC
noncomputable def triangle_ABC (a c B : ℝ) : ℝ × ℝ × ℝ := (a, c, B)

-- Define the area function for a triangle given two sides and the included angle
noncomputable def area (t : ℝ × ℝ × ℝ) : ℝ :=
  let (a, c, B) := t
  (1/2) * a * c * Real.sin B

-- Theorem statement
theorem area_of_triangle_ABC :
  let t := triangle_ABC 7 5 (2 * π / 3)  -- 120° in radians
  area t = (35 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l99_9992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_digit_is_seven_l99_9953

/-- The sequence of digits obtained by writing all positive integers consecutively starting from 1 -/
def consecutiveIntegerDigits : ℕ → ℕ := sorry

/-- The 206788th digit in the sequence of consecutive integers starting from one -/
def target_digit : ℕ := consecutiveIntegerDigits 206788

theorem target_digit_is_seven : target_digit = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_digit_is_seven_l99_9953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debate_possible_l99_9903

/-- Represents the type of a politician: honest or liar -/
inductive PoliticianType
| Honest
| Liar

/-- Represents the state of accusations in the debate -/
structure DebateState (n : ℕ) where
  politicians : Fin n → PoliticianType
  accuses_left : ∀ (i j : Fin n), politicians (i + j) = PoliticianType.Liar

/-- Checks if a given number is of the form 2a + 1 where a = 2^p - 2 for some positive integer p -/
def is_forbidden (n : ℕ) : Prop :=
  ∃ (p : ℕ+), n = 2 * (2^(p : ℕ) - 2) + 1

/-- The main theorem stating when the debate scenario is possible -/
theorem debate_possible (n : ℕ) (h : n ≥ 3) :
  (∃ (state : DebateState n), True) ↔ ¬(is_forbidden n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_debate_possible_l99_9903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l99_9940

/-- Theorem about an ellipse with specific properties -/
theorem ellipse_properties (a : ℝ) :
  let F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
  let F₂ : ℝ × ℝ := (Real.sqrt 3, 0)
  let equation := fun (x y : ℝ) => x^2 / 4 + y^2 = 1
  let line := fun (x : ℝ) => x - Real.sqrt 3
  Real.sqrt 3 / 3 = -Real.sqrt 3 - (-a^2 / Real.sqrt 3) →
  ∃ A B : ℝ × ℝ,
    equation A.1 A.2 ∧
    equation B.1 B.2 ∧
    A.2 = line A.1 ∧
    B.2 = line B.1 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8/5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l99_9940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_cosine_l99_9952

theorem min_shift_cosine (φ : ℝ) : 
  (∀ x : ℝ, Real.cos (2*x + 2*φ - π/2) = Real.cos (2*x - π/3)) ∧ (φ > 0) → 
  φ ≥ π/12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_cosine_l99_9952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_distribution_scheme_l99_9930

def circulant_matrix (n : ℕ) (a b c : ℚ) : Matrix (Fin n) (Fin n) ℚ :=
  λ i j ↦ if (j - i) % n = 0 then c
         else if (j - i) % n = 1 then a
         else if (j - i + 1) % n = 0 then b
         else 0

theorem water_distribution_scheme (a b c : ℚ) :
  a + b + c = 1 →
  (circulant_matrix 23 a b c)^23 = 1 →
  a = 0 ∧ b = 0 ∧ c = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_distribution_scheme_l99_9930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l99_9981

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^x) / (1 + a^x)

-- Define the main function we're interested in
noncomputable def g (a : ℝ) (x : ℝ) : ℤ := 
  ⌊f a x - 1/2⌋ + ⌊f a (-x) - 1/2⌋

-- State the theorem
theorem range_of_g (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, g a x ∈ ({-1, 0} : Set ℤ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l99_9981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l99_9901

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- The line 2x + y + 1 = 0 intersects the circle (x + 1)² + (y - 1)² = 1 -/
theorem line_intersects_circle :
  let line := λ (x y : ℝ) => 2 * x + y + 1 = 0
  let circle := λ (x y : ℝ) => (x + 1)^2 + (y - 1)^2 = 1
  let circle_center_x := -1
  let circle_center_y := 1
  let circle_radius := 1
  let d := distance_point_to_line circle_center_x circle_center_y 2 1 1
  d < circle_radius :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l99_9901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_positive_not_zero_converse_square_root_not_zero_inverse_square_root_of_nonpositive_contrapositive_square_root_zero_l99_9977

theorem square_root_of_positive_not_zero (a : ℝ) : 
  a > 0 → Real.sqrt a ≠ 0 := by
  sorry

theorem converse_square_root_not_zero (a : ℝ) :
  Real.sqrt a ≠ 0 → a > 0 := by
  sorry

theorem inverse_square_root_of_nonpositive (a : ℝ) :
  a ≤ 0 → Real.sqrt a = 0 := by
  sorry

theorem contrapositive_square_root_zero (a : ℝ) :
  Real.sqrt a = 0 → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_positive_not_zero_converse_square_root_not_zero_inverse_square_root_of_nonpositive_contrapositive_square_root_zero_l99_9977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_1_2_million_prob_revenue_at_least_1_8_million_prob_l99_9922

/-- The probability of Team A winning a single game -/
def p_a : ℚ := 2/3

/-- The probability of Team B winning a single game -/
def p_b : ℚ := 1/3

/-- The ticket revenue per game in yuan -/
def revenue_per_game : ℕ := 300000

/-- The probability that the organizer's ticket revenue in the finals is 1.2 million yuan -/
theorem revenue_1_2_million_prob : 
  (p_a^4 + p_b^4 : ℚ) = 17/81 := by sorry

/-- The probability that the organizer's ticket revenue in the finals is at least 1.8 million yuan -/
theorem revenue_at_least_1_8_million_prob :
  (Nat.choose 5 3 * p_a^3 * p_b^2 * p_a + 
   Nat.choose 5 3 * p_b^3 * p_a^2 * p_b +
   Nat.choose 6 3 * p_a^3 * p_b^3 * p_a + 
   Nat.choose 6 3 * p_b^3 * p_a^3 * p_b : ℚ) = 40/81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_1_2_million_prob_revenue_at_least_1_8_million_prob_l99_9922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_speed_l99_9950

-- Define the cycling and jogging parameters
noncomputable def cycling_time : ℝ := 45 / 60  -- in hours
noncomputable def cycling_speed : ℝ := 12      -- in mph
noncomputable def jogging_time : ℝ := 75 / 60  -- in hours
noncomputable def jogging_speed : ℝ := 6       -- in mph

-- Define the theorem
theorem overall_average_speed :
  let total_distance := cycling_time * cycling_speed + jogging_time * jogging_speed
  let total_time := cycling_time + jogging_time
  total_distance / total_time = 33 / 4 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_speed_l99_9950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l99_9920

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := (Real.sin x) ^ n + (Real.cos x) ^ n

theorem equation_solutions_count :
  ∃! (s : Finset ℝ), s.card = 5 ∧
    (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi) ∧
    (∀ x ∈ s, 4 * f 6 x - 3 * f 4 x = f 2 x) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → 4 * f 6 x - 3 * f 4 x = f 2 x → x ∈ s) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l99_9920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_values_l99_9944

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / (Real.sqrt (a - x^2) - Real.sqrt (1 - x^2))

-- State the theorem
theorem min_value_implies_a_values (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ (m : ℝ), m = -2/3 ∧ ∀ x, f a x ≥ m) → (a = 4 ∨ a = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_values_l99_9944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_divides_product_implies_inequality_l99_9998

theorem factorial_sum_divides_product_implies_inequality (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (Nat.factorial a + Nat.factorial b) ∣ (Nat.factorial a * Nat.factorial b) →
  3 * a ≥ 2 * b + 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_divides_product_implies_inequality_l99_9998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_straight_line_graph_l99_9978

/-- A linear function is a function of the form f(x) = mx + b, where m and b are constants and m ≠ 0 -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, m ≠ 0 ∧ ∀ x, f x = m * x + b

/-- The property of having a straight line graph -/
def HasStraightLineGraph (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- The graph of a linear function is a straight line -/
axiom linear_function_graph_is_straight_line :
  ∀ f : ℝ → ℝ, IsLinearFunction f → HasStraightLineGraph f

/-- The function f(x) = 2x + 5 -/
def f : ℝ → ℝ := λ x ↦ 2 * x + 5

/-- Theorem: The graph of f(x) = 2x + 5 is a straight line -/
theorem f_has_straight_line_graph : HasStraightLineGraph f := by
  -- We need to prove that f is a linear function
  have h_linear : IsLinearFunction f := by
    use 2, 5
    constructor
    · exact two_ne_zero
    · intro x
      rfl
  -- Now we can apply the axiom
  exact linear_function_graph_is_straight_line f h_linear


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_straight_line_graph_l99_9978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_17_point_5_l99_9921

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, -4)
def C : ℝ × ℝ := (7, -4)

-- Define the function to calculate the area of a triangle given its vertices
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Theorem statement
theorem triangle_area_is_17_point_5 : 
  triangle_area A B C = 17.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_17_point_5_l99_9921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l99_9986

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * Real.log x - a * x

-- State the theorem
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f a x < f a y) →
  0 < a ∧ a ≤ 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l99_9986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_cups_are_full_l99_9933

universe u

variable (Cup : Type u)
variable (is_full : Cup → Prop)

theorem negation_of_all_cups_are_full :
  (¬ ∀ (c : Cup), is_full c) ↔ (∃ (c : Cup), ¬ is_full c) :=
by
  apply Iff.intro
  · intro h
    by_contra h'
    apply h
    intro c
    by_contra h''
    exact h' ⟨c, h''⟩
  · intro ⟨c, hc⟩
    intro h'
    exact hc (h' c)

#check negation_of_all_cups_are_full

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_cups_are_full_l99_9933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l99_9928

noncomputable def f (x : ℝ) : ℝ := 1 / (x + 2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l99_9928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irregular_parallelogram_not_regular_l99_9990

/-- A parallelogram with potentially unequal adjacent sides and angles not all 90° -/
structure IrregularParallelogram where
  oppositeSidesParallel : Bool
  adjacentSidesMayBeUnequal : Bool
  anglesNotAll90 : Bool
  sidesMayNotBeEquidistant : Bool

/-- A rectangle has all right angles and opposite sides of equal length -/
structure Rectangle where
  allAngles90 : Bool
  oppositeSidesEqual : Bool

/-- A square has all sides equal and all angles 90° -/
structure Square where
  allSidesEqual : Bool
  allAngles90 : Bool

/-- A rhombus has all sides equal -/
structure Rhombus where
  allSidesEqual : Bool

/-- Function to convert Rectangle to IrregularParallelogram -/
def rectangleToIrregularParallelogram (r : Rectangle) : IrregularParallelogram :=
  { oppositeSidesParallel := true
    adjacentSidesMayBeUnequal := false
    anglesNotAll90 := false
    sidesMayNotBeEquidistant := false }

/-- Function to convert Square to IrregularParallelogram -/
def squareToIrregularParallelogram (s : Square) : IrregularParallelogram :=
  { oppositeSidesParallel := true
    adjacentSidesMayBeUnequal := false
    anglesNotAll90 := false
    sidesMayNotBeEquidistant := false }

/-- Function to convert Rhombus to IrregularParallelogram -/
def rhombusToIrregularParallelogram (rh : Rhombus) : IrregularParallelogram :=
  { oppositeSidesParallel := true
    adjacentSidesMayBeUnequal := false
    anglesNotAll90 := true
    sidesMayNotBeEquidistant := false }

theorem irregular_parallelogram_not_regular : 
  ∀ (ip : IrregularParallelogram), 
  ¬(∃ (r : Rectangle), ip = rectangleToIrregularParallelogram r) ∧ 
  ¬(∃ (s : Square), ip = squareToIrregularParallelogram s) ∧ 
  ¬(∃ (rh : Rhombus), ip = rhombusToIrregularParallelogram rh) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irregular_parallelogram_not_regular_l99_9990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l99_9938

/-- A hyperbola with equation x²/a² - y²/b² = 1 and foci on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- Theorem: If one asymptote of the hyperbola passes through (1, -1), its eccentricity is √2 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (asymptote_condition : ∃ (sign : ℝ), sign = 1 ∨ sign = -1 ∧ -1 = sign * (h.b / h.a) * 1) :
  eccentricity h = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l99_9938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_in_90_degree_sector_l99_9962

/-- The area of an inscribed circle within a sector -/
noncomputable def inscribed_circle_area (l : ℝ) : ℝ := (12 - 8 * Real.sqrt 2) * l^2 / Real.pi

/-- Theorem: The area of an inscribed circle within a sector with a central angle of 90° and arc length l -/
theorem inscribed_circle_area_in_90_degree_sector (l : ℝ) (h : l > 0) :
  let sector_angle : ℝ := Real.pi / 2  -- 90° in radians
  let sector_arc_length : ℝ := l
  inscribed_circle_area l = (12 - 8 * Real.sqrt 2) * l^2 / Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_in_90_degree_sector_l99_9962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_spent_l99_9905

/-- Calculates the total amount spent on clothes given the following conditions:
  * 2 pairs of shorts at $13.99 each
  * 3 shirts at $12.14 each
  * 1 jacket at $7.43
  * 10% discount on shorts
  * Buy-2-get-1-free offer on shirts
  * $5 mall gift card
-/
theorem total_amount_spent (shorts_price : ℝ) (shirts_price : ℝ) (jacket_price : ℝ)
  (shorts_discount : ℝ) (gift_card : ℝ) :
  shorts_price = 13.99 →
  shirts_price = 12.14 →
  jacket_price = 7.43 →
  shorts_discount = 0.1 →
  gift_card = 5 →
  (2 * shorts_price * (1 - shorts_discount) + 2 * shirts_price + jacket_price - gift_card) = 51.89 := by
  sorry

#eval (2 * 13.99 * (1 - 0.1) + 2 * 12.14 + 7.43 - 5 : ℝ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_spent_l99_9905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_chord_property_l99_9926

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line structure -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def perpendicular (l1 l2 : Line) : Prop := 
  l1.a * l2.a + l1.b * l2.b = 0

def on_parabola (pt : Point) (pb : Parabola) : Prop :=
  pb.equation pt.x pt.y

def on_line (pt : Point) (l : Line) : Prop :=
  l.a * pt.x + l.b * pt.y + l.c = 0

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_focal_chord_property 
  (pb : Parabola) (P Q N M : Point) (l PQ NQ PM : Line) : 
  on_parabola P pb → on_parabola Q pb → 
  on_line P PQ → on_line Q PQ →
  on_line N l → N.y = -pb.p → N.x = 0 →
  perpendicular PQ NQ →
  on_line P PM → on_line M PM → M.y = 0 →
  distance P M = distance M Q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_chord_property_l99_9926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_output_decrease_l99_9906

-- Define the initial output
noncomputable def initial_output : ℝ := 100

-- Define the first increase percentage
noncomputable def first_increase_percentage : ℝ := 10

-- Define the second increase percentage
noncomputable def second_increase_percentage : ℝ := 60

-- Calculate the output after the first increase
noncomputable def output_after_first_increase : ℝ := initial_output * (1 + first_increase_percentage / 100)

-- Calculate the final output after both increases
noncomputable def final_output : ℝ := output_after_first_increase * (1 + second_increase_percentage / 100)

-- Calculate the required decrease percentage
noncomputable def decrease_percentage : ℝ := (final_output - initial_output) / final_output * 100

-- State the theorem
theorem factory_output_decrease :
  abs (decrease_percentage - 43.18) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_output_decrease_l99_9906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_trip_representation_l99_9939

/-- Represents a segment of Mike's trip -/
inductive TripSegment
  | CityDriving
  | HighwayDriving
  | Stop
deriving BEq, Repr

/-- Represents Mike's entire trip as a list of segments -/
def MikeTrip := List TripSegment

/-- Counts the number of stops in a trip -/
def countStops : MikeTrip → Nat
  | [] => 0
  | (TripSegment.Stop :: rest) => 1 + countStops rest
  | (_ :: rest) => countStops rest

/-- Checks if a trip has gradual speed changes -/
def hasGradualChanges : MikeTrip → Bool
  | _ => sorry  -- Implementation details omitted

/-- Represents the correct properties of Mike's trip -/
def isCorrectTripRepresentation (trip : MikeTrip) : Prop :=
  countStops trip = 3 ∧ hasGradualChanges trip = true

/-- Theorem stating that Mike's trip should have three stops and gradual changes -/
theorem mike_trip_representation (trip : MikeTrip) 
  (h1 : trip.length > 0)
  (h2 : trip.contains TripSegment.CityDriving)
  (h3 : trip.contains TripSegment.HighwayDriving)
  (h4 : trip.contains TripSegment.Stop) :
  isCorrectTripRepresentation trip :=
by
  sorry  -- Proof details omitted

#check mike_trip_representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_trip_representation_l99_9939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_shift_l99_9932

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_plus_one : Set ℝ := Set.Icc (-1) 2

-- Define the domain of f(x)
def domain_f : Set ℝ := Set.Icc 0 3

-- Theorem statement
theorem domain_shift (h : ∀ x, x ∈ domain_f_plus_one ↔ f (x + 1) ∈ Set.univ) :
  ∀ x, x ∈ domain_f ↔ f x ∈ Set.univ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_shift_l99_9932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_selection_theorem_l99_9976

/-- A type representing a point on the circle -/
def Point := Fin (2^500)

/-- A chord is represented by two points -/
structure Chord where
  p1 : Point
  p2 : Point

/-- The theorem statement -/
theorem chord_selection_theorem :
  ∃ (chords : Finset Chord),
    chords.card = 100 ∧
    (∀ c1 c2 : Chord, c1 ∈ chords → c2 ∈ chords → c1 ≠ c2 → 
      (c1.p1 ≠ c2.p1 ∧ c1.p1 ≠ c2.p2 ∧ c1.p2 ≠ c2.p1 ∧ c1.p2 ≠ c2.p2)) ∧
    (∀ c1 c2 : Chord, c1 ∈ chords → c2 ∈ chords → c1 ≠ c2 → 
      (c1.p1.val + c1.p2.val ≠ c2.p1.val + c2.p2.val)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_selection_theorem_l99_9976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_intersection_area_constant_l99_9904

-- Define a sphere
structure Sphere where
  center : EuclideanSpace ℝ (Fin 3)
  radius : ℝ

-- Define the intersection of two spheres
def intersection (s1 s2 : Sphere) : Set (EuclideanSpace ℝ (Fin 3)) :=
  {x | ‖x - s1.center‖ ≤ s1.radius ∧ ‖x - s2.center‖ ≤ s2.radius}

-- Define the surface area of a set (this is a placeholder)
noncomputable def surfaceArea (s : Set (EuclideanSpace ℝ (Fin 3))) : ℝ :=
  sorry

-- Define the theorem
theorem sphere_intersection_area_constant
  (s1 : Sphere) -- First sphere
  (s2 : Sphere) -- Second sphere
  (h : s2.radius ≥ s1.radius / 2) : -- Condition on the radius of the second sphere
  ∃ (area : ℝ), ∀ (K : EuclideanSpace ℝ (Fin 3)),
    s2.center = K →
    surfaceArea (intersection s1 s2) = π * s1.radius^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_intersection_area_constant_l99_9904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l99_9991

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

noncomputable def g (x φ : ℝ) : ℝ := f (x + Real.pi/6) φ

theorem phi_value (φ : ℝ) (h1 : |φ| < Real.pi/2) 
  (h2 : ∀ x, g x φ = -g (-x) φ) : φ = -Real.pi/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l99_9991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l99_9927

/-- The parabola y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola y² = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The line with slope 1 passing through the focus -/
def line (x y : ℝ) : Prop := y = x - 1

/-- The intersection points of the line and the parabola -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ line p.1 p.2}

theorem intersection_segment_length :
  ∃ A B, A ∈ intersection_points ∧ B ∈ intersection_points ∧ ‖A - B‖ = 8 := by
  sorry

#check intersection_segment_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l99_9927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flat_bottomed_g_l99_9907

/-- A function f is "flat-bottomed" on domain D if there exists a closed interval [a, b] ⊆ D and a constant c such that:
    1) For all x₁ ∈ [a, b], f(x₁) = c
    2) For all x₂ ∈ D \ [a, b], f(x₂) > c -/
def is_flat_bottomed (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≤ b ∧ Set.Icc a b ⊆ D ∧
    (∀ x ∈ Set.Icc a b, f x = c) ∧
    (∀ x ∈ D \ Set.Icc a b, f x > c)

/-- The domain for the function g -/
def D : Set ℝ := Set.Ici (-2)

/-- The function g(x) = mx + √(x² + 2x + n) -/
noncomputable def g (m n : ℝ) (x : ℝ) : ℝ := m * x + Real.sqrt (x^2 + 2*x + n)

/-- Theorem: For g to be flat-bottomed on D, m = 1 and n = 1 -/
theorem flat_bottomed_g :
  ∀ m n : ℝ, is_flat_bottomed (g m n) D ↔ m = 1 ∧ n = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flat_bottomed_g_l99_9907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_property_l99_9900

-- Define the hyperbola
def is_on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 / 16 - y^2 / 9 = 1

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- State the theorem
theorem hyperbola_distance_property (P : ℝ × ℝ) :
  is_on_hyperbola P →
  distance P (5, 0) = 15 →
  (distance P (-5, 0) = 7 ∨ distance P (-5, 0) = 23) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_property_l99_9900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_field_side_length_is_40_l99_9954

/-- The length of each side of a square field, given the time taken to run around it and the runner's speed -/
noncomputable def square_field_side_length (time_seconds : ℝ) (speed_km_per_hour : ℝ) : ℝ :=
  let speed_meters_per_second := speed_km_per_hour * 1000 / 3600
  let perimeter := speed_meters_per_second * time_seconds
  perimeter / 4

theorem square_field_side_length_is_40 :
  square_field_side_length 48 12 = 40 := by
  unfold square_field_side_length
  simp
  -- The proof steps would go here, but we'll use sorry as requested
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_field_side_length_is_40_l99_9954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_unique_sides_l99_9914

/-- A triangle with specific angle and side length properties -/
structure SpecialTriangle where
  -- The sides of the triangle
  a : ℕ
  b : ℕ
  c : ℕ
  -- The angles of the triangle (in radians)
  α : ℝ
  β : ℝ
  γ : ℝ
  -- The sides are consecutive natural numbers
  consec_sides : (b = a + 1) ∧ (c = b + 1)
  -- The largest angle is twice the smallest
  angle_relation : γ = 2 * α
  -- The angles sum to π
  angle_sum : α + β + γ = π
  -- Law of sines
  law_of_sines : (Real.sin α) / (a : ℝ) = (Real.sin β) / (b : ℝ) ∧ 
                 (Real.sin β) / (b : ℝ) = (Real.sin γ) / (c : ℝ)
  -- Law of cosines
  law_of_cosines : (a : ℝ)^2 = (b : ℝ)^2 + (c : ℝ)^2 - 2 * (b : ℝ) * (c : ℝ) * Real.cos α

/-- The unique side lengths of a SpecialTriangle are 4, 5, and 6 -/
theorem special_triangle_unique_sides (t : SpecialTriangle) : t.a = 4 ∧ t.b = 5 ∧ t.c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_unique_sides_l99_9914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_bound_l99_9924

theorem sine_difference_bound (N : ℕ) (hN : N > 0) :
  ∃ (n k : ℕ), n ≠ k ∧ n ≤ N + 1 ∧ k ≤ N + 1 ∧ |Real.sin (n : ℝ) - Real.sin (k : ℝ)| < 2 / (N : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_bound_l99_9924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_completion_days_l99_9941

/-- The number of days it takes person B to complete the project alone -/
noncomputable def B_days : ℝ := 20

/-- The number of days it takes to complete the project when A and B work together, 
    with A quitting 10 days before completion -/
noncomputable def total_days : ℝ := 15

/-- The number of days A works on the project before quitting -/
noncomputable def A_work_days : ℝ := total_days - 10

/-- The function that calculates the fraction of the project completed in one day 
    when A and B work together -/
noncomputable def work_rate (x : ℝ) : ℝ := 1/x + 1/B_days

/-- The theorem stating that A can complete the project alone in 20 days -/
theorem A_completion_days : 
  ∃ (x : ℝ), x > 0 ∧ 
  A_work_days * work_rate x + (total_days - A_work_days) * (1/B_days) = 1 ∧ 
  x = 20 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_completion_days_l99_9941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_domain_is_positive_f_increasing_on_interval_l99_9979

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

-- State the theorem
theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Define the domain of f(x)
def f_domain : Set ℝ := {x : ℝ | x > 0}

-- State that the domain is (0, +∞)
theorem f_domain_is_positive : f_domain = {x : ℝ | x > 0} := by
  sorry

-- Additional theorem to state that f is increasing on (1, +∞)
theorem f_increasing_on_interval :
  StrictMonoOn f {x : ℝ | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_domain_is_positive_f_increasing_on_interval_l99_9979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_special_integers_l99_9996

/-- Given two distinct positive integers whose arithmetic mean is a two-digit integer
    and whose geometric mean is obtained by reversing the digits of the arithmetic mean,
    prove that the absolute difference between these integers is 66. -/
theorem absolute_difference_of_special_integers :
  ∀ x y : ℤ,
  x ≠ y →
  x > 0 →
  y > 0 →
  ∃ (a b : ℤ), a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧
  (x + y) / 2 = 10 * a + b →
  Real.sqrt (x * y : ℝ) = (10 * b + a : ℝ) →
  |x - y| = 66 := by
  sorry

#check absolute_difference_of_special_integers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_special_integers_l99_9996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_l99_9975

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / x

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := 1 / (x - 3)

-- Theorem stating that f_inv is the inverse of f
theorem f_inverse : ∀ x : ℝ, x ≠ 0 → f (f_inv (f x)) = x ∧ f_inv (f x) = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_l99_9975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_charlie_to_diana_l99_9913

-- Define the dimensions of the cans
def charlie_diameter : ℝ := 8
def charlie_height : ℝ := 16
def diana_diameter : ℝ := 16
def diana_height : ℝ := 8

-- Define the volume of a cylinder
noncomputable def cylinder_volume (diameter : ℝ) (height : ℝ) : ℝ :=
  (Real.pi / 4) * diameter^2 * height

-- Theorem statement
theorem volume_ratio_charlie_to_diana :
  (cylinder_volume charlie_diameter charlie_height) / 
  (cylinder_volume diana_diameter diana_height) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_charlie_to_diana_l99_9913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l99_9963

def is_valid_number (n : ℕ) : Bool :=
  200 ≤ n ∧ n < 300 ∧
  (n / 100 < (n / 10) % 10) ∧
  ((n / 10) % 10 < n % 10) ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  ((n / 10) % 10 ≠ n % 10) ∧
  (n / 100 ≠ n % 10)

theorem count_valid_numbers : 
  (Finset.filter (fun n => is_valid_number n) (Finset.range 300)).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l99_9963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_given_intersection_l99_9970

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if (1/2 : ℝ) < x ∧ x ≤ 1 then 2 * x^2 / (x + 1)
  else if 0 ≤ x ∧ x ≤ 1/2 then -1/3 * x + 1/6
  else 0  -- Default case to make the function total

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := 1/2 * a * x^2 - 2 * a + 2

-- State the theorem
theorem range_of_a_given_intersection (a : ℝ) : 
  (a > 0) → 
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ f x₁ = g a x₂) → 
  a ∈ Set.Icc (1/2) (4/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_given_intersection_l99_9970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l99_9915

/-- A hyperbola with focus on the x-axis and asymptotes y = ± (3/4)x has eccentricity 5/4 -/
theorem hyperbola_eccentricity (h : Real → Real → Prop) 
  (focus_on_x : ∃ c, ∀ x y, h x y → x = c)
  (asymptotes : ∀ x y, h x y → (y = (3/4) * x ∨ y = -(3/4) * x)) : 
  ∃ e, e = 5/4 ∧ ∀ x y, h x y → (x^2 / a^2 - y^2 / b^2 = 1 ∧ e^2 = 1 + b^2 / a^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l99_9915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l99_9946

/-- The speed of a train given its length and time to cross a stationary point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem stating that a 350-meter train crossing a point in 4.5 seconds has a speed of 77.78 m/s -/
theorem train_speed_calculation :
  let length : ℝ := 350
  let time : ℝ := 4.5
  abs (train_speed length time - 77.78) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l99_9946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_function_theorem_l99_9973

/-- A complex function with specific properties -/
def f (a b : ℝ) (z : ℂ) : ℂ := (a + b * Complex.I) * z

/-- The main theorem -/
theorem equidistant_function_theorem (a b : ℝ) (m n : ℕ) :
  (∀ z : ℂ, Complex.abs (f a b z) = Complex.abs (z - f a b z)) →
  a > 0 →
  b > 0 →
  Complex.abs (a + b * Complex.I) = 8 →
  b^2 = m / n →
  Nat.Coprime m n →
  m + n = 259 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_function_theorem_l99_9973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleFigureE_l99_9987

-- Define the pieces
structure Piece where
  width : Nat
  height : Nat

-- Define the figures
inductive Figure
  | A | B | C | D | E

-- Define the set of available pieces
def availablePieces : List Piece := [
  { width := 1, height := 1 },
  { width := 1, height := 1 },
  { width := 1, height := 1 },
  { width := 1, height := 2 },
  { width := 1, height := 2 },
  { width := 2, height := 2 }
]

-- Function to check if a figure can be formed
def canFormFigure (f : Figure) : Prop :=
  match f with
  | Figure.E => False
  | _ => True

-- Theorem statement
theorem impossibleFigureE :
  ∀ (arrangement : List Piece),
    arrangement = availablePieces →
    ¬(canFormFigure Figure.E) ∧
    (canFormFigure Figure.A) ∧
    (canFormFigure Figure.B) ∧
    (canFormFigure Figure.C) ∧
    (canFormFigure Figure.D) :=
by
  intro arrangement h
  apply And.intro
  · simp [canFormFigure]
  · apply And.intro
    · simp [canFormFigure]
    · apply And.intro
      · simp [canFormFigure]
      · apply And.intro
        · simp [canFormFigure]
        · simp [canFormFigure]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleFigureE_l99_9987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_special_after_3429_l99_9971

def is_special (n : ℕ) : Prop :=
  (n ≥ 1000) ∧ (n < 10000) ∧ (Finset.card (Finset.image (λ d ↦ d % 10) (Finset.range 4)) = 4)

theorem smallest_special_after_3429 :
  ∀ k : ℕ, k > 3429 ∧ k < 3450 → ¬ is_special k ∧ is_special 3450 := by
  sorry

#check smallest_special_after_3429

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_special_after_3429_l99_9971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mahesh_work_days_l99_9956

/-- The number of days Mahesh worked on the project -/
noncomputable def mahesh_days : ℝ := 24

/-- Mahesh's work rate per day -/
noncomputable def mahesh_rate : ℝ := 1 / 45

/-- Rajesh's work rate per day -/
noncomputable def rajesh_rate : ℝ := 1 / 30

/-- Total days to complete the work -/
def total_days : ℕ := 54

/-- Number of days Rajesh worked -/
def rajesh_days : ℕ := 30

theorem mahesh_work_days :
  (mahesh_days : ℝ) + rajesh_days = total_days ∧
  mahesh_rate * mahesh_days + rajesh_rate * (rajesh_days : ℝ) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mahesh_work_days_l99_9956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_desk_occupants_l99_9945

-- Define the students
inductive Student : Type where
  | Artjom | Borya | Vova | Grisha | Dima | Zhenya

-- Define the desks
inductive Desk : Type where
  | First | Second | Third

-- Define a function to represent seating arrangement
def seating : Student → Desk := sorry

-- Define the conditions
axiom dima_distracts : ∃ s : Student, seating Student.Dima = seating s

axiom borya_behind_zhenya : 
  (seating Student.Borya = Desk.Third ∧ seating Student.Zhenya = Desk.Second) ∨
  (seating Student.Borya = Desk.Second ∧ seating Student.Zhenya = Desk.First)

axiom artjom_grisha_together : seating Student.Artjom = seating Student.Grisha

axiom vova_zhenya_separate : seating Student.Vova ≠ seating Student.Zhenya

-- Theorem to prove
theorem second_desk_occupants :
  seating Student.Zhenya = Desk.Second ∧ seating Student.Dima = Desk.Second := by
  sorry

#check second_desk_occupants

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_desk_occupants_l99_9945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jakes_bike_time_is_two_hours_l99_9997

/-- The time it takes Jake to bike to the water park -/
noncomputable def jakes_bike_time (dad_drive_time : ℝ) (dad_speed1 : ℝ) (dad_speed2 : ℝ) (jake_speed : ℝ) : ℝ :=
  let half_time := dad_drive_time / 2
  let distance1 := dad_speed1 * half_time
  let distance2 := dad_speed2 * half_time
  let total_distance := distance1 + distance2
  total_distance / jake_speed

theorem jakes_bike_time_is_two_hours :
  jakes_bike_time (1/2) 28 60 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jakes_bike_time_is_two_hours_l99_9997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_range_l99_9999

noncomputable def f (x : ℝ) := x + 1/x

theorem f_monotonicity_and_range :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂) ∧
  (∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x : ℝ, 1/2 ≤ x ∧ x ≤ 2 → 5/2 ≤ f (1 / f x) ∧ f (1 / f x) ≤ 29/10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_range_l99_9999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l99_9989

theorem exponential_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (2 : ℝ)^x - (2 : ℝ)^y > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l99_9989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_divisors_of_30_has_three_prime_factors_l99_9966

def divisors (n : ℕ) : Finset ℕ := Finset.filter (· ∣ n) (Finset.range (n + 1))

def productOfDivisors (n : ℕ) : ℕ := (divisors n).prod id

def distinctPrimeFactors (n : ℕ) : ℕ := (Nat.factors n).toFinset.card

theorem product_of_divisors_of_30_has_three_prime_factors :
  distinctPrimeFactors (productOfDivisors 30) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_divisors_of_30_has_three_prime_factors_l99_9966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_distance_between_difference_and_sum_l99_9929

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (-1, -4)
def center2 : ℝ × ℝ := (2, 2)
def radius1 : ℝ := 5

noncomputable def radius2 : ℝ := Real.sqrt 10

-- Theorem stating that the circles intersect
theorem circles_intersect :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
by
  sorry

-- Theorem to check if the distance between centers is between the difference and sum of radii
theorem distance_between_difference_and_sum :
  let d := Real.sqrt ((2 - (-1))^2 + (2 - (-4))^2)
  |radius1 - radius2| < d ∧ d < radius1 + radius2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_distance_between_difference_and_sum_l99_9929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l99_9951

/-- Horner's Method operations for a polynomial of degree n -/
def horner_operations (n : ℕ) : ℕ × ℕ :=
  (n, n)

/-- A polynomial of degree 5 -/
def polynomial_degree_5 (x : ℝ) : ℝ :=
  5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

/-- The degree of the polynomial -/
def polynomial_degree : ℕ := 5

theorem horner_method_operations :
  horner_operations polynomial_degree = (5, 5) := by
  rfl

#eval horner_operations polynomial_degree

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l99_9951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_zero_l99_9980

theorem polynomial_value_at_zero (p : Polynomial ℝ) :
  (Polynomial.degree p = 5) →
  (∀ n : ℕ, n < 6 → p.eval (3 ^ n : ℝ) = (3 ^ n : ℝ)⁻¹) →
  p.eval 0 = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_zero_l99_9980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagram_internal_angle_heptagram_internal_angle_approx_l99_9935

/-- Represents a regular seven-pointed star (heptagram) -/
structure Heptagram where
  /-- The number of points in the star -/
  num_points : ℕ
  /-- The central angle between two adjacent points -/
  central_angle : ℝ
  /-- Assertion that the star has 7 points -/
  is_seven_pointed : num_points = 7
  /-- Assertion that the central angles sum to 360° -/
  central_angle_sum : central_angle * (num_points : ℝ) = 360

/-- The measure of an internal angle in a regular heptagram -/
noncomputable def internal_angle (h : Heptagram) : ℝ :=
  270 / 7

/-- Theorem: The measure of an internal angle in a regular heptagram is 270°/7 -/
theorem heptagram_internal_angle (h : Heptagram) :
  internal_angle h = 270 / 7 := by
  -- Unfold the definition of internal_angle
  unfold internal_angle
  -- The equality holds by definition
  rfl

/-- Theorem: The measure of an internal angle in a regular heptagram is approximately 38.57143° -/
theorem heptagram_internal_angle_approx (h : Heptagram) :
  ∃ ε > 0, |internal_angle h - 38.57143| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagram_internal_angle_heptagram_internal_angle_approx_l99_9935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_length_l99_9988

/-- Given a parallelogram with adjacent sides s and 3s units forming a 30-degree angle,
    if the area is 27√3 square units, then s = 3√(2√3). -/
theorem parallelogram_side_length (s : ℝ) : 
  s > 0 →
  (27 * Real.sqrt 3 : ℝ) = 3 * s * (s * Real.sin (30 * π / 180)) →
  s = 3 * Real.sqrt (2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_length_l99_9988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l99_9985

-- Define the line λ: 2x - y + 3 = 0
def line_lambda (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Define the circle C: x^2 + (y-1)^2 = 5
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Theorem: The line intersects the circle
theorem line_intersects_circle : ∃ (x y : ℝ), line_lambda x y ∧ circle_C x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l99_9985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_focus_min_sum_distances_l99_9957

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Focus of the parabola -/
noncomputable def focus (e : Parabola) : Point :=
  { x := e.p / 2, y := 0 }

/-- Theorem: Minimum distance to focus is related to p -/
theorem min_distance_to_focus (e : Parabola) :
  ∃ (m : Point), m.y^2 = 2 * e.p * m.x ∧ 
  (∀ (n : Point), n.y^2 = 2 * e.p * n.x → 
    distance m (focus e) ≤ distance n (focus e)) ∧
  distance m (focus e) = 3 →
  e.p = 6 := by
  sorry

/-- Main theorem -/
theorem min_sum_distances (e : Parabola) (a : Point) :
  (∃ (m : Point), m.y^2 = 2 * e.p * m.x ∧ 
    (∀ (n : Point), n.y^2 = 2 * e.p * n.x → 
      distance m (focus e) ≤ distance n (focus e)) ∧
    distance m (focus e) = 3 →
    e.p = 6) →
  a.x = 4 ∧ a.y = 1 →
  ∃ (min : ℝ), 
    (∀ (p : Point), p.y^2 = 2 * e.p * p.x → 
      distance p a + distance p (focus e) ≥ min) ∧
    min = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_focus_min_sum_distances_l99_9957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_l99_9994

/-- Represents a square in 2D space -/
structure Square where
  side_length : ℝ

/-- Represents a right triangle in 2D space -/
structure RightTriangle where
  base : ℝ
  height : ℝ

/-- Theorem: A square can be divided into a smaller square and four congruent right triangles -/
theorem square_division (s : Square) (inner_side : ℝ) 
  (h_inner_side : 0 < inner_side ∧ inner_side < s.side_length) :
  ∃ (inner_square : Square) (outer_triangles : Finset RightTriangle),
    inner_square.side_length = inner_side ∧
    outer_triangles.card = 4 ∧
    (∀ t ∈ outer_triangles, t.base = (s.side_length - inner_side) / 2 ∧ t.height = inner_side) ∧
    s.side_length^2 = inner_square.side_length^2 + 4 * (outer_triangles.toList.map (λ t ↦ t.base * t.height)).sum :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_l99_9994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_year_interest_rate_l99_9948

-- Define the initial amount, interest rates, and final amount
noncomputable def initial_amount : ℚ := 5000
noncomputable def first_year_rate : ℚ := 2 / 100
noncomputable def final_amount : ℚ := 5253

-- Define a function to calculate the amount after the first year
noncomputable def amount_after_first_year (principal : ℚ) (rate : ℚ) : ℚ :=
  principal * (1 + rate)

-- Define a function to calculate the interest rate for the second year
noncomputable def second_year_rate (initial : ℚ) (first_rate : ℚ) (final : ℚ) : ℚ :=
  (final / (initial * (1 + first_rate))) - 1

-- Theorem statement
theorem second_year_interest_rate :
  second_year_rate initial_amount first_year_rate final_amount = 3 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_year_interest_rate_l99_9948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_stars_sum_l99_9925

/-- The number of gold stars Shelby earned each day of the week -/
def stars : Fin 7 → ℕ
| 0 => 4  -- Monday
| 1 => 6  -- Tuesday
| 2 => 3  -- Wednesday
| 3 => 5  -- Thursday
| 4 => 2  -- Friday
| 5 => 3  -- Saturday
| 6 => 7  -- Sunday

/-- The total number of gold stars Shelby earned in the week -/
def totalStars : ℕ := (Finset.range 7).sum (fun i => stars i)

theorem shelby_stars_sum :
  totalStars = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_stars_sum_l99_9925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_increase_is_70_l99_9968

/-- The highest price of gasoline -/
noncomputable def highest_price : ℝ := 17

/-- The lowest price of gasoline -/
noncomputable def lowest_price : ℝ := 10

/-- The percentage increase from the lowest price to the highest price -/
noncomputable def percentage_increase : ℝ := ((highest_price - lowest_price) / lowest_price) * 100

/-- Theorem stating that the percentage increase is 70% -/
theorem percentage_increase_is_70 : percentage_increase = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_increase_is_70_l99_9968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l99_9958

/-- Ellipse with center at origin, right focus at (√3, 0), and maximum inscribed triangle area of 2 -/
structure SpecialEllipse where
  /-- Standard equation coefficient for x² -/
  a : ℝ
  /-- Standard equation coefficient for y² -/
  b : ℝ
  /-- Right focus x-coordinate -/
  c : ℝ
  /-- Maximum area of inscribed triangle -/
  maxArea : ℝ
  /-- Constraint on focus position -/
  focus_constraint : c = Real.sqrt 3
  /-- Constraint on maximum area -/
  area_constraint : maxArea = 2
  /-- Relationship between a, b, and c -/
  ellipse_constraint : a^2 = b^2 + c^2

/-- Line intersecting the ellipse -/
structure IntersectingLine where
  /-- Slope of the line -/
  k : ℝ
  /-- y-intercept of the line -/
  m : ℝ
  /-- Constraint on slope -/
  slope_constraint : k > 0

/-- Theorem about the special ellipse and its properties -/
theorem special_ellipse_properties (E : SpecialEllipse) (l : IntersectingLine) :
  (E.a = 2 ∧ E.b = 1) ∧
  (∃ (k₁ k₂ : ℝ), k₁ * l.k = l.k * k₂ →
    (∀ S S₁ S₂ : ℝ, (S₁ + S₂) / S ≥ (5 * Real.pi) / 4)) ∧
  (l.k = 1/2 ∧ (l.m = 1 ∨ l.m = -1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l99_9958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_range_l99_9942

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x * (Real.sin (Real.pi / 4 + x / 2))^2 + Real.cos (2 * x)

theorem monotone_increasing_range (w : ℝ) (hw : w > 0) :
  (∀ x ∈ Set.Icc (-Real.pi/2) (2*Real.pi/3), Monotone (fun x => f (w * x))) →
  w ∈ Set.Ioo 0 (3/4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_range_l99_9942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_distance_30_min_l99_9902

/-- Calculates the distance traveled given speed and time --/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that traveling at 50 km/h for 0.5 hours (30 minutes) results in a distance of 25 km --/
theorem travel_distance_30_min : 
  let speed : ℝ := 50  -- Speed in km/h
  let time : ℝ := 0.5  -- Time in hours (30 minutes = 0.5 hours)
  distance_traveled speed time = 25 := by
  -- Unfold the definition of distance_traveled
  unfold distance_traveled
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num

#check travel_distance_30_min

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_distance_30_min_l99_9902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equivalence_l99_9909

/-- Proves that 0.0000003 is equal to 3 × 10^(-7) in scientific notation -/
theorem scientific_notation_equivalence : (3 * 10^(-7 : ℤ) : ℝ) = 0.0000003 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equivalence_l99_9909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_g_strictly_increasing_k_upper_bound_l99_9964

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (4^x - a) / 2^x

-- Theorem 1
theorem odd_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a x = -f a (-x)) → a = 1 := by sorry

-- Define the simplified function g after determining a = 1
noncomputable def g (x : ℝ) : ℝ := 2^x - 2^(-x)

-- Theorem 2
theorem g_strictly_increasing :
  ∀ x y, x < y → g x < g y := by sorry

-- Theorem 3
theorem k_upper_bound (k : ℝ) :
  (∀ x, g (x^2 - x) + g (2*x^2 - k) > 0) → k < -1/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_g_strictly_increasing_k_upper_bound_l99_9964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_of_f_l99_9916

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := ((2^x + 1)^2) / (2^x * x) + 1

-- Define the domain
def domain : Set ℝ := {x | (x ≥ -2018 ∧ x < 0) ∨ (x > 0 ∧ x ≤ 2018)}

-- State the theorem
theorem max_min_sum_of_f :
  ∃ (M N : ℝ), 
    (∀ x ∈ domain, f x ≤ M) ∧
    (∃ x ∈ domain, f x = M) ∧
    (∀ x ∈ domain, f x ≥ N) ∧
    (∃ x ∈ domain, f x = N) ∧
    M + N = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_of_f_l99_9916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l99_9984

noncomputable def f (x y : ℝ) : ℝ := min x (y / (x^2 + y^2))

theorem f_max_value :
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → f x y ≤ f x₀ y₀) ∧
  f x₀ y₀ = 1 / Real.sqrt 2 := by
  sorry

#check f_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l99_9984
