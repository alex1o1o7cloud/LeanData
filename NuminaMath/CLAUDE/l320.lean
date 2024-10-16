import Mathlib

namespace NUMINAMATH_CALUDE_some_number_value_l320_32049

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * n * 49) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l320_32049


namespace NUMINAMATH_CALUDE_division_problem_l320_32092

theorem division_problem (dividend quotient divisor remainder multiple : ℕ) :
  remainder = 6 →
  dividend = 86 →
  divisor = 5 * quotient →
  divisor = multiple * remainder + 2 →
  dividend = divisor * quotient + remainder →
  multiple = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l320_32092


namespace NUMINAMATH_CALUDE_cheese_wedge_volume_l320_32097

/-- The volume of a wedge of cheese that represents one-third of a cylindrical log --/
theorem cheese_wedge_volume (d h r : ℝ) : 
  d = 12 →  -- diameter is 12 cm
  h = d →   -- height is equal to diameter
  r = d / 2 →  -- radius is half the diameter
  (1 / 3) * (π * r^2 * h) = 144 * π := by
  sorry

end NUMINAMATH_CALUDE_cheese_wedge_volume_l320_32097


namespace NUMINAMATH_CALUDE_playground_max_area_l320_32096

theorem playground_max_area :
  ∀ (width height : ℕ),
    width + height = 75 →
    width * height ≤ 1406 :=
by
  sorry

end NUMINAMATH_CALUDE_playground_max_area_l320_32096


namespace NUMINAMATH_CALUDE_digit_puzzle_solution_l320_32004

theorem digit_puzzle_solution :
  ∀ (E F G H : ℕ),
  (E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10) →
  (10 * E + F) + (10 * G + E) = 10 * H + E →
  (10 * E + F) - (10 * G + E) = E →
  H = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_puzzle_solution_l320_32004


namespace NUMINAMATH_CALUDE_yard_length_is_700_l320_32001

/-- The length of a yard with trees planted at equal distances -/
def yard_length (num_trees : ℕ) (tree_distance : ℕ) : ℕ :=
  (num_trees - 1) * tree_distance

/-- Proof that the yard length is 700 meters -/
theorem yard_length_is_700 :
  yard_length 26 28 = 700 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_is_700_l320_32001


namespace NUMINAMATH_CALUDE_new_triangle_area_ratio_l320_32056

/-- Represents a triangle -/
structure Triangle where
  area : ℝ

/-- Represents a point on a side of a triangle -/
structure PointOnSide where
  distance_ratio : ℝ

/-- Creates a new triangle from points on the sides of an original triangle -/
def new_triangle_from_points (original : Triangle) (p1 p2 p3 : PointOnSide) : Triangle :=
  sorry

theorem new_triangle_area_ratio (T : Triangle) :
  let p1 : PointOnSide := { distance_ratio := 1/3 }
  let p2 : PointOnSide := { distance_ratio := 1/3 }
  let p3 : PointOnSide := { distance_ratio := 1/3 }
  let new_T := new_triangle_from_points T p1 p2 p3
  new_T.area = (1/9) * T.area := by
  sorry

end NUMINAMATH_CALUDE_new_triangle_area_ratio_l320_32056


namespace NUMINAMATH_CALUDE_football_yards_gained_l320_32065

theorem football_yards_gained (initial_loss : ℤ) (final_progress : ℤ) (yards_gained : ℤ) : 
  initial_loss = -5 → final_progress = 2 → yards_gained = initial_loss + final_progress →
  yards_gained = 7 := by
sorry

end NUMINAMATH_CALUDE_football_yards_gained_l320_32065


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l320_32014

theorem negative_sixty_four_to_four_thirds (x : ℝ) : x = (-64)^(4/3) → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l320_32014


namespace NUMINAMATH_CALUDE_initial_girls_count_l320_32018

theorem initial_girls_count (total : ℕ) : 
  (total ≠ 0) →
  (total / 2 : ℚ) = (total / 2 : ℕ) →
  ((total / 2 : ℕ) - 5 : ℚ) / total = 2 / 5 →
  total / 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l320_32018


namespace NUMINAMATH_CALUDE_hyperbola_ac_range_l320_32028

-- Define the hyperbola and its properties
structure Hyperbola where
  focal_distance : ℝ
  a : ℝ
  left_focus : ℝ × ℝ
  right_focus : ℝ × ℝ
  point_on_right_branch : ℝ × ℝ

-- Define the theorem
theorem hyperbola_ac_range (h : Hyperbola) : 
  h.focal_distance = 4 → 
  h.a < 2 →
  let A := h.left_focus
  let B := h.right_focus
  let C := h.point_on_right_branch
  (dist C B - dist C A = 2 * h.a) →
  (dist A C + dist B C + dist A B = 10) →
  (3 < dist A C) ∧ (dist A C < 5) := by
  sorry

-- Note: dist is assumed to be a function that calculates the distance between two points

end NUMINAMATH_CALUDE_hyperbola_ac_range_l320_32028


namespace NUMINAMATH_CALUDE_sandy_clothes_spending_l320_32055

/-- The amount Sandy spent on clothes -/
def total_spent (shorts_cost shirt_cost jacket_cost : ℚ) : ℚ :=
  shorts_cost + shirt_cost + jacket_cost

/-- Theorem: Sandy's total spending on clothes -/
theorem sandy_clothes_spending :
  total_spent 13.99 12.14 7.43 = 33.56 := by
  sorry

end NUMINAMATH_CALUDE_sandy_clothes_spending_l320_32055


namespace NUMINAMATH_CALUDE_factorial_ratio_l320_32017

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l320_32017


namespace NUMINAMATH_CALUDE_equation_solution_l320_32036

theorem equation_solution :
  ∃ x : ℚ, (0.05 * x + 0.12 * (30 + x) = 18) ∧ (x = 144 / 17) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l320_32036


namespace NUMINAMATH_CALUDE_constant_expression_l320_32061

theorem constant_expression (x y : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hsum : x + y = 1) :
  x / (y^3 - 1) + y / (1 - x^3) + 2 * (x - y) / (x^2 * y^2 + 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_constant_expression_l320_32061


namespace NUMINAMATH_CALUDE_right_triangle_k_values_right_triangle_k_values_only_l320_32038

/-- A right-angled triangle in 2D space -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angled : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∨
                    (C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2) = 0 ∨
                    (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

/-- The theorem stating the possible values of k -/
theorem right_triangle_k_values (t : RightTriangle) (k : ℝ) : 
  t.B = (t.A.1 + 2, t.A.2 + 1) ∧ t.C = (t.A.1 + 3, t.A.2 + k) → k = -6 ∨ k = -1 := by
  sorry

/-- The main theorem proving that -6 and -1 are the only possible values for k -/
theorem right_triangle_k_values_only (t : RightTriangle) :
  t.B = (t.A.1 + 2, t.A.2 + 1) ∧ t.C = (t.A.1 + 3, t.A.2 + k) →
  (k = -6 ∨ k = -1) ∧ ∀ m, (m = -6 ∨ m = -1 ∨ ¬∃ s : RightTriangle, 
    s.B = (s.A.1 + 2, s.A.2 + 1) ∧ s.C = (s.A.1 + 3, s.A.2 + m)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_k_values_right_triangle_k_values_only_l320_32038


namespace NUMINAMATH_CALUDE_vertically_opposite_angles_equal_l320_32082

-- Define a type for angles
def Angle : Type := ℝ

-- Define a function to represent vertically opposite angles
def verticallyOpposite (α β : Angle) : Prop := sorry

-- Theorem: Vertically opposite angles are equal
theorem vertically_opposite_angles_equal (α β : Angle) :
  verticallyOpposite α β → α = β :=
sorry

end NUMINAMATH_CALUDE_vertically_opposite_angles_equal_l320_32082


namespace NUMINAMATH_CALUDE_isosceles_triangle_l320_32058

theorem isosceles_triangle (A B C : Real) (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) : A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l320_32058


namespace NUMINAMATH_CALUDE_parabola_equation_l320_32033

-- Define the parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the line x - y + 2 = 0
def focus_line (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the conditions for the parabola
def parabola_conditions (p : Parabola) : Prop :=
  -- Vertex at origin
  p.equation 0 0
  -- Axis of symmetry is a coordinate axis
  ∧ (∀ x y : ℝ, p.equation x y → p.equation x (-y) ∨ p.equation (-x) y)
  -- Focus on the line x - y + 2 = 0
  ∧ ∃ fx fy : ℝ, focus_line fx fy ∧ 
    ((∀ x y : ℝ, p.equation x y ↔ (x - fx)^2 + (y - fy)^2 = (x + fx)^2 + (y + fy)^2)
    ∨ (∀ x y : ℝ, p.equation x y ↔ (x - fx)^2 + (y - fy)^2 = (x - fx)^2 + (y + fy)^2))

-- Theorem statement
theorem parabola_equation (p : Parabola) (h : parabola_conditions p) :
  (∀ x y : ℝ, p.equation x y ↔ y^2 = -8*x) ∨ (∀ x y : ℝ, p.equation x y ↔ x^2 = 8*y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l320_32033


namespace NUMINAMATH_CALUDE_annies_ride_distance_l320_32067

/-- Represents the taxi fare structure -/
structure TaxiFare where
  startFee : ℝ
  perMileFee : ℝ
  tollFee : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (fare : TaxiFare) (miles : ℝ) : ℝ :=
  fare.startFee + fare.tollFee + fare.perMileFee * miles

theorem annies_ride_distance (mikeFare annieFare : TaxiFare) 
  (h1 : mikeFare.startFee = 2.5)
  (h2 : mikeFare.perMileFee = 0.25)
  (h3 : mikeFare.tollFee = 0)
  (h4 : annieFare.startFee = 2.5)
  (h5 : annieFare.perMileFee = 0.25)
  (h6 : annieFare.tollFee = 5)
  (h7 : totalFare mikeFare 36 = totalFare annieFare (annies_miles : ℝ)) :
  annies_miles = 16 := by
  sorry

end NUMINAMATH_CALUDE_annies_ride_distance_l320_32067


namespace NUMINAMATH_CALUDE_saltwater_solution_bounds_l320_32075

theorem saltwater_solution_bounds :
  let solution_A : ℝ := 5  -- Concentration of solution A (%)
  let solution_B : ℝ := 8  -- Concentration of solution B (%)
  let solution_C : ℝ := 9  -- Concentration of solution C (%)
  let weight_A : ℝ := 60   -- Weight of solution A (g)
  let weight_B : ℝ := 60   -- Weight of solution B (g)
  let weight_C : ℝ := 47   -- Weight of solution C (g)
  let target_concentration : ℝ := 7  -- Target concentration (%)
  let target_weight : ℝ := 100       -- Target weight (g)

  ∀ x y z : ℝ,
    x + y + z = target_weight →
    solution_A * x + solution_B * y + solution_C * z = target_concentration * target_weight →
    0 ≤ x ∧ x ≤ weight_A →
    0 ≤ y ∧ y ≤ weight_B →
    0 ≤ z ∧ z ≤ weight_C →
    (∃ x_max : ℝ, x ≤ x_max ∧ x_max = 49) ∧
    (∃ x_min : ℝ, x_min ≤ x ∧ x_min = 35) :=
by sorry

end NUMINAMATH_CALUDE_saltwater_solution_bounds_l320_32075


namespace NUMINAMATH_CALUDE_tv_watch_time_two_weeks_l320_32026

/-- Calculates the total hours of TV watched in two weeks -/
def tvWatchTimeInTwoWeeks (minutesPerDay : ℕ) (daysPerWeek : ℕ) : ℚ :=
  let minutesPerWeek : ℕ := minutesPerDay * daysPerWeek
  let hoursPerWeek : ℚ := minutesPerWeek / 60
  hoursPerWeek * 2

/-- Theorem: Children watching 45 minutes of TV per day, 4 days a week, watch 6 hours in two weeks -/
theorem tv_watch_time_two_weeks :
  tvWatchTimeInTwoWeeks 45 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_tv_watch_time_two_weeks_l320_32026


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_l320_32053

/-- The ellipse C₁ -/
def C₁ (x y a b : ℝ) : Prop := y^2 / a^2 + x^2 / b^2 = 1

/-- The parabola C₂ -/
def C₂ (x y p : ℝ) : Prop := x^2 = 2 * p * y

/-- The directrix l of C₂ -/
def l (y : ℝ) : Prop := y = -2

/-- Intersection point of l and C₁ -/
def intersection_point (x y : ℝ) : Prop := x = Real.sqrt 2 ∧ y = -2

/-- Common focus condition -/
def common_focus (a b p : ℝ) : Prop := sorry

theorem ellipse_parabola_intersection (a b p : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : p > 0)
  (h4 : common_focus a b p)
  (h5 : ∃ x y, C₁ x y a b ∧ l y ∧ intersection_point x y) :
  (∀ x y, C₁ x y a b ↔ y^2 / 8 + x^2 / 4 = 1) ∧
  (∀ x y, C₂ x y p ↔ x^2 = 8 * y) ∧
  (∃ min max : ℝ, min = -8 ∧ max = 2 ∧
    ∀ t : ℝ, ∃ x₃ y₃ x₄ y₄ : ℝ,
      C₁ x₃ y₃ a b ∧ C₁ x₄ y₄ a b ∧
      min < x₃ * x₄ + y₃ * y₄ ∧ x₃ * x₄ + y₃ * y₄ ≤ max) :=
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_l320_32053


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l320_32047

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def P : Set Nat := {1, 2, 3, 4, 5}
def Q : Set Nat := {3, 4, 5, 6, 7}

theorem intersection_complement_equality : P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l320_32047


namespace NUMINAMATH_CALUDE_square_sum_and_product_l320_32079

theorem square_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 49) : 
  x^2 + y^2 = 25 ∧ x * y = -12 := by
sorry

end NUMINAMATH_CALUDE_square_sum_and_product_l320_32079


namespace NUMINAMATH_CALUDE_regular_polygon_with_405_diagonals_has_30_sides_l320_32007

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 405 diagonals has 30 sides -/
theorem regular_polygon_with_405_diagonals_has_30_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 405 → n = 30 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_405_diagonals_has_30_sides_l320_32007


namespace NUMINAMATH_CALUDE_product_of_roots_l320_32023

theorem product_of_roots (x : ℝ) : 
  (24 * x^2 - 72 * x + 200 = 0) → 
  (∃ r₁ r₂ : ℝ, (24 * r₁^2 - 72 * r₁ + 200 = 0) ∧ 
                (24 * r₂^2 - 72 * r₂ + 200 = 0) ∧ 
                (r₁ * r₂ = 25 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l320_32023


namespace NUMINAMATH_CALUDE_sum_of_roots_l320_32037

theorem sum_of_roots (h1 : 32 = 2^5) (h2 : 27 = 3^3) :
  2 * (32 : ℝ)^(1/5) + 3 * (27 : ℝ)^(1/3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l320_32037


namespace NUMINAMATH_CALUDE_vertical_angles_equal_l320_32064

-- Define a line as a type
def Line : Type := ℝ → ℝ → Prop

-- Define a point as a pair of real numbers
def Point : Type := ℝ × ℝ

-- Define the notion of two lines intersecting at a point
def intersect (l1 l2 : Line) (p : Point) : Prop :=
  l1 p.1 p.2 ∧ l2 p.1 p.2

-- Define vertical angles
def vertical_angles (l1 l2 : Line) (p1 p2 p3 p4 : Point) : Prop :=
  ∃ (i : Point), intersect l1 l2 i ∧
  (p1 ≠ i ∧ p2 ≠ i ∧ p3 ≠ i ∧ p4 ≠ i) ∧
  (l1 p1.1 p1.2 ∧ l1 p3.1 p3.2) ∧
  (l2 p2.1 p2.2 ∧ l2 p4.1 p4.2)

-- Define angle measure
def angle_measure (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem: Vertical angles are equal
theorem vertical_angles_equal (l1 l2 : Line) (p1 p2 p3 p4 : Point) :
  vertical_angles l1 l2 p1 p2 p3 p4 →
  angle_measure p1 i p2 = angle_measure p3 i p4 :=
sorry

end NUMINAMATH_CALUDE_vertical_angles_equal_l320_32064


namespace NUMINAMATH_CALUDE_smallest_three_digit_number_l320_32006

def digits : Finset Nat := {3, 0, 2, 5, 7}

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : Nat), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  n = 100 * a + 10 * b + c

theorem smallest_three_digit_number :
  ∀ n, is_valid_number n → n ≥ 203 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_number_l320_32006


namespace NUMINAMATH_CALUDE_min_bushes_for_zucchinis_l320_32069

/-- The number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries needed to trade for one zucchini -/
def containers_per_zucchini : ℚ := 4 / 3

/-- The number of zucchinis Natalie wants to obtain -/
def target_zucchinis : ℕ := 72

/-- The minimum number of bushes needed to obtain at least the target number of zucchinis -/
def min_bushes_needed : ℕ :=
  (target_zucchinis * containers_per_zucchini / containers_per_bush).ceil.toNat

theorem min_bushes_for_zucchinis :
  min_bushes_needed = 10 := by sorry

end NUMINAMATH_CALUDE_min_bushes_for_zucchinis_l320_32069


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l320_32094

/-- The eccentricity of a hyperbola with the given conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (A B F : ℝ × ℝ) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  -- F is the right focus of C
  (F.1 = c ∧ F.2 = 0) →
  -- A and B are on the asymptotes of C
  (A.2 = (b / a) * A.1 ∧ B.2 = -(b / a) * B.1) →
  -- AF is perpendicular to the x-axis
  (A.1 = c ∧ A.2 = (b * c) / a) →
  -- AB is perpendicular to OB
  ((A.2 - B.2) / (A.1 - B.1) = a / b) →
  -- BF is parallel to OA
  ((F.2 - B.2) / (F.1 - B.1) = A.2 / A.1) →
  -- The eccentricity of the hyperbola
  c / a = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l320_32094


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l320_32085

theorem complex_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l320_32085


namespace NUMINAMATH_CALUDE_cliff_rock_collection_l320_32051

theorem cliff_rock_collection :
  let total_rocks : ℕ := 180
  let sedimentary_rocks : ℕ := total_rocks * 2 / 3
  let igneous_rocks : ℕ := sedimentary_rocks / 2
  let shiny_igneous_ratio : ℚ := 2 / 3
  shiny_igneous_ratio * igneous_rocks = 40 := by
  sorry

end NUMINAMATH_CALUDE_cliff_rock_collection_l320_32051


namespace NUMINAMATH_CALUDE_f_equality_iff_a_half_l320_32021

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then (4 : ℝ) ^ x else (2 : ℝ) ^ (a - x)

theorem f_equality_iff_a_half (a : ℝ) (h : a ≠ 1) :
  f a (1 - a) = f a (a - 1) ↔ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_equality_iff_a_half_l320_32021


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l320_32041

-- Define the function y = mx^2 - mx - 1
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | y (1/2) x < 0} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Part 2
theorem solution_set_part2 (m : ℝ) :
  {x : ℝ | y m x < (1 - m) * x - 1} =
    if m = 0 then
      {x : ℝ | x > 0}
    else if m > 0 then
      {x : ℝ | 0 < x ∧ x < 1/m}
    else
      {x : ℝ | x < 1/m ∨ x > 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l320_32041


namespace NUMINAMATH_CALUDE_school_year_days_is_180_l320_32080

/-- The number of days in a school year. -/
def school_year_days : ℕ := 180

/-- The maximum percentage of days that can be missed without taking exams. -/
def max_missed_percentage : ℚ := 5 / 100

/-- The number of days Hazel has already missed. -/
def days_already_missed : ℕ := 6

/-- The additional number of days Hazel can miss without taking exams. -/
def additional_days_can_miss : ℕ := 3

/-- Theorem stating that the number of days in the school year is 180. -/
theorem school_year_days_is_180 :
  (days_already_missed + additional_days_can_miss : ℚ) / school_year_days = max_missed_percentage :=
by sorry

end NUMINAMATH_CALUDE_school_year_days_is_180_l320_32080


namespace NUMINAMATH_CALUDE_series_sum_equals_one_l320_32086

/-- The sum of the series ∑(k=0 to ∞) 2^(2^k) / (4^(2^k) - 1) is equal to 1 -/
theorem series_sum_equals_one : 
  ∑' (k : ℕ), (2^(2^k)) / ((4^(2^k)) - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_l320_32086


namespace NUMINAMATH_CALUDE_preimage_of_two_three_l320_32076

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Theorem statement
theorem preimage_of_two_three :
  ∃ (x y : ℝ), f (x, y) = (2, 3) ∧ x = 5/2 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_two_three_l320_32076


namespace NUMINAMATH_CALUDE_tangent_line_circle_l320_32030

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x - y + 3 = 0

/-- The circle equation -/
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 - 2*x + 2 - a = 0

/-- The theorem statement -/
theorem tangent_line_circle (a : ℝ) :
  (∃ x y : ℝ, line_equation x y ∧ circle_equation x y a ∧
    ∀ x' y' : ℝ, line_equation x' y' → circle_equation x' y' a → (x = x' ∧ y = y')) →
  a = 9 := by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_l320_32030


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l320_32071

theorem sufficient_not_necessary (a : ℝ) : 
  (a > 1 → 1/a < 1) ∧ (∃ b : ℝ, b ≤ 1 ∧ 1/b < 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l320_32071


namespace NUMINAMATH_CALUDE_product_selection_theorem_product_display_theorem_l320_32098

def total_products : ℕ := 10
def ineligible_products : ℕ := 2
def products_to_select : ℕ := 4
def display_positions : ℕ := 6
def gold_medal_products : ℕ := 2

def arrangement (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem product_selection_theorem :
  arrangement (total_products - ineligible_products) products_to_select = 1680 :=
sorry

theorem product_display_theorem :
  arrangement display_positions gold_medal_products *
  arrangement (total_products - gold_medal_products) (display_positions - gold_medal_products) = 50400 :=
sorry

end NUMINAMATH_CALUDE_product_selection_theorem_product_display_theorem_l320_32098


namespace NUMINAMATH_CALUDE_expected_value_unfair_coin_expected_value_zero_l320_32057

/-- The expected monetary value of a single flip of an unfair coin -/
theorem expected_value_unfair_coin (p_heads : ℝ) (p_tails : ℝ) 
  (value_heads : ℝ) (value_tails : ℝ) : ℝ :=
  p_heads * value_heads + p_tails * value_tails

/-- Proof that the expected monetary value of the specific unfair coin is 0 -/
theorem expected_value_zero : 
  expected_value_unfair_coin (2/3) (1/3) 5 (-10) = 0 := by
sorry

end NUMINAMATH_CALUDE_expected_value_unfair_coin_expected_value_zero_l320_32057


namespace NUMINAMATH_CALUDE_container_count_l320_32073

theorem container_count (container_capacity : ℝ) (total_capacity : ℝ) : 
  (8 : ℝ) = 0.2 * container_capacity →
  total_capacity = 1600 →
  (total_capacity / container_capacity : ℝ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_container_count_l320_32073


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l320_32012

-- Define the function f for x > 0
def f_pos (x : ℝ) : ℝ := x^2 - x - 1

-- Define the property of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_pos : ∀ x > 0, f x = f_pos x) : 
  ∀ x < 0, f x = -x^2 - x + 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l320_32012


namespace NUMINAMATH_CALUDE_chord_division_theorem_l320_32029

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an annulus -/
structure Annulus where
  inner : Circle
  outer : Circle

/-- Theorem: Given an annulus and a point inside it, there exists a chord passing through the point
    that divides the chord in a given ratio -/
theorem chord_division_theorem (A : Annulus) (P : Point) (m n : ℝ) 
    (h_concentric : A.inner.center = A.outer.center)
    (h_inside : (P.x - A.inner.center.x)^2 + (P.y - A.inner.center.y)^2 > A.inner.radius^2 ∧ 
                (P.x - A.outer.center.x)^2 + (P.y - A.outer.center.y)^2 < A.outer.radius^2)
    (h_positive : m > 0 ∧ n > 0) :
  ∃ (A₁ A₂ : Point),
    -- A₁ is on the inner circle
    (A₁.x - A.inner.center.x)^2 + (A₁.y - A.inner.center.y)^2 = A.inner.radius^2 ∧
    -- A₂ is on the outer circle
    (A₂.x - A.outer.center.x)^2 + (A₂.y - A.outer.center.y)^2 = A.outer.radius^2 ∧
    -- P is on the line segment A₁A₂
    ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ P.x = A₁.x + t * (A₂.x - A₁.x) ∧ P.y = A₁.y + t * (A₂.y - A₁.y) ∧
    -- The ratio A₁P:PA₂ is m:n
    t / (1 - t) = m / n :=
by sorry

end NUMINAMATH_CALUDE_chord_division_theorem_l320_32029


namespace NUMINAMATH_CALUDE_gcd_lcm_multiple_relationship_l320_32032

theorem gcd_lcm_multiple_relationship (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 6) :
  Nat.gcd a b = b ∧ Nat.lcm a b = a := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_multiple_relationship_l320_32032


namespace NUMINAMATH_CALUDE_locus_equation_l320_32063

/-- Parabola type representing y^2 = 4px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (par : Parabola) where
  y : ℝ
  eq : y^2 = 4 * par.p * (y^2 / (4 * par.p))

/-- The locus of point M given two points on a parabola -/
def locusM (par : Parabola) (A B : ParabolaPoint par) (M : ℝ × ℝ) : Prop :=
  let OA := (A.y^2 / (4 * par.p), A.y)
  let OB := (B.y^2 / (4 * par.p), B.y)
  let (x, y) := M
  (OA.1 * OB.1 + OA.2 * OB.2 = 0) ∧  -- OA ⊥ OB
  (x * (B.y^2 - A.y^2) / (4 * par.p) + y * (B.y - A.y) = 0) ∧  -- OM ⊥ AB
  (x - A.y^2 / (4 * par.p)) * (B.y - A.y) = 
    ((B.y^2 - A.y^2) / (4 * par.p)) * (y - A.y)  -- M is on line AB

theorem locus_equation (par : Parabola) (A B : ParabolaPoint par) (M : ℝ × ℝ) :
  locusM par A B M → M.1^2 + M.2^2 - 4 * par.p * M.1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_equation_l320_32063


namespace NUMINAMATH_CALUDE_expression_equals_three_l320_32050

theorem expression_equals_three : 
  (1/2)⁻¹ + 4 * Real.cos (45 * π / 180) - Real.sqrt 8 + (2023 - Real.pi)^0 = 3 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_three_l320_32050


namespace NUMINAMATH_CALUDE_leo_caught_40_l320_32025

/-- The number of fish Leo caught -/
def leo_fish : ℕ := sorry

/-- The number of fish Agrey caught -/
def agrey_fish : ℕ := sorry

/-- Agrey caught 20 more fish than Leo -/
axiom agrey_more : agrey_fish = leo_fish + 20

/-- They caught a total of 100 fish together -/
axiom total_fish : leo_fish + agrey_fish = 100

/-- Prove that Leo caught 40 fish -/
theorem leo_caught_40 : leo_fish = 40 := by sorry

end NUMINAMATH_CALUDE_leo_caught_40_l320_32025


namespace NUMINAMATH_CALUDE_factors_of_1320_l320_32089

/-- The number of distinct, positive factors of 1320 -/
def num_factors_1320 : ℕ := sorry

/-- 1320 has exactly 24 distinct, positive factors -/
theorem factors_of_1320 : num_factors_1320 = 24 := by sorry

end NUMINAMATH_CALUDE_factors_of_1320_l320_32089


namespace NUMINAMATH_CALUDE_max_weekly_profit_l320_32088

/-- Represents the weekly sales profit as a function of the price increase -/
def weekly_profit (x : ℝ) : ℝ := -10 * x^2 + 100 * x + 6000

/-- Represents the number of items sold per week as a function of the price increase -/
def items_sold (x : ℝ) : ℝ := 300 - 10 * x

theorem max_weekly_profit :
  ∀ x : ℝ, x ≤ 20 → weekly_profit x ≤ 6250 ∧
  ∃ x₀ : ℝ, x₀ ≤ 20 ∧ weekly_profit x₀ = 6250 :=
sorry

end NUMINAMATH_CALUDE_max_weekly_profit_l320_32088


namespace NUMINAMATH_CALUDE_import_tax_percentage_l320_32046

/-- The import tax percentage problem -/
theorem import_tax_percentage 
  (total_value : ℝ) 
  (tax_threshold : ℝ) 
  (tax_paid : ℝ) 
  (h1 : total_value = 2590)
  (h2 : tax_threshold = 1000)
  (h3 : tax_paid = 111.30)
  : (tax_paid / (total_value - tax_threshold)) = 0.07 := by
  sorry

end NUMINAMATH_CALUDE_import_tax_percentage_l320_32046


namespace NUMINAMATH_CALUDE_binary_operation_proof_l320_32087

/-- Convert a binary number (represented as a list of bits) to a natural number -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- The first binary number 11001₂ -/
def num1 : List Bool := [true, true, false, false, true]

/-- The second binary number 1101₂ -/
def num2 : List Bool := [true, true, false, true]

/-- The third binary number 101₂ -/
def num3 : List Bool := [true, false, true]

/-- The result 100111010₂ -/
def result : List Bool := [true, false, false, true, true, true, false, true, false]

/-- Theorem stating that (11001₂ * 1101₂) - 101₂ = 100111010₂ -/
theorem binary_operation_proof :
  (binary_to_nat num1 * binary_to_nat num2) - binary_to_nat num3 = binary_to_nat result := by
  sorry

end NUMINAMATH_CALUDE_binary_operation_proof_l320_32087


namespace NUMINAMATH_CALUDE_blue_balls_count_l320_32066

/-- Given a jar with white and blue balls in a 5:3 ratio, 
    prove that 15 white balls implies 9 blue balls -/
theorem blue_balls_count (white_balls blue_balls : ℕ) : 
  (white_balls : ℚ) / blue_balls = 5 / 3 → 
  white_balls = 15 → 
  blue_balls = 9 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l320_32066


namespace NUMINAMATH_CALUDE_masha_sasha_numbers_l320_32074

theorem masha_sasha_numbers 
  (a b : ℕ) 
  (h_distinct : a ≠ b) 
  (h_greater_11 : a > 11 ∧ b > 11) 
  (h_sum_known : ∃ s, s = a + b) 
  (h_one_even : Even a ∨ Even b) 
  (h_unique : ∀ x y : ℕ, x ≠ y → x > 11 → y > 11 → x + y = a + b → (Even x ∨ Even y) → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
sorry

end NUMINAMATH_CALUDE_masha_sasha_numbers_l320_32074


namespace NUMINAMATH_CALUDE_original_number_proof_l320_32083

theorem original_number_proof (N : ℕ) : N = 28 ↔ 
  (∃ k : ℕ, N - 11 = 17 * k) ∧ 
  (∀ x : ℕ, x < 11 → ¬(∃ m : ℕ, N - x = 17 * m)) ∧
  (∀ M : ℕ, M < N → ¬(∃ k : ℕ, M - 11 = 17 * k) ∨ 
    (∃ x : ℕ, x < 11 ∧ ∃ m : ℕ, M - x = 17 * m)) :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l320_32083


namespace NUMINAMATH_CALUDE_set_intersection_problem_l320_32081

def M : Set ℝ := {x | 0 < x ∧ x < 8}
def N : Set ℝ := {x | ∃ n : ℕ, x = 2 * n + 1}

theorem set_intersection_problem : M ∩ N = {1, 3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l320_32081


namespace NUMINAMATH_CALUDE_project_hours_proof_l320_32016

theorem project_hours_proof (kate mark pat : ℕ) 
  (h1 : pat = 2 * kate)
  (h2 : 3 * pat = mark)
  (h3 : mark = kate + 105) :
  kate + mark + pat = 189 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_proof_l320_32016


namespace NUMINAMATH_CALUDE_pyramid_cases_l320_32008

/-- The sum of the first n triangular numbers -/
def sum_triangular (n : ℕ) : ℕ :=
  (n * (n + 1) * (n + 2)) / 6

/-- The pyramid has 6 levels -/
def pyramid_levels : ℕ := 6

theorem pyramid_cases : sum_triangular pyramid_levels = 56 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_cases_l320_32008


namespace NUMINAMATH_CALUDE_disk_contains_origin_l320_32035

theorem disk_contains_origin (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ a b c : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₄ > 0) (h₃ : y₁ > 0) (h₄ : y₂ > 0)
  (h₅ : x₂ < 0) (h₆ : x₃ < 0) (h₇ : y₃ < 0) (h₈ : y₄ < 0)
  (h₉ : (x₁ - a)^2 + (y₁ - b)^2 ≤ c^2)
  (h₁₀ : (x₂ - a)^2 + (y₂ - b)^2 ≤ c^2)
  (h₁₁ : (x₃ - a)^2 + (y₃ - b)^2 ≤ c^2)
  (h₁₂ : (x₄ - a)^2 + (y₄ - b)^2 ≤ c^2) :
  a^2 + b^2 ≤ c^2 := by
sorry

end NUMINAMATH_CALUDE_disk_contains_origin_l320_32035


namespace NUMINAMATH_CALUDE_female_listeners_l320_32000

theorem female_listeners (total_listeners male_listeners : ℕ) 
  (h1 : total_listeners = 180) 
  (h2 : male_listeners = 80) : 
  total_listeners - male_listeners = 100 := by
  sorry

end NUMINAMATH_CALUDE_female_listeners_l320_32000


namespace NUMINAMATH_CALUDE_tan_alpha_neg_three_l320_32005

theorem tan_alpha_neg_three (α : ℝ) (h : Real.tan α = -3) :
  (Real.cos α + 2 * Real.sin α) / (Real.cos α - 3 * Real.sin α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_neg_three_l320_32005


namespace NUMINAMATH_CALUDE_invitation_combinations_l320_32002

theorem invitation_combinations (n m : ℕ) (h : n = 10 ∧ m = 6) : 
  (Nat.choose n m) - (Nat.choose (n - 2) (m - 2)) = 140 :=
sorry

end NUMINAMATH_CALUDE_invitation_combinations_l320_32002


namespace NUMINAMATH_CALUDE_no_three_way_partition_of_positive_integers_l320_32099

theorem no_three_way_partition_of_positive_integers :
  ¬ ∃ (A B C : Set ℕ+),
    (A ∪ B ∪ C = Set.univ) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
    (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧
    (∀ x ∈ A, ∀ y ∈ B, (x^2 - x*y + y^2) ∈ C) ∧
    (∀ x ∈ B, ∀ y ∈ C, (x^2 - x*y + y^2) ∈ A) ∧
    (∀ x ∈ C, ∀ y ∈ A, (x^2 - x*y + y^2) ∈ B) :=
by sorry

end NUMINAMATH_CALUDE_no_three_way_partition_of_positive_integers_l320_32099


namespace NUMINAMATH_CALUDE_garden_minimum_width_l320_32045

theorem garden_minimum_width :
  ∀ w : ℝ,
  w > 0 →
  w * (w + 12) ≥ 120 →
  w ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_garden_minimum_width_l320_32045


namespace NUMINAMATH_CALUDE_batsman_highest_score_l320_32027

-- Define the given conditions
def total_innings : ℕ := 46
def average : ℚ := 62
def score_difference : ℕ := 150
def average_excluding_extremes : ℚ := 58

-- Define the theorem
theorem batsman_highest_score :
  ∃ (highest lowest : ℕ),
    (highest - lowest = score_difference) ∧
    (highest + lowest = total_innings * average - (total_innings - 2) * average_excluding_extremes) ∧
    (highest = 225) := by
  sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l320_32027


namespace NUMINAMATH_CALUDE_projection_matrix_values_l320_32034

-- Define the matrix P
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, 20/49],
    ![c, 29/49]]

-- Define the property of being a projection matrix
def is_projection_matrix (M : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  M * M = M

-- Theorem statement
theorem projection_matrix_values :
  ∀ a c : ℚ, is_projection_matrix (P a c) → a = 41/49 ∧ c = 204/1225 :=
by sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l320_32034


namespace NUMINAMATH_CALUDE_adams_change_l320_32072

/-- Given that Adam has $5 and an airplane costs $4.28, prove that the change Adam will receive is $0.72. -/
theorem adams_change (adam_money : ℝ) (airplane_cost : ℝ) (change : ℝ) 
  (h1 : adam_money = 5)
  (h2 : airplane_cost = 4.28)
  (h3 : change = adam_money - airplane_cost) :
  change = 0.72 := by
sorry

end NUMINAMATH_CALUDE_adams_change_l320_32072


namespace NUMINAMATH_CALUDE_function_properties_l320_32040

noncomputable section

variables (a b : ℝ) (x : ℝ)

def f (x : ℝ) := -a * x + b + a * x * Real.log x

theorem function_properties :
  a ≠ 0 →
  f e = 2 →
  (b = 2) ∧
  (a > 0 →
    (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂)) ∧
  (a < 0 →
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂) ∧
    (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂)) :=
by sorry

end

end NUMINAMATH_CALUDE_function_properties_l320_32040


namespace NUMINAMATH_CALUDE_always_true_inequality_l320_32078

theorem always_true_inequality (a b x y : ℝ) (h1 : x < a) (h2 : y < b) : x * y < a * b := by
  sorry

end NUMINAMATH_CALUDE_always_true_inequality_l320_32078


namespace NUMINAMATH_CALUDE_sea_lion_penguin_ratio_l320_32015

theorem sea_lion_penguin_ratio :
  let sea_lions : ℕ := 48
  let penguins : ℕ := sea_lions + 84
  (sea_lions : ℚ) / penguins = 4 / 11 := by
sorry

end NUMINAMATH_CALUDE_sea_lion_penguin_ratio_l320_32015


namespace NUMINAMATH_CALUDE_laws_in_concept_l320_32019

/-- The probability that exactly M laws are included in the Concept -/
def prob_exactly_M (K N M : ℕ) (p : ℝ) : ℝ :=
  Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)

/-- The expected number of laws included in the Concept -/
def expected_laws (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

/-- Theorem stating the probability of exactly M laws being included and the expected number of laws -/
theorem laws_in_concept (K N M : ℕ) (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : M ≤ K) :
  (prob_exactly_M K N M p = Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)) ∧
  (expected_laws K N p = K * (1 - (1 - p)^N)) := by
  sorry

#check laws_in_concept

end NUMINAMATH_CALUDE_laws_in_concept_l320_32019


namespace NUMINAMATH_CALUDE_no_solution_exists_l320_32022

theorem no_solution_exists : ¬∃ x : ℝ, (|x^2 - 14*x + 40| = 3) ∧ (x^2 - 14*x + 45 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l320_32022


namespace NUMINAMATH_CALUDE_music_school_population_l320_32093

/-- Given a music school with boys, girls, and teachers, prove that the total number of people is 9b/7, where b is the number of boys. -/
theorem music_school_population (b g t : ℚ) : 
  b = 4 * g ∧ g = 7 * t → b + g + t = 9 * b / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_music_school_population_l320_32093


namespace NUMINAMATH_CALUDE_weighted_average_is_correct_l320_32070

def english_score : Rat := 76 / 120
def english_weight : Nat := 2

def math_score : Rat := 65 / 150
def math_weight : Nat := 3

def physics_score : Rat := 82 / 100
def physics_weight : Nat := 2

def chemistry_score : Rat := 67 / 80
def chemistry_weight : Nat := 1

def biology_score : Rat := 85 / 100
def biology_weight : Nat := 2

def history_score : Rat := 92 / 150
def history_weight : Nat := 1

def geography_score : Rat := 58 / 75
def geography_weight : Nat := 1

def total_weight : Nat := english_weight + math_weight + physics_weight + chemistry_weight + biology_weight + history_weight + geography_weight

def weighted_average_score : Rat :=
  (english_score * english_weight +
   math_score * math_weight +
   physics_score * physics_weight +
   chemistry_score * chemistry_weight +
   biology_score * biology_weight +
   history_score * history_weight +
   geography_score * geography_weight) / total_weight

theorem weighted_average_is_correct : weighted_average_score = 67755 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_is_correct_l320_32070


namespace NUMINAMATH_CALUDE_factor_of_expression_l320_32024

theorem factor_of_expression (x y z : ℝ) :
  ∃ (f : ℝ → ℝ → ℝ → ℝ), (x^2 - y^2 - z^2 + 2*y*z + x + y - z) = (x - y + z + 1) * f x y z := by
  sorry

end NUMINAMATH_CALUDE_factor_of_expression_l320_32024


namespace NUMINAMATH_CALUDE_cradle_cup_d_score_l320_32003

/-- Represents the scores of the five participants in the "Cradle Cup" math competition. -/
structure CradleCupScores where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ

/-- The conditions of the "Cradle Cup" math competition. -/
def CradleCupConditions (scores : CradleCupScores) : Prop :=
  scores.A = 94 ∧
  scores.B ≥ scores.A ∧ scores.B ≥ scores.C ∧ scores.B ≥ scores.D ∧ scores.B ≥ scores.E ∧
  scores.C = (scores.A + scores.D) / 2 ∧
  5 * scores.D = scores.A + scores.B + scores.C + scores.D + scores.E ∧
  scores.E = scores.C + 2 ∧
  scores.B ≤ 100 ∧ scores.C ≤ 100 ∧ scores.D ≤ 100 ∧ scores.E ≤ 100

/-- The theorem stating that given the conditions of the "Cradle Cup" math competition,
    participant D must have scored 96 points. -/
theorem cradle_cup_d_score (scores : CradleCupScores) :
  CradleCupConditions scores → scores.D = 96 :=
by sorry

end NUMINAMATH_CALUDE_cradle_cup_d_score_l320_32003


namespace NUMINAMATH_CALUDE_lcm_of_5_6_10_12_l320_32060

theorem lcm_of_5_6_10_12 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 12)) = 60 := by sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_10_12_l320_32060


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l320_32054

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 5 → b = 5 → c = 2 →
  (a = b) →  -- isosceles condition
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →  -- triangle inequality
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l320_32054


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l320_32009

theorem rectangle_area_problem (w : ℝ) (L L' A : ℝ) : 
  w = 10 →                      -- Width is 10 m
  A = L * w →                   -- Original area
  L' * w = (4/3) * A →          -- New area is 1 1/3 times original
  2 * L' + 2 * w = 60 →         -- New perimeter is 60 m
  A = 150                       -- Original area is 150 square meters
:= by sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l320_32009


namespace NUMINAMATH_CALUDE_representation_bound_l320_32052

def f (n : ℕ) : ℕ := sorry

theorem representation_bound (n : ℕ) (h : n ≥ 3) :
  (2 : ℝ) ^ (n^2/4) < (f (2^n) : ℝ) ∧ (f (2^n) : ℝ) < (2 : ℝ) ^ (n^2/2) := by
  sorry

end NUMINAMATH_CALUDE_representation_bound_l320_32052


namespace NUMINAMATH_CALUDE_fraction_irreducible_l320_32039

theorem fraction_irreducible (n : ℤ) : Nat.gcd (2 * n^2 + 9 * n - 17).natAbs (n + 6).natAbs = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l320_32039


namespace NUMINAMATH_CALUDE_imaginary_power_difference_l320_32095

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_difference : i^23 - i^210 = -i + 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_power_difference_l320_32095


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_l320_32011

theorem reciprocal_of_negative_one :
  ∃ x : ℚ, x * (-1) = 1 ∧ x = -1 :=
sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_l320_32011


namespace NUMINAMATH_CALUDE_expression_simplification_l320_32013

theorem expression_simplification (a : ℝ) 
  (h1 : a ≠ -8) 
  (h2 : a ≠ 1) 
  (h3 : a ≠ -1) : 
  (9 / (a + 8) - (a^(1/3) + 2) / (a^(2/3) - 2*a^(1/3) + 4)) * 
  ((a^(4/3) + 8*a^(1/3)) / (1 - a^(2/3))) + 
  (5 - a^(2/3)) / (1 + a^(1/3)) = 5 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l320_32013


namespace NUMINAMATH_CALUDE_post_office_mail_count_l320_32043

/-- The number of pieces of mail handled by a post office in six months -/
def mail_in_six_months (letters_per_day : ℕ) (packages_per_day : ℕ) (days_per_month : ℕ) (num_months : ℕ) : ℕ :=
  (letters_per_day + packages_per_day) * (days_per_month * num_months)

/-- Theorem stating that the post office handles 14,400 pieces of mail in six months -/
theorem post_office_mail_count :
  mail_in_six_months 60 20 30 6 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_post_office_mail_count_l320_32043


namespace NUMINAMATH_CALUDE_problem_solution_l320_32091

theorem problem_solution (x y : ℚ) : 
  x = 152 → 
  x^3*y - 3*x^2*y + 3*x*y = 912000 → 
  y = 3947/15200 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l320_32091


namespace NUMINAMATH_CALUDE_three_digit_sum_theorem_l320_32068

/-- The sum of all three-digit numbers -/
def sum_three_digit : ℕ := 494550

/-- Predicate to check if a number is three-digit -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem three_digit_sum_theorem (x y : ℕ) :
  is_three_digit x ∧ is_three_digit y ∧ 
  sum_three_digit - x - y = 600 * x →
  x = 823 ∧ y = 527 := by
sorry

end NUMINAMATH_CALUDE_three_digit_sum_theorem_l320_32068


namespace NUMINAMATH_CALUDE_A_equals_B_l320_32077

def A (a : ℕ) : Set ℕ :=
  {k : ℕ | ∃ x y : ℤ, x > Real.sqrt a ∧ k = (x^2 - a) / (x^2 - y^2)}

def B (a : ℕ) : Set ℕ :=
  {k : ℕ | ∃ x y : ℤ, 0 ≤ x ∧ x < Real.sqrt a ∧ k = (x^2 - a) / (x^2 - y^2)}

theorem A_equals_B (a : ℕ) (h : ¬ ∃ n : ℕ, n^2 = a) : A a = B a := by
  sorry

end NUMINAMATH_CALUDE_A_equals_B_l320_32077


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l320_32062

theorem parabola_focus_distance (p : ℝ) (h1 : p > 0) :
  ∃ (x y : ℝ),
    y^2 = 2*p*x ∧
    x + p/2 = 2 →
    Real.sqrt (x - p/2)^2 + y^2 = Real.sqrt (2*p*(2 - p/2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l320_32062


namespace NUMINAMATH_CALUDE_solve_linear_equation_l320_32048

theorem solve_linear_equation (x : ℤ) : 9823 + x = 13200 → x = 3377 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l320_32048


namespace NUMINAMATH_CALUDE_prob_three_non_defective_pencils_l320_32084

/-- The probability of selecting 3 non-defective pencils from a box of 8 pencils, where 2 are defective. -/
theorem prob_three_non_defective_pencils :
  let total_pencils : ℕ := 8
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils := total_pencils - defective_pencils
  Nat.choose non_defective_pencils selected_pencils / Nat.choose total_pencils selected_pencils = 5 / 14 :=
by sorry

end NUMINAMATH_CALUDE_prob_three_non_defective_pencils_l320_32084


namespace NUMINAMATH_CALUDE_renovation_theorem_l320_32042

/-- Represents a city grid --/
structure CityGrid where
  rows : Nat
  cols : Nat

/-- Calculates the minimum number of buildings after renovation --/
def minBuildingsAfterRenovation (grid : CityGrid) : Nat :=
  sorry

theorem renovation_theorem :
  (minBuildingsAfterRenovation ⟨20, 20⟩ = 25) ∧
  (minBuildingsAfterRenovation ⟨50, 90⟩ = 282) := by
  sorry

end NUMINAMATH_CALUDE_renovation_theorem_l320_32042


namespace NUMINAMATH_CALUDE_fraction_sum_equals_two_l320_32059

theorem fraction_sum_equals_two : 
  (1 : ℚ) / 2 + (1 : ℚ) / 2 + (1 : ℚ) / 3 + (1 : ℚ) / 3 + (1 : ℚ) / 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_two_l320_32059


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l320_32031

theorem compare_negative_fractions : -2/3 > -5/7 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l320_32031


namespace NUMINAMATH_CALUDE_area_of_region_l320_32020

/-- The area of the region defined by x^2 + y^2 + 8x - 18y = 0 is 97π -/
theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 97 ∧ 
   A = Real.pi * (Real.sqrt ((x + 4)^2 + (y - 9)^2)) ^ 2 ∧
   x^2 + y^2 + 8*x - 18*y = 0) := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l320_32020


namespace NUMINAMATH_CALUDE_remainder_theorem_l320_32044

theorem remainder_theorem : ∃ q : ℕ, 3^303 + 303 = q * (3^151 + 3^75 + 1) + 294 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l320_32044


namespace NUMINAMATH_CALUDE_clock_hands_coincidence_time_l320_32010

/-- Represents the state of a clock's hands -/
structure ClockState where
  minute_angle : ℝ
  hour_angle : ℝ

/-- Represents the movement rates of clock hands -/
structure ClockRates where
  minute_rate : ℝ
  hour_rate : ℝ

/-- Calculates the time taken for clock hands to move from one state to another -/
def time_between_states (initial : ClockState) (final : ClockState) (rates : ClockRates) : ℝ :=
  sorry

theorem clock_hands_coincidence_time :
  let initial_state : ClockState := { minute_angle := 0, hour_angle := 180 }
  let final_state : ClockState := { minute_angle := 0, hour_angle := 0 }
  let rates : ClockRates := { minute_rate := 6, hour_rate := 0.5 }
  let time := time_between_states initial_state final_state rates
  time = 360 ∧ time < 12 * 60 := by sorry

end NUMINAMATH_CALUDE_clock_hands_coincidence_time_l320_32010


namespace NUMINAMATH_CALUDE_fundraiser_percentage_increase_l320_32090

def fundraiser (initial_rate : ℝ) (total_hours : ℕ) (initial_hours : ℕ) (total_amount : ℝ) : Prop :=
  let remaining_hours := total_hours - initial_hours
  let initial_amount := initial_rate * initial_hours
  let remaining_amount := total_amount - initial_amount
  let new_rate := remaining_amount / remaining_hours
  let percentage_increase := (new_rate - initial_rate) / initial_rate * 100
  percentage_increase = 20

theorem fundraiser_percentage_increase :
  fundraiser 5000 26 12 144000 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_percentage_increase_l320_32090
