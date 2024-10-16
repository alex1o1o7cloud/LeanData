import Mathlib

namespace NUMINAMATH_CALUDE_tetrahedron_inscribed_circumscribed_inequality_l2544_254456

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The inscribed sphere of a tetrahedron -/
def inscribedSphere (t : Tetrahedron) : Sphere := sorry

/-- The circumscribed sphere of a tetrahedron -/
def circumscribedSphere (t : Tetrahedron) : Sphere := sorry

/-- The intersection of the planes of the remaining faces -/
def planesIntersection (t : Tetrahedron) : Point3D := sorry

/-- The intersection of a line segment with a sphere -/
def lineIntersectSphere (p1 p2 : Point3D) (s : Sphere) : Point3D := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

theorem tetrahedron_inscribed_circumscribed_inequality (t : Tetrahedron) :
  let I := (inscribedSphere t).center
  let J := planesIntersection t
  let K := lineIntersectSphere I J (circumscribedSphere t)
  distance I K > distance J K := by sorry

end NUMINAMATH_CALUDE_tetrahedron_inscribed_circumscribed_inequality_l2544_254456


namespace NUMINAMATH_CALUDE_exactly_two_win_probability_l2544_254468

/-- The probability that exactly two out of three players win a game, given their individual probabilities of success. -/
theorem exactly_two_win_probability 
  (p_alice : ℚ) 
  (p_benjamin : ℚ) 
  (p_carol : ℚ) 
  (h_alice : p_alice = 1/5) 
  (h_benjamin : p_benjamin = 3/8) 
  (h_carol : p_carol = 2/7) : 
  (p_alice * p_benjamin * (1 - p_carol) + 
   p_alice * p_carol * (1 - p_benjamin) + 
   p_benjamin * p_carol * (1 - p_alice)) = 49/280 := by
sorry


end NUMINAMATH_CALUDE_exactly_two_win_probability_l2544_254468


namespace NUMINAMATH_CALUDE_trick_deck_cost_l2544_254463

/-- The cost of a trick deck satisfies the given conditions -/
theorem trick_deck_cost : ∃ (cost : ℕ), 
  6 * cost + 2 * cost = 64 ∧ cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_trick_deck_cost_l2544_254463


namespace NUMINAMATH_CALUDE_cross_section_distance_l2544_254428

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Side length of the base hexagon -/
  base_side : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- Distance from the apex of the pyramid -/
  distance_from_apex : ℝ
  /-- Area of the cross section -/
  area : ℝ

/-- Theorem about the distance of a cross section in a right hexagonal pyramid -/
theorem cross_section_distance
  (pyramid : RightHexagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h_parallel : cs1.distance_from_apex < cs2.distance_from_apex)
  (h_areas : cs1.area = 150 * Real.sqrt 3 ∧ cs2.area = 600 * Real.sqrt 3)
  (h_distance : cs2.distance_from_apex - cs1.distance_from_apex = 8) :
  cs2.distance_from_apex = 16 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_distance_l2544_254428


namespace NUMINAMATH_CALUDE_ellipse_line_slope_l2544_254455

/-- An ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e_eq : e = 1/2

/-- A line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  h_slope_pos : 0 < slope

/-- Radii of incircles of triangles formed by points on the ellipse and foci -/
structure IncircleRadii where
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  h_radii_rel : r₁ + r₃ = 2 * r₂

/-- The main theorem statement -/
theorem ellipse_line_slope (E : Ellipse) (l : Line) (R : IncircleRadii) :
  l.slope = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_line_slope_l2544_254455


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l2544_254430

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {3, 4, 5}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem statement
theorem complement_intersection_equals_set :
  (U \ N) ∩ M = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l2544_254430


namespace NUMINAMATH_CALUDE_heartsuit_inequality_l2544_254489

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem statement
theorem heartsuit_inequality : ∃ x y : ℝ, 3 * (heartsuit x y) ≠ heartsuit (3*x) y := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_inequality_l2544_254489


namespace NUMINAMATH_CALUDE_solution_difference_l2544_254448

theorem solution_difference (a b : ℝ) : 
  (∀ x : ℝ, (3 * x - 9) / (x^2 + 3 * x - 18) = x + 1 ↔ x = a ∨ x = b) →
  a ≠ b →
  a > b →
  a - b = 1 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l2544_254448


namespace NUMINAMATH_CALUDE_angle_sum_l2544_254459

theorem angle_sum (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.sin (α - β) = 5/6) (h4 : Real.tan α / Real.tan β = -1/4) :
  α + β = 7 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_l2544_254459


namespace NUMINAMATH_CALUDE_second_box_weight_l2544_254418

/-- The weight of the second box in a set of three boxes -/
def weight_of_second_box (weight_first weight_last total_weight : ℕ) : ℕ :=
  total_weight - weight_first - weight_last

/-- Theorem: The weight of the second box is 11 pounds -/
theorem second_box_weight :
  weight_of_second_box 2 5 18 = 11 := by
  sorry

end NUMINAMATH_CALUDE_second_box_weight_l2544_254418


namespace NUMINAMATH_CALUDE_parallelogram_diagonals_fixed_points_l2544_254451

/-- A line in a 2D plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A parallelogram in a 2D plane -/
structure Parallelogram :=
  (a b c d : Point)

/-- The diagonal of a parallelogram -/
def diagonal (p : Parallelogram) : Line :=
  sorry

/-- Check if a point lies on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  sorry

/-- The intersection point of two lines -/
def intersection (l1 l2 : Line) : Point :=
  sorry

theorem parallelogram_diagonals_fixed_points 
  (l : Line) (A B C D : Point) 
  (hA : on_line A l) (hB : on_line B l) (hC : on_line C l) (hD : on_line D l)
  (p1 p2 : Parallelogram) 
  (hp1 : p1.a = A ∧ p1.c = B) (hp2 : p2.a = C ∧ p2.c = D) :
  ∃ (P Q : Point), 
    (on_line P l ∧ on_line Q l) ∧
    ((on_line (intersection (diagonal p1) l) l ∧ 
      (intersection (diagonal p1) l = P ∨ intersection (diagonal p1) l = Q)) ∧
     (on_line (intersection (diagonal p2) l) l ∧ 
      (intersection (diagonal p2) l = P ∨ intersection (diagonal p2) l = Q))) :=
  sorry

end NUMINAMATH_CALUDE_parallelogram_diagonals_fixed_points_l2544_254451


namespace NUMINAMATH_CALUDE_jeremy_watermelons_l2544_254492

/-- The number of watermelons Jeremy eats per week -/
def jeremy_eats_per_week : ℕ := 3

/-- The number of watermelons Jeremy gives to his dad per week -/
def jeremy_gives_dad_per_week : ℕ := 2

/-- The number of weeks the watermelons will last -/
def weeks_watermelons_last : ℕ := 6

/-- The total number of watermelons Jeremy bought -/
def total_watermelons : ℕ := 30

theorem jeremy_watermelons :
  total_watermelons = (jeremy_eats_per_week + jeremy_gives_dad_per_week) * weeks_watermelons_last :=
by sorry

end NUMINAMATH_CALUDE_jeremy_watermelons_l2544_254492


namespace NUMINAMATH_CALUDE_triangle_problem_l2544_254464

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition: 2c - a = 2b*cos(A) -/
def satisfiesCondition (t : Triangle) : Prop :=
  2 * t.c - t.a = 2 * t.b * Real.cos t.A

/-- Theorem stating the two parts of the problem -/
theorem triangle_problem (t : Triangle) 
  (h : satisfiesCondition t) : 
  t.B = π / 3 ∧ 
  (t.a = 2 ∧ t.b = Real.sqrt 7 → t.c = 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2544_254464


namespace NUMINAMATH_CALUDE_sphere_triangle_distance_l2544_254452

-- Define the sphere
def sphere_radius : ℝ := 9

-- Define the triangle
def triangle_side1 : ℝ := 20
def triangle_side2 : ℝ := 20
def triangle_side3 : ℝ := 30

-- State the theorem
theorem sphere_triangle_distance :
  let s := (triangle_side1 + triangle_side2 + triangle_side3) / 2
  let area := Real.sqrt (s * (s - triangle_side1) * (s - triangle_side2) * (s - triangle_side3))
  let inradius := area / s
  Real.sqrt (sphere_radius ^ 2 - inradius ^ 2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sphere_triangle_distance_l2544_254452


namespace NUMINAMATH_CALUDE_c_value_l2544_254485

theorem c_value (a b c : ℚ) : 
  8 = (2 / 100) * a → 
  2 = (8 / 100) * b → 
  c = b / a → 
  c = 1 / 16 := by
sorry

end NUMINAMATH_CALUDE_c_value_l2544_254485


namespace NUMINAMATH_CALUDE_inequality_of_roots_l2544_254419

theorem inequality_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_roots_l2544_254419


namespace NUMINAMATH_CALUDE_stamp_cost_theorem_l2544_254400

/-- The total cost of stamps in cents -/
def total_cost (type_a_cost type_b_cost type_c_cost : ℕ) 
               (type_a_quantity type_b_quantity type_c_quantity : ℕ) : ℕ :=
  type_a_cost * type_a_quantity + 
  type_b_cost * type_b_quantity + 
  type_c_cost * type_c_quantity

/-- Theorem: The total cost of stamps is 594 cents -/
theorem stamp_cost_theorem : 
  total_cost 34 52 73 4 6 2 = 594 := by
  sorry

end NUMINAMATH_CALUDE_stamp_cost_theorem_l2544_254400


namespace NUMINAMATH_CALUDE_lollipop_sharing_ratio_l2544_254477

theorem lollipop_sharing_ratio : 
  ∀ (total_lollipops : ℕ) (total_cost : ℚ) (shared_cost : ℚ),
  total_lollipops = 12 →
  total_cost = 3 →
  shared_cost = 3/4 →
  (shared_cost / (total_cost / total_lollipops)) / total_lollipops = 1/4 := by
sorry

end NUMINAMATH_CALUDE_lollipop_sharing_ratio_l2544_254477


namespace NUMINAMATH_CALUDE_clothing_production_solution_l2544_254404

/-- Represents the solution to the clothing production problem -/
def clothingProduction (totalFabric : ℝ) (topsPerUnit : ℝ) (pantsPerUnit : ℝ) (unitFabric : ℝ) 
  (fabricForTops : ℝ) (fabricForPants : ℝ) : Prop :=
  totalFabric > 0 ∧
  topsPerUnit > 0 ∧
  pantsPerUnit > 0 ∧
  unitFabric > 0 ∧
  fabricForTops ≥ 0 ∧
  fabricForPants ≥ 0 ∧
  fabricForTops + fabricForPants = totalFabric ∧
  (fabricForTops / unitFabric) * topsPerUnit = (fabricForPants / unitFabric) * pantsPerUnit

theorem clothing_production_solution :
  clothingProduction 600 2 3 3 360 240 := by
  sorry

end NUMINAMATH_CALUDE_clothing_production_solution_l2544_254404


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l2544_254480

def num_men : ℕ := 8
def num_women : ℕ := 4
def num_selected : ℕ := 4

theorem probability_at_least_one_woman :
  let total_people := num_men + num_women
  let prob_all_men := (num_men.choose num_selected : ℚ) / (total_people.choose num_selected : ℚ)
  (1 : ℚ) - prob_all_men = 85 / 99 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l2544_254480


namespace NUMINAMATH_CALUDE_pen_distribution_l2544_254469

theorem pen_distribution (num_pencils : ℕ) (num_students : ℕ) (num_pens : ℕ) : 
  num_pencils = 910 →
  num_students = 91 →
  num_pencils % num_students = 0 →
  num_pens % num_students = 0 →
  ∃ k : ℕ, num_pens = k * num_students :=
by sorry

end NUMINAMATH_CALUDE_pen_distribution_l2544_254469


namespace NUMINAMATH_CALUDE_square_root_and_arithmetic_square_root_l2544_254474

variable (m : ℝ)

theorem square_root_and_arithmetic_square_root :
  (∀ x : ℝ, x^2 = (5 + m)^2 → x = (5 + m) ∨ x = -(5 + m)) ∧
  (Real.sqrt ((5 + m)^2) = |5 + m|) := by
  sorry

end NUMINAMATH_CALUDE_square_root_and_arithmetic_square_root_l2544_254474


namespace NUMINAMATH_CALUDE_three_fraction_equality_l2544_254415

theorem three_fraction_equality (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hdiff : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (heq : (y + 1) / (x - z + 1) = (x + y + 2) / (z + 2) ∧ 
         (x + y + 2) / (z + 2) = (x + 1) / (y + 1)) : 
  (x + 1) / (y + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_fraction_equality_l2544_254415


namespace NUMINAMATH_CALUDE_smallest_root_of_equation_l2544_254446

theorem smallest_root_of_equation (x : ℚ) : 
  (x - 5/6)^2 + (x - 5/6)*(x - 2/3) = 0 ∧ x^2 - 2*x + 1 ≥ 0 → 
  x ≥ 5/6 ∧ (∀ y : ℚ, y < 5/6 → (y - 5/6)^2 + (y - 5/6)*(y - 2/3) ≠ 0 ∨ y^2 - 2*y + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_root_of_equation_l2544_254446


namespace NUMINAMATH_CALUDE_min_value_a_l2544_254490

theorem min_value_a : ∃ (a : ℝ),
  (∃ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) 1 ∧ 1 + 2^x + a * 4^x ≥ 0) ∧
  (∀ (b : ℝ), (∃ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) 1 ∧ 1 + 2^x + b * 4^x ≥ 0) → b ≥ a) ∧
  a = -6 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l2544_254490


namespace NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l2544_254443

theorem modulo_equivalence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2839 [ZMOD 10] ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l2544_254443


namespace NUMINAMATH_CALUDE_triangle_area_l2544_254420

def line1 (x : ℝ) : ℝ := x
def line2 (x : ℝ) : ℝ := -2 * x + 3

theorem triangle_area : 
  let x_intercept := (3 : ℝ) / 2
  let intersection_x := (1 : ℝ)
  let intersection_y := line1 intersection_x
  let base := x_intercept
  let height := intersection_y
  (1 / 2) * base * height = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2544_254420


namespace NUMINAMATH_CALUDE_expression_simplification_l2544_254479

/-- Given nonzero real numbers a, b, c, and a constant real number θ,
    define x, y, z as specified, and prove that x^2 + y^2 + z^2 - xyz = 4 -/
theorem expression_simplification 
  (a b c : ℝ) (θ : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (x : ℝ := b / c + c / b + Real.sin θ)
  (y : ℝ := a / c + c / a + Real.cos θ)
  (z : ℝ := a / b + b / a + Real.tan θ) :
  x^2 + y^2 + z^2 - x*y*z = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2544_254479


namespace NUMINAMATH_CALUDE_russian_chess_championship_games_l2544_254439

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 18 players, 153 games are played -/
theorem russian_chess_championship_games : 
  num_games 18 = 153 := by
  sorry

end NUMINAMATH_CALUDE_russian_chess_championship_games_l2544_254439


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l2544_254405

def z (a : ℝ) : ℂ := Complex.mk (a^2 - 1) (a - 2)

theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → z a = Complex.I * Complex.im (z a)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ z a = Complex.I * Complex.im (z a)) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l2544_254405


namespace NUMINAMATH_CALUDE_car_cost_l2544_254466

/-- Calculates the cost of the car given the costs of other gifts and total worth --/
theorem car_cost (ring_cost bracelet_cost total_worth : ℕ) 
  (h1 : ring_cost = 4000)
  (h2 : bracelet_cost = 2 * ring_cost)
  (h3 : total_worth = 14000) :
  total_worth - (ring_cost + bracelet_cost) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_car_cost_l2544_254466


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2544_254498

theorem rectangle_dimensions :
  ∀ w l : ℝ,
  w > 0 →
  l > 0 →
  l = 2 * w →
  w * l = (1/2) * (2 * (w + l)) →
  w = (3/2) ∧ l = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2544_254498


namespace NUMINAMATH_CALUDE_f_properties_l2544_254401

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

-- Theorem statement
theorem f_properties :
  f (f 4) = 1/2 ∧ ∀ x, f x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2544_254401


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l2544_254491

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l2544_254491


namespace NUMINAMATH_CALUDE_pie_crust_flour_redistribution_l2544_254476

theorem pie_crust_flour_redistribution 
  (initial_crusts : ℕ) 
  (initial_flour_per_crust : ℚ) 
  (new_crusts : ℕ) 
  (total_flour : ℚ) 
  (h1 : initial_crusts = 40)
  (h2 : initial_flour_per_crust = 1 / 8)
  (h3 : new_crusts = 25)
  (h4 : total_flour = initial_crusts * initial_flour_per_crust)
  : (total_flour / new_crusts : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pie_crust_flour_redistribution_l2544_254476


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l2544_254433

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l2544_254433


namespace NUMINAMATH_CALUDE_positive_numbers_l2544_254454

theorem positive_numbers (a b c : ℝ) 
  (sum_pos : a + b + c > 0)
  (sum_prod_pos : a * b + b * c + c * a > 0)
  (prod_pos : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_l2544_254454


namespace NUMINAMATH_CALUDE_king_paths_count_l2544_254460

/-- The number of paths for a king on a 7x7 chessboard -/
def numPaths : Fin 7 → Fin 7 → ℕ
| ⟨i, hi⟩, ⟨j, hj⟩ =>
  if i = 3 ∧ j = 3 then 0  -- Central cell (4,4) is forbidden
  else if i = 0 ∨ j = 0 then 1  -- First row and column
  else 
    have hi' : i - 1 < 7 := by sorry
    have hj' : j - 1 < 7 := by sorry
    numPaths ⟨i - 1, hi'⟩ ⟨j, hj⟩ + 
    numPaths ⟨i, hi⟩ ⟨j - 1, hj'⟩ + 
    numPaths ⟨i - 1, hi'⟩ ⟨j - 1, hj'⟩

/-- The theorem stating the number of paths for the king -/
theorem king_paths_count : numPaths ⟨6, by simp⟩ ⟨6, by simp⟩ = 5020 := by
  sorry

end NUMINAMATH_CALUDE_king_paths_count_l2544_254460


namespace NUMINAMATH_CALUDE_negative_power_division_l2544_254413

theorem negative_power_division : -3^7 / 3^2 = -3^5 := by sorry

end NUMINAMATH_CALUDE_negative_power_division_l2544_254413


namespace NUMINAMATH_CALUDE_eighteenth_replacement_november_l2544_254423

/-- Represents months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Converts a number of months to a Month -/
def monthsToMonth (n : ℕ) : Month :=
  match n % 12 with
  | 0 => Month.December
  | 1 => Month.January
  | 2 => Month.February
  | 3 => Month.March
  | 4 => Month.April
  | 5 => Month.May
  | 6 => Month.June
  | 7 => Month.July
  | 8 => Month.August
  | 9 => Month.September
  | 10 => Month.October
  | _ => Month.November

/-- The month of the nth wheel replacement, given a 7-month cycle starting in January -/
def wheelReplacementMonth (n : ℕ) : Month :=
  monthsToMonth ((n - 1) * 7 + 1)

theorem eighteenth_replacement_november :
  wheelReplacementMonth 18 = Month.November := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_replacement_november_l2544_254423


namespace NUMINAMATH_CALUDE_problem_solution_l2544_254412

theorem problem_solution (a b : ℕ+) (h : (a.val^3 - a.val^2 + 1) * (b.val^3 - b.val^2 + 2) = 2020) :
  10 * a.val + b.val = 53 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2544_254412


namespace NUMINAMATH_CALUDE_non_prime_sequence_300th_term_l2544_254438

/-- A sequence of positive integers with primes omitted -/
def non_prime_sequence : ℕ → ℕ := sorry

/-- The 300th term of the non-prime sequence -/
def term_300 : ℕ := 609

theorem non_prime_sequence_300th_term :
  non_prime_sequence 300 = term_300 := by sorry

end NUMINAMATH_CALUDE_non_prime_sequence_300th_term_l2544_254438


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2544_254432

/-- Given that i^2 = -1, prove that (3-2i)/(4+5i) = 2/41 - (23/41)i -/
theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - 2*i) / (4 + 5*i) = 2/41 - (23/41)*i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2544_254432


namespace NUMINAMATH_CALUDE_triangle_area_l2544_254410

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 3) (h2 : b = 5) (h3 : Real.cos C = -3/5) :
  let S := (1/2) * a * b * Real.sin C
  S = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2544_254410


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2544_254426

/-- A quadratic function f(x) with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + a

/-- The function g(x) represents f(x) - x -/
def g (a : ℝ) (x : ℝ) : ℝ := f a x - x

theorem quadratic_roots_theorem (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : g a x₁ = 0) (h₂ : g a x₂ = 0) (h₃ : 0 < x₁) (h₄ : x₁ < x₂) (h₅ : x₂ < 1) :
  (0 < a ∧ a < 3 - Real.sqrt 2) ∧ f a 0 * f a 1 - f a 0 < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2544_254426


namespace NUMINAMATH_CALUDE_digit_150_of_17_70_l2544_254470

/-- The decimal representation of 17/70 has a repeating sequence of digits. -/
def decimal_rep_17_70 : ℕ → ℕ
| 0 => 2
| 1 => 4
| n + 2 => match (n + 2) % 6 with
  | 0 => 4
  | 1 => 2
  | 2 => 8
  | 3 => 5
  | 4 => 7
  | 5 => 1
  | _ => 0  -- This case should never occur

/-- The 150th digit after the decimal point in the decimal representation of 17/70 is 4. -/
theorem digit_150_of_17_70 : decimal_rep_17_70 149 = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_17_70_l2544_254470


namespace NUMINAMATH_CALUDE_quadratic_properties_l2544_254453

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_neg : a < 0
  root_neg_one : a * (-1)^2 + b * (-1) + c = 0
  symmetry_axis : -b / (2 * a) = 1

/-- Properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  (f.a - f.b + f.c = 0) ∧
  (∀ m : ℝ, f.a * m^2 + f.b * m + f.c ≤ -4 * f.a) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → 
    f.a * x₁^2 + f.b * x₁ + f.c + 1 = 0 → 
    f.a * x₂^2 + f.b * x₂ + f.c + 1 = 0 → 
    x₁ < -1 ∧ x₂ > 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2544_254453


namespace NUMINAMATH_CALUDE_binomial_plus_three_l2544_254444

theorem binomial_plus_three : Nat.choose 6 2 + 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_binomial_plus_three_l2544_254444


namespace NUMINAMATH_CALUDE_min_value_sum_product_l2544_254431

theorem min_value_sum_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 1 → a + 2 * b ≤ x + 2 * y ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 1 ∧ x + 2 * y = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l2544_254431


namespace NUMINAMATH_CALUDE_prob_six_queen_is_4_663_l2544_254483

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of sixes in a standard deck -/
def NumSixes : ℕ := 4

/-- Number of queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Probability of drawing a 6 as the first card and a Queen as the second card -/
def ProbSixQueen : ℚ := (NumSixes : ℚ) / StandardDeck * NumQueens / (StandardDeck - 1)

theorem prob_six_queen_is_4_663 : ProbSixQueen = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_six_queen_is_4_663_l2544_254483


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2544_254458

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |2*x - 8| = 5 - x ↔ x = 13/3 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2544_254458


namespace NUMINAMATH_CALUDE_joan_found_79_seashells_l2544_254427

/-- The number of seashells Mike gave to Joan -/
def mike_seashells : ℕ := 63

/-- The total number of seashells Joan has -/
def total_seashells : ℕ := 142

/-- The number of seashells Joan found initially -/
def joan_initial_seashells : ℕ := total_seashells - mike_seashells

theorem joan_found_79_seashells : joan_initial_seashells = 79 := by sorry

end NUMINAMATH_CALUDE_joan_found_79_seashells_l2544_254427


namespace NUMINAMATH_CALUDE_sample_size_is_300_l2544_254425

/-- Represents the population ratios of the districts -/
def district_ratios : List ℕ := [2, 3, 5, 2, 6]

/-- The number of individuals contributed by the largest district -/
def largest_district_contribution : ℕ := 100

/-- Calculates the total sample size based on the district ratios and the contribution of the largest district -/
def calculate_sample_size (ratios : List ℕ) (largest_contribution : ℕ) : ℕ :=
  let total_ratio := ratios.sum
  let largest_ratio := ratios.maximum?
  match largest_ratio with
  | some max_ratio => (total_ratio * largest_contribution) / max_ratio
  | none => 0

/-- Theorem stating that the calculated sample size is 300 -/
theorem sample_size_is_300 :
  calculate_sample_size district_ratios largest_district_contribution = 300 := by
  sorry

#eval calculate_sample_size district_ratios largest_district_contribution

end NUMINAMATH_CALUDE_sample_size_is_300_l2544_254425


namespace NUMINAMATH_CALUDE_alcohol_amount_l2544_254422

/-- Represents the amount of alcohol in liters -/
def alcohol : ℝ := 14

/-- Represents the amount of water in liters -/
def water : ℝ := 10.5

/-- The amount of water added to the mixture in liters -/
def water_added : ℝ := 7

/-- The initial ratio of alcohol to water -/
def initial_ratio : ℚ := 4/3

/-- The final ratio of alcohol to water after adding more water -/
def final_ratio : ℚ := 4/5

theorem alcohol_amount :
  (alcohol / water = initial_ratio) ∧
  (alcohol / (water + water_added) = final_ratio) →
  alcohol = 14 := by
sorry

end NUMINAMATH_CALUDE_alcohol_amount_l2544_254422


namespace NUMINAMATH_CALUDE_complex_number_problem_l2544_254429

theorem complex_number_problem (z : ℂ) (hz : z ≠ 0) :
  Complex.abs (z + 2) = 2 ∧ (z + 4 / z).im = 0 →
  z = -1 + Complex.I * Real.sqrt 3 ∨ z = -1 - Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2544_254429


namespace NUMINAMATH_CALUDE_f_properties_l2544_254450

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - 2 * (a ^ x) - (a ^ (2 * x))

theorem f_properties (a : ℝ) (h_a : a > 1) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y ∧ y < 1) ∧
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-2) 1 ∧ f a x₀ = -7 →
    a = 2 ∧ ∃ x_max : ℝ, x_max ∈ Set.Icc (-2) 1 ∧ f a x_max = 7/16 ∧
    ∀ x : ℝ, x ∈ Set.Icc (-2) 1 → f a x ≤ 7/16) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2544_254450


namespace NUMINAMATH_CALUDE_units_digit_difference_l2544_254441

/-- Returns the units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Predicate for a natural number being even -/
def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

theorem units_digit_difference (p : ℕ) 
  (h1 : p > 0)
  (h2 : isEven p)
  (h3 : unitsDigit p > 0)
  (h4 : unitsDigit (p + 1) = 7) :
  unitsDigit (p^3) - unitsDigit (p^2) = 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_difference_l2544_254441


namespace NUMINAMATH_CALUDE_bea_earned_more_than_dawn_l2544_254462

/-- Calculates the difference in earnings between two lemonade sellers -/
def lemonade_earnings_difference (bea_price bea_sold dawn_price dawn_sold : ℕ) : ℕ :=
  bea_price * bea_sold - dawn_price * dawn_sold

/-- Proves that Bea earned 26 cents more than Dawn given the conditions -/
theorem bea_earned_more_than_dawn :
  lemonade_earnings_difference 25 10 28 8 = 26 := by
  sorry

end NUMINAMATH_CALUDE_bea_earned_more_than_dawn_l2544_254462


namespace NUMINAMATH_CALUDE_polygon_sides_l2544_254436

theorem polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → 
  (exterior_angle = 30) → 
  (n * exterior_angle = 360) → 
  n = 12 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_l2544_254436


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2544_254414

theorem solution_set_inequality (x : ℝ) : 
  (0 < x ∧ x < 2) ↔ (4 / x > |x| ∧ x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2544_254414


namespace NUMINAMATH_CALUDE_rotation_sum_110_l2544_254440

/-- A structure representing a triangle in a 2D coordinate plane -/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The rotation parameters -/
structure RotationParams where
  n : ℝ
  u : ℝ
  v : ℝ

/-- Predicate to check if a rotation transforms one triangle to another -/
def rotates (t1 t2 : Triangle) (r : RotationParams) : Prop :=
  sorry  -- Definition of rotation transformation

theorem rotation_sum_110 (DEF D'E'F' : Triangle) (r : RotationParams) :
  DEF.D = (0, 0) →
  DEF.E = (0, 10) →
  DEF.F = (20, 0) →
  D'E'F'.D = (30, 20) →
  D'E'F'.E = (40, 20) →
  D'E'F'.F = (30, 6) →
  0 < r.n →
  r.n < 180 →
  rotates DEF D'E'F' r →
  r.n + r.u + r.v = 110 := by
  sorry

#check rotation_sum_110

end NUMINAMATH_CALUDE_rotation_sum_110_l2544_254440


namespace NUMINAMATH_CALUDE_students_with_dogs_l2544_254496

theorem students_with_dogs 
  (total_students : ℕ) 
  (girls_percentage : ℚ) 
  (boys_percentage : ℚ) 
  (girls_with_dogs_percentage : ℚ) 
  (boys_with_dogs_percentage : ℚ) :
  total_students = 100 →
  girls_percentage = 1/2 →
  boys_percentage = 1/2 →
  girls_with_dogs_percentage = 1/5 →
  boys_with_dogs_percentage = 1/10 →
  (girls_percentage * total_students * girls_with_dogs_percentage +
   boys_percentage * total_students * boys_with_dogs_percentage : ℚ) = 15 := by
sorry

end NUMINAMATH_CALUDE_students_with_dogs_l2544_254496


namespace NUMINAMATH_CALUDE_pizza_problem_l2544_254482

/-- The sum of a geometric series with first term a, common ratio r, and n terms -/
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The fraction of pizza eaten after n trips to the refrigerator -/
def pizza_eaten (n : ℕ) : ℚ :=
  geometric_sum (1/3) (1/3) n

theorem pizza_problem : pizza_eaten 6 = 364/729 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l2544_254482


namespace NUMINAMATH_CALUDE_second_meeting_time_l2544_254407

/-- The time (in seconds) it takes for the racing magic to complete one round -/
def racing_magic_time : ℕ := 60

/-- The time (in seconds) it takes for the charging bull to complete one round -/
def charging_bull_time : ℕ := 90

/-- The time (in minutes) it takes for both objects to meet at the starting point for the second time -/
def meeting_time : ℕ := 3

/-- Theorem stating that the meeting time is correct given the individual round times -/
theorem second_meeting_time (racing_time : ℕ) (bull_time : ℕ) (meet_time : ℕ) 
  (h1 : racing_time = racing_magic_time)
  (h2 : bull_time = charging_bull_time)
  (h3 : meet_time = meeting_time) :
  Nat.lcm racing_time bull_time = meet_time * 60 := by
  sorry

end NUMINAMATH_CALUDE_second_meeting_time_l2544_254407


namespace NUMINAMATH_CALUDE_odd_cube_plus_23_divisible_by_24_l2544_254472

theorem odd_cube_plus_23_divisible_by_24 (n : ℤ) (h : Odd n) : 
  ∃ k : ℤ, n^3 + 23*n = 24*k := by
sorry

end NUMINAMATH_CALUDE_odd_cube_plus_23_divisible_by_24_l2544_254472


namespace NUMINAMATH_CALUDE_max_area_triangle_l2544_254467

/-- Given points A, B, C, and P in a plane with specific distances, 
    prove that the maximum possible area of triangle ABC is 18.5 -/
theorem max_area_triangle (A B C P : ℝ × ℝ) : 
  let PA := Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)
  let PB := Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)
  let PC := Real.sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  PA = 5 ∧ PB = 4 ∧ PC = 3 ∧ BC = 5 →
  (∀ A' : ℝ × ℝ, 
    let PA' := Real.sqrt ((A'.1 - P.1)^2 + (A'.2 - P.2)^2)
    PA' = 5 →
    let area := abs ((A'.1 - B.1) * (C.2 - B.2) - (A'.2 - B.2) * (C.1 - B.1)) / 2
    area ≤ 18.5) :=
by sorry


end NUMINAMATH_CALUDE_max_area_triangle_l2544_254467


namespace NUMINAMATH_CALUDE_product_of_one_plus_roots_l2544_254416

theorem product_of_one_plus_roots (p q r : ℝ) : 
  p^3 - 15*p^2 + 25*p - 10 = 0 →
  q^3 - 15*q^2 + 25*q - 10 = 0 →
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  (1 + p) * (1 + q) * (1 + r) = 51 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_roots_l2544_254416


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l2544_254465

noncomputable def P (α β : ℝ) (x : ℝ) : ℝ := α * x^4 + α * x^3 + α * x^2 + α * x + β

noncomputable def Q (α : ℝ) (x : ℝ) : ℝ := α * x^3 + α * x

theorem polynomial_equation_solution (α β : ℝ) (hα : α ≠ 0) :
  (∀ x : ℝ, P α β (x^2) + Q α x = P α β x + x^5 * Q α x) ∧
  (∀ P' Q' : ℝ → ℝ, (∀ x : ℝ, P' (x^2) + Q' x = P' x + x^5 * Q' x) →
    (∃ c : ℝ, P' = P (c * α) (c * β) ∧ Q' = Q (c * α))) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l2544_254465


namespace NUMINAMATH_CALUDE_sin_squared_sum_three_angles_l2544_254495

theorem sin_squared_sum_three_angles (α : ℝ) : 
  (Real.sin (α - Real.pi / 3))^2 + (Real.sin α)^2 + (Real.sin (α + Real.pi / 3))^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_sum_three_angles_l2544_254495


namespace NUMINAMATH_CALUDE_solve_equation_l2544_254402

theorem solve_equation (x : ℝ) : 
  (x^4)^(1/3) = 32 * 32^(1/12) → x = 16 * 2^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2544_254402


namespace NUMINAMATH_CALUDE_kenny_monday_jumping_jacks_l2544_254478

/-- Represents the number of jumping jacks Kenny did on each day of the week. -/
structure WeeklyJumpingJacks where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total number of jumping jacks for the week. -/
def totalJumpingJacks (week : WeeklyJumpingJacks) : ℕ :=
  week.sunday + week.monday + week.tuesday + week.wednesday + week.thursday + week.friday + week.saturday

/-- Theorem stating that Kenny must have done 20 jumping jacks on Monday. -/
theorem kenny_monday_jumping_jacks :
  ∃ (this_week : WeeklyJumpingJacks),
    this_week.sunday = 34 ∧
    this_week.tuesday = 0 ∧
    this_week.wednesday = 123 ∧
    this_week.thursday = 64 ∧
    this_week.friday = 23 ∧
    this_week.saturday = 61 ∧
    totalJumpingJacks this_week = 325 ∧
    this_week.monday = 20 := by
  sorry

#check kenny_monday_jumping_jacks

end NUMINAMATH_CALUDE_kenny_monday_jumping_jacks_l2544_254478


namespace NUMINAMATH_CALUDE_product_of_numbers_l2544_254457

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 218) : x * y = 13 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2544_254457


namespace NUMINAMATH_CALUDE_olympiad_colors_l2544_254488

-- Define the colors
inductive Color
  | Red
  | Yellow
  | Blue

-- Define a person's outfit
structure Outfit :=
  (dress : Color)
  (notebook : Color)

-- Define the problem statement
theorem olympiad_colors :
  ∃ (sveta tanya ira : Outfit),
    -- All dress colors are different
    sveta.dress ≠ tanya.dress ∧ sveta.dress ≠ ira.dress ∧ tanya.dress ≠ ira.dress ∧
    -- All notebook colors are different
    sveta.notebook ≠ tanya.notebook ∧ sveta.notebook ≠ ira.notebook ∧ tanya.notebook ≠ ira.notebook ∧
    -- Only Sveta's dress and notebook colors match
    (sveta.dress = sveta.notebook) ∧
    (tanya.dress ≠ tanya.notebook) ∧
    (ira.dress ≠ ira.notebook) ∧
    -- Tanya's dress and notebook are not red
    (tanya.dress ≠ Color.Red) ∧ (tanya.notebook ≠ Color.Red) ∧
    -- Ira has a yellow notebook
    (ira.notebook = Color.Yellow) ∧
    -- The solution
    sveta = Outfit.mk Color.Red Color.Red ∧
    ira = Outfit.mk Color.Blue Color.Yellow ∧
    tanya = Outfit.mk Color.Yellow Color.Blue :=
by
  sorry

end NUMINAMATH_CALUDE_olympiad_colors_l2544_254488


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2544_254406

theorem complex_equation_solution (z : ℂ) (h : (1 + 2*I)*z = 4 + 3*I) : z = 2 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2544_254406


namespace NUMINAMATH_CALUDE_tan_123_negative_l2544_254493

theorem tan_123_negative (a : ℝ) (h : Real.sin (123 * π / 180) = a) :
  Real.tan (123 * π / 180) < 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_123_negative_l2544_254493


namespace NUMINAMATH_CALUDE_shortest_paths_count_l2544_254471

/-- The number of shortest paths on an m × n grid -/
def numShortestPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) n

/-- Theorem: The number of shortest paths on an m × n grid
    from point A to point B is equal to (m+n choose n) -/
theorem shortest_paths_count (m n : ℕ) :
  numShortestPaths m n = Nat.choose (m + n) n := by
  sorry

end NUMINAMATH_CALUDE_shortest_paths_count_l2544_254471


namespace NUMINAMATH_CALUDE_special_key_102_presses_l2544_254434

def f (x : ℚ) : ℚ := 1 / (1 - x)

def iterate_f (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_f n x)

theorem special_key_102_presses :
  iterate_f 102 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_special_key_102_presses_l2544_254434


namespace NUMINAMATH_CALUDE_sum_of_rearranged_digits_l2544_254497

theorem sum_of_rearranged_digits : 1357 + 3571 + 5713 + 7135 = 17776 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rearranged_digits_l2544_254497


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l2544_254403

theorem smallest_sum_of_squares (x y : ℕ+) : 
  (x.val * (x.val + 1) ∣ y.val * (y.val + 1)) ∧ 
  (¬ (x.val ∣ y.val) ∧ ¬ (x.val ∣ (y.val + 1)) ∧ ¬ ((x.val + 1) ∣ y.val) ∧ ¬ ((x.val + 1) ∣ (y.val + 1))) →
  (∀ a b : ℕ+, 
    (a.val * (a.val + 1) ∣ b.val * (b.val + 1)) ∧ 
    (¬ (a.val ∣ b.val) ∧ ¬ (a.val ∣ (b.val + 1)) ∧ ¬ ((a.val + 1) ∣ b.val) ∧ ¬ ((a.val + 1) ∣ (b.val + 1))) →
    x.val^2 + y.val^2 ≤ a.val^2 + b.val^2) →
  x.val^2 + y.val^2 = 1421 := by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l2544_254403


namespace NUMINAMATH_CALUDE_matts_current_age_matts_age_is_65_l2544_254411

/-- Given that James turned 27 three years ago and in 5 years, Matt will be twice James' age,
    prove that Matt's current age is 65. -/
theorem matts_current_age : ℕ → Prop :=
  fun age_matt : ℕ =>
    let age_james_3_years_ago : ℕ := 27
    let years_since_james_27 : ℕ := 3
    let years_until_matt_twice_james : ℕ := 5
    let age_james : ℕ := age_james_3_years_ago + years_since_james_27
    let age_james_in_5_years : ℕ := age_james + years_until_matt_twice_james
    let age_matt_in_5_years : ℕ := 2 * age_james_in_5_years
    age_matt = age_matt_in_5_years - years_until_matt_twice_james ∧ age_matt = 65

/-- Proof of Matt's current age -/
theorem matts_age_is_65 : matts_current_age 65 := by
  sorry

end NUMINAMATH_CALUDE_matts_current_age_matts_age_is_65_l2544_254411


namespace NUMINAMATH_CALUDE_auto_finance_to_total_auto_ratio_l2544_254445

def total_consumer_credit : ℝ := 855
def auto_finance_credit : ℝ := 57
def auto_credit_percentage : ℝ := 0.20

theorem auto_finance_to_total_auto_ratio :
  let total_auto_credit := total_consumer_credit * auto_credit_percentage
  auto_finance_credit / total_auto_credit = 1/3 := by
sorry

end NUMINAMATH_CALUDE_auto_finance_to_total_auto_ratio_l2544_254445


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l2544_254481

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 5, 6}
def B : Set ℕ := {x ∈ U | x^2 - 5*x ≥ 0}

theorem intersection_complement_equal : A ∩ (U \ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l2544_254481


namespace NUMINAMATH_CALUDE_sqrt_divided_by_two_is_ten_l2544_254424

theorem sqrt_divided_by_two_is_ten (x : ℝ) : (Real.sqrt x) / 2 = 10 → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_divided_by_two_is_ten_l2544_254424


namespace NUMINAMATH_CALUDE_ten_factorial_mod_thirteen_l2544_254475

-- Define factorial for natural numbers
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem ten_factorial_mod_thirteen : factorial 10 % 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ten_factorial_mod_thirteen_l2544_254475


namespace NUMINAMATH_CALUDE_steves_return_speed_l2544_254486

/-- Proves that given a round trip of 60 km (30 km each way), where the return speed is twice 
    the outbound speed, and the total travel time is 6 hours, the return speed is 15 km/h. -/
theorem steves_return_speed 
  (distance : ℝ) 
  (total_time : ℝ) 
  (speed_to_work : ℝ) 
  (speed_from_work : ℝ) : 
  distance = 30 →
  total_time = 6 →
  speed_from_work = 2 * speed_to_work →
  distance / speed_to_work + distance / speed_from_work = total_time →
  speed_from_work = 15 := by
  sorry


end NUMINAMATH_CALUDE_steves_return_speed_l2544_254486


namespace NUMINAMATH_CALUDE_mauve_red_parts_l2544_254449

/-- Represents the composition of paint mixtures -/
structure PaintMixture where
  red : ℝ
  blue : ℝ

/-- Defines the fuchsia paint mixture -/
def fuchsia : PaintMixture := { red := 5, blue := 3 }

/-- Defines the mauve paint mixture with unknown red parts -/
def mauve (x : ℝ) : PaintMixture := { red := x, blue := 6 }

/-- Theorem stating the number of red parts in mauve paint -/
theorem mauve_red_parts : 
  ∃ (x : ℝ), 
    (16 * (fuchsia.red / (fuchsia.red + fuchsia.blue))) = 
    (x * 20 / (x + (mauve x).blue)) ∧ 
    x = 3 := by sorry

end NUMINAMATH_CALUDE_mauve_red_parts_l2544_254449


namespace NUMINAMATH_CALUDE_ray_initial_cents_l2544_254409

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of cents Ray gives to Peter -/
def cents_to_peter : ℕ := 25

/-- The number of nickels Ray has left after giving away cents -/
def nickels_left : ℕ := 4

/-- The initial number of cents Ray had -/
def initial_cents : ℕ := 95

theorem ray_initial_cents :
  initial_cents = 
    cents_to_peter + 
    (2 * cents_to_peter) + 
    (nickels_left * nickel_value) :=
by sorry

end NUMINAMATH_CALUDE_ray_initial_cents_l2544_254409


namespace NUMINAMATH_CALUDE_license_plate_count_l2544_254437

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 4

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of possible positions for the letter block -/
def block_positions : ℕ := digits_count + 1

theorem license_plate_count : 
  num_digits ^ digits_count * num_letters ^ letters_count * block_positions = 878800000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2544_254437


namespace NUMINAMATH_CALUDE_functional_equation_properties_l2544_254487

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem functional_equation_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (f 0 = 0) ∧ (f 1 = 0) ∧ (∀ x : ℝ, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l2544_254487


namespace NUMINAMATH_CALUDE_smallest_number_l2544_254421

theorem smallest_number (a b c d : ℤ) (ha : a = 0) (hb : b = -3) (hc : c = 1) (hd : d = -1) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
sorry

end NUMINAMATH_CALUDE_smallest_number_l2544_254421


namespace NUMINAMATH_CALUDE_mall_sales_optimal_profit_l2544_254499

/-- Represents the selling prices and profit calculation for products A and B --/
structure ProductSales where
  cost_price : ℝ
  price_a : ℝ
  price_b : ℝ
  sales_a : ℝ → ℝ
  sales_b : ℝ → ℝ
  profit : ℝ → ℝ

/-- The theorem statement based on the given problem --/
theorem mall_sales_optimal_profit (s : ProductSales) : 
  s.cost_price = 20 ∧ 
  20 * s.price_a + 10 * s.price_b = 840 ∧ 
  10 * s.price_a + 15 * s.price_b = 660 ∧
  s.sales_a 0 = 40 ∧
  (∀ m, s.sales_a m = s.sales_a 0 + 10 * m) ∧
  (∀ m, s.price_a - m ≥ s.price_b) ∧
  (∀ m, s.profit m = (s.price_a - m - s.cost_price) * s.sales_a m + (s.price_b - s.cost_price) * s.sales_b m) →
  s.price_a = 30 ∧ 
  s.price_b = 24 ∧ 
  (∃ m, s.sales_a m = s.sales_b m ∧ 
       s.profit m = 810 ∧ 
       ∀ n, s.profit n ≤ s.profit m) := by
  sorry

end NUMINAMATH_CALUDE_mall_sales_optimal_profit_l2544_254499


namespace NUMINAMATH_CALUDE_a_13_value_l2544_254417

/-- An arithmetic sequence with specific terms -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_13_value (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) 
  (h_a5 : a 5 = 6)
  (h_a8 : a 8 = 15) : 
  a 13 = 30 := by
sorry

end NUMINAMATH_CALUDE_a_13_value_l2544_254417


namespace NUMINAMATH_CALUDE_V₃_at_one_horner_equiv_f_l2544_254435

-- Define the polynomial f(x) = 3x^5 + 2x^3 - 8x + 5
def f (x : ℝ) : ℝ := 3 * x^5 + 2 * x^3 - 8 * x + 5

-- Define Horner's method for this polynomial
def horner (x : ℝ) : ℝ := (((((3 * x + 0) * x + 2) * x + 0) * x - 8) * x + 5)

-- Define V₃ in Horner's method
def V₃ (x : ℝ) : ℝ := ((3 * x + 0) * x + 2) * x + 0

-- Theorem: V₃(1) = 2
theorem V₃_at_one : V₃ 1 = 2 := by
  sorry

-- Prove that Horner's method is equivalent to the original polynomial
theorem horner_equiv_f : ∀ x, horner x = f x := by
  sorry

end NUMINAMATH_CALUDE_V₃_at_one_horner_equiv_f_l2544_254435


namespace NUMINAMATH_CALUDE_custom_calculator_results_l2544_254442

-- Define the custom operation *
noncomputable def customOp (a b : ℤ) : ℤ := 2 * a - b

-- Properties of the custom operation
axiom prop_i (a : ℤ) : customOp a a = a
axiom prop_ii (a : ℤ) : customOp a 0 = 2 * a
axiom prop_iii (a b c d : ℤ) : customOp a b + customOp c d = customOp (a + c) (b + d)

-- Theorem to prove
theorem custom_calculator_results :
  (customOp 2 3 + customOp 0 3 = -2) ∧ (customOp 1024 48 = 2000) := by
  sorry

end NUMINAMATH_CALUDE_custom_calculator_results_l2544_254442


namespace NUMINAMATH_CALUDE_only_solution_is_two_l2544_254461

/-- Represents the number constructed in the problem -/
def constructNumber (k : ℕ) : ℕ :=
  (10^2000 - 1) - (10^k - 1) * 10^(2000 - k) - (10^1001 - 1)

/-- The main theorem stating that k = 2 is the only solution -/
theorem only_solution_is_two :
  ∃! k : ℕ, k > 0 ∧ ∃ m : ℕ, constructNumber k = m^2 :=
sorry

end NUMINAMATH_CALUDE_only_solution_is_two_l2544_254461


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2544_254473

theorem simplify_and_evaluate (x y : ℤ) (hx : x = -1) (hy : y = -2) :
  2 * (x - 2*y)^2 - (2*y + x) * (-2*y + x) = 33 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2544_254473


namespace NUMINAMATH_CALUDE_blue_cards_count_l2544_254494

theorem blue_cards_count (red_cards : ℕ) (blue_prob : ℝ) (blue_cards : ℕ) : 
  red_cards = 10 → 
  blue_prob = 0.8 → 
  (blue_cards : ℝ) / ((blue_cards : ℝ) + (red_cards : ℝ)) = blue_prob → 
  blue_cards = 40 := by
sorry

end NUMINAMATH_CALUDE_blue_cards_count_l2544_254494


namespace NUMINAMATH_CALUDE_sculpture_cost_in_inr_l2544_254447

/-- Exchange rate between US dollars and Namibian dollars -/
def usd_to_nad : ℝ := 10

/-- Exchange rate between US dollars and Chinese yuan -/
def usd_to_cny : ℝ := 7

/-- Exchange rate between Chinese yuan and Indian Rupees -/
def cny_to_inr : ℝ := 10

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 200

/-- Theorem stating the cost of the sculpture in Indian Rupees -/
theorem sculpture_cost_in_inr :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny * cny_to_inr = 1400 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_inr_l2544_254447


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l2544_254408

theorem basketball_lineup_count :
  let total_players : ℕ := 20
  let lineup_size : ℕ := 5
  let specific_role : ℕ := 1
  let interchangeable : ℕ := 4
  total_players.choose specific_role * (total_players - specific_role).choose interchangeable = 77520 :=
by sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l2544_254408


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2544_254484

theorem negation_of_proposition :
  (¬∃ x : ℝ, x > 0 ∧ Real.sin x > 2^x - 1) ↔ (∀ x : ℝ, x > 0 → Real.sin x ≤ 2^x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2544_254484
