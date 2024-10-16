import Mathlib

namespace NUMINAMATH_CALUDE_total_points_is_65_l3018_301884

/-- Represents the types of enemies in the game -/
inductive EnemyType
  | A
  | B
  | C

/-- The number of points earned for defeating each type of enemy -/
def pointsForEnemy (t : EnemyType) : ℕ :=
  match t with
  | EnemyType.A => 10
  | EnemyType.B => 15
  | EnemyType.C => 20

/-- The total number of enemies in the level -/
def totalEnemies : ℕ := 8

/-- The number of each type of enemy in the level -/
def enemyCount (t : EnemyType) : ℕ :=
  match t with
  | EnemyType.A => 3
  | EnemyType.B => 2
  | EnemyType.C => 3

/-- The number of enemies defeated for each type -/
def defeatedCount (t : EnemyType) : ℕ :=
  match t with
  | EnemyType.A => 3  -- All Type A enemies
  | EnemyType.B => 1  -- Half of Type B enemies
  | EnemyType.C => 1  -- One Type C enemy

/-- Calculates the total points earned -/
def totalPointsEarned : ℕ :=
  (defeatedCount EnemyType.A * pointsForEnemy EnemyType.A) +
  (defeatedCount EnemyType.B * pointsForEnemy EnemyType.B) +
  (defeatedCount EnemyType.C * pointsForEnemy EnemyType.C)

/-- Theorem stating that the total points earned is 65 -/
theorem total_points_is_65 : totalPointsEarned = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_65_l3018_301884


namespace NUMINAMATH_CALUDE_sum_with_abs_zero_implies_triple_l3018_301806

theorem sum_with_abs_zero_implies_triple (a : ℝ) : a + |a| = 0 → a - |2*a| = 3*a := by
  sorry

end NUMINAMATH_CALUDE_sum_with_abs_zero_implies_triple_l3018_301806


namespace NUMINAMATH_CALUDE_circle_ratio_l3018_301848

theorem circle_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  π * b^2 - π * a^2 = 5 * (π * a^2) → a / b = Real.sqrt 6 / 6 :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_l3018_301848


namespace NUMINAMATH_CALUDE_largest_x_value_l3018_301827

theorem largest_x_value (x : ℝ) : 
  (10 * x^2 - 52 * x + 48 = 8) → x ≤ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_x_value_l3018_301827


namespace NUMINAMATH_CALUDE_kittens_given_to_friends_l3018_301859

/-- Given that Joan initially had 8 kittens and now has 6 kittens,
    prove that she gave 2 kittens to her friends. -/
theorem kittens_given_to_friends : 
  ∀ (initial current given : ℕ), 
    initial = 8 → 
    current = 6 → 
    given = initial - current → 
    given = 2 := by
  sorry

end NUMINAMATH_CALUDE_kittens_given_to_friends_l3018_301859


namespace NUMINAMATH_CALUDE_remainder_274_pow_274_mod_13_l3018_301808

theorem remainder_274_pow_274_mod_13 : 274^274 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_274_pow_274_mod_13_l3018_301808


namespace NUMINAMATH_CALUDE_division_problem_l3018_301897

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/5) : 
  c / a = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3018_301897


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l3018_301839

theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
    sheep / horses = 4 / 7 →
    horses * 230 = 12880 →
    sheep = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l3018_301839


namespace NUMINAMATH_CALUDE_restaurant_sales_problem_l3018_301852

/-- Represents the dinner sales for a restaurant over four days. -/
structure RestaurantSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- The conditions and proof goal for the restaurant sales problem. -/
theorem restaurant_sales_problem (sales : RestaurantSales) : 
  sales.monday = 40 →
  sales.tuesday = sales.monday + 40 →
  sales.wednesday = sales.tuesday / 2 →
  sales.thursday > sales.wednesday →
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 203 →
  sales.thursday - sales.wednesday = 3 := by
  sorry


end NUMINAMATH_CALUDE_restaurant_sales_problem_l3018_301852


namespace NUMINAMATH_CALUDE_cistern_fill_time_l3018_301803

theorem cistern_fill_time (empty_time second_tap : ℝ) (fill_time_both : ℝ) 
  (h1 : empty_time = 9)
  (h2 : fill_time_both = 7.2) : 
  ∃ (fill_time_first : ℝ), 
    fill_time_first = 4 ∧ 
    (1 / fill_time_first - 1 / empty_time = 1 / fill_time_both) :=
by sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l3018_301803


namespace NUMINAMATH_CALUDE_least_number_of_cans_l3018_301826

def maaza_volume : ℕ := 20
def pepsi_volume : ℕ := 144
def sprite_volume : ℕ := 368

theorem least_number_of_cans : 
  let gcd := Nat.gcd maaza_volume (Nat.gcd pepsi_volume sprite_volume)
  let maaza_cans := maaza_volume / gcd
  let pepsi_cans := pepsi_volume / gcd
  let sprite_cans := sprite_volume / gcd
  maaza_cans + pepsi_cans + sprite_cans = 133 := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l3018_301826


namespace NUMINAMATH_CALUDE_all_representable_l3018_301813

def powers_of_three : List ℕ := [1, 3, 9, 27, 81, 243, 729]

def can_represent (n : ℕ) (powers : List ℕ) : Prop :=
  ∃ (subset : List ℕ) (signs : List Bool),
    subset ⊆ powers ∧
    signs.length = subset.length ∧
    (List.zip subset signs).foldl
      (λ acc (p, sign) => if sign then acc + p else acc - p) 0 = n

theorem all_representable :
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 1093 → can_represent n powers_of_three :=
sorry

end NUMINAMATH_CALUDE_all_representable_l3018_301813


namespace NUMINAMATH_CALUDE_valid_triples_l3018_301844

-- Define the type for our triples
def Triple := (Nat × Nat × Nat)

-- Define the conditions
def satisfiesConditions (t : Triple) : Prop :=
  let (a, b, c) := t
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧  -- positive integers
  (a ≤ b) ∧ (b ≤ c) ∧  -- ordered
  (Nat.gcd a (Nat.gcd b c) = 1) ∧  -- gcd(a,b,c) = 1
  ((a + b + c) ∣ (a^12 + b^12 + c^12)) ∧
  ((a + b + c) ∣ (a^23 + b^23 + c^23)) ∧
  ((a + b + c) ∣ (a^11004 + b^11004 + c^11004))

-- The theorem
theorem valid_triples :
  {t : Triple | satisfiesConditions t} = {(1,1,1), (1,1,4)} := by
  sorry

end NUMINAMATH_CALUDE_valid_triples_l3018_301844


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l3018_301863

/-- Given that B is a digit in base 5 and c is a base greater than 6, 
    if BBB₅ = 44ₖ, then the smallest possible sum of B + c is 34. -/
theorem smallest_sum_B_plus_c : 
  ∀ (B c : ℕ), 
    0 ≤ B ∧ B ≤ 4 →  -- B is a digit in base 5
    c > 6 →  -- c is a base greater than 6
    31 * B = 4 * c + 4 →  -- BBB₅ = 44ₖ
    ∀ (B' c' : ℕ), 
      0 ≤ B' ∧ B' ≤ 4 →
      c' > 6 →
      31 * B' = 4 * c' + 4 →
      B + c ≤ B' + c' →
      B + c = 34 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l3018_301863


namespace NUMINAMATH_CALUDE_maximum_marks_l3018_301876

theorem maximum_marks (percentage : ℝ) (obtained_marks : ℝ) (max_marks : ℝ) : 
  percentage = 92 / 100 → 
  obtained_marks = 184 → 
  percentage * max_marks = obtained_marks → 
  max_marks = 200 := by
sorry

end NUMINAMATH_CALUDE_maximum_marks_l3018_301876


namespace NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l3018_301881

/-- Proves that 1 cubic foot equals 1728 cubic inches, given that 1 foot equals 12 inches. -/
theorem cubic_foot_to_cubic_inches :
  (1 : ℝ) * (1 : ℝ) * (1 : ℝ) = 1728 * (1 / 12 : ℝ) * (1 / 12 : ℝ) * (1 / 12 : ℝ) :=
by
  sorry

#check cubic_foot_to_cubic_inches

end NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l3018_301881


namespace NUMINAMATH_CALUDE_cuboid_vertices_sum_l3018_301865

theorem cuboid_vertices_sum (n : ℕ) (h : 6 * n + 12 * n = 216) : 8 * n = 96 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_vertices_sum_l3018_301865


namespace NUMINAMATH_CALUDE_noah_has_largest_result_l3018_301874

def starting_number : ℕ := 15

def liam_result : ℕ := ((starting_number - 2) * 3) + 3
def mia_result : ℕ := ((starting_number * 3) - 4) + 3
def noah_result : ℕ := ((starting_number - 3) + 4) * 3

theorem noah_has_largest_result :
  noah_result > liam_result ∧ noah_result > mia_result :=
by sorry

end NUMINAMATH_CALUDE_noah_has_largest_result_l3018_301874


namespace NUMINAMATH_CALUDE_cube_root_product_simplification_l3018_301820

theorem cube_root_product_simplification :
  (1 + 27) ^ (1/3) * (1 + 27 ^ (1/3)) ^ (1/3) = 112 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_product_simplification_l3018_301820


namespace NUMINAMATH_CALUDE_base_sum_theorem_l3018_301860

theorem base_sum_theorem : ∃ (R₁ R₂ : ℕ), 
  (R₁ > 1 ∧ R₂ > 1) ∧
  (4 * R₁ + 5) * (R₂^2 - 1) = (3 * R₂ + 4) * (R₁^2 - 1) ∧
  (5 * R₁ + 4) * (R₂^2 - 1) = (4 * R₂ + 3) * (R₁^2 - 1) ∧
  R₁ + R₂ = 23 := by
  sorry

end NUMINAMATH_CALUDE_base_sum_theorem_l3018_301860


namespace NUMINAMATH_CALUDE_intersection_point_l3018_301886

def line1 (x y : ℚ) : Prop := 12 * x - 5 * y = 8
def line2 (x y : ℚ) : Prop := 10 * x + 2 * y = 20

theorem intersection_point :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (58/37, 667/370) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3018_301886


namespace NUMINAMATH_CALUDE_min_sum_squares_l3018_301821

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3018_301821


namespace NUMINAMATH_CALUDE_solution_proof_l3018_301869

/-- The equation we're solving -/
def equation (x : ℝ) : Prop :=
  4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20) = x^2 - 13*x - 6

/-- The largest real solution to the equation -/
noncomputable def n : ℝ := 13 + Real.sqrt 61

/-- The decomposition of n into d + √(e + √f) -/
def d : ℕ := 13
def e : ℕ := 61
def f : ℕ := 0

theorem solution_proof :
  equation n ∧ 
  n = d + Real.sqrt (e + Real.sqrt f) ∧
  d + e + f = 74 := by sorry

end NUMINAMATH_CALUDE_solution_proof_l3018_301869


namespace NUMINAMATH_CALUDE_no_triangle_tangent_to_both_curves_l3018_301851

/-- C₁ is the unit circle -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- C₂ is an ellipse with semi-major axis a and semi-minor axis b -/
def C₂ (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- A triangle is externally tangent to C₁ if all its vertices lie outside or on C₁ 
    and each side is tangent to C₁ -/
def externally_tangent_C₁ (A B C : ℝ × ℝ) : Prop := sorry

/-- A triangle is internally tangent to C₂ if all its vertices lie inside or on C₂ 
    and each side is tangent to C₂ -/
def internally_tangent_C₂ (a b : ℝ) (A B C : ℝ × ℝ) : Prop := sorry

theorem no_triangle_tangent_to_both_curves (a b : ℝ) :
  a > b ∧ b > 0 ∧ C₂ a b 1 1 →
  ¬ ∃ (A B C : ℝ × ℝ), externally_tangent_C₁ A B C ∧ internally_tangent_C₂ a b A B C :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_tangent_to_both_curves_l3018_301851


namespace NUMINAMATH_CALUDE_triangle_property_l3018_301840

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition from the problem -/
def satisfies_condition (t : Triangle) : Prop :=
  1 - 2 * Real.sin t.B * Real.sin t.C = Real.cos (2 * t.B) + Real.cos (2 * t.C) - Real.cos (2 * t.A)

theorem triangle_property (t : Triangle) (h : satisfies_condition t) :
  t.A = Real.pi / 3 ∧ ∃ (x : ℝ), x ≤ Real.pi ∧ ∀ (y : ℝ), Real.sin t.B + Real.sin t.C ≤ Real.sin y := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l3018_301840


namespace NUMINAMATH_CALUDE_lcm_23_46_827_l3018_301857

theorem lcm_23_46_827 (h1 : 46 = 23 * 2) (h2 : Nat.Prime 827) :
  Nat.lcm 23 (Nat.lcm 46 827) = 38042 :=
by sorry

end NUMINAMATH_CALUDE_lcm_23_46_827_l3018_301857


namespace NUMINAMATH_CALUDE_symmetry_implies_axis_l3018_301817

/-- A function g is symmetric about x = 1.5 if g(x) = g(3-x) for all x -/
def IsSymmetricAbout1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

/-- The line x = 1.5 is an axis of symmetry for a function g if 
    for every point (x, g(x)) on the graph, (3-x, g(x)) is also on the graph -/
def IsAxisOfSymmetry1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

theorem symmetry_implies_axis (g : ℝ → ℝ) :
  IsSymmetricAbout1_5 g → IsAxisOfSymmetry1_5 g := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_axis_l3018_301817


namespace NUMINAMATH_CALUDE_car_distance_ratio_l3018_301899

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ  -- Speed in km/hr
  time : ℝ   -- Time in hours

/-- Calculates the distance traveled by a car -/
def distance (c : Car) : ℝ := c.speed * c.time

/-- The problem statement -/
theorem car_distance_ratio :
  let car_a : Car := ⟨50, 8⟩
  let car_b : Car := ⟨25, 4⟩
  (distance car_a) / (distance car_b) = 4
  := by sorry

end NUMINAMATH_CALUDE_car_distance_ratio_l3018_301899


namespace NUMINAMATH_CALUDE_cylinder_volume_theorem_l3018_301812

/-- The volume of a cylinder with a rectangular net of dimensions 2a and a -/
def cylinder_volume (a : ℝ) : Set ℝ :=
  {v | v = a^3 / Real.pi ∨ v = a^3 / (2 * Real.pi)}

/-- Theorem stating that the volume of the cylinder is either a³/π or a³/(2π) -/
theorem cylinder_volume_theorem (a : ℝ) (h : a > 0) :
  ∀ v ∈ cylinder_volume a, v = a^3 / Real.pi ∨ v = a^3 / (2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_theorem_l3018_301812


namespace NUMINAMATH_CALUDE_complement_intersection_empty_l3018_301875

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {2, 4, 5}

theorem complement_intersection_empty :
  (Aᶜ ∩ Bᶜ : Set Nat) = ∅ :=
by
  sorry

#check complement_intersection_empty

end NUMINAMATH_CALUDE_complement_intersection_empty_l3018_301875


namespace NUMINAMATH_CALUDE_simplify_expression_l3018_301832

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : x^2 + y^2 + z^2 = x*y + y*z + z*x) :
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2) = 3 / x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3018_301832


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3018_301871

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) ↔ -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3018_301871


namespace NUMINAMATH_CALUDE_circle_path_in_right_triangle_l3018_301834

theorem circle_path_in_right_triangle (a b c : ℝ) (r : ℝ) :
  a = 5 →
  b = 12 →
  c = 13 →
  r = 2 →
  a^2 + b^2 = c^2 →
  let path_length := (a - 2*r) + (b - 2*r) + (c - 2*r)
  path_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_path_in_right_triangle_l3018_301834


namespace NUMINAMATH_CALUDE_inequality_solution_l3018_301890

theorem inequality_solution (x : ℝ) : 2*x + 6 > 5*x - 3 → x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3018_301890


namespace NUMINAMATH_CALUDE_unique_pair_l3018_301842

def is_valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ a + b = 28 ∧ (Even a ∨ Even b)

theorem unique_pair : ∀ a b : ℕ, is_valid_pair a b → (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_l3018_301842


namespace NUMINAMATH_CALUDE_line_not_in_plane_implies_parallel_l3018_301830

-- Define the types for lines and planes
variable (L P : Type*)

-- Define the relation for a line being contained in a plane
variable (containedIn : L → P → Prop)

-- Define the relation for a line being parallel to a plane
variable (parallelTo : L → P → Prop)

-- State the theorem
theorem line_not_in_plane_implies_parallel (l : L) (α : P) :
  (¬ containedIn l α) → parallelTo l α := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_plane_implies_parallel_l3018_301830


namespace NUMINAMATH_CALUDE_cos_eight_arccos_one_fourth_l3018_301877

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1/4)) = 172546/1048576 := by
  sorry

end NUMINAMATH_CALUDE_cos_eight_arccos_one_fourth_l3018_301877


namespace NUMINAMATH_CALUDE_symmetric_sine_function_value_l3018_301891

theorem symmetric_sine_function_value (a φ : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (3 * x + φ)
  (∀ x, f (a + x) = f (a - x)) →
  f (a + π / 6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sine_function_value_l3018_301891


namespace NUMINAMATH_CALUDE_product_80641_9999_l3018_301819

theorem product_80641_9999 : 80641 * 9999 = 806329359 := by
  sorry

end NUMINAMATH_CALUDE_product_80641_9999_l3018_301819


namespace NUMINAMATH_CALUDE_sales_department_replacement_l3018_301895

/-- Represents the ages and work experience of employees in a sales department. -/
structure SalesDepartment where
  initialMenCount : ℕ
  initialAvgAge : ℝ
  initialAvgExperience : ℝ
  replacedMenAges : Fin 2 → ℕ
  womenAgeRanges : Fin 2 → Set ℕ
  newAvgAge : ℝ
  newAvgExperience : ℝ

/-- Theorem stating the average age of the two women and the change in work experience. -/
theorem sales_department_replacement
  (dept : SalesDepartment)
  (h_men_count : dept.initialMenCount = 8)
  (h_age_increase : dept.newAvgAge = dept.initialAvgAge + 2)
  (h_exp_change : dept.newAvgExperience = dept.initialAvgExperience + 1)
  (h_replaced_ages : dept.replacedMenAges 0 = 20 ∧ dept.replacedMenAges 1 = 24)
  (h_women_ages : dept.womenAgeRanges 0 = Set.Icc 26 30 ∧ dept.womenAgeRanges 1 = Set.Icc 32 36) :
  ∃ (w₁ w₂ : ℕ), w₁ ∈ dept.womenAgeRanges 0 ∧ w₂ ∈ dept.womenAgeRanges 1 ∧
  (w₁ + w₂) / 2 = 30 ∧
  (dept.initialMenCount * dept.newAvgExperience - dept.initialMenCount * dept.initialAvgExperience) = 8 := by
  sorry


end NUMINAMATH_CALUDE_sales_department_replacement_l3018_301895


namespace NUMINAMATH_CALUDE_netflix_shows_l3018_301873

/-- The number of shows watched per week by Gina and her sister on Netflix. -/
def total_shows (gina_minutes : ℕ) (show_length : ℕ) (gina_ratio : ℕ) : ℕ :=
  let gina_shows := gina_minutes / show_length
  let sister_shows := gina_shows / gina_ratio
  gina_shows + sister_shows

/-- Theorem stating the total number of shows watched per week given the conditions. -/
theorem netflix_shows : total_shows 900 50 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_netflix_shows_l3018_301873


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l3018_301889

/-- A trapezoid with mutually perpendicular diagonals -/
structure Trapezoid :=
  (height : ℝ)
  (diagonal : ℝ)
  (diagonals_perpendicular : Bool)

/-- The area of a trapezoid with given properties -/
def trapezoid_area (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem: The area of a trapezoid with mutually perpendicular diagonals, 
    height 4, and one diagonal of length 5 is equal to 50/3 -/
theorem trapezoid_area_theorem (t : Trapezoid) 
  (h1 : t.height = 4)
  (h2 : t.diagonal = 5)
  (h3 : t.diagonals_perpendicular = true) : 
  trapezoid_area t = 50 / 3 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_area_theorem_l3018_301889


namespace NUMINAMATH_CALUDE_inequality_proof_l3018_301888

theorem inequality_proof (p q r x y θ : ℝ) :
  p * x^(q - y) + q * x^(r - y) + r * x^(y - θ) ≥ p + q + r := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3018_301888


namespace NUMINAMATH_CALUDE_quadratic_other_x_intercept_l3018_301809

/-- A quadratic function with vertex (5, 8) and one x-intercept at (1, 0) has its other x-intercept at x = 9 -/
theorem quadratic_other_x_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 8 - a * (x - 5)^2) →  -- vertex form of quadratic with vertex (5, 8)
  (a * 1^2 + b * 1 + c = 0) →                       -- (1, 0) is an x-intercept
  (∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_other_x_intercept_l3018_301809


namespace NUMINAMATH_CALUDE_intersection_point_l3018_301887

-- Define the system of equations
def line1 (x y : ℚ) : Prop := 6 * x - 3 * y = 18
def line2 (x y : ℚ) : Prop := 8 * x + 2 * y = 20

-- State the theorem
theorem intersection_point :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (8/3, -2/3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3018_301887


namespace NUMINAMATH_CALUDE_baby_sea_turtles_on_sand_l3018_301846

theorem baby_sea_turtles_on_sand (total : ℕ) (swept_fraction : ℚ) (on_sand : ℕ) : 
  total = 42 → 
  swept_fraction = 1/3 → 
  on_sand = total - (swept_fraction * total).num → 
  on_sand = 28 := by
sorry

end NUMINAMATH_CALUDE_baby_sea_turtles_on_sand_l3018_301846


namespace NUMINAMATH_CALUDE_base9_perfect_square_multiple_of_3_l3018_301883

/-- Represents a number in base 9 of the form ab4c -/
structure Base9Number where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a ≠ 0

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.b + 36 + n.c

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem base9_perfect_square_multiple_of_3 (n : Base9Number) 
  (h1 : isPerfectSquare (toDecimal n))
  (h2 : (toDecimal n) % 3 = 0) :
  n.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_base9_perfect_square_multiple_of_3_l3018_301883


namespace NUMINAMATH_CALUDE_fifth_scroll_age_l3018_301837

def scroll_age (n : ℕ) : ℕ → ℕ
  | 0 => 4080
  | (m + 1) => scroll_age n m + (scroll_age n m) / 2

theorem fifth_scroll_age : scroll_age 5 4 = 20655 := by
  sorry

end NUMINAMATH_CALUDE_fifth_scroll_age_l3018_301837


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3018_301856

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * Real.pi * r^2 = 36 * Real.pi →
  (4 / 3) * Real.pi * r^3 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3018_301856


namespace NUMINAMATH_CALUDE_total_towels_weight_lb_l3018_301836

-- Define the given conditions
def mary_towels : ℕ := 24
def frances_towels : ℕ := mary_towels / 4
def frances_towels_weight_oz : ℚ := 128

-- Define the weight of one towel in ounces
def towel_weight_oz : ℚ := frances_towels_weight_oz / frances_towels

-- Define the total number of towels
def total_towels : ℕ := mary_towels + frances_towels

-- Define the conversion factor from ounces to pounds
def oz_to_lb : ℚ := 1 / 16

-- Theorem to prove
theorem total_towels_weight_lb :
  (total_towels : ℚ) * towel_weight_oz * oz_to_lb = 40 :=
sorry

end NUMINAMATH_CALUDE_total_towels_weight_lb_l3018_301836


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l3018_301849

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 + (2-a)*x + 1

-- Define the solution range
def in_range (x : ℝ) : Prop := -1 < x ∧ x ≤ 3 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2

-- Define the uniqueness of the solution
def unique_solution (a : ℝ) : Prop :=
  ∃! x, quadratic a x = 0 ∧ in_range x

-- State the theorem
theorem quadratic_unique_solution :
  ∀ a : ℝ, unique_solution a ↔ 
    (a = 4.5) ∨ 
    (a < 0) ∨ 
    (a > 16/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l3018_301849


namespace NUMINAMATH_CALUDE_unique_prime_square_plus_two_prime_l3018_301807

theorem unique_prime_square_plus_two_prime : 
  ∃! p : ℕ, Prime p ∧ Prime (p^2 + 2) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_prime_square_plus_two_prime_l3018_301807


namespace NUMINAMATH_CALUDE_unique_intersection_l3018_301896

/-- The curve C in the xy-plane -/
def curve (x y : ℝ) : Prop :=
  y = x^2 ∧ x ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2)

/-- The line l in the xy-plane -/
def line (x y : ℝ) : Prop :=
  y - x = 2

/-- The intersection point of the curve and the line -/
def intersection_point : ℝ × ℝ := (-1, 1)

theorem unique_intersection :
  ∃! p : ℝ × ℝ, curve p.1 p.2 ∧ line p.1 p.2 ∧ p = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_l3018_301896


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l3018_301831

theorem function_satisfies_conditions (m n : ℕ) : 
  let f : ℕ → ℕ → ℕ := λ m n => m * n
  (m ≥ 1 ∧ n ≥ 1 → 2 * (f m n) = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  f m 0 = 0 ∧ f 0 n = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l3018_301831


namespace NUMINAMATH_CALUDE_trail_mix_problem_l3018_301853

/-- Given 16 bags of nuts and an unknown number of bags of dried fruit,
    if the maximum number of identical portions that can be made without
    leftover bags is 2, then the number of bags of dried fruit must be 2. -/
theorem trail_mix_problem (dried_fruit : ℕ) : 
  (∃ (portion_size : ℕ), 
    portion_size > 0 ∧ 
    (16 + dried_fruit) % 2 = 0 ∧
    (16 + dried_fruit) / 2 = portion_size ∧
    ∀ n : ℕ, n > 2 → (16 + dried_fruit) % n ≠ 0) →
  dried_fruit = 2 := by
sorry

end NUMINAMATH_CALUDE_trail_mix_problem_l3018_301853


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3018_301850

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    prove that the common difference d is 2 when (S_2020 / 2020) - (S_20 / 20) = 2000 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0)  -- Arithmetic sequence property
  (h_sum : ∀ n, S n = n * a 0 + n * (n - 1) / 2 * (a 1 - a 0))  -- Sum formula
  (h_condition : S 2020 / 2020 - S 20 / 20 = 2000)  -- Given condition
  : a 1 - a 0 = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3018_301850


namespace NUMINAMATH_CALUDE_tangent_sum_problem_l3018_301835

theorem tangent_sum_problem (x y m : ℝ) :
  x^3 + Real.sin (2*x) = m →
  y^3 + Real.sin (2*y) = -m →
  x ∈ Set.Ioo (-π/4) (π/4) →
  y ∈ Set.Ioo (-π/4) (π/4) →
  Real.tan (x + y + π/3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_problem_l3018_301835


namespace NUMINAMATH_CALUDE_exists_winning_strategy_l3018_301828

/-- Strategy for guessing numbers -/
def GuessingStrategy := ℕ → ℕ

/-- The set of integers from 1 to 2002 -/
def NumberSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 2002}

/-- Function to determine if a guess is correct -/
def isCorrectGuess (n : ℕ) (guess : ℕ) : Bool := n = guess

/-- Function to determine if a number is higher or lower than the guess -/
def compareGuess (n : ℕ) (guess : ℕ) : Bool := n > guess

/-- Function to determine if a strategy results in an odd number of guesses -/
def isOddNumberOfGuesses (strategy : GuessingStrategy) (n : ℕ) : Prop := sorry

/-- Function to calculate the probability of winning with a given strategy -/
def winningProbability (strategy : GuessingStrategy) : ℚ := sorry

/-- Theorem stating that there exists a strategy with winning probability greater than 2/3 -/
theorem exists_winning_strategy :
  ∃ (strategy : GuessingStrategy),
    (∀ n ∈ NumberSet, isOddNumberOfGuesses strategy n) ∧
    winningProbability strategy > 2/3 := by sorry

end NUMINAMATH_CALUDE_exists_winning_strategy_l3018_301828


namespace NUMINAMATH_CALUDE_number_of_divisors_l3018_301801

theorem number_of_divisors 
  (p q r : Nat) 
  (m : Nat) 
  (h_p_prime : Nat.Prime p) 
  (h_q_prime : Nat.Prime q) 
  (h_r_prime : Nat.Prime r) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) 
  (h_m_pos : m > 0) 
  (h_n_def : n = 7^m * p^2 * q * r) : 
  Nat.card (Nat.divisors n) = 12 * (m + 1) := by
sorry

end NUMINAMATH_CALUDE_number_of_divisors_l3018_301801


namespace NUMINAMATH_CALUDE_tom_jogging_distance_l3018_301841

/-- The distance Tom jogs in 15 minutes given his rate -/
theorem tom_jogging_distance (rate : ℝ) (time : ℝ) (h1 : rate = 1 / 18) (h2 : time = 15) :
  rate * time = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_tom_jogging_distance_l3018_301841


namespace NUMINAMATH_CALUDE_reciprocal_expression_l3018_301882

theorem reciprocal_expression (a b : ℝ) (h : a * b = 1) :
  a^2 * b - (a - 2023) = 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_expression_l3018_301882


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_original_statement_l3018_301833

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ (∃ x > 0, ¬ P x) :=
by sorry

theorem negation_of_original_statement :
  (¬ ∀ x > 0, x^2 - 3*x + 2 < 0) ↔ (∃ x > 0, x^2 - 3*x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_original_statement_l3018_301833


namespace NUMINAMATH_CALUDE_inequality_proof_l3018_301893

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  (2 + x)^2 / (1 + x)^2 + (2 + y)^2 / (1 + y)^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3018_301893


namespace NUMINAMATH_CALUDE_parallel_vectors_m_l3018_301814

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem parallel_vectors_m (m : ℝ) : 
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (-2, m)
  are_parallel a (a.1 + 2 * b.1, a.2 + 2 * b.2) → m = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_l3018_301814


namespace NUMINAMATH_CALUDE_angle_measure_l3018_301861

/-- Given two angles AOB and BOC, proves that angle AOC is either the sum or difference of these angles. -/
theorem angle_measure (α β : ℝ) (hα : α = 30) (hβ : β = 15) :
  ∃ γ : ℝ, (γ = α + β ∨ γ = α - β) ∧ (γ = 45 ∨ γ = 15) := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3018_301861


namespace NUMINAMATH_CALUDE_tangent_slope_angle_range_l3018_301845

theorem tangent_slope_angle_range (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n = Real.sqrt 3 / 2) :
  let f : ℝ → ℝ := λ x ↦ (1/3) * x^3 + n^2 * x
  let k := (m^2 + n^2)
  let θ := Real.arctan k
  θ ∈ Set.Ici (π/3) ∩ Set.Iio (π/2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_range_l3018_301845


namespace NUMINAMATH_CALUDE_rectangular_field_dimension_exists_unique_l3018_301825

theorem rectangular_field_dimension_exists_unique (area : ℝ) :
  ∃! m : ℝ, m > 0 ∧ (3 * m + 8) * (m - 3) = area :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_dimension_exists_unique_l3018_301825


namespace NUMINAMATH_CALUDE_smallest_pyramid_height_approx_l3018_301847

/-- Represents a square-based pyramid with a cylinder inside. -/
structure PyramidWithCylinder where
  base_side_length : ℝ
  cylinder_diameter : ℝ
  cylinder_length : ℝ

/-- Calculates the smallest possible height of the pyramid given its configuration. -/
def smallest_pyramid_height (p : PyramidWithCylinder) : ℝ :=
  sorry

/-- The theorem stating the smallest possible height of the pyramid. -/
theorem smallest_pyramid_height_approx :
  let p := PyramidWithCylinder.mk 20 10 10
  ∃ ε > 0, abs (smallest_pyramid_height p - 22.1) < ε :=
sorry

end NUMINAMATH_CALUDE_smallest_pyramid_height_approx_l3018_301847


namespace NUMINAMATH_CALUDE_game_points_sum_l3018_301805

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allieRolls : List ℕ := [6, 3, 2, 4]
def charlieRolls : List ℕ := [5, 3, 1, 6]

theorem game_points_sum : 
  (List.sum (List.map g allieRolls)) + (List.sum (List.map g charlieRolls)) = 38 := by
  sorry

end NUMINAMATH_CALUDE_game_points_sum_l3018_301805


namespace NUMINAMATH_CALUDE_linear_function_proof_l3018_301868

/-- A linear function passing through points (1, -1) and (-2, 8) -/
def f (x : ℝ) : ℝ := -3 * x + 2

theorem linear_function_proof :
  (f 1 = -1) ∧ 
  (f (-2) = 8) ∧ 
  (∀ x : ℝ, f x = -3 * x + 2) ∧
  (f (-10) = 32) := by
  sorry


end NUMINAMATH_CALUDE_linear_function_proof_l3018_301868


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3018_301867

theorem polynomial_factorization (x : ℝ) :
  (x^3 - x + 3)^2 = x^6 - 2*x^4 + 6*x^3 + x^2 - 6*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3018_301867


namespace NUMINAMATH_CALUDE_bug_path_distance_l3018_301872

theorem bug_path_distance (r : Real) (leg : Real) (h1 : r = 40) (h2 : leg = 50) :
  let diameter := 2 * r
  let other_leg := Real.sqrt (diameter ^ 2 - leg ^ 2)
  diameter + leg + other_leg = 192.45 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_distance_l3018_301872


namespace NUMINAMATH_CALUDE_percentage_problem_l3018_301829

theorem percentage_problem (x : ℝ) (h : 24 = 75 / 100 * x) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3018_301829


namespace NUMINAMATH_CALUDE_probability_at_least_one_defective_l3018_301811

theorem probability_at_least_one_defective (total : ℕ) (defective : ℕ) (chosen : ℕ) 
  (h1 : total = 20) (h2 : defective = 4) (h3 : chosen = 2) :
  (1 : ℚ) - (Nat.choose (total - defective) chosen : ℚ) / (Nat.choose total chosen : ℚ) = 7/19 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_defective_l3018_301811


namespace NUMINAMATH_CALUDE_problem_1_l3018_301870

theorem problem_1 (a : ℚ) (h : a = 1/2) : 2*a^2 - 5*a + a^2 + 4*a - 3*a^2 - 2 = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3018_301870


namespace NUMINAMATH_CALUDE_inequality_solution_l3018_301892

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 8 / (x + 4) ≥ 2) ↔ (x < -4 ∨ x ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3018_301892


namespace NUMINAMATH_CALUDE_seating_arrangement_l3018_301802

theorem seating_arrangement (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l3018_301802


namespace NUMINAMATH_CALUDE_double_sized_cube_weight_l3018_301885

/-- Given a cubical block of metal, this function calculates the weight of another cube of the same metal with sides twice as long. -/
def weight_of_double_sized_cube (original_weight : ℝ) : ℝ :=
  8 * original_weight

/-- Theorem stating that if a cubical block of metal weighs 3 pounds, then another cube of the same metal with sides twice as long will weigh 24 pounds. -/
theorem double_sized_cube_weight :
  weight_of_double_sized_cube 3 = 24 := by
  sorry

#eval weight_of_double_sized_cube 3

end NUMINAMATH_CALUDE_double_sized_cube_weight_l3018_301885


namespace NUMINAMATH_CALUDE_consecutive_sum_equality_l3018_301810

theorem consecutive_sum_equality :
  ∃ (a b : ℕ), 
    a ≥ 1 ∧ 
    5 * (a + 2) = 2 * b + 1 ∧ 
    ∀ (x : ℕ), x < a → ¬∃ (y : ℕ), 5 * (x + 2) = 2 * y + 1 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_sum_equality_l3018_301810


namespace NUMINAMATH_CALUDE_min_perimeter_non_congruent_isosceles_triangles_l3018_301843

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- The area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 2 : ℝ) * Real.sqrt ((t.leg : ℝ)^2 - (t.base / 2 : ℝ)^2)

theorem min_perimeter_non_congruent_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    t1.base * 9 = t2.base * 10 ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      s1.base * 9 = s2.base * 10 →
      perimeter t1 ≤ perimeter s1 ∧
      perimeter t1 = 728 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_non_congruent_isosceles_triangles_l3018_301843


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3018_301858

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (Real.sin θ)^6 / a^2 + (Real.cos θ)^6 / b^2 = 1 / (a^2 + b^2) →
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = (a^7 + b^7) / (a^2 + b^2)^6 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3018_301858


namespace NUMINAMATH_CALUDE_a_always_gets_half_rule_independent_l3018_301898

/-- The game rules for counter division --/
inductive Rule
| R1  -- B takes the biggest and smallest heaps
| R2  -- B takes the two middling heaps
| R3  -- B chooses between R1 and R2

/-- The optimal number of counters A can obtain --/
def optimal_counters (N : ℕ) (r : Rule) : ℕ :=
  N / 2

/-- Theorem: A always gets ⌊N/2⌋ counters regardless of the rule --/
theorem a_always_gets_half (N : ℕ) (h : N ≥ 4) (r : Rule) :
  optimal_counters N r = N / 2 := by
  sorry

/-- Corollary: The result is independent of the chosen rule --/
theorem rule_independent (N : ℕ) (h : N ≥ 4) (r1 r2 : Rule) :
  optimal_counters N r1 = optimal_counters N r2 := by
  sorry

end NUMINAMATH_CALUDE_a_always_gets_half_rule_independent_l3018_301898


namespace NUMINAMATH_CALUDE_cost_of_500_pencils_l3018_301878

/-- The cost of n pencils in dollars, given the price of one pencil in cents and the number of cents in a dollar. -/
def cost_of_pencils (n : ℕ) (price_per_pencil : ℕ) (cents_per_dollar : ℕ) : ℚ :=
  (n * price_per_pencil : ℚ) / cents_per_dollar

/-- Theorem stating that the cost of 500 pencils is 10 dollars, given the specified conditions. -/
theorem cost_of_500_pencils : 
  cost_of_pencils 500 2 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_500_pencils_l3018_301878


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l3018_301855

/-- Represents a mapping of letters to digits -/
def LetterMapping := Char → Fin 10

/-- Check if a mapping is valid for the cryptarithm -/
def is_valid_mapping (m : LetterMapping) : Prop :=
  m 'Г' ≠ 0 ∧
  m 'О' ≠ 0 ∧
  m 'В' ≠ 0 ∧
  (∀ c₁ c₂, c₁ ≠ c₂ → m c₁ ≠ m c₂) ∧
  (m 'Г' * 1000 + m 'О' * 100 + m 'Р' * 10 + m 'А') +
  (m 'О' * 10000 + m 'Г' * 1000 + m 'О' * 100 + m 'Н' * 10 + m 'Ь') =
  (m 'В' * 100000 + m 'У' * 10000 + m 'Л' * 1000 + m 'К' * 100 + m 'А' * 10 + m 'Н')

theorem cryptarithm_solution :
  ∃! m : LetterMapping, is_valid_mapping m ∧
    m 'Г' = 6 ∧ m 'О' = 9 ∧ m 'Р' = 4 ∧ m 'А' = 7 ∧
    m 'Н' = 2 ∧ m 'Ь' = 5 ∧
    m 'В' = 1 ∧ m 'У' = 0 ∧ m 'Л' = 3 ∧ m 'К' = 8 :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l3018_301855


namespace NUMINAMATH_CALUDE_adele_age_fraction_l3018_301854

/-- Given the ages of Jackson, Mandy, and Adele, prove that Adele's age is 3/4 of Jackson's age. -/
theorem adele_age_fraction (jackson_age mandy_age adele_age : ℕ) : 
  jackson_age = 20 →
  mandy_age = jackson_age + 10 →
  (jackson_age + 10) + (mandy_age + 10) + (adele_age + 10) = 95 →
  ∃ f : ℚ, adele_age = f * jackson_age ∧ f = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_adele_age_fraction_l3018_301854


namespace NUMINAMATH_CALUDE_gcd_65536_49152_l3018_301862

theorem gcd_65536_49152 : Nat.gcd 65536 49152 = 16384 := by
  sorry

end NUMINAMATH_CALUDE_gcd_65536_49152_l3018_301862


namespace NUMINAMATH_CALUDE_distinct_collections_l3018_301818

/-- Represents the count of each letter in MATHEMATICIAN --/
def letterCounts : Finset (Char × ℕ) := 
  {('M', 1), ('A', 3), ('T', 2), ('H', 1), ('E', 1), ('I', 3), ('C', 1), ('N', 1)}

/-- Represents the set of vowels in MATHEMATICIAN --/
def vowels : Finset Char := {'A', 'I', 'E'}

/-- Represents the set of consonants in MATHEMATICIAN --/
def consonants : Finset Char := {'M', 'T', 'H', 'C', 'N'}

/-- Calculates the number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of distinct vowel selections --/
def vowelSelections : ℕ := 
  choose 3 3 + 3 * choose 3 2 + 3 * choose 3 1 + choose 3 0

/-- Calculates the number of distinct consonant selections --/
def consonantSelections : ℕ := 
  choose 4 3 + 2 * choose 4 2 + choose 4 1

/-- The main theorem stating the total number of distinct collections --/
theorem distinct_collections : 
  vowelSelections * consonantSelections = 112 := by sorry

end NUMINAMATH_CALUDE_distinct_collections_l3018_301818


namespace NUMINAMATH_CALUDE_pet_ownership_l3018_301864

theorem pet_ownership (total_students : ℕ) (dog_owners : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 45)
  (h2 : dog_owners = 25)
  (h3 : cat_owners = 34)
  (h4 : ∀ s, s ∈ Finset.range total_students → 
    (s ∈ Finset.range dog_owners ∨ s ∈ Finset.range cat_owners)) :
  Finset.card (Finset.range dog_owners ∩ Finset.range cat_owners) = 14 := by
  sorry

end NUMINAMATH_CALUDE_pet_ownership_l3018_301864


namespace NUMINAMATH_CALUDE_critical_point_and_zeros_l3018_301824

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + (1 - 3 * Real.log x) / a

theorem critical_point_and_zeros (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → (deriv (f a)) x = 0 → x = 1 → a = 1) ∧
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ↔ 0 < a ∧ a < Real.exp (-1)) :=
sorry

end NUMINAMATH_CALUDE_critical_point_and_zeros_l3018_301824


namespace NUMINAMATH_CALUDE_smallest_n_for_interval_condition_l3018_301880

theorem smallest_n_for_interval_condition : ∃ (n : ℕ), n > 0 ∧
  (∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1992 →
    ∃ (k : ℕ), (m : ℚ) / 1993 < (k : ℚ) / n ∧ (k : ℚ) / n < ((m : ℚ) + 1) / 1994) ∧
  (∀ (n' : ℕ), 0 < n' ∧ n' < n →
    ∃ (m : ℕ), 1 ≤ m ∧ m ≤ 1992 ∧
      ∀ (k : ℕ), ((m : ℚ) / 1993 ≥ (k : ℚ) / n' ∨ (k : ℚ) / n' ≥ ((m : ℚ) + 1) / 1994)) ∧
  n = 3987 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_interval_condition_l3018_301880


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l3018_301823

theorem halfway_between_fractions : (2 / 9 + 5 / 12) / 2 = 23 / 72 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l3018_301823


namespace NUMINAMATH_CALUDE_eustace_milford_age_ratio_l3018_301838

theorem eustace_milford_age_ratio :
  ∀ (eustace_age milford_age : ℕ),
    eustace_age + 3 = 39 →
    milford_age + 3 = 21 →
    eustace_age / milford_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_eustace_milford_age_ratio_l3018_301838


namespace NUMINAMATH_CALUDE_first_die_sides_l3018_301816

theorem first_die_sides (p : ℝ) (n : ℕ) : 
  p = 0.023809523809523808 →  -- Given probability
  p = 1 / (n * 7) →           -- Probability formula
  n = 6                       -- Number of sides on first die
:= by sorry

end NUMINAMATH_CALUDE_first_die_sides_l3018_301816


namespace NUMINAMATH_CALUDE_sets_problem_l3018_301815

def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

theorem sets_problem (a : ℝ) :
  (A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 8}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | 5 ≤ x ∧ x < 8}) ∧
  (C a ∩ A = C a → a ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_sets_problem_l3018_301815


namespace NUMINAMATH_CALUDE_reeses_height_l3018_301894

theorem reeses_height (parker daisy reese : ℝ) 
  (h1 : parker = daisy - 4)
  (h2 : daisy = reese + 8)
  (h3 : (parker + daisy + reese) / 3 = 64) :
  reese = 60 := by sorry

end NUMINAMATH_CALUDE_reeses_height_l3018_301894


namespace NUMINAMATH_CALUDE_dave_performance_weeks_l3018_301822

/-- Given that Dave breaks 2 guitar strings per night, performs 6 shows per week,
    and needs to replace 144 guitar strings in total, prove that he performs for 12 weeks. -/
theorem dave_performance_weeks 
  (strings_per_night : ℕ)
  (shows_per_week : ℕ)
  (total_strings : ℕ)
  (h1 : strings_per_night = 2)
  (h2 : shows_per_week = 6)
  (h3 : total_strings = 144) :
  total_strings / (strings_per_night * shows_per_week) = 12 := by
sorry

end NUMINAMATH_CALUDE_dave_performance_weeks_l3018_301822


namespace NUMINAMATH_CALUDE_brady_record_chase_l3018_301800

/-- The minimum average yards per game needed to beat the record -/
def min_avg_yards_per_game (current_record : ℕ) (current_yards : ℕ) (games_left : ℕ) : ℚ :=
  (current_record + 1 - current_yards) / games_left

theorem brady_record_chase (current_record : ℕ) (current_yards : ℕ) (games_left : ℕ)
  (h1 : current_record = 5999)
  (h2 : current_yards = 4200)
  (h3 : games_left = 6) :
  min_avg_yards_per_game current_record current_yards games_left = 300 := by
sorry

end NUMINAMATH_CALUDE_brady_record_chase_l3018_301800


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3018_301879

-- Problem 1
theorem problem_1 : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + |(-3)| = 4 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : a ≠ 1) : (1 - 1/a) / ((a^2 - 2*a + 1) / a) = 1 / (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3018_301879


namespace NUMINAMATH_CALUDE_laptop_final_price_l3018_301866

/-- The final price of a laptop after successive discounts --/
theorem laptop_final_price (original_price : ℝ) (discount1 discount2 discount3 : ℝ) :
  original_price = 1200 →
  discount1 = 0.1 →
  discount2 = 0.2 →
  discount3 = 0.05 →
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 820.80 := by
  sorry

#check laptop_final_price

end NUMINAMATH_CALUDE_laptop_final_price_l3018_301866


namespace NUMINAMATH_CALUDE_math_books_together_probability_l3018_301804

def total_textbooks : ℕ := 15
def math_textbooks : ℕ := 4
def box_sizes : List ℕ := [4, 5, 6]

def is_valid_distribution (dist : List ℕ) : Prop :=
  dist.length = 3 ∧ 
  dist.sum = total_textbooks ∧
  ∀ b ∈ dist, b ≤ total_textbooks - math_textbooks + 1

def probability_math_books_together : ℚ :=
  4 / 273

theorem math_books_together_probability :
  probability_math_books_together = 
    (number_of_valid_distributions_with_math_books_together : ℚ) / 
    (total_number_of_valid_distributions : ℚ) :=
by sorry

#check math_books_together_probability

end NUMINAMATH_CALUDE_math_books_together_probability_l3018_301804
