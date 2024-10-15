import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_of_seven_power_l1057_105740

theorem units_digit_of_seven_power (n : ℕ) : (7^(6^5) : ℕ) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_power_l1057_105740


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l1057_105774

theorem quadratic_two_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (2 * x₁^2 + x₁ - 1 = 0) ∧ (2 * x₂^2 + x₂ - 1 = 0) ∧
  (∀ x : ℝ, 2 * x^2 + x - 1 = 0 → (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l1057_105774


namespace NUMINAMATH_CALUDE_only_C_suitable_for_census_C_unique_suitable_for_census_l1057_105766

/-- Represents a survey option -/
inductive SurveyOption
| A  -- Understanding the vision of middle school students in our province
| B  -- Investigating the viewership of "The Reader"
| C  -- Inspecting the components of a newly developed fighter jet to ensure successful test flights
| D  -- Testing the lifespan of a batch of light bulbs

/-- Defines what makes a survey suitable for a census -/
def isSuitableForCensus (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.C => True
  | _ => False

/-- Theorem stating that only option C is suitable for a census -/
theorem only_C_suitable_for_census :
  ∀ (option : SurveyOption), isSuitableForCensus option ↔ option = SurveyOption.C :=
by
  sorry

/-- Corollary: Option C is the unique survey suitable for a census -/
theorem C_unique_suitable_for_census :
  ∃! (option : SurveyOption), isSuitableForCensus option :=
by
  sorry

end NUMINAMATH_CALUDE_only_C_suitable_for_census_C_unique_suitable_for_census_l1057_105766


namespace NUMINAMATH_CALUDE_tangent_lines_perpendicular_PQR_inequality_l1057_105760

-- Define the function f and its inverse g
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

-- Define the slopes of the tangent lines
noncomputable def k₁ : ℝ := -1 / Real.exp 1
noncomputable def k₂ : ℝ := Real.exp 1

-- Define P, Q, and R
noncomputable def P (a b : ℝ) : ℝ := g ((a + b) / 2)
noncomputable def Q (a b : ℝ) : ℝ := (g a - g b) / (a - b)
noncomputable def R (a b : ℝ) : ℝ := (g a + g b) / 2

-- State the theorems to be proved
theorem tangent_lines_perpendicular : k₁ * k₂ = -1 := by sorry

theorem PQR_inequality (a b : ℝ) (h : a ≠ b) : P a b < Q a b ∧ Q a b < R a b := by sorry

end NUMINAMATH_CALUDE_tangent_lines_perpendicular_PQR_inequality_l1057_105760


namespace NUMINAMATH_CALUDE_first_group_size_l1057_105723

/-- Represents the work rate of a single beaver -/
def BeaverWorkRate : ℝ := 1

/-- Represents the total amount of work required to build the dam -/
def DamWork : ℝ := 1

theorem first_group_size (time1 : ℝ) (time2 : ℝ) (num_beavers2 : ℕ) :
  time1 > 0 → time2 > 0 → num_beavers2 > 0 →
  time1 = 3 → time2 = 5 → num_beavers2 = 12 →
  ∃ (num_beavers1 : ℕ), 
    num_beavers1 > 0 ∧ 
    (num_beavers1 : ℝ) * BeaverWorkRate * time1 = DamWork ∧
    (num_beavers2 : ℝ) * BeaverWorkRate * time2 = DamWork ∧
    num_beavers1 = 20 :=
by sorry

#check first_group_size

end NUMINAMATH_CALUDE_first_group_size_l1057_105723


namespace NUMINAMATH_CALUDE_problem_statement_l1057_105770

theorem problem_statement (x y z : ℝ) (hx : x = 3) (hy : y = 4) (hz : z = 2) :
  (x^5 + 3*y^3 + z^2) / 12 = 439/12 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1057_105770


namespace NUMINAMATH_CALUDE_min_square_side_for_given_dimensions_l1057_105773

/-- Represents the dimensions of a table and cube -/
structure TableDimensions where
  length : ℕ
  breadth : ℕ
  cube_side : ℕ

/-- Calculates the minimum side length of a square formed by arranging tables -/
def min_square_side (td : TableDimensions) : ℕ :=
  2 * td.length + 2 * td.breadth

/-- Theorem stating the minimum side length of the square formed by tables -/
theorem min_square_side_for_given_dimensions :
  ∀ (td : TableDimensions),
    td.length = 12 →
    td.breadth = 16 →
    td.cube_side = 4 →
    min_square_side td = 56 :=
by
  sorry

#eval min_square_side ⟨12, 16, 4⟩

end NUMINAMATH_CALUDE_min_square_side_for_given_dimensions_l1057_105773


namespace NUMINAMATH_CALUDE_range_of_sum_l1057_105778

def f (x : ℝ) : ℝ := |2 - x^2|

theorem range_of_sum (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  2 < a + b ∧ a + b < 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l1057_105778


namespace NUMINAMATH_CALUDE_twenty_fifth_in_base5_l1057_105749

/-- Converts a natural number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Checks if a list of digits is a valid base 5 number -/
def isValidBase5 (l : List ℕ) : Prop :=
  l.all (· < 5)

theorem twenty_fifth_in_base5 :
  let base5Repr := toBase5 25
  isValidBase5 base5Repr ∧ base5Repr = [1, 0, 0] := by sorry

end NUMINAMATH_CALUDE_twenty_fifth_in_base5_l1057_105749


namespace NUMINAMATH_CALUDE_central_angle_specific_points_l1057_105759

/-- A point on a sphere represented by its latitude and longitude -/
structure SpherePoint where
  latitude : Real
  longitude : Real

/-- The central angle between two points on a sphere -/
def centralAngle (center : Point) (p1 p2 : SpherePoint) : Real :=
  sorry

theorem central_angle_specific_points :
  let center : Point := sorry
  let pointA : SpherePoint := { latitude := 0, longitude := 110 }
  let pointB : SpherePoint := { latitude := 45, longitude := -115 }
  centralAngle center pointA pointB = 120 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_specific_points_l1057_105759


namespace NUMINAMATH_CALUDE_pencils_per_row_l1057_105726

/-- Given a total of 154 pencils arranged in 14 rows with an equal number of pencils in each row,
    prove that there are 11 pencils in each row. -/
theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 154)
  (h2 : num_rows = 14)
  (h3 : total_pencils = num_rows * pencils_per_row) :
  pencils_per_row = 11 := by
  sorry

#check pencils_per_row

end NUMINAMATH_CALUDE_pencils_per_row_l1057_105726


namespace NUMINAMATH_CALUDE_max_product_combination_l1057_105701

def digits : List Nat := [1, 3, 5, 8, 9]

def is_valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def product (a b c d e : Nat) : Nat :=
  (100 * a + 10 * b + c) * (10 * d + e)

theorem max_product_combination :
  ∀ a b c d e : Nat,
    is_valid_combination a b c d e →
    product a b c d e ≤ product 8 3 1 9 5 :=
by sorry

end NUMINAMATH_CALUDE_max_product_combination_l1057_105701


namespace NUMINAMATH_CALUDE_brown_family_probability_l1057_105724

/-- The number of children Mr. Brown has. -/
def num_children : ℕ := 8

/-- The probability of a child being a twin. -/
def twin_probability : ℚ := 1/10

/-- The probability of a child being male (or female). -/
def gender_probability : ℚ := 1/2

/-- Calculates the probability of having an unequal number of sons and daughters
    given the number of children, twin probability, and gender probability. -/
def unequal_gender_probability (n : ℕ) (p_twin : ℚ) (p_gender : ℚ) : ℚ :=
  sorry

theorem brown_family_probability :
  unequal_gender_probability num_children twin_probability gender_probability = 95/128 :=
sorry

end NUMINAMATH_CALUDE_brown_family_probability_l1057_105724


namespace NUMINAMATH_CALUDE_fruit_picking_combinations_l1057_105719

def fruit_types : ℕ := 3
def picks : ℕ := 2

theorem fruit_picking_combinations : (fruit_types.choose picks) = 6 := by
  sorry

end NUMINAMATH_CALUDE_fruit_picking_combinations_l1057_105719


namespace NUMINAMATH_CALUDE_bricks_used_in_scenario_l1057_105785

/-- The number of bricks used in a construction project -/
def bricks_used (walls : ℕ) (courses_per_wall : ℕ) (bricks_per_course : ℕ) (uncompleted_courses : ℕ) : ℕ :=
  walls * courses_per_wall * bricks_per_course - uncompleted_courses * bricks_per_course

/-- Theorem stating that the number of bricks used in the given scenario is 220 -/
theorem bricks_used_in_scenario : bricks_used 4 6 10 2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_bricks_used_in_scenario_l1057_105785


namespace NUMINAMATH_CALUDE_park_population_l1057_105720

/-- Calculates the total population of lions, leopards, and elephants in a park. -/
theorem park_population (num_lions : ℕ) (num_leopards : ℕ) (num_elephants : ℕ) : 
  num_lions = 200 →
  num_lions = 2 * num_leopards →
  num_elephants = (num_lions + num_leopards) / 2 →
  num_lions + num_leopards + num_elephants = 450 := by
  sorry

#check park_population

end NUMINAMATH_CALUDE_park_population_l1057_105720


namespace NUMINAMATH_CALUDE_triangle_side_length_l1057_105779

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 2 →
  b = Real.sqrt 6 →
  A = 30 * π / 180 →
  (c = Real.sqrt 2 ∨ c = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1057_105779


namespace NUMINAMATH_CALUDE_fraction_count_l1057_105798

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def satisfies_condition (n m : ℕ) : Prop :=
  is_two_digit n ∧ is_two_digit m ∧ n * (m + 19) = m * (n + 20)

theorem fraction_count : 
  ∃! (count : ℕ), ∃ (pairs : Finset (ℕ × ℕ)),
    pairs.card = count ∧
    (∀ (pair : ℕ × ℕ), pair ∈ pairs ↔ satisfies_condition pair.1 pair.2) ∧
    count = 3 :=
sorry

end NUMINAMATH_CALUDE_fraction_count_l1057_105798


namespace NUMINAMATH_CALUDE_angle_equality_l1057_105777

theorem angle_equality (θ : Real) (h1 : Real.sqrt 2 * Real.sin (10 * π / 180) = Real.cos θ - Real.sin θ) 
                       (h2 : 0 < θ ∧ θ < π / 2) : θ = 35 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l1057_105777


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1057_105732

theorem fraction_evaluation : 
  (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1057_105732


namespace NUMINAMATH_CALUDE_baking_powder_inventory_l1057_105715

/-- Given that Kelly had 0.4 box of baking powder yesterday and 0.1 more box
    yesterday compared to now, prove that she has 0.3 box of baking powder now. -/
theorem baking_powder_inventory (yesterday : ℝ) (difference : ℝ) (now : ℝ)
    (h1 : yesterday = 0.4)
    (h2 : difference = 0.1)
    (h3 : yesterday = now + difference) :
    now = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_baking_powder_inventory_l1057_105715


namespace NUMINAMATH_CALUDE_algebraic_identity_l1057_105763

theorem algebraic_identity (x y : ℝ) : -y^2 * x + x * y^2 = 0 := by sorry

end NUMINAMATH_CALUDE_algebraic_identity_l1057_105763


namespace NUMINAMATH_CALUDE_target_not_reachable_l1057_105718

/-- Represents a point in 3D space -/
structure Point3D where
  x : Int
  y : Int
  z : Int

/-- The set of initial vertices of the cube -/
def initialVertices : Set Point3D :=
  { ⟨0,0,0⟩, ⟨0,0,1⟩, ⟨0,1,0⟩, ⟨1,0,0⟩, ⟨0,1,1⟩, ⟨1,0,1⟩, ⟨1,1,0⟩ }

/-- The target vertex we want to reach -/
def targetVertex : Point3D := ⟨1,1,1⟩

/-- Performs a symmetry operation on a point relative to another point -/
def symmetryOperation (p : Point3D) (center : Point3D) : Point3D :=
  ⟨2 * center.x - p.x, 2 * center.y - p.y, 2 * center.z - p.z⟩

/-- Defines the set of points reachable through symmetry operations -/
def reachablePoints : Set Point3D :=
  sorry  -- Definition of reachable points would go here

/-- Theorem: The target vertex is not reachable from the initial vertices -/
theorem target_not_reachable : targetVertex ∉ reachablePoints := by
  sorry


end NUMINAMATH_CALUDE_target_not_reachable_l1057_105718


namespace NUMINAMATH_CALUDE_prob_not_losing_l1057_105707

/-- The probability of A not losing in a chess game -/
theorem prob_not_losing (prob_draw prob_win : ℚ) : 
  prob_draw = 1/2 → prob_win = 1/3 → prob_draw + prob_win = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_losing_l1057_105707


namespace NUMINAMATH_CALUDE_only_triangle_combines_l1057_105712

/-- Represents a regular polygon --/
structure RegularPolygon where
  interior_angle : ℝ

/-- Checks if two regular polygons can be combined to form a 360° vertex --/
def can_combine (p1 p2 : RegularPolygon) : Prop :=
  ∃ (n m : ℕ), n * p1.interior_angle + m * p2.interior_angle = 360

/-- The given regular polygon with 150° interior angle --/
def given_polygon : RegularPolygon :=
  { interior_angle := 150 }

/-- Regular quadrilateral --/
def quadrilateral : RegularPolygon :=
  { interior_angle := 90 }

/-- Regular hexagon --/
def hexagon : RegularPolygon :=
  { interior_angle := 120 }

/-- Regular octagon --/
def octagon : RegularPolygon :=
  { interior_angle := 135 }

/-- Equilateral triangle --/
def equilateral_triangle : RegularPolygon :=
  { interior_angle := 60 }

/-- Theorem stating that only the equilateral triangle can be combined with the given polygon --/
theorem only_triangle_combines :
  ¬(can_combine given_polygon quadrilateral) ∧
  ¬(can_combine given_polygon hexagon) ∧
  ¬(can_combine given_polygon octagon) ∧
  (can_combine given_polygon equilateral_triangle) :=
sorry

end NUMINAMATH_CALUDE_only_triangle_combines_l1057_105712


namespace NUMINAMATH_CALUDE_cubic_polynomial_w_value_l1057_105755

theorem cubic_polynomial_w_value (p q r : ℂ) (u v w : ℂ) : 
  p^3 + 5*p^2 + 7*p - 18 = 0 →
  q^3 + 5*q^2 + 7*q - 18 = 0 →
  r^3 + 5*r^2 + 7*r - 18 = 0 →
  (p+q)^3 + u*(p+q)^2 + v*(p+q) + w = 0 →
  (q+r)^3 + u*(q+r)^2 + v*(q+r) + w = 0 →
  (r+p)^3 + u*(r+p)^2 + v*(r+p) + w = 0 →
  w = 179 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_w_value_l1057_105755


namespace NUMINAMATH_CALUDE_train_crossing_time_l1057_105721

theorem train_crossing_time (speed1 speed2 length1 length2 : ℝ) 
  (h1 : speed1 = 110)
  (h2 : speed2 = 90)
  (h3 : length1 = 1.10)
  (h4 : length2 = 0.9)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0)
  (h7 : length1 > 0)
  (h8 : length2 > 0) :
  (length1 + length2) / (speed1 + speed2) * 60 = 0.6 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1057_105721


namespace NUMINAMATH_CALUDE_expression_lower_bound_l1057_105794

theorem expression_lower_bound (a : ℝ) (h : a > 1) :
  a + 4 / (a - 1) ≥ 5 ∧ (a + 4 / (a - 1) = 5 ↔ a = 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_lower_bound_l1057_105794


namespace NUMINAMATH_CALUDE_hotel_air_conditioned_rooms_l1057_105790

theorem hotel_air_conditioned_rooms 
  (total_rooms : ℝ) 
  (air_conditioned_rooms : ℝ) 
  (h1 : 3 / 4 * total_rooms = total_rooms - (total_rooms - air_conditioned_rooms + (air_conditioned_rooms - 2 / 3 * air_conditioned_rooms)))
  (h2 : 2 / 3 * air_conditioned_rooms = air_conditioned_rooms - (air_conditioned_rooms - 2 / 3 * air_conditioned_rooms))
  (h3 : 4 / 5 * (1 / 4 * total_rooms) = 1 / 3 * air_conditioned_rooms) :
  air_conditioned_rooms / total_rooms = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_hotel_air_conditioned_rooms_l1057_105790


namespace NUMINAMATH_CALUDE_product_of_system_l1057_105792

theorem product_of_system (a b c d : ℚ) : 
  (4 * a + 2 * b + 5 * c + 8 * d = 67) →
  (4 * (d + c) = b) →
  (2 * b + 3 * c = a) →
  (c + 1 = d) →
  (a * b * c * d = (1201 * 572 * 19 * 124 : ℚ) / 105^4) := by
  sorry

end NUMINAMATH_CALUDE_product_of_system_l1057_105792


namespace NUMINAMATH_CALUDE_parallel_distance_theorem_l1057_105706

/-- Represents a line in a plane -/
structure Line where
  -- We don't need to define the internal structure of a line for this problem

/-- Represents the distance between two lines -/
def distance (l1 l2 : Line) : ℝ := sorry

/-- States that two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

theorem parallel_distance_theorem (a b c : Line) :
  parallel a b ∧ parallel b c ∧ parallel a c →
  distance a b = 5 →
  distance a c = 2 →
  (distance b c = 3 ∨ distance b c = 7) := by sorry

end NUMINAMATH_CALUDE_parallel_distance_theorem_l1057_105706


namespace NUMINAMATH_CALUDE_no_geometric_progression_of_2n_plus_1_l1057_105771

theorem no_geometric_progression_of_2n_plus_1 :
  ¬ ∃ (k m n : ℕ), k ≠ m ∧ m ≠ n ∧ k ≠ n ∧
    (2^m + 1)^2 = (2^k + 1) * (2^n + 1) :=
sorry

end NUMINAMATH_CALUDE_no_geometric_progression_of_2n_plus_1_l1057_105771


namespace NUMINAMATH_CALUDE_sets_problem_l1057_105752

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*x - a^2 - 2*a < 0}
def B (a : ℝ) : Set ℝ := {y | ∃ x ≤ 2, y = 3^x - 2*a}

-- State the theorem
theorem sets_problem (a : ℝ) :
  (a = 3 → A a ∪ B a = Set.Ioo (-6) 5) ∧
  (A a ∩ B a = A a ↔ a = -1 ∨ (0 ≤ a ∧ a ≤ 7/3)) := by
  sorry


end NUMINAMATH_CALUDE_sets_problem_l1057_105752


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1057_105741

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (5 * x₁^2 + 3 * x₁ - 7 = 0) →
  (5 * x₂^2 + 3 * x₂ - 7 = 0) →
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 79/25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1057_105741


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_zero_one_four_l1057_105772

-- Define set A
def A : Set ℤ := {x | x^2 - 4*x ≤ 0}

-- Define set B
def B : Set ℤ := {y | ∃ m ∈ A, y = m^2}

-- Theorem statement
theorem A_intersect_B_equals_zero_one_four : A ∩ B = {0, 1, 4} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_zero_one_four_l1057_105772


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_l1057_105783

/-- A piecewise function f(x) defined by three parts -/
noncomputable def f (a c : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then 2*a*x + 6
  else if -2 ≤ x ∧ x ≤ 2 then 3*x - 2
  else 4*x + 2*c

/-- The theorem stating that if f is continuous, then a + c = -1/2 -/
theorem continuous_piecewise_function (a c : ℝ) :
  Continuous (f a c) → a + c = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_l1057_105783


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1057_105761

/-- Vector in R^2 -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if two vectors are parallel -/
def areParallel (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

/-- Proposition p: vectors a and b are parallel -/
def p (m : ℝ) : Prop :=
  areParallel ⟨m, -2⟩ ⟨4, -2*m⟩

/-- Proposition q: m = 2 -/
def q (m : ℝ) : Prop :=
  m = 2

/-- p is necessary but not sufficient for q -/
theorem p_necessary_not_sufficient :
  (∀ m, q m → p m) ∧ (∃ m, p m ∧ ¬q m) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1057_105761


namespace NUMINAMATH_CALUDE_hoodies_solution_l1057_105780

/-- Represents the number of hoodies owned by each person -/
structure HoodieOwnership where
  fiona : ℕ
  casey : ℕ
  alex : ℕ

/-- The conditions of the hoodies problem -/
def hoodies_problem (h : HoodieOwnership) : Prop :=
  h.fiona + h.casey + h.alex = 15 ∧
  h.casey = h.fiona + 2 ∧
  h.alex = 3

/-- The solution to the hoodies problem -/
theorem hoodies_solution :
  ∃ h : HoodieOwnership, hoodies_problem h ∧ h.fiona = 5 ∧ h.casey = 7 ∧ h.alex = 3 := by
  sorry

end NUMINAMATH_CALUDE_hoodies_solution_l1057_105780


namespace NUMINAMATH_CALUDE_geometric_series_squared_sum_l1057_105727

/-- For a convergent geometric series with first term a and common ratio r,
    the sum of the series formed by the absolute values of the squares of the terms
    is a^2 / (1 - |r|^2). -/
theorem geometric_series_squared_sum (a r : ℝ) (h : |r| < 1) :
  ∑' n, |a^2 * r^(2*n)| = a^2 / (1 - |r|^2) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_squared_sum_l1057_105727


namespace NUMINAMATH_CALUDE_max_wickets_bowler_l1057_105731

/-- Represents the maximum number of wickets a bowler can take in an over -/
def max_wickets_per_over : ℕ := 3

/-- Represents the number of overs bowled by the bowler in an innings -/
def overs_bowled : ℕ := 6

/-- Represents the total number of players in a cricket team -/
def players_in_team : ℕ := 11

/-- Represents the maximum number of wickets that can be taken in an innings -/
def max_wickets_in_innings : ℕ := players_in_team - 1

/-- Theorem stating the maximum number of wickets a bowler can take in an innings -/
theorem max_wickets_bowler (wickets_per_over : ℕ) (overs : ℕ) (team_size : ℕ) :
  wickets_per_over = max_wickets_per_over →
  overs = overs_bowled →
  team_size = players_in_team →
  min (wickets_per_over * overs) (team_size - 1) = max_wickets_in_innings := by
  sorry

end NUMINAMATH_CALUDE_max_wickets_bowler_l1057_105731


namespace NUMINAMATH_CALUDE_parsley_sprigs_remaining_l1057_105748

/-- Calculates the number of parsley sprigs remaining after decorating plates. -/
theorem parsley_sprigs_remaining 
  (initial_sprigs : ℕ) 
  (whole_sprig_plates : ℕ) 
  (half_sprig_plates : ℕ) : 
  initial_sprigs = 25 → 
  whole_sprig_plates = 8 → 
  half_sprig_plates = 12 → 
  initial_sprigs - (whole_sprig_plates + half_sprig_plates / 2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_parsley_sprigs_remaining_l1057_105748


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l1057_105737

theorem part_to_whole_ratio 
  (N P : ℚ) 
  (h1 : (1/4) * (2/5) * P = 15) 
  (h2 : (40/100) * N = 180) : 
  P/N = 1/6 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l1057_105737


namespace NUMINAMATH_CALUDE_fernanda_savings_calculation_l1057_105750

/-- Calculates the final amount in Fernanda's savings account after receiving payments from debtors and making a withdrawal. -/
theorem fernanda_savings_calculation 
  (aryan_debt kyro_debt jordan_debt imani_debt : ℚ)
  (aryan_payment_percent kyro_payment_percent jordan_payment_percent imani_payment_percent : ℚ)
  (initial_savings withdrawal : ℚ) : 
  aryan_debt = 2 * kyro_debt →
  aryan_debt = 1200 →
  jordan_debt = 800 →
  imani_debt = 500 →
  aryan_payment_percent = 0.6 →
  kyro_payment_percent = 0.8 →
  jordan_payment_percent = 0.5 →
  imani_payment_percent = 0.25 →
  initial_savings = 300 →
  withdrawal = 120 →
  initial_savings + 
    (aryan_debt * aryan_payment_percent +
     kyro_debt * kyro_payment_percent +
     jordan_debt * jordan_payment_percent +
     imani_debt * imani_payment_percent) -
    withdrawal = 1905 :=
by sorry


end NUMINAMATH_CALUDE_fernanda_savings_calculation_l1057_105750


namespace NUMINAMATH_CALUDE_negation_equivalence_l1057_105710

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1057_105710


namespace NUMINAMATH_CALUDE_triangle_area_l1057_105768

/-- The area of a triangle is half the product of two adjacent sides and the sine of the angle between them. -/
theorem triangle_area (a b : ℝ) (γ : ℝ) (ha : a > 0) (hb : b > 0) (hγ : 0 < γ ∧ γ < π) :
  ∃ (S : ℝ), S = (1 / 2) * a * b * Real.sin γ ∧ S > 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1057_105768


namespace NUMINAMATH_CALUDE_problem_solution_l1057_105747

theorem problem_solution :
  (∀ x : ℝ, (-2 * x^2)^3 + x^2 * x^4 - (-3 * x^3)^2 = -16 * x^6) ∧
  (∀ a b : ℝ, (a - b)^2 * (b - a)^4 + (b - a)^3 * (a - b)^3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1057_105747


namespace NUMINAMATH_CALUDE_callie_caught_seven_frogs_l1057_105781

def alster_frogs : ℕ := 2

def quinn_frogs (alster : ℕ) : ℕ := 2 * alster

def bret_frogs (quinn : ℕ) : ℕ := 3 * quinn

def callie_frogs (bret : ℕ) : ℕ := (5 * bret) / 8

theorem callie_caught_seven_frogs :
  callie_frogs (bret_frogs (quinn_frogs alster_frogs)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_callie_caught_seven_frogs_l1057_105781


namespace NUMINAMATH_CALUDE_unique_polynomial_property_l1057_105764

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The property that P satisfies for all real x and y -/
def SatisfiesProperty (P : RealPolynomial) : Prop :=
  ∀ x y : ℝ, |y^2 - P x| ≤ 2 * |x| ↔ |x^2 - P y| ≤ 2 * |y|

/-- The theorem stating that x^2 + 1 is the unique polynomial satisfying the given properties -/
theorem unique_polynomial_property : 
  ∃! P : RealPolynomial, 
    (P 0 = 1) ∧ 
    SatisfiesProperty P ∧ 
    ∀ x : ℝ, P x = x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_polynomial_property_l1057_105764


namespace NUMINAMATH_CALUDE_ingot_growth_theorem_l1057_105704

def gold_good_multiplier : ℝ := 1.3
def silver_good_multiplier : ℝ := 1.2
def gold_bad_multiplier : ℝ := 0.7
def silver_bad_multiplier : ℝ := 0.8
def total_days : ℕ := 7

theorem ingot_growth_theorem (good_days : ℕ) 
  (h1 : good_days ≤ total_days) 
  (h2 : gold_good_multiplier ^ good_days * gold_bad_multiplier ^ (total_days - good_days) < 1)
  (h3 : silver_good_multiplier ^ good_days * silver_bad_multiplier ^ (total_days - good_days) > 1) : 
  good_days = 4 := by
  sorry

end NUMINAMATH_CALUDE_ingot_growth_theorem_l1057_105704


namespace NUMINAMATH_CALUDE_income_relationship_l1057_105716

-- Define the incomes as real numbers
variable (juan_income tim_income mary_income : ℝ)

-- State the theorem
theorem income_relationship :
  tim_income = 0.6 * juan_income →
  mary_income = 1.5 * tim_income →
  mary_income = 0.9 * juan_income :=
by
  sorry

#check income_relationship

end NUMINAMATH_CALUDE_income_relationship_l1057_105716


namespace NUMINAMATH_CALUDE_ben_mm_count_l1057_105736

theorem ben_mm_count (bryan_skittles : ℕ) (difference : ℕ) (ben_mm : ℕ) : 
  bryan_skittles = 50 →
  difference = 30 →
  bryan_skittles = ben_mm + difference →
  ben_mm = 20 := by
sorry

end NUMINAMATH_CALUDE_ben_mm_count_l1057_105736


namespace NUMINAMATH_CALUDE_board_cutting_l1057_105775

theorem board_cutting (total_length : ℝ) (shorter_length : ℝ) : 
  total_length = 69 →
  shorter_length + 2 * shorter_length = total_length →
  shorter_length = 23 := by
sorry

end NUMINAMATH_CALUDE_board_cutting_l1057_105775


namespace NUMINAMATH_CALUDE_rhinoceros_population_increase_l1057_105769

/-- Calculates the percentage increase in rhinoceros population given initial conditions --/
theorem rhinoceros_population_increase 
  (initial_rhinos : ℕ)
  (watering_area : ℕ)
  (grazing_per_rhino : ℕ)
  (total_expanded_area : ℕ)
  (h1 : initial_rhinos = 8000)
  (h2 : watering_area = 10000)
  (h3 : grazing_per_rhino = 100)
  (h4 : total_expanded_area = 890000) :
  (((total_expanded_area - (initial_rhinos * grazing_per_rhino + watering_area)) / grazing_per_rhino) / initial_rhinos : ℚ) = 1/10 :=
by sorry

end NUMINAMATH_CALUDE_rhinoceros_population_increase_l1057_105769


namespace NUMINAMATH_CALUDE_last_two_digits_of_expression_l1057_105713

theorem last_two_digits_of_expression : 
  (1941^3846 + 1961^4181 - 1981^4556 * 2141^4917) % 100 = 81 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_expression_l1057_105713


namespace NUMINAMATH_CALUDE_matrix_determinant_l1057_105753

/-- The determinant of the matrix [[x + 2, x, x+1], [x, x + 3, x], [x+1, x, x + 4]] is 6x^2 + 36x + 48 -/
theorem matrix_determinant (x : ℝ) : 
  Matrix.det !![x + 2, x, x + 1; x, x + 3, x; x + 1, x, x + 4] = 6*x^2 + 36*x + 48 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_l1057_105753


namespace NUMINAMATH_CALUDE_investment_exceeds_4_million_in_2020_l1057_105745

/-- The year when the investment first exceeds 4 million CNY -/
def first_year_exceeding_4_million : ℕ :=
  2020

/-- The initial investment in 2010 in millions of CNY -/
def initial_investment : ℝ :=
  1.3

/-- The annual increase rate as a decimal -/
def annual_increase_rate : ℝ :=
  0.12

/-- The target investment in millions of CNY -/
def target_investment : ℝ :=
  4.0

theorem investment_exceeds_4_million_in_2020 :
  initial_investment * (1 + annual_increase_rate) ^ (first_year_exceeding_4_million - 2010) > target_investment ∧
  ∀ year : ℕ, year < first_year_exceeding_4_million →
    initial_investment * (1 + annual_increase_rate) ^ (year - 2010) ≤ target_investment :=
by sorry

end NUMINAMATH_CALUDE_investment_exceeds_4_million_in_2020_l1057_105745


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1057_105799

/-- Given two vectors a and b in ℝ², prove that the angle between them is 2π/3 -/
theorem angle_between_vectors (a b : ℝ × ℝ) : 
  a = (1, Real.sqrt 3) → 
  ‖b‖ = 1 → 
  ‖a + b‖ = Real.sqrt 3 → 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖)) = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1057_105799


namespace NUMINAMATH_CALUDE_discounted_shoe_price_l1057_105703

/-- The price paid for a pair of shoes after a discount -/
theorem discounted_shoe_price (original_price : ℝ) (discount_percent : ℝ) 
  (h1 : original_price = 204)
  (h2 : discount_percent = 75) : 
  original_price * (1 - discount_percent / 100) = 51 := by
  sorry

end NUMINAMATH_CALUDE_discounted_shoe_price_l1057_105703


namespace NUMINAMATH_CALUDE_complex_equation_real_solution_l1057_105705

theorem complex_equation_real_solution (a : ℝ) :
  (∃ x : ℝ, (1 + Complex.I) * x^2 - 2 * (a + Complex.I) * x + (5 - 3 * Complex.I) = 0) →
  (a = 7/3 ∨ a = -3) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_real_solution_l1057_105705


namespace NUMINAMATH_CALUDE_neutron_electron_difference_l1057_105743

/-- Represents an atomic element -/
structure Element where
  protonNumber : ℕ
  massNumber : ℕ

/-- Calculates the number of neutrons in an element -/
def neutronCount (e : Element) : ℕ :=
  e.massNumber - e.protonNumber

/-- The number of electrons in a neutral atom is equal to the proton number -/
def electronCount (e : Element) : ℕ :=
  e.protonNumber

/-- Theorem: For an element with proton number 118 and mass number 293,
    the difference between the number of neutrons and electrons is 57 -/
theorem neutron_electron_difference (e : Element) 
    (h1 : e.protonNumber = 118) (h2 : e.massNumber = 293) : 
    neutronCount e - electronCount e = 57 := by
  sorry

end NUMINAMATH_CALUDE_neutron_electron_difference_l1057_105743


namespace NUMINAMATH_CALUDE_square_side_lengths_average_l1057_105742

theorem square_side_lengths_average (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 144) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_lengths_average_l1057_105742


namespace NUMINAMATH_CALUDE_line_parallel_plane_l1057_105757

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)
variable (parallelPlane : Plane → Plane → Prop)

-- Define the intersection of planes
variable (intersect : Plane → Plane → Line)

-- Define the "not subset of" relation for a line and a plane
variable (notSubset : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_plane 
  (l m : Line) (α β : Plane) 
  (h1 : intersect α β = m) 
  (h2 : notSubset l α) 
  (h3 : parallelLine l m) : 
  parallelLinePlane l α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_l1057_105757


namespace NUMINAMATH_CALUDE_bobs_password_probability_l1057_105789

theorem bobs_password_probability :
  let single_digit_numbers : ℕ := 10
  let even_single_digit_numbers : ℕ := 5
  let alphabet_letters : ℕ := 26
  let vowels : ℕ := 5
  let non_zero_single_digit_numbers : ℕ := 9
  let prob_even : ℚ := even_single_digit_numbers / single_digit_numbers
  let prob_vowel : ℚ := vowels / alphabet_letters
  let prob_non_zero : ℚ := non_zero_single_digit_numbers / single_digit_numbers
  prob_even * prob_vowel * prob_non_zero = 9 / 52 := by
sorry

end NUMINAMATH_CALUDE_bobs_password_probability_l1057_105789


namespace NUMINAMATH_CALUDE_multiplicative_inverse_290_mod_1721_l1057_105796

theorem multiplicative_inverse_290_mod_1721 : ∃ n : ℕ, 
  51^2 + 140^2 = 149^2 → 
  n < 1721 ∧ 
  (290 * n) % 1721 = 1 ∧ 
  n = 1456 := by
sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_290_mod_1721_l1057_105796


namespace NUMINAMATH_CALUDE_probability_of_identical_cubes_l1057_105787

/-- Represents the colors available for painting the cube faces -/
inductive Color
| Red
| Blue
| Green

/-- Represents a cube with painted faces -/
def Cube := Fin 6 → Color

/-- The total number of ways to paint a single cube -/
def totalWaysToPaintOneCube : ℕ := 729

/-- The total number of ways to paint two cubes -/
def totalWaysToPaintTwoCubes : ℕ := totalWaysToPaintOneCube * totalWaysToPaintOneCube

/-- Checks if two cubes are identical after rotation -/
def areIdenticalAfterRotation (cube1 cube2 : Cube) : Prop := sorry

/-- The number of ways two cubes can be painted to be identical after rotation -/
def waysToBeIdentical : ℕ := 66

/-- The probability that two independently painted cubes are identical after rotation -/
theorem probability_of_identical_cubes :
  (waysToBeIdentical : ℚ) / totalWaysToPaintTwoCubes = 2 / 16101 := by sorry

end NUMINAMATH_CALUDE_probability_of_identical_cubes_l1057_105787


namespace NUMINAMATH_CALUDE_domino_coloring_l1057_105739

/-- 
Given an m × n board where m and n are natural numbers and mn is even,
prove that the smallest non-negative integer V such that in each row,
the number of squares covered by red dominoes and the number of squares
covered by blue dominoes are each at most V, is equal to n.
-/
theorem domino_coloring (m n : ℕ) (h : Even (m * n)) : 
  ∃ V : ℕ, (∀ row_red row_blue : ℕ, row_red + row_blue = n → row_red ≤ V ∧ row_blue ≤ V) ∧
  (∀ W : ℕ, W < n → ∃ row_red row_blue : ℕ, row_red + row_blue = n ∧ (row_red > W ∨ row_blue > W)) :=
by sorry

end NUMINAMATH_CALUDE_domino_coloring_l1057_105739


namespace NUMINAMATH_CALUDE_f_negative_l1057_105730

-- Define an even function f
def f (x : ℝ) : ℝ := sorry

-- State the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_positive : ∀ x > 0, f x = x^2 + x

-- Theorem to prove
theorem f_negative : ∀ x < 0, f x = x^2 - x := by sorry

end NUMINAMATH_CALUDE_f_negative_l1057_105730


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1057_105797

theorem inequality_solution_set (x : ℝ) : 2 * x + 3 ≤ 1 ↔ x ≤ -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1057_105797


namespace NUMINAMATH_CALUDE_point_motion_l1057_105765

/-- The position function of a point moving in a straight line -/
def s (t : ℝ) : ℝ := 3 * t^2 - 3 * t + 8

/-- The velocity function derived from the position function -/
def v (t : ℝ) : ℝ := 6 * t - 3

/-- The acceleration function derived from the velocity function -/
def a : ℝ := 6

theorem point_motion :
  v 4 = 21 ∧ a = 6 := by sorry

end NUMINAMATH_CALUDE_point_motion_l1057_105765


namespace NUMINAMATH_CALUDE_solve_for_C_l1057_105758

theorem solve_for_C : ∃ C : ℝ, 4 * C + 3 = 31 ∧ C = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_C_l1057_105758


namespace NUMINAMATH_CALUDE_bubble_pass_probability_correct_l1057_105776

/-- Given a sequence of 50 distinct real numbers, this function calculates
    the probability that the number initially in the 25th position
    will end up in the 35th position after one bubble pass. -/
def bubble_pass_probability (seq : Fin 50 → ℝ) (h : Function.Injective seq) : ℚ :=
  1 / 1190

/-- The theorem stating that the probability is correct -/
theorem bubble_pass_probability_correct (seq : Fin 50 → ℝ) (h : Function.Injective seq) :
    bubble_pass_probability seq h = 1 / 1190 := by
  sorry

end NUMINAMATH_CALUDE_bubble_pass_probability_correct_l1057_105776


namespace NUMINAMATH_CALUDE_ratio_equality_l1057_105733

theorem ratio_equality (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1057_105733


namespace NUMINAMATH_CALUDE_certain_number_problem_l1057_105735

theorem certain_number_problem (x : ℝ) : 
  ((x + 20) * 2) / 2 - 2 = 88 / 2 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1057_105735


namespace NUMINAMATH_CALUDE_senior_sample_size_l1057_105786

theorem senior_sample_size (total : ℕ) (freshmen : ℕ) (sophomores : ℕ) (sample : ℕ) 
  (h_total : total = 900)
  (h_freshmen : freshmen = 240)
  (h_sophomores : sophomores = 260)
  (h_sample : sample = 45) :
  let seniors := total - freshmen - sophomores
  let sampling_fraction := sample / total
  seniors * sampling_fraction = 20 := by
sorry

end NUMINAMATH_CALUDE_senior_sample_size_l1057_105786


namespace NUMINAMATH_CALUDE_equation_proof_l1057_105756

theorem equation_proof : 169 + 2 * 13 * 7 + 49 = 400 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1057_105756


namespace NUMINAMATH_CALUDE_curve_symmetry_l1057_105717

/-- The curve represented by the equation xy(x+y)=1 is symmetric about the line y=x -/
theorem curve_symmetry (x y : ℝ) : x * y * (x + y) = 1 ↔ y * x * (y + x) = 1 := by sorry

end NUMINAMATH_CALUDE_curve_symmetry_l1057_105717


namespace NUMINAMATH_CALUDE_quadratic_roots_l1057_105784

theorem quadratic_roots (a : ℝ) : 
  (3^2 - 2*3 + a = 0) → 
  ((-1)^2 - 2*(-1) + a = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1057_105784


namespace NUMINAMATH_CALUDE_parallel_lines_shapes_l1057_105734

/-- Two parallel lines with marked points -/
structure ParallelLines :=
  (line1 : Finset ℕ)
  (line2 : Finset ℕ)
  (h1 : line1.card = 10)
  (h2 : line2.card = 11)

/-- The number of triangles formed by the points on parallel lines -/
def num_triangles (pl : ParallelLines) : ℕ :=
  pl.line1.card * Nat.choose pl.line2.card 2 + pl.line2.card * Nat.choose pl.line1.card 2

/-- The number of quadrilaterals formed by the points on parallel lines -/
def num_quadrilaterals (pl : ParallelLines) : ℕ :=
  Nat.choose pl.line1.card 2 * Nat.choose pl.line2.card 2

theorem parallel_lines_shapes (pl : ParallelLines) :
  num_triangles pl = 1045 ∧ num_quadrilaterals pl = 2475 := by
  sorry

#eval num_triangles ⟨Finset.range 10, Finset.range 11, rfl, rfl⟩
#eval num_quadrilaterals ⟨Finset.range 10, Finset.range 11, rfl, rfl⟩

end NUMINAMATH_CALUDE_parallel_lines_shapes_l1057_105734


namespace NUMINAMATH_CALUDE_equation_solution_l1057_105746

theorem equation_solution : 
  ∀ s : ℝ, (s^2 - 3*s + 2) / (s^2 - 6*s + 5) = (s^2 - 4*s - 5) / (s^2 - 2*s - 15) ↔ s = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1057_105746


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l1057_105788

theorem polynomial_equality_implies_sum (m n : ℝ) : 
  (∀ x : ℝ, (x + 5) * (x + n) = x^2 + m*x - 5) → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l1057_105788


namespace NUMINAMATH_CALUDE_tom_roses_count_l1057_105722

/-- The number of roses in a dozen -/
def dozen : ℕ := 12

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of dozens of roses Tom sends per day -/
def dozens_per_day : ℕ := 2

/-- The total number of roses Tom sent in a week -/
def total_roses : ℕ := dozens_per_day * dozen * days_in_week

theorem tom_roses_count : total_roses = 168 := by
  sorry

end NUMINAMATH_CALUDE_tom_roses_count_l1057_105722


namespace NUMINAMATH_CALUDE_computer_price_increase_l1057_105729

def price_increase (d : ℝ) : Prop :=
  2 * d = 540 →
  ((351 - d) / d) * 100 = 30

theorem computer_price_increase : price_increase 270 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l1057_105729


namespace NUMINAMATH_CALUDE_find_positive_integer_l1057_105762

def first_seven_multiples_of_six : List ℕ := [6, 12, 18, 24, 30, 36, 42]

def a : ℚ := (first_seven_multiples_of_six.sum : ℚ) / 7

def b (n : ℕ) : ℚ := 2 * n

theorem find_positive_integer (n : ℕ) (h : n > 0) :
  a ^ 2 - (b n) ^ 2 = 0 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_find_positive_integer_l1057_105762


namespace NUMINAMATH_CALUDE_symmetric_point_xoz_l1057_105714

/-- A point in 3D space represented by its coordinates -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoz plane in 3D space -/
def xoz_plane : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- Two points are symmetric with respect to the xoz plane if their x and z coordinates are equal,
    and their y coordinates are opposite -/
def symmetric_wrt_xoz (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = -q.y ∧ p.z = q.z

/-- Given a point P(1, 2, 3), its symmetric point Q with respect to the xoz plane
    has coordinates (1, -2, 3) -/
theorem symmetric_point_xoz :
  let P : Point3D := ⟨1, 2, 3⟩
  ∃ Q : Point3D, symmetric_wrt_xoz P Q ∧ Q = ⟨1, -2, 3⟩ :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_xoz_l1057_105714


namespace NUMINAMATH_CALUDE_seashells_total_l1057_105709

theorem seashells_total (sam_shells mary_shells : ℕ) 
  (h1 : sam_shells = 18) (h2 : mary_shells = 47) : 
  sam_shells + mary_shells = 65 := by
  sorry

end NUMINAMATH_CALUDE_seashells_total_l1057_105709


namespace NUMINAMATH_CALUDE_penelope_food_amount_l1057_105711

/-- Amount of food animals eat per day -/
structure AnimalFood where
  greta : ℝ
  penelope : ℝ
  milton : ℝ
  elmer : ℝ

/-- Conditions for animal food consumption -/
def valid_food_amounts (food : AnimalFood) : Prop :=
  food.penelope = 10 * food.greta ∧
  food.milton = food.greta / 100 ∧
  food.elmer = 4000 * food.milton ∧
  food.elmer = food.penelope + 60

theorem penelope_food_amount (food : AnimalFood) 
  (h : valid_food_amounts food) : food.penelope = 20 := by
  sorry

#check penelope_food_amount

end NUMINAMATH_CALUDE_penelope_food_amount_l1057_105711


namespace NUMINAMATH_CALUDE_wire_cutting_l1057_105728

theorem wire_cutting (total_length : ℝ) (cut_fraction : ℝ) (remaining_length : ℝ) : 
  total_length = 3 → 
  cut_fraction = 1/3 → 
  remaining_length = total_length * (1 - cut_fraction) → 
  remaining_length = 2 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l1057_105728


namespace NUMINAMATH_CALUDE_cable_length_l1057_105702

/-- Given a curve in 3D space defined by the system of equations:
    x + y + z = 10
    xy + yz + xz = 18
    This theorem states that the length of the curve is 4π√(23/3) -/
theorem cable_length (x y z : ℝ) 
  (eq1 : x + y + z = 10)
  (eq2 : x * y + y * z + x * z = 18) : 
  ∃ (curve_length : ℝ), curve_length = 4 * Real.pi * Real.sqrt (23 / 3) :=
by sorry

end NUMINAMATH_CALUDE_cable_length_l1057_105702


namespace NUMINAMATH_CALUDE_shelf_arrangement_l1057_105708

/-- The number of ways to choose k items from n items without considering order -/
def combination (n k : ℕ) : ℕ := Nat.choose n k

/-- The problem statement -/
theorem shelf_arrangement : combination 8 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_shelf_arrangement_l1057_105708


namespace NUMINAMATH_CALUDE_beka_jackson_miles_difference_l1057_105700

/-- The difference in miles flown between two people -/
def miles_difference (beka_miles jackson_miles : ℕ) : ℕ :=
  beka_miles - jackson_miles

/-- Theorem stating the difference in miles flown between Beka and Jackson -/
theorem beka_jackson_miles_difference :
  miles_difference 873 563 = 310 := by
  sorry

end NUMINAMATH_CALUDE_beka_jackson_miles_difference_l1057_105700


namespace NUMINAMATH_CALUDE_modular_exponentiation_l1057_105738

theorem modular_exponentiation (n : ℕ) :
  (47^2051 - 25^2051) % 5 = 3 := by sorry

end NUMINAMATH_CALUDE_modular_exponentiation_l1057_105738


namespace NUMINAMATH_CALUDE_three_repeated_digit_sum_theorem_l1057_105791

theorem three_repeated_digit_sum_theorem : ∃ (a b c : ℕ), 
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  let sum := 11111 * a + 1111 * b + 111 * c
  10000 ≤ sum ∧ sum < 100000 ∧
  (∀ (d₁ d₂ : ℕ), 
    d₁ < 5 ∧ d₂ < 5 ∧ d₁ ≠ d₂ → 
    (sum / (10^d₁) % 10) ≠ (sum / (10^d₂) % 10)) :=
by sorry

end NUMINAMATH_CALUDE_three_repeated_digit_sum_theorem_l1057_105791


namespace NUMINAMATH_CALUDE_even_monotonic_function_property_l1057_105744

def is_even_on (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, -a ≤ x ∧ x ≤ a → f x = f (-x)

def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ b → (f x ≤ f y ∨ f y ≤ f x)

theorem even_monotonic_function_property (f : ℝ → ℝ) 
  (h1 : is_even_on f 5)
  (h2 : is_monotonic_on f 0 5)
  (h3 : f (-3) < f 1) :
  f 1 < f 0 := by
  sorry

end NUMINAMATH_CALUDE_even_monotonic_function_property_l1057_105744


namespace NUMINAMATH_CALUDE_share_of_y_is_54_l1057_105754

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The total amount in rupees -/
def total_amount : ℝ := 234

/-- The share distribution satisfies the given conditions -/
def is_valid_distribution (s : ShareDistribution) : Prop :=
  s.y = 0.45 * s.x ∧ s.z = 0.5 * s.x ∧ s.x + s.y + s.z = total_amount

/-- The share of y in a valid distribution is 54 rupees -/
theorem share_of_y_is_54 (s : ShareDistribution) (h : is_valid_distribution s) : s.y = 54 := by
  sorry


end NUMINAMATH_CALUDE_share_of_y_is_54_l1057_105754


namespace NUMINAMATH_CALUDE_playground_transfer_l1057_105751

theorem playground_transfer (x : ℤ) : 
  (54 + x = 2 * (48 - x)) ↔ 
  (54 + x = 2 * (48 - x) ∧ 
   54 + x > 0 ∧ 
   48 - x > 0) :=
by sorry

end NUMINAMATH_CALUDE_playground_transfer_l1057_105751


namespace NUMINAMATH_CALUDE_min_disks_for_profit_l1057_105795

/-- The number of disks Maria buys for $7 -/
def buy_rate : ℕ := 5

/-- The number of disks Maria sells for $7 -/
def sell_rate : ℕ := 4

/-- The price in dollars for buying or selling the respective number of disks -/
def price : ℚ := 7

/-- The desired profit in dollars -/
def target_profit : ℚ := 125

/-- The cost of buying one disk -/
def cost_per_disk : ℚ := price / buy_rate

/-- The revenue from selling one disk -/
def revenue_per_disk : ℚ := price / sell_rate

/-- The profit made from selling one disk -/
def profit_per_disk : ℚ := revenue_per_disk - cost_per_disk

theorem min_disks_for_profit : 
  ∀ n : ℕ, (n : ℚ) * profit_per_disk ≥ target_profit → n ≥ 358 :=
by sorry

end NUMINAMATH_CALUDE_min_disks_for_profit_l1057_105795


namespace NUMINAMATH_CALUDE_committee_count_l1057_105767

theorem committee_count (n m k : ℕ) (h1 : n = 8) (h2 : m = 2) (h3 : k = 5) :
  (Nat.choose n k) - (Nat.choose (n - m) (k - m)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_l1057_105767


namespace NUMINAMATH_CALUDE_factory_employee_increase_l1057_105782

theorem factory_employee_increase (initial_employees : ℕ) (increase_percentage : ℚ) 
  (h1 : initial_employees = 852)
  (h2 : increase_percentage = 25 / 100) :
  initial_employees + (increase_percentage * initial_employees).floor = 1065 := by
  sorry

end NUMINAMATH_CALUDE_factory_employee_increase_l1057_105782


namespace NUMINAMATH_CALUDE_triangle_inequality_l1057_105793

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  a * (b^2 + c^2 - a^2) + b * (c^2 + a^2 - b^2) + c * (a^2 + b^2 - c^2) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1057_105793


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1057_105725

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_roots : a 2 * a 6 = 81 ∧ a 2 + a 6 = 34) : 
  a 4 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1057_105725
