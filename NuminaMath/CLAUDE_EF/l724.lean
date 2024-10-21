import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_bounce_count_l724_72465

noncomputable def initial_height : ℝ := 256
noncomputable def bounce_factor : ℝ := 3/4
noncomputable def target_height : ℝ := 20

noncomputable def height_after_bounces (n : ℕ) : ℝ :=
  initial_height * bounce_factor ^ n

theorem smallest_bounce_count : ∃ (n : ℕ), n = 8 ∧ 
  (∀ (m : ℕ), height_after_bounces m < target_height → m ≥ n) ∧
  height_after_bounces n < target_height := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_bounce_count_l724_72465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_with_remainders_l724_72499

theorem smallest_positive_integer_with_remainders : ∃ b : ℕ, b > 0 ∧
  (∀ n : ℕ, 0 < n → n < b → ¬(n % 3 = 2 ∧ n % 5 = 3)) ∧ 
  (b % 3 = 2 ∧ b % 5 = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_with_remainders_l724_72499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_score_product_l724_72467

def basketball_scores (scores : List Nat) : Prop :=
  scores.length = 8 ∧
  scores.take 6 = [10, 7, 8, 12, 9, 2] ∧
  scores[6]! < 15 ∧
  scores[7]! < 15 ∧
  (scores.take 7).sum % 7 = 0 ∧
  scores.sum % 8 = 0

theorem basketball_score_product (scores : List Nat) :
  basketball_scores scores → scores[6]! * scores[7]! = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_score_product_l724_72467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_sqrt_minus_one_over_sqrt_l724_72497

def summation (f : ℕ → ℝ) (a b : ℕ) : ℝ :=
  (Finset.range (b - a + 1)).sum (fun i => f (i + a))

theorem sum_equals_sqrt_minus_one_over_sqrt (n : ℕ) :
  summation (fun n => 1 / (n * Real.sqrt (n^2 - 9) + 9 * Real.sqrt n)) 3 100 =
  (Real.sqrt 103 - 1) / Real.sqrt 103 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_sqrt_minus_one_over_sqrt_l724_72497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_defined_for_all_reals_f_defined_for_all_reals_l724_72486

theorem function_defined_for_all_reals (k : ℝ) (h : k > 4/3) :
  ∀ x : ℝ, (3 * x^2 - 4 * x + k) ≠ 0 := by
  sorry

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k * x^2 - 3 * x + 4) / (3 * x^2 - 4 * x + k)

theorem f_defined_for_all_reals (k : ℝ) (h : k > 4/3) :
  ∀ x : ℝ, ∃ y : ℝ, f k x = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_defined_for_all_reals_f_defined_for_all_reals_l724_72486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_visits_correct_l724_72488

def factory_visits (total_factories : ℕ) (visits_required : ℕ) 
  (group1_visits : ℕ) (group2_visits : ℕ) (group3_visits : ℕ) : ℕ :=
  let total_visits_required := total_factories * visits_required
  let visits_made := group1_visits + group2_visits + group3_visits
  total_visits_required - visits_made

theorem factory_visits_correct (total_factories : ℕ) (visits_required : ℕ) 
  (group1_visits : ℕ) (group2_visits : ℕ) (group3_visits : ℕ) :
  factory_visits total_factories visits_required group1_visits group2_visits group3_visits =
  total_factories * visits_required - (group1_visits + group2_visits + group3_visits) :=
by
  simp [factory_visits]
  
#eval factory_visits 395 2 135 112 97

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_visits_correct_l724_72488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l724_72477

theorem trigonometric_problem (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.sin α = 3/5) 
  (h4 : Real.tan (α - β) = -1/3) : 
  Real.sin (α - β) = -Real.sqrt 10 / 10 ∧ 
  Real.cos β = 9 * Real.sqrt 10 / 50 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l724_72477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l724_72491

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- The foci of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def Foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (a^2 - b^2)
  ((c, 0), (-c, 0))

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: For any point on an ellipse, if its distance to one focus is 5,
    then its distance to the other focus is also 5 -/
theorem ellipse_focus_distance (a b : ℝ) (h : a > b) (p : ℝ × ℝ) 
    (hp : p ∈ Ellipse a b) (hd : distance p (Foci a b).1 = 5) :
  distance p (Foci a b).2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l724_72491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_square_nonagon_l724_72475

-- Define the regular polygon type
structure RegularPolygon where
  sides : ℕ
  side_length : ℝ

-- Define the square
def square : RegularPolygon :=
  { sides := 4, side_length := 1 }

-- Define the nonagon
def nonagon : RegularPolygon :=
  { sides := 9, side_length := 1 }

-- Function to calculate interior angle of a regular polygon
noncomputable def interior_angle (p : RegularPolygon) : ℝ :=
  (180 * (p.sides - 2 : ℝ)) / p.sides

-- Theorem statement
theorem exterior_angle_square_nonagon :
  let square_angle : ℝ := 90
  let nonagon_angle : ℝ := interior_angle nonagon
  360 - (square_angle + nonagon_angle) = 130 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_square_nonagon_l724_72475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l724_72494

open Real

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 6)

noncomputable def g (x : ℝ) : ℝ := sin (2 * (x + π / 6) + π / 6)

theorem shift_equivalence (x : ℝ) :
  ∃ (d : ℝ), d = -4/3 ∧
  (∀ (n : ℤ), f (x + n * d) = -4/3) →
  g x = f (x + π / 6) := by
  sorry

#check shift_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l724_72494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l724_72439

theorem range_of_a (a : ℝ) : 
  (∃ t : ℝ, t > 0 ∧ a * (2 * Real.exp 1 - t) * Real.log t = 1) → 
  a ∈ Set.Iio 0 ∪ Set.Ici (1 / Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l724_72439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_height_greater_than_103_3_l724_72481

theorem tower_height_greater_than_103_3 (angle : Real) (distance : Real) (height : Real) : 
  angle = 46 * Real.pi / 180 →
  distance = 100 →
  height = distance * (Real.tan angle) →
  height > 103.3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_height_greater_than_103_3_l724_72481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sparrow_population_decline_l724_72484

/-- The smallest positive integer n such that 100% × (0.7)^n < 20% is 5 -/
theorem sparrow_population_decline : ∃ n : ℕ, (n = 5 ∧ 100 * (0.7 : ℝ)^n < 20) ∧ ∀ m : ℕ, m < n → 100 * (0.7 : ℝ)^m ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sparrow_population_decline_l724_72484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_fourth_root_l724_72487

theorem closest_integer_to_fourth_root : 
  ∃ (n : ℤ), n = 15 ∧ 
  ∀ (m : ℤ), |m - ((15:ℝ)^4 + (10:ℝ)^4)^(1/4)| ≥ |n - ((15:ℝ)^4 + (10:ℝ)^4)^(1/4)| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_fourth_root_l724_72487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_duralumin_count_possible_l724_72408

/-- Represents a metal cube that can be either aluminum or duralumin -/
inductive MetalCube
| aluminum
| duralumin

/-- Represents the result of weighing two sets of cubes -/
inductive WeighResult
| leftHeavier
| equal
| rightHeavier

/-- Represents a two-pan balance -/
def Balance := List MetalCube → List MetalCube → WeighResult

/-- The total number of metal cubes -/
def totalCubes : Nat := 20

/-- The maximum number of weighings allowed -/
def maxWeighings : Nat := 11

/-- A function that determines the number of duralumin cubes -/
def determineDuraluminCount (cubes : List MetalCube) (balance : Balance) : Nat := sorry

/-- Theorem stating that it's possible to determine the number of duralumin cubes -/
theorem determine_duralumin_count_possible :
  ∀ (cubes : List MetalCube) (balance : Balance),
    cubes.length = totalCubes →
    ∃ (count : Nat),
      count ≤ totalCubes ∧
      count = (determineDuraluminCount cubes balance) ∧
      (∃ (weighings : Nat), weighings ≤ maxWeighings) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_duralumin_count_possible_l724_72408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exterior_angles_triangle_l724_72406

-- Define Triangle type if not already defined in Mathlib
structure Triangle where
  -- You might want to define the properties of a triangle here
  -- For example: vertices : Fin 3 → ℝ × ℝ

-- Define sum_exterior_angles function
def sum_exterior_angles (t : Triangle) : ℝ := sorry

theorem sum_exterior_angles_triangle (t : Triangle) : 
  sum_exterior_angles t = 360 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exterior_angles_triangle_l724_72406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_correct_l724_72464

/-- Given two samples with averages x and y, and their combined average z -/
def combined_average (x y z : ℝ) (l m : ℝ) : Prop :=
  y ≠ x ∧ z = l * x + m * y

/-- The line l defined by the given equation -/
def line_l (x y l m : ℝ) : Prop :=
  (l + 2) * x - (1 + 2 * m) * y + 1 - 3 * l = 0

/-- The circle with center (1, 1) and radius 2 -/
def circle_c (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 4

/-- The perpendicular line l' -/
def line_l' (x y l m : ℝ) : Prop :=
  (2 * l - 3) * x - (3 - m) * y = 0

theorem all_statements_correct (x y z l m : ℝ) 
  (h1 : combined_average x y z l m) (h2 : line_l x y l m) : 
  (line_l 1 1 l m) ∧ 
  (∃ x y, line_l x y l m ∧ circle_c x y) ∧
  (∀ x y, line_l x y l m → x^2 + y^2 ≤ 2) ∧
  (∃ k, ∀ x y, line_l x y l m ↔ line_l' (k * x) (k * y) l m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_correct_l724_72464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l724_72489

/-- Ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  shortest_distance : ℝ
  major_axis_length : ℝ
  eq_shortest_distance : shortest_distance = Real.sqrt 6 - 2
  eq_major_axis : major_axis_length = 2 * Real.sqrt 6

/-- The main theorem about the ellipse C -/
theorem ellipse_properties (C : Ellipse) :
  ∃ (E : ℝ × ℝ),
    -- Standard equation of the ellipse
    (fun (x y : ℝ) ↦ x^2 / 6 + y^2 / 2 = 1) =
    (fun (x y : ℝ) ↦ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
    -- Fixed point E on x-axis
    E.1 = 7/3 ∧ E.2 = 0 ∧
    -- Constant dot product for any line through right focus
    ∀ (k : ℝ), k ≠ 0 →
      let line := fun (x : ℝ) ↦ k * (x - 2)
      ∃ (A B : ℝ × ℝ),
        -- A and B are on the ellipse and the line
        A.2 = line A.1 ∧ B.2 = line B.1 ∧
        A.1^2 / 6 + A.2^2 / 2 = 1 ∧
        B.1^2 / 6 + B.2^2 / 2 = 1 ∧
        -- Constant dot product
        (A.1 - E.1) * (B.1 - E.1) + (A.2 - E.2) * (B.2 - E.2) = -5/9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l724_72489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l724_72427

/-- The eccentricity of a hyperbola with the given conditions is √5 -/
theorem hyperbola_eccentricity (a b c : ℝ) (F E P O : ℝ × ℝ) : 
  a > 0 → 
  b > 0 → 
  F = (-c, 0) → 
  (∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1}) → 
  (∀ (x y : ℝ), x^2 + y^2 = a^2 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2}) → 
  E ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2} → 
  (∃ t : ℝ, P = F + t • (E - F) ∧ P ∈ {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1}) → 
  O = (0, 0) → 
  E - O = (1/2) • ((F - O) + (P - O)) → 
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l724_72427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_product_integer_l724_72470

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | 1 => (3 : ℝ) ^ (1 / 17)
  | n + 2 => a (n + 1) * (a n) ^ 3

noncomputable def product_up_to (k : ℕ) : ℝ :=
  (Finset.range k).prod (λ i => a (i + 1))

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem smallest_k_product_integer :
  ∀ k : ℕ, k < 3 → ¬(is_integer (product_up_to k)) ∧
  is_integer (product_up_to 3) :=
by sorry

#check smallest_k_product_integer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_product_integer_l724_72470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_representatives_selection_l724_72474

theorem student_representatives_selection (n m k : ℕ) (hn : n = 10) (hm : m = 3) (hk : k = 2) :
  (Finset.sum (Finset.range 2) (λ i ↦ Nat.choose m (i + 1) * Nat.choose (n - m) (k - (i + 1)))) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_representatives_selection_l724_72474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_triangle_angle_sum_l724_72413

/-- The sum of interior angles of a regular polygon with n sides -/
noncomputable def interior_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

/-- The measure of each interior angle in a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ := interior_angle_sum n / n

/-- Theorem: The sum of an interior angle of a regular octagon and an interior angle of a regular triangle is 195° -/
theorem octagon_triangle_angle_sum :
  interior_angle 8 + interior_angle 3 = 195 := by
  -- Expand the definitions
  unfold interior_angle
  unfold interior_angle_sum
  -- Simplify the expressions
  simp [div_eq_mul_inv]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_triangle_angle_sum_l724_72413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_car_efficiency_l724_72483

/-- Represents the fuel efficiency of Tony's car in miles per gallon -/
def fuel_efficiency : ℝ → Prop := sorry

/-- Represents the daily round trip distance Tony drives to work -/
def daily_trip_distance : ℝ := 50

/-- Represents the number of work days per week -/
def work_days_per_week : ℕ := 5

/-- Represents the tank capacity of Tony's car in gallons -/
def tank_capacity : ℝ := 10

/-- Represents the price of gas per gallon -/
def gas_price : ℝ := 2

/-- Represents the total amount Tony spends on gas in 4 weeks -/
def total_gas_spending : ℝ := 80

/-- Represents the number of weeks considered -/
def weeks : ℕ := 4

theorem tony_car_efficiency :
  fuel_efficiency 25 :=
by
  sorry

#check tony_car_efficiency

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_car_efficiency_l724_72483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_unit_vectors_in_halfplane_l724_72414

/-- A type representing a vector in a plane --/
structure PlaneVector where
  x : ℝ
  y : ℝ

/-- The sum of a list of vectors --/
def vectorSum (vectors : List PlaneVector) : PlaneVector :=
  { x := vectors.map (·.x) |>.sum,
    y := vectors.map (·.y) |>.sum }

/-- The magnitude (length) of a vector --/
noncomputable def magnitude (v : PlaneVector) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

/-- Predicate to check if a vector is a unit vector --/
def isUnitVector (v : PlaneVector) : Prop :=
  magnitude v = 1

/-- Predicate to check if vectors lie in the same half-plane --/
def inSameHalfPlane (vectors : List PlaneVector) : Prop :=
  ∃ (n : PlaneVector), ∀ v ∈ vectors, n.x * v.x + n.y * v.y ≥ 0

theorem sum_of_odd_unit_vectors_in_halfplane
  (n : ℕ)
  (vectors : List PlaneVector)
  (h_odd : Odd n)
  (h_count : vectors.length = n)
  (h_unit : ∀ v ∈ vectors, isUnitVector v)
  (h_halfplane : inSameHalfPlane vectors) :
  magnitude (vectorSum vectors) ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_unit_vectors_in_halfplane_l724_72414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_sum_equals_one_l724_72450

/-- A triangle with altitudes and an internal point -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  M : ℝ × ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- The triangle inequality for the internal point -/
def is_internal (t : Triangle) : Prop :=
  let (xM, yM) := t.M
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  let (xC, yC) := t.C
  0 < (xB - xA) * (yM - yA) - (yB - yA) * (xM - xA) ∧
  0 < (xC - xB) * (yM - yB) - (yC - yB) * (xM - xB) ∧
  0 < (xA - xC) * (yM - yC) - (yA - yC) * (xM - xC)

/-- The theorem to be proved -/
theorem distance_ratio_sum_equals_one (t : Triangle) (h : is_internal t) :
  t.x / t.h_a + t.y / t.h_b + t.z / t.h_c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_sum_equals_one_l724_72450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_positive_l724_72431

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x > 0 then x^2 - x
  else -((-x)^2 - (-x))

-- State the theorem
theorem solution_set_of_f_positive :
  {x : ℝ | f x > 0} = Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_positive_l724_72431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n_l724_72498

theorem existence_of_n : ∃ (n : ℕ), 
  ∀ (x y : ℝ), ∃ (a : Fin n → ℝ),
    (x = (Finset.sum Finset.univ (λ i => a i))) ∧ 
    (y = (Finset.sum Finset.univ (λ i => 1 / (a i)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n_l724_72498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_fraction_of_grid_l724_72445

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Calculates the area of a right triangle given three vertices on a grid -/
def rightTriangleArea (p1 p2 p3 : GridPoint) : ℚ :=
  let base := (p3.y - p1.y : ℚ)
  let height := (p2.x - p1.x : ℚ)
  base * height / 2

/-- Calculates the area of a square grid -/
def gridArea (sideLength : ℕ) : ℚ :=
  (sideLength * sideLength : ℚ)

theorem triangle_area_fraction_of_grid :
  let p1 : GridPoint := ⟨3, 3⟩
  let p2 : GridPoint := ⟨5, 5⟩
  let p3 : GridPoint := ⟨3, 5⟩
  let gridSideLength : ℕ := 6
  (rightTriangleArea p1 p2 p3) / (gridArea gridSideLength) = 1 / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_fraction_of_grid_l724_72445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_second_quadrant_angle_l724_72411

/-- An angle in the second quadrant -/
def SecondQuadrantAngle (α : Real) : Prop :=
  Real.pi / 2 < α ∧ α < Real.pi

/-- A point on the terminal side of an angle -/
def TerminalSidePoint (α : Real) (P : Real × Real) : Prop :=
  ∃ r > 0, P.1 = r * (Real.cos α) ∧ P.2 = r * (Real.sin α)

theorem point_on_second_quadrant_angle 
  (α : Real) (x : Real) 
  (h1 : SecondQuadrantAngle α)
  (h2 : TerminalSidePoint α (x, Real.sqrt 5))
  (h3 : Real.cos α = (Real.sqrt 2 / 4) * x) :
  x = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_second_quadrant_angle_l724_72411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l724_72476

/-- Represents an ellipse with foci F₁(-c,0) and F₂(c,0), and equation x²/a² + y²/b² = 1 --/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- A point on the ellipse --/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The eccentricity of an ellipse --/
noncomputable def eccentricity (E : Ellipse) : ℝ := E.c / E.a

/-- Theorem: The eccentricity of the ellipse is in the range [√3/3, √2/2] --/
theorem eccentricity_range (E : Ellipse) (P : PointOnEllipse E) 
  (h_dot_product : (P.x + E.c) * (P.x - E.c) + P.y^2 = E.c^2) :
  Real.sqrt 3 / 3 ≤ eccentricity E ∧ eccentricity E ≤ Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l724_72476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_quarter_increase_20_percent_l724_72444

/-- Represents the share price increase percentage at the end of the first quarter -/
def first_quarter_increase (x : ℝ) : Prop := True  -- Placeholder definition

/-- The share price at the end of the second quarter is 50% higher than at the beginning of the year -/
axiom second_quarter_increase : ∀ (P : ℝ), P > 0 → P * 1.5 = P + P * 0.5

/-- The increase from the end of the first quarter to the end of the second quarter is 25% -/
axiom first_to_second_quarter_increase : 
  ∀ (P : ℝ) (x : ℝ), P > 0 → first_quarter_increase x → 
    P * (1 + x/100) * 1.25 = P * 1.5

/-- Theorem stating that the first quarter increase was 20% -/
theorem first_quarter_increase_20_percent : 
  ∀ (P : ℝ), P > 0 → first_quarter_increase 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_quarter_increase_20_percent_l724_72444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_itself_l724_72493

noncomputable def f (p q : ℝ) (x : ℝ) : ℝ :=
  if x < -1 then p * x + q else 5 * x - 10

theorem f_inverse_of_itself (p q : ℝ) :
  (∀ x, f p q (f p q x) = x) → p + q = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_itself_l724_72493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pair_inequality_l724_72460

theorem unique_pair_inequality :
  ∃! (p q : ℝ), ∀ x : ℝ, x ∈ Set.Icc 0 1 →
    |Real.sqrt (1 - x^2) - p * x - q| ≤ (Real.sqrt 2 - 1) / 2 ∧
    p = -1 ∧
    q = (Real.sqrt 2 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pair_inequality_l724_72460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l724_72425

theorem trigonometric_problem (θ : Real) 
  (h1 : Real.cos (π/4 - θ) = Real.sqrt 2/10) 
  (h2 : θ ∈ Set.Ioo 0 π) : 
  (Real.sin (π/4 + θ) = Real.sqrt 2/10) ∧ 
  (Real.sin θ^4 - Real.cos θ^4 = 7/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l724_72425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fixed_point_and_min_area_l724_72423

noncomputable section

/-- Parabola E: y² = 4x -/
def parabola_E (x y : ℝ) : Prop := y^2 = 4*x

/-- Point on parabola E -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola_E x y

/-- Fixed point Q -/
def Q : ℝ × ℝ := (9/2, 0)

/-- Dot product of vectors OA and OB -/
def dot_product (A B : PointOnParabola) : ℝ := A.x * B.x + A.y * B.y

/-- Line passing through two points -/
def line_through (A B : PointOnParabola) (x y : ℝ) : Prop :=
  (y - A.y) * (B.x - A.x) = (x - A.x) * (B.y - A.y)

/-- Perpendicular line from Q to AB -/
def perpendicular_line (A B : PointOnParabola) (x y : ℝ) : Prop :=
  (y - Q.2) * (B.x - A.x) = -(x - Q.1) * (B.y - A.y)

/-- Intersection points of perpendicular line with parabola -/
structure IntersectionPoints (A B : PointOnParabola) where
  G : PointOnParabola
  D : PointOnParabola
  on_perp_line : perpendicular_line A B G.x G.y ∧ perpendicular_line A B D.x D.y

/-- Area of quadrilateral AGBD -/
def area_AGBD (A B : PointOnParabola) (I : IntersectionPoints A B) : ℝ :=
  sorry  -- Area calculation would go here

theorem parabola_fixed_point_and_min_area 
  (A B : PointOnParabola) 
  (h_opposite : A.y * B.y < 0) 
  (h_dot_product : dot_product A B = 9/4) :
  (line_through A B Q.1 Q.2) ∧ 
  (∃ (I : IntersectionPoints A B), ∀ (J : IntersectionPoints A B), area_AGBD A B I ≤ area_AGBD A B J) ∧
  (∃ (I : IntersectionPoints A B), area_AGBD A B I = 88) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fixed_point_and_min_area_l724_72423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_square_axial_cylinder_l724_72495

/-- A cylinder with a square axial section -/
structure SquareAxialCylinder where
  /-- The side length of the square axial section -/
  side_length : ℝ
  /-- Assumption that the side length is positive -/
  side_length_pos : side_length > 0

/-- Calculate the surface area of a cylinder with a square axial section -/
noncomputable def surface_area (c : SquareAxialCylinder) : ℝ :=
  2 * Real.pi * c.side_length * c.side_length + 2 * Real.pi * (c.side_length / 2) ^ 2

/-- Theorem: The surface area of a cylinder with a square axial section of side length 2 is 6π -/
theorem surface_area_square_axial_cylinder :
  ∃ c : SquareAxialCylinder, c.side_length = 2 ∧ surface_area c = 6 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_square_axial_cylinder_l724_72495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_inverse_property_l724_72400

-- Define the functions
noncomputable def f₁ (x : ℝ) : ℝ := x - 1/x

noncomputable def f₂ (x : ℝ) : ℝ := Real.log x

noncomputable def f₃ (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x
  else if x = 1 then 0
  else -1/x

noncomputable def f₄ (x : ℝ) : ℝ := x + 1/x

-- Define the negative inverse property
def has_negative_inverse_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f (1/x) = -f x

-- State the theorem
theorem negative_inverse_property :
  (has_negative_inverse_property f₁) ∧
  (has_negative_inverse_property f₂) ∧
  (has_negative_inverse_property f₃) ∧
  ¬(has_negative_inverse_property f₄) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_inverse_property_l724_72400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_b_always_one_l724_72496

def b (n : ℕ) : ℚ := (5^n - 1) / 4

theorem gcd_b_always_one (n : ℕ) : Nat.gcd (Int.natAbs ((b n).num)) (Int.natAbs ((b (n+1)).num)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_b_always_one_l724_72496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_negative_one_fourth_l724_72418

theorem opposite_of_negative_one_fourth : 
  -(-(1 / 4 : ℚ)) = 1 / 4 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_negative_one_fourth_l724_72418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l724_72401

noncomputable def a (x : ℝ) (m : ℝ) : ℝ × ℝ := (Real.sin x, m * Real.cos x)
def b : ℝ × ℝ := (3, -1)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def f (x m : ℝ) : ℝ := dot_product (a x m) b

def symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c - x) = f (c + x)

theorem part1 (x : ℝ) :
  parallel (a x 1) b → 2 * (Real.sin x)^2 - 3 * (Real.cos x)^2 = 3/2 := by sorry

theorem part2 (m : ℝ) :
  symmetric_about (f · m) (2 * Real.pi / 3) →
  (∃ y ∈ Set.Icc (Real.pi / 8) (2 * Real.pi / 3),
    f (2 * y) m = -Real.sqrt 3 ∧
    ∃ z ∈ Set.Icc (Real.pi / 8) (2 * Real.pi / 3),
      f (2 * z) m = 2 * Real.sqrt 3) ∨
  (∃ y ∈ Set.Icc (Real.pi / 8) (2 * Real.pi / 3),
    f (2 * y) m = -2 * Real.sqrt 3 ∧
    ∃ z ∈ Set.Icc (Real.pi / 8) (2 * Real.pi / 3),
      f (2 * z) m = Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l724_72401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l724_72404

def complex (a : ℝ) : ℂ := Complex.mk 1 a * Complex.mk 2 1

theorem pure_imaginary_condition (a : ℝ) : 
  complex a = Complex.I * (complex a).im → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l724_72404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_walk_l724_72436

/-- Calculates the straight-line distance from the starting point given an eastward distance and a northeast distance at a 45-degree angle. -/
noncomputable def straightLineDistance (eastDistance : ℝ) (northeastDistance : ℝ) : ℝ :=
  Real.sqrt ((eastDistance + northeastDistance / Real.sqrt 2) ^ 2 + (northeastDistance / Real.sqrt 2) ^ 2)

/-- Theorem stating that walking 3 miles east and then 8 miles northeast at a 45-degree angle results in a straight-line distance of √73 miles from the starting point. -/
theorem distance_after_walk : straightLineDistance 3 8 = Real.sqrt 73 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_walk_l724_72436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_properties_l724_72429

noncomputable def Z : ℂ := (1/2)/(1+Complex.I) + (-5/4 + 9/4*Complex.I)

theorem Z_properties :
  (Complex.abs Z = Real.sqrt 5) ∧
  (∃ (p q : ℝ), 2 * Z^2 + p * Z + q = 0 ∧ p = 4 ∧ q = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_properties_l724_72429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_g_max_l724_72459

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x + 1 / Real.sqrt x + Real.sqrt (x + 1 / x + 1)

noncomputable def g (x : ℝ) : ℝ := Real.sqrt x + 1 / Real.sqrt x - Real.sqrt (x + 1 / x + 1)

theorem f_min_g_max (x : ℝ) (hx : x > 0) : f x ≥ 2 + Real.sqrt 3 ∧ g x ≤ 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_g_max_l724_72459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_is_only_even_l724_72478

-- Define the functions
noncomputable def f (x : ℝ) := Real.sin x
noncomputable def g (x : ℝ) := Real.cos x
noncomputable def h (x : ℝ) := Real.tan x
noncomputable def k (x : ℝ) := Real.sin (2 * x)

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statement
theorem cos_is_only_even :
  is_even g ∧ ¬is_even f ∧ ¬is_even h ∧ ¬is_even k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_is_only_even_l724_72478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_for_even_m_l724_72455

theorem solution_for_even_m (m : ℕ) (h_m_even : Even m) (h_m_pos : m > 0) :
  ∃! (n x y : ℕ), 
    n > 0 ∧ x > 0 ∧ y > 0 ∧
    Nat.Coprime m n ∧
    (x^2 + y^2)^m = (x * y)^n ∧
    n = m + 1 ∧ x = 2^(m / 2) ∧ y = 2^(m / 2) := by
  sorry

#check solution_for_even_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_for_even_m_l724_72455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_tether_problem_l724_72403

/-- Represents the grazing area of a horse tethered to the corner of a rectangular field -/
noncomputable def grazingArea (ropeLength : ℝ) : ℝ := (1/4) * Real.pi * ropeLength^2

/-- Theorem stating the relationship between the grazing area and rope length -/
theorem horse_tether_problem (fieldLength fieldWidth grazedArea : ℝ) 
  (hLength : fieldLength = 45)
  (hWidth : fieldWidth = 25)
  (hArea : grazedArea = 380.132711084365) :
  ∃ ropeLength : ℝ, grazingArea ropeLength = grazedArea ∧ 
  |ropeLength - 22| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_tether_problem_l724_72403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_good_numbers_l724_72416

/-- A natural number is a "good number" if its ones digit is 0, 1, or 2,
    its tens digit is 0, 1, 2, or 3 (excluding leading zeros for multi-digit numbers),
    and its hundreds digit is 0, 1, 2, or 3 (excluding leading zeros for three-digit numbers). -/
def isGoodNumber (n : ℕ) : Prop :=
  n < 1000 ∧
  n % 10 ≤ 2 ∧
  (n / 10) % 10 ≤ 3 ∧
  (n / 100) ≤ 3

/-- Decidable predicate for isGoodNumber -/
def isGoodNumber_decidable : DecidablePred isGoodNumber :=
  fun n => decidable_of_iff
    (n < 1000 ∧ n % 10 ≤ 2 ∧ (n / 10) % 10 ≤ 3 ∧ (n / 100) ≤ 3)
    (by simp [isGoodNumber])

instance : DecidablePred isGoodNumber := isGoodNumber_decidable

/-- The count of "good numbers" less than 1000 is 48. -/
theorem count_good_numbers : (Finset.filter isGoodNumber (Finset.range 1000)).card = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_good_numbers_l724_72416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_network_no_airline_connects_more_than_50_l724_72473

/-- Represents a city in the network -/
structure City where
  id : Nat

/-- Represents an airline in the network -/
structure Airline where
  id : Nat

/-- Represents a flight connection between two cities -/
structure Flight where
  fromCity : City
  toCity : City
  airline : Airline

/-- Represents the entire flight network -/
structure FlightNetwork where
  cities : List City
  airlines : List Airline
  flights : List Flight

/-- Checks if there's a path between two cities using a specific airline -/
def hasPath (network : FlightNetwork) (airline : Airline) (fromCity toCity : City) : Prop :=
  sorry

/-- Counts the number of cities an airline can connect -/
def connectedCitiesCount (network : FlightNetwork) (airline : Airline) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem exists_network_no_airline_connects_more_than_50 :
  ∃ (network : FlightNetwork),
    (network.cities.length = 200) ∧
    (network.airlines.length = 8) ∧
    (∀ c1 c2 : City, c1 ∈ network.cities → c2 ∈ network.cities → c1 ≠ c2 →
      ∃ f : Flight, f ∈ network.flights ∧ f.fromCity = c1 ∧ f.toCity = c2) ∧
    (∀ a : Airline, a ∈ network.airlines →
      connectedCitiesCount network a ≤ 50) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_network_no_airline_connects_more_than_50_l724_72473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_triangle_properties_l724_72492

/-- Definition of a golden triangle with vertex angle 108° --/
structure GoldenTriangle :=
  (a b c : ℝ)  -- sides of the triangle
  (α β γ : ℝ)  -- angles of the triangle in radians
  (h_α : α = 108 * π / 180)  -- vertex angle is 108°
  (h_isosceles : b = c)  -- isosceles triangle
  (h_angle_sum : α + β + γ = π)  -- angle sum in a triangle
  (h_golden_ratio : a / b = (Real.sqrt 5 - 1) / 2)  -- golden ratio property

/-- The golden ratio --/
noncomputable def φ : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_triangle_properties (T : GoldenTriangle) :
  (1 - 2 * Real.sin (9 * π / 180)^2) / (2 * φ * Real.sqrt (4 - φ^2)) = (Real.sqrt 5 + 1) / 8 ∧
  Real.cos (36 * π / 180) = (Real.sqrt 5 + 1) / 4 ∧
  Real.sin (36 * π / 180) / Real.sin (108 * π / 180) = φ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_triangle_properties_l724_72492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximized_at_40_l724_72407

/-- The volume function of the box -/
noncomputable def V (x : ℝ) : ℝ := x^2 * ((60 - x) / 2)

/-- The theorem stating that the volume is maximized when x = 40 -/
theorem volume_maximized_at_40 :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 60 ∧
  (∀ (y : ℝ), y ∈ Set.Ioo 0 60 → V y ≤ V x) ∧
  x = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximized_at_40_l724_72407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l724_72448

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
def has_solution_set (a b c : ℝ) : Prop :=
  ∀ x, f a b c x > -2 * x ↔ 1 < x ∧ x < 3

def has_equal_roots (a b c : ℝ) : Prop :=
  ∃ x, f a b c x + 6 * a = 0 ∧ 
    ∀ y, f a b c y + 6 * a = 0 → y = x

def max_value_positive (a b c : ℝ) : Prop :=
  ∃ m, (∀ x, f a b c x ≤ m) ∧ m > 0

-- State the theorem
theorem quadratic_function_properties (a b c : ℝ) 
  (h1 : a < 0) 
  (h2 : has_solution_set a b c) :
  (has_equal_roots a b c → 
    f a b c = λ x ↦ -1/5 * x^2 - 6/5 * x - 3/5) ∧
  (max_value_positive a b c → 
    (a < -2 - Real.sqrt 3 ∨ (-2 + Real.sqrt 3 < a ∧ a < 0))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l724_72448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group2_more_stable_l724_72462

noncomputable def group1_scores : List ℝ := [92, 90, 91, 96, 96]
noncomputable def group2_scores : List ℝ := [92, 96, 90, 95, 92]

noncomputable def variance (scores : List ℝ) : ℝ :=
  let mean := scores.sum / scores.length
  (scores.map (λ x => (x - mean) ^ 2)).sum / scores.length

theorem group2_more_stable : variance group2_scores < variance group1_scores := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group2_more_stable_l724_72462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_three_l724_72412

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (Real.pi * x / 3 + Real.pi / 4)

theorem min_distance_is_three (x₁ x₂ : ℝ) :
  (∀ x : ℝ, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  ∃ y₁ y₂ : ℝ, (∀ x : ℝ, f y₁ ≤ f x ∧ f x ≤ f y₂) ∧ |y₁ - y₂| = 3 ∧
  ∀ z₁ z₂ : ℝ, (∀ x : ℝ, f z₁ ≤ f x ∧ f x ≤ f z₂) → |z₁ - z₂| ≥ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_three_l724_72412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l724_72469

noncomputable def f (w : ℝ) (x : ℝ) : ℝ := Real.sin (w * x + Real.pi / 4)

theorem f_properties (w : ℝ) (h1 : w < 0) (h2 : |w| < 1) :
  -- Part 1
  (w = -1/2 →
    ∃ (T : ℝ), T = 4 * Real.pi ∧ ∀ (x : ℝ), f w (x + T) = f w x) ∧
  (w = -1/2 →
    ∀ (k : ℤ), f w (Real.pi/2 - 2*k*Real.pi) = 0) ∧
  (w = -1/2 →
    ∀ (k : ℤ) (x : ℝ), f w (-Real.pi/2 - 2*k*Real.pi + x) = -f w (-Real.pi/2 - 2*k*Real.pi - x)) ∧
  -- Part 2
  (∀ (x y : ℝ), Real.pi/2 < x ∧ x < y ∧ y < Real.pi → f w y ≤ f w x) →
  -3/4 ≤ w ∧ w < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l724_72469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l724_72458

def P : Set ℝ := {x | x^2 ≥ 1}
def M (a : ℝ) : Set ℝ := {a}

theorem range_of_a (a : ℝ) (h : P ∪ M a = P) : a ∈ Set.Iic (-1) ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l724_72458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_coefficients_l724_72447

-- Define the parametric curve
noncomputable def x (t : ℝ) : ℝ := 3 * Real.cos t + 2 * Real.sin t
noncomputable def y (t : ℝ) : ℝ := 3 * Real.sin t

-- Define the equation of the curve
def curve_equation (a b c : ℝ) (t : ℝ) : Prop :=
  a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1

-- Theorem statement
theorem curve_coefficients :
  (∀ t, curve_equation (1/9) (-4/27) (17/81) t) ∧
  ∀ a b c, (∀ t, curve_equation a b c t) → a = 1/9 ∧ b = -4/27 ∧ c = 17/81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_coefficients_l724_72447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l724_72468

/-- There exists a real number x that satisfies the given equation and is approximately 0.7 -/
theorem equation_solution : ∃ x : ℝ, 
  (x^3 - 0.5^3) / (x^2 + 0.40 + 0.5^2) = 0.3000000000000001 ∧ 
  abs (x - 0.7) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l724_72468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_calculation_l724_72440

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

/-- Rounds a real number to the nearest multiple of 10 -/
noncomputable def round_to_nearest_ten (x : ℝ) : ℤ :=
  10 * round_to_nearest (x / 10)

/-- The correct result of the calculation -/
theorem correct_calculation : round_to_nearest_ten ((57 + 68) * 2) = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_calculation_l724_72440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stop_theorem_l724_72428

/-- Given a train with speeds excluding and including stoppages, 
    calculate the number of minutes the train stops per hour -/
noncomputable def train_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : ℝ :=
  let distance_lost := speed_without_stops - speed_with_stops
  distance_lost / speed_without_stops * 60

theorem train_stop_theorem (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 45)
  (h2 : speed_with_stops = 36) :
  train_stop_time speed_without_stops speed_with_stops = 12 := by
  sorry

#check train_stop_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stop_theorem_l724_72428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_proof_l724_72420

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square ABCD with side length 3 -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

/-- A point on a line segment -/
structure PointOnSegment (A B : Point) where
  point : Point
  ratio : ℝ

/-- The intersection of two line segments -/
structure Intersection (A B C D : Point) where
  point : Point

/-- The area of a quadrilateral -/
noncomputable def quadrilateralArea (P Q R S : Point) : ℝ := sorry

theorem square_area_proof (ABCD : Square)
  (hA : ABCD.A = ⟨0, 3⟩) (hB : ABCD.B = ⟨0, 0⟩) (hC : ABCD.C = ⟨3, 0⟩) (hD : ABCD.D = ⟨3, 3⟩)
  (E : PointOnSegment ABCD.A ABCD.B) (hE : E.ratio = 2/3)
  (F : PointOnSegment ABCD.B ABCD.C) (hF : F.ratio = 2/3)
  (I : Intersection ABCD.A F.point ABCD.D E.point)
  (H : Intersection ABCD.B ABCD.D ABCD.A F.point) :
  quadrilateralArea ABCD.B E.point I.point H.point = 63/65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_proof_l724_72420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_card_all_sets_l724_72453

/-- A collection of sets satisfying the given conditions -/
structure SetCollection where
  sets : Finset (Finset ℕ)
  card_sets : sets.card = 1985
  card_each : ∀ s, s ∈ sets → s.card = 45
  union_any_two : ∀ s t, s ∈ sets → t ∈ sets → s ≠ t → (s ∪ t).card = 89

/-- The main theorem stating the cardinality of the union of all sets -/
theorem union_card_all_sets (c : SetCollection) : 
  (c.sets.biUnion id).card = 87381 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_card_all_sets_l724_72453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_any_position_reachable_l724_72482

/-- Represents the state of the stone game on an infinite line -/
def StoneGame := ℕ → Bool

/-- The initial state of the game, where stones fill one half-line -/
def initial_state : StoneGame := λ n ↦ n = 0

/-- Performs a valid move in the game -/
def make_move (state : StoneGame) (pos : ℕ) : StoneGame :=
  λ n ↦ if n = pos + 2 then true
       else if n = pos ∨ n = pos + 1 then false
       else state n

/-- A sequence of moves in the game -/
def move_sequence := List ℕ

/-- Applies a sequence of moves to the initial state -/
def apply_moves (moves : move_sequence) : StoneGame :=
  moves.foldl make_move initial_state

/-- Predicate to check if a position is reachable -/
def is_reachable (pos : ℕ) : Prop :=
  ∃ (moves : move_sequence), (apply_moves moves pos) = true

/-- The main theorem: any position is reachable -/
theorem any_position_reachable :
  ∀ (pos : ℕ), is_reachable pos := by
  sorry

#check any_position_reachable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_any_position_reachable_l724_72482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_theorem_l724_72424

noncomputable def discount_rate (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 625 then 0.8
  else if 625 ≤ x ∧ x ≤ 1000 then (0.8 * x - 100) / x
  else 0  -- undefined for other values

def low_discount_set : Set ℝ :=
  {x | (2500 ≤ x ∧ x < 3000) ∨ (3125 ≤ x ∧ x ≤ 3500)}

theorem discount_theorem :
  (∀ x, 0 < x ∧ x ≤ 1000 → discount_rate x = if x < 625 then 0.8 else (0.8 * x - 100) / x) ∧
  (discount_rate 1000 = 0.7) ∧
  (∀ x, x ∈ low_discount_set ↔ (2500 ≤ x ∧ x ≤ 3500 ∧ discount_rate x < 2/3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_theorem_l724_72424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_and_locus_l724_72410

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1/4
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 49/4

-- Define the curve E (locus of center of circle D)
def curve_E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + 4

-- Define the point P
def point_P : ℝ × ℝ := (0, 4)

-- Define the theorem
theorem circle_tangency_and_locus :
  ∀ (x_Q y_Q : ℝ),
  (∃ (k : ℝ), k ≠ 0 ∧ line_l k x_Q y_Q) →
  (∃ (x_A y_A x_B y_B : ℝ),
    hyperbola_C x_A y_A ∧
    hyperbola_C x_B y_B ∧
    line_l k x_A y_A ∧
    line_l k x_B y_B) →
  y_Q = 0 →
  x_Q ≠ 1 ∧ x_Q ≠ -1 →
  (∃ (lambda1 lambda2 : ℝ),
    lambda1 + lambda2 = -8/3 ∧
    ((x_Q = 2 ∧ y_Q = 0) ∨ (x_Q = -2 ∧ y_Q = 0))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_and_locus_l724_72410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circle_coverage_l724_72443

/-- The total area covered by circles in a regular hexagon arrangement -/
noncomputable def total_covered_area (a r : ℝ) : ℝ := Real.pi * (3 * r^2 - 4 * a * r + 2 * a^2)

/-- Theorem stating the conditions for minimum and maximum covered area -/
theorem hexagon_circle_coverage (a : ℝ) (h : a > 0) :
  ∃ (r_min r_max : ℝ),
    (a / 2 ≤ r_min ∧ r_min ≤ a * Real.sqrt 3 / 2) ∧
    (a / 2 ≤ r_max ∧ r_max ≤ a * Real.sqrt 3 / 2) ∧
    (∀ r, a / 2 ≤ r ∧ r ≤ a * Real.sqrt 3 / 2 →
      total_covered_area a r_min ≤ total_covered_area a r ∧
      total_covered_area a r ≤ total_covered_area a r_max) ∧
    r_min = 2 * a / 3 ∧
    r_max = a * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circle_coverage_l724_72443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_bounds_prime_permutation_bound_l724_72432

/-- a_{n,k} is the maximum number of permutations of a set with n elements
    where every two permutations have at least k common components -/
def a (n k : ℕ) : ℕ := sorry

/-- b_{n,k} is the maximum number of permutations of a set with n elements
    where every two permutations have at most k common components -/
def b (n k : ℕ) : ℕ := sorry

theorem permutation_bounds (n k : ℕ) (h : k ≤ n) :
  (a n k) * (b n (k-1)) ≤ Nat.factorial n := by
  sorry

theorem prime_permutation_bound (p : ℕ) (hp : Nat.Prime p) :
  a p 2 = Nat.factorial (p-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_bounds_prime_permutation_bound_l724_72432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_l724_72402

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) : ℝ → ℝ → Prop := λ x y ↦ x + m * y + 6 = 0
def l₂ (m : ℝ) : ℝ → ℝ → Prop := λ x y ↦ (m - 2) * x + 3 * y + 2 * m = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ x y, f x y ↔ g (k * x) (k * y)

-- Theorem statement
theorem lines_parallel : parallel (l₁ (-1)) (l₂ (-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_l724_72402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jan_extra_distance_l724_72480

-- Define variables
variable (d t s : ℝ)

-- Define conditions
def hans_distance (d : ℝ) : ℝ := d + 100
def hans_time (t : ℝ) : ℝ := t + 2
def hans_speed (s : ℝ) : ℝ := s + 10

def jans_time (t : ℝ) : ℝ := t + 3
def jans_speed (s : ℝ) : ℝ := s + 15

-- Theorem to prove
theorem jan_extra_distance :
  hans_distance d = hans_speed s * hans_time t →
  s * t = d →
  ∃ m : ℝ, m = jans_speed s * jans_time t ∧ m - d = 165 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jan_extra_distance_l724_72480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l724_72438

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- The problem statement -/
theorem interest_rate_problem (principal : ℝ) (time : ℝ) (rate : ℝ) :
  principal = 2000 →
  ∀ (interest_increase : ℝ),
    interest_increase = 40 →
    simple_interest principal rate (time + 4) - simple_interest principal rate time = interest_increase →
    rate = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l724_72438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_monomial_l724_72451

-- Define the monomial as noncomputable due to its dependency on Real.pi
noncomputable def monomial : ℝ → ℝ → ℝ := fun x y ↦ -((3 * Real.pi * x^2 * y) / 5)

-- State the theorem
theorem coefficient_of_monomial :
  ∃ (c : ℝ), ∀ (x y : ℝ), monomial x y = c * x^2 * y ∧ c = -(3 * Real.pi / 5) := by
  -- Introduce the coefficient
  let c := -(3 * Real.pi / 5)
  
  -- Prove existence
  use c
  
  -- Prove the equality for all x and y
  intro x y
  
  -- Split the conjunction
  constructor
  
  -- Prove the first part: monomial x y = c * x^2 * y
  · simp [monomial, c]
    -- The proof steps would go here, but we'll use sorry for now
    sorry
  
  -- Prove the second part: c = -(3 * Real.pi / 5)
  · -- This is true by definition of c
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_monomial_l724_72451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l724_72446

open Real

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * (x + 1)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + a / (Real.exp x)

theorem problem_statement :
  (∃ a : ℝ, deriv (f a) 0 = 0 ∧ a = 1) ∧
  (∀ a : ℝ, a ≤ -1 →
    (∃ m : ℝ, m = 3 ∧
      (∀ x₁ x₂ : ℝ, x₁ < x₂ → g a x₂ - g a x₁ > m * (x₂ - x₁)) ∧
      (∀ m' : ℝ, m' > m →
        ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ g a x₂ - g a x₁ ≤ m' * (x₂ - x₁)))) ∧
  (∀ n : ℕ, n > 0 →
    (Finset.sum (Finset.range n) (λ i => (2 * i + 1) ^ n) : ℝ) <
      Real.sqrt e / (e - 1) * (2 * n) ^ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l724_72446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l724_72490

/-- The area of a quadrilateral given its vertices -/
def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  sorry

/-- Given a quadrilateral ABCD with the following properties:
  - AC is a diagonal of length 10 cm
  - The perpendicular distance from B to AC is 7 cm
  - The perpendicular distance from D to AC is 3 cm
  - Angle BAC is 75 degrees
  This theorem states that the area of quadrilateral ABCD is 50 cm². -/
theorem area_of_quadrilateral (A B C D : ℝ × ℝ) (AC : ℝ) (BE DF : ℝ) (angle_BAC : ℝ) : 
  AC = 10 →
  BE = 7 →
  DF = 3 →
  angle_BAC = 75 →
  area_quadrilateral A B C D = 50 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l724_72490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_with_specific_divisor_property_l724_72479

def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem unique_integer_with_specific_divisor_property :
  ∃! x : ℕ, 
    x > 0 ∧
    num_divisors (2 * x) = num_divisors x + 2 ∧
    num_divisors (3 * x) = num_divisors x + 3 ∧
    x = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_with_specific_divisor_property_l724_72479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_APB_is_60_degrees_l724_72463

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x + 1)^2 + (y - 5)^2 = 2

-- Define the line that P is on
def line (x y : ℝ) : Prop := x + y = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := y = -x

-- Define the point P
def P : ℝ × ℝ := (-3, 3)

-- State that P is on the line
axiom P_on_line : line P.1 P.2

-- State that CP is perpendicular to y = -x
axiom CP_perp : perp_line ((P.1 + 1) / 2) ((P.2 + 5) / 2)

-- Define the center of the circle
def C : ℝ × ℝ := (-1, 5)

-- Theorem statement
theorem angle_APB_is_60_degrees :
  ∃ (A B : ℝ × ℝ), 
    circleC A.1 A.2 ∧ 
    circleC B.1 B.2 ∧ 
    (∀ (x y : ℝ), circleC x y → (x - P.1)^2 + (y - P.2)^2 ≥ (A.1 - P.1)^2 + (A.2 - P.2)^2) ∧
    (∀ (x y : ℝ), circleC x y → (x - P.1)^2 + (y - P.2)^2 ≥ (B.1 - P.1)^2 + (B.2 - P.2)^2) ∧
    Real.arccos ((A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2)) / 
              (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) * Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)) = π / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_APB_is_60_degrees_l724_72463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_A_l724_72419

def divisors_of_60 : Finset ℕ := {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}

def A : ℕ := (divisors_of_60.prod id)

theorem divisors_of_A :
  (Nat.factors A).toFinset.card = 3 ∧ Nat.card (Nat.divisors A) = 637 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_A_l724_72419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_printing_problem_solution_l724_72454

/-- Represents the printing problem with two printers -/
structure PrintingProblem where
  total_pages : ℕ
  printer_a_time : ℕ
  printer_b_extra_rate : ℕ

/-- Calculates the time taken for both printers to finish the task together -/
noncomputable def time_taken (p : PrintingProblem) : ℚ :=
  let printer_a_rate : ℚ := p.total_pages / p.printer_a_time
  let printer_b_rate : ℚ := printer_a_rate + p.printer_b_extra_rate
  let combined_rate : ℚ := printer_a_rate + printer_b_rate
  p.total_pages / combined_rate

/-- Theorem stating that for the given conditions, the time taken is 8.4 minutes -/
theorem printing_problem_solution :
  let p : PrintingProblem := ⟨35, 60, 3⟩
  time_taken p = 21/2.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_printing_problem_solution_l724_72454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l724_72437

-- Define the function f(x) = lg x + 2x - 8
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10 + 2 * x - 8

-- State the theorem
theorem root_interval (k : ℤ) : 
  (∃ x : ℝ, x > k ∧ x < k + 1 ∧ f x = 0) → k = 3 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l724_72437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_in_K_l724_72449

/-- The set of natural numbers formed by alternating digits 0 and 1, 
    with both the first and last digits being 1 -/
def K : Set ℕ :=
  {n : ℕ | ∃ (d : List ℕ), 
    n = (List.foldl (fun acc x => acc * 10 + x) 0 d) ∧
    d.head? = some 1 ∧
    d.getLast? = some 1 ∧
    (∀ i, i + 1 < d.length → d[i]? = some (1 - (d[i+1]?.getD 0)))}

/-- There is only one prime number in the set K -/
theorem unique_prime_in_K : ∃! p, p ∈ K ∧ Nat.Prime p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_in_K_l724_72449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_house_time_l724_72426

/-- Time it takes Matt to paint the house -/
noncomputable def matt_time : ℝ := 12

/-- Time it takes Patty to paint the house -/
noncomputable def patty_time (m : ℝ) : ℝ := m / 3

/-- Time it takes Rachel to paint the house -/
noncomputable def rachel_time (p : ℝ) : ℝ := 2 * p + 5

theorem paint_house_time : 
  rachel_time (patty_time matt_time) = 13 ∧ matt_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_house_time_l724_72426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l724_72442

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.sin α = -12/13) 
  (h2 : π < α ∧ α < 3*π/2) : 
  Real.tan α = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l724_72442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l724_72405

noncomputable def f (x : ℝ) := Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
    T = 4 * Real.pi ∧
    (∀ (x : ℝ), x ∈ Set.Icc (-2 * Real.pi) (2 * Real.pi) →
      (StrictMonoOn f (Set.Icc (-5 * Real.pi / 3) (Real.pi / 3)) ∧
       Set.Icc (-5 * Real.pi / 3) (Real.pi / 3) ⊆ Set.Icc (-2 * Real.pi) (2 * Real.pi))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l724_72405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_trucks_l724_72434

theorem sarah_trucks (x : ℝ) : 
  x - 13.5 - (x - 13.5) * 0.25 = 38 → 
  Int.floor x = 64 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_trucks_l724_72434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_speed_at_max_power_l724_72430

/-- The force exerted by the airflow on a sail -/
noncomputable def F (C S ρ v₀ v : ℝ) : ℝ := (C * S * ρ * (v₀ - v)^2) / 2

/-- The instantaneous power of the wind -/
noncomputable def N (C S ρ v₀ v : ℝ) : ℝ := F C S ρ v₀ v * v

/-- The theorem stating that the sailboat speed is v₀/3 when power is maximized -/
theorem sailboat_speed_at_max_power (C S ρ v₀ : ℝ) (hC : C > 0) (hS : S > 0) (hρ : ρ > 0) (hv₀ : v₀ > 0) :
  ∃ (v : ℝ), v = v₀ / 3 ∧ ∀ (u : ℝ), N C S ρ v₀ v ≥ N C S ρ v₀ u := by
  sorry

#check sailboat_speed_at_max_power

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_speed_at_max_power_l724_72430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l724_72435

theorem tan_double_angle (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.cos x = 4/5) : 
  Real.tan (2*x) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l724_72435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_width_is_60_l724_72461

/-- Represents the dimensions and road configuration of a rectangular lawn -/
structure LawnConfig where
  length : ℚ
  roadWidth : ℚ
  totalRoadArea : ℚ

/-- Calculates the width of the lawn based on the given configuration -/
def calculateLawnWidth (config : LawnConfig) : ℚ :=
  (config.totalRoadArea - config.length * config.roadWidth + config.roadWidth * config.roadWidth) / config.roadWidth

/-- Theorem stating that the width of the lawn is 60 meters given the specified configuration -/
theorem lawn_width_is_60 (config : LawnConfig) 
    (h1 : config.length = 80)
    (h2 : config.roadWidth = 10)
    (h3 : config.totalRoadArea = 1300) : 
  calculateLawnWidth config = 60 := by
  sorry

#eval calculateLawnWidth { length := 80, roadWidth := 10, totalRoadArea := 1300 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_width_is_60_l724_72461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_arithmetic_sequence_l724_72433

theorem max_lambda_arithmetic_sequence (n : ℕ+) (a : ℕ → ℝ) (S : ℕ+ → ℝ) :
  (∀ (n : ℕ+), S n = (n : ℝ) * (a 1 + a n) / 2) →
  (∀ (k : ℕ+), ∃ (d : ℝ), ∀ (i : ℕ), a (i + 1) = a i + d) →
  (∀ (n : ℕ+), n^2 * (a n)^2 + 4 * (S n)^2 ≥ (1/2) * n^2 * (a 1)^2) →
  ¬∃ (lambda : ℝ), lambda > 1/2 ∧ ∀ (n : ℕ+), n^2 * (a n)^2 + 4 * (S n)^2 ≥ lambda * n^2 * (a 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_arithmetic_sequence_l724_72433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_inequality_l724_72417

theorem incorrect_inequality (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) :
  ¬ (∀ (x y : ℝ), x < 0 → 0 < y → x^2 < x*y) := by
  intro h
  specialize h a b h1 h2
  have h3 : a^2 > 0 := by
    apply pow_two_pos_of_ne_zero
    exact ne_of_lt h1
  have h4 : a*b < 0 := by
    apply mul_neg_of_neg_of_pos h1 h2
  exact not_lt_of_gt (lt_trans h4 h3) h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_inequality_l724_72417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l724_72441

/-- Represents the work rate of one man per hour -/
noncomputable def man_work_rate : ℝ := 1

/-- Represents the work rate of one woman per hour -/
noncomputable def woman_work_rate : ℝ := (2/3) * man_work_rate

/-- The total amount of work to be done -/
noncomputable def total_work : ℝ := 21 * 20 * 9 * woman_work_rate

/-- The number of men needed to complete the work -/
def num_men : ℕ := 34

theorem work_completion :
  ∃ (x : ℕ), (x : ℝ) * 21 * 8 * man_work_rate = total_work ∧ 
  x ≥ num_men ∧ 
  ∀ (y : ℕ), (y : ℝ) * 21 * 8 * man_work_rate = total_work → y ≥ x :=
by sorry

#check work_completion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l724_72441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rotation_volume_l724_72456

/-- The volume of a cylinder formed by rotating a rectangle around one of its sides. -/
def cylinderVolume (length width : ℝ) : Set ℝ :=
  {v | v = Real.pi * length * width * width ∨ v = Real.pi * length * length * width}

/-- Theorem: The volume of a cylinder formed by rotating a 4x2 rectangle is either 16π or 32π. -/
theorem rectangle_rotation_volume :
  cylinderVolume 4 2 = {16 * Real.pi, 32 * Real.pi} := by
  sorry

#check rectangle_rotation_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rotation_volume_l724_72456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_numbers_problem_l724_72409

theorem three_numbers_problem (a b c : ℕ+) 
  (sum_condition : a + b + c = 78)
  (product_condition : a * b * c = 9240) :
  |Int.ofNat (max a.val (max b.val c.val)) - Int.ofNat (min a.val (min b.val c.val))| = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_numbers_problem_l724_72409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_distance_is_8_l724_72422

/-- The distance Bob walked when he met Yolanda -/
noncomputable def distance_bob_walked (total_distance : ℝ) (yolanda_speed : ℝ) (bob_speed : ℝ) (bob_delay : ℝ) : ℝ :=
  let meeting_time := (total_distance - yolanda_speed * bob_delay) / (yolanda_speed + bob_speed)
  bob_speed * meeting_time

/-- Theorem stating that Bob walked 8 miles when he met Yolanda -/
theorem bob_distance_is_8 :
  distance_bob_walked 17 3 4 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_distance_is_8_l724_72422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_l724_72457

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 - 2

theorem zero_of_f : 
  ∃ x : ℝ, x > 0 ∧ f x = 0 ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_l724_72457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balanced_coloring_exists_l724_72466

-- Define the Color type
inductive Color
  | Blue
  | White

-- Define a Point type for lattice points
structure Point where
  x : Int
  y : Int

-- Define the set M
def M : Set Point := sorry

-- Define a line parallel to coordinate axes
structure Line where
  isHorizontal : Bool
  coordinate : Int

-- Coloring function
def coloringFunction : Point → Color := sorry

-- Count function for points of a specific color on a line
def countColorOnLine (L : Line) (c : Color) : Nat := sorry

-- Theorem statement
theorem balanced_coloring_exists :
  ∃ (f : Point → Color), ∀ (L : Line),
    (Int.natAbs (countColorOnLine L Color.Blue - countColorOnLine L Color.White) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balanced_coloring_exists_l724_72466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_sum_l724_72471

theorem equation_solution_sum : ∃ (a b c d : ℕ+) (x y : ℝ),
  (x + y = 4) ∧ 
  (3 * x * y = 4) ∧ 
  (x = (a : ℝ) + (b : ℝ) * Real.sqrt (c : ℝ) / (d : ℝ) ∨ 
   x = (a : ℝ) - (b : ℝ) * Real.sqrt (c : ℝ) / (d : ℝ)) ∧
  (∀ (a' b' c' d' : ℕ+), 
    (x = (a' : ℝ) + (b' : ℝ) * Real.sqrt (c' : ℝ) / (d' : ℝ) ∨ 
     x = (a' : ℝ) - (b' : ℝ) * Real.sqrt (c' : ℝ) / (d' : ℝ)) →
    ((a' : ℝ) ≤ (a : ℝ) ∧ (b' : ℝ) ≤ (b : ℝ) ∧ (c' : ℝ) ≤ (c : ℝ) ∧ (d' : ℝ) ≤ (d : ℝ))) →
  a + b + c + d = 17 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_sum_l724_72471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_number_is_263_l724_72421

def sequenceNum : ℕ → ℕ
  | 0 => 11
  | 1 => 23
  | 2 => 47
  | 3 => 83
  | 4 => 131
  | 5 => 191
  | n + 6 => sequenceNum (n + 5) + (72 + n * 12)

theorem seventh_number_is_263 : sequenceNum 6 = 263 := by
  rfl

#eval sequenceNum 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_number_is_263_l724_72421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_negative_product_l724_72472

def sequenceA (n : ℕ) : ℚ :=
  if n = 0 then 3 else (11 - 2 * (n + 1)) / 3

theorem first_negative_product :
  let a := sequenceA
  (∀ k < 5, a k > 0) ∧ (a 5 > 0) ∧ (a 6 < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_negative_product_l724_72472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l724_72485

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from point P(3,-1) to the line x+3y-20=0 is 2√10 -/
theorem distance_point_to_line_example : distance_point_to_line 3 (-1) 1 3 (-20) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l724_72485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_line_to_circle_l724_72415

-- Define the line
def line (x : ℝ) : ℝ := x - 1

-- Define the circle
def mycircle (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y + 4 = 0

-- State the theorem
theorem shortest_distance_line_to_circle :
  ∃ (p : ℝ × ℝ), p.2 = line p.1 ∧
  (∀ (q : ℝ × ℝ), mycircle q.1 q.2 →
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ 2 * Real.sqrt 2 - 1) ∧
  (∃ (r : ℝ × ℝ), mycircle r.1 r.2 ∧
    Real.sqrt ((p.1 - r.1)^2 + (p.2 - r.2)^2) = 2 * Real.sqrt 2 - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_line_to_circle_l724_72415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_time_difference_l724_72452

/-- Represents the number of widgets -/
noncomputable def W : ℝ := sorry

/-- Rate of production for machine X in widgets per day -/
noncomputable def rate_X : ℝ := W / 6

/-- Rate of production for machine Y in widgets per day -/
noncomputable def rate_Y : ℝ := W / 4

/-- Combined rate of production for machines X and Y in widgets per day -/
noncomputable def combined_rate : ℝ := 5 * W / 12

/-- Time taken by machine X to produce W widgets -/
noncomputable def time_X : ℝ := W / rate_X

/-- Time taken by machine Y to produce W widgets -/
noncomputable def time_Y : ℝ := W / rate_Y

theorem machine_time_difference :
  time_X - time_Y = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_time_difference_l724_72452
