import Mathlib

namespace NUMINAMATH_CALUDE_z_modulus_l3384_338434

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition z(i+1) = i
def condition (z : ℂ) : Prop := z * (i + 1) = i

-- State the theorem
theorem z_modulus (z : ℂ) (h : condition z) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_z_modulus_l3384_338434


namespace NUMINAMATH_CALUDE_sequence_value_at_50_l3384_338459

def f (n : ℕ) : ℕ := 2 * n^3 + 3 * n^2 + n + 1

theorem sequence_value_at_50 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 25 ∧ f 3 = 65 → f 50 = 257551 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_value_at_50_l3384_338459


namespace NUMINAMATH_CALUDE_min_value_of_function_l3384_338414

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  8 + x/2 + 2/x ≥ 10 ∧ ∃ y > 0, 8 + y/2 + 2/y = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3384_338414


namespace NUMINAMATH_CALUDE_original_order_cost_l3384_338462

def original_tomatoes : ℝ := 0.99
def new_tomatoes : ℝ := 2.20
def original_lettuce : ℝ := 1.00
def new_lettuce : ℝ := 1.75
def original_celery : ℝ := 1.96
def new_celery : ℝ := 2.00
def delivery_and_tip : ℝ := 8.00
def new_bill : ℝ := 35.00

theorem original_order_cost :
  let tomatoes_diff := new_tomatoes - original_tomatoes
  let lettuce_diff := new_lettuce - original_lettuce
  let celery_diff := new_celery - original_celery
  let total_diff := tomatoes_diff + lettuce_diff + celery_diff
  new_bill - delivery_and_tip - total_diff = 25 := by sorry

end NUMINAMATH_CALUDE_original_order_cost_l3384_338462


namespace NUMINAMATH_CALUDE_late_secondary_spermatocyte_homomorphic_l3384_338472

-- Define the stages of meiosis
inductive MeiosisStage
  | PrimaryMidFirst
  | PrimaryLateFirst
  | SecondaryMidSecond
  | SecondaryLateSecond

-- Define the types of sex chromosome pairs
inductive SexChromosomePair
  | Heteromorphic
  | Homomorphic

-- Define a function that determines the sex chromosome pair for each stage
def sexChromosomePairAtStage (stage : MeiosisStage) : SexChromosomePair :=
  match stage with
  | MeiosisStage.PrimaryMidFirst => SexChromosomePair.Heteromorphic
  | MeiosisStage.PrimaryLateFirst => SexChromosomePair.Heteromorphic
  | MeiosisStage.SecondaryMidSecond => SexChromosomePair.Heteromorphic
  | MeiosisStage.SecondaryLateSecond => SexChromosomePair.Homomorphic

-- State the theorem
theorem late_secondary_spermatocyte_homomorphic :
  ∀ (stage : MeiosisStage),
    sexChromosomePairAtStage stage = SexChromosomePair.Homomorphic
    ↔ stage = MeiosisStage.SecondaryLateSecond :=
by sorry

end NUMINAMATH_CALUDE_late_secondary_spermatocyte_homomorphic_l3384_338472


namespace NUMINAMATH_CALUDE_specific_theater_seats_l3384_338409

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increment : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increment + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- Theorem stating that a theater with specific seat arrangement has 416 seats -/
theorem specific_theater_seats :
  let t : Theater := {
    first_row_seats := 14,
    seat_increment := 3,
    last_row_seats := 50
  }
  total_seats t = 416 := by
  sorry


end NUMINAMATH_CALUDE_specific_theater_seats_l3384_338409


namespace NUMINAMATH_CALUDE_babysitting_earnings_l3384_338475

/-- Calculates the earnings for a given hourly rate and number of minutes worked. -/
def calculate_earnings (hourly_rate : ℚ) (minutes_worked : ℚ) : ℚ :=
  hourly_rate * minutes_worked / 60

/-- Proves that given an hourly rate of $12 and 50 minutes of work, the earnings are equal to $10. -/
theorem babysitting_earnings :
  calculate_earnings 12 50 = 10 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_earnings_l3384_338475


namespace NUMINAMATH_CALUDE_max_soccer_balls_l3384_338412

/-- Represents the cost and quantity of soccer balls and basketballs -/
structure BallPurchase where
  soccer_cost : ℕ
  basketball_cost : ℕ
  total_balls : ℕ
  max_cost : ℕ

/-- Defines the conditions of the ball purchase problem -/
def ball_purchase_problem : BallPurchase where
  soccer_cost := 80
  basketball_cost := 60
  total_balls := 50
  max_cost := 3600

/-- Theorem stating the maximum number of soccer balls that can be purchased -/
theorem max_soccer_balls (bp : BallPurchase) : 
  bp.soccer_cost * 4 + bp.basketball_cost * 7 = 740 →
  bp.soccer_cost * 7 + bp.basketball_cost * 5 = 860 →
  ∃ (m : ℕ), m ≤ bp.total_balls ∧ 
             bp.soccer_cost * m + bp.basketball_cost * (bp.total_balls - m) ≤ bp.max_cost ∧
             ∀ (n : ℕ), n > m → 
               bp.soccer_cost * n + bp.basketball_cost * (bp.total_balls - n) > bp.max_cost :=
by sorry

#eval ball_purchase_problem.soccer_cost -- Expected output: 80
#eval ball_purchase_problem.basketball_cost -- Expected output: 60

end NUMINAMATH_CALUDE_max_soccer_balls_l3384_338412


namespace NUMINAMATH_CALUDE_cistern_water_breadth_l3384_338481

/-- Proves that for a cistern with given dimensions and wet surface area, 
    the breadth of water is 1.25 meters. -/
theorem cistern_water_breadth 
  (length : ℝ) 
  (width : ℝ) 
  (wet_surface_area : ℝ) 
  (h_length : length = 6) 
  (h_width : width = 4) 
  (h_wet_area : wet_surface_area = 49) : 
  ∃ (breadth : ℝ), 
    breadth = 1.25 ∧ 
    wet_surface_area = length * width + 2 * (length + width) * breadth :=
by sorry

end NUMINAMATH_CALUDE_cistern_water_breadth_l3384_338481


namespace NUMINAMATH_CALUDE_inscribe_smaller_circles_l3384_338417

-- Define a triangle type
structure Triangle where
  -- We don't need to specify the exact properties of a triangle here

-- Define a circle type
structure Circle where
  radius : ℝ

-- Define a function that checks if a circle can be inscribed in a triangle
def can_inscribe (t : Triangle) (c : Circle) : Prop :=
  sorry -- The exact definition is not important for this statement

-- Main theorem
theorem inscribe_smaller_circles 
  (t : Triangle) (r : ℝ) (n : ℕ) 
  (h : can_inscribe t (Circle.mk r)) :
  ∃ (circles : Finset Circle), 
    (circles.card = n^2) ∧ 
    (∀ c ∈ circles, c.radius = r / n) ∧
    (∀ c ∈ circles, can_inscribe t c) :=
sorry


end NUMINAMATH_CALUDE_inscribe_smaller_circles_l3384_338417


namespace NUMINAMATH_CALUDE_equal_diff_squares_properties_l3384_338448

-- Definition of "equal difference of squares sequence"
def is_equal_diff_squares (a : ℕ → ℝ) : Prop :=
  ∃ p : ℝ, ∀ n ≥ 2, a n ^ 2 - a (n - 1) ^ 2 = p

theorem equal_diff_squares_properties :
  -- Statement 1
  is_equal_diff_squares (fun n => (-1) ^ n) ∧
  -- Statement 2
  (∀ a : ℕ → ℝ, is_equal_diff_squares a →
    ∃ d : ℝ, ∀ n ≥ 2, a n ^ 2 - a (n - 1) ^ 2 = d) ∧
  -- Statement 3
  (∀ a : ℕ → ℝ, is_equal_diff_squares a →
    (∃ d : ℝ, ∀ n ≥ 2, a n - a (n - 1) = d) →
    ∃ c : ℝ, ∀ n, a n = c) ∧
  -- Statement 4
  ∃ a : ℕ → ℝ, is_equal_diff_squares a ∧
    ∀ k : ℕ+, is_equal_diff_squares (fun n => a (k * n)) :=
by sorry

end NUMINAMATH_CALUDE_equal_diff_squares_properties_l3384_338448


namespace NUMINAMATH_CALUDE_milk_cartons_calculation_l3384_338406

/-- Calculates the number of 1L milk cartons needed for lasagna -/
def milk_cartons_needed (servings_per_person : ℕ) : ℕ :=
  let people : ℕ := 8
  let cup_per_serving : ℚ := 1/2
  let ml_per_cup : ℕ := 250
  let ml_per_carton : ℕ := 1000
  ⌈(people * servings_per_person : ℚ) * cup_per_serving * ml_per_cup / ml_per_carton⌉₊

theorem milk_cartons_calculation (s : ℕ) :
  milk_cartons_needed s = ⌈(8 * s : ℚ) * (1/2) * 250 / 1000⌉₊ :=
by sorry

end NUMINAMATH_CALUDE_milk_cartons_calculation_l3384_338406


namespace NUMINAMATH_CALUDE_number_line_points_l3384_338471

theorem number_line_points (A B : ℝ) : 
  (|A - B| = 4 * Real.sqrt 2) → 
  (A = 3 * Real.sqrt 2) → 
  (B = -Real.sqrt 2 ∨ B = 7 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_number_line_points_l3384_338471


namespace NUMINAMATH_CALUDE_valid_a_value_l3384_338461

-- Define the linear equation
def linear_equation (a x : ℝ) : Prop := (a - 1) * x - 6 = 0

-- State the theorem
theorem valid_a_value : ∃ (a : ℝ), a ≠ 1 ∧ ∀ (x : ℝ), linear_equation a x → True :=
by
  sorry

end NUMINAMATH_CALUDE_valid_a_value_l3384_338461


namespace NUMINAMATH_CALUDE_number_of_factors_27648_l3384_338405

theorem number_of_factors_27648 : Nat.card (Nat.divisors 27648) = 44 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_27648_l3384_338405


namespace NUMINAMATH_CALUDE_binary_253_ones_minus_zeros_l3384_338493

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Count the number of true values in a list of booleans -/
def countTrue (l : List Bool) : ℕ :=
  l.foldl (fun acc b => if b then acc + 1 else acc) 0

/-- Count the number of false values in a list of booleans -/
def countFalse (l : List Bool) : ℕ :=
  l.length - countTrue l

theorem binary_253_ones_minus_zeros : 
  let binary := toBinary 253
  let ones := countTrue binary
  let zeros := countFalse binary
  ones - zeros = 6 := by sorry

end NUMINAMATH_CALUDE_binary_253_ones_minus_zeros_l3384_338493


namespace NUMINAMATH_CALUDE_sum_of_squares_residuals_l3384_338473

/-- Linear Regression Sum of Squares -/
structure LinearRegressionSS where
  SST : ℝ  -- Total sum of squares
  SSR : ℝ  -- Sum of squares due to regression
  SSE : ℝ  -- Sum of squares for residuals

/-- Theorem: Sum of Squares for Residuals in Linear Regression -/
theorem sum_of_squares_residuals 
  (lr : LinearRegressionSS) 
  (h1 : lr.SST = 13) 
  (h2 : lr.SSR = 10) 
  (h3 : lr.SST = lr.SSR + lr.SSE) : 
  lr.SSE = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_residuals_l3384_338473


namespace NUMINAMATH_CALUDE_parabola_equation_l3384_338428

/-- Given a point A(1,1) and a parabola C: y^2 = 2px (p > 0) whose focus lies on the perpendicular
    bisector of OA, prove that the equation of the parabola C is y^2 = 4x. -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) : 
  let A : ℝ × ℝ := (1, 1)
  let O : ℝ × ℝ := (0, 0)
  let perpendicular_bisector := {(x, y) : ℝ × ℝ | x + y = 1}
  let focus : ℝ × ℝ := (p / 2, 0)
  focus ∈ perpendicular_bisector →
  ∀ x y : ℝ, y^2 = 2*p*x ↔ y^2 = 4*x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3384_338428


namespace NUMINAMATH_CALUDE_race_distance_l3384_338411

/-- Represents the race scenario -/
structure Race where
  distance : ℝ
  time_A : ℝ
  time_B : ℝ
  speed_A : ℝ
  speed_B : ℝ

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  r.time_A = 33 ∧
  r.speed_A = r.distance / r.time_A ∧
  r.speed_B = (r.distance - 35) / r.time_A ∧
  r.speed_B = 35 / 7 ∧
  r.time_B = r.time_A + 7

/-- The theorem stating that the race distance is 200 meters -/
theorem race_distance (r : Race) (h : race_conditions r) : r.distance = 200 :=
sorry

end NUMINAMATH_CALUDE_race_distance_l3384_338411


namespace NUMINAMATH_CALUDE_binomial_coefficient_is_integer_l3384_338466

theorem binomial_coefficient_is_integer (m n : ℕ) (h : m > n) :
  ∃ k : ℕ, (m.choose n) = k := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_is_integer_l3384_338466


namespace NUMINAMATH_CALUDE_symmetry_correctness_l3384_338456

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry operations in 3D space -/
def symmetry_x_axis (p : Point3D) : Point3D := ⟨p.x, -p.y, -p.z⟩
def symmetry_yoz_plane (p : Point3D) : Point3D := ⟨-p.x, p.y, p.z⟩
def symmetry_y_axis (p : Point3D) : Point3D := ⟨-p.x, p.y, -p.z⟩
def symmetry_origin (p : Point3D) : Point3D := ⟨-p.x, -p.y, -p.z⟩

/-- The theorem to be proved -/
theorem symmetry_correctness (a b c : ℝ) : 
  let M : Point3D := ⟨a, b, c⟩
  (symmetry_x_axis M ≠ ⟨a, -b, c⟩) ∧ 
  (symmetry_yoz_plane M ≠ ⟨a, -b, -c⟩) ∧ 
  (symmetry_y_axis M ≠ ⟨a, -b, c⟩) ∧ 
  (symmetry_origin M = ⟨-a, -b, -c⟩) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_correctness_l3384_338456


namespace NUMINAMATH_CALUDE_benjamin_weekly_miles_l3384_338457

/-- Calculates the total miles Benjamin walks in a week --/
def total_miles_walked (work_distance : ℕ) (dog_walk_distance : ℕ) (friend_distance : ℕ) (store_distance : ℕ) : ℕ :=
  let work_trips := 2 * work_distance * 5
  let dog_walks := 2 * dog_walk_distance * 7
  let friend_visit := 2 * friend_distance
  let store_trips := 2 * store_distance * 2
  work_trips + dog_walks + friend_visit + store_trips

/-- Theorem stating that Benjamin walks 102 miles in a week --/
theorem benjamin_weekly_miles :
  total_miles_walked 6 2 1 3 = 102 := by
  sorry

#eval total_miles_walked 6 2 1 3

end NUMINAMATH_CALUDE_benjamin_weekly_miles_l3384_338457


namespace NUMINAMATH_CALUDE_intersection_A_B_solution_set_a_eq_1_solution_set_a_gt_1_solution_set_a_lt_1_l3384_338437

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -4 < x ∧ x < 1 ∨ 3 < x ∧ x < 4} := by sorry

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := x^2 - (a + 1)*x + a < 0

-- Theorems for the solution sets of the inequality
theorem solution_set_a_eq_1 : {x | inequality 1 x} = ∅ := by sorry

theorem solution_set_a_gt_1 (a : ℝ) (h : a > 1) : 
  {x | inequality a x} = {x | 1 < x ∧ x < a} := by sorry

theorem solution_set_a_lt_1 (a : ℝ) (h : a < 1) : 
  {x | inequality a x} = {x | a < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_solution_set_a_eq_1_solution_set_a_gt_1_solution_set_a_lt_1_l3384_338437


namespace NUMINAMATH_CALUDE_russian_alphabet_symmetry_partition_l3384_338489

-- Define the set of Russian alphabet letters
inductive RussianLetter
| A | B | V | G | D | E | Zh | Z | I | K | L | M | N | O | P | R | S | T | U | F | Kh | Ts | Ch | Sh | Shch | Eh | Yu | Ya

-- Define symmetry types
inductive SymmetryType
| Vertical
| Horizontal
| Central
| All
| None

-- Define a function that assigns a symmetry type to each letter
def letterSymmetry : RussianLetter → SymmetryType
| RussianLetter.A => SymmetryType.Vertical
| RussianLetter.D => SymmetryType.Vertical
| RussianLetter.M => SymmetryType.Vertical
| RussianLetter.P => SymmetryType.Vertical
| RussianLetter.T => SymmetryType.Vertical
| RussianLetter.Sh => SymmetryType.Vertical
| RussianLetter.V => SymmetryType.Horizontal
| RussianLetter.E => SymmetryType.Horizontal
| RussianLetter.Z => SymmetryType.Horizontal
| RussianLetter.K => SymmetryType.Horizontal
| RussianLetter.S => SymmetryType.Horizontal
| RussianLetter.Eh => SymmetryType.Horizontal
| RussianLetter.Yu => SymmetryType.Horizontal
| RussianLetter.I => SymmetryType.Central
| RussianLetter.Zh => SymmetryType.All
| RussianLetter.N => SymmetryType.All
| RussianLetter.O => SymmetryType.All
| RussianLetter.F => SymmetryType.All
| RussianLetter.Kh => SymmetryType.All
| _ => SymmetryType.None

-- Define the five groups
def group1 := {l : RussianLetter | letterSymmetry l = SymmetryType.Vertical}
def group2 := {l : RussianLetter | letterSymmetry l = SymmetryType.Horizontal}
def group3 := {l : RussianLetter | letterSymmetry l = SymmetryType.Central}
def group4 := {l : RussianLetter | letterSymmetry l = SymmetryType.All}
def group5 := {l : RussianLetter | letterSymmetry l = SymmetryType.None}

-- Theorem: The groups form a partition of the Russian alphabet
theorem russian_alphabet_symmetry_partition :
  (∀ l : RussianLetter, l ∈ group1 ∨ l ∈ group2 ∨ l ∈ group3 ∨ l ∈ group4 ∨ l ∈ group5) ∧
  (group1 ∩ group2 = ∅) ∧ (group1 ∩ group3 = ∅) ∧ (group1 ∩ group4 = ∅) ∧ (group1 ∩ group5 = ∅) ∧
  (group2 ∩ group3 = ∅) ∧ (group2 ∩ group4 = ∅) ∧ (group2 ∩ group5 = ∅) ∧
  (group3 ∩ group4 = ∅) ∧ (group3 ∩ group5 = ∅) ∧
  (group4 ∩ group5 = ∅) :=
sorry

end NUMINAMATH_CALUDE_russian_alphabet_symmetry_partition_l3384_338489


namespace NUMINAMATH_CALUDE_opposite_sides_iff_m_range_l3384_338446

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation 3x - y + m = 0 -/
def lineEquation (p : Point) (m : ℝ) : ℝ := 3 * p.x - p.y + m

/-- Two points are on opposite sides of the line if the product of their line equations is negative -/
def oppositeSides (p1 p2 : Point) (m : ℝ) : Prop :=
  lineEquation p1 m * lineEquation p2 m < 0

/-- The theorem stating the equivalence between the points being on opposite sides and the range of m -/
theorem opposite_sides_iff_m_range (m : ℝ) :
  oppositeSides (Point.mk 1 2) (Point.mk 1 1) m ↔ -2 < m ∧ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_iff_m_range_l3384_338446


namespace NUMINAMATH_CALUDE_candy_mixture_price_l3384_338404

/-- Given two types of candy mixed together, prove the price per pound of the mixture -/
theorem candy_mixture_price (X : ℝ) (price_X : ℝ) (weight_Y : ℝ) (price_Y : ℝ) 
  (total_weight : ℝ) (h1 : price_X = 3.50) (h2 : weight_Y = 6.25) (h3 : price_Y = 4.30) 
  (h4 : total_weight = 10) (h5 : X + weight_Y = total_weight) : 
  (X * price_X + weight_Y * price_Y) / total_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_price_l3384_338404


namespace NUMINAMATH_CALUDE_min_sphere_surface_area_for_pyramid_l3384_338490

/-- Minimum surface area of a sphere containing a specific triangular pyramid -/
theorem min_sphere_surface_area_for_pyramid (V : ℝ) (h : ℝ) (angle : ℝ) : 
  V = 8 * Real.sqrt 3 →
  h = 4 →
  angle = π / 3 →
  ∃ (S : ℝ), S = 48 * π ∧ 
    ∀ (S' : ℝ), (∃ (r : ℝ), S' = 4 * π * r^2 ∧ 
      ∃ (a b c : ℝ), 
        a^2 + (h/2)^2 ≤ r^2 ∧
        b^2 + (h/2)^2 ≤ r^2 ∧
        c^2 + h^2 ≤ r^2 ∧
        (1/3) * (1/2) * a * b * Real.sin angle * h = V) → 
    S ≤ S' :=
sorry

end NUMINAMATH_CALUDE_min_sphere_surface_area_for_pyramid_l3384_338490


namespace NUMINAMATH_CALUDE_bus_passengers_problem_l3384_338424

/-- Proves that the initial number of people on a bus was 50, given the conditions of passenger changes at three stops. -/
theorem bus_passengers_problem (initial : ℕ) : 
  (((initial - 15) - (8 - 2)) - (4 - 3) = 28) → initial = 50 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_problem_l3384_338424


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l3384_338482

theorem system_of_equations_sum (x y z : ℝ) 
  (eq1 : y + z = 16 - 4*x)
  (eq2 : x + z = -18 - 4*y)
  (eq3 : x + y = 13 - 4*z) :
  2*x + 2*y + 2*z = 11/3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l3384_338482


namespace NUMINAMATH_CALUDE_total_arrangements_l3384_338476

-- Define the number of people
def total_people : ℕ := 5

-- Define the number of positions for person A
def positions_for_A : ℕ := 2

-- Define the number of positions for person B
def positions_for_B : ℕ := 3

-- Define the number of remaining people
def remaining_people : ℕ := total_people - 2

-- Theorem statement
theorem total_arrangements :
  (positions_for_A * positions_for_B * (Nat.factorial remaining_people)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_arrangements_l3384_338476


namespace NUMINAMATH_CALUDE_greater_number_proof_l3384_338422

theorem greater_number_proof (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 8) (h3 : x > y) : x = 19 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l3384_338422


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l3384_338451

theorem abs_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ - 3| = 15) ∧ 
  (|x₂ - 3| = 15) ∧ 
  (x₁ ≠ x₂) ∧ 
  (|x₁ - x₂| = 30) := by
sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l3384_338451


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3384_338450

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 - 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3384_338450


namespace NUMINAMATH_CALUDE_yummy_kibble_percentage_proof_l3384_338449

/-- The number of vets in the state -/
def total_vets : ℕ := 1000

/-- The percentage of vets recommending Puppy Kibble -/
def puppy_kibble_percentage : ℚ := 20 / 100

/-- The number of additional vets recommending Yummy Dog Kibble compared to Puppy Kibble -/
def additional_yummy_kibble_vets : ℕ := 100

/-- The percentage of vets recommending Yummy Dog Kibble -/
def yummy_kibble_percentage : ℚ := 30 / 100

theorem yummy_kibble_percentage_proof :
  (puppy_kibble_percentage * total_vets + additional_yummy_kibble_vets : ℚ) / total_vets = yummy_kibble_percentage := by
  sorry

end NUMINAMATH_CALUDE_yummy_kibble_percentage_proof_l3384_338449


namespace NUMINAMATH_CALUDE_max_pogs_purchase_l3384_338496

theorem max_pogs_purchase (x y z : ℕ) : 
  x ≥ 1 → y ≥ 1 → z ≥ 1 →
  3 * x + 4 * y + 9 * z = 75 →
  z ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_pogs_purchase_l3384_338496


namespace NUMINAMATH_CALUDE_angle_x_value_l3384_338465

/-- Given a configuration where AB and CD are straight lines, with specific angle measurements, prove that angle x equals 35 degrees. -/
theorem angle_x_value (AXB CYX XYB : ℝ) (h1 : AXB = 150) (h2 : CYX = 130) (h3 : XYB = 55) : ∃ x : ℝ, x = 35 := by
  sorry

end NUMINAMATH_CALUDE_angle_x_value_l3384_338465


namespace NUMINAMATH_CALUDE_negative_polynomial_count_l3384_338420

theorem negative_polynomial_count : 
  ∃ (S : Finset ℤ), (∀ x ∈ S, x^5 - 51*x^3 + 50*x < 0) ∧ 
                    (∀ x : ℤ, x^5 - 51*x^3 + 50*x < 0 → x ∈ S) ∧ 
                    Finset.card S = 12 :=
by sorry

end NUMINAMATH_CALUDE_negative_polynomial_count_l3384_338420


namespace NUMINAMATH_CALUDE_gcd_15_70_l3384_338497

theorem gcd_15_70 : Nat.gcd 15 70 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_15_70_l3384_338497


namespace NUMINAMATH_CALUDE_liquid_volume_in_tin_l3384_338444

/-- The volume of liquid in a cylindrical tin with a conical cavity -/
theorem liquid_volume_in_tin (tin_diameter tin_height : ℝ) 
  (liquid_fill_ratio : ℝ) (cavity_height cavity_diameter : ℝ) : 
  tin_diameter = 10 →
  tin_height = 5 →
  liquid_fill_ratio = 2/3 →
  cavity_height = 2 →
  cavity_diameter = 4 →
  (liquid_fill_ratio * tin_height * π * (tin_diameter/2)^2 - 
   (1/3) * π * (cavity_diameter/2)^2 * cavity_height) = (242/3) * π := by
  sorry

#check liquid_volume_in_tin

end NUMINAMATH_CALUDE_liquid_volume_in_tin_l3384_338444


namespace NUMINAMATH_CALUDE_acid_dilution_l3384_338464

/-- Given an initial acid solution with concentration p% and volume p ounces,
    adding y ounces of water results in a (p-15)% acid solution.
    This theorem proves that y = 15p / (p-15) when p > 30. -/
theorem acid_dilution (p : ℝ) (y : ℝ) (h : p > 30) :
  (p * p / 100 = (p - 15) / 100 * (p + y)) → y = 15 * p / (p - 15) := by
  sorry


end NUMINAMATH_CALUDE_acid_dilution_l3384_338464


namespace NUMINAMATH_CALUDE_inclination_angle_range_l3384_338467

theorem inclination_angle_range (α : ℝ) (θ : ℝ) : 
  (∃ x y : ℝ, x * Real.sin α - y + 1 = 0) →
  0 ≤ θ ∧ θ < π →
  (θ ∈ Set.Icc 0 (π/4) ∪ Set.Ico (3*π/4) π) ↔ 
  (∃ x y : ℝ, x * Real.sin α - y + 1 = 0 ∧ θ = Real.arctan (Real.sin α)) :=
by sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l3384_338467


namespace NUMINAMATH_CALUDE_set_A_characterization_l3384_338435

theorem set_A_characterization (A : Set ℕ) : 
  ({1} ∪ A = {1, 3, 5}) → (A = {1, 3, 5} ∨ A = {3, 5}) := by
  sorry

end NUMINAMATH_CALUDE_set_A_characterization_l3384_338435


namespace NUMINAMATH_CALUDE_mistake_correction_l3384_338425

theorem mistake_correction (x : ℤ) : x - 23 = 4 → x * 23 = 621 := by
  sorry

end NUMINAMATH_CALUDE_mistake_correction_l3384_338425


namespace NUMINAMATH_CALUDE_complete_square_and_calculate_l3384_338401

theorem complete_square_and_calculate :
  ∀ m n p : ℝ,
  (∀ x : ℝ, 2 * x^2 - 8 * x + 19 = m * (x - n)^2 + p) →
  2017 + m * p - 5 * n = 2029 := by
sorry

end NUMINAMATH_CALUDE_complete_square_and_calculate_l3384_338401


namespace NUMINAMATH_CALUDE_shaded_square_area_fraction_l3384_338419

/-- The area of a square with vertices at (3,2), (5,4), (3,6), and (1,4) on a 6x6 grid is 2/9 of the total grid area. -/
theorem shaded_square_area_fraction :
  let grid_size : ℕ := 6
  let total_area : ℝ := (grid_size : ℝ) ^ 2
  let shaded_square_vertices : List (ℕ × ℕ) := [(3, 2), (5, 4), (3, 6), (1, 4)]
  let shaded_square_side : ℝ := 2 * Real.sqrt 2
  let shaded_square_area : ℝ := shaded_square_side ^ 2
  shaded_square_area / total_area = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_shaded_square_area_fraction_l3384_338419


namespace NUMINAMATH_CALUDE_corners_count_is_even_l3384_338427

/-- A corner is a shape on a grid paper -/
structure Corner where
  position : ℤ × ℤ

/-- A rectangle is a 1x4 shape on a grid paper -/
structure Rectangle where
  position : ℤ × ℤ

/-- A centrally symmetric figure on a grid paper -/
structure CentrallySymmetricFigure where
  corners : List Corner
  rectangles : List Rectangle
  is_centrally_symmetric : Bool

/-- The theorem states that in a centrally symmetric figure composed of corners and 1x4 rectangles, 
    the number of corners must be even -/
theorem corners_count_is_even (figure : CentrallySymmetricFigure) 
  (h : figure.is_centrally_symmetric = true) : 
  Even (figure.corners.length) := by
  sorry

end NUMINAMATH_CALUDE_corners_count_is_even_l3384_338427


namespace NUMINAMATH_CALUDE_probability_yellow_marble_l3384_338415

theorem probability_yellow_marble (blue red yellow : ℕ) 
  (h_blue : blue = 7)
  (h_red : red = 11)
  (h_yellow : yellow = 6) :
  (yellow : ℚ) / (blue + red + yellow) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_marble_l3384_338415


namespace NUMINAMATH_CALUDE_excircle_incircle_similarity_l3384_338469

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A B C : Point)

/-- Represents a circle defined by its center and a point on the circumference -/
structure Circle :=
  (center : Point) (point : Point)

/-- Defines an excircle of a triangle -/
def excircle (T : Triangle) (vertex : Point) : Circle :=
  sorry

/-- Defines the incircle of a triangle -/
def incircle (T : Triangle) : Circle :=
  sorry

/-- Defines the circumcircle of a triangle -/
def circumcircle (T : Triangle) : Circle :=
  sorry

/-- Defines the point where a circle touches a line segment -/
def touchPoint (C : Circle) (A B : Point) : Point :=
  sorry

/-- Defines the intersection points of two circles -/
def circleIntersection (C1 C2 : Circle) : Set Point :=
  sorry

/-- Checks if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop :=
  sorry

theorem excircle_incircle_similarity
  (ABC : Triangle)
  (A' : Point) (B' : Point) (C' : Point)
  (C1 : Point) (A1 : Point) (B1 : Point) :
  A' = touchPoint (excircle ABC ABC.A) ABC.B ABC.C →
  B' = touchPoint (excircle ABC ABC.B) ABC.C ABC.A →
  C' = touchPoint (excircle ABC ABC.C) ABC.A ABC.B →
  C1 ∈ circleIntersection (circumcircle ABC) (circumcircle ⟨A', B', C⟩) →
  A1 ∈ circleIntersection (circumcircle ABC) (circumcircle ⟨ABC.A, B', C'⟩) →
  B1 ∈ circleIntersection (circumcircle ABC) (circumcircle ⟨A', ABC.B, C'⟩) →
  let incirclePoints := Triangle.mk
    (touchPoint (incircle ABC) ABC.B ABC.C)
    (touchPoint (incircle ABC) ABC.C ABC.A)
    (touchPoint (incircle ABC) ABC.A ABC.B)
  areSimilar ⟨A1, B1, C1⟩ incirclePoints :=
by
  sorry

end NUMINAMATH_CALUDE_excircle_incircle_similarity_l3384_338469


namespace NUMINAMATH_CALUDE_total_cost_is_eight_times_shorts_l3384_338470

/-- The cost of football equipment relative to shorts -/
def FootballEquipmentCost (x : ℝ) : Prop :=
  let shorts := x
  let tshirt := x
  let boots := 4 * x
  let shinguards := 2 * x
  (shorts + tshirt = 2 * x) ∧
  (shorts + boots = 5 * x) ∧
  (shorts + shinguards = 3 * x) ∧
  (shorts + tshirt + boots + shinguards = 8 * x)

/-- Theorem: The total cost of all items is 8 times the cost of shorts -/
theorem total_cost_is_eight_times_shorts (x : ℝ) (h : FootballEquipmentCost x) :
  ∃ (shorts tshirt boots shinguards : ℝ),
    shorts = x ∧
    shorts + tshirt + boots + shinguards = 8 * x :=
by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_eight_times_shorts_l3384_338470


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3384_338421

theorem quadratic_roots_range (m : ℝ) : 
  (∀ x, x^2 + (m-2)*x + (5-m) = 0 → x > 2) →
  m ∈ Set.Ioc (-5) (-4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3384_338421


namespace NUMINAMATH_CALUDE_fabric_sales_fraction_l3384_338498

theorem fabric_sales_fraction (total_sales stationery_sales : ℕ) 
  (h1 : total_sales = 36)
  (h2 : stationery_sales = 15)
  (h3 : ∃ jewelry_sales : ℕ, jewelry_sales = total_sales / 4)
  (h4 : ∃ fabric_sales : ℕ, fabric_sales + total_sales / 4 + stationery_sales = total_sales) :
  ∃ fabric_sales : ℕ, (fabric_sales : ℚ) / total_sales = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fabric_sales_fraction_l3384_338498


namespace NUMINAMATH_CALUDE_jason_initial_cards_l3384_338432

/-- The number of Pokemon cards Jason had initially -/
def initial_cards : ℕ := sorry

/-- The number of Pokemon cards Benny bought from Jason -/
def cards_bought : ℕ := 2

/-- The number of Pokemon cards Jason has left -/
def cards_left : ℕ := 1

/-- Theorem stating that Jason's initial number of Pokemon cards was 3 -/
theorem jason_initial_cards : initial_cards = 3 := by sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l3384_338432


namespace NUMINAMATH_CALUDE_equation_solution_l3384_338488

theorem equation_solution : ∃ x : ℝ, x ≠ 1 ∧ (2 * x + 4) / (x^2 + 4 * x - 5) = (2 - x) / (x - 1) ∧ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3384_338488


namespace NUMINAMATH_CALUDE_diagonal_cubes_140_320_360_l3384_338430

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def diagonal_cubes (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: The internal diagonal of a 140 × 320 × 360 rectangular solid passes through 760 unit cubes -/
theorem diagonal_cubes_140_320_360 :
  diagonal_cubes 140 320 360 = 760 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_cubes_140_320_360_l3384_338430


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3384_338483

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = Real.sqrt 3)
  (θ : ℝ) (h4 : Real.tan θ = Real.sqrt 21 / 2)
  (P Q : ℝ × ℝ) (F2 : ℝ × ℝ) (h5 : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (h6 : Q.1 = 0) (h7 : (Q.2 - F2.2) / (Q.1 - F2.1) = Real.tan θ)
  (h8 : dist P Q / dist P F2 = 1/2) :
  ∃ (k : ℝ), ∀ (x y : ℝ), 3 * x^2 - y^2 = k :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3384_338483


namespace NUMINAMATH_CALUDE_exponential_inequality_solution_set_l3384_338445

theorem exponential_inequality_solution_set :
  {x : ℝ | (4 : ℝ)^(8 - x) > (4 : ℝ)^(-2 * x)} = {x : ℝ | x > -8} := by sorry

end NUMINAMATH_CALUDE_exponential_inequality_solution_set_l3384_338445


namespace NUMINAMATH_CALUDE_students_suggesting_both_l3384_338480

/-- Given the total number of students suggesting bacon and the number of students suggesting only bacon,
    prove that the number of students suggesting both mashed potatoes and bacon
    is equal to the difference between these two values. -/
theorem students_suggesting_both (total_bacon : ℕ) (only_bacon : ℕ)
    (h : total_bacon = 569 ∧ only_bacon = 351) :
    total_bacon - only_bacon = 218 := by
  sorry

end NUMINAMATH_CALUDE_students_suggesting_both_l3384_338480


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l3384_338460

/-- A convex polyhedron inscribed around a sphere -/
structure InscribedPolyhedron where
  -- The radius of the inscribed sphere
  r : ℝ
  -- The surface area of the polyhedron
  surface_area : ℝ
  -- The volume of the polyhedron
  volume : ℝ
  -- Assumption that the polyhedron is inscribed around the sphere
  inscribed : True

/-- 
Theorem: For any convex polyhedron inscribed around a sphere,
the ratio of its volume to its surface area is equal to r/3,
where r is the radius of the inscribed sphere.
-/
theorem volume_to_surface_area_ratio (P : InscribedPolyhedron) :
  P.volume / P.surface_area = P.r / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l3384_338460


namespace NUMINAMATH_CALUDE_min_saplings_needed_l3384_338440

theorem min_saplings_needed (road_length : ℕ) (tree_spacing : ℕ) : road_length = 1000 → tree_spacing = 100 → 
  (road_length / tree_spacing + 1) * 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_min_saplings_needed_l3384_338440


namespace NUMINAMATH_CALUDE_dot_product_OM_ON_l3384_338458

/-- Regular triangle OAB with side length 1 -/
structure RegularTriangle where
  O : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  is_regular : sorry
  side_length : sorry

/-- Points M and N divide AB into three equal parts -/
def divide_side (t : RegularTriangle) : (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

/-- Vector representation -/
def vec (p q : ℝ × ℝ) : ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Main theorem -/
theorem dot_product_OM_ON (t : RegularTriangle) : 
  let (M, N) := divide_side t
  let m := vec t.O M
  let n := vec t.O N
  dot_product m n = 1/6 := by
    sorry

end NUMINAMATH_CALUDE_dot_product_OM_ON_l3384_338458


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3384_338491

/-- The dimensions of a rectangle satisfying specific conditions -/
theorem rectangle_dimensions :
  ∀ x y : ℝ,
  x > 0 ∧ y > 0 →  -- Ensure positive dimensions
  y = 2 * x →      -- Length is twice the width
  2 * (x + y) = 2 * (x * y) →  -- Perimeter is twice the area
  (x, y) = (3/2, 3) := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3384_338491


namespace NUMINAMATH_CALUDE_max_value_of_x_l3384_338436

theorem max_value_of_x (x : ℝ) : 
  (((4 * x - 16) / (3 * x - 4)) ^ 2 + (4 * x - 16) / (3 * x - 4) = 18) →
  x ≤ (3 * Real.sqrt 73 + 28) / (11 - Real.sqrt 73) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_l3384_338436


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l3384_338416

-- Define a regular decagon inscribed in a circle
def RegularDecagon : Type := Unit

-- Define a segment in the decagon
def Segment (d : RegularDecagon) : Type := Unit

-- Define a function to check if three segments form a triangle with positive area
def formsTriangle (d : RegularDecagon) (s1 s2 s3 : Segment d) : Prop := sorry

-- Define a function to calculate the probability
def probabilityOfTriangle (d : RegularDecagon) : ℚ := sorry

-- Theorem statement
theorem decagon_triangle_probability (d : RegularDecagon) : 
  probabilityOfTriangle d = 153 / 190 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l3384_338416


namespace NUMINAMATH_CALUDE_problem_solution_l3384_338407

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - t*x + 1

-- Define the predicate p
def p (t : ℝ) : Prop := ∃ x, f t x = 0

-- Define the predicate q
def q (t : ℝ) : Prop := ∀ x, |x - 1| ≥ 2 - t^2

theorem problem_solution (t : ℝ) :
  (q t → t ∈ Set.Ici (Real.sqrt 2) ∪ Set.Iic (-Real.sqrt 2)) ∧
  (¬p t ∧ ¬q t → t ∈ Set.Ioo (-Real.sqrt 2) (Real.sqrt 2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3384_338407


namespace NUMINAMATH_CALUDE_absolute_value_quadratic_equivalence_l3384_338454

theorem absolute_value_quadratic_equivalence :
  ∀ b c : ℝ,
  (∀ x : ℝ, |x + 4| = 3 ↔ x^2 + b*x + c = 0) →
  b = 8 ∧ c = 7 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_quadratic_equivalence_l3384_338454


namespace NUMINAMATH_CALUDE_curve_and_function_relation_l3384_338408

-- Define a curve C as a set of points in ℝ²
def C : Set (ℝ × ℝ) := sorry

-- Define the function F
def F : ℝ → ℝ → ℝ := sorry

-- Theorem statement
theorem curve_and_function_relation :
  (∀ p : ℝ × ℝ, p ∈ C → F p.1 p.2 = 0) ∧
  (∀ p : ℝ × ℝ, F p.1 p.2 ≠ 0 → p ∉ C) :=
sorry

end NUMINAMATH_CALUDE_curve_and_function_relation_l3384_338408


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l3384_338431

/-- Given two graphs that intersect at (3,4) and (7,2), prove that a+c = 10 -/
theorem intersection_implies_sum (a b c d : ℝ) : 
  (∀ x, -|x - (a + 1)| + b = |x - (c - 1)| + (d - 1) → x = 3 ∨ x = 7) →
  -|3 - (a + 1)| + b = |3 - (c - 1)| + (d - 1) →
  -|7 - (a + 1)| + b = |7 - (c - 1)| + (d - 1) →
  -|3 - (a + 1)| + b = 4 →
  -|7 - (a + 1)| + b = 2 →
  |3 - (c - 1)| + (d - 1) = 4 →
  |7 - (c - 1)| + (d - 1) = 2 →
  a + c = 10 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l3384_338431


namespace NUMINAMATH_CALUDE_tank_capacity_l3384_338474

/-- Represents the flow rate in kiloliters per minute -/
def flow_rate (volume : ℚ) (time : ℚ) : ℚ := volume / time

/-- Calculates the net flow rate into the tank -/
def net_flow_rate (fill_rate drain_rate1 drain_rate2 : ℚ) : ℚ :=
  fill_rate - (drain_rate1 + drain_rate2)

/-- Calculates the amount of water added to the tank -/
def water_added (net_rate : ℚ) (time : ℚ) : ℚ := net_rate * time

/-- Converts kiloliters to liters -/
def kiloliters_to_liters (kl : ℚ) : ℚ := kl * 1000

theorem tank_capacity :
  let fill_rate := flow_rate 1 2
  let drain_rate1 := flow_rate 1 4
  let drain_rate2 := flow_rate 1 6
  let net_rate := net_flow_rate fill_rate drain_rate1 drain_rate2
  let added_water := water_added net_rate 36
  let full_capacity := 2 * added_water
  kiloliters_to_liters full_capacity = 6000 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l3384_338474


namespace NUMINAMATH_CALUDE_f_2019_equals_2_l3384_338403

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2019_equals_2 (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : ∀ x, f x = f (4 - x))
  (h_f_neg3 : f (-3) = 2) :
  f 2019 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2019_equals_2_l3384_338403


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3384_338468

def A : Set ℝ := {x : ℝ | |x| < 2}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3384_338468


namespace NUMINAMATH_CALUDE_cubic_equation_equivalence_l3384_338413

theorem cubic_equation_equivalence (x : ℝ) :
  x^3 + (x + 1)^4 + (x + 2)^3 = (x + 3)^4 ↔ 7 * (x^3 + 6 * x^2 + 13.14 * x + 10.29) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_equivalence_l3384_338413


namespace NUMINAMATH_CALUDE_clothing_pricing_solution_l3384_338438

/-- Represents the pricing strategy for a piece of clothing --/
structure ClothingPricing where
  markedPrice : ℝ
  costPrice : ℝ

/-- Defines the conditions for the clothing pricing problem --/
def validPricing (p : ClothingPricing) : Prop :=
  (0.5 * p.markedPrice + 20 = p.costPrice) ∧ 
  (0.8 * p.markedPrice - 40 = p.costPrice)

/-- Theorem stating the unique solution to the clothing pricing problem --/
theorem clothing_pricing_solution :
  ∃! p : ClothingPricing, validPricing p ∧ p.markedPrice = 200 ∧ p.costPrice = 120 := by
  sorry


end NUMINAMATH_CALUDE_clothing_pricing_solution_l3384_338438


namespace NUMINAMATH_CALUDE_test_questions_missed_l3384_338495

theorem test_questions_missed (T : ℕ) (X Y : ℝ) : 
  T > 0 → 
  0 ≤ X ∧ X ≤ 100 →
  0 ≤ Y ∧ Y ≤ 100 →
  ∃ (M F : ℕ),
    M = 5 * F ∧
    M + F = 216 ∧
    M = T * (1 - X / 100) ∧
    F = T * (1 - Y / 100) →
  M = 180 := by
sorry

end NUMINAMATH_CALUDE_test_questions_missed_l3384_338495


namespace NUMINAMATH_CALUDE_fourth_person_height_l3384_338441

/-- Represents the heights of four people standing in order of increasing height. -/
structure HeightGroup where
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ

/-- The conditions of the height problem. -/
def height_conditions (h : HeightGroup) : Prop :=
  h.second = h.first + 2 ∧
  h.third = h.second + 2 ∧
  h.fourth = h.third + 6 ∧
  (h.first + h.second + h.third + h.fourth) / 4 = 79

/-- The theorem stating that under the given conditions, the fourth person is 85 inches tall. -/
theorem fourth_person_height (h : HeightGroup) :
  height_conditions h → h.fourth = 85 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l3384_338441


namespace NUMINAMATH_CALUDE_sequence_sum_l3384_338453

/-- Given a geometric sequence {a_n} and an arithmetic sequence {b_n}, prove that
    b_3 + b_11 = 6 under the given conditions. -/
theorem sequence_sum (a : ℕ → ℝ) (b : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- geometric sequence with common ratio 2
  a 2 * a 3 * a 4 = 27 / 64 →   -- product condition
  b 7 = a 5 →                   -- relation between sequences
  (∃ d, ∀ n, b (n + 1) = b n + d) →  -- arithmetic sequence
  b 3 + b 11 = 6 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3384_338453


namespace NUMINAMATH_CALUDE_tan_two_pi_fifth_plus_theta_l3384_338499

theorem tan_two_pi_fifth_plus_theta (θ : ℝ) 
  (h : Real.sin ((12 / 5) * Real.pi + θ) + 2 * Real.sin ((11 / 10) * Real.pi - θ) = 0) : 
  Real.tan ((2 / 5) * Real.pi + θ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_pi_fifth_plus_theta_l3384_338499


namespace NUMINAMATH_CALUDE_green_bows_count_l3384_338402

theorem green_bows_count (total : ℕ) (white : ℕ) : 
  (1 / 5 : ℚ) * total + (1 / 2 : ℚ) * total + (1 / 10 : ℚ) * total + white = total →
  white = 30 →
  (1 / 10 : ℚ) * total = 15 := by
sorry

end NUMINAMATH_CALUDE_green_bows_count_l3384_338402


namespace NUMINAMATH_CALUDE_other_sales_is_fifteen_percent_l3384_338485

/-- The percentage of sales not attributed to books, magazines, or stationery -/
def other_sales_percentage (books magazines stationery : ℝ) : ℝ :=
  100 - (books + magazines + stationery)

/-- Theorem stating that the percentage of other sales is 15% -/
theorem other_sales_is_fifteen_percent :
  other_sales_percentage 45 30 10 = 15 := by
  sorry

#eval other_sales_percentage 45 30 10

end NUMINAMATH_CALUDE_other_sales_is_fifteen_percent_l3384_338485


namespace NUMINAMATH_CALUDE_homologous_functions_count_l3384_338418

def f (x : ℝ) : ℝ := x^2

def isValidDomain (D : Set ℝ) : Prop :=
  (∀ x ∈ D, f x ∈ ({0, 1} : Set ℝ)) ∧
  (∀ y ∈ ({0, 1} : Set ℝ), ∃ x ∈ D, f x = y)

theorem homologous_functions_count :
  ∃! (domains : Finset (Set ℝ)), domains.card = 3 ∧
    ∀ D ∈ domains, isValidDomain D :=
sorry

end NUMINAMATH_CALUDE_homologous_functions_count_l3384_338418


namespace NUMINAMATH_CALUDE_max_savings_l3384_338410

structure Flight where
  airline : String
  basePrice : ℕ
  discountPercentage : ℕ
  layovers : ℕ
  travelTime : ℕ

def calculateDiscountedPrice (flight : Flight) : ℚ :=
  flight.basePrice - (flight.basePrice * flight.discountPercentage / 100)

def flightOptions : List Flight := [
  ⟨"Delta Airlines", 850, 20, 1, 6⟩,
  ⟨"United Airlines", 1100, 30, 1, 7⟩,
  ⟨"American Airlines", 950, 25, 2, 9⟩,
  ⟨"Southwest Airlines", 900, 15, 1, 5⟩,
  ⟨"JetBlue Airways", 1200, 40, 0, 4⟩
]

theorem max_savings (options : List Flight := flightOptions) :
  let discountedPrices := options.map calculateDiscountedPrice
  let minPrice := discountedPrices.minimum?
  let maxPrice := discountedPrices.maximum?
  ∀ min max, minPrice = some min → maxPrice = some max →
    max - min = 90 :=
by sorry

end NUMINAMATH_CALUDE_max_savings_l3384_338410


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3384_338442

theorem geometric_sequence_sum (a₀ r : ℚ) (n : ℕ) (h₁ : a₀ = 1/3) (h₂ : r = 1/3) (h₃ : n = 10) :
  let S := a₀ * (1 - r^n) / (1 - r)
  S = 29524/59049 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3384_338442


namespace NUMINAMATH_CALUDE_polynomial_coefficient_equality_l3384_338484

theorem polynomial_coefficient_equality (k d m : ℚ) : 
  (∀ x : ℚ, (6 * x^3 - 4 * x^2 + 9/4) * (d * x^3 + k * x^2 + m) = 
   18 * x^6 - 17 * x^5 + 34 * x^4 - (36/4) * x^3 + (18/4) * x^2) → 
  k = -5/6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_equality_l3384_338484


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3384_338455

-- Define propositions p and q
def p (a b : ℝ) : Prop := a > 0 ∧ 0 > b

def q (a b : ℝ) : Prop := |a + b| < |a| + |b|

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ a b : ℝ, p a b → q a b) ∧
  ¬(∀ a b : ℝ, q a b → p a b) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3384_338455


namespace NUMINAMATH_CALUDE_max_remainder_theorem_l3384_338439

theorem max_remainder_theorem :
  (∀ n : ℕ, n < 120 → ∃ k : ℕ, 209 = k * n + 104 ∧ ∀ m : ℕ, m < n → 209 % m ≤ 104) ∧
  (∀ n : ℕ, n < 90 → ∃ k : ℕ, 209 = k * n + 69 ∧ ∀ m : ℕ, m < n → 209 % m ≤ 69) :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_theorem_l3384_338439


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3384_338479

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (-1, m)
  parallel a b → m = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3384_338479


namespace NUMINAMATH_CALUDE_probability_one_of_each_color_l3384_338487

def total_marbles : ℕ := 12
def marbles_per_color : ℕ := 3
def colors : ℕ := 4
def selected_marbles : ℕ := 4

/-- The probability of selecting one marble of each color when randomly selecting 4 marbles
    without replacement from a bag containing 3 red, 3 blue, 3 green, and 3 yellow marbles. -/
theorem probability_one_of_each_color : 
  (marbles_per_color ^ colors : ℚ) / (total_marbles.choose selected_marbles) = 9 / 55 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_each_color_l3384_338487


namespace NUMINAMATH_CALUDE_dividend_percentage_calculation_l3384_338429

/-- Calculates the dividend percentage given investment details and dividend received -/
theorem dividend_percentage_calculation
  (investment : ℝ)
  (share_face_value : ℝ)
  (premium_percentage : ℝ)
  (dividend_received : ℝ)
  (h1 : investment = 14400)
  (h2 : share_face_value = 100)
  (h3 : premium_percentage = 20)
  (h4 : dividend_received = 840.0000000000001) :
  let share_cost := share_face_value * (1 + premium_percentage / 100)
  let num_shares := investment / share_cost
  let dividend_per_share := dividend_received / num_shares
  let dividend_percentage := (dividend_per_share / share_face_value) * 100
  dividend_percentage = 7 := by
sorry

end NUMINAMATH_CALUDE_dividend_percentage_calculation_l3384_338429


namespace NUMINAMATH_CALUDE_candidate_a_democratic_votes_l3384_338423

theorem candidate_a_democratic_votes 
  (total_voters : ℝ) 
  (dem_percent : ℝ) 
  (rep_percent : ℝ) 
  (rep_for_a_percent : ℝ) 
  (total_for_a_percent : ℝ) 
  (h1 : dem_percent = 0.60)
  (h2 : rep_percent = 1 - dem_percent)
  (h3 : rep_for_a_percent = 0.20)
  (h4 : total_for_a_percent = 0.47) :
  let dem_for_a_percent := (total_for_a_percent * total_voters - rep_for_a_percent * rep_percent * total_voters) / (dem_percent * total_voters)
  dem_for_a_percent = 0.65 := by
sorry

end NUMINAMATH_CALUDE_candidate_a_democratic_votes_l3384_338423


namespace NUMINAMATH_CALUDE_normal_trip_time_l3384_338452

theorem normal_trip_time 
  (normal_distance : ℝ) 
  (additional_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : normal_distance = 150) 
  (h2 : additional_distance = 100) 
  (h3 : total_time = 5) :
  (normal_distance / ((normal_distance + additional_distance) / total_time)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_normal_trip_time_l3384_338452


namespace NUMINAMATH_CALUDE_min_value_of_f_l3384_338447

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 - 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

-- Theorem statement
theorem min_value_of_f (m : ℝ) (h1 : -1 ≤ m) (h2 : m ≤ 1) :
  f m ≥ -4 ∧ ∃ m₀, -1 ≤ m₀ ∧ m₀ ≤ 1 ∧ f m₀ = -4 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3384_338447


namespace NUMINAMATH_CALUDE_product_of_roots_l3384_338433

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 18 → ∃ y : ℝ, (x + 3) * (x - 4) = 18 ∧ (y + 3) * (y - 4) = 18 ∧ x * y = -30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3384_338433


namespace NUMINAMATH_CALUDE_complement_of_union_equals_four_l3384_338426

def U : Set Nat := {1, 2, 3, 4}
def P : Set Nat := {1, 2}
def Q : Set Nat := {2, 3}

theorem complement_of_union_equals_four : 
  (U \ (P ∪ Q)) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_four_l3384_338426


namespace NUMINAMATH_CALUDE_parallel_vectors_condition_l3384_338494

/-- Two vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The statement that m = 1 is a necessary but not sufficient condition for
    vectors (m, 1) and (1, m) to be parallel -/
theorem parallel_vectors_condition :
  ∃ m : ℝ, (m = 1 → are_parallel (m, 1) (1, m)) ∧
           ¬(are_parallel (m, 1) (1, m) → m = 1) :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_condition_l3384_338494


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3384_338443

theorem imaginary_part_of_complex_number (b : ℝ) :
  let z : ℂ := 2 + b * Complex.I
  (Complex.abs z = 2 * Real.sqrt 2) → (b = 2 ∨ b = -2) :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3384_338443


namespace NUMINAMATH_CALUDE_equation_solution_l3384_338463

theorem equation_solution :
  ∀ x : ℝ, (1 / 7 : ℝ) + 7 / x = 15 / x + (1 / 15 : ℝ) → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3384_338463


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3384_338478

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of five consecutive terms starting from the third term is 250 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 250

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : 
  a 2 + a 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3384_338478


namespace NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l3384_338486

theorem coefficient_x4_in_expansion : 
  (Finset.range 9).sum (fun k => Nat.choose 8 k * 3^(8 - k) * if k = 4 then 1 else 0) = 5670 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l3384_338486


namespace NUMINAMATH_CALUDE_restaurant_check_amount_l3384_338400

theorem restaurant_check_amount
  (tax_rate : Real)
  (total_payment : Real)
  (tip_amount : Real)
  (h1 : tax_rate = 0.20)
  (h2 : total_payment = 20)
  (h3 : tip_amount = 2) :
  ∃ (original_amount : Real),
    original_amount * (1 + tax_rate) = total_payment - tip_amount ∧
    original_amount = 15 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_check_amount_l3384_338400


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3384_338477

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 16 feet is 16√13 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 16 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3384_338477


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3384_338492

/-- Proves that the cost price is 17500 given the selling price, discount rate, and profit rate --/
theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  selling_price = 21000 →
  discount_rate = 0.1 →
  profit_rate = 0.08 →
  (selling_price * (1 - discount_rate)) / (1 + profit_rate) = 17500 := by
  sorry

#check cost_price_calculation

end NUMINAMATH_CALUDE_cost_price_calculation_l3384_338492
