import Mathlib

namespace NUMINAMATH_CALUDE_max_points_in_plane_max_points_in_space_l1599_159932

/-- A point in a Euclidean space -/
structure Point (n : Nat) where
  coords : Fin n → ℝ

/-- Checks if three points form an obtuse angle -/
def is_obtuse_angle (n : Nat) (p1 p2 p3 : Point n) : Prop :=
  sorry -- Definition of obtuse angle check

/-- A configuration of points in a Euclidean space -/
structure PointConfiguration (n : Nat) where
  dim : Nat -- dimension of the space (2 for plane, 3 for space)
  points : Fin n → Point dim
  no_obtuse_angles : ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    ¬ is_obtuse_angle dim (points i) (points j) (points k)

/-- The maximum number of points in a plane configuration without obtuse angles -/
theorem max_points_in_plane :
  (∃ (c : PointConfiguration 4), c.dim = 2) ∧
  (∀ (n : Nat), n > 4 → ¬ ∃ (c : PointConfiguration n), c.dim = 2) :=
sorry

/-- The maximum number of points in a space configuration without obtuse angles -/
theorem max_points_in_space :
  (∃ (c : PointConfiguration 8), c.dim = 3) ∧
  (∀ (n : Nat), n > 8 → ¬ ∃ (c : PointConfiguration n), c.dim = 3) :=
sorry

end NUMINAMATH_CALUDE_max_points_in_plane_max_points_in_space_l1599_159932


namespace NUMINAMATH_CALUDE_arccos_one_equals_zero_l1599_159946

theorem arccos_one_equals_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_equals_zero_l1599_159946


namespace NUMINAMATH_CALUDE_smallest_n_for_subset_sequence_l1599_159944

theorem smallest_n_for_subset_sequence (X : Finset ℕ) (h : X.card = 100) :
  let n := 2 * Nat.choose 100 50 + 2 * Nat.choose 100 49 + 1
  ∀ (A : Fin n → Finset ℕ), (∀ i, A i ⊆ X) →
    (∃ i j k, i < j ∧ j < k ∧ (A i ⊆ A j ∧ A j ⊆ A k ∨ A i ⊇ A j ∧ A j ⊇ A k)) ∧
  ∀ m < n, ∃ (B : Fin m → Finset ℕ), (∀ i, B i ⊆ X) ∧
    ¬(∃ i j k, i < j ∧ j < k ∧ (B i ⊆ B j ∧ B j ⊆ B k ∨ B i ⊇ B j ∧ B j ⊇ B k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_subset_sequence_l1599_159944


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l1599_159960

theorem sum_of_roots_cubic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 3*x^2 + 2*x
  (∃ a b c : ℝ, f x = (x - a) * (x - b) * (x - c)) → 
  (a + b + c = 3) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l1599_159960


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l1599_159919

theorem student_multiplication_problem (x : ℝ) : 40 * x - 150 = 130 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l1599_159919


namespace NUMINAMATH_CALUDE_max_roses_is_316_l1599_159902

/-- Represents the pricing options and budget for purchasing roses -/
structure RosePurchase where
  single_price : ℚ  -- Price of a single rose
  dozen_price : ℚ   -- Price of a dozen roses
  two_dozen_price : ℚ -- Price of two dozen roses
  budget : ℚ        -- Total budget

/-- Calculates the maximum number of roses that can be purchased given the pricing options and budget -/
def max_roses (rp : RosePurchase) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of roses that can be purchased is 316 -/
theorem max_roses_is_316 (rp : RosePurchase) 
  (h1 : rp.single_price = 63/10)
  (h2 : rp.dozen_price = 36)
  (h3 : rp.two_dozen_price = 50)
  (h4 : rp.budget = 680) :
  max_roses rp = 316 :=
by sorry

end NUMINAMATH_CALUDE_max_roses_is_316_l1599_159902


namespace NUMINAMATH_CALUDE_evaluate_expression_l1599_159909

theorem evaluate_expression : 3000^3 - 2998*3000^2 - 2998^2*3000 + 2998^3 = 23992 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1599_159909


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_triangle_height_equation_l1599_159990

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem 1
theorem perpendicular_line_equation (A : Point) (h : A.x = -2 ∧ A.y = 3) :
  ∃ l : Line, perpendicular l ⟨A.y, -A.x, 0⟩ ∧ on_line A l ∧ l.a = 2 ∧ l.b = -3 ∧ l.c = 13 :=
sorry

-- Theorem 2
theorem triangle_height_equation (A B C : Point) 
  (hA : A.x = 4 ∧ A.y = 0) (hB : B.x = 6 ∧ B.y = 7) (hC : C.x = 0 ∧ C.y = 3) :
  ∃ l : Line, perpendicular l ⟨B.y - A.y, A.x - B.x, B.x * A.y - A.x * B.y⟩ ∧ 
    on_line C l ∧ l.a = 2 ∧ l.b = 7 ∧ l.c = -21 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_triangle_height_equation_l1599_159990


namespace NUMINAMATH_CALUDE_parallel_vectors_subtraction_l1599_159975

/-- Given two parallel vectors a and b, prove that 2a - b equals (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (m, 4)
  (a.1 * b.2 = a.2 * b.1) →  -- Condition for parallel vectors
  (2 * a.1 - b.1, 2 * a.2 - b.2) = (4, -8) := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_subtraction_l1599_159975


namespace NUMINAMATH_CALUDE_trivia_game_total_score_luke_total_score_l1599_159961

/-- 
Given a player in a trivia game who:
- Plays a certain number of rounds
- Scores the same number of points each round
- Scores a specific number of points per round

This theorem proves that the total points scored is equal to 
the product of the number of rounds and the points per round.
-/
theorem trivia_game_total_score 
  (rounds : ℕ) 
  (points_per_round : ℕ) : 
  rounds * points_per_round = rounds * points_per_round := by
  sorry

/-- 
This theorem applies the general trivia_game_total_score theorem 
to Luke's specific case, where he played 5 rounds and scored 60 points per round.
-/
theorem luke_total_score : 5 * 60 = 300 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_total_score_luke_total_score_l1599_159961


namespace NUMINAMATH_CALUDE_exists_x_less_than_x_cubed_less_than_x_squared_l1599_159921

theorem exists_x_less_than_x_cubed_less_than_x_squared :
  ∃ x : ℝ, x < x^3 ∧ x^3 < x^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_less_than_x_cubed_less_than_x_squared_l1599_159921


namespace NUMINAMATH_CALUDE_kindergarten_tissue_problem_l1599_159907

theorem kindergarten_tissue_problem :
  ∀ (group1 : ℕ), 
    (group1 * 40 + 10 * 40 + 11 * 40 = 1200) → 
    group1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_tissue_problem_l1599_159907


namespace NUMINAMATH_CALUDE_no_real_solutions_l1599_159988

theorem no_real_solutions :
  ∀ x y : ℝ, x^2 + 3*y^2 - 4*x - 6*y + 10 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1599_159988


namespace NUMINAMATH_CALUDE_at_least_eight_empty_columns_at_least_eight_people_in_one_column_l1599_159913

/-- Represents the state of people on columns -/
structure ColumnState where
  num_people : Nat
  num_columns : Nat
  initial_column : Nat

/-- Proves that at least 8 columns are empty after any number of steps -/
theorem at_least_eight_empty_columns (state : ColumnState) 
  (h1 : state.num_people = 65)
  (h2 : state.num_columns = 17)
  (h3 : state.initial_column = 9) :
  ∀ (steps : Nat), ∃ (empty_columns : Nat), empty_columns ≥ 8 := by
  sorry

/-- Proves that there is always at least one column with at least 8 people -/
theorem at_least_eight_people_in_one_column (state : ColumnState) 
  (h1 : state.num_people = 65)
  (h2 : state.num_columns = 17)
  (h3 : state.initial_column = 9) :
  ∀ (steps : Nat), ∃ (column : Nat), ∃ (people_in_column : Nat), 
    people_in_column ≥ 8 ∧ column ≤ state.num_columns := by
  sorry

end NUMINAMATH_CALUDE_at_least_eight_empty_columns_at_least_eight_people_in_one_column_l1599_159913


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l1599_159998

/-- The area of a right triangle with side lengths in the ratio 5:12:13, 
    inscribed in a circle of radius 5, is equal to 3000/169. -/
theorem triangle_area_in_circle (r : ℝ) (h : r = 5) : 
  let s := r * 2 / 13  -- Scale factor
  let a := 5 * s       -- First side
  let b := 12 * s      -- Second side
  let c := 13 * s      -- Third side (hypotenuse)
  (a^2 + b^2 = c^2) ∧  -- Pythagorean theorem
  (c = 2 * r) →        -- Diameter equals hypotenuse
  (1/2 * a * b = 3000/169) := by
sorry

end NUMINAMATH_CALUDE_triangle_area_in_circle_l1599_159998


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l1599_159925

/-- The number of ways to arrange books on a shelf with specific conditions. -/
def arrange_books (num_math_books num_english_books : ℕ) : ℕ :=
  Nat.factorial (num_english_books - 1) * Nat.factorial num_math_books

/-- Theorem stating the number of ways to arrange 4 math books and 6 English books
    with specific conditions. -/
theorem book_arrangement_theorem :
  arrange_books 4 6 = 2880 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l1599_159925


namespace NUMINAMATH_CALUDE_average_age_increase_l1599_159972

theorem average_age_increase (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 15 →
  student_avg_age = 10 →
  teacher_age = 26 →
  (((num_students : ℝ) * student_avg_age + teacher_age) / ((num_students : ℝ) + 1)) - student_avg_age = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l1599_159972


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_geometric_sequence_problem_l1599_159958

-- Arithmetic Sequence
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ := (n : ℚ) * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem arithmetic_sequence_problem (a₁ d Sn : ℚ) (n : ℕ) :
  a₁ = 3/2 ∧ d = -1/2 ∧ Sn = -15 →
  n = 12 ∧ arithmetic_sequence a₁ d n = -4 :=
sorry

-- Geometric Sequence
def geometric_sequence (a₁ q : ℚ) (n : ℕ) : ℚ := a₁ * q ^ (n - 1)

def geometric_sum (a₁ q : ℚ) (n : ℕ) : ℚ := a₁ * (q ^ n - 1) / (q - 1)

theorem geometric_sequence_problem (a₁ q Sn : ℚ) (n : ℕ) :
  q = 2 ∧ geometric_sequence a₁ q n = 96 ∧ Sn = 189 →
  a₁ = 3 ∧ n = 6 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_geometric_sequence_problem_l1599_159958


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1599_159994

/-- Given a quadratic equation x^2 - x + 1 - m = 0 with two real roots α and β 
    satisfying |α| + |β| ≤ 5, the range of m is [3/4, 7]. -/
theorem quadratic_roots_range (m : ℝ) (α β : ℝ) : 
  (∀ x, x^2 - x + 1 - m = 0 ↔ x = α ∨ x = β) →
  (|α| + |β| ≤ 5) →
  (3/4 ≤ m ∧ m ≤ 7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1599_159994


namespace NUMINAMATH_CALUDE_min_value_sum_l1599_159942

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / x + 1 / y = 1) :
  x + y ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l1599_159942


namespace NUMINAMATH_CALUDE_exists_continuous_random_variable_point_P_in_plane_ABC_l1599_159959

-- Define a random variable type
def RandomVariable := ℝ → ℝ

-- Define vectors in ℝ³
def AB : Fin 3 → ℝ := ![2, -1, -4]
def AC : Fin 3 → ℝ := ![4, 2, 0]
def AP : Fin 3 → ℝ := ![0, -4, -8]

-- Theorem for the existence of a continuous random variable
theorem exists_continuous_random_variable :
  ∃ (X : RandomVariable), ∀ (a b : ℝ), a < b → ∃ (x : ℝ), a < X x ∧ X x < b :=
sorry

-- Function to check if a point is in a plane defined by three vectors
def is_in_plane (p v1 v2 : Fin 3 → ℝ) : Prop :=
  ∃ (a b : ℝ), p = λ i => a * v1 i + b * v2 i

-- Theorem for point P lying in plane ABC
theorem point_P_in_plane_ABC :
  is_in_plane AP AB AC :=
sorry

end NUMINAMATH_CALUDE_exists_continuous_random_variable_point_P_in_plane_ABC_l1599_159959


namespace NUMINAMATH_CALUDE_total_consumption_30_days_l1599_159943

/-- Represents the daily food consumption for each dog -/
structure DogConsumption where
  a : Float
  b : Float
  c : Float
  d : Float
  e : Float

/-- Calculates the total daily consumption for all dogs -/
def totalDailyConsumption (dc : DogConsumption) : Float :=
  dc.a + dc.b + dc.c + dc.d + dc.e

/-- Represents the food consumption for each dog on Sundays -/
structure SundayConsumption where
  a : Float
  b : Float
  c : Float
  d : Float
  e : Float

/-- Calculates the total consumption for all dogs on a Sunday -/
def totalSundayConsumption (sc : SundayConsumption) : Float :=
  sc.a + sc.b + sc.c + sc.d + sc.e

/-- Theorem: Total dog food consumption over 30 days is 60 scoops -/
theorem total_consumption_30_days 
  (dc : DogConsumption)
  (sc : SundayConsumption)
  (h1 : dc.a = 0.125)
  (h2 : dc.b = 0.25)
  (h3 : dc.c = 0.375)
  (h4 : dc.d = 0.5)
  (h5 : dc.e = 0.75)
  (h6 : sc.a = dc.a)
  (h7 : sc.b = dc.b)
  (h8 : sc.c = dc.c + 0.1)
  (h9 : sc.d = dc.d)
  (h10 : sc.e = dc.e - 0.1)
  (h11 : totalDailyConsumption dc = totalSundayConsumption sc) :
  30 * totalDailyConsumption dc = 60 := by
  sorry


end NUMINAMATH_CALUDE_total_consumption_30_days_l1599_159943


namespace NUMINAMATH_CALUDE_coins_problem_l1599_159939

theorem coins_problem (x : ℚ) : 
  let lost := (2 : ℚ) / 3 * x
  let found := (4 : ℚ) / 5 * lost
  let remaining := x - lost + found
  x - remaining = (2 : ℚ) / 15 * x :=
by sorry

end NUMINAMATH_CALUDE_coins_problem_l1599_159939


namespace NUMINAMATH_CALUDE_weight_meets_standard_l1599_159912

/-- The nominal weight of the strawberry box in kilograms -/
def nominal_weight : ℝ := 5

/-- The allowed deviation from the nominal weight in kilograms -/
def allowed_deviation : ℝ := 0.03

/-- The actual weight of the strawberry box in kilograms -/
def actual_weight : ℝ := 4.98

/-- Theorem stating that the actual weight meets the standard -/
theorem weight_meets_standard : 
  nominal_weight - allowed_deviation ≤ actual_weight ∧ 
  actual_weight ≤ nominal_weight + allowed_deviation := by
  sorry

end NUMINAMATH_CALUDE_weight_meets_standard_l1599_159912


namespace NUMINAMATH_CALUDE_log_xyz_equals_one_l1599_159945

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- State the theorem
theorem log_xyz_equals_one 
  (x y z : ℝ) 
  (h1 : log (x^2 * y^2 * z) = 2) 
  (h2 : log (x * y * z^3) = 2) : 
  log (x * y * z) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_xyz_equals_one_l1599_159945


namespace NUMINAMATH_CALUDE_prime_q_value_l1599_159970

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_q_value (p : ℕ) (hp : is_prime p) :
  let q := 13 * p + 2
  is_prime q ∧ (q - 1) % 3 = 0 → q = 67 := by
  sorry

end NUMINAMATH_CALUDE_prime_q_value_l1599_159970


namespace NUMINAMATH_CALUDE_diamond_calculation_l1599_159983

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation : 
  let x := diamond (diamond 2 3) 4
  let y := diamond 2 (diamond 3 4)
  x - y = -29 / 132 := by sorry

end NUMINAMATH_CALUDE_diamond_calculation_l1599_159983


namespace NUMINAMATH_CALUDE_inequality_proof_l1599_159911

theorem inequality_proof (x a : ℝ) (h : x < a ∧ a < 0) : x^3 > a*x ∧ a*x < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1599_159911


namespace NUMINAMATH_CALUDE_barney_average_speed_l1599_159993

def initial_reading : ℕ := 2332
def final_reading : ℕ := 2772
def total_time : ℕ := 12

def distance : ℕ := final_reading - initial_reading

def average_speed : ℚ := distance / total_time

theorem barney_average_speed : 
  initial_reading = 2332 → 
  final_reading = 2772 → 
  total_time = 12 → 
  ⌊average_speed⌋ = 36 := by sorry

end NUMINAMATH_CALUDE_barney_average_speed_l1599_159993


namespace NUMINAMATH_CALUDE_sequence_sum_l1599_159965

theorem sequence_sum (a b c d : ℕ) (h1 : 0 < a ∧ a < b ∧ b < c ∧ c < d)
  (h2 : b - a = c - b) (h3 : c * c = b * d) (h4 : d - a = 30) :
  a + b + c + d = 129 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l1599_159965


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1599_159978

theorem complex_fraction_evaluation :
  (2 : ℂ) / (Complex.I * (3 - Complex.I)) = (1 / 5 : ℂ) - (3 / 5 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1599_159978


namespace NUMINAMATH_CALUDE_sin_18_deg_identity_l1599_159967

theorem sin_18_deg_identity :
  let x : ℝ := Real.sin (18 * π / 180)
  4 * x^2 + 2 * x = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_deg_identity_l1599_159967


namespace NUMINAMATH_CALUDE_parallelogram_smaller_angle_l1599_159992

theorem parallelogram_smaller_angle (smaller_angle larger_angle : ℝ) : 
  larger_angle = smaller_angle + 90 →
  smaller_angle + larger_angle = 180 →
  smaller_angle = 45 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_smaller_angle_l1599_159992


namespace NUMINAMATH_CALUDE_number_sequence_problem_l1599_159916

theorem number_sequence_problem :
  ∃ k : ℕ+,
    let a : ℕ+ → ℤ := λ n => (-2) ^ n.val
    let b : ℕ+ → ℤ := λ n => a n + 2
    let c : ℕ+ → ℚ := λ n => (1 / 2 : ℚ) * (a n)
    (a k + b k + c k = 642) ∧ (a k = 256) :=
by
  sorry

end NUMINAMATH_CALUDE_number_sequence_problem_l1599_159916


namespace NUMINAMATH_CALUDE_compound_mass_percentage_sum_l1599_159938

/-- Given a compound with two parts, where one part's mass percentage is known,
    prove that the sum of both parts' mass percentages is 100%. -/
theorem compound_mass_percentage_sum (part1_percentage : ℝ) :
  part1_percentage = 80.12 →
  100 - part1_percentage = 19.88 := by
  sorry

end NUMINAMATH_CALUDE_compound_mass_percentage_sum_l1599_159938


namespace NUMINAMATH_CALUDE_fabric_theorem_l1599_159935

def fabric_problem (checkered_cost plain_cost yard_cost : ℚ) : Prop :=
  let checkered_yards := checkered_cost / yard_cost
  let plain_yards := plain_cost / yard_cost
  let total_yards := checkered_yards + plain_yards
  total_yards = 16

theorem fabric_theorem :
  fabric_problem 75 45 7.5 := by
  sorry

end NUMINAMATH_CALUDE_fabric_theorem_l1599_159935


namespace NUMINAMATH_CALUDE_sad_girls_l1599_159997

/-- Given information about children's emotions and genders -/
structure ChildrenInfo where
  total : ℕ
  happy : ℕ
  sad : ℕ
  neither : ℕ
  boys : ℕ
  girls : ℕ
  happyBoys : ℕ
  neitherBoys : ℕ

/-- Theorem stating the number of sad girls -/
theorem sad_girls (info : ChildrenInfo)
  (h1 : info.total = 60)
  (h2 : info.happy = 30)
  (h3 : info.sad = 10)
  (h4 : info.neither = 20)
  (h5 : info.boys = 17)
  (h6 : info.girls = 43)
  (h7 : info.happyBoys = 6)
  (h8 : info.neitherBoys = 5)
  (h9 : info.total = info.happy + info.sad + info.neither)
  (h10 : info.total = info.boys + info.girls)
  : info.sad - (info.boys - info.happyBoys - info.neitherBoys) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sad_girls_l1599_159997


namespace NUMINAMATH_CALUDE_solution_set_characterization_l1599_159950

/-- A differentiable function satisfying certain conditions -/
class SpecialFunction (f : ℝ → ℝ) : Prop where
  differentiable : Differentiable ℝ f
  domain : ∀ x, x < 0 → f x ≠ 0
  condition : ∀ x, x < 0 → 3 * f x + x * deriv f x < 0

/-- The solution set of the inequality -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | (x + 2016)^3 * f (x + 2016) + 8 * f (-2) < 0}

theorem solution_set_characterization (f : ℝ → ℝ) [SpecialFunction f] :
  SolutionSet f = Set.Ioo (-2018) (-2016) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l1599_159950


namespace NUMINAMATH_CALUDE_tortoise_age_problem_l1599_159929

theorem tortoise_age_problem (tailor_age tortoise_age tree_age : ℕ) : 
  tailor_age + tortoise_age + tree_age = 264 →
  tailor_age = 4 * (tailor_age - tortoise_age) →
  tortoise_age = 7 * (tortoise_age - tree_age) →
  tortoise_age = 77 := by
sorry

end NUMINAMATH_CALUDE_tortoise_age_problem_l1599_159929


namespace NUMINAMATH_CALUDE_sequence_equivalence_l1599_159963

theorem sequence_equivalence (n : ℕ+) : (2*n - 1)^2 - 1 = 4*n*(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_equivalence_l1599_159963


namespace NUMINAMATH_CALUDE_set_operations_l1599_159915

def A : Set ℤ := {1,2,3,4,5}
def B : Set ℤ := {-1,1,2,3}
def U : Set ℤ := {x | -1 ≤ x ∧ x < 6}

theorem set_operations :
  (A ∩ B = {1,2,3}) ∧
  (A ∪ B = {-1,1,2,3,4,5}) ∧
  ((U \ B) ∩ A = {4,5}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l1599_159915


namespace NUMINAMATH_CALUDE_some_number_value_l1599_159964

theorem some_number_value (some_number : ℝ) : 
  (3.242 * some_number) / 100 = 0.032420000000000004 → some_number = 1 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1599_159964


namespace NUMINAMATH_CALUDE_remainder_problem_l1599_159927

theorem remainder_problem (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 19) → N % 13 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1599_159927


namespace NUMINAMATH_CALUDE_remainder_102938475610_div_12_l1599_159951

theorem remainder_102938475610_div_12 : 102938475610 % 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_102938475610_div_12_l1599_159951


namespace NUMINAMATH_CALUDE_smallest_fraction_l1599_159977

theorem smallest_fraction (x : ℝ) (hx : x = 9) : 
  min ((x - 3) / 8) (min (8 / x) (min (8 / (x + 2)) (min (8 / (x - 2)) ((x + 3) / 8)))) = (x - 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l1599_159977


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1599_159936

/-- A rhombus with side length 37 units and shorter diagonal 40 units has a longer diagonal of 62 units. -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side = 37 → shorter_diagonal = 40 → longer_diagonal = 62 → 
  side^2 = (shorter_diagonal / 2)^2 + (longer_diagonal / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1599_159936


namespace NUMINAMATH_CALUDE_optimal_meal_plan_l1599_159914

/-- Represents the nutritional content of a meal -/
structure Nutrition :=
  (carbs : ℕ)
  (protein : ℕ)
  (vitaminC : ℕ)

/-- Represents the meal plan -/
structure MealPlan :=
  (lunch : ℕ)
  (dinner : ℕ)

def lunch_nutrition : Nutrition := ⟨12, 6, 6⟩
def dinner_nutrition : Nutrition := ⟨8, 6, 10⟩

def lunch_cost : ℚ := 2.5
def dinner_cost : ℚ := 4

def minimum_nutrition : Nutrition := ⟨64, 42, 54⟩

def total_nutrition (plan : MealPlan) : Nutrition :=
  ⟨plan.lunch * lunch_nutrition.carbs + plan.dinner * dinner_nutrition.carbs,
   plan.lunch * lunch_nutrition.protein + plan.dinner * dinner_nutrition.protein,
   plan.lunch * lunch_nutrition.vitaminC + plan.dinner * dinner_nutrition.vitaminC⟩

def meets_requirements (plan : MealPlan) : Prop :=
  let total := total_nutrition plan
  total.carbs ≥ minimum_nutrition.carbs ∧
  total.protein ≥ minimum_nutrition.protein ∧
  total.vitaminC ≥ minimum_nutrition.vitaminC

def total_cost (plan : MealPlan) : ℚ :=
  plan.lunch * lunch_cost + plan.dinner * dinner_cost

theorem optimal_meal_plan :
  ∃ (plan : MealPlan),
    meets_requirements plan ∧
    (∀ (other : MealPlan), meets_requirements other → total_cost plan ≤ total_cost other) ∧
    plan.lunch = 4 ∧ plan.dinner = 3 :=
sorry

end NUMINAMATH_CALUDE_optimal_meal_plan_l1599_159914


namespace NUMINAMATH_CALUDE_sum_of_integers_l1599_159985

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 4)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1599_159985


namespace NUMINAMATH_CALUDE_tan_315_degrees_l1599_159981

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l1599_159981


namespace NUMINAMATH_CALUDE_inequality_of_distinct_reals_l1599_159954

theorem inequality_of_distinct_reals (a b c : ℝ) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  |a / (b - c)| + |b / (c - a)| + |c / (a - b)| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_distinct_reals_l1599_159954


namespace NUMINAMATH_CALUDE_discount_order_difference_l1599_159956

-- Define the original price and discounts
def original_price : ℝ := 30
def flat_discount : ℝ := 5
def percentage_discount : ℝ := 0.25

-- Define the two discount application orders
def discount_flat_then_percent : ℝ := (original_price - flat_discount) * (1 - percentage_discount)
def discount_percent_then_flat : ℝ := (original_price * (1 - percentage_discount)) - flat_discount

-- Theorem statement
theorem discount_order_difference :
  discount_flat_then_percent - discount_percent_then_flat = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_discount_order_difference_l1599_159956


namespace NUMINAMATH_CALUDE_fraction_value_l1599_159941

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 3 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 4 * d) : 
  a * c / (b * d) = 12 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l1599_159941


namespace NUMINAMATH_CALUDE_rectangle_to_hexagon_area_l1599_159996

/-- Given a rectangle with sides of length a and 36, prove that when transformed into a hexagon
    with parallel sides of length a separated by 24, and the hexagon has the same area as the
    original rectangle, then a² = 720. -/
theorem rectangle_to_hexagon_area (a : ℝ) : 
  (0 < a) →
  (24 * a + 30 * Real.sqrt (a^2 - 36) = 36 * a) →
  a^2 = 720 := by
sorry

end NUMINAMATH_CALUDE_rectangle_to_hexagon_area_l1599_159996


namespace NUMINAMATH_CALUDE_rounding_accuracy_of_1_35_billion_l1599_159995

theorem rounding_accuracy_of_1_35_billion :
  ∃ n : ℕ, (1350000000 : ℕ) = n * 10000000 ∧ n % 10 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_rounding_accuracy_of_1_35_billion_l1599_159995


namespace NUMINAMATH_CALUDE_black_lambs_count_all_lambs_accounted_l1599_159968

/-- The number of black lambs in Farmer Cunningham's flock -/
def black_lambs : ℕ := 5855

/-- The total number of lambs in Farmer Cunningham's flock -/
def total_lambs : ℕ := 6048

/-- The number of white lambs in Farmer Cunningham's flock -/
def white_lambs : ℕ := 193

/-- Theorem stating that the number of black lambs is correct -/
theorem black_lambs_count : black_lambs = total_lambs - white_lambs := by
  sorry

/-- Theorem stating that all lambs are accounted for -/
theorem all_lambs_accounted : total_lambs = black_lambs + white_lambs := by
  sorry

end NUMINAMATH_CALUDE_black_lambs_count_all_lambs_accounted_l1599_159968


namespace NUMINAMATH_CALUDE_fourth_power_inequality_l1599_159904

theorem fourth_power_inequality (a b c : ℝ) : 
  a^4 + b^4 + c^4 ≥ a^2*b^2 + b^2*c^2 + c^2*a^2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_inequality_l1599_159904


namespace NUMINAMATH_CALUDE_alley_width_l1599_159980

/-- Given a ladder of length a in an alley, making angles of 60° and 45° with the ground on opposite walls, 
    the width of the alley w is equal to (√3 * a) / 2. -/
theorem alley_width (a : ℝ) (w : ℝ) (h : ℝ) (k : ℝ) : 
  a > 0 → 
  k = a * (1 / 2) → 
  h = a * (Real.sqrt 2 / 2) → 
  w ^ 2 = h ^ 2 + k ^ 2 → 
  w = (Real.sqrt 3 * a) / 2 := by
sorry

end NUMINAMATH_CALUDE_alley_width_l1599_159980


namespace NUMINAMATH_CALUDE_least_value_quadratic_l1599_159957

theorem least_value_quadratic (y : ℝ) :
  (2 * y^2 + 7 * y + 3 = 5) → (y ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l1599_159957


namespace NUMINAMATH_CALUDE_parabola_point_value_l1599_159910

/-- Prove that for a parabola y = x^2 + (a+1)x + a passing through (-1, m), m must equal 0 -/
theorem parabola_point_value (a m : ℝ) : 
  ((-1)^2 + (a + 1)*(-1) + a = m) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_value_l1599_159910


namespace NUMINAMATH_CALUDE_max_value_constraint_l1599_159987

theorem max_value_constraint (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  ∃ (max : ℝ), max = 37 / 2 ∧ ∀ (a b c : ℝ), 9 * a^2 + 4 * b^2 + 25 * c^2 = 1 → 8 * a + 3 * b + 10 * c ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1599_159987


namespace NUMINAMATH_CALUDE_largest_term_at_125_l1599_159940

/-- The binomial coefficient (n choose k) -/
def choose (n k : ℕ) : ℚ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The kth term in the expansion of (1 + 0.3)^500 -/
def A (k : ℕ) : ℚ :=
  choose 500 k * (3/10)^k

/-- The theorem stating that A_k is largest when k = 125 -/
theorem largest_term_at_125 :
  ∀ j ∈ Finset.range 501, A 125 ≥ A j :=
sorry

end NUMINAMATH_CALUDE_largest_term_at_125_l1599_159940


namespace NUMINAMATH_CALUDE_grandfather_age_relationship_l1599_159923

/-- Represents the ages and relationships in the family problem -/
structure FamilyAges where
  fatherCurrentAge : ℕ
  sonCurrentAge : ℕ
  grandfatherAgeFiveYearsAgo : ℕ
  fatherAgeSameAsSonAtBirth : fatherCurrentAge = sonCurrentAge + sonCurrentAge
  fatherCurrentAge58 : fatherCurrentAge = 58
  sonAgeFiveYearsAgoHalfGrandfather : sonCurrentAge - 5 = (grandfatherAgeFiveYearsAgo - 5) / 2

/-- Theorem stating the relationship between the grandfather's age 5 years ago and the son's current age -/
theorem grandfather_age_relationship (f : FamilyAges) : 
  f.grandfatherAgeFiveYearsAgo = 2 * f.sonCurrentAge - 5 := by
  sorry

#check grandfather_age_relationship

end NUMINAMATH_CALUDE_grandfather_age_relationship_l1599_159923


namespace NUMINAMATH_CALUDE_cos_ninety_degrees_equals_zero_l1599_159917

theorem cos_ninety_degrees_equals_zero : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_ninety_degrees_equals_zero_l1599_159917


namespace NUMINAMATH_CALUDE_expected_hearts_in_modified_deck_l1599_159953

/-- A circular arrangement of cards -/
structure CircularDeck :=
  (total : ℕ)
  (hearts : ℕ)
  (h_total : total ≥ hearts)

/-- Expected number of adjacent heart pairs in a circular arrangement -/
def expected_adjacent_hearts (deck : CircularDeck) : ℚ :=
  (deck.hearts : ℚ) * (deck.hearts - 1) / (deck.total - 1)

theorem expected_hearts_in_modified_deck :
  let deck := CircularDeck.mk 40 10 (by norm_num)
  expected_adjacent_hearts deck = 30 / 13 := by
sorry

end NUMINAMATH_CALUDE_expected_hearts_in_modified_deck_l1599_159953


namespace NUMINAMATH_CALUDE_total_carrot_sticks_l1599_159974

def before_dinner : ℕ := 22
def after_dinner : ℕ := 15

theorem total_carrot_sticks : before_dinner + after_dinner = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_carrot_sticks_l1599_159974


namespace NUMINAMATH_CALUDE_henry_trips_problem_l1599_159922

def henry_trips (carry_capacity : ℕ) (table1_trays : ℕ) (table2_trays : ℕ) : ℕ :=
  (table1_trays + table2_trays + carry_capacity - 1) / carry_capacity

theorem henry_trips_problem : henry_trips 9 29 52 = 9 := by
  sorry

end NUMINAMATH_CALUDE_henry_trips_problem_l1599_159922


namespace NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l1599_159984

/-- The complex number z = (2-i)/(1+i) is located in the fourth quadrant of the complex plane. -/
theorem complex_in_fourth_quadrant : ∃ (x y : ℝ), 
  (x > 0 ∧ y < 0) ∧ 
  (Complex.I : ℂ) * ((2 : ℂ) - Complex.I) = ((1 : ℂ) + Complex.I) * (x + y * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l1599_159984


namespace NUMINAMATH_CALUDE_prob_B_not_occur_expected_value_B_l1599_159949

-- Define the sample space for a single die roll
def Ω : Finset ℕ := Finset.range 6

-- Define events A and B
def A : Finset ℕ := {0, 1, 2}
def B : Finset ℕ := {0, 1, 3}

-- Number of rolls
def n : ℕ := 10

-- Number of times event A occurred
def k : ℕ := 6

-- Probability of event A
def p_A : ℚ := (A.card : ℚ) / Ω.card

-- Probability of event B given A
def p_B_given_A : ℚ := ((A ∩ B).card : ℚ) / A.card

-- Probability of event B given not A
def p_B_given_not_A : ℚ := ((B \ A).card : ℚ) / (Ω \ A).card

-- Theorem for part (a)
theorem prob_B_not_occur (h : k = 6) :
  (Finset.card Ω)^n * (A.card)^k * ((Ω \ A).card)^(n - k) * ((A \ B).card)^k * ((Ω \ (A ∪ B)).card)^(n - k) / 
  (Finset.card Ω)^n / (Finset.card Ω)^n * Nat.choose n k = 64 / 236486 := by sorry

-- Theorem for part (b)
theorem expected_value_B (h : k = 6) :
  k * p_B_given_A + (n - k) * p_B_given_not_A = 16 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_B_not_occur_expected_value_B_l1599_159949


namespace NUMINAMATH_CALUDE_program_arrangement_count_l1599_159906

def num_singing_programs : ℕ := 4
def num_skit_programs : ℕ := 2
def num_singing_between_skits : ℕ := 3

def arrange_programs : ℕ := sorry

theorem program_arrangement_count :
  arrange_programs = 96 := by sorry

end NUMINAMATH_CALUDE_program_arrangement_count_l1599_159906


namespace NUMINAMATH_CALUDE_intersection_point_a_l1599_159920

/-- A function f(x) = 4x + b where b is an integer -/
def f (b : ℤ) : ℝ → ℝ := λ x ↦ 4 * x + b

/-- The inverse of f -/
noncomputable def f_inv (b : ℤ) : ℝ → ℝ := λ x ↦ (x - b) / 4

theorem intersection_point_a (b : ℤ) (a : ℤ) :
  f b (-4) = a ∧ f_inv b (-4) = a → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_a_l1599_159920


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l1599_159966

def ellipse (b : ℝ) (x y : ℝ) : Prop := x^2/4 + y^2/b^2 = 1

theorem max_value_on_ellipse (b : ℝ) (h : b > 0) :
  (∃ (x y : ℝ), ellipse b x y ∧ 
    ∀ (x' y' : ℝ), ellipse b x' y' → x^2 + 2*y ≥ x'^2 + 2*y') ∧
  (∃ (max : ℝ), 
    (0 < b ∧ b ≤ 4 → max = b^2/4 + 4) ∧
    (b > 4 → max = 2*b) ∧
    ∀ (x y : ℝ), ellipse b x y → x^2 + 2*y ≤ max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l1599_159966


namespace NUMINAMATH_CALUDE_intersection_empty_range_necessary_not_sufficient_range_l1599_159901

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 4}
def B : Set ℝ := {x | (x - 2) * (x - 3) ≤ 0}

-- Theorem 1: If A ∩ B = ∅, then a ∈ (-∞, -2) ∪ (7, ∞)
theorem intersection_empty_range (a : ℝ) : 
  A a ∩ B = ∅ → a < -2 ∨ a > 7 := by sorry

-- Theorem 2: If B is a necessary but not sufficient condition for A, then a ∈ [1, 6]
theorem necessary_not_sufficient_range (a : ℝ) :
  (B ⊆ A a ∧ ¬(A a ⊆ B)) → 1 ≤ a ∧ a ≤ 6 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_range_necessary_not_sufficient_range_l1599_159901


namespace NUMINAMATH_CALUDE_pentagon_area_fraction_l1599_159930

/-- Represents a rectangle with length 3 times its width -/
structure Rectangle where
  width : ℝ
  length : ℝ
  length_eq_3width : length = 3 * width

/-- Represents a pentagon formed by folding the rectangle -/
structure Pentagon where
  original : Rectangle
  area : ℝ

/-- The theorem to be proved -/
theorem pentagon_area_fraction (r : Rectangle) (p : Pentagon) 
  (h : p.original = r) : 
  p.area = (13 / 18) * (r.width * r.length) := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_fraction_l1599_159930


namespace NUMINAMATH_CALUDE_candles_remaining_l1599_159948

def total_candles : ℕ := 40
def alyssa_fraction : ℚ := 1/2
def chelsea_fraction : ℚ := 70/100

theorem candles_remaining (total : ℕ) (alyssa_frac chelsea_frac : ℚ) : 
  total - (alyssa_frac * total).floor - (chelsea_frac * (total - (alyssa_frac * total).floor)).floor = 6 :=
by sorry

#check candles_remaining total_candles alyssa_fraction chelsea_fraction

end NUMINAMATH_CALUDE_candles_remaining_l1599_159948


namespace NUMINAMATH_CALUDE_nonCongruentTrianglesCount_l1599_159908

-- Define the grid
def Grid := Fin 3 → Fin 3 → ℝ × ℝ

-- Define the grid with 0.5 unit spacing
def standardGrid : Grid :=
  λ i j => (0.5 * i.val, 0.5 * j.val)

-- Define a triangle as a tuple of three points
def Triangle := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

-- Define congruence for triangles
def areCongruent (t1 t2 : Triangle) : Prop := sorry

-- Define a function to generate all possible triangles from the grid
def allTriangles (g : Grid) : List Triangle := sorry

-- Define a function to count non-congruent triangles
def countNonCongruentTriangles (triangles : List Triangle) : Nat := sorry

-- The main theorem
theorem nonCongruentTrianglesCount :
  countNonCongruentTriangles (allTriangles standardGrid) = 9 := by sorry

end NUMINAMATH_CALUDE_nonCongruentTrianglesCount_l1599_159908


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1599_159937

/-- Given a line L1 defined by 4x + 5y = 15, prove that the x-intercept of the line L2
    that is perpendicular to L1 and has a y-intercept of -3 is 12/5. -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 5 * y = 15
  let m1 : ℝ := -4 / 5  -- slope of L1
  let m2 : ℝ := 5 / 4   -- slope of L2 (perpendicular to L1)
  let b2 : ℝ := -3      -- y-intercept of L2
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = m2 * x + b2
  let x_intercept : ℝ := 12 / 5
  (∀ x y, L2 x y → y = 0 → x = x_intercept) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1599_159937


namespace NUMINAMATH_CALUDE_oil_in_partial_tank_l1599_159962

theorem oil_in_partial_tank (tank_capacity : ℕ) (total_oil : ℕ) : 
  tank_capacity = 32 → total_oil = 728 → 
  total_oil % tank_capacity = 24 := by sorry

end NUMINAMATH_CALUDE_oil_in_partial_tank_l1599_159962


namespace NUMINAMATH_CALUDE_sum_of_digits_up_to_1000_l1599_159905

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := (Finset.range n).sum sumOfDigits

theorem sum_of_digits_up_to_1000 : sumOfDigitsUpTo 1000 = 14446 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_up_to_1000_l1599_159905


namespace NUMINAMATH_CALUDE_grass_field_width_l1599_159982

/-- Represents the width of the grass field -/
def field_width : ℝ := sorry

/-- The length of the grass field in meters -/
def field_length : ℝ := 75

/-- The width of the path around the field in meters -/
def path_width : ℝ := 2.5

/-- The cost of constructing the path per square meter in Rupees -/
def cost_per_sqm : ℝ := 2

/-- The total cost of constructing the path in Rupees -/
def total_cost : ℝ := 1350

/-- Theorem stating that given the conditions, the width of the grass field is 55 meters -/
theorem grass_field_width : 
  field_width = 55 := by sorry

end NUMINAMATH_CALUDE_grass_field_width_l1599_159982


namespace NUMINAMATH_CALUDE_area_ratio_S₂_to_S₁_l1599_159926

-- Define the sets S₁ and S₂
def S₁ : Set (ℝ × ℝ) := {p | Real.log (1 + p.1^2 + p.2^2) ≤ 1 + Real.log (p.1 + p.2)}
def S₂ : Set (ℝ × ℝ) := {p | Real.log (2 + p.1^2 + p.2^2) ≤ 2 + Real.log (p.1 + p.2)}

-- Define the areas of S₁ and S₂
noncomputable def area_S₁ : ℝ := Real.pi * 49
noncomputable def area_S₂ : ℝ := Real.pi * 4998

-- Theorem statement
theorem area_ratio_S₂_to_S₁ : area_S₂ / area_S₁ = 102 := by sorry

end NUMINAMATH_CALUDE_area_ratio_S₂_to_S₁_l1599_159926


namespace NUMINAMATH_CALUDE_c_value_l1599_159999

def f (a c x : ℝ) : ℝ := a * x^3 + c

theorem c_value (a c : ℝ) :
  (∃ x, f a c x = 20 ∧ x ∈ Set.Icc 1 2) ∧
  (∀ x, x ∈ Set.Icc 1 2 → f a c x ≤ 20) ∧
  (deriv (f a c) 1 = 6) →
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_c_value_l1599_159999


namespace NUMINAMATH_CALUDE_max_k_value_l1599_159924

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 1

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := Real.log x + k * x

theorem max_k_value :
  (∃ k : ℝ, ∀ x : ℝ, x > 0 → f x ≥ g k x) →
  (∀ k : ℝ, (∀ x : ℝ, x > 0 → f x ≥ g k x) → k ≤ 1) ∧
  (∀ x : ℝ, x > 0 → f x ≥ g 1 x) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l1599_159924


namespace NUMINAMATH_CALUDE_triangle_side_length_l1599_159928

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sinB : ℝ
  area : ℝ

-- Define the conditions
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

def isArithmeticSequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

def hasSinB (t : Triangle) : Prop :=
  t.sinB = 4/5

def hasArea (t : Triangle) : Prop :=
  t.area = 3/2

-- Theorem statement
theorem triangle_side_length (t : Triangle) 
  (h1 : isValidTriangle t)
  (h2 : isArithmeticSequence t)
  (h3 : hasSinB t)
  (h4 : hasArea t) :
  t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1599_159928


namespace NUMINAMATH_CALUDE_sum_of_first_five_primes_l1599_159971

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem sum_of_first_five_primes :
  first_five_primes.sum = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_five_primes_l1599_159971


namespace NUMINAMATH_CALUDE_campground_distance_l1599_159979

theorem campground_distance (speed1 speed2 speed3 time1 time2 time3 : ℝ) 
  (h1 : speed1 = 60)
  (h2 : speed2 = 50)
  (h3 : speed3 = 55)
  (h4 : time1 = 2)
  (h5 : time2 = 3)
  (h6 : time3 = 4) :
  speed1 * time1 + speed2 * time2 + speed3 * time3 = 490 :=
by
  sorry

end NUMINAMATH_CALUDE_campground_distance_l1599_159979


namespace NUMINAMATH_CALUDE_correct_cookies_in_partial_bag_edgars_cookies_l1599_159947

/-- Represents the number of cookies in a paper bag that is not full. -/
def cookiesInPartialBag (totalCookies bagCapacity : ℕ) : ℕ :=
  totalCookies % bagCapacity

/-- Proves that the number of cookies in a partial bag is correct. -/
theorem correct_cookies_in_partial_bag (totalCookies bagCapacity : ℕ) 
    (h1 : bagCapacity > 0) (h2 : totalCookies ≥ bagCapacity) :
  cookiesInPartialBag totalCookies bagCapacity = 
    totalCookies - bagCapacity * (totalCookies / bagCapacity) :=
by sorry

/-- The specific problem instance. -/
theorem edgars_cookies :
  cookiesInPartialBag 292 16 = 4 :=
by sorry

end NUMINAMATH_CALUDE_correct_cookies_in_partial_bag_edgars_cookies_l1599_159947


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_2x_l1599_159900

theorem factorization_x_squared_minus_2x (x : ℝ) : x^2 - 2*x = x*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_2x_l1599_159900


namespace NUMINAMATH_CALUDE_lauren_mail_total_l1599_159952

/-- The total number of pieces of mail sent by Lauren over four days -/
def total_mail (monday tuesday wednesday thursday : ℕ) : ℕ :=
  monday + tuesday + wednesday + thursday

/-- Theorem stating the total number of pieces of mail sent by Lauren -/
theorem lauren_mail_total : ∃ (monday tuesday wednesday thursday : ℕ),
  monday = 65 ∧
  tuesday = monday + 10 ∧
  wednesday = tuesday - 5 ∧
  thursday = wednesday + 15 ∧
  total_mail monday tuesday wednesday thursday = 295 :=
by sorry

end NUMINAMATH_CALUDE_lauren_mail_total_l1599_159952


namespace NUMINAMATH_CALUDE_proportion_not_true_l1599_159986

theorem proportion_not_true (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : 3 * a = 5 * b) :
  ¬(a / b = 3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_proportion_not_true_l1599_159986


namespace NUMINAMATH_CALUDE_unit_digit_of_product_l1599_159969

def numbers : List Nat := [6245, 7083, 9137, 4631, 5278, 3974]

theorem unit_digit_of_product (nums : List Nat := numbers) :
  (nums.foldl (· * ·) 1) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_product_l1599_159969


namespace NUMINAMATH_CALUDE_intersection_point_a_l1599_159955

/-- A linear function f(x) = 4x + b -/
def f (b : ℤ) : ℝ → ℝ := λ x ↦ 4 * x + b

/-- The inverse of f -/
noncomputable def f_inv (b : ℤ) : ℝ → ℝ := λ x ↦ (x - b) / 4

theorem intersection_point_a (b : ℤ) (a : ℤ) :
  f b 4 = a ∧ f_inv b a = 4 → a = 4 := by sorry

end NUMINAMATH_CALUDE_intersection_point_a_l1599_159955


namespace NUMINAMATH_CALUDE_quiz_probability_l1599_159934

theorem quiz_probability : 
  let n_questions : ℕ := 5
  let n_choices : ℕ := 6
  let p_correct : ℚ := 1 / n_choices
  let p_incorrect : ℚ := 1 - p_correct
  1 - p_incorrect ^ n_questions = 4651 / 7776 :=
by sorry

end NUMINAMATH_CALUDE_quiz_probability_l1599_159934


namespace NUMINAMATH_CALUDE_jacob_phoebe_age_fraction_l1599_159918

/-- Represents the ages and relationships of Rehana, Phoebe, and Jacob -/
structure AgeRelationship where
  rehana_current_age : ℕ
  jacob_current_age : ℕ
  years_until_comparison : ℕ
  rehana_phoebe_ratio : ℕ

/-- The fraction of Phoebe's age that Jacob's age represents -/
def age_fraction (ar : AgeRelationship) : ℚ :=
  ar.jacob_current_age / (ar.rehana_current_age + ar.years_until_comparison - ar.years_until_comparison * ar.rehana_phoebe_ratio)

/-- Theorem stating that given the conditions, Jacob's age is 3/5 of Phoebe's age -/
theorem jacob_phoebe_age_fraction :
  ∀ (ar : AgeRelationship),
  ar.rehana_current_age = 25 →
  ar.jacob_current_age = 3 →
  ar.years_until_comparison = 5 →
  ar.rehana_phoebe_ratio = 3 →
  age_fraction ar = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_jacob_phoebe_age_fraction_l1599_159918


namespace NUMINAMATH_CALUDE_hyperbola_dot_product_theorem_l1599_159991

/-- The hyperbola in the Cartesian coordinate system -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := (-2, 0)

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := (2, 0)

/-- A point on the hyperbola -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The vector from one point to another -/
def vector (a b : ℝ × ℝ) : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)

/-- The theorem to be proved -/
theorem hyperbola_dot_product_theorem 
  (P Q : HyperbolaPoint) 
  (h_line : ∃ (m b : ℝ), P.y = m * P.x + b ∧ Q.y = m * Q.x + b ∧ F₁.2 = m * F₁.1 + b) 
  (h_dot_product : dot_product (vector F₁ F₂) (vector F₁ (P.x, P.y)) = 16) :
  dot_product (vector F₂ (P.x, P.y)) (vector F₂ (Q.x, Q.y)) = 27 / 13 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_dot_product_theorem_l1599_159991


namespace NUMINAMATH_CALUDE_decimal_division_division_result_l1599_159989

theorem decimal_division (x y : ℚ) : x / y = (x * 1000) / (y * 1000) := by sorry

theorem division_result : (0.25 : ℚ) / (0.005 : ℚ) = 50 := by sorry

end NUMINAMATH_CALUDE_decimal_division_division_result_l1599_159989


namespace NUMINAMATH_CALUDE_square_sum_equality_l1599_159931

theorem square_sum_equality (a b : ℕ) (h1 : a^2 = 225) (h2 : b^2 = 25) :
  a^2 + 2*a*b + b^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l1599_159931


namespace NUMINAMATH_CALUDE_tangent_equality_at_neg_one_tangent_equality_range_l1599_159976

/-- The function f(x) = x³ - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The function g(x) = x² + a -/
def g (a x : ℝ) : ℝ := x^2 + a

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The derivative of g -/
def g' (x : ℝ) : ℝ := 2 * x

/-- Theorem 1: When the tangent lines of f and g are equal at x₁ = -1, a = 3 -/
theorem tangent_equality_at_neg_one (a : ℝ) :
  (∃ x₂, f' (-1) = g' x₂ ∧ f (-1) - f' (-1) * (-1) = g a x₂ - g' x₂ * x₂) → a = 3 :=
sorry

/-- Theorem 2: The range of a for which the tangent line of f at some point 
    is also a tangent line of g is [-1, +∞) -/
theorem tangent_equality_range :
  {a : ℝ | ∃ x₁ x₂, f' x₁ = g' x₂ ∧ f x₁ - f' x₁ * x₁ = g a x₂ - g' x₂ * x₂} = Set.Ici (-1) :=
sorry

end NUMINAMATH_CALUDE_tangent_equality_at_neg_one_tangent_equality_range_l1599_159976


namespace NUMINAMATH_CALUDE_parabola_translation_l1599_159973

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The standard parabola y = x^2 -/
def standard_parabola : Parabola := ⟨1, 0, 0⟩

/-- The final parabola y = (x+4)^2 - 5 -/
def final_parabola : Parabola := ⟨1, 8, 11⟩

/-- Translate a parabola horizontally -/
def translate_horizontal (p : Parabola) (d : ℝ) : Parabola :=
  ⟨p.a, p.b - 2 * p.a * d, p.a * d^2 + p.b * d + p.c⟩

/-- Translate a parabola vertically -/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  ⟨p.a, p.b, p.c + d⟩

/-- Theorem: The final parabola can be obtained by translating the standard parabola
    4 units to the left and then 5 units downward -/
theorem parabola_translation :
  translate_vertical (translate_horizontal standard_parabola (-4)) (-5) = final_parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l1599_159973


namespace NUMINAMATH_CALUDE_tank_capacity_after_adding_gas_l1599_159903

/-- 
Given a tank with a capacity of 48 gallons, initially filled to 3/4 of its capacity,
prove that after adding 8 gallons of gasoline, the tank will be filled to 11/12 of its capacity.
-/
theorem tank_capacity_after_adding_gas (tank_capacity : ℚ) (initial_fill_fraction : ℚ) 
  (added_gas : ℚ) (final_fill_fraction : ℚ) : 
  tank_capacity = 48 → 
  initial_fill_fraction = 3/4 → 
  added_gas = 8 → 
  final_fill_fraction = (initial_fill_fraction * tank_capacity + added_gas) / tank_capacity →
  final_fill_fraction = 11/12 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_after_adding_gas_l1599_159903


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l1599_159933

theorem smallest_n_satisfying_inequality : 
  ∀ n : ℕ, n > 0 → (1 / n - 1 / (n + 1) < 1 / 15) ↔ n ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l1599_159933
