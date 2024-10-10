import Mathlib

namespace decreasing_quadratic_implies_a_geq_6_l2615_261588

/-- A function f(x) = x^2 - 2(a-1)x + 2 is decreasing on the interval (-∞, 5] -/
def is_decreasing_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, x < y → x ≤ 5 → y ≤ 5 → f x ≥ f y

/-- The quadratic function f(x) = x^2 - 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*(a-1)*x + 2

theorem decreasing_quadratic_implies_a_geq_6 :
  ∀ a : ℝ, is_decreasing_on_interval (f a) a → a ≥ 6 :=
sorry

end decreasing_quadratic_implies_a_geq_6_l2615_261588


namespace smaller_circle_radius_l2615_261551

/-- Given two concentric circles with areas A1 and A1 + A2, where the larger circle
    has radius 5 and A1, A2, A1 + A2 form an arithmetic progression,
    prove that the radius of the smaller circle is 5√2/2 -/
theorem smaller_circle_radius
  (A1 A2 : ℝ)
  (h1 : A1 > 0)
  (h2 : A2 > 0)
  (h3 : (A1 + A2) = π * 5^2)
  (h4 : A2 = (A1 + (A1 + A2)) / 2)
  : ∃ (r : ℝ), r > 0 ∧ A1 = π * r^2 ∧ r = 5 * Real.sqrt 2 / 2 :=
sorry

end smaller_circle_radius_l2615_261551


namespace fixed_points_of_f_squared_l2615_261581

def X := ℤ × ℤ × ℤ

def f (x : X) : X :=
  let (a, b, c) := x
  (a + b + c, a * b + b * c + c * a, a * b * c)

theorem fixed_points_of_f_squared (a b c : ℤ) :
  f (f (a, b, c)) = (a, b, c) ↔ 
    ((∃ k : ℤ, (a, b, c) = (k, 0, 0)) ∨ (a, b, c) = (-1, -1, 1)) := by
  sorry

end fixed_points_of_f_squared_l2615_261581


namespace expression_evaluation_l2615_261589

theorem expression_evaluation : 
  let c : ℝ := 2
  let d : ℝ := 1/4
  (Real.sqrt (c - d) / (c^2 * Real.sqrt (2*c))) * 
  (Real.sqrt ((c - d)/(c + d)) + Real.sqrt ((c^2 + c*d)/(c^2 - c*d))) = 1/3 := by
sorry

end expression_evaluation_l2615_261589


namespace unique_function_satisfying_equation_l2615_261561

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, 
    f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x^2 * f y + 4 * y^2 :=
by
  -- The unique function is f(x) = x^2
  use fun x => x^2
  sorry

end unique_function_satisfying_equation_l2615_261561


namespace books_to_pens_ratio_l2615_261572

def total_stationery : ℕ := 400
def num_books : ℕ := 280

theorem books_to_pens_ratio :
  let num_pens := total_stationery - num_books
  (num_books / (Nat.gcd num_books num_pens)) = 7 ∧
  (num_pens / (Nat.gcd num_books num_pens)) = 3 := by
  sorry

end books_to_pens_ratio_l2615_261572


namespace smallest_n_for_inequality_l2615_261544

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ 
  (∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
  (∀ (m : ℕ), m < n → ∃ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) :=
by sorry

end smallest_n_for_inequality_l2615_261544


namespace price_reduction_order_invariance_l2615_261568

theorem price_reduction_order_invariance :
  let reduction1 := 0.1
  let reduction2 := 0.15
  let total_reduction1 := 1 - (1 - reduction1) * (1 - reduction2)
  let total_reduction2 := 1 - (1 - reduction2) * (1 - reduction1)
  total_reduction1 = total_reduction2 ∧ total_reduction1 = 0.235 := by
  sorry

end price_reduction_order_invariance_l2615_261568


namespace resulting_surface_area_l2615_261579

/-- Represents the dimensions of the large cube -/
def large_cube_dim : ℕ := 12

/-- Represents the dimensions of the small cubes -/
def small_cube_dim : ℕ := 3

/-- The number of small cubes in the original large cube -/
def total_small_cubes : ℕ := 64

/-- The number of small cubes removed -/
def removed_cubes : ℕ := 8

/-- The number of remaining small cubes after removal -/
def remaining_cubes : ℕ := total_small_cubes - removed_cubes

/-- The surface area of a single small cube before modification -/
def small_cube_surface : ℕ := 6 * small_cube_dim ^ 2

/-- The number of new surfaces exposed per small cube after modification -/
def new_surfaces_per_cube : ℕ := 12

/-- The number of edge-shared internal faces -/
def edge_shared_faces : ℕ := 12

/-- The area of each edge-shared face -/
def edge_shared_face_area : ℕ := small_cube_dim ^ 2

/-- Theorem stating the surface area of the resulting structure -/
theorem resulting_surface_area :
  (remaining_cubes * (small_cube_surface + new_surfaces_per_cube)) -
  (4 * 3 * edge_shared_faces * edge_shared_face_area) = 3408 := by
  sorry

end resulting_surface_area_l2615_261579


namespace smallest_solution_congruences_l2615_261519

theorem smallest_solution_congruences :
  ∃ x : ℕ, x > 0 ∧
    x % 2 = 1 ∧
    x % 3 = 2 ∧
    x % 4 = 3 ∧
    x % 5 = 4 ∧
    (∀ y : ℕ, y > 0 →
      y % 2 = 1 →
      y % 3 = 2 →
      y % 4 = 3 →
      y % 5 = 4 →
      y ≥ x) ∧
  x = 59 := by
  sorry

end smallest_solution_congruences_l2615_261519


namespace max_coins_distribution_l2615_261585

theorem max_coins_distribution (n : ℕ) : 
  n < 150 ∧ 
  ∃ k : ℕ, n = 13 * k + 3 →
  n ≤ 146 :=
by sorry

end max_coins_distribution_l2615_261585


namespace no_common_terms_except_one_l2615_261527

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem no_common_terms_except_one : ∀ n : ℕ, x n = y n → x n = 1 := by sorry

end no_common_terms_except_one_l2615_261527


namespace dean_books_count_l2615_261555

theorem dean_books_count (tony_books breanna_books total_different_books : ℕ)
  (tony_dean_shared all_shared : ℕ) :
  tony_books = 23 →
  breanna_books = 17 →
  tony_dean_shared = 3 →
  all_shared = 1 →
  total_different_books = 47 →
  ∃ dean_books : ℕ,
    dean_books = 16 ∧
    total_different_books =
      (tony_books - tony_dean_shared - all_shared) +
      (dean_books - tony_dean_shared - all_shared) +
      (breanna_books - all_shared) :=
by sorry

end dean_books_count_l2615_261555


namespace equidistant_point_x_coordinate_l2615_261570

/-- The x-coordinate of the point on the x-axis equidistant from A(-4, 0) and B(2, 6) is 2 -/
theorem equidistant_point_x_coordinate : 
  ∃ (x : ℝ), (x + 4)^2 = (x - 2)^2 + 36 ∧ x = 2 := by
  sorry

end equidistant_point_x_coordinate_l2615_261570


namespace ellipse_properties_l2615_261598

/-- An ellipse with equation x^2/2 + y^2 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

/-- The circle with diameter F₁F₂ -/
def Circle : Set (ℝ × ℝ) :=
  {p | (p.1^2 + p.2^2) = 1}

/-- The line x + y - √2 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p | p.1 + p.2 = Real.sqrt 2}

/-- The dot product of vectors PF₁ and PF₂ -/
def dotProduct (P : ℝ × ℝ) : ℝ :=
  (F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2)

theorem ellipse_properties :
  (∀ p ∈ Circle, p ∈ Line) ∧
  (∀ P ∈ Ellipse, dotProduct P ≥ 0 ∧ ∃ Q ∈ Ellipse, dotProduct Q = 0) :=
sorry

end ellipse_properties_l2615_261598


namespace star_commutative_l2615_261587

/-- Binary operation ★ defined for integers -/
def star (a b : ℤ) : ℤ := a^2 + b^2

/-- Theorem stating that ★ is commutative for all integers -/
theorem star_commutative : ∀ (a b : ℤ), star a b = star b a := by
  sorry

end star_commutative_l2615_261587


namespace soda_quarters_needed_l2615_261567

theorem soda_quarters_needed (total_amount : ℚ) (quarters_per_soda : ℕ) : 
  total_amount = 213.75 ∧ quarters_per_soda = 7 →
  (⌊(total_amount / 0.25) / quarters_per_soda⌋ + 1) * quarters_per_soda - 
  (total_amount / 0.25).floor = 6 := by
  sorry

end soda_quarters_needed_l2615_261567


namespace bezout_identity_solutions_l2615_261595

theorem bezout_identity_solutions (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (u₀ v₀ : ℤ), ∀ (u v : ℤ),
    (a * u + b * v = Int.gcd a b) ↔ ∃ (k : ℤ), u = u₀ - k * b ∧ v = v₀ + k * a :=
by sorry

end bezout_identity_solutions_l2615_261595


namespace sequence_not_in_interval_l2615_261563

/-- Given an infinite sequence of real numbers {aₙ} where aₙ₊₁ = √(aₙ² + aₙ - 1) for all n ≥ 1,
    prove that a₁ ∉ (-2, 1). -/
theorem sequence_not_in_interval (a : ℕ → ℝ) 
    (h : ∀ n : ℕ, a (n + 1) = Real.sqrt ((a n)^2 + a n - 1)) : 
    a 1 ∉ Set.Ioo (-2 : ℝ) 1 := by
  sorry

end sequence_not_in_interval_l2615_261563


namespace correct_sampling_methods_l2615_261518

/-- Represents a sampling method -/
inductive SamplingMethod
  | Systematic
  | SimpleRandom
  | Stratified

/-- Represents a sampling scenario -/
structure SamplingScenario where
  method : SamplingMethod
  description : String

/-- The milk production line sampling scenario -/
def milkProductionScenario : SamplingScenario :=
  { method := SamplingMethod.Systematic,
    description := "Sampling a bag every 30 minutes on a milk production line" }

/-- The math enthusiasts sampling scenario -/
def mathEnthusiastsScenario : SamplingScenario :=
  { method := SamplingMethod.SimpleRandom,
    description := "Selecting 3 individuals from 30 math enthusiasts in a middle school" }

/-- Theorem stating that the sampling methods are correctly identified -/
theorem correct_sampling_methods :
  (milkProductionScenario.method = SamplingMethod.Systematic) ∧
  (mathEnthusiastsScenario.method = SamplingMethod.SimpleRandom) :=
sorry

end correct_sampling_methods_l2615_261518


namespace sum_f_equals_1326_l2615_261556

/-- The number of integer points on the line segment from (0,0) to (n, n+3), excluding endpoints -/
def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

/-- The sum of f(n) for n from 1 to 1990 -/
def sum_f : ℕ := (Finset.range 1990).sum f

theorem sum_f_equals_1326 : sum_f = 1326 := by sorry

end sum_f_equals_1326_l2615_261556


namespace parallel_subset_parallel_perpendicular_planes_parallel_l2615_261554

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (lineparallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Theorem 1
theorem parallel_subset_parallel 
  (a : Line) (α β : Plane) :
  parallel α β → subset a α → lineparallel a β := by sorry

-- Theorem 2
theorem perpendicular_planes_parallel 
  (a : Line) (α β : Plane) :
  perpendicular a α → perpendicular a β → parallel α β := by sorry

end parallel_subset_parallel_perpendicular_planes_parallel_l2615_261554


namespace smallest_fixed_point_of_R_l2615_261537

/-- The transformation R that reflects a line first on l₁: y = √3x and then on l₂: y = -√3x -/
def R (l : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The n-th iteration of R -/
def R_iter (n : ℕ) (l : ℝ → ℝ) : ℝ → ℝ :=
  match n with
  | 0 => l
  | n + 1 => R (R_iter n l)

/-- Any line can be represented as y = kx for some k -/
def line (k : ℝ) : ℝ → ℝ := λ x => k * x

theorem smallest_fixed_point_of_R :
  ∀ k : ℝ, ∃ m : ℕ, m > 0 ∧ R_iter m (line k) = line k ∧
  ∀ n : ℕ, 0 < n → n < m → R_iter n (line k) ≠ line k :=
by sorry

end smallest_fixed_point_of_R_l2615_261537


namespace ratio_sum_squares_implies_sum_l2615_261523

theorem ratio_sum_squares_implies_sum (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a →
  c = 3 * a →
  a^2 + b^2 + c^2 = 2016 →
  a + b + c = 72 := by
sorry

end ratio_sum_squares_implies_sum_l2615_261523


namespace vision_survey_is_sampling_l2615_261536

/-- Represents a survey method -/
inductive SurveyMethod
| Sampling
| Census
| Other

/-- Represents a school with a given population of eighth-grade students -/
structure School where
  population : ℕ

/-- Represents a vision survey conducted in a school -/
structure VisionSurvey where
  school : School
  sample_size : ℕ
  selection_method : String

/-- Determines the survey method based on the vision survey parameters -/
def determine_survey_method (survey : VisionSurvey) : SurveyMethod :=
  if survey.sample_size < survey.school.population ∧ survey.selection_method = "Random" then
    SurveyMethod.Sampling
  else if survey.sample_size = survey.school.population then
    SurveyMethod.Census
  else
    SurveyMethod.Other

/-- Theorem stating that the given vision survey uses a sampling survey method -/
theorem vision_survey_is_sampling (school : School) (survey : VisionSurvey) :
  school.population = 400 →
  survey.school = school →
  survey.sample_size = 80 →
  survey.selection_method = "Random" →
  determine_survey_method survey = SurveyMethod.Sampling :=
by
  sorry

#check vision_survey_is_sampling

end vision_survey_is_sampling_l2615_261536


namespace triangle_properties_l2615_261599

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) :
  (t.a / t.b = (1 + Real.cos t.A) / Real.cos t.C) →
  (t.A = π / 2) ∧
  (t.a = 1 → ∃ S : ℝ, S ≤ 1/4 ∧ 
    ∀ S' : ℝ, (∃ t' : Triangle, t'.a = 1 ∧ t'.A = π/2 ∧ S' = 1/2 * t'.b * t'.c) → 
      S' ≤ S) :=
by sorry

end triangle_properties_l2615_261599


namespace monomial_existence_l2615_261505

/-- A monomial in variables a and b -/
structure Monomial where
  coeff : ℤ
  a_power : ℕ
  b_power : ℕ

/-- Multiplication of monomials -/
def mul_monomial (x y : Monomial) : Monomial :=
  { coeff := x.coeff * y.coeff,
    a_power := x.a_power + y.a_power,
    b_power := x.b_power + y.b_power }

/-- Addition of monomials -/
def add_monomial (x y : Monomial) : Option Monomial :=
  if x.a_power = y.a_power ∧ x.b_power = y.b_power then
    some { coeff := x.coeff + y.coeff,
           a_power := x.a_power,
           b_power := x.b_power }
  else
    none

theorem monomial_existence : ∃ (x y : Monomial),
  (mul_monomial x y = { coeff := -12, a_power := 4, b_power := 2 }) ∧
  (∃ (z : Monomial), add_monomial x y = some z ∧ z.coeff = 1) :=
sorry

end monomial_existence_l2615_261505


namespace initial_fund_calculation_l2615_261524

theorem initial_fund_calculation (initial_per_employee final_per_employee undistributed : ℕ) : 
  initial_per_employee = 50 →
  final_per_employee = 45 →
  undistributed = 95 →
  (initial_per_employee - final_per_employee) * (undistributed / (initial_per_employee - final_per_employee)) = 950 := by
  sorry

end initial_fund_calculation_l2615_261524


namespace parabola_equation_l2615_261515

/-- A parabola with vertex at the origin, coordinate axes as axes of symmetry, 
    and passing through (-2, 3) has equation x^2 = (4/3)y or y^2 = -(9/2)x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ (x = 0 ∧ y = 0) ∨ (x = -2 ∧ y = 3)) →  -- vertex at origin and passes through (-2, 3)
  (∀ x y, p (x, y) ↔ p (-x, y)) →  -- symmetry about y-axis
  (∀ x y, p (x, y) ↔ p (x, -y)) →  -- symmetry about x-axis
  (∃ a b : ℝ, (∀ x y, p (x, y) ↔ x^2 = a*y) ∨ (∀ x y, p (x, y) ↔ y^2 = b*x)) →
  (∀ x y, p (x, y) ↔ x^2 = (4/3)*y ∨ y^2 = -(9/2)*x) :=
by sorry

end parabola_equation_l2615_261515


namespace zoe_songs_total_l2615_261509

theorem zoe_songs_total (country_albums : ℕ) (pop_albums : ℕ) (songs_per_album : ℕ) : 
  country_albums = 3 → pop_albums = 5 → songs_per_album = 3 →
  (country_albums + pop_albums) * songs_per_album = 24 := by
sorry

end zoe_songs_total_l2615_261509


namespace largest_parabolic_slice_l2615_261534

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a circle in 3D space -/
structure Circle3D where
  center : Point3D
  radius : ℝ

/-- Represents a cone in 3D space -/
structure Cone where
  vertex : Point3D
  base : Circle3D

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  point : Point3D

/-- Calculates the area of a parabolic slice -/
def parabolicSliceArea (cone : Cone) (plane : Plane) : ℝ := sorry

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (p : Point3D) (a : Point3D) (b : Point3D) : Prop := sorry

/-- Theorem: The largest area parabolic slice is obtained when the midpoint of
    the intersection of the cutting plane with the base circle bisects AO -/
theorem largest_parabolic_slice (cone : Cone) (plane : Plane) :
  let A := sorry -- Point on base circle
  let O := cone.base.center
  let E := sorry -- Midpoint of intersection of plane with base circle
  (∀ p : Plane, parabolicSliceArea cone p ≤ parabolicSliceArea cone plane) ↔
  isMidpoint E A O := by sorry

end largest_parabolic_slice_l2615_261534


namespace man_downstream_speed_l2615_261525

/-- Calculates the downstream speed of a man given his upstream and still water speeds -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem: Given a man's upstream speed of 26 km/h and still water speed of 28 km/h, 
    his downstream speed is 30 km/h -/
theorem man_downstream_speed :
  downstream_speed 26 28 = 30 := by
  sorry

end man_downstream_speed_l2615_261525


namespace problem_statement_l2615_261577

theorem problem_statement (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a^2 + b^2 = a*b + 1) (hcd : c*d > 1) :
  (a + b ≤ 2) ∧ (Real.sqrt (a*c) + Real.sqrt (b*d) < c + d) := by
  sorry

end problem_statement_l2615_261577


namespace smallest_four_digit_divisible_by_12_l2615_261553

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_12 :
  ∀ n : ℕ, is_four_digit n → (sum_of_first_n n % 12 = 0) → n ≥ 1001 :=
by sorry

end smallest_four_digit_divisible_by_12_l2615_261553


namespace servant_worked_months_l2615_261533

def yearly_salary : ℚ := 90
def turban_value : ℚ := 50
def received_cash : ℚ := 55

def total_yearly_salary : ℚ := yearly_salary + turban_value
def monthly_salary : ℚ := total_yearly_salary / 12
def total_received : ℚ := received_cash + turban_value

theorem servant_worked_months : 
  ∃ (months : ℚ), months * monthly_salary = total_received ∧ months = 9 := by
  sorry

end servant_worked_months_l2615_261533


namespace square_rolling_octagon_l2615_261586

/-- Represents the faces of a square -/
inductive SquareFace
  | Left
  | Right
  | Top
  | Bottom

/-- Represents the rotation of a square -/
def squareRotation (n : ℕ) : ℕ := n * 135

/-- The final position of an object on a square face after rolling around an octagon -/
def finalPosition (initialFace : SquareFace) : SquareFace :=
  match (squareRotation 4) % 360 with
  | 180 => match initialFace with
    | SquareFace.Left => SquareFace.Right
    | SquareFace.Right => SquareFace.Left
    | SquareFace.Top => SquareFace.Bottom
    | SquareFace.Bottom => SquareFace.Top
  | _ => initialFace

theorem square_rolling_octagon :
  finalPosition SquareFace.Left = SquareFace.Right :=
by sorry

end square_rolling_octagon_l2615_261586


namespace problem_solution_l2615_261584

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem problem_solution (a : ℝ) (m : ℝ) : 
  (∀ x : ℝ, f a x ≤ 3 ↔ x ∈ Set.Icc (-6) 0) →
  (a = -3 ∧ 
   (∀ x : ℝ, f a x + f a (x + 5) ≥ 2 * m) → m ≤ 5/2) :=
by sorry

end problem_solution_l2615_261584


namespace smallest_solution_floor_equation_l2615_261508

theorem smallest_solution_floor_equation :
  ∀ x : ℝ, (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ)^2 = 19 → x ≥ Real.sqrt 119 := by
  sorry

end smallest_solution_floor_equation_l2615_261508


namespace complex_magnitude_product_l2615_261514

theorem complex_magnitude_product : 
  Complex.abs ((Real.sqrt 8 - Complex.I * 2) * (Real.sqrt 3 * 2 + Complex.I * 6)) = 24 := by
  sorry

end complex_magnitude_product_l2615_261514


namespace notebooks_distribution_l2615_261530

/-- 
Given a class where:
- The total number of notebooks distributed is 512
- Each child initially receives a number of notebooks equal to 1/8 of the total number of children
Prove that if the number of children is halved, each child would receive 16 notebooks.
-/
theorem notebooks_distribution (C : ℕ) (h1 : C > 0) : 
  (C * (C / 8) = 512) → ((512 / (C / 2)) = 16) :=
by sorry

end notebooks_distribution_l2615_261530


namespace area_ratio_ABJ_ADE_l2615_261564

/-- Represents a regular octagon with vertices ABCDEFGH -/
structure RegularOctagon where
  vertices : Fin 8 → Point

/-- Represents a triangle within the regular octagon -/
structure OctagonTriangle where
  vertices : Fin 3 → Point

/-- The area of a triangle -/
def area (t : OctagonTriangle) : ℝ := sorry

/-- The regular octagon ABCDEFGH -/
def octagon : RegularOctagon := sorry

/-- Triangle ABJ formed by two smaller equilateral triangles -/
def triangle_ABJ : OctagonTriangle := sorry

/-- Triangle ADE formed by connecting every third vertex of the octagon -/
def triangle_ADE : OctagonTriangle := sorry

theorem area_ratio_ABJ_ADE :
  area triangle_ABJ / area triangle_ADE = 2 / 3 := by sorry

end area_ratio_ABJ_ADE_l2615_261564


namespace original_people_count_l2615_261539

theorem original_people_count (x : ℚ) : 
  (2 * x / 3 + 6 - x / 6 = 15) → x = 27 := by
  sorry

end original_people_count_l2615_261539


namespace detergent_amount_in_altered_solution_l2615_261543

/-- The ratio of bleach to detergent to water in a solution -/
structure SolutionRatio :=
  (bleach : ℚ)
  (detergent : ℚ)
  (water : ℚ)

/-- The amount of each component in a solution -/
structure SolutionAmount :=
  (bleach : ℚ)
  (detergent : ℚ)
  (water : ℚ)

def original_ratio : SolutionRatio :=
  { bleach := 2, detergent := 25, water := 100 }

def altered_ratio (r : SolutionRatio) : SolutionRatio :=
  { bleach := 3 * r.bleach,
    detergent := r.detergent,
    water := 2 * r.water }

def water_amount : ℚ := 300

theorem detergent_amount_in_altered_solution :
  ∀ (r : SolutionRatio) (w : ℚ),
  r = original_ratio →
  w = water_amount →
  ∃ (a : SolutionAmount),
    a.water = w ∧
    a.detergent = 37.5 ∧
    a.bleach / a.detergent = (altered_ratio r).bleach / (altered_ratio r).detergent ∧
    a.detergent / a.water = (altered_ratio r).detergent / (altered_ratio r).water :=
by sorry

end detergent_amount_in_altered_solution_l2615_261543


namespace congruence_solution_l2615_261532

theorem congruence_solution (n : ℤ) : 19 * n ≡ 13 [ZMOD 47] → n ≡ 25 [ZMOD 47] := by
  sorry

end congruence_solution_l2615_261532


namespace smaller_circle_radius_l2615_261507

/-- Given two circles where one encloses the other, this theorem proves
    the radius of the smaller circle given specific conditions. -/
theorem smaller_circle_radius
  (R : ℝ) -- Radius of the larger circle
  (r : ℝ) -- Radius of the smaller circle
  (A₁ : ℝ) -- Area of the smaller circle
  (A₂ : ℝ) -- Area difference between the two circles
  (h1 : R = 5) -- The larger circle has a radius of 5 units
  (h2 : A₁ = π * r^2) -- Area formula for the smaller circle
  (h3 : A₂ = π * R^2 - A₁) -- Area difference
  (h4 : ∃ (d : ℝ), A₁ + d = A₂ ∧ A₂ + d = A₁ + A₂) -- Arithmetic progression condition
  : r = 5 * Real.sqrt 2 / 2 := by
  sorry

end smaller_circle_radius_l2615_261507


namespace point_not_in_region_l2615_261513

def plane_region (x y : ℝ) : Prop := 3 * x + 2 * y < 6

theorem point_not_in_region : ¬ plane_region 2 0 := by
  sorry

end point_not_in_region_l2615_261513


namespace max_a_value_l2615_261592

theorem max_a_value (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) :
  a ≤ 59 ∧ ∃ (a' b' : ℤ), a' = 59 ∧ b' = 1 ∧ a' > b' ∧ b' > 0 ∧ a' + b' + a' * b' = 119 :=
sorry

end max_a_value_l2615_261592


namespace divisor_pairing_l2615_261512

theorem divisor_pairing (n : ℕ+) (h : ¬ ∃ m : ℕ, n = m ^ 2) :
  ∃ f : {d : ℕ // d ∣ n} → {d : ℕ // d ∣ n},
    ∀ d : {d : ℕ // d ∣ n}, 
      (f (f d) = d) ∧ 
      ((d.val ∣ (f d).val) ∨ ((f d).val ∣ d.val)) :=
sorry

end divisor_pairing_l2615_261512


namespace expression_factorization_l2615_261575

theorem expression_factorization (b : ℝ) : 
  (8 * b^3 + 45 * b^2 - 10) - (-12 * b^3 + 5 * b^2 - 10) = 20 * b^2 * (b + 2) := by
  sorry

end expression_factorization_l2615_261575


namespace T_is_integer_at_smallest_n_l2615_261573

/-- Sum of reciprocals of non-zero digits from 1 to 5^n -/
def T (n : ℕ) : ℚ :=
  sorry

/-- The smallest positive integer n for which T n is an integer -/
def smallest_n : ℕ := 504

theorem T_is_integer_at_smallest_n :
  (T smallest_n).isInt ∧ ∀ m : ℕ, m > 0 ∧ m < smallest_n → ¬(T m).isInt :=
sorry

end T_is_integer_at_smallest_n_l2615_261573


namespace living_room_set_cost_l2615_261549

theorem living_room_set_cost (coach_cost sectional_cost paid_amount : ℚ)
  (h1 : coach_cost = 2500)
  (h2 : sectional_cost = 3500)
  (h3 : paid_amount = 7200)
  (discount_rate : ℚ)
  (h4 : discount_rate = 0.1) :
  ∃ (additional_cost : ℚ),
    paid_amount = (1 - discount_rate) * (coach_cost + sectional_cost + additional_cost) ∧
    additional_cost = 2000 := by
sorry

end living_room_set_cost_l2615_261549


namespace area_of_inner_rectangle_l2615_261522

theorem area_of_inner_rectangle (s : ℝ) (h : s > 0) : 
  let larger_square_area := s^2
  let half_larger_square_area := larger_square_area / 2
  let inner_rectangle_side := s / 2
  let inner_rectangle_area := inner_rectangle_side^2
  half_larger_square_area = 80 → inner_rectangle_area = 40 := by
  sorry

end area_of_inner_rectangle_l2615_261522


namespace chocolates_distribution_l2615_261535

theorem chocolates_distribution (total_children boys girls : ℕ) 
  (boys_chocolates girls_chocolates : ℕ) :
  total_children = 120 →
  boys = 60 →
  girls = 60 →
  boys + girls = total_children →
  boys_chocolates = 2 →
  girls_chocolates = 3 →
  boys * boys_chocolates + girls * girls_chocolates = 300 :=
by sorry

end chocolates_distribution_l2615_261535


namespace simplify_trig_expression_l2615_261540

theorem simplify_trig_expression :
  1 / Real.sin (15 * π / 180) - 1 / Real.cos (15 * π / 180) = 2 * Real.sqrt 2 := by
  sorry

end simplify_trig_expression_l2615_261540


namespace log_product_equals_one_l2615_261516

theorem log_product_equals_one : 
  Real.log 3 / Real.log 2 * (Real.log 4 / Real.log 9) = 1 := by
  sorry

end log_product_equals_one_l2615_261516


namespace tom_car_distribution_l2615_261510

theorem tom_car_distribution (total_packages : ℕ) (cars_per_package : ℕ) (num_nephews : ℕ) (cars_remaining : ℕ) :
  total_packages = 10 →
  cars_per_package = 5 →
  num_nephews = 2 →
  cars_remaining = 30 →
  (total_packages * cars_per_package - cars_remaining) / (num_nephews * (total_packages * cars_per_package)) = 1 / 5 := by
  sorry

end tom_car_distribution_l2615_261510


namespace triangle_area_l2615_261552

/-- Given a triangle with perimeter 60 cm and inradius 2.5 cm, its area is 75 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 60 → inradius = 2.5 → area = perimeter / 2 * inradius → area = 75 := by
  sorry

end triangle_area_l2615_261552


namespace soda_survey_result_l2615_261597

/-- Given a survey of 520 people and a central angle of 220° for the "Soda" sector,
    prove that 317 people chose "Soda". -/
theorem soda_survey_result (total_surveyed : ℕ) (soda_angle : ℝ) :
  total_surveyed = 520 →
  soda_angle = 220 →
  ∃ (soda_count : ℕ),
    soda_count = 317 ∧
    (soda_count : ℝ) / total_surveyed * 360 ≥ soda_angle - 0.5 ∧
    (soda_count : ℝ) / total_surveyed * 360 < soda_angle + 0.5 :=
by sorry


end soda_survey_result_l2615_261597


namespace inverse_proportion_problem_l2615_261503

/-- Two real numbers are inversely proportional -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h_prop : InverselyProportional x₁ y₁)
  (h_init : x₁ = 40 ∧ y₁ = 8)
  (h_final : y₂ = 10) :
  x₂ = 32 ∧ InverselyProportional x₂ y₂ :=
sorry

end inverse_proportion_problem_l2615_261503


namespace seventh_term_of_geometric_sequence_l2615_261502

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem seventh_term_of_geometric_sequence :
  let a₁ : ℚ := 3
  let a₂ : ℚ := -1/2
  let r : ℚ := a₂ / a₁
  geometric_sequence a₁ r 7 = 1/15552 := by
sorry

end seventh_term_of_geometric_sequence_l2615_261502


namespace puzzle_solution_l2615_261582

def puzzle_problem (total_pieces : ℕ) (num_sons : ℕ) (reyn_pieces : ℕ) : ℕ :=
  let pieces_per_son := total_pieces / num_sons
  let rhys_pieces := 2 * reyn_pieces
  let rory_pieces := 3 * reyn_pieces
  let placed_pieces := reyn_pieces + rhys_pieces + rory_pieces
  total_pieces - placed_pieces

theorem puzzle_solution :
  puzzle_problem 300 3 25 = 150 := by
  sorry

end puzzle_solution_l2615_261582


namespace hyperbola_asymptote_slope_l2615_261596

theorem hyperbola_asymptote_slope (m : ℝ) : m > 0 →
  (∀ x y : ℝ, x^2/16 - y^2/25 = 1 ↔ (y = m*x ∨ y = -m*x)) →
  m = 5/4 := by
sorry

end hyperbola_asymptote_slope_l2615_261596


namespace ageOfReplacedManIs42_l2615_261594

/-- Given a group of 6 men where:
    - The average age increases by 3 years when two women replace two men
    - One of the men is 26 years old
    - The average age of the women is 34
    This function calculates the age of the other man who was replaced. -/
def ageOfReplacedMan (averageIncrease : ℕ) (knownManAge : ℕ) (womenAverageAge : ℕ) : ℕ :=
  2 * womenAverageAge - knownManAge

/-- Theorem stating that under the given conditions, 
    the age of the other replaced man is 42 years. -/
theorem ageOfReplacedManIs42 :
  ageOfReplacedMan 3 26 34 = 42 := by
  sorry


end ageOfReplacedManIs42_l2615_261594


namespace parallel_lines_a_value_l2615_261504

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x - y + b₁ = 0 ↔ m₂ * x - y + b₂ = 0) ↔ m₁ = m₂

/-- The value of a for which ax-2y+2=0 is parallel to x+(a-3)y+1=0 -/
theorem parallel_lines_a_value :
  ∃ a : ℝ, (∀ x y : ℝ, a * x - 2 * y + 2 = 0 ↔ x + (a - 3) * y + 1 = 0) → a = 1 := by
  sorry


end parallel_lines_a_value_l2615_261504


namespace fruit_punch_ratio_l2615_261565

theorem fruit_punch_ratio (orange_punch apple_juice cherry_punch total_punch : ℝ) : 
  orange_punch = 4.5 →
  apple_juice = cherry_punch - 1.5 →
  total_punch = orange_punch + cherry_punch + apple_juice →
  total_punch = 21 →
  cherry_punch / orange_punch = 2 := by
sorry

end fruit_punch_ratio_l2615_261565


namespace smallest_integer_y_l2615_261566

theorem smallest_integer_y (y : ℤ) : (7 - 3 * y < 20) ∧ (∀ z : ℤ, z < y → ¬(7 - 3 * z < 20)) → y = -4 := by
  sorry

end smallest_integer_y_l2615_261566


namespace original_bottle_caps_l2615_261576

theorem original_bottle_caps (removed : ℕ) (left : ℕ) (original : ℕ) : 
  removed = 47 → left = 40 → original = removed + left → original = 87 := by
  sorry

end original_bottle_caps_l2615_261576


namespace heroes_on_back_l2615_261520

/-- The number of heroes Will drew on the front of the paper -/
def heroes_on_front : ℕ := 2

/-- The total number of heroes Will drew -/
def total_heroes : ℕ := 9

/-- Theorem: The number of heroes Will drew on the back of the paper is 7 -/
theorem heroes_on_back : total_heroes - heroes_on_front = 7 := by
  sorry

end heroes_on_back_l2615_261520


namespace intersection_line_slope_l2615_261559

/-- The slope of the line passing through the intersection points of two circles -/
theorem intersection_line_slope (x y : ℝ) : 
  (x^2 + y^2 - 6*x + 4*y - 8 = 0) ∧ 
  (x^2 + y^2 - 10*x + 18*y + 40 = 0) →
  (∃ m : ℚ, m = 2/7 ∧ 
   ∀ (x₁ y₁ x₂ y₂ : ℝ), 
   (x₁^2 + y₁^2 - 6*x₁ + 4*y₁ - 8 = 0) ∧ 
   (x₁^2 + y₁^2 - 10*x₁ + 18*y₁ + 40 = 0) ∧
   (x₂^2 + y₂^2 - 6*x₂ + 4*y₂ - 8 = 0) ∧ 
   (x₂^2 + y₂^2 - 10*x₂ + 18*y₂ + 40 = 0) ∧
   x₁ ≠ x₂ →
   m = (y₂ - y₁) / (x₂ - x₁)) :=
sorry

end intersection_line_slope_l2615_261559


namespace remainder_8347_div_9_l2615_261511

theorem remainder_8347_div_9 : 8347 % 9 = 4 := by
  sorry

end remainder_8347_div_9_l2615_261511


namespace gas_measurement_l2615_261545

/-- Represents the ratio of inches to liters per minute for liquid -/
def liquid_ratio : ℚ := 2.5 / 60

/-- Represents the movement ratio of gas compared to liquid -/
def gas_movement_ratio : ℚ := 1 / 2

/-- Represents the amount of gas that passed through the rotameter in liters -/
def gas_volume : ℚ := 192

/-- Calculates the number of inches measured for the gas phase -/
def gas_inches : ℚ := (gas_volume * liquid_ratio) / gas_movement_ratio

/-- Theorem stating that the number of inches measured for the gas phase is 4 -/
theorem gas_measurement :
  gas_inches = 4 := by sorry

end gas_measurement_l2615_261545


namespace smallest_winning_number_l2615_261557

theorem smallest_winning_number : ∃ N : ℕ, 
  (N = 6) ∧ 
  (8 * N + 450 < 500) ∧ 
  (N ≤ 499) ∧ 
  (∀ m : ℕ, m < N → (8 * m + 450 ≥ 500) ∨ m > 499) :=
by sorry

end smallest_winning_number_l2615_261557


namespace fraction_evaluation_l2615_261521

theorem fraction_evaluation : (15 : ℚ) / 45 - 2 / 9 + 1 / 4 * 8 / 3 = 7 / 9 := by
  sorry

end fraction_evaluation_l2615_261521


namespace boat_upstream_downstream_ratio_l2615_261541

/-- Given a boat with speed in still water and a stream with its own speed,
    prove that the ratio of time taken to row upstream to downstream is 2:1 -/
theorem boat_upstream_downstream_ratio
  (boat_speed : ℝ) (stream_speed : ℝ)
  (h1 : boat_speed = 54)
  (h2 : stream_speed = 18) :
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
  sorry

#check boat_upstream_downstream_ratio

end boat_upstream_downstream_ratio_l2615_261541


namespace k_squared_test_probability_two_males_l2615_261591

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![30, 45],
    ![15, 10]]

-- Define the total number of people surveyed
def total_surveyed : ℕ := 100

-- Define the K² formula
def k_squared (a b c d : ℕ) : ℚ :=
  (total_surveyed * (a * d - b * c)^2 : ℚ) /
  ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 95% confidence
def critical_value : ℚ := 3841 / 1000

-- Theorem for the K² test
theorem k_squared_test :
  k_squared (contingency_table 0 0) (contingency_table 0 1)
            (contingency_table 1 0) (contingency_table 1 1) < critical_value := by
  sorry

-- Define the number of healthy living people
def healthy_living : ℕ := 45

-- Define the number of healthy living males
def healthy_males : ℕ := 30

-- Theorem for the probability of selecting two males
theorem probability_two_males :
  (Nat.choose healthy_males 2 : ℚ) / (Nat.choose healthy_living 2) = 2 / 5 := by
  sorry

end k_squared_test_probability_two_males_l2615_261591


namespace max_value_difference_l2615_261529

noncomputable def f (x : ℝ) := x^3 - 3*x^2 - x + 1

theorem max_value_difference (x₀ m : ℝ) : 
  (∀ x, f x ≤ f x₀) →  -- f attains maximum at x₀
  m ≠ x₀ →             -- m is not equal to x₀
  f x₀ = f m →         -- f(x₀) = f(m)
  |m - x₀| = 2 * Real.sqrt 3 := by
sorry

end max_value_difference_l2615_261529


namespace distance_to_point_l2615_261517

theorem distance_to_point : Real.sqrt ((8 - 0)^2 + (-15 - 0)^2) = 17 := by sorry

end distance_to_point_l2615_261517


namespace gas_diffusion_rate_and_molar_mass_l2615_261548

theorem gas_diffusion_rate_and_molar_mass 
  (r_unknown r_O2 : ℝ) 
  (M_unknown M_O2 : ℝ) 
  (h1 : r_unknown / r_O2 = 1 / 3) 
  (h2 : r_unknown / r_O2 = Real.sqrt (M_O2 / M_unknown)) :
  M_unknown = 9 * M_O2 := by
  sorry

end gas_diffusion_rate_and_molar_mass_l2615_261548


namespace barber_total_loss_l2615_261562

/-- Represents the barber's financial transactions and losses --/
def barber_loss : ℕ → Prop :=
  fun loss =>
    ∃ (haircut_cost change_given flower_shop_exchange bakery_exchange counterfeit_50 counterfeit_10 replacement_50 replacement_10 : ℕ),
      haircut_cost = 25 ∧
      change_given = 25 ∧
      flower_shop_exchange = 50 ∧
      bakery_exchange = 10 ∧
      counterfeit_50 = 50 ∧
      counterfeit_10 = 10 ∧
      replacement_50 = 50 ∧
      replacement_10 = 10 ∧
      loss = haircut_cost + change_given + counterfeit_50 + counterfeit_10 + replacement_50 + replacement_10 - flower_shop_exchange

theorem barber_total_loss :
  barber_loss 120 :=
sorry

end barber_total_loss_l2615_261562


namespace tiffany_bags_l2615_261528

/-- The number of bags found the next day -/
def bags_found (initial_bags total_bags : ℕ) : ℕ := total_bags - initial_bags

/-- Proof that Tiffany found 8 bags the next day -/
theorem tiffany_bags : bags_found 4 12 = 8 := by
  sorry

end tiffany_bags_l2615_261528


namespace average_of_abc_l2615_261574

theorem average_of_abc (A B C : ℝ) 
  (eq1 : 1001 * C - 2002 * A = 4004)
  (eq2 : 1001 * B + 3003 * A = 5005) : 
  (A + B + C) / 3 = 3 := by
  sorry

end average_of_abc_l2615_261574


namespace probability_is_seventy_percent_l2615_261571

/-- Represents a frequency interval with its lower bound, upper bound, and frequency count -/
structure FrequencyInterval where
  lower : ℝ
  upper : ℝ
  frequency : ℕ

/-- The sample data -/
def sample : List FrequencyInterval := [
  ⟨10, 20, 2⟩,
  ⟨20, 30, 3⟩,
  ⟨30, 40, 4⟩,
  ⟨40, 50, 5⟩,
  ⟨50, 60, 4⟩,
  ⟨60, 70, 2⟩
]

/-- The total sample size -/
def sampleSize : ℕ := 20

/-- The upper bound of the interval in question -/
def intervalUpperBound : ℝ := 50

/-- Calculates the probability of the sample data falling within (-∞, intervalUpperBound) -/
def probabilityWithinInterval (sample : List FrequencyInterval) (sampleSize : ℕ) (intervalUpperBound : ℝ) : ℚ :=
  sorry

/-- Theorem stating that the probability of the sample data falling within (-∞, 50) is 70% -/
theorem probability_is_seventy_percent :
  probabilityWithinInterval sample sampleSize intervalUpperBound = 7/10 := by
  sorry

end probability_is_seventy_percent_l2615_261571


namespace unique_a_with_integer_solutions_l2615_261547

theorem unique_a_with_integer_solutions : 
  ∃! a : ℕ+, (a : ℝ) ≤ 100 ∧ 
  ∃ x y : ℤ, x ≠ y ∧ 
  (x : ℝ)^2 + (2 * (a : ℝ) - 3) * (x : ℝ) + ((a : ℝ) - 1)^2 = 0 ∧
  (y : ℝ)^2 + (2 * (a : ℝ) - 3) * (y : ℝ) + ((a : ℝ) - 1)^2 = 0 :=
sorry

end unique_a_with_integer_solutions_l2615_261547


namespace highest_score_for_given_stats_l2615_261558

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  totalInnings : ℕ
  overallAverage : ℚ
  scoreDifference : ℕ
  averageExcludingExtremes : ℚ

/-- Calculates the highest score given a batsman's statistics -/
def highestScore (stats : BatsmanStats) : ℕ :=
  sorry

/-- Theorem stating the highest score for the given conditions -/
theorem highest_score_for_given_stats :
  let stats : BatsmanStats := {
    totalInnings := 46,
    overallAverage := 59,
    scoreDifference := 150,
    averageExcludingExtremes := 58
  }
  highestScore stats = 151 := by
  sorry

end highest_score_for_given_stats_l2615_261558


namespace power_of_three_expression_l2615_261500

theorem power_of_three_expression : 3^(1+2+3) - (3^1 + 3^2 + 3^4) = 636 := by
  sorry

end power_of_three_expression_l2615_261500


namespace max_value_of_f_on_interval_l2615_261501

def f (x : ℝ) := x^3 - 12*x

theorem max_value_of_f_on_interval :
  ∃ (M : ℝ), M = 16 ∧ ∀ x ∈ Set.Icc (-3) 3, f x ≤ M :=
sorry

end max_value_of_f_on_interval_l2615_261501


namespace triangular_difference_2015_l2615_261569

theorem triangular_difference_2015 : ∃ (n k : ℕ), 
  1000 ≤ n * (n + 1) / 2 ∧ n * (n + 1) / 2 < 10000 ∧
  1000 ≤ k * (k + 1) / 2 ∧ k * (k + 1) / 2 < 10000 ∧
  n * (n + 1) / 2 - k * (k + 1) / 2 = 2015 :=
by sorry


end triangular_difference_2015_l2615_261569


namespace log_base_six_two_point_five_l2615_261531

theorem log_base_six_two_point_five (x : ℝ) :
  (Real.log x) / (Real.log 6) = 2.5 → x = 36 * Real.sqrt 6 := by
  sorry

end log_base_six_two_point_five_l2615_261531


namespace ngon_triangle_partition_l2615_261542

/-- A function that checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A structure representing an n-gon -/
structure Ngon (n : ℕ) :=
  (vertices : Fin n → ℝ × ℝ)
  (is_convex : sorry)  -- Additional property to ensure the n-gon is convex

/-- 
Given an n-gon and three vertex indices, return the lengths of the three parts 
of the boundary divided by these vertices
-/
def boundary_parts (n : ℕ) (poly : Ngon n) (i j k : Fin n) : ℝ × ℝ × ℝ := sorry

/-- The main theorem -/
theorem ngon_triangle_partition (n : ℕ) (h : n ≥ 3 ∧ n ≠ 4) :
  ∀ (poly : Ngon n), ∃ (i j k : Fin n),
    let (a, b, c) := boundary_parts n poly i j k
    can_form_triangle a b c :=
  sorry

end ngon_triangle_partition_l2615_261542


namespace pharmacy_loss_l2615_261578

theorem pharmacy_loss (a b : ℝ) (h : a < b) : 
  100 * ((a + b) / 2) - (41 * a + 59 * b) < 0 := by
  sorry

#check pharmacy_loss

end pharmacy_loss_l2615_261578


namespace water_depth_is_60_feet_l2615_261550

def ron_height : ℝ := 12

def water_depth : ℝ := 5 * ron_height

theorem water_depth_is_60_feet : water_depth = 60 := by
  sorry

end water_depth_is_60_feet_l2615_261550


namespace lcm_hcf_problem_l2615_261590

theorem lcm_hcf_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 →
  Nat.gcd a b = 30 →
  b = 330 →
  a = 210 := by
sorry

end lcm_hcf_problem_l2615_261590


namespace student_skills_l2615_261593

theorem student_skills (total : ℕ) (chess_unable : ℕ) (puzzle_unable : ℕ) (code_unable : ℕ) :
  total = 120 →
  chess_unable = 50 →
  puzzle_unable = 75 →
  code_unable = 40 →
  (∃ (two_skills : ℕ), two_skills = 75 ∧
    two_skills = (total - chess_unable) + (total - puzzle_unable) + (total - code_unable) - total) :=
by sorry

end student_skills_l2615_261593


namespace expression_factorization_l2615_261506

theorem expression_factorization (x : ℝ) : 
  (16 * x^7 + 81 * x^4 - 9) - (4 * x^7 - 18 * x^4 + 3) = 3 * (4 * x^7 + 33 * x^4 - 4) := by
  sorry

end expression_factorization_l2615_261506


namespace min_value_expression_l2615_261526

theorem min_value_expression (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_one : x + y + z + w = 1) (x_eq_y : x = y) :
  ∀ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d = 1 ∧ a = b →
  (a + b + c) / (a * b * c * d) ≥ (x + y + z) / (x * y * z * w) ∧
  (x + y + z) / (x * y * z * w) ≥ 1024 :=
by sorry

end min_value_expression_l2615_261526


namespace cistern_fill_time_l2615_261546

theorem cistern_fill_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) : 
  empty_rate = 1 / 9 →
  combined_fill_time = 7 / 3 →
  1 / fill_time - empty_rate = 1 / combined_fill_time →
  fill_time = 63 / 34 := by
sorry

end cistern_fill_time_l2615_261546


namespace line_intersection_l2615_261580

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 7*x + 12

-- Define the linear function
def g (m b x : ℝ) : ℝ := m*x + b

-- Define the distance between two points on the same vertical line
def distance (k m b : ℝ) : ℝ := |f k - g m b k|

theorem line_intersection (m b : ℝ) : 
  (∃ k, distance k m b = 8) ∧ 
  g m b 2 = 7 ∧ 
  b ≠ 0 →
  (m = 1 ∧ b = 5) ∨ (m = 5 ∧ b = -3) :=
sorry

end line_intersection_l2615_261580


namespace factorization_of_quadratic_l2615_261583

theorem factorization_of_quadratic (a : ℝ) : a^2 + 3*a = a*(a + 3) := by
  sorry

end factorization_of_quadratic_l2615_261583


namespace min_value_z_l2615_261538

theorem min_value_z (x y : ℝ) : 3 * x^2 + 5 * y^2 + 12 * x - 10 * y + 40 ≥ 23 := by
  sorry

end min_value_z_l2615_261538


namespace candy_boxes_problem_l2615_261560

theorem candy_boxes_problem (a b c : ℕ) : 
  a = b + c - 8 → 
  b = a + c - 12 → 
  c = 10 := by
  sorry

end candy_boxes_problem_l2615_261560
