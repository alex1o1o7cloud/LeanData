import Mathlib

namespace NUMINAMATH_CALUDE_equal_sum_of_intervals_l3315_331568

-- Define the function f on the interval [a, b]
variable (f : ℝ → ℝ)
variable (a b : ℝ)

-- Define the property of f being continuous on [a, b]
def IsContinuousOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → ContinuousAt f x

-- Define the property of f(a) = f(b)
def HasEqualEndpoints (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f a = f b

-- Define the sum of lengths of intervals where f is increasing
def SumOfIncreasingIntervals (f : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

-- Define the sum of lengths of intervals where f is decreasing
def SumOfDecreasingIntervals (f : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

-- State the theorem
theorem equal_sum_of_intervals
  (h1 : IsContinuousOn f a b)
  (h2 : HasEqualEndpoints f a b)
  (h3 : a ≤ b) :
  SumOfIncreasingIntervals f a b = SumOfDecreasingIntervals f a b :=
by sorry

end NUMINAMATH_CALUDE_equal_sum_of_intervals_l3315_331568


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3315_331533

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := (n + 1) / 2

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 1 / (n * a n)

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := (2 * n) / (n + 1)

theorem arithmetic_sequence_problem :
  (a 7 = 4) ∧ (a 19 = 2 * a 9) ∧
  (∀ n : ℕ, n > 0 → b n = 1 / (n * a n)) ∧
  (∀ n : ℕ, n > 0 → S n = (2 * n) / (n + 1)) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3315_331533


namespace NUMINAMATH_CALUDE_minimal_radius_inscribed_triangle_l3315_331559

theorem minimal_radius_inscribed_triangle (a b : ℝ) (ha : a = 3) (hb : b = 4) :
  let c := Real.sqrt (a^2 + b^2)
  let R := c / 2
  R = (5 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_minimal_radius_inscribed_triangle_l3315_331559


namespace NUMINAMATH_CALUDE_f_properties_l3315_331530

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - 2 * Real.sqrt 3 * Real.sin x * Real.sin (x - Real.pi / 2)

def is_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem f_properties :
  ∃ (T : ℝ) (A B : ℝ) (a b c : ℝ),
    T > 0 ∧
    is_period T f ∧
    (∀ T' > 0, is_period T' f → T ≤ T') ∧
    f (A / 2) = 3 ∧
    (1 / 4 * (a^2 + c^2 - b^2) = 1 / 2 * a * c * Real.sin B) →
    (T = Real.pi ∧ b / a = Real.sqrt 6 / 3) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3315_331530


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l3315_331593

/-- The cost ratio of a muffin to a banana given Susie's and Nathan's purchases -/
theorem muffin_banana_cost_ratio :
  ∀ (muffin_cost banana_cost : ℝ),
  muffin_cost > 0 →
  banana_cost > 0 →
  5 * muffin_cost + 4 * banana_cost > 0 →
  4 * (5 * muffin_cost + 4 * banana_cost) = 4 * muffin_cost + 12 * banana_cost →
  muffin_cost / banana_cost = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l3315_331593


namespace NUMINAMATH_CALUDE_ball_bounce_problem_l3315_331535

def bounce_height (k : ℕ) : ℝ :=
  1500 * (0.40 ^ k) * (0.95 ^ (k * (k - 1) / 2))

def is_first_bounce_below_two (k : ℕ) : Prop :=
  bounce_height k < 2 ∧ ∀ j : ℕ, j < k → bounce_height j ≥ 2

theorem ball_bounce_problem :
  ∃ k : ℕ, is_first_bounce_below_two k ∧ k = 6 :=
sorry

end NUMINAMATH_CALUDE_ball_bounce_problem_l3315_331535


namespace NUMINAMATH_CALUDE_AtLeastOneSolution_l3315_331513

-- Define the property that the function f should satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f (y^2)) = x^2 + y

-- Theorem statement
theorem AtLeastOneSolution : ∃ f : ℝ → ℝ, SatisfiesProperty f := by
  sorry

end NUMINAMATH_CALUDE_AtLeastOneSolution_l3315_331513


namespace NUMINAMATH_CALUDE_system_solution_l3315_331572

theorem system_solution (x y u v : ℝ) : 
  (x = -2 ∧ y = 2 ∧ u = 2 ∧ v = -2) →
  (x + 7*y + 3*v + 5*u = 16) ∧
  (8*x + 4*y + 6*v + 2*u = -16) ∧
  (2*x + 6*y + 4*v + 8*u = 16) ∧
  (5*x + 3*y + 7*v + u = -16) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3315_331572


namespace NUMINAMATH_CALUDE_variable_conditions_l3315_331521

theorem variable_conditions (a b c d e : ℝ) 
  (h : (a + b + e) / (b + c) = (c + d + e) / (d + a)) : 
  a = c ∨ a + b + c + d + e = 0 := by sorry

end NUMINAMATH_CALUDE_variable_conditions_l3315_331521


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3315_331520

/-- The quadratic inequality mx^2 - mx - 1 < 0 has all real numbers as its solution set -/
def all_reals_solution (m : ℝ) : Prop :=
  ∀ x : ℝ, m * x^2 - m * x - 1 < 0

/-- The range of values for m -/
def m_range (m : ℝ) : Prop := -4 < m ∧ m < 0

/-- Theorem stating that if the quadratic inequality has all real numbers as its solution set,
    then m is in the range (-4, 0) -/
theorem quadratic_inequality_range :
  ∀ m : ℝ, all_reals_solution m → m_range m :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3315_331520


namespace NUMINAMATH_CALUDE_negative_x_is_directly_proportional_l3315_331506

/-- A function f : ℝ → ℝ is directly proportional if there exists a constant k such that f x = k * x for all x -/
def DirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function f(x) = -x is directly proportional -/
theorem negative_x_is_directly_proportional :
  DirectlyProportional (fun x => -x) := by
  sorry

#check negative_x_is_directly_proportional

end NUMINAMATH_CALUDE_negative_x_is_directly_proportional_l3315_331506


namespace NUMINAMATH_CALUDE_jennifer_garden_max_area_l3315_331598

/-- Represents a rectangular garden with integer side lengths. -/
structure RectangularGarden where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangular garden. -/
def perimeter (g : RectangularGarden) : ℕ := 2 * (g.length + g.width)

/-- Calculates the area of a rectangular garden. -/
def area (g : RectangularGarden) : ℕ := g.length * g.width

/-- Theorem stating the maximum area of Jennifer's garden. -/
theorem jennifer_garden_max_area :
  ∃ (g : RectangularGarden),
    g.length = 30 ∧
    perimeter g = 160 ∧
    (∀ (h : RectangularGarden), h.length = 30 ∧ perimeter h = 160 → area h ≤ area g) ∧
    area g = 1500 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_garden_max_area_l3315_331598


namespace NUMINAMATH_CALUDE_area_triangle_QDA_l3315_331582

/-- The area of triangle QDA given the coordinates of points D, Q, A, and B -/
theorem area_triangle_QDA (p : ℝ) : 
  let D : ℝ × ℝ := (0, p + 3)
  let Q : ℝ × ℝ := (0, 15)
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let area := (1/2) * (A.1 - Q.1) * (Q.2 - D.2)
  area = 18 - (3/2) * p :=
by
  sorry

end NUMINAMATH_CALUDE_area_triangle_QDA_l3315_331582


namespace NUMINAMATH_CALUDE_nested_average_equals_29_18_l3315_331566

def average_2 (a b : ℚ) : ℚ := (a + b) / 2

def average_3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem nested_average_equals_29_18 : 
  average_3 (average_3 (-1) 2 3) (average_2 2 3) 1 = 29 / 18 := by
  sorry

end NUMINAMATH_CALUDE_nested_average_equals_29_18_l3315_331566


namespace NUMINAMATH_CALUDE_stratified_sampling_appropriate_l3315_331514

/-- Represents a group of teachers -/
structure TeacherGroup where
  size : ℕ

/-- Represents the entire population of teachers -/
structure TeacherPopulation where
  senior : TeacherGroup
  intermediate : TeacherGroup
  junior : TeacherGroup

/-- Represents a sampling method -/
inductive SamplingMethod
  | Stratified
  | Simple
  | Cluster

/-- States that stratified sampling is appropriate for a population with distinct groups -/
theorem stratified_sampling_appropriate (population : TeacherPopulation) (sample_size : ℕ) :
  SamplingMethod.Stratified = 
    (fun (pop : TeacherPopulation) (s : ℕ) => 
      if pop.senior.size ≠ pop.intermediate.size ∧ 
         pop.senior.size ≠ pop.junior.size ∧ 
         pop.intermediate.size ≠ pop.junior.size
      then SamplingMethod.Stratified
      else SamplingMethod.Simple) population sample_size :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_appropriate_l3315_331514


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l3315_331563

theorem perpendicular_lines_b_value (b : ℝ) : 
  (∀ x y : ℝ, 3 * y + 2 * x - 6 = 0 ∨ 4 * y + b * x + 8 = 0) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    3 * y₁ + 2 * x₁ - 6 = 0 ∧ 
    4 * y₂ + b * x₂ + 8 = 0 ∧
    (y₂ - y₁) * (x₂ - x₁) = 0) →
  b = -6 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l3315_331563


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3315_331596

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B : Set ℝ := {x | x > Real.sqrt 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | Real.sqrt 3 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3315_331596


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3315_331599

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (3 * I - 5) / (2 + I)
  Complex.im z = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3315_331599


namespace NUMINAMATH_CALUDE_triangle_problem_l3315_331503

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (c - Real.sqrt 3 * b * Real.sin A = (a^2 + c^2 - b^2) / (2 * c) - b) →
  (A = π / 3) ∧
  ((b = c / 4) → 
   (a * 2 * Real.sqrt 3 = b * c * Real.sin A) → 
   (a = 13)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3315_331503


namespace NUMINAMATH_CALUDE_complement_of_A_l3315_331550

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}

theorem complement_of_A : Aᶜ = {3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l3315_331550


namespace NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l3315_331579

theorem negation_of_all_squares_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l3315_331579


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l3315_331583

/-- The number of valid combinations for a magical elixir --/
def validCombinations (herbs : ℕ) (gems : ℕ) (incompatibleGems : ℕ) (incompatibleHerbs : ℕ) : ℕ :=
  herbs * gems - incompatibleGems * incompatibleHerbs

/-- Theorem: The wizard can prepare his elixir in 17 different ways --/
theorem wizard_elixir_combinations : 
  validCombinations 4 5 1 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l3315_331583


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_6_l3315_331580

/-- The area of a circle with diameter 6 meters is 9π square meters. -/
theorem circle_area_with_diameter_6 :
  let diameter : ℝ := 6
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius^2
  area = 9 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_6_l3315_331580


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_distance_to_8_15_l3315_331574

theorem distance_from_origin_to_point : ℝ → ℝ → ℝ
  | x, y => Real.sqrt (x^2 + y^2)

theorem distance_to_8_15 :
  distance_from_origin_to_point 8 15 = 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_distance_to_8_15_l3315_331574


namespace NUMINAMATH_CALUDE_rental_car_cost_sharing_l3315_331551

theorem rental_car_cost_sharing (n : ℕ) (C : ℝ) (h : n > 1) :
  (C / (n - 1 : ℝ) - C / n = 0.125) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_rental_car_cost_sharing_l3315_331551


namespace NUMINAMATH_CALUDE_divided_triangle_properties_l3315_331531

structure DividedTriangle where
  u : ℝ
  v : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  S : ℝ
  u_pos : u > 0
  v_pos : v > 0
  x_pos : x > 0
  y_pos : y > 0
  z_pos : z > 0
  S_pos : S > 0

theorem divided_triangle_properties (t : DividedTriangle) :
  t.u * t.v = t.y * t.z ∧ t.S ≤ (t.x * t.z) / t.y := by
  sorry

end NUMINAMATH_CALUDE_divided_triangle_properties_l3315_331531


namespace NUMINAMATH_CALUDE_correct_seat_increase_l3315_331541

/-- Represents the seating arrangement in a theater --/
structure Theater where
  first_row_seats : ℕ
  last_row_seats : ℕ
  total_seats : ℕ
  seat_increase_per_row : ℕ

/-- Calculates the number of rows in the theater --/
def num_rows (t : Theater) : ℕ :=
  (t.last_row_seats - t.first_row_seats) / t.seat_increase_per_row + 1

/-- Calculates the sum of seats in all rows --/
def sum_of_seats (t : Theater) : ℕ :=
  (num_rows t * (t.first_row_seats + t.last_row_seats)) / 2

/-- Theorem stating the correct seat increase per row --/
theorem correct_seat_increase (t : Theater) 
  (h1 : t.first_row_seats = 12)
  (h2 : t.last_row_seats = 48)
  (h3 : t.total_seats = 570)
  (h4 : sum_of_seats t = t.total_seats) :
  t.seat_increase_per_row = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_seat_increase_l3315_331541


namespace NUMINAMATH_CALUDE_pencil_count_l3315_331529

theorem pencil_count (initial_pencils added_pencils : ℕ) : 
  initial_pencils = 2 → added_pencils = 3 → initial_pencils + added_pencils = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l3315_331529


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l3315_331517

theorem quadratic_solution_range (a b c : ℝ) :
  (a * 0^2 + b * 0 + c = -15) →
  (a * 0.5^2 + b * 0.5 + c = -8.75) →
  (a * 1^2 + b * 1 + c = -2) →
  (a * 1.5^2 + b * 1.5 + c = 5.25) →
  (a * 2^2 + b * 2 + c = 13) →
  ∃ x : ℝ, (1 < x ∧ x < 1.5) ∧ (a * x^2 + b * x + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l3315_331517


namespace NUMINAMATH_CALUDE_triangle_configuration_l3315_331524

theorem triangle_configuration (AB : ℝ) (cosA sinC : ℝ) :
  AB = 30 →
  cosA = 4/5 →
  sinC = 4/5 →
  ∃ (DA DB BC DC : ℝ),
    DA = AB * cosA ∧
    DB ^ 2 = AB ^ 2 - DA ^ 2 ∧
    BC = DB / sinC ∧
    DC ^ 2 = BC ^ 2 - DB ^ 2 ∧
    DC = 13.5 := by sorry

end NUMINAMATH_CALUDE_triangle_configuration_l3315_331524


namespace NUMINAMATH_CALUDE_circle_equation_implies_sum_l3315_331597

theorem circle_equation_implies_sum (x y : ℝ) :
  x^2 + y^2 - 2*x + 4*y + 5 = 0 → 2*x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_implies_sum_l3315_331597


namespace NUMINAMATH_CALUDE_maxwell_walking_speed_l3315_331576

/-- Proves that Maxwell's walking speed is 4 km/h given the problem conditions -/
theorem maxwell_walking_speed
  (total_distance : ℝ)
  (brad_speed : ℝ)
  (maxwell_distance : ℝ)
  (h1 : total_distance = 50)
  (h2 : brad_speed = 6)
  (h3 : maxwell_distance = 20)
  : ∃ (maxwell_speed : ℝ), maxwell_speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_maxwell_walking_speed_l3315_331576


namespace NUMINAMATH_CALUDE_inheritance_tax_equation_l3315_331585

/-- The inheritance amount in dollars -/
def inheritance : ℝ := 35300

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The state tax rate as a decimal -/
def state_tax_rate : ℝ := 0.12

/-- The total tax paid in dollars -/
def total_tax_paid : ℝ := 12000

theorem inheritance_tax_equation :
  federal_tax_rate * inheritance + 
  state_tax_rate * (inheritance - federal_tax_rate * inheritance) = 
  total_tax_paid := by sorry

end NUMINAMATH_CALUDE_inheritance_tax_equation_l3315_331585


namespace NUMINAMATH_CALUDE_sign_of_c_l3315_331552

theorem sign_of_c (a b c : ℝ) 
  (h1 : a * b / c < 0) 
  (h2 : a * b < 0) : 
  c > 0 := by sorry

end NUMINAMATH_CALUDE_sign_of_c_l3315_331552


namespace NUMINAMATH_CALUDE_divisible_by_24_l3315_331534

theorem divisible_by_24 (n : ℕ+) : ∃ k : ℤ, (n : ℤ) * (n + 2) * (5 * n - 1) * (5 * n + 1) = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l3315_331534


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3315_331504

/-- Represents the number of employees in each title category -/
structure EmployeeCount where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  general : ℕ

/-- Represents the number of employees in each title category for a sample -/
structure SampleCount where
  senior : ℕ
  intermediate : ℕ
  general : ℕ

/-- Checks if the sample count is proportional to the employee count -/
def is_proportional_sample (ec : EmployeeCount) (sc : SampleCount) (sample_size : ℕ) : Prop :=
  sc.senior = (ec.senior * sample_size) / ec.total ∧
  sc.intermediate = (ec.intermediate * sample_size) / ec.total ∧
  sc.general = (ec.general * sample_size) / ec.total

theorem stratified_sampling_theorem (ec : EmployeeCount) (sc : SampleCount) (sample_size : ℕ) :
  ec.total = 150 ∧ ec.senior = 15 ∧ ec.intermediate = 45 ∧ ec.general = 90 ∧
  sample_size = 30 ∧
  is_proportional_sample ec sc sample_size →
  sc.senior = 3 ∧ sc.intermediate = 9 ∧ sc.general = 18 :=
by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3315_331504


namespace NUMINAMATH_CALUDE_count_divisible_integers_l3315_331501

theorem count_divisible_integers :
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ (1764 : ℤ) ∣ (m^2 - 4)) ∧
    (∀ m : ℕ, m > 0 ∧ (1764 : ℤ) ∣ (m^2 - 4) → m ∈ S) ∧
    Finset.card S = 3 :=
by sorry

end NUMINAMATH_CALUDE_count_divisible_integers_l3315_331501


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l3315_331539

/-- Represents a rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  total_area : ℝ
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  sum_of_areas : total_area = area1 + area2 + area3 + area4

/-- Theorem: Given a divided rectangle with specific areas, the fourth area is 28 -/
theorem fourth_rectangle_area
  (rect : DividedRectangle)
  (h1 : rect.total_area = 100)
  (h2 : rect.area1 = 24)
  (h3 : rect.area2 = 30)
  (h4 : rect.area3 = 18) :
  rect.area4 = 28 := by
  sorry


end NUMINAMATH_CALUDE_fourth_rectangle_area_l3315_331539


namespace NUMINAMATH_CALUDE_three_color_cubes_l3315_331553

theorem three_color_cubes (total red blue green : ℕ) 
  (h_total : total = 100)
  (h_red : red = 80)
  (h_blue : blue = 85)
  (h_green : green = 75)
  (h_red_le : red ≤ total)
  (h_blue_le : blue ≤ total)
  (h_green_le : green ≤ total) :
  ∃ n : ℕ, 40 ≤ n ∧ n ≤ 75 ∧ n = total - ((total - red) + (total - blue) + (total - green)) :=
sorry

end NUMINAMATH_CALUDE_three_color_cubes_l3315_331553


namespace NUMINAMATH_CALUDE_t_range_theorem_max_radius_theorem_l3315_331564

-- Define the circle equation
def circle_equation (x y t : ℝ) : Prop :=
  x^2 + y^2 - 2*(t+3)*x + 2*(1-4*t^2)*y + 16*t^4 + 9 = 0

-- Define the range of t
def t_range (t : ℝ) : Prop :=
  -1/7 < t ∧ t < 1

-- Define the radius squared as a function of t
def radius_squared (t : ℝ) : ℝ :=
  -7*t^2 + 6*t + 1

-- Theorem for the range of t
theorem t_range_theorem :
  ∀ t : ℝ, (∃ x y : ℝ, circle_equation x y t) ↔ t_range t :=
sorry

-- Theorem for the maximum radius
theorem max_radius_theorem :
  ∃ t : ℝ, t_range t ∧ 
    ∀ t' : ℝ, t_range t' → radius_squared t ≥ radius_squared t' ∧
    t = 3/7 :=
sorry

end NUMINAMATH_CALUDE_t_range_theorem_max_radius_theorem_l3315_331564


namespace NUMINAMATH_CALUDE_smallest_divisible_by_hundred_threes_l3315_331527

/-- A number consisting of n ones -/
def a (n : ℕ) : ℕ := (10^n - 1) / 9

/-- A number consisting of 100 threes -/
def hundred_threes : ℕ := a 100 * 37

theorem smallest_divisible_by_hundred_threes :
  ∀ k : ℕ, k < 300 → ¬(hundred_threes ∣ a k) ∧ (hundred_threes ∣ a 300) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_hundred_threes_l3315_331527


namespace NUMINAMATH_CALUDE_fraction_multiplication_result_l3315_331546

theorem fraction_multiplication_result : (3 / 4 : ℚ) * (1 / 2 : ℚ) * (2 / 5 : ℚ) * 5100 = 765 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_result_l3315_331546


namespace NUMINAMATH_CALUDE_fish_population_calculation_l3315_331587

/-- Calculates the number of fish in a pond on April 1 based on sampling data --/
theorem fish_population_calculation (tagged_april : ℕ) (sample_august : ℕ) (tagged_in_sample : ℕ)
  (death_rate : ℚ) (birth_rate : ℚ) :
  tagged_april = 80 →
  sample_august = 100 →
  tagged_in_sample = 4 →
  death_rate = 30 / 100 →
  birth_rate = 50 / 100 →
  ∃ (total_fish : ℕ), total_fish = 1000 := by
  sorry

#check fish_population_calculation

end NUMINAMATH_CALUDE_fish_population_calculation_l3315_331587


namespace NUMINAMATH_CALUDE_local_minimum_is_global_minimum_l3315_331578

-- Define a triangular lattice
structure TriangularLattice where
  vertices : Set Point
  edges : Set (Point × Point)
  -- Add necessary lattice properties

-- Define a distance function on the lattice
def distance (lattice : TriangularLattice) (p q : Point) : ℝ := sorry

-- Define the sum of distances from a point to n constant points
def sumDistances (lattice : TriangularLattice) (p : Point) (constants : List Point) : ℝ :=
  (constants.map (distance lattice p)).sum

-- Define the neighborhood of a point
def neighbors (lattice : TriangularLattice) (p : Point) : Set Point := sorry

-- Main theorem
theorem local_minimum_is_global_minimum 
  (lattice : TriangularLattice) 
  (constants : List Point) 
  (A : Point) : 
  (∀ n ∈ neighbors lattice A, sumDistances lattice A constants ≤ sumDistances lattice n constants) → 
  (∀ p ∈ lattice.vertices, sumDistances lattice A constants ≤ sumDistances lattice p constants) :=
by sorry

end NUMINAMATH_CALUDE_local_minimum_is_global_minimum_l3315_331578


namespace NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l3315_331542

-- Problem 1
theorem calculate_expression : 
  Real.sqrt ((-4)^2) + 2 * (Real.sqrt 2 - 3) - |-(2 * Real.sqrt 2)| = -2 := by sorry

-- Problem 2
theorem solve_system_of_equations :
  ∃ (x y : ℝ), x / 2 + y / 3 = 4 ∧ x + 2 * y = 16 ∧ x = 4 ∧ y = 6 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l3315_331542


namespace NUMINAMATH_CALUDE_max_moves_21x21_max_moves_20x21_l3315_331536

/-- Represents a rectangular grid with lights -/
structure Grid where
  rows : ℕ
  cols : ℕ
  lights : Set (ℕ × ℕ)

/-- Represents a move in the light-turning game -/
structure Move where
  line : (ℝ × ℝ) → Prop
  affected_lights : Set (ℕ × ℕ)

/-- The maximum number of moves possible for a given grid -/
def max_moves (g : Grid) : ℕ := sorry

/-- Theorem stating the maximum number of moves for a 21×21 square grid -/
theorem max_moves_21x21 :
  ∀ (g : Grid), g.rows = 21 ∧ g.cols = 21 → max_moves g = 3 := by sorry

/-- Theorem stating the maximum number of moves for a 20×21 rectangular grid -/
theorem max_moves_20x21 :
  ∀ (g : Grid), g.rows = 20 ∧ g.cols = 21 → max_moves g = 4 := by sorry

end NUMINAMATH_CALUDE_max_moves_21x21_max_moves_20x21_l3315_331536


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3315_331575

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with S₃ = 9 and S₆ = 36, a₇ + a₈ + a₉ = 45 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
    (h₃ : seq.S 3 = 9) (h₆ : seq.S 6 = 36) : 
    seq.a 7 + seq.a 8 + seq.a 9 = 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3315_331575


namespace NUMINAMATH_CALUDE_original_numerator_proof_l3315_331544

theorem original_numerator_proof (n : ℚ) : 
  (n + 3) / 12 = 2 / 3 → n / 9 = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_original_numerator_proof_l3315_331544


namespace NUMINAMATH_CALUDE_office_episodes_l3315_331548

theorem office_episodes (total_episodes : ℕ) (weeks : ℕ) (monday_episodes : ℕ) 
  (h1 : total_episodes = 201)
  (h2 : weeks = 67)
  (h3 : monday_episodes = 1) :
  ∃ wednesday_episodes : ℕ, 
    wednesday_episodes * weeks + monday_episodes * weeks = total_episodes ∧ 
    wednesday_episodes = 2 := by
  sorry

end NUMINAMATH_CALUDE_office_episodes_l3315_331548


namespace NUMINAMATH_CALUDE_distance_to_focus_l3315_331595

/-- A point on a parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- The distance between a point and a vertical line -/
def distance_to_vertical_line (p : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  |p.1 - line_x|

/-- The focus of the parabola y^2 = 4x -/
def parabola_focus : ℝ × ℝ := (1, 0)

theorem distance_to_focus (P : PointOnParabola) 
  (h : distance_to_vertical_line (P.x, P.y) (-2) = 6) : 
  distance_to_vertical_line (P.x, P.y) (parabola_focus.1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_l3315_331595


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_inequality_l3315_331590

theorem unique_integer_satisfying_inequality :
  ∃! x : ℤ, 3 * x^2 + 14 * x + 24 ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_inequality_l3315_331590


namespace NUMINAMATH_CALUDE_cos_330_degrees_l3315_331592

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l3315_331592


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3315_331577

theorem trig_identity_proof :
  Real.sin (50 * π / 180) * Real.cos (20 * π / 180) -
  Real.cos (50 * π / 180) * Real.sin (20 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3315_331577


namespace NUMINAMATH_CALUDE_airplane_altitude_l3315_331528

theorem airplane_altitude (alice_bob_distance : ℝ) (alice_elevation : ℝ) (bob_elevation : ℝ) 
  (h_distance : alice_bob_distance = 15)
  (h_alice_elevation : alice_elevation = 25 * π / 180)
  (h_bob_elevation : bob_elevation = 45 * π / 180) :
  ∃ (altitude : ℝ), 3.7 < altitude ∧ altitude < 3.9 := by
  sorry


end NUMINAMATH_CALUDE_airplane_altitude_l3315_331528


namespace NUMINAMATH_CALUDE_negative_three_cubed_equality_l3315_331565

theorem negative_three_cubed_equality : -3^3 = (-3)^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_cubed_equality_l3315_331565


namespace NUMINAMATH_CALUDE_geometric_sum_111_l3315_331569

def is_geometric_progression (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r > 0 ∧ b = a * r ∧ c = a * r^2

def valid_triple (a b c : ℕ) : Prop :=
  is_geometric_progression a b c ∧ a + b + c = 111

theorem geometric_sum_111 :
  ∀ a b c : ℕ, valid_triple a b c ↔ 
    ((a, b, c) = (1, 10, 100) ∨ (a, b, c) = (100, 10, 1) ∨ (a, b, c) = (37, 37, 37)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_111_l3315_331569


namespace NUMINAMATH_CALUDE_age_sum_theorem_l3315_331584

def mother_age : ℕ := 40

def daughter_age (m : ℕ) : ℕ := (70 - m) / 2

theorem age_sum_theorem (m : ℕ) (h : m = mother_age) : 
  2 * m + daughter_age m = 95 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_theorem_l3315_331584


namespace NUMINAMATH_CALUDE_regression_variable_nature_l3315_331508

/-- A variable in regression analysis -/
inductive RegressionVariable
  | Independent
  | Dependent

/-- The nature of a variable -/
inductive VariableNature
  | Deterministic
  | Random

/-- Determines the nature of a regression variable -/
def variableNature (v : RegressionVariable) : VariableNature :=
  match v with
  | RegressionVariable.Independent => VariableNature.Deterministic
  | RegressionVariable.Dependent => VariableNature.Random

theorem regression_variable_nature :
  (variableNature RegressionVariable.Independent = VariableNature.Deterministic) ∧
  (variableNature RegressionVariable.Dependent = VariableNature.Random) := by
  sorry

end NUMINAMATH_CALUDE_regression_variable_nature_l3315_331508


namespace NUMINAMATH_CALUDE_miles_trombones_l3315_331515

/-- Represents the number of musical instruments Miles owns -/
structure MilesInstruments where
  trumpets : ℕ
  guitars : ℕ
  french_horns : ℕ
  trombones : ℕ

/-- The total number of Miles' instruments -/
def total_instruments (m : MilesInstruments) : ℕ :=
  m.trumpets + m.guitars + m.french_horns + m.trombones

theorem miles_trombones :
  ∃ (m : MilesInstruments),
    m.trumpets = 10 - 3 ∧
    m.guitars = 2 + 2 ∧
    m.french_horns = m.guitars - 1 ∧
    m.trombones = 1 + 2 ∧
    total_instruments m = 17 →
    m.trombones = 3 := by
  sorry

end NUMINAMATH_CALUDE_miles_trombones_l3315_331515


namespace NUMINAMATH_CALUDE_sum_of_intercepts_is_zero_l3315_331556

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x = y^3 - 3*y^2 + 3*y

/-- Theorem stating that the sum of x and y intercepts is 0 -/
theorem sum_of_intercepts_is_zero (a b c d : ℝ) : 
  (parabola a 0) ∧ 
  (parabola 0 b) ∧ 
  (parabola 0 c) ∧ 
  (parabola 0 d) → 
  a + b + c + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_intercepts_is_zero_l3315_331556


namespace NUMINAMATH_CALUDE_derivative_of_even_function_l3315_331561

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the derivative of f as g
variable (g : ℝ → ℝ)

-- State the theorem
theorem derivative_of_even_function 
  (h1 : ∀ x, f (-x) = f x)  -- f is an even function
  (h2 : ∀ x, HasDerivAt f (g x) x)  -- g is the derivative of f
  : ∀ x, g (-x) = -g x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_even_function_l3315_331561


namespace NUMINAMATH_CALUDE_scalene_not_unique_from_two_angles_l3315_331522

-- Define a triangle
structure Triangle :=
  (a b c : ℝ) -- side lengths
  (α β γ : ℝ) -- angles
  (h1 : a > 0 ∧ b > 0 ∧ c > 0) -- positive side lengths
  (h2 : α > 0 ∧ β > 0 ∧ γ > 0) -- positive angles
  (h3 : α + β + γ = π) -- sum of angles is π

-- Define a scalene triangle
def isScalene (t : Triangle) : Prop :=
  t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c

-- Theorem statement
theorem scalene_not_unique_from_two_angles :
  ∃ (t1 t2 : Triangle) (α β : ℝ),
    isScalene t1 ∧ isScalene t2 ∧
    t1.α = α ∧ t1.β = β ∧
    t2.α = α ∧ t2.β = β ∧
    t1 ≠ t2 :=
sorry

end NUMINAMATH_CALUDE_scalene_not_unique_from_two_angles_l3315_331522


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l3315_331537

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l3315_331537


namespace NUMINAMATH_CALUDE_alison_bought_six_small_tubs_l3315_331512

/-- Represents the number of small tubs Alison bought -/
def num_small_tubs : ℕ := sorry

/-- Represents the number of large tubs Alison bought -/
def num_large_tubs : ℕ := 3

/-- Represents the cost of a large tub in dollars -/
def cost_large_tub : ℕ := 6

/-- Represents the cost of a small tub in dollars -/
def cost_small_tub : ℕ := 5

/-- Represents the total cost of all tubs in dollars -/
def total_cost : ℕ := 48

/-- Theorem stating that Alison bought 6 small tubs -/
theorem alison_bought_six_small_tubs :
  num_large_tubs * cost_large_tub + num_small_tubs * cost_small_tub = total_cost →
  num_small_tubs = 6 := by
  sorry

end NUMINAMATH_CALUDE_alison_bought_six_small_tubs_l3315_331512


namespace NUMINAMATH_CALUDE_gcd_4370_13824_l3315_331511

theorem gcd_4370_13824 : Nat.gcd 4370 13824 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4370_13824_l3315_331511


namespace NUMINAMATH_CALUDE_partnership_investment_timing_l3315_331540

/-- A partnership problem with three investors --/
theorem partnership_investment_timing
  (x : ℝ)  -- A's investment amount
  (m : ℝ)  -- Months after which B invests
  (total_gain : ℝ)  -- Total annual gain
  (a_share : ℝ)  -- A's share of the gain
  (h1 : total_gain = 21000)  -- Given total gain
  (h2 : a_share = 7000)  -- Given A's share
  (h3 : a_share / total_gain = (x * 12) / (x * 12 + 2 * x * (12 - m) + 3 * x * 4))  -- Profit ratio equation
  : m = 6 :=
sorry

end NUMINAMATH_CALUDE_partnership_investment_timing_l3315_331540


namespace NUMINAMATH_CALUDE_binomial_expansion_arithmetic_sequence_l3315_331500

/-- 
Given a natural number n, prove that if the coefficients of the first three terms 
in the expansion of (x/2 + 1)^n form an arithmetic sequence, then n = 8.
-/
theorem binomial_expansion_arithmetic_sequence (n : ℕ) : 
  (∃ d : ℚ, 1 = (n.choose 0) ∧ 
             (n.choose 1) / 2 = 1 + d ∧ 
             (n.choose 2) / 4 = 1 + 2*d) → 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_arithmetic_sequence_l3315_331500


namespace NUMINAMATH_CALUDE_intersection_distance_l3315_331570

/-- Given a linear function f(x) = ax + b, if the distance between intersection points
    of y = x^2 - 2 and y = f(x) is √26, and the distance between intersection points
    of y = x^2 and y = f(x) + 1 is 3√2, then the distance between intersection points
    of y = x^2 and y = f(x) is √10. -/
theorem intersection_distance (a b : ℝ) : 
  let f := fun (x : ℝ) => a * x + b
  (∃ x₁ x₂ : ℝ, x₁^2 - 2 = f x₁ ∧ x₂^2 - 2 = f x₂ ∧ (x₂ - x₁)^2 = 26) →
  (∃ y₁ y₂ : ℝ, y₁^2 = f y₁ + 1 ∧ y₂^2 = f y₂ + 1 ∧ (y₂ - y₁)^2 = 18) →
  ∃ z₁ z₂ : ℝ, z₁^2 = f z₁ ∧ z₂^2 = f z₂ ∧ (z₂ - z₁)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l3315_331570


namespace NUMINAMATH_CALUDE_cave_depth_remaining_l3315_331557

/-- Given a cave of depth 974 feet and a current position of 588 feet,
    the remaining distance to the end of the cave is 386 feet. -/
theorem cave_depth_remaining (cave_depth : ℕ) (current_position : ℕ) :
  cave_depth = 974 → current_position = 588 → cave_depth - current_position = 386 := by
  sorry

end NUMINAMATH_CALUDE_cave_depth_remaining_l3315_331557


namespace NUMINAMATH_CALUDE_negation_of_and_l3315_331518

theorem negation_of_and (x y : ℝ) : ¬(x > 1 ∧ y > 2) ↔ (x ≤ 1 ∨ y ≤ 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_and_l3315_331518


namespace NUMINAMATH_CALUDE_josh_marbles_l3315_331558

/-- The number of marbles Josh initially had -/
def initial_marbles : ℕ := 19

/-- The number of marbles Josh lost -/
def lost_marbles : ℕ := 11

/-- The number of marbles Josh has now -/
def current_marbles : ℕ := initial_marbles - lost_marbles

/-- Theorem stating that Josh now has 8 marbles -/
theorem josh_marbles : current_marbles = 8 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l3315_331558


namespace NUMINAMATH_CALUDE_two_point_questions_l3315_331589

theorem two_point_questions (total_points total_questions : ℕ) 
  (h1 : total_points = 100)
  (h2 : total_questions = 40)
  (h3 : ∃ (x y : ℕ), x + y = total_questions ∧ 2*x + 4*y = total_points) :
  ∃ (x : ℕ), x = 30 ∧ 
    ∃ (y : ℕ), x + y = total_questions ∧ 2*x + 4*y = total_points :=
by
  sorry

end NUMINAMATH_CALUDE_two_point_questions_l3315_331589


namespace NUMINAMATH_CALUDE_jerry_remaining_money_l3315_331547

def mustard_oil_quantity : ℕ := 2
def mustard_oil_price : ℕ := 13
def pasta_quantity : ℕ := 3
def pasta_price : ℕ := 4
def sauce_quantity : ℕ := 1
def sauce_price : ℕ := 5
def total_budget : ℕ := 50

theorem jerry_remaining_money :
  total_budget - (mustard_oil_quantity * mustard_oil_price + 
                  pasta_quantity * pasta_price + 
                  sauce_quantity * sauce_price) = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerry_remaining_money_l3315_331547


namespace NUMINAMATH_CALUDE_triangle_side_length_l3315_331510

/-- Given a triangle ABC with sides a, b, and c, if b = 5, c = 4, and cos(B - C) = 31/32, then a = 6 -/
theorem triangle_side_length (a b c : ℝ) (B C : ℝ) : 
  b = 5 → c = 4 → Real.cos (B - C) = 31/32 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3315_331510


namespace NUMINAMATH_CALUDE_mischief_meet_handshakes_l3315_331509

/-- Calculates the number of handshakes in a group where everyone shakes hands with everyone else -/
def handshakes_in_group (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the Regional Mischief Meet -/
structure MischiefMeet where
  num_gremlins : ℕ
  num_imps : ℕ
  num_cooperative_imps : ℕ

/-- Calculates the total number of handshakes at the Regional Mischief Meet -/
def total_handshakes (meet : MischiefMeet) : ℕ :=
  handshakes_in_group meet.num_gremlins +
  handshakes_in_group meet.num_cooperative_imps +
  meet.num_gremlins * meet.num_imps

theorem mischief_meet_handshakes :
  let meet : MischiefMeet := {
    num_gremlins := 30,
    num_imps := 20,
    num_cooperative_imps := 10
  }
  total_handshakes meet = 1080 := by sorry

end NUMINAMATH_CALUDE_mischief_meet_handshakes_l3315_331509


namespace NUMINAMATH_CALUDE_base_6_conversion_l3315_331502

def base_6_to_decimal (a b c d e : ℕ) : ℕ := 
  a * (6^4) + b * (6^3) + c * (6^2) + d * (6^1) + e * (6^0)

theorem base_6_conversion (m : ℕ) : 
  base_6_to_decimal 3 m 5 0 2 = 4934 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_6_conversion_l3315_331502


namespace NUMINAMATH_CALUDE_norine_retirement_age_l3315_331525

/-- Represents Norine's retirement conditions and calculates her retirement age -/
def norineRetirement (currentAge : ℕ) (yearsWorked : ℕ) (retirementSum : ℕ) : ℕ :=
  let currentSum := currentAge + yearsWorked
  let yearsToRetirement := (retirementSum - currentSum) / 2
  currentAge + yearsToRetirement

/-- Theorem stating that Norine will retire at age 58 given the problem conditions -/
theorem norine_retirement_age :
  norineRetirement 50 19 85 = 58 := by
  sorry

end NUMINAMATH_CALUDE_norine_retirement_age_l3315_331525


namespace NUMINAMATH_CALUDE_triangle_sine_problem_l3315_331594

theorem triangle_sine_problem (a b c : ℝ) (A B C : ℝ) :
  a = 1 →
  b = Real.sqrt 2 →
  Real.sin A = 1/3 →
  (a / Real.sin A = b / Real.sin B) →
  Real.sin B = Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_problem_l3315_331594


namespace NUMINAMATH_CALUDE_equation_system_solution_l3315_331532

theorem equation_system_solution (a b c : ℝ) : 
  (∀ x y : ℝ, a * x + y = 5 ∧ b * x - c * y = -1) →
  (3 * 2 + 3 = 5 ∧ b * 2 - c * 3 = -1) →
  (a * 1 + 2 = 5 ∧ b * 1 - c * 2 = -1) →
  a + b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3315_331532


namespace NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l3315_331555

theorem four_digit_number_with_specific_remainders :
  ∃! N : ℕ,
    N % 131 = 112 ∧
    N % 132 = 98 ∧
    1000 ≤ N ∧
    N < 10000 ∧
    N = 1946 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l3315_331555


namespace NUMINAMATH_CALUDE_liouvilles_theorem_l3315_331538

theorem liouvilles_theorem (p m : ℕ) (h_prime : Nat.Prime p) (h_p_gt_5 : p > 5) (h_m_pos : m > 0) :
  (Nat.factorial (p - 1) + 1) ≠ p ^ m := by
  sorry

end NUMINAMATH_CALUDE_liouvilles_theorem_l3315_331538


namespace NUMINAMATH_CALUDE_fifteen_foot_string_wicks_l3315_331507

/-- Calculates the total number of wicks that can be cut from a string of given length,
    where the wicks are of two different lengths and there are an equal number of each. -/
def total_wicks (total_length_feet : ℕ) (wick_length_1 : ℕ) (wick_length_2 : ℕ) : ℕ :=
  let total_length_inches := total_length_feet * 12
  let pair_length := wick_length_1 + wick_length_2
  let num_pairs := total_length_inches / pair_length
  2 * num_pairs

/-- Theorem stating that a 15-foot string cut into equal numbers of 6-inch and 12-inch wicks
    results in a total of 20 wicks. -/
theorem fifteen_foot_string_wicks :
  total_wicks 15 6 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_foot_string_wicks_l3315_331507


namespace NUMINAMATH_CALUDE_trigonometric_values_for_point_l3315_331573

theorem trigonometric_values_for_point (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = -4 ∧ r * Real.sin α = 3) →
  Real.sin α = 3/5 ∧ Real.cos α = -4/5 ∧ Real.tan α = -3/4 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_values_for_point_l3315_331573


namespace NUMINAMATH_CALUDE_composition_of_transformations_l3315_331581

-- Define the transformations
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)
def g (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- State the theorem
theorem composition_of_transformations :
  g (f (-6, 7)) = (-7, 6) := by sorry

end NUMINAMATH_CALUDE_composition_of_transformations_l3315_331581


namespace NUMINAMATH_CALUDE_total_fruits_after_changes_l3315_331526

def initial_oranges : Nat := 40
def initial_apples : Nat := 25
def initial_bananas : Nat := 15

def removed_oranges : Nat := 37
def added_oranges : Nat := 7
def removed_apples : Nat := 10
def added_bananas : Nat := 12

def final_oranges : Nat := initial_oranges - removed_oranges + added_oranges
def final_apples : Nat := initial_apples - removed_apples
def final_bananas : Nat := initial_bananas + added_bananas

theorem total_fruits_after_changes :
  final_oranges + final_apples + final_bananas = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_after_changes_l3315_331526


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l3315_331588

theorem right_triangle_third_side_product (a b c d : ℝ) : 
  a = 6 → b = 8 → 
  ((a^2 + b^2 = c^2) ∨ (a^2 + d^2 = b^2)) →
  c * d = 20 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l3315_331588


namespace NUMINAMATH_CALUDE_hyperbola_condition_l3315_331591

-- Define the equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 3) - y^2 / (k + 3) = 1

-- State the theorem
theorem hyperbola_condition (k : ℝ) :
  (k > 3 → is_hyperbola k) ∧ (∃ k₀ ≤ 3, is_hyperbola k₀) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l3315_331591


namespace NUMINAMATH_CALUDE_ap_terms_count_l3315_331519

theorem ap_terms_count (a d : ℚ) (n : ℕ) (h_even : Even n) (h_odd_sum : (n / 2) * (2 * a + (n - 2) * d) = 30)
    (h_even_sum : (n / 2) * (2 * a + n * d) = 50) (h_last_first : (n - 1) * d = 15) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ap_terms_count_l3315_331519


namespace NUMINAMATH_CALUDE_greatest_x_value_l3315_331545

theorem greatest_x_value (x : ℝ) : 
  (x^2 - x - 90) / (x - 9) = 4 / (x + 6) → x ≤ -7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3315_331545


namespace NUMINAMATH_CALUDE_not_adjacent_probability_l3315_331554

theorem not_adjacent_probability (n : ℕ) (h : n = 10) : 
  (n.choose 2 - (n - 1)) / n.choose 2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_not_adjacent_probability_l3315_331554


namespace NUMINAMATH_CALUDE_remainder_27_pow_27_plus_27_mod_28_l3315_331567

theorem remainder_27_pow_27_plus_27_mod_28 :
  (27^27 + 27) % 28 = 26 := by
sorry

end NUMINAMATH_CALUDE_remainder_27_pow_27_plus_27_mod_28_l3315_331567


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l3315_331543

/-- Represents the dimensions of the triangular brownie pan -/
structure PanDimensions where
  base : ℕ
  height : ℕ

/-- Represents the dimensions of a single brownie piece -/
structure PieceDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of full brownie pieces that can be obtained from a triangular pan -/
def maxBrowniePieces (pan : PanDimensions) (piece : PieceDimensions) : ℕ :=
  (pan.base / piece.width) * (pan.height / piece.height)

/-- Theorem stating the maximum number of brownie pieces for the given dimensions -/
theorem brownie_pieces_count :
  let pan := PanDimensions.mk 30 24
  let piece := PieceDimensions.mk 3 4
  maxBrowniePieces pan piece = 60 := by
sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l3315_331543


namespace NUMINAMATH_CALUDE_novel_series_arrangement_l3315_331516

def number_of_series : ℕ := 3
def volumes_per_series : ℕ := 4

def arrangement_count : ℕ := 34650

theorem novel_series_arrangement :
  (Nat.factorial (number_series * volumes_per_series)) / 
  (Nat.factorial volumes_per_series)^number_of_series = arrangement_count := by
  sorry

end NUMINAMATH_CALUDE_novel_series_arrangement_l3315_331516


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_under_cube_root_2000_l3315_331571

theorem greatest_multiple_of_four_under_cube_root_2000 :
  ∀ x : ℕ, 
    x > 0 → 
    x % 4 = 0 → 
    x^3 < 2000 → 
    x ≤ 12 ∧ 
    ∃ y : ℕ, y > 0 ∧ y % 4 = 0 ∧ y^3 < 2000 ∧ y = 12 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_under_cube_root_2000_l3315_331571


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l3315_331586

theorem triangle_inequality_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l3315_331586


namespace NUMINAMATH_CALUDE_theater_line_arrangements_l3315_331505

theorem theater_line_arrangements (n : ℕ) (h : n = 7) : 
  Nat.factorial n = 5040 := by
  sorry

end NUMINAMATH_CALUDE_theater_line_arrangements_l3315_331505


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3315_331549

theorem quadratic_one_root (a : ℝ) : 
  (∃! x, (a + 2) * x^2 + 2 * a * x + 1 = 0) → (a = 2 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3315_331549


namespace NUMINAMATH_CALUDE_range_of_expression_l3315_331562

theorem range_of_expression (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  1 / a + a / b ≥ 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l3315_331562


namespace NUMINAMATH_CALUDE_cars_between_black_and_white_l3315_331523

/-- Given a row of 20 cars, with a black car 16th from the right and a white car 11th from the left,
    the number of cars between the black and white cars is 5. -/
theorem cars_between_black_and_white :
  ∀ (total_cars : ℕ) (black_from_right white_from_left : ℕ),
    total_cars = 20 →
    black_from_right = 16 →
    white_from_left = 11 →
    (white_from_left - (total_cars - black_from_right + 1) - 1 = 5) :=
by sorry

end NUMINAMATH_CALUDE_cars_between_black_and_white_l3315_331523


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l3315_331560

theorem quadratic_solution_range (t : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - t = 0 → -1 < x ∧ x < 4) →
  -1 ≤ t ∧ t < 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l3315_331560
