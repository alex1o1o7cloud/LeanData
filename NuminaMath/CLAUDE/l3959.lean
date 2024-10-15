import Mathlib

namespace NUMINAMATH_CALUDE_no_product_equality_l3959_395917

def a : ℕ → ℤ
  | 0 => 2
  | 1 => 5
  | (n + 2) => (2 - n^2) * a (n + 1) + (2 + n^2) * a n

theorem no_product_equality : ¬∃ (p q r : ℕ+), a p.val * a q.val = a r.val := by
  sorry

end NUMINAMATH_CALUDE_no_product_equality_l3959_395917


namespace NUMINAMATH_CALUDE_expression_values_l3959_395951

theorem expression_values (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  let e := x / |x| + y / |y| + z / |z| + (x*y*z) / |x*y*z|
  e = 4 ∨ e = 0 ∨ e = -4 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l3959_395951


namespace NUMINAMATH_CALUDE_coefficient_equals_168_l3959_395941

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the coefficient of x^2y^2 in (1+x)^8(1+y)^4
def coefficient_x2y2 : ℕ := binomial 8 2 * binomial 4 2

-- Theorem statement
theorem coefficient_equals_168 : coefficient_x2y2 = 168 := by sorry

end NUMINAMATH_CALUDE_coefficient_equals_168_l3959_395941


namespace NUMINAMATH_CALUDE_verify_scenario1_verify_scenario2_prove_gizmo_production_l3959_395967

/-- Represents the time (in hours) for one worker to produce one gizmo -/
def gizmo_time : ℚ := 1/5

/-- Represents the time (in hours) for one worker to produce one gadget -/
def gadget_time : ℚ := 1/5

/-- Verifies that 80 workers in 1 hour produce 160 gizmos and 240 gadgets -/
theorem verify_scenario1 : 
  80 * (1 / gizmo_time) = 160 ∧ 80 * (1 / gadget_time) = 240 := by sorry

/-- Verifies that 100 workers in 3 hours produce 900 gizmos and 600 gadgets -/
theorem verify_scenario2 : 
  100 * (3 / gizmo_time) = 900 ∧ 100 * (3 / gadget_time) = 600 := by sorry

/-- Proves that 70 workers in 5 hours produce 70 gizmos -/
theorem prove_gizmo_production : 
  70 * (5 / gizmo_time) = 70 := by sorry

end NUMINAMATH_CALUDE_verify_scenario1_verify_scenario2_prove_gizmo_production_l3959_395967


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3959_395920

theorem sum_of_three_numbers : 72.52 + 12.23 + 5.21 = 89.96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3959_395920


namespace NUMINAMATH_CALUDE_right_triangle_vector_relation_l3959_395966

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define vectors
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem right_triangle_vector_relation (t : ℝ) (ABC : Triangle) :
  (ABC.C.1 - ABC.A.1 = 2 ∧ ABC.C.2 - ABC.A.2 = 2) →  -- AC = (2, 2)
  (ABC.B.1 - ABC.A.1 = t ∧ ABC.B.2 - ABC.A.2 = 1) →  -- AB = (t, 1)
  dot_product (vector ABC.A ABC.C) (vector ABC.A ABC.B) = 0 →  -- Angle C is 90 degrees
  t = 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_vector_relation_l3959_395966


namespace NUMINAMATH_CALUDE_teaching_fee_sum_l3959_395971

theorem teaching_fee_sum (k : ℚ) : 
  (5 * k) / (4 * k) = 5 / 4 →
  (5 * k + 20) / (4 * k + 20) = 6 / 5 →
  (5 * k + 20) + (4 * k + 20) = 220 := by
  sorry

end NUMINAMATH_CALUDE_teaching_fee_sum_l3959_395971


namespace NUMINAMATH_CALUDE_pipe_crate_height_difference_l3959_395997

/-- The height difference between two crates of cylindrical pipes -/
theorem pipe_crate_height_difference (pipe_diameter : ℝ) (crate_a_rows : ℕ) (crate_b_rows : ℕ) :
  pipe_diameter = 20 →
  crate_a_rows = 10 →
  crate_b_rows = 9 →
  let crate_a_height := crate_a_rows * pipe_diameter
  let crate_b_height := crate_b_rows * pipe_diameter + (crate_b_rows - 1) * pipe_diameter * Real.sqrt 3
  crate_a_height - crate_b_height = 20 - 160 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_pipe_crate_height_difference_l3959_395997


namespace NUMINAMATH_CALUDE_unique_solution_l3959_395932

def SatisfiesEquation (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f n + f (f n) + f (f (f n)) = 3 * n

theorem unique_solution :
  ∀ f : ℕ → ℕ, SatisfiesEquation f → (∀ n : ℕ, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l3959_395932


namespace NUMINAMATH_CALUDE_prime_pairs_dividing_power_sum_l3959_395950

theorem prime_pairs_dividing_power_sum :
  ∀ p q : ℕ,
  Nat.Prime p → Nat.Prime q →
  (p * q ∣ 2^p + 2^q) ↔ ((p = 2 ∧ q = 2) ∨ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_dividing_power_sum_l3959_395950


namespace NUMINAMATH_CALUDE_supermarket_pricing_problem_l3959_395935

-- Define the linear function
def sales_function (x : ℝ) : ℝ := -2 * x + 60

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - 10) * (sales_function x)

theorem supermarket_pricing_problem :
  -- 1. The linear function satisfies the given data points
  (sales_function 12 = 36 ∧ sales_function 13 = 34) ∧
  -- 2. When the profit is 192 yuan, the selling price is 18 yuan
  (profit_function 18 = 192) ∧
  -- 3. The maximum profit is 198 yuan when the selling price is 19 yuan, given the constraints
  (∀ x : ℝ, 10 ≤ x ∧ x ≤ 19 → profit_function x ≤ profit_function 19) ∧
  (profit_function 19 = 198) :=
by sorry

end NUMINAMATH_CALUDE_supermarket_pricing_problem_l3959_395935


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3959_395994

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3959_395994


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l3959_395959

def M : Set ℤ := {0, 1, 2, 3, 4}
def N : Set ℤ := {-2, 0, 2}

theorem set_intersection_theorem : M ∩ N = {0, 2} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l3959_395959


namespace NUMINAMATH_CALUDE_equation_solution_l3959_395908

def solution_set : Set (ℤ × ℤ) :=
  {(-2, 4), (-2, 6), (0, 10), (4, -2), (4, 12), (6, -2), (6, 12), (10, 0), (10, 10), (12, 4), (12, 6)}

theorem equation_solution (x y : ℤ) :
  x + y ≠ 0 →
  (((x^2 + y^2) : ℚ) / (x + y : ℚ) = 10) ↔ (x, y) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3959_395908


namespace NUMINAMATH_CALUDE_matrix_product_equality_l3959_395954

theorem matrix_product_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A - B = A * B)
  (h2 : A * B = ![![7, -2], ![4, -3]]) :
  B * A = ![![6, -2], ![4, -4]] := by sorry

end NUMINAMATH_CALUDE_matrix_product_equality_l3959_395954


namespace NUMINAMATH_CALUDE_account_balance_increase_l3959_395928

theorem account_balance_increase (initial_deposit : ℝ) (first_year_balance : ℝ) (total_increase_percent : ℝ) :
  initial_deposit = 1000 →
  first_year_balance = 1100 →
  total_increase_percent = 32 →
  let final_balance := initial_deposit * (1 + total_increase_percent / 100)
  let second_year_increase := final_balance - first_year_balance
  let second_year_increase_percent := (second_year_increase / first_year_balance) * 100
  second_year_increase_percent = 20 := by
sorry

end NUMINAMATH_CALUDE_account_balance_increase_l3959_395928


namespace NUMINAMATH_CALUDE_no_intersection_l3959_395972

/-- The line 3x + 4y = 12 and the circle x^2 + y^2 = 4 have no points of intersection -/
theorem no_intersection : 
  ∀ x y : ℝ, (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → False :=
by sorry

end NUMINAMATH_CALUDE_no_intersection_l3959_395972


namespace NUMINAMATH_CALUDE_gift_cost_per_teacher_l3959_395960

/-- Proves that if a person buys gifts for 7 teachers and spends $70 in total, then each gift costs $10. -/
theorem gift_cost_per_teacher (num_teachers : ℕ) (total_spent : ℚ) : 
  num_teachers = 7 → total_spent = 70 → total_spent / num_teachers = 10 := by
  sorry

end NUMINAMATH_CALUDE_gift_cost_per_teacher_l3959_395960


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l3959_395999

theorem geometric_progression_proof (b₁ q : ℝ) (h_decreasing : |q| < 1) :
  (b₁^3 / (1 - q^3)) / (b₁ / (1 - q)) = 48/7 →
  (b₁^4 / (1 - q^4)) / (b₁^2 / (1 - q^2)) = 144/17 →
  (b₁ = 3 ∨ b₁ = -3) ∧ q = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l3959_395999


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l3959_395986

/-- The volume of a tetrahedron with an inscribed sphere -/
theorem tetrahedron_volume (S₁ S₂ S₃ S₄ r : ℝ) (h₁ : S₁ > 0) (h₂ : S₂ > 0) (h₃ : S₃ > 0) (h₄ : S₄ > 0) (hr : r > 0) :
  ∃ V : ℝ, V = (1/3) * (S₁ + S₂ + S₃ + S₄) * r ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l3959_395986


namespace NUMINAMATH_CALUDE_sequence_increasing_iff_l3959_395902

theorem sequence_increasing_iff (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = 2^n - 3 * a n) →
  (∀ n : ℕ, a (n + 1) > a n) ↔
  a 0 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_iff_l3959_395902


namespace NUMINAMATH_CALUDE_fruiting_plants_given_away_l3959_395979

/-- Represents the number of plants in Roxy's garden -/
structure GardenState where
  flowering : ℕ
  fruiting : ℕ

/-- Calculates the total number of plants -/
def GardenState.total (s : GardenState) : ℕ := s.flowering + s.fruiting

def initial_state : GardenState :=
  { flowering := 7,
    fruiting := 2 * 7 }

def after_buying : GardenState :=
  { flowering := initial_state.flowering + 3,
    fruiting := initial_state.fruiting + 2 }

def plants_remaining : ℕ := 21

def flowering_given_away : ℕ := 1

theorem fruiting_plants_given_away :
  ∃ (x : ℕ), 
    after_buying.fruiting - x = plants_remaining - (after_buying.flowering - flowering_given_away) ∧
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fruiting_plants_given_away_l3959_395979


namespace NUMINAMATH_CALUDE_percentage_relation_l3959_395934

/-- Given the relationships between j, k, l, and m, prove that 150% of k equals 50% of l -/
theorem percentage_relation (j k l m : ℝ) : 
  (1.25 * j = 0.25 * k) →
  (∃ x : ℝ, 0.01 * x * k = 0.5 * l) →
  (1.75 * l = 0.75 * m) →
  (0.2 * m = 7 * j) →
  1.5 * k = 0.5 * l := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l3959_395934


namespace NUMINAMATH_CALUDE_f_derivative_positive_at_midpoint_l3959_395968

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a-2)*x - a * log x

theorem f_derivative_positive_at_midpoint (a c x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) 
  (h₄ : f a x₁ = c) (h₅ : f a x₂ = c) : 
  deriv (f a) ((x₁ + x₂) / 2) > 0 := by
sorry

end NUMINAMATH_CALUDE_f_derivative_positive_at_midpoint_l3959_395968


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_l3959_395901

theorem smallest_x_absolute_value (x : ℝ) : 
  (∀ y, |y + 4| = 15 → x ≤ y) ↔ x = -19 := by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_l3959_395901


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3959_395922

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (30 * π / 180 + α) = 3/5)
  (h2 : 60 * π / 180 < α)
  (h3 : α < 150 * π / 180) :
  Real.cos α = (3 - 4 * Real.sqrt 3) / 10 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3959_395922


namespace NUMINAMATH_CALUDE_product_sequence_equals_32_l3959_395918

theorem product_sequence_equals_32 : 
  (1 / 4 : ℚ) * 8 * (1 / 16 : ℚ) * 32 * (1 / 64 : ℚ) * 128 * (1 / 256 : ℚ) * 512 * (1 / 1024 : ℚ) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_equals_32_l3959_395918


namespace NUMINAMATH_CALUDE_angle_complement_supplement_relation_l3959_395958

/-- 
Given an angle x in degrees, if its complement (90° - x) is 75% of its supplement (180° - x), 
then x = 180°.
-/
theorem angle_complement_supplement_relation (x : ℝ) : 
  (90 - x) = 0.75 * (180 - x) → x = 180 := by sorry

end NUMINAMATH_CALUDE_angle_complement_supplement_relation_l3959_395958


namespace NUMINAMATH_CALUDE_cauchy_schwarz_and_max_value_l3959_395919

theorem cauchy_schwarz_and_max_value :
  (∀ a b c d : ℝ, (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2) ∧
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a + b = 1 → (Real.sqrt (3*a + 1) + Real.sqrt (3*b + 1))^2 ≤ 10) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_and_max_value_l3959_395919


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_first_two_increasing_l3959_395939

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_increasing_iff_first_two_increasing
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : a 1 > 0) :
  IncreasingSequence a ↔ a 1 < a 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_first_two_increasing_l3959_395939


namespace NUMINAMATH_CALUDE_atticus_marbles_l3959_395985

theorem atticus_marbles (a j c : ℕ) : 
  3 * (a + j + c) = 60 →
  a = j / 2 →
  c = 8 →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_atticus_marbles_l3959_395985


namespace NUMINAMATH_CALUDE_bananas_per_visit_l3959_395915

theorem bananas_per_visit (store_visits : ℕ) (total_bananas : ℕ) (bananas_per_visit : ℕ) : 
  store_visits = 2 → total_bananas = 20 → bananas_per_visit * store_visits = total_bananas → bananas_per_visit = 10 := by
  sorry

end NUMINAMATH_CALUDE_bananas_per_visit_l3959_395915


namespace NUMINAMATH_CALUDE_enemies_left_proof_l3959_395992

def enemies_left_undefeated (total_enemies : ℕ) (points_per_enemy : ℕ) (total_points : ℕ) : ℕ :=
  total_enemies - (total_points / points_per_enemy)

theorem enemies_left_proof (total_enemies : ℕ) (points_per_enemy : ℕ) (total_points : ℕ)
  (h1 : total_enemies = 11)
  (h2 : points_per_enemy = 9)
  (h3 : total_points = 72) :
  enemies_left_undefeated total_enemies points_per_enemy total_points = 3 :=
by
  sorry

#eval enemies_left_undefeated 11 9 72

end NUMINAMATH_CALUDE_enemies_left_proof_l3959_395992


namespace NUMINAMATH_CALUDE_inequality_for_natural_numbers_l3959_395906

theorem inequality_for_natural_numbers (n : ℕ) :
  (2 * n + 1)^n ≥ (2 * n)^n + (2 * n - 1)^n := by sorry

end NUMINAMATH_CALUDE_inequality_for_natural_numbers_l3959_395906


namespace NUMINAMATH_CALUDE_angle_U_measure_l3959_395955

/-- A hexagon with specific angle properties -/
structure SpecialHexagon where
  -- Define the angles of the hexagon
  F : ℝ
  G : ℝ
  I : ℝ
  R : ℝ
  U : ℝ
  E : ℝ
  -- Conditions from the problem
  angle_sum : F + G + I + R + U + E = 720
  angle_congruence : F = I ∧ I = U
  supplementary_GR : G + R = 180
  supplementary_EU : E + U = 180

/-- The measure of angle U in the special hexagon is 120 degrees -/
theorem angle_U_measure (h : SpecialHexagon) : h.U = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_U_measure_l3959_395955


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l3959_395944

/-- Prove that given the conditions of the cyclist problem, the speed of cyclist C is 10 mph. -/
theorem cyclist_speed_problem (c d : ℝ) : 
  d = c + 5 →  -- C travels 5 mph slower than D
  (80 - 16) / c = (80 + 16) / d →  -- Travel times are equal
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l3959_395944


namespace NUMINAMATH_CALUDE_triangle_inequality_l3959_395964

theorem triangle_inequality (a b c : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) → 
  (¬(a + b > c ∧ b + c > a ∧ c + a > b) ↔ a + b ≤ c ∨ b + c ≤ a ∨ c + a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3959_395964


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l3959_395975

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- Theorem statement
theorem line_perp_parallel_planes 
  (α β : Plane) (a : Line) 
  (h1 : parallel α β) 
  (h2 : perpendicular a β) : 
  perpendicular a α :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l3959_395975


namespace NUMINAMATH_CALUDE_intersection_M_N_l3959_395931

def M : Set ℝ := {x | (x - 1)^2 < 4}

def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3959_395931


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3959_395996

/-- An equilateral triangle is a triangle with all sides of equal length -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The perimeter of a triangle is the sum of its side lengths -/
def perimeter (triangle : EquilateralTriangle) : ℝ :=
  3 * triangle.side_length

/-- Theorem: The perimeter of an equilateral triangle with side length 'a' is 3a -/
theorem equilateral_triangle_perimeter (a : ℝ) (ha : a > 0) :
  let triangle : EquilateralTriangle := ⟨a, ha⟩
  perimeter triangle = 3 * a := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3959_395996


namespace NUMINAMATH_CALUDE_total_travel_time_is_144_hours_l3959_395929

/-- Represents the travel times between different locations -/
structure TravelTimes where
  ngaparaToZipra : ℝ
  ningiToZipra : ℝ
  ziproToVarnasi : ℝ

/-- Calculates the total travel time given the travel times between locations -/
def totalTravelTime (t : TravelTimes) : ℝ :=
  t.ngaparaToZipra + t.ningiToZipra + t.ziproToVarnasi

/-- Theorem stating the total travel time given the conditions in the problem -/
theorem total_travel_time_is_144_hours :
  ∀ t : TravelTimes,
  t.ngaparaToZipra = 60 →
  t.ningiToZipra = 0.8 * t.ngaparaToZipra →
  t.ziproToVarnasi = 0.75 * t.ningiToZipra →
  totalTravelTime t = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_is_144_hours_l3959_395929


namespace NUMINAMATH_CALUDE_inequality_proof_l3959_395980

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3959_395980


namespace NUMINAMATH_CALUDE_flower_perimeter_l3959_395905

/-- The perimeter of a flower-like figure created by a regular hexagon inscribed in a circle -/
theorem flower_perimeter (c : ℝ) (h : c = 16) : 
  let hexagon_arc := c / 6
  let petal_arc := 2 * hexagon_arc
  let num_petals := 6
  num_petals * petal_arc = 32 := by
  sorry

end NUMINAMATH_CALUDE_flower_perimeter_l3959_395905


namespace NUMINAMATH_CALUDE_angus_tokens_l3959_395930

def token_value : ℕ := 4
def elsa_tokens : ℕ := 60
def token_difference : ℕ := 20

theorem angus_tokens : 
  ∃ (angus_tokens : ℕ), 
    angus_tokens * token_value = elsa_tokens * token_value - token_difference ∧ 
    angus_tokens = 55 :=
by sorry

end NUMINAMATH_CALUDE_angus_tokens_l3959_395930


namespace NUMINAMATH_CALUDE_unique_root_of_equation_l3959_395965

theorem unique_root_of_equation :
  ∃! x : ℝ, (3 : ℝ)^x + (5 : ℝ)^x + (11 : ℝ)^x = (19 : ℝ)^x * Real.sqrt (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_of_equation_l3959_395965


namespace NUMINAMATH_CALUDE_cake_division_theorem_l3959_395916

-- Define the cake and its properties
structure Cake where
  length : ℝ
  width : ℝ
  area : ℝ
  h_area_positive : area > 0
  h_area_calc : area = length * width

-- Define the cuts
structure Cuts where
  x : ℝ
  y : ℝ
  z : ℝ
  h_x_positive : x > 0
  h_y_positive : y > 0
  h_z_positive : z > 0
  h_sum : x + y + z = 1

-- Define the theorem
theorem cake_division_theorem (cake : Cake) (cuts : Cuts) :
  ∃ (piece1 piece2 : ℝ),
    piece1 + piece2 ≥ 0.25 * cake.area ∧
    piece1 = max (cake.length * cuts.x * cake.width) (cake.length * cuts.y * cake.width) ∧
    piece2 = min (cake.length * cuts.x * cake.width) (cake.length * cuts.y * cake.width) ∧
    cake.area - (piece1 + piece2) ≤ 0.75 * cake.area :=
by sorry

end NUMINAMATH_CALUDE_cake_division_theorem_l3959_395916


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3959_395973

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3959_395973


namespace NUMINAMATH_CALUDE_apple_price_difference_l3959_395938

/-- Given the prices of Shimla apples (S), Red Delicious apples (R), and Fuji apples (F) in rupees,
    prove that the difference in price between Shimla and Fuji apples can be expressed as shown,
    given the condition from the problem. -/
theorem apple_price_difference (S R F : ℝ) 
  (h : 1.05 * (S + R) = R + 0.90 * F + 250) :
  S - F = (-0.15 * S - 0.05 * R) / 0.90 + 250 / 0.90 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_difference_l3959_395938


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l3959_395927

theorem number_of_boys_in_class (num_girls : ℕ) (avg_boys : ℚ) (avg_girls : ℚ) (avg_class : ℚ) :
  num_girls = 4 ∧ avg_boys = 84 ∧ avg_girls = 92 ∧ avg_class = 86 →
  ∃ (num_boys : ℕ), num_boys = 12 ∧
    (avg_boys * num_boys + avg_girls * num_girls) / (num_boys + num_girls) = avg_class :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_l3959_395927


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3959_395907

theorem absolute_value_inequality (x : ℝ) :
  |((3 * x + 2) / (x - 2))| ≥ 3 ↔ x ∈ Set.Ici (2/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3959_395907


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3959_395974

theorem fraction_inequality_solution_set (x : ℝ) :
  x ≠ 0 → ((x - 1) / x ≤ 0 ↔ 0 < x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3959_395974


namespace NUMINAMATH_CALUDE_product_increased_by_four_l3959_395981

theorem product_increased_by_four (x : ℝ) (h : x = 3) : 5 * x + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_product_increased_by_four_l3959_395981


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_six_l3959_395912

theorem sin_thirteen_pi_six : Real.sin (13 * π / 6) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_six_l3959_395912


namespace NUMINAMATH_CALUDE_hayden_earnings_354_l3959_395936

/-- Represents Hayden's work day at the limousine company -/
structure HaydenWorkDay where
  shortRideReimbursement : ℕ := 3
  longRideReimbursement : ℕ := 4
  baseHourlyWage : ℕ := 15
  shortRideBonus : ℕ := 7
  longRideBonus : ℕ := 10
  goodReviewBonus : ℕ := 20
  excellentReviewBonus : ℕ := 30
  totalRides : ℕ := 5
  longRides : ℕ := 2
  hoursWorked : ℕ := 11
  shortRideGas : ℕ := 10
  longRideGas : ℕ := 15
  tollFee : ℕ := 6
  numTolls : ℕ := 2
  goodReviews : ℕ := 2
  excellentReviews : ℕ := 1

/-- Calculates Hayden's total earnings for the day -/
def calculateEarnings (day : HaydenWorkDay) : ℕ :=
  let baseEarnings := day.baseHourlyWage * day.hoursWorked
  let shortRides := day.totalRides - day.longRides
  let rideBonuses := shortRides * day.shortRideBonus + day.longRides * day.longRideBonus
  let gasReimbursement := day.shortRideGas * day.shortRideReimbursement + day.longRideGas * day.longRideReimbursement
  let reviewBonuses := day.goodReviews * day.goodReviewBonus + day.excellentReviews * day.excellentReviewBonus
  let totalBeforeTolls := baseEarnings + rideBonuses + gasReimbursement + reviewBonuses
  totalBeforeTolls - (day.numTolls * day.tollFee)

/-- Theorem stating that Hayden's earnings for the day equal $354 -/
theorem hayden_earnings_354 (day : HaydenWorkDay) : calculateEarnings day = 354 := by
  sorry

end NUMINAMATH_CALUDE_hayden_earnings_354_l3959_395936


namespace NUMINAMATH_CALUDE_largest_cylinder_radius_largest_cylinder_radius_is_4_l3959_395947

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : Real
  width : Real
  height : Real

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : Real
  height : Real

/-- Checks if a cylinder fits in a crate when the crate is on its length-width side -/
def cylinderFitsInCrate (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  cylinder.height ≤ crate.height ∧ 2 * cylinder.radius ≤ min crate.length crate.width

/-- The largest cylinder that fits in the crate has a radius equal to half the smaller of the crate's length or width -/
theorem largest_cylinder_radius (crate : CrateDimensions) :
  ∃ (cylinder : Cylinder),
    cylinderFitsInCrate crate cylinder ∧
    ∀ (other : Cylinder), cylinderFitsInCrate crate other → other.radius ≤ cylinder.radius :=
  sorry

/-- The specific crate dimensions given in the problem -/
def problemCrate : CrateDimensions :=
  { length := 12, width := 8, height := 6 }

/-- The theorem stating that the largest cylinder that fits in the problem crate has a radius of 4 feet -/
theorem largest_cylinder_radius_is_4 :
  ∃ (cylinder : Cylinder),
    cylinderFitsInCrate problemCrate cylinder ∧
    cylinder.radius = 4 ∧
    ∀ (other : Cylinder), cylinderFitsInCrate problemCrate other → other.radius ≤ cylinder.radius :=
  sorry

end NUMINAMATH_CALUDE_largest_cylinder_radius_largest_cylinder_radius_is_4_l3959_395947


namespace NUMINAMATH_CALUDE_lunch_packet_cost_l3959_395963

/-- Represents the field trip scenario -/
structure FieldTrip where
  total_students : Nat
  lunch_buyers : Nat
  packet_cost : Nat
  apples_per_packet : Nat
  total_cost : Nat

/-- The field trip satisfies the given conditions -/
def valid_field_trip (ft : FieldTrip) : Prop :=
  ft.total_students = 50 ∧
  ft.lunch_buyers > ft.total_students / 2 ∧
  ft.apples_per_packet < ft.packet_cost ∧
  ft.lunch_buyers * ft.packet_cost = ft.total_cost ∧
  ft.total_cost = 3087

theorem lunch_packet_cost (ft : FieldTrip) :
  valid_field_trip ft → ft.packet_cost = 9 := by
  sorry

#check lunch_packet_cost

end NUMINAMATH_CALUDE_lunch_packet_cost_l3959_395963


namespace NUMINAMATH_CALUDE_power_product_simplification_l3959_395998

theorem power_product_simplification :
  (-3/2 : ℚ)^2023 * (-2/3 : ℚ)^2022 = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l3959_395998


namespace NUMINAMATH_CALUDE_integer_solution_of_inequality_l3959_395940

theorem integer_solution_of_inequality (x : ℤ) : 3 ≤ 3 * x + 3 ∧ 3 * x + 3 ≤ 5 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_of_inequality_l3959_395940


namespace NUMINAMATH_CALUDE_harmonic_sets_theorem_l3959_395993

-- Define a circle
class Circle where
  -- Add any necessary properties for a circle

-- Define a point on a circle
class PointOnCircle (c : Circle) where
  -- Add any necessary properties for a point on a circle

-- Define a line
class Line where
  -- Add any necessary properties for a line

-- Define the property of lines intersecting at a single point
def intersectAtSinglePoint (l1 l2 l3 l4 : Line) : Prop :=
  sorry

-- Define a harmonic set of points
def isHarmonic {c : Circle} (A B C D : PointOnCircle c) : Prop :=
  sorry

-- Define the line connecting two points
def connectingLine {c : Circle} (P Q : PointOnCircle c) : Line :=
  sorry

theorem harmonic_sets_theorem
  {c : Circle}
  (A B C D A₁ B₁ C₁ D₁ : PointOnCircle c)
  (h_intersect : intersectAtSinglePoint
    (connectingLine A A₁)
    (connectingLine B B₁)
    (connectingLine C C₁)
    (connectingLine D D₁))
  (h_harmonic : isHarmonic A B C D ∨ isHarmonic A₁ B₁ C₁ D₁) :
  isHarmonic A B C D ∧ isHarmonic A₁ B₁ C₁ D₁ :=
sorry

end NUMINAMATH_CALUDE_harmonic_sets_theorem_l3959_395993


namespace NUMINAMATH_CALUDE_opening_weekend_revenue_calculation_l3959_395937

/-- Represents the movie's financial data in millions of dollars -/
structure MovieFinancials where
  openingWeekendRevenue : ℝ
  totalRevenue : ℝ
  productionCompanyRevenue : ℝ
  productionCost : ℝ
  profit : ℝ

/-- Theorem stating the opening weekend revenue given the movie's financial conditions -/
theorem opening_weekend_revenue_calculation (m : MovieFinancials) 
  (h1 : m.totalRevenue = 3.5 * m.openingWeekendRevenue)
  (h2 : m.productionCompanyRevenue = 0.6 * m.totalRevenue)
  (h3 : m.profit = m.productionCompanyRevenue - m.productionCost)
  (h4 : m.profit = 192)
  (h5 : m.productionCost = 60) :
  m.openingWeekendRevenue = 120 := by
  sorry

end NUMINAMATH_CALUDE_opening_weekend_revenue_calculation_l3959_395937


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_143_l3959_395988

def sum_of_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_prime_factors_143 : sum_of_prime_factors 143 = 24 := by sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_143_l3959_395988


namespace NUMINAMATH_CALUDE_candy_jar_theorem_l3959_395970

/-- Represents the number of candies of each type in a jar -/
structure CandyJar where
  orange : ℕ
  purple : ℕ
  white : ℕ
  green : ℕ
  black : ℕ

/-- The total number of candies in the jar -/
def CandyJar.total (jar : CandyJar) : ℕ :=
  jar.orange + jar.purple + jar.white + jar.green + jar.black

/-- Replaces a third of purple candies with white candies -/
def CandyJar.replacePurpleWithWhite (jar : CandyJar) : CandyJar :=
  { jar with
    purple := jar.purple - (jar.purple / 3)
    white := jar.white + (jar.purple / 3)
  }

theorem candy_jar_theorem (jar : CandyJar) :
  jar.total = 100 ∧
  jar.orange = 40 ∧
  jar.purple = 30 ∧
  jar.white = 20 ∧
  jar.green = 10 ∧
  jar.black = 10 →
  (jar.replacePurpleWithWhite).white = 30 := by
  sorry


end NUMINAMATH_CALUDE_candy_jar_theorem_l3959_395970


namespace NUMINAMATH_CALUDE_cyclic_inequality_l3959_395987

theorem cyclic_inequality (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_abc : a + b + c = 3) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) + 
  Real.sqrt 2 * (Real.sqrt (a / (b + c)) + Real.sqrt (b / (c + a)) + Real.sqrt (c / (a + b))) 
  ≥ 9/2 := by
sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l3959_395987


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l3959_395921

def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, p a b c x = p a b c (15 - x)) →
  p a b c 4 = -4 →
  p a b c 11 = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l3959_395921


namespace NUMINAMATH_CALUDE_triangle_longest_side_l3959_395900

theorem triangle_longest_side (x : ℝ) : 
  9 + (2 * x + 3) + (3 * x - 2) = 45 →
  max 9 (max (2 * x + 3) (3 * x - 2)) = 19 := by
sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l3959_395900


namespace NUMINAMATH_CALUDE_charity_raffle_winnings_l3959_395989

theorem charity_raffle_winnings (winnings : ℝ) : 
  (winnings / 2 - 2 = 55) → winnings = 114 := by
  sorry

end NUMINAMATH_CALUDE_charity_raffle_winnings_l3959_395989


namespace NUMINAMATH_CALUDE_whale_weight_precision_l3959_395923

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : Float
  exponent : Int

/-- Represents the level of precision for a number -/
inductive Precision
  | Hundreds
  | Thousands
  | TenThousands
  | HundredThousands

/-- Determines the precision of a number in scientific notation -/
def getPrecision (n : ScientificNotation) : Precision :=
  sorry

/-- The approximate weight of the whale in scientific notation -/
def whaleWeight : ScientificNotation :=
  { coefficient := 1.36, exponent := 5 }

/-- Theorem stating that the whale weight is precise to the thousands place -/
theorem whale_weight_precision :
  getPrecision whaleWeight = Precision.Thousands :=
sorry

end NUMINAMATH_CALUDE_whale_weight_precision_l3959_395923


namespace NUMINAMATH_CALUDE_dividing_line_coefficients_l3959_395926

/-- A region formed by nine unit circles tightly packed in the first quadrant -/
def Region : Set (ℝ × ℝ) := sorry

/-- A line with slope 2 that divides the region into two equal-area parts -/
def dividingLine : Set (ℝ × ℝ) := sorry

/-- The coefficients of the line equation ax = by + c -/
def lineCoefficients : ℕ × ℕ × ℕ := sorry

theorem dividing_line_coefficients :
  ∀ (a b c : ℕ),
    lineCoefficients = (a, b, c) →
    dividingLine = {(x, y) | a * x = b * y + c} →
    (∀ (x y : ℝ), (x, y) ∈ dividingLine → y = 2 * x) →
    Nat.gcd a (Nat.gcd b c) = 1 →
    a^2 + b^2 + c^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dividing_line_coefficients_l3959_395926


namespace NUMINAMATH_CALUDE_combined_cost_price_theorem_l3959_395957

def stock_price_1 : ℝ := 100
def stock_price_2 : ℝ := 150
def stock_price_3 : ℝ := 200

def discount_1 : ℝ := 0.06
def discount_2 : ℝ := 0.10
def discount_3 : ℝ := 0.07

def brokerage_1 : ℝ := 0.015
def brokerage_2 : ℝ := 0.02
def brokerage_3 : ℝ := 0.025

def taxation_rate : ℝ := 0.15

def combined_cost_price : ℝ :=
  let discounted_price_1 := stock_price_1 * (1 - discount_1)
  let discounted_price_2 := stock_price_2 * (1 - discount_2)
  let discounted_price_3 := stock_price_3 * (1 - discount_3)
  let cost_price_1 := discounted_price_1 * (1 + brokerage_1)
  let cost_price_2 := discounted_price_2 * (1 + brokerage_2)
  let cost_price_3 := discounted_price_3 * (1 + brokerage_3)
  let total_investing_amount := cost_price_1 + cost_price_2 + cost_price_3
  total_investing_amount * (1 + taxation_rate)

theorem combined_cost_price_theorem : combined_cost_price = 487.324 := by
  sorry

end NUMINAMATH_CALUDE_combined_cost_price_theorem_l3959_395957


namespace NUMINAMATH_CALUDE_limit_x2y_over_x2_plus_y2_is_zero_l3959_395952

open Real

/-- The limit of (x^2 * y) / (x^2 + y^2) as x and y approach 0 is 0. -/
theorem limit_x2y_over_x2_plus_y2_is_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    0 < Real.sqrt (x^2 + y^2) ∧ Real.sqrt (x^2 + y^2) < δ →
    |x^2 * y / (x^2 + y^2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_x2y_over_x2_plus_y2_is_zero_l3959_395952


namespace NUMINAMATH_CALUDE_helium_pressure_change_l3959_395976

/-- Boyle's Law for ideal gases at constant temperature -/
axiom boyles_law {V1 P1 V2 P2 : ℝ} (hV1 : V1 > 0) (hP1 : P1 > 0) (hV2 : V2 > 0) (hP2 : P2 > 0) :
  V1 * P1 = V2 * P2

theorem helium_pressure_change (V1 P1 V2 P2 : ℝ) 
  (hV1 : V1 = 3.4) (hP1 : P1 = 8) (hV2 : V2 = 8.5) 
  (hV1pos : V1 > 0) (hP1pos : P1 > 0) (hV2pos : V2 > 0) (hP2pos : P2 > 0) :
  P2 = 3.2 := by
  sorry

#check helium_pressure_change

end NUMINAMATH_CALUDE_helium_pressure_change_l3959_395976


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l3959_395948

theorem complementary_angles_difference (x : ℝ) (h1 : x > 0) (h2 : 3*x + x = 90) : |3*x - x| = 45 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l3959_395948


namespace NUMINAMATH_CALUDE_john_total_spent_l3959_395943

/-- Calculates the total amount spent by John in USD -/
def total_spent (umbrella_price : ℝ) (raincoat_price : ℝ) (bag_price : ℝ)
                (umbrella_count : ℕ) (raincoat_count : ℕ) (bag_count : ℕ)
                (umbrella_raincoat_discount : ℝ) (bag_discount : ℝ)
                (refund_percentage : ℝ) (initial_conversion_rate : ℝ)
                (refund_conversion_rate : ℝ) : ℝ :=
  sorry

theorem john_total_spent :
  total_spent 8 15 25 2 3 1 0.1 0.05 0.8 1.15 1.17 = 77.81 := by
  sorry

end NUMINAMATH_CALUDE_john_total_spent_l3959_395943


namespace NUMINAMATH_CALUDE_red_apples_count_l3959_395983

def basket_problem (total_apples green_apples : ℕ) : Prop :=
  total_apples = 9 ∧ green_apples = 2 → total_apples - green_apples = 7

theorem red_apples_count : basket_problem 9 2 := by
  sorry

end NUMINAMATH_CALUDE_red_apples_count_l3959_395983


namespace NUMINAMATH_CALUDE_f_value_at_one_l3959_395949

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 2*x + 15
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 120*x + c

-- State the theorem
theorem f_value_at_one (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0 ∧
    f b c r₁ = 0 ∧ f b c r₂ = 0 ∧ f b c r₃ = 0) →
  f b c 1 = -3682.25 :=
sorry

end NUMINAMATH_CALUDE_f_value_at_one_l3959_395949


namespace NUMINAMATH_CALUDE_diplomats_not_speaking_russian_l3959_395982

theorem diplomats_not_speaking_russian (total : ℕ) (french : ℕ) (both_percent : ℚ) (neither_percent : ℚ) 
  (h_total : total = 70)
  (h_french : french = 25)
  (h_both : both_percent = 1/10)
  (h_neither : neither_percent = 1/5) : 
  total - (total : ℚ) * (1 - neither_percent) + french - total * both_percent = 39 := by
  sorry

end NUMINAMATH_CALUDE_diplomats_not_speaking_russian_l3959_395982


namespace NUMINAMATH_CALUDE_compute_expression_l3959_395991

theorem compute_expression : 15 * (30 / 6)^2 = 375 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l3959_395991


namespace NUMINAMATH_CALUDE_percentage_of_boys_with_dogs_l3959_395995

/-- Proves that 10% of boys have dogs at home given the conditions of the problem -/
theorem percentage_of_boys_with_dogs (total_students : ℕ) (girls_with_dogs : ℕ) (total_with_dogs : ℕ) :
  total_students = 100 →
  girls_with_dogs = (20 * (total_students / 2)) / 100 →
  total_with_dogs = 15 →
  (total_with_dogs - girls_with_dogs) * 100 / (total_students / 2) = 10 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_boys_with_dogs_l3959_395995


namespace NUMINAMATH_CALUDE_deck_restoration_l3959_395911

/-- Represents a cut operation on a deck of cards -/
def cut (n : ℕ) (deck : List ℕ) : List ℕ := sorry

/-- Represents the composition of multiple cuts -/
def compose_cuts (cuts : List ℕ) (deck : List ℕ) : List ℕ := sorry

theorem deck_restoration (x : ℕ) :
  let deck := List.range 52
  let cuts := [28, 31, 2, x, 21]
  compose_cuts cuts deck = deck →
  x = 22 := by sorry

end NUMINAMATH_CALUDE_deck_restoration_l3959_395911


namespace NUMINAMATH_CALUDE_power_of_three_remainder_l3959_395984

theorem power_of_three_remainder (k : ℕ) : (3 ^ (4 * k + 3)) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_remainder_l3959_395984


namespace NUMINAMATH_CALUDE_doraemon_dorayakis_l3959_395977

/-- Represents the possible moves in rock-paper-scissors game -/
inductive Move
| Rock
| Scissors

/-- Represents the outcome of a single round -/
inductive Outcome
| Win
| Lose
| Tie

/-- Calculates the outcome of a round given two moves -/
def roundOutcome (move1 move2 : Move) : Outcome :=
  match move1, move2 with
  | Move.Rock, Move.Scissors => Outcome.Win
  | Move.Scissors, Move.Rock => Outcome.Lose
  | _, _ => Outcome.Tie

/-- Calculates the number of dorayakis received based on the outcome -/
def dorayakisForOutcome (outcome : Outcome) : Nat :=
  match outcome with
  | Outcome.Win => 2
  | Outcome.Lose => 0
  | Outcome.Tie => 1

/-- Represents a player's strategy -/
structure Strategy where
  move : Nat → Move

/-- Doraemon's strategy of always playing Rock -/
def doraemonStrategy : Strategy :=
  { move := λ _ => Move.Rock }

/-- Nobita's strategy of playing Scissors once every 10 rounds, Rock otherwise -/
def nobitaStrategy : Strategy :=
  { move := λ round => if round % 10 == 0 then Move.Scissors else Move.Rock }

/-- Calculates the total dorayakis received by a player over multiple rounds -/
def totalDorayakis (playerStrategy opponentStrategy : Strategy) (rounds : Nat) : Nat :=
  (List.range rounds).foldl (λ acc round =>
    acc + dorayakisForOutcome (roundOutcome (playerStrategy.move round) (opponentStrategy.move round))
  ) 0

theorem doraemon_dorayakis :
  totalDorayakis doraemonStrategy nobitaStrategy 20 = 10 ∧
  totalDorayakis nobitaStrategy doraemonStrategy 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_doraemon_dorayakis_l3959_395977


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3959_395978

/-- The sum of an infinite geometric series with first term 1 and common ratio 1/5 is 5/4 -/
theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/5
  let S : ℝ := ∑' n, a * r^n
  S = 5/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3959_395978


namespace NUMINAMATH_CALUDE_triangle_solution_l3959_395953

theorem triangle_solution (a b c : ℝ) (A B C : ℝ) : 
  a = 42 →
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  C = π - A - B →
  b = a * Real.sin B / Real.sin A →
  c = a * Real.sin C / Real.sin A →
  b = 21 * Real.sqrt 6 ∧ c = 21 * (Real.sqrt 3 + 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_solution_l3959_395953


namespace NUMINAMATH_CALUDE_fraction_equality_l3959_395945

theorem fraction_equality : (2023^2 - 2016^2) / (2042^2 - 1997^2) = 7 / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3959_395945


namespace NUMINAMATH_CALUDE_strictly_increasing_implies_a_geq_one_l3959_395969

/-- A function f(x) = x^3 - 2x^2 + ax + 3 is strictly increasing on the interval [1, 2] -/
def StrictlyIncreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x < y → f x < f y

/-- The function f(x) = x^3 - 2x^2 + ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x + 3

theorem strictly_increasing_implies_a_geq_one (a : ℝ) :
  StrictlyIncreasing (f a) a → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_strictly_increasing_implies_a_geq_one_l3959_395969


namespace NUMINAMATH_CALUDE_raw_materials_cost_calculation_raw_materials_cost_value_l3959_395961

/-- The amount Kanul spent on raw materials -/
def raw_materials_cost : ℝ := sorry

/-- The total amount Kanul had -/
def total_amount : ℝ := 5555.56

/-- The amount Kanul spent on machinery -/
def machinery_cost : ℝ := 2000

/-- The amount Kanul kept as cash -/
def cash : ℝ := 0.1 * total_amount

theorem raw_materials_cost_calculation : 
  raw_materials_cost = total_amount - machinery_cost - cash := by sorry

theorem raw_materials_cost_value : 
  raw_materials_cost = 3000 := by sorry

end NUMINAMATH_CALUDE_raw_materials_cost_calculation_raw_materials_cost_value_l3959_395961


namespace NUMINAMATH_CALUDE_triangle_third_angle_l3959_395913

theorem triangle_third_angle (a b : ℝ) (ha : a = 115) (hb : b = 30) : 180 - a - b = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_angle_l3959_395913


namespace NUMINAMATH_CALUDE_arrangement_from_combination_l3959_395990

theorem arrangement_from_combination (n : ℕ) (h1 : n ≥ 2) (h2 : Nat.choose n 2 = 15) : 
  n * (n - 1) = 30 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_from_combination_l3959_395990


namespace NUMINAMATH_CALUDE_hot_dog_packaging_l3959_395942

theorem hot_dog_packaging :
  let total_hot_dogs : ℕ := 25197625
  let package_size : ℕ := 5
  let full_sets : ℕ := 5039525
  total_hot_dogs / package_size = full_sets ∧
  total_hot_dogs % package_size = 0 := by
sorry

end NUMINAMATH_CALUDE_hot_dog_packaging_l3959_395942


namespace NUMINAMATH_CALUDE_bird_migration_distance_l3959_395910

/-- The combined distance traveled by a group of birds migrating between three lakes over two seasons. -/
def combined_distance (num_birds : ℕ) (distance1 : ℝ) (distance2 : ℝ) : ℝ :=
  num_birds * (distance1 + distance2)

/-- Theorem: The combined distance traveled by 20 birds over two seasons between three lakes is 2200 miles. -/
theorem bird_migration_distance :
  combined_distance 20 50 60 = 2200 := by
  sorry

end NUMINAMATH_CALUDE_bird_migration_distance_l3959_395910


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3959_395946

theorem complex_fraction_simplification (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 3) * (x - 4) * (x - 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3959_395946


namespace NUMINAMATH_CALUDE_fifth_month_sales_l3959_395909

def sales_1 : ℕ := 5124
def sales_2 : ℕ := 5366
def sales_3 : ℕ := 5808
def sales_4 : ℕ := 5399
def sales_6 : ℕ := 4579
def average_sale : ℕ := 5400
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sale ∧
    sales_5 = 6124 :=
by sorry

end NUMINAMATH_CALUDE_fifth_month_sales_l3959_395909


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l3959_395933

/-- Given a function f(x) = x^5 + ax^3 + bx - 8, if f(-2) = 10, then f(2) = -26 -/
theorem polynomial_symmetry (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^5 + a*x^3 + b*x - 8
  f (-2) = 10 → f 2 = -26 := by
sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l3959_395933


namespace NUMINAMATH_CALUDE_fred_seashell_count_l3959_395904

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The difference between Fred's and Tom's seashell counts -/
def fred_tom_difference : ℕ := 28

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := tom_seashells + fred_tom_difference

theorem fred_seashell_count : fred_seashells = 43 := by
  sorry

end NUMINAMATH_CALUDE_fred_seashell_count_l3959_395904


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3959_395924

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n where a_4 = 4, prove that a_2 * a_6 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geo : IsGeometricSequence a) (h_a4 : a 4 = 4) :
  a 2 * a 6 = 16 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l3959_395924


namespace NUMINAMATH_CALUDE_competitive_examination_selection_l3959_395925

theorem competitive_examination_selection (total_candidates : ℕ) 
  (selection_rate_A : ℚ) (selection_rate_B : ℚ) : 
  total_candidates = 8100 → 
  selection_rate_A = 6 / 100 → 
  selection_rate_B = 7 / 100 → 
  (selection_rate_B - selection_rate_A) * total_candidates = 81 := by
  sorry

end NUMINAMATH_CALUDE_competitive_examination_selection_l3959_395925


namespace NUMINAMATH_CALUDE_move_right_three_units_l3959_395962

/-- Represents a point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point horizontally by a given distance -/
def moveHorizontal (p : Point) (distance : ℝ) : Point :=
  { x := p.x + distance, y := p.y }

theorem move_right_three_units :
  let P : Point := { x := -2, y := -3 }
  let Q : Point := moveHorizontal P 3
  Q.x = 1 ∧ Q.y = -3 := by
  sorry

end NUMINAMATH_CALUDE_move_right_three_units_l3959_395962


namespace NUMINAMATH_CALUDE_sum_of_sequences_l3959_395914

def sequence1 : List ℕ := [3, 13, 23, 33, 43]
def sequence2 : List ℕ := [11, 21, 31, 41, 51]

theorem sum_of_sequences : (sequence1.sum + sequence2.sum) = 270 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l3959_395914


namespace NUMINAMATH_CALUDE_angle_equality_l3959_395956

theorem angle_equality (C : Real) (h1 : 0 < C) (h2 : C < π) 
  (h3 : Real.cos C = Real.sin C) : C = π/4 ∨ C = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l3959_395956


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l3959_395903

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l3959_395903
