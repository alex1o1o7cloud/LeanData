import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_minimum_l3703_370370

def f (x : ℝ) := x^2 - 12*x + 28

theorem quadratic_minimum (x : ℝ) : f x ≥ f 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3703_370370


namespace NUMINAMATH_CALUDE_factorization_equality_l3703_370354

theorem factorization_equality (m n : ℝ) : 2 * m^2 * n - 8 * m * n + 8 * n = 2 * n * (m - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3703_370354


namespace NUMINAMATH_CALUDE_sarah_bowled_160_l3703_370384

def sarahs_score (gregs_score : ℕ) : ℕ := gregs_score + 60

theorem sarah_bowled_160 (gregs_score : ℕ) :
  sarahs_score gregs_score = 160 ∧ 
  (sarahs_score gregs_score + gregs_score) / 2 = 130 :=
by sorry

end NUMINAMATH_CALUDE_sarah_bowled_160_l3703_370384


namespace NUMINAMATH_CALUDE_identical_views_sphere_or_cube_l3703_370331

-- Define a type for solids
structure Solid where
  -- Add any necessary properties

-- Define a function to represent the view of a solid
def view (s : Solid) : Set Point := sorry

-- Define spheres and cubes as specific types of solids
def Sphere : Solid := sorry
def Cube : Solid := sorry

-- Theorem stating that a solid with three identical views could be a sphere or a cube
theorem identical_views_sphere_or_cube (s : Solid) :
  (∃ v : Set Point, view s = v ∧ view s = v ∧ view s = v) →
  s = Sphere ∨ s = Cube :=
sorry

end NUMINAMATH_CALUDE_identical_views_sphere_or_cube_l3703_370331


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3703_370307

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3703_370307


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3703_370363

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((b + c) * (a^4 - b^2 * c^2)) / (a*b + 2*b*c + c*a) +
  ((c + a) * (b^4 - c^2 * a^2)) / (b*c + 2*c*a + a*b) +
  ((a + b) * (c^4 - a^2 * b^2)) / (c*a + 2*a*b + b*c) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3703_370363


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_seven_thirds_l3703_370371

theorem negative_sixty_four_to_seven_thirds : (-64 : ℝ) ^ (7/3) = -65536 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_seven_thirds_l3703_370371


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l3703_370383

/-- Given that Jason initially has 3 Pokemon cards and Benny buys 2 of them,
    prove that Jason will have 1 Pokemon card left. -/
theorem jason_pokemon_cards (initial_cards : ℕ) (cards_bought : ℕ) 
  (h1 : initial_cards = 3)
  (h2 : cards_bought = 2) :
  initial_cards - cards_bought = 1 := by
  sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l3703_370383


namespace NUMINAMATH_CALUDE_calculation_proof_l3703_370378

theorem calculation_proof :
  (1) * (Real.sqrt 2 + 2)^2 = 6 + 4 * Real.sqrt 2 ∧
  (2) * (Real.sqrt 3 - Real.sqrt 8) - (1/2) * (Real.sqrt 18 + Real.sqrt 12) = -(7/2) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3703_370378


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3703_370374

/-- The eccentricity of a hyperbola with given equation and asymptotes -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (heq : ∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1 ↔ (y = x / 2 ∨ y = -x / 2)) :
  let e := Real.sqrt (a^2 + b^2) / a
  e = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3703_370374


namespace NUMINAMATH_CALUDE_mole_winter_survival_l3703_370359

/-- Represents the Mole's food storage --/
structure MoleStorage :=
  (grain : ℕ)
  (millet : ℕ)

/-- Represents a monthly consumption plan --/
inductive ConsumptionPlan
  | AllGrain
  | MixedDiet

/-- The Mole's winter survival problem --/
theorem mole_winter_survival 
  (initial_grain : ℕ)
  (storage_capacity : ℕ)
  (exchange_rate : ℕ)
  (winter_duration : ℕ)
  (h_initial_grain : initial_grain = 8)
  (h_storage_capacity : storage_capacity = 12)
  (h_exchange_rate : exchange_rate = 2)
  (h_winter_duration : winter_duration = 3)
  : ∃ (exchange_amount : ℕ) 
      (final_storage : MoleStorage) 
      (consumption_plan : Fin winter_duration → ConsumptionPlan),
    -- Exchange constraint
    exchange_amount ≤ initial_grain ∧
    -- Storage capacity constraint
    final_storage.grain + final_storage.millet ≤ storage_capacity ∧
    -- Exchange calculation
    final_storage.grain = initial_grain - exchange_amount ∧
    final_storage.millet = exchange_amount * exchange_rate ∧
    -- Survival constraint
    (∀ month : Fin winter_duration,
      (consumption_plan month = ConsumptionPlan.AllGrain → 
        ∃ remaining_storage : MoleStorage,
          remaining_storage.grain = final_storage.grain - 3 * (month.val + 1) ∧
          remaining_storage.millet = final_storage.millet) ∧
      (consumption_plan month = ConsumptionPlan.MixedDiet →
        ∃ remaining_storage : MoleStorage,
          remaining_storage.grain = final_storage.grain - (month.val + 1) ∧
          remaining_storage.millet = final_storage.millet - 3 * (month.val + 1))) ∧
    -- Final state
    ∃ final_state : MoleStorage,
      final_state.grain = 0 ∧ final_state.millet = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_mole_winter_survival_l3703_370359


namespace NUMINAMATH_CALUDE_polynomial_equality_l3703_370355

theorem polynomial_equality (x : ℝ) : ∃ (t a b : ℝ),
  (3 * x^2 - 4 * x + 5) * (5 * x^2 + t * x + 12) = 15 * x^4 - 47 * x^3 + a * x^2 + b * x + 60 ∧
  t = -9 ∧ a = -53 ∧ b = -156 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3703_370355


namespace NUMINAMATH_CALUDE_arrangements_count_is_correct_l3703_370385

/-- The number of arrangements of 4 boys and 3 girls in a row,
    where exactly two girls are standing next to each other. -/
def arrangements_count : ℕ := 2880

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 3

/-- Theorem stating that the number of arrangements of 4 boys and 3 girls in a row,
    where exactly two girls are standing next to each other, is equal to 2880. -/
theorem arrangements_count_is_correct :
  arrangements_count = num_girls * (num_girls - 1) / 2 * 
    (num_boys * (num_boys - 1) * (num_boys - 2) * (num_boys - 3)) *
    ((num_boys + 1) * num_boys) :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_is_correct_l3703_370385


namespace NUMINAMATH_CALUDE_equation_solution_set_l3703_370325

theorem equation_solution_set : 
  let f : ℝ → ℝ := λ x => 1 / (x^2 + 13*x - 12) + 1 / (x^2 + 4*x - 12) + 1 / (x^2 - 15*x - 12)
  {x : ℝ | f x = 0} = {1, -12, 12, -1} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_set_l3703_370325


namespace NUMINAMATH_CALUDE_rally_ticket_cost_l3703_370376

theorem rally_ticket_cost 
  (total_attendance : ℕ)
  (door_ticket_price : ℚ)
  (total_receipts : ℚ)
  (pre_rally_tickets : ℕ)
  (h1 : total_attendance = 750)
  (h2 : door_ticket_price = 2.75)
  (h3 : total_receipts = 1706.25)
  (h4 : pre_rally_tickets = 475) :
  ∃ (pre_rally_price : ℚ), 
    pre_rally_price * pre_rally_tickets + 
    door_ticket_price * (total_attendance - pre_rally_tickets) = total_receipts ∧
    pre_rally_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_rally_ticket_cost_l3703_370376


namespace NUMINAMATH_CALUDE_exists_decreasing_linear_function_through_origin_l3703_370358

/-- A linear function that decreases and passes through (0,2) -/
def decreasingLinearFunction (k : ℝ) (x : ℝ) : ℝ := k * x + 2

theorem exists_decreasing_linear_function_through_origin :
  ∃ (k : ℝ), k < 0 ∧
    (∀ (x y : ℝ), x < y → decreasingLinearFunction k x > decreasingLinearFunction k y) ∧
    decreasingLinearFunction k 0 = 2 :=
sorry

end NUMINAMATH_CALUDE_exists_decreasing_linear_function_through_origin_l3703_370358


namespace NUMINAMATH_CALUDE_intersection_A_B_l3703_370347

-- Define set A
def A : Set ℝ := {x | x ≥ 1}

-- Define set B
def B : Set ℝ := {y | y ≤ 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3703_370347


namespace NUMINAMATH_CALUDE_plane_equation_satisfies_conditions_l3703_370390

/-- A plane in 3D space represented by its equation coefficients -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Check if two planes are parallel -/
def planesParallel (plane1 plane2 : Plane) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ plane1.a = k * plane2.a ∧ plane1.b = k * plane2.b ∧ plane1.c = k * plane2.c

theorem plane_equation_satisfies_conditions : 
  let plane := Plane.mk 24 20 (-16) (-20)
  let point1 := Point3D.mk 2 (-1) 3
  let point2 := Point3D.mk 1 3 (-4)
  let parallelPlane := Plane.mk 3 (-4) 1 (-5)
  pointOnPlane plane point1 ∧ 
  pointOnPlane plane point2 ∧ 
  planesParallel plane parallelPlane :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_satisfies_conditions_l3703_370390


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3703_370353

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3703_370353


namespace NUMINAMATH_CALUDE_triangle_properties_l3703_370311

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle satisfying certain conditions -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.a + t.c) * Real.cos t.B + t.b * Real.cos t.C = 0)
  (h2 : t.b = Real.sqrt 13)
  (h3 : t.a + t.c = 4) :
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3703_370311


namespace NUMINAMATH_CALUDE_system_solution_l3703_370300

theorem system_solution :
  ∃ (a b c d : ℝ),
    (a * b + c + d = 3) ∧
    (b * c + d + a = 5) ∧
    (c * d + a + b = 2) ∧
    (d * a + b + c = 6) ∧
    (a = 2 ∧ b = 0 ∧ c = 0 ∧ d = 3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3703_370300


namespace NUMINAMATH_CALUDE_remainder_385857_div_6_l3703_370314

theorem remainder_385857_div_6 : 385857 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_385857_div_6_l3703_370314


namespace NUMINAMATH_CALUDE_square_of_sum_m_plus_two_n_l3703_370348

theorem square_of_sum_m_plus_two_n (m n : ℝ) : (m + 2*n)^2 = m^2 + 4*n^2 + 4*m*n := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_m_plus_two_n_l3703_370348


namespace NUMINAMATH_CALUDE_class_size_ratio_l3703_370339

theorem class_size_ratio : 
  let finley_class_size : ℕ := 24
  let johnson_class_size : ℕ := 22
  let half_finley_class_size : ℚ := (finley_class_size : ℚ) / 2
  (johnson_class_size : ℚ) / half_finley_class_size = 11 / 6 := by sorry

end NUMINAMATH_CALUDE_class_size_ratio_l3703_370339


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l3703_370337

theorem complex_arithmetic_equality : (80 + (5 * 12) / (180 / 3)) ^ 2 * (7 - (3^2)) = -13122 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l3703_370337


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l3703_370333

/-- The perimeter of a semicircle with radius 6.7 cm is equal to π * 6.7 + 13.4 cm. -/
theorem semicircle_perimeter : 
  let r : ℝ := 6.7
  ∀ π : ℝ, π * r + 2 * r = π * r + 13.4 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l3703_370333


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3703_370386

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 150 → volume = (surface_area / 6) ^ (3/2) → volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3703_370386


namespace NUMINAMATH_CALUDE_chord_equation_of_ellipse_l3703_370373

/-- Given an ellipse and a point that bisects a chord of the ellipse, 
    prove the equation of the line containing the chord. -/
theorem chord_equation_of_ellipse (x y : ℝ → ℝ) :
  (∀ t, (x t)^2 / 36 + (y t)^2 / 9 = 1) →  -- Ellipse equation
  (∃ t₁ t₂, t₁ ≠ t₂ ∧ 
    (x t₁ + x t₂) / 2 = 4 ∧ 
    (y t₁ + y t₂) / 2 = 2) →  -- Midpoint condition
  (∃ A B : ℝ, ∀ t, A * (x t) + B * (y t) = 8) →  -- Line equation
  A = 1 ∧ B = 2 := by
sorry

end NUMINAMATH_CALUDE_chord_equation_of_ellipse_l3703_370373


namespace NUMINAMATH_CALUDE_triangle_side_length_l3703_370340

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the angles and side length
def angle_X (t : Triangle) : ℝ := sorry
def angle_Y (t : Triangle) : ℝ := sorry
def length_XZ (t : Triangle) : ℝ := sorry
def length_YZ (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_side_length (t : Triangle) :
  angle_X t = π / 4 →  -- 45°
  angle_Y t = π / 3 →  -- 60°
  length_XZ t = 6 * Real.sqrt 3 →
  length_YZ t = 6 * Real.sqrt (2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3703_370340


namespace NUMINAMATH_CALUDE_smallest_a_for_nonprime_polynomial_l3703_370304

theorem smallest_a_for_nonprime_polynomial : ∃ (a : ℕ), a > 0 ∧
  (∀ (x : ℤ), ¬ Prime (x^4 + a^3)) ∧
  (∀ (b : ℕ), b > 0 ∧ b < a → ∃ (x : ℤ), Prime (x^4 + b^3)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_nonprime_polynomial_l3703_370304


namespace NUMINAMATH_CALUDE_simplify_expression_l3703_370342

theorem simplify_expression (x : ℝ) : (3*x - 6)*(x + 9) - (x + 6)*(3*x - 2) = 5*x - 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3703_370342


namespace NUMINAMATH_CALUDE_x_value_in_sequence_l3703_370398

def fibonacci_like_sequence (a : ℤ → ℤ) : Prop :=
  ∀ n, a (n + 2) = a (n + 1) + a n

theorem x_value_in_sequence (a : ℤ → ℤ) :
  fibonacci_like_sequence a →
  a 3 = 10 →
  a 4 = 5 →
  a 5 = 15 →
  a 6 = 20 →
  a 7 = 35 →
  a 8 = 55 →
  a 9 = 90 →
  a 0 = -20 :=
by
  sorry

end NUMINAMATH_CALUDE_x_value_in_sequence_l3703_370398


namespace NUMINAMATH_CALUDE_polynomial_ratio_theorem_l3703_370395

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^2009 - 19*x^2008 + 1

-- Define the set of distinct zeros of f
def zeros (f : ℝ → ℝ) : Set ℝ := {x | f x = 0}

-- Define the polynomial P
def P (z : ℝ) : ℝ := sorry

-- Theorem statement
theorem polynomial_ratio_theorem 
  (h1 : ∀ r ∈ zeros f, P (r - 1/r) = 0) 
  (h2 : Fintype (zeros f)) 
  (h3 : Fintype.card (zeros f) = 2009) :
  P 2 / P (-2) = 36 / 49 := by sorry

end NUMINAMATH_CALUDE_polynomial_ratio_theorem_l3703_370395


namespace NUMINAMATH_CALUDE_isosceles_triangle_areas_l3703_370315

theorem isosceles_triangle_areas (W X Y : ℝ) : 
  (W = (5 * 5) / 2) →
  (X = (12 * 12) / 2) →
  (Y = (13 * 13) / 2) →
  (X + Y ≠ 2 * W + X) ∧
  (W + X ≠ Y) ∧
  (2 * X ≠ W + Y) ∧
  (X + W ≠ X) ∧
  (W + Y ≠ 2 * X) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_areas_l3703_370315


namespace NUMINAMATH_CALUDE_knights_count_l3703_370303

/-- Represents the statement made by the i-th person on the island -/
def statement (i : ℕ) (num_knights : ℕ) : Prop :=
  num_knights ∣ i

/-- Represents whether a person at position i is telling the truth -/
def is_truthful (i : ℕ) (num_knights : ℕ) : Prop :=
  statement i num_knights

/-- The total number of inhabitants on the island -/
def total_inhabitants : ℕ := 100

/-- Theorem stating that the only possible numbers of knights are 0 and 10 -/
theorem knights_count : 
  ∃ (num_knights : ℕ), 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ total_inhabitants → 
      (is_truthful i num_knights ↔ i % num_knights = 0)) ∧
    (num_knights = 0 ∨ num_knights = 10) :=
sorry

end NUMINAMATH_CALUDE_knights_count_l3703_370303


namespace NUMINAMATH_CALUDE_mouse_jump_distance_l3703_370352

/-- The jumping distances of animals in a contest -/
def jumping_contest (grasshopper frog mouse : ℕ) : Prop :=
  grasshopper = 39 ∧ 
  grasshopper = frog + 19 ∧ 
  mouse + 12 = frog

theorem mouse_jump_distance :
  ∀ grasshopper frog mouse : ℕ, 
  jumping_contest grasshopper frog mouse → 
  mouse = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_mouse_jump_distance_l3703_370352


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_integers_l3703_370324

/-- Given three consecutive odd integers where the largest is -47, their sum is -141 -/
theorem sum_of_three_consecutive_odd_integers :
  ∀ (a b c : ℤ),
  (a < b ∧ b < c) →                   -- a, b, c are in ascending order
  (∃ k : ℤ, a = 2*k + 1) →            -- a is odd
  (∃ k : ℤ, b = 2*k + 1) →            -- b is odd
  (∃ k : ℤ, c = 2*k + 1) →            -- c is odd
  (b = a + 2) →                       -- b is the next consecutive odd integer after a
  (c = b + 2) →                       -- c is the next consecutive odd integer after b
  (c = -47) →                         -- the largest number is -47
  (a + b + c = -141) :=               -- their sum is -141
by sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_integers_l3703_370324


namespace NUMINAMATH_CALUDE_f_monotonic_intervals_f_inequality_solution_f_max_value_l3703_370345

-- Define the function f(x) = x|x-2|
def f (x : ℝ) : ℝ := x * abs (x - 2)

-- Theorem for monotonic intervals
theorem f_monotonic_intervals :
  (∀ x y, x ≤ y ∧ y ≤ 1 → f x ≤ f y) ∧
  (∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f x ≥ f y) ∧
  (∀ x y, 2 ≤ x ∧ x ≤ y → f x ≤ f y) :=
sorry

-- Theorem for the inequality solution
theorem f_inequality_solution :
  ∀ x, f x < 3 ↔ x < 3 :=
sorry

-- Theorem for the maximum value
theorem f_max_value (a : ℝ) (h : 0 < a ∧ a ≤ 2) :
  (∀ x, 0 ≤ x ∧ x ≤ a → f x ≤ (if a ≤ 1 then a * (2 - a) else 1)) ∧
  (∃ x, 0 ≤ x ∧ x ≤ a ∧ f x = (if a ≤ 1 then a * (2 - a) else 1)) :=
sorry

end NUMINAMATH_CALUDE_f_monotonic_intervals_f_inequality_solution_f_max_value_l3703_370345


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3703_370335

/-- Given a boat that travels 8 km downstream and 2 km upstream in one hour,
    its speed in still water is 5 km/hr. -/
theorem boat_speed_in_still_water : ∀ (b s : ℝ),
  b + s = 8 →  -- Speed downstream
  b - s = 2 →  -- Speed upstream
  b = 5 :=     -- Speed in still water
by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3703_370335


namespace NUMINAMATH_CALUDE_platform_length_l3703_370380

/-- Calculates the length of a platform given the speed of a train, time to cross the platform, and length of the train. -/
theorem platform_length
  (train_speed_kmh : ℝ)
  (crossing_time_s : ℝ)
  (train_length_m : ℝ)
  (h1 : train_speed_kmh = 72)
  (h2 : crossing_time_s = 26)
  (h3 : train_length_m = 270) :
  let train_speed_ms : ℝ := train_speed_kmh * (5 / 18)
  let total_distance : ℝ := train_speed_ms * crossing_time_s
  let platform_length : ℝ := total_distance - train_length_m
  platform_length = 250 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3703_370380


namespace NUMINAMATH_CALUDE_checker_center_on_boundary_l3703_370322

/-- Represents a circular checker on a checkerboard -/
structure Checker where
  center : ℝ × ℝ
  radius : ℝ
  is_on_board : Bool
  covers_equal_areas : Bool

/-- Represents a checkerboard -/
structure Checkerboard where
  size : ℕ
  square_size : ℝ

/-- Checks if a point is on a boundary or junction of squares -/
def is_on_boundary_or_junction (board : Checkerboard) (point : ℝ × ℝ) : Prop :=
  ∃ (n m : ℕ), (n ≤ board.size ∧ m ≤ board.size) ∧
    (point.1 = n * board.square_size ∨ point.2 = m * board.square_size)

/-- Main theorem -/
theorem checker_center_on_boundary (board : Checkerboard) (c : Checker) :
    c.is_on_board = true → c.covers_equal_areas = true →
    is_on_boundary_or_junction board c.center :=
  sorry


end NUMINAMATH_CALUDE_checker_center_on_boundary_l3703_370322


namespace NUMINAMATH_CALUDE_factor_expression_l3703_370301

theorem factor_expression (x : ℝ) : 16 * x^3 + 4 * x^2 = 4 * x^2 * (4 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3703_370301


namespace NUMINAMATH_CALUDE_min_value_expression_l3703_370305

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c) ≥ 216 ∧
  (∃ a b c, (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c) = 216) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3703_370305


namespace NUMINAMATH_CALUDE_sum_on_real_axis_l3703_370343

theorem sum_on_real_axis (a : ℝ) : 
  let z₁ : ℂ := 2 + I
  let z₂ : ℂ := 3 + a * I
  (z₁ + z₂).im = 0 → a = -1 := by sorry

end NUMINAMATH_CALUDE_sum_on_real_axis_l3703_370343


namespace NUMINAMATH_CALUDE_popping_corn_probability_l3703_370397

theorem popping_corn_probability (white yellow blue : ℝ)
  (white_pop yellow_pop blue_pop : ℝ) :
  white = 1/2 →
  yellow = 1/3 →
  blue = 1/6 →
  white_pop = 3/4 →
  yellow_pop = 1/2 →
  blue_pop = 1/3 →
  (white * white_pop) / (white * white_pop + yellow * yellow_pop + blue * blue_pop) = 27/43 := by
  sorry

end NUMINAMATH_CALUDE_popping_corn_probability_l3703_370397


namespace NUMINAMATH_CALUDE_joes_bath_shop_problem_l3703_370308

theorem joes_bath_shop_problem (bottles_per_box : ℕ) (total_sold : ℕ) 
  (h1 : bottles_per_box = 19)
  (h2 : total_sold = 95)
  (h3 : ∃ (bar_boxes bottle_boxes : ℕ), bar_boxes * total_sold = bottle_boxes * total_sold)
  (h4 : ∀ x : ℕ, x > 1 ∧ x * total_sold = bottles_per_box * total_sold → x ≥ 5) :
  ∃ (bars_per_box : ℕ), bars_per_box > 1 ∧ bars_per_box * total_sold = bottles_per_box * total_sold ∧ bars_per_box = 5 := by
  sorry

end NUMINAMATH_CALUDE_joes_bath_shop_problem_l3703_370308


namespace NUMINAMATH_CALUDE_circles_intersection_triangle_similarity_l3703_370328

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the centers of the circles
variable (O₁ O₂ : Point)

-- Define the circles
variable (Γ₁ Γ₂ : Circle)

-- Define the intersection points
variable (X Y : Point)

-- Define point A on Γ₁
variable (A : Point)

-- Define point B as the intersection of AY and Γ₂
variable (B : Point)

-- Define the property of being on a circle
variable (on_circle : Point → Circle → Prop)

-- Define the property of two circles intersecting
variable (intersect : Circle → Circle → Point → Point → Prop)

-- Define the property of a point being on a line
variable (on_line : Point → Point → Point → Prop)

-- Define the property of triangle similarity
variable (similar_triangles : Point → Point → Point → Point → Point → Point → Prop)

-- State the theorem
theorem circles_intersection_triangle_similarity
  (h1 : on_circle O₁ Γ₁)
  (h2 : on_circle O₂ Γ₂)
  (h3 : intersect Γ₁ Γ₂ X Y)
  (h4 : on_circle A Γ₁)
  (h5 : A ≠ X)
  (h6 : A ≠ Y)
  (h7 : on_line A Y B)
  (h8 : on_circle B Γ₂) :
  similar_triangles X O₁ O₂ X A B :=
sorry

end NUMINAMATH_CALUDE_circles_intersection_triangle_similarity_l3703_370328


namespace NUMINAMATH_CALUDE_park_visitors_total_l3703_370316

theorem park_visitors_total (saturday_visitors : ℕ) (sunday_extra : ℕ) : 
  saturday_visitors = 200 → sunday_extra = 40 → 
  saturday_visitors + (saturday_visitors + sunday_extra) = 440 := by
sorry

end NUMINAMATH_CALUDE_park_visitors_total_l3703_370316


namespace NUMINAMATH_CALUDE_total_revenue_is_3610_l3703_370310

/-- Represents the quantity and price information for a fruit --/
structure Fruit where
  quantity : ℕ
  originalPrice : ℚ
  priceChange : ℚ

/-- Calculates the total revenue for a single fruit type --/
def calculateFruitRevenue (fruit : Fruit) : ℚ :=
  fruit.quantity * (fruit.originalPrice * (1 + fruit.priceChange))

/-- Theorem stating that the total revenue from all fruits is $3610 --/
theorem total_revenue_is_3610 
  (lemons : Fruit)
  (grapes : Fruit)
  (oranges : Fruit)
  (apples : Fruit)
  (kiwis : Fruit)
  (pineapples : Fruit)
  (h1 : lemons = { quantity := 80, originalPrice := 8, priceChange := 0.5 })
  (h2 : grapes = { quantity := 140, originalPrice := 7, priceChange := 0.25 })
  (h3 : oranges = { quantity := 60, originalPrice := 5, priceChange := 0.1 })
  (h4 : apples = { quantity := 100, originalPrice := 4, priceChange := 0.2 })
  (h5 : kiwis = { quantity := 50, originalPrice := 6, priceChange := -0.15 })
  (h6 : pineapples = { quantity := 30, originalPrice := 12, priceChange := 0 }) :
  calculateFruitRevenue lemons + calculateFruitRevenue grapes + 
  calculateFruitRevenue oranges + calculateFruitRevenue apples + 
  calculateFruitRevenue kiwis + calculateFruitRevenue pineapples = 3610 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_is_3610_l3703_370310


namespace NUMINAMATH_CALUDE_rectangular_field_breadth_breadth_approximation_l3703_370369

/-- The breadth of a rectangular field with length 90 meters, 
    whose area is equal to a square plot with diagonal 120 meters. -/
theorem rectangular_field_breadth : ℝ :=
  let rectangular_length : ℝ := 90
  let square_diagonal : ℝ := 120
  let square_side : ℝ := square_diagonal / Real.sqrt 2
  let square_area : ℝ := square_side ^ 2
  let rectangular_area : ℝ := square_area
  rectangular_area / rectangular_length

/-- The breadth of the rectangular field is approximately 80 meters. -/
theorem breadth_approximation (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, abs (rectangular_field_breadth - 80) < δ ∧ δ < ε :=
sorry

end NUMINAMATH_CALUDE_rectangular_field_breadth_breadth_approximation_l3703_370369


namespace NUMINAMATH_CALUDE_series_sum_is_zero_l3703_370392

/-- The sum of the series -1 + 0 + 1 - 2 + 0 + 2 - 3 + 0 + 3 - ... + (-4001) + 0 + 4001 -/
def seriesSum : ℤ := sorry

/-- The number of terms in the series -/
def numTerms : ℕ := 12003

/-- The series ends at 4001 -/
def lastTerm : ℕ := 4001

theorem series_sum_is_zero :
  seriesSum = 0 :=
by sorry

end NUMINAMATH_CALUDE_series_sum_is_zero_l3703_370392


namespace NUMINAMATH_CALUDE_function_periodicity_l3703_370313

/-- Given a > 0 and f satisfying f(x) + f(x+a) + f(x) f(x+a) = 1 for all x,
    prove that f is periodic with period 2a -/
theorem function_periodicity (a : ℝ) (f : ℝ → ℝ) (ha : a > 0)
  (hf : ∀ x : ℝ, f x + f (x + a) + f x * f (x + a) = 1) :
  ∀ x : ℝ, f (x + 2 * a) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l3703_370313


namespace NUMINAMATH_CALUDE_orange_apple_cost_l3703_370356

theorem orange_apple_cost : ∃ (x y : ℚ),
  (7 * x + 5 * y = 13) ∧
  (3 * x + 4 * y = 8) →
  (37 * x + 45 * y = 93) := by
  sorry

end NUMINAMATH_CALUDE_orange_apple_cost_l3703_370356


namespace NUMINAMATH_CALUDE_pokemon_cards_total_l3703_370365

/-- The number of people with Pokemon cards -/
def num_people : ℕ := 4

/-- The number of Pokemon cards each person has -/
def cards_per_person : ℕ := 14

/-- The total number of Pokemon cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem pokemon_cards_total : total_cards = 56 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_total_l3703_370365


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l3703_370346

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 → x % 4 = 0 → x^3 < 8000 → x ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l3703_370346


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_reciprocal_l3703_370349

theorem repeating_decimal_equals_reciprocal (a : ℕ) : 
  (1 ≤ a ∧ a ≤ 9) → 
  ((10 + a - 1) / 90 : ℚ) = 1 / a → 
  a = 6 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_reciprocal_l3703_370349


namespace NUMINAMATH_CALUDE_points_not_on_any_circle_l3703_370312

-- Define the circle equation
def circle_equation (x y α β : ℝ) : Prop :=
  α * ((x - 2)^2 + y^2 - 1) + β * ((x + 2)^2 + y^2 - 1) = 0

-- Define the set of points not on any circle
def points_not_on_circles : Set (ℝ × ℝ) :=
  {p | p.1 = 0 ∨ (p.1 = Real.sqrt 3 ∧ p.2 = 0) ∨ (p.1 = -Real.sqrt 3 ∧ p.2 = 0)}

-- Theorem statement
theorem points_not_on_any_circle :
  ∀ (p : ℝ × ℝ), p ∈ points_not_on_circles →
  ∀ (α β : ℝ), ¬(circle_equation p.1 p.2 α β) :=
by sorry

end NUMINAMATH_CALUDE_points_not_on_any_circle_l3703_370312


namespace NUMINAMATH_CALUDE_dad_steps_l3703_370327

/-- Represents the number of steps taken by each person --/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- The ratio of steps between Dad and Masha --/
def dad_masha_ratio (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha

/-- The ratio of steps between Masha and Yasha --/
def masha_yasha_ratio (s : Steps) : Prop :=
  5 * s.masha = 3 * s.yasha

/-- The total number of steps taken by Masha and Yasha --/
def masha_yasha_total (s : Steps) : Prop :=
  s.masha + s.yasha = 400

/-- The main theorem: Given the conditions, Dad took 90 steps --/
theorem dad_steps (s : Steps) 
  (h1 : dad_masha_ratio s) 
  (h2 : masha_yasha_ratio s) 
  (h3 : masha_yasha_total s) : 
  s.dad = 90 := by
  sorry

end NUMINAMATH_CALUDE_dad_steps_l3703_370327


namespace NUMINAMATH_CALUDE_min_value_of_objective_function_l3703_370321

def objective_function (x y : ℝ) : ℝ := x + 3 * y

def constraint1 (x y : ℝ) : Prop := x + y - 2 ≥ 0
def constraint2 (x y : ℝ) : Prop := x - y - 2 ≤ 0
def constraint3 (y : ℝ) : Prop := y ≥ 1

theorem min_value_of_objective_function :
  ∀ x y : ℝ, constraint1 x y → constraint2 x y → constraint3 y →
  ∀ x' y' : ℝ, constraint1 x' y' → constraint2 x' y' → constraint3 y' →
  objective_function x y ≥ 4 ∧
  (∃ x₀ y₀ : ℝ, constraint1 x₀ y₀ ∧ constraint2 x₀ y₀ ∧ constraint3 y₀ ∧ objective_function x₀ y₀ = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_objective_function_l3703_370321


namespace NUMINAMATH_CALUDE_area_between_circles_first_quadrant_l3703_370377

/-- The area of the region between two concentric circles with radii 15 and 9,
    extending only within the first quadrant, is equal to 36π. -/
theorem area_between_circles_first_quadrant :
  let r₁ : ℝ := 15
  let r₂ : ℝ := 9
  let full_area := π * (r₁^2 - r₂^2)
  let quadrant_area := full_area / 4
  quadrant_area = 36 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_circles_first_quadrant_l3703_370377


namespace NUMINAMATH_CALUDE_minimum_value_sqrt_plus_reciprocal_l3703_370366

theorem minimum_value_sqrt_plus_reciprocal (x : ℝ) (hx : x > 0) :
  3 * Real.sqrt x + 1 / x ≥ 4 ∧
  (3 * Real.sqrt x + 1 / x = 4 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_sqrt_plus_reciprocal_l3703_370366


namespace NUMINAMATH_CALUDE_victoria_beacon_ratio_l3703_370381

/-- The population of Richmond -/
def richmond_population : ℕ := 3000

/-- The population of Beacon -/
def beacon_population : ℕ := 500

/-- The difference between Richmond's and Victoria's populations -/
def richmond_victoria_diff : ℕ := 1000

/-- The population of Victoria -/
def victoria_population : ℕ := richmond_population - richmond_victoria_diff

theorem victoria_beacon_ratio : 
  (victoria_population : ℚ) / (beacon_population : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_victoria_beacon_ratio_l3703_370381


namespace NUMINAMATH_CALUDE_total_problems_l3703_370319

/-- The number of problems Georgia completes in the first 20 minutes -/
def problems_first_20 : ℕ := 10

/-- The number of problems Georgia completes in the second 20 minutes -/
def problems_second_20 : ℕ := 2 * problems_first_20

/-- The number of problems Georgia has left to solve -/
def problems_left : ℕ := 45

/-- Theorem: The total number of problems on the test is 75 -/
theorem total_problems : 
  problems_first_20 + problems_second_20 + problems_left = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_l3703_370319


namespace NUMINAMATH_CALUDE_a_gt_b_relation_l3703_370375

theorem a_gt_b_relation (a b : ℝ) :
  (∀ a b, a - 1 > b + 1 → a > b) ∧
  (∃ a b, a > b ∧ a - 1 ≤ b + 1) :=
sorry

end NUMINAMATH_CALUDE_a_gt_b_relation_l3703_370375


namespace NUMINAMATH_CALUDE_steve_total_cost_theorem_l3703_370364

def steve_total_cost (mike_dvd_price : ℝ) (steve_extra_dvd_price : ℝ) 
  (steve_extra_dvd_count : ℕ) (shipping_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let steve_favorite_dvd_price := 2 * mike_dvd_price
  let steve_extra_dvds_cost := steve_extra_dvd_count * steve_extra_dvd_price
  let total_dvds_cost := steve_favorite_dvd_price + steve_extra_dvds_cost
  let shipping_cost := shipping_rate * total_dvds_cost
  let subtotal := total_dvds_cost + shipping_cost
  let tax := tax_rate * subtotal
  subtotal + tax

theorem steve_total_cost_theorem :
  steve_total_cost 5 7 2 0.8 0.1 = 47.52 := by
  sorry

end NUMINAMATH_CALUDE_steve_total_cost_theorem_l3703_370364


namespace NUMINAMATH_CALUDE_order_of_abc_l3703_370367

-- Define the constants
noncomputable def a : ℝ := Real.log 2 / Real.log 10
noncomputable def b : ℝ := Real.cos 2
noncomputable def c : ℝ := 2 ^ (1 / 5)

-- State the theorem
theorem order_of_abc : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l3703_370367


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l3703_370351

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l3703_370351


namespace NUMINAMATH_CALUDE_congruence_system_solutions_l3703_370332

theorem congruence_system_solutions (a b c : ℤ) : 
  ∃ (s : Finset ℤ), 
    (∀ x ∈ s, x ≥ 0 ∧ x < 2000 ∧ 
      x % 14 = a % 14 ∧ 
      x % 15 = b % 15 ∧ 
      x % 16 = c % 16) ∧
    s.card = 3 :=
by sorry

end NUMINAMATH_CALUDE_congruence_system_solutions_l3703_370332


namespace NUMINAMATH_CALUDE_M_not_subset_P_l3703_370372

-- Define the sets M and P
def M : Set ℝ := {x | x > 1}
def P : Set ℝ := {x | x^2 > 1}

-- State the theorem
theorem M_not_subset_P : ¬(M ⊆ P) := by sorry

end NUMINAMATH_CALUDE_M_not_subset_P_l3703_370372


namespace NUMINAMATH_CALUDE_bug_return_probability_l3703_370361

/-- Represents the probability of the bug being at the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 - Q n) / 2

/-- The probability of returning to the starting vertex on the 12th move in a square -/
theorem bug_return_probability : Q 12 = 683 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l3703_370361


namespace NUMINAMATH_CALUDE_min_difference_in_sample_l3703_370389

theorem min_difference_in_sample (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e →
  c = 12 →
  (a + b + c + d + e) / 5 = 10 →
  e - a ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_difference_in_sample_l3703_370389


namespace NUMINAMATH_CALUDE_num_children_picked_apples_l3703_370344

/-- The number of baskets -/
def num_baskets : ℕ := 11

/-- The sum of apples picked by each child from all baskets -/
def apples_per_child : ℕ := (num_baskets * (num_baskets + 1)) / 2

/-- The total number of apples picked by all children -/
def total_apples_picked : ℕ := 660

/-- Theorem stating that the number of children who picked apples is 10 -/
theorem num_children_picked_apples : 
  total_apples_picked / apples_per_child = 10 := by
  sorry

end NUMINAMATH_CALUDE_num_children_picked_apples_l3703_370344


namespace NUMINAMATH_CALUDE_kitchen_cleaning_time_l3703_370382

theorem kitchen_cleaning_time (alice_time bob_fraction : ℚ) (h1 : alice_time = 40) (h2 : bob_fraction = 3/8) :
  bob_fraction * alice_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_cleaning_time_l3703_370382


namespace NUMINAMATH_CALUDE_real_part_of_z_l3703_370362

theorem real_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2) : z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3703_370362


namespace NUMINAMATH_CALUDE_gcd_105_88_l3703_370320

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_88_l3703_370320


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3703_370399

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 3 ∧ 
  (∀ (m : ℕ), m < n → ¬(15 ∣ (427398 - m))) ∧ 
  (15 ∣ (427398 - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3703_370399


namespace NUMINAMATH_CALUDE_history_class_grades_l3703_370309

theorem history_class_grades (total_students : ℕ) 
  (prob_A : ℚ) (prob_B : ℚ) (prob_C : ℚ) :
  total_students = 31 →
  prob_A = 0.7 * prob_B →
  prob_C = 1.4 * prob_B →
  prob_A + prob_B + prob_C = 1 →
  (total_students : ℚ) * prob_B = 10 := by
  sorry

end NUMINAMATH_CALUDE_history_class_grades_l3703_370309


namespace NUMINAMATH_CALUDE_sum_of_coordinates_of_B_l3703_370379

/-- Given two points A and B in a 2D plane, where A is at the origin and B is on the line y = 5,
    with the slope of the line AB being 3/4, prove that the sum of the x- and y-coordinates of B is 35/3. -/
theorem sum_of_coordinates_of_B (A B : ℝ × ℝ) : 
  A = (0, 0) →
  B.2 = 5 →
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 →
  B.1 + B.2 = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_of_B_l3703_370379


namespace NUMINAMATH_CALUDE_pat_stickers_end_of_week_l3703_370330

/-- The number of stickers Pat had at the end of the week -/
def total_stickers (initial : ℕ) (earned : ℕ) : ℕ := initial + earned

/-- Theorem: Pat had 61 stickers at the end of the week -/
theorem pat_stickers_end_of_week :
  total_stickers 39 22 = 61 := by
  sorry

end NUMINAMATH_CALUDE_pat_stickers_end_of_week_l3703_370330


namespace NUMINAMATH_CALUDE_only_zero_and_198_satisfy_l3703_370336

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for numbers equal to 11 times the sum of their digits -/
def is_eleven_times_sum_of_digits (n : ℕ) : Prop :=
  n = 11 * sum_of_digits n

theorem only_zero_and_198_satisfy :
  ∀ n : ℕ, is_eleven_times_sum_of_digits n ↔ n = 0 ∨ n = 198 := by sorry

end NUMINAMATH_CALUDE_only_zero_and_198_satisfy_l3703_370336


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l3703_370350

-- Define the properties of quadrilaterals
def is_square (q : Type) : Prop := sorry
def is_rectangle (q : Type) : Prop := sorry

-- Given statement
axiom square_implies_rectangle : ∀ (q : Type), is_square q → is_rectangle q

-- Theorem to prove
theorem converse_and_inverse_false :
  (∃ (q : Type), is_rectangle q ∧ ¬is_square q) ∧
  (∃ (q : Type), ¬is_square q ∧ is_rectangle q) :=
sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l3703_370350


namespace NUMINAMATH_CALUDE_subset_condition_nonempty_intersection_l3703_370368

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- Theorem 1: A ⊆ B iff a < -1
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ a < -1 := by sorry

-- Theorem 2: A ∩ B ≠ ∅ iff a < 3
theorem nonempty_intersection (a : ℝ) : (A ∩ B a).Nonempty ↔ a < 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_nonempty_intersection_l3703_370368


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l3703_370302

theorem two_digit_reverse_sum (x y m : ℕ) : 
  (10 ≤ x ∧ x < 100) →  -- x is a two-digit integer
  (10 ≤ y ∧ y < 100) →  -- y is a two-digit integer
  (∃ (a b : ℕ), x = 10 * a + b ∧ y = 10 * b + a) →  -- y is obtained by reversing the digits of x
  (0 < m) →  -- m is a positive integer
  (x^2 - y^2 = 9 * m^2) →  -- given equation
  x + y + 2 * m = 143 := by
sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l3703_370302


namespace NUMINAMATH_CALUDE_min_sides_for_80_intersections_l3703_370388

/-- The number of intersection points between two n-sided polygons -/
def intersection_points (n : ℕ) : ℕ := 80

/-- Proposition: The minimum value of n for which two n-sided polygons can have exactly 80 intersection points is 10 -/
theorem min_sides_for_80_intersections :
  ∀ n : ℕ, intersection_points n = 80 → n ≥ 10 ∧ 
  ∃ (m : ℕ), m = 10 ∧ intersection_points m = 80 :=
sorry

end NUMINAMATH_CALUDE_min_sides_for_80_intersections_l3703_370388


namespace NUMINAMATH_CALUDE_total_time_calculation_l3703_370357

-- Define the constants
def performance_time : ℕ := 6
def practice_ratio : ℕ := 3
def tantrum_ratio : ℕ := 5

-- Define the theorem
theorem total_time_calculation :
  performance_time * (1 + practice_ratio + tantrum_ratio) = 54 :=
by sorry

end NUMINAMATH_CALUDE_total_time_calculation_l3703_370357


namespace NUMINAMATH_CALUDE_divisible_by_two_l3703_370338

theorem divisible_by_two (a b : ℕ) : 
  (2 ∣ (a * b)) → (2 ∣ a) ∨ (2 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_two_l3703_370338


namespace NUMINAMATH_CALUDE_riding_to_total_ratio_l3703_370394

/-- Represents the number of horses and owners -/
def total_count : ℕ := 18

/-- Represents the number of legs walking on the ground -/
def legs_on_ground : ℕ := 90

/-- Represents the number of owners riding their horses -/
def riding_owners : ℕ := total_count - (legs_on_ground - 4 * total_count) / 2

/-- Theorem stating the ratio of riding owners to total owners -/
theorem riding_to_total_ratio :
  (riding_owners : ℚ) / total_count = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_riding_to_total_ratio_l3703_370394


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_average_l3703_370329

theorem consecutive_even_numbers_sum_average (x y z : ℤ) : 
  (∃ k : ℤ, x = 2*k ∧ y = 2*k + 2 ∧ z = 2*k + 4) →  -- consecutive even numbers
  (x + y + z = (x + y + z) / 3 + 44) →               -- sum is 44 more than average
  z = 24 :=                                          -- largest number is 24
by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_average_l3703_370329


namespace NUMINAMATH_CALUDE_apple_vendor_discard_percent_l3703_370326

/-- Represents the vendor's apple selling and discarding pattern -/
structure AppleVendor where
  initial_apples : ℝ
  day1_sell_percent : ℝ
  day1_discard_percent : ℝ
  day2_sell_percent : ℝ
  total_discard_percent : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem apple_vendor_discard_percent 
  (v : AppleVendor) 
  (h1 : v.day1_sell_percent = 50)
  (h2 : v.day2_sell_percent = 50)
  (h3 : v.total_discard_percent = 30)
  : v.day1_discard_percent = 20 := by
  sorry


end NUMINAMATH_CALUDE_apple_vendor_discard_percent_l3703_370326


namespace NUMINAMATH_CALUDE_approximate_number_properties_l3703_370387

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : Float
  exponent : Int

/-- Determines if a number is accurate to a specific place value -/
def is_accurate_to (n : ScientificNotation) (place : Int) : Prop :=
  sorry

/-- Counts the number of significant figures in a number -/
def count_significant_figures (n : ScientificNotation) : Nat :=
  sorry

/-- The hundreds place value -/
def hundreds : Int :=
  2

theorem approximate_number_properties (n : ScientificNotation) 
  (h1 : n.coefficient = 8.8)
  (h2 : n.exponent = 3) :
  is_accurate_to n hundreds ∧ count_significant_figures n = 2 := by
  sorry

end NUMINAMATH_CALUDE_approximate_number_properties_l3703_370387


namespace NUMINAMATH_CALUDE_g_sum_equals_one_l3703_370318

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom func_property : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y
axiom f_nonzero : ∀ x : ℝ, f x ≠ 0
axiom f_equal : f 1 = f 2

-- State the theorem
theorem g_sum_equals_one : g (-1) + g 1 = 1 := by sorry

end NUMINAMATH_CALUDE_g_sum_equals_one_l3703_370318


namespace NUMINAMATH_CALUDE_polynomial_value_l3703_370341

theorem polynomial_value : ∀ a b : ℝ, 
  (a * 1^3 + b * 1 + 1 = 2023) → 
  (a * (-1)^3 + b * (-1) - 2 = -2024) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l3703_370341


namespace NUMINAMATH_CALUDE_cost_per_square_meter_l3703_370334

def initial_land : ℝ := 300
def final_land : ℝ := 900
def total_cost : ℝ := 12000

theorem cost_per_square_meter :
  (total_cost / (final_land - initial_land)) = 20 := by sorry

end NUMINAMATH_CALUDE_cost_per_square_meter_l3703_370334


namespace NUMINAMATH_CALUDE_gcd_459_357_l3703_370306

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l3703_370306


namespace NUMINAMATH_CALUDE_solve_for_y_l3703_370317

theorem solve_for_y (x y : ℤ) (h1 : x^2 - x + 6 = y + 2) (h2 : x = -8) : y = 76 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3703_370317


namespace NUMINAMATH_CALUDE_one_bedroom_apartment_fraction_l3703_370396

theorem one_bedroom_apartment_fraction :
  let two_bedroom_fraction : ℝ := 0.33
  let total_fraction : ℝ := 0.5
  let one_bedroom_fraction : ℝ := total_fraction - two_bedroom_fraction
  one_bedroom_fraction = 0.17 := by
sorry

end NUMINAMATH_CALUDE_one_bedroom_apartment_fraction_l3703_370396


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l3703_370393

/-- A triangle with consecutive even number side lengths. -/
structure EvenTriangle where
  n : ℕ
  side1 : ℕ := 2 * n
  side2 : ℕ := 2 * n + 2
  side3 : ℕ := 2 * n + 4

/-- The triangle inequality holds for an EvenTriangle. -/
def satisfiesTriangleInequality (t : EvenTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- The perimeter of an EvenTriangle. -/
def perimeter (t : EvenTriangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem: The smallest possible perimeter of a triangle with consecutive even number
    side lengths that satisfies the triangle inequality is 18. -/
theorem smallest_even_triangle_perimeter :
  ∃ (t : EvenTriangle), satisfiesTriangleInequality t ∧
    perimeter t = 18 ∧
    ∀ (t' : EvenTriangle), satisfiesTriangleInequality t' → perimeter t' ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l3703_370393


namespace NUMINAMATH_CALUDE_intersection_A_B_m3_union_A_B_eq_A_l3703_370323

-- Define sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 < 0}

-- Theorem 1: Intersection of A and B when m = 3
theorem intersection_A_B_m3 : A ∩ B 3 = {x | 2 < x ∧ x < 3} := by sorry

-- Theorem 2: Condition for A ∪ B = A
theorem union_A_B_eq_A (m : ℝ) : A ∪ B m = A ↔ 0 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_m3_union_A_B_eq_A_l3703_370323


namespace NUMINAMATH_CALUDE_rental_duration_proof_l3703_370360

/-- Calculates the number of rental days given the daily rate, weekly rate, and total payment -/
def rentalDays (dailyRate weeklyRate totalPayment : ℕ) : ℕ :=
  let fullWeeks := totalPayment / weeklyRate
  let remainingPayment := totalPayment % weeklyRate
  let additionalDays := remainingPayment / dailyRate
  fullWeeks * 7 + additionalDays

/-- Proves that given the specified rates and payment, the rental duration is 11 days -/
theorem rental_duration_proof :
  rentalDays 30 190 310 = 11 := by
  sorry

end NUMINAMATH_CALUDE_rental_duration_proof_l3703_370360


namespace NUMINAMATH_CALUDE_geometric_mean_of_sqrt2_plus_minus_one_l3703_370391

theorem geometric_mean_of_sqrt2_plus_minus_one :
  let a := Real.sqrt 2 + 1
  let b := Real.sqrt 2 - 1
  ∃ x : ℝ, x^2 = a * b ∧ (x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_of_sqrt2_plus_minus_one_l3703_370391
