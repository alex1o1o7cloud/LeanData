import Mathlib

namespace NUMINAMATH_CALUDE_no_extrema_iff_a_nonpositive_l2610_261024

/-- The function f(x) = x^2 - a * ln(x) has no extrema if and only if a ≤ 0 -/
theorem no_extrema_iff_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y > 0 ∧ (x^2 - a * Real.log x < y^2 - a * Real.log y ∨ 
                                      x^2 - a * Real.log x > y^2 - a * Real.log y)) ↔ 
  a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_no_extrema_iff_a_nonpositive_l2610_261024


namespace NUMINAMATH_CALUDE_congruence_implication_l2610_261052

theorem congruence_implication (a b c d n : ℤ) 
  (h1 : a * c ≡ 0 [ZMOD n])
  (h2 : b * c + a * d ≡ 0 [ZMOD n]) :
  b * c ≡ 0 [ZMOD n] ∧ a * d ≡ 0 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_congruence_implication_l2610_261052


namespace NUMINAMATH_CALUDE_max_subway_commuters_l2610_261081

theorem max_subway_commuters (total_employees : ℕ) 
  (h_total : total_employees = 48) 
  (part_time full_time : ℕ) 
  (h_sum : part_time + full_time = total_employees) 
  (h_both_exist : part_time > 0 ∧ full_time > 0) :
  ∃ (subway_commuters : ℕ), 
    subway_commuters = ⌊(1 / 3 : ℚ) * part_time⌋ + ⌊(1 / 4 : ℚ) * full_time⌋ ∧
    subway_commuters ≤ 15 ∧
    (∀ (pt ft : ℕ), 
      pt + ft = total_employees → 
      pt > 0 → 
      ft > 0 → 
      ⌊(1 / 3 : ℚ) * pt⌋ + ⌊(1 / 4 : ℚ) * ft⌋ ≤ subway_commuters) :=
by sorry

end NUMINAMATH_CALUDE_max_subway_commuters_l2610_261081


namespace NUMINAMATH_CALUDE_triangle_side_comparison_l2610_261015

theorem triangle_side_comparison (A B C : ℝ) (a b c : ℝ) :
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a / Real.sin A = b / Real.sin B) →
  (Real.sin A > Real.sin B) →
  (a > b) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_comparison_l2610_261015


namespace NUMINAMATH_CALUDE_bicycle_cost_l2610_261006

def hourly_rate : ℕ := 5
def monday_hours : ℕ := 2
def wednesday_hours : ℕ := 1
def friday_hours : ℕ := 3
def weeks_to_work : ℕ := 6

def weekly_hours : ℕ := monday_hours + wednesday_hours + friday_hours

def weekly_earnings : ℕ := weekly_hours * hourly_rate

theorem bicycle_cost : weekly_earnings * weeks_to_work = 180 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_l2610_261006


namespace NUMINAMATH_CALUDE_parabola_equation_l2610_261045

/-- A parabola in the Cartesian coordinate system with directrix y = 4 -/
structure Parabola where
  /-- The equation of the parabola -/
  equation : ℝ → ℝ → Prop

/-- The standard form of a parabola equation -/
def StandardForm (p : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 = -4*p*y

/-- Theorem: The standard equation of a parabola with directrix y = 4 is x^2 = -16y -/
theorem parabola_equation (P : Parabola) : 
  P.equation = StandardForm 4 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2610_261045


namespace NUMINAMATH_CALUDE_like_terms_exponent_product_l2610_261098

theorem like_terms_exponent_product (x y : ℝ) (m n : ℕ) : 
  (∀ (a b : ℝ), a * x^3 * y^n = b * x^m * y^2 → a ≠ 0 → b ≠ 0 → m = 3 ∧ n = 2) →
  m * n = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_product_l2610_261098


namespace NUMINAMATH_CALUDE_existence_of_non_triangle_forming_numbers_l2610_261088

theorem existence_of_non_triangle_forming_numbers : 
  ∃ (a b : ℕ), a > 1000 ∧ b > 1000 ∧ 
  (∀ (c : ℕ), ∃ (k : ℕ), c = k^2 → 
    ¬(a + b > c ∧ a + c > b ∧ b + c > a)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_non_triangle_forming_numbers_l2610_261088


namespace NUMINAMATH_CALUDE_equations_not_equivalent_l2610_261017

theorem equations_not_equivalent : 
  ¬(∀ x : ℝ, (2 * (x - 10)) / (x^2 - 13*x + 30) = 1 ↔ x^2 - 15*x + 50 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equations_not_equivalent_l2610_261017


namespace NUMINAMATH_CALUDE_divisible_by_five_l2610_261042

theorem divisible_by_five (a b : ℕ) : 
  (5 ∣ a * b) → (5 ∣ a) ∨ (5 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l2610_261042


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l2610_261037

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 3 * x * y) : 1 / x + 1 / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l2610_261037


namespace NUMINAMATH_CALUDE_crayon_box_count_l2610_261010

theorem crayon_box_count : ∀ (total : ℕ),
  (total : ℚ) / 3 + (total : ℚ) * (1 / 5) + 56 = total →
  total = 120 := by
sorry

end NUMINAMATH_CALUDE_crayon_box_count_l2610_261010


namespace NUMINAMATH_CALUDE_min_face_sum_l2610_261029

/-- Represents the arrangement of numbers on a cube's vertices -/
def CubeArrangement := Fin 8 → Fin 8

/-- The sum of any three numbers on the same face is at least 10 -/
def ValidArrangement (arr : CubeArrangement) : Prop :=
  ∀ (face : Fin 6) (v1 v2 v3 : Fin 4), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 →
    (arr (face * 4 + v1) + arr (face * 4 + v2) + arr (face * 4 + v3) : ℕ) ≥ 10

/-- The sum of numbers on one face -/
def FaceSum (arr : CubeArrangement) (face : Fin 6) : ℕ :=
  (arr (face * 4) : ℕ) + (arr (face * 4 + 1) : ℕ) + (arr (face * 4 + 2) : ℕ) + (arr (face * 4 + 3) : ℕ)

/-- The minimum possible sum of numbers on one face is 16 -/
theorem min_face_sum :
  ∀ (arr : CubeArrangement), ValidArrangement arr →
    ∃ (face : Fin 6), FaceSum arr face = 16 ∧
      ∀ (other_face : Fin 6), FaceSum arr other_face ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_min_face_sum_l2610_261029


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l2610_261038

theorem ceiling_floor_difference : 
  ⌈(15 / 8 : ℝ) * (-34 / 4 : ℝ)⌉ - ⌊(15 / 8 : ℝ) * ⌊-34 / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l2610_261038


namespace NUMINAMATH_CALUDE_equation_solution_l2610_261058

theorem equation_solution : 
  ∀ x y z : ℕ, 2^x + 3^y + 7 = z! ↔ (x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5) :=
by sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2610_261058


namespace NUMINAMATH_CALUDE_reflection_result_l2610_261096

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -(p.2))

/-- The original point F -/
def F : ℝ × ℝ := (3, -3)

theorem reflection_result :
  (reflect_x (reflect_y F)) = (-3, 3) := by sorry

end NUMINAMATH_CALUDE_reflection_result_l2610_261096


namespace NUMINAMATH_CALUDE_fertilizer_pesticide_cost_l2610_261083

/-- Proves the amount spent on fertilizers and pesticides for a small farm operation --/
theorem fertilizer_pesticide_cost
  (seed_cost : ℝ)
  (labor_cost : ℝ)
  (num_bags : ℕ)
  (price_per_bag : ℝ)
  (profit_percentage : ℝ)
  (h1 : seed_cost = 50)
  (h2 : labor_cost = 15)
  (h3 : num_bags = 10)
  (h4 : price_per_bag = 11)
  (h5 : profit_percentage = 0.1)
  : ∃ (fertilizer_pesticide_cost : ℝ),
    fertilizer_pesticide_cost = 35 ∧
    price_per_bag * num_bags = (1 + profit_percentage) * (seed_cost + labor_cost + fertilizer_pesticide_cost) :=
by sorry

end NUMINAMATH_CALUDE_fertilizer_pesticide_cost_l2610_261083


namespace NUMINAMATH_CALUDE_spheres_in_unit_cube_radius_l2610_261033

/-- A configuration of spheres in a unit cube -/
structure SpheresInCube where
  /-- The number of spheres in the cube -/
  num_spheres : ℕ
  /-- The radius of each sphere -/
  radius : ℝ
  /-- One sphere is at a vertex of the cube -/
  vertex_sphere : Prop
  /-- Each of the remaining spheres is tangent to the vertex sphere and three faces of the cube -/
  remaining_spheres_tangent : Prop

/-- The theorem stating the radius of spheres in the given configuration -/
theorem spheres_in_unit_cube_radius (config : SpheresInCube) :
  config.num_spheres = 12 →
  config.vertex_sphere →
  config.remaining_spheres_tangent →
  config.radius = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_spheres_in_unit_cube_radius_l2610_261033


namespace NUMINAMATH_CALUDE_school_pencil_order_l2610_261047

/-- The number of pencils each student receives -/
def pencils_per_student : ℕ := 3

/-- The number of students in the school -/
def number_of_students : ℕ := 65

/-- The total number of pencils ordered by the school -/
def total_pencils : ℕ := pencils_per_student * number_of_students

theorem school_pencil_order : total_pencils = 195 := by
  sorry

end NUMINAMATH_CALUDE_school_pencil_order_l2610_261047


namespace NUMINAMATH_CALUDE_eden_has_fourteen_bears_l2610_261092

/-- The number of stuffed bears Eden has after receiving her share from Daragh --/
def edens_final_bear_count (initial_bears : ℕ) (favorite_bears : ℕ) (sisters : ℕ) (edens_initial_bears : ℕ) : ℕ :=
  let remaining_bears := initial_bears - favorite_bears
  let bears_per_sister := remaining_bears / sisters
  edens_initial_bears + bears_per_sister

/-- Theorem stating that Eden will have 14 stuffed bears after receiving her share --/
theorem eden_has_fourteen_bears :
  edens_final_bear_count 20 8 3 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_eden_has_fourteen_bears_l2610_261092


namespace NUMINAMATH_CALUDE_min_resistance_optimal_l2610_261084

noncomputable def min_resistance (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : ℝ :=
  let r₁₂ := (a₁ * a₂) / (a₁ + a₂)
  let r₁₂₃ := r₁₂ + a₃
  let r₄₅ := (a₄ * a₅) / (a₄ + a₅)
  let r₄₅₆ := r₄₅ + a₆
  (r₁₂₃ * r₄₅₆) / (r₁₂₃ + r₄₅₆)

theorem min_resistance_optimal
  (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)
  (h : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄ ∧ a₄ > a₅ ∧ a₅ > a₆) :
  ∀ (r : ℝ), r ≥ min_resistance a₁ a₂ a₃ a₄ a₅ a₆ :=
by sorry

end NUMINAMATH_CALUDE_min_resistance_optimal_l2610_261084


namespace NUMINAMATH_CALUDE_spade_calculation_l2610_261007

def spade (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem spade_calculation : spade 2 (spade 3 (spade 1 2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l2610_261007


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_6_real_l2610_261050

theorem sqrt_2x_minus_6_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 2 * x - 6) ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_6_real_l2610_261050


namespace NUMINAMATH_CALUDE_train_passing_time_l2610_261019

/-- Proves that a train with given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 285 →
  train_speed_kmh = 54 →
  passing_time = 19 →
  train_length / (train_speed_kmh * 1000 / 3600) = passing_time :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l2610_261019


namespace NUMINAMATH_CALUDE_square_difference_equality_l2610_261078

theorem square_difference_equality : (19 + 12)^2 - (12^2 + 19^2) = 456 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2610_261078


namespace NUMINAMATH_CALUDE_interval_length_implies_c_minus_three_l2610_261025

theorem interval_length_implies_c_minus_three (c : ℝ) : 
  (∃ x : ℝ, 3 ≤ 5*x - 4 ∧ 5*x - 4 ≤ c) →
  (∀ x : ℝ, 3 ≤ 5*x - 4 ∧ 5*x - 4 ≤ c → (7/5 : ℝ) ≤ x ∧ x ≤ (c + 4)/5) →
  ((c + 4)/5 - 7/5 = 15) →
  c - 3 = 75 := by
sorry

end NUMINAMATH_CALUDE_interval_length_implies_c_minus_three_l2610_261025


namespace NUMINAMATH_CALUDE_people_in_room_l2610_261073

theorem people_in_room (total_chairs : ℕ) (people : ℕ) : 
  (3 * people : ℚ) / 5 = (5 * total_chairs : ℚ) / 6 →  -- Three-fifths of people are seated in five-sixths of chairs
  total_chairs - (5 * total_chairs) / 6 = 10 →         -- 10 chairs are empty
  people = 83 := by
sorry

end NUMINAMATH_CALUDE_people_in_room_l2610_261073


namespace NUMINAMATH_CALUDE_lana_extra_flowers_l2610_261089

/-- The number of extra flowers Lana picked -/
def extra_flowers (tulips roses used : ℕ) : ℕ :=
  tulips + roses - used

/-- Theorem: Lana picked 3 extra flowers -/
theorem lana_extra_flowers :
  extra_flowers 36 37 70 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lana_extra_flowers_l2610_261089


namespace NUMINAMATH_CALUDE_quadratic_sum_l2610_261059

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x = a * x^2 + b * x + c) →
  (QuadraticFunction a b c 1 = 64) →
  (QuadraticFunction a b c (-2) = 0) →
  (QuadraticFunction a b c 4 = 0) →
  a + b + c = 64 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2610_261059


namespace NUMINAMATH_CALUDE_spears_from_sapling_proof_l2610_261070

/-- The number of spears that can be made from a log -/
def spears_per_log : ℕ := 9

/-- The number of spears that can be made from 6 saplings and a log -/
def spears_from_6_saplings_and_log : ℕ := 27

/-- The number of saplings used along with a log -/
def number_of_saplings : ℕ := 6

/-- The number of spears that can be made from a single sapling -/
def spears_per_sapling : ℕ := 3

theorem spears_from_sapling_proof :
  number_of_saplings * spears_per_sapling + spears_per_log = spears_from_6_saplings_and_log :=
by sorry

end NUMINAMATH_CALUDE_spears_from_sapling_proof_l2610_261070


namespace NUMINAMATH_CALUDE_cookie_box_cost_josh_cookie_box_cost_l2610_261065

/-- The cost of a box of cookies given Josh's bracelet-making business --/
theorem cookie_box_cost (cost_per_bracelet : ℚ) (price_per_bracelet : ℚ) 
  (num_bracelets : ℕ) (money_left : ℚ) : ℚ :=
  let profit_per_bracelet := price_per_bracelet - cost_per_bracelet
  let total_profit := profit_per_bracelet * num_bracelets
  total_profit - money_left

/-- The cost of Josh's box of cookies is $3 --/
theorem josh_cookie_box_cost : cookie_box_cost 1 1.5 12 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookie_box_cost_josh_cookie_box_cost_l2610_261065


namespace NUMINAMATH_CALUDE_log_equation_solution_l2610_261093

/-- Given that log₃ₓ(343) = x and x is real, prove that x is a non-square, non-cube, non-integral rational number -/
theorem log_equation_solution (x : ℝ) (h : Real.log 343 / Real.log (3 * x) = x) :
  ∃ (a b : ℤ), x = (a : ℝ) / (b : ℝ) ∧ 
  b ≠ 0 ∧ 
  ¬ ∃ (n : ℤ), x = n ∧
  ¬ ∃ (n : ℝ), x = n ^ 2 ∧
  ¬ ∃ (n : ℝ), x = n ^ 3 :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2610_261093


namespace NUMINAMATH_CALUDE_circle_x_axis_intersection_l2610_261076

/-- Given a circle with diameter endpoints (0,0) and (10,10), 
    the x-coordinate of the second intersection point with the x-axis is 10 -/
theorem circle_x_axis_intersection :
  ∀ (C : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ C ↔ (x - 5)^2 + (y - 5)^2 = 50) →
    (0, 0) ∈ C →
    (10, 10) ∈ C →
    ∃ (x : ℝ), x ≠ 0 ∧ (x, 0) ∈ C ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_x_axis_intersection_l2610_261076


namespace NUMINAMATH_CALUDE_power_of_three_mod_eleven_l2610_261056

theorem power_of_three_mod_eleven : 3^2023 % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eleven_l2610_261056


namespace NUMINAMATH_CALUDE_senior_count_l2610_261021

/-- Represents the count of students in each grade level -/
structure StudentCounts where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- Given the conditions of the student sample, proves the number of seniors -/
theorem senior_count (total : ℕ) (counts : StudentCounts) : 
  total = 800 ∧ 
  counts.juniors = (23 * total) / 100 ∧ 
  counts.sophomores = (25 * total) / 100 ∧ 
  counts.freshmen = counts.sophomores + 56 ∧ 
  total = counts.freshmen + counts.sophomores + counts.juniors + counts.seniors → 
  counts.seniors = 160 := by
sorry


end NUMINAMATH_CALUDE_senior_count_l2610_261021


namespace NUMINAMATH_CALUDE_perpendicular_line_angle_l2610_261046

-- Define the perpendicularity condition
def isPerpendicular (θ : Real) : Prop :=
  ∃ t : Real, (1 + t * Real.cos θ = t * Real.sin θ) ∧ 
              (Real.tan θ = -1)

-- State the theorem
theorem perpendicular_line_angle :
  ∀ θ : Real, 0 ≤ θ ∧ θ < π → isPerpendicular θ → θ = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_angle_l2610_261046


namespace NUMINAMATH_CALUDE_framing_needed_l2610_261016

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered photograph. -/
theorem framing_needed (orig_width orig_height border_width : ℕ) : 
  orig_width = 5 →
  orig_height = 7 →
  border_width = 3 →
  let enlarged_width := 2 * orig_width
  let enlarged_height := 2 * orig_height
  let framed_width := enlarged_width + 2 * border_width
  let framed_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (framed_width + framed_height)
  let feet := (perimeter + 11) / 12  -- Ceiling division to get the next whole foot
  feet = 6 := by
  sorry

#check framing_needed

end NUMINAMATH_CALUDE_framing_needed_l2610_261016


namespace NUMINAMATH_CALUDE_lisa_equal_earnings_l2610_261014

/-- Given Greta's work hours, Greta's hourly rate, and Lisa's hourly rate,
    calculates the number of hours Lisa needs to work to equal Greta's earnings. -/
def lisa_work_hours (greta_hours : ℕ) (greta_rate : ℚ) (lisa_rate : ℚ) : ℚ :=
  (greta_hours : ℚ) * greta_rate / lisa_rate

/-- Proves that Lisa needs to work 32 hours to equal Greta's earnings,
    given the specified conditions. -/
theorem lisa_equal_earnings : lisa_work_hours 40 12 15 = 32 := by
  sorry

end NUMINAMATH_CALUDE_lisa_equal_earnings_l2610_261014


namespace NUMINAMATH_CALUDE_circular_matrix_determinant_properties_l2610_261040

def circularMatrix (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a, b, c],
    ![c, a, b],
    ![b, c, a]]

theorem circular_matrix_determinant_properties :
  (∃ (S : Set (ℚ × ℚ × ℚ)), Set.Infinite S ∧
    ∀ (abc : ℚ × ℚ × ℚ), abc ∈ S →
      Matrix.det (circularMatrix abc.1 abc.2.1 abc.2.2) = 1) ∧
  (∃ (T : Set (ℤ × ℤ × ℤ)), Set.Finite T ∧
    ∀ (abc : ℤ × ℤ × ℤ), Matrix.det (circularMatrix ↑abc.1 ↑abc.2.1 ↑abc.2.2) = 1 →
      abc ∈ T) := by
  sorry

end NUMINAMATH_CALUDE_circular_matrix_determinant_properties_l2610_261040


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2610_261067

def A : Set ℝ := {x | x + 2 = 0}
def B : Set ℝ := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : A ∩ B = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2610_261067


namespace NUMINAMATH_CALUDE_largest_y_value_l2610_261063

theorem largest_y_value : 
  (∃ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y) → 
  (∀ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y → y ≤ 3/25) ∧
  (∃ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y ∧ y = 3/25) := by
  sorry

end NUMINAMATH_CALUDE_largest_y_value_l2610_261063


namespace NUMINAMATH_CALUDE_equation_solution_l2610_261011

theorem equation_solution :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ 4/5 →
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 22*x - 48)/(5*x - 4) = -7 →
  x = -4/3 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2610_261011


namespace NUMINAMATH_CALUDE_hexagon_extension_length_l2610_261079

-- Define the regular hexagon
def RegularHexagon (C D E F G H : ℝ × ℝ) : Prop :=
  -- Add conditions for a regular hexagon with side length 4
  sorry

-- Define the extension of CD to Y
def ExtendCD (C D Y : ℝ × ℝ) : Prop :=
  dist C Y = 2 * dist C D

-- Main theorem
theorem hexagon_extension_length 
  (C D E F G H Y : ℝ × ℝ) 
  (hex : RegularHexagon C D E F G H) 
  (ext : ExtendCD C D Y) : 
  dist H Y = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_extension_length_l2610_261079


namespace NUMINAMATH_CALUDE_angle_sum_at_point_l2610_261005

/-- 
Given three angles that meet at a point in a plane, 
if two of the angles are 145° and 95°, 
then the third angle is 120°.
-/
theorem angle_sum_at_point (a b c : ℝ) : 
  a + b + c = 360 → a = 145 → b = 95 → c = 120 := by sorry

end NUMINAMATH_CALUDE_angle_sum_at_point_l2610_261005


namespace NUMINAMATH_CALUDE_pizza_combinations_l2610_261085

theorem pizza_combinations (n : ℕ) (k : ℕ) : n = 8 ∧ k = 5 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l2610_261085


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2610_261082

theorem polynomial_divisibility (a b c d e : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 7 * k) →
  (∃ k₁ k₂ k₃ k₄ k₅ : ℤ, a = 7 * k₁ ∧ b = 7 * k₂ ∧ c = 7 * k₃ ∧ d = 7 * k₄ ∧ e = 7 * k₅) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2610_261082


namespace NUMINAMATH_CALUDE_bookstore_sales_after_returns_l2610_261053

/-- Calculates the total sales after returns for a bookstore --/
theorem bookstore_sales_after_returns 
  (total_customers : ℕ) 
  (return_rate : ℚ) 
  (price_per_book : ℕ) : 
  total_customers = 1000 → 
  return_rate = 37 / 100 → 
  price_per_book = 15 → 
  (total_customers : ℚ) * (1 - return_rate) * (price_per_book : ℚ) = 9450 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_sales_after_returns_l2610_261053


namespace NUMINAMATH_CALUDE_employee_devices_l2610_261001

theorem employee_devices (total : ℝ) (h_total : total > 0) : 
  let cell_phone := (2/3 : ℝ) * total
  let pager := (2/5 : ℝ) * total
  let neither := (1/3 : ℝ) * total
  let both := cell_phone + pager - (total - neither)
  both / total = 2/5 := by
sorry

end NUMINAMATH_CALUDE_employee_devices_l2610_261001


namespace NUMINAMATH_CALUDE_new_person_weight_l2610_261074

/-- Given a group of 10 persons, if replacing one person weighing 65 kg
    with a new person increases the average weight by 3.2 kg,
    then the weight of the new person is 97 kg. -/
theorem new_person_weight
  (n : ℕ) (old_weight average_increase : ℝ)
  (h1 : n = 10)
  (h2 : old_weight = 65)
  (h3 : average_increase = 3.2) :
  let new_weight := old_weight + n * average_increase
  new_weight = 97 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2610_261074


namespace NUMINAMATH_CALUDE_square_of_98_l2610_261044

theorem square_of_98 : (98 : ℕ) ^ 2 = 9604 := by sorry

end NUMINAMATH_CALUDE_square_of_98_l2610_261044


namespace NUMINAMATH_CALUDE_exam_mean_score_l2610_261094

/-- Given an exam score distribution where 58 is 2 standard deviations below the mean
    and 98 is 3 standard deviations above the mean, the mean score is 74. -/
theorem exam_mean_score (μ σ : ℝ) 
  (h1 : 58 = μ - 2 * σ) 
  (h2 : 98 = μ + 3 * σ) : 
  μ = 74 := by
  sorry

end NUMINAMATH_CALUDE_exam_mean_score_l2610_261094


namespace NUMINAMATH_CALUDE_calculator_battery_life_l2610_261066

/-- Calculates the remaining battery life of a calculator after partial use and an exam -/
theorem calculator_battery_life 
  (full_battery : ℝ) 
  (used_fraction : ℝ) 
  (exam_duration : ℝ) 
  (h1 : full_battery = 60) 
  (h2 : used_fraction = 3/4) 
  (h3 : exam_duration = 2) :
  full_battery * (1 - used_fraction) - exam_duration = 13 := by
  sorry

end NUMINAMATH_CALUDE_calculator_battery_life_l2610_261066


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l2610_261064

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes is 7 -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 7 := by sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l2610_261064


namespace NUMINAMATH_CALUDE_cos_330_degrees_l2610_261043

theorem cos_330_degrees : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l2610_261043


namespace NUMINAMATH_CALUDE_inverse_inequality_for_negative_reals_l2610_261090

theorem inverse_inequality_for_negative_reals (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  1 / a > 1 / b := by
sorry

end NUMINAMATH_CALUDE_inverse_inequality_for_negative_reals_l2610_261090


namespace NUMINAMATH_CALUDE_expression_simplification_l2610_261071

theorem expression_simplification (x : ℝ) : 
  x * (x * (x * (3 - x) - 6) + 7) + 2 = -x^4 + 3*x^3 - 6*x^2 + 7*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2610_261071


namespace NUMINAMATH_CALUDE_triangle_toothpicks_count_l2610_261049

/-- The number of small triangles in the base of the large triangle -/
def base_triangles : ℕ := 101

/-- The total number of small triangles in the large triangle -/
def total_triangles (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of shared toothpicks in the structure -/
def shared_toothpicks (n : ℕ) : ℕ := 3 * total_triangles n / 2

/-- The number of boundary toothpicks -/
def boundary_toothpicks (n : ℕ) : ℕ := 3 * n

/-- The number of support toothpicks on the boundary -/
def support_toothpicks : ℕ := 3

/-- The total number of toothpicks required for the structure -/
def total_toothpicks (n : ℕ) : ℕ :=
  shared_toothpicks n + boundary_toothpicks n + support_toothpicks

theorem triangle_toothpicks_count :
  total_toothpicks base_triangles = 8032 :=
sorry

end NUMINAMATH_CALUDE_triangle_toothpicks_count_l2610_261049


namespace NUMINAMATH_CALUDE_linear_function_shift_l2610_261060

/-- 
A linear function y = -2x + b is shifted 3 units upwards.
This theorem proves that if the shifted function passes through the point (2, 0),
then b = 1.
-/
theorem linear_function_shift (b : ℝ) : 
  (∀ x y : ℝ, y = -2 * x + b + 3 → (x = 2 ∧ y = 0) → b = 1) := by
sorry

end NUMINAMATH_CALUDE_linear_function_shift_l2610_261060


namespace NUMINAMATH_CALUDE_solve_system_l2610_261036

theorem solve_system (x y : ℝ) (eq1 : x - y = 8) (eq2 : x + 2*y = 14) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2610_261036


namespace NUMINAMATH_CALUDE_eight_power_91_greater_than_seven_power_92_l2610_261008

theorem eight_power_91_greater_than_seven_power_92 : 8^91 > 7^92 := by
  sorry

end NUMINAMATH_CALUDE_eight_power_91_greater_than_seven_power_92_l2610_261008


namespace NUMINAMATH_CALUDE_marks_deck_cost_l2610_261000

/-- Calculates the total cost of constructing and sealing a rectangular deck. -/
def deck_total_cost (length width construction_cost_per_sqft sealant_cost_per_sqft : ℝ) : ℝ :=
  let area := length * width
  let construction_cost := area * construction_cost_per_sqft
  let sealant_cost := area * sealant_cost_per_sqft
  construction_cost + sealant_cost

/-- Theorem stating that the total cost of Mark's deck is $4800. -/
theorem marks_deck_cost : 
  deck_total_cost 30 40 3 1 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_marks_deck_cost_l2610_261000


namespace NUMINAMATH_CALUDE_marbles_in_jar_l2610_261039

/-- The number of marbles in a jar when two boys combine their collections -/
theorem marbles_in_jar (ben_marbles : ℕ) (leo_extra_marbles : ℕ) : 
  ben_marbles = 56 → leo_extra_marbles = 20 → 
  ben_marbles + (ben_marbles + leo_extra_marbles) = 132 := by
  sorry

#check marbles_in_jar

end NUMINAMATH_CALUDE_marbles_in_jar_l2610_261039


namespace NUMINAMATH_CALUDE_mancino_garden_width_is_5_l2610_261012

/-- The width of Mancino's gardens -/
def mancino_garden_width : ℝ := 5

/-- The number of Mancino's gardens -/
def mancino_garden_count : ℕ := 3

/-- The length of Mancino's gardens -/
def mancino_garden_length : ℝ := 16

/-- The number of Marquita's gardens -/
def marquita_garden_count : ℕ := 2

/-- The length of Marquita's gardens -/
def marquita_garden_length : ℝ := 8

/-- The width of Marquita's gardens -/
def marquita_garden_width : ℝ := 4

/-- The total area of all gardens -/
def total_garden_area : ℝ := 304

theorem mancino_garden_width_is_5 :
  mancino_garden_width = 5 ∧
  mancino_garden_count * mancino_garden_length * mancino_garden_width +
  marquita_garden_count * marquita_garden_length * marquita_garden_width =
  total_garden_area :=
by sorry

end NUMINAMATH_CALUDE_mancino_garden_width_is_5_l2610_261012


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l2610_261030

theorem divisibility_by_seven (n : ℕ) : 7 ∣ (3^(12*n + 1) + 2^(6*n + 2)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l2610_261030


namespace NUMINAMATH_CALUDE_twins_shirts_l2610_261034

/-- The number of shirts Hazel and Razel have in total -/
def total_shirts (hazel_shirts : ℕ) (razel_shirts : ℕ) : ℕ :=
  hazel_shirts + razel_shirts

/-- Theorem: If Hazel received 6 shirts and Razel received twice the number of shirts as Hazel,
    then the total number of shirts they have is 18. -/
theorem twins_shirts :
  let hazel_shirts : ℕ := 6
  let razel_shirts : ℕ := 2 * hazel_shirts
  total_shirts hazel_shirts razel_shirts = 18 := by
sorry

end NUMINAMATH_CALUDE_twins_shirts_l2610_261034


namespace NUMINAMATH_CALUDE_min_value_theorem_l2610_261009

theorem min_value_theorem (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m > 0) (h3 : n > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 2 → 1 / m + 2 / n ≤ 1 / x + 2 / y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 2 ∧ 1 / x + 2 / y = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2610_261009


namespace NUMINAMATH_CALUDE_line_segment_proportion_l2610_261022

theorem line_segment_proportion (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_proportion_l2610_261022


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2610_261048

-- Define a point in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define collinearity for three points
def collinear (p1 p2 p3 : Point3D) : Prop := sorry

-- State the theorem
theorem collinear_points_sum (a b : ℝ) :
  collinear (1, b, a) (b, 2, a) (b, a, 3) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l2610_261048


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2610_261095

/-- A polynomial with integer coefficients where each coefficient is between 0 and 4 inclusive -/
def IntPolynomial (m : ℕ) := { b : Fin (m + 1) → ℤ // ∀ i, 0 ≤ b i ∧ b i < 5 }

/-- Evaluation of an IntPolynomial at a given value -/
def evalPoly {m : ℕ} (P : IntPolynomial m) (x : ℝ) : ℝ :=
  (Finset.range (m + 1)).sum (fun i => (P.val i : ℝ) * x ^ i)

theorem polynomial_evaluation (m : ℕ) (P : IntPolynomial m) :
  evalPoly P (Real.sqrt 5) = 23 + 19 * Real.sqrt 5 →
  evalPoly P 3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2610_261095


namespace NUMINAMATH_CALUDE_min_additional_wins_correct_l2610_261075

/-- The minimum number of additional wins required to achieve a 90% winning percentage -/
def min_additional_wins : ℕ := 26

/-- The initial number of games played -/
def initial_games : ℕ := 4

/-- The initial number of games won -/
def initial_wins : ℕ := 1

/-- The target winning percentage -/
def target_percentage : ℚ := 9/10

theorem min_additional_wins_correct :
  ∀ n : ℕ, 
    (n ≥ min_additional_wins) ↔ 
    ((initial_wins + n : ℚ) / (initial_games + n)) ≥ target_percentage :=
sorry

end NUMINAMATH_CALUDE_min_additional_wins_correct_l2610_261075


namespace NUMINAMATH_CALUDE_square_of_negative_sqrt_two_l2610_261018

theorem square_of_negative_sqrt_two : (-Real.sqrt 2)^2 = 2 := by sorry

end NUMINAMATH_CALUDE_square_of_negative_sqrt_two_l2610_261018


namespace NUMINAMATH_CALUDE_f_two_zeros_l2610_261032

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x - a else 4*(x-a)*(x-2*a)

theorem f_two_zeros (a : ℝ) :
  (∃! (z1 z2 : ℝ), z1 ≠ z2 ∧ f a z1 = 0 ∧ f a z2 = 0 ∧ ∀ z, f a z = 0 → z = z1 ∨ z = z2) ↔
  (1/2 ≤ a ∧ a < 1) ∨ (2 ≤ a) :=
sorry

end NUMINAMATH_CALUDE_f_two_zeros_l2610_261032


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l2610_261027

/-- Given that x is inversely proportional to y, this function represents their relationship -/
def inverse_proportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_ratio
  (x₁ x₂ y₁ y₂ : ℝ)
  (hx₁ : x₁ ≠ 0)
  (hx₂ : x₂ ≠ 0)
  (hy₁ : y₁ ≠ 0)
  (hy₂ : y₂ ≠ 0)
  (hxy₁ : inverse_proportion x₁ y₁)
  (hxy₂ : inverse_proportion x₂ y₂)
  (hx_ratio : x₁ / x₂ = 3 / 4) :
  y₁ / y₂ = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l2610_261027


namespace NUMINAMATH_CALUDE_f_theorem_l2610_261068

def f_properties (f : ℝ → ℝ) : Prop :=
  Continuous f ∧
  (∀ x, f (-x) = f x) ∧
  (∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) ∧
  f (-1) = 0

theorem f_theorem (f : ℝ → ℝ) (h : f_properties f) :
  f 3 > f 4 ∧
  (∀ m, f (m - 1) < f 2 → m < -1 ∨ m > 3) ∧
  ∃ M, ∀ x, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_theorem_l2610_261068


namespace NUMINAMATH_CALUDE_total_age_proof_l2610_261057

/-- Given three people a, b, and c, where a is two years older than b, b is twice as old as c, 
    and b is 10 years old, prove that the total of their ages is 27 years. -/
theorem total_age_proof (a b c : ℕ) : 
  b = 10 → a = b + 2 → b = 2 * c → a + b + c = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_age_proof_l2610_261057


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_negation_of_specific_proposition_l2610_261080

theorem negation_of_existential_proposition (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_specific_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_negation_of_specific_proposition_l2610_261080


namespace NUMINAMATH_CALUDE_triangle_expression_l2610_261077

theorem triangle_expression (A B C : ℝ) : 
  A = 15 * π / 180 →
  A + B + C = π →
  Real.sqrt 3 * Real.sin A - Real.cos (B + C) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_expression_l2610_261077


namespace NUMINAMATH_CALUDE_money_ratio_l2610_261028

/-- Jake's feeding allowance in dollars -/
def feeding_allowance : ℚ := 4

/-- Cost of one candy in dollars -/
def candy_cost : ℚ := 1/5

/-- Number of candies Jake's friend can purchase -/
def candies_purchased : ℕ := 5

/-- Amount of money Jake gave to his friend in dollars -/
def money_given : ℚ := candy_cost * candies_purchased

/-- Theorem stating the ratio of money given to feeding allowance -/
theorem money_ratio : money_given / feeding_allowance = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_money_ratio_l2610_261028


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2610_261069

theorem inequality_solution_set (x : ℝ) :
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2610_261069


namespace NUMINAMATH_CALUDE_suitcase_lock_settings_l2610_261087

/-- The number of dials on the suitcase lock. -/
def num_dials : ℕ := 4

/-- The number of digits available for each dial. -/
def num_digits : ℕ := 10

/-- Calculates the number of different settings for a suitcase lock. -/
def lock_settings : ℕ := num_digits * (num_digits - 1) * (num_digits - 2) * (num_digits - 3)

/-- Theorem stating that the number of different settings for the suitcase lock is 5040. -/
theorem suitcase_lock_settings :
  lock_settings = 5040 := by sorry

end NUMINAMATH_CALUDE_suitcase_lock_settings_l2610_261087


namespace NUMINAMATH_CALUDE_claudia_coins_l2610_261026

/-- Represents the number of different coin combinations possible with n coins -/
def combinations (n : ℕ) : ℕ := sorry

/-- Represents the number of different values that can be formed with n coins -/
def values (n : ℕ) : ℕ := sorry

theorem claudia_coins :
  ∀ x y : ℕ,
  x + y = 15 →                           -- Total number of coins is 15
  combinations (x + y) = 23 →            -- 23 different combinations possible
  (∀ n : ℕ, n ≤ 10 → values n ≥ 15) →    -- At least 15 values with no more than 10 coins
  y = 9                                  -- Claudia has 9 10-cent coins
  := by sorry

end NUMINAMATH_CALUDE_claudia_coins_l2610_261026


namespace NUMINAMATH_CALUDE_five_from_eight_l2610_261002

/-- The number of ways to choose k items from a set of n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The problem statement -/
theorem five_from_eight : choose 8 5 = 56 := by sorry

end NUMINAMATH_CALUDE_five_from_eight_l2610_261002


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l2610_261062

theorem parabola_shift_theorem (m : ℝ) (x₁ x₂ : ℝ) : 
  x₁ * x₂ = x₁ + x₂ + 49 →
  x₁ * x₂ = -6 * m →
  x₁ + x₂ = 2 * m - 1 →
  min (abs x₁) (abs x₂) = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l2610_261062


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l2610_261099

/-- Represents the number 56.9 billion -/
def billion_value : ℝ := 56.9 * 1000000000

/-- Represents the scientific notation of 56.9 billion -/
def scientific_notation : ℝ := 5.69 * 10^9

/-- Theorem stating that 56.9 billion is equal to 5.69 × 10^9 in scientific notation -/
theorem billion_to_scientific_notation : billion_value = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l2610_261099


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2610_261004

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 + 3*Complex.I) / (3 - Complex.I) → z.im = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2610_261004


namespace NUMINAMATH_CALUDE_asymptote_sum_l2610_261061

/-- Given a function g(x) = (x+5) / (x^2 + cx + d) with vertical asymptotes at x = 2 and x = -3,
    prove that the sum of c and d is -5. -/
theorem asymptote_sum (c d : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -3 → 
    (x + 5) / (x^2 + c*x + d) = (x + 5) / ((x - 2) * (x + 3))) →
  c + d = -5 := by
sorry

end NUMINAMATH_CALUDE_asymptote_sum_l2610_261061


namespace NUMINAMATH_CALUDE_apple_juice_quantity_l2610_261055

/-- Given the total apple production and export percentage, calculate the quantity of apples used for juice -/
theorem apple_juice_quantity (total_production : ℝ) (export_percentage : ℝ) (juice_percentage : ℝ) : 
  total_production = 6 →
  export_percentage = 0.25 →
  juice_percentage = 0.60 →
  juice_percentage * (total_production * (1 - export_percentage)) = 2.7 := by
sorry

end NUMINAMATH_CALUDE_apple_juice_quantity_l2610_261055


namespace NUMINAMATH_CALUDE_star_op_equation_solution_l2610_261091

-- Define the "※" operation
def star_op (a b : ℝ) : ℝ := a * b^2 + 2 * a * b

-- State the theorem
theorem star_op_equation_solution :
  ∃! x : ℝ, star_op 1 x = -1 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_star_op_equation_solution_l2610_261091


namespace NUMINAMATH_CALUDE_num_lions_seen_l2610_261054

/-- The number of legs Borgnine wants to see at the zoo -/
def total_legs : ℕ := 1100

/-- The number of chimps Borgnine has seen -/
def num_chimps : ℕ := 12

/-- The number of lizards Borgnine has seen -/
def num_lizards : ℕ := 5

/-- The number of tarantulas Borgnine will see -/
def num_tarantulas : ℕ := 125

/-- The number of legs a chimp has -/
def chimp_legs : ℕ := 4

/-- The number of legs a lion has -/
def lion_legs : ℕ := 4

/-- The number of legs a lizard has -/
def lizard_legs : ℕ := 4

/-- The number of legs a tarantula has -/
def tarantula_legs : ℕ := 8

theorem num_lions_seen : ℕ := by
  sorry

end NUMINAMATH_CALUDE_num_lions_seen_l2610_261054


namespace NUMINAMATH_CALUDE_alphabet_letters_l2610_261041

theorem alphabet_letters (total : ℕ) (both : ℕ) (line_only : ℕ) (h1 : total = 60) (h2 : both = 20) (h3 : line_only = 36) :
  total = both + line_only + (total - (both + line_only)) →
  total - (both + line_only) = 24 :=
by sorry

end NUMINAMATH_CALUDE_alphabet_letters_l2610_261041


namespace NUMINAMATH_CALUDE_cd_total_length_l2610_261003

theorem cd_total_length : 
  let cd1 : ℝ := 1.5
  let cd2 : ℝ := 1.5
  let cd3 : ℝ := 2 * cd1
  let cd4 : ℝ := 0.5 * cd2
  let cd5 : ℝ := cd1 + cd2
  cd1 + cd2 + cd3 + cd4 + cd5 = 9.75 := by
sorry

end NUMINAMATH_CALUDE_cd_total_length_l2610_261003


namespace NUMINAMATH_CALUDE_max_a_value_l2610_261013

theorem max_a_value (a k x₁ x₂ : ℝ) : 
  (∀ k ∈ Set.Icc 0 2, 
   ∀ x₁ ∈ Set.Icc k (k + a), 
   ∀ x₂ ∈ Set.Icc (k + 2*a) (k + 4*a), 
   (x₁^2 - (k^2 - 5*a*k + 3)*x₁ + 7) ≥ (x₂^2 - (k^2 - 5*a*k + 3)*x₂ + 7)) →
  a ≤ (2 * Real.sqrt 6 - 4) / 5 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2610_261013


namespace NUMINAMATH_CALUDE_perpendicular_line_through_center_l2610_261086

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 - 3 = 0

/-- The given line equation -/
def given_line_equation (x y : ℝ) : Prop := x + y - 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 0)

/-- The perpendicular line equation -/
def perpendicular_line_equation (x y : ℝ) : Prop := x - y + 1 = 0

/-- Theorem stating that the line passing through the center of the circle and perpendicular to the given line has the equation x - y + 1 = 0 -/
theorem perpendicular_line_through_center :
  ∀ (x y : ℝ), 
    (x, y) = circle_center → 
    (∀ (x' y' : ℝ), given_line_equation x' y' → (x - x') * (y - y') = -1) → 
    perpendicular_line_equation x y :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_center_l2610_261086


namespace NUMINAMATH_CALUDE_difference_of_squares_l2610_261051

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2610_261051


namespace NUMINAMATH_CALUDE_coefficient_d_nonzero_l2610_261020

/-- A polynomial of degree 5 with five distinct roots including 0 and 1 -/
def Q (a b c d f : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + f

/-- The theorem stating that the coefficient d must be nonzero -/
theorem coefficient_d_nonzero (a b c d f : ℝ) :
  (∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ≠ 0 ∧ p ≠ 1 ∧ q ≠ 0 ∧ q ≠ 1 ∧ r ≠ 0 ∧ r ≠ 1 ∧
    ∀ x : ℝ, Q a b c d f x = 0 ↔ x = 0 ∨ x = 1 ∨ x = p ∨ x = q ∨ x = r) →
  d ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_coefficient_d_nonzero_l2610_261020


namespace NUMINAMATH_CALUDE_smallest_c_for_inverse_l2610_261097

def g (x : ℝ) : ℝ := (x + 3)^2 - 10

theorem smallest_c_for_inverse : 
  ∀ c : ℝ, (∀ x y : ℝ, x ≥ c → y ≥ c → g x = g y → x = y) ↔ c ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_inverse_l2610_261097


namespace NUMINAMATH_CALUDE_factorial_sum_square_solutions_l2610_261035

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_square_solutions :
  ∀ m n : ℕ+, m^2 = factorial_sum n ↔ (m = 1 ∧ n = 1) ∨ (m = 3 ∧ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_square_solutions_l2610_261035


namespace NUMINAMATH_CALUDE_fence_sections_count_l2610_261023

/-- The number of posts in the nth section -/
def posts_in_section (n : ℕ) : ℕ := 2 * n + 1

/-- The total number of posts used for n sections -/
def total_posts (n : ℕ) : ℕ := n^2

/-- The total number of posts available -/
def available_posts : ℕ := 435

theorem fence_sections_count :
  ∃ (n : ℕ), total_posts n = available_posts ∧ n = 21 :=
sorry

end NUMINAMATH_CALUDE_fence_sections_count_l2610_261023


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2610_261072

theorem diophantine_equation_solutions (p : ℕ) (h_prime : Nat.Prime p) :
  ∀ x y n : ℕ, x > 0 ∧ y > 0 ∧ n > 0 →
  p^n = x^3 + y^3 ↔
  (p = 2 ∧ ∃ k : ℕ, x = 2^k ∧ y = 2^k ∧ n = 3*k + 1) ∨
  (p = 3 ∧ ∃ k : ℕ, (x = 3^k ∧ y = 2 * 3^k ∧ n = 3*k + 2) ∨
                    (x = 2 * 3^k ∧ y = 3^k ∧ n = 3*k + 2)) :=
by sorry


end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2610_261072


namespace NUMINAMATH_CALUDE_negation_of_existence_l2610_261031

theorem negation_of_existence (p : Prop) :
  (¬ ∃ (x y : ℤ), x^2 + y^2 = 2015) ↔ (∀ (x y : ℤ), x^2 + y^2 ≠ 2015) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2610_261031
