import Mathlib

namespace NUMINAMATH_CALUDE_ceiling_sqrt_250_l809_80999

theorem ceiling_sqrt_250 : ⌈Real.sqrt 250⌉ = 16 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_250_l809_80999


namespace NUMINAMATH_CALUDE_cubes_with_three_painted_faces_l809_80984

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  painted_outside : Bool

/-- Represents a smaller cube cut from a larger cube -/
structure SmallCube where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Function to count the number of painted faces of a small cube -/
def count_painted_faces (c : Cube 4) (sc : SmallCube) : ℕ :=
  sorry

/-- Function to count the number of small cubes with at least three painted faces -/
def count_cubes_with_three_painted_faces (c : Cube 4) : ℕ :=
  sorry

/-- Theorem: In a 4x4x4 cube that is fully painted on the outside and then cut into 1x1x1 cubes,
    the number of 1x1x1 cubes with at least three faces painted is equal to 8 -/
theorem cubes_with_three_painted_faces (c : Cube 4) (h : c.painted_outside = true) :
  count_cubes_with_three_painted_faces c = 8 :=
by sorry

end NUMINAMATH_CALUDE_cubes_with_three_painted_faces_l809_80984


namespace NUMINAMATH_CALUDE_circle_cartesian_to_polar_l809_80960

/-- Proves that a circle with Cartesian equation x² + y² - 2y = 0 has polar equation ρ = 2sin(θ) -/
theorem circle_cartesian_to_polar :
  ∀ (x y ρ θ : ℝ),
  (x^2 + y^2 - 2*y = 0) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (ρ = 2 * Real.sin θ) :=
by sorry

end NUMINAMATH_CALUDE_circle_cartesian_to_polar_l809_80960


namespace NUMINAMATH_CALUDE_remaining_roots_equation_l809_80904

theorem remaining_roots_equation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hab : a ≠ b) :
  ∃ x₁ : ℝ, (x₁^2 + a*x₁ + b*c = 0 ∧ x₁^2 + b*x₁ + c*a = 0) →
  ∃ x₂ x₃ : ℝ, x₂ ≠ x₁ ∧ x₃ ≠ x₁ ∧ x₂^2 + c*x₂ + a*b = 0 ∧ x₃^2 + c*x₃ + a*b = 0 :=
sorry

end NUMINAMATH_CALUDE_remaining_roots_equation_l809_80904


namespace NUMINAMATH_CALUDE_corner_sum_l809_80920

/-- Represents a 10x10 array filled with integers from 1 to 100 -/
def CheckerBoard := Fin 10 → Fin 10 → Fin 100

/-- The checkerboard is filled in sequence -/
def is_sequential (board : CheckerBoard) : Prop :=
  ∀ i j, board i j = i.val * 10 + j.val + 1

/-- The sum of the numbers in the four corners of the checkerboard is 202 -/
theorem corner_sum (board : CheckerBoard) (h : is_sequential board) :
  (board 0 0).val + (board 0 9).val + (board 9 0).val + (board 9 9).val = 202 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_l809_80920


namespace NUMINAMATH_CALUDE_total_drivers_l809_80982

theorem total_drivers (N : ℕ) 
  (drivers_A : ℕ) 
  (sample_A sample_B sample_C sample_D : ℕ) :
  drivers_A = 96 →
  sample_A = 8 →
  sample_B = 23 →
  sample_C = 27 →
  sample_D = 43 →
  (sample_A : ℚ) / drivers_A = (sample_A + sample_B + sample_C + sample_D : ℚ) / N →
  N = 1212 :=
by sorry

end NUMINAMATH_CALUDE_total_drivers_l809_80982


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l809_80954

open Real

theorem triangle_ABC_properties (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 3 →
  C = π / 3 →
  2 * sin (2 * A) + sin (A - B) = sin C →
  (A = π / 2 ∨ A = π / 6) ∧
  2 * Real.sqrt 3 ≤ a + b + c ∧ a + b + c ≤ 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l809_80954


namespace NUMINAMATH_CALUDE_simplify_polynomial_l809_80924

theorem simplify_polynomial (b : ℝ) : (1 : ℝ) * (2 * b) * (3 * b^2) * (4 * b^3) * (5 * b^4) * (6 * b^5) = 720 * b^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l809_80924


namespace NUMINAMATH_CALUDE_stationery_sales_l809_80995

theorem stationery_sales (total_sales : ℕ) (fabric_fraction : ℚ) (jewelry_fraction : ℚ)
  (h_total : total_sales = 36)
  (h_fabric : fabric_fraction = 1/3)
  (h_jewelry : jewelry_fraction = 1/4)
  (h_stationery : fabric_fraction + jewelry_fraction < 1) :
  total_sales - (total_sales * fabric_fraction).floor - (total_sales * jewelry_fraction).floor = 15 :=
by sorry

end NUMINAMATH_CALUDE_stationery_sales_l809_80995


namespace NUMINAMATH_CALUDE_square_ends_in_001_l809_80981

theorem square_ends_in_001 (x : ℤ) : 
  x^2 ≡ 1 [ZMOD 1000] → 
  (x ≡ 1 [ZMOD 500] ∨ x ≡ -1 [ZMOD 500] ∨ x ≡ 249 [ZMOD 500] ∨ x ≡ -249 [ZMOD 500]) :=
by sorry

end NUMINAMATH_CALUDE_square_ends_in_001_l809_80981


namespace NUMINAMATH_CALUDE_subtraction_of_negative_integers_l809_80919

theorem subtraction_of_negative_integers : -3 - 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_negative_integers_l809_80919


namespace NUMINAMATH_CALUDE_x_axis_coefficients_l809_80937

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate to check if a line is the x-axis -/
def is_x_axis (l : Line) : Prop :=
  ∀ x y : ℝ, l.A * x + l.B * y + l.C = 0 ↔ y = 0

/-- Theorem stating that if a line is the x-axis, then B ≠ 0 and A = C = 0 -/
theorem x_axis_coefficients (l : Line) :
  is_x_axis l → l.B ≠ 0 ∧ l.A = 0 ∧ l.C = 0 :=
sorry

end NUMINAMATH_CALUDE_x_axis_coefficients_l809_80937


namespace NUMINAMATH_CALUDE_complex_root_coefficients_l809_80933

theorem complex_root_coefficients (b c : ℝ) : 
  (Complex.I * Real.sqrt 2 + 1) ^ 2 + b * (Complex.I * Real.sqrt 2 + 1) + c = 0 → 
  b = -2 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_coefficients_l809_80933


namespace NUMINAMATH_CALUDE_dual_polyhedron_properties_l809_80968

/-- A regular polyhedron with its dual -/
structure RegularPolyhedronWithDual where
  G : ℕ  -- number of faces
  P : ℕ  -- number of edges
  B : ℕ  -- number of vertices
  n : ℕ  -- number of edges meeting at each vertex

/-- Properties of the dual of a regular polyhedron -/
def dual_properties (poly : RegularPolyhedronWithDual) : Prop :=
  ∃ (dual_faces dual_edges dual_vertices : ℕ),
    dual_faces = poly.B ∧
    dual_edges = poly.P ∧
    dual_vertices = poly.G

/-- Theorem stating the properties of the dual polyhedron -/
theorem dual_polyhedron_properties (poly : RegularPolyhedronWithDual) :
  dual_properties poly :=
sorry

end NUMINAMATH_CALUDE_dual_polyhedron_properties_l809_80968


namespace NUMINAMATH_CALUDE_min_value_expression_l809_80928

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 - 1) * (1 / b^2 - 1) ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l809_80928


namespace NUMINAMATH_CALUDE_same_solution_value_of_b_l809_80987

theorem same_solution_value_of_b : ∀ x b : ℝ, 
  (3 * x + 9 = 6) ∧ 
  (5 * b * x - 15 = 5) → 
  b = -4 := by sorry

end NUMINAMATH_CALUDE_same_solution_value_of_b_l809_80987


namespace NUMINAMATH_CALUDE_quadratic_roots_equal_roots_condition_l809_80922

-- Part 1: Roots of x^2 - 2x - 8 = 0
theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 8
  ∃ (x₁ x₂ : ℝ), x₁ = 4 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
sorry

-- Part 2: Value of a when roots of x^2 - ax + 1 = 0 are equal
theorem equal_roots_condition :
  let g : ℝ → ℝ → ℝ := λ a x => x^2 - a*x + 1
  ∀ a : ℝ, (∃! x : ℝ, g a x = 0) → (a = 2 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_equal_roots_condition_l809_80922


namespace NUMINAMATH_CALUDE_min_value_quadratic_l809_80958

theorem min_value_quadratic (x : ℝ) :
  x ∈ Set.Icc (-2 : ℝ) (2 : ℝ) →
  (x^2 + 2*x + 1) ≥ 0 ∧ ∃ y ∈ Set.Icc (-2 : ℝ) (2 : ℝ), y^2 + 2*y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l809_80958


namespace NUMINAMATH_CALUDE_total_pages_read_l809_80934

-- Define the reading rates (pages per 60 minutes)
def rene_rate : ℕ := 30
def lulu_rate : ℕ := 27
def cherry_rate : ℕ := 25

-- Define the total reading time in minutes
def total_time : ℕ := 240

-- Define the function to calculate pages read given rate and time
def pages_read (rate : ℕ) (time : ℕ) : ℕ :=
  rate * (time / 60)

-- Theorem statement
theorem total_pages_read :
  pages_read rene_rate total_time +
  pages_read lulu_rate total_time +
  pages_read cherry_rate total_time = 328 := by
  sorry


end NUMINAMATH_CALUDE_total_pages_read_l809_80934


namespace NUMINAMATH_CALUDE_martha_age_is_32_l809_80969

-- Define Ellen's current age
def ellen_current_age : ℕ := 10

-- Define Ellen's age in 6 years
def ellen_future_age : ℕ := ellen_current_age + 6

-- Define Martha's age in terms of Ellen's future age
def martha_age : ℕ := 2 * ellen_future_age

-- Theorem to prove Martha's age
theorem martha_age_is_32 : martha_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_martha_age_is_32_l809_80969


namespace NUMINAMATH_CALUDE_mixture_combination_theorem_l809_80970

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℕ
  water : ℕ

/-- Combines two mixtures -/
def combineMixtures (m1 m2 : Mixture) : Mixture :=
  { milk := m1.milk + m2.milk
    water := m1.water + m2.water }

/-- Simplifies a ratio by dividing both parts by their GCD -/
def simplifyRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem mixture_combination_theorem :
  let m1 : Mixture := { milk := 7, water := 2 }
  let m2 : Mixture := { milk := 8, water := 1 }
  let combined := combineMixtures m1 m2
  simplifyRatio combined.milk combined.water = (5, 1) := by
  sorry

end NUMINAMATH_CALUDE_mixture_combination_theorem_l809_80970


namespace NUMINAMATH_CALUDE_roots_difference_squared_l809_80926

theorem roots_difference_squared (α β : ℝ) : 
  α^2 - 3*α + 1 = 0 → β^2 - 3*β + 1 = 0 → (α - β)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_difference_squared_l809_80926


namespace NUMINAMATH_CALUDE_joan_has_16_seashells_l809_80923

/-- The number of seashells Joan has now, given that she found 79 and gave away 63. -/
def joans_remaining_seashells (found : ℕ) (gave_away : ℕ) : ℕ :=
  found - gave_away

/-- Theorem stating that Joan has 16 seashells now. -/
theorem joan_has_16_seashells : 
  joans_remaining_seashells 79 63 = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_16_seashells_l809_80923


namespace NUMINAMATH_CALUDE_gcd_condition_equivalence_l809_80985

theorem gcd_condition_equivalence (m n : ℕ+) :
  (∀ (x y : ℕ+), x ∣ m → y ∣ n → Nat.gcd (x + y) (m * n) > 1) ↔ Nat.gcd m n > 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_condition_equivalence_l809_80985


namespace NUMINAMATH_CALUDE_mashed_potatoes_suggestion_l809_80914

theorem mashed_potatoes_suggestion (bacon : ℕ) (tomatoes : ℕ) (total : ℕ) 
  (h1 : bacon = 374) 
  (h2 : tomatoes = 128) 
  (h3 : total = 826) :
  total - (bacon + tomatoes) = 324 :=
by sorry

end NUMINAMATH_CALUDE_mashed_potatoes_suggestion_l809_80914


namespace NUMINAMATH_CALUDE_smallest_positive_integer_2002m_44444n_l809_80976

theorem smallest_positive_integer_2002m_44444n : 
  (∃ (k : ℕ+), ∀ (a : ℕ+), (∃ (m n : ℤ), a.val = 2002 * m + 44444 * n) → k ≤ a) ∧ 
  (∃ (m n : ℤ), (2 : ℕ+).val = 2002 * m + 44444 * n) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_2002m_44444n_l809_80976


namespace NUMINAMATH_CALUDE_tv_screen_horizontal_length_l809_80991

/-- Represents a rectangular TV screen --/
structure TVScreen where
  horizontal : ℝ
  vertical : ℝ
  diagonal : ℝ

/-- Theorem: Given a TV screen with horizontal to vertical ratio of 9:12 and
    diagonal of 32 inches, the horizontal length is 25.6 inches --/
theorem tv_screen_horizontal_length 
  (tv : TVScreen) 
  (ratio : tv.horizontal / tv.vertical = 9 / 12) 
  (diag : tv.diagonal = 32) :
  tv.horizontal = 25.6 := by
  sorry

#check tv_screen_horizontal_length

end NUMINAMATH_CALUDE_tv_screen_horizontal_length_l809_80991


namespace NUMINAMATH_CALUDE_pencil_cost_is_15_cents_l809_80906

/-- The cost of a pen in cents -/
def pen_cost : ℚ := sorry

/-- The cost of a pencil in cents -/
def pencil_cost : ℚ := sorry

/-- The total cost of 5 pens and 4 pencils in cents -/
def total_cost_1 : ℚ := 315

/-- The total cost of 3 pens and 6 pencils in cents -/
def total_cost_2 : ℚ := 243

theorem pencil_cost_is_15_cents :
  (5 * pen_cost + 4 * pencil_cost = total_cost_1) →
  (3 * pen_cost + 6 * pencil_cost = total_cost_2) →
  pencil_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_is_15_cents_l809_80906


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l809_80902

/-- 
Given a quadratic polynomial of the form 6x^2 + nx + 144, where n is an integer,
this theorem states that the largest value of n for which the polynomial 
can be factored as the product of two linear factors with integer coefficients is 865.
-/
theorem largest_n_for_factorization : 
  (∃ (n : ℤ), ∀ (A B : ℤ), 
    (6 * A = 6 ∧ A + 6 * B = n ∧ A * B = 144) → 
    (∀ (m : ℤ), (∃ (C D : ℤ), 6 * C = 6 ∧ C + 6 * D = m ∧ C * D = 144) → m ≤ n)) ∧
  (∃ (A B : ℤ), 6 * A = 6 ∧ A + 6 * B = 865 ∧ A * B = 144) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l809_80902


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l809_80900

theorem smallest_number_of_eggs :
  ∀ (total_eggs : ℕ) (num_containers : ℕ),
    total_eggs > 150 →
    total_eggs = 15 * num_containers - 3 →
    (∀ smaller_total : ℕ, smaller_total > 150 → smaller_total = 15 * (smaller_total / 15) - 3 → smaller_total ≥ total_eggs) →
    total_eggs = 162 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l809_80900


namespace NUMINAMATH_CALUDE_negation_of_existence_l809_80986

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x, x^2 - 1 < 0) ↔ (∀ x, x^2 - 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l809_80986


namespace NUMINAMATH_CALUDE_expression_equals_point_one_l809_80966

-- Define the expression
def expression : ℝ := (0.000001 ^ (1/2)) ^ (1/3)

-- State the theorem
theorem expression_equals_point_one : expression = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_point_one_l809_80966


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_surface_area_l809_80935

/-- The total surface area of a right pyramid with a regular hexagonal base -/
theorem hexagonal_pyramid_surface_area 
  (base_edge : ℝ) 
  (slant_height : ℝ) 
  (h : base_edge = 8) 
  (k : slant_height = 10) : 
  ∃ (area : ℝ), area = 48 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_surface_area_l809_80935


namespace NUMINAMATH_CALUDE_sum_first_10_terms_sequence_is_constant_sum_equals_first_term_times_n_l809_80908

/-- Sum of the first n terms of a geometric sequence with a₁ = 2 and r = 1 -/
def geometricSum (n : ℕ) : ℝ := 2 * n

/-- The geometric sequence with a₁ = 2 and r = 1 -/
def geometricSequence : ℕ → ℝ
  | 0 => 2
  | n + 1 => geometricSequence n

theorem sum_first_10_terms :
  geometricSum 10 = 20 := by sorry

theorem sequence_is_constant (n : ℕ) :
  geometricSequence n = 2 := by sorry

theorem sum_equals_first_term_times_n (n : ℕ) :
  geometricSum n = 2 * n := by sorry

end NUMINAMATH_CALUDE_sum_first_10_terms_sequence_is_constant_sum_equals_first_term_times_n_l809_80908


namespace NUMINAMATH_CALUDE_train_length_l809_80983

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (cross_time_s : ℝ) (h1 : speed_kmh = 90) (h2 : cross_time_s = 10) :
  speed_kmh * (1000 / 3600) * cross_time_s = 250 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l809_80983


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_2_sqrt_2_min_value_equality_l809_80997

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  ∀ x y, x > 0 → y > 0 → (x + 1)⁻¹ + (y + 1)⁻¹ = 1 → a + 2 * b ≤ x + 2 * y :=
by
  sorry

theorem min_value_is_2_sqrt_2 (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  a + 2 * b ≥ 2 * Real.sqrt 2 :=
by
  sorry

theorem min_value_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  (a + 2 * b = 2 * Real.sqrt 2) ↔ (a + 1 = Real.sqrt 2 * (b + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_2_sqrt_2_min_value_equality_l809_80997


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l809_80916

/-- The length of a rectangular plot in meters -/
def length : ℝ := 55

/-- The breadth of a rectangular plot in meters -/
def breadth : ℝ := 45

/-- The cost of fencing per meter in rupees -/
def cost_per_meter : ℝ := 26.50

/-- The total cost of fencing in rupees -/
def total_cost : ℝ := 5300

/-- Theorem stating the length of the rectangular plot -/
theorem rectangular_plot_length :
  (length = breadth + 10) ∧
  (total_cost = cost_per_meter * (2 * (length + breadth))) →
  length = 55 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l809_80916


namespace NUMINAMATH_CALUDE_ethan_reading_pages_l809_80921

/-- Represents the number of pages Ethan read on Saturday morning -/
def saturday_morning_pages : ℕ := sorry

/-- Represents the total number of pages in the book -/
def total_pages : ℕ := 360

/-- Represents the number of pages Ethan read on Saturday night -/
def saturday_night_pages : ℕ := 10

/-- Represents the number of pages left to read after Sunday -/
def pages_left : ℕ := 210

/-- The main theorem to prove -/
theorem ethan_reading_pages : 
  saturday_morning_pages = 40 ∧
  (saturday_morning_pages + saturday_night_pages) * 3 = total_pages - pages_left :=
sorry

end NUMINAMATH_CALUDE_ethan_reading_pages_l809_80921


namespace NUMINAMATH_CALUDE_candy_cost_l809_80990

theorem candy_cost (initial_amount pencil_cost remaining_amount : ℕ) 
  (h1 : initial_amount = 43)
  (h2 : pencil_cost = 20)
  (h3 : remaining_amount = 18) :
  initial_amount - pencil_cost - remaining_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_l809_80990


namespace NUMINAMATH_CALUDE_min_percentage_both_subjects_l809_80912

theorem min_percentage_both_subjects (total : ℝ) (physics_percentage : ℝ) (chemistry_percentage : ℝ)
  (h_physics : physics_percentage = 68)
  (h_chemistry : chemistry_percentage = 72)
  (h_total : total > 0) :
  (physics_percentage + chemistry_percentage - 100 : ℝ) = 40 := by
sorry

end NUMINAMATH_CALUDE_min_percentage_both_subjects_l809_80912


namespace NUMINAMATH_CALUDE_min_ratio_two_digit_integers_l809_80978

theorem min_ratio_two_digit_integers (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 → -- x and y are two-digit positive integers
  (x + y) / 2 = 55 → -- mean of x and y is 55
  ∀ a b : ℕ, 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ (a + b) / 2 = 55 →
  x / y ≤ a / b →
  x / y = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_min_ratio_two_digit_integers_l809_80978


namespace NUMINAMATH_CALUDE_min_distance_parallel_lines_l809_80941

/-- Given two parallel lines and a point between them, prove the minimum distance from the point to a fixed point. -/
theorem min_distance_parallel_lines (x₀ y₀ : ℝ) :
  (∃ (xp yp xq yq : ℝ),
    (xp - 2*yp - 1 = 0) ∧
    (xq - 2*yq + 3 = 0) ∧
    (x₀ = (xp + xq) / 2) ∧
    (y₀ = (yp + yq) / 2) ∧
    (y₀ > -x₀ + 2)) →
  Real.sqrt ((x₀ - 4)^2 + y₀^2) ≥ Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parallel_lines_l809_80941


namespace NUMINAMATH_CALUDE_sinusoidal_function_parameters_l809_80994

/-- 
Given a sinusoidal function y = a * sin(b * x + φ) where a > 0 and b > 0,
if the maximum value is 3 and the period is 2π/4, then a = 3 and b = 4.
-/
theorem sinusoidal_function_parameters 
  (a b φ : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : ∀ x, a * Real.sin (b * x + φ) ≤ 3)
  (h4 : ∃ x, a * Real.sin (b * x + φ) = 3)
  (h5 : (2 * Real.pi) / b = Real.pi / 2) : 
  a = 3 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_function_parameters_l809_80994


namespace NUMINAMATH_CALUDE_shopping_discount_l809_80989

theorem shopping_discount (shoe_price : ℝ) (dress_price : ℝ) 
  (shoe_discount : ℝ) (dress_discount : ℝ) (num_shoes : ℕ) :
  shoe_price = 50 →
  dress_price = 100 →
  shoe_discount = 0.4 →
  dress_discount = 0.2 →
  num_shoes = 2 →
  (num_shoes : ℝ) * shoe_price * (1 - shoe_discount) + 
    dress_price * (1 - dress_discount) = 140 := by
  sorry

end NUMINAMATH_CALUDE_shopping_discount_l809_80989


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l809_80975

theorem sum_of_reciprocals (a b : ℝ) (ha : a^2 - 3*a + 2 = 0) (hb : b^2 - 3*b + 2 = 0) (hab : a ≠ b) :
  1/a + 1/b = 3/2 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l809_80975


namespace NUMINAMATH_CALUDE_yang_hui_theorem_l809_80959

theorem yang_hui_theorem (a b : ℝ) 
  (sum : a + b = 3)
  (product : a * b = 1)
  (sum_squares : a^2 + b^2 = 7)
  (sum_cubes : a^3 + b^3 = 18)
  (sum_fourth_powers : a^4 + b^4 = 47) :
  a^5 + b^5 = 123 := by sorry

end NUMINAMATH_CALUDE_yang_hui_theorem_l809_80959


namespace NUMINAMATH_CALUDE_sqrt_inequality_fraction_product_inequality_l809_80979

-- Part 1
theorem sqrt_inequality : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

-- Part 2
theorem fraction_product_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_fraction_product_inequality_l809_80979


namespace NUMINAMATH_CALUDE_work_equivalence_first_group_size_correct_l809_80956

/-- The number of hours it takes the first group to complete the work -/
def first_group_hours : ℕ := 20

/-- The number of men in the second group -/
def second_group_men : ℕ := 15

/-- The number of hours it takes the second group to complete the work -/
def second_group_hours : ℕ := 48

/-- The number of men in the first group -/
def first_group_men : ℕ := 36

theorem work_equivalence :
  first_group_men * first_group_hours = second_group_men * second_group_hours :=
by sorry

/-- Proves that the number of men in the first group is correct -/
theorem first_group_size_correct :
  first_group_men = (second_group_men * second_group_hours) / first_group_hours :=
by sorry

end NUMINAMATH_CALUDE_work_equivalence_first_group_size_correct_l809_80956


namespace NUMINAMATH_CALUDE_year_2020_is_gengzi_l809_80962

/-- Represents the Heavenly Stems in the Sexagenary Cycle -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches in the Sexagenary Cycle -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Sexagenary Cycle -/
structure SexagenaryYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

/-- The Sexagenary Cycle system -/
def sexagenaryCycle : List SexagenaryYear := sorry

/-- Function to get the Sexagenary year for a given Gregorian year -/
def getSexagenaryYear (gregorianYear : Nat) : SexagenaryYear := sorry

/-- Theorem stating that 2020 corresponds to the GengZi year in the Sexagenary Cycle -/
theorem year_2020_is_gengzi :
  getSexagenaryYear 2020 = SexagenaryYear.mk HeavenlyStem.Geng EarthlyBranch.Zi :=
sorry

end NUMINAMATH_CALUDE_year_2020_is_gengzi_l809_80962


namespace NUMINAMATH_CALUDE_tree_planting_group_size_l809_80903

/-- Proves that the number of people in the first group is 3, given the conditions of the tree planting activity. -/
theorem tree_planting_group_size :
  ∀ (x : ℕ), 
    (12 : ℚ) / x = (36 : ℚ) / (x + 6) →
    x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_planting_group_size_l809_80903


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l809_80930

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation x^2 + 3x - 1 = 0 -/
def a : ℝ := 1
def b : ℝ := 3
def c : ℝ := -1

theorem quadratic_discriminant : discriminant a b c = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l809_80930


namespace NUMINAMATH_CALUDE_max_point_of_f_l809_80910

def f (x : ℝ) := 3 * x - x^3

theorem max_point_of_f :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≤ f x₀ ∧ x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_point_of_f_l809_80910


namespace NUMINAMATH_CALUDE_chessboard_number_property_l809_80947

theorem chessboard_number_property (n : ℕ) (X : Matrix (Fin n) (Fin n) ℝ) 
  (h : ∀ (i j k : Fin n), X i j + X j k + X k i = 0) :
  ∃ (t : Fin n → ℝ), ∀ (i j : Fin n), X i j = t i - t j := by
sorry

end NUMINAMATH_CALUDE_chessboard_number_property_l809_80947


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l809_80988

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof :
  rectangle_area 2500 10 = 200 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l809_80988


namespace NUMINAMATH_CALUDE_conditional_probability_l809_80952

/-- The total number of products in the box -/
def total_products : ℕ := 4

/-- The number of first-class products in the box -/
def first_class_products : ℕ := 3

/-- The number of second-class products in the box -/
def second_class_products : ℕ := 1

/-- Event A: "the first draw is a first-class product" -/
def event_A : Set ℕ := {1, 2, 3}

/-- Event B: "the second draw is a first-class product" -/
def event_B : Set ℕ := {1, 2}

/-- The probability of event A -/
def prob_A : ℚ := first_class_products / total_products

/-- The probability of event B given event A has occurred -/
def prob_B_given_A : ℚ := (first_class_products - 1) / (total_products - 1)

/-- The conditional probability of event B given event A -/
theorem conditional_probability :
  prob_B_given_A = 2/3 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_l809_80952


namespace NUMINAMATH_CALUDE_election_votes_l809_80925

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (60 * total_votes) / 100 - (40 * total_votes) / 100 = 240) : 
  (60 * total_votes) / 100 = 720 :=
sorry

end NUMINAMATH_CALUDE_election_votes_l809_80925


namespace NUMINAMATH_CALUDE_geometric_seq_increasing_condition_l809_80917

/-- A sequence is geometric if there exists a constant r such that aₙ₊₁ = r * aₙ for all n. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence is increasing if aₙ₊₁ > aₙ for all n. -/
def IsIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem geometric_seq_increasing_condition (a : ℕ → ℝ) (h : IsGeometric a) :
  (IsIncreasing a → a 2 > a 1) ∧ ¬(a 2 > a 1 → IsIncreasing a) :=
by sorry

end NUMINAMATH_CALUDE_geometric_seq_increasing_condition_l809_80917


namespace NUMINAMATH_CALUDE_same_wage_proportional_earnings_l809_80909

/-- Proves that maintaining the same hourly wage and weekly hours results in proportional earnings -/
theorem same_wage_proportional_earnings
  (seasonal_weeks : ℕ)
  (seasonal_earnings : ℝ)
  (new_weeks : ℕ)
  (new_earnings : ℝ)
  (h_seasonal_weeks : seasonal_weeks = 36)
  (h_seasonal_earnings : seasonal_earnings = 7200)
  (h_new_weeks : new_weeks = 18)
  (h_new_earnings : new_earnings = 3600)
  : (new_earnings / new_weeks) = (seasonal_earnings / seasonal_weeks) :=
by sorry

end NUMINAMATH_CALUDE_same_wage_proportional_earnings_l809_80909


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l809_80974

def is_solution (x y z : ℕ+) : Prop :=
  x^4 + y^4 + z^4 = 2*x^2*y^2 + 2*y^2*z^2 + 2*z^2*x^2 - 63

def solution_set : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(1, 4, 4), (4, 1, 4), (4, 4, 1), (2, 2, 3), (2, 3, 2), (3, 2, 2)}

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+, is_solution x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l809_80974


namespace NUMINAMATH_CALUDE_determine_b_l809_80943

theorem determine_b (a b : ℝ) (h1 : 3 * a + 4 = 1) (h2 : b - 2 * a = 5) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_determine_b_l809_80943


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l809_80980

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  swimmer : ℝ
  stream : ℝ

/-- Calculates the effective speed of the swimmer given the direction. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Represents the conditions of the swimming problem. -/
structure SwimmingProblem where
  downstreamDistance : ℝ
  upstreamDistance : ℝ
  time : ℝ

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 6 km/h. -/
theorem swimmer_speed_in_still_water (p : SwimmingProblem)
  (h1 : p.downstreamDistance = 72)
  (h2 : p.upstreamDistance = 36)
  (h3 : p.time = 9)
  : ∃ (s : SwimmerSpeeds),
    effectiveSpeed s true * p.time = p.downstreamDistance ∧
    effectiveSpeed s false * p.time = p.upstreamDistance ∧
    s.swimmer = 6 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l809_80980


namespace NUMINAMATH_CALUDE_factor_proof_l809_80951

theorem factor_proof (x y z : ℝ) : 
  ∃ (k : ℝ), x^2 - y^2 - z^2 + 2*y*z + x - y - z + 2 = (x - y + z + 1) * k :=
by sorry

end NUMINAMATH_CALUDE_factor_proof_l809_80951


namespace NUMINAMATH_CALUDE_one_nonnegative_solution_l809_80931

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -6*x :=
sorry

end NUMINAMATH_CALUDE_one_nonnegative_solution_l809_80931


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l809_80927

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 240) :
  a 9 - (1/3) * a 11 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l809_80927


namespace NUMINAMATH_CALUDE_polynomial_characterization_l809_80965

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that must be satisfied by the polynomial -/
def SatisfiesCondition (P : RealPolynomial) : Prop :=
  ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z ≠ 0 → 2 * x * y * z = x + y + z →
    (P x) / (y * z) + (P y) / (z * x) + (P z) / (x * y) = P (x - y) + P (y - z) + P (z - x)

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem polynomial_characterization (P : RealPolynomial) 
  (h : SatisfiesCondition P) : 
  ∃ (c : ℝ), ∀ (x : ℝ), P x = c * (x^2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l809_80965


namespace NUMINAMATH_CALUDE_smallest_AAAB_l809_80964

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def AB (a b : ℕ) : ℕ := 10 * a + b

def AAAB (a b : ℕ) : ℕ := 1000 * a + 100 * a + 10 * a + b

theorem smallest_AAAB :
  ∀ a b : ℕ,
    a ≠ b →
    a < 10 →
    b < 10 →
    is_two_digit (AB a b) →
    is_four_digit (AAAB a b) →
    7 * (AB a b) = AAAB a b →
    ∀ a' b' : ℕ,
      a' ≠ b' →
      a' < 10 →
      b' < 10 →
      is_two_digit (AB a' b') →
      is_four_digit (AAAB a' b') →
      7 * (AB a' b') = AAAB a' b' →
      AAAB a b ≤ AAAB a' b' →
    AAAB a b = 6661 :=
by sorry

end NUMINAMATH_CALUDE_smallest_AAAB_l809_80964


namespace NUMINAMATH_CALUDE_water_drinking_time_l809_80950

/-- Proves that given a goal of drinking 3 liters of water and drinking 500 milliliters every 2 hours, it will take 12 hours to reach the goal. -/
theorem water_drinking_time (goal : ℕ) (intake : ℕ) (frequency : ℕ) (h1 : goal = 3) (h2 : intake = 500) (h3 : frequency = 2) : 
  (goal * 1000) / intake * frequency = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_drinking_time_l809_80950


namespace NUMINAMATH_CALUDE_arrangement_theorem_l809_80940

/-- The number of ways to arrange n distinct objects --/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n distinct objects with k specific objects always adjacent --/
def permutations_with_adjacent (n k : ℕ) : ℕ :=
  permutations (n - k + 1) * permutations k

/-- The number of ways to arrange n distinct objects with k specific objects always adjacent
    and m specific objects never adjacent --/
def permutations_with_adjacent_and_not_adjacent (n k m : ℕ) : ℕ :=
  permutations_with_adjacent n k - permutations_with_adjacent (n - m + 1) (k + m - 1)

theorem arrangement_theorem :
  (permutations_with_adjacent 5 2 = 48) ∧
  (permutations_with_adjacent_and_not_adjacent 5 2 1 = 36) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l809_80940


namespace NUMINAMATH_CALUDE_max_profit_is_45_6_l809_80911

/-- Profit function for location A -/
def L₁ (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def L₂ (x : ℝ) : ℝ := 2 * x

/-- Total profit function -/
def S (x : ℝ) : ℝ := L₁ x + L₂ (15 - x)

/-- The maximum total profit is 45.6 when selling 15 cars across both locations -/
theorem max_profit_is_45_6 :
  ∃ x : ℕ, x ≤ 15 ∧ S x = 45.6 ∧ ∀ y : ℕ, y ≤ 15 → S y ≤ S x := by
  sorry

end NUMINAMATH_CALUDE_max_profit_is_45_6_l809_80911


namespace NUMINAMATH_CALUDE_min_disks_is_fifteen_l809_80955

/-- Represents the storage problem with given file sizes and quantities --/
structure StorageProblem where
  total_files : Nat
  disk_capacity : Real
  files_09mb : Nat
  files_08mb : Nat
  files_05mb : Nat
  h_total : total_files = files_09mb + files_08mb + files_05mb
  h_capacity : disk_capacity = 2

/-- Calculates the minimum number of disks required for the given storage problem --/
def min_disks_required (problem : StorageProblem) : Nat :=
  sorry

/-- The main theorem stating that the minimum number of disks required is 15 --/
theorem min_disks_is_fifteen :
  ∀ (problem : StorageProblem),
    problem.total_files = 35 →
    problem.files_09mb = 4 →
    problem.files_08mb = 15 →
    problem.files_05mb = 16 →
    min_disks_required problem = 15 :=
  sorry

end NUMINAMATH_CALUDE_min_disks_is_fifteen_l809_80955


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l809_80905

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and eccentricity √2,
    prove that the angle between its two asymptotes is π/2 -/
theorem hyperbola_asymptote_angle (a b : ℝ) (h : a > 0) (k : b > 0) :
  let e := Real.sqrt 2
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let eccentricity := fun (c : ℝ) => c / a
  eccentricity (Real.sqrt (a^2 + b^2)) = e →
  let asymptote_angle := fun (m₁ m₂ : ℝ) => Real.arctan ((m₂ - m₁) / (1 + m₁ * m₂))
  asymptote_angle (b/a) (-b/a) = π/2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l809_80905


namespace NUMINAMATH_CALUDE_mary_has_ten_marbles_l809_80913

/-- The number of blue marbles Dan has -/
def dans_marbles : ℕ := 5

/-- The ratio of Mary's marbles to Dan's marbles -/
def mary_to_dan_ratio : ℕ := 2

/-- The number of blue marbles Mary has -/
def marys_marbles : ℕ := mary_to_dan_ratio * dans_marbles

theorem mary_has_ten_marbles : marys_marbles = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_ten_marbles_l809_80913


namespace NUMINAMATH_CALUDE_remainder_of_1543_base12_div_9_l809_80961

/-- Converts a base-12 number to decimal --/
def base12ToDecimal (n : ℕ) : ℕ :=
  let d0 := n % 12
  let d1 := (n / 12) % 12
  let d2 := (n / 144) % 12
  let d3 := n / 1728
  d3 * 1728 + d2 * 144 + d1 * 12 + d0

/-- The base-12 number 1543 --/
def base12_1543 : ℕ := 1543

theorem remainder_of_1543_base12_div_9 :
  (base12ToDecimal base12_1543) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1543_base12_div_9_l809_80961


namespace NUMINAMATH_CALUDE_circle_number_determinable_l809_80971

/-- Represents a system of six circles connected by line segments -/
structure CircleSystem where
  /-- Numbers in the circles -/
  circle_numbers : Fin 6 → ℝ
  /-- Numbers on the segments connecting the circles -/
  segment_numbers : Fin 6 → ℝ
  /-- Each circle contains the sum of its incoming segment numbers -/
  sum_property : ∀ i : Fin 6, circle_numbers i = segment_numbers i + segment_numbers ((i + 5) % 6)

/-- The theorem stating that any circle's number can be determined from the other five -/
theorem circle_number_determinable (cs : CircleSystem) (i : Fin 6) :
  cs.circle_numbers i =
    cs.circle_numbers ((i + 1) % 6) +
    cs.circle_numbers ((i + 3) % 6) +
    cs.circle_numbers ((i + 5) % 6) -
    cs.circle_numbers ((i + 2) % 6) -
    cs.circle_numbers ((i + 4) % 6) :=
  sorry


end NUMINAMATH_CALUDE_circle_number_determinable_l809_80971


namespace NUMINAMATH_CALUDE_bags_given_away_bags_given_away_equals_two_l809_80918

def initial_purchase : ℕ := 3
def second_purchase : ℕ := 3
def remaining_bags : ℕ := 4

theorem bags_given_away : ℕ := by
  sorry

theorem bags_given_away_equals_two : bags_given_away = 2 := by
  sorry

end NUMINAMATH_CALUDE_bags_given_away_bags_given_away_equals_two_l809_80918


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l809_80942

theorem simplify_fraction_product : (360 / 32) * (10 / 240) * (16 / 6) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l809_80942


namespace NUMINAMATH_CALUDE_smallest_surface_area_l809_80998

def cube_surface_area (side : ℝ) : ℝ := 6 * side^2

def min_combined_surface_area (side1 side2 side3 : ℝ) : ℝ :=
  cube_surface_area side1 + cube_surface_area side2 + cube_surface_area side3 -
  (2 * side1^2 + 2 * side2^2 + 2 * side3^2)

theorem smallest_surface_area :
  min_combined_surface_area 3 5 8 = 502 := by
  sorry

end NUMINAMATH_CALUDE_smallest_surface_area_l809_80998


namespace NUMINAMATH_CALUDE_history_book_cost_l809_80993

/-- Given the following conditions:
  - Total number of books is 90
  - Math books cost $4 each
  - Total price of all books is $396
  - Number of math books bought is 54
  Prove that the cost of a history book is $5 -/
theorem history_book_cost (total_books : ℕ) (math_book_cost : ℕ) (total_price : ℕ) (math_books : ℕ) :
  total_books = 90 →
  math_book_cost = 4 →
  total_price = 396 →
  math_books = 54 →
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 :=
by sorry

end NUMINAMATH_CALUDE_history_book_cost_l809_80993


namespace NUMINAMATH_CALUDE_fruit_pie_theorem_l809_80939

/-- Represents the number of fruits needed for different types of pies -/
structure FruitRequirement where
  apples : ℕ
  pears : ℕ
  peaches : ℕ

/-- Calculates the total fruits needed for a given number of pies -/
def total_fruits (req : FruitRequirement) (num_pies : ℕ) : FruitRequirement :=
  { apples := req.apples * num_pies
  , pears := req.pears * num_pies
  , peaches := req.peaches * num_pies }

/-- Adds two FruitRequirement structures -/
def add_requirements (a b : FruitRequirement) : FruitRequirement :=
  { apples := a.apples + b.apples
  , pears := a.pears + b.pears
  , peaches := a.peaches + b.peaches }

theorem fruit_pie_theorem :
  let fruit_pie_req : FruitRequirement := { apples := 4, pears := 3, peaches := 0 }
  let apple_peach_pie_req : FruitRequirement := { apples := 6, pears := 0, peaches := 2 }
  let fruit_pies := 357
  let apple_peach_pies := 712
  let total_req := add_requirements (total_fruits fruit_pie_req fruit_pies) (total_fruits apple_peach_pie_req apple_peach_pies)
  total_req.apples = 5700 ∧ total_req.pears = 1071 ∧ total_req.peaches = 1424 := by
  sorry

end NUMINAMATH_CALUDE_fruit_pie_theorem_l809_80939


namespace NUMINAMATH_CALUDE_jerusha_earnings_l809_80949

theorem jerusha_earnings (L : ℝ) : 
  L + 4 * L = 85 → 4 * L = 68 := by
  sorry

end NUMINAMATH_CALUDE_jerusha_earnings_l809_80949


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l809_80915

theorem more_girls_than_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 35 → 
  3 * girls = 4 * boys → 
  total = boys + girls → 
  girls - boys = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l809_80915


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l809_80992

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  a₁ : ℤ  -- First term
  d : ℤ   -- Common difference

/-- The nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.a₁ + (n - 1 : ℤ) * seq.d

theorem tenth_term_of_specific_sequence :
  ∃ (seq : ArithmeticSequence),
    seq.a₁ = 10 ∧
    nthTerm seq 2 = 7 ∧
    nthTerm seq 3 = 4 ∧
    nthTerm seq 10 = -17 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l809_80992


namespace NUMINAMATH_CALUDE_negation_of_all_squares_positive_l809_80932

theorem negation_of_all_squares_positive :
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_positive_l809_80932


namespace NUMINAMATH_CALUDE_functional_equation_solution_l809_80948

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = x + f y) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l809_80948


namespace NUMINAMATH_CALUDE_triangle_division_2005_l809_80996

theorem triangle_division_2005 : ∃ n : ℕ, n^2 + (2005 - n^2)^2 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_triangle_division_2005_l809_80996


namespace NUMINAMATH_CALUDE_multiple_of_z_l809_80936

theorem multiple_of_z (x y z k : ℕ+) : 
  (3 * x.val = 4 * y.val) → 
  (3 * x.val = k * z.val) → 
  (x.val - y.val + z.val = 19) → 
  (∀ (x' y' z' : ℕ+), 3 * x'.val = 4 * y'.val → 3 * x'.val = k * z'.val → x'.val - y'.val + z'.val ≥ 19) →
  k = 12 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_z_l809_80936


namespace NUMINAMATH_CALUDE_unique_base_twelve_l809_80929

/-- Convert a base-b number to its decimal representation -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + b * acc) 0

/-- Check if all digits in a list are less than a given base -/
def valid_digits (digits : List Nat) (b : Nat) : Prop :=
  digits.all (· < b)

/-- The main theorem statement -/
theorem unique_base_twelve :
  ∃! b : Nat, 
    b > 1 ∧
    valid_digits [3, 0, 6] b ∧
    valid_digits [4, 2, 9] b ∧
    valid_digits [7, 4, 3] b ∧
    to_decimal [3, 0, 6] b + to_decimal [4, 2, 9] b = to_decimal [7, 4, 3] b :=
by
  sorry

end NUMINAMATH_CALUDE_unique_base_twelve_l809_80929


namespace NUMINAMATH_CALUDE_monotone_sin_range_l809_80957

theorem monotone_sin_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, Monotone (fun x ↦ f x)) →
  (∀ x ∈ Set.Icc 0 a, f x = Real.sin (2 * x + π / 3)) →
  a > 0 →
  0 < a ∧ a ≤ π / 12 := by
  sorry

end NUMINAMATH_CALUDE_monotone_sin_range_l809_80957


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l809_80907

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- Distance between a point and a vertical line -/
def distance_to_vertical_line (p : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  |p.1 - line_x|

theorem parabola_focus_distance 
  (P : ParabolaPoint) 
  (h : distance_to_vertical_line (P.x, P.y) (-2) = 5) :
  distance_to_vertical_line (P.x, P.y) 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l809_80907


namespace NUMINAMATH_CALUDE_q_function_equality_l809_80972

/-- Given a function q(x) that satisfies the equation
    q(x) + (2x^6 + 5x^4 + 10x) = (8x^4 + 35x^3 + 40x^2 + 2),
    prove that q(x) = -2x^6 + 3x^4 + 35x^3 + 40x^2 - 10x + 2 -/
theorem q_function_equality (q : ℝ → ℝ) :
  (∀ x, q x + (2 * x^6 + 5 * x^4 + 10 * x) = (8 * x^4 + 35 * x^3 + 40 * x^2 + 2)) →
  (∀ x, q x = -2 * x^6 + 3 * x^4 + 35 * x^3 + 40 * x^2 - 10 * x + 2) :=
by sorry

end NUMINAMATH_CALUDE_q_function_equality_l809_80972


namespace NUMINAMATH_CALUDE_fourth_root_equation_l809_80977

theorem fourth_root_equation (x : ℝ) (h : x > 0) :
  (x^3 * x^(1/2))^(1/4) = 4 → x = 4^(8/7) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_l809_80977


namespace NUMINAMATH_CALUDE_smallest_variance_most_stable_city_D_most_stable_l809_80963

/-- Represents a city with its cabbage price variance -/
structure City where
  name : String
  variance : ℝ

/-- Defines stability of cabbage prices based on variance -/
def is_most_stable (cities : List City) (c : City) : Prop :=
  ∀ city ∈ cities, c.variance ≤ city.variance

/-- The theorem stating that the city with the smallest variance is the most stable -/
theorem smallest_variance_most_stable (cities : List City) (c : City) 
    (h₁ : c ∈ cities) 
    (h₂ : ∀ city ∈ cities, c.variance ≤ city.variance) : 
    is_most_stable cities c := by
  sorry

/-- The specific problem instance -/
def problem_instance : List City :=
  [⟨"A", 18.3⟩, ⟨"B", 17.4⟩, ⟨"C", 20.1⟩, ⟨"D", 12.5⟩]

/-- The theorem applied to the specific problem instance -/
theorem city_D_most_stable : 
    is_most_stable problem_instance ⟨"D", 12.5⟩ := by
  sorry

end NUMINAMATH_CALUDE_smallest_variance_most_stable_city_D_most_stable_l809_80963


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l809_80953

theorem complex_fraction_simplification :
  (1/2 * 1/3 * 1/4 * 1/5 + 3/2 * 3/4 * 3/5) / (1/2 * 2/3 * 2/5) = 41/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l809_80953


namespace NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l809_80967

theorem geometric_sequence_eighth_term 
  (a : ℕ → ℝ) 
  (is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (fifth_term : a 5 = 11) 
  (eleventh_term : a 11 = 5) : 
  a 8 = Real.sqrt 55 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l809_80967


namespace NUMINAMATH_CALUDE_square_root_of_nine_l809_80973

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l809_80973


namespace NUMINAMATH_CALUDE_abs_neg_six_equals_six_l809_80938

theorem abs_neg_six_equals_six : |(-6 : ℤ)| = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_six_equals_six_l809_80938


namespace NUMINAMATH_CALUDE_find_number_l809_80946

theorem find_number (x : ℝ) : 6 + (1/2) * (1/3) * (1/5) * x = (1/15) * x → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l809_80946


namespace NUMINAMATH_CALUDE_inner_tangent_circle_radius_l809_80944

/-- Given a right triangle with legs 3 and 4 units, the radius of the circle
    tangent to both legs and the circumcircle internally is 2 units. -/
theorem inner_tangent_circle_radius (a b c r : ℝ) : 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → r = a + b - c → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_inner_tangent_circle_radius_l809_80944


namespace NUMINAMATH_CALUDE_intersecting_lines_coefficient_sum_l809_80901

/-- Two lines intersecting at a point implies a specific sum of their coefficients -/
theorem intersecting_lines_coefficient_sum 
  (m b : ℝ) 
  (h1 : 8 = m * 5 + 3) 
  (h2 : 8 = 4 * 5 + b) : 
  b + m = -11 := by sorry

end NUMINAMATH_CALUDE_intersecting_lines_coefficient_sum_l809_80901


namespace NUMINAMATH_CALUDE_trig_sum_equals_two_l809_80945

theorem trig_sum_equals_two : Real.cos (π / 4) ^ 2 + Real.tan (π / 3) * Real.cos (π / 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_two_l809_80945
