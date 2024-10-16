import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_for_square_root_solution_l968_96860

def is_square_integer (x : ℚ) : Prop :=
  ∃ m : ℤ, x = m^2

theorem smallest_n_for_square_root (n : ℕ) : Prop :=
  n ≥ 2 ∧ 
  is_square_integer ((n + 1) * (2 * n + 1) / 6) ∧
  ∀ k : ℕ, k ≥ 2 ∧ k < n → ¬is_square_integer ((k + 1) * (2 * k + 1) / 6)

theorem solution : smallest_n_for_square_root 337 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_square_root_solution_l968_96860


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l968_96849

theorem simplify_and_rationalize (a b c d e f g h i : ℝ) 
  (ha : a = 3) (hb : b = 7) (hc : c = 5) (hd : d = 8) (he : e = 6) (hf : f = 9) :
  (Real.sqrt a / Real.sqrt b) * (Real.sqrt c / Real.sqrt d) * (Real.sqrt e / Real.sqrt f) = 
  Real.sqrt 35 / 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l968_96849


namespace NUMINAMATH_CALUDE_range_of_x_l968_96895

theorem range_of_x (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -2) :
  x > 1 / 3 ∨ x < -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l968_96895


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l968_96851

theorem contrapositive_equivalence (a b : ℝ) :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l968_96851


namespace NUMINAMATH_CALUDE_complex_cube_root_l968_96811

theorem complex_cube_root (x y d : ℤ) (z : ℂ) : 
  x > 0 → y > 0 → z = x + y * Complex.I → z^3 = -54 + d * Complex.I → z = 3 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l968_96811


namespace NUMINAMATH_CALUDE_school_sample_size_l968_96875

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  stratum_size : ℕ
  stratum_sample : ℕ
  total_sample : ℕ

/-- Checks if the given stratified sample is proportional -/
def is_proportional_sample (s : StratifiedSample) : Prop :=
  s.stratum_sample * s.total_population = s.total_sample * s.stratum_size

/-- Theorem stating that for the given population and sample sizes, 
    the total sample size is 45 -/
theorem school_sample_size :
  ∀ (s : StratifiedSample), 
    s.total_population = 1500 →
    s.stratum_size = 400 →
    s.stratum_sample = 12 →
    is_proportional_sample s →
    s.total_sample = 45 := by
  sorry

end NUMINAMATH_CALUDE_school_sample_size_l968_96875


namespace NUMINAMATH_CALUDE_union_nonempty_iff_in_range_l968_96877

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | x^2 + (2*a - 3)*x + 2*a^2 - a - 3 = 0}

-- Define the set A (inferred from the problem)
def A (a : ℝ) : Set ℝ := {x | x^2 - (a - 2)*x - 2*a + 4 = 0}

-- Define the range of a
def range_a : Set ℝ := {a | a ≤ -6 ∨ (-7/2 ≤ a ∧ a ≤ 3/2) ∨ a ≥ 2}

-- Theorem statement
theorem union_nonempty_iff_in_range (a : ℝ) :
  (A a ∪ B a).Nonempty ↔ a ∈ range_a :=
sorry

end NUMINAMATH_CALUDE_union_nonempty_iff_in_range_l968_96877


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l968_96818

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 1 / (x - 3) + 1 / (x - 5) - 4 / (x - 4)
  ∃ (s : ℝ), s = 4 - Real.sqrt 2 ∧ 
    (f s = 0) ∧ 
    (∀ (x : ℝ), f x = 0 → x ≥ s) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l968_96818


namespace NUMINAMATH_CALUDE_ellipse_equation_l968_96816

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci are on the x-axis
  foci_on_x_axis : Bool
  -- Passes through (0,1) and (3,0)
  passes_through_points : (ℝ × ℝ) → (ℝ × ℝ) → Prop
  -- Eccentricity is 3/5
  eccentricity : ℚ
  -- Length of minor axis is 8
  minor_axis_length : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 16 = 1

/-- Theorem: The standard equation of the ellipse with given properties -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.foci_on_x_axis = true)
  (h2 : e.passes_through_points (0, 1) (3, 0))
  (h3 : e.eccentricity = 3/5)
  (h4 : e.minor_axis_length = 8) :
  ∀ x y : ℝ, standard_equation e x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l968_96816


namespace NUMINAMATH_CALUDE_product_of_numbers_l968_96838

theorem product_of_numbers (x y : ℝ) 
  (h1 : x - y = 9) 
  (h2 : x^2 + y^2 = 157) : 
  x * y = 22 := by sorry

end NUMINAMATH_CALUDE_product_of_numbers_l968_96838


namespace NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l968_96806

/-- The area of the shaded region formed by a square with side length 10 cm
    and four quarter circles drawn at its corners is equal to 100 - 25π cm². -/
theorem shaded_area_square_with_quarter_circles :
  let square_side : ℝ := 10
  let square_area : ℝ := square_side ^ 2
  let quarter_circle_radius : ℝ := square_side / 2
  let full_circle_area : ℝ := π * quarter_circle_radius ^ 2
  let shaded_area : ℝ := square_area - full_circle_area
  shaded_area = 100 - 25 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l968_96806


namespace NUMINAMATH_CALUDE_bus_rental_optimization_l968_96853

theorem bus_rental_optimization (total_people : ℕ) (small_bus_seats small_bus_cost : ℕ)
  (large_bus_seats large_bus_cost : ℕ) (total_buses : ℕ) :
  total_people = 600 →
  small_bus_seats = 32 →
  large_bus_seats = 45 →
  small_bus_cost + 2 * large_bus_cost = 2800 →
  large_bus_cost = (125 * small_bus_cost) / 100 →
  total_buses = 14 →
  ∃ (small_buses large_buses : ℕ),
    small_buses + large_buses = total_buses ∧
    small_buses * small_bus_seats + large_buses * large_bus_seats ≥ total_people ∧
    small_buses * small_bus_cost + large_buses * large_bus_cost = 13600 ∧
    ∀ (other_small other_large : ℕ),
      other_small + other_large = total_buses →
      other_small * small_bus_seats + other_large * large_bus_seats ≥ total_people →
      other_small * small_bus_cost + other_large * large_bus_cost ≥ 13600 :=
by sorry

end NUMINAMATH_CALUDE_bus_rental_optimization_l968_96853


namespace NUMINAMATH_CALUDE_divisors_of_720_l968_96881

theorem divisors_of_720 : Finset.card (Nat.divisors 720) = 30 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_720_l968_96881


namespace NUMINAMATH_CALUDE_marble_jar_problem_l968_96847

theorem marble_jar_problem (jar1_blue_ratio : ℚ) (jar1_green_ratio : ℚ)
  (jar2_blue_ratio : ℚ) (jar2_green_ratio : ℚ) (total_green : ℕ) :
  jar1_blue_ratio = 7 / 10 →
  jar1_green_ratio = 3 / 10 →
  jar2_blue_ratio = 6 / 10 →
  jar2_green_ratio = 4 / 10 →
  total_green = 80 →
  ∃ (total_jar1 total_jar2 : ℕ),
    total_jar1 = total_jar2 ∧
    (jar1_green_ratio * total_jar1 + jar2_green_ratio * total_jar2 : ℚ) = total_green ∧
    ⌊jar1_blue_ratio * total_jar1 - jar2_blue_ratio * total_jar2⌋ = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l968_96847


namespace NUMINAMATH_CALUDE_square_root_equation_l968_96896

theorem square_root_equation (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l968_96896


namespace NUMINAMATH_CALUDE_alice_bob_meeting_l968_96845

/-- The number of points on the circle -/
def n : ℕ := 18

/-- Alice's movement (clockwise) -/
def a : ℕ := 7

/-- Bob's movement (counterclockwise) -/
def b : ℕ := 13

/-- The number of turns it takes for Alice and Bob to meet again -/
def meetingTurns : ℕ := 9

/-- Theorem stating that Alice and Bob meet after 'meetingTurns' turns -/
theorem alice_bob_meeting :
  (meetingTurns * (a + n - b)) % n = 0 := by sorry

end NUMINAMATH_CALUDE_alice_bob_meeting_l968_96845


namespace NUMINAMATH_CALUDE_distribute_4_3_l968_96891

/-- The number of ways to distribute n indistinguishable objects into k distinct containers,
    with each container receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 36 ways to distribute 4 indistinguishable objects into 3 distinct containers,
    with each container receiving at least one object. -/
theorem distribute_4_3 : distribute 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_4_3_l968_96891


namespace NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l968_96862

theorem sphere_cylinder_equal_area (h : ℝ) (d : ℝ) (r : ℝ) :
  h = 16 →
  d = 16 →
  4 * Real.pi * r^2 = 2 * Real.pi * (d / 2) * h →
  r = 8 :=
by sorry

end NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l968_96862


namespace NUMINAMATH_CALUDE_spaghetti_pizza_ratio_l968_96825

/-- The total number of students surveyed -/
def total_students : ℕ := 800

/-- The number of students preferring lasagna -/
def lasagna_preference : ℕ := 150

/-- The number of students preferring manicotti -/
def manicotti_preference : ℕ := 120

/-- The number of students preferring ravioli -/
def ravioli_preference : ℕ := 180

/-- The number of students preferring spaghetti -/
def spaghetti_preference : ℕ := 200

/-- The number of students preferring pizza -/
def pizza_preference : ℕ := 150

/-- Theorem stating that the ratio of students preferring spaghetti to students preferring pizza is 4/3 -/
theorem spaghetti_pizza_ratio : 
  (spaghetti_preference : ℚ) / (pizza_preference : ℚ) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_spaghetti_pizza_ratio_l968_96825


namespace NUMINAMATH_CALUDE_f_seven_equals_f_nine_l968_96868

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f being decreasing on (8, +∞)
def DecreasingAfterEight (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 8 ∧ y > x → f y < f x

-- Define the property of f(x+8) being an even function
def EvenShiftedByEight (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 8) = f (-x + 8)

-- State the theorem
theorem f_seven_equals_f_nine
  (h1 : DecreasingAfterEight f)
  (h2 : EvenShiftedByEight f) :
  f 7 = f 9 := by
  sorry

end NUMINAMATH_CALUDE_f_seven_equals_f_nine_l968_96868


namespace NUMINAMATH_CALUDE_fraction_inverse_addition_l968_96832

theorem fraction_inverse_addition (a b : ℚ) (h : a ≠ b) :
  let c := -(a + b)
  (a + c) / (b + c) = b / a :=
by sorry

end NUMINAMATH_CALUDE_fraction_inverse_addition_l968_96832


namespace NUMINAMATH_CALUDE_descending_order_proof_l968_96855

theorem descending_order_proof (a b c d : ℝ) : 
  a = Real.sin (33 * π / 180) →
  b = Real.cos (55 * π / 180) →
  c = Real.tan (35 * π / 180) →
  d = Real.log 5 →
  d > c ∧ c > b ∧ b > a :=
by sorry

end NUMINAMATH_CALUDE_descending_order_proof_l968_96855


namespace NUMINAMATH_CALUDE_problem_statement_l968_96865

theorem problem_statement (a b x y : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h1 : a = 2 * b)
  (h2 : x = 3 * y)
  (h3 : a + b = x * y)
  (h4 : b = 4)
  (h5 : y = 2) :
  x * a = 48 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l968_96865


namespace NUMINAMATH_CALUDE_fraction_problem_l968_96856

theorem fraction_problem (a b : ℝ) (f : ℝ) : 
  a - b = 8 → 
  a + b = 24 → 
  f * (a + b) = 6 → 
  f = 1/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l968_96856


namespace NUMINAMATH_CALUDE_no_integer_solution_l968_96801

theorem no_integer_solution : ∀ x y : ℤ, 5 * x^2 - 4 * y^2 ≠ 2017 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l968_96801


namespace NUMINAMATH_CALUDE_garden_occupation_fraction_l968_96864

theorem garden_occupation_fraction :
  ∀ (garden_length garden_width : ℝ)
    (trapezoid_short_side trapezoid_long_side : ℝ)
    (sandbox_side : ℝ),
  garden_length = 40 →
  garden_width = 8 →
  trapezoid_long_side - trapezoid_short_side = 10 →
  trapezoid_short_side + trapezoid_long_side = garden_length →
  sandbox_side = 5 →
  let triangle_leg := (trapezoid_long_side - trapezoid_short_side) / 2
  let triangle_area := triangle_leg ^ 2 / 2
  let total_triangles_area := 2 * triangle_area
  let sandbox_area := sandbox_side ^ 2
  let occupied_area := total_triangles_area + sandbox_area
  let garden_area := garden_length * garden_width
  occupied_area / garden_area = 5 / 32 :=
by sorry

end NUMINAMATH_CALUDE_garden_occupation_fraction_l968_96864


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l968_96841

theorem perfect_square_trinomial (x y k : ℝ) : 
  (∃ a : ℝ, x^2 + k*x*y + 64*y^2 = a^2) → k = 16 ∨ k = -16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l968_96841


namespace NUMINAMATH_CALUDE_modular_inverse_17_mod_800_l968_96803

theorem modular_inverse_17_mod_800 : ∃ x : ℕ, x < 800 ∧ (17 * x) % 800 = 1 :=
by
  use 47
  sorry

end NUMINAMATH_CALUDE_modular_inverse_17_mod_800_l968_96803


namespace NUMINAMATH_CALUDE_nine_rooks_on_checkerboard_l968_96886

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (is_checkerboard : Bool)

/-- Represents a rook placement on a chessboard -/
structure RookPlacement :=
  (board : Chessboard)
  (num_rooks : Nat)
  (same_color : Bool)
  (non_attacking : Bool)

/-- Counts the number of valid rook placements -/
def count_rook_placements (placement : RookPlacement) : Nat :=
  sorry

/-- Theorem: The number of ways to place 9 non-attacking rooks on cells of the same color on a 9x9 checkerboard is 2880 -/
theorem nine_rooks_on_checkerboard :
  ∀ (board : Chessboard) (placement : RookPlacement),
    board.size = 9 ∧
    board.is_checkerboard = true ∧
    placement.board = board ∧
    placement.num_rooks = 9 ∧
    placement.same_color = true ∧
    placement.non_attacking = true →
    count_rook_placements placement = 2880 :=
  sorry

end NUMINAMATH_CALUDE_nine_rooks_on_checkerboard_l968_96886


namespace NUMINAMATH_CALUDE_average_shift_l968_96834

theorem average_shift (x₁ x₂ x₃ : ℝ) (h : (x₁ + x₂ + x₃) / 3 = 40) :
  ((x₁ + 40) + (x₂ + 40) + (x₃ + 40)) / 3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_shift_l968_96834


namespace NUMINAMATH_CALUDE_max_visible_cubes_9x9x9_l968_96854

/-- Represents a cube made of unit cubes -/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point -/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  let face_size := cube.size^2
  let edge_size := cube.size - 1
  3 * face_size - 3 * edge_size + 1

/-- Theorem: For a 9×9×9 cube, the maximum number of visible unit cubes is 220 -/
theorem max_visible_cubes_9x9x9 :
  max_visible_cubes ⟨9⟩ = 220 := by sorry

end NUMINAMATH_CALUDE_max_visible_cubes_9x9x9_l968_96854


namespace NUMINAMATH_CALUDE_student_subject_assignment_l968_96800

/-- The number of ways to assign students to subjects. -/
def num_assignments (n : ℕ) (k : ℕ) : ℕ := k^n

/-- The number of students. -/
def num_students : ℕ := 4

/-- The number of subjects. -/
def num_subjects : ℕ := 3

theorem student_subject_assignment :
  num_assignments num_students num_subjects = 81 := by
  sorry

end NUMINAMATH_CALUDE_student_subject_assignment_l968_96800


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l968_96815

theorem negation_of_existence_proposition :
  (¬∃ x : ℝ, x > 0 ∧ Real.sin x > 2^x - 1) ↔ (∀ x : ℝ, x > 0 → Real.sin x ≤ 2^x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l968_96815


namespace NUMINAMATH_CALUDE_parabola_reflection_difference_l968_96870

/-- Represents a quadratic function of the form ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The function representing the original parabola translated up by 3 units --/
def f (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c + 3

/-- The function representing the reflected parabola translated down by 3 units --/
def g (p : Parabola) (x : ℝ) : ℝ :=
  -p.a * x^2 - p.b * x - p.c - 3

/-- Theorem stating that (f-g)(x) equals 2ax^2 + 2bx + 2c + 6 --/
theorem parabola_reflection_difference (p : Parabola) (x : ℝ) :
  f p x - g p x = 2 * p.a * x^2 + 2 * p.b * x + 2 * p.c + 6 := by
  sorry


end NUMINAMATH_CALUDE_parabola_reflection_difference_l968_96870


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l968_96878

theorem sqrt_product_equality (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt (12 * x) * Real.sqrt (20 * x) * Real.sqrt (5 * x) * Real.sqrt (30 * x) = 30) :
  x = 1 / Real.sqrt 20 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l968_96878


namespace NUMINAMATH_CALUDE_no_double_application_function_l968_96809

theorem no_double_application_function :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 2019 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l968_96809


namespace NUMINAMATH_CALUDE_discount_difference_l968_96821

theorem discount_difference : 
  let original_bill : ℝ := 12000
  let single_discount_rate : ℝ := 0.35
  let first_successive_discount_rate : ℝ := 0.30
  let second_successive_discount_rate : ℝ := 0.06
  let single_discount_amount : ℝ := original_bill * (1 - single_discount_rate)
  let successive_discount_amount : ℝ := original_bill * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)
  successive_discount_amount - single_discount_amount = 96 := by
sorry

end NUMINAMATH_CALUDE_discount_difference_l968_96821


namespace NUMINAMATH_CALUDE_optimal_production_consumption_theorem_l968_96822

/-- Represents a country's production capabilities and consumption --/
structure Country where
  eggplant_production : ℝ
  corn_production : ℝ
  consumption : ℝ × ℝ

/-- The global market for agricultural products --/
structure Market where
  price : ℝ

/-- Calculates the optimal production and consumption for two countries --/
def optimal_production_and_consumption (a b : Country) (m : Market) : (Country × Country) :=
  sorry

/-- Main theorem: Optimal production and consumption for countries A and B --/
theorem optimal_production_consumption_theorem (a b : Country) (m : Market) :
  a.eggplant_production = 10 ∧
  a.corn_production = 8 ∧
  b.eggplant_production = 18 ∧
  b.corn_production = 12 ∧
  m.price > 0 →
  let (a', b') := optimal_production_and_consumption a b m
  a'.consumption = (4, 4) ∧ b'.consumption = (9, 9) :=
sorry

end NUMINAMATH_CALUDE_optimal_production_consumption_theorem_l968_96822


namespace NUMINAMATH_CALUDE_certain_number_problem_l968_96839

theorem certain_number_problem : ∃ x : ℚ, (24 : ℚ) = (4/5) * x + 4 ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l968_96839


namespace NUMINAMATH_CALUDE_amazon_tide_problem_l968_96810

theorem amazon_tide_problem (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (2 * x + φ)) →
  (abs φ < π / 2) →
  (∀ x, f (x - π / 3) = -f (-x - π / 3)) →
  (φ = -π / 3) ∧
  (∀ x, f (5 * π / 12 + x) = f (5 * π / 12 - x)) ∧
  (∀ x ∈ Set.Icc (-π / 3) (-π / 6), ∀ y ∈ Set.Icc (-π / 3) (-π / 6), x < y → f x > f y) ∧
  (∃ x ∈ Set.Ioo 0 (π / 2), (deriv f) x = 0) :=
by sorry

end NUMINAMATH_CALUDE_amazon_tide_problem_l968_96810


namespace NUMINAMATH_CALUDE_cubic_fraction_value_l968_96824

theorem cubic_fraction_value : 
  let a : ℝ := 8
  let b : ℝ := 8 - 1
  (a^3 + b^3) / (a^2 - a*b + b^2) = 15 := by sorry

end NUMINAMATH_CALUDE_cubic_fraction_value_l968_96824


namespace NUMINAMATH_CALUDE_point_on_x_axis_l968_96872

/-- 
A point P with coordinates (3+a, a-5) lies on the x-axis in a Cartesian coordinate system.
Prove that a = 5.
-/
theorem point_on_x_axis (a : ℝ) : (a - 5 = 0) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l968_96872


namespace NUMINAMATH_CALUDE_average_of_multiples_10_to_400_l968_96873

def multiples_of_10 (n : ℕ) : List ℕ :=
  (List.range ((400 - 10) / 10 + 1)).map (λ i => 10 * (i + 1))

theorem average_of_multiples_10_to_400 :
  (List.sum (multiples_of_10 400)) / (List.length (multiples_of_10 400)) = 205 := by
  sorry

end NUMINAMATH_CALUDE_average_of_multiples_10_to_400_l968_96873


namespace NUMINAMATH_CALUDE_average_age_is_35_l968_96831

/-- The average age of Omi, Kimiko, and Arlette is 35 years old. -/
theorem average_age_is_35 (kimiko_age omi_age arlette_age : ℕ) : 
  kimiko_age = 28 →
  omi_age = 2 * kimiko_age →
  arlette_age = 3 * kimiko_age / 4 →
  (omi_age + kimiko_age + arlette_age) / 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_age_is_35_l968_96831


namespace NUMINAMATH_CALUDE_slope_range_ordinate_range_l968_96867

-- Define the point A
def A : ℝ × ℝ := (0, 3)

-- Define the line l
def line_l (x : ℝ) : ℝ := 2 * x - 4

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def conditions (C : Circle) (k : ℝ) : Prop :=
  C.radius = 1 ∧
  C.center.2 = line_l C.center.1 ∧
  C.center.2 = C.center.1 - 1 ∧
  ∃ (x y : ℝ), (x - C.center.1)^2 + (y - C.center.2)^2 = 1 ∧ y = k * x + 3

-- Define the theorems to be proved
theorem slope_range (C : Circle) :
  (∃ k, conditions C k) → ∃ k, -3/4 ≤ k ∧ k ≤ 0 :=
sorry

theorem ordinate_range (C : Circle) :
  (∃ M : ℝ × ℝ, (M.1 - C.center.1)^2 + (M.2 - C.center.2)^2 = 1 ∧
   (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4 * ((M.1 - 0)^2 + (M.2 - 0)^2)) →
  -4 ≤ C.center.2 ∧ C.center.2 ≤ 4/5 :=
sorry

end NUMINAMATH_CALUDE_slope_range_ordinate_range_l968_96867


namespace NUMINAMATH_CALUDE_positive_function_condition_l968_96880

theorem positive_function_condition (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → (2 - a^2) * x + a > 0) ↔ (0 < a ∧ a < 2) := by
  sorry

end NUMINAMATH_CALUDE_positive_function_condition_l968_96880


namespace NUMINAMATH_CALUDE_finite_correct_numbers_l968_96899

theorem finite_correct_numbers (k : ℕ) (h : k > 1) : 
  Set.Finite {n : ℕ | n > 1 ∧ 
                      Nat.Coprime n k ∧ 
                      ∀ d : ℕ, d < n → d ∣ n → ¬Nat.Coprime (d + k) n} := by
  sorry

end NUMINAMATH_CALUDE_finite_correct_numbers_l968_96899


namespace NUMINAMATH_CALUDE_value_of_x_l968_96869

theorem value_of_x (w y z x : ℕ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 8) : 
  x = 138 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l968_96869


namespace NUMINAMATH_CALUDE_breath_holding_improvement_l968_96814

/-- Calculates the final breath-holding time after three weeks of practice --/
def final_breath_holding_time (initial_time : ℝ) : ℝ :=
  let after_first_week := initial_time * 2
  let after_second_week := after_first_week * 2
  after_second_week * 1.5

/-- Theorem stating that given an initial breath-holding time of 10 seconds,
    the final time after three weeks of practice is 60 seconds --/
theorem breath_holding_improvement :
  final_breath_holding_time 10 = 60 := by
  sorry

#eval final_breath_holding_time 10

end NUMINAMATH_CALUDE_breath_holding_improvement_l968_96814


namespace NUMINAMATH_CALUDE_bottles_needed_to_fill_container_l968_96823

def craft_bottle_volume : ℕ := 150
def decorative_container_volume : ℕ := 2650

theorem bottles_needed_to_fill_container : 
  ∃ n : ℕ, n * craft_bottle_volume ≥ decorative_container_volume ∧ 
  ∀ m : ℕ, m * craft_bottle_volume ≥ decorative_container_volume → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_bottles_needed_to_fill_container_l968_96823


namespace NUMINAMATH_CALUDE_meal_cost_calculation_l968_96888

/-- Proves that given a meal with specific tax and tip rates, and a total cost,
    the original meal cost can be determined. -/
theorem meal_cost_calculation (total_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
    (h_total : total_cost = 36.90)
    (h_tax : tax_rate = 0.09)
    (h_tip : tip_rate = 0.18) :
    ∃ (original_cost : ℝ), 
      original_cost * (1 + tax_rate + tip_rate) = total_cost ∧ 
      original_cost = 29 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_calculation_l968_96888


namespace NUMINAMATH_CALUDE_inscribed_square_area_l968_96804

def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/8 = 1

def inscribed_square (s : ℝ) : Prop :=
  ellipse s s ∧ ellipse (-s) s ∧ ellipse s (-s) ∧ ellipse (-s) (-s)

theorem inscribed_square_area :
  ∃ s : ℝ, inscribed_square s ∧ (2*s)^2 = 32/3 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l968_96804


namespace NUMINAMATH_CALUDE_fraction_existence_and_nonexistence_l968_96840

theorem fraction_existence_and_nonexistence :
  (∀ n : ℕ+, ∃ a b : ℕ+, (Real.sqrt n : ℝ) ≤ (a : ℝ) / (b : ℝ) ∧
                         (a : ℝ) / (b : ℝ) ≤ Real.sqrt (n + 1) ∧
                         (b : ℝ) ≤ Real.sqrt n + 1) ∧
  (∃ f : ℕ → ℕ+, ∀ k : ℕ, ∀ a b : ℕ+,
    (Real.sqrt (f k) : ℝ) ≤ (a : ℝ) / (b : ℝ) →
    (a : ℝ) / (b : ℝ) ≤ Real.sqrt (f k + 1) →
    (b : ℝ) > Real.sqrt (f k)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_existence_and_nonexistence_l968_96840


namespace NUMINAMATH_CALUDE_prism_21_edges_has_9_faces_l968_96842

/-- Represents a prism with a given number of edges. -/
structure Prism where
  edges : ℕ

/-- Calculates the number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  (p.edges / 3) + 2

/-- Theorem stating that a prism with 21 edges has 9 faces. -/
theorem prism_21_edges_has_9_faces :
  ∀ (p : Prism), p.edges = 21 → num_faces p = 9 := by
  sorry

#eval num_faces { edges := 21 }

end NUMINAMATH_CALUDE_prism_21_edges_has_9_faces_l968_96842


namespace NUMINAMATH_CALUDE_reservoir_capacity_proof_l968_96848

theorem reservoir_capacity_proof (current_level : ℝ) (normal_level : ℝ) (total_capacity : ℝ)
  (h1 : current_level = 30)
  (h2 : current_level = 2 * normal_level)
  (h3 : current_level = 0.75 * total_capacity) :
  total_capacity - normal_level = 25 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_capacity_proof_l968_96848


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l968_96844

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (3 ∣ n) ∧ (4 ∣ n) ∧ (7 ∣ n) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (3 ∣ m) ∧ (4 ∣ m) ∧ (7 ∣ m) → n ≤ m) ∧
  n = 168 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l968_96844


namespace NUMINAMATH_CALUDE_target_practice_probabilities_l968_96805

/-- Represents a shooter in the target practice scenario -/
structure Shooter where
  hit_probability : ℝ
  num_shots : ℕ

/-- Calculates the probability of the given event -/
def calculate_probability (s1 s2 : Shooter) (event : Shooter → Shooter → ℝ) : ℝ :=
  event s1 s2

/-- The scenario with two shooters -/
def target_practice_scenario : Prop :=
  ∃ (s1 s2 : Shooter),
    s1.hit_probability = 0.8 ∧
    s2.hit_probability = 0.6 ∧
    s1.num_shots = 2 ∧
    s2.num_shots = 3 ∧
    (calculate_probability s1 s2 (λ _ _ => 0.99744) = 
     calculate_probability s1 s2 (λ s1 s2 => 1 - (1 - s1.hit_probability)^s1.num_shots * (1 - s2.hit_probability)^s2.num_shots)) ∧
    (calculate_probability s1 s2 (λ _ _ => 0.13824) = 
     calculate_probability s1 s2 (λ s1 s2 => (s1.num_shots * s1.hit_probability * (1 - s1.hit_probability)) * 
                                             (Nat.choose s2.num_shots 2 * s2.hit_probability^2 * (1 - s2.hit_probability)))) ∧
    (calculate_probability s1 s2 (λ _ _ => 0.87328) = 
     calculate_probability s1 s2 (λ s1 s2 => 1 - (1 - s1.hit_probability^2) * 
                                             (1 - s2.hit_probability^2 - s2.hit_probability^3))) ∧
    (calculate_probability s1 s2 (λ _ _ => 0.032) = 
     calculate_probability s1 s2 (λ s1 s2 => (s1.num_shots * s1.hit_probability * (1 - s1.hit_probability)^(s1.num_shots - 1) * (1 - s2.hit_probability)^s2.num_shots) + 
                                             ((1 - s1.hit_probability)^s1.num_shots * s2.num_shots * s2.hit_probability * (1 - s2.hit_probability)^(s2.num_shots - 1))))

theorem target_practice_probabilities : target_practice_scenario := sorry

end NUMINAMATH_CALUDE_target_practice_probabilities_l968_96805


namespace NUMINAMATH_CALUDE_triangle_formation_l968_96817

/-- A function that checks if three stick lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem stating which set of stick lengths can form a triangle -/
theorem triangle_formation :
  can_form_triangle 2 3 4 ∧
  ¬can_form_triangle 3 7 2 ∧
  ¬can_form_triangle 3 3 7 ∧
  ¬can_form_triangle 1 2 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l968_96817


namespace NUMINAMATH_CALUDE_real_root_quadratic_l968_96813

theorem real_root_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - (1 - Complex.I) * x + m + 2 * Complex.I = 0) → m = -6 := by
  sorry

end NUMINAMATH_CALUDE_real_root_quadratic_l968_96813


namespace NUMINAMATH_CALUDE_divisibility_problem_l968_96885

theorem divisibility_problem (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 24)
  (h2 : Nat.gcd b c = 36)
  (h3 : Nat.gcd c d = 54)
  (h4 : 70 < Nat.gcd d a ∧ Nat.gcd d a < 100) :
  13 ∣ a.val := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l968_96885


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l968_96893

/-- Represents a systematic sampling scheme. -/
structure SystematicSampling where
  totalItems : ℕ
  sampleSize : ℕ
  startingPoint : ℕ
  samplingInterval : ℕ

/-- Generates the sample numbers for a given systematic sampling scheme. -/
def generateSampleNumbers (s : SystematicSampling) : List ℕ :=
  List.range s.sampleSize |>.map (fun i => s.startingPoint + i * s.samplingInterval)

/-- Theorem: The correct sample numbers for systematic sampling of 5 items from 50 products
    are 9, 19, 29, 39, 49. -/
theorem correct_systematic_sample :
  let s : SystematicSampling := {
    totalItems := 50,
    sampleSize := 5,
    startingPoint := 9,
    samplingInterval := 10
  }
  generateSampleNumbers s = [9, 19, 29, 39, 49] := by
  sorry


end NUMINAMATH_CALUDE_correct_systematic_sample_l968_96893


namespace NUMINAMATH_CALUDE_divisible_by_nine_l968_96887

theorem divisible_by_nine : ∃ k : ℤ, 2^10 - 2^8 + 2^6 - 2^4 + 2^2 - 1 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l968_96887


namespace NUMINAMATH_CALUDE_constant_b_value_l968_96866

theorem constant_b_value (x y : ℝ) (b : ℝ) 
  (h1 : (7 * x + b * y) / (x - 2 * y) = 13)
  (h2 : x / (2 * y) = 5 / 2) :
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_constant_b_value_l968_96866


namespace NUMINAMATH_CALUDE_amc_10_2007_scoring_l968_96826

/-- The minimum number of correctly solved problems to achieve the target score -/
def min_correct_problems (total_problems : ℕ) (attempted_problems : ℕ) (correct_points : ℕ) (unanswered_points : ℕ) (target_score : ℕ) : ℕ :=
  let unanswered := total_problems - attempted_problems
  let unanswered_score := unanswered * unanswered_points
  let required_score := target_score - unanswered_score
  (required_score + correct_points - 1) / correct_points

theorem amc_10_2007_scoring :
  min_correct_problems 30 27 7 2 150 = 21 := by
  sorry

end NUMINAMATH_CALUDE_amc_10_2007_scoring_l968_96826


namespace NUMINAMATH_CALUDE_simplified_expression_l968_96858

theorem simplified_expression : 
  (2 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7) = 
  Real.sqrt 6 + 2 * Real.sqrt 2 - Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_l968_96858


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l968_96882

theorem quadratic_equation_integer_roots :
  let S : Set ℝ := {a : ℝ | a > 0 ∧ ∃ x y : ℤ, x ≠ y ∧ a^2 * x^2 + a * x + 1 - 13 * a^2 = 0 ∧ a^2 * y^2 + a * y + 1 - 13 * a^2 = 0}
  S = {1, 1/3, 1/4} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l968_96882


namespace NUMINAMATH_CALUDE_total_books_and_magazines_l968_96827

def books_per_shelf : ℕ := 23
def magazines_per_shelf : ℕ := 61
def number_of_shelves : ℕ := 29

theorem total_books_and_magazines :
  books_per_shelf * number_of_shelves + magazines_per_shelf * number_of_shelves = 2436 := by
  sorry

end NUMINAMATH_CALUDE_total_books_and_magazines_l968_96827


namespace NUMINAMATH_CALUDE_bisection_next_step_l968_96836

/-- The bisection method's next step for a function with given properties -/
theorem bisection_next_step (f : ℝ → ℝ) (h1 : f 1 < 0) (h2 : f 1.5 > 0) :
  let x₀ : ℝ := (1 + 1.5) / 2
  x₀ = 1.25 := by sorry

end NUMINAMATH_CALUDE_bisection_next_step_l968_96836


namespace NUMINAMATH_CALUDE_max_kitchen_towel_sets_is_13_l968_96837

-- Define the given parameters
def budget : ℚ := 600
def guest_bathroom_sets : ℕ := 2
def master_bathroom_sets : ℕ := 4
def hand_towel_sets : ℕ := 3
def guest_bathroom_price : ℚ := 40
def master_bathroom_price : ℚ := 50
def hand_towel_price : ℚ := 30
def kitchen_towel_price : ℚ := 20
def guest_bathroom_discount : ℚ := 0.15
def master_bathroom_discount : ℚ := 0.20
def hand_towel_discount : ℚ := 0.15
def kitchen_towel_discount : ℚ := 0.10
def sales_tax : ℚ := 0.08

-- Define the function to calculate the maximum number of kitchen towel sets
def max_kitchen_towel_sets : ℕ :=
  let guest_bathroom_cost := guest_bathroom_sets * guest_bathroom_price * (1 - guest_bathroom_discount)
  let master_bathroom_cost := master_bathroom_sets * master_bathroom_price * (1 - master_bathroom_discount)
  let hand_towel_cost := hand_towel_sets * hand_towel_price * (1 - hand_towel_discount)
  let total_cost_before_tax := guest_bathroom_cost + master_bathroom_cost + hand_towel_cost
  let total_cost_after_tax := total_cost_before_tax * (1 + sales_tax)
  let remaining_budget := budget - total_cost_after_tax
  let kitchen_towel_cost_after_tax := kitchen_towel_price * (1 - kitchen_towel_discount) * (1 + sales_tax)
  (remaining_budget / kitchen_towel_cost_after_tax).floor.toNat

-- Theorem statement
theorem max_kitchen_towel_sets_is_13 : max_kitchen_towel_sets = 13 := by
  sorry


end NUMINAMATH_CALUDE_max_kitchen_towel_sets_is_13_l968_96837


namespace NUMINAMATH_CALUDE_tablet_down_payment_is_100_l968_96892

/-- The down payment for a tablet purchase with given conditions. -/
def tablet_down_payment (cash_price installment_total first_4_months next_4_months last_4_months cash_savings : ℕ) : ℕ :=
  installment_total - (4 * first_4_months + 4 * next_4_months + 4 * last_4_months)

/-- Theorem stating that the down payment for the tablet is $100 under given conditions. -/
theorem tablet_down_payment_is_100 :
  tablet_down_payment 450 520 40 35 30 70 = 100 := by
  sorry

end NUMINAMATH_CALUDE_tablet_down_payment_is_100_l968_96892


namespace NUMINAMATH_CALUDE_expression_evaluation_l968_96812

theorem expression_evaluation :
  let a : ℚ := 2
  let b : ℚ := 1
  let expr := -1/3 * (a^3*b - a*b) + a*b^3 - (a*b - b)/2 - 1/2*b + 1/3*a^3*b
  expr = 5/3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l968_96812


namespace NUMINAMATH_CALUDE_total_pennies_donated_l968_96843

def cassandra_pennies : ℕ := 5000
def james_difference : ℕ := 276

theorem total_pennies_donated (cassandra : ℕ) (james_diff : ℕ) 
  (h1 : cassandra = cassandra_pennies) 
  (h2 : james_diff = james_difference) : 
  cassandra + (cassandra - james_diff) = 9724 :=
by
  sorry

end NUMINAMATH_CALUDE_total_pennies_donated_l968_96843


namespace NUMINAMATH_CALUDE_no_pythagorean_solution_for_prime_congruent_to_neg_one_mod_four_l968_96884

theorem no_pythagorean_solution_for_prime_congruent_to_neg_one_mod_four 
  (p : Nat) (hp : Prime p) (hp_cong : p % 4 = 3) :
  ∀ n : Nat, n > 0 → ¬∃ x y : Nat, x > 0 ∧ y > 0 ∧ x^2 + y^2 = p^n :=
by sorry

end NUMINAMATH_CALUDE_no_pythagorean_solution_for_prime_congruent_to_neg_one_mod_four_l968_96884


namespace NUMINAMATH_CALUDE_dodecahedron_path_count_l968_96889

/-- Represents a path on a dodecahedron -/
structure DodecahedronPath where
  start : (Int × Int × Int)
  finish : (Int × Int × Int)
  length : Nat
  visitsAllCorners : Bool
  cannotReturnToStart : Bool

/-- The number of valid paths on a dodecahedron meeting specific conditions -/
def countValidPaths : Nat :=
  sorry

theorem dodecahedron_path_count :
  let validPath : DodecahedronPath :=
    { start := (0, 0, 0),
      finish := (1, 1, 0),
      length := 19,
      visitsAllCorners := true,
      cannotReturnToStart := true }
  countValidPaths = 90 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_path_count_l968_96889


namespace NUMINAMATH_CALUDE_multiple_calculation_l968_96830

theorem multiple_calculation (a b m : ℤ) : 
  b = 8 → 
  b - a = 3 → 
  a * b = m * (a + b) + 14 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_multiple_calculation_l968_96830


namespace NUMINAMATH_CALUDE_pigs_joined_l968_96879

def initial_pigs : ℕ := 64
def final_pigs : ℕ := 86

theorem pigs_joined (initial : ℕ) (final : ℕ) (h1 : initial = initial_pigs) (h2 : final = final_pigs) :
  final - initial = 22 :=
by sorry

end NUMINAMATH_CALUDE_pigs_joined_l968_96879


namespace NUMINAMATH_CALUDE_train_length_l968_96871

/-- The length of a train given its passing times over different distances -/
theorem train_length (tree_time platform_time platform_length : ℝ) 
  (h1 : tree_time = 120)
  (h2 : platform_time = 170)
  (h3 : platform_length = 500)
  (h4 : tree_time > 0)
  (h5 : platform_time > 0)
  (h6 : platform_length > 0) :
  let train_length := (platform_time * platform_length) / (platform_time - tree_time)
  train_length = 1200 := by
sorry


end NUMINAMATH_CALUDE_train_length_l968_96871


namespace NUMINAMATH_CALUDE_compute_expression_l968_96835

theorem compute_expression : 4 * 4^3 - 16^60 / 16^57 = -3840 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l968_96835


namespace NUMINAMATH_CALUDE_total_tomatoes_l968_96898

def tomato_problem (plant1 plant2 plant3 : ℕ) : Prop :=
  plant1 = 24 ∧
  plant2 = (plant1 / 2) + 5 ∧
  plant3 = plant2 + 2 ∧
  plant1 + plant2 + plant3 = 60

theorem total_tomatoes : ∃ (plant1 plant2 plant3 : ℕ), tomato_problem plant1 plant2 plant3 :=
  sorry

end NUMINAMATH_CALUDE_total_tomatoes_l968_96898


namespace NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l968_96828

theorem sqrt_50_plus_sqrt_32 : Real.sqrt 50 + Real.sqrt 32 = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l968_96828


namespace NUMINAMATH_CALUDE_square_perimeter_from_p_shape_l968_96833

/-- Given a square cut into four equal rectangles arranged to form a 'P' shape with a perimeter of 56,
    the perimeter of the original square is 74 2/3. -/
theorem square_perimeter_from_p_shape (x : ℚ) : 
  (2 * (4 * x) + 4 * x = 56) →  -- Perimeter of 'P' shape
  (4 * (4 * x) = 74 + 2/3) -- Perimeter of original square
  := by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_p_shape_l968_96833


namespace NUMINAMATH_CALUDE_louise_pencils_l968_96861

theorem louise_pencils (box_capacity : ℕ) (red_pencils : ℕ) (yellow_pencils : ℕ) (total_boxes : ℕ)
  (h1 : box_capacity = 20)
  (h2 : red_pencils = 20)
  (h3 : yellow_pencils = 40)
  (h4 : total_boxes = 8) :
  let blue_pencils := 2 * red_pencils
  let total_capacity := box_capacity * total_boxes
  let other_pencils := red_pencils + blue_pencils + yellow_pencils
  let green_pencils := total_capacity - other_pencils
  green_pencils = 60 ∧ green_pencils = red_pencils + blue_pencils :=
by sorry


end NUMINAMATH_CALUDE_louise_pencils_l968_96861


namespace NUMINAMATH_CALUDE_saree_price_calculation_l968_96857

/-- Calculate the final price after applying multiple discounts and a tax increase --/
def finalPrice (initialPrice : ℝ) (discounts : List ℝ) (taxRate : ℝ) (finalDiscount : ℝ) : ℝ :=
  let priceAfterDiscounts := discounts.foldl (fun price discount => price * (1 - discount)) initialPrice
  let priceAfterTax := priceAfterDiscounts * (1 + taxRate)
  priceAfterTax * (1 - finalDiscount)

/-- The theorem stating the final price of the sarees --/
theorem saree_price_calculation :
  let initialPrice : ℝ := 495
  let discounts : List ℝ := [0.20, 0.15, 0.10]
  let taxRate : ℝ := 0.05
  let finalDiscount : ℝ := 0.03
  abs (finalPrice initialPrice discounts taxRate finalDiscount - 308.54) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_saree_price_calculation_l968_96857


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_l968_96890

theorem geometric_arithmetic_progression (a b c : ℝ) (q : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positivity for decreasing sequence
  a > b ∧ b > c →  -- Decreasing sequence
  b = a * q ∧ c = a * q^2 →  -- Geometric progression
  2 * (2020 * b / 7) = 577 * a + c / 7 →  -- Arithmetic progression
  q = 1/2 := by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_l968_96890


namespace NUMINAMATH_CALUDE_ordering_abc_l968_96807

theorem ordering_abc : 
  let a : ℝ := 1/11
  let b : ℝ := Real.sqrt (1/10)
  let c : ℝ := Real.log (11/10)
  b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l968_96807


namespace NUMINAMATH_CALUDE_min_covering_ge_max_disjoint_l968_96894

/-- A polygon in a 2D plane -/
structure Polygon where
  -- Add necessary fields for a polygon

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle covers a polygon -/
def covers (c : Circle) (p : Polygon) : Prop :=
  sorry

/-- Predicate to check if a circle is inside a polygon -/
def inside (c : Circle) (p : Polygon) : Prop :=
  sorry

/-- Predicate to check if two circles are disjoint -/
def disjoint (c1 c2 : Circle) : Prop :=
  sorry

/-- The minimum number of unit circles required to cover a polygon -/
def min_covering_circles (p : Polygon) : ℕ :=
  sorry

/-- The maximum number of disjoint unit circles inside a polygon -/
def max_disjoint_circles (p : Polygon) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of covering circles is greater than or equal to
    the maximum number of disjoint circles inside the polygon -/
theorem min_covering_ge_max_disjoint (p : Polygon) :
  min_covering_circles p ≥ max_disjoint_circles p :=
sorry

end NUMINAMATH_CALUDE_min_covering_ge_max_disjoint_l968_96894


namespace NUMINAMATH_CALUDE_unique_number_property_l968_96820

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def remove_digit (n : ℕ) (pos : Fin 5) : ℕ :=
  let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  let removed := digits.removeNth pos
  removed.foldl (fun acc d => acc * 10 + d) 0

theorem unique_number_property :
  ∃! n : ℕ, is_five_digit n ∧
    ∃ pos : Fin 5, n + remove_digit n pos = 54321 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l968_96820


namespace NUMINAMATH_CALUDE_sum_of_first_seven_primes_with_units_digit_3_l968_96859

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def has_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def first_seven_primes_with_units_digit_3 : List ℕ := [3, 13, 23, 43, 53, 73, 83]

theorem sum_of_first_seven_primes_with_units_digit_3 :
  (∀ n ∈ first_seven_primes_with_units_digit_3, is_prime n ∧ has_units_digit_3 n) →
  (∀ p : ℕ, is_prime p → has_units_digit_3 p → 
    p ∉ first_seven_primes_with_units_digit_3 → 
    p > (List.maximum first_seven_primes_with_units_digit_3).getD 0) →
  List.sum first_seven_primes_with_units_digit_3 = 291 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_seven_primes_with_units_digit_3_l968_96859


namespace NUMINAMATH_CALUDE_sin_660_deg_l968_96846

theorem sin_660_deg : Real.sin (660 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_660_deg_l968_96846


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l968_96819

theorem min_value_sum_squares (x y z : ℝ) (h : x + y + z = 1) :
  ∃ m : ℝ, m = 4/9 ∧ ∀ a b c : ℝ, a + b + c = 1 → a^2 + b^2 + 4*c^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l968_96819


namespace NUMINAMATH_CALUDE_bedroom_area_l968_96852

/-- Proves that the area of each bedroom is 121 square feet given the specified house layout --/
theorem bedroom_area (total_area : ℝ) (num_bedrooms : ℕ) (num_bathrooms : ℕ)
  (bathroom_length bathroom_width : ℝ) (kitchen_area : ℝ) :
  total_area = 1110 →
  num_bedrooms = 4 →
  num_bathrooms = 2 →
  bathroom_length = 6 →
  bathroom_width = 8 →
  kitchen_area = 265 →
  ∃ (bedroom_area : ℝ),
    bedroom_area = 121 ∧
    total_area = num_bedrooms * bedroom_area + 
                 num_bathrooms * bathroom_length * bathroom_width +
                 2 * kitchen_area :=
by
  sorry

end NUMINAMATH_CALUDE_bedroom_area_l968_96852


namespace NUMINAMATH_CALUDE_solve_equation_l968_96876

theorem solve_equation (x y : ℝ) : y = 2 / (5 * x + 3) → y = 2 → x = -2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l968_96876


namespace NUMINAMATH_CALUDE_no_valid_coloring_l968_96863

/-- A coloring function that assigns one of three colors to each natural number -/
def Coloring := ℕ → Fin 3

/-- Predicate checking if a coloring satisfies the required property -/
def ValidColoring (c : Coloring) : Prop :=
  (∃ n : ℕ, c n = 0) ∧
  (∃ n : ℕ, c n = 1) ∧
  (∃ n : ℕ, c n = 2) ∧
  (∀ x y : ℕ, c x ≠ c y → c (x + y) ≠ c x ∧ c (x + y) ≠ c y)

theorem no_valid_coloring : ¬∃ c : Coloring, ValidColoring c := by
  sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l968_96863


namespace NUMINAMATH_CALUDE_equation_solutions_l968_96874

theorem equation_solutions : 
  ∀ x : ℝ, x^4 + (4 - x)^4 = 272 ↔ x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l968_96874


namespace NUMINAMATH_CALUDE_sandy_molly_age_ratio_l968_96883

/-- Proves that the ratio of Sandy's current age to Molly's current age is 4:3 -/
theorem sandy_molly_age_ratio :
  let sandy_future_age : ℕ := 34
  let years_to_future : ℕ := 6
  let molly_current_age : ℕ := 21
  let sandy_current_age : ℕ := sandy_future_age - years_to_future
  (sandy_current_age : ℚ) / (molly_current_age : ℚ) = 4 / 3 := by
  sorry

#check sandy_molly_age_ratio

end NUMINAMATH_CALUDE_sandy_molly_age_ratio_l968_96883


namespace NUMINAMATH_CALUDE_max_radius_of_circle_max_radius_achieved_l968_96850

/-- A circle in a rectangular coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point lies on a circle -/
def pointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem max_radius_of_circle (c : Circle) :
  pointOnCircle c (8, 0) → pointOnCircle c (-8, 0) → c.radius ≤ 8 := by
  sorry

theorem max_radius_achieved (r : ℝ) :
  r ≤ 8 →
  ∃ c : Circle, pointOnCircle c (8, 0) ∧ pointOnCircle c (-8, 0) ∧ c.radius = r := by
  sorry

end NUMINAMATH_CALUDE_max_radius_of_circle_max_radius_achieved_l968_96850


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l968_96808

theorem sqrt_equation_solutions : 
  {x : ℝ | Real.sqrt (4 * x - 3) + 18 / Real.sqrt (4 * x - 3) = 9} = {3, 9.75} := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l968_96808


namespace NUMINAMATH_CALUDE_martha_cakes_l968_96897

/-- The number of whole cakes Martha needs to buy -/
def cakes_needed (num_children : ℕ) (cakes_per_child : ℕ) (special_children : ℕ) 
  (parts_per_cake : ℕ) : ℕ :=
  let total_small_cakes := num_children * cakes_per_child
  let special_whole_cakes := (special_children * cakes_per_child + parts_per_cake - 1) / parts_per_cake
  let remaining_small_cakes := total_small_cakes - special_whole_cakes * parts_per_cake
  special_whole_cakes + (remaining_small_cakes + parts_per_cake - 1) / parts_per_cake

/-- The theorem stating the number of cakes Martha needs to buy -/
theorem martha_cakes : cakes_needed 5 25 2 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_martha_cakes_l968_96897


namespace NUMINAMATH_CALUDE_pencil_notebook_cost_l968_96829

/-- Given the cost of pencils and notebooks, calculate the cost of a different quantity -/
theorem pencil_notebook_cost 
  (pencil_price notebook_price : ℕ) 
  (h1 : 4 * pencil_price + 3 * notebook_price = 9600)
  (h2 : 2 * pencil_price + 2 * notebook_price = 5400) :
  8 * pencil_price + 7 * notebook_price = 20400 :=
by sorry

end NUMINAMATH_CALUDE_pencil_notebook_cost_l968_96829


namespace NUMINAMATH_CALUDE_transport_cost_proof_l968_96802

def cost_per_kg : ℝ := 18000
def instrument_mass_g : ℝ := 300

theorem transport_cost_proof :
  let instrument_mass_kg : ℝ := instrument_mass_g / 1000
  instrument_mass_kg * cost_per_kg = 5400 := by
  sorry

end NUMINAMATH_CALUDE_transport_cost_proof_l968_96802
