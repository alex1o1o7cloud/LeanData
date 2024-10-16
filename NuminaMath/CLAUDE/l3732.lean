import Mathlib

namespace NUMINAMATH_CALUDE_package_contains_100_masks_l3732_373206

/-- The number of masks in a package used by a family -/
def number_of_masks (family_members : ℕ) (days_per_mask : ℕ) (total_days : ℕ) : ℕ :=
  family_members * (total_days / days_per_mask)

/-- Theorem: The package contains 100 masks -/
theorem package_contains_100_masks :
  number_of_masks 5 4 80 = 100 := by
  sorry

end NUMINAMATH_CALUDE_package_contains_100_masks_l3732_373206


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3732_373203

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 1

theorem tangent_line_equation (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f) x₀ = 2 →
  ∃ y₀ : ℝ, y₀ = f x₀ ∧ 2 * x - y - Real.exp + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3732_373203


namespace NUMINAMATH_CALUDE_mrs_hilt_pecan_pies_l3732_373299

/-- The number of pecan pies Mrs. Hilt baked -/
def pecan_pies : ℕ := sorry

/-- The number of apple pies Mrs. Hilt baked -/
def apple_pies : ℕ := 14

/-- The number of rows in the pie arrangement -/
def rows : ℕ := 30

/-- The number of pies in each row -/
def pies_per_row : ℕ := 5

/-- The total number of pies -/
def total_pies : ℕ := rows * pies_per_row

theorem mrs_hilt_pecan_pies :
  pecan_pies = 136 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pecan_pies_l3732_373299


namespace NUMINAMATH_CALUDE_opposite_reciprocal_calc_l3732_373260

theorem opposite_reciprocal_calc 
  (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 3) 
  (h4 : m < 0) : 
  m^3 + c*d + (a+b)/m = -26 := by
sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_calc_l3732_373260


namespace NUMINAMATH_CALUDE_balloons_given_correct_fred_balloons_l3732_373261

/-- The number of balloons Fred gave to Sandy -/
def balloons_given (initial current : ℕ) : ℕ := initial - current

theorem balloons_given_correct (initial current : ℕ) (h : initial ≥ current) :
  balloons_given initial current = initial - current :=
by sorry

/-- Fred's scenario -/
theorem fred_balloons :
  let initial : ℕ := 709
  let current : ℕ := 488
  balloons_given initial current = 221 :=
by sorry

end NUMINAMATH_CALUDE_balloons_given_correct_fred_balloons_l3732_373261


namespace NUMINAMATH_CALUDE_average_age_first_group_l3732_373290

theorem average_age_first_group (total_students : Nat) (avg_age_all : ℝ) 
  (first_group_size second_group_size : Nat) (avg_age_second_group : ℝ) 
  (age_last_student : ℝ) :
  total_students = 15 →
  avg_age_all = 15 →
  first_group_size = 7 →
  second_group_size = 7 →
  avg_age_second_group = 16 →
  age_last_student = 15 →
  (total_students * avg_age_all - second_group_size * avg_age_second_group - age_last_student) / first_group_size = 14 := by
sorry

end NUMINAMATH_CALUDE_average_age_first_group_l3732_373290


namespace NUMINAMATH_CALUDE_area_bounded_region_area_is_four_l3732_373209

/-- The area of the region bounded by x = 2, y = 2, x = 0, and y = 0 is 4 -/
theorem area_bounded_region : ℝ :=
  let x_bound : ℝ := 2
  let y_bound : ℝ := 2
  x_bound * y_bound

#check area_bounded_region

theorem area_is_four : area_bounded_region = 4 := by sorry

end NUMINAMATH_CALUDE_area_bounded_region_area_is_four_l3732_373209


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_sqrt_three_squared_l3732_373255

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by sorry

theorem sqrt_three_squared : Real.sqrt (3 ^ 2) = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_sqrt_three_squared_l3732_373255


namespace NUMINAMATH_CALUDE_f_satisfies_properties_l3732_373253

-- Define the function f(x) = x²
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_satisfies_properties :
  -- Property 1: f(x₁x₂) = f(x₁)f(x₂)
  (∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂) ∧
  -- Property 2: f'(x) > 0 for x ∈ (0, +∞)
  (∀ x : ℝ, x > 0 → HasDerivAt f (2 * x) x) ∧
  (∀ x : ℝ, x > 0 → 2 * x > 0) ∧
  -- Property 3: f'(x) is an odd function
  (∀ x : ℝ, HasDerivAt f (2 * (-x)) (-x) ∧ HasDerivAt f (2 * x) x ∧ 2 * (-x) = -(2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_properties_l3732_373253


namespace NUMINAMATH_CALUDE_intersection_problem_l3732_373280

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c * x + d

-- State the theorem
theorem intersection_problem (a b c d : ℝ) :
  (f a b 2 = 4) →  -- The graphs intersect at x = 2
  (g c d 2 = 4) →  -- The graphs intersect at x = 2
  (b + c = 1) →    -- Given condition
  (4 * a + d = 1)  -- What we want to prove
:= by sorry

end NUMINAMATH_CALUDE_intersection_problem_l3732_373280


namespace NUMINAMATH_CALUDE_point_on_line_value_l3732_373216

theorem point_on_line_value (a b : ℝ) : 
  b = 3 * a - 2 → 2 * b - 6 * a + 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_value_l3732_373216


namespace NUMINAMATH_CALUDE_donnas_card_shop_wage_l3732_373229

/-- Calculates Donna's hourly wage at the card shop based on her weekly earnings --/
theorem donnas_card_shop_wage (
  dog_walking_hours : ℕ)
  (dog_walking_rate : ℚ)
  (dog_walking_days : ℕ)
  (card_shop_hours : ℕ)
  (card_shop_days : ℕ)
  (babysitting_hours : ℕ)
  (babysitting_rate : ℚ)
  (total_earnings : ℚ)
  (h1 : dog_walking_hours = 2)
  (h2 : dog_walking_rate = 10)
  (h3 : dog_walking_days = 5)
  (h4 : card_shop_hours = 2)
  (h5 : card_shop_days = 5)
  (h6 : babysitting_hours = 4)
  (h7 : babysitting_rate = 10)
  (h8 : total_earnings = 305) :
  (total_earnings - (dog_walking_hours * dog_walking_rate * dog_walking_days + babysitting_hours * babysitting_rate)) / (card_shop_hours * card_shop_days) = 33/2 := by
  sorry

#eval (33 : ℚ) / 2

end NUMINAMATH_CALUDE_donnas_card_shop_wage_l3732_373229


namespace NUMINAMATH_CALUDE_cone_lateral_surface_is_sector_l3732_373284

/-- Represents the possible shapes of an unfolded lateral surface of a cone -/
inductive UnfoldedShape
  | Triangle
  | Rectangle
  | Square
  | Sector

/-- Represents a cone -/
structure Cone where
  -- Add any necessary properties of a cone here

/-- The lateral surface of a cone when unfolded -/
def lateralSurface (c : Cone) : UnfoldedShape := sorry

/-- Theorem stating that the lateral surface of a cone, when unfolded, is shaped like a sector -/
theorem cone_lateral_surface_is_sector (c : Cone) : lateralSurface c = UnfoldedShape.Sector := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_is_sector_l3732_373284


namespace NUMINAMATH_CALUDE_triangle_similarity_l3732_373228

-- Define the basic structures
structure Point := (x y : ℝ)

structure Triangle :=
  (A B C : Point)

-- Define the perpendicular foot point
def perp_foot (P : Point) (line : Point × Point) : Point := sorry

-- Define the similarity relation between triangles
def similar (T1 T2 : Triangle) : Prop := sorry

-- Define the construction process
def construct_next_triangle (T : Triangle) (P : Point) : Triangle :=
  let B1 := perp_foot P (T.B, T.C)
  let B2 := perp_foot P (T.C, T.A)
  let B3 := perp_foot P (T.A, T.B)
  Triangle.mk B1 B2 B3

-- Theorem statement
theorem triangle_similarity 
  (A : Triangle) 
  (P : Point) 
  (h_interior : sorry) -- Assumption that P is interior to A
  : 
  let B := construct_next_triangle A P
  let C := construct_next_triangle B P
  let D := construct_next_triangle C P
  similar A D := by sorry

end NUMINAMATH_CALUDE_triangle_similarity_l3732_373228


namespace NUMINAMATH_CALUDE_parabola_c_value_l3732_373276

-- Define the parabola equation
def parabola (a b c : ℝ) (x y : ℝ) : Prop := x = a * y^2 + b * y + c

-- Define the vertex of the parabola
def vertex (x y : ℝ) : Prop := x = 5 ∧ y = 3

-- Define a point on the parabola
def point_on_parabola (x y : ℝ) : Prop := x = 3 ∧ y = 5

-- Theorem statement
theorem parabola_c_value :
  ∀ (a b c : ℝ),
  (∀ x y, vertex x y → parabola a b c x y) →
  (∀ x y, point_on_parabola x y → parabola a b c x y) →
  a = -1 →
  c = -4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3732_373276


namespace NUMINAMATH_CALUDE_equation_solutions_l3732_373214

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 49 = 0 ↔ x = 7/2 ∨ x = -7/2) ∧
  (∀ x : ℝ, (x + 1)^3 - 27 = 0 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3732_373214


namespace NUMINAMATH_CALUDE_angle_E_measure_l3732_373275

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ)

-- Define the conditions for the quadrilateral
def is_special_quadrilateral (q : Quadrilateral) : Prop :=
  q.E = 3 * q.F ∧ q.E = 4 * q.G ∧ q.E = 6 * q.H ∧
  q.E + q.F + q.G + q.H = 360

-- Theorem statement
theorem angle_E_measure (q : Quadrilateral) 
  (h : is_special_quadrilateral q) : 
  205 < q.E ∧ q.E < 206 :=
sorry

end NUMINAMATH_CALUDE_angle_E_measure_l3732_373275


namespace NUMINAMATH_CALUDE_square_root_sequence_l3732_373277

theorem square_root_sequence (n : ℕ) : 
  (∀ k ∈ Finset.range 35, Int.floor (Real.sqrt (n^2 + k : ℝ)) = n) ↔ n = 17 := by
sorry

end NUMINAMATH_CALUDE_square_root_sequence_l3732_373277


namespace NUMINAMATH_CALUDE_congruence_problem_l3732_373222

theorem congruence_problem (c d m : ℤ) : 
  c ≡ 25 [ZMOD 53] →
  d ≡ 98 [ZMOD 53] →
  m ∈ Finset.Icc 150 200 →
  (c - d ≡ m [ZMOD 53] ↔ m = 192) := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l3732_373222


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3732_373201

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 7) (h2 : x ≠ -2) :
  (3 * x + 12) / (x^2 - 5*x - 14) = (11/3) / (x - 7) + (-2/3) / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3732_373201


namespace NUMINAMATH_CALUDE_triangle_area_with_angle_bisector_l3732_373205

/-- The area of a triangle given two sides and the angle bisector between them -/
theorem triangle_area_with_angle_bisector (a b l : ℝ) (ha : a > 0) (hb : b > 0) (hl : l > 0) :
  let area := l * (a + b) / (4 * a * b) * Real.sqrt (4 * a^2 * b^2 - l^2 * (a + b)^2)
  ∃ (α : ℝ), α > 0 ∧ α < π/2 ∧
    (l * (a + b) / (2 * a * b) = Real.cos α) ∧
    area = (1/2) * a * b * Real.sin (2 * α) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_with_angle_bisector_l3732_373205


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l3732_373248

theorem girls_to_boys_ratio (total : ℕ) (difference : ℕ) : 
  total = 36 →
  difference = 6 →
  ∃ (girls boys : ℕ),
    girls = boys + difference ∧
    girls + boys = total ∧
    girls * 5 = boys * 7 :=
by sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l3732_373248


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3732_373263

theorem polynomial_coefficient_sum : 
  ∀ (A B C D E : ℝ), 
  (∀ x : ℝ, (2*x + 3)*(4*x^3 - 2*x^2 + x - 7) = A*x^4 + B*x^3 + C*x^2 + D*x + E) →
  A + B + C + D + E = -20 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3732_373263


namespace NUMINAMATH_CALUDE_max_value_theorem_l3732_373234

structure Point where
  x : ℝ
  y : ℝ

def ellipse (p : Point) : Prop :=
  p.x^2 / 4 + 9 * p.y^2 / 4 = 1

def condition (p q : Point) : Prop :=
  p.x * q.x + 9 * p.y * q.y = -2

def expression (p q : Point) : ℝ :=
  |2 * p.x + 3 * p.y - 3| + |2 * q.x + 3 * q.y - 3|

theorem max_value_theorem (p q : Point) 
  (h1 : p ≠ q) 
  (h2 : ellipse p) 
  (h3 : ellipse q) 
  (h4 : condition p q) : 
  ∃ (max : ℝ), max = 6 + 2 * Real.sqrt 5 ∧ 
    ∀ (p' q' : Point), 
      p' ≠ q' → 
      ellipse p' → 
      ellipse q' → 
      condition p' q' → 
      expression p' q' ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3732_373234


namespace NUMINAMATH_CALUDE_gcd_459_357_l3732_373271

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l3732_373271


namespace NUMINAMATH_CALUDE_davids_math_marks_l3732_373252

def english_marks : ℕ := 76
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def average_marks : ℕ := 75
def num_subjects : ℕ := 5

theorem davids_math_marks :
  ∃ (math_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / num_subjects = average_marks ∧
    math_marks = 65 :=
by sorry

end NUMINAMATH_CALUDE_davids_math_marks_l3732_373252


namespace NUMINAMATH_CALUDE_sequence_properties_l3732_373285

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define arithmetic progression
def is_arithmetic_progression (a : Sequence) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric progression
def is_geometric_progression (a : Sequence) :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem sequence_properties (a : Sequence) 
  (h1 : a 4 + a 7 = 2) 
  (h2 : a 5 * a 6 = -8) :
  (is_arithmetic_progression a → a 1 * a 10 = -728) ∧
  (is_geometric_progression a → a 1 + a 10 = -7) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3732_373285


namespace NUMINAMATH_CALUDE_optimal_ships_l3732_373262

/-- The maximum annual shipbuilding capacity -/
def max_capacity : ℕ := 20

/-- The output value function -/
def R (x : ℕ) : ℚ := 3700 * x + 45 * x^2 - 10 * x^3

/-- The cost function -/
def C (x : ℕ) : ℚ := 460 * x + 5000

/-- The profit function -/
def p (x : ℕ) : ℚ := R x - C x

/-- The marginal function of a function f -/
def M (f : ℕ → ℚ) (x : ℕ) : ℚ := f (x + 1) - f x

/-- The theorem stating the optimal number of ships to build -/
theorem optimal_ships : 
  ∃ (x : ℕ), x ≤ max_capacity ∧ x > 0 ∧
  ∀ (y : ℕ), y ≤ max_capacity ∧ y > 0 → p x ≥ p y ∧
  x = 12 :=
sorry

end NUMINAMATH_CALUDE_optimal_ships_l3732_373262


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_l3732_373298

theorem product_of_roots_cubic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 15*x^2 + 75*x - 50
  ∃ a b c : ℝ, (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ a * b * c = 50 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_l3732_373298


namespace NUMINAMATH_CALUDE_congruence_power_l3732_373240

theorem congruence_power (a b m n : ℕ) (h : a ≡ b [MOD m]) : a^n ≡ b^n [MOD m] := by
  sorry

end NUMINAMATH_CALUDE_congruence_power_l3732_373240


namespace NUMINAMATH_CALUDE_longest_chord_of_circle_with_radius_five_l3732_373273

/-- A circle with a given radius. -/
structure Circle where
  radius : ℝ

/-- The longest chord of a circle is its diameter, which is twice the radius. -/
def longestChordLength (c : Circle) : ℝ := 2 * c.radius

theorem longest_chord_of_circle_with_radius_five :
  ∃ (c : Circle), c.radius = 5 ∧ longestChordLength c = 10 := by
  sorry

end NUMINAMATH_CALUDE_longest_chord_of_circle_with_radius_five_l3732_373273


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3732_373239

theorem sum_of_coefficients (a b : ℚ) : 
  (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2 → a + b = 70 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3732_373239


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3732_373292

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 = 3 →
  a 1 + a 6 = 12 →
  a 7 + a 8 + a 9 = 45 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3732_373292


namespace NUMINAMATH_CALUDE_pencil_difference_l3732_373218

theorem pencil_difference (price : ℚ) (liam_count mia_count : ℕ) : 
  price > 0.01 →
  price * liam_count = 2.10 →
  price * mia_count = 2.82 →
  mia_count - liam_count = 12 := by
sorry

end NUMINAMATH_CALUDE_pencil_difference_l3732_373218


namespace NUMINAMATH_CALUDE_f_neg_l3732_373293

-- Define an even function f
def f : ℝ → ℝ := sorry

-- Define the property of an even function
axiom f_even : ∀ x : ℝ, f x = f (-x)

-- Define f for positive x
axiom f_pos : ∀ x : ℝ, x > 0 → f x = x^2 - 2*x

-- Theorem to prove
theorem f_neg : ∀ x : ℝ, x < 0 → f x = x^2 + 2*x := by sorry

end NUMINAMATH_CALUDE_f_neg_l3732_373293


namespace NUMINAMATH_CALUDE_quadratic_function_value_l3732_373208

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_value (a b c : ℝ) (h : a ≠ 0) :
  f a b c (-3) = 7 →
  f a b c (-2) = 0 →
  f a b c 0 = -8 →
  f a b c 1 = -9 →
  f a b c 3 = -5 →
  f a b c 5 = 7 →
  f a b c 2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l3732_373208


namespace NUMINAMATH_CALUDE_hannah_cutting_speed_l3732_373288

/-- The number of strands Hannah can cut per minute -/
def hannah_strands_per_minute : ℕ := 8

/-- The total number of strands of duct tape -/
def total_strands : ℕ := 22

/-- The number of strands Hannah's son can cut per minute -/
def son_strands_per_minute : ℕ := 3

/-- The time it takes to cut all strands (in minutes) -/
def total_time : ℕ := 2

theorem hannah_cutting_speed :
  hannah_strands_per_minute = 8 ∧
  total_strands = 22 ∧
  son_strands_per_minute = 3 ∧
  total_time = 2 ∧
  total_time * (hannah_strands_per_minute + son_strands_per_minute) = total_strands :=
by sorry

end NUMINAMATH_CALUDE_hannah_cutting_speed_l3732_373288


namespace NUMINAMATH_CALUDE_marble_difference_l3732_373236

theorem marble_difference (drew_initial marcus_initial : ℕ) : 
  drew_initial - marcus_initial = 24 ∧ 
  drew_initial - 12 = 25 ∧ 
  marcus_initial + 12 = 25 :=
by
  sorry

#check marble_difference

end NUMINAMATH_CALUDE_marble_difference_l3732_373236


namespace NUMINAMATH_CALUDE_mixed_groups_count_l3732_373296

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ) 
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  (∃ (mixed_groups : ℕ), 
    mixed_groups = 72 ∧
    mixed_groups * 2 + boy_boy_photos + girl_girl_photos = total_groups * group_size) :=
by sorry


end NUMINAMATH_CALUDE_mixed_groups_count_l3732_373296


namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l3732_373227

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2) 
  (h2 : Complex.abs (w^2 + z^2) = 8) : 
  Complex.abs (w^3 + z^3) = 20 := by sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l3732_373227


namespace NUMINAMATH_CALUDE_highest_page_number_l3732_373230

/-- Represents the count of available digits --/
def DigitCount := Fin 10 → ℕ

/-- The set of digits where all digits except 5 are unlimited --/
def unlimitedExceptFive (d : DigitCount) : Prop :=
  ∀ i : Fin 10, i.val ≠ 5 → d i = 0 ∧ d 5 = 18

/-- Counts the occurrences of a digit in a natural number --/
def countDigit (digit : Fin 10) (n : ℕ) : ℕ :=
  sorry

/-- Counts the total occurrences of a digit in all numbers up to n --/
def totalDigitCount (digit : Fin 10) (n : ℕ) : ℕ :=
  sorry

/-- The main theorem --/
theorem highest_page_number (d : DigitCount) (h : unlimitedExceptFive d) :
  ∀ n : ℕ, n > 99 → totalDigitCount 5 n > 18 :=
sorry

end NUMINAMATH_CALUDE_highest_page_number_l3732_373230


namespace NUMINAMATH_CALUDE_race_head_start_l3732_373291

theorem race_head_start (vA vB L H : ℝ) : 
  vA = (15 / 13) * vB →
  (L - H) / vB = L / vA - 0.4 * L / vB →
  H = (8 / 15) * L :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l3732_373291


namespace NUMINAMATH_CALUDE_cos_135_degrees_l3732_373268

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l3732_373268


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l3732_373294

theorem integer_solutions_of_equation : 
  {(a, b) : ℤ × ℤ | a^2 + b = b^2022} = {(0, 0), (0, 1)} := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l3732_373294


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l3732_373217

-- Define the statements p and q
def p (a : ℝ) : Prop := a ≥ 1
def q (a : ℝ) : Prop := a ≤ 2

-- Theorem stating that ¬p is a sufficient but not necessary condition for q
theorem not_p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, ¬(p a) → q a) ∧ ¬(∀ a : ℝ, q a → ¬(p a)) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l3732_373217


namespace NUMINAMATH_CALUDE_price_reduction_sales_increase_l3732_373237

theorem price_reduction_sales_increase (price_reduction : ℝ) 
  (net_revenue_increase : ℝ) (sales_increase : ℝ) : 
  price_reduction = 0.3 → 
  net_revenue_increase = 0.26 → 
  (1 - price_reduction) * (1 + sales_increase) = 1 + net_revenue_increase → 
  sales_increase = 0.8 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_sales_increase_l3732_373237


namespace NUMINAMATH_CALUDE_opposite_face_is_E_l3732_373202

/-- Represents the faces of a cube --/
inductive Face
  | A | B | C | D | E | F

/-- Represents a cube --/
structure Cube where
  faces : List Face
  top : Face
  adjacent : Face → List Face
  opposite : Face → Face

/-- The cube configuration described in the problem --/
def problem_cube : Cube :=
  { faces := [Face.A, Face.B, Face.C, Face.D, Face.E, Face.F]
  , top := Face.F
  , adjacent := fun f => match f with
    | Face.D => [Face.A, Face.B, Face.C]
    | _ => sorry  -- We don't have information about other adjacencies
  , opposite := sorry  -- To be proven
  }

theorem opposite_face_is_E :
  problem_cube.opposite Face.A = Face.E :=
by sorry

end NUMINAMATH_CALUDE_opposite_face_is_E_l3732_373202


namespace NUMINAMATH_CALUDE_min_value_3x_plus_y_l3732_373235

theorem min_value_3x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 2 / (x + 4) + 1 / (y + 3) = 1 / 4) :
  3 * x + y ≥ -8 + 20 * Real.sqrt 2 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧
    2 / (x₀ + 4) + 1 / (y₀ + 3) = 1 / 4 ∧
    3 * x₀ + y₀ = -8 + 20 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_y_l3732_373235


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l3732_373250

theorem min_sum_absolute_values (x : ℝ) :
  ∃ (m : ℝ), m = -2 ∧ (∀ x, |x + 3| + |x + 5| + |x + 6| ≥ m) ∧ (∃ x, |x + 3| + |x + 5| + |x + 6| = m) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l3732_373250


namespace NUMINAMATH_CALUDE_definite_integral_sin_plus_one_l3732_373223

theorem definite_integral_sin_plus_one :
  ∫ x in (-1)..(1), (Real.sin x + 1) = 2 - 2 * Real.cos 1 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_sin_plus_one_l3732_373223


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3732_373233

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I * ((a - Complex.I) * (1 + Complex.I))).re = 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3732_373233


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l3732_373272

/-- The volume of a rectangular box with face areas 36, 18, and 12 square inches -/
theorem rectangular_box_volume (l w h : ℝ) 
  (face1 : l * w = 36)
  (face2 : w * h = 18)
  (face3 : l * h = 12) :
  l * w * h = 36 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l3732_373272


namespace NUMINAMATH_CALUDE_base9_arithmetic_l3732_373200

/-- Represents a number in base 9 --/
def Base9 : Type := ℕ

/-- Addition in base 9 --/
def add_base9 (a b : Base9) : Base9 := sorry

/-- Subtraction in base 9 --/
def sub_base9 (a b : Base9) : Base9 := sorry

/-- Conversion from decimal to base 9 --/
def to_base9 (n : ℕ) : Base9 := sorry

theorem base9_arithmetic :
  sub_base9 (add_base9 (to_base9 374) (to_base9 625)) (to_base9 261) = to_base9 738 := by sorry

end NUMINAMATH_CALUDE_base9_arithmetic_l3732_373200


namespace NUMINAMATH_CALUDE_candy_distribution_l3732_373244

theorem candy_distribution (n : ℕ) : 
  (n > 0) → 
  (120 % n = 1) → 
  (n = 7 ∨ n = 17) :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l3732_373244


namespace NUMINAMATH_CALUDE_sphere_diameter_sum_l3732_373225

theorem sphere_diameter_sum (r : ℝ) (d : ℝ) (a b : ℕ) : 
  r = 6 →
  d = 2 * (3 * (4 / 3 * π * r^3))^(1/3) →
  d = a * (b : ℝ)^(1/3) →
  b > 0 →
  ∀ k : ℕ, k > 1 → k^3 ∣ b → k = 1 →
  a + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_sphere_diameter_sum_l3732_373225


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l3732_373241

-- First expression
theorem factorization_1 (x y : ℝ) : -x^2 + 12*x*y - 36*y^2 = -(x - 6*y)^2 := by
  sorry

-- Second expression
theorem factorization_2 (x : ℝ) : x^4 - 9*x^2 = x^2 * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l3732_373241


namespace NUMINAMATH_CALUDE_crows_left_on_branch_l3732_373278

/-- The number of crows remaining on a tree branch after some birds flew away -/
def remaining_crows (initial_parrots initial_total initial_crows remaining_parrots : ℕ) : ℕ :=
  initial_crows - (initial_parrots - remaining_parrots)

/-- Theorem stating the number of crows remaining on the branch -/
theorem crows_left_on_branch :
  ∀ (initial_parrots initial_total initial_crows remaining_parrots : ℕ),
    initial_parrots = 7 →
    initial_total = 13 →
    initial_crows = initial_total - initial_parrots →
    remaining_parrots = 2 →
    remaining_crows initial_parrots initial_total initial_crows remaining_parrots = 1 := by
  sorry

#eval remaining_crows 7 13 6 2

end NUMINAMATH_CALUDE_crows_left_on_branch_l3732_373278


namespace NUMINAMATH_CALUDE_prob_second_red_given_first_red_is_half_l3732_373232

/-- Represents the probability of drawing a red ball as the second draw, given that the first draw was red, from a box containing red and white balls. -/
def probability_second_red_given_first_red (total_red : ℕ) (total_white : ℕ) : ℚ :=
  if total_red > 0 then
    (total_red - 1 : ℚ) / (total_red + total_white - 1 : ℚ)
  else
    0

/-- Theorem stating that in a box with 4 red balls and 3 white balls, 
    if two balls are drawn without replacement and the first ball is red, 
    the probability that the second ball is also red is 1/2. -/
theorem prob_second_red_given_first_red_is_half :
  probability_second_red_given_first_red 4 3 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_second_red_given_first_red_is_half_l3732_373232


namespace NUMINAMATH_CALUDE_stock_sale_total_amount_l3732_373282

/-- Calculates the total amount including brokerage for a stock sale -/
theorem stock_sale_total_amount 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : cash_realized = 104.25)
  (h2 : brokerage_rate = 1/4) : 
  ∃ (total_amount : ℝ), total_amount = 104.51 ∧ 
  total_amount = cash_realized + (brokerage_rate / 100) * cash_realized := by
  sorry

end NUMINAMATH_CALUDE_stock_sale_total_amount_l3732_373282


namespace NUMINAMATH_CALUDE_original_phone_number_proof_l3732_373267

def is_valid_phone_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def first_upgrade (n : ℕ) : ℕ :=
  let d1 := n / 100000
  let rest := n % 100000
  d1 * 1000000 + 800000 + rest

def second_upgrade (n : ℕ) : ℕ :=
  2000000 + n

theorem original_phone_number_proof :
  ∃! n : ℕ, is_valid_phone_number n ∧
    second_upgrade (first_upgrade n) = 81 * n ∧
    n = 282500 :=
sorry

end NUMINAMATH_CALUDE_original_phone_number_proof_l3732_373267


namespace NUMINAMATH_CALUDE_two_consecutive_sets_sum_100_l3732_373247

/-- A structure representing a set of consecutive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  sum_is_100 : start * length + (length * (length - 1)) / 2 = 100
  at_least_two : length ≥ 2

/-- The theorem stating that there are exactly two sets of consecutive positive integers
    whose sum is 100 and contain at least two integers -/
theorem two_consecutive_sets_sum_100 :
  ∃! (sets : Finset ConsecutiveSet), sets.card = 2 ∧ 
    (∀ s ∈ sets, s.start > 0 ∧ s.length ≥ 2 ∧ 
      s.start * s.length + (s.length * (s.length - 1)) / 2 = 100) :=
sorry

end NUMINAMATH_CALUDE_two_consecutive_sets_sum_100_l3732_373247


namespace NUMINAMATH_CALUDE_unit_price_sum_l3732_373286

theorem unit_price_sum (x y z : ℝ) 
  (eq1 : 3 * x + 7 * y + z = 24)
  (eq2 : 4 * x + 10 * y + z = 33) : 
  x + y + z = 6 := by
sorry

end NUMINAMATH_CALUDE_unit_price_sum_l3732_373286


namespace NUMINAMATH_CALUDE_zoom_download_time_ratio_l3732_373221

/-- Prove that the ratio of Windows download time to Mac download time is 3:1 -/
theorem zoom_download_time_ratio :
  let total_time := 82
  let mac_download_time := 10
  let audio_glitch_time := 2 * 4
  let video_glitch_time := 6
  let glitch_time := audio_glitch_time + video_glitch_time
  let no_glitch_time := 2 * glitch_time
  let windows_download_time := total_time - (mac_download_time + glitch_time + no_glitch_time)
  (windows_download_time : ℚ) / mac_download_time = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_zoom_download_time_ratio_l3732_373221


namespace NUMINAMATH_CALUDE_zeros_after_one_in_10000_to_50_l3732_373210

theorem zeros_after_one_in_10000_to_50 : 
  (∃ n : ℕ, 10000^50 = 10^n ∧ n = 200) :=
by sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_10000_to_50_l3732_373210


namespace NUMINAMATH_CALUDE_pentagonal_prism_with_pyramid_sum_l3732_373257

/-- A shape formed by adding a pyramid to one pentagonal face of a pentagonal prism -/
structure PentagonalPrismWithPyramid where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- The sum of faces, vertices, and edges for a PentagonalPrismWithPyramid -/
def PentagonalPrismWithPyramid.sum (shape : PentagonalPrismWithPyramid) : ℕ :=
  shape.faces + shape.vertices + shape.edges

theorem pentagonal_prism_with_pyramid_sum :
  ∃ (shape : PentagonalPrismWithPyramid), shape.sum = 42 :=
sorry

end NUMINAMATH_CALUDE_pentagonal_prism_with_pyramid_sum_l3732_373257


namespace NUMINAMATH_CALUDE_condition_relationship_l3732_373279

theorem condition_relationship (x : ℝ) :
  (∀ x, -1 < x ∧ x < 3 → x < 3) ∧ 
  ¬(∀ x, x < 3 → -1 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l3732_373279


namespace NUMINAMATH_CALUDE_quadratic_root_distance_translation_l3732_373283

/-- Given a quadratic function f(x) = ax^2 + bx + c with a > 0 and two distinct roots
    a distance p apart, the downward translation needed to make the distance
    between the roots 2p is (3b^2)/(4a) - 3c. -/
theorem quadratic_root_distance_translation
  (a b c p : ℝ)
  (h_a_pos : a > 0)
  (h_distinct_roots : ∃ x y, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0)
  (h_distance : ∃ x y, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ |x - y| = p) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := a * x^2 + b * x + (c - ((3 * b^2) / (4 * a) - 3 * c))
  ∃ x y, x ≠ y ∧ g x = 0 ∧ g y = 0 ∧ |x - y| = 2 * p :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_distance_translation_l3732_373283


namespace NUMINAMATH_CALUDE_g_of_three_equals_five_l3732_373289

theorem g_of_three_equals_five (g : ℝ → ℝ) (h : ∀ x, g (x + 2) = 2 * x + 3) :
  g 3 = 5 := by sorry

end NUMINAMATH_CALUDE_g_of_three_equals_five_l3732_373289


namespace NUMINAMATH_CALUDE_certain_amount_calculation_l3732_373259

theorem certain_amount_calculation (A : ℝ) : 
  (0.65 * 150 = 0.20 * A) → A = 487.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_calculation_l3732_373259


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l3732_373213

-- Define the triangles and their side lengths
structure Triangle :=
  (a b c : ℝ)

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

-- State the theorem
theorem similar_triangles_side_length 
  (PQR STU : Triangle) 
  (h_similar : similar PQR STU) 
  (h_PQ : PQR.a = 7) 
  (h_QR : PQR.b = 10) 
  (h_ST : STU.a = 4.9) : 
  STU.b = 7 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l3732_373213


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l3732_373204

-- Define the grid dimensions
def grid_width : Nat := 4
def grid_height : Nat := 5

-- Define a type for grid positions
structure GridPosition where
  x : Nat
  y : Nat

-- Define the initially shaded squares
def initial_shaded : List GridPosition := [
  { x := 1, y := 4 },
  { x := 4, y := 1 }
]

-- Define a function to check if a position is within the grid
def is_valid_position (pos : GridPosition) : Prop :=
  pos.x ≥ 0 ∧ pos.x < grid_width ∧ pos.y ≥ 0 ∧ pos.y < grid_height

-- Define a function to check if a list of positions creates horizontal and vertical symmetry
def is_symmetric (shaded : List GridPosition) : Prop :=
  ∀ pos : GridPosition, is_valid_position pos →
    (pos ∈ shaded ↔ 
     { x := grid_width - 1 - pos.x, y := pos.y } ∈ shaded ∧
     { x := pos.x, y := grid_height - 1 - pos.y } ∈ shaded)

-- The main theorem
theorem min_additional_squares_for_symmetry :
  ∃ (additional : List GridPosition),
    (∀ pos ∈ additional, pos ∉ initial_shaded) ∧
    is_symmetric (initial_shaded ++ additional) ∧
    additional.length = 6 ∧
    (∀ (other : List GridPosition),
      (∀ pos ∈ other, pos ∉ initial_shaded) →
      is_symmetric (initial_shaded ++ other) →
      other.length ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l3732_373204


namespace NUMINAMATH_CALUDE_function_sum_negative_l3732_373224

theorem function_sum_negative
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (x + 2) = -f (-x + 2))
  (h_increasing : ∀ x y, x > 2 → y > 2 → x < y → f x < f y)
  (x₁ x₂ : ℝ)
  (h_sum : x₁ + x₂ > 4)
  (h_product : (x₁ - 2) * (x₂ - 2) < 0) :
  f x₁ + f x₂ < 0 := by
sorry

end NUMINAMATH_CALUDE_function_sum_negative_l3732_373224


namespace NUMINAMATH_CALUDE_triangle_third_side_l3732_373215

theorem triangle_third_side (a b area c : ℝ) : 
  a = 2 * Real.sqrt 2 →
  b = 3 →
  area = 3 →
  area = (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) →
  (c = Real.sqrt 5 ∨ c = Real.sqrt 29) := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_l3732_373215


namespace NUMINAMATH_CALUDE_problem_solution_l3732_373245

theorem problem_solution (a b c : ℤ) : 
  (abs a = 2) → (b = -7) → (c = 5) → (a^2 + (-b) + (-c) = 6) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3732_373245


namespace NUMINAMATH_CALUDE_max_value_implies_a_l3732_373220

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x * (x - 2)^2

/-- Theorem stating that if f(x) has a maximum value of 16/9 and a ≠ 0, then a = 3/2 -/
theorem max_value_implies_a (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ (M : ℝ), M = 16/9 ∧ ∀ (x : ℝ), f a x ≤ M) : 
  a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l3732_373220


namespace NUMINAMATH_CALUDE_wax_needed_for_SUV_l3732_373251

/-- The amount of wax needed to detail Kellan's SUV -/
def wax_for_SUV : ℕ := by sorry

/-- The amount of wax needed to detail Kellan's car -/
def wax_for_car : ℕ := 3

/-- The amount of wax in the bottle Kellan bought -/
def wax_bought : ℕ := 11

/-- The amount of wax Kellan spilled -/
def wax_spilled : ℕ := 2

/-- The amount of wax left after detailing both vehicles -/
def wax_left : ℕ := 2

theorem wax_needed_for_SUV : 
  wax_for_SUV = 4 := by sorry

end NUMINAMATH_CALUDE_wax_needed_for_SUV_l3732_373251


namespace NUMINAMATH_CALUDE_largest_m_value_l3732_373264

/-- A triangle with integer coordinate vertices -/
structure IntTriangle where
  A : ℤ × ℤ
  B : ℤ × ℤ
  C : ℤ × ℤ

/-- The number of lattice points on each side of the triangle -/
def latticePointsOnSides (T : IntTriangle) : ℕ := sorry

/-- The area of the triangle -/
def triangleArea (T : IntTriangle) : ℚ := sorry

/-- The theorem stating the largest possible value of m -/
theorem largest_m_value (T : IntTriangle) :
  (latticePointsOnSides T > 0) →
  (triangleArea T < 2020) →
  (latticePointsOnSides T ≤ 64) ∧ 
  (∃ T', latticePointsOnSides T' = 64 ∧ triangleArea T' < 2020) :=
sorry

end NUMINAMATH_CALUDE_largest_m_value_l3732_373264


namespace NUMINAMATH_CALUDE_orange_juice_ratio_l3732_373242

-- Define the given quantities
def servings : Nat := 280
def serving_size : Nat := 6  -- in ounces
def concentrate_cans : Nat := 35
def concentrate_can_size : Nat := 12  -- in ounces

-- Define the theorem
theorem orange_juice_ratio :
  let total_juice := servings * serving_size
  let total_concentrate := concentrate_cans * concentrate_can_size
  let water_needed := total_juice - total_concentrate
  let water_cans := water_needed / concentrate_can_size
  (water_cans : Int) / (concentrate_cans : Int) = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_ratio_l3732_373242


namespace NUMINAMATH_CALUDE_set_equality_l3732_373269

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the set we want to prove equal to ℂᵤ(M ∪ N)
def target_set : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem set_equality : target_set = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l3732_373269


namespace NUMINAMATH_CALUDE_nested_average_calculation_l3732_373258

def average (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem nested_average_calculation : 
  let x := average 2 3 1
  let y := average 4 1 0
  average x y 5 = 26 / 9 := by sorry

end NUMINAMATH_CALUDE_nested_average_calculation_l3732_373258


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_specific_sqrt_equality_l3732_373238

theorem sqrt_fraction_equality (n : ℕ) (hn : n > 0) :
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = 1 + 1 / (n * (n+1) : ℝ) := by sorry

theorem specific_sqrt_equality :
  Real.sqrt (101/100 + 1/121) = 1 + 1/110 := by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_specific_sqrt_equality_l3732_373238


namespace NUMINAMATH_CALUDE_galyas_number_puzzle_l3732_373231

theorem galyas_number_puzzle (N : ℕ) : (∀ k : ℝ, ((k * N + N) / N - N = k - 2021)) ↔ N = 2022 := by sorry

end NUMINAMATH_CALUDE_galyas_number_puzzle_l3732_373231


namespace NUMINAMATH_CALUDE_length_PQ_is_two_l3732_373274

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Represents a circle in polar form -/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

/-- Represents a ray in polar form -/
structure PolarRay where
  θ : ℝ

theorem length_PQ_is_two 
  (l : ParametricLine)
  (C : PolarCircle)
  (OM : PolarRay)
  (h1 : l.x = fun t ↦ -1/2 * t)
  (h2 : l.y = fun t ↦ 3 * Real.sqrt 3 + (Real.sqrt 3 / 2) * t)
  (h3 : C.equation = fun ρ θ ↦ ρ = 2 * Real.cos θ)
  (h4 : OM.θ = π / 3)
  (P : PolarPoint)
  (Q : PolarPoint)
  (h5 : C.equation P.ρ P.θ)
  (h6 : P.θ = OM.θ)
  (h7 : Q.θ = OM.θ)
  (h8 : Real.sqrt 3 * Q.ρ * Real.cos Q.θ + Q.ρ * Real.sin Q.θ - 3 * Real.sqrt 3 = 0) :
  abs (P.ρ - Q.ρ) = 2 := by
sorry


end NUMINAMATH_CALUDE_length_PQ_is_two_l3732_373274


namespace NUMINAMATH_CALUDE_third_dimension_of_large_box_l3732_373281

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of small boxes that can fit into a larger box -/
def maxSmallBoxes (largeBox : BoxDimensions) (smallBox : BoxDimensions) : ℕ :=
  (largeBox.length / smallBox.length) * (largeBox.width / smallBox.width) * (largeBox.height / smallBox.height)

theorem third_dimension_of_large_box 
  (largeBox : BoxDimensions) 
  (smallBox : BoxDimensions) 
  (h : ℕ) :
  largeBox.length = 12 ∧ 
  largeBox.width = 14 ∧ 
  largeBox.height = h ∧
  smallBox.length = 3 ∧ 
  smallBox.width = 7 ∧ 
  smallBox.height = 2 ∧
  maxSmallBoxes largeBox smallBox = 64 →
  h = 16 :=
by sorry

end NUMINAMATH_CALUDE_third_dimension_of_large_box_l3732_373281


namespace NUMINAMATH_CALUDE_reading_time_difference_l3732_373249

/-- Prove that the difference in reading time between two people is 360 minutes -/
theorem reading_time_difference 
  (xanthia_speed molly_speed : ℕ) 
  (book_length : ℕ) 
  (h1 : xanthia_speed = 120)
  (h2 : molly_speed = 40)
  (h3 : book_length = 360) :
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = 360 := by
  sorry

#check reading_time_difference

end NUMINAMATH_CALUDE_reading_time_difference_l3732_373249


namespace NUMINAMATH_CALUDE_faye_country_albums_l3732_373297

/-- The number of country albums Faye bought -/
def country_albums : ℕ := sorry

/-- The number of pop albums Faye bought -/
def pop_albums : ℕ := 3

/-- The number of songs per album -/
def songs_per_album : ℕ := 6

/-- The total number of songs Faye bought -/
def total_songs : ℕ := 30

theorem faye_country_albums : 
  country_albums = 2 := by sorry

end NUMINAMATH_CALUDE_faye_country_albums_l3732_373297


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3732_373254

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 40 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_rhombus_perimeter_l3732_373254


namespace NUMINAMATH_CALUDE_area_of_triangle_DBC_l3732_373219

/-- Given points A, B, C, D, and E in a coordinate plane, where D and E are midpoints of AB and BC respectively, prove that the area of triangle DBC is 30 square units. -/
theorem area_of_triangle_DBC (A B C D E : ℝ × ℝ) : 
  A = (0, 10) → 
  B = (0, 0) → 
  C = (12, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) → 
  (1 / 2) * (C.1 - B.1) * D.2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_DBC_l3732_373219


namespace NUMINAMATH_CALUDE_hash_property_l3732_373212

/-- Operation # for non-negative integers -/
def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

/-- Theorem: If a # b = 100, then (a + b) + 5 = 10 for non-negative integers a and b -/
theorem hash_property (a b : ℕ) (h : hash a b = 100) : (a + b) + 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hash_property_l3732_373212


namespace NUMINAMATH_CALUDE_honda_production_l3732_373211

theorem honda_production (day_shift : ℕ) (second_shift : ℕ) : 
  day_shift = 4 * second_shift → 
  day_shift + second_shift = 5500 → 
  day_shift = 4400 := by
sorry

end NUMINAMATH_CALUDE_honda_production_l3732_373211


namespace NUMINAMATH_CALUDE_smallest_whole_number_larger_than_triangle_perimeter_l3732_373295

theorem smallest_whole_number_larger_than_triangle_perimeter : 
  ∀ s : ℝ, 
  s > 0 → 
  7 + s > 17 → 
  17 + s > 7 → 
  7 + 17 > s → 
  48 > 7 + 17 + s ∧ 
  ∀ n : ℕ, n < 48 → ∃ t : ℝ, t > 0 ∧ 7 + t > 17 ∧ 17 + t > 7 ∧ 7 + 17 > t ∧ n ≤ 7 + 17 + t :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_larger_than_triangle_perimeter_l3732_373295


namespace NUMINAMATH_CALUDE_coursework_materials_expense_l3732_373270

def budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

theorem coursework_materials_expense : 
  budget * (1 - (food_percentage + accommodation_percentage + entertainment_percentage)) = 300 := by
  sorry

end NUMINAMATH_CALUDE_coursework_materials_expense_l3732_373270


namespace NUMINAMATH_CALUDE_sum_of_solutions_g_l3732_373266

def f (x : ℝ) : ℝ := -x^2 + 10*x - 20

def g : ℝ → ℝ := (f^[2010])

theorem sum_of_solutions_g (h : ∃ (S : Finset ℝ), S.card = 2^2010 ∧ ∀ x ∈ S, g x = 2) :
  ∃ (S : Finset ℝ), S.card = 2^2010 ∧ (∀ x ∈ S, g x = 2) ∧ (S.sum id = 5 * 2^2010) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_g_l3732_373266


namespace NUMINAMATH_CALUDE_existence_of_m_l3732_373265

/-- n(m) denotes the number of factors of 2 in m! -/
def n (m : ℕ) : ℕ := sorry

theorem existence_of_m : ∃ m : ℕ, m > 1990^1990 ∧ m = 3^1990 + n m := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_l3732_373265


namespace NUMINAMATH_CALUDE_total_chicken_pieces_is_74_l3732_373226

/-- Represents the number of chicken pieces used in each dish -/
def chicken_pieces : (String → Nat) := fun
  | "Chicken Pasta" => 2
  | "Barbecue Chicken" => 4
  | "Fried Chicken Dinner" => 8
  | "Grilled Chicken Salad" => 1
  | _ => 0

/-- Represents the number of orders for each dish -/
def orders : (String → Nat) := fun
  | "Chicken Pasta" => 8
  | "Barbecue Chicken" => 5
  | "Fried Chicken Dinner" => 4
  | "Grilled Chicken Salad" => 6
  | _ => 0

/-- Calculates the total number of chicken pieces needed -/
def total_chicken_pieces : Nat :=
  (chicken_pieces "Chicken Pasta" * orders "Chicken Pasta") +
  (chicken_pieces "Barbecue Chicken" * orders "Barbecue Chicken") +
  (chicken_pieces "Fried Chicken Dinner" * orders "Fried Chicken Dinner") +
  (chicken_pieces "Grilled Chicken Salad" * orders "Grilled Chicken Salad")

theorem total_chicken_pieces_is_74 : total_chicken_pieces = 74 := by
  sorry

end NUMINAMATH_CALUDE_total_chicken_pieces_is_74_l3732_373226


namespace NUMINAMATH_CALUDE_right_triangle_parity_l3732_373207

theorem right_triangle_parity (a b c : ℕ) (h_right : a^2 + b^2 = c^2) :
  (Even a ∧ Even b ∧ Even c) ∨
  ((Odd a ∧ Even b ∧ Odd c) ∨ (Even a ∧ Odd b ∧ Odd c)) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_parity_l3732_373207


namespace NUMINAMATH_CALUDE_droid_coffee_usage_l3732_373246

/-- The number of bags of coffee beans Droid uses in a week -/
def weekly_coffee_usage (morning_usage : ℕ) (days_per_week : ℕ) : ℕ :=
  let afternoon_usage := 3 * morning_usage
  let evening_usage := 2 * morning_usage
  let daily_usage := morning_usage + afternoon_usage + evening_usage
  daily_usage * days_per_week

/-- Theorem stating that Droid uses 126 bags of coffee beans per week -/
theorem droid_coffee_usage :
  weekly_coffee_usage 3 7 = 126 := by
  sorry

end NUMINAMATH_CALUDE_droid_coffee_usage_l3732_373246


namespace NUMINAMATH_CALUDE_expression_evaluation_l3732_373243

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  (x^3 + 1) / x * (y^3 + 1) / y + (x^3 - 1) / y * (y^3 - 1) / x = 2 * x^2 * y^2 + 2 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3732_373243


namespace NUMINAMATH_CALUDE_kim_cousins_count_l3732_373256

theorem kim_cousins_count (gum_per_cousin : ℕ) (total_gum : ℕ) (h1 : gum_per_cousin = 5) (h2 : total_gum = 20) :
  total_gum / gum_per_cousin = 4 := by
sorry

end NUMINAMATH_CALUDE_kim_cousins_count_l3732_373256


namespace NUMINAMATH_CALUDE_train_length_calculation_l3732_373287

theorem train_length_calculation (pole_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  pole_time = 50 →
  platform_length = 500 →
  platform_time = 100 →
  ∃ (train_length : ℝ) (train_speed : ℝ),
    train_length = train_speed * pole_time ∧
    train_length + platform_length = train_speed * platform_time ∧
    train_length = 500 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3732_373287
