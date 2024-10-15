import Mathlib

namespace NUMINAMATH_CALUDE_locus_of_P_perpendicular_line_through_focus_l2329_232935

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the point M on the ellipse
def point_M (x y : ℝ) : Prop := ellipse_C x y

-- Define the point N as the foot of the perpendicular from M to x-axis
def point_N (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the point P
def point_P (x y : ℝ) (mx my : ℝ) : Prop :=
  point_M mx my ∧ (x - mx)^2 + y^2 = 2 * my^2

-- Define the point Q
def point_Q (y : ℝ) : ℝ × ℝ := (-3, y)

-- Theorem 1: The locus of P is a circle
theorem locus_of_P (x y : ℝ) :
  (∃ mx my, point_P x y mx my) → x^2 + y^2 = 2 :=
sorry

-- Theorem 2: Line through P perpendicular to OQ passes through left focus
theorem perpendicular_line_through_focus (x y qy : ℝ) (mx my : ℝ) :
  point_P x y mx my →
  (x * (-3 - x) + y * (qy - y) = 1) →
  (∃ t : ℝ, x + t * (qy - y) = -1 ∧ y - t * (-3 - x) = 0) :=
sorry

end NUMINAMATH_CALUDE_locus_of_P_perpendicular_line_through_focus_l2329_232935


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l2329_232983

theorem like_terms_exponent_sum (m n : ℤ) : 
  (∃ (x y : ℝ), -5 * x^m * y^(m+1) = x^(n-1) * y^3) → m + n = 5 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l2329_232983


namespace NUMINAMATH_CALUDE_jelly_bean_theorem_l2329_232946

def jelly_bean_problem (total_jelly_beans : ℕ) (total_people : ℕ) (last_four_take : ℕ) (remaining : ℕ) : Prop :=
  let last_four_total := 4 * last_four_take
  let taken_by_others := total_jelly_beans - remaining - last_four_total
  let others_take_each := 2 * last_four_take
  let num_others := taken_by_others / others_take_each
  num_others = 6 ∧ 
  num_others + 4 = total_people ∧
  taken_by_others + last_four_total + remaining = total_jelly_beans

theorem jelly_bean_theorem : jelly_bean_problem 8000 10 400 1600 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_theorem_l2329_232946


namespace NUMINAMATH_CALUDE_sum_of_five_variables_l2329_232924

theorem sum_of_five_variables (a b c d e : ℝ) 
  (eq1 : a + b = 16)
  (eq2 : b + c = 9)
  (eq3 : c + d = 3)
  (eq4 : d + e = 5)
  (eq5 : e + a = 7) :
  a + b + c + d + e = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_variables_l2329_232924


namespace NUMINAMATH_CALUDE_lateral_face_base_angle_l2329_232953

/-- A regular quadrilateral pyramid with specific properties -/
structure RegularQuadPyramid where
  /-- The angle between a lateral edge and the base plane -/
  edge_base_angle : ℝ
  /-- The angle at the apex of the pyramid -/
  apex_angle : ℝ
  /-- The condition that edge_base_angle equals apex_angle -/
  edge_base_eq_apex : edge_base_angle = apex_angle

/-- The theorem stating the angle between the lateral face and the base plane -/
theorem lateral_face_base_angle (p : RegularQuadPyramid) :
  Real.arctan (Real.sqrt (1 + Real.sqrt 5)) =
    Real.arctan (Real.sqrt (1 + Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_lateral_face_base_angle_l2329_232953


namespace NUMINAMATH_CALUDE_no_real_solutions_l2329_232917

theorem no_real_solutions : ¬∃ (x y : ℝ), 3*x^2 + y^2 - 9*x - 6*y + 23 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2329_232917


namespace NUMINAMATH_CALUDE_shop_c_tv_sets_l2329_232985

theorem shop_c_tv_sets (a b c d e : ℕ) : 
  a = 20 ∧ b = 30 ∧ d = 80 ∧ e = 50 ∧ 
  (a + b + c + d + e) / 5 = 48 →
  c = 60 := by
sorry

end NUMINAMATH_CALUDE_shop_c_tv_sets_l2329_232985


namespace NUMINAMATH_CALUDE_sqrt_2700_minus_37_cube_l2329_232992

theorem sqrt_2700_minus_37_cube (a b : ℕ+) :
  (Real.sqrt 2700 - 37 : ℝ) = (Real.sqrt a.val - b.val)^3 →
  a.val + b.val = 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2700_minus_37_cube_l2329_232992


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2329_232959

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence. -/
def common_difference (a : ℕ → ℝ) : ℝ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a2 : a 2 = 14) 
  (h_a5 : a 5 = 5) : 
  common_difference a = -3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2329_232959


namespace NUMINAMATH_CALUDE_sin_1320_degrees_l2329_232993

theorem sin_1320_degrees : Real.sin (1320 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_1320_degrees_l2329_232993


namespace NUMINAMATH_CALUDE_total_arrangements_l2329_232923

def news_reports : ℕ := 5
def interviews : ℕ := 4
def total_programs : ℕ := 5
def min_news : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

def permute (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

def arrangements (news interviews total min_news : ℕ) : ℕ :=
  (choose news min_news * choose interviews (total - min_news) * permute total total) +
  (choose news (min_news + 1) * choose interviews (total - min_news - 1) * permute total total) +
  (choose news total * permute total total)

theorem total_arrangements :
  arrangements news_reports interviews total_programs min_news = 9720 := by
  sorry

end NUMINAMATH_CALUDE_total_arrangements_l2329_232923


namespace NUMINAMATH_CALUDE_rice_quantity_calculation_rice_quantity_proof_l2329_232940

/-- Calculates the final quantity of rice that can be bought given initial conditions and price changes -/
theorem rice_quantity_calculation (initial_quantity : ℝ) 
  (first_price_reduction : ℝ) (second_price_reduction : ℝ) 
  (kg_to_pound_ratio : ℝ) (currency_exchange_rate : ℝ) : ℝ :=
  let after_first_reduction := initial_quantity * (1 / (1 - first_price_reduction))
  let after_second_reduction := after_first_reduction * (1 / (1 - second_price_reduction))
  let in_pounds := after_second_reduction * kg_to_pound_ratio
  let after_exchange_rate := in_pounds * (1 + currency_exchange_rate)
  let final_quantity := after_exchange_rate / kg_to_pound_ratio
  final_quantity

/-- The final quantity of rice that can be bought is approximately 29.17 kg -/
theorem rice_quantity_proof :
  ∃ ε > 0, |rice_quantity_calculation 20 0.2 0.1 2.2 0.05 - 29.17| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_rice_quantity_calculation_rice_quantity_proof_l2329_232940


namespace NUMINAMATH_CALUDE_number_problem_l2329_232997

theorem number_problem (x : ℝ) : x^2 + 100 = (x - 20)^2 → x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2329_232997


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l2329_232943

theorem pizza_payment_difference :
  let total_slices : ℕ := 12
  let pepperoni_slices : ℕ := total_slices / 3
  let plain_cost : ℚ := 12
  let pepperoni_cost : ℚ := 3
  let total_cost : ℚ := plain_cost + pepperoni_cost
  let cost_per_slice : ℚ := total_cost / total_slices
  let pepperoni_slice_cost : ℚ := cost_per_slice + pepperoni_cost / pepperoni_slices
  let plain_slice_cost : ℚ := cost_per_slice
  let mark_pepperoni_slices : ℕ := pepperoni_slices
  let mark_plain_slices : ℕ := 2
  let anne_slices : ℕ := total_slices - mark_pepperoni_slices - mark_plain_slices
  let mark_cost : ℚ := mark_pepperoni_slices * pepperoni_slice_cost + mark_plain_slices * plain_slice_cost
  let anne_cost : ℚ := anne_slices * plain_slice_cost
  mark_cost - anne_cost = 3 := by sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l2329_232943


namespace NUMINAMATH_CALUDE_village_population_decrease_rate_l2329_232913

/-- Proves that the rate of decrease in Village X's population is 1,200 people per year -/
theorem village_population_decrease_rate 
  (initial_x : ℕ) 
  (initial_y : ℕ) 
  (growth_rate_y : ℕ) 
  (years : ℕ) 
  (h1 : initial_x = 70000)
  (h2 : initial_y = 42000)
  (h3 : growth_rate_y = 800)
  (h4 : years = 14)
  (h5 : ∃ (decrease_rate : ℕ), initial_x - years * decrease_rate = initial_y + years * growth_rate_y) :
  ∃ (decrease_rate : ℕ), decrease_rate = 1200 := by
sorry

end NUMINAMATH_CALUDE_village_population_decrease_rate_l2329_232913


namespace NUMINAMATH_CALUDE_system_sample_fourth_number_l2329_232932

/-- Represents a system sampling of employees -/
structure SystemSample where
  total : Nat
  sample_size : Nat
  sample : Finset Nat

/-- Checks if a given set of numbers forms an arithmetic sequence -/
def is_arithmetic_sequence (s : Finset Nat) : Prop :=
  ∃ (a d : Nat), ∀ (x : Nat), x ∈ s → ∃ (k : Nat), x = a + k * d

/-- The main theorem about the system sampling -/
theorem system_sample_fourth_number
  (s : SystemSample)
  (h_total : s.total = 52)
  (h_size : s.sample_size = 4)
  (h_contains : {6, 32, 45} ⊆ s.sample)
  (h_arithmetic : is_arithmetic_sequence s.sample) :
  19 ∈ s.sample :=
sorry

end NUMINAMATH_CALUDE_system_sample_fourth_number_l2329_232932


namespace NUMINAMATH_CALUDE_opposite_reciprocal_absolute_value_l2329_232927

theorem opposite_reciprocal_absolute_value (a b c d m : ℝ) : 
  (a = -b) →  -- a and b are opposite numbers
  (c * d = 1) →  -- c and d are reciprocals
  (m = 3 ∨ m = -3) →  -- |m| = 3
  ((a + b) / m - c * d + m = 2 ∨ (a + b) / m - c * d + m = -4) := by
sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_absolute_value_l2329_232927


namespace NUMINAMATH_CALUDE_spelling_contest_problem_l2329_232931

theorem spelling_contest_problem (drew_wrong carla_correct total : ℕ) 
  (h1 : drew_wrong = 6)
  (h2 : carla_correct = 14)
  (h3 : total = 52)
  (h4 : 2 * drew_wrong = carla_correct + (total - (carla_correct + drew_wrong + (total - (2 * drew_wrong + carla_correct))))) :
  total - (2 * drew_wrong + carla_correct) = 20 := by
  sorry

end NUMINAMATH_CALUDE_spelling_contest_problem_l2329_232931


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_sum_l2329_232998

theorem quadratic_root_ratio_sum (x₁ x₂ : ℝ) : 
  x₁^2 + 2*x₁ - 8 = 0 →
  x₂^2 + 2*x₂ - 8 = 0 →
  x₁ ≠ 0 →
  x₂ ≠ 0 →
  x₂/x₁ + x₁/x₂ = -5/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_sum_l2329_232998


namespace NUMINAMATH_CALUDE_binomial_consecutive_ratio_l2329_232996

theorem binomial_consecutive_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 3 / 5 →
  n + k = 8 :=
by sorry

end NUMINAMATH_CALUDE_binomial_consecutive_ratio_l2329_232996


namespace NUMINAMATH_CALUDE_find_k_l2329_232937

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem find_k (k : ℤ) (h_odd : k % 2 = 1) (h_eq : f (f (f k)) = 31) : k = 119 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2329_232937


namespace NUMINAMATH_CALUDE_max_books_borrowed_l2329_232950

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (avg_books : ℚ) (h1 : total_students = 38) (h2 : zero_books = 2) (h3 : one_book = 12) 
  (h4 : two_books = 10) (h5 : avg_books = 2) : ∃ (max_books : ℕ), max_books = 5 ∧ 
  (∀ (student_books : ℕ), student_books ≤ max_books) := by
  sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l2329_232950


namespace NUMINAMATH_CALUDE_lattice_polygon_extension_l2329_232900

/-- A point with integer coordinates -/
def LatticePoint (p : ℝ × ℝ) : Prop :=
  ∃ (x y : ℤ), p = (↑x, ↑y)

/-- A polygon with all vertices being lattice points -/
def LatticePolygon (vertices : List (ℝ × ℝ)) : Prop :=
  ∀ v ∈ vertices, LatticePoint v

/-- A convex polygon -/
def ConvexPolygon (vertices : List (ℝ × ℝ)) : Prop :=
  sorry  -- Definition of convex polygon

/-- Theorem: For any convex lattice polygon, there exists another convex lattice polygon
    that contains it and has exactly one additional vertex -/
theorem lattice_polygon_extension
  (Γ : List (ℝ × ℝ))
  (h_lattice : LatticePolygon Γ)
  (h_convex : ConvexPolygon Γ) :
  ∃ (Γ' : List (ℝ × ℝ)),
    LatticePolygon Γ' ∧
    ConvexPolygon Γ' ∧
    (∀ v ∈ Γ, v ∈ Γ') ∧
    (∃! v, v ∈ Γ' ∧ v ∉ Γ) :=
  sorry

end NUMINAMATH_CALUDE_lattice_polygon_extension_l2329_232900


namespace NUMINAMATH_CALUDE_third_class_duration_l2329_232981

/-- Calculates the duration of the third class in a course --/
theorem third_class_duration 
  (weeks : ℕ) 
  (fixed_class_hours : ℕ) 
  (fixed_classes_per_week : ℕ) 
  (homework_hours : ℕ) 
  (total_hours : ℕ) 
  (h1 : weeks = 24)
  (h2 : fixed_class_hours = 3)
  (h3 : fixed_classes_per_week = 2)
  (h4 : homework_hours = 4)
  (h5 : total_hours = 336) :
  ∃ (third_class_hours : ℕ), 
    (fixed_classes_per_week * fixed_class_hours + third_class_hours + homework_hours) * weeks = total_hours ∧
    third_class_hours = 4 :=
by sorry

end NUMINAMATH_CALUDE_third_class_duration_l2329_232981


namespace NUMINAMATH_CALUDE_min_representatives_per_table_l2329_232960

/-- Represents the number of representatives for each country -/
structure Representatives where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ

/-- The condition that country ratios are satisfied -/
def satisfies_ratios (r : Representatives) : Prop :=
  r.A = 2 * r.B ∧ r.A = 3 * r.C ∧ r.A = 4 * r.D

/-- The condition that each country is outnumbered by others at a table -/
def is_outnumbered (r : Representatives) (total : ℕ) : Prop :=
  r.A < r.B + r.C + r.D ∧
  r.B < r.A + r.C + r.D ∧
  r.C < r.A + r.B + r.D ∧
  r.D < r.A + r.B + r.C

/-- The main theorem stating the minimum number of representatives per table -/
theorem min_representatives_per_table (r : Representatives) 
  (h_ratios : satisfies_ratios r) : 
  (∃ (n : ℕ), n > 0 ∧ is_outnumbered r n ∧ 
    ∀ (m : ℕ), m > 0 ∧ is_outnumbered r m → n ≤ m) → 
  (∃ (n : ℕ), n > 0 ∧ is_outnumbered r n ∧ 
    ∀ (m : ℕ), m > 0 ∧ is_outnumbered r m → n ≤ m) ∧ n = 25 :=
sorry

end NUMINAMATH_CALUDE_min_representatives_per_table_l2329_232960


namespace NUMINAMATH_CALUDE_center_radius_sum_l2329_232951

/-- Definition of the circle D -/
def D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - 14*p.1 + p.2^2 + 10*p.2 = -34}

/-- Center of the circle D -/
def center : ℝ × ℝ := sorry

/-- Radius of the circle D -/
def radius : ℝ := sorry

/-- Theorem stating the sum of center coordinates and radius -/
theorem center_radius_sum :
  center.1 + center.2 + radius = 2 + 2 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_center_radius_sum_l2329_232951


namespace NUMINAMATH_CALUDE_v_2004_equals_1_l2329_232958

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 1
| 4 => 2
| 5 => 4
| _ => 0  -- Default case for completeness

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 3
| (n + 1) => g (v n + 1)

-- Theorem statement
theorem v_2004_equals_1 : v 2004 = 1 := by
  sorry

end NUMINAMATH_CALUDE_v_2004_equals_1_l2329_232958


namespace NUMINAMATH_CALUDE_locus_of_vertex_C_l2329_232995

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define an equilateral triangle
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def IsEquilateral (t : EquilateralTriangle) : Prop :=
  let d_AB := ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)^(1/2)
  let d_BC := ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)^(1/2)
  let d_CA := ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)^(1/2)
  d_AB = d_BC ∧ d_BC = d_CA

-- Define the theorem
theorem locus_of_vertex_C (c : Circle) (t : EquilateralTriangle) :
  IsEquilateral t →
  PointOnCircle c t.A →
  PointOnCircle c t.B →
  ∃ c1 c2 : Circle,
    c1.center = c.center ∧
    c2.center = c.center ∧
    c1.radius = c.radius ∧
    c2.radius = c.radius ∧
    PointOnCircle c1 t.C ∨ PointOnCircle c2 t.C :=
by sorry

end NUMINAMATH_CALUDE_locus_of_vertex_C_l2329_232995


namespace NUMINAMATH_CALUDE_complex_magnitude_l2329_232948

theorem complex_magnitude (z₁ z₂ : ℂ) 
  (h1 : z₁ + z₂ = Complex.I * z₁) 
  (h2 : z₂^2 = 2 * Complex.I) : 
  Complex.abs z₁ = 1 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2329_232948


namespace NUMINAMATH_CALUDE_unique_expression_value_l2329_232989

theorem unique_expression_value (m n : ℤ) : 
  (∃! z : ℤ, m * n + 13 * m + 13 * n - m^2 - n^2 = z ∧ 
   (∀ k l : ℤ, k * l + 13 * k + 13 * l - k^2 - l^2 = z → k = m ∧ l = n)) →
  m * n + 13 * m + 13 * n - m^2 - n^2 = 169 :=
by sorry

end NUMINAMATH_CALUDE_unique_expression_value_l2329_232989


namespace NUMINAMATH_CALUDE_distance_between_blue_lights_l2329_232963

/-- Represents the pattern of lights -/
inductive LightColor
| Blue
| Yellow

/-- Represents the recurring pattern of lights -/
def lightPattern : List LightColor :=
  [LightColor.Blue, LightColor.Blue, LightColor.Blue,
   LightColor.Yellow, LightColor.Yellow, LightColor.Yellow, LightColor.Yellow]

/-- The spacing between lights in inches -/
def lightSpacing : ℕ := 7

/-- The number of inches in a foot -/
def inchesPerFoot : ℕ := 12

/-- Calculates the position of the nth blue light in the sequence -/
def bluePosition (n : ℕ) : ℕ :=
  ((n - 1) / 3) * lightPattern.length + ((n - 1) % 3) + 1

/-- The main theorem to prove -/
theorem distance_between_blue_lights :
  (bluePosition 25 - bluePosition 4) * lightSpacing / inchesPerFoot = 28 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_blue_lights_l2329_232963


namespace NUMINAMATH_CALUDE_no_solution_condition_l2329_232942

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (m * x) / (x - 3) ≠ 3 / (x - 3)) ↔ (m = 1 ∨ m = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l2329_232942


namespace NUMINAMATH_CALUDE_income_ratio_l2329_232941

/-- Represents a person's financial information -/
structure Person where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- The problem setup -/
def financialProblem (p1 p2 : Person) : Prop :=
  p1.income = 3500 ∧
  p1.savings = 1400 ∧
  p2.savings = 1400 ∧
  p1.expenditure * 2 = p2.expenditure * 3 ∧
  p1.income = p1.expenditure + p1.savings ∧
  p2.income = p2.expenditure + p2.savings

/-- The theorem to prove -/
theorem income_ratio (p1 p2 : Person) 
  (h : financialProblem p1 p2) : 
  p1.income * 4 = p2.income * 5 := by
  sorry


end NUMINAMATH_CALUDE_income_ratio_l2329_232941


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l2329_232990

/-- Calculates the simple interest rate given loan amounts and repayment details -/
theorem calculate_interest_rate 
  (initial_loan : ℝ) 
  (additional_loan : ℝ) 
  (total_repayment : ℝ) 
  (initial_period : ℝ) 
  (total_period : ℝ)
  (h1 : initial_loan = 10000)
  (h2 : additional_loan = 12000)
  (h3 : total_repayment = 27160)
  (h4 : initial_period = 2)
  (h5 : total_period = 5) :
  ∃ r : ℝ, r = 6 ∧ 
    initial_loan * (1 + r / 100 * initial_period) + 
    (initial_loan + additional_loan) * (1 + r / 100 * (total_period - initial_period)) = 
    total_repayment :=
by sorry

end NUMINAMATH_CALUDE_calculate_interest_rate_l2329_232990


namespace NUMINAMATH_CALUDE_last_twelve_average_l2329_232936

theorem last_twelve_average (total_count : Nat) (total_average : ℚ) (first_twelve_average : ℚ) (thirteenth_result : ℚ) :
  total_count = 25 →
  total_average = 24 →
  first_twelve_average = 14 →
  thirteenth_result = 228 →
  (total_count * total_average = 12 * first_twelve_average + thirteenth_result + 12 * ((total_count * total_average - 12 * first_twelve_average - thirteenth_result) / 12)) ∧
  ((total_count * total_average - 12 * first_twelve_average - thirteenth_result) / 12 = 17) := by
sorry

end NUMINAMATH_CALUDE_last_twelve_average_l2329_232936


namespace NUMINAMATH_CALUDE_train_length_l2329_232902

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 150 → time = 12 → ∃ length : ℝ, abs (length - 500.04) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2329_232902


namespace NUMINAMATH_CALUDE_correct_city_determination_l2329_232933

/-- Represents the two cities on Mars -/
inductive City
| MarsPolis
| MarsCity

/-- Represents the possible answers to a question -/
inductive Answer
| Yes
| No

/-- A Martian's response to the question "Do you live here?" -/
def martianResponse (city : City) (martianOrigin : City) : Answer :=
  match city, martianOrigin with
  | City.MarsPolis, _ => Answer.Yes
  | City.MarsCity, _ => Answer.No

/-- Determines the city based on the Martian's response -/
def determineCity (response : Answer) : City :=
  match response with
  | Answer.Yes => City.MarsPolis
  | Answer.No => City.MarsCity

/-- Theorem stating that asking "Do you live here?" always determines the correct city -/
theorem correct_city_determination (actualCity : City) (martianOrigin : City) :
  determineCity (martianResponse actualCity martianOrigin) = actualCity :=
by sorry

end NUMINAMATH_CALUDE_correct_city_determination_l2329_232933


namespace NUMINAMATH_CALUDE_gcd_of_sums_of_squares_l2329_232987

theorem gcd_of_sums_of_squares : Nat.gcd (130^2 + 240^2 + 350^2) (131^2 + 241^2 + 351^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_sums_of_squares_l2329_232987


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2329_232956

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = 4 ∧ 
  (x₁^2 - 6*x₁ + 8 = 0) ∧ (x₂^2 - 6*x₂ + 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2329_232956


namespace NUMINAMATH_CALUDE_sequence_sum_l2329_232979

-- Define the sequence type
def Sequence := Fin 10 → ℝ

-- Define the property of consecutive terms summing to 20
def ConsecutiveSum (s : Sequence) : Prop :=
  ∀ i : Fin 8, s i + s (i + 1) + s (i + 2) = 20

-- Define the theorem
theorem sequence_sum (s : Sequence) 
  (h1 : ConsecutiveSum s) 
  (h2 : s 4 = 8) : 
  s 0 + s 9 = 8 := by
  sorry


end NUMINAMATH_CALUDE_sequence_sum_l2329_232979


namespace NUMINAMATH_CALUDE_prime_product_digital_sum_difference_l2329_232962

def digital_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digital_sum (n / 10)

theorem prime_product_digital_sum_difference 
  (p q r : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime q) 
  (hr : Nat.Prime r) 
  (hpqr : p * q * r = 18 * 962) 
  (hdiff : p ≠ q ∧ q ≠ r ∧ p ≠ r) : 
  ∃ (result : ℕ), digital_sum p + digital_sum q + digital_sum r - digital_sum (p * q * r) = result :=
sorry

end NUMINAMATH_CALUDE_prime_product_digital_sum_difference_l2329_232962


namespace NUMINAMATH_CALUDE_counterexample_exists_l2329_232929

theorem counterexample_exists : ∃ n : ℕ+, 
  ¬(Nat.Prime n.val) ∧ Nat.Prime (n.val - 2) ∧ n.val = 33 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2329_232929


namespace NUMINAMATH_CALUDE_book_selection_problem_l2329_232903

/-- The number of ways to choose 2 books from 15 books, excluding 3 pairs that cannot be chosen. -/
theorem book_selection_problem (total_books : Nat) (books_to_choose : Nat) (prohibited_pairs : Nat) : 
  total_books = 15 → books_to_choose = 2 → prohibited_pairs = 3 →
  Nat.choose total_books books_to_choose - prohibited_pairs = 102 := by
sorry

end NUMINAMATH_CALUDE_book_selection_problem_l2329_232903


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2329_232909

theorem sum_of_fractions : (1 : ℚ) / 3 + 2 / 7 + 3 / 8 = 167 / 168 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2329_232909


namespace NUMINAMATH_CALUDE_tangent_points_and_circle_area_l2329_232918

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the point P
def P : ℝ × ℝ := (1, -1)

-- Define the tangent points M and N
def M (x₁ : ℝ) : ℝ × ℝ := (x₁, parabola x₁)
def N (x₂ : ℝ) : ℝ × ℝ := (x₂, parabola x₂)

-- State the theorem
theorem tangent_points_and_circle_area 
  (x₁ x₂ : ℝ) 
  (h_tangent : ∃ (k b : ℝ), ∀ x, k * x + b = parabola x → x = x₁ ∨ x = x₂)
  (h_order : x₁ < x₂) :
  (x₁ = 1 - Real.sqrt 2 ∧ x₂ = 1 + Real.sqrt 2) ∧ 
  (∃ (r : ℝ), r > 0 ∧ 
    (∃ (x y : ℝ), (x - P.1)^2 + (y - P.2)^2 = r^2 ∧
      ∃ (k b : ℝ), k * x + b = y ∧ k * x₁ + b = parabola x₁ ∧ k * x₂ + b = parabola x₂) ∧
    π * r^2 = 16 * π / 5) := by
  sorry

end NUMINAMATH_CALUDE_tangent_points_and_circle_area_l2329_232918


namespace NUMINAMATH_CALUDE_floor_identity_l2329_232922

theorem floor_identity (x : ℝ) : 
  ⌊(3+x)/6⌋ - ⌊(4+x)/6⌋ + ⌊(5+x)/6⌋ = ⌊(1+x)/2⌋ - ⌊(1+x)/3⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_identity_l2329_232922


namespace NUMINAMATH_CALUDE_inverse_variation_sqrt_l2329_232921

/-- Given that y varies inversely as √x, prove that when y = 2 for x = 4, then x = 1/4 when y = 8 -/
theorem inverse_variation_sqrt (k : ℝ) (h1 : k > 0) : 
  (∀ x y, x > 0 → y = k / Real.sqrt x) → 
  (2 = k / Real.sqrt 4) → 
  (8 = k / Real.sqrt (1/4)) := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_sqrt_l2329_232921


namespace NUMINAMATH_CALUDE_point_on_segment_l2329_232975

-- Define the space we're working in (Euclidean plane)
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

-- Define points A, B, C, and M
variable (A B C M : E)

-- Define the condition that for any M, either MA ≤ MB or MA ≤ MC
def condition (A B C : E) : Prop :=
  ∀ M : E, ‖M - A‖ ≤ ‖M - B‖ ∨ ‖M - A‖ ≤ ‖M - C‖

-- Define what it means for A to lie on the segment BC
def lies_on_segment (A B C : E) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A = (1 - t) • B + t • C

-- State the theorem
theorem point_on_segment (A B C : E) :
  condition A B C → lies_on_segment A B C :=
by
  sorry

end NUMINAMATH_CALUDE_point_on_segment_l2329_232975


namespace NUMINAMATH_CALUDE_shopping_tax_free_cost_l2329_232965

/-- Given a shopping trip with a total spend, sales tax paid, and tax rate,
    calculate the cost of tax-free items. -/
theorem shopping_tax_free_cost
  (total_spend : ℚ)
  (sales_tax : ℚ)
  (tax_rate : ℚ)
  (h1 : total_spend = 40)
  (h2 : sales_tax = 3/10)
  (h3 : tax_rate = 6/100)
  : ∃ (tax_free_cost : ℚ), tax_free_cost = 35 :=
by
  sorry


end NUMINAMATH_CALUDE_shopping_tax_free_cost_l2329_232965


namespace NUMINAMATH_CALUDE_square_triangle_perimeter_ratio_l2329_232968

theorem square_triangle_perimeter_ratio (s_square s_triangle : ℝ) 
  (h_positive_square : s_square > 0)
  (h_positive_triangle : s_triangle > 0)
  (h_equal_perimeter : 4 * s_square = 3 * s_triangle) :
  s_triangle / s_square = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_perimeter_ratio_l2329_232968


namespace NUMINAMATH_CALUDE_equation_C_violates_basic_properties_l2329_232966

-- Define the equations
def equation_A (a b c : ℝ) : Prop := (a / c = b / c) → (a = b)
def equation_B (a b : ℝ) : Prop := (-a = -b) → (2 - a = 2 - b)
def equation_C (a b c : ℝ) : Prop := (a * c = b * c) → (a = b)
def equation_D (a b m : ℝ) : Prop := ((m^2 + 1) * a = (m^2 + 1) * b) → (a = b)

-- Theorem statement
theorem equation_C_violates_basic_properties :
  (∃ a b c : ℝ, ¬(equation_C a b c)) ∧
  (∀ a b c : ℝ, c ≠ 0 → equation_A a b c) ∧
  (∀ a b : ℝ, equation_B a b) ∧
  (∀ a b m : ℝ, equation_D a b m) :=
by sorry

end NUMINAMATH_CALUDE_equation_C_violates_basic_properties_l2329_232966


namespace NUMINAMATH_CALUDE_equation_solution_l2329_232901

theorem equation_solution :
  ∃ y : ℝ, (5 : ℝ)^(2*y) * (25 : ℝ)^y = (625 : ℝ)^3 ∧ y = 3 :=
by
  -- Define 25 and 625 in terms of 5
  have h1 : (25 : ℝ) = (5 : ℝ)^2 := by sorry
  have h2 : (625 : ℝ) = (5 : ℝ)^4 := by sorry

  -- Prove the existence of y
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2329_232901


namespace NUMINAMATH_CALUDE_number_greater_than_three_l2329_232974

theorem number_greater_than_three (x : ℝ) : 7 * x - 15 > 2 * x → x > 3 := by
  sorry

end NUMINAMATH_CALUDE_number_greater_than_three_l2329_232974


namespace NUMINAMATH_CALUDE_kennel_total_is_45_l2329_232939

/-- Represents the number of dogs in a kennel with specific characteristics. -/
structure KennelDogs where
  long_fur : ℕ
  brown : ℕ
  neither : ℕ
  long_fur_and_brown : ℕ

/-- Calculates the total number of dogs in the kennel. -/
def total_dogs (k : KennelDogs) : ℕ :=
  k.long_fur + k.brown - k.long_fur_and_brown + k.neither

/-- Theorem stating the total number of dogs in the kennel is 45. -/
theorem kennel_total_is_45 (k : KennelDogs) 
    (h1 : k.long_fur = 29)
    (h2 : k.brown = 17)
    (h3 : k.neither = 8)
    (h4 : k.long_fur_and_brown = 9) :
  total_dogs k = 45 := by
  sorry

end NUMINAMATH_CALUDE_kennel_total_is_45_l2329_232939


namespace NUMINAMATH_CALUDE_polynomial_roots_in_arithmetic_progression_l2329_232978

theorem polynomial_roots_in_arithmetic_progression (j k : ℝ) : 
  (∃ a b c d : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
    (∃ r : ℝ, b - a = r ∧ c - b = r ∧ d - c = r) ∧
    (∀ x : ℝ, x^4 + j*x^2 + k*x + 900 = (x - a)*(x - b)*(x - c)*(x - d))) →
  j = -900 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_in_arithmetic_progression_l2329_232978


namespace NUMINAMATH_CALUDE_big_al_bananas_l2329_232952

theorem big_al_bananas (a : ℕ) (h : a + 2*a + 4*a + 8*a + 16*a = 155) : 16*a = 80 := by
  sorry

end NUMINAMATH_CALUDE_big_al_bananas_l2329_232952


namespace NUMINAMATH_CALUDE_f_max_min_l2329_232971

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

-- State the theorem
theorem f_max_min :
  (∃ (x : ℝ), f x = 5 ∧ ∀ (y : ℝ), f y ≤ 5) ∧
  (∃ (x : ℝ), f x = -27 ∧ ∀ (y : ℝ), f y ≥ -27) := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_l2329_232971


namespace NUMINAMATH_CALUDE_loan_split_l2329_232930

/-- Given a total sum of 2691 Rs. split into two parts, if the interest on the first part
    for 8 years at 3% per annum is equal to the interest on the second part for 3 years
    at 5% per annum, then the second part of the sum is 1656 Rs. -/
theorem loan_split (x : ℚ) : 
  (x ≥ 0) →
  (2691 - x ≥ 0) →
  (x * 3 * 8 / 100 = (2691 - x) * 5 * 3 / 100) →
  (2691 - x = 1656) :=
by sorry

end NUMINAMATH_CALUDE_loan_split_l2329_232930


namespace NUMINAMATH_CALUDE_scientific_notation_of_60000_l2329_232976

theorem scientific_notation_of_60000 : 60000 = 6 * (10 ^ 4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_60000_l2329_232976


namespace NUMINAMATH_CALUDE_sum_product_inequality_l2329_232988

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + a * c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l2329_232988


namespace NUMINAMATH_CALUDE_sara_salad_cost_l2329_232916

/-- The cost of Sara's lunch items -/
structure LunchCost where
  hotdog : ℝ
  total : ℝ

/-- Calculates the cost of the salad given the total lunch cost and hotdog cost -/
def salad_cost (lunch : LunchCost) : ℝ :=
  lunch.total - lunch.hotdog

/-- Theorem stating that Sara's salad cost $5.10 -/
theorem sara_salad_cost :
  let lunch : LunchCost := { hotdog := 5.36, total := 10.46 }
  salad_cost lunch = 5.10 := by
  sorry

end NUMINAMATH_CALUDE_sara_salad_cost_l2329_232916


namespace NUMINAMATH_CALUDE_required_run_rate_is_6_15_l2329_232970

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  targetScore : ℕ
  firstSegmentOvers : ℕ
  firstSegmentRunRate : ℚ
  firstSegmentWicketsLost : ℕ
  maxTotalWicketsLost : ℕ
  personalMilestone : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstSegmentOvers
  let runsScored := game.firstSegmentRunRate * game.firstSegmentOvers
  let runsNeeded := game.targetScore - runsScored
  runsNeeded / remainingOvers

/-- Theorem stating the required run rate for the given game scenario -/
theorem required_run_rate_is_6_15 (game : CricketGame) 
    (h1 : game.totalOvers = 50)
    (h2 : game.targetScore = 282)
    (h3 : game.firstSegmentOvers = 10)
    (h4 : game.firstSegmentRunRate = 3.6)
    (h5 : game.firstSegmentWicketsLost = 2)
    (h6 : game.maxTotalWicketsLost = 5)
    (h7 : game.personalMilestone = 75) :
    requiredRunRate game = 6.15 := by
  sorry


end NUMINAMATH_CALUDE_required_run_rate_is_6_15_l2329_232970


namespace NUMINAMATH_CALUDE_laundry_day_lcm_l2329_232949

theorem laundry_day_lcm : Nat.lcm 6 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_laundry_day_lcm_l2329_232949


namespace NUMINAMATH_CALUDE_group_size_problem_l2329_232991

theorem group_size_problem (x : ℕ) : 
  (5 * x + 45 = 7 * x + 3) → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l2329_232991


namespace NUMINAMATH_CALUDE_class_visual_conditions_most_suitable_l2329_232934

/-- Represents a survey method --/
inductive SurveyMethod
  | EnergyLamps
  | ClassVisualConditions
  | ProvinceInternetUsage
  | CanalFishTypes

/-- Defines what constitutes a comprehensive investigation --/
def isComprehensive (method : SurveyMethod) : Prop :=
  match method with
  | .ClassVisualConditions => true
  | _ => false

/-- Theorem stating that understanding the visual conditions of Class 803 
    is the most suitable method for a comprehensive investigation --/
theorem class_visual_conditions_most_suitable :
  ∀ (method : SurveyMethod), 
    isComprehensive method → method = SurveyMethod.ClassVisualConditions :=
by sorry

end NUMINAMATH_CALUDE_class_visual_conditions_most_suitable_l2329_232934


namespace NUMINAMATH_CALUDE_mistaken_calculation_l2329_232928

theorem mistaken_calculation (x : ℕ) : 
  (x / 16 = 8) → (x % 16 = 4) → (x * 16 + 8 = 2120) :=
by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_l2329_232928


namespace NUMINAMATH_CALUDE_product_equality_l2329_232957

theorem product_equality (h : 213 * 16 = 3408) : 1.6 * 213.0 = 340.8 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l2329_232957


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l2329_232977

theorem logarithm_expression_equality : 
  (Real.log 160 / Real.log 4) / (Real.log 4 / Real.log 80) - 
  (Real.log 40 / Real.log 4) / (Real.log 4 / Real.log 10) = 
  4.25 + (3/2) * (Real.log 5 / Real.log 4) := by sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l2329_232977


namespace NUMINAMATH_CALUDE_right_triangle_area_and_height_l2329_232994

theorem right_triangle_area_and_height :
  let a : ℝ := 9
  let b : ℝ := 40
  let c : ℝ := 41
  -- Condition: it's a right triangle
  a ^ 2 + b ^ 2 = c ^ 2 →
  -- Prove the area
  (1 / 2 : ℝ) * a * b = 180 ∧
  -- Prove the height
  (2 * ((1 / 2 : ℝ) * a * b)) / c = 360 / 41 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_and_height_l2329_232994


namespace NUMINAMATH_CALUDE_average_weight_increase_l2329_232982

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 65 →
  new_weight = 97 →
  (new_weight - old_weight) / initial_count = 4 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2329_232982


namespace NUMINAMATH_CALUDE_base_seven_23456_equals_6068_l2329_232905

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_23456_equals_6068 :
  base_seven_to_ten [6, 5, 4, 3, 2] = 6068 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_23456_equals_6068_l2329_232905


namespace NUMINAMATH_CALUDE_sum_parity_when_sum_of_squares_odd_l2329_232947

theorem sum_parity_when_sum_of_squares_odd (n m : ℤ) (h : Odd (n^2 + m^2)) : Odd (n + m) := by
  sorry

end NUMINAMATH_CALUDE_sum_parity_when_sum_of_squares_odd_l2329_232947


namespace NUMINAMATH_CALUDE_purchase_costs_l2329_232945

def cost (x y : ℕ) : ℕ := x + 2 * y

theorem purchase_costs : 
  (cost 5 5 ≤ 18) ∧ 
  (cost 9 4 ≤ 18) ∧ 
  (cost 9 5 > 18) ∧ 
  (cost 2 6 ≤ 18) ∧ 
  (cost 16 0 ≤ 18) :=
by sorry

end NUMINAMATH_CALUDE_purchase_costs_l2329_232945


namespace NUMINAMATH_CALUDE_hannah_mugs_theorem_l2329_232912

def hannah_mugs (total_mugs : ℕ) (total_colors : ℕ) (yellow_mugs : ℕ) : Prop :=
  ∃ (red_mugs blue_mugs other_mugs : ℕ),
    total_mugs = red_mugs + blue_mugs + yellow_mugs + other_mugs ∧
    blue_mugs = 3 * red_mugs ∧
    red_mugs = yellow_mugs / 2 ∧
    other_mugs = 4

theorem hannah_mugs_theorem :
  hannah_mugs 40 4 12 :=
by sorry

end NUMINAMATH_CALUDE_hannah_mugs_theorem_l2329_232912


namespace NUMINAMATH_CALUDE_min_guesses_theorem_l2329_232954

/-- The minimum number of guesses required to determine the leader's binary string -/
def minGuesses (n k : ℕ+) : ℕ :=
  if n = 2 * k then 2 else 1

/-- Theorem stating the minimum number of guesses required -/
theorem min_guesses_theorem (n k : ℕ+) (h : n > k) :
  minGuesses n k = 2 ↔ n = 2 * k :=
sorry

end NUMINAMATH_CALUDE_min_guesses_theorem_l2329_232954


namespace NUMINAMATH_CALUDE_log_properties_l2329_232986

-- Define the logarithm function
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_properties (b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) :
  (f b b = 1) ∧
  (f b 1 = 0) ∧
  (∀ x, 0 < x → x < b → f b x < 1) ∧
  (∀ x, x > b → f b x > 1) :=
by sorry

end NUMINAMATH_CALUDE_log_properties_l2329_232986


namespace NUMINAMATH_CALUDE_trapezoid_parallel_line_length_l2329_232919

/-- Represents a trapezoid with bases of lengths a and b -/
structure Trapezoid (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- 
Given a trapezoid with bases of lengths a and b, 
if a line parallel to the bases divides the trapezoid into two equal-area trapezoids,
then the length of the segment of this line between the non-parallel sides 
is sqrt((a^2 + b^2)/2).
-/
theorem trapezoid_parallel_line_length 
  (a b : ℝ) (trap : Trapezoid a b) : 
  ∃ (x : ℝ), x > 0 ∧ x = Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_parallel_line_length_l2329_232919


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2329_232926

theorem water_tank_capacity (c : ℚ) : 
  (1 / 5 : ℚ) * c + 5 = (2 / 7 : ℚ) * c → c = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l2329_232926


namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l2329_232920

/-- The x-coordinate of a point on the parabola y^2 = 6x that is twice as far from the focus as from the y-axis -/
theorem parabola_point_x_coordinate 
  (x y : ℝ) 
  (h1 : y^2 = 6*x) -- Point is on the parabola y^2 = 6x
  (h2 : (x - 3/2)^2 + y^2 = 4 * x^2) -- Distance to focus is twice distance to y-axis
  : x = 3/2 := by sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l2329_232920


namespace NUMINAMATH_CALUDE_min_distance_midpoint_to_origin_l2329_232973

/-- Given two parallel lines in a 2D plane, this theorem states that 
    the minimum distance from the midpoint of any line segment 
    connecting points on these lines to the origin is 3√2. -/
theorem min_distance_midpoint_to_origin 
  (l₁ l₂ : Set (ℝ × ℝ)) 
  (h₁ : l₁ = {(x, y) | x + y = 7})
  (h₂ : l₂ = {(x, y) | x + y = 5})
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A ∈ l₁) (hB : B ∈ l₂) :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 ∧ 
    ∀ (A' : ℝ × ℝ) (B' : ℝ × ℝ), A' ∈ l₁ → B' ∈ l₂ → 
      let M' := ((A'.1 + B'.1) / 2, (A'.2 + B'.2) / 2)
      d ≤ Real.sqrt (M'.1^2 + M'.2^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_midpoint_to_origin_l2329_232973


namespace NUMINAMATH_CALUDE_condition_property_l2329_232984

theorem condition_property :
  (∀ x y : ℝ, x + y ≠ 5 → (x ≠ 1 ∨ y ≠ 4)) ∧
  (∃ x y : ℝ, (x ≠ 1 ∨ y ≠ 4) ∧ x + y = 5) := by
  sorry

end NUMINAMATH_CALUDE_condition_property_l2329_232984


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2329_232906

/-- 
For a quadratic equation (a-1)x^2 - 2x + 1 = 0 to have real roots, 
a must satisfy: a ≤ 2 and a ≠ 1 
-/
theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 2 ∧ a ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2329_232906


namespace NUMINAMATH_CALUDE_days_in_year_l2329_232915

theorem days_in_year (a b c : ℕ+) (h : 29 * a + 30 * b + 31 * c = 366) : 
  19 * a + 20 * b + 21 * c = 246 := by
  sorry

end NUMINAMATH_CALUDE_days_in_year_l2329_232915


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l2329_232911

/-- Given two circles with centers at (0, 0) and (17, 0) and radii 3 and 8 respectively,
    the x-coordinate of the point where a line tangent to both circles intersects the x-axis
    (to the right of the origin) is equal to 51/11. -/
theorem tangent_line_intersection (x : ℝ) : x > 0 →
  (x^2 = 3^2 + x^2) ∧ ((17 - x)^2 = 8^2 + x^2) → x = 51 / 11 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l2329_232911


namespace NUMINAMATH_CALUDE_girls_in_class_l2329_232904

theorem girls_in_class (total_students : ℕ) (girls : ℕ) (boys : ℕ) :
  total_students = 250 →
  girls + boys = total_students →
  girls = 2 * (total_students - (girls + boys - girls)) →
  girls = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_in_class_l2329_232904


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l2329_232999

open Set Real

-- Define set A
def A : Set ℝ := {x | |x - 2| ≤ 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem complement_intersection_A_B :
  (Aᶜ ∪ Bᶜ) = {x : ℝ | x ≠ 0} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l2329_232999


namespace NUMINAMATH_CALUDE_solve_equation_l2329_232914

theorem solve_equation (x : ℝ) : x^6 = 3^12 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2329_232914


namespace NUMINAMATH_CALUDE_distinct_integers_sum_l2329_232964

theorem distinct_integers_sum (b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℕ) : 
  b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₂ ≠ b₅ ∧ b₂ ≠ b₆ ∧ b₂ ≠ b₇ ∧ b₂ ≠ b₈ ∧ b₂ ≠ b₉ ∧
  b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₃ ≠ b₆ ∧ b₃ ≠ b₇ ∧ b₃ ≠ b₈ ∧ b₃ ≠ b₉ ∧
  b₄ ≠ b₅ ∧ b₄ ≠ b₆ ∧ b₄ ≠ b₇ ∧ b₄ ≠ b₈ ∧ b₄ ≠ b₉ ∧
  b₅ ≠ b₆ ∧ b₅ ≠ b₇ ∧ b₅ ≠ b₈ ∧ b₅ ≠ b₉ ∧
  b₆ ≠ b₇ ∧ b₆ ≠ b₈ ∧ b₆ ≠ b₉ ∧
  b₇ ≠ b₈ ∧ b₇ ≠ b₉ ∧
  b₈ ≠ b₉ →
  (7 : ℚ) / 11 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 + b₈ / 40320 + b₉ / 362880 →
  0 ≤ b₂ ∧ b₂ < 2 →
  0 ≤ b₃ ∧ b₃ < 3 →
  0 ≤ b₄ ∧ b₄ < 4 →
  0 ≤ b₅ ∧ b₅ < 5 →
  0 ≤ b₆ ∧ b₆ < 6 →
  0 ≤ b₇ ∧ b₇ < 7 →
  0 ≤ b₈ ∧ b₈ < 8 →
  0 ≤ b₉ ∧ b₉ < 9 →
  b₂ + b₃ + b₄ + b₅ + b₆ + b₇ + b₈ + b₉ = 16 := by
sorry

end NUMINAMATH_CALUDE_distinct_integers_sum_l2329_232964


namespace NUMINAMATH_CALUDE_complex_subtraction_l2329_232980

theorem complex_subtraction (z₁ z₂ : ℂ) (h₁ : z₁ = 2 + 3*I) (h₂ : z₂ = 3 + I) : 
  z₁ - z₂ = -1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l2329_232980


namespace NUMINAMATH_CALUDE_volleyball_team_starters_l2329_232967

theorem volleyball_team_starters (n : ℕ) (q : ℕ) (s : ℕ) (h1 : n = 16) (h2 : q = 4) (h3 : s = 6) :
  (Nat.choose (n - q) s) + q * (Nat.choose (n - q) (s - 1)) = 4092 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_starters_l2329_232967


namespace NUMINAMATH_CALUDE_delacroix_band_size_l2329_232955

theorem delacroix_band_size (n : ℕ) : 
  (∃ k : ℕ, 30 * n = 28 * k + 6) →
  30 * n < 1200 →
  (∀ m : ℕ, (∃ j : ℕ, 30 * m = 28 * j + 6) → 30 * m < 1200 → 30 * m ≤ 30 * n) →
  30 * n = 930 :=
by sorry

end NUMINAMATH_CALUDE_delacroix_band_size_l2329_232955


namespace NUMINAMATH_CALUDE_boxwood_count_proof_l2329_232944

/-- The cost to trim up each boxwood -/
def trim_cost : ℚ := 5

/-- The cost to trim a boxwood into a fancy shape -/
def fancy_trim_cost : ℚ := 15

/-- The number of boxwoods to be shaped into spheres -/
def fancy_trim_count : ℕ := 4

/-- The total charge for the service -/
def total_charge : ℚ := 210

/-- The number of boxwood hedges the customer wants trimmed up -/
def boxwood_count : ℕ := 30

theorem boxwood_count_proof :
  trim_cost * boxwood_count + fancy_trim_cost * fancy_trim_count = total_charge :=
by sorry

end NUMINAMATH_CALUDE_boxwood_count_proof_l2329_232944


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2329_232925

theorem polynomial_divisibility : ∀ (x : ℂ),
  (x^5 + x^4 + x^3 + x^2 + x + 1 = 0) →
  (x^55 + x^44 + x^33 + x^22 + x^11 + 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2329_232925


namespace NUMINAMATH_CALUDE_kyle_lifting_improvement_l2329_232907

theorem kyle_lifting_improvement (current_capacity : ℕ) (ratio : ℕ) : 
  current_capacity = 80 ∧ ratio = 3 → 
  current_capacity - (current_capacity / ratio) = 53 := by
sorry

end NUMINAMATH_CALUDE_kyle_lifting_improvement_l2329_232907


namespace NUMINAMATH_CALUDE_franks_chips_purchase_franks_chips_purchase_correct_l2329_232961

theorem franks_chips_purchase (chocolate_bars : ℕ) (chocolate_price : ℕ) 
  (chip_price : ℕ) (paid : ℕ) (change : ℕ) : ℕ :=
  let total_spent := paid - change
  let chocolate_cost := chocolate_bars * chocolate_price
  let chips_cost := total_spent - chocolate_cost
  chips_cost / chip_price

#check franks_chips_purchase 5 2 3 20 4 = 2

theorem franks_chips_purchase_correct : franks_chips_purchase 5 2 3 20 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_franks_chips_purchase_franks_chips_purchase_correct_l2329_232961


namespace NUMINAMATH_CALUDE_twelve_triangles_fit_l2329_232969

/-- Represents a right triangle with integer leg lengths -/
structure RightTriangle where
  base : ℕ
  height : ℕ

/-- Calculates the area of a right triangle -/
def area (t : RightTriangle) : ℕ := t.base * t.height / 2

/-- Counts the number of small triangles that fit into a large triangle -/
def count_triangles (large : RightTriangle) (small : RightTriangle) : ℕ :=
  area large / area small

/-- Theorem stating that 12 small triangles fit into the large triangle -/
theorem twelve_triangles_fit (large small : RightTriangle) 
  (h1 : large.base = 6) (h2 : large.height = 4)
  (h3 : small.base = 2) (h4 : small.height = 1) :
  count_triangles large small = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_triangles_fit_l2329_232969


namespace NUMINAMATH_CALUDE_quadratic_function_properties_g_zero_for_negative_m_g_max_abs_value_case1_g_max_abs_value_case2_l2329_232938

def f (x : ℝ) := (x + 1)^2 - 4

def g (m : ℝ) (x : ℝ) := m * f x + 1

theorem quadratic_function_properties :
  (∀ x, f x ≥ -4) ∧ f (-2) = -3 ∧ f 0 = -3 := by sorry

theorem g_zero_for_negative_m (m : ℝ) (hm : m < 0) :
  ∃! x, x ≤ 1 ∧ g m x = 0 := by sorry

theorem g_max_abs_value_case1 (m : ℝ) (hm : 0 < m ∧ m ≤ 8/7) :
  ∀ x ∈ [-3, 3/2], |g m x| ≤ 9/4 * m + 1 := by sorry

theorem g_max_abs_value_case2 (m : ℝ) (hm : m > 8/7) :
  ∀ x ∈ [-3, 3/2], |g m x| ≤ 4 * m - 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_g_zero_for_negative_m_g_max_abs_value_case1_g_max_abs_value_case2_l2329_232938


namespace NUMINAMATH_CALUDE_pen_pencil_ratio_proof_l2329_232972

def number_of_pencils : ℕ := 48
def pencil_pen_difference : ℕ := 8

def number_of_pens : ℕ := number_of_pencils - pencil_pen_difference

def pen_pencil_ratio : ℚ × ℚ := (5, 6)

theorem pen_pencil_ratio_proof :
  (number_of_pens : ℚ) / (number_of_pencils : ℚ) = pen_pencil_ratio.1 / pen_pencil_ratio.2 :=
by sorry

end NUMINAMATH_CALUDE_pen_pencil_ratio_proof_l2329_232972


namespace NUMINAMATH_CALUDE_solve_sandwich_cost_l2329_232910

def sandwich_cost_problem (total_cost soda_cost : ℚ) : Prop :=
  let num_sandwiches : ℕ := 2
  let num_sodas : ℕ := 4
  let sandwich_cost : ℚ := (total_cost - num_sodas * soda_cost) / num_sandwiches
  total_cost = 838/100 ∧ soda_cost = 87/100 → sandwich_cost = 245/100

theorem solve_sandwich_cost : 
  sandwich_cost_problem (838/100) (87/100) := by
  sorry

end NUMINAMATH_CALUDE_solve_sandwich_cost_l2329_232910


namespace NUMINAMATH_CALUDE_irrational_sum_two_l2329_232908

theorem irrational_sum_two : ∃ (a b : ℝ), Irrational a ∧ Irrational b ∧ a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_irrational_sum_two_l2329_232908
