import Mathlib

namespace NUMINAMATH_CALUDE_triangle_angle_bounds_l179_17925

noncomputable def largest_angle (a b : ℝ) : ℝ := 
  Real.arccos (a / (2 * b))

noncomputable def smallest_angle_case1 (a b : ℝ) : ℝ := 
  Real.arcsin (a / b)

noncomputable def smallest_angle_case2 (a b : ℝ) : ℝ := 
  Real.arccos (b / (2 * a))

theorem triangle_angle_bounds (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (x y : ℝ),
    (largest_angle a b ≤ x ∧ x < π) ∧
    ((b ≥ a * Real.sqrt 2 → 0 < y ∧ y ≤ smallest_angle_case1 a b) ∧
     (b ≤ a * Real.sqrt 2 → 0 < y ∧ y ≤ smallest_angle_case2 a b)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_bounds_l179_17925


namespace NUMINAMATH_CALUDE_derivative_y_l179_17934

noncomputable def y (x : ℝ) : ℝ := x * Real.sin (2 * x)

theorem derivative_y (x : ℝ) :
  deriv y x = Real.sin (2 * x) + 2 * x * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_y_l179_17934


namespace NUMINAMATH_CALUDE_mayoral_election_votes_l179_17948

theorem mayoral_election_votes (x y z : ℕ) : 
  x = y + y / 2 →
  y = z - 2 * z / 5 →
  x = 22500 →
  z = 25000 :=
by sorry

end NUMINAMATH_CALUDE_mayoral_election_votes_l179_17948


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_l179_17931

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem fibonacci_divisibility (k m n s : ℕ) (h : m > 0) (h1 : n > 0) :
  m ∣ fibonacci k → m^n ∣ fibonacci (k * m^(n-1) * s) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_l179_17931


namespace NUMINAMATH_CALUDE_backyard_fence_problem_l179_17914

theorem backyard_fence_problem (back_length : ℝ) (fence_cost_per_foot : ℝ) 
  (owner_back_fraction : ℝ) (owner_left_fraction : ℝ) (owner_total_cost : ℝ) :
  back_length = 18 →
  fence_cost_per_foot = 3 →
  owner_back_fraction = 1/2 →
  owner_left_fraction = 2/3 →
  owner_total_cost = 72 →
  ∃ side_length : ℝ,
    side_length * fence_cost_per_foot * owner_left_fraction + 
    side_length * fence_cost_per_foot +
    back_length * fence_cost_per_foot * owner_back_fraction = owner_total_cost ∧
    side_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_backyard_fence_problem_l179_17914


namespace NUMINAMATH_CALUDE_equality_of_fractions_l179_17982

theorem equality_of_fractions (x y z k : ℝ) : 
  (5 / (x + y) = k / (x + z)) ∧ (k / (x + z) = 9 / (z - y)) → k = 14 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l179_17982


namespace NUMINAMATH_CALUDE_tims_medical_cost_tims_out_of_pocket_cost_l179_17907

/-- Calculates the out-of-pocket cost for Tim's medical visit --/
theorem tims_medical_cost (mri_cost : ℚ) (doctor_rate : ℚ) (exam_time : ℚ) 
  (visit_fee : ℚ) (insurance_coverage : ℚ) : ℚ :=
  let total_cost := mri_cost + doctor_rate * exam_time / 2 + visit_fee
  let insurance_payment := total_cost * insurance_coverage
  total_cost - insurance_payment

/-- Proves that Tim's out-of-pocket cost is $300 --/
theorem tims_out_of_pocket_cost : 
  tims_medical_cost 1200 300 (1/2) 150 (4/5) = 300 := by
  sorry

end NUMINAMATH_CALUDE_tims_medical_cost_tims_out_of_pocket_cost_l179_17907


namespace NUMINAMATH_CALUDE_point_not_on_line_l179_17908

theorem point_not_on_line (m b : ℝ) (h : m * b > 0) : 
  ¬(∃ y : ℝ, y = 3 * m * 4 + 4 * b ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_point_not_on_line_l179_17908


namespace NUMINAMATH_CALUDE_james_car_transactions_l179_17901

/-- Calculates the total amount James was out of pocket after car transactions --/
theorem james_car_transactions (old_car_value : ℝ) (new_car_price : ℝ) 
  (wife_old_car_value : ℝ) (wife_new_car_price : ℝ) 
  (old_car_sale_percentage : ℝ) (new_car_purchase_percentage : ℝ) 
  (wife_old_car_sale_percentage : ℝ) (wife_new_car_purchase_percentage : ℝ) 
  (sales_tax_rate : ℝ) (processing_fee_rate : ℝ) :
  old_car_value = 20000 →
  new_car_price = 30000 →
  wife_old_car_value = 15000 →
  wife_new_car_price = 25000 →
  old_car_sale_percentage = 0.80 →
  new_car_purchase_percentage = 0.90 →
  wife_old_car_sale_percentage = 0.70 →
  wife_new_car_purchase_percentage = 0.85 →
  sales_tax_rate = 0.07 →
  processing_fee_rate = 0.02 →
  (new_car_price * new_car_purchase_percentage + wife_new_car_price * wife_new_car_purchase_percentage) * 
    (1 + sales_tax_rate) - 
  (old_car_value * old_car_sale_percentage + wife_old_car_value * wife_old_car_sale_percentage) * 
    (1 - processing_fee_rate) = 25657.50 := by
  sorry


end NUMINAMATH_CALUDE_james_car_transactions_l179_17901


namespace NUMINAMATH_CALUDE_original_number_proof_l179_17977

theorem original_number_proof (x : ℝ) : 
  (x * 1.125 - x * 0.75 = 30) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l179_17977


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_properties_l179_17918

/-- An ellipse and hyperbola with shared properties -/
structure EllipseHyperbola where
  /-- The distance between the foci -/
  focal_distance : ℝ
  /-- The difference between the major axis of the ellipse and the real axis of the hyperbola -/
  axis_difference : ℝ
  /-- The ratio of eccentricities (ellipse:hyperbola) -/
  eccentricity_ratio : ℝ × ℝ

/-- The equations of the ellipse and hyperbola -/
def curve_equations (eh : EllipseHyperbola) : (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop) :=
  (λ x y ↦ x^2/49 + y^2/36 = 1, λ x y ↦ x^2/9 - y^2/4 = 1)

/-- The area of the triangle formed by the foci and an intersection point -/
def triangle_area (eh : EllipseHyperbola) : ℝ := 12

/-- Theorem stating the properties of the ellipse and hyperbola -/
theorem ellipse_hyperbola_properties (eh : EllipseHyperbola)
    (h1 : eh.focal_distance = 2 * Real.sqrt 13)
    (h2 : eh.axis_difference = 4)
    (h3 : eh.eccentricity_ratio = (3, 7)) :
    curve_equations eh = (λ x y ↦ x^2/49 + y^2/36 = 1, λ x y ↦ x^2/9 - y^2/4 = 1) ∧
    triangle_area eh = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_properties_l179_17918


namespace NUMINAMATH_CALUDE_m_values_l179_17922

def A : Set ℝ := {1, 3}

def B (m : ℝ) : Set ℝ := {x | m * x - 3 = 0}

theorem m_values (m : ℝ) : A ∪ B m = A → m ∈ ({0, 1, 3} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_m_values_l179_17922


namespace NUMINAMATH_CALUDE_articles_sold_l179_17968

theorem articles_sold (cost_price : ℝ) (h : cost_price > 0) : 
  ∃ (N : ℕ), (20 : ℝ) * cost_price = N * (2 * cost_price) ∧ N = 10 :=
by sorry

end NUMINAMATH_CALUDE_articles_sold_l179_17968


namespace NUMINAMATH_CALUDE_xy_equals_three_l179_17954

theorem xy_equals_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdist : x ≠ y)
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_three_l179_17954


namespace NUMINAMATH_CALUDE_equal_roots_values_l179_17987

theorem equal_roots_values (x m : ℝ) : 
  (x^2 * (x - 2) - (m + 2)) / ((x - 2) * (m - 2)) = x^2 / m → 
  (∀ x, 2*x^2 - 4*x - m^2 - 2*m = 0) → 
  (m = -1 + Real.sqrt 3 ∨ m = -1 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_values_l179_17987


namespace NUMINAMATH_CALUDE_unique_equal_intercept_line_l179_17957

/-- A line with equal intercepts on both axes passing through (2,3) -/
def EqualInterceptLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ a : ℝ, p.1 + p.2 = a ∧ (2 : ℝ) + (3 : ℝ) = a}

/-- The theorem stating that there is exactly one line with equal intercepts passing through (2,3) -/
theorem unique_equal_intercept_line : 
  ∃! a : ℝ, (2 : ℝ) + (3 : ℝ) = a ∧ EqualInterceptLine = {p : ℝ × ℝ | p.1 + p.2 = a} :=
by sorry

end NUMINAMATH_CALUDE_unique_equal_intercept_line_l179_17957


namespace NUMINAMATH_CALUDE_steven_skittles_count_l179_17951

/-- The number of groups of Skittles in Steven's collection -/
def num_groups : ℕ := 77

/-- The number of Skittles in each group -/
def skittles_per_group : ℕ := 77

/-- The total number of Skittles in Steven's collection -/
def total_skittles : ℕ := num_groups * skittles_per_group

theorem steven_skittles_count : total_skittles = 5929 := by
  sorry

end NUMINAMATH_CALUDE_steven_skittles_count_l179_17951


namespace NUMINAMATH_CALUDE_ord₂_3n_minus_1_l179_17927

-- Define ord₂ function
def ord₂ (i : ℤ) : ℕ :=
  if i = 0 then 0 else (i.natAbs.factors.filter (· = 2)).length

-- Main theorem
theorem ord₂_3n_minus_1 (n : ℕ) (h : n > 0) :
  (ord₂ (3^n - 1) = 1 ↔ n % 2 = 1) ∧
  (¬ ∃ n, ord₂ (3^n - 1) = 2) ∧
  (ord₂ (3^n - 1) = 3 ↔ n % 4 = 2) :=
sorry

-- Additional lemma to ensure ord₂(3ⁿ - 1) > 0 for n > 0
lemma ord₂_3n_minus_1_pos (n : ℕ) (h : n > 0) :
  ord₂ (3^n - 1) > 0 :=
sorry

end NUMINAMATH_CALUDE_ord₂_3n_minus_1_l179_17927


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_squared_l179_17923

theorem sum_of_fourth_powers_squared (A B C : ℤ) (h : A + B + C = 0) :
  2 * (A^4 + B^4 + C^4) = (A^2 + B^2 + C^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_squared_l179_17923


namespace NUMINAMATH_CALUDE_japanese_students_fraction_l179_17973

theorem japanese_students_fraction (J : ℚ) (h : J > 0) : 
  let S := 3 * J
  let seniors_studying := (1/3) * S
  let juniors_studying := (3/4) * J
  let total_students := S + J
  (seniors_studying + juniors_studying) / total_students = 7/16 := by
sorry

end NUMINAMATH_CALUDE_japanese_students_fraction_l179_17973


namespace NUMINAMATH_CALUDE_system_solution_l179_17996

theorem system_solution (a₁ a₂ a₃ a₄ : ℝ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄) :
  ∃! (x₁ x₂ x₃ x₄ : ℝ),
    (|a₁ - a₂| * x₂ + |a₁ - a₃| * x₃ + |a₁ - a₄| * x₄ = 1) ∧
    (|a₂ - a₁| * x₁ + |a₂ - a₃| * x₃ + |a₂ - a₄| * x₄ = 1) ∧
    (|a₃ - a₁| * x₁ + |a₃ - a₂| * x₂ + |a₃ - a₄| * x₄ = 1) ∧
    (|a₄ - a₁| * x₁ + |a₄ - a₂| * x₂ + |a₄ - a₃| * x₃ = 1) ∧
    (x₁ = x₂) ∧ (x₂ = x₃) ∧ (x₃ = x₄) ∧
    (x₁ = 1 / (3 * a₁ - (a₂ + a₃ + a₄))) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l179_17996


namespace NUMINAMATH_CALUDE_triangle_count_is_53_l179_17994

/-- Represents a rectangle divided into triangles --/
structure TriangulatedRectangle where
  columns : Nat
  rows : Nat
  has_full_diagonals : Bool
  has_half_diagonals : Bool

/-- Counts the number of triangles in a TriangulatedRectangle --/
def count_triangles (rect : TriangulatedRectangle) : Nat :=
  sorry

/-- The specific rectangle described in the problem --/
def problem_rectangle : TriangulatedRectangle :=
  { columns := 6
  , rows := 3
  , has_full_diagonals := true
  , has_half_diagonals := true }

theorem triangle_count_is_53 : count_triangles problem_rectangle = 53 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_is_53_l179_17994


namespace NUMINAMATH_CALUDE_combine_numbers_to_24_l179_17969

theorem combine_numbers_to_24 : (10 * 10 - 4) / 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_combine_numbers_to_24_l179_17969


namespace NUMINAMATH_CALUDE_certain_number_proof_l179_17911

theorem certain_number_proof (x : ℝ) :
  (1.12 * x) / 4.98 = 528.0642570281125 → x = 2350 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l179_17911


namespace NUMINAMATH_CALUDE_quadrilateral_theorem_l179_17916

-- Define a quadrilateral
structure Quadrilateral :=
(A B C D : ℝ × ℝ)

-- Define the angle between two vectors
def angle (v w : ℝ × ℝ) : ℝ := sorry

-- Define the length of a vector
def length (v : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem quadrilateral_theorem (q : Quadrilateral) 
  (h : angle (q.C.1 - q.A.1, q.C.2 - q.A.2) (q.A.1 - q.C.1, q.A.2 - q.C.2) = 120) :
  (length (q.A.1 - q.C.1, q.A.2 - q.C.2) * length (q.B.1 - q.D.1, q.B.2 - q.D.2))^2 =
  (length (q.A.1 - q.B.1, q.A.2 - q.B.2) * length (q.C.1 - q.D.1, q.C.2 - q.D.2))^2 +
  (length (q.B.1 - q.C.1, q.B.2 - q.C.2) * length (q.A.1 - q.D.1, q.A.2 - q.D.2))^2 +
  length (q.A.1 - q.B.1, q.A.2 - q.B.2) * length (q.B.1 - q.C.1, q.B.2 - q.C.2) *
  length (q.C.1 - q.D.1, q.C.2 - q.D.2) * length (q.D.1 - q.A.1, q.D.2 - q.A.2) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_theorem_l179_17916


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l179_17965

/-- The function f(x) = x^3 + x + 2 -/
def f (x : ℝ) : ℝ := x^3 + x + 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_parallel_points :
  ∀ x y : ℝ, (f x = y ∧ f' x = 4) ↔ (x = -1 ∧ y = 0) ∨ (x = 1 ∧ y = 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l179_17965


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l179_17935

theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 18 →
  length = 3 * breadth →
  area = length * breadth →
  area = 972 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l179_17935


namespace NUMINAMATH_CALUDE_salt_solution_dilution_l179_17952

theorem salt_solution_dilution (initial_volume : ℝ) (initial_salt_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 64 ∧ 
  initial_salt_percentage = 0.1 ∧ 
  added_water = 16 →
  let salt_amount := initial_volume * initial_salt_percentage
  let new_volume := initial_volume + added_water
  let final_salt_percentage := salt_amount / new_volume
  final_salt_percentage = 0.08 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_dilution_l179_17952


namespace NUMINAMATH_CALUDE_trig_identity_l179_17942

theorem trig_identity : 
  Real.sqrt (1 + Real.sin 6) + Real.sqrt (1 - Real.sin 6) = -2 * Real.cos 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l179_17942


namespace NUMINAMATH_CALUDE_men_count_in_line_l179_17989

/-- Represents the number of arrangements where men and women stand alternately -/
def alternating_arrangements (men women : ℕ) : ℕ :=
  if men ≥ women then men + 1 - women else women + 1 - men

/-- The problem statement -/
theorem men_count_in_line (women : ℕ) (arrangements : ℕ) :
  women = 2 →
  arrangements = 4 →
  ∃ (men : ℕ), alternating_arrangements men women = arrangements :=
by
  sorry

#eval alternating_arrangements 4 2  -- Should output 4

end NUMINAMATH_CALUDE_men_count_in_line_l179_17989


namespace NUMINAMATH_CALUDE_log_equation_solution_l179_17980

theorem log_equation_solution (x : ℝ) :
  Real.log x / Real.log 2 = 5/2 → x = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l179_17980


namespace NUMINAMATH_CALUDE_f_2019_equals_2_l179_17900

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2019_equals_2 (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : ∀ x, f x = f (4 - x))
  (h_f_neg3 : f (-3) = 2) :
  f 2019 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2019_equals_2_l179_17900


namespace NUMINAMATH_CALUDE_winter_olympics_merchandise_l179_17986

def total_items : ℕ := 180
def figurine_cost : ℕ := 80
def pendant_cost : ℕ := 50
def total_spent : ℕ := 11400
def figurine_price : ℕ := 100
def pendant_price : ℕ := 60
def min_profit : ℕ := 2900

theorem winter_olympics_merchandise (x y : ℕ) (m : ℕ) : 
  x + y = total_items ∧ 
  figurine_cost * x + pendant_cost * y = total_spent ∧
  (pendant_price - pendant_cost) * m + (figurine_price - figurine_cost) * (total_items - m) ≥ min_profit →
  x = 80 ∧ y = 100 ∧ m ≤ 70 := by
  sorry

end NUMINAMATH_CALUDE_winter_olympics_merchandise_l179_17986


namespace NUMINAMATH_CALUDE_sin_minus_abs_sin_range_l179_17966

theorem sin_minus_abs_sin_range :
  ∀ y : ℝ, (∃ x : ℝ, y = Real.sin x - |Real.sin x|) ↔ -2 ≤ y ∧ y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_abs_sin_range_l179_17966


namespace NUMINAMATH_CALUDE_total_students_present_l179_17936

/-- Calculates the total number of students present across four kindergarten sessions -/
theorem total_students_present
  (morning_registered : ℕ) (morning_absent : ℕ)
  (early_afternoon_registered : ℕ) (early_afternoon_absent : ℕ)
  (late_afternoon_registered : ℕ) (late_afternoon_absent : ℕ)
  (evening_registered : ℕ) (evening_absent : ℕ)
  (h1 : morning_registered = 25) (h2 : morning_absent = 3)
  (h3 : early_afternoon_registered = 24) (h4 : early_afternoon_absent = 4)
  (h5 : late_afternoon_registered = 30) (h6 : late_afternoon_absent = 5)
  (h7 : evening_registered = 35) (h8 : evening_absent = 7) :
  (morning_registered - morning_absent) +
  (early_afternoon_registered - early_afternoon_absent) +
  (late_afternoon_registered - late_afternoon_absent) +
  (evening_registered - evening_absent) = 95 :=
by sorry

end NUMINAMATH_CALUDE_total_students_present_l179_17936


namespace NUMINAMATH_CALUDE_train_speed_conversion_l179_17990

/-- Converts speed from meters per second to kilometers per hour -/
def mps_to_kmph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

/-- The speed of the train in meters per second -/
def train_speed_mps : ℝ := 52.5042

theorem train_speed_conversion :
  mps_to_kmph train_speed_mps = 189.01512 := by sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l179_17990


namespace NUMINAMATH_CALUDE_coffee_tea_overlap_l179_17961

theorem coffee_tea_overlap (total population : ℝ) 
  (coffee_drinkers : ℝ) (tea_drinkers : ℝ) (neither_drinkers : ℝ) :
  coffee_drinkers = 0.6 * total →
  tea_drinkers = 0.5 * total →
  neither_drinkers = 0.1 * total →
  population > 0 →
  ∃ (both_drinkers : ℝ), 
    both_drinkers ≥ 0.2 * total ∧
    both_drinkers ≤ coffee_drinkers ∧
    both_drinkers ≤ tea_drinkers ∧
    coffee_drinkers + tea_drinkers - both_drinkers + neither_drinkers = total :=
by
  sorry

end NUMINAMATH_CALUDE_coffee_tea_overlap_l179_17961


namespace NUMINAMATH_CALUDE_sin_alpha_value_l179_17904

theorem sin_alpha_value (α : Real) :
  (∃ (t : Real), t * (Real.sin (30 * π / 180)) = Real.cos α ∧
                 t * (-Real.cos (30 * π / 180)) = Real.sin α) →
  Real.sin α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l179_17904


namespace NUMINAMATH_CALUDE_sequence_2018th_term_l179_17949

theorem sequence_2018th_term (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : ∀ n, 3 * S n = 2 * a n - 3 * n) : 
  a 2018 = 2^2018 - 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_2018th_term_l179_17949


namespace NUMINAMATH_CALUDE_even_function_derivative_is_odd_l179_17902

theorem even_function_derivative_is_odd
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_diff : Differentiable ℝ f)
  (h_even : ∀ x, f (-x) = f x)
  (h_deriv : ∀ x, HasDerivAt f (g x) x) :
  ∀ x, g (-x) = -g x :=
sorry

end NUMINAMATH_CALUDE_even_function_derivative_is_odd_l179_17902


namespace NUMINAMATH_CALUDE_smallest_angle_in_ratio_triangle_l179_17975

theorem smallest_angle_in_ratio_triangle (α β γ : Real) : 
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- Angles are positive
  β = 2 * α ∧ γ = 3 * α →  -- Angle ratio is 1 : 2 : 3
  α + β + γ = π →         -- Sum of angles in a triangle
  α = π / 6 := by
    sorry

end NUMINAMATH_CALUDE_smallest_angle_in_ratio_triangle_l179_17975


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l179_17976

theorem quadratic_roots_product (c d : ℝ) : 
  (3 * c ^ 2 + 9 * c - 21 = 0) → 
  (3 * d ^ 2 + 9 * d - 21 = 0) → 
  (3 * c - 4) * (6 * d - 8) = -22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l179_17976


namespace NUMINAMATH_CALUDE_min_value_expression_l179_17984

theorem min_value_expression (x y : ℤ) (h : 4*x + 5*y = 7) :
  ∃ (m : ℤ), m = 1 ∧ ∀ (a b : ℤ), 4*a + 5*b = 7 → 5*|a| - 3*|b| ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l179_17984


namespace NUMINAMATH_CALUDE_cos_angle_POQ_l179_17903

/-- Given two points P and Q on the unit circle centered at the origin O,
    where P is in the first quadrant with x-coordinate 4/5,
    and Q is in the fourth quadrant with x-coordinate 5/13,
    prove that the cosine of angle POQ is 56/65. -/
theorem cos_angle_POQ (P Q : ℝ × ℝ) : 
  (P.1^2 + P.2^2 = 1) →  -- P is on the unit circle
  (Q.1^2 + Q.2^2 = 1) →  -- Q is on the unit circle
  (P.1 = 4/5) →          -- x-coordinate of P is 4/5
  (P.2 ≥ 0) →            -- P is in the first quadrant
  (Q.1 = 5/13) →         -- x-coordinate of Q is 5/13
  (Q.2 ≤ 0) →            -- Q is in the fourth quadrant
  Real.cos (Real.arccos P.1 + Real.arccos Q.1) = 56/65 := by
  sorry


end NUMINAMATH_CALUDE_cos_angle_POQ_l179_17903


namespace NUMINAMATH_CALUDE_polynomial_degree_problem_l179_17940

theorem polynomial_degree_problem (m n : ℤ) : 
  (m + 1 + 2 = 6) →  -- Degree of the polynomial term x^(m+1)y^2 is 6
  (2*n + (5 - m) = 6) →  -- Degree of the monomial x^(2n)y^(5-m) is 6
  (-m)^3 + 2*n = -23 := by
sorry

end NUMINAMATH_CALUDE_polynomial_degree_problem_l179_17940


namespace NUMINAMATH_CALUDE_smallest_c_value_l179_17921

theorem smallest_c_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : ∀ x, a * Real.sin (b * x + c) ≤ a * Real.sin (b * (-π/4) + c))
  (h5 : a = 3) :
  c ≥ 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l179_17921


namespace NUMINAMATH_CALUDE_bread_loaves_l179_17997

theorem bread_loaves (slices_per_loaf : ℕ) (num_friends : ℕ) (slices_per_friend : ℕ) :
  slices_per_loaf = 15 →
  num_friends = 10 →
  slices_per_friend = 6 →
  (num_friends * slices_per_friend) / slices_per_loaf = 4 :=
by sorry

end NUMINAMATH_CALUDE_bread_loaves_l179_17997


namespace NUMINAMATH_CALUDE_complex_power_215_36_l179_17919

theorem complex_power_215_36 : (Complex.exp (215 * π / 180 * Complex.I))^36 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_215_36_l179_17919


namespace NUMINAMATH_CALUDE_no_x_squared_term_l179_17985

theorem no_x_squared_term (a : ℚ) : 
  (∀ x, (x + 1) * (x^2 - 5*a*x + a) = x^3 + (-5*a + 1)*x^2 + (-9*a)*x + a) → 
  a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l179_17985


namespace NUMINAMATH_CALUDE_vacuum_time_calculation_l179_17988

theorem vacuum_time_calculation (total_free_time dusting_time mopping_time cat_brushing_time num_cats remaining_free_time : ℕ) 
  (h1 : total_free_time = 180)
  (h2 : dusting_time = 60)
  (h3 : mopping_time = 30)
  (h4 : cat_brushing_time = 5)
  (h5 : num_cats = 3)
  (h6 : remaining_free_time = 30) :
  total_free_time - remaining_free_time - (dusting_time + mopping_time + cat_brushing_time * num_cats) = 45 := by
  sorry

end NUMINAMATH_CALUDE_vacuum_time_calculation_l179_17988


namespace NUMINAMATH_CALUDE_scalene_triangle_distinct_lines_l179_17945

/-- A scalene triangle is a triangle where all sides and angles are different -/
structure ScaleneTriangle where
  -- We don't need to define the specific properties here, just the existence of the triangle
  exists_triangle : True

/-- An altitude of a triangle is a line segment from a vertex perpendicular to the opposite side -/
def altitude (t : ScaleneTriangle) : ℕ := 3

/-- A median of a triangle is a line segment from a vertex to the midpoint of the opposite side -/
def median (t : ScaleneTriangle) : ℕ := 3

/-- An angle bisector of a triangle is a line that divides an angle into two equal parts -/
def angle_bisector (t : ScaleneTriangle) : ℕ := 3

/-- The total number of distinct lines in a scalene triangle -/
def total_distinct_lines (t : ScaleneTriangle) : ℕ :=
  altitude t + median t + angle_bisector t

theorem scalene_triangle_distinct_lines (t : ScaleneTriangle) :
  total_distinct_lines t = 9 := by
  sorry

end NUMINAMATH_CALUDE_scalene_triangle_distinct_lines_l179_17945


namespace NUMINAMATH_CALUDE_bread_in_pond_l179_17944

theorem bread_in_pond (total_bread : ℕ) : 
  (total_bread / 2 : ℕ) + 13 + 7 + 30 = total_bread → total_bread = 100 := by
  sorry

end NUMINAMATH_CALUDE_bread_in_pond_l179_17944


namespace NUMINAMATH_CALUDE_birthday_age_multiple_l179_17910

theorem birthday_age_multiple :
  let current_age : ℕ := 9
  let years_ago : ℕ := 6
  let age_then : ℕ := current_age - years_ago
  current_age / age_then = 3 :=
by sorry

end NUMINAMATH_CALUDE_birthday_age_multiple_l179_17910


namespace NUMINAMATH_CALUDE_min_degree_g_l179_17932

/-- Given polynomials f, g, and h satisfying the equation 4f + 5g = h, 
    with deg(f) = 10 and deg(h) = 12, the minimum possible degree of g is 12 -/
theorem min_degree_g (f g h : Polynomial ℝ) 
  (eq : 4 • f + 5 • g = h) 
  (deg_f : Polynomial.degree f = 10)
  (deg_h : Polynomial.degree h = 12) :
  Polynomial.degree g ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_degree_g_l179_17932


namespace NUMINAMATH_CALUDE_simplify_expression_l179_17983

theorem simplify_expression (x : ℝ) : (2*x + 25) + (150*x^2 + 2*x + 25) = 150*x^2 + 4*x + 50 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l179_17983


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l179_17915

theorem quadratic_root_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 10*x + k = 0 ∧ y^2 + 10*y + k = 0) →
  k = 18.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l179_17915


namespace NUMINAMATH_CALUDE_crab_meat_per_dish_l179_17967

/-- Proves that given the conditions of Johnny's crab dish production, he uses 1.5 pounds of crab meat per dish. -/
theorem crab_meat_per_dish (dishes_per_day : ℕ) (crab_cost_per_pound : ℚ) 
  (weekly_crab_cost : ℚ) (operating_days : ℕ) :
  dishes_per_day = 40 →
  crab_cost_per_pound = 8 →
  weekly_crab_cost = 1920 →
  operating_days = 4 →
  (weekly_crab_cost / crab_cost_per_pound) / operating_days / dishes_per_day = 3/2 := by
  sorry

#check crab_meat_per_dish

end NUMINAMATH_CALUDE_crab_meat_per_dish_l179_17967


namespace NUMINAMATH_CALUDE_jaylen_green_beans_l179_17912

/-- Prove that Jaylen has 7 green beans given the conditions of the vegetable problem. -/
theorem jaylen_green_beans :
  ∀ (jaylen_carrots jaylen_cucumbers jaylen_bell_peppers jaylen_green_beans kristin_bell_peppers kristin_green_beans : ℕ),
  jaylen_carrots = 5 →
  jaylen_cucumbers = 2 →
  kristin_bell_peppers = 2 →
  jaylen_bell_peppers = 2 * kristin_bell_peppers →
  kristin_green_beans = 20 →
  jaylen_carrots + jaylen_cucumbers + jaylen_bell_peppers + jaylen_green_beans = 18 →
  jaylen_green_beans = kristin_green_beans / 2 - 3 →
  jaylen_green_beans = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_jaylen_green_beans_l179_17912


namespace NUMINAMATH_CALUDE_angles_same_terminal_side_as_15_l179_17938

-- Define the set of angles with the same terminal side as 15°
def sameTerminalSideAs15 : Set ℝ := {β | ∃ k : ℤ, β = 15 + k * 360}

-- Theorem stating that the set of angles with the same terminal side as 15° 
-- is exactly the set we defined
theorem angles_same_terminal_side_as_15 : 
  {β : ℝ | ∃ k : ℤ, β = 15 + k * 360} = sameTerminalSideAs15 := by
  sorry

-- Note: In Lean, we typically work with radians. For simplicity, we're using degrees here.
-- In a more rigorous setting, we might want to define a conversion between degrees and radians.

end NUMINAMATH_CALUDE_angles_same_terminal_side_as_15_l179_17938


namespace NUMINAMATH_CALUDE_total_trees_after_planting_l179_17917

def current_trees : ℕ := 33
def new_trees : ℕ := 44

theorem total_trees_after_planting :
  current_trees + new_trees = 77 := by sorry

end NUMINAMATH_CALUDE_total_trees_after_planting_l179_17917


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l179_17906

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internals of the plane for this problem
  dummy : Unit

/-- A line in 3D space -/
structure Line3D where
  -- We don't need to define the internals of the line for this problem
  dummy : Unit

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two planes -/
def perpendicular_planes (p1 : Plane3D) (p2 : Plane3D) : Prop :=
  sorry

/-- The main theorem -/
theorem line_perp_parallel_implies_planes_perp 
  (a b g : Plane3D) (l : Line3D) 
  (h1 : a ≠ b) (h2 : a ≠ g) (h3 : b ≠ g)
  (h4 : perpendicular l a) (h5 : parallel l b) : 
  perpendicular_planes a b :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l179_17906


namespace NUMINAMATH_CALUDE_geometric_sequence_m_value_l179_17943

/-- A geometric sequence with common ratio not equal to 1 -/
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  r ≠ 1 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_m_value
  (a : ℕ → ℝ) (r : ℝ) (m : ℕ)
  (h_geom : geometric_sequence a r)
  (h_eq1 : a 5 * a 6 + a 4 * a 7 = 18)
  (h_eq2 : a 1 * a m = 9) :
  m = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_m_value_l179_17943


namespace NUMINAMATH_CALUDE_cos_alpha_value_l179_17905

theorem cos_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.sin (π/6 - α) = -1/3) : Real.cos α = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l179_17905


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l179_17941

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The given complex number -/
def z : ℂ := (1 + 2 * i) * i

/-- A complex number is in the second quadrant if its real part is negative and its imaginary part is positive -/
def is_in_second_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im > 0

theorem z_in_second_quadrant : is_in_second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l179_17941


namespace NUMINAMATH_CALUDE_triangle_inequality_l179_17970

theorem triangle_inequality (a b c S r R : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0 ∧ r > 0 ∧ R > 0) :
  (9 * r) / (2 * S) ≤ (1 / a + 1 / b + 1 / c) ∧ (1 / a + 1 / b + 1 / c) ≤ (9 * R) / (4 * S) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l179_17970


namespace NUMINAMATH_CALUDE_sqrt_x_squared_y_simplification_l179_17955

theorem sqrt_x_squared_y_simplification (x y : ℝ) (h : x * y < 0) :
  Real.sqrt (x^2 * y) = -x * Real.sqrt y := by sorry

end NUMINAMATH_CALUDE_sqrt_x_squared_y_simplification_l179_17955


namespace NUMINAMATH_CALUDE_cone_base_circumference_l179_17993

theorem cone_base_circumference 
  (V : ℝ) (l : ℝ) (θ : ℝ) (h : ℝ) (r : ℝ) (C : ℝ) :
  V = 27 * Real.pi ∧ 
  l = 6 ∧ 
  θ = Real.pi / 3 ∧ 
  h = l * Real.cos θ ∧ 
  V = 1/3 * Real.pi * r^2 * h ∧ 
  C = 2 * Real.pi * r
  → C = 6 * Real.sqrt 3 * Real.pi := by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l179_17993


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l179_17947

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence where S_9 = 72, a_2 + a_4 + a_9 = 24 -/
theorem arithmetic_seq_sum (seq : ArithmeticSequence) (h : seq.S 9 = 72) :
  seq.a 2 + seq.a 4 + seq.a 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_sum_l179_17947


namespace NUMINAMATH_CALUDE_complex_equality_implies_a_equals_three_l179_17979

theorem complex_equality_implies_a_equals_three (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (2 - Complex.I)
  (z.re = z.im) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_a_equals_three_l179_17979


namespace NUMINAMATH_CALUDE_gcd_180_450_l179_17971

theorem gcd_180_450 : Nat.gcd 180 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_450_l179_17971


namespace NUMINAMATH_CALUDE_martha_coffee_savings_l179_17991

/-- Martha's weekly latte cost -/
def weekly_latte_cost : ℚ := 4 * 5

/-- Martha's weekly iced coffee cost -/
def weekly_iced_coffee_cost : ℚ := 2 * 3

/-- Martha's total weekly coffee cost -/
def weekly_coffee_cost : ℚ := weekly_latte_cost + weekly_iced_coffee_cost

/-- Number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Martha's annual coffee cost -/
def annual_coffee_cost : ℚ := weekly_coffee_cost * weeks_per_year

/-- Percentage of coffee spending Martha wants to cut -/
def spending_cut_percentage : ℚ := 25 / 100

/-- Martha's annual savings from reducing coffee spending -/
def annual_savings : ℚ := annual_coffee_cost * spending_cut_percentage

theorem martha_coffee_savings : annual_savings = 338 := by sorry

end NUMINAMATH_CALUDE_martha_coffee_savings_l179_17991


namespace NUMINAMATH_CALUDE_range_of_a_l179_17963

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then x + 4 else x^2 - 2*x

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a ∈ Set.Icc (-5 : ℝ) (4 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l179_17963


namespace NUMINAMATH_CALUDE_shirley_sold_at_least_20_boxes_l179_17958

/-- The number of cases Shirley needs to deliver -/
def num_cases : ℕ := 5

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 4

/-- The number of extra boxes, which is unknown but non-negative -/
def extra_boxes : ℕ := sorry

/-- The total number of boxes Shirley sold -/
def total_boxes : ℕ := num_cases * boxes_per_case + extra_boxes

theorem shirley_sold_at_least_20_boxes : total_boxes ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_shirley_sold_at_least_20_boxes_l179_17958


namespace NUMINAMATH_CALUDE_joe_initial_cars_l179_17937

/-- Given that Joe will have 62 cars after getting 12 more, prove that he initially had 50 cars. -/
theorem joe_initial_cars : 
  ∀ (initial_cars : ℕ), 
  (initial_cars + 12 = 62) → 
  initial_cars = 50 := by
sorry

end NUMINAMATH_CALUDE_joe_initial_cars_l179_17937


namespace NUMINAMATH_CALUDE_beetle_speed_l179_17998

/-- Beetle's speed in km/h given ant's speed and relative distance -/
theorem beetle_speed (ant_distance : ℝ) (time_minutes : ℝ) (beetle_relative_distance : ℝ) :
  ant_distance = 1000 →
  time_minutes = 30 →
  beetle_relative_distance = 0.9 →
  (ant_distance * beetle_relative_distance / time_minutes) * (60 / 1000) = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_beetle_speed_l179_17998


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l179_17999

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1 → x^3 > x^(1/3)) ↔ (∃ x : ℝ, x > 1 ∧ x^3 ≤ x^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l179_17999


namespace NUMINAMATH_CALUDE_function_growth_l179_17930

theorem function_growth (f : ℕ+ → ℝ) 
  (h : ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2) 
  (h4 : f 4 ≥ 25) :
  ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 := by
sorry

end NUMINAMATH_CALUDE_function_growth_l179_17930


namespace NUMINAMATH_CALUDE_independence_test_suitable_for_categorical_variables_l179_17950

/-- Independence test is a statistical method used to determine the relationship between two variables -/
structure IndependenceTest where
  is_statistical_method : Bool
  determines_relationship : Bool
  between_two_variables : Bool

/-- Categorical variables are a type of variable -/
structure CategoricalVariable where
  is_variable : Bool

/-- The statement that the independence test is suitable for examining the relationship between categorical variables -/
theorem independence_test_suitable_for_categorical_variables 
  (test : IndependenceTest) 
  (cat_var : CategoricalVariable) : 
  test.is_statistical_method ∧ 
  test.determines_relationship ∧ 
  test.between_two_variables → 
  (∃ (relationship : CategoricalVariable → CategoricalVariable → Prop), 
    test.determines_relationship ∧ 
    ∀ (x y : CategoricalVariable), relationship x y) := by
  sorry

end NUMINAMATH_CALUDE_independence_test_suitable_for_categorical_variables_l179_17950


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l179_17956

theorem framed_painting_ratio : 
  let painting_size : ℝ := 20
  let frame_side (x : ℝ) := x
  let frame_top_bottom (x : ℝ) := 3 * x
  let framed_width (x : ℝ) := painting_size + 2 * frame_side x
  let framed_height (x : ℝ) := painting_size + 2 * frame_top_bottom x
  let frame_area (x : ℝ) := framed_width x * framed_height x - painting_size^2
  ∃ x : ℝ, 
    x > 0 ∧ 
    frame_area x = painting_size^2 ∧
    (min (framed_width x) (framed_height x)) / (max (framed_width x) (framed_height x)) = 4/7 :=
by sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l179_17956


namespace NUMINAMATH_CALUDE_english_test_question_count_l179_17939

/-- Calculates the number of questions on an English test given the following conditions:
  * There is a math test with 40 questions
  * A student gets 75% of math questions right
  * The student gets 98% of English test questions right
  * The student gets a total of 79 questions right on both tests
-/
def english_test_questions (math_questions : ℕ) (math_correct_percentage : ℚ)
  (english_correct_percentage : ℚ) (total_correct : ℕ) : ℕ :=
  sorry

theorem english_test_question_count :
  english_test_questions 40 (75 / 100) (98 / 100) 79 = 50 := by
  sorry

end NUMINAMATH_CALUDE_english_test_question_count_l179_17939


namespace NUMINAMATH_CALUDE_plane_equation_proof_l179_17959

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Parametric representation of the plane -/
def parametricPlane (s t : ℝ) : Point3D :=
  { x := 2 + 2*s - 3*t
  , y := 1 - 2*s
  , z := 4 + 3*s + 4*t }

/-- The plane equation we want to prove -/
def targetPlane : Plane :=
  { a := 8
  , b := 17
  , c := 6
  , d := -57 }

theorem plane_equation_proof :
  (∀ s t : ℝ, pointOnPlane (parametricPlane s t) targetPlane) ∧
  targetPlane.a > 0 ∧
  Int.gcd (Int.natAbs targetPlane.a) (Int.gcd (Int.natAbs targetPlane.b) (Int.gcd (Int.natAbs targetPlane.c) (Int.natAbs targetPlane.d))) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l179_17959


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l179_17946

theorem arithmetic_series_sum (k : ℕ) : 
  let a₁ : ℤ := k^2 - 1
  let d : ℤ := 1
  let n : ℕ := 2 * k
  let S := (n : ℤ) * (2 * a₁ + (n - 1) * d) / 2
  S = 2 * k^3 + 2 * k^2 - 3 * k :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l179_17946


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l179_17978

theorem quadratic_equation_solution :
  let x₁ := -2 + Real.sqrt 2
  let x₂ := -2 - Real.sqrt 2
  x₁^2 + 4*x₁ + 2 = 0 ∧ x₂^2 + 4*x₂ + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l179_17978


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l179_17964

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l179_17964


namespace NUMINAMATH_CALUDE_arrangement_exists_l179_17992

theorem arrangement_exists (n : ℕ) : 
  ∃ (p : Fin n → ℕ), Function.Injective p ∧ Set.range p = Finset.range n ∧
    ∀ (i j k : Fin n), i < j → j < k → 
      p j ≠ (p i + p k) / 2 := by sorry

end NUMINAMATH_CALUDE_arrangement_exists_l179_17992


namespace NUMINAMATH_CALUDE_mango_community_ratio_l179_17933

/-- Represents the mango harvest and sales problem. -/
structure MangoHarvest where
  total_kg : ℕ  -- Total kilograms of mangoes harvested
  sold_market_kg : ℕ  -- Kilograms of mangoes sold to the market
  mangoes_per_kg : ℕ  -- Number of mangoes per kilogram
  mangoes_left : ℕ  -- Number of mangoes left after sales

/-- The ratio of mangoes sold to the community to total mangoes harvested is 1/3. -/
theorem mango_community_ratio (h : MangoHarvest) 
  (h_total : h.total_kg = 60)
  (h_market : h.sold_market_kg = 20)
  (h_per_kg : h.mangoes_per_kg = 8)
  (h_left : h.mangoes_left = 160) :
  (h.total_kg * h.mangoes_per_kg - h.sold_market_kg * h.mangoes_per_kg - h.mangoes_left) / 
  (h.total_kg * h.mangoes_per_kg) = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_mango_community_ratio_l179_17933


namespace NUMINAMATH_CALUDE_problem_statement_l179_17974

theorem problem_statement : (1 / ((-8^2)^4)) * (-8)^9 = -8 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l179_17974


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l179_17929

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 = 2 →                     -- a_1 = 2
  d ≠ 0 →                       -- d ≠ 0
  (a 2) ^ 2 = a 1 * a 5 →       -- a_1, a_2, a_5 form a geometric sequence
  d = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l179_17929


namespace NUMINAMATH_CALUDE_sum_of_s_and_u_l179_17972

-- Define complex numbers
variable (p q r s t u : ℝ)

-- Define the conditions
def complex_sum_condition (p q r s t u : ℝ) : Prop :=
  Complex.mk (p + r + t) (q + s + u) = Complex.I * (-7)

-- Theorem statement
theorem sum_of_s_and_u 
  (h1 : q = 5)
  (h2 : t = -p - r)
  (h3 : complex_sum_condition p q r s t u) :
  s + u = -12 := by sorry

end NUMINAMATH_CALUDE_sum_of_s_and_u_l179_17972


namespace NUMINAMATH_CALUDE_largest_base5_3digit_in_base10_l179_17995

/-- The largest three-digit number in base-5 -/
def largest_base5_3digit : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

/-- Theorem: The largest three-digit number in base-5, when converted to base-10, is equal to 124 -/
theorem largest_base5_3digit_in_base10 : largest_base5_3digit = 124 := by
  sorry

end NUMINAMATH_CALUDE_largest_base5_3digit_in_base10_l179_17995


namespace NUMINAMATH_CALUDE_gas_usage_difference_l179_17960

theorem gas_usage_difference (adhira_usage felicity_usage : ℝ) : 
  felicity_usage = 23 →
  adhira_usage + felicity_usage = 30 →
  4 * adhira_usage - felicity_usage = 5 := by
sorry

end NUMINAMATH_CALUDE_gas_usage_difference_l179_17960


namespace NUMINAMATH_CALUDE_no_solutions_fibonacci_equation_l179_17920

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n+2) => fib (n+1) + fib n

theorem no_solutions_fibonacci_equation :
  ∀ n : ℕ, n * (fib n) * (fib (n+1)) ≠ (fib (n+2) - 1)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_no_solutions_fibonacci_equation_l179_17920


namespace NUMINAMATH_CALUDE_johns_purchase_cost_l179_17926

/-- Calculates the total cost of John's purchase given the number of gum packs, candy bars, and the cost of a candy bar. -/
def total_cost (gum_packs : ℕ) (candy_bars : ℕ) (candy_bar_cost : ℚ) : ℚ :=
  let gum_cost := candy_bar_cost / 2
  gum_packs * gum_cost + candy_bars * candy_bar_cost

/-- Proves that John's total cost for 2 packs of gum and 3 candy bars is $6, given that each candy bar costs $1.5 and gum costs half as much. -/
theorem johns_purchase_cost : total_cost 2 3 (3/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_johns_purchase_cost_l179_17926


namespace NUMINAMATH_CALUDE_pushups_percentage_l179_17909

def jumping_jacks : ℕ := 12
def pushups : ℕ := 8
def situps : ℕ := 20

def total_exercises : ℕ := jumping_jacks + pushups + situps

def percentage_pushups : ℚ := (pushups : ℚ) / (total_exercises : ℚ) * 100

theorem pushups_percentage : percentage_pushups = 20 := by
  sorry

end NUMINAMATH_CALUDE_pushups_percentage_l179_17909


namespace NUMINAMATH_CALUDE_square_minus_double_eq_one_implies_double_square_minus_quadruple_l179_17928

theorem square_minus_double_eq_one_implies_double_square_minus_quadruple (m : ℝ) :
  m^2 - 2*m = 1 → 2*m^2 - 4*m = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_double_eq_one_implies_double_square_minus_quadruple_l179_17928


namespace NUMINAMATH_CALUDE_inequality_solution_set_l179_17953

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x^2 - 2 ≥ 2*x - a*x}
  (a = 0 → S = {x : ℝ | x ≤ -1}) ∧
  (a > 0 → S = {x : ℝ | x ≥ 2/a ∨ x ≤ -1}) ∧
  (-2 < a ∧ a < 0 → S = {x : ℝ | 2/a ≤ x ∧ x ≤ -1}) ∧
  (a = -2 → S = {x : ℝ | x = -1}) ∧
  (a < -2 → S = {x : ℝ | -1 ≤ x ∧ x ≤ 2/a}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l179_17953


namespace NUMINAMATH_CALUDE_apples_on_table_l179_17913

/-- The number of green apples on the table -/
def green_apples : ℕ := 2

/-- The number of red apples on the table -/
def red_apples : ℕ := 3

/-- The number of yellow apples on the table -/
def yellow_apples : ℕ := 14

/-- The total number of apples on the table -/
def total_apples : ℕ := green_apples + red_apples + yellow_apples

theorem apples_on_table : total_apples = 19 := by
  sorry

end NUMINAMATH_CALUDE_apples_on_table_l179_17913


namespace NUMINAMATH_CALUDE_city_park_highest_difference_l179_17924

/-- Snowfall data for different locations --/
structure SnowfallData where
  mrsHilt : Float
  brecknockSchool : Float
  townLibrary : Float
  cityPark : Float

/-- Calculate the absolute difference between two snowfall measurements --/
def snowfallDifference (a b : Float) : Float :=
  (a - b).abs

/-- Determine the location with the highest snowfall difference compared to Mrs. Hilt's house --/
def highestSnowfallDifference (data : SnowfallData) : String :=
  let schoolDiff := snowfallDifference data.mrsHilt data.brecknockSchool
  let libraryDiff := snowfallDifference data.mrsHilt data.townLibrary
  let parkDiff := snowfallDifference data.mrsHilt data.cityPark
  if parkDiff > schoolDiff && parkDiff > libraryDiff then
    "City Park"
  else if schoolDiff > libraryDiff then
    "Brecknock Elementary School"
  else
    "Town Library"

/-- Theorem: The city park has the highest snowfall difference compared to Mrs. Hilt's house --/
theorem city_park_highest_difference (data : SnowfallData)
  (h1 : data.mrsHilt = 29.7)
  (h2 : data.brecknockSchool = 17.3)
  (h3 : data.townLibrary = 23.8)
  (h4 : data.cityPark = 12.6) :
  highestSnowfallDifference data = "City Park" := by
  sorry

end NUMINAMATH_CALUDE_city_park_highest_difference_l179_17924


namespace NUMINAMATH_CALUDE_skyler_song_count_l179_17981

/-- The number of songs Skyler wrote in total -/
def total_songs (hit_songs top_100_songs unreleased_songs : ℕ) : ℕ :=
  hit_songs + top_100_songs + unreleased_songs

/-- Theorem stating the total number of songs Skyler wrote -/
theorem skyler_song_count :
  ∀ (hit_songs : ℕ),
    hit_songs = 25 →
    ∀ (top_100_songs : ℕ),
      top_100_songs = hit_songs + 10 →
      ∀ (unreleased_songs : ℕ),
        unreleased_songs = hit_songs - 5 →
        total_songs hit_songs top_100_songs unreleased_songs = 80 := by
  sorry

end NUMINAMATH_CALUDE_skyler_song_count_l179_17981


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l179_17962

/-- The surface area of a cube with the same volume as a 9x3x27 rectangular prism is 486 square inches. -/
theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_side : ℝ) : 
  l = 9 ∧ w = 3 ∧ h = 27 →
  cube_side^3 = l * w * h →
  6 * cube_side^2 = 486 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l179_17962
