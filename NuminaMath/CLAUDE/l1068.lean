import Mathlib

namespace NUMINAMATH_CALUDE_max_non_managers_l1068_106849

/-- The maximum number of non-managers in a department with 9 managers, 
    given that the ratio of managers to non-managers must be greater than 7:37 -/
theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 9 →
  (managers : ℚ) / non_managers > 7 / 37 →
  non_managers ≤ 47 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l1068_106849


namespace NUMINAMATH_CALUDE_min_value_of_f_l1068_106895

/-- The quadratic function we're minimizing -/
def f (x y : ℝ) : ℝ := 3*x^2 + 4*x*y + 2*y^2 - 6*x + 8*y + 9

theorem min_value_of_f :
  (∀ x y : ℝ, f x y ≥ -15) ∧ (∃ x y : ℝ, f x y = -15) := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1068_106895


namespace NUMINAMATH_CALUDE_two_integer_tangent_lengths_l1068_106886

def circle_circumference : ℝ := 10

def is_valid_arc_length (x : ℝ) : Prop :=
  0 < x ∧ x < circle_circumference

theorem two_integer_tangent_lengths :
  ∃ (t₁ t₂ : ℕ), t₁ ≠ t₂ ∧
  (∀ m : ℕ, is_valid_arc_length m →
    (∃ n : ℝ, is_valid_arc_length n ∧
      m + n = circle_circumference ∧
      (t₁ : ℝ)^2 = m * n ∨ (t₂ : ℝ)^2 = m * n)) ∧
  (∀ t : ℕ, (∃ m : ℕ, is_valid_arc_length m ∧
    (∃ n : ℝ, is_valid_arc_length n ∧
      m + n = circle_circumference ∧
      (t : ℝ)^2 = m * n)) →
    t = t₁ ∨ t = t₂) :=
by sorry

end NUMINAMATH_CALUDE_two_integer_tangent_lengths_l1068_106886


namespace NUMINAMATH_CALUDE_daily_expense_reduction_l1068_106827

theorem daily_expense_reduction (total_expense : ℕ) (original_days : ℕ) (extended_days : ℕ) :
  total_expense = 360 →
  original_days = 20 →
  extended_days = 24 →
  (total_expense / original_days) - (total_expense / extended_days) = 3 := by
  sorry

end NUMINAMATH_CALUDE_daily_expense_reduction_l1068_106827


namespace NUMINAMATH_CALUDE_bread_slices_problem_l1068_106803

theorem bread_slices_problem (initial_slices : ℕ) : 
  (initial_slices : ℚ) * (2/3) - 2 = 6 → initial_slices = 12 := by
  sorry

end NUMINAMATH_CALUDE_bread_slices_problem_l1068_106803


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l1068_106864

theorem inserted_numbers_sum (a b : ℝ) : 
  0 < a ∧ 0 < b ∧ 
  2 < a ∧ a < b ∧ b < 12 ∧
  (∃ r : ℝ, r > 0 ∧ a = 2 * r ∧ b = 2 * r^2) ∧
  (∃ d : ℝ, b = a + d ∧ 12 = b + d) →
  a + b = 12 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l1068_106864


namespace NUMINAMATH_CALUDE_expression_evaluation_l1068_106802

theorem expression_evaluation (a b : ℚ) (h1 : a = -1) (h2 : b = 1/2) :
  5*a*b - 2*(3*a*b - (4*a*b^2 + 1/2*a*b)) - 5*a*b^2 = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1068_106802


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_l1068_106819

/-- Represents a triangular figure made of toothpicks -/
structure TriangularFigure where
  toothpicks : ℕ
  triangles : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : TriangularFigure) : ℕ :=
  15

/-- Theorem: For a triangular figure with 40 toothpicks and at least 35 triangles,
    the minimum number of toothpicks to remove to eliminate all triangles is 15 -/
theorem min_toothpicks_removal (figure : TriangularFigure) 
    (h1 : figure.toothpicks = 40) 
    (h2 : figure.triangles ≥ 35) : 
  min_toothpicks_to_remove figure = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_min_toothpicks_removal_l1068_106819


namespace NUMINAMATH_CALUDE_new_student_weight_l1068_106861

/-- Calculates the weight of a new student given the initial and final conditions of a group of students. -/
theorem new_student_weight
  (initial_count : ℕ)
  (initial_avg : ℝ)
  (final_count : ℕ)
  (final_avg : ℝ)
  (h1 : initial_count = 19)
  (h2 : initial_avg = 15)
  (h3 : final_count = initial_count + 1)
  (h4 : final_avg = 14.6) :
  final_count * final_avg - initial_count * initial_avg = 7 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l1068_106861


namespace NUMINAMATH_CALUDE_distance_to_centroid_l1068_106859

-- Define a triangle by its side lengths
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define a point inside the triangle by its distances from the vertices
structure InnerPoint (t : Triangle) where
  p : ℝ
  q : ℝ
  r : ℝ
  pos_p : 0 < p
  pos_q : 0 < q
  pos_r : 0 < r

-- Theorem statement
theorem distance_to_centroid (t : Triangle) (d : InnerPoint t) :
  ∃ (ds : ℝ), ds^2 = (3 * (d.p^2 + d.q^2 + d.r^2) - (t.a^2 + t.b^2 + t.c^2)) / 9 :=
sorry

end NUMINAMATH_CALUDE_distance_to_centroid_l1068_106859


namespace NUMINAMATH_CALUDE_arrangements_with_restriction_l1068_106835

def num_actors : ℕ := 6

-- Define a function to calculate the number of arrangements
def num_arrangements (n : ℕ) (restricted_positions : ℕ) : ℕ :=
  (n - restricted_positions) * (n - 1).factorial

-- Theorem statement
theorem arrangements_with_restriction :
  num_arrangements num_actors 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_restriction_l1068_106835


namespace NUMINAMATH_CALUDE_ellipse_properties_l1068_106899

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Checks if two foci and one endpoint of minor axis form an equilateral triangle -/
def equilateral_triangle (e : Ellipse) : Prop :=
  e.a = 2 * Real.sqrt (e.a^2 - e.b^2)

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  ∀ x y, x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Represents a line with slope k passing through point (0, 2) -/
structure Line where
  k : ℝ

/-- Checks if a line intersects the ellipse at two distinct points M and N -/
def intersects_ellipse (e : Ellipse) (l : Line) : Prop :=
  ∃ x₁ y₁ x₂ y₂,
    x₁ ≠ x₂ ∧
    x₁^2 / e.a^2 + y₁^2 / e.b^2 = 1 ∧
    x₂^2 / e.a^2 + y₂^2 / e.b^2 = 1 ∧
    y₁ = l.k * x₁ + 2 ∧
    y₂ = l.k * x₂ + 2

/-- Checks if OM · ON = 2 for intersection points M and N -/
def satisfies_dot_product (e : Ellipse) (l : Line) : Prop :=
  ∃ x₁ y₁ x₂ y₂,
    intersects_ellipse e l ∧
    x₁ * x₂ + y₁ * y₂ = 2

theorem ellipse_properties (e : Ellipse)
    (h_minor_axis : e.b = Real.sqrt 3)
    (h_equilateral : equilateral_triangle e) :
    standard_equation { a := 2, b := Real.sqrt 3, h_positive := sorry } ∧
    ∃ l : Line, l.k = Real.sqrt 2 / 2 ∨ l.k = -Real.sqrt 2 / 2 ∧
              satisfies_dot_product e l :=
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1068_106899


namespace NUMINAMATH_CALUDE_train_length_train_length_example_l1068_106896

/-- The length of a train given its speed and time to cross a pole --/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ := by
  sorry

/-- Proof that a train with speed 60 km/hr crossing a pole in 4 seconds has a length of approximately 66.68 meters --/
theorem train_length_example : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |train_length 60 4 - 66.68| < ε := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_example_l1068_106896


namespace NUMINAMATH_CALUDE_edward_binders_l1068_106897

def total_cards : ℕ := 763
def cards_per_binder : ℕ := 109

theorem edward_binders : 
  total_cards / cards_per_binder = 7 :=
by sorry

end NUMINAMATH_CALUDE_edward_binders_l1068_106897


namespace NUMINAMATH_CALUDE_max_value_of_symmetric_f_l1068_106832

/-- A function f that is symmetric about x = -2 and has the form (1-x^2)(x^2+ax+b) -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

/-- The symmetry condition for f about x = -2 -/
def symmetric_about_neg_two (a b : ℝ) : Prop :=
  ∀ t, f a b (-2 + t) = f a b (-2 - t)

/-- The theorem stating that if f is symmetric about x = -2, its maximum value is 16 -/
theorem max_value_of_symmetric_f (a b : ℝ) 
  (h : symmetric_about_neg_two a b) : 
  ∃ x, f a b x = 16 ∧ ∀ y, f a b y ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_symmetric_f_l1068_106832


namespace NUMINAMATH_CALUDE_rachel_essay_time_l1068_106836

/-- Calculates the total time spent on an essay in hours -/
def total_essay_time (pages_written : ℕ) (writing_rate : ℚ) (research_time : ℕ) (editing_time : ℕ) : ℚ :=
  let writing_time : ℚ := pages_written * writing_rate
  let total_minutes : ℚ := research_time + writing_time + editing_time
  total_minutes / 60

/-- Theorem: Rachel spends 5 hours completing the essay -/
theorem rachel_essay_time : 
  total_essay_time 6 (30 : ℚ) 45 75 = 5 := by
  sorry

end NUMINAMATH_CALUDE_rachel_essay_time_l1068_106836


namespace NUMINAMATH_CALUDE_lattice_triangle_area_l1068_106884

/-- A lattice point is a point with integer coordinates. -/
def LatticePoint (p : ℝ × ℝ) : Prop := ∃ (x y : ℤ), p = (↑x, ↑y)

/-- A triangle with vertices at lattice points. -/
structure LatticeTriangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v1_lattice : LatticePoint v1
  v2_lattice : LatticePoint v2
  v3_lattice : LatticePoint v3

/-- The number of lattice points strictly inside the triangle. -/
def interior_points (t : LatticeTriangle) : ℕ := sorry

/-- The number of lattice points on the sides of the triangle (excluding vertices). -/
def boundary_points (t : LatticeTriangle) : ℕ := sorry

/-- The area of a triangle. -/
def triangle_area (t : LatticeTriangle) : ℝ := sorry

/-- Theorem: The area of a lattice triangle with n interior points and m boundary points
    (excluding vertices) is equal to n + m/2 + 1/2. -/
theorem lattice_triangle_area (t : LatticeTriangle) :
  triangle_area t = interior_points t + (boundary_points t : ℝ) / 2 + 1 / 2 := by sorry

end NUMINAMATH_CALUDE_lattice_triangle_area_l1068_106884


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1068_106879

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, prove that if they are parallel, then x = 3 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1068_106879


namespace NUMINAMATH_CALUDE_divisibility_problem_l1068_106853

theorem divisibility_problem (p : ℕ) (hp : p.Prime) (hp_gt_7 : p > 7) 
  (hp_mod_6 : p % 6 = 1) (m : ℕ) (hm : m = 2^p - 1) : 
  (127 * m) ∣ (2^(m-1) - 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1068_106853


namespace NUMINAMATH_CALUDE_subset_P_l1068_106889

-- Define the set P
def P : Set ℝ := {x | x ≤ 3}

-- State the theorem
theorem subset_P : {-1} ⊆ P := by sorry

end NUMINAMATH_CALUDE_subset_P_l1068_106889


namespace NUMINAMATH_CALUDE_parking_duration_for_5_5_yuan_l1068_106870

/-- Calculates the parking duration given the total fee paid -/
def parking_duration (total_fee : ℚ) : ℚ :=
  (total_fee - 0.5) / (0.5 + 0.5) + 1

/-- Theorem stating that given the specific fee paid, the parking duration is 6 hours -/
theorem parking_duration_for_5_5_yuan :
  parking_duration 5.5 = 6 := by sorry

end NUMINAMATH_CALUDE_parking_duration_for_5_5_yuan_l1068_106870


namespace NUMINAMATH_CALUDE_maddie_spent_95_l1068_106869

/-- Calculates the total amount spent on T-shirts with a bulk discount -/
def total_spent (white_packs blue_packs : ℕ) 
                (white_per_pack blue_per_pack : ℕ) 
                (white_price blue_price : ℚ) 
                (discount_percent : ℚ) : ℚ :=
  let white_total := white_packs * white_per_pack * white_price
  let blue_total := blue_packs * blue_per_pack * blue_price
  let subtotal := white_total + blue_total
  let discount := subtotal * (discount_percent / 100)
  subtotal - discount

/-- Proves that Maddie spent $95 on T-shirts -/
theorem maddie_spent_95 : 
  total_spent 2 4 5 3 4 5 5 = 95 := by
  sorry

end NUMINAMATH_CALUDE_maddie_spent_95_l1068_106869


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l1068_106878

theorem integer_pairs_satisfying_equation : 
  {(x, y) : ℤ × ℤ | x * (x + 1) * (x + 2) * (x + 3) = y * (y + 1)} = 
  {(0, 0), (-1, 0), (-2, 0), (-3, 0), (0, -1), (-1, -1), (-2, -1), (-3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l1068_106878


namespace NUMINAMATH_CALUDE_power_of_power_equals_ten_l1068_106828

theorem power_of_power_equals_ten (m : ℝ) : (m^2)^5 = m^10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_equals_ten_l1068_106828


namespace NUMINAMATH_CALUDE_exchange_rate_20_percent_increase_same_digits_l1068_106801

/-- Represents an exchange rate as a pair of integers (whole, fraction) -/
structure ExchangeRate where
  whole : ℕ
  fraction : ℕ
  h_fraction : fraction < 100

/-- Checks if two exchange rates have the same digits in different order -/
def same_digits_different_order (x y : ExchangeRate) : Prop := sorry

/-- Calculates the 20% increase of an exchange rate -/
def increase_by_20_percent (x : ExchangeRate) : ExchangeRate := sorry

/-- Main theorem: There exists an exchange rate that, when increased by 20%,
    results in a new rate with the same digits in a different order -/
theorem exchange_rate_20_percent_increase_same_digits :
  ∃ (x : ExchangeRate), same_digits_different_order x (increase_by_20_percent x) := by
  sorry

end NUMINAMATH_CALUDE_exchange_rate_20_percent_increase_same_digits_l1068_106801


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l1068_106858

theorem purely_imaginary_condition (a : ℝ) :
  a = -1 ↔ (∃ b : ℝ, Complex.mk (a^2 - 1) (a - 1) = Complex.I * b) := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l1068_106858


namespace NUMINAMATH_CALUDE_compound_weight_l1068_106812

/-- Given a compound with a molecular weight of 2670 grams/mole, 
    prove that the total weight of 10 moles of this compound is 26700 grams. -/
theorem compound_weight (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 2670 → moles = 10 → moles * molecular_weight = 26700 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_l1068_106812


namespace NUMINAMATH_CALUDE_problem_solution_l1068_106809

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem problem_solution :
  let x := sum_integers 20 30
  let y := count_even_integers 20 30
  x + y = 281 → y = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1068_106809


namespace NUMINAMATH_CALUDE_wedding_decoration_cost_l1068_106874

/-- Calculates the total cost of decorations for a wedding reception --/
def total_decoration_cost (num_tables : ℕ) (tablecloth_cost : ℕ) (place_settings_per_table : ℕ) 
  (place_setting_cost : ℕ) (roses_per_centerpiece : ℕ) (rose_cost : ℕ) (lilies_per_centerpiece : ℕ) 
  (lily_cost : ℕ) : ℕ :=
  num_tables * (tablecloth_cost + place_settings_per_table * place_setting_cost + 
  roses_per_centerpiece * rose_cost + lilies_per_centerpiece * lily_cost)

/-- Theorem stating that the total decoration cost for the given parameters is 3500 --/
theorem wedding_decoration_cost : 
  total_decoration_cost 20 25 4 10 10 5 15 4 = 3500 := by
  sorry

#eval total_decoration_cost 20 25 4 10 10 5 15 4

end NUMINAMATH_CALUDE_wedding_decoration_cost_l1068_106874


namespace NUMINAMATH_CALUDE_symmetry_of_sine_function_l1068_106872

/-- Given a function f(x) = sin(wx + π/4) where w > 0 and 
    the minimum positive period of f(x) is π, 
    prove that the graph of f(x) is symmetrical about the line x = π/8 -/
theorem symmetry_of_sine_function (w : ℝ) (h1 : w > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (w * x + π / 4)
  (∀ x : ℝ, f (x + π) = f x) →  -- minimum positive period is π
  ∀ x : ℝ, f (π / 4 - x) = f (π / 4 + x) := by
sorry

end NUMINAMATH_CALUDE_symmetry_of_sine_function_l1068_106872


namespace NUMINAMATH_CALUDE_combine_like_terms_l1068_106877

-- Define the theorem
theorem combine_like_terms (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l1068_106877


namespace NUMINAMATH_CALUDE_chef_cherries_remaining_l1068_106807

theorem chef_cherries_remaining (initial_cherries used_cherries : ℕ) 
  (h1 : initial_cherries = 77)
  (h2 : used_cherries = 60) :
  initial_cherries - used_cherries = 17 := by
  sorry

end NUMINAMATH_CALUDE_chef_cherries_remaining_l1068_106807


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1068_106850

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 11 = 24)
  (h_a4 : a 4 = 3) :
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1068_106850


namespace NUMINAMATH_CALUDE_solve_for_A_l1068_106829

theorem solve_for_A (x₁ x₂ A : ℂ) : 
  x₁ ≠ x₂ →
  x₁ * (x₁ + 1) = A →
  x₂ * (x₂ + 1) = A →
  x₁^4 + 3*x₁^3 + 5*x₁ = x₂^4 + 3*x₂^3 + 5*x₂ →
  A = -7 := by sorry

end NUMINAMATH_CALUDE_solve_for_A_l1068_106829


namespace NUMINAMATH_CALUDE_theater_seats_l1068_106826

theorem theater_seats : ∃ n : ℕ, n < 60 ∧ n % 9 = 5 ∧ n % 6 = 3 ∧ n = 41 := by
  sorry

end NUMINAMATH_CALUDE_theater_seats_l1068_106826


namespace NUMINAMATH_CALUDE_austin_starting_amount_l1068_106833

def robot_cost : ℚ := 875 / 100
def discount_rate : ℚ := 1 / 10
def coupon_discount : ℚ := 5
def tax_rate : ℚ := 2 / 25
def total_tax : ℚ := 722 / 100
def shipping_fee : ℚ := 499 / 100
def gift_card : ℚ := 25
def change : ℚ := 1153 / 100

def total_robots : ℕ := 2 * 1 + 3 * 2 + 2 * 3

theorem austin_starting_amount (initial_amount : ℚ) :
  (∃ (discounted_price : ℚ),
    discounted_price = total_robots * robot_cost * (1 - discount_rate) - coupon_discount ∧
    total_tax = discounted_price * tax_rate ∧
    initial_amount = discounted_price + total_tax + shipping_fee - gift_card + change) →
  initial_amount = 7746 / 100 :=
by sorry

end NUMINAMATH_CALUDE_austin_starting_amount_l1068_106833


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1068_106868

/-- Given a principal amount P and an interest rate R (as a percentage),
    if the amount after 2 years is 780 and after 7 years is 1020,
    then the principal amount P is 684. -/
theorem simple_interest_problem (P R : ℚ) : 
  P + (P * R * 2) / 100 = 780 →
  P + (P * R * 7) / 100 = 1020 →
  P = 684 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1068_106868


namespace NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l1068_106847

theorem roots_of_cubic_polynomial :
  let p : ℝ → ℝ := λ x => x^3 - 2*x^2 - 5*x + 6
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l1068_106847


namespace NUMINAMATH_CALUDE_product_of_powers_equals_sum_l1068_106822

theorem product_of_powers_equals_sum (w x y z k : ℕ) : 
  2^w * 3^x * 5^y * 7^z * 11^k = 900 → 2*w + 3*x + 5*y + 7*z + 11*k = 20 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_sum_l1068_106822


namespace NUMINAMATH_CALUDE_geometric_sequence_302nd_term_l1068_106856

/-- Given a geometric sequence with first term 8 and second term -16, 
    the 302nd term is -2^304 -/
theorem geometric_sequence_302nd_term : 
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n + 2) = a (n + 1) * (a (n + 1) / a n)) →  -- geometric sequence condition
    a 1 = 8 →                                           -- first term
    a 2 = -16 →                                         -- second term
    a 302 = -2^304 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_302nd_term_l1068_106856


namespace NUMINAMATH_CALUDE_min_value_constraint_l1068_106820

theorem min_value_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  x^2 + 8*x*y + 25*y^2 + 16*y*z + 9*z^2 ≥ 403/9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_constraint_l1068_106820


namespace NUMINAMATH_CALUDE_hyperbola_equation_theorem_l1068_106825

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  eccentricity : ℝ
  real_axis_length : ℝ

/-- The equation of a hyperbola with given properties -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (x^2 / 4 - y^2 / 12 = 1) ∨ (y^2 / 4 - x^2 / 12 = 1)

/-- Theorem stating that a hyperbola with the given properties has one of the two specified equations -/
theorem hyperbola_equation_theorem (h : Hyperbola) 
    (h_center : h.center = (0, 0))
    (h_eccentricity : h.eccentricity = 2)
    (h_real_axis : h.real_axis_length = 4) :
    ∀ x y : ℝ, hyperbola_equation h x y := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_equation_theorem_l1068_106825


namespace NUMINAMATH_CALUDE_sum_of_3rd_4th_5th_terms_l1068_106873

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem sum_of_3rd_4th_5th_terms
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : geometric_sequence a)
  (h_ratio : ∃ (r : ℝ), ∀ n, a (n + 1) = 2 * a n)
  (h_sum_first_three : a 1 + a 2 + a 3 = 21) :
  a 3 + a 4 + a 5 = 84 :=
sorry

end NUMINAMATH_CALUDE_sum_of_3rd_4th_5th_terms_l1068_106873


namespace NUMINAMATH_CALUDE_remainder_when_divided_by_x_plus_2_l1068_106867

/-- A polynomial of degree 4 -/
def q (x : ℝ) : ℝ := 2 * x^4 - 3 * x^2 - 13 * x + 6

/-- The remainder theorem -/
def remainder_theorem (p : ℝ → ℝ) (a : ℝ) : ℝ := p a

theorem remainder_when_divided_by_x_plus_2 :
  remainder_theorem q 2 = 6 →
  remainder_theorem q (-2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_remainder_when_divided_by_x_plus_2_l1068_106867


namespace NUMINAMATH_CALUDE_elective_courses_schemes_l1068_106841

theorem elective_courses_schemes (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 10 → k = 3 → m = 3 →
  (Nat.choose (n - m) k + m * Nat.choose (n - m) (k - 1) = 98) :=
by sorry

end NUMINAMATH_CALUDE_elective_courses_schemes_l1068_106841


namespace NUMINAMATH_CALUDE_square_minus_four_times_plus_four_equals_six_l1068_106813

theorem square_minus_four_times_plus_four_equals_six (a : ℝ) :
  a = Real.sqrt 6 + 2 → a^2 - 4*a + 4 = 6 := by sorry

end NUMINAMATH_CALUDE_square_minus_four_times_plus_four_equals_six_l1068_106813


namespace NUMINAMATH_CALUDE_regression_line_equation_l1068_106852

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear equation in the form y = mx + b -/
structure LinearEquation where
  slope : ℝ
  intercept : ℝ

/-- Check if a point lies on a given linear equation -/
def pointOnLine (p : Point) (eq : LinearEquation) : Prop :=
  p.y = eq.slope * p.x + eq.intercept

/-- The theorem to be proved -/
theorem regression_line_equation 
  (slope : ℝ) 
  (center : Point) 
  (h_slope : slope = 1.23)
  (h_center : center = ⟨4, 5⟩) :
  ∃ (eq : LinearEquation), 
    eq.slope = slope ∧ 
    pointOnLine center eq ∧ 
    eq = ⟨1.23, 0.08⟩ := by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l1068_106852


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l1068_106846

theorem solve_fraction_equation :
  ∃ x : ℚ, (1 / 4 : ℚ) - (1 / 5 : ℚ) = 1 / x ∧ x = 20 := by
sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l1068_106846


namespace NUMINAMATH_CALUDE_problem_1_l1068_106824

theorem problem_1 (x y : ℝ) : (-3 * x^2 * y)^2 * (2 * x * y^2) / (-6 * x^3 * y^4) = -3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1068_106824


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1068_106823

theorem sqrt_product_equality : Real.sqrt 54 * Real.sqrt 50 * Real.sqrt 6 = 90 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1068_106823


namespace NUMINAMATH_CALUDE_calculate_expression_l1068_106837

theorem calculate_expression (a : ℝ) : a * a^2 - 2 * a^3 = -a^3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1068_106837


namespace NUMINAMATH_CALUDE_curve_symmetry_l1068_106875

/-- A curve in the xy-plane -/
class Curve (f : ℝ → ℝ → Prop) : Prop

/-- Symmetry of a curve with respect to a line -/
def symmetricTo (f : ℝ → ℝ → Prop) (l : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y ↔ f (y + 3) (x - 3)

/-- The line x - y - 3 = 0 -/
def symmetryLine (x y : ℝ) : ℝ := x - y - 3

/-- Theorem: If a curve f is symmetric with respect to the line x - y - 3 = 0,
    then its equation is f(y + 3, x - 3) = 0 -/
theorem curve_symmetry (f : ℝ → ℝ → Prop) [Curve f] 
    (h : symmetricTo f symmetryLine) :
  ∀ x y, f x y ↔ f (y + 3) (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetry_l1068_106875


namespace NUMINAMATH_CALUDE_min_norm_v_l1068_106817

open Real
open Vector

/-- Given a vector v such that ‖v + (4, 2)‖ = 10, the minimum value of ‖v‖ is 10 - 2√5 -/
theorem min_norm_v (v : ℝ × ℝ) (h : ‖v + (4, 2)‖ = 10) :
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * sqrt 5 ∧ ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ := by
sorry


end NUMINAMATH_CALUDE_min_norm_v_l1068_106817


namespace NUMINAMATH_CALUDE_necessary_condition_for_inequality_l1068_106811

theorem necessary_condition_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 2 3 → x^2 - a ≤ 0) → 
  (a ≥ 8 ∧ ∃ b : ℝ, b ≥ 8 ∧ ∃ y : ℝ, y ∈ Set.Icc 2 3 ∧ y^2 - b > 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_condition_for_inequality_l1068_106811


namespace NUMINAMATH_CALUDE_oranges_picked_total_l1068_106815

/-- The number of oranges Joan picked -/
def joan_oranges : ℕ := 37

/-- The number of oranges Sara picked -/
def sara_oranges : ℕ := 10

/-- The total number of oranges picked -/
def total_oranges : ℕ := joan_oranges + sara_oranges

theorem oranges_picked_total :
  total_oranges = 47 := by sorry

end NUMINAMATH_CALUDE_oranges_picked_total_l1068_106815


namespace NUMINAMATH_CALUDE_smallest_and_largest_with_digit_sum_17_l1068_106831

def digit_sum (n : ℕ) : ℕ := sorry

def all_digits_different (n : ℕ) : Prop := sorry

theorem smallest_and_largest_with_digit_sum_17 :
  ∃ (smallest largest : ℕ),
    (∀ n : ℕ, digit_sum n = 17 → all_digits_different n →
      smallest ≤ n ∧ n ≤ largest) ∧
    digit_sum smallest = 17 ∧
    all_digits_different smallest ∧
    digit_sum largest = 17 ∧
    all_digits_different largest ∧
    smallest = 89 ∧
    largest = 743210 :=
sorry

end NUMINAMATH_CALUDE_smallest_and_largest_with_digit_sum_17_l1068_106831


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1068_106840

theorem inequality_equivalence (x y : ℝ) : y - x < Real.sqrt (x^2) ↔ y < 0 ∨ y < 2*x := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1068_106840


namespace NUMINAMATH_CALUDE_missing_digit_is_five_l1068_106891

def largest_number (x : ℕ) : ℕ :=
  if x ≥ 2 then 9000 + 100 * x + 21 else 9000 + 200 + x

def smallest_number (x : ℕ) : ℕ := 1000 + 200 + 90 + x

theorem missing_digit_is_five :
  ∀ x : ℕ, x < 10 →
    largest_number x - smallest_number x = 8262 →
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_missing_digit_is_five_l1068_106891


namespace NUMINAMATH_CALUDE_water_left_is_84_ounces_l1068_106848

/-- Represents the water cooler problem --/
def water_cooler_problem (initial_gallons : ℕ) (ounces_per_cup : ℕ) (rows : ℕ) (chairs_per_row : ℕ) (ounces_per_gallon : ℕ) : ℕ :=
  let initial_ounces := initial_gallons * ounces_per_gallon
  let total_cups := rows * chairs_per_row
  let ounces_poured := total_cups * ounces_per_cup
  initial_ounces - ounces_poured

/-- Theorem stating that the water left in the cooler is 84 ounces --/
theorem water_left_is_84_ounces :
  water_cooler_problem 3 6 5 10 128 = 84 := by
  sorry

end NUMINAMATH_CALUDE_water_left_is_84_ounces_l1068_106848


namespace NUMINAMATH_CALUDE_tan_strictly_increasing_interval_l1068_106888

/-- The function f(x) = tan(x - π/4) is strictly increasing on the interval (kπ - π/4, kπ + 3π/4) for all k ∈ ℤ -/
theorem tan_strictly_increasing_interval (k : ℤ) :
  StrictMonoOn (fun x => Real.tan (x - π/4)) (Set.Ioo (k * π - π/4) (k * π + 3*π/4)) := by
  sorry

end NUMINAMATH_CALUDE_tan_strictly_increasing_interval_l1068_106888


namespace NUMINAMATH_CALUDE_f_g_derivatives_neg_l1068_106843

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
variable (hf : ∀ x, f (-x) = -f x)
variable (hg : ∀ x, g (-x) = g x)

-- Define the derivative properties for x > 0
variable (hf_deriv_pos : ∀ x, x > 0 → deriv f x > 0)
variable (hg_deriv_pos : ∀ x, x > 0 → deriv g x > 0)

-- State the theorem
theorem f_g_derivatives_neg (x : ℝ) (hx : x < 0) : 
  deriv f x > 0 ∧ deriv g x < 0 :=
sorry

end NUMINAMATH_CALUDE_f_g_derivatives_neg_l1068_106843


namespace NUMINAMATH_CALUDE_sum_of_min_max_cubic_expression_l1068_106893

theorem sum_of_min_max_cubic_expression (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 30) :
  let f := fun (x y z w : ℝ) => 4 * (x^3 + y^3 + z^3 + w^3) - 3 * (x^2 + y^2 + z^2 + w^2)^2
  (⨅ (p : Fin 4 → ℝ) (h : p 0 + p 1 + p 2 + p 3 = 10 ∧ p 0^2 + p 1^2 + p 2^2 + p 3^2 = 30), f (p 0) (p 1) (p 2) (p 3)) +
  (⨆ (p : Fin 4 → ℝ) (h : p 0 + p 1 + p 2 + p 3 = 10 ∧ p 0^2 + p 1^2 + p 2^2 + p 3^2 = 30), f (p 0) (p 1) (p 2) (p 3)) = 404 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_min_max_cubic_expression_l1068_106893


namespace NUMINAMATH_CALUDE_angle_D_measure_l1068_106855

structure CyclicQuadrilateral where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_360 : A + B + C + D = 360
  ratio_ABC : ∃ (x : ℝ), A = 3*x ∧ B = 4*x ∧ C = 6*x

theorem angle_D_measure (q : CyclicQuadrilateral) : q.D = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l1068_106855


namespace NUMINAMATH_CALUDE_adjacent_probability_in_two_by_three_l1068_106880

/-- Represents a 2x3 seating arrangement -/
def SeatingArrangement := Fin 2 → Fin 3 → Fin 6

/-- Two positions are adjacent if they are next to each other in the same row or column -/
def adjacent (pos1 pos2 : Fin 2 × Fin 3) : Prop :=
  (pos1.1 = pos2.1 ∧ (pos1.2.val + 1 = pos2.2.val ∨ pos2.2.val + 1 = pos1.2.val)) ∨
  (pos1.2 = pos2.2 ∧ pos1.1 ≠ pos2.1)

/-- The probability of two specific students being adjacent in a random seating arrangement -/
def probability_adjacent : ℚ :=
  7 / 15

theorem adjacent_probability_in_two_by_three :
  probability_adjacent = 7 / 15 := by sorry

end NUMINAMATH_CALUDE_adjacent_probability_in_two_by_three_l1068_106880


namespace NUMINAMATH_CALUDE_f_value_at_3_l1068_106805

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (x + Real.sqrt (x^2 + 1)) + a * x^7 + b * x^3 - 4

theorem f_value_at_3 (a b : ℝ) (h : f a b (-3) = 4) : f a b 3 = -12 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l1068_106805


namespace NUMINAMATH_CALUDE_sum_digits_first_2002_even_integers_l1068_106865

/-- The number of digits in a positive integer -/
def numDigits (n : ℕ) : ℕ := sorry

/-- The nth positive even integer -/
def nthEvenInteger (n : ℕ) : ℕ := sorry

/-- The sum of digits for the first n positive even integers -/
def sumDigitsFirstNEvenIntegers (n : ℕ) : ℕ := sorry

/-- Theorem: The total number of digits used to write the first 2002 positive even integers is 7456 -/
theorem sum_digits_first_2002_even_integers : 
  sumDigitsFirstNEvenIntegers 2002 = 7456 := by sorry

end NUMINAMATH_CALUDE_sum_digits_first_2002_even_integers_l1068_106865


namespace NUMINAMATH_CALUDE_cricket_team_captain_age_l1068_106814

theorem cricket_team_captain_age (team_size : ℕ) (whole_team_avg_age : ℕ) 
  (captain_age wicket_keeper_age : ℕ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 3 →
  whole_team_avg_age = 21 →
  (whole_team_avg_age * team_size - captain_age - wicket_keeper_age) / (team_size - 2) + 1 = whole_team_avg_age →
  captain_age = 24 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_captain_age_l1068_106814


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l1068_106838

theorem lcm_gcd_product : Nat.lcm 6 (Nat.lcm 8 12) * Nat.gcd 6 (Nat.gcd 8 12) = 48 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l1068_106838


namespace NUMINAMATH_CALUDE_merchant_profit_l1068_106842

theorem merchant_profit (C S : ℝ) (h : 17 * C = 16 * S) :
  (S - C) / C * 100 = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_l1068_106842


namespace NUMINAMATH_CALUDE_loops_per_day_l1068_106806

def weekly_goal : ℕ := 3500
def track_length : ℕ := 50
def days_in_week : ℕ := 7

theorem loops_per_day : 
  ∀ (goal : ℕ) (track : ℕ) (days : ℕ),
  goal = weekly_goal → 
  track = track_length → 
  days = days_in_week →
  (goal / track) / days = 10 := by sorry

end NUMINAMATH_CALUDE_loops_per_day_l1068_106806


namespace NUMINAMATH_CALUDE_speed_ratio_is_two_l1068_106845

/-- Given a round trip with total distance 60 km, total time 6 hours, and return speed 15 km/h,
    the ratio of return speed to outbound speed is 2. -/
theorem speed_ratio_is_two 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (return_speed : ℝ) 
  (h1 : total_distance = 60) 
  (h2 : total_time = 6) 
  (h3 : return_speed = 15) : 
  return_speed / ((total_distance / 2) / (total_time - total_distance / (2 * return_speed))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_is_two_l1068_106845


namespace NUMINAMATH_CALUDE_blue_has_most_marbles_blue_greater_than_others_l1068_106810

/-- Represents the colors of marbles -/
inductive Color
| Red
| Blue
| Yellow

/-- Represents the marble counting problem -/
structure MarbleCounting where
  total : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- The conditions of the marble counting problem -/
def marbleProblem : MarbleCounting where
  total := 24
  red := 24 / 4
  blue := 24 / 4 + 6
  yellow := 24 - (24 / 4 + (24 / 4 + 6))

/-- Function to determine which color has the most marbles -/
def mostMarbles (mc : MarbleCounting) : Color :=
  if mc.blue > mc.red ∧ mc.blue > mc.yellow then Color.Blue
  else if mc.red > mc.blue ∧ mc.red > mc.yellow then Color.Red
  else Color.Yellow

/-- Theorem stating that blue has the most marbles in the given problem -/
theorem blue_has_most_marbles :
  mostMarbles marbleProblem = Color.Blue :=
by
  sorry

/-- Theorem proving that the number of blue marbles is greater than both red and yellow -/
theorem blue_greater_than_others (mc : MarbleCounting) :
  mc.blue > mc.red ∧ mc.blue > mc.yellow →
  mostMarbles mc = Color.Blue :=
by
  sorry

end NUMINAMATH_CALUDE_blue_has_most_marbles_blue_greater_than_others_l1068_106810


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1068_106808

theorem sum_of_solutions_quadratic (x : ℝ) :
  let a : ℝ := -3
  let b : ℝ := -18
  let c : ℝ := 81
  let sum_of_roots := -b / a
  (a * x^2 + b * x + c = 0) → sum_of_roots = -6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1068_106808


namespace NUMINAMATH_CALUDE_power_equality_l1068_106882

theorem power_equality (p : ℕ) : 81^6 = 3^p → p = 24 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1068_106882


namespace NUMINAMATH_CALUDE_ab_max_and_sum_min_l1068_106863

theorem ab_max_and_sum_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 7 * b = 10) :
  (ab ≤ 25 / 21) ∧ (3 / a + 7 / b ≥ 10) := by
  sorry

end NUMINAMATH_CALUDE_ab_max_and_sum_min_l1068_106863


namespace NUMINAMATH_CALUDE_quadratic_function_equal_values_l1068_106866

theorem quadratic_function_equal_values (a m n : ℝ) (h1 : a ≠ 0) (h2 : m ≠ n) :
  (a * m^2 - 4 * a * m - 3 = a * n^2 - 4 * a * n - 3) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_equal_values_l1068_106866


namespace NUMINAMATH_CALUDE_cricket_team_throwers_l1068_106854

theorem cricket_team_throwers (total_players : ℕ) (right_handed : ℕ) : 
  total_players = 61 → right_handed = 53 → ∃ (throwers : ℕ), 
    throwers = 37 ∧ 
    throwers ≤ right_handed ∧
    throwers ≤ total_players ∧
    3 * (right_handed - throwers) = 2 * (total_players - throwers) := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_throwers_l1068_106854


namespace NUMINAMATH_CALUDE_george_has_twelve_blocks_l1068_106871

/-- The number of blocks George has -/
def georgesBlocks (numBoxes : ℕ) (blocksPerBox : ℕ) : ℕ :=
  numBoxes * blocksPerBox

/-- Theorem: George has 12 blocks given 2 boxes with 6 blocks each -/
theorem george_has_twelve_blocks :
  georgesBlocks 2 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_george_has_twelve_blocks_l1068_106871


namespace NUMINAMATH_CALUDE_probability_same_team_l1068_106818

def q (a : ℕ) : ℚ :=
  (Nat.choose (a - 4) 2 + Nat.choose (52 - a) 2) / 1225

theorem probability_same_team (a : ℕ) (h : 4 ≤ a ∧ a ≤ 52) :
  q a = (Nat.choose (a - 4) 2 + Nat.choose (52 - a) 2) / 1225 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_team_l1068_106818


namespace NUMINAMATH_CALUDE_profit_percentage_l1068_106800

theorem profit_percentage (selling_price : ℝ) (cost_price : ℝ) (h : cost_price = 0.81 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (1 - 0.81) / 0.81 * 100 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_l1068_106800


namespace NUMINAMATH_CALUDE_a_plus_b_squared_l1068_106834

theorem a_plus_b_squared (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 2) (h3 : a < b) :
  (a + b)^2 = 1 ∨ (a + b)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_squared_l1068_106834


namespace NUMINAMATH_CALUDE_books_vs_figures_difference_l1068_106881

theorem books_vs_figures_difference :
  ∀ (initial_figures initial_books added_figures : ℕ),
    initial_figures = 2 →
    initial_books = 10 →
    added_figures = 4 →
    initial_books - (initial_figures + added_figures) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_books_vs_figures_difference_l1068_106881


namespace NUMINAMATH_CALUDE_range_of_m_l1068_106804

theorem range_of_m (m : ℝ) : 
  (m + 4)^(-1/2 : ℝ) < (3 - 2*m)^(-1/2 : ℝ) → 
  -1/3 < m ∧ m < 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1068_106804


namespace NUMINAMATH_CALUDE_pencil_count_l1068_106885

theorem pencil_count (num_boxes : ℝ) (pencils_per_box : ℝ) (h1 : num_boxes = 4.0) (h2 : pencils_per_box = 648.0) :
  num_boxes * pencils_per_box = 2592.0 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l1068_106885


namespace NUMINAMATH_CALUDE_monotonicity_condition_l1068_106844

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the monotonicity property on the interval (-∞, 8]
def is_monotonic_on_interval (k : ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 8 → f k x < f k y ∨ f k x > f k y

-- Theorem statement
theorem monotonicity_condition (k : ℝ) :
  is_monotonic_on_interval k ↔ k ≥ 64 := by
  sorry

end NUMINAMATH_CALUDE_monotonicity_condition_l1068_106844


namespace NUMINAMATH_CALUDE_student_survey_l1068_106894

theorem student_survey (french_and_english : ℕ) (french_not_english : ℕ) 
  (h1 : french_and_english = 20)
  (h2 : french_not_english = 60)
  (h3 : french_and_english + french_not_english = (2 : ℝ) / 5 * total_students) :
  total_students = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_student_survey_l1068_106894


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equal_sums_l1068_106898

/-- The sum of the first n terms of an arithmetic sequence with first term a and common difference d -/
def arithmeticSum (a d n : ℤ) : ℤ := n * (2 * a + (n - 1) * d) / 2

/-- The problem statement -/
theorem arithmetic_sequence_equal_sums (n : ℤ) : n ≠ 0 →
  (arithmeticSum 5 6 n = arithmeticSum 3 5 n) ↔ n = -3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equal_sums_l1068_106898


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l1068_106830

-- Define the inequality
def inequality (x a : ℝ) : Prop := (x + a) / (x^2 + 4*x + 3) > 0

-- Define the solution set
def solution_set (x : ℝ) : Prop := (-3 < x ∧ x < -1) ∨ x > 2

-- Theorem statement
theorem inequality_solution_implies_a_value :
  (∀ x : ℝ, inequality x a ↔ solution_set x) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l1068_106830


namespace NUMINAMATH_CALUDE_marble_probability_l1068_106816

def total_marbles : ℕ := 15
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def trials : ℕ := 6
def green_choices : ℕ := 3

def prob_green : ℚ := green_marbles / total_marbles
def prob_purple : ℚ := purple_marbles / total_marbles

theorem marble_probability : 
  (Nat.choose trials green_choices : ℚ) * 
  (prob_green ^ green_choices) * 
  (prob_purple ^ (trials - green_choices)) * 
  prob_purple = 4913248/34171875 := by sorry

end NUMINAMATH_CALUDE_marble_probability_l1068_106816


namespace NUMINAMATH_CALUDE_tangent_count_depends_on_position_l1068_106887

/-- Represents the position of a point relative to a circle -/
inductive PointPosition
  | OnCircle
  | OutsideCircle
  | InsideCircle

/-- Represents the number of tangents that can be drawn -/
inductive TangentCount
  | Zero
  | One
  | Two

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines the position of a point relative to a circle -/
def pointPosition (c : Circle) (p : ℝ × ℝ) : PointPosition :=
  sorry

/-- Counts the number of tangents that can be drawn from a point to a circle -/
def tangentCount (c : Circle) (p : ℝ × ℝ) : TangentCount :=
  sorry

/-- Theorem: The number of tangents depends on the point's position relative to the circle -/
theorem tangent_count_depends_on_position (c : Circle) (p : ℝ × ℝ) :
  (pointPosition c p = PointPosition.OnCircle → tangentCount c p = TangentCount.One) ∧
  (pointPosition c p = PointPosition.OutsideCircle → tangentCount c p = TangentCount.Two) ∧
  (pointPosition c p = PointPosition.InsideCircle → tangentCount c p = TangentCount.Zero) :=
  sorry

end NUMINAMATH_CALUDE_tangent_count_depends_on_position_l1068_106887


namespace NUMINAMATH_CALUDE_pies_per_row_l1068_106857

theorem pies_per_row (total_pies : ℕ) (num_rows : ℕ) (h1 : total_pies = 30) (h2 : num_rows = 6) :
  total_pies / num_rows = 5 := by
sorry

end NUMINAMATH_CALUDE_pies_per_row_l1068_106857


namespace NUMINAMATH_CALUDE_right_triangle_area_l1068_106876

theorem right_triangle_area (a c : ℝ) (h1 : a = 15) (h2 : c = 17) : ∃ b : ℝ, 
  a^2 + b^2 = c^2 ∧ (1/2) * a * b = 60 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1068_106876


namespace NUMINAMATH_CALUDE_f_g_deriv_signs_l1068_106860

-- Define f and g as real-valued functions
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_deriv_pos : ∀ x : ℝ, x > 0 → deriv f x > 0
axiom g_deriv_pos : ∀ x : ℝ, x > 0 → deriv g x > 0

-- State the theorem
theorem f_g_deriv_signs :
  ∀ x : ℝ, x < 0 → deriv f x > 0 ∧ deriv g x < 0 :=
sorry

end NUMINAMATH_CALUDE_f_g_deriv_signs_l1068_106860


namespace NUMINAMATH_CALUDE_power_of_product_with_negative_l1068_106821

theorem power_of_product_with_negative (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_with_negative_l1068_106821


namespace NUMINAMATH_CALUDE_percentage_spent_on_hats_l1068_106883

def total_money : ℕ := 90
def scarf_count : ℕ := 18
def scarf_price : ℕ := 2
def hat_to_scarf_ratio : ℕ := 2

theorem percentage_spent_on_hats :
  let money_spent_on_scarves := scarf_count * scarf_price
  let money_spent_on_hats := total_money - money_spent_on_scarves
  let percentage_on_hats := (money_spent_on_hats : ℚ) / total_money * 100
  percentage_on_hats = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_spent_on_hats_l1068_106883


namespace NUMINAMATH_CALUDE_inheritance_distribution_l1068_106892

structure Relative where
  name : String
  amount : ℕ

structure Couple where
  husband : Relative
  wife : Relative

def total_inheritance : ℕ := 1000
def wives_total : ℕ := 396

theorem inheritance_distribution (john henry tom : Relative) (katherine jane mary : Relative) :
  john.name = "John Smith" →
  henry.name = "Henry Snooks" →
  tom.name = "Tom Crow" →
  katherine.name = "Katherine" →
  jane.name = "Jane" →
  mary.name = "Mary" →
  jane.amount = katherine.amount + 10 →
  mary.amount = jane.amount + 10 →
  katherine.amount + jane.amount + mary.amount = wives_total →
  john.amount = katherine.amount →
  henry.amount = (3 * jane.amount) / 2 →
  tom.amount = 2 * mary.amount →
  john.amount + henry.amount + tom.amount + katherine.amount + jane.amount + mary.amount = total_inheritance →
  ∃ (c1 c2 c3 : Couple),
    c1.husband = john ∧ c1.wife = katherine ∧
    c2.husband = henry ∧ c2.wife = jane ∧
    c3.husband = tom ∧ c3.wife = mary :=
by
  sorry

end NUMINAMATH_CALUDE_inheritance_distribution_l1068_106892


namespace NUMINAMATH_CALUDE_equation_system_solutions_l1068_106839

theorem equation_system_solutions :
  ∀ (x y z : ℝ),
  (x = (2 * z^2) / (1 + z^2)) ∧
  (y = (2 * x^2) / (1 + x^2)) ∧
  (z = (2 * y^2) / (1 + y^2)) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solutions_l1068_106839


namespace NUMINAMATH_CALUDE_max_value_theorem_l1068_106890

theorem max_value_theorem (x y : ℝ) (h : x^2 - 3*x + 4*y = 7) :
  ∃ (M : ℝ), M = 16 ∧ ∀ (x' y' : ℝ), x'^2 - 3*x' + 4*y' = 7 → 3*x' + 4*y' ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1068_106890


namespace NUMINAMATH_CALUDE_flour_salt_difference_l1068_106851

theorem flour_salt_difference (total_flour sugar total_salt flour_added : ℕ) : 
  total_flour = 12 → 
  sugar = 14 →
  total_salt = 7 → 
  flour_added = 2 → 
  (total_flour - flour_added) - total_salt = 3 := by
sorry

end NUMINAMATH_CALUDE_flour_salt_difference_l1068_106851


namespace NUMINAMATH_CALUDE_square_root_of_four_l1068_106862

theorem square_root_of_four :
  {y : ℝ | y^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l1068_106862
