import Mathlib

namespace NUMINAMATH_CALUDE_nested_floor_equation_solution_l1613_161375

theorem nested_floor_equation_solution :
  ∃! x : ℝ, x * ⌊x * ⌊x * ⌊x * ⌊x⌋⌋⌋⌋ = 122 :=
by
  -- The unique solution is 122/41
  use 122/41
  sorry

end NUMINAMATH_CALUDE_nested_floor_equation_solution_l1613_161375


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1613_161328

theorem quadratic_inequality_equivalence (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x - 4*a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1613_161328


namespace NUMINAMATH_CALUDE_charity_book_donation_l1613_161305

theorem charity_book_donation (initial_books : ℕ) (donors : ℕ) (borrowed_books : ℕ) (final_books : ℕ)
  (h1 : initial_books = 300)
  (h2 : donors = 10)
  (h3 : borrowed_books = 140)
  (h4 : final_books = 210) :
  (final_books + borrowed_books - initial_books) / donors = 5 := by
  sorry

end NUMINAMATH_CALUDE_charity_book_donation_l1613_161305


namespace NUMINAMATH_CALUDE_f_properties_l1613_161388

/-- A function f from positive integers to positive integers with a parameter k -/
def f (k : ℕ+) : ℕ+ → ℕ+ :=
  fun n => if n > k then n - k else sorry

/-- The number of different functions f when k = 5 and 1 ≤ f(n) ≤ 2 for n ≤ 5 -/
def count_functions : ℕ := sorry

theorem f_properties :
  (∃ (a : ℕ+), f 1 1 = a) ∧
  count_functions = 32 := by sorry

end NUMINAMATH_CALUDE_f_properties_l1613_161388


namespace NUMINAMATH_CALUDE_triangle_inequality_l1613_161360

/-- For any triangle ABC and real numbers x, y, and z, 
    x^2 + y^2 + z^2 ≥ 2xy cos C + 2yz cos A + 2zx cos B -/
theorem triangle_inequality (A B C : ℝ) (x y z : ℝ) : 
  x^2 + y^2 + z^2 ≥ 2*x*y*(Real.cos C) + 2*y*z*(Real.cos A) + 2*z*x*(Real.cos B) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1613_161360


namespace NUMINAMATH_CALUDE_inequality_system_no_solution_l1613_161336

/-- The inequality system has no solution if and only if a ≥ -1 -/
theorem inequality_system_no_solution (a : ℝ) : 
  (∀ x : ℝ, ¬(x < a - 3 ∧ x + 2 > 2 * a)) ↔ a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_no_solution_l1613_161336


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1613_161399

/-- The function f(x) = (3/2)x^2 - 9x + 7 attains its minimum value when x = 3 -/
theorem min_value_quadratic (x : ℝ) : 
  (∀ y : ℝ, (3/2 : ℝ) * x^2 - 9*x + 7 ≤ (3/2 : ℝ) * y^2 - 9*y + 7) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1613_161399


namespace NUMINAMATH_CALUDE_sequence_general_term_l1613_161365

theorem sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ) (h : ∀ n, S n = 2 * n - a n) :
  ∀ n, a n = (2^n - 1) / 2^(n-1) := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1613_161365


namespace NUMINAMATH_CALUDE_irreducible_polynomial_l1613_161306

/-- A polynomial of the form x^n + 5x^(n-1) + 3 is irreducible over ℤ[X] for any integer n > 1 -/
theorem irreducible_polynomial (n : ℕ) (hn : n > 1) :
  Irreducible (Polynomial.monomial n 1 + Polynomial.monomial (n-1) 5 + Polynomial.monomial 0 3 : Polynomial ℤ) := by
  sorry

end NUMINAMATH_CALUDE_irreducible_polynomial_l1613_161306


namespace NUMINAMATH_CALUDE_refrigerator_discount_proof_l1613_161354

/-- The original price of the refrigerator -/
def original_price : ℝ := 250.00

/-- The first discount rate -/
def first_discount : ℝ := 0.20

/-- The second discount rate -/
def second_discount : ℝ := 0.15

/-- The final price as a percentage of the original price -/
def final_percentage : ℝ := 0.68

theorem refrigerator_discount_proof :
  original_price * (1 - first_discount) * (1 - second_discount) = original_price * final_percentage :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_discount_proof_l1613_161354


namespace NUMINAMATH_CALUDE_nonzero_real_equation_solution_l1613_161397

theorem nonzero_real_equation_solution (x : ℝ) (h : x ≠ 0) :
  (9 * x)^18 = (18 * x)^9 ↔ x = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_real_equation_solution_l1613_161397


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1613_161358

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define the line l passing through the origin
def line_l (k x y : ℝ) : Prop := y = k * x

-- Define the trajectory Γ
def trajectory_Γ (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0 ∧ 3/2 < x ∧ x ≤ 2

-- Define the line m
def line_m (a x y : ℝ) : Prop := y = a * x + 4

theorem circle_line_intersection
  (k : ℝ) -- Slope of line l
  (a : ℝ) -- Parameter for line m
  : 
  (∃ (x1 y1 x2 y2 : ℝ), 
    x1 ≠ x2 ∧
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    line_l k x1 y1 ∧ line_l k x2 y2) →
  (-Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3) ∧
  (∀ x y, trajectory_Γ x y ↔ 
    ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ 
    x = (x1 + x2) / 2 * (1 - t) + 2 * t ∧
    y = (y1 + y2) / 2 * (1 - t)) ∧
  ((∃! x y, trajectory_Γ x y ∧ line_m a x y) →
    (a = -15/8 ∨ (-Real.sqrt 3 - 8)/3 < a ∧ a ≤ (Real.sqrt 3 - 8)/3)) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1613_161358


namespace NUMINAMATH_CALUDE_skateboard_distance_l1613_161356

/-- The distance traveled by the skateboard in the nth second -/
def distance (n : ℕ) : ℕ := 8 + 9 * (n - 1)

/-- The total distance traveled by the skateboard after n seconds -/
def total_distance (n : ℕ) : ℕ := n * (distance 1 + distance n) / 2

theorem skateboard_distance :
  total_distance 20 = 1870 := by sorry

end NUMINAMATH_CALUDE_skateboard_distance_l1613_161356


namespace NUMINAMATH_CALUDE_third_number_in_ratio_l1613_161310

theorem third_number_in_ratio (a b c : ℝ) : 
  a / 5 = b / 6 ∧ b / 6 = c / 8 ∧  -- numbers are in ratio 5 : 6 : 8
  a + c = b + 49 →                -- sum of longest and smallest equals sum of third and 49
  b = 42 :=                       -- prove that the third number (b) is 42
by sorry

end NUMINAMATH_CALUDE_third_number_in_ratio_l1613_161310


namespace NUMINAMATH_CALUDE_sin_two_alpha_value_l1613_161344

theorem sin_two_alpha_value (α : Real) (h : Real.sin α - Real.cos α = 4/3) :
  Real.sin (2 * α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_value_l1613_161344


namespace NUMINAMATH_CALUDE_incorrect_equation_simplification_l1613_161357

theorem incorrect_equation_simplification (x : ℝ) : 
  (1 / (x + 1) = 2 * x / (3 * x + 3) - 1) ≠ (3 = 2 * x - 3 * x + 3) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_equation_simplification_l1613_161357


namespace NUMINAMATH_CALUDE_f_neg_one_lt_f_one_l1613_161315

/-- A function f: ℝ → ℝ that satisfies the given conditions -/
def f : ℝ → ℝ := sorry

/-- f is differentiable on ℝ -/
axiom f_differentiable : Differentiable ℝ f

/-- The functional equation for f -/
axiom f_eq (x : ℝ) : f x = x^2 + 2 * x * (deriv f 2)

/-- Theorem: f(-1) < f(1) -/
theorem f_neg_one_lt_f_one : f (-1) < f 1 := by sorry

end NUMINAMATH_CALUDE_f_neg_one_lt_f_one_l1613_161315


namespace NUMINAMATH_CALUDE_min_triangles_to_cover_l1613_161338

theorem min_triangles_to_cover (large_side : ℝ) (small_side : ℝ) : 
  large_side = 8 → small_side = 2 → 
  (large_side^2 / small_side^2 : ℝ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_triangles_to_cover_l1613_161338


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1613_161396

theorem min_value_of_expression (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  x*y/z + y*z/x + z*x/y ≥ Real.sqrt 3 ∧
  (x*y/z + y*z/x + z*x/y = Real.sqrt 3 ↔ x = y ∧ y = z ∧ z = Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1613_161396


namespace NUMINAMATH_CALUDE_correct_average_l1613_161302

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  incorrect_avg = 16 →
  incorrect_num = 25 →
  correct_num = 45 →
  (n : ℚ) * incorrect_avg = (n - 1 : ℚ) * incorrect_avg + incorrect_num →
  ((n : ℚ) * incorrect_avg - incorrect_num + correct_num) / n = 18 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l1613_161302


namespace NUMINAMATH_CALUDE_binomial_square_example_l1613_161364

theorem binomial_square_example : 16^2 + 2*(16*5) + 5^2 = 441 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_example_l1613_161364


namespace NUMINAMATH_CALUDE_prime_of_square_minus_one_l1613_161350

theorem prime_of_square_minus_one (a : ℕ) (h : a ≥ 2) :
  Nat.Prime (a^2 - 1) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_of_square_minus_one_l1613_161350


namespace NUMINAMATH_CALUDE_zoo_visitors_l1613_161352

theorem zoo_visitors (friday_visitors : ℕ) (sunday_visitors : ℕ) : 
  friday_visitors = 1250 →
  sunday_visitors = 500 →
  5250 = 3 * (friday_visitors + sunday_visitors) :=
by sorry

end NUMINAMATH_CALUDE_zoo_visitors_l1613_161352


namespace NUMINAMATH_CALUDE_max_xyz_value_l1613_161308

theorem max_xyz_value (x y z : ℝ) 
  (eq1 : x + x*y + x*y*z = 1)
  (eq2 : y + y*z + x*y*z = 2)
  (eq3 : z + x*z + x*y*z = 4) :
  x*y*z ≤ (5 + Real.sqrt 17) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_xyz_value_l1613_161308


namespace NUMINAMATH_CALUDE_store_sales_total_l1613_161334

/-- Represents the number of DVDs and CDs sold in a store in one day. -/
structure StoreSales where
  dvds : ℕ
  cds : ℕ

/-- Given a store that sells 1.6 times as many DVDs as CDs and sells 168 DVDs in one day,
    the total number of DVDs and CDs sold is 273. -/
theorem store_sales_total (s : StoreSales) 
    (h1 : s.dvds = 168)
    (h2 : s.dvds = (1.6 : ℝ) * s.cds) : 
    s.dvds + s.cds = 273 := by
  sorry

end NUMINAMATH_CALUDE_store_sales_total_l1613_161334


namespace NUMINAMATH_CALUDE_chameleon_color_impossibility_l1613_161339

/-- Represents the state of chameleons on the island -/
structure ChameleonSystem :=
  (num_chameleons : Nat)
  (num_colors : Nat)
  (color_change : Nat → Nat → Nat)  -- Function representing color change

/-- Represents the property that all chameleons have been all colors -/
def all_chameleons_all_colors (system : ChameleonSystem) : Prop :=
  ∀ c : Nat, c < system.num_chameleons → 
    ∃ t1 t2 t3 : Nat, 
      system.color_change c t1 = 0 ∧ 
      system.color_change c t2 = 1 ∧ 
      system.color_change c t3 = 2

theorem chameleon_color_impossibility :
  ∀ system : ChameleonSystem, 
    system.num_chameleons = 35 → 
    system.num_colors = 3 → 
    ¬(all_chameleons_all_colors system) := by
  sorry

end NUMINAMATH_CALUDE_chameleon_color_impossibility_l1613_161339


namespace NUMINAMATH_CALUDE_green_pill_cost_l1613_161323

theorem green_pill_cost (daily_green : ℕ) (daily_pink : ℕ) (days : ℕ) 
  (green_pink_diff : ℚ) (total_cost : ℚ) :
  daily_green = 2 →
  daily_pink = 1 →
  days = 21 →
  green_pink_diff = 1 →
  total_cost = 819 →
  ∃ (green_cost : ℚ), 
    green_cost = 40 / 3 ∧ 
    (daily_green * green_cost + daily_pink * (green_cost - green_pink_diff)) * days = total_cost :=
by sorry

end NUMINAMATH_CALUDE_green_pill_cost_l1613_161323


namespace NUMINAMATH_CALUDE_tan_theta_value_l1613_161394

theorem tan_theta_value (θ : Real) : 
  (Real.sin (π - θ) + Real.cos (θ - 2*π)) / (Real.sin θ + Real.cos (π + θ)) = 1/2 → 
  Real.tan θ = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1613_161394


namespace NUMINAMATH_CALUDE_max_value_fraction_l1613_161335

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (x * y) / (x + 8*y) ≤ 1/18 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1613_161335


namespace NUMINAMATH_CALUDE_max_value_x_plus_inverse_l1613_161304

theorem max_value_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 15 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_plus_inverse_l1613_161304


namespace NUMINAMATH_CALUDE_square_of_real_not_always_positive_l1613_161312

theorem square_of_real_not_always_positive : 
  ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_of_real_not_always_positive_l1613_161312


namespace NUMINAMATH_CALUDE_min_value_of_sum_min_value_reached_min_value_is_27_l1613_161379

theorem min_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 27) :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x * y * z = 27 → a + 3 * b + 9 * c ≤ x + 3 * y + 9 * z :=
by
  sorry

theorem min_value_reached (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 27) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 27 ∧ a + 3 * b + 9 * c = x + 3 * y + 9 * z :=
by
  sorry

theorem min_value_is_27 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_min_value_reached_min_value_is_27_l1613_161379


namespace NUMINAMATH_CALUDE_qualified_light_bulb_probability_l1613_161301

def market_probability (factory_A_share : ℝ) (factory_B_share : ℝ) 
                       (factory_A_qualification : ℝ) (factory_B_qualification : ℝ) : ℝ :=
  factory_A_share * factory_A_qualification + factory_B_share * factory_B_qualification

theorem qualified_light_bulb_probability :
  market_probability 0.7 0.3 0.9 0.8 = 0.87 := by
  sorry

end NUMINAMATH_CALUDE_qualified_light_bulb_probability_l1613_161301


namespace NUMINAMATH_CALUDE_triangle_side_length_l1613_161384

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (c^2 - a^2 = 5*b) → 
  (3 * Real.sin A * Real.cos C = Real.cos A * Real.sin C) → 
  b = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1613_161384


namespace NUMINAMATH_CALUDE_odd_power_congruence_l1613_161361

theorem odd_power_congruence (x : ℤ) (n : ℕ) (h_odd : Odd x) (h_n : n ≥ 1) :
  ∃ k : ℤ, x^(2^n) = 1 + k * 2^(n+2) := by
  sorry

end NUMINAMATH_CALUDE_odd_power_congruence_l1613_161361


namespace NUMINAMATH_CALUDE_area_equality_iff_concyclic_l1613_161316

-- Define the triangle ABC
variable (A B C : Point)

-- Define the altitudes and their intersection
variable (U V W H : Point)

-- Define points X, Y, Z on the altitudes
variable (X Y Z : Point)

-- Define the property of being an acute-angled triangle
def is_acute_angled (A B C : Point) : Prop := sorry

-- Define the property of a point being on a line segment
def on_segment (P Q R : Point) : Prop := sorry

-- Define the property of points being different
def are_different (P Q : Point) : Prop := sorry

-- Define the property of points being concyclic
def are_concyclic (P Q R S : Point) : Prop := sorry

-- Define the area of a triangle
def area (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem area_equality_iff_concyclic :
  is_acute_angled A B C →
  on_segment A U H → on_segment B V H → on_segment C W H →
  on_segment A U X → on_segment B V Y → on_segment C W Z →
  are_different X H → are_different Y H → are_different Z H →
  (are_concyclic X Y Z H ↔ area A B C = area A B Z + area A Y C + area X B C) :=
by sorry

end NUMINAMATH_CALUDE_area_equality_iff_concyclic_l1613_161316


namespace NUMINAMATH_CALUDE_incorrect_permutations_of_error_l1613_161395

def word : String := "error"

theorem incorrect_permutations_of_error (n : ℕ) :
  (n = word.length) →
  (n.choose 2 * 1 - 1 = 19) :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_permutations_of_error_l1613_161395


namespace NUMINAMATH_CALUDE_expression_evaluation_l1613_161332

theorem expression_evaluation : 3 * 3^4 - 9^20 / 9^18 + 5^3 = 287 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1613_161332


namespace NUMINAMATH_CALUDE_fraction_simplification_l1613_161366

theorem fraction_simplification :
  (3 / 7 + 5 / 8) / (5 / 12 + 1 / 4) = 177 / 112 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1613_161366


namespace NUMINAMATH_CALUDE_complement_of_35_degree_angle_l1613_161345

theorem complement_of_35_degree_angle (A : Real) : 
  A = 35 → 90 - A = 55 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_35_degree_angle_l1613_161345


namespace NUMINAMATH_CALUDE_ellipse_inequality_l1613_161329

theorem ellipse_inequality (a b x y : ℝ) (ha : a > 0) (hb : b > 0)
  (h : x^2 / a^2 + y^2 / b^2 ≤ 1) : a^2 + b^2 ≥ (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_inequality_l1613_161329


namespace NUMINAMATH_CALUDE_smallest_tree_height_l1613_161307

/-- Given three trees with specific height relationships, prove the height of the smallest tree -/
theorem smallest_tree_height (tallest middle smallest : ℝ) : 
  tallest = 108 →
  middle = tallest / 2 - 6 →
  smallest = middle / 4 →
  smallest = 12 := by sorry

end NUMINAMATH_CALUDE_smallest_tree_height_l1613_161307


namespace NUMINAMATH_CALUDE_inequality_proof_l1613_161325

theorem inequality_proof (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (h : 1/x + 1/y + 1/z = 1) : 
  a^x + b^y + c^z ≥ (4*a*b*c*x*y*z) / ((x+y+z-3)^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1613_161325


namespace NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l1613_161314

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The sum of the interior angles of a hexagon is 720° -/
theorem hexagon_interior_angles_sum :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l1613_161314


namespace NUMINAMATH_CALUDE_smallest_positive_solution_tan_equation_l1613_161386

theorem smallest_positive_solution_tan_equation :
  let x : ℝ := π / 26
  (∀ y : ℝ, y > 0 ∧ y < x → ¬(Real.tan (4 * y) + Real.tan (3 * y) = 1 / Real.cos (3 * y))) ∧
  (Real.tan (4 * x) + Real.tan (3 * x) = 1 / Real.cos (3 * x)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_tan_equation_l1613_161386


namespace NUMINAMATH_CALUDE_toy_cost_proof_l1613_161337

-- Define the number of toys
def num_toys : ℕ := 5

-- Define the discount rate (80% of original price)
def discount_rate : ℚ := 4/5

-- Define the total paid after discount
def total_paid : ℚ := 12

-- Define the cost per toy before discount
def cost_per_toy : ℚ := 3

-- Theorem statement
theorem toy_cost_proof :
  discount_rate * (num_toys : ℚ) * cost_per_toy = total_paid :=
sorry

end NUMINAMATH_CALUDE_toy_cost_proof_l1613_161337


namespace NUMINAMATH_CALUDE_two_books_total_cost_l1613_161383

/-- Proves that the total cost of two books is 420 given the specified conditions -/
theorem two_books_total_cost :
  ∀ (cost_loss cost_gain selling_price : ℝ),
  cost_loss = 245 →
  selling_price = cost_loss * 0.85 →
  selling_price = cost_gain * 1.19 →
  cost_loss + cost_gain = 420 :=
by
  sorry

end NUMINAMATH_CALUDE_two_books_total_cost_l1613_161383


namespace NUMINAMATH_CALUDE_line_is_integral_curve_no_inflection_points_l1613_161393

/-- Represents a function y(x) that satisfies the differential equation y' = 2x - y -/
def IntegralCurve (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv y) x = 2 * x - y x

/-- The line y = 2x - 2 is an integral curve of the differential equation y' = 2x - y -/
theorem line_is_integral_curve :
  IntegralCurve (λ x ↦ 2 * x - 2) := by sorry

/-- For any integral curve of y' = 2x - y, its second derivative is never zero -/
theorem no_inflection_points (y : ℝ → ℝ) (h : IntegralCurve y) :
  ∀ x, (deriv (deriv y)) x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_line_is_integral_curve_no_inflection_points_l1613_161393


namespace NUMINAMATH_CALUDE_max_value_fraction_l1613_161381

theorem max_value_fraction (x : ℝ) (h : x > 1) :
  (x^4 - x^2) / (x^6 + 2*x^3 - 1) ≤ 1/5 := by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1613_161381


namespace NUMINAMATH_CALUDE_parallel_iff_intersects_both_parallel_transitive_l1613_161377

-- Define the basic structures
structure Plane :=
(p : Type)

structure Line :=
(l : Type)

-- Define the relation for a line intersecting a plane at a single point
def intersects_at_single_point (l : Line) (α : Plane) : Prop :=
  ∃ (p : α.p), ∀ (q : α.p), l.l → (p = q)

-- Define the parallelism relation between planes
def parallel (α β : Plane) : Prop :=
  ∀ (l : Line), intersects_at_single_point l α → intersects_at_single_point l β

-- State the theorem
theorem parallel_iff_intersects_both (α β : Plane) :
  parallel α β ↔ ∀ (l : Line), intersects_at_single_point l α → intersects_at_single_point l β :=
sorry

-- State the transitivity of parallelism
theorem parallel_transitive (α β γ : Plane) :
  parallel α β → parallel β γ → parallel α γ :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_intersects_both_parallel_transitive_l1613_161377


namespace NUMINAMATH_CALUDE_faster_train_length_l1613_161319

/-- Given two trains moving in the same direction, this theorem calculates the length of the faster train. -/
theorem faster_train_length (v_fast v_slow : ℝ) (t_cross : ℝ) (h1 : v_fast = 72) (h2 : v_slow = 36) (h3 : t_cross = 18) :
  (v_fast - v_slow) * (5 / 18) * t_cross = 180 :=
by sorry

end NUMINAMATH_CALUDE_faster_train_length_l1613_161319


namespace NUMINAMATH_CALUDE_intersection_seq_100th_term_l1613_161348

def geometric_seq (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

def arithmetic_seq (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

def intersection_seq (n : ℕ) : ℝ := 2^(4 * n - 3)

theorem intersection_seq_100th_term :
  intersection_seq 100 = 2^397 :=
by sorry

end NUMINAMATH_CALUDE_intersection_seq_100th_term_l1613_161348


namespace NUMINAMATH_CALUDE_infinite_sum_equals_five_twentyfourths_l1613_161347

/-- The infinite sum of n / (n^4 - 4n^2 + 8) from n=1 to infinity equals 5/24 -/
theorem infinite_sum_equals_five_twentyfourths :
  ∑' n : ℕ+, (n : ℝ) / ((n : ℝ)^4 - 4*(n : ℝ)^2 + 8) = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_five_twentyfourths_l1613_161347


namespace NUMINAMATH_CALUDE_quadratic_completing_square_sum_l1613_161351

theorem quadratic_completing_square_sum (x q t : ℝ) : 
  (9 * x^2 - 54 * x - 36 = 0) →
  ((x + q)^2 = t) →
  (9 * (x + q)^2 = 9 * t) →
  (q + t = 10) := by
sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_sum_l1613_161351


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1613_161342

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^2 + 4 * x - 5) - (2 * x^2 - 3 * x + 8) = x^2 + 7 * x - 13 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1613_161342


namespace NUMINAMATH_CALUDE_regular_polygon_area_l1613_161387

theorem regular_polygon_area (n : ℕ) (R : ℝ) (h : n > 0) :
  (n * R^2 / 2) * (Real.sin (2 * Real.pi / n) + Real.cos (Real.pi / n)) = 4 * R^2 →
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_area_l1613_161387


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l1613_161374

theorem triangle_area_ratio (d : ℝ) : d > 0 →
  (1/2 * 3 * 6) = (1/4) * (1/2 * (d - 3) * (2*d - 6)) →
  d = 9 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l1613_161374


namespace NUMINAMATH_CALUDE_xyz_product_l1613_161378

/-- Given real numbers x, y, z, a, b, and c satisfying certain conditions,
    prove that their product xyz equals (a³ - 3ab² + 2c³) / 6 -/
theorem xyz_product (x y z a b c : ℝ) 
  (sum_eq : x + y + z = a)
  (sum_squares_eq : x^2 + y^2 + z^2 = b^2)
  (sum_cubes_eq : x^3 + y^3 + z^3 = c^3) :
  x * y * z = (a^3 - 3*a*b^2 + 2*c^3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l1613_161378


namespace NUMINAMATH_CALUDE_min_value_3x_plus_2y_min_value_attained_l1613_161317

theorem min_value_3x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x = 4 * x * y - 2 * y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a = 4 * a * b - 2 * b → 3 * x + 2 * y ≤ 3 * a + 2 * b :=
by sorry

theorem min_value_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x = 4 * x * y - 2 * y) :
  3 * x + 2 * y = 2 + Real.sqrt 3 ↔ x = (3 + Real.sqrt 3) / 6 ∧ y = (Real.sqrt 3 + 1) / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_2y_min_value_attained_l1613_161317


namespace NUMINAMATH_CALUDE_russian_doll_price_l1613_161392

theorem russian_doll_price (original_quantity : ℕ) (discounted_quantity : ℕ) (discounted_price : ℚ) :
  original_quantity = 15 →
  discounted_quantity = 20 →
  discounted_price = 3 →
  (discounted_quantity * discounted_price) / original_quantity = 4 := by
  sorry

end NUMINAMATH_CALUDE_russian_doll_price_l1613_161392


namespace NUMINAMATH_CALUDE_vector_angle_theorem_l1613_161391

/-- Given two vectors in 2D space, if the angle between them is 5π/6 and the magnitude of one vector
    equals the magnitude of their sum, then the angle between that vector and their sum is 2π/3. -/
theorem vector_angle_theorem (a b : ℝ × ℝ) :
  let angle_between := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  let magnitude (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
  angle_between = 5 * Real.pi / 6 ∧ magnitude a = magnitude (a.1 + b.1, a.2 + b.2) →
  Real.arccos ((a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2)) /
    (magnitude a * magnitude (a.1 + b.1, a.2 + b.2))) = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_theorem_l1613_161391


namespace NUMINAMATH_CALUDE_solve_for_a_l1613_161313

theorem solve_for_a : ∀ (x a : ℝ), (3 * x - 5 = x + a) ∧ (x = 2) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1613_161313


namespace NUMINAMATH_CALUDE_roberts_reading_capacity_l1613_161367

def reading_speed : ℝ := 75
def book_length : ℝ := 300
def available_time : ℝ := 9

theorem roberts_reading_capacity :
  ⌊available_time / (book_length / reading_speed)⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_roberts_reading_capacity_l1613_161367


namespace NUMINAMATH_CALUDE_square_difference_equals_690_l1613_161343

theorem square_difference_equals_690 : (23 + 15)^2 - (23^2 + 15^2) = 690 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_690_l1613_161343


namespace NUMINAMATH_CALUDE_point_C_coordinates_l1613_161331

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-3, 1)
def B : ℝ × ℝ := (0, 5)

def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem point_C_coordinates :
  ∀ C : ℝ × ℝ,
    parallel (vector A C) (vector O A) →
    perpendicular (vector B C) (vector A B) →
    C = (12, -4) := by sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l1613_161331


namespace NUMINAMATH_CALUDE_dog_park_ratio_l1613_161355

theorem dog_park_ratio (total_dogs : ℕ) (spotted_dogs : ℕ) (pointy_ear_dogs : ℕ) :
  pointy_ear_dogs = total_dogs / 5 →
  pointy_ear_dogs = 6 →
  spotted_dogs = 15 →
  (spotted_dogs : ℚ) / total_dogs = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_dog_park_ratio_l1613_161355


namespace NUMINAMATH_CALUDE_average_of_data_l1613_161321

def data : List ℕ := [5, 6, 5, 6, 4, 4]

theorem average_of_data : (data.sum : ℚ) / data.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_data_l1613_161321


namespace NUMINAMATH_CALUDE_hotel_rooms_l1613_161324

/-- Proves that a hotel with the given conditions has 10 rooms -/
theorem hotel_rooms (R : ℕ) 
  (people_per_room : ℕ) 
  (towels_per_person : ℕ) 
  (total_towels : ℕ) 
  (h1 : people_per_room = 3) 
  (h2 : towels_per_person = 2) 
  (h3 : total_towels = 60) 
  (h4 : R * people_per_room * towels_per_person = total_towels) : 
  R = 10 := by
  sorry

#check hotel_rooms

end NUMINAMATH_CALUDE_hotel_rooms_l1613_161324


namespace NUMINAMATH_CALUDE_marbles_problem_l1613_161340

theorem marbles_problem (jinwoo seonghyeon cheolsu : ℕ) : 
  jinwoo = (2 * seonghyeon) / 3 →
  cheolsu = 72 →
  jinwoo + cheolsu = 2 * seonghyeon →
  jinwoo = 36 := by
  sorry

end NUMINAMATH_CALUDE_marbles_problem_l1613_161340


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_B_subset_A_iff_a_in_range_l1613_161311

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - a| ≤ 4}
def B : Set ℝ := {x : ℝ | (x - 2) * (x - 3) ≤ 0}

-- Theorem 1: A ∩ B = ∅ if and only if a ∈ (-∞, -2) ∪ (7, +∞)
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a < -2 ∨ a > 7 := by sorry

-- Theorem 2: B ⊆ A if and only if a ∈ [1, 6]
theorem B_subset_A_iff_a_in_range (a : ℝ) :
  B ⊆ A a ↔ 1 ≤ a ∧ a ≤ 6 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_B_subset_A_iff_a_in_range_l1613_161311


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1613_161330

/-- The sum of all sides of an equilateral triangle with side length 13/12 meters is 13/4 meters. -/
theorem equilateral_triangle_perimeter (side_length : ℚ) (h : side_length = 13 / 12) :
  3 * side_length = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1613_161330


namespace NUMINAMATH_CALUDE_common_solutions_iff_y_values_l1613_161341

theorem common_solutions_iff_y_values (x y : ℝ) : 
  (∃ x : ℝ, x^2 + y^2 - 16 = 0 ∧ x^2 - 3*y - 12 = 0) ↔ (y = -4 ∨ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_common_solutions_iff_y_values_l1613_161341


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1613_161327

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem quadratic_function_properties :
  (∀ x, f (x + 1) - f x = 2*x - 1) ∧
  f 0 = 2 ∧
  (∀ x ∈ Set.Icc (-2) 2, 1 ≤ f x ∧ f x ≤ 10) ∧
  (∀ t, 
    let min_value := 
      if t ≥ 1 then t^2 - 2*t + 2
      else if 0 < t ∧ t < 1 then 1
      else t^2 + 2*t + 1
    ∀ x ∈ Set.Icc t (t + 1), f x ≥ min_value) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1613_161327


namespace NUMINAMATH_CALUDE_stock_value_change_l1613_161380

theorem stock_value_change (initial_value : ℝ) (h : initial_value > 0) :
  let day1_value := initial_value * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.40)
  day2_value = initial_value * 1.05 := by
    sorry

end NUMINAMATH_CALUDE_stock_value_change_l1613_161380


namespace NUMINAMATH_CALUDE_average_difference_approx_l1613_161303

def total_students : ℕ := 180
def total_teachers : ℕ := 6
def class_enrollments : List ℕ := [80, 40, 40, 10, 5, 5]

def teacher_average (students : ℕ) (teachers : ℕ) (enrollments : List ℕ) : ℚ :=
  (enrollments.sum : ℚ) / teachers

def student_average (students : ℕ) (enrollments : List ℕ) : ℚ :=
  (enrollments.map (λ n => n * n)).sum / students

theorem average_difference_approx (ε : ℚ) (hε : ε > 0) :
  ∃ δ : ℚ, δ > 0 ∧ 
    |teacher_average total_students total_teachers class_enrollments - 
     student_average total_students class_enrollments + 24.17| < δ ∧ δ < ε :=
by sorry

end NUMINAMATH_CALUDE_average_difference_approx_l1613_161303


namespace NUMINAMATH_CALUDE_no_polynomial_satisfies_condition_l1613_161363

/-- A polynomial function of degree exactly 3 -/
def PolynomialDegree3 (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^3 + b * x + c

/-- The condition that f(x^2) = [f(x)]^2 = f(f(x)) -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x^2) = (f x)^2 ∧ f (x^2) = f (f x)

theorem no_polynomial_satisfies_condition :
  ¬∃ f : ℝ → ℝ, PolynomialDegree3 f ∧ SatisfiesCondition f :=
sorry

end NUMINAMATH_CALUDE_no_polynomial_satisfies_condition_l1613_161363


namespace NUMINAMATH_CALUDE_function_properties_l1613_161326

-- Define the function f from X to Y
variable {X Y : Type*}
variable (f : X → Y)

-- Theorem stating that none of the given statements are necessarily true for all functions
theorem function_properties :
  (∃ y : Y, ∀ x : X, f x ≠ y) ∧  -- Some elements in Y might not have a preimage in X
  (∃ x₁ x₂ : X, x₁ ≠ x₂ ∧ f x₁ = f x₂) ∧  -- Different elements in X can have the same image in Y
  (∃ y : Y, True)  -- Y is not empty
  :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1613_161326


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1613_161371

theorem inequality_solution_set (x : ℝ) : 
  (2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5) ↔ (5 / 2 < x ∧ x ≤ 14 / 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1613_161371


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l1613_161390

/-- The curve defined by the polar equation r = 1 / (2sin(θ) - cos(θ)) is a line. -/
theorem polar_to_cartesian_line :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  ∀ (θ : ℝ), ∀ (r : ℝ), r > 0 →
  r = 1 / (2 * Real.sin θ - Real.cos θ) →
  a * (r * Real.cos θ) + b * (r * Real.sin θ) + c = 0 :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_line_l1613_161390


namespace NUMINAMATH_CALUDE_justin_tim_emily_games_l1613_161346

/-- The total number of players in the four-square league -/
def total_players : ℕ := 12

/-- The number of players in the larger game -/
def larger_game_players : ℕ := 7

/-- The number of specific players (Justin, Tim, and Emily) -/
def specific_players : ℕ := 3

theorem justin_tim_emily_games (h : total_players = 12 ∧ larger_game_players = 7 ∧ specific_players = 3) :
  Nat.choose (total_players - specific_players) (larger_game_players - specific_players) = 126 := by
  sorry

end NUMINAMATH_CALUDE_justin_tim_emily_games_l1613_161346


namespace NUMINAMATH_CALUDE_sum_of_odd_and_multiples_of_three_l1613_161398

/-- The number of six-digit odd numbers -/
def A : ℕ := 450000

/-- The number of six-digit multiples of 3 -/
def B : ℕ := 300000

/-- The sum of six-digit odd numbers and six-digit multiples of 3 is 750000 -/
theorem sum_of_odd_and_multiples_of_three : A + B = 750000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_and_multiples_of_three_l1613_161398


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1613_161300

/-- Given a geometric sequence {a_n} with common ratio q ≠ 1, where a₁a₂a₃ = -1/8
    and a₂, a₄, a₃ form an arithmetic sequence, the sum of the first 4 terms
    of the sequence {a_n} is equal to 5/8. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n : ℕ, a (n + 1) = a n * q) →
  a 1 * a 2 * a 3 = -1/8 →
  2 * a 4 = a 2 + a 3 →
  (a 1 + a 2 + a 3 + a 4 : ℝ) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1613_161300


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1613_161370

theorem min_value_of_expression (x y : ℝ) 
  (hx : |x| ≤ 1) (hy : |y| ≤ 1) : 
  ∃ (min_val : ℝ), min_val = 3 ∧ 
  ∀ (a b : ℝ), |a| ≤ 1 → |b| ≤ 1 → |b + 1| + |2*b - a - 4| ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1613_161370


namespace NUMINAMATH_CALUDE_prob_at_least_two_fruits_l1613_161349

/-- The probability of choosing a specific fruit at a meal -/
def prob_single_fruit : ℚ := 1 / 3

/-- The number of meals in a day -/
def num_meals : ℕ := 4

/-- The probability of choosing the same fruit for all meals -/
def prob_same_fruit : ℚ := prob_single_fruit ^ num_meals

/-- The number of fruit types -/
def num_fruit_types : ℕ := 3

theorem prob_at_least_two_fruits : 
  1 - (num_fruit_types : ℚ) * prob_same_fruit = 26 / 27 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_fruits_l1613_161349


namespace NUMINAMATH_CALUDE_max_volume_cross_section_area_l1613_161353

/-- Sphere with radius 2 -/
def Sphere : Type := Unit

/-- Points on the surface of the sphere -/
def Point : Type := Unit

/-- Angle between two points and the center of the sphere -/
def angle (p q : Point) : ℝ := sorry

/-- Volume of the triangular pyramid formed by three points and the center of the sphere -/
def pyramidVolume (a b c : Point) : ℝ := sorry

/-- Area of the circular cross-section formed by a plane through three points on the sphere -/
def crossSectionArea (a b c : Point) : ℝ := sorry

/-- The theorem statement -/
theorem max_volume_cross_section_area (o : Sphere) (a b c : Point) :
  (∀ (p q : Point), angle p q = angle a b) →
  (∀ (p q r : Point), pyramidVolume p q r ≤ pyramidVolume a b c) →
  crossSectionArea a b c = 8 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_max_volume_cross_section_area_l1613_161353


namespace NUMINAMATH_CALUDE_star_two_four_star_neg_three_x_l1613_161373

-- Define the new operation ※
def star (a b : ℝ) : ℝ := a^2 + 2*a*b

-- Theorem 1
theorem star_two_four : star 2 4 = 20 := by sorry

-- Theorem 2
theorem star_neg_three_x (x : ℝ) : star (-3) x = -3 + x → x = 12/7 := by sorry

end NUMINAMATH_CALUDE_star_two_four_star_neg_three_x_l1613_161373


namespace NUMINAMATH_CALUDE_max_candy_leftover_l1613_161320

theorem max_candy_leftover (x : ℕ) : ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l1613_161320


namespace NUMINAMATH_CALUDE_set_operations_l1613_161309

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- State the theorem
theorem set_operations :
  (Set.univ \ A = {x | x ≥ 3 ∨ x ≤ -2}) ∧
  (A ∩ B = {x | -2 < x ∧ x < 3}) ∧
  (Set.univ \ (A ∩ B) = {x | x ≥ 3 ∨ x ≤ -2}) ∧
  ((Set.univ \ A) ∩ B = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1613_161309


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l1613_161382

/-- Calculates the final price percentage after three consecutive price reductions -/
theorem price_reduction_theorem (initial_price : ℝ) 
  (reduction1 reduction2 reduction3 : ℝ) 
  (h1 : reduction1 = 0.09)
  (h2 : reduction2 = 0.10)
  (h3 : reduction3 = 0.15) : 
  (initial_price * (1 - reduction1) * (1 - reduction2) * (1 - reduction3)) / initial_price = 0.69615 := by
  sorry

#check price_reduction_theorem

end NUMINAMATH_CALUDE_price_reduction_theorem_l1613_161382


namespace NUMINAMATH_CALUDE_sequence_problem_l1613_161333

theorem sequence_problem (m : ℕ+) (a : ℕ → ℝ) 
  (h0 : a 0 = 37)
  (h1 : a 1 = 72)
  (hm : a m = 0)
  (h_rec : ∀ k : ℕ, 1 ≤ k → k < m → a (k + 1) = a (k - 1) - 3 / a k) :
  m = 889 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1613_161333


namespace NUMINAMATH_CALUDE_negative_one_three_in_M_l1613_161376

def M : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, y = 2 * x ∧ p = (x - y, x + y)}

theorem negative_one_three_in_M : ((-1 : ℝ), (3 : ℝ)) ∈ M := by
  sorry

end NUMINAMATH_CALUDE_negative_one_three_in_M_l1613_161376


namespace NUMINAMATH_CALUDE_sum_of_first_cards_theorem_l1613_161389

/-- The sum of points of the first cards in card piles -/
def sum_of_first_cards (a b c d : ℕ) : ℕ :=
  b * (c + 1) + d - a

/-- Theorem stating the sum of points of the first cards in card piles -/
theorem sum_of_first_cards_theorem (a b c d : ℕ) :
  ∃ x : ℕ, x = sum_of_first_cards a b c d :=
by
  sorry

#check sum_of_first_cards_theorem

end NUMINAMATH_CALUDE_sum_of_first_cards_theorem_l1613_161389


namespace NUMINAMATH_CALUDE_route_down_length_is_18_l1613_161368

/-- A hiking trip up and down a mountain -/
structure HikingTrip where
  rate_up : ℝ
  time_up : ℝ
  rate_down_factor : ℝ
  time_down : ℝ

/-- The length of the route down the mountain -/
def route_down_length (trip : HikingTrip) : ℝ :=
  trip.rate_up * trip.rate_down_factor * trip.time_down

/-- Theorem stating the length of the route down the mountain -/
theorem route_down_length_is_18 (trip : HikingTrip) 
  (h1 : trip.time_up = trip.time_down)
  (h2 : trip.rate_down_factor = 1.5)
  (h3 : trip.rate_up = 6)
  (h4 : trip.time_up = 2) : 
  route_down_length trip = 18 := by
  sorry

#eval route_down_length ⟨6, 2, 1.5, 2⟩

end NUMINAMATH_CALUDE_route_down_length_is_18_l1613_161368


namespace NUMINAMATH_CALUDE_positive_A_value_l1613_161385

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- State the theorem
theorem positive_A_value (A : ℝ) (h : hash A 3 = 130) : A = 11 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l1613_161385


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l1613_161318

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 12 * X^3 + 24 * X^2 - 10 * X + 5
  let divisor : Polynomial ℚ := 3 * X + 4
  let quotient : Polynomial ℚ := 4 * X^2 - 22/3
  dividend = divisor * quotient + (Polynomial.C (-197/9) : Polynomial ℚ) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l1613_161318


namespace NUMINAMATH_CALUDE_max_objective_value_l1613_161362

/-- The system of inequalities and objective function --/
def LinearProgram (x y : ℝ) : Prop :=
  x + 7 * y ≤ 32 ∧
  2 * x + 5 * y ≤ 42 ∧
  3 * x + 4 * y ≤ 62 ∧
  2 * x + y = 34 ∧
  x ≥ 0 ∧ y ≥ 0

/-- The objective function --/
def ObjectiveFunction (x y : ℝ) : ℝ :=
  3 * x + 8 * y

/-- The theorem stating the maximum value of the objective function --/
theorem max_objective_value :
  ∃ (x y : ℝ), LinearProgram x y ∧
  ∀ (x' y' : ℝ), LinearProgram x' y' →
  ObjectiveFunction x y ≥ ObjectiveFunction x' y' ∧
  ObjectiveFunction x y = 64 :=
sorry

end NUMINAMATH_CALUDE_max_objective_value_l1613_161362


namespace NUMINAMATH_CALUDE_nabla_computation_l1613_161359

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem nabla_computation : (nabla (nabla 2 3) 4) = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_computation_l1613_161359


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l1613_161372

/-- Given a line with equation x - y + 1 = 0, 
    the line symmetric to it with respect to the y-axis has equation x + y - 1 = 0 -/
theorem symmetric_line_wrt_y_axis : 
  ∀ (l : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ x - y + 1 = 0) →
    ∃ (l' : Set (ℝ × ℝ)), 
      (∀ (x y : ℝ), (x, y) ∈ l' ↔ x + y - 1 = 0) ∧
      (∀ (x y : ℝ), (x, y) ∈ l' ↔ (-x, y) ∈ l) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l1613_161372


namespace NUMINAMATH_CALUDE_product_first_fifth_l1613_161369

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  third_term : a 3 = 3
  sum_reciprocals : 1 / a 1 + 1 / a 5 = 6 / 5

/-- The product of the first and fifth terms of the arithmetic sequence is 5 -/
theorem product_first_fifth (seq : ArithmeticSequence) : seq.a 1 * seq.a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_first_fifth_l1613_161369


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1613_161322

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℝ) / (2 * Real.sqrt 7 + 3 * Real.sqrt 13) = (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -6 ∧ B = 7 ∧ C = -9 ∧ D = 13 ∧ E = 89 ∧
    Int.gcd (Int.gcd A C) E = 1 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1613_161322
