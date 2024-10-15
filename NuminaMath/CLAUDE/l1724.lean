import Mathlib

namespace NUMINAMATH_CALUDE_solve_linear_equation_l1724_172491

theorem solve_linear_equation :
  ∃ x : ℚ, -3 * x - 12 = 8 * x + 5 ∧ x = -17 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1724_172491


namespace NUMINAMATH_CALUDE_integral_sqrt_x_2_minus_x_l1724_172469

theorem integral_sqrt_x_2_minus_x (x : ℝ) : ∫ x in (0:ℝ)..1, Real.sqrt (x * (2 - x)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_x_2_minus_x_l1724_172469


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1724_172427

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that in an arithmetic sequence {aₙ} where
    a₅ + a₆ = 16 and a₈ = 12, the third term a₃ equals 4. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 5 + a 6 = 16)
    (h_eighth : a 8 = 12) : 
  a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1724_172427


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_15_l1724_172448

/-- Represents a triangle divided into four smaller triangles and a quadrilateral -/
structure DividedTriangle where
  total_area : ℝ
  triangle1_area : ℝ
  triangle2_area : ℝ
  triangle3_area : ℝ
  triangle4_area : ℝ
  quadrilateral_area : ℝ
  area_sum : total_area = triangle1_area + triangle2_area + triangle3_area + triangle4_area + quadrilateral_area

/-- Theorem stating that if the areas of the four triangles are 5, 10, 10, and 8, 
    then the area of the quadrilateral is 15 -/
theorem quadrilateral_area_is_15 (t : DividedTriangle) 
    (h1 : t.triangle1_area = 5)
    (h2 : t.triangle2_area = 10)
    (h3 : t.triangle3_area = 10)
    (h4 : t.triangle4_area = 8) :
    t.quadrilateral_area = 15 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_is_15_l1724_172448


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1724_172419

theorem quadratic_equal_roots : ∃ x : ℝ, 4 * x^2 - 4 * x + 1 = 0 ∧
  ∀ y : ℝ, 4 * y^2 - 4 * y + 1 = 0 → y = x := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1724_172419


namespace NUMINAMATH_CALUDE_fence_poles_for_given_plot_l1724_172411

/-- Calculates the number of fence poles needed to enclose a rectangular plot -/
def fence_poles (length width pole_distance : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  (perimeter + pole_distance - 1) / pole_distance

/-- Theorem stating the number of fence poles needed for the given plot -/
theorem fence_poles_for_given_plot :
  fence_poles 250 150 7 = 115 := by
  sorry

end NUMINAMATH_CALUDE_fence_poles_for_given_plot_l1724_172411


namespace NUMINAMATH_CALUDE_inequality_problem_l1724_172453

theorem inequality_problem (r p q : ℝ) 
  (hr : r < 0) 
  (hpq : p * q ≠ 0) 
  (hineq : p^2 * r > q^2 * r) : 
  ¬((-p > -q) ∧ (-p < q) ∧ (1 < -q/p) ∧ (1 > q/p)) :=
sorry

end NUMINAMATH_CALUDE_inequality_problem_l1724_172453


namespace NUMINAMATH_CALUDE_class_test_theorem_l1724_172429

/-- A theorem about a class test where some students didn't take the test -/
theorem class_test_theorem 
  (total_students : ℕ) 
  (answered_q2 : ℕ) 
  (did_not_take : ℕ) 
  (answered_both : ℕ) 
  (h1 : total_students = 30)
  (h2 : answered_q2 = 22)
  (h3 : did_not_take = 5)
  (h4 : answered_both = 22)
  (h5 : answered_both ≤ answered_q2)
  (h6 : did_not_take + answered_q2 ≤ total_students) :
  ∃ (answered_q1 : ℕ), answered_q1 = answered_both ∧ 
    answered_q1 + (answered_q2 - answered_both) + did_not_take ≤ total_students :=
by
  sorry

end NUMINAMATH_CALUDE_class_test_theorem_l1724_172429


namespace NUMINAMATH_CALUDE_proposition_relationship_l1724_172494

theorem proposition_relationship :
  (∀ x : ℝ, (0 < x ∧ x < 5) → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_relationship_l1724_172494


namespace NUMINAMATH_CALUDE_min_expense_is_2200_l1724_172472

/-- Represents the types of trucks available --/
inductive TruckType
| A
| B

/-- Represents the characteristics of a truck type --/
structure TruckInfo where
  cost : ℕ
  capacity : ℕ

/-- The problem setup --/
def problem_setup : (TruckType → TruckInfo) × ℕ × ℕ × ℕ :=
  (λ t => match t with
    | TruckType.A => ⟨400, 20⟩
    | TruckType.B => ⟨300, 10⟩,
   4,  -- number of Type A trucks
   8,  -- number of Type B trucks
   100) -- total air conditioners to transport

/-- Calculate the minimum transportation expense --/
def min_transportation_expense (setup : (TruckType → TruckInfo) × ℕ × ℕ × ℕ) : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem min_expense_is_2200 :
  min_transportation_expense problem_setup = 2200 :=
sorry

end NUMINAMATH_CALUDE_min_expense_is_2200_l1724_172472


namespace NUMINAMATH_CALUDE_shelly_thread_calculation_l1724_172454

/-- The number of friends Shelly made in classes -/
def class_friends : ℕ := 10

/-- The number of friends Shelly made from after-school clubs -/
def club_friends : ℕ := 2 * class_friends

/-- The amount of thread needed for each keychain for class friends (in inches) -/
def class_thread_per_keychain : ℕ := 16

/-- The amount of thread needed for each keychain for after-school club friends (in inches) -/
def club_thread_per_keychain : ℕ := 20

/-- The total amount of thread Shelly needs (in inches) -/
def total_thread_needed : ℕ := class_friends * class_thread_per_keychain + club_friends * club_thread_per_keychain

theorem shelly_thread_calculation :
  total_thread_needed = 560 := by
  sorry

end NUMINAMATH_CALUDE_shelly_thread_calculation_l1724_172454


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1724_172402

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {y | ∃ x, y = -x^2 + 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1724_172402


namespace NUMINAMATH_CALUDE_ellipse_distance_to_y_axis_l1724_172417

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the foci
def foci (f : ℝ) : Prop := f^2 = 3

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse x y

-- Define the perpendicularity condition
def perpendicular_vectors (x y f : ℝ) : Prop :=
  (x + f) * (x - f) + y * y = 0

-- Theorem statement
theorem ellipse_distance_to_y_axis 
  (x y f : ℝ) 
  (h1 : ellipse x y) 
  (h2 : foci f) 
  (h3 : perpendicular_vectors x y f) : 
  x^2 = 8/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_distance_to_y_axis_l1724_172417


namespace NUMINAMATH_CALUDE_find_b_l1724_172442

theorem find_b (p q : ℝ → ℝ) (b : ℝ) 
  (h1 : ∀ x, p x = 2 * x - 7)
  (h2 : ∀ x, q x = 3 * x - b)
  (h3 : p (q 4) = 7) : 
  b = 5 := by sorry

end NUMINAMATH_CALUDE_find_b_l1724_172442


namespace NUMINAMATH_CALUDE_sum_and_difference_problem_l1724_172463

theorem sum_and_difference_problem (a b : ℤ) : 
  a + b = 56 → 
  a = b + 12 → 
  a = 22 → 
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_and_difference_problem_l1724_172463


namespace NUMINAMATH_CALUDE_book_price_increase_l1724_172433

theorem book_price_increase (original_price : ℝ) (h : original_price > 0) :
  let price_after_first_increase := original_price * 1.15
  let final_price := price_after_first_increase * 1.15
  (final_price - original_price) / original_price = 0.3225 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l1724_172433


namespace NUMINAMATH_CALUDE_trig_sum_problem_l1724_172425

theorem trig_sum_problem (α : Real) 
  (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin α * Real.cos α = -1/2) :
  1 / (1 + Real.sin α) + 1 / (1 + Real.cos α) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_problem_l1724_172425


namespace NUMINAMATH_CALUDE_bucket_sand_problem_l1724_172443

theorem bucket_sand_problem (capacity_A : ℝ) (initial_sand_A : ℝ) :
  capacity_A > 0 →
  initial_sand_A ≥ 0 →
  initial_sand_A ≤ capacity_A →
  let capacity_B := capacity_A / 2
  let sand_B := 3 / 8 * capacity_B
  let total_sand := initial_sand_A + sand_B
  total_sand = 0.4375 * capacity_A →
  initial_sand_A = 1 / 4 * capacity_A :=
by sorry

end NUMINAMATH_CALUDE_bucket_sand_problem_l1724_172443


namespace NUMINAMATH_CALUDE_coeff_x_squared_is_thirteen_l1724_172449

/-- The coefficient of x^2 in the expansion of (1-x)^3(2x^2+1)^5 -/
def coeff_x_squared : ℕ :=
  (Nat.choose 5 4) * 2 + 3 * (Nat.choose 5 5)

/-- Theorem stating that the coefficient of x^2 in the expansion of (1-x)^3(2x^2+1)^5 is 13 -/
theorem coeff_x_squared_is_thirteen : coeff_x_squared = 13 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x_squared_is_thirteen_l1724_172449


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1724_172445

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- State the theorem
theorem sufficient_but_not_necessary : 
  (p 2) ∧ (∃ a : ℝ, a ≠ 2 ∧ p a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1724_172445


namespace NUMINAMATH_CALUDE_series_sum_l1724_172422

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) equals -7/24 -/
theorem series_sum : ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3)) = -7/24 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l1724_172422


namespace NUMINAMATH_CALUDE_regular_tetrahedron_properties_regular_tetrahedron_all_properties_l1724_172428

/-- Definition of a regular tetrahedron -/
structure RegularTetrahedron where
  /-- All edges of the tetrahedron are equal -/
  edges_equal : Bool
  /-- All faces of the tetrahedron are congruent equilateral triangles -/
  faces_congruent : Bool
  /-- The angle between any two edges at the same vertex is equal -/
  vertex_angles_equal : Bool
  /-- The dihedral angle between any two adjacent faces is equal -/
  dihedral_angles_equal : Bool

/-- Theorem: Properties of a regular tetrahedron -/
theorem regular_tetrahedron_properties (t : RegularTetrahedron) : 
  t.edges_equal ∧ 
  t.faces_congruent ∧ 
  t.vertex_angles_equal ∧ 
  t.dihedral_angles_equal := by
  sorry

/-- Corollary: All three properties mentioned in the problem are true for a regular tetrahedron -/
theorem regular_tetrahedron_all_properties (t : RegularTetrahedron) :
  (t.edges_equal ∧ t.vertex_angles_equal) ∧
  (t.faces_congruent ∧ t.dihedral_angles_equal) ∧
  (t.faces_congruent ∧ t.vertex_angles_equal) := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_properties_regular_tetrahedron_all_properties_l1724_172428


namespace NUMINAMATH_CALUDE_division_theorem_l1724_172480

theorem division_theorem (b : ℕ) (hb : b ≠ 0) :
  ∀ n : ℕ, ∃! (q r : ℕ), r < b ∧ n = q * b + r :=
sorry

end NUMINAMATH_CALUDE_division_theorem_l1724_172480


namespace NUMINAMATH_CALUDE_curve_expression_bound_l1724_172485

theorem curve_expression_bound :
  ∀ x y : ℝ, x^2 + (y^2)/4 = 4 → 
  ∃ t : ℝ, x = 2*Real.cos t ∧ y = 4*Real.sin t ∧ 
  -4 ≤ Real.sqrt 3 * x + (1/2) * y ∧ Real.sqrt 3 * x + (1/2) * y ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_curve_expression_bound_l1724_172485


namespace NUMINAMATH_CALUDE_phi_bound_l1724_172424

def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def satisfies_functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f x + 1

def iterate (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

def phi (f : ℝ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  iterate f n x - x

theorem phi_bound (f : ℝ → ℝ) (n : ℕ) :
  is_non_decreasing f →
  satisfies_functional_equation f →
  ∀ x y, |phi f n x - phi f n y| < 1 := by
  sorry

end NUMINAMATH_CALUDE_phi_bound_l1724_172424


namespace NUMINAMATH_CALUDE_xiaos_speed_correct_l1724_172459

/-- Xiao Hu Ma's speed in meters per minute -/
def xiaos_speed : ℝ := 80

/-- Distance between Xiao Hu Ma's house and school in meters -/
def total_distance : ℝ := 1800

/-- Distance from the meeting point to school in meters -/
def remaining_distance : ℝ := 200

/-- Time difference between Xiao Hu Ma and his father starting in minutes -/
def time_difference : ℝ := 10

theorem xiaos_speed_correct :
  xiaos_speed * (total_distance - remaining_distance) / xiaos_speed -
  (total_distance - remaining_distance) / (2 * xiaos_speed) = time_difference := by
  sorry

end NUMINAMATH_CALUDE_xiaos_speed_correct_l1724_172459


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1724_172414

theorem polynomial_factorization (x y z : ℝ) :
  x * (y - z)^3 + y * (z - x)^3 + z * (x - y)^3 = (x - y) * (y - z) * (z - x) * (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1724_172414


namespace NUMINAMATH_CALUDE_range_of_m_l1724_172457

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 2 / x + 1 / y = 1) (h2 : ∀ x y, x > 0 → y > 0 → 2 / x + 1 / y = 1 → x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1724_172457


namespace NUMINAMATH_CALUDE_coin_toss_and_die_roll_probability_l1724_172436

/-- The probability of getting exactly three heads and one tail when tossing four coins -/
def prob_three_heads_one_tail : ℚ := 1 / 4

/-- The probability of rolling a number greater than 4 on a six-sided die -/
def prob_die_greater_than_four : ℚ := 1 / 3

/-- The number of coins tossed -/
def num_coins : ℕ := 4

/-- The number of sides on the die -/
def num_die_sides : ℕ := 6

theorem coin_toss_and_die_roll_probability :
  prob_three_heads_one_tail * prob_die_greater_than_four = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_coin_toss_and_die_roll_probability_l1724_172436


namespace NUMINAMATH_CALUDE_measuring_rod_with_rope_l1724_172476

theorem measuring_rod_with_rope (x y : ℝ) 
  (h1 : x - y = 5)
  (h2 : y - (1/2) * x = 5) : 
  x - y = 5 ∧ y - (1/2) * x = 5 := by
  sorry

end NUMINAMATH_CALUDE_measuring_rod_with_rope_l1724_172476


namespace NUMINAMATH_CALUDE_dave_spent_22_tickets_l1724_172421

def tickets_spent_on_beanie (initial_tickets : ℕ) (additional_tickets : ℕ) (remaining_tickets : ℕ) : ℕ :=
  initial_tickets + additional_tickets - remaining_tickets

theorem dave_spent_22_tickets : 
  tickets_spent_on_beanie 25 15 18 = 22 := by
  sorry

end NUMINAMATH_CALUDE_dave_spent_22_tickets_l1724_172421


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1724_172481

def A : Set ℝ := {2, 3, 4, 5, 6}
def B : Set ℝ := {x : ℝ | x^2 - 8*x + 12 ≥ 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1724_172481


namespace NUMINAMATH_CALUDE_skyscraper_arrangement_impossible_l1724_172473

/-- The number of cyclic permutations of n elements -/
def cyclic_permutations (n : ℕ) : ℕ := (n - 1).factorial

/-- The maximum number of regions that n lines can divide a plane into -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The number of lines connecting n points -/
def connecting_lines (n : ℕ) : ℕ := n.choose 2

theorem skyscraper_arrangement_impossible :
  let n := 7
  let permutations := cyclic_permutations n
  let lines := connecting_lines n
  let regions := max_regions lines
  regions < permutations := by sorry

end NUMINAMATH_CALUDE_skyscraper_arrangement_impossible_l1724_172473


namespace NUMINAMATH_CALUDE_set_equality_implies_m_equals_negative_one_l1724_172468

theorem set_equality_implies_m_equals_negative_one (m : ℝ) :
  let A : Set ℝ := {m, 2}
  let B : Set ℝ := {m^2 - 2, 2}
  A = B → m = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_m_equals_negative_one_l1724_172468


namespace NUMINAMATH_CALUDE_pen_purchasing_plans_l1724_172438

theorem pen_purchasing_plans :
  ∃! (solutions : List (ℕ × ℕ)), 
    solutions.length = 3 ∧
    (∀ (x y : ℕ), (x, y) ∈ solutions ↔ 
      x > 0 ∧ y > 0 ∧ 15 * x + 10 * y = 105) :=
by sorry

end NUMINAMATH_CALUDE_pen_purchasing_plans_l1724_172438


namespace NUMINAMATH_CALUDE_min_sum_abs_values_l1724_172434

theorem min_sum_abs_values (x : ℝ) :
  ∃ (m : ℝ), (∀ (y : ℝ), |y + 1| + |y + 2| + |y + 6| ≥ m) ∧
             (∃ (z : ℝ), |z + 1| + |z + 2| + |z + 6| = m) ∧
             (m = 5) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_abs_values_l1724_172434


namespace NUMINAMATH_CALUDE_playground_children_count_l1724_172474

theorem playground_children_count (boys girls : ℕ) 
  (h1 : boys = 40) 
  (h2 : girls = 77) : 
  boys + girls = 117 := by
sorry

end NUMINAMATH_CALUDE_playground_children_count_l1724_172474


namespace NUMINAMATH_CALUDE_circle_through_points_l1724_172490

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 2*y + 12 = 0

-- Define the points
def P : ℝ × ℝ := (2, 2)
def M : ℝ × ℝ := (5, 3)
def N : ℝ × ℝ := (3, -1)

-- Theorem statement
theorem circle_through_points :
  circle_equation P.1 P.2 ∧ circle_equation M.1 M.2 ∧ circle_equation N.1 N.2 :=
sorry

end NUMINAMATH_CALUDE_circle_through_points_l1724_172490


namespace NUMINAMATH_CALUDE_initial_shoe_collection_l1724_172464

theorem initial_shoe_collection (initial_collection : ℕ) : 
  (initial_collection : ℝ) * 0.7 + 6 = 62 → initial_collection = 80 :=
by sorry

end NUMINAMATH_CALUDE_initial_shoe_collection_l1724_172464


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1724_172404

theorem unique_integer_solution :
  ∃! (a b c : ℤ), a^2 + b^2 + c^2 + 3 < a*b + 3*b + 2*c ∧ a = 1 ∧ b = 2 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1724_172404


namespace NUMINAMATH_CALUDE_division_instead_of_multiplication_error_l1724_172452

theorem division_instead_of_multiplication_error (y : ℝ) (h : y > 0) :
  (|8 * y - y / 8| / (8 * y)) * 100 = 98 := by
  sorry

end NUMINAMATH_CALUDE_division_instead_of_multiplication_error_l1724_172452


namespace NUMINAMATH_CALUDE_not_divides_two_pow_minus_one_l1724_172470

theorem not_divides_two_pow_minus_one (n : ℕ) (hn : n > 1) : ¬(n ∣ 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_two_pow_minus_one_l1724_172470


namespace NUMINAMATH_CALUDE_min_value_theorem_l1724_172446

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  1/x + 8/(1 - 2*x) ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1724_172446


namespace NUMINAMATH_CALUDE_largest_prime_for_integer_sqrt_l1724_172423

theorem largest_prime_for_integer_sqrt : ∃ (p : ℕ), 
  Prime p ∧ 
  (∃ (q : ℕ), q^2 = 17*p + 625) ∧
  (∀ (p' : ℕ), Prime p' → (∃ (q' : ℕ), q'^2 = 17*p' + 625) → p' ≤ p) ∧
  p = 67 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_for_integer_sqrt_l1724_172423


namespace NUMINAMATH_CALUDE_find_vector_c_l1724_172415

/-- Given vectors a and b in ℝ², find vector c satisfying the given conditions -/
theorem find_vector_c (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (-3, 2)) : 
  ∃ c : ℝ × ℝ, 
    (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = 0) ∧ 
    (∃ k : ℝ, (c.1 - a.1, c.2 - a.2) = (k * b.1, k * b.2)) → 
    c = (7/3, 7/9) := by
  sorry

end NUMINAMATH_CALUDE_find_vector_c_l1724_172415


namespace NUMINAMATH_CALUDE_line_contains_point_l1724_172467

theorem line_contains_point (k : ℝ) : 
  (2 + 3 * k * (-1/3) = -4 * 1) → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l1724_172467


namespace NUMINAMATH_CALUDE_classification_theorem_l1724_172486

def expressions : List String := [
  "4xy", "m^2n/2", "y^2 + y + 2/y", "2x^3 - 3", "0", "-3/(ab) + a",
  "m", "(m-n)/(m+n)", "(x-1)/2", "3/x"
]

def is_monomial (expr : String) : Bool := sorry

def is_polynomial (expr : String) : Bool := sorry

theorem classification_theorem :
  let monomials := expressions.filter is_monomial
  let polynomials := expressions.filter (λ e => is_polynomial e ∧ ¬is_monomial e)
  let all_polynomials := expressions.filter is_polynomial
  (monomials = ["4xy", "m^2n/2", "0", "m"]) ∧
  (polynomials = ["2x^3 - 3", "(x-1)/2"]) ∧
  (all_polynomials = ["4xy", "m^2n/2", "2x^3 - 3", "0", "m", "(x-1)/2"]) := by
  sorry

end NUMINAMATH_CALUDE_classification_theorem_l1724_172486


namespace NUMINAMATH_CALUDE_line_points_product_l1724_172489

/-- Given a line k passing through the origin with slope √7 / 3,
    if points (x, 8) and (20, y) lie on this line, then x * y = 160. -/
theorem line_points_product (x y : ℝ) : 
  (∃ k : ℝ → ℝ, k 0 = 0 ∧ 
   (∀ x₁ x₂, x₁ ≠ x₂ → (k x₂ - k x₁) / (x₂ - x₁) = Real.sqrt 7 / 3) ∧
   k x = 8 ∧ k 20 = y) →
  x * y = 160 := by
sorry


end NUMINAMATH_CALUDE_line_points_product_l1724_172489


namespace NUMINAMATH_CALUDE_expression_simplification_l1724_172447

theorem expression_simplification (x : ℝ) (h : x = 1) :
  (2 * x) / (x + 2) - x / (x - 2) + (4 * x) / (x^2 - 4) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1724_172447


namespace NUMINAMATH_CALUDE_line_AB_equation_l1724_172441

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point P
def P : ℝ × ℝ := (3, 2)

-- Define that P is the midpoint of AB
def is_midpoint (A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem line_AB_equation (A B : ℝ × ℝ) :
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ is_midpoint A B →
  ∀ x y : ℝ, (y = x - 1) ↔ (y - A.2 = A.2 - A.1 * (x - A.1)) :=
by sorry

end NUMINAMATH_CALUDE_line_AB_equation_l1724_172441


namespace NUMINAMATH_CALUDE_intersection_sum_l1724_172458

/-- Two lines intersect at a point if the point satisfies both line equations -/
def intersect_at (x y a b : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 = (1/3) * p.2 + a ∧ p.2 = (1/3) * p.1 + b

/-- The problem statement -/
theorem intersection_sum (a b : ℝ) :
  intersect_at 3 1 a b (3, 1) → a + b = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l1724_172458


namespace NUMINAMATH_CALUDE_distinct_collections_l1724_172478

/-- Represents the letter counts in CALCULATOR --/
structure LetterCounts where
  a : Nat
  c : Nat
  l : Nat
  other_vowels : Nat
  other_consonants : Nat

/-- Represents a selection of letters --/
structure Selection where
  a : Nat
  c : Nat
  l : Nat
  other_vowels : Nat
  other_consonants : Nat

/-- Checks if a selection is valid --/
def is_valid_selection (s : Selection) : Prop :=
  s.a + s.other_vowels = 3 ∧ 
  s.c + s.l + s.other_consonants = 6

/-- Counts distinct vowel selections --/
def count_vowel_selections (total : LetterCounts) : Nat :=
  3 -- This is a simplification based on the problem's specifics

/-- Counts distinct consonant selections --/
noncomputable def count_consonant_selections (total : LetterCounts) : Nat :=
  sorry -- This would be calculated based on the combinations in the solution

/-- The main theorem --/
theorem distinct_collections (total : LetterCounts) 
  (h1 : total.a = 2)
  (h2 : total.c = 2)
  (h3 : total.l = 2)
  (h4 : total.other_vowels = 2)
  (h5 : total.other_consonants = 2) :
  (count_vowel_selections total) * (count_consonant_selections total) = 
  3 * (count_consonant_selections total) := by
  sorry

#check distinct_collections

end NUMINAMATH_CALUDE_distinct_collections_l1724_172478


namespace NUMINAMATH_CALUDE_max_distinct_dance_counts_l1724_172416

/-- Represents the dance count for a person -/
def DanceCount := Nat

/-- Represents a set of distinct dance counts -/
def DistinctCounts := Finset DanceCount

theorem max_distinct_dance_counts 
  (num_boys : Nat) 
  (num_girls : Nat) 
  (h_boys : num_boys = 29) 
  (h_girls : num_girls = 15) :
  ∃ (dc : DistinctCounts), dc.card ≤ 29 ∧ 
  ∀ (dc' : DistinctCounts), dc'.card ≤ dc.card :=
sorry

end NUMINAMATH_CALUDE_max_distinct_dance_counts_l1724_172416


namespace NUMINAMATH_CALUDE_no_solution_equation1_unique_solution_equation2_l1724_172444

-- Define the first equation
def equation1 (x : ℝ) : Prop :=
  x ≠ 2 ∧ 3*x ≠ 6 ∧ (5*x - 4) / (x - 2) = (4*x + 10) / (3*x - 6) - 1

-- Define the second equation
def equation2 (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ -2 ∧ 1 - (x - 2) / (2 + x) = 16 / (x^2 - 4)

-- Theorem for the first equation
theorem no_solution_equation1 : ¬∃ x, equation1 x :=
  sorry

-- Theorem for the second equation
theorem unique_solution_equation2 : ∃! x, equation2 x ∧ x = 6 :=
  sorry

end NUMINAMATH_CALUDE_no_solution_equation1_unique_solution_equation2_l1724_172444


namespace NUMINAMATH_CALUDE_product_equals_one_l1724_172499

theorem product_equals_one (a b c : ℝ) 
  (h1 : a^2 + 2 = b^4) 
  (h2 : b^2 + 2 = c^4) 
  (h3 : c^2 + 2 = a^4) : 
  (a^2 - 1) * (b^2 - 1) * (c^2 - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_one_l1724_172499


namespace NUMINAMATH_CALUDE_max_digit_sum_for_reciprocal_decimal_l1724_172495

/-- Given digits a, b, c forming a decimal 0.abc that equals 1/y for some integer y between 1 and 12,
    the sum a + b + c is at most 8. -/
theorem max_digit_sum_for_reciprocal_decimal (a b c y : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, c are digits
  (0 < y ∧ y ≤ 12) →            -- 0 < y ≤ 12
  (a * 100 + b * 10 + c : ℚ) / 1000 = 1 / y →  -- 0.abc = 1/y
  a + b + c ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_max_digit_sum_for_reciprocal_decimal_l1724_172495


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_15_12_l1724_172410

theorem half_abs_diff_squares_15_12 : (1/2 : ℝ) * |15^2 - 12^2| = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_15_12_l1724_172410


namespace NUMINAMATH_CALUDE_vector_operations_and_parallel_condition_l1724_172400

def a : Fin 2 → ℝ := ![2, 0]
def b : Fin 2 → ℝ := ![1, 4]

theorem vector_operations_and_parallel_condition :
  (2 • a + 3 • b = ![7, 12]) ∧
  (a - 2 • b = ![0, -8]) ∧
  (∃ (k : ℝ), ∃ (t : ℝ), k • a + b = t • (a + 2 • b) → k = 1/2) := by sorry

end NUMINAMATH_CALUDE_vector_operations_and_parallel_condition_l1724_172400


namespace NUMINAMATH_CALUDE_rectangle_length_l1724_172455

/-- Given a rectangle with area 28 square centimeters and width 4 centimeters, its length is 7 centimeters. -/
theorem rectangle_length (area width : ℝ) (h_area : area = 28) (h_width : width = 4) :
  area / width = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l1724_172455


namespace NUMINAMATH_CALUDE_bicycle_costs_l1724_172413

theorem bicycle_costs (B H L : ℝ) 
  (total_cost : B + H + L = 480)
  (bicycle_helmet_ratio : B = 5 * H)
  (lock_helmet_ratio : L = 0.5 * H)
  (lock_total_ratio : L = 0.1 * 480) : 
  B = 360 ∧ H = 72 ∧ L = 48 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_costs_l1724_172413


namespace NUMINAMATH_CALUDE_yard_area_l1724_172498

theorem yard_area (fence_length : ℝ) (unfenced_side : ℝ) (h1 : fence_length = 64) (h2 : unfenced_side = 40) :
  ∃ (width : ℝ), 
    unfenced_side + 2 * width = fence_length ∧ 
    unfenced_side * width = 480 :=
by sorry

end NUMINAMATH_CALUDE_yard_area_l1724_172498


namespace NUMINAMATH_CALUDE_root_equation_value_l1724_172440

theorem root_equation_value (m : ℝ) : 
  m^2 - m - 2 = 0 → m^2 - m + 2023 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l1724_172440


namespace NUMINAMATH_CALUDE_jump_rope_competition_theorem_l1724_172484

/-- Represents a jump rope competition for a class of students. -/
structure JumpRopeCompetition where
  totalStudents : ℕ
  initialParticipants : ℕ
  initialAverage : ℕ
  lateStudentScores : List ℕ

/-- Calculates the new average score for the entire class after late students participate. -/
def newAverageScore (comp : JumpRopeCompetition) : ℚ :=
  let initialTotal := comp.initialParticipants * comp.initialAverage
  let lateTotal := comp.lateStudentScores.sum
  let totalJumps := initialTotal + lateTotal
  totalJumps / comp.totalStudents

/-- The main theorem stating that for the given competition parameters, 
    the new average score is 21. -/
theorem jump_rope_competition_theorem (comp : JumpRopeCompetition) 
  (h1 : comp.totalStudents = 30)
  (h2 : comp.initialParticipants = 26)
  (h3 : comp.initialAverage = 20)
  (h4 : comp.lateStudentScores = [26, 27, 28, 29]) :
  newAverageScore comp = 21 := by
  sorry

#eval newAverageScore {
  totalStudents := 30,
  initialParticipants := 26,
  initialAverage := 20,
  lateStudentScores := [26, 27, 28, 29]
}

end NUMINAMATH_CALUDE_jump_rope_competition_theorem_l1724_172484


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l1724_172497

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (-9/7, 20/7)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: -2y = 6x + 2 -/
def line2 (x y : ℚ) : Prop := -2 * y = 6 * x + 2

theorem intersection_point_is_unique :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = intersection_point :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l1724_172497


namespace NUMINAMATH_CALUDE_baker_pastry_cake_difference_l1724_172492

/-- The number of cakes made by the baker -/
def cakes_made : ℕ := 19

/-- The number of pastries made by the baker -/
def pastries_made : ℕ := 131

/-- The difference between pastries and cakes made by the baker -/
def pastry_cake_difference : ℕ := pastries_made - cakes_made

theorem baker_pastry_cake_difference :
  pastry_cake_difference = 112 := by sorry

end NUMINAMATH_CALUDE_baker_pastry_cake_difference_l1724_172492


namespace NUMINAMATH_CALUDE_boat_license_combinations_l1724_172432

def possible_letters : Nat := 3
def digits_per_license : Nat := 6
def possible_digits : Nat := 10

theorem boat_license_combinations :
  possible_letters * possible_digits ^ digits_per_license = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_combinations_l1724_172432


namespace NUMINAMATH_CALUDE_inheritance_division_l1724_172430

theorem inheritance_division (A B : ℝ) : 
  A + B = 100 ∧ 
  (1/4 : ℝ) * B - (1/3 : ℝ) * A = 11 →
  A = 24 ∧ B = 76 := by
sorry

end NUMINAMATH_CALUDE_inheritance_division_l1724_172430


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l1724_172466

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 → 
  Nat.gcd a b = 55 → 
  a = 210 → 
  b = 605 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l1724_172466


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_sum_binomial_coefficient_sum_equals_six_largest_n_is_six_l1724_172496

theorem largest_binomial_coefficient_sum (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

theorem binomial_coefficient_sum_equals_six : 
  Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 6 :=
by sorry

theorem largest_n_is_six : 
  ∃ (n : ℕ), n = 6 ∧ 
    Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n ∧
    ∀ (m : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_binomial_coefficient_sum_binomial_coefficient_sum_equals_six_largest_n_is_six_l1724_172496


namespace NUMINAMATH_CALUDE_shop_owner_profit_l1724_172483

/-- Calculates the percentage profit of a shop owner who cheats with weights -/
theorem shop_owner_profit (buying_cheat : ℝ) (selling_cheat : ℝ) : 
  buying_cheat = 0.14 →
  selling_cheat = 0.20 →
  (((1 + buying_cheat) / (1 - selling_cheat)) - 1) * 100 = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_shop_owner_profit_l1724_172483


namespace NUMINAMATH_CALUDE_four_machines_copies_l1724_172460

/-- Represents a copying machine with a specific rate --/
structure Machine where
  copies : ℕ
  minutes : ℕ

/-- Calculates the total number of copies produced by multiple machines in a given time --/
def totalCopies (machines : List Machine) (workTime : ℕ) : ℕ :=
  machines.foldl (fun acc m => acc + workTime * m.copies / m.minutes) 0

/-- Theorem stating the total number of copies produced by four specific machines in 40 minutes --/
theorem four_machines_copies : 
  let machineA : Machine := ⟨100, 8⟩
  let machineB : Machine := ⟨150, 10⟩
  let machineC : Machine := ⟨200, 12⟩
  let machineD : Machine := ⟨250, 15⟩
  let machines : List Machine := [machineA, machineB, machineC, machineD]
  totalCopies machines 40 = 2434 := by
  sorry

end NUMINAMATH_CALUDE_four_machines_copies_l1724_172460


namespace NUMINAMATH_CALUDE_boy_and_bus_speeds_l1724_172450

/-- Represents the problem of finding the speeds of a boy and a bus given certain conditions. -/
theorem boy_and_bus_speeds
  (total_distance : ℝ)
  (first_meeting_time : ℝ)
  (boy_additional_distance : ℝ)
  (stop_time : ℝ) :
  total_distance = 4.5 ∧
  first_meeting_time = 0.25 ∧
  boy_additional_distance = 9 / 28 ∧
  stop_time = 4 / 60 →
  ∃ (boy_speed bus_speed : ℝ),
    boy_speed = 3 ∧
    bus_speed = 45 ∧
    boy_speed > 0 ∧
    bus_speed > 0 ∧
    boy_speed * first_meeting_time + boy_additional_distance =
      bus_speed * first_meeting_time - total_distance ∧
    bus_speed * (first_meeting_time + 2 * stop_time) = 2 * total_distance :=
by sorry

end NUMINAMATH_CALUDE_boy_and_bus_speeds_l1724_172450


namespace NUMINAMATH_CALUDE_power_of_two_equals_quadratic_plus_linear_plus_one_l1724_172471

theorem power_of_two_equals_quadratic_plus_linear_plus_one
  (x y : ℕ) (h : 2^x = y^2 + y + 1) : x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equals_quadratic_plus_linear_plus_one_l1724_172471


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1724_172403

/-- An arithmetic sequence with the given property -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) :
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1724_172403


namespace NUMINAMATH_CALUDE_xiaoming_class_ratio_l1724_172462

theorem xiaoming_class_ratio (n : ℕ) (h1 : 30 < n) (h2 : n < 40) : ¬ ∃ k : ℕ, n = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_class_ratio_l1724_172462


namespace NUMINAMATH_CALUDE_shaded_area_is_32_5_l1724_172405

/-- Represents a rectangular grid -/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Represents a right-angled triangle -/
structure RightTriangle where
  base : ℕ
  height : ℕ

/-- Calculates the area of a shaded region in a grid, excluding a right-angled triangle -/
def shadedArea (g : Grid) (t : RightTriangle) : ℚ :=
  (g.rows * g.cols : ℚ) - (t.base * t.height : ℚ) / 2

/-- Theorem stating that the shaded area in the given problem is 32.5 square units -/
theorem shaded_area_is_32_5 :
  let g : Grid := ⟨4, 13⟩
  let t : RightTriangle := ⟨13, 3⟩
  shadedArea g t = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_32_5_l1724_172405


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1724_172487

theorem quadratic_equation_roots (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 8 * x + 10 = 0 ↔ x = 2 ∨ x = -5/3) → k = 24 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1724_172487


namespace NUMINAMATH_CALUDE_modulus_of_3_minus_4i_l1724_172488

theorem modulus_of_3_minus_4i : Complex.abs (3 - 4 * Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_3_minus_4i_l1724_172488


namespace NUMINAMATH_CALUDE_equation_solution_l1724_172493

theorem equation_solution : 
  ∀ x : ℝ, (Real.sqrt (4 * x + 10) / Real.sqrt (8 * x + 2) = 2 / Real.sqrt 5) → x = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1724_172493


namespace NUMINAMATH_CALUDE_x_minus_y_equals_fourteen_l1724_172420

theorem x_minus_y_equals_fourteen (x y : ℝ) (h : x^2 + y^2 = 16*x - 12*y + 100) : x - y = 14 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_fourteen_l1724_172420


namespace NUMINAMATH_CALUDE_cryptarithm_solutions_l1724_172482

def is_valid_solution (tuk : ℕ) (ctuk : ℕ) : Prop :=
  tuk ≥ 100 ∧ tuk < 1000 ∧ ctuk ≥ 1000 ∧ ctuk < 10000 ∧
  5 * tuk = ctuk ∧
  (tuk.digits 10).card = 3 ∧ (ctuk.digits 10).card = 4

theorem cryptarithm_solutions :
  (∀ tuk ctuk : ℕ, is_valid_solution tuk ctuk → (tuk = 250 ∧ ctuk = 1250) ∨ (tuk = 750 ∧ ctuk = 3750)) ∧
  is_valid_solution 250 1250 ∧
  is_valid_solution 750 3750 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solutions_l1724_172482


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1724_172409

theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1724_172409


namespace NUMINAMATH_CALUDE_union_of_sets_l1724_172407

theorem union_of_sets : 
  let M : Set ℕ := {2, 3, 5}
  let N : Set ℕ := {3, 4, 5}
  M ∪ N = {2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1724_172407


namespace NUMINAMATH_CALUDE_coefficient_equals_k_squared_minus_one_l1724_172456

theorem coefficient_equals_k_squared_minus_one (k : ℝ) (h1 : k > 0) :
  (∃ b : ℝ, (k * b^2 - b)^2 = k^2 * b^4 - 2 * k * b^3 + k^2 * b^2 - b^2) →
  k = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_coefficient_equals_k_squared_minus_one_l1724_172456


namespace NUMINAMATH_CALUDE_max_unused_cubes_l1724_172475

/-- The side length of the original cube in small cube units -/
def original_side_length : ℕ := 10

/-- The total number of small cubes in the original cube -/
def total_cubes : ℕ := original_side_length ^ 3

/-- The function that calculates the number of small cubes used in a hollow cube of side length x -/
def cubes_used (x : ℕ) : ℕ := 6 * (x - 1) ^ 2 + 2

/-- The side length of the largest possible hollow cube -/
def largest_hollow_side : ℕ := 13

theorem max_unused_cubes :
  ∃ (unused : ℕ), unused = total_cubes - cubes_used largest_hollow_side ∧
  unused = 134 ∧
  ∀ (x : ℕ), x > largest_hollow_side → cubes_used x > total_cubes :=
sorry

end NUMINAMATH_CALUDE_max_unused_cubes_l1724_172475


namespace NUMINAMATH_CALUDE_zodiac_pigeonhole_l1724_172461

/-- The number of Greek Zodiac signs -/
def greek_zodiac_count : ℕ := 12

/-- The number of Chinese Zodiac signs -/
def chinese_zodiac_count : ℕ := 12

/-- The minimum number of people required to ensure at least 3 people have the same Greek Zodiac sign -/
def min_people_same_greek_sign : ℕ := greek_zodiac_count * 2 + 1

/-- The minimum number of people required to ensure at least 2 people have the same combination of Greek and Chinese Zodiac signs -/
def min_people_same_combined_signs : ℕ := greek_zodiac_count * chinese_zodiac_count + 1

theorem zodiac_pigeonhole :
  (min_people_same_greek_sign = 25) ∧
  (min_people_same_combined_signs = 145) := by
  sorry

end NUMINAMATH_CALUDE_zodiac_pigeonhole_l1724_172461


namespace NUMINAMATH_CALUDE_square_plus_double_equals_one_implies_double_square_plus_quadruple_plus_one_equals_three_l1724_172477

theorem square_plus_double_equals_one_implies_double_square_plus_quadruple_plus_one_equals_three
  (a : ℝ) (h : a^2 + 2*a = 1) : 2*a^2 + 4*a + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_double_equals_one_implies_double_square_plus_quadruple_plus_one_equals_three_l1724_172477


namespace NUMINAMATH_CALUDE_triangle_circumcircle_radius_l1724_172418

theorem triangle_circumcircle_radius 
  (a : ℝ) 
  (A : ℝ) 
  (h1 : a = 2) 
  (h2 : A = 2 * π / 3) : 
  ∃ R : ℝ, R = (2 * Real.sqrt 3) / 3 ∧ 
  R = a / (2 * Real.sin A) := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_radius_l1724_172418


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l1724_172412

/-- The range of m for which the quadratic equation (m-1)x^2 + 2x + 1 = 0 has two real roots -/
theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (m - 1) * x₁^2 + 2 * x₁ + 1 = 0 ∧ 
    (m - 1) * x₂^2 + 2 * x₂ + 1 = 0) ↔ 
  (m ≤ 2 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l1724_172412


namespace NUMINAMATH_CALUDE_distance_to_left_focus_l1724_172406

/-- Given an ellipse and a hyperbola, prove that the distance from their intersection point
    in the first quadrant to the left focus of the ellipse is 4. -/
theorem distance_to_left_focus (x y : ℝ) : 
  x > 0 → y > 0 →  -- P is in the first quadrant
  x^2 / 9 + y^2 / 5 = 1 →  -- Ellipse equation
  x^2 - y^2 / 3 = 1 →  -- Hyperbola equation
  ∃ (f₁ : ℝ × ℝ), -- Left focus of the ellipse
    Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_left_focus_l1724_172406


namespace NUMINAMATH_CALUDE_square_root_equation_implies_product_l1724_172437

theorem square_root_equation_implies_product (x : ℝ) :
  Real.sqrt (8 + x) + Real.sqrt (25 - x^2) = 9 →
  (8 + x) * (25 - x^2) = 576 := by
sorry

end NUMINAMATH_CALUDE_square_root_equation_implies_product_l1724_172437


namespace NUMINAMATH_CALUDE_ab_product_l1724_172465

theorem ab_product (a b : ℚ) (h : 6 * a = 20 ∧ 7 * b = 20) : 84 * a * b = 800 := by
  sorry

end NUMINAMATH_CALUDE_ab_product_l1724_172465


namespace NUMINAMATH_CALUDE_initial_owls_count_l1724_172408

theorem initial_owls_count (initial_owls final_owls joined_owls : ℕ) : 
  initial_owls + joined_owls = final_owls →
  joined_owls = 2 →
  final_owls = 5 →
  initial_owls = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_owls_count_l1724_172408


namespace NUMINAMATH_CALUDE_circle_arrangement_impossibility_l1724_172431

theorem circle_arrangement_impossibility :
  ¬ ∃ (arrangement : Fin 2017 → ℕ),
    (∀ i, arrangement i ∈ Finset.range 2017 ∧ arrangement i ≠ 0) ∧
    (∀ i j, i ≠ j → arrangement i ≠ arrangement j) ∧
    (∀ i, Even ((arrangement i) + (arrangement ((i + 1) % 2017)) + (arrangement ((i + 2) % 2017)))) :=
by sorry

end NUMINAMATH_CALUDE_circle_arrangement_impossibility_l1724_172431


namespace NUMINAMATH_CALUDE_complement_of_M_l1724_172426

-- Define the set M
def M : Set ℝ := {x : ℝ | x * (x - 3) > 0}

-- State the theorem
theorem complement_of_M : 
  (Set.univ : Set ℝ) \ M = Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l1724_172426


namespace NUMINAMATH_CALUDE_a_5_equals_13_l1724_172479

/-- A sequence defined by a_n = pn + q -/
def a (p q : ℝ) : ℕ+ → ℝ := fun n ↦ p * n.val + q

/-- Given a sequence a_n where a_1 = 5, a_8 = 19, and a_n = pn + q for all n ∈ ℕ+
    (where p and q are constants), prove that a_5 = 13 -/
theorem a_5_equals_13 (p q : ℝ) (h1 : a p q 1 = 5) (h8 : a p q 8 = 19) : a p q 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_a_5_equals_13_l1724_172479


namespace NUMINAMATH_CALUDE_parabola_transformation_l1724_172451

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = -2x^2 + 1 -/
def original_parabola : Parabola := ⟨-2, 0, 1⟩

/-- Moves a parabola horizontally by h units -/
def move_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  ⟨p.a, -2 * p.a * h + p.b, p.a * h^2 - p.b * h + p.c⟩

/-- Moves a parabola vertically by k units -/
def move_vertical (p : Parabola) (k : ℝ) : Parabola :=
  ⟨p.a, p.b, p.c + k⟩

/-- The final parabola after moving right by 1 and up by 1 -/
def final_parabola : Parabola :=
  move_vertical (move_horizontal original_parabola 1) 1

theorem parabola_transformation :
  final_parabola = ⟨-2, 4, 2⟩ := by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l1724_172451


namespace NUMINAMATH_CALUDE_odd_function_sum_l1724_172439

/-- A function f is odd on an interval [a, b] -/
def IsOddOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ Set.Icc a b, f (-x) = -f x) ∧ a + b = 0

/-- The main theorem -/
theorem odd_function_sum (a b c : ℝ) :
  IsOddOn (fun x ↦ a * x^3 + x + c) a b →
  a + b + c + 2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_odd_function_sum_l1724_172439


namespace NUMINAMATH_CALUDE_rahim_average_book_price_l1724_172435

/-- The average price of books bought by Rahim -/
def average_price (books1 books2 : ℕ) (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / (books1 + books2)

/-- Theorem stating the average price of books bought by Rahim -/
theorem rahim_average_book_price :
  let books1 := 65
  let books2 := 50
  let price1 := 1160
  let price2 := 920
  abs (average_price books1 books2 price1 price2 - 18.09) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_rahim_average_book_price_l1724_172435


namespace NUMINAMATH_CALUDE_shopping_cost_l1724_172401

def toilet_paper_quantity : ℕ := 10
def paper_towel_quantity : ℕ := 7
def tissue_quantity : ℕ := 3

def toilet_paper_price : ℚ := 3/2
def paper_towel_price : ℚ := 2
def tissue_price : ℚ := 2

def total_cost : ℚ := 
  toilet_paper_quantity * toilet_paper_price + 
  paper_towel_quantity * paper_towel_price + 
  tissue_quantity * tissue_price

theorem shopping_cost : total_cost = 35 := by
  sorry

end NUMINAMATH_CALUDE_shopping_cost_l1724_172401
