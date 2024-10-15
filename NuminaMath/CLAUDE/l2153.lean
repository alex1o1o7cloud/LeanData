import Mathlib

namespace NUMINAMATH_CALUDE_oranges_per_group_l2153_215387

theorem oranges_per_group (total_oranges : ℕ) (num_groups : ℕ) 
  (h1 : total_oranges = 384) (h2 : num_groups = 16) :
  total_oranges / num_groups = 24 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_group_l2153_215387


namespace NUMINAMATH_CALUDE_friendship_theorem_l2153_215327

/-- A simple graph with 17 vertices where each vertex has degree 4 -/
structure FriendshipGraph where
  vertices : Finset (Fin 17)
  edges : Finset (Fin 17 × Fin 17)
  edge_symmetric : ∀ a b, (a, b) ∈ edges → (b, a) ∈ edges
  no_self_loops : ∀ a, (a, a) ∉ edges
  degree_four : ∀ v, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 4

/-- Two vertices are acquainted if there's an edge between them -/
def acquainted (G : FriendshipGraph) (a b : Fin 17) : Prop :=
  (a, b) ∈ G.edges

/-- Two vertices share a common neighbor if there exists a vertex connected to both -/
def share_neighbor (G : FriendshipGraph) (a b : Fin 17) : Prop :=
  ∃ c, acquainted G a c ∧ acquainted G b c

/-- Main theorem: There exist two vertices that are not acquainted and do not share a neighbor -/
theorem friendship_theorem (G : FriendshipGraph) : 
  ∃ a b, a ≠ b ∧ ¬(acquainted G a b) ∧ ¬(share_neighbor G a b) := by
  sorry

end NUMINAMATH_CALUDE_friendship_theorem_l2153_215327


namespace NUMINAMATH_CALUDE_complex_subtraction_l2153_215312

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 4 + I) :
  a - 3*b = -7 - 6*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l2153_215312


namespace NUMINAMATH_CALUDE_remainder_theorem_l2153_215366

theorem remainder_theorem (r : ℤ) : (r^15 - 1) % (r + 2) = -32769 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2153_215366


namespace NUMINAMATH_CALUDE_no_valid_rectangle_l2153_215397

theorem no_valid_rectangle (a b x y : ℝ) : 
  a < b →
  x < a →
  y < a →
  2 * (x + y) = (2 * (a + b)) / 3 →
  x * y = (a * b) / 3 →
  False :=
by sorry

end NUMINAMATH_CALUDE_no_valid_rectangle_l2153_215397


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_l2153_215351

theorem units_digit_of_sum_of_powers (a b : ℕ) (ha : a = 15) (hb : b = 220) :
  ∃ k : ℤ, (a + Real.sqrt b)^19 + (a - Real.sqrt b)^19 = 10 * k + 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_l2153_215351


namespace NUMINAMATH_CALUDE_complex_fraction_equals_seven_plus_i_l2153_215317

theorem complex_fraction_equals_seven_plus_i :
  let i : ℂ := Complex.I
  (1 + i) * (3 + 4*i) / i = 7 + i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_seven_plus_i_l2153_215317


namespace NUMINAMATH_CALUDE_algorithm_steps_are_determinate_l2153_215383

/-- Represents a step in an algorithm -/
structure AlgorithmStep where
  precise : Bool
  effective : Bool
  determinate : Bool

/-- Represents an algorithm -/
structure Algorithm where
  steps : List AlgorithmStep
  solvesProblem : Bool
  finite : Bool

/-- Theorem: Given an algorithm with finite, precise, and effective steps that solve a problem, 
    prove that all steps in the algorithm are determinate -/
theorem algorithm_steps_are_determinate (a : Algorithm) 
  (h1 : a.solvesProblem)
  (h2 : a.finite)
  (h3 : ∀ s ∈ a.steps, s.precise)
  (h4 : ∀ s ∈ a.steps, s.effective) :
  ∀ s ∈ a.steps, s.determinate := by
  sorry


end NUMINAMATH_CALUDE_algorithm_steps_are_determinate_l2153_215383


namespace NUMINAMATH_CALUDE_sum_of_triangle_ops_l2153_215346

-- Define the triangle operation
def triangle_op (a b c : ℤ) : ℤ := 2*a + 3*b - 4*c

-- State the theorem
theorem sum_of_triangle_ops : 
  triangle_op 2 3 5 + triangle_op 4 6 1 = 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_triangle_ops_l2153_215346


namespace NUMINAMATH_CALUDE_bees_after_six_days_l2153_215349

/-- The number of bees after n days in the hive process -/
def bees (n : ℕ) : ℕ := 6^n

/-- The process starts with 1 bee and continues for 6 days -/
def days : ℕ := 6

/-- The theorem stating the number of bees after 6 days -/
theorem bees_after_six_days : bees days = 46656 := by sorry

end NUMINAMATH_CALUDE_bees_after_six_days_l2153_215349


namespace NUMINAMATH_CALUDE_range_of_g_l2153_215323

/-- A function g defined on the interval [-1, 1] with g(x) = cx + d, where c < 0 and d > 0 -/
def g (c d : ℝ) (hc : c < 0) (hd : d > 0) : ℝ → ℝ :=
  fun x => c * x + d

/-- The range of g is [c + d, -c + d] -/
theorem range_of_g (c d : ℝ) (hc : c < 0) (hd : d > 0) :
  Set.range (g c d hc hd) = Set.Icc (c + d) (-c + d) := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l2153_215323


namespace NUMINAMATH_CALUDE_increasing_condition_m_range_l2153_215311

-- Define the linear function
def y (m x : ℝ) : ℝ := (m - 2) * x + 6

-- Part 1: y increases as x increases iff m > 2
theorem increasing_condition (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → y m x₁ < y m x₂) ↔ m > 2 :=
sorry

-- Part 2: Range of m when -2 ≤ x ≤ 4 and y ≤ 10
theorem m_range (m : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 4 → y m x ≤ 10) ↔ (2 < m ∧ m ≤ 3) ∨ (0 ≤ m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_increasing_condition_m_range_l2153_215311


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2153_215378

-- Define the universal set U
def U : Set ℕ := {x | 0 ≤ x ∧ x < 6}

-- Define set A
def A : Set ℕ := {1, 3, 5}

-- Define set B
def B : Set ℕ := {x ∈ U | x^2 + 4 = 5*x}

-- Theorem statement
theorem complement_union_theorem : 
  (U \ A) ∪ (U \ B) = {0, 2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2153_215378


namespace NUMINAMATH_CALUDE_abes_age_problem_l2153_215350

theorem abes_age_problem (present_age : ℕ) (sum_ages : ℕ) (years_ago : ℕ) :
  present_age = 19 →
  sum_ages = 31 →
  sum_ages = present_age + (present_age - years_ago) →
  years_ago = 7 := by
sorry

end NUMINAMATH_CALUDE_abes_age_problem_l2153_215350


namespace NUMINAMATH_CALUDE_min_gumballs_for_five_colors_l2153_215339

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- The minimum number of gumballs needed to guarantee five of the same color -/
def minGumballsForFive (machine : GumballMachine) : Nat :=
  16

/-- Theorem stating that for a machine with 10 red, 10 white, 10 blue, and 6 green gumballs,
    the minimum number of gumballs needed to guarantee five of the same color is 16 -/
theorem min_gumballs_for_five_colors (machine : GumballMachine)
    (h_red : machine.red = 10)
    (h_white : machine.white = 10)
    (h_blue : machine.blue = 10)
    (h_green : machine.green = 6) :
    minGumballsForFive machine = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_gumballs_for_five_colors_l2153_215339


namespace NUMINAMATH_CALUDE_max_sum_removed_numbers_l2153_215314

theorem max_sum_removed_numbers (n : ℕ) (m k : ℕ) 
  (h1 : n > 2) 
  (h2 : 1 < m ∧ m < n) 
  (h3 : 1 < k ∧ k < n) 
  (h4 : (n * (n + 1) / 2 - m - k) / (n - 2) = 17) :
  m + k ≤ 51 ∧ ∃ (m' k' : ℕ), 1 < m' ∧ m' < n ∧ 1 < k' ∧ k' < n ∧ m' + k' = 51 := by
  sorry

#check max_sum_removed_numbers

end NUMINAMATH_CALUDE_max_sum_removed_numbers_l2153_215314


namespace NUMINAMATH_CALUDE_laptop_price_calculation_l2153_215386

def original_price : ℝ := 1200
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.12

def discounted_price : ℝ := original_price * (1 - discount_rate)
def total_price : ℝ := discounted_price * (1 + tax_rate)

theorem laptop_price_calculation :
  total_price = 940.8 := by sorry

end NUMINAMATH_CALUDE_laptop_price_calculation_l2153_215386


namespace NUMINAMATH_CALUDE_product_of_powers_l2153_215319

theorem product_of_powers (y : ℝ) (hy : y ≠ 0) :
  (18 * y^3) * (4 * y^2) * (1 / (2*y)^3) = 9 * y^2 := by
sorry

end NUMINAMATH_CALUDE_product_of_powers_l2153_215319


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l2153_215370

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: The sum of the interior angles of a pentagon is 540° -/
theorem sum_interior_angles_pentagon :
  sum_interior_angles 5 = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l2153_215370


namespace NUMINAMATH_CALUDE_xy_value_l2153_215390

theorem xy_value (x y : ℝ) (h : x^2 + y^2 - 22*x - 20*y + 221 = 0) : x * y = 110 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2153_215390


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4652_l2153_215379

theorem largest_prime_factor_of_4652 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4652 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 4652 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4652_l2153_215379


namespace NUMINAMATH_CALUDE_meeting_distance_l2153_215358

theorem meeting_distance (initial_speed : ℝ) (speed_increase : ℝ) (initial_distance : ℝ) 
  (late_time : ℝ) (early_time : ℝ) :
  initial_speed = 45 ∧ 
  speed_increase = 20 ∧ 
  initial_distance = 45 ∧ 
  late_time = 0.75 ∧ 
  early_time = 0.25 → 
  ∃ (total_distance : ℝ),
    total_distance = initial_speed * (total_distance / initial_speed + late_time) ∧
    total_distance - initial_distance = (initial_speed + speed_increase) * 
      (total_distance / initial_speed - 1 - early_time) ∧
    total_distance = 191.25 := by
  sorry

end NUMINAMATH_CALUDE_meeting_distance_l2153_215358


namespace NUMINAMATH_CALUDE_unique_bezout_bounded_l2153_215330

theorem unique_bezout_bounded (a b : ℕ) (ha : a > 1) (hb : b > 1) (hgcd : Nat.gcd a b = 1) :
  ∃! (r s : ℕ), a * r - b * s = 1 ∧ 0 < r ∧ r < b ∧ 0 < s ∧ s < a := by
  sorry

end NUMINAMATH_CALUDE_unique_bezout_bounded_l2153_215330


namespace NUMINAMATH_CALUDE_z_minimum_l2153_215313

/-- The function z(x, y) defined in the problem -/
def z (x y : ℝ) : ℝ := x^2 + 2*x*y + 2*y^2 + 2*x + 4*y + 3

/-- Theorem stating the minimum value of z and where it occurs -/
theorem z_minimum :
  (∀ x y : ℝ, z x y ≥ 1) ∧ (z 0 (-1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_z_minimum_l2153_215313


namespace NUMINAMATH_CALUDE_total_amount_is_1800_l2153_215355

/-- Calculates the total amount spent on courses for two semesters --/
def total_amount_spent (
  units_per_semester : ℕ
  ) (science_cost_per_unit : ℚ
  ) (humanities_cost_per_unit : ℚ
  ) (science_units_first : ℕ
  ) (humanities_units_first : ℕ
  ) (science_units_second : ℕ
  ) (humanities_units_second : ℕ
  ) (scholarship_percentage : ℚ
  ) : ℚ :=
  let first_semester_cost := 
    science_cost_per_unit * science_units_first + 
    humanities_cost_per_unit * humanities_units_first
  let second_semester_cost := 
    (1 - scholarship_percentage) * science_cost_per_unit * science_units_second + 
    humanities_cost_per_unit * humanities_units_second
  first_semester_cost + second_semester_cost

theorem total_amount_is_1800 :
  total_amount_spent 20 60 45 12 8 12 8 (1/2) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_1800_l2153_215355


namespace NUMINAMATH_CALUDE_triangle_properties_l2153_215316

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)
  (angle_sum : A + B + C = π)

theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt 3 * Real.cos t.C + Real.sin t.C = (Real.sqrt 3 * t.a) / t.b)
  (h2 : t.a + t.c = 5 * Real.sqrt 7)
  (h3 : t.b = 7) :
  t.B = π / 3 ∧ t.a * t.c * Real.cos t.B = -21 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2153_215316


namespace NUMINAMATH_CALUDE_prime_divisibility_l2153_215344

theorem prime_divisibility (p q : ℕ) : 
  Prime p → Prime q → q ∣ (3^p - 2^p) → p ∣ (q - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l2153_215344


namespace NUMINAMATH_CALUDE_f_composition_of_one_l2153_215393

def f (x : ℤ) : ℤ :=
  if x % 2 = 0 then 3 * x / 2 else 2 * x + 1

theorem f_composition_of_one : f (f (f (f 1))) = 31 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_one_l2153_215393


namespace NUMINAMATH_CALUDE_temporary_employee_percentage_is_51_5_l2153_215343

/-- Represents the composition of workers in a factory -/
structure WorkerComposition where
  technicians : Real
  skilled_laborers : Real
  unskilled_laborers : Real
  permanent_technicians : Real
  permanent_skilled : Real
  permanent_unskilled : Real

/-- Calculates the percentage of temporary employees in the factory -/
def temporary_employee_percentage (wc : WorkerComposition) : Real :=
  100 - (wc.technicians * wc.permanent_technicians + 
         wc.skilled_laborers * wc.permanent_skilled + 
         wc.unskilled_laborers * wc.permanent_unskilled)

/-- Theorem stating that given the conditions, the percentage of temporary employees is 51.5% -/
theorem temporary_employee_percentage_is_51_5 (wc : WorkerComposition) 
  (h1 : wc.technicians = 40)
  (h2 : wc.skilled_laborers = 35)
  (h3 : wc.unskilled_laborers = 25)
  (h4 : wc.permanent_technicians = 60)
  (h5 : wc.permanent_skilled = 45)
  (h6 : wc.permanent_unskilled = 35) :
  temporary_employee_percentage wc = 51.5 := by
  sorry

end NUMINAMATH_CALUDE_temporary_employee_percentage_is_51_5_l2153_215343


namespace NUMINAMATH_CALUDE_age_difference_l2153_215362

theorem age_difference (sachin_age rahul_age : ℝ) : 
  sachin_age = 24.5 → 
  sachin_age / rahul_age = 7 / 9 → 
  rahul_age - sachin_age = 7 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2153_215362


namespace NUMINAMATH_CALUDE_abc_is_50_l2153_215334

def repeating_decimal (a b c : ℕ) : ℚ :=
  1 + (100 * a + 10 * b + c : ℚ) / 999

theorem abc_is_50 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  12 * (repeating_decimal a b c - (1 + (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000)) = 0.6 →
  100 * a + 10 * b + c = 50 := by
sorry

end NUMINAMATH_CALUDE_abc_is_50_l2153_215334


namespace NUMINAMATH_CALUDE_power_of_negative_product_l2153_215348

theorem power_of_negative_product (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l2153_215348


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l2153_215300

/-- The equation (m-2)x^2 - 3x = 0 is quadratic in x if and only if m ≠ 2 -/
theorem quadratic_equation_condition (m : ℝ) :
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m - 2) * x^2 - 3 * x = a * x^2 + b * x + c) ↔ m ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l2153_215300


namespace NUMINAMATH_CALUDE_equation_root_condition_l2153_215376

/-- The equation has a root greater than zero if and only if a = -8 -/
theorem equation_root_condition (a : ℝ) : 
  (∃ x > 0, (3*x - 1) / (x - 3) = a / (3 - x) - 1) ↔ a = -8 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_condition_l2153_215376


namespace NUMINAMATH_CALUDE_arithmetic_progression_not_power_l2153_215391

theorem arithmetic_progression_not_power (n : ℕ) (k : ℕ) : 
  let a : ℕ → ℕ := λ i => 4 * i - 2
  ∀ i : ℕ, ∀ r : ℕ, 2 ≤ r → r ≤ n → ¬ ∃ m : ℕ, a i = m ^ r :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_not_power_l2153_215391


namespace NUMINAMATH_CALUDE_sequence_problem_l2153_215305

theorem sequence_problem (n : ℕ+) (b : ℕ → ℝ)
  (h0 : b 0 = 40)
  (h1 : b 1 = 70)
  (hn : b n = 0)
  (hk : ∀ k : ℕ, 1 ≤ k → k < n → b (k + 1) = b (k - 1) - 2 / b k) :
  n = 1401 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l2153_215305


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_l2153_215367

-- Define the parametric equations
def x_param (t : ℝ) : ℝ := t + 1
def y_param (t : ℝ) : ℝ := 3 - t^2

-- State the theorem
theorem parametric_to_cartesian :
  ∀ (x y : ℝ), (∃ t : ℝ, x = x_param t ∧ y = y_param t) ↔ y = -x^2 + 2*x + 2 :=
by sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_l2153_215367


namespace NUMINAMATH_CALUDE_complex_power_four_l2153_215304

theorem complex_power_four (i : ℂ) : i^2 = -1 → (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l2153_215304


namespace NUMINAMATH_CALUDE_largest_lcm_with_18_l2153_215399

theorem largest_lcm_with_18 :
  max (Nat.lcm 18 3) (max (Nat.lcm 18 6) (max (Nat.lcm 18 9) (max (Nat.lcm 18 12) (max (Nat.lcm 18 15) (Nat.lcm 18 18))))) = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_18_l2153_215399


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2153_215324

theorem rectangle_perimeter (L W : ℝ) (h1 : L * W = (L + 6) * (W - 2)) (h2 : L * W = (L - 12) * (W + 6)) : 
  2 * (L + W) = 132 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2153_215324


namespace NUMINAMATH_CALUDE_min_exponent_sum_520_l2153_215364

/-- Given a natural number n, returns the minimum sum of exponents when expressing n as a sum of at least two distinct powers of 2 -/
def min_exponent_sum (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the minimum sum of exponents when expressing 520 as a sum of at least two distinct powers of 2 is 12 -/
theorem min_exponent_sum_520 : min_exponent_sum 520 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_exponent_sum_520_l2153_215364


namespace NUMINAMATH_CALUDE_z_value_l2153_215372

theorem z_value (a : ℕ) (z : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * z) : z = 49 := by
  sorry

end NUMINAMATH_CALUDE_z_value_l2153_215372


namespace NUMINAMATH_CALUDE_parallelogram_most_analogous_to_parallelepiped_l2153_215302

-- Define the types for 2D and 3D figures
inductive PlaneFigure
| Triangle
| Trapezoid
| Rectangle
| Parallelogram

inductive SpaceFigure
| Parallelepiped

-- Define the property of being formed by translation
def FormedByTranslation (plane : PlaneFigure) (space : SpaceFigure) : Prop :=
  match plane, space with
  | PlaneFigure.Parallelogram, SpaceFigure.Parallelepiped => True
  | _, _ => False

-- Define the concept of being analogous
def Analogous (plane : PlaneFigure) (space : SpaceFigure) : Prop :=
  FormedByTranslation plane space

-- Theorem statement
theorem parallelogram_most_analogous_to_parallelepiped :
  ∀ (plane : PlaneFigure),
    Analogous plane SpaceFigure.Parallelepiped →
    plane = PlaneFigure.Parallelogram :=
sorry

end NUMINAMATH_CALUDE_parallelogram_most_analogous_to_parallelepiped_l2153_215302


namespace NUMINAMATH_CALUDE_f_two_equals_six_l2153_215357

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 6

theorem f_two_equals_six (a b : ℝ) (h : f a b (-1) = f a b 3) : f a b 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_two_equals_six_l2153_215357


namespace NUMINAMATH_CALUDE_unique_rectangle_with_half_perimeter_quarter_area_l2153_215320

theorem unique_rectangle_with_half_perimeter_quarter_area 
  (a b : ℝ) (hab : a < b) : 
  ∃! (x y : ℝ), x < b ∧ y < b ∧ 
  2 * (x + y) = a + b ∧ 
  x * y = (a * b) / 4 := by
sorry

end NUMINAMATH_CALUDE_unique_rectangle_with_half_perimeter_quarter_area_l2153_215320


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_240_l2153_215303

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_240 :
  rectangle_area 3600 10 = 240 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_240_l2153_215303


namespace NUMINAMATH_CALUDE_artworks_per_quarter_is_two_l2153_215368

/-- The number of students in the art club -/
def num_students : ℕ := 15

/-- The number of quarters in a school year -/
def quarters_per_year : ℕ := 4

/-- The total number of artworks collected in two school years -/
def total_artworks : ℕ := 240

/-- The number of artworks each student makes by the end of each quarter -/
def artworks_per_student_per_quarter : ℕ := 2

/-- Theorem stating that the number of artworks each student makes by the end of each quarter is 2 -/
theorem artworks_per_quarter_is_two :
  artworks_per_student_per_quarter * num_students * quarters_per_year * 2 = total_artworks :=
by sorry

end NUMINAMATH_CALUDE_artworks_per_quarter_is_two_l2153_215368


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l2153_215322

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∃ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) = 7) ∧
  (∀ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 7) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l2153_215322


namespace NUMINAMATH_CALUDE_sum_of_products_l2153_215301

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 20) (hxz : x * z = 60) (hyz : y * z = 90) :
  x + y + z = 11 * Real.sqrt 30 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2153_215301


namespace NUMINAMATH_CALUDE_profit_at_10_yuan_increase_profit_maximum_at_5_yuan_increase_l2153_215325

/-- Represents the product pricing model -/
structure PricingModel where
  currentPrice : ℝ
  weeklySales : ℝ
  salesDecrease : ℝ
  costPrice : ℝ

/-- Calculates the profit for a given price increase -/
def profit (model : PricingModel) (priceIncrease : ℝ) : ℝ :=
  (model.currentPrice + priceIncrease - model.costPrice) *
  (model.weeklySales - model.salesDecrease * priceIncrease)

/-- The pricing model for the given problem -/
def givenModel : PricingModel :=
  { currentPrice := 60
    weeklySales := 300
    salesDecrease := 10
    costPrice := 40 }

/-- Theorem: A price increase of 10 yuan results in a weekly profit of 6000 yuan -/
theorem profit_at_10_yuan_increase (ε : ℝ) :
  |profit givenModel 10 - 6000| < ε := by sorry

/-- Theorem: A price increase of 5 yuan maximizes the weekly profit -/
theorem profit_maximum_at_5_yuan_increase :
  ∀ x, profit givenModel 5 ≥ profit givenModel x := by sorry

end NUMINAMATH_CALUDE_profit_at_10_yuan_increase_profit_maximum_at_5_yuan_increase_l2153_215325


namespace NUMINAMATH_CALUDE_lesser_fraction_l2153_215308

theorem lesser_fraction (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : 
  min x y = 1/4 := by
sorry

end NUMINAMATH_CALUDE_lesser_fraction_l2153_215308


namespace NUMINAMATH_CALUDE_quadratic_has_two_real_roots_roots_equal_absolute_value_l2153_215363

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (m + 3) * x + m + 2

-- Theorem 1: The quadratic equation always has two real roots
theorem quadratic_has_two_real_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

-- Theorem 2: The absolute values of the roots are equal iff m = -1 or m = -3
theorem roots_equal_absolute_value (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ |x₁| = |x₂|) ↔
  (m = -1 ∨ m = -3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_real_roots_roots_equal_absolute_value_l2153_215363


namespace NUMINAMATH_CALUDE_circle_center_proof_l2153_215307

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation x² - 10x + y² - 8y = 16, 
    prove that its center is (5, 4) -/
theorem circle_center_proof (eq : CircleEquation) 
  (h1 : eq.a = 1)
  (h2 : eq.b = -10)
  (h3 : eq.c = 1)
  (h4 : eq.d = -8)
  (h5 : eq.e = -16) :
  CircleCenter.mk 5 4 = CircleCenter.mk (-eq.b / (2 * eq.a)) (-eq.d / (2 * eq.c)) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_proof_l2153_215307


namespace NUMINAMATH_CALUDE_sum_f_positive_l2153_215345

def f (x : ℝ) : ℝ := x^5 + x

theorem sum_f_positive (a b c : ℝ) (hab : a + b > 0) (hbc : b + c > 0) (hca : c + a > 0) :
  f a + f b + f c > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l2153_215345


namespace NUMINAMATH_CALUDE_carlos_pesos_l2153_215340

/-- The exchange rate from Mexican pesos to U.S. dollars -/
def exchange_rate : ℚ := 8 / 14

/-- The amount spent in U.S. dollars -/
def amount_spent : ℕ := 50

/-- The remaining amount is three times the spent amount -/
def remaining_ratio : ℕ := 3

/-- The number of Mexican pesos Carlos had -/
def p : ℕ := 350

theorem carlos_pesos :
  p * exchange_rate - amount_spent = remaining_ratio * amount_spent := by
  sorry

end NUMINAMATH_CALUDE_carlos_pesos_l2153_215340


namespace NUMINAMATH_CALUDE_double_shot_espresso_price_l2153_215375

/-- Represents the cost of a coffee order -/
structure CoffeeOrder where
  drip_coffee : ℕ
  drip_coffee_price : ℚ
  latte : ℕ
  latte_price : ℚ
  vanilla_syrup : ℕ
  vanilla_syrup_price : ℚ
  cold_brew : ℕ
  cold_brew_price : ℚ
  cappuccino : ℕ
  cappuccino_price : ℚ
  double_shot_espresso : ℕ
  total_price : ℚ

/-- Calculates the cost of the double shot espresso -/
def double_shot_espresso_cost (order : CoffeeOrder) : ℚ :=
  order.total_price -
  (order.drip_coffee * order.drip_coffee_price +
   order.latte * order.latte_price +
   order.vanilla_syrup * order.vanilla_syrup_price +
   order.cold_brew * order.cold_brew_price +
   order.cappuccino * order.cappuccino_price)

/-- Theorem stating that the double shot espresso costs $3.50 -/
theorem double_shot_espresso_price (order : CoffeeOrder) 
  (h1 : order.drip_coffee = 2)
  (h2 : order.drip_coffee_price = 2.25)
  (h3 : order.latte = 2)
  (h4 : order.latte_price = 4)
  (h5 : order.vanilla_syrup = 1)
  (h6 : order.vanilla_syrup_price = 0.5)
  (h7 : order.cold_brew = 2)
  (h8 : order.cold_brew_price = 2.5)
  (h9 : order.cappuccino = 1)
  (h10 : order.cappuccino_price = 3.5)
  (h11 : order.double_shot_espresso = 1)
  (h12 : order.total_price = 25) :
  double_shot_espresso_cost order = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_double_shot_espresso_price_l2153_215375


namespace NUMINAMATH_CALUDE_expression_simplification_l2153_215352

theorem expression_simplification (x y z : ℝ) : 
  (x - (2*y + z)) - ((x + 2*y) - 3*z) = -4*y + 2*z := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2153_215352


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_shared_foci_l2153_215374

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the ellipse
def ellipse (a x y : ℝ) : Prop := x^2 / a^2 + y^2 / 16 = 1

-- Define the condition that a > 0
def a_positive (a : ℝ) : Prop := a > 0

-- Define the condition that the hyperbola and ellipse share the same foci
def same_foci (a : ℝ) : Prop := ∃ c : ℝ, c^2 = 9 ∧ 
  (∀ x y : ℝ, hyperbola x y ↔ x^2 / 4 - y^2 / 5 = 1) ∧
  (∀ x y : ℝ, ellipse a x y ↔ x^2 / a^2 + y^2 / 16 = 1)

-- Theorem statement
theorem hyperbola_ellipse_shared_foci (a : ℝ) :
  a_positive a → same_foci a → a = 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_shared_foci_l2153_215374


namespace NUMINAMATH_CALUDE_max_dominoes_20x19_grid_l2153_215329

/-- Represents a rectangular grid --/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Represents a domino --/
structure Domino where
  length : ℕ
  width : ℕ

/-- The maximum number of dominoes that can be placed on a grid --/
def max_dominoes (g : Grid) (d : Domino) : ℕ :=
  (g.rows * g.cols) / (d.length * d.width)

/-- The theorem stating the maximum number of 3×1 dominoes on a 20×19 grid --/
theorem max_dominoes_20x19_grid :
  let grid : Grid := ⟨20, 19⟩
  let domino : Domino := ⟨3, 1⟩
  max_dominoes grid domino = 126 := by
  sorry

#eval max_dominoes ⟨20, 19⟩ ⟨3, 1⟩

end NUMINAMATH_CALUDE_max_dominoes_20x19_grid_l2153_215329


namespace NUMINAMATH_CALUDE_reflect_F_coordinates_l2153_215384

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point over the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- The original point F -/
def F : ℝ × ℝ := (-1, -1)

theorem reflect_F_coordinates :
  (reflect_y_eq_x (reflect_x F)) = (1, -1) := by
sorry

end NUMINAMATH_CALUDE_reflect_F_coordinates_l2153_215384


namespace NUMINAMATH_CALUDE_function_equation_implies_identity_l2153_215394

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f x + y) * (f (x - y) + 1) = f (f (x * f (x + 1)) - y * f (y - 1))) :
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_implies_identity_l2153_215394


namespace NUMINAMATH_CALUDE_probability_of_eight_in_three_elevenths_l2153_215309

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : List ℕ := sorry

/-- The probability of a digit occurring in a list of digits -/
def digit_probability (d : ℕ) (l : List ℕ) : ℚ := sorry

theorem probability_of_eight_in_three_elevenths :
  digit_probability 8 (decimal_representation (3/11)) = 0 := by sorry

end NUMINAMATH_CALUDE_probability_of_eight_in_three_elevenths_l2153_215309


namespace NUMINAMATH_CALUDE_water_balloon_ratio_l2153_215338

/-- Prove that the ratio of Randy's water balloons to Janice's water balloons is 1:2 -/
theorem water_balloon_ratio 
  (cynthia_balloons : ℕ) 
  (janice_balloons : ℕ) 
  (h1 : cynthia_balloons = 12)
  (h2 : janice_balloons = 6)
  (h3 : cynthia_balloons = 4 * (cynthia_balloons / 4)) :
  (cynthia_balloons / 4) / janice_balloons = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_water_balloon_ratio_l2153_215338


namespace NUMINAMATH_CALUDE_max_sum_with_product_2665_l2153_215354

theorem max_sum_with_product_2665 :
  ∀ A B C : ℕ+,
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2665 →
  A + B + C ≤ 539 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_product_2665_l2153_215354


namespace NUMINAMATH_CALUDE_ratio_cubes_equals_twentyseven_l2153_215388

theorem ratio_cubes_equals_twentyseven : (81000 ^ 3) / (27000 ^ 3) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ratio_cubes_equals_twentyseven_l2153_215388


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2153_215341

theorem complex_equation_solution :
  ∃ x : ℂ, (3 : ℂ) + 2 * Complex.I * x = 4 - 5 * Complex.I * x ∧ x = -Complex.I / 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2153_215341


namespace NUMINAMATH_CALUDE_sin_negative_three_pi_plus_alpha_l2153_215353

theorem sin_negative_three_pi_plus_alpha (α : ℝ) (h : Real.sin (π + α) = 1/3) :
  Real.sin (-3*π + α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_three_pi_plus_alpha_l2153_215353


namespace NUMINAMATH_CALUDE_slope_of_line_l2153_215337

theorem slope_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → (y - 4) / (-x) = -4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l2153_215337


namespace NUMINAMATH_CALUDE_largest_c_for_negative_five_in_range_l2153_215361

/-- The function f(x) defined as x^2 + 5x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + c

/-- Theorem stating that 5/4 is the largest value of c such that -5 is in the range of f(x) -/
theorem largest_c_for_negative_five_in_range :
  ∀ c : ℝ, (∃ x : ℝ, f c x = -5) ↔ c ≤ 5/4 := by sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_five_in_range_l2153_215361


namespace NUMINAMATH_CALUDE_average_first_14_even_numbers_l2153_215336

theorem average_first_14_even_numbers :
  let first_14_even : List ℕ := List.range 14 |>.map (fun n => 2 * (n + 1))
  (first_14_even.sum / first_14_even.length : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_first_14_even_numbers_l2153_215336


namespace NUMINAMATH_CALUDE_garden_potato_yield_l2153_215385

/-- Calculates the expected potato yield from a rectangular garden --/
theorem garden_potato_yield 
  (length_steps width_steps : ℕ) 
  (step_length : ℝ) 
  (planting_ratio : ℝ) 
  (yield_rate : ℝ) :
  length_steps = 10 →
  width_steps = 30 →
  step_length = 3 →
  planting_ratio = 0.9 →
  yield_rate = 3/4 →
  (length_steps : ℝ) * step_length * (width_steps : ℝ) * step_length * planting_ratio * yield_rate = 1822.5 := by
  sorry

end NUMINAMATH_CALUDE_garden_potato_yield_l2153_215385


namespace NUMINAMATH_CALUDE_joe_game_buying_duration_l2153_215318

/-- Calculates the number of months before running out of money given initial amount, monthly spending, and monthly income. -/
def monthsBeforeBroke (initialAmount : ℕ) (monthlySpending : ℕ) (monthlyIncome : ℕ) : ℕ :=
  initialAmount / (monthlySpending - monthlyIncome)

/-- Theorem stating that given the specific conditions, Joe can buy and sell games for 12 months before running out of money. -/
theorem joe_game_buying_duration :
  monthsBeforeBroke 240 50 30 = 12 := by
  sorry

end NUMINAMATH_CALUDE_joe_game_buying_duration_l2153_215318


namespace NUMINAMATH_CALUDE_no_simultaneously_safe_numbers_l2153_215342

def is_p_safe (n p : ℕ) : Prop :=
  ∀ k : ℤ, |n - k * p| ≥ 3

def simultaneously_safe (n : ℕ) : Prop :=
  is_p_safe n 5 ∧ is_p_safe n 7 ∧ is_p_safe n 11

theorem no_simultaneously_safe_numbers : 
  ¬ ∃ n : ℕ, n > 0 ∧ n ≤ 500 ∧ simultaneously_safe n := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneously_safe_numbers_l2153_215342


namespace NUMINAMATH_CALUDE_berry_picking_pattern_l2153_215369

/-- A sequence of 5 numbers where the differences between consecutive terms
    form an arithmetic sequence with a common difference of 2 -/
def BerrySequence (a b c d e : ℕ) : Prop :=
  (c - b) - (b - a) = 2 ∧
  (d - c) - (c - b) = 2 ∧
  (e - d) - (d - c) = 2

theorem berry_picking_pattern (a b c d e : ℕ) :
  BerrySequence a b c d e →
  a = 3 →
  c = 7 →
  d = 12 →
  e = 19 →
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_berry_picking_pattern_l2153_215369


namespace NUMINAMATH_CALUDE_polygon_properties_l2153_215315

/-- Proves that a polygon with n sides, where the sum of interior angles is 5 times
    the sum of exterior angles, has 12 sides and 54 diagonals. -/
theorem polygon_properties (n : ℕ) : 
  (n - 2) * 180 = 5 * 360 → 
  n = 12 ∧ 
  n * (n - 3) / 2 = 54 := by
sorry

end NUMINAMATH_CALUDE_polygon_properties_l2153_215315


namespace NUMINAMATH_CALUDE_max_candies_eaten_l2153_215347

/-- The maximum sum of products of pairs from a set of n elements -/
def maxProductSum (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of initial elements on the board -/
def initialCount : ℕ := 30

/-- The theorem stating the maximum number of candies Karlson could eat -/
theorem max_candies_eaten :
  maxProductSum initialCount = 435 := by
  sorry

end NUMINAMATH_CALUDE_max_candies_eaten_l2153_215347


namespace NUMINAMATH_CALUDE_max_value_of_g_l2153_215310

-- Define the function g(x)
def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

-- State the theorem
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 25 / 8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_g_l2153_215310


namespace NUMINAMATH_CALUDE_investment_problem_l2153_215380

def total_investment : ℝ := 1000
def silver_rate : ℝ := 0.04
def gold_rate : ℝ := 0.06
def years : ℕ := 3
def final_amount : ℝ := 1206.11

def silver_investment (x : ℝ) : ℝ := x * (1 + silver_rate) ^ years
def gold_investment (x : ℝ) : ℝ := (total_investment - x) * (1 + gold_rate) ^ years

theorem investment_problem (x : ℝ) :
  silver_investment x + gold_investment x = final_amount →
  x = 228.14 := by sorry

end NUMINAMATH_CALUDE_investment_problem_l2153_215380


namespace NUMINAMATH_CALUDE_slope_of_OP_l2153_215335

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 2

-- Define the line
def is_on_line (x y k : ℝ) : Prop := x + y = k

-- Define the intersection points
def are_intersection_points (M N : ℝ × ℝ) (k : ℝ) : Prop :=
  is_on_ellipse M.1 M.2 ∧ is_on_ellipse N.1 N.2 ∧
  is_on_line M.1 M.2 k ∧ is_on_line N.1 N.2 k

-- Define the midpoint
def is_midpoint (P M N : ℝ × ℝ) : Prop :=
  P.1 = (M.1 + N.1) / 2 ∧ P.2 = (M.2 + N.2) / 2

-- Theorem statement
theorem slope_of_OP (k : ℝ) (M N P : ℝ × ℝ) :
  are_intersection_points M N k →
  is_midpoint P M N →
  P.2 / P.1 = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_slope_of_OP_l2153_215335


namespace NUMINAMATH_CALUDE_quadratic_roots_l2153_215381

theorem quadratic_roots (a : ℝ) : 
  (3^2 - 2*3 + a = 0) → 
  ((-1)^2 - 2*(-1) + a = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2153_215381


namespace NUMINAMATH_CALUDE_ages_of_peter_and_grace_l2153_215326

/-- Represents a person's age -/
structure Age where
  value : ℕ

/-- Represents the ages of Peter, Jacob, and Grace -/
structure AgeGroup where
  peter : Age
  jacob : Age
  grace : Age

/-- Check if the given ages satisfy the problem conditions -/
def satisfies_conditions (ages : AgeGroup) : Prop :=
  (ages.peter.value - 10 = (ages.jacob.value - 10) / 3) ∧
  (ages.jacob.value = ages.peter.value + 12) ∧
  (ages.grace.value = (ages.peter.value + ages.jacob.value) / 2)

theorem ages_of_peter_and_grace (ages : AgeGroup) 
  (h : satisfies_conditions ages) : 
  ages.peter.value = 16 ∧ ages.grace.value = 22 := by
  sorry

#check ages_of_peter_and_grace

end NUMINAMATH_CALUDE_ages_of_peter_and_grace_l2153_215326


namespace NUMINAMATH_CALUDE_brush_square_ratio_l2153_215377

/-- Given a square with side length s and a brush width w, 
    if the brush covers exactly one-third of the square's area 
    when swept along both diagonals, then the ratio s/w is equal to 2√3 - 2. -/
theorem brush_square_ratio (s w : ℝ) (h : s > 0) (h' : w > 0) : 
  w^2 + ((s - w)^2) / 2 = (1/3) * s^2 → s / w = 2 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_brush_square_ratio_l2153_215377


namespace NUMINAMATH_CALUDE_final_number_after_combinations_l2153_215331

def combineNumbers (a b : ℕ) : ℕ := a * b + a + b

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem final_number_after_combinations : 
  ∀ (permutation : List ℕ), 
    permutation.length = 20 ∧ 
    (∀ n, n ∈ permutation ↔ 1 ≤ n ∧ n ≤ 20) →
    (permutation.foldl combineNumbers 0) = factorial 21 - 1 :=
by sorry

end NUMINAMATH_CALUDE_final_number_after_combinations_l2153_215331


namespace NUMINAMATH_CALUDE_sharp_constant_is_20_l2153_215373

/-- The function # defined for any real number -/
def sharp (C : ℝ) (p : ℝ) : ℝ := 2 * p - C

/-- Theorem stating that the constant in the sharp function is 20 -/
theorem sharp_constant_is_20 : ∃ C : ℝ, 
  (sharp C (sharp C (sharp C 18.25)) = 6) ∧ C = 20 := by
  sorry

end NUMINAMATH_CALUDE_sharp_constant_is_20_l2153_215373


namespace NUMINAMATH_CALUDE_f_extrema_l2153_215382

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 1) * (x + a)

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 1

theorem f_extrema (a : ℝ) (h : f_derivative a (-1) = 0) :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-3/2 : ℝ) 1, f a x ≤ max) ∧ 
    (∃ x ∈ Set.Icc (-3/2 : ℝ) 1, f a x = max) ∧
    (∀ x ∈ Set.Icc (-3/2 : ℝ) 1, min ≤ f a x) ∧ 
    (∃ x ∈ Set.Icc (-3/2 : ℝ) 1, f a x = min) ∧
    max = 6 ∧ min = 13/8 := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_l2153_215382


namespace NUMINAMATH_CALUDE_machine_quality_comparison_l2153_215392

/-- Data for machine production quality --/
structure MachineData where
  first_class : ℕ
  second_class : ℕ

/-- Calculate the frequency of first-class products --/
def frequency (data : MachineData) : ℚ :=
  data.first_class / (data.first_class + data.second_class)

/-- Calculate K² statistic --/
def k_squared (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating the frequencies and significance of difference --/
theorem machine_quality_comparison 
  (machine_a machine_b : MachineData)
  (h_a : machine_a = ⟨150, 50⟩)
  (h_b : machine_b = ⟨120, 80⟩) :
  (frequency machine_a = 3/4) ∧ 
  (frequency machine_b = 3/5) ∧ 
  (k_squared machine_a.first_class machine_a.second_class 
              machine_b.first_class machine_b.second_class > 6635/1000) := by
  sorry

#eval frequency ⟨150, 50⟩
#eval frequency ⟨120, 80⟩
#eval k_squared 150 50 120 80

end NUMINAMATH_CALUDE_machine_quality_comparison_l2153_215392


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2153_215333

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + m = 0) → m ≤ 9/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2153_215333


namespace NUMINAMATH_CALUDE_sum_a_2b_is_zero_l2153_215328

theorem sum_a_2b_is_zero (a b : ℝ) (h : (a^2 + 4*a + 6)*(2*b^2 - 4*b + 7) ≤ 10) : 
  a + 2*b = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_a_2b_is_zero_l2153_215328


namespace NUMINAMATH_CALUDE_circle_properties_l2153_215389

/-- Theorem about a circle's properties given a specific sum of circumference, diameter, and radius -/
theorem circle_properties (r : ℝ) (h : 2 * Real.pi * r + 2 * r + r = 27.84) : 
  2 * r = 6 ∧ Real.pi * r^2 = 28.26 := by
  sorry

#check circle_properties

end NUMINAMATH_CALUDE_circle_properties_l2153_215389


namespace NUMINAMATH_CALUDE_sin_double_angle_with_tan_three_l2153_215321

theorem sin_double_angle_with_tan_three (θ : ℝ) :
  (∃ (x y : ℝ), x > 0 ∧ y = 3 * x ∧ Real.cos θ * x = Real.sin θ * y) →
  Real.sin (2 * θ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_with_tan_three_l2153_215321


namespace NUMINAMATH_CALUDE_intersection_value_l2153_215396

theorem intersection_value (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (3 / a = b) ∧ (a - 1 = b) → 1 / a - 1 / b = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_value_l2153_215396


namespace NUMINAMATH_CALUDE_onion_saute_time_l2153_215332

def calzone_problem (onion_time : ℝ) : Prop :=
  let garlic_pepper_time := (1/4) * onion_time
  let knead_time := 30
  let rest_time := 2 * knead_time
  let assemble_time := (1/10) * (knead_time + rest_time)
  onion_time + garlic_pepper_time + knead_time + rest_time + assemble_time = 124

theorem onion_saute_time :
  ∃ (t : ℝ), calzone_problem t ∧ t = 20 := by
  sorry

end NUMINAMATH_CALUDE_onion_saute_time_l2153_215332


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_unchanged_l2153_215395

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- The origin of the Cartesian coordinate system -/
def origin : CartesianPoint := ⟨0, 0⟩

/-- The coordinates of a point with respect to the origin -/
def coordinatesWrtOrigin (p : CartesianPoint) : CartesianPoint := p

theorem coordinates_wrt_origin_unchanged (p : CartesianPoint) :
  coordinatesWrtOrigin p = p := by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_unchanged_l2153_215395


namespace NUMINAMATH_CALUDE_squared_sum_equals_cube_root_l2153_215398

theorem squared_sum_equals_cube_root (x y : ℝ) 
  (h1 : x^2 - 3*y^2 = 17/x) 
  (h2 : 3*x^2 - y^2 = 23/y) : 
  x^2 + y^2 = 818^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_squared_sum_equals_cube_root_l2153_215398


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2153_215365

/-- The area of an isosceles trapezoid with bases 4x and 3x, and height x, is 7x²/2 -/
theorem isosceles_trapezoid_area (x : ℝ) : 
  let base1 : ℝ := 4 * x
  let base2 : ℝ := 3 * x
  let height : ℝ := x
  let area : ℝ := (base1 + base2) / 2 * height
  area = 7 * x^2 / 2 := by
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2153_215365


namespace NUMINAMATH_CALUDE_BA_equals_AB_l2153_215360

def matrix_2x2 (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![a, b; c, d]

theorem BA_equals_AB (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B)
  (h2 : A * B = matrix_2x2 5 2 (-2) 4) :
  B * A = matrix_2x2 5 2 (-2) 4 := by
  sorry

end NUMINAMATH_CALUDE_BA_equals_AB_l2153_215360


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2153_215371

theorem isosceles_triangle_perimeter (equilateral_perimeter : ℝ) (isosceles_base : ℝ) : 
  equilateral_perimeter = 45 → 
  isosceles_base = 10 → 
  ∃ (isosceles_side : ℝ), 
    isosceles_side = equilateral_perimeter / 3 ∧ 
    2 * isosceles_side + isosceles_base = 40 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2153_215371


namespace NUMINAMATH_CALUDE_chloe_recycled_28_pounds_l2153_215359

/-- Represents the recycling scenario with Chloe and her friends -/
structure RecyclingScenario where
  pounds_per_point : ℕ
  friends_recycled : ℕ
  total_points : ℕ

/-- Calculates the amount of paper Chloe recycled given the recycling scenario -/
def chloe_recycled (scenario : RecyclingScenario) : ℕ :=
  scenario.pounds_per_point * scenario.total_points - scenario.friends_recycled

/-- Theorem stating that Chloe recycled 28 pounds given the specific scenario -/
theorem chloe_recycled_28_pounds : 
  let scenario : RecyclingScenario := {
    pounds_per_point := 6,
    friends_recycled := 2,
    total_points := 5
  }
  chloe_recycled scenario = 28 := by
  sorry

end NUMINAMATH_CALUDE_chloe_recycled_28_pounds_l2153_215359


namespace NUMINAMATH_CALUDE_print_gift_wrap_price_l2153_215306

/-- The price of print gift wrap per roll -/
def print_price : ℝ := 6

/-- The price of solid color gift wrap per roll -/
def solid_price : ℝ := 4

/-- The total number of rolls sold -/
def total_rolls : ℕ := 480

/-- The total amount of money collected -/
def total_money : ℝ := 2340

/-- The number of print rolls sold -/
def print_rolls : ℕ := 210

theorem print_gift_wrap_price :
  print_price * print_rolls + solid_price * (total_rolls - print_rolls) = total_money :=
sorry

end NUMINAMATH_CALUDE_print_gift_wrap_price_l2153_215306


namespace NUMINAMATH_CALUDE_total_blocks_l2153_215356

theorem total_blocks (red : ℕ) (yellow : ℕ) (green : ℕ) (blue : ℕ) (orange : ℕ) (purple : ℕ)
  (h1 : red = 24)
  (h2 : yellow = red + 8)
  (h3 : green = yellow - 10)
  (h4 : blue = 2 * green)
  (h5 : orange = blue + 15)
  (h6 : purple = red + orange - 7) :
  red + yellow + green + blue + orange + purple = 257 := by
  sorry

end NUMINAMATH_CALUDE_total_blocks_l2153_215356
