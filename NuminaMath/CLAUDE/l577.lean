import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_value_at_2_l577_57733

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The properties of the quadratic function -/
structure QuadraticProperties (a b c : ℝ) : Prop where
  max_value : ∃ (y : ℝ), ∀ (x : ℝ), f a b c x ≤ y ∧ f a b c (-2) = y
  max_is_10 : f a b c (-2) = 10
  passes_through : f a b c 0 = -6

theorem quadratic_value_at_2 {a b c : ℝ} (h : QuadraticProperties a b c) : 
  f a b c 2 = -54 := by
  sorry

#check quadratic_value_at_2

end NUMINAMATH_CALUDE_quadratic_value_at_2_l577_57733


namespace NUMINAMATH_CALUDE_max_side_length_11_l577_57739

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  perimeter_24 : a + b + c = 24
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The maximum length of any side in a triangle with integer side lengths and perimeter 24 is 11 -/
theorem max_side_length_11 (t : Triangle) : t.a ≤ 11 ∧ t.b ≤ 11 ∧ t.c ≤ 11 := by
  sorry

end NUMINAMATH_CALUDE_max_side_length_11_l577_57739


namespace NUMINAMATH_CALUDE_inequality_holds_for_all_real_x_l577_57785

theorem inequality_holds_for_all_real_x : ∀ x : ℝ, 2^(Real.sin x)^2 + 2^(Real.cos x)^2 ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_all_real_x_l577_57785


namespace NUMINAMATH_CALUDE_negation_of_proposition_l577_57734

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a^2 + b^2 = 4 → a ≥ 2*b) ↔
  (∃ (a b : ℝ), a^2 + b^2 = 4 ∧ a < 2*b) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l577_57734


namespace NUMINAMATH_CALUDE_y_minus_x_values_l577_57799

theorem y_minus_x_values (x y : ℝ) 
  (h1 : |x + 1| = 3)
  (h2 : |y| = 5)
  (h3 : -y/x > 0) :
  y - x = -7 ∨ y - x = 9 := by
sorry

end NUMINAMATH_CALUDE_y_minus_x_values_l577_57799


namespace NUMINAMATH_CALUDE_no_solution_iff_a_equals_two_l577_57787

theorem no_solution_iff_a_equals_two (a : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (a * x) / (x - 1) + 3 / (1 - x) ≠ 2) ↔ a = 2 := by
sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_equals_two_l577_57787


namespace NUMINAMATH_CALUDE_triangle_area_l577_57747

theorem triangle_area (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  a = Real.sqrt 2 → 
  A = π / 4 → 
  B = π / 3 → 
  C = π - A - B →
  S = (1 / 2) * a * b * Real.sin C →
  S = (3 + Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l577_57747


namespace NUMINAMATH_CALUDE_inequality_proof_l577_57757

theorem inequality_proof (x y : ℝ) : 2^(-Real.cos x^2) + 2^(-Real.sin x^2) ≥ Real.sin y + Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l577_57757


namespace NUMINAMATH_CALUDE_tank_fill_level_l577_57793

theorem tank_fill_level (tank_capacity : ℚ) (added_amount : ℚ) (final_fraction : ℚ) 
  (h1 : tank_capacity = 42)
  (h2 : added_amount = 7)
  (h3 : final_fraction = 9/10)
  (h4 : (final_fraction * tank_capacity) = (added_amount + (initial_fraction * tank_capacity))) :
  initial_fraction = 733/1000 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_level_l577_57793


namespace NUMINAMATH_CALUDE_total_cookies_l577_57768

def num_bags : ℕ := 37
def cookies_per_bag : ℕ := 19

theorem total_cookies : num_bags * cookies_per_bag = 703 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l577_57768


namespace NUMINAMATH_CALUDE_f_g_f_3_equals_630_l577_57786

def f (x : ℝ) : ℝ := 5 * x + 5

def g (x : ℝ) : ℝ := 6 * x + 5

theorem f_g_f_3_equals_630 : f (g (f 3)) = 630 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_3_equals_630_l577_57786


namespace NUMINAMATH_CALUDE_rons_height_l577_57717

/-- Proves that Ron's height is 13 feet given the water depth and its relation to Ron's height -/
theorem rons_height (water_depth : ℝ) (h1 : water_depth = 208) 
  (h2 : ∃ (rons_height : ℝ), water_depth = 16 * rons_height) : 
  ∃ (rons_height : ℝ), rons_height = 13 := by
  sorry

end NUMINAMATH_CALUDE_rons_height_l577_57717


namespace NUMINAMATH_CALUDE_non_red_cubes_count_total_small_cubes_correct_l577_57728

/-- Represents the number of small cubes without red faces in a 6x6x6 cube with three faces painted red -/
def non_red_cubes : Set ℕ :=
  {n : ℕ | n = 120 ∨ n = 125}

/-- The main theorem stating that the number of non-red cubes is either 120 or 125 -/
theorem non_red_cubes_count :
  ∀ n : ℕ, n ∈ non_red_cubes ↔ (n = 120 ∨ n = 125) :=
by
  sorry

/-- The cube is 6x6x6 -/
def cube_size : ℕ := 6

/-- The number of small cubes the large cube is cut into -/
def total_small_cubes : ℕ := 216

/-- The number of faces painted red -/
def painted_faces : ℕ := 3

/-- The size of each small cube -/
def small_cube_size : ℕ := 1

/-- Theorem stating that the total number of small cubes is correct -/
theorem total_small_cubes_correct :
  cube_size ^ 3 = total_small_cubes :=
by
  sorry

end NUMINAMATH_CALUDE_non_red_cubes_count_total_small_cubes_correct_l577_57728


namespace NUMINAMATH_CALUDE_M_intersect_N_l577_57796

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}

theorem M_intersect_N : M ∩ N = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l577_57796


namespace NUMINAMATH_CALUDE_rectangle_to_square_l577_57773

theorem rectangle_to_square (k : ℕ) (h1 : k > 5) : 
  (∃ (n : ℕ), k * (k - 5) = n^2) → k * (k - 5) = 6^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l577_57773


namespace NUMINAMATH_CALUDE_xy_minus_two_equals_negative_one_l577_57737

theorem xy_minus_two_equals_negative_one 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : (Real.sqrt (x^2 + 1) - x + 1) * (Real.sqrt (y^2 + 1) - y + 1) = 2) : 
  x * y - 2 = -1 := by
sorry

end NUMINAMATH_CALUDE_xy_minus_two_equals_negative_one_l577_57737


namespace NUMINAMATH_CALUDE_complex_multiplication_l577_57794

theorem complex_multiplication (i : ℂ) :
  i^2 = -1 →
  (3 - 4*i) * (-6 + 2*i) = -10 + 30*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l577_57794


namespace NUMINAMATH_CALUDE_room_length_proof_l577_57720

/-- Given the width, total cost, and rate of paving a room's floor, 
    prove that the length of the room is 5.5 meters. -/
theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) 
    (h1 : width = 3.75)
    (h2 : total_cost = 20625)
    (h3 : paving_rate = 1000) : 
  (total_cost / paving_rate) / width = 5.5 := by
  sorry

#check room_length_proof

end NUMINAMATH_CALUDE_room_length_proof_l577_57720


namespace NUMINAMATH_CALUDE_max_pairs_sum_l577_57761

theorem max_pairs_sum (n : ℕ) (h : n = 3009) :
  let S := Finset.range n
  ∃ (k : ℕ) (f : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ f → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ f → q ∈ f → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ f → q ∈ f → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ f → p.1 + p.2 ≤ n + 1) ∧
    f.card = k ∧
    k = 1203 ∧
    (∀ (m : ℕ) (g : Finset (ℕ × ℕ)),
      (∀ (p : ℕ × ℕ), p ∈ g → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) →
      (∀ (p q : ℕ × ℕ), p ∈ g → q ∈ g → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
      (∀ (p q : ℕ × ℕ), p ∈ g → q ∈ g → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
      (∀ (p : ℕ × ℕ), p ∈ g → p.1 + p.2 ≤ n + 1) →
      g.card = m →
      m ≤ k) :=
by
  sorry

end NUMINAMATH_CALUDE_max_pairs_sum_l577_57761


namespace NUMINAMATH_CALUDE_find_unknown_number_l577_57778

theorem find_unknown_number : ∃ x : ℝ, (213 * 16 = 3408) ∧ (1.6 * x = 3.408) → x = 2.13 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_number_l577_57778


namespace NUMINAMATH_CALUDE_tony_saturday_sandwiches_l577_57755

/-- The number of sandwiches Tony made on Saturday -/
def sandwiches_on_saturday (
  slices_per_sandwich : ℕ)
  (days_in_week : ℕ)
  (initial_slices : ℕ)
  (remaining_slices : ℕ)
  (sandwiches_per_day : ℕ) : ℕ :=
  ((initial_slices - remaining_slices) - (days_in_week - 1) * sandwiches_per_day * slices_per_sandwich) / slices_per_sandwich

theorem tony_saturday_sandwiches :
  sandwiches_on_saturday 2 6 22 6 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tony_saturday_sandwiches_l577_57755


namespace NUMINAMATH_CALUDE_real_roots_condition_l577_57711

theorem real_roots_condition (x : ℝ) :
  (∃ y : ℝ, y^2 + 5*x*y + 2*x + 9 = 0) ↔ (x ≤ -0.6 ∨ x ≥ 0.92) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_condition_l577_57711


namespace NUMINAMATH_CALUDE_some_employees_not_managers_l577_57743

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Employee : U → Prop)
variable (Manager : U → Prop)
variable (Punctual : U → Prop)
variable (Shareholder : U → Prop)

-- State the theorem
theorem some_employees_not_managers
  (h1 : ∃ x, Employee x ∧ ¬Punctual x)
  (h2 : ∀ x, Manager x → Punctual x)
  (h3 : ∃ x, Manager x ∧ Shareholder x) :
  ∃ x, Employee x ∧ ¬Manager x :=
sorry

end NUMINAMATH_CALUDE_some_employees_not_managers_l577_57743


namespace NUMINAMATH_CALUDE_triangle_problem_l577_57715

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem statement --/
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * Real.sin t.B * Real.sin t.C + Real.cos t.B + 2 * Real.cos (t.B + t.C) = 0)
  (h2 : Real.sin t.B ≠ 1)
  (h3 : 5 * Real.sin t.B = 3 * Real.sin t.A)
  (h4 : (1/2) * t.a * t.b * Real.sin t.C = 15 * Real.sqrt 3 / 4) :
  t.C = 2 * Real.pi / 3 ∧ t.a + t.b + t.c = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l577_57715


namespace NUMINAMATH_CALUDE_circle_area_ratio_l577_57741

theorem circle_area_ratio (C D : Real) (r_C r_D : ℝ) :
  (60 / 360) * (2 * Real.pi * r_C) = (40 / 360) * (2 * Real.pi * r_D) →
  (Real.pi * r_C^2) / (Real.pi * r_D^2) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l577_57741


namespace NUMINAMATH_CALUDE_min_side_length_two_triangles_l577_57703

/-- Given two triangles ABC and DBC sharing side BC, with known side lengths,
    prove that the minimum integral length of BC is 16 cm. -/
theorem min_side_length_two_triangles 
  (AB AC DC BD : ℝ) 
  (h_AB : AB = 7)
  (h_AC : AC = 18)
  (h_DC : DC = 10)
  (h_BD : BD = 25) :
  (∃ (BC : ℕ), BC ≥ 16 ∧ ∀ (n : ℕ), n < 16 → 
    (n : ℝ) ≤ AC - AB ∨ (n : ℝ) ≤ BD - DC) :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_two_triangles_l577_57703


namespace NUMINAMATH_CALUDE_profit_is_152_l577_57764

/-- The profit made from selling jerseys -/
def profit_from_jerseys (profit_per_jersey : ℕ) (jerseys_sold : ℕ) : ℕ :=
  profit_per_jersey * jerseys_sold

/-- Theorem: The profit from selling jerseys is $152 -/
theorem profit_is_152 :
  profit_from_jerseys 76 2 = 152 := by
  sorry

end NUMINAMATH_CALUDE_profit_is_152_l577_57764


namespace NUMINAMATH_CALUDE_average_score_is_correct_total_students_is_correct_l577_57795

/-- Calculates the average score given a list of (score, number of students) pairs -/
def averageScore (scores : List (ℚ × ℕ)) : ℚ :=
  let totalScore := scores.foldl (fun acc (score, count) => acc + score * count) 0
  let totalStudents := scores.foldl (fun acc (_, count) => acc + count) 0
  totalScore / totalStudents

/-- The given score distribution -/
def scoreDistribution : List (ℚ × ℕ) :=
  [(100, 10), (95, 20), (85, 40), (70, 40), (60, 20), (55, 10), (45, 10)]

/-- The total number of students -/
def totalStudents : ℕ := 150

/-- Theorem stating that the average score is 75.33 (11300/150) -/
theorem average_score_is_correct :
  averageScore scoreDistribution = 11300 / 150 := by
  sorry

/-- Theorem verifying the total number of students -/
theorem total_students_is_correct :
  (scoreDistribution.foldl (fun acc (_, count) => acc + count) 0) = totalStudents := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_correct_total_students_is_correct_l577_57795


namespace NUMINAMATH_CALUDE_blue_car_fraction_l577_57759

theorem blue_car_fraction (total : ℕ) (black : ℕ) : 
  total = 516 →
  black = 86 →
  (total - (total / 2 + black)) / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_car_fraction_l577_57759


namespace NUMINAMATH_CALUDE_m_eq_n_necessary_not_sufficient_l577_57760

/-- Defines a circle in R^2 --/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b r : ℝ), r > 0 ∧ ∀ (x y : ℝ), f x y = 0 ↔ (x - a)^2 + (y - b)^2 = r^2

/-- The equation mx^2 + ny^2 = 3 --/
def equation (m n : ℝ) (x y : ℝ) : ℝ := m * x^2 + n * y^2 - 3

theorem m_eq_n_necessary_not_sufficient :
  (∀ m n : ℝ, is_circle (equation m n) → m = n) ∧
  (∃ m n : ℝ, m = n ∧ ¬is_circle (equation m n)) :=
sorry

end NUMINAMATH_CALUDE_m_eq_n_necessary_not_sufficient_l577_57760


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l577_57740

/-- The surface area of a cuboid with given dimensions -/
def cuboidSurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

/-- Theorem: The surface area of a cuboid with edges 4 cm, 5 cm, and 6 cm is 148 cm² -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 4 5 6 = 148 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l577_57740


namespace NUMINAMATH_CALUDE_angie_drinks_three_cups_per_day_l577_57718

/-- Represents the number of cups of coffee per pound -/
def cupsPerPound : ℕ := 40

/-- Represents the number of pounds of coffee bought -/
def poundsBought : ℕ := 3

/-- Represents the number of days the coffee lasts -/
def daysLasting : ℕ := 40

/-- Calculates the number of cups of coffee Angie drinks per day -/
def cupsPerDay : ℕ := (poundsBought * cupsPerPound) / daysLasting

/-- Theorem stating that Angie drinks 3 cups of coffee per day -/
theorem angie_drinks_three_cups_per_day : cupsPerDay = 3 := by
  sorry

end NUMINAMATH_CALUDE_angie_drinks_three_cups_per_day_l577_57718


namespace NUMINAMATH_CALUDE_blue_hat_cost_l577_57775

/-- Proves that the cost of each blue hat is $6 given the conditions of the hat purchase problem -/
theorem blue_hat_cost (total_hats : ℕ) (green_hats : ℕ) (green_cost : ℕ) (total_cost : ℕ) : 
  total_hats = 85 →
  green_hats = 40 →
  green_cost = 7 →
  total_cost = 550 →
  (total_cost - green_hats * green_cost) / (total_hats - green_hats) = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_hat_cost_l577_57775


namespace NUMINAMATH_CALUDE_angle_C_sides_a_b_max_area_l577_57742

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 2 ∧ Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A

-- Theorem 1: Angle C
theorem angle_C (t : Triangle) (h : triangle_conditions t) : t.C = π/3 := by
  sorry

-- Theorem 2: Sides a and b
theorem sides_a_b (t : Triangle) (h : triangle_conditions t) 
  (area : (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3) : 
  t.a = 2 ∧ t.b = 2 := by
  sorry

-- Theorem 3: Maximum area
theorem max_area (t : Triangle) (h : triangle_conditions t) :
  ∀ (s : Triangle), triangle_conditions s → 
    (1/2) * s.a * s.b * Real.sin s.C ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_sides_a_b_max_area_l577_57742


namespace NUMINAMATH_CALUDE_system_solution_conditions_l577_57702

/-- Given a system of equations:
    a x + b y = c z
    a √(1 - x²) + b √(1 - y²) = c √(1 - z²)
    where x, y, z are real variables,
    prove that for a real solution to exist:
    1. a, b, c must satisfy the triangle inequalities
    2. At least one of a or b must have the same sign as c -/
theorem system_solution_conditions (a b c : ℝ) : 
  (∃ x y z : ℝ, a * x + b * y = c * z ∧ 
   a * Real.sqrt (1 - x^2) + b * Real.sqrt (1 - y^2) = c * Real.sqrt (1 - z^2)) →
  (abs a ≤ abs b + abs c ∧ abs b ≤ abs a + abs c ∧ abs c ≤ abs a + abs b) ∧
  (a * c ≥ 0 ∨ b * c ≥ 0) := by
sorry

end NUMINAMATH_CALUDE_system_solution_conditions_l577_57702


namespace NUMINAMATH_CALUDE_probability_three_hearts_is_correct_l577_57708

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of hearts in a standard deck -/
def heartsCount : ℕ := 13

/-- Calculates the probability of drawing three hearts in a row from a standard deck without replacement -/
def probabilityThreeHearts : ℚ :=
  (heartsCount : ℚ) / deckSize *
  ((heartsCount - 1) : ℚ) / (deckSize - 1) *
  ((heartsCount - 2) : ℚ) / (deckSize - 2)

/-- Theorem stating that the probability of drawing three hearts in a row is 26/2025 -/
theorem probability_three_hearts_is_correct :
  probabilityThreeHearts = 26 / 2025 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_hearts_is_correct_l577_57708


namespace NUMINAMATH_CALUDE_inequality_proof_l577_57724

theorem inequality_proof (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b * c > 0) :
  a * b + b * c + c * a < Real.sqrt (a * b * c) / 2 + 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l577_57724


namespace NUMINAMATH_CALUDE_smallest_m_is_671_l577_57732

def is_valid (m n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    a = 2015^(3*m+1) ∧
    b = 2015^(6*n+2) ∧
    a < b ∧
    a % 10^2014 = b % 10^2014

theorem smallest_m_is_671 :
  (∃ (n : ℕ), is_valid 671 n) ∧
  (∀ (m : ℕ), m < 671 → ¬∃ (n : ℕ), is_valid m n) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_671_l577_57732


namespace NUMINAMATH_CALUDE_intersection_complement_A_with_B_l577_57798

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 4)}
def B : Set ℝ := {x | -1 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 0}

theorem intersection_complement_A_with_B :
  (Set.univ \ A) ∩ B = Set.Icc (0 : ℝ) (1/2) := by sorry

end NUMINAMATH_CALUDE_intersection_complement_A_with_B_l577_57798


namespace NUMINAMATH_CALUDE_number_of_proper_subsets_of_union_l577_57766

def A : Finset Nat := {2, 3}
def B : Finset Nat := {2, 4, 5}

theorem number_of_proper_subsets_of_union : (Finset.powerset (A ∪ B)).card - 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_proper_subsets_of_union_l577_57766


namespace NUMINAMATH_CALUDE_initial_fee_value_l577_57780

/-- The initial fee of the first car rental plan -/
def initial_fee : ℝ := sorry

/-- The cost per mile for the first car rental plan -/
def cost_per_mile_plan1 : ℝ := 0.40

/-- The cost per mile for the second car rental plan -/
def cost_per_mile_plan2 : ℝ := 0.60

/-- The number of miles driven for which both plans cost the same -/
def miles_driven : ℝ := 325

theorem initial_fee_value :
  initial_fee = 65 :=
by
  have h1 : initial_fee + cost_per_mile_plan1 * miles_driven = cost_per_mile_plan2 * miles_driven :=
    sorry
  sorry

end NUMINAMATH_CALUDE_initial_fee_value_l577_57780


namespace NUMINAMATH_CALUDE_tangent_line_parallel_point_l577_57736

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^4 - x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 4*x^3 - 1

-- Theorem statement
theorem tangent_line_parallel_point (P : ℝ × ℝ) :
  f' P.1 = 3 →  -- The slope of the tangent line at P is 3
  f P.1 = P.2 → -- P lies on the curve f(x)
  P = (1, 0) := by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_point_l577_57736


namespace NUMINAMATH_CALUDE_m_properties_l577_57782

/-- The smallest positive integer with both 5 and 6 as digits, each appearing at least once, and divisible by both 3 and 7 -/
def m : ℕ := 5665665660

/-- Checks if a natural number contains both 5 and 6 as digits -/
def has_five_and_six (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a * 10 + 5 ∧ ∃ (c d : ℕ), n = c * 10 + 6

/-- Returns the last four digits of a natural number -/
def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem m_properties :
  has_five_and_six m ∧ 
  m % 3 = 0 ∧ 
  m % 7 = 0 ∧ 
  ∀ k < m, ¬(has_five_and_six k ∧ k % 3 = 0 ∧ k % 7 = 0) ∧
  last_four_digits m = 5660 :=
sorry

end NUMINAMATH_CALUDE_m_properties_l577_57782


namespace NUMINAMATH_CALUDE_intersection_perpendicular_line_l577_57726

-- Define the lines l1, l2, and l3
def l1 (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def l2 (x y : ℝ) : Prop := 2*x + y + 2 = 0
def l3 (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (x, y) where
  x := -2
  y := 2

-- Define perpendicularity of lines
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem intersection_perpendicular_line :
  ∃ (m b : ℝ), 
    (l1 P.1 P.2) ∧ 
    (l2 P.1 P.2) ∧ 
    (perpendicular m ((1 : ℝ) / 2)) ∧ 
    (∀ (x y : ℝ), m * x + y + b = 0 ↔ 2 * x + y + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_line_l577_57726


namespace NUMINAMATH_CALUDE_solutions_count_l577_57738

/-- The number of solutions to the Diophantine equation 3x + 5y = 805 where x and y are positive integers -/
def num_solutions : ℕ :=
  (Finset.filter (fun t : ℕ => 265 - 5 * t > 0 ∧ 2 + 3 * t > 0) (Finset.range 53)).card

theorem solutions_count : num_solutions = 53 := by
  sorry

end NUMINAMATH_CALUDE_solutions_count_l577_57738


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l577_57751

theorem perfect_square_trinomial_m_value (m : ℝ) :
  (∃ (a b : ℝ), ∀ y : ℝ, y^2 - m*y + 9 = (a*y + b)^2) →
  m = 6 ∨ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l577_57751


namespace NUMINAMATH_CALUDE_probability_of_selection_l577_57789

/-- The number of shirts in the drawer -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the drawer -/
def num_shorts : ℕ := 7

/-- The number of pairs of socks in the drawer -/
def num_socks : ℕ := 8

/-- The total number of articles of clothing -/
def total_items : ℕ := num_shirts + num_shorts + num_socks

/-- The number of items to be selected -/
def items_selected : ℕ := 5

/-- The probability of selecting two shirts, two pairs of shorts, and one pair of socks -/
theorem probability_of_selection : 
  (Nat.choose num_shirts 2 * Nat.choose num_shorts 2 * Nat.choose num_socks 1) / 
  Nat.choose total_items items_selected = 280 / 2261 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_l577_57789


namespace NUMINAMATH_CALUDE_bridge_length_l577_57719

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 250 →
  crossing_time = 32 →
  train_speed_kmh = 45 →
  ∃ (bridge_length : ℝ), bridge_length = 150 := by
  sorry


end NUMINAMATH_CALUDE_bridge_length_l577_57719


namespace NUMINAMATH_CALUDE_parallelepiped_with_surroundings_volume_l577_57714

/-- The volume of a set consisting of a rectangular parallelepiped and its surrounding elements -/
theorem parallelepiped_with_surroundings_volume 
  (l w h : ℝ) 
  (hl : l = 2) 
  (hw : w = 3) 
  (hh : h = 6) 
  (r : ℝ) 
  (hr : r = 1) : 
  (l * w * h) + 
  (2 * (r * w * h + r * l * h + r * l * w)) + 
  (π * r^2 * (l + w + h)) + 
  (2 * π * r^3) = 
  108 + (41/3) * π := by sorry

end NUMINAMATH_CALUDE_parallelepiped_with_surroundings_volume_l577_57714


namespace NUMINAMATH_CALUDE_opera_house_earnings_l577_57735

/-- Opera house earnings calculation -/
theorem opera_house_earnings : 
  let total_rows : ℕ := 150
  let section_a_rows : ℕ := 50
  let section_b_rows : ℕ := 60
  let section_c_rows : ℕ := 40
  let seats_per_row : ℕ := 10
  let section_a_price : ℕ := 20
  let section_b_price : ℕ := 15
  let section_c_price : ℕ := 10
  let convenience_fee : ℕ := 3
  let section_a_occupancy : ℚ := 9/10
  let section_b_occupancy : ℚ := 3/4
  let section_c_occupancy : ℚ := 7/10

  let section_a_earnings := (section_a_price + convenience_fee) * (section_a_rows * seats_per_row : ℕ) * section_a_occupancy
  let section_b_earnings := (section_b_price + convenience_fee) * (section_b_rows * seats_per_row : ℕ) * section_b_occupancy
  let section_c_earnings := (section_c_price + convenience_fee) * (section_c_rows * seats_per_row : ℕ) * section_c_occupancy

  let total_earnings := section_a_earnings + section_b_earnings + section_c_earnings

  total_earnings = 22090 := by sorry

end NUMINAMATH_CALUDE_opera_house_earnings_l577_57735


namespace NUMINAMATH_CALUDE_curve_E_and_min_distance_l577_57792

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 5 = 0

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 91 = 0

/-- Definition of curve E as the locus of centers of moving circles -/
def E (x y : ℝ) : Prop := ∃ (r : ℝ), 
  (∀ (x₁ y₁ : ℝ), C₁ x₁ y₁ → (x - x₁)^2 + (y - y₁)^2 = (r + 2)^2) ∧
  (∀ (x₂ y₂ : ℝ), C₂ x₂ y₂ → (x - x₂)^2 + (y - y₂)^2 = (10 - r)^2)

/-- The right focus of curve E -/
def F : ℝ × ℝ := (3, 0)

/-- Theorem stating the equation of curve E and the minimum value of |PO|²+|PF|² -/
theorem curve_E_and_min_distance : 
  (∀ x y : ℝ, E x y ↔ x^2/36 + y^2/27 = 1) ∧
  (∃ min : ℝ, min = 45 ∧ 
    ∀ x y : ℝ, E x y → x^2 + y^2 + (x - F.1)^2 + (y - F.2)^2 ≥ min) :=
sorry

end NUMINAMATH_CALUDE_curve_E_and_min_distance_l577_57792


namespace NUMINAMATH_CALUDE_mailing_cost_calculation_l577_57723

/-- Calculates the total cost of mailing letters and packages -/
def total_mailing_cost (letter_cost package_cost : ℚ) (num_letters : ℕ) : ℚ :=
  let num_packages := num_letters - 2
  letter_cost * num_letters + package_cost * num_packages

/-- Theorem: Given the conditions, the total mailing cost is $4.49 -/
theorem mailing_cost_calculation :
  total_mailing_cost (37/100) (88/100) 5 = 449/100 := by
sorry

end NUMINAMATH_CALUDE_mailing_cost_calculation_l577_57723


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l577_57707

theorem fraction_to_decimal : (17 : ℚ) / 625 = 0.0272 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l577_57707


namespace NUMINAMATH_CALUDE_f_composition_nonnegative_iff_a_geq_three_l577_57749

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x + 1

theorem f_composition_nonnegative_iff_a_geq_three (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ 0) ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_nonnegative_iff_a_geq_three_l577_57749


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_parallel_l577_57790

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- Given vectors a = (m,1) and b = (n,1), m/n = 1 is a sufficient but not necessary condition for a ∥ b -/
theorem sufficient_not_necessary_parallel (m n : ℝ) :
  (m / n = 1 → parallel (m, 1) (n, 1)) ∧
  ¬(parallel (m, 1) (n, 1) → m / n = 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_parallel_l577_57790


namespace NUMINAMATH_CALUDE_smallest_n_for_m_disjoint_monochromatic_edges_l577_57701

/-- A two-coloring of a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- Predicate for m pairwise disjoint edges of the same color -/
def HasMDisjointMonochromaticEdges (n m : ℕ) (coloring : TwoColoring n) : Prop :=
  ∃ (edges : Fin m → Fin n × Fin n),
    (∀ i : Fin m, (edges i).1 ≠ (edges i).2) ∧
    (∀ i j : Fin m, i ≠ j → (edges i).1 ≠ (edges j).1 ∧ (edges i).1 ≠ (edges j).2 ∧
                            (edges i).2 ≠ (edges j).1 ∧ (edges i).2 ≠ (edges j).2) ∧
    (∃ c : Fin 2, ∀ i : Fin m, coloring (edges i).1 (edges i).2 = c)

/-- The main theorem -/
theorem smallest_n_for_m_disjoint_monochromatic_edges (m : ℕ) (hm : m > 0) :
  (∀ n : ℕ, n ≥ 3 * m - 1 → ∀ coloring : TwoColoring n, HasMDisjointMonochromaticEdges n m coloring) ∧
  (∀ n : ℕ, n < 3 * m - 1 → ∃ coloring : TwoColoring n, ¬HasMDisjointMonochromaticEdges n m coloring) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_m_disjoint_monochromatic_edges_l577_57701


namespace NUMINAMATH_CALUDE_percentage_of_B_grades_l577_57762

def scores : List Nat := [91, 68, 58, 99, 82, 94, 88, 76, 79, 62, 87, 81, 65, 85, 89, 73, 77, 84, 59, 72]

def is_grade_B (score : Nat) : Bool :=
  85 ≤ score ∧ score ≤ 94

def count_grade_B (scores : List Nat) : Nat :=
  scores.filter is_grade_B |>.length

theorem percentage_of_B_grades :
  (count_grade_B scores : ℚ) / scores.length * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_B_grades_l577_57762


namespace NUMINAMATH_CALUDE_olivers_score_l577_57772

theorem olivers_score (n : ℕ) (avg_24 : ℚ) (avg_25 : ℚ) (oliver_score : ℚ) :
  n = 25 →
  avg_24 = 76 →
  avg_25 = 78 →
  (n - 1) * avg_24 + oliver_score = n * avg_25 →
  oliver_score = 126 := by
  sorry

end NUMINAMATH_CALUDE_olivers_score_l577_57772


namespace NUMINAMATH_CALUDE_sqrt_fraction_equals_half_l577_57753

theorem sqrt_fraction_equals_half : 
  Real.sqrt ((16^6 + 8^8) / (16^3 + 8^9)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equals_half_l577_57753


namespace NUMINAMATH_CALUDE_some_number_is_four_l577_57729

theorem some_number_is_four : ∃ n : ℚ, (27 / n) * 12 - 18 = 3 * 12 + 27 ∧ n = 4 := by sorry

end NUMINAMATH_CALUDE_some_number_is_four_l577_57729


namespace NUMINAMATH_CALUDE_units_digit_of_27_times_36_l577_57797

theorem units_digit_of_27_times_36 : (27 * 36) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_27_times_36_l577_57797


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l577_57769

/-- Represents the number of students to be selected in a stratified sampling -/
def total_sample : ℕ := 45

/-- Represents the total number of male students -/
def male_population : ℕ := 500

/-- Represents the total number of female students -/
def female_population : ℕ := 400

/-- Represents the number of male students selected in the sample -/
def male_sample : ℕ := 25

/-- Calculates the number of female students to be selected in the sample -/
def female_sample : ℕ := (male_sample * female_population) / male_population

/-- Proves that the calculated female sample size maintains the stratified sampling proportion -/
theorem stratified_sampling_proportion :
  female_sample = 20 ∧
  (male_sample : ℚ) / male_population = (female_sample : ℚ) / female_population ∧
  male_sample + female_sample = total_sample :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l577_57769


namespace NUMINAMATH_CALUDE_x_polynomial_equality_l577_57705

theorem x_polynomial_equality (x : ℝ) (h : x + 1/x = 3) :
  x^12 - 7*x^8 + x^4 = 35292*x - 13652 := by
  sorry

end NUMINAMATH_CALUDE_x_polynomial_equality_l577_57705


namespace NUMINAMATH_CALUDE_parallel_condition_not_sufficient_nor_necessary_l577_57713

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Two lines are parallel -/
def parallel_lines (l m : Line3D) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def parallel_line_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

theorem parallel_condition_not_sufficient_nor_necessary 
  (l m : Line3D) (α : Plane3D) 
  (h_diff : l ≠ m) (h_parallel : parallel_lines l m) : 
  (¬ (∀ α, parallel_line_plane l α → parallel_line_plane m α)) ∧ 
  (¬ (∀ α, parallel_line_plane m α → parallel_line_plane l α)) :=
by sorry

end NUMINAMATH_CALUDE_parallel_condition_not_sufficient_nor_necessary_l577_57713


namespace NUMINAMATH_CALUDE_hyperbola_C_properties_l577_57716

/-- Hyperbola C with distance √2 from focus to asymptote -/
structure HyperbolaC where
  b : ℝ
  b_pos : b > 0
  focus_to_asymptote : ∃ (c : ℝ), b * c / Real.sqrt (b^2 + 2) = Real.sqrt 2

/-- Intersection points of line l with hyperbola C -/
structure IntersectionPoints (h : HyperbolaC) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  l_passes_through_2_0 : ∃ (m : ℝ), A.1 = m * A.2 + 2 ∧ B.1 = m * B.2 + 2
  A_on_C : A.1^2 - A.2^2 = 2
  B_on_C : B.1^2 - B.2^2 = 2
  A_B_right_branch : A.1 > 0 ∧ B.1 > 0

/-- Main theorem -/
theorem hyperbola_C_properties (h : HyperbolaC) :
  (∀ (x y : ℝ), x^2/2 - y^2/h.b^2 = 1 ↔ x^2 - y^2 = 2) ∧
  (∀ (i : IntersectionPoints h),
    ∃ (N : ℝ × ℝ), N = (1, 0) ∧
      (i.A.1 - N.1) * (i.B.1 - N.1) + (i.A.2 - N.2) * (i.B.2 - N.2) = -1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_C_properties_l577_57716


namespace NUMINAMATH_CALUDE_smallest_year_after_2010_with_digit_sum_16_l577_57777

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Predicate to check if a year is after 2010 -/
def is_after_2010 (year : ℕ) : Prop :=
  year > 2010

/-- Theorem stating that 2059 is the smallest year after 2010 with digit sum 16 -/
theorem smallest_year_after_2010_with_digit_sum_16 :
  (∀ year : ℕ, is_after_2010 year → sum_of_digits year = 16 → year ≥ 2059) ∧
  (is_after_2010 2059 ∧ sum_of_digits 2059 = 16) :=
sorry

end NUMINAMATH_CALUDE_smallest_year_after_2010_with_digit_sum_16_l577_57777


namespace NUMINAMATH_CALUDE_product_equality_l577_57727

theorem product_equality : 250 * 24.98 * 2.498 * 1250 = 19484012.5 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l577_57727


namespace NUMINAMATH_CALUDE_mabel_katrina_marble_ratio_l577_57779

/-- Prove that Mabel has 5 times as many marbles as Katrina -/
theorem mabel_katrina_marble_ratio : 
  ∀ (amanda katrina mabel : ℕ),
  amanda + 12 = 2 * katrina →
  mabel = 85 →
  mabel = amanda + 63 →
  mabel / katrina = 5 := by
sorry

end NUMINAMATH_CALUDE_mabel_katrina_marble_ratio_l577_57779


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l577_57710

theorem opposite_of_negative_2023 : 
  ∃ x : ℤ, x + (-2023) = 0 ∧ x = 2023 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l577_57710


namespace NUMINAMATH_CALUDE_student_ticket_price_l577_57700

theorem student_ticket_price 
  (total_sales : ℝ)
  (student_ticket_surplus : ℕ)
  (nonstudent_tickets : ℕ)
  (nonstudent_price : ℝ)
  (h1 : total_sales = 10500)
  (h2 : student_ticket_surplus = 250)
  (h3 : nonstudent_tickets = 850)
  (h4 : nonstudent_price = 9) :
  ∃ (student_price : ℝ), 
    student_price = 2.59 ∧ 
    (nonstudent_tickets : ℝ) * nonstudent_price + 
    ((nonstudent_tickets : ℝ) + (student_ticket_surplus : ℝ)) * student_price = total_sales :=
by sorry

end NUMINAMATH_CALUDE_student_ticket_price_l577_57700


namespace NUMINAMATH_CALUDE_anne_cleaning_time_l577_57781

-- Define the cleaning rates
def bruce_rate : ℝ := sorry
def anne_rate : ℝ := sorry

-- Define the conditions
axiom combined_rate : bruce_rate + anne_rate = 1 / 4
axiom doubled_anne_rate : bruce_rate + 2 * anne_rate = 1 / 3

-- Theorem to prove
theorem anne_cleaning_time :
  1 / anne_rate = 12 := by sorry

end NUMINAMATH_CALUDE_anne_cleaning_time_l577_57781


namespace NUMINAMATH_CALUDE_hyperbola_transverse_axis_length_l577_57770

/-- Given a hyperbola with equation x²/a² - y² = 1 where a > 0 and eccentricity = 2,
    prove that the length of its transverse axis is 2√3/3 -/
theorem hyperbola_transverse_axis_length (a : ℝ) (h1 : a > 0) :
  let e := 2  -- eccentricity
  let c := Real.sqrt (a^2 + 1)  -- focal distance
  e = c / a →  -- definition of eccentricity
  2 * a = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_transverse_axis_length_l577_57770


namespace NUMINAMATH_CALUDE_boxes_in_marker_carton_l577_57771

def pencil_cartons : ℕ := 20
def pencil_boxes_per_carton : ℕ := 10
def pencil_box_cost : ℕ := 2
def marker_cartons : ℕ := 10
def marker_carton_cost : ℕ := 4
def total_spent : ℕ := 600

theorem boxes_in_marker_carton :
  ∃ (x : ℕ), 
    x * marker_carton_cost * marker_cartons + 
    pencil_cartons * pencil_boxes_per_carton * pencil_box_cost = 
    total_spent ∧ 
    x = 5 := by sorry

end NUMINAMATH_CALUDE_boxes_in_marker_carton_l577_57771


namespace NUMINAMATH_CALUDE_complex_number_equality_l577_57706

theorem complex_number_equality : (1 + Complex.I)^10 / (1 - Complex.I) = -16 + 16 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l577_57706


namespace NUMINAMATH_CALUDE_mixture_volume_proof_l577_57744

theorem mixture_volume_proof (initial_water_percent : Real) 
                             (final_water_percent : Real)
                             (added_water : Real) :
  initial_water_percent = 0.20 →
  final_water_percent = 0.25 →
  added_water = 10 →
  ∃ (initial_volume : Real),
    initial_volume * initial_water_percent + added_water = 
    final_water_percent * (initial_volume + added_water) ∧
    initial_volume = 150 := by
  sorry

end NUMINAMATH_CALUDE_mixture_volume_proof_l577_57744


namespace NUMINAMATH_CALUDE_rational_reachability_l577_57712

-- Define the operations
def f (x : ℚ) : ℚ := (1 + x) / x
def g (x : ℚ) : ℚ := (1 - x) / x

-- Define a type for sequences of operations
inductive Op
| F : Op
| G : Op

def apply_op (op : Op) (x : ℚ) : ℚ :=
  match op with
  | Op.F => f x
  | Op.G => g x

def apply_ops (ops : List Op) (x : ℚ) : ℚ :=
  ops.foldl (λ acc op => apply_op op acc) x

theorem rational_reachability (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (ops : List Op), apply_ops ops a = b :=
sorry

end NUMINAMATH_CALUDE_rational_reachability_l577_57712


namespace NUMINAMATH_CALUDE_two_positive_real_roots_condition_no_real_roots_necessary_condition_l577_57784

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : Prop := x^2 + (m - 3) * x + m = 0

-- Define the condition for two positive real roots
def has_two_positive_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
  quadratic_equation m x₁ ∧ quadratic_equation m x₂

-- Define the condition for no real roots
def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, ¬(quadratic_equation m x)

-- Theorem for two positive real roots
theorem two_positive_real_roots_condition :
  ∀ m : ℝ, has_two_positive_real_roots m ↔ (0 < m ∧ m ≤ 1) :=
sorry

-- Theorem for necessary condition of no real roots
theorem no_real_roots_necessary_condition :
  ∀ m : ℝ, has_no_real_roots m → m > 1 :=
sorry

end NUMINAMATH_CALUDE_two_positive_real_roots_condition_no_real_roots_necessary_condition_l577_57784


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l577_57746

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_a5 : a 5 = 2)
  (h_a9 : a 9 = 32) :
  a 4 * a 10 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l577_57746


namespace NUMINAMATH_CALUDE_even_function_properties_l577_57752

def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def DecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

def HasMinimumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≥ m) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

def HasMaximumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ m) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

theorem even_function_properties (f : ℝ → ℝ) :
  EvenFunction f →
  IncreasingOn f 3 7 →
  HasMinimumOn f 3 7 2 →
  DecreasingOn f (-7) (-3) ∧ HasMaximumOn f (-7) (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_properties_l577_57752


namespace NUMINAMATH_CALUDE_sum_is_zero_l577_57730

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Three non-zero vectors with specified properties -/
structure ThreeVectors (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (a b c : V)
  (a_nonzero : a ≠ 0)
  (b_nonzero : b ≠ 0)
  (c_nonzero : c ≠ 0)
  (ab_noncollinear : ¬ ∃ (r : ℝ), a = r • b)
  (bc_noncollinear : ¬ ∃ (r : ℝ), b = r • c)
  (ca_noncollinear : ¬ ∃ (r : ℝ), c = r • a)
  (ab_parallel_c : ∃ (m : ℝ), a + b = m • c)
  (bc_parallel_a : ∃ (n : ℝ), b + c = n • a)

/-- The sum of three vectors with the given properties is zero -/
theorem sum_is_zero (v : ThreeVectors V) : v.a + v.b + v.c = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_is_zero_l577_57730


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l577_57748

theorem diophantine_equation_solution (a b : ℕ+) 
  (h1 : (b ^ 619 : ℕ) ∣ (a ^ 1000 : ℕ) + 1)
  (h2 : (a ^ 619 : ℕ) ∣ (b ^ 1000 : ℕ) + 1) :
  a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l577_57748


namespace NUMINAMATH_CALUDE_peanut_butter_cookie_probability_l577_57722

/-- The probability of selecting a peanut butter cookie -/
def peanut_butter_probability (peanut_butter_cookies : ℕ) (chocolate_chip_cookies : ℕ) (lemon_cookies : ℕ) : ℚ :=
  peanut_butter_cookies / (peanut_butter_cookies + chocolate_chip_cookies + lemon_cookies)

theorem peanut_butter_cookie_probability :
  peanut_butter_probability 70 50 20 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_cookie_probability_l577_57722


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l577_57763

theorem consecutive_integers_product (a : ℤ) (h : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7 = 20) :
  (a + 6) * a = 391 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l577_57763


namespace NUMINAMATH_CALUDE_initial_time_calculation_l577_57731

theorem initial_time_calculation (distance : ℝ) (new_speed : ℝ) (time_ratio : ℝ) 
  (h1 : distance = 180)
  (h2 : new_speed = 20)
  (h3 : time_ratio = 3/2) :
  let new_time := distance / new_speed
  let initial_time := new_time * time_ratio
  initial_time = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_initial_time_calculation_l577_57731


namespace NUMINAMATH_CALUDE_stating_distribution_schemes_eq_60_l577_57704

/-- Represents the number of female students -/
def num_female : ℕ := 5

/-- Represents the number of male students -/
def num_male : ℕ := 2

/-- Represents the number of groups -/
def num_groups : ℕ := 2

/-- 
Calculates the number of ways to distribute students into groups
such that each group has at least one female and one male student
-/
def distribution_schemes (f : ℕ) (m : ℕ) (g : ℕ) : ℕ :=
  2 * (2^f - 2)

/-- 
Theorem stating that the number of distribution schemes
for the given problem is 60
-/
theorem distribution_schemes_eq_60 :
  distribution_schemes num_female num_male num_groups = 60 := by
  sorry

end NUMINAMATH_CALUDE_stating_distribution_schemes_eq_60_l577_57704


namespace NUMINAMATH_CALUDE_pythagorean_sum_inequality_l577_57756

theorem pythagorean_sum_inequality (a b c x y z : ℕ) 
  (h1 : a^2 + b^2 = c^2) (h2 : x^2 + y^2 = z^2) :
  (a + x)^2 + (b + y)^2 ≤ (c + z)^2 ∧ 
  ((a + x)^2 + (b + y)^2 = (c + z)^2 ↔ (a * z = c * x ∧ b * z = c * y)) :=
sorry

end NUMINAMATH_CALUDE_pythagorean_sum_inequality_l577_57756


namespace NUMINAMATH_CALUDE_johns_father_age_l577_57709

theorem johns_father_age (john_age father_age : ℕ) : 
  john_age + father_age = 77 →
  father_age = 2 * john_age + 32 →
  john_age = 15 →
  father_age = 62 := by
sorry

end NUMINAMATH_CALUDE_johns_father_age_l577_57709


namespace NUMINAMATH_CALUDE_basketball_match_loss_percentage_l577_57765

theorem basketball_match_loss_percentage 
  (won lost : ℕ) 
  (h1 : won > 0 ∧ lost > 0) 
  (h2 : won / lost = 7 / 3) : 
  (lost : ℚ) / ((won : ℚ) + lost) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_basketball_match_loss_percentage_l577_57765


namespace NUMINAMATH_CALUDE_average_running_distance_l577_57758

def monday_distance : ℝ := 4.2
def tuesday_distance : ℝ := 3.8
def wednesday_distance : ℝ := 3.6
def thursday_distance : ℝ := 4.4
def number_of_days : ℕ := 4

def total_distance : ℝ := monday_distance + tuesday_distance + wednesday_distance + thursday_distance

theorem average_running_distance :
  total_distance / number_of_days = 4 := by sorry

end NUMINAMATH_CALUDE_average_running_distance_l577_57758


namespace NUMINAMATH_CALUDE_product_expansion_l577_57783

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l577_57783


namespace NUMINAMATH_CALUDE_simplify_fraction_l577_57725

theorem simplify_fraction : (88 : ℚ) / 7744 = 1 / 88 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l577_57725


namespace NUMINAMATH_CALUDE_max_ladles_l577_57791

/-- Represents the cost of a pan in dollars -/
def pan_cost : ℕ := 3

/-- Represents the cost of a pot in dollars -/
def pot_cost : ℕ := 5

/-- Represents the cost of a ladle in dollars -/
def ladle_cost : ℕ := 9

/-- Represents the total amount Sarah will spend in dollars -/
def total_spend : ℕ := 100

/-- Represents the minimum number of each item Sarah must buy -/
def min_items : ℕ := 2

theorem max_ladles :
  ∃ (p q l : ℕ),
    p ≥ min_items ∧
    q ≥ min_items ∧
    l ≥ min_items ∧
    pan_cost * p + pot_cost * q + ladle_cost * l = total_spend ∧
    l = 9 ∧
    ∀ (p' q' l' : ℕ),
      p' ≥ min_items →
      q' ≥ min_items →
      l' ≥ min_items →
      pan_cost * p' + pot_cost * q' + ladle_cost * l' = total_spend →
      l' ≤ l :=
by sorry

end NUMINAMATH_CALUDE_max_ladles_l577_57791


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l577_57750

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 ≤ 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x ≤ 3}

-- Theorem for A ∪ B = B
theorem union_condition (a : ℝ) : A ∪ B a = B a ↔ a < 2 := by sorry

-- Theorem for A ∩ B = B
theorem intersection_condition (a : ℝ) : A ∩ B a = B a ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l577_57750


namespace NUMINAMATH_CALUDE_set_operation_result_l577_57767

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 2, 3}

-- Define set B
def B : Set Nat := {3, 4, 5}

-- Theorem to prove
theorem set_operation_result :
  ((U \ A) ∩ B) = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_result_l577_57767


namespace NUMINAMATH_CALUDE_sequence_eventually_periodic_l577_57776

def sequence_next (a : ℕ) (un : ℕ) : ℕ :=
  if un % 2 = 0 then un / 2 else a + un

def is_periodic (s : ℕ → ℕ) (k p : ℕ) : Prop :=
  ∀ n, n ≥ k → s (n + p) = s n

theorem sequence_eventually_periodic (a : ℕ) (h_a : Odd a) (u : ℕ → ℕ) 
  (h_u : ∀ n, u (n + 1) = sequence_next a (u n)) :
  ∃ k p, p > 0 ∧ is_periodic u k p :=
sorry

end NUMINAMATH_CALUDE_sequence_eventually_periodic_l577_57776


namespace NUMINAMATH_CALUDE_walter_age_theorem_l577_57774

/-- Walter's age at the end of 1998 -/
def walter_age_1998 : ℕ := 34

/-- Walter's grandmother's age at the end of 1998 -/
def grandmother_age_1998 : ℕ := 3 * walter_age_1998

/-- The sum of Walter's and his grandmother's birth years -/
def birth_years_sum : ℕ := 3860

/-- Walter's age at the end of 2003 -/
def walter_age_2003 : ℕ := walter_age_1998 + 5

theorem walter_age_theorem :
  (1998 - walter_age_1998) + (1998 - grandmother_age_1998) = birth_years_sum ∧
  walter_age_2003 = 39 := by
  sorry

end NUMINAMATH_CALUDE_walter_age_theorem_l577_57774


namespace NUMINAMATH_CALUDE_power_function_through_point_l577_57745

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, ∀ x : ℝ, f x = x ^ n

-- Theorem statement
theorem power_function_through_point :
  ∀ f : ℝ → ℝ, isPowerFunction f → f 3 = Real.sqrt 3 → f = fun x => Real.sqrt x :=
by sorry


end NUMINAMATH_CALUDE_power_function_through_point_l577_57745


namespace NUMINAMATH_CALUDE_reflection_count_theorem_l577_57788

/-- Represents a semicircular room -/
structure SemicircularRoom where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a light beam -/
structure LightBeam where
  start : ℝ × ℝ
  angle : ℝ

/-- Counts the number of reflections before the light beam returns to its starting point -/
def count_reflections (room : SemicircularRoom) (beam : LightBeam) : ℕ :=
  sorry

/-- The main theorem stating the number of reflections -/
theorem reflection_count_theorem (room : SemicircularRoom) (beam : LightBeam) :
  room.center = (0, 0) →
  room.radius = 1 →
  beam.start = (-1, 0) →
  beam.angle = 46 * π / 180 →
  count_reflections room beam = 65 :=
sorry

end NUMINAMATH_CALUDE_reflection_count_theorem_l577_57788


namespace NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l577_57754

theorem square_area_to_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁^2 / s₂^2 = 16 / 81) :
  (4 * s₁) / (4 * s₂) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l577_57754


namespace NUMINAMATH_CALUDE_gemstones_for_four_sets_l577_57721

/-- Calculates the number of gemstones needed for earring sets -/
def gemstones_needed (num_sets : ℕ) : ℕ :=
  let earrings_per_set := 2
  let magnets_per_earring := 2
  let buttons_per_earring := magnets_per_earring / 2
  let gemstones_per_earring := buttons_per_earring * 3
  num_sets * earrings_per_set * gemstones_per_earring

/-- Proves that 4 sets of earrings require 24 gemstones -/
theorem gemstones_for_four_sets : gemstones_needed 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gemstones_for_four_sets_l577_57721
