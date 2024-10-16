import Mathlib

namespace NUMINAMATH_CALUDE_restaurant_bill_share_l353_35343

/-- Calculate each person's share of a restaurant bill with tip -/
theorem restaurant_bill_share 
  (total_bill : ℝ) 
  (num_people : ℕ) 
  (tip_percentage : ℝ) 
  (h1 : total_bill = 211)
  (h2 : num_people = 5)
  (h3 : tip_percentage = 0.15) : 
  (total_bill * (1 + tip_percentage)) / num_people = 48.53 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_share_l353_35343


namespace NUMINAMATH_CALUDE_move_point_theorem_l353_35387

/-- A point in 2D space represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point left by a given distance. -/
def moveLeft (p : Point) (distance : ℝ) : Point :=
  { x := p.x - distance, y := p.y }

/-- Moves a point up by a given distance. -/
def moveUp (p : Point) (distance : ℝ) : Point :=
  { x := p.x, y := p.y + distance }

/-- Theorem stating that moving point P(0, 3) left by 2 units and up by 1 unit results in P₁(-2, 4). -/
theorem move_point_theorem : 
  let P : Point := { x := 0, y := 3 }
  let P₁ : Point := moveUp (moveLeft P 2) 1
  P₁.x = -2 ∧ P₁.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_move_point_theorem_l353_35387


namespace NUMINAMATH_CALUDE_will_money_distribution_l353_35350

theorem will_money_distribution (x : ℚ) :
  let amount1 := 5 * x
  let amount2 := 3 * x
  let amount3 := 2 * x
  amount2 = 42 →
  amount1 + amount2 + amount3 = 140 :=
by sorry

end NUMINAMATH_CALUDE_will_money_distribution_l353_35350


namespace NUMINAMATH_CALUDE_complement_union_M_N_l353_35389

open Set

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def M : Set ℕ := {1,3,5,7}
def N : Set ℕ := {5,6,7}

theorem complement_union_M_N : 
  (U \ (M ∪ N)) = {2,4,8} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l353_35389


namespace NUMINAMATH_CALUDE_point_inside_ellipse_l353_35353

theorem point_inside_ellipse (a : ℝ) : 
  (a^2 / 4 + 1 / 2 < 1) → (-Real.sqrt 2 < a ∧ a < Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_point_inside_ellipse_l353_35353


namespace NUMINAMATH_CALUDE_mendez_family_mean_age_l353_35354

/-- The Mendez family children problem -/
theorem mendez_family_mean_age :
  let ages : List ℝ := [5, 5, 10, 12, 15]
  let mean_age := (ages.sum) / (ages.length : ℝ)
  mean_age = 9.4 := by
sorry

end NUMINAMATH_CALUDE_mendez_family_mean_age_l353_35354


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l353_35332

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the original plywood and its cuts -/
structure Plywood where
  length : ℝ
  width : ℝ
  num_pieces : ℕ

/-- Checks if a rectangle is a valid cut of the plywood -/
def is_valid_cut (p : Plywood) (r : Rectangle) : Prop :=
  p.length * p.width = p.num_pieces * r.length * r.width

theorem plywood_cut_perimeter_difference (p : Plywood) :
  p.length = 6 ∧ p.width = 9 ∧ p.num_pieces = 6 →
  ∃ (max_r min_r : Rectangle),
    is_valid_cut p max_r ∧
    is_valid_cut p min_r ∧
    ∀ (r : Rectangle), is_valid_cut p r →
      perimeter r ≤ perimeter max_r ∧
      perimeter min_r ≤ perimeter r ∧
      perimeter max_r - perimeter min_r = 10 := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l353_35332


namespace NUMINAMATH_CALUDE_contest_ranking_l353_35326

theorem contest_ranking (A B C D : ℝ) 
  (eq1 : B + D = 2*(A + C) - 20)
  (ineq1 : A + 2*C < 2*B + D)
  (ineq2 : D > 2*(B + C)) :
  D > B ∧ B > A ∧ A > C :=
sorry

end NUMINAMATH_CALUDE_contest_ranking_l353_35326


namespace NUMINAMATH_CALUDE_part_one_part_two_l353_35388

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | (x - a) * (x + 1) > 0}
def Q : Set ℝ := {x | |x - 1| ≤ 1}

-- Part 1
theorem part_one : (Set.univ \ P 1) ∪ Q = {x | -1 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : P a ∩ Q = ∅) : a > 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l353_35388


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l353_35394

theorem rectangular_to_polar_conversion :
  let x : ℝ := Real.sqrt 3
  let y : ℝ := -1
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 2 * Real.pi - Real.arctan (1 / Real.sqrt 3)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 2 ∧ θ = 11 * Real.pi / 6 :=
by
  sorry

#check rectangular_to_polar_conversion

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l353_35394


namespace NUMINAMATH_CALUDE_ladybug_leaf_count_l353_35333

theorem ladybug_leaf_count (ladybugs_per_leaf : ℕ) (total_ladybugs : ℕ) (h1 : ladybugs_per_leaf = 139) (h2 : total_ladybugs = 11676) :
  total_ladybugs / ladybugs_per_leaf = 84 := by
  sorry

end NUMINAMATH_CALUDE_ladybug_leaf_count_l353_35333


namespace NUMINAMATH_CALUDE_diagonal_division_ratio_equality_l353_35337

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The orthocenter of a triangle -/
def orthocenter (A B C : Point) : Point :=
  sorry

/-- The ratio in which a line segment is divided by a point -/
def divisionRatio (A B P : Point) : ℝ :=
  sorry

/-- The intersection point of two line segments -/
def intersectionPoint (A B C D : Point) : Point :=
  sorry

/-- Theorem: In convex quadrilaterals ABCD and A'B'C'D', where A', B', C', D' are orthocenters
    of triangles BCD, CDA, DAB, ABC respectively, the corresponding diagonals are divided by 
    the points of intersection in the same ratio -/
theorem diagonal_division_ratio_equality 
  (ABCD : Quadrilateral) 
  (A' B' C' D' : Point) 
  (h_convex : sorry) -- Assume ABCD is convex
  (h_A' : A' = orthocenter ABCD.B ABCD.C ABCD.D)
  (h_B' : B' = orthocenter ABCD.C ABCD.D ABCD.A)
  (h_C' : C' = orthocenter ABCD.D ABCD.A ABCD.B)
  (h_D' : D' = orthocenter ABCD.A ABCD.B ABCD.C) :
  let P := intersectionPoint ABCD.A ABCD.C ABCD.B ABCD.D
  let P' := intersectionPoint A' C' B' D'
  divisionRatio ABCD.A ABCD.C P = divisionRatio A' C' P' ∧
  divisionRatio ABCD.B ABCD.D P = divisionRatio B' D' P' :=
sorry

end NUMINAMATH_CALUDE_diagonal_division_ratio_equality_l353_35337


namespace NUMINAMATH_CALUDE_f_value_l353_35336

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ := Int.ceil x

-- Define the function f
def f (x y : ℚ) : ℚ := x - y * ceiling (x / y)

-- State the theorem
theorem f_value : f (1/3) (-3/7) = -2/21 := by
  sorry

end NUMINAMATH_CALUDE_f_value_l353_35336


namespace NUMINAMATH_CALUDE_inequalities_proof_l353_35398

theorem inequalities_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (a^3 * b > a * b^3) ∧ (a - b/a > b - a/b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l353_35398


namespace NUMINAMATH_CALUDE_smallest_positive_omega_l353_35391

/-- Given a function f(x) = sin(ωx + π/3), if f(x - π/3) = -f(x) for all x, 
    then the smallest positive value of ω is 3. -/
theorem smallest_positive_omega (ω : ℝ) : 
  (∀ x, Real.sin (ω * (x - π/3) + π/3) = -Real.sin (ω * x + π/3)) → 
  (∀ δ > 0, δ < ω → δ ≤ 3) ∧ ω = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_omega_l353_35391


namespace NUMINAMATH_CALUDE_opposites_sum_zero_l353_35316

theorem opposites_sum_zero (a b : ℚ) : a + b = 0 → a = -b := by
  sorry

end NUMINAMATH_CALUDE_opposites_sum_zero_l353_35316


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l353_35306

def i : ℂ := Complex.I

def z : ℂ := 2 * i * (1 + i)

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l353_35306


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l353_35383

theorem tan_double_angle_special_case (α : Real) 
  (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.cos α + Real.sin α = -1/5) : 
  Real.tan (2 * α) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l353_35383


namespace NUMINAMATH_CALUDE_tournament_games_count_l353_35380

/-- Calculates the number of games in a round-robin tournament for a given number of teams -/
def roundRobinGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the number of games in knockout rounds for a given number of teams -/
def knockoutGames (n : ℕ) : ℕ := n - 1

theorem tournament_games_count :
  let totalTeams : ℕ := 32
  let groupCount : ℕ := 8
  let teamsPerGroup : ℕ := 4
  let advancingTeams : ℕ := 2
  
  totalTeams = groupCount * teamsPerGroup →
  
  (groupCount * roundRobinGames teamsPerGroup) +
  (knockoutGames (groupCount * advancingTeams)) = 63 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_count_l353_35380


namespace NUMINAMATH_CALUDE_inequality_proof_l353_35373

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l353_35373


namespace NUMINAMATH_CALUDE_right_triangle_ratio_bound_l353_35381

/-- A non-degenerate right triangle with sides a, b, and c (c being the hypotenuse) -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  right_angle : a^2 + b^2 = c^2
  c_largest : c ≥ a ∧ c ≥ b

/-- The theorem stating that the least upper bound of (a^2 + b^2 + c^2) / a^2 for all non-degenerate right triangles is 4 -/
theorem right_triangle_ratio_bound :
  ∃ N : ℝ, (∀ t : RightTriangle, (t.a^2 + t.b^2 + t.c^2) / t.a^2 ≤ N) ∧
  (∀ ε > 0, ∃ t : RightTriangle, N - ε < (t.a^2 + t.b^2 + t.c^2) / t.a^2) ∧
  N = 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_bound_l353_35381


namespace NUMINAMATH_CALUDE_triangle_side_length_l353_35363

theorem triangle_side_length (a b c : ℝ) (C : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : C = π/3) :
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) → c = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l353_35363


namespace NUMINAMATH_CALUDE_scientific_notation_of_229000_l353_35360

theorem scientific_notation_of_229000 :
  229000 = 2.29 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_229000_l353_35360


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l353_35345

/-- Represents the number of employees in each title category -/
structure EmployeeCount where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  general : ℕ

/-- Represents the number of employees in each title category for a sample -/
structure SampleCount where
  senior : ℕ
  intermediate : ℕ
  general : ℕ

/-- Checks if the sample count is proportional to the employee count -/
def is_proportional_sample (ec : EmployeeCount) (sc : SampleCount) (sample_size : ℕ) : Prop :=
  sc.senior = (ec.senior * sample_size) / ec.total ∧
  sc.intermediate = (ec.intermediate * sample_size) / ec.total ∧
  sc.general = (ec.general * sample_size) / ec.total

theorem stratified_sampling_theorem (ec : EmployeeCount) (sc : SampleCount) (sample_size : ℕ) :
  ec.total = 150 ∧ ec.senior = 15 ∧ ec.intermediate = 45 ∧ ec.general = 90 ∧
  sample_size = 30 ∧
  is_proportional_sample ec sc sample_size →
  sc.senior = 3 ∧ sc.intermediate = 9 ∧ sc.general = 18 :=
by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l353_35345


namespace NUMINAMATH_CALUDE_complex_inequality_l353_35335

open Complex

theorem complex_inequality (x y : ℂ) (z : ℂ) 
  (h1 : abs x = 1) (h2 : abs y = 1)
  (h3 : π / 3 ≤ arg x - arg y) (h4 : arg x - arg y ≤ 5 * π / 3) :
  abs z + abs (z - x) + abs (z - y) ≥ abs (z * x - y) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l353_35335


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l353_35351

/-- Given an arithmetic sequence with 5 terms, first term 13, and last term 49,
    prove that the middle term (3rd term) is 31. -/
theorem arithmetic_sequence_middle_term :
  ∀ (a : ℕ → ℝ),
    (∀ i j, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence
    a 0 = 13 →                                   -- first term is 13
    a 4 = 49 →                                   -- last term is 49
    a 2 = 31 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l353_35351


namespace NUMINAMATH_CALUDE_power_of_36_l353_35313

theorem power_of_36 : (36 : ℝ) ^ (5/2 : ℝ) = 7776 := by
  sorry

end NUMINAMATH_CALUDE_power_of_36_l353_35313


namespace NUMINAMATH_CALUDE_ratio_equivalence_l353_35374

theorem ratio_equivalence (a b : ℚ) (h : 5 * a = 6 * b) :
  (a / b = 6 / 5) ∧ (b / a = 5 / 6) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalence_l353_35374


namespace NUMINAMATH_CALUDE_dot_product_bounds_l353_35399

theorem dot_product_bounds (a b : ℝ) :
  let v : ℝ × ℝ := (a, b)
  let u : ℝ → ℝ × ℝ := fun θ ↦ (Real.cos θ, Real.sin θ)
  ∀ θ, -Real.sqrt (a^2 + b^2) ≤ (v.1 * (u θ).1 + v.2 * (u θ).2) ∧
       (v.1 * (u θ).1 + v.2 * (u θ).2) ≤ Real.sqrt (a^2 + b^2) ∧
       (∃ θ₁, v.1 * (u θ₁).1 + v.2 * (u θ₁).2 = Real.sqrt (a^2 + b^2)) ∧
       (∃ θ₂, v.1 * (u θ₂).1 + v.2 * (u θ₂).2 = -Real.sqrt (a^2 + b^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_dot_product_bounds_l353_35399


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l353_35356

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 2 + a 7 = 12)
  (h_product : a 4 * a 5 = 35) :
  (∀ n : ℕ, a n = 2 * n - 3) ∨ (∀ n : ℕ, a n = 15 - 2 * n) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l353_35356


namespace NUMINAMATH_CALUDE_games_given_to_friend_l353_35340

theorem games_given_to_friend (initial_games : ℕ) (remaining_games : ℕ) 
  (h1 : initial_games = 9) 
  (h2 : remaining_games = 5) : 
  initial_games - remaining_games = 4 := by
  sorry

end NUMINAMATH_CALUDE_games_given_to_friend_l353_35340


namespace NUMINAMATH_CALUDE_sarah_apple_ratio_l353_35393

theorem sarah_apple_ratio : 
  let sarah_apples : ℕ := 45
  let brother_apples : ℕ := 9
  (sarah_apples : ℚ) / brother_apples = 5 := by sorry

end NUMINAMATH_CALUDE_sarah_apple_ratio_l353_35393


namespace NUMINAMATH_CALUDE_log_xy_z_in_terms_of_log_x_z_and_log_y_z_l353_35358

theorem log_xy_z_in_terms_of_log_x_z_and_log_y_z
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.log z / Real.log (x * y) = (Real.log z / Real.log x * Real.log z / Real.log y) /
                                  (Real.log z / Real.log x + Real.log z / Real.log y) :=
by sorry

end NUMINAMATH_CALUDE_log_xy_z_in_terms_of_log_x_z_and_log_y_z_l353_35358


namespace NUMINAMATH_CALUDE_fourth_root_equation_l353_35307

theorem fourth_root_equation (X : ℝ) : 
  (X^5)^(1/4) = 32 * (32^(1/16)) → X = 16 * (2^(1/4)) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_l353_35307


namespace NUMINAMATH_CALUDE_sandbox_volume_calculation_l353_35390

/-- The volume of a rectangular box with given dimensions -/
def sandbox_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

/-- Theorem: The volume of the sandbox is 3,429,000 cubic centimeters -/
theorem sandbox_volume_calculation :
  sandbox_volume 312 146 75 = 3429000 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_volume_calculation_l353_35390


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l353_35301

/-- The Easter egg hunt problem -/
theorem easter_egg_hunt (kevin_eggs bonnie_eggs george_eggs cheryl_eggs : ℕ) 
  (hk : kevin_eggs = 5)
  (hb : bonnie_eggs = 13)
  (hg : george_eggs = 9)
  (hc : cheryl_eggs = 56) :
  cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 := by
sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l353_35301


namespace NUMINAMATH_CALUDE_cookies_problem_l353_35305

theorem cookies_problem (mona jasmine rachel : ℕ) : 
  mona = 20 →
  jasmine < mona →
  rachel = jasmine + 10 →
  mona + jasmine + rachel = 60 →
  jasmine = 15 := by
sorry

end NUMINAMATH_CALUDE_cookies_problem_l353_35305


namespace NUMINAMATH_CALUDE_no_right_triangle_perimeter_twice_hypotenuse_l353_35385

theorem no_right_triangle_perimeter_twice_hypotenuse :
  ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b + Real.sqrt (a^2 + b^2) = 2 * Real.sqrt (a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_no_right_triangle_perimeter_twice_hypotenuse_l353_35385


namespace NUMINAMATH_CALUDE_number_of_baskets_is_one_l353_35372

-- Define the total number of peaches
def total_peaches : ℕ := 10

-- Define the number of red peaches per basket
def red_peaches_per_basket : ℕ := 4

-- Define the number of green peaches per basket
def green_peaches_per_basket : ℕ := 6

-- Theorem to prove
theorem number_of_baskets_is_one :
  let peaches_per_basket := red_peaches_per_basket + green_peaches_per_basket
  total_peaches / peaches_per_basket = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_of_baskets_is_one_l353_35372


namespace NUMINAMATH_CALUDE_advertising_agency_clients_l353_35357

theorem advertising_agency_clients (total : ℕ) (tv radio mag tv_mag tv_radio radio_mag : ℕ) 
  (h_total : total = 180)
  (h_tv : tv = 115)
  (h_radio : radio = 110)
  (h_mag : mag = 130)
  (h_tv_mag : tv_mag = 85)
  (h_tv_radio : tv_radio = 75)
  (h_radio_mag : radio_mag = 95) :
  total = tv + radio + mag - tv_mag - tv_radio - radio_mag + 80 :=
sorry

end NUMINAMATH_CALUDE_advertising_agency_clients_l353_35357


namespace NUMINAMATH_CALUDE_vegetable_project_profit_l353_35395

-- Define the constants
def initial_investment : ℝ := 600000
def first_year_expense : ℝ := 80000
def annual_expense_increase : ℝ := 20000
def annual_income : ℝ := 260000

-- Define the net profit function
def f (n : ℝ) : ℝ := -n^2 + 19*n - 60

-- Define the theorem to prove
theorem vegetable_project_profit (n : ℝ) :
  f n = n * annual_income - 
    (n * first_year_expense + (n * (n - 1) / 2) * annual_expense_increase) - 
    initial_investment / 10000 ∧
  (∀ m : ℝ, m < 5 → f m ≤ 0) ∧
  (∀ m : ℝ, m ≥ 5 → f m > 0) :=
sorry

end NUMINAMATH_CALUDE_vegetable_project_profit_l353_35395


namespace NUMINAMATH_CALUDE_negation_equivalence_l353_35315

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 ≥ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l353_35315


namespace NUMINAMATH_CALUDE_combined_molecular_weight_l353_35355

-- Define atomic weights
def carbon_weight : ℝ := 12.01
def chlorine_weight : ℝ := 35.45
def sulfur_weight : ℝ := 32.07
def fluorine_weight : ℝ := 19.00

-- Define molecular compositions
def ccl4_carbon_count : ℕ := 1
def ccl4_chlorine_count : ℕ := 4
def sf6_sulfur_count : ℕ := 1
def sf6_fluorine_count : ℕ := 6

-- Define number of moles
def ccl4_moles : ℕ := 9
def sf6_moles : ℕ := 5

-- Theorem statement
theorem combined_molecular_weight :
  let ccl4_weight := carbon_weight * ccl4_carbon_count + chlorine_weight * ccl4_chlorine_count
  let sf6_weight := sulfur_weight * sf6_sulfur_count + fluorine_weight * sf6_fluorine_count
  ccl4_weight * ccl4_moles + sf6_weight * sf6_moles = 2114.64 := by
  sorry

end NUMINAMATH_CALUDE_combined_molecular_weight_l353_35355


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l353_35359

-- Problem 1
theorem problem_1 : (Real.sqrt 7 - Real.sqrt 3) * (Real.sqrt 7 + Real.sqrt 3) - (Real.sqrt 6 + Real.sqrt 2)^2 = -4 - 4 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 : (3 * Real.sqrt 12 - 3 * Real.sqrt (1/3) + Real.sqrt 48) / (2 * Real.sqrt 3) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l353_35359


namespace NUMINAMATH_CALUDE_last_day_same_as_fifteenth_day_l353_35368

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a year -/
structure DayInYear where
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- A function to determine the day of the week for any day in the year,
    given the day of the week for the 15th day -/
def dayOfWeekFor (fifteenthDay : DayOfWeek) (dayNumber : Nat) : DayOfWeek :=
  sorry

theorem last_day_same_as_fifteenth_day 
  (year : Nat) 
  (h1 : year = 2005) 
  (h2 : (dayOfWeekFor DayOfWeek.Tuesday 15) = DayOfWeek.Tuesday) 
  (h3 : (dayOfWeekFor DayOfWeek.Tuesday 365) = (dayOfWeekFor DayOfWeek.Tuesday 15)) :
  (dayOfWeekFor DayOfWeek.Tuesday 365) = DayOfWeek.Tuesday := by
  sorry

end NUMINAMATH_CALUDE_last_day_same_as_fifteenth_day_l353_35368


namespace NUMINAMATH_CALUDE_jerry_initial_figures_l353_35312

/-- The number of books on Jerry's shelf -/
def num_books : ℕ := 9

/-- The number of action figures added later -/
def added_figures : ℕ := 7

/-- The difference between action figures and books after adding -/
def difference : ℕ := 3

/-- The initial number of action figures on Jerry's shelf -/
def initial_figures : ℕ := 5

theorem jerry_initial_figures :
  initial_figures + added_figures = num_books + difference := by sorry

end NUMINAMATH_CALUDE_jerry_initial_figures_l353_35312


namespace NUMINAMATH_CALUDE_sequence_representation_l353_35314

theorem sequence_representation (a : ℕ → ℕ) 
  (h_increasing : ∀ k : ℕ, k ≥ 1 → a k < a (k + 1)) :
  ∀ N : ℕ, ∃ m p q x y : ℕ, 
    m > N ∧ 
    p ≠ q ∧ 
    x > 0 ∧ 
    y > 0 ∧ 
    a m = x * a p + y * a q :=
sorry

end NUMINAMATH_CALUDE_sequence_representation_l353_35314


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l353_35364

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product (m₁ m₂ : ℝ) : 
  m₁ * m₂ = -1 ↔ (∃ (x₁ y₁ x₂ y₂ : ℝ), y₁ = m₁ * x₁ ∧ y₂ = m₂ * x₂ ∧ (y₂ - y₁) * (x₂ - x₁) = 0)

/-- The slope of a line ax + y + c = 0 is -a -/
axiom line_slope (a c : ℝ) : ∃ (m : ℝ), m = -a ∧ ∀ (x y : ℝ), a * x + y + c = 0 → y = m * x - c

theorem perpendicular_lines_b_value :
  ∀ (b : ℝ), 
  (∀ (x y : ℝ), 3 * x + y - 5 = 0 → bx + y + 2 = 0 → 
    ∃ (m₁ m₂ : ℝ), (m₁ * m₂ = -1 ∧ 
      (∀ (x₁ y₁ : ℝ), 3 * x₁ + y₁ - 5 = 0 → y₁ = m₁ * x₁ + 5) ∧
      (∀ (x₂ y₂ : ℝ), b * x₂ + y₂ + 2 = 0 → y₂ = m₂ * x₂ - 2))) →
  b = -1/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l353_35364


namespace NUMINAMATH_CALUDE_handshake_theorem_l353_35334

/-- The number of handshakes in a group where each person shakes hands with a fixed number of others -/
def total_handshakes (n : ℕ) (k : ℕ) : ℕ := n * k / 2

/-- Theorem: In a group of 30 people, where each person shakes hands with exactly 3 others, 
    the total number of handshakes is 45 -/
theorem handshake_theorem : 
  total_handshakes 30 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_handshake_theorem_l353_35334


namespace NUMINAMATH_CALUDE_binomial_coefficient_7_4_l353_35370

theorem binomial_coefficient_7_4 : Nat.choose 7 4 = 35 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_7_4_l353_35370


namespace NUMINAMATH_CALUDE_angle_bisector_segments_l353_35349

/-- 
Given a triangle ABC with sides a, b, c, and angle bisectors AD (internal) and AD₁ (external) of angle A,
this theorem proves the relationships between the segments formed by these angle bisectors.
-/
theorem angle_bisector_segments (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let BD := a * c / (b + c)
  let DC := a * b / (b + c)
  let D₁B := a * c / (b - c)
  let D₁C := a * b / (b - c)
  let DD₁ := 2 * a * b * c / (b^2 - c^2)
  (BD = a * c / (b + c)) ∧
  (DC = a * b / (b + c)) ∧
  (D₁B = a * c / (b - c)) ∧
  (D₁C = a * b / (b - c)) ∧
  (DD₁ = 2 * a * b * c / (b^2 - c^2)) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_segments_l353_35349


namespace NUMINAMATH_CALUDE_max_gcd_of_product_600_l353_35367

theorem max_gcd_of_product_600 :
  ∃ (a b : ℕ), a * b = 600 ∧ 
  (∀ (c d : ℕ), c * d = 600 → Nat.gcd a b ≥ Nat.gcd c d) ∧
  Nat.gcd a b = 10 := by
sorry

end NUMINAMATH_CALUDE_max_gcd_of_product_600_l353_35367


namespace NUMINAMATH_CALUDE_infinite_series_sum_l353_35322

/-- The sum of the infinite series ∑(k=1 to ∞) k^2 / 3^k is equal to 6 -/
theorem infinite_series_sum : ∑' k, (k^2 : ℝ) / 3^k = 6 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l353_35322


namespace NUMINAMATH_CALUDE_sector_central_angle_measures_l353_35379

/-- A circular sector with given perimeter and area -/
structure Sector where
  perimeter : ℝ
  area : ℝ

/-- The possible radian measures of the central angle for a sector with given perimeter and area -/
def centralAngleMeasures (s : Sector) : Set ℝ :=
  {α : ℝ | ∃ r : ℝ, r > 0 ∧ 2 * r + α * r = s.perimeter ∧ 1 / 2 * α * r^2 = s.area}

/-- Theorem: For a sector with perimeter 6 and area 2, the central angle measure is either 1 or 4 radians -/
theorem sector_central_angle_measures :
  centralAngleMeasures ⟨6, 2⟩ = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_measures_l353_35379


namespace NUMINAMATH_CALUDE_sum_of_pairwise_quotients_geq_three_halves_l353_35366

theorem sum_of_pairwise_quotients_geq_three_halves 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_pairwise_quotients_geq_three_halves_l353_35366


namespace NUMINAMATH_CALUDE_sodas_sold_in_afternoon_l353_35327

theorem sodas_sold_in_afternoon (morning_sodas : ℕ) (total_sodas : ℕ) 
  (h1 : morning_sodas = 77) 
  (h2 : total_sodas = 96) : 
  total_sodas - morning_sodas = 19 := by
  sorry

end NUMINAMATH_CALUDE_sodas_sold_in_afternoon_l353_35327


namespace NUMINAMATH_CALUDE_divide_8900_by_6_and_4_l353_35310

theorem divide_8900_by_6_and_4 : (8900 / 6) / 4 = 370.8333333333333 := by
  sorry

end NUMINAMATH_CALUDE_divide_8900_by_6_and_4_l353_35310


namespace NUMINAMATH_CALUDE_base_value_l353_35352

theorem base_value (b x y : ℕ) (h1 : b^x * 4^y = 59049) (h2 : x - y = 10) (h3 : x = 10) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_value_l353_35352


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l353_35397

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l353_35397


namespace NUMINAMATH_CALUDE_parallel_condition_l353_35347

/-- Two lines are parallel if and only if their slopes are equal -/
def are_parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ m₁ ≠ 0 ∧ m₂ ≠ 0

/-- The first line: ax - y + 3 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x - y + 3 = 0

/-- The second line: 2x - (a + 1)y + 4 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  2 * x - (a + 1) * y + 4 = 0

/-- a = -2 is a sufficient but not necessary condition for the lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  (a = -2 → are_parallel a (-1) 3 2 (-(a + 1)) 4) ∧
  ¬(are_parallel a (-1) 3 2 (-(a + 1)) 4 → a = -2) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l353_35347


namespace NUMINAMATH_CALUDE_sqrt_sum_equal_product_equal_l353_35331

-- Problem 1
theorem sqrt_sum_equal : Real.sqrt 2 * Real.sqrt 3 + Real.sqrt 24 = 3 * Real.sqrt 6 := by sorry

-- Problem 2
theorem product_equal : (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3) = 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equal_product_equal_l353_35331


namespace NUMINAMATH_CALUDE_arithmetic_computation_l353_35377

theorem arithmetic_computation : 7^2 - 4*5 + 4^3 = 93 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l353_35377


namespace NUMINAMATH_CALUDE_cricket_score_product_l353_35346

def first_ten_scores : List Nat := [11, 6, 7, 5, 12, 8, 3, 10, 9, 4]

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

theorem cricket_score_product :
  ∀ (score_11 score_12 : Nat),
    score_11 < 15 →
    score_12 < 15 →
    is_integer ((List.sum first_ten_scores + score_11) / 11) →
    is_integer ((List.sum first_ten_scores + score_11 + score_12) / 12) →
    score_11 * score_12 = 14 :=
by sorry

end NUMINAMATH_CALUDE_cricket_score_product_l353_35346


namespace NUMINAMATH_CALUDE_range_of_m_l353_35324

-- Define the linear function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x + m + 1

-- Define the condition for the graph passing through the first, second, and fourth quadrants
def passes_through_quadrants (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₄ : ℝ), 
    (x₁ > 0 ∧ f m x₁ > 0) ∧  -- First quadrant
    (x₂ < 0 ∧ f m x₂ > 0) ∧  -- Second quadrant
    (x₄ > 0 ∧ f m x₄ < 0)    -- Fourth quadrant

-- Theorem statement
theorem range_of_m (m : ℝ) :
  passes_through_quadrants m → -1 < m ∧ m < 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l353_35324


namespace NUMINAMATH_CALUDE_eliza_age_l353_35384

theorem eliza_age (aunt_ellen_age : ℕ) (dina_age : ℕ) (eliza_age : ℕ) : 
  aunt_ellen_age = 48 →
  dina_age = aunt_ellen_age / 2 →
  eliza_age = dina_age - 6 →
  eliza_age = 18 := by
sorry

end NUMINAMATH_CALUDE_eliza_age_l353_35384


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l353_35342

def num_red_balls : ℕ := 4
def num_white_balls : ℕ := 6

def score_red : ℕ := 2
def score_white : ℕ := 1

def ways_to_draw (n r w : ℕ) : ℕ := Nat.choose num_red_balls r * Nat.choose num_white_balls w

theorem ball_drawing_theorem :
  (ways_to_draw 4 4 0 + ways_to_draw 4 3 1 + ways_to_draw 4 2 2 = 115) ∧
  (ways_to_draw 5 2 3 + ways_to_draw 5 3 2 + ways_to_draw 5 4 1 = 186) := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l353_35342


namespace NUMINAMATH_CALUDE_train_crossing_time_l353_35302

/-- Proves that a train with given length and speed takes a specific time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 1500 ∧ 
  train_speed_kmh = 108 →
  crossing_time = 50 :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l353_35302


namespace NUMINAMATH_CALUDE_f_difference_l353_35304

/-- The function f(x) as defined in the problem -/
def f (x : ℝ) : ℝ := x^6 + x^4 + 3*x^3 + 4*x^2 + 8*x

/-- Theorem stating that f(3) - f(-3) = 210 -/
theorem f_difference : f 3 - f (-3) = 210 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l353_35304


namespace NUMINAMATH_CALUDE_tan_4530_degrees_l353_35365

theorem tan_4530_degrees : Real.tan (4530 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_4530_degrees_l353_35365


namespace NUMINAMATH_CALUDE_mabels_daisy_problem_l353_35311

/-- Given a number of daisies and petals per daisy, calculate the total number of petals --/
def total_petals (num_daisies : ℕ) (petals_per_daisy : ℕ) : ℕ :=
  num_daisies * petals_per_daisy

/-- Given an initial number of daisies and the number of daisies given away,
    calculate the remaining number of daisies --/
def remaining_daisies (initial_daisies : ℕ) (daisies_given : ℕ) : ℕ :=
  initial_daisies - daisies_given

theorem mabels_daisy_problem (initial_daisies : ℕ) (petals_per_daisy : ℕ) (daisies_given : ℕ)
    (h1 : initial_daisies = 5)
    (h2 : petals_per_daisy = 8)
    (h3 : daisies_given = 2) :
  total_petals (remaining_daisies initial_daisies daisies_given) petals_per_daisy = 24 := by
  sorry


end NUMINAMATH_CALUDE_mabels_daisy_problem_l353_35311


namespace NUMINAMATH_CALUDE_poster_ratio_l353_35382

theorem poster_ratio (total : ℕ) (small_fraction : ℚ) (large : ℕ) : 
  total = 50 → 
  small_fraction = 2 / 5 → 
  large = 5 → 
  (total - (small_fraction * total).num - large) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_poster_ratio_l353_35382


namespace NUMINAMATH_CALUDE_everton_college_calculator_cost_l353_35308

/-- The total cost of calculators purchased by Everton college -/
def total_cost (scientific_count : ℕ) (graphing_count : ℕ) (scientific_price : ℕ) (graphing_price : ℕ) : ℕ :=
  scientific_count * scientific_price + graphing_count * graphing_price

/-- Theorem stating the total cost of calculators for Everton college -/
theorem everton_college_calculator_cost :
  total_cost 20 25 10 57 = 1625 := by
  sorry

end NUMINAMATH_CALUDE_everton_college_calculator_cost_l353_35308


namespace NUMINAMATH_CALUDE_pies_dropped_l353_35338

/-- Proves the number of pies Marcus dropped given the conditions of the problem -/
theorem pies_dropped (pies_per_batch : ℕ) (num_batches : ℕ) (pies_left : ℕ) : 
  pies_per_batch = 5 → num_batches = 7 → pies_left = 27 →
  pies_per_batch * num_batches - pies_left = 8 := by sorry

end NUMINAMATH_CALUDE_pies_dropped_l353_35338


namespace NUMINAMATH_CALUDE_partner_c_investment_l353_35348

/-- Calculates the investment of partner C in a partnership business --/
theorem partner_c_investment 
  (a_investment : ℕ) 
  (b_investment : ℕ) 
  (total_profit : ℕ) 
  (a_profit_share : ℕ) 
  (h1 : a_investment = 6300)
  (h2 : b_investment = 4200)
  (h3 : total_profit = 13000)
  (h4 : a_profit_share = 3900) :
  ∃ c_investment : ℕ, 
    c_investment = 10500 ∧ 
    a_profit_share * (a_investment + b_investment + c_investment) = 
      total_profit * a_investment :=
by sorry

end NUMINAMATH_CALUDE_partner_c_investment_l353_35348


namespace NUMINAMATH_CALUDE_line_equation_point_slope_l353_35309

/-- Theorem: Equation of a line with given slope passing through a point -/
theorem line_equation_point_slope (k x₀ y₀ : ℝ) :
  ∀ x y : ℝ, (y - y₀ = k * (x - x₀)) ↔ (y = k * x + (y₀ - k * x₀)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_point_slope_l353_35309


namespace NUMINAMATH_CALUDE_range_of_a_l353_35329

-- Define the propositions p and q
def p (m a : ℝ) : Prop := m^2 - 7*m*a + 12*a^2 < 0

def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧ 
  ∃ (c : ℝ), c > 0 ∧ x^2 / (m - 1) + y^2 / (2 - m - c) = 1

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (a > 0) → 
  (∀ m : ℝ, (¬(p m a) → ¬(q m)) ∧ ∃ m : ℝ, ¬(p m a) ∧ q m) →
  a ∈ Set.Icc (1/3 : ℝ) (3/8 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l353_35329


namespace NUMINAMATH_CALUDE_parabola_translation_l353_35396

/-- Given a parabola y = 3x² in the original coordinate system,
    if the x-axis is translated 2 units up and the y-axis is translated 2 units to the right,
    then the equation of the parabola in the new coordinate system is y = 3(x + 2)² - 2 -/
theorem parabola_translation (x y : ℝ) :
  (y = 3 * x^2) →
  (∀ x' y', x' = x - 2 ∧ y' = y - 2) →
  (y = 3 * (x + 2)^2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l353_35396


namespace NUMINAMATH_CALUDE_pencil_pen_combinations_l353_35386

theorem pencil_pen_combinations (pencil_types : Nat) (pen_types : Nat) :
  pencil_types = 4 → pen_types = 3 → pencil_types * pen_types = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_combinations_l353_35386


namespace NUMINAMATH_CALUDE_sequence_ratio_values_l353_35319

/-- Two sequences where one is arithmetic and the other is geometric -/
structure SequencePair :=
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arithmetic : (∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d) ∨ 
                  (∃ d : ℝ, ∀ n : ℕ, b (n + 1) - b n = d))
  (h_geometric : (∃ r : ℝ, ∀ n : ℕ, a (n + 1) / a n = r) ∨ 
                 (∃ r : ℝ, ∀ n : ℕ, b (n + 1) / b n = r))

/-- The theorem stating the possible values of a_3 / b_3 -/
theorem sequence_ratio_values (s : SequencePair)
  (h1 : s.a 1 = s.b 1)
  (h2 : s.a 2 / s.b 2 = 2)
  (h4 : s.a 4 / s.b 4 = 8) :
  s.a 3 / s.b 3 = -5 ∨ s.a 3 / s.b 3 = -16/5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_values_l353_35319


namespace NUMINAMATH_CALUDE_spinner_probability_l353_35325

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_D = 1/6 → p_A + p_B + p_C + p_D = 1 → p_C = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l353_35325


namespace NUMINAMATH_CALUDE_max_cookies_eaten_l353_35362

/-- Given two people sharing 30 cookies, where one eats twice as many as the other,
    the maximum number of cookies the person eating fewer could have eaten is 10. -/
theorem max_cookies_eaten (total : ℕ) (andy_cookies : ℕ) (bella_cookies : ℕ) : 
  total = 30 →
  bella_cookies = 2 * andy_cookies →
  total = andy_cookies + bella_cookies →
  andy_cookies ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_cookies_eaten_l353_35362


namespace NUMINAMATH_CALUDE_fraction_equality_l353_35376

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x - 2 * y) / (3 * x + y) = 3) : 
  (5 * x - y) / (2 * x + 4 * y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l353_35376


namespace NUMINAMATH_CALUDE_inscribed_circle_triangle_sides_l353_35375

/-- A triangle with an inscribed circle of radius 3, where one side is divided into segments of 4 and 3 by the point of tangency. -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  s₁ : ℝ
  /-- The length of the second segment of the divided side -/
  s₂ : ℝ
  /-- Condition that the radius is 3 -/
  h_r : r = 3
  /-- Condition that the first segment is 4 -/
  h_s₁ : s₁ = 4
  /-- Condition that the second segment is 3 -/
  h_s₂ : s₂ = 3

/-- The lengths of the sides of the triangle -/
def sideLengths (t : InscribedCircleTriangle) : Fin 3 → ℝ
| 0 => 24
| 1 => 25
| 2 => 7

theorem inscribed_circle_triangle_sides (t : InscribedCircleTriangle) :
  ∀ i, sideLengths t i = if i = 0 then 24 else if i = 1 then 25 else 7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_triangle_sides_l353_35375


namespace NUMINAMATH_CALUDE_marvins_substitution_l353_35300

theorem marvins_substitution (a b c d f : ℤ) : 
  a = 3 → b = 4 → c = 7 → d = 5 →
  (a + b - c + d - f = a + (b - (c + (d - f)))) →
  f = 5 := by sorry

end NUMINAMATH_CALUDE_marvins_substitution_l353_35300


namespace NUMINAMATH_CALUDE_factorization_yx_squared_minus_y_l353_35323

theorem factorization_yx_squared_minus_y (x y : ℝ) : y * x^2 - y = y * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_yx_squared_minus_y_l353_35323


namespace NUMINAMATH_CALUDE_sum_of_number_and_reverse_is_99_l353_35392

/-- Definition of a two-digit number -/
def TwoDigitNumber (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

/-- The property that the difference between the number and its reverse
    is 7 times the sum of its digits -/
def SatisfiesEquation (a b : ℕ) : Prop :=
  (10 * a + b) - (10 * b + a) = 7 * (a + b)

/-- Theorem stating that for a two-digit number satisfying the given equation,
    the sum of the number and its reverse is 99 -/
theorem sum_of_number_and_reverse_is_99 (a b : ℕ) 
  (h1 : TwoDigitNumber a b) (h2 : SatisfiesEquation a b) : 
  (10 * a + b) + (10 * b + a) = 99 := by
  sorry

#check sum_of_number_and_reverse_is_99

end NUMINAMATH_CALUDE_sum_of_number_and_reverse_is_99_l353_35392


namespace NUMINAMATH_CALUDE_price_decrease_l353_35339

/-- Given an article with a 40% price decrease resulting in a price of 1050,
    prove that the original price was 1750. -/
theorem price_decrease (original_price : ℝ) : 
  (original_price * (1 - 0.4) = 1050) → original_price = 1750 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_l353_35339


namespace NUMINAMATH_CALUDE_periodic_function_roots_l353_35361

theorem periodic_function_roots (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (2 + x) = f (2 - x))
  (h2 : ∀ x : ℝ, f (7 + x) = f (7 - x))
  (h3 : f 0 = 0) :
  ∃ (roots : Finset ℝ), (∀ x ∈ roots, f x = 0 ∧ x ∈ Set.Icc (-1000) 1000) ∧ roots.card ≥ 201 :=
sorry

end NUMINAMATH_CALUDE_periodic_function_roots_l353_35361


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_2y_l353_35303

theorem min_values_xy_and_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / x + 9 / y = 1) : 
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / a + 9 / b = 1 → x * y ≤ a * b) ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / a + 9 / b = 1 → x + 2 * y ≤ a + 2 * b) ∧
  x * y = 36 ∧ 
  x + 2 * y = 20 + 6 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_2y_l353_35303


namespace NUMINAMATH_CALUDE_vector_problem_l353_35344

/-- Given four points in a plane, prove that if certain vector conditions are met,
    then the coordinates of point D and the value of k are as specified. -/
theorem vector_problem (A B C D : ℝ × ℝ) (k : ℝ) : 
  A = (1, 3) →
  B = (2, -2) →
  C = (4, -1) →
  B - A = D - C →
  ∃ (t : ℝ), t • (k • (B - A) - (C - B)) = (B - A) + 3 • (C - B) →
  D = (5, -6) ∧ k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l353_35344


namespace NUMINAMATH_CALUDE_value_of_one_item_l353_35330

/-- Given two persons with equal capitals, each consisting of items of equal value and coins,
    prove that the value of one item is (p - m) / (a - b) --/
theorem value_of_one_item
  (a b : ℕ) (m p : ℝ) (h : a ≠ b)
  (equal_capitals : a * x + m = b * x + p)
  (x : ℝ) :
  x = (p - m) / (a - b) :=
by sorry

end NUMINAMATH_CALUDE_value_of_one_item_l353_35330


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l353_35328

def U : Set Int := {-2, 0, 1, 2}

def A : Set Int := {x ∈ U | x^2 + x - 2 = 0}

theorem complement_of_A_in_U :
  (U \ A) = {0, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l353_35328


namespace NUMINAMATH_CALUDE_product_inequality_l353_35378

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l353_35378


namespace NUMINAMATH_CALUDE_largest_solution_is_25_l353_35341

theorem largest_solution_is_25 :
  ∃ (x : ℝ), (x^2 + x - 1 + |x^2 - (x - 1)|) / 2 = 35*x - 250 ∧
  x = 25 ∧
  ∀ (y : ℝ), (y^2 + y - 1 + |y^2 - (y - 1)|) / 2 = 35*y - 250 → y ≤ 25 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_is_25_l353_35341


namespace NUMINAMATH_CALUDE_amaya_total_marks_l353_35371

/-- Represents the marks scored in different subjects -/
structure Marks where
  music : ℕ
  social_studies : ℕ
  arts : ℕ
  maths : ℕ

/-- Calculates the total marks scored in all subjects -/
def total_marks (m : Marks) : ℕ :=
  m.music + m.social_studies + m.arts + m.maths

/-- Theorem stating the total marks Amaya scored -/
theorem amaya_total_marks (m : Marks) 
  (h1 : m.maths + 20 = m.arts)
  (h2 : m.social_studies = m.music + 10)
  (h3 : m.maths = m.arts - m.arts / 10)
  (h4 : m.music = 70) :
  total_marks m = 530 := by
  sorry

#eval total_marks ⟨70, 80, 200, 180⟩

end NUMINAMATH_CALUDE_amaya_total_marks_l353_35371


namespace NUMINAMATH_CALUDE_least_odd_integer_given_mean_l353_35320

theorem least_odd_integer_given_mean (integers : List Int) : 
  integers.length = 10 ∧ 
  (∀ i ∈ integers, i % 2 = 1) ∧ 
  (∀ i j, i ∈ integers → j ∈ integers → i ≠ j → |i - j| % 2 = 0) ∧
  (integers.sum / integers.length : ℚ) = 154 →
  integers.minimum? = some 144 := by
sorry

end NUMINAMATH_CALUDE_least_odd_integer_given_mean_l353_35320


namespace NUMINAMATH_CALUDE_polynomial_coefficient_b_l353_35369

theorem polynomial_coefficient_b (a b c d : ℝ) : 
  (∃ (z w : ℂ), z * w = 9 - 3*I ∧ z + w = -2 - 6*I) →
  (∀ (r : ℂ), r^4 + a*r^3 + b*r^2 + c*r + d = 0 → r.im ≠ 0) →
  b = 58 := by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_b_l353_35369


namespace NUMINAMATH_CALUDE_mod_nine_equality_l353_35318

theorem mod_nine_equality : 54^2023 - 27^2023 ≡ 0 [ZMOD 9] := by sorry

end NUMINAMATH_CALUDE_mod_nine_equality_l353_35318


namespace NUMINAMATH_CALUDE_steves_gum_pieces_l353_35321

/-- Given Todd's initial and final number of gum pieces, prove that the number of gum pieces
    Steve gave Todd is equal to the difference between the final and initial numbers. -/
theorem steves_gum_pieces (todd_initial todd_final steve_gave : ℕ) 
    (h1 : todd_initial = 38)
    (h2 : todd_final = 54)
    (h3 : todd_final = todd_initial + steve_gave) :
  steve_gave = todd_final - todd_initial := by
  sorry

end NUMINAMATH_CALUDE_steves_gum_pieces_l353_35321


namespace NUMINAMATH_CALUDE_mn_length_is_two_l353_35317

-- Define the line l
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the intersection points M and N
def intersection_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    y₁ = line_l k x₁ ∧ y₂ = line_l k x₂ ∧
    x₁ ≠ x₂

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 12

-- Main theorem
theorem mn_length_is_two (k : ℝ) :
  intersection_points k →
  (∃ x₁ y₁ x₂ y₂ : ℝ, circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
                      y₁ = line_l k x₁ ∧ y₂ = line_l k x₂ ∧
                      dot_product_condition x₁ y₁ x₂ y₂) →
  ∃ x₁ y₁ x₂ y₂ : ℝ, circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
                     y₁ = line_l k x₁ ∧ y₂ = line_l k x₂ ∧
                     (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_mn_length_is_two_l353_35317
