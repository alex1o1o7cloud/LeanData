import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_products_l3265_326513

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 75)
  (eq2 : y^2 + y*z + z^2 = 64)
  (eq3 : z^2 + x*z + x^2 = 139) :
  x*y + y*z + x*z = 80 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l3265_326513


namespace NUMINAMATH_CALUDE_max_sides_touched_l3265_326522

/-- Represents a regular hexagon -/
structure RegularHexagon where
  -- Add any necessary fields

/-- Represents a circle -/
structure Circle where
  -- Add any necessary fields

/-- Predicate to check if a circle is entirely contained within a hexagon -/
def is_contained (c : Circle) (h : RegularHexagon) : Prop :=
  sorry

/-- Predicate to check if a circle touches a side of a hexagon -/
def touches_side (c : Circle) (h : RegularHexagon) (side : Nat) : Prop :=
  sorry

/-- Predicate to check if a circle touches all sides of a hexagon -/
def touches_all_sides (c : Circle) (h : RegularHexagon) : Prop :=
  sorry

/-- The main theorem -/
theorem max_sides_touched (h : RegularHexagon) :
  ∃ (c : Circle), is_contained c h ∧ ¬touches_all_sides c h ∧
  (∃ (n : Nat), n = 2 ∧ 
    (∀ (m : Nat), (∃ (sides : Finset Nat), sides.card = m ∧ 
      (∀ (side : Nat), side ∈ sides → touches_side c h side)) → m ≤ n)) :=
sorry

end NUMINAMATH_CALUDE_max_sides_touched_l3265_326522


namespace NUMINAMATH_CALUDE_original_kittens_correct_l3265_326567

/-- The number of kittens Tim's cat originally had -/
def original_kittens : ℕ := 6

/-- The number of kittens Tim gave away -/
def kittens_given_away : ℕ := 3

/-- The number of kittens Tim received -/
def kittens_received : ℕ := 9

/-- The number of kittens Tim has now -/
def current_kittens : ℕ := 12

/-- Theorem stating that the original number of kittens is correct -/
theorem original_kittens_correct : 
  original_kittens + kittens_received - kittens_given_away = current_kittens := by
  sorry

end NUMINAMATH_CALUDE_original_kittens_correct_l3265_326567


namespace NUMINAMATH_CALUDE_third_group_size_l3265_326591

/-- Represents a choir split into three groups -/
structure Choir :=
  (total_members : ℕ)
  (group1_members : ℕ)
  (group2_members : ℕ)
  (group3_members : ℕ)

/-- The choir has 70 total members, with 25 in the first group and 30 in the second group -/
def choir_setup : Choir :=
  { total_members := 70,
    group1_members := 25,
    group2_members := 30,
    group3_members := 15 }

/-- Theorem: The third group of the choir has 15 members -/
theorem third_group_size (c : Choir) (h1 : c.total_members = 70) 
    (h2 : c.group1_members = 25) (h3 : c.group2_members = 30) : 
    c.group3_members = 15 := by
  sorry

#check third_group_size

end NUMINAMATH_CALUDE_third_group_size_l3265_326591


namespace NUMINAMATH_CALUDE_parabola_equation_proof_l3265_326505

/-- A parabola is defined by three points: A(4,0), C(0,-4), and B(-1,0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of the parabola is y = ax^2 + bx + c -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The parabola passes through point A(4,0) -/
def passes_through_A (p : Parabola) : Prop :=
  parabola_equation p 4 = 0

/-- The parabola passes through point C(0,-4) -/
def passes_through_C (p : Parabola) : Prop :=
  parabola_equation p 0 = -4

/-- The parabola passes through point B(-1,0) -/
def passes_through_B (p : Parabola) : Prop :=
  parabola_equation p (-1) = 0

/-- The theorem states that the parabola passing through A, C, and B
    has the equation y = x^2 - 3x - 4 -/
theorem parabola_equation_proof :
  ∃ p : Parabola,
    passes_through_A p ∧
    passes_through_C p ∧
    passes_through_B p ∧
    p.a = 1 ∧ p.b = -3 ∧ p.c = -4 :=
  sorry

end NUMINAMATH_CALUDE_parabola_equation_proof_l3265_326505


namespace NUMINAMATH_CALUDE_good_number_proof_l3265_326570

theorem good_number_proof :
  ∃! n : ℕ, n ∈ Finset.range 2016 ∧
  (Finset.sum (Finset.range 2016) id - n) % 2016 = 0 ∧
  n = 1008 := by
sorry

end NUMINAMATH_CALUDE_good_number_proof_l3265_326570


namespace NUMINAMATH_CALUDE_obtuse_angle_range_l3265_326584

-- Define the vectors a and b
def a (x : ℝ) : Fin 3 → ℝ := ![x, 2, 0]
def b (x : ℝ) : Fin 3 → ℝ := ![3, 2 - x, x^2]

-- Define the dot product of two vectors
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

-- Define the condition for an obtuse angle
def is_obtuse_angle (v w : Fin 3 → ℝ) : Prop :=
  dot_product v w < 0

-- State the theorem
theorem obtuse_angle_range (x : ℝ) :
  is_obtuse_angle (a x) (b x) → x < -4 :=
sorry

end NUMINAMATH_CALUDE_obtuse_angle_range_l3265_326584


namespace NUMINAMATH_CALUDE_triangle_angle_from_side_relation_l3265_326527

theorem triangle_angle_from_side_relation (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  0 < a ∧ 0 < b ∧ 0 < c →
  Real.sqrt 2 * a = 2 * b * Real.sin A →
  B = π / 4 ∨ B = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_from_side_relation_l3265_326527


namespace NUMINAMATH_CALUDE_baking_contest_votes_l3265_326583

theorem baking_contest_votes (witch_votes dragon_votes unicorn_votes : ℕ) : 
  witch_votes = 7 →
  unicorn_votes = 3 * witch_votes →
  dragon_votes > witch_votes →
  witch_votes + unicorn_votes + dragon_votes = 60 →
  dragon_votes - witch_votes = 25 := by
sorry

end NUMINAMATH_CALUDE_baking_contest_votes_l3265_326583


namespace NUMINAMATH_CALUDE_santiago_has_58_roses_l3265_326509

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := 24

/-- The number of additional roses Mrs. Santiago has compared to Mrs. Garrett -/
def additional_roses : ℕ := 34

/-- The total number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := garrett_roses + additional_roses

/-- Theorem stating that Mrs. Santiago has 58 red roses -/
theorem santiago_has_58_roses : santiago_roses = 58 := by
  sorry

end NUMINAMATH_CALUDE_santiago_has_58_roses_l3265_326509


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3265_326515

theorem trigonometric_identities (θ : Real) 
  (h1 : π/2 < θ ∧ θ < π) -- θ is in the second quadrant
  (h2 : Real.tan (2*θ) = -2*Real.sqrt 2) : -- tan 2θ = -2√2
  (Real.tan θ = -Real.sqrt 2 / 2) ∧ 
  ((2 * (Real.cos (θ/2))^2 - Real.sin θ - Real.tan (5*π/4)) / 
   (Real.sqrt 2 * Real.sin (θ + π/4)) = 4 + 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3265_326515


namespace NUMINAMATH_CALUDE_triangle_ratio_greater_than_two_l3265_326555

/-- In a right triangle ABC with ∠BAC = 90°, AB = 5, BC = 6, and point K dividing AC in ratio 3:1 from A,
    the ratio BK/AH is greater than 2, where AH is the altitude from A to BC. -/
theorem triangle_ratio_greater_than_two (A B C K H : ℝ × ℝ) : 
  -- Triangle ABC is right-angled at A
  (A.1 = 0 ∧ A.2 = 0) → 
  (B.1 = 5 ∧ B.2 = 0) → 
  (C.1 = 0 ∧ C.2 = 6) → 
  -- K divides AC in ratio 3:1 from A
  (K.1 = (3/4) * C.1 ∧ K.2 = (3/4) * C.2) →
  -- H is the foot of the altitude from A to BC
  (H.1 = 0 ∧ H.2 = 30 / Real.sqrt 61) →
  -- The ratio BK/AH is greater than 2
  Real.sqrt ((K.1 - B.1)^2 + (K.2 - B.2)^2) / Real.sqrt (H.1^2 + H.2^2) > 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_greater_than_two_l3265_326555


namespace NUMINAMATH_CALUDE_certain_number_proof_l3265_326538

theorem certain_number_proof : ∃ x : ℕ, x * 12 = 173 * 240 ∧ x = 3460 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3265_326538


namespace NUMINAMATH_CALUDE_soccer_team_captains_l3265_326516

theorem soccer_team_captains (n : ℕ) (k : ℕ) (h1 : n = 14) (h2 : k = 3) :
  Nat.choose n k = 364 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_captains_l3265_326516


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3265_326596

/-- Given that i is the imaginary unit, prove that (3 + 2i) / (1 - i) = 1/2 + 5/2 * i -/
theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 + 2 * i) / (1 - i) = 1/2 + 5/2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3265_326596


namespace NUMINAMATH_CALUDE_positive_root_negative_root_zero_root_l3265_326592

-- Define the equation
def equation (a b x : ℝ) : Prop := b + x = 4 * x + a

-- Theorem for positive root
theorem positive_root (a b : ℝ) : 
  b > a → ∃ x : ℝ, x > 0 ∧ equation a b x := by sorry

-- Theorem for negative root
theorem negative_root (a b : ℝ) : 
  b < a → ∃ x : ℝ, x < 0 ∧ equation a b x := by sorry

-- Theorem for zero root
theorem zero_root (a b : ℝ) : 
  b = a → ∃ x : ℝ, x = 0 ∧ equation a b x := by sorry

end NUMINAMATH_CALUDE_positive_root_negative_root_zero_root_l3265_326592


namespace NUMINAMATH_CALUDE_total_bottles_bought_l3265_326563

-- Define the variables
def bottles_per_day : ℕ := 9
def days_lasted : ℕ := 17

-- Define the theorem
theorem total_bottles_bought : 
  bottles_per_day * days_lasted = 153 := by
  sorry

end NUMINAMATH_CALUDE_total_bottles_bought_l3265_326563


namespace NUMINAMATH_CALUDE_inequality_preservation_l3265_326502

theorem inequality_preservation (a b : ℝ) (h : a > b) : a / 3 > b / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3265_326502


namespace NUMINAMATH_CALUDE_existence_of_a_i_for_x_ij_l3265_326501

theorem existence_of_a_i_for_x_ij (n : ℕ) (x : Fin n → Fin n → ℝ)
  (h : ∀ (i j k : Fin n), x i j + x j k + x k i = 0) :
  ∃ (a : Fin n → ℝ), ∀ (i j : Fin n), x i j = a i - a j := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_i_for_x_ij_l3265_326501


namespace NUMINAMATH_CALUDE_solve_for_p_l3265_326508

theorem solve_for_p (n m p : ℚ) : 
  (5 / 6 : ℚ) = n / 72 ∧ 
  (5 / 6 : ℚ) = (m + n) / 84 ∧ 
  (5 / 6 : ℚ) = (p - m) / 120 → 
  p = 110 := by
sorry

end NUMINAMATH_CALUDE_solve_for_p_l3265_326508


namespace NUMINAMATH_CALUDE_power_of_two_equality_l3265_326511

theorem power_of_two_equality : (16^3) * (4^4) * (32^2) = 2^30 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l3265_326511


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_and_contained_line_l3265_326528

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem perpendicular_line_to_plane_and_contained_line 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) 
  (h2 : contained_in n α) : 
  perpendicular_lines m n := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_and_contained_line_l3265_326528


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l3265_326535

theorem quadratic_function_minimum (a b c : ℝ) (h_a : a ≠ 0) :
  let f := fun x => a * x^2 + b * x + c
  let f' := fun x => 2 * a * x + b
  (f' 0 > 0) →
  (∀ x, f x ≥ 0) →
  ∀ ε > 0, ∃ x, f x / f' 0 < 2 + ε :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l3265_326535


namespace NUMINAMATH_CALUDE_alpha_beta_inequality_l3265_326559

theorem alpha_beta_inequality (α β : ℝ) : α > β ↔ α - β > Real.sin α - Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_inequality_l3265_326559


namespace NUMINAMATH_CALUDE_cube_digits_convergence_l3265_326500

/-- The function that cubes each digit of a natural number and sums the results -/
def cube_digits_sum (n : ℕ) : ℕ :=
  n.digits 10
    |>.map (fun d => d^3)
    |>.sum

/-- The sequence generated by repeatedly applying cube_digits_sum -/
def cube_digits_sequence (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => cube_digits_sum (cube_digits_sequence start n)

/-- The theorem stating that the sequence converges to 153 for multiples of 3 -/
theorem cube_digits_convergence (n : ℕ) (h : 3 ∣ n) :
  ∃ k, ∀ m ≥ k, cube_digits_sequence n m = 153 := by
  sorry

end NUMINAMATH_CALUDE_cube_digits_convergence_l3265_326500


namespace NUMINAMATH_CALUDE_solve_for_z_l3265_326560

theorem solve_for_z : ∃ z : ℝ, (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt z = 2.4 → z = 75 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_z_l3265_326560


namespace NUMINAMATH_CALUDE_min_workers_theorem_l3265_326598

/-- Represents the company's profit scenario -/
structure CompanyProfit where
  maintenance_cost : ℕ
  worker_wage : ℕ
  production_rate : ℕ
  gadget_price : ℚ
  workday_length : ℕ

/-- Calculates the minimum number of workers required for profit -/
def min_workers_for_profit (c : CompanyProfit) : ℕ :=
  Nat.succ (Nat.ceil ((c.maintenance_cost : ℚ) / 
    (c.production_rate * c.workday_length * c.gadget_price - c.worker_wage * c.workday_length)))

/-- Theorem stating the minimum number of workers required for profit -/
theorem min_workers_theorem (c : CompanyProfit) 
  (h1 : c.maintenance_cost = 800)
  (h2 : c.worker_wage = 20)
  (h3 : c.production_rate = 6)
  (h4 : c.gadget_price = 9/2)
  (h5 : c.workday_length = 9) :
  min_workers_for_profit c = 13 := by
  sorry

#eval min_workers_for_profit { 
  maintenance_cost := 800, 
  worker_wage := 20, 
  production_rate := 6, 
  gadget_price := 9/2, 
  workday_length := 9 
}

end NUMINAMATH_CALUDE_min_workers_theorem_l3265_326598


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_implies_a_value_l3265_326529

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

-- State the theorem
theorem monotonic_decreasing_interval_implies_a_value (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y ≤ 4 → f a x > f a y) →
  a = -3 :=
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_implies_a_value_l3265_326529


namespace NUMINAMATH_CALUDE_juan_number_problem_l3265_326503

theorem juan_number_problem (n : ℝ) : 
  (2 * ((n + 3)^2) - 2) / 3 = 14 ↔ (n = -3 + Real.sqrt 22 ∨ n = -3 - Real.sqrt 22) :=
by sorry

end NUMINAMATH_CALUDE_juan_number_problem_l3265_326503


namespace NUMINAMATH_CALUDE_point_not_on_line_l3265_326585

/-- Given m > 2 and mb > 0, prove that (0, -2023) cannot lie on y = mx + b -/
theorem point_not_on_line (m b : ℝ) (hm : m > 2) (hmb : m * b > 0) :
  ¬ (∃ (x y : ℝ), x = 0 ∧ y = -2023 ∧ y = m * x + b) := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_line_l3265_326585


namespace NUMINAMATH_CALUDE_selling_price_calculation_l3265_326510

theorem selling_price_calculation (cost_price : ℝ) (profit_percentage : ℝ) : 
  cost_price = 240 → profit_percentage = 20 → 
  cost_price * (1 + profit_percentage / 100) = 288 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l3265_326510


namespace NUMINAMATH_CALUDE_milk_solution_l3265_326536

def milk_problem (A B C : ℝ) : Prop :=
  A > 0 ∧  -- A is positive (implicitly assumed as it represents a quantity)
  B = 0.375 * A ∧  -- B is 62.5% less than A
  C = A - B ∧  -- C contains the remainder
  B + 148 = C - 148  -- After transfer, B and C are equal

theorem milk_solution :
  ∀ A B C, milk_problem A B C → A = 1184 := by
  sorry

end NUMINAMATH_CALUDE_milk_solution_l3265_326536


namespace NUMINAMATH_CALUDE_factorization_equalities_l3265_326575

theorem factorization_equalities (x y : ℝ) : 
  (2 * x^3 - 8 * x = 2 * x * (x + 2) * (x - 2)) ∧ 
  (x^3 - 5 * x^2 + 6 * x = x * (x - 3) * (x - 2)) ∧ 
  (4 * x^4 * y^2 - 5 * x^2 * y^2 - 9 * y^2 = y^2 * (2 * x + 3) * (2 * x - 3) * (x^2 + 1)) ∧ 
  (3 * x^2 - 10 * x * y + 3 * y^2 = (3 * x - y) * (x - 3 * y)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equalities_l3265_326575


namespace NUMINAMATH_CALUDE_expression_factorization_l3265_326533

theorem expression_factorization (b : ℝ) :
  (4 * b^3 - 84 * b^2 - 12 * b) - (-3 * b^3 - 9 * b^2 + 3 * b) = b * (7 * b + 3) * (b - 5) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3265_326533


namespace NUMINAMATH_CALUDE_smallest_k_value_l3265_326552

theorem smallest_k_value (m n k : ℤ) (h : 221 * m + 247 * n + 323 * k = 2001) :
  (k > 100 ∧ ∀ k' > 100, 221 * m + 247 * n + 323 * k' = 2001 → k ≤ k') → k = 111 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_value_l3265_326552


namespace NUMINAMATH_CALUDE_company_fund_proof_l3265_326531

theorem company_fund_proof (n : ℕ) (initial_fund : ℕ) : 
  (80 * n - 20 = initial_fund) →  -- Planned $80 bonus, $20 short
  (70 * n + 75 = initial_fund) →  -- Actual $70 bonus, $75 left
  initial_fund = 700 := by
sorry

end NUMINAMATH_CALUDE_company_fund_proof_l3265_326531


namespace NUMINAMATH_CALUDE_x_squared_plus_3xy_plus_y_squared_l3265_326512

theorem x_squared_plus_3xy_plus_y_squared (x y : ℝ) 
  (h1 : x * y = -3) 
  (h2 : x + y = -4) : 
  x^2 + 3*x*y + y^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_3xy_plus_y_squared_l3265_326512


namespace NUMINAMATH_CALUDE_line_x_intercept_l3265_326580

/-- A line passing through two points (-3, 3) and (2, 10) has x-intercept -36/7 -/
theorem line_x_intercept : 
  let p₁ : ℝ × ℝ := (-3, 3)
  let p₂ : ℝ × ℝ := (2, 10)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  let b : ℝ := p₁.2 - m * p₁.1
  (0 - b) / m = -36/7 :=
by sorry

end NUMINAMATH_CALUDE_line_x_intercept_l3265_326580


namespace NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l3265_326540

theorem largest_n_for_sin_cos_inequality : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (2 * ↑n)) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), (Real.sin y)^m + (Real.cos y)^m < 1 / (2 * ↑m)) ∧
  n = 8 :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l3265_326540


namespace NUMINAMATH_CALUDE_shelf_filling_l3265_326550

theorem shelf_filling (A H C S M N E : ℕ) (x y z : ℝ) (l : ℝ) 
  (hA : A > 0) (hH : H > 0) (hC : C > 0) (hS : S > 0) (hM : M > 0) (hN : N > 0) (hE : E > 0)
  (hDistinct : A ≠ H ∧ A ≠ C ∧ A ≠ S ∧ A ≠ M ∧ A ≠ N ∧ A ≠ E ∧
               H ≠ C ∧ H ≠ S ∧ H ≠ M ∧ H ≠ N ∧ H ≠ E ∧
               C ≠ S ∧ C ≠ M ∧ C ≠ N ∧ C ≠ E ∧
               S ≠ M ∧ S ≠ N ∧ S ≠ E ∧
               M ≠ N ∧ M ≠ E ∧
               N ≠ E)
  (hThickness : 0 < x ∧ x < y ∧ x < z)
  (hFill1 : A * x + H * y + C * z = l)
  (hFill2 : S * x + M * y + N * z = l)
  (hFill3 : E * x = l) :
  E = (A * M + C * N - S * H - N * H) / (M + N - H) :=
sorry

end NUMINAMATH_CALUDE_shelf_filling_l3265_326550


namespace NUMINAMATH_CALUDE_f_monotonicity_and_range_l3265_326587

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + (1/2) * a * x^2 - x

theorem f_monotonicity_and_range :
  (∀ x > -1, ∀ y ∈ (Set.Ioo (-1 : ℝ) (-1/2) ∪ Set.Ioi 0), x < y → f 2 x < f 2 y) ∧
  (∀ x > -1, ∀ y ∈ Set.Ioo (-1/2 : ℝ) 0, x < y → f 2 x > f 2 y) ∧
  (∀ a : ℝ, (∀ x > 0, f a x ≥ a * x - x) ↔ 0 ≤ a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_range_l3265_326587


namespace NUMINAMATH_CALUDE_sin_cos_difference_74_14_l3265_326586

theorem sin_cos_difference_74_14 :
  Real.sin (74 * π / 180) * Real.cos (14 * π / 180) -
  Real.cos (74 * π / 180) * Real.sin (14 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_74_14_l3265_326586


namespace NUMINAMATH_CALUDE_minimum_containers_needed_l3265_326590

theorem minimum_containers_needed (pool_capacity : ℕ) (container_capacity : ℕ) 
  (h1 : pool_capacity = 2250)
  (h2 : container_capacity = 75)
  (h3 : pool_capacity > 0)
  (h4 : container_capacity > 0) :
  (pool_capacity + container_capacity - 1) / container_capacity = 30 :=
by sorry

end NUMINAMATH_CALUDE_minimum_containers_needed_l3265_326590


namespace NUMINAMATH_CALUDE_custom_mult_five_four_l3265_326572

-- Define the custom multiplication operation
def custom_mult (a b : ℤ) : ℤ := a^2 + a*b - b^2

-- Theorem statement
theorem custom_mult_five_four :
  custom_mult 5 4 = 29 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_five_four_l3265_326572


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_three_l3265_326542

theorem sqrt_expression_equals_three :
  (Real.sqrt 2 + 1)^2 - Real.sqrt 18 + 2 * Real.sqrt (1/2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_three_l3265_326542


namespace NUMINAMATH_CALUDE_people_in_room_l3265_326576

theorem people_in_room (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) : 
  (3 : ℚ) / 5 * total_people = seated_people →
  (4 : ℚ) / 5 * total_chairs = seated_people →
  total_chairs - seated_people = 5 →
  total_people = 33 := by
sorry

end NUMINAMATH_CALUDE_people_in_room_l3265_326576


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3265_326579

/-- A geometric sequence with positive terms and common ratio 2 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3265_326579


namespace NUMINAMATH_CALUDE_combined_weight_of_new_men_weight_problem_l3265_326532

/-- The combined weight of two new men replacing one man in a group, given certain conditions -/
theorem combined_weight_of_new_men (initial_count : ℕ) (weight_increase : ℝ) 
  (replaced_weight : ℝ) (new_count : ℕ) : ℝ :=
  let total_weight_increase := weight_increase * new_count
  let combined_weight := total_weight_increase + replaced_weight
  combined_weight

/-- The theorem statement matching the original problem -/
theorem weight_problem : 
  combined_weight_of_new_men 10 2.5 68 11 = 95.5 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_of_new_men_weight_problem_l3265_326532


namespace NUMINAMATH_CALUDE_proportion_equality_l3265_326524

theorem proportion_equality (m n : ℝ) (h1 : 6 * m = 7 * n) (h2 : n ≠ 0) :
  m / 7 = n / 6 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l3265_326524


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l3265_326595

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 7071 : ℤ) ≡ 3540 [ZMOD 15] ∧
  ∀ (y : ℕ), y > 0 → (y + 7071 : ℤ) ≡ 3540 [ZMOD 15] → x ≤ y ∧
  x = 9 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l3265_326595


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l3265_326551

def p (x : ℝ) : ℝ := -3 * x^3 - 4 * x^2 - 8 * x + 2
def q (x : ℝ) : ℝ := -2 * x^2 - 7 * x + 3

theorem coefficient_x_squared_in_product :
  ∃ (a b c d e : ℝ), p x * q x = a * x^4 + b * x^3 + 40 * x^2 + d * x + e :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l3265_326551


namespace NUMINAMATH_CALUDE_sisters_height_l3265_326597

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Converts inches to feet and remaining inches -/
def inches_to_feet_and_inches (inches : ℕ) : ℕ × ℕ :=
  (inches / 12, inches % 12)

/-- Represents a height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ
  h_valid : inches < 12

theorem sisters_height 
  (sunflower_height_feet : ℕ)
  (height_difference_inches : ℕ)
  (h_sunflower : sunflower_height_feet = 6)
  (h_difference : height_difference_inches = 21) :
  let sunflower_height_inches := feet_to_inches sunflower_height_feet
  let sister_height_inches := sunflower_height_inches - height_difference_inches
  let (sister_feet, sister_inches) := inches_to_feet_and_inches sister_height_inches
  Height.mk sister_feet sister_inches (by sorry) = Height.mk 4 3 (by sorry) :=
by sorry

end NUMINAMATH_CALUDE_sisters_height_l3265_326597


namespace NUMINAMATH_CALUDE_hyperbola_condition_l3265_326561

theorem hyperbola_condition (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (m + 2) - y^2 / (m - 1) = 1) → (m < -2 ∨ m > 1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l3265_326561


namespace NUMINAMATH_CALUDE_second_bag_weight_is_10_l3265_326548

/-- The weight of the second bag of dog food Elise bought -/
def second_bag_weight (initial_weight first_bag_weight final_weight : ℕ) : ℕ :=
  final_weight - (initial_weight + first_bag_weight)

theorem second_bag_weight_is_10 :
  second_bag_weight 15 15 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_bag_weight_is_10_l3265_326548


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l3265_326517

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∀ x y : ℝ, (m + 2) * x + 3 * m * y + 7 = 0 ∧ 
               (m - 2) * x + (m + 2) * y - 5 = 0 → 
               ((m + 2) * (m - 2) + 3 * m * (m + 2) = 0)) → 
  m = 1/2 ∨ m = -2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l3265_326517


namespace NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l3265_326574

theorem complex_exp_13pi_over_2 : Complex.exp ((13 * Real.pi / 2) * Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l3265_326574


namespace NUMINAMATH_CALUDE_number_with_percentage_increase_l3265_326564

theorem number_with_percentage_increase : ∃ x : ℝ, x + 0.35 * x = x + 150 := by
  sorry

end NUMINAMATH_CALUDE_number_with_percentage_increase_l3265_326564


namespace NUMINAMATH_CALUDE_vampire_blood_consumption_l3265_326507

/-- The amount of blood a vampire needs per week in gallons -/
def blood_needed_per_week : ℚ := 7

/-- The number of people the vampire sucks blood from each day -/
def people_per_day : ℕ := 4

/-- The number of pints in a gallon -/
def pints_per_gallon : ℕ := 8

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem: Given a vampire who needs 7 gallons of blood per week and sucks blood from 4 people each day, 
    the amount of blood sucked per person is 2 pints. -/
theorem vampire_blood_consumption :
  (blood_needed_per_week * pints_per_gallon) / (people_per_day * days_per_week) = 2 := by
  sorry

end NUMINAMATH_CALUDE_vampire_blood_consumption_l3265_326507


namespace NUMINAMATH_CALUDE_amount_with_r_l3265_326526

/-- Given three people sharing a total amount of money, where one person has
    two-thirds of what the other two have combined, this theorem proves
    the amount held by that person. -/
theorem amount_with_r (total : ℝ) (amount_r : ℝ) : 
  total = 7000 →
  amount_r = (2/3) * (total - amount_r) →
  amount_r = 2800 := by
sorry


end NUMINAMATH_CALUDE_amount_with_r_l3265_326526


namespace NUMINAMATH_CALUDE_compound_interest_rate_is_10_percent_l3265_326518

/-- Prove that the compound interest rate is 10% given the problem conditions -/
theorem compound_interest_rate_is_10_percent 
  (simple_interest_principal : ℝ) 
  (simple_interest_rate : ℝ) 
  (simple_interest_time : ℝ) 
  (compound_interest_principal : ℝ) 
  (compound_interest_time : ℝ) 
  (h1 : simple_interest_principal = 1750)
  (h2 : simple_interest_rate = 8)
  (h3 : simple_interest_time = 3)
  (h4 : compound_interest_principal = 4000)
  (h5 : compound_interest_time = 2)
  (h6 : simple_interest_principal * simple_interest_rate * simple_interest_time / 100 = 
        compound_interest_principal * ((1 + compound_interest_rate / 100)^compound_interest_time - 1) / 2) :
  compound_interest_rate = 10 := by
  sorry


end NUMINAMATH_CALUDE_compound_interest_rate_is_10_percent_l3265_326518


namespace NUMINAMATH_CALUDE_cos_sum_seventh_roots_l3265_326594

theorem cos_sum_seventh_roots : 
  Real.cos (2 * π / 7) + Real.cos (4 * π / 7) + Real.cos (6 * π / 7) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_seventh_roots_l3265_326594


namespace NUMINAMATH_CALUDE_y_derivative_l3265_326571

noncomputable def y (x : ℝ) : ℝ := -2 * Real.exp x * Real.sin x

theorem y_derivative (x : ℝ) : 
  deriv y x = -2 * Real.exp x * (Real.cos x + Real.sin x) := by sorry

end NUMINAMATH_CALUDE_y_derivative_l3265_326571


namespace NUMINAMATH_CALUDE_sunrise_is_certain_event_l3265_326534

-- Define the type for events
inductive Event
| TV : Event
| Dice : Event
| Sunrise : Event
| SeedGermination : Event

-- Define the property of being a certain event
def isCertainEvent (e : Event) : Prop :=
  match e with
  | Event.TV => False
  | Event.Dice => False
  | Event.Sunrise => True
  | Event.SeedGermination => False

-- Theorem statement
theorem sunrise_is_certain_event : isCertainEvent Event.Sunrise := by
  sorry

end NUMINAMATH_CALUDE_sunrise_is_certain_event_l3265_326534


namespace NUMINAMATH_CALUDE_moon_speed_km_per_hour_l3265_326546

/-- The speed of the moon in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 0.2

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem: The moon's speed in kilometers per hour -/
theorem moon_speed_km_per_hour :
  moon_speed_km_per_sec * (seconds_per_hour : ℝ) = 720 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_km_per_hour_l3265_326546


namespace NUMINAMATH_CALUDE_alexander_pencils_per_picture_l3265_326582

theorem alexander_pencils_per_picture
  (first_exhibition_pictures : ℕ)
  (new_galleries : ℕ)
  (pictures_per_new_gallery : ℕ)
  (signing_pencils_per_exhibition : ℕ)
  (total_pencils_used : ℕ)
  (h1 : first_exhibition_pictures = 9)
  (h2 : new_galleries = 5)
  (h3 : pictures_per_new_gallery = 2)
  (h4 : signing_pencils_per_exhibition = 2)
  (h5 : total_pencils_used = 88) :
  (total_pencils_used - (signing_pencils_per_exhibition * (new_galleries + 1))) /
  (first_exhibition_pictures + new_galleries * pictures_per_new_gallery) = 4 := by
sorry

end NUMINAMATH_CALUDE_alexander_pencils_per_picture_l3265_326582


namespace NUMINAMATH_CALUDE_pauls_and_sarahs_ages_l3265_326558

theorem pauls_and_sarahs_ages (p s : ℕ) : 
  p = s + 8 →                   -- Paul is eight years older than Sarah
  p + 6 = 3 * (s - 2) →         -- In six years, Paul will be three times as old as Sarah was two years ago
  p + s = 28                    -- The sum of their current ages is 28
  := by sorry

end NUMINAMATH_CALUDE_pauls_and_sarahs_ages_l3265_326558


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l3265_326557

theorem triangle_side_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + c^2) / b^2 ≥ (1 : ℝ) / 2 ∧ ∃ a' b' c' : ℝ, 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧ (a'^2 + c'^2) / b'^2 = (1 : ℝ) / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_inequality_l3265_326557


namespace NUMINAMATH_CALUDE_field_trip_attendance_calculation_l3265_326549

/-- The number of people on a field trip -/
def field_trip_attendance (num_vans : ℕ) (num_buses : ℕ) (people_per_van : ℕ) (people_per_bus : ℕ) : ℕ :=
  num_vans * people_per_van + num_buses * people_per_bus

/-- Theorem stating the total number of people on the field trip -/
theorem field_trip_attendance_calculation :
  field_trip_attendance 6 8 6 18 = 180 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_attendance_calculation_l3265_326549


namespace NUMINAMATH_CALUDE_real_part_of_z_l3265_326525

theorem real_part_of_z (z : ℂ) : z = (2 + I) / (1 + I)^2 → z.re = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3265_326525


namespace NUMINAMATH_CALUDE_restaurant_table_difference_l3265_326588

/-- Represents the number of tables and seating capacity in a restaurant --/
structure Restaurant where
  new_tables : ℕ
  original_tables : ℕ
  new_table_capacity : ℕ
  original_table_capacity : ℕ

/-- Calculates the total number of tables in the restaurant --/
def Restaurant.total_tables (r : Restaurant) : ℕ :=
  r.new_tables + r.original_tables

/-- Calculates the total seating capacity of the restaurant --/
def Restaurant.total_capacity (r : Restaurant) : ℕ :=
  r.new_tables * r.new_table_capacity + r.original_tables * r.original_table_capacity

/-- Theorem stating the difference between new and original tables --/
theorem restaurant_table_difference (r : Restaurant) 
  (h1 : r.total_tables = 40)
  (h2 : r.total_capacity = 212)
  (h3 : r.new_table_capacity = 6)
  (h4 : r.original_table_capacity = 4) :
  r.new_tables - r.original_tables = 12 := by
  sorry


end NUMINAMATH_CALUDE_restaurant_table_difference_l3265_326588


namespace NUMINAMATH_CALUDE_sum_first_15_not_multiple_of_57_l3265_326565

theorem sum_first_15_not_multiple_of_57 :
  ¬ (∃ k : ℕ, (List.range 15).sum + 15 = 57 * k) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_first_15_not_multiple_of_57_l3265_326565


namespace NUMINAMATH_CALUDE_inequality_solution_l3265_326562

def solution_set : Set ℝ := {x : ℝ | -5 < x ∧ x < 1 ∨ x > 6}

theorem inequality_solution :
  {x : ℝ | (x - 1) / (x^2 - x - 30) > 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3265_326562


namespace NUMINAMATH_CALUDE_stating_race_result_l3265_326599

/-- Represents a runner in the race -/
inductive Runner
| Primus
| Secundus
| Tertius

/-- Represents the order of runners -/
def RunnerOrder := List Runner

/-- The number of place changes between pairs of runners -/
structure PlaceChanges where
  primus_secundus : Nat
  secundus_tertius : Nat
  primus_tertius : Nat

/-- The initial order of runners -/
def initial_order : RunnerOrder := [Runner.Primus, Runner.Secundus, Runner.Tertius]

/-- The place changes during the race -/
def race_changes : PlaceChanges := {
  primus_secundus := 9,
  secundus_tertius := 10,
  primus_tertius := 11
}

/-- The final order of runners -/
def final_order : RunnerOrder := [Runner.Secundus, Runner.Tertius, Runner.Primus]

/-- 
Theorem stating that given the initial order and place changes,
the final order is [Secundus, Tertius, Primus]
-/
theorem race_result (order : RunnerOrder) (changes : PlaceChanges) :
  order = initial_order ∧ changes = race_changes →
  final_order = [Runner.Secundus, Runner.Tertius, Runner.Primus] :=
by sorry

end NUMINAMATH_CALUDE_stating_race_result_l3265_326599


namespace NUMINAMATH_CALUDE_object_is_cylinder_l3265_326568

-- Define the possible shapes
inductive Shape
  | Rectangle
  | Cylinder
  | Cuboid
  | Cone

-- Define the types of views
inductive View
  | Rectangular
  | Circular

-- Define the object's properties
structure Object where
  frontView : View
  topView : View
  sideView : View

-- Theorem statement
theorem object_is_cylinder (obj : Object)
  (h1 : obj.frontView = View.Rectangular)
  (h2 : obj.sideView = View.Rectangular)
  (h3 : obj.topView = View.Circular) :
  Shape.Cylinder = 
    match obj.frontView, obj.topView, obj.sideView with
    | View.Rectangular, View.Circular, View.Rectangular => Shape.Cylinder
    | _, _, _ => Shape.Rectangle  -- default case, won't be reached
  := by sorry

end NUMINAMATH_CALUDE_object_is_cylinder_l3265_326568


namespace NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l3265_326543

theorem cubic_root_reciprocal_sum (p q r : ℝ) : 
  p^3 - 9*p^2 + 8*p + 2 = 0 →
  q^3 - 9*q^2 + 8*q + 2 = 0 →
  r^3 - 9*r^2 + 8*r + 2 = 0 →
  p ≠ q → p ≠ r → q ≠ r →
  1/p^2 + 1/q^2 + 1/r^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l3265_326543


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l3265_326547

-- Define the color type
inductive Color
  | White
  | Red
  | Black

-- Define the coloring function type
def ColoringFunction := ℤ × ℤ → Color

-- Define the property of a color appearing on infinitely many lines
def InfiniteLines (f : ColoringFunction) (c : Color) : Prop :=
  ∀ N : ℕ, ∃ y > N, ∀ M : ℕ, ∃ x > M, f (x, y) = c

-- Define the parallelogram property
def ParallelogramProperty (f : ColoringFunction) : Prop :=
  ∀ a b c : ℤ × ℤ,
    f a = Color.White → f b = Color.Red → f c = Color.Black →
    ∃ d : ℤ × ℤ, f d = Color.Red ∧ d = (a.1 + c.1 - b.1, a.2 + c.2 - b.2)

-- The main theorem
theorem exists_valid_coloring : ∃ f : ColoringFunction,
  (InfiniteLines f Color.White) ∧
  (InfiniteLines f Color.Red) ∧
  (InfiniteLines f Color.Black) ∧
  ParallelogramProperty f :=
sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l3265_326547


namespace NUMINAMATH_CALUDE_xy_value_l3265_326514

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3265_326514


namespace NUMINAMATH_CALUDE_floor_product_equation_l3265_326589

theorem floor_product_equation : ∃! (x : ℝ), x > 0 ∧ x * ⌊x⌋ = 50 ∧ |x - (50 / 7)| < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_floor_product_equation_l3265_326589


namespace NUMINAMATH_CALUDE_parabola_area_ratio_l3265_326577

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Line passing through a point -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- Intersection of a line with x = -2 -/
def intersectionWithM (l : Line) : ℝ × ℝ :=
  (-2, l.slope * (-3 - l.point.1) + l.point.2)

/-- Area of a triangle -/
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  sorry

/-- Main theorem -/
theorem parabola_area_ratio 
  (C : Parabola)
  (F : ℝ × ℝ)
  (L : Line)
  (A B : ℝ × ℝ)
  (O : ℝ × ℝ := (0, 0))
  (M N : ℝ × ℝ) :
  C.p = 2 →
  F = (1, 0) →
  L.point = F →
  C.equation A.1 A.2 →
  C.equation B.1 B.2 →
  M = intersectionWithM ⟨O, (A.2 - O.2) / (A.1 - O.1)⟩ →
  N = intersectionWithM ⟨O, (B.2 - O.2) / (B.1 - O.1)⟩ →
  (triangleArea A B O) / (triangleArea M N O) = 1/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_area_ratio_l3265_326577


namespace NUMINAMATH_CALUDE_theater_ticket_area_l3265_326553

/-- The area of a rectangular theater ticket -/
theorem theater_ticket_area (perimeter width : ℝ) (h1 : perimeter = 28) (h2 : width = 6) :
  let length := (perimeter - 2 * width) / 2
  width * length = 48 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_area_l3265_326553


namespace NUMINAMATH_CALUDE_smith_family_mean_age_l3265_326556

def smith_family_ages : List ℕ := [8, 8, 8, 12, 11, 3, 4]

theorem smith_family_mean_age :
  (smith_family_ages.sum : ℚ) / smith_family_ages.length = 54 / 7 := by
  sorry

end NUMINAMATH_CALUDE_smith_family_mean_age_l3265_326556


namespace NUMINAMATH_CALUDE_promotion_difference_l3265_326581

/-- Represents a shoe promotion strategy -/
inductive Promotion
  | A  -- Buy one pair, get second pair half price
  | B  -- Buy one pair, get $15 off second pair

/-- Calculate the total cost of two pairs of shoes under a given promotion -/
def calculateCost (p : Promotion) (price : ℕ) : ℕ :=
  match p with
  | Promotion.A => price + price / 2
  | Promotion.B => price + price - 15

/-- The difference in cost between Promotion B and Promotion A is $5 -/
theorem promotion_difference (shoePrice : ℕ) (h : shoePrice = 40) :
  calculateCost Promotion.B shoePrice - calculateCost Promotion.A shoePrice = 5 := by
  sorry

#eval calculateCost Promotion.B 40 - calculateCost Promotion.A 40

end NUMINAMATH_CALUDE_promotion_difference_l3265_326581


namespace NUMINAMATH_CALUDE_rational_expression_iff_perfect_square_l3265_326537

theorem rational_expression_iff_perfect_square (x : ℝ) :
  ∃ (q : ℚ), x + Real.sqrt (x^2 + 9) - 1 / (x + Real.sqrt (x^2 + 9)) = q ↔ 
  ∃ (n : ℕ), x^2 + 9 = n^2 := by
sorry

end NUMINAMATH_CALUDE_rational_expression_iff_perfect_square_l3265_326537


namespace NUMINAMATH_CALUDE_exp_properties_l3265_326519

-- Define the exponential function as a power series
noncomputable def Exp (z : ℂ) : ℂ := ∑' n, z^n / n.factorial

-- State the properties to be proved
theorem exp_properties :
  -- Property 1: The derivative of Exp(z) is equal to Exp(z) itself
  (∀ z : ℂ, HasDerivAt Exp (Exp z) z) ∧
  -- Property 2: Exp((α+β)z) = Exp(αz) · Exp(βz) for any complex α and β
  (∀ α β z : ℂ, Exp ((α + β) * z) = Exp (α * z) * Exp (β * z)) :=
by sorry

end NUMINAMATH_CALUDE_exp_properties_l3265_326519


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l3265_326544

theorem convex_polygon_sides (sum_except_one : ℝ) (missing_angle : ℝ) : 
  sum_except_one = 2970 ∧ missing_angle = 150 → 
  (∃ (n : ℕ), n = 20 ∧ 180 * (n - 2) = sum_except_one + missing_angle) :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l3265_326544


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l3265_326506

theorem cement_mixture_weight :
  ∀ (total_weight : ℝ),
    (1/5 : ℝ) * total_weight +     -- Sand
    (3/4 : ℝ) * total_weight +     -- Water
    6 = total_weight →             -- Gravel
    total_weight = 120 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l3265_326506


namespace NUMINAMATH_CALUDE_function_domain_condition_l3265_326566

/-- Given a function f(x) = √(kx² - 4x + 3), prove that for f to have a domain of ℝ, 
    k must be in the range [4/3, +∞). -/
theorem function_domain_condition (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (k * x^2 - 4 * x + 3)) ↔ k ≥ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_function_domain_condition_l3265_326566


namespace NUMINAMATH_CALUDE_bottle_cap_distance_difference_l3265_326539

/-- Calculates the total distance traveled by Jenny's bottle cap -/
def jennys_distance : ℝ := 18 + 6 + 7.2 + 3.6 + 3.96

/-- Calculates the total distance traveled by Mark's bottle cap -/
def marks_distance : ℝ := 15 + 30 + 34.5 + 25.875 + 24.58125 + 7.374375 + 9.21796875

/-- The difference in distance between Mark's and Jenny's bottle caps -/
def distance_difference : ℝ := marks_distance - jennys_distance

theorem bottle_cap_distance_difference :
  distance_difference = 107.78959375 := by sorry

end NUMINAMATH_CALUDE_bottle_cap_distance_difference_l3265_326539


namespace NUMINAMATH_CALUDE_shortest_side_in_triangle_l3265_326554

theorem shortest_side_in_triangle (A B C : Real) (a b c : Real) :
  B = 45 * π / 180 →  -- Convert 45° to radians
  C = 60 * π / 180 →  -- Convert 60° to radians
  c = 1 →
  A + B + C = π →     -- Sum of angles in a triangle
  a / Real.sin A = b / Real.sin B →  -- Law of Sines
  b / Real.sin B = c / Real.sin C →  -- Law of Sines
  b < a ∧ b < c →     -- b is the shortest side
  b = Real.sqrt 6 / 3 :=
by sorry

end NUMINAMATH_CALUDE_shortest_side_in_triangle_l3265_326554


namespace NUMINAMATH_CALUDE_x_minus_y_equals_four_l3265_326530

theorem x_minus_y_equals_four (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_four_l3265_326530


namespace NUMINAMATH_CALUDE_twelfth_number_with_digit_sum_12_l3265_326569

/-- A function that returns the sum of the digits of a positive integer -/
def digit_sum (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 12 -/
def nth_number_with_digit_sum_12 (n : ℕ+) : ℕ+ := sorry

/-- Theorem stating that the 12th number with digit sum 12 is 165 -/
theorem twelfth_number_with_digit_sum_12 : 
  nth_number_with_digit_sum_12 12 = 165 := by sorry

end NUMINAMATH_CALUDE_twelfth_number_with_digit_sum_12_l3265_326569


namespace NUMINAMATH_CALUDE_vector_parallelism_l3265_326541

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 1]
def b (x : ℝ) : Fin 2 → ℝ := ![2, x]

-- Define the condition for parallel vectors
def parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

-- State the theorem
theorem vector_parallelism (x : ℝ) :
  parallel (λ i => a i + b x i) (λ i => a i - b x i) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallelism_l3265_326541


namespace NUMINAMATH_CALUDE_max_m_value_min_objective_value_l3265_326523

-- Define the inequality function
def inequality (x m : ℝ) : Prop := |x - 3| + |x - m| ≥ 2 * m

-- Theorem for the maximum value of m
theorem max_m_value : 
  (∀ x : ℝ, inequality x 1) ∧ 
  (∀ m : ℝ, m > 1 → ∃ x : ℝ, ¬(inequality x m)) :=
sorry

-- Define the constraint function
def constraint (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1

-- Define the objective function
def objective (a b c : ℝ) : ℝ := 4 * a^2 + 9 * b^2 + c^2

-- Theorem for the minimum value of the objective function
theorem min_objective_value :
  (∀ a b c : ℝ, constraint a b c → objective a b c ≥ 36/49) ∧
  (∃ a b c : ℝ, constraint a b c ∧ objective a b c = 36/49 ∧ 
    a = 9/49 ∧ b = 4/49 ∧ c = 36/49) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_min_objective_value_l3265_326523


namespace NUMINAMATH_CALUDE_sheridan_fish_problem_l3265_326593

/-- The problem of calculating Mrs. Sheridan's initial number of fish -/
theorem sheridan_fish_problem (fish_from_sister fish_total : ℕ) 
  (h1 : fish_from_sister = 47)
  (h2 : fish_total = 69)
  (h3 : fish_total = fish_from_sister + initial_fish) :
  initial_fish = 22 :=
by
  sorry

#check sheridan_fish_problem

end NUMINAMATH_CALUDE_sheridan_fish_problem_l3265_326593


namespace NUMINAMATH_CALUDE_trey_decorations_l3265_326521

theorem trey_decorations (total : ℕ) (nails thumbtacks sticky : ℕ) : 
  (nails = (2 * total) / 3) →
  (thumbtacks = (2 * (total - nails)) / 5) →
  (sticky = total - nails - thumbtacks) →
  (sticky = 15) →
  (nails = 50) := by
  sorry

end NUMINAMATH_CALUDE_trey_decorations_l3265_326521


namespace NUMINAMATH_CALUDE_cooking_time_for_remaining_potatoes_l3265_326504

/-- Given a chef cooking potatoes with the following conditions:
  - The total number of potatoes to cook is 16
  - The number of potatoes already cooked is 7
  - Each potato takes 5 minutes to cook
  Prove that the time required to cook the remaining potatoes is 45 minutes. -/
theorem cooking_time_for_remaining_potatoes :
  ∀ (total_potatoes cooked_potatoes cooking_time_per_potato : ℕ),
    total_potatoes = 16 →
    cooked_potatoes = 7 →
    cooking_time_per_potato = 5 →
    (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 45 :=
by sorry

end NUMINAMATH_CALUDE_cooking_time_for_remaining_potatoes_l3265_326504


namespace NUMINAMATH_CALUDE_M_intersect_N_is_empty_l3265_326545

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = x + 2}

-- Define the set N
def N : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

-- Theorem statement
theorem M_intersect_N_is_empty : M ∩ (N.image Prod.snd) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_is_empty_l3265_326545


namespace NUMINAMATH_CALUDE_prob_at_most_two_cars_is_one_sixth_l3265_326573

/-- The number of cars in the metro train -/
def num_cars : ℕ := 6

/-- The number of deceased passengers -/
def num_deceased : ℕ := 4

/-- The probability that at most two cars have deceased passengers -/
def prob_at_most_two_cars : ℚ := 1 / 6

/-- Theorem stating that the probability of at most two cars having deceased passengers is 1/6 -/
theorem prob_at_most_two_cars_is_one_sixth :
  prob_at_most_two_cars = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_at_most_two_cars_is_one_sixth_l3265_326573


namespace NUMINAMATH_CALUDE_unique_solution_l3265_326578

theorem unique_solution : ∃! (x y z : ℕ), 
  2 ≤ x ∧ x ≤ y ∧ y ≤ z ∧
  (x * y) % z = 1 ∧
  (x * z) % y = 1 ∧
  (y * z) % x = 1 ∧
  x = 2 ∧ y = 3 ∧ z = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l3265_326578


namespace NUMINAMATH_CALUDE_x_eq_y_sufficient_not_necessary_for_abs_x_eq_abs_y_l3265_326520

theorem x_eq_y_sufficient_not_necessary_for_abs_x_eq_abs_y :
  (∀ x y : ℝ, x = y → |x| = |y|) ∧
  (∃ x y : ℝ, |x| = |y| ∧ x ≠ y) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_y_sufficient_not_necessary_for_abs_x_eq_abs_y_l3265_326520
