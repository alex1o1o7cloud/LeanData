import Mathlib

namespace NUMINAMATH_CALUDE_table_wobbles_l3072_307277

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a table with four legs -/
structure Table where
  leg1 : Point3D
  leg2 : Point3D
  leg3 : Point3D
  leg4 : Point3D

/-- Checks if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧ 
    a * p1.x + b * p1.y + c * p1.z + d = 0 ∧
    a * p2.x + b * p2.y + c * p2.z + d = 0 ∧
    a * p3.x + b * p3.y + c * p3.z + d = 0 ∧
    a * p4.x + b * p4.y + c * p4.z + d = 0

/-- Defines a square table with given leg lengths -/
def squareTable : Table :=
  { leg1 := ⟨0, 0, 70⟩
  , leg2 := ⟨1, 0, 71⟩
  , leg3 := ⟨1, 1, 72.5⟩
  , leg4 := ⟨0, 1, 72⟩ }

/-- Theorem: The square table with given leg lengths wobbles -/
theorem table_wobbles : ¬areCoplanar squareTable.leg1 squareTable.leg2 squareTable.leg3 squareTable.leg4 := by
  sorry

end NUMINAMATH_CALUDE_table_wobbles_l3072_307277


namespace NUMINAMATH_CALUDE_solution_difference_l3072_307213

theorem solution_difference (p q : ℝ) : 
  (p - 4) * (p + 4) = 17 * p - 68 →
  (q - 4) * (q + 4) = 17 * q - 68 →
  p ≠ q →
  p > q →
  p - q = 9 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l3072_307213


namespace NUMINAMATH_CALUDE_complex_expression_equality_l3072_307214

theorem complex_expression_equality : ((7 - 3*I) - 3*(2 - 5*I)) * I = I - 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l3072_307214


namespace NUMINAMATH_CALUDE_not_all_angles_exceed_90_l3072_307218

/-- A plane quadrilateral is a geometric figure with four sides and four angles in a plane. -/
structure PlaneQuadrilateral where
  angles : Fin 4 → ℝ
  sum_of_angles : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 360

/-- Theorem: In a plane quadrilateral, it is impossible for all four internal angles to exceed 90°. -/
theorem not_all_angles_exceed_90 (q : PlaneQuadrilateral) : 
  ¬(∀ i : Fin 4, q.angles i > 90) := by
  sorry

end NUMINAMATH_CALUDE_not_all_angles_exceed_90_l3072_307218


namespace NUMINAMATH_CALUDE_triangle_angles_from_area_equation_l3072_307245

theorem triangle_angles_from_area_equation (α β γ : Real) (a b c : Real) (t : Real) :
  α = 43 * Real.pi / 180 →
  γ + β + α = Real.pi →
  2 * t = a * b * Real.sqrt (Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin α * Real.sin β) →
  β = 17 * Real.pi / 180 ∧ γ = 120 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_from_area_equation_l3072_307245


namespace NUMINAMATH_CALUDE_smallest_a_is_2_pow_16_l3072_307227

/-- The number of factors of a natural number n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- The smallest natural number satisfying the given condition -/
def smallest_a : ℕ := sorry

/-- The theorem statement -/
theorem smallest_a_is_2_pow_16 :
  (∀ a : ℕ, num_factors (a^2) = num_factors a + 16 → a ≥ smallest_a) ∧
  num_factors (smallest_a^2) = num_factors smallest_a + 16 ∧
  smallest_a = 2^16 := by sorry

end NUMINAMATH_CALUDE_smallest_a_is_2_pow_16_l3072_307227


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3072_307229

theorem simplify_and_evaluate (x y : ℝ) (hx : x = 2023) (hy : y = 2) :
  (x + 2*y)^2 - (x^3 + 4*x^2*y) / x = 16 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3072_307229


namespace NUMINAMATH_CALUDE_nine_pointed_star_angle_sum_l3072_307297

/-- A 9-pointed star is formed by connecting every fourth point of 9 evenly spaced points on a circle. -/
structure NinePointedStar where
  /-- The number of points on the circle -/
  num_points : ℕ
  /-- The number of points to skip when forming the star -/
  skip_points : ℕ
  /-- The number of points is 9 -/
  points_eq_nine : num_points = 9
  /-- We skip every 3 points (connect every 4th) -/
  skip_three : skip_points = 3

/-- The sum of the angles at the tips of a 9-pointed star is 540 degrees -/
theorem nine_pointed_star_angle_sum (star : NinePointedStar) : 
  (star.num_points : ℝ) * (360 / (2 * star.num_points : ℝ) * star.skip_points) = 540 := by
  sorry

end NUMINAMATH_CALUDE_nine_pointed_star_angle_sum_l3072_307297


namespace NUMINAMATH_CALUDE_hannahs_quarters_l3072_307294

def is_valid_quarter_count (n : ℕ) : Prop :=
  40 < n ∧ n < 400 ∧
  n % 6 = 3 ∧
  n % 7 = 3 ∧
  n % 8 = 3

theorem hannahs_quarters :
  ∀ n : ℕ, is_valid_quarter_count n ↔ (n = 171 ∨ n = 339) :=
by sorry

end NUMINAMATH_CALUDE_hannahs_quarters_l3072_307294


namespace NUMINAMATH_CALUDE_stone_blocks_per_step_l3072_307211

theorem stone_blocks_per_step 
  (levels : ℕ) 
  (steps_per_level : ℕ) 
  (total_blocks : ℕ) 
  (h1 : levels = 4) 
  (h2 : steps_per_level = 8) 
  (h3 : total_blocks = 96) : 
  total_blocks / (levels * steps_per_level) = 3 := by
sorry

end NUMINAMATH_CALUDE_stone_blocks_per_step_l3072_307211


namespace NUMINAMATH_CALUDE_negative_number_identification_l3072_307239

theorem negative_number_identification :
  let a := -3^2
  let b := (-3)^2
  let c := |-3|
  let d := -(-3)
  (a < 0) ∧ (b ≥ 0) ∧ (c ≥ 0) ∧ (d ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_number_identification_l3072_307239


namespace NUMINAMATH_CALUDE_employed_females_percentage_is_16_percent_l3072_307278

/-- Represents an age group in Town X -/
inductive AgeGroup
  | Young    -- 18-34
  | Middle   -- 35-54
  | Senior   -- 55+

/-- The percentage of employed population in each age group -/
def employed_percentage : ℝ := 64

/-- The percentage of employed males in each age group -/
def employed_males_percentage : ℝ := 48

/-- The percentage of employed females in each age group -/
def employed_females_percentage : ℝ := employed_percentage - employed_males_percentage

theorem employed_females_percentage_is_16_percent :
  employed_females_percentage = 16 := by sorry

end NUMINAMATH_CALUDE_employed_females_percentage_is_16_percent_l3072_307278


namespace NUMINAMATH_CALUDE_polynomial_coefficient_square_difference_l3072_307235

theorem polynomial_coefficient_square_difference (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_square_difference_l3072_307235


namespace NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l3072_307201

/-- Given a point M with coordinates (3,-4), its symmetric point M' about the x-axis has coordinates (3,4). -/
theorem symmetric_point_about_x_axis :
  let M : ℝ × ℝ := (3, -4)
  let M' : ℝ × ℝ := (M.1, -M.2)
  M' = (3, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l3072_307201


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3072_307230

theorem polynomial_simplification (p : ℝ) :
  (2 * p^4 + 5 * p^3 - 3 * p + 4) + (-p^4 + 2 * p^3 - 7 * p^2 + 4 * p - 2) =
  p^4 + 7 * p^3 - 7 * p^2 + p + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3072_307230


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l3072_307224

theorem min_value_2x_plus_y :
  ∀ x y : ℝ, (|y| ≤ 2 - x ∧ x ≥ -1) → (∀ x' y' : ℝ, |y'| ≤ 2 - x' ∧ x' ≥ -1 → 2*x + y ≤ 2*x' + y') ∧ (∃ x₀ y₀ : ℝ, |y₀| ≤ 2 - x₀ ∧ x₀ ≥ -1 ∧ 2*x₀ + y₀ = -5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l3072_307224


namespace NUMINAMATH_CALUDE_ratio_from_percentage_l3072_307236

theorem ratio_from_percentage (x y : ℝ) (h : y = x * (1 - 0.909090909090909)) :
  x / y = 11 := by
sorry

end NUMINAMATH_CALUDE_ratio_from_percentage_l3072_307236


namespace NUMINAMATH_CALUDE_frog_jumps_l3072_307206

-- Define the hexagon vertices
inductive Vertex : Type
| A | B | C | D | E | F

-- Define the neighbor relation
def isNeighbor : Vertex → Vertex → Prop :=
  sorry

-- Define the number of paths from A to C in n jumps
def numPaths (n : ℕ) : ℕ :=
  if n % 2 = 0 then (1/3) * (4^(n/2) - 1) else 0

-- Define the number of paths from A to C in n jumps avoiding D
def numPathsAvoidD (n : ℕ) : ℕ :=
  if n % 2 = 0 then 3^(n/2 - 1) else 0

-- Define the survival probability after n jumps with mine at D
def survivalProb (n : ℕ) : ℚ :=
  if n % 2 = 0 then (3/4)^(n/2 - 1) else (3/4)^((n-1)/2)

-- Define the expected lifespan
def expectedLifespan : ℚ := 9

-- Main theorem
theorem frog_jumps :
  (∀ n : ℕ, numPaths n = if n % 2 = 0 then (1/3) * (4^(n/2) - 1) else 0) ∧
  (∀ n : ℕ, numPathsAvoidD n = if n % 2 = 0 then 3^(n/2 - 1) else 0) ∧
  (∀ n : ℕ, survivalProb n = if n % 2 = 0 then (3/4)^(n/2 - 1) else (3/4)^((n-1)/2)) ∧
  expectedLifespan = 9 :=
sorry

end NUMINAMATH_CALUDE_frog_jumps_l3072_307206


namespace NUMINAMATH_CALUDE_solution_verification_l3072_307260

theorem solution_verification :
  let x : ℚ := 425
  let y : ℝ := (270 + 90 * Real.sqrt 2) / 7
  (x - (11/17) * x = 150) ∧ (y - ((Real.sqrt 2)/3) * y = 90) := by sorry

end NUMINAMATH_CALUDE_solution_verification_l3072_307260


namespace NUMINAMATH_CALUDE_quadratic_intersection_with_x_axis_l3072_307246

theorem quadratic_intersection_with_x_axis (a b c : ℝ) :
  ∃ x : ℝ, (x - a) * (x - b) - c^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_with_x_axis_l3072_307246


namespace NUMINAMATH_CALUDE_manager_salary_is_220000_l3072_307247

/-- Represents the average salary of managers at Plutarch Enterprises -/
def manager_salary : ℝ := 220000

/-- Theorem stating that the average salary of managers at Plutarch Enterprises is $220,000 -/
theorem manager_salary_is_220000 
  (marketer_percent : ℝ) (engineer_percent : ℝ) (sales_percent : ℝ) (manager_percent : ℝ)
  (marketer_salary : ℝ) (engineer_salary : ℝ) (sales_salary : ℝ) (total_avg_salary : ℝ)
  (h1 : marketer_percent = 0.60)
  (h2 : engineer_percent = 0.20)
  (h3 : sales_percent = 0.10)
  (h4 : manager_percent = 0.10)
  (h5 : marketer_salary = 50000)
  (h6 : engineer_salary = 80000)
  (h7 : sales_salary = 70000)
  (h8 : total_avg_salary = 75000)
  (h9 : marketer_percent + engineer_percent + sales_percent + manager_percent = 1) :
  manager_salary = 220000 := by
  sorry

#check manager_salary_is_220000

end NUMINAMATH_CALUDE_manager_salary_is_220000_l3072_307247


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3072_307225

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Define set B
def B : Set ℝ := {x | 2*x - 3 > 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (3/2) 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3072_307225


namespace NUMINAMATH_CALUDE_system_solvability_l3072_307298

/-- The system of equations has real solutions if and only if a, b, c form a triangle -/
theorem system_solvability (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∃ x y z : ℝ, a * x + b * y - c * z = 0 ∧
               a * Real.sqrt (1 - x^2) + b * Real.sqrt (1 - y^2) - c * Real.sqrt (1 - z^2) = 0)
  ↔ (abs (a - b) ≤ c ∧ c ≤ a + b) :=
by sorry

end NUMINAMATH_CALUDE_system_solvability_l3072_307298


namespace NUMINAMATH_CALUDE_rock_collection_problem_l3072_307287

theorem rock_collection_problem (minerals_yesterday : ℕ) (gemstones : ℕ) (new_minerals : ℕ) :
  gemstones = minerals_yesterday / 2 →
  new_minerals = 6 →
  gemstones = 21 →
  minerals_yesterday + new_minerals = 48 :=
by sorry

end NUMINAMATH_CALUDE_rock_collection_problem_l3072_307287


namespace NUMINAMATH_CALUDE_final_ball_properties_l3072_307284

/-- Represents the types of balls in the simulator -/
inductive BallType
  | A
  | B
  | C

/-- Represents the state of the simulator -/
structure SimulatorState where
  a_count : Nat
  b_count : Nat
  c_count : Nat

/-- Defines the collision rules -/
def collide (t1 t2 : BallType) : BallType :=
  match t1, t2 with
  | BallType.A, BallType.A => BallType.C
  | BallType.B, BallType.B => BallType.C
  | BallType.C, BallType.C => BallType.C
  | BallType.A, BallType.B => BallType.C
  | BallType.B, BallType.A => BallType.C
  | BallType.A, BallType.C => BallType.B
  | BallType.C, BallType.A => BallType.B
  | BallType.B, BallType.C => BallType.A
  | BallType.C, BallType.B => BallType.A

/-- The initial state of the simulator -/
def initial_state : SimulatorState :=
  { a_count := 12, b_count := 9, c_count := 10 }

/-- Defines a valid final state with only one ball remaining -/
def is_valid_final_state (s : SimulatorState) : Prop :=
  s.a_count + s.b_count + s.c_count = 1

/-- Represents a sequence of collisions -/
def collision_sequence := List (BallType × BallType)

/-- Applies a collision sequence to a simulator state -/
def apply_collisions : SimulatorState → collision_sequence → SimulatorState
  | s, [] => s
  | s, (t1, t2) :: rest => sorry  -- Implementation details omitted

theorem final_ball_properties :
  ∃ (seq : collision_sequence), 
    let final_state := apply_collisions initial_state seq
    is_valid_final_state final_state ∧ 
    final_state.a_count = 1 ∧
    ∀ (seq' : collision_sequence), 
      let final_state' := apply_collisions initial_state seq'
      is_valid_final_state final_state' → 
      final_state'.c_count ≠ 1 :=
by sorry


end NUMINAMATH_CALUDE_final_ball_properties_l3072_307284


namespace NUMINAMATH_CALUDE_sammy_cheese_ratio_l3072_307209

/-- Represents the number of slices in each pizza -/
def slices_per_pizza : ℕ := 12

/-- Represents the number of pizzas ordered -/
def total_pizzas : ℕ := 2

/-- Represents the number of slices Dean ate from the Hawaiian pizza -/
def dean_slices : ℕ := slices_per_pizza / 2

/-- Represents the number of slices Frank ate from the Hawaiian pizza -/
def frank_slices : ℕ := 3

/-- Represents the total number of slices left over -/
def leftover_slices : ℕ := 11

/-- Theorem stating the ratio of slices Sammy ate from the cheese pizza to the total slices of the cheese pizza -/
theorem sammy_cheese_ratio :
  ∃ (sammy_slices : ℕ),
    sammy_slices = slices_per_pizza - (leftover_slices - (slices_per_pizza - (dean_slices + frank_slices))) ∧
    sammy_slices * 3 = slices_per_pizza :=
by sorry

end NUMINAMATH_CALUDE_sammy_cheese_ratio_l3072_307209


namespace NUMINAMATH_CALUDE_range_of_a_l3072_307249

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, Real.exp x + Real.log a / a > Real.log x / a) ↔ a > Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3072_307249


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_30_l3072_307228

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ,
    p.Prime ∧ q.Prime ∧
    p > 30 ∧ q > 30 ∧
    p ≠ q ∧
    p * q = 1147 ∧
    ∀ p' q' : ℕ,
      p'.Prime → q'.Prime →
      p' > 30 → q' > 30 →
      p' ≠ q' →
      p' * q' ≥ 1147 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_30_l3072_307228


namespace NUMINAMATH_CALUDE_equivalence_proof_l3072_307271

variable (P Q : Prop)

theorem equivalence_proof :
  (P → Q) ↔ (¬Q → ¬P) ∧ (¬P ∨ Q) ∧ ¬((Q → P) ↔ (P → Q)) :=
sorry

end NUMINAMATH_CALUDE_equivalence_proof_l3072_307271


namespace NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l3072_307233

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 - a*x + 3*b = 0) 
  (h2 : ∃ x : ℝ, x^2 - 3*b*x + a = 0) : 
  a + b ≥ 3.3442 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l3072_307233


namespace NUMINAMATH_CALUDE_no_solution_mod_five_l3072_307243

theorem no_solution_mod_five : ¬∃ (n : ℕ), n^2 % 5 = 1 ∧ n^3 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_mod_five_l3072_307243


namespace NUMINAMATH_CALUDE_original_cost_of_dvd_pack_l3072_307263

theorem original_cost_of_dvd_pack (discount : ℕ) (price_after_discount : ℕ) 
  (h1 : discount = 25)
  (h2 : price_after_discount = 51) :
  discount + price_after_discount = 76 := by
sorry

end NUMINAMATH_CALUDE_original_cost_of_dvd_pack_l3072_307263


namespace NUMINAMATH_CALUDE_inverse_f_l3072_307285

/-- Given a function f: ℝ → ℝ satisfying f(4) = 3 and f(2x) = 2f(x) + 1 for all x,
    prove that f(128) = 127 -/
theorem inverse_f (f : ℝ → ℝ) (h1 : f 4 = 3) (h2 : ∀ x, f (2 * x) = 2 * f x + 1) :
  f 128 = 127 := by sorry

end NUMINAMATH_CALUDE_inverse_f_l3072_307285


namespace NUMINAMATH_CALUDE_only_lottery_is_random_l3072_307250

-- Define the events
def event_A := "No moisture, seed germination"
def event_B := "At least 2 people out of 367 have the same birthday"
def event_C := "Melting of ice at -1°C under standard pressure"
def event_D := "Xiao Ying bought a lottery ticket and won a 5 million prize"

-- Define a predicate for random events
def is_random_event (e : String) : Prop := sorry

-- Theorem stating that only event_D is a random event
theorem only_lottery_is_random :
  ¬(is_random_event event_A) ∧
  ¬(is_random_event event_B) ∧
  ¬(is_random_event event_C) ∧
  is_random_event event_D :=
by sorry

end NUMINAMATH_CALUDE_only_lottery_is_random_l3072_307250


namespace NUMINAMATH_CALUDE_odd_function_property_l3072_307290

/-- A function f(x) = ax^5 - bx^3 + cx is odd and f(-3) = 7 implies f(3) = -7 -/
theorem odd_function_property (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^5 - b * x^3 + c * x)
  (h2 : f (-3) = 7) : 
  f 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3072_307290


namespace NUMINAMATH_CALUDE_soccer_stars_league_teams_l3072_307208

theorem soccer_stars_league_teams (n : ℕ) : n > 1 → (n * (n - 1)) / 2 = 28 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_soccer_stars_league_teams_l3072_307208


namespace NUMINAMATH_CALUDE_sheep_problem_l3072_307240

theorem sheep_problem (n : ℕ) (h1 : n > 0) :
  let total := n * n
  let remainder := total % 10
  let elder_share := total - remainder
  let younger_share := remainder
  (remainder < 10 ∧ elder_share % 20 = 10) →
  (elder_share + younger_share + 2) / 2 = (elder_share + 2) / 2 ∧
  (elder_share + younger_share + 2) / 2 = (younger_share + 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_sheep_problem_l3072_307240


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3072_307248

theorem quadratic_equation_roots (k : ℝ) :
  let f := fun x : ℝ => x^2 + (2*k - 1)*x + k^2
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0) →
  (k < 1/4 ∧
   (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 = 0 → f x2 = 0 → x1 + x2 + x1*x2 - 1 = 0 → k = 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3072_307248


namespace NUMINAMATH_CALUDE_valid_pairs_l3072_307253

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digits_form_arithmetic_sequence (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ : ℕ), n = 100 * d₁ + 10 * d₂ + d₃ ∧ 
    d₁ < d₂ ∧ d₂ < d₃ ∧ d₂ - d₁ = d₃ - d₂

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def all_digits_same (n : ℕ) : Prop :=
  ∃ (d : ℕ), n = d * 11111

theorem valid_pairs : 
  ∀ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ≤ b ∧ 
    is_three_digit (a + b) ∧ 
    digits_form_arithmetic_sequence (a + b) ∧
    is_five_digit (a * b) ∧
    all_digits_same (a * b) →
  ((a = 41 ∧ b = 271) ∨ 
   (a = 164 ∧ b = 271) ∨ 
   (a = 82 ∧ b = 542) ∨ 
   (a = 123 ∧ b = 813)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l3072_307253


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3072_307295

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt ((6 / (x + 1)) - 1)}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 2*x + 3)}

-- Statement to prove
theorem intersection_A_complement_B : 
  A ∩ (Set.univ \ B) = {x : ℝ | 3 ≤ x ∧ x ≤ 5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3072_307295


namespace NUMINAMATH_CALUDE_refund_calculation_l3072_307296

/-- Calculates the refund amount for returned cans given specific conditions -/
theorem refund_calculation (total_cans brand_a_price brand_b_price average_price discount restocking_fee tax : ℚ)
  (h1 : total_cans = 6)
  (h2 : brand_a_price = 33 / 100)
  (h3 : brand_b_price = 40 / 100)
  (h4 : average_price = 365 / 1000)
  (h5 : discount = 20 / 100)
  (h6 : restocking_fee = 5 / 100)
  (h7 : tax = 8 / 100)
  (h8 : ∃ (brand_a_count brand_b_count : ℚ), 
    brand_a_count + brand_b_count = total_cans ∧ 
    brand_a_count * brand_a_price + brand_b_count * brand_b_price = total_cans * average_price ∧
    brand_a_count > brand_b_count) :
  ∃ (refund : ℚ), refund = 55 / 100 := by
  sorry


end NUMINAMATH_CALUDE_refund_calculation_l3072_307296


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l3072_307216

/-- Calculates the total receipts from ticket sales given the ticket prices and quantities sold. -/
def total_receipts (adult_price child_price : ℕ) (adult_qty child_qty : ℕ) : ℕ :=
  adult_price * adult_qty + child_price * child_qty

/-- Proves that the total receipts from ticket sales is $840 given the specified conditions. -/
theorem theater_ticket_sales : 
  let adult_price : ℕ := 12
  let child_price : ℕ := 4
  let total_tickets : ℕ := 130
  let child_tickets : ℕ := 90
  let adult_tickets : ℕ := total_tickets - child_tickets
  total_receipts adult_price child_price adult_tickets child_tickets = 840 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l3072_307216


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l3072_307232

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 1

-- Define the condition p
def condition_p (m : ℝ) : Prop := -1 < m ∧ m < 5

-- Define the condition q
def condition_q (m : ℝ) : Prop :=
  ∀ x, quadratic_equation m x = 0 → -2 < x ∧ x < 4

-- Theorem: p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient :
  (∀ m, condition_q m → condition_p m) ∧
  ¬(∀ m, condition_p m → condition_q m) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l3072_307232


namespace NUMINAMATH_CALUDE_license_plate_difference_l3072_307221

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of letter slots in Georgia license plates -/
def georgia_letters : ℕ := 5

/-- The number of digit slots in Georgia license plates -/
def georgia_digits : ℕ := 2

/-- The number of letter slots in Texas license plates -/
def texas_letters : ℕ := 4

/-- The number of digit slots in Texas license plates -/
def texas_digits : ℕ := 3

/-- The difference in the number of possible license plates between Georgia and Texas -/
theorem license_plate_difference : 
  (num_letters ^ georgia_letters) * (num_digits ^ georgia_digits) - 
  (num_letters ^ texas_letters) * (num_digits ^ texas_digits) = 731161600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l3072_307221


namespace NUMINAMATH_CALUDE_line_parameterization_l3072_307276

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 2 * x - 30

/-- The parameterization of the line -/
def parameterization (g : ℝ → ℝ) (t : ℝ) : ℝ × ℝ := (g t, 18 * t - 10)

/-- The theorem stating that g(t) = 9t + 10 satisfies the line equation and parameterization -/
theorem line_parameterization (g : ℝ → ℝ) :
  (∀ t, line_equation (g t) (18 * t - 10)) ↔ (∀ t, g t = 9 * t + 10) :=
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3072_307276


namespace NUMINAMATH_CALUDE_min_value_theorem_l3072_307265

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 1/b = 2) :
  ∃ (m : ℝ), m = 4 ∧ ∀ x y, x > 0 → y > 0 → x + 1/y = 2 → 2/x + 2*y ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3072_307265


namespace NUMINAMATH_CALUDE_perfect_square_coefficient_l3072_307281

/-- A quadratic trinomial in a and b with coefficient m -/
def quadratic_trinomial (a b : ℝ) (m : ℝ) : ℝ := a^2 + m*a*b + b^2

/-- Definition of a perfect square trinomial -/
def is_perfect_square_trinomial (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (g : ℝ → ℝ), ∀ (x y : ℝ), f x y = (g x + y)^2 ∨ f x y = (g x - y)^2

/-- If a^2 + mab + b^2 is a perfect square trinomial, then m = 2 or m = -2 -/
theorem perfect_square_coefficient (m : ℝ) :
  is_perfect_square_trinomial (quadratic_trinomial · · m) → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_coefficient_l3072_307281


namespace NUMINAMATH_CALUDE_total_wheels_in_lot_l3072_307207

/-- The number of wheels on a standard car -/
def wheels_per_car : ℕ := 4

/-- The number of cars in the parking lot -/
def cars_in_lot : ℕ := 12

/-- Theorem: The total number of car wheels in the parking lot is 48 -/
theorem total_wheels_in_lot : cars_in_lot * wheels_per_car = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_in_lot_l3072_307207


namespace NUMINAMATH_CALUDE_cos_pi_minus_theta_point_l3072_307264

theorem cos_pi_minus_theta_point (θ : Real) :
  (∃ r : Real, r > 0 ∧ r * Real.cos θ = 4 ∧ r * Real.sin θ = -3) →
  Real.cos (Real.pi - θ) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_theta_point_l3072_307264


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3072_307262

/-- 
For a geometric sequence {a_n}, if a_2 + a_4 = 2, 
then a_1a_3 + 2a_2a_4 + a_3a_5 = 4
-/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_sum : a 2 + a 4 = 2) : 
  a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3072_307262


namespace NUMINAMATH_CALUDE_remainder_2007_div_25_l3072_307203

theorem remainder_2007_div_25 : 2007 % 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2007_div_25_l3072_307203


namespace NUMINAMATH_CALUDE_initial_girls_count_l3072_307266

theorem initial_girls_count (b g : ℕ) : 
  (2 * (g - 15) = b) →
  (5 * (b - 45) = g - 15) →
  g = 40 := by
sorry

end NUMINAMATH_CALUDE_initial_girls_count_l3072_307266


namespace NUMINAMATH_CALUDE_probability_red_green_white_l3072_307268

def red_marbles : ℕ := 4
def green_marbles : ℕ := 5
def white_marbles : ℕ := 11

def total_marbles : ℕ := red_marbles + green_marbles + white_marbles

theorem probability_red_green_white :
  (red_marbles : ℚ) / total_marbles *
  green_marbles / (total_marbles - 1) *
  white_marbles / (total_marbles - 2) = 11 / 342 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_green_white_l3072_307268


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l3072_307269

-- Define the geometric sequence
def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ :=
  b * x^2 + c * x + a

-- State the theorem
theorem unique_root_quadratic (a b c : ℝ) :
  is_geometric_sequence a b c →
  a ≤ b →
  b ≤ c →
  c ≤ 1 →
  (∃! x : ℝ, quadratic a b c x = 0) →
  (∃ x : ℝ, quadratic a b c x = 0 ∧ x = -(Real.rpow 4 (1/3) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l3072_307269


namespace NUMINAMATH_CALUDE_f_nonnegative_implies_a_eq_four_l3072_307219

/-- The function f(x) = ax^3 - 3x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3*x + 1

/-- Theorem: If f(x) ≥ 0 for all x in [-1, 1], then a = 4 -/
theorem f_nonnegative_implies_a_eq_four (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_nonnegative_implies_a_eq_four_l3072_307219


namespace NUMINAMATH_CALUDE_inequality_proof_l3072_307273

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) :
  2 / ((a + b) * (c + d)) ≤ 1 / Real.sqrt (a * b) + 1 / Real.sqrt (c * d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3072_307273


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3072_307200

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 9*x - 20) → (∃ y : ℝ, y^2 = 9*y - 20 ∧ x + y = 9) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3072_307200


namespace NUMINAMATH_CALUDE_function_inequality_implies_domain_restriction_l3072_307280

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing
def Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem function_inequality_implies_domain_restriction
  (h_increasing : Increasing f)
  (h_inequality : ∀ x, f x < f (2*x - 3)) :
  ∀ x, f x < f (2*x - 3) → x > 3 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_domain_restriction_l3072_307280


namespace NUMINAMATH_CALUDE_boxes_of_apples_l3072_307202

/-- The number of boxes of apples after processing a delivery -/
def number_of_boxes (apples_per_crate : ℕ) (crates_delivered : ℕ) (rotten_apples : ℕ) (apples_per_box : ℕ) : ℕ :=
  ((apples_per_crate * crates_delivered - rotten_apples) / apples_per_box)

/-- Theorem stating that the number of boxes is 50 given the specific conditions -/
theorem boxes_of_apples :
  number_of_boxes 42 12 4 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_boxes_of_apples_l3072_307202


namespace NUMINAMATH_CALUDE_backpack_music_player_problem_l3072_307222

/-- Represents the prices and discounts for the backpack and music player problem -/
structure Prices where
  backpack : ℝ
  music_player : ℝ
  renmin_discount : ℝ
  carrefour_voucher : ℝ
  budget : ℝ

/-- The conditions of the problem -/
def problem_conditions (p : Prices) : Prop :=
  p.backpack + p.music_player = 452 ∧
  p.music_player = 4 * p.backpack - 8 ∧
  p.renmin_discount = 0.2 ∧
  p.carrefour_voucher = 30 ∧
  p.budget = 400

/-- The cost at Renmin Department Store after discount -/
def renmin_cost (p : Prices) : ℝ :=
  (p.backpack + p.music_player) * (1 - p.renmin_discount)

/-- The cost at Carrefour after applying vouchers -/
def carrefour_cost (p : Prices) : ℝ :=
  p.music_player + p.backpack - 3 * p.carrefour_voucher

/-- The main theorem stating the solution to the problem -/
theorem backpack_music_player_problem (p : Prices) 
  (h : problem_conditions p) : 
  p.backpack = 92 ∧ 
  p.music_player = 360 ∧ 
  renmin_cost p < carrefour_cost p ∧
  renmin_cost p ≤ p.budget ∧
  carrefour_cost p ≤ p.budget :=
sorry

end NUMINAMATH_CALUDE_backpack_music_player_problem_l3072_307222


namespace NUMINAMATH_CALUDE_floor_equality_iff_range_l3072_307275

theorem floor_equality_iff_range (x : ℝ) : 
  ⌊⌊3 * x⌋ - 1⌋ = ⌊x + 1⌋ ↔ 2/3 ≤ x ∧ x < 4/3 :=
by sorry

end NUMINAMATH_CALUDE_floor_equality_iff_range_l3072_307275


namespace NUMINAMATH_CALUDE_tangent_line_to_ellipse_l3072_307261

/-- Given a line y = mx + 3 tangent to the ellipse 4x^2 + y^2 = 4, m^2 = 5 -/
theorem tangent_line_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 3 → 4 * x^2 + y^2 = 4) → 
  (∃! x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + y^2 = 4) → 
  m^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ellipse_l3072_307261


namespace NUMINAMATH_CALUDE_largest_number_from_digits_l3072_307292

def digits : List ℕ := [1, 7, 0]

def formNumber (d : List ℕ) : ℕ :=
  d.foldl (fun acc x => acc * 10 + x) 0

def isPermutation (l1 l2 : List ℕ) : Prop :=
  l1.length = l2.length ∧ ∀ x, x ∈ l1 ↔ x ∈ l2

theorem largest_number_from_digits : 
  ∀ p : List ℕ, isPermutation digits p → formNumber p ≤ 710 :=
sorry

end NUMINAMATH_CALUDE_largest_number_from_digits_l3072_307292


namespace NUMINAMATH_CALUDE_equation_solution_l3072_307220

theorem equation_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + 1 = d + Real.sqrt (a + b + c - d) →
  d = 5/4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3072_307220


namespace NUMINAMATH_CALUDE_mothers_day_bouquet_l3072_307270

/-- Represents the flower shop problem --/
structure FlowerShop where
  carnation_price : ℚ
  rose_price : ℚ
  processing_fee : ℚ
  total_budget : ℚ
  total_flowers : ℕ

/-- Represents a bouquet composition --/
structure Bouquet where
  carnations : ℕ
  roses : ℕ

/-- Checks if a bouquet satisfies the conditions of the flower shop problem --/
def is_valid_bouquet (shop : FlowerShop) (bouquet : Bouquet) : Prop :=
  let total_cost := shop.carnation_price * bouquet.carnations + shop.rose_price * bouquet.roses + shop.processing_fee
  bouquet.carnations + bouquet.roses = shop.total_flowers ∧
  total_cost = shop.total_budget

/-- The main theorem to prove --/
theorem mothers_day_bouquet : 
  let shop := FlowerShop.mk 1.5 2 2 21 10
  let bouquet := Bouquet.mk 2 8
  is_valid_bouquet shop bouquet := by
  sorry

end NUMINAMATH_CALUDE_mothers_day_bouquet_l3072_307270


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3072_307267

/-- Given an inequality (ax-1)(x+1) < 0 with respect to x, where the solution set
    is (-∞, 1/a) ∪ (-1, +∞), prove that the range of the real number a is -1 ≤ a < 0. -/
theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, (a * x - 1) * (x + 1) < 0 ↔ x ∈ ({y : ℝ | y < (1 : ℝ) / a} ∪ {y : ℝ | y > -1})) →
  -1 ≤ a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3072_307267


namespace NUMINAMATH_CALUDE_train_length_l3072_307242

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 18 → ∃ (length : ℝ), 
  (length ≥ 299.5 ∧ length ≤ 300.5) ∧ 
  length = speed * (1000 / 3600) * time := by
  sorry


end NUMINAMATH_CALUDE_train_length_l3072_307242


namespace NUMINAMATH_CALUDE_range_of_x_l3072_307241

theorem range_of_x (x y : ℝ) (h : 4 * x * y + 4 * y^2 + x + 6 = 0) :
  x ≤ -2 ∨ x ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l3072_307241


namespace NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l3072_307256

-- Define the solution sets
def solution_set_1 : Set ℝ := {x | -4 < x ∧ x < 2}
def solution_set_2 : Set ℝ := {x | (-2/3 ≤ x ∧ x < 1/3) ∨ (1 < x ∧ x ≤ 2)}

-- Theorem for the first inequality
theorem inequality_solution_1 : 
  {x : ℝ | |x - 1| + |x + 3| < 6} = solution_set_1 := by sorry

-- Theorem for the second inequality
theorem inequality_solution_2 :
  {x : ℝ | 1 < |3*x - 2| ∧ |3*x - 2| < 4} = solution_set_2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l3072_307256


namespace NUMINAMATH_CALUDE_relationship_holds_l3072_307251

/-- The relationship between x and y is defined by the function f --/
def f (x : ℕ) : ℕ := x^2 + 3*x + 1

/-- The set of x values --/
def X : Finset ℕ := {1, 2, 3, 4}

/-- The corresponding y values for each x in X --/
def Y : Finset ℕ := {5, 13, 25, 41}

/-- A function that checks if a given pair (x, y) satisfies the relationship --/
def satisfies_relationship (pair : ℕ × ℕ) : Prop :=
  f pair.1 = pair.2

theorem relationship_holds : ∀ (x : ℕ), x ∈ X → (x, f x) ∈ X.product Y ∧ satisfies_relationship (x, f x) := by
  sorry

end NUMINAMATH_CALUDE_relationship_holds_l3072_307251


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l3072_307258

theorem vector_magnitude_proof (a b : ℝ × ℝ) :
  let angle := Real.pi / 3
  (a.1^2 + a.2^2 = 1) →
  (b.1^2 + b.2^2 = 4) →
  (a.1 * b.1 + a.2 * b.2 = 2 * Real.cos angle) →
  ((2*a.1 - b.1)^2 + (2*a.2 - b.2)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l3072_307258


namespace NUMINAMATH_CALUDE_town_budget_theorem_l3072_307237

/-- Represents the town's budget allocation problem -/
def TownBudget (total : ℝ) (policing_fraction : ℝ) (education : ℝ) : Prop :=
  let policing := total * policing_fraction
  let remaining := total - policing - education
  remaining = 4

/-- The theorem statement for the town's budget allocation problem -/
theorem town_budget_theorem :
  TownBudget 32 0.5 12 := by
  sorry

end NUMINAMATH_CALUDE_town_budget_theorem_l3072_307237


namespace NUMINAMATH_CALUDE_impossibleDivision_l3072_307259

/-- Represents an employee with their salary -/
structure Employee :=
  (salary : ℝ)

/-- Represents a region with its employees -/
structure Region :=
  (employees : List Employee)

/-- The total salary of a region -/
def totalSalary (r : Region) : ℝ :=
  (r.employees.map Employee.salary).sum

/-- The condition that 10% of employees get 90% of total salary -/
def salaryDistributionCondition (employees : List Employee) : Prop :=
  ∃ (highPaidEmployees : List Employee),
    highPaidEmployees.length = (employees.length / 10) ∧
    (highPaidEmployees.map Employee.salary).sum ≥ 0.9 * ((employees.map Employee.salary).sum)

/-- The condition for a valid region division -/
def validRegionDivision (regions : List Region) : Prop :=
  ∀ r ∈ regions, ∀ subset : List Employee,
    subset.length = (r.employees.length / 10) →
    (subset.map Employee.salary).sum ≤ 0.11 * totalSalary r

/-- The main theorem -/
theorem impossibleDivision :
  ∃ (employees : List Employee),
    salaryDistributionCondition employees ∧
    ¬∃ (regions : List Region),
      (regions.map Region.employees).join = employees ∧
      validRegionDivision regions :=
sorry

end NUMINAMATH_CALUDE_impossibleDivision_l3072_307259


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l3072_307223

theorem circle_area_from_circumference (k : ℝ) : 
  let circumference := 18 * Real.pi
  let radius := circumference / (2 * Real.pi)
  let area := k * Real.pi
  area = Real.pi * radius^2 → k = 81 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l3072_307223


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l3072_307288

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 187 ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 187 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l3072_307288


namespace NUMINAMATH_CALUDE_expression_simplification_l3072_307282

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  ((2 * x - 3) / (x - 2) - 1) / ((x^2 - 2*x + 1) / (x - 2)) = 1 / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3072_307282


namespace NUMINAMATH_CALUDE_repeating_decimal_prime_l3072_307244

/-- A function that determines if a rational number has a repeating decimal representation with a given period length. -/
def has_repeating_decimal_period (q : ℚ) (period : ℕ) : Prop :=
  ∃ (k : ℕ) (r : ℚ), q = k + r ∧ r < 1 ∧ (10 ^ period : ℚ) * r = r

theorem repeating_decimal_prime (n : ℕ) (h1 : n > 1) 
  (h2 : has_repeating_decimal_period (1 / n : ℚ) (n - 1)) : 
  Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_prime_l3072_307244


namespace NUMINAMATH_CALUDE_coat_price_calculation_l3072_307291

/-- The total selling price of a coat after discount and tax -/
def totalSellingPrice (originalPrice discount taxRate : ℝ) : ℝ :=
  let salePrice := originalPrice * (1 - discount)
  salePrice * (1 + taxRate)

/-- Theorem: The total selling price of a $120 coat with 30% discount and 8% tax is $90.72 -/
theorem coat_price_calculation :
  totalSellingPrice 120 0.3 0.08 = 90.72 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_calculation_l3072_307291


namespace NUMINAMATH_CALUDE_triangle_side_expression_l3072_307215

theorem triangle_side_expression (a b c : ℝ) (h1 : a > c) (h2 : a + b > c) (h3 : b + c > a) (h4 : c + a > b) :
  |c - a| - Real.sqrt ((a + c - b) ^ 2) = b - 2 * c :=
sorry

end NUMINAMATH_CALUDE_triangle_side_expression_l3072_307215


namespace NUMINAMATH_CALUDE_tangent_line_determines_coefficients_l3072_307299

theorem tangent_line_determines_coefficients :
  ∀ (a b : ℝ),
  let f : ℝ → ℝ := λ x => x^2 + a*x + b
  let tangent_line : ℝ → ℝ := λ x => x + 1
  (f 0 = 1) →
  (∀ x, tangent_line x = x - f x + 1) →
  (∀ h : ℝ, h ≠ 0 → (f h - f 0) / h = tangent_line 0) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_determines_coefficients_l3072_307299


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3072_307231

theorem regular_polygon_sides (exterior_angle : ℝ) : 
  exterior_angle = 30 → (360 / exterior_angle : ℝ) = 12 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3072_307231


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_with_same_terminal_side_l3072_307238

theorem same_terminal_side (α β : Real) : 
  ∃ k : Int, α = β + 2 * π * (k : Real) → 
  α.cos = β.cos ∧ α.sin = β.sin :=
by sorry

theorem angle_with_same_terminal_side : 
  ∃ k : Int, (11 * π / 8 : Real) = (-5 * π / 8 : Real) + 2 * π * (k : Real) :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_with_same_terminal_side_l3072_307238


namespace NUMINAMATH_CALUDE_blue_to_black_pen_ratio_l3072_307289

/-- Given the conditions of John's pen collection, prove the ratio of blue to black pens --/
theorem blue_to_black_pen_ratio :
  ∀ (blue black red : ℕ),
  blue + black + red = 31 →
  black = red + 5 →
  blue = 18 →
  blue / black = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_to_black_pen_ratio_l3072_307289


namespace NUMINAMATH_CALUDE_mean_temperature_l3072_307254

def temperatures : List ℚ := [-6, -3, -3, -4, 2, 4, 1]

def mean (list : List ℚ) : ℚ :=
  (list.sum) / list.length

theorem mean_temperature : mean temperatures = -6/7 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l3072_307254


namespace NUMINAMATH_CALUDE_printer_pages_theorem_l3072_307257

/-- Represents a printer with specific crumpling and blurring patterns -/
structure Printer where
  crumple_interval : Nat
  blur_interval : Nat

/-- Calculates the number of pages that are neither crumpled nor blurred -/
def good_pages (p : Printer) (total : Nat) : Nat :=
  total - (total / p.crumple_interval + total / p.blur_interval - total / (Nat.lcm p.crumple_interval p.blur_interval))

/-- Theorem: For a printer that crumples every 7th page and blurs every 3rd page,
    if 24 pages are neither crumpled nor blurred, then 42 pages were printed in total -/
theorem printer_pages_theorem (p : Printer) (h1 : p.crumple_interval = 7) (h2 : p.blur_interval = 3) :
  good_pages p 42 = 24 := by
  sorry

#eval good_pages ⟨7, 3⟩ 42  -- Should output 24

end NUMINAMATH_CALUDE_printer_pages_theorem_l3072_307257


namespace NUMINAMATH_CALUDE_blocks_per_color_l3072_307217

theorem blocks_per_color 
  (total_blocks : ℕ) 
  (num_colors : ℕ) 
  (h1 : total_blocks = 49)
  (h2 : num_colors = 7)
  (h3 : num_colors > 0)
  (h4 : total_blocks % num_colors = 0) :
  total_blocks / num_colors = 7 := by
sorry

end NUMINAMATH_CALUDE_blocks_per_color_l3072_307217


namespace NUMINAMATH_CALUDE_problem_statement_l3072_307286

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x > 0, x > 0 → 6 - 1 / x ≤ 9 * x) ∧
  (a^2 + 9 * b^2 + 2 * a * b = a^2 * b^2 → a * b ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3072_307286


namespace NUMINAMATH_CALUDE_farm_animals_l3072_307283

theorem farm_animals (total_legs total_animals : ℕ) 
  (h_legs : total_legs = 38)
  (h_animals : total_animals = 12)
  (h_positive : total_legs > 0 ∧ total_animals > 0) :
  ∃ (chickens sheep : ℕ),
    chickens + sheep = total_animals ∧
    2 * chickens + 4 * sheep = total_legs ∧
    chickens = 5 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l3072_307283


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l3072_307205

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x, (a - 1) * x < a - 1 ↔ x > 1) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l3072_307205


namespace NUMINAMATH_CALUDE_chord_length_l3072_307274

-- Define the circle C
def Circle (m n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - m)^2 + (p.2 - n)^2 = 4}

-- Define the theorem
theorem chord_length
  (m n : ℝ) -- Center of the circle
  (A B : ℝ × ℝ) -- Points on the circle
  (hA : A ∈ Circle m n) -- A is on the circle
  (hB : B ∈ Circle m n) -- B is on the circle
  (hAB : A ≠ B) -- A and B are different points
  (h_sum : ‖(A.1 - m, A.2 - n) + (B.1 - m, B.2 - n)‖ = 2 * Real.sqrt 3) -- |→CA + →CB| = 2√3
  : ‖(A.1 - B.1, A.2 - B.2)‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l3072_307274


namespace NUMINAMATH_CALUDE_sum_abc_equals_22_l3072_307252

theorem sum_abc_equals_22 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (eq1 : a^2 + b*c = 115)
  (eq2 : b^2 + a*c = 127)
  (eq3 : c^2 + a*b = 115) :
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_equals_22_l3072_307252


namespace NUMINAMATH_CALUDE_quadratic_constant_term_l3072_307234

theorem quadratic_constant_term 
  (x : ℝ) 
  (some_number : ℝ) 
  (h1 : x = 0.5) 
  (h2 : 2 * x^2 + 9 * x + some_number = 0) : 
  some_number = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_constant_term_l3072_307234


namespace NUMINAMATH_CALUDE_point_conditions_imply_m_value_l3072_307226

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the point P based on the parameter m -/
def P (m : ℝ) : Point :=
  { x := 3 - m, y := 2 * m + 6 }

/-- Condition: P is in the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Condition: P is equidistant from the coordinate axes -/
def equidistant_from_axes (p : Point) : Prop :=
  abs p.x = abs p.y

/-- Theorem: If P(3-m, 2m+6) is in the fourth quadrant and equidistant from the axes, then m = -9 -/
theorem point_conditions_imply_m_value :
  ∀ m : ℝ, in_fourth_quadrant (P m) ∧ equidistant_from_axes (P m) → m = -9 :=
by sorry

end NUMINAMATH_CALUDE_point_conditions_imply_m_value_l3072_307226


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_bisector_bound_l3072_307272

-- Define the ellipse
def is_on_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the theorem
theorem ellipse_perpendicular_bisector_bound 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (A B : ℝ × ℝ) 
  (h_A : is_on_ellipse A.1 A.2 a b) 
  (h_B : is_on_ellipse B.1 B.2 a b) 
  (x₀ : ℝ) 
  (h_perp_bisector : ∃ (k : ℝ), 
    k * (A.1 - B.1) = A.2 - B.2 ∧ 
    x₀ = (A.1 + B.1) / 2 + k * (A.2 + B.2) / 2) :
  -((a^2 - b^2) / a) < x₀ ∧ x₀ < (a^2 - b^2) / a :=
sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_bisector_bound_l3072_307272


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l3072_307293

theorem isosceles_triangle_condition (a b : ℝ) (A B : ℝ) : 
  0 < a → 0 < b → 0 < A → A < π → 0 < B → B < π →
  a * Real.cos B = b * Real.cos A → A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l3072_307293


namespace NUMINAMATH_CALUDE_bath_frequency_l3072_307279

/-- 
Given a person who takes a bath B times per week and a shower once per week,
prove that if they clean themselves 156 times in 52 weeks, then B = 2.
-/
theorem bath_frequency (B : ℕ) 
  (h1 : B + 1 = (156 : ℕ) / 52) : B = 2 := by
  sorry

#check bath_frequency

end NUMINAMATH_CALUDE_bath_frequency_l3072_307279


namespace NUMINAMATH_CALUDE_log_square_plus_one_neither_sufficient_nor_necessary_l3072_307204

theorem log_square_plus_one_neither_sufficient_nor_necessary (a b : ℝ) :
  ¬(∀ a b : ℝ, (Real.log (a^2 + 1) < Real.log (b^2 + 1)) → (a < b)) ∧
  ¬(∀ a b : ℝ, (a < b) → (Real.log (a^2 + 1) < Real.log (b^2 + 1))) :=
sorry

end NUMINAMATH_CALUDE_log_square_plus_one_neither_sufficient_nor_necessary_l3072_307204


namespace NUMINAMATH_CALUDE_parking_lot_bikes_l3072_307255

theorem parking_lot_bikes (cars : ℕ) (total_wheels : ℕ) (car_wheels : ℕ) (bike_wheels : ℕ) : 
  cars = 14 → total_wheels = 66 → car_wheels = 4 → bike_wheels = 2 → 
  (total_wheels - cars * car_wheels) / bike_wheels = 5 := by
sorry

end NUMINAMATH_CALUDE_parking_lot_bikes_l3072_307255


namespace NUMINAMATH_CALUDE_factor_implies_m_value_l3072_307212

theorem factor_implies_m_value (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - m*x - 15 = (x + 3) * k) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_m_value_l3072_307212


namespace NUMINAMATH_CALUDE_chief_permutations_l3072_307210

/-- The number of letters in the word CHIEF -/
def word_length : ℕ := 5

/-- The total number of permutations of the word CHIEF -/
def total_permutations : ℕ := Nat.factorial word_length

/-- The number of permutations where I appears after E -/
def permutations_i_after_e : ℕ := total_permutations / 2

theorem chief_permutations :
  permutations_i_after_e = total_permutations / 2 :=
by sorry

end NUMINAMATH_CALUDE_chief_permutations_l3072_307210
