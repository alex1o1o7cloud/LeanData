import Mathlib

namespace NUMINAMATH_CALUDE_max_value_3m_plus_4n_l3925_392505

theorem max_value_3m_plus_4n (m n : ℕ) (even_nums : Finset ℕ) (odd_nums : Finset ℕ) : 
  m = 15 →
  even_nums.card = m →
  odd_nums.card = n →
  (∀ x ∈ even_nums, x % 2 = 0 ∧ x > 0) →
  (∀ x ∈ odd_nums, x % 2 = 1 ∧ x > 0) →
  (even_nums.sum id + odd_nums.sum id = 1987) →
  (3 * m + 4 * n ≤ 221) :=
by sorry

end NUMINAMATH_CALUDE_max_value_3m_plus_4n_l3925_392505


namespace NUMINAMATH_CALUDE_bernardo_win_smallest_number_l3925_392589

def game_winner (N : ℕ) : Prop :=
  N ≤ 999 ∧ 8 * N + 600 < 1000 ∧ 8 * N + 700 ≥ 1000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem bernardo_win_smallest_number :
  ∃ (N : ℕ), N = 38 ∧ game_winner N ∧
  (∀ (M : ℕ), M < N → ¬game_winner M) ∧
  sum_of_digits N = 11 :=
sorry

end NUMINAMATH_CALUDE_bernardo_win_smallest_number_l3925_392589


namespace NUMINAMATH_CALUDE_direction_vector_b_l3925_392564

def point_1 : ℝ × ℝ := (-3, 4)
def point_2 : ℝ × ℝ := (2, -1)

theorem direction_vector_b (b : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ (point_2.1 - point_1.1, point_2.2 - point_1.2) = (k * b, k * (-1))) → 
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_b_l3925_392564


namespace NUMINAMATH_CALUDE_doodads_produced_l3925_392552

/-- Represents the production rate of gizmos per worker per hour -/
def gizmo_rate (workers : ℕ) (hours : ℕ) (gizmos : ℕ) : ℚ :=
  (gizmos : ℚ) / ((workers : ℚ) * (hours : ℚ))

/-- Represents the production rate of doodads per worker per hour -/
def doodad_rate (workers : ℕ) (hours : ℕ) (doodads : ℕ) : ℚ :=
  (doodads : ℚ) / ((workers : ℚ) * (hours : ℚ))

/-- Theorem stating the number of doodads produced by 40 workers in 4 hours -/
theorem doodads_produced
  (h1 : gizmo_rate 80 2 160 = gizmo_rate 70 3 210)
  (h2 : doodad_rate 80 2 240 = doodad_rate 70 3 420)
  (h3 : gizmo_rate 40 4 160 = gizmo_rate 80 2 160) :
  (doodad_rate 80 2 240 * (40 : ℚ) * 4) = 320 := by
  sorry

end NUMINAMATH_CALUDE_doodads_produced_l3925_392552


namespace NUMINAMATH_CALUDE_min_lcm_ac_l3925_392544

theorem min_lcm_ac (a b c : ℕ+) (hab : Nat.lcm a b = 12) (hbc : Nat.lcm b c = 15) :
  ∃ (a' b' c' : ℕ+), Nat.lcm a' b' = 12 ∧ Nat.lcm b' c' = 15 ∧ 
  Nat.lcm a' c' = 20 ∧ ∀ (x y : ℕ+), Nat.lcm x y ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_min_lcm_ac_l3925_392544


namespace NUMINAMATH_CALUDE_ellipse_product_l3925_392593

/-- Given an ellipse with center O, major axis AB, minor axis CD, and focus F,
    prove that if OF = 8 and the diameter of the inscribed circle of triangle OCF is 4,
    then (AB)(CD) = 240 -/
theorem ellipse_product (O A B C D F : ℝ × ℝ) : 
  let OA := dist O A
  let OB := dist O B
  let OC := dist O C
  let OD := dist O D
  let OF := dist O F
  let a := OA
  let b := OC
  let inscribed_diameter := 4
  (OA = OB) →  -- A and B are equidistant from O (major axis)
  (OC = OD) →  -- C and D are equidistant from O (minor axis)
  (a > b) →    -- major axis is longer than minor axis
  (OF = 8) →   -- given condition
  (b + OF - a = inscribed_diameter / 2) →  -- inradius formula for triangle OCF
  (2 * a) * (2 * b) = 240 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_product_l3925_392593


namespace NUMINAMATH_CALUDE_lily_account_balance_l3925_392507

def calculate_remaining_balance (initial_balance shirt_cost book_cost num_books gift_percentage : ℚ) : ℚ :=
  let remaining_after_shirt := initial_balance - shirt_cost
  let shoe_cost := 3 * shirt_cost
  let remaining_after_shoes := remaining_after_shirt - shoe_cost
  let total_book_cost := book_cost * num_books
  let remaining_after_books := remaining_after_shoes - total_book_cost
  let gift_cost := gift_percentage * remaining_after_books
  remaining_after_books - gift_cost

theorem lily_account_balance :
  calculate_remaining_balance 55 7 4 5 0.2 = 5.6 := by
  sorry

end NUMINAMATH_CALUDE_lily_account_balance_l3925_392507


namespace NUMINAMATH_CALUDE_power_equality_l3925_392557

theorem power_equality (m n : ℕ) (h1 : 3^m = 5) (h2 : 9^n = 10) : 3^(m+2*n) = 50 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3925_392557


namespace NUMINAMATH_CALUDE_decimal_digits_sum_l3925_392515

theorem decimal_digits_sum (a : ℕ) : ∃ (n m : ℕ),
  (10^(n-1) ≤ a ∧ a < 10^n) ∧
  (10^(3*(n-1)) ≤ a^3 ∧ a^3 < 10^(3*n)) ∧
  (3*n - 2 ≤ m ∧ m ≤ 3*n) →
  n + m ≠ 2001 := by
sorry

end NUMINAMATH_CALUDE_decimal_digits_sum_l3925_392515


namespace NUMINAMATH_CALUDE_fair_hair_percentage_l3925_392558

/-- Given a company where 10% of all employees are women with fair hair,
    and 40% of fair-haired employees are women,
    prove that 25% of all employees have fair hair. -/
theorem fair_hair_percentage
  (total_employees : ℝ)
  (women_fair_hair_percentage : ℝ)
  (women_among_fair_hair_percentage : ℝ)
  (h1 : women_fair_hair_percentage = 0.1)
  (h2 : women_among_fair_hair_percentage = 0.4)
  : (women_fair_hair_percentage / women_among_fair_hair_percentage) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_fair_hair_percentage_l3925_392558


namespace NUMINAMATH_CALUDE_micah_envelope_stamps_l3925_392562

/-- Represents the stamp distribution problem for Micah's envelopes --/
theorem micah_envelope_stamps :
  ∀ (total_stamps : ℕ) 
    (total_envelopes : ℕ) 
    (light_envelopes : ℕ) 
    (stamps_per_light : ℕ) 
    (stamps_per_heavy : ℕ),
  total_stamps = 52 →
  total_envelopes = 14 →
  light_envelopes = 6 →
  stamps_per_light = 2 →
  stamps_per_heavy = 5 →
  total_stamps = light_envelopes * stamps_per_light + 
                 (total_envelopes - light_envelopes) * stamps_per_heavy :=
by
  sorry


end NUMINAMATH_CALUDE_micah_envelope_stamps_l3925_392562


namespace NUMINAMATH_CALUDE_gcd_problem_l3925_392559

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2027 * k) :
  Nat.gcd (Int.natAbs (b^2 + 7*b + 18)) (Int.natAbs (b + 6)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3925_392559


namespace NUMINAMATH_CALUDE_range_of_even_quadratic_function_l3925_392553

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the domain of the function
def domain (a : ℝ) : Set ℝ := Set.Icc (1 + a) 2

-- State the theorem
theorem range_of_even_quadratic_function (a b : ℝ) :
  is_even (f a b) ∧ (∀ x ∈ domain a, f a b x ∈ Set.Icc (-10) 2) →
  Set.range (f a b) = Set.Icc (-10) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_even_quadratic_function_l3925_392553


namespace NUMINAMATH_CALUDE_parabola_coefficient_ratio_l3925_392566

/-- A parabola with equation y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given two points on a parabola with the same y-coordinate and x-coordinates
    equidistant from x = 1, the ratio a/b of the parabola coefficients is -1/2 -/
theorem parabola_coefficient_ratio 
  (p : Parabola) 
  (A B : Point) 
  (h1 : A.x = -1 ∧ A.y = 2) 
  (h2 : B.x = 3 ∧ B.y = 2) 
  (h3 : A.y = p.a * A.x^2 + p.b * A.x + p.c) 
  (h4 : B.y = p.a * B.x^2 + p.b * B.x + p.c) :
  p.a / p.b = -1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_ratio_l3925_392566


namespace NUMINAMATH_CALUDE_first_degree_function_theorem_l3925_392594

/-- A first-degree function from ℝ to ℝ -/
structure FirstDegreeFunction where
  f : ℝ → ℝ
  k : ℝ
  b : ℝ
  h : ∀ x, f x = k * x + b
  k_nonzero : k ≠ 0

/-- Theorem: If f is a first-degree function satisfying f(f(x)) = 4x + 9 for all x,
    then f(x) = 2x + 3 or f(x) = -2x - 9 -/
theorem first_degree_function_theorem (f : FirstDegreeFunction) 
  (h : ∀ x, f.f (f.f x) = 4 * x + 9) :
  (∀ x, f.f x = 2 * x + 3) ∨ (∀ x, f.f x = -2 * x - 9) := by
  sorry

end NUMINAMATH_CALUDE_first_degree_function_theorem_l3925_392594


namespace NUMINAMATH_CALUDE_binary_111011_equals_59_l3925_392536

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- The binary representation of the number we're converting -/
def binary_111011 : List Bool := [true, true, true, false, true, true]

/-- Theorem stating that the decimal representation of 111011(2) is 59 -/
theorem binary_111011_equals_59 : binary_to_decimal binary_111011 = 59 := by
  sorry

end NUMINAMATH_CALUDE_binary_111011_equals_59_l3925_392536


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l3925_392573

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1)
  (h2 : Complex.abs b = 1)
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b^2 * c) + b^3 / (a^2 * c) + c^3 / (a^2 * b) = 1) :
  Complex.abs (a + b + c) = Real.sqrt 15 / 3 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l3925_392573


namespace NUMINAMATH_CALUDE_min_value_sin_cos_l3925_392517

theorem min_value_sin_cos (α β : ℝ) (h1 : α ≥ 0) (h2 : β ≥ 0) (h3 : α + β ≤ 2 * Real.pi) :
  Real.sin α + 2 * Real.cos β ≥ -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sin_cos_l3925_392517


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3925_392581

open Real

theorem trigonometric_simplification (α x : ℝ) :
  ((sin (π - α) * cos (3*π - α) * tan (-α - π) * tan (α - 2*π)) / 
   (tan (4*π - α) * sin (5*π + α)) = sin α) ∧
  ((sin (3*π - x) / tan (5*π - x)) * 
   (1 / (tan (5*π/2 - x) * tan (4.5*π - x))) * 
   (cos (2*π - x) / sin (-x)) = sin x) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3925_392581


namespace NUMINAMATH_CALUDE_tire_price_problem_l3925_392545

theorem tire_price_problem (total_cost : ℝ) (fifth_tire_cost : ℝ) :
  total_cost = 485 →
  fifth_tire_cost = 5 →
  ∃ (regular_price : ℝ),
    4 * regular_price + fifth_tire_cost = total_cost ∧
    regular_price = 120 := by
  sorry

end NUMINAMATH_CALUDE_tire_price_problem_l3925_392545


namespace NUMINAMATH_CALUDE_opposite_face_is_U_l3925_392546

-- Define the faces of the cube
inductive Face : Type
  | P | Q | R | S | T | U

-- Define the property of being adjacent in the net
def adjacent_in_net : Face → Face → Prop :=
  sorry

-- Define the property of being opposite in the cube
def opposite_in_cube : Face → Face → Prop :=
  sorry

-- State the theorem
theorem opposite_face_is_U :
  (adjacent_in_net Face.P Face.Q) →
  (adjacent_in_net Face.P Face.R) →
  (adjacent_in_net Face.P Face.S) →
  (¬adjacent_in_net Face.P Face.T ∨ ¬adjacent_in_net Face.P Face.U) →
  opposite_in_cube Face.P Face.U :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_face_is_U_l3925_392546


namespace NUMINAMATH_CALUDE_max_notebooks_purchase_l3925_392578

theorem max_notebooks_purchase (available : ℚ) (cost : ℚ) : 
  available = 12 → cost = 1.25 → 
  ⌊available / cost⌋ = 9 := by sorry

end NUMINAMATH_CALUDE_max_notebooks_purchase_l3925_392578


namespace NUMINAMATH_CALUDE_smallest_positive_largest_negative_smallest_absolute_l3925_392501

theorem smallest_positive_largest_negative_smallest_absolute (triangle : ℕ) (O : ℤ) (square : ℚ) : 
  (∀ n : ℕ, n > 0 → triangle ≤ n) →
  (∀ z : ℤ, z < 0 → z ≤ O) →
  (∀ q : ℚ, q ≠ 0 → |square| ≤ |q|) →
  triangle > 0 →
  O < 0 →
  (square + triangle) * O = -1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_largest_negative_smallest_absolute_l3925_392501


namespace NUMINAMATH_CALUDE_project_completion_time_project_completion_time_solution_l3925_392599

/-- Represents the project completion time problem -/
theorem project_completion_time 
  (initial_workers : ℕ) 
  (initial_days : ℕ) 
  (additional_workers : ℕ) 
  (efficiency_improvement : ℚ) : ℕ :=
  let total_work := initial_workers * initial_days
  let new_workers := initial_workers + additional_workers
  let new_efficiency := 1 + efficiency_improvement
  let new_daily_work := new_workers * new_efficiency
  ⌊(total_work / new_daily_work : ℚ)⌋₊
    
/-- The solution to the specific problem instance -/
theorem project_completion_time_solution :
  project_completion_time 10 20 5 (1/10) = 12 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_project_completion_time_solution_l3925_392599


namespace NUMINAMATH_CALUDE_leak_drain_time_l3925_392538

/-- Given a pump that can fill a tank in 2 hours, and with a leak it takes 2 1/3 hours to fill the tank,
    prove that the time it takes for the leak to drain all the water of the tank is 14 hours. -/
theorem leak_drain_time (pump_fill_time leak_fill_time : ℚ) : 
  pump_fill_time = 2 →
  leak_fill_time = 7/3 →
  (1 / (1 / pump_fill_time - 1 / leak_fill_time)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_leak_drain_time_l3925_392538


namespace NUMINAMATH_CALUDE_matt_jump_time_l3925_392540

/-- Given that Matt skips rope 3 times per second and gets 1800 skips in total,
    prove that he jumped for 10 minutes. -/
theorem matt_jump_time (skips_per_second : ℕ) (total_skips : ℕ) (jump_time : ℕ) :
  skips_per_second = 3 →
  total_skips = 1800 →
  jump_time * 60 * skips_per_second = total_skips →
  jump_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_matt_jump_time_l3925_392540


namespace NUMINAMATH_CALUDE_no_real_solutions_system_l3925_392549

theorem no_real_solutions_system :
  ¬∃ (x y z : ℝ), (x + y = 3) ∧ (3*x*y - z^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_system_l3925_392549


namespace NUMINAMATH_CALUDE_unknown_cube_edge_length_l3925_392569

/-- The edge length of the unknown cube -/
def x : ℝ := 6

/-- The volume of a cube given its edge length -/
def cube_volume (edge : ℝ) : ℝ := edge ^ 3

theorem unknown_cube_edge_length :
  let cube1_edge : ℝ := 8
  let cube2_edge : ℝ := 10
  let new_cube_edge : ℝ := 12
  cube_volume new_cube_edge = cube_volume cube1_edge + cube_volume cube2_edge + cube_volume x :=
by sorry

end NUMINAMATH_CALUDE_unknown_cube_edge_length_l3925_392569


namespace NUMINAMATH_CALUDE_intersection_min_a_l3925_392584

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := (1/2)^x
def g (a x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem intersection_min_a (a : ℝ) (x₀ y₀ : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : x₀ ≥ 2) 
  (h4 : f x₀ = g a x₀) 
  (h5 : y₀ = f x₀) :
  a ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_intersection_min_a_l3925_392584


namespace NUMINAMATH_CALUDE_tank_cart_friction_l3925_392527

/-- The frictional force acting on a tank resting on an accelerating cart --/
theorem tank_cart_friction (m₁ m₂ a μ g : ℝ) (h₁ : m₁ = 3) (h₂ : m₂ = 15) (h₃ : a = 4) (h₄ : μ = 0.6) (h₅ : g = 9.8) :
  let F_friction := m₁ * a
  let F_max_static := μ * m₁ * g
  F_friction ≤ F_max_static ∧ F_friction = 12 := by
  sorry

end NUMINAMATH_CALUDE_tank_cart_friction_l3925_392527


namespace NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_unique_l3925_392571

theorem quadratic_minimum (x : ℝ) : 
  2 * x^2 - 8 * x + 1 ≥ 2 * 2^2 - 8 * 2 + 1 := by
  sorry

theorem quadratic_minimum_unique (x : ℝ) : 
  (2 * x^2 - 8 * x + 1 = 2 * 2^2 - 8 * 2 + 1) → (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_unique_l3925_392571


namespace NUMINAMATH_CALUDE_hexagonal_dish_volume_l3925_392586

/-- Represents a pyramidal frustum formed by bending four regular hexagons attached to a square --/
structure HexagonalDish where
  side_length : ℝ
  volume : ℝ

/-- The volume of the hexagonal dish is √(49/12) cubic meters when the side length is 1 meter --/
theorem hexagonal_dish_volume (dish : HexagonalDish) (h1 : dish.side_length = 1) :
  dish.volume = Real.sqrt (49 / 12) := by
  sorry

#check hexagonal_dish_volume

end NUMINAMATH_CALUDE_hexagonal_dish_volume_l3925_392586


namespace NUMINAMATH_CALUDE_soccer_goals_product_l3925_392583

def first_ten_games : List Nat := [5, 2, 4, 3, 6, 2, 7, 4, 1, 3]

def goals_sum (games : List Nat) : Nat :=
  games.sum

def is_integer (n : ℚ) : Prop :=
  ∃ m : ℤ, n = m

theorem soccer_goals_product :
  ∀ (g11 g12 : Nat),
    g11 < 10 →
    g12 < 10 →
    is_integer ((goals_sum first_ten_games + g11) / 11) →
    is_integer ((goals_sum first_ten_games + g11 + g12) / 12) →
    g11 * g12 = 28 :=
by sorry

end NUMINAMATH_CALUDE_soccer_goals_product_l3925_392583


namespace NUMINAMATH_CALUDE_final_cucumber_count_l3925_392587

/-- Given the sum of carrots and cucumbers is 10, and the number of carrots is 4,
    if 2 more cucumbers are bought, the final number of cucumbers is 8. -/
theorem final_cucumber_count (total : ℕ) (carrots : ℕ) (additional : ℕ) : 
  total = 10 → carrots = 4 → additional = 2 → 
  (total - carrots) + additional = 8 := by
  sorry

end NUMINAMATH_CALUDE_final_cucumber_count_l3925_392587


namespace NUMINAMATH_CALUDE_initial_strawberries_l3925_392555

/-- The number of strawberries Paul picked -/
def picked : ℕ := 78

/-- The total number of strawberries Paul had after picking more -/
def total : ℕ := 120

/-- The initial number of strawberries in Paul's basket -/
def initial : ℕ := total - picked

theorem initial_strawberries : initial + picked = total := by
  sorry

end NUMINAMATH_CALUDE_initial_strawberries_l3925_392555


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l3925_392503

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_eight_ten : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l3925_392503


namespace NUMINAMATH_CALUDE_channel_probabilities_l3925_392526

/-- Represents a binary communication channel with error probabilities -/
structure Channel where
  α : Real
  β : Real
  h_α_pos : 0 < α
  h_α_lt_one : α < 1
  h_β_pos : 0 < β
  h_β_lt_one : β < 1

/-- Probability of receiving 1,0,1 when sending 1,0,1 in single transmission -/
def singleTransProb (c : Channel) : Real :=
  (1 - c.α) * (1 - c.β)^2

/-- Probability of receiving 1,0,1 when sending 1 in triple transmission -/
def tripleTransProb (c : Channel) : Real :=
  c.β * (1 - c.β)^2

/-- Probability of decoding as 1 when sending 1 in triple transmission -/
def tripleTransDecodeProb (c : Channel) : Real :=
  c.β * (1 - c.β)^2 + (1 - c.β)^3

/-- Probability of decoding as 0 when sending 0 in single transmission -/
def singleTransDecodeZeroProb (c : Channel) : Real :=
  1 - c.α

/-- Probability of decoding as 0 when sending 0 in triple transmission -/
def tripleTransDecodeZeroProb (c : Channel) : Real :=
  3 * c.α * (1 - c.α)^2 + (1 - c.α)^3

theorem channel_probabilities (c : Channel) :
  (singleTransProb c = (1 - c.α) * (1 - c.β)^2) ∧
  (tripleTransProb c = c.β * (1 - c.β)^2) ∧
  (tripleTransDecodeProb c = c.β * (1 - c.β)^2 + (1 - c.β)^3) ∧
  (∀ h : 0 < c.α ∧ c.α < 0.5,
    tripleTransDecodeZeroProb c > singleTransDecodeZeroProb c) :=
by sorry

end NUMINAMATH_CALUDE_channel_probabilities_l3925_392526


namespace NUMINAMATH_CALUDE_new_weekly_earnings_l3925_392535

-- Define the original weekly earnings
def original_earnings : ℝ := 60

-- Define the percentage increase
def percentage_increase : ℝ := 0.30

-- Theorem to prove the new weekly earnings
theorem new_weekly_earnings :
  original_earnings * (1 + percentage_increase) = 78 := by
  sorry

end NUMINAMATH_CALUDE_new_weekly_earnings_l3925_392535


namespace NUMINAMATH_CALUDE_min_value_equality_l3925_392585

/-- Given a function f(x) = x^2 + ax, prove that the minimum value of f(f(x)) 
    is equal to the minimum value of f(x) if and only if a ≤ 0 or a ≥ 2. -/
theorem min_value_equality (a : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + a*x
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ y : ℝ, f y = m) →
  (∃ n : ℝ, ∀ x : ℝ, f (f x) ≥ n ∧ ∃ y : ℝ, f (f y) = n) →
  (∃ k : ℝ, (∀ x : ℝ, f x ≥ k ∧ ∃ y : ℝ, f y = k) ∧
            (∀ x : ℝ, f (f x) ≥ k ∧ ∃ y : ℝ, f (f y) = k)) ↔
  (a ≤ 0 ∨ a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_equality_l3925_392585


namespace NUMINAMATH_CALUDE_ratio_problem_l3925_392534

theorem ratio_problem (a b x y : ℕ) : 
  a > b → 
  a - b = 5 → 
  a * 5 = b * 6 → 
  (a - x) * 4 = (b - x) * 5 → 
  (a + y) * 6 = (b + y) * 7 → 
  x = 5 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3925_392534


namespace NUMINAMATH_CALUDE_sarah_max_correct_l3925_392560

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  total_questions : ℕ
  correct_score : ℤ
  incorrect_score : ℤ

/-- Represents a student's exam results. -/
structure ExamResult where
  exam : Exam
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  total_score : ℤ

/-- Checks if the exam result is valid according to the exam rules. -/
def is_valid_result (result : ExamResult) : Prop :=
  result.correct + result.incorrect + result.unanswered = result.exam.total_questions ∧
  result.correct * result.exam.correct_score + result.incorrect * result.exam.incorrect_score = result.total_score

/-- The specific exam Sarah took. -/
def sarah_exam : Exam :=
  { total_questions := 25
  , correct_score := 4
  , incorrect_score := -3 }

/-- Sarah's exam result. -/
def sarah_result (correct : ℕ) : ExamResult :=
  { exam := sarah_exam
  , correct := correct
  , incorrect := (4 * correct - 40) / 3
  , unanswered := 25 - correct - (4 * correct - 40) / 3
  , total_score := 40 }

theorem sarah_max_correct :
  ∀ c : ℕ, c > 13 → ¬(is_valid_result (sarah_result c)) ∧
  is_valid_result (sarah_result 13) :=
sorry

end NUMINAMATH_CALUDE_sarah_max_correct_l3925_392560


namespace NUMINAMATH_CALUDE_unique_favorite_number_l3925_392591

def is_favorite_number (n : ℕ) : Prop :=
  80 < n ∧ n ≤ 130 ∧
  n % 13 = 0 ∧
  n % 3 ≠ 0 ∧
  (n / 100 + (n / 10) % 10 + n % 10) % 4 = 0

theorem unique_favorite_number : ∃! n, is_favorite_number n :=
  sorry

end NUMINAMATH_CALUDE_unique_favorite_number_l3925_392591


namespace NUMINAMATH_CALUDE_sequence_sum_l3925_392502

theorem sequence_sum (x : Fin 10 → ℝ) 
  (h : ∀ i : Fin 9, x i + 2 * x (i.succ) = 1) :
  x 0 + 512 * x 9 = 171 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3925_392502


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_450_l3925_392531

/-- The number of perfect square factors of 450 -/
def perfect_square_factors_of_450 : ℕ :=
  (Finset.filter (fun n => n^2 ∣ 450) (Finset.range (450 + 1))).card

/-- Theorem: The number of perfect square factors of 450 is 4 -/
theorem count_perfect_square_factors_450 : perfect_square_factors_of_450 = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_450_l3925_392531


namespace NUMINAMATH_CALUDE_even_function_inequality_l3925_392525

/-- Given a function f(x) = a^(|x+b|) where a > 0, a ≠ 1, b ∈ ℝ, and f is even, prove f(b-3) < f(a+2) -/
theorem even_function_inequality (a b : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(|x + b|)
  (∀ x, f x = f (-x)) →
  f (b - 3) < f (a + 2) := by
sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3925_392525


namespace NUMINAMATH_CALUDE_secretary_project_hours_l3925_392554

/-- Proves that given three secretaries whose work times are in the ratio of 2:3:5 and who worked a combined total of 80 hours, the secretary who worked the longest spent 40 hours on the project. -/
theorem secretary_project_hours (t1 t2 t3 : ℝ) : 
  t1 + t2 + t3 = 80 ∧ 
  t2 = (3/2) * t1 ∧ 
  t3 = (5/2) * t1 → 
  t3 = 40 := by
sorry

end NUMINAMATH_CALUDE_secretary_project_hours_l3925_392554


namespace NUMINAMATH_CALUDE_least_positive_angle_theta_l3925_392522

/-- The least positive angle θ (in degrees) satisfying cos 10° = sin 15° + sin θ is 32.5° -/
theorem least_positive_angle_theta : 
  let θ : ℝ := 32.5
  ∀ φ : ℝ, 0 < φ ∧ φ < θ → Real.cos (10 * π / 180) ≠ Real.sin (15 * π / 180) + Real.sin (φ * π / 180) ∧
  Real.cos (10 * π / 180) = Real.sin (15 * π / 180) + Real.sin (θ * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_least_positive_angle_theta_l3925_392522


namespace NUMINAMATH_CALUDE_g_difference_theorem_l3925_392597

/-- The function g(x) = 3x^2 + x - 4 -/
def g (x : ℝ) : ℝ := 3 * x^2 + x - 4

/-- Theorem stating that [g(x+h) - g(x)] - [g(x) - g(x-h)] = 6h^2 for all real x and h -/
theorem g_difference_theorem (x h : ℝ) : 
  (g (x + h) - g x) - (g x - g (x - h)) = 6 * h^2 := by
  sorry

end NUMINAMATH_CALUDE_g_difference_theorem_l3925_392597


namespace NUMINAMATH_CALUDE_solve_equation_and_evaluate_l3925_392533

theorem solve_equation_and_evaluate (x : ℝ) : 
  (5 * x - 7 = 15 * x + 21) → 3 * (x + 10) = 21.6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_and_evaluate_l3925_392533


namespace NUMINAMATH_CALUDE_ashley_cocktail_calories_l3925_392590

/-- Represents the ingredients of Ashley's cocktail -/
structure Cocktail :=
  (mango_juice : ℝ)
  (honey : ℝ)
  (water : ℝ)
  (vodka : ℝ)

/-- Calculates the total calories in the cocktail -/
def total_calories (c : Cocktail) : ℝ :=
  c.mango_juice * 0.6 + c.honey * 6.4 + c.vodka * 0.7

/-- Calculates the total weight of the cocktail -/
def total_weight (c : Cocktail) : ℝ :=
  c.mango_juice + c.honey + c.water + c.vodka

/-- Ashley's cocktail recipe -/
def ashley_cocktail : Cocktail :=
  { mango_juice := 150
  , honey := 200
  , water := 300
  , vodka := 100 }

/-- Theorem stating that 300g of Ashley's cocktail contains 576 calories -/
theorem ashley_cocktail_calories :
  (300 / total_weight ashley_cocktail) * total_calories ashley_cocktail = 576 := by
  sorry


end NUMINAMATH_CALUDE_ashley_cocktail_calories_l3925_392590


namespace NUMINAMATH_CALUDE_binary_multiplication_correct_l3925_392519

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNumber := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : ℕ) : BinaryNumber :=
  sorry

/-- Converts a binary number to its decimal representation -/
def toDecimal (b : BinaryNumber) : ℕ :=
  sorry

/-- Multiplies two binary numbers -/
def binaryMultiply (a b : BinaryNumber) : BinaryNumber :=
  sorry

theorem binary_multiplication_correct :
  binaryMultiply (toBinary 13) (toBinary 7) = toBinary 91 :=
by sorry

end NUMINAMATH_CALUDE_binary_multiplication_correct_l3925_392519


namespace NUMINAMATH_CALUDE_rectangle_to_square_l3925_392500

theorem rectangle_to_square (x y : ℚ) :
  (x - 5 = y + 2) →
  (x * y = (x - 5) * (y + 2)) →
  (x = 25/3 ∧ y = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l3925_392500


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_50_and_288_l3925_392550

theorem smallest_n_divisible_by_50_and_288 :
  ∃ (n : ℕ), n > 0 ∧ 
    50 ∣ n^2 ∧ 
    288 ∣ n^3 ∧ 
    ∀ (m : ℕ), m > 0 → 50 ∣ m^2 → 288 ∣ m^3 → n ≤ m :=
by
  use 60
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_50_and_288_l3925_392550


namespace NUMINAMATH_CALUDE_cost_of_traveling_specific_roads_l3925_392508

/-- Calculates the cost of traveling two intersecting roads on a rectangular lawn. -/
def cost_of_traveling_roads (lawn_length lawn_width road_width cost_per_sqm : ℕ) : ℕ :=
  let road1_area := road_width * lawn_width
  let road2_area := road_width * lawn_length
  let intersection_area := road_width * road_width
  let total_road_area := road1_area + road2_area - intersection_area
  total_road_area * cost_per_sqm

/-- Proves that the cost of traveling two intersecting roads on a specific rectangular lawn is 6500. -/
theorem cost_of_traveling_specific_roads :
  cost_of_traveling_roads 80 60 10 5 = 6500 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_traveling_specific_roads_l3925_392508


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l3925_392509

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  (|x| - 2) / (x - 2) = 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l3925_392509


namespace NUMINAMATH_CALUDE_puddle_depth_calculation_l3925_392513

/-- Represents the rainfall rate in centimeters per hour -/
def rainfall_rate : ℝ := 10

/-- Represents the duration of rainfall in hours -/
def rainfall_duration : ℝ := 3

/-- Represents the base area of the puddle in square centimeters -/
def puddle_base_area : ℝ := 300

/-- Calculates the depth of the puddle given the rainfall rate and duration -/
def puddle_depth : ℝ := rainfall_rate * rainfall_duration

theorem puddle_depth_calculation :
  puddle_depth = 30 := by sorry

end NUMINAMATH_CALUDE_puddle_depth_calculation_l3925_392513


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l3925_392551

/-- Represents a regular polygon -/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ
  diagonals : ℕ

/-- Calculate the number of diagonals in a polygon with n sides -/
def diagonalsCount (n : ℕ) : ℕ :=
  n * (n - 3) / 2

/-- Calculate the perimeter of a regular polygon -/
def perimeter (p : RegularPolygon) : ℝ :=
  p.sides * p.sideLength

theorem regular_polygon_properties :
  ∃ (p : RegularPolygon),
    p.diagonals = 15 ∧
    p.sideLength = 6 ∧
    p.sides = 7 ∧
    perimeter p = 42 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l3925_392551


namespace NUMINAMATH_CALUDE_surface_sum_bounds_l3925_392548

/-- Represents a standard die with 6 faces -/
structure Die :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7)

/-- Represents the large cube assembled from smaller dice -/
structure LargeCube :=
  (dice : Fin 125 → Die)

/-- The sum of visible numbers on the large cube's surface -/
def surface_sum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the bounds of the surface sum -/
theorem surface_sum_bounds (cube : LargeCube) :
  210 ≤ surface_sum cube ∧ surface_sum cube ≤ 840 := by
  sorry

end NUMINAMATH_CALUDE_surface_sum_bounds_l3925_392548


namespace NUMINAMATH_CALUDE_max_volume_open_top_box_l3925_392592

/-- Given a square sheet metal of width 60 cm, the maximum volume of an open-top box 
    with a square base that can be created from it is 16000 cm³. -/
theorem max_volume_open_top_box (sheet_width : ℝ) (h : sheet_width = 60) :
  ∃ (x : ℝ), 0 < x ∧ x < sheet_width / 2 ∧
  (∀ (y : ℝ), 0 < y → y < sheet_width / 2 → 
    x * (sheet_width - 2 * x)^2 ≥ y * (sheet_width - 2 * y)^2) ∧
  x * (sheet_width - 2 * x)^2 = 16000 :=
by sorry


end NUMINAMATH_CALUDE_max_volume_open_top_box_l3925_392592


namespace NUMINAMATH_CALUDE_inscribed_circles_theorem_l3925_392576

theorem inscribed_circles_theorem (N : ℕ) (r : ℝ) (h_pos : r > 0) : 
  let R := N * r
  let area_small_circles := N * Real.pi * r^2
  let area_large_circle := Real.pi * R^2
  let area_remaining := area_large_circle - area_small_circles
  (area_small_circles / area_remaining = 1 / 3) → N = 4 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circles_theorem_l3925_392576


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_WXYZ_l3925_392541

-- Define the quadrilateral WXYZ
structure Quadrilateral :=
  (W X Y Z : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex_quadrilateral_WXYZ (q : Quadrilateral) : Prop :=
  -- We assume the quadrilateral is convex without explicitly defining it
  True

def has_correct_side_lengths (q : Quadrilateral) : Prop :=
  let (xw, yw) := q.W
  let (xx, xy) := q.X
  let (xy', yy) := q.Y
  let (xz, yz) := q.Z
  (xz - xw)^2 + (yz - yw)^2 = 9^2 ∧  -- WZ = 9
  (xy' - xx)^2 + (yy - xy)^2 = 5^2 ∧  -- XY = 5
  (xz - xy')^2 + (yz - yy)^2 = 12^2 ∧  -- YZ = 12
  (xw - xy')^2 + (yw - yy)^2 = 15^2    -- YW = 15

def has_right_angle_WXY (q : Quadrilateral) : Prop :=
  let (xw, yw) := q.W
  let (xx, xy) := q.X
  let (xy', yy) := q.Y
  (xw - xx) * (xy' - xx) + (yw - xy) * (yy - xy) = 0  -- ∠WXY = 90°

-- Define the area of the quadrilateral
def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem area_of_quadrilateral_WXYZ (q : Quadrilateral) 
  (h1 : is_convex_quadrilateral_WXYZ q)
  (h2 : has_correct_side_lengths q)
  (h3 : has_right_angle_WXY q) :
  area q = 76.5 := by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_WXYZ_l3925_392541


namespace NUMINAMATH_CALUDE_altitude_of_equal_area_triangle_trapezoid_l3925_392598

/-- The altitude of a triangle and trapezoid with equal areas -/
theorem altitude_of_equal_area_triangle_trapezoid
  (h : ℝ) -- altitude
  (b : ℝ) -- base of the triangle
  (m : ℝ) -- median of the trapezoid
  (h_pos : h > 0) -- altitude is positive
  (b_val : b = 24) -- base of triangle is 24 inches
  (m_val : m = b / 2) -- median of trapezoid is half of triangle base
  (area_eq : 1/2 * b * h = m * h) -- areas are equal
  : h ∈ Set.Ioi 0 :=
by sorry

end NUMINAMATH_CALUDE_altitude_of_equal_area_triangle_trapezoid_l3925_392598


namespace NUMINAMATH_CALUDE_largest_b_value_l3925_392579

theorem largest_b_value : 
  ∃ (b : ℚ), (2 * b + 5) * (b - 1) = 6 * b ∧ 
  ∀ (x : ℚ), (2 * x + 5) * (x - 1) = 6 * x → x ≤ b ∧ 
  b = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_largest_b_value_l3925_392579


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l3925_392530

theorem greatest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, y > x → 7 - 5 * y + y^2 ≥ 28) ∧ 
  (7 - 5 * x + x^2 < 28) → 
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l3925_392530


namespace NUMINAMATH_CALUDE_bat_survey_result_l3925_392521

theorem bat_survey_result (total : ℕ) 
  (blind_percent : ℚ) (deaf_percent : ℚ) (deaf_count : ℕ) 
  (h1 : blind_percent = 784/1000) 
  (h2 : deaf_percent = 532/1000) 
  (h3 : deaf_count = 33) : total = 79 :=
by
  sorry

end NUMINAMATH_CALUDE_bat_survey_result_l3925_392521


namespace NUMINAMATH_CALUDE_expression_equality_l3925_392512

theorem expression_equality : 
  (84 + 4 / 19 : ℚ) * (1375 / 1000 : ℚ) + (105 + 5 / 19 : ℚ) * (9 / 10 : ℚ) = 210 + 10 / 19 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3925_392512


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3925_392542

theorem quadratic_equation_roots (c : ℝ) :
  (2 + Real.sqrt 3 : ℝ) ^ 2 - 4 * (2 + Real.sqrt 3) + c = 0 →
  (2 - Real.sqrt 3 : ℝ) ^ 2 - 4 * (2 - Real.sqrt 3) + c = 0 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3925_392542


namespace NUMINAMATH_CALUDE_fourth_grade_students_at_end_of_year_l3925_392514

theorem fourth_grade_students_at_end_of_year 
  (initial_students : ℝ) 
  (students_left : ℝ) 
  (students_transferred : ℝ) 
  (h1 : initial_students = 42.0)
  (h2 : students_left = 4.0)
  (h3 : students_transferred = 10.0) :
  initial_students - students_left - students_transferred = 28.0 := by
sorry

end NUMINAMATH_CALUDE_fourth_grade_students_at_end_of_year_l3925_392514


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3925_392543

theorem geometric_sequence_common_ratio
  (x : ℝ)
  (h : ∃ r : ℝ, (x + Real.log 2 / Real.log 27) * r = x + Real.log 2 / Real.log 9 ∧
                (x + Real.log 2 / Real.log 9) * r = x + Real.log 2 / Real.log 3) :
  ∃ r : ℝ, r = 3 ∧
    (x + Real.log 2 / Real.log 27) * r = x + Real.log 2 / Real.log 9 ∧
    (x + Real.log 2 / Real.log 9) * r = x + Real.log 2 / Real.log 3 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3925_392543


namespace NUMINAMATH_CALUDE_exists_student_won_all_l3925_392574

/-- Represents a competition --/
def Competition := Fin 44

/-- Represents a student --/
structure Student where
  id : ℕ

/-- The set of students who won a given competition --/
def winners : Competition → Finset Student :=
  sorry

/-- The number of competitions a student has won --/
def wins (s : Student) : ℕ :=
  sorry

/-- Statement: There exists a student who won all competitions --/
theorem exists_student_won_all :
  (∀ c : Competition, (winners c).card = 7) →
  (∀ c₁ c₂ : Competition, c₁ ≠ c₂ → ∃! s : Student, s ∈ winners c₁ ∧ s ∈ winners c₂) →
  ∃ s : Student, ∀ c : Competition, s ∈ winners c :=
sorry

end NUMINAMATH_CALUDE_exists_student_won_all_l3925_392574


namespace NUMINAMATH_CALUDE_possible_x_values_l3925_392520

theorem possible_x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 10) (h2 : y + 1 / x = 5 / 12) :
  x = 4 ∨ x = 6 := by
sorry

end NUMINAMATH_CALUDE_possible_x_values_l3925_392520


namespace NUMINAMATH_CALUDE_sin_arccos_three_fifths_l3925_392523

theorem sin_arccos_three_fifths : Real.sin (Real.arccos (3/5)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_three_fifths_l3925_392523


namespace NUMINAMATH_CALUDE_digits_zeros_equality_l3925_392575

/-- Count the number of digits in a natural number -/
def countDigits (n : ℕ) : ℕ := sorry

/-- Count the number of zeros in a natural number -/
def countZeros (n : ℕ) : ℕ := sorry

/-- Sum of digits in a sequence from 1 to n -/
def sumDigits (n : ℕ) : ℕ := (Finset.range n).sum (λ i => countDigits (i + 1))

/-- Sum of zeros in a sequence from 1 to n -/
def sumZeros (n : ℕ) : ℕ := (Finset.range n).sum (λ i => countZeros (i + 1))

/-- Theorem: For any natural number k, the number of all digits in the sequence
    1, 2, 3, ..., 10^k is equal to the number of all zeros in the sequence
    1, 2, 3, ..., 10^(k+1) -/
theorem digits_zeros_equality (k : ℕ) :
  sumDigits (10^k) = sumZeros (10^(k+1)) := by sorry

end NUMINAMATH_CALUDE_digits_zeros_equality_l3925_392575


namespace NUMINAMATH_CALUDE_cubic_factorization_l3925_392524

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3925_392524


namespace NUMINAMATH_CALUDE_max_value_range_l3925_392547

noncomputable def f (a x : ℝ) : ℝ := ((1 - a) * x^2 - a * x + a) / Real.exp x

theorem max_value_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≤ a) ↔ a ∈ Set.Ici (4 / (Real.exp 2 + 5)) :=
sorry

end NUMINAMATH_CALUDE_max_value_range_l3925_392547


namespace NUMINAMATH_CALUDE_xy_value_l3925_392588

theorem xy_value (x y : ℝ) (h : |x - 2*y| + (5*x - 7*y - 3)^2 = 0) : x^y = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3925_392588


namespace NUMINAMATH_CALUDE_point_outside_circle_l3925_392577

theorem point_outside_circle (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 → (x - 1)^2 + (y - 1)^2 > 0) ↔ 
  (0 < m ∧ m < 1/4) ∨ m > 1 := by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l3925_392577


namespace NUMINAMATH_CALUDE_second_month_sale_l3925_392567

theorem second_month_sale (
  average_sale : ℕ)
  (month1_sale : ℕ)
  (month3_sale : ℕ)
  (month4_sale : ℕ)
  (month5_sale : ℕ)
  (month6_sale : ℕ)
  (h1 : average_sale = 6500)
  (h2 : month1_sale = 6635)
  (h3 : month3_sale = 7230)
  (h4 : month4_sale = 6562)
  (h5 : month6_sale = 4791)
  : ∃ (month2_sale : ℕ),
    month2_sale = 13782 ∧
    (month1_sale + month2_sale + month3_sale + month4_sale + month5_sale + month6_sale) / 6 = average_sale :=
by sorry

end NUMINAMATH_CALUDE_second_month_sale_l3925_392567


namespace NUMINAMATH_CALUDE_sin_two_alpha_zero_l3925_392565

theorem sin_two_alpha_zero (α : Real) (f : Real → Real)
  (h1 : ∀ x, f x = Real.sin x - Real.cos x)
  (h2 : f α = 1) : Real.sin (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_zero_l3925_392565


namespace NUMINAMATH_CALUDE_range_of_a_l3925_392582

-- Define the condition from the problem
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → (2 : ℝ)^(-2*x) - Real.log x / Real.log a < 0

-- State the theorem
theorem range_of_a (a : ℝ) : condition a → 1/4 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3925_392582


namespace NUMINAMATH_CALUDE_infinitely_many_triples_divisible_by_p_cubed_l3925_392595

theorem infinitely_many_triples_divisible_by_p_cubed :
  ∀ n : ℕ, ∃ p a b : ℕ,
    p > n ∧
    Nat.Prime p ∧
    a < p ∧
    b < p ∧
    (p^3 : ℕ) ∣ ((a + b)^p - a^p - b^p) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_triples_divisible_by_p_cubed_l3925_392595


namespace NUMINAMATH_CALUDE_quadratic_root_sum_inequality_l3925_392528

theorem quadratic_root_sum_inequality (a b c x₁ : ℝ) (h₁ : x₁ > 0) (h₂ : a * x₁^2 + b * x₁ + c = 0) :
  ∃ x₂ : ℝ, x₂ > 0 ∧ c * x₂^2 + b * x₂ + a = 0 ∧ x₁ + x₂ ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_inequality_l3925_392528


namespace NUMINAMATH_CALUDE_product_is_zero_matrix_l3925_392529

def skew_symmetric_matrix (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, d, -e;
     -d, 0, f;
     e, -f, 0]

def symmetric_matrix (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![d^2, d*e, d*f;
     d*e, e^2, e*f;
     d*f, e*f, f^2]

theorem product_is_zero_matrix (d e f : ℝ) : 
  skew_symmetric_matrix d e f * symmetric_matrix d e f = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_is_zero_matrix_l3925_392529


namespace NUMINAMATH_CALUDE_probability_yellow_or_green_l3925_392504

def yellow_marbles : ℕ := 4
def green_marbles : ℕ := 3
def red_marbles : ℕ := 2
def blue_marbles : ℕ := 1

def total_marbles : ℕ := yellow_marbles + green_marbles + red_marbles + blue_marbles
def favorable_marbles : ℕ := yellow_marbles + green_marbles

theorem probability_yellow_or_green : 
  (favorable_marbles : ℚ) / total_marbles = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_or_green_l3925_392504


namespace NUMINAMATH_CALUDE_non_adjacent_arrangement_count_l3925_392561

/-- The number of ways to arrange 4 boys and 2 girls in a row such that the 2 girls are not adjacent --/
def non_adjacent_arrangements : ℕ := 480

/-- The number of boys --/
def num_boys : ℕ := 4

/-- The number of girls --/
def num_girls : ℕ := 2

/-- The number of spaces available for girls (including ends) --/
def num_spaces : ℕ := num_boys + 1

theorem non_adjacent_arrangement_count :
  non_adjacent_arrangements = num_boys.factorial * (num_spaces.choose num_girls) := by
  sorry

end NUMINAMATH_CALUDE_non_adjacent_arrangement_count_l3925_392561


namespace NUMINAMATH_CALUDE_incorrect_number_calculation_l3925_392556

theorem incorrect_number_calculation (n : ℕ) (initial_avg correct_avg incorrect_num : ℚ) :
  n = 10 ∧ 
  initial_avg = 16 ∧ 
  correct_avg = 18 ∧ 
  incorrect_num = 25 →
  ∃ actual_num : ℚ,
    n * initial_avg + actual_num = n * correct_avg ∧
    actual_num = incorrect_num - (correct_avg - initial_avg) * n ∧
    actual_num = 5 := by sorry

end NUMINAMATH_CALUDE_incorrect_number_calculation_l3925_392556


namespace NUMINAMATH_CALUDE_regular_octagon_diagonal_l3925_392580

theorem regular_octagon_diagonal (s : ℝ) (h : s = 12) : 
  let diagonal := s * Real.sqrt 2
  diagonal = 12 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_regular_octagon_diagonal_l3925_392580


namespace NUMINAMATH_CALUDE_max_n_for_consecutive_product_l3925_392532

theorem max_n_for_consecutive_product (n : ℕ) : 
  (∃ k : ℕ, 9*n^2 + 5*n + 26 = k * (k + 1)) → n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_n_for_consecutive_product_l3925_392532


namespace NUMINAMATH_CALUDE_temperature_data_inconsistency_l3925_392537

theorem temperature_data_inconsistency 
  (x_bar : ℝ) 
  (m : ℝ) 
  (S_squared : ℝ) 
  (h_x_bar : x_bar = 0) 
  (h_m : m = 4) 
  (h_S_squared : S_squared = 15.917) : 
  ¬(|x_bar - m| ≤ Real.sqrt S_squared) := by
  sorry

end NUMINAMATH_CALUDE_temperature_data_inconsistency_l3925_392537


namespace NUMINAMATH_CALUDE_picasso_prints_probability_l3925_392568

/-- The probability of arranging 4 specific items consecutively in a random arrangement of n items -/
def consecutive_probability (n : ℕ) (k : ℕ) : ℚ :=
  if n < k then 0
  else (k.factorial * (n - k + 1).factorial) / n.factorial

theorem picasso_prints_probability :
  consecutive_probability 12 4 = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_picasso_prints_probability_l3925_392568


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3925_392539

theorem max_value_sqrt_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (a_ge : a ≥ -1/2)
  (b_ge : b ≥ -2)
  (c_ge : c ≥ -7/3) :
  ∃ (x y z : ℝ), x + y + z = 3 ∧ 
    x ≥ -1/2 ∧ y ≥ -2 ∧ z ≥ -7/3 ∧
    Real.sqrt (4*x + 2) + Real.sqrt (4*y + 8) + Real.sqrt (4*z + 10) = 4 * Real.sqrt 6 ∧
    ∀ (a b c : ℝ), a + b + c = 3 → a ≥ -1/2 → b ≥ -2 → c ≥ -7/3 →
      Real.sqrt (4*a + 2) + Real.sqrt (4*b + 8) + Real.sqrt (4*c + 10) ≤ 4 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3925_392539


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l3925_392570

theorem parallelogram_side_length 
  (s : ℝ) 
  (angle : ℝ) 
  (area : ℝ) 
  (h₁ : angle = π / 3) -- 60 degrees in radians
  (h₂ : area = 27 * Real.sqrt 3)
  (h₃ : area = 3 * s * s * Real.sin angle) :
  s = 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l3925_392570


namespace NUMINAMATH_CALUDE_marble_203_is_blue_l3925_392572

/-- Represents the color of a marble -/
inductive Color
| Red
| Blue
| Green

/-- The length of one complete cycle of marbles -/
def cycleLength : Nat := 6 + 5 + 4

/-- The position of a marble within its cycle -/
def positionInCycle (n : Nat) : Nat :=
  n % cycleLength

/-- The color of a marble at a given position within a cycle -/
def colorInCycle (pos : Nat) : Color :=
  if pos ≤ 6 then Color.Red
  else if pos ≤ 11 then Color.Blue
  else Color.Green

/-- The color of the nth marble in the sequence -/
def marbleColor (n : Nat) : Color :=
  colorInCycle (positionInCycle n)

/-- Theorem: The 203rd marble is blue -/
theorem marble_203_is_blue : marbleColor 203 = Color.Blue := by
  sorry

end NUMINAMATH_CALUDE_marble_203_is_blue_l3925_392572


namespace NUMINAMATH_CALUDE_probability_diamond_or_ace_l3925_392511

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (target_cards : ℕ)
  (h_total : total_cards = 52)
  (h_target : target_cards = 16)

/-- The probability of drawing at least one target card in two draws with replacement -/
def probability_at_least_one (d : Deck) : ℚ :=
  1 - (((d.total_cards - d.target_cards : ℚ) / d.total_cards) ^ 2)

theorem probability_diamond_or_ace (d : Deck) :
  probability_at_least_one d = 88 / 169 := by
  sorry

end NUMINAMATH_CALUDE_probability_diamond_or_ace_l3925_392511


namespace NUMINAMATH_CALUDE_grid_sum_l3925_392510

theorem grid_sum (X Y Z : ℝ) 
  (row1_sum : 1 + X + 3 = 9)
  (row2_sum : 2 + Y + Z = 9) :
  X + Y + Z = 12 := by sorry

end NUMINAMATH_CALUDE_grid_sum_l3925_392510


namespace NUMINAMATH_CALUDE_cos_72_sin_78_plus_sin_72_sin_12_equals_half_l3925_392563

theorem cos_72_sin_78_plus_sin_72_sin_12_equals_half :
  Real.cos (72 * π / 180) * Real.sin (78 * π / 180) +
  Real.sin (72 * π / 180) * Real.sin (12 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_72_sin_78_plus_sin_72_sin_12_equals_half_l3925_392563


namespace NUMINAMATH_CALUDE_units_digit_sum_cubes_l3925_392506

theorem units_digit_sum_cubes : (24^3 + 17^3) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_cubes_l3925_392506


namespace NUMINAMATH_CALUDE_system_solution_is_one_two_l3925_392516

theorem system_solution_is_one_two :
  ∃! (s : Set ℝ), s = {1, 2} ∧
  (∀ x y : ℝ, (x^4 + y^4 = 17 ∧ x + y = 3) ↔ (x ∈ s ∧ y ∈ s ∧ x ≠ y)) :=
sorry

end NUMINAMATH_CALUDE_system_solution_is_one_two_l3925_392516


namespace NUMINAMATH_CALUDE_markus_marbles_l3925_392596

theorem markus_marbles (mara_bags : ℕ) (mara_marbles_per_bag : ℕ) 
  (markus_bags : ℕ) (markus_extra_marbles : ℕ) :
  mara_bags = 12 →
  mara_marbles_per_bag = 2 →
  markus_bags = 2 →
  markus_extra_marbles = 2 →
  (mara_bags * mara_marbles_per_bag + markus_extra_marbles) / markus_bags = 13 := by
  sorry

end NUMINAMATH_CALUDE_markus_marbles_l3925_392596


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3925_392518

/-- A geometric sequence with common ratio q > 1 -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (4 * a 2005 ^ 2 - 8 * a 2005 + 3 = 0) →
  (4 * a 2006 ^ 2 - 8 * a 2006 + 3 = 0) →
  a 2007 + a 2008 = 18 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3925_392518
