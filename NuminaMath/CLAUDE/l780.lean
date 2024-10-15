import Mathlib

namespace NUMINAMATH_CALUDE_cuboid_edge_length_l780_78089

/-- Represents the length of an edge in centimeters -/
def Edge := ℝ

/-- Represents the volume in cubic centimeters -/
def Volume := ℝ

/-- Given a cuboid with edges a, x, and b, and volume v,
    prove that if a = 4, b = 6, and v = 96, then x = 4 -/
theorem cuboid_edge_length (a x b v : ℝ) :
  a = 4 → b = 6 → v = 96 → v = a * x * b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l780_78089


namespace NUMINAMATH_CALUDE_f_max_value_inequality_proof_l780_78042

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| - |2*x + 4|

-- Statement 1: The maximum value of f(x) is 3
theorem f_max_value : ∃ (M : ℝ), M = 3 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

-- Statement 2: For positive real numbers x, y, z such that x + y + z = 3, y²/x + z²/y + x²/z ≥ 3
theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 3) :
  y^2 / x + z^2 / y + x^2 / z ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_f_max_value_inequality_proof_l780_78042


namespace NUMINAMATH_CALUDE_marks_change_factor_l780_78098

theorem marks_change_factor (n : ℕ) (initial_avg final_avg : ℝ) (h1 : n = 10) (h2 : initial_avg = 40) (h3 : final_avg = 80) :
  ∃ (factor : ℝ), factor * (n * initial_avg) = n * final_avg ∧ factor = 2 := by
sorry

end NUMINAMATH_CALUDE_marks_change_factor_l780_78098


namespace NUMINAMATH_CALUDE_cookie_calculation_l780_78062

theorem cookie_calculation (initial_cookies given_cookies received_cookies : ℕ) :
  initial_cookies ≥ given_cookies →
  initial_cookies - given_cookies + received_cookies =
    initial_cookies - given_cookies + received_cookies :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_calculation_l780_78062


namespace NUMINAMATH_CALUDE_binomial_15_12_l780_78049

theorem binomial_15_12 : Nat.choose 15 12 = 2730 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_12_l780_78049


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l780_78084

theorem quadratic_equation_roots : 
  let f : ℝ → ℝ := λ x => x^2 + 2*x - 3
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l780_78084


namespace NUMINAMATH_CALUDE_simplify_expression_l780_78047

theorem simplify_expression (a : ℝ) (ha : a > 0) :
  a^2 / (a^(1/2) * a^(2/3)) = a^(5/6) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l780_78047


namespace NUMINAMATH_CALUDE_multiply_decimal_l780_78086

theorem multiply_decimal : (3.6 : ℝ) * 0.25 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimal_l780_78086


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l780_78004

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x ≥ 0}
def N : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l780_78004


namespace NUMINAMATH_CALUDE_money_distribution_l780_78054

/-- Given three people A, B, and C with some amount of money, prove that B and C together have 340 rupees. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 500 →  -- Total money between A, B, and C
  A + C = 200 →      -- Money A and C have together
  C = 40 →           -- Money C has
  B + C = 340 :=     -- Prove that B and C have 340 together
by sorry

end NUMINAMATH_CALUDE_money_distribution_l780_78054


namespace NUMINAMATH_CALUDE_strips_intersection_angle_l780_78020

/-- A strip is defined as the region between two parallel lines. -/
structure Strip where
  width : ℝ

/-- The intersection of two strips forms a parallelogram. -/
structure StripIntersection where
  strip1 : Strip
  strip2 : Strip
  area : ℝ

/-- The angle between two strips is the angle between their defining lines. -/
def angleBetweenStrips (intersection : StripIntersection) : ℝ := sorry

theorem strips_intersection_angle (intersection : StripIntersection) :
  intersection.strip1.width = 1 →
  intersection.strip2.width = 1 →
  intersection.area = 2 →
  angleBetweenStrips intersection = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_strips_intersection_angle_l780_78020


namespace NUMINAMATH_CALUDE_problem_1_l780_78071

theorem problem_1 (a : ℝ) : a^3 * a + (2*a^2)^2 = 5*a^4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l780_78071


namespace NUMINAMATH_CALUDE_equation_solution_l780_78063

theorem equation_solution (a b c : ℤ) : 
  (∀ x, (x - a) * (x - 12) + 4 = (x + b) * (x + c)) → (a = 7 ∨ a = 17) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l780_78063


namespace NUMINAMATH_CALUDE_adams_farm_animals_l780_78026

theorem adams_farm_animals (cows : ℕ) (sheep : ℕ) (pigs : ℕ) : 
  cows = 12 →
  sheep = 2 * cows →
  pigs = 3 * sheep →
  cows + sheep + pigs = 108 := by
sorry

end NUMINAMATH_CALUDE_adams_farm_animals_l780_78026


namespace NUMINAMATH_CALUDE_binomial_15_4_l780_78096

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_4_l780_78096


namespace NUMINAMATH_CALUDE_bank_queue_properties_l780_78077

/-- Represents a queue of people with different operation times -/
structure BankQueue where
  total_people : Nat
  simple_ops : Nat
  long_ops : Nat
  simple_time : Nat
  long_time : Nat

/-- Calculates the minimum wasted person-minutes -/
def min_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the maximum wasted person-minutes -/
def max_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the expected wasted person-minutes for a random order -/
def expected_wasted_time (q : BankQueue) : Rat :=
  sorry

/-- Theorem stating the properties of the bank queue problem -/
theorem bank_queue_properties (q : BankQueue)
  (h1 : q.total_people = 8)
  (h2 : q.simple_ops = 5)
  (h3 : q.long_ops = 3)
  (h4 : q.simple_time = 1)
  (h5 : q.long_time = 5) :
  min_wasted_time q = 40 ∧
  max_wasted_time q = 100 ∧
  expected_wasted_time q = 84 := by
  sorry

end NUMINAMATH_CALUDE_bank_queue_properties_l780_78077


namespace NUMINAMATH_CALUDE_min_angle_function_l780_78043

/-- For any triangle with internal angles α, β, and γ in radians, 
    the minimum value of 4/α + 1/(β + γ) is 9/π. -/
theorem min_angle_function (α β γ : ℝ) (h1 : 0 < α) (h2 : 0 < β) (h3 : 0 < γ) 
    (h4 : α + β + γ = π) : 
  (∀ α' β' γ' : ℝ, 0 < α' ∧ 0 < β' ∧ 0 < γ' ∧ α' + β' + γ' = π → 
    4 / α + 1 / (β + γ) ≤ 4 / α' + 1 / (β' + γ')) → 
  4 / α + 1 / (β + γ) = 9 / π := by
sorry

end NUMINAMATH_CALUDE_min_angle_function_l780_78043


namespace NUMINAMATH_CALUDE_maria_average_balance_l780_78041

def maria_balance : List ℝ := [50, 250, 100, 200, 150, 250]

theorem maria_average_balance :
  (maria_balance.sum / maria_balance.length : ℝ) = 1000 / 6 := by sorry

end NUMINAMATH_CALUDE_maria_average_balance_l780_78041


namespace NUMINAMATH_CALUDE_hyperbola_C_eccentricity_l780_78065

/-- Hyperbola C with foci F₁ and F₂, and points P and Q satisfying given conditions -/
structure HyperbolaC where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_P_on_C : P.1^2 / a^2 - P.2^2 / b^2 = 1
  h_Q_on_asymptote : Q.2 / Q.1 = b / a
  h_first_quadrant : P.1 > 0 ∧ P.2 > 0 ∧ Q.1 > 0 ∧ Q.2 > 0
  h_QP_eq_PF₂ : (Q.1 - P.1, Q.2 - P.2) = (P.1 - F₂.1, P.2 - F₂.2)
  h_QF₁_perp_QF₂ : (Q.1 - F₁.1) * (Q.1 - F₂.1) + (Q.2 - F₁.2) * (Q.2 - F₂.2) = 0

/-- The eccentricity of hyperbola C is √5 - 1 -/
theorem hyperbola_C_eccentricity (hC : HyperbolaC) : 
  ∃ e : ℝ, e = Real.sqrt 5 - 1 ∧ e^2 = (hC.a^2 + hC.b^2) / hC.a^2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_C_eccentricity_l780_78065


namespace NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l780_78050

/-- The maximum area of a rectangle inscribed in a circular segment -/
theorem max_area_inscribed_rectangle (r : ℝ) (α : ℝ) (h : 0 < α ∧ α ≤ π / 2) :
  ∃ (T_max : ℝ), T_max = (r^2 / 8) * (-3 * Real.cos α + Real.sqrt (8 + Real.cos α ^ 2)) *
    Real.sqrt (8 - 2 * Real.cos α ^ 2 - 2 * Real.cos α * Real.sqrt (8 + Real.cos α ^ 2)) ∧
  ∀ (T : ℝ), T ≤ T_max := by
  sorry

end NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l780_78050


namespace NUMINAMATH_CALUDE_car_distance_traveled_l780_78073

/-- Given a train speed and a car's relative speed to the train, 
    calculate the distance traveled by the car in a given time. -/
theorem car_distance_traveled 
  (train_speed : ℝ) 
  (car_relative_speed : ℝ) 
  (time_minutes : ℝ) : 
  train_speed = 90 →
  car_relative_speed = 2/3 →
  time_minutes = 30 →
  (car_relative_speed * train_speed) * (time_minutes / 60) = 30 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_traveled_l780_78073


namespace NUMINAMATH_CALUDE_circumcenter_equidistant_closest_vertex_l780_78074

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem: The circumcenter is equidistant from all vertices of the triangle
theorem circumcenter_equidistant (t : Triangle) :
  distance (circumcenter t) t.A = distance (circumcenter t) t.B ∧
  distance (circumcenter t) t.B = distance (circumcenter t) t.C :=
sorry

-- Theorem: Any point in the plane is closest to one of the three vertices
theorem closest_vertex (t : Triangle) (p : ℝ × ℝ) :
  (distance p t.A ≤ distance p t.B ∧ distance p t.A ≤ distance p t.C) ∨
  (distance p t.B ≤ distance p t.A ∧ distance p t.B ≤ distance p t.C) ∨
  (distance p t.C ≤ distance p t.A ∧ distance p t.C ≤ distance p t.B) :=
sorry

end NUMINAMATH_CALUDE_circumcenter_equidistant_closest_vertex_l780_78074


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l780_78030

/-- Given an arithmetic sequence with common difference 2 where a₁, a₃, a₄ form a geometric sequence, a₆ = 2 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →         -- a₁, a₃, a₄ form a geometric sequence
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l780_78030


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l780_78002

theorem solution_set_quadratic_inequality :
  let S : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}
  S = {x | -1 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l780_78002


namespace NUMINAMATH_CALUDE_binomial_26_6_l780_78024

theorem binomial_26_6 (h1 : Nat.choose 25 5 = 53130) (h2 : Nat.choose 25 6 = 177100) :
  Nat.choose 26 6 = 230230 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_6_l780_78024


namespace NUMINAMATH_CALUDE_equation_solutions_l780_78032

theorem equation_solutions :
  (∃ x : ℚ, 2 * x - 1 = 5 * x + 2 ↔ x = -1) ∧
  (∃ x : ℚ, (5 * x + 1) / 2 - (2 * x - 1) / 4 = 1 ↔ x = 1 / 8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l780_78032


namespace NUMINAMATH_CALUDE_games_won_l780_78018

/-- Proves that the number of games won is 8, given the total games and lost games. -/
theorem games_won (total_games lost_games : ℕ) 
  (h1 : total_games = 12) 
  (h2 : lost_games = 4) : 
  total_games - lost_games = 8 := by
  sorry

#check games_won

end NUMINAMATH_CALUDE_games_won_l780_78018


namespace NUMINAMATH_CALUDE_machine_time_difference_l780_78000

/- Define the variables -/
variable (W : ℝ) -- Number of widgets
variable (X : ℝ) -- Rate of machine X in widgets per day
variable (Y : ℝ) -- Rate of machine Y in widgets per day

/- Define the conditions -/
axiom machine_X_rate : X = W / 6
axiom combined_rate : X + Y = 5 * W / 12
axiom machine_X_alone : 30 * X = 5 * W

/- State the theorem -/
theorem machine_time_difference : 
  W / X - W / Y = 2 := by sorry

end NUMINAMATH_CALUDE_machine_time_difference_l780_78000


namespace NUMINAMATH_CALUDE_problem_solution_l780_78061

theorem problem_solution (m n : ℝ) 
  (hm : m^2 - 2*m - 1 = 0) 
  (hn : n^2 + 2*n - 1 = 0) 
  (hmn : m*n ≠ 1) : 
  (m*n + n + 1) / n = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l780_78061


namespace NUMINAMATH_CALUDE_equation_solution_l780_78025

theorem equation_solution (x a b : ℝ) (h : b / x^3 + 1 / x^2 + 1 / x + 1 = 0) :
  x^3 + x^2 + x + a = a - b := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l780_78025


namespace NUMINAMATH_CALUDE_line_through_circle_center_l780_78006

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := 3*x + y + a = 0

-- Define the center of a circle
def is_center (h k : ℝ) : Prop := ∀ x y, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 5

-- Theorem statement
theorem line_through_circle_center (a : ℝ) : 
  (∃ h k, is_center h k ∧ line_equation h k a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l780_78006


namespace NUMINAMATH_CALUDE_farm_sale_earnings_l780_78027

/-- Calculates the total money earned from selling farm animals -/
def total_money_earned (num_cows : ℕ) (pig_cow_ratio : ℕ) (price_per_pig : ℕ) (price_per_cow : ℕ) : ℕ :=
  let num_pigs := num_cows * pig_cow_ratio
  let money_from_pigs := num_pigs * price_per_pig
  let money_from_cows := num_cows * price_per_cow
  money_from_pigs + money_from_cows

/-- Theorem stating that given the specific conditions, the total money earned is $48,000 -/
theorem farm_sale_earnings : total_money_earned 20 4 400 800 = 48000 := by
  sorry

end NUMINAMATH_CALUDE_farm_sale_earnings_l780_78027


namespace NUMINAMATH_CALUDE_tricycle_count_l780_78075

theorem tricycle_count (total_children : ℕ) (total_wheels : ℕ) (walking_children : ℕ) :
  total_children = 10 →
  total_wheels = 24 →
  walking_children = 2 →
  ∃ (bicycles tricycles : ℕ),
    bicycles + tricycles + walking_children = total_children ∧
    2 * bicycles + 3 * tricycles = total_wheels ∧
    tricycles = 8 :=
by sorry

end NUMINAMATH_CALUDE_tricycle_count_l780_78075


namespace NUMINAMATH_CALUDE_theta_range_l780_78097

theorem theta_range (θ : Real) : 
  θ ∈ Set.Icc 0 π ∧ 
  (∀ x ∈ Set.Icc (-1) 0, x^2 * Real.cos θ + (x+1)^2 * Real.sin θ + x^2 + x > 0) →
  θ ∈ Set.Ioo (π/12) (5*π/12) := by
sorry

end NUMINAMATH_CALUDE_theta_range_l780_78097


namespace NUMINAMATH_CALUDE_sum_reciprocal_squares_cubic_l780_78033

/-- Given a cubic equation x^3 - 12x^2 + 17x + 4 = 0 with real roots a, b, and c,
    prove that the sum of reciprocals of squares of roots equals 385/16 -/
theorem sum_reciprocal_squares_cubic (a b c : ℝ) : 
  a^3 - 12*a^2 + 17*a + 4 = 0 → 
  b^3 - 12*b^2 + 17*b + 4 = 0 → 
  c^3 - 12*c^2 + 17*c + 4 = 0 → 
  (1/a^2) + (1/b^2) + (1/c^2) = 385/16 :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squares_cubic_l780_78033


namespace NUMINAMATH_CALUDE_green_face_probability_l780_78005

/-- Probability of rolling a green face on an octahedron with 5 green faces out of 8 total faces -/
theorem green_face_probability (total_faces : ℕ) (green_faces : ℕ) 
  (h1 : total_faces = 8) 
  (h2 : green_faces = 5) : 
  (green_faces : ℚ) / total_faces = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_green_face_probability_l780_78005


namespace NUMINAMATH_CALUDE_xw_value_l780_78021

theorem xw_value (x w : ℝ) (h1 : 7 * x = 28) (h2 : x + w = 9) : x * w = 20 := by
  sorry

end NUMINAMATH_CALUDE_xw_value_l780_78021


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l780_78010

theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ+) : 
  Nat.lcm a b = 108 → 
  (a : ℚ) / (b : ℚ) = 3 / 7 → 
  (a : ℕ) + b = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l780_78010


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l780_78045

/-- A box containing pairs of shoes -/
structure ShoeBox where
  pairs : ℕ
  total : ℕ
  total_eq_twice_pairs : total = 2 * pairs

/-- The probability of selecting two matching shoes from a ShoeBox -/
def matchingProbability (box : ShoeBox) : ℚ :=
  1 / (box.total - 1)

theorem matching_shoes_probability (box : ShoeBox) 
  (h : box.pairs = 100) : 
  matchingProbability box = 1 / 199 := by
  sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l780_78045


namespace NUMINAMATH_CALUDE_lydia_apple_tree_age_l780_78003

theorem lydia_apple_tree_age (tree_fruit_time : ℕ) (planting_age : ℕ) : 
  tree_fruit_time = 10 → planting_age = 6 → planting_age + tree_fruit_time = 16 := by
  sorry

end NUMINAMATH_CALUDE_lydia_apple_tree_age_l780_78003


namespace NUMINAMATH_CALUDE_infiniteNestedSqrtEqualThree_l780_78034

/-- The value of the infinite expression sqrt(6 + sqrt(6 + sqrt(6 + ...))) -/
noncomputable def infiniteNestedSqrt : ℝ :=
  Real.sqrt (6 + Real.sqrt (6 + Real.sqrt (6 + Real.sqrt 6)))

/-- Theorem stating that the infinite nested square root equals 3 -/
theorem infiniteNestedSqrtEqualThree : infiniteNestedSqrt = 3 := by
  sorry

end NUMINAMATH_CALUDE_infiniteNestedSqrtEqualThree_l780_78034


namespace NUMINAMATH_CALUDE_min_value_theorem_l780_78094

theorem min_value_theorem (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 9) : 
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l780_78094


namespace NUMINAMATH_CALUDE_program_output_l780_78044

def program (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
  let a' := b
  let b' := c
  let c' := a'
  (a', b', c')

theorem program_output : program 10 20 30 = (20, 30, 20) := by
  sorry

end NUMINAMATH_CALUDE_program_output_l780_78044


namespace NUMINAMATH_CALUDE_largest_non_sum_is_correct_l780_78055

/-- The largest natural number not exceeding 50 that cannot be expressed as a sum of 5s and 6s -/
def largest_non_sum : ℕ := 19

/-- A predicate that checks if a natural number can be expressed as a sum of 5s and 6s -/
def is_sum_of_5_and_6 (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

theorem largest_non_sum_is_correct :
  (largest_non_sum ≤ 50) ∧
  ¬(is_sum_of_5_and_6 largest_non_sum) ∧
  ∀ (m : ℕ), m > largest_non_sum → m ≤ 50 → is_sum_of_5_and_6 m :=
by sorry

end NUMINAMATH_CALUDE_largest_non_sum_is_correct_l780_78055


namespace NUMINAMATH_CALUDE_building_stories_l780_78064

/-- Represents the number of stories in the building -/
def n : ℕ := sorry

/-- Time taken by Lola to climb one story -/
def lola_time_per_story : ℕ := 10

/-- Time taken by the elevator to go up one story -/
def elevator_time_per_story : ℕ := 8

/-- Time the elevator stops on each floor -/
def elevator_stop_time : ℕ := 3

/-- Total time taken by the slower person to reach the top -/
def total_time : ℕ := 220

/-- Time taken by Lola to reach the top -/
def lola_total_time : ℕ := n * lola_time_per_story

/-- Time taken by Tara (using the elevator) to reach the top -/
def tara_total_time : ℕ := n * elevator_time_per_story + (n - 1) * elevator_stop_time

theorem building_stories :
  (tara_total_time ≥ lola_total_time) ∧ (tara_total_time = total_time) → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_building_stories_l780_78064


namespace NUMINAMATH_CALUDE_certain_number_equation_l780_78082

theorem certain_number_equation : ∃ x : ℚ, 1038 * x = 173 * 240 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l780_78082


namespace NUMINAMATH_CALUDE_garden_flowers_l780_78029

/-- Represents a rectangular garden with a rose planted at a specific position -/
structure Garden where
  rows_front : Nat  -- Number of rows in front of the rose
  rows_back : Nat   -- Number of rows behind the rose
  cols_left : Nat   -- Number of columns to the left of the rose
  cols_right : Nat  -- Number of columns to the right of the rose

/-- Calculates the total number of flowers in the garden -/
def total_flowers (g : Garden) : Nat :=
  (g.rows_front + 1 + g.rows_back) * (g.cols_left + 1 + g.cols_right)

/-- Theorem stating the total number of flowers in the specified garden -/
theorem garden_flowers :
  let g : Garden := {
    rows_front := 6,
    rows_back := 15,
    cols_left := 8,
    cols_right := 12
  }
  total_flowers g = 462 := by
  sorry

#eval total_flowers { rows_front := 6, rows_back := 15, cols_left := 8, cols_right := 12 }

end NUMINAMATH_CALUDE_garden_flowers_l780_78029


namespace NUMINAMATH_CALUDE_expression_equality_l780_78008

theorem expression_equality : (2^1006 + 5^1007)^2 - (2^1006 - 5^1007)^2 = 40 * 10^1006 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l780_78008


namespace NUMINAMATH_CALUDE_pizza_bill_theorem_l780_78040

/-- The total bill amount for a group of people dividing equally -/
def total_bill (num_people : ℕ) (amount_per_person : ℕ) : ℕ :=
  num_people * amount_per_person

/-- Theorem: For a group of 5 people paying $8 each, the total bill is $40 -/
theorem pizza_bill_theorem :
  total_bill 5 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_pizza_bill_theorem_l780_78040


namespace NUMINAMATH_CALUDE_journal_pages_per_session_l780_78056

/-- Given the number of journal-writing sessions per week and the total number of pages written
    in a certain number of weeks, calculate the number of pages written per session. -/
def pages_per_session (sessions_per_week : ℕ) (total_pages : ℕ) (num_weeks : ℕ) : ℕ :=
  total_pages / (sessions_per_week * num_weeks)

/-- Theorem stating that under the given conditions, each student writes 4 pages per session. -/
theorem journal_pages_per_session :
  pages_per_session 3 72 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_journal_pages_per_session_l780_78056


namespace NUMINAMATH_CALUDE_set_properties_l780_78081

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {0, |x|}
def B : Set ℝ := {1, 0, -1}

-- State the theorem
theorem set_properties (x : ℝ) (h : A x ⊆ B) :
  (x = 1 ∨ x = -1) ∧
  (A x ∪ B = {-1, 0, 1}) ∧
  (B \ A x = {-1}) :=
by sorry

end NUMINAMATH_CALUDE_set_properties_l780_78081


namespace NUMINAMATH_CALUDE_power_of_six_tens_digit_one_l780_78007

theorem power_of_six_tens_digit_one : ∃ n : ℕ, (6^n) % 100 ≥ 10 ∧ (6^n) % 100 < 20 := by
  sorry

end NUMINAMATH_CALUDE_power_of_six_tens_digit_one_l780_78007


namespace NUMINAMATH_CALUDE_problem_solution_l780_78092

theorem problem_solution (a b : ℤ) 
  (h1 : 3015 * a + 3019 * b = 3023)
  (h2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l780_78092


namespace NUMINAMATH_CALUDE_bertha_family_no_daughters_bertha_family_no_daughters_is_32_l780_78060

/-- Represents the family structure of Bertha and her descendants --/
structure BerthaFamily where
  daughters : Nat
  granddaughters : Nat
  daughters_with_children : Nat

/-- The properties of Bertha's family --/
def bertha_family : BerthaFamily where
  daughters := 8
  granddaughters := 32
  daughters_with_children := 8

theorem bertha_family_no_daughters : Nat :=
  let total := bertha_family.daughters + bertha_family.granddaughters
  let with_daughters := bertha_family.daughters_with_children
  total - with_daughters
  
#check bertha_family_no_daughters

theorem bertha_family_no_daughters_is_32 :
  bertha_family_no_daughters = 32 := by
  sorry

#check bertha_family_no_daughters_is_32

end NUMINAMATH_CALUDE_bertha_family_no_daughters_bertha_family_no_daughters_is_32_l780_78060


namespace NUMINAMATH_CALUDE_cube_root_square_l780_78039

theorem cube_root_square (x : ℝ) : (x - 1)^(1/3) = 3 → (x - 1)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_square_l780_78039


namespace NUMINAMATH_CALUDE_probability_non_expired_bottle_l780_78019

theorem probability_non_expired_bottle (total_bottles : ℕ) (expired_bottles : ℕ) 
  (h1 : total_bottles = 5) (h2 : expired_bottles = 1) : 
  (total_bottles - expired_bottles : ℚ) / total_bottles = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_non_expired_bottle_l780_78019


namespace NUMINAMATH_CALUDE_factory_production_rate_solve_factory_production_rate_l780_78016

/-- Proves that the daily production rate in the first year was 3650 televisions,
    given a 10% reduction in the second year and a total production of 3285 televisions
    in the second year. -/
theorem factory_production_rate : ℝ → Prop :=
  fun daily_rate =>
    let reduction_factor : ℝ := 0.9
    let second_year_production : ℝ := 3285
    daily_rate * reduction_factor * 365 = second_year_production →
    daily_rate = 3650

/-- The actual theorem statement -/
theorem solve_factory_production_rate : 
  ∃ (rate : ℝ), factory_production_rate rate :=
sorry

end NUMINAMATH_CALUDE_factory_production_rate_solve_factory_production_rate_l780_78016


namespace NUMINAMATH_CALUDE_sidorov_cash_calculation_l780_78068

/-- The disposable cash of the Sidorov family as of June 1, 2018 -/
def sidorov_cash : ℝ := 724506.3

/-- The first component of the Sidorov family's cash -/
def cash_component1 : ℝ := 496941.3

/-- The second component of the Sidorov family's cash -/
def cash_component2 : ℝ := 227565.0

/-- Theorem stating that the sum of the two cash components equals the total disposable cash -/
theorem sidorov_cash_calculation : 
  cash_component1 + cash_component2 = sidorov_cash := by
  sorry

end NUMINAMATH_CALUDE_sidorov_cash_calculation_l780_78068


namespace NUMINAMATH_CALUDE_actual_height_of_boy_l780_78001

/-- Proves that the actual height of a boy in a class of 35 boys is 226 cm, given the conditions of the problem. -/
theorem actual_height_of_boy (n : ℕ) (initial_avg : ℝ) (wrong_height : ℝ) (actual_avg : ℝ)
  (h1 : n = 35)
  (h2 : initial_avg = 181)
  (h3 : wrong_height = 166)
  (h4 : actual_avg = 179) :
  ∃ (actual_height : ℝ), actual_height = 226 ∧
    n * actual_avg = n * initial_avg - wrong_height + actual_height :=
by sorry

end NUMINAMATH_CALUDE_actual_height_of_boy_l780_78001


namespace NUMINAMATH_CALUDE_add_10000_seconds_to_5_45_00_l780_78058

def seconds_to_time (seconds : ℕ) : ℕ × ℕ × ℕ :=
  let total_minutes := seconds / 60
  let remaining_seconds := seconds % 60
  let hours := total_minutes / 60
  let minutes := total_minutes % 60
  (hours, minutes, remaining_seconds)

def add_time (start : ℕ × ℕ × ℕ) (duration : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (start_h, start_m, start_s) := start
  let (duration_h, duration_m, duration_s) := duration
  let total_seconds := start_s + start_m * 60 + start_h * 3600 +
                       duration_s + duration_m * 60 + duration_h * 3600
  seconds_to_time total_seconds

theorem add_10000_seconds_to_5_45_00 :
  add_time (5, 45, 0) (seconds_to_time 10000) = (8, 31, 40) :=
sorry

end NUMINAMATH_CALUDE_add_10000_seconds_to_5_45_00_l780_78058


namespace NUMINAMATH_CALUDE_replaced_person_weight_is_65_l780_78079

/-- The weight of the replaced person when the average weight of 6 persons
    increases by 2.5 kg after replacing one person with a new 80 kg person -/
def replacedPersonWeight (initialCount : ℕ) (averageIncrease : ℝ) (newPersonWeight : ℝ) : ℝ :=
  newPersonWeight - (initialCount : ℝ) * averageIncrease

/-- Theorem stating that under the given conditions, the weight of the replaced person is 65 kg -/
theorem replaced_person_weight_is_65 :
  replacedPersonWeight 6 2.5 80 = 65 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_is_65_l780_78079


namespace NUMINAMATH_CALUDE_parabola_directrix_l780_78099

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x = -(1/4) * y^2

/-- The directrix equation -/
def directrix_equation (x : ℝ) : Prop := x = 1

/-- Theorem: The directrix of the parabola x = -1/4 * y^2 is x = 1 -/
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola_equation x y → 
  ∃ (d : ℝ), directrix_equation d ∧
  ∀ (p q : ℝ × ℝ), 
    parabola_equation p.1 p.2 →
    (p.1 - d)^2 = (p.1 - q.1)^2 + (p.2 - q.2)^2 →
    q.1 = -1 ∧ q.2 = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l780_78099


namespace NUMINAMATH_CALUDE_min_skittles_proof_l780_78048

def min_skittles : ℕ := 150

theorem min_skittles_proof :
  (∀ n : ℕ, n ≥ min_skittles ∧ n % 19 = 17 → n ≥ min_skittles) ∧
  min_skittles % 19 = 17 :=
sorry

end NUMINAMATH_CALUDE_min_skittles_proof_l780_78048


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l780_78070

/-- Given a line segment with one endpoint (6, -1) and midpoint (3, 7),
    the sum of the coordinates of the other endpoint is 15. -/
theorem endpoint_coordinate_sum : ∀ (x y : ℝ),
  (6 + x) / 2 = 3 →
  (-1 + y) / 2 = 7 →
  x + y = 15 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l780_78070


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l780_78036

theorem quadratic_complete_square (a b c : ℝ) (h : 4 * a^2 - 8 * a - 320 = 0) :
  ∃ s : ℝ, s = 81 ∧ ∃ k : ℝ, (a - k)^2 = s :=
sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l780_78036


namespace NUMINAMATH_CALUDE_third_layer_sugar_l780_78057

/-- The amount of sugar needed for each layer of the cake -/
def sugar_amount (layer : Nat) : ℕ :=
  match layer with
  | 1 => 2  -- First layer requires 2 cups of sugar
  | 2 => 2 * sugar_amount 1  -- Second layer is twice as big as the first
  | 3 => 3 * sugar_amount 2  -- Third layer is three times larger than the second
  | _ => 0  -- We only consider 3 layers in this problem

theorem third_layer_sugar : sugar_amount 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_third_layer_sugar_l780_78057


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l780_78038

theorem hyperbola_eccentricity (a : ℝ) : 
  a > 0 → 
  (∃ x y : ℝ, x^2/a^2 - y^2/3 = 1) → 
  (∃ c : ℝ, c/a = 2) → 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l780_78038


namespace NUMINAMATH_CALUDE_remainder_2357912_div_8_l780_78053

theorem remainder_2357912_div_8 : 2357912 % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2357912_div_8_l780_78053


namespace NUMINAMATH_CALUDE_punch_bowl_ratio_l780_78011

/-- Proves that the ratio of punch the cousin drank to the initial amount is 1:1 -/
theorem punch_bowl_ratio : 
  ∀ (initial_amount cousin_drink : ℚ),
  initial_amount > 0 →
  cousin_drink > 0 →
  initial_amount - cousin_drink + 4 - 2 + 12 = 16 →
  initial_amount + 14 = 16 →
  cousin_drink / initial_amount = 1 := by
sorry

end NUMINAMATH_CALUDE_punch_bowl_ratio_l780_78011


namespace NUMINAMATH_CALUDE_prob_qualified_volleyball_expected_net_profit_l780_78023

-- Define the supply percentages and qualification rates
def supply_A : ℝ := 0.4
def supply_B : ℝ := 0.3
def supply_C : ℝ := 0.3
def qual_rate_A : ℝ := 0.95
def qual_rate_B : ℝ := 0.92
def qual_rate_C : ℝ := 0.96

-- Define profit and loss for each factory
def profit_A : ℝ := 10
def loss_A : ℝ := 5
def profit_C : ℝ := 8
def loss_C : ℝ := 6

-- Theorem 1: Probability of purchasing a qualified volleyball
theorem prob_qualified_volleyball :
  supply_A * qual_rate_A + supply_B * qual_rate_B + supply_C * qual_rate_C = 0.944 :=
sorry

-- Theorem 2: Expected net profit from purchasing one volleyball from Factory A and one from Factory C
theorem expected_net_profit :
  qual_rate_A * qual_rate_C * (profit_A + profit_C) +
  qual_rate_A * (1 - qual_rate_C) * (profit_A - loss_C) +
  (1 - qual_rate_A) * qual_rate_C * (profit_C - loss_A) +
  (1 - qual_rate_A) * (1 - qual_rate_C) * (-loss_A - loss_C) = 16.69 :=
sorry

end NUMINAMATH_CALUDE_prob_qualified_volleyball_expected_net_profit_l780_78023


namespace NUMINAMATH_CALUDE_x_power_2048_minus_reciprocal_l780_78095

theorem x_power_2048_minus_reciprocal (x : ℂ) (h : x + 1/x = Complex.I * Real.sqrt 2) :
  x^2048 - 1/x^2048 = 14^512 - 1024 := by
  sorry

end NUMINAMATH_CALUDE_x_power_2048_minus_reciprocal_l780_78095


namespace NUMINAMATH_CALUDE_perimeter_triangle_pst_l780_78078

/-- Given a triangle PQR with points S on PQ, T on PR, and U on ST, 
    prove that the perimeter of triangle PST is 36 under specific conditions. -/
theorem perimeter_triangle_pst (P Q R S T U : ℝ × ℝ) : 
  dist P Q = 19 →
  dist Q R = 18 →
  dist P R = 17 →
  ∃ t₁ : ℝ, S = (1 - t₁) • P + t₁ • Q →
  ∃ t₂ : ℝ, T = (1 - t₂) • P + t₂ • R →
  ∃ t₃ : ℝ, U = (1 - t₃) • S + t₃ • T →
  dist Q S = dist S U →
  dist U T = dist T R →
  dist P S + dist S T + dist P T = 36 :=
sorry

end NUMINAMATH_CALUDE_perimeter_triangle_pst_l780_78078


namespace NUMINAMATH_CALUDE_quadratic_linear_system_solution_l780_78014

theorem quadratic_linear_system_solution : 
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    (x₁^2 - 6*x₁ + 8 = 0) ∧
    (x₂^2 - 6*x₂ + 8 = 0) ∧
    (2*x₁ - y₁ = 6) ∧
    (2*x₂ - y₂ = 6) ∧
    (y₁ = 2) ∧
    (y₂ = -2) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_linear_system_solution_l780_78014


namespace NUMINAMATH_CALUDE_expected_distinct_faces_formula_l780_78090

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of times the die is rolled -/
def numRolls : ℕ := 6

/-- The probability that a specific face does not appear in any of the rolls -/
def probNoAppearance : ℚ := (numFaces - 1 : ℚ) / numFaces ^ numRolls

/-- The expected number of distinct faces that appear when a die is rolled multiple times -/
def expectedDistinctFaces : ℚ := numFaces * (1 - probNoAppearance)

/-- Theorem: The expected number of distinct faces that appear when a die is rolled six times 
    is equal to (6^6 - 5^6) / 6^5 -/
theorem expected_distinct_faces_formula : 
  expectedDistinctFaces = (numFaces ^ numRolls - (numFaces - 1) ^ numRolls : ℚ) / numFaces ^ (numRolls - 1) := by
  sorry

end NUMINAMATH_CALUDE_expected_distinct_faces_formula_l780_78090


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l780_78009

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l780_78009


namespace NUMINAMATH_CALUDE_expand_polynomial_simplify_expression_l780_78028

-- Problem 1
theorem expand_polynomial (x : ℝ) : x * (x + 3) * (x + 5) = x^3 + 8*x^2 + 15*x := by
  sorry

-- Problem 2
theorem simplify_expression (x y : ℝ) : (5*x + 2*y) * (5*x - 2*y) - 5*x * (5*x - 3*y) = -4*y^2 + 15*x*y := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_simplify_expression_l780_78028


namespace NUMINAMATH_CALUDE_range_of_a_l780_78067

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 2 3, x^2 - a ≥ 0) → a ∈ Set.Iic 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l780_78067


namespace NUMINAMATH_CALUDE_ammonia_composition_l780_78066

/-- The mass percentage of Nitrogen in Ammonia -/
def nitrogen_percentage : ℝ := 77.78

/-- The mass percentage of Hydrogen in Ammonia -/
def hydrogen_percentage : ℝ := 100 - nitrogen_percentage

theorem ammonia_composition :
  hydrogen_percentage = 22.22 := by sorry

end NUMINAMATH_CALUDE_ammonia_composition_l780_78066


namespace NUMINAMATH_CALUDE_ice_cream_bill_l780_78069

/-- The cost of ice cream scoops for Pierre and his mom -/
theorem ice_cream_bill (cost_per_scoop : ℕ) (pierre_scoops : ℕ) (mom_scoops : ℕ) :
  cost_per_scoop = 2 → pierre_scoops = 3 → mom_scoops = 4 →
  cost_per_scoop * (pierre_scoops + mom_scoops) = 14 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_bill_l780_78069


namespace NUMINAMATH_CALUDE_drainpipe_time_l780_78088

/-- Given a tank and three pipes:
    - Pipe1 fills the tank in 5 hours
    - Pipe2 fills the tank in 4 hours
    - Drainpipe empties the tank in x hours
    - All three pipes together fill the tank in 2.5 hours
    Prove that x = 20 -/
theorem drainpipe_time (pipe1_time pipe2_time all_pipes_time : ℝ) 
  (h1 : pipe1_time = 5)
  (h2 : pipe2_time = 4)
  (h3 : all_pipes_time = 2.5)
  (drain_time : ℝ) :
  (1 / pipe1_time + 1 / pipe2_time - 1 / drain_time = 1 / all_pipes_time) → 
  drain_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_drainpipe_time_l780_78088


namespace NUMINAMATH_CALUDE_system_solution_l780_78083

theorem system_solution : 
  ∃ (x y : ℝ), 
    (6.751 * x + 3.249 * y = 26.751) ∧ 
    (3.249 * x + 6.751 * y = 23.249) ∧ 
    (x = 3) ∧ 
    (y = 2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l780_78083


namespace NUMINAMATH_CALUDE_original_number_is_five_l780_78080

theorem original_number_is_five : ∃ x : ℚ, ((x / 4) * 12) - 6 = 9 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_five_l780_78080


namespace NUMINAMATH_CALUDE_group_interval_calculation_l780_78059

/-- Given a group [a,b) in a frequency distribution histogram with frequency 0.3 and height 0.06, |a-b| = 5 -/
theorem group_interval_calculation (a b : ℝ) 
  (frequency : ℝ) (height : ℝ) 
  (h1 : frequency = 0.3) 
  (h2 : height = 0.06) : 
  |a - b| = 5 := by sorry

end NUMINAMATH_CALUDE_group_interval_calculation_l780_78059


namespace NUMINAMATH_CALUDE_shoe_probability_l780_78087

def total_pairs : ℕ := 15
def black_pairs : ℕ := 8
def red_pairs : ℕ := 4
def white_pairs : ℕ := 3

def total_shoes : ℕ := total_pairs * 2

def favorable_outcomes : ℕ := black_pairs * black_pairs + red_pairs * red_pairs + white_pairs * white_pairs

theorem shoe_probability : 
  (favorable_outcomes : ℚ) / (total_shoes.choose 2) = 89 / 435 := by
  sorry

end NUMINAMATH_CALUDE_shoe_probability_l780_78087


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l780_78031

def A : Set ℝ := {x | x^2 ≤ 1}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l780_78031


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l780_78051

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => (2*x + 1)^2 - 3*(2*x + 1)
  ∃ (x₁ x₂ : ℝ), x₁ = -1/2 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l780_78051


namespace NUMINAMATH_CALUDE_double_burgers_count_l780_78052

/-- Represents the purchase of hamburgers for the marching band. -/
structure HamburgerPurchase where
  total_cost : ℚ
  total_burgers : ℕ
  single_burger_price : ℚ
  double_burger_price : ℚ

/-- Calculates the number of double burgers purchased. -/
def number_of_double_burgers (purchase : HamburgerPurchase) : ℕ := 
  sorry

/-- Theorem stating that the number of double burgers purchased is 41. -/
theorem double_burgers_count (purchase : HamburgerPurchase) 
  (h1 : purchase.total_cost = 70.5)
  (h2 : purchase.total_burgers = 50)
  (h3 : purchase.single_burger_price = 1)
  (h4 : purchase.double_burger_price = 1.5) :
  number_of_double_burgers purchase = 41 := by
  sorry

end NUMINAMATH_CALUDE_double_burgers_count_l780_78052


namespace NUMINAMATH_CALUDE_solution_verification_l780_78093

/-- Proves that (3, 2020, 4) and (-1, 2018, -2) are solutions to the given system of equations -/
theorem solution_verification :
  (∃ (x y z : ℤ), 
    (x + y - 2018 = (y - 2019) * x) ∧
    (y + z - 2017 = (y - 2019) * z) ∧
    (x + z + 5 = x * z) ∧
    ((x = 3 ∧ y = 2020 ∧ z = 4) ∨ (x = -1 ∧ y = 2018 ∧ z = -2))) := by
  sorry

end NUMINAMATH_CALUDE_solution_verification_l780_78093


namespace NUMINAMATH_CALUDE_valid_combinations_count_l780_78022

def digits : List Nat := [1, 1, 2, 2, 3, 3, 3, 3]

def is_valid_price (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 9999

def count_valid_combinations (digits : List Nat) : Nat :=
  sorry

theorem valid_combinations_count :
  count_valid_combinations digits = 14700 := by sorry

end NUMINAMATH_CALUDE_valid_combinations_count_l780_78022


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l780_78091

theorem sqrt_meaningful_range (a : ℝ) : 
  (∃ x : ℝ, x^2 = 2 - a) ↔ a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l780_78091


namespace NUMINAMATH_CALUDE_cos_15_degrees_l780_78015

theorem cos_15_degrees : Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_15_degrees_l780_78015


namespace NUMINAMATH_CALUDE_arrangements_theorem_l780_78037

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The total number of students in the group -/
def total_students : ℕ := num_boys + num_girls

/-- Function to calculate the number of arrangements where girls are not next to each other -/
def arrangements_girls_not_adjacent : ℕ := sorry

/-- Function to calculate the number of arrangements with girl A not at left end and girl B not at right end -/
def arrangements_girl_A_B_restricted : ℕ := sorry

/-- Function to calculate the number of arrangements where all boys stand next to each other -/
def arrangements_boys_together : ℕ := sorry

/-- Function to calculate the number of arrangements where A, B, C stand in height order -/
def arrangements_ABC_height_order : ℕ := sorry

theorem arrangements_theorem :
  arrangements_girls_not_adjacent = 480 ∧
  arrangements_girl_A_B_restricted = 504 ∧
  arrangements_boys_together = 144 ∧
  arrangements_ABC_height_order = 120 := by sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l780_78037


namespace NUMINAMATH_CALUDE_number_of_divisors_of_n_l780_78072

def n : ℕ := 293601000

theorem number_of_divisors_of_n : Nat.card {d : ℕ | d ∣ n} = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_n_l780_78072


namespace NUMINAMATH_CALUDE_line_equation_l780_78076

/-- Proves that the equation 4x + 3y - 13 = 0 represents the line passing through (1, 3)
    with a slope that is 1/3 of the slope of y = -4x -/
theorem line_equation (x y : ℝ) : 
  (∃ (k : ℝ), k = (-4 : ℝ) / 3 ∧ 
   y - 3 = k * (x - 1) ∧
   (∀ (x' y' : ℝ), y' = -4 * x' → k = (1 : ℝ) / 3 * (-4))) → 
  (4 * x + 3 * y - 13 = 0) := by
sorry

end NUMINAMATH_CALUDE_line_equation_l780_78076


namespace NUMINAMATH_CALUDE_coefficient_x2y1_is_60_l780_78085

/-- The coefficient of x^m y^n in the expansion of (1+x)^6(1+y)^4 -/
def f (m n : ℕ) : ℕ := Nat.choose 6 m * Nat.choose 4 n

/-- The theorem stating that the coefficient of x^2y^1 in the expansion of (1+x)^6(1+y)^4 is 60 -/
theorem coefficient_x2y1_is_60 : f 2 1 = 60 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y1_is_60_l780_78085


namespace NUMINAMATH_CALUDE_percentage_problem_l780_78046

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 40 + (8 / 100) * 24 = 5.92 ↔ P = 10 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l780_78046


namespace NUMINAMATH_CALUDE_function_value_2010_l780_78035

theorem function_value_2010 (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 1) = -f x) 
  (h2 : f 1 = 4) : 
  f 2010 = -4 := by sorry

end NUMINAMATH_CALUDE_function_value_2010_l780_78035


namespace NUMINAMATH_CALUDE_prob_select_copresidents_from_random_club_l780_78013

/-- Represents a math club with a given number of students and two co-presidents -/
structure MathClub where
  students : Nat
  has_two_copresidents : Bool

/-- Calculates the probability of selecting two co-presidents when choosing three members from a club -/
def prob_select_copresidents (club : MathClub) : Rat :=
  if club.has_two_copresidents then
    (Nat.choose (club.students - 2) 1 : Rat) / (Nat.choose club.students 3 : Rat)
  else
    0

/-- The list of math clubs in the school district -/
def math_clubs : List MathClub := [
  { students := 5, has_two_copresidents := true },
  { students := 7, has_two_copresidents := true },
  { students := 8, has_two_copresidents := true }
]

/-- Theorem stating the probability of selecting two co-presidents when randomly choosing
    three members from a randomly selected club among the given math clubs -/
theorem prob_select_copresidents_from_random_club : 
  (1 / (math_clubs.length : Rat)) * (math_clubs.map prob_select_copresidents).sum = 11 / 60 := by
  sorry

end NUMINAMATH_CALUDE_prob_select_copresidents_from_random_club_l780_78013


namespace NUMINAMATH_CALUDE_students_studying_all_subjects_l780_78017

theorem students_studying_all_subjects (total : ℕ) (math : ℕ) (latin : ℕ) (chem : ℕ) 
  (multi : ℕ) (none : ℕ) (h1 : total = 425) (h2 : math = 351) (h3 : latin = 71) 
  (h4 : chem = 203) (h5 : multi = 199) (h6 : none = 8) : 
  ∃ x : ℕ, x = 9 ∧ 
  total - none = math + latin + chem - multi + x := by
  sorry

end NUMINAMATH_CALUDE_students_studying_all_subjects_l780_78017


namespace NUMINAMATH_CALUDE_circles_intersection_product_of_coordinates_l780_78012

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 25 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 10*y + 37 = 0

-- Theorem stating that (2, 5) is the intersection point of the two circles
theorem circles_intersection :
  ∃! (x y : ℝ), circle1 x y ∧ circle2 x y ∧ x = 2 ∧ y = 5 := by
  sorry

-- Theorem stating that the product of the coordinates of the intersection point is 10
theorem product_of_coordinates :
  ∀ (x y : ℝ), circle1 x y ∧ circle2 x y → x * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersection_product_of_coordinates_l780_78012
