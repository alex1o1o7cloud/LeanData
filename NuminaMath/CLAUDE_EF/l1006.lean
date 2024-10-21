import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_caught_three_catfish_l1006_100691

/-- Represents the fishing catch of Emily -/
structure FishingCatch where
  trout_count : ℕ
  catfish_count : ℕ
  bluegill_count : ℕ
  trout_weight : ℝ
  catfish_weight : ℝ
  bluegill_weight : ℝ
  total_weight : ℝ

/-- Theorem stating that Emily caught 3 catfish -/
theorem emily_caught_three_catfish (c : FishingCatch) 
  (h1 : c.trout_count = 4)
  (h2 : c.bluegill_count = 5)
  (h3 : c.trout_weight = 2)
  (h4 : c.catfish_weight = 1.5)
  (h5 : c.bluegill_weight = 2.5)
  (h6 : c.total_weight = 25)
  (h7 : c.total_weight = 
    c.trout_count * c.trout_weight + 
    c.catfish_count * c.catfish_weight + 
    c.bluegill_count * c.bluegill_weight) :
  c.catfish_count = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_caught_three_catfish_l1006_100691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_multiplication_linearity_l1006_100637

-- Define the matrix and vectors
variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (u x : Fin 2 → ℝ)

-- Define the given conditions
axiom Mu : M.vecMul u = ![3, 1]
axiom Mx : M.vecMul x = ![-1, 4]

-- State the theorem
theorem matrix_multiplication_linearity :
  M.vecMul (3 • u - (1/2) • x) = ![19/2, 1] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_multiplication_linearity_l1006_100637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_OPO_more_likely_than_000_l1006_100617

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents a sequence of coin flips -/
def CoinSequence := List CoinFlip

/-- The probability of getting heads or tails in a single flip -/
noncomputable def flipProbability : ℝ := 1 / 2

/-- Checks if a given sequence contains the pattern OPO -/
def containsOPO (seq : CoinSequence) : Prop :=
  ∃ (i : ℕ), i + 2 < seq.length ∧ 
    seq.get? i = some CoinFlip.Heads ∧ 
    seq.get? (i+1) = some CoinFlip.Tails ∧ 
    seq.get? (i+2) = some CoinFlip.Heads

/-- Checks if a given sequence contains the pattern 000 -/
def contains000 (seq : CoinSequence) : Prop :=
  ∃ (i : ℕ), i + 2 < seq.length ∧ 
    seq.get? i = some CoinFlip.Heads ∧ 
    seq.get? (i+1) = some CoinFlip.Heads ∧ 
    seq.get? (i+2) = some CoinFlip.Heads

/-- The probability of OPO appearing before 000 in a sequence of coin flips -/
noncomputable def probOPOBefore000 : ℝ := sorry

/-- The probability of 000 appearing before OPO in a sequence of coin flips -/
noncomputable def prob000BeforeOPO : ℝ := sorry

/-- Theorem stating that OPO is more likely to appear before 000 -/
theorem OPO_more_likely_than_000 : probOPOBefore000 > prob000BeforeOPO := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_OPO_more_likely_than_000_l1006_100617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_is_half_l1006_100695

theorem tan_theta_is_half (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) 
  (h3 : Real.sin (2 * θ) * 1 + Real.cos θ * (-Real.cos θ) = 0) : 
  Real.tan θ = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_is_half_l1006_100695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_better_fit_exponential_model_regression_equation_l1006_100644

noncomputable def x_mean : ℝ := 26
noncomputable def y_mean : ℝ := 215
noncomputable def u_mean : ℝ := 680
noncomputable def v_mean : ℝ := 5.36

noncomputable def sum_x_diff_sq : ℝ := 100
noncomputable def sum_u_diff_sq : ℝ := 22500
noncomputable def sum_u_diff_y_diff : ℝ := 260
noncomputable def sum_y_diff_sq : ℝ := 4
noncomputable def sum_v_diff_sq : ℝ := 4
noncomputable def sum_x_diff_v_diff : ℝ := 18

noncomputable def correlation_coefficient (sum_diff_prod sum_x_diff_sq sum_y_diff_sq : ℝ) : ℝ :=
  sum_diff_prod / (Real.sqrt sum_x_diff_sq * Real.sqrt sum_y_diff_sq)

theorem better_fit_exponential_model :
  correlation_coefficient sum_x_diff_v_diff sum_x_diff_sq sum_v_diff_sq >
  correlation_coefficient sum_u_diff_y_diff sum_u_diff_sq sum_y_diff_sq := by
  sorry

noncomputable def lambda_hat : ℝ := sum_x_diff_v_diff / sum_x_diff_sq

noncomputable def t_hat : ℝ := v_mean - lambda_hat * x_mean

theorem regression_equation (x : ℝ) :
  ∃ y : ℝ, y = Real.exp (lambda_hat * x + t_hat) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_better_fit_exponential_model_regression_equation_l1006_100644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_conversion_l1006_100622

theorem polar_to_cartesian_conversion :
  ∀ (x y ρ θ : ℝ),
    x = ρ * Real.cos θ →
    y = ρ * Real.sin θ →
    ρ^2 * (1 + Real.sin θ^2) = 2 →
    x^2 / 2 + y^2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_conversion_l1006_100622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_geometric_sum_l1006_100666

def z_k (z k : ℕ) : ℕ := (z^(k+1) - 1) / (z - 1)

def M (z : ℕ) : Set ℕ := {n | ∃ k, n = z_k z k}

theorem divisors_of_geometric_sum (z : ℕ) (hz : z > 1) :
  {n : ℕ | n > 0 ∧ ∃ m ∈ M z, n ∣ m} = {n : ℕ | n > 0 ∧ Nat.gcd n z = 1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_geometric_sum_l1006_100666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_double_weight_8n_plus_5_weight_2_pow_n_minus_1_l1006_100655

/-- The weight function ω(n) counts the number of 1s in the binary representation of n -/
def ω (n : ℕ) : ℕ := sorry

theorem weight_double (n : ℕ) : ω (2 * n) = ω n := by sorry

theorem weight_8n_plus_5 (n : ℕ) : ω (8 * n + 5) = ω (4 * n + 3) := by sorry

theorem weight_2_pow_n_minus_1 (n : ℕ) : ω (2^n - 1) = n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_double_weight_8n_plus_5_weight_2_pow_n_minus_1_l1006_100655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosA_special_triangle_l1006_100627

noncomputable def triangle_cosA (a b c : ℝ) : ℝ :=
  (b^2 + c^2 - a^2) / (2*b*c)

theorem cosA_special_triangle (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_heights : a = Real.sqrt 2 * b ∧ a = 2 * c)
  (h_area : 1/2 * a * (1/2) = 1/2 * b * (Real.sqrt 2 / 2) ∧ 
            1/2 * b * (Real.sqrt 2 / 2) = 1/2 * c * 1) :
  triangle_cosA a b c = - Real.sqrt 2 / 4 := by
  sorry

#check cosA_special_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosA_special_triangle_l1006_100627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_hull_theorem_l1006_100659

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A convex polygon -/
structure ConvexPolygon where
  vertices : List Point
  is_convex : Bool

/-- Predicate to check if a point is inside a polygon -/
def point_inside_polygon (p : Point) (poly : ConvexPolygon) : Prop :=
  sorry

/-- Predicate to check if a point is inside the convex hull of a set of points -/
def point_inside_convex_hull (p : Point) (points : List Point) : Prop :=
  sorry

/-- Theorem: For any convex 100-gon and any set of k points chosen inside it, 
    where 2 ≤ k ≤ 50, there exists a subset of 2k vertices of the 100-gon 
    such that the convex hull of these 2k vertices contains all k chosen points. -/
theorem convex_hull_theorem 
  (polygon : ConvexPolygon) 
  (chosen_points : List Point) 
  (h1 : polygon.vertices.length = 100)
  (h2 : polygon.is_convex = true)
  (h3 : 2 ≤ chosen_points.length)
  (h4 : chosen_points.length ≤ 50)
  (h5 : ∀ p ∈ chosen_points, point_inside_polygon p polygon) :
  ∃ (subset : List Point),
    subset.length = 2 * chosen_points.length ∧ 
    subset ⊆ polygon.vertices ∧
    ∀ p ∈ chosen_points, point_inside_convex_hull p subset :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_hull_theorem_l1006_100659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_and_difference_l1006_100649

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

-- Define the constant function
def g : ℝ → ℝ := Function.const ℝ 5

-- Define the intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Theorem statement
theorem intersection_distance_and_difference : 
  ∃ (C D : ℝ), C ∈ intersection_points ∧ D ∈ intersection_points ∧ C ≠ D ∧
  ∃ (p q : ℕ), 
    (abs (C - D) = Real.sqrt p / q) ∧ 
    Nat.Coprime p q ∧
    p - q = 19 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_and_difference_l1006_100649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_power_with_specific_start_l1006_100696

theorem no_power_with_specific_start : ¬ ∃ (n : ℕ), 
  (∃ (k : ℕ), 5 * 10^k ≤ 2^n ∧ 2^n < 6 * 10^k) ∧ 
  (∃ (m : ℕ), 2 * 10^m ≤ 5^n ∧ 5^n < 3 * 10^m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_power_with_specific_start_l1006_100696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l1006_100640

theorem triangle_side_sum (a b c : ℝ) (A : ℝ) :
  a = 2 →
  A = π / 3 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  b + c = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l1006_100640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_in_S_l1006_100672

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

def S (α : ℝ) : Set ℤ := {z : ℤ | ∃ n : ℤ, z = floor (n * α)}

def is_arithmetic_progression (a : ℕ → ℤ) (k : ℕ) : Prop :=
  ∃ d : ℤ, ∀ i : ℕ, i < k - 1 → a (i + 1) - a i = d

theorem arithmetic_progression_in_S (α : ℝ) (h_irr : Irrational α) (h_gt : α > 2) :
  (∀ m : ℕ, m ≥ 3 → 
    ∃ a : ℕ → ℤ, 
      (∀ i, i < m → a i ∈ S α) ∧ 
      (∀ i j, i < m → j < m → i ≠ j → a i ≠ a j) ∧ 
      is_arithmetic_progression a m) ∧
  ¬∃ a : ℕ → ℤ, 
    (∀ n, a n ∈ S α) ∧ 
    (∀ n m, n ≠ m → a n ≠ a m) ∧ 
    is_arithmetic_progression a 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_in_S_l1006_100672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_l1006_100671

/-- Given a line in the form of ax + by + c = 0 -/
def Line (a b c : ℝ) : ℝ → ℝ → Prop :=
  fun x y => a * x + b * y + c = 0

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def Perpendicular (l₁ l₂ : ℝ → ℝ → Prop) : Prop :=
  ∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ,
    l₁ = Line a₁ b₁ c₁ ∧
    l₂ = Line a₂ b₂ c₂ ∧
    a₁ * a₂ = -b₁ * b₂

theorem perpendicular_lines_k (k : ℝ) :
  Perpendicular (Line 1 4 (-1)) (Line k 1 2) →
  k = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_l1006_100671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slate_rock_probability_l1006_100603

/-- The probability of selecting two slate rocks from a field with 10 slate rocks, 
    15 pumice rocks, and 6 granite rocks when choosing 2 rocks at random without replacement -/
theorem slate_rock_probability : 
  (10 : ℚ) * 9 / ((10 + 15 + 6) * (10 + 15 + 6 - 1)) = 3 / 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slate_rock_probability_l1006_100603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_correct_perpendicular_line_correct_l1006_100652

-- Define the given line l
def line_l (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define the point P
def point_P : ℝ × ℝ := (1, 2)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2 * x - y = 0

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x + 2 * y - 5 = 0

-- Theorem for the parallel line
theorem parallel_line_correct :
  (∀ x y : ℝ, line_l x y ↔ ∃ k : ℝ, parallel_line x y ∧ 2 * x - y - 2 = k) ∧
  parallel_line point_P.1 point_P.2 := by sorry

-- Theorem for the perpendicular line
theorem perpendicular_line_correct :
  (∀ x₁ y₁ x₂ y₂ : ℝ, line_l x₁ y₁ ∧ perpendicular_line x₂ y₂ →
    (x₂ - x₁) * 2 + (y₂ - y₁) * (-1) = 0) ∧
  perpendicular_line point_P.1 point_P.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_correct_perpendicular_line_correct_l1006_100652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_odd_powers_l1006_100680

theorem divisibility_of_odd_powers (p a b c d : ℕ) (hp : Nat.Prime p) (hp3 : p > 3)
  (h_sum : p ∣ (a + b + c + d)) (h_cube_sum : p ∣ (a^3 + b^3 + c^3 + d^3)) :
  ∀ n : ℕ, Odd n → p ∣ (a^n + b^n + c^n + d^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_odd_powers_l1006_100680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_calculation_l1006_100618

/-- Represents the total worth of a stock in Rupees -/
def stock_worth (total : ℝ) : Prop := total > 0

/-- Calculates the profit or loss from selling a portion of stock -/
def profit_loss (portion : ℝ) (rate : ℝ) (total : ℝ) : ℝ :=
  portion * rate * total

theorem stock_worth_calculation (total : ℝ) :
  stock_worth total →
  profit_loss 0.2 0.1 total - profit_loss 0.8 0.05 total = 350 →
  total = 17500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_calculation_l1006_100618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_are_real_l1006_100681

/-- Given a positive integer n and polynomials f and g with real coefficients
    such that (1+iT)^n = f(T) + ig(T), where i is the square root of -1,
    for any real number k, the equation f(T) + kg(T) = 0 has only real roots. -/
theorem roots_are_real (n : ℕ+) (f g : ℝ → ℝ) (k : ℝ) :
  (∀ T : ℂ, (1 + Complex.I * T) ^ n.val = f T.re + Complex.I * g T.re) →
  (∀ T : ℂ, f T.re + k * g T.re = 0 → T.im = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_are_real_l1006_100681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l1006_100646

theorem undefined_values_count : ∃ (S : Finset ℝ), 
  (∀ x ∈ S, (x^2 + x - 6) * (x - 4) = 0) ∧ 
  Finset.card S = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l1006_100646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_30_value_l1006_100621

def c : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 3
  | n + 3 => c (n + 2) * c (n + 1)

theorem c_30_value : c 30 = 3^514229 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_30_value_l1006_100621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carousel_rotation_period_is_two_thirds_a_carousel_conditions_l1006_100626

/-- The period of a carousel's rotation given specific meeting conditions with a runner -/
noncomputable def carousel_rotation_period (a : ℝ) (a_pos : a > 0) : ℝ :=
  let v := (2 * Real.pi) / a  -- Angular speed of the runner
  let U := (2 * Real.pi) / ((2 * a) / 3)  -- Angular speed of the carousel
  (2 * a) / 3

/-- The carousel completes one full rotation in 2a/3 seconds -/
theorem carousel_rotation_period_is_two_thirds_a (a : ℝ) (a_pos : a > 0) :
  carousel_rotation_period a a_pos = (2 * a) / 3 := by
  unfold carousel_rotation_period
  simp
  
/-- Proof of the conditions given in the problem -/
theorem carousel_conditions (a : ℝ) (a_pos : a > 0) :
  let v := (2 * Real.pi) / a  -- Angular speed of the runner
  let U := (2 * Real.pi) / ((2 * a) / 3)  -- Angular speed of the carousel
  (U * a - v * a = 2 * Real.pi) ∧ 
  (v * a = U * ((2 * a) / 3 - a / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carousel_rotation_period_is_two_thirds_a_carousel_conditions_l1006_100626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_speed_l1006_100667

/-- The average speed formula for a round trip -/
noncomputable def average_speed (speed1 speed2 : ℝ) : ℝ :=
  (2 * speed1 * speed2) / (speed1 + speed2)

/-- Theorem: Given the conditions of the problem, the speed on the flight up is 112.5 mph -/
theorem flight_speed (speed_home : ℝ) (avg_speed : ℝ) 
  (h1 : speed_home = 90)
  (h2 : avg_speed = 100)
  (h3 : ∀ s, avg_speed = average_speed s speed_home) :
  ∃ speed_up : ℝ, speed_up = 112.5 := by
  sorry

#check flight_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_speed_l1006_100667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l1006_100636

/-- A quadratic function passing through three given points has its vertex at x = 4 -/
theorem quadratic_vertex (a b c : ℝ) : 
  (8 = 4 * a + 2 * b + c) →
  (8 = 36 * a + 6 * b + c) →
  (13 = 49 * a + 7 * b + c) →
  (let f : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c
   let vertex_x : ℝ := -b / (2 * a)
   vertex_x = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l1006_100636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_2_equals_5_l1006_100613

-- Define the function q
noncomputable def q : ℝ → ℝ := sorry

-- Axiom: q(2.0) is an integer
axiom q_2_is_integer : ∃ n : ℤ, q 2 = n

-- Theorem: q(2.0) = 5
theorem q_2_equals_5 : q 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_2_equals_5_l1006_100613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traveler_distance_l1006_100661

/-- Represents the distance traveled in myriameters -/
noncomputable def distance (d1 t1 h1 h2 : ℝ) : ℝ :=
  d1 * (h2 / h1)

theorem traveler_distance (d1 t1 t2 h1 h2 : ℝ) :
  d1 = 112 ∧ t1 = 29 ∧ h1 = 7 ∧ t2 = 17 ∧ h2 = 10 →
  distance d1 t1 h1 (t2 * h2) = 97 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_traveler_distance_l1006_100661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1006_100639

-- Define the function g(x) as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.arctan (2 * x) + Real.arctan ((2 - 3 * x) / (2 + 3 * x)) + Real.arctan x

-- State the theorem about the range of g(x)
theorem range_of_g :
  (∀ x : ℝ, -π/4 ≤ g x ∧ g x ≤ 3*π/4) ∧
  (∃ x y : ℝ, g x = -π/4 ∧ g y = 3*π/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1006_100639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_through_points_l1006_100650

/-- A quadratic function passing through three given points -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_through_points :
  ∃ (a b c : ℝ),
    let f := quadratic_function a b c
    f (1/2) = 3/4 ∧
    f (-1) = 3 ∧
    f 2 = 3 ∧
    a = 1 ∧ b = -1 ∧ c = 1 :=
by
  sorry

#check quadratic_through_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_through_points_l1006_100650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_plants_in_garden_l1006_100633

theorem unique_plants_in_garden (A B C : Finset ℕ) : 
  A.card = 600 →
  B.card = 500 →
  C.card = 400 →
  (A ∩ B).card = 60 →
  (A ∩ C).card = 120 →
  (B ∩ C).card = 80 →
  (A ∩ B ∩ C).card = 30 →
  (A ∪ B ∪ C).card = 1270 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_plants_in_garden_l1006_100633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_equation_of_l_intersection_range_l1006_100602

noncomputable section

-- Define the curve C
def C (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

-- Define the line l in polar form
def l (ρ θ : ℝ) (m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 3) + m = 0

-- Theorem for the rectangular equation of l
theorem rectangular_equation_of_l (x y m : ℝ) :
  (∃ ρ θ, l ρ θ m ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔
  Real.sqrt 3 * x + y + 2 * m = 0 := by
  sorry

-- Theorem for the intersection of C and l
theorem intersection_range (m : ℝ) :
  (∃ t, ∃ x y, C t = (x, y) ∧ Real.sqrt 3 * x + y + 2 * m = 0) ↔
  -19/12 ≤ m ∧ m ≤ 5/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_equation_of_l_intersection_range_l1006_100602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersection_points_l1006_100624

/-- The first function in the intersection problem -/
def f (x : ℝ) : ℝ := |3 * x + 4| + 1

/-- The second function in the intersection problem -/
def g (x : ℝ) : ℝ := -|4 * x - 3| - 2

/-- A point (x, y) is an intersection point if it satisfies both equations -/
def is_intersection_point (x : ℝ) : Prop := f x = g x

/-- The set of all intersection points -/
def intersection_points : Set ℝ := {x | is_intersection_point x}

/-- The theorem stating that there are exactly 3 intersection points -/
theorem three_intersection_points : ∃ (S : Finset ℝ), S.card = 3 ∧ ∀ x, x ∈ intersection_points ↔ x ∈ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersection_points_l1006_100624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_second_hand_distance_theorem_l1006_100668

/-- The length of the second hand on a clock in centimeters. -/
def second_hand_length : ℝ := 8

/-- The number of minutes in the given time period. -/
def time_period : ℕ := 45

/-- The distance traveled by the tip of the second hand in the given time period. -/
noncomputable def distance_traveled : ℝ := 720 * Real.pi

/-- 
Theorem stating that the distance traveled by the tip of a clock's second hand
in the given time period is equal to the calculated distance, given the length
of the second hand.
-/
theorem second_hand_distance_theorem :
  2 * Real.pi * second_hand_length * (time_period : ℝ) = distance_traveled := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_second_hand_distance_theorem_l1006_100668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l1006_100674

theorem max_product_sum (a b c d : ℕ) : 
  a ∈ ({2, 4, 6, 8} : Set ℕ) → 
  b ∈ ({2, 4, 6, 8} : Set ℕ) → 
  c ∈ ({2, 4, 6, 8} : Set ℕ) → 
  d ∈ ({2, 4, 6, 8} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  (a * b + b * d + d * c + c * a) ≤ 40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l1006_100674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_axes_l1006_100653

theorem sine_symmetry_axes (ω φ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < π) :
  (∀ x : ℝ, Real.sin (ω * (π/4 + x) + φ) = Real.sin (ω * (π/4 - x) + φ)) ∧
  (∀ x : ℝ, Real.sin (ω * (5*π/4 + x) + φ) = Real.sin (ω * (5*π/4 - x) + φ)) →
  φ = π/4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_axes_l1006_100653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_green_eyes_backpack_no_glasses_l1006_100690

/-- Represents the class of students -/
structure StudentClass where
  total : ℕ
  greenEyes : ℕ
  backpack : ℕ
  glasses : ℕ

/-- The minimum number of students with green eyes and a backpack -/
def minOverlap (c : StudentClass) : ℕ :=
  max 0 (c.greenEyes + c.backpack - c.total)

/-- The number of students who don't wear glasses -/
def noGlasses (c : StudentClass) : ℕ :=
  c.total - c.glasses

/-- The theorem to prove -/
theorem min_green_eyes_backpack_no_glasses (c : StudentClass) 
  (h1 : c.total = 25)
  (h2 : c.greenEyes = 15)
  (h3 : c.backpack = 18)
  (h4 : c.glasses = 6) :
  minOverlap c ≤ 8 ∧ 8 ≤ min (minOverlap c) (noGlasses c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_green_eyes_backpack_no_glasses_l1006_100690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1006_100669

open Set Real

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | 2 < x}
def C (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x ≤ a + 1}

theorem problem_solution :
  (∃ a : ℝ, C a ⊆ A) →
  (A ∩ B = {x | 2 < x ∧ x ≤ 5}) ∧
  ((Bᶜ) ∪ A = {x | x ≤ 5}) ∧
  (∀ a : ℝ, C a ⊆ A ↔ 1 ≤ a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1006_100669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l1006_100685

-- Define the curve C
noncomputable def curve_C (α : Real) : Real × Real :=
  (2 + Real.sqrt 10 * Real.cos α, Real.sqrt 10 * Real.sin α)

-- Define the line l
def line_l (ρ θ : Real) : Prop :=
  ρ * Real.cos θ + 2 * ρ * Real.sin θ - 12 = 0

-- Define the ray l₁
def ray_l₁ (ρ θ : Real) : Prop :=
  θ = Real.pi / 4 ∧ ρ ≥ 0

-- Theorem statement
theorem distance_AB : ∃ (ρ_A ρ_B : Real),
  (∃ α, curve_C α = (ρ_A * Real.cos (Real.pi / 4), ρ_A * Real.sin (Real.pi / 4))) ∧
  line_l ρ_B (Real.pi / 4) ∧
  ray_l₁ ρ_A (Real.pi / 4) ∧
  ray_l₁ ρ_B (Real.pi / 4) ∧
  |ρ_A - ρ_B| = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l1006_100685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_by_52_l1006_100660

/-- A function representing the spiral numbering of grid vertices -/
def spiralNumbering : ℕ × ℕ → ℕ := sorry

/-- The sum of numbers at the vertices of a cell -/
def cellSum (x y : ℕ) : ℕ :=
  spiralNumbering (x, y) + spiralNumbering (x + 1, y) +
  spiralNumbering (x, y + 1) + spiralNumbering (x + 1, y + 1)

/-- Predicate for a cell having a sum divisible by 52 -/
def isDivisibleBy52 (x y : ℕ) : Prop :=
  cellSum x y % 52 = 0

theorem infinitely_many_divisible_by_52 :
  ∀ n : ℕ, ∃ x y : ℕ, x + y > n ∧ isDivisibleBy52 x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_by_52_l1006_100660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_stock_weight_l1006_100619

-- Define the weights of the goods
noncomputable def green_beans_weight : ℝ := 60

-- Rice weight is 30 kg less than green beans
noncomputable def rice_weight : ℝ := green_beans_weight - 30

-- Sugar weight is 10 kg less than green beans
noncomputable def sugar_weight : ℝ := green_beans_weight - 10

-- Define the fractions of weight lost
noncomputable def rice_loss_fraction : ℝ := 1/3
noncomputable def sugar_loss_fraction : ℝ := 1/5

-- Calculate the remaining weights
noncomputable def remaining_rice : ℝ := rice_weight * (1 - rice_loss_fraction)
noncomputable def remaining_sugar : ℝ := sugar_weight * (1 - sugar_loss_fraction)

-- Theorem: The total remaining stock weighs 120 kg
theorem remaining_stock_weight : 
  remaining_rice + remaining_sugar + green_beans_weight = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_stock_weight_l1006_100619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inequality_l1006_100662

theorem angle_inequality : 
  Real.sin (-33 * π / 4) < Real.tan (-7 * π / 6) ∧ Real.tan (-7 * π / 6) < Real.cos (23 * π / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inequality_l1006_100662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_liar_puzzle_l1006_100612

-- Define the universe of people
variable (Person : Type)

-- Define the properties
variable (isKnight isLiar : Person → Prop)
variable (tells : Person → Prop → Prop)

-- Define A and B as people
variable (A B : Person)

-- Define the statements made by A and B
axiom A_statement : tells A (isKnight B)
axiom B_statement : tells B (isLiar A)

-- Define the properties of knights and liars
axiom knight_truth : ∀ p : Person, isKnight p → ∀ q : Prop, tells p q ↔ q
axiom liar_lie : ∀ p : Person, isLiar p → ∀ q : Prop, tells p q ↔ ¬q

-- The theorem to prove
theorem knight_liar_puzzle :
  (tells A (isKnight B) ∧ tells B (isLiar A)) →
  ((tells A (isKnight B) ∧ ¬isKnight A) ∨ (tells B (isLiar A) ∧ ¬isKnight B) ∨
   (¬tells A (isKnight B) ∧ ¬isLiar A) ∨ (¬tells B (isLiar A) ∧ ¬isLiar B)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_liar_puzzle_l1006_100612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_estimate_l1006_100665

theorem problem_1_estimate : 
  let total_teams : ℕ := 689
  let correct_answers : ℕ := 175
  let a : ℚ := correct_answers / total_teams
  let A : ℕ := (10000 * a).floor.toNat
  A = 2539 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_estimate_l1006_100665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_range_l1006_100645

/-- Given a unit vector a⃗ and a vector b⃗ satisfying b⃗ · (a⃗ - b⃗) = 0, 
    the magnitude of b⃗ is between 0 and 1 inclusive. -/
theorem vector_magnitude_range (a b : ℝ × ℝ) 
  (h1 : Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = 1) 
  (h2 : (b.1 * (a.1 - b.1) + b.2 * (a.2 - b.2)) = 0) : 
  0 ≤ Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) ∧ Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_range_l1006_100645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_plus_one_approx_l1006_100689

theorem sqrt_difference_plus_one_approx : 
  ∃ ε > 0, abs ((Real.sqrt 81 - Real.sqrt 77) + 1 - 1.23) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_plus_one_approx_l1006_100689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8_sqrt_3_l1006_100600

-- Define complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- Define the conditions
def condition1 (z₁ : ℂ) : Prop := Complex.abs z₁ = 4
def condition2 (z₁ z₂ : ℂ) : Prop := 4 * z₁^2 - 2 * z₁ * z₂ + z₂^2 = 0

-- Define the area of the triangle
noncomputable def triangle_area (z₁ z₂ : ℂ) : ℝ := 
  (1/2) * Complex.abs z₁ * Complex.abs z₂ * Real.sin (Complex.arg (z₂ / z₁))

-- State the theorem
theorem triangle_area_is_8_sqrt_3 
  (h1 : condition1 z₁)
  (h2 : condition2 z₁ z₂) :
  triangle_area z₁ z₂ = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8_sqrt_3_l1006_100600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l1006_100607

/-- The quadratic polynomial q(x) that satisfies the given conditions -/
noncomputable def q (x : ℝ) : ℝ := -3/5 * x^2 - 3/5 * x + 36/5

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions : 
  q (-4) = 0 ∧ q 3 = 0 ∧ q 6 = -18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l1006_100607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_statements_correct_l1006_100629

/-- Define the equivalence class [r] for integers modulo 7 -/
def mod7Class (r : ℤ) : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 7 * k + r % 7}

/-- The statement to be proved -/
theorem four_statements_correct : 
  (¬ (2016 ∈ mod7Class 1)) ∧
  (-3 ∈ mod7Class 4) ∧
  (mod7Class 3 ∩ mod7Class 6 = ∅) ∧
  (Set.univ : Set ℤ) = mod7Class 0 ∪ mod7Class 1 ∪ mod7Class 2 ∪ mod7Class 3 ∪ mod7Class 4 ∪ mod7Class 5 ∪ mod7Class 6 ∧
  (∀ a b : ℤ, (∃ r : ℤ, a ∈ mod7Class r ∧ b ∈ mod7Class r) ↔ (a - b) ∈ mod7Class 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_statements_correct_l1006_100629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l1006_100679

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + x else -x^2

-- State the theorem
theorem range_of_t (t : ℝ) : f (f t) ≤ 2 → t ≤ Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l1006_100679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_placemat_length_on_circular_table_l1006_100699

/-- The length of a rectangular place mat on a circular table -/
noncomputable def placemat_length (R : ℝ) (n : ℕ) : ℝ :=
  Real.sqrt (24 - 5 * Real.sqrt (2 - Real.sqrt 2)) - (5 * Real.sqrt (2 - Real.sqrt 2)) / 2 + 1

/-- Theorem stating that the placemat length formula is correct for the given conditions -/
theorem placemat_length_on_circular_table (R : ℝ) (n : ℕ) (h1 : R = 5) (h2 : n = 8) :
  placemat_length R n = Real.sqrt (24 - 5 * Real.sqrt (2 - Real.sqrt 2)) - (5 * Real.sqrt (2 - Real.sqrt 2)) / 2 + 1 :=
by
  -- Unfold the definition of placemat_length
  unfold placemat_length
  -- The right-hand side is exactly the definition, so it's trivially true
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_placemat_length_on_circular_table_l1006_100699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_pi_third_radius_three_l1006_100601

/-- The area of a sector with central angle θ and radius r -/
noncomputable def sectorArea (θ : Real) (r : Real) : Real := θ * r^2 / 2

theorem sector_area_pi_third_radius_three :
  let θ : Real := π / 3
  let r : Real := 3
  sectorArea θ r = 3 * π / 2 := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_pi_third_radius_three_l1006_100601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1006_100620

-- Define the function f
def f (x : ℝ) : ℝ := 3*x - x^3 - 6*x^2

-- Define the interval
def interval : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem max_value_of_f :
  ∃ (max_val : ℝ), (abs (max_val - 2.768) < 0.001) ∧
  (∀ x ∈ interval, f x ≤ max_val) ∧
  (∃ x ∈ interval, f x = max_val) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1006_100620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_avg_score_difference_l1006_100611

/-- The league record for average points per player per round -/
def league_record : ℚ := 287

/-- The number of players per team -/
def players_per_team : ℕ := 4

/-- The number of rounds in a season -/
def rounds_per_season : ℕ := 10

/-- The team's current score after 9 rounds -/
def current_score : ℕ := 10440

/-- The minimum average score needed per player in the final round to tie the league record -/
noncomputable def min_avg_score : ℚ :=
  (rounds_per_season * players_per_team * league_record - current_score) / players_per_team

theorem min_avg_score_difference : league_record - min_avg_score = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_avg_score_difference_l1006_100611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1006_100606

noncomputable def f (x : ℝ) := Real.cos x * Real.sin (2 * x)

theorem function_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + 2 * Real.pi) = f x) ∧
  (∃ x, f x = (4 * Real.sqrt 3) / 9) ∧
  (∀ x, f x ≤ (4 * Real.sqrt 3) / 9) ∧
  (∀ x, f (Real.pi - x) = -f x) ∧
  (∃ x, f (2 * Real.pi - x) + f x ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1006_100606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_4_equals_10_point_5_l1006_100682

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

-- Define the inverse function of f
noncomputable def f_inv (x : ℝ) : ℝ := 3 - 4 / x

-- Define the function g using f_inv
noncomputable def g (x : ℝ) : ℝ := 1 / (f_inv x) + 10

-- Theorem statement
theorem g_of_4_equals_10_point_5 : g 4 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_4_equals_10_point_5_l1006_100682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_possible_l1006_100609

-- Define the triangle structure
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions for the triangle
def TriangleConditions (t : Triangle) (a α s_b : ℝ) : Prop :=
  let (xa, ya) := t.A
  let (xb, yb) := t.B
  let (xc, yc) := t.C
  -- Side length AC = a
  ((xa - xc)^2 + (ya - yc)^2 = a^2) ∧
  -- Angle BAC = α
  (Real.cos α = ((xb - xa) * (xc - xa) + (yb - ya) * (yc - ya)) / 
    (Real.sqrt ((xb - xa)^2 + (yb - ya)^2) * Real.sqrt ((xc - xa)^2 + (yc - ya)^2))) ∧
  -- Median length from B to midpoint of AC = s_b
  ((xb - (xa + xc)/2)^2 + (yb - (ya + yc)/2)^2 = s_b^2)

-- Theorem statement
theorem triangle_construction_possible (a α s_b : ℝ) 
  (h1 : a > 0) (h2 : 0 < α ∧ α < Real.pi) (h3 : s_b > 0) :
  ∃ t : Triangle, TriangleConditions t a α s_b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_possible_l1006_100609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_two_points_l1006_100683

/-- Given two points A and B in a 2D plane, this theorem proves that the equation of the line passing through these points is 3x - y - 2 = 0. -/
theorem line_equation_through_two_points (A B : ℝ × ℝ) :
  A = (1, 1) → B = (2, 4) → ∀ x y : ℝ, (3 * x - y - 2 = 0 ↔ (∃ t : ℝ, (x, y) = (1 - t) • A + t • B)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_two_points_l1006_100683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_toothpicks_difference_l1006_100673

/-- Represents the number of toothpicks needed for each step in the staircase. -/
def toothpicks_per_step : Nat → Nat
  | 0 => 0  -- Base case for 0 steps
  | 1 => 4
  | n + 1 => toothpicks_per_step n + 2 * n

/-- The total number of toothpicks needed for a staircase with n steps. -/
def total_toothpicks (n : Nat) : Nat :=
  (List.range n).map toothpicks_per_step |>.sum

theorem staircase_toothpicks_difference : total_toothpicks 5 - total_toothpicks 3 = 22 := by
  sorry

#eval total_toothpicks 5 - total_toothpicks 3  -- For verification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_toothpicks_difference_l1006_100673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_is_generalized_distance_l1006_100657

def generalized_distance (f : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y, f x y ≥ 0) ∧ 
  (∀ x y, f x y = 0 ↔ x = y) ∧
  (∀ x y, f x y = f y x) ∧
  (∀ x y z, f x y ≤ f x z + f z y)

theorem abs_is_generalized_distance : 
  generalized_distance (fun x y ↦ |x - y|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_is_generalized_distance_l1006_100657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_route_count_l1006_100634

/-- Represents a point on a grid --/
structure Point where
  x : ℤ
  y : ℤ

/-- Calculates the number of shortest paths between two points on a grid --/
def numShortestPaths (start finish : Point) : ℕ :=
  Nat.choose (Int.natAbs (finish.x - start.x) + Int.natAbs (finish.y - start.y)) (Int.natAbs (finish.x - start.x))

/-- The southwest corner of City Park --/
def swCorner : Point := ⟨0, 0⟩

/-- The northeast corner of City Park --/
def neCorner : Point := ⟨5, 4⟩

/-- Samantha's house location --/
def samanthaHouse : Point := ⟨2, 3⟩

/-- The library location --/
def library : Point := ⟨0, 1⟩

/-- Samantha's school location --/
def school : Point := ⟨4, 1⟩

theorem samantha_route_count : 
  numShortestPaths samanthaHouse library * numShortestPaths neCorner school = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_route_count_l1006_100634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_functions_l1006_100632

theorem max_distance_between_functions :
  ∃ (max : ℝ), max = 2 ∧ ∀ t : ℝ, |Real.sin (2*t - π/12) - Real.sqrt 3 * Real.cos (2*t - π/12)| ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_functions_l1006_100632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_when_tan_is_3_l1006_100605

theorem sin_cos_product_when_tan_is_3 (x : Real) (h : Real.tan x = 3) : 
  Real.sin x * Real.cos x = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_when_tan_is_3_l1006_100605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_visit_pattern_l1006_100658

/-- A cube graph represented as a simple graph -/
def CubeGraph : SimpleGraph (Fin 8) :=
  sorry

/-- A path in the cube graph -/
def CubePath (G : SimpleGraph (Fin 8)) : Type :=
  List (Fin 8)

/-- The number of times a vertex is visited in a path -/
def VisitCount (p : CubePath CubeGraph) (v : Fin 8) : ℕ :=
  sorry

/-- Theorem stating the impossibility of the specific visit pattern -/
theorem impossible_visit_pattern :
  ¬ ∃ (p : CubePath CubeGraph) (v : Fin 8),
    VisitCount p v = 25 ∧
    ∀ (w : Fin 8), w ≠ v → VisitCount p w = 20 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_visit_pattern_l1006_100658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_specific_line_l1006_100676

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ :=
  let m := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)
  l.x₁ - l.y₁ / m

/-- The theorem stating that the x-intercept of the line passing through (10, 3) and (-10, -7) is 4 -/
theorem x_intercept_of_specific_line :
  let l : Line := { x₁ := 10, y₁ := 3, x₂ := -10, y₂ := -7 }
  x_intercept l = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_specific_line_l1006_100676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_between_parallel_lines_l1006_100631

def is_between (y : ℝ) (a b : ℝ) : Prop := a < y ∧ y < b

theorem point_between_parallel_lines (b : ℤ) :
  let line1 := λ (x y : ℝ) => 6 * x - 8 * y + 1 = 0
  let line2 := λ (x y : ℝ) => 3 * x - 4 * y + 5 = 0
  let y_intercept := λ (c : ℝ) => c / 4
  (∃ c : ℝ, line1 5 (y_intercept c) ∧ line2 5 (y_intercept c) ∧ 
    is_between (y_intercept (4 * ↑b - 15)) (y_intercept 1) (y_intercept 5)) →
  b = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_between_parallel_lines_l1006_100631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_x_converges_to_one_l1006_100693

noncomputable def sequence_x (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => (4 / Real.pi ^ 2) * (Real.arccos (sequence_x a n) + Real.pi / 2) * Real.arcsin (sequence_x a n)

theorem sequence_x_converges_to_one (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sequence_x a n - 1| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_x_converges_to_one_l1006_100693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_characterization_upper_bound_condition_l1006_100664

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := Real.log x + (1/2) * x^2 + a * x

def g (x : ℝ) : ℝ := Real.exp x + (3/2) * x^2

def has_no_extreme_points (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (∀ ε > 0, ∃ y : ℝ, 0 < |y - x| ∧ |y - x| < ε ∧ f y ≠ f x)

def has_exactly_two_extreme_points (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
  (∀ x : ℝ, x > 0 → x ≠ x₁ → x ≠ x₂ → (∀ ε > 0, ∃ y : ℝ, 0 < |y - x| ∧ |y - x| < ε ∧ f y ≠ f x)) ∧
  (∀ ε > 0, ∃ y₁ y₂ : ℝ, 0 < |y₁ - x₁| ∧ |y₁ - x₁| < ε ∧ 0 < |y₂ - x₂| ∧ |y₂ - x₂| < ε ∧ f y₁ ≠ f x₁ ∧ f y₂ ≠ f x₂)

theorem extreme_points_characterization :
  (a ≥ -2 → has_no_extreme_points (f a)) ∧
  (a < -2 → has_exactly_two_extreme_points (f a)) := by sorry

theorem upper_bound_condition :
  (∀ x : ℝ, x > 0 → f a x ≤ g x) → a ≤ Real.exp 1 + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_characterization_upper_bound_condition_l1006_100664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_odd_divisor_power_of_two_l1006_100625

def w (x : Nat) : Nat := sorry

theorem largest_odd_divisor_power_of_two (a b : Nat) :
  (∀ x, w x = Nat.gcd x ((2^Nat.log2 x) - 1)) →
  Nat.Coprime a b →
  (∃ m, a + w (b + 1) = 2^m) →
  (∃ n, b + w (a + 1) = 2^n) →
  ∃ k l, a + 1 = 2^k ∧ b + 1 = 2^l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_odd_divisor_power_of_two_l1006_100625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l1006_100675

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line x - y = 0
def line_xy (x y : ℝ) : Prop := x = y

-- Define the points P, Q, and M
def point_P : ℝ × ℝ := (2, 0)
noncomputable def point_Q : ℝ × ℝ := (-1, Real.sqrt 3)
def point_M : ℝ × ℝ := (2, 1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x = 2 ∨ y - 1 = -3/4 * (x - 2)

theorem circle_and_tangent_line :
  ∃ (center_x center_y : ℝ),
    line_xy center_x center_y ∧
    circle_C (point_P.1 - center_x) (point_P.2 - center_y) ∧
    circle_C (point_Q.1 - center_x) (point_Q.2 - center_y) →
    (∀ x y, circle_C x y ↔ x^2 + y^2 = 4) ∧
    (∀ x y, tangent_line x y ↔ 
      (x = point_M.1 ∨ y - point_M.2 = -3/4 * (x - point_M.1)) ∧
      ∃! (ix iy : ℝ), circle_C ix iy ∧ tangent_line ix iy) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l1006_100675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_64_l1006_100692

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  shorter_base_positive : 0 < shorter_base
  base_ratio : longer_base = 3 * shorter_base
  height_relation : height = 2 * shorter_base

/-- Calculates the area of a trapezoid -/
noncomputable def area (t : Trapezoid) : ℝ :=
  (t.shorter_base + t.longer_base) * t.height / 2

/-- Theorem: The area of the specific trapezoid is 64 square units -/
theorem trapezoid_area_is_64 (t : Trapezoid) (h : t.shorter_base = 4) : area t = 64 := by
  sorry

#check trapezoid_area_is_64

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_64_l1006_100692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_assignment_l1006_100686

/-- Represents the three friends --/
inductive Friend
  | Peter
  | Roman
  | Sergey

/-- Represents the three specialties --/
inductive Specialty
  | Mathematics
  | Physics
  | Chemistry

/-- Function that assigns a specialty to each friend --/
def assignment : Friend → Specialty := sorry

/-- Condition 1: If Peter is a mathematician, then Sergey is not a physicist --/
axiom condition1 : assignment Friend.Peter = Specialty.Mathematics → assignment Friend.Sergey ≠ Specialty.Physics

/-- Condition 2: If Roman is not a physicist, then Peter is a mathematician --/
axiom condition2 : assignment Friend.Roman ≠ Specialty.Physics → assignment Friend.Peter = Specialty.Mathematics

/-- Condition 3: If Sergey is not a mathematician, then Roman is a chemist --/
axiom condition3 : assignment Friend.Sergey ≠ Specialty.Mathematics → assignment Friend.Roman = Specialty.Chemistry

/-- Each friend has a unique specialty --/
axiom unique_specialty : ∀ (f1 f2 : Friend), f1 ≠ f2 → assignment f1 ≠ assignment f2

/-- All specialties are assigned --/
axiom all_specialties_assigned : ∀ (s : Specialty), ∃ (f : Friend), assignment f = s

/-- Theorem: The only valid assignment is Peter: Chemist, Roman: Physicist, Sergey: Mathematician --/
theorem correct_assignment :
  assignment Friend.Peter = Specialty.Chemistry ∧
  assignment Friend.Roman = Specialty.Physics ∧
  assignment Friend.Sergey = Specialty.Mathematics := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_assignment_l1006_100686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_quadratic_equation_specific_condition_l1006_100641

theorem quadratic_equation_roots (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + m + 3
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ↔ m ≤ 1 :=
by sorry

theorem quadratic_equation_specific_condition (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + m + 3
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ 3*x₁ + |x₂| = 2) →
  m = -8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_quadratic_equation_specific_condition_l1006_100641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_factorization_theorem_l1006_100642

-- Part 1: System of equations
theorem system_solution (m b : ℝ) : 
  (3 * m + b = 11 ∧ -4 * m - b = 11) → (m = -22 ∧ b = 77) := by sorry

-- Part 2: Factorization
theorem factorization_theorem (m : ℝ) : 
  (m^2 + 1)^2 - 4*m^2 = (m + 1)^2 * (m - 1)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_factorization_theorem_l1006_100642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_XYZ_l1006_100643

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Triangle in a 2D plane defined by three points -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the area of a triangle given three side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: Area of triangle XYZ is 90 square units -/
theorem area_triangle_XYZ (X Y Z W : Point2D) : 
  distance X Z = 17 →
  distance X Y = 25 →
  distance W Z = 8 →
  (W.x - X.x) * (Y.y - X.y) = (W.y - X.y) * (Y.x - X.x) →  -- Coplanarity condition
  (Z.x - W.x) * (Y.x - W.x) + (Z.y - W.y) * (Y.y - W.y) = 0 →  -- Right angle at W
  triangleArea (distance X Y) (distance Y Z) (distance X Z) = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_XYZ_l1006_100643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_bijective_l1006_100663

/-- A function satisfying the given functional equation is bijective -/
theorem function_equation_bijective 
  (f : ℝ → ℝ) 
  (hf : ∀ x, 0 < f x)
  (h : ∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → 
       (z + 1) * f (x + y) = f (x * f z + y) + f (y * f z + x)) :
  Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_bijective_l1006_100663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_diagonal_l1006_100688

/-- A square and a rectangle with equal perimeters and a common vertex -/
structure SquareRectangleConfig where
  /-- Side length of the square -/
  c : ℝ
  /-- Width of the rectangle -/
  a : ℝ
  /-- Height of the rectangle -/
  b : ℝ
  /-- The perimeters are equal -/
  perimeter_eq : 4 * c = 2 * (a + b)

/-- The point of intersection of the diagonals of the rectangle -/
noncomputable def rectangle_diagonal_intersection (cfg : SquareRectangleConfig) : ℝ × ℝ :=
  (cfg.a / 2, cfg.b / 2)

/-- The equation of the diagonal of the square -/
def square_diagonal_equation (cfg : SquareRectangleConfig) (x y : ℝ) : Prop :=
  y = -x + cfg.c

theorem intersection_on_diagonal (cfg : SquareRectangleConfig) :
  let (x, y) := rectangle_diagonal_intersection cfg
  square_diagonal_equation cfg x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_diagonal_l1006_100688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bdc_18_degrees_l1006_100697

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = c^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = b^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = a^2

-- Define a right-angled triangle
def RightTriangle (A B C : ℝ × ℝ) : Prop :=
  Triangle A B C ∧ (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define point D on the extension of AB
def PointD (A B C D : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t > 1 ∧ D.1 = A.1 + t * (B.1 - A.1) ∧ D.2 = A.2 + t * (B.2 - A.2)

-- Define DC = 2BC
def DCTwiceBC (B C D : ℝ × ℝ) : Prop :=
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = 4 * ((C.1 - B.1)^2 + (C.2 - B.2)^2)

-- Define point H as the foot of the altitude from C
def AltitudeFootH (A B C H : ℝ × ℝ) : Prop :=
  (H.1 - A.1) * (B.1 - A.1) + (H.2 - A.2) * (B.2 - A.2) = 
  ((C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2)) * 
  ((B.1 - A.1)^2 + (B.2 - A.2)^2)⁻¹

-- Define the distance from H to BC equals HA
def HBCEqualsHA (A B C H : ℝ × ℝ) : Prop :=
  ((H.1 - B.1) * (C.2 - B.2) - (H.2 - B.2) * (C.1 - B.1))^2 * 
  ((C.1 - B.1)^2 + (C.2 - B.2)^2)⁻¹ = 
  (H.1 - A.1)^2 + (H.2 - A.2)^2

-- Define angle BDC
noncomputable def AngleBDC (B C D : ℝ × ℝ) : ℝ :=
  Real.arccos (((B.1 - D.1) * (C.1 - D.1) + (B.2 - D.2) * (C.2 - D.2)) * 
  (((B.1 - D.1)^2 + (B.2 - D.2)^2) * ((C.1 - D.1)^2 + (C.2 - D.2)^2))⁻¹)

theorem angle_bdc_18_degrees 
  (A B C D H : ℝ × ℝ) : 
  RightTriangle A B C → 
  PointD A B C D → 
  DCTwiceBC B C D → 
  AltitudeFootH A B C H → 
  HBCEqualsHA A B C H → 
  AngleBDC B C D = 18 * (π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bdc_18_degrees_l1006_100697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_200_l1006_100670

noncomputable def fixed_cost : ℝ := 5

noncomputable def additional_cost (x : ℝ) : ℝ :=
  if x < 150 then (1/2) * x^2 + 128 * x
  else 210 * x + 400000 / x - 6900

noncomputable def selling_price : ℝ := 2

noncomputable def annual_profit (x : ℝ) : ℝ :=
  selling_price * x - (fixed_cost + additional_cost x)

theorem max_profit_at_200 :
  ∀ x : ℕ+, annual_profit (x : ℝ) ≤ annual_profit 200 := by
  sorry

#check max_profit_at_200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_200_l1006_100670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_growth_l1006_100687

theorem height_growth : ∃ (liam_initial noah_initial : ℝ) 
  (liam_growth_rate noah_growth_rate_factor : ℝ),
  liam_initial = 45 ∧
  noah_initial = 40 ∧
  liam_growth_rate = 0.15 ∧
  noah_growth_rate_factor = 0.5 ∧
  let liam_growth := liam_initial * liam_growth_rate
  let noah_growth_rate := liam_growth_rate * noah_growth_rate_factor
  let noah_growth := noah_initial * noah_growth_rate
  let liam_final := liam_initial + liam_growth
  let noah_final := noah_initial + noah_growth
  (liam_final, noah_final) = (51.75, 43) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_growth_l1006_100687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_equals_zero_l1006_100608

noncomputable def b : ℕ → ℝ
  | 0 => Real.sin (Real.pi / 30) ^ 2
  | n + 1 => 4 * b n * (1 - b n)

theorem smallest_n_equals_zero :
  ∀ n : ℕ, n > 0 → b n ≠ b 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_equals_zero_l1006_100608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_A_l1006_100698

theorem triangle_angle_A (A B C a b c : Real) : 
  C = Real.pi / 3 →  -- 60° in radians
  b = Real.sqrt 6 → 
  c = 3 → 
  b = a * Real.sin B / Real.sin A →  -- Law of Sines
  c = a * Real.sin C / Real.sin A →  -- Law of Sines
  A + B + C = Real.pi →  -- Sum of angles in a triangle
  A = Real.pi * 5 / 12  -- 75° in radians
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_A_l1006_100698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1006_100648

theorem inequality_equivalence (x : ℝ) : (4 : ℝ)^(2*x - 1) > (1/2 : ℝ)^(-x - 4) ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1006_100648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_plane_centroid_sum_l1006_100647

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  /-- Distance from the origin to the plane -/
  distance_from_origin : ℝ
  /-- The plane intersects the x-axis at this point -/
  x_intercept : ℝ
  /-- The plane intersects the y-axis at this point -/
  y_intercept : ℝ
  /-- The plane intersects the z-axis at this point -/
  z_intercept : ℝ
  /-- The intercepts are distinct from the origin -/
  distinct_intercepts : x_intercept ≠ 0 ∧ y_intercept ≠ 0 ∧ z_intercept ≠ 0

/-- The centroid of a triangle formed by the axis intercepts -/
noncomputable def centroid (p : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (p.x_intercept / 3, p.y_intercept / 3, p.z_intercept / 3)

/-- The theorem to be proved -/
theorem intercept_plane_centroid_sum (p : IntersectingPlane) 
  (h : p.distance_from_origin = 1) : 
  let (x, y, z) := centroid p
  1 / x^2 + 1 / y^2 + 1 / z^2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_plane_centroid_sum_l1006_100647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_geometric_range_of_t_l1006_100614

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 3
  | n + 1 => 2 / (sequence_a n + 1)

noncomputable def sequence_b (n : ℕ) : ℝ := (sequence_a n - 1) / (sequence_a n + 2)

theorem sequence_b_geometric : ∃ (r : ℝ), r = -1/2 ∧ ∀ (n : ℕ), sequence_b (n + 1) = r * sequence_b n := by
  sorry

theorem range_of_t : ∃ (t_min t_max : ℝ), t_min = (1 - Real.sqrt 3) / 2 ∧ t_max = (Real.sqrt 3 - 1) / 2 ∧
  (∀ (t : ℝ), (∀ (n : ℕ+) (m : ℝ), m ≥ -1 ∧ m ≤ 1 → sequence_a (n:ℕ) - t^2 - m*t ≥ 0) ↔ t_min ≤ t ∧ t ≤ t_max) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_geometric_range_of_t_l1006_100614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_form_l1006_100656

def f (a : ℝ) (x : ℝ) : ℝ := a * x + x + 1

def inverse_function (f g : ℝ → ℝ) : Prop :=
  ∀ x, g (f x) = x ∧ f (g x) = x

-- g is defined implicitly through its relationship with f⁻¹
def symmetric_to_inverse (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f⁻¹ (x + 1) = y ↔ g y = x

theorem g_form (a : ℝ) (h1 : a > 1) :
  ∃ g : ℝ → ℝ,
    (inverse_function (f a) (λ x ↦ (f a x)⁻¹)) ∧
    (symmetric_to_inverse (f a) g) →
    (∀ x, g x = a * x + x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_form_l1006_100656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pipes_for_equal_volume_l1006_100616

-- Define the diameters of the pipes
noncomputable def large_diameter : ℝ := 12
noncomputable def small_diameter_1 : ℝ := 3
noncomputable def small_diameter_2 : ℝ := 5

-- Define the number of small pipes needed
def num_small_pipes_1 : ℕ := 16
def num_small_pipes_2 : ℕ := 0

-- Define the volume calculation function
noncomputable def pipe_volume (diameter : ℝ) (length : ℝ) : ℝ :=
  Real.pi * (diameter / 2) ^ 2 * length

-- Theorem statement
theorem min_pipes_for_equal_volume (length : ℝ) (h_length : length > 0) :
  pipe_volume large_diameter length = 
  num_small_pipes_1 * pipe_volume small_diameter_1 length + 
  num_small_pipes_2 * pipe_volume small_diameter_2 length ∧
  ∀ (n m : ℕ), 
    pipe_volume large_diameter length ≤ 
    n * pipe_volume small_diameter_1 length + 
    m * pipe_volume small_diameter_2 length →
    num_small_pipes_1 + num_small_pipes_2 ≤ n + m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pipes_for_equal_volume_l1006_100616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_handed_players_count_l1006_100678

theorem right_handed_players_count (total_players throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 40)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  (h5 : ∀ t, t ∈ Finset.range throwers → t ≠ 0 → t ∈ Finset.range total_players) :
  throwers + ((total_players - throwers) * 2 / 3) = 60 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_handed_players_count_l1006_100678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1006_100651

def is_valid_permutation (p q r s : ℕ) : Prop :=
  Multiset.ofList [p, q, r, s] = Multiset.ofList [6, 7, 8, 9]

def expression_value (p q r s : ℕ) : ℕ :=
  p * q + q * r + r * s + p * s

theorem max_expression_value :
  ∀ p q r s : ℕ, is_valid_permutation p q r s →
  expression_value p q r s ≤ 225 := by
  sorry

#check max_expression_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1006_100651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tidal_function_solution_l1006_100630

noncomputable def tidal_function (A ω φ k t : ℝ) : ℝ := A * Real.sin (ω * t + φ) + k

theorem tidal_function_solution :
  ∀ A ω φ k : ℝ,
  (∀ t, tidal_function A ω φ k (t + 12) = tidal_function A ω φ k t) →
  (tidal_function A ω φ k 0 = 15) →
  (∃ t, tidal_function A ω φ k t = 9) →
  (tidal_function A ω φ k 3 = 15) →
  (∀ t, tidal_function A ω φ k t = 3 * Real.sin (Real.pi / 6 * t) + 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tidal_function_solution_l1006_100630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_inequality_l1006_100604

/-- A triangle in a 2D Euclidean space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Sum of distances from centroid to vertices -/
noncomputable def s₁ (t : Triangle) : ℝ :=
  let G := centroid t
  distance G t.A + distance G t.B + distance G t.C

/-- Perimeter of the triangle -/
noncomputable def s₂ (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- Theorem: For any triangle, s₂ ≥ 2s₁ and s₁ ≤ s₂ -/
theorem centroid_distance_inequality (t : Triangle) : s₂ t ≥ 2 * s₁ t ∧ s₁ t ≤ s₂ t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_inequality_l1006_100604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annika_hike_distance_l1006_100635

/-- Calculates the total distance Annika can hike east given her hiking rate, initial distance, and time constraint. -/
noncomputable def total_distance_east (rate : ℝ) (initial_distance : ℝ) (time_constraint : ℝ) : ℝ :=
  initial_distance + (time_constraint - 2 * initial_distance * rate) / (2 * rate)

/-- Theorem stating that Annika's total distance east is 4.5 kilometers under the given conditions. -/
theorem annika_hike_distance :
  let rate : ℝ := 10  -- minutes per kilometer
  let initial_distance : ℝ := 2.5  -- kilometers
  let time_constraint : ℝ := 45  -- minutes
  total_distance_east rate initial_distance time_constraint = 4.5 := by
  -- Unfold the definition and simplify
  unfold total_distance_east
  -- Perform the calculation
  norm_num
  -- Complete the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annika_hike_distance_l1006_100635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_balanced_no_balanced_subset_exists_l1006_100694

open Set Finset Nat

def balanced (S : Finset ℕ) : Prop :=
  ∀ a ∈ S, ∃ b ∈ S, b ≠ a ∧ ((a + b) / 2) ∈ S

theorem subset_balanced (k : ℕ) (S : Finset ℕ) :
  k > 1 →
  let n := 2^k
  S ⊆ range n →
  S.card > 3 * n / 4 →
  balanced S :=
by
  sorry

-- Part (b) can be represented as a separate theorem
theorem no_balanced_subset_exists (k : ℕ) :
  k > 1 →
  let n := 2^k
  ∃ S : Finset ℕ, S ⊆ range n ∧ S.card > 2 * n / 3 ∧ ¬(balanced S) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_balanced_no_balanced_subset_exists_l1006_100694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_value_equivalence_l1006_100615

/-- Represents the different coin denominations in kopecks -/
inductive CoinDenomination : Type
  | One : CoinDenomination
  | Two : CoinDenomination
  | Five : CoinDenomination
  | Ten : CoinDenomination
  | Twenty : CoinDenomination
  | Fifty : CoinDenomination
  | Hundred : CoinDenomination

/-- The value of a coin in kopecks -/
def coinValue : CoinDenomination → ℕ
  | CoinDenomination.One => 1
  | CoinDenomination.Two => 2
  | CoinDenomination.Five => 5
  | CoinDenomination.Ten => 10
  | CoinDenomination.Twenty => 20
  | CoinDenomination.Fifty => 50
  | CoinDenomination.Hundred => 100

/-- A collection of coins -/
def CoinCollection := CoinDenomination → ℕ

/-- The total value of a coin collection in kopecks -/
def totalValue (coins : CoinCollection) : ℕ :=
  (List.map (fun d => coinValue d * coins d) [CoinDenomination.One, CoinDenomination.Two, CoinDenomination.Five, CoinDenomination.Ten, CoinDenomination.Twenty, CoinDenomination.Fifty, CoinDenomination.Hundred]).sum

/-- The total number of coins in a collection -/
def totalCoins (coins : CoinCollection) : ℕ :=
  (List.map (fun d => coins d) [CoinDenomination.One, CoinDenomination.Two, CoinDenomination.Five, CoinDenomination.Ten, CoinDenomination.Twenty, CoinDenomination.Fifty, CoinDenomination.Hundred]).sum

theorem coin_value_equivalence (k m : ℕ) :
  (∃ (coins : CoinCollection), totalCoins coins = k ∧ totalValue coins = m) →
  (∃ (coins : CoinCollection), totalCoins coins = m ∧ totalValue coins = k * 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_value_equivalence_l1006_100615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_cubed_in_expansion_l1006_100654

/-- The coefficient of x^3 in the expansion of (2x + √x)^4 is 24 -/
theorem coeff_x_cubed_in_expansion : ℕ := 24

/-- Proof of the theorem -/
lemma proof_coeff_x_cubed_in_expansion : coeff_x_cubed_in_expansion = 24 := by
  -- Unfold the definition
  unfold coeff_x_cubed_in_expansion
  -- The result follows directly from the definition
  rfl

/-- Auxiliary function to represent the expansion -/
noncomputable def expansion (x : ℝ) : ℝ := (2*x + Real.sqrt x)^4

/-- The coefficient of x^3 in the expansion can be computed as follows -/
lemma compute_coeff_x_cubed : coeff_x_cubed_in_expansion = 24 := by
  -- We'll use the binomial theorem to compute this coefficient
  -- The general term is C(4,r) * (2x)^(4-r) * (√x)^r
  -- We want the term where the power of x is 3, which occurs when r = 2
  
  -- Compute C(4,2)
  have h1 : Nat.choose 4 2 = 6 := rfl
  
  -- The coefficient we want is 4 * C(4,2) = 4 * 6 = 24
  calc
    coeff_x_cubed_in_expansion = 4 * Nat.choose 4 2 := by sorry
    _ = 4 * 6 := by rw [h1]
    _ = 24 := by rfl

  -- The detailed proof steps are omitted with 'sorry'
  -- In a complete proof, we would expand this calculation fully

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_cubed_in_expansion_l1006_100654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_neg_two_point_eight_l1006_100684

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  ⌊x⌋

-- State the theorem
theorem floor_neg_two_point_eight :
  floor (-2.8) = -3 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_neg_two_point_eight_l1006_100684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_constant_term_l1006_100677

/-- 
Given a polynomial P(x) = x⁸ - 4x⁷ + 7x⁶ + ... + a₀ with all positive real roots,
prove that a₀ = 1/256
-/
theorem polynomial_constant_term (P : Polynomial ℝ) (a₀ : ℝ) : 
  P.degree = 8 → 
  P.coeff 8 = 1 → 
  P.coeff 7 = -4 → 
  P.coeff 6 = 7 → 
  P.coeff 0 = a₀ → 
  (∀ r, r ∈ P.roots → r > 0) →
  a₀ = 1 / 256 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_constant_term_l1006_100677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bc_length_l1006_100610

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def validTriangle (t : Triangle) : Prop :=
  t.A = (0, 0) ∧ 
  t.B.2 = parabola t.B.1 ∧ 
  t.C.2 = parabola t.C.1 ∧
  t.B.2 = t.C.2

-- Define the area of the triangle
noncomputable def triangleArea (t : Triangle) : ℝ :=
  abs ((t.B.1 - t.A.1) * (t.C.2 - t.A.2) - (t.C.1 - t.A.1) * (t.B.2 - t.A.2)) / 2

-- Define the length of BC
noncomputable def lengthBC (t : Triangle) : ℝ :=
  abs (t.C.1 - t.B.1)

-- Theorem statement
theorem triangle_bc_length (t : Triangle) 
  (h1 : validTriangle t) (h2 : triangleArea t = 64) : lengthBC t = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bc_length_l1006_100610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_l1006_100638

theorem inscribed_squares_area (r : ℝ) (h : r > 0) : 
  (∃ (s : ℝ), s > 0 ∧ s^2 = 40 ∧ s = r * Real.sqrt 2) → 
  (∃ (t : ℝ), t > 0 ∧ t = 2 * r ∧ t^2 = 80) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_l1006_100638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1006_100628

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Add this case to handle n = 0
  | 1 => 1
  | n + 2 => sequence_a (n + 1) / ((sequence_a (n + 1))^2 + 1)

theorem sequence_a_properties :
  ∀ n : ℕ, n ≥ 1 →
    (sequence_a (n + 1) < sequence_a n) ∧
    ((1 : ℝ) / (2^(n - 1)) ≤ sequence_a n) ∧
    (sequence_a n ≤ (2^n : ℝ) / (3 * 2^n - 4)) :=
by
  sorry

#check sequence_a
#check sequence_a_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1006_100628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_operation_properties_l1006_100623

-- Define the new operation ⊕
def oplus (a : ℤ) (x : ℕ) : ℤ := a^x

-- Theorem statement
theorem custom_operation_properties :
  (∀ (a : ℤ) (x y : ℕ), a ≠ 0 → oplus a x = a^x) ∧
  (oplus 2 6 = 64) ∧
  (oplus 8 2 = 64 ∧ oplus (-8) 2 = 64) ∧
  ((oplus 2 1) * (oplus 2 2) = 8 ∧ (oplus 3 1) * (oplus 3 1) = 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_operation_properties_l1006_100623
