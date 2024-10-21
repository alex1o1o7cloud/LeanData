import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_l545_54595

theorem min_abs_diff (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : a * b - 3 * a + 4 * b = 137) : 
  ∃ (a' b' : ℤ), a' > 0 ∧ b' > 0 ∧ a' * b' - 3 * a' + 4 * b' = 137 ∧ 
    ∀ (x y : ℤ), x > 0 → y > 0 → x * y - 3 * x + 4 * y = 137 → |a' - b'| ≤ |x - y| ∧ |a' - b'| = 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_l545_54595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l545_54594

/-- Represents a hyperbola with given asymptotic equations and a point it passes through -/
structure Hyperbola where
  asymptote_slope : ℝ
  point : ℝ × ℝ

/-- Standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : ℝ × ℝ → Prop :=
  λ (x, y) ↦ (y^2) / (9/4) - (x^2) / 4 = 1

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity : ℝ := 5/3

/-- Theorem stating the standard equation and eccentricity of the given hyperbola -/
theorem hyperbola_properties (h : Hyperbola) 
  (h_asymptote : h.asymptote_slope = 3/4)
  (h_point : h.point = (2 * Real.sqrt 3, -3)) :
  (standard_equation h h.point ∧ eccentricity = 5/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l545_54594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_common_chord_and_angle_l545_54509

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 - 4*x + y^2 - 2*y = 8
def circle2 (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 4*y = -8

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the angle with x-axis (in radians)
noncomputable def angle_with_x_axis : ℝ := Real.pi/4

-- Theorem statement
theorem circles_common_chord_and_angle :
  ∀ x y : ℝ, 
  (circle1 x y ∧ circle2 x y) → 
  (common_chord x y ∧ 
   angle_with_x_axis = Real.arctan 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_common_chord_and_angle_l545_54509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_sqrt_2x_l545_54548

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x)

theorem average_rate_of_change_sqrt_2x :
  let x₁ : ℝ := 1/2
  let x₂ : ℝ := 2
  (f x₂ - f x₁) / (x₂ - x₁) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_sqrt_2x_l545_54548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l545_54534

noncomputable def f (A ω x : Real) : Real := A * Real.sin (ω * x + Real.pi / 6)

theorem function_properties (A ω : Real) (h1 : A > 0) (h2 : ω > 0)
  (h3 : ∀ x, f A ω x ≤ 2)
  (h4 : ∀ x, f A ω (x + 2*Real.pi/ω) = f A ω x)
  (h5 : ∀ T, T > 0 → (∀ x, f A ω (x + T) = f A ω x) → T ≥ 2*Real.pi/ω) :
  (∀ x, f A ω x = 2 * Real.sin (x + Real.pi / 6)) ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), 
    Real.cos x * (2 * Real.sin (x + Real.pi / 6)) ≤ 3/2) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), 
    Real.cos x * (2 * Real.sin (x + Real.pi / 6)) = 3/2) ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), 
    Real.cos x * (2 * Real.sin (x + Real.pi / 6)) ≥ 0) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), 
    Real.cos x * (2 * Real.sin (x + Real.pi / 6)) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l545_54534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubeRoot_increasing_l545_54538

-- Define the cube root function as noncomputable
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Statement: The cube root function is increasing on ℝ
theorem cubeRoot_increasing : 
  ∀ x y : ℝ, x < y → cubeRoot x < cubeRoot y :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubeRoot_increasing_l545_54538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_order_proof_l545_54506

noncomputable def circle_A_radius : ℝ := 2
noncomputable def circle_B_circumference : ℝ := 10 * Real.pi
noncomputable def circle_C_area : ℝ := 16 * Real.pi

theorem circle_order_proof :
  let r_B := circle_B_circumference / (2 * Real.pi)
  let r_C := Real.sqrt (circle_C_area / Real.pi)
  circle_A_radius < r_C ∧ r_C < r_B := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_order_proof_l545_54506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_squared_in_S_l545_54560

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the set S
def S : Set ℂ := {-1, 0, 1}

-- State the theorem
theorem imaginary_unit_squared_in_S : i^2 ∈ S := by
  -- Proof that i^2 = -1
  have h1 : i^2 = -1 := Complex.I_sq
  
  -- Show that -1 is in S
  have h2 : (-1 : ℂ) ∈ S := by simp [S]
  
  -- Conclude that i^2 is in S
  rw [h1]
  exact h2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_squared_in_S_l545_54560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_tosses_correct_l545_54562

/-- The expected number of tosses in the coin game -/
def expected_tosses : ℚ := 4 + 135 / 256

/-- A fair coin -/
def fair_coin : Type := Unit

/-- The outcome of a coin toss -/
inductive Outcome
  | Heads
  | Tails
deriving BEq, Repr

/-- The game state -/
structure GameState where
  tosses : List Outcome
  max_tosses : Nat

/-- Check if the game should stop -/
def stop_condition (state : GameState) : Bool :=
  match state.tosses.reverse with
  | Outcome.Tails :: rest => 
    let heads_count := rest.takeWhile (· == Outcome.Heads) |>.length
    heads_count % 2 == 1
  | _ => false

/-- The coin game -/
noncomputable def coin_game (coin : fair_coin) (max_tosses : Nat) : GameState → ℚ
| state => sorry

theorem expected_tosses_correct (coin : fair_coin) :
  coin_game coin 10 ⟨[], 10⟩ = expected_tosses := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_tosses_correct_l545_54562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorable_proof_l545_54592

/-- A function that checks if a quadratic expression can be factored into two binomials with integer coefficients -/
def is_factorable (b : ℤ) : Prop :=
  ∃ (r s : ℤ), ∀ (x : ℤ), x^2 + b*x + 1998 = (x + r) * (x + s)

/-- The smallest positive integer b for which x^2 + bx + 1998 factors into two binomials with integer coefficients -/
def smallest_factorable : ℤ := 91

theorem smallest_factorable_proof :
  (is_factorable smallest_factorable) ∧ 
  (∀ b : ℤ, 0 < b → b < smallest_factorable → ¬(is_factorable b)) := by
  sorry

#check smallest_factorable_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorable_proof_l545_54592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l545_54590

/-- Given that the sum of coefficients of (2x + a/x)(x - 2/x)^5 is -1, 
    the constant term in the expansion is -200. -/
theorem constant_term_expansion (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (2*x + a/x) * (x - 2/x)^5 = -1) → 
  ∃ f : Polynomial ℝ, 
    (∀ x : ℝ, x ≠ 0 → Polynomial.eval x f = (2*x + a/x) * (x - 2/x)^5) ∧ 
    f.coeff 0 = -200 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l545_54590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_polar_equation_point_T_coordinates_l545_54589

open Real

-- Define the semicircle C
noncomputable def C (a : ℝ) : ℝ × ℝ :=
  (cos a, 1 + sin a)

-- Define the range of parameter a
def a_range (a : ℝ) : Prop :=
  -π/2 ≤ a ∧ a ≤ π/2

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = 2 * sin θ

-- Define the range of θ
def θ_range (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ ≤ π/2

-- Define point T
noncomputable def T : ℝ × ℝ :=
  (Real.sqrt 3, π/3)

-- Theorem for the polar equation of semicircle C
theorem semicircle_polar_equation :
  ∀ a ρ θ, a_range a → C a = (ρ * cos θ, ρ * sin θ) → θ_range θ → polar_equation ρ θ := by
  sorry

-- Theorem for the polar coordinates of point T
theorem point_T_coordinates :
  ∀ a, a_range a → C a = T → ((Real.sqrt 3) * cos (π/3), (Real.sqrt 3) * sin (π/3)) = T := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_polar_equation_point_T_coordinates_l545_54589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l545_54518

def A : Set ℝ := {x | |x - 2| ≤ 1}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l545_54518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_is_06_l545_54572

/-- Represents the time difference between two routes in minutes -/
noncomputable def route_time_difference (
  route_x_distance : ℝ
  ) (route_x_speed : ℝ
  ) (route_y_total_distance : ℝ
  ) (route_y_normal_speed : ℝ
  ) (route_y_construction_distance : ℝ
  ) (route_y_construction_speed : ℝ
  ) : ℝ :=
  let route_x_time := route_x_distance / route_x_speed * 60
  let route_y_normal_distance := route_y_total_distance - route_y_construction_distance
  let route_y_normal_time := route_y_normal_distance / route_y_normal_speed * 60
  let route_y_construction_time := route_y_construction_distance / route_y_construction_speed * 60
  let route_y_time := route_y_normal_time + route_y_construction_time
  route_x_time - route_y_time

/-- The time difference between Route X and Route Y is 0.6 minutes -/
theorem route_time_difference_is_06 :
  route_time_difference 9 45 8 50 1.5 25 = 0.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_is_06_l545_54572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_NH4I_molecular_weight_l545_54527

/-- The atomic weight of Nitrogen in g/mol -/
def N_weight : ℝ := 14.01

/-- The atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- The atomic weight of Iodine in g/mol -/
def I_weight : ℝ := 126.90

/-- The number of Nitrogen atoms in NH4I -/
def N_count : ℕ := 1

/-- The number of Hydrogen atoms in NH4I -/
def H_count : ℕ := 4

/-- The number of Iodine atoms in NH4I -/
def I_count : ℕ := 1

/-- The molecular weight of NH4I in g/mol -/
def NH4I_weight : ℝ := N_weight * N_count + H_weight * H_count + I_weight * I_count

/-- Theorem stating that the molecular weight of NH4I is approximately 144.942 g/mol -/
theorem NH4I_molecular_weight : 
  |NH4I_weight - 144.942| < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_NH4I_molecular_weight_l545_54527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_packets_needed_to_fill_gunny_bag_l545_54597

-- Define constants
noncomputable def ton_to_pounds : ℝ := 2200
noncomputable def pound_to_ounces : ℝ := 16
noncomputable def gunny_bag_capacity_tons : ℝ := 13.75
noncomputable def packet_weight_pounds : ℝ := 16
noncomputable def packet_weight_ounces : ℝ := 4
noncomputable def packet_weight_grams : ℝ := 250
noncomputable def grams_to_ounces : ℝ := 1 / 28.3495

-- Define the theorem
theorem packets_needed_to_fill_gunny_bag :
  let gunny_bag_capacity_ounces := gunny_bag_capacity_tons * ton_to_pounds * pound_to_ounces
  let packet_weight_ounces := packet_weight_pounds * pound_to_ounces + packet_weight_ounces + packet_weight_grams * grams_to_ounces
  let packets_needed := gunny_bag_capacity_ounces / packet_weight_ounces
  ⌈packets_needed⌉ = 1375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_packets_needed_to_fill_gunny_bag_l545_54597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_2_prop_4_l545_54516

-- Define real numbers a, b, and c
variable (a b c : ℝ)

-- Proposition 2
theorem prop_2 : ¬(∃ (q : ℚ), (a + 5 : ℝ) = ↑q) ↔ ¬(∃ (q : ℚ), (a : ℝ) = ↑q) := by sorry

-- Proposition 4
theorem prop_4 : a < 3 → a < 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_2_prop_4_l545_54516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_is_square_l545_54500

-- Define a quadrilateral
structure Quadrilateral :=
  (a b c d : ℝ)  -- Side lengths
  (α β γ δ : ℝ)  -- Angles in radians

-- Define the properties of the quadrilateral
def is_special_quadrilateral (q : Quadrilateral) : Prop :=
  -- Diagonals are perpendicular (we can't directly represent this with side lengths and angles)
  -- Perimeter is 30 units
  q.a + q.b + q.c + q.d = 30 ∧
  -- One angle is 90 degrees (π/2 radians)
  (q.α = Real.pi/2 ∨ q.β = Real.pi/2 ∨ q.γ = Real.pi/2 ∨ q.δ = Real.pi/2)

-- Define what it means for a quadrilateral to be a square
def is_square (q : Quadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d ∧
  q.α = Real.pi/2 ∧ q.β = Real.pi/2 ∧ q.γ = Real.pi/2 ∧ q.δ = Real.pi/2

-- The theorem to prove
theorem special_quadrilateral_is_square (q : Quadrilateral) :
  is_special_quadrilateral q → is_square q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_is_square_l545_54500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_bound_l545_54584

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x
noncomputable def g (x : ℝ) : ℝ := -1/2 * x^(3/2)

-- State the theorem
theorem function_inequality_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, f a x < g x) → a < -3/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_bound_l545_54584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_inequality_l545_54544

open Real

-- Define the variables and conditions
variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)

-- Define x, y, z as noncomputable
noncomputable def x (a b : ℝ) : ℝ := a + 1/b - 1
noncomputable def y (b c : ℝ) : ℝ := b + 1/c - 1
noncomputable def z (c a : ℝ) : ℝ := c + 1/a - 1

-- Conditions for x, y, z > 0
variable (hx : x a b > 0)
variable (hy : y b c > 0)
variable (hz : z c a > 0)

-- Define the cyclic sum function
noncomputable def cyclicSum (f : ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f (x a b) (y b c) + f (y b c) (z c a) + f (z c a) (x a b)

-- State the theorem
theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hx : x a b > 0) (hy : y b c > 0) (hz : z c a > 0) :
  cyclicSum (fun u v => u * v / (sqrt (u * v) + 2)) a b c ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_inequality_l545_54544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l545_54554

-- Define the function f(x) = ln(x^2 - x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - x)

-- Define the domain of f
def domain_f : Set ℝ := {x | x < 0 ∨ x > 1}

-- Theorem stating that the domain of f is (-∞, 0) ∪ (1, +∞)
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l545_54554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_upper_hemisphere_l545_54580

noncomputable def cube_side_length : ℝ := 4

noncomputable def in_cube (x y z : ℝ) : Prop :=
  -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2 ∧ -2 ≤ z ∧ z ≤ 2

noncomputable def in_upper_hemisphere (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 ≤ 4 ∧ x + y + z ≥ 0

noncomputable def cube_volume : ℝ := cube_side_length ^ 3

noncomputable def hemisphere_volume : ℝ := (4 * Real.pi / 3) * 2^3 / 2

theorem probability_in_upper_hemisphere :
  hemisphere_volume / cube_volume = Real.pi / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_upper_hemisphere_l545_54580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_problem_l545_54546

theorem sheep_problem (a b : ℕ) : 
  (∃ (x y : ℕ), a = x^2 ∧ b = y^2) →  -- a and b are perfect squares
  97 ≤ a + b →                       -- lower bound
  a + b ≤ 108 →                      -- upper bound
  a > b →                            -- Noémie has more sheep
  2 ≤ Nat.sqrt a →                   -- Noémie has at least 2 sheep
  2 ≤ Nat.sqrt b →                   -- Tristan has at least 2 sheep
  Odd (a + b) →                      -- total is odd
  a = 81 ∧ b = 16 := by              -- conclusion
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_problem_l545_54546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_eight_divisors_l545_54563

theorem factorial_eight_divisors : 
  (Finset.filter (λ x : ℕ => x ∣ Nat.factorial 8) (Finset.range (Nat.factorial 8 + 1))).card = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_eight_divisors_l545_54563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_cos_l545_54555

theorem tan_value_given_cos (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.cos x = 4/5) : 
  Real.tan x = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_cos_l545_54555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_N_power_five_l545_54569

theorem det_N_power_five {n : Type*} [Fintype n] [DecidableEq n] 
  (N : Matrix n n ℝ) (h : Matrix.det N = 3) : Matrix.det (N^5) = 243 := by
  have h1 : Matrix.det (N^5) = (Matrix.det N)^5 := by
    exact Matrix.det_pow N 5
  rw [h1, h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_N_power_five_l545_54569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l545_54561

noncomputable def f (x : ℝ) := Real.sqrt (16 - x^2) + 1 / Real.sqrt (Real.sin x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ∈ Set.Ioc (-4) (-Real.pi) ∪ Set.Ioo 0 Real.pi} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l545_54561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l545_54588

/-- Given function f with parameter ω > 0 -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) + (Real.cos (ω * x))^2 - 1/2

/-- Function g derived from f by translation and stretching -/
noncomputable def g (x : ℝ) : ℝ := Real.cos x

/-- Theorem stating the value of ω and the range of k -/
theorem function_properties (ω : ℝ) (h_ω : ω > 0) :
  (∀ x : ℝ, f ω (x + π/(2*ω)) = f ω x) → -- symmetry condition
  ω = 1 ∧ 
  ∀ k : ℝ, (∃ x : ℝ, x ∈ Set.Icc (-π/6) (2*π/3) ∧ g x = k) ↔ k ∈ Set.Icc (-1/2) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l545_54588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_in_interval_l545_54553

/-- The function f(x) = ln x + 2x - 6 --/
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

/-- Theorem: There exists a root of f(x) in the interval (2.5, 2.75) --/
theorem f_has_root_in_interval :
  ∃ x ∈ Set.Ioo 2.5 2.75, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_in_interval_l545_54553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_term_l545_54593

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r^(n - 1)

def is_divisible_by (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

theorem smallest_divisible_term (a₁ : ℚ) (a₂ : ℚ) (n : ℕ) :
  a₁ = 1/2 →
  a₂ = 10 →
  (∀ k < n, ¬ is_divisible_by (Int.floor (geometric_sequence a₁ (a₂ / a₁) k).num) 100000) →
  is_divisible_by (Int.floor (geometric_sequence a₁ (a₂ / a₁) n).num) 100000 →
  n = 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_term_l545_54593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_grid_is_unique_solution_l545_54577

/-- A 3x3 grid represented as a list of 9 natural numbers -/
def Grid := List ℕ

/-- Check if a number is prime -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

/-- Check if the sum of a list of numbers is prime -/
def sumIsPrime (l : List ℕ) : Prop := isPrime (l.sum)

/-- Get the nth row of a 3x3 grid -/
def getRow (g : Grid) (n : Fin 3) : List ℕ :=
  match n with
  | ⟨0, _⟩ => [g.get! 0, g.get! 1, g.get! 2]
  | ⟨1, _⟩ => [g.get! 3, g.get! 4, g.get! 5]
  | ⟨2, _⟩ => [g.get! 6, g.get! 7, g.get! 8]

/-- Get the nth column of a 3x3 grid -/
def getColumn (g : Grid) (n : Fin 3) : List ℕ :=
  match n with
  | ⟨0, _⟩ => [g.get! 0, g.get! 3, g.get! 6]
  | ⟨1, _⟩ => [g.get! 1, g.get! 4, g.get! 7]
  | ⟨2, _⟩ => [g.get! 2, g.get! 5, g.get! 8]

/-- Check if a grid is valid according to the problem conditions -/
def isValidGrid (g : Grid) : Prop :=
  g.length = 9 ∧
  g.toFinset = Finset.range 9 ∧
  (∀ i : Fin 3, sumIsPrime (getRow g i)) ∧
  (∀ i : Fin 3, sumIsPrime (getColumn g i))

/-- The specific grid we want to prove is the only valid solution -/
def specificGrid : Grid := [1, 7, 9, 2, 6, 3, 8, 4, 5]

/-- Theorem stating that the specific grid is the only valid solution -/
theorem specific_grid_is_unique_solution :
  isValidGrid specificGrid ∧ 
  ∀ g : Grid, isValidGrid g → g = specificGrid := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_grid_is_unique_solution_l545_54577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_function_b_squared_l545_54581

/-- A complex number. -/
structure C where
  re : ℝ
  im : ℝ

/-- The modulus (absolute value) of a complex number. -/
noncomputable def C.abs (z : C) : ℝ := Real.sqrt (z.re ^ 2 + z.im ^ 2)

/-- The function f(z) = (a + bi)z. -/
def f (a b : ℝ) (z : C) : C :=
  { re := a * z.re - b * z.im,
    im := b * z.re + a * z.im }

/-- Subtraction for complex numbers. -/
instance : Sub C where
  sub z w := { re := z.re - w.re, im := z.im - w.im }

/-- The theorem statement. -/
theorem equidistant_function_b_squared (a b : ℝ) :
  b > 0 →
  ((C.abs { re := a, im := b }) = 10) →
  (∀ z : C, C.abs (f a b z - z) = C.abs (f a b z)) →
  b ^ 2 = 99.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_function_b_squared_l545_54581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expectation_is_one_l545_54575

/-- Represents the number of specified white mice assigned to the control group out of two -/
def X : ℕ → ℝ := sorry

/-- The total number of mice in the experiment -/
def total_mice : ℕ := 40

/-- The number of mice in the control group -/
def control_mice : ℕ := 20

/-- The number of mice in the experimental group -/
def experimental_mice : ℕ := 20

/-- The probability mass function of X -/
noncomputable def pmf (x : ℕ) : ℝ :=
  if x = 0 then 19 / 78
  else if x = 1 then 20 / 39
  else if x = 2 then 19 / 78
  else 0

/-- X is a valid probability distribution -/
axiom pmf_sum_one : ∑' x, pmf x = 1

/-- The mathematical expectation of X -/
noncomputable def expectation : ℝ := ∑' x, x * pmf x

/-- Theorem: The mathematical expectation of X is 1 -/
theorem expectation_is_one : expectation = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expectation_is_one_l545_54575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_equation_special_circle_passes_through_points_l545_54583

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The standard equation of a circle -/
def standardEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The circle passing through A(-6, 0), B(0, 2), and O(0, 0) -/
noncomputable def specialCircle : Circle :=
  { center := (-3, 1)
    radius := Real.sqrt 10 }

/-- Theorem: The standard equation of the circle passing through A(-6, 0), B(0, 2), and O(0, 0) -/
theorem special_circle_equation (x y : ℝ) :
  standardEquation specialCircle x y ↔ (x + 3)^2 + (y - 1)^2 = 10 := by
  sorry

/-- The circle passes through the given points -/
theorem special_circle_passes_through_points :
  standardEquation specialCircle (-6) 0 ∧
  standardEquation specialCircle 0 2 ∧
  standardEquation specialCircle 0 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_equation_special_circle_passes_through_points_l545_54583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_equation_l545_54566

/-- The equation of the directrix of a parabola -/
noncomputable def directrix_equation (a : ℝ) : ℝ := -1 / (4 * a)

/-- Theorem: The directrix equation of the parabola y = -1/4 * x^2 is y = 1/4 -/
theorem parabola_directrix_equation :
  directrix_equation (-1/4 : ℝ) = 1/4 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_equation_l545_54566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_digit_problem_l545_54526

/-- Represents a mapping of Chinese characters to digits --/
def ChineseToDigit := Char → Fin 9

/-- The set of Chinese characters used in the problem --/
def ChineseChars : Finset Char := {'华', '杯', '赛', '祝', '贺'}

/-- Condition: Different characters represent different digits --/
def is_injective (f : ChineseToDigit) : Prop :=
  ∀ x y, x ∈ ChineseChars → y ∈ ChineseChars → f x = f y → x = y

/-- Condition: "祝" represents 4 --/
def zhu_is_four (f : ChineseToDigit) : Prop :=
  f '祝' = 4

/-- Condition: "贺" represents 8 --/
def he_is_eight (f : ChineseToDigit) : Prop :=
  f '贺' = 8

/-- Convert a sequence of three Chinese characters to a number --/
def to_number (f : ChineseToDigit) (a b c : Char) : ℕ :=
  100 * (f a).val + 10 * (f b).val + (f c).val

/-- The main theorem --/
theorem chinese_digit_problem :
  ∃ (f : ChineseToDigit),
    is_injective f ∧
    zhu_is_four f ∧
    he_is_eight f ∧
    to_number f '华' '杯' '赛' = 7632 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_digit_problem_l545_54526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tileD_in_areaZ_l545_54515

-- Define the tiles and areas
inductive Tile | A | B | C | D
inductive Area | X | Y | Z | W

-- Define the structure of a tile with numbers on its edges
structure TileNumbers where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the numbers for each tile
def tileA : TileNumbers := ⟨5, 3, 2, 4⟩
def tileB : TileNumbers := ⟨2, 4, 5, 3⟩
def tileC : TileNumbers := ⟨3, 6, 1, 5⟩
def tileD : TileNumbers := ⟨5, 2, 3, 6⟩

-- Define a function to get the numbers for a given tile
def getTileNumbers (t : Tile) : TileNumbers :=
  match t with
  | Tile.A => tileA
  | Tile.B => tileB
  | Tile.C => tileC
  | Tile.D => tileD

-- Define the placement of tiles in areas
axiom placement : Tile → Area

-- Define the condition that adjacent tiles must have matching numbers
def matchingEdges (t1 t2 : Tile) (a1 a2 : Area) : Prop :=
  (a1 = Area.X ∧ a2 = Area.Y) ∨ (a1 = Area.Y ∧ a2 = Area.X) →
    (getTileNumbers t1).right = (getTileNumbers t2).left ∧
  (a1 = Area.Y ∧ a2 = Area.Z) ∨ (a1 = Area.Z ∧ a2 = Area.Y) →
    (getTileNumbers t1).bottom = (getTileNumbers t2).top ∧
  (a1 = Area.Z ∧ a2 = Area.W) ∨ (a1 = Area.W ∧ a2 = Area.Z) →
    (getTileNumbers t1).right = (getTileNumbers t2).left

-- Theorem: Tile D is positioned in area Z
theorem tileD_in_areaZ : placement Tile.D = Area.Z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tileD_in_areaZ_l545_54515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_one_equals_negative_three_l545_54582

-- Define the function f as a parameter instead of a definition
theorem f_negative_one_equals_negative_three 
  (f : ℝ → ℝ) (m : ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_nonneg : ∀ x ≥ 0, f x = 2^x + 2*x + m) :
  f (-1) = -3 :=
by
  -- Use the odd function property
  have h1 : f (-1) = -f 1 := h_odd 1
  
  -- Use the definition for non-negative x
  have h2 : f 1 = 2^1 + 2*1 + m := h_nonneg 1 (by norm_num)
  
  -- Combine the results
  rw [h1, h2]
  
  -- Simplify
  ring_nf
  
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_one_equals_negative_three_l545_54582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameters_from_dot_product_range_l545_54539

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The left focus of the ellipse -/
noncomputable def leftFocus (e : Ellipse) : ℝ × ℝ := (-Real.sqrt (e.a^2 - e.b^2), 0)

/-- The right focus of the ellipse -/
noncomputable def rightFocus (e : Ellipse) : ℝ × ℝ := (Real.sqrt (e.a^2 - e.b^2), 0)

/-- The dot product of vectors from a point to the foci -/
noncomputable def fociDotProduct (e : Ellipse) (p : PointOnEllipse e) : ℝ :=
  let f1 := leftFocus e
  let f2 := rightFocus e
  (p.x - f1.1) * (p.x - f2.1) + (p.y - f1.2) * (p.y - f2.2)

/-- The theorem stating the relationship between the ellipse parameters and the dot product range -/
theorem ellipse_parameters_from_dot_product_range (e : Ellipse) :
  (∀ p : PointOnEllipse e, -3 ≤ fociDotProduct e p ∧ fociDotProduct e p ≤ 3) →
  e.a^2 = 9 ∧ e.b^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameters_from_dot_product_range_l545_54539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_radii_l545_54535

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the theorem
theorem triangle_circle_radii 
  (abc : Triangle) 
  (h1 : abc.b = 7)
  (h2 : (abc.a + abc.b) / abc.c = (Real.sin abc.A - Real.sin abc.C) / (Real.sin abc.A - Real.sin abc.B)) :
  ∃ (R r : ℝ), 
    (R = 7 * Real.sqrt 3 / 3) ∧ 
    (0 < r) ∧ 
    (r ≤ 7 * Real.sqrt 3 / 6) ∧
    (R = abc.a / (2 * Real.sin abc.A)) ∧  -- Circumradius formula
    (r = (abc.a * Real.sin (abc.B / 2) * Real.sin (abc.C / 2)) / Real.sin (abc.A / 2)) :=  -- Inradius formula
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_radii_l545_54535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l545_54564

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y = 0

-- Define the line
def my_line (t x y : ℝ) : Prop := 2*t*x - y - 2 - 2*t = 0

-- Theorem statement
theorem line_intersects_circle :
  ∃ (t x y : ℝ), my_circle x y ∧ my_line t x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l545_54564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_midpoint_relation_l545_54576

/-- Given a triangle ABC with D as the midpoint of AB, 
    prove that vector CD equals (1/2)AB - AC -/
theorem vector_midpoint_relation (A B C D : ℝ × ℝ) : 
  D = (A + B) / 2 → (C - D) = (B - A) / 2 - (C - A) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_midpoint_relation_l545_54576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_DE_l545_54504

-- Define the rectangle ABCD
def AB : ℝ := 8
def AD : ℝ := 9

-- Define the area of rectangle ABCD
noncomputable def area_ABCD : ℝ := AB * AD

-- Define the area of triangle DCE
noncomputable def area_DCE : ℝ := area_ABCD / 3

-- Define DC as the base of triangle DCE
def DC : ℝ := AB

-- Define CE
noncomputable def CE : ℝ := 2 * area_DCE / DC

-- Define DE using Pythagorean theorem
noncomputable def DE : ℝ := Real.sqrt (DC^2 + CE^2)

-- Theorem to prove
theorem length_of_DE : DE = 10 := by
  -- Expand the definition of DE
  unfold DE
  -- Expand the definition of CE
  unfold CE
  -- Expand the definition of area_DCE
  unfold area_DCE
  -- Expand the definition of area_ABCD
  unfold area_ABCD
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_DE_l545_54504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_solutions_l545_54568

theorem positive_integer_solutions (n : ℕ) :
  (∃ a b c d : ℕ+, (a + b + c + d : ℝ) = n * Real.sqrt (a * b * c * d : ℝ)) ↔ n ∈ ({1, 2, 3, 4} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_solutions_l545_54568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_equals_zero_l545_54596

/-- The function f(x) = -1/2 * x^2 + x -/
noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + x

theorem n_equals_zero
  (m n k : ℝ)
  (h1 : m < n)
  (h2 : k > 1)
  (h3 : Set.Icc m n ⊆ Set.range f)
  (h4 : Set.range f ∩ Set.Icc m n = Set.Icc (k * m) (k * n)) :
  n = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_equals_zero_l545_54596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l545_54579

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2017 * x + Real.pi / 6) + Real.cos (2017 * x - Real.pi / 3)

/-- The theorem statement -/
theorem min_value_theorem (A : ℝ) (x₁ x₂ : ℝ) :
  (∀ x : ℝ, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  (∀ x : ℝ, f x ≤ A) →
  (∃ x : ℝ, f x = A) →
  (∃ δ : ℝ, δ > 0 ∧ ∀ y z : ℝ, (∀ x : ℝ, f y ≤ f x ∧ f x ≤ f z) → A * |y - z| ≥ δ) →
  (∃ δ : ℝ, δ > 0 ∧ ∀ y z : ℝ, (∀ x : ℝ, f y ≤ f x ∧ f x ≤ f z) → A * |y - z| ≥ δ ∧ δ = 2 * Real.pi / 2017) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l545_54579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l545_54524

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle (renamed to avoid conflict)
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y = 0

-- Define the conditions
def conditions (a b : ℝ) (P F₁ F₂ : ℝ × ℝ) : Prop :=
  a > b ∧ b > 0 ∧
  ellipse_C P.1 P.2 a b ∧
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0 ∧
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 4/3 ∧
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 14/3

-- Define the center of the circle
def M : ℝ × ℝ := (-2, 1)

-- Define the symmetry condition for A and B
def symmetric_points (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2

-- State the theorem
theorem ellipse_and_line_equations 
  (a b : ℝ) (P F₁ F₂ A B : ℝ × ℝ) (L : ℝ → ℝ) :
  conditions a b P F₁ F₂ →
  circle_eq M.1 M.2 →
  ellipse_C A.1 A.2 a b →
  ellipse_C B.1 B.2 a b →
  symmetric_points A B →
  (∀ x, L x = (56 * x - 193) / 81) →
  ellipse_C x y 3 (Real.sqrt (28/9)) ∧
  L M.1 = M.2 ∧ L A.1 = A.2 ∧ L B.1 = B.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l545_54524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_of_solutions_l545_54505

open Real BigOperators

theorem sum_squares_of_solutions : ∃ (S : Finset ℝ), 
  (∀ x ∈ S, x^128 - 128^16 = 0) ∧ 
  (∀ x : ℝ, x^128 - 128^16 = 0 → x ∈ S) ∧
  (∑ x in S, x^2) = 2^(9/4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_of_solutions_l545_54505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l545_54503

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = 2 * f x + f y) →
  (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l545_54503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l545_54599

/-- Represents the time (in minutes) to fill a fraction of the cistern -/
def fill_time (fraction : ℚ) : ℕ := sorry

/-- The time to fill 1/11 of the cistern is 6 minutes -/
axiom partial_fill : fill_time (1/11) = 6

/-- The time to fill the whole cistern -/
def total_fill_time : ℕ := fill_time 1

theorem cistern_fill_time : total_fill_time = 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l545_54599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l545_54519

/-- Given two vectors a and b in a real inner product space, 
    prove that the angle between them is 60 degrees (π/3 radians) 
    under the given conditions. -/
theorem angle_between_vectors (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (ha : ‖a‖ = 4) (hb : ‖b‖ = 3) (hab : ‖a - b‖ = Real.sqrt 13) : 
  Real.arccos (inner a b / (‖a‖ * ‖b‖)) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l545_54519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_equals_one_l545_54532

-- Define acute angles
def is_acute (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- State the theorem
theorem tan_sum_equals_one (α β : Real) 
  (h_acute_α : is_acute α) 
  (h_acute_β : is_acute β)
  (h_tan_β : Real.tan β = (Real.cos α - Real.sin α) / (Real.cos α + Real.sin α)) : 
  Real.tan (α + β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_equals_one_l545_54532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volumes_l545_54542

-- Define the structure of our triangular pyramid
structure TriangularPyramid where
  -- Two isosceles right-angled triangles
  isosceles_right_face1 : Bool
  isosceles_right_face2 : Bool
  -- One equilateral triangle with side length 1
  equilateral_face : Bool
  side_length : ℝ

-- Define a function to calculate the volume of the pyramid
noncomputable def calculate_volume (p : TriangularPyramid) : Finset ℝ :=
  sorry

-- Theorem statement
theorem triangular_pyramid_volumes 
  (p : TriangularPyramid) 
  (h1 : p.isosceles_right_face1 = true) 
  (h2 : p.isosceles_right_face2 = true)
  (h3 : p.equilateral_face = true)
  (h4 : p.side_length = 1) : 
  (calculate_volume p).card = 3 := by
  sorry

#check triangular_pyramid_volumes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volumes_l545_54542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l545_54501

/-- The cubic function f(x) = x³ - 3ax + b, where a ≠ 0 -/
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x + b

/-- The derivative of f(x) -/
def f' (a x : ℝ) : ℝ := 3*x^2 - 3*a

theorem cubic_function_properties (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, f' a x = 0 → x = Real.sqrt a ∨ x = -Real.sqrt a) ∧
  (f' a 2 = 0 ∧ f a b 2 = 8 → a = 4 ∧ b = 24) ∧
  (∀ x, x < -Real.sqrt a → f' a x > 0) ∧
  (∀ x, -Real.sqrt a < x ∧ x < Real.sqrt a → f' a x < 0) ∧
  (∀ x, Real.sqrt a < x → f' a x > 0) ∧
  f a b (-Real.sqrt a) = 2*a*Real.sqrt a + b ∧
  f a b (Real.sqrt a) = -2*a*Real.sqrt a + b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l545_54501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_left_faces_points_l545_54567

/-- Represents a standard six-sided die -/
structure Die where
  faces : Fin 6 → Nat
  valid_faces : ∀ i, faces i ∈ ({1, 2, 3, 4, 5, 6} : Set Nat)
  opposite_sum : ∀ i, faces i + faces (5 - i) = 7

/-- Represents the configuration of four dice glued together -/
structure DiceConfiguration where
  dice : Fin 4 → Die
  left_faces : Fin 4 → Nat

/-- Theorem stating the number of points on each left face -/
theorem left_faces_points (config : DiceConfiguration) :
  config.left_faces 0 = 3 ∧
  config.left_faces 1 = 5 ∧
  config.left_faces 2 = 6 ∧
  config.left_faces 3 = 5 := by
  sorry

#check left_faces_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_left_faces_points_l545_54567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l545_54591

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define points E and F
def E (rect : Rectangle) : ℝ × ℝ := (1, rect.C.2)
def F (rect : Rectangle) : ℝ × ℝ := (rect.D.1, 2)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_dot_product (rect : Rectangle) 
  (h1 : rect.A = (0, 0))
  (h2 : rect.B = (1, 0))
  (h3 : rect.C = (1, 2))
  (h4 : rect.D = (0, 2))
  (h5 : distance (E rect) (F rect) = 1) :
  ∃ (e : ℝ × ℝ) (f : ℝ × ℝ),
    e.1 = 1 ∧ 0 ≤ e.2 ∧ e.2 ≤ 2 ∧
    0 ≤ f.1 ∧ f.1 ≤ 1 ∧ f.2 = 2 ∧
    distance e f = 1 ∧
    dot_product (e.1 - rect.A.1, e.2 - rect.A.2) (f.1 - rect.A.1, f.2 - rect.A.2) =
      5 - Real.sqrt 5 ∧
    ∀ (e' f' : ℝ × ℝ),
      e'.1 = 1 → 0 ≤ e'.2 → e'.2 ≤ 2 →
      0 ≤ f'.1 → f'.1 ≤ 1 → f'.2 = 2 →
      distance e' f' = 1 →
      dot_product (e'.1 - rect.A.1, e'.2 - rect.A.2) (f'.1 - rect.A.1, f'.2 - rect.A.2) ≥
        5 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l545_54591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_A_l545_54520

theorem max_value_A : 
  (∀ (A : ℝ), (∀ (x y : ℕ+), (3 : ℝ) * x^2 + y^2 + 1 ≥ A * (x^2 + x * y + x)) → A ≤ 5/3) ∧ 
  (∀ (x y : ℕ+), (3 : ℝ) * x^2 + y^2 + 1 ≥ 5/3 * (x^2 + x * y + x)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_A_l545_54520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l545_54552

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- Define the domain A
def A : Set ℝ := Set.Ioo (-1) 1

-- Define the set B
def B (a : ℝ) : Set ℝ := Set.Ioo a (a + 1)

-- Theorem statement
theorem f_properties :
  (∃ a : ℝ, a ∈ Set.Icc (-1) 0 ∧ B a ⊆ A) ∧
  (∀ x ∈ A, f (-x) = -f x) ∧
  (∃ x ∈ A, f x ≠ f (-x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l545_54552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_tiling_l545_54556

noncomputable section

open Set

def is_equilateral_triangle (t : Set (ℝ × ℝ)) : Prop := sorry
def side_length (t : Set (ℝ × ℝ)) : ℝ := sorry
def length (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem hexagon_tiling (n : ℕ) :
  let hexagon_perimeter := 1 + 2 + 3 + 4 + 5 + 6
  let total_triangle_sides := 3 * n
  let interior_sides := total_triangle_sides - hexagon_perimeter
  (∃ (hexagon : Set (ℝ × ℝ)), 
    (∃ (s₁ s₂ s₃ s₄ s₅ s₆ : Set (ℝ × ℝ)), 
      s₁ ∪ s₂ ∪ s₃ ∪ s₄ ∪ s₅ ∪ s₆ = hexagon ∧
      length s₁ = 1 ∧ length s₂ = 2 ∧ length s₃ = 3 ∧
      length s₄ = 4 ∧ length s₅ = 5 ∧ length s₆ = 6) ∧
    (∃ (triangles : Finset (Set (ℝ × ℝ))),
      triangles.card = n ∧
      (∀ t ∈ triangles, is_equilateral_triangle t ∧ side_length t = 1) ∧
      (⋃ t ∈ triangles, t) = hexagon)) →
  n % 2 = 1 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_tiling_l545_54556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_system_l545_54533

theorem solve_equation_system (c o u n t s : ℚ) 
  (h1 : c + o = u)
  (h2 : u + n = t)
  (h3 : t + c = s)
  (h4 : o + n + s = 15)
  (h5 : s = t + 3)
  (h6 : c ≠ 0 ∧ o ≠ 0 ∧ u ≠ 0 ∧ n ≠ 0 ∧ t ≠ 0 ∧ s ≠ 0) :
  t = 7.5 := by
  sorry

#check solve_equation_system

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_system_l545_54533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_vector_zero_l545_54529

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem scalar_vector_zero (a : V) (r : ℝ) : r • a = 0 → r = 0 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_vector_zero_l545_54529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_pi_over_2_properties_l545_54545

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 2)

theorem cos_2x_plus_pi_over_2_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + Real.pi) = f x) ∧ 
  (∀ p, p > 0 → (∀ x, f (x + p) = f x) → p ≥ Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_pi_over_2_properties_l545_54545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_covered_by_pentagons_l545_54598

/-- A pentagon consisting of a square and a triangle -/
structure Pentagon where
  square_side : ℝ
  triangle_height : ℝ

/-- The area of a pentagon -/
noncomputable def pentagon_area (p : Pentagon) : ℝ :=
  p.square_side ^ 2 + (1/2) * p.square_side * p.triangle_height

/-- A cube with a given edge length -/
structure Cube where
  edge_length : ℝ

/-- The surface area of a cube -/
noncomputable def cube_surface_area (c : Cube) : ℝ :=
  6 * c.edge_length ^ 2

theorem cube_covered_by_pentagons (c : Cube) (p : Pentagon) :
  c.edge_length = 5 ∧ p.square_side = 5 ∧ p.triangle_height = 2 →
  cube_surface_area c = 5 * pentagon_area p :=
by
  intro h
  have h1 : c.edge_length = 5 := h.left
  have h2 : p.square_side = 5 := h.right.left
  have h3 : p.triangle_height = 2 := h.right.right
  
  calc
    cube_surface_area c = 6 * c.edge_length ^ 2 := rfl
    _ = 6 * 5 ^ 2 := by rw [h1]
    _ = 150 := by norm_num
    _ = 5 * (5 ^ 2 + (1/2) * 5 * 2) := by norm_num
    _ = 5 * (p.square_side ^ 2 + (1/2) * p.square_side * p.triangle_height) := by rw [h2, h3]
    _ = 5 * pentagon_area p := rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_covered_by_pentagons_l545_54598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_reciprocal_distances_l545_54574

-- Define the parametric equation of line l
noncomputable def line_l (t α : ℝ) : ℝ × ℝ :=
  (-1 + t * Real.cos α, 1 + t * Real.sin α)

-- Define the polar equation of curve C
noncomputable def curve_C (θ : ℝ) : ℝ :=
  -4 * Real.cos θ

-- Define point P
def point_P : ℝ × ℝ := (-1, 1)

-- Define the function to calculate the sum of reciprocals of distances
noncomputable def sum_reciprocal_distances (α : ℝ) : ℝ :=
  Real.sqrt (4 * Real.sin (2 * α) + 12) / 2

-- State the theorem
theorem range_of_sum_reciprocal_distances :
  ∀ α ∈ Set.Icc 0 Real.pi,
    ∃ A B : ℝ × ℝ,
      A ≠ B ∧
      (∃ t₁ : ℝ, line_l t₁ α = A) ∧
      (∃ t₂ : ℝ, line_l t₂ α = B) ∧
      (∃ θ₁ : ℝ, (Real.cos θ₁ * curve_C θ₁, Real.sin θ₁ * curve_C θ₁) = A) ∧
      (∃ θ₂ : ℝ, (Real.cos θ₂ * curve_C θ₂, Real.sin θ₂ * curve_C θ₂) = B) ∧
      Real.sqrt 2 ≤ sum_reciprocal_distances α ∧
      sum_reciprocal_distances α ≤ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_reciprocal_distances_l545_54574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_ellipse_l545_54508

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 2 / (1 - 2 * Real.cos θ)

-- Define the property of being an ellipse
def is_ellipse (f : ℝ × ℝ → Prop) : Prop :=
  ∃ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), f (x, y) ↔ a * x^2 + b * y^2 + c * x + d * y + e = 0

-- Theorem statement
theorem polar_equation_is_ellipse :
  is_ellipse (λ (p : ℝ × ℝ) ↦ ∃ θ : ℝ, 
    polar_equation (Real.sqrt (p.1^2 + p.2^2)) θ ∧
    p.1 = (Real.sqrt (p.1^2 + p.2^2)) * Real.cos θ ∧
    p.2 = (Real.sqrt (p.1^2 + p.2^2)) * Real.sin θ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_ellipse_l545_54508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anthony_work_distance_l545_54557

/-- The distance from Anthony's apartment to work, in miles. -/
def distance_to_work : ℝ := sorry

/-- The distance from Anthony's apartment to the gym, in miles. -/
def distance_to_gym : ℝ := 7

/-- The relationship between the distance to work and the distance to the gym. -/
axiom gym_distance_relation : distance_to_gym = distance_to_work / 2 + 2

theorem anthony_work_distance : distance_to_work = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anthony_work_distance_l545_54557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l545_54510

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧ 
  (∀ x : ℝ, f (2 * Real.pi / 3 - x) = f (2 * Real.pi / 3 + x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l545_54510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_invariance_regression_interpretation_regression_through_mean_confidence_interpretation_l545_54502

-- Define a dataset as a list of real numbers
def Dataset := List ℝ

-- Define the mean of a dataset
noncomputable def mean (D : Dataset) : ℝ := (D.sum) / D.length

-- Define the variance of a dataset
noncomputable def variance (D : Dataset) : ℝ :=
  let μ := mean D
  (D.map (fun x => (x - μ)^2)).sum / D.length

-- Statement 1
theorem variance_invariance (D : Dataset) (c : ℝ) :
  variance (D.map (fun x => x + c)) = variance D :=
sorry

-- Statement 2
theorem regression_interpretation (x y : ℝ) :
  y = 5 - 3 * x → (x + 1, y - 3) ∈ {(x', y') | y' = 5 - 3 * x'} :=
sorry

-- Statement 3
theorem regression_through_mean (a b x_bar y_bar : ℝ) :
  y_bar = b * x_bar + a :=
sorry

-- Statement 4
theorem confidence_interpretation (confidence_level : ℝ) :
  confidence_level = 0.99 → (1 - confidence_level) = 0.01 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_invariance_regression_interpretation_regression_through_mean_confidence_interpretation_l545_54502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_divisibility_l545_54565

theorem coprime_divisibility (A n : ℕ) (h : Nat.Coprime A n) :
  ∃ X Y : ℤ, (|X| < Real.sqrt (n : ℝ)) ∧ (|Y| < Real.sqrt (n : ℝ)) ∧ (n : ℤ) ∣ A * X - Y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_divisibility_l545_54565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_l545_54573

/-- Given an acute triangle ABC with side AB = 2 and 1/tan(A) + 1/tan(B) = 4/tan(C),
    the length of the median on side AB is √2. -/
theorem median_length (A B C : ℝ) (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) 
  (h_side : 2 = 2) -- This represents AB = 2
  (h_tan : 1 / Real.tan A + 1 / Real.tan B = 4 / Real.tan C) : 
  ∃ (D : ℝ), D = Real.sqrt 2 ∧ D^2 = (1/4) * (2^2 + 2^2 + 2 * 2 * 2 * Real.cos C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_l545_54573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_ratio_l545_54587

/-- Represents the daily work rate of a man -/
noncomputable def M : ℝ := sorry

/-- Represents the daily work rate of a boy -/
noncomputable def B : ℝ := sorry

/-- Represents the total amount of work in the project -/
noncomputable def W : ℝ := sorry

/-- Group 1 completes the project in 5 days -/
axiom group1 : (12 * M + 16 * B) * 5 = W

/-- Group 2 completes the project in 4 days -/
axiom group2 : (13 * M + 24 * B) * 4 = W

/-- Task A takes twice as long as Task B -/
axiom task_time_ratio : ∃ (a b : ℝ), a = 2 * b

/-- Workers spend half time on each task -/
axiom time_allocation : ∃ (t : ℝ), t > 0 ∧ t / 2 + t / 2 = t

/-- The ratio of daily work done by a man to that of a boy is 2:1 for both tasks -/
theorem work_ratio : M / B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_ratio_l545_54587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_B_l545_54522

theorem polynomial_coefficient_B (A C D : ℤ) : 
  ∃ (roots : List ℤ), 
    (∀ r ∈ roots, r > 0) ∧ 
    (roots.length = 7) ∧
    (roots.sum = 15) ∧
    (∀ z, (z^7 - 15*z^6 + A*z^5 - 306*z^3 + C*z^2 + D*z + 32 = 0) ↔ (z ∈ roots)) →
    -306 = -(List.sum (List.map (λ r₁ => 
      List.sum (List.map (λ r₂ => 
        List.sum (List.map (λ r₃ => 
          if r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ then r₁ * r₂ * r₃ else 0
        ) roots)
      ) roots)
    ) roots))
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_B_l545_54522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l545_54586

noncomputable section

variable (a b : ℝ) (c : ℕ)

axiom eq1 : Real.sqrt (a - 3) + Real.sqrt (3 - a) = 0
axiom eq2 : (3 * b - 4) ^ (1/3 : ℝ) = 2
axiom def_c : c = Int.floor (Real.sqrt 6)

theorem problem_solution :
  a = 3 ∧ b = 4 ∧ c = 2 ∧ Real.sqrt (a + 6 * b - c) = 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l545_54586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_distribution_scenarios_l545_54531

/-- A random variable follows a binomial distribution if it represents the number of successes
    in a fixed number of independent trials, each with the same probability of success. -/
def is_binomial_distribution (ξ : ℕ → ℝ) (n : ℕ) (p : ℝ) : Prop :=
  ∀ k, 0 ≤ k ∧ k ≤ n → ξ k = Nat.choose n k * p^k * (1 - p)^(n - k)

/-- Scenario A: Die throwing -/
def die_throwing (n : ℕ) (ξ : ℕ → ℝ) : Prop :=
  is_binomial_distribution ξ n (1/3)

/-- Scenario B: Shooter -/
def shooter (ξ : ℕ → ℝ) : Prop :=
  ∀ k, ξ k = 0.9 * (1 - 0.9)^(k - 1)

/-- Scenario C: Sampling with replacement -/
def sampling_with_replacement (N M n : ℕ) (ξ : ℕ → ℝ) : Prop :=
  is_binomial_distribution ξ n (M / N)

/-- Scenario D: Sampling without replacement -/
def sampling_without_replacement (N M n : ℕ) (ξ : ℕ → ℝ) : Prop :=
  ∀ k, ξ k = (Nat.choose M k * Nat.choose (N - M) (n - k)) / Nat.choose N n

theorem binomial_distribution_scenarios
  (n N M : ℕ) (ξA ξB ξC ξD : ℕ → ℝ)
  (hA : die_throwing n ξA)
  (hB : shooter ξB)
  (hC : sampling_with_replacement N M n ξC)
  (hD : sampling_without_replacement N M n ξD)
  (hM : M < N) :
  (is_binomial_distribution ξA n (1/3) ∧ is_binomial_distribution ξC n (M / N)) ∧
  (¬ is_binomial_distribution ξB n 0.9 ∧ ¬ is_binomial_distribution ξD n (M / N)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_distribution_scenarios_l545_54531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l545_54537

noncomputable def f (x : Real) : Real := (Real.sin x + Real.sqrt 3 * Real.cos x) * (Real.cos x - Real.sqrt 3 * Real.sin x)

theorem smallest_positive_period_of_f :
  ∃ (T : Real), T > 0 ∧ (∀ (x : Real), f (x + T) = f x) ∧
  (∀ (T' : Real), T' > 0 ∧ (∀ (x : Real), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l545_54537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_surface_area_l545_54511

/-- A right pyramid with a rectangular base -/
structure RightPyramid where
  base_length : ℝ
  base_width : ℝ
  height : ℝ

/-- Calculate the total surface area of a right pyramid -/
noncomputable def total_surface_area (p : RightPyramid) : ℝ :=
  let base_area := p.base_length * p.base_width
  let half_diagonal := Real.sqrt ((p.base_length / 2) ^ 2 + (p.base_width / 2) ^ 2)
  let slant_height := Real.sqrt (p.height ^ 2 + half_diagonal ^ 2)
  let lateral_area := (p.base_length + p.base_width) * slant_height
  base_area + lateral_area

/-- The theorem stating the total surface area of the specific pyramid -/
theorem specific_pyramid_surface_area :
  let p := RightPyramid.mk 8 6 15
  total_surface_area p = 48 + 55 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_surface_area_l545_54511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_satisfying_inequality_l545_54551

theorem least_integer_satisfying_inequality :
  ∀ y : ℤ, (3 * y.natAbs + 6 < 24) → y ≥ -5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_satisfying_inequality_l545_54551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_proportion_l545_54523

/-- Triangle PQR with angle bisector PE -/
structure AngleBisectorTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  E : ℝ × ℝ
  p : ℝ
  q : ℝ
  r : ℝ
  x : ℝ
  y : ℝ

/-- The angle bisector theorem holds for this triangle -/
axiom angle_bisector_theorem {t : AngleBisectorTriangle} : t.x / t.r = t.y / t.q

/-- PE is indeed an angle bisector -/
axiom is_angle_bisector {t : AngleBisectorTriangle} : t.x + t.y = t.p

/-- Theorem: In triangle PQR, if PE bisects angle P and meets QR at E, 
    then y/q = p/(r+q), where y = RE, q is opposite to angle Q, 
    p is opposite to angle P, and r is opposite to angle R -/
theorem angle_bisector_proportion (t : AngleBisectorTriangle) : 
  t.y / t.q = t.p / (t.r + t.q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_proportion_l545_54523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_theorem_l545_54514

/-- The centroid theorem for triangles -/
theorem centroid_theorem {V : Type*} [AddCommGroup V] [Module ℚ V] 
  (O A B C Q : V) : 
  (∃ (M N K : V), 
    M = (1/2 : ℚ) • (B + C) ∧ 
    N = (1/2 : ℚ) • (A + C) ∧ 
    K = (1/2 : ℚ) • (A + B) ∧ 
    Q - A = (2/3 : ℚ) • (M - A) ∧
    Q - B = (2/3 : ℚ) • (N - B) ∧
    Q - C = (2/3 : ℚ) • (K - C)) →
  Q = (1/3 : ℚ) • (A + B + C) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_theorem_l545_54514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_flow_maximization_l545_54559

/-- Safe interval distance as a function of speed -/
noncomputable def d (v : ℝ) : ℝ := (1 / 4000) * v^2

/-- Unit time flow as a function of speed -/
noncomputable def Q (v : ℝ) : ℝ := v / (0.4 + d v)

/-- Maximum unit time flow -/
def Q_max : ℝ := 50

/-- Optimal speed -/
def v_optimal : ℝ := 40

theorem train_flow_maximization :
  ∀ v : ℝ, v > 0 → Q v ≤ Q_max ∧ Q v_optimal = Q_max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_flow_maximization_l545_54559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_value_l545_54549

-- Define a quadratic polynomial
def quadratic_polynomial (a b c : ℚ) : ℚ → ℚ := λ x ↦ a * x^2 + b * x + c

-- Define the divisibility condition
def divisibility_condition (q : ℚ → ℚ) : Prop :=
  ∃ p : ℚ → ℚ, ∀ x, q x^3 - x = p x * (x - 2) * (x + 2) * (x - 5)

theorem quadratic_polynomial_value (a b c : ℚ) :
  divisibility_condition (quadratic_polynomial a b c) →
  quadratic_polynomial a b c 7 = 9/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_value_l545_54549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_negative_half_l545_54530

-- Define the line L
noncomputable def line_L (t α : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, 1 + t * Real.sin α)

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the midpoint condition
def is_midpoint (m a b : ℝ × ℝ) : Prop :=
  m.1 = (a.1 + b.1)/2 ∧ m.2 = (a.2 + b.2)/2

theorem line_slope_is_negative_half (α : ℝ) :
  ∃ (t₁ t₂ : ℝ),
    let a := line_L t₁ α
    let b := line_L t₂ α
    on_ellipse a.1 a.2 ∧
    on_ellipse b.1 b.2 ∧
    is_midpoint (2, 1) a b →
    Real.tan α = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_negative_half_l545_54530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_debt_time_is_50_div_3_l545_54585

/-- The time (in days) when Darren's and Fergie's debts are equal -/
noncomputable def equal_debt_time (darren_initial_debt : ℝ) (darren_borrowed : ℝ) (darren_interest_rate : ℝ)
                    (fergie_borrowed : ℝ) (fergie_interest_rate : ℝ) : ℝ :=
  (fergie_borrowed - darren_initial_debt - darren_borrowed) /
  (darren_borrowed * darren_interest_rate - fergie_borrowed * fergie_interest_rate)

/-- Theorem stating that the time when Darren's and Fergie's debts are equal is 50/3 days -/
theorem equal_debt_time_is_50_div_3 :
  equal_debt_time 50 200 0.12 300 0.07 = 50 / 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_debt_time_is_50_div_3_l545_54585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_count_l545_54521

def is_valid_permutation (perm : List Char) : Bool :=
  perm.length = 4 ∧
  perm.toFinset = {'a', 'b', 'c', 'd'} ∧
  (List.zip perm perm.tail).all (fun (x, y) =>
    (x, y) ∉ [('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'b'), ('c', 'd'), ('d', 'c')])

def count_valid_permutations : Nat :=
  (List.permutations ['a', 'b', 'c', 'd']).filter is_valid_permutation |>.length

theorem valid_permutations_count :
  count_valid_permutations = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_count_l545_54521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_marked_price_l545_54528

/-- The marked price of an article -/
noncomputable def marked_price (initial_price discount_on_initial gain_percentage discount_on_marked : ℝ) : ℝ :=
  let cost_price := initial_price * (1 - discount_on_initial)
  let selling_price := cost_price * (1 + gain_percentage)
  selling_price / (1 - discount_on_marked)

/-- Theorem stating the correct marked price given the conditions -/
theorem correct_marked_price :
  marked_price 36 0.15 0.25 0.1 = 42.5 := by
  -- Unfold the definition of marked_price
  unfold marked_price
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_marked_price_l545_54528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsequence_arithmetic_common_difference_subsequence_geometric_sum_bound_l545_54543

def sequence_a : ℕ → ℚ := λ n => if n = 0 then 0 else 1 / n

def is_subsequence {α : Type*} (s : ℕ → α) (t : ℕ → α) : Prop :=
  ∃ f : ℕ → ℕ, Monotone f ∧ StrictMono f ∧ ∀ n, t n = s (f n)

def arithmetic_sequence (s : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n, s (n + 1) - s n = d

def geometric_sequence (s : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n, s (n + 1) = s n * q

theorem subsequence_arithmetic_common_difference
  (b : ℕ → ℚ)
  (h_subseq : is_subsequence sequence_a b)
  (h_arith : arithmetic_sequence b)
  (h_len : ∀ n < 5, b n ≠ 0) :
  ∃ d : ℚ, -1/8 < d ∧ d < 0 ∧ ∀ n, b (n + 1) - b n = d :=
sorry

theorem subsequence_geometric_sum_bound
  (c : ℕ → ℚ)
  (m : ℕ)
  (h_subseq : is_subsequence sequence_a c)
  (h_geom : geometric_sequence c)
  (h_m : m ≥ 3) :
  Finset.sum (Finset.range m) c ≤ 2 - 1 / (2^(m-1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsequence_arithmetic_common_difference_subsequence_geometric_sum_bound_l545_54543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l545_54513

def t (n : ℕ) : ℚ :=
  if n % 2 = 1 then (1 : ℚ) / 7^n else (2 : ℚ) / 7^n

theorem sequence_sum : ∑' (n : ℕ), t (n + 1) = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l545_54513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_give_change_all_minimum_coins_needed_l545_54578

/-- Represents the fare for a minibus ride in rubles -/
def fare : ℕ := 75

/-- Represents the number of passengers -/
def num_passengers : ℕ := 15

/-- Represents the probability of a passenger paying with a 100-ruble bill -/
def prob_100_bill : ℚ := 1/2

/-- Represents the probability of a passenger paying with exact change -/
def prob_exact_change : ℚ := 1/2

/-- Represents the change given to a passenger paying with a 100-ruble bill -/
def change_amount : ℕ := 25

/-- Represents the initial number of coins the driver has -/
def initial_coins : ℕ := 0

/-- Function representing the probability of giving change to all passengers -/
noncomputable def probability_of_giving_change_to_all (initial_coins : ℕ) (num_passengers : ℕ) (prob_100_bill : ℚ) : ℚ :=
sorry

/-- Theorem stating the probability of giving change to all passengers paying with 100-ruble bills -/
theorem probability_give_change_all : 
  ∃ (p : ℚ), p = 6435 / 32768 ∧ 
  (probability_of_giving_change_to_all initial_coins num_passengers prob_100_bill = p) := by
  sorry

/-- Theorem stating the minimum number of coins needed for a 0.95 probability of giving change -/
theorem minimum_coins_needed : 
  ∃ (n : ℕ), n = 11 ∧ 
  (probability_of_giving_change_to_all n num_passengers prob_100_bill ≥ 95/100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_give_change_all_minimum_coins_needed_l545_54578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_region_implies_a_eq_neg_eight_l545_54525

/-- A function f(x) = √(ax² + bx + c) with domain D forms a square region -/
def forms_square_region (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∃ (a b c : ℝ), a < 0 ∧
  (∀ x ∈ D, f x = Real.sqrt (a * x^2 + b * x + c)) ∧
  (∃ (s t : Set ℝ), s ⊆ D ∧ t ⊆ D ∧ 
    (∀ (x y : ℝ), x ∈ s ∧ y ∈ t → (x, f y) ∈ (Set.prod s (Set.range f))) ∧
    ∃ (side : ℝ), side > 0 ∧ Set.prod s (Set.range f) = Set.Icc 0 side ×ˢ Set.Icc 0 side)

theorem square_region_implies_a_eq_neg_eight
  (f : ℝ → ℝ) (D : Set ℝ) (hf : forms_square_region f D) :
  ∃ (a b c : ℝ), a = -8 ∧ a < 0 ∧
  (∀ x ∈ D, f x = Real.sqrt (a * x^2 + b * x + c)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_region_implies_a_eq_neg_eight_l545_54525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sine_fraction_at_pi_l545_54547

theorem limit_sine_fraction_at_pi :
  ∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ x : ℝ, 0 < |x - π| ∧ |x - π| < δ →
    |(1 - Real.sin (x / 2)) / (π - x) - 0| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sine_fraction_at_pi_l545_54547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_l545_54517

/-- A rhombus with given area and one diagonal -/
structure Rhombus where
  area : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  side : ℝ

/-- The rhombus satisfies the given conditions -/
def is_valid_rhombus (r : Rhombus) : Prop :=
  r.area = 24 ∧ r.diagonal1 = 6 ∧ r.area = (1/2) * r.diagonal1 * r.diagonal2

/-- The side length of the rhombus can be calculated using its diagonals -/
noncomputable def side_from_diagonals (d1 d2 : ℝ) : ℝ :=
  Real.sqrt ((d1/2)^2 + (d2/2)^2)

/-- Theorem: The side length of the given rhombus is 5 -/
theorem rhombus_side_length (r : Rhombus) (h : is_valid_rhombus r) : r.side = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_l545_54517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l545_54550

-- Define the function f(x) = 2x - m/x
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * x - m / x

-- Theorem statement
theorem function_properties (m : ℝ) :
  (f m 1 = 1) →  -- The graph passes through (1,1)
  (m = 1) ∧  -- m equals 1
  (∀ x : ℝ, x ≠ 0 → f m (-x) = -(f m x)) ∧  -- f is an odd function
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → f m x < f m y)  -- f is increasing on (0, +∞)
  := by
    intro h1
    have h2 : m = 1 := by
      -- Proof that m = 1
      sorry
    have h3 : ∀ x : ℝ, x ≠ 0 → f m (-x) = -(f m x) := by
      -- Proof that f is an odd function
      sorry
    have h4 : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f m x < f m y := by
      -- Proof that f is increasing on (0, +∞)
      sorry
    exact ⟨h2, h3, h4⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l545_54550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_sum_of_c_and_d_l545_54512

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 9 * x^2)

-- State the theorem about the range of h
theorem range_of_h :
  ∀ y : ℝ, (∃ x : ℝ, h x = y) ↔ 0 < y ∧ y ≤ 1 := by sorry

-- Define c and d based on the range
def c : ℝ := 0
def d : ℝ := 1

-- State the theorem about c + d
theorem sum_of_c_and_d : c + d = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_sum_of_c_and_d_l545_54512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_is_twelve_polygon_sides_satisfies_condition_l545_54541

/-- The number of sides in a polygon where the sum of interior angles is 1/4 more than 
    the sum of exterior angles by 90° -/
def polygon_sides : ℕ :=
  let n : ℕ := 12  -- The number of sides we want to prove
  let interior_sum : ℝ := (n - 2) * 180
  let exterior_sum : ℝ := 360
  n

theorem polygon_sides_is_twelve : polygon_sides = 12 := by
  -- Unfold the definition of polygon_sides
  unfold polygon_sides
  -- The definition directly returns 12, so this is trivially true
  rfl

theorem polygon_sides_satisfies_condition : 
  let n := polygon_sides
  let interior_sum : ℝ := (n - 2) * 180
  let exterior_sum : ℝ := 360
  interior_sum = (1 / 4) * exterior_sum + 90 := by
  -- This is where we would prove that the condition is satisfied
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_is_twelve_polygon_sides_satisfies_condition_l545_54541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_arrangements_all_are_permutations_all_permutations_included_l545_54558

-- Define a type for the girls
inductive Girl : Type
| A : Girl  -- Anya
| S : Girl  -- Sanya
| T : Girl  -- Tanya

-- Define a type for a queue arrangement
def Arrangement := List Girl

-- Define the set of all possible arrangements
def AllArrangements : List Arrangement :=
  [[Girl.A, Girl.S, Girl.T],
   [Girl.A, Girl.T, Girl.S],
   [Girl.S, Girl.A, Girl.T],
   [Girl.S, Girl.T, Girl.A],
   [Girl.T, Girl.A, Girl.S],
   [Girl.T, Girl.S, Girl.A]]

-- Theorem: There are exactly 6 possible arrangements
theorem six_arrangements :
  List.length AllArrangements = 6 := by
  rfl

-- Theorem: All arrangements are permutations of the three girls
theorem all_are_permutations (arr : Arrangement) :
  arr ∈ AllArrangements →
  Multiset.ofList arr = Multiset.ofList [Girl.A, Girl.S, Girl.T] := by
  sorry

-- Theorem: All permutations are included in the arrangements
theorem all_permutations_included (perm : Arrangement) :
  Multiset.ofList perm = Multiset.ofList [Girl.A, Girl.S, Girl.T] →
  perm ∈ AllArrangements := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_arrangements_all_are_permutations_all_permutations_included_l545_54558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_distance_inequality_l545_54571

/-- Predicate to check if a quadrilateral is convex -/
def ConvexQuadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

/-- Predicate to check if a point is inside a quadrilateral -/
def PointInsideQuadrilateral (M A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

/-- Given a convex quadrilateral ABCD and a point M inside it,
    the sum of distances from M to the vertices is less than
    the sum of pairwise distances between vertices. -/
theorem quadrilateral_distance_inequality
  (A B C D M : EuclideanSpace ℝ (Fin 2))
  (h_convex : ConvexQuadrilateral A B C D)
  (h_inside : PointInsideQuadrilateral M A B C D) :
  dist M A + dist M B + dist M C + dist M D <
  dist A B + dist B C + dist C D + dist D A :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_distance_inequality_l545_54571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_l545_54536

/-- Proves that borrowing $60 at 6% simple annual interest results in owing $63.6 after one year -/
theorem borrowed_amount (P : ℝ) (interest_rate : ℝ) (owed_amount : ℝ) : 
  interest_rate = 0.06 →
  owed_amount = 63.6 →
  P * (1 + interest_rate) = owed_amount →
  P = 60 := by
  intros h_rate h_owed h_equation
  -- The proof steps would go here
  sorry

#check borrowed_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_l545_54536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_n_values_l545_54507

theorem triangle_n_values :
  let valid_n := {n : ℕ+ | 2 ≤ n ∧ n ≤ 12}
  Finset.card (Finset.filter (λ n => 2 ≤ n ∧ n ≤ 12) (Finset.range 13)) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_n_values_l545_54507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l545_54540

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and the condition 3a² + 3b² - c² = 4ab, prove that cos(cos A) ≤ cos(sin B) -/
theorem triangle_inequality (a b c A B C : ℝ) : 
  3 * a^2 + 3 * b^2 - c^2 = 4 * a * b →
  0 < a → 0 < b → 0 < c →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  a * Real.sin B = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin C →
  Real.cos (Real.cos A) ≤ Real.cos (Real.sin B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l545_54540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_eq_one_iff_x_eq_kpi_plus_pi_fourth_l545_54570

theorem tan_eq_one_iff_x_eq_kpi_plus_pi_fourth :
  ∀ x : ℝ, Real.tan x = 1 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_eq_one_iff_x_eq_kpi_plus_pi_fourth_l545_54570
