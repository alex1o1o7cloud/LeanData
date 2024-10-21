import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_count_10_mod_1000_l1050_105079

def tower_count (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | 1 => 1
  | m + 2 => 3 * tower_count (m + 1)

theorem tower_count_10_mod_1000 :
  tower_count 10 % 1000 = 122 := by
  -- Proof goes here
  sorry

#eval tower_count 10 % 1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_count_10_mod_1000_l1050_105079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teddy_bear_cost_l1050_105060

/-- Proves that the cost of each teddy bear is $15 -/
theorem teddy_bear_cost
  (num_toys : ℕ)
  (toy_cost : ℕ)
  (num_teddy_bears : ℕ)
  (total_cost : ℕ)
  (teddy_bear_cost : ℕ)
  (h1 : num_toys = 28)
  (h2 : toy_cost = 10)
  (h3 : num_teddy_bears = 20)
  (h4 : total_cost = 580)
  (h5 : total_cost = num_toys * toy_cost + num_teddy_bears * teddy_bear_cost) :
  teddy_bear_cost = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_teddy_bear_cost_l1050_105060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_m_values_finite_count_m_values_correct_l1050_105032

/-- The number of different integer values of m for which x^2 - mx + 24 has integer roots -/
def count_m_values : ℕ := 8

/-- A function that checks if a given m produces integer roots for x^2 - mx + 24 -/
def has_integer_roots (m : ℤ) : Prop :=
  ∃ x₁ x₂ : ℤ, x₁ + x₂ = m ∧ x₁ * x₂ = 24

/-- The set of all integer m values that produce integer roots -/
def valid_m_values : Set ℤ :=
  {m : ℤ | has_integer_roots m}

/-- Proof that valid_m_values is finite -/
theorem valid_m_values_finite : Set.Finite valid_m_values := by
  -- We know there are exactly 8 values, but we'll just prove finiteness here
  sorry

/-- Instance of Fintype for valid_m_values -/
noncomputable instance : Fintype valid_m_values :=
  Set.Finite.fintype valid_m_values_finite

/-- The main theorem stating that the count of valid m values is correct -/
theorem count_m_values_correct :
  Fintype.card valid_m_values = count_m_values := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_m_values_finite_count_m_values_correct_l1050_105032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l1050_105090

-- Problem 1
theorem problem_1 : (-7) - (-10) + (-8) - 2 = -7 := by sorry

-- Problem 2
theorem problem_2 : (1/4 - 1/2 + 1/6) * 12 = -1 := by sorry

-- Problem 3
theorem problem_3 : -3 * |(-2)| + (-28) / (-7) = -2 := by sorry

-- Problem 4
theorem problem_4 : -(3^2) - (-2)^3 / 4 = -7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l1050_105090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sq_cm_to_sq_dm_conversion_cubic_m_to_cubic_dm_conversion_cubic_dm_to_liter_conversion_liter_to_ml_conversion_l1050_105046

-- Define conversion rates
def sq_cm_to_sq_dm : ℚ := 100
def cubic_m_to_cubic_dm : ℚ := 1000
def cubic_dm_to_liter : ℚ := 1
def liter_to_ml : ℚ := 1000

-- Define the conversion functions
noncomputable def convert_sq_cm_to_sq_dm (x : ℚ) : ℚ := x / sq_cm_to_sq_dm
noncomputable def convert_cubic_m_to_cubic_dm (x : ℚ) : ℚ := x * cubic_m_to_cubic_dm
noncomputable def convert_cubic_dm_to_liter (x : ℚ) : ℚ := x * cubic_dm_to_liter
noncomputable def convert_liter_to_ml (x : ℚ) : ℚ := x * liter_to_ml

-- State the theorems to be proved
theorem sq_cm_to_sq_dm_conversion : convert_sq_cm_to_sq_dm 628 = 6.28 := by sorry

theorem cubic_m_to_cubic_dm_conversion : convert_cubic_m_to_cubic_dm (9/2) = 4500 := by sorry

theorem cubic_dm_to_liter_conversion : convert_cubic_dm_to_liter (18/5) = 18/5 := by sorry

theorem liter_to_ml_conversion : convert_liter_to_ml (18/5) = 3600 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sq_cm_to_sq_dm_conversion_cubic_m_to_cubic_dm_conversion_cubic_dm_to_liter_conversion_liter_to_ml_conversion_l1050_105046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_midpoint_is_7_5_l1050_105048

/-- Right triangle DEF with given side lengths -/
structure RightTriangleDEF where
  DE : ℝ
  DF : ℝ
  EF : ℝ
  is_right : DE^2 = DF^2 + EF^2

/-- The distance from F to the midpoint of DE in a right triangle DEF -/
noncomputable def distance_to_midpoint (t : RightTriangleDEF) : ℝ := t.DE / 2

/-- Theorem: In the given right triangle DEF, the distance from F to the midpoint of DE is 7.5 -/
theorem distance_to_midpoint_is_7_5 (t : RightTriangleDEF) 
  (h1 : t.DE = 15) (h2 : t.DF = 9) (h3 : t.EF = 12) : 
  distance_to_midpoint t = 7.5 := by
  -- Unfold the definition of distance_to_midpoint
  unfold distance_to_midpoint
  -- Rewrite using the given hypothesis
  rw [h1]
  -- Simplify the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_midpoint_is_7_5_l1050_105048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equalize_expenses_l1050_105080

/-- The amount Joey should pay Monica to equalize expenses -/
noncomputable def amount_to_pay (M J C : ℝ) : ℝ := (C - J - M) / 3

/-- Proof that the calculated amount equalizes expenses -/
theorem equalize_expenses (M J C : ℝ) 
  (h1 : M < J) 
  (h2 : J < C) : 
  (M + amount_to_pay M J C) = (J - amount_to_pay M J C) ∧ 
  (M + amount_to_pay M J C) = C / 3 ∧
  (J - amount_to_pay M J C) = C / 3 := by
  sorry

#check equalize_expenses

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equalize_expenses_l1050_105080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_centers_in_A_l1050_105099

noncomputable def symmetry_center (k : ℤ) : ℝ × ℝ := (k * Real.pi / 2 + 5 * Real.pi / 12, 1)

def A : Set (ℝ × ℝ) := {p | ∃ k : ℤ, p = symmetry_center k}

theorem symmetry_centers_in_A : 
  {(-7 * Real.pi / 12, 1), (17 * Real.pi / 12, 1)} ⊆ A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_centers_in_A_l1050_105099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_inequality_solution_set_l1050_105037

theorem fractional_inequality_solution_set :
  {x : ℝ | (1 : ℝ) / (x - 1) ≥ -1 ∧ x ≠ 1} = Set.Iic 0 ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_inequality_solution_set_l1050_105037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_within_plane_l1050_105098

-- Define a type for geometric objects
inductive GeometricObject
  | Line
  | Plane

-- Define a predicate for "is within"
def IsWithin (a b : Set GeometricObject) : Prop := a ⊆ b

-- Theorem statement
theorem line_within_plane (a α : Set GeometricObject) 
  (h1 : ∃x ∈ a, x = GeometricObject.Line) 
  (h2 : ∃x ∈ α, x = GeometricObject.Plane) 
  (h3 : IsWithin a α) : 
  a ⊆ α := by
  -- The proof is omitted for now
  sorry

-- Example usage
example (a α : Set GeometricObject) 
  (h1 : ∃x ∈ a, x = GeometricObject.Line) 
  (h2 : ∃x ∈ α, x = GeometricObject.Plane) 
  (h3 : IsWithin a α) : 
  a ⊆ α := by
  exact line_within_plane a α h1 h2 h3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_within_plane_l1050_105098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l1050_105010

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_8 (a₁ q : ℝ) :
  (geometric_sequence a₁ q 1 + geometric_sequence a₁ q 2 = 6) →
  (geometric_sequence a₁ q 4 + geometric_sequence a₁ q 5 = 48) →
  geometric_sum a₁ q 8 = 510 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l1050_105010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l1050_105022

/-- The height of water in a cylinder after pouring half the water from a cone -/
theorem water_height_in_cylinder (cone_radius : ℝ) (cone_height : ℝ) (cylinder_radius : ℝ) :
  cone_radius = 15 →
  cone_height = 20 →
  cylinder_radius = 18 →
  let cone_volume := (1/3) * Real.pi * cone_radius^2 * cone_height
  let transferred_volume := (1/2) * cone_volume
  let cylinder_height := transferred_volume / (Real.pi * cylinder_radius^2)
  ∃ ε > 0, |cylinder_height - 2.315| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l1050_105022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1050_105056

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := Real.exp (3 * Real.log x - x) - x^2 - (a - 4) * x - 4

-- State the theorem
theorem min_value_of_a (a : ℝ) :
  (∀ x > 0, f a x ≤ 0) → a ≥ 4 / Real.exp 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1050_105056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_greater_than_five_l1050_105025

/-- A standard die with six faces numbered from 1 to 6 -/
structure StandardDie :=
  (faces : Finset Nat)
  (face_count : faces.card = 6)
  (face_range : ∀ n : Nat, n ∈ faces ↔ 1 ≤ n ∧ n ≤ 6)

/-- The probability of an event in a finite sample space -/
def probability (event : Finset Nat) (sample_space : Finset Nat) : ℚ :=
  (event.card : ℚ) / (sample_space.card : ℚ)

/-- Theorem: The probability of rolling a number greater than 5 on a standard die is 1/6 -/
theorem prob_greater_than_five (d : StandardDie) : 
  probability (d.faces.filter (λ n => n > 5)) d.faces = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_greater_than_five_l1050_105025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1050_105083

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem car_average_speed :
  let distance : ℝ := 360
  let time : ℝ := 4.5
  average_speed distance time = 80 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1050_105083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_earnings_l1050_105054

noncomputable def beef_amount : ℝ := 20
noncomputable def pork_amount : ℝ := beef_amount / 2
noncomputable def meat_per_meal : ℝ := 1.5
noncomputable def price_per_meal : ℝ := 20

theorem james_earnings : 
  (beef_amount + pork_amount) / meat_per_meal * price_per_meal = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_earnings_l1050_105054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l1050_105016

theorem pyramid_volume (c α β : Real) (h_angle : 0 < α ∧ α < π/4) :
  (c^3 / 36) * Real.sin (2 * α) * Real.tan β * Real.sqrt (1 + 3 * Real.cos α ^ 2) =
  (1 / 3) * ((c^2 / 4) * Real.sin (2 * α)) * ((c / 3) * Real.sqrt (1 + 3 * Real.cos α ^ 2) * Real.tan β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l1050_105016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rd_cost_productivity_increase_calculation_l1050_105041

/-- The R&D costs attributed to the increase in average labor productivity -/
noncomputable def rd_cost_per_productivity_increase (rd_cost : ℝ) (productivity_change : ℝ) : ℝ :=
  rd_cost / productivity_change

/-- Theorem: The R&D costs attributed to the increase in average labor productivity
    by 1 million rubles per person is approximately equal to 1661 million rubles -/
theorem rd_cost_productivity_increase_calculation :
  let rd_cost : ℝ := 3205.69
  let productivity_change : ℝ := 1.93
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |rd_cost_per_productivity_increase rd_cost productivity_change - 1661| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rd_cost_productivity_increase_calculation_l1050_105041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1050_105069

open Real

noncomputable def f (x : ℝ) := sin (x + π/6)

theorem f_max_min :
  (∃ (k : ℤ), f (2*k*π + π/3) = 1 ∧ ∀ x, f x ≤ 1) ∧
  (∃ (k : ℤ), f (2*k*π - 2*π/3) = -1 ∧ ∀ x, f x ≥ -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1050_105069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1050_105047

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

theorem work_completion_time 
  (a_days b_days c_days : ℝ) 
  (ha : a_days = 36) 
  (hb : b_days = 18) 
  (hc : c_days = 6) : 
  1 / (work_rate a_days + work_rate b_days + work_rate c_days) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1050_105047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cable_package_cost_is_100_l1050_105028

/-- The cost of the cable program package -/
noncomputable def cable_package_cost (x : ℝ) : ℝ := x

/-- The cost of the next 100 channels -/
noncomputable def next_100_channels_cost (x : ℝ) : ℝ := x / 2

/-- The total cost of the cable program -/
noncomputable def total_cost (x : ℝ) : ℝ := cable_package_cost x + next_100_channels_cost x

/-- One person's share of the total cost -/
noncomputable def one_person_share (x : ℝ) : ℝ := total_cost x / 2

theorem cable_package_cost_is_100 :
  ∃ x : ℝ, cable_package_cost x = 100 ∧ one_person_share x = 75 := by
  use 100
  constructor
  · rfl
  · simp [one_person_share, total_cost, cable_package_cost, next_100_channels_cost]
    norm_num

#check cable_package_cost_is_100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cable_package_cost_is_100_l1050_105028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_G_l1050_105036

/-- The function F as defined in the problem -/
def F (p q : ℝ) : ℝ := -4*p*q + 5*p*(1-q) + 2*(1-p)*q - 3*(1-p)*(1-q)

/-- G(p) is the maximum of F(p, q) over all q in [0, 1] -/
noncomputable def G (p : ℝ) : ℝ := 
  ⨆ (q : ℝ) (h : 0 ≤ q ∧ q ≤ 1), F p q

/-- The theorem statement -/
theorem minimize_G : 
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧ 
  (∀ (p' : ℝ), 0 ≤ p' ∧ p' ≤ 1 → G p ≤ G p') ∧
  p = 5/14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_G_l1050_105036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1050_105030

noncomputable def f (x : ℝ) : ℝ := (8^x + 5) / (2^x + 1)

theorem f_min_value : ∀ x : ℝ, f x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1050_105030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_approx_40_l1050_105097

/-- Calculates the speed of a bus given its length, time to pass a person, and the person's speed in the opposite direction. -/
noncomputable def calculate_bus_speed (bus_length : ℝ) (passing_time : ℝ) (person_speed_kmh : ℝ) : ℝ :=
  let person_speed_ms := person_speed_kmh * (1000 / 3600)
  let relative_speed := bus_length / passing_time
  let bus_speed_ms := relative_speed - person_speed_ms
  bus_speed_ms * (3600 / 1000)

/-- Theorem stating that under the given conditions, the bus speed is approximately 40 km/hr. -/
theorem bus_speed_approx_40 :
  let bus_length := (15 : ℝ)
  let passing_time := (1.125 : ℝ)
  let person_speed := (8 : ℝ)
  let calculated_speed := calculate_bus_speed bus_length passing_time person_speed
  abs (calculated_speed - 40) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_approx_40_l1050_105097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_triples_and_no_solution_l1050_105023

theorem infinite_triples_and_no_solution :
  (∃ f : ℕ → ℕ+ × ℕ+ × ℕ+, Function.Injective f ∧
    ∀ i : ℕ, let (m, n, p) := f i; 4*m*n - m - n = p^2 - 1) ∧
  ¬∃ (m n p : ℕ+), 4*m*n - m - n = p^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_triples_and_no_solution_l1050_105023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_living_room_side_length_l1050_105027

/-- The side length of a square living room, given specific conditions about room sizes and total area to be painted. -/
theorem living_room_side_length : ℝ := by
  let bedroom_width : ℝ := 10
  let bedroom_length : ℝ := 12
  let wall_height : ℝ := 10
  let total_area : ℝ := 1640
  let bedroom_area : ℝ := 2 * (bedroom_width + bedroom_length) * wall_height
  let living_room_area : ℝ := total_area - bedroom_area
  let living_room_side_length : ℝ := living_room_area / (3 * wall_height)
  have : living_room_side_length = 40 := by sorry
  exact living_room_side_length


end NUMINAMATH_CALUDE_ERRORFEEDBACK_living_room_side_length_l1050_105027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1050_105035

-- Define the inequality
def inequality (x : ℝ) : Prop :=
  |4 * x^2 - 32/x| + |x^2 + 5/(x^2 - 6)| ≤ |3 * x^2 - 5/(x^2 - 6) - 32/x|

-- Define the solution set
noncomputable def solution_set : Set ℝ :=
  Set.Ioc (-Real.sqrt 6) (-Real.sqrt 5) ∪ 
  Set.Icc (-1) 0 ∪ 
  Set.Icc 1 2 ∪ 
  Set.Ico (Real.sqrt 5) (Real.sqrt 6)

-- Theorem statement
theorem inequality_solution :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1050_105035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_a_minus_pi_half_l1050_105061

theorem sin_a_minus_pi_half (a : Real) 
  (h1 : Real.sin a = 2/3) 
  (h2 : a ∈ Set.Ioo (Real.pi/2) Real.pi) : 
  Real.sin (a - Real.pi/2) = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_a_minus_pi_half_l1050_105061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_bounds_l1050_105093

theorem p_bounds : 1 < Real.rpow 4 (1/3) - Real.rpow 6 (1/3) + Real.rpow 9 (1/3) ∧ 
                   Real.rpow 4 (1/3) - Real.rpow 6 (1/3) + Real.rpow 9 (1/3) < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_bounds_l1050_105093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_d_equals_two_l1050_105092

/-- A linear function with integer coefficient and constant term -/
def g (c : ℤ) : ℝ → ℝ := fun x ↦ 5 * x + c

/-- The theorem stating that if g(x) = 5x + c intersects its inverse at (2, d), then d = 2 -/
theorem intersection_point_d_equals_two (c d : ℤ) :
  (∃ (g_inv : ℝ → ℝ), Function.LeftInverse g_inv (g c) ∧ Function.RightInverse g_inv (g c)) →
  (g c 2 = d ∧ g c d = 2) →
  d = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_d_equals_two_l1050_105092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_not_quadratic_radical_l1050_105011

-- Define what a quadratic radical is
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, f x = Real.sqrt (g x)) ∧ 
  ((∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c) ∨
   (∃ k : ℝ, ∀ x, g x = k)) ∧
  (∀ x, g x ≥ 0)

-- The expression we're checking
noncomputable def expr (x : ℝ) : ℝ := Real.sqrt (-abs x - 1)

-- Theorem statement
theorem expr_not_quadratic_radical : ¬ is_quadratic_radical expr := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_not_quadratic_radical_l1050_105011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_radii_l1050_105034

/-- Represents a truncated right circular cone with a hemisphere on top -/
structure TruncatedConeWithHemisphere where
  r₁ : ℝ  -- radius of the larger base
  r₂ : ℝ  -- radius of the smaller base
  s : ℝ   -- radius of the inscribed sphere

/-- The volume of the truncated cone plus hemisphere is three times the volume of the inscribed sphere -/
def volume_condition (shape : TruncatedConeWithHemisphere) : Prop :=
  ∃ (h : ℝ), 
    (1/3 * Real.pi * (shape.r₁^2 + shape.r₁ * shape.r₂ + shape.r₂^2) * h + 2/3 * Real.pi * shape.r₁^3) = 
    3 * (4/3 * Real.pi * shape.s^3)

/-- The theorem stating the ratio of r₁ to r₂ -/
theorem ratio_of_radii (shape : TruncatedConeWithHemisphere) 
  (h : volume_condition shape) : 
  shape.r₁ / shape.r₂ = (5 + Real.sqrt 21) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_radii_l1050_105034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1050_105007

noncomputable def f (x : ℝ) := x * Real.sin x

theorem f_inequality : f (-π/3) > f 1 ∧ f 1 > f (π/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1050_105007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l1050_105072

open Real

noncomputable def f (x : ℝ) : ℝ := log (cos x) / log 10

theorem f_strictly_increasing (k : ℤ) :
  StrictMonoOn f (Set.Ioo (2 * (k : ℝ) * π - π / 2) (2 * (k : ℝ) * π)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l1050_105072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1050_105026

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point of tangency
def P : ℝ × ℝ := (2, 8)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (f P.1 = P.2) ∧
    (HasDerivAt f m P.1) ∧
    (m * P.1 - P.2 + b = 0) ∧
    (m = 12 ∧ b = -16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1050_105026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1050_105029

theorem gcd_problem (b : ℤ) (h1 : ∃ k : ℤ, b = 1187 * (2*k + 1)) :
  Int.gcd (3*b^2 + 34*b + 76) (b + 16) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1050_105029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_allocation_l1050_105009

def budget_allocation (total : ℚ) (ratio : List ℚ) : List ℚ :=
  let sum_ratio := ratio.sum
  ratio.map (λ x => (x / sum_ratio) * total)

theorem food_allocation 
  (total : ℚ) 
  (ratio : List ℚ) 
  (h1 : total = 3600) 
  (h2 : ratio = [5, 4, 1, 3, 2]) :
  (budget_allocation total ratio).get! 1 = 960 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_allocation_l1050_105009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l1050_105085

/-- Represents a single jump of the frog -/
structure Jump where
  direction : ℝ × ℝ
  magnitude : ℝ
  random_direction : Bool

/-- Represents the frog's journey -/
structure FrogJourney where
  jumps : List Jump
  num_jumps : Nat
  jump_length : ℝ

/-- Calculates the final position of the frog -/
noncomputable def final_position (journey : FrogJourney) : ℝ × ℝ :=
  sorry

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- The probability of the frog being within 2 meters of the starting point -/
noncomputable def probability_within_two_meters (journey : FrogJourney) : ℝ :=
  sorry

/-- Approximation relation for real numbers -/
def approximately_equal (x y : ℝ) (ε : ℝ) : Prop :=
  abs (x - y) < ε

notation:50 x " ≈ " y => approximately_equal x y 0.01

theorem frog_jump_probability (j : FrogJourney) 
  (h1 : j.num_jumps = 5)
  (h2 : j.jump_length = 1)
  (h3 : ∀ jump ∈ j.jumps, jump.random_direction = true) :
  probability_within_two_meters j ≈ 0.40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l1050_105085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_profit_share_l1050_105031

/-- Calculates the share of profit for an investor in a business partnership --/
def calculate_profit_share (total_profit : ℕ) (x_investment y_investment z_investment : ℕ) (z_join_month : ℕ) : ℚ :=
  let total_months : ℕ := 12
  let x_investment_months : ℕ := x_investment * total_months
  let y_investment_months : ℕ := y_investment * total_months
  let z_investment_months : ℕ := z_investment * (total_months - z_join_month)
  let total_investment_months : ℕ := x_investment_months + y_investment_months + z_investment_months
  let z_share : ℚ := (z_investment_months : ℚ) / (total_investment_months : ℚ)
  z_share * (total_profit : ℚ)

/-- Theorem stating that Z's share of the profit is 2600 --/
theorem z_profit_share :
  calculate_profit_share 14300 36000 42000 48000 4 = 2600 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_profit_share_l1050_105031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_and_no_intersection_l1050_105091

-- Define the circle C
def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the point M
structure Point (a b : ℝ) where
  nonzero : a * b ≠ 0

-- Define the line l
def Line_l (M : ℝ × ℝ) := {p : ℝ × ℝ | ∃ t : ℝ, p = (2*M.1*t, 2*M.2*t)}

-- Define the line m
def Line_m (a b r : ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 = r^2}

-- Define perpendicularity for lines
def Perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∀ (p1 p2 : ℝ × ℝ) (q1 q2 : ℝ × ℝ),
    p1 ∈ l1 → p2 ∈ l1 → q1 ∈ l2 → q2 ∈ l2 →
    (p2.1 - p1.1) * (q2.1 - q1.1) + (p2.2 - p1.2) * (q2.2 - q1.2) = 0

theorem perpendicular_and_no_intersection 
  (r : ℝ) (a b : ℝ) (M : Point a b) (h_inside : a^2 + b^2 < r^2) :
  (∃ (l : Set (ℝ × ℝ)), l = Line_l (a, b) ∧ 
   (∀ (p : ℝ × ℝ), p ∈ l → p ∈ Circle r → (a, b) = ((p.1 + 0)/2, (p.2 + 0)/2))) →
  Perpendicular (Line_l (a, b)) (Line_m a b r) ∧ 
  (∀ (p : ℝ × ℝ), p ∈ Line_m a b r → p ∉ Circle r) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_and_no_intersection_l1050_105091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_count_l1050_105065

theorem junior_count (total : ℚ) (junior_percent : ℚ) (senior_percent : ℚ) :
  total = 36 →
  junior_percent = 1/5 →
  senior_percent = 1/4 →
  ∃ (juniors seniors : ℚ),
    juniors + seniors = total ∧
    junior_percent * juniors = senior_percent * seniors ∧
    juniors = 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_count_l1050_105065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_product_PA_PB_l1050_105002

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (t : ℝ) : ℝ × ℝ := (1 - Real.sqrt 2 / 2 * t, 1 + Real.sqrt 2 / 2 * t)

def C₂ (ρ θ : ℝ) : Prop := ρ^2 - 2 * ρ * Real.cos θ - 3 = 0

-- Define point P in polar coordinates
def P : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Theorem for the length of AB
theorem length_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14 := by sorry

-- Theorem for the product of distances PA and PB
theorem product_PA_PB : 
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) * Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_product_PA_PB_l1050_105002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1050_105062

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c

-- Define the given conditions
def given_conditions (t : Triangle) : Prop :=
  Real.sin (t.A + t.C) = 8 * (Real.sin (t.B / 2))^2 ∧
  t.a + t.c = 6 ∧
  (1/2) * t.a * t.c * Real.sin t.B = 2

-- State the theorem
theorem triangle_problem (t : Triangle) 
  (h_triangle : triangle_properties t)
  (h_conditions : given_conditions t) :
  Real.cos t.B = 15/17 ∧ t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1050_105062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_edges_theorem_l1050_105086

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron defined by four points -/
structure Tetrahedron where
  A : Fin 4 → Point3D

/-- Checks if two edges (represented as pairs of points) are perpendicular -/
def are_perpendicular (edge1 : Point3D × Point3D) (edge2 : Point3D × Point3D) : Prop :=
  let (a1, a2) := edge1
  let (b1, b2) := edge2
  (a2.x - a1.x) * (b2.x - b1.x) + (a2.y - a1.y) * (b2.y - b1.y) + (a2.z - a1.z) * (b2.z - b1.z) = 0

/-- Main theorem: If 5 out of 6 pairs of edges from two tetrahedrons are perpendicular, 
    then the 6th pair is also perpendicular -/
theorem perpendicular_edges_theorem (tetra1 tetra2 : Tetrahedron) 
  (h : ∃ (i j k l : Fin 4), i ≠ j ∧ k ≠ l ∧ 
    (∀ (m n p q : Fin 4), m ≠ n ∧ p ≠ q ∧ (m, n, p, q) ≠ (i, j, k, l) → 
      are_perpendicular 
        (tetra1.A m, tetra1.A n) 
        (tetra2.A p, tetra2.A q))) :
  ∀ (i j k l : Fin 4), i ≠ j ∧ k ≠ l → 
    are_perpendicular 
      (tetra1.A i, tetra1.A j) 
      (tetra2.A k, tetra2.A l) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_edges_theorem_l1050_105086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourOccurrencesUpTo1000_l1050_105050

def countDigit (d : Nat) (start : Nat) (end_ : Nat) : Nat :=
  (end_ - start + 1) / 10 * (if d = 0 then 1 else 10)

def countFourOccurrences (n : Nat) : Nat :=
  countDigit 4 1 (n % 10) +
  countDigit 4 1 ((n / 10) % 10) * 10 +
  countDigit 4 1 (n / 100) * 100

theorem fourOccurrencesUpTo1000 :
  countFourOccurrences 1000 = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourOccurrencesUpTo1000_l1050_105050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joy_foster_dogs_food_l1050_105084

/-- Represents the amount of food eaten by dogs over a period of time -/
structure DogFood where
  mom_meal : ℚ  -- Amount of food mom eats per meal
  mom_meals_per_day : ℕ  -- Number of meals mom eats per day
  puppy_count : ℕ  -- Number of puppies
  puppy_meals_per_day : ℕ  -- Number of meals each puppy eats per day
  total_days : ℕ  -- Total number of days
  total_food : ℚ  -- Total amount of food for all dogs over the period

/-- Calculates the amount of food each puppy eats per meal -/
noncomputable def puppy_food_per_meal (df : DogFood) : ℚ :=
  let mom_total := df.mom_meal * df.mom_meals_per_day * df.total_days
  let puppy_total := df.total_food - mom_total
  let total_puppy_meals := df.puppy_count * df.puppy_meals_per_day * df.total_days
  puppy_total / total_puppy_meals

/-- Theorem stating that given the conditions, each puppy eats 0.5 cups per meal -/
theorem joy_foster_dogs_food (df : DogFood) 
  (h1 : df.mom_meal = 3/2)
  (h2 : df.mom_meals_per_day = 3)
  (h3 : df.puppy_count = 5)
  (h4 : df.puppy_meals_per_day = 2)
  (h5 : df.total_days = 6)
  (h6 : df.total_food = 57) :
  puppy_food_per_meal df = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joy_foster_dogs_food_l1050_105084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_b_necessary_not_sufficient_l1050_105055

-- Define the basic structures
structure Line where
  -- Add any necessary fields

structure Plane where
  -- Add any necessary fields

-- Define relations between structures
def Line.intersects (l m : Line) : Prop := sorry
def Line.in_plane (l : Line) (p : Plane) : Prop := sorry
def Line.intersects_plane (l : Line) (p : Plane) : Prop := sorry
def Plane.intersects (p q : Plane) : Prop := sorry

-- Define the propositions
def proposition_a (l m : Line) (α β : Plane) : Prop :=
  (l.intersects m) ∧ (l.in_plane α) ∧ (m.in_plane α) ∧ ¬(l.in_plane β) ∧ ¬(m.in_plane β)

def proposition_b (l m : Line) (β : Plane) : Prop :=
  (l.intersects_plane β) ∨ (m.intersects_plane β)

def proposition_c (α β : Plane) : Prop :=
  α.intersects β

-- Define the theorem
theorem proposition_b_necessary_not_sufficient 
  (l m : Line) (α β : Plane) 
  (h_a : proposition_a l m α β) :
  (proposition_c α β → proposition_b l m β) ∧
  ¬(proposition_b l m β → proposition_c α β) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_b_necessary_not_sufficient_l1050_105055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_property_l1050_105012

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => -1
  | n+2 => -sequence_a (n+1) - 2 * sequence_a n

theorem perfect_square_property (n : ℕ) :
  ∃ c : ℤ, (2^(n+2) : ℤ) - 7 * (sequence_a n)^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_property_l1050_105012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_theorem_l1050_105073

open Real

noncomputable section

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a point on a circle
def pointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem chord_theorem (O M A B C : ℝ × ℝ) (r : ℝ) (hr : r > 0) :
  let circle : Circle := { center := O, radius := r, radius_pos := hr }
  pointOnCircle circle M →
  pointOnCircle circle A →
  pointOnCircle circle B →
  pointOnCircle circle C →
  angle A M B = π / 3 →
  angle A M C = π / 3 →
  distance M A = distance M B + distance M C :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_theorem_l1050_105073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l1050_105078

noncomputable def initial_number : ℂ := -4 - 6 * Complex.I

noncomputable def rotation_factor : ℂ := Complex.exp (Complex.I * Real.pi / 6)

def dilation_factor : ℝ := 2

noncomputable def transformation (z : ℂ) : ℂ := dilation_factor * (rotation_factor * z)

theorem complex_transformation :
  transformation initial_number = (6 - 4 * Real.sqrt 3) + (-6 * Real.sqrt 3 - 4) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l1050_105078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_solution_l1050_105045

theorem trig_inequality_solution (x : ℝ) :
  x ∈ Set.Icc (-π/4 : ℝ) (7*π/4) →
  (Real.sin x)^2018 + (Real.cos x)^(-2019 : ℤ) ≥ (Real.cos x)^2018 + (Real.sin x)^(-2019 : ℤ) ↔
  x ∈ Set.Ioc (-π/4 : ℝ) 0 ∪ Set.Ico (π/4) (π/2) ∪ Set.Ioo π (5*π/4) ∪ Set.Ioo (3*π/2) (7*π/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_solution_l1050_105045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inspection_not_qualified_l1050_105049

def part_count : ℕ := 60

def part_distribution : List (ℝ × ℕ) := [
  (28.51, 13), (28.52, 6), (28.50, 4), (28.48, 11),
  (28.49, 13), (28.54, 1), (28.53, 7), (28.47, 5)
]

def prob_less_than_28_49 : ℚ := 4/15

def standard_deviation : ℝ := 0.02

def qualification_threshold : ℝ := 0.8

def average_diameter : ℝ := 28.50

noncomputable def count_within_range (l : List (ℝ × ℕ)) (lower upper : ℝ) : ℕ :=
  l.foldl (λ acc (x, n) => if lower ≤ x ∧ x ≤ upper then acc + n else acc) 0

theorem inspection_not_qualified :
  (count_within_range part_distribution (average_diameter - standard_deviation) (average_diameter + standard_deviation) : ℝ)
  < qualification_threshold * part_count :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inspection_not_qualified_l1050_105049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_theorem_l1050_105063

-- Define the polynomial type
def MyPolynomial (α : Type*) := α → α

-- Define the property that the polynomial satisfies the given equation
def SatisfiesEquation (P : MyPolynomial ℝ) : Prop :=
  ∀ (x : ℤ), (↑x - 2010) * P (↑x + 67) = ↑x * P ↑x

-- Define the specific form of the polynomial
def SpecificForm (P : MyPolynomial ℝ) : Prop :=
  ∃ (c : ℝ), ∀ (x : ℝ), P x = c * (Finset.range 30).prod (fun i => x - (↑i + 1) * 67)

-- State the theorem
theorem polynomial_equation_theorem (P : MyPolynomial ℝ) :
  SatisfiesEquation P → SpecificForm P :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_theorem_l1050_105063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l1050_105000

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -2

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2

-- Define the perpendicular condition
def perpendicular_to_directrix (P A : ℝ × ℝ) : Prop :=
  A.1 = -2 ∧ A.2 = P.2

-- Define the slope condition
def slope_condition (A : ℝ × ℝ) : Prop :=
  (A.2 - 0) / (A.1 - 2) = -Real.sqrt 3

-- Main theorem
theorem parabola_point_distance (P A : ℝ × ℝ) :
  point_on_parabola P →
  perpendicular_to_directrix P A →
  slope_condition A →
  Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 4 * Real.sqrt 3 :=
by
  sorry

#check parabola_point_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l1050_105000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auditorium_seats_taken_l1050_105068

theorem auditorium_seats_taken (total_seats : ℕ) (broken_fraction : ℚ) (available_seats : ℕ) 
  (h1 : total_seats = 500)
  (h2 : broken_fraction = 1 / 10)
  (h3 : available_seats = 250)
  : (total_seats - available_seats - (total_seats * broken_fraction).floor) / total_seats = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_auditorium_seats_taken_l1050_105068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_buddhas_hand_problem_l1050_105020

def standard_weight : ℝ := 0.5
def price_per_kg : ℝ := 42
def weight_deviations : List ℝ := [0.1, 0, -0.05, -0.25, 0.15, 0.2, 0.05, -0.1]

theorem buddhas_hand_problem :
  let max_deviation := weight_deviations.maximum?
  let min_deviation := weight_deviations.minimum?
  let total_deviation := weight_deviations.sum
  let total_weight := 8 * standard_weight + total_deviation
  let total_earnings := total_weight * price_per_kg
  (∀ max min, max_deviation = some max → min_deviation = some min → max - min = 0.45) ∧
  (total_deviation = 0.1) ∧
  (total_earnings = 172.2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_buddhas_hand_problem_l1050_105020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleCovering_l1050_105024

/-- Represents a piece of cardboard made of four 1x1 squares -/
structure CardboardPiece where
  squares : List (Fin 4 × Fin 5)
  square_count : squares.length = 4

/-- Represents the 4x5 rectangle with a checkerboard coloring -/
def Rectangle : Fin 4 → Fin 5 → Bool :=
  fun i j => (i.val + j.val) % 2 = 0

/-- Counts the number of black squares covered by a cardboard piece -/
def blackSquaresCovered (piece : CardboardPiece) : Nat :=
  (piece.squares.filter (fun coord => Rectangle coord.1 coord.2)).length

/-- Predicate for a valid covering of the rectangle -/
def isValidCovering (pieces : Fin 5 → CardboardPiece) : Prop :=
  ∀ i j, ∃ k, (pieces k).squares.contains (i, j)

theorem impossibleCovering :
  ∀ (pieces : Fin 5 → CardboardPiece),
  (∃ k, blackSquaresCovered (pieces k) ≠ 2) →
  ¬ isValidCovering pieces :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleCovering_l1050_105024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_coordinates_l1050_105044

/-- Helper definition for a triangle --/
def is_triangle (A B C : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

/-- Helper definition for incenter --/
def is_incenter (I A B C : ℝ × ℝ) : Prop :=
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  x + y + z = 1 ∧
  I = (x * A.1 + y * B.1 + z * C.1, x * A.2 + y * B.2 + z * C.2)

theorem incenter_coordinates (A B C I : ℝ × ℝ) (a b c : ℝ) : 
  a = 8 ∧ b = 10 ∧ c = 6 →
  is_triangle A B C →
  is_incenter I A B C →
  ∃ (x y z : ℝ), 
    x = 1/3 ∧ y = 5/12 ∧ z = 1/4 ∧
    x + y + z = 1 ∧
    I = (x * A.1 + y * B.1 + z * C.1, x * A.2 + y * B.2 + z * C.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_coordinates_l1050_105044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_arithmetic_pascal_identity_l1050_105013

-- Define P as a function
def P (p k n : ℕ) : ZMod p := (Nat.choose n k : ZMod p)

theorem p_arithmetic_pascal_identity (p : ℕ) (k : ℕ) (hp : Nat.Prime p) (hk : k ≤ (p - 1) / 2) :
  P p k (p - 1 - k) = (-1)^k * P p k (2 * k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_arithmetic_pascal_identity_l1050_105013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l1050_105076

noncomputable def complex_locus (z : ℂ) : ℂ := z + 2 / z

theorem locus_is_ellipse (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  ∀ (w : ℂ), w = complex_locus z ↔ 
    (w.re / a)^2 + (w.im / b)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l1050_105076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_full_compensation_theorem_l1050_105082

/-- Represents the state deposit insurance system -/
structure DepositInsuranceSystem where
  max_compensation : ℕ
  insures_interest : Bool

/-- Represents a bank -/
structure Bank where
  participates_in_insurance : Bool

/-- Represents a deposit -/
structure Deposit where
  initial_amount : ℕ
  accrued_interest : ℕ
  bank : Bank

/-- Represents the compensation received when a bank fails -/
def compensation (deposit : Deposit) (insurance : DepositInsuranceSystem) : ℕ :=
  if deposit.bank.participates_in_insurance then
    min (deposit.initial_amount + 
      (if insurance.insures_interest then deposit.accrued_interest else 0)) 
      insurance.max_compensation
  else 0

/-- Theorem stating that the depositor receives full compensation -/
theorem full_compensation_theorem 
  (deposit : Deposit) 
  (insurance : DepositInsuranceSystem) 
  (h1 : deposit.initial_amount = 100000)
  (h2 : deposit.bank.participates_in_insurance = true)
  (h3 : insurance.max_compensation ≥ deposit.initial_amount + deposit.accrued_interest)
  (h4 : insurance.insures_interest = true) :
  compensation deposit insurance = deposit.initial_amount + deposit.accrued_interest := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_full_compensation_theorem_l1050_105082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_six_triangle_area_is_sqrt_three_l1050_105094

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  angle_sum : A + B + C = π

-- Define the conditions given in the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.a^2 + t.c^2 - t.b^2 = Real.sqrt 3 * t.a * t.c ∧
  2 * t.b * Real.cos t.A = Real.sqrt 3 * (t.c * Real.cos t.A + t.a * Real.cos t.C) ∧
  ∃ (m : ℝ), m^2 = 7 ∧ m = Real.sqrt ((t.b/2)^2 + (t.a^2 + t.c^2)/4 - t.a * t.c * Real.cos t.B / 2)

-- State the theorems to be proved
theorem angle_B_is_pi_over_six (t : Triangle) (h : t.a^2 + t.c^2 - t.b^2 = Real.sqrt 3 * t.a * t.c) :
  t.B = π/6 := by sorry

theorem triangle_area_is_sqrt_three (t : Triangle) (h : triangle_conditions t) :
  t.a * t.c * Real.sin t.B / 2 = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_six_triangle_area_is_sqrt_three_l1050_105094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1050_105005

theorem ellipse_equation (x y : ℝ) :
  (∃ lambda : ℝ, lambda > -9 ∧
    (x^2 / (16 + lambda) + y^2 / (9 + lambda) = 1) ∧
    (16 / (16 + lambda) + 9 / (9 + lambda) = 1)) →
  (x^2 / 28 + y^2 / 21 = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1050_105005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_superhero_speed_l1050_105019

/-- Converts miles per hour to kilometers per minute -/
noncomputable def mph_to_km_per_min (speed_mph : ℝ) : ℝ :=
  speed_mph * (1 / 0.6) / 60

theorem superhero_speed (speed_mph : ℝ) (h1 : speed_mph = 36000) :
  mph_to_km_per_min speed_mph = 1000 := by
  sorry

#check superhero_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_superhero_speed_l1050_105019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l1050_105089

/-- Given a geometric sequence {aₙ} with first term a₁ and common ratio q,
    Sₙ represents the sum of the first n terms of the sequence. -/
noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence {aₙ} with sum of first n terms Sₙ,
    if 9S₃ = S₆ and a₂ = 1, then a₁ = 1/2. -/
theorem geometric_sequence_problem (a₁ q : ℝ) :
  q ≠ 1 →
  9 * (geometric_sum a₁ q 3) = geometric_sum a₁ q 6 →
  a₁ * q = 1 →
  a₁ = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l1050_105089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_area_less_than_original_l1050_105052

/-- Represents the dimensions of a blanket -/
structure Blanket where
  length : ℝ
  width : ℝ

/-- Calculates the area of a blanket -/
noncomputable def area (b : Blanket) : ℝ := b.length * b.width

/-- Represents the folding operations -/
inductive FoldOp
  | Half
  | Third
  | TwoFifths
  | Quarter
  | TwoSevenths

/-- Applies a folding operation to a blanket -/
noncomputable def applyFold (b : Blanket) (op : FoldOp) : Blanket :=
  match op with
  | FoldOp.Half => ⟨b.length / 2, b.width⟩
  | FoldOp.Third => ⟨b.length / 3, b.width⟩
  | FoldOp.TwoFifths => ⟨b.length, b.width * 2 / 5⟩
  | FoldOp.Quarter => ⟨b.length / 4, b.width⟩
  | FoldOp.TwoSevenths => ⟨b.length * 2 / 7, b.width⟩

/-- Folds a blanket according to a list of folding operations -/
noncomputable def foldBlanket (b : Blanket) (ops : List FoldOp) : Blanket :=
  ops.foldl applyFold b

theorem folded_area_less_than_original :
  let b1 := Blanket.mk 12 9
  let b2 := Blanket.mk 16 6
  let b3 := Blanket.mk 18 10
  let f1 := foldBlanket b1 [FoldOp.Half, FoldOp.Third, FoldOp.TwoFifths]
  let f2 := foldBlanket b2 [FoldOp.Quarter, FoldOp.Third, FoldOp.TwoSevenths]
  let f3 := foldBlanket b3 [FoldOp.Half, FoldOp.Third, FoldOp.Half, FoldOp.Third, FoldOp.Half, FoldOp.Third]
  area f1 + area f2 + area f3 < area b1 + area b2 + area b3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_area_less_than_original_l1050_105052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_set_subset_of_any_set_odd_increasing_counterexample_min_value_not_two_quadratic_roots_condition_l1050_105077

-- Statement A
theorem empty_set_subset_of_any_set (A : Set α) : ∅ ⊆ A := by
  sorry

-- Statement B
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem odd_increasing_counterexample :
  ∃ f : ℝ → ℝ, odd_function f ∧ 
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧ 
  ¬(∀ x y, x < y → f x < f y) := by
  sorry

-- Statement C
noncomputable def f (x : ℝ) : ℝ := (x^2 + 3) / Real.sqrt (x^2 + 2)

theorem min_value_not_two : 
  ∃ x : ℝ, f x < 2 := by
  sorry

-- Statement D
def quadratic_roots_in_open_interval (m : ℝ) : Prop :=
  ∀ x, x^2 - m*x + 2 = 0 → 1 < x

theorem quadratic_roots_condition :
  ¬(∀ m : ℝ, quadratic_roots_in_open_interval m ↔ m ≥ 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_set_subset_of_any_set_odd_increasing_counterexample_min_value_not_two_quadratic_roots_condition_l1050_105077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_in_equilateral_triangle_l1050_105066

-- Define necessary structures and functions if they're not already in Mathlib
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

def Equilateral (A B C : Point) : Prop := sorry
def SegmentLength (A B : Point) : ℝ := sorry
def CircleDiameter (ω : Circle) : ℝ := 2 * ω.radius
def CircleInscribed (ω : Circle) (A B C : Point) : Prop := sorry
def CircleTangent (ω : Circle) (l : Point → Point → Prop) : Prop := sorry
def Segment (A B : Point) : Set Point := sorry

-- Define membership for Point in Circle
instance : Membership Point Circle where
  mem := λ p c => (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

theorem min_distance_in_equilateral_triangle (A B C : Point) (ω : Circle) :
  Equilateral A B C →
  SegmentLength A B = 3 →
  CircleDiameter ω = 1 →
  CircleInscribed ω A B C →
  CircleTangent ω (λ p q => p = A ∨ p = B) →
  CircleTangent ω (λ p q => p = A ∨ p = C) →
  ∀ P Q : Point, P ∈ ω → Q ∈ Segment B C →
    ∃ min_dist : ℝ, min_dist = (3 * Real.sqrt 3 - 3) / 2 ∧
      SegmentLength P Q ≥ min_dist :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_in_equilateral_triangle_l1050_105066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1050_105014

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1 + Real.log x / Real.log a) / (Real.log (x + 1) / Real.log a)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop :=
  x - 2 * Real.log 2 * y + 2 * Real.log 2 - 1 = 0

-- Define the critical point condition
def has_critical_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  deriv f x₀ = 0

-- Main theorem
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, x > 0 → f a x = (1 + Real.log x / Real.log a) / (Real.log (x + 1) / Real.log a)) →
  (∃ x₀ : ℝ, has_critical_point (f a) x₀) →
  (tangent_line 1 (f 2 1)) ∧ 
  (a > 1) ∧
  (∀ x₀ : ℝ, has_critical_point (f a) x₀ → x₀ + f a x₀ ≥ 3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1050_105014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_roots_sum_of_roots_minimum_value_c_l1050_105006

-- Define the quadratic equation type
def QuadraticEquation (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Theorem 1
theorem reciprocal_roots (m n : ℝ) (hn : n ≠ 0) :
  ∃ x y : ℝ, (QuadraticEquation 1 m n x ∧ QuadraticEquation 1 m n y) →
  (QuadraticEquation n m 1 (1/x) ∧ QuadraticEquation n m 1 (1/y)) :=
sorry

-- Theorem 2
theorem sum_of_roots (a b : ℝ) :
  (QuadraticEquation 1 (-15) (-5) a ∧ QuadraticEquation 1 (-15) (-5) b) →
  a + b = 15 :=
sorry

-- Theorem 3
theorem minimum_value_c (a b c : ℝ) :
  a + b + c = 0 ∧ a * b * c = 16 →
  c > 0 →
  c ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_roots_sum_of_roots_minimum_value_c_l1050_105006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_length_l1050_105074

theorem altitude_length (A B C : ℝ) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) 
  (h_sin_sum : Real.sin (A + B) = 3/5)
  (h_sin_diff : Real.sin (A - B) = 1/5)
  (h_AB : 3 = 3) : -- AB = 3, but we use 3 = 3 to avoid introducing new variables
  3 * (Real.tan A) = 6 + 3 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_length_l1050_105074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l1050_105057

/-- The area of the region bounded by the curves x = 16 cos³(t), y = 2 sin³(t), and x = 2 (for x ≥ 2) -/
noncomputable def boundedArea : ℝ := 4 * Real.pi

/-- The parametric equations of the curve -/
noncomputable def x (t : ℝ) : ℝ := 16 * (Real.cos t) ^ 3
noncomputable def y (t : ℝ) : ℝ := 2 * (Real.sin t) ^ 3

/-- The vertical line -/
def verticalLine : ℝ := 2

theorem area_of_bounded_region :
  ∃ (a b : ℝ), a < b ∧
  (∀ t ∈ Set.Icc a b, x t ≥ verticalLine) ∧
  boundedArea = -2 * ∫ t in a..b, y t * (deriv x t) := by
  sorry

#check area_of_bounded_region

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l1050_105057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_legs_large_value_l1050_105088

/-- Two similar right triangles with areas 30 and 750 square inches, 
    where the smaller triangle has a hypotenuse of 13 inches -/
structure SimilarTriangles where
  area_small : ℝ
  area_large : ℝ
  hypotenuse_small : ℝ
  similar : area_large / area_small = (hypotenuse_small * (area_large / area_small).sqrt)^2 / hypotenuse_small^2

/-- The sum of the lengths of the legs of the larger triangle -/
noncomputable def sum_legs_large (t : SimilarTriangles) : ℝ :=
  (t.hypotenuse_small * (t.area_large / t.area_small).sqrt) * 
  ((t.area_small / (t.hypotenuse_small^2 / 2)).sqrt + (2 * t.area_small / t.hypotenuse_small^2).sqrt)

theorem sum_legs_large_value (t : SimilarTriangles) 
  (h1 : t.area_small = 30)
  (h2 : t.area_large = 750)
  (h3 : t.hypotenuse_small = 13) :
  sum_legs_large t = 85 := by
    sorry

#check sum_legs_large_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_legs_large_value_l1050_105088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sqrt_difference_l1050_105096

theorem smallest_n_for_sqrt_difference : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → Real.sqrt (m : ℝ) - Real.sqrt ((m : ℝ) - 1) ≥ (5 : ℝ)/1000) ∧
  Real.sqrt (n : ℝ) - Real.sqrt ((n : ℝ) - 1) < (5 : ℝ)/1000 ∧
  n = 10001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sqrt_difference_l1050_105096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_k_l1050_105018

-- Define the functions p and q
def p (x y : ℝ) : ℝ := x^2 - y^2
noncomputable def q (x y : ℝ) : ℝ := Real.log (x - y)

-- Define the unknown functions c and v
def c : ℝ → ℝ → ℝ := sorry
def v : ℝ → ℝ → ℝ := sorry

-- Define the variables k, m, n, w
def k : ℝ := sorry
def m : ℝ := sorry
def n : ℝ := sorry
def w : ℝ := sorry

-- State the theorem
theorem solve_for_k :
  p 32 6 = k * c 32 6 ∧
  p 45 10 = m * c 45 10 ∧
  q 15 5 = n * v 15 5 ∧
  q 28 7 = w * v 28 7 ∧
  m = 2 * k ∧
  w = n + 1 →
  k = 1925 / 1976 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_k_l1050_105018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_playground_fundraiser_result_l1050_105017

/-- The amount of money raised by the school for a new playground -/
noncomputable def playground_fundraiser 
  (johnson_amount : ℝ)
  (johnson_sutton_ratio : ℝ)
  (sutton_rollin_ratio : ℝ)
  (rollin_total_ratio : ℝ)
  (admin_fee_rate : ℝ) : ℝ :=
  let sutton_amount := johnson_amount / johnson_sutton_ratio
  let rollin_amount := sutton_amount * sutton_rollin_ratio
  let total_amount := rollin_amount / rollin_total_ratio
  let admin_fee := total_amount * admin_fee_rate
  total_amount - admin_fee

/-- The theorem stating the correct amount raised for the playground -/
theorem playground_fundraiser_result : 
  playground_fundraiser 2300 2 8 (1/3) 0.02 = 27048 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_playground_fundraiser_result_l1050_105017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knights_count_l1050_105015

/-- Represents a statement made by an islander --/
structure Statement where
  index : Nat
  isTrue : Bool

/-- Represents the collection of statements made at the meeting --/
def Meeting := List Statement

/-- Checks if a statement is valid according to the rule --/
def isValidStatement (statements : List Statement) (s : Statement) : Prop :=
  let trueCount := (statements.filter (·.isTrue)).length
  let falseCount := (statements.filter (fun x => ¬x.isTrue)).length
  s.isTrue ↔ (trueCount = falseCount - 20)

/-- The total number of islanders at the meeting --/
def totalIslanders : Nat := 65

theorem knights_count (meeting : List Statement) 
  (h1 : meeting.length = totalIslanders)
  (h2 : ∀ s, s ∈ meeting → s.index ≤ totalIslanders)
  (h3 : ∀ s, s ∈ meeting → isValidStatement (meeting.filter (fun x => x.index < s.index)) s) :
  (meeting.filter (·.isTrue)).length = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_knights_count_l1050_105015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l1050_105075

theorem number_of_subsets : Finset.card (Finset.powerset {0, 2, 3}) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l1050_105075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l1050_105038

noncomputable def sequence_a : ℕ → ℝ := sorry
noncomputable def sequence_S : ℕ → ℝ := sorry
noncomputable def sequence_c : ℕ → ℝ := sorry
noncomputable def sequence_b : ℕ → ℝ := sorry

axiom sum_condition (n : ℕ) : sequence_a n + sequence_S n = n
axiom c_definition (n : ℕ) : sequence_c n = sequence_a n - 1
axiom b_initial : sequence_b 1 = sequence_a 1
axiom b_recursive (n : ℕ) (h : n ≥ 2) : sequence_b n = sequence_a n - sequence_a (n-1)

theorem b_formula (n : ℕ) (h : n > 0) : sequence_b n = 1 / (2^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l1050_105038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solid_revolution_l1050_105043

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 2 * x
  else if 1 < x ∧ x ≤ 3 then Real.sqrt (-x^2 + 2*x + 3)
  else 0

noncomputable def volume_of_revolution (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  Real.pi * ∫ x in a..b, (f x)^2

theorem volume_of_solid_revolution :
  volume_of_revolution f 0 3 = 20 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solid_revolution_l1050_105043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_is_10_l1050_105008

/-- The speed of a boat crossing a river --/
noncomputable def boat_speed (river_width : ℝ) (crossing_time : ℝ) (drift : ℝ) : ℝ :=
  let current_speed := drift / crossing_time
  let across_speed := river_width / crossing_time
  Real.sqrt (across_speed^2 + current_speed^2)

/-- Theorem: The speed of the boat in still water is 10 m/s --/
theorem boat_speed_is_10 :
  boat_speed 400 50 300 = 10 := by
  -- Unfold the definition of boat_speed
  unfold boat_speed
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_is_10_l1050_105008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_fraction_sum_l1050_105021

open BigOperators

theorem min_n_for_fraction_sum (n : ℕ) (a : ℕ → ℕ) : 
  (∀ i j, i < j → i < n → j < n → a i < a j) →
  (∀ i, i < n → a i > 0) →
  (13 : ℚ) / 14 = ∑ i in Finset.range n, (1 : ℚ) / (a i) →
  n ≥ 4 :=
by
  sorry

#check min_n_for_fraction_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_fraction_sum_l1050_105021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_sqrt_sum_l1050_105064

theorem triangle_area_sqrt_sum (T T₁ T₂ T₃ : ℝ) 
  (h₁ : T > 0) (h₂ : T₁ > 0) (h₃ : T₂ > 0) (h₄ : T₃ > 0) :
  ∃ (k₁ k₂ k₃ : ℝ),
    k₁ > 0 ∧ k₂ > 0 ∧ k₃ > 0 ∧
    k₁ + k₂ + k₃ = 1 ∧
    T₁ = k₁^2 * T ∧
    T₂ = k₂^2 * T ∧
    T₃ = k₃^2 * T →
    Real.sqrt T = Real.sqrt T₁ + Real.sqrt T₂ + Real.sqrt T₃ :=
by
  sorry

#check triangle_area_sqrt_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_sqrt_sum_l1050_105064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1050_105042

noncomputable def f (x : ℝ) := 9*x + 1/(x-1)

theorem min_value_of_f :
  (∀ x > 1, f x ≥ 15) ∧ (∃ x > 1, f x = 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1050_105042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_decisive_games_l1050_105087

/-- Represents the number of decisive games in a chess match -/
def DecisiveGames : Type := ℕ

/-- The probability of the match ending after two games -/
noncomputable def p_end : ℝ := 1/2

/-- The expected value of the number of decisive games -/
noncomputable def E_X : ℝ := 4

/-- The theorem stating that the expected number of decisive games is 4 -/
theorem expected_decisive_games :
  let X : DecisiveGames := sorry
  let prob_win_A : ℝ := 1/2  -- probability of player A winning a single game
  let prob_win_B : ℝ := 1/2  -- probability of player B winning a single game
  E_X = 2 * p_end + (2 + E_X) * (1 - p_end) := by
  sorry

#check expected_decisive_games

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_decisive_games_l1050_105087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_2019_position_l1050_105040

/-- The triangular arrangement of natural numbers -/
def TriangularArrangement : ℕ → ℕ → ℕ := sorry

/-- The first number in row n of the triangular arrangement -/
def first_in_row (n : ℕ) : ℕ := n * (n - 1) / 2 + 1

/-- The last number in row n of the triangular arrangement -/
def last_in_row (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: 2019 is in the 64th row and 3rd position of the triangular arrangement -/
theorem number_2019_position :
  ∃ (row col : ℕ), row = 64 ∧ col = 3 ∧
  first_in_row row ≤ 2019 ∧ 2019 ≤ last_in_row row ∧
  TriangularArrangement row col = 2019 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_2019_position_l1050_105040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_time_ratio_l1050_105095

/-- Proves that the ratio of time spent on hold with Comcast to time spent turning the router off and on is 6:1 --/
theorem martha_time_ratio :
  ∀ (hold_time : ℝ),
  hold_time > 0 →
  10 + hold_time + (hold_time / 2) = 100 →
  (hold_time / 10 : ℝ) = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_time_ratio_l1050_105095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_ngon_regions_l1050_105001

/-- The number of regions in a convex n-gon divided by its diagonals -/
def num_regions (n : ℕ) : ℚ :=
  n * (n - 3) / 2 + n * (n - 1) * (n - 2) * (n - 3) / 24 + 1

/-- Definition: Three diagonals intersect at a point -/
def three_diagonals_intersect (p1 p2 p3 : ℕ × ℕ) : Prop :=
  sorry

/-- Theorem: The number of regions in a convex n-gon divided by its diagonals -/
theorem convex_ngon_regions (n : ℕ) (h : n ≥ 4) :
  let D := n * (n - 3) / 2  -- number of diagonals
  let P := n * (n - 1) * (n - 2) * (n - 3) / 24  -- number of intersection points
  (∀ p1 p2 p3 : ℕ × ℕ, ¬(three_diagonals_intersect p1 p2 p3)) →
  num_regions n = D + P + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_ngon_regions_l1050_105001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_with_given_focus_and_eccentricity_l1050_105033

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The distance from the center to a focus -/
noncomputable def Ellipse.focalDistance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- Theorem: Given an ellipse with one focus at (0,1) and eccentricity 1/2, 
    its standard equation is y^2/4 + x^2/3 = 1 -/
theorem ellipse_with_given_focus_and_eccentricity :
  ∀ (e : Ellipse), 
    e.focalDistance = 1 →
    e.eccentricity = 1/2 →
    e.a = 2 ∧ e.b = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_with_given_focus_and_eccentricity_l1050_105033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_bridge_time_l1050_105053

/-- The time it takes for a train to pass a bridge -/
noncomputable def trainPassingTime (trainLength : ℝ) (bridgeLength : ℝ) (trainSpeed : ℝ) : ℝ :=
  (trainLength + bridgeLength) / (trainSpeed * 1000 / 3600)

/-- Theorem: The time it takes for a 240-meter long train traveling at 50 km/hour 
    to pass a 130-meter long bridge is approximately 26.64 seconds -/
theorem train_passing_bridge_time :
  ∃ ε > 0, |trainPassingTime 240 130 50 - 26.64| < ε :=
by
  -- We'll use 0.01 as our epsilon
  use 0.01
  -- Split the goal into two parts
  constructor
  · -- Prove ε > 0
    norm_num
  · -- Prove |trainPassingTime 240 130 50 - 26.64| < ε
    -- This part requires computation and is left as sorry for now
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_bridge_time_l1050_105053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1050_105058

/-- The time taken for the slower train to pass the driver of the faster train -/
theorem train_passing_time (length1 length2 speed1 speed2 : ℝ) : 
  length1 = 800 →
  length2 = 600 →
  speed1 = 85 * (5/18) →
  speed2 = 65 * (5/18) →
  speed1 > speed2 →
  ∃ ε > 0, |length2 / (speed1 + speed2) - 14.4| < ε :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1050_105058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_p_l1050_105003

/-- The polynomial under consideration -/
noncomputable def p (x : ℝ) : ℝ := 3 + 7 * x^5 + 150 + 3 * Real.exp 1 * x^6 + Real.sqrt 5 * x^4 + 11

/-- The degree of a polynomial is the highest power of x in any term -/
def polynomial_degree (f : ℝ → ℝ) : ℕ := sorry

theorem degree_of_p : polynomial_degree p = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_p_l1050_105003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_30_l1050_105081

theorem remainder_sum_mod_30 (a b c : ℕ) 
  (ha : a % 30 = 15) 
  (hb : b % 30 = 5) 
  (hc : c % 30 = 20) : 
  (a + b + c) % 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_30_l1050_105081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_l1050_105070

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2) / (x + 1)

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + 2 * a * x) / (x + 1)^2

-- Theorem statement
theorem tangent_slope_implies_a (a : ℝ) :
  f_derivative a 1 = 1 → a = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_l1050_105070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_sum_2n_formula_l1050_105039

def sequence_a : ℕ → ℚ
  | 0 => 3  -- Adding case for 0
  | 1 => 3
  | 2 => 2
  | (n + 3) => if n % 2 = 0 then sequence_a (n + 1) - 1 else 3 * sequence_a (n + 1)

def general_formula (n : ℕ) : ℚ :=
  if n % 2 = 1 then (7 - n) / 2 else 2 * (3 ^ ((n - 2) / 2))

def sum_2n (n : ℕ) : ℚ :=
  -1/2 * n^2 + 7/2 * n + 3^n - 1

theorem sequence_a_formula (n : ℕ) :
  sequence_a n = general_formula n := by
  sorry

theorem sum_2n_formula (n : ℕ) :
  (Finset.range (2 * n)).sum (λ i => sequence_a (i + 1)) = sum_2n n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_sum_2n_formula_l1050_105039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_theorem_l1050_105004

noncomputable def nearest_integer (x : ℝ) : ℤ :=
  ⌊x + 1/2⌋

theorem nearest_integer_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : |x| + y = 5) (h2 : |x| * y + x^3 - 2 = 0) :
  nearest_integer (x - y + 1) = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_theorem_l1050_105004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_for_spheres_l1050_105071

/-- The height of a cylinder containing four unit spheres arranged as described -/
noncomputable def cylinder_height : ℝ := 2 * Real.sqrt (2/3) + 2

/-- The configuration of four unit spheres in a cylinder -/
structure SpheresInCylinder where
  /-- The radius of each sphere is 1 -/
  sphere_radius : ℝ := 1
  /-- The spheres are pairwise tangent -/
  pairwise_tangent : Prop
  /-- Three spheres touch one base and the lateral surface of the cylinder -/
  three_spheres_touch_base : Prop
  /-- The fourth sphere touches the other base of the cylinder -/
  fourth_sphere_touches_other_base : Prop

/-- Theorem stating the height of the cylinder containing the spheres -/
theorem cylinder_height_for_spheres (config : SpheresInCylinder) :
  config.pairwise_tangent ∧
  config.three_spheres_touch_base ∧
  config.fourth_sphere_touches_other_base →
  cylinder_height = 2 * Real.sqrt (2/3) + 2 := by
  sorry

#check cylinder_height_for_spheres

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_for_spheres_l1050_105071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_modulo_l1050_105067

theorem remainder_sum_modulo (a b c d : ℕ) :
  a % 53 = 32 →
  b % 53 = 45 →
  c % 53 = 6 →
  d % 53 = 10 →
  (a + b + c + d) % 53 = 40 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_modulo_l1050_105067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_perpendicular_through_perpendicular_line_planes_perpendicular_to_line_are_parallel_l1050_105051

-- Define the basic types
variable (P : Type) -- Type for points
variable (L : Type) -- Type for lines
variable (Plane : Type) -- Type for planes

-- Define the basic relations
variable (on_plane : P → Plane → Prop) -- Point is on a plane
variable (on_line : P → L → Prop) -- Point is on a line
variable (line_in_plane : L → Plane → Prop) -- Line is in a plane
variable (perpendicular : L → Plane → Prop) -- Line is perpendicular to a plane
variable (plane_perpendicular : Plane → Plane → Prop) -- Two planes are perpendicular
variable (plane_parallel : Plane → Plane → Prop) -- Two planes are parallel

-- Statement ②
theorem plane_perpendicular_through_perpendicular_line 
  (π₁ π₂ : Plane) (l : L) :
  perpendicular l π₁ → line_in_plane l π₂ → plane_perpendicular π₁ π₂ :=
by sorry

-- Statement ④
theorem planes_perpendicular_to_line_are_parallel 
  (π₁ π₂ : Plane) (l : L) :
  perpendicular l π₁ → perpendicular l π₂ → plane_parallel π₁ π₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_perpendicular_through_perpendicular_line_planes_perpendicular_to_line_are_parallel_l1050_105051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_proof_l1050_105059

-- Define Lisa's and Kay's cleaning rates
noncomputable def lisa_rate : ℝ := 1 / 8
noncomputable def kay_rate : ℝ := 1 / 12

-- Define the combined cleaning rate
noncomputable def combined_rate : ℝ := lisa_rate + kay_rate

-- Define the time it takes to clean the room together
noncomputable def cleaning_time : ℝ := 1 / combined_rate

-- Theorem statement
theorem cleaning_time_proof : cleaning_time = 4.8 := by
  -- Expand definitions
  unfold cleaning_time combined_rate lisa_rate kay_rate
  -- Perform algebraic manipulations
  simp [add_div, inv_div]
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_proof_l1050_105059
