import Mathlib

namespace NUMINAMATH_CALUDE_calculation_proof_l490_49096

theorem calculation_proof : (36 / (9 + 2 - 6)) * 4 = 28.8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l490_49096


namespace NUMINAMATH_CALUDE_compound_interest_rate_interest_rate_calculation_l490_49015

/-- Compound interest calculation -/
theorem compound_interest_rate (P : ℝ) (A : ℝ) (n : ℝ) (t : ℝ) (h1 : P > 0) (h2 : A > P) (h3 : n > 0) (h4 : t > 0) :
  A = P * (1 + 0.2 / n) ^ (n * t) → 
  A - P = 240 ∧ P = 1200 ∧ n = 1 ∧ t = 1 :=
by sorry

/-- Main theorem: Interest rate calculation -/
theorem interest_rate_calculation (P : ℝ) (A : ℝ) (n : ℝ) (t : ℝ) 
  (h1 : P > 0) (h2 : A > P) (h3 : n > 0) (h4 : t > 0) 
  (h5 : A - P = 240) (h6 : P = 1200) (h7 : n = 1) (h8 : t = 1) :
  ∃ r : ℝ, A = P * (1 + r) ∧ r = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_rate_interest_rate_calculation_l490_49015


namespace NUMINAMATH_CALUDE_committee_formation_l490_49001

theorem committee_formation (total : ℕ) (mathematicians : ℕ) (economists : ℕ) (committee_size : ℕ) :
  total = mathematicians + economists →
  mathematicians = 3 →
  economists = 10 →
  committee_size = 7 →
  (Nat.choose total committee_size) - (Nat.choose economists committee_size) = 1596 :=
by sorry

end NUMINAMATH_CALUDE_committee_formation_l490_49001


namespace NUMINAMATH_CALUDE_land_area_scientific_notation_l490_49056

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  norm_coeff : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The land area in square kilometers -/
def land_area : ℝ := 9600000

/-- The scientific notation of the land area -/
def land_area_scientific : ScientificNotation :=
  { coefficient := 9.6
  , exponent := 6
  , norm_coeff := by sorry }

theorem land_area_scientific_notation :
  land_area = land_area_scientific.coefficient * (10 : ℝ) ^ land_area_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_land_area_scientific_notation_l490_49056


namespace NUMINAMATH_CALUDE_unknown_number_problem_l490_49083

theorem unknown_number_problem (x : ℝ) : 
  (50 : ℝ) / 100 * 100 = (20 : ℝ) / 100 * x + 47 → x = 15 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_problem_l490_49083


namespace NUMINAMATH_CALUDE_expression_simplification_l490_49092

theorem expression_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) :
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2) = 
  3 / (2 * (-b - c + b * c)) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l490_49092


namespace NUMINAMATH_CALUDE_max_product_constraint_l490_49007

theorem max_product_constraint (a b : ℝ) : 
  a > 0 → b > 0 → 3 * a + 8 * b = 72 → ab ≤ 54 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 8 * b₀ = 72 ∧ a₀ * b₀ = 54 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l490_49007


namespace NUMINAMATH_CALUDE_evaluate_expression_l490_49020

theorem evaluate_expression : 3002^3 - 3001 * 3002^2 - 3001^2 * 3002 + 3001^3 + 1 = 6004 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l490_49020


namespace NUMINAMATH_CALUDE_other_roots_of_polynomial_l490_49005

def f (a b x : ℝ) : ℝ := x^3 + 4*x^2 + a*x + b

theorem other_roots_of_polynomial (a b : ℚ) :
  (f a b (2 + Real.sqrt 3) = 0) →
  (f a b (2 - Real.sqrt 3) = 0) ∧ (f a b (-8) = 0) :=
by sorry

end NUMINAMATH_CALUDE_other_roots_of_polynomial_l490_49005


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l490_49046

/-- Regular hexagon with vertices A and C -/
structure RegularHexagon where
  A : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a regular hexagon -/
def hexagon_area (h : RegularHexagon) : ℝ := sorry

/-- Theorem: Area of the specific regular hexagon -/
theorem specific_hexagon_area :
  let h : RegularHexagon := { A := (0, 0), C := (8, 2) }
  hexagon_area h = 34 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l490_49046


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l490_49070

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l490_49070


namespace NUMINAMATH_CALUDE_undefined_function_roots_sum_l490_49032

theorem undefined_function_roots_sum : 
  let f (x : ℝ) := 3 * x^2 - 9 * x + 6
  ∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ ≠ r₂ ∧ r₁ + r₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_undefined_function_roots_sum_l490_49032


namespace NUMINAMATH_CALUDE_six_selected_in_interval_l490_49061

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ
  (population_positive : population > 0)
  (sample_size_positive : sample_size > 0)
  (sample_size_le_population : sample_size ≤ population)
  (interval_valid : interval_start ≤ interval_end)
  (interval_in_range : interval_end ≤ population)

/-- Calculates the number of selected individuals within a given interval -/
def selected_in_interval (s : SystematicSample) : ℕ :=
  ((s.interval_end - s.interval_start + 1) + (s.population / s.sample_size - 1)) / (s.population / s.sample_size)

/-- Theorem stating that for the given parameters, 6 individuals are selected within the interval -/
theorem six_selected_in_interval (s : SystematicSample) 
  (h_pop : s.population = 420)
  (h_sample : s.sample_size = 21)
  (h_start : s.interval_start = 241)
  (h_end : s.interval_end = 360) :
  selected_in_interval s = 6 := by
  sorry

#eval selected_in_interval {
  population := 420,
  sample_size := 21,
  interval_start := 241,
  interval_end := 360,
  population_positive := by norm_num,
  sample_size_positive := by norm_num,
  sample_size_le_population := by norm_num,
  interval_valid := by norm_num,
  interval_in_range := by norm_num
}

end NUMINAMATH_CALUDE_six_selected_in_interval_l490_49061


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l490_49036

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 1734 → s^3 = 4913 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l490_49036


namespace NUMINAMATH_CALUDE_max_value_fraction_l490_49093

theorem max_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 3 ≤ y ∧ y ≤ 6) :
  (∀ a b, -3 ≤ a ∧ a ≤ -1 ∧ 3 ≤ b ∧ b ≤ 6 → (a + b) / a ≤ (x + y) / x) →
  (x + y) / x = -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l490_49093


namespace NUMINAMATH_CALUDE_constant_term_expansion_l490_49098

/-- The constant term in the expansion of (3x + 2/x)^8 -/
def constant_term : ℕ := 90720

/-- The binomial coefficient (8 choose 4) -/
def binom_8_4 : ℕ := 70

theorem constant_term_expansion :
  constant_term = binom_8_4 * 3^4 * 2^4 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l490_49098


namespace NUMINAMATH_CALUDE_intersection_midpoint_theorem_l490_49085

-- Define the curve
def curve (x : ℝ) : ℝ := -2 * x^2 + 5 * x - 2

-- Define the line
def line (x : ℝ) : ℝ := -2 * x

-- Theorem statement
theorem intersection_midpoint_theorem (s : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    curve x₁ = line x₁ ∧ 
    curve x₂ = line x₂ ∧ 
    (curve x₁ + curve x₂) / 2 = 7 / s) →
  s = -2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_midpoint_theorem_l490_49085


namespace NUMINAMATH_CALUDE_rectangle_areas_sum_l490_49069

theorem rectangle_areas_sum : 
  let base_width : ℕ := 2
  let lengths : List ℕ := [1, 4, 9, 16, 25, 36, 49]
  let areas : List ℕ := lengths.map (λ l => base_width * l)
  areas.sum = 280 := by sorry

end NUMINAMATH_CALUDE_rectangle_areas_sum_l490_49069


namespace NUMINAMATH_CALUDE_fundamental_theorem_of_calculus_l490_49016

open Set
open Interval
open MeasureTheory
open Real

-- Define the theorem
theorem fundamental_theorem_of_calculus 
  (f : ℝ → ℝ) (f' : ℝ → ℝ) (a b : ℝ) 
  (h1 : ContinuousOn f (Icc a b))
  (h2 : DifferentiableOn ℝ f (Ioc a b))
  (h3 : ∀ x ∈ Ioc a b, deriv f x = f' x) :
  ∫ x in a..b, f' x = f b - f a :=
sorry

end NUMINAMATH_CALUDE_fundamental_theorem_of_calculus_l490_49016


namespace NUMINAMATH_CALUDE_ram_krish_work_time_l490_49078

/-- Represents the efficiency of a worker -/
structure Efficiency : Type :=
  (value : ℝ)

/-- Represents the time taken to complete a task -/
structure Time : Type :=
  (days : ℝ)

/-- Represents the amount of work in a task -/
structure Work : Type :=
  (amount : ℝ)

/-- The theorem stating the relationship between Ram and Krish's efficiency and their combined work time -/
theorem ram_krish_work_time 
  (ram_efficiency : Efficiency)
  (krish_efficiency : Efficiency)
  (ram_alone_time : Time)
  (task : Work)
  (h1 : ram_efficiency.value = (1 / 2) * krish_efficiency.value)
  (h2 : ram_alone_time.days = 30)
  (h3 : task.amount = ram_efficiency.value * ram_alone_time.days) :
  ∃ (combined_time : Time),
    combined_time.days = 10 ∧
    task.amount = (ram_efficiency.value + krish_efficiency.value) * combined_time.days :=
sorry

end NUMINAMATH_CALUDE_ram_krish_work_time_l490_49078


namespace NUMINAMATH_CALUDE_bird_percentage_problem_l490_49033

theorem bird_percentage_problem (total : ℝ) (pigeons sparrows crows parakeets : ℝ) :
  pigeons = 0.4 * total →
  sparrows = 0.2 * total →
  crows = 0.15 * total →
  parakeets = total - (pigeons + sparrows + crows) →
  crows / (total - sparrows) = 0.1875 := by
sorry

end NUMINAMATH_CALUDE_bird_percentage_problem_l490_49033


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l490_49010

theorem profit_percentage_calculation (C S : ℝ) (h1 : C > 0) (h2 : S > 0) :
  20 * C = 16 * S →
  (S - C) / C * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l490_49010


namespace NUMINAMATH_CALUDE_equilateral_is_peculiar_specific_right_triangle_is_peculiar_right_angled_peculiar_triangle_ratio_l490_49031

-- Definition of a peculiar triangle
def is_peculiar_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2*c^2 ∨ a^2 + c^2 = 2*b^2 ∨ b^2 + c^2 = 2*a^2

-- Definition of an equilateral triangle
def is_equilateral_triangle (a b c : ℝ) : Prop :=
  a = b ∧ b = c

-- Definition of a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Theorem 1: An equilateral triangle is a peculiar triangle
theorem equilateral_is_peculiar (a b c : ℝ) :
  is_equilateral_triangle a b c → is_peculiar_triangle a b c :=
sorry

-- Theorem 2: A right triangle with sides 5√2, 10, and 5√6 is a peculiar triangle
theorem specific_right_triangle_is_peculiar :
  let a : ℝ := 5 * Real.sqrt 2
  let b : ℝ := 5 * Real.sqrt 6
  let c : ℝ := 10
  is_right_triangle a b c ∧ is_peculiar_triangle a b c :=
sorry

-- Theorem 3: In a right-angled peculiar triangle, the ratio of sides is 1:√2:√3
theorem right_angled_peculiar_triangle_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > a) :
  is_right_triangle a b c ∧ is_peculiar_triangle a b c →
  ∃ (k : ℝ), a = k ∧ b = k * Real.sqrt 2 ∧ c = k * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_equilateral_is_peculiar_specific_right_triangle_is_peculiar_right_angled_peculiar_triangle_ratio_l490_49031


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l490_49095

theorem greatest_power_of_two_factor : ∃ k : ℕ, k = 502 ∧ 
  (∀ m : ℕ, 2^m ∣ (12^1002 - 6^501) → m ≤ k) ∧
  (2^k ∣ (12^1002 - 6^501)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l490_49095


namespace NUMINAMATH_CALUDE_fraction_decomposition_l490_49094

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 5/3) :
  (7 * x - 15) / (3 * x^2 - x - 10) = (29/11) / (x + 2) + (-9/11) / (3*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l490_49094


namespace NUMINAMATH_CALUDE_max_value_and_x_l490_49087

theorem max_value_and_x (x : ℝ) (y : ℝ) (h : x < 0) (h1 : y = 3*x + 4/x) :
  (∀ z, z < 0 → 3*z + 4/z ≤ y) → y = -4*Real.sqrt 3 ∧ x = -2*Real.sqrt 3/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_and_x_l490_49087


namespace NUMINAMATH_CALUDE_angle_terminal_side_l490_49090

theorem angle_terminal_side (α : Real) :
  let P : ℝ × ℝ := (Real.tan α, Real.cos α)
  (P.1 < 0 ∧ P.2 < 0) →  -- Point P is in the third quadrant
  (Real.cos α < 0 ∧ Real.sin α > 0) -- Terminal side of α is in the second quadrant
:= by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l490_49090


namespace NUMINAMATH_CALUDE_nyusha_ate_28_candies_l490_49048

-- Define the number of candies eaten by each person
variable (K E N B : ℕ)

-- Define the conditions
axiom total_candies : K + E + N + B = 86
axiom minimum_candies : K ≥ 5 ∧ E ≥ 5 ∧ N ≥ 5 ∧ B ≥ 5
axiom nyusha_ate_most : N > K ∧ N > E ∧ N > B
axiom kros_yozhik_total : K + E = 53

-- Theorem to prove
theorem nyusha_ate_28_candies : N = 28 := by
  sorry


end NUMINAMATH_CALUDE_nyusha_ate_28_candies_l490_49048


namespace NUMINAMATH_CALUDE_rhombus_min_rotation_l490_49013

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  sides : Fin 4 → ℝ
  sides_equal : ∀ i j : Fin 4, sides i = sides j

/-- The minimum rotation angle for a rhombus to coincide with its original position -/
def min_rotation_angle (r : Rhombus) : ℝ := 180

/-- Theorem: The minimum rotation angle for a rhombus with a 60° angle to coincide with its original position is 180° -/
theorem rhombus_min_rotation (r : Rhombus) (angle : ℝ) (h : angle = 60) :
  min_rotation_angle r = 180 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_min_rotation_l490_49013


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l490_49052

theorem sqrt_product_plus_one : 
  Real.sqrt ((21 : ℝ) * 20 * 19 * 18 + 1) = 379 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l490_49052


namespace NUMINAMATH_CALUDE_aquarium_visit_cost_difference_l490_49074

/-- Represents the cost structure and family composition for an aquarium visit -/
structure AquariumVisit where
  family_pass_cost : ℚ
  adult_ticket_cost : ℚ
  child_ticket_cost : ℚ
  num_adults : ℕ
  num_children : ℕ

/-- Calculates the cost of separate tickets with the special offer applied -/
def separate_tickets_cost (visit : AquariumVisit) : ℚ :=
  let discounted_adults := visit.num_children / 3
  let full_price_adults := visit.num_adults - discounted_adults
  let discounted_adult_cost := visit.adult_ticket_cost * (1/2)
  discounted_adults * discounted_adult_cost +
  full_price_adults * visit.adult_ticket_cost +
  visit.num_children * visit.child_ticket_cost

/-- Theorem stating the difference between separate tickets and family pass -/
theorem aquarium_visit_cost_difference (visit : AquariumVisit) 
  (h1 : visit.family_pass_cost = 150)
  (h2 : visit.adult_ticket_cost = 35)
  (h3 : visit.child_ticket_cost = 20)
  (h4 : visit.num_adults = 2)
  (h5 : visit.num_children = 5) :
  separate_tickets_cost visit - visit.family_pass_cost = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_visit_cost_difference_l490_49074


namespace NUMINAMATH_CALUDE_compare_negative_roots_l490_49068

theorem compare_negative_roots : -6 * Real.sqrt 5 < -5 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_roots_l490_49068


namespace NUMINAMATH_CALUDE_x_squared_minus_two_is_quadratic_l490_49054

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 2 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- Theorem: x^2 - 2 = 0 is a quadratic equation -/
theorem x_squared_minus_two_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_two_is_quadratic_l490_49054


namespace NUMINAMATH_CALUDE_greatest_m_value_l490_49040

def reverse_number (n : ℕ) : ℕ := sorry

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_m_value (m : ℕ) 
  (h1 : is_four_digit m)
  (h2 : is_four_digit (reverse_number m))
  (h3 : m % 63 = 0)
  (h4 : (reverse_number m) % 63 = 0)
  (h5 : m % 11 = 0) :
  m ≤ 9811 ∧ ∃ (m : ℕ), m = 9811 ∧ 
    is_four_digit m ∧
    is_four_digit (reverse_number m) ∧
    m % 63 = 0 ∧
    (reverse_number m) % 63 = 0 ∧
    m % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_greatest_m_value_l490_49040


namespace NUMINAMATH_CALUDE_arcsin_arccos_range_l490_49039

theorem arcsin_arccos_range (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (z : ℝ), z = 2 * Real.arcsin x - Real.arccos y ∧ -3 * π / 2 ≤ z ∧ z ≤ π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_arccos_range_l490_49039


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l490_49099

-- Define the binary operation ◇
noncomputable def diamond (a b : ℝ) : ℝ := a / b

-- State the theorem
theorem diamond_equation_solution :
  (∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 → diamond a (diamond b c) = diamond (diamond a b) c) →
  (∀ (a : ℝ), a ≠ 0 → diamond a a = 1) →
  diamond 504 (diamond 12 (25 / 21)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l490_49099


namespace NUMINAMATH_CALUDE_particle_position_after_1989_minutes_l490_49026

/-- Represents the position of a particle in 2D space -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Calculates the time taken to enclose n squares -/
def timeForSquares (n : ℕ) : ℕ :=
  (n + 1)^2 - 1

/-- Calculates the position of the particle after a given number of minutes -/
def particlePosition (minutes : ℕ) : Position :=
  sorry

/-- The theorem stating the position of the particle after 1989 minutes -/
theorem particle_position_after_1989_minutes :
  particlePosition 1989 = Position.mk 44 35 := by sorry

end NUMINAMATH_CALUDE_particle_position_after_1989_minutes_l490_49026


namespace NUMINAMATH_CALUDE_santos_salvadore_earnings_ratio_l490_49065

/-- Proves that the ratio of Santo's earnings to Salvadore's earnings is 1:2 -/
theorem santos_salvadore_earnings_ratio :
  let salvadore_earnings : ℚ := 1956
  let total_earnings : ℚ := 2934
  let santo_earnings : ℚ := total_earnings - salvadore_earnings
  santo_earnings / salvadore_earnings = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_santos_salvadore_earnings_ratio_l490_49065


namespace NUMINAMATH_CALUDE_at_least_one_true_l490_49071

theorem at_least_one_true (p q : Prop) : ¬(¬(p ∨ q)) → (p ∨ q) := by sorry

end NUMINAMATH_CALUDE_at_least_one_true_l490_49071


namespace NUMINAMATH_CALUDE_total_floor_area_square_slabs_l490_49018

/-- Calculates the total floor area covered by square stone slabs. -/
theorem total_floor_area_square_slabs 
  (num_slabs : ℕ) 
  (slab_length : ℝ) 
  (h1 : num_slabs = 30)
  (h2 : slab_length = 200)
  : (num_slabs * (slab_length / 100)^2 : ℝ) = 120 := by
  sorry

#check total_floor_area_square_slabs

end NUMINAMATH_CALUDE_total_floor_area_square_slabs_l490_49018


namespace NUMINAMATH_CALUDE_chocolate_chip_calculation_l490_49044

/-- The number of cups of chocolate chips needed for one recipe -/
def cups_per_recipe : ℕ := 2

/-- The number of recipes to be made -/
def number_of_recipes : ℕ := 23

/-- The total number of cups of chocolate chips needed -/
def total_cups : ℕ := cups_per_recipe * number_of_recipes

theorem chocolate_chip_calculation : total_cups = 46 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_calculation_l490_49044


namespace NUMINAMATH_CALUDE_gcd_of_136_and_1275_l490_49066

theorem gcd_of_136_and_1275 : Nat.gcd 136 1275 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_136_and_1275_l490_49066


namespace NUMINAMATH_CALUDE_zero_shaded_area_l490_49027

/-- Represents a square tile with a pattern of triangles -/
structure Tile where
  sideLength : ℝ
  triangleArea : ℝ

/-- Represents a rectangular floor tiled with square tiles -/
structure Floor where
  length : ℝ
  width : ℝ
  tile : Tile

/-- Calculates the total shaded area of the floor -/
def totalShadedArea (floor : Floor) : ℝ :=
  let totalTiles := floor.length * floor.width
  let tileArea := floor.tile.sideLength ^ 2
  let shadedAreaPerTile := tileArea - 4 * floor.tile.triangleArea
  totalTiles * shadedAreaPerTile

/-- Theorem stating that the total shaded area of the specific floor is 0 -/
theorem zero_shaded_area :
  let tile : Tile := {
    sideLength := 1,
    triangleArea := 1/4
  }
  let floor : Floor := {
    length := 12,
    width := 9,
    tile := tile
  }
  totalShadedArea floor = 0 := by sorry

end NUMINAMATH_CALUDE_zero_shaded_area_l490_49027


namespace NUMINAMATH_CALUDE_harry_worked_36_hours_l490_49067

/-- Payment structure for Harry and James -/
structure PaymentStructure where
  base_rate : ℝ
  harry_base_hours : ℕ := 30
  harry_overtime_rate : ℝ := 2
  james_base_hours : ℕ := 40
  james_overtime_rate : ℝ := 1.5

/-- Calculate pay for a given number of hours worked -/
def calculate_pay (ps : PaymentStructure) (base_hours : ℕ) (overtime_rate : ℝ) (hours_worked : ℕ) : ℝ :=
  if hours_worked ≤ base_hours then
    ps.base_rate * hours_worked
  else
    ps.base_rate * base_hours + ps.base_rate * overtime_rate * (hours_worked - base_hours)

/-- Theorem stating that Harry worked 36 hours if paid the same as James who worked 41 hours -/
theorem harry_worked_36_hours (ps : PaymentStructure) :
  calculate_pay ps ps.james_base_hours ps.james_overtime_rate 41 =
  calculate_pay ps ps.harry_base_hours ps.harry_overtime_rate 36 :=
sorry

end NUMINAMATH_CALUDE_harry_worked_36_hours_l490_49067


namespace NUMINAMATH_CALUDE_min_pizzas_break_even_l490_49060

/-- The minimum number of whole pizzas John must deliver to break even -/
def min_pizzas : ℕ := 1000

/-- The cost of the car -/
def car_cost : ℕ := 8000

/-- The earning per pizza -/
def earning_per_pizza : ℕ := 12

/-- The gas cost per pizza -/
def gas_cost_per_pizza : ℕ := 4

/-- Theorem stating that min_pizzas is the minimum number of whole pizzas
    John must deliver to at least break even on his car purchase -/
theorem min_pizzas_break_even :
  min_pizzas = (car_cost + gas_cost_per_pizza - 1) / (earning_per_pizza - gas_cost_per_pizza) :=
by sorry

end NUMINAMATH_CALUDE_min_pizzas_break_even_l490_49060


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l490_49057

/-- Given a line L1 with equation x - 2y - 2 = 0 and a point P (5, 3),
    the line L2 passing through P and perpendicular to L1 has equation 2x + y - 13 = 0 -/
theorem perpendicular_line_equation (L1 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  (L1 = {(x, y) | x - 2*y - 2 = 0}) →
  (P = (5, 3)) →
  (∃ L2 : Set (ℝ × ℝ), 
    (P ∈ L2) ∧ 
    (∀ (v w : ℝ × ℝ), v ∈ L1 → w ∈ L1 → v ≠ w → 
      ∀ (p q : ℝ × ℝ), p ∈ L2 → q ∈ L2 → p ≠ q → 
        ((v.1 - w.1) * (p.1 - q.1) + (v.2 - w.2) * (p.2 - q.2) = 0)) ∧
    (L2 = {(x, y) | 2*x + y - 13 = 0})) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l490_49057


namespace NUMINAMATH_CALUDE_minimum_value_inequality_l490_49034

theorem minimum_value_inequality (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : 2 * m + n = 1) :
  1 / m + 2 / n ≥ 8 ∧ (1 / m + 2 / n = 8 ↔ n = 2 * m ∧ n = 1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_inequality_l490_49034


namespace NUMINAMATH_CALUDE_squirrel_nut_difference_squirrel_nut_difference_example_l490_49079

theorem squirrel_nut_difference : ℕ → ℕ → ℕ
  | num_squirrels, num_nuts =>
    num_squirrels - num_nuts

theorem squirrel_nut_difference_example : squirrel_nut_difference 4 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_nut_difference_squirrel_nut_difference_example_l490_49079


namespace NUMINAMATH_CALUDE_harmonic_mean_closest_to_ten_l490_49029

theorem harmonic_mean_closest_to_ten :
  let a := 5
  let b := 2023
  let harmonic_mean := 2 * a * b / (a + b)
  ∀ n : ℤ, n ≠ 10 → |harmonic_mean - 10| < |harmonic_mean - n| :=
by sorry

end NUMINAMATH_CALUDE_harmonic_mean_closest_to_ten_l490_49029


namespace NUMINAMATH_CALUDE_range_of_a_l490_49081

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2) 3, 2 * x > x^2 + a) → a < -8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l490_49081


namespace NUMINAMATH_CALUDE_percentage_calculation_l490_49023

theorem percentage_calculation (x : ℝ) : 
  x = 0.18 * 4750 → 1.5 * x = 1282.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l490_49023


namespace NUMINAMATH_CALUDE_orphanage_donation_percentage_l490_49076

def total_income : ℝ := 1000000
def children_percentage : ℝ := 0.2
def num_children : ℕ := 3
def wife_percentage : ℝ := 0.3
def final_amount : ℝ := 50000

theorem orphanage_donation_percentage :
  let family_distribution := children_percentage * num_children + wife_percentage
  let remaining_before_donation := total_income * (1 - family_distribution)
  let donation_amount := remaining_before_donation - final_amount
  (donation_amount / remaining_before_donation) * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_orphanage_donation_percentage_l490_49076


namespace NUMINAMATH_CALUDE_shares_problem_l490_49097

theorem shares_problem (total : ℕ) (a b c : ℕ) : 
  total = 1760 →
  a + b + c = total →
  3 * b = 4 * a →
  5 * a = 3 * c →
  6 * a = 8 * b →
  8 * b = 20 * c →
  c = 250 := by
sorry

end NUMINAMATH_CALUDE_shares_problem_l490_49097


namespace NUMINAMATH_CALUDE_probability_A_and_B_selected_l490_49011

/-- The number of students in the group -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The probability of selecting both A and B -/
def prob_select_A_and_B : ℚ := 3 / 10

/-- Theorem stating that the probability of selecting both A and B
    when randomly choosing 3 students from a group of 5 students is 3/10 -/
theorem probability_A_and_B_selected :
  (Nat.choose (total_students - 2) (selected_students - 2) : ℚ) /
  (Nat.choose total_students selected_students : ℚ) = prob_select_A_and_B :=
sorry

end NUMINAMATH_CALUDE_probability_A_and_B_selected_l490_49011


namespace NUMINAMATH_CALUDE_rope_length_problem_l490_49082

theorem rope_length_problem (total_ropes : ℕ) (avg_length_all : ℝ) (avg_length_third : ℝ) :
  total_ropes = 6 →
  avg_length_all = 80 →
  avg_length_third = 70 →
  let third_ropes := total_ropes / 3
  let remaining_ropes := total_ropes - third_ropes
  let total_length := total_ropes * avg_length_all
  let third_length := third_ropes * avg_length_third
  let remaining_length := total_length - third_length
  remaining_length / remaining_ropes = 85 := by
sorry

end NUMINAMATH_CALUDE_rope_length_problem_l490_49082


namespace NUMINAMATH_CALUDE_union_with_complement_l490_49047

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem union_with_complement : A ∪ (U \ B) = {1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_with_complement_l490_49047


namespace NUMINAMATH_CALUDE_triangle_implies_s_range_l490_49086

-- Define the system of inequalities
def SystemOfInequalities : Type := Unit  -- Placeholder, as we don't have specific inequalities

-- Define what it means for a region to be a triangle
def IsTriangle (region : SystemOfInequalities) : Prop := sorry

-- Define the range of s
def SRange (s : ℝ) : Prop := (0 < s ∧ s ≤ 2) ∨ s ≥ 4

-- Theorem statement
theorem triangle_implies_s_range (region : SystemOfInequalities) :
  IsTriangle region → ∀ s, SRange s :=
sorry

end NUMINAMATH_CALUDE_triangle_implies_s_range_l490_49086


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l490_49064

/-- Given a cube with surface area 864 square centimeters, its volume is 1728 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ (a : ℝ), 
  (6 * a^2 = 864) →  -- Surface area of cube is 864 sq cm
  (a^3 = 1728)       -- Volume of cube is 1728 cubic cm
:= by sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l490_49064


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_6_l490_49014

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Hyperbola type -/
structure Hyperbola where
  equation : ℝ → ℝ → ℝ → Prop
  a : ℝ

/-- Theorem: Eccentricity of hyperbola given specific conditions -/
theorem hyperbola_eccentricity_sqrt_6
  (p : Parabola)
  (h : Hyperbola)
  (A B : ℝ × ℝ)
  (h_parabola : p.equation = fun x y ↦ y^2 = 4*x)
  (h_hyperbola : h.equation = fun x y a ↦ x^2/a^2 - y^2 = 1)
  (h_a_pos : h.a > 0)
  (h_intersection : p.equation A.1 A.2 ∧ h.equation A.1 A.2 h.a ∧
                    p.equation B.1 B.2 ∧ h.equation B.1 B.2 h.a)
  (h_right_angle : (A.1 - p.focus.1) * (B.1 - p.focus.1) +
                   (A.2 - p.focus.2) * (B.2 - p.focus.2) = 0) :
  ∃ (e : ℝ), e^2 = 6 ∧ e = (Real.sqrt ((h.a^2 + 1) / h.a^2)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_6_l490_49014


namespace NUMINAMATH_CALUDE_log_problem_l490_49089

theorem log_problem (x : ℝ) (h : x = (Real.log 2 / Real.log 4) ^ ((Real.log 16 / Real.log 2) ^ 2)) :
  Real.log x / Real.log 5 = -16 / Real.log 5 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l490_49089


namespace NUMINAMATH_CALUDE_family_probability_l490_49062

/-- The probability of having a boy or a girl -/
def child_probability : ℚ := 1 / 2

/-- The number of children in the family -/
def family_size : ℕ := 4

/-- The probability of having at least one boy and one girl in a family of four children -/
def prob_at_least_one_boy_and_girl : ℚ := 7 / 8

theorem family_probability : 
  1 - (child_probability ^ family_size + child_probability ^ family_size) = prob_at_least_one_boy_and_girl :=
sorry

end NUMINAMATH_CALUDE_family_probability_l490_49062


namespace NUMINAMATH_CALUDE_ellipse_and_line_problem_l490_49059

structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  e : ℝ

def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 + 1}

theorem ellipse_and_line_problem (C : Ellipse) (L : ℝ → Set (ℝ × ℝ)) :
  C.c = 4 * Real.sqrt 3 →
  C.e = Real.sqrt 3 / 2 →
  (∀ x y, (x, y) ∈ {p : ℝ × ℝ | x^2 / C.a^2 + y^2 / C.b^2 = 1} ↔ (x, y) ∈ {p : ℝ × ℝ | x^2 / 16 + y^2 / 4 = 1}) →
  (∃ A B : ℝ × ℝ, A ∈ L k ∧ B ∈ L k ∧ A ∈ {p : ℝ × ℝ | x^2 / 16 + y^2 / 4 = 1} ∧ B ∈ {p : ℝ × ℝ | x^2 / 16 + y^2 / 4 = 1}) →
  (∀ A B : ℝ × ℝ, A ∈ L k ∧ B ∈ L k → A.1 = -2 * B.1) →
  (k = Real.sqrt 15 / 10 ∨ k = -Real.sqrt 15 / 10) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_problem_l490_49059


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l490_49049

theorem consecutive_odd_integers_sum (x : ℤ) : 
  x > 0 ∧ 
  Odd x ∧ 
  Odd (x + 2) ∧ 
  x * (x + 2) = 945 → 
  x + (x + 2) = 60 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l490_49049


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l490_49042

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b + b / c + c / a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l490_49042


namespace NUMINAMATH_CALUDE_parabola_point_relation_l490_49072

theorem parabola_point_relation (a : ℝ) (y₁ y₂ : ℝ) 
  (h1 : a > 0) 
  (h2 : y₁ = a * (-2)^2) 
  (h3 : y₂ = a * 1^2) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_relation_l490_49072


namespace NUMINAMATH_CALUDE_unique_solution_l490_49028

/-- Definition of the diamond operation -/
def diamond (a b c d : ℝ) : ℝ × ℝ :=
  (a * c - b * d, a * d + b * c)

/-- Theorem stating the unique solution to the equation -/
theorem unique_solution :
  ∀ x y : ℝ, diamond x 3 x y = (6, 0) ↔ x = 0 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l490_49028


namespace NUMINAMATH_CALUDE_aj_has_370_stamps_l490_49003

/-- The number of stamps each person has -/
structure Stamps where
  aj : ℕ
  kj : ℕ
  cj : ℕ

/-- The conditions of the stamp collection problem -/
def stamp_problem (s : Stamps) : Prop :=
  s.cj = 5 + 2 * s.kj ∧
  s.kj = s.aj / 2 ∧
  s.aj + s.kj + s.cj = 930

/-- The theorem stating that AJ has 370 stamps -/
theorem aj_has_370_stamps :
  ∃ (s : Stamps), stamp_problem s ∧ s.aj = 370 :=
by
  sorry


end NUMINAMATH_CALUDE_aj_has_370_stamps_l490_49003


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equation_l490_49075

theorem sum_of_reciprocal_equation (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (eq1 : 1 / x + 1 / y = 4)
  (eq2 : 1 / x - 1 / y = -8) :
  x + y = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equation_l490_49075


namespace NUMINAMATH_CALUDE_largest_c_value_l490_49000

-- Define the function g
def g (x : ℝ) : ℝ := x^2 + 3*x + 1

-- State the theorem
theorem largest_c_value (d : ℝ) (hd : d > 0) :
  (∃ (c : ℝ), c > 0 ∧ 
    (∀ (x : ℝ), |x - 1| ≤ d → |g x - 1| ≤ c) ∧
    (∀ (c' : ℝ), c' > c → ∃ (x : ℝ), |x - 1| ≤ d ∧ |g x - 1| > c')) ∧
  (∀ (c : ℝ), 
    (c > 0 ∧ 
     (∀ (x : ℝ), |x - 1| ≤ d → |g x - 1| ≤ c) ∧
     (∀ (c' : ℝ), c' > c → ∃ (x : ℝ), |x - 1| ≤ d ∧ |g x - 1| > c'))
    → c = 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_value_l490_49000


namespace NUMINAMATH_CALUDE_product_abcd_l490_49091

theorem product_abcd (a b c d : ℚ) : 
  3*a + 2*b + 4*c + 6*d = 48 →
  4*(d+c) = b →
  4*b + 2*c = a →
  2*c - 2 = d →
  a * b * c * d = -58735360 / 81450625 :=
by sorry

end NUMINAMATH_CALUDE_product_abcd_l490_49091


namespace NUMINAMATH_CALUDE_fence_decoration_combinations_l490_49088

def num_colors : ℕ := 6
def num_techniques : ℕ := 5

theorem fence_decoration_combinations :
  num_colors * num_techniques = 30 := by
  sorry

end NUMINAMATH_CALUDE_fence_decoration_combinations_l490_49088


namespace NUMINAMATH_CALUDE_luke_game_points_per_round_l490_49045

/-- Given a total score and number of rounds in a game where equal points are gained in each round,
    calculate the points gained per round. -/
def points_per_round (total_score : ℕ) (num_rounds : ℕ) : ℚ :=
  total_score / num_rounds

/-- Theorem stating that for Luke's game with 154 total points over 14 rounds,
    the points gained per round is 11. -/
theorem luke_game_points_per_round :
  points_per_round 154 14 = 11 := by
  sorry

end NUMINAMATH_CALUDE_luke_game_points_per_round_l490_49045


namespace NUMINAMATH_CALUDE_symmetry_line_equation_l490_49080

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := y = -x^2 + 4*x - 2
def C2 (x y : ℝ) : Prop := y^2 = x

-- Define symmetry about a line
def symmetric_about_line (l : ℝ → ℝ → Prop) (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), C1 x1 y1 → C2 x2 y2 → 
    ∃ (x' y' : ℝ), l x' y' ∧ 
      x' = (x1 + x2) / 2 ∧ 
      y' = (y1 + y2) / 2

-- Theorem statement
theorem symmetry_line_equation :
  ∀ (l : ℝ → ℝ → Prop),
  symmetric_about_line l C1 C2 →
  (∀ x y, l x y ↔ x + y - 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_symmetry_line_equation_l490_49080


namespace NUMINAMATH_CALUDE_werewolf_identity_l490_49006

/-- Represents a forest dweller -/
inductive Dweller
| A
| B
| C

/-- Represents the status of a dweller -/
structure Status where
  is_werewolf : Bool
  is_knight : Bool

/-- The statement made by B -/
def b_statement (status : Dweller → Status) : Prop :=
  (status Dweller.C).is_werewolf

theorem werewolf_identity (status : Dweller → Status) :
  (∃! d : Dweller, (status d).is_werewolf ∧ (status d).is_knight) →
  (∀ d : Dweller, d ≠ Dweller.A → d ≠ Dweller.B → ¬(status d).is_knight) →
  b_statement status →
  (status Dweller.A).is_werewolf := by
  sorry

end NUMINAMATH_CALUDE_werewolf_identity_l490_49006


namespace NUMINAMATH_CALUDE_sin_equals_cos_810_deg_l490_49019

theorem sin_equals_cos_810_deg (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * Real.pi / 180) = Real.cos (810 * Real.pi / 180) ↔ n = -180 ∨ n = 0 ∨ n = 180) :=
by sorry

end NUMINAMATH_CALUDE_sin_equals_cos_810_deg_l490_49019


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l490_49035

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

-- Theorem statement
theorem hyperbola_asymptote :
  ∀ x y : ℝ, hyperbola x y → (∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ hyperbola x' y' ∧ asymptote x' y') :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l490_49035


namespace NUMINAMATH_CALUDE_total_treats_eq_155_l490_49063

/-- The number of chewing gums -/
def chewing_gums : ℕ := 60

/-- The number of chocolate bars -/
def chocolate_bars : ℕ := 55

/-- The number of candies of different flavors -/
def candies : ℕ := 40

/-- The total number of treats -/
def total_treats : ℕ := chewing_gums + chocolate_bars + candies

theorem total_treats_eq_155 : total_treats = 155 := by sorry

end NUMINAMATH_CALUDE_total_treats_eq_155_l490_49063


namespace NUMINAMATH_CALUDE_ab_range_l490_49002

def f (x : ℝ) : ℝ := |2 - x^2|

theorem ab_range (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  ∃ (l u : ℝ), l = 0 ∧ u = 2 ∧ ∀ x, a * b = x → l < x ∧ x < u :=
sorry

end NUMINAMATH_CALUDE_ab_range_l490_49002


namespace NUMINAMATH_CALUDE_value_of_a_l490_49037

theorem value_of_a (a b c : ℤ) 
  (eq1 : a + b = 2 * c - 1)
  (eq2 : b + c = 7)
  (eq3 : c = 4) : 
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l490_49037


namespace NUMINAMATH_CALUDE_fraction_equality_l490_49077

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 4 / 3) 
  (h2 : r / t = 9 / 14) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -11 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l490_49077


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_subtraction_for_509_divisible_by_9_least_subtraction_509_divisible_by_9_l490_49051

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem subtraction_for_509_divisible_by_9 :
  ∃ (k : ℕ), k < 9 ∧ (509 - k) % 9 = 0 ∧ ∀ (m : ℕ), m < k → (509 - m) % 9 ≠ 0 :=
by
  sorry

#eval (509 - 5) % 9  -- Should output 0

theorem least_subtraction_509_divisible_by_9 :
  5 < 9 ∧ (509 - 5) % 9 = 0 ∧ ∀ (m : ℕ), m < 5 → (509 - m) % 9 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_subtraction_for_509_divisible_by_9_least_subtraction_509_divisible_by_9_l490_49051


namespace NUMINAMATH_CALUDE_percentage_calculation_l490_49024

theorem percentage_calculation : 
  (0.47 * 1442 - 0.36 * 1412) + 66 = 235.42 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l490_49024


namespace NUMINAMATH_CALUDE_shoe_percentage_gain_l490_49012

/-- Prove that the percentage gain on the selling price of a shoe is approximately 16.67% -/
theorem shoe_percentage_gain :
  let manufacturing_cost : ℝ := 210
  let transportation_cost_per_100 : ℝ := 500
  let selling_price : ℝ := 258
  let total_cost : ℝ := manufacturing_cost + transportation_cost_per_100 / 100
  let gain : ℝ := selling_price - total_cost
  let percentage_gain : ℝ := gain / selling_price * 100
  ∃ ε > 0, abs (percentage_gain - 16.67) < ε :=
by sorry

end NUMINAMATH_CALUDE_shoe_percentage_gain_l490_49012


namespace NUMINAMATH_CALUDE_solution_difference_l490_49041

theorem solution_difference (r s : ℝ) : 
  (∀ x, x ≠ 3 → (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) →
  r ≠ s →
  r > s →
  r - s = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l490_49041


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l490_49058

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_increasing : increasing_function f)
  (h_f_0 : f 0 = -2)
  (h_f_3 : f 3 = 2) :
  {x : ℝ | |f (x + 1)| ≥ 2} = {x : ℝ | x ≤ -1 ∨ x ≥ 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l490_49058


namespace NUMINAMATH_CALUDE_committee_formations_count_l490_49008

/-- The number of ways to form a committee of 5 members from a club of 15 people,
    where the committee must include exactly 2 designated roles and 3 additional members. -/
def committeeFormations (clubSize : ℕ) (committeeSize : ℕ) (designatedRoles : ℕ) (additionalMembers : ℕ) : ℕ :=
  (clubSize * (clubSize - 1)) * Nat.choose (clubSize - designatedRoles) additionalMembers

/-- Theorem stating that the number of committee formations
    for the given conditions is 60060. -/
theorem committee_formations_count :
  committeeFormations 15 5 2 3 = 60060 := by
  sorry

end NUMINAMATH_CALUDE_committee_formations_count_l490_49008


namespace NUMINAMATH_CALUDE_infinite_series_sum_l490_49084

/-- The sum of the infinite series Σ(k^2 / 3^k) from k=1 to infinity is equal to 4 -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (k : ℝ)^2 / 3^k) = 4 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l490_49084


namespace NUMINAMATH_CALUDE_thirty_five_million_scientific_notation_l490_49050

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℕ) : ℝ × ℤ :=
  sorry

theorem thirty_five_million_scientific_notation :
  scientific_notation 35000000 = (3.5, 7) :=
sorry

end NUMINAMATH_CALUDE_thirty_five_million_scientific_notation_l490_49050


namespace NUMINAMATH_CALUDE_concentric_circles_chord_count_l490_49043

theorem concentric_circles_chord_count
  (angle_ABC : ℝ)
  (is_tangent : Bool)
  (h1 : angle_ABC = 60)
  (h2 : is_tangent = true) :
  ∃ n : ℕ, n * angle_ABC = 180 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_chord_count_l490_49043


namespace NUMINAMATH_CALUDE_parallelogram_base_l490_49022

theorem parallelogram_base (area height base : ℝ) : 
  area = 336 ∧ height = 24 ∧ area = base * height → base = 14 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l490_49022


namespace NUMINAMATH_CALUDE_common_root_of_quadratic_equations_l490_49021

theorem common_root_of_quadratic_equations (x : ℚ) :
  (6 * x^2 + 5 * x - 1 = 0) ∧ (18 * x^2 + 41 * x - 7 = 0) → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_common_root_of_quadratic_equations_l490_49021


namespace NUMINAMATH_CALUDE_compare_a_and_b_l490_49073

theorem compare_a_and_b (a b : ℝ) (h : 5 * (a - 1) = b + a^2) : a > b := by
  sorry

end NUMINAMATH_CALUDE_compare_a_and_b_l490_49073


namespace NUMINAMATH_CALUDE_xiao_ying_score_l490_49009

/-- Given an average score and a student's score relative to the average,
    calculate the student's actual score. -/
def calculate_score (average : ℕ) (relative_score : ℤ) : ℕ :=
  (average : ℤ) + relative_score |>.toNat

/-- The problem statement -/
theorem xiao_ying_score :
  let average_score : ℕ := 83
  let xiao_ying_relative_score : ℤ := -3
  calculate_score average_score xiao_ying_relative_score = 80 := by
  sorry

#eval calculate_score 83 (-3)

end NUMINAMATH_CALUDE_xiao_ying_score_l490_49009


namespace NUMINAMATH_CALUDE_ordering_abc_l490_49053

theorem ordering_abc : ∃ (a b c : ℝ), 
  a = Real.sqrt 1.2 ∧ 
  b = Real.exp 0.1 ∧ 
  c = 1 + Real.log 1.1 ∧ 
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ordering_abc_l490_49053


namespace NUMINAMATH_CALUDE_share_calculation_l490_49038

theorem share_calculation (total : ℝ) (a b c : ℝ) : 
  total = 500 →
  a = (2/3) * (b + c) →
  b = (2/3) * (a + c) →
  a + b + c = total →
  a = 200 := by
sorry

end NUMINAMATH_CALUDE_share_calculation_l490_49038


namespace NUMINAMATH_CALUDE_prob_at_least_7_heads_theorem_l490_49017

/-- A fair coin is flipped 10 times. -/
def total_flips : ℕ := 10

/-- The number of heads required for the event. -/
def min_heads : ℕ := 7

/-- The number of fixed heads at the end. -/
def fixed_heads : ℕ := 2

/-- The probability of getting heads on a single flip of a fair coin. -/
def prob_heads : ℚ := 1/2

/-- The probability of getting at least 7 heads in 10 flips, given that the last two are heads. -/
def prob_at_least_7_heads_given_last_2_heads : ℚ := 93/256

/-- 
Theorem: The probability of getting at least 7 heads in 10 flips of a fair coin, 
given that the last two flips are heads, is equal to 93/256.
-/
theorem prob_at_least_7_heads_theorem : 
  prob_at_least_7_heads_given_last_2_heads = 93/256 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_7_heads_theorem_l490_49017


namespace NUMINAMATH_CALUDE_inequality_solution_l490_49055

theorem inequality_solution :
  {x : ℝ | (x^2 - 9) / (x^2 - 16) > 0} = {x : ℝ | x < -4 ∨ x > 4} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l490_49055


namespace NUMINAMATH_CALUDE_polar_to_parabola_l490_49004

/-- The polar equation r = 1 / (1 - sin θ) represents a parabola -/
theorem polar_to_parabola :
  ∃ (x y : ℝ), (∃ (r θ : ℝ), r = 1 / (1 - Real.sin θ) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  x^2 = 1 + 2*y := by
  sorry

end NUMINAMATH_CALUDE_polar_to_parabola_l490_49004


namespace NUMINAMATH_CALUDE_election_percentage_l490_49030

theorem election_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) 
  (h1 : winner_votes = 744)
  (h2 : margin = 288)
  (h3 : total_votes = winner_votes + (winner_votes - margin)) :
  (winner_votes : ℚ) / total_votes * 100 = 62 := by
  sorry

end NUMINAMATH_CALUDE_election_percentage_l490_49030


namespace NUMINAMATH_CALUDE_circle_radius_with_modified_area_formula_l490_49025

/-- Given a circle with a modified area formula, prove that its radius is 10√2 units. -/
theorem circle_radius_with_modified_area_formula 
  (area : ℝ) 
  (k : ℝ) 
  (h1 : area = 100 * Real.pi)
  (h2 : k = 0.5)
  (h3 : ∀ r, Real.pi * k * r^2 = area) :
  ∃ r, r = 10 * Real.sqrt 2 ∧ Real.pi * k * r^2 = area := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_with_modified_area_formula_l490_49025
