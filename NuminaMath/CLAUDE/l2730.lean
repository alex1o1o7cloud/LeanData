import Mathlib

namespace NUMINAMATH_CALUDE_uncle_jude_cookies_l2730_273022

/-- The number of cookies Uncle Jude baked -/
def total_cookies : ℕ := 256

/-- The number of cookies given to Tim -/
def tim_cookies : ℕ := 15

/-- The number of cookies given to Mike -/
def mike_cookies : ℕ := 23

/-- The number of cookies kept in the fridge -/
def fridge_cookies : ℕ := 188

/-- The number of cookies given to Anna -/
def anna_cookies : ℕ := 2 * tim_cookies

theorem uncle_jude_cookies : 
  total_cookies = tim_cookies + mike_cookies + anna_cookies + fridge_cookies := by
  sorry

end NUMINAMATH_CALUDE_uncle_jude_cookies_l2730_273022


namespace NUMINAMATH_CALUDE_power_equality_l2730_273080

theorem power_equality (q : ℕ) : 64^4 = 8^q → q = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2730_273080


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2730_273082

theorem trigonometric_equation_solution (k : ℤ) :
  let x₁ := π / 60 + k * π / 10
  let x₂ := -π / 24 - k * π / 4
  (∀ x, x = x₁ ∨ x = x₂ → (Real.sin (3 * x) + Real.sqrt 3 * Real.cos (3 * x))^2 - 2 * Real.cos (14 * x) = 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2730_273082


namespace NUMINAMATH_CALUDE_largest_lcm_with_15_l2730_273023

theorem largest_lcm_with_15 :
  let lcm_list := [lcm 15 3, lcm 15 5, lcm 15 6, lcm 15 9, lcm 15 10, lcm 15 12]
  List.maximum lcm_list = some 60 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_15_l2730_273023


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_four_l2730_273064

/-- Original parabola function -/
def original_parabola (x : ℝ) : ℝ := (x + 3)^2 - 2

/-- Transformed parabola function -/
def transformed_parabola (x : ℝ) : ℝ := -(x - 2)^2 + 2

/-- The zeros of the transformed parabola -/
def zeros : Set ℝ := {x | transformed_parabola x = 0}

theorem sum_of_zeros_is_four :
  ∃ (a b : ℝ), a ∈ zeros ∧ b ∈ zeros ∧ a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_four_l2730_273064


namespace NUMINAMATH_CALUDE_newspaper_conference_max_both_l2730_273068

theorem newspaper_conference_max_both (total : ℕ) (writers : ℕ) (editors : ℕ) (neither : ℕ) (both : ℕ) :
  total = 90 →
  writers = 45 →
  editors > 38 →
  neither = 2 * both →
  total = writers + editors + neither - both →
  both ≤ 4 ∧ (∃ (e : ℕ), editors = 38 + e ∧ both = 4) :=
by sorry

end NUMINAMATH_CALUDE_newspaper_conference_max_both_l2730_273068


namespace NUMINAMATH_CALUDE_sin_pi_minus_alpha_l2730_273094

theorem sin_pi_minus_alpha (α : Real) :
  (∃ (x y : Real), x = -4 ∧ y = 3 ∧ x = r * Real.cos α ∧ y = r * Real.sin α ∧ r > 0) →
  Real.sin (Real.pi - α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_minus_alpha_l2730_273094


namespace NUMINAMATH_CALUDE_max_baggies_of_cookies_l2730_273063

def chocolateChipCookies : ℕ := 23
def oatmealCookies : ℕ := 25
def cookiesPerBaggie : ℕ := 6

theorem max_baggies_of_cookies : 
  (chocolateChipCookies + oatmealCookies) / cookiesPerBaggie = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_baggies_of_cookies_l2730_273063


namespace NUMINAMATH_CALUDE_saree_price_calculation_l2730_273028

theorem saree_price_calculation (P : ℝ) : 
  (P * (1 - 0.20) * (1 - 0.05) = 133) → P = 175 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l2730_273028


namespace NUMINAMATH_CALUDE_substitution_method_correctness_l2730_273083

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 2 * x - y = 5
def equation2 (x y : ℝ) : Prop := y = 1 + x

-- Define the correct substitution
def correct_substitution (x : ℝ) : Prop := 2 * x - 1 - x = 5

-- Theorem statement
theorem substitution_method_correctness :
  ∀ x y : ℝ, equation1 x y ∧ equation2 x y → correct_substitution x :=
by sorry

end NUMINAMATH_CALUDE_substitution_method_correctness_l2730_273083


namespace NUMINAMATH_CALUDE_initial_concentration_proof_l2730_273048

/-- Proves that the initial concentration of a hydrochloric acid solution is 20%
    given the conditions of the problem. -/
theorem initial_concentration_proof (
  initial_amount : ℝ)
  (drained_amount : ℝ)
  (added_concentration : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_amount = 300)
  (h2 : drained_amount = 25)
  (h3 : added_concentration = 80 / 100)
  (h4 : final_concentration = 25 / 100)
  : ∃ (initial_concentration : ℝ),
    initial_concentration = 20 / 100 ∧
    (initial_amount - drained_amount) * initial_concentration +
    drained_amount * added_concentration =
    initial_amount * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_initial_concentration_proof_l2730_273048


namespace NUMINAMATH_CALUDE_ellipse_properties_l2730_273098

noncomputable section

-- Define the ellipse parameters
def a : ℝ := 2
def b : ℝ := Real.sqrt 3
def c : ℝ := 1

-- Define the eccentricity
def e : ℝ := 1 / 2

-- Define the maximum area of triangle PAB
def max_area : ℝ := 2 * Real.sqrt 3

-- Define the coordinates of point D
def d_x : ℝ := -11 / 8
def d_y : ℝ := 0

-- Define the constant dot product value
def constant_dot_product : ℝ := -135 / 64

-- Theorem statement
theorem ellipse_properties :
  (a > b ∧ b > 0) ∧
  (e = c / a) ∧
  (max_area = a * b) ∧
  (a^2 = b^2 + c^2) →
  (∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∀ t : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ,
    (x₁^2 / a^2 + y₁^2 / b^2 = 1) ∧
    (x₂^2 / a^2 + y₂^2 / b^2 = 1) ∧
    (x₁ = t * y₁ - 1) ∧
    (x₂ = t * y₂ - 1) ∧
    ((x₁ - d_x) * (x₂ - d_x) + y₁ * y₂ = constant_dot_product)) :=
by sorry

end

end NUMINAMATH_CALUDE_ellipse_properties_l2730_273098


namespace NUMINAMATH_CALUDE_selling_price_ratio_l2730_273033

theorem selling_price_ratio (CP : ℝ) (SP1 SP2 : ℝ) 
  (h1 : SP1 = CP + 0.5 * CP) 
  (h2 : SP2 = CP + 3 * CP) : 
  SP2 / SP1 = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l2730_273033


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l2730_273061

/-- Calculates the cost of plastering a rectangular tank's walls and bottom -/
def plasteringCost (length width height rate : ℝ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * rate

/-- Theorem stating the cost of plastering the specific tank -/
theorem tank_plastering_cost :
  plasteringCost 25 12 6 0.75 = 558 := by
  sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l2730_273061


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2730_273054

/-- An isosceles triangle with side lengths 3 and 6 has a perimeter of 15 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 6 → b = 6 → c = 3 →
  (a = b ∨ a = c ∨ b = c) →  -- Isosceles condition
  a + b + c = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2730_273054


namespace NUMINAMATH_CALUDE_wild_sorghum_and_corn_different_species_reproductive_isolation_wild_sorghum_corn_l2730_273059

/-- Represents a plant species -/
structure Species where
  name : String
  chromosomes : Nat

/-- Defines reproductive isolation between two species -/
def reproductiveIsolation (s1 s2 : Species) : Prop :=
  s1.chromosomes ≠ s2.chromosomes

/-- Defines whether two species are the same -/
def sameSpecies (s1 s2 : Species) : Prop :=
  s1.chromosomes = s2.chromosomes ∧ ¬reproductiveIsolation s1 s2

/-- Wild sorghum species -/
def wildSorghum : Species :=
  { name := "Wild Sorghum", chromosomes := 22 }

/-- Corn species -/
def corn : Species :=
  { name := "Corn", chromosomes := 20 }

/-- Theorem stating that wild sorghum and corn are not the same species -/
theorem wild_sorghum_and_corn_different_species :
  ¬sameSpecies wildSorghum corn :=
by
  sorry

/-- Theorem stating that there is reproductive isolation between wild sorghum and corn -/
theorem reproductive_isolation_wild_sorghum_corn :
  reproductiveIsolation wildSorghum corn :=
by
  sorry

end NUMINAMATH_CALUDE_wild_sorghum_and_corn_different_species_reproductive_isolation_wild_sorghum_corn_l2730_273059


namespace NUMINAMATH_CALUDE_quadratic_root_bounds_l2730_273047

theorem quadratic_root_bounds (a b : ℝ) (α β : ℝ) : 
  (α^2 + a*α + b = 0) → 
  (β^2 + a*β + b = 0) → 
  (∀ x, x^2 + a*x + b = 0 → x = α ∨ x = β) →
  (|α| < 2 ∧ |β| < 2 ↔ 2*|a| < 4 + b ∧ |b| < 4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_bounds_l2730_273047


namespace NUMINAMATH_CALUDE_bc_length_l2730_273044

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
def is_right_triangle (X Y Z : ℝ × ℝ) : Prop := sorry

-- Define the lengths
def length (X Y : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem bc_length 
  (h1 : is_right_triangle A B C)
  (h2 : is_right_triangle A B D)
  (h3 : length A D = 50)
  (h4 : length C D = 25)
  (h5 : length A C = 20)
  (h6 : length A B = 15) :
  length B C = 25 :=
sorry

end NUMINAMATH_CALUDE_bc_length_l2730_273044


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2730_273058

theorem sum_of_squares_lower_bound (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2730_273058


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2730_273005

theorem circle_area_ratio (C D : Real) (r_C r_D : ℝ) : 
  (60 / 360) * (2 * Real.pi * r_C) = (40 / 360) * (2 * Real.pi * r_D) →
  (Real.pi * r_C^2) / (Real.pi * r_D^2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2730_273005


namespace NUMINAMATH_CALUDE_prob_king_queen_heart_l2730_273070

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of Queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Number of Hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of King of Hearts in a standard deck -/
def NumKingOfHearts : ℕ := 1

/-- Probability of drawing a King, then a Queen, then a Heart from a standard 52-card deck -/
theorem prob_king_queen_heart : 
  (NumKings * (NumQueens - 1) * NumHearts + 
   NumKingOfHearts * NumQueens * (NumHearts - 1) + 
   NumKingOfHearts * (NumQueens - 1) * (NumHearts - 1)) / 
  (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2)) = 67 / 44200 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_queen_heart_l2730_273070


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l2730_273032

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (plane_perp : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (line_perp_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (line_perp : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (m n : Line)
  (h1 : line_perp_plane m α)
  (h2 : line_perp_plane n β)
  (h3 : plane_perp α β) :
  line_perp m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l2730_273032


namespace NUMINAMATH_CALUDE_range_of_r_l2730_273084

-- Define the function r(x)
def r (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

-- State the theorem
theorem range_of_r :
  Set.range (fun x : ℝ => r x) = Set.Ici 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_r_l2730_273084


namespace NUMINAMATH_CALUDE_perpendicular_bisector_y_intercept_range_l2730_273073

/-- Given two distinct points on a parabola y = 2x², prove that the y-intercept of their perpendicular bisector with slope 2 is greater than 9/32. -/
theorem perpendicular_bisector_y_intercept_range 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂))
  (h_parabola₁ : y₁ = 2 * x₁^2)
  (h_parabola₂ : y₂ = 2 * x₂^2)
  (b : ℝ) 
  (h_perpendicular_bisector : ∃ (m : ℝ), 
    y₁ = -1/(2*m) * x₁ + b + 1/(4*m) ∧ 
    y₂ = -1/(2*m) * x₂ + b + 1/(4*m) ∧ 
    m = 2) : 
  b > 9/32 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_y_intercept_range_l2730_273073


namespace NUMINAMATH_CALUDE_cistern_length_l2730_273055

theorem cistern_length (width : ℝ) (depth : ℝ) (wet_area : ℝ) (length : ℝ) : 
  width = 4 →
  depth = 1.25 →
  wet_area = 49 →
  wet_area = (length * width) + (2 * length * depth) + (2 * width * depth) →
  length = 6 := by
sorry

end NUMINAMATH_CALUDE_cistern_length_l2730_273055


namespace NUMINAMATH_CALUDE_inlet_pipe_rate_l2730_273046

/-- Prove that the inlet pipe rate is 3 cubic inches/min given the tank and pipe conditions -/
theorem inlet_pipe_rate (tank_volume : ℝ) (outlet_rate1 outlet_rate2 : ℝ) (empty_time : ℝ) :
  tank_volume = 51840 ∧ 
  outlet_rate1 = 9 ∧ 
  outlet_rate2 = 6 ∧ 
  empty_time = 4320 →
  ∃ inlet_rate : ℝ, 
    inlet_rate = 3 ∧ 
    (outlet_rate1 + outlet_rate2 - inlet_rate) * empty_time = tank_volume :=
by
  sorry

end NUMINAMATH_CALUDE_inlet_pipe_rate_l2730_273046


namespace NUMINAMATH_CALUDE_line_parallel_plane_theorem_l2730_273069

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the contained relation for lines and planes
variable (containedInPlane : Line → Plane → Prop)

-- Theorem statement
theorem line_parallel_plane_theorem 
  (a b : Line) (α : Plane) :
  parallelLine a b → parallelLinePlane a α →
  containedInPlane b α ∨ parallelLinePlane b α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_theorem_l2730_273069


namespace NUMINAMATH_CALUDE_cos_A_right_triangle_l2730_273097

theorem cos_A_right_triangle (adjacent hypotenuse : ℝ) 
  (h1 : adjacent = 5)
  (h2 : hypotenuse = 13)
  (h3 : adjacent > 0)
  (h4 : hypotenuse > 0)
  (h5 : adjacent < hypotenuse) : 
  Real.cos (Real.arccos (adjacent / hypotenuse)) = 5 / 13 := by
sorry

end NUMINAMATH_CALUDE_cos_A_right_triangle_l2730_273097


namespace NUMINAMATH_CALUDE_square_root_meaningful_l2730_273089

theorem square_root_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) → x ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_square_root_meaningful_l2730_273089


namespace NUMINAMATH_CALUDE_mass_percentage_cl_l2730_273091

/-- Given a compound where the mass percentage of Cl is 92.11%,
    prove that the mass percentage of Cl in the compound is 92.11%. -/
theorem mass_percentage_cl (compound_mass_percentage : ℝ) 
  (h : compound_mass_percentage = 92.11) : 
  compound_mass_percentage = 92.11 := by
  sorry

end NUMINAMATH_CALUDE_mass_percentage_cl_l2730_273091


namespace NUMINAMATH_CALUDE_expression_simplification_l2730_273065

theorem expression_simplification (a b : ℝ) 
  (h : (a - 2)^2 + Real.sqrt (b + 1) = 0) :
  (a^2 - 2*a*b + b^2) / (a^2 - b^2) / ((a^2 - a*b) / a) - 2 / (a + b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2730_273065


namespace NUMINAMATH_CALUDE_g_derivative_l2730_273026

noncomputable def g (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem g_derivative (x : ℝ) : 
  deriv g x = Real.exp x * Real.sin x + Real.exp x * Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_g_derivative_l2730_273026


namespace NUMINAMATH_CALUDE_tate_additional_tickets_l2730_273000

/-- The number of additional tickets Tate bought -/
def additional_tickets : ℕ := sorry

theorem tate_additional_tickets : 
  let initial_tickets : ℕ := 32
  let total_tickets : ℕ := initial_tickets + additional_tickets
  let peyton_tickets : ℕ := total_tickets / 2
  51 = total_tickets + peyton_tickets →
  additional_tickets = 2 := by sorry

end NUMINAMATH_CALUDE_tate_additional_tickets_l2730_273000


namespace NUMINAMATH_CALUDE_fraction_value_l2730_273015

theorem fraction_value (a b c d : ℚ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 4 * d) :
  a * c / (b * d) = 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2730_273015


namespace NUMINAMATH_CALUDE_identity_function_satisfies_equation_l2730_273087

theorem identity_function_satisfies_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) →
  (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_identity_function_satisfies_equation_l2730_273087


namespace NUMINAMATH_CALUDE_ceiling_minus_x_eq_half_l2730_273072

theorem ceiling_minus_x_eq_half (x : ℝ) (h : x - ⌊x⌋ = 1/2) : ⌈x⌉ - x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_eq_half_l2730_273072


namespace NUMINAMATH_CALUDE_large_long_furred_brown_dogs_l2730_273053

/-- Represents the characteristics of dogs in a kennel -/
structure DogKennel where
  total : ℕ
  longFurred : ℕ
  brown : ℕ
  neitherLongFurredNorBrown : ℕ
  large : ℕ
  small : ℕ
  smallAndBrown : ℕ
  onlyLargeAndLongFurred : ℕ

/-- Theorem stating the number of large, long-furred, brown dogs -/
theorem large_long_furred_brown_dogs (k : DogKennel)
  (h1 : k.total = 60)
  (h2 : k.longFurred = 35)
  (h3 : k.brown = 25)
  (h4 : k.neitherLongFurredNorBrown = 10)
  (h5 : k.large = 30)
  (h6 : k.small = 30)
  (h7 : k.smallAndBrown = 14)
  (h8 : k.onlyLargeAndLongFurred = 7) :
  ∃ n : ℕ, n = 6 ∧ n = k.large - k.onlyLargeAndLongFurred - (k.brown - k.smallAndBrown) :=
by sorry


end NUMINAMATH_CALUDE_large_long_furred_brown_dogs_l2730_273053


namespace NUMINAMATH_CALUDE_square_less_than_four_times_l2730_273062

theorem square_less_than_four_times : ∀ n : ℤ, n^2 < 4*n ↔ n = 1 ∨ n = 2 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_four_times_l2730_273062


namespace NUMINAMATH_CALUDE_cassette_tape_cost_cassette_tape_cost_proof_l2730_273041

theorem cassette_tape_cost 
  (initial_amount : ℝ) 
  (headphone_cost : ℝ) 
  (num_tapes : ℕ) 
  (remaining_amount : ℝ) : ℝ :=
  let total_tape_cost := initial_amount - headphone_cost - remaining_amount
  total_tape_cost / num_tapes

#check cassette_tape_cost 50 25 2 7 = 9

theorem cassette_tape_cost_proof 
  (initial_amount : ℝ)
  (headphone_cost : ℝ)
  (num_tapes : ℕ)
  (remaining_amount : ℝ)
  (h1 : initial_amount = 50)
  (h2 : headphone_cost = 25)
  (h3 : num_tapes = 2)
  (h4 : remaining_amount = 7) :
  cassette_tape_cost initial_amount headphone_cost num_tapes remaining_amount = 9 := by
  sorry

end NUMINAMATH_CALUDE_cassette_tape_cost_cassette_tape_cost_proof_l2730_273041


namespace NUMINAMATH_CALUDE_logarithm_equation_solution_l2730_273077

theorem logarithm_equation_solution (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (∀ x : ℝ, x > 0 → 5 * (Real.log x / Real.log a)^2 + 2 * (Real.log x / Real.log b)^2 = 
    (10 * (Real.log x)^2) / (Real.log a * Real.log b) + (Real.log x)^2) →
  b = a^(2 / (5 + Real.sqrt 17)) ∨ b = a^(2 / (5 - Real.sqrt 17)) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_equation_solution_l2730_273077


namespace NUMINAMATH_CALUDE_solution_to_system_l2730_273035

theorem solution_to_system (x y z : ℝ) :
  3 * (x^2 + y^2 + z^2) = 1 →
  x^2*y^2 + y^2*z^2 + z^2*x^2 = x*y*z*(x + y + z)^3 →
  ((x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ (x = -1/3 ∧ y = -1/3 ∧ z = -1/3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_to_system_l2730_273035


namespace NUMINAMATH_CALUDE_bread_pieces_theorem_l2730_273066

/-- The number of pieces a slice of bread becomes when torn in half twice -/
def pieces_per_slice : ℕ := 4

/-- The number of slices of bread used -/
def num_slices : ℕ := 2

/-- The total number of bread pieces after tearing -/
def total_pieces : ℕ := num_slices * pieces_per_slice

theorem bread_pieces_theorem : total_pieces = 8 := by
  sorry

end NUMINAMATH_CALUDE_bread_pieces_theorem_l2730_273066


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l2730_273019

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersection_of_three_lines (x y : ℝ) :
  y = 4 * x + 5 ∧ 
  y = -3 * x + 10 ∧ 
  y = 2 * x + k →
  k = 45 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l2730_273019


namespace NUMINAMATH_CALUDE_point_B_coordinates_l2730_273052

-- Define the vector type
def Vec := ℝ × ℝ

-- Define point A
def A : Vec := (-1, -5)

-- Define vector a
def a : Vec := (2, 3)

-- Define vector AB in terms of a
def AB : Vec := (3 * a.1, 3 * a.2)

-- Define point B
def B : Vec := (A.1 + AB.1, A.2 + AB.2)

-- Theorem statement
theorem point_B_coordinates : B = (5, 4) := by
  sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l2730_273052


namespace NUMINAMATH_CALUDE_even_digits_in_base9_567_l2730_273007

/-- Converts a natural number from base 10 to base 9 -/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

theorem even_digits_in_base9_567 : 
  countEvenDigits (toBase9 567) = 2 :=
sorry

end NUMINAMATH_CALUDE_even_digits_in_base9_567_l2730_273007


namespace NUMINAMATH_CALUDE_scientific_notation_3462_23_l2730_273074

theorem scientific_notation_3462_23 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 3462.23 = a * (10 : ℝ) ^ n ∧ a = 3.46223 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_3462_23_l2730_273074


namespace NUMINAMATH_CALUDE_symmetry_center_x_value_l2730_273006

/-- Given a function f(x) = 1/2 * sin(ω*x + π/6) with ω > 0, 
    if its graph is tangent to a line y = m with distance π between adjacent tangent points,
    and A(x₀, y₀) is a symmetry center of y = f(x) with x₀ ∈ [0, π/2],
    then x₀ = 5π/12 -/
theorem symmetry_center_x_value (ω : ℝ) (m : ℝ) (x₀ : ℝ) (y₀ : ℝ) :
  ω > 0 →
  (∃ (k : ℤ), x₀ = k * π - π / 12) →
  x₀ ∈ Set.Icc 0 (π / 2) →
  (∀ (x : ℝ), (1 / 2) * Real.sin (ω * x + π / 6) = m → 
    ∃ (n : ℤ), x = n * π / ω) →
  x₀ = 5 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_center_x_value_l2730_273006


namespace NUMINAMATH_CALUDE_school_distance_l2730_273031

theorem school_distance (speed_to_school : ℝ) (speed_from_school : ℝ) (total_time : ℝ) 
  (h1 : speed_to_school = 3)
  (h2 : speed_from_school = 2)
  (h3 : total_time = 5) :
  ∃ (distance : ℝ), distance = 6 ∧ 
    (distance / speed_to_school + distance / speed_from_school = total_time) :=
by
  sorry

end NUMINAMATH_CALUDE_school_distance_l2730_273031


namespace NUMINAMATH_CALUDE_total_repair_cost_is_50_95_l2730_273071

def tire_repair_cost (num_tires : ℕ) (cost_per_tire : ℚ) (sales_tax : ℚ) 
                     (discount_rate : ℚ) (discount_valid : Bool) (city_fee : ℚ) : ℚ :=
  let base_cost := num_tires * cost_per_tire
  let tax_cost := num_tires * sales_tax
  let fee_cost := num_tires * city_fee
  let discount := if discount_valid then discount_rate * base_cost else 0
  base_cost + tax_cost + fee_cost - discount

theorem total_repair_cost_is_50_95 :
  let car_a_cost := tire_repair_cost 3 7 0.5 0.05 true 2.5
  let car_b_cost := tire_repair_cost 2 8.5 0 0.1 false 2.5
  car_a_cost + car_b_cost = 50.95 := by
sorry

#eval tire_repair_cost 3 7 0.5 0.05 true 2.5 + tire_repair_cost 2 8.5 0 0.1 false 2.5

end NUMINAMATH_CALUDE_total_repair_cost_is_50_95_l2730_273071


namespace NUMINAMATH_CALUDE_right_triangle_identification_l2730_273078

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_identification :
  ¬(is_right_triangle 6 15 17) ∧
  ¬(is_right_triangle 7 12 15) ∧
  is_right_triangle 7 24 25 ∧
  ¬(is_right_triangle 13 15 20) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l2730_273078


namespace NUMINAMATH_CALUDE_probability_point_on_subsegment_l2730_273024

/-- The probability of a randomly chosen point on a segment also lying on its subsegment -/
theorem probability_point_on_subsegment 
  (L ℓ : ℝ) 
  (hL : L = 40) 
  (hℓ : ℓ = 15) 
  (h_pos_L : L > 0) 
  (h_pos_ℓ : ℓ > 0) 
  (h_subsegment : ℓ ≤ L) :
  ℓ / L = 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_point_on_subsegment_l2730_273024


namespace NUMINAMATH_CALUDE_initial_balloon_count_balloon_package_problem_l2730_273038

theorem initial_balloon_count (num_friends : ℕ) (balloons_given_back : ℕ) (final_balloons_per_friend : ℕ) : ℕ :=
  let initial_balloons_per_friend := final_balloons_per_friend + balloons_given_back
  num_friends * initial_balloons_per_friend

theorem balloon_package_problem :
  initial_balloon_count 5 11 39 = 250 := by
  sorry

end NUMINAMATH_CALUDE_initial_balloon_count_balloon_package_problem_l2730_273038


namespace NUMINAMATH_CALUDE_sine_midpoint_inequality_l2730_273079

theorem sine_midpoint_inequality (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : x₁ < π) (h₃ : 0 < x₂) (h₄ : x₂ < π) (h₅ : x₁ ≠ x₂) : 
  (Real.sin x₁ + Real.sin x₂) / 2 < Real.sin ((x₁ + x₂) / 2) := by
  sorry

end NUMINAMATH_CALUDE_sine_midpoint_inequality_l2730_273079


namespace NUMINAMATH_CALUDE_min_seats_for_adjacency_l2730_273040

/-- Represents a row of seats -/
structure SeatRow where
  total_seats : ℕ
  occupied_seats : ℕ

/-- Checks if the next person must sit next to someone -/
def must_sit_next (row : SeatRow) : Prop :=
  ∀ i : ℕ, i < row.total_seats - 1 → (i % 4 = 0 → i < row.occupied_seats * 4)

/-- The main theorem to be proved -/
theorem min_seats_for_adjacency (row : SeatRow) :
  row.total_seats = 150 →
  (∀ r : SeatRow, r.total_seats = 150 → r.occupied_seats < 37 → ¬ must_sit_next r) →
  must_sit_next row →
  row.occupied_seats ≥ 37 :=
sorry

end NUMINAMATH_CALUDE_min_seats_for_adjacency_l2730_273040


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2730_273075

theorem cubic_equation_root (a b : ℚ) : 
  (3 + Real.sqrt 5 : ℝ) ^ 3 + a * (3 + Real.sqrt 5 : ℝ) ^ 2 + b * (3 + Real.sqrt 5 : ℝ) - 20 = 0 → 
  b = -26 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2730_273075


namespace NUMINAMATH_CALUDE_batsman_average_increase_l2730_273086

/-- Represents a batsman's cricket performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  average : Rat

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : Nat) : Rat :=
  (b.totalRuns + runsScored) / (b.innings + 1)

theorem batsman_average_increase 
  (b : Batsman) 
  (h1 : b.innings = 19) 
  (h2 : newAverage b 85 = b.average + 4) 
  (h3 : b.average > 0) : 
  newAverage b 85 = 9 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l2730_273086


namespace NUMINAMATH_CALUDE_negation_of_nonnegative_product_l2730_273021

theorem negation_of_nonnegative_product (a b : ℝ) :
  ¬(a ≥ 0 ∧ b ≥ 0 → a * b ≥ 0) ↔ (a < 0 ∨ b < 0 → a * b < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_nonnegative_product_l2730_273021


namespace NUMINAMATH_CALUDE_log_inequality_l2730_273011

theorem log_inequality (c a b : ℝ) (h1 : 0 < c) (h2 : c < 1) (h3 : b > 1) (h4 : a > b) :
  Real.log c / Real.log a > Real.log c / Real.log b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2730_273011


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2730_273014

theorem complex_equation_solution (m A B : ℝ) :
  (((2 : ℂ) - m * I) / ((1 : ℂ) + 2 * I) = A + B * I) →
  A + B = 0 →
  m = -2/3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2730_273014


namespace NUMINAMATH_CALUDE_third_day_income_l2730_273085

def cab_driver_problem (day1 day2 day3 day4 day5 : ℝ) : Prop :=
  day1 = 300 ∧ 
  day2 = 150 ∧ 
  day4 = 200 ∧ 
  day5 = 600 ∧ 
  (day1 + day2 + day3 + day4 + day5) / 5 = 400

theorem third_day_income (day1 day2 day3 day4 day5 : ℝ) 
  (h : cab_driver_problem day1 day2 day3 day4 day5) : day3 = 750 := by
  sorry

end NUMINAMATH_CALUDE_third_day_income_l2730_273085


namespace NUMINAMATH_CALUDE_sparrow_swallow_weight_system_l2730_273042

theorem sparrow_swallow_weight_system :
  ∀ (x y : ℝ),
    (∃ (sparrow_count swallow_count : ℕ),
      sparrow_count = 5 ∧
      swallow_count = 6 ∧
      (4 * x + y = 5 * y + x) ∧
      (sparrow_count * x + swallow_count * y = 1)) →
    (4 * x + y = 5 * y + x ∧ 5 * x + 6 * y = 1) :=
by sorry

end NUMINAMATH_CALUDE_sparrow_swallow_weight_system_l2730_273042


namespace NUMINAMATH_CALUDE_inequality_proof_l2730_273036

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b * c) + b / (a * c) + c / (a * b) ≥ 2 / a + 2 / b - 2 / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2730_273036


namespace NUMINAMATH_CALUDE_tory_fundraising_problem_l2730_273027

/-- Represents the fundraising problem for Tory's cookie sale --/
theorem tory_fundraising_problem (goal : ℕ) (chocolate_price oatmeal_price sugar_price : ℕ)
  (chocolate_sold oatmeal_sold sugar_sold : ℕ) :
  goal = 250 →
  chocolate_price = 6 →
  oatmeal_price = 5 →
  sugar_price = 4 →
  chocolate_sold = 5 →
  oatmeal_sold = 10 →
  sugar_sold = 15 →
  goal - (chocolate_price * chocolate_sold + oatmeal_price * oatmeal_sold + sugar_price * sugar_sold) = 110 := by
  sorry

#check tory_fundraising_problem

end NUMINAMATH_CALUDE_tory_fundraising_problem_l2730_273027


namespace NUMINAMATH_CALUDE_sons_age_l2730_273009

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2730_273009


namespace NUMINAMATH_CALUDE_nell_initial_cards_l2730_273056

theorem nell_initial_cards (cards_given_to_jeff cards_left : ℕ) 
  (h1 : cards_given_to_jeff = 301)
  (h2 : cards_left = 154) :
  cards_given_to_jeff + cards_left = 455 :=
by sorry

end NUMINAMATH_CALUDE_nell_initial_cards_l2730_273056


namespace NUMINAMATH_CALUDE_g_satisfies_conditions_l2730_273037

/-- A monic polynomial of degree 3 satisfying specific conditions -/
def g (x : ℝ) : ℝ := x^3 + 4*x^2 + 3*x + 6

/-- Theorem stating that g satisfies the given conditions -/
theorem g_satisfies_conditions :
  (∀ x, g x = x^3 + 4*x^2 + 3*x + 6) ∧
  g 0 = 6 ∧
  g 1 = 14 ∧
  g (-1) = 6 :=
by sorry

end NUMINAMATH_CALUDE_g_satisfies_conditions_l2730_273037


namespace NUMINAMATH_CALUDE_largest_k_for_distinct_roots_l2730_273004

theorem largest_k_for_distinct_roots : 
  ∀ k : ℤ, 
  (∃ x y : ℝ, x ≠ y ∧ 
    (k - 2 : ℝ) * x^2 - 4 * x + 4 = 0 ∧ 
    (k - 2 : ℝ) * y^2 - 4 * y + 4 = 0) →
  k ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_distinct_roots_l2730_273004


namespace NUMINAMATH_CALUDE_min_value_ab_l2730_273034

theorem min_value_ab (a b : ℝ) (h : 0 < a ∧ 0 < b) (eq : 1/a + 4/b = Real.sqrt (a*b)) : 
  4 ≤ a * b ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 1/a₀ + 4/b₀ = Real.sqrt (a₀*b₀) ∧ a₀ * b₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l2730_273034


namespace NUMINAMATH_CALUDE_tan_double_angle_l2730_273051

theorem tan_double_angle (x : ℝ) (h : Real.tan (Real.pi - x) = 3 / 4) : 
  Real.tan (2 * x) = -24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l2730_273051


namespace NUMINAMATH_CALUDE_inequality_proof_l2730_273018

theorem inequality_proof (a b x : ℝ) (ha : a > 0) (hb : b > 0) :
  a * b ≤ (a * Real.sin x ^ 2 + b * Real.cos x ^ 2) * (a * Real.cos x ^ 2 + b * Real.sin x ^ 2) ∧
  (a * Real.sin x ^ 2 + b * Real.cos x ^ 2) * (a * Real.cos x ^ 2 + b * Real.sin x ^ 2) ≤ (a + b) ^ 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2730_273018


namespace NUMINAMATH_CALUDE_triangle_area_condition_l2730_273096

/-- The area of the triangle formed by the line x - 2y + 2m = 0 and the coordinate axes is not less than 1 if and only if m ∈ (-∞, -1] ∪ [1, +∞) -/
theorem triangle_area_condition (m : ℝ) : 
  (∃ (x y : ℝ), x - 2*y + 2*m = 0 ∧ 
   (1/2) * |x| * |y| ≥ 1) ↔ 
  (m ≤ -1 ∨ m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_condition_l2730_273096


namespace NUMINAMATH_CALUDE_max_u_coordinate_is_two_l2730_273092

-- Define the transformation
def transform (x y : ℝ) : ℝ × ℝ :=
  (x^2 + y^2, x - y)

-- Define the unit square vertices
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (0, 1)

-- Define the set of points in the unit square
def unitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Theorem: The maximum u-coordinate of the transformed unit square is 2
theorem max_u_coordinate_is_two :
  ∃ (p : ℝ × ℝ), p ∈ unitSquare ∧
    (∀ (q : ℝ × ℝ), q ∈ unitSquare →
      (transform p.1 p.2).1 ≥ (transform q.1 q.2).1) ∧
    (transform p.1 p.2).1 = 2 :=
  sorry

end NUMINAMATH_CALUDE_max_u_coordinate_is_two_l2730_273092


namespace NUMINAMATH_CALUDE_tangent_line_problem_l2730_273010

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x^2
def g (b : ℝ) (x : ℝ) : ℝ := Real.sin x + b * x

def is_tangent_at (l : ℝ → ℝ) (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  l x₀ = f x₀ ∧ (deriv l) x₀ = (deriv f) x₀

theorem tangent_line_problem (a b : ℝ) (l : ℝ → ℝ) :
  is_tangent_at l (f a) 0 →
  is_tangent_at l (g b) (Real.pi / 2) →
  (a = 1 ∧ b = 1) ∧
  (∀ x, l x = x + 1) ∧
  (∀ x, Real.exp x + x^2 - x - Real.sin x > 0) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_problem_l2730_273010


namespace NUMINAMATH_CALUDE_sum_divisors_bound_l2730_273050

/-- σ(n) is the sum of the divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- ω(n) is the number of distinct prime divisors of n -/
def omega (n : ℕ+) : ℕ := sorry

/-- The sum of divisors of n is less than n multiplied by one more than
    the number of its distinct prime divisors -/
theorem sum_divisors_bound (n : ℕ+) : sigma n < n * (omega n + 1) := by sorry

end NUMINAMATH_CALUDE_sum_divisors_bound_l2730_273050


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l2730_273045

theorem sum_of_four_numbers : 2367 + 3672 + 6723 + 7236 = 19998 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l2730_273045


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_l2730_273016

theorem sqrt_50_between_consecutive_integers :
  ∃ (n : ℕ), (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1 : ℝ) ∧ n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_l2730_273016


namespace NUMINAMATH_CALUDE_largest_square_from_wire_l2730_273090

/-- Given a wire of length 28 centimeters forming the largest possible square,
    the length of one side of the square is 7 centimeters. -/
theorem largest_square_from_wire (wire_length : ℝ) (side_length : ℝ) :
  wire_length = 28 →
  side_length * 4 = wire_length →
  side_length = 7 := by sorry

end NUMINAMATH_CALUDE_largest_square_from_wire_l2730_273090


namespace NUMINAMATH_CALUDE_company_dividend_percentage_l2730_273001

/-- Calculates the dividend percentage paid by a company given the face value of a share,
    the investor's return on investment, and the investor's purchase price per share. -/
def dividend_percentage (face_value : ℚ) (roi : ℚ) (purchase_price : ℚ) : ℚ :=
  (roi * purchase_price / face_value) * 100

/-- Theorem stating that under the given conditions, the dividend percentage is 18.5% -/
theorem company_dividend_percentage :
  let face_value : ℚ := 50
  let roi : ℚ := 25 / 100
  let purchase_price : ℚ := 37
  dividend_percentage face_value roi purchase_price = 185 / 10 := by
  sorry

end NUMINAMATH_CALUDE_company_dividend_percentage_l2730_273001


namespace NUMINAMATH_CALUDE_sum_of_possible_y_values_l2730_273020

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- Two angles of the triangle
  angle1 : ℝ
  angle2 : ℝ
  -- The triangle is isosceles
  isIsosceles : angle1 = angle2 ∨ angle1 = 180 - angle1 - angle2 ∨ angle2 = 180 - angle1 - angle2
  -- The sum of angles in a triangle is 180°
  sumOfAngles : angle1 + angle2 + (180 - angle1 - angle2) = 180

-- Theorem statement
theorem sum_of_possible_y_values (t : IsoscelesTriangle) (h1 : t.angle1 = 40 ∨ t.angle2 = 40) :
  ∃ y1 y2 : ℝ, (y1 = t.angle1 ∨ y1 = t.angle2) ∧ 
             (y2 = t.angle1 ∨ y2 = t.angle2) ∧
             y1 ≠ y2 ∧
             y1 + y2 = 140 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_y_values_l2730_273020


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l2730_273017

theorem product_mod_seventeen : (2021 * 2023 * 2025 * 2027 * 2029) % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l2730_273017


namespace NUMINAMATH_CALUDE_sarahs_test_score_l2730_273002

theorem sarahs_test_score 
  (hunter_score : ℕ) 
  (john_score : ℕ) 
  (grant_score : ℕ) 
  (sarah_score : ℕ) 
  (hunter_score_val : hunter_score = 45)
  (john_score_def : john_score = 2 * hunter_score)
  (grant_score_def : grant_score = john_score + 10)
  (sarah_score_def : sarah_score = grant_score - 5) :
  sarah_score = 95 := by
sorry

end NUMINAMATH_CALUDE_sarahs_test_score_l2730_273002


namespace NUMINAMATH_CALUDE_olga_aquarium_fish_count_l2730_273093

/-- The number of fish in Olga's aquarium -/
def fish_count (yellow blue green : ℕ) : ℕ := yellow + blue + green

/-- Theorem stating the total number of fish in Olga's aquarium -/
theorem olga_aquarium_fish_count :
  ∀ (yellow blue green : ℕ),
    yellow = 12 →
    blue = yellow / 2 →
    green = yellow * 2 →
    fish_count yellow blue green = 42 :=
by
  sorry

#check olga_aquarium_fish_count

end NUMINAMATH_CALUDE_olga_aquarium_fish_count_l2730_273093


namespace NUMINAMATH_CALUDE_min_dist_point_on_line_l2730_273039

/-- The point that minimizes the sum of distances to two given points on a line -/
def minDistPoint (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

/-- The line 3x - 4y + 4 = 0 -/
def line (p : ℝ × ℝ) : Prop :=
  3 * p.1 - 4 * p.2 + 4 = 0

theorem min_dist_point_on_line :
  let A : ℝ × ℝ := (-3, 5)
  let B : ℝ × ℝ := (2, 15)
  let P : ℝ × ℝ := (8/3, 3)
  line P ∧
  ∀ Q : ℝ × ℝ, line Q →
    distance P A + distance P B ≤ distance Q A + distance Q B :=
sorry

end NUMINAMATH_CALUDE_min_dist_point_on_line_l2730_273039


namespace NUMINAMATH_CALUDE_team_pizza_consumption_l2730_273060

theorem team_pizza_consumption (total_slices : ℕ) (slices_left : ℕ) : 
  total_slices = 32 → slices_left = 7 → total_slices - slices_left = 25 := by
  sorry

end NUMINAMATH_CALUDE_team_pizza_consumption_l2730_273060


namespace NUMINAMATH_CALUDE_digit_1234_is_4_l2730_273067

/-- The number of digits in the representation of an integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The nth digit in the decimal expansion of x -/
def nth_digit (x : ℝ) (n : ℕ) : ℕ := sorry

/-- The number formed by concatenating the decimal representations of integers from 1 to n -/
def concat_integers (n : ℕ) : ℝ := sorry

theorem digit_1234_is_4 :
  let x := concat_integers 500
  nth_digit x 1234 = 4 := by sorry

end NUMINAMATH_CALUDE_digit_1234_is_4_l2730_273067


namespace NUMINAMATH_CALUDE_total_people_in_program_l2730_273029

theorem total_people_in_program : 
  let parents : ℕ := 105
  let pupils : ℕ := 698
  let staff : ℕ := 45
  let performers : ℕ := 32
  parents + pupils + staff + performers = 880 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l2730_273029


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2730_273099

theorem smallest_integer_with_remainders : ∃ b : ℕ+, 
  (b : ℕ) % 4 = 3 ∧ 
  (b : ℕ) % 6 = 5 ∧ 
  ∀ c : ℕ+, (c : ℕ) % 4 = 3 → (c : ℕ) % 6 = 5 → b ≤ c :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2730_273099


namespace NUMINAMATH_CALUDE_convex_polygon_equal_area_division_l2730_273049

-- Define a convex polygon
structure ConvexPolygon where
  -- Add necessary properties to define a convex polygon
  is_convex : Bool

-- Define a line in 2D space
structure Line where
  -- Add necessary properties to define a line
  slope : ℝ
  intercept : ℝ

-- Define the concept of perpendicular lines
def perpendicular (l1 l2 : Line) : Prop :=
  -- Add condition for perpendicularity
  sorry

-- Define the concept of a region in the polygon
structure Region where
  -- Add necessary properties to define a region
  area : ℝ

-- Define the division of a polygon by two lines
def divide_polygon (p : ConvexPolygon) (l1 l2 : Line) : List Region :=
  -- Function to divide the polygon into regions
  sorry

-- Theorem statement
theorem convex_polygon_equal_area_division (p : ConvexPolygon) :
  ∃ (l1 l2 : Line), 
    perpendicular l1 l2 ∧ 
    let regions := divide_polygon p l1 l2
    regions.length = 4 ∧ 
    ∀ (r1 r2 : Region), r1 ∈ regions → r2 ∈ regions → r1.area = r2.area :=
by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_equal_area_division_l2730_273049


namespace NUMINAMATH_CALUDE_group_size_l2730_273012

theorem group_size (B F BF : ℕ) (h1 : B = 13) (h2 : F = 15) (h3 : BF = 18) : 
  B + F - BF + 3 = 13 := by
sorry

end NUMINAMATH_CALUDE_group_size_l2730_273012


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2730_273008

theorem simplify_polynomial (s : ℝ) : (2*s^2 - 5*s + 3) - (s^2 + 4*s - 6) = s^2 - 9*s + 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2730_273008


namespace NUMINAMATH_CALUDE_survey_questions_l2730_273081

-- Define the number of questions per survey
def questionsPerSurvey : ℕ := sorry

-- Define the payment per question
def paymentPerQuestion : ℚ := 1/5

-- Define the number of surveys completed on Monday
def mondaySurveys : ℕ := 3

-- Define the number of surveys completed on Tuesday
def tuesdaySurveys : ℕ := 4

-- Define the total earnings
def totalEarnings : ℚ := 14

-- Theorem statement
theorem survey_questions :
  questionsPerSurvey * (mondaySurveys + tuesdaySurveys : ℚ) * paymentPerQuestion = totalEarnings ∧
  questionsPerSurvey = 10 := by
  sorry

end NUMINAMATH_CALUDE_survey_questions_l2730_273081


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l2730_273088

theorem complex_modulus_equation : ∃ (n : ℝ), n > 0 ∧ Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 26 ∧ n = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l2730_273088


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2730_273003

theorem cubic_roots_sum (a b : ℝ) : 
  (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (∀ w : ℝ, w^3 - 9*w^2 + a*w - b = 0 ↔ (w = x ∨ w = y ∨ w = z))) →
  a + b = 38 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2730_273003


namespace NUMINAMATH_CALUDE_mn_perpendicular_pq_l2730_273057

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Line : Type :=
  (a b : Point)

-- Define the quadrilateral and its properties
structure Quadrilateral : Type :=
  (A B C D : Point)
  (convex : Bool)

-- Define the intersection point of diagonals
def intersectionPoint (q : Quadrilateral) : Point :=
  sorry

-- Define centroid of a triangle
def centroid (p1 p2 p3 : Point) : Point :=
  sorry

-- Define orthocenter of a triangle
def orthocenter (p1 p2 p3 : Point) : Point :=
  sorry

-- Define perpendicularity of lines
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

-- Theorem statement
theorem mn_perpendicular_pq (q : Quadrilateral) :
  let O := intersectionPoint q
  let M := centroid q.A O q.B
  let N := centroid q.C O q.D
  let P := orthocenter q.B O q.C
  let Q := orthocenter q.D O q.A
  perpendicular (Line.mk M N) (Line.mk P Q) :=
sorry

end NUMINAMATH_CALUDE_mn_perpendicular_pq_l2730_273057


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2730_273013

theorem inequality_equivalence :
  ∀ x : ℝ, |(7 - 2*x) / 4| < 3 ↔ -2.5 < x ∧ x < 9.5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2730_273013


namespace NUMINAMATH_CALUDE_sum_of_xy_l2730_273030

theorem sum_of_xy (x y : ℕ) (hx : x > 0) (hy : y > 0) (hx_bound : x < 30) (hy_bound : y < 30) 
  (h_eq : x + y + x * y = 119) : x + y = 24 ∨ x + y = 21 ∨ x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l2730_273030


namespace NUMINAMATH_CALUDE_arcsin_sqrt2_over_2_l2730_273043

theorem arcsin_sqrt2_over_2 : Real.arcsin (Real.sqrt 2 / 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sqrt2_over_2_l2730_273043


namespace NUMINAMATH_CALUDE_platform_length_l2730_273095

/-- Given a train of length 300 meters that crosses a signal pole in 20 seconds
    and a platform in 39 seconds, the length of the platform is 285 meters. -/
theorem platform_length (train_length : ℝ) (pole_time : ℝ) (platform_time : ℝ) :
  train_length = 300 →
  pole_time = 20 →
  platform_time = 39 →
  ∃ platform_length : ℝ,
    platform_length = 285 ∧
    train_length / pole_time * platform_time = train_length + platform_length :=
by
  sorry


end NUMINAMATH_CALUDE_platform_length_l2730_273095


namespace NUMINAMATH_CALUDE_max_value_on_interval_max_value_is_11_l2730_273025

def f (x : ℝ) : ℝ := x^4 - 8*x^2 + 2

theorem max_value_on_interval (a b : ℝ) (h : a ≤ b) :
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c :=
sorry

theorem max_value_is_11 :
  ∃ c ∈ Set.Icc (-1) 3, f c = 11 ∧ ∀ x ∈ Set.Icc (-1) 3, f x ≤ f c :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_max_value_is_11_l2730_273025


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2730_273076

theorem fraction_subtraction :
  (4 : ℚ) / 5 - (1 : ℚ) / 5 = (3 : ℚ) / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2730_273076
