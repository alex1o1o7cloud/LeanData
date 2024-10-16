import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2738_273806

-- Define the repeating decimals
def repeating_2 : ℚ := 2/9
def repeating_03 : ℚ := 1/33

-- Theorem statement
theorem sum_of_repeating_decimals : 
  repeating_2 + repeating_03 = 25/99 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2738_273806


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l2738_273847

theorem rectangular_garden_area (perimeter width length : ℝ) : 
  perimeter = 72 →
  length = 3 * width →
  2 * length + 2 * width = perimeter →
  length * width = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l2738_273847


namespace NUMINAMATH_CALUDE_factor_expression_l2738_273827

theorem factor_expression (x : ℝ) : 45 * x + 30 = 15 * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2738_273827


namespace NUMINAMATH_CALUDE_divisor_problem_l2738_273812

theorem divisor_problem (w : ℤ) (x : ℤ) :
  (∃ k : ℤ, w = 13 * k) →
  (∃ m : ℤ, w + 3 = x * m) →
  x = 3 :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l2738_273812


namespace NUMINAMATH_CALUDE_trapezoid_area_l2738_273862

/-- Represents a trapezoid ABCD with a circle passing through A, B, and touching C -/
structure TrapezoidWithCircle where
  /-- Length of CD -/
  cd : ℝ
  /-- Length of AE -/
  ae : ℝ
  /-- The circle is centered on diagonal AC -/
  circle_on_diagonal : Bool
  /-- BC is parallel to AD -/
  bc_parallel_ad : Bool
  /-- The circle passes through A and B -/
  circle_through_ab : Bool
  /-- The circle touches CD at C -/
  circle_touches_cd : Bool
  /-- The circle intersects AD at E -/
  circle_intersects_ad : Bool

/-- Calculate the area of the trapezoid ABCD -/
def calculate_area (t : TrapezoidWithCircle) : ℝ :=
  sorry

/-- Theorem stating that the area of the trapezoid ABCD is 204 -/
theorem trapezoid_area (t : TrapezoidWithCircle) 
  (h1 : t.cd = 6 * Real.sqrt 13)
  (h2 : t.ae = 8)
  (h3 : t.circle_on_diagonal)
  (h4 : t.bc_parallel_ad)
  (h5 : t.circle_through_ab)
  (h6 : t.circle_touches_cd)
  (h7 : t.circle_intersects_ad) :
  calculate_area t = 204 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2738_273862


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l2738_273869

/-- The minimum distance between a point on the line x - 2y + 2 = 0 and the origin -/
theorem min_distance_to_origin : 
  ∃ (d : ℝ), d = 2 * Real.sqrt 5 / 5 ∧ 
  ∀ (P : ℝ × ℝ), P.1 - 2 * P.2 + 2 = 0 → 
  Real.sqrt (P.1^2 + P.2^2) ≥ d := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l2738_273869


namespace NUMINAMATH_CALUDE_quarter_probability_is_3_28_l2738_273825

/-- Represents the types of coins in the jar -/
inductive Coin
| Quarter
| Nickel
| Penny
| Dime

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
| Coin.Quarter => 25
| Coin.Nickel => 5
| Coin.Penny => 1
| Coin.Dime => 10

/-- The total value of each coin type in cents -/
def total_value : Coin → ℕ
| Coin.Quarter => 1200
| Coin.Nickel => 500
| Coin.Penny => 200
| Coin.Dime => 1000

/-- The number of coins of each type -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Quarter + coin_count Coin.Nickel + 
                       coin_count Coin.Penny + coin_count Coin.Dime

/-- The probability of choosing a quarter -/
def quarter_probability : ℚ := coin_count Coin.Quarter / total_coins

theorem quarter_probability_is_3_28 : quarter_probability = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_quarter_probability_is_3_28_l2738_273825


namespace NUMINAMATH_CALUDE_sine_cosine_relation_l2738_273809

theorem sine_cosine_relation (x : ℝ) : 
  Real.sin (2 * x + π / 6) = -1 / 3 → Real.cos (π / 3 - 2 * x) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_relation_l2738_273809


namespace NUMINAMATH_CALUDE_union_of_sets_l2738_273896

/-- Given sets A and B with specific properties, prove their union -/
theorem union_of_sets (a : ℝ) : 
  let A : Set ℝ := {|a + 1|, 3, 5}
  let B : Set ℝ := {2*a + 1, a^(2*a + 2), a^2 + 2*a - 1}
  (A ∩ B = {2, 3}) → (A ∪ B = {1, 2, 3, 5}) := by
  sorry


end NUMINAMATH_CALUDE_union_of_sets_l2738_273896


namespace NUMINAMATH_CALUDE_initial_deposit_proof_l2738_273860

/-- Proves that the initial deposit is correct given the total savings goal,
    saving period, and weekly saving amount. -/
theorem initial_deposit_proof (total_goal : ℕ) (weeks : ℕ) (weekly_saving : ℕ) 
    (h1 : total_goal = 500)
    (h2 : weeks = 19)
    (h3 : weekly_saving = 17) : 
  total_goal - (weeks * weekly_saving) = 177 := by
  sorry

end NUMINAMATH_CALUDE_initial_deposit_proof_l2738_273860


namespace NUMINAMATH_CALUDE_not_perfect_square_l2738_273844

theorem not_perfect_square (n : ℕ) : ¬∃ (m : ℕ), 4 * n^2 + 4 * n + 4 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2738_273844


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l2738_273813

theorem polynomial_value_theorem (f : ℝ → ℝ) :
  (∃ a b c d : ℝ, ∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  (|f 1| = 12 ∧ |f 2| = 12 ∧ |f 3| = 12 ∧ |f 5| = 12 ∧ |f 6| = 12 ∧ |f 7| = 12) →
  |f 0| = 72 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l2738_273813


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l2738_273810

-- Define the bowties operation
noncomputable def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt b^2)))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ h : ℝ, bowtie 5 h = 10 ∧ h = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l2738_273810


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2738_273826

theorem sqrt_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2738_273826


namespace NUMINAMATH_CALUDE_veterinary_clinic_payment_l2738_273865

/-- Veterinary clinic problem -/
theorem veterinary_clinic_payment
  (dog_charge : ℕ)
  (cat_charge : ℕ)
  (parrot_charge : ℕ)
  (rabbit_charge : ℕ)
  (dogs : ℕ)
  (cats : ℕ)
  (parrots : ℕ)
  (rabbits : ℕ)
  (h1 : dog_charge = 60)
  (h2 : cat_charge = 40)
  (h3 : parrot_charge = 70)
  (h4 : rabbit_charge = 50)
  (h5 : dogs = 25)
  (h6 : cats = 45)
  (h7 : parrots = 15)
  (h8 : rabbits = 10) :
  dog_charge * dogs + cat_charge * cats + parrot_charge * parrots + rabbit_charge * rabbits = 4850 := by
  sorry

end NUMINAMATH_CALUDE_veterinary_clinic_payment_l2738_273865


namespace NUMINAMATH_CALUDE_sin_2theta_value_l2738_273845

theorem sin_2theta_value (θ : Real) 
  (h1 : Real.cos (π/4 - θ) * Real.cos (π/4 + θ) = Real.sqrt 2 / 6)
  (h2 : 0 < θ) (h3 : θ < π/2) : 
  Real.sin (2*θ) = Real.sqrt 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l2738_273845


namespace NUMINAMATH_CALUDE_intersection_point_determines_d_l2738_273841

theorem intersection_point_determines_d : ∀ d : ℝ,
  (∃ x y : ℝ, 3 * x - 4 * y = d ∧ 6 * x + 8 * y = -d ∧ x = 2 ∧ y = -3) →
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_determines_d_l2738_273841


namespace NUMINAMATH_CALUDE_strawberry_area_l2738_273808

/-- Given a garden with the following properties:
  * The total area is 64 square feet
  * Half of the garden is for fruits
  * A quarter of the fruit section is for strawberries
  Prove that the area for strawberries is 8 square feet. -/
theorem strawberry_area (garden_area : ℝ) (fruit_ratio : ℝ) (strawberry_ratio : ℝ) : 
  garden_area = 64 → 
  fruit_ratio = 1/2 → 
  strawberry_ratio = 1/4 → 
  garden_area * fruit_ratio * strawberry_ratio = 8 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_area_l2738_273808


namespace NUMINAMATH_CALUDE_store_prices_existence_l2738_273843

theorem store_prices_existence (S : ℕ) (h : S ≥ 100) :
  ∃ (T C B P : ℕ), T > C ∧ C > B ∧ T + C + B = S ∧ T * C * B = P ∧
  ∃ (T' C' B' : ℕ), (T', C', B') ≠ (T, C, B) ∧
    T' > C' ∧ C' > B' ∧ T' + C' + B' = S ∧ T' * C' * B' = P :=
by sorry

end NUMINAMATH_CALUDE_store_prices_existence_l2738_273843


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2738_273866

theorem polynomial_factorization (a b c d : ℤ) :
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + x^3 - 5*x^2 + x - 6) →
  a + b + c + d = -4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2738_273866


namespace NUMINAMATH_CALUDE_tan_sum_eq_neg_one_l2738_273816

theorem tan_sum_eq_neg_one (α β : ℝ) 
  (h : 2 * Real.sin β * Real.sin (α - π/4) = Real.sin (α - β + π/4)) : 
  Real.tan (α + β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_eq_neg_one_l2738_273816


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2738_273859

theorem hemisphere_surface_area (V : ℝ) (h : V = (500 / 3) * Real.pi) :
  ∃ (r : ℝ), V = (2 / 3) * Real.pi * r^3 ∧
             (2 * Real.pi * r^2 + Real.pi * r^2) = 3 * Real.pi * 250^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2738_273859


namespace NUMINAMATH_CALUDE_triangle_area_sum_properties_l2738_273857

/-- Represents a rectangular prism with dimensions 1, 2, and 3 -/
structure RectangularPrism where
  length : ℝ := 1
  width : ℝ := 2
  height : ℝ := 3

/-- Represents the sum of areas of all triangles with vertices on the prism -/
def triangleAreaSum (prism : RectangularPrism) : ℝ := sorry

/-- Theorem stating the properties of the triangle area sum -/
theorem triangle_area_sum_properties (prism : RectangularPrism) :
  ∃ (m a n : ℤ), 
    triangleAreaSum prism = m + a * Real.sqrt n ∧ 
    m + n + a = 49 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_sum_properties_l2738_273857


namespace NUMINAMATH_CALUDE_max_height_formula_l2738_273800

/-- Triangle ABC with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The maximum possible height of the table formed by right angle folds -/
def max_table_height (t : Triangle) : ℝ := sorry

/-- The specific triangle in the problem -/
def problem_triangle : Triangle := { a := 25, b := 28, c := 31 }

theorem max_height_formula : 
  max_table_height problem_triangle = 42 * Real.sqrt 2582 / 28 := by sorry

end NUMINAMATH_CALUDE_max_height_formula_l2738_273800


namespace NUMINAMATH_CALUDE_relationship_abcd_l2738_273805

theorem relationship_abcd (a b c d : ℝ) :
  (a + 2*b) / (2*b + c) = (c + 2*d) / (2*d + a) →
  (a = c ∨ a + c + 2*(b + d) = 0) :=
by sorry

end NUMINAMATH_CALUDE_relationship_abcd_l2738_273805


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2738_273838

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h : (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8/3) : 
  Real.tan (α + π/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2738_273838


namespace NUMINAMATH_CALUDE_f_value_at_2_f_equals_f_horner_f_2_equals_62_l2738_273877

/-- The polynomial function f(x) = 2x^4 + 3x^3 + 5x - 4 -/
def f (x : ℝ) : ℝ := 2*x^4 + 3*x^3 + 5*x - 4

/-- Horner's method representation of f(x) -/
def f_horner (x : ℝ) : ℝ := x*(x*(x*(2*x + 3)) + 5) - 4

theorem f_value_at_2 : f 2 = 62 := by sorry

theorem f_equals_f_horner : ∀ x, f x = f_horner x := by sorry

theorem f_2_equals_62 : f_horner 2 = 62 := by sorry

end NUMINAMATH_CALUDE_f_value_at_2_f_equals_f_horner_f_2_equals_62_l2738_273877


namespace NUMINAMATH_CALUDE_root_product_expression_l2738_273829

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 - 2*p*α + p^2 - 1 = 0) → 
  (β^2 - 2*p*β + p^2 - 1 = 0) → 
  (γ^2 - 2*q*γ + q^2 - 1 = 0) → 
  (δ^2 - 2*q*δ + q^2 - 1 = 0) → 
  (α - γ)*(β - δ)*(α - δ)*(β - γ) = (q^2 + p^2)^2 + 4*p^2 - 4*p*q*(q^2 + p^2) := by
sorry

end NUMINAMATH_CALUDE_root_product_expression_l2738_273829


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2738_273846

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {-3, -2, 2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {-3, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2738_273846


namespace NUMINAMATH_CALUDE_number_equals_twenty_l2738_273890

theorem number_equals_twenty : ∃ x : ℝ, 5 * x = 100 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_twenty_l2738_273890


namespace NUMINAMATH_CALUDE_pairwise_sums_problem_l2738_273875

theorem pairwise_sums_problem (a b c d e x y : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
    {183, 186, 187, 190, 191, 192, 193, 194, 196, x} ∧
  x > 196 ∧
  y = 10 * x + 3 →
  a = 91 ∧ b = 92 ∧ c = 95 ∧ d = 99 ∧ e = 101 ∧ x = 200 ∧ y = 2003 :=
by sorry

end NUMINAMATH_CALUDE_pairwise_sums_problem_l2738_273875


namespace NUMINAMATH_CALUDE_basketball_team_cutoff_l2738_273830

/-- The number of students who didn't make the cut for the basketball team -/
theorem basketball_team_cutoff (girls boys callback : ℕ) 
  (h1 : girls = 39)
  (h2 : boys = 4)
  (h3 : callback = 26) :
  girls + boys - callback = 17 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_cutoff_l2738_273830


namespace NUMINAMATH_CALUDE_max_garden_area_l2738_273802

theorem max_garden_area (L : ℝ) (h : L > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = L ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a + 2*b = L → x*y ≥ a*b ∧
  x*y = L^2/8 :=
sorry

end NUMINAMATH_CALUDE_max_garden_area_l2738_273802


namespace NUMINAMATH_CALUDE_most_cost_effective_boat_rental_l2738_273842

/-- Represents the cost and capacity of a boat type -/
structure BoatType where
  capacity : Nat
  cost : Nat

/-- Represents a combination of boats -/
structure BoatCombination where
  largeboats : Nat
  smallboats : Nat

def totalPeople (b : BoatCombination) (large : BoatType) (small : BoatType) : Nat :=
  b.largeboats * large.capacity + b.smallboats * small.capacity

def totalCost (b : BoatCombination) (large : BoatType) (small : BoatType) : Nat :=
  b.largeboats * large.cost + b.smallboats * small.cost

def isSufficient (b : BoatCombination) (large : BoatType) (small : BoatType) (people : Nat) : Prop :=
  totalPeople b large small ≥ people

def isMoreCostEffective (b1 b2 : BoatCombination) (large : BoatType) (small : BoatType) : Prop :=
  totalCost b1 large small < totalCost b2 large small

theorem most_cost_effective_boat_rental :
  let large : BoatType := { capacity := 6, cost := 24 }
  let small : BoatType := { capacity := 4, cost := 20 }
  let people : Nat := 46
  let optimal : BoatCombination := { largeboats := 7, smallboats := 1 }
  (isSufficient optimal large small people) ∧
  (∀ b : BoatCombination, 
    isSufficient b large small people → 
    totalCost optimal large small ≤ totalCost b large small) := by
  sorry

#check most_cost_effective_boat_rental

end NUMINAMATH_CALUDE_most_cost_effective_boat_rental_l2738_273842


namespace NUMINAMATH_CALUDE_total_worth_is_correct_l2738_273803

-- Define the given conditions
def initial_packs : ℕ := 4
def new_packs : ℕ := 2
def price_per_pack : ℚ := 2.5
def discount_rate : ℚ := 0.15
def tax_rate : ℚ := 0.07

-- Define the function to calculate the total worth
def total_worth : ℚ :=
  let initial_cost := initial_packs * price_per_pack
  let new_cost := new_packs * price_per_pack
  let discount := new_cost * discount_rate
  let discounted_cost := new_cost - discount
  let tax := new_cost * tax_rate
  let total_new_cost := discounted_cost + tax
  initial_cost + total_new_cost

-- State the theorem
theorem total_worth_is_correct : total_worth = 14.6 := by
  sorry

end NUMINAMATH_CALUDE_total_worth_is_correct_l2738_273803


namespace NUMINAMATH_CALUDE_pitcher_juice_distribution_l2738_273853

theorem pitcher_juice_distribution (pitcher_capacity : ℝ) (num_cups : ℕ) :
  pitcher_capacity > 0 →
  num_cups = 8 →
  let juice_amount := pitcher_capacity / 2
  let juice_per_cup := juice_amount / num_cups
  juice_per_cup / pitcher_capacity = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_pitcher_juice_distribution_l2738_273853


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2738_273884

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 65 / 6 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 1 / b₀ = 65 / 6 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2738_273884


namespace NUMINAMATH_CALUDE_abs_inequality_and_fraction_inequality_l2738_273850

theorem abs_inequality_and_fraction_inequality :
  (∀ x : ℝ, |x + 3| - |x - 2| ≥ 3 ↔ x ≥ 1) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → (a^2 - b^2) / (a^2 + b^2) > (a - b) / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_and_fraction_inequality_l2738_273850


namespace NUMINAMATH_CALUDE_henrys_action_figures_l2738_273817

def action_figure_problem (total_needed : ℕ) (cost_per_figure : ℕ) (money_needed : ℕ) : Prop :=
  let figures_to_buy : ℕ := money_needed / cost_per_figure
  let initial_figures : ℕ := total_needed - figures_to_buy
  initial_figures = 3

theorem henrys_action_figures :
  action_figure_problem 8 6 30 := by
  sorry

end NUMINAMATH_CALUDE_henrys_action_figures_l2738_273817


namespace NUMINAMATH_CALUDE_park_diameter_is_40_l2738_273819

/-- Represents the circular park with its components -/
structure CircularPark where
  pond_diameter : ℝ
  garden_width : ℝ
  path_width : ℝ

/-- Calculates the diameter of the outer boundary of the jogging path -/
def outer_boundary_diameter (park : CircularPark) : ℝ :=
  park.pond_diameter + 2 * (park.garden_width + park.path_width)

/-- Theorem stating that for the given park dimensions, the outer boundary diameter is 40 feet -/
theorem park_diameter_is_40 :
  let park : CircularPark := {
    pond_diameter := 12,
    garden_width := 10,
    path_width := 4
  }
  outer_boundary_diameter park = 40 := by sorry

end NUMINAMATH_CALUDE_park_diameter_is_40_l2738_273819


namespace NUMINAMATH_CALUDE_sqrt_ratio_simplification_l2738_273895

theorem sqrt_ratio_simplification :
  (Real.sqrt (8^2 + 15^2)) / (Real.sqrt (49 + 64)) = 17 / Real.sqrt 113 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ratio_simplification_l2738_273895


namespace NUMINAMATH_CALUDE_fraction_simplification_l2738_273861

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) :
  (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2738_273861


namespace NUMINAMATH_CALUDE_kale_spring_mowings_l2738_273840

/-- The number of times Kale mowed his lawn in the spring -/
def spring_mowings : ℕ := sorry

/-- The number of times Kale mowed his lawn in the summer -/
def summer_mowings : ℕ := 5

/-- The difference between spring and summer mowings -/
def mowing_difference : ℕ := 3

/-- Theorem stating that Kale mowed his lawn 8 times in the spring -/
theorem kale_spring_mowings :
  spring_mowings = 8 ∧
  summer_mowings = 5 ∧
  spring_mowings - summer_mowings = mowing_difference :=
sorry

end NUMINAMATH_CALUDE_kale_spring_mowings_l2738_273840


namespace NUMINAMATH_CALUDE_larger_number_proof_l2738_273889

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) :
  L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2738_273889


namespace NUMINAMATH_CALUDE_total_income_calculation_l2738_273891

/-- Calculates the total income for a clothing store sale --/
def calculate_total_income (tshirt_price : ℚ) (pants_price : ℚ) (skirt_price : ℚ) 
                           (refurbished_tshirt_price : ℚ) (skirt_discount_rate : ℚ) 
                           (tshirt_discount_rate : ℚ) (sales_tax_rate : ℚ) 
                           (tshirts_sold : ℕ) (refurbished_tshirts_sold : ℕ) 
                           (pants_sold : ℕ) (skirts_sold : ℕ) : ℚ :=
  sorry

theorem total_income_calculation :
  let tshirt_price : ℚ := 5
  let pants_price : ℚ := 4
  let skirt_price : ℚ := 6
  let refurbished_tshirt_price : ℚ := tshirt_price / 2
  let skirt_discount_rate : ℚ := 1 / 10
  let tshirt_discount_rate : ℚ := 1 / 5
  let sales_tax_rate : ℚ := 2 / 25
  let tshirts_sold : ℕ := 15
  let refurbished_tshirts_sold : ℕ := 7
  let pants_sold : ℕ := 6
  let skirts_sold : ℕ := 12
  calculate_total_income tshirt_price pants_price skirt_price refurbished_tshirt_price 
                         skirt_discount_rate tshirt_discount_rate sales_tax_rate
                         tshirts_sold refurbished_tshirts_sold pants_sold skirts_sold = 1418 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_total_income_calculation_l2738_273891


namespace NUMINAMATH_CALUDE_sport_water_amount_l2738_273833

/-- Represents the ratios in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard drink formulation -/
def standard_ratio : DrinkRatio :=
  ⟨1, 12, 30⟩

/-- The sport drink formulation -/
def sport_ratio : DrinkRatio :=
  ⟨1, 4, 60⟩

/-- The amount of corn syrup in the sport formulation -/
def sport_corn_syrup : ℚ := 3

theorem sport_water_amount :
  let water_amount := sport_corn_syrup * sport_ratio.water / sport_ratio.corn_syrup
  water_amount = 45 := by sorry

end NUMINAMATH_CALUDE_sport_water_amount_l2738_273833


namespace NUMINAMATH_CALUDE_fibonacci_mod_4_2012_eq_1_l2738_273828

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def fibonacci_mod_4 (n : ℕ) : ℕ := fibonacci n % 4

theorem fibonacci_mod_4_2012_eq_1 : fibonacci_mod_4 2012 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_mod_4_2012_eq_1_l2738_273828


namespace NUMINAMATH_CALUDE_function_fixed_point_l2738_273876

def iterateF (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (iterateF f n)

theorem function_fixed_point
  (f : ℝ → ℝ)
  (hf : Continuous f)
  (h : ∀ x : ℝ, ∃ n : ℕ, iterateF f n x = 1) :
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_fixed_point_l2738_273876


namespace NUMINAMATH_CALUDE_f_symmetric_about_x_eq_2_l2738_273821

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if -2 ≤ x ∧ x ≤ 0 then 2^x - 2^(-x) + x else 0  -- We define f on [-2,0] as given, and 0 elsewhere

-- State the theorem
theorem f_symmetric_about_x_eq_2 :
  (∀ x, x * f x = -x * f (-x)) →  -- y = xf(x) is even
  (∀ x, f (x - 1) + f (x + 3) = 0) →  -- given condition
  (∀ x, f (x - 2) = f (-x + 2)) :=  -- symmetry about x = 2
by sorry

end NUMINAMATH_CALUDE_f_symmetric_about_x_eq_2_l2738_273821


namespace NUMINAMATH_CALUDE_age_difference_l2738_273873

/-- Given the ages of three people a, b, and c, prove that a is 2 years older than b -/
theorem age_difference (a b c : ℕ) : 
  b = 2 * c →
  a + b + c = 17 →
  b = 6 →
  a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2738_273873


namespace NUMINAMATH_CALUDE_currency_exchange_problem_l2738_273823

def exchange_rate : ℚ := 8 / 6

theorem currency_exchange_problem (d : ℕ) :
  (d : ℚ) * exchange_rate - 96 = d →
  d = 288 := by sorry

end NUMINAMATH_CALUDE_currency_exchange_problem_l2738_273823


namespace NUMINAMATH_CALUDE_system_solutions_l2738_273839

def equation1 (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 6*y = -20

def equation2 (x z : ℝ) : Prop := x^2 + z^2 + 8*x + 4*z = -10

def equation3 (y z : ℝ) : Prop := y^2 + z^2 - 6*y + 4*z = 0

def is_solution (x y z : ℝ) : Prop :=
  equation1 x y ∧ equation2 x z ∧ equation3 y z

theorem system_solutions :
  (is_solution (-3) 1 1) ∧
  (is_solution (-3) 1 (-5)) ∧
  (is_solution (-3) 5 1) ∧
  (is_solution (-3) 5 (-5)) ∧
  (is_solution (-5) 1 1) ∧
  (is_solution (-5) 1 (-5)) ∧
  (is_solution (-5) 5 1) ∧
  (is_solution (-5) 5 (-5)) ∧
  (∀ x y z, is_solution x y z →
    ((x = -3 ∧ y = 1 ∧ z = 1) ∨
     (x = -3 ∧ y = 1 ∧ z = -5) ∨
     (x = -3 ∧ y = 5 ∧ z = 1) ∨
     (x = -3 ∧ y = 5 ∧ z = -5) ∨
     (x = -5 ∧ y = 1 ∧ z = 1) ∨
     (x = -5 ∧ y = 1 ∧ z = -5) ∨
     (x = -5 ∧ y = 5 ∧ z = 1) ∨
     (x = -5 ∧ y = 5 ∧ z = -5))) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l2738_273839


namespace NUMINAMATH_CALUDE_probability_JQKA_same_suit_value_l2738_273872

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards dealt -/
def CardsDealt : ℕ := 4

/-- Represents the number of Jacks in a standard deck -/
def JacksInDeck : ℕ := 4

/-- Probability of drawing a specific sequence of four cards (Jack, Queen, King, Ace) 
    of the same suit from a standard 52-card deck without replacement -/
def probability_JQKA_same_suit : ℚ :=
  (JacksInDeck : ℚ) / StandardDeck *
  1 / (StandardDeck - 1) *
  1 / (StandardDeck - 2) *
  1 / (StandardDeck - 3)

theorem probability_JQKA_same_suit_value : 
  probability_JQKA_same_suit = 1 / 1624350 := by sorry

end NUMINAMATH_CALUDE_probability_JQKA_same_suit_value_l2738_273872


namespace NUMINAMATH_CALUDE_g_determinant_identity_g_1002_1004_minus_1003_squared_l2738_273858

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -1; 1, 0]

-- Define the sequence G
def G : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => G (n + 1) - G n

-- State the theorem
theorem g_determinant_identity (n : ℕ) :
  A ^ n = !![G (n + 1), G n; G n, G (n - 1)] →
  G n * G (n + 2) - G (n + 1) ^ 2 = 1 := by
  sorry

-- The specific case for n = 1003
theorem g_1002_1004_minus_1003_squared :
  G 1002 * G 1004 - G 1003 ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_determinant_identity_g_1002_1004_minus_1003_squared_l2738_273858


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_cubes_l2738_273888

theorem units_digit_of_sum_of_cubes : (24^3 + 42^3) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_cubes_l2738_273888


namespace NUMINAMATH_CALUDE_zeta_sum_eight_l2738_273824

theorem zeta_sum_eight (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 8)
  (h3 : ζ₁^4 + ζ₂^4 + ζ₃^4 = 26) :
  ζ₁^8 + ζ₂^8 + ζ₃^8 = 219 := by sorry

end NUMINAMATH_CALUDE_zeta_sum_eight_l2738_273824


namespace NUMINAMATH_CALUDE_max_volume_box_l2738_273868

/-- Represents a rectangular box without a lid -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The volume of a box -/
def volume (b : Box) : ℝ := b.length * b.width * b.height

/-- The surface area of a box without a lid -/
def surfaceArea (b : Box) : ℝ := 
  b.length * b.width + 2 * b.height * (b.length + b.width)

/-- Theorem: Maximum volume of a box with given constraints -/
theorem max_volume_box : 
  ∃ (b : Box), 
    b.width = 2 ∧ 
    surfaceArea b = 32 ∧ 
    (∀ (b' : Box), b'.width = 2 → surfaceArea b' = 32 → volume b' ≤ volume b) ∧
    volume b = 16 := by
  sorry


end NUMINAMATH_CALUDE_max_volume_box_l2738_273868


namespace NUMINAMATH_CALUDE_polynomial_evaluation_at_negative_two_l2738_273852

theorem polynomial_evaluation_at_negative_two :
  let f : ℝ → ℝ := λ x ↦ x^3 + x^2 + 2*x + 2
  f (-2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_at_negative_two_l2738_273852


namespace NUMINAMATH_CALUDE_at_least_one_accepted_l2738_273863

theorem at_least_one_accepted (prob_A prob_B : ℝ) 
  (h1 : 0 ≤ prob_A ∧ prob_A ≤ 1)
  (h2 : 0 ≤ prob_B ∧ prob_B ≤ 1)
  (independence : True) -- Assumption of independence
  : 1 - (1 - prob_A) * (1 - prob_B) = prob_A + prob_B - prob_A * prob_B :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_accepted_l2738_273863


namespace NUMINAMATH_CALUDE_group_sum_difference_l2738_273898

/-- S_n represents the sum of the n-th group in a sequence where
    the n-th group contains n consecutive natural numbers starting from
    n(n-1)/2 + 1 -/
def S (n : ℕ) : ℕ := n * (n^2 + 1) / 2

/-- The theorem states that S_16 - S_4 - S_1 = 2021 -/
theorem group_sum_difference : S 16 - S 4 - S 1 = 2021 := by
  sorry

end NUMINAMATH_CALUDE_group_sum_difference_l2738_273898


namespace NUMINAMATH_CALUDE_fraction_simplification_l2738_273897

theorem fraction_simplification (x : ℝ) : (x + 3) / 4 - (5 - 2*x) / 3 = (11*x - 11) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2738_273897


namespace NUMINAMATH_CALUDE_line_through_point_intersecting_circle_l2738_273893

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a line intersects a circle with a given chord length -/
def lineIntersectsCircle (l : Line) (c : Circle) (chordLength : ℝ) : Prop :=
  ∃ (p1 p2 : Point),
    pointOnLine p1 l ∧ pointOnLine p2 l ∧
    (p1.x - c.center.x)^2 + (p1.y - c.center.y)^2 = c.radius^2 ∧
    (p2.x - c.center.x)^2 + (p2.y - c.center.y)^2 = c.radius^2 ∧
    (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = chordLength^2

theorem line_through_point_intersecting_circle (p : Point) (c : Circle) (chordLength : ℝ) :
  let l1 : Line := { a := 1, b := 0, c := -3 }
  let l2 : Line := { a := 3, b := -4, c := 15 }
  p.x = 3 ∧ p.y = 6 ∧
  c.center.x = 0 ∧ c.center.y = 0 ∧ c.radius = 5 ∧
  chordLength = 8 →
  (pointOnLine p l1 ∧ lineIntersectsCircle l1 c chordLength) ∨
  (pointOnLine p l2 ∧ lineIntersectsCircle l2 c chordLength) := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_intersecting_circle_l2738_273893


namespace NUMINAMATH_CALUDE_cosine_difference_l2738_273885

theorem cosine_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 2) : 
  Real.cos (A - B) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_cosine_difference_l2738_273885


namespace NUMINAMATH_CALUDE_binary_to_quinary_conversion_l2738_273831

/-- Converts a natural number from base 2 to base 10 -/
def base2_to_base10 (n : List Bool) : ℕ :=
  n.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number from base 10 to base 5 -/
def base10_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_to_quinary_conversion :
  base10_to_base5 (base2_to_base10 [true, false, true, true, true]) = [1, 0, 4] := by
  sorry

end NUMINAMATH_CALUDE_binary_to_quinary_conversion_l2738_273831


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l2738_273883

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l2738_273883


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2738_273807

theorem coin_flip_probability : 
  let n : ℕ := 12  -- total number of flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads for a fair coin
  Nat.choose n k * p^k * (1-p)^(n-k) = 55/1024 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2738_273807


namespace NUMINAMATH_CALUDE_six_digit_same_digits_prime_divisor_sum_l2738_273815

def is_six_digit_same_digits (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ ∃ d : ℕ, n = d * 111111

def sum_distinct_prime_divisors (n : ℕ) : ℕ := sorry

theorem six_digit_same_digits_prime_divisor_sum 
  (n : ℕ) (h : is_six_digit_same_digits n) : 
  sum_distinct_prime_divisors n ≠ 70 ∧ sum_distinct_prime_divisors n ≠ 80 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_same_digits_prime_divisor_sum_l2738_273815


namespace NUMINAMATH_CALUDE_jean_grandchildren_gift_l2738_273836

/-- Calculates the total amount given to grandchildren per year -/
def total_given_to_grandchildren (num_grandchildren : ℕ) (cards_per_grandchild : ℕ) (amount_per_card : ℕ) : ℕ :=
  num_grandchildren * cards_per_grandchild * amount_per_card

/-- Proves that Jean gives $480 to her grandchildren per year -/
theorem jean_grandchildren_gift :
  total_given_to_grandchildren 3 2 80 = 480 := by
  sorry

#eval total_given_to_grandchildren 3 2 80

end NUMINAMATH_CALUDE_jean_grandchildren_gift_l2738_273836


namespace NUMINAMATH_CALUDE_water_bucket_problem_l2738_273834

theorem water_bucket_problem (bucket3 bucket5 bucket6 : ℕ) 
  (h1 : bucket3 = 3)
  (h2 : bucket5 = 5)
  (h3 : bucket6 = 6) :
  bucket6 - (bucket5 - bucket3) = 4 :=
by sorry

end NUMINAMATH_CALUDE_water_bucket_problem_l2738_273834


namespace NUMINAMATH_CALUDE_unique_ticket_number_l2738_273892

def is_valid_ticket (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  let x := n / 100
  let y := (n / 10) % 10
  let z := n % 10
  n = (10*x + y + 10*x + z + 10*y + x + 10*y + z + 10*z + x + 10*z + y) / 2

theorem unique_ticket_number :
  ∃! n : ℕ, is_valid_ticket n ∧ n = 198 := by
  sorry

end NUMINAMATH_CALUDE_unique_ticket_number_l2738_273892


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l2738_273867

theorem sandy_correct_sums (total_sums : ℕ) (total_marks : ℤ) 
  (correct_marks : ℕ) (incorrect_marks : ℕ) :
  total_sums = 30 →
  total_marks = 50 →
  correct_marks = 3 →
  incorrect_marks = 2 →
  ∃ (correct : ℕ) (incorrect : ℕ),
    correct + incorrect = total_sums ∧
    correct_marks * correct - incorrect_marks * incorrect = total_marks ∧
    correct = 22 := by
  sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l2738_273867


namespace NUMINAMATH_CALUDE_existence_of_non_one_start_l2738_273870

def begins_with_same_digit (x : ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧
  ∀ n : ℕ, n ≤ 2015 →
    ∃ k : ℕ, x^n = d * 10^k + (x^n - d * 10^k) ∧
             d * 10^k ≤ x^n ∧
             x^n < (d + 1) * 10^k

theorem existence_of_non_one_start :
  ∃ x : ℕ, begins_with_same_digit x ∧
    ∃ d : ℕ, d ≠ 1 ∧ d < 10 ∧
    ∀ n : ℕ, n ≤ 2015 →
      ∃ k : ℕ, x^n = d * 10^k + (x^n - d * 10^k) ∧
               d * 10^k ≤ x^n ∧
               x^n < (d + 1) * 10^k :=
by sorry

end NUMINAMATH_CALUDE_existence_of_non_one_start_l2738_273870


namespace NUMINAMATH_CALUDE_log_calculation_l2738_273854

theorem log_calculation : Real.log 25 / Real.log 10 + 
  (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) + 
  (Real.log 2 / Real.log 10)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_calculation_l2738_273854


namespace NUMINAMATH_CALUDE_max_term_binomial_expansion_l2738_273822

theorem max_term_binomial_expansion :
  let n : ℕ := 213
  let x : ℝ := Real.sqrt 5
  let term (k : ℕ) := (n.choose k) * x^k
  ∃ k_max : ℕ, k_max = 147 ∧ ∀ k : ℕ, k ≤ n → term k ≤ term k_max :=
by sorry

end NUMINAMATH_CALUDE_max_term_binomial_expansion_l2738_273822


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l2738_273804

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 36
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define the moving circle M
def M (x y r : ℝ) : Prop := ∃ (center : ℝ × ℝ), center = (x, y) ∧ r > 0

-- Define internal tangency of M to C₁
def internalTangent (x y r : ℝ) : Prop :=
  M x y r ∧ C₁ (x + r) y

-- Define external tangency of M to C₂
def externalTangent (x y r : ℝ) : Prop :=
  M x y r ∧ C₂ (x - r) y

-- Theorem statement
theorem trajectory_is_ellipse :
  ∀ (x y : ℝ),
    (∃ (r : ℝ), internalTangent x y r ∧ externalTangent x y r) →
    x^2 / 16 + y^2 / 15 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l2738_273804


namespace NUMINAMATH_CALUDE_nested_radical_solution_l2738_273820

theorem nested_radical_solution :
  ∃! x : ℝ, x > 0 ∧ x = Real.sqrt (3 + x) :=
by
  use (1 + Real.sqrt 13) / 2
  sorry

end NUMINAMATH_CALUDE_nested_radical_solution_l2738_273820


namespace NUMINAMATH_CALUDE_exactly_one_two_defective_mutually_exclusive_at_least_one_defective_all_genuine_mutually_exclusive_mutually_exclusive_pairs_l2738_273811

/-- Represents the number of genuine items in the box -/
def genuine_items : ℕ := 4

/-- Represents the number of defective items in the box -/
def defective_items : ℕ := 3

/-- Represents the number of items randomly selected -/
def selected_items : ℕ := 2

/-- Represents the event "Exactly one defective item" -/
def exactly_one_defective : Set (Fin genuine_items × Fin defective_items) := sorry

/-- Represents the event "Exactly two defective items" -/
def exactly_two_defective : Set (Fin genuine_items × Fin defective_items) := sorry

/-- Represents the event "At least one defective item" -/
def at_least_one_defective : Set (Fin genuine_items × Fin defective_items) := sorry

/-- Represents the event "All are genuine" -/
def all_genuine : Set (Fin genuine_items × Fin defective_items) := sorry

/-- Theorem stating that "Exactly one defective item" and "Exactly two defective items" are mutually exclusive -/
theorem exactly_one_two_defective_mutually_exclusive :
  exactly_one_defective ∩ exactly_two_defective = ∅ := sorry

/-- Theorem stating that "At least one defective item" and "All are genuine" are mutually exclusive -/
theorem at_least_one_defective_all_genuine_mutually_exclusive :
  at_least_one_defective ∩ all_genuine = ∅ := sorry

/-- Main theorem proving that only the specified pairs of events are mutually exclusive -/
theorem mutually_exclusive_pairs :
  (exactly_one_defective ∩ exactly_two_defective = ∅) ∧
  (at_least_one_defective ∩ all_genuine = ∅) ∧
  (exactly_one_defective ∩ at_least_one_defective ≠ ∅) ∧
  (exactly_two_defective ∩ at_least_one_defective ≠ ∅) ∧
  (exactly_one_defective ∩ all_genuine ≠ ∅) ∧
  (exactly_two_defective ∩ all_genuine ≠ ∅) := sorry

end NUMINAMATH_CALUDE_exactly_one_two_defective_mutually_exclusive_at_least_one_defective_all_genuine_mutually_exclusive_mutually_exclusive_pairs_l2738_273811


namespace NUMINAMATH_CALUDE_quotient_in_fourth_quadrant_l2738_273881

/-- Given two complex numbers z₁ and z₂, prove that their quotient lies in the fourth quadrant. -/
theorem quotient_in_fourth_quadrant (z₁ z₂ : ℂ) 
  (hz₁ : z₁ = 2 + Complex.I) 
  (hz₂ : z₂ = 1 + Complex.I) : 
  let q := z₁ / z₂
  0 < q.re ∧ q.im < 0 :=
by sorry


end NUMINAMATH_CALUDE_quotient_in_fourth_quadrant_l2738_273881


namespace NUMINAMATH_CALUDE_absolute_value_expression_l2738_273855

theorem absolute_value_expression (x : ℝ) (h : x < 0) : 
  |x - 3 * Real.sqrt ((x - 2)^2)| = 6 - 4*x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l2738_273855


namespace NUMINAMATH_CALUDE_toothpicks_for_ten_base_triangles_l2738_273856

/-- The number of toothpicks needed to construct a large equilateral triangle -/
def toothpicks_needed (base_triangles : ℕ) : ℕ :=
  let total_triangles := base_triangles * (base_triangles + 1) / 2
  let total_sides := 3 * total_triangles
  let shared_sides := (total_sides - 3 * base_triangles) / 2
  let boundary_sides := 3 * base_triangles
  shared_sides + boundary_sides

/-- Theorem stating that 98 toothpicks are needed for a large equilateral triangle with 10 small triangles on its base -/
theorem toothpicks_for_ten_base_triangles :
  toothpicks_needed 10 = 98 := by
  sorry


end NUMINAMATH_CALUDE_toothpicks_for_ten_base_triangles_l2738_273856


namespace NUMINAMATH_CALUDE_intersection_empty_intersection_equals_A_l2738_273837

-- Define set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 4*x - 12 > 0}

-- Theorem for the first question
theorem intersection_empty (a : ℝ) : 
  A a ∩ B = ∅ ↔ -2 ≤ a ∧ a ≤ 3 := by sorry

-- Theorem for the second question
theorem intersection_equals_A (a : ℝ) : 
  A a ∩ B = A a ↔ a < -5 ∨ a > 6 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_intersection_equals_A_l2738_273837


namespace NUMINAMATH_CALUDE_dot_product_bound_l2738_273871

theorem dot_product_bound (a b c m n p : ℝ) 
  (sum_abc : a + b + c = 1) 
  (sum_mnp : m + n + p = 1) : 
  -1 ≤ a*m + b*n + c*p ∧ a*m + b*n + c*p ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_dot_product_bound_l2738_273871


namespace NUMINAMATH_CALUDE_workday_percentage_theorem_l2738_273801

/-- Represents the duration of a workday in minutes -/
def workday_duration : ℕ := 10 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_duration : ℕ := 60

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_duration : ℕ := 2 * first_meeting_duration

/-- Represents the duration of the break in minutes -/
def break_duration : ℕ := 30

/-- Calculates the total time spent in meetings and on break -/
def total_meeting_and_break_time : ℕ :=
  first_meeting_duration + second_meeting_duration + break_duration

/-- Theorem: The percentage of the workday spent in meetings or on break is 35% -/
theorem workday_percentage_theorem :
  (total_meeting_and_break_time : ℚ) / (workday_duration : ℚ) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_workday_percentage_theorem_l2738_273801


namespace NUMINAMATH_CALUDE_third_candidate_votes_l2738_273880

theorem third_candidate_votes :
  ∀ (total_votes : ℕ) (winning_votes second_votes third_votes : ℕ),
    winning_votes = 11628 →
    second_votes = 7636 →
    winning_votes = (49.69230769230769 / 100 : ℚ) * total_votes →
    total_votes = winning_votes + second_votes + third_votes →
    third_votes = 4136 := by
  sorry

end NUMINAMATH_CALUDE_third_candidate_votes_l2738_273880


namespace NUMINAMATH_CALUDE_complete_square_sum_l2738_273814

/-- Given a quadratic equation x^2 - 6x + 5 = 0, when rewritten in the form (x + b)^2 = c
    where b and c are integers, prove that b + c = 1 -/
theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + b)^2 = c) → b + c = 1 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l2738_273814


namespace NUMINAMATH_CALUDE_marble_arrangement_mod_1000_l2738_273886

/-- The number of blue marbles -/
def blue_marbles : ℕ := 6

/-- The maximum number of yellow marbles that allows for a valid arrangement -/
def yellow_marbles : ℕ := 18

/-- The total number of marbles -/
def total_marbles : ℕ := blue_marbles + yellow_marbles

/-- The number of ways to arrange the marbles -/
def arrangements : ℕ := Nat.choose total_marbles blue_marbles

theorem marble_arrangement_mod_1000 :
  arrangements % 1000 = 700 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_mod_1000_l2738_273886


namespace NUMINAMATH_CALUDE_correct_commission_calculation_l2738_273849

/-- Calculates the total commission for a salesperson selling appliances -/
def calculate_commission (num_appliances : ℕ) (total_selling_price : ℚ) : ℚ :=
  let fixed_commission := 50 * num_appliances
  let percentage_commission := 0.1 * total_selling_price
  fixed_commission + percentage_commission

/-- Theorem stating the correct commission calculation for the given scenario -/
theorem correct_commission_calculation :
  calculate_commission 6 3620 = 662 := by
  sorry

end NUMINAMATH_CALUDE_correct_commission_calculation_l2738_273849


namespace NUMINAMATH_CALUDE_expression_value_l2738_273851

theorem expression_value (x : ℝ) (h : x^2 - 4*x - 1 = 0) : 
  (x - 3) / (x - 4) - 1 / x = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2738_273851


namespace NUMINAMATH_CALUDE_fourth_root_13824000_l2738_273894

theorem fourth_root_13824000 : (62 : ℕ)^4 = 13824000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_13824000_l2738_273894


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2738_273874

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 < x + 6} = {x : ℝ | -2 < x ∧ x < 3} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2738_273874


namespace NUMINAMATH_CALUDE_x_minus_y_values_l2738_273835

theorem x_minus_y_values (x y : ℝ) (h1 : |x + 1| = 4) (h2 : (y + 2)^2 = 0) :
  x - y = 5 ∨ x - y = -3 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l2738_273835


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2738_273818

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, n < 1000 → n ≥ 100 → n % 17 = 0 → n ≤ 986 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2738_273818


namespace NUMINAMATH_CALUDE_negation_of_set_implication_l2738_273878

theorem negation_of_set_implication (A B : Set α) :
  ¬(A ∪ B = A → A ∩ B = B) ↔ (A ∪ B ≠ A → A ∩ B ≠ B) :=
sorry

end NUMINAMATH_CALUDE_negation_of_set_implication_l2738_273878


namespace NUMINAMATH_CALUDE_f_one_half_equals_two_l2738_273864

-- Define the function f
noncomputable def f (y : ℝ) : ℝ := (4 : ℝ) ^ y

-- State the theorem
theorem f_one_half_equals_two :
  f (1/2) = 2 :=
sorry

end NUMINAMATH_CALUDE_f_one_half_equals_two_l2738_273864


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2738_273899

theorem rectangle_dimensions (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (harea : x * y = 36) (hperim : 2 * x + 2 * y = 30) :
  (x = 12 ∧ y = 3) ∨ (x = 3 ∧ y = 12) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2738_273899


namespace NUMINAMATH_CALUDE_max_blue_points_l2738_273832

/-- The maximum number of blue points when 2016 spheres are colored red or green -/
theorem max_blue_points (total_spheres : Nat) (h : total_spheres = 2016) :
  ∃ (red_spheres : Nat),
    red_spheres ≤ total_spheres ∧
    red_spheres * (total_spheres - red_spheres) = 1016064 ∧
    ∀ (x : Nat), x ≤ total_spheres →
      x * (total_spheres - x) ≤ 1016064 := by
  sorry

end NUMINAMATH_CALUDE_max_blue_points_l2738_273832


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l2738_273848

theorem product_remainder_mod_five :
  (14452 * 15652 * 16781) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l2738_273848


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2738_273887

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x = 7) ∧ 
  (3 * (-1)^2 + m * (-1) = 7) → 
  (∃ x : ℝ, x ≠ -1 ∧ 3 * x^2 + m * x = 7 ∧ x = 7/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2738_273887


namespace NUMINAMATH_CALUDE_girl_scout_pool_trip_expenses_l2738_273882

/-- Girl Scout Pool Trip Expenses Theorem -/
theorem girl_scout_pool_trip_expenses
  (earnings : ℝ)
  (pool_entry_cost : ℝ)
  (transportation_fee : ℝ)
  (snack_cost : ℝ)
  (num_people : ℕ)
  (h1 : earnings = 30)
  (h2 : pool_entry_cost = 2.5)
  (h3 : transportation_fee = 1.25)
  (h4 : snack_cost = 3)
  (h5 : num_people = 10) :
  earnings - (pool_entry_cost + transportation_fee + snack_cost) * num_people = -37.5 :=
sorry

end NUMINAMATH_CALUDE_girl_scout_pool_trip_expenses_l2738_273882


namespace NUMINAMATH_CALUDE_tiger_speed_l2738_273879

/-- Proves that the tiger's speed is 30 kmph given the problem conditions -/
theorem tiger_speed (tiger_head_start : ℝ) (zebra_chase_time : ℝ) (zebra_speed : ℝ)
  (h1 : tiger_head_start = 5)
  (h2 : zebra_chase_time = 6)
  (h3 : zebra_speed = 55) :
  tiger_head_start * (zebra_speed * zebra_chase_time / (tiger_head_start + zebra_chase_time)) = 30 * tiger_head_start :=
by sorry

end NUMINAMATH_CALUDE_tiger_speed_l2738_273879
