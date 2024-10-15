import Mathlib

namespace NUMINAMATH_CALUDE_max_dot_product_OM_OC_l1312_131293

/-- Given points in a 2D Cartesian coordinate system -/
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (2, -1)

/-- M is a moving point with x-coordinate between -2 and 2 -/
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | -2 ≤ p.1 ∧ p.1 ≤ 2}

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem: The maximum value of OM · OC is 4 -/
theorem max_dot_product_OM_OC :
  ∃ (m : ℝ × ℝ), m ∈ M ∧ 
    ∀ (n : ℝ × ℝ), n ∈ M → 
      dot_product (m.1 - O.1, m.2 - O.2) (C.1 - O.1, C.2 - O.2) ≥ 
      dot_product (n.1 - O.1, n.2 - O.2) (C.1 - O.1, C.2 - O.2) ∧
    dot_product (m.1 - O.1, m.2 - O.2) (C.1 - O.1, C.2 - O.2) = 4 :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_OM_OC_l1312_131293


namespace NUMINAMATH_CALUDE_certain_number_sum_l1312_131263

theorem certain_number_sum (n : ℕ) : 
  (n % 423 = 0) → 
  (n / 423 = 423 - 421) → 
  (n + 421 = 1267) := by
sorry

end NUMINAMATH_CALUDE_certain_number_sum_l1312_131263


namespace NUMINAMATH_CALUDE_true_compound_propositions_l1312_131288

-- Define propositions
variable (p₁ p₂ p₃ p₄ : Prop)

-- Define the truth values of the propositions
axiom p₁_true : p₁
axiom p₂_false : ¬p₂
axiom p₃_false : ¬p₃
axiom p₄_true : p₄

-- Theorem to prove
theorem true_compound_propositions :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) :=
by sorry

end NUMINAMATH_CALUDE_true_compound_propositions_l1312_131288


namespace NUMINAMATH_CALUDE_parabola_line_intersection_chord_length_l1312_131297

/-- Represents a parabola with equation y² = 6x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  eq_def : equation = fun y x => y^2 = 6*x

/-- Represents a line with a 45° inclination passing through a point -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ
  eq_def : slope = 1

/-- The length of the chord formed by the intersection of a parabola and a line -/
def chord_length (p : Parabola) (l : Line) : ℝ := 12

/-- Theorem: The length of the chord formed by the intersection of the parabola y² = 6x
    and a line passing through its focus with a 45° inclination is 12 -/
theorem parabola_line_intersection_chord_length (p : Parabola) (l : Line) :
  p.equation = fun y x => y^2 = 6*x →
  l.point = (3/2, 0) →
  l.slope = 1 →
  chord_length p l = 12 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_chord_length_l1312_131297


namespace NUMINAMATH_CALUDE_factor_million_three_ways_l1312_131275

/-- The number of ways to factor 1,000,000 into three factors, ignoring order -/
def factor_ways : ℕ := 139

/-- The prime factorization of 1,000,000 -/
def million_factorization : ℕ × ℕ := (6, 6)

theorem factor_million_three_ways :
  let (a, b) := million_factorization
  (2^a * 5^b = 1000000) →
  (factor_ways = 
    (1 : ℕ) + -- case where all factors are equal
    15 + -- case where exactly two factors are equal
    ((28 * 28 - 15 * 3 - 1) / 6 : ℕ) -- case where all factors are different
  ) := by sorry

end NUMINAMATH_CALUDE_factor_million_three_ways_l1312_131275


namespace NUMINAMATH_CALUDE_warehouse_optimization_l1312_131285

/-- Represents the warehouse dimensions and costs -/
structure Warehouse where
  x : ℝ  -- length of the iron fence (front)
  y : ℝ  -- length of one brick wall (side)
  iron_cost : ℝ := 40  -- cost per meter of iron fence
  brick_cost : ℝ := 45  -- cost per meter of brick wall
  top_cost : ℝ := 20  -- cost per square meter of the top
  budget : ℝ := 3200  -- total budget

/-- The total cost of the warehouse -/
def total_cost (w : Warehouse) : ℝ :=
  w.iron_cost * w.x + 2 * w.brick_cost * w.y + w.top_cost * w.x * w.y

/-- The area of the warehouse -/
def area (w : Warehouse) : ℝ :=
  w.x * w.y

/-- Theorem stating the maximum area and optimal dimensions -/
theorem warehouse_optimization (w : Warehouse) :
  (∀ w' : Warehouse, total_cost w' ≤ w.budget → area w' ≤ 100) ∧
  (∃ w' : Warehouse, total_cost w' ≤ w.budget ∧ area w' = 100) ∧
  (area w = 100 → total_cost w ≤ w.budget → w.x = 15) :=
sorry

end NUMINAMATH_CALUDE_warehouse_optimization_l1312_131285


namespace NUMINAMATH_CALUDE_purification_cost_is_one_l1312_131266

/-- The cost to purify a gallon of fresh water -/
def purification_cost (water_per_person : ℚ) (family_size : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (water_per_person * family_size)

/-- Theorem: The cost to purify a gallon of fresh water is $1 -/
theorem purification_cost_is_one :
  purification_cost (1/2) 6 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_purification_cost_is_one_l1312_131266


namespace NUMINAMATH_CALUDE_ring_toss_revenue_l1312_131242

/-- The daily revenue of a ring toss game at a carnival -/
def daily_revenue (total_revenue : ℕ) (num_days : ℕ) : ℚ :=
  total_revenue / num_days

/-- Theorem stating that the daily revenue is 140 given the conditions -/
theorem ring_toss_revenue :
  daily_revenue 420 3 = 140 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_revenue_l1312_131242


namespace NUMINAMATH_CALUDE_function_upper_bound_l1312_131203

open Real

theorem function_upper_bound 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : a ∈ Set.Icc (-1 / Real.exp 1) 0) 
  (h2 : ∀ x > 0, f x = (x + 1) / Real.exp x - a * log x) :
  ∀ x ∈ Set.Ioo 0 2, f x < (1 - a - a^2) / Real.exp (-a) := by
sorry

end NUMINAMATH_CALUDE_function_upper_bound_l1312_131203


namespace NUMINAMATH_CALUDE_reciprocal_and_opposite_of_negative_four_l1312_131289

theorem reciprocal_and_opposite_of_negative_four :
  (1 / (-4 : ℝ) = -1/4) ∧ (-((-4) : ℝ) = 4) := by sorry

end NUMINAMATH_CALUDE_reciprocal_and_opposite_of_negative_four_l1312_131289


namespace NUMINAMATH_CALUDE_modulo_congruence_solution_l1312_131222

theorem modulo_congruence_solution :
  ∃! k : ℤ, 0 ≤ k ∧ k < 17 ∧ -175 ≡ k [ZMOD 17] := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_solution_l1312_131222


namespace NUMINAMATH_CALUDE_river_rectification_l1312_131299

theorem river_rectification 
  (total_length : ℝ) 
  (rate_A : ℝ) 
  (rate_B : ℝ) 
  (total_time : ℝ) 
  (h1 : total_length = 180)
  (h2 : rate_A = 8)
  (h3 : rate_B = 12)
  (h4 : total_time = 20) :
  ∃ (length_A length_B : ℝ),
    length_A + length_B = total_length ∧
    length_A / rate_A + length_B / rate_B = total_time ∧
    length_A = 120 ∧
    length_B = 60 :=
by sorry

end NUMINAMATH_CALUDE_river_rectification_l1312_131299


namespace NUMINAMATH_CALUDE_original_number_proof_l1312_131201

theorem original_number_proof (x : ℝ) : 3 * (2 * x + 5) = 129 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1312_131201


namespace NUMINAMATH_CALUDE_arbor_day_saplings_l1312_131282

theorem arbor_day_saplings 
  (rate_A rate_B : ℚ) 
  (saplings_A saplings_B : ℕ) : 
  rate_A = (3 : ℚ) / 4 * rate_B → 
  saplings_B = saplings_A + 36 → 
  saplings_A + saplings_B = 252 := by
sorry

end NUMINAMATH_CALUDE_arbor_day_saplings_l1312_131282


namespace NUMINAMATH_CALUDE_arithmetic_mean_function_constant_l1312_131267

/-- A function from ℤ² to ℤ⁺ satisfying the arithmetic mean property -/
def ArithmeticMeanFunction (f : ℤ × ℤ → ℕ+) : Prop :=
  ∀ x y : ℤ, (f (x, y) : ℚ) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4

/-- If a function satisfies the arithmetic mean property, then it is constant -/
theorem arithmetic_mean_function_constant (f : ℤ × ℤ → ℕ+) 
  (h : ArithmeticMeanFunction f) : 
  ∀ p q : ℤ × ℤ, f p = f q :=
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_function_constant_l1312_131267


namespace NUMINAMATH_CALUDE_cats_in_meow_and_paw_l1312_131226

/-- The number of cats in Cat Cafe Cool -/
def cool_cats : ℕ := 5

/-- The number of cats in Cat Cafe Paw -/
def paw_cats : ℕ := 2 * cool_cats

/-- The number of cats in Cat Cafe Meow -/
def meow_cats : ℕ := 3 * paw_cats

/-- The total number of cats in Cat Cafe Meow and Cat Cafe Paw -/
def total_cats : ℕ := meow_cats + paw_cats

/-- Theorem stating that the total number of cats in Cat Cafe Meow and Cat Cafe Paw is 40 -/
theorem cats_in_meow_and_paw : total_cats = 40 := by
  sorry

end NUMINAMATH_CALUDE_cats_in_meow_and_paw_l1312_131226


namespace NUMINAMATH_CALUDE_sales_tax_difference_example_l1312_131270

/-- The difference between two sales tax amounts on a given price -/
def salesTaxDifference (price : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  price * rate2 - price * rate1

/-- Theorem stating that the difference between 8% and 7.5% sales tax on $50 is $0.25 -/
theorem sales_tax_difference_example : salesTaxDifference 50 0.075 0.08 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_example_l1312_131270


namespace NUMINAMATH_CALUDE_real_roots_iff_k_leq_one_l1312_131224

theorem real_roots_iff_k_leq_one (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 4 * x + 4 = 0) ↔ k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_iff_k_leq_one_l1312_131224


namespace NUMINAMATH_CALUDE_polynomial_roots_k_values_l1312_131283

/-- The set of all distinct possible values of k for the polynomial x^2 - kx + 36 
    with only positive integer roots -/
def possible_k_values : Set ℤ := {12, 13, 15, 20, 37}

/-- A polynomial of the form x^2 - kx + 36 -/
def polynomial (k : ℤ) (x : ℝ) : ℝ := x^2 - k*x + 36

theorem polynomial_roots_k_values :
  ∀ k : ℤ, (∃ r₁ r₂ : ℤ, r₁ > 0 ∧ r₂ > 0 ∧ 
    ∀ x : ℝ, polynomial k x = 0 ↔ x = r₁ ∨ x = r₂) ↔ 
  k ∈ possible_k_values :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_k_values_l1312_131283


namespace NUMINAMATH_CALUDE_trig_identity_l1312_131272

theorem trig_identity : 
  1 / Real.cos (70 * π / 180) - 2 / Real.sin (70 * π / 180) = 
  2 * Real.tan (50 * π / 180) - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1312_131272


namespace NUMINAMATH_CALUDE_train_crossing_poles_time_l1312_131218

/-- Calculates the total time for a train to cross multiple poles -/
theorem train_crossing_poles_time
  (train_speed : ℝ)
  (first_pole_crossing_time : ℝ)
  (pole_distances : List ℝ)
  (h1 : train_speed = 75)  -- 75 kmph
  (h2 : first_pole_crossing_time = 3)  -- 3 seconds
  (h3 : pole_distances = [500, 800, 1500, 2200]) :  -- distances in meters
  ∃ (total_time : ℝ),
    total_time = 243 ∧  -- 243 seconds
    total_time = first_pole_crossing_time +
      (pole_distances.map (λ d => d / (train_speed * 1000 / 3600))).sum :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_poles_time_l1312_131218


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l1312_131291

/-- The volume of a cube with space diagonal 3√3 is 27 -/
theorem cube_volume_from_space_diagonal :
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 3 * Real.sqrt 3 → s^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l1312_131291


namespace NUMINAMATH_CALUDE_pure_imaginary_sum_l1312_131210

theorem pure_imaginary_sum (a b c d : ℝ) : 
  let z₁ : ℂ := a + b * Complex.I
  let z₂ : ℂ := c + d * Complex.I
  (z₁ + z₂).re = 0 ∧ (z₁ + z₂).im ≠ 0 → a + c = 0 ∧ b + d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_sum_l1312_131210


namespace NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_sixth_l1312_131202

theorem arcsin_one_half_equals_pi_sixth : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_sixth_l1312_131202


namespace NUMINAMATH_CALUDE_total_interest_calculation_l1312_131204

/-- Calculate the total interest for two principal amounts -/
def totalInterest (principal1 principal2 : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal1 + principal2) * rate * time

theorem total_interest_calculation :
  let principal1 : ℝ := 1000
  let principal2 : ℝ := 1400
  let rate : ℝ := 0.03
  let time : ℝ := 4.861111111111111
  abs (totalInterest principal1 principal2 rate time - 350) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_calculation_l1312_131204


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l1312_131292

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 3 = 4 →
  a 7 - 2 * a 5 = 32 →
  a 7 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l1312_131292


namespace NUMINAMATH_CALUDE_function_with_two_symmetries_is_periodic_l1312_131281

/-- A function with two lines of symmetry is periodic -/
theorem function_with_two_symmetries_is_periodic
  (f : ℝ → ℝ) (m n : ℝ) (hm : m ≠ n)
  (sym_m : ∀ x, f x = f (2 * m - x))
  (sym_n : ∀ x, f x = f (2 * n - x)) :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

end NUMINAMATH_CALUDE_function_with_two_symmetries_is_periodic_l1312_131281


namespace NUMINAMATH_CALUDE_system_solution_l1312_131230

theorem system_solution :
  ∀ x y z : ℝ,
  x = Real.sqrt (2 * y + 3) →
  y = Real.sqrt (2 * z + 3) →
  z = Real.sqrt (2 * x + 3) →
  x = 3 ∧ y = 3 ∧ z = 3 :=
by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l1312_131230


namespace NUMINAMATH_CALUDE_original_holes_additional_holes_l1312_131298

-- Define the circumference of the circular road
def circumference : ℕ := 400

-- Define the original interval between streetlamps
def original_interval : ℕ := 50

-- Define the new interval between streetlamps
def new_interval : ℕ := 40

-- Theorem for the number of holes in the original plan
theorem original_holes : circumference / original_interval = 8 := by sorry

-- Theorem for the number of additional holes in the new plan
theorem additional_holes : 
  circumference / new_interval - (circumference / (Nat.lcm original_interval new_interval)) = 8 := by sorry

end NUMINAMATH_CALUDE_original_holes_additional_holes_l1312_131298


namespace NUMINAMATH_CALUDE_volumes_equal_l1312_131265

/-- The region bounded by x² = 4y, x² = -4y, x = 4, x = -4 -/
def Region1 (x y : ℝ) : Prop :=
  (x^2 = 4*y ∨ x^2 = -4*y) ∧ (x ≤ 4 ∧ x ≥ -4)

/-- The region defined by x²y² ≤ 16, x² + (y-2)² ≥ 4, x² + (y+2)² ≥ 4 -/
def Region2 (x y : ℝ) : Prop :=
  x^2 * y^2 ≤ 16 ∧ x^2 + (y-2)^2 ≥ 4 ∧ x^2 + (y+2)^2 ≥ 4

/-- The volume of the solid obtained by rotating Region1 around the y-axis -/
noncomputable def V1 : ℝ := sorry

/-- The volume of the solid obtained by rotating Region2 around the y-axis -/
noncomputable def V2 : ℝ := sorry

/-- The volumes of the two solids are equal -/
theorem volumes_equal : V1 = V2 := by sorry

end NUMINAMATH_CALUDE_volumes_equal_l1312_131265


namespace NUMINAMATH_CALUDE_matinee_children_count_l1312_131231

/-- Proves the number of children at a movie theater matinee --/
theorem matinee_children_count :
  let child_price : ℚ := 9/2
  let adult_price : ℚ := 27/4
  let total_receipts : ℚ := 405
  ∀ (num_adults : ℕ),
    (child_price * (num_adults + 20 : ℚ) + adult_price * num_adults = total_receipts) →
    (num_adults + 20 = 48) :=
by
  sorry

#check matinee_children_count

end NUMINAMATH_CALUDE_matinee_children_count_l1312_131231


namespace NUMINAMATH_CALUDE_circle_product_values_l1312_131251

noncomputable section

open Real Set

def circle_product (α β : ℝ × ℝ) : ℝ := 
  (α.1 * β.1 + α.2 * β.2) / (β.1 * β.1 + β.2 * β.2)

def angle (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem circle_product_values (a b : ℝ × ℝ) 
  (h1 : a ≠ (0, 0)) 
  (h2 : b ≠ (0, 0)) 
  (h3 : π/6 < angle a b ∧ angle a b < π/2) 
  (h4 : ∃ n : ℤ, circle_product a b = n/2) 
  (h5 : ∃ m : ℤ, circle_product b a = m/2) : 
  circle_product a b = 1 ∨ circle_product a b = 1/2 := by
sorry

end

end NUMINAMATH_CALUDE_circle_product_values_l1312_131251


namespace NUMINAMATH_CALUDE_circle_sector_area_equality_l1312_131244

theorem circle_sector_area_equality (r : ℝ) (φ : ℝ) 
  (h1 : 0 < r) (h2 : 0 < φ) (h3 : φ < π / 4) :
  (φ * r^2 / 2 + r^2 * Real.sin φ / 2 = φ * r^2 + r^2 * Real.sin (2 * φ) / 2) ↔ φ = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_sector_area_equality_l1312_131244


namespace NUMINAMATH_CALUDE_max_value_inequality_l1312_131213

theorem max_value_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + 2*b + 3*c = 6) : 
  Real.sqrt (a + 1) + Real.sqrt (2*b + 1) + Real.sqrt (3*c + 1) ≤ 3 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1312_131213


namespace NUMINAMATH_CALUDE_min_value_A_l1312_131241

theorem min_value_A (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_abc : a + b + c = 3) :
  let A := (a^3 + b^3)/(8*a*b + 9 - c^2) + (b^3 + c^3)/(8*b*c + 9 - a^2) + (c^3 + a^3)/(8*c*a + 9 - b^2)
  A ≥ 3/8 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 3 ∧
    (a₀^3 + b₀^3)/(8*a₀*b₀ + 9 - c₀^2) + (b₀^3 + c₀^3)/(8*b₀*c₀ + 9 - a₀^2) + (c₀^3 + a₀^3)/(8*c₀*a₀ + 9 - b₀^2) = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_A_l1312_131241


namespace NUMINAMATH_CALUDE_water_molecule_radius_scientific_notation_l1312_131249

theorem water_molecule_radius_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.00000000192 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -9 :=
by sorry

end NUMINAMATH_CALUDE_water_molecule_radius_scientific_notation_l1312_131249


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1312_131209

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1312_131209


namespace NUMINAMATH_CALUDE_unique_quartic_polynomial_l1312_131243

/-- A monic quartic polynomial with real coefficients -/
def QuarticPolynomial (a b c d : ℝ) : ℝ → ℂ :=
  fun x ↦ (x^4 : ℂ) + a*(x^3 : ℂ) + b*(x^2 : ℂ) + c*(x : ℂ) + d

theorem unique_quartic_polynomial 
  (q : ℝ → ℂ) 
  (monic : q = QuarticPolynomial (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)) 
  (root_complex : q (1 - 3*I) = 0) 
  (root_zero : q 0 = -48) 
  (root_one : q 1 = 0) : 
  q = QuarticPolynomial (-7.8) 25.4 (-23.8) 48 := by
  sorry

end NUMINAMATH_CALUDE_unique_quartic_polynomial_l1312_131243


namespace NUMINAMATH_CALUDE_fixed_point_linear_function_l1312_131235

theorem fixed_point_linear_function (k : ℝ) : 
  (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_linear_function_l1312_131235


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1312_131257

/-- Hyperbola with center at origin, focus at (3,0), and intersection points with midpoint (-12,-15) -/
def Hyperbola (E : Set (ℝ × ℝ)) : Prop :=
  ∃ (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)),
    (0, 0) ∈ E ∧  -- Center at origin
    (3, 0) ∈ E ∧  -- Focus at (3,0)
    (A ∈ E ∧ B ∈ E) ∧  -- A and B are on the hyperbola
    (A ∈ l ∧ B ∈ l ∧ (3, 0) ∈ l) ∧  -- A, B, and focus are on line l
    ((A.1 + B.1) / 2 = -12 ∧ (A.2 + B.2) / 2 = -15)  -- Midpoint of A and B is (-12,-15)

/-- The equation of the hyperbola -/
def HyperbolaEquation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

theorem hyperbola_equation (E : Set (ℝ × ℝ)) (h : Hyperbola E) :
  ∀ (x y : ℝ), (x, y) ∈ E ↔ HyperbolaEquation x y := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1312_131257


namespace NUMINAMATH_CALUDE_andrea_pony_pasture_cost_l1312_131287

/-- Calculates the monthly pasture cost for Andrea's pony --/
theorem andrea_pony_pasture_cost :
  let daily_food_cost : ℕ := 10
  let lesson_cost : ℕ := 60
  let lessons_per_week : ℕ := 2
  let total_annual_expense : ℕ := 15890
  let days_per_year : ℕ := 365
  let weeks_per_year : ℕ := 52

  let annual_food_cost := daily_food_cost * days_per_year
  let annual_lesson_cost := lesson_cost * lessons_per_week * weeks_per_year
  let annual_pasture_cost := total_annual_expense - (annual_food_cost + annual_lesson_cost)
  let monthly_pasture_cost := annual_pasture_cost / 12

  monthly_pasture_cost = 500 := by sorry

end NUMINAMATH_CALUDE_andrea_pony_pasture_cost_l1312_131287


namespace NUMINAMATH_CALUDE_ana_dress_count_l1312_131216

/-- The number of dresses Ana has -/
def ana_dresses : ℕ := 15

/-- The number of dresses Lisa has -/
def lisa_dresses : ℕ := ana_dresses + 18

/-- The total number of dresses Ana and Lisa have combined -/
def total_dresses : ℕ := 48

theorem ana_dress_count : ana_dresses = 15 := by sorry

end NUMINAMATH_CALUDE_ana_dress_count_l1312_131216


namespace NUMINAMATH_CALUDE_smallest_number_for_2_and_4_l1312_131239

def smallest_number (a b : ℕ) : ℕ := 
  if a ≤ b then 10 * a + b else 10 * b + a

theorem smallest_number_for_2_and_4 : 
  smallest_number 2 4 = 24 := by sorry

end NUMINAMATH_CALUDE_smallest_number_for_2_and_4_l1312_131239


namespace NUMINAMATH_CALUDE_divisibility_implies_power_l1312_131228

theorem divisibility_implies_power (m n : ℕ+) 
  (h : (m * n) ∣ (m ^ 2010 + n ^ 2010 + n)) :
  ∃ k : ℕ+, n = k ^ 2010 := by sorry

end NUMINAMATH_CALUDE_divisibility_implies_power_l1312_131228


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1312_131260

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  (a 20) ^ 2 - 10 * (a 20) + 16 = 0 →                   -- a_20 is a root
  (a 60) ^ 2 - 10 * (a 60) + 16 = 0 →                   -- a_60 is a root
  (a 30 * a 40 * a 50) / 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1312_131260


namespace NUMINAMATH_CALUDE_complement_union_sets_l1312_131254

def I : Finset ℕ := {0,1,2,3,4,5,6,7,8}
def M : Finset ℕ := {1,2,4,5}
def N : Finset ℕ := {0,3,5,7}

theorem complement_union_sets : (I \ (M ∪ N)) = {6,8} := by sorry

end NUMINAMATH_CALUDE_complement_union_sets_l1312_131254


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_deg_has_20_sides_l1312_131221

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_deg_has_20_sides :
  ∀ n : ℕ,
  n > 0 →
  (360 : ℝ) / n = 18 →
  n = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_deg_has_20_sides_l1312_131221


namespace NUMINAMATH_CALUDE_least_square_tiles_l1312_131200

theorem least_square_tiles (length width : ℕ) (h1 : length = 544) (h2 : width = 374) :
  let tile_size := Nat.gcd length width
  let num_tiles := (length * width) / (tile_size * tile_size)
  num_tiles = 50864 := by
sorry

end NUMINAMATH_CALUDE_least_square_tiles_l1312_131200


namespace NUMINAMATH_CALUDE_three_rug_overlap_l1312_131273

theorem three_rug_overlap (total_area floor_area double_layer_area : ℝ) 
  (h1 : total_area = 200)
  (h2 : floor_area = 140)
  (h3 : double_layer_area = 24) : 
  ∃ (triple_layer_area : ℝ), 
    triple_layer_area = 18 ∧ 
    total_area = floor_area + double_layer_area + 2 * triple_layer_area :=
by
  sorry

end NUMINAMATH_CALUDE_three_rug_overlap_l1312_131273


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1312_131279

theorem sqrt_sum_inequality (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 19/3) :
  Real.sqrt (x - 1) + Real.sqrt (2 * x + 9) + Real.sqrt (19 - 3 * x) < 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1312_131279


namespace NUMINAMATH_CALUDE_softball_team_savings_l1312_131214

/-- Calculates the total savings for a softball team's uniform purchase with group discount --/
theorem softball_team_savings
  (team_size : ℕ)
  (brand_a_shirt_cost brand_a_pants_cost brand_a_socks_cost : ℚ)
  (brand_b_shirt_cost brand_b_pants_cost brand_b_socks_cost : ℚ)
  (brand_a_customization_cost brand_b_customization_cost : ℚ)
  (brand_a_group_shirt_cost brand_a_group_pants_cost brand_a_group_socks_cost : ℚ)
  (brand_b_group_shirt_cost brand_b_group_pants_cost brand_b_group_socks_cost : ℚ)
  (individual_socks_players non_customized_shirts_players brand_b_socks_players : ℕ)
  (h1 : team_size = 12)
  (h2 : brand_a_shirt_cost = 7.5)
  (h3 : brand_a_pants_cost = 15)
  (h4 : brand_a_socks_cost = 4.5)
  (h5 : brand_b_shirt_cost = 10)
  (h6 : brand_b_pants_cost = 20)
  (h7 : brand_b_socks_cost = 6)
  (h8 : brand_a_customization_cost = 6)
  (h9 : brand_b_customization_cost = 8)
  (h10 : brand_a_group_shirt_cost = 6.5)
  (h11 : brand_a_group_pants_cost = 13)
  (h12 : brand_a_group_socks_cost = 4)
  (h13 : brand_b_group_shirt_cost = 8.5)
  (h14 : brand_b_group_pants_cost = 17)
  (h15 : brand_b_group_socks_cost = 5)
  (h16 : individual_socks_players = 3)
  (h17 : non_customized_shirts_players = 2)
  (h18 : brand_b_socks_players = 1) :
  (team_size * (brand_a_shirt_cost + brand_a_customization_cost + brand_b_pants_cost + brand_a_socks_cost)) -
  (team_size * (brand_a_group_shirt_cost + brand_a_customization_cost + brand_b_group_pants_cost + brand_a_group_socks_cost) +
   individual_socks_players * (brand_a_socks_cost - brand_a_group_socks_cost) -
   non_customized_shirts_players * brand_a_customization_cost +
   brand_b_socks_players * (brand_b_socks_cost - brand_a_group_socks_cost)) = 46.5 := by
  sorry


end NUMINAMATH_CALUDE_softball_team_savings_l1312_131214


namespace NUMINAMATH_CALUDE_shirt_cost_problem_l1312_131232

/-- Proves that the original cost of one of the remaining shirts is $12.50 -/
theorem shirt_cost_problem (total_original_cost : ℝ) (discounted_shirt_price : ℝ) 
  (discount_rate : ℝ) (current_total_cost : ℝ) :
  total_original_cost = 100 →
  discounted_shirt_price = 25 →
  discount_rate = 0.4 →
  current_total_cost = 85 →
  ∃ (remaining_shirt_cost : ℝ),
    remaining_shirt_cost = 12.5 ∧
    3 * discounted_shirt_price * (1 - discount_rate) + 2 * remaining_shirt_cost = current_total_cost ∧
    3 * discounted_shirt_price + 2 * remaining_shirt_cost = total_original_cost :=
by
  sorry


end NUMINAMATH_CALUDE_shirt_cost_problem_l1312_131232


namespace NUMINAMATH_CALUDE_special_dog_food_weight_l1312_131259

/-- The weight of each bag of special dog food for a puppy -/
theorem special_dog_food_weight :
  let first_period_days : ℕ := 60
  let total_days : ℕ := 365
  let first_period_consumption : ℕ := 2  -- ounces per day
  let second_period_consumption : ℕ := 4  -- ounces per day
  let ounces_per_pound : ℕ := 16
  let number_of_bags : ℕ := 17
  
  let total_consumption : ℕ := 
    first_period_days * first_period_consumption + 
    (total_days - first_period_days) * second_period_consumption
  
  let total_pounds : ℚ := total_consumption / ounces_per_pound
  let bag_weight : ℚ := total_pounds / number_of_bags
  
  ∃ (weight : ℚ), abs (weight - bag_weight) < 0.005 ∧ weight = 4.93 :=
by sorry

end NUMINAMATH_CALUDE_special_dog_food_weight_l1312_131259


namespace NUMINAMATH_CALUDE_fingernail_growth_rate_l1312_131255

/-- Proves that the rate of fingernail growth is 0.1 inch per month given the specified conditions. -/
theorem fingernail_growth_rate 
  (current_age : ℕ) 
  (record_age : ℕ) 
  (current_length : ℚ) 
  (record_length : ℚ) 
  (h1 : current_age = 12) 
  (h2 : record_age = 32) 
  (h3 : current_length = 2) 
  (h4 : record_length = 26) : 
  (record_length - current_length) / ((record_age - current_age) * 12 : ℚ) = 1/10 := by
  sorry

#eval (26 - 2 : ℚ) / ((32 - 12) * 12 : ℚ)

end NUMINAMATH_CALUDE_fingernail_growth_rate_l1312_131255


namespace NUMINAMATH_CALUDE_min_value_theorem_l1312_131276

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  1/a + 2/b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 1 ∧ 1/a₀ + 2/b₀ = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1312_131276


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1312_131220

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h : (Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 2) : 
  Real.tan (α + π / 4) = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1312_131220


namespace NUMINAMATH_CALUDE_oliver_shelf_capacity_l1312_131236

/-- Given the total number of books, the number of books taken by the librarian,
    and the number of shelves needed, calculate the number of books that can fit on each shelf. -/
def books_per_shelf (total_books : ℕ) (books_taken : ℕ) (shelves_needed : ℕ) : ℕ :=
  (total_books - books_taken) / shelves_needed

/-- Prove that Oliver can fit 4 books on each shelf given the problem conditions. -/
theorem oliver_shelf_capacity :
  books_per_shelf 46 10 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_oliver_shelf_capacity_l1312_131236


namespace NUMINAMATH_CALUDE_angle_sine_relation_l1312_131269

theorem angle_sine_relation (A B : Real) (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2) :
  A > B ↔ Real.sin A > Real.sin B :=
sorry

end NUMINAMATH_CALUDE_angle_sine_relation_l1312_131269


namespace NUMINAMATH_CALUDE_son_work_time_l1312_131219

/-- Given a man and his son working on a job, this theorem proves how long it takes the son to complete the job alone. -/
theorem son_work_time (man_time son_time combined_time : ℚ)
  (hman : man_time = 5)
  (hcombined : combined_time = 4)
  (hwork : (1 / man_time) + (1 / son_time) = 1 / combined_time) :
  son_time = 20 := by
  sorry

#check son_work_time

end NUMINAMATH_CALUDE_son_work_time_l1312_131219


namespace NUMINAMATH_CALUDE_table_runner_coverage_l1312_131237

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) 
  (two_layer_area : ℝ) (three_layer_area : ℝ) 
  (h1 : total_runner_area = 204)
  (h2 : table_area = 175)
  (h3 : two_layer_area = 24)
  (h4 : three_layer_area = 20) :
  (((total_runner_area - 2 * two_layer_area - 3 * three_layer_area) + 
    two_layer_area + three_layer_area) / table_area) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_table_runner_coverage_l1312_131237


namespace NUMINAMATH_CALUDE_quadratic_solution_existence_l1312_131253

theorem quadratic_solution_existence (a : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + a = 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_existence_l1312_131253


namespace NUMINAMATH_CALUDE_fraction_equality_l1312_131280

theorem fraction_equality : (3 * 4 * 5) / (2 * 3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1312_131280


namespace NUMINAMATH_CALUDE_given_equation_is_quadratic_l1312_131290

/-- A quadratic equation is of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation x² = 2x - 3x² -/
def given_equation (x : ℝ) : ℝ := x^2 - 2*x + 3*x^2

theorem given_equation_is_quadratic :
  is_quadratic_equation given_equation :=
sorry

end NUMINAMATH_CALUDE_given_equation_is_quadratic_l1312_131290


namespace NUMINAMATH_CALUDE_two_possible_w_values_l1312_131295

theorem two_possible_w_values 
  (w : ℂ) 
  (h_exists : ∃ (u v : ℂ), u ≠ v ∧ ∀ (z : ℂ), (z - u) * (z - v) = (z - w * u) * (z - w * v)) : 
  w = 1 ∨ w = -1 :=
sorry

end NUMINAMATH_CALUDE_two_possible_w_values_l1312_131295


namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l1312_131247

/-- Given a parabola y = ax^2 where a < 0, its latus rectum has the equation y = -1/(4a) -/
theorem latus_rectum_of_parabola (a : ℝ) (h : a < 0) :
  let parabola := λ x : ℝ => a * x^2
  let latus_rectum := λ y : ℝ => y = -1 / (4 * a)
  ∀ x : ℝ, latus_rectum (parabola x) := by sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l1312_131247


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l1312_131274

theorem solve_system_of_equations (x y z : ℤ) 
  (eq1 : 4 * x + y + z = 80)
  (eq2 : 2 * x - y - z = 40)
  (eq3 : 3 * x + y - z = 20) :
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l1312_131274


namespace NUMINAMATH_CALUDE_canoe_upstream_speed_l1312_131284

/-- The speed of a canoe rowing upstream, given its downstream speed and the stream speed -/
theorem canoe_upstream_speed (downstream_speed stream_speed : ℝ) : 
  downstream_speed = 10 → stream_speed = 2 → 
  downstream_speed - 2 * stream_speed = 6 := by sorry

end NUMINAMATH_CALUDE_canoe_upstream_speed_l1312_131284


namespace NUMINAMATH_CALUDE_right_handed_players_count_l1312_131225

/-- Calculates the number of right-handed players on a cricket team given specific conditions -/
theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) (left_handed_thrower_percent : ℚ)
  (right_handed_thrower_avg : ℕ) (left_handed_thrower_avg : ℕ) (total_runs : ℕ)
  (left_handed_non_thrower_runs : ℕ) :
  total_players = 120 →
  throwers = 55 →
  left_handed_thrower_percent = 1/5 →
  right_handed_thrower_avg = 25 →
  left_handed_thrower_avg = 30 →
  total_runs = 3620 →
  left_handed_non_thrower_runs = 720 →
  ∃ (right_handed_players : ℕ), right_handed_players = 164 :=
by sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l1312_131225


namespace NUMINAMATH_CALUDE_dormitory_students_l1312_131268

theorem dormitory_students (F S : ℝ) (h1 : F + S = 1) 
  (h2 : 4/5 * F = F - F/5) 
  (h3 : S - 4 * (F/5) = 4/5 * F) 
  (h4 : S - (S - 4 * (F/5)) = 0.2) : 
  S = 2/3 := by sorry

end NUMINAMATH_CALUDE_dormitory_students_l1312_131268


namespace NUMINAMATH_CALUDE_problem_solution_l1312_131223

theorem problem_solution (x y : ℝ) : (x - 1)^2 + Real.sqrt (y + 2) = 0 → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1312_131223


namespace NUMINAMATH_CALUDE_problem_solution_l1312_131252

theorem problem_solution (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : x * y = 9) 
  (h3 : x < y) : 
  (Real.sqrt x - Real.sqrt y) / (Real.sqrt x + Real.sqrt y) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1312_131252


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l1312_131246

theorem square_perimeter_sum (y : ℝ) (h1 : y^2 + (2*y)^2 = 145) (h2 : (2*y)^2 - y^2 = 105) :
  4*y + 8*y = 12 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l1312_131246


namespace NUMINAMATH_CALUDE_min_sum_at_6_l1312_131278

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = -14
  sum_5_6 : a 5 + a 6 = -4
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The value of n for which the sum reaches its minimum -/
def min_sum_index (seq : ArithmeticSequence) : ℕ :=
  6

/-- Theorem: The sum of the arithmetic sequence reaches its minimum when n = 6 -/
theorem min_sum_at_6 (seq : ArithmeticSequence) :
  ∀ n : ℕ, sum_n seq (min_sum_index seq) ≤ sum_n seq n :=
sorry

end NUMINAMATH_CALUDE_min_sum_at_6_l1312_131278


namespace NUMINAMATH_CALUDE_candy_mixture_theorem_l1312_131229

def candy_mixture (initial_blue initial_red added_blue added_red final_blue : ℚ) : Prop :=
  ∃ (x y : ℚ),
    x > 0 ∧ y > 0 ∧
    initial_blue = 1/10 ∧
    initial_red = 1/4 ∧
    added_blue = 1/4 ∧
    added_red = 3/4 ∧
    (initial_blue * x + added_blue * y) / (x + y) = final_blue

theorem candy_mixture_theorem :
  ∀ initial_blue initial_red added_blue added_red final_blue,
    candy_mixture initial_blue initial_red added_blue added_red final_blue →
    final_blue = 4/25 →
    ∃ final_red : ℚ, final_red = 9/20 :=
by sorry

end NUMINAMATH_CALUDE_candy_mixture_theorem_l1312_131229


namespace NUMINAMATH_CALUDE_probability_red_second_draw_three_five_l1312_131238

/-- Represents the color of a ball -/
inductive Color
  | Red
  | Blue

/-- Represents a box of balls -/
structure Box where
  total : Nat
  red : Nat
  blue : Nat
  h_total : total = red + blue

/-- Calculates the probability of drawing a red ball on the second draw -/
def probability_red_second_draw (box : Box) : Rat :=
  (box.red * (box.total - 1) + box.blue * box.red) / (box.total * (box.total - 1))

/-- Theorem stating the probability of drawing a red ball on the second draw -/
theorem probability_red_second_draw_three_five :
  let box : Box := ⟨5, 3, 2, rfl⟩
  probability_red_second_draw box = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_second_draw_three_five_l1312_131238


namespace NUMINAMATH_CALUDE_xiao_li_commute_l1312_131212

/-- Xiao Li's commute problem -/
theorem xiao_li_commute 
  (distance : ℝ) 
  (walk_late : ℝ) 
  (bike_early : ℝ) 
  (bike_speed_factor : ℝ) 
  (breakdown_distance : ℝ) 
  (early_arrival : ℝ)
  (h1 : distance = 4.5)
  (h2 : walk_late = 5 / 60)
  (h3 : bike_early = 10 / 60)
  (h4 : bike_speed_factor = 1.5)
  (h5 : breakdown_distance = 1.5)
  (h6 : early_arrival = 5 / 60) :
  ∃ (walk_speed bike_speed min_run_speed : ℝ),
    walk_speed = 6 ∧ 
    bike_speed = 9 ∧ 
    min_run_speed = 7.2 ∧
    distance / walk_speed - walk_late = distance / bike_speed + bike_early ∧
    bike_speed = bike_speed_factor * walk_speed ∧
    breakdown_distance + (distance / bike_speed + bike_early - breakdown_distance / bike_speed - early_arrival) * min_run_speed ≥ distance :=
by
  sorry

end NUMINAMATH_CALUDE_xiao_li_commute_l1312_131212


namespace NUMINAMATH_CALUDE_chinese_digit_mapping_l1312_131277

/-- A function that maps Chinese characters to unique digits 1-9 -/
def ChineseToDigit : Type := Char → Fin 9

/-- The condition that the function maps different characters to different digits -/
def isInjective (f : ChineseToDigit) : Prop :=
  ∀ (c1 c2 : Char), f c1 = f c2 → c1 = c2

/-- The theorem statement -/
theorem chinese_digit_mapping (f : ChineseToDigit) 
  (h_injective : isInjective f)
  (h_zhu : f '祝' = 4)
  (h_he : f '贺' = 8) :
  (f '华') * 100 + (f '杯') * 10 + (f '赛') = 7632 := by
  sorry


end NUMINAMATH_CALUDE_chinese_digit_mapping_l1312_131277


namespace NUMINAMATH_CALUDE_even_quadratic_function_l1312_131240

/-- A function f is even on an interval if f(-x) = f(x) for all x in the interval -/
def IsEvenOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x, x ∈ [min (-b) (-a), max a b] → f (-x) = f x

theorem even_quadratic_function (a : ℝ) :
  let f := fun x ↦ a * x^2 + 1
  IsEvenOn f (3 - a) 5 → a = 8 := by
sorry

end NUMINAMATH_CALUDE_even_quadratic_function_l1312_131240


namespace NUMINAMATH_CALUDE_paper_goods_cost_l1312_131233

/-- Given that 100 paper plates and 200 paper cups cost $6.00, 
    prove that 20 paper plates and 40 paper cups cost $1.20 -/
theorem paper_goods_cost (plate_cost cup_cost : ℝ) 
    (h : 100 * plate_cost + 200 * cup_cost = 6) :
  20 * plate_cost + 40 * cup_cost = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_paper_goods_cost_l1312_131233


namespace NUMINAMATH_CALUDE_a_fourth_plus_b_fourth_l1312_131286

theorem a_fourth_plus_b_fourth (a b : ℝ) 
  (h1 : a^2 - b^2 = 5) 
  (h2 : a * b = 2) : 
  a^4 + b^4 = 833 := by sorry

end NUMINAMATH_CALUDE_a_fourth_plus_b_fourth_l1312_131286


namespace NUMINAMATH_CALUDE_min_product_a_purchase_l1312_131217

theorem min_product_a_purchase (cost_a cost_b total_items max_cost : ℕ) 
  (h1 : cost_a = 20)
  (h2 : cost_b = 50)
  (h3 : total_items = 10)
  (h4 : max_cost = 350) : 
  ∃ min_a : ℕ, min_a = 5 ∧ 
  ∀ x : ℕ, (x ≤ total_items ∧ x * cost_a + (total_items - x) * cost_b ≤ max_cost) → x ≥ min_a := by
  sorry

end NUMINAMATH_CALUDE_min_product_a_purchase_l1312_131217


namespace NUMINAMATH_CALUDE_toy_store_problem_l1312_131215

/-- Toy store problem -/
theorem toy_store_problem 
  (purchase_price : ℝ) 
  (base_price : ℝ) 
  (base_sales : ℝ) 
  (price_increment : ℝ) 
  (sales_decrement : ℝ) 
  (max_price : ℝ) 
  (max_cost : ℝ) 
  (profit : ℝ) :
  purchase_price = 49 →
  base_price = 50 →
  base_sales = 50 →
  price_increment = 0.5 →
  sales_decrement = 3 →
  max_price = 60 →
  max_cost = 686 →
  profit = 147 →
  ∃ (x : ℝ) (a : ℝ),
    -- Part 1: Price range
    56 ≤ x ∧ x ≤ 60 ∧
    x ≤ max_price ∧
    purchase_price * (base_sales - sales_decrement * ((x - base_price) / price_increment)) ≤ max_cost ∧
    -- Part 2: Value of a
    a = 25 ∧
    (x * (1 + a / 100) - purchase_price) * (base_sales - sales_decrement * ((x - base_price) / price_increment)) * (1 - 2 * a / 100) = profit :=
by sorry

end NUMINAMATH_CALUDE_toy_store_problem_l1312_131215


namespace NUMINAMATH_CALUDE_x_squared_equals_three_l1312_131258

theorem x_squared_equals_three (x : ℝ) (h1 : x > 0) (h2 : Real.sin (Real.arctan x) = x / 2) : x^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_equals_three_l1312_131258


namespace NUMINAMATH_CALUDE_log_expression_equality_l1312_131271

theorem log_expression_equality (a b c d e x y : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hx : x > 0) (hy : y > 0) :
  Real.log (a^2 / b) + Real.log (b^3 / c^2) + Real.log (c / d) + Real.log (d^2 / e) - Real.log (a^3 * y / (e^2 * x)) = Real.log ((b^2 * e * x) / (c * a * y)) :=
by sorry

end NUMINAMATH_CALUDE_log_expression_equality_l1312_131271


namespace NUMINAMATH_CALUDE_retail_price_maximizes_profit_l1312_131294

/-- The profit function for a shopping mall selling items -/
def profit_function (p : ℝ) : ℝ := (8300 - 170*p - p^2)*(p - 20)

/-- The derivative of the profit function -/
def profit_derivative (p : ℝ) : ℝ := -3*p^2 - 300*p + 11700

theorem retail_price_maximizes_profit :
  ∃ (p : ℝ), p > 20 ∧ 
  ∀ (q : ℝ), q > 20 → profit_function p ≥ profit_function q ∧
  p = 30 := by
  sorry

#check retail_price_maximizes_profit

end NUMINAMATH_CALUDE_retail_price_maximizes_profit_l1312_131294


namespace NUMINAMATH_CALUDE_largest_after_removal_l1312_131256

/-- Represents the initial number as a string -/
def initial_number : String := "123456789101112131415...99100"

/-- Represents the final number after digit removal as a string -/
def final_number : String := "9999978596061...99100"

/-- Function to remove digits from a string -/
def remove_digits (s : String) (n : Nat) : String := sorry

/-- Function to compare two strings as numbers -/
def compare_as_numbers (s1 s2 : String) : Bool := sorry

/-- Theorem stating that the final_number is the largest possible after removing 100 digits -/
theorem largest_after_removal :
  ∀ (s : String),
    s.length = initial_number.length - 100 →
    s = remove_digits initial_number 100 →
    compare_as_numbers final_number s = true :=
sorry

end NUMINAMATH_CALUDE_largest_after_removal_l1312_131256


namespace NUMINAMATH_CALUDE_dorothy_age_problem_l1312_131248

theorem dorothy_age_problem :
  let dorothy_age : ℕ := 15
  let sister_age : ℕ := dorothy_age / 3
  let years_later : ℕ := 5
  (dorothy_age + years_later) = 2 * (sister_age + years_later) :=
by sorry

end NUMINAMATH_CALUDE_dorothy_age_problem_l1312_131248


namespace NUMINAMATH_CALUDE_congruence_solution_l1312_131261

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 14589 [ZMOD 15] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1312_131261


namespace NUMINAMATH_CALUDE_joe_tax_fraction_l1312_131211

/-- The fraction of income that goes to taxes -/
def tax_fraction (tax_payment : ℚ) (income : ℚ) : ℚ :=
  tax_payment / income

theorem joe_tax_fraction :
  let monthly_tax : ℚ := 848
  let monthly_income : ℚ := 2120
  tax_fraction monthly_tax monthly_income = 106 / 265 := by
sorry

end NUMINAMATH_CALUDE_joe_tax_fraction_l1312_131211


namespace NUMINAMATH_CALUDE_vertex_C_coordinates_l1312_131250

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the median CM and altitude BH
def median_CM (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => 2 * p.1 - p.2 - 5 = 0

def altitude_BH (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => p.1 - 2 * p.2 - 5 = 0

-- Theorem statement
theorem vertex_C_coordinates (t : Triangle) :
  t.A = (5, 1) →
  median_CM t t.C →
  altitude_BH t t.B →
  (t.C.1 - t.A.1) * (t.B.1 - t.C.1) + (t.C.2 - t.A.2) * (t.B.2 - t.C.2) = 0 →
  t.C = (4, 3) := by
  sorry

end NUMINAMATH_CALUDE_vertex_C_coordinates_l1312_131250


namespace NUMINAMATH_CALUDE_crayons_theorem_l1312_131205

/-- The number of crayons in the drawer at the end of Thursday. -/
def crayons_at_end_of_thursday (initial : ℕ) (mary_adds : ℕ) (john_removes : ℕ) (lisa_adds : ℕ) (jeremy_adds : ℕ) (sarah_removes : ℕ) : ℕ :=
  initial + mary_adds - john_removes + lisa_adds + jeremy_adds - sarah_removes

/-- Theorem stating that the number of crayons at the end of Thursday is 13. -/
theorem crayons_theorem : crayons_at_end_of_thursday 7 3 5 4 6 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_crayons_theorem_l1312_131205


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1312_131206

theorem polynomial_divisibility (n : ℤ) : 
  (∀ x : ℝ, (3 * x^2 + 5 * x + n) = (x - 2) * (3 * x + 11)) ↔ n = -22 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1312_131206


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1312_131296

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) →
  a ∈ Set.Ioi 3 ∪ Set.Iic 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1312_131296


namespace NUMINAMATH_CALUDE_complex_equality_l1312_131264

theorem complex_equality (z₁ z₂ : ℂ) (h : Complex.abs (z₁ + 2 * z₂) = Complex.abs (2 * z₁ + z₂)) :
  ∀ a : ℝ, Complex.abs (z₁ + a * z₂) = Complex.abs (a * z₁ + z₂) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l1312_131264


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l1312_131207

theorem stratified_sampling_medium_stores 
  (total_stores : ℕ) 
  (medium_stores : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_stores = 300) 
  (h2 : medium_stores = 75) 
  (h3 : sample_size = 20) :
  ⌊(medium_stores : ℚ) / total_stores * sample_size⌋ = 5 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l1312_131207


namespace NUMINAMATH_CALUDE_lines_symmetric_about_y_axis_l1312_131234

/-- Two lines are symmetric about the y-axis if and only if their coefficients satisfy certain conditions -/
theorem lines_symmetric_about_y_axis 
  (m n p : ℝ) : 
  (∀ x y : ℝ, x + m * y + 5 = 0 ↔ x + n * y + p = 0) ∧ 
  (∀ x y : ℝ, x + m * y + 5 = 0 ↔ -x + n * y + p = 0) ↔ 
  m = -n ∧ p = -5 := by sorry

end NUMINAMATH_CALUDE_lines_symmetric_about_y_axis_l1312_131234


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1312_131245

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ ∀ x > 0, ¬ P x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x > 0, x^2 - x ≤ 0) ↔ (∀ x > 0, x^2 - x > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1312_131245


namespace NUMINAMATH_CALUDE_gcd_with_35_is_7_l1312_131262

theorem gcd_with_35_is_7 : 
  ∃ (s : Finset Nat), s = {n : Nat | 70 < n ∧ n < 90 ∧ Nat.gcd 35 n = 7} ∧ s = {77, 84} := by
  sorry

end NUMINAMATH_CALUDE_gcd_with_35_is_7_l1312_131262


namespace NUMINAMATH_CALUDE_math_team_combinations_l1312_131208

theorem math_team_combinations (girls : ℕ) (boys : ℕ) 
  (h1 : girls = 4) (h2 : boys = 6) : 
  (Nat.choose girls 3) * (Nat.choose boys 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l1312_131208


namespace NUMINAMATH_CALUDE_function_properties_l1312_131227

def StrictlyDecreasing (f : ℝ → ℝ) :=
  ∀ x y, x < y → f x > f y

def StrictlyConvex (f : ℝ → ℝ) :=
  ∀ x y t, 0 < t → t < 1 → f (t * x + (1 - t) * y) < t * f x + (1 - t) * f y

theorem function_properties (f : ℝ → ℝ) (h1 : StrictlyDecreasing f) (h2 : StrictlyConvex f) :
  ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → x₂ < 1 →
    (x₂ * f x₁ > x₁ * f x₂) ∧
    ((f x₁ + f x₂) / 2 < f ((x₁ + x₂) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1312_131227
