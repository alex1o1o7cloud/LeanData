import Mathlib

namespace NUMINAMATH_CALUDE_profit_360_implies_price_increase_4_price_13_implies_profit_350_l382_38251

/-- Represents the daily profit function for a company selling goods -/
def profit_function (x : ℕ) : ℤ := 10 * (x + 2) * (10 - x)

/-- Theorem stating that when the daily profit is 360 yuan, the selling price has increased by 4 yuan -/
theorem profit_360_implies_price_increase_4 :
  ∃ (x : ℕ), 0 ≤ x ∧ x ≤ 10 ∧ profit_function x = 360 → x = 4 := by
  sorry

/-- Theorem stating that when the selling price increases by 3 yuan (to 13 yuan), the profit is 350 yuan -/
theorem price_13_implies_profit_350 :
  profit_function 3 = 350 := by
  sorry

end NUMINAMATH_CALUDE_profit_360_implies_price_increase_4_price_13_implies_profit_350_l382_38251


namespace NUMINAMATH_CALUDE_circle_condition_m_set_l382_38278

/-- A set in R² represents a circle if it can be expressed as 
    {(x, y) | (x - h)² + (y - k)² = r²} for some h, k, and r > 0 -/
def IsCircle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ h k r, r > 0 ∧ S = {p : ℝ × ℝ | (p.1 - h)^2 + (p.2 - k)^2 = r^2}

/-- The set of points (x, y) satisfying the given equation -/
def S (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*m*p.1 - 2*m*p.2 + 2*m^2 + m - 1 = 0}

theorem circle_condition (m : ℝ) : IsCircle (S m) → m < 1 := by
  sorry

theorem m_set : {m : ℝ | IsCircle (S m)} = {m : ℝ | m < 1} := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_m_set_l382_38278


namespace NUMINAMATH_CALUDE_complex_sum_to_polar_l382_38242

theorem complex_sum_to_polar : ∃ (r θ : ℝ), 
  12 * Complex.exp (3 * π * Complex.I / 13) + 12 * Complex.exp (7 * π * Complex.I / 26) = 
  r * Complex.exp (θ * Complex.I) ∧ 
  r = 12 * Real.sqrt (2 + Real.sqrt 2) ∧ 
  θ = 3.25 * π / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_to_polar_l382_38242


namespace NUMINAMATH_CALUDE_percentage_difference_l382_38241

theorem percentage_difference : (0.6 * 50) - (0.45 * 30) = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l382_38241


namespace NUMINAMATH_CALUDE_actual_distance_calculation_l382_38233

-- Define the map scale
def map_scale : ℚ := 200

-- Define the measured distance on the map
def map_distance : ℚ := 9/2

-- Theorem to prove
theorem actual_distance_calculation :
  map_scale * map_distance / 100 = 9 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_calculation_l382_38233


namespace NUMINAMATH_CALUDE_snack_pack_suckers_l382_38249

/-- The number of suckers needed for snack packs --/
def suckers_needed (pretzels : ℕ) (goldfish_multiplier : ℕ) (kids : ℕ) (items_per_baggie : ℕ) : ℕ :=
  kids * items_per_baggie - (pretzels + goldfish_multiplier * pretzels)

theorem snack_pack_suckers :
  suckers_needed 64 4 16 22 = 32 := by
  sorry

end NUMINAMATH_CALUDE_snack_pack_suckers_l382_38249


namespace NUMINAMATH_CALUDE_second_race_lead_l382_38275

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ

/-- Represents a race between two runners -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner

theorem second_race_lead (h d : ℝ) (first_race second_race : Race) :
  h > 0 ∧ d > 0 ∧
  first_race.distance = h ∧
  second_race.distance = h ∧
  first_race.runner_a = second_race.runner_a ∧
  first_race.runner_b = second_race.runner_b ∧
  first_race.runner_a.speed * h = first_race.runner_b.speed * (h - 2 * d) →
  let finish_time := (h + 2 * d) / first_race.runner_a.speed
  finish_time * first_race.runner_a.speed - finish_time * first_race.runner_b.speed = 4 * d^2 / h :=
by sorry

end NUMINAMATH_CALUDE_second_race_lead_l382_38275


namespace NUMINAMATH_CALUDE_boxes_in_case_l382_38237

/-- Proves the number of boxes in a case given the total boxes, eggs per box, and total eggs -/
theorem boxes_in_case 
  (total_boxes : ℕ) 
  (eggs_per_box : ℕ) 
  (total_eggs : ℕ) 
  (h1 : total_boxes = 5)
  (h2 : eggs_per_box = 3)
  (h3 : total_eggs = 15)
  (h4 : total_eggs = total_boxes * eggs_per_box) :
  total_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_in_case_l382_38237


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_and_necessary_not_sufficient_l382_38283

theorem sufficient_not_necessary_and_necessary_not_sufficient :
  (∀ a : ℝ, a > 1 → 1 / a < 1) ∧
  (∃ a : ℝ, 1 / a < 1 ∧ a ≤ 1) ∧
  (∀ a b : ℝ, a * b ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ a * b = 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_and_necessary_not_sufficient_l382_38283


namespace NUMINAMATH_CALUDE_remove_horizontal_eliminates_triangles_l382_38298

/-- Represents a triangular grid constructed with toothpicks -/
structure TriangularGrid where
  toothpicks : ℕ
  rows : ℕ
  columns : ℕ
  triangles : ℕ

/-- The specific triangular grid in the problem -/
def problemGrid : TriangularGrid :=
  { toothpicks := 36
  , rows := 3
  , columns := 5
  , triangles := 35 }

/-- The number of horizontal toothpicks in the grid -/
def horizontalToothpicks (grid : TriangularGrid) : ℕ := grid.rows * grid.columns

/-- Theorem stating that removing all horizontal toothpicks eliminates all triangles -/
theorem remove_horizontal_eliminates_triangles (grid : TriangularGrid) :
  horizontalToothpicks grid = 15 ∧
  horizontalToothpicks grid ≤ grid.toothpicks ∧
  grid.triangles > 35 →
  (∀ n : ℕ, n < 15 → ∃ t : ℕ, t > 0) ∧
  (∀ t : ℕ, t = 0) :=
sorry

#check remove_horizontal_eliminates_triangles problemGrid

end NUMINAMATH_CALUDE_remove_horizontal_eliminates_triangles_l382_38298


namespace NUMINAMATH_CALUDE_luncheon_absence_l382_38299

/-- The number of people who didn't show up to a luncheon --/
def people_absent (invited : ℕ) (table_capacity : ℕ) (tables_needed : ℕ) : ℕ :=
  invited - (table_capacity * tables_needed)

/-- Proof that 50 people didn't show up to the luncheon --/
theorem luncheon_absence : people_absent 68 3 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_absence_l382_38299


namespace NUMINAMATH_CALUDE_scientific_notation_of_nanometers_l382_38279

theorem scientific_notation_of_nanometers : 
  ∃ (a : ℝ) (n : ℤ), 0.000000007 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7 ∧ n = -9 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_nanometers_l382_38279


namespace NUMINAMATH_CALUDE_no_large_lattice_regular_ngon_l382_38204

/-- A lattice point in the coordinate plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A regular n-gon in the coordinate plane -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → LatticePoint
  is_regular : ∀ (i j : Fin n), 
    (vertices i).x^2 + (vertices i).y^2 = (vertices j).x^2 + (vertices j).y^2

/-- There does not exist a regular n-gon with n ≥ 7 whose vertices are all lattice points -/
theorem no_large_lattice_regular_ngon :
  ∀ (n : ℕ), n ≥ 7 → ¬∃ (ngon : RegularNGon n), True :=
sorry

end NUMINAMATH_CALUDE_no_large_lattice_regular_ngon_l382_38204


namespace NUMINAMATH_CALUDE_outfits_count_l382_38225

/-- The number of shirts available. -/
def num_shirts : ℕ := 6

/-- The number of ties available. -/
def num_ties : ℕ := 5

/-- The number of pants available. -/
def num_pants : ℕ := 4

/-- The total number of possible outfits. -/
def total_outfits : ℕ := num_shirts * num_ties * num_pants

/-- Theorem stating that the total number of outfits is 120. -/
theorem outfits_count : total_outfits = 120 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l382_38225


namespace NUMINAMATH_CALUDE_prices_and_min_cost_l382_38244

/-- Represents the price of a thermometer in yuan -/
def thermometer_price : ℝ := sorry

/-- Represents the price of a barrel of disinfectant in yuan -/
def disinfectant_price : ℝ := sorry

/-- The total cost of 4 thermometers and 2 barrels of disinfectant is 400 yuan -/
axiom equation1 : 4 * thermometer_price + 2 * disinfectant_price = 400

/-- The total cost of 2 thermometers and 4 barrels of disinfectant is 320 yuan -/
axiom equation2 : 2 * thermometer_price + 4 * disinfectant_price = 320

/-- The total number of items to be purchased -/
def total_items : ℕ := 80

/-- The constraint that the number of thermometers is no less than 1/4 of the number of disinfectant -/
def constraint (m : ℕ) : Prop := m ≥ (total_items - m) / 4

/-- The cost function for m thermometers and (80 - m) barrels of disinfectant -/
def cost (m : ℕ) : ℝ := thermometer_price * m + disinfectant_price * (total_items - m)

/-- The theorem stating the unit prices and minimum cost -/
theorem prices_and_min_cost :
  thermometer_price = 80 ∧
  disinfectant_price = 40 ∧
  ∃ m : ℕ, constraint m ∧ cost m = 3840 ∧ ∀ n : ℕ, constraint n → cost m ≤ cost n :=
sorry

end NUMINAMATH_CALUDE_prices_and_min_cost_l382_38244


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l382_38211

/-- Given a hyperbola and a point P on it forming a unit-area parallelogram with the origin and points on the asymptotes, prove the equations of the asymptotes. -/
theorem hyperbola_asymptotes (a : ℝ) (h_a : a > 0) (P : ℝ × ℝ) :
  (P.1^2 / a^2 - P.2^2 = 1) →  -- P is on the hyperbola
  (∃ (A B : ℝ × ℝ), 
    (A.2 = (A.1 / a)) ∧  -- A is on one asymptote
    (B.2 = -(B.1 / a)) ∧  -- B is on the other asymptote
    (P.2 - A.2 = (P.1 - A.1) / a) ∧  -- PA is parallel to one asymptote
    (P.2 - B.2 = -(P.1 - B.1) / a) ∧  -- PB is parallel to the other asymptote
    (abs ((A.1 - 0) * (P.2 - 0) - (A.2 - 0) * (P.1 - 0)) = 1)  -- Area of OBPA is 1
  ) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l382_38211


namespace NUMINAMATH_CALUDE_no_valid_p_exists_l382_38215

theorem no_valid_p_exists (p M : ℝ) (hp : 0 < p) (hM : 0 < M) (hp2 : p < 2) : 
  ¬∃ p, M * (1 + p / 100) * (1 - 50 * p / 100) > M :=
by
  sorry

end NUMINAMATH_CALUDE_no_valid_p_exists_l382_38215


namespace NUMINAMATH_CALUDE_equation_roots_l382_38209

theorem equation_roots : 
  let f (x : ℝ) := 18 / (x^2 - 9) - 3 / (x - 3) - 2
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -4.5 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_l382_38209


namespace NUMINAMATH_CALUDE_min_value_a_l382_38254

theorem min_value_a (a : ℝ) : 
  (¬ ∃ x₀ : ℝ, -1 < x₀ ∧ x₀ ≤ 2 ∧ x₀ - a > 0) → 
  (∀ b : ℝ, (¬ ∃ x₀ : ℝ, -1 < x₀ ∧ x₀ ≤ 2 ∧ x₀ - b > 0) → a ≤ b) → 
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l382_38254


namespace NUMINAMATH_CALUDE_not_always_swap_cities_l382_38222

/-- A graph representing cities and their railroad connections. -/
structure CityGraph where
  V : Type
  E : V → V → Prop

/-- A bijective function representing a renaming of cities. -/
def IsRenaming (G : CityGraph) (f : G.V → G.V) : Prop :=
  Function.Bijective f

/-- A renaming that preserves the graph structure (i.e., a graph isomorphism). -/
def IsValidRenaming (G : CityGraph) (f : G.V → G.V) : Prop :=
  IsRenaming G f ∧ ∀ u v : G.V, G.E u v ↔ G.E (f u) (f v)

/-- For any two cities, there exists a valid renaming that maps one to the other. -/
axiom any_city_can_be_renamed (G : CityGraph) :
  ∀ u v : G.V, ∃ f : G.V → G.V, IsValidRenaming G f ∧ f u = v

/-- The theorem to be proved. -/
theorem not_always_swap_cities (G : CityGraph) :
  ¬(∀ x y : G.V, ∃ f : G.V → G.V, IsValidRenaming G f ∧ f x = y ∧ f y = x) :=
sorry

end NUMINAMATH_CALUDE_not_always_swap_cities_l382_38222


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_pi_sixth_l382_38231

theorem sin_2alpha_minus_pi_sixth (α : ℝ) 
  (h : Real.cos (α + π / 6) = Real.sqrt 3 / 3) : 
  Real.sin (2 * α - π / 6) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_pi_sixth_l382_38231


namespace NUMINAMATH_CALUDE_parallelogram_height_l382_38282

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 200 → base = 10 → area = base * height → height = 20 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l382_38282


namespace NUMINAMATH_CALUDE_base_conversion_equality_l382_38247

def base_8_to_10 (n : ℕ) : ℕ := sorry

def base_10_to_7 (n : ℕ) : ℕ := sorry

theorem base_conversion_equality :
  base_10_to_7 (base_8_to_10 5314) = 11026 := by sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l382_38247


namespace NUMINAMATH_CALUDE_median_and_altitude_length_l382_38276

/-- An isosceles triangle DEF with DE = DF = 10 and EF = 12 -/
structure IsoscelesTriangle where
  /-- Length of side DE -/
  de : ℝ
  /-- Length of side DF -/
  df : ℝ
  /-- Length of side EF -/
  ef : ℝ
  /-- DE equals DF -/
  de_eq_df : de = df
  /-- DE equals 10 -/
  de_eq_ten : de = 10
  /-- EF equals 12 -/
  ef_eq_twelve : ef = 12

/-- The median DM from vertex D to side EF in the isosceles triangle -/
def median (t : IsoscelesTriangle) : ℝ := sorry

/-- The altitude DH from vertex D to side EF in the isosceles triangle -/
def altitude (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem: Both the median and altitude have length 8 -/
theorem median_and_altitude_length (t : IsoscelesTriangle) : 
  median t = 8 ∧ altitude t = 8 := by sorry

end NUMINAMATH_CALUDE_median_and_altitude_length_l382_38276


namespace NUMINAMATH_CALUDE_carriage_hire_cost_l382_38267

/-- The cost of hiring a carriage for a journey, given:
  * The distance to the destination
  * The speed of the horse
  * The hourly rate for the carriage
  * A flat fee for the service
-/
theorem carriage_hire_cost 
  (distance : ℝ) 
  (speed : ℝ) 
  (hourly_rate : ℝ) 
  (flat_fee : ℝ) 
  (h1 : distance = 20)
  (h2 : speed = 10)
  (h3 : hourly_rate = 30)
  (h4 : flat_fee = 20)
  : (distance / speed) * hourly_rate + flat_fee = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_carriage_hire_cost_l382_38267


namespace NUMINAMATH_CALUDE_jia_test_probability_l382_38246

/-- The probability of passing a test with given parameters -/
def test_pass_probability (total_questions n_correct_known n_selected n_to_pass : ℕ) : ℚ :=
  let favorable_outcomes := Nat.choose n_correct_known 2 * Nat.choose (total_questions - n_correct_known) 1 +
                            Nat.choose n_correct_known 3
  let total_outcomes := Nat.choose total_questions n_selected
  favorable_outcomes / total_outcomes

/-- Theorem stating the probability of Jia passing the test -/
theorem jia_test_probability :
  test_pass_probability 10 5 3 2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_jia_test_probability_l382_38246


namespace NUMINAMATH_CALUDE_sin_value_at_pi_over_four_l382_38277

theorem sin_value_at_pi_over_four 
  (φ : Real) 
  (ω : Real)
  (h1 : (- 4 : Real) / 5 = Real.cos φ ∧ (3 : Real) / 5 = Real.sin φ)
  (h2 : (2 * Real.pi) / ω = Real.pi)
  (h3 : ω > 0) :
  Real.sin ((2 : Real) * Real.pi / 4 + φ) = - (4 : Real) / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_value_at_pi_over_four_l382_38277


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_m_half_l382_38285

/-- Given two non-zero parallel vectors (m^2-1, m+1) and (1, -2), prove that m = 1/2 -/
theorem parallel_vectors_imply_m_half (m : ℝ) : 
  (m^2 - 1 ≠ 0 ∨ m + 1 ≠ 0) →  -- Vector a is non-zero
  (∃ (k : ℝ), k ≠ 0 ∧ k * (1 : ℝ) = m^2 - 1 ∧ k * (-2 : ℝ) = m + 1) →  -- Vectors are parallel
  m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_m_half_l382_38285


namespace NUMINAMATH_CALUDE_A_power_98_l382_38223

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]

theorem A_power_98 : A ^ 98 = !![0, 0, 0; 0, -1, 0; 0, 0, -1] := by sorry

end NUMINAMATH_CALUDE_A_power_98_l382_38223


namespace NUMINAMATH_CALUDE_organization_size_after_five_years_l382_38210

def organization_growth (initial_members : ℕ) (leaders : ℕ) (recruitment : ℕ) (years : ℕ) : ℕ :=
  let rec growth (k : ℕ) (members : ℕ) : ℕ :=
    if k = 0 then
      members
    else
      growth (k - 1) (4 * members - 18)
  growth years initial_members

theorem organization_size_after_five_years :
  organization_growth 20 6 3 5 = 14382 :=
by sorry

end NUMINAMATH_CALUDE_organization_size_after_five_years_l382_38210


namespace NUMINAMATH_CALUDE_division_problem_l382_38253

theorem division_problem (n : ℤ) : 
  (n / 20 = 15) ∧ (n % 20 = 6) → n = 306 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l382_38253


namespace NUMINAMATH_CALUDE_intersection_point_ratio_l382_38262

/-- Parabola type representing y² = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Line type with 60° inclination passing through (1, 0) -/
structure Line where
  x : ℝ
  y : ℝ
  eq : y = Real.sqrt 3 * (x - 1)

/-- Intersection point of the parabola and the line -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  on_parabola : Parabola
  on_line : Line

/-- Theorem stating the ratio of distances from intersection points to focus -/
theorem intersection_point_ratio 
  (A B : IntersectionPoint) 
  (h1 : A.x + 1 > B.x + 1) : 
  (A.x + 1) / (B.x + 1) = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_point_ratio_l382_38262


namespace NUMINAMATH_CALUDE_f_properties_l382_38239

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin x ^ 2 + Real.sin x * Real.cos x

theorem f_properties :
  (f (25 * Real.pi / 6) = 0) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1 - Real.sqrt 3 / 2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1 - Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l382_38239


namespace NUMINAMATH_CALUDE_mars_ticket_cost_after_30_years_l382_38245

/-- The cost of a ticket to Mars after a given number of decades, 
    given an initial cost and a halving rate every decade. -/
def mars_ticket_cost (initial_cost : ℚ) (decades : ℕ) : ℚ :=
  initial_cost / (2 ^ decades)

/-- Theorem stating that the cost of a ticket to Mars after 3 decades
    is $125,000, given an initial cost of $1,000,000 and halving every decade. -/
theorem mars_ticket_cost_after_30_years 
  (initial_cost : ℚ) (h_initial : initial_cost = 1000000) :
  mars_ticket_cost initial_cost 3 = 125000 := by
  sorry

#eval mars_ticket_cost 1000000 3

end NUMINAMATH_CALUDE_mars_ticket_cost_after_30_years_l382_38245


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l382_38265

theorem quadratic_equation_k_value :
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := 3
  let k : ℝ := 16/3
  (4 * b^2 - k * a * c = 0) ∧
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 4 * b * x + c = 0 ∧ a * y^2 + 4 * b * y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l382_38265


namespace NUMINAMATH_CALUDE_complex_fraction_equality_complex_fraction_equality_proof_l382_38270

theorem complex_fraction_equality : Complex → Prop :=
  fun i => (3 : ℂ) / (1 - i)^2 = (3 / 2 : ℂ) * i

-- The proof is omitted
theorem complex_fraction_equality_proof : complex_fraction_equality Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_complex_fraction_equality_proof_l382_38270


namespace NUMINAMATH_CALUDE_dodecagon_arrangement_impossible_l382_38291

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def arrangement := Fin 12 → Fin 12

def valid_arrangement (a : arrangement) : Prop :=
  ∀ i : Fin 12, ∃ j : Fin 12, a j = i

def adjacent_sum_prime (a : arrangement) : Prop :=
  ∀ i : Fin 12, is_prime ((a i).val + 1 + (a ((i + 1) % 12)).val + 1)

def skip_two_sum_prime (a : arrangement) : Prop :=
  ∀ i : Fin 12, is_prime ((a i).val + 1 + (a ((i + 3) % 12)).val + 1)

theorem dodecagon_arrangement_impossible :
  ¬∃ a : arrangement, valid_arrangement a ∧ adjacent_sum_prime a ∧ skip_two_sum_prime a :=
sorry

end NUMINAMATH_CALUDE_dodecagon_arrangement_impossible_l382_38291


namespace NUMINAMATH_CALUDE_unique_valid_number_l382_38240

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  n / 1000 = 764 ∧
  n % 8 = 0 ∧ n % 9 = 0 ∧ n % 11 = 0

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 764280 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l382_38240


namespace NUMINAMATH_CALUDE_athletes_same_first_digit_know_each_other_l382_38271

/-- Represents an athlete with an assigned number -/
structure Athlete where
  id : Nat
  number : Nat

/-- Represents the relation of two athletes knowing each other -/
def knows (a b : Athlete) : Prop := sorry

/-- Returns the first digit of a natural number -/
def firstDigit (n : Nat) : Nat := sorry

/-- Theorem: Given 19100 athletes, where among any 12 athletes at least 2 know each other,
    there exist 2 athletes who know each other and whose assigned numbers start with the same digit -/
theorem athletes_same_first_digit_know_each_other 
  (athletes : Finset Athlete) 
  (h1 : athletes.card = 19100) 
  (h2 : ∀ s : Finset Athlete, s ⊆ athletes → s.card = 12 → 
        ∃ a b : Athlete, a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ knows a b) :
  ∃ a b : Athlete, a ∈ athletes ∧ b ∈ athletes ∧ a ≠ b ∧ 
    knows a b ∧ firstDigit a.number = firstDigit b.number := by
  sorry

end NUMINAMATH_CALUDE_athletes_same_first_digit_know_each_other_l382_38271


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l382_38227

theorem discount_percentage_calculation (MP : ℝ) (h1 : MP > 0) : 
  let CP := 0.36 * MP
  let gain_percent := 122.22222222222223
  let SP := CP * (1 + gain_percent / 100)
  (MP - SP) / MP * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_discount_percentage_calculation_l382_38227


namespace NUMINAMATH_CALUDE_student_weight_loss_l382_38261

theorem student_weight_loss (student_weight sister_weight : ℝ) 
  (h1 : student_weight = 90)
  (h2 : student_weight + sister_weight = 132) : 
  ∃ (weight_loss : ℝ), 
    weight_loss = 6 ∧ 
    student_weight - weight_loss = 2 * sister_weight :=
by sorry

end NUMINAMATH_CALUDE_student_weight_loss_l382_38261


namespace NUMINAMATH_CALUDE_laura_charge_account_theorem_l382_38202

/-- Calculates the total amount owed after one year with simple interest -/
def total_amount_owed (principal : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * interest_rate * time

/-- Proves that the total amount owed after one year is $37.45 -/
theorem laura_charge_account_theorem :
  let principal : ℝ := 35
  let interest_rate : ℝ := 0.07
  let time : ℝ := 1
  total_amount_owed principal interest_rate time = 37.45 := by
sorry

end NUMINAMATH_CALUDE_laura_charge_account_theorem_l382_38202


namespace NUMINAMATH_CALUDE_hike_attendance_l382_38260

/-- The number of people who went on the hike --/
def total_hikers (num_cars num_taxis num_vans : ℕ) 
                 (car_capacity taxi_capacity van_capacity : ℕ) : ℕ :=
  num_cars * car_capacity + num_taxis * taxi_capacity + num_vans * van_capacity

/-- Theorem stating that 58 people went on the hike --/
theorem hike_attendance : 
  total_hikers 3 6 2 4 6 5 = 58 := by
  sorry

end NUMINAMATH_CALUDE_hike_attendance_l382_38260


namespace NUMINAMATH_CALUDE_frog_jump_probability_l382_38248

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the rectangle boundary -/
def Rectangle := {p : Point | p.x = 0 ∨ p.x = 5 ∨ p.y = 0 ∨ p.y = 5}

/-- Represents a vertical side of the rectangle -/
def VerticalSide := {p : Point | p.x = 0 ∨ p.x = 5}

/-- The probability of ending on a vertical side starting from a given point -/
def probabilityVerticalSide (p : Point) : ℚ := sorry

/-- The frog's starting point -/
def startPoint : Point := ⟨2, 3⟩

/-- Theorem stating the probability of ending on a vertical side -/
theorem frog_jump_probability : probabilityVerticalSide startPoint = 2/3 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l382_38248


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l382_38207

theorem complex_magnitude_theorem (x y : ℝ) (z : ℂ) :
  z = x + y * Complex.I →
  (2 * x) / (1 - Complex.I) = 1 + y * Complex.I →
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l382_38207


namespace NUMINAMATH_CALUDE_sparrow_population_decline_l382_38226

/-- Proves that the smallest integer t satisfying (0.6^t ≤ 0.05) is 6 -/
theorem sparrow_population_decline (t : ℕ) : 
  (∀ k : ℕ, k < t → (0.6 : ℝ)^k > 0.05) ∧ (0.6 : ℝ)^t ≤ 0.05 → t = 6 :=
by sorry

end NUMINAMATH_CALUDE_sparrow_population_decline_l382_38226


namespace NUMINAMATH_CALUDE_count_divisible_integers_l382_38266

theorem count_divisible_integers : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ (2310 : ℤ) ∣ (m^2 - 2)) ∧ 
    (∀ m : ℕ, m > 0 ∧ (2310 : ℤ) ∣ (m^2 - 2) → m ∈ S) ∧
    S.card = 3 :=
by sorry

end NUMINAMATH_CALUDE_count_divisible_integers_l382_38266


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l382_38205

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (4 - a/2) * x + 2 else a^x

/-- The theorem stating the range of a for which f is increasing on ℝ -/
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (4 ≤ a ∧ a < 8) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l382_38205


namespace NUMINAMATH_CALUDE_survey_respondents_count_l382_38259

theorem survey_respondents_count :
  let brand_x_count : ℕ := 360
  let brand_x_to_y_ratio : ℚ := 9 / 1
  let total_respondents : ℕ := brand_x_count + (brand_x_count / brand_x_to_y_ratio.num * brand_x_to_y_ratio.den).toNat
  total_respondents = 400 :=
by sorry

end NUMINAMATH_CALUDE_survey_respondents_count_l382_38259


namespace NUMINAMATH_CALUDE_arctan_sum_three_seven_l382_38274

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_seven_l382_38274


namespace NUMINAMATH_CALUDE_fraction_difference_l382_38224

theorem fraction_difference (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : m^2 - n^2 = m*n) : 
  n/m - m/n = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l382_38224


namespace NUMINAMATH_CALUDE_hypotenuse_sum_of_two_triangles_l382_38263

theorem hypotenuse_sum_of_two_triangles : 
  let triangle1_leg1 : ℝ := 120
  let triangle1_leg2 : ℝ := 160
  let triangle2_leg1 : ℝ := 30
  let triangle2_leg2 : ℝ := 40
  let hypotenuse1 := Real.sqrt (triangle1_leg1^2 + triangle1_leg2^2)
  let hypotenuse2 := Real.sqrt (triangle2_leg1^2 + triangle2_leg2^2)
  hypotenuse1 + hypotenuse2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_sum_of_two_triangles_l382_38263


namespace NUMINAMATH_CALUDE_circus_tickets_cost_l382_38229

def adult_ticket_price : ℚ := 44
def child_ticket_price : ℚ := 28
def num_adults : ℕ := 2
def num_children : ℕ := 5
def discount_rate : ℚ := 0.1
def discount_threshold : ℕ := 6

def total_cost : ℚ :=
  let total_tickets := num_adults + num_children
  let subtotal := num_adults * adult_ticket_price + num_children * child_ticket_price
  if total_tickets > discount_threshold then
    subtotal * (1 - discount_rate)
  else
    subtotal

theorem circus_tickets_cost :
  total_cost = 205.2 := by sorry

end NUMINAMATH_CALUDE_circus_tickets_cost_l382_38229


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l382_38286

theorem geometric_sequence_seventh_term 
  (a : ℝ) (a₃ : ℝ) (n : ℕ) (h₁ : a = 3) (h₂ : a₃ = 3/64) (h₃ : n = 7) :
  a * (a₃ / a) ^ ((n - 1) / 2) = 3/262144 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l382_38286


namespace NUMINAMATH_CALUDE_bakery_roll_combinations_l382_38213

theorem bakery_roll_combinations :
  let total_rolls : ℕ := 9
  let fixed_rolls : ℕ := 6
  let remaining_rolls : ℕ := total_rolls - fixed_rolls
  let num_types : ℕ := 4
  (Nat.choose (remaining_rolls + num_types - 1) (num_types - 1)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_bakery_roll_combinations_l382_38213


namespace NUMINAMATH_CALUDE_min_value_quadratic_l382_38243

theorem min_value_quadratic (x : ℝ) : 
  ∃ (min_y : ℝ), ∀ (y : ℝ), y = 2 * x^2 - 8 * x + 10 → y ≥ min_y ∧ min_y = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l382_38243


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l382_38287

theorem solution_set_quadratic_inequality :
  let S := {x : ℝ | 2 * x^2 - x - 3 ≥ 0}
  S = {x : ℝ | x ≤ -1 ∨ x ≥ 3/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l382_38287


namespace NUMINAMATH_CALUDE_sphere_containment_l382_38288

/-- A point in 3-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A sphere in 3-dimensional space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Predicate to check if a point is inside or on a sphere -/
def Point3D.inSphere (p : Point3D) (s : Sphere) : Prop :=
  (p.x - s.center.x)^2 + (p.y - s.center.y)^2 + (p.z - s.center.z)^2 ≤ s.radius^2

/-- The main theorem -/
theorem sphere_containment (n : ℕ) (points : Fin n → Point3D) 
    (h : n ≥ 5)
    (h_four : ∀ (a b c d : Fin n), ∃ (s : Sphere), 
      s.radius = 1 ∧ 
      (points a).inSphere s ∧ 
      (points b).inSphere s ∧ 
      (points c).inSphere s ∧ 
      (points d).inSphere s) :
    ∃ (s : Sphere), s.radius = 1 ∧ ∀ (i : Fin n), (points i).inSphere s := by
  sorry

end NUMINAMATH_CALUDE_sphere_containment_l382_38288


namespace NUMINAMATH_CALUDE_age_ratio_in_one_year_l382_38235

/-- Represents the current ages of Jack and Alex -/
structure Ages where
  jack : ℕ
  alex : ℕ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  (ages.jack - 3 = 2 * (ages.alex - 3)) ∧ 
  (ages.jack - 5 = 3 * (ages.alex - 5))

/-- The future ratio of their ages will be 3:2 -/
def future_ratio (ages : Ages) (years : ℕ) : Prop :=
  3 * (ages.alex + years) = 2 * (ages.jack + years)

/-- The theorem to be proved -/
theorem age_ratio_in_one_year (ages : Ages) :
  age_conditions ages → ∃ (y : ℕ), y = 1 ∧ future_ratio ages y :=
sorry

end NUMINAMATH_CALUDE_age_ratio_in_one_year_l382_38235


namespace NUMINAMATH_CALUDE_inequality_relationship_l382_38293

theorem inequality_relationship (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_relationship_l382_38293


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l382_38294

theorem polynomial_product_expansion (x : ℝ) :
  (1 + x + x^3) * (1 - x^4) = 1 + x + x^3 - x^4 - x^5 - x^7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l382_38294


namespace NUMINAMATH_CALUDE_fraction_simplification_l382_38219

theorem fraction_simplification 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (h_sum : a^2 + b^2 + c^2 ≠ 0) :
  (a^2*b^2 + 2*a^2*b*c + a^2*c^2 - b^4) / (a^4 - b^2*c^2 + 2*a*b*c^2 + c^4) = 
  ((a*b+a*c+b^2)*(a*b+a*c-b^2)) / ((a^2 + b^2 - c^2)*(a^2 - b^2 + c^2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l382_38219


namespace NUMINAMATH_CALUDE_x_value_l382_38290

theorem x_value (x : ℝ) : x = 80 * (1 + 13 / 100) → x = 90.4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l382_38290


namespace NUMINAMATH_CALUDE_ball_cost_price_l382_38256

theorem ball_cost_price (cost : ℕ → ℝ) (h1 : cost 11 - 720 = cost 5) : cost 1 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ball_cost_price_l382_38256


namespace NUMINAMATH_CALUDE_equilateral_triangle_max_area_l382_38200

/-- The area of a triangle is maximum when it is equilateral, given a fixed perimeter -/
theorem equilateral_triangle_max_area 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let p := (a + b + c) / 2
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  ∀ a' b' c' : ℝ, 
    a' > 0 → b' > 0 → c' > 0 →
    a' + b' + c' = a + b + c →
    let p' := (a' + b' + c') / 2
    let S' := Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c'))
    S' ≤ S ∧ (S' = S → a' = b' ∧ b' = c') :=
by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_max_area_l382_38200


namespace NUMINAMATH_CALUDE_arccos_difference_equals_negative_pi_over_six_l382_38272

theorem arccos_difference_equals_negative_pi_over_six :
  Real.arccos ((Real.sqrt 6 + 1) / (2 * Real.sqrt 3)) - Real.arccos (Real.sqrt (2/3)) = -π/6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_difference_equals_negative_pi_over_six_l382_38272


namespace NUMINAMATH_CALUDE_line_inclination_angle_l382_38284

/-- The inclination angle of a line is the angle between the positive x-axis and the line, measured counterclockwise. -/
def inclination_angle (a b c : ℝ) : ℝ := sorry

/-- A line is defined by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem line_inclination_angle :
  let l : Line := { a := 1, b := 1, c := 1 }
  inclination_angle l.a l.b l.c = 135 * π / 180 := by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l382_38284


namespace NUMINAMATH_CALUDE_jerrys_breakfast_theorem_l382_38273

/-- Calculates the total calories in Jerry's breakfast -/
def jerrys_breakfast_calories : ℕ :=
  let pancake_calories : ℕ := 120
  let bacon_calories : ℕ := 100
  let cereal_calories : ℕ := 200
  let num_pancakes : ℕ := 6
  let num_bacon_strips : ℕ := 2
  let num_cereal_bowls : ℕ := 1
  (pancake_calories * num_pancakes) + (bacon_calories * num_bacon_strips) + (cereal_calories * num_cereal_bowls)

/-- Proves that Jerry's breakfast contains 1120 calories -/
theorem jerrys_breakfast_theorem : jerrys_breakfast_calories = 1120 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_breakfast_theorem_l382_38273


namespace NUMINAMATH_CALUDE_complex_point_first_quadrant_l382_38289

theorem complex_point_first_quadrant : 
  let z : ℂ := 1 / Complex.I
  (z^3 + 1).re > 0 ∧ (z^3 + 1).im > 0 := by sorry

end NUMINAMATH_CALUDE_complex_point_first_quadrant_l382_38289


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_is_right_angled_l382_38212

/-- Given a triangle ABC where the ratio of angles A : B : C is 2 : 3 : 5, 
    prove that one of the angles is 90°. -/
theorem triangle_with_angle_ratio_is_right_angled (A B C : ℝ) 
  (h_triangle : A + B + C = 180) 
  (h_ratio : ∃ (x : ℝ), A = 2*x ∧ B = 3*x ∧ C = 5*x) : 
  A = 90 ∨ B = 90 ∨ C = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_is_right_angled_l382_38212


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l382_38250

theorem quadratic_roots_condition (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ →
  x₁^2 + (a - 1) * x₁ + 2 * a - 5 = 0 →
  x₂^2 + (a - 1) * x₂ + 2 * a - 5 = 0 →
  1 / x₁ + 1 / x₂ < -3 / 5 →
  a > 5 / 2 ∧ a < 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l382_38250


namespace NUMINAMATH_CALUDE_absolute_value_minus_self_nonnegative_l382_38238

theorem absolute_value_minus_self_nonnegative (m : ℚ) : 0 ≤ |m| - m := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_minus_self_nonnegative_l382_38238


namespace NUMINAMATH_CALUDE_right_triangle_ratio_squared_l382_38203

/-- Given a right triangle with legs a and b, and hypotenuse c, 
    where b > a, a/b = (1/2) * (b/c), and a + b + c = 12, 
    prove that (a/b)² = 1/2 -/
theorem right_triangle_ratio_squared (a b c : ℝ) 
  (h1 : b > a)
  (h2 : a / b = (1 / 2) * (b / c))
  (h3 : a + b + c = 12)
  (h4 : c^2 = a^2 + b^2) : 
  (a / b)^2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_squared_l382_38203


namespace NUMINAMATH_CALUDE_vertex_sum_is_ten_l382_38297

/-- Two parabolas intersecting at two points -/
structure IntersectingParabolas where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h1 : -(3 - a)^2 + b = 6
  h2 : (3 - c)^2 + d = 6
  h3 : -(7 - a)^2 + b = 2
  h4 : (7 - c)^2 + d = 2

/-- The sum of x-coordinates of the vertices of two intersecting parabolas -/
def vertexSum (p : IntersectingParabolas) : ℝ := p.a + p.c

/-- Theorem: The sum of x-coordinates of the vertices of two intersecting parabolas is 10 -/
theorem vertex_sum_is_ten (p : IntersectingParabolas) : vertexSum p = 10 := by
  sorry

end NUMINAMATH_CALUDE_vertex_sum_is_ten_l382_38297


namespace NUMINAMATH_CALUDE_units_digit_of_3_pow_2011_l382_38206

def units_digit (n : ℕ) : ℕ := n % 10

def power_of_3_units_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

theorem units_digit_of_3_pow_2011 :
  units_digit (3^2011) = power_of_3_units_digit 2011 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_3_pow_2011_l382_38206


namespace NUMINAMATH_CALUDE_cube_face_area_l382_38232

theorem cube_face_area (V : ℝ) (h : V = 125) : ∃ (A : ℝ), A = 25 ∧ A = (V ^ (1/3)) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_area_l382_38232


namespace NUMINAMATH_CALUDE_fraction_evaluation_l382_38208

theorem fraction_evaluation : (((5 * 4) + 6) : ℝ) / 10 = 2.6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l382_38208


namespace NUMINAMATH_CALUDE_total_birds_count_l382_38269

/-- The number of geese in the marsh -/
def num_geese : ℕ := 58

/-- The number of ducks in the marsh -/
def num_ducks : ℕ := 37

/-- The total number of birds in the marsh -/
def total_birds : ℕ := num_geese + num_ducks

/-- Theorem: The total number of birds in the marsh is 95 -/
theorem total_birds_count : total_birds = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_count_l382_38269


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l382_38268

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a₃_eq : a 3 = -2
  aₙ_eq : ∃ n : ℕ, a n = 3/2
  Sₙ_eq : ∃ n : ℕ, (n : ℚ) * (a 1 + a n) / 2 = -15/2

/-- The first term of the arithmetic sequence is either -3 or -19/6 -/
theorem arithmetic_sequence_first_term (seq : ArithmeticSequence) :
  seq.a 1 = -3 ∨ seq.a 1 = -19/6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l382_38268


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_l382_38218

theorem units_digit_of_7_power : 7^(100^6) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_l382_38218


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l382_38264

theorem simplify_and_evaluate (a b : ℝ) (ha : a = 3) (hb : b = 2) :
  (a + b) / (a * b) / ((a / b) - (b / a)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l382_38264


namespace NUMINAMATH_CALUDE_product_inequality_l382_38236

theorem product_inequality (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hab : a * b > 1) (hn : n ≥ 2) :
  (a + b)^n > a^n + b^n + 2^n - 2 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l382_38236


namespace NUMINAMATH_CALUDE_power_of_three_squared_l382_38234

theorem power_of_three_squared : 3^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_squared_l382_38234


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l382_38230

theorem stratified_sampling_sample_size 
  (total_population : ℕ) 
  (selection_probability : ℝ) 
  (sample_size : ℕ) :
  total_population = 1200 →
  selection_probability = 0.4 →
  (sample_size : ℝ) / total_population = selection_probability →
  sample_size = 480 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l382_38230


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l382_38296

/-- Represents a hyperbola with equation x²/m - y²/3 = 1 -/
structure Hyperbola where
  m : ℝ
  focus : ℝ × ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  h.focus = (2, 0) → eccentricity h = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l382_38296


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_from_one_vertex_l382_38252

/-- A dodecagon is a polygon with 12 sides. -/
def Dodecagon : Nat := 12

/-- The number of diagonals that can be drawn from one vertex of a polygon with n sides. -/
def diagonalsFromOneVertex (n : Nat) : Nat := n - 3

theorem dodecagon_diagonals_from_one_vertex :
  diagonalsFromOneVertex Dodecagon = 9 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_from_one_vertex_l382_38252


namespace NUMINAMATH_CALUDE_violet_buddy_hiking_time_l382_38221

/-- Represents the hiking scenario of Violet and Buddy -/
structure HikingScenario where
  violet_water_rate : Real  -- ml per hour
  buddy_water_rate : Real   -- ml per hour
  violet_capacity : Real    -- L
  buddy_capacity : Real     -- L
  hiking_speed : Real       -- km/h
  break_interval : Real     -- hours
  break_duration : Real     -- hours

/-- Calculates the total time Violet and Buddy can spend on the trail before running out of water -/
def total_trail_time (scenario : HikingScenario) : Real :=
  sorry

/-- Theorem stating that Violet and Buddy can spend 6.25 hours on the trail before running out of water -/
theorem violet_buddy_hiking_time :
  let scenario : HikingScenario := {
    violet_water_rate := 800,
    buddy_water_rate := 400,
    violet_capacity := 4.8,
    buddy_capacity := 1.5,
    hiking_speed := 4,
    break_interval := 2,
    break_duration := 0.5
  }
  total_trail_time scenario = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_violet_buddy_hiking_time_l382_38221


namespace NUMINAMATH_CALUDE_maia_remaining_requests_l382_38214

/-- Calculates the number of remaining client requests after a given number of days -/
def remaining_requests (daily_intake : ℕ) (daily_completion : ℕ) (days : ℕ) : ℕ :=
  (daily_intake - daily_completion) * days

/-- Theorem: Given Maia's work pattern, she will have 10 remaining requests after 5 days -/
theorem maia_remaining_requests :
  remaining_requests 6 4 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_maia_remaining_requests_l382_38214


namespace NUMINAMATH_CALUDE_cosine_value_on_unit_circle_l382_38292

theorem cosine_value_on_unit_circle (α : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = 1/2 ∧ y = Real.sqrt 3 / 2) →
  Real.cos α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_on_unit_circle_l382_38292


namespace NUMINAMATH_CALUDE_jenny_garden_area_l382_38295

/-- Represents a rectangular garden with fence posts. -/
structure Garden where
  total_posts : ℕ
  post_spacing : ℕ
  shorter_side_posts : ℕ
  longer_side_posts : ℕ

/-- Calculates the area of a rectangular garden given its specifications. -/
def garden_area (g : Garden) : ℕ :=
  (g.shorter_side_posts - 1) * g.post_spacing * ((g.longer_side_posts - 1) * g.post_spacing)

/-- Theorem: The area of Jenny's garden is 144 square yards. -/
theorem jenny_garden_area :
  ∀ g : Garden,
    g.total_posts = 24 →
    g.post_spacing = 3 →
    g.longer_side_posts = 3 * g.shorter_side_posts →
    g.total_posts = 2 * (g.shorter_side_posts + g.longer_side_posts - 2) →
    garden_area g = 144 := by
  sorry

#eval garden_area { total_posts := 24, post_spacing := 3, shorter_side_posts := 3, longer_side_posts := 9 }

end NUMINAMATH_CALUDE_jenny_garden_area_l382_38295


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l382_38220

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  first_sum : a 1 + a 8 = 10
  second_sum : a 2 + a 9 = 18

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) :
  common_difference seq = 4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l382_38220


namespace NUMINAMATH_CALUDE_vector_magnitude_l382_38255

theorem vector_magnitude (m : ℝ) : 
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![m, 2]
  (∃ (k : ℝ), b = k • a) → 
  ‖a + 2 • b‖ = 3 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l382_38255


namespace NUMINAMATH_CALUDE_cube_sum_gt_product_sum_l382_38216

theorem cube_sum_gt_product_sum {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^3 + b^3 > a^2*b + a*b^2 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_gt_product_sum_l382_38216


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l382_38258

/-- 
Given a two-digit number where the difference between the original number 
and the number with interchanged digits is 81, the difference between its 
two digits is 9.
-/
theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 81 → x - y = 9 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l382_38258


namespace NUMINAMATH_CALUDE_additional_money_needed_mrs_smith_shopping_l382_38257

/-- Calculates the additional money needed for Mrs. Smith's shopping trip --/
theorem additional_money_needed (total_budget : ℚ) (dress_budget : ℚ) (shoe_budget : ℚ) (accessory_budget : ℚ)
  (increase_ratio : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_needed := (dress_budget + shoe_budget + accessory_budget) * (1 + increase_ratio)
  let discounted_total := total_needed * (1 - discount_rate)
  discounted_total - total_budget

/-- Proves that Mrs. Smith needs $95 more --/
theorem mrs_smith_shopping : 
  additional_money_needed 500 300 150 50 (2/5) (15/100) = 95 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_mrs_smith_shopping_l382_38257


namespace NUMINAMATH_CALUDE_combined_bus_capacity_l382_38201

/-- The capacity of the train -/
def train_capacity : ℕ := 120

/-- The number of buses -/
def num_buses : ℕ := 2

/-- The capacity of each bus as a fraction of the train's capacity -/
def bus_capacity_fraction : ℚ := 1 / 6

/-- Theorem stating the combined capacity of the buses -/
theorem combined_bus_capacity :
  (↑train_capacity * bus_capacity_fraction * ↑num_buses : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_combined_bus_capacity_l382_38201


namespace NUMINAMATH_CALUDE_graduates_distribution_l382_38217

def distribute_graduates (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem graduates_distribution :
  distribute_graduates 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_graduates_distribution_l382_38217


namespace NUMINAMATH_CALUDE_exists_special_function_l382_38280

/-- A function from pairs of positive integers to positive integers -/
def PositiveIntegerFunction := ℕ+ → ℕ+ → ℕ+

/-- Predicate for a function being a polynomial in one variable when the other is fixed -/
def IsPolynomialInOneVariable (f : PositiveIntegerFunction) : Prop :=
  (∀ x : ℕ+, ∃ Px : ℕ+ → ℕ+, ∀ y : ℕ+, f x y = Px y) ∧
  (∀ y : ℕ+, ∃ Qy : ℕ+ → ℕ+, ∀ x : ℕ+, f x y = Qy x)

/-- Predicate for a function not being a polynomial in both variables -/
def IsNotPolynomialInBothVariables (f : PositiveIntegerFunction) : Prop :=
  ¬∃ P : ℕ+ → ℕ+ → ℕ+, ∀ x y : ℕ+, f x y = P x y

/-- The main theorem stating the existence of a function with the required properties -/
theorem exists_special_function : 
  ∃ f : PositiveIntegerFunction, 
    IsPolynomialInOneVariable f ∧ IsNotPolynomialInBothVariables f := by
  sorry

end NUMINAMATH_CALUDE_exists_special_function_l382_38280


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l382_38228

theorem sqrt_sum_equality : 
  (Real.sqrt 54 - Real.sqrt 27) + Real.sqrt 3 + 8 * Real.sqrt (1/2) = 
  3 * Real.sqrt 6 - 2 * Real.sqrt 3 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l382_38228


namespace NUMINAMATH_CALUDE_olivias_albums_l382_38281

def number_of_albums (pictures_from_phone : ℕ) (pictures_from_camera : ℕ) (pictures_per_album : ℕ) : ℕ :=
  (pictures_from_phone + pictures_from_camera) / pictures_per_album

theorem olivias_albums :
  let pictures_from_phone : ℕ := 5
  let pictures_from_camera : ℕ := 35
  let pictures_per_album : ℕ := 5
  number_of_albums pictures_from_phone pictures_from_camera pictures_per_album = 8 := by
  sorry

end NUMINAMATH_CALUDE_olivias_albums_l382_38281
