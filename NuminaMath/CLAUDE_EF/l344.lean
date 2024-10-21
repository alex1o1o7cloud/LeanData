import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_median_impossibility_l344_34406

/-- A polynomial of degree 2 -/
def polynomial (a₂ a₁ a₀ : ℝ) (x : ℝ) : ℝ := a₂ * x^2 + a₁ * x + a₀

/-- The median of three distinct real numbers -/
noncomputable def median (a b c : ℝ) : ℝ := 
  if a < b ∧ b < c ∨ c < b ∧ b < a then b
  else if b < a ∧ a < c ∨ c < a ∧ a < b then a
  else c

theorem polynomial_median_impossibility (a₂ a₁ a₀ : ℝ) 
  (h_distinct : a₂ ≠ a₁ ∧ a₁ ≠ a₀ ∧ a₂ ≠ a₀)
  (h_p_minus_one : polynomial a₂ a₁ a₀ (-1) = 1)
  (h_p_zero : polynomial a₂ a₁ a₀ 0 = 2)
  (h_p_one : polynomial a₂ a₁ a₀ 1 = 3) :
  ¬(polynomial (median a₂ a₁ a₀) (median a₂ a₁ a₀) (median a₂ a₁ a₀) (-1) = 3 ∧
    polynomial (median a₂ a₁ a₀) (median a₂ a₁ a₀) (median a₂ a₁ a₀) 0 = 1 ∧
    polynomial (median a₂ a₁ a₀) (median a₂ a₁ a₀) (median a₂ a₁ a₀) 1 = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_median_impossibility_l344_34406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_distribution_l344_34465

/-- A distribution of peanuts among monkeys -/
def Distribution := Fin 100 → ℕ

/-- The total number of peanuts -/
def total_peanuts : ℕ := 1600

/-- A valid distribution sums to the total number of peanuts -/
def valid_distribution (d : Distribution) : Prop :=
  (Finset.univ.sum d) = total_peanuts

/-- At least 4 monkeys receive the same number of peanuts -/
def at_least_four_same (d : Distribution) : Prop :=
  ∃ (n : ℕ), (Finset.filter (fun i => d i = n) Finset.univ).card ≥ 4

/-- No 5 monkeys receive the same number of peanuts -/
def no_five_same (d : Distribution) : Prop :=
  ∀ (n : ℕ), (Finset.filter (fun i => d i = n) Finset.univ).card < 5

theorem peanut_distribution :
  (∀ d : Distribution, valid_distribution d → at_least_four_same d) ∧
  (∃ d : Distribution, valid_distribution d ∧ no_five_same d) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_distribution_l344_34465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_ones_correct_prob_two_ones_approx_l344_34462

/-- The probability of rolling exactly two 1s when rolling twelve standard 6-sided dice -/
def prob_two_ones : ℚ :=
  (66 * 9765625) / 2176782336

/-- The number of standard dice being rolled -/
def num_dice : ℕ := 12

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of dice expected to show a 1 -/
def target_ones : ℕ := 2

theorem prob_two_ones_correct :
  prob_two_ones = (Nat.choose num_dice target_ones : ℚ) *
    (1 / num_sides) ^ target_ones *
    ((num_sides - 1) / num_sides) ^ (num_dice - target_ones) :=
by
  sorry

theorem prob_two_ones_approx :
  abs (prob_two_ones - 293/1000) < 1/1000 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_ones_correct_prob_two_ones_approx_l344_34462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l344_34411

/-- Curve C defined in polar coordinates -/
def C : Set (ℝ × ℝ) :=
  {(x, y) | ∃ θ : ℝ, x = (6 / Real.sqrt (4 * Real.cos θ^2 + 9 * Real.sin θ^2)) * Real.cos θ ∧
                      y = (6 / Real.sqrt (4 * Real.cos θ^2 + 9 * Real.sin θ^2)) * Real.sin θ}

/-- Theorem stating the Cartesian equation of curve C and the maximum value of 3x + 4y -/
theorem curve_C_properties :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / 9 + y^2 / 4 = 1) ∧
  (∃ (M : ℝ), M = Real.sqrt 145 ∧ ∀ (x y : ℝ), (x, y) ∈ C → 3*x + 4*y ≤ M) := by
  sorry

#check curve_C_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l344_34411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_tangent_l344_34494

theorem perpendicular_vectors_tangent (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π / 2))
  (a b : Fin 2 → Real)
  (h2 : a = λ i => if i = 0 then Real.cos θ else 2)
  (h3 : b = λ i => if i = 0 then -1 else Real.sin θ)
  (h4 : (a 0) * (b 0) + (a 1) * (b 1) = 0) : 
  Real.tan θ = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_tangent_l344_34494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_decrease_percentage_l344_34474

theorem wage_decrease_percentage (original_wage original_hours : ℝ) 
  (h_positive_wage : original_wage > 0)
  (h_positive_hours : original_hours > 0) :
  let new_hours := 1.25 * original_hours
  let new_wage := (original_wage * original_hours) / new_hours
  (original_wage - new_wage) / original_wage = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_decrease_percentage_l344_34474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_1027_terms_equals_2010_l344_34427

def mySequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 
    let block := (n / (n + 2)) + 1
    let position_in_block := n % (block + 1)
    if position_in_block = 0 then 1 else 2

def sum_first_n_terms (n : ℕ) : ℕ :=
  (List.range n).map mySequence |> List.sum

theorem sum_of_first_1027_terms_equals_2010 : sum_first_n_terms 1027 = 2010 := by
  sorry

#eval sum_first_n_terms 1027

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_1027_terms_equals_2010_l344_34427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_is_350_l344_34467

/-- Represents a hotel with pricing and occupancy rules -/
structure Hotel where
  totalRooms : ℕ
  basePrice : ℝ
  priceIncrement : ℝ
  vacancyIncrement : ℝ
  maintenanceCost : ℝ

/-- Calculates the number of occupied rooms based on price increase -/
noncomputable def occupiedRooms (h : Hotel) (priceIncrease : ℝ) : ℝ :=
  h.totalRooms - (priceIncrease / h.priceIncrement) * h.vacancyIncrement

/-- Calculates the profit for a given price increase -/
noncomputable def profit (h : Hotel) (priceIncrease : ℝ) : ℝ :=
  let occupied := occupiedRooms h priceIncrease
  let price := h.basePrice + priceIncrease
  occupied * (price - h.maintenanceCost)

/-- The optimal price increase that maximizes profit -/
noncomputable def optimalPriceIncrease (h : Hotel) : ℝ :=
  h.priceIncrement * (h.totalRooms / 2 - (h.basePrice - h.maintenanceCost) / (2 * h.priceIncrement))

/-- Theorem stating that 350 yuan maximizes profit for the given hotel conditions -/
theorem optimal_price_is_350 (h : Hotel) 
    (h_rooms : h.totalRooms = 50)
    (h_base : h.basePrice = 180)
    (h_increment : h.priceIncrement = 10)
    (h_vacancy : h.vacancyIncrement = 1)
    (h_maintenance : h.maintenanceCost = 20) :
    h.basePrice + optimalPriceIncrease h = 350 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_is_350_l344_34467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l344_34487

-- Define the function f
noncomputable def f (x : ℝ) := (2017 : ℝ)^x + Real.log (Real.sqrt (x^2 + 1) + x) - (2017 : ℝ)^(-x) + 1

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  (f (2*x - 1) + f x > 2) ↔ (x > 1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l344_34487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_equality_l344_34489

theorem power_product_equality : ((-27 : ℝ) ^ (2/3 : ℝ)) * (9 : ℝ) ^ (-(3/2) : ℝ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_equality_l344_34489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finger_movement_explanation_center_of_mass_balance_ruler_balance_behavior_l344_34417

/-- Represents a ruler with a center of mass -/
structure Ruler where
  length : ℝ
  center_of_mass : ℝ

/-- Represents a finger supporting the ruler -/
structure Finger where
  position : ℝ
  friction : ℝ

/-- Represents the state of the ruler balanced on two fingers -/
structure BalancedRuler where
  ruler : Ruler
  finger1 : Finger
  finger2 : Finger

/-- The weight distribution on each finger -/
noncomputable def weight_distribution (b : BalancedRuler) : ℝ × ℝ :=
  sorry

/-- The movement of fingers based on weight distribution and friction -/
noncomputable def finger_movement (b : BalancedRuler) : ℝ × ℝ :=
  sorry

/-- Theorem stating that the movement of fingers is determined by weight distribution and friction -/
theorem finger_movement_explanation (b : BalancedRuler) :
  ∃ (f : (ℝ × ℝ) × ℝ × ℝ → ℝ × ℝ),
    finger_movement b = f (weight_distribution b, b.finger1.friction, b.finger2.friction) := by
  sorry

/-- Theorem stating that the center of mass remains balanced between the two fingers -/
theorem center_of_mass_balance (b : BalancedRuler) :
  b.ruler.center_of_mass = (b.finger1.position + b.finger2.position) / 2 := by
  sorry

/-- Main theorem explaining the observed behavior -/
theorem ruler_balance_behavior (b : BalancedRuler) :
  ∃ (f : (ℝ × ℝ) × ℝ × ℝ → ℝ × ℝ),
    finger_movement b = f (weight_distribution b, b.finger1.friction, b.finger2.friction) ∧
    b.ruler.center_of_mass = (b.finger1.position + b.finger2.position) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finger_movement_explanation_center_of_mass_balance_ruler_balance_behavior_l344_34417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lindsay_dolls_l344_34434

/-- The number of blonde-haired dolls Lindsay has -/
def blonde : ℕ := by sorry

/-- The number of brown-haired dolls Lindsay has -/
def brown : ℕ := 4 * blonde

/-- The number of black-haired dolls Lindsay has -/
def black : ℕ := brown - 2

theorem lindsay_dolls : blonde = 4 := by
  have h1 : black + brown = blonde + 26 := by sorry
  -- Rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lindsay_dolls_l344_34434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_labelings_l344_34415

/-- Represents a labeling of a tetrahedron's edges -/
def Labeling := Fin 6 → Fin 2

/-- Checks if a labeling is valid (sum of labels on each face equals 1) -/
def is_valid_labeling (l : Labeling) : Prop :=
  ∀ face : Fin 4, (l (face + 0) + l (face + 1) + l (face + 2)) = 1

/-- Two labelings are considered distinct even if they can be obtained from each other by rotations or reflections -/
def are_distinct (l1 l2 : Labeling) : Prop :=
  l1 ≠ l2

/-- The set of all valid labelings -/
def valid_labelings : Set Labeling :=
  {l : Labeling | is_valid_labeling l}

theorem tetrahedron_labelings :
  ∃ (labelings : Finset Labeling),
    Finset.card labelings = 8 ∧
    (∀ l ∈ labelings, l ∈ valid_labelings) ∧
    (∀ l1 l2, l1 ∈ labelings → l2 ∈ labelings → l1 ≠ l2 → are_distinct l1 l2) ∧
    (∀ l, l ∈ valid_labelings → ∃ l' ∈ labelings, ¬(are_distinct l l')) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_labelings_l344_34415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_is_92_l344_34469

/-- A regular polygon inscribed in a circle -/
structure InscribedPolygon where
  sides : ℕ
  vertices : Finset (ℝ × ℝ)

/-- The set of inscribed polygons in the problem -/
def problem_polygons : Finset InscribedPolygon := sorry

/-- Two polygons do not share a vertex -/
def no_shared_vertex (p q : InscribedPolygon) : Prop := 
  p.vertices ∩ q.vertices = ∅

/-- Consecutive primes in the range of polygon sides -/
def consecutive_primes (n m : ℕ) : Prop :=
  Nat.Prime n ∧ Nat.Prime m ∧ ∀ k, n < k ∧ k < m → ¬Nat.Prime k

/-- Polygons with consecutive prime sides do not intersect -/
def no_intersection_consecutive_primes (p q : InscribedPolygon) : Prop :=
  consecutive_primes p.sides q.sides → 
    (∀ x y : ℝ × ℝ, x ∈ p.vertices → y ∈ q.vertices → 
      ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (1 - t) • x + t • y ∉ Set.univ)

/-- The number of intersection points between two polygons -/
noncomputable def intersection_count (p q : InscribedPolygon) : ℕ := sorry

/-- The total number of intersection points for all pairs of polygons -/
noncomputable def total_intersections : ℕ := sorry

theorem intersection_count_is_92 : total_intersections = 92 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_is_92_l344_34469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l344_34422

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

noncomputable def g (x : ℝ) : ℝ := f (Real.pi/2 - x)

theorem function_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (k : ℤ), ∀ (x : ℝ), f (k * Real.pi + 5 * Real.pi / 6 - x) = f (k * Real.pi + 5 * Real.pi / 6 + x)) ∧
  (∀ (k : ℤ), ∀ (x y : ℝ), k * Real.pi - Real.pi/12 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 5 * Real.pi / 12 → f x < f y) ∧
  (∀ (x : ℝ), g x = -(1/2) * Real.sin (2 * x - 2 * Real.pi / 3)) ∧
  (∃ (x : ℝ), -Real.pi/4 ≤ x ∧ x ≤ Real.pi/2 ∧ g x = 1/2 ∧ ∀ (y : ℝ), -Real.pi/4 ≤ y ∧ y ≤ Real.pi/2 → g y ≤ g x) ∧
  (∃ (x : ℝ), -Real.pi/4 ≤ x ∧ x ≤ Real.pi/2 ∧ g x = -Real.sqrt 3 / 4 ∧ ∀ (y : ℝ), -Real.pi/4 ≤ y ∧ y ≤ Real.pi/2 → g x ≤ g y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l344_34422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l344_34484

/-- The minimum distance between two points on the hyperbola -/
noncomputable def min_distance (M N : ℝ × ℝ) : ℝ := sorry

/-- The equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (a b : ℝ) (k_PA k_PB : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : k_PA * k_PB = 3) (h4 : ∃ (M N : ℝ × ℝ), min_distance M N = 4) :
  ((∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
   ((x^2 / 4 - y^2 / 12 = 1) ∨ (9 * x^2 / 4 - 3 * y^2 / 4 = 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l344_34484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_constant_l344_34440

noncomputable section

/-- The ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 / (5/3) = 1

/-- The line that intersects the ellipse -/
def line (k x : ℝ) : ℝ := k * (x + 1)

/-- The point M -/
def M : ℝ × ℝ := (-7/3, 0)

/-- Theorem stating that the dot product MA · MB is constant -/
theorem dot_product_constant (k : ℝ) (A B : ℝ × ℝ) :
  ellipse A.1 A.2 ∧ 
  ellipse B.1 B.2 ∧ 
  A.2 = line k A.1 ∧ 
  B.2 = line k B.1 ∧
  (A.1 + B.1) / 2 = -1/2 →
  ((A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) : ℝ) = 4/9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_constant_l344_34440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_17th_roots_of_unity_l344_34420

open BigOperators
open Complex

theorem sum_of_17th_roots_of_unity (ω : ℂ) : 
  ω^17 = 1 → ∑ k in Finset.range 16, ω^(k + 1) = -1 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_17th_roots_of_unity_l344_34420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l344_34437

/-- The equation of a cubic curve -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

/-- The point on the curve -/
def point : ℝ × ℝ := (1, -1)

/-- The slope of the tangent line at the given point -/
def m : ℝ := f' point.1

/-- The equation of the tangent line -/
def tangent_line (x : ℝ) : ℝ := -3*x + 2

theorem tangent_line_equation :
  (∀ x, f x = x^3 - 3*x^2 + 1) →
  f point.1 = point.2 →
  (∀ x, f' x = 3*x^2 - 6*x) →
  m = f' point.1 →
  (∀ x, tangent_line x = -3*x + 2) →
  ∀ x, tangent_line x = m * (x - point.1) + point.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l344_34437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l344_34404

noncomputable section

def f (x : ℝ) : ℝ := -2 * x

def g (x : ℝ) : ℝ := x / Real.log x + x - 2

theorem min_value_theorem (x₁ x₂ : ℝ) (h₁ : x₁ < 0) (h₂ : x₂ > 0) (h₃ : f x₁ = g x₂) :
  ∃ (m : ℝ), m = 4 * Real.sqrt (Real.exp 1) - 2 ∧ x₂ - 2 * x₁ ≥ m :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l344_34404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_bound_l344_34414

-- Define the square
def square_side : ℝ := 10

-- Define the diamond
def diamond_area : ℝ := 50

-- Define the semicircle
noncomputable def semicircle_area : ℝ := Real.pi * 25 / 2

-- Define the shaded area (intersection of diamond and semicircle)
noncomputable def shaded_area : ℝ := sorry

-- Theorem statement
theorem shaded_area_bound : 
  shaded_area ≤ min diamond_area semicircle_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_bound_l344_34414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_probability_after_removal_l344_34400

/-- Represents a deck of cards with counts for each number -/
def Deck := Fin 10 → ℕ

/-- The initial deck with five cards of each number -/
def initialDeck : Deck := fun _ => 5

/-- Removes two cards of a given number from the deck -/
def removeTwo (d : Deck) (n : Fin 10) : Deck :=
  fun i => if i = n then d i - 2 else d i

/-- Calculates the total number of cards in the deck -/
def totalCards (d : Deck) : ℕ := (Finset.sum Finset.univ d)

/-- Calculates the number of pairs in the deck -/
def numberOfPairs (d : Deck) : ℕ := Finset.sum Finset.univ (fun i => (d i).choose 2)

/-- The probability of selecting a pair from the deck -/
noncomputable def pairProbability (d : Deck) : ℚ :=
  (numberOfPairs d : ℚ) / (totalCards d).choose 2

theorem pair_probability_after_removal :
  ∀ a b : Fin 10, a ≠ b →
  pairProbability (removeTwo (removeTwo initialDeck a) b) = 16 / 207 := by
  sorry

#eval (16 : ℕ) + 207

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_probability_after_removal_l344_34400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_surjective_except_zero_l344_34416

noncomputable def g (x : ℝ) : ℤ :=
  if x > -2 then Int.ceil (1 / (2 * x + 4))
  else if x < -2 then Int.floor (1 / (2 * x + 4))
  else 0  -- arbitrary value for x = -2, as g is not defined there

theorem g_surjective_except_zero :
  ∀ z : ℤ, z ≠ 0 → ∃ x : ℝ, x ≠ -2 ∧ g x = z :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_surjective_except_zero_l344_34416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_set_characterization_l344_34471

def is_good (A : Finset ℕ) : Prop :=
  (∀ a b c, a ∈ A → b ∈ A → c ∈ A → a ≠ b → b ≠ c → a ≠ c → Nat.gcd a (Nat.gcd b c) = 1) ∧
  (∀ b c, b ∈ A → c ∈ A → b ≠ c → ∃ a ∈ A, a ≠ b ∧ a ≠ c ∧ (b * c) % a = 0)

def is_form1 (A : Finset ℕ) : Prop :=
  ∃ a b : ℕ, A = {a, b, a * b} ∧ Nat.gcd a b = 1

def is_form2 (A : Finset ℕ) : Prop :=
  ∃ p q r : ℕ, A = {p * q, q * r, r * p} ∧ 
    Nat.gcd p q = 1 ∧ Nat.gcd q r = 1 ∧ Nat.gcd r p = 1

theorem good_set_characterization (A : Finset ℕ) :
  is_good A ↔ is_form1 A ∨ is_form2 A := by
  sorry

#check good_set_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_set_characterization_l344_34471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pine_lake_soccer_swimmers_l344_34482

theorem pine_lake_soccer_swimmers (N : ℝ) (hN : N > 0) : 
  let soccer_players := 0.7 * N
  let swimmers := 0.5 * N
  let soccer_swimmers := 0.3 * soccer_players
  let non_swimming_soccer_players := soccer_players - soccer_swimmers
  let non_swimmers := N - swimmers
  (non_swimming_soccer_players / non_swimmers) * 100 = 98 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pine_lake_soccer_swimmers_l344_34482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l344_34473

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 / Real.sqrt (x + 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > -1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l344_34473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_closer_to_F_l344_34401

/-- Triangle DEF with sides DE = 7, EF = 6, FD = 8 -/
structure Triangle (V : Type*) [MetricSpace V] where
  D : V
  E : V
  F : V
  de : dist D E = 7
  ef : dist E F = 6
  fd : dist F D = 8

/-- The probability that a randomly selected point in triangle DEF 
    is closer to vertex F than to either D or E -/
noncomputable def closerToF {V : Type*} [MetricSpace V] (t : Triangle V) : ℝ :=
  1 / 6

/-- Theorem stating that the probability is 1/6 -/
theorem prob_closer_to_F {V : Type*} [MetricSpace V] (t : Triangle V) :
  closerToF t = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_closer_to_F_l344_34401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_common_shares_l344_34495

/-- Calculates the number of common shares given the total dividend, preferred shares details, and dividend rates -/
theorem calculate_common_shares
  (par_value : ℚ)
  (preferred_shares : ℕ)
  (preferred_dividend_rate : ℚ)
  (common_dividend_rate : ℚ)
  (total_dividend : ℚ)
  (h1 : par_value = 50)
  (h2 : preferred_shares = 1200)
  (h3 : preferred_dividend_rate = 1/10)
  (h4 : common_dividend_rate = 7/100)
  (h5 : total_dividend = 16500) :
  (total_dividend - preferred_dividend_rate * par_value * preferred_shares) / (common_dividend_rate * par_value) = 3000 :=
by
  sorry

-- The #eval command is removed as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_common_shares_l344_34495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l344_34459

-- Define the power function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (-2 * m^2 + m + 2) * x^(m + 1)

-- Define the function y
noncomputable def y (m a x : ℝ) : ℝ := f m x - 4 * (a - 1) * x

-- Define the property of f being an even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the property of y being monotonic on an interval
def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∨
  (∀ x y, a < x ∧ x < y ∧ y < b → f x > f y)

-- State the theorem
theorem range_of_a (m : ℝ) :
  is_even_function (f m) →
  (∀ a : ℝ, is_monotonic_on (y m a) 2 4 ↔ a ≤ 2 ∨ a ≥ 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l344_34459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_lines_l344_34468

-- Define the points
def A : Fin 3 → ℝ := ![3, -2, 4]
def B : Fin 3 → ℝ := ![13, -12, 9]
def C : Fin 3 → ℝ := ![1, 6, -8]
def D : Fin 3 → ℝ := ![3, -1, 2]

-- Define the intersection point
def intersection_point : Fin 3 → ℝ := ![-7, 8, -1]

-- Theorem statement
theorem intersection_of_lines :
  ∃ (t s : ℝ),
    (∀ i : Fin 3, A i + t * (B i - A i) = intersection_point i) ∧
    (∀ i : Fin 3, C i + s * (D i - C i) = intersection_point i) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_lines_l344_34468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_even_l344_34472

-- Define g as an odd function
def g_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Define h using g
def h (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^5)|

-- Theorem statement
theorem h_is_even (g : ℝ → ℝ) (g_odd : g_odd g) : 
  ∀ x, h g (-x) = h g x := by
  intro x
  simp [h, g_odd]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_even_l344_34472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_l344_34413

/-- Given a circle with radius 12 meters and a sector with area 49.02857142857143 square meters,
    the angle at the center of the circle is approximately 39 degrees. -/
theorem sector_angle (r : ℝ) (area : ℝ) (h1 : r = 12) (h2 : area = 49.02857142857143) :
  ∃ θ : ℝ, abs (θ - 39) < 0.1 ∧ area = (θ / 360) * Real.pi * r^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_l344_34413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l344_34448

theorem angle_in_second_quadrant (θ : Real) :
  (∃ (P : ℝ × ℝ), P.1 = Real.sin θ * Real.cos θ ∧ P.2 = 2 * Real.cos θ ∧ P.1 < 0 ∧ P.2 < 0) →
  (0 < θ ∧ θ < Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l344_34448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l344_34444

-- Define a line in the form ax + by = c
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the slope of a line
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

-- Define when two lines are parallel
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

-- The given line 3x - 6y = 9
def given_line : Line := { a := 3, b := -6, c := 9 }

-- Theorem statement
theorem parallel_line_slope :
  ∀ l : Line, parallel l given_line → l.slope = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l344_34444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_integer_representation_l344_34456

theorem two_digit_integer_representation (a b : ℕ) :
  a < 10 → b < 10 → (10 * b + a = 10 * b + a) :=
by
  intros ha hb
  rfl

#check two_digit_integer_representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_integer_representation_l344_34456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_is_CD_l344_34425

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the angles in the quadrilateral
noncomputable def angle_ABD (q : Quadrilateral) : ℝ := sorry
noncomputable def angle_ADB (q : Quadrilateral) : ℝ := sorry
noncomputable def angle_BDC (q : Quadrilateral) : ℝ := sorry
noncomputable def angle_CBD (q : Quadrilateral) : ℝ := sorry

-- Define the segments in the quadrilateral
noncomputable def segment_AB (q : Quadrilateral) : ℝ := sorry
noncomputable def segment_BC (q : Quadrilateral) : ℝ := sorry
noncomputable def segment_CD (q : Quadrilateral) : ℝ := sorry
noncomputable def segment_DA (q : Quadrilateral) : ℝ := sorry
noncomputable def segment_BD (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem longest_segment_is_CD (q : Quadrilateral) 
  (h1 : angle_ABD q = 35)
  (h2 : angle_ADB q = 50)
  (h3 : angle_BDC q = 60)
  (h4 : angle_CBD q = 65) :
  segment_CD q > segment_AB q ∧
  segment_CD q > segment_BC q ∧
  segment_CD q > segment_DA q ∧
  segment_CD q > segment_BD q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_is_CD_l344_34425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_applicability_l344_34485

noncomputable def f (x : ℝ) := Real.log x + x
noncomputable def g (x : ℝ) := Real.exp x - 3 * x
def h (x : ℝ) := x^3 - 3 * x + 1
noncomputable def k (x : ℝ) := 4 * x^2 - 4 * Real.sqrt 5 * x + 5

theorem bisection_method_applicability :
  (∃ a b, a < b ∧ f a * f b < 0) ∧
  (∃ a b, a < b ∧ g a * g b < 0) ∧
  (∃ a b, a < b ∧ h a * h b < 0) ∧
  (∀ x, k x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_applicability_l344_34485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_coin_game_rule_exists_l344_34441

/-- Represents the binary expansion of an irrational number -/
def BinaryExpansion := ℕ → Bool

/-- A rule for the coin-tossing game -/
structure GameRule where
  decide : ℕ → Bool → Bool

theorem irrational_coin_game_rule_exists (p : ℝ) 
  (h_p_irrational : Irrational p) 
  (h_p_in_01 : p ∈ Set.Ioo 0 1) :
  ∃ (b : BinaryExpansion) (r : GameRule),
    (∀ n : ℕ, (1 / 2 : ℝ)^n * (if b n then 1 else 0) = p) ∧
    (∑' n : ℕ, (1 / 2 : ℝ)^n = (1 : ℝ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_coin_game_rule_exists_l344_34441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_five_terms_l344_34443

noncomputable def mySequence (n : ℕ) : ℚ := (-1)^n * ((2 * n - 1 : ℚ) / (2^n : ℚ))

theorem sequence_first_five_terms :
  (mySequence 1 = -1/2) ∧
  (mySequence 2 = 3/4) ∧
  (mySequence 3 = -5/8) ∧
  (mySequence 4 = 7/16) ∧
  (mySequence 5 = -9/32) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_five_terms_l344_34443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l344_34478

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

/-- The shifted function g(x, φ) -/
noncomputable def g (x φ : ℝ) : ℝ := f (x - φ)

/-- A function is symmetric about the y-axis if f(x) = f(-x) for all x -/
def is_symmetric_about_y_axis (h : ℝ → ℝ) : Prop :=
  ∀ x, h x = h (-x)

/-- The main theorem -/
theorem min_shift_for_symmetry :
  ∃ φ_min : ℝ, φ_min > 0 ∧ 
    is_symmetric_about_y_axis (g · φ_min) ∧
    (∀ φ, φ > 0 → is_symmetric_about_y_axis (g · φ) → φ ≥ φ_min) ∧
    φ_min = Real.pi / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l344_34478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_is_two_subset_condition_l344_34476

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |2*x - a| ≥ 2}
def B : Set ℝ := {x | 2 < Real.exp (x * Real.log 2) ∧ Real.exp (x * Real.log 2) < 8}

-- Theorem 1: When a = 2, A ∩ B = [2, 3)
theorem intersection_when_a_is_two :
  A 2 ∩ B = Set.Icc 2 3 := by sorry

-- Theorem 2: Given A ∩ B ≠ ∅, B ⊆ A iff a ∈ (-∞, 0] ∪ [8, +∞)
theorem subset_condition (a : ℝ) :
  (A a ∩ B).Nonempty → (B ⊆ A a ↔ a ≤ 0 ∨ a ≥ 8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_is_two_subset_condition_l344_34476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l344_34438

def A : Set ℝ := {x : ℝ | x ∈ ({0, 1, 2, 3, 4, 5} : Set ℝ)}
def B : Set ℝ := {x : ℝ | |x - 2| ≤ 1}

theorem intersection_A_B : A ∩ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l344_34438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_36_l344_34479

/-- Calculates the downstream distance given the upstream distance, time, and speed in still water. -/
noncomputable def downstream_distance (upstream_distance : ℝ) (time : ℝ) (still_water_speed : ℝ) : ℝ :=
  let stream_speed := (still_water_speed - upstream_distance / time) / 2
  (still_water_speed + stream_speed) * time

/-- Theorem stating that under the given conditions, the downstream distance is 36 km. -/
theorem downstream_distance_is_36 :
  downstream_distance 26 2 15.5 = 36 := by
  -- Unfold the definition of downstream_distance
  unfold downstream_distance
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_36_l344_34479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l344_34490

/-- Ellipse C with equation (x^2/16) + (y^2/12) = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

/-- Right focus of the ellipse -/
def F : ℝ × ℝ := (2, 0)

/-- Line l: x = 8 -/
def line_l (x : ℝ) : Prop := x = 8

/-- Fixed point E -/
def E : ℝ × ℝ := (5, 0)

/-- Origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- Area of a triangle given three points -/
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- Theorem stating that BD passes through E and maximum area of OBD is 15 -/
theorem ellipse_theorem (A B D : ℝ × ℝ) 
  (hA : ellipse_C A.1 A.2) 
  (hB : ellipse_C B.1 B.2)
  (hD : line_l D.1)
  (h_line_AB : ∃ (m : ℝ), A.1 = m * A.2 + F.1 ∧ B.1 = m * B.2 + F.1)
  (h_perp_AD : (A.2 - D.2) * (A.1 - D.1) = 0) :
  (∃ (t : ℝ), (1 - t) • B + t • D = E) ∧
  (∀ (A' B' D' : ℝ × ℝ), 
    ellipse_C A'.1 A'.2 → 
    ellipse_C B'.1 B'.2 → 
    line_l D'.1 → 
    (∃ (m : ℝ), A'.1 = m * A'.2 + F.1 ∧ B'.1 = m * B'.2 + F.1) →
    (A'.2 - D'.2) * (A'.1 - D'.1) = 0 →
    area_triangle O B' D' ≤ 15) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l344_34490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_brad_meeting_time_l344_34408

/-- The time it takes for Maxwell and Brad to meet -/
noncomputable def meeting_time (distance : ℝ) (maxwell_speed : ℝ) (brad_speed : ℝ) (head_start : ℝ) : ℝ :=
  (distance - maxwell_speed * head_start) / (maxwell_speed + brad_speed)

/-- The total time Maxwell walks before meeting Brad -/
noncomputable def maxwell_total_time (distance : ℝ) (maxwell_speed : ℝ) (brad_speed : ℝ) (head_start : ℝ) : ℝ :=
  meeting_time distance maxwell_speed brad_speed head_start + head_start

theorem maxwell_brad_meeting_time :
  maxwell_total_time 24 4 6 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_brad_meeting_time_l344_34408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_ring_circle_sum_bounds_l344_34446

/-- Represents the sum of numbers in each circle -/
def CircleSum (s : ℕ) : Prop := true

/-- The set of numbers to be placed in the circles -/
def NumberSet : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The number of circles in the five-ring configuration -/
def NumCircles : ℕ := 5

/-- Predicate to check if a given sum is valid for the five-ring problem -/
def IsValidSum (s : ℕ) : Prop :=
  ∃ (a b c d e f g h i : ℕ),
    a ∈ NumberSet ∧ b ∈ NumberSet ∧ c ∈ NumberSet ∧ d ∈ NumberSet ∧
    e ∈ NumberSet ∧ f ∈ NumberSet ∧ g ∈ NumberSet ∧ h ∈ NumberSet ∧
    i ∈ NumberSet ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
    g ≠ h ∧ g ≠ i ∧
    h ≠ i ∧
    CircleSum s ∧
    a + b = s ∧
    b + c + d = s ∧
    d + e + f = s ∧
    f + g + h = s ∧
    h + i = s

theorem five_ring_circle_sum_bounds :
  (∀ s, IsValidSum s → s ≤ 14) ∧
  (∀ s, IsValidSum s → s ≥ 11) ∧
  IsValidSum 14 ∧
  IsValidSum 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_ring_circle_sum_bounds_l344_34446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l344_34405

theorem triangle_trigonometric_identity 
  (A B C : Real) (a b c : Real) (h : Real) :
  A + B + C = π → -- sum of angles in a triangle
  a > 0 → b > 0 → c > 0 → -- positive side lengths
  c * Real.sin A = h → -- definition of height
  c - a = h → -- given condition
  (Real.cos (A/2) - Real.sin (A/2)) * (Real.sin (C/2) + Real.cos (C/2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l344_34405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_cost_per_mile_l344_34419

/-- Represents the cost per mile for a taxi ride. -/
def cost_per_mile : ℝ := sorry

/-- Represents the initial fee for a taxi ride. -/
def initial_fee : ℝ := 2.50

/-- Represents the bridge toll fee. -/
def bridge_toll : ℝ := 5.00

/-- Represents the length of Annie's ride in miles. -/
def annie_miles : ℝ := 16

/-- Represents the length of Mike's ride in miles. -/
def mike_miles : ℝ := 36

/-- Theorem stating that the cost per mile for the taxi ride is $0.25. -/
theorem taxi_cost_per_mile :
  (initial_fee + mike_miles * cost_per_mile = 
   initial_fee + bridge_toll + annie_miles * cost_per_mile) →
  cost_per_mile = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_cost_per_mile_l344_34419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l344_34461

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define the function for which A is the domain
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x + Real.sqrt (3 - x)

-- Define set A as the domain of f
def A : Set ℝ := {x | x ∈ U ∧ x ≥ 0 ∧ 3 - x ≥ 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = (2:ℝ)^x ∧ 1 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem problem_solution :
  A = Set.Icc 0 3 ∧
  B = Set.Icc 2 4 ∧
  (Aᶜ ∩ B) = Set.Ioc 3 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l344_34461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angelinas_speed_l344_34410

/-- Angelina's walking problem -/
theorem angelinas_speed
  (distance_home_grocery : ℝ)
  (distance_grocery_gym : ℝ)
  (speed_home_grocery : ℝ)
  (speed_grocery_gym : ℝ)
  (time_difference : ℝ)
  (h1 : distance_home_grocery = 150)
  (h2 : distance_grocery_gym = 200)
  (h3 : speed_grocery_gym = 2 * speed_home_grocery)
  (h4 : distance_grocery_gym / speed_grocery_gym = distance_home_grocery / speed_home_grocery - time_difference)
  (h5 : time_difference = 10) :
  speed_grocery_gym = 10 := by
  sorry

#check angelinas_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angelinas_speed_l344_34410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l344_34435

noncomputable section

open Real

def f (x : ℝ) := (1/2) * (cos x)^2 + (sqrt 3 / 2) * sin x * cos x

def g (x : ℝ) := f (x - π/6)

theorem f_properties :
  (∀ x : ℝ, f x ≥ -1/4) ∧
  (∀ k : ℤ, f (k * π - π/3) = -1/4) ∧
  (∀ x : ℝ, ∃ k : ℤ, StrictMonoOn g (Set.Ioo (-π/4 + k * π) (π/4 + k * π))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l344_34435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_positive_when_sin_positive_l344_34458

theorem tan_half_positive_when_sin_positive (α : ℝ) (h : Real.sin α > 0) :
  Real.tan (α / 2) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_positive_when_sin_positive_l344_34458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_to_increase_avg_l344_34402

noncomputable def current_scores : List ℝ := [95, 85, 75, 65, 90]
def score_increase : ℝ := 4

noncomputable def min_next_score (scores : List ℝ) (increase : ℝ) : ℝ :=
  let current_avg := scores.sum / scores.length
  let target_avg := current_avg + increase
  let total_required := target_avg * (scores.length + 1)
  total_required - scores.sum

theorem min_score_to_increase_avg :
  min_next_score current_scores score_increase = 106 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_to_increase_avg_l344_34402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_form_l344_34423

/-- Triangle vertices -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (10, -2)
def C : ℝ × ℝ := (7, 5)

/-- Point Q -/
def Q : ℝ × ℝ := (5, 3)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Sum of distances from Q to A, B, and C -/
noncomputable def sum_distances : ℝ := distance Q A + distance Q B + distance Q C

/-- Theorem: The sum of distances can be expressed as p + q√r with p + q = 7 -/
theorem sum_distances_form :
  ∃ (p q r : ℤ), sum_distances = ↑p + ↑q * Real.sqrt ↑r ∧ p + q = 7 ∧ 
  (∀ (p' q' r' : ℤ), sum_distances = ↑p' + ↑q' * Real.sqrt ↑r' → r ≤ r') := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_form_l344_34423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_with_given_radii_l344_34496

/-- The radius of the inscribed circle of a triangle -/
def inscribedCircleRadius (a b c : ℝ) : ℝ := sorry

/-- The radius of the circumscribed circle of a triangle -/
def circumscribedCircleRadius (a b c : ℝ) : ℝ := sorry

/-- Predicate to check if three lengths can form a triangle -/
def IsTriangle (a b c : ℝ) : Prop := sorry

/-- An isosceles triangle with given inscribed and circumscribed circle radii has specific integer side lengths -/
theorem isosceles_triangle_with_given_radii :
  ∀ (a b c : ℝ) (r R : ℝ),
  a = b →
  r = 3/2 →
  R = 25/8 →
  IsTriangle a b c →
  inscribedCircleRadius a b c = r →
  circumscribedCircleRadius a b c = R →
  ∃ (x y z : ℕ), (a = x ∧ b = y ∧ c = z) →
  (a = 5 ∧ b = 5 ∧ c = 6) ∨ (a = 5 ∧ b = 5 ∧ c = 6) ∨ (a = 6 ∧ b = 5 ∧ c = 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_with_given_radii_l344_34496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_l344_34455

/-- A function satisfying certain symmetry properties -/
def g : ℝ → ℝ := sorry

/-- The function g is symmetric about x = 3 -/
axiom g_sym_3 : ∀ x : ℝ, g (3 + x) = g (3 - x)

/-- The function g is symmetric about x = 8 -/
axiom g_sym_8 : ∀ x : ℝ, g (8 + x) = g (8 - x)

/-- The function g equals zero at x = 0 -/
axiom g_zero : g 0 = 0

/-- The count of roots of g in the interval [-1000, 1000] -/
def root_count : ℕ := sorry

/-- The theorem stating the minimum number of roots of g in [-1000, 1000] -/
theorem min_roots : root_count ≥ 401 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_l344_34455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_weight_is_14_l344_34497

-- Define the weight of one canoe
def canoe_weight : ℚ := 35

-- Define the number of bowling balls and canoes
def num_bowling_balls : ℕ := 10
def num_canoes : ℕ := 4

-- Define the total weight of bowling balls and canoes
def total_weight : ℚ := canoe_weight * num_canoes

-- Define the weight of one bowling ball
noncomputable def bowling_ball_weight : ℚ := total_weight / num_bowling_balls

-- Theorem to prove
theorem bowling_ball_weight_is_14 : bowling_ball_weight = 14 := by
  -- Unfold the definitions
  unfold bowling_ball_weight total_weight canoe_weight num_bowling_balls num_canoes
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_weight_is_14_l344_34497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_lower_bound_and_function_inequality_l344_34442

theorem cosine_lower_bound_and_function_inequality (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ≥ 0, Real.cos x ≥ 1 - (1/2) * x^2) ∧
  (a ≥ 1 → ∀ x ≥ 0, f x ≥ (1 + Real.sin x)^2) :=
by
  have h1 : ∀ x ≥ 0, Real.cos x ≥ 1 - (1/2) * x^2 := by sorry
  have h2 : a ≥ 1 → ∀ x ≥ 0, f x ≥ (1 + Real.sin x)^2 := by
    intro ha x hx
    sorry
  exact ⟨h1, h2⟩

-- Definition of the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp (a * x) + x * Real.cos x + 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_lower_bound_and_function_inequality_l344_34442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l344_34491

-- Define the logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Theorem statement
theorem log_properties (x₁ x₂ : ℝ) (h1 : x₁ > 0) (h2 : x₂ > 0) (h3 : x₁ ≠ x₂) :
  (f (x₁ * x₂) = f x₁ + f x₂) ∧
  ((f x₁ - f x₂) / (x₁ - x₂) > 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l344_34491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_numbers_sum_l344_34463

theorem board_numbers_sum (n : ℕ) (h : n = 10) :
  ∃ (S : Finset ℕ),
    Finset.card S = n ∧
    (∀ x y, x ∈ S → y ∈ S → x ≠ y → x ≠ y) ∧
    (∃ A : Finset ℕ, A ⊆ S ∧ Finset.card A = 3 ∧ ∀ x ∈ A, x % 5 = 0) ∧
    (∃ B : Finset ℕ, B ⊆ S ∧ Finset.card B = 4 ∧ ∀ x ∈ B, x % 4 = 0) ∧
    Finset.sum S id < 75 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_numbers_sum_l344_34463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_nines_to_700_l344_34403

def count_digit (d : Nat) (start : Nat) (stop : Nat) : Nat :=
  sorry

theorem count_nines_to_700 :
  count_digit 9 1 700 = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_nines_to_700_l344_34403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_three_fixed_point_of_f_converges_to_fixed_point_l344_34498

-- Define the continued fraction function
noncomputable def continuedFraction (x : ℝ) : ℕ → ℝ
  | 0 => x / (1 + Real.sqrt (1 + x))
  | n + 1 => x / (2 + continuedFraction x n)

-- State the theorem
theorem solution_is_three :
  ∃ (x : ℝ), x > 0 ∧ continuedFraction x 1984 = 1 ∧ x = 3 := by
  -- Proof goes here
  sorry

-- Define the fixed point function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x) + 1

-- State a lemma about the fixed point
theorem fixed_point_of_f :
  f 3 = 3 := by
  -- Proof goes here
  sorry

-- State a theorem about the convergence to the fixed point
theorem converges_to_fixed_point :
  ∀ (x : ℝ), x > 0 → continuedFraction x 1984 = 1 → x = 3 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_three_fixed_point_of_f_converges_to_fixed_point_l344_34498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l344_34470

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time / 100

theorem interest_rate_problem (principal interest time : ℝ) 
  (h1 : principal = 800)
  (h2 : interest = 144)
  (h3 : time = 4)
  (h4 : simple_interest principal (45/10) time = interest) : 
  45/10 = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l344_34470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_intersection_product_of_distances_l344_34492

noncomputable section

/-- Line l with angle of inclination α -/
def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (3 + t * Real.cos α, t * Real.sin α)

/-- Curve C -/
def curve_C (θ : ℝ) : ℝ × ℝ :=
  (1 / Real.cos θ, Real.tan θ)

/-- Intersection points of line l and curve C -/
def intersection_points (α : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t θ, line_l α t = p ∧ curve_C θ = p}

/-- Midpoint of two points -/
def midpoint_of_segment (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

theorem midpoint_of_intersection :
  ∃ A B, A ∈ intersection_points (π/3) ∧
         B ∈ intersection_points (π/3) ∧
         A ≠ B ∧ 
         midpoint_of_segment A B = (9/2, 3*Real.sqrt 3/2) := by sorry

theorem product_of_distances :
  ∃ l : ℝ → ℝ × ℝ,
    (∀ t, (l t).2 = 2 * ((l t).1 - 3)) ∧
    (l 0 = (3, 0)) ∧
    ∃ A B, A ∈ intersection_points (Real.arctan 2) ∧
           B ∈ intersection_points (Real.arctan 2) ∧
           A ≠ B ∧ 
           let P := (3, 0)
           let PA := Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)
           let PB := Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)
           PA * PB = 40/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_intersection_product_of_distances_l344_34492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_negative_in_fourth_quadrant_l344_34451

/-- An angle is in the fourth quadrant if it's between 270° and 360° (exclusive) -/
def is_fourth_quadrant (α : Real) : Prop :=
  3/2 * Real.pi < α ∧ α < 2 * Real.pi

theorem tan_half_angle_negative_in_fourth_quadrant (α : Real) 
  (h : is_fourth_quadrant α) : 
  Real.tan (α/2) < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_negative_in_fourth_quadrant_l344_34451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_filled_tank_l344_34432

/-- Represents a cubical water tank -/
structure CubicalTank where
  side_length : ℝ
  water_height : ℝ
  water_volume : ℝ

/-- Calculates the fraction of a cubical tank filled with water -/
noncomputable def fraction_filled (tank : CubicalTank) : ℝ :=
  tank.water_volume / (tank.side_length ^ 3)

/-- Theorem: For a cubical tank with 16 cubic feet of water at 1 foot height, 
    the fraction filled is 1/4 -/
theorem quarter_filled_tank :
  ∀ (tank : CubicalTank), 
    tank.water_height = 1 → 
    tank.water_volume = 16 → 
    fraction_filled tank = 1/4 := by
  intro tank h1 h2
  sorry

#check quarter_filled_tank

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_filled_tank_l344_34432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_midpoint_slope_of_line_l_l344_34430

-- Define the circle C
noncomputable def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Define point B
def point_B : ℝ × ℝ := (0, 3)

-- Define the midpoint M of AB
noncomputable def midpoint_M (xa ya : ℝ) : ℝ × ℝ := ((xa + 0) / 2, (ya + 3) / 2)

-- Define the trajectory of M
noncomputable def trajectory_M (x y : ℝ) : Prop := x^2 + (y - 1.5)^2 = 1

-- Define the line l
noncomputable def line_l (k : ℝ) (x y : ℝ) : Prop := y - 3 = k * (x - 0)

-- Define the length of chord AB
noncomputable def chord_length : ℝ := 2 * Real.sqrt 19 / 5

-- Theorem for the trajectory of midpoint M
theorem trajectory_of_midpoint :
  ∀ (xa ya : ℝ), circle_C xa ya →
  let (xm, ym) := midpoint_M xa ya
  trajectory_M xm ym := by
  sorry

-- Theorem for the slope of line l
theorem slope_of_line_l :
  ∃ (k : ℝ), (k = 3 + Real.sqrt 22 / 2 ∨ k = 3 - Real.sqrt 22 / 2) ∧
  ∃ (xa ya : ℝ), circle_C xa ya ∧ line_l k xa ya ∧
  (xa - 0)^2 + (ya - 3)^2 = chord_length^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_midpoint_slope_of_line_l_l344_34430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_regular_triangular_pyramid_l344_34454

/-- The volume of a regular triangular pyramid with an inscribed sphere -/
noncomputable def triangularPyramidVolume (r : ℝ) (α : ℝ) : ℝ :=
  (r^3 * Real.sqrt 3 * (1 + Real.sqrt (1 + 4 * Real.tan α ^ 2))^3) / (12 * Real.tan α ^ 2)

/-- Axiom for the volume of a regular triangular pyramid with inscribed sphere -/
axiom volume_of_regular_triangular_pyramid (r : ℝ) (α : ℝ) : ℝ

/-- Theorem stating the volume of a regular triangular pyramid with an inscribed sphere -/
theorem volume_regular_triangular_pyramid
  (r : ℝ) (α : ℝ) (h_r : r > 0) (h_α : 0 < α ∧ α < π / 2) :
  ∃ (V : ℝ), V = triangularPyramidVolume r α ∧
    V = volume_of_regular_triangular_pyramid r α :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_regular_triangular_pyramid_l344_34454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_sphere_to_hemisphere_l344_34486

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def hemisphere_volume (r : ℝ) : ℝ := (1 / 2) * (4 / 3) * Real.pi * r^3

theorem volume_ratio_sphere_to_hemisphere (r : ℝ) (h : r > 0) :
  sphere_volume r / hemisphere_volume (3 * r) = 1 / 13.5 := by
  -- Expand definitions
  unfold sphere_volume hemisphere_volume
  -- Simplify the expression
  simp [Real.pi]
  -- Perform algebraic manipulations
  field_simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_sphere_to_hemisphere_l344_34486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_level_rise_ratio_specific_l344_34452

/-- Represents the ratio of liquid level rise between two cones -/
noncomputable def liquid_level_rise_ratio (r1 r2 m1 m2 : ℝ) : ℝ :=
  let h1 := 4 * r2^3 / r1^3
  let x := (3 * r1^3 + 4 * m1^3) / r1^3
  let y := (17 * r2^3 + 4 * m2^3) / (16 * r2^3)
  4 * (x^(1/3) - 1) / (y^(1/3) - 1)

/-- The ratio of liquid level rise in two cones with specific dimensions -/
theorem liquid_level_rise_ratio_specific :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |liquid_level_rise_ratio 4 8 2 1 - 8.156| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_level_rise_ratio_specific_l344_34452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_probability_l344_34481

/-- Probability of getting exactly n points in a coin toss game -/
noncomputable def P (n : ℕ) : ℝ :=
  (1/3) * (2 + (-1/2)^n)

/-- The coin toss game where heads give 1 point and tails give 2 points -/
theorem coin_toss_probability (n : ℕ) :
  P n = (1/3) * (2 + (-1/2)^n) :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_probability_l344_34481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_l344_34460

/-- The function for which we want to find the oblique asymptote -/
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 5 * x + 8) / (2 * x + 3)

/-- The proposed oblique asymptote function -/
noncomputable def g (x : ℝ) : ℝ := (3/2) * x + 1/4

/-- Theorem stating that g is the oblique asymptote of f -/
theorem oblique_asymptote : 
  ∀ ε > 0, ∃ M, ∀ x, abs x > M → abs (f x - g x) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_l344_34460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_lines_l344_34436

noncomputable section

/-- The curve equation -/
def curve (x : ℝ) : ℝ := (1/3) * x^3 + 4/3

/-- The point P -/
def P : ℝ × ℝ := (2, 4)

/-- Tangent line equation at P(2,4) -/
def tangent_at_P (x y : ℝ) : Prop := 4*x - y - 4 = 0

/-- Tangent lines passing through P(2,4) -/
def tangent_through_P (x y : ℝ) : Prop := 
  (4*x - y - 4 = 0) ∨ (x - y + 2 = 0)

/-- Derivative of the curve -/
def curve_deriv (x : ℝ) : ℝ := x^2

theorem curve_tangent_lines :
  (curve P.1 = P.2) →  -- P lies on the curve
  (∀ x y, tangent_at_P x y ↔ 
    (y - P.2 = (curve_deriv P.1) * (x - P.1))) ∧
  (∃ x₀, x₀ ≠ 2 ∧ 
    (∀ x y, tangent_through_P x y ↔ 
      (y - curve x₀ = (curve_deriv x₀) * (x - x₀)) ∧
      (4 - curve x₀ = (curve_deriv x₀) * (2 - x₀)))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_lines_l344_34436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_600_plus_tan_240_l344_34418

theorem sin_600_plus_tan_240 : 
  Real.sin (600 * π / 180) + Real.tan (240 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_600_plus_tan_240_l344_34418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l344_34493

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : b * (Real.sin B - Real.sin C) + (c - a) * (Real.sin A + Real.sin C) = 0)
  (h_a : a = Real.sqrt 3)
  (h_sin_c : Real.sin C = (1 + Real.sqrt 3) / 2 * Real.sin B) :
  A = Real.pi / 3 ∧ 
  (1/2) * a * b * Real.sin C = (3 + Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l344_34493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_m_l344_34453

theorem sum_of_valid_m : ∃ (S : Finset ℤ), 
  (∀ m ∈ S, (∀ x : ℝ, 3 * x + m = x - 5 → x < 0) ∧ 
             (∀ y : ℝ, (y + 2) / 3 - y / 2 < 1 ∧ 3 * (y - m) ≥ 0 ↔ y > -2)) ∧
  (S.sum id = -9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_m_l344_34453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_porridge_l344_34464

/-- Represents the amount of porridge each chick receives -/
def Porridge := Fin 7 → ℝ

/-- The conditions of the porridge distribution -/
structure PorridgeDistribution (p : Porridge) : Prop where
  third_eq_first_second : p 2 = p 0 + p 1
  fourth_eq_second_third : p 3 = p 1 + p 2
  fifth_eq_third_fourth : p 4 = p 2 + p 3
  sixth_eq_fourth_fifth : p 5 = p 3 + p 4
  seventh_empty : p 6 = 0
  fifth_amount : p 4 = 10

/-- The theorem stating the total amount of porridge -/
theorem total_porridge (p : Porridge) (h : PorridgeDistribution p) :
  Finset.sum Finset.univ p = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_porridge_l344_34464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_repair_cost_l344_34428

/-- Calculates the repair cost of a scooter given the profit percentage and total profit -/
noncomputable def repair_cost (profit_percentage : ℝ) (total_profit : ℝ) : ℝ :=
  let original_cost := total_profit / profit_percentage
  0.1 * original_cost

/-- Theorem: Given the conditions of the scooter sale, the repair cost is $550 -/
theorem scooter_repair_cost :
  repair_cost 0.2 1100 = 550 := by
  -- Unfold the definition of repair_cost
  unfold repair_cost
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_repair_cost_l344_34428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_XW_l344_34431

-- Define the triangle XYZ
def X : ℝ × ℝ := (0, 0)
def Y : ℝ × ℝ := (3, 0)
def Z : ℝ × ℝ := (0, 4)

-- Define W as a point on YZ
variable (W : ℝ × ℝ)

-- Define XW as the angle bisector of YXZ
def is_angle_bisector (X Y Z W : ℝ × ℝ) : Prop :=
  (W.1 - X.1) * (Z.2 - X.2) = (W.2 - X.2) * (Z.1 - X.1) ∧
  (W.1 - X.1) * (Y.2 - X.2) = (W.2 - X.2) * (Y.1 - X.1)

-- W is on YZ
axiom W_on_YZ : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ W = (Y.1 + t * (Z.1 - Y.1), Y.2 + t * (Z.2 - Y.2))

-- XW is the angle bisector
axiom XW_bisects : is_angle_bisector X Y Z W

-- Theorem to prove
theorem length_of_XW : 
  Real.sqrt ((W.1 - X.1)^2 + (W.2 - X.2)^2) = 12 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_XW_l344_34431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l344_34475

/-- Represents a square pyramid -/
structure SquarePyramid where
  base_edge : ℝ
  altitude : ℝ

/-- Calculate the volume of a square pyramid -/
noncomputable def pyramid_volume (p : SquarePyramid) : ℝ :=
  (1 / 3) * p.base_edge^2 * p.altitude

/-- Calculate the volume of a frustum formed by cutting a smaller pyramid from a larger one -/
noncomputable def frustum_volume (large : SquarePyramid) (scale : ℝ) : ℝ :=
  pyramid_volume large - pyramid_volume { base_edge := scale * large.base_edge, altitude := scale * large.altitude }

theorem frustum_volume_ratio (p : SquarePyramid) :
  frustum_volume p (1/3) = (26/27) * pyramid_volume p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l344_34475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_of_b_l344_34433

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m * p.1 + b}

-- State the theorem
theorem intersection_range_of_b :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) ↔ b ∈ Set.Icc (-(Real.sqrt 6)/2) ((Real.sqrt 6)/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_of_b_l344_34433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_calories_burned_l344_34480

/-- Represents the calories burned per minute while running -/
def calories_per_minute_running : ℝ → ℝ := sorry

/-- Represents the total calories burned in 60 minutes -/
def total_calories : ℝ → ℝ := sorry

/-- Represents the time spent running in minutes -/
def time_running : ℝ → ℝ := sorry

/-- Represents the calories burned per minute while walking -/
def calories_per_minute_walking : ℝ → ℝ := sorry

theorem athlete_calories_burned 
  (h1 : calories_per_minute_walking 1 = 4)
  (h2 : total_calories 1 = 450)
  (h3 : time_running 1 = 35) :
  calories_per_minute_running 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_calories_burned_l344_34480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l344_34450

/-- The function f(x) = sin(ωx + π/4) has exactly one zero in [0, π) -/
def has_one_zero (ω : ℕ) : Prop :=
  ∃! x, x ∈ Set.Icc 0 Real.pi ∧ Real.sin (ω * x + Real.pi / 4) = 0

/-- If f(x) = sin(ωx + π/4) where ω ∈ ℕ has exactly one zero in [0, π), then ω = 1 -/
theorem omega_value (ω : ℕ) (h : has_one_zero ω) : ω = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l344_34450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_is_120_degrees_l344_34424

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 2 = 1

-- Define the foci
noncomputable def F1 : ℝ × ℝ := sorry
noncomputable def F2 : ℝ × ℝ := sorry

-- Define a point on the ellipse
noncomputable def P : ℝ × ℝ := sorry

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_measure_is_120_degrees :
  is_on_ellipse P.1 P.2 →
  distance P F1 = 4 →
  ∃ θ : ℝ, θ = 120 * Real.pi / 180 ∧ 
    Real.cos θ = (distance P F1)^2 + (distance P F2)^2 - (distance F1 F2)^2 / (2 * distance P F1 * distance P F2) := by
  sorry

#check angle_measure_is_120_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_is_120_degrees_l344_34424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l344_34429

-- Define the function f(x) = ln x + 1/x
noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

-- State the theorem
theorem f_monotone_increasing :
  ∀ x y : ℝ, 1 < x → x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l344_34429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_b_l344_34426

def a : Fin 3 → ℝ := ![-2, 2, 0]
def b : Fin 3 → ℝ := ![1, 0, -1]

noncomputable def angle_between (v w : Fin 3 → ℝ) : ℝ :=
  Real.arccos ((v 0 * w 0 + v 1 * w 1 + v 2 * w 2) /
    (Real.sqrt (v 0^2 + v 1^2 + v 2^2) * Real.sqrt (w 0^2 + w 1^2 + w 2^2)))

theorem angle_between_a_b :
  angle_between a b = Real.pi * (2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_b_l344_34426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_l344_34445

-- Define the variables
noncomputable def a : ℝ := Real.log 3 / Real.log 4
noncomputable def b : ℝ := Real.log 3
noncomputable def c : ℝ := Real.sqrt 10

-- State the theorem
theorem order_of_logarithms : a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_l344_34445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l344_34477

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x
def g (a : ℝ) (x : ℝ) : ℝ := (2 - a) * x^3

-- State the theorem
theorem sufficient_but_not_necessary
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  (∀ x y : ℝ, x < y → g a x < g a y) ∧
  ¬ ((∀ x y : ℝ, x < y → g a x < g a y) →
     (∀ x y : ℝ, x < y → f a x > f a y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l344_34477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_married_men_fraction_l344_34421

theorem married_men_fraction 
  (total_people : ℕ) 
  (prob_single_woman : ℚ) 
  (h1 : total_people = 24) 
  (h2 : prob_single_woman = 1/3) :
  (total_people : ℚ) / 4 = (total_people - (total_people / 3 : ℕ)) / 2 := by
  sorry

#check married_men_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_married_men_fraction_l344_34421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_set_properties_l344_34457

-- Define the property of being a "good set"
def is_good_set (A : Set ℚ) : Prop :=
  (0 ∈ A ∧ 1 ∈ A) ∧
  (∀ x y, x ∈ A → y ∈ A → x - y ∈ A) ∧
  (∀ x, x ∈ A → x ≠ 0 → x⁻¹ ∈ A)

theorem good_set_properties :
  (is_good_set (Set.univ : Set ℚ)) ∧
  (∀ A : Set ℚ, is_good_set A → ∀ x y, x ∈ A → y ∈ A → x + y ∈ A) ∧
  (∀ A : Set ℚ, is_good_set A → ∀ x y, x ∈ A → y ∈ A → x * y ∈ A) ∧
  (∀ A : Set ℚ, is_good_set A → ∀ x y, x ∈ A → y ∈ A → x ≠ 0 → y / x ∈ A) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_set_properties_l344_34457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_special_modular_property_l344_34409

theorem characterization_of_special_modular_property (n : ℕ) :
  (∃ a : ℤ, a^(Nat.totient n / 2) ≡ -1 [ZMOD n]) ↔
  (n = 4 ∨ 
   (∃ p k : ℕ, Nat.Prime p ∧ p > 2 ∧ k > 0 ∧ (n = p^k ∨ n = 2 * p^k))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_special_modular_property_l344_34409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_conversion_l344_34407

noncomputable def rectangularToSpherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 ∧ y ≥ 0 then Real.pi + Real.arctan (y / x)
           else if x < 0 ∧ y < 0 then -Real.pi + Real.arctan (y / x)
           else if x = 0 ∧ y > 0 then Real.pi / 2
           else if x = 0 ∧ y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  let φ := Real.arccos (z / ρ)
  (ρ, θ, φ)

theorem rectangular_to_spherical_conversion (x y z : ℝ) :
  let (ρ, θ, φ) := rectangularToSpherical x y z
  x = 1 ∧ y = -2 ∧ z = 2 * Real.sqrt 2 →
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi →
  ρ = Real.sqrt 13 ∧
  θ = Real.pi - Real.arctan 2 ∧
  φ = Real.arccos (2 * Real.sqrt 2 / Real.sqrt 13) := by
  sorry

#check rectangular_to_spherical_conversion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_conversion_l344_34407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l344_34499

theorem f_derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = (1/3) * x^3 - f' 1 * x^2 - x) →
  f' 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l344_34499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_max_product_l344_34466

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := m * y = -x
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x = y + m - 3

-- Define points A and B
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 3)

-- Define the distance function
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem intersection_max_product (m : ℝ) :
  ∀ x y : ℝ, l₁ m x y → l₂ m x y →
  distance (x, y) A * distance (x, y) B ≤ 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_max_product_l344_34466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_arrangements_l344_34449

def total_students : ℕ := 360

def is_valid_arrangement (students_per_row : ℕ) : Bool :=
  students_per_row ≥ 18 &&
  (total_students / students_per_row) ≥ 12 &&
  total_students % students_per_row = 0

def valid_arrangements : List ℕ :=
  (List.range (total_students + 1)).filter is_valid_arrangement

theorem sum_of_valid_arrangements :
  valid_arrangements.sum = 92 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_arrangements_l344_34449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l344_34412

noncomputable def f (x : ℝ) := 2 * Real.sin (x - Real.pi/6)

theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Ioo (Real.pi/6) ((2:ℝ)*Real.pi/3)) := by
  sorry

#check f_increasing_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l344_34412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_point_sum_l344_34439

noncomputable section

-- Define the line equation
def line_equation (x : ℝ) : ℝ := -5/3 * x + 15

-- Define points P and Q
def P : ℝ × ℝ := (9, 0)
def Q : ℝ × ℝ := (0, 15)

-- Define point T
def T (r s : ℝ) : ℝ × ℝ := (r, s)

-- Define the condition that T is on line segment PQ
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ T r s = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2)

-- Define the area ratio condition
def area_ratio_condition (r s : ℝ) : Prop :=
  (1/2 * P.1 * Q.2) = 2 * (1/2 * P.1 * s)

-- Theorem statement
theorem line_segment_point_sum (r s : ℝ) 
  (h1 : T_on_PQ r s) 
  (h2 : area_ratio_condition r s) 
  (h3 : s = line_equation r) : 
  r + s = 12 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_point_sum_l344_34439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_l344_34447

def A : Set ℝ := {x | abs x < 4}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

theorem set_equality : {x : ℝ | x ∈ A ∧ x ∉ A ∩ B} = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_l344_34447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_n_product_greater_than_5000_l344_34488

open Real Nat

theorem smallest_odd_n_product_greater_than_5000 :
  ∃ (n : ℕ), Odd n ∧ 
    (∀ m : ℕ, Odd m → m < n → (2 : ℝ)^((m + 1)^2 / 9) ≤ 5000) ∧
    (2 : ℝ)^((n + 1)^2 / 9) > 5000 ∧
    n = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_n_product_greater_than_5000_l344_34488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_tangent_l344_34483

/-- Given an angle α whose terminal side passes through the point (-1, y) and tan α = 1/2, prove that y = -1/2 -/
theorem terminal_side_tangent (α : ℝ) (y : ℝ) 
  (h1 : ∃ (t : ℝ), t * (-1) = Real.cos α ∧ t * y = Real.sin α) -- terminal side passes through (-1, y)
  (h2 : Real.tan α = 1/2) : 
  y = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_tangent_l344_34483
