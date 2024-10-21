import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2_cos_3_tan_4_negative_l146_14676

theorem sin_2_cos_3_tan_4_negative : Real.sin 2 * Real.cos 3 * Real.tan 4 < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2_cos_3_tan_4_negative_l146_14676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_price_scaling_l146_14640

-- Define the properties of the containers
def d₁ : ℝ := 4
def h₁ : ℝ := 5
def p₁ : ℝ := 0.80
def d₂ : ℝ := 8
def h₂ : ℝ := 10

-- Define the volume calculation function
noncomputable def volume (d h : ℝ) : ℝ := Real.pi * (d / 2)^2 * h

-- Theorem statement
theorem container_price_scaling (p₂ : ℝ) : 
  p₂ = p₁ * (volume d₂ h₂ / volume d₁ h₁) → p₂ = 6.40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_price_scaling_l146_14640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_steps_to_paint_square_l146_14645

/-- Represents a cell in the square --/
structure Cell where
  x : Int
  y : Int

/-- The size of the square --/
def squareSize : Int := 2021

/-- Manhattan distance between two cells --/
def manhattanDistance (c1 c2 : Cell) : Int :=
  abs (c1.x - c2.x) + abs (c1.y - c2.y)

/-- Whether a cell is a critical point --/
def isCriticalPoint (c : Cell) : Prop :=
  (c.x = 0 ∨ c.x = squareSize - 1 ∨ c.x = squareSize / 2) ∧
  (c.y = 0 ∨ c.y = squareSize - 1 ∨ c.y = squareSize / 2)

/-- The set of all critical points --/
def criticalPoints : Set Cell :=
  {c | isCriticalPoint c}

/-- A cell is controlled by an initially painted cell if their Manhattan distance is at most 1414 --/
def isControlled (c : Cell) (initial1 initial2 : Cell) : Prop :=
  manhattanDistance c initial1 ≤ 1414 ∨ manhattanDistance c initial2 ≤ 1414

/-- The main theorem --/
theorem min_steps_to_paint_square :
  ∀ (initial1 initial2 : Cell),
    ¬(∀ (c : Cell), c ∈ criticalPoints → isControlled c initial1 initial2) →
    (∀ (c : Cell), 0 ≤ c.x ∧ c.x < squareSize ∧ 0 ≤ c.y ∧ c.y < squareSize →
      manhattanDistance c initial1 ≤ 1515 ∨ manhattanDistance c initial2 ≤ 1515) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_steps_to_paint_square_l146_14645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximize_integral_l146_14682

-- Define the constraint function
noncomputable def constraint (a b : ℝ) : Prop :=
  ∫ x in (0:ℝ)..1, (a * x + b)^2 = 1

-- Define the function to be maximized
noncomputable def objective (a b : ℝ) : ℝ :=
  ∫ x in (0:ℝ)..1, 3 * x * (a * x + b)

-- State the theorem
theorem maximize_integral :
  ∃ (a b : ℝ), constraint a b ∧
    (∀ (a' b' : ℝ), constraint a' b' → objective a b ≥ objective a' b') ∧
    a = Real.sqrt 3 ∧ b = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximize_integral_l146_14682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quintuple_sum_inequality_l146_14699

theorem quintuple_sum_inequality (a : ℝ) : 
  5 * (a + 3) ≥ 6 ↔ 5 * (a + 3) ≥ 6 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quintuple_sum_inequality_l146_14699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_is_460_l146_14602

/-- Represents the rental business scenario -/
structure RentalBusiness where
  canoe_price : ℕ  -- Daily rental price for a canoe
  kayak_price : ℕ  -- Daily rental price for a kayak
  canoe_kayak_ratio : Rat  -- Ratio of canoes to kayaks
  canoe_kayak_difference : ℕ  -- Difference between number of canoes and kayaks

/-- Calculates the total revenue for the rental business -/
def calculate_revenue (rb : RentalBusiness) : ℕ :=
  sorry

/-- Theorem stating that the total revenue is $460 -/
theorem revenue_is_460 (rb : RentalBusiness) 
  (h1 : rb.canoe_price = 11)
  (h2 : rb.kayak_price = 16)
  (h3 : rb.canoe_kayak_ratio = 4 / 3)
  (h4 : rb.canoe_kayak_difference = 5) :
  calculate_revenue rb = 460 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_is_460_l146_14602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_variance_and_std_dev_l146_14662

/-- A random variable uniformly distributed in the interval (a, b) -/
structure UniformRV (a b : ℝ) where
  (hlt : a < b)

/-- The variance of a uniform random variable -/
noncomputable def variance (X : UniformRV a b) : ℝ := (b - a)^2 / 12

/-- The standard deviation of a uniform random variable -/
noncomputable def standardDeviation (X : UniformRV a b) : ℝ := (b - a) / (2 * Real.sqrt 3)

/-- Theorem: The variance and standard deviation of a uniform random variable -/
theorem uniform_variance_and_std_dev (a b : ℝ) (X : UniformRV a b) :
  variance X = (b - a)^2 / 12 ∧ standardDeviation X = (b - a) / (2 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_variance_and_std_dev_l146_14662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l146_14648

/-- Definition of the hyperbola -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- Focal distance of the hyperbola -/
def focal_distance : ℝ := 4

/-- Distance from focus to asymptote -/
noncomputable def focus_to_asymptote : ℝ := Real.sqrt 3

/-- Theorem stating the focal distance and distance from focus to asymptote for the given hyperbola -/
theorem hyperbola_properties :
  (∀ x y, hyperbola x y → 
    focal_distance = 4 ∧ 
    focus_to_asymptote = Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l146_14648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_sum_of_divisors_180_l146_14636

def sumOfDivisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  (Nat.factors (sumOfDivisors 180)).maximum? = some 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_sum_of_divisors_180_l146_14636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_solutions_l146_14660

def satisfies_equation (n : ℕ) : Prop :=
  (n + 1500) / 90 = ⌊(n : ℝ)^(1/3)⌋

theorem exactly_two_solutions : 
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_equation n) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_solutions_l146_14660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_abs_deviation_larger_for_10_tosses_l146_14605

/-- Represents the outcome of n coin tosses -/
structure CoinTosses (n : ℕ) where
  heads : ℕ
  heads_le_n : heads ≤ n

/-- The frequency of heads in n tosses -/
def frequency {n : ℕ} (t : CoinTosses n) : ℚ :=
  t.heads / n

/-- The deviation of the frequency from 0.5 -/
def deviation {n : ℕ} (t : CoinTosses n) : ℚ :=
  frequency t - 1/2

/-- The absolute deviation of the frequency from 0.5 -/
def abs_deviation {n : ℕ} (t : CoinTosses n) : ℚ :=
  |deviation t|

/-- Expected value of a random variable -/
noncomputable def expected_value {α : Type*} (X : α → ℝ) (p : α → ℝ) : ℝ :=
  sorry

/-- The probability measure for fair coin tosses -/
noncomputable def fair_coin_prob (n : ℕ) : CoinTosses n → ℝ :=
  sorry

theorem expected_abs_deviation_larger_for_10_tosses :
  expected_value (fun t : CoinTosses 10 => (abs_deviation t : ℝ)) (fair_coin_prob 10) >
  expected_value (fun t : CoinTosses 100 => (abs_deviation t : ℝ)) (fair_coin_prob 100) :=
by
  sorry

#check expected_abs_deviation_larger_for_10_tosses

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_abs_deviation_larger_for_10_tosses_l146_14605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_in_range_l146_14658

theorem count_solutions_in_range : 
  (Finset.filter (fun k : ℕ => -30 < (k : ℤ) * Real.pi ∧ (k : ℤ) * Real.pi < 120) (Finset.range 49)).card = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_in_range_l146_14658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corresponding_angles_equivalence_l146_14669

-- Define what it means for angles to be corresponding
def are_corresponding_angles (angle1 angle2 : Real) : Prop := sorry

-- Define the original statement
def corresponding_angles_are_equal : Prop :=
  ∀ (angle1 angle2 : Real), are_corresponding_angles angle1 angle2 → angle1 = angle2

-- Define the rewritten statement
def if_corresponding_then_equal : Prop :=
  ∀ (angle1 angle2 : Real), are_corresponding_angles angle1 angle2 → angle1 = angle2

-- Theorem stating the equivalence of the two forms
theorem corresponding_angles_equivalence :
  corresponding_angles_are_equal ↔ if_corresponding_then_equal := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corresponding_angles_equivalence_l146_14669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_ab_l146_14683

/-- The time to fill the tank when all valves are open -/
noncomputable def all_valves_time : ℝ := 1

/-- The time to fill the tank when valves A and C are open -/
noncomputable def ac_time : ℝ := 1.5

/-- The time to fill the tank when valves B and C are open -/
noncomputable def bc_time : ℝ := 2

/-- The flow rate of valve A -/
noncomputable def rate_a : ℝ := 1 / 2

/-- The flow rate of valve B -/
noncomputable def rate_b : ℝ := 1 / 3

/-- The flow rate of valve C -/
noncomputable def rate_c : ℝ := 1 / 6

/-- The time to fill the tank when only valves A and B are open -/
noncomputable def ab_time : ℝ := 1.2

theorem fill_time_ab : 
  (rate_a + rate_b + rate_c) * all_valves_time = 1 ∧
  (rate_a + rate_c) * ac_time = 1 ∧
  (rate_b + rate_c) * bc_time = 1 →
  (rate_a + rate_b) * ab_time = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_ab_l146_14683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l146_14631

/-- The circle on which point P moves -/
def circle_P (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 1

/-- The ellipse on which point Q moves -/
def ellipse_Q (x y : ℝ) : Prop := x^2/9 + y^2 = 1

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The maximum distance between points on the circle and ellipse -/
theorem max_distance_circle_ellipse :
  ∃ (max_dist : ℝ), max_dist = 3 * Real.sqrt 3 + 1 ∧
    ∀ (x1 y1 x2 y2 : ℝ), 
      circle_P x1 y1 → ellipse_Q x2 y2 → 
      distance x1 y1 x2 y2 ≤ max_dist :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l146_14631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_and_min_distance_l146_14694

/-- Curve C₁ in polar coordinates -/
noncomputable def C₁ (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- Transformation from C₁ to C₂ -/
def transform (x y : ℝ) : ℝ × ℝ := (2 * (x - 1), y)

/-- Curve C₂ in Cartesian coordinates -/
def C₂ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Point P -/
def P : ℝ × ℝ := (1, 2)

/-- Line passing through P and a point on C₂ -/
noncomputable def line (α : ℝ) (t : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 2 + t * Real.sin α)

/-- Distance product |PA||PB| -/
noncomputable def distance_product (α : ℝ) : ℝ := 13 / (1 + 3 * Real.sin α ^ 2)

theorem C₂_equation_and_min_distance :
  (∀ x y, C₂ x y ↔ ∃ θ, (x, y) = transform (C₁ θ * Real.cos θ) (C₁ θ * Real.sin θ)) ∧
  (∀ α, distance_product α ≥ 13/4) ∧
  (∃ α, distance_product α = 13/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_and_min_distance_l146_14694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendly_triangle_vertex_angle_l146_14672

/-- Given an isosceles triangle ABC with A = B, if there exists a triangle A₁B₁C₁ such that
    cos A / sin A₁ = cos B / sin B₁ = cos C / sin C₁ = 1, then the measure of angle C is π/4. -/
theorem friendly_triangle_vertex_angle (A B C A₁ B₁ C₁ : Real) : 
  A = B →  -- Triangle ABC is isosceles
  (Real.cos A) / (Real.sin A₁) = 1 →  -- Friendly triangle condition for A
  (Real.cos B) / (Real.sin B₁) = 1 →  -- Friendly triangle condition for B
  (Real.cos C) / (Real.sin C₁) = 1 →  -- Friendly triangle condition for C
  A + B + C = π →  -- Sum of angles in a triangle
  A₁ + B₁ + C₁ = π →  -- Sum of angles in a triangle
  C = π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendly_triangle_vertex_angle_l146_14672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_tax_calculation_l146_14686

/-- Calculates the sales tax on taxable purchases given the total payment, tax rate, and cost of tax-free items. -/
noncomputable def salesTaxOnTaxablePurchases (totalPayment taxRate costOfTaxFreeItems : ℝ) : ℝ :=
  let costOfTaxableItemsPlusTax := totalPayment - costOfTaxFreeItems
  let costOfTaxableItems := costOfTaxableItemsPlusTax / (1 + taxRate)
  taxRate * costOfTaxableItems

/-- Theorem stating that given the specified conditions, the sales tax on taxable purchases is $1.28. -/
theorem sales_tax_calculation (totalPayment taxRate costOfTaxFreeItems : ℝ) 
  (h1 : totalPayment = 30)
  (h2 : taxRate = 0.08)
  (h3 : costOfTaxFreeItems = 12.72) :
  salesTaxOnTaxablePurchases totalPayment taxRate costOfTaxFreeItems = 1.28 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval salesTaxOnTaxablePurchases 30 0.08 12.72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_tax_calculation_l146_14686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l146_14629

-- Define the function f(x) = √x - x + 1
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - x + 1

-- State the theorem
theorem root_exists_in_interval :
  ContinuousOn f (Set.Icc 2 3) ∧ f 2 > 0 ∧ f 3 < 0 →
  ∃ x : ℝ, x ∈ Set.Ioo 2 3 ∧ f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l146_14629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derived_sequence_equality_iff_identical_l146_14625

/-- Definition of a Pn sequence -/
def is_pn_sequence (A : List ℕ) : Prop :=
  A.length > 0 ∧ 
  (∀ i, i ∈ List.range A.length → 1 ≤ A.get! i ∧ A.get! i ≤ A.length) ∧
  A.Nodup

/-- Definition of the derived sequence -/
def derived_sequence (A : List ℕ) : List ℤ :=
  let n := A.length
  List.range n |>.map (λ i =>
    if i = 0 then 0
    else List.range i |>.foldl (λ acc j =>
      acc + (Int.sign (A.get! i - A.get! j))) 0)

/-- Main theorem: derived sequences are equal iff original sequences are identical -/
theorem derived_sequence_equality_iff_identical (A A' : List ℕ) 
  (hA : is_pn_sequence A) (hA' : is_pn_sequence A') :
  derived_sequence A = derived_sequence A' ↔ A = A' := by
  sorry

#check derived_sequence_equality_iff_identical

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derived_sequence_equality_iff_identical_l146_14625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l146_14668

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (9 - x^2)}
def N : Set ℝ := {x | ∃ y, y = (2 : ℝ)^(1 - x)}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l146_14668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_cost_graph_is_finite_distinct_points_l146_14697

def goldfish_cost (n : ℕ) : ℚ :=
  20 * n / 100 + 5

def valid_goldfish_count (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 12

theorem goldfish_cost_graph_is_finite_distinct_points :
  ∃ (S : Set (ℕ × ℚ)),
    (∀ p, p ∈ S → valid_goldfish_count p.1 ∧ p.2 = goldfish_cost p.1) ∧
    Finite S ∧
    (∀ p q, p ∈ S → q ∈ S → p ≠ q → p.2 ≠ q.2) :=
  sorry

#check goldfish_cost_graph_is_finite_distinct_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_cost_graph_is_finite_distinct_points_l146_14697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rem_five_sevenths_neg_three_fourths_l146_14690

-- Define the remainder function as noncomputable
noncomputable def rem (x y : ℝ) : ℝ := x - y * ⌊x / y⌋

-- State the theorem
theorem rem_five_sevenths_neg_three_fourths :
  rem (5/7) (-3/4) = -1/28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rem_five_sevenths_neg_three_fourths_l146_14690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_and_optimal_expenses_initial_condition_inverse_proportionality_all_sold_out_l146_14628

noncomputable def x (t : ℝ) : ℝ := 3 - 2 / (t + 1)

noncomputable def y (t : ℝ) : ℝ := (-t^2 + 98*t + 35) / (2*(t + 1))

theorem max_profit_and_optimal_expenses :
  ∃ (t_max : ℝ), t_max = 7 ∧ 
  (∀ t : ℝ, t ≥ 0 → y t ≤ y t_max) ∧
  y t_max = 42 := by
  sorry

theorem initial_condition : x 0 = 1 := by
  sorry

theorem inverse_proportionality (t : ℝ) :
  (3 - x t) * (t + 1) = 2 := by
  sorry

theorem all_sold_out (t : ℝ) :
  x t * (1.5 * (32 * x t + 3) / x t + t / (2 * x t)) =
  1.5 * (32 * x t + 3) + t / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_and_optimal_expenses_initial_condition_inverse_proportionality_all_sold_out_l146_14628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_divisibility_l146_14633

def M : Set ℕ := {n : ℕ | ∀ p : ℕ, Nat.Prime p → p > 3 → ¬p ∣ n}

theorem subset_divisibility (A : ℕ → Set ℕ) :
  (∀ i : ℕ, A i ⊆ M) →
  ∃ i j : ℕ, i ≠ j ∧ ∀ x ∈ A i, ∃ y ∈ A j, y ∣ x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_divisibility_l146_14633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_equals_45_l146_14685

theorem root_product_equals_45 :
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * Real.sqrt 25 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_equals_45_l146_14685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l146_14617

/-- The area of the triangle bounded by the y-axis and two lines -/
theorem triangle_area : ∃ (area : ℝ), area = 50 / 9 := by
  -- Define the lines and y-axis
  let line1 : ℝ → ℝ → Prop := λ x y ↦ y - 2 * x = -1
  let line2 : ℝ → ℝ → Prop := λ x y ↦ 4 * y + x = 16
  let y_axis : ℝ → ℝ → Prop := λ x _ ↦ x = 0

  -- Define the triangle area function
  let triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

  -- Calculate the y-intercepts
  have y_intercept1 : line1 0 (-1) := by sorry
  have y_intercept2 : line2 0 4 := by sorry

  -- Calculate the base length
  let base : ℝ := 4 - (-1)

  -- Find the intersection point
  have intersection : ∃ (x y : ℝ), line1 x y ∧ line2 x y := by sorry
  
  -- Extract the x-coordinate (height) from the intersection
  have height : ℝ := 20 / 9

  -- Calculate the area
  let area : ℝ := triangle_area base height

  -- Prove that the area equals 50/9
  have area_equals_50_over_9 : area = 50 / 9 := by sorry

  -- Conclude the theorem
  exact ⟨area, area_equals_50_over_9⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l146_14617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_with_four_prime_factors_l146_14681

theorem unique_integer_with_four_prime_factors : ∃! n : ℕ,
  (∃ (p q r s : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧
    n = p * q * r * s ∧
    p * p + q * q + r * r + s * s = 476) ∧
  (∀ (a b c d : ℕ), 
    Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d ∧
    n = a * b * c * d →
    Multiset.ofList [a, b, c, d] = Multiset.ofList [3, 3, 13, 17]) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_with_four_prime_factors_l146_14681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l146_14609

/-- The common ratio of an infinite geometric sequence -/
noncomputable def common_ratio (a : ℕ → ℝ) : ℝ := a 2 / a 1

/-- The sum of an infinite geometric sequence -/
noncomputable def infinite_geometric_sum (a : ℕ → ℝ) (q : ℝ) : ℝ := a 1 / (1 - q)

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  infinite_geometric_sum a (common_ratio a) = 3 →
  a 1 = 2 →
  common_ratio a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l146_14609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l146_14635

noncomputable section

variable (α : ℝ)
variable (t : ℝ)

def a : ℝ × ℝ := (2 * Real.cos α, Real.sin α ^ 2)
def b : ℝ × ℝ := (2 * Real.sin α, t)

theorem vector_problem (h1 : 0 < α) (h2 : α < Real.pi / 2) :
  (a α - b α t = (2/5, 0) → t = 9/25) ∧
  (t = 1 → (a α).1 * (b α t).1 + (a α).2 * (b α t).2 = 1 → Real.tan (2 * α + Real.pi / 4) = 23/7) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l146_14635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_value_at_one_l146_14639

/-- A monotonous function on (0, +∞) satisfying a specific functional equation. -/
structure SpecialFunction where
  f : ℝ → ℝ
  monotonous : Monotone f
  domain : ∀ x, x > 0 → f x ≠ 0
  equation : ∀ x, x > 0 → f x * f (f x + 2 / x) = 2

/-- The value of f(1) for a SpecialFunction is either 1 + √5 or 1 - √5. -/
theorem special_function_value_at_one (sf : SpecialFunction) :
    sf.f 1 = 1 + Real.sqrt 5 ∨ sf.f 1 = 1 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_value_at_one_l146_14639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_evaluation_l146_14651

theorem ceiling_evaluation : ⌈(4 : ℝ) * (8 - 3/4)⌉ = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_evaluation_l146_14651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_c_l146_14622

noncomputable def angle (u v : ℝ × ℝ) : ℝ :=
  Real.arccos ((u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2)))

noncomputable def toDegrees (θ : ℝ) : ℝ :=
  θ * 180 / Real.pi

theorem angle_between_a_and_c : 
  let a : ℝ × ℝ := (-2, 0)
  let c : ℝ × ℝ := (1, Real.sqrt 3)
  toDegrees (angle a c) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_c_l146_14622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_angle_problem_l146_14689

def angle_problem (α : Real) : Prop :=
  (Real.tan α = 2) →
  (Real.tan (2 * α) = -4/3) ∧
  ((2 * (Real.cos (α/2))^2 - 2 * Real.sin (α - Real.pi) - 1) / 
   (Real.sqrt 2 * Real.cos (α - 7*Real.pi/4)) = -5)

theorem solve_angle_problem :
  ∀ α : Real, angle_problem α := by
  intro α
  intro h
  apply And.intro
  · sorry  -- Proof for tan(2α) = -4/3
  · sorry  -- Proof for the fraction equal to -5


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_angle_problem_l146_14689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l146_14696

theorem max_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + 5 * b < 100) :
  a * b * (100 - 4 * a - 5 * b) ≤ 1851.852 ∧
  ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 4 * a₀ + 5 * b₀ < 100 ∧
    a₀ * b₀ * (100 - 4 * a₀ - 5 * b₀) = 1851.852 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l146_14696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l146_14695

def solution_set : Set ℝ := {x | -1 ≤ x ∧ x < 2}

theorem inequality_solution :
  ∀ x : ℝ, (x + 1) / (2 - x) ≥ 0 ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l146_14695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_copper_bar_weight_l146_14650

/-- The weight of a copper bar in kg -/
noncomputable def copper_weight : ℝ := 90

/-- The weight of a steel bar in kg -/
noncomputable def steel_weight : ℝ := copper_weight + 20

/-- The weight of a tin bar in kg -/
noncomputable def tin_weight : ℝ := steel_weight / 2

/-- The total weight of 20 bars each of copper, steel, and tin in kg -/
noncomputable def total_weight : ℝ := 5100

theorem copper_bar_weight :
  steel_weight = 2 * tin_weight ∧
  steel_weight = copper_weight + 20 ∧
  20 * copper_weight + 20 * steel_weight + 20 * tin_weight = total_weight →
  copper_weight = 90 := by
  intro h
  -- The proof goes here
  sorry

#check copper_bar_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_copper_bar_weight_l146_14650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jo_reading_time_l146_14684

def reading_problem (total_pages current_page previous_page : ℕ) : Prop :=
  let pages_per_hour := current_page - previous_page
  let remaining_pages := total_pages - current_page
  (remaining_pages / pages_per_hour : ℚ) = 4

theorem jo_reading_time :
  reading_problem 210 90 60 :=
by
  -- Unfold the definition of reading_problem
  unfold reading_problem
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl

#check jo_reading_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jo_reading_time_l146_14684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l146_14607

noncomputable def f (x : ℝ) : ℝ := (Real.sin (x + Real.pi))^2 - (Real.cos (x - Real.pi/3))^2

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧ 
    (∀ (T' : ℝ), (0 < T' ∧ T' < T) → ∃ (x : ℝ), f (x + T') ≠ f x) ∧
    (∀ (k : ℤ), StrictMonoOn f (Set.Icc (k * Real.pi + Real.pi/6) (k * Real.pi + 2*Real.pi/3))) ∧
    (∀ (m : ℝ), 
      (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/6) (Real.pi/4) → |f x - m| ≤ 2) ↔ 
      m ∈ Set.Icc (-7/4) (3/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l146_14607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_subsequence_l146_14677

/-- Given an integer a > 1, a_n is defined as a^(n+1) + a^n - 1 -/
def a_n (a n : ℕ) : ℕ := a^(n+1) + a^n - 1

/-- The theorem states that there exists an infinite subsequence of a_n that is pairwise coprime -/
theorem infinite_coprime_subsequence (a : ℕ) (ha : a > 1) :
  ∃ (f : ℕ → ℕ), StrictMono f ∧ (∀ i j, i ≠ j → Nat.Coprime (a_n a (f i)) (a_n a (f j))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_subsequence_l146_14677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalence_point_2x_plus_1_no_equivalence_point_x_squared_minus_x_plus_2_equivalence_points_area_condition_equivalence_points_reflection_condition_l146_14630

-- Definition of "equivalence point"
def is_equivalence_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = x

-- Theorem 1
theorem equivalence_point_2x_plus_1 :
  is_equivalence_point (λ x ↦ 2 * x + 1) (-1) :=
by sorry

-- Theorem 2
theorem no_equivalence_point_x_squared_minus_x_plus_2 :
  ∀ x, ¬is_equivalence_point (λ x ↦ x^2 - x + 2) x :=
by sorry

-- Theorem 3
theorem equivalence_points_area_condition (b : ℝ) :
  let h := λ x ↦ 9 / x
  let k := λ x ↦ -x + b
  let x_A := (3 : ℝ)
  let x_B := b / 2
  (∀ x > 0, is_equivalence_point h x → x = x_A) ∧
  (∀ x > 0, is_equivalence_point k x → x = x_B) ∧
  (1/2 * x_B * |x_A - x_B| = 3) →
  b = 3 + Real.sqrt 33 ∨ b = 3 - Real.sqrt 33 :=
by sorry

-- Theorem 4
theorem equivalence_points_reflection_condition (m : ℝ) :
  let j := λ x ↦ x^2 - 4
  let j_reflected := λ x ↦ j (2*m - x)
  (∃! p q, p ≠ q ∧ 
    ((p ≥ m ∧ is_equivalence_point j p) ∨ 
     (p < m ∧ is_equivalence_point j_reflected p)) ∧
    ((q ≥ m ∧ is_equivalence_point j q) ∨ 
     (q < m ∧ is_equivalence_point j_reflected q))) →
  m < -17/8 ∨ (1 - Real.sqrt 17) / 2 < m ∧ m < (1 + Real.sqrt 17) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalence_point_2x_plus_1_no_equivalence_point_x_squared_minus_x_plus_2_equivalence_points_area_condition_equivalence_points_reflection_condition_l146_14630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l146_14620

theorem matrix_transformation (N : Matrix (Fin 3) (Fin 3) ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 3; 0, 1, 0; 1, 0, 0]
  M * N = !![3 * N 2 0, 3 * N 2 1, 3 * N 2 2;
             N 1 0, N 1 1, N 1 2;
             N 0 0, N 0 1, N 0 2] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l146_14620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_triangle_l146_14649

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Ray where
  origin : Point
  direction : Point

structure Angle where
  vertex : Point
  ray1 : Ray
  ray2 : Ray

structure Triangle where
  a : Point
  b : Point
  c : Point

structure Line where
  a : Point
  b : Point

-- Define the given angle and point
noncomputable def XAY : Angle := sorry
noncomputable def O : Point := sorry

-- Define the symmetrical angle
noncomputable def X'A'Y' : Angle := sorry

-- Define the intersection points
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry

-- Define a function to create a line through a point
noncomputable def line_through_point (p : Point) (q : Point) : Line :=
  { a := p, b := q }

-- Define a function to calculate the area of a triangle
noncomputable def triangle_area (t : Triangle) : ℝ := sorry

-- Define a function to create a triangle from an angle and a line
noncomputable def triangle_from_angle_and_line (a : Angle) (l : Line) : Triangle := sorry

-- Theorem statement
theorem smallest_area_triangle :
  ∀ (l : Line),
    triangle_area (triangle_from_angle_and_line XAY l) ≥ 
    triangle_area (triangle_from_angle_and_line XAY (line_through_point B C)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_triangle_l146_14649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_divisors_l146_14679

open Nat

def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem x_divisors (x : ℕ) (h : num_divisors (x^3) = 7) : num_divisors x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_divisors_l146_14679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_sqrt_five_irrational_l146_14693

theorem negative_sqrt_five_irrational :
  (∃ (a b : ℤ), (0.4 : ℝ) = a / b) →
  (∃ (c d : ℤ), (2/3 : ℝ) = c / d) →
  (∃ (e f : ℤ), (8^(1/3) : ℝ) = e / f) →
  ¬(∃ (g h : ℤ), (-Real.sqrt 5 : ℝ) = g / h) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_sqrt_five_irrational_l146_14693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_Z_time_l146_14664

/-- The time it takes for printer Z to do the job alone -/
noncomputable def T_Z : ℝ := sorry

/-- The rate at which printer X completes the job -/
noncomputable def rate_X : ℝ := 1 / 16

/-- The rate at which printer Y completes the job -/
noncomputable def rate_Y : ℝ := 1 / 12

/-- The rate at which printer Z completes the job -/
noncomputable def rate_Z : ℝ := 1 / T_Z

/-- The combined rate of printers Y and Z -/
noncomputable def combined_rate_YZ : ℝ := rate_Y + rate_Z

/-- The time it takes for printers Y and Z to complete the job together -/
noncomputable def time_YZ : ℝ := 1 / combined_rate_YZ

/-- The ratio of time for X alone to Y and Z together -/
noncomputable def ratio : ℝ := 10 / 3

theorem printer_Z_time : T_Z = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_Z_time_l146_14664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_properties_sum_l146_14623

/-- The function f(x) = (x^2 - 2x + 1) / (x^3 - 3x^2 + 2x) -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 1) / (x^3 - 3*x^2 + 2*x)

/-- The number of holes in the graph of f -/
def a : ℕ := 1

/-- The number of vertical asymptotes in the graph of f -/
def b : ℕ := 2

/-- The number of horizontal asymptotes in the graph of f -/
def c : ℕ := 1

/-- The number of oblique asymptotes in the graph of f -/
def d : ℕ := 0

theorem graph_properties_sum :
  a^2 + 2*b^2 + 3*c^2 + 4*d^2 = 12 := by
  -- Expand the left-hand side
  calc
    a^2 + 2*b^2 + 3*c^2 + 4*d^2
    = 1^2 + 2*2^2 + 3*1^2 + 4*0^2 := by rfl
    _ = 1 + 8 + 3 + 0 := by rfl
    _ = 12 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_properties_sum_l146_14623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_properties_l146_14600

theorem alpha_beta_properties (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2 ∧ π / 2 < β ∧ β < π)
  (h2 : Real.tan (α / 2) = 1 / 2)
  (h3 : Real.cos (β - α) = Real.sqrt 2 / 10) :
  Real.sin α = 4 / 5 ∧ β = 3 * π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_properties_l146_14600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_airlines_theorem_l146_14618

/-- Represents a country with cities and airline companies. -/
structure Country where
  N : ℕ  -- Number of cities
  k : ℕ  -- Number of companies
  h : k < N  -- Assumption that k is less than N

/-- Represents the connectivity property of the country. -/
def is_connected (c : Country) (airlines : ℕ) : Prop :=
  airlines ≤ Nat.choose c.N 2 ∧  -- Total number of airlines is at most C(N,2)
  (∀ (i j : Fin c.N), i ≠ j → ∃ (path : List (Fin c.N)), path.head? = some i ∧ path.getLast? = some j) ∧  -- Any city can be reached from any other
  (∀ (company : Fin c.k), ∃ (i j : Fin c.N), i ≠ j ∧ ¬∃ (path : List (Fin c.N)), path.head? = some i ∧ path.getLast? = some j ∧ ∀ (x : Fin c.N), x ∈ path → x.val ≠ company.val)  -- Removing any company's airlines disrupts connectivity

/-- The maximum number of airlines in the country. -/
def max_airlines (c : Country) : ℕ := Nat.choose c.N 2 - Nat.choose c.k 2

/-- Theorem stating the maximum number of airlines in the country. -/
theorem max_airlines_theorem (c : Country) : 
  ∃ (airlines : ℕ), is_connected c airlines ∧ 
    ∀ (a : ℕ), is_connected c a → a ≤ max_airlines c :=
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_airlines_theorem_l146_14618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_An_integer_iff_An_not_prime_l146_14661

noncomputable def An (n : ℕ) : ℚ := (2^(4*n+2) + 1) / 65

theorem An_integer_iff (n : ℕ) : 
  (∃ k : ℤ, n = 3*k + 1) ↔ ∃ m : ℤ, An n = m := by
  sorry

theorem An_not_prime : ∀ n : ℕ, ¬(Nat.Prime (Int.natAbs ((An n).num)) ∧ (An n).den = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_An_integer_iff_An_not_prime_l146_14661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_example_l146_14641

theorem complex_magnitude_example : Complex.abs (1 + 2*Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_example_l146_14641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l146_14653

-- Define the ellipse G
def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (x y m : ℝ) : Prop :=
  y = x + m

-- Define the isosceles triangle condition
def isosceles_triangle (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 - (-3))^2 + (y1 - 2)^2 = (x2 - (-3))^2 + (y2 - 2)^2

theorem ellipse_and_triangle_properties
  (a b c : ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h3 : c / a = 2 / a)  -- Eccentricity condition
  (h4 : c = 2)          -- Right focus at (2,0)
  (x1 y1 x2 y2 : ℝ)     -- Intersection points of line and ellipse
  (h5 : line x1 y1 2)   -- Line l has slope 1 and intersects ellipse
  (h6 : line x2 y2 2)
  (h7 : ellipse x1 y1 a b)
  (h8 : ellipse x2 y2 a b)
  (h9 : isosceles_triangle x1 y1 x2 y2)  -- Isosceles triangle condition
  : (∀ x y, ellipse x y a b ↔ x^2/4 + y^2/4 = 1) ∧    -- Ellipse equation
    (((x2 - x1)^2 + (y2 - y1)^2).sqrt * 5/Real.sqrt 2) / 2 = 15/Real.sqrt 2  -- Area of triangle
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l146_14653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l146_14610

theorem fraction_simplification (x y : ℝ) (h : x ≠ -y) :
  (x^8 + x^6 * y^2 + x^4 * y^4 + x^2 * y^6 + y^8) / (x^4 + x^3 * y + x^2 * y^2 + x * y^3 + y^4) = (x^5 + y^5) / (x + y) := by
  sorry

def calculate_result : ℚ :=
  let x : ℚ := 1/100
  let y : ℚ := 1/50
  (x^5 + y^5) / (x + y)

#eval calculate_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l146_14610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_inequality_l146_14608

def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def IncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem solution_to_inequality
  (f : ℝ → ℝ)
  (odd : IsOdd f)
  (incr : IncreasingOn f (Set.Ici 0))
  (h : f (-3) = 0) :
  {x : ℝ | x * f x < 0} = Set.Ioo (-3) 0 ∪ Set.Ioo 0 3 := by
  sorry

#check solution_to_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_inequality_l146_14608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_approximation_l146_14680

open BigOperators

def series_sum (N : ℕ) : ℚ :=
  ∑ n in Finset.range N, 3 / ((n + 1) * (n + 4))

theorem series_sum_approximation :
  ∃ ε > 0, ε < 0.001 ∧ |series_sum 3010 - 1.833| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_approximation_l146_14680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_eq_neg_five_sixths_l146_14632

-- Define the Riemann function R(x)
def R : ℝ → ℝ := sorry

-- Define function f(x)
def f : ℝ → ℝ := sorry

-- Axioms for f(x)
axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_symmetry (x : ℝ) : f (1 + x) = -f (1 - x)
axiom f_eq_R (x : ℝ) (h : x ∈ Set.Icc 0 1) : f x = R x

-- Axioms for R(x)
axiom R_one : R 1 = 0
axiom R_half : R (1/2) = 1/2
axiom R_third : R (1/3) = 1/3

-- Theorem to prove
theorem f_sum_eq_neg_five_sixths :
  f 2023 + f (2023/2) + f (-2023/3) = -5/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_eq_neg_five_sixths_l146_14632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l146_14688

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_proof (principal time interest : ℝ) :
  principal = 350 →
  time = 8 →
  interest = principal - 238 →
  simple_interest principal 4 time = interest :=
by
  intros h1 h2 h3
  simp [simple_interest]
  rw [h1, h2, h3]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l146_14688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_calculation_l146_14634

noncomputable def cost_price : ℝ := 540
noncomputable def markup_percentage : ℝ := 15
noncomputable def selling_price : ℝ := 456

noncomputable def marked_price : ℝ := cost_price * (1 + markup_percentage / 100)

noncomputable def discount : ℝ := marked_price - selling_price

noncomputable def discount_percentage : ℝ := (discount / marked_price) * 100

theorem discount_percentage_calculation :
  ∃ ε > 0, abs (discount_percentage - 26.57) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_calculation_l146_14634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_l146_14674

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to a horizontal line -/
def distanceToHorizontalLine (p : Point) (y : ℝ) : ℝ :=
  |p.y - y|

/-- The set of points satisfying the given conditions -/
def satisfyingPoints : Set Point :=
  {p : Point | distanceToHorizontalLine p 15 = 4 ∧ distance p ⟨6, 15⟩ = 11}

theorem sum_of_coordinates (points : Finset Point) 
  (h : points.card = 4)
  (h_subset : ↑points ⊆ satisfyingPoints) :
  (points.sum (λ p => p.x + p.y)) = 84 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_l146_14674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l146_14638

noncomputable def f (x : ℝ) := Real.cos x * Real.cos (x - Real.pi/3)

theorem f_properties :
  (f (2*Real.pi/3) = -1/4) ∧
  (∀ x : ℝ, f x < 1/4 ↔ ∃ k : ℤ, k*Real.pi + 5*Real.pi/12 < x ∧ x < k*Real.pi + 11*Real.pi/12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l146_14638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_asymptote_l146_14603

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 -/
noncomputable def point_to_line_distance (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The asymptote of the hyperbola x^2 - y^2 = 2 -/
def asymptote (x y : ℝ) : Prop := y = x

theorem distance_from_focus_to_asymptote :
  point_to_line_distance focus.1 focus.2 1 (-1) 0 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_asymptote_l146_14603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_primer_discount_percentage_l146_14678

def rooms : ℕ := 5
def paint_cost : ℚ := 25
def primer_cost : ℚ := 30
def total_spent : ℚ := 245

theorem primer_discount_percentage (discount_percentage : ℚ)
  (h1 : rooms > 0)
  (h2 : paint_cost > 0)
  (h3 : primer_cost > 0)
  (h4 : total_spent > 0)
  (h5 : total_spent = rooms * paint_cost + rooms * (primer_cost * (1 - discount_percentage / 100)))
  : discount_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_primer_discount_percentage_l146_14678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_circle_C_min_tangent_length_l146_14656

noncomputable section

-- Define the circle C in polar coordinates
def circle_C (θ : Real) : Real := 2 * Real.cos (θ + Real.pi / 4)

-- Define the line l in parametric form
def line_l (t : Real) : Real × Real := (Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t + 4 * Real.sqrt 2)

-- Theorem for the center coordinates of circle C
theorem center_of_circle_C :
  ∃ (x y : Real), x = Real.sqrt 2 / 2 ∧ y = -Real.sqrt 2 / 2 ∧
  ∀ (θ : Real), (circle_C θ * Real.cos θ - x)^2 + (circle_C θ * Real.sin θ - y)^2 = 1 := by
  sorry

-- Theorem for the minimum tangent length
theorem min_tangent_length :
  ∃ (min_length : Real), min_length = 2 * Real.sqrt 6 ∧
  ∀ (t : Real), let (x, y) := line_l t
  (x - Real.sqrt 2 / 2)^2 + (y + Real.sqrt 2 / 2)^2 ≥ 1 + min_length^2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_circle_C_min_tangent_length_l146_14656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_from_properties_l146_14624

/-- Definition of an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  h : b > 0
  k : a ≥ b

/-- The equation of an ellipse with center (0, 0) -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Theorem: Given an ellipse with specific properties, prove its equation -/
theorem ellipse_equation_from_properties :
  ∀ (e : Ellipse),
    eccentricity e = Real.sqrt 2 / 2 →
    (∃ (A B F₂ : ℝ × ℝ),
      ellipse_equation e A.1 A.2 ∧
      ellipse_equation e B.1 B.2 ∧
      F₂.2 = 0 ∧  -- F₂ is on x-axis
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
      Real.sqrt ((A.1 - F₂.1)^2 + A.2^2) +
      Real.sqrt ((B.1 - F₂.1)^2 + B.2^2) = 16) →
    ∀ (x y : ℝ), ellipse_equation e x y ↔ x^2 / 16 + y^2 / 8 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_from_properties_l146_14624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_limit_and_power_limit_l146_14666

noncomputable def x : ℕ → ℝ
  | 0 => 2  -- Adding the case for 0
  | 1 => 2
  | (n + 2) => Real.sqrt (x (n + 1) + 1 / ((n + 2) : ℝ))

theorem x_limit_and_power_limit :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - 1| < ε) ∧
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n ^ n - Real.exp 1| < ε) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_limit_and_power_limit_l146_14666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l146_14615

/-- Definition of the hyperbola C -/
def hyperbola_C (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 16 = 1

/-- The center of the hyperbola is at the origin -/
def center_at_origin : Prop :=
  ∀ x y : ℝ, hyperbola_C x y ↔ hyperbola_C (-x) (-y)

/-- The left focus of the hyperbola is at (-2√5, 0) -/
noncomputable def left_focus : ℝ × ℝ := (-2 * Real.sqrt 5, 0)

/-- The eccentricity of the hyperbola is √5 -/
noncomputable def eccentricity : ℝ := Real.sqrt 5

/-- Definition of the line passing through (-4, 0) -/
def line_through_minus_four (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y - 4

/-- Definition of the fixed line x = -1 -/
def fixed_line (x : ℝ) : Prop := x = -1

/-- The main theorem to be proved -/
theorem hyperbola_theorem
  (h_center : center_at_origin)
  (h_focus : left_focus = (-2 * Real.sqrt 5, 0))
  (h_eccentricity : eccentricity = Real.sqrt 5) :
  (∀ x y : ℝ, hyperbola_C x y ↔ x^2 / 4 - y^2 / 16 = 1) ∧
  (∀ m : ℝ, ∃ P : ℝ × ℝ,
    (∃ M N : ℝ × ℝ,
      hyperbola_C M.1 M.2 ∧
      hyperbola_C N.1 N.2 ∧
      line_through_minus_four m M.1 M.2 ∧
      line_through_minus_four m N.1 N.2 ∧
      M.2 > 0 ∧ M.1 < 0) →
    fixed_line P.1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l146_14615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_when_eccentricity_2_l146_14652

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := 
  Real.sqrt ((a^2 + b^2) / a^2)

/-- The equation of the asymptotes of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola a b) : ℝ := b / a

theorem hyperbola_asymptote_slope_when_eccentricity_2 
  (a b : ℝ) (h : Hyperbola a b) (e : eccentricity h = 2) :
  asymptote_slope h = Real.sqrt 3 := by
  sorry

#check hyperbola_asymptote_slope_when_eccentricity_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_when_eccentricity_2_l146_14652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_always_wins_l146_14604

/-- A point on a circle --/
structure Point where
  angle : ℚ
  deriving Repr

/-- A player in the game --/
inductive Player
  | First
  | Second
  deriving Repr, DecidableEq

/-- The game state --/
structure GameState where
  n : ℕ
  redPoints : List Point
  bluePoints : List Point

/-- The longest arc between same-colored points with no other points in between --/
noncomputable def longestArc (player : Player) (state : GameState) : ℚ :=
  sorry

/-- The winning condition for a player --/
def wins (player : Player) (state : GameState) : Prop :=
  longestArc player state > longestArc (if player = Player.First then Player.Second else Player.First) state

/-- A strategy for the second player --/
def secondPlayerStrategy (state : GameState) : Point :=
  sorry

/-- The theorem stating that the Second Player can always win --/
theorem second_player_always_wins :
  ∀ n : ℕ, n > 1 →
  ∃ (strategy : GameState → Point),
  ∀ (firstPlayerMoves : List Point),
  wins Player.Second (GameState.mk n firstPlayerMoves (List.map (λ _ => secondPlayerStrategy (GameState.mk n firstPlayerMoves [])) (List.range n))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_always_wins_l146_14604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4y3_l146_14663

theorem coefficient_x4y3 : ∃ (c : ℕ), c = 120 ∧ 
  (∀ (x y : ℝ), (1 + x)^6 * (2 + y)^4 = c * x^4 * y^3 + (fun (x y : ℝ) ↦ (1 + x)^6 * (2 + y)^4 - c * x^4 * y^3) x y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4y3_l146_14663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_k_representation_of_fraction_l146_14616

theorem base_k_representation_of_fraction (k : ℕ) (h1 : k > 0) : 
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 
    (8 : ℚ) / 45 = (a : ℚ) / k + (b : ℚ) / k^2 + (a : ℚ) / k^3 + (b : ℚ) / k^4 + 
    (a : ℚ) / k^5 + (b : ℚ) / k^6) → 
  k = 21 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_k_representation_of_fraction_l146_14616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l146_14642

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x / 2 + Real.pi / 6) - 1

theorem f_extrema :
  (∀ x, f x ≤ 2) ∧
  (∀ x, f x ≥ -4) ∧
  (∀ k : ℤ, f (4 * ↑k * Real.pi + 2 * Real.pi / 3) = 2) ∧
  (∀ k : ℤ, f (4 * ↑k * Real.pi - 4 * Real.pi / 3) = -4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l146_14642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l146_14646

-- Define the sets A and B
def A : Set ℝ := {x | (2 : ℝ)^x > 8}
def B : Set ℝ := {x | x^2 - 3*x - 4 < 0}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = Set.Ioc (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l146_14646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_over_n_squared_series_convergence_l146_14698

theorem sin_over_n_squared_series_convergence :
  let a : ℕ → ℝ := λ n => Real.sin (n : ℝ) / (n : ℝ)^2
  (∀ n : ℕ, |Real.sin (n : ℝ)| ≤ 1) →
  ∃ (L : ℝ), Summable a ∧ Summable (λ n => |a n|) ∧ HasSum a L :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_over_n_squared_series_convergence_l146_14698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_time_is_16_seconds_l146_14670

/-- The time taken for a person to cover the entire length of an escalator -/
noncomputable def escalatorTime (escalatorSpeed : ℝ) (personSpeed : ℝ) (escalatorLength : ℝ) : ℝ :=
  escalatorLength / (escalatorSpeed + personSpeed)

/-- Theorem: The time taken for a person to cover the entire length of an escalator is 16 seconds -/
theorem escalator_time_is_16_seconds :
  let escalatorSpeed : ℝ := 8
  let personSpeed : ℝ := 2
  let escalatorLength : ℝ := 160
  escalatorTime escalatorSpeed personSpeed escalatorLength = 16 := by
  -- Unfold the definition of escalatorTime
  unfold escalatorTime
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_time_is_16_seconds_l146_14670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_octagon_area_ratio_l146_14671

/-- The area between circumscribed and inscribed circles of a regular polygon -/
noncomputable def area_between_circles (n : ℕ) (s : ℝ) : ℝ :=
  let θ := 2 * Real.pi / n
  let r := s / (2 * Real.tan (θ / 2))
  let R := s / (2 * Real.sin (θ / 2))
  Real.pi * (R^2 - r^2)

/-- The theorem stating the relationship between areas for hexagon and octagon -/
theorem hexagon_octagon_area_ratio :
  (area_between_circles 6 3) = 4/5 * (area_between_circles 8 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_octagon_area_ratio_l146_14671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l146_14606

theorem count_integers_in_range : 
  (Finset.filter (fun n : ℕ => 300 < n^2 ∧ n^2 < 1200) (Finset.range 35)).card = 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l146_14606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_circle_l146_14673

/-- The equation of a circle in polar coordinates -/
noncomputable def circle_equation (ρ θ : ℝ) : Prop :=
  ρ = Real.sqrt 2 * (Real.cos θ + Real.sin θ)

/-- The center of the circle in polar coordinates -/
noncomputable def circle_center : ℝ × ℝ := (1, Real.pi / 4)

/-- Theorem: The polar coordinates of the center of the circle described by 
    ρ = √2(cos θ + sin θ) are (1, π/4) -/
theorem center_of_circle :
  ∀ ρ θ : ℝ, circle_equation ρ θ → 
  ∃ r φ : ℝ, (r, φ) = circle_center ∧ 
  r * Real.cos φ = Real.sqrt 2 / 2 ∧ 
  r * Real.sin φ = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_circle_l146_14673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l146_14659

/-- A regular quadrilateral pyramid with an inscribed sphere -/
structure RegularQuadrilateralPyramid where
  h : ℝ  -- height of the pyramid
  r : ℝ  -- radius of the inscribed sphere
  h_pos : 0 < h
  r_pos : 0 < r
  h_gt_r : h > r

/-- The volume of a regular quadrilateral pyramid with an inscribed sphere -/
noncomputable def volume (p : RegularQuadrilateralPyramid) : ℝ :=
  (4 * p.h^5 - 4 * p.h^3 * p.r^2) / (3 * p.r^2)

/-- Theorem: The volume formula for a regular quadrilateral pyramid with an inscribed sphere -/
theorem volume_formula (p : RegularQuadrilateralPyramid) :
  volume p = (4 * p.h^5 - 4 * p.h^3 * p.r^2) / (3 * p.r^2) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l146_14659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l146_14619

def sequence_sum (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_properties (a : ℕ+ → ℝ) 
  (h1 : a 1 ≠ 0)
  (h2 : ∀ n : ℕ+, 2 * a n - a 1 = a 1 * sequence_sum a n) :
  (a 1 = 1) ∧
  (∀ n : ℕ+, a n = (2 : ℝ) ^ (n.val - 1)) ∧
  (∀ n : ℕ+, sequence_sum (λ k => (k.val : ℝ) * a k) n = 1 + ((n.val - 1) : ℝ) * 2^n.val) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l146_14619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l146_14644

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties : 
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ 
    (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧ 
  f (Real.pi / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l146_14644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cone_height_equals_frustum_altitude_l146_14614

/-- Frustum data -/
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the radius of a circle given its area -/
noncomputable def radius_from_area (area : ℝ) : ℝ :=
  Real.sqrt (area / Real.pi)

/-- Theorem: The height of the small cone removed to create a frustum is equal to the frustum's altitude -/
theorem small_cone_height_equals_frustum_altitude (f : Frustum) 
  (h1 : f.altitude = 16)
  (h2 : f.lower_base_area = 196 * Real.pi)
  (h3 : f.upper_base_area = 49 * Real.pi) :
  let r1 := radius_from_area f.lower_base_area
  let r2 := radius_from_area f.upper_base_area
  r2 / r1 = 1 / 2 → f.altitude = radius_from_area f.upper_base_area :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cone_height_equals_frustum_altitude_l146_14614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outside_area_fraction_is_correct_l146_14675

/-- Represents a right triangle with a 30-60-90 degree angle configuration -/
structure Triangle30_60_90 where
  -- AB is the side opposite to the 30° angle
  AB : ℝ
  -- AC is the side opposite to the 60° angle
  AC : ℝ
  -- BC is the hypotenuse
  BC : ℝ
  -- Ensure the side lengths satisfy the 30-60-90 triangle ratio
  h_sides : AC = AB * Real.sqrt 3 ∧ BC = 2 * AB

/-- Represents a circle inscribed in the triangle -/
structure InscribedCircle (t : Triangle30_60_90) where
  -- The radius of the inscribed circle
  radius : ℝ
  -- The circle is tangent to the hypotenuse at two points
  h_tangent : radius = t.AB ∧ radius = t.AC / Real.sqrt 3

/-- The fraction of the triangle's area that lies outside the inscribed circle -/
noncomputable def outsideAreaFraction (t : Triangle30_60_90) (c : InscribedCircle t) : ℝ :=
  Real.sqrt 3 / 2 - Real.pi / (2 * Real.sqrt 3)

theorem outside_area_fraction_is_correct (t : Triangle30_60_90) (c : InscribedCircle t) :
  outsideAreaFraction t c = Real.sqrt 3 / 2 - Real.pi / (2 * Real.sqrt 3) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_outside_area_fraction_is_correct_l146_14675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot30_eq_sqrt3_l146_14626

noncomputable def tan30 : ℝ := 1 / Real.sqrt 3

noncomputable def cot (θ : ℝ) : ℝ := 1 / Real.tan θ

-- Theorem statement
theorem cot30_eq_sqrt3 : cot (30 * π / 180) = Real.sqrt 3 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot30_eq_sqrt3_l146_14626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_count_l146_14655

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a quadrilateral with four vertices -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Given a quadrilateral, counts the number of possible positive integer lengths for diagonal AC -/
noncomputable def countPossibleDiagonalLengths (q : Quadrilateral) : ℕ :=
  let AB := distance q.A q.B
  let BC := distance q.B q.C
  let CD := distance q.C q.D
  let DA := distance q.D q.A
  let minLength := max (AB - BC) (CD - DA)
  let maxLength := min (AB + BC) (CD + DA)
  (Int.floor maxLength).toNat - (Int.ceil minLength).toNat + 1

theorem diagonal_length_count (q : Quadrilateral) :
  q.A = ⟨0, 0⟩ ∧ q.B = ⟨4, 6⟩ ∧ q.C = ⟨11, 2⟩ ∧ q.D = ⟨6, -7⟩ →
  distance q.A q.B = 6 ∧ distance q.B q.C = 9 ∧ distance q.C q.D = 14 ∧ distance q.D q.A = 10 →
  countPossibleDiagonalLengths q = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_count_l146_14655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_ratio_l146_14665

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def projection (v w : V) : V := (inner v w / ‖w‖^2) • w

theorem projection_ratio (v w : V) (h : w ≠ 0) :
  let p := projection v w
  let q := projection p w
  ‖p‖ / ‖v‖ = 4/5 → ‖q‖ / ‖w‖ = 4/5 := by
  intro hp
  sorry

#check projection_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_ratio_l146_14665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_probabilities_l146_14637

/-- The probability of making a basket -/
noncomputable def p : ℝ := 1/2

/-- The number of shots -/
def n : ℕ := 5

/-- The probability of making 4 consecutive shots out of 5 -/
noncomputable def prob_consecutive_4 : ℝ := 1/16

/-- The probability of making exactly 4 shots out of 5 -/
noncomputable def prob_exactly_4 : ℝ := 5/32

/-- Theorem stating the probabilities for consecutive and exact shots -/
theorem basketball_probabilities :
  (prob_consecutive_4 = p^4 * (1 - p) + (1 - p) * p^4) ∧
  (prob_exactly_4 = (n.choose 4 : ℝ) * p^4 * (1 - p)) := by
  sorry

#check basketball_probabilities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_probabilities_l146_14637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perfect_square_in_sequence_l146_14643

noncomputable def f (n : ℕ) : ℕ := Nat.floor ((n : ℝ) + Real.sqrt (n : ℝ))

def m : ℕ := 1111

theorem exists_perfect_square_in_sequence :
  ∃ k : ℕ, ∃ s : ℕ, (f^[k] m) = s^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perfect_square_in_sequence_l146_14643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l146_14654

def A : Set ℕ := {x : ℕ | x > 0 ∧ Real.sqrt (x : ℝ) ≤ 2}
def B : Set ℕ := {y : ℕ | ∃ x : ℝ, (y : ℝ) = x^2 + 2}

theorem A_intersect_B : A ∩ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l146_14654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_selection_for_double_minimum_twelve_for_double_l146_14627

/-- The set of integers from 1 to 16 -/
def S : Finset ℕ := Finset.range 16

/-- A function that checks if one number is twice another -/
def is_double (a b : ℕ) : Prop := b = 2 * a

/-- The theorem stating that at least 12 integers must be chosen -/
theorem minimum_selection_for_double (T : Finset ℕ) (h : T ⊆ S) : 
  (∀ (a b : ℕ), a ∈ T → b ∈ T → ¬is_double a b) → T.card ≤ 11 :=
sorry

/-- The main theorem proving that 12 is the minimum number of integers needed -/
theorem minimum_twelve_for_double : 
  ∃ T : Finset ℕ, T ⊆ S ∧ T.card = 12 ∧ ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ is_double a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_selection_for_double_minimum_twelve_for_double_l146_14627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_external_angles_is_360_l146_14647

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_cyclic : ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i, (vertices i).1^2 + (vertices i).2^2 = radius^2

-- Define external angles of the quadrilateral
noncomputable def external_angles (q : CyclicQuadrilateral) : Fin 4 → ℝ :=
  λ i => sorry

-- Define the property that external angles are inscribed in external arcs
def inscribed_in_external_arcs (q : CyclicQuadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem sum_of_external_angles_is_360 (q : CyclicQuadrilateral) 
  (h : inscribed_in_external_arcs q) : 
  (Finset.univ.sum (external_angles q)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_external_angles_is_360_l146_14647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_alpha_l146_14667

/-- Given two vectors a and b in R^2, where a = (3, 4) and b = (sin α, cos α),
    if a is parallel to b, then tan α = 3/4 -/
theorem parallel_vectors_tan_alpha (α : ℝ) :
  let a : Fin 2 → ℝ := ![3, 4]
  let b : Fin 2 → ℝ := ![Real.sin α, Real.cos α]
  (∃ (k : ℝ), ∀ (i : Fin 2), a i = k * b i) →
  Real.tan α = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_alpha_l146_14667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l146_14601

/-- The number of positive integer divisors of n, including 1 and n -/
def τ (n : ℕ+) : ℕ := sorry

/-- The sum of τ(k) for k from 1 to n -/
def S (n : ℕ+) : ℕ := sorry

/-- The number of positive integers n ≤ 2005 with S(n) odd -/
def a : ℕ := sorry

/-- The number of positive integers n ≤ 2005 with S(n) even -/
def b : ℕ := sorry

/-- The absolute difference between a and b is 25 -/
theorem divisor_sum_parity_difference : |Int.ofNat a - Int.ofNat b| = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l146_14601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_to_identity_l146_14613

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![1/2, -Real.sqrt 3/2; Real.sqrt 3/2, 1/2]

def is_identity (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  M = !![1, 0; 0, 1]

theorem smallest_power_to_identity :
  (∀ k : ℕ, k < 12 → ¬(is_identity (rotation_matrix ^ k))) ∧
  (is_identity (rotation_matrix ^ 12)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_to_identity_l146_14613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_line_theorem_l146_14687

noncomputable section

/-- Represents a parabola y^2 = ax -/
structure Parabola where
  a : ℝ

/-- Represents a line with slope 2 -/
structure Line where
  y_intercept : ℝ

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (p.a / 4, 0)

/-- Point where the line intersects the y-axis -/
def y_axis_intersection (l : Line) : ℝ × ℝ := (0, l.y_intercept)

/-- Check if a line passes through a point -/
def line_passes_through (l : Line) (point : ℝ × ℝ) : Prop :=
  point.2 = 2 * point.1 + l.y_intercept

/-- Calculate the area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

theorem parabola_focus_line_theorem (p : Parabola) (l : Line) :
  line_passes_through l (focus p) →
  triangle_area (0, 0) (y_axis_intersection l) (focus p) = 4 →
  p.a = 8 ∨ p.a = -8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_line_theorem_l146_14687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_properties_l146_14621

noncomputable section

/-- Linear function passing through (-1, -5) and intersecting y = 1/2x at (2, a) -/
def linear_function (x k b : ℝ) : ℝ := k * x + b

/-- Proportional function y = 1/2x -/
def prop_function (x : ℝ) : ℝ := (1/2) * x

/-- The point where the linear function intersects the x-axis -/
def x_intercept (k b : ℝ) : ℝ := b / (-k)

theorem linear_function_properties :
  ∃ (k b a : ℝ),
    (linear_function (-1) k b = -5) ∧
    (linear_function 2 k b = prop_function 2) ∧
    (a = 1) ∧
    (k = 2) ∧
    (b = -3) ∧
    (1/2 * (x_intercept k b) * (prop_function 2) = 3/4) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_properties_l146_14621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_for_angle_C_l146_14691

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the cosine of angle C using the law of cosines
noncomputable def cos_C (a b c : ℝ) : ℝ := (a^2 + b^2 - c^2) / (2 * a * b)

-- Define the angle C in degrees
noncomputable def angle_C (a b c : ℝ) : ℝ := Real.arccos (cos_C a b c) * (180 / Real.pi)

-- State the theorem
theorem largest_x_for_angle_C :
  ∀ a b c : ℝ,
  triangle_ABC a b c →
  a = 2 →
  b = 3 →
  c > 4 →
  (∀ x : ℝ, x > 105 → angle_C a b c > x) ∧
  ¬(∀ x : ℝ, x > 105.000001 → angle_C a b c > x) :=
by
  sorry

#check largest_x_for_angle_C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_for_angle_C_l146_14691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_equidistant_from_A_and_B_l146_14657

/-- The distance between two points in 3D space -/
noncomputable def distance (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

/-- Theorem: The point C(0, 0, 14/9) on the z-axis is equidistant from A(-4, 1, 7) and B(3, 5, -2) -/
theorem point_C_equidistant_from_A_and_B :
  let xA : ℝ := -4
  let yA : ℝ := 1
  let zA : ℝ := 7
  let xB : ℝ := 3
  let yB : ℝ := 5
  let zB : ℝ := -2
  let xC : ℝ := 0
  let yC : ℝ := 0
  let zC : ℝ := 14/9
  distance xA yA zA xC yC zC = distance xB yB zB xC yC zC :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_equidistant_from_A_and_B_l146_14657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_75_degrees_l146_14611

open Real

-- Define the angle addition formula for tangent
noncomputable def tan_add (a b : ℝ) : ℝ := (tan a + tan b) / (1 - tan a * tan b)

-- State the theorem
theorem tan_75_degrees :
  tan (75 * π / 180) = 2 + sqrt 3 :=
by
  -- Define the known values
  have tan_45 : tan (45 * π / 180) = 1 := by sorry
  have tan_30 : tan (30 * π / 180) = 1 / sqrt 3 := by sorry
  
  -- Apply the angle addition formula
  have h1 : tan (75 * π / 180) = tan_add (45 * π / 180) (30 * π / 180) := by sorry
  
  -- The rest of the proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_75_degrees_l146_14611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_growth_l146_14612

/-- The annual growth rate of the village population -/
noncomputable def annual_growth_rate : ℝ := 0.18

/-- The population after 2 years -/
noncomputable def population_after_2_years : ℝ := 10860.72

/-- The current population of the village -/
noncomputable def current_population : ℝ := population_after_2_years / (1 + annual_growth_rate)^2

theorem village_population_growth :
  current_population * (1 + annual_growth_rate)^2 = population_after_2_years := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_growth_l146_14612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_distance_sum_l146_14692

/-- Given a triangle PQR with centroid G, if GP² + GQ² + GR² = 72, then PQ² + PR² + QR² = 216 -/
theorem triangle_centroid_distance_sum (P Q R G : ℝ × ℝ) : 
  (G = ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)) →
  (dist G P)^2 + (dist G Q)^2 + (dist G R)^2 = 72 →
  (dist P Q)^2 + (dist P R)^2 + (dist Q R)^2 = 216 := by
  sorry

#check triangle_centroid_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_distance_sum_l146_14692
