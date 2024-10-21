import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_mileage_approx_l297_29733

/-- Calculates the highway mileage per gallon given the following conditions:
  * City mileage is 30 miles per gallon
  * Trip consists of 60 city miles and 200 highway miles (one way)
  * Gas costs $3.00 per gallon
  * Total spent on gas is $42
-/
noncomputable def highway_mileage (city_mpg : ℝ) (city_miles : ℝ) (highway_miles : ℝ) 
                    (gas_cost : ℝ) (total_spent : ℝ) : ℝ :=
  let city_gallons := city_miles / city_mpg
  let city_cost := city_gallons * gas_cost
  let highway_cost := total_spent - city_cost
  let highway_gallons := highway_cost / gas_cost
  highway_miles / highway_gallons

/-- Theorem stating that the highway mileage is approximately 16.67 mpg -/
theorem highway_mileage_approx :
  |highway_mileage 30 60 200 3 42 - 50/3| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_mileage_approx_l297_29733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_is_odd_l297_29723

/-- The sine function is odd -/
theorem sine_is_odd : ∀ x : ℝ, Real.sin (-x) = -Real.sin x := by
  intro x
  exact Real.sin_neg x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_is_odd_l297_29723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_expression_l297_29797

theorem log_sum_expression (n : ℝ) (h : (3 : ℝ)^n = 2) : 
  Real.log 6 / Real.log 3 + Real.log 8 / Real.log 3 = 4*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_expression_l297_29797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_sliced_cone_l297_29779

/-- A right circular cone sliced into five equal-height sections -/
structure SlicedCone where
  base_radius : ℝ
  height : ℝ
  (positive_radius : base_radius > 0)
  (positive_height : height > 0)

/-- Volume of a cone segment given its base radius and height -/
noncomputable def cone_segment_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- Volume of the largest piece (bottom piece) -/
noncomputable def largest_piece_volume (cone : SlicedCone) : ℝ :=
  cone_segment_volume (cone.base_radius) (cone.height / 5) -
  cone_segment_volume ((4/5) * cone.base_radius) ((4/5) * cone.height / 5)

/-- Volume of the second-largest piece (second from bottom) -/
noncomputable def second_largest_piece_volume (cone : SlicedCone) : ℝ :=
  cone_segment_volume ((4/5) * cone.base_radius) ((4/5) * cone.height / 5) -
  cone_segment_volume ((3/5) * cone.base_radius) ((3/5) * cone.height / 5)

/-- The main theorem stating the ratio of volumes -/
theorem volume_ratio_of_sliced_cone (cone : SlicedCone) :
  second_largest_piece_volume cone / largest_piece_volume cone = 37 / 61 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_sliced_cone_l297_29779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_upstream_time_l297_29746

-- Define the boat's speed in still water
noncomputable def boat_speed : ℝ := 15

-- Define the stream's speed
noncomputable def stream_speed : ℝ := 3

-- Define the time taken to travel downstream
noncomputable def downstream_time : ℝ := 1

-- Define the function to calculate upstream time
noncomputable def upstream_time (boat_speed stream_speed downstream_time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * downstream_time / (boat_speed - stream_speed)

-- Theorem statement
theorem boat_upstream_time :
  upstream_time boat_speed stream_speed downstream_time = 1.5 := by
  -- Unfold the definitions
  unfold upstream_time boat_speed stream_speed downstream_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_upstream_time_l297_29746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_intersects_side_l297_29792

/-- A polygon inscribed in a circle -/
structure InscribedPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  inscribed : ∀ i : Fin n, (vertices i).1^2 + (vertices i).2^2 = 1

/-- The opposite side of a vertex in a polygon -/
def oppositeSide (p : InscribedPolygon 101) (i : Fin 101) : Set (ℝ × ℝ) :=
  let j := (i + 50) % 101
  let k := (i + 51) % 101
  {x : ℝ × ℝ | ∃ t : ℝ, x = (1 - t) • p.vertices j + t • p.vertices k}

/-- The perpendicular from a vertex to a line -/
def perpendicular (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {x : ℝ × ℝ | ∃ q ∈ l, (x.1 - p.1) * (q.1 - x.1) + (x.2 - p.2) * (q.2 - x.2) = 0}

/-- The theorem statement -/
theorem perpendicular_intersects_side (p : InscribedPolygon 101) :
  ∃ i : Fin 101, (perpendicular (p.vertices i) (oppositeSide p i)) ∩ (oppositeSide p i) ≠ ∅ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_intersects_side_l297_29792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_example_permutations_l297_29710

theorem example_permutations :
  Finset.card (Finset.univ.image (fun σ : Equiv.Perm (Fin 7) => σ)) = Nat.factorial 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_example_permutations_l297_29710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_perimeter_l297_29748

/-- A rectangular garden with specific length-width relationship and area -/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  area : ℝ
  length_width_relation : length = 3 * width + 15
  area_equation : area = width * length
  area_value : area = 4050

/-- The perimeter of a rectangular garden -/
def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.width)

/-- Theorem stating the perimeter of the specific rectangular garden -/
theorem garden_perimeter (g : RectangularGarden) : 
  ∃ ε > 0, |perimeter g - 304.64| < ε := by
  sorry

#check garden_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_perimeter_l297_29748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_optimization_l297_29754

noncomputable def production_cost (x : ℝ) : ℝ := x^2/10 - 30*x + 4000

noncomputable def average_cost (x : ℝ) : ℝ := production_cost x / x

noncomputable def annual_profit (x : ℝ) : ℝ := 16*x - production_cost x

theorem production_optimization :
  ∃ (x_min x_max : ℝ),
    150 ≤ x_min ∧ x_min ≤ 250 ∧
    150 ≤ x_max ∧ x_max ≤ 250 ∧
    (∀ x, 150 ≤ x → x ≤ 250 → average_cost x_min ≤ average_cost x) ∧
    average_cost x_min = 10 ∧
    (∀ x, 150 ≤ x → x ≤ 250 → annual_profit x ≤ annual_profit x_max) ∧
    annual_profit x_max = 1290 ∧
    x_min = 200 ∧
    x_max = 230 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_optimization_l297_29754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l297_29726

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then x^2 - a*x + 1 else x^2 - 3*a*x + 2*a^2 + 1

-- State the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∃ θ : ℝ, π/4 < θ ∧ θ < π/2 ∧ f a (Real.sin θ) = f a (Real.cos θ)) →
  (1/2 < a ∧ a < Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l297_29726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l297_29758

/-- A parabola with equation y² = 20x -/
structure Parabola where
  equation : ∀ x y : ℝ, y^2 = 20 * x

/-- A hyperbola with equation x²/a² - y²/b² = 1, where a > b > 0 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a
  equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

/-- The focus of a parabola y² = 20x -/
noncomputable def focus (p : Parabola) : ℝ × ℝ := (5, 0)

/-- The asymptote of a hyperbola x²/a² - y²/b² = 1 -/
noncomputable def asymptote (h : Hyperbola) : ℝ → ℝ := fun x ↦ - (h.b / h.a) * x

/-- The distance from a point to a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) (f : ℝ → ℝ) : ℝ := 
  |p.2 - f p.1| / Real.sqrt (1 + (deriv f p.1)^2)

theorem hyperbola_equation (p : Parabola) (h : Hyperbola) :
  (focus p = (h.a, 0) ∨ focus p = (-h.a, 0)) →
  distanceToLine (focus p) (asymptote h) = 4 →
  h.a = 3 ∧ h.b = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l297_29758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_condition_l297_29788

theorem quadruple_condition (a₁ b₁ c₁ d₁ : ℕ) :
  (∃ (a₀ b₀ c₀ d₀ : ℤ), |a₀ - b₀| = a₁ ∧ |b₀ - c₀| = b₁ ∧ |c₀ - d₀| = c₁ ∧ |d₀ - a₀| = d₁) ↔
  (∃ (s₁ s₂ s₃ s₄ : ℤ), s₁ * a₁ + s₂ * b₁ + s₃ * c₁ + s₄ * d₁ = 0 ∧ 
   s₁^2 = 1 ∧ s₂^2 = 1 ∧ s₃^2 = 1 ∧ s₄^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_condition_l297_29788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_gain_calculation_l297_29778

/-- Continuous compound interest calculation -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * Real.exp (rate * time)

/-- Currency conversion -/
def convert_currency (amount : ℝ) (exchange_rate : ℝ) : ℝ :=
  amount * exchange_rate

theorem b_gain_calculation (initial_loan : ℝ) (ab_rate : ℝ) (ab_time : ℝ)
  (usd_to_eur : ℝ) (bc_rate : ℝ) (bc_time : ℝ) (eur_to_usd : ℝ) :
  initial_loan = 3500 →
  ab_rate = 0.10 →
  ab_time = 2 →
  usd_to_eur = 0.85 →
  bc_rate = 0.14 →
  bc_time = 3 →
  eur_to_usd = 1.17 →
  ∃ (gain : ℝ), abs (gain - 2199.70) < 0.01 ∧
    gain = convert_currency 
      (compound_interest 
        (convert_currency 
          (compound_interest initial_loan ab_rate ab_time) 
        usd_to_eur) 
      bc_rate bc_time) 
    eur_to_usd - compound_interest initial_loan ab_rate ab_time :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_gain_calculation_l297_29778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_roots_for_all_p_infinite_p_with_equal_roots_l297_29794

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- A quadratic equation ax^2 + bx + c = 0 has equal roots if and only if its discriminant is zero -/
def has_equal_roots (a b c : ℝ) : Prop := discriminant a b c = 0

/-- The quadratic equation x^2 - 2px + p^2 = 0 has equal roots for all real p -/
theorem equal_roots_for_all_p : ∀ p : ℝ, has_equal_roots 1 (-2*p) (p^2) := by
  sorry

/-- The set of real values of p for which x^2 - 2px + p^2 = 0 has equal roots is not finite -/
theorem infinite_p_with_equal_roots : ¬ Set.Finite {p : ℝ | has_equal_roots 1 (-2*p) (p^2)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_roots_for_all_p_infinite_p_with_equal_roots_l297_29794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_count_l297_29790

/-- Represents a seating arrangement around a circular table -/
def SeatingArrangement := Fin 8 → Fin 8

/-- Represents a married couple -/
structure Couple :=
  (husband : Fin 8)
  (wife : Fin 8)

/-- The set of all possible seating arrangements -/
def AllArrangements : Set SeatingArrangement := Set.univ

/-- Predicate to check if a seating arrangement is valid -/
def IsValidArrangement (arrangement : SeatingArrangement) (couples : Fin 4 → Couple) : Prop :=
  (∀ i : Fin 8, (arrangement i).val % 2 ≠ i.val % 2) ∧  -- Men and women alternate
  (∀ c : Fin 4, arrangement (couples c).husband ≠ (couples c).wife) ∧  -- Spouses not across
  (∀ c : Fin 4, ∀ i : Fin 8, arrangement i = (couples c).husband →
    arrangement ((i + 1) % 8) ≠ (couples c).wife ∧
    arrangement ((i + 7) % 8) ≠ (couples c).wife)  -- Spouses not adjacent

/-- The set of valid seating arrangements -/
def ValidArrangements (couples : Fin 4 → Couple) : Set SeatingArrangement :=
  { arr ∈ AllArrangements | IsValidArrangement arr couples }

/-- Provide an instance of Fintype for ValidArrangements -/
instance (couples : Fin 4 → Couple) : Fintype (ValidArrangements couples) :=
  sorry

theorem seating_arrangements_count (couples : Fin 4 → Couple) :
  Fintype.card (ValidArrangements couples) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_count_l297_29790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_properties_l297_29795

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set M
def M (a b c : ℝ) := {x : ℝ | f a b c x < 0}

-- Define the discriminant
def Δ (a b c : ℝ) := b^2 - 4*a*c

theorem quadratic_inequality_properties :
  -- Statement 1
  ∃ (a b c : ℝ), M a b c = ∅ ∧ a > 0 ∧ Δ a b c ≤ 0 ∧

  -- Statement 2
  ∀ (a b c : ℝ), M a b c = Set.Ioo (-1) 3 →
    {x : ℝ | -c * x^2 - b * x - b > c * x + 4 * a} = Set.Iic (-2) ∪ Set.Ioi (1/3) ∧

  -- Statement 3
  ∀ (a b c x₀ : ℝ), a < b →
    M a b c = {x : ℝ | x ≠ x₀} →
    (∃ (t : ℝ), (a + 4*c) / (b - a) = t ∧ t ≥ 2 - 2*Real.sqrt 2) ∧

  -- Statement 4
  ∀ (a b c : ℝ), a < 0 → M a b c ≠ ∅ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_properties_l297_29795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supermartingale_switch_l297_29744

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsFiniteMeasure P]
variable {ι : Type*} [Fintype ι] [LinearOrder ι]

variable (ξ η : ι → Ω → ℝ) (𝒟 : ι → Set (Set Ω))
variable (τ : Ω → ι)

def is_supermartingale (X : ι → Ω → ℝ) (ℱ : ι → Set (Set Ω)) : Prop :=
  ∀ i j, i ≤ j → MeasurableSet (ℱ i) → 
    ∀ A ∈ ℱ i, ∫ ω in A, X j ω ∂P ≤ ∫ ω in A, X i ω ∂P

def is_stopping_time (τ : Ω → ι) (ℱ : ι → Set (Set Ω)) : Prop :=
  ∀ i, MeasurableSet {ω | τ ω ≤ i}

def ζ (ξ η : ι → Ω → ℝ) (τ : Ω → ι) : ι → Ω → ℝ :=
  λ k ω ↦ if τ ω > k then ξ k ω else η k ω

theorem supermartingale_switch
  (hξ : is_supermartingale P ξ 𝒟)
  (hη : is_supermartingale P η 𝒟)
  (hτ : is_stopping_time τ 𝒟)
  (h_geq : P {ω | ξ (τ ω) ω ≥ η (τ ω) ω} = 1) :
  is_supermartingale P (ζ ξ η τ) 𝒟 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supermartingale_switch_l297_29744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l297_29714

theorem simplify_expression : 
  Real.sqrt 8 * (2 : ℝ)^(1/2 : ℝ) + (18 + 6 * 3) / 3 - (8 : ℝ)^(3/2 : ℝ) = 4 + 12 - 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l297_29714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_property_l297_29768

theorem inverse_function_property (f : ℝ → ℝ) (a : ℝ) (h₁ : Function.Bijective f) 
  (h₂ : ∀ x, (Function.invFun f) (x + a) = (Function.invFun (λ x => f (x + a))) x)
  (h₃ : f a = a) (h₄ : a ≠ 0) : f (2 * a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_property_l297_29768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_condition_equivalence_l297_29786

theorem partition_condition_equivalence (a : ℝ) (h : a > 0) :
  (∃ (n : ℕ) (A : ℕ → Set ℕ), 
    (∀ i j, i ≠ j → i ≤ n → j ≤ n → Disjoint (A i) (A j)) ∧ 
    (∀ i ≤ n, Set.Infinite (A i)) ∧
    (⋃ i ∈ Finset.range (n + 1), A i) = {x : ℕ | x > 0} ∧
    (∀ i ≤ n, ∀ b c, b ∈ A i → c ∈ A i → b > c → b - c ≥ a ^ i))
  ↔
  (a < 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_condition_equivalence_l297_29786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_foot_on_circle_l297_29732

/-- Two parallel lines in a plane -/
structure ParallelLines :=
  (r s : Set (ℝ × ℝ))
  (parallel : sorry) -- We'll need to define parallelism properly

/-- A point equidistant from two lines -/
def EquidistantPoint (l₁ l₂ : Set (ℝ × ℝ)) (A : ℝ × ℝ) : Prop :=
  sorry -- We'll need to define distance to a line

/-- The foot of a perpendicular from a point to a line segment -/
noncomputable def FootOfPerpendicular (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The theorem statement -/
theorem perpendicular_foot_on_circle 
  (L : ParallelLines) 
  (A : ℝ × ℝ) 
  (h_equidistant : EquidistantPoint L.r L.s A) :
  ∀ (B : ℝ × ℝ), B ∈ L.r →
    ∀ (C : ℝ × ℝ), C ∈ L.s →
      (∃ (d : ℝ), d > 0 ∧ sorry) → -- We'll need to define the angle condition
        ∃ (circle : Set (ℝ × ℝ)),
          sorry ∧ -- IsCircle condition
          (∃ (center : ℝ × ℝ), center = A) ∧
          (∃ (radius : ℝ), radius > 0 ∧ sorry) ∧ -- We'll need to define the radius condition
          FootOfPerpendicular A B C ∈ circle :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_foot_on_circle_l297_29732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l297_29703

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - x / (x + 1)

-- Define the domain of f
def f_domain (x : ℝ) : Prop := x > -1

-- State the theorem about monotonicity and tangent line
theorem f_properties :
  ∀ x : ℝ, f_domain x →
  (∀ y : ℝ, y > 0 → (deriv f) y > 0) ∧
  (∀ z : ℝ, -1 < z ∧ z < 0 → (deriv f) z < 0) ∧
  (∃ a b : ℝ, a = 1 ∧ b = 4*Real.log 2 - 3 ∧
    ∀ x y : ℝ, y = f x → x - 4*y + b = 0 ↔ x = a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l297_29703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_roots_implies_m_bound_l297_29772

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then Real.exp (-x) else Real.exp x

def g (m : ℝ) (x : ℝ) : ℝ := m * x^2

theorem four_roots_implies_m_bound (m : ℝ) : 
  (∃ w x y z : ℝ, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    f w + g m w = 0 ∧ f x + g m x = 0 ∧ f y + g m y = 0 ∧ f z + g m z = 0) →
  m < -(Real.exp 2 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_roots_implies_m_bound_l297_29772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_numbers_printable_l297_29782

def next_number (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else (n + 1001) / 2

def is_printable (n : ℕ) : Prop :=
  ∃ k : ℕ, Nat.iterate next_number k 1 = n

theorem not_all_numbers_printable :
  ∃ m : ℕ, m ≤ 100 ∧ m > 0 ∧ ¬(is_printable m) := by
  sorry

#check not_all_numbers_printable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_numbers_printable_l297_29782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_modulo_l297_29760

theorem remainder_sum_modulo (n m : ℕ) : 
  n % 157 = 53 → m % 193 = 76 → (n + m) % 61 = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_modulo_l297_29760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_xy_l297_29798

noncomputable def data_set (x y : ℝ) : List ℝ := [4, 5, 6, x, y]

noncomputable def mean (l : List ℝ) : ℝ := (l.sum) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  (l.map (λ x => (x - mean l)^2)).sum / l.length

theorem absolute_difference_xy (x y : ℝ) :
  mean (data_set x y) = 5 ∧
  variance (data_set x y) = 4/5 →
  |x - y| = 2 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_xy_l297_29798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_side_range_l297_29700

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of the sides
noncomputable def side_length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define an obtuse triangle
def is_obtuse (t : Triangle) : Prop :=
  let a := side_length t.B t.C
  let b := side_length t.A t.C
  let c := side_length t.A t.B
  a^2 > b^2 + c^2 ∨ b^2 > a^2 + c^2 ∨ c^2 > a^2 + b^2

-- Theorem statement
theorem obtuse_triangle_side_range (t : Triangle) :
  is_obtuse t →
  side_length t.A t.B = 2 →
  side_length t.A t.C = 5 →
  let x := side_length t.B t.C
  (3 < x ∧ x < Real.sqrt 21) ∨ (Real.sqrt 29 < x ∧ x < 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_side_range_l297_29700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_travels_to_beijing_l297_29720

-- Define IndepEvents as a structure to represent independent events
structure IndepEvents where
  -- Add any necessary fields or conditions here

-- Define Probability.atLeastOne function
def Probability.atLeastOne (pA pB pC : ℝ) : ℝ :=
  1 - (1 - pA) * (1 - pB) * (1 - pC)

theorem at_least_one_travels_to_beijing 
  (pA pB pC : ℝ) 
  (hA : pA = 1/3) 
  (hB : pB = 1/4) 
  (hC : pC = 1/5) 
  (hIndep : IndepEvents) : 
  Probability.atLeastOne pA pB pC = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_travels_to_beijing_l297_29720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_is_isosceles_right_l297_29784

def BA : Fin 2 → ℝ := ![1, -3]
def BC : Fin 2 → ℝ := ![4, -2]

def AC : Fin 2 → ℝ := BC - BA

theorem triangle_ABC_is_isosceles_right :
  (Finset.sum Finset.univ (λ i => BA i ^ 2) = Finset.sum Finset.univ (λ i => AC i ^ 2)) ∧
  (Finset.sum Finset.univ (λ i => BA i * AC i) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_is_isosceles_right_l297_29784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_increasing_sum_implies_positive_ratio_positive_ratio_not_sufficient_for_increasing_sum_l297_29739

/-- Represents a geometric sequence with first term a₁ and common ratio q -/
structure GeometricSequence where
  a₁ : ℝ
  q : ℝ

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sumFirstNTerms (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.a₁ * (1 - g.q^n) / (1 - g.q)

/-- Predicate to check if a sequence of real numbers is increasing -/
def isIncreasing (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, s (n + 1) > s n

theorem geometric_sequence_increasing_sum_implies_positive_ratio
  (g : GeometricSequence) :
  isIncreasing (sumFirstNTerms g) → g.q > 0 :=
by sorry

theorem positive_ratio_not_sufficient_for_increasing_sum :
  ∃ g : GeometricSequence, g.q > 0 ∧ ¬isIncreasing (sumFirstNTerms g) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_increasing_sum_implies_positive_ratio_positive_ratio_not_sufficient_for_increasing_sum_l297_29739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_exp_curve_l297_29783

noncomputable def curve (x : ℝ) : ℝ := Real.exp x
def line (k : ℝ) (x : ℝ) : ℝ := x + k

def is_tangent (k : ℝ) : Prop :=
  ∃ x₀ : ℝ, 
    curve x₀ = line k x₀ ∧ 
    (deriv curve) x₀ = (deriv (line k)) x₀

theorem tangent_line_to_exp_curve (k : ℝ) :
  is_tangent k → k = 1 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_exp_curve_l297_29783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l297_29751

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 3*y + 2 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 3*y + 1 = 0

-- Define the centers and radii
noncomputable def center₁ : ℝ × ℝ := (-2, -3/2)
noncomputable def center₂ : ℝ × ℝ := (-1, -3/2)
noncomputable def radius₁ : ℝ := Real.sqrt 17 / 2
noncomputable def radius₂ : ℝ := 3/2

-- Define the distance between centers
noncomputable def distance : ℝ := 1

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance < radius₁ + radius₂ ∧ distance > abs (radius₁ - radius₂) := by
  sorry

#check circles_intersect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l297_29751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_equals_set_l297_29712

noncomputable def U : Finset ℝ := {-2, -8, 0, Real.pi, 6, 10}
noncomputable def A : Finset ℝ := {-2, Real.pi, 6}
def B : Finset ℝ := {1}

theorem complement_union_equals_set : (U \ A) ∪ B = {0, 1, -8, 10} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_equals_set_l297_29712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_range_l297_29737

/-- The function f(x) = log_a(2x - a) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x - a) / Real.log a

/-- The theorem statement -/
theorem log_function_range (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) (2/3 : ℝ), f a x > 0) → 
  a ∈ Set.Ioo (1/3 : ℝ) (1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_range_l297_29737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l297_29709

-- Define the integer part function
noncomputable def intPart (x : ℝ) : ℤ := ⌊x⌋

-- Define the decimal part function
noncomputable def decPart (x : ℝ) : ℝ := x - (intPart x : ℝ)

-- Define the equation
def equation (x : ℝ) : Prop := 2 * (intPart x : ℝ) = x + 2 * decPart x

-- Theorem statement
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 0 ∨ x = 4/3 ∨ x = 8/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l297_29709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_cost_price_l297_29771

/-- Given a selling price and a markup percentage, calculate the cost price. -/
noncomputable def cost_price (selling_price : ℝ) (markup_percent : ℝ) : ℝ :=
  selling_price / (1 + markup_percent / 100)

/-- Theorem: The cost price of an item with a selling price of 5400 and a 32% markup is approximately 4090.91 -/
theorem furniture_cost_price :
  let selling_price : ℝ := 5400
  let markup_percent : ℝ := 32
  let calculated_cost_price := cost_price selling_price markup_percent
  ∃ ε > 0, |calculated_cost_price - 4090.91| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_cost_price_l297_29771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_power_sum_l297_29752

theorem cos_power_sum (α : ℝ) (x : ℂ) (n : ℕ) (h : x + 1 / x = 2 * Real.cos α) :
  x^n + 1 / x^n = 2 * Real.cos (n * α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_power_sum_l297_29752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_range_inequality_condition_range_l297_29777

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((2 / x) + a)

/-- The function F(x) -/
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x - Real.log ((2 - a) * x + 3 * a - 3)

/-- Theorem for the range of a where F has a unique zero -/
theorem unique_zero_range (a : ℝ) :
  (∃! x, F a x = 0) ↔ a ∈ Set.Ioc (-1) (4/3) ∪ {2, 5/2} := by sorry

/-- Theorem for the range of positive a satisfying the inequality condition -/
theorem inequality_condition_range (a : ℝ) :
  (∀ m, m ∈ Set.Icc (3/4) 1 → ∀ x₁ x₂, x₁ ∈ Set.Icc m (4*m - 1) → x₂ ∈ Set.Icc m (4*m - 1) →
   |f a x₁ - f a x₂| ≤ Real.log 2) ↔ a ≥ 12 - 8 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_range_inequality_condition_range_l297_29777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_work_days_l297_29766

/-- The work ratio between a man and a boy -/
def m : ℝ := sorry

/-- The number of days it takes 60 boys to complete the work -/
def days_for_boys : ℝ := 40 * m

/-- The work completed by 240 men in 20 days -/
def work_by_men : ℝ := 240 * 20

/-- The work completed by 60 boys in the unknown number of days -/
def work_by_boys : ℝ := 60 * days_for_boys

theorem boys_work_days :
  work_by_men = 2 * work_by_boys :=
by
  -- Expand definitions
  unfold work_by_men work_by_boys days_for_boys
  -- Perform algebraic manipulations
  simp [mul_assoc, mul_comm]
  -- The proof is incomplete, so we use sorry
  sorry

#check boys_work_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_work_days_l297_29766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_unit_vectors_l297_29761

/-- Given points M(1, 1) and N(4, -3), prove that the collinear unit vectors
    with vector MN are (3/5, -4/5) and (-3/5, 4/5). -/
theorem collinear_unit_vectors (M N : ℝ × ℝ) (h : M = (1, 1) ∧ N = (4, -3)) :
  let MN := (N.1 - M.1, N.2 - M.2)
  let magnitude := Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2)
  let unit_vector1 := ((N.1 - M.1) / magnitude, (N.2 - M.2) / magnitude)
  let unit_vector2 := (-(N.1 - M.1) / magnitude, -(N.2 - M.2) / magnitude)
  unit_vector1 = (3/5, -4/5) ∧ unit_vector2 = (-3/5, 4/5) := by
  sorry

#check collinear_unit_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_unit_vectors_l297_29761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_wrong_answers_for_prize_l297_29775

/-- Represents the competition rules and conditions -/
structure CompetitionRules where
  totalQuestions : Nat
  correctPoints : Nat
  notAnsweredPoints : Nat
  wrongPoints : Int
  minPrizeScore : Nat
  notAnswered : Nat

/-- Calculates the maximum number of questions that can be answered incorrectly while still winning a prize -/
def maxWrongAnswers (rules : CompetitionRules) : Nat :=
  Int.toNat <| Int.floor (((rules.correctPoints : Int) * ((rules.totalQuestions : Int) - (rules.notAnswered : Int)) - (rules.minPrizeScore : Int)) / 
   ((rules.correctPoints : Int) + rules.wrongPoints))

/-- The main theorem stating that given the specific competition rules, 
    the maximum number of questions that can be answered incorrectly while still winning a prize is 3 -/
theorem max_wrong_answers_for_prize : 
  let rules := CompetitionRules.mk 25 4 0 (-1) 80 1
  maxWrongAnswers rules = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_wrong_answers_for_prize_l297_29775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l297_29755

theorem trigonometric_identities (α β : Real) 
  (h1 : Real.cos (α - β/2) = -2*Real.sqrt 7/7)
  (h2 : Real.sin (α/2 - β) = 1/2)
  (h3 : α ∈ Set.Ioo (π/2) π)
  (h4 : β ∈ Set.Ioo 0 (π/2)) :
  Real.cos ((α + β)/2) = -Real.sqrt 21/14 ∧ Real.tan (α + β) = 5*Real.sqrt 3/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l297_29755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l297_29789

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 / (2 * Real.sin x - 1)

-- Define the range set
def range_set : Set ℝ := {y | ∃ x, f x = y}

-- Theorem statement
theorem f_range : range_set = Set.Iic (-2/3) ∪ Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l297_29789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_copies_cover_l297_29780

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ
  width_positive : 0 < width
  length_positive : 0 < length
  width_le_length : width ≤ length

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.length

/-- The diagonal of a rectangle -/
noncomputable def Rectangle.diagonal (r : Rectangle) : ℝ := Real.sqrt (r.width^2 + r.length^2)

/-- Predicate to check if two copies of one rectangle can cover another -/
def can_cover (r1 r2 : Rectangle) : Prop :=
  2 * r1.width ≥ r2.width ∧ 2 * r1.length ≥ r2.length

/-- Theorem: If P and Q have the same area, P's diagonal is longer, and two copies of P can cover Q,
    then two copies of Q can cover P -/
theorem two_copies_cover (P Q : Rectangle) 
  (h_area : P.area = Q.area)
  (h_diagonal : P.diagonal > Q.diagonal)
  (h_P_covers_Q : can_cover P Q) :
  can_cover Q P :=
by
  sorry  -- The proof is skipped for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_copies_cover_l297_29780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l297_29741

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

-- Theorem stating that g is an odd function
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l297_29741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_area_of_triangle_ABC_l297_29721

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x - Real.pi/6)

-- Statement for the range of f(x)
theorem range_of_f :
  ∀ x ∈ Set.Icc 0 (Real.pi/2),
  f x ∈ Set.Icc (-1/2) (1/4) := by
  sorry

-- Define the properties of triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  f A = 1/4 ∧ a = Real.sqrt 3 ∧ Real.sin B = 2 * Real.sin C

-- Statement for the area of triangle ABC
theorem area_of_triangle_ABC (A B C : ℝ) (a b c : ℝ) :
  triangle_ABC A B C a b c →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_area_of_triangle_ABC_l297_29721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l297_29787

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := x^2 + (Real.log a + 2) * x + Real.log b

-- State the theorem
theorem function_values (a b : ℝ) : 
  (f a b (-1) = -2) ∧ 
  (∀ x : ℝ, f a b x ≥ 2 * x) → 
  a = 100 ∧ b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l297_29787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_to_g_transformation_l297_29750

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := f ((x + Real.pi / 6) / 2)

theorem f_to_g_transformation : g = Real.cos := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_to_g_transformation_l297_29750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_tangent_circles_l297_29705

-- Define the circles and triangle
def circle_small : ℝ := 3
def circle_large : ℝ := 5

-- Define the theorem
theorem triangle_area_tangent_circles : 
  ∀ (D E F : ℝ × ℝ),
  let DE := dist D E
  let DF := dist D F
  let EF := dist E F
  -- Conditions
  (∃ (P Q : ℝ × ℝ), 
    -- P and Q are centers of the circles
    (dist P D = circle_small + DE ∧ dist P F = circle_small + DF) ∧
    (dist Q D = circle_large + DE ∧ dist Q F = circle_large + DF) ∧
    -- Circles are tangent
    dist P Q = circle_small + circle_large ∧
    -- DE is longer than DF by the diameter of the smaller circle
    DE = DF + 2 * circle_small) →
  -- Conclusion
  (1/2 : ℝ) * EF * ((DE + DF) / 2) = 50 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_tangent_circles_l297_29705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exercise_problem_l297_29756

-- Define the given parameters
def distance_AB : ℝ := 9000
def speed_ratio : ℝ := 1.2
def time_difference : ℝ := 5
def initial_calorie_rate : ℝ := 10
def initial_calorie_duration : ℝ := 30
def calorie_increase_rate : ℝ := 1
def total_calories : ℝ := 2300

-- Define the speeds and total exercise time as variables to be proved
def xiaohong_speed : ℝ := 300
def xiaoming_speed : ℝ := 360
def total_exercise_time : ℝ := 70

-- Theorem statement
theorem exercise_problem :
  -- Conditions
  (distance_AB / xiaohong_speed - distance_AB / xiaoming_speed = time_difference) ∧
  (xiaoming_speed = speed_ratio * xiaohong_speed) ∧
  -- Calorie calculation
  (let time_to_B := distance_AB / xiaoming_speed
   let extra_time := total_exercise_time - time_to_B
   let initial_calories := min initial_calorie_duration total_exercise_time * initial_calorie_rate
   let extra_calories := (max (total_exercise_time - initial_calorie_duration) 0) * 
                         (initial_calorie_rate + (max (total_exercise_time - initial_calorie_duration) 0 + 1) / 2 * calorie_increase_rate)
   initial_calories + extra_calories = total_calories) →
  -- Conclusion
  xiaohong_speed = 300 ∧ xiaoming_speed = 360 ∧ total_exercise_time = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exercise_problem_l297_29756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_abs_sum_l297_29753

theorem floor_abs_sum : ⌊|((-42/10) : ℝ)|⌋ + |⌊(-42/10 : ℝ)⌋| = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_abs_sum_l297_29753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_plough_time_l297_29704

/-- The time it takes for s to plough the field alone -/
noncomputable def s_time (r_and_s_time r_time t_time : ℝ) : ℝ :=
  1 / (1 / r_and_s_time - 1 / r_time)

/-- Theorem: Given the conditions, s takes 30 hours to plough the field alone -/
theorem s_plough_time (r_and_s_time r_time t_time : ℝ) 
  (h1 : r_and_s_time = 10)
  (h2 : r_time = 15)
  (h3 : t_time = 20) :
  s_time r_and_s_time r_time t_time = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_plough_time_l297_29704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l297_29718

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem range_of_f :
  Set.range f = Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l297_29718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_equals_sqrt3_over_2_l297_29727

theorem sin_cos_sum_equals_sqrt3_over_2 :
  Real.sin (15 * π / 180) * Real.cos (45 * π / 180) +
  Real.sin (105 * π / 180) * Real.sin (135 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_equals_sqrt3_over_2_l297_29727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solvability_condition_l297_29716

open Real

theorem triangle_solvability_condition 
  (a b : ℝ) 
  (h_positive_a : 0 < a) 
  (h_positive_b : 0 < b) 
  (h_angle_ratio : ∃ (α β : ℝ), 0 < α ∧ 0 < β ∧ α / β = 1 / 2) :
  (∃ (α β γ : ℝ), α + β + γ = π ∧ a / Real.sin α = b / Real.sin β) ↔ b / 2 < a ∧ a < b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solvability_condition_l297_29716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l297_29757

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ)  -- Angles
variable (a b c : ℝ)  -- Sides

-- State the given conditions
axiom triangle_condition : b * Real.sin A = a * Real.cos (B - Real.pi/6)
axiom side_b : b = 3
axiom triangle_area : (1/2) * a * c * Real.sin B = 2 * Real.sqrt 3

-- State the theorem to be proved
theorem triangle_properties :
  B = Real.pi/3 ∧ a + b + c = 3 + Real.sqrt 33 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l297_29757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_sum_of_squares_modulo_l297_29747

open Nat

theorem unique_non_sum_of_squares_modulo (n : ℕ) :
  n ≥ 2 ∧
  (∃! x : Fin n, ¬∃ a b : Fin n, x = a * a + b * b) ↔
  n = 4 := by
  sorry  -- The proof goes here

#check unique_non_sum_of_squares_modulo

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_sum_of_squares_modulo_l297_29747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_central_angle_l297_29725

-- Define a circular sector
structure CircularSector where
  radius : ℝ
  arcLength : ℝ
  centralAngle : ℝ

-- Define the properties of the sector
def sectorProperties (s : CircularSector) : Prop :=
  s.radius > 0 ∧
  s.arcLength > 0 ∧
  s.centralAngle > 0 ∧
  s.centralAngle = s.arcLength / s.radius

-- Define the area of the sector
noncomputable def sectorArea (s : CircularSector) : ℝ :=
  1/2 * s.radius * s.arcLength

-- Define the perimeter of the sector
noncomputable def sectorPerimeter (s : CircularSector) : ℝ :=
  2 * s.radius + s.arcLength

-- Theorem statement
theorem sector_central_angle (s : CircularSector) :
  sectorProperties s →
  sectorArea s = 1 →
  sectorPerimeter s = 4 →
  s.centralAngle = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_central_angle_l297_29725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_inequality_implies_a_range_l297_29706

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x / Real.log 3)^2 + (a - 1) * (Real.log x / Real.log 3) + 3 * a - 2

-- Part 1: Minimum value of f is 2 implies a = 7 + 4√2
theorem min_value_implies_a (a : ℝ) :
  (∀ x > 0, f a x ≥ 2) ∧ (∃ x > 0, f a x = 2) → a = 7 + 4 * Real.sqrt 2 := by
  sorry

-- Part 2: Inequality condition implies a ≤ -4/3
theorem inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 3 9, f a (3 * x) + Real.log (9 * x) / Real.log 3 ≤ 0) → a ≤ -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_inequality_implies_a_range_l297_29706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l297_29740

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x + y - 4| / Real.sqrt 2

-- Theorem statement
theorem distance_range :
  ∀ x y : ℝ, circle_C x y →
  0 ≤ distance_to_line x y ∧ distance_to_line x y ≤ 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l297_29740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_methods_necessary_l297_29728

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents the household distribution in the urban district -/
structure HouseholdDistribution where
  total : Nat
  farmers : Nat
  workers : Nat
  intellectuals : Nat
  farmer_majority : farmers > workers + intellectuals

/-- Represents the sampling plan -/
structure SamplingPlan where
  methods : List SamplingMethod
  sample_size : Nat

/-- Checks if a sampling plan is valid for a given household distribution -/
def is_valid_sampling_plan (dist : HouseholdDistribution) (plan : SamplingPlan) : Prop :=
  plan.sample_size ≤ dist.total ∧
  plan.methods.length ≥ 1 ∧
  (dist.farmers > dist.workers + dist.intellectuals → SamplingMethod.Systematic ∈ plan.methods) ∧
  (dist.workers > 0 → SamplingMethod.SimpleRandom ∈ plan.methods) ∧
  (dist.intellectuals > 0 → SamplingMethod.SimpleRandom ∈ plan.methods) ∧
  SamplingMethod.Stratified ∈ plan.methods

/-- The main theorem stating that all three sampling methods are necessary -/
theorem all_methods_necessary (dist : HouseholdDistribution) (plan : SamplingPlan) :
  is_valid_sampling_plan dist plan →
  plan.methods = [SamplingMethod.SimpleRandom, SamplingMethod.Systematic, SamplingMethod.Stratified] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_methods_necessary_l297_29728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l297_29708

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) : 
  -- Part 1
  ((5/4 * t.c - t.a) * Real.cos t.B = t.b * Real.cos t.A ∧ 
   Real.sin t.A = 2/5 ∧ 
   t.a + t.b = 10) → 
  t.a = 4 ∧
  -- Part 2
  ((5/4 * t.c - t.a) * Real.cos t.B = t.b * Real.cos t.A ∧ 
   t.b = 3 * Real.sqrt 5 ∧ 
   t.a = 5) → 
  (1/2 * t.a * t.c * Real.sin t.B) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l297_29708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l297_29713

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * |x - a|

-- Part 1
theorem part_one :
  {x : ℝ | |x - 1/3| + f 2 x ≥ 1} = {x : ℝ | x ≤ 0 ∨ x ≥ 1} := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  (Set.Icc (1/3 : ℝ) (1/2 : ℝ)) ⊆ {x : ℝ | |x - 1/3| + f a x ≤ x} →
  -1/2 ≤ a ∧ a ≤ 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l297_29713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l297_29781

-- Define the function f(x) = |2^x - 1| - |2^x + 1|
noncomputable def f (x : ℝ) : ℝ := |Real.rpow 2 x - 1| - |Real.rpow 2 x + 1|

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x = a + 1) ↔ a ∈ Set.Icc (-3) (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l297_29781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_nine_percent_l297_29730

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time / 100

/-- Given conditions -/
noncomputable def principal : ℝ := 8925
noncomputable def time : ℝ := 5
noncomputable def total_interest : ℝ := 4016.25

/-- Theorem: The rate of interest per annum is 9% -/
theorem interest_rate_is_nine_percent : 
  ∃ (rate : ℝ), simple_interest principal rate time = total_interest ∧ rate = 9 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_nine_percent_l297_29730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_sum_zero_l297_29796

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x + 2| < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - m) * (x - 2) < 0}

-- State the theorem
theorem intersection_implies_sum_zero (m n : ℝ) : 
  A ∩ B m = Set.Ioo (-1) n → m + n = 0 := by
  sorry

-- Note: Set.Ioo represents an open interval (a, b) in Lean

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_sum_zero_l297_29796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_in_aquarium_l297_29729

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- A cube with side length 2 -/
def Cube := { p : Point3D | 0 ≤ p.x ∧ p.x ≤ 2 ∧ 0 ≤ p.y ∧ p.y ≤ 2 ∧ 0 ≤ p.z ∧ p.z ≤ 2 }

theorem fish_in_aquarium (fish : Finset Point3D) :
  fish.card = 9 → fish.toSet ⊆ Cube →
  ∃ p q : Point3D, p ∈ fish ∧ q ∈ fish ∧ p ≠ q ∧ distance p q < Real.sqrt 3 := by
  sorry

#check fish_in_aquarium

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_in_aquarium_l297_29729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cone_volume_ratio_l297_29702

theorem right_triangle_cone_volume_ratio :
  ∀ (leg1 leg2 : ℝ),
    leg1 = 1 →
    leg2 = Real.sqrt 3 →
    (1 / 3) * Real.pi * leg2^2 * leg1 / ((1 / 3) * Real.pi * leg1^2 * leg2) = Real.sqrt 3 :=
by
  intros leg1 leg2 h1 h2
  rw [h1, h2]
  norm_num
  field_simp
  ring_nf
  sorry  -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cone_volume_ratio_l297_29702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_no_minimum_l297_29799

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -2^x / (2^x + 1)

-- State the theorem
theorem f_strictly_decreasing_no_minimum :
  (∀ x y : ℝ, x < y → f x > f y) ∧ 
  (∀ m : ℝ, ∃ x : ℝ, f x < m) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_no_minimum_l297_29799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_hexagon_shaded_area_is_half_hexagon_l297_29715

/-- Given a regular hexagon with area 60 and midpoints marked on four of its sides,
    the area of the region formed by connecting these midpoints is half the hexagon's area. -/
theorem shaded_area_of_hexagon (hexagon_area : ℝ) 
  (h_area : hexagon_area = 60) : ℝ :=
by
  -- Define the shaded region
  sorry

/-- The area of the shaded region in the hexagon -/
def shaded_area : ℝ := 30

theorem shaded_area_is_half_hexagon (hexagon_area : ℝ) 
  (h_area : hexagon_area = 60) :
  shaded_area = hexagon_area / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_hexagon_shaded_area_is_half_hexagon_l297_29715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l297_29749

/-- Line passing through a point with given angle of inclination -/
structure Line where
  point : ℝ × ℝ
  angle : ℝ

/-- Circle with center at origin -/
structure Circle where
  radius : ℝ

/-- The intersection points of a line and a circle -/
def intersectionPoints (l : Line) (c : Circle) : Set (ℝ × ℝ) :=
  sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  sorry

theorem intersection_length (l : Line) (c : Circle) :
  l.point = (2, 1) →
  l.angle = π / 4 →
  c.radius = 2 →
  ∃ A B, A ∈ intersectionPoints l c ∧ B ∈ intersectionPoints l c ∧
    distance A B = Real.sqrt 14 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l297_29749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l297_29724

/-- The area of a trapezium with given parallel side lengths and height -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of 20 cm and 18 cm, 
    and height 15 cm, is 285 square centimeters -/
theorem trapezium_area_example : 
  trapeziumArea 20 18 15 = 285 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic
  simp [mul_add, mul_div_assoc]
  -- Check that the result is correct
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l297_29724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_shuffle_eventually_returns_to_original_order_l297_29767

/-- Perfect shuffle function -/
def perfectShuffle (n : ℕ) (k : ℕ) : ℕ :=
  (2 * k) % (2 * n - 1)

/-- Represents r perfect shuffles -/
def rShuffles (n : ℕ) (r : ℕ) (k : ℕ) : ℕ :=
  (2^r * k) % (2 * n - 1)

theorem perfect_shuffle_eventually_returns_to_original_order :
  ∀ n : ℕ, n > 0 → ∃ r : ℕ, r > 0 ∧ ∀ k : ℕ, k ≤ 2*n → rShuffles n r k = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_shuffle_eventually_returns_to_original_order_l297_29767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l297_29762

/-- The eccentricity of an ellipse under specific conditions -/
theorem ellipse_eccentricity (a b c : ℝ) (P : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ Set.range (λ t => (a * Real.cos t, b * Real.sin t))) ∧
  (∃ x y : ℝ, P = (x, y) ∧ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∃ x y : ℝ, P = (x, y) ∧ x^2 + y^2 = c^2) ∧
  Real.arccos ((P.1 + c) / (2 * c)) = 2 * Real.arccos ((-c - P.1) / (2 * c)) →
  c / a = Real.sqrt 3 - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l297_29762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l297_29770

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (Real.sqrt 5 * Real.cos θ, Real.sqrt 5 * Real.sin θ)
noncomputable def C₂ (t : ℝ) : ℝ × ℝ := (1 - (Real.sqrt 2 / 2) * t, -(Real.sqrt 2 / 2) * t)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(-1, -2), (2, 1)}

-- Theorem statement
theorem curves_intersection :
  ∀ (p : ℝ × ℝ), p ∈ intersection_points ↔
    (∃ θ : ℝ, C₁ θ = p) ∧ (∃ t : ℝ, C₂ t = p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l297_29770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l297_29743

theorem sin_pi_plus_alpha (α : ℝ) (k : ℝ) 
  (h1 : Real.cos α = k) 
  (h2 : α > π / 2) 
  (h3 : α < π) : 
  Real.sin (π + α) = -Real.sqrt (1 - k^2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l297_29743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_value_l297_29736

def my_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1  -- We define a₀ = 1 to align with 1-based indexing
  | n+1 => my_sequence n + 2

theorem eighth_term_value :
  my_sequence 7 = 15 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_value_l297_29736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_combined_transformation_l297_29759

-- Define the rotation angle
noncomputable def θ : Real := Real.pi / 4  -- 45 degrees in radians

-- Define the scaling factor
def k : Real := 3

-- Define the rotation matrix
noncomputable def R : Matrix (Fin 2) (Fin 2) Real := 
  Matrix.of ![
    ![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]
  ]

-- Define the scaling matrix
def K : Matrix (Fin 2) (Fin 2) Real :=
  Matrix.of ![
    ![k, 0],
    ![0, k]
  ]

-- Define the combined transformation matrix
noncomputable def S : Matrix (Fin 2) (Fin 2) Real := K * R

-- Theorem statement
theorem det_combined_transformation :
  Matrix.det S = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_combined_transformation_l297_29759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crate_height_difference_l297_29793

/-- The height difference between two crate packing methods for cylindrical pipes -/
theorem crate_height_difference (pipe_diameter : ℝ) (num_pipes : ℕ) (pipes_per_row : ℕ) :
  pipe_diameter = 12 →
  num_pipes = 200 →
  pipes_per_row = 10 →
  (num_pipes / pipes_per_row : ℝ) * pipe_diameter -
  ((num_pipes / pipes_per_row + 1 : ℝ) * (pipe_diameter / 2) +
   (num_pipes / pipes_per_row : ℝ) * (Real.sqrt 3 * pipe_diameter / 2)) =
  114 - 120 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crate_height_difference_l297_29793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_perimeter_l297_29763

/-- Given four pieces of paper A, B, C, D, prove that their total perimeter is 184 cm -/
theorem paper_perimeter (a b c d : ℝ) : 
  -- A, C, D are rectangular, B is square
  (∃ (w₁ h₁ w₃ h₃ w₄ h₄ : ℝ), a = w₁ * h₁ ∧ c = w₃ * h₃ ∧ d = w₄ * h₄) →
  (∃ (s : ℝ), b = s * s) →
  -- Total area when assembled into a large rectangle
  a + b + c + d = 480 →
  -- Areas of B, C, and D are each 3 times the area of A
  b = 3 * a ∧ c = 3 * a ∧ d = 3 * a →
  -- Total perimeter
  ∃ (p : ℝ), p = 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d) ∧ p = 184 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_perimeter_l297_29763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l297_29742

noncomputable section

open Real

def f (x : ℝ) : ℝ := 2 * sin (π / 6 - 2 * x)

theorem f_increasing_interval :
  ∀ x ∈ Set.Icc (π / 3) (5 * π / 6),
    x ∈ Set.Icc 0 π →
    ∀ y ∈ Set.Icc (π / 3) (5 * π / 6),
      x < y → f x < f y :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l297_29742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_satisfying_equation_l297_29735

theorem count_pairs_satisfying_equation : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ ↦ 
      p.1 ≥ p.2 ∧ 
      p.1 > 0 ∧ 
      p.2 > 0 ∧ 
      1 / (p.1 : ℚ) + 1 / (p.2 : ℚ) = 1 / 6)
    (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_satisfying_equation_l297_29735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_kola_volume_l297_29711

/-- Represents the composition and volume of a kola solution --/
structure KolaSolution where
  volume : ℝ
  water_percent : ℝ
  kola_percent : ℝ
  sugar_percent : ℝ

/-- Theorem stating the initial volume of the kola solution --/
theorem initial_kola_volume
  (initial : KolaSolution)
  (h1 : initial.water_percent = 88)
  (h2 : initial.kola_percent = 8)
  (h3 : initial.sugar_percent = 100 - initial.water_percent - initial.kola_percent)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_kola : ℝ)
  (h4 : added_sugar = 3.2)
  (h5 : added_water = 10)
  (h6 : added_kola = 6.8)
  (h7 : (initial.sugar_percent / 100 * initial.volume + added_sugar) / 
        (initial.volume + added_sugar + added_water + added_kola) = 0.04521739130434784) :
  ∃ (ε : ℝ), abs (initial.volume - 440) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_kola_volume_l297_29711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_solution_exists_l297_29731

/-- Represents a 3D coordinate --/
structure Coordinate where
  x : Nat
  y : Nat
  z : Nat

/-- Represents a 1 × 1 × 2 parallelepiped --/
structure SmallParallelepiped where
  base : Coordinate
  vertical : Bool

/-- Represents the large 5 × 5 × 3 parallelepiped --/
def LargeParallelepiped : Set Coordinate :=
  {c : Coordinate | c.x < 5 ∧ c.y < 5 ∧ c.z < 3}

/-- Predicate to check if a coordinate is at the center of a side face --/
def isExitPoint (c : Coordinate) : Prop :=
  (c.x = 2 ∧ c.y = 0 ∧ c.z = 1) ∨ (c.x = 2 ∧ c.y = 4 ∧ c.z = 1)

/-- The set of removed cubes --/
def RemovedCubes : Set Coordinate := sorry

/-- The remaining structure after removing cubes --/
def RemainingStructure : Set Coordinate :=
  LargeParallelepiped \ RemovedCubes

/-- Predicate to check if the cave has only two exits --/
def hasValidExits : Prop :=
  ∃ (e1 e2 : Coordinate),
    isExitPoint e1 ∧ isExitPoint e2 ∧ e1 ≠ e2 ∧
    ∀ (c : Coordinate), c ∈ RemovedCubes ∩ LargeParallelepiped → (c = e1 ∨ c = e2)

/-- Predicate to check if the remaining structure can be assembled from 1 × 1 × 2 parallelepipeds --/
def canBeAssembled : Prop :=
  ∃ (parallelepipeds : Set SmallParallelepiped),
    ∀ (c : Coordinate), c ∈ RemainingStructure ↔
      ∃ (p : SmallParallelepiped), p ∈ parallelepipeds ∧
        (c = p.base ∨ (p.vertical ∧ c = ⟨p.base.x, p.base.y, p.base.z + 1⟩))

/-- The main theorem stating that a valid solution exists --/
theorem valid_solution_exists : ∃ (RemovedCubes : Set Coordinate),
  hasValidExits ∧ canBeAssembled := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_solution_exists_l297_29731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l297_29773

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  (1/2) * abs (v.1 * w.2 - v.2 * w.1)

/-- Theorem: The area of the triangle with vertices at (1, 3), (5, -2), and (8, 6) is 23.5 -/
theorem triangle_area_example : triangleArea (1, 3) (5, -2) (8, 6) = 23.5 := by
  -- Expand the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l297_29773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_leq_one_l297_29774

theorem negation_of_forall_sin_leq_one :
  (¬ ∀ x : ℝ, x ≥ 0 → Real.sin x ≤ 1) ↔ (∃ x : ℝ, x ≥ 0 ∧ Real.sin x > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_leq_one_l297_29774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_can_give_C_125m_start_l297_29701

-- Define the race distance
noncomputable def race_distance : ℝ := 1000

-- Define the start advantages
noncomputable def A_B_start : ℝ := 100
noncomputable def A_C_start : ℝ := 200

-- Define the speeds of A, B, and C as ratios
noncomputable def speed_ratio_A_B : ℝ := race_distance / (race_distance - A_B_start)
noncomputable def speed_ratio_A_C : ℝ := race_distance / (race_distance - A_C_start)

-- Define the speed ratio of B to C
noncomputable def speed_ratio_B_C : ℝ := speed_ratio_A_B / speed_ratio_A_C

-- Define the start B can give C
noncomputable def B_C_start : ℝ := race_distance * (1 - speed_ratio_B_C)

-- Theorem statement
theorem B_can_give_C_125m_start :
  B_C_start = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_can_give_C_125m_start_l297_29701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023rd_term_l297_29776

-- Define a function to calculate the sum of cubes of digits
def sumCubesOfDigits (n : Nat) : Nat := sorry

-- Define the sequence
def sequenceterm : Nat → Nat
  | 0 => 2023
  | n + 1 => sumCubesOfDigits (sequenceterm n)

-- State the theorem
theorem sequence_2023rd_term :
  sequenceterm 2022 = 370 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023rd_term_l297_29776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_point_connections_l297_29765

theorem plane_point_connections (m : ℕ) (l : ℕ) 
  (h1 : m > 0) 
  (h2 : ∀ p : Fin m, ∃ (connected : Finset (Fin m)), 
    Finset.card connected = l ∧ p ∉ connected ∧ 
    ∀ q ∈ connected, ∃ (segment : Fin m × Fin m), 
      segment = (p, q) ∨ segment = (q, p)) :
  1 ≤ l ∧ l < m ∧ Even (m * l) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_point_connections_l297_29765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l297_29791

def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

theorem function_properties (a b c : ℝ) :
  f a b c 0 = 1 ∧ 
  (∃ (k : ℝ), ∀ x, f a b c x = k * (x - 1) + f a b c 1) →
  a = 5/2 ∧ c = 1 ∧
  (∀ x, (x > -3 * Real.sqrt 10 / 10 ∧ x < 0) ∨ (x > 3 * Real.sqrt 10 / 10) →
    ∃ h > 0, ∀ y ∈ Set.Ioo (x - h) (x + h), f a b c y > f a b c x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l297_29791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_quadrilateral_l297_29707

/-- The area of a quadrilateral given its vertices -/
noncomputable def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  let (x4, y4) := v4
  (1/2) * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

theorem area_of_specific_quadrilateral :
  quadrilateralArea (2, 1) (1, 6) (4, 5) (9, 9) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_quadrilateral_l297_29707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_queue_length_reduction_l297_29734

/-- The number of people in the queue -/
def num_people : ℕ := 11

/-- The radius of an umbrella in centimeters -/
noncomputable def umbrella_radius : ℝ := 50

/-- The spacing between people after closing umbrellas in centimeters -/
noncomputable def spacing : ℝ := 50

/-- The initial length of the queue with open umbrellas -/
noncomputable def initial_length : ℝ := num_people * (2 * umbrella_radius)

/-- The final length of the queue with closed umbrellas -/
noncomputable def final_length : ℝ := (num_people - 1) * spacing

/-- The ratio of initial length to final length -/
noncomputable def length_ratio : ℝ := initial_length / final_length

theorem queue_length_reduction : length_ratio = 2.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_queue_length_reduction_l297_29734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_intersections_is_80_l297_29745

/-- Represents a regular polygon inscribed in a circle -/
structure InscribedPolygon where
  sides : Nat
  deriving Repr

/-- Counts the number of intersections between two inscribed polygons -/
def countIntersections (p1 p2 : InscribedPolygon) : Nat :=
  2 * min p1.sides p2.sides

/-- The set of inscribed polygons in our problem -/
def polygons : List InscribedPolygon :=
  [⟨6⟩, ⟨7⟩, ⟨8⟩, ⟨9⟩]

/-- Theorem: The total number of intersection points is 80 -/
theorem total_intersections_is_80 :
  (List.sum (List.join (List.map (fun p1 =>
    List.map (fun p2 =>
      if p1.sides < p2.sides then countIntersections p1 p2 else 0
    ) polygons
  ) polygons))) = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_intersections_is_80_l297_29745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_circle_tangent_l297_29717

theorem right_triangle_circle_tangent (X Y Z Q O : ℝ × ℝ) (r : ℝ) : 
  -- XYZ is a right triangle with right angle at Y
  (Y.1 - X.1) * (Z.2 - Y.2) = (Y.2 - X.2) * (Z.1 - Y.1) →
  -- XZ = √85
  Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2) = Real.sqrt 85 →
  -- XY = 7
  Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) = 7 →
  -- O is the center of the circle on XY
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ O = (X.1 + t * (Y.1 - X.1), X.2 + t * (Y.2 - X.2))) →
  -- Circle is tangent to XZ and YZ
  Real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2) = r ∧ 
  Real.sqrt ((Z.1 - O.1)^2 + (Z.2 - O.2)^2) = r →
  -- Q is on XZ
  (∃ s : ℝ, 0 < s ∧ s < 1 ∧ Q = (X.1 + s * (Z.1 - X.1), X.2 + s * (Z.2 - X.2))) →
  -- ZQ = 6
  Real.sqrt ((Z.1 - Q.1)^2 + (Z.2 - Q.2)^2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_circle_tangent_l297_29717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_given_p_q_condition_l297_29722

/-- Given statements p and q, and the condition that ¬p is a necessary but not sufficient
    condition for ¬q, prove that the range of values for a is [0, 1/2]. -/
theorem range_of_a_given_p_q_condition (a : ℝ) :
  (∀ x : ℝ, (x^2 - (2 * a + 1) * x + a * (a + 1) > 0) → ((4 * x - 3)^2 > 1)) ∧
  (∃ x : ℝ, ((4 * x - 3)^2 > 1) ∧ (x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0)) →
  0 ≤ a ∧ a ≤ 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_given_p_q_condition_l297_29722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_is_pi_l297_29769

/-- A function representing a cosine wave with amplitude 3, frequency 2, and phase shift φ. -/
noncomputable def f (φ : ℝ) : ℝ → ℝ := λ x ↦ 3 * Real.cos (2 * x + φ)

/-- Symmetry condition: the function is symmetric about a point if there exists an x₀ such that
    f(x₀ + x) = f(x₀ - x) for all x -/
def is_symmetric_about_point (g : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, ∀ x : ℝ, g (x₀ + x) = g (x₀ - x)

/-- The main theorem: if f(x) is symmetric about a point, then the minimum φ is π -/
theorem min_phi_is_pi :
  (∃ φ : ℝ, is_symmetric_about_point (f φ)) →
  (∃ φ_min : ℝ, φ_min = Real.pi ∧
    ∀ φ : ℝ, is_symmetric_about_point (f φ) → φ_min ≤ φ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_is_pi_l297_29769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_connectivity_l297_29719

/-- Represents a diagonal in a 1×1 square -/
inductive Diagonal
| NorthEast
| NorthWest

/-- Represents a grid of m×n squares, each containing a diagonal -/
def Grid (m n : ℕ) := Fin m → Fin n → Diagonal

/-- Represents a path in the grid -/
inductive GridPath (m n : ℕ)
| Start : GridPath m n
| Move : GridPath m n → Fin m → Fin n → GridPath m n

/-- Checks if a path connects the left and right sides of the grid -/
def connects_left_right (m n : ℕ) (p : GridPath m n) : Prop := sorry

/-- Checks if a path connects the top and bottom sides of the grid -/
def connects_top_bottom (m n : ℕ) (p : GridPath m n) : Prop := sorry

/-- Main theorem: There exists a path connecting either left-right or top-bottom -/
theorem diagonal_connectivity (m n : ℕ) (grid : Grid m n) :
  (∃ p : GridPath m n, connects_left_right m n p) ∨ 
  (∃ p : GridPath m n, connects_top_bottom m n p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_connectivity_l297_29719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_k_l297_29785

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (3 * x + 4) / (k * x - 3)

theorem no_valid_k : ¬∃ k : ℝ, 
  (∀ x : ℝ, f k x = (3 * x + 4) / (k * x - 3)) ∧ 
  (∀ x : ℝ, x ≠ 3/k → f k (f k x) = x) ∧ 
  (f k 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_k_l297_29785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_l297_29764

/-- The cost of fencing around a circular field -/
theorem fencing_cost (d : ℝ) (rate : ℝ) (h1 : d = 42) (h2 : rate = 3) :
  ∃ (cost : ℝ), |cost - (Real.pi * d * rate)| < 0.01 ∧ |cost - 395.85| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_l297_29764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_y_coord_l297_29738

/-- 
Given an equilateral triangle with two vertices at (0,6) and (10,6),
and the third vertex in the first quadrant, 
the y-coordinate of the third vertex is 6 + 5√3.
-/
theorem equilateral_triangle_third_vertex_y_coord : 
  ∀ (x y : ℝ),
  let A : ℝ × ℝ := (0, 6)
  let B : ℝ × ℝ := (10, 6)
  let C : ℝ × ℝ := (x, y)
  (‖A - B‖ = ‖B - C‖) ∧ 
  (‖B - C‖ = ‖C - A‖) ∧ 
  (x ≥ 0) ∧ (y ≥ 0) →
  y = 6 + 5 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_y_coord_l297_29738
