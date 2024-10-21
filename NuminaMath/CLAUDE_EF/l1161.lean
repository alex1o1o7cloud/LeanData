import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1161_116135

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp (1/2 * x)

-- Define the point of tangency
noncomputable def point : ℝ × ℝ := (4, Real.exp 2)

-- State the theorem
theorem tangent_triangle_area :
  let tangent_line (x : ℝ) := (Real.exp 2 / 2) * x - Real.exp 2
  let x_intercept := 2
  let y_intercept := Real.exp 2
  (1/2) * x_intercept * y_intercept = Real.exp 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1161_116135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_c_l1161_116188

/-- The function c(x) parameterized by k -/
noncomputable def c (k : ℝ) (x : ℝ) : ℝ := (3*k*x^2 - 4*x + 1) / (-3*x^2 - 4*x + k)

/-- The domain of c(x) is all real numbers iff k < -4/3 -/
theorem domain_of_c (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, c k x = y) ↔ k < -4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_c_l1161_116188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peak_on_november_12_l1161_116118

/-- Represents the day number in November (1-30) -/
def Day := Fin 30

/-- The number of new infections on a given day -/
def new_infections : Day → ℕ := sorry

/-- The total number of infections over the 30-day period -/
def total_infections : ℕ := sorry

/-- The day when the number of new infections peaked -/
def peak_day : Day := sorry

/-- Conditions of the problem -/
axiom initial_infections : new_infections ⟨1, by norm_num⟩ = 20
axiom total_infected : total_infections = 8670
axiom increasing_phase (d : Day) : d.val < peak_day.val → new_infections ⟨d.val + 1, sorry⟩ = new_infections d + 50
axiom decreasing_phase (d : Day) : d.val > peak_day.val → new_infections ⟨d.val + 1, sorry⟩ = new_infections d - 30

/-- The theorem to prove -/
theorem peak_on_november_12 : peak_day = ⟨12, by norm_num⟩ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peak_on_november_12_l1161_116118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hellys_theorem_l1161_116136

/-- A type representing a convex shape on a plane -/
structure ConvexShape where
  -- We'll use a set of points to represent the shape
  points : Set (ℝ × ℝ)
  -- Add convexity property (this is a placeholder and should be properly defined)
  convex : True

/-- Membership instance for ConvexShape -/
instance : Membership (ℝ × ℝ) ConvexShape where
  mem p s := p ∈ s.points

/-- Given n convex shapes on a plane, if any three have a common point, 
    then all n shapes have a common point (Helly's theorem) -/
theorem hellys_theorem {n : ℕ} (shapes : Fin n → ConvexShape) 
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    ∃ (p : ℝ × ℝ), p ∈ shapes i ∧ p ∈ shapes j ∧ p ∈ shapes k) :
  ∃ (p : ℝ × ℝ), ∀ (i : Fin n), p ∈ shapes i := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hellys_theorem_l1161_116136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_direct_age_height_correlation_l1161_116107

-- Define the residents
inductive Resident : Type
| A : Resident
| B : Resident
| C : Resident

-- Define the age relation
def olderThan : Resident → Resident → Prop := sorry

-- Define the height relation
def tallerThan : Resident → Resident → Prop := sorry

-- Define the statement relation
def states : Resident → (Resident → Resident → Prop) → Resident → Resident → Prop := sorry

-- Axioms based on the problem conditions
axiom younger_dont_contradict (x y : Resident) : 
  olderThan y x → ¬(states x tallerThan y x ∧ states y tallerThan x y)

axiom elders_not_wrong (x y : Resident) :
  olderThan x y → states x tallerThan y x → tallerThan y x

-- Given statements
axiom statement_A : states Resident.A tallerThan Resident.B Resident.A
axiom statement_B : states Resident.B tallerThan Resident.A Resident.B
axiom statement_C : states Resident.C tallerThan Resident.C Resident.B

-- Theorem to prove
theorem no_direct_age_height_correlation :
  ¬(∀ (x y : Resident), olderThan x y → tallerThan y x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_direct_age_height_correlation_l1161_116107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_fibonacci_is_eight_l1161_116191

def fibonacci_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci_sequence n + fibonacci_sequence (n + 1)

theorem sixth_fibonacci_is_eight :
  fibonacci_sequence 5 = 8 := by
  rw [fibonacci_sequence]
  rw [fibonacci_sequence]
  rw [fibonacci_sequence]
  rw [fibonacci_sequence]
  rw [fibonacci_sequence]
  rfl

#eval fibonacci_sequence 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_fibonacci_is_eight_l1161_116191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_parallelogram_triangle_sides_l1161_116123

/-- A parallelogram inscribed in a triangle -/
structure InscribedParallelogram where
  /-- Length of the shorter side of the parallelogram -/
  short_side : ℝ
  /-- Length of the longer side of the parallelogram -/
  long_side : ℝ
  /-- Length of one diagonal of the parallelogram -/
  diagonal : ℝ
  /-- The shorter side is indeed shorter than the longer side -/
  short_side_shorter : short_side < long_side
  /-- The diagonal is longer than both sides -/
  diagonal_longest : diagonal > long_side ∧ diagonal > short_side

/-- The sides of the triangle containing the inscribed parallelogram -/
noncomputable def triangle_sides (p : InscribedParallelogram) : (ℝ × ℝ × ℝ) :=
  (9, 9, 6 * Real.sqrt 2)

/-- Theorem stating the relationship between the inscribed parallelogram and the triangle sides -/
theorem inscribed_parallelogram_triangle_sides 
  (p : InscribedParallelogram) 
  (h1 : p.short_side = 3) 
  (h2 : p.long_side = 5) 
  (h3 : p.diagonal = 6) : 
  triangle_sides p = (9, 9, 6 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_parallelogram_triangle_sides_l1161_116123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1161_116130

/-- Given functions f and g, we define G and explore its properties. -/
theorem function_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ m * x + 3
  let g : ℝ → ℝ := λ x ↦ x^2 + 2*x + m
  let G : ℝ → ℝ := λ x ↦ f x - g x - 1

  -- Part 1: f(x) - g(x) has a zero point
  (∃ x, f x = g x) ∧

  -- Part 2: Range of m when |G(x)| is decreasing on [-1, 0]
  (∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → |G x| ≥ |G y| → m ∈ Set.Icc 2 6 ∪ Set.Iic 0) ∧

  -- Part 3: Existence of integers a and b such that a ≤ G(x) ≤ b is exactly [a, b]
  ((∃ a b : ℤ, (∀ x : ℝ, (↑a : ℝ) ≤ G x ∧ G x ≤ (↑b : ℝ) ↔ x ∈ Set.Icc (↑a : ℝ) (↑b : ℝ)) ∧
    ((a = -1 ∧ b = 1) ∨ (a = 2 ∧ b = 4))) ∨
   (¬∃ a b : ℤ, ∀ x : ℝ, (↑a : ℝ) ≤ G x ∧ G x ≤ (↑b : ℝ) ↔ x ∈ Set.Icc (↑a : ℝ) (↑b : ℝ)))
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1161_116130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_circles_area_l1161_116141

/-- The area of the overlapping region of two circles with given radii and intersection angle -/
noncomputable def overlapping_area (r₁ r₂ : ℝ) (θ : ℝ) : ℝ :=
  let sector_area := min (r₁^2 * θ / 2) (r₂^2 * θ / 2)
  let triangle_area := r₁ * r₂ * Real.sin θ / 2
  (sector_area - triangle_area) / 2

/-- Theorem stating the area of the overlapping region for specific circle radii and angle -/
theorem overlapping_circles_area :
  overlapping_area 5 3 (2 * Real.pi / 3) = 9 * Real.pi / 6 - 15 * Real.sqrt 3 / 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_circles_area_l1161_116141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AD_between_44_and_45_l1161_116159

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
axiom B_east_of_A : A.1 < B.1 ∧ A.2 = B.2
axiom C_north_of_B : B.1 = C.1 ∧ B.2 < C.2
axiom AC_distance : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 15 * Real.sqrt 2
axiom BAC_angle : Real.arctan ((C.2 - A.2) / (C.1 - A.1)) = 30 * Real.pi / 180
axiom D_north_of_C : C.1 = D.1 ∧ D.2 - C.2 = 25

-- Define the distance AD
noncomputable def distance_AD (A D : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)

-- Theorem to prove
theorem AD_between_44_and_45 (A B C D : ℝ × ℝ) : 
  44 < distance_AD A D ∧ distance_AD A D < 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_AD_between_44_and_45_l1161_116159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_divisor_of_50_factorial_l1161_116195

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

theorem smallest_non_divisor_of_50_factorial :
  ∃ (k : ℕ), k = 53 ∧ ¬(k ∣ factorial 50) ∧ ∀ m < k, m ∣ factorial 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_divisor_of_50_factorial_l1161_116195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_of_30_l1161_116116

def factors (n : ℕ) : Finset ℕ := Finset.filter (λ d => n % d = 0) (Finset.range (n + 1) \ {0})

theorem sum_of_factors_of_30 : (factors 30).sum id = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_of_30_l1161_116116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_neg_x_implies_a_geq_one_l1161_116113

-- Define the function f(x) = ln(x) - ax^2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2

-- Theorem statement
theorem f_leq_neg_x_implies_a_geq_one :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → f a x ≤ -x) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_neg_x_implies_a_geq_one_l1161_116113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1161_116190

-- Define the functions f and g as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + x^2
noncomputable def g (x : ℝ) : ℝ := x^3 - x^2 - 3

-- Define the theorem
theorem function_inequality (a : ℝ) :
  (∀ s t : ℝ, s ∈ Set.Icc (1/2 : ℝ) 2 → t ∈ Set.Icc (1/2 : ℝ) 2 → f a s ≥ g t) →
  a ≥ 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1161_116190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_one_third_l1161_116109

noncomputable def f (x : ℝ) : ℝ := (6 * x^2 + x - 1) / (x - 1/3)

theorem limit_of_f_at_one_third (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1/3| → |x - 1/3| < δ → |f x - 5| < ε ∧ δ = ε/6 :=
by
  -- We choose δ = ε/6
  let δ := ε/6
  
  -- Show that δ > 0
  have hδ : δ > 0 := by
    apply div_pos hε
    norm_num
  
  -- Provide δ and prove it satisfies the conditions
  use δ
  constructor
  · exact hδ
  · intro x hx_nonzero hx_delta
    constructor
    · sorry -- Proof of |f x - 5| < ε goes here
    · rfl -- δ = ε/6 by definition


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_one_third_l1161_116109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_for_equation_l1161_116106

theorem solutions_count_for_equation : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ ↦ p.1 + p.2 + 2 * p.1 * p.2 = 2023) (Finset.product (Finset.range 2024) (Finset.range 2024))).card ∧ n = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_for_equation_l1161_116106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1161_116132

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  Real.sin (A - π/6) = Real.cos A →
  a = 1 →
  b + c = 2 →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  (A = π/3 ∧ (1/2) * b * c * (Real.sin A) = Real.sqrt 3/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1161_116132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_lt_id_lt_tan_l1161_116152

theorem sin_lt_id_lt_tan {α : ℝ} (h : α ∈ Set.Ioo 0 (π / 2)) : Real.sin α < α ∧ α < Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_lt_id_lt_tan_l1161_116152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_f_min_f_extrema_l1161_116187

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6)

-- Theorem for the maximum value
theorem f_max (k : ℤ) : f (2 * Real.pi * (k : ℝ) + Real.pi / 3) = 1 := by sorry

-- Theorem for the minimum value
theorem f_min (k : ℤ) : f (2 * Real.pi * (k : ℝ) - 2 * Real.pi / 3) = -1 := by sorry

-- Theorem stating that these are indeed the maximum and minimum values
theorem f_extrema :
  (∀ x : ℝ, f x ≤ 1) ∧
  (∀ x : ℝ, f x ≥ -1) ∧
  (∃ k : ℤ, f (2 * Real.pi * (k : ℝ) + Real.pi / 3) = 1) ∧
  (∃ k : ℤ, f (2 * Real.pi * (k : ℝ) - 2 * Real.pi / 3) = -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_f_min_f_extrema_l1161_116187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_watermelons_weight_l1161_116173

/-- The weight of a watermelon in grams -/
def watermelon_weight : ℕ → Prop := sorry

/-- The weight of a pineapple in grams -/
def pineapple_weight : ℕ → Prop := sorry

/-- The relationship between watermelon and pineapple weights -/
axiom weight_difference : ∀ w p : ℕ, watermelon_weight w ∧ pineapple_weight p → w = p + 850

/-- The total weight of 3 watermelons and 4 pineapples -/
axiom total_weight : ∀ w p : ℕ, watermelon_weight w ∧ pineapple_weight p → 3 * w + 4 * p = 5700

theorem four_watermelons_weight :
  ∃ w : ℕ, watermelon_weight w ∧ 4 * w = 5200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_watermelons_weight_l1161_116173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1161_116168

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def sum_of_geometric_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i ↦ a i)

def arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (let S := sum_of_geometric_sequence a
   arithmetic_progression (S 4) (S 2) (S 3)) →
  a 1 + a 2 + a 3 = -18 →
  (∀ n : ℕ, a n = 3 * (-2)^(n - 1)) ∧
  (∃ n : ℕ, sum_of_geometric_sequence a n ≥ 2013) ∧
  (∀ n : ℕ, sum_of_geometric_sequence a n ≥ 2013 ↔ ∃ k : ℕ, k ≥ 5 ∧ n = 2 * k + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1161_116168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_is_forty_percent_l1161_116172

-- Define the total number of passengers
variable (P : ℝ)

-- Define the percentage of passengers with round-trip tickets
def round_trip_percentage (P : ℝ) : ℝ := 
  -- 20% of all passengers held round-trip tickets and took their cars
  let with_car := 0.2 * P
  -- 50% of passengers with round-trip tickets did not take their cars
  let without_car := with_car
  -- Total passengers with round-trip tickets
  with_car + without_car

-- Theorem statement
theorem round_trip_is_forty_percent (P : ℝ) (h : P > 0) : 
  round_trip_percentage P = 0.4 * P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_is_forty_percent_l1161_116172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1161_116111

theorem power_equation_solution (x : ℝ) : (8 : ℝ)^x = 2^9 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1161_116111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_almond_peanut_cost_ratio_l1161_116114

/-- The cost of a jar of peanut butter in dollars -/
noncomputable def peanut_butter_cost : ℚ := 3

/-- The fraction of a jar needed to make a batch of cookies -/
noncomputable def jar_fraction_per_batch : ℚ := 1/2

/-- The additional cost per batch for almond butter cookies compared to peanut butter cookies -/
noncomputable def additional_cost_per_batch : ℚ := 3

/-- The cost of a jar of almond butter in dollars -/
noncomputable def almond_butter_cost : ℚ := 2 * (jar_fraction_per_batch * peanut_butter_cost + additional_cost_per_batch)

/-- The ratio of almond butter cost to peanut butter cost -/
noncomputable def cost_ratio : ℚ := almond_butter_cost / peanut_butter_cost

theorem almond_peanut_cost_ratio :
  cost_ratio = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_almond_peanut_cost_ratio_l1161_116114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_characterization_l1161_116134

/-- A quadratic polynomial -/
noncomputable def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The reciprocal function -/
noncomputable def ReciprocalFunction (f : ℝ → ℝ) : ℝ → ℝ := fun x ↦ 1 / (f x)

/-- Theorem: Characterization of a specific quadratic polynomial -/
theorem quadratic_polynomial_characterization 
  (q : ℝ → ℝ) 
  (h_quad : ∃ a b c : ℝ, q = QuadraticPolynomial a b c) 
  (h_asym_neg2 : ∀ ε > 0, ∃ δ > 0, ∀ x, |x + 2| < δ → |(ReciprocalFunction q) x| > 1/ε)
  (h_asym_3 : ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |(ReciprocalFunction q) x| > 1/ε)
  (h_point : (ReciprocalFunction q) 4 = 1/18) :
  q = QuadraticPolynomial 3 (-3) (-18) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_characterization_l1161_116134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_preserving_cross_ratio_is_projective_l1161_116102

/-- A line in a projective space -/
structure ProjectiveLine where
  points : Type
  -- Add other necessary fields here

/-- A mapping between projective lines -/
structure ProjectiveLineMapping (a b : ProjectiveLine) where
  to_fun : a.points → b.points
  -- Add necessary properties here

/-- The cross-ratio of four points on a projective line -/
noncomputable def cross_ratio (l : ProjectiveLine) (p q r s : l.points) : ℝ :=
  sorry

/-- A mapping is projective if it's a bijection that preserves cross-ratios -/
def is_projective {a b : ProjectiveLine} (f : ProjectiveLineMapping a b) : Prop :=
  Function.Bijective f.to_fun ∧
  ∀ (p q r s : a.points), cross_ratio a p q r s = cross_ratio b (f.to_fun p) (f.to_fun q) (f.to_fun r) (f.to_fun s)

/-- Main theorem: A mapping that preserves cross-ratios is projective -/
theorem mapping_preserving_cross_ratio_is_projective
  {a b : ProjectiveLine} (f : ProjectiveLineMapping a b)
  (h : ∀ (p q r s : a.points), cross_ratio a p q r s = cross_ratio b (f.to_fun p) (f.to_fun q) (f.to_fun r) (f.to_fun s)) :
  is_projective f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_preserving_cross_ratio_is_projective_l1161_116102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airline_route_within_republic_l1161_116126

/-- Represents a city in the country -/
structure City where
  id : Nat
  republic : Nat
  routes : Nat

/-- Represents the country with its cities and airline routes -/
structure Country where
  cities : Finset City
  routes : Finset (City × City)

/-- The theorem to be proved -/
theorem airline_route_within_republic (country : Country) :
  (country.cities.card = 100) →
  (∃ r1 r2 r3, ∀ c ∈ country.cities, c.republic = r1 ∨ c.republic = r2 ∨ c.republic = r3) →
  (country.cities.filter (λ c => c.routes ≥ 70)).card ≥ 70 →
  ∃ c1 c2, c1 ∈ country.cities ∧ c2 ∈ country.cities ∧ c1.republic = c2.republic ∧ (c1, c2) ∈ country.routes := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airline_route_within_republic_l1161_116126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_four_ninths_l1161_116164

/-- The sum of the series k/(4^k) from k=1 to infinity -/
noncomputable def series_sum : ℝ := ∑' k, k / (4 ^ k)

/-- Theorem: The sum of the series k/(4^k) from k=1 to infinity equals 4/9 -/
theorem series_sum_equals_four_ninths : series_sum = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_four_ninths_l1161_116164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1161_116160

theorem right_triangle_area (x : ℝ) (h : x > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b = Real.sqrt 3 * a ∧ a^2 + b^2 = x^2 ∧
  (1/2 * a * b) = (Real.sqrt 3 * x^2) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1161_116160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_beta_l1161_116158

theorem sin_alpha_plus_beta (α β : ℝ) 
  (h1 : π/4 < α ∧ α < 3*π/4)
  (h2 : 0 < β ∧ β < π/4)
  (h3 : Real.sin (α + π/4) = 3/5)
  (h4 : Real.cos (π/4 + β) = 5/13) :
  Real.sin (α + β) = 56/65 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_beta_l1161_116158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_distance_properties_l1161_116150

/-- Polynomial satisfying the distance-preserving property -/
def IsDistancePreserving (P : Polynomial ℝ) (a : Fin m → ℝ) : Prop :=
  ∀ i j, i < j → |P.eval (a i) - P.eval (a j)| = |a i - a j|

/-- Polynomial satisfying the distance-reducing property -/
def IsDistanceReducing (Q : Polynomial ℝ) (a : Fin m → ℝ) : Prop :=
  ∀ i j, i < j → |Q.eval (a i) - Q.eval (a j)| < |a i - a j|

/-- Main theorem combining both parts of the problem -/
theorem polynomial_distance_properties
  (n m : ℕ+) (hn : n < m) (a : Fin m → ℝ) (ha : Function.Injective a) :
  (∃ (P : Polynomial ℝ) (c d : ℝ), P.degree ≤ n ∧ IsDistancePreserving P a ∧
    ((P = Polynomial.X + Polynomial.C c) ∨ (P = -Polynomial.X + Polynomial.C d))) ∧
  (2 ≤ n → 2 ≤ m → ∃ (Q : Polynomial ℝ) (c : ℝ), c > 0 ∧ Q.degree = n ∧
    IsDistanceReducing Q a ∧ Q = Polynomial.monomial n c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_distance_properties_l1161_116150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_function_inequality_l1161_116133

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a * Real.log x

noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 1) / x - 1

theorem tangent_line_and_function_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ = 0 ∧ (deriv (f a)) x₀ = 0) →
  (a > 0 →
    (∀ x₁ x₂ : ℝ, x₁ ≥ 3 → x₂ ≥ 3 → x₁ ≠ x₂ →
      |f a x₁ - f a x₂| < |g x₁ - g x₂|)) →
  a = -Real.exp 1 ∨ (a > 0 ∧ a ≤ 2 * Real.exp 2 / 3 - 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_function_inequality_l1161_116133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1161_116175

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.a * (1 - g.r^n) / (1 - g.r)

/-- Theorem stating the property of the specific geometric sequence -/
theorem geometric_sequence_property (g : GeometricSequence) :
  geometricSum g 20 = 30 ∧ geometricSum g 30 = 70 →
  geometricSum g 10 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1161_116175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l1161_116145

theorem sin_plus_cos_value (θ : ℝ) (x : ℝ) :
  x ≠ 0 ∧
  (∃ y : ℝ, y = -1 ∧ x = Real.tan θ * y) ∧
  Real.tan θ = -x →
  Real.sin θ + Real.cos θ = 0 ∨ Real.sin θ + Real.cos θ = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l1161_116145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nthSum_formula_l1161_116169

/-- The sum of the first k positive integers -/
def S (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The n-th sum in the sequence 1, 2+3, 4+5+6, 7+8+9+10, ... -/
def nthSum (n : ℕ) : ℚ :=
  (S (n * (n + 1) / 2) - S (n * (n - 1) / 2)) / 1

theorem nthSum_formula (n : ℕ) : nthSum n = n^3 / 2 + n / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nthSum_formula_l1161_116169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_candidates_count_l1161_116112

theorem exam_candidates_count :
  ∀ (total_candidates : ℕ) (boys : ℕ),
    -- Given conditions
    total_candidates = boys + 900 →
    (0.28 * (boys : ℝ) + 0.32 * 900) / (total_candidates : ℝ) = 0.298 →
    -- Conclusion
    total_candidates = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_candidates_count_l1161_116112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_term_l1161_116165

/-- An arithmetic sequence with first three terms a-1, a+1, and 2a+3 -/
def arithmetic_sequence (a : ℝ) : ℕ → ℝ := sorry

/-- The first term of the sequence is a-1 -/
axiom first_term (a : ℝ) : arithmetic_sequence a 1 = a - 1

/-- The second term of the sequence is a+1 -/
axiom second_term (a : ℝ) : arithmetic_sequence a 2 = a + 1

/-- The third term of the sequence is 2a+3 -/
axiom third_term (a : ℝ) : arithmetic_sequence a 3 = 2*a + 3

/-- The sequence is arithmetic, so the difference between consecutive terms is constant -/
axiom is_arithmetic (a : ℝ) (n : ℕ) : 
  arithmetic_sequence a (n + 1) - arithmetic_sequence a n = 
  arithmetic_sequence a 2 - arithmetic_sequence a 1

/-- Theorem: The n-th term of the sequence is 2n-3 -/
theorem nth_term (a : ℝ) (n : ℕ) : arithmetic_sequence a n = 2 * n - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_term_l1161_116165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1161_116182

/-- Parabola struct representing y² = 2px -/
structure Parabola where
  p : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Distance between two points -/
noncomputable def distance (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Theorem about parabola properties -/
theorem parabola_properties (C : Parabola) (M A B T : Point) (l : Line) :
  M.x = 1 ∧ M.y = 2 ∧  -- M(1,2) is on the parabola
  M.y^2 = 2 * C.p * M.x ∧  -- M satisfies parabola equation
  T.x = 0 ∧ T.y = 1 ∧  -- T(0,1) is given
  (A.y - M.y) / (A.x - M.x) + (B.y - M.y) / (B.x - M.x) = 0 ∧  -- MA and MB slopes are complementary
  A.y = l.slope * A.x + l.intercept ∧  -- A is on line l
  B.y = l.slope * B.x + l.intercept ∧  -- B is on line l
  A.y^2 = 2 * C.p * A.x ∧  -- A is on parabola C
  B.y^2 = 2 * C.p * B.x  -- B is on parabola C
  →
  C.p = 2 ∧  -- Implies p = 2
  (λ x : ℝ => x = -1) = (λ x : ℝ => x = -C.p / 2) ∧  -- Directrix equation is x = -1
  distance T A * distance T B = 2  -- |TA| * |TB| = 2
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1161_116182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_product_l1161_116155

theorem consecutive_integer_product :
  ∀ a : ℕ,
  (∃ i j k l m n : ℕ,
    i ∈ Finset.range 6 ∧
    j ∈ Finset.range 6 ∧
    k ∈ Finset.range 6 ∧
    l ∈ Finset.range 6 ∧
    m ∈ Finset.range 6 ∧
    n ∈ Finset.range 6 ∧
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ i ≠ n ∧
    j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ j ≠ n ∧
    k ≠ l ∧ k ≠ m ∧ k ≠ n ∧
    l ≠ m ∧ l ≠ n ∧
    m ≠ n ∧
    (a + i) * (a + j) + (a + k) * (a + l) = (a + m) * (a + n)) ↔ a = 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_product_l1161_116155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_sin_is_odd_l1161_116110

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10
noncomputable def g (x : ℝ) : ℝ := Real.cos x
def h (x : ℝ) : ℝ := abs x
noncomputable def k (x : ℝ) : ℝ := Real.sin x

-- Define the property of being an odd function
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statement
theorem only_sin_is_odd :
  ¬ isOdd f ∧ ¬ isOdd g ∧ ¬ isOdd h ∧ isOdd k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_sin_is_odd_l1161_116110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_theorem_l1161_116194

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem smallest_difference_theorem (p q r s : ℕ+) : 
  p < q → q < r → r < s → p * q * r * s = factorial 9 → 
  (∀ (p' q' r' s' : ℕ+), p' < q' → q' < r' → r' < s' → p' * q' * r' * s' = factorial 9 → s - p ≤ s' - p') →
  s - p = 52 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_theorem_l1161_116194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l1161_116124

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 6)

theorem monotone_increasing_interval 
  (ω : ℝ) 
  (h_pos : ω > 0) 
  (h_dist : (2 * Real.pi) / ω = Real.pi) : 
  StrictMonoOn (f ω) (Set.Icc (-Real.pi/6) (Real.pi/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l1161_116124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_tripled_l1161_116156

/-- The radius of a circle that triples in area when increased by n -/
noncomputable def radius_tripled_area (n : ℝ) : ℝ :=
  n * (Real.sqrt 3 - 1) / 2

/-- Theorem: If the area of a circle is tripled when its radius r is increased by n,
    then r equals n(√3 - 1) / 2 -/
theorem circle_area_tripled (r n : ℝ) (h : r > 0) (hn : n > 0) :
  π * (r + n)^2 = 3 * π * r^2 → r = radius_tripled_area n := by
  sorry

#check circle_area_tripled

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_tripled_l1161_116156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_jogger_l1161_116174

/-- Time taken for a train to pass a jogger --/
theorem train_passing_jogger (jogger_speed train_speed train_length initial_distance : ℝ) : 
  jogger_speed = 9 / 3.6 →
  train_speed = 45 / 3.6 →
  train_length = 130 →
  initial_distance = 240 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 37 :=
by
  -- Introduce the hypotheses
  intro h1 h2 h3 h4

  -- Calculate the relative speed
  have relative_speed : ℝ := train_speed - jogger_speed

  -- Calculate the total distance to be covered
  have total_distance : ℝ := initial_distance + train_length

  -- Calculate the time taken
  have time_taken : ℝ := total_distance / relative_speed

  -- Prove that the time taken is 37 seconds
  sorry

-- The actual proof is omitted and replaced with 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_jogger_l1161_116174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_not_sufficient_nor_necessary_for_sine_sum_l1161_116138

theorem angle_sum_not_sufficient_nor_necessary_for_sine_sum :
  ¬(∀ α β : Real, α + β = 90 → Real.sin α + Real.sin β > 1) ∧
  ¬(∀ α β : Real, Real.sin α + Real.sin β > 1 → α + β = 90) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_not_sufficient_nor_necessary_for_sine_sum_l1161_116138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1161_116171

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x φ : Real) : Real :=
  3 * Real.sqrt 2 * Real.cos (x + φ) + Real.sin x

-- State the theorem
theorem min_value_of_f (φ : Real) 
  (h1 : φ > -Real.pi/2 ∧ φ < Real.pi/2) 
  (h2 : f (Real.pi/2) φ = 4) :
  ∃ (x : Real), ∀ (y : Real), f y φ ≥ -5 ∧ f x φ = -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1161_116171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1161_116139

-- Define the hyperbola
def is_hyperbola (x y a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the foci
def are_foci (F₁ F₂ : ℝ × ℝ) (a b : ℝ) : Prop :=
  F₁.1 = -F₂.1 ∧ F₁.1^2 = a^2 + b^2

-- Define a point on the hyperbola
def on_hyperbola (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  is_hyperbola P.1 P.2 a b

-- Define right angle condition
def right_angle (F₁ P F₂ : ℝ × ℝ) : Prop :=
  (F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2) = 0

-- Define arithmetic sequence condition for triangle sides
noncomputable def arithmetic_sequence (F₁ P F₂ : ℝ × ℝ) : Prop :=
  ∃ d : ℝ, 
    Real.sqrt ((F₁.1 - P.1)^2 + (F₁.2 - P.2)^2) + d = 
    Real.sqrt ((F₂.1 - P.1)^2 + (F₂.2 - P.2)^2) ∧
    Real.sqrt ((F₂.1 - P.1)^2 + (F₂.2 - P.2)^2) + d = 
    Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  is_hyperbola P.1 P.2 a b →
  are_foci F₁ F₂ a b →
  on_hyperbola P a b →
  right_angle F₁ P F₂ →
  arithmetic_sequence F₁ P F₂ →
  eccentricity a b = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1161_116139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_domain_l1161_116142

noncomputable def k (x : ℝ) : ℝ := 1 / (x + 8) + 1 / (x^2 + 8) + 1 / (x^4 + x^3 + 8)

theorem k_domain : 
  {x : ℝ | IsRegular (k x)} = {x : ℝ | x < -8 ∨ x > -8} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_domain_l1161_116142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_solutions_count_constrained_solutions_count_l1161_116177

/-- The number of positive integer solutions to x₁ + x₂ + x₃ + ... + xₘ = n -/
def num_positive_solutions (m n : ℕ) : ℕ :=
  Nat.choose (n - 1) (m - 1)

/-- The number of integer solutions to x₁ + x₂ + x₃ + ... + xₘ = n with constraints -/
def num_constrained_solutions (m n r : ℕ) : ℕ :=
  Nat.choose (n + (r - 1) * (2 - m)) (m - 1)

theorem positive_solutions_count (m n : ℕ) (h : m < n) :
  num_positive_solutions m n = Nat.choose (n - 1) (m - 1) := by
  rfl

theorem constrained_solutions_count (m n r : ℕ) 
  (h1 : m > 0) (h2 : n > 0) (h3 : n ≥ (m - 2) * r + 1) :
  num_constrained_solutions m n r = Nat.choose (n + (r - 1) * (2 - m)) (m - 1) := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_solutions_count_constrained_solutions_count_l1161_116177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1161_116115

/-- The function f(x) defined as sin(ωx) + 1 --/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + 1

/-- The statement that f has exactly three zeros in the interval [-π/3, 0] --/
def has_three_zeros (ω : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    x₁ ∈ Set.Icc (-Real.pi/3) 0 ∧ x₂ ∈ Set.Icc (-Real.pi/3) 0 ∧ x₃ ∈ Set.Icc (-Real.pi/3) 0 ∧
    f ω x₁ = 0 ∧ f ω x₂ = 0 ∧ f ω x₃ = 0 ∧
    ∀ x ∈ Set.Icc (-Real.pi/3) 0, f ω x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃

/-- The theorem stating the range of ω --/
theorem omega_range :
  ∀ ω : ℝ, ω > 0 → has_three_zeros ω → ω ∈ Set.Icc (51/4) (75/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1161_116115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_of_q_zeros_l1161_116153

/-- The polynomial Q(z) -/
noncomputable def Q (z : ℂ) : ℂ := z^8 + (6 * Real.sqrt 2 + 8) * z^4 - (6 * Real.sqrt 2 + 9)

/-- The set of zeros of Q(z) -/
def zeros : Set ℂ := {z : ℂ | Q z = 0}

/-- An 8-sided polygon in the complex plane -/
structure Octagon :=
  (vertices : Fin 8 → ℂ)

/-- The perimeter of an octagon -/
noncomputable def perimeter (o : Octagon) : ℝ :=
  (Finset.sum Finset.univ fun i => Complex.abs (o.vertices i.succ - o.vertices i))

/-- An octagon whose vertices are the zeros of Q(z) -/
def isValidOctagon (o : Octagon) : Prop :=
  (Finset.univ.image o.vertices : Set ℂ) = zeros

theorem min_perimeter_of_q_zeros :
  ∃ (o : Octagon), isValidOctagon o ∧
    (∀ (o' : Octagon), isValidOctagon o' → perimeter o ≤ perimeter o') ∧
    perimeter o = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_of_q_zeros_l1161_116153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fraction_with_20_percent_increase_l1161_116186

theorem no_fraction_with_20_percent_increase : ¬∃ (x y : ℕ), 
  x > 0 ∧ y > 0 ∧ Nat.gcd x y = 1 ∧ (x + 1) * y = 6 * x * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fraction_with_20_percent_increase_l1161_116186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_error_calculation_l1161_116105

-- Define the actual side length and the measured side length
variable (S : ℝ) -- actual side length
variable (S' : ℝ) -- measured side length

-- Define the error in area calculation
def area_error : ℝ := 25.44

-- Define the relationship between S and S'
def side_error_relation (S S' e : ℝ) : Prop := S' = S * (1 + e / 100)

-- Theorem statement
theorem side_error_calculation :
  ∃ e : ℝ, side_error_relation S S' e ∧ 
  (S'^2 - S^2) / S^2 * 100 = area_error ∧
  e = 12.72 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_error_calculation_l1161_116105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_radius_l1161_116162

-- Define the unit circle
def unitCircle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define a point P on the unit circle
noncomputable def P : ℝ × ℝ := sorry

-- Define point A as the projection of P on the x-axis
noncomputable def A : ℝ × ℝ := (P.1, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the triangle OAP
def triangleOAP : Set (ℝ × ℝ) := {O, A, P}

-- Define the inscribed circle radius function
noncomputable def inscribedCircleRadius (α : ℝ) : ℝ := 
  (Real.cos (α/2) - Real.sin (α/2)) * Real.sin (α/2)

-- The theorem to prove
theorem largest_inscribed_circle_radius :
  ∃ (r : ℝ), r = (Real.sqrt 2 - 1) / 2 ∧
  ∀ (α : ℝ), inscribedCircleRadius α ≤ r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_radius_l1161_116162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_inscribed_squares_l1161_116189

/-- Predicate to check if a point is a vertex of the inner square -/
def is_vertex_inner_square (p : ℝ × ℝ) (perimeter : ℝ) : Prop :=
  ∃ (x y : ℝ), (x = 0 ∨ x = perimeter / 4) ∧ (y = 0 ∨ y = perimeter / 4) ∧ p = (x, y)

/-- Predicate to check if a point is a vertex of the outer square -/
def is_vertex_outer_square (p : ℝ × ℝ) (perimeter : ℝ) : Prop :=
  ∃ (x y : ℝ), (x = 0 ∨ x = perimeter / 4) ∧ (y = 0 ∨ y = perimeter / 4) ∧ p = (x, y)

/-- The maximum distance between vertices of two squares, one inscribed in the other -/
theorem max_distance_inscribed_squares (inner_perimeter outer_perimeter : ℝ) 
  (h_inner : inner_perimeter = 24) 
  (h_outer : outer_perimeter = 36) :
  ∃ (d : ℝ), d = 7.5 * Real.sqrt 2 ∧ 
  ∀ (p₁ p₂ : ℝ × ℝ), 
    is_vertex_inner_square p₁ inner_perimeter → 
    is_vertex_outer_square p₂ outer_perimeter → 
    Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) ≤ d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_inscribed_squares_l1161_116189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_b_value_l1161_116137

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- Theorem: For a hyperbola with equation x^2/a^2 - y^2/b^2 = 1, where b > 0 and eccentricity is 2, the value of b is √3 -/
theorem hyperbola_b_value (a b : ℝ) (h1 : b > 0) (h2 : eccentricity a b = 2) : b = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_b_value_l1161_116137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1161_116129

/-- For any positive integer n, the polynomial f(x) = x^(n+2) + (x+1)^(2n+1) is divisible by x^2 + x + 1 -/
theorem polynomial_divisibility (n : ℕ) :
  ∃ q : Polynomial ℤ, x^(n+2) + (x+1)^(2*n+1) = (x^2 + x + 1) * q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1161_116129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sequence_properties_l1161_116198

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the tangent line at point n
noncomputable def tangent_line (n : ℕ) (x : ℝ) : ℝ := f n + (f n) * (x - n)

-- Define the x-coordinate of the intersection point
noncomputable def x_n (n : ℕ) : ℝ := n + 1 / (Real.exp 1 - 1)

-- Define the y-coordinate of the intersection point
noncomputable def y_n (n : ℕ) : ℝ := (Real.exp (n + 1)) / (Real.exp 1 - 1)

-- Theorem statement
theorem intersection_sequence_properties :
  (∀ n : ℕ, x_n (n + 1) - x_n n = x_n 2 - x_n 1) ∧
  (∀ n : ℕ, y_n (n + 1) / y_n n = y_n 2 / y_n 1) :=
by
  sorry  -- Skip the proof for now

#check intersection_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sequence_properties_l1161_116198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_dmitry_before_father_is_two_thirds_l1161_116117

/-- Represents the time intervals for family members' arrivals -/
structure TimeIntervals where
  m : ℝ  -- Total time between Mary Kuzminichna's visit and Galina Efimovna's arrival
  x : ℝ  -- Time from Mary Kuzminichna's visit to Anatoly's arrival
  y : ℝ  -- Time from Mary Kuzminichna's visit to Dmitry's arrival
  z : ℝ  -- Time from Mary Kuzminichna's visit to Svetlana's arrival
  h_m_pos : m > 0
  h_x : 0 < x ∧ x < m
  h_y : 0 < y
  h_z : y < z ∧ z < m

/-- The probability that Dmitry returned home before his father -/
noncomputable def probability_dmitry_before_father (t : TimeIntervals) : ℝ :=
  2 / 3

/-- Theorem stating that the probability of Dmitry returning home before his father is 2/3 -/
theorem probability_dmitry_before_father_is_two_thirds (t : TimeIntervals) :
  probability_dmitry_before_father t = 2 / 3 := by
  -- The proof is omitted for now
  sorry

#check probability_dmitry_before_father_is_two_thirds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_dmitry_before_father_is_two_thirds_l1161_116117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_expected_sides_general_l1161_116108

/-- Represents the number of sides of the initial polygon -/
def n : ℕ := 4

/-- Represents the number of cuts made -/
def num_cuts : ℕ := 3600

/-- Represents the total number of sides after all cuts -/
def total_sides (n : ℕ) : ℕ := n + 4 * num_cuts

/-- Represents the total number of polygons after all cuts -/
def total_polygons : ℕ := num_cuts + 1

/-- The theorem stating the expected number of sides -/
theorem expected_sides : 
  (total_sides n : ℚ) / (total_polygons : ℚ) = (n + 14400 : ℚ) / 3601 := by
  sorry

/-- The theorem for the general case with arbitrary initial polygon -/
theorem expected_sides_general (m : ℕ) : 
  (total_sides m : ℚ) / (total_polygons : ℚ) = (m + 14400 : ℚ) / 3601 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_expected_sides_general_l1161_116108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_interval_l1161_116193

theorem a_interval (a : ℝ) (h : a^5 - a^3 + a = 2) : (3 : ℝ)^(1/6) < a ∧ a < (2 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_interval_l1161_116193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mixture_characterization_l1161_116176

/-- A material point representing a mixture -/
structure MaterialPoint where
  K : ℝ  -- Concentration
  m : ℝ  -- Mass
  m_pos : m > 0

/-- Addition of material points -/
noncomputable def MaterialPoint.add (p q : MaterialPoint) : MaterialPoint where
  K := (p.K * p.m + q.K * q.m) / (p.m + q.m)
  m := p.m + q.m
  m_pos := by
    have h1 : p.m > 0 := p.m_pos
    have h2 : q.m > 0 := q.m_pos
    exact add_pos h1 h2

/-- Theorem: The combined mixture is characterized by the sum of the original material points -/
theorem combined_mixture_characterization (p q : MaterialPoint) :
  ∃ (r : MaterialPoint), r = MaterialPoint.add p q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mixture_characterization_l1161_116176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_valid_routes_l1161_116170

-- Define the graph structure
structure CityGraph where
  cities : Finset String
  roads : Finset (String × String)
  num_cities : cities.card = 15
  num_roads : roads.card = 20
  b_exists : "B" ∈ cities
  m_exists : "M" ∈ cities

-- Define a valid route
def ValidRoute (g : CityGraph) (route : List (String × String)) : Prop :=
  route.length = 15 ∧
  (route.head?.map Prod.fst = some "B") ∧
  (route.getLast?.map Prod.snd = some "M") ∧
  (∀ r ∈ route, r ∈ g.roads) ∧
  (∀ r ∈ route, ∀ s ∈ route, r ≠ s → r.1 ≠ s.2 ∨ r.2 ≠ s.1)

-- Main theorem
theorem exactly_two_valid_routes (g : CityGraph) : 
  ∃! (routes : Finset (List (String × String))), 
    routes.card = 2 ∧ 
    ∀ route ∈ routes, ValidRoute g route :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_valid_routes_l1161_116170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_series_diverges_l1161_116179

open BigOperators Real

/-- A bijection from ℕ to ℕ -/
noncomputable def f : ℕ → ℕ := sorry

/-- The statement that at least one of the series diverges -/
theorem at_least_one_series_diverges :
  (¬ ∃ M : ℝ, ∀ N : ℕ, (∑ n in Finset.range N, (1 : ℝ) / (n + f n)) ≤ M) ∨
  (¬ ∃ M : ℝ, ∀ N : ℕ, (∑ n in Finset.range N, (1 : ℝ) / n - (1 : ℝ) / f n) ≤ M) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_series_diverges_l1161_116179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_zero_of_f_l1161_116127

-- Define the function f(x) = log x + x - 3
noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 3

-- Define the approximate values for log given in the reference data
def log_2_5 : ℝ := 0.398
def log_2_75 : ℝ := 0.439
def log_2_5625 : ℝ := 0.409

-- Define the accuracy requirement
def accuracy : ℝ := 0.1

-- State the theorem
theorem approximate_zero_of_f :
  ∃ (x : ℝ), x > 2.5 ∧ x < 2.75 ∧ |f x| < accuracy ∧ |x - 2.6| ≤ accuracy := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_zero_of_f_l1161_116127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_loss_percentage_l1161_116180

/-- Represents the cost and selling prices of a radio type -/
structure RadioPrices where
  costPrice : ℝ
  sellingPrice : ℝ

/-- Calculates the loss percentage for a single radio type -/
noncomputable def lossPercentage (radio : RadioPrices) : ℝ :=
  (radio.costPrice - radio.sellingPrice) / radio.costPrice * 100

/-- Calculates the overall loss percentage for multiple radio types -/
noncomputable def overallLossPercentage (radios : List RadioPrices) : ℝ :=
  let totalCost := radios.map (·.costPrice) |>.sum
  let totalSelling := radios.map (·.sellingPrice) |>.sum
  (totalCost - totalSelling) / totalCost * 100

/-- The main theorem stating the overall loss percentage -/
theorem radio_loss_percentage : 
  let radioA := RadioPrices.mk 490 465.50
  let radioB := RadioPrices.mk 765 720
  let radioC := RadioPrices.mk 1220 1185
  let radios := [radioA, radioB, radioC]
  abs (overallLossPercentage radios - 4.222222222) < 0.000000001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_loss_percentage_l1161_116180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_in_divided_square_l1161_116167

/-- Given a square divided into six rectangles of equal area, 
    the ratio of the longer side to the shorter side of any rectangle 
    that spans half the width of the square is 1.5 -/
theorem rectangle_ratio_in_divided_square (a : ℝ) (h : a > 0) : 
  (a / 2) / ((a^2 / 6) / (a / 2)) = 3/2 := by
  -- Simplify the expression
  have h1 : (a / 2) / ((a^2 / 6) / (a / 2)) = (a / 2) / (a / 3) := by
    -- Simplification steps would go here
    sorry
  -- Show that (a / 2) / (a / 3) = 3/2
  have h2 : (a / 2) / (a / 3) = 3/2 := by
    -- Proof steps would go here
    sorry
  -- Combine the results
  rw [h1, h2]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_in_divided_square_l1161_116167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l1161_116192

theorem min_k_value (k : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 (π / 2) → Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x ≤ k + 1) ↔ 
  k ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l1161_116192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_camp_sampling_l1161_116125

/-- Represents a camp with a range of student numbers -/
structure Camp where
  start : Nat
  stop : Nat

/-- Represents the systematic sampling parameters -/
structure SamplingParams where
  totalStudents : Nat
  sampleSize : Nat
  startNumber : Nat

/-- Counts the number of selected students in a given camp -/
def countSelectedInCamp (camp : Camp) (params : SamplingParams) : Nat :=
  sorry

/-- Theorem statement for the summer camp problem -/
theorem summer_camp_sampling 
  (params : SamplingParams)
  (campI campII campIII : Camp)
  (h1 : params.totalStudents = 600)
  (h2 : params.sampleSize = 50)
  (h3 : params.startNumber = 3)
  (h4 : campI.start = 1 ∧ campI.stop = 300)
  (h5 : campII.start = 301 ∧ campII.stop = 495)
  (h6 : campIII.start = 496 ∧ campIII.stop = 600) :
  (countSelectedInCamp campI params = 25) ∧
  (countSelectedInCamp campII params = 17) ∧
  (countSelectedInCamp campIII params = 8) :=
by
  sorry

#check summer_camp_sampling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_camp_sampling_l1161_116125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1161_116119

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 3 ≤ x ∧ x ≤ m + 3}

-- Part 1
theorem part_one (m : ℝ) : A ∩ B m = Set.Icc 2 3 → m = 5 := by sorry

-- Part 2
theorem part_two (m : ℝ) : A ⊆ (B m)ᶜ → m < -4 ∨ m > 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1161_116119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_form_l1161_116144

/-- A parabola with directrix x = 3 and opening to the left has the standard form equation y² = -12x -/
theorem parabola_standard_form (p : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ p ↔ (x - 3)^2 = (y^2 / 4)) →  -- directrix x = 3
  (∃ (a : ℝ), a > 0 ∧ ∀ (x y : ℝ), (x, y) ∈ p ↔ y^2 = -4*a*x) →  -- opens to the left
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ p ↔ y^2 = -12*x + k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_form_l1161_116144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_total_amount_l1161_116140

/-- Calculate the total amount Janet owes for wages and taxes for one month -/
def total_amount_owed : ℕ :=
  let warehouse_workers : ℕ := 4
  let managers : ℕ := 2
  let warehouse_wage : ℕ := 15
  let manager_wage : ℕ := 20
  let days_per_month : ℕ := 25
  let hours_per_day : ℕ := 8
  let fica_tax_rate : Rat := 1/10

  let warehouse_total : ℕ := warehouse_workers * warehouse_wage * days_per_month * hours_per_day
  let manager_total : ℕ := managers * manager_wage * days_per_month * hours_per_day
  let total_wages : ℕ := warehouse_total + manager_total
  let fica_taxes : ℕ := (((total_wages : Rat) * fica_tax_rate).floor.toNat)

  total_wages + fica_taxes

theorem janet_total_amount : total_amount_owed = 22000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_total_amount_l1161_116140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1161_116146

theorem relationship_abc (a b c : ℝ) 
  (ha : a = (1/2)^3) 
  (hb : b = 3^(1/2)) 
  (hc : c = Real.log 3 / Real.log (1/2)) : 
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1161_116146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_one_l1161_116120

theorem integral_sqrt_minus_one :
  (∫ x in (0 : ℝ)..1, (Real.sqrt (1 - (1 - x)^2) - 1)) = π / 4 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_one_l1161_116120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_phone_time_l1161_116128

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem jill_phone_time : geometric_sum 5 2 5 = 155 := by
  -- Unfold the definition of geometric_sum
  unfold geometric_sum
  -- Simplify the expression
  simp [pow_succ, Real.rpow_nat_cast]
  -- Perform the numerical calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_phone_time_l1161_116128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_is_AB_l1161_116148

/-- Represents the cost in millions of yuan -/
abbrev Cost : Type := ℝ

/-- Represents the probability of an event occurring -/
abbrev Probability : Type := ℝ

/-- The probability of the unexpected event occurring without preventive measures -/
def p_event : Probability := 0.3

/-- The loss in millions of yuan if the unexpected event occurs -/
def loss : Cost := 4

/-- The cost of preventive measure A in millions of yuan -/
def cost_A_measure : Cost := 0.45

/-- The cost of preventive measure B in millions of yuan -/
def cost_B_measure : Cost := 0.3

/-- The probability of the event not occurring with measure A -/
def p_not_A : Probability := 0.9

/-- The probability of the event not occurring with measure B -/
def p_not_B : Probability := 0.85

/-- Calculate the total cost given the prevention cost and the probability of the event occurring -/
def total_cost (prevention_cost : Cost) (p_event : Probability) : Cost :=
  prevention_cost + p_event * loss

/-- The total cost without any preventive measures -/
def cost_none : Cost := total_cost 0 p_event

/-- The total cost with preventive measure A -/
def cost_A : Cost := total_cost cost_A_measure (1 - p_not_A)

/-- The total cost with preventive measure B -/
def cost_B : Cost := total_cost cost_B_measure (1 - p_not_B)

/-- The total cost with both preventive measures A and B -/
def cost_AB : Cost := total_cost (cost_A_measure + cost_B_measure) ((1 - p_not_A) * (1 - p_not_B))

theorem min_cost_is_AB :
  cost_AB < cost_none ∧ cost_AB < cost_A ∧ cost_AB < cost_B := by
  sorry

#eval cost_none
#eval cost_A
#eval cost_B
#eval cost_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_is_AB_l1161_116148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_at_two_equals_seven_l1161_116121

-- Define the function q
noncomputable def q (x : ℝ) : ℝ :=
  Real.sign (3*x - 3) * |3*x - 3|^(1/3) + 
  3 * Real.sign (3*x - 3) * |3*x - 3|^(1/7) + 
  |3*x - 3|^(1/9)

-- State the theorem
theorem q_at_two_equals_seven : q 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_at_two_equals_seven_l1161_116121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_e₁_e₂_norm_of_a_min_value_of_norm_a_l1161_116157

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors and parameters
variable (e₁ e₂ e₃ : V) (x : ℝ) (n : ℕ)

-- Define the hypotheses
variable (h₁ : ‖e₁‖ = 1) (h₂ : ‖e₂‖ = 1) (h₃ : ‖e₃‖ = 1)
variable (h₄ : e₁ + e₂ + e₃ = 0)
variable (h₅ : x ≠ 0)
variable (h₆ : 0 < n)

-- Define vector a
noncomputable def a : V := x • e₁ + (n / x) • e₂ + (x + n / x) • e₃

-- State the theorems to be proved
theorem angle_between_e₁_e₂ : 
  Real.arccos (inner e₁ e₂) = (2 * Real.pi) / 3 := by
  sorry

theorem norm_of_a : 
  ‖a e₁ e₂ e₃ x n‖ = Real.sqrt (x^2 + (n / x)^2 - n) := by
  sorry

theorem min_value_of_norm_a :
  ∃ (x_min : ℝ), ∀ (x : ℝ), x ≠ 0 → ‖a e₁ e₂ e₃ x_min n‖ ≤ ‖a e₁ e₂ e₃ x n‖ ∧
  ‖a e₁ e₂ e₃ x_min n‖ = Real.sqrt n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_e₁_e₂_norm_of_a_min_value_of_norm_a_l1161_116157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_sums_l1161_116199

/- Define the arithmetic sequence -/
def arithmetic_sequence (a₁ : ℕ) (d : ℕ) : ℕ → ℕ
  | n => a₁ + (n - 1) * d

/- Define the condition that all terms are two-digit even numbers -/
def is_two_digit_even (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 2 = 0

/- Define the sum of odd-indexed terms -/
def sum_odd_terms (seq : ℕ → ℕ) (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => seq (2 * i + 1))

/- Define the new sequence formed by combining odd terms with following even terms -/
def new_sequence (seq : ℕ → ℕ) : ℕ → ℕ
  | n => if n % 2 = 1 then seq n * 100 + seq (n + 1) else 0

/- Main theorem -/
theorem difference_of_sums (a₁ d : ℕ) (h₁ : ∀ n, is_two_digit_even (arithmetic_sequence a₁ d n))
    (h₂ : ∃ N, sum_odd_terms (arithmetic_sequence a₁ d) N = 100) :
  ∃ M, Finset.sum (Finset.range M) (new_sequence (arithmetic_sequence a₁ d)) -
       Finset.sum (Finset.range (2 * M)) (arithmetic_sequence a₁ d) = 9900 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_sums_l1161_116199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_distance_l1161_116196

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem hyperbola_circle_distance (h : Hyperbola) (c : Circle) :
  h.a = 3 →
  h.b = 4 →
  (c.center.1 = 4 ∨ c.center.1 = -4) →
  c.center.1^2 / h.a^2 - c.center.2^2 / h.b^2 = 1 →
  distance c.center (0, 0) = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_distance_l1161_116196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_is_correct_l1161_116161

/-- The discount rate offered by the shop -/
noncomputable def discount_rate : ℝ := 0.40

/-- The price Smith paid for the shirt after discount -/
noncomputable def discounted_price : ℝ := 560

/-- The original selling price of the shirt -/
noncomputable def original_price : ℝ := discounted_price / (1 - discount_rate)

/-- Theorem stating that the original price is approximately 933.33 -/
theorem original_price_is_correct : 
  ∃ ε > 0, |original_price - 933.33| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_is_correct_l1161_116161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_from_cosine_relation_l1161_116122

/-- Given a triangle ABC where b*cos(C) = c*cos(B), prove that the triangle is isosceles. -/
theorem triangle_isosceles_from_cosine_relation (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Side lengths are positive
  0 < A ∧ 0 < B ∧ 0 < C →  -- Angles are positive
  A + B + C = π →  -- Sum of angles in a triangle
  a * Real.sin B = b * Real.sin A →  -- Law of sines
  b * Real.sin C = c * Real.sin B →  -- Law of sines
  a * Real.sin C = c * Real.sin A →  -- Law of sines
  b * Real.cos C = c * Real.cos B →  -- Given condition
  B = C  -- Conclusion: two angles are equal, hence isosceles
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_from_cosine_relation_l1161_116122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_cost_is_76950_l1161_116147

/-- Represents the cost calculation for filling a trapezoidal prism-shaped pool --/
noncomputable def pool_cost_calculation 
  (base_width : ℝ) 
  (base_length : ℝ) 
  (top_width : ℝ) 
  (top_length : ℝ) 
  (depth : ℝ) 
  (liters_per_cubic_foot : ℝ) 
  (price_per_liter : ℝ) 
  (tax_rate : ℝ) 
  (discount_rate : ℝ) : ℝ :=
  let volume := (1/2) * depth * (base_width + top_width) * base_length
  let total_liters := volume * liters_per_cubic_foot
  let initial_cost := total_liters * price_per_liter
  let taxed_cost := initial_cost * (1 + tax_rate)
  taxed_cost * (1 - discount_rate)

/-- Theorem stating the cost to fill the pool with the given parameters --/
theorem pool_cost_is_76950 : 
  pool_cost_calculation 6 20 4 18 10 25 3 0.08 0.05 = 76950 := by
  sorry

-- Remove this line as it's causing issues with noncomputable definitions
-- #eval pool_cost_calculation 6 20 4 18 10 25 3 0.08 0.05

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_cost_is_76950_l1161_116147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1161_116197

-- Define the function f
noncomputable def f (a b r : ℝ) : ℝ := (a^(1/2 + r) + b^(1/2 + r)) / (a^(1/2 - r) + b^(1/2 - r))

-- State the theorem
theorem f_inequality (a b r : ℝ) (ha : a > 0) (hb : b > 0) :
  (r < 0 → f a b r ≤ (a * b)^r) ∧ (r > 0 → f a b r ≥ (a * b)^r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1161_116197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_problem_l1161_116184

/-- Represents an infinite geometric series -/
structure InfiniteGeometricSeries where
  firstTerm : ℝ
  secondTerm : ℝ

/-- Calculates the sum of an infinite geometric series -/
noncomputable def sumInfiniteGeometricSeries (s : InfiniteGeometricSeries) : ℝ :=
  s.firstTerm / (1 - s.secondTerm / s.firstTerm)

theorem geometric_series_problem (m : ℝ) : 
  let s1 := InfiniteGeometricSeries.mk 20 10
  let s2 := InfiniteGeometricSeries.mk 20 (10 + m)
  sumInfiniteGeometricSeries s2 = 3 * sumInfiniteGeometricSeries s1 → m = 20 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_problem_l1161_116184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_satisfying_combination_l1161_116163

/-- A set of the numbers 1, 2, 3, 5, 6 -/
def NumberSet : Finset ℕ := {1, 2, 3, 5, 6}

/-- The property that a combination of numbers satisfies the equation -/
def SatisfiesEquation (a b c d e : ℕ) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ c ∈ NumberSet ∧ d ∈ NumberSet ∧ e ∈ NumberSet ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  (a + b - c) * d / e = 4

/-- There exists a combination of numbers from NumberSet that satisfies the equation -/
theorem exists_satisfying_combination : ∃ (a b c d e : ℕ), SatisfiesEquation a b c d e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_satisfying_combination_l1161_116163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_gas_work_l1161_116100

-- Define the gas constant
def R : ℝ := 8.31

-- Define the temperature change in Kelvin
def ΔT : ℝ := 100

-- Define the relation between pressure and volume
def pressure_volume_relation (p V : ℝ) : Prop :=
  ∃ α : ℝ, p = α * V

-- Define the work done by the gas
noncomputable def work_done (p V₁ V₂ : ℝ) : ℝ :=
  ∫ v in V₁..V₂, p * v

-- State the theorem
theorem ideal_gas_work :
  ∀ (p V₁ V₂ : ℝ),
  pressure_volume_relation p V₁ →
  pressure_volume_relation p V₂ →
  work_done p V₁ V₂ = (1/2) * R * ΔT :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_gas_work_l1161_116100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l1161_116151

open Real

noncomputable def curve_C (θ : Real) : Real := 2 * cos (θ - π / 3)

noncomputable def point_A : Real × Real := (3/2, sqrt 3 / 2)
noncomputable def point_B : Real × Real := (0, sqrt 3)

noncomputable def line_l (t : Real) : Real × Real := (1/2 - sqrt 3 / 2 * t, sqrt 3 / 2 + 1/2 * t)

theorem curve_and_line_properties :
  -- 1. The curve C in Cartesian coordinates
  (∀ x y, (x - 1/2)^2 + (y - sqrt 3 / 2)^2 = 1 ↔
    ∃ θ, x = curve_C θ * cos θ ∧ y = curve_C θ * sin θ) ∧
  -- 2. The line l is parallel to AB and bisects the area enclosed by C
  (∃ t, line_l t = (1/2, sqrt 3 / 2)) ∧
  -- 3. Range of |MA|² + |MB|² for any point M on C
  (∀ θ, let M := (1/2 + cos θ, sqrt 3 / 2 + sin θ);
        2 ≤ (M.1 - point_A.1)^2 + (M.2 - point_A.2)^2 +
           (M.1 - point_B.1)^2 + (M.2 - point_B.2)^2 ∧
        (M.1 - point_A.1)^2 + (M.2 - point_A.2)^2 +
        (M.1 - point_B.1)^2 + (M.2 - point_B.2)^2 ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l1161_116151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l1161_116131

/-- The ratio of areas of triangles associated with sides a and b in a triangle --/
theorem triangle_area_ratio (a b c s : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semiperimeter : s = (a + b + c) / 2) :
  (s - b) / (s - a) = (s - b) / (s - a) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l1161_116131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_correct_proposition_l1161_116183

theorem one_correct_proposition (x y : ℝ) : 
  (∃! n : Nat, n ∈ ({1, 2, 3, 4, 5} : Set Nat) ∧
    (n = 1 → (x > y → x^2 > y^2)) ∧
    (n = 2 → (x^2 > y^2 → x > y)) ∧
    (n = 3 → (x > abs y → x^2 > y^2)) ∧
    (n = 4 → (abs x > y → x^2 > y^2)) ∧
    (n = 5 → (x ≠ y → x^2 ≠ y^2))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_correct_proposition_l1161_116183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_demand_decrease_for_constant_revenue_l1161_116149

/-- Calculates the required percentage decrease in demand to maintain constant revenue
    given an original price increase and a subsidy factor. -/
noncomputable def required_demand_decrease (original_increase : ℝ) (subsidy_factor : ℝ) : ℝ :=
  1 - 1 / (1 + original_increase * subsidy_factor)

/-- The theorem states that given a 20% price increase reduced by half,
    the required percentage decrease in demand is approximately 9.09%. -/
theorem demand_decrease_for_constant_revenue :
  let original_increase : ℝ := 0.2
  let subsidy_factor : ℝ := 0.5
  let required_decrease : ℝ := required_demand_decrease original_increase subsidy_factor
  abs (required_decrease - 0.0909) < 0.0001 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval required_demand_decrease 0.2 0.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_demand_decrease_for_constant_revenue_l1161_116149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_plate_probability_l1161_116103

theorem same_color_plate_probability 
  (red_plates : Nat) 
  (blue_plates : Nat) 
  (green_plates : Nat) 
  (h_red : red_plates = 6) 
  (h_blue : blue_plates = 5) 
  (h_green : green_plates = 4) : 
  (((red_plates * (red_plates - 1)) / 2 + 
    (blue_plates * (blue_plates - 1)) / 2 + 
    (green_plates * (green_plates - 1)) / 2 : Rat) / 
   ((red_plates + blue_plates + green_plates) * 
    (red_plates + blue_plates + green_plates - 1) / 2)) = 31 / 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_plate_probability_l1161_116103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l1161_116154

/-- A conic section C is defined by the equation x²+my²=1 where m is a real number. -/
def ConicSection (m : ℝ) := {(x, y) : ℝ × ℝ | x^2 + m*y^2 = 1}

/-- The eccentricity of a conic section. -/
noncomputable def Eccentricity (m : ℝ) : ℝ := Real.sqrt (1 + (-1/m))

/-- Theorem: If the eccentricity of the conic section C: x²+my²=1 is 2, then m = -1/3. -/
theorem conic_section_eccentricity (m : ℝ) :
  m < 0 → Eccentricity m = 2 → m = -1/3 := by
  sorry

#check conic_section_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l1161_116154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1161_116166

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 25 - y^2 / 9 = 1

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Define the foci of the hyperbola
noncomputable def left_focus : ℝ × ℝ := (-Real.sqrt 34, 0)
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 34, 0)

-- Theorem statement
theorem hyperbola_focal_distance 
  (x y : ℝ) 
  (h_on_hyperbola : hyperbola x y) 
  (h_dist_left : distance x y (left_focus.1) (left_focus.2) = 18) :
  distance x y (right_focus.1) (right_focus.2) = 8 ∨ 
  distance x y (right_focus.1) (right_focus.2) = 28 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1161_116166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_model_height_approx_l1161_116181

/-- Represents the dimensions and volumes of the original landmark and its scaled model -/
structure LandmarkScaling where
  originalHeight : ℝ
  originalVolume : ℝ
  modelVolume : ℝ

/-- Calculates the height of a scaled model given the original landmark's dimensions and the model's volume -/
noncomputable def scaledModelHeight (l : LandmarkScaling) : ℝ :=
  l.originalHeight * (l.modelVolume / l.originalVolume) ^ (1/3 : ℝ)

/-- Theorem stating that the scaled model's height is approximately 1.29 meters -/
theorem scaled_model_height_approx (l : LandmarkScaling) 
  (h1 : l.originalHeight = 60)
  (h2 : l.originalVolume = 150000)
  (h3 : l.modelVolume = 1.5) :
  ‖scaledModelHeight l - 1.29‖ < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_model_height_approx_l1161_116181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_esha_behind_anusha_l1161_116104

/-- Represents a runner in the race -/
inductive Runner
| Anusha
| Banu
| Esha

/-- Represents a section of the race -/
inductive Section
| Flat
| Uphill
| Downhill

/-- The race configuration -/
structure Race where
  total_distance : ℝ
  flat_distance : ℝ
  uphill_distance : ℝ
  downhill_distance : ℝ

/-- The relative speed of runners on different sections -/
def faster_than (r1 r2 : Runner) (s : Section) : Prop :=
  match s, r1, r2 with
  | Section.Flat, Runner.Anusha, Runner.Banu => True
  | Section.Flat, Runner.Banu, Runner.Esha => True
  | Section.Uphill, Runner.Banu, Runner.Anusha => True
  | Section.Uphill, Runner.Banu, Runner.Esha => True
  | _, _, _ => False

/-- The distance between two runners at a given point -/
noncomputable def distance_between (r1 r2 : Runner) (at_finish : Runner) : ℝ := 
  sorry

/-- The main theorem to prove -/
theorem esha_behind_anusha (race : Race)
  (h_total : race.total_distance = 100)
  (h_flat : race.flat_distance = 40)
  (h_uphill : race.uphill_distance = 30)
  (h_downhill : race.downhill_distance = 30)
  (h_flat_order : faster_than Runner.Anusha Runner.Banu Section.Flat ∧ 
                  faster_than Runner.Banu Runner.Esha Section.Flat)
  (h_uphill_order : faster_than Runner.Banu Runner.Anusha Section.Uphill ∧ 
                    faster_than Runner.Banu Runner.Esha Section.Uphill)
  (h_anusha_banu : distance_between Runner.Anusha Runner.Banu Runner.Anusha = 10)
  (h_banu_esha : distance_between Runner.Banu Runner.Esha Runner.Banu = 10) :
  distance_between Runner.Anusha Runner.Esha Runner.Anusha = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_esha_behind_anusha_l1161_116104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1161_116178

-- Define the complex number
noncomputable def z : ℂ := Complex.abs (3 + 4*Complex.I) / (2 + Complex.I)

-- Statement to prove
theorem z_in_fourth_quadrant : 
  Real.sign z.re = 1 ∧ Real.sign z.im = -1 :=
by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1161_116178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_average_mark_l1161_116185

theorem first_class_average_mark (students_first : ℕ) (students_second : ℕ) 
  (average_second : ℝ) (average_total : ℝ) :
  students_first = 22 →
  students_second = 28 →
  average_second = 60 →
  average_total = 51.2 →
  (students_first + students_second) * average_total - students_second * average_second = students_first * 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_average_mark_l1161_116185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_five_hundredth_term_l1161_116101

/-- Defines the sequence where the nth positive integer appears n times -/
def my_sequence (n : ℕ) : ℕ := Nat.sqrt (2 * n + 1)

theorem two_thousand_five_hundredth_term :
  my_sequence 2500 = 71 ∧ my_sequence 2500 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_five_hundredth_term_l1161_116101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_N_power_five_l1161_116143

theorem det_N_power_five {n : Type*} [Fintype n] [DecidableEq n] 
  (N : Matrix n n ℝ) (h : Matrix.det N = 3) : Matrix.det (N^5) = 243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_N_power_five_l1161_116143
