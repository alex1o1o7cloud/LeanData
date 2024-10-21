import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1081_108157

/-- Given a principal amount P and an annual interest rate r (as a percentage),
    calculates the compound interest after t years with n compoundings per year. -/
noncomputable def compound_interest (P r t n : ℝ) : ℝ :=
  P * ((1 + r / (100 * n)) ^ (n * t) - 1)

/-- Theorem stating that if the compound interest on a principal P for 5 years
    with semi-annual compounding is one-third of P, then the annual interest rate r
    satisfies the equation r = 200 * ((4/3)^(1/10) - 1). -/
theorem interest_rate_calculation (P : ℝ) (h : P > 0) :
  ∃ r : ℝ, compound_interest P r 5 2 = P / 3 ∧
  r = 200 * ((4/3)^(1/10) - 1) := by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1081_108157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_in_AlBr3_approx_l1081_108141

/-- The mass percentage of aluminum in aluminum bromide -/
noncomputable def mass_percentage_Al_in_AlBr3 : ℝ :=
  let molar_mass_Al : ℝ := 26.98
  let molar_mass_Br : ℝ := 79.90
  let molar_mass_AlBr3 : ℝ := molar_mass_Al + 3 * molar_mass_Br
  (molar_mass_Al / molar_mass_AlBr3) * 100

/-- Theorem stating that the mass percentage of aluminum in aluminum bromide is approximately 10.11% -/
theorem mass_percentage_Al_in_AlBr3_approx :
  abs (mass_percentage_Al_in_AlBr3 - 10.11) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_in_AlBr3_approx_l1081_108141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1081_108124

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + Real.sqrt (a * b) + (a * b * c) ^ (1/3) ≤ 4/3 * (a + b + c) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1081_108124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_probability_l1081_108103

theorem coin_probability (p : ℝ) (h1 : 0 < p) (h2 : p < 1/2) :
  (Nat.choose 5 2 : ℝ) * p^2 * (1-p)^3 = 1/5 → p = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_probability_l1081_108103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_interval_of_g_l1081_108138

open Real

theorem monotonic_interval_of_g (w : ℝ) (h_w : w > 0) :
  let f := λ x => 1/2 - (cos (w*x))^2
  let T := π / 2
  let m := π / 8
  let g := λ x => 2 * sin (2*x - m) + 1
  (∀ x, f (x + T) = f x) →
  (∀ x₁ x₂, x₁ < x₂ → x₁ ∈ Set.Icc π (5*π/4) → x₂ ∈ Set.Icc π (5*π/4) → g x₁ < g x₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_interval_of_g_l1081_108138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_A_and_C_l1081_108116

-- Define the line segment AB and points A, B, C, D
noncomputable def AB : ℝ := 1  -- We can normalize AB to 1 for simplicity
noncomputable def A : ℝ := 0   -- A is at the start of the segment
noncomputable def B : ℝ := AB  -- B is at the end of the segment
noncomputable def D : ℝ := AB / 4  -- AD = AB/4 since AB = 4AD
noncomputable def C : ℝ := AB - AB / 8  -- C is at AB - BC, and BC = AB/8 since AB = 8BC

-- State the theorem
theorem probability_between_A_and_C : (C - A) / (B - A) = 7 / 8 := by
  -- Expand the definitions
  unfold C B A AB
  -- Simplify the expression
  simp [sub_zero, div_self]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_A_and_C_l1081_108116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_properties_l1081_108102

/-- Recursive definition of sequence P -/
def P : ℕ → ℤ → List ℕ → ℤ
| 0, a₀, _ => a₀
| 1, a₀, a => a.head! * a₀ + 1
| (k+2), a₀, a => 
    let a_k := a.get! k
    a_k * P (k+1) a₀ a + P k a₀ a

/-- Recursive definition of sequence Q -/
def Q : ℕ → ℤ → List ℕ → ℤ
| 0, _, _ => 1
| 1, _, a => a.head!
| (k+2), a₀, a => 
    let a_k := a.get! k
    a_k * Q (k+1) a₀ a + Q k a₀ a

/-- Definition of continued fraction -/
noncomputable def contFrac (a₀ : ℤ) (a : List ℕ) : ℚ :=
  sorry

/-- Main theorem -/
theorem sequences_properties 
  (a₀ : ℤ) (a : List ℕ) (n : ℕ) (h_n : n ≤ a.length) :
  (∀ k : ℕ, k ≤ n → 
    ((P k a₀ a : ℚ) / (Q k a₀ a : ℚ) = contFrac a₀ (a.take k)) ∧ 
    (P k a₀ a * Q (k-1) a₀ a - P (k-1) a₀ a * Q k a₀ a = (-1 : ℤ)^(k+1)) ∧
    (Nat.gcd (P k a₀ a).natAbs (Q k a₀ a).natAbs = 1)) :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_properties_l1081_108102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_negative_interval_l1081_108173

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4) / Real.log (1/2)

-- State the theorem
theorem f_increasing_on_negative_interval :
  StrictMonoOn f (Set.Iio (-2)) :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_negative_interval_l1081_108173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_factor_power_greatest_factor_is_14_l1081_108151

theorem greatest_factor_power (x : ℕ) : (3 : ℕ) ^ x ∣ 9 ^ 7 ↔ x ≤ 14 :=
by sorry

theorem greatest_factor_is_14 : 
  (14 : ℕ) = Finset.sup (Finset.range 15) (λ x => if (3 : ℕ) ^ x ∣ 9 ^ 7 then x else 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_factor_power_greatest_factor_is_14_l1081_108151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_area_l1081_108130

/-- The area of a right triangle with hypotenuse c, where the projection of the right angle
    vertex onto the hypotenuse divides it into two segments with a special ratio property. -/
theorem right_triangle_special_area (c : ℝ) (h : c > 0) :
  ∃ (x : ℝ), 0 < x ∧ x < c ∧
  (c - x) / x = x / c ∧
  ∃ (S : ℝ), S = (c^2 * Real.sqrt (Real.sqrt 5 - 2)) / 2 ∧
  S = (1/2) * c * Real.sqrt (c^2 * (Real.sqrt 5 - 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_area_l1081_108130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_price_before_discount_l1081_108194

/-- 
Given an article whose price after a 42% decrease is 1050 (in some currency units),
this theorem states that the original price of the article was approximately 1810.34 (in the same currency units).
-/
theorem article_price_before_discount (price_after_discount : ℝ) : 
  price_after_discount = 1050 → 
  ∃ (original_price : ℝ), 
    original_price * (1 - 0.42) = price_after_discount ∧ 
    abs (original_price - 1810.34) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_price_before_discount_l1081_108194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_nonnegative_l1081_108191

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if abs x ≥ 1 then x^2 else x

-- Define the properties of g
def is_quadratic (g : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, g x = a * x^2 + b * x + c

-- Theorem statement
theorem range_of_g_nonnegative
  (g : ℝ → ℝ)
  (h_quad : is_quadratic g)
  (h_range : Set.range (f ∘ g) = Set.Ici 0) :
  Set.range g = Set.Ici 0 := by
  sorry

#check range_of_g_nonnegative

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_nonnegative_l1081_108191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_equals_one_l1081_108176

/-- The modulus of the complex number z = (1-2i)/(2+i) is equal to 1. -/
theorem modulus_of_z_equals_one : Complex.abs ((1 - 2*Complex.I) / (2 + Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_equals_one_l1081_108176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l1081_108131

-- Define the function f(x) = 2^x - x^3
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) - x^3

-- Theorem statement
theorem f_has_zero_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l1081_108131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_time_is_80_minutes_l1081_108187

-- Define the time taken for each pump to empty the pool
noncomputable def pump_a_time : ℝ := 4
noncomputable def pump_b_time : ℝ := 2

-- Define the rate at which each pump empties the pool
noncomputable def pump_a_rate : ℝ := 1 / pump_a_time
noncomputable def pump_b_rate : ℝ := 1 / pump_b_time

-- Define the combined rate of both pumps
noncomputable def combined_rate : ℝ := pump_a_rate + pump_b_rate

-- Define the time taken for both pumps to empty the pool together
noncomputable def combined_time : ℝ := 1 / combined_rate

-- Theorem stating that the combined time is equal to 80 minutes
theorem combined_time_is_80_minutes : 
  combined_time * 60 = 80 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_time_is_80_minutes_l1081_108187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_union_A_complement_B_l1081_108115

-- Define the sets A, B, and U
def A : Set ℝ := {x | -2 < x ∧ x ≤ 4}
def B : Set ℝ := {x | 2 - x < 1}
def U : Set ℝ := Set.univ

-- Theorem for part (1)
theorem intersection_A_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 4} :=
by sorry

-- Theorem for part (2)
theorem union_A_complement_B :
  A ∪ (Set.compl B) = {x : ℝ | x ≤ 4} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_union_A_complement_B_l1081_108115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangency_l1081_108132

/-- Represents a parabola with equation y^2 = 2px --/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a circle with equation x^2 + y^2 + 6x + 8 = 0 --/
def Circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + 6*p.1 + 8 = 0}

/-- The directrix of a parabola --/
def directrix (pa : Parabola) : Set ℝ := {x : ℝ | x = -pa.p/2}

/-- A line is tangent to a circle if it intersects the circle at exactly one point --/
def is_tangent (l : Set ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ c ∧ p.1 ∈ l

theorem parabola_circle_tangency (pa : Parabola) :
  is_tangent (directrix pa) Circle → pa.p = 4 ∨ pa.p = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangency_l1081_108132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perfect_sequence_l1081_108192

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, n > 0 → a n > 0

def b (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  Nat.gcd (a n) (a (n + 1))

def combined_sequence (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n % 2 = 0 then a (n / 2) else b a ((n - 1) / 2)

theorem exists_perfect_sequence :
  ∃ a : ℕ → ℕ,
    is_valid_sequence a ∧
    (∀ n : ℕ, ∃! m : ℕ, combined_sequence a m = n + 1) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perfect_sequence_l1081_108192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_120_deg_l1081_108175

/-- The area of a sector with given radius and central angle -/
noncomputable def sectorArea (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  (centralAngle / 360) * Real.pi * radius^2

/-- Theorem: The area of a sector with radius 3 and central angle 120° is 3π -/
theorem sector_area_120_deg : sectorArea 3 120 = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_120_deg_l1081_108175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_divisible_by_6_l1081_108156

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 10 ∧ n = 746302 + d * 10

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem valid_number_divisible_by_6 (n : ℕ) :
  is_valid_number n → (is_divisible_by_6 n ↔ n = 746322 ∨ n = 746382) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_divisible_by_6_l1081_108156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1081_108119

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 6) / (2*x - 4)

noncomputable def g (x a b c d : ℝ) : ℝ := (a*x^2 + b*x + c) / (x - d)

-- State the theorem
theorem intersection_point (a b c d : ℝ) :
  -- Conditions
  (∀ x, ¬(2*x - 4 = 0) → ¬(x - d = 0)) →  -- Same vertical asymptote
  (∃ m₁ m₂ b₁ b₂, m₁ * m₂ = -1 ∧ m₁ * 0 + b₁ = m₂ * 0 + b₂) →  -- Perpendicular oblique asymptotes intersecting on y-axis
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = g x₁ a b c d ∧ f x₂ = g x₂ a b c d) →  -- Two intersection points
  (∃ y, f (-2) = g (-2) a b c d ∧ f (-2) = y) →  -- One intersection at x = -2
  -- Conclusion
  ∃ y, f 4 = g 4 a b c d ∧ f 4 = y ∧ y = -1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1081_108119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_correct_l1081_108113

/-- Calculate the selling price of a machine given various expenses, exchange rates, taxes, and profit margin -/
def calculate_selling_price (
  purchase_price : ℝ)
  (repair_cost_eur : ℝ)
  (transportation_cost_gbp : ℝ)
  (usd_to_eur_initial : ℝ)
  (usd_to_gbp_initial : ℝ)
  (eur_to_gbp_initial : ℝ)
  (usd_to_eur_final : ℝ)
  (usd_to_gbp_final : ℝ)
  (eur_to_gbp_final : ℝ)
  (import_tax_rate : ℝ)
  (repair_tax_rate : ℝ)
  (transportation_tax_rate : ℝ)
  (profit_margin : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the calculated selling price matches the expected result -/
theorem selling_price_correct : 
  calculate_selling_price 
    12000 -- purchase_price
    4000 -- repair_cost_eur
    1000 -- transportation_cost_gbp
    0.85 -- usd_to_eur_initial
    0.75 -- usd_to_gbp_initial
    0.9 -- eur_to_gbp_initial
    0.8 -- usd_to_eur_final
    0.7 -- usd_to_gbp_final
    0.875 -- eur_to_gbp_final
    0.05 -- import_tax_rate
    0.1 -- repair_tax_rate
    0.03 -- transportation_tax_rate
    0.5 -- profit_margin
  = 28724.70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_correct_l1081_108113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_xy_value_l1081_108163

theorem min_xy_value (x y : ℝ) 
  (h : 2 * (Real.cos (x + y - 1))^2 = ((x + 1)^2 + (y - 1)^2 - 2*x*y) / (x - y + 1)) : 
  x * y ≥ (1/4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_xy_value_l1081_108163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1081_108166

noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arccos (x^3 - x))

theorem domain_of_g :
  {x : ℝ | g x ≠ 0 ∧ x ∈ Set.Icc (-1) 1} = Set.Ioc (-1) 0 ∪ Set.Ioc 0 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1081_108166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_faces_are_trapezoids_prism_lateral_edges_properties_l1081_108155

-- Define a point in 3D space
structure Point3D where
  x : Real
  y : Real
  z : Real

-- Define a polygon
structure Polygon where
  vertices : List Point3D

-- Define a frustum
structure Frustum where
  base : Polygon
  top : Polygon
  lateral_faces : List Polygon

-- Define a prism
structure Prism where
  base : Polygon
  top : Polygon
  lateral_edges : List (Point3D × Point3D)

-- Define properties
def is_trapezoid (p : Polygon) : Prop := sorry

def are_equal_length (edges : List (Point3D × Point3D)) : Prop := sorry

def may_not_be_perpendicular (edges : List (Point3D × Point3D)) (base : Polygon) : Prop := sorry

-- Theorem statements
theorem frustum_lateral_faces_are_trapezoids (f : Frustum) :
  ∀ face ∈ f.lateral_faces, is_trapezoid face := by sorry

theorem prism_lateral_edges_properties (p : Prism) :
  are_equal_length p.lateral_edges ∧ may_not_be_perpendicular p.lateral_edges p.base := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_faces_are_trapezoids_prism_lateral_edges_properties_l1081_108155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l1081_108111

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (1 - Complex.I) / (1 - 2 * Complex.I)
  Complex.im z = 1/5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l1081_108111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l1081_108107

-- Define f as a noncomputable function from ℝ to ℝ
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x^2 - 1)
def domain_f_comp (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Define the domain of f(x)
def domain_f (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 8

theorem domain_equivalence :
  (∀ x, domain_f_comp x ↔ 0 ≤ x^2 - 1 ∧ x^2 - 1 ≤ 3) →
  (∀ y, domain_f y ↔ ∃ x, domain_f_comp x ∧ y = x^2 - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l1081_108107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_a_to_b_l1081_108145

/-- Proves that the ratio of a's speed to b's speed is 2:1 given the conditions -/
theorem speed_ratio_a_to_b (k : ℝ) (b_speed : ℝ) : 
  b_speed > 0 →
  k * b_speed + b_speed = 1 / 8 →
  b_speed = 1 / 24 →
  k = 2 := by
  intros h1 h2 h3
  -- Proof steps would go here
  sorry

#check speed_ratio_a_to_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_a_to_b_l1081_108145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_l1081_108160

/-- A function that returns true if a five-digit number satisfies the given conditions -/
def satisfiesConditions (n : ℕ) : Bool :=
  n ≥ 40000 && n < 100000 &&
  (let digits := n.digits 10
   digits.length = 5 &&
   (digits[1]! * digits[2]! * digits[3]!) < 10)

/-- The count of five-digit numbers satisfying the conditions -/
def countSatisfyingNumbers : ℕ := (List.range 100000).filter satisfiesConditions |>.length

/-- Theorem stating that the count of satisfying numbers is 2580 -/
theorem count_satisfying_numbers : countSatisfyingNumbers = 2580 := by
  sorry

#eval countSatisfyingNumbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_l1081_108160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_greater_than_S_S_4_equals_32_T_3_equals_16_l1081_108158

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n + 3

-- Define b_n
def b (n : ℕ) : ℚ :=
  if n % 2 = 1 then a n - 6 else 2 * a n

-- Define S_n (sum of first n terms of a_n)
def S (n : ℕ) : ℚ := (n + 4) * n

-- Define T_n (sum of first n terms of b_n)
noncomputable def T (n : ℕ) : ℚ :=
  if n % 2 = 0 then
    (n * (3 * n + 7)) / 2
  else
    (3 * n^2 + 5 * n - 10) / 2

-- State the theorem
theorem T_greater_than_S (n : ℕ) (h : n > 5) : T n > S n := by
  sorry

-- Additional conditions to verify the setup
theorem S_4_equals_32 : S 4 = 32 := by
  sorry

theorem T_3_equals_16 : T 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_greater_than_S_S_4_equals_32_T_3_equals_16_l1081_108158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_some_number_value_l1081_108167

theorem some_number_value : ∀ some_number : ℝ, 
  (some_number * 10^2) * (4 * 10^(-2 : ℤ)) = 12 → some_number = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_some_number_value_l1081_108167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_product_equality_l1081_108199

theorem sqrt_product_equality (p : ℝ) (hp : p > 0) :
  Real.sqrt (45 * p) * Real.sqrt (15 * p) * (Real.sqrt (10 * p^3))^(1/3) = 150 * (5 : ℝ)^(1/3) * p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_product_equality_l1081_108199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_inscribed_quadrilateral_l1081_108184

-- Define the circle
variable (r : ℝ) -- radius of the circle

-- Define the points A, B, C, D on the circle
variable (A B C D : ℝ × ℝ)

-- Define the inscribed quadrilateral ABCD
def inscribed_quadrilateral (A B C D : ℝ × ℝ) (r : ℝ) : Prop :=
  (A.1 - r)^2 + A.2^2 = r^2 ∧
  (B.1 - r)^2 + B.2^2 = r^2 ∧
  (C.1 - r)^2 + C.2^2 = r^2 ∧
  (D.1 - r)^2 + D.2^2 = r^2

-- AC is a diameter
def ac_diameter (A C : ℝ × ℝ) (r : ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (2*r)^2

-- Angle DAC is 30°
def angle_dac_30 (A C D : ℝ × ℝ) : Prop :=
  let v1 := (D.1 - A.1, D.2 - A.2)
  let v2 := (C.1 - A.1, C.2 - A.2)
  Real.cos (Real.pi/6) = (v1.1*v2.1 + v1.2*v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2))

-- Angle BAC is 45°
def angle_bac_45 (A B C : ℝ × ℝ) : Prop :=
  let v1 := (B.1 - A.1, B.2 - A.2)
  let v2 := (C.1 - A.1, C.2 - A.2)
  Real.cos (Real.pi/4) = (v1.1*v2.1 + v1.2*v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2))

-- The main theorem
theorem area_ratio_inscribed_quadrilateral
  (h1 : inscribed_quadrilateral A B C D r)
  (h2 : ac_diameter A C r)
  (h3 : angle_dac_30 A C D)
  (h4 : angle_bac_45 A B C) :
  -- The ratio of the area of ABCD to the area of the circle
  (Real.sqrt 3 + 2) / (2 * Real.pi) = sorry := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_inscribed_quadrilateral_l1081_108184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_32_67892_to_thousandth_l1081_108179

/-- Rounds a real number to the nearest thousandth -/
noncomputable def roundToThousandth (x : ℝ) : ℝ :=
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

/-- The theorem states that rounding 32.67892 to the nearest thousandth equals 32.679 -/
theorem round_32_67892_to_thousandth :
  roundToThousandth 32.67892 = 32.679 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_32_67892_to_thousandth_l1081_108179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_6575_to_hundredth_l1081_108162

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The theorem stating that rounding 2.6575 to the nearest hundredth results in 2.66 -/
theorem round_2_6575_to_hundredth :
  roundToHundredth 2.6575 = 2.66 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_6575_to_hundredth_l1081_108162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circumsphere_radius_exists_min_circumsphere_radius_l1081_108136

/-- A right triangular prism with a right angle at A and lateral face area of 16 -/
structure RightTriangularPrism where
  -- BC is the base edge opposite to the right angle
  bc : ℝ
  -- BB₁ is the height of the prism
  bb₁ : ℝ
  -- The area of the lateral face BCC₁B₁ is 16
  lateral_area : bc * bb₁ = 16
  -- Ensure bc and bb₁ are positive
  bc_pos : 0 < bc
  bb₁_pos : 0 < bb₁

/-- The radius of the circumscribed sphere of a right triangular prism -/
noncomputable def circumsphere_radius (p : RightTriangularPrism) : ℝ :=
  Real.sqrt ((p.bc / 2) ^ 2 + (p.bb₁ / 2) ^ 2)

theorem min_circumsphere_radius (p : RightTriangularPrism) :
    circumsphere_radius p ≥ 2 * Real.sqrt 2 := by
  sorry

theorem exists_min_circumsphere_radius :
    ∃ p : RightTriangularPrism, circumsphere_radius p = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circumsphere_radius_exists_min_circumsphere_radius_l1081_108136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_347_sequence_l1081_108135

def has_consecutive_347 (m n : ℕ) : Prop :=
  ∃ k : ℕ, (1000 * (10^k * m % n)) / n = 347

theorem smallest_n_with_347_sequence :
  (∃ m : ℕ, m < 1000 ∧ Nat.Coprime m 1000 ∧ has_consecutive_347 m 1000) ∧
  (∀ n : ℕ, n < 1000 → ¬∃ m : ℕ, m < n ∧ Nat.Coprime m n ∧ has_consecutive_347 m n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_347_sequence_l1081_108135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mixture_nuts_percentage_l1081_108181

/-- Represents the composition of a trail mix -/
structure TrailMix where
  nuts : ℝ
  dried_fruit : ℝ
  chocolate_chips : ℝ

/-- The combined mixture of two trail mixes -/
def combine_trail_mix (tm1 tm2 : TrailMix) : TrailMix :=
  { nuts := tm1.nuts + tm2.nuts,
    dried_fruit := tm1.dried_fruit + tm2.dried_fruit,
    chocolate_chips := tm1.chocolate_chips + tm2.chocolate_chips }

theorem combined_mixture_nuts_percentage 
  (sue_mix : TrailMix)
  (jane_mix : TrailMix)
  (h1 : sue_mix.nuts = 0.3)
  (h2 : sue_mix.dried_fruit = 0.7)
  (h3 : jane_mix.nuts = 0.6)
  (h4 : jane_mix.chocolate_chips = 0.4)
  (h5 : (combine_trail_mix sue_mix jane_mix).dried_fruit / 2 = 0.35) :
  (combine_trail_mix sue_mix jane_mix).nuts / 2 = 0.45 := by
  sorry

#check combined_mixture_nuts_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mixture_nuts_percentage_l1081_108181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_score_is_two_l1081_108112

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define a function to represent the score of each student
def score : Student → ℕ := sorry

-- Define the quiz properties
def num_questions : ℕ := 5

-- Axioms based on the problem conditions
axiom b_score_higher : score Student.B = score Student.C + 1
axiom a_d_contrary : ∀ q : Fin 5, q ≠ 4 → (score Student.A + score Student.D = 4)
axiom fourth_question_wrong : score Student.A + score Student.D = 4

-- The theorem to prove
theorem d_score_is_two : score Student.D = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_score_is_two_l1081_108112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yoga_studio_average_weight_l1081_108161

/-- Represents the average weight of a group of people -/
structure GroupWeight where
  count : ℕ
  avgWeight : ℝ

/-- Calculates the total weight of a group -/
def totalWeight (g : GroupWeight) : ℝ := g.count * g.avgWeight

/-- Yoga studio participant data -/
def yogaStudioData : List GroupWeight := [
  ⟨10, 200⟩,  -- Men above 50
  ⟨7, 180⟩,   -- Men below 50
  ⟨12, 156⟩,  -- Women above 40
  ⟨5, 120⟩,   -- Women below 40
  ⟨6, 100⟩,   -- Children aged 10-15
  ⟨4, 80⟩     -- Children below 10
]

/-- Theorem: The average weight of all participants in the yoga studio is approximately 151.18 pounds -/
theorem yoga_studio_average_weight :
  let totalWeight := (yogaStudioData.map totalWeight).sum
  let totalParticipants := (yogaStudioData.map (·.count)).sum
  abs ((totalWeight / totalParticipants : ℝ) - 151.18) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yoga_studio_average_weight_l1081_108161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_with_unique_grades_l1081_108106

/-- Represents a student's grades -/
def Grades (n : ℕ) := Fin (2 * n) → Fin 2

/-- Checks if one set of grades is better than another -/
def isBetter (n : ℕ) (g1 g2 : Grades n) : Prop :=
  (∀ i, g1 i ≥ g2 i) ∧ (∃ i, g1 i > g2 i)

/-- The set of all possible grade combinations -/
def AllGrades (n : ℕ) := {g : Grades n | ∀ i, g i < 2}

/-- The theorem to be proved -/
theorem max_students_with_unique_grades (n : ℕ) :
  ∃ (S : Set (Grades n)), S ⊆ AllGrades n ∧
    (∀ g1 g2, g1 ∈ S → g2 ∈ S → g1 ≠ g2) ∧
    (∀ g1 g2, g1 ∈ S → g2 ∈ S → ¬(isBetter n g1 g2)) ∧
    (∀ g, g ∈ AllGrades n → g ∉ S → ∃ g', g' ∈ S ∧ (isBetter n g' g ∨ isBetter n g g')) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_with_unique_grades_l1081_108106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eleven_equals_eleven_l1081_108183

/-- An arithmetic sequence -/
def arithmetic_sequence : ℕ → ℝ := sorry

/-- Sum of first n terms of the arithmetic sequence -/
noncomputable def S (n : ℕ) : ℝ := (n : ℝ) / 2 * (arithmetic_sequence 1 + arithmetic_sequence n)

/-- Point O -/
def O : ℝ × ℝ × ℝ := (0, 0, 0)

/-- Vector OA -/
def OA : ℝ × ℝ × ℝ := sorry

/-- Vector OB -/
def OB : ℝ × ℝ × ℝ := sorry

/-- Vector OC -/
def OC : ℝ × ℝ × ℝ := sorry

/-- Vectors are not collinear -/
axiom not_collinear : ¬ ∃ (t : ℝ), OC = t • OA + (1 - t) • OB

/-- Given condition -/
axiom condition : 2 • OC = (arithmetic_sequence 4) • OA + (arithmetic_sequence 8) • OB

theorem sum_eleven_equals_eleven : S 11 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eleven_equals_eleven_l1081_108183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_p_q_is_zero_l1081_108190

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def is_unit_vector (v : V) : Prop := ‖v‖ = 1

theorem angle_between_p_q_is_zero
  (p q r : V)
  (hp : is_unit_vector p)
  (hq : is_unit_vector q)
  (hr : is_unit_vector r)
  (h_sum : p + q + (2 : ℝ) • r = 0) :
  Real.arccos (inner p q) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_p_q_is_zero_l1081_108190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_lower_bound_l1081_108104

open Real

/-- The function f(x) = x^2 - bx + ln(x) -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + log x

/-- The derivative of f(x) -/
noncomputable def f' (b : ℝ) (x : ℝ) : ℝ := (2*x^2 - b*x + 1) / x

theorem f_difference_lower_bound 
  (b : ℝ) 
  (hb : b > 9/2) 
  (x₁ x₂ : ℝ) 
  (hx : x₁ < x₂) 
  (hroots : f' b x₁ = 0 ∧ f' b x₂ = 0) :
  f b x₁ - f b x₂ > 63/16 - 3 * log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_lower_bound_l1081_108104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crew_and_passenger_assignment_l1081_108100

-- Define the crew members and passengers
inductive CrewMember | Smirnov | Zhukov | Romanov
inductive Passenger | Smirnov | Zhukov | Romanov

-- Define the roles
inductive Role | Engineer | Conductor | Stoker

-- Define the cities
inductive City | Lviv | Omsk | Chita

-- Define the assignment of roles to crew members
def role_assignment : CrewMember → Role := sorry

-- Define the city where each passenger lives
def passenger_city : Passenger → City := sorry

-- Define the conditions
axiom conductor_lives_in_omsk :
  ∃ c : CrewMember, role_assignment c = Role.Conductor ∧ 
    passenger_city (match c with
      | CrewMember.Smirnov => Passenger.Smirnov
      | CrewMember.Zhukov => Passenger.Zhukov
      | CrewMember.Romanov => Passenger.Romanov) = City.Omsk

axiom passenger_romanov_in_lviv :
  passenger_city Passenger.Romanov = City.Lviv

axiom passenger_zhukov_not_math_prof :
  ∀ p : Passenger, p = Passenger.Zhukov → ¬(∃ c : City, passenger_city p = c ∧ c ≠ City.Omsk)

axiom conductor_namesake_in_chita :
  ∃ c : CrewMember, role_assignment c = Role.Conductor ∧ 
    passenger_city (match c with
      | CrewMember.Smirnov => Passenger.Smirnov
      | CrewMember.Zhukov => Passenger.Zhukov
      | CrewMember.Romanov => Passenger.Romanov) = City.Chita

axiom smirnov_beats_stoker :
  ∃ s : CrewMember, role_assignment s = Role.Stoker ∧ s ≠ CrewMember.Smirnov

-- Theorem to prove
theorem crew_and_passenger_assignment :
  (role_assignment CrewMember.Smirnov = Role.Engineer) ∧
  (role_assignment CrewMember.Zhukov = Role.Conductor) ∧
  (role_assignment CrewMember.Romanov = Role.Stoker) ∧
  (passenger_city Passenger.Romanov = City.Lviv) ∧
  (passenger_city Passenger.Zhukov = City.Omsk) ∧
  (passenger_city Passenger.Smirnov = City.Chita) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crew_and_passenger_assignment_l1081_108100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l1081_108165

def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2 : ℝ) + (m^2 - 3 * m + 2 : ℝ) * Complex.I

theorem z_properties :
  (∀ m : ℝ, (z m).im = 0 ↔ m = 1 ∨ m = 2) ∧
  (∀ m : ℝ, (z m).re = 0 ↔ m = -1/2) ∧
  z 0^2 / (z 0 + 5 + 2 * Complex.I) = -32/25 - 24/25 * Complex.I :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l1081_108165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_reciprocals_l1081_108149

/-- Parabola defined by y^2 = 4x -/
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Line with slope 1 passing through (1,0) -/
def Line (x y : ℝ) : Prop := y = x - 1

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_line_intersection_sum_reciprocals :
  ∃ (P Q : ℝ × ℝ),
    Parabola P.1 P.2 ∧
    Parabola Q.1 Q.2 ∧
    Line P.1 P.2 ∧
    Line Q.1 Q.2 ∧
    P ≠ Q ∧
    (1 / distance P Focus + 1 / distance Q Focus = 1) := by
  sorry

#check parabola_line_intersection_sum_reciprocals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_reciprocals_l1081_108149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_audio_space_per_hour_approx_l1081_108153

/-- Represents the digital music library --/
structure MusicLibrary where
  total_days : ℕ
  total_space : ℕ
  audio_hours : ℕ
  video_hours : ℕ
  audio_space_per_hour : ℚ
  video_space_per_hour : ℚ

/-- The given conditions of the problem --/
def problem_library : MusicLibrary where
  total_days := 15
  total_space := 20000
  audio_hours := 24 * 15 / 2
  video_hours := 24 * 15 / 2
  audio_space_per_hour := 20000 / (540 : ℚ)
  video_space_per_hour := 2 * (20000 / (540 : ℚ))

/-- Theorem stating the solution to the problem --/
theorem audio_space_per_hour_approx (lib : MusicLibrary) :
  lib = problem_library →
  (lib.audio_space_per_hour * 1).floor = 37 := by
  sorry

#check audio_space_per_hour_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_audio_space_per_hour_approx_l1081_108153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_pricing_problem_l1081_108148

/-- A convenience store sells goods with the following conditions:
  - Original price: 20 yuan per piece
  - Cost price: 12 yuan per piece
  - Initial sales: 240 pieces per day
  - If price increases by 0.5 yuan, sales decrease by 10 pieces per day
  - If price decreases by 0.5 yuan, sales increase by 20 pieces per day
  - Target daily profit: 1980 yuan
-/
theorem store_pricing_problem :
  ∃ (price : ℝ), (price = 21 ∨ price = 23) ∧
  (let original_price : ℝ := 20
   let cost_price : ℝ := 12
   let initial_sales : ℝ := 240
   let price_increase : ℝ := 0.5
   let sales_decrease : ℝ := 10
   let price_decrease : ℝ := 0.5
   let sales_increase : ℝ := 20
   let target_profit : ℝ := 1980
   let sales_at_price : ℝ → ℝ := λ p ↦ initial_sales + (original_price - p) * (sales_increase + sales_decrease) / (price_increase + price_decrease)
   (price - cost_price) * (sales_at_price price) = target_profit) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_pricing_problem_l1081_108148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_perpendicular_lines_l1081_108120

/-- Definition of line l₁ -/
def l₁ (m : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ x + m * y + 6 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ (m - 2) * x + 3 * m * y + 2 * m = 0

/-- Theorem stating the conditions for l₁ and l₂ to be parallel -/
theorem parallel_lines (m : ℝ) : 
  (∀ x y, l₁ m x y ↔ l₂ m x y) ↔ (m = 0 ∨ m = 5) := by
  sorry

/-- Theorem stating the conditions for l₁ and l₂ to be perpendicular -/
theorem perpendicular_lines (m : ℝ) : 
  (∀ x₁ y₁ x₂ y₂, l₁ m x₁ y₁ ∧ l₂ m x₂ y₂ → (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) = 0) 
  ↔ (m = -1 ∨ m = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_perpendicular_lines_l1081_108120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l1081_108127

/-- An ellipse with a vertex at (2,0) and eccentricity √2/2 -/
noncomputable def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) + (p.2^2 / 2) = 1}

/-- A line passing through (1,0) -/
noncomputable def Line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * (p.1 - 1)}

/-- The area of triangle AMN -/
noncomputable def TriangleArea (m : ℝ) : ℝ :=
  (4 * Real.sqrt 2) / 5

theorem ellipse_and_line_equations :
  ∃ (m : ℝ),
    (∀ (x y : ℝ), (x, y) ∈ Ellipse ↔ (x^2 / 4) + (y^2 / 2) = 1) ∧
    (Line m = {p : ℝ × ℝ | Real.sqrt 2 * p.1 + p.2 - Real.sqrt 2 = 0} ∨
     Line m = {p : ℝ × ℝ | Real.sqrt 2 * p.1 - p.2 - Real.sqrt 2 = 0}) ∧
    TriangleArea m = (4 * Real.sqrt 2) / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l1081_108127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_combined_set_l1081_108144

def set_x : Set Nat := {n : Nat | 100 ≤ n ∧ n < 1000 ∧ Nat.Prime n}
def set_y : Set Nat := {n : Nat | n > 0 ∧ n < 100 ∧ n % 4 = 0}
def set_z : Set Nat := {n : Nat | n > 0 ∧ n < 200 ∧ n % 5 = 0}

def combined_set : Set Nat := set_x ∪ set_y ∪ set_z

theorem range_of_combined_set :
  ∃ (max min : Nat), max ∈ combined_set ∧ min ∈ combined_set ∧
  (∀ n ∈ combined_set, min ≤ n ∧ n ≤ max) ∧
  max - min = 993 :=
by
  -- We know from the problem that the max is 997 and the min is 4
  use 997, 4
  apply And.intro
  · sorry -- Prove 997 ∈ combined_set
  apply And.intro
  · sorry -- Prove 4 ∈ combined_set
  apply And.intro
  · sorry -- Prove ∀ n ∈ combined_set, 4 ≤ n ∧ n ≤ 997
  · norm_num -- Prove 997 - 4 = 993

#check range_of_combined_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_combined_set_l1081_108144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_odd_and_increasing_l1081_108110

-- Define the function f(x) = x^(1/3)
noncomputable def f (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem cube_root_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → x < 0 → f x < f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_odd_and_increasing_l1081_108110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percentage_l1081_108134

theorem car_profit_percentage (purchase_price repair_cost selling_price : ℝ)
  (h1 : purchase_price = 42000)
  (h2 : repair_cost = 12000)
  (h3 : selling_price = 64900) :
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  abs (profit_percentage - 20.19) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percentage_l1081_108134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1081_108133

noncomputable def f (x : ℝ) := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem f_inequality (x : ℝ) : 
  f x > f (2*x - 1) ↔ 1/3 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1081_108133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_theorem_l1081_108198

-- Define the triangle and its properties
structure Triangle where
  A : Point
  B : Point
  C : Point
  F : Point  -- Point for the exterior angle
  angleABC : ℝ
  angleBCA : ℝ
  angleFAB : ℝ

-- Define the theorem
theorem exterior_angle_theorem (t : Triangle) (x : ℝ) 
  (h1 : t.angleFAB = 70)
  (h2 : t.angleABC = x + 20)
  (h3 : t.angleBCA = 20 + x)
  : x = 15 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_theorem_l1081_108198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterexample_exists_l1081_108168

def select : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | 1, x, _, _, _, _, _ => x
  | 2, _, x, _, _, _, _ => x
  | 3, _, _, x, _, _, _ => x
  | 4, _, _, _, x, _, _ => x
  | 5, _, _, _, _, x, _ => x
  | 6, _, _, _, _, _, x => x
  | _, _, _, _, _, _, _ => 0

theorem counterexample_exists : ∃ (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ),
  (∀ i j k, i ≠ j → j ≠ k → i ≠ k → i < 7 → j < 7 → k < 7 →
    ¬(Nat.gcd (select i a₁ a₂ a₃ a₄ a₅ a₆) (select j a₁ a₂ a₃ a₄ a₅ a₆) = 1 ∧
      Nat.gcd (select j a₁ a₂ a₃ a₄ a₅ a₆) (select k a₁ a₂ a₃ a₄ a₅ a₆) = 1 ∧
      Nat.gcd (select i a₁ a₂ a₃ a₄ a₅ a₆) (select k a₁ a₂ a₃ a₄ a₅ a₆) = 1)) ∧
  (∀ i j k, i ≠ j → j ≠ k → i ≠ k → i < 7 → j < 7 → k < 7 →
    Nat.gcd (Nat.gcd (select i a₁ a₂ a₃ a₄ a₅ a₆) (select j a₁ a₂ a₃ a₄ a₅ a₆)) (select k a₁ a₂ a₃ a₄ a₅ a₆) = 1) := by
  sorry

#check counterexample_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterexample_exists_l1081_108168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_and_side_a_l1081_108122

noncomputable def f (x : ℝ) := Real.sin x * Real.sin x + Real.sin x * Real.cos x

theorem range_of_f_and_side_a :
  (∀ x ∈ Set.Icc (-π/4) (π/4), f x ∈ Set.Icc (-(Real.sqrt 2)/2 + 1/2) 1) ∧
  (∀ A B C a b c : ℝ,
    f B = 1 →
    b = Real.sqrt 2 →
    c = Real.sqrt 3 →
    Real.sin A * b = Real.sin B * a →
    Real.sin B * c = Real.sin C * b →
    Real.sin C * a = Real.sin A * c →
    a = (Real.sqrt 6 + Real.sqrt 2) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_and_side_a_l1081_108122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_to_diagonal_ratio_l1081_108143

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define the diagonal of a square
noncomputable def diagonal (s : Square) : ℝ := s.side * Real.sqrt 2

-- Theorem: The ratio of side length to diagonal length in a square is √2/2
theorem square_side_to_diagonal_ratio (s : Square) :
  s.side / (diagonal s) = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_to_diagonal_ratio_l1081_108143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_time_double_discount_l1081_108147

/-- True discount calculation for a given bill amount, time period, and discount rate. -/
noncomputable def trueDiscount (billAmount : ℝ) (timePeriod : ℝ) (discountRate : ℝ) : ℝ :=
  (billAmount * discountRate * timePeriod) / (1 + discountRate * timePeriod)

/-- Theorem stating that doubling the time period doubles the true discount. -/
theorem double_time_double_discount (billAmount timePeriod discountRate : ℝ) :
  billAmount > 0 →
  timePeriod > 0 →
  discountRate > 0 →
  trueDiscount billAmount timePeriod discountRate = 10 →
  billAmount = 110 →
  trueDiscount billAmount (2 * timePeriod) discountRate = 20 :=
by
  sorry

#check double_time_double_discount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_time_double_discount_l1081_108147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_not_won_is_fifty_l1081_108121

/-- Represents the ratio of games won to games lost -/
def wonToLostRatio : ℚ := 5 / 3

/-- Represents the ratio of games won to games tied -/
def wonToTiedRatio : ℚ := 5 / 2

/-- Calculate the percentage of games not won -/
def percentageNotWon (wonToLost : ℚ) (wonToTied : ℚ) : ℚ :=
  let total := wonToLost.num + wonToLost.den + (wonToLost.num * wonToTied.den) / wonToTied.num
  let notWon := wonToLost.den + (wonToLost.num * wonToTied.den) / wonToTied.num
  (notWon / total) * 100

theorem percentage_not_won_is_fifty :
  percentageNotWon wonToLostRatio wonToTiedRatio = 50 := by
  -- Expand the definition of percentageNotWon
  unfold percentageNotWon
  -- Simplify the rational expressions
  simp [wonToLostRatio, wonToTiedRatio]
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_not_won_is_fifty_l1081_108121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_l1081_108174

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  AB : ℝ  -- Side length AB
  BC : ℝ  -- Side length BC

-- Define the properties of the specific triangle
noncomputable def special_triangle : Triangle where
  A := 2 * Real.pi / 3 - Real.pi / 3  -- Derived from 2B = A + C and A + B + C = π
  B := Real.pi / 3  -- Given in the solution, but not directly in the problem
  C := 2 * Real.pi / 3 - Real.pi / 3  -- Derived from 2B = A + C and A + B + C = π
  AB := 1
  BC := 4

-- Theorem statement
theorem median_length (t : Triangle) (h1 : t.A + t.B + t.C = Real.pi) 
  (h2 : 2 * t.B = t.A + t.C) (h3 : t.AB = 1) (h4 : t.BC = 4) :
  let AD := Real.sqrt 3
  AD^2 = t.AB^2 + (t.BC/2)^2 - 2 * t.AB * (t.BC/2) * Real.cos t.B :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_l1081_108174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_to_prism_volume_ratio_l1081_108185

/-- A structure representing a right circular cone inscribed in a rectangular prism -/
structure InscribedCone where
  h : ℝ  -- height of both cone and prism
  w : ℝ  -- width of the rectangular base
  hw : w > 0  -- width is positive

/-- The volume of the inscribed cone -/
noncomputable def cone_volume (c : InscribedCone) : ℝ :=
  (1 / 3) * Real.pi * (c.w / 2)^2 * c.h

/-- The volume of the rectangular prism -/
def prism_volume (c : InscribedCone) : ℝ :=
  2 * c.w^2 * c.h

/-- The theorem stating the ratio of cone volume to prism volume -/
theorem cone_to_prism_volume_ratio (c : InscribedCone) :
  cone_volume c / prism_volume c = Real.pi / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_to_prism_volume_ratio_l1081_108185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_size_possibilities_l1081_108177

/-- Represents the number of people in the group -/
def n : ℕ := sorry

/-- Represents the number of messages sent by each person -/
def k : ℕ := sorry

/-- The total number of messages received by all people -/
def total_messages : ℕ := 440

/-- Each person sends messages to everyone except themselves -/
axiom messages_sent : ∀ (i : ℕ), i < n → k * (n - 1) = total_messages / n

/-- The theorem stating the possible values for the number of people -/
theorem group_size_possibilities : n = 2 ∨ n = 5 ∨ n = 11 := by
  sorry

#check group_size_possibilities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_size_possibilities_l1081_108177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_finite_set_with_property_l1081_108170

theorem no_finite_set_with_property : ¬∃ (M : Set ℝ), 
  (Finite M) ∧ 
  (∃ x y, x ∈ M ∧ y ∈ M ∧ x ≠ y) ∧ 
  (∀ a b, a ∈ M → b ∈ M → (2 * a - b^2) ∈ M) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_finite_set_with_property_l1081_108170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1081_108164

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

def satisfiesConditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 3 ∧
  Real.cos (t.B - t.C) = 9/10 ∧
  t.A = Real.pi/3

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : satisfiesConditions t) : 
  (∃ S : ℝ, S = (7 * Real.sqrt 3) / 10 ∧ S = (1/2) * t.b * t.c * Real.sin t.A) ∧
  (∀ x : ℝ, x = (t.b + t.c) / t.a → Real.sqrt 3 < x ∧ x ≤ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1081_108164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extremes_consecutive_integers_l1081_108101

theorem sum_of_extremes_consecutive_integers (n : ℕ) (z : ℚ) (h1 : Even n) (h2 : n > 0) :
  let b := z - (n - 1) / 2
  let seq := List.range n |>.map (λ i => b + ↑i)
  (seq.sum / ↑n = z) → (seq.head! + seq.getLast! = 2 * z) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extremes_consecutive_integers_l1081_108101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_roundings_same_direction_l1081_108128

/-- An infinite arithmetic progression of natural numbers -/
def ArithmeticProgression (a₀ d : ℕ) : ℕ → ℕ := fun n ↦ a₀ + n * d

/-- Rounds a real number to the nearest integer -/
noncomputable def roundToNearest (x : ℝ) : ℤ :=
  if x - ⌊x⌋ < 1/2 then ⌊x⌋ else ⌈x⌉

theorem not_all_roundings_same_direction (a₀ d : ℕ) (hd : d ≠ 0) :
  ¬ (∀ n : ℕ, (roundToNearest (Real.sqrt (ArithmeticProgression a₀ d n : ℝ)) : ℝ) >
               (Real.sqrt (ArithmeticProgression a₀ d n : ℝ))) ∧
  ¬ (∀ n : ℕ, (roundToNearest (Real.sqrt (ArithmeticProgression a₀ d n : ℝ)) : ℝ) <
               (Real.sqrt (ArithmeticProgression a₀ d n : ℝ))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_roundings_same_direction_l1081_108128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_of_f_l1081_108152

/-- The function f(x) = ((x+1)^2 + sin x) / (x^2 + 1) -/
noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_min_sum_of_f :
  ∃ (M N : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, N ≤ f x) ∧ (M + N = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_of_f_l1081_108152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersect_unique_intersections_l1081_108129

/-- The first curve -/
def curve1 (x : ℝ) : ℝ := 3 * x^2 - 15 * x - 15

/-- The second curve -/
def curve2 (x : ℝ) : ℝ := x^3 - 5 * x^2 + 8 * x - 4

/-- The x-coordinates of the intersection points -/
noncomputable def intersection_x : List ℝ := [-1, (9 - Real.sqrt 37) / 2, (9 + Real.sqrt 37) / 2]

/-- Theorem stating that the curves intersect at the given points -/
theorem curves_intersect :
  ∀ x ∈ intersection_x, curve1 x = curve2 x :=
by sorry

/-- Theorem stating that these are the only intersection points -/
theorem unique_intersections :
  ∀ x : ℝ, curve1 x = curve2 x → x ∈ intersection_x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersect_unique_intersections_l1081_108129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_quadrilateral_diagonal_length_l1081_108193

/-- A parallelogram with sides a and b -/
structure Parallelogram (a b : ℝ) where
  a_positive : 0 < a
  b_positive : 0 < b

/-- The quadrilateral formed by the angle bisectors of a parallelogram -/
def AngleBisectorQuadrilateral (a b : ℝ) (p : Parallelogram a b) : Type := sorry

/-- The length of a diagonal of the AngleBisectorQuadrilateral -/
noncomputable def DiagonalLength (a b : ℝ) (p : Parallelogram a b) : ℝ := sorry

/-- Theorem: The length of the diagonals of the quadrilateral formed by 
    the angle bisectors of a parallelogram with sides a and b is |a - b| -/
theorem angle_bisector_quadrilateral_diagonal_length 
  (a b : ℝ) (p : Parallelogram a b) : 
  DiagonalLength a b p = |a - b| := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_quadrilateral_diagonal_length_l1081_108193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_10_0_l1081_108189

theorem binomial_10_0 : Nat.choose 10 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_10_0_l1081_108189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_to_concentric_l1081_108154

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Inversion transformation with respect to a circle -/
noncomputable def inversion (c : Circle) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Two circles are concentric if they have the same center -/
def concentric (c1 c2 : Circle) : Prop :=
  c1.center = c2.center

/-- Two circles are non-intersecting if the distance between their centers
    is greater than the sum of their radii -/
def non_intersecting (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 > (c1.radius + c2.radius)^2

/-- A point is on a circle if its distance from the center equals the radius -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

theorem inversion_to_concentric
  (S1 S2 : Circle)
  (h : non_intersecting S1 S2) :
  ∃ (c : Circle), ∃ (S1' S2' : Circle),
    (∀ p, inversion c (inversion c p) = p) ∧
    (concentric S1' S2') ∧
    (∀ p, on_circle p S1 ↔ on_circle (inversion c p) S1') ∧
    (∀ p, on_circle p S2 ↔ on_circle (inversion c p) S2') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_to_concentric_l1081_108154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sine_sum_l1081_108196

noncomputable def arithmetic_sequence (a₁ : ℝ) : ℕ → ℝ := fun n => a₁ + (n - 1 : ℝ) * Real.pi

def S (a₁ : ℝ) : Set ℝ := {x | ∃ n : ℕ+, x = Real.sin (arithmetic_sequence a₁ n)}

theorem arithmetic_sequence_sine_sum (a₁ : ℝ) :
  (∃ a b : ℝ, S a₁ = {a, b} ∧ a ≠ b) → 
  ∃ a b : ℝ, S a₁ = {a, b} ∧ a + b = 0 := by
  sorry

#check arithmetic_sequence_sine_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sine_sum_l1081_108196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dev_tina_time_ratio_l1081_108139

/-- Represents the time taken by Dev and Tina together to complete the task -/
def T : ℝ := sorry

/-- Time taken by Dev working alone -/
def dev_time : ℝ := T + 20

/-- Time taken by Tina working alone -/
def tina_time : ℝ := T + 5

/-- The equation representing the combined work rate -/
axiom work_rate_equation : 1 / dev_time + 1 / tina_time = 1 / T

theorem dev_tina_time_ratio :
  dev_time / tina_time = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dev_tina_time_ratio_l1081_108139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_a_ln_a_equals_one_l1081_108150

-- Define the function f(x) = x^a - log_a(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^a - (Real.log x) / (Real.log a)

-- State the theorem
theorem inequality_implies_a_ln_a_equals_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x > 0, f a x ≥ 1) → a * Real.log a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_a_ln_a_equals_one_l1081_108150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_height_A_is_three_halves_l1081_108159

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 3 ∧
  Real.sqrt 3 * Real.sin t.C = (Real.sin t.A + Real.sqrt 3 * Real.cos t.A) * Real.sin t.B

-- Define the height from A to BC
noncomputable def height_A (t : Triangle) : Real :=
  (2 * triangle_area t) / t.b
  where
    triangle_area (t : Triangle) : Real := (1/2) * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem max_height_A_is_three_halves (t : Triangle) :
  triangle_conditions t →
  ∃ (h_max : Real), (∀ (t' : Triangle), triangle_conditions t' → height_A t' ≤ h_max) ∧ h_max = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_height_A_is_three_halves_l1081_108159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_speeding_problem_l1081_108142

theorem liam_speeding_problem (distance : ℝ) (actual_speed : ℝ) (early_time : ℝ) :
  distance = 20 →
  actual_speed = 40 →
  early_time = 4 / 60 →
  ∃ (ideal_speed : ℝ), 
    (distance / ideal_speed = distance / actual_speed + early_time) ∧
    (abs (actual_speed - ideal_speed - 4.71) < 0.01) :=
by
  intro h_distance h_speed h_early
  sorry

#check liam_speeding_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_speeding_problem_l1081_108142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1081_108188

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 / (x + 1)
noncomputable def g (a x : ℝ) : ℝ := a * x + 5 - 2 * a

-- State the theorem
theorem range_of_a :
  ∀ (a : ℝ), a > 0 →
  (∀ (x₁ : ℝ), x₁ ∈ Set.Icc 0 1 →
    ∃ (x₀ : ℝ), x₀ ∈ Set.Icc 0 1 ∧ g a x₀ = f x₁) →
  a ∈ Set.Icc (5/2) 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1081_108188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_to_polar_l1081_108182

theorem complex_sum_to_polar : 
  12 * Complex.exp (3 * Real.pi * Complex.I / 13) + 12 * Complex.exp (12 * Real.pi * Complex.I / 26) = 
  24 * Real.cos (Real.pi / 13) * Complex.exp (9 * Real.pi * Complex.I / 26) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_to_polar_l1081_108182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_is_integer_l1081_108108

theorem factorial_ratio_is_integer (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ k : ℤ, (Nat.factorial a * Nat.factorial b * Nat.factorial (a + b) : ℚ) / (Nat.factorial (2 * a) * Nat.factorial (2 * b)) = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_is_integer_l1081_108108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_squared_pure_imaginary_necessary_not_sufficient_l1081_108186

/-- Given non-zero real numbers a and b, and z = a + bi, "z^2 is a pure imaginary number" 
    is a necessary but not sufficient condition for "a = b". -/
theorem z_squared_pure_imaginary_necessary_not_sufficient (a b : ℝ) (hz : ℂ) 
    (ha : a ≠ 0) (hb : b ≠ 0) (hz_def : hz = a + b * Complex.I) : 
    (∃ k : ℝ, hz ^ 2 = k * Complex.I) → (a = b ∨ a = -b) ∧
    ¬((a = b ∨ a = -b) → (∃ k : ℝ, hz ^ 2 = k * Complex.I)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_squared_pure_imaginary_necessary_not_sufficient_l1081_108186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_m_range_l1081_108109

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (x + Real.pi/3) * (Real.sin (x + Real.pi/3) - Real.sqrt 3 * Real.cos (x + Real.pi/3))

theorem f_range_and_m_range :
  ∀ m : ℝ, 
  (∀ x ∈ Set.Icc 0 (Real.pi/6), m * (f x + Real.sqrt 3) + 2 = 0) →
  m ∈ Set.Icc (-2 * Real.sqrt 3 / 3) (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_m_range_l1081_108109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1081_108180

noncomputable section

/-- Triangle DEF with vertices D(0,0), E(0,3), and F(10,0) -/
def triangle_DEF : Set (ℝ × ℝ) :=
  {p | ∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ 0 ≤ t₂ ∧ t₁ + t₂ ≤ 1 ∧
    p = (10 * t₁, 3 * t₂)}

/-- The area of a triangle given its base and height -/
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

/-- The area of the region above the line y = b in triangle DEF -/
def area_above (b : ℝ) : ℝ := triangle_area 10 (3 - b)

/-- The total area of triangle DEF -/
def total_area : ℝ := triangle_area 10 3

/-- The theorem stating that the horizontal line y = b divides the triangle into two equal areas
    if and only if b = 1.5 -/
theorem equal_area_division (b : ℝ) : 
  area_above b = (1 / 2) * total_area ↔ b = 1.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1081_108180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_washington_trip_cost_per_person_l1081_108114

/-- The total cost per person for a group trip to Washington D.C. -/
theorem washington_trip_cost_per_person 
  (num_people : ℕ)
  (airfare_hotel_cost : ℚ)
  (food_cost : ℚ)
  (transportation_cost : ℚ)
  (h1 : num_people = 15)
  (h2 : airfare_hotel_cost = 13500)
  (h3 : food_cost = 4500)
  (h4 : transportation_cost = 3000) :
  (airfare_hotel_cost + food_cost + transportation_cost) / num_people = 1400 := by
  sorry

-- Remove the #eval line as it's causing issues with universe levels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_washington_trip_cost_per_person_l1081_108114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_length_is_65_l1081_108118

/-- Represents the properties of a rectangular lawn with roads --/
structure LawnWithRoads where
  width : ℝ
  roadWidth : ℝ
  travelCost : ℝ
  costPerSqm : ℝ

/-- Calculates the length of a lawn given its properties --/
noncomputable def calculateLawnLength (lawn : LawnWithRoads) : ℝ :=
  let totalRoadArea := lawn.travelCost / lawn.costPerSqm
  (totalRoadArea + 2 * lawn.roadWidth * lawn.width) / (2 * lawn.roadWidth)

/-- Theorem stating the length of the lawn under given conditions --/
theorem lawn_length_is_65 (lawn : LawnWithRoads) 
  (h_width : lawn.width = 40)
  (h_roadWidth : lawn.roadWidth = 10)
  (h_travelCost : lawn.travelCost = 3300)
  (h_costPerSqm : lawn.costPerSqm = 3) :
  calculateLawnLength lawn = 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_length_is_65_l1081_108118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_s4_s2_l1081_108140

/-- An arithmetic sequence with a non-zero common difference,
    where a_1, a_2, a_4 form a geometric sequence. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  d_nonzero : d ≠ 0
  geometric : a 2 ^ 2 = a 1 * a 4

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1 : ℚ) * seq.d)

/-- The ratio of S_4 to S_2 is 10/3 -/
theorem ratio_s4_s2 (seq : ArithmeticSequence) :
  S seq 4 / S seq 2 = 10 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_s4_s2_l1081_108140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1081_108172

/-- The circle C in Cartesian coordinates -/
def circle_C (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - a*y = 0

/-- The line l in Cartesian coordinates -/
def line_l (x y : ℝ) : Prop := 4*x + 3*y - 8 = 0

/-- The radius of the circle C -/
noncomputable def radius (a : ℝ) : ℝ := a/2

/-- The distance from the center of circle C to line l -/
noncomputable def center_to_line_distance (a : ℝ) : ℝ := |3*a - 16|/10

/-- The chord length of circle C cut by line l -/
noncomputable def chord_length (a : ℝ) : ℝ := 
  2 * Real.sqrt ((radius a)^2 - (center_to_line_distance a)^2)

theorem circle_line_intersection (a : ℝ) :
  a ≠ 0 →
  chord_length a = Real.sqrt 3 * radius a →
  a = 32 ∨ a = 32/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1081_108172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1081_108125

theorem problem_statement : 
  let p := ∃ x : ℝ, Real.sin x = Real.sqrt 2
  let q := ∀ x : ℝ, x^2 - x + 1 > 0
  (¬p) ∨ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1081_108125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_expression_l1081_108171

theorem divisibility_of_expression (a b : ℕ) 
  (ha : a > 0) (hb : b > 0)
  (h : (a + b^3) % (a^2 + 3*a*b + 3*b^2 - 1) = 0) :
  ∃ (k : ℕ), k > 0 ∧ (a^2 + 3*a*b + 3*b^2 - 1) % k^3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_expression_l1081_108171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_complex_expression_l1081_108123

theorem min_value_complex_expression (w : ℂ) (h : Complex.abs (w - (3 - 3*I)) = 3) :
  ∃ (min : ℝ), min = 17 ∧ 
  ∀ (z : ℂ), Complex.abs (z - (3 - 3*I)) = 3 → 
    (Complex.abs (z + (1 - I)))^2 + (Complex.abs (z - (7 - 2*I)))^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_complex_expression_l1081_108123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_cunninghams_lambs_l1081_108117

theorem farmer_cunninghams_lambs 
  (total_lambs : ℕ) 
  (white_lambs : ℕ) 
  (h1 : total_lambs = 6048)
  (h2 : white_lambs = 193) :
  total_lambs - white_lambs = 5855 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_cunninghams_lambs_l1081_108117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_cone_volume_ratio_l1081_108195

/-- A cone inscribed in a hemisphere with specific geometric properties -/
structure InscribedCone where
  R : ℝ  -- Radius of the hemisphere
  α : ℝ  -- Angle between the line and the base plane
  vertex_at_center : True  -- Vertex of cone at center of hemisphere base
  bases_parallel : True  -- Bases of cone and hemisphere are parallel

/-- The ratio of volumes of a hemisphere to its inscribed cone -/
noncomputable def volume_ratio (c : InscribedCone) : ℝ :=
  (2 * (Real.cos c.α)^2) / (Real.cos (2 * c.α) * Real.tan c.α)

/-- Theorem stating the volume ratio of a hemisphere to its inscribed cone -/
theorem hemisphere_cone_volume_ratio (c : InscribedCone) :
  volume_ratio c = (2 * (Real.cos c.α)^2) / (Real.cos (2 * c.α) * Real.tan c.α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_cone_volume_ratio_l1081_108195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l1081_108105

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line equation
def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p | a * p.1 + b * p.2 + c = 0}

-- Define the tangent line condition
def IsTangent (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) :=
  ∃ p, p ∈ l ∩ c ∧ ∀ q, q ∈ l ∩ c → q = p

theorem circle_and_tangent_line 
  (C : Set (ℝ × ℝ))
  (h_C : C = Circle (3, 3) (Real.sqrt 10))
  (A : (2, 0) ∈ C)
  (B : (0, 4) ∈ C)
  (center_on_line : (3, 3) ∈ Line 1 2 (-9))
  (P : ℝ × ℝ)
  (P_coord : P = (-2, 8))
  (l : Set (ℝ × ℝ))
  (l_through_P : P ∈ l)
  (l_tangent : IsTangent l C) :
  (l = Line 1 3 (-22) ∨ l = Line 3 1 (-2)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l1081_108105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_transformation_defense_l1081_108169

-- Define the types of natural disasters
inductive NaturalDisaster
| ColdWave
| Earthquake
| HeavyRain
| Typhoon
deriving Inhabited

-- Define the project location
def projectLocation : String := "rural Fujian"

-- Define the roof types
inductive RoofType
| Flat
| Sloped

-- Define the project
structure SlopeTransformationProject where
  location : String
  originalRoof : RoofType
  transformedRoof : RoofType

-- Define the main natural disasters in Fujian
def mainDisastersInFujian : List NaturalDisaster := [NaturalDisaster.HeavyRain, NaturalDisaster.Typhoon]

-- Define the properties of roof types
def canWithstandHeavyRain (roof : RoofType) : Prop :=
  match roof with
  | RoofType.Flat => true
  | RoofType.Sloped => true

def improvesWaterDrainage (roof : RoofType) : Prop :=
  match roof with
  | RoofType.Flat => false
  | RoofType.Sloped => true

def increasesWindAffectedArea (roof : RoofType) : Prop :=
  match roof with
  | RoofType.Flat => false
  | RoofType.Sloped => true

-- Define the theorem
theorem slope_transformation_defense (project : SlopeTransformationProject) :
  project.location = projectLocation ∧
  project.originalRoof = RoofType.Flat ∧
  project.transformedRoof = RoofType.Sloped ∧
  mainDisastersInFujian = [NaturalDisaster.HeavyRain, NaturalDisaster.Typhoon] ∧
  canWithstandHeavyRain project.originalRoof ∧
  improvesWaterDrainage project.transformedRoof ∧
  increasesWindAffectedArea project.transformedRoof →
  NaturalDisaster.Typhoon = (
    mainDisastersInFujian.filter (fun d => 
      match d with
      | NaturalDisaster.Typhoon => true
      | _ => false
    )
  ).head! :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_transformation_defense_l1081_108169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_candidate_l1081_108137

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def candidate_number (A : ℕ) : ℕ := 103860 + A

theorem unique_prime_candidate : 
  ∃! A : ℕ, A < 10 ∧ is_prime (candidate_number A) ∧ candidate_number A = 103861 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_candidate_l1081_108137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l1081_108126

theorem sin_cos_relation (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : 
  Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l1081_108126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1081_108178

-- Define the conversion factor from km/hr to m/s
noncomputable def kmhr_to_ms (v : ℝ) : ℝ := v * (5/18)

-- Define the train's properties
def train_length : ℝ := 300
def train_speed_kmhr : ℝ := 250

-- Define the time to cross function
noncomputable def time_to_cross (length : ℝ) (speed : ℝ) : ℝ := length / speed

-- Theorem statement
theorem train_crossing_time :
  let train_speed_ms := kmhr_to_ms train_speed_kmhr
  abs (time_to_cross train_length train_speed_ms - 4.32) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1081_108178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_parabola_focus_l1081_108146

/-- Represents a line in 2D space -/
structure Line (α : Type*) [LinearOrderedField α] where
  slope : α
  intercept : α

/-- Represents a hyperbola -/
structure Hyperbola (α : Type*) [LinearOrderedField α] where
  a : α
  b : α

/-- Checks if a point is on a line -/
def Line.contains {α : Type*} [LinearOrderedField α] (l : Line α) (p : α × α) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

/-- Checks if a line is an asymptote of a hyperbola -/
def Line.isAsymptoteOf {α : Type*} [LinearOrderedField α] (l : Line α) (h : Hyperbola α) : Prop :=
  l.slope = h.b / h.a ∨ l.slope = -h.b / h.a

/-- Checks if two lines are parallel -/
def Line.isParallelTo {α : Type*} [LinearOrderedField α] (l1 l2 : Line α) : Prop :=
  l1.slope = l2.slope

/-- Checks if two lines are perpendicular -/
def Line.isPerpendicularTo {α : Type*} [LinearOrderedField α] (l1 l2 : Line α) : Prop :=
  l1.slope * l2.slope = -1

/-- Given a hyperbola C and a parabola with the following properties:
    1. The equation of C is (x²/a²) - (y²/b²) = 1, where a > 0 and b > 0
    2. There's a parabola with equation y² = 4x
    3. Line l passes through the focus of the parabola and point (0, b)
    4. One asymptote of C is parallel to l
    5. The other asymptote of C is perpendicular to l
    Then, the equation of hyperbola C is x² - y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (l : Line ℝ), 
    (l.contains (1, 0)) ∧ 
    (l.contains (0, b)) ∧
    (∃ (asym1 asym2 : Line ℝ),
      (asym1.isAsymptoteOf ⟨a, b⟩) ∧
      (asym2.isAsymptoteOf ⟨a, b⟩) ∧
      (asym1.isParallelTo l) ∧
      (asym2.isPerpendicularTo l))) →
  (∀ x y : ℝ, (x^2/a^2) - (y^2/b^2) = 1 ↔ x^2 - y^2 = 1) :=
sorry

/-- The focus of the parabola y² = 4x is at (1, 0) -/
theorem parabola_focus : 
  ∀ x y : ℝ, y^2 = 4*x → (1, 0) = (x, y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_parabola_focus_l1081_108146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_39_l1081_108197

-- Define the vertices of the triangle
noncomputable def A : ℝ × ℝ := (-2, 3)
noncomputable def B : ℝ × ℝ := (8, -1)
noncomputable def C : ℝ × ℝ := (10, 6)

-- Define the function to calculate the area of a triangle given its vertices
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p
  let (x2, y2) := q
  let (x3, y3) := r
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Theorem statement
theorem triangle_area_is_39 : triangleArea A B C = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_39_l1081_108197
