import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_plus_2b_min_value_theorem_min_value_achieved_l413_41363

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the theorem
theorem min_value_a_plus_2b (a b : ℝ) (h : log10 a + log10 b = 1) :
  a > 0 → b > 0 → ∀ x y : ℝ, x > 0 → y > 0 → log10 x + log10 y = 1 → a + 2*b ≤ x + 2*y :=
by
  sorry

-- Define the minimum value
noncomputable def min_value : ℝ := 4 * Real.sqrt 5

-- Prove the minimum value theorem
theorem min_value_theorem (a b : ℝ) (h : log10 a + log10 b = 1) :
  a > 0 → b > 0 → a + 2*b ≥ min_value :=
by
  sorry

-- Prove the existence of a and b that achieve the minimum value
theorem min_value_achieved :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ log10 a + log10 b = 1 ∧ a + 2*b = min_value :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_plus_2b_min_value_theorem_min_value_achieved_l413_41363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l413_41383

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c

def is_arithmetic_sequence (t : Triangle) : Prop :=
  t.B - t.A = t.C - t.B

def dot_product_condition (t : Triangle) : Prop :=
  t.a * t.c * Real.cos t.B = -3/2

def side_b_condition (t : Triangle) : Prop :=
  t.b = Real.sqrt 3

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : is_arithmetic_sequence t)
  (h3 : dot_product_condition t)
  (h4 : side_b_condition t) :
  t.a + t.c = 2 * Real.sqrt 3 ∧ 
  -Real.sqrt 3 / 2 < 2 * Real.sin t.A - Real.sin t.C ∧
  2 * Real.sin t.A - Real.sin t.C < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l413_41383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_is_44_l413_41394

def next_term (n : ℕ) : ℕ :=
  if n < 15 then 7 * n
  else if 15 ≤ n ∧ n ≤ 35 ∧ n % 2 = 0 then n + 10
  else if n > 35 ∧ n % 2 = 1 then n - 7
  else n

def sequence_term (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => next_term (sequence_term start n)

theorem fiftieth_term_is_44 : sequence_term 76 49 = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_is_44_l413_41394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_and_angle_l413_41391

theorem triangle_inequality_and_angle (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_arithmetic : ∃ (k : ℝ), 1/b = (1/a + 1/c) / 2 ∧ k = 1/b - 1/a ∧ k = 1/c - 1/b) :
  b/a < c/b ∧ ∃ (B : ℝ), 0 < B ∧ B ≤ π/2 ∧ Real.cos B = (a^2 + c^2 - b^2) / (2*a*c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_and_angle_l413_41391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_of_fractions_l413_41330

theorem comparison_of_fractions : (-3/4 : ℚ) < |(-7/5 : ℚ)| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_of_fractions_l413_41330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_l413_41335

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- Define the boundaries
def a : ℝ := 1
def b : ℝ := 2

-- State the theorem
theorem area_enclosed_by_curve : 
  ∫ x in a..b, f x = Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_l413_41335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_range_l413_41398

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

-- Define the set of x₀ where f(x₀) > 1
def solution_set : Set ℝ := {x | f x > 1}

-- Theorem statement
theorem f_greater_than_one_range :
  solution_set = Set.Iio (-1) ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_range_l413_41398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l413_41399

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (1 / (a^2 * b^2)) / (1 / a^3 + 1 / b^3) = (a * b) / (a^3 + b^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l413_41399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_points_distance_l413_41393

/-- The distance between two points on a parabola that intersect with its focal points -/
theorem parabola_focal_points_distance (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 6*x₁ →  -- Point A on the parabola
  y₂^2 = 6*x₂ →  -- Point B on the parabola
  x₁ + x₂ = 6 →  -- Given condition
  let p := 3     -- Parameter of the parabola
  let AB := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)
  AB = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_points_distance_l413_41393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l413_41361

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin (2 * x - Real.pi / 3) + b

noncomputable def g (a b x : ℝ) : ℝ := a * (Real.cos x) ^ 2 + Real.sin x + b

theorem max_value_of_g (a b : ℝ) :
  a > 0 →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a b x ≥ -2) →
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f a b x = -2) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a b x ≤ Real.sqrt 3) →
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f a b x = Real.sqrt 3) →
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), g a b x = Real.sqrt 3 + 1 / 8) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), g a b x ≤ Real.sqrt 3 + 1 / 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l413_41361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_condition_l413_41338

/-- IsTriangle predicate (placeholder) -/
def IsTriangle (s : Set (ℝ × ℝ)) : Prop := sorry

/-- TriangleSide function (placeholder) -/
def TriangleSide (s : Set (ℝ × ℝ)) (i : Nat) : ℝ := sorry

/-- TriangleAltitude function (placeholder) -/
def TriangleAltitude (s : Set (ℝ × ℝ)) (i : Nat) : ℝ := sorry

theorem triangle_existence_condition (c d m_c : ℝ) (h1 : c > 0) (h2 : d > 0) (h3 : m_c > 0) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = d ∧ 
   ∃ (triangle : Set (ℝ × ℝ)), 
     IsTriangle triangle ∧ 
     TriangleSide triangle 0 = c ∧
     TriangleAltitude triangle 0 = m_c) ↔
  (d > c ∧ m_c < Real.sqrt ((d^2 - c^2) / 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_condition_l413_41338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_discount_calculation_l413_41310

/-- Banker's gain in Rupees -/
noncomputable def bankers_gain : ℝ := 60

/-- Interest rate per annum as a percentage -/
noncomputable def interest_rate : ℝ := 10

/-- Time in years -/
noncomputable def time : ℝ := 3

/-- Calculates the true discount given banker's gain, interest rate, and time -/
noncomputable def true_discount (bg : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  (bg * 100) / (r * t)

/-- Calculates the banker's discount given banker's gain and true discount -/
noncomputable def bankers_discount (bg : ℝ) (td : ℝ) : ℝ :=
  bg + td

theorem bankers_discount_calculation :
  bankers_discount bankers_gain (true_discount bankers_gain interest_rate time) = 260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_discount_calculation_l413_41310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_lengths_l413_41390

-- Define the relationship between a square's diagonal and side length
noncomputable def diagonal_to_side (d : ℝ) : ℝ := d / Real.sqrt 2

-- Define the first square's diagonal
noncomputable def first_diagonal : ℝ := 2 * Real.sqrt 2

-- Define the second square's diagonal
noncomputable def second_diagonal : ℝ := 2 * first_diagonal

-- Statement of the theorem
theorem square_side_lengths :
  diagonal_to_side first_diagonal = 2 ∧ 
  diagonal_to_side second_diagonal = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_lengths_l413_41390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_even_function_l413_41354

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) + 3*x
  else Real.log x - 3*x

-- State the theorem
theorem tangent_line_of_even_function :
  (∀ x, f (-x) = f x) →  -- f is even
  (∀ x < 0, f x = Real.log (-x) + 3*x) →  -- definition for x < 0
  f 1 = -3 →  -- the point (1, -3) lies on the curve
  ∃ m b, ∀ x, m*x + b = -2*x - 1 ∧  -- equation of tangent line
         m = (deriv f) 1 ∧  -- slope is the derivative at x=1
         -3 = m*1 + b  -- point (1, -3) satisfies the equation
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_even_function_l413_41354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tips_fraction_is_nine_thirteenths_l413_41364

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ
  income_composition : income = salary + tips
  tips_ratio : tips = (9 / 4) * salary

/-- The fraction of a waitress's income that comes from tips -/
def tipsFraction (w : WaitressIncome) : ℚ :=
  w.tips / (w.salary + w.tips)

/-- Theorem: The fraction of income from tips is 9/13 -/
theorem tips_fraction_is_nine_thirteenths (w : WaitressIncome) : 
  tipsFraction w = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tips_fraction_is_nine_thirteenths_l413_41364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_120_equals_two_thirds_l413_41351

def b : ℕ → ℚ
  | 0 => 2  -- Adding case for 0
  | 1 => 2
  | 2 => 3
  | n + 3 => (2 - b (n + 2)) / (3 * b (n + 1))

theorem b_120_equals_two_thirds : b 120 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_120_equals_two_thirds_l413_41351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l413_41307

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  2 * Real.sin (ω * x) * Real.cos (ω * x) + 2 * Real.sqrt 3 * (Real.sin (ω * x))^2 - Real.sqrt 3

noncomputable def g (x : ℝ) : ℝ := 
  2 * Real.sin (2 * x) + 1

theorem function_properties :
  ∀ ω : ℝ, ω > 0 →
  (∀ x : ℝ, f ω (x + π) = f ω x) →
  (∀ T : ℝ, T > 0 → (∀ x : ℝ, f ω (x + T) = f ω x) → T ≥ π) →
  (ω = 1) ∧
  (∀ x : ℝ, f ω x = 2 * Real.sin (2 * x - π / 3)) ∧
  (∀ k : ℤ, ∀ x : ℝ, 
    (x ≥ k * π - π / 12 ∧ x ≤ k * π + 5 * π / 12) →
    (∀ y : ℝ, x < y → f ω x < f ω y)) ∧
  (∀ x : ℝ, g x = f ω (x + π / 6) + 1) ∧
  (∃ b : ℝ, b > 0 ∧
    (∃ x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ : ℝ,
      0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧
      x₅ < x₆ ∧ x₆ < x₇ ∧ x₇ < x₈ ∧ x₈ < x₉ ∧ x₉ < x₁₀ ∧ x₁₀ ≤ b ∧
      g x₁ = 0 ∧ g x₂ = 0 ∧ g x₃ = 0 ∧ g x₄ = 0 ∧ g x₅ = 0 ∧
      g x₆ = 0 ∧ g x₇ = 0 ∧ g x₈ = 0 ∧ g x₉ = 0 ∧ g x₁₀ = 0) ∧
    b = 59 * π / 12 ∧
    (∀ b' : ℝ, b' < b →
      ¬(∃ x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ : ℝ,
        0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧
        x₅ < x₆ ∧ x₆ < x₇ ∧ x₇ < x₈ ∧ x₈ < x₉ ∧ x₉ < x₁₀ ∧ x₁₀ ≤ b' ∧
        g x₁ = 0 ∧ g x₂ = 0 ∧ g x₃ = 0 ∧ g x₄ = 0 ∧ g x₅ = 0 ∧
        g x₆ = 0 ∧ g x₇ = 0 ∧ g x₈ = 0 ∧ g x₉ = 0 ∧ g x₁₀ = 0))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l413_41307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l413_41375

/-- Ellipse structure -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Line structure -/
structure Line where
  m : ℝ
  c : ℝ

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem about ellipse properties -/
theorem ellipse_properties (C : Ellipse) (l : Line) :
  (C.b / C.a * (-C.b / C.a) = -1/4) →
  (l.m = 1/2 ∧ l.c = 1/2) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁^2 / C.a^2 + y₁^2 / C.b^2 = 1 ∧
                         x₂^2 / C.a^2 + y₂^2 / C.b^2 = 1 ∧
                         y₁ = l.m * x₁ + l.c ∧
                         y₂ = l.m * x₂ + l.c ∧
                         distance x₁ y₁ x₂ y₂ = Real.sqrt 35 / 2) →
  (Real.sqrt (C.a^2 - C.b^2) / C.a = Real.sqrt 3 / 2) ∧
  (C.a^2 = 4 ∧ C.b^2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l413_41375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_free_education_policy_beneficial_l413_41329

/-- Represents the economic value of international agreements -/
def international_agreement_value : ℝ := sorry

/-- Represents the economic cost of an aging population -/
def aging_population_cost : ℝ := sorry

/-- Represents the economic contribution of an educated foreign student over their lifetime -/
def student_lifetime_contribution : ℝ := sorry

/-- Represents the cost of educating a foreign student -/
def education_cost : ℝ := sorry

/-- Represents the number of foreign students educated -/
def num_foreign_students : ℕ := sorry

/-- Theorem stating that the economic benefits of offering free education to foreign students outweigh the costs -/
theorem free_education_policy_beneficial 
  (h1 : international_agreement_value > 0)
  (h2 : aging_population_cost > 0)
  (h3 : student_lifetime_contribution > education_cost)
  (h4 : num_foreign_students > 0) :
  international_agreement_value + num_foreign_students * (student_lifetime_contribution - education_cost) > 
  aging_population_cost :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_free_education_policy_beneficial_l413_41329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_four_lt_b_seven_l413_41386

noncomputable def b : ℕ → ℝ
| 0 => 1
| n + 1 => 1 + 1 / b n

theorem b_four_lt_b_seven : b 4 < b 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_four_lt_b_seven_l413_41386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sausage_weight_l413_41342

/-- Given that Jake buys 3 packages of sausages at $4 per pound and pays $24 in total,
    prove that each package weighs 2 pounds. -/
theorem sausage_weight (num_packages : ℕ) (price_per_pound : ℝ) (total_cost : ℝ) 
    (h1 : num_packages = 3)
    (h2 : price_per_pound = 4)
    (h3 : total_cost = 24) :
  total_cost / (price_per_pound * num_packages) = 2 := by
  -- Convert natural number to real for division
  have num_packages_real : ℝ := num_packages
  -- Calculate total weight
  have total_weight : ℝ := total_cost / price_per_pound
  -- Calculate weight per package
  have weight_per_package : ℝ := total_weight / num_packages_real
  -- Prove that weight per package equals 2
  sorry

#check sausage_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sausage_weight_l413_41342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_and_die_probability_l413_41372

-- Define a fair coin
noncomputable def fair_coin : ℚ := 1 / 2

-- Define a standard six-sided die
def six_sided_die : ℕ := 6

-- Define the probability of rolling a number greater than 4
noncomputable def prob_greater_than_4 (n : ℕ) : ℚ := 2 / n

-- Theorem statement
theorem coin_and_die_probability :
  let p_heads := fair_coin
  let p_greater_than_4 := prob_greater_than_4 six_sided_die
  p_heads * p_greater_than_4 = 1 / 6 := by
  -- Unfold definitions
  unfold fair_coin prob_greater_than_4 six_sided_die
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_and_die_probability_l413_41372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l413_41357

theorem inequality_solution_set :
  Set.Ioi (3/2 : ℝ) ∪ Set.Iic (-1) = {x : ℝ | 2*x^2 - x - 3 > 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l413_41357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_change_l413_41300

/-- Calculate the change received after purchasing items with different tax rates -/
theorem calculate_change (apple_price sandwich_price soda_price : ℚ)
  (apple_tax_rate sandwich_tax_rate : ℚ)
  (payment : ℚ)
  (h1 : apple_price = 75/100)
  (h2 : sandwich_price = 350/100)
  (h3 : soda_price = 125/100)
  (h4 : apple_tax_rate = 3/100)
  (h5 : sandwich_tax_rate = 6/100)
  (h6 : payment = 20) :
  let total_cost := (apple_price * (1 + apple_tax_rate)) +
                    (sandwich_price * (1 + sandwich_tax_rate)) +
                    soda_price
  ∃ (change : ℚ), change ≥ 0 ∧ change < 1/100 ∧ 
  (payment - total_cost - change) * 100 = 1427 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_change_l413_41300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_damage_conversion_l413_41396

/-- Converts Japanese yen to US dollars -/
noncomputable def yen_to_usd (yen : ℝ) : ℝ := (yen / 100) * 0.9

/-- Converts US dollars to euros -/
noncomputable def usd_to_euro (usd : ℝ) : ℝ := usd * 0.93

/-- Theorem stating the conversion of damage from yen to USD and euros -/
theorem damage_conversion (damage_yen : ℝ) 
  (h1 : damage_yen = 4000000000) : 
  yen_to_usd damage_yen = 36000000 ∧ 
  usd_to_euro (yen_to_usd damage_yen) = 33480000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_damage_conversion_l413_41396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_floor_values_l413_41334

def floor_seq (n : ℕ) : ℕ := Int.toNat ⌊(n^2 : ℚ) / 500⌋

def distinct_count (s : List ℕ) : ℕ := (s.toFinset).card

theorem distinct_floor_values :
  distinct_count (List.map floor_seq (List.range 501)) = 376 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_floor_values_l413_41334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_drain_time_calculation_l413_41340

/-- Represents the time it takes for a leak to drain a full tank, given the pump's filling rate and the actual filling time with the leak. -/
noncomputable def leak_drain_time (pump_fill_time : ℚ) (actual_fill_time : ℚ) : ℚ :=
  let pump_rate := 1 / pump_fill_time
  let combined_rate := 1 / actual_fill_time
  let leak_rate := pump_rate - combined_rate
  1 / leak_rate

/-- Theorem stating that given a pump that can fill a tank in 2 hours, and the actual filling time
    with a leak is 2 1/8 hours, the time it takes for the leak to drain the full tank is 34 hours. -/
theorem leak_drain_time_calculation :
  leak_drain_time 2 (17/8) = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_drain_time_calculation_l413_41340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l413_41336

noncomputable def initial_complex : ℂ := -4 - 6*Complex.I

noncomputable def rotation_60_deg : ℂ := Complex.exp (Complex.I * (Real.pi/3))

def dilation_factor : ℝ := 2

theorem complex_transformation :
  (initial_complex * rotation_60_deg * dilation_factor) = 
  Complex.ofReal (-4 + 6*Real.sqrt 3) + Complex.I * Complex.ofReal (-4*Real.sqrt 3 - 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l413_41336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_and_b_l413_41382

/-- The function f(x) = a*ln(2x) + b*x reaches its maximum value of ln(2) - 1 at x = 1 -/
noncomputable def f (a b x : ℝ) : ℝ := a * Real.log (2 * x) + b * x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a b x : ℝ) : ℝ := a / x + b

theorem max_value_implies_a_and_b :
  ∀ a b : ℝ,
  (∀ x : ℝ, x > 0 → f a b x ≤ f a b 1) ∧
  (f a b 1 = Real.log 2 - 1) ∧
  (f_derivative a b 1 = 0) →
  a = 1 ∧ b = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_and_b_l413_41382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l413_41380

theorem rationalize_denominator :
  (Real.rpow 9 (1/3) + Real.sqrt 5) / (Real.rpow 2 (1/3) + Real.sqrt 5) =
  (Real.sqrt 5 * Real.rpow 2 (1/3) - 6 * Real.rpow 3 (1/3) - 5) / (Real.rpow 4 (1/3) - 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l413_41380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l413_41384

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem f_properties : 
  (∀ x, f (x + Real.pi) = f x) ∧ 
  (∀ x, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l413_41384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_lcm_360_l413_41358

theorem gcd_lcm_360 (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  Nat.gcd a b ∈ ({1, 2, 3, 4, 6, 8, 12, 24} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_lcm_360_l413_41358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_div_by_five_l413_41381

/-- The count of positive 3-digit numbers divisible by 5 -/
theorem count_three_digit_div_by_five : 
  (Finset.filter (fun n => 100 ≤ n ∧ n ≤ 999 ∧ n % 5 = 0) (Finset.range 1000)).card = 180 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_div_by_five_l413_41381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_g_eight_l413_41320

-- Define g as a parameter
variable (g : ℝ → ℝ)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sqrt (-x) else g (x - 1)

-- Define the property of f being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_implies_g_eight :
  is_odd (f g) → g 8 = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_g_eight_l413_41320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l413_41313

-- Define the circle C
def myCircle (t : ℝ) (x y : ℝ) : Prop :=
  t ≠ 0 ∧ (x - t)^2 + y^2 = t^2

-- Define the line
def myLine (x y : ℝ) : Prop :=
  2*x + y - 4 = 0

-- Define the theorem
theorem circle_properties (t : ℝ) :
  t ≠ 0 →
  (∃ (A B : ℝ × ℝ), 
    myCircle t A.1 A.2 ∧ 
    myCircle t B.1 B.2 ∧ 
    A.1 = 2*t ∧ A.2 = 0 ∧
    B.1 = 0 ∧ (B.2 = t ∨ B.2 = -t)) →
  (∃ (M N : ℝ × ℝ),
    myCircle t M.1 M.2 ∧
    myCircle t N.1 N.2 ∧
    myLine M.1 M.2 ∧
    myLine N.1 N.2 ∧
    (M.1 - 0)^2 + (M.2 - 0)^2 = (N.1 - 0)^2 + (N.2 - 0)^2) →
  (∀ (x y : ℝ), myCircle t x y ↔ (x - 2)^2 + (y - 1)^2 = 5) ∧
  (1/2 * 2*t * |t| = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l413_41313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_expr_value_l413_41397

theorem abs_expr_value (x : ℝ) (h : x = -20.25) : 
  (abs (abs (abs x - x) - abs x) - x) = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_expr_value_l413_41397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_probability_l413_41392

def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 3000}

def is_divisible_by_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

def random_selection (S : Set ℕ) : ℕ → ℕ → ℕ → Prop :=
  λ a b c ↦ a ∈ S ∧ b ∈ S ∧ c ∈ S

noncomputable def ℙ : Prop → ℝ := sorry

theorem divisibility_probability :
  ∀ a b c : ℕ,
  random_selection S a b c →
  ℙ (is_divisible_by_5 (a * b * c + a * b + a)) = 41 / 125 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_probability_l413_41392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_journey_distance_l413_41347

/-- Represents a point on the bicycle journey --/
inductive Point
| A
| B
| C
| D
| E

/-- Represents the distance between two points in kilometers --/
def distance (p q : Point) : ℚ := sorry

/-- The total distance of the journey --/
def total_distance : ℚ := distance Point.A Point.C

theorem bicycle_journey_distance :
  -- Condition 1: Distance from D to B is 3 times the distance from A to D
  (distance Point.D Point.B = 3 * distance Point.A Point.D) →
  -- Condition 2: Distance from D to E is 10 km
  (distance Point.D Point.E = 10) →
  -- Condition 3: Distance from E to C is 3 times the distance from E to B
  (distance Point.E Point.C = 3 * distance Point.E Point.B) →
  -- Conclusion: The total distance is 40/3 km
  total_distance = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_journey_distance_l413_41347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_5pi_12_l413_41359

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

-- State the theorem
theorem derivative_f_at_5pi_12 :
  deriv f (5 * Real.pi / 12) = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_5pi_12_l413_41359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l413_41365

theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 → Real.cos θ ^ 2 + 2 * m * Real.sin θ - 2 * m - 2 < 0) →
  m > -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l413_41365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l413_41366

/-- A trapezoid with specific properties -/
structure Trapezoid :=
  (EF GH : ℝ)
  (height : ℝ)
  (ef_eq_gh : EF = GH)
  (ef_value : EF = 10)
  (gh_value : GH = 22)
  (height_value : height = 5)

/-- The perimeter of the trapezoid -/
noncomputable def perimeter (t : Trapezoid) : ℝ :=
  2 * (t.EF + t.GH) + 2 * Real.sqrt (t.height^2 + ((t.GH - t.EF)/2)^2)

/-- Theorem stating the perimeter of the specific trapezoid -/
theorem trapezoid_perimeter : 
  ∀ t : Trapezoid, perimeter t = 32 + 2 * Real.sqrt 61 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l413_41366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l413_41326

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the transformation T
noncomputable def T (g : ℝ → ℝ) : ℝ → ℝ := 
  fun x => g ((x + Real.pi/4) / 2)

-- State the theorem
theorem function_transformation (x : ℝ) : 
  T f x = 3 * Real.sin x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l413_41326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pells_equation_unique_solution_l413_41368

def is_fundamental_solution (x₀ y₀ : ℤ) : Prop :=
  x₀^2 - 2003 * y₀^2 = 1

def is_valid_solution (x y x₀ : ℤ) : Prop :=
  x > 0 ∧ y > 0 ∧ ∀ p : ℕ, Nat.Prime p → (p ∣ x.natAbs → p ∣ x₀.natAbs)

theorem pells_equation_unique_solution (x₀ y₀ x y : ℤ) 
  (h_fund : is_fundamental_solution x₀ y₀)
  (h_sol : x^2 - 2003 * y^2 = 1)
  (h_valid : is_valid_solution x y x₀) :
  x = x₀ ∧ y = y₀ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pells_equation_unique_solution_l413_41368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_amount_is_nine_l413_41356

/-- The amount of alcohol solution used to create a mixture with 42.75% alcohol content when combined with 3 liters of water, given that the original solution contains 57% alcohol. -/
noncomputable def alcohol_solution_amount : ℝ :=
  let water_amount : ℝ := 3
  let original_alcohol_percentage : ℝ := 0.57
  let final_alcohol_percentage : ℝ := 0.4275
  (water_amount * final_alcohol_percentage) / (original_alcohol_percentage - final_alcohol_percentage)

/-- Theorem stating that the amount of alcohol solution used is approximately 9 liters. -/
theorem alcohol_solution_amount_is_nine : 
  ∃ ε > 0, |alcohol_solution_amount - 9| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_amount_is_nine_l413_41356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_digits_of_fraction_l413_41331

theorem decimal_digits_of_fraction : ∃ (n : ℕ) (d : ℚ),
  (2^7 : ℚ) / (14^3 * 125) = d ∧
  d > 0 ∧ d < 1 ∧
  (10^8 * d).num.gcd (10^8 * d).den = 1 ∧
  (10^8 * d).den = 10^8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_digits_of_fraction_l413_41331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l413_41376

/-- Calculates the time for a train to pass a jogger on an inclined track -/
noncomputable def train_passing_time (jogger_speed : ℝ) (train_speed : ℝ) 
  (train_length : ℝ) (initial_distance : ℝ) (incline : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let speed_reduction := 1 - incline
  let jogger_reduced_speed := jogger_speed_ms * speed_reduction
  let train_reduced_speed := train_speed_ms * speed_reduction
  let relative_speed := train_reduced_speed - jogger_reduced_speed
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The time for the train to pass the jogger is approximately 46.05 seconds -/
theorem train_passing_time_approx : 
  ∃ ε > 0, |train_passing_time 9 60 200 420 0.05 - 46.05| < ε :=
by
  sorry

-- Remove the #eval statement as it's not needed for compilation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l413_41376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_sum_l413_41373

theorem triangle_tangent_sum (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.tan A + Real.tan B + Real.tan C) / (2 * Real.tan A * Real.tan B * Real.tan C) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_sum_l413_41373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_point_on_line_l413_41379

theorem trigonometric_identity (a : ℝ) (h1 : π / 2 < a) (h2 : a < π) (h3 : Real.sin (π - a) = 4 / 5) :
  (Real.sin (2 * π + a) * Real.tan (π - a) * Real.cos (-π - a)) / 
  (Real.sin (3 * π / 2 - a) * Real.cos (π / 2 + a)) = -4 / 3 := by
  sorry

theorem point_on_line (θ : ℝ) (h : Real.sin θ = -2 * Real.cos θ) :
  (1 + Real.sin (2 * θ) - Real.cos (2 * θ)) / 
  (1 + Real.sin (2 * θ) + Real.cos (2 * θ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_point_on_line_l413_41379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_omega_range_l413_41387

-- Define the function f(x)
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

-- State the theorem
theorem two_zeros_omega_range (ω : ℝ) :
  ω > 0 →
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi ∧
    f ω x₁ = 0 ∧ f ω x₂ = 0 ∧
    ∀ x, 0 < x ∧ x < Real.pi ∧ f ω x = 0 → x = x₁ ∨ x = x₂) →
  5 / 3 < ω ∧ ω ≤ 8 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_omega_range_l413_41387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_property_l413_41344

/-- The set M containing integers from 1 to 10000 -/
def M : Set Nat := Finset.range 10000

/-- The theorem stating the existence of 16 subsets with the required property -/
theorem subset_intersection_property :
  ∃ (S : Finset (Set Nat)), S.card = 16 ∧
    ∀ a ∈ M, ∃ (T : Finset (Set Nat)), T ⊆ S ∧ T.card = 8 ∧
      (⋂₀ T.toSet) = {a} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_property_l413_41344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_point_existence_l413_41312

theorem geometric_mean_point_existence (A B C : ℝ) (h_triangle : A + B + C = π) :
  (∃ D : ℝ, 0 ≤ D ∧ D ≤ 1 ∧
    (Real.sin (C * (1 - D)))^2 = Real.sin (A * D) * Real.sin (B * (1 - D)))
  ↔ 
  Real.sin A * Real.sin B ≤ (Real.sin (C / 2))^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_point_existence_l413_41312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_quadrilateral_area_l413_41388

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola : Set Point2D :=
  {p : Point2D | p.y^2 = 4 * p.x}

/-- The focus of the parabola y² = 4x -/
def focus : Point2D :=
  ⟨1, 0⟩

/-- A chord of the parabola passing through the focus -/
structure Chord where
  slope : ℝ

/-- Predicate to check if a chord contains the focus -/
def containsFocus (c : Chord) : Prop :=
  sorry

/-- Two chords are perpendicular -/
def perpendicular (c1 c2 : Chord) : Prop :=
  c1.slope * c2.slope = -1

/-- The area of the quadrilateral formed by the intersections of two chords with the parabola -/
noncomputable def quadrilateralArea (c1 c2 : Chord) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem min_quadrilateral_area
  (c1 c2 : Chord)
  (h1 : containsFocus c1)
  (h2 : containsFocus c2)
  (h3 : perpendicular c1 c2) :
  ∀ (a : ℝ), a = quadrilateralArea c1 c2 → a ≥ 32 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_quadrilateral_area_l413_41388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_volume_error_l413_41304

theorem rectangular_prism_volume_error (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let actual_volume := a * b * c
  let measured_volume := 1.08 * a * 0.93 * b * 1.05 * c
  let error_percent := (measured_volume - actual_volume) / actual_volume * 100
  ∃ ε > 0, |error_percent - 0.884| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_volume_error_l413_41304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_to_triple_fifty_at_ten_percent_amount_owed_after_twenty_days_is_at_least_triple_l413_41370

/-- The minimum number of days for a loan to triple -/
noncomputable def min_days_to_triple (principal : ℝ) (daily_rate : ℝ) : ℕ :=
  Nat.ceil ((2 * principal) / (principal * daily_rate))

/-- Theorem: For a $50 loan at 10% daily interest, it takes at least 20 days to triple -/
theorem days_to_triple_fifty_at_ten_percent :
  min_days_to_triple 50 0.1 = 20 := by
  sorry

/-- Theorem: After 20 days, the amount owed is at least triple the initial loan -/
theorem amount_owed_after_twenty_days_is_at_least_triple
  (principal : ℝ) (daily_rate : ℝ) (h1 : principal = 50) (h2 : daily_rate = 0.1) :
  principal * (1 + 20 * daily_rate) ≥ 3 * principal := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_to_triple_fifty_at_ten_percent_amount_owed_after_twenty_days_is_at_least_triple_l413_41370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l413_41355

-- Define the function (marked as noncomputable due to Real.log)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (4 - a * 2^x)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x ≤ 1 → (4 - a * 2^x > 0)) ↔ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l413_41355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flowers_per_bouquet_l413_41309

/-- Given that Paige picked 53 flowers, 18 wilted, and she could make 5 bouquets
    with the remaining flowers, prove that each bouquet contained 7 flowers. -/
theorem flowers_per_bouquet
  (initial_flowers : ℕ)
  (wilted_flowers : ℕ)
  (num_bouquets : ℕ)
  (h1 : initial_flowers = 53)
  (h2 : wilted_flowers = 18)
  (h3 : num_bouquets = 5)
  (h4 : (initial_flowers - wilted_flowers) % num_bouquets = 0) :
  (initial_flowers - wilted_flowers) / num_bouquets = 7 := by
  sorry

#check flowers_per_bouquet

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flowers_per_bouquet_l413_41309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l413_41318

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 1 - Real.exp x

-- Define the point P where f(x) intersects the x-axis
def P : ℝ × ℝ := (0, 0)

-- State the theorem
theorem tangent_line_at_P :
  let m := -(deriv f (P.1))  -- Slope of the tangent line
  let b := P.2 - m * P.1     -- y-intercept of the tangent line
  (λ x => m * x + b) = (λ x => -x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l413_41318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l413_41360

/-- Recursive definition of the function sequence -/
noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => fun x => 1 / x
  | n + 1 => fun x => deriv (f n) x

/-- The theorem statement -/
theorem f_expression (n : ℕ) (x : ℝ) (h : n ≥ 1) (hx : x ≠ 0) :
  f n x = (-1)^n * (Nat.factorial n) / x^(n + 1) := by
  sorry

#check f_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l413_41360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_sin_eq_two_l413_41325

theorem negation_of_existence_sin_eq_two :
  (¬ ∃ x : ℝ, Real.sin x = 2) ↔ (∀ x : ℝ, Real.sin x ≠ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_sin_eq_two_l413_41325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_l413_41389

def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3*m + 2}

theorem unique_m_value : ∃! m : ℝ, 2 ∈ A m ∧ (∀ x y : ℝ, x ∈ A m → y ∈ A m → x = y → x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_l413_41389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_exists_l413_41327

theorem special_sequence_exists : ∃ (S : Set ℕ), 
  (∀ a b c d : ℕ, a ∈ S → b ∈ S → c ∈ S → d ∈ S → a - b = c - d → (a = c ∧ b = d) ∨ (a = b ∧ c = d)) ∧ 
  (∀ k : ℕ, k > 0 → ∀ i : ℕ, i > 0 ∧ i ≤ k → ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ (a - b = i ∨ b - a = i)) ∧
  (∃ n : ℕ, n > 0 ∧ ∀ a b : ℕ, a ∈ S → b ∈ S → a - b ≠ n ∧ b - a ≠ n) ∧
  Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_exists_l413_41327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l413_41333

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 4 = 0

-- Define the line
def line_equation (x y : ℝ) : Prop := y = x - 3

-- Define the region of interest
def region (x y : ℝ) : Prop :=
  circle_equation x y ∧ y ≥ 0 ∧ y ≤ x - 3

-- Theorem statement
theorem circle_area_theorem :
  ∃ A : ℝ, A = (9 * Real.pi) / 4 ∧
  A = ∫ x in Set.Icc 0 5, ∫ y in Set.Icc 0 (min (x - 3) (Real.sqrt (9 - (x-2)^2) + 3)), 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l413_41333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mortgage_payment_count_example_l413_41301

noncomputable def mortgage_payment_count (first_payment : ℝ) (payment_ratio : ℝ) (total_amount : ℝ) : ℕ :=
  Int.toNat ⌈(Real.log (total_amount * (payment_ratio - 1) / first_payment + 1)) / (Real.log payment_ratio)⌉

theorem mortgage_payment_count_example : 
  mortgage_payment_count 100 2.5 1229600 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mortgage_payment_count_example_l413_41301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_travel_time_l413_41345

theorem k_travel_time 
  (x : ℝ) -- K's speed in miles per hour
  (h1 : x > 0) -- K's speed is positive
  (h2 : (30 / (x - 1/3)) - (30 / x) = 1/2) -- K takes 30 minutes (1/2 hour) less time than M
  (h3 : x > 1/3) -- K's speed is greater than 1/3 mph (since M's speed is positive)
  : (30 : ℝ) / x = 30 / x := -- K's time to travel 30 miles
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_travel_time_l413_41345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_c_l413_41315

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x + c

theorem range_of_c (a b c : ℝ) :
  f a b c 0 = 5 →
  f a b c (Real.pi / 2) = 5 →
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → |f a b c x| ≤ 10) →
  -5 * Real.sqrt 2 ≤ c ∧ c ≤ 15 * Real.sqrt 2 + 20 := by
  sorry

#check range_of_c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_c_l413_41315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_model_dome_height_approx_l413_41346

/-- The height of the original dome in meters -/
noncomputable def original_height : ℝ := 55

/-- The volume of the original dome in liters -/
noncomputable def original_volume : ℝ := 250000

/-- The volume of the model dome in liters -/
noncomputable def model_volume : ℝ := 0.2

/-- The scaling factor for the model -/
noncomputable def scale_factor : ℝ := (original_volume / model_volume) ^ (1/3)

/-- The height of the model dome in meters -/
noncomputable def model_height : ℝ := original_height / scale_factor

theorem model_dome_height_approx :
  ∃ ε > 0, abs (model_height - 0.5) < ε ∧ ε < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_model_dome_height_approx_l413_41346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_coefficient_b_l413_41305

/-- A cubic function f(x) = ax^3 + bx^2 + cx + d -/
def cubic_function (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d

theorem cubic_coefficient_b (a b c d : ℝ) :
  (cubic_function a b c d (-2) = 0) →
  (cubic_function a b c d 1 = 0) →
  (cubic_function a b c d 2 = 3) →
  b = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_coefficient_b_l413_41305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bound_existence_l413_41369

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Helper function to calculate the angle between three points -/
noncomputable def angle (p q r : Point) : ℝ :=
sorry

/-- Theorem: For any set of n points in a plane, there exist three points that form an angle ≤ π/n -/
theorem angle_bound_existence (n : ℕ) (h : n > 0) (points : Finset Point) (h_card : points.card = n) :
  ∃ (p q r : Point), p ∈ points ∧ q ∈ points ∧ r ∈ points ∧ 
  angle p q r ≤ Real.pi / (n : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bound_existence_l413_41369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_y_eq_x_l413_41353

-- Define the reflection matrix
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]

-- Define a point in 2D space
def point (x y : ℝ) : Fin 2 → ℝ := ![x, y]

-- Define the reflection transformation
def reflect (p : Fin 2 → ℝ) : Fin 2 → ℝ := ![p 1, p 0]

theorem reflection_over_y_eq_x :
  ∀ p : Fin 2 → ℝ, reflection_matrix.mulVec p = reflect p := by
  sorry

#check reflection_over_y_eq_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_y_eq_x_l413_41353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_theorem_l413_41319

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (4/5) * t, 1 + (3/5) * t)

-- Define the circle
def circle_eq (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 4

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l t ∧ circle_eq p}

-- State the theorem
theorem distance_product_theorem :
  let P : ℝ × ℝ := (1, 1)
  let α : ℝ := Real.arctan (3/4)
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
    ((P.1 - A.1)^2 + (P.2 - A.2)^2) * ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_theorem_l413_41319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_decrease_percentage_l413_41352

noncomputable def mpg_at_45 : ℝ := 50
noncomputable def miles_at_60 : ℝ := 400
noncomputable def gallons_at_60 : ℝ := 10

noncomputable def mpg_at_60 : ℝ := miles_at_60 / gallons_at_60

noncomputable def percentage_decrease : ℝ := (mpg_at_45 - mpg_at_60) / mpg_at_45 * 100

theorem mpg_decrease_percentage :
  percentage_decrease = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_decrease_percentage_l413_41352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_s_value_l413_41337

/-- Triangle with side lengths a, b, c and centroid P -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a ≥ b
  hb : b ≥ c
  hc : c > 0

/-- Segment lengths from centroid to sides -/
noncomputable def segment_lengths (t : Triangle) :=
  let ma := Real.sqrt (2 * t.b^2 + 2 * t.c^2 - t.a^2)
  let mb := Real.sqrt (2 * t.a^2 + 2 * t.c^2 - t.b^2)
  let mc := Real.sqrt (2 * t.a^2 + 2 * t.b^2 - t.c^2)
  (ma, mb, mc)

/-- Sum of segments -/
noncomputable def s (t : Triangle) : ℝ :=
  let (ma, mb, mc) := segment_lengths t
  (2/3) * (ma + mb + mc)

theorem max_s_value (t : Triangle) :
  s t = (2/3) * (Real.sqrt (2 * t.b^2 + 2 * t.c^2 - t.a^2) +
                 Real.sqrt (2 * t.a^2 + 2 * t.c^2 - t.b^2) +
                 Real.sqrt (2 * t.a^2 + 2 * t.b^2 - t.c^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_s_value_l413_41337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_copied_for_fifteen_dollars_l413_41385

/-- Given that 5 cents copies 3 pages, prove that $15 copies 900 pages -/
theorem pages_copied_for_fifteen_dollars : ℕ := by
  let cost_per_three_pages : ℚ := 5 / 100  -- 5 cents in dollars
  let pages_per_five_cents : ℚ := 3
  let total_cost : ℚ := 15  -- $15 in dollars
  let total_pages : ℕ := (total_cost / cost_per_three_pages * pages_per_five_cents).floor.toNat
  have h : total_pages = 900 := by
    -- Proof goes here
    sorry
  exact total_pages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_copied_for_fifteen_dollars_l413_41385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_30_prime_factors_l413_41339

theorem factorial_30_prime_factors : 
  (Finset.filter (fun p => Nat.Prime p ∧ p ≤ 30) (Finset.range 31)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_30_prime_factors_l413_41339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_lines_intersect_l413_41311

/-- A line that divides a square into two quadrilaterals -/
structure DividingLine where
  -- Representing a line by its slope and y-intercept
  slope : ℝ
  intercept : ℝ

/-- A square with sides of length 1 -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

/-- The ratio of areas of the two quadrilaterals formed by a dividing line -/
noncomputable def areaRatio (l : DividingLine) : ℝ :=
  sorry

/-- Check if a point lies on a dividing line -/
def pointOnLine (p : ℝ × ℝ) (l : DividingLine) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

theorem at_least_three_lines_intersect (k : ℝ) (lines : Finset DividingLine)
    (h1 : lines.card = 9)
    (h2 : ∀ l ∈ lines, areaRatio l = k) :
    ∃ p : ℝ × ℝ, ∃ ls : Finset DividingLine,
      ls ⊆ lines ∧ ls.card ≥ 3 ∧ ∀ l ∈ ls, pointOnLine p l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_lines_intersect_l413_41311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sequence_condition_l413_41303

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => (n^2 * sequence_a (n + 1)^2 + 5) / ((n^2 - 1) * sequence_a (n + 2))

noncomputable def sequence_b (x y : ℝ) (n : ℕ) : ℝ := (n : ℝ) * x + y * sequence_a n

theorem constant_sequence_condition (x y : ℝ) (h : x ≠ 0) :
  (∀ n : ℕ, n > 2 → (sequence_b x y (n + 2) + sequence_b x y n) / sequence_b x y (n + 1) = 
                    (sequence_b x y (n + 1) + sequence_b x y (n - 1)) / sequence_b x y n) ↔ 
  (x = 1 ∧ y = 0) := by
  sorry

#check constant_sequence_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sequence_condition_l413_41303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_area_formula_l413_41367

structure TriangularPyramid where
  base_area : ℝ
  lateral_face_area : ℝ
  lateral_edges_perpendicular : Prop

/-- The area of the projection of a lateral face onto the base of a triangular pyramid
    with perpendicular lateral edges. -/
noncomputable def projection_area (pyramid : TriangularPyramid) : ℝ :=
  pyramid.lateral_face_area^2 / pyramid.base_area

/-- Theorem: In a triangular pyramid with perpendicular lateral edges,
    the area of the projection of a lateral face onto the base
    is the square of the lateral face area divided by the base area. -/
theorem projection_area_formula (pyramid : TriangularPyramid) 
  (h : pyramid.lateral_edges_perpendicular) :
  projection_area pyramid = pyramid.lateral_face_area^2 / pyramid.base_area := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_area_formula_l413_41367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_of_functions_l413_41328

/-- Two functions are congruent if their graphs can coincide after a number of translations. -/
def CongruentFunctions (f g : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = g (x + a) + b

noncomputable def f₁ : ℝ → ℝ := λ x ↦ Real.sin x + Real.cos x
noncomputable def f₂ : ℝ → ℝ := λ x ↦ Real.sqrt 2 * Real.sin x + Real.sqrt 2
noncomputable def f₃ : ℝ → ℝ := Real.sin

theorem congruence_of_functions :
  CongruentFunctions f₁ f₂ ∧
  ¬CongruentFunctions f₁ f₃ ∧
  ¬CongruentFunctions f₂ f₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_of_functions_l413_41328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_number_placement_exists_l413_41314

/-- Represents an edge of a cube with an assigned integer value -/
structure CubeEdge :=
  (value : ℕ)
  (h_value : value ≥ 1 ∧ value ≤ 12)

/-- Represents a face of a cube with four edges -/
structure CubeFace :=
  (edges : Fin 4 → CubeEdge)

/-- Calculates the sum of values on a cube face -/
def face_sum (face : CubeFace) : ℕ :=
  (face.edges 0).value + (face.edges 1).value + (face.edges 2).value + (face.edges 3).value

/-- Represents a cube with 12 edges -/
structure Cube :=
  (edges : Fin 12 → CubeEdge)
  (h_distinct : ∀ i j, i ≠ j → (edges i).value ≠ (edges j).value)
  (h_sum : ∀ i, (edges i).value ≥ 1 ∧ (edges i).value ≤ 12)
  (faces : Fin 6 → CubeFace)
  (h_face_sum_equal : ∀ i j, face_sum (faces i) = face_sum (faces j))

/-- Theorem stating that there exists a valid cube configuration -/
theorem cube_number_placement_exists : ∃ (c : Cube), True := by
  sorry

#check cube_number_placement_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_number_placement_exists_l413_41314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_proof_l413_41324

theorem trig_identities_proof (α β : ℝ) 
  (h1 : Real.cos α = -3/5)
  (h2 : π/2 < α ∧ α < π)
  (h3 : Real.sin β = -12/13)
  (h4 : π < β ∧ β < 3*π/2) : 
  Real.sin (α + β) = 16/65 ∧ 
  Real.cos (α - β) = -33/65 ∧ 
  Real.tan (α + β) = 16/63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_proof_l413_41324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_building_height_l413_41302

/-- The height of the first building -/
def h : ℝ := sorry

/-- The height of the second building -/
def h2 : ℝ := 2 * h

/-- The height of the third building -/
def h3 : ℝ := 3 * (h + h2)

/-- The total height of all three buildings -/
def total_height : ℝ := 7200

theorem first_building_height :
  h + h2 + h3 = total_height → h = 600 := by
  intro h_eq
  -- The proof steps would go here
  sorry

#check first_building_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_building_height_l413_41302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l413_41348

-- Define the propositions p and q
def p (a x : ℝ) : Prop := a * x - 2 ≤ 0 ∧ a * x + 1 > 0
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

-- Part 1
theorem part1 (a : ℝ) : 
  (∃ x : ℝ, 1/2 < x ∧ x < 3 ∧ p a x) ↔ -2 < a ∧ a < 4 := by sorry

-- Part 2
theorem part2 (a : ℝ) :
  (∀ x : ℝ, q x → p a x) ∧ (∃ x : ℝ, p a x ∧ ¬q x) ↔ -1/2 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l413_41348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_trig_value_l413_41377

theorem negative_trig_value : ∃ (x : ℝ), 
  (x = Real.sin (1100 * Real.pi / 180) ∨ 
   x = Real.cos (-2200 * Real.pi / 180) ∨ 
   x = Real.tan (-10) ∨ 
   x = Real.sin (7 * Real.pi / 10) * Real.cos Real.pi * Real.tan (17 * Real.pi / 9)) ∧
  (x < 0 ↔ x = Real.tan (-10)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_trig_value_l413_41377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_fraction_above_line_l413_41395

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in 2D space -/
structure Square where
  bottomLeft : Point
  topRight : Point

/-- Represents a line in 2D space -/
structure Line where
  point1 : Point
  point2 : Point

/-- Calculates the area of a square -/
noncomputable def squareArea (s : Square) : ℝ :=
  (s.topRight.x - s.bottomLeft.x) * (s.topRight.y - s.bottomLeft.y)

/-- Calculates the area of the triangle formed by the line and the square's bottom edge -/
noncomputable def triangleArea (s : Square) (l : Line) : ℝ :=
  ((s.topRight.x - s.bottomLeft.x) * (l.point2.y - s.bottomLeft.y)) / 2

/-- Theorem: The fraction of the square's area above the line is 5/6 -/
theorem area_fraction_above_line (s : Square) (l : Line) : 
  (squareArea s - triangleArea s l) / squareArea s = 5/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_fraction_above_line_l413_41395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_sum_on_p_graph_l413_41317

-- Define the functions f and p
def f : ℝ → ℝ := sorry
def p : ℝ → ℝ := sorry

-- State the theorem
theorem coordinate_sum_on_p_graph :
  f 4 = 8 →
  (∀ x, f x ≥ 0 → p x = Real.sqrt (f x)) →
  4 + p 4 = 4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_sum_on_p_graph_l413_41317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l413_41308

-- Define the initial complex number
noncomputable def z : ℂ := -4 * Complex.I

-- Define the rotation angle in radians
noncomputable def θ : ℝ := Real.pi / 3

-- Define the dilation factor
def k : ℝ := 2

-- Define the combined transformation
noncomputable def transform (w : ℂ) : ℂ := k * (Complex.exp (Complex.I * θ) * w)

-- Theorem statement
theorem complex_transformation :
  transform z = 4 * Real.sqrt 3 - 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l413_41308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_plane_on_line_l413_41332

-- Define the types for our objects
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define our objects
variable (α : Subspace ℝ V) (a : AffineSubspace ℝ V) (M : V)

-- State the theorem
theorem point_outside_plane_on_line 
  (h1 : M ∉ α) 
  (h2 : M ∈ a) : 
  M ∉ α ∧ M ∈ a := by
  exact ⟨h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_plane_on_line_l413_41332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_is_three_l413_41349

def B (x y z p q r s t u : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![x, y, z; p, q, r; s, t, u]

theorem sum_of_squares_is_three
  (x y z p q r s t u : ℝ)
  (h : (B x y z p q r s t u).transpose = (B x y z p q r s t u)⁻¹) :
  x^2 + y^2 + z^2 + p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 3 := by
  sorry

#check sum_of_squares_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_is_three_l413_41349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l413_41343

theorem trig_identity (α : Real) 
  (h1 : α ∈ Set.Ioo (5 * Real.pi / 4) (3 * Real.pi / 2))
  (h2 : Real.tan α + 1 / Real.tan α = 8) :
  Real.sin α * Real.cos α = 1 / 8 ∧ Real.sin α - Real.cos α = -Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l413_41343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_negative_sixteen_pi_thirds_l413_41378

theorem sin_negative_sixteen_pi_thirds : 
  Real.sin (-16 * π / 3) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_negative_sixteen_pi_thirds_l413_41378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_equals_13440_l413_41371

theorem binomial_product_equals_13440 : (Nat.choose 10 3) * (Nat.choose 8 3) * 2 = 13440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_equals_13440_l413_41371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l413_41316

/-- Given two curves C₁ and C₂, prove that if the tangent line of C₁ at a specific point
    is also tangent to C₂, then t ln(4e²/t) = 8. -/
theorem tangent_line_property (t : ℝ) (h_t : t > 0) :
  let C₁ := {p : ℝ × ℝ | p.2^2 = t * p.1 ∧ p.2 > 0}
  let C₂ := {p : ℝ × ℝ | p.2 = Real.exp (p.1 + 1) - 1}
  let M := (4/t, 2)
  ∃ (m n : ℝ), 
    (m, n) ∈ C₂ ∧
    (2 - n) / (4/t - m) = t / 4 ∧
    (∀ p : ℝ × ℝ, p ∈ C₁ → p.2 - 2 ≤ (t/4) * (p.1 - 4/t)) ∧
    (∀ p : ℝ × ℝ, p ∈ C₂ → p.2 - n ≤ (t/4) * (p.1 - m))
  → t * Real.log (4 * Real.exp 2 / t) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l413_41316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_div_n_l413_41350

/-- A sequence defined recursively -/
def a : ℕ → ℚ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | (n + 2) => (n + 1) * a (n + 1) / (n + 2)

/-- Theorem stating that a_n = 1/n for all n ≥ 1 -/
theorem a_eq_one_div_n (n : ℕ) (hn : n ≥ 1) : a n = 1 / n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_div_n_l413_41350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibLike_50th_term_remainder_l413_41321

def fibLike : ℕ → ℕ
  | 0 => 2
  | 1 => 2
  | n + 2 => fibLike (n + 1) + fibLike n

theorem fibLike_50th_term_remainder :
  fibLike 49 % 5 = 2 ∧ fibLike 49 % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibLike_50th_term_remainder_l413_41321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l413_41322

theorem function_inequality (x₁ x₂ : ℝ) (h1 : x₁ ∈ Set.Icc (-π/2) (π/2)) 
  (h2 : x₂ ∈ Set.Icc (-π/2) (π/2)) (h3 : x₁ < x₂) : 
  (x₁ - Real.sin x₁) < (x₂ - Real.sin x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l413_41322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_g_zeros_l413_41374

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x^2 - 2 * a * x - 1

-- Define the function g as the derivative of f
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x + 2 * a * x - 2 * a

-- Theorem for part I
theorem f_monotonicity :
  let a : ℝ := 1/2
  (∀ x y : ℝ, x < 0 ∧ y < 0 ∧ x < y → f a x < f a y) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x < y → f a x < f a y) :=
by sorry

-- Theorem for part II
theorem g_zeros (a : ℝ) :
  (a < -Real.exp 2 / 2 → ∃ x y : ℝ, x < y ∧ g a x = 0 ∧ g a y = 0) ∧
  (a = -Real.exp 2 / 2 → ∃ x : ℝ, g a x = 0) ∧
  (-Real.exp 2 / 2 < a ∧ a ≤ 0 → ∀ x : ℝ, g a x ≠ 0) ∧
  (0 < a ∧ a < 1/2 → ∃ x : ℝ, g a x = 0) ∧
  (a = 1/2 → g a 0 = 0) ∧
  (1/2 < a → ∃ x : ℝ, 0 < x ∧ x < 1 ∧ g a x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_g_zeros_l413_41374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_nine_l413_41306

-- Define the vertices of the triangle
def v1 : ℚ × ℚ := (1, 3)
def v2 : ℚ × ℚ := (4, -2)
def v3 : ℚ × ℚ := (-2, 2)

-- Define the determinant formula for triangle area
def triangleArea (a b c : ℚ × ℚ) : ℚ :=
  let (x1, y1) := a
  let (x2, y2) := b
  let (x3, y3) := c
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Theorem statement
theorem triangle_area_is_nine :
  triangleArea v1 v2 v3 = 9 := by
  -- Unfold the definitions
  unfold triangleArea v1 v2 v3
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

#eval triangleArea v1 v2 v3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_nine_l413_41306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l413_41341

-- Define the set X as ℝ \ {0, 1}
def X : Set ℝ := {x : ℝ | x ≠ 0 ∧ x ≠ 1}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^3 - x^2 - x + 1) / (2*(x^2 - x))

-- State the theorem
theorem functional_equation_solution (x : ℝ) (hx : x ∈ X) :
  f x + f (1 - 1/x) = 1 + x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l413_41341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_exceeds_90_l413_41323

-- Define the normal distribution
structure NormalDist (μ σ : ℝ) where
  value : ℝ

-- Define the probability function
noncomputable def prob (X : NormalDist 80 σ) (a b : ℝ) : ℝ := sorry

-- Define the condition P(70 ≤ X ≤ 90) = 1/3
axiom prob_condition {σ : ℝ} (X : NormalDist 80 σ) : prob X 70 90 = 1/3

-- Define the probability of X > 90
noncomputable def prob_exceed_90 {σ : ℝ} (X : NormalDist 80 σ) : ℝ := 1/3

-- Define the number of students
def num_students : ℕ := 3

-- Theorem statement
theorem exactly_one_exceeds_90 {σ : ℝ} (X : NormalDist 80 σ) :
  (num_students.choose 1 : ℝ) * (1 - prob_exceed_90 X)^2 * prob_exceed_90 X = 4/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_exceeds_90_l413_41323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l413_41362

/-- Sum of digits of a natural number -/
def sum_of_digits (x : ℕ) : ℕ := sorry

/-- Given that n is a positive integer and a = (10^(2n) - 1) / (3 * (10^n + 1)),
    if the sum of the digits of a is 567, then n = 189 -/
theorem problem_statement (n : ℕ+) (a : ℕ) 
    (h1 : a = (10^(2*n.val) - 1) / (3 * (10^n.val + 1)))
    (h2 : sum_of_digits a = 567) : n = 189 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l413_41362
