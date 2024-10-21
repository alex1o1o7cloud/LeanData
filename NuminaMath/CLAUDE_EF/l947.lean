import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_ratio_l947_94790

theorem profit_share_ratio (p_investment q_investment : ℚ) 
  (h1 : p_investment = 12000)
  (h2 : q_investment = 30000) :
  p_investment / q_investment = 2 / 5 := by
  rw [h1, h2]
  norm_num

#check profit_share_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_ratio_l947_94790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_C_in_CaCO3_approx_l947_94759

/-- The mass percentage of carbon in calcium carbonate -/
noncomputable def mass_percentage_C_in_CaCO3 : ℝ :=
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_C : ℝ := 12.01
  let molar_mass_O : ℝ := 16.00
  let molar_mass_CaCO3 : ℝ := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  (molar_mass_C / molar_mass_CaCO3) * 100

/-- Theorem stating that the mass percentage of carbon in calcium carbonate is approximately 12.00% -/
theorem mass_percentage_C_in_CaCO3_approx :
  ∃ ε > 0, |mass_percentage_C_in_CaCO3 - 12.00| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_C_in_CaCO3_approx_l947_94759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_russian_doll_discount_l947_94789

/-- The discounted price of Russian dolls -/
noncomputable def discounted_price (original_price : ℚ) (original_quantity : ℕ) (new_quantity : ℕ) : ℚ :=
  (original_price * original_quantity) / new_quantity

/-- Theorem stating the discounted price of Russian dolls -/
theorem russian_doll_discount :
  let original_price : ℚ := 4
  let original_quantity : ℕ := 15
  let new_quantity : ℕ := 20
  discounted_price original_price original_quantity new_quantity = 3 := by
  -- Unfold the definition and simplify
  unfold discounted_price
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_russian_doll_discount_l947_94789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l947_94706

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≥ 0 then (a - 5) * x - 1 else (x + a) / (x - 1)

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x > f a y) → 
  a ∈ Set.Ioc (-1 : ℝ) 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l947_94706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_cubic_equation_l947_94722

theorem solve_cubic_equation :
  ∃ y : ℝ, (y - 3)^3 = (1/27)⁻¹ ∧ y = 6 :=
by
  use 6
  constructor
  · norm_num
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_cubic_equation_l947_94722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_beta_l947_94788

theorem cosine_of_beta (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.cos α = Real.sqrt 5 / 5 →
  Real.sin (α + β) = 3/5 →
  Real.cos β = 2 * Real.sqrt 5 / 25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_beta_l947_94788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l947_94772

/-- A projection in R^2 -/
noncomputable def Projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let norm_squared := w.1 * w.1 + w.2 * w.2
  (dot_product / norm_squared * w.1, dot_product / norm_squared * w.2)

theorem projection_property :
  let proj := Projection
  let v₁ : ℝ × ℝ := (3, 3)
  let v₂ : ℝ × ℝ := (1, -1)
  let w : ℝ × ℝ := (3, 1)
  proj v₁ w = (4.5, 1.5) →
  proj v₂ w = (0.6, 0.2) := by
  sorry

#check projection_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l947_94772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_square_area_ratio_l947_94763

theorem rectangle_square_area_ratio :
  ∀ (s : ℝ), s > 0 →
  (let longer_side := 1.2 * s
   let shorter_side := 0.85 * s
   let area_R := longer_side * shorter_side
   let area_S := s * s
   area_R / area_S) = 51 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_square_area_ratio_l947_94763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l947_94783

theorem tan_double_angle (x : ℝ) (h : Real.tan x - (Real.tan x)⁻¹ = 3/2) : 
  Real.tan (2*x) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l947_94783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_equation_roots_l947_94734

theorem right_triangle_equation_roots (m : ℝ) : 
  (∃ α β : ℝ, α + β = π / 2 ∧ 
    (4 * (Real.sin α)^2 - 2*(m+1)*(Real.sin α) + m = 0) ∧ 
    (4 * (Real.sin β)^2 - 2*(m+1)*(Real.sin β) + m = 0)) → 
  m = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_equation_roots_l947_94734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplifies_to_neg_cos_f_value_at_specific_alpha_l947_94747

/-- The function f(α) as defined in the problem -/
noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

/-- Theorem stating that f(α) simplifies to -cos(α) -/
theorem f_simplifies_to_neg_cos (α : Real) : f α = -Real.cos α := by sorry

/-- Theorem stating the value of f(α) when α = -31π/3 -/
theorem f_value_at_specific_alpha : f (-31 * Real.pi / 3) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplifies_to_neg_cos_f_value_at_specific_alpha_l947_94747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_ratio_proof_l947_94795

/-- Represents the efficiency of a worker -/
structure Efficiency where
  value : ℝ
  pos : value > 0

/-- The problem statement -/
theorem efficiency_ratio_proof 
  (ram_efficiency : Efficiency) 
  (krish_efficiency : Efficiency) 
  (ram_alone_time : ℝ) 
  (together_time : ℝ) 
  (h1 : ram_efficiency.value = (1/2) * krish_efficiency.value)
  (h2 : ram_alone_time = 27)
  (h3 : together_time = 9)
  (h4 : ram_efficiency.value * ram_alone_time = 
        (ram_efficiency.value + krish_efficiency.value) * together_time) :
  ram_efficiency.value / krish_efficiency.value = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_ratio_proof_l947_94795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l947_94732

def spinner_numbers : List ℕ := [4, 6, 7, 1, 8, 9, 10, 3]

def is_prime (n : ℕ) : Bool :=
  n > 1 && (Nat.factors n).length == 1

def count_primes (l : List ℕ) : ℕ := (l.filter is_prime).length

theorem spinner_prime_probability : 
  (count_primes spinner_numbers : ℚ) / (spinner_numbers.length : ℚ) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l947_94732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_greater_than_N_l947_94757

theorem M_greater_than_N (x y : ℝ) : 
  (x^2 + y^2 + 1) > 2*(x + y - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_greater_than_N_l947_94757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_lines_l947_94710

/-- Represents a 2D point -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a 2D line -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a 3D line with first and second projections -/
structure Line3D where
  first_proj : Line2D
  second_proj : Line2D

/-- Represents a circle in 2D space -/
structure Circle2D where
  center : Point2D
  radius : ℝ

/-- Represents a direction in 2D space -/
structure Direction2D where
  angle : ℝ

/-- The problem setup -/
structure ProblemSetup where
  g : Line3D
  k : Circle2D
  dir : Direction2D
  angle : ℝ

/-- A function that counts the number of valid lines -/
noncomputable def count_valid_lines (setup : ProblemSetup) : ℕ :=
  sorry

/-- The main theorem -/
theorem exactly_two_lines (setup : ProblemSetup) 
  (h1 : setup.k.center.y = 0)  -- Circle k is in the first image plane
  (h2 : setup.angle = 30 * π / 180) : -- The angle is 30 degrees
  count_valid_lines setup = 2 := by
  sorry

#check exactly_two_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_lines_l947_94710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_work_hours_l947_94797

/-- Represents Janet's work scenario -/
structure WorkScenario where
  hours_per_week : ℝ
  current_hourly_rate : ℝ
  freelance_hourly_rate : ℝ
  freelance_weekly_fica : ℝ
  freelance_monthly_healthcare : ℝ
  weeks_per_month : ℝ
  monthly_freelance_increase : ℝ

/-- Calculates the weekly earnings difference between freelance and current job -/
noncomputable def weekly_earnings_difference (w : WorkScenario) : ℝ :=
  w.freelance_hourly_rate * w.hours_per_week - 
  w.freelance_weekly_fica - 
  w.freelance_monthly_healthcare / w.weeks_per_month - 
  w.current_hourly_rate * w.hours_per_week

/-- Janet's work hours theorem -/
theorem janet_work_hours (w : WorkScenario) 
  (h1 : w.current_hourly_rate = 30)
  (h2 : w.freelance_hourly_rate = 40)
  (h3 : w.freelance_weekly_fica = 25)
  (h4 : w.freelance_monthly_healthcare = 400)
  (h5 : w.weeks_per_month = 4)
  (h6 : w.monthly_freelance_increase = 1100)
  (h7 : weekly_earnings_difference w = w.monthly_freelance_increase / w.weeks_per_month) :
  w.hours_per_week = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_work_hours_l947_94797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_sufficient_line_l_not_necessary_l947_94752

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 5 = 0

-- Define what it means for a line to bisect the circumference of a circle
def bisects_circle (line : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), line x y ∧ circle x y ∧ 
  (∀ (x' y' : ℝ), circle x' y' → (x' - x)^2 + (y' - y)^2 = 1)

-- Statement 1: Sufficiency
theorem line_l_sufficient : 
  ∀ (x y : ℝ), line_l x y → bisects_circle line_l circle_eq :=
sorry

-- Statement 2: Not Necessary
theorem line_l_not_necessary : 
  ∃ (line : ℝ → ℝ → Prop), 
    (∃ (x y : ℝ), line x y ∧ ¬line_l x y) ∧ 
    bisects_circle line circle_eq :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_sufficient_line_l_not_necessary_l947_94752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_side_l947_94754

theorem smallest_triangle_side (t : ℤ) : 
  (7 + t > (23 : ℝ) / 2 ∧ 7 + (23 : ℝ) / 2 > t ∧ (23 : ℝ) / 2 + t > 7) → t ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_side_l947_94754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_contains_perfect_square_l947_94712

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem subset_contains_perfect_square 
  (A : Finset ℕ) 
  (h1 : A ⊆ Finset.range 170) 
  (h2 : A.card = 84) 
  (h3 : ∀ x y, x ∈ A → y ∈ A → x + y ≠ 169) : 
  ∃ n, n ∈ A ∧ is_perfect_square n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_contains_perfect_square_l947_94712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_base_mixture_ratio_l947_94707

theorem acid_base_mixture_ratio (r s t : ℝ) (hr : r > 0) (hs : s > 0) (ht : t > 0) :
  (r / (r + 1) + s / (s + 1) + t / (t + 1)) / 
  (1 / (r + 1) + 1 / (s + 1) + 1 / (t + 1)) = 
  (r * s * t + r * t + r * s + s * t) / 
  (r * s + r * t + s * t + r + s + t + 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_base_mixture_ratio_l947_94707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_difference_l947_94773

open Set MeasureTheory Interval Real

theorem integral_difference (f : ℝ → ℝ) (A B : ℝ) 
  (h1 : ∫ x in Set.Icc 0 1, f x = A)
  (h2 : ∫ x in Set.Icc 0 2, f x = B) :
  ∫ x in Set.Icc 1 2, f x = B - A := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_difference_l947_94773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_l947_94714

theorem smallest_positive_angle (α : ℝ) : 
  (∃ (x y : ℝ), x = Real.sin (5 * Real.pi / 6) ∧ y = Real.cos (5 * Real.pi / 6) ∧ 
   x = Real.sin α ∧ y = Real.cos α ∧ 0 ≤ α ∧ α < 2 * Real.pi) →
  α = 5 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_l947_94714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logo_enlargement_l947_94713

/-- Calculates the height of a proportionally enlarged rectangle -/
noncomputable def enlargedHeight (w₁ h₁ w₂ : ℝ) : ℝ :=
  (w₂ / w₁) * h₁

/-- Theorem: The height of the proportionally enlarged logo is 8 inches -/
theorem logo_enlargement (w₁ h₁ w₂ : ℝ) (hw₁ : w₁ = 3) (hh₁ : h₁ = 2) (hw₂ : w₂ = 12) :
  enlargedHeight w₁ h₁ w₂ = 8 := by
  sorry

#check logo_enlargement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logo_enlargement_l947_94713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_closer_to_side_probability_l947_94709

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the probability that a randomly selected point in the rectangle
    is closer to a side than to either diagonal -/
noncomputable def closerToSideProbability (r : Rectangle) : ℚ :=
  sorry

theorem rectangle_closer_to_side_probability :
  let r : Rectangle := { width := 6, height := 8 }
  let prob : ℚ := closerToSideProbability r
  ∃ (m n : ℕ), Nat.Coprime m n ∧ prob = m / n ∧ m + n = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_closer_to_side_probability_l947_94709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l947_94724

/-- A line passing through the point (1,3) and tangent to the circle x^2 + y^2 = 1 
    has the equation x = 1 or 4x - 3y + 5 = 0 -/
theorem tangent_line_equation (l : Set (ℝ × ℝ)) : 
  (∃ (P : ℝ × ℝ), P.1 = 1 ∧ P.2 = 3 ∧ P ∈ l) →
  (∀ (Q : ℝ × ℝ), Q ∈ l → Q.1^2 + Q.2^2 ≥ 1) →
  (∃ (R : ℝ × ℝ), R ∈ l ∧ R.1^2 + R.2^2 = 1) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x = 1 ∨ 4*x - 3*y + 5 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l947_94724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l947_94715

noncomputable def f (x : ℝ) : ℝ := (x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1) / (x^2 - 9)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -3 ∨ (-3 < x ∧ x < 3) ∨ 3 < x} := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l947_94715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_dividend_l947_94721

/-- Calculate the dividend received from an investment in shares --/
theorem calculate_dividend 
  (investment : ℝ) 
  (share_face_value : ℝ) 
  (premium_rate : ℝ) 
  (dividend_rate : ℝ) 
  (h1 : investment = 14400) 
  (h2 : share_face_value = 100) 
  (h3 : premium_rate = 0.20) 
  (h4 : dividend_rate = 0.05) : 
  (investment / (share_face_value * (1 + premium_rate))) * (share_face_value * dividend_rate) = 600 :=
by
  -- Define intermediate calculations
  let share_cost := share_face_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := share_face_value * dividend_rate

  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_dividend_l947_94721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_optimization_l947_94735

/-- Cost function for producing x units -/
noncomputable def C (x : ℝ) : ℝ := 25000 + 200 * x + (1 / 40) * x^2

/-- Average cost function -/
noncomputable def A (x : ℝ) : ℝ := C x / x

/-- Profit function when selling price is 500 yuan per unit -/
noncomputable def P (x : ℝ) : ℝ := 500 * x - C x

theorem production_optimization :
  (∀ x > 0, A 1000 ≤ A x) ∧
  (∀ x ≥ 0, P 6000 ≥ P x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_optimization_l947_94735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_nonzero_digit_of_1_over_137_l947_94765

theorem first_nonzero_digit_of_1_over_137 : 
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ d < 10 ∧ 
  (∀ (k : ℕ), k < n → (((10 : ℚ)^k * (1 : ℚ) / 137).floor = 0)) ∧
  ((10 : ℚ)^n * (1 : ℚ) / 137).floor = d := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_nonzero_digit_of_1_over_137_l947_94765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marching_band_formations_l947_94771

theorem marching_band_formations :
  let total_musicians : ℕ := 360
  let valid_formation (s t : ℕ) : Prop :=
    s * t = total_musicians ∧ 12 ≤ t ∧ t ≤ 50 ∧ s ≥ 12
  (∃! (x : ℕ), x = Finset.card (Finset.filter (fun p => valid_formation p.1 p.2) (Finset.product (Finset.range 51) (Finset.range 51)))) ∧
  (∃ (x : ℕ), x = Finset.card (Finset.filter (fun p => valid_formation p.1 p.2) (Finset.product (Finset.range 51) (Finset.range 51))) ∧ x = 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marching_band_formations_l947_94771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_l947_94793

theorem car_sale_profit (P : ℝ) (h : P > 0) :
  let discount_rate : ℝ := 0.20
  let profit_rate : ℝ := 0.19999999999999996
  let buying_price : ℝ := P * (1 - discount_rate)
  let selling_price : ℝ := P * (1 + profit_rate)
  ∃ ε > 0, abs ((selling_price / buying_price - 1) * 100 - 50) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_l947_94793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_subset_l947_94727

theorem divisibility_in_subset (S : Finset ℕ) : 
  S.card = 11 → (∀ n ∈ S, n ≤ 20) → (∀ m n : ℕ, m ∈ S → n ∈ S → m ≠ n) →
  ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_subset_l947_94727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_determines_magnitude_l947_94770

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def min_value_is_one (a b : E) : Prop :=
  ∀ t : ℝ, ‖a + t • b‖ ≥ (1 : ℝ) ∧ ∃ t₀ : ℝ, ‖a + t₀ • b‖ = (1 : ℝ)

theorem angle_determines_magnitude (a b : E) (θ : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  min_value_is_one a b →
  inner a b = ‖a‖ * ‖b‖ * Real.cos θ →
  ∃! r : ℝ, r > 0 ∧ ‖r • a‖ = r * ‖a‖ ∧ r * ‖a‖ = (1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_determines_magnitude_l947_94770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l947_94717

theorem sufficient_not_necessary_condition :
  (∀ (α : ℝ) (k : ℤ), α = π/6 + k*π → Real.cos (2*α) = 1/2) ∧
  (∃ (α : ℝ), Real.cos (2*α) = 1/2 ∧ ∀ (k : ℤ), α ≠ π/6 + k*π) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l947_94717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_l947_94700

/-- A right triangle with leg lengths 1 -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  leg_lengths_eq_one : A.1 - C.1 = 1 ∧ B.2 - C.2 = 1
  right_angle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

/-- A point on one of the sides of the triangle -/
def PointOnSide (t : RightTriangle) := 
  {P : ℝ × ℝ | P ∈ Set.range (fun x => (x * t.A.1 + (1 - x) * t.C.1, x * t.A.2 + (1 - x) * t.C.2)) ∨
               P ∈ Set.range (fun x => (x * t.B.1 + (1 - x) * t.C.1, x * t.B.2 + (1 - x) * t.C.2)) ∨
               P ∈ Set.range (fun x => (x * t.A.1 + (1 - x) * t.B.1, x * t.A.2 + (1 - x) * t.B.2))}

/-- The product of distances from a point to the vertices of the triangle -/
noncomputable def DistanceProduct (t : RightTriangle) (P : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - t.A.1)^2 + (P.2 - t.A.2)^2) *
  Real.sqrt ((P.1 - t.B.1)^2 + (P.2 - t.B.2)^2) *
  Real.sqrt ((P.1 - t.C.1)^2 + (P.2 - t.C.2)^2)

/-- The theorem stating the maximum value of PA · PB · PC -/
theorem max_distance_product (t : RightTriangle) : 
  ∀ P ∈ PointOnSide t, DistanceProduct t P ≤ Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_l947_94700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l947_94748

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the properties of a
def a_properties (a : ℝ) : Prop := a > 0 ∧ a ≠ 1

-- Define the inverse function property
def inverse_property (a : ℝ) : Prop :=
  ∃ (f_inv : ℝ → ℝ), (∀ x, f_inv (f a x) = x) ∧ f_inv (Real.sqrt 2 / 2) = 1/2

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 
  if -2 ≤ x ∧ x ≤ 2 then f a x else 0 -- Placeholder definition

-- Define the property that g equals f on [-2, 2]
def g_equals_f_on_interval (a : ℝ) : Prop :=
  ∀ x, -2 ≤ x ∧ x ≤ 2 → g a x = f a x

-- Define the evenness of g(x+2)
def g_shifted_even (a : ℝ) : Prop :=
  ∀ x, g a (x + 2) = g a (-x + 2)

-- The main theorem
theorem main_theorem (a : ℝ) 
  (h1 : a_properties a)
  (h2 : inverse_property a)
  (h3 : g_equals_f_on_interval a)
  (h4 : g_shifted_even a) :
  a = 1/2 ∧ g a (Real.sqrt 2) < g a 3 ∧ g a 3 < g a Real.pi :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l947_94748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_p1_p2_is_sqrt6_l947_94737

/-- The distance between two points in 3D space -/
noncomputable def distance3D (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- The theorem stating that the distance between P₁(2, 3, 5) and P₂(3, 1, 4) is √6 -/
theorem distance_p1_p2_is_sqrt6 :
  distance3D (2, 3, 5) (3, 1, 4) = Real.sqrt 6 := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check distance3D (2, 3, 5) (3, 1, 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_p1_p2_is_sqrt6_l947_94737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_distance_altitude_sum_l947_94716

-- Define a tetrahedron
structure Tetrahedron where
  -- We don't need to specify the exact structure, just that it exists
  mk :: 

-- Define a point inside a tetrahedron
structure PointInTetrahedron (T : Tetrahedron) where
  -- Again, we don't need to specify the exact structure
  mk ::

-- Define the distance from a point to a face of the tetrahedron
noncomputable def distance_to_face (T : Tetrahedron) (P : PointInTetrahedron T) (face : Fin 4) : ℝ := 
  sorry

-- Define the altitude of the tetrahedron corresponding to a face
noncomputable def altitude (T : Tetrahedron) (face : Fin 4) : ℝ := 
  sorry

-- The theorem to be proved
theorem tetrahedron_distance_altitude_sum (T : Tetrahedron) (P : PointInTetrahedron T) :
  (Finset.sum Finset.univ (fun i => (distance_to_face T P i) / (altitude T i))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_distance_altitude_sum_l947_94716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approx_98_percent_l947_94761

/-- Given a profit percentage, calculates the cost price as a percentage of the selling price -/
noncomputable def cost_price_percentage (profit_percentage : ℝ) : ℝ :=
  100 / (1 + profit_percentage / 100)

/-- Theorem stating that for a profit percentage of 2.0408163265306123%, 
    the cost price is approximately 98% of the selling price -/
theorem cost_price_approx_98_percent :
  let profit_percentage : ℝ := 2.0408163265306123
  abs (cost_price_percentage profit_percentage - 98) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval cost_price_percentage 2.0408163265306123

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approx_98_percent_l947_94761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_expressions_correct_l947_94778

/-- The number of ways to arrange n elements from 10 digits and 4 operators to form an arithmetic expression -/
noncomputable def arithmetic_expressions (n : ℕ) : ℝ :=
  (1 / (4 * Real.sqrt 65)) * 
  ((15 + Real.sqrt 65) * (5 + Real.sqrt 65) ^ n - 
   (15 - Real.sqrt 65) * (5 - Real.sqrt 65) ^ n)

/-- The recurrence relation for the number of arithmetic expressions -/
def arithmetic_expressions_recurrence : ℕ → ℝ
  | 0 => 0
  | 1 => 10
  | 2 => 120
  | n + 3 => 10 * arithmetic_expressions_recurrence (n + 2) + 
             40 * arithmetic_expressions_recurrence (n + 1)

/-- Theorem stating that the closed form solution satisfies the recurrence relation -/
theorem arithmetic_expressions_correct (n : ℕ) : 
  arithmetic_expressions n = arithmetic_expressions_recurrence n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_expressions_correct_l947_94778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_series_6_eq_three_eighths_l947_94745

def sum_series (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => 1 / ((i + 2) * (i + 3)))

theorem sum_series_6_eq_three_eighths :
  sum_series 6 = 3 / 8 := by
  sorry

#eval sum_series 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_series_6_eq_three_eighths_l947_94745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_average_distance_l947_94777

theorem lemming_average_distance (square_side : ℝ) (diagonal_move : ℝ) (turn_angle : ℝ) (final_move : ℝ) : 
  square_side = 15 →
  diagonal_move = 9.3 →
  turn_angle = 45 →
  final_move = 3 →
  let diagonal_length := square_side * Real.sqrt 2
  let fraction_traveled := diagonal_move / diagonal_length
  let initial_pos := (fraction_traveled * square_side, fraction_traveled * square_side)
  let final_pos := (
    initial_pos.1 + final_move * Real.cos (turn_angle * π / 180),
    initial_pos.2 + final_move * Real.sin (turn_angle * π / 180)
  )
  let distances := [
    final_pos.1,
    square_side - final_pos.1,
    final_pos.2,
    square_side - final_pos.2
  ]
  (distances.sum / distances.length) = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_average_distance_l947_94777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ratio_maximizes_volume_l947_94749

/-- The ratio that maximizes the volume of a square-based, open-top box created from a circular piece of paper. -/
noncomputable def optimal_ratio : ℝ := (1 + Real.sqrt 17) / 2

/-- The volume of the box as a function of the base side length and height. -/
def volume (a b : ℝ) : ℝ := a^2 * b

/-- The Pythagorean theorem relating the base side length, height, and diameter of the circular paper. -/
def pythagorean_relation (a b D : ℝ) : Prop := a^2 + (a + 2*b)^2 = D^2

/-- Theorem stating that the optimal ratio maximizes the volume of the box. -/
theorem optimal_ratio_maximizes_volume :
  ∀ D : ℝ, D > 0 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    pythagorean_relation a b D ∧
    (∀ a' b' : ℝ, a' > 0 → b' > 0 → pythagorean_relation a' b' D →
      volume a' b' ≤ volume a b) ∧
    a / b = optimal_ratio :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ratio_maximizes_volume_l947_94749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l947_94796

/-- The time taken for a faster train to pass a slower train -/
noncomputable def time_to_pass (faster_speed slower_speed train_length : ℝ) : ℝ :=
  (2 * train_length) / ((faster_speed - slower_speed) * (5 / 18))

/-- Theorem stating that the time taken for the faster train to pass the slower train is approximately 54 seconds -/
theorem train_passing_time :
  let faster_speed : ℝ := 46
  let slower_speed : ℝ := 36
  let train_length : ℝ := 75
  abs (time_to_pass faster_speed slower_speed train_length - 54) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l947_94796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l947_94731

open Real

theorem trigonometric_problem (α β : ℝ) 
  (h_α : α ∈ Set.Ioo 0 (π/4))
  (h_β : β ∈ Set.Ioo (3*π/4) π)
  (h_sin_diff : Real.sin (α - β) = -3/5) :
  (Real.sin β = 5/13 → Real.cos α = 63/65) ∧
  (Real.tan β / Real.tan α = -11 → α + β = 5*π/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l947_94731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_throws_for_repeated_sum_l947_94744

/-- Represents an eight-sided die -/
def EightSidedDie : Type := Fin 8

/-- The sum of rolling four eight-sided dice -/
def DiceSum : Type := Fin 29

/-- The minimum number of throws required to guarantee a repeated sum -/
def MinThrows : ℕ := 30

/-- Theorem stating the minimum number of throws required to guarantee a repeated sum -/
theorem min_throws_for_repeated_sum :
  ∀ (throws : ℕ), throws ≥ MinThrows →
    ∃ (rolls : Fin throws → EightSidedDie × EightSidedDie × EightSidedDie × EightSidedDie),
      ∃ (i j : Fin throws), i ≠ j ∧
        (rolls i).1.val + (rolls i).2.1.val + (rolls i).2.2.1.val + (rolls i).2.2.2.val =
        (rolls j).1.val + (rolls j).2.1.val + (rolls j).2.2.1.val + (rolls j).2.2.2.val :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_throws_for_repeated_sum_l947_94744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_interpolation_l947_94730

/-- Given distinct real numbers a, b, and c, this theorem states that the constructed quadratic
polynomial p(x) satisfies p(a) = a^4, p(b) = b^4, and p(c) = c^4. -/
theorem quadratic_polynomial_interpolation (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  let p : ℝ → ℝ := λ x ↦ (a^2 + b^2 + c^2 + a*b + b*c + a*c)*(x - a)*(x - b) +
                       (a^3 + a^2*b + a*b^2 + b^3)*(x - a) + a^4
  p a = a^4 ∧ p b = b^4 ∧ p c = c^4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_interpolation_l947_94730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_every_positive_integer_representable_l947_94705

/-- Represents the allowed operations on numbers -/
inductive Operation
  | Add : Operation → Operation → Operation
  | Sub : Operation → Operation → Operation
  | Mul : Operation → Operation → Operation
  | Div : Operation → Operation → Operation
  | Sqrt : Operation → Operation
  | Floor : Operation → Operation
  | Four : Operation

/-- Counts the number of 4's used in an operation -/
def countFours : Operation → Nat
  | Operation.Four => 1
  | Operation.Add a b => countFours a + countFours b
  | Operation.Sub a b => countFours a + countFours b
  | Operation.Mul a b => countFours a + countFours b
  | Operation.Div a b => countFours a + countFours b
  | Operation.Sqrt a => countFours a
  | Operation.Floor a => countFours a

/-- Evaluates an operation to a real number -/
noncomputable def evaluate : Operation → ℝ
  | Operation.Four => 4
  | Operation.Add a b => evaluate a + evaluate b
  | Operation.Sub a b => evaluate a - evaluate b
  | Operation.Mul a b => evaluate a * evaluate b
  | Operation.Div a b => evaluate a / evaluate b
  | Operation.Sqrt a => Real.sqrt (evaluate a)
  | Operation.Floor a => ⌊evaluate a⌋

/-- The main theorem stating that every positive integer can be represented -/
theorem every_positive_integer_representable :
  ∀ n : ℕ+, ∃ op : Operation, 
    countFours op ≤ 3 ∧ 
    evaluate op = n.val := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_every_positive_integer_representable_l947_94705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smoking_harms_health_l947_94741

-- Define a simple enum for correlation types
inductive Correlation
  | Positive
  | Negative
  | None
  | Uncertain

-- Define variables for smoking and health
def smoking : Prop := true
def health : Prop := true

-- Define the relationship between smoking and health
def smoking_health_relationship : Correlation := Correlation.Negative

-- Theorem stating that the relationship is negative
theorem smoking_harms_health : 
  smoking_health_relationship = Correlation.Negative := by
  -- The proof is trivial as we defined it as such
  rfl

#check smoking_harms_health

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smoking_harms_health_l947_94741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_q_p_equals_0_002_l947_94785

/-- Rounds a number to the nearest hundredth -/
def roundToHundredth (x : ℚ) : ℚ :=
  (x * 100).floor / 100 + if (x * 100 - (x * 100).floor ≥ 1/2) then 1/100 else 0

/-- Rounds a number to three decimal places -/
def roundToThreeDecimals (x : ℚ) : ℚ :=
  (x * 1000).floor / 1000 + if (x * 1000 - (x * 1000).floor ≥ 1/2) then 1/1000 else 0

/-- Calculates p as defined in the problem -/
def calculateP (a b c d : ℚ) : ℚ :=
  roundToHundredth (a + b + c + d)

/-- Calculates q as defined in the problem -/
def calculateQ (a b c d : ℚ) : ℚ :=
  roundToThreeDecimals a + roundToThreeDecimals b + roundToThreeDecimals c + roundToThreeDecimals d

theorem difference_q_p_equals_0_002 :
  let a : ℚ := 5457 / 1000
  let b : ℚ := 2951 / 1000
  let c : ℚ := 3746 / 1000
  let d : ℚ := 4398 / 1000
  calculateQ a b c d - calculateP a b c d = 1 / 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_q_p_equals_0_002_l947_94785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinates_of_P_l947_94776

/-- Converts Cartesian coordinates to polar coordinates -/
noncomputable def cartesianToPolar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 && y < 0 
           then Real.arctan (y / x) + 2 * Real.pi 
           else Real.arctan (y / x)
  (r, θ)

/-- Theorem: The polar coordinates of P(1/2, -√3/2) are (1, 5π/3) -/
theorem polar_coordinates_of_P : 
  cartesianToPolar (1/2) (-Real.sqrt 3/2) = (1, 5*Real.pi/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinates_of_P_l947_94776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_is_pi_over_four_l947_94718

theorem sum_of_angles_is_pi_over_four (a : ℝ) (α β : ℝ) 
  (h1 : a > 2)
  (h2 : α ∈ Set.Ioo (-π/2) (π/2))
  (h3 : β ∈ Set.Ioo (-π/2) (π/2))
  (h4 : (Real.tan α)^2 + 3*a*(Real.tan α) + 3*a + 1 = 0)
  (h5 : (Real.tan β)^2 + 3*a*(Real.tan β) + 3*a + 1 = 0) :
  α + β = π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_is_pi_over_four_l947_94718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_implies_a_bound_l947_94794

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a / x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a + a * x - 6 * Real.log x

theorem g_increasing_implies_a_bound (a : ℝ) :
  (∀ x > 0, Monotone (fun x => g x a)) → a ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_implies_a_bound_l947_94794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_santa_candy_distribution_l947_94792

theorem santa_candy_distribution :
  ∃! (bags : Fin 10 → ℕ), 
    (∀ i j, i ≠ j → bags i ≠ bags j) ∧
    (bags 0 = 5) ∧
    (Finset.sum (Finset.univ : Finset (Fin 10)) bags = 115) ∧
    (bags 0 + bags 1 + bags 2 = 20) ∧
    (bags 7 + bags 8 + bags 9 = 50) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_santa_candy_distribution_l947_94792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l947_94702

noncomputable def car_cost : ℝ := 34000
noncomputable def repair_cost : ℝ := 12000
noncomputable def selling_price : ℝ := 65000

noncomputable def total_cost : ℝ := car_cost + repair_cost
noncomputable def profit : ℝ := selling_price - total_cost
noncomputable def profit_percent : ℝ := (profit / total_cost) * 100

theorem profit_percentage_calculation :
  |profit_percent - 41.30| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l947_94702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_price_is_52_5_l947_94739

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The price of 2 chairs and 1 table is 60% of the price of 1 chair and 2 tables -/
axiom price_ratio : 2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

/-- The price of 1 table and 1 chair is $60 -/
axiom total_price : chair_price + table_price = 60

/-- The price of 1 table is $52.5 -/
theorem table_price_is_52_5 : table_price = 52.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_price_is_52_5_l947_94739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l947_94786

theorem no_solution_exists (p : ℕ) (hp : Nat.Prime p) :
  ¬ ∃ (a b : ℕ), (p ∣ (a * b)) ∧ (p ∣ (a + b)) ∧ (¬(p ∣ a) ∨ ¬(p ∣ b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l947_94786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_is_two_l947_94746

-- Define a point as a pair of integers
def Point := ℤ × ℤ

-- Define the circle x^2 + y^2 = 16
def onCircle (p : Point) : Prop := p.1^2 + p.2^2 = 16

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define what it means for a distance to be irrational
def isIrrationalDistance (p q : Point) : Prop :=
  ¬(∃ (n : ℕ), (distance p q)^2 = n)

-- Main theorem
theorem max_ratio_is_two :
  ∃ (A B C D : Point),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    onCircle A ∧ onCircle B ∧ onCircle C ∧ onCircle D ∧
    isIrrationalDistance A B ∧ isIrrationalDistance C D ∧
    (∀ (P Q R S : Point),
      P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S →
      onCircle P ∧ onCircle Q ∧ onCircle R ∧ onCircle S →
      isIrrationalDistance P Q ∧ isIrrationalDistance R S →
      distance P Q / distance R S ≤ distance A B / distance C D) ∧
    distance A B / distance C D = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_is_two_l947_94746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_zeros_l947_94704

def AbsoluteDifferenceSequence (a : ℕ → ℕ) : Prop :=
  (a 1 > 0) ∧ (a 2 > 0) ∧ ∀ n ≥ 3, a n = Int.natAbs (a (n - 1) - a (n - 2))

theorem infinite_zeros (a : ℕ → ℕ) (h : AbsoluteDifferenceSequence a) :
  ∀ k : ℕ, ∃ n > k, a n = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_zeros_l947_94704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_parallel_vectors_l947_94758

theorem triangle_area_with_parallel_vectors (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a = Real.sqrt 7 →
  b = 2 →
  c > 0 →
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  a^2 = b^2 + c^2 - 2*b*c * Real.cos A →
  (1/2) * b * c * Real.sin A = (3*Real.sqrt 3)/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_parallel_vectors_l947_94758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_part1_problem_part2_l947_94766

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

noncomputable def angle_to_point (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)

theorem problem_part1 (m n : ℝ) (α : ℝ) :
  unit_circle m n →
  second_quadrant m n →
  n = 12/13 →
  angle_to_point α = (m, n) →
  Real.tan α = -12/5 ∧
  (2 * Real.sin (Real.pi + α) + Real.cos α) / (Real.cos (Real.pi/2 + α) + 2 * Real.cos α) = 29/22 := by
  sorry

theorem problem_part2 (m n : ℝ) (α : ℝ) :
  unit_circle m n →
  second_quadrant m n →
  Real.sin α + Real.cos α = 1/5 →
  angle_to_point α = (m, n) →
  m = -3/5 ∧ n = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_part1_problem_part2_l947_94766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l947_94703

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - a*x + a + 1) * Real.exp x

/-- Theorem stating the range of m -/
theorem range_of_m (a : ℝ) (x₁ x₂ : ℝ) (h_a : a > 0) (h_extreme : x₁ < x₂) 
  (h_extreme_points : ∀ x, x ≠ x₁ ∧ x ≠ x₂ → (deriv (f a)) x ≠ 0)
  (h_inequality : ∀ m : ℝ, m * x₁ - (f a x₂) / Real.exp x₁ > 0) :
  ∀ m : ℝ, m ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l947_94703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_imply_sum_l947_94738

/-- A rational function with specific asymptotes -/
noncomputable def rational_function (A B C : ℤ) (x : ℝ) : ℝ :=
  x / (x^3 + A*x^2 + B*x + C)

/-- The property of having vertical asymptotes at specific points -/
def has_asymptotes (A B C : ℤ) : Prop :=
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 1 ∧ x ≠ 3 → rational_function A B C x ≠ 0) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    (abs (x + 3) < δ ∨ abs (x - 1) < δ ∨ abs (x - 3) < δ) → 
    abs (rational_function A B C x) > 1/ε)

/-- The main theorem -/
theorem asymptotes_imply_sum (A B C : ℤ) : 
  has_asymptotes A B C → A + B + C = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_imply_sum_l947_94738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_seven_l947_94740

theorem product_remainder_mod_seven (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_seven_l947_94740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l947_94799

noncomputable section

/-- The line y = 3x - 5 -/
def line (x y : ℝ) : Prop := y = 3 * x - 5

/-- Parameterization A -/
def paramA (t : ℝ) : ℝ × ℝ := (0 + t, -5 + 3 * t)

/-- Parameterization B -/
def paramB (t : ℝ) : ℝ × ℝ := (5/3 + 3 * t, 0 + t)

/-- Parameterization C -/
def paramC (t : ℝ) : ℝ × ℝ := (1 + 2 * t, -2 + 6 * t)

/-- Parameterization D -/
def paramD (t : ℝ) : ℝ × ℝ := (-5/3 - t, 0 - 3 * t)

/-- Parameterization E -/
def paramE (t : ℝ) : ℝ × ℝ := (-5 + t/3, -20 + t)

theorem line_parameterization :
  (∀ t, line (paramA t).1 (paramA t).2) ∧
  (∀ t, line (paramC t).1 (paramC t).2) ∧
  (∀ t, line (paramD t).1 (paramD t).2) ∧
  ¬(∀ t, line (paramB t).1 (paramB t).2) ∧
  ¬(∀ t, line (paramE t).1 (paramE t).2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l947_94799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l947_94725

noncomputable def e : ℝ := Real.exp 1

noncomputable def a : ℝ := 1 / e - Real.log (1 / e)
noncomputable def b : ℝ := 1 / (2 * e) - Real.log (1 / (2 * e))
noncomputable def c : ℝ := 2 / e - Real.log (2 / e)

theorem abc_inequality : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l947_94725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_formula_not_valid_l947_94756

noncomputable def my_sequence (n : ℕ) : ℝ :=
  if n % 2 = 0 then 1 else 0

noncomputable def my_formula (n : ℕ) : ℝ :=
  (1/2) * (1 + (-1)^n)

theorem formula_not_valid : ¬ (∀ n, my_sequence n = my_formula n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_formula_not_valid_l947_94756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l947_94787

/-- Calculates the average speed given two trip segments -/
noncomputable def average_speed (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) : ℝ :=
  (distance1 + distance2) / (distance1 / speed1 + distance2 / speed2)

theorem average_speed_calculation :
  let d1 : ℝ := 50  -- distance of first segment in miles
  let s1 : ℝ := 20  -- speed of first segment in miles per hour
  let d2 : ℝ := 25  -- distance of second segment in miles
  let s2 : ℝ := 25  -- speed of second segment in miles per hour
  abs (average_speed d1 s1 d2 s2 - 21.4286) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l947_94787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_plane_intersection_volume_l947_94711

-- Define the cube
def Cube : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 2}

-- Define the plane
def Plane : Set (Fin 3 → ℝ) :=
  {p | -3 * p 0 + 2 * p 1 - 4 * p 2 = 0}

-- Define the intersection of the cube and the plane
def Intersection := Cube ∩ Plane

-- Define the volume function
noncomputable def volume : Set (Fin 3 → ℝ) → ℝ := sorry

-- Theorem statement
theorem cube_plane_intersection_volume :
  volume Intersection = 1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_plane_intersection_volume_l947_94711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_in_chromic_acid_l947_94750

/-- Molar mass of Hydrogen in g/mol -/
noncomputable def molar_mass_H : ℝ := 1.01

/-- Molar mass of Chromium in g/mol -/
noncomputable def molar_mass_Cr : ℝ := 51.996

/-- Molar mass of Oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- Molar mass of Chromic acid (H2CrO4) in g/mol -/
noncomputable def molar_mass_H2CrO4 : ℝ := 2 * molar_mass_H + molar_mass_Cr + 4 * molar_mass_O

/-- Mass of Hydrogen in Chromic acid (H2CrO4) in g/mol -/
noncomputable def mass_H_in_H2CrO4 : ℝ := 2 * molar_mass_H

/-- Mass percentage of Hydrogen in Chromic acid (H2CrO4) -/
noncomputable def mass_percentage_H : ℝ := (mass_H_in_H2CrO4 / molar_mass_H2CrO4) * 100

theorem mass_percentage_H_in_chromic_acid :
  |mass_percentage_H - 1.712| < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_in_chromic_acid_l947_94750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_two_digit_prime_saturated_l947_94798

def isPrimeSaturated (n : ℕ) : Prop :=
  (Nat.factors n).prod < Real.sqrt (n : ℝ)

def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem greatest_two_digit_prime_saturated :
  ∀ n : ℕ, isTwoDigit n → isPrimeSaturated n → n ≤ 98 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_two_digit_prime_saturated_l947_94798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l947_94767

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides

-- Define the properties of the triangle
def IsValidTriangle (t : Triangle) : Prop :=
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c

-- Define the conditions given in the problem
def ProblemConditions (t : Triangle) : Prop :=
  IsValidTriangle t ∧
  Real.pi/2 < t.C ∧  -- C is obtuse
  t.c - t.b = 2 * t.b * Real.cos t.A

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : ProblemConditions t) : 
  t.a^2 = t.b * (t.b + t.c) ∧ 
  t.A = 2 * t.B ∧ 
  0 < Real.sin t.B ∧ Real.sin t.B < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l947_94767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_count_relation_l947_94719

theorem root_count_relation (a : ℝ) :
  (∃ S : Finset ℝ, (∀ x ∈ S, 4^x - 4^(-x) = 2 * Real.cos (a * x)) ∧ S.card = 2007) →
  ∃ T : Finset ℝ, (∀ x ∈ T, 4^x + 4^(-x) = 2 * Real.cos (a * x) + 4) ∧ T.card = 4014 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_count_relation_l947_94719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_is_infinite_l947_94753

-- Define the set P
def P : Set Nat := {p | ∃ k : Nat, k > 0 ∧ Nat.Prime p ∧ p ∣ (k^3 + 6)}

-- State the theorem
theorem P_is_infinite : Set.Infinite P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_is_infinite_l947_94753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l947_94764

noncomputable def area_triangle (a b c : ℝ) : ℝ := 
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_theorem (a b c A B C : ℝ) (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A)
  (h2 : a = Real.sqrt 7) (h3 : c = 2) :
  A = π / 3 ∧ area_triangle a b c = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l947_94764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_three_digit_number_l947_94774

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  (∃ a b c : ℕ, a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ n = 100 * a + 10 * b + c ∧ n = 5 * a * b * c) ∧
  n = 175 := by
  sorry

#check unique_three_digit_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_three_digit_number_l947_94774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_represents_two_circles_l947_94760

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 - ρ * (2 + Real.sin θ) + 2 * Real.sin θ = 0

-- Define what it means for an equation to represent two circles
def represents_two_circles (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
    r₁ > 0 ∧ r₂ > 0 ∧
    (∀ x y, eq x y ↔ 
      ((x - c₁.1)^2 + (y - c₁.2)^2 = r₁^2) ∨
      ((x - c₂.1)^2 + (y - c₂.2)^2 = r₂^2))

-- Theorem statement
theorem polar_equation_represents_two_circles :
  represents_two_circles (fun x y ↦ ∃ ρ θ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ polar_equation ρ θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_represents_two_circles_l947_94760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l947_94723

-- Define the hyperbola
def hyperbola : ℝ × ℝ → Prop := λ p => p.1^2 - p.2^2/4 = 1

-- Define the asymptote
def is_asymptote (f : ℝ → ℝ) (h : ℝ × ℝ → Prop) : Prop :=
  ∀ ε > 0, ∃ M > 0, ∀ x, |x| > M → 
    ∃ y, h (x, y) ∧ |y - f x| < ε * |x|

-- Theorem statement
theorem hyperbola_asymptote :
  is_asymptote (λ x => 2*x) hyperbola ∨ 
  is_asymptote (λ x => -2*x) hyperbola :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l947_94723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_ellipse_tangency_point_enclosed_area_exists_l947_94751

-- Define positive real numbers a and b
variable (a b : ℝ) (ha : a > 0) (hb : b > 0)

-- Define the circle C₁: (x-a)² + y² = a²
def C₁ (x y : ℝ) : Prop := (x - a)^2 + y^2 = a^2

-- Define the ellipse C₂: x² + y²/b² = 1
def C₂ (x y : ℝ) : Prop := x^2 + y^2/b^2 = 1

-- Define the condition for C₁ to be inscribed in C₂
def inscribed_condition (a b : ℝ) : Prop := a^2 = b^2 * (1 + b^2)

-- Define the point of tangency (p, q) in the first quadrant
noncomputable def p (a b : ℝ) : ℝ := a / (1 + b^2)
noncomputable def q (a b : ℝ) : ℝ := b * Real.sqrt (1 - (a / (1 + b^2))^2)

-- Theorem: If C₁ is inscribed in C₂, then the inscribed_condition holds
theorem inscribed_circle_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, C₁ a x y → C₂ b x y) → inscribed_condition a b :=
sorry

-- Theorem: If b = 1/√3 and C₁ is inscribed in C₂, then (p, q) = (1/2, 1/2)
theorem tangency_point (a : ℝ) (ha : a > 0) (h : b = 1 / Real.sqrt 3) :
  inscribed_condition a (1 / Real.sqrt 3) → p a (1 / Real.sqrt 3) = 1/2 ∧ q a (1 / Real.sqrt 3) = 1/2 :=
sorry

-- Define the area enclosed by C₁ and C₂ for x ≥ p
noncomputable def enclosed_area (a b : ℝ) : ℝ :=
  ∫ x in p a b..(2*a), (b * Real.sqrt (1 - x^2/a^2) - Real.sqrt (a^2 - (x-a)^2))

-- Theorem: The enclosed area exists and is finite
theorem enclosed_area_exists (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  inscribed_condition a b → ∃ A : ℝ, enclosed_area a b = A :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_ellipse_tangency_point_enclosed_area_exists_l947_94751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l947_94720

noncomputable def f (x : ℝ) := (Real.sin x + Real.cos x) * Real.sin x

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x, f x = m) ∧ (m = 1/2 - Real.sqrt 2/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l947_94720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_max_area_l947_94726

/-- Represents a quadrilateral with side lengths a, b, c, and d -/
structure Quadrilateral (a b c d : ℝ) where
  convex : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
  triangle_inequality : a + b + c > d ∧ b + c + d > a ∧ c + d + a > b ∧ d + a + b > c

/-- The area of a quadrilateral -/
noncomputable def area (q : Quadrilateral a b c d) : ℝ := sorry

/-- Indicates if a quadrilateral is cyclic (can be inscribed in a circle) -/
def is_cyclic (q : Quadrilateral a b c d) : Prop := sorry

/-- The theorem stating that cyclic quadrilaterals have the maximum area -/
theorem cyclic_max_area {a b c d : ℝ} (q : Quadrilateral a b c d) :
  ∃ (q' : Quadrilateral a b c d), is_cyclic q' ∧ area q ≤ area q' := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_max_area_l947_94726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l947_94742

/-- Represents a hyperbola in the real plane -/
structure Hyperbola where
  /-- Equation of the hyperbola -/
  equation : ℝ → ℝ → Prop

/-- Checks if a point is on the hyperbola -/
def IsOn (h : Hyperbola) (x y : ℝ) : Prop :=
  h.equation x y

/-- Checks if a line is an asymptote of the hyperbola -/
def IsAsymptote (h : Hyperbola) (f : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → ∃ y, IsOn h x y ∧ |y - f x| < ε

/-- Length of the real axis of a hyperbola -/
noncomputable def RealAxisLength (h : Hyperbola) : ℝ :=
  sorry -- Definition would go here

theorem hyperbola_real_axis_length :
  ∀ (h : Hyperbola),
    (∀ (x y : ℝ), (y = 2*x ∨ y = -2*x) → IsAsymptote h (fun z ↦ if y = 2*x then 2*z else -2*z)) →
    (∃ (x y : ℝ), IsOn h x y ∧ x + y - 3 = 0 ∧ 2*x - y + 6 = 0) →
    RealAxisLength h = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l947_94742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_2002_eq_1_l947_94762

noncomputable section

variable (f : ℝ → ℝ)

axiom f_1 : f 1 = 1
axiom f_5 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_1_le : ∀ x : ℝ, f (x + 1) ≤ f x + 1

def g (x : ℝ) : ℝ := f x + 1 - x

theorem g_2002_eq_1 : g f 2002 = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_2002_eq_1_l947_94762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_equals_one_l947_94784

theorem sum_of_reciprocals_equals_one (a b : ℝ) (h1 : (2 : ℝ)^a = 10) (h2 : (5 : ℝ)^b = 10) : 
  1/a + 1/b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_equals_one_l947_94784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_sums_l947_94708

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

/-- There is no positive integer n for which the sums of two specific arithmetic sequences are equal -/
theorem no_equal_sums : ∀ n : ℕ+, 
  arithmetic_sum 5 3 n.val ≠ arithmetic_sum 20 3 n.val := by
  intro n
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_sums_l947_94708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l947_94781

noncomputable section

-- Define the ellipse E
def ellipse_equation (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the distance between the right focus and the line x/a + y/b = 1
def focus_line_distance (a b : ℝ) : ℝ := 
  |b * Real.sqrt (a^2 - b^2) - a * b| / Real.sqrt (a^2 + b^2)

-- Define the theorem
theorem ellipse_properties (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = 1/2) 
  (h4 : focus_line_distance a b = Real.sqrt 21 / 7) :
  (∀ x y, ellipse_equation a b x y ↔ ellipse_equation 2 (Real.sqrt 3) x y) ∧
  (∀ A B : ℝ × ℝ, 
    (ellipse_equation a b A.1 A.2 ∧ 
     ellipse_equation a b B.1 B.2 ∧ 
     A.1 * B.1 + A.2 * B.2 = 0) →
    (let k := (B.2 - A.2) / (B.1 - A.1);
     let m := A.2 - k * A.1;
     |m| / Real.sqrt (k^2 + 1) = 2 * Real.sqrt 21 / 7)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l947_94781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_angle_theorem_l947_94775

/-- A parallelogram with vertices E, F, G, and H -/
structure Parallelogram (E F G H : EuclideanSpace ℝ (Fin 2)) : Prop where
  is_parallelogram : (F - E) = (H - G) ∧ (G - F) = (H - E)

/-- The measure of an angle in degrees -/
noncomputable def angle_measure (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry

/-- Theorem: In a parallelogram EFGH, if the measure of angle EFG is twice
    the measure of angle FGH, then the measure of angle EHG is 120° -/
theorem parallelogram_angle_theorem
  (E F G H : EuclideanSpace ℝ (Fin 2))
  (para : Parallelogram E F G H)
  (angle_relation : angle_measure E F G = 2 * angle_measure F G H) :
  angle_measure E H G = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_angle_theorem_l947_94775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stationery_box_sheets_l947_94779

/-- The number of sheets of paper in each box of stationery -/
def sheets_per_box : ℕ := sorry

/-- The number of envelopes in each box of stationery -/
def envelopes_per_box : ℕ := sorry

/-- The number of sheets John had left after using all envelopes -/
def john_sheets_left : ℕ := 60

/-- The number of envelopes Mary had left after using all sheets -/
def mary_envelopes_left : ℕ := 60

/-- The number of sheets Mary used per letter -/
def mary_sheets_per_letter : ℕ := 5

theorem stationery_box_sheets :
  (sheets_per_box - envelopes_per_box = john_sheets_left) →
  (mary_sheets_per_letter * envelopes_per_box = sheets_per_box) →
  sheets_per_box = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stationery_box_sheets_l947_94779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_increasing_l947_94736

-- Define the interval (0,1)
def openInterval : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := 3 - x
noncomputable def h (x : ℝ) : ℝ := 1 / x
def k (x : ℝ) : ℝ := -x^2 + 4

-- State the theorem
theorem only_f_increasing :
  (∀ x y, x ∈ openInterval → y ∈ openInterval → x < y → f x < f y) ∧
  (∃ x y, x ∈ openInterval ∧ y ∈ openInterval ∧ x < y ∧ g x ≥ g y) ∧
  (∃ x y, x ∈ openInterval ∧ y ∈ openInterval ∧ x < y ∧ h x ≥ h y) ∧
  (∃ x y, x ∈ openInterval ∧ y ∈ openInterval ∧ x < y ∧ k x ≥ k y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_increasing_l947_94736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_chord_implies_trisected_chord_l947_94728

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ
  radius_positive : r > 0

/-- A chord of a circle -/
structure Chord (c : Circle) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  on_circle : dist A c.O = c.r ∧ dist B c.O = c.r

/-- The length of a chord -/
def chord_length (c : Circle) (chord : Chord c) : ℝ := dist chord.A chord.B

theorem equal_chord_implies_trisected_chord 
  (c : Circle) 
  (AB AC : Chord c) 
  (h : chord_length c AB = chord_length c AC) :
  ∃ (FG : Chord c) (F₁ G₁ : ℝ × ℝ),
    F₁ ∈ Set.Icc FG.A FG.B ∧
    G₁ ∈ Set.Icc FG.A FG.B ∧
    F₁ ∈ Set.Icc AB.A AB.B ∧
    G₁ ∈ Set.Icc AC.A AC.B ∧
    dist FG.A F₁ = dist F₁ G₁ ∧
    dist F₁ G₁ = dist G₁ FG.B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_chord_implies_trisected_chord_l947_94728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_leq_1_1_l947_94733

noncomputable def numbers : List ℚ := [14/10, 9/10, 12/10, 1/2, 13/10]

theorem sum_of_numbers_leq_1_1 : 
  (numbers.filter (λ x => (x : ℝ) ≤ 1.1)).sum = 14/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_leq_1_1_l947_94733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_sum_nonnegative_l947_94791

theorem inequality_implies_sum_nonnegative (x y : ℝ) :
  (2 : ℝ)^x - (3 : ℝ)^(-x) ≥ (2 : ℝ)^(-y) - (3 : ℝ)^y → x + y ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_sum_nonnegative_l947_94791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_modulo_thirteen_l947_94701

theorem sum_remainder_modulo_thirteen (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_modulo_thirteen_l947_94701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_satisfy_equation_l947_94743

noncomputable def a : ℝ × ℝ × ℝ := (2, 2, 2)
noncomputable def b : ℝ × ℝ × ℝ := (1, -4, 1)
noncomputable def c : ℝ × ℝ × ℝ := (-2, 1, 3)

noncomputable def p : ℝ := -1/2
noncomputable def q : ℝ := 26/9
noncomputable def r : ℝ := -1/7

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  let (v1, v2, v3) := v
  let (w1, w2, w3) := w
  v1 * w1 + v2 * w2 + v3 * w3

def scale_vector (s : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (v1, v2, v3) := v
  (s * v1, s * v2, s * v3)

def add_vectors (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (v1, v2, v3) := v
  let (w1, w2, w3) := w
  (v1 + w1, v2 + w2, v3 + w3)

theorem vectors_satisfy_equation :
  (dot_product a b = 0) →
  (dot_product a c = 0) →
  (dot_product b c = 0) →
  add_vectors (scale_vector p a) (add_vectors (scale_vector q b) (scale_vector r c)) = (3, -11, 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_satisfy_equation_l947_94743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l947_94755

def M : Set ℤ := {x | -4 < x ∧ x < 2}
def N : Set ℤ := {x : ℤ | x^2 < 4}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l947_94755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l947_94769

theorem remainder_problem (s t u v : ℕ) 
  (h_order : s > t ∧ t > u ∧ u > v)
  (h_s : s % 23 = 6)
  (h_t : t % 23 = 9)
  (h_u : u % 23 = 13)
  (h_v : v % 23 = 17) :
  (2 * (s - t) - 3 * (t - u) + 4 * (u - v)) % 23 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l947_94769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_given_sum_l947_94729

theorem cos_difference_given_sum (α β : ℝ) 
  (h1 : Real.cos α + Real.cos β = 1/2) 
  (h2 : Real.sin α + Real.sin β = 1/3) : 
  Real.cos (α - β) = -59/72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_given_sum_l947_94729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_of_month_l947_94782

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Returns the day of the week that is n days after the given day -/
def daysAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => daysAfter (nextDay d) n

theorem first_day_of_month (d : DayOfWeek) :
  daysAfter d 17 = DayOfWeek.Wednesday → d = DayOfWeek.Sunday :=
by
  sorry

#check first_day_of_month

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_of_month_l947_94782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equals_minus_two_plus_two_i_l947_94780

noncomputable def z : ℂ := 4 / (-1 - Complex.I)

theorem z_equals_minus_two_plus_two_i : z = -2 + 2 * Complex.I := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equals_minus_two_plus_two_i_l947_94780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_approx_l947_94768

/-- The molar mass of Barium in g/mol -/
noncomputable def molar_mass_Ba : ℝ := 137.327

/-- The molar mass of Oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 15.999

/-- The molar mass of Hydrogen in g/mol -/
noncomputable def molar_mass_H : ℝ := 1.008

/-- The number of Barium atoms in Ba(OH)₂ -/
def num_Ba : ℕ := 1

/-- The number of Oxygen atoms in Ba(OH)₂ -/
def num_O : ℕ := 2

/-- The number of Hydrogen atoms in Ba(OH)₂ -/
def num_H : ℕ := 2

/-- The molar mass of Ba(OH)₂ in g/mol -/
noncomputable def molar_mass_BaOH2 : ℝ := 
  num_Ba * molar_mass_Ba + num_O * molar_mass_O + num_H * molar_mass_H

/-- The mass percentage of Hydrogen in Ba(OH)₂ -/
noncomputable def mass_percentage_H : ℝ := 
  (num_H * molar_mass_H / molar_mass_BaOH2) * 100

theorem mass_percentage_H_approx : 
  |mass_percentage_H - 1.176| < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_approx_l947_94768
