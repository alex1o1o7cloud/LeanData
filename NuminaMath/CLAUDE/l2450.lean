import Mathlib

namespace NUMINAMATH_CALUDE_p_percentage_of_x_l2450_245073

theorem p_percentage_of_x (x y z w t u p : ℝ) 
  (h1 : 0.37 * z = 0.84 * y)
  (h2 : y = 0.62 * x)
  (h3 : 0.47 * w = 0.73 * z)
  (h4 : w = t - u)
  (h5 : u = 0.25 * t)
  (h6 : p = z + t + u) :
  p = 5.05675 * x := by sorry

end NUMINAMATH_CALUDE_p_percentage_of_x_l2450_245073


namespace NUMINAMATH_CALUDE_gardens_area_difference_l2450_245017

/-- Represents a rectangular garden with length and width -/
structure Garden where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Calculates the usable area of a garden with a path around the perimeter -/
def Garden.usableArea (g : Garden) (pathWidth : ℝ) : ℝ :=
  (g.length - 2 * pathWidth) * (g.width - 2 * pathWidth)

theorem gardens_area_difference : 
  let karlGarden : Garden := { length := 22, width := 50 }
  let makennaGarden : Garden := { length := 30, width := 46 }
  let pathWidth : ℝ := 1
  makennaGarden.usableArea pathWidth - karlGarden.area = 132 := by sorry

end NUMINAMATH_CALUDE_gardens_area_difference_l2450_245017


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l2450_245023

theorem sqrt_sum_squares_eq_sum (a b : ℝ) :
  Real.sqrt (a^2 + b^2) = a + b ↔ a * b = 0 ∧ a + b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l2450_245023


namespace NUMINAMATH_CALUDE_waiter_tips_fraction_l2450_245088

theorem waiter_tips_fraction (base_salary : ℚ) : 
  let tips := (5 / 4) * base_salary
  let total_income := base_salary + tips
  let expenses := (1 / 8) * base_salary
  let taxes := (1 / 5) * total_income
  let after_tax_income := total_income - taxes
  (tips / after_tax_income) = 25 / 36 :=
by sorry

end NUMINAMATH_CALUDE_waiter_tips_fraction_l2450_245088


namespace NUMINAMATH_CALUDE_cubic_term_of_line_l2450_245027

-- Define the line equation
def line_equation (x : ℝ) : ℝ := x^2 - x^3

-- State the theorem
theorem cubic_term_of_line : 
  ∃ (a b c d : ℝ), 
    (∀ x, line_equation x = a*x^3 + b*x^2 + c*x + d) ∧ 
    (a = -1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_term_of_line_l2450_245027


namespace NUMINAMATH_CALUDE_smallest_square_area_for_two_rectangles_l2450_245044

/-- The smallest square area that can contain two non-overlapping rectangles -/
theorem smallest_square_area_for_two_rectangles :
  ∀ (w₁ h₁ w₂ h₂ : ℕ),
    w₁ = 2 ∧ h₁ = 4 ∧ w₂ = 3 ∧ h₂ = 5 →
    ∃ (s : ℕ),
      s^2 = 81 ∧
      ∀ (a : ℕ),
        (a ≥ w₁ ∧ a ≥ h₁ ∧ a ≥ w₂ ∧ a ≥ h₂ ∧ a ≥ w₁ + w₂ ∧ a ≥ h₁ + h₂) →
        a^2 ≥ s^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_area_for_two_rectangles_l2450_245044


namespace NUMINAMATH_CALUDE_calculation_one_l2450_245071

theorem calculation_one :
  (27 : ℝ) ^ (1/3) + (1/9).sqrt / (-2/3) + |(-(1/2))| = 3 := by sorry

end NUMINAMATH_CALUDE_calculation_one_l2450_245071


namespace NUMINAMATH_CALUDE_pet_shop_grooming_l2450_245052

/-- The pet shop grooming problem -/
theorem pet_shop_grooming (poodle_time terrier_time total_time : ℕ) 
  (terrier_count : ℕ) (poodle_count : ℕ) : 
  poodle_time = 30 →
  terrier_time = poodle_time / 2 →
  terrier_count = 8 →
  total_time = 210 →
  poodle_count * poodle_time + terrier_count * terrier_time = total_time →
  poodle_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_grooming_l2450_245052


namespace NUMINAMATH_CALUDE_circle_M_properties_l2450_245079

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 4

-- Define the line that contains the center of the circle
def center_line (x y : ℝ) : Prop :=
  x + y = 2

-- Define the points C and D
def point_C : ℝ × ℝ := (1, -1)
def point_D : ℝ × ℝ := (-1, 1)

-- Theorem statement
theorem circle_M_properties :
  (∀ x y, circle_M x y → center_line x y) ∧
  circle_M point_C.1 point_C.2 ∧
  circle_M point_D.1 point_D.2 ∧
  (∀ x y, circle_M x y → 2 - 2 * Real.sqrt 2 ≤ x + y ∧ x + y ≤ 2 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_circle_M_properties_l2450_245079


namespace NUMINAMATH_CALUDE_dog_weight_ratio_l2450_245095

theorem dog_weight_ratio (chihuahua pitbull great_dane : ℝ) : 
  chihuahua + pitbull + great_dane = 439 →
  great_dane = 307 →
  great_dane = 3 * pitbull + 10 →
  pitbull / chihuahua = 3 := by
  sorry

end NUMINAMATH_CALUDE_dog_weight_ratio_l2450_245095


namespace NUMINAMATH_CALUDE_chord_equation_l2450_245056

/-- Given a hyperbola and a point that bisects a chord, find the equation of the line containing the chord. -/
theorem chord_equation (x y : ℝ → ℝ) (t : ℝ) :
  (∀ t, (x t)^2 - 4*(y t)^2 = 4) →  -- Curve equation
  (∃ t₁ t₂, x ((t₁ + t₂)/2) = 3 ∧ y ((t₁ + t₂)/2) = -1) →  -- Point A(3, -1) bisects the chord
  (∃ a b c : ℝ, ∀ t, a*(x t) + b*(y t) + c = 0 ∧ a = 3 ∧ b = 4 ∧ c = -5) -- Line equation
  := by sorry

end NUMINAMATH_CALUDE_chord_equation_l2450_245056


namespace NUMINAMATH_CALUDE_standard_equation_of_C_l2450_245031

-- Define the ellipse C
def C : Set (ℝ × ℝ) := sorry

-- Define that C passes through the point (2,3)
def C_passes_through : (2, 3) ∈ C := sorry

-- Define that (2,0) is the right focus of C
def right_focus : (2, 0) ∈ C := sorry

-- Theorem: The standard equation of C is x²/16 + y²/12 = 1
theorem standard_equation_of_C : 
  ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2/16 + y^2/12 = 1 := by sorry

end NUMINAMATH_CALUDE_standard_equation_of_C_l2450_245031


namespace NUMINAMATH_CALUDE_zeros_and_range_of_f_l2450_245020

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + b - 1

theorem zeros_and_range_of_f (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f 1 (-2) x = 0 ↔ x = 3 ∨ x = -1) ∧
  (∀ b : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) ↔ 0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_zeros_and_range_of_f_l2450_245020


namespace NUMINAMATH_CALUDE_inequality_solution_l2450_245041

open Real

theorem inequality_solution (x : ℝ) : 
  (2 * x + 3) / (x + 5) > (5 * x + 7) / (3 * x + 14) ↔ 
  (x > -103.86 ∧ x < -14/3) ∨ (x > -5 ∧ x < -0.14) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2450_245041


namespace NUMINAMATH_CALUDE_power_function_properties_l2450_245078

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (2 * m^2 - 2 * m - 3) * x^2

-- State the theorem
theorem power_function_properties (m : ℝ) :
  (∀ x y, 0 < x ∧ x < y → f m y < f m x) →  -- f is monotonically decreasing on (0, +∞)
  (f m 8 = Real.sqrt 2 / 4) ∧               -- f(8) = √2/4
  (∀ x, f m (x^2 + 2*x) < f m (x + 6) ↔ x ∈ Set.Ioo (-6) (-3) ∪ Set.Ioi 2) :=
by sorry


end NUMINAMATH_CALUDE_power_function_properties_l2450_245078


namespace NUMINAMATH_CALUDE_average_difference_l2450_245059

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = (20 + 60 + x) / 3 + 5 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l2450_245059


namespace NUMINAMATH_CALUDE_rectangular_field_distance_l2450_245064

/-- The distance run around a rectangular field -/
def distance_run (length width : ℕ) (laps : ℕ) : ℕ :=
  2 * (length + width) * laps

/-- Theorem: Running 3 laps around a 75m by 15m rectangular field results in a total distance of 540m -/
theorem rectangular_field_distance :
  distance_run 75 15 3 = 540 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_distance_l2450_245064


namespace NUMINAMATH_CALUDE_lizzys_final_money_l2450_245057

/-- Calculates Lizzy's final amount of money in cents -/
def lizzys_money (mother_gave uncle_gave father_gave candy_cost toy_cost change_dollars change_cents : ℕ) : ℕ :=
  let initial := mother_gave + father_gave
  let after_candy := initial - candy_cost
  let after_uncle := after_candy + uncle_gave
  let after_toy := after_uncle - toy_cost
  let final := after_toy + change_dollars * 100 + change_cents
  final

/-- Proves that Lizzy's final amount of money is 160 cents -/
theorem lizzys_final_money :
  lizzys_money 80 70 40 50 90 1 10 = 160 := by
  sorry

end NUMINAMATH_CALUDE_lizzys_final_money_l2450_245057


namespace NUMINAMATH_CALUDE_problem_polygon_area_l2450_245043

/-- A point in a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- A polygon defined by a list of grid points --/
def Polygon := List GridPoint

/-- The polygon described in the problem --/
def problemPolygon : Polygon := [
  ⟨0, 0⟩, ⟨10, 0⟩, ⟨20, 0⟩, ⟨30, 10⟩, ⟨20, 30⟩,
  ⟨10, 30⟩, ⟨0, 30⟩, ⟨0, 20⟩, ⟨10, 20⟩, ⟨10, 10⟩
]

/-- Calculate the area of a polygon given its vertices --/
def calculatePolygonArea (p : Polygon) : Int :=
  sorry

theorem problem_polygon_area :
  calculatePolygonArea problemPolygon = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_polygon_area_l2450_245043


namespace NUMINAMATH_CALUDE_gym_signup_fee_l2450_245062

theorem gym_signup_fee 
  (cheap_monthly : ℕ)
  (expensive_monthly : ℕ)
  (expensive_signup : ℕ)
  (total_cost : ℕ)
  (h1 : cheap_monthly = 10)
  (h2 : expensive_monthly = 3 * cheap_monthly)
  (h3 : expensive_signup = 4 * expensive_monthly)
  (h4 : total_cost = 650)
  (h5 : total_cost = 12 * cheap_monthly + 12 * expensive_monthly + expensive_signup + cheap_signup) :
  cheap_signup = 50 := by
  sorry

end NUMINAMATH_CALUDE_gym_signup_fee_l2450_245062


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2450_245054

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the solution set
def solution_set : Set ℝ := {x | -1 < x ∧ x < 3}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2450_245054


namespace NUMINAMATH_CALUDE_point_on_inverse_graph_and_coordinate_sum_l2450_245026

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- Theorem statement
theorem point_on_inverse_graph_and_coordinate_sum 
  (h : f 3 = 5/3) : 
  (f_inv (5/3) = 3) ∧ 
  ((1/3) * (f_inv (5/3)) = 1) ∧ 
  (5/3 + 1 = 8/3) := by
sorry

end NUMINAMATH_CALUDE_point_on_inverse_graph_and_coordinate_sum_l2450_245026


namespace NUMINAMATH_CALUDE_triangle_angle_B_l2450_245075

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_B (t : Triangle) : 
  t.a = 2 * Real.sqrt 3 → 
  t.b = 2 → 
  t.A = π / 3 → 
  t.B = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_B_l2450_245075


namespace NUMINAMATH_CALUDE_percentage_of_330_l2450_245077

theorem percentage_of_330 : (33 + 1 / 3 : ℚ) / 100 * 330 = 110 := by sorry

end NUMINAMATH_CALUDE_percentage_of_330_l2450_245077


namespace NUMINAMATH_CALUDE_janes_numbers_l2450_245018

def is_between (n : ℕ) (a b : ℕ) : Prop := a ≤ n ∧ n ≤ b

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def satisfies_conditions (n : ℕ) : Prop :=
  is_between n 100 150 ∧
  n % 7 = 0 ∧
  n % 3 ≠ 0 ∧
  sum_of_digits n % 4 = 0

theorem janes_numbers : 
  {n : ℕ | satisfies_conditions n} = {112, 147} := by sorry

end NUMINAMATH_CALUDE_janes_numbers_l2450_245018


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2450_245068

theorem quadratic_equation_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ 
   (a^2 - 1) * x^2 - 2*(5*a + 1)*x + 24 = 0 ∧
   (a^2 - 1) * y^2 - 2*(5*a + 1)*y + 24 = 0) → 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2450_245068


namespace NUMINAMATH_CALUDE_locus_equation_rectangle_perimeter_bound_l2450_245000

-- Define the locus W
def W : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.2| = Real.sqrt (p.1^2 + (p.2 - 1/2)^2)}

-- Define a rectangle with three vertices on W
structure RectangleOnW where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  d : ℝ × ℝ
  h_a_on_w : a ∈ W
  h_b_on_w : b ∈ W
  h_c_on_w : c ∈ W
  h_is_rectangle : (a.1 - b.1) * (c.1 - b.1) + (a.2 - b.2) * (c.2 - b.2) = 0 ∧
                   (a.1 - d.1) * (c.1 - d.1) + (a.2 - d.2) * (c.2 - d.2) = 0

-- Theorem statements
theorem locus_equation (p : ℝ × ℝ) :
  p ∈ W ↔ p.2 = p.1^2 + 1/4 := by sorry

theorem rectangle_perimeter_bound (rect : RectangleOnW) :
  let perimeter := 2 * (Real.sqrt ((rect.a.1 - rect.b.1)^2 + (rect.a.2 - rect.b.2)^2) +
                        Real.sqrt ((rect.b.1 - rect.c.1)^2 + (rect.b.2 - rect.c.2)^2))
  perimeter > 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_locus_equation_rectangle_perimeter_bound_l2450_245000


namespace NUMINAMATH_CALUDE_parabola_vertex_m_value_l2450_245040

theorem parabola_vertex_m_value (m : ℝ) :
  let f (x : ℝ) := 3 * x^2 + 6 * Real.sqrt m * x + 36
  let vertex_y := f (-(Real.sqrt m) / 3)
  vertex_y = 33 → m = 1 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_m_value_l2450_245040


namespace NUMINAMATH_CALUDE_admission_score_theorem_l2450_245096

-- Define the parameters of the normal distribution
def μ : ℝ := 500
def σ : ℝ := 100

-- Define the admission rate
def admission_rate : ℝ := 0.4

-- Define the standard normal cumulative distribution function
noncomputable def Φ : ℝ → ℝ := sorry

-- State the theorem
theorem admission_score_theorem :
  ∃ (z : ℝ), Φ z = 1 - admission_rate ∧ μ + σ * z = 525 :=
sorry

end NUMINAMATH_CALUDE_admission_score_theorem_l2450_245096


namespace NUMINAMATH_CALUDE_no_prime_with_perfect_square_131_base_l2450_245080

theorem no_prime_with_perfect_square_131_base : ¬∃ n : ℕ, 
  (5 ≤ n ∧ n ≤ 15) ∧ 
  Nat.Prime n ∧ 
  ∃ m : ℕ, n^2 + 3*n + 1 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_with_perfect_square_131_base_l2450_245080


namespace NUMINAMATH_CALUDE_appropriate_presentation_lengths_l2450_245022

/-- Represents the duration of a presentation in minutes -/
def PresentationDuration : Set ℝ := { x | 20 ≤ x ∧ x ≤ 40 }

/-- The ideal speech rate in words per minute -/
def SpeechRate : ℝ := 120

/-- Calculates the range of appropriate word counts for a presentation -/
def AppropriateWordCount : Set ℕ :=
  { w | ∃ (d : ℝ), d ∈ PresentationDuration ∧ 
    (↑w : ℝ) ≥ 20 * SpeechRate ∧ (↑w : ℝ) ≤ 40 * SpeechRate }

/-- Theorem stating that 2700, 3900, and 4500 words are appropriate presentation lengths -/
theorem appropriate_presentation_lengths :
  2700 ∈ AppropriateWordCount ∧
  3900 ∈ AppropriateWordCount ∧
  4500 ∈ AppropriateWordCount :=
by sorry

end NUMINAMATH_CALUDE_appropriate_presentation_lengths_l2450_245022


namespace NUMINAMATH_CALUDE_parabola_properties_l2450_245097

-- Define the parabola
def parabola (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem parabola_properties :
  ∃ (a b : ℝ),
    (parabola a b (-2) = 0) ∧
    (parabola a b (-1) = 3) ∧
    (a = 6 ∧ b = 8) ∧
    (let vertex_x := -a / 2
     let vertex_y := parabola a b vertex_x
     vertex_x = -3 ∧ vertex_y = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2450_245097


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l2450_245055

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (plane_perp : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a : Line) (α β : Plane) 
  (h : subset a α) : 
  (∀ (b : Line), subset b α → (perp b β → plane_perp α β)) ∧ 
  (∃ (c : Line), subset c α ∧ plane_perp α β ∧ ¬perp c β) :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l2450_245055


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2450_245038

def snake_toy_price : ℚ := 1176 / 100
def cage_price : ℚ := 1454 / 100
def heat_lamp_price : ℚ := 625 / 100
def cage_discount_rate : ℚ := 10 / 100
def sales_tax_rate : ℚ := 8 / 100
def found_money : ℚ := 1

def total_cost : ℚ :=
  let discounted_cage_price := cage_price * (1 - cage_discount_rate)
  let subtotal := snake_toy_price + discounted_cage_price + heat_lamp_price
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  total_with_tax - found_money

theorem total_cost_calculation :
  (total_cost * 100).floor / 100 = 3258 / 100 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2450_245038


namespace NUMINAMATH_CALUDE_inequality_proof_l2450_245001

theorem inequality_proof (s x y z : ℝ) 
  (hs : s > 0) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (h : s * x > z * y) : 
  ¬ (
    (x > z ∧ -x > -z ∧ s > z / x ∧ s < y / x) ∨
    (x > z ∧ -x > -z ∧ s > z / x) ∨
    (x > z ∧ -x > -z ∧ s < y / x) ∨
    (x > z ∧ s > z / x ∧ s < y / x) ∨
    (-x > -z ∧ s > z / x ∧ s < y / x) ∨
    (x > z ∧ -x > -z) ∨
    (x > z ∧ s > z / x) ∨
    (x > z ∧ s < y / x) ∨
    (-x > -z ∧ s > z / x) ∨
    (-x > -z ∧ s < y / x) ∨
    (s > z / x ∧ s < y / x) ∨
    (x > z) ∨
    (-x > -z) ∨
    (s > z / x) ∨
    (s < y / x)
  ) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2450_245001


namespace NUMINAMATH_CALUDE_consecutive_numbers_divisibility_l2450_245053

theorem consecutive_numbers_divisibility (k : ℕ) :
  let r₁ := k % 2022
  let r₂ := (k + 1) % 2022
  let r₃ := (k + 2) % 2022
  Prime (r₁ + r₂ + r₃) →
  (k % 2022 = 0) ∨ ((k + 1) % 2022 = 0) ∨ ((k + 2) % 2022 = 0) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_divisibility_l2450_245053


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l2450_245067

theorem gcd_of_powers_of_two : Nat.gcd (2^2025 - 1) (2^2016 - 1) = 2^9 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l2450_245067


namespace NUMINAMATH_CALUDE_base8_54321_equals_22737_l2450_245047

-- Define a function to convert a base-8 number to base-10
def base8ToBase10 (n : List Nat) : Nat :=
  List.foldl (fun acc d => 8 * acc + d) 0 n

-- Define the base-8 number 54321
def base8Number : List Nat := [5, 4, 3, 2, 1]

-- State the theorem
theorem base8_54321_equals_22737 :
  base8ToBase10 base8Number = 22737 := by
  sorry

end NUMINAMATH_CALUDE_base8_54321_equals_22737_l2450_245047


namespace NUMINAMATH_CALUDE_no_inscribed_circle_l2450_245069

/-- A pentagon is represented by a list of its side lengths -/
def Pentagon := List ℝ

/-- Check if a list represents a valid pentagon with sides 1, 2, 5, 6, 7 -/
def isValidPentagon (p : Pentagon) : Prop :=
  p.length = 5 ∧ p.toFinset = {1, 2, 5, 6, 7}

/-- Sum of three elements in a list -/
def sumThree (l : List ℝ) (i j k : ℕ) : ℝ :=
  (l.get? i).getD 0 + (l.get? j).getD 0 + (l.get? k).getD 0

/-- Check if the sum of two non-adjacent sides is greater than or equal to
    the sum of the remaining three sides -/
def hasInvalidPair (p : Pentagon) : Prop :=
  (p.get? 0).getD 0 + (p.get? 2).getD 0 ≥ sumThree p 1 3 4 ∨
  (p.get? 0).getD 0 + (p.get? 3).getD 0 ≥ sumThree p 1 2 4 ∨
  (p.get? 1).getD 0 + (p.get? 3).getD 0 ≥ sumThree p 0 2 4 ∨
  (p.get? 1).getD 0 + (p.get? 4).getD 0 ≥ sumThree p 0 2 3 ∨
  (p.get? 2).getD 0 + (p.get? 4).getD 0 ≥ sumThree p 0 1 3

theorem no_inscribed_circle (p : Pentagon) (h : isValidPentagon p) :
  hasInvalidPair p := by
  sorry


end NUMINAMATH_CALUDE_no_inscribed_circle_l2450_245069


namespace NUMINAMATH_CALUDE_cricket_players_count_l2450_245060

/-- The number of cricket players in a games hour -/
def cricket_players (total_players hockey_players football_players softball_players : ℕ) : ℕ :=
  total_players - (hockey_players + football_players + softball_players)

/-- Theorem: There are 16 cricket players given the conditions -/
theorem cricket_players_count : cricket_players 59 12 18 13 = 16 := by
  sorry

end NUMINAMATH_CALUDE_cricket_players_count_l2450_245060


namespace NUMINAMATH_CALUDE_total_potatoes_l2450_245045

def nancys_potatoes : ℕ := 6
def sandys_potatoes : ℕ := 7

theorem total_potatoes : nancys_potatoes + sandys_potatoes = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_potatoes_l2450_245045


namespace NUMINAMATH_CALUDE_max_value_at_13_l2450_245048

-- Define the function f(x) = x - 5
def f (x : ℝ) : ℝ := x - 5

-- Theorem statement
theorem max_value_at_13 :
  ∃ (x : ℝ), x ≤ 13 ∧ ∀ (y : ℝ), y ≤ 13 → f y ≤ f x ∧ f x = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_at_13_l2450_245048


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l2450_245076

/-- Calculates the total charge for a taxi trip -/
def taxiCharge (initialFee : ℚ) (additionalChargePerIncrement : ℚ) (incrementDistance : ℚ) (tripDistance : ℚ) : ℚ :=
  initialFee + (tripDistance / incrementDistance) * additionalChargePerIncrement

theorem taxi_charge_calculation :
  let initialFee : ℚ := 235/100
  let additionalChargePerIncrement : ℚ := 35/100
  let incrementDistance : ℚ := 2/5
  let tripDistance : ℚ := 36/10
  taxiCharge initialFee additionalChargePerIncrement incrementDistance tripDistance = 550/100 := by
  sorry

#eval taxiCharge (235/100) (35/100) (2/5) (36/10)

end NUMINAMATH_CALUDE_taxi_charge_calculation_l2450_245076


namespace NUMINAMATH_CALUDE_log_inequality_solution_l2450_245006

theorem log_inequality_solution (x : ℝ) : 
  (4 * (Real.log (Real.cos (2 * x)) / Real.log 16) + 
   2 * (Real.log (Real.sin x) / Real.log 4) + 
   Real.log (Real.cos x) / Real.log 2 + 3 < 0) ↔ 
  (0 < x ∧ x < Real.pi / 24) ∨ (5 * Real.pi / 24 < x ∧ x < Real.pi / 4) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_solution_l2450_245006


namespace NUMINAMATH_CALUDE_alyssa_toy_cost_l2450_245028

/-- Calculates the total cost of toys with various discounts and special offers -/
def total_cost (football_price marbles_price puzzle_price toy_car_price board_game_price 
                stuffed_animal_price action_figure_price : ℝ) : ℝ :=
  let marbles_discounted := marbles_price * (1 - 0.05)
  let puzzle_discounted := puzzle_price * (1 - 0.10)
  let toy_car_discounted := toy_car_price * (1 - 0.15)
  let stuffed_animals_total := stuffed_animal_price * 1.5
  let action_figures_total := action_figure_price * (1 + 0.4)
  football_price + marbles_discounted + puzzle_discounted + toy_car_discounted + 
  board_game_price + stuffed_animals_total + action_figures_total

/-- Theorem stating the total cost of Alyssa's toys -/
theorem alyssa_toy_cost : 
  total_cost 5.71 6.59 4.25 3.95 10.49 8.99 12.39 = 60.468 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_toy_cost_l2450_245028


namespace NUMINAMATH_CALUDE_sticker_distribution_l2450_245030

/-- Given the ratio of stickers and Kate's sticker count, prove the equal distribution result -/
theorem sticker_distribution 
  (kate_stickers : ℝ) 
  (ratio_kate : ℝ) 
  (ratio_jenna : ℝ) 
  (ratio_ava : ℝ) 
  (h1 : kate_stickers = 45) 
  (h2 : ratio_kate = 7.5) 
  (h3 : ratio_jenna = 4.25) 
  (h4 : ratio_ava = 5.75) : 
  (kate_stickers + (kate_stickers * ratio_jenna / ratio_kate) + (kate_stickers * ratio_ava / ratio_kate)) / 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2450_245030


namespace NUMINAMATH_CALUDE_position_of_81st_number_l2450_245021

/-- Represents the triangular number pattern where each row has one more number than the previous row. -/
def TriangularPattern : Nat → Nat → Nat
  | row, pos => if pos ≤ row then (row * (row - 1)) / 2 + pos else 0

/-- The position of a number in the triangular pattern. -/
structure Position where
  row : Nat
  pos : Nat

/-- Finds the position of the nth number in the triangular pattern. -/
def findPosition (n : Nat) : Position :=
  let row := (Nat.sqrt (8 * n + 1) - 1) / 2 + 1
  let pos := n - (row * (row - 1)) / 2
  ⟨row, pos⟩

theorem position_of_81st_number :
  findPosition 81 = ⟨13, 3⟩ := by sorry

end NUMINAMATH_CALUDE_position_of_81st_number_l2450_245021


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_sqrt2_plus_minus_one_l2450_245011

theorem arithmetic_mean_of_sqrt2_plus_minus_one :
  (((Real.sqrt 2) + 1) + ((Real.sqrt 2) - 1)) / 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_sqrt2_plus_minus_one_l2450_245011


namespace NUMINAMATH_CALUDE_used_car_percentage_l2450_245036

theorem used_car_percentage (used_price original_price : ℝ) 
  (h1 : used_price = 15000)
  (h2 : original_price = 37500) :
  (used_price / original_price) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_used_car_percentage_l2450_245036


namespace NUMINAMATH_CALUDE_a_value_l2450_245099

def P (a : ℝ) : Set ℝ := {1, 2, a}
def Q : Set ℝ := {x | x^2 - 9 = 0}

theorem a_value (a : ℝ) : P a ∩ Q = {3} → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l2450_245099


namespace NUMINAMATH_CALUDE_golden_ratio_comparison_l2450_245015

theorem golden_ratio_comparison : (Real.sqrt 5 - 1) / 2 > 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_comparison_l2450_245015


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l2450_245070

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof that the bridge length is 215 meters -/
theorem bridge_length_proof :
  bridge_length 160 45 30 = 215 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l2450_245070


namespace NUMINAMATH_CALUDE_original_price_calculation_l2450_245003

/-- Given an article sold for $35 with a 75% gain, prove that the original price was $20. -/
theorem original_price_calculation (sale_price : ℝ) (gain_percent : ℝ) 
  (h1 : sale_price = 35)
  (h2 : gain_percent = 75) :
  ∃ (original_price : ℝ), 
    sale_price = original_price * (1 + gain_percent / 100) ∧ 
    original_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2450_245003


namespace NUMINAMATH_CALUDE_dave_has_more_cats_l2450_245090

/-- The number of pets owned by Teddy, Ben, and Dave -/
structure PetOwnership where
  teddy_dogs : ℕ
  teddy_cats : ℕ
  ben_dogs : ℕ
  dave_dogs : ℕ
  dave_cats : ℕ

/-- The conditions of the pet ownership problem -/
def pet_problem (p : PetOwnership) : Prop :=
  p.teddy_dogs = 7 ∧
  p.teddy_cats = 8 ∧
  p.ben_dogs = p.teddy_dogs + 9 ∧
  p.dave_dogs = p.teddy_dogs - 5 ∧
  p.teddy_dogs + p.teddy_cats + p.ben_dogs + p.dave_dogs + p.dave_cats = 54

/-- The theorem stating that Dave has 13 more cats than Teddy -/
theorem dave_has_more_cats (p : PetOwnership) (h : pet_problem p) :
  p.dave_cats = p.teddy_cats + 13 := by
  sorry

end NUMINAMATH_CALUDE_dave_has_more_cats_l2450_245090


namespace NUMINAMATH_CALUDE_damaged_books_count_damaged_books_proof_l2450_245087

theorem damaged_books_count : ℕ → ℕ → Prop :=
  fun obsolete damaged =>
    (obsolete = 6 * damaged - 8) →
    (obsolete + damaged = 69) →
    (damaged = 11)

-- The proof is omitted
theorem damaged_books_proof : damaged_books_count 58 11 := by sorry

end NUMINAMATH_CALUDE_damaged_books_count_damaged_books_proof_l2450_245087


namespace NUMINAMATH_CALUDE_car_wash_price_l2450_245085

theorem car_wash_price (truck_price suv_price total_raised : ℕ) 
  (num_suvs num_trucks num_cars : ℕ) :
  truck_price = 6 →
  suv_price = 7 →
  num_suvs = 5 →
  num_trucks = 5 →
  num_cars = 7 →
  total_raised = 100 →
  ∃ (car_price : ℕ), 
    car_price = 5 ∧ 
    total_raised = num_suvs * suv_price + num_trucks * truck_price + num_cars * car_price :=
by sorry

end NUMINAMATH_CALUDE_car_wash_price_l2450_245085


namespace NUMINAMATH_CALUDE_f_minimum_value_l2450_245029

open Real

noncomputable def f (x : ℝ) : ℝ := (3 * sin x - 4 * cos x - 10) * (3 * sin x + 4 * cos x - 10)

theorem f_minimum_value :
  ∃ (min : ℝ), (∀ (x : ℝ), f x ≥ min) ∧ (min = 25 / 9 - 10 - 80 * Real.sqrt 2 / 3 - 116) := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_value_l2450_245029


namespace NUMINAMATH_CALUDE_expected_twos_is_one_third_l2450_245002

/-- Represents a standard six-sided die -/
def StandardDie := Fin 6

/-- The probability of rolling a 2 on a standard die -/
def prob_two : ℚ := 1 / 6

/-- The probability of not rolling a 2 on a standard die -/
def prob_not_two : ℚ := 5 / 6

/-- The expected number of 2's when rolling two standard dice -/
def expected_twos : ℚ := 1 / 3

/-- Theorem: The expected number of 2's when rolling two standard dice is 1/3 -/
theorem expected_twos_is_one_third :
  expected_twos = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_twos_is_one_third_l2450_245002


namespace NUMINAMATH_CALUDE_mean_of_combined_sets_l2450_245074

theorem mean_of_combined_sets (set1_count : Nat) (set1_mean : ℚ) (set2_count : Nat) (set2_mean : ℚ) 
  (h1 : set1_count = 7)
  (h2 : set1_mean = 15)
  (h3 : set2_count = 8)
  (h4 : set2_mean = 18) :
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  total_sum / total_count = 249 / 15 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_combined_sets_l2450_245074


namespace NUMINAMATH_CALUDE_total_students_l2450_245058

/-- The number of students who went to the movie -/
def M : ℕ := 10

/-- The number of students who went to the picnic -/
def P : ℕ := 20

/-- The number of students who played games -/
def G : ℕ := 5

/-- The number of students who went to both the movie and the picnic -/
def MP : ℕ := 4

/-- The number of students who went to both the movie and games -/
def MG : ℕ := 2

/-- The number of students who went to both the picnic and games -/
def PG : ℕ := 0

/-- The number of students who participated in all three activities -/
def MPG : ℕ := 2

/-- The total number of students -/
def T : ℕ := M + P + G - MP - MG - PG + MPG

theorem total_students : T = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l2450_245058


namespace NUMINAMATH_CALUDE_find_starting_number_l2450_245066

theorem find_starting_number :
  ∀ n : ℤ,
  (300 : ℝ) = (n + 200 : ℝ) / 2 + 150 →
  n = 100 :=
by sorry

end NUMINAMATH_CALUDE_find_starting_number_l2450_245066


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2450_245016

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ - 3 = 0 → 
  x₂^2 - 4*x₂ - 3 = 0 → 
  x₁ + x₂ = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2450_245016


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l2450_245063

theorem initial_markup_percentage (C : ℝ) (h : C > 0) : 
  ∃ M : ℝ, 
    M ≥ 0 ∧ 
    C * (1 + M) * 1.25 * 0.93 = C * (1 + 0.395) ∧ 
    M = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l2450_245063


namespace NUMINAMATH_CALUDE_card_length_is_three_inches_l2450_245039

-- Define the poster board size in inches
def posterBoardSize : ℕ := 12

-- Define the width of the cards in inches
def cardWidth : ℕ := 2

-- Define the maximum number of cards that can be made
def maxCards : ℕ := 24

-- Theorem statement
theorem card_length_is_three_inches :
  ∀ (cardLength : ℕ),
    (posterBoardSize / cardWidth) * (posterBoardSize / cardLength) = maxCards →
    cardLength = 3 := by
  sorry

end NUMINAMATH_CALUDE_card_length_is_three_inches_l2450_245039


namespace NUMINAMATH_CALUDE_solution_in_first_and_second_quadrants_l2450_245012

-- Define the inequalities
def inequality1 (x y : ℝ) : Prop := y > 3 * x
def inequality2 (x y : ℝ) : Prop := y > 6 - 2 * x

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem solution_in_first_and_second_quadrants :
  ∀ x y : ℝ, inequality1 x y ∧ inequality2 x y →
  first_quadrant x y ∨ second_quadrant x y :=
sorry

end NUMINAMATH_CALUDE_solution_in_first_and_second_quadrants_l2450_245012


namespace NUMINAMATH_CALUDE_probability_at_least_two_special_items_l2450_245086

theorem probability_at_least_two_special_items (total : Nat) (special : Nat) (select : Nat) 
  (h1 : total = 8) (h2 : special = 3) (h3 : select = 4) : 
  (Nat.choose special 2 * Nat.choose (total - special) (select - 2) + 
   Nat.choose special 3 * Nat.choose (total - special) (select - 3)) / 
  Nat.choose total select = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_special_items_l2450_245086


namespace NUMINAMATH_CALUDE_stuarts_initial_marbles_l2450_245014

/-- Stuart's initial marble count problem -/
theorem stuarts_initial_marbles (betty_marbles : ℕ) (stuart_final : ℕ) 
  (h1 : betty_marbles = 60)
  (h2 : stuart_final = 80) :
  ∃ (stuart_initial : ℕ), 
    stuart_initial + (betty_marbles * 2/5 : ℕ) = stuart_final ∧ 
    stuart_initial = 56 := by
  sorry

end NUMINAMATH_CALUDE_stuarts_initial_marbles_l2450_245014


namespace NUMINAMATH_CALUDE_binary_ones_condition_theorem_l2450_245010

/-- The number of 1's in the binary representation of a natural number -/
def binary_ones (n : ℕ) : ℕ := sorry

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, binary_ones (f x + y) = binary_ones (f y + x)

/-- The main theorem -/
theorem binary_ones_condition_theorem (f : ℕ → ℕ) :
  satisfies_condition f → ∃ c : ℕ, ∀ x : ℕ, f x = x + c := by sorry

end NUMINAMATH_CALUDE_binary_ones_condition_theorem_l2450_245010


namespace NUMINAMATH_CALUDE_a_10_ends_with_1000_nines_l2450_245019

def a : ℕ → ℕ
  | 0 => 9
  | (n + 1) => 3 * (a n)^4 + 4 * (a n)^3

def ends_with_nines (n : ℕ) (k : ℕ) : Prop :=
  ∃ m : ℕ, n = m * 10^k + (10^k - 1)

theorem a_10_ends_with_1000_nines : ends_with_nines (a 10) 1000 := by
  sorry

end NUMINAMATH_CALUDE_a_10_ends_with_1000_nines_l2450_245019


namespace NUMINAMATH_CALUDE_set_operations_l2450_245005

def A : Set ℝ := {x | x < 0 ∨ x ≥ 2}
def B : Set ℝ := {x | -1 < x ∧ x < 1}

theorem set_operations :
  (A ∪ B = {x | x ≥ 2 ∨ x < 1}) ∧
  ((Aᶜ ∩ B) = {x | 0 ≤ x ∧ x < 1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2450_245005


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l2450_245098

theorem x_fourth_plus_inverse_x_fourth (x : ℝ) (h : x + 1/x = 5) : x^4 + 1/x^4 = 527 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l2450_245098


namespace NUMINAMATH_CALUDE_three_additional_trams_needed_l2450_245084

/-- The number of trams needed to reduce intervals by one-fifth -/
def additional_trams (initial_trams : ℕ) : ℕ :=
  let total_distance := 60
  let initial_interval := total_distance / initial_trams
  let new_interval := initial_interval * 4 / 5
  let new_total_trams := total_distance / new_interval
  new_total_trams - initial_trams

/-- Theorem stating that 3 additional trams are needed -/
theorem three_additional_trams_needed :
  additional_trams 12 = 3 := by
  sorry

#eval additional_trams 12

end NUMINAMATH_CALUDE_three_additional_trams_needed_l2450_245084


namespace NUMINAMATH_CALUDE_gene_mutation_not_valid_for_AaB_l2450_245065

/-- Represents a genotype --/
inductive Genotype
  | AaB
  | AABb

/-- Represents possible reasons for lacking a gene --/
inductive Reason
  | GeneMutation
  | ChromosomalVariation
  | ChromosomalStructuralVariation
  | MaleIndividual

/-- Determines if a reason is valid for explaining the lack of a gene --/
def is_valid_reason (g : Genotype) (r : Reason) : Prop :=
  match g, r with
  | Genotype.AaB, Reason.GeneMutation => False
  | _, _ => True

/-- Theorem stating that gene mutation is not a valid reason for individual A's genotype --/
theorem gene_mutation_not_valid_for_AaB :
  ¬(is_valid_reason Genotype.AaB Reason.GeneMutation) :=
by
  sorry


end NUMINAMATH_CALUDE_gene_mutation_not_valid_for_AaB_l2450_245065


namespace NUMINAMATH_CALUDE_sum_of_C_and_D_l2450_245046

/-- Represents a 4x4 table with numbers 1 to 4 -/
def Table := Fin 4 → Fin 4 → Fin 4

/-- Checks if a row contains all numbers from 1 to 4 -/
def validRow (t : Table) (row : Fin 4) : Prop :=
  ∀ n : Fin 4, ∃ col : Fin 4, t row col = n

/-- Checks if a column contains all numbers from 1 to 4 -/
def validColumn (t : Table) (col : Fin 4) : Prop :=
  ∀ n : Fin 4, ∃ row : Fin 4, t row col = n

/-- Checks if the table satisfies all given constraints -/
def validTable (t : Table) : Prop :=
  (∀ row : Fin 4, validRow t row) ∧
  (∀ col : Fin 4, validColumn t col) ∧
  t 0 0 = 1 ∧
  t 1 1 = 2 ∧
  t 3 3 = 4

theorem sum_of_C_and_D (t : Table) (h : validTable t) :
  t 1 2 + t 2 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_C_and_D_l2450_245046


namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l2450_245094

def is_valid_student_count (n : ℕ) : Prop :=
  n > 0 ∧ 120 % (2 * n) = 0

theorem candy_distribution_theorem :
  ∀ n : ℕ, is_valid_student_count n ↔ n ∈ ({5, 6, 10, 12, 15} : Finset ℕ) :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l2450_245094


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2450_245032

theorem mod_equivalence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -2023 [ZMOD 9] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2450_245032


namespace NUMINAMATH_CALUDE_union_A_B_when_m_2_intersection_A_B_empty_iff_l2450_245035

-- Define sets A and B
def A : Set ℝ := {x | (4 : ℝ) / (x + 1) > 1}
def B (m : ℝ) : Set ℝ := {x | (x - m - 4) * (x - m + 1) > 0}

-- Part 1
theorem union_A_B_when_m_2 : A ∪ B 2 = {x : ℝ | x < 3 ∨ x > 6} := by sorry

-- Part 2
theorem intersection_A_B_empty_iff (m : ℝ) : A ∩ B m = ∅ ↔ -1 ≤ m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_m_2_intersection_A_B_empty_iff_l2450_245035


namespace NUMINAMATH_CALUDE_largest_divisible_digit_l2450_245037

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def last_digit (n : ℕ) : ℕ := n % 10

def number_with_digit (d : ℕ) : ℕ := 78120 + d

theorem largest_divisible_digit : 
  (∀ d : ℕ, d ≤ 9 → is_divisible_by_6 (number_with_digit d) → d ≤ 6) ∧ 
  is_divisible_by_6 (number_with_digit 6) :=
sorry

end NUMINAMATH_CALUDE_largest_divisible_digit_l2450_245037


namespace NUMINAMATH_CALUDE_distance_rides_to_car_l2450_245089

/-- The distance Heather walked from the car to the entrance -/
def distance_car_to_entrance : ℝ := 0.3333333333333333

/-- The distance Heather walked from the entrance to the carnival rides -/
def distance_entrance_to_rides : ℝ := 0.3333333333333333

/-- The total distance Heather walked -/
def total_distance : ℝ := 0.75

/-- The theorem states that given the above distances, 
    the distance Heather walked from the carnival rides back to the car 
    is 0.08333333333333337 miles -/
theorem distance_rides_to_car : 
  total_distance - (distance_car_to_entrance + distance_entrance_to_rides) = 0.08333333333333337 := by
  sorry

end NUMINAMATH_CALUDE_distance_rides_to_car_l2450_245089


namespace NUMINAMATH_CALUDE_linear_equation_exponent_l2450_245082

theorem linear_equation_exponent (k : ℕ) : 
  (∀ x, ∃ a b, x^(k-1) + 3 = a*x + b) → k = 2 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_l2450_245082


namespace NUMINAMATH_CALUDE_seymour_fertilizer_calculation_l2450_245049

/-- Calculates the total fertilizer needed for Seymour's plant shop --/
theorem seymour_fertilizer_calculation : 
  let petunia_flats : ℕ := 4
  let petunias_per_flat : ℕ := 8
  let petunia_fertilizer : ℕ := 8
  let rose_flats : ℕ := 3
  let roses_per_flat : ℕ := 6
  let rose_fertilizer : ℕ := 3
  let sunflower_flats : ℕ := 5
  let sunflowers_per_flat : ℕ := 10
  let sunflower_fertilizer : ℕ := 6
  let orchid_flats : ℕ := 2
  let orchids_per_flat : ℕ := 4
  let orchid_fertilizer : ℕ := 4
  let venus_flytraps : ℕ := 2
  let venus_flytrap_fertilizer : ℕ := 2
  
  petunia_flats * petunias_per_flat * petunia_fertilizer +
  rose_flats * roses_per_flat * rose_fertilizer +
  sunflower_flats * sunflowers_per_flat * sunflower_fertilizer +
  orchid_flats * orchids_per_flat * orchid_fertilizer +
  venus_flytraps * venus_flytrap_fertilizer = 646 := by
  sorry

#check seymour_fertilizer_calculation

end NUMINAMATH_CALUDE_seymour_fertilizer_calculation_l2450_245049


namespace NUMINAMATH_CALUDE_calculation_proof_l2450_245091

theorem calculation_proof : 
  Real.sqrt 5 * (-Real.sqrt 10) - (1/7)⁻¹ + |-(2^3)| = -5 * Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l2450_245091


namespace NUMINAMATH_CALUDE_william_has_more_money_l2450_245013

/-- Represents the amount of money in different currencies --/
structure Money where
  usd_20 : ℕ
  usd_10 : ℕ
  usd_5 : ℕ
  gbp_10 : ℕ
  eur_20 : ℕ

/-- Converts Money to USD --/
def to_usd (m : Money) (gbp_rate : ℚ) (eur_rate : ℚ) : ℚ :=
  (m.usd_20 * 20 + m.usd_10 * 10 + m.usd_5 * 5 + m.gbp_10 * 10 * gbp_rate + m.eur_20 * 20 * eur_rate : ℚ)

/-- Oliver's money --/
def oliver : Money := ⟨10, 0, 3, 12, 0⟩

/-- William's money --/
def william : Money := ⟨0, 15, 4, 0, 20⟩

/-- The exchange rates --/
def gbp_rate : ℚ := 138 / 100
def eur_rate : ℚ := 118 / 100

theorem william_has_more_money :
  to_usd william gbp_rate eur_rate - to_usd oliver gbp_rate eur_rate = 2614 / 10 := by
  sorry

end NUMINAMATH_CALUDE_william_has_more_money_l2450_245013


namespace NUMINAMATH_CALUDE_smallest_coin_count_fifty_seven_satisfies_conditions_smallest_coin_count_is_57_l2450_245093

theorem smallest_coin_count (n : ℕ) : 
  (n % 5 = 2) ∧ (n % 4 = 1) ∧ (n % 3 = 0) → n ≥ 57 :=
by
  sorry

theorem fifty_seven_satisfies_conditions : 
  (57 % 5 = 2) ∧ (57 % 4 = 1) ∧ (57 % 3 = 0) :=
by
  sorry

theorem smallest_coin_count_is_57 : 
  ∃ (n : ℕ), (n % 5 = 2) ∧ (n % 4 = 1) ∧ (n % 3 = 0) ∧ 
  (∀ (m : ℕ), (m % 5 = 2) ∧ (m % 4 = 1) ∧ (m % 3 = 0) → m ≥ n) ∧
  n = 57 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_coin_count_fifty_seven_satisfies_conditions_smallest_coin_count_is_57_l2450_245093


namespace NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l2450_245050

def contains_seven (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + 7 * b ∧ b ≤ 9

theorem smallest_n_with_seven_in_squares : 
  (∀ m : ℕ, m < 26 → ¬(contains_seven (m^2) ∧ contains_seven ((m+1)^2))) ∧
  (contains_seven (26^2) ∧ contains_seven (27^2)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l2450_245050


namespace NUMINAMATH_CALUDE_age_sum_five_years_ago_l2450_245024

theorem age_sum_five_years_ago (djibo_age : ℕ) (sister_age : ℕ) : 
  djibo_age = 17 → sister_age = 28 → djibo_age - 5 + (sister_age - 5) = 35 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_five_years_ago_l2450_245024


namespace NUMINAMATH_CALUDE_base_10_sum_45_l2450_245081

/-- The sum of single-digit numbers in base b -/
def sum_single_digits (b : ℕ) : ℕ := (b - 1) * b / 2

/-- Checks if a number in base b has 5 as its units digit -/
def has_units_digit_5 (n : ℕ) (b : ℕ) : Prop := n % b = 5

theorem base_10_sum_45 :
  ∃ (b : ℕ), b > 1 ∧ sum_single_digits b = 45 ∧ has_units_digit_5 (sum_single_digits b) b ∧ b = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_10_sum_45_l2450_245081


namespace NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l2450_245008

theorem polynomial_multiplication_simplification (x : ℝ) :
  (3 * x - 2) * (5 * x^12 + 3 * x^11 + 5 * x^10 + 3 * x^9) =
  15 * x^13 - x^12 + 9 * x^11 - x^10 - 6 * x^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l2450_245008


namespace NUMINAMATH_CALUDE_constructible_heights_count_l2450_245025

/-- A function that returns the number of constructible heights given a number of bricks and possible height increments. -/
def countConstructibleHeights (numBricks : ℕ) (heightIncrements : List ℕ) : ℕ :=
  sorry

/-- The theorem stating that with 25 bricks and height increments of 0, 3, and 4, there are 98 constructible heights. -/
theorem constructible_heights_count : 
  countConstructibleHeights 25 [0, 3, 4] = 98 :=
sorry

end NUMINAMATH_CALUDE_constructible_heights_count_l2450_245025


namespace NUMINAMATH_CALUDE_price_reduction_effect_l2450_245009

theorem price_reduction_effect (P Q : ℝ) (P_positive : P > 0) (Q_positive : Q > 0) :
  let new_price := P * (1 - 0.35)
  let new_quantity := Q * (1 + 0.8)
  let original_revenue := P * Q
  let new_revenue := new_price * new_quantity
  (new_revenue - original_revenue) / original_revenue = 0.17 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_effect_l2450_245009


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l2450_245004

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l2450_245004


namespace NUMINAMATH_CALUDE_poultry_farm_solution_l2450_245051

/-- Represents the poultry farm problem --/
def poultry_farm_problem (initial_chickens initial_guinea_fowls : ℕ)
  (daily_loss_chickens daily_loss_turkeys daily_loss_guinea_fowls : ℕ)
  (days : ℕ) (total_birds_left : ℕ) : Prop :=
  let initial_turkeys := 200
  let total_initial_birds := initial_chickens + initial_turkeys + initial_guinea_fowls
  let total_loss := (daily_loss_chickens + daily_loss_turkeys + daily_loss_guinea_fowls) * days
  total_initial_birds - total_loss = total_birds_left

/-- Theorem stating the solution to the poultry farm problem --/
theorem poultry_farm_solution :
  poultry_farm_problem 300 80 20 8 5 7 349 := by
  sorry

#check poultry_farm_solution

end NUMINAMATH_CALUDE_poultry_farm_solution_l2450_245051


namespace NUMINAMATH_CALUDE_max_value_of_f_l2450_245061

noncomputable def f (x : ℝ) : ℝ := 2 * (-1) * Real.log x - 1 / x

theorem max_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≤ f x ∧ f x = 2 * Real.log 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2450_245061


namespace NUMINAMATH_CALUDE_y_not_between_l2450_245072

theorem y_not_between (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∀ x y : ℝ, y = (a * Real.sin x + b) / (a * Real.sin x - b) →
  (a > b → (y ≥ (a - b) / (a + b) ∨ y ≤ (a + b) / (a - b))) :=
by sorry

end NUMINAMATH_CALUDE_y_not_between_l2450_245072


namespace NUMINAMATH_CALUDE_solution_exists_l2450_245007

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the system of equations
def equation_system (x y : ℝ) : Prop :=
  x ≠ 0 ∧ y > 0 ∧ log10 (x^2 / y^3) = 1 ∧ log10 (x^2 * y^3) = 7

-- Theorem statement
theorem solution_exists :
  ∃ x y : ℝ, equation_system x y ∧ (x = 100 ∨ x = -100) ∧ y = 10 :=
by sorry

end NUMINAMATH_CALUDE_solution_exists_l2450_245007


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2450_245042

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a - b) / (b + c) + (b - c) / (c + d) + (c - d) / (d + a) + (d - a) / (a + b) ≥ 0 ∧
  ((a - b) / (b + c) + (b - c) / (c + d) + (c - d) / (d + a) + (d - a) / (a + b) = 0 ↔ a = c ∧ b = d) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2450_245042


namespace NUMINAMATH_CALUDE_max_value_x3_minus_y3_l2450_245033

theorem max_value_x3_minus_y3 (x y : ℝ) 
  (h1 : 3 * (x^3 + y^3) = x + y) 
  (h2 : x + y = 1) : 
  ∃ (max : ℝ), max = 7/27 ∧ ∀ (a b : ℝ), 3 * (a^3 + b^3) = a + b → a + b = 1 → a^3 - b^3 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_x3_minus_y3_l2450_245033


namespace NUMINAMATH_CALUDE_axis_of_symmetry_sine_curve_l2450_245092

/-- The axis of symmetry for the sine curve y = sin(2πx - π/3) is x = 5/12 -/
theorem axis_of_symmetry_sine_curve (x : ℝ) : 
  (∃ (k : ℤ), x = k / 2 + 5 / 12) ↔ 
  (∃ (n : ℤ), 2 * π * x - π / 3 = n * π + π / 2) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_sine_curve_l2450_245092


namespace NUMINAMATH_CALUDE_gcd_102_238_l2450_245083

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l2450_245083


namespace NUMINAMATH_CALUDE_existence_equivalence_l2450_245034

/-- Proves the equivalence between the existence of x in [1, 2] satisfying 
    2x^2 - ax + 2 > 0 and a < 4 for any real number a -/
theorem existence_equivalence (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 2 * x^2 - a * x + 2 > 0) ↔ a < 4 := by
  sorry

#check existence_equivalence

end NUMINAMATH_CALUDE_existence_equivalence_l2450_245034
