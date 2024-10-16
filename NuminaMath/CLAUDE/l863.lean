import Mathlib

namespace NUMINAMATH_CALUDE_max_area_semicircle_l863_86311

/-- A semicircle with diameter AB and radius R -/
structure Semicircle where
  R : ℝ
  A : Point
  B : Point

/-- Points C and D on the semicircle -/
structure PointsOnSemicircle (S : Semicircle) where
  C : Point
  D : Point

/-- The area of quadrilateral ACDB -/
def area (S : Semicircle) (P : PointsOnSemicircle S) : ℝ :=
  sorry

/-- C and D divide the semicircle into three equal parts -/
def equalParts (S : Semicircle) (P : PointsOnSemicircle S) : Prop :=
  sorry

theorem max_area_semicircle (S : Semicircle) :
  ∃ (P : PointsOnSemicircle S),
    equalParts S P ∧
    ∀ (Q : PointsOnSemicircle S), area S Q ≤ area S P ∧
    area S P = (3 * Real.sqrt 3 / 4) * S.R^2 :=
  sorry

end NUMINAMATH_CALUDE_max_area_semicircle_l863_86311


namespace NUMINAMATH_CALUDE_somu_present_age_l863_86349

/-- Somu's age -/
def somu_age : ℕ := sorry

/-- Somu's father's age -/
def father_age : ℕ := sorry

/-- Somu's age is one-third of his father's age -/
axiom current_age_ratio : somu_age = father_age / 3

/-- 5 years ago, Somu's age was one-fifth of his father's age -/
axiom past_age_ratio : somu_age - 5 = (father_age - 5) / 5

theorem somu_present_age : somu_age = 10 := by sorry

end NUMINAMATH_CALUDE_somu_present_age_l863_86349


namespace NUMINAMATH_CALUDE_spinsters_to_cats_ratio_l863_86301

theorem spinsters_to_cats_ratio 
  (spinsters : ℕ) 
  (cats : ℕ) 
  (x : ℚ)
  (ratio_condition : spinsters / cats = x / 9)
  (difference_condition : cats = spinsters + 63)
  (spinsters_count : spinsters = 18) :
  spinsters / cats = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_spinsters_to_cats_ratio_l863_86301


namespace NUMINAMATH_CALUDE_linear_equations_compatibility_l863_86390

theorem linear_equations_compatibility (a b c d : ℝ) :
  (∃ x : ℝ, a * x + b = 0 ∧ c * x + d = 0) ↔ a * d - b * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_equations_compatibility_l863_86390


namespace NUMINAMATH_CALUDE_distinct_power_representations_l863_86389

theorem distinct_power_representations : ∃ (N : ℕ) 
  (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℕ), 
  (∃ (x y : ℕ), a₁ = x^2 ∧ a₂ = y^2) ∧
  (∃ (x y : ℕ), b₁ = x^3 ∧ b₂ = y^3) ∧
  (∃ (x y : ℕ), c₁ = x^5 ∧ c₂ = y^5) ∧
  (∃ (x y : ℕ), d₁ = x^7 ∧ d₂ = y^7) ∧
  N = a₁ - a₂ ∧
  N = b₁ - b₂ ∧
  N = c₁ - c₂ ∧
  N = d₁ - d₂ ∧
  a₁ ≠ b₁ ∧ a₁ ≠ c₁ ∧ a₁ ≠ d₁ ∧ b₁ ≠ c₁ ∧ b₁ ≠ d₁ ∧ c₁ ≠ d₁ :=
by
  sorry

end NUMINAMATH_CALUDE_distinct_power_representations_l863_86389


namespace NUMINAMATH_CALUDE_log_3_base_5_l863_86310

theorem log_3_base_5 (a : ℝ) (h : Real.log 45 / Real.log 5 = a) :
  Real.log 3 / Real.log 5 = (a - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_3_base_5_l863_86310


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l863_86346

theorem interest_rate_calculation 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 6100) 
  (h2 : time = 2) 
  (h3 : interest_difference = 61) : 
  ∃ (rate : ℝ), 
    rate = 1 ∧ 
    principal * ((1 + rate / 100) ^ time - 1) - principal * rate * time / 100 = interest_difference :=
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l863_86346


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l863_86338

/-- Proves that 3 liters of 33% alcohol solution mixed with 1 liter of water results in 24.75% alcohol concentration -/
theorem alcohol_mixture_proof (x : ℝ) :
  (x > 0) →
  (0.33 * x = 0.2475 * (x + 1)) →
  x = 3 := by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l863_86338


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l863_86318

theorem arcsin_equation_solution (x : ℝ) :
  (Real.arcsin x + Real.arcsin (3 * x) = π / 4) →
  (x = Real.sqrt (1 / (9 + 4 * Real.sqrt 2)) ∨
   x = -Real.sqrt (1 / (9 + 4 * Real.sqrt 2))) ∧
  (x ≥ -1 ∧ x ≤ 1) ∧ (3 * x ≥ -1 ∧ 3 * x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l863_86318


namespace NUMINAMATH_CALUDE_equation_solution_solution_set_l863_86378

theorem equation_solution (x : ℝ) : 
  x ≠ 7 → (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 3*x - 18) / (x^2 - 4*x - 21) :=
by sorry

theorem solution_set : 
  {x : ℝ | (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 3*x - 18) / (x^2 - 4*x - 21)} = {x : ℝ | x ≠ 7} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_solution_set_l863_86378


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l863_86351

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 2 + a 4 = 5) :
  a 5 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l863_86351


namespace NUMINAMATH_CALUDE_total_seeds_calculation_l863_86313

/-- The number of seeds planted in each flower bed -/
def seeds_per_bed : ℕ := 6

/-- The number of flower beds -/
def num_beds : ℕ := 9

/-- The total number of seeds planted -/
def total_seeds : ℕ := seeds_per_bed * num_beds

theorem total_seeds_calculation : total_seeds = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_calculation_l863_86313


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l863_86330

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd f) (h_periodic : is_periodic_2 f) (h_value : f (1 + a) = 1) :
  f (1 - a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l863_86330


namespace NUMINAMATH_CALUDE_inequality_solution_l863_86316

theorem inequality_solution (x : ℝ) : 2 * (3 * x - 2) > x + 1 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l863_86316


namespace NUMINAMATH_CALUDE_exp_two_pi_third_in_second_quadrant_l863_86344

-- Define the complex exponential function
noncomputable def complex_exp (z : ℂ) : ℂ := Real.exp z.re * (Real.cos z.im + Complex.I * Real.sin z.im)

-- Define the second quadrant of the complex plane
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem exp_two_pi_third_in_second_quadrant :
  second_quadrant (complex_exp (Complex.I * (2 * Real.pi / 3))) :=
sorry

end NUMINAMATH_CALUDE_exp_two_pi_third_in_second_quadrant_l863_86344


namespace NUMINAMATH_CALUDE_max_value_of_z_l863_86334

theorem max_value_of_z (x y : ℝ) 
  (h1 : |2*x + y + 1| ≤ |x + 2*y + 2|) 
  (h2 : -1 ≤ y) (h3 : y ≤ 1) : 
  (∀ (x' y' : ℝ), |2*x' + y' + 1| ≤ |x' + 2*y' + 2| → -1 ≤ y' → y' ≤ 1 → 2*x' + y' ≤ 2*x + y) →
  2*x + y = 5 := by sorry

end NUMINAMATH_CALUDE_max_value_of_z_l863_86334


namespace NUMINAMATH_CALUDE_hemisphere_container_volume_l863_86317

/-- Given a total volume of water and the number of hemisphere containers needed,
    calculate the volume of each hemisphere container. -/
theorem hemisphere_container_volume
  (total_volume : ℝ)
  (num_containers : ℕ)
  (h_total_volume : total_volume = 10976)
  (h_num_containers : num_containers = 2744) :
  total_volume / num_containers = 4 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_container_volume_l863_86317


namespace NUMINAMATH_CALUDE_tshirt_purchase_cost_l863_86320

theorem tshirt_purchase_cost : 
  let num_fandoms : ℕ := 4
  let shirts_per_fandom : ℕ := 5
  let original_price : ℚ := 15
  let initial_discount : ℚ := 0.2
  let additional_discount : ℚ := 0.1
  let seasonal_discount : ℚ := 0.25
  let seasonal_discount_portion : ℚ := 0.5
  let tax_rate : ℚ := 0.1

  let total_shirts := num_fandoms * shirts_per_fandom
  let original_total := total_shirts * original_price
  let after_initial_discount := original_total * (1 - initial_discount)
  let after_additional_discount := after_initial_discount * (1 - additional_discount)
  let seasonal_discount_amount := (original_total * seasonal_discount_portion) * seasonal_discount
  let after_all_discounts := after_additional_discount - seasonal_discount_amount
  let final_cost := after_all_discounts * (1 + tax_rate)

  final_cost = 196.35 := by sorry

end NUMINAMATH_CALUDE_tshirt_purchase_cost_l863_86320


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l863_86356

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 2 ∧ y ≥ 2 → x + y ≥ 4) ∧
  (∃ x y : ℝ, x + y ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l863_86356


namespace NUMINAMATH_CALUDE_total_cookies_l863_86319

/-- Given 26 bags of cookies with 2 cookies in each bag, prove that the total number of cookies is 52. -/
theorem total_cookies (num_bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : num_bags = 26) 
  (h2 : cookies_per_bag = 2) : 
  num_bags * cookies_per_bag = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l863_86319


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l863_86393

-- Define a triangle ABC
structure Triangle (α : Type) [Field α] where
  A : α
  B : α
  C : α
  a : α
  b : α
  c : α

-- State the theorem
theorem triangle_angle_proof {α : Type} [Field α] (ABC : Triangle α) :
  ABC.b = 2 * ABC.a →
  ABC.B = ABC.A + 60 →
  ABC.A = 30 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l863_86393


namespace NUMINAMATH_CALUDE_men_earnings_l863_86362

/-- Represents the total earnings of workers over a week. -/
structure Earnings where
  men : ℝ
  women : ℝ
  boys : ℝ

/-- Represents the work rates and hours of different groups of workers. -/
structure WorkData where
  X : ℝ  -- Number of women equivalent to 5 men
  M : ℝ  -- Hours worked by men
  W : ℝ  -- Hours worked by women
  B : ℝ  -- Hours worked by boys
  rm : ℝ  -- Wage rate for men per hour
  rw : ℝ  -- Wage rate for women per hour
  rb : ℝ  -- Wage rate for boys per hour

/-- Theorem stating the total earnings for men given the problem conditions. -/
theorem men_earnings (data : WorkData) (total : Earnings) :
  (5 : ℝ) * data.X * data.W * data.rw = (8 : ℝ) * data.B * data.rb →
  total.men + total.women + total.boys = 180 →
  total.men = (5 : ℝ) * data.M * data.rm :=
by sorry

end NUMINAMATH_CALUDE_men_earnings_l863_86362


namespace NUMINAMATH_CALUDE_largest_common_term_l863_86382

theorem largest_common_term (n m : ℕ) : 
  163 = 3 + 8 * n ∧ 
  163 = 5 + 9 * m ∧ 
  163 ≤ 200 ∧ 
  ∀ k, k > 163 → k ≤ 200 → (k - 3) % 8 ≠ 0 ∨ (k - 5) % 9 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l863_86382


namespace NUMINAMATH_CALUDE_binomial_25_2_l863_86341

theorem binomial_25_2 : Nat.choose 25 2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_binomial_25_2_l863_86341


namespace NUMINAMATH_CALUDE_complex_calculation_l863_86376

theorem complex_calculation (A M N : ℂ) (Q : ℝ) :
  A = 5 - 2*I →
  M = -3 + 2*I →
  N = 3*I →
  Q = 3 →
  (A - M + N - Q) * I = 1 + 5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_calculation_l863_86376


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l863_86355

theorem smallest_n_congruence (n : ℕ) : 
  (∀ m : ℕ, 0 < m → m < 15 → ¬(890 * m ≡ 1426 * m [ZMOD 30])) ∧ 
  (890 * 15 ≡ 1426 * 15 [ZMOD 30]) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l863_86355


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_c_l863_86336

theorem quadratic_roots_imply_c (c : ℚ) : 
  (∀ x : ℚ, 2 * x^2 + 14 * x + c = 0 ↔ x = (-14 + Real.sqrt 10) / 4 ∨ x = (-14 - Real.sqrt 10) / 4) →
  c = 93 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_c_l863_86336


namespace NUMINAMATH_CALUDE_pat_stickers_l863_86357

theorem pat_stickers (initial_stickers earned_stickers : ℕ) 
  (h1 : initial_stickers = 39)
  (h2 : earned_stickers = 22) :
  initial_stickers + earned_stickers = 61 := by
  sorry

end NUMINAMATH_CALUDE_pat_stickers_l863_86357


namespace NUMINAMATH_CALUDE_expected_score_is_seven_sixths_l863_86395

/-- Represents the score obtained from a single die roll -/
inductive Score
| one
| two
| three

/-- The probability of getting each score -/
def prob (s : Score) : ℚ :=
  match s with
  | Score.one => 1/2
  | Score.two => 1/3
  | Score.three => 1/6

/-- The point value associated with each score -/
def value (s : Score) : ℕ :=
  match s with
  | Score.one => 1
  | Score.two => 2
  | Score.three => 3

/-- The expected score for a single roll of the die -/
def expected_score : ℚ :=
  (prob Score.one * value Score.one) +
  (prob Score.two * value Score.two) +
  (prob Score.three * value Score.three)

theorem expected_score_is_seven_sixths :
  expected_score = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_expected_score_is_seven_sixths_l863_86395


namespace NUMINAMATH_CALUDE_cosine_angle_equality_l863_86332

theorem cosine_angle_equality (n : ℤ) : 
  (0 ≤ n ∧ n ≤ 180) ∧ (Real.cos (n * π / 180) = Real.cos (1124 * π / 180)) ↔ n = 44 := by
  sorry

end NUMINAMATH_CALUDE_cosine_angle_equality_l863_86332


namespace NUMINAMATH_CALUDE_equation_solution_l863_86354

theorem equation_solution : ∃ x : ℚ, (5 * x + 12 * x = 540 - 12 * (x - 5)) ∧ (x = 600 / 29) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l863_86354


namespace NUMINAMATH_CALUDE_sum_of_multiples_l863_86308

def smallest_three_digit_multiple_of_5 : ℕ := 100

def smallest_four_digit_multiple_of_7 : ℕ := 1001

theorem sum_of_multiples : 
  smallest_three_digit_multiple_of_5 + smallest_four_digit_multiple_of_7 = 1101 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l863_86308


namespace NUMINAMATH_CALUDE_train_length_l863_86379

/-- The length of a train given its crossing time, bridge length, and speed. -/
theorem train_length (crossing_time : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) :
  crossing_time = 29.997600191984642 →
  bridge_length = 200 →
  train_speed_kmph = 36 →
  ∃ (train_length : ℝ), abs (train_length - 99.976) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l863_86379


namespace NUMINAMATH_CALUDE_ratio_of_angles_l863_86329

-- Define the circle and triangle
def Circle : Type := Unit
def Point : Type := Unit
def Triangle : Type := Unit

-- Define the center of the circle
def O : Point := sorry

-- Define the vertices of the triangle
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry

-- Define point E
def E : Point := sorry

-- Define the inscribed triangle
def triangle_ABC : Triangle := sorry

-- Define the arcs
def arc_AB : ℝ := 100
def arc_BC : ℝ := 80

-- Define the perpendicular condition
def OE_perp_AC : Prop := sorry

-- Define the angles
def angle_OBE : ℝ := sorry
def angle_BAC : ℝ := sorry

-- State the theorem
theorem ratio_of_angles (circle : Circle) (triangle_ABC : Triangle) 
  (h1 : arc_AB = 100)
  (h2 : arc_BC = 80)
  (h3 : OE_perp_AC) :
  angle_OBE / angle_BAC = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_ratio_of_angles_l863_86329


namespace NUMINAMATH_CALUDE_function_inequality_l863_86399

open Real

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x, deriv f x < f x) : 
  f 1 < ℯ * f 0 ∧ f 2014 < ℯ^2014 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l863_86399


namespace NUMINAMATH_CALUDE_f_g_four_zeros_implies_a_range_l863_86342

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*a*x - a + 1 else Real.log (-x)

def g (a : ℝ) (x : ℝ) : ℝ := x^2 + 1 - 2*a

def has_four_zeros (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧
    ∀ x, f x = 0 → (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

theorem f_g_four_zeros_implies_a_range (a : ℝ) :
  has_four_zeros (f a ∘ g a) →
  a ∈ Set.Ioo ((Real.sqrt 5 - 1) / 2) 1 ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_f_g_four_zeros_implies_a_range_l863_86342


namespace NUMINAMATH_CALUDE_point_not_in_region_iff_m_in_interval_l863_86326

/-- The function representing the left side of the inequality -/
def f (m x y : ℝ) : ℝ := x - (m^2 - 2*m + 4)*y - 6

/-- The theorem stating the equivalence between the point (-1, -1) not being in the region
    and m being in the interval [-1, 3] -/
theorem point_not_in_region_iff_m_in_interval :
  ∀ m : ℝ, f m (-1) (-1) ≤ 0 ↔ -1 ≤ m ∧ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_point_not_in_region_iff_m_in_interval_l863_86326


namespace NUMINAMATH_CALUDE_cross_product_scalar_m_l863_86327

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the cross product operation
variable (cross : V → V → V)

-- Axioms for cross product
variable (cross_anticomm : ∀ a b : V, cross a b = - cross b a)
variable (cross_distributive : ∀ a b c : V, cross a (b + c) = cross a b + cross a c)
variable (cross_zero : ∀ a : V, cross a a = 0)

-- The main theorem
theorem cross_product_scalar_m (m : ℝ) : 
  (∀ u v w : V, u + v + w = 0 → 
    m • (cross v u) + cross v w + cross w u = cross v u) → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_cross_product_scalar_m_l863_86327


namespace NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l863_86328

theorem vector_subtraction_scalar_multiplication (a b : ℝ × ℝ) :
  a = (3, -8) → b = (2, -6) → a - 5 • b = (-7, 22) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l863_86328


namespace NUMINAMATH_CALUDE_inequality_proof_l863_86315

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ 2 / (1 - x*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l863_86315


namespace NUMINAMATH_CALUDE_smallest_steps_l863_86303

theorem smallest_steps (n : ℕ) : 
  n > 20 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 2 ∧ 
  (∀ m : ℕ, m > 20 ∧ m % 6 = 4 ∧ m % 7 = 2 → n ≤ m) → 
  n = 58 := by
sorry

end NUMINAMATH_CALUDE_smallest_steps_l863_86303


namespace NUMINAMATH_CALUDE_triangular_intersection_solids_l863_86373

-- Define the types for geometric solids and plane
inductive GeometricSolid
| Cone
| Cylinder
| Pyramid
| Cube

structure Plane

-- Define the intersection of a plane and a geometric solid
def Intersection (p : Plane) (s : GeometricSolid) : Set (ℝ × ℝ × ℝ) := sorry

-- Define what it means for an intersection to be triangular
def IsTriangularIntersection (i : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- The main theorem
theorem triangular_intersection_solids 
  (p : Plane) (s : GeometricSolid) 
  (h : IsTriangularIntersection (Intersection p s)) : 
  s = GeometricSolid.Cone ∨ s = GeometricSolid.Pyramid ∨ s = GeometricSolid.Cube := by
  sorry


end NUMINAMATH_CALUDE_triangular_intersection_solids_l863_86373


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l863_86364

theorem perfect_square_trinomial_m_value (m : ℝ) : 
  (∃ k : ℝ, ∀ y : ℝ, 4*y^2 - m*y + 25 = (2*y - k)^2) → 
  (m = 20 ∨ m = -20) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l863_86364


namespace NUMINAMATH_CALUDE_sams_remaining_dimes_l863_86375

theorem sams_remaining_dimes 
  (initial_dimes : ℕ) 
  (borrowed_dimes : ℕ) 
  (h1 : initial_dimes = 8) 
  (h2 : borrowed_dimes = 4) :
  initial_dimes - borrowed_dimes = 4 :=
by sorry

end NUMINAMATH_CALUDE_sams_remaining_dimes_l863_86375


namespace NUMINAMATH_CALUDE_total_weight_is_8040_l863_86302

/-- Represents the catering setup for an event -/
structure CateringSetup where
  numTables : Nat
  settingsPerTable : Nat
  backupPercentage : Rat
  forkWeight : Rat
  knifeWeight : Rat
  spoonWeight : Rat
  largePlateWeight : Rat
  smallPlateWeight : Rat
  wineGlassWeight : Rat
  waterGlassWeight : Rat
  tableDecorationWeight : Rat

/-- Calculates the total weight of all items for the catering setup -/
def totalWeight (setup : CateringSetup) : Rat :=
  let totalSettings := setup.numTables * setup.settingsPerTable * (1 + setup.backupPercentage)
  let silverwareWeight := totalSettings * (setup.forkWeight + setup.knifeWeight + setup.spoonWeight)
  let plateWeight := totalSettings * (setup.largePlateWeight + setup.smallPlateWeight)
  let glassWeight := totalSettings * (setup.wineGlassWeight + setup.waterGlassWeight)
  let decorationWeight := setup.numTables * setup.tableDecorationWeight
  silverwareWeight + plateWeight + glassWeight + decorationWeight

/-- Theorem stating that the total weight for the given setup is 8040 ounces -/
theorem total_weight_is_8040 (setup : CateringSetup) 
    (h1 : setup.numTables = 15)
    (h2 : setup.settingsPerTable = 8)
    (h3 : setup.backupPercentage = 1/4)
    (h4 : setup.forkWeight = 7/2)
    (h5 : setup.knifeWeight = 4)
    (h6 : setup.spoonWeight = 9/2)
    (h7 : setup.largePlateWeight = 14)
    (h8 : setup.smallPlateWeight = 10)
    (h9 : setup.wineGlassWeight = 7)
    (h10 : setup.waterGlassWeight = 9)
    (h11 : setup.tableDecorationWeight = 16) :
    totalWeight setup = 8040 := by
  sorry


end NUMINAMATH_CALUDE_total_weight_is_8040_l863_86302


namespace NUMINAMATH_CALUDE_quadratic_sum_l863_86380

/-- Given a quadratic expression 5x^2 - 20x + 8, when expressed in the form a(x - h)^2 + k,
    the sum a + h + k equals -5. -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a h k : ℝ), (5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) ∧ (a + h + k = -5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l863_86380


namespace NUMINAMATH_CALUDE_expand_expression_l863_86343

theorem expand_expression (x : ℝ) : 3 * (8 * x^2 - 2 * x + 1) = 24 * x^2 - 6 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l863_86343


namespace NUMINAMATH_CALUDE_function_is_zero_l863_86372

def is_logarithmic_property (f : ℕ+ → ℝ) : Prop :=
  ∀ m n : ℕ+, f (m * n) = f m + f n

def is_non_decreasing (f : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, f (n + 1) ≥ f n

theorem function_is_zero
  (f : ℕ+ → ℝ)
  (h1 : is_logarithmic_property f)
  (h2 : is_non_decreasing f) :
  ∀ n : ℕ+, f n = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_is_zero_l863_86372


namespace NUMINAMATH_CALUDE_unique_function_is_identity_l863_86353

-- Define a type for our function
def ContinuousRealFunction := ℝ → ℝ

-- Define the properties of our function
def SatisfiesConditions (f : ContinuousRealFunction) : Prop :=
  Continuous f ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → f x > M) ∧
  (∀ φ : ℝ, ∀ x y : ℝ, |x - y| > φ → 
    ∃ n : ℕ, n < φ^2023 ∧ (f^[n] x + f^[n] y = x + y))

-- State the theorem
theorem unique_function_is_identity 
  (f : ContinuousRealFunction) 
  (h : SatisfiesConditions f) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_unique_function_is_identity_l863_86353


namespace NUMINAMATH_CALUDE_book_arrangements_eq_1440_l863_86350

/-- The number of ways to arrange 8 books (3 Russian, 2 French, and 3 Italian) on a shelf,
    keeping the Russian books together and the French books together. -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 8
  let russian_books : ℕ := 3
  let french_books : ℕ := 2
  let italian_books : ℕ := 3
  let russian_unit : ℕ := 1
  let french_unit : ℕ := 1
  let total_units : ℕ := russian_unit + french_unit + italian_books
  Nat.factorial total_units * Nat.factorial russian_books * Nat.factorial french_books

/-- Theorem stating that the number of book arrangements is 1440. -/
theorem book_arrangements_eq_1440 : book_arrangements = 1440 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangements_eq_1440_l863_86350


namespace NUMINAMATH_CALUDE_abs_g_zero_eq_forty_l863_86371

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- The property that the absolute value of g at specific points is 10 -/
def HasSpecificValues (g : ThirdDegreePolynomial) : Prop :=
  |g (-2)| = 10 ∧ |g 1| = 10 ∧ |g 4| = 10 ∧ |g 5| = 10 ∧ |g 6| = 10 ∧ |g 9| = 10

/-- The theorem stating that if g satisfies the specific values, then |g(0)| = 40 -/
theorem abs_g_zero_eq_forty (g : ThirdDegreePolynomial) 
  (h : HasSpecificValues g) : |g 0| = 40 := by
  sorry

end NUMINAMATH_CALUDE_abs_g_zero_eq_forty_l863_86371


namespace NUMINAMATH_CALUDE_survey_B_most_suitable_for_census_l863_86396

-- Define the characteristics of a survey
structure Survey where
  population : Set String
  method : String
  is_destructive : Bool
  is_manageable : Bool

-- Define the conditions for a census
def is_census_suitable (s : Survey) : Prop :=
  s.is_manageable ∧ ¬s.is_destructive ∧ s.method = "complete enumeration"

-- Define the surveys
def survey_A : Survey := {
  population := {"televisions"},
  method := "sampling",
  is_destructive := true,
  is_manageable := false
}

def survey_B : Survey := {
  population := {"ninth grade students in a certain middle school class"},
  method := "complete enumeration",
  is_destructive := false,
  is_manageable := true
}

def survey_C : Survey := {
  population := {"middle school students in Chongqing"},
  method := "sampling",
  is_destructive := false,
  is_manageable := false
}

def survey_D : Survey := {
  population := {"middle school students in Chongqing"},
  method := "sampling",
  is_destructive := false,
  is_manageable := false
}

-- Theorem stating that survey B is the most suitable for a census
theorem survey_B_most_suitable_for_census :
  is_census_suitable survey_B ∧
  ¬is_census_suitable survey_A ∧
  ¬is_census_suitable survey_C ∧
  ¬is_census_suitable survey_D :=
sorry

end NUMINAMATH_CALUDE_survey_B_most_suitable_for_census_l863_86396


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_five_l863_86306

theorem sqrt_sum_equals_five (x y : ℝ) (h : y = Real.sqrt (x - 9) - Real.sqrt (9 - x) + 4) :
  Real.sqrt x + Real.sqrt y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_five_l863_86306


namespace NUMINAMATH_CALUDE_average_salary_feb_to_may_l863_86361

def average_salary_jan_to_apr : ℕ := 8000
def salary_jan : ℕ := 6100
def salary_may : ℕ := 6500
def target_average : ℕ := 8100

theorem average_salary_feb_to_may :
  (4 * average_salary_jan_to_apr - salary_jan + salary_may) / 4 = target_average :=
sorry

end NUMINAMATH_CALUDE_average_salary_feb_to_may_l863_86361


namespace NUMINAMATH_CALUDE_collinear_vectors_magnitude_l863_86358

def a : ℝ × ℝ := (1, 2)
def b (k : ℝ) : ℝ × ℝ := (-2, k)

def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v = t • w ∨ w = t • v

theorem collinear_vectors_magnitude (k : ℝ) :
  collinear a (b k) →
  ‖(3 • a) + (b k)‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_magnitude_l863_86358


namespace NUMINAMATH_CALUDE_bryan_bookshelves_l863_86314

/-- Given that Bryan has a total number of books and each bookshelf contains a fixed number of books,
    calculate the number of bookshelves he has. -/
def calculate_bookshelves (total_books : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books / books_per_shelf

/-- Prove that Bryan has 19 bookshelves given the conditions. -/
theorem bryan_bookshelves :
  calculate_bookshelves 38 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_bryan_bookshelves_l863_86314


namespace NUMINAMATH_CALUDE_base_5_sum_l863_86388

/-- Represents a digit in base 5 -/
def Base5Digit := { n : ℕ // n > 0 ∧ n < 5 }

/-- Converts a three-digit number in base 5 to its decimal representation -/
def toDecimal (a b c : Base5Digit) : ℕ := 25 * a.val + 5 * b.val + c.val

theorem base_5_sum (A B C : Base5Digit) 
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hSum : toDecimal A B C + toDecimal B C A + toDecimal C A B = 25 * 31 * A.val) :
  B.val + C.val = 4 :=
sorry

end NUMINAMATH_CALUDE_base_5_sum_l863_86388


namespace NUMINAMATH_CALUDE_triangle_sides_l863_86360

theorem triangle_sides (average_length : ℝ) (perimeter : ℝ) (n : ℕ) :
  average_length = 12 →
  perimeter = 36 →
  average_length * n = perimeter →
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_l863_86360


namespace NUMINAMATH_CALUDE_expenditure_ratio_l863_86325

/-- Given a person's income and savings pattern over two years, prove the ratio of total expenditure to first year expenditure --/
theorem expenditure_ratio (income : ℝ) (h1 : income > 0) : 
  let first_year_savings := 0.25 * income
  let first_year_expenditure := income - first_year_savings
  let second_year_income := 1.25 * income
  let second_year_savings := 2 * first_year_savings
  let second_year_expenditure := second_year_income - second_year_savings
  let total_expenditure := first_year_expenditure + second_year_expenditure
  (total_expenditure / first_year_expenditure) = 2 := by
  sorry


end NUMINAMATH_CALUDE_expenditure_ratio_l863_86325


namespace NUMINAMATH_CALUDE_original_aotd_votes_l863_86365

/-- Represents the vote counts for three books --/
structure VoteCounts where
  got : ℕ  -- Game of Thrones
  twi : ℕ  -- Twilight
  aotd : ℕ  -- The Art of the Deal

/-- Represents the vote alteration process --/
def alter_votes (v : VoteCounts) : ℚ × ℚ × ℚ :=
  (v.got, v.twi / 2, v.aotd / 5)

/-- The theorem to be proved --/
theorem original_aotd_votes (v : VoteCounts) : 
  v.got = 10 ∧ v.twi = 12 ∧ 
  (let (got, twi, aotd) := alter_votes v
   got = (got + twi + aotd) / 2) →
  v.aotd = 20 :=
by sorry

end NUMINAMATH_CALUDE_original_aotd_votes_l863_86365


namespace NUMINAMATH_CALUDE_education_allocation_l863_86307

def town_budget : ℕ := 32000000

theorem education_allocation :
  let policing : ℕ := town_budget / 2
  let public_spaces : ℕ := 4000000
  let education : ℕ := town_budget - (policing + public_spaces)
  education = 12000000 := by sorry

end NUMINAMATH_CALUDE_education_allocation_l863_86307


namespace NUMINAMATH_CALUDE_chicken_rabbit_equations_l863_86367

/-- Represents the "chicken-rabbit in the same cage" problem --/
def chicken_rabbit_problem (x y : ℕ) : Prop :=
  let total_heads : ℕ := 35
  let total_feet : ℕ := 94
  let chicken_feet : ℕ := 2
  let rabbit_feet : ℕ := 4
  (x + y = total_heads) ∧ (chicken_feet * x + rabbit_feet * y = total_feet)

/-- Proves that the system of equations correctly represents the problem --/
theorem chicken_rabbit_equations : 
  ∀ x y : ℕ, chicken_rabbit_problem x y ↔ (x + y = 35 ∧ 2*x + 4*y = 94) := by
  sorry

end NUMINAMATH_CALUDE_chicken_rabbit_equations_l863_86367


namespace NUMINAMATH_CALUDE_science_tech_group_size_l863_86337

theorem science_tech_group_size :
  ∀ (girls boys : ℕ),
  girls = 18 →
  girls = 2 * boys - 2 →
  girls + boys = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_science_tech_group_size_l863_86337


namespace NUMINAMATH_CALUDE_trigonometric_identities_l863_86359

theorem trigonometric_identities (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi) (h3 : Real.tan α = -2) : 
  ((2 * Real.cos (Real.pi / 2 + α) - Real.cos (Real.pi - α)) / 
   (Real.sin (Real.pi / 2 - α) - 3 * Real.sin (Real.pi + α)) = -1) ∧
  (2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 11/5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l863_86359


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l863_86369

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 70) 
  (h2 : average_speed = 80) : 
  (2 * average_speed - speed_first_hour) = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l863_86369


namespace NUMINAMATH_CALUDE_two_a_minus_b_value_l863_86377

theorem two_a_minus_b_value (a b : ℝ) 
  (ha : |a| = 4)
  (hb : |b| = 5)
  (hab : |a + b| = -(a + b)) :
  2*a - b = 13 ∨ 2*a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_two_a_minus_b_value_l863_86377


namespace NUMINAMATH_CALUDE_f_behavior_at_infinity_l863_86335

def f (x : ℝ) := -3 * x^3 + 4 * x^2 + 1

theorem f_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → f x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → f x > M) :=
by sorry

end NUMINAMATH_CALUDE_f_behavior_at_infinity_l863_86335


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l863_86304

theorem ratio_x_to_y (x y : ℚ) (h : (14*x - 5*y) / (17*x - 3*y) = 2/7) : x/y = 29/64 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l863_86304


namespace NUMINAMATH_CALUDE_product_divisible_by_17_l863_86385

theorem product_divisible_by_17 : 
  17 ∣ (2002 + 3) * (2003 + 3) * (2004 + 3) * (2005 + 3) * (2006 + 3) * (2007 + 3) := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_17_l863_86385


namespace NUMINAMATH_CALUDE_triangle_angle_c_l863_86391

theorem triangle_angle_c (A B C : ℝ) (h1 : A + B + C = π) 
  (h2 : |Real.cos A - 1/2| + 2*(1 - Real.tan B)^2 = 0) : C = π/2.4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l863_86391


namespace NUMINAMATH_CALUDE_last_digit_is_11_l863_86383

def fibonacci_mod_12 : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => (fibonacci_mod_12 (n + 1) + fibonacci_mod_12 n) % 12

def digit_appears (d : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ n ∧ fibonacci_mod_12 k = d

theorem last_digit_is_11 :
  ∀ d : ℕ, d < 12 →
    ∃ n : ℕ, digit_appears d n ∧
      ¬∃ m : ℕ, m > n ∧ digit_appears 11 m ∧ ¬digit_appears 11 n :=
by sorry

end NUMINAMATH_CALUDE_last_digit_is_11_l863_86383


namespace NUMINAMATH_CALUDE_solve_equation_l863_86348

theorem solve_equation (x : ℝ) : 11 + Real.sqrt (-4 + 6 * x / 3) = 13 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l863_86348


namespace NUMINAMATH_CALUDE_knowledge_competition_probabilities_l863_86394

/-- Represents the outcome of answering a question -/
inductive Answer
| Correct
| Incorrect

/-- Represents the state of a contestant in the competition -/
structure ContestantState where
  score : ℕ
  questions_answered : ℕ

/-- Represents the probabilities of correctly answering each question -/
structure QuestionProbabilities where
  pA : ℚ
  pB : ℚ
  pC : ℚ
  pD : ℚ

/-- Updates the contestant's state based on their answer -/
def updateState (state : ContestantState) (answer : Answer) (questionNumber : ℕ) : ContestantState :=
  match answer with
  | Answer.Correct =>
    let points := match questionNumber with
      | 1 => 1
      | 2 => 2
      | 3 => 3
      | 4 => 6
      | _ => 0
    { score := state.score + points, questions_answered := state.questions_answered + 1 }
  | Answer.Incorrect =>
    { score := state.score - 2, questions_answered := state.questions_answered + 1 }

/-- Checks if a contestant is eliminated based on their current state -/
def isEliminated (state : ContestantState) : Bool :=
  state.score < 8 || (state.questions_answered = 4 && state.score < 14)

/-- Checks if a contestant has advanced to the next round -/
def hasAdvanced (state : ContestantState) : Bool :=
  state.score ≥ 14

/-- Main theorem statement -/
theorem knowledge_competition_probabilities 
  (probs : QuestionProbabilities)
  (h1 : probs.pA = 3/4)
  (h2 : probs.pB = 1/2)
  (h3 : probs.pC = 1/3)
  (h4 : probs.pD = 1/4) :
  ∃ (advanceProb : ℚ) (ξDist : ℕ → ℚ) (ξExpected : ℚ),
    (advanceProb = 1/2) ∧ 
    (ξDist 2 = 1/8) ∧ (ξDist 3 = 1/2) ∧ (ξDist 4 = 3/8) ∧
    (ξExpected = 7/4) := by
  sorry

end NUMINAMATH_CALUDE_knowledge_competition_probabilities_l863_86394


namespace NUMINAMATH_CALUDE_oranges_per_sack_l863_86386

/-- Proves that the number of oranges per sack is 50, given the harvest conditions --/
theorem oranges_per_sack (total_sacks : ℕ) (discarded_sacks : ℕ) (total_oranges : ℕ)
  (h1 : total_sacks = 76)
  (h2 : discarded_sacks = 64)
  (h3 : total_oranges = 600) :
  total_oranges / (total_sacks - discarded_sacks) = 50 := by
  sorry

#check oranges_per_sack

end NUMINAMATH_CALUDE_oranges_per_sack_l863_86386


namespace NUMINAMATH_CALUDE_fruit_eating_arrangements_l863_86392

theorem fruit_eating_arrangements : 
  let total_fruits : ℕ := 4 + 2 + 1
  let apples : ℕ := 4
  let oranges : ℕ := 2
  let bananas : ℕ := 1
  (Nat.factorial total_fruits) / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas) = 105 := by
  sorry

end NUMINAMATH_CALUDE_fruit_eating_arrangements_l863_86392


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l863_86340

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given that the man is 46 years older than his son and the son's current age is 44. -/
theorem mans_age_to_sons_age_ratio :
  ∀ (son_age man_age : ℕ),
    son_age = 44 →
    man_age = son_age + 46 →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l863_86340


namespace NUMINAMATH_CALUDE_sum_of_even_is_even_l863_86381

theorem sum_of_even_is_even (a b : ℤ) (ha : Even a) (hb : Even b) : Even (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_is_even_l863_86381


namespace NUMINAMATH_CALUDE_max_container_weight_is_26_l863_86352

/-- Represents a warehouse with a total weight of goods and a maximum container weight. -/
structure Warehouse where
  totalWeight : ℕ
  maxContainerWeight : ℕ

/-- Represents a train with a number of platforms and the capacity of each platform. -/
structure Train where
  numPlatforms : ℕ
  platformCapacity : ℕ

/-- Checks if a given warehouse's goods can be transported by a given train. -/
def canTransport (w : Warehouse) (t : Train) : Prop :=
  ∀ (containerWeights : List ℕ),
    (containerWeights.sum = w.totalWeight) →
    (∀ weight ∈ containerWeights, 0 < weight ∧ weight ≤ w.maxContainerWeight) →
    ∃ (platformLoads : List ℕ),
      (platformLoads.length = t.numPlatforms) ∧
      (platformLoads.sum = w.totalWeight) ∧
      (∀ load ∈ platformLoads, load ≤ t.platformCapacity)

/-- The main theorem stating that 26 is the maximum container weight that guarantees transport. -/
theorem max_container_weight_is_26 :
  let w : Warehouse := ⟨1500, 26⟩
  let t : Train := ⟨25, 80⟩
  (canTransport w t) ∧ ¬(canTransport ⟨1500, 27⟩ t) :=
by sorry

end NUMINAMATH_CALUDE_max_container_weight_is_26_l863_86352


namespace NUMINAMATH_CALUDE_line_direction_vector_l863_86312

def point_a : ℝ × ℝ := (-3, 1)
def point_b : ℝ × ℝ := (2, 5)

def direction_vector (b : ℝ) : ℝ × ℝ := (1, b)

theorem line_direction_vector : 
  ∃ (b : ℝ), (point_b.1 - point_a.1, point_b.2 - point_a.2) = 
    (point_b.1 - point_a.1) • direction_vector b ∧ b = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l863_86312


namespace NUMINAMATH_CALUDE_coin_sum_problem_l863_86374

/-- Calculates the total sum of money in rupees given the number of 20 paise and 25 paise coins. -/
def total_sum_in_rupees (coins_20_paise : ℕ) (coins_25_paise : ℕ) : ℚ :=
  (coins_20_paise * 20 + coins_25_paise * 25) / 100

/-- Proves that given 336 total coins, with 260 coins of 20 paise and the rest being 25 paise coins, 
    the total sum of money is 71 rupees. -/
theorem coin_sum_problem (total_coins : ℕ) (coins_20_paise : ℕ) 
  (h1 : total_coins = 336)
  (h2 : coins_20_paise = 260)
  (h3 : total_coins = coins_20_paise + (total_coins - coins_20_paise)) :
  total_sum_in_rupees coins_20_paise (total_coins - coins_20_paise) = 71 := by
  sorry

#eval total_sum_in_rupees 260 76

end NUMINAMATH_CALUDE_coin_sum_problem_l863_86374


namespace NUMINAMATH_CALUDE_calculate_expression_l863_86323

theorem calculate_expression : -5^2 - (-3)^3 * (2/9) - 9 * |-(2/3)| = -25 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l863_86323


namespace NUMINAMATH_CALUDE_soda_cost_l863_86309

theorem soda_cost (burger_cost soda_cost : ℕ) : 
  (3 * burger_cost + 2 * soda_cost = 450) →
  (2 * burger_cost + 3 * soda_cost = 480) →
  soda_cost = 108 := by
sorry

end NUMINAMATH_CALUDE_soda_cost_l863_86309


namespace NUMINAMATH_CALUDE_balance_difference_approx_l863_86331

def angela_deposit : ℝ := 9000
def bob_deposit : ℝ := 11000
def angela_rate : ℝ := 0.08
def bob_rate : ℝ := 0.09
def years : ℕ := 25

def angela_balance : ℝ := angela_deposit * (1 + angela_rate) ^ years
def bob_balance : ℝ := bob_deposit * (1 + bob_rate * years)

theorem balance_difference_approx :
  ‖angela_balance - bob_balance - 25890‖ < 1 := by sorry

end NUMINAMATH_CALUDE_balance_difference_approx_l863_86331


namespace NUMINAMATH_CALUDE_mother_carrots_count_l863_86324

/-- The number of carrots Haley picked -/
def haley_carrots : ℕ := 39

/-- The number of good carrots -/
def good_carrots : ℕ := 64

/-- The number of bad carrots -/
def bad_carrots : ℕ := 13

/-- The number of carrots Haley's mother picked -/
def mother_carrots : ℕ := (good_carrots + bad_carrots) - haley_carrots

theorem mother_carrots_count : mother_carrots = 38 := by
  sorry

end NUMINAMATH_CALUDE_mother_carrots_count_l863_86324


namespace NUMINAMATH_CALUDE_b_fourth_plus_inverse_l863_86387

theorem b_fourth_plus_inverse (b : ℝ) (h : (b + 1/b)^2 = 5) : b^4 + 1/b^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_b_fourth_plus_inverse_l863_86387


namespace NUMINAMATH_CALUDE_intersection_plane_sphere_sum_l863_86384

/-- Given a plane x + 2y + 3z = 6 that passes through a point (a, b, c) and intersects
    the coordinate axes at points A, B, C distinct from the origin O,
    prove that a/p + b/q + c/r = 2, where (p, q, r) is the center of the sphere
    passing through A, B, C, and O. -/
theorem intersection_plane_sphere_sum (a b c p q r : ℝ) : 
  (∃ (x y z : ℝ), x + 2*y + 3*z = 6 ∧ 
                   a + 2*b + 3*c = 6 ∧
                   (x = 0 ∨ y = 0 ∨ z = 0) ∧
                   (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) →
  (p^2 + q^2 + r^2 = (p - 6)^2 + q^2 + r^2 ∧
   p^2 + q^2 + r^2 = p^2 + (q - 3)^2 + r^2 ∧
   p^2 + q^2 + r^2 = p^2 + q^2 + (r - 2)^2) →
  a/p + b/q + c/r = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_plane_sphere_sum_l863_86384


namespace NUMINAMATH_CALUDE_rectangle_cut_squares_l863_86368

/-- Given a rectangle with length 90 cm and width 42 cm, prove that when cut into the largest possible squares with integer side lengths, the minimum number of squares is 105 and their total perimeter is 2520 cm. -/
theorem rectangle_cut_squares (length width : ℕ) (h1 : length = 90) (h2 : width = 42) :
  let side_length := Nat.gcd length width
  let num_squares := (length / side_length) * (width / side_length)
  let total_perimeter := num_squares * (4 * side_length)
  num_squares = 105 ∧ total_perimeter = 2520 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_cut_squares_l863_86368


namespace NUMINAMATH_CALUDE_factorization_of_four_minus_n_squared_l863_86339

theorem factorization_of_four_minus_n_squared (n : ℝ) : 4 - n^2 = (2 + n) * (2 - n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_four_minus_n_squared_l863_86339


namespace NUMINAMATH_CALUDE_bernardo_silvia_game_sum_of_digits_l863_86345

theorem bernardo_silvia_game (N : ℕ) : N = 38 ↔ 
  (27 * N + 900 < 2000) ∧ 
  (27 * N + 900 ≥ 1925) ∧ 
  (∀ k : ℕ, k < N → (27 * k + 900 < 1925 ∨ 27 * k + 900 ≥ 2000)) :=
sorry

theorem sum_of_digits (N : ℕ) : N = 38 → (N % 10 + N / 10) = 11 :=
sorry

end NUMINAMATH_CALUDE_bernardo_silvia_game_sum_of_digits_l863_86345


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l863_86322

-- Define the vectors
def a : Fin 2 → ℝ := ![(-1 : ℝ), 3]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 2]

-- Define the dot product
def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

-- Define perpendicularity
def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  dot_product u v = 0

-- The main theorem
theorem perpendicular_vectors_x_value :
  ∃ x : ℝ, perpendicular (a + b x) a ∧ x = 16 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l863_86322


namespace NUMINAMATH_CALUDE_tan_periodic_equality_l863_86398

theorem tan_periodic_equality (m : ℤ) : 
  -180 < m ∧ m < 180 ∧ Real.tan (m * π / 180) = Real.tan (1500 * π / 180) → m = 60 := by
  sorry

end NUMINAMATH_CALUDE_tan_periodic_equality_l863_86398


namespace NUMINAMATH_CALUDE_f_properties_f_inv_property_l863_86300

/-- A function f(x) that is directly proportional to x-3 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x - 3 * k

/-- The theorem stating the properties of f -/
theorem f_properties (k : ℝ) :
  f k 4 = 3 →
  (∀ x, f k x = 3 * x - 9) ∧
  (∃ x, f k x = -12 ∧ x = -1) := by
  sorry

/-- The inverse function of f -/
noncomputable def f_inv (k : ℝ) (y : ℝ) : ℝ := (y + 3 * k) / k

/-- Theorem stating that f_inv(-12) = -1 when f(4) = 3 -/
theorem f_inv_property (k : ℝ) :
  f k 4 = 3 →
  f_inv k (-12) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_properties_f_inv_property_l863_86300


namespace NUMINAMATH_CALUDE_abc_acute_angle_implies_m_values_l863_86370

def OA : Fin 2 → ℝ := ![3, -4]
def OB : Fin 2 → ℝ := ![6, -3]
def OC (m : ℝ) : Fin 2 → ℝ := ![5 - m, -3 - m]

def BA : Fin 2 → ℝ := ![OA 0 - OB 0, OA 1 - OB 1]
def BC (m : ℝ) : Fin 2 → ℝ := ![OC m 0 - OB 0, OC m 1 - OB 1]

def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1)

def is_acute_angle (m : ℝ) : Prop := dot_product BA (BC m) > 0

theorem abc_acute_angle_implies_m_values :
  ∀ m : ℝ, is_acute_angle m → (m = 0 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_abc_acute_angle_implies_m_values_l863_86370


namespace NUMINAMATH_CALUDE_nabla_calculation_l863_86363

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem nabla_calculation : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l863_86363


namespace NUMINAMATH_CALUDE_even_count_pascal_15_rows_l863_86347

/-- Counts the number of even entries in a single row of Pascal's Triangle -/
def countEvenInRow (n : ℕ) : ℕ := sorry

/-- Counts the total number of even entries in the first n rows of Pascal's Triangle -/
def countEvenInTriangle (n : ℕ) : ℕ := sorry

/-- The number of even integers in the first 15 rows of Pascal's Triangle is 97 -/
theorem even_count_pascal_15_rows : countEvenInTriangle 15 = 97 := by sorry

end NUMINAMATH_CALUDE_even_count_pascal_15_rows_l863_86347


namespace NUMINAMATH_CALUDE_multiplication_value_proof_l863_86333

theorem multiplication_value_proof (x : ℚ) : (3 / 4) * x = 9 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_value_proof_l863_86333


namespace NUMINAMATH_CALUDE_bill_amount_calculation_l863_86305

/-- The amount of a bill given its true discount and banker's discount -/
def bill_amount (true_discount : ℚ) (bankers_discount : ℚ) : ℚ :=
  true_discount + true_discount

/-- Theorem stating that given a true discount of 360 and a banker's discount of 424.8, 
    the amount of the bill is 720 -/
theorem bill_amount_calculation :
  bill_amount 360 424.8 = 720 := by
  sorry

end NUMINAMATH_CALUDE_bill_amount_calculation_l863_86305


namespace NUMINAMATH_CALUDE_M_factor_count_l863_86321

def M : ℕ := 2^6 * 3^5 * 5^3 * 7^4 * 11^1

def count_factors (n : ℕ) : ℕ := sorry

theorem M_factor_count : count_factors M = 1680 := by sorry

end NUMINAMATH_CALUDE_M_factor_count_l863_86321


namespace NUMINAMATH_CALUDE_arrangement_count_is_2880_l863_86366

/-- The number of ways to choose k items from n items without replacement and where order matters. -/
def permutations (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def combinations (n k : ℕ) : ℕ := sorry

/-- The number of different arrangements of 4 boys and 3 girls in a row, where exactly 2 of the 3 girls are adjacent. -/
def arrangement_count : ℕ := sorry

theorem arrangement_count_is_2880 : arrangement_count = 2880 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_2880_l863_86366


namespace NUMINAMATH_CALUDE_sqrt_three_divided_by_sum_l863_86397

theorem sqrt_three_divided_by_sum : 
  Real.sqrt 3 / (Real.sqrt (1/3) + Real.sqrt (3/16)) = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_divided_by_sum_l863_86397
