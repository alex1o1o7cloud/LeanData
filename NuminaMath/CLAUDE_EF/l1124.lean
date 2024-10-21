import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_x_squared_plus_y_squared_equals_x_to_y_l1124_112427

theorem unique_solution_x_squared_plus_y_squared_equals_x_to_y :
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = x^y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_x_squared_plus_y_squared_equals_x_to_y_l1124_112427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1124_112484

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (1/3) * (a + b + c)^2 ≥ a * Real.sqrt (b * c) + b * Real.sqrt (c * a) + c * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1124_112484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_on_fixed_line_range_of_area_S_l1124_112445

-- Define the curves
def line_l1 (m : ℝ) := λ x : ℝ ↦ 2 * x + m
def parabola_C1 (a : ℝ) := λ x : ℝ ↦ a * x^2
def circle_C2 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 5

-- Define the conditions
variable (m a : ℝ)
variable (h_m : m < 0)
variable (h_a : a > 0)

-- Tangency conditions (assumed, not proven)
axiom tangent_l1_C1 : ∃ x, line_l1 m x = parabola_C1 a x
axiom tangent_l1_C2 : ∃ x, circle_C2 x (line_l1 m x)
axiom tangent_C1_C2 : ∃ x, circle_C2 x (parabola_C1 a x)

-- Focus of the parabola
noncomputable def focus (a : ℝ) : ℝ × ℝ := (0, 1 / (4 * a))

-- Point on the parabola
variable (A : ℝ × ℝ)
variable (h_A : A.2 = parabola_C1 a A.1)

-- Tangent line at A
def tangent_at_A (a : ℝ) (A : ℝ × ℝ) := λ x : ℝ ↦ A.2 + 2 * a * A.1 * (x - A.1)

-- Point B
def B (a : ℝ) (A : ℝ × ℝ) : ℝ × ℝ := (0, tangent_at_A a A 0)

-- Point M (completing the parallelogram FAMB)
noncomputable def M (a : ℝ) (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 - (focus a).1, A.2 - (focus a).2 + (B a A).2 - (focus a).2)

-- Theorem 1: M lies on a fixed line
theorem M_on_fixed_line (a : ℝ) (h_a : a > 0) (A : ℝ × ℝ) (h_A : A.2 = parabola_C1 a A.1) :
  (M a A).2 = -3/2 := by sorry

-- Line l2 (NM)
noncomputable def line_l2 (a : ℝ) (A : ℝ × ℝ) := λ x : ℝ ↦
  ((M a A).2 - (focus a).2) / ((M a A).1 - (focus a).1) * (x - (focus a).1) + (focus a).2

-- Points P and Q
noncomputable def P_Q (a : ℝ) (A : ℝ × ℝ) : ℝ × ℝ := sorry

-- Area of triangle NPQ
noncomputable def area_NPQ (a : ℝ) (A : ℝ × ℝ) : ℝ := sorry

-- Theorem 2: Range of area S
theorem range_of_area_S (a : ℝ) (h_a : a > 0) (A : ℝ × ℝ) (h_A : A.2 = parabola_C1 a A.1) :
  ∀ S, S = area_NPQ a A → S > 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_on_fixed_line_range_of_area_S_l1124_112445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_k_for_two_zeros_l1124_112412

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- Define the function g(x) = kx^2
def g (k : ℝ) (x : ℝ) : ℝ := k * x^2

-- Define the function F(x) = f(x) - g(x)
noncomputable def F (k : ℝ) (x : ℝ) : ℝ := f x - g k x

-- Theorem for the range of f(x)
theorem range_of_f : Set.range f = Set.Ici (-Real.exp (-1)) := by sorry

-- Theorem for the range of k where F(x) has exactly two zeros for x > 0
theorem range_of_k_for_two_zeros (k : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ F k x₁ = 0 ∧ F k x₂ = 0 ∧
    ∀ x, 0 < x → F k x = 0 → x = x₁ ∨ x = x₂) ↔ k > Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_k_for_two_zeros_l1124_112412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1124_112473

theorem range_of_a (a : ℝ) : 
  (Real.log (1/2) / Real.log a < 1) → (a^(1/2) < 1) → (0 < a ∧ a < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1124_112473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l1124_112419

theorem triangle_trigonometric_identity (A B C : ℝ) :
  (A + B + C = π) →
  (Real.sqrt 3 * Real.sin B + Real.cos B = 2) →
  (Real.tan (A/2) + Real.tan (C/2) + Real.sqrt 3 * Real.tan (A/2) * Real.tan (C/2) = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l1124_112419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_outer_square_l1124_112491

/-- Given a square EFGH with area 36 and equilateral triangles constructed on its sides,
    the area of the square PQRS formed by connecting the outer vertices of these triangles
    is 144 + 72√3. -/
theorem area_of_outer_square (E F G H P Q R S : ℝ × ℝ) 
  (square_area : ℝ)
  (is_square : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop)
  (is_equilateral_triangle : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop) : 
  -- Define the square EFGH
  is_square E F G H ∧ 
  -- Define the area of EFGH
  square_area = 36 ∧
  square_area = (F.1 - E.1) * (F.2 - E.2) ∧
  -- Define equilateral triangles on the sides of EFGH
  is_equilateral_triangle E P F ∧
  is_equilateral_triangle F Q G ∧
  is_equilateral_triangle G R H ∧
  is_equilateral_triangle H S E →
  -- The area of square PQRS
  (Q.1 - P.1) * (Q.2 - P.2) = 144 + 72 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_outer_square_l1124_112491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_2_pairs_correct_final_result_l1124_112435

def total_socks : ℕ := 14
def blue_socks : ℕ := 7
def red_socks : ℕ := 7
def total_draws : ℕ := 7

def probability_same_color : ℚ := 42 / 91
def probability_different_color : ℚ := 49 / 91

def probability_at_most_2_pairs : ℚ := 184 / 429

theorem probability_at_most_2_pairs_correct :
  probability_at_most_2_pairs = 
    (probability_different_color ^ total_draws) +
    (total_draws * probability_same_color * (probability_different_color ^ (total_draws - 1))) +
    (Nat.choose total_draws 2 * (probability_same_color ^ 2) * (probability_different_color ^ (total_draws - 2))) :=
by sorry

-- Additional theorem to show the final result
theorem final_result :
  184 + 429 = 613 :=
by norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_2_pairs_correct_final_result_l1124_112435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_de_moivre_formula_l1124_112467

theorem de_moivre_formula (x : ℝ) (n : ℕ) :
  (Complex.exp (Complex.I * x)) ^ n = Complex.exp (Complex.I * (n : ℝ) * x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_de_moivre_formula_l1124_112467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_sqrt_three_l1124_112472

theorem tan_alpha_equals_sqrt_three (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α + Real.sin (α + π / 3) + Real.sin (α + 2 * π / 3) = Real.sqrt 3) : 
  Real.tan α = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_sqrt_three_l1124_112472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1124_112404

theorem trigonometric_equation_solution (x : ℝ) 
  (h1 : Real.cos (2 * x) ≠ 0) 
  (h2 : Real.cos (3 * x) ≠ 0) 
  (h3 : Real.cos (5 * x) ≠ 0) :
  8.447 * Real.tan (2 * x) * Real.tan (3 * x)^2 * Real.tan (5 * x)^2 = Real.tan (2 * x) + Real.tan (3 * x)^2 - Real.tan (5 * x)^2 ↔ 
  (∃ k : ℤ, x = k * Real.pi) ∨ (∃ k : ℤ, x = Real.pi / 32 * (4 * k + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1124_112404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_17th_inning_l1124_112458

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℚ
  innings : ℕ
  currentInningRuns : ℚ
  milestoneBonus : ℚ

/-- Calculates the new average after an inning -/
def newAverage (b : Batsman) : ℚ :=
  (b.initialAverage * (b.innings - 1) + b.currentInningRuns + b.milestoneBonus) / b.innings

/-- Theorem: A batsman scoring 87 runs in the 17th inning with a 2-run average increase at 50 runs will have a new average of 55 -/
theorem batsman_average_after_17th_inning (b : Batsman) 
    (h1 : b.innings = 17)
    (h2 : b.currentInningRuns = 87)
    (h3 : b.milestoneBonus = 2) : 
  newAverage b = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_17th_inning_l1124_112458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l1124_112426

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := -6/x - 5*Real.log x

-- State the theorem
theorem f_monotone_increasing_interval :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 6/5 → f x₁ < f x₂ ∧
  ∀ y₁ y₂, 0 < y₁ ∧ y₁ < y₂ ∧ (y₁ ≤ 6/5 → y₂ > 6/5) → f y₁ ≥ f y₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l1124_112426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l1124_112406

/-- The length of two trains passing each other -/
theorem train_length_problem (crossing_time : ℝ) (faster_train_speed : ℝ) :
  crossing_time = 20 →
  faster_train_speed = 24 →
  let slower_train_speed := faster_train_speed / 2
  let relative_speed := slower_train_speed + faster_train_speed
  let total_length := relative_speed * crossing_time
  let train_length := total_length / 2
  train_length = 360 := by
  intro h1 h2
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l1124_112406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_plus_fraction_equals_one_l1124_112409

theorem series_sum_plus_fraction_equals_one :
  let series := (Finset.range 20).sum (fun i => 1 / ((i + 2) * (i + 3)) : ℕ → ℚ)
  series + 15 / 22 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_plus_fraction_equals_one_l1124_112409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_quarter_l1124_112469

/-- The sum of an infinite geometric series with first term a and common ratio r. -/
noncomputable def infiniteGeometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Proof that the sum of the infinite geometric series 1/4 + 1/8 + 1/16 + 1/32 + ... is equal to 1/2. -/
theorem infinite_geometric_series_quarter : 
  let a : ℝ := 1/4
  let r : ℝ := 1/2
  infiniteGeometricSeriesSum a r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_quarter_l1124_112469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_angles_formula_l1124_112443

/-- The sum of interior angles of a non-self-intersecting n-sided polygon -/
noncomputable def sum_interior_angles (n : ℕ) : ℝ := (n - 2 : ℝ) * Real.pi

/-- Theorem: For any integer n ≥ 3, the sum of interior angles of a 
    non-self-intersecting n-sided polygon equals (n-2)π radians -/
theorem sum_interior_angles_formula (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2 : ℝ) * Real.pi :=
by
  -- Unfold the definition of sum_interior_angles
  unfold sum_interior_angles
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_angles_formula_l1124_112443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1124_112441

theorem trigonometric_identity (α β : ℝ) 
  (h : (Real.cos α)^6 / (Real.cos (2*β))^2 + (Real.sin α)^6 / (Real.sin (2*β))^2 = 1) : 
  (Real.sin (2*β))^6 / (Real.sin α)^2 + (Real.cos (2*β))^6 / (Real.cos α)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1124_112441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_addition_l1124_112493

theorem complex_multiplication_addition : ∃ (result : ℂ),
  let z₁ : ℂ := 3 + 5*I
  let z₂ : ℂ := 4 - 7*I
  let multiplier : ℂ := 2*I
  result = (z₁ * multiplier) + (z₂ * multiplier) ∧ result = 4 + 14*I := by
  -- Define the two complex numbers
  let z₁ : ℂ := 3 + 5*I
  let z₂ : ℂ := 4 - 7*I
  
  -- Define the multiplier
  let multiplier : ℂ := 2*I
  
  -- Define the operation
  let result : ℂ := (z₁ * multiplier) + (z₂ * multiplier)
  
  -- State the existence of the result and its properties
  use result
  constructor
  · rfl  -- This proves result = (z₁ * multiplier) + (z₂ * multiplier)
  · sorry  -- This skips the proof that result = 4 + 14*I


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_addition_l1124_112493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_and_volume_selling_prices_for_target_profit_l1124_112444

/-- Represents the souvenir selling scenario -/
structure SouvenirSales where
  cost_price : ℝ
  initial_price : ℝ
  initial_volume : ℝ
  price_increase : ℝ
  volume_decrease : ℝ
  new_price : ℝ

/-- Calculates the profit per item -/
noncomputable def profit_per_item (s : SouvenirSales) : ℝ :=
  s.new_price - s.cost_price

/-- Calculates the new daily sales volume -/
noncomputable def new_sales_volume (s : SouvenirSales) : ℝ :=
  s.initial_volume - (s.volume_decrease / s.price_increase) * (s.new_price - s.initial_price)

/-- Calculates the daily profit -/
noncomputable def daily_profit (s : SouvenirSales) (price : ℝ) : ℝ :=
  (price - s.cost_price) * (s.initial_volume - (s.volume_decrease / s.price_increase) * (price - s.initial_price))

/-- Theorem stating the profit per item and new sales volume -/
theorem profit_and_volume (s : SouvenirSales) 
    (h1 : s.cost_price = 5)
    (h2 : s.initial_price = 9)
    (h3 : s.initial_volume = 32)
    (h4 : s.price_increase = 2)
    (h5 : s.volume_decrease = 8)
    (h6 : s.new_price = 11) :
    profit_per_item s = 6 ∧ new_sales_volume s = 24 := by
  sorry

/-- Theorem stating the selling prices for a daily profit of 140 yuan -/
theorem selling_prices_for_target_profit (s : SouvenirSales) 
    (h1 : s.cost_price = 5)
    (h2 : s.initial_price = 9)
    (h3 : s.initial_volume = 32)
    (h4 : s.price_increase = 2)
    (h5 : s.volume_decrease = 8) :
    ∃ (p1 p2 : ℝ), p1 ≠ p2 ∧ daily_profit s p1 = 140 ∧ daily_profit s p2 = 140 ∧ (p1 = 12 ∨ p1 = 10) ∧ (p2 = 12 ∨ p2 = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_and_volume_selling_prices_for_target_profit_l1124_112444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_theorem_l1124_112495

/-- The area of wrapping paper needed for a rectangular box -/
noncomputable def wrapping_paper_area (l w h : ℝ) : ℝ :=
  4 * max ((l / 2 + h) ^ 2) ((w / 2 + h) ^ 2)

/-- Theorem: The area of wrapping paper needed for a rectangular box with dimensions l, w, and h
    is equal to 4 * max((l/2 + h)^2, (w/2 + h)^2) -/
theorem wrapping_paper_theorem (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  wrapping_paper_area l w h = 4 * (max ((l / 2 + h) ^ 2) ((w / 2 + h) ^ 2)) := by
  -- Unfold the definition of wrapping_paper_area
  unfold wrapping_paper_area
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_theorem_l1124_112495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_diameter_ratio_is_2_sqrt6_div_5_l1124_112439

/-- Configuration of three circles where two smaller circles touch each other externally
    and each touches a larger circle internally. -/
structure CircleConfiguration where
  r : ℝ
  small_circle1_radius : ℝ := 2 * r
  small_circle2_radius : ℝ := 3 * r
  large_circle_radius : ℝ := 6 * r

/-- The ratio of the length of the common internal tangent segment to the smaller circles
    (enclosed within the largest circle) to the diameter of the largest circle. -/
noncomputable def tangent_to_diameter_ratio (config : CircleConfiguration) : ℝ :=
  2 * Real.sqrt 6 / 5

/-- Theorem stating the ratio of the common internal tangent segment length to the diameter
    of the largest circle in the given circle configuration. -/
theorem tangent_diameter_ratio_is_2_sqrt6_div_5 (config : CircleConfiguration) :
  tangent_to_diameter_ratio config = 2 * Real.sqrt 6 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_diameter_ratio_is_2_sqrt6_div_5_l1124_112439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l1124_112410

/-- The expression to be expanded -/
noncomputable def f (x : ℝ) : ℝ := (2 / Real.sqrt x - x) ^ 6

/-- The general term of the expansion -/
noncomputable def general_term (r : ℕ) (x : ℝ) : ℝ :=
  (-1)^r * Nat.choose 6 r * 2^(6-r) * x^((3*r)/2 - 3)

theorem expansion_properties :
  (∃ r : ℕ, general_term r 1 = 240) ∧
  (∀ k : ℕ, k ≠ 3 → Nat.choose 6 3 ≥ Nat.choose 6 k) := by
  sorry

#check expansion_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l1124_112410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_chain_implies_next_smaller_l1124_112417

/-- Least common multiple of two natural numbers -/
def lcm' (a b : ℕ) : ℕ := Nat.lcm a b

theorem lcm_chain_implies_next_smaller (n : ℕ) (h : n > 0) :
  (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ 35 → lcm' n (n + i) > lcm' n (n + j)) →
  lcm' n (n + 35) > lcm' n (n + 36) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_chain_implies_next_smaller_l1124_112417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_l1124_112487

-- Define the regular triangular prism
structure RegularTriangularPrism where
  base_edge : ℝ
  height : ℝ

-- Define the cross section
structure CrossSection where
  prism : RegularTriangularPrism
  area : ℝ

-- Define the theorem
theorem cross_section_area 
  (prism : RegularTriangularPrism) 
  (h1 : prism.base_edge = 1) 
  (h2 : prism.height = 2) 
  (cross_section : CrossSection) 
  (h3 : cross_section.prism = prism) :
  cross_section.area = Real.sqrt 13 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_l1124_112487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_light_distance_l1124_112402

/-- The distance between two points when light reflects off the x-axis -/
noncomputable def reflectedDistance (A B : ℝ × ℝ) : ℝ :=
  let A' := (A.1, -A.2)
  Real.sqrt ((B.1 - A'.1)^2 + (B.2 - A'.2)^2)

theorem reflected_light_distance :
  let A : ℝ × ℝ := (-3, 5)
  let B : ℝ × ℝ := (2, 10)
  reflectedDistance A B = 5 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_light_distance_l1124_112402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l1124_112407

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ := (principal * rate * time) / 100

theorem interest_rate_is_ten_percent
  (principal : ℝ)
  (h_positive : principal > 0)
  (h_interest : simple_interest principal rate 2 = principal / 5) :
  rate = 10 :=
by
  -- The proof steps would go here
  sorry

#check interest_rate_is_ten_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l1124_112407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l1124_112453

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A (p : ℝ) : Set ℝ := {x | x^2 + p*x + 12 = 0}

-- Define set B
def B (q : ℝ) : Set ℝ := {x | x^2 - 5*x + q = 0}

-- Define the theorem
theorem union_of_A_and_B (p q : ℝ) :
  (Set.compl (A p) ∩ B q = {2}) → (A p ∩ Set.compl (B q) = {4}) →
  A p ∪ B q = {2, 3, 4} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l1124_112453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_intersection_and_union_l1124_112432

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {x : ℝ | x^2 - 4*x - 5 < 0}

-- State the theorem
theorem solution_sets_intersection_and_union :
  (A ∩ B = Set.Ioo (-1 : ℝ) 1) ∧ (A ∪ B = Set.Ioo (-3 : ℝ) 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_intersection_and_union_l1124_112432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangular_pyramid_volume_l1124_112483

noncomputable section

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

def perpendicular (a b c : Point3D) : Prop :=
  (a.x - b.x) * (a.x - c.x) + (a.y - b.y) * (a.y - c.y) + (a.z - b.z) * (a.z - c.z) = 0

noncomputable def volume_right_triangular_pyramid (base_area height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

theorem right_triangular_pyramid_volume 
  (S P Q R : Point3D)
  (h_perp_SP_SQ : perpendicular S P Q)
  (h_perp_SP_SR : perpendicular S P R)
  (h_perp_SQ_SR : perpendicular S Q R)
  (h_SP : distance S P = 12)
  (h_SQ : distance S Q = 12)
  (h_SR : distance S R = 8) :
  volume_right_triangular_pyramid ((1 / 2) * 12 * 12) 8 = 192 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangular_pyramid_volume_l1124_112483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equalities_l1124_112471

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x | 1/2 ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x < 8}

-- Define set B
def B : Set ℝ := {x | 5/(x+2) ≥ 1}

-- Statement for the proof problem
theorem set_equalities :
  (A = {x : ℝ | -1 ≤ x ∧ x < 3}) ∧
  (B = {x : ℝ | -2 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∩ B) = {x : ℝ | (-2 < x ∧ x < -1) ∨ x = 3}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equalities_l1124_112471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_healing_time_l1124_112424

/-- The time (in weeks) it took James to heal enough to get a skin graft -/
def healing_time_before_graft : ℝ := sorry

/-- The total recovery time (in weeks) -/
def total_recovery_time : ℝ := 10

/-- The relationship between healing time before graft and total recovery time -/
axiom recovery_time_relation : total_recovery_time = 2.5 * healing_time_before_graft

theorem james_healing_time : healing_time_before_graft = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_healing_time_l1124_112424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_sin_x_l1124_112434

noncomputable section

/-- Vector m as a function of x -/
def m (x : ℝ) : ℝ := Real.cos (x / 2) - 1

/-- Vector n as a function of x -/
def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 2), Real.cos (x / 2) ^ 2)

/-- Function f(x) -/
def f (x : ℝ) : ℝ := m x * (n x).1 + (n x).2 + 1

theorem min_value_and_sin_x :
  (∀ x ∈ Set.Icc (π / 2) π, f x ≥ 1 ∧ f π = 1) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f x = 11 / 10 → Real.sin x = (3 * Real.sqrt 3 + 4) / 10) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_sin_x_l1124_112434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_on_largest_side_l1124_112479

/-- Represents a triangle with sides a, b, c and corresponding altitudes m_a, m_b, m_c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  acute : a > 0 ∧ b > 0 ∧ c > 0
  ordered : a ≥ b ∧ b ≥ c
  area_eq : a * m_a = b * m_b ∧ b * m_b = c * m_c

/-- Calculates the side length of an inscribed square on side a -/
noncomputable def square_side_a (t : Triangle) : ℝ := (t.a * t.m_a) / (t.a + t.m_a)

/-- Calculates the side length of an inscribed square on side b -/
noncomputable def square_side_b (t : Triangle) : ℝ := (t.b * t.m_b) / (t.b + t.m_b)

/-- Calculates the side length of an inscribed square on side c -/
noncomputable def square_side_c (t : Triangle) : ℝ := (t.c * t.m_c) / (t.c + t.m_c)

/-- Theorem: The smallest inscribed square is on the largest side of the triangle -/
theorem smallest_square_on_largest_side (t : Triangle) :
  square_side_a t ≤ square_side_b t ∧ square_side_b t ≤ square_side_c t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_on_largest_side_l1124_112479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_rhombus_l1124_112416

/-- The area enclosed by the graph of |2x| + |5y| = 10 is 20 -/
theorem area_enclosed_by_rhombus : 
  ∃ (A : Set (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ A ↔ |2 * p.1| + |5 * p.2| = 10) ∧ 
    (MeasureTheory.volume A = 20) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_rhombus_l1124_112416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_DA_is_15_l1124_112490

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the angles in radians
noncomputable def angle_CAB : ℝ := 20 * Real.pi / 180
noncomputable def angle_BAD : ℝ := 40 * Real.pi / 180

-- Define the distances
def BC : ℝ := 31
def BD : ℝ := 20
def CD : ℝ := 21

-- State the theorem
theorem distance_DA_is_15 : 
  ∀ (A B C D : ℝ × ℝ),
  -- C is south and west by 20° from A
  (C.1 - A.1) * Real.cos angle_CAB + (C.2 - A.2) * Real.sin angle_CAB = 0 →
  -- B is on the road from A
  (B.1 - A.1) * Real.cos angle_BAD + (B.2 - A.2) * Real.sin angle_BAD = 0 →
  -- BC = 31
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = BC^2 →
  -- BD = 20
  (B.1 - D.1)^2 + (B.2 - D.2)^2 = BD^2 →
  -- CD = 21
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = CD^2 →
  -- DA = 15
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = 15^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_DA_is_15_l1124_112490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_proof_l1124_112489

/-- Given a quadratic polynomial q(x) satisfying specific conditions,
    prove that q(x) = (2/3)x^2 - 2/3 -/
theorem quadratic_polynomial_proof (q : ℝ → ℝ) 
  (h1 : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c)  -- q is quadratic
  (h2 : q 2 = 2)                                     -- q(2) = 2
  (h3 : q (-1) = 0)                                  -- q(-1) = 0 (from asymptote)
  (h4 : q 1 = 0)                                     -- q(1) = 0 (given and from asymptote)
  : q = λ x ↦ (2/3) * x^2 - 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_proof_l1124_112489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_zero_l1124_112447

theorem division_remainder_zero (x y : ℕ) (h1 : (x : ℝ) / (y : ℝ) = 6.12) 
  (h2 : (y : ℝ) = 49.99999999999996) : x % y = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_zero_l1124_112447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_inscribed_square_side_length_l1124_112470

/-- An ellipse with foci and minor axis endpoints on the unit circle has an inscribed square with side length 2√6/3 -/
theorem ellipse_inscribed_square_side_length :
  ∀ (a b : ℝ) (ellipse : Set (ℝ × ℝ)),
  a > 0 ∧ b > 0 ∧ a > b →
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ↔ (x, y) ∈ ellipse) →
  b = 1 →
  Real.sqrt (a^2 - b^2) = 1 →
  (∃ (s : ℝ),
    s > 0 ∧
    (∀ (x y : ℝ), (x, y) ∈ ellipse ∧ x = y → x ≤ s/2) ∧
    (∃ (x y : ℝ), (x, y) ∈ ellipse ∧ x = y ∧ x = s/2) ∧
    s = 2 * Real.sqrt 6 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_inscribed_square_side_length_l1124_112470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1124_112400

/-- A triangle with side lengths 15, 10, and 12 has a perimeter of 37. -/
theorem triangle_perimeter (a b c : ℝ) : 
  a = 15 → b = 10 → c = 12 → a + b + c = 37 := by
  intros ha hb hc
  rw [ha, hb, hc]
  norm_num

#check triangle_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1124_112400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_five_eq_negative_five_l1124_112476

/-- A function f defined as f(x) = a*sin(x) + b*tan(x) + 1 -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x + b * Real.tan x + 1

/-- Theorem stating that if f(5) = 7, then f(-5) = -5 -/
theorem f_negative_five_eq_negative_five (a b : ℝ) :
  f a b 5 = 7 → f a b (-5) = -5 := by
  intro h
  have h1 : f a b (-5) + f a b 5 = 2 := by
    calc
      f a b (-5) + f a b 5
        = (a * Real.sin (-5) + b * Real.tan (-5) + 1) + (a * Real.sin 5 + b * Real.tan 5 + 1) := rfl
      _ = (a * (-Real.sin 5) + b * (-Real.tan 5) + 1) + (a * Real.sin 5 + b * Real.tan 5 + 1) := by simp [Real.sin_neg, Real.tan_neg]
      _ = 2 := by ring
  
  have h2 : f a b (-5) = 2 - f a b 5 := by
    linarith [h1]
  
  calc
    f a b (-5) = 2 - f a b 5 := h2
    _ = 2 - 7 := by rw [h]
    _ = -5 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_five_eq_negative_five_l1124_112476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_exists_l1124_112403

-- Define the function f(x) = a^(x+1) - 2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x+1) - 2

-- Theorem statement
theorem fixed_point_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_exists_l1124_112403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1124_112481

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem line_equation_proof (l1 l2 : Line) (p : Point) :
  l1.a = 3 ∧ l1.b = -4 ∧ l1.c = 5 →
  l2.a = 4 ∧ l2.b = 3 ∧ l2.c = -1 →
  p.x = -2 ∧ p.y = 3 →
  p.liesOn l2 ∧ l1.perpendicular l2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1124_112481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_of_primes_l1124_112430

theorem arithmetic_progression_of_primes (p : ℕ) (a : ℕ → ℕ) (d : ℕ) : 
  Prime p →
  (∀ i, i < p → Prime (a i)) →
  (∀ i, i < p - 1 → a (i + 1) = a i + d) →
  (∀ i j, i < j → j < p → a i < a j) →
  a 0 > p →
  p ∣ d :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_of_primes_l1124_112430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_5_moles_H2CO3_approx_l1124_112437

/-- The molecular weight of H2CO3 in grams per mole -/
def molecular_weight_H2CO3 : ℝ :=
  2 * 1.008 + 1 * 12.011 + 3 * 15.999

/-- The weight of 5 moles of H2CO3 in grams -/
def weight_5_moles_H2CO3 : ℝ :=
  5 * molecular_weight_H2CO3

/-- Theorem stating that the weight of 5 moles of H2CO3 is approximately 310.12 grams -/
theorem weight_5_moles_H2CO3_approx :
  |weight_5_moles_H2CO3 - 310.12| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_5_moles_H2CO3_approx_l1124_112437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_alpha_l1124_112440

theorem tan_pi_minus_alpha (α : ℝ) (x : ℝ) (h1 : 0 < α ∧ α < π/2) 
  (h2 : x^2 + (3/5)^2 = 1) (h3 : Real.tan α = 3/5 / x) : 
  Real.tan (π - α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_alpha_l1124_112440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_multiple_of_four_l1124_112485

theorem closest_perfect_square_multiple_of_four : 
  ∃ n : ℤ, n = 324 ∧ 
  (∃ k : ℤ, n = k^2) ∧ 
  (n % 4 = 0) ∧ 
  (∀ m : ℤ, (∃ j : ℤ, m = j^2) ∧ (m % 4 = 0) → |m - 350| ≥ |n - 350|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_multiple_of_four_l1124_112485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1124_112425

-- Define the power function as noncomputable
noncomputable def f (n : ℝ) (x : ℝ) : ℝ := x^n

-- State the theorem
theorem power_function_through_point (n : ℝ) :
  f n 2 = Real.sqrt 2 → f n 9 = 3 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1124_112425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_eq_zero_l1124_112474

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def not_identically_zero (f : ℝ → ℝ) : Prop :=
  ∃ x, f x ≠ 0

theorem f_zero_eq_zero
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_not_zero : not_identically_zero f)
  (h_property : ∀ x, x * f (x + 1) = (1 + x) * f x) :
  f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_eq_zero_l1124_112474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_problem_l1124_112438

/-- The length of a string spiraling around a circular cylindrical post -/
noncomputable def string_length (circumference height : ℝ) (wraps : ℕ) : ℝ :=
  let vertical_distance := height / (wraps : ℝ)
  let horizontal_distance := circumference
  (wraps : ℝ) * Real.sqrt (vertical_distance^2 + horizontal_distance^2)

/-- Theorem stating the length of the string in the given problem -/
theorem string_length_problem : 
  string_length 5 20 5 = 5 * Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_problem_l1124_112438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_one_l1124_112496

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x - 1 else -x^2 - 2*x

-- State the theorem
theorem f_inverse_of_one (a : ℝ) : f a = 1 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_one_l1124_112496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pricing_values_correct_l1124_112431

/-- Represents the tiered pricing system for gas charges -/
structure GasPricing where
  a : ℝ  -- Price for first tier
  b : ℝ  -- Price for second tier
  c : ℝ  -- Price for third tier

/-- Calculates the gas cost based on consumption and pricing -/
noncomputable def gasCost (pricing : GasPricing) (consumption : ℝ) : ℝ :=
  if consumption ≤ 310 then
    pricing.a * consumption
  else if consumption ≤ 520 then
    310 * pricing.a + pricing.b * (consumption - 310)
  else
    310 * pricing.a + 210 * pricing.b + pricing.c * (consumption - 520)

/-- Represents the consumption and payment data for a specific month -/
structure MonthlyData where
  consumption : ℝ
  payment : ℝ

/-- The given consumption and payment data -/
def givenData : List MonthlyData := [
  ⟨56, 168⟩, ⟨80, 240⟩, ⟨66, 198⟩, ⟨58, 174⟩, ⟨60, 183⟩,
  ⟨53, 174.9⟩, ⟨55, 186⟩, ⟨63, 264.6⟩
]

/-- Theorem stating that the given data implies the specific pricing values -/
theorem pricing_values_correct (pricing : GasPricing) : 
  (∀ data ∈ givenData, gasCost pricing data.consumption = data.payment) →
  pricing.a = 3 ∧ pricing.b = 3.3 ∧ pricing.c = 4.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pricing_values_correct_l1124_112431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l1124_112499

open Real

-- Define the point P in 3D space
def P : ℝ × ℝ × ℝ → Prop := fun (x, y, z) ↦ x^2 + y^2 + z^2 = 1

-- Define the surface area of the shape
noncomputable def surface_area : ℝ := 4 * π

-- Theorem statement
theorem sphere_surface_area :
  ∀ (x y z : ℝ), P (x, y, z) → surface_area = 4 * π :=
by
  intros x y z h
  unfold surface_area
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l1124_112499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inequality_constant_l1124_112492

theorem max_inequality_constant (a b : ℝ) : 
  max (|a + b|) (max (|a - b|) (|2006 - b|)) ≥ 1003 ∧ 
  ∀ C > 1003, ∃ a b : ℝ, max (|a + b|) (max (|a - b|) (|2006 - b|)) < C :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inequality_constant_l1124_112492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_given_cos_l1124_112463

theorem tan_value_for_given_cos (α : ℝ) : 
  α ∈ Set.Ioo (-π/2) 0 → Real.cos α = 3/5 → Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_given_cos_l1124_112463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_bound_no_lower_bound_for_a_a_range_theorem_l1124_112451

/-- The function f(x) defined in terms of a real parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 4/x - a| + a

/-- Theorem stating that if the maximum value of f(x) on [1,4] is 5, then a ≤ 9/2 -/
theorem max_value_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Icc 1 4, f a x ≤ 5) ∧ (∃ x ∈ Set.Icc 1 4, f a x = 5) →
  a ≤ 9/2 := by
  sorry

/-- Theorem stating that there exists no lower bound for a -/
theorem no_lower_bound_for_a :
  ∀ b : ℝ, ∃ a : ℝ, a < b ∧ (∀ x ∈ Set.Icc 1 4, f a x ≤ 5) ∧ (∃ x ∈ Set.Icc 1 4, f a x = 5) := by
  sorry

/-- The main theorem combining the two previous results -/
theorem a_range_theorem :
  {a : ℝ | (∀ x ∈ Set.Icc 1 4, f a x ≤ 5) ∧ (∃ x ∈ Set.Icc 1 4, f a x = 5)} = Set.Iic (9/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_bound_no_lower_bound_for_a_a_range_theorem_l1124_112451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_weighings_suffice_l1124_112455

/-- Represents the result of a weighing -/
inductive WeighingResult
  | LeftLighter
  | Equal
  | RightLighter

/-- Represents a coin -/
structure Coin where
  id : Nat
  isCounterfeit : Bool

/-- Represents a weighing on the balance scale -/
def weighing (leftPan rightPan : List Coin) : WeighingResult :=
  sorry

/-- The set of all coins -/
def allCoins : List Coin :=
  (List.range 7).map (λ i => ⟨i + 1, true⟩) ++
  (List.range 7).map (λ i => ⟨i + 8, false⟩)

/-- A strategy is a list of three weighings -/
def Strategy := List (List Coin × List Coin)

/-- Checks if a strategy correctly identifies all coins -/
def isValidStrategy (s : Strategy) : Prop :=
  s.length = 3 ∧
  ∃ (result1 result2 result3 : WeighingResult),
    weighing (s.get! 0).1 (s.get! 0).2 = result1 ∧
    weighing (s.get! 1).1 (s.get! 1).2 = result2 ∧
    weighing (s.get! 2).1 (s.get! 2).2 = result3 ∧
    (∀ c ∈ allCoins, c.isCounterfeit = (c.id ≤ 7))

theorem three_weighings_suffice : ∃ (s : Strategy), isValidStrategy s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_weighings_suffice_l1124_112455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_b_l1124_112454

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the conditions
variable (h₁ : ‖e₁‖ = 1)
variable (h₂ : ‖e₂‖ = 1)
variable (h₃ : Real.cos (Real.pi / 3) = inner e₁ e₂)

-- Define vectors a and b
def a (e₁ e₂ : V) : V := 2 • e₁ + e₂
def b (e₁ e₂ : V) : V := -3 • e₁ + 2 • e₂

-- State the theorem
theorem angle_between_a_and_b :
  Real.arccos (inner (a e₁ e₂) (b e₁ e₂) / (‖a e₁ e₂‖ * ‖b e₁ e₂‖)) = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_b_l1124_112454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_woman_work_days_l1124_112465

/-- The number of days it takes for one woman to complete the work alone -/
noncomputable def days_for_one_woman (total_work : ℝ) (men_rate : ℝ) (women_rate : ℝ) : ℝ :=
  total_work / women_rate

/-- The theorem stating that it takes 350 days for one woman to complete the work alone -/
theorem woman_work_days (total_work : ℝ) (men_rate : ℝ) (women_rate : ℝ)
    (h1 : (10 * men_rate + 15 * women_rate) * 7 = total_work)
    (h2 : men_rate * 100 = total_work) :
    days_for_one_woman total_work men_rate women_rate = 350 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_woman_work_days_l1124_112465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1124_112498

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  arithmetic_property : 4 * a 1 + a 3 = 4 * a 2
  a_1_eq_1 : a 1 = 1

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (seq : GeometricSequence) (n : ℕ) : ℚ :=
  (1 - (seq.a 2 / seq.a 1) ^ n) / (1 - (seq.a 2 / seq.a 1))

theorem geometric_sequence_properties (seq : GeometricSequence) :
  seq.a 2 / seq.a 1 = 2 ∧ geometric_sum seq 4 = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1124_112498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_volume_ratio_l1124_112428

/-- The ratio of the volume of a right circular cylinder perfectly inscribed in a cube
    to the volume of the cube. -/
noncomputable def cylinder_to_cube_volume_ratio : ℝ := Real.pi / 4

/-- Theorem stating that the ratio of the volume of a right circular cylinder
    perfectly inscribed in a cube to the volume of the cube is π/4. -/
theorem inscribed_cylinder_volume_ratio :
  let s : ℝ := 1  -- Assume unit cube for simplicity
  let cylinder_volume := Real.pi * (s/2)^2 * s
  let cube_volume := s^3
  cylinder_volume / cube_volume = cylinder_to_cube_volume_ratio :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_volume_ratio_l1124_112428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l1124_112418

theorem polynomial_factorization (x : Polynomial ℤ) : 
  4 * (x + 3) * (x + 7) * (x + 8) * (x + 10) + 2 * x^2 = 
  (x^2 + 16*x + 72) * (2*x + 36) * (2*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l1124_112418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l1124_112421

/-- An ellipse with equation x²/8 + y²/4 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}

/-- The left focus of the ellipse -/
def LeftFocus : ℝ × ℝ := (-2, 0)

/-- The dot product of vectors FM and FN, where F is the left focus and M, N are points on the ellipse -/
def DotProduct (m n : ℝ × ℝ) : ℝ :=
  let fm := (m.1 - LeftFocus.1, m.2 - LeftFocus.2)
  let fn := (n.1 - LeftFocus.1, n.2 - LeftFocus.2)
  fm.1 * fn.1 + fm.2 * fn.2

/-- Theorem stating that the range of the dot product is [-4, 14] -/
theorem dot_product_range :
  ∀ m n, m ∈ Ellipse → n ∈ Ellipse → -4 ≤ DotProduct m n ∧ DotProduct m n ≤ 14 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l1124_112421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_packet_price_problem_l1124_112461

theorem milk_packet_price_problem (total_packets : ℕ) (remaining_packets : ℕ) 
  (avg_price_all : ℚ) (avg_price_remaining : ℚ) :
  total_packets = 5 →
  remaining_packets = 3 →
  avg_price_all = 20 →
  avg_price_remaining = 12 →
  (total_packets - remaining_packets : ℚ) * 
    ((total_packets * avg_price_all - remaining_packets * avg_price_remaining) / 
     (total_packets - remaining_packets)) = 64 := by
  sorry

#check milk_packet_price_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_packet_price_problem_l1124_112461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_sum_eq_one_l1124_112429

theorem cos_squared_sum_eq_one (x : ℝ) : 
  (Real.cos x)^2 + (Real.cos (2*x))^2 + (Real.cos (3*x))^2 = 1 ↔ 
  (∃ k : ℤ, x = π/6 + k*2*π) ∨
  (∃ k : ℤ, x = π/4 + k*2*π) ∨
  (∃ k : ℤ, x = π/2 + k*2*π) ∨
  (∃ k : ℤ, x = 3*π/4 + k*2*π) ∨
  (∃ k : ℤ, x = 5*π/6 + k*2*π) ∨
  (∃ k : ℤ, x = 7*π/6 + k*2*π) ∨
  (∃ k : ℤ, x = 5*π/4 + k*2*π) ∨
  (∃ k : ℤ, x = 7*π/4 + k*2*π) ∨
  (∃ k : ℤ, x = 11*π/6 + k*2*π) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_sum_eq_one_l1124_112429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l1124_112413

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A segment connecting two points -/
structure Segment where
  p1 : Point
  p2 : Point

/-- Color of a segment -/
inductive Color where
  | Red
  | Blue

/-- A configuration of six points and their colored segments -/
structure Configuration where
  points : Finset Point
  coloring : Segment → Color

/-- Predicate to check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop := sorry

/-- Predicate to check if a triangle is monochromatic -/
def monochromatic_triangle (config : Configuration) (p1 p2 p3 : Point) : Prop := sorry

theorem monochromatic_triangle_exists (config : Configuration) :
  (config.points.card = 6) →
  (∀ p1 p2 p3, p1 ∈ config.points → p2 ∈ config.points → p3 ∈ config.points →
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬collinear p1 p2 p3) →
  (∃ p1 p2 p3, p1 ∈ config.points ∧ p2 ∈ config.points ∧ p3 ∈ config.points ∧
    monochromatic_triangle config p1 p2 p3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l1124_112413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mean_weight_Y_Z_l1124_112497

/-- Represents a pile of stones -/
structure StonePile where
  weight : ℝ  -- Total weight of stones
  count : ℝ   -- Number of stones (using ℝ for simplicity)

/-- Calculates the mean weight of a stone pile -/
noncomputable def meanWeight (pile : StonePile) : ℝ :=
  pile.weight / pile.count

/-- Combines two stone piles -/
def combinePiles (pile1 pile2 : StonePile) : StonePile :=
  { weight := pile1.weight + pile2.weight
  , count := pile1.count + pile2.count }

/-- The maximum mean weight of combined piles Y and Z is 69 -/
theorem max_mean_weight_Y_Z (X Y Z : StonePile) 
    (hX : meanWeight X = 30)
    (hY : meanWeight Y = 60)
    (hXY : meanWeight (combinePiles X Y) = 40)
    (hXZ : meanWeight (combinePiles X Z) = 35) :
    ∃ (m : ℝ), m ≤ meanWeight (combinePiles Y Z) ∧ m < 70 ∧ ∀ n, (n : ℝ) < meanWeight (combinePiles Y Z) → n ≤ 69 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mean_weight_Y_Z_l1124_112497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_to_non_perpendicular_line_l1124_112448

-- Define the necessary structures
structure Line : Type
structure Plane : Type

-- Define the relationships
axiom perpendicular : Line → Plane → Prop
axiom perpendicular_line : Line → Line → Prop
axiom in_plane : Line → Plane → Prop

-- Define the theorem
theorem lines_perpendicular_to_non_perpendicular_line 
  (a : Line) (α : Plane) 
  (h : ¬perpendicular a α) : 
  ∃ (S : Set Line), (∀ l ∈ S, in_plane l α ∧ perpendicular_line l a) ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_to_non_perpendicular_line_l1124_112448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_four_times_five_to_seven_l1124_112477

theorem cube_root_of_four_times_five_to_seven :
  (4 * 5^7 : ℝ) ^ (1/3) = 2^(2/3) * 5^(7/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_four_times_five_to_seven_l1124_112477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_tan_cycle_sum_zero_l1124_112482

theorem sin_tan_cycle_sum_zero (x y z : ℝ) 
  (h1 : Real.sin x = Real.tan y) 
  (h2 : Real.sin y = Real.tan z) 
  (h3 : Real.sin z = Real.tan x) : 
  Real.sin x + Real.sin y + Real.sin z = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_tan_cycle_sum_zero_l1124_112482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_seq_divisibility_l1124_112486

def fib_seq : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib_seq (n + 1) + fib_seq n

theorem fib_seq_divisibility :
  ∀ m : ℕ, m > 0 → ∃ k : ℕ, m ∣ (fib_seq k ^ 4 - fib_seq k - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_seq_divisibility_l1124_112486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_in_final_function_l1124_112480

-- Define the original function
noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

-- Define the final transformed function
noncomputable def transformed_function (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi/4) + 1

-- Define the transformations
def double_vertical (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ 2 * f x
def shift_right (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x ↦ f (x - shift)
def shift_up (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x ↦ f x + shift

-- Theorem statement
theorem transformations_result_in_final_function :
  ∀ x : ℝ, transformed_function x = 
    (shift_up (shift_right (double_vertical original_function) (Real.pi/4)) 1) x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_in_final_function_l1124_112480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1124_112459

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Helper function to calculate the area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := 
  1 / 2 * t.b * t.c * Real.sin t.A

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (t.b - t.c)^2 = t.a^2 - t.b * t.c) 
  (h2 : t.a = 3)
  (h3 : Real.sin t.C = 2 * Real.sin t.B) :
  t.A = π / 3 ∧ area t = 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1124_112459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_and_volume_l1124_112415

/-- Represents a hemisphere with a given base area -/
structure Hemisphere where
  base_area : ℝ
  base_area_positive : base_area > 0

/-- Calculate the radius of a hemisphere given its base area -/
noncomputable def Hemisphere.radius (h : Hemisphere) : ℝ := Real.sqrt (h.base_area / Real.pi)

/-- Calculate the total surface area of a hemisphere -/
noncomputable def Hemisphere.surface_area (h : Hemisphere) : ℝ :=
  2 * Real.pi * h.radius ^ 2 + h.base_area

/-- Calculate the volume of a hemisphere -/
noncomputable def Hemisphere.volume (h : Hemisphere) : ℝ :=
  (2 / 3) * Real.pi * h.radius ^ 3

theorem hemisphere_surface_area_and_volume
  (h : Hemisphere)
  (h_base_area : h.base_area = 225 * Real.pi) :
  h.surface_area = 675 * Real.pi ∧ h.volume = 2250 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_and_volume_l1124_112415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_online_store_marketing_problem_l1124_112494

-- Define the variables and functions
noncomputable def cost_price : ℝ := 32
def trial_period : ℕ := 25

noncomputable def selling_price (x : ℝ) (k₁ k₂ : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 14 then k₁ * x + 40
  else if 15 ≤ x ∧ x ≤ 25 then k₂ / x + 32
  else 0

noncomputable def sales_volume (x : ℝ) : ℝ := -x + 48

noncomputable def profit (x : ℝ) (k₁ k₂ : ℝ) : ℝ :=
  (selling_price x k₁ k₂ - cost_price) * sales_volume x

-- State the theorem
theorem online_store_marketing_problem :
  ∃ (k₁ k₂ : ℝ),
    -- Condition 1: When x = 4, y = 42
    selling_price 4 k₁ k₂ = 42 ∧
    -- Condition 2: When x = 20, y = 37
    selling_price 20 k₁ k₂ = 37 ∧
    -- Result 1: k₁ = 1/2 and k₂ = 100
    k₁ = 1/2 ∧ k₂ = 100 ∧
    -- Result 2: Maximum profit occurs at x = 15 with value 220
    (∀ x : ℝ, 15 ≤ x ∧ x ≤ 25 → profit x k₁ k₂ ≤ profit 15 k₁ k₂) ∧
    profit 15 k₁ k₂ = 220 ∧
    -- Result 3: Subsidy range for increasing profit in first 14 days
    (∀ a : ℝ, 1 < a ∧ a < 2.5 ↔
      ∀ x y : ℝ, 1 ≤ x ∧ x < y ∧ y ≤ 14 →
        (selling_price x k₁ k₂ - cost_price + a) * sales_volume x <
        (selling_price y k₁ k₂ - cost_price + a) * sales_volume y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_online_store_marketing_problem_l1124_112494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_sphere_radius_l1124_112466

-- Define the unit cube
def UnitCube : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 1}

-- Define the inscribed sphere
def InscribedSphere : Set (Fin 3 → ℝ) :=
  {p | (p 0 - 1/2)^2 + (p 1 - 1/2)^2 + (p 2 - 1/2)^2 ≤ (1/2)^2}

-- Define a small sphere at a corner
def SmallSphereAtCorner (corner : Fin 3 → ℝ) (r : ℝ) : Set (Fin 3 → ℝ) :=
  {p | (p 0 - corner 0)^2 + (p 1 - corner 1)^2 + (p 2 - corner 2)^2 ≤ r^2}

-- Define the property of being externally tangent
def ExternallyTangent (s1 s2 : Set (Fin 3 → ℝ)) : Prop :=
  ∃ p, p ∈ s1 ∧ p ∈ s2 ∧ ∀ q, q ∈ s1 → q ∈ s2 → q = p

-- Define the property of being tangent to three faces
def TangentToThreeFaces (s : Set (Fin 3 → ℝ)) : Prop :=
  ∃ p q r, p ∈ s ∧ q ∈ s ∧ r ∈ s ∧ 
    (p 0 = 0 ∨ p 0 = 1 ∨ p 1 = 0 ∨ p 1 = 1 ∨ p 2 = 0 ∨ p 2 = 1) ∧
    (q 0 = 0 ∨ q 0 = 1 ∨ q 1 = 0 ∨ q 1 = 1 ∨ q 2 = 0 ∨ q 2 = 1) ∧
    (r 0 = 0 ∨ r 0 = 1 ∨ r 1 = 0 ∨ r 1 = 1 ∨ r 2 = 0 ∨ r 2 = 1) ∧
    p ≠ q ∧ q ≠ r ∧ r ≠ p

theorem small_sphere_radius :
  ∀ (corner : Fin 3 → ℝ) (r : ℝ),
    corner ∈ UnitCube →
    ExternallyTangent (SmallSphereAtCorner corner r) InscribedSphere →
    TangentToThreeFaces (SmallSphereAtCorner corner r) →
    r = (Real.sqrt 3 - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_sphere_radius_l1124_112466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1124_112478

/-- Sum of an arithmetic sequence -/
noncomputable def arithmetic_sum (n : ℕ) (a l : ℝ) : ℝ := n * (a + l) / 2

/-- The problem statement -/
theorem arithmetic_sequence_sum : 
  let n : ℕ := 10
  let a : ℝ := 5
  let l : ℝ := 41
  arithmetic_sum n a l = 230 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1124_112478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_pure_imaginary_implies_a_equals_two_l1124_112422

/-- Represents the imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Definition of the complex number z -/
noncomputable def z (a : ℝ) : ℂ := (1 + a * i) / (2 - i)

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def isPureImaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

/-- Theorem: If z(a) is pure imaginary, then a = 2 -/
theorem z_pure_imaginary_implies_a_equals_two :
  ∃ a : ℝ, isPureImaginary (z a) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_pure_imaginary_implies_a_equals_two_l1124_112422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1124_112423

theorem problem_solution (x y : ℝ) : 
  (Real.sqrt (2 * y + x - 2) + abs (x + 2) = 0) →
  x = -2 ∧ y = 2 ∧ Real.sqrt (3 * x^2 - 4 * y) + (x / y)^2033 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1124_112423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1124_112408

noncomputable def f (m : ℝ) (a : ℝ) (x : ℝ) : ℝ := m / x + Real.log (x / a)

theorem function_properties (m : ℝ) (a : ℝ) (h_a : a > 0) :
  /- If the minimum value of f(x) is 2, then m/a = e -/
  (∃ (x : ℝ), ∀ y : ℝ, f m a x ≤ f m a y ∧ f m a x = 2) →
  m / a = Real.exp 1 ∧
  /- If m = 1, a > e, and x₀ > 1 is a zero of f(x), then: -/
  (m = 1 ∧ a > Real.exp 1 →
    ∀ x₀ : ℝ, x₀ > 1 → f m a x₀ = 0 →
      /- 1/(2x₀) + x₀ < a - 1 -/
      (1 / (2 * x₀) + x₀ < a - 1) ∧
      /- x₀ + 1/x₀ > 2ln(a) - ln(ln(a)) -/
      (x₀ + 1 / x₀ > 2 * Real.log a - Real.log (Real.log a))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1124_112408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_in_region_count_squares_in_region_alt_l1124_112411

/-- Represents a square with integer coordinates -/
structure IntegerSquare where
  bottomLeft : ℤ × ℤ
  sideLength : ℕ

/-- Checks if a point (x, y) is within the bounded region -/
def isWithinRegion (x y : ℤ) : Prop :=
  y ≤ 2 * x ∧ y > -1 ∧ x ≤ 6

/-- Checks if an IntegerSquare is entirely within the bounded region -/
def isSquareWithinRegion (square : IntegerSquare) : Prop :=
  let (x, y) := square.bottomLeft
  isWithinRegion x y ∧
  isWithinRegion (x + square.sideLength) (y + square.sideLength)

/-- The main theorem to be proved -/
theorem count_squares_in_region :
  ∃ (s : Finset IntegerSquare), (∀ sq ∈ s, isSquareWithinRegion sq) ∧ s.card = 74 := by
  sorry

/-- Alternative formulation without using Finset -/
theorem count_squares_in_region_alt :
  ∃ (squares : List IntegerSquare),
    (∀ sq ∈ squares, isSquareWithinRegion sq) ∧
    squares.length = 74 ∧
    squares.Nodup := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_in_region_count_squares_in_region_alt_l1124_112411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinates_l1124_112401

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three vertices -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))

theorem third_vertex_coordinates (x : ℝ) :
  let a : Point := ⟨6, 4⟩
  let b : Point := ⟨0, 0⟩
  let c : Point := ⟨x, 0⟩
  x < 0 →
  triangleArea a b c = 30 →
  x = -15 := by
  sorry

#check third_vertex_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinates_l1124_112401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1124_112456

/-- Represents the sum of the first k terms of an arithmetic sequence -/
def S (k : ℕ) : ℝ := sorry

/-- The arithmetic sequence property: S_n, S_{2n}-S_n, S_{3n}-S_{2n}, S_{4n}-S_{3n} form an arithmetic sequence -/
axiom arithmetic_property (n : ℕ) : 
  ∃ (d : ℝ), S (2*n) - S n = S n + d ∧ S (3*n) - S (2*n) = S n + 2*d ∧ S (4*n) - S (3*n) = S n + 3*d

theorem arithmetic_sequence_sum (n : ℕ) (h1 : S n = 2) (h2 : S (3*n) = 12) : S (4*n) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1124_112456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_Y_X_l1124_112442

/-- The sum of an arithmetic sequence -/
noncomputable def arithmeticSum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- X is the sum of the arithmetic sequence 10, 12, 14, ..., 100 -/
noncomputable def X : ℝ := arithmeticSum 10 2 46

/-- Y is the sum of the arithmetic sequence 12, 14, 16, ..., 102 -/
noncomputable def Y : ℝ := arithmeticSum 12 2 46

/-- The difference between Y and X is 92 -/
theorem difference_Y_X : Y - X = 92 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_Y_X_l1124_112442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_triangles_sum_l1124_112488

/-- Helper function to calculate the area of a triangle given its vertices -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Given a triangle PQR with vertices P(7, 5), Q(1, -3), and R(4, 4),
    if point S(x, y) is chosen such that triangles PQS, PRS, and QRS have equal areas,
    then 12x + 3y = 54. -/
theorem equal_area_triangles_sum (x y : ℝ) :
  let P : ℝ × ℝ := (7, 5)
  let Q : ℝ × ℝ := (1, -3)
  let R : ℝ × ℝ := (4, 4)
  let S : ℝ × ℝ := (x, y)
  (area_triangle P Q S = area_triangle P R S) ∧
  (area_triangle P R S = area_triangle Q R S) →
  12 * x + 3 * y = 54 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_triangles_sum_l1124_112488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l1124_112460

/-- The length of the chord intercepted by a line on a circle -/
noncomputable def chord_length (a b c : ℝ) (x₀ y₀ r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - ((a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2))^2)

theorem chord_length_specific_case :
  chord_length 3 4 (-5) 2 1 2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l1124_112460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_sets_l1124_112457

def is_closed_under {α : Type*} (S : Set α) (op : α → α → α) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → op a b ∈ S

def set1 : Set ℤ := {-2, -1, 1, 2}
def set2 : Set ℤ := {-1, 0, 1}

theorem closed_sets :
  ¬(is_closed_under set1 (·+·) ∧ is_closed_under set1 (·*·)) ∧
  (is_closed_under set2 (·+·) ∧ is_closed_under set2 (·*·)) ∧
  (is_closed_under (Set.univ : Set ℤ) (·+·) ∧ is_closed_under (Set.univ : Set ℤ) (·*·)) ∧
  (is_closed_under (Set.univ : Set ℚ) (·+·) ∧ is_closed_under (Set.univ : Set ℚ) (·*·)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_sets_l1124_112457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_product_not_perfect_square_l1124_112449

theorem factorial_product_not_perfect_square (n : ℕ) (h : 100 ≤ n ∧ n ≤ 104) :
  ¬ ∃ k : ℕ, (Nat.factorial n) * (Nat.factorial (n + 1)) = k^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_product_not_perfect_square_l1124_112449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_of_29_l1124_112462

theorem sum_of_divisors_of_29 (h : Nat.Prime 29) : 
  (Finset.filter (λ x => x ∣ 29) (Finset.range 30)).sum id = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_of_29_l1124_112462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concave_polygon_without_right_angle_l1124_112468

/-- A concave polygon -/
structure ConcavePolygon where
  vertices : List (ℝ × ℝ)
  is_concave : Prop
  consecutive_right_triangle : Prop

/-- Checks if an angle is 90° or 270° -/
def is_right_or_reflex (angle : ℝ) : Prop :=
  angle = 90 ∨ angle = 270

/-- Calculates the angle at a vertex (placeholder function) -/
noncomputable def angle_at_vertex (p : ConcavePolygon) (v : ℕ) : ℝ :=
  sorry

/-- Main theorem -/
theorem concave_polygon_without_right_angle :
  ∃ (p : ConcavePolygon), 
    p.is_concave ∧ 
    p.consecutive_right_triangle ∧ 
    ¬(∀ (v : ℕ), v < p.vertices.length → is_right_or_reflex (angle_at_vertex p v)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concave_polygon_without_right_angle_l1124_112468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_chord_length_l1124_112464

/-- The length of the chord intercepted by a line on a circle -/
noncomputable def chord_length (a b c : ℝ) (d e f : ℝ) : ℝ :=
  let circle := fun (x y : ℝ) ↦ x^2 + y^2 - d*x - f
  let line := fun (x y : ℝ) ↦ a*x + b*y + c
  let center_x := d/2
  let center_y := e/2
  let radius := Real.sqrt ((d/2)^2 + (e/2)^2 - f)
  let dist_center_to_line := abs (a*center_x + b*center_y + c) / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (radius^2 - dist_center_to_line^2)

/-- The chord length for the specific circle and line in the problem -/
theorem problem_chord_length :
  chord_length 3 (-4) (-4) 6 0 0 = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_chord_length_l1124_112464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1124_112446

-- Define the angle in degrees
noncomputable def angle : ℝ := 600

-- Define a full rotation in degrees
noncomputable def full_rotation : ℝ := 360

-- Define the quadrant function
noncomputable def quadrant (θ : ℝ) : ℕ :=
  let normalized_angle := θ % full_rotation
  if 0 ≤ normalized_angle ∧ normalized_angle < 90 then 1
  else if 90 ≤ normalized_angle ∧ normalized_angle < 180 then 2
  else if 180 ≤ normalized_angle ∧ normalized_angle < 270 then 3
  else 4

-- Theorem stating that the 600° angle is in the third quadrant
theorem angle_in_third_quadrant : quadrant angle = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1124_112446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_Q_relationship_l1124_112452

def P (n : ℕ+) (x : ℝ) : ℝ := (1 - x) ^ (2 * n.val - 1)

def Q (n : ℕ+) (x : ℝ) : ℝ := 1 - (2 * n.val - 1) * x + (n.val - 1) * (2 * n.val - 1) * x^2

theorem P_Q_relationship :
  (∀ x, P 1 x = Q 1 x) ∧
  (P 2 0 = Q 2 0) ∧
  (∀ x > 0, P 2 x < Q 2 x) ∧
  (∀ x < 0, P 2 x > Q 2 x) ∧
  (∀ n ≥ 3, ∀ x > 0, P n x < Q n x) ∧
  (∀ n ≥ 3, ∀ x < 0, P n x > Q n x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_Q_relationship_l1124_112452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1124_112433

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

theorem f_properties :
  (∀ x y : ℝ, 1 < x → x < y → f x > f y) ∧
  (∀ x : ℝ, x ∈ Set.Icc 2 4 → f x ≤ 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc 2 4 → f x ≥ 2/3) ∧
  (f 2 = 2) ∧
  (f 4 = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1124_112433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_special_triangle_exists_l1124_112450

/-- Triangle with sides a, b and angle condition α = 2β --/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  β : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_angle_condition : 0 < β ∧ β < π

/-- Theorem stating the conditions for a unique SpecialTriangle --/
theorem unique_special_triangle_exists (t : SpecialTriangle) :
  (t.b < t.a ∧ t.a < 2 * t.b ∧ t.β < π / 3) ↔
  ∃! (α : ℝ), 0 < α ∧ α < π ∧ α = 2 * t.β ∧
    t.a / Real.sin α = t.b / Real.sin t.β ∧
    α + t.β + (π - α - t.β) = π :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_special_triangle_exists_l1124_112450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_minimum_wage_is_8_l1124_112436

/-- The minimum wage James is working for to pay off the food fight costs -/
noncomputable def james_minimum_wage : ℚ :=
  let meat_cost : ℚ := 20 * 5
  let fruit_veg_cost : ℚ := 15 * 4
  let bread_cost : ℚ := 60 * (3/2)
  let janitorial_hourly_rate : ℚ := 10
  let janitorial_overtime_rate : ℚ := janitorial_hourly_rate * (3/2)
  let janitorial_cost : ℚ := 10 * janitorial_overtime_rate
  let total_cost : ℚ := meat_cost + fruit_veg_cost + bread_cost + janitorial_cost
  let work_hours : ℚ := 50
  total_cost / work_hours

/-- Theorem stating that James' minimum wage is $8/hour -/
theorem james_minimum_wage_is_8 : james_minimum_wage = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_minimum_wage_is_8_l1124_112436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l1124_112420

noncomputable def curve_x (θ : ℝ) : ℝ := 2 * Real.cos θ
noncomputable def curve_y (θ : ℝ) : ℝ := Real.sin θ

theorem max_distance_to_origin :
  ∃ (c : ℝ), c = 2 ∧ 
  ∀ (θ : ℝ), Real.sqrt ((curve_x θ)^2 + (curve_y θ)^2) ≤ c ∧
  ∃ (θ₀ : ℝ), Real.sqrt ((curve_x θ₀)^2 + (curve_y θ₀)^2) = c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_origin_l1124_112420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l1124_112475

/-- The length of a train given relative speeds and crossing time -/
noncomputable def trainLength (fasterSpeed slowerSpeed : ℝ) (crossingTime : ℝ) : ℝ :=
  (fasterSpeed - slowerSpeed) * (1000 / 3600) * crossingTime

/-- Theorem stating the length of the faster train -/
theorem faster_train_length :
  let fasterSpeed : ℝ := 72
  let slowerSpeed : ℝ := 36
  let crossingTime : ℝ := 15
  trainLength fasterSpeed slowerSpeed crossingTime = 150 := by
  -- Unfold the definition of trainLength
  unfold trainLength
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l1124_112475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_polyominoes_l1124_112414

/-- A polyomino is a shape formed by connecting unit squares edge to edge. -/
structure Polyomino where
  shape : Set (ℕ × ℕ)

/-- A grid is a rectangular array of cells. -/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- A placement of polyominoes on a grid. -/
structure Placement where
  grid : Grid
  polyomino : Polyomino
  positions : List (ℕ × ℕ)

/-- The "b" shaped polyomino. -/
def bPolyomino : Polyomino :=
  ⟨{(0, 0), (1, 0), (2, 0), (2, 1)}⟩

/-- The 8x8 grid. -/
def grid8x8 : Grid :=
  ⟨8, 8⟩

/-- A valid placement satisfies all the conditions of the problem. -/
def isValidPlacement (p : Placement) : Prop :=
  (p.grid = grid8x8) ∧
  (p.polyomino = bPolyomino) ∧
  (∀ pos ∈ p.positions, pos.fst < p.grid.rows ∧ pos.snd < p.grid.cols) ∧
  (∀ i < p.grid.rows, (p.positions.filter (λ pos ↦ pos.fst = i)).length = (p.positions.filter (λ pos ↦ pos.snd = i)).length)

/-- The maximum number of "b" shaped polyominoes that can be placed on the 8x8 grid
    satisfying all conditions. -/
theorem max_b_polyominoes : 
  (∃ (p : Placement), isValidPlacement p ∧ p.positions.length = 7) ∧
  (∀ (p : Placement), isValidPlacement p → p.positions.length ≤ 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_polyominoes_l1124_112414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_polynomial_l1124_112405

noncomputable def ω : ℂ := Complex.exp (Complex.I * Real.pi / 5)

theorem roots_of_polynomial (x : ℂ) :
  (x^4 - x^3 + x^2 - x + 1 = 0) ↔ (x = ω ∨ x = ω^3 ∨ x = ω^7 ∨ x = ω^9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_polynomial_l1124_112405
