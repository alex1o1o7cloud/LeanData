import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_minus_single_l896_89667

theorem tan_double_minus_single (α β : ℝ) 
  (h1 : Real.tan α = 1/2) 
  (h2 : Real.tan (α - β) = -2/5) : 
  Real.tan (2*α - β) = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_minus_single_l896_89667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_price_increase_l896_89697

theorem candy_bar_price_increase 
  (original_weight : ℝ) 
  (original_price : ℝ) 
  (weight_reduction : ℝ) 
  (old_tax_rate : ℝ) 
  (new_tax_rate : ℝ) 
  (old_exchange_rate : ℝ) 
  (new_exchange_rate : ℝ) 
  (h1 : weight_reduction = 0.4) 
  (h2 : old_tax_rate = 0.05) 
  (h3 : new_tax_rate = 0.08) 
  (h4 : old_exchange_rate = 1.2) 
  (h5 : new_exchange_rate = 1.35) :
  let old_effective_price := (original_price * (1 + old_tax_rate) * old_exchange_rate) / original_weight
  let new_effective_price := (original_price * (1 + new_tax_rate) * new_exchange_rate) / (original_weight * (1 - weight_reduction))
  let percent_increase := (new_effective_price / old_effective_price - 1) * 100
  ∃ (ε : ℝ), abs (percent_increase - 92.86) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_price_increase_l896_89697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l896_89685

noncomputable def f (x : ℝ) := (3/4) * x^2 - 3*x + 4

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x, a ≤ f x ∧ f x ≤ b ↔ a ≤ x ∧ x ≤ b) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l896_89685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_number_of_presidencies_l896_89664

-- Define number_of_presidencies as a function
def number_of_presidencies (n : ℕ) : Fin n → ℕ := sorry

theorem different_number_of_presidencies (n : ℕ) (h : n = 1001) : 
  ¬ ∃ (k : ℕ), (∀ resident : Fin n, 
    (number_of_presidencies n resident = k) ∧ 
    (Nat.choose n 13 = n * k)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_number_of_presidencies_l896_89664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_change_l896_89624

/-- The population change over two years -/
theorem population_change 
  (initial_population : ℕ) 
  (first_year_increase : ℚ) 
  (second_year_decrease : ℚ) : 
  initial_population = 415600 →
  first_year_increase = 25 / 100 →
  second_year_decrease = 30 / 100 →
  (↑initial_population * (1 + first_year_increase) * (1 - second_year_decrease)).floor = 363650 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_change_l896_89624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_proof_l896_89690

/-- A line passing through (2,1) with an inclination angle of π/4 -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + Real.sqrt 2/2 * t, 1 + Real.sqrt 2/2 * t)

/-- The curve C defined by parametric equations -/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 * Real.cos θ, 2 * Real.sin θ)

/-- The distance between intersection points of line_l and curve_C -/
noncomputable def intersection_distance : ℝ := 4 * Real.sqrt 26 / 3

theorem intersection_distance_proof :
  ∃ (t₁ t₂ : ℝ), 
    (∃ (θ₁ θ₂ : ℝ), line_l t₁ = curve_C θ₁ ∧ line_l t₂ = curve_C θ₂) ∧
    Real.sqrt ((t₁ - t₂)^2 * ((Real.sqrt 2/2)^2 + (Real.sqrt 2/2)^2)) = intersection_distance := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_proof_l896_89690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_dress_price_l896_89609

/-- Calculates the final price of a dress for a staff member after discounts and tax --/
noncomputable def final_price (d : ℝ) (t : ℝ) : ℝ :=
  let initial_discount := 0.65
  let staff_discount := 0.60
  let price_after_first_discount := d * (1 - initial_discount)
  let price_after_staff_discount := price_after_first_discount * (1 - staff_discount)
  price_after_staff_discount * (1 + t / 100)

/-- Theorem stating the final price calculation for a staff member --/
theorem staff_dress_price (d : ℝ) (t : ℝ) :
  final_price d t = 0.14 * d + 0.0014 * d * t :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_dress_price_l896_89609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_never_meet_l896_89620

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  initialPosition : ℝ

/-- The problem setup -/
def runnerProblem (C : ℝ) : Prop :=
  C > 0 ∧
  ∃ (r1 r2 r3 : Runner),
    r1.speed = 1 ∧
    r2.speed = 2 ∧
    r3.speed = 4 ∧
    r1.initialPosition = 0 ∧
    r2.initialPosition = C / 3 ∧
    r3.initialPosition = 2 * C / 3

/-- Position of a runner at time t -/
noncomputable def position (r : Runner) (C : ℝ) (t : ℝ) : ℝ :=
  (r.initialPosition + r.speed * t) % C

/-- The main theorem: runners never meet -/
theorem runners_never_meet (C : ℝ) (h : runnerProblem C) :
  ∀ t : ℝ, t > 0 →
    ∀ r1 r2 r3 : Runner,
      (r1.speed = 1 ∧ r2.speed = 2 ∧ r3.speed = 4 ∧
       r1.initialPosition = 0 ∧ r2.initialPosition = C / 3 ∧ r3.initialPosition = 2 * C / 3) →
      ¬(position r1 C t = position r2 C t ∧ position r2 C t = position r3 C t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_never_meet_l896_89620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l896_89651

noncomputable def a : ℝ × ℝ := (1/2, Real.sqrt 3 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := a.1 * (b x).1 + a.2 * (b x).2 + 2

theorem f_properties :
  (∀ x, f x ≤ 3) ∧
  (∀ x, f x ≥ 1) ∧
  (∃ x, f x = 3) ∧
  (∃ x, f x = 1) ∧
  (∀ x y, π/6 ≤ x ∧ x < y ∧ y ≤ 7*π/6 → f y < f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l896_89651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_monotonic_interval_l896_89601

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := tan (x + π / 4)

-- State the theorem
theorem tan_monotonic_interval :
  ∀ k : ℤ, StrictMonoOn f (Set.Ioo ((k : ℝ) * π - 3 * π / 4) ((k : ℝ) * π + π / 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_monotonic_interval_l896_89601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_line_equation_l896_89671

-- Define the circle C₁
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}

-- Define the tangent line
def tangentLine (x y : ℝ) : Prop := 3*x + 4*y - 10 = 0

-- Define the point M
def M : ℝ × ℝ := (1, 2)

-- Define the length of the chord
noncomputable def chordLength : ℝ := 2 * Real.sqrt 3

-- Theorem for the equation of circle C₁
theorem circle_equation :
  C₁ = {p : ℝ × ℝ | p.1^2 + p.2^2 = 4} := by
  sorry

-- Theorem for the equations of line l
theorem line_equation (l : Set (ℝ × ℝ)) :
  (∀ p, p ∈ l → (p.1 = 1 ∨ 3*p.1 - 4*p.2 + 5 = 0)) →
  M ∈ l →
  (∃ p q, p ∈ l ∩ C₁ ∧ q ∈ l ∩ C₁ ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chordLength) →
  (∀ p, p ∈ l → p.1 = 1) ∨ (∀ p, p ∈ l → 3*p.1 - 4*p.2 + 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_line_equation_l896_89671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parameter_sum_l896_89680

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  focus : ℝ × ℝ
  vertex : ℝ × ℝ

/-- Calculates the parameters of the hyperbola -/
noncomputable def hyperbola_parameters (H : Hyperbola) : ℝ × ℝ × ℝ × ℝ := 
  let (h, k) := H.center
  let a := abs (H.vertex.2 - k)
  let c := abs (H.focus.2 - k)
  let b := Real.sqrt (c^2 - a^2)
  (h, k, a, b)

/-- Theorem stating the sum of hyperbola parameters -/
theorem hyperbola_parameter_sum (H : Hyperbola) 
  (h_center : H.center = (3, 0)) 
  (h_focus : H.focus = (3, 10)) 
  (h_vertex : H.vertex = (3, 4)) : 
  let (h, k, a, b) := hyperbola_parameters H
  h + k + a + b = 7 + 2 * Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parameter_sum_l896_89680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_ratio_is_7_10_l896_89694

/-- Represents the investment and profit details of a partner -/
structure Partner where
  investment : ℕ
  time : ℕ

/-- Calculates the profit factor of a partner -/
def profitFactor (p : Partner) : ℕ := p.investment * p.time

/-- Proves that the profit ratio of two partners is 7:10 given the specified conditions -/
theorem profit_ratio_is_7_10 (p q : Partner) 
  (h1 : p.investment = 7 ∧ q.investment = 5)  -- Investment ratio is 7:5
  (h2 : p.time = 8)                           -- p invests for 8 months
  (h3 : q.time = 16)                          -- q invests for 16 months
  : (profitFactor p : ℚ) / (profitFactor q : ℚ) = 7 / 10 := by
  sorry

-- Remove the #eval statement as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_ratio_is_7_10_l896_89694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_y_equals_x_l896_89628

theorem angle_on_y_equals_x (α : ℝ) : 
  (∃ x y : ℝ, y = x ∧ x = Real.cos α ∧ y = Real.sin α) → 
  1 - 2 * Real.sin α * Real.cos α - 3 * (Real.cos α)^2 = -3/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_y_equals_x_l896_89628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petrol_price_theorem_l896_89652

/-- Given a price reduction, additional gallons bought, total cost, and exchange rate,
    calculate the original price of petrol in euros per gallon. -/
noncomputable def original_price_euros (price_reduction : ℝ) (additional_gallons : ℝ) 
                         (total_cost : ℝ) (exchange_rate : ℝ) : ℝ :=
  let price_usd := total_cost * price_reduction / 
                   (additional_gallons * (1 - (1 - price_reduction)))
  price_usd / exchange_rate

/-- Theorem stating that given the specific conditions, the original price
    of petrol is approximately 38.98 euros per gallon. -/
theorem petrol_price_theorem :
  let price_reduction := 0.135
  let additional_gallons := 7.25
  let total_cost := 325
  let exchange_rate := 1.15
  abs (original_price_euros price_reduction additional_gallons total_cost exchange_rate - 38.98) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval original_price_euros 0.135 7.25 325 1.15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petrol_price_theorem_l896_89652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_third_eq_sqrt_two_div_two_l896_89693

def g (x : ℝ) : ℝ := 1 - x^2

noncomputable def f (y : ℝ) : ℝ :=
  Real.sqrt ((1 - (1 - y)) / (1 - y))

theorem f_one_third_eq_sqrt_two_div_two :
  f (1/3) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_third_eq_sqrt_two_div_two_l896_89693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_over_square_root_of_five_l896_89616

theorem sixth_root_over_square_root_of_five :
  (5 : ℝ) ^ (1/6 : ℝ) / (5 : ℝ) ^ (1/2 : ℝ) = (5 : ℝ) ^ (-1/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_over_square_root_of_five_l896_89616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_traffic_light_timing_l896_89672

/-- Probability of passing through the first intersection without stopping -/
noncomputable def P1 (x : ℝ) : ℝ := x / (x + 30)

/-- Probability of passing through the second intersection without stopping -/
noncomputable def P2 (x : ℝ) : ℝ := 120 / (x + 120)

/-- Total probability of passing through both intersections without stopping -/
noncomputable def P (x : ℝ) : ℝ := P1 x * P2 x

theorem optimal_traffic_light_timing :
  ∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → P y ≤ P x) ∧
    x = 60 ∧ 
    P x = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_traffic_light_timing_l896_89672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l896_89647

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (3/5) * t, (4/5) * t)

-- Define the curve C
noncomputable def curve_C (k : ℝ) : ℝ × ℝ := (4 * k^2, 4 * k)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t k, line_l t = p ∧ curve_C k = p}

-- Define the length of a segment given two points
noncomputable def segment_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem length_of_AB : 
  ∀ A B, A ∈ intersection_points → B ∈ intersection_points → A ≠ B → segment_length A B = 25/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l896_89647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_coins_distribution_l896_89699

/-- Represents the number of pirates --/
def num_pirates : ℕ := 10

/-- Calculates the fraction of remaining coins taken by the k-th pirate --/
def pirate_fraction (k : ℕ) : ℚ :=
  if k ≤ num_pirates then k / num_pirates else 0

/-- Calculates the product of fractions for all pirates --/
def fraction_product : ℚ :=
  Finset.prod (Finset.range num_pirates) (fun k => 1 - pirate_fraction (k + 1))

/-- The smallest initial number of coins --/
def smallest_initial_coins : ℕ := 3628800

/-- The number of coins the 10th pirate receives --/
def tenth_pirate_coins : ℕ := 362880

theorem pirate_coins_distribution :
  (smallest_initial_coins : ℚ) * fraction_product = (tenth_pirate_coins : ℚ) ∧
  (∀ k, k ≤ num_pirates →
    ∃ n : ℕ, (smallest_initial_coins : ℚ) * 
      (Finset.prod (Finset.range k) (fun i => 1 - pirate_fraction (i + 1))) * 
      pirate_fraction k = (n : ℚ)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_coins_distribution_l896_89699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l896_89670

/-- Given a circle with center O and a point A not on the circle,
    if a line from A intersects the circle at points B and C,
    AB•AC = 60, and OA = 8, then the radius of the circle is 2. -/
theorem circle_radius_proof (O A B C : EuclideanSpace ℝ (Fin 2)) (r : ℝ) :
  A ∉ Metric.sphere O r →
  B ∈ Metric.sphere O r →
  C ∈ Metric.sphere O r →
  dist A B * dist A C = 60 →
  dist O A = 8 →
  r = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l896_89670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_for_cosine_function_l896_89632

theorem omega_value_for_cosine_function : 
  ∀ (ω : ℝ) (y : ℝ → ℝ),
  y = (λ x => 2 * Real.cos (π / 3 - ω * x)) →
  (∃ (T : ℝ), T > 0 ∧ T = 4 * π ∧ ∀ (x : ℝ), y (x + T) = y x) →
  ω = 1/2 ∨ ω = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_for_cosine_function_l896_89632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_exponents_l896_89689

theorem smallest_sum_of_exponents (a b : ℕ) (h : 3^8 * 5^2 * 2 = a^b) :
  ∃ (a' b' : ℕ), 3^8 * 5^2 * 2 = a'^b' ∧ a' + b' ≤ a + b ∧ a' + b' = 812 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_exponents_l896_89689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l896_89676

theorem train_crossing_time (length1 length2 time1 time2 : ℝ) 
  (h1 : length1 = 120)
  (h2 : length2 = 150)
  (h3 : time1 = 10)
  (h4 : time2 = 15)
  (h5 : length1 > 0)
  (h6 : length2 > 0)
  (h7 : time1 > 0)
  (h8 : time2 > 0) :
  (length1 + length2) / ((length1 / time1) - (length2 / time2)) = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l896_89676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l896_89691

noncomputable section

def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - (Real.cos (x + Real.pi / 4))^2

theorem triangle_max_area (A B C : ℝ) (hA : 0 < A ∧ A < Real.pi/2) (hB : 0 < B ∧ B < Real.pi/2) 
  (hC : 0 < C ∧ C < Real.pi/2) (hABC : A + B + C = Real.pi) (hf : f (A/2) = 0) 
  (ha : 1 = 2 * (Real.sin (B/2) * Real.sin (C/2) / Real.sin (A/2))) :
  ∃ (S : ℝ), S ≤ (2 + Real.sqrt 3) / 4 ∧ 
  (∀ (S' : ℝ), S' = Real.sin B * Real.sin C * Real.sin A / (2 * Real.sin (A/2)) → S' ≤ S) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l896_89691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l896_89625

noncomputable def f (x a b : ℝ) : ℝ := (x + 5) / (x^2 + a*x + b)

theorem asymptote_sum (a b : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -3 → f x a b ≠ 0) →
  (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |f x a b| > 1/ε) →
  (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x + 3| ∧ |x + 3| < δ → |f x a b| > 1/ε) →
  a + b = -5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l896_89625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_and_properties_l896_89648

noncomputable section

def f : ℝ → ℝ := sorry

axiom f_bounds : ∀ x : ℝ, 0 < x → x ≤ 5 → x ≤ f x ∧ f x ≤ 2 * |x - 1| + 1

axiom f_has_min : ∃ (x : ℝ), ∀ (y : ℝ), f x ≤ f y

axiom f_at_one : ∃ a : ℝ, f 1 = 4 * a + 1

axiom f_inequality : 
  ∀ m : ℝ, m > 1 → ∃ (t : ℝ), ∀ (x : ℝ), 1 ≤ x → x ≤ m → f t ≤ (1/4) * (x + t + 1)^2

theorem f_formula_and_properties :
  (∀ x : ℝ, f x = (1/4) * (x^2 + 2*x + 1)) ∧
  (∃ a : ℝ, a = (1/4) ∧ f 1 = 4 * a + 1) ∧
  (∀ m : ℝ, m > 5 → ¬(∃ (t : ℝ), ∀ (x : ℝ), 1 ≤ x → x ≤ m → f t ≤ (1/4) * (x + t + 1)^2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_and_properties_l896_89648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_half_solution_l896_89600

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

theorem f_half_solution (b : ℝ) :
  f b = 1/2 → (b = -3/2 ∨ b = Real.sqrt 2/2 ∨ b = -Real.sqrt 2/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_half_solution_l896_89600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematics_value_l896_89653

def letter_value (n : ℕ) : ℤ :=
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -3
  | _ => 0

def letter_position (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'b' => 2
  | 'c' => 3
  | 'd' => 4
  | 'e' => 5
  | 'f' => 6
  | 'g' => 7
  | 'h' => 8
  | 'i' => 9
  | 'j' => 10
  | 'k' => 11
  | 'l' => 12
  | 'm' => 13
  | 'n' => 14
  | 'o' => 15
  | 'p' => 16
  | 'q' => 17
  | 'r' => 18
  | 's' => 19
  | 't' => 20
  | 'u' => 21
  | 'v' => 22
  | 'w' => 23
  | 'x' => 24
  | 'y' => 25
  | 'z' => 26
  | _ => 0

def word_value (word : String) : ℤ :=
  word.toList.map (fun c => letter_value (letter_position c)) |>.sum

theorem mathematics_value :
  word_value "mathematics" = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematics_value_l896_89653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_unwrap_angle_l896_89619

/-- Given a cone with base radius 2 and slant height 6, 
    the central angle of the sector formed when the lateral surface is unwrapped is 120°. -/
theorem cone_unwrap_angle (r l : ℝ) (h1 : r = 2) (h2 : l = 6) : 
  (2 * Real.pi * r / l) * (180 / Real.pi) = 120 := by
  -- Substitute the given values
  rw [h1, h2]
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_unwrap_angle_l896_89619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_bisector_quadrilateral_area_l896_89641

/-- The area of the quadrilateral formed by the intersection of angle bisectors in a rectangle --/
theorem rectangle_bisector_quadrilateral_area (a b : ℝ) (h : b > a) (h' : a > 0) (h'' : b > 0) :
  let S := (b - a)^2 / 2
  S = S :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_bisector_quadrilateral_area_l896_89641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_cosine_value_l896_89617

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem function_and_cosine_value :
  ∃ (A ω φ : ℝ) (x₀ : ℝ),
    A > 0 ∧ ω > 0 ∧ |φ| ≤ Real.pi / 2 ∧
    x₀ ∈ Set.Icc (Real.pi / 2) Real.pi ∧
    (∀ x, f x = A * Real.sin (ω * x + φ)) ∧
    (∀ x, f x ≥ -2) ∧
    (∀ x, f (x + Real.pi / (2 * ω)) = f x) ∧
    (∀ x, f (x + Real.pi / 12) = f (-x - Real.pi / 12)) ∧
    f (x₀ / 2) = -3 / 8 ∧
    Real.cos (x₀ + Real.pi / 6) = -Real.sqrt 741 / 32 - 3 / 32 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_cosine_value_l896_89617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_same_color_count_is_even_l896_89610

/-- A color assignment for a grid point -/
inductive Color
| Red
| Blue

/-- Represents a square grid with colored vertices -/
structure ColoredGrid (n : ℕ+) where
  grid : Fin (n + 1) → Fin (n + 1) → Color
  corner_constraint :
    grid 0 0 = Color.Red ∧
    grid n 0 = Color.Blue ∧
    grid 0 n = Color.Blue ∧
    grid n n = Color.Red

/-- Counts the number of unit squares with exactly two vertices of the same color -/
def count_two_same_color (n : ℕ+) (g : ColoredGrid n) : ℕ :=
  sorry

/-- The main theorem: the count of unit squares with exactly two vertices of the same color is even -/
theorem two_same_color_count_is_even (n : ℕ+) (g : ColoredGrid n) :
  Even (count_two_same_color n g) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_same_color_count_is_even_l896_89610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_phi_square_l896_89611

-- Define Euler's totient function
def phi (n : Nat) : Nat := Nat.totient n

-- Define a predicate for perfect squares
def is_perfect_square (n : Nat) : Prop :=
  ∃ m : Nat, n = m * m

-- Define the property we want to prove
def is_largest_n_phi_square (n : Nat) : Prop :=
  is_perfect_square (n * phi n) ∧ 
  ∀ m : Nat, m > n → ¬is_perfect_square (m * phi m)

-- State the theorem
theorem largest_n_phi_square : 
  is_largest_n_phi_square 1 := by
  sorry

-- Additional lemma to help with the proof
lemma phi_one : phi 1 = 1 := by
  sorry

-- Another helpful lemma
lemma not_perfect_square_for_greater_than_one :
  ∀ n : Nat, n > 1 → ¬is_perfect_square (n * phi n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_phi_square_l896_89611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_length_l896_89686

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 5 = 0

-- Define the chord length
noncomputable def chord_length : ℝ := 4 * Real.sqrt 2

-- Theorem statement
theorem chord_intersection_length :
  ∃ (A B : ℝ × ℝ),
    (line_eq A.1 A.2 ∧ circle_eq A.1 A.2) ∧
    (line_eq B.1 B.2 ∧ circle_eq B.1 B.2) ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = chord_length := by
  sorry

#check chord_intersection_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_length_l896_89686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_profit_percent_l896_89663

theorem pen_profit_percent : 
  ∀ (marked_price : ℝ) (purchase_quantity : ℕ) (marked_quantity : ℕ) (discount_percent : ℝ),
  marked_price > 0 → 
  purchase_quantity = 120 → 
  marked_quantity = 100 → 
  discount_percent = 5 →
  let cost_price := marked_price * (marked_quantity : ℝ) / (purchase_quantity : ℝ)
  let selling_price := marked_price * (1 - discount_percent / 100)
  let profit_per_pen := selling_price - cost_price
  let profit_percent := (profit_per_pen / cost_price) * 100
  profit_percent = 70 :=
by
  intros marked_price purchase_quantity marked_quantity discount_percent
  intro h_marked_price h_purchase_quantity h_marked_quantity h_discount_percent
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_profit_percent_l896_89663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_solution_l896_89656

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3 * Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, -Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + Real.sqrt 3 / 2

theorem symmetry_and_solution (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : x₂ < π) 
  (h₄ : f x₁ = 1/3) (h₅ : f x₂ = 1/3) : 
  Real.cos (x₁ - x₂) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_solution_l896_89656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercept_lines_perpendicular_intersection_line_l896_89683

-- Define the point P
def P : ℝ × ℝ := (2, 2)

-- Define the lines from the conditions
def line1 (x y : ℝ) : Prop := 3 * x - 2 * y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 3 * y + 4 = 0

-- Define the function for lines with equal intercepts
def equal_intercepts (a b c : ℝ) : Prop := a * c = b * c ∧ a ≠ 0 ∧ b ≠ 0

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem for the first condition
theorem equal_intercept_lines :
  ∃ (a b c : ℝ), (a * P.1 + b * P.2 + c = 0 ∧ equal_intercepts a b c) →
    (a = 1 ∧ b = -1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -4) := by
  sorry

-- Theorem for the second condition
theorem perpendicular_intersection_line :
  ∃ (x y : ℝ), (line1 x y ∧ line2 x y) →
    ∃ (a b c : ℝ), (a * x + b * y + c = 0 ∧
      perpendicular (-3 : ℝ) (a / b)) →
        a = 3 ∧ b = -1 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercept_lines_perpendicular_intersection_line_l896_89683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_number_with_at_most_one_prime_l896_89606

/-- A function that checks if a number is prime -/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

/-- A function that removes seven digits from a nine-digit number -/
def removeSeven (n : Nat) (i j : Fin 9) : Nat :=
  sorry

/-- The theorem statement -/
theorem nine_digit_number_with_at_most_one_prime :
  ∃ (N : Nat),
    (N ≥ 100000000 ∧ N < 1000000000) ∧  -- Nine-digit number
    (∀ i j : Nat, i < 9 → j < 9 → i ≠ j → (N / (10^i)) % 10 ≠ (N / (10^j)) % 10) ∧  -- Distinct digits
    (∃! (i j : Fin 9), isPrime (removeSeven N i j)) ∧  -- At most one prime after removing seven digits
    N = 391524680  -- The specific number
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_number_with_at_most_one_prime_l896_89606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_is_ellipse_answer_is_correct_l896_89626

/-- The equation of a conic section -/
noncomputable def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-1)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 12

/-- The two fixed points (foci) -/
def focus1 : ℝ × ℝ := (0, 1)
def focus2 : ℝ × ℝ := (6, -4)

/-- Distance between the two foci -/
noncomputable def focal_distance : ℝ := Real.sqrt 61

/-- Theorem stating that the equation describes an ellipse -/
theorem conic_is_ellipse :
  (∀ x y, conic_equation x y ↔ 
    Real.sqrt ((x - focus1.1)^2 + (y - focus1.2)^2) +
    Real.sqrt ((x - focus2.1)^2 + (y - focus2.2)^2) = 12) ∧
  focal_distance < 12 →
  True := by
  sorry

/-- The answer to the problem -/
def answer : String := "E"

/-- Theorem stating that the answer is correct -/
theorem answer_is_correct : answer = "E" := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_is_ellipse_answer_is_correct_l896_89626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l896_89645

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ (k1 k2 k3 : ℤ), 0 ≤ (1 + 4 * k1) * Real.pi / (4 * ω) ∧ 
                          (1 + 4 * k2) * Real.pi / (4 * ω) ≤ Real.pi ∧ 
                          k1 < k2 ∧ k2 < k3 ∧
                          ∀ (k : ℤ), (0 ≤ (1 + 4 * k) * Real.pi / (4 * ω) ∧ (1 + 4 * k) * Real.pi / (4 * ω) ≤ Real.pi) → 
                          (k = k1 ∨ k = k2 ∨ k = k3)) : 
  (∃ (ω' : ℝ), ω' > 0 ∧ 2 * Real.pi / 3 = 2 * Real.pi / ω') ∧ 
  (∀ (x y : ℝ), 0 < x ∧ x < y ∧ y < Real.pi / 15 → f ω x < f ω y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l896_89645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_ratio_in_special_triangle_l896_89638

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.A = (1/6) * Real.pi ∧ t.B = (1/3) * Real.pi ∧ t.C = (1/2) * Real.pi

-- Theorem statement
theorem side_ratio_in_special_triangle (t : Triangle) 
  (h : validTriangle t) : 
  ∃ (k : ℝ), k > 0 ∧ t.a = k ∧ t.b = k * Real.sqrt 3 ∧ t.c = 2 * k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_ratio_in_special_triangle_l896_89638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_9_l896_89659

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else x^2

theorem f_value_9 (α : ℝ) (h : f α = 9) : α = -9 ∨ α = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_9_l896_89659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l896_89627

noncomputable section

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the circle
def circle_eq (a c x y : ℝ) : Prop := (x - a)^2 + y^2 = c^2 / 16

-- Define the tangent line
def tangent_line (a b c x y : ℝ) : Prop := a * x + b * y - a * c = 0

-- Define the asymptote
def asymptote (a b x y : ℝ) : Prop := y = (b / a) * x

-- Define perpendicularity of lines
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Define eccentricity
def eccentricity (c a : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : ∃ x y, hyperbola a b x y ∧ circle_eq a c x y)
  (h4 : ∃ x y, tangent_line a b c x y)
  (h5 : perpendicular (b / a) (-a / b)) :
  eccentricity c a = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l896_89627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l896_89604

noncomputable def f (x : ℝ) := Real.sqrt (2 * Real.cos x + 1)

theorem domain_of_f :
  {x : ℝ | ∃ (k : ℤ), 2 * π * ↑k - 2 * π / 3 ≤ x ∧ x ≤ 2 * π * ↑k + 2 * π / 3} =
  {x : ℝ | f x ≥ 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l896_89604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_equals_bullet_l896_89696

-- Define the symbols as variables
variable (delta square theta bullet : ℕ)

-- Define the conditions
axiom cond1 : delta + delta = square
axiom cond2 : square + delta = theta
axiom cond3 : theta = bullet + square + delta

-- Theorem to prove
theorem delta_equals_bullet : bullet = 3 * delta := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_equals_bullet_l896_89696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_given_equation_is_parabola_l896_89675

/-- 
A conic section is defined by the equation |x-3| = √((y+4)² + (x-1)²).
This function returns true if the conic section is a parabola, false otherwise.
-/
def isParabola (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ x y : ℝ, f x y = 0 ↔ (y + b)^2 = a * x + c)

/-- 
The main theorem stating that the given equation describes a parabola.
-/
theorem given_equation_is_parabola :
  isParabola (fun x y => |x - 3| - Real.sqrt ((y + 4)^2 + (x - 1)^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_given_equation_is_parabola_l896_89675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sum_l896_89605

-- Define the quadratic equation
noncomputable def quadratic (x : ℝ) : ℝ := 5 * x^2 - 11 * x + 2

-- Define the sum of roots
noncomputable def sum_of_roots (m n : ℕ) : ℝ := Real.sqrt (m : ℝ) / n

-- State the theorem
theorem quadratic_root_sum :
  ∃ (m n : ℕ), 
    (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ m)) ∧ 
    (∀ (x : ℝ), quadratic x = 0 → 
      sum_of_roots m n = (x + (11 - x) / 5)) ∧
    m + n = 126 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sum_l896_89605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l896_89640

noncomputable def f (A : ℝ) (x : ℝ) : ℝ := A * Real.sin (2 * x + Real.pi / 6)

theorem f_inequality (A : ℝ) (h : A > 0) : f A 2 < f A (-2) ∧ f A (-2) < f A 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l896_89640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graphical_method_correct_l896_89633

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ
deriving Inhabited

/-- Calculates the intersection point of a line segment with a vertical line -/
noncomputable def intersectionPoint (p1 p2 : Point) (x : ℝ) : Point :=
  { x := x
  , y := p1.y + (p2.y - p1.y) * (x - p1.x) / (p2.x - p1.x) }

/-- Constructs the sequence of B points as described in the problem -/
noncomputable def constructBPoints (dataPoints : List ℝ) : List Point :=
  let n := dataPoints.length
  let aPoints := List.zipWith (λ i y => Point.mk (2 * i) y) (List.range n) dataPoints
  let bPoints := List.range (n - 1) |>.scanl
    (λ acc i =>
      let p1 := if i = 0 then aPoints[0]! else acc
      let p2 := aPoints[i + 1]!
      intersectionPoint p1 p2 (i + 1))
    aPoints[0]!
  bPoints

/-- Calculates the arithmetic mean of a list of real numbers -/
noncomputable def arithmeticMean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

/-- Theorem stating that the graphical method correctly calculates the arithmetic mean -/
theorem graphical_method_correct (dataPoints : List ℝ) :
  let bPoints := constructBPoints dataPoints
  let n := dataPoints.length
  n > 1 → bPoints[n - 2]!.y = arithmeticMean dataPoints := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graphical_method_correct_l896_89633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l896_89649

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

-- Statement for part (I)
theorem part_one (a b : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) :
  f a + f b = f ((a + b) / (1 + a * b)) := by sorry

-- Statement for part (II)
theorem part_two :
  {m : ℝ | ∃ x, 0 ≤ x ∧ x < Real.pi / 2 ∧ f (Real.sin x ^ 2) + f (m * Real.cos x + 2 * m) = 0} =
  {m : ℝ | -1/2 < m ∧ m ≤ 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l896_89649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_speed_approx_l896_89614

/-- The speed of a sailboat given wind speed and sail angles -/
noncomputable def sailboat_speed (c : ℝ) (α : ℝ) (β : ℝ) : ℝ :=
  c * Real.sin α * Real.sin β

/-- Theorem: The speed of a sailboat is approximately 6.5 m/s under given conditions -/
theorem sailboat_speed_approx :
  let c := 12 -- wind speed in m/s
  let α := 50 * π / 180 -- sail angle relative to boat direction in radians
  let β := 45 * π / 180 -- sail angle relative to wind direction in radians
  ∃ ε > 0, abs (sailboat_speed c α β - 6.5) < ε :=
by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check sailboat_speed 12 (50 * Real.pi / 180) (45 * Real.pi / 180)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_speed_approx_l896_89614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l896_89646

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/x) * Real.log (Real.sqrt (x^2 - 3*x + 2) + Real.sqrt (-x^2 - 3*x + 4))

-- Define the set representing the domain
def domain : Set ℝ := Set.Icc (-4) 0 ∪ Set.Ioo 0 1

-- State the theorem
theorem f_domain : 
  ∀ x : ℝ, x ∈ domain ↔ 
    (x ≠ 0 ∧ 
     x^2 - 3*x + 2 ≥ 0 ∧ 
     -x^2 - 3*x + 4 ≥ 0 ∧ 
     Real.sqrt (x^2 - 3*x + 2) + Real.sqrt (-x^2 - 3*x + 4) > 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l896_89646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_cartesian_l896_89634

variable (R : ℝ) (t : ℝ)

noncomputable def x (R t : ℝ) : ℝ := R * Real.cos t
noncomputable def y (R t : ℝ) : ℝ := R * Real.sin t

theorem parametric_to_cartesian (R t : ℝ) :
  (x R t) ^ 2 + (y R t) ^ 2 = R ^ 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_cartesian_l896_89634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l896_89666

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  B = π / 4 →
  a = 3 * Real.sqrt 2 →
  Real.sin A ≠ 0 →
  Real.sin B ≠ 0 →
  a / Real.sin A = b / Real.sin B →
  b = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l896_89666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_on_imported_car_l896_89607

/-- Calculates the total tax for a two-tiered tax system on imported cars -/
noncomputable def calculate_tax (car_price : ℝ) (first_tier_level : ℝ) (first_tier_rate : ℝ) (second_tier_rate : ℝ) : ℝ :=
  let first_tier_tax := min car_price first_tier_level * first_tier_rate
  let second_tier_tax := max (car_price - first_tier_level) 0 * second_tier_rate
  first_tier_tax + second_tier_tax

/-- Theorem stating that the tax on a $30,000 car in the given two-tiered system is $5,500 -/
theorem tax_on_imported_car :
  calculate_tax 30000 10000 0.25 0.15 = 5500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_on_imported_car_l896_89607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_parallel_to_given_line_l896_89608

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the slope of the given line
def m : ℝ := 4

-- Theorem statement
theorem tangent_lines_parallel_to_given_line :
  ∃ (a b : ℝ), 
    (∀ x, (f' a) * (x - a) + f a = m * (x - a) + b) ∧
    (b = 0 ∨ b = -4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_parallel_to_given_line_l896_89608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_y_axis_l896_89612

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + Real.log x

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1 / x

-- Theorem statement
theorem tangent_perpendicular_to_y_axis (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ ¬ ∃ y : ℝ, |f_derivative a x| < y) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_y_axis_l896_89612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_fraction_theorem_l896_89639

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  short_base : ℝ
  long_base : ℝ
  side : ℝ
  angle : ℝ

/-- The fraction of the area closer to the longest base in a trapezoid -/
noncomputable def fraction_closest_to_long_base (t : Trapezoid) : ℝ :=
  1 / 2

/-- Theorem stating that for a trapezoid with specific dimensions, 
    the fraction of the area closer to the longest base is 1/2 -/
theorem trapezoid_fraction_theorem (t : Trapezoid) 
    (h1 : t.short_base = 120)
    (h2 : t.long_base = 180)
    (h3 : t.side = 130)
    (h4 : t.angle = Real.pi / 3) : 
  fraction_closest_to_long_base t = 1 / 2 := by
  sorry

#check trapezoid_fraction_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_fraction_theorem_l896_89639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_zeros_iff_a_pm_one_l896_89615

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.cos (2 * x) + a * Real.sin x

theorem nine_zeros_iff_a_pm_one :
  ∀ a : ℝ, (∃! (s : Finset ℝ), s.card = 9 ∧ 
    (∀ x ∈ s, 0 < x ∧ x < 6 * Real.pi ∧ f a x = 0) ∧
    (∀ x, 0 < x → x < 6 * Real.pi → f a x = 0 → x ∈ s)) ↔ 
  (a = 1 ∨ a = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_zeros_iff_a_pm_one_l896_89615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l896_89662

/-- Given a hyperbola with equation (y^2 / 16) - (x^2 / 9) = 1, 
    prove that the positive slope of its asymptotes is 4/3 -/
theorem hyperbola_asymptote_slope :
  ∃ (m : ℝ), m > 0 ∧
  (∀ (x y : ℝ), y^2 / 16 - x^2 / 9 = 1 →
    ∃ (k : ℝ), k = 1 ∨ k = -1 ∧ y = k * m * x) ∧
  m = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l896_89662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_quotient_l896_89635

/-- The cubic equation whose roots we're interested in -/
def cubic_equation (x : ℝ) : Prop := x^3 - 8*x^2 + 8*x = 1

/-- p is a root of the cubic equation -/
axiom p_is_root : ∃ p : ℝ, cubic_equation p

/-- q is a root of the cubic equation -/
axiom q_is_root : ∃ q : ℝ, cubic_equation q

/-- p is the largest root -/
axiom p_is_largest : ∃ p : ℝ, cubic_equation p ∧ ∀ x, cubic_equation x → x ≤ p

/-- q is the smallest root -/
axiom q_is_smallest : ∃ q : ℝ, cubic_equation q ∧ ∀ x, cubic_equation x → q ≤ x

/-- p and q are distinct -/
axiom p_neq_q : ∃ p q : ℝ, cubic_equation p ∧ cubic_equation q ∧ p ≠ q

theorem root_sum_quotient : ∃ p q : ℝ, cubic_equation p ∧ cubic_equation q ∧ p/q + q/p = 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_quotient_l896_89635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l896_89674

open Real

theorem integral_proof (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -2) :
  let f := λ y => (y^3 + 6*y^2 + 11*y + 7) / ((y+1)*(y+2)^3)
  let F := λ y => log (abs (y + 1)) + 1 / (2*(y + 2)^2)
  (deriv F x) = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l896_89674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_vertical_asymptote_l896_89678

/-- The function f(x) = (x+2)/(x^2 - 4) -/
noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x^2 - 4)

/-- The number of vertical asymptotes of f -/
def num_vertical_asymptotes : ℕ := 1

/-- Theorem stating that f has exactly one vertical asymptote -/
theorem f_has_one_vertical_asymptote :
  ∃! x : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, 
    0 < |y - x| ∧ |y - x| < δ → |f y| > 1/ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_vertical_asymptote_l896_89678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_135_deg_equals_3pi_div_4_l896_89682

/-- Conversion factor from degrees to radians -/
noncomputable def deg_to_rad : ℝ := Real.pi / 180

/-- 135 degrees in radians -/
noncomputable def angle_135_deg_in_rad : ℝ := 135 * deg_to_rad

theorem angle_135_deg_equals_3pi_div_4 : angle_135_deg_in_rad = 3 * Real.pi / 4 := by
  -- Unfold the definitions
  unfold angle_135_deg_in_rad deg_to_rad
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_135_deg_equals_3pi_div_4_l896_89682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_load_capacity_proof_l896_89613

/-- The load capacity formula for cylindrical columns -/
noncomputable def load_capacity (T H S : ℝ) : ℝ := (50 * T^3) / (S * H^2)

/-- Proof of the load capacity for given parameters -/
theorem load_capacity_proof (T H S : ℝ) (hT : T = 5) (hH : H = 10) (hS : S = 2) :
  load_capacity T H S = 31.25 := by
  -- Unfold the definition of load_capacity
  unfold load_capacity
  -- Substitute the given values
  rw [hT, hH, hS]
  -- Simplify the expression
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_load_capacity_proof_l896_89613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_satisfies_conditions_l896_89631

noncomputable section

/-- The distance between two points in a coordinate plane -/
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The perimeter of a triangle given the coordinates of its vertices -/
def triangle_perimeter (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  distance x₁ y₁ x₂ y₂ + distance x₂ y₂ x₃ y₃ + distance x₃ y₃ x₁ y₁

/-- The area of a triangle given the coordinates of its vertices -/
def triangle_area (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  (1/2) * abs ((x₁ - x₃) * (y₂ - y₃) - (x₂ - x₃) * (y₁ - y₃))

theorem no_triangle_satisfies_conditions :
  ∀ x y : ℝ, 
    triangle_perimeter 0 0 12 0 x y ≠ 60 ∨ 
    triangle_area 0 0 12 0 x y ≠ 240 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_satisfies_conditions_l896_89631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_l896_89618

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := x^3 + x^2 * k

-- State the theorem
theorem derivative_at_one (k : ℝ) : 
  deriv (f k) 1 = -3 ↔ k = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_l896_89618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pillows_is_1600_l896_89669

/-- Amount of fluffy foam material per pillow in pounds -/
noncomputable def fluffyFoamPerPillow : ℝ := 2

/-- Amount of microfiber per pillow in pounds -/
noncomputable def microfiberPerPillow : ℝ := 2.5

/-- Amount of cotton fabric per pillow in pounds -/
noncomputable def cottonFabricPerPillow : ℝ := 1.75

/-- Total amount of fluffy foam material available in pounds -/
noncomputable def totalFluffyFoam : ℝ := 6000

/-- Total amount of microfiber available in pounds -/
noncomputable def totalMicrofiber : ℝ := 4000

/-- Total amount of cotton fabric available in pounds -/
noncomputable def totalCottonFabric : ℝ := 3000

/-- Function to calculate the number of pillows that can be made from a given material -/
noncomputable def pillowsFromMaterial (totalMaterial : ℝ) (materialPerPillow : ℝ) : ℝ :=
  totalMaterial / materialPerPillow

/-- Theorem: The maximum number of pillows that can be made is 1600 -/
theorem max_pillows_is_1600 :
  min (pillowsFromMaterial totalFluffyFoam fluffyFoamPerPillow)
      (min (pillowsFromMaterial totalMicrofiber microfiberPerPillow)
           (pillowsFromMaterial totalCottonFabric cottonFabricPerPillow)) = 1600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pillows_is_1600_l896_89669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l896_89692

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  k : ℝ
  eq : (x y : ℝ) → Prop := fun x y ↦ x^2 / a^2 - y^2 / b^2 = k

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem stating the properties of the given hyperbola -/
theorem hyperbola_properties :
  ∃ (C : Hyperbola),
    C.eq 4 3 ∧
    C.a^2 = 12 ∧
    C.b^2 = 27 ∧
    C.k = 1 ∧
    eccentricity C = Real.sqrt 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l896_89692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_zero_l896_89622

theorem cos_2beta_zero (α β : Real) 
  (h1 : Real.sin α = Real.sqrt 5 / 5)
  (h2 : Real.sin (α - β) = -(Real.sqrt 10) / 10)
  (h3 : 0 < α ∧ α < Real.pi/2)
  (h4 : 0 < β ∧ β < Real.pi/2) :
  Real.cos (2 * β) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_zero_l896_89622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_five_hours_l896_89681

/-- Represents the time it takes to fill a tank given different pipe configurations -/
structure TankFilling where
  ab_time : ℝ  -- Time for pipes A and B to fill the tank
  cd_time : ℝ  -- Time for pipes C and D to fill the tank
  e_time : ℝ   -- Time for pipe E to fill the tank alone

/-- The time it takes to fill the tank with all pipes open, including E draining -/
noncomputable def fill_time (tf : TankFilling) : ℝ :=
  1 / (1 / tf.ab_time + 1 / tf.cd_time - 1 / tf.e_time)

/-- Theorem stating that for the given configuration, the fill time is 5 hours -/
theorem fill_time_is_five_hours (tf : TankFilling) 
  (hab : tf.ab_time = 6) 
  (hcd : tf.cd_time = 10) 
  (he : tf.e_time = 15) : 
  fill_time tf = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_five_hours_l896_89681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_required_for_reaction_l896_89623

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  nh4cl : Moles  -- Ammonium chloride
  h2o : Moles    -- Water
  nh4oh : Moles  -- Ammonium hydroxide
  hcl : Moles    -- Hydrochloric acid

/-- The balanced equation for the reaction -/
def balanced_equation (r : Reaction) : Prop :=
  r.nh4cl = r.h2o ∧ r.nh4cl = r.nh4oh ∧ r.nh4cl = r.hcl

/-- The theorem stating the amount of water required for the reaction -/
theorem water_required_for_reaction (r : Reaction) 
  (h1 : r.nh4cl = (2 : ℝ)) 
  (h2 : r.hcl = (2 : ℝ)) 
  (h3 : r.nh4oh = (2 : ℝ)) 
  (h4 : balanced_equation r) : 
  r.h2o = (2 : ℝ) := by
  sorry

-- Instance to allow using natural numbers as Moles
instance : Coe ℕ Moles where
  coe := λ n => (n : ℝ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_required_for_reaction_l896_89623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_and_uv_sum_l896_89660

-- Define the points
def X : ℝ × ℝ := (0, 0)
def Y : ℝ × ℝ := (10, 0)
def Z : ℝ × ℝ := (5, 3)
def Q : ℝ × ℝ := (5, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem sum_of_distances_and_uv_sum :
  ∃ (u v a b : ℕ), 
    distance X Q + distance Y Q + distance Z Q = u * Real.sqrt a + v * Real.sqrt b ∧
    u + v = 4 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_and_uv_sum_l896_89660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l896_89679

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l896_89679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_equals_four_to_four_thirds_l896_89665

-- Define the sequence of exponents
def exponent_sequence (n : ℕ) : ℚ := 1 / (2^n : ℚ)

-- Define the sequence of bases
def base_sequence (n : ℕ) : ℕ := 4^n

-- Define the nth term of the product
noncomputable def product_term (n : ℕ) : ℝ := (base_sequence n : ℝ) ^ (exponent_sequence n : ℝ)

-- State the theorem
theorem infinite_product_equals_four_to_four_thirds :
  (∏' n : ℕ, product_term n) = (4 : ℝ)^(4/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_equals_four_to_four_thirds_l896_89665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_cos_over_x_l896_89684

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.cos x / x

-- State the theorem
theorem derivative_of_cos_over_x :
  deriv f = λ x ↦ -(x * Real.sin x + Real.cos x) / (x^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_cos_over_x_l896_89684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_half_l896_89668

open Real

theorem series_sum_equals_half :
  let S : ℕ → ℝ := λ n => (3 : ℝ)^n / ((9 : ℝ)^n - 1)
  ∑' n, S n = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_half_l896_89668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l896_89636

theorem triangle_angle_calculation (A B C : ℝ) (h1 : A + B + C = π) 
  (h2 : Real.tan A = 1/2) (h3 : Real.cos B = 3 * Real.sqrt 10 / 10) : C = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l896_89636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_hyperbola_standard_equation_l896_89629

/-- An ellipse with specified properties has a standard equation -/
theorem ellipse_standard_equation :
  ∀ (e : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ e ↔ ∃ (t : ℝ), x = 2 * Real.cos t ∧ y = 4 * Real.sin t) →
  (2, 0) ∈ e →
  ∃ (a b : ℝ), a = 2 * b ∧ ∀ (x y : ℝ), (x, y) ∈ e ↔ (y^2 / 16) + (x^2 / 4) = 1 :=
by
  sorry

/-- A hyperbola with specified properties has a standard equation -/
theorem hyperbola_standard_equation :
  ∀ (h : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), x + 2*y = 0 → (x, y) ∈ h) →
  (2, 2) ∈ h →
  ∃ (a b : ℝ), ∀ (x y : ℝ), (x, y) ∈ h ↔ (y^2 / 3) - (x^2 / 12) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_hyperbola_standard_equation_l896_89629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_polar_line_l896_89637

/-- The distance from a point to a line in polar coordinates -/
noncomputable def distance_point_to_polar_line (x y : ℝ) (a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- The line equation in polar coordinates -/
def polar_line_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ + Real.sin θ) = 2

theorem distance_to_polar_line :
  distance_point_to_polar_line 1 0 1 1 (-2) = Real.sqrt 2 / 2 := by
  sorry

#check distance_to_polar_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_polar_line_l896_89637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_wins_first_given_conditions_l896_89642

/-- A game series between two teams A and B -/
structure GameSeries where
  /-- The probability of team A winning a single game -/
  prob_A_win : ℚ
  /-- The probability of team B winning a single game -/
  prob_B_win : ℚ
  /-- The number of games needed to win the series -/
  games_to_win : ℕ

/-- The conditions of the specific game series in the problem -/
def problem_series : GameSeries where
  prob_A_win := 1/2
  prob_B_win := 1/2
  games_to_win := 3

/-- The event that team B wins the first game -/
def B_wins_first (series : GameSeries) : Prop := sorry

/-- The event that team B wins the second game -/
def B_wins_second (series : GameSeries) : Prop := sorry

/-- The event that team A wins the series -/
def A_wins_series (series : GameSeries) : Prop := sorry

/-- The probability of an event given the game series -/
noncomputable def prob (series : GameSeries) (event : Prop) : ℚ := sorry

/-- The conditional probability of event A given event B -/
noncomputable def conditional_prob (series : GameSeries) (A B : Prop) : ℚ := 
  (prob series (A ∧ B)) / (prob series B)

/-- The main theorem to be proved -/
theorem prob_B_wins_first_given_conditions (series : GameSeries) : 
  conditional_prob series 
    (B_wins_first series) 
    (B_wins_second series ∧ A_wins_series series) = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_wins_first_given_conditions_l896_89642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l896_89661

theorem vector_magnitude_proof {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (ha : ‖a‖ = 3) (hb : ‖b‖ = 2) (hab : ‖a + b‖ = 4) :
  ‖a - b‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l896_89661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l896_89630

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt ((a^2 + b^2) / a^2)

/-- The slope of the asymptote of a hyperbola -/
noncomputable def asymptote_slope (a b : ℝ) : ℝ := b / a

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_intersect : ∀ (m : ℝ), m ≠ 0 → ∃! (x y : ℝ), y = 2*x + m ∧ x^2/a^2 - y^2/b^2 = 1) :
  eccentricity a b = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l896_89630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l896_89655

def p₁ : Prop := ∀ x : ℝ, Monotone (fun x => Real.exp (x * Real.log 2) - Real.exp (-x * Real.log 2))

def p₂ : Prop := ∀ x : ℝ, Antitone (fun x => Real.exp (x * Real.log 2) + Real.exp (-x * Real.log 2))

def q₁ : Prop := p₁ ∨ p₂

def q₂ : Prop := p₁ ∧ p₂

def q₃ : Prop := (¬p₁) ∨ p₂

def q₄ : Prop := p₁ ∨ (¬p₂)

theorem problem_solution : q₁ ∧ q₄ ∧ ¬q₂ ∧ ¬q₃ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l896_89655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l896_89654

-- Define the constants
noncomputable def a : ℝ := Real.rpow 2 0.8
noncomputable def b : ℝ := Real.rpow (1/2) (-1.2)
noncomputable def c : ℝ := 2 * (Real.log 2 / Real.log 5)

-- State the theorem
theorem relationship_abc : c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l896_89654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l896_89621

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ p = 6 * Real.pi ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ (x : ℝ), f x ≤ M ∧ 
    ∃ (y : ℝ), f y = M) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l896_89621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l896_89643

theorem log_inequality : ∃ (a b c : ℝ), 
  a = Real.log 2 / Real.log 3 ∧ 
  b = Real.log 3 / Real.log 2 ∧ 
  c = Real.log 5 / Real.log 2 ∧ 
  a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l896_89643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l896_89658

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Theorem about the area of a triangle formed by a point on a hyperbola and its foci -/
theorem hyperbola_triangle_area 
  (h : Hyperbola) 
  (f1 f2 p : Point) 
  (h_eq : h.a^2 - h.b^2 = 25) 
  (h_on : isOnHyperbola h p) 
  (h_foci : f1 = Point.mk (-5) 0 ∧ f2 = Point.mk 5 0) 
  (h_ratio : 3 * distance p f1 = 4 * distance p f2) :
  (1/2) * distance f1 p * distance f2 p = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l896_89658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_equality_find_m_value_l896_89698

-- Part 1
theorem logarithm_expression_equality : 
  Real.log 5^2 + (2/3) * Real.log 8 + Real.log 5 * Real.log 20 + (Real.log 2)^2 = 3 := by sorry

-- Part 2
theorem find_m_value (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_equality_find_m_value_l896_89698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equality_l896_89602

theorem sequence_equality (x : Fin 2021 → ℝ) 
  (h_positive : ∀ i, x i > 0)
  (h_recurrence : ∀ i : Fin 2020, x (Fin.succ i) = (x i ^ 3 + 2) / (3 * x i ^ 2))
  (h_cycle : x ⟨2020, by norm_num⟩ = x ⟨0, by norm_num⟩) :
  ∀ i, x i = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equality_l896_89602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_divisibility_l896_89695

theorem perfect_square_divisibility (x y : ℕ) (hx : x > 0) (hy : y > 0)
  (h : (x^2 + y^2 - x) % (2*x*y) = 0) : 
  ∃ (n : ℕ), x = n^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_divisibility_l896_89695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_zero_possible_l896_89657

theorem existence_of_zero_possible (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b)) (h_prod : f a * f b > 0) :
  (∃ c ∈ Set.Icc a b, f c = 0) ∨ (∀ c ∈ Set.Icc a b, f c ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_zero_possible_l896_89657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l896_89677

theorem sqrt_calculations :
  (Real.sqrt 18 * Real.sqrt 6 = 6 * Real.sqrt 3) ∧
  (Real.sqrt 8 - Real.sqrt 2 + 2 * Real.sqrt (1/2) = 3 * Real.sqrt 2) ∧
  (Real.sqrt 12 * (Real.sqrt 9 / 3) / (Real.sqrt 3 / 3) = 6) ∧
  ((Real.sqrt 7 + Real.sqrt 5) * (Real.sqrt 7 - Real.sqrt 5) = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l896_89677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_triangle_area_l896_89650

-- Define the function g on the domain {x₁, x₂, x₃}
def g (x : ℝ) : ℝ := sorry

-- Define the area of a triangle given three points
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem transformed_triangle_area 
  (x₁ x₂ x₃ : ℝ) 
  (h₁ : triangleArea (x₁, g x₁) (x₂, g x₂) (x₃, g x₃) = 45) :
  triangleArea (x₁/3, 3 * g x₁) (x₂/3, 3 * g x₂) (x₃/3, 3 * g x₃) = 45 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_triangle_area_l896_89650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l896_89687

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a²cos(A)sin(B) = b²sin(A)cos(B), then the triangle is either
    isosceles (A = B) or right-angled (A + B = π/2) -/
theorem triangle_shape (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a^2 * Real.cos A * Real.sin B = b^2 * Real.sin A * Real.cos B →
  A = B ∨ A + B = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l896_89687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_range_l896_89644

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x^2 - 6 * x + a^2 + 1
  else x^(5 - 2*a)

-- State the theorem
theorem monotone_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Ioo (5/2) 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_range_l896_89644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_from_origin_l896_89603

/-- Definition of a 2D plane --/
def Plane := ℝ × ℝ

/-- Definition of a circle through three points --/
def Circle := Plane → Prop

/-- Definition to calculate the length of a tangent from a point to a circle --/
noncomputable def Circle.tangentLength (circle : Circle) (point : Plane) : ℝ := sorry

/-- Given a circle passing through three points, calculate the length of the tangent from the origin --/
theorem tangent_length_from_origin (c : ℝ) : 
  ∃ (A B C : Plane) (circle : Circle),
    A = (1, 3) ∧
    B = (4, 7) ∧
    C = (6, 11) ∧
    circle A ∧ circle B ∧ circle C ∧
    Circle.tangentLength circle (0, 0) = Real.sqrt c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_from_origin_l896_89603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_distance_l896_89688

/-- Represents the scenario of a car trip with an accident -/
structure CarTrip where
  s : ℝ  -- Original speed of the car in mph
  D : ℝ  -- Total distance of the trip in miles

/-- Calculates the total travel time for the first scenario -/
noncomputable def total_time_scenario1 (trip : CarTrip) : ℝ :=
  2 + 1/3 + (6 * (trip.D - 2 * trip.s)) / (5 * trip.s)

/-- Calculates the total travel time for the second scenario -/
noncomputable def total_time_scenario2 (trip : CarTrip) : ℝ :=
  (2 * trip.s + 60) / trip.s + 1/3 + (6 * (trip.D - 2 * trip.s - 60)) / (5 * trip.s)

/-- Theorem stating the total distance of the trip -/
theorem trip_distance : ∃ (trip : CarTrip), 
  trip.s > 0 ∧ 
  total_time_scenario1 trip = trip.D / trip.s + 2 ∧ 
  total_time_scenario2 trip = trip.D / trip.s + 1.5 ∧ 
  trip.D = 280 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_distance_l896_89688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_expression_range_l896_89673

/-- Predicate for arithmetic sequence -/
def is_arithmetic_seq (x a₁ a₂ y : ℝ) : Prop :=
  ∃ d : ℝ, a₁ = x + d ∧ a₂ = x + 2*d ∧ y = x + 3*d

/-- Predicate for geometric sequence -/
def is_geometric_seq (x b₁ b₂ y : ℝ) : Prop :=
  ∃ r : ℝ, b₁ = x * r ∧ b₂ = x * r^2 ∧ y = x * r^3

/-- Given real numbers forming arithmetic and geometric sequences, prove the range of a specific expression -/
theorem sequence_expression_range (x y : ℝ) (a₁ a₂ b₁ b₂ : ℝ) 
  (h_arith : is_arithmetic_seq x a₁ a₂ y)
  (h_geom : is_geometric_seq x b₁ b₂ y) :
  (a₁ + a₂)^2 / (b₁ * b₂) ∈ Set.Iic 0 ∪ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_expression_range_l896_89673
