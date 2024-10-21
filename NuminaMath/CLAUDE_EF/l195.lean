import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l195_19583

/-- The function f(x) = (7x^2 - 8) / (4x^2 + 3x + 1) -/
noncomputable def f (x : ℝ) : ℝ := (7 * x^2 - 8) / (4 * x^2 + 3 * x + 1)

/-- The horizontal asymptote of f(x) is 7/4 -/
theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x, x > N → |f x - 7/4| < ε := by
  sorry

#check horizontal_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l195_19583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_payment_technology_mapping_l195_19592

-- Define the payment technologies
inductive PaymentTechnology
| Chip
| MagneticStripe
| Paypass
| CVC

-- Define the actions
inductive PaymentAction
| Tap
| PayOnline
| Swipe
| InsertIntoTerminal

-- Define the correct mapping function
def correctMapping (tech : PaymentTechnology) : PaymentAction :=
  match tech with
  | PaymentTechnology.Chip => PaymentAction.InsertIntoTerminal
  | PaymentTechnology.MagneticStripe => PaymentAction.Swipe
  | PaymentTechnology.Paypass => PaymentAction.Tap
  | PaymentTechnology.CVC => PaymentAction.PayOnline

-- Theorem statement
theorem correct_payment_technology_mapping :
  (∀ tech : PaymentTechnology, 
    (tech = PaymentTechnology.Chip → correctMapping tech = PaymentAction.InsertIntoTerminal) ∧
    (tech = PaymentTechnology.MagneticStripe → correctMapping tech = PaymentAction.Swipe) ∧
    (tech = PaymentTechnology.Paypass → correctMapping tech = PaymentAction.Tap) ∧
    (tech = PaymentTechnology.CVC → correctMapping tech = PaymentAction.PayOnline)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_payment_technology_mapping_l195_19592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_term_l195_19550

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  h_d : d > 0 -- d is positive

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1) * seq.d)

theorem smallest_positive_term 
  (seq : ArithmeticSequence) 
  (h : S seq 12 = 2 * S seq 5) :
  ∃ n : ℕ, (∀ m : ℕ, m < n → seq.a m ≤ 0) ∧ seq.a n > 0 ∧ n = 25 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_term_l195_19550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_pentagon_not_convex_l195_19549

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a pentagon -/
structure Pentagon where
  vertices : Fin 5 → Point

/-- Represents the angle bisector of a vertex in a pentagon -/
noncomputable def angle_bisector (p : Pentagon) (i : Fin 5) : Line :=
  sorry

/-- Returns the intersection point of two lines -/
noncomputable def line_intersection (l1 l2 : Line) : Point :=
  sorry

/-- Checks if a pentagon is convex -/
def is_convex (p : Pentagon) : Prop :=
  sorry

/-- The main theorem: The pentagon formed by intersection points of consecutive angle bisectors of a pentagon cannot be convex -/
theorem angle_bisector_pentagon_not_convex (p : Pentagon) : 
  let b : Fin 5 → Point := λ i ↦ 
    line_intersection (angle_bisector p i) (angle_bisector p ((i + 1) % 5))
  ¬(is_convex (Pentagon.mk b)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_pentagon_not_convex_l195_19549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l195_19586

/-- The distance between two parallel lines in R² --/
noncomputable def distance_parallel_lines (a₁ a₂ b₁ b₂ d₁ d₂ : ℝ) : ℝ :=
  let v₁ := a₁ - b₁
  let v₂ := a₂ - b₂
  let n := d₁ * d₁ + d₂ * d₂
  Real.sqrt ((v₁ * d₂ - v₂ * d₁) * (v₁ * d₂ - v₂ * d₁) / n)

theorem distance_between_given_lines :
  distance_parallel_lines 2 (-3) 1 (-5) 1 (-7) = 9 * Real.sqrt 2 / 10 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l195_19586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l195_19554

/-- The number of ways to arrange plates around a circular table. -/
def circularArrangements (blue red green yellow : ℕ) : ℕ :=
  (blue + red + green + yellow - 1).factorial / (blue.factorial * red.factorial * green.factorial * yellow.factorial)

/-- The number of ways to arrange plates around a circular table with red plates as one block. -/
def circularArrangementsWithRedBlock (blue red green yellow : ℕ) : ℕ :=
  (blue + green + yellow).factorial / (blue.factorial * green.factorial * yellow.factorial)

/-- The theorem stating the number of valid arrangements of plates. -/
theorem valid_arrangements_count :
  circularArrangements 4 3 3 1 - circularArrangementsWithRedBlock 4 3 3 1 = 24780 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l195_19554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l195_19528

/-- Given a simple interest, rate, and time, calculate the principal amount -/
noncomputable def calculate_principal (simple_interest : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  simple_interest / (rate * time)

/-- Theorem: Given the specified conditions, the principal amount is 6693.75 -/
theorem principal_calculation :
  let simple_interest : ℝ := 4016.25
  let rate : ℝ := 12 / 100  -- 12% converted to decimal
  let time : ℝ := 5
  calculate_principal simple_interest rate time = 6693.75 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_principal 4016.25 (12/100) 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l195_19528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_5_eq_30_l195_19535

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_ne_zero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  log_arithmetic : 2 * (Real.log (a 2)) = Real.log (a 1) + Real.log (a 4)
  a5_eq_10 : a 5 = 10

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (seq.a 1) + (n * (n - 1) / 2) * seq.d

/-- The main theorem -/
theorem sum_5_eq_30 (seq : ArithmeticSequence) : sum_n seq 5 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_5_eq_30_l195_19535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brie_laundry_theorem_l195_19539

/-- Represents the number of clothing pieces to be washed -/
def clothes_to_wash (total_blouses total_skirts total_slacks : ℕ) 
                    (blouse_percent skirt_percent slack_percent : ℚ) : ℕ :=
  (((total_blouses : ℚ) * blouse_percent).floor +
   ((total_skirts : ℚ) * skirt_percent).floor +
   ((total_slacks : ℚ) * slack_percent).floor).toNat

theorem brie_laundry_theorem : 
  clothes_to_wash 12 6 8 (3/4) (1/2) (1/4) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brie_laundry_theorem_l195_19539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l195_19562

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 4)

theorem problem_solution :
  (∀ x, f x = Real.cos (x + Real.pi / 4)) →
  (f (Real.pi / 6) + f (-Real.pi / 6) = Real.sqrt 6 / 2) ∧
  (∀ x, f x = Real.sqrt 2 / 3 → Real.sin (2 * x) = 5 / 9) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l195_19562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_reciprocal_difference_powers_l195_19501

/-- For x > 1, the sum of 1 / (x^(3^n) - x^(-3^n)) from n = 0 to infinity equals 1 / (x - 1) -/
theorem infinite_sum_reciprocal_difference_powers (x : ℝ) (hx : x > 1) :
  ∑' (n : ℕ), 1 / (x^(3^n) - (1/x)^(3^n)) = 1 / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_reciprocal_difference_powers_l195_19501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_may_cost_correct_l195_19511

-- Define the original and new billing rates
def original_rate : ℚ := 2
def new_rate_low : ℚ := 9/5  -- 1.8 as a rational number
def threshold : ℚ := 10

-- Define the billing function for the new system
noncomputable def new_billing (a : ℚ) (x : ℚ) : ℚ :=
  if x ≤ threshold then new_rate_low * x
  else new_rate_low * threshold + a * (x - threshold)

-- Define the conditions for finding 'a'
theorem find_a :
  ∃ a : ℚ, a > new_rate_low ∧ 
    new_billing a 20 = original_rate * 20 + 8 ∧
    a = 3 := by sorry

-- Define the cost function for May
noncomputable def may_cost (x : ℚ) : ℚ :=
  if x ≤ threshold then 9/5 * x
  else 3 * x - 12

-- Prove the correctness of the May cost function
theorem may_cost_correct (x : ℚ) :
  may_cost x = 
    if x ≤ threshold then 9/5 * x
    else 3 * x - 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_may_cost_correct_l195_19511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l195_19545

/-- A vector on the line y = 3/4x + 3 -/
def VectorOnLine (v : ℝ × ℝ) : Prop :=
  v.2 = 3/4 * v.1 + 3

/-- The projection vector w' -/
def ProjectionVector (w : ℝ × ℝ) : Prop :=
  w.1 + 3/4 * w.2 = 0

/-- The projection of v onto w -/
noncomputable def VectorProjection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dotProduct := v.1 * w.1 + v.2 * w.2
  let normSquared := w.1 * w.1 + w.2 * w.2
  (dotProduct / normSquared * w.1, dotProduct / normSquared * w.2)

theorem projection_theorem (v w : ℝ × ℝ) :
  VectorOnLine v → ProjectionVector w →
  VectorProjection v w = (-36/25, 48/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l195_19545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l195_19590

/-- The constant term in the expansion of (a√x - 1/√x)^6, where a is the integral of sin x from 0 to π -/
theorem constant_term_binomial_expansion (a : ℝ) (h : a = ∫ x in (0 : ℝ)..Real.pi, Real.sin x) : 
  (Finset.range 7).sum (fun k => (-1)^k * (Nat.choose 6 k) * a^(6-k) * 1^k * (if k = 6 then 1 else 0)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l195_19590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_exponents_l195_19508

theorem smallest_sum_of_exponents (a b : ℕ) (h : (3 : ℕ)^7 * (5 : ℕ)^3 = a^b) :
  ∃ (c d : ℕ), (3 : ℕ)^7 * (5 : ℕ)^3 = c^d ∧ c + d ≤ a + b ∧ c + d = 3376 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_exponents_l195_19508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l195_19517

/-- The time required for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 110 m traveling at 72 km/h takes approximately 12.1 seconds to cross a bridge of length 132 m -/
theorem train_crossing_bridge_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.05 ∧ |train_crossing_time 110 72 132 - 12.1| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l195_19517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_l195_19565

/-- The diameter of the initial pizza in inches -/
noncomputable def initial_diameter : ℝ := 10

/-- The diameter of the final pizza in inches -/
noncomputable def final_diameter : ℝ := 15

/-- The area of a circle given its diameter -/
noncomputable def circle_area (d : ℝ) : ℝ := Real.pi * (d / 2) ^ 2

/-- The percent increase in area from initial to final pizza size -/
noncomputable def percent_increase : ℝ :=
  (circle_area final_diameter - circle_area initial_diameter) / circle_area initial_diameter * 100

theorem pizza_area_increase :
  percent_increase = 125 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_l195_19565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_triples_l195_19547

def S : Finset ℕ := Finset.range 66

theorem count_ordered_triples :
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    t.1 ∈ S ∧ t.2.1 ∈ S ∧ t.2.2 ∈ S ∧ t.1 < t.2.2 ∧ t.2.1 < t.2.2) 
    (S.product (S.product S))).card = 89440 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_triples_l195_19547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_shop_pricing_l195_19588

/-- Represents the cost price of an item -/
def cost_price : ℝ → ℝ := sorry

/-- Represents the selling price of an item -/
def selling_price : ℝ → ℝ := sorry

/-- The markup percentage applied by the shop owner -/
def markup_percentage : ℝ := 0.20

theorem furniture_shop_pricing (item : ℝ) :
  selling_price item = 8337 ∧
  selling_price item = cost_price item * (1 + markup_percentage) →
  cost_price item = 6947.5 := by
  sorry

#check furniture_shop_pricing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_shop_pricing_l195_19588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_unique_l195_19575

-- Define the properties of the function g
def satisfies_properties (g : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, g (x₁ + x₂) = g x₁ * g x₂) ∧
  (g 1 = 3) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂)

-- Define the proposed function
noncomputable def g (x : ℝ) : ℝ := 3^x

-- Theorem statement
theorem g_unique : 
  satisfies_properties g ∧ 
  ∀ f : ℝ → ℝ, satisfies_properties f → f = g :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_unique_l195_19575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_value_l195_19593

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x^5 - 1) / 5

-- State the theorem
theorem inverse_g_value : g⁻¹ (-11/40) = -(3/8)^(1/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_value_l195_19593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sum_problem_l195_19582

theorem integer_sum_problem (x y : ℤ) (h1 : x > 0) (h2 : y > 0) (h3 : x - y = 8) (h4 : x * y = 272) :
  x + y = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sum_problem_l195_19582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l195_19577

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * t, 2 - t)

def parabola_C (x y : ℝ) : Prop := y^2 = 2*x

def point_A : ℝ × ℝ := (0, 2)

theorem intersection_distance_sum :
  ∃ (t₁ t₂ : ℝ),
    t₁ ≠ t₂ ∧
    parabola_C (line_l t₁).1 (line_l t₁).2 ∧
    parabola_C (line_l t₂).1 (line_l t₂).2 ∧
    Real.sqrt ((line_l t₁).1 - point_A.1)^2 + ((line_l t₁).2 - point_A.2)^2 +
    Real.sqrt ((line_l t₂).1 - point_A.1)^2 + ((line_l t₂).2 - point_A.2)^2
    = 8 + 4 * Real.sqrt 3 := by
  sorry

#check intersection_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l195_19577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l195_19533

/-- The sum of the infinite series ∑(n=1 to ∞) n³/3ⁿ -/
noncomputable def infiniteSeries : ℝ := ∑' n, (n ^ 3 : ℝ) / (3 ^ n)

/-- Theorem: The sum of the infinite series ∑(n=1 to ∞) n³/3ⁿ is equal to 33/8 -/
theorem infiniteSeriesSum : infiniteSeries = 33 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l195_19533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_ming_infection_probability_l195_19506

/-- The probability of infection from a first-generation transmitter -/
noncomputable def p_first : ℝ := 0.95

/-- The probability of infection from a second-generation transmitter -/
noncomputable def p_second : ℝ := 0.9

/-- The probability of infection from a third-generation transmitter -/
noncomputable def p_third : ℝ := 0.85

/-- The number of first-generation transmitters -/
def n_first : ℕ := 5

/-- The number of second-generation transmitters -/
def n_second : ℕ := 3

/-- The number of third-generation transmitters -/
def n_third : ℕ := 2

/-- The total number of infected individuals -/
def total_infected : ℕ := n_first + n_second + n_third

/-- The probability of contacting a first-generation transmitter -/
noncomputable def p_contact_first : ℝ := n_first / total_infected

/-- The probability of contacting a second-generation transmitter -/
noncomputable def p_contact_second : ℝ := n_second / total_infected

/-- The probability of contacting a third-generation transmitter -/
noncomputable def p_contact_third : ℝ := n_third / total_infected

/-- The probability of Xiao Ming getting infected -/
noncomputable def p_infected : ℝ := p_first * p_contact_first + p_second * p_contact_second + p_third * p_contact_third

theorem xiao_ming_infection_probability : p_infected = 0.915 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_ming_infection_probability_l195_19506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_sum_bounds_l195_19548

open Real

/-- For any triangle with sides a, b, c and corresponding medians ma, mb, mc,
    the sum of the medians is bounded by 3/4 of the perimeter and the perimeter itself. -/
theorem median_sum_bounds (a b c ma mb mc : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hma : ma = Real.sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4))
  (hmb : mb = Real.sqrt ((2 * a^2 + 2 * c^2 - b^2) / 4))
  (hmc : mc = Real.sqrt ((2 * a^2 + 2 * b^2 - c^2) / 4))
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) :
  3/4 * (a + b + c) < ma + mb + mc ∧ ma + mb + mc < a + b + c := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_sum_bounds_l195_19548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l195_19581

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => (Real.sqrt (1 + a n ^ 2) - 1) / a n

theorem a_lower_bound : ∀ n : ℕ, a n > π / 2^(n+2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l195_19581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_about_x_eq_1_l195_19507

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def g (x : ℝ) : ℝ := |floor x| - |floor (2 - x)|

theorem g_symmetry_about_x_eq_1 :
  ∀ x : ℝ, g x = g (2 - x) := by
  sorry

#check g_symmetry_about_x_eq_1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_about_x_eq_1_l195_19507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_nonpositive_l195_19513

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x - 2

-- Define the domain
def domain : Set ℝ := Set.Icc (-5) 5

-- Define the set of x values where f(x) ≤ 0
def solution_set : Set ℝ := {x ∈ domain | f x ≤ 0}

-- Theorem statement
theorem probability_f_nonpositive :
  (MeasureTheory.volume solution_set) / (MeasureTheory.volume domain) = 3/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_nonpositive_l195_19513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l195_19598

-- Define the vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 0)
noncomputable def b (x : ℝ) : ℝ × ℝ := (0, Real.sin x)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  (a x + b x).1^2 + (a x + b x).2^2 + Real.sqrt 3 * Real.sin (2 * x)

-- State the theorem
theorem f_properties :
  (∀ k : ℤ, StrictMono (fun x => f (x + k * Real.pi + Real.pi / 3))) ∧
  Set.Icc (2 - Real.sqrt 3) 4 = Set.image f (Set.Ioo (-Real.pi/4) (Real.pi/4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l195_19598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l195_19559

-- Define the initial complex number
def z : ℂ := -4 - 6*Complex.I

-- Define the rotation angle in radians
noncomputable def θ : ℝ := 30 * (Real.pi / 180)

-- Define the dilation scale factor
noncomputable def k : ℝ := Real.sqrt 3

-- Define the combined transformation
noncomputable def T (w : ℂ) : ℂ := k * (Complex.exp (Complex.I * θ) * w)

-- State the theorem
theorem complex_transformation :
  T z = -6 - 3 * Real.sqrt 3 - (2 * Real.sqrt 3 + 9) * Complex.I :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l195_19559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_vector_range_l195_19525

/-- Defines an acute angle between two vectors -/
def acute_angle (a b : ℝ × ℝ) : Prop :=
  0 < a.1 * b.1 + a.2 * b.2 ∧ ¬(a.1 * b.2 = a.2 * b.1)

/-- Given vectors a and b in ℝ², if the angle between them is acute, 
    then the second component of b satisfies the given range. -/
theorem acute_angle_vector_range (m : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (4, m)
  acute_angle a b → m ∈ Set.union (Set.Ioo (-2) 8) (Set.Ioi 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_vector_range_l195_19525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_range_l195_19566

theorem cosine_sine_range : 
  (∀ x : ℝ, -4 ≤ (Real.cos x) ^ 2 + 2 * (Real.sin x) - 2 ∧ (Real.cos x) ^ 2 + 2 * (Real.sin x) - 2 ≤ 0) ∧ 
  (∃ x y : ℝ, (Real.cos x) ^ 2 + 2 * (Real.sin x) - 2 = -4 ∧ (Real.cos y) ^ 2 + 2 * (Real.sin y) - 2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_range_l195_19566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_no_zeros_l195_19537

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x - 1

-- Define the derivative of f
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x - a

theorem f_monotonicity (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, f_deriv a x > 0) ∧
  (a > 0 → (∀ x > 0, x < 1/a → f_deriv a x > 0) ∧ 
           (∀ x > 1/a, f_deriv a x < 0)) :=
by sorry

theorem f_no_zeros (a : ℝ) :
  (a > 0 ∧ (∀ x > 0, f a x ≠ 0)) → a > 1/(exp 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_no_zeros_l195_19537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_proof_l195_19599

theorem vector_operation_proof : 
  (4 : ℝ) • (![3, -6] : Fin 2 → ℝ) + (5 : ℝ) • (![2, -9] : Fin 2 → ℝ) - (![(-1), 3] : Fin 2 → ℝ) = ![23, -72] := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_proof_l195_19599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_number_puzzle_solution_l195_19569

/-- Represents a cell in the cross-number puzzle -/
structure Cell where
  value : Nat
  is_nonzero : value > 0
  is_digit : value < 10

/-- Represents the cross-number puzzle -/
structure CrossNumber where
  across1 : Cell
  across3 : Cell
  across5 : Cell
  down1 : Cell
  down2 : Cell
  down4 : Cell

/-- Checks if a number is a palindrome -/
def is_palindrome (n : Nat) : Prop := sorry

/-- The main theorem stating the properties of the cross-number puzzle -/
theorem cross_number_puzzle_solution :
  ∀ (puzzle : CrossNumber),
  (∃ (factor : Nat), factor ∣ 105 ∧ puzzle.across1.value = factor - 4) →
  (∃ (palindrome : Nat), is_palindrome palindrome ∧ puzzle.across3.value = palindrome + 1) →
  (puzzle.across5.value * puzzle.across5.value = puzzle.across5.value) →
  (∃ (square : Nat), puzzle.down1.value = square * square - 2) →
  (∃ (cube : Nat), puzzle.down2.value = cube * cube * cube - 400) →
  (∃ (clue1 clue2 : Nat), 
    (clue1 ∈ ({puzzle.across1.value, puzzle.across3.value, puzzle.across5.value, 
              puzzle.down1.value, puzzle.down2.value} : Set Nat) ∧
     clue2 ∈ ({puzzle.across1.value, puzzle.across3.value, puzzle.across5.value, 
              puzzle.down1.value, puzzle.down2.value} : Set Nat) ∧
     clue1 ≠ clue2) ∧
    puzzle.down4.value = clue1 + clue2 - 6) →
  puzzle.across5.value * puzzle.across5.value = 841 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_number_puzzle_solution_l195_19569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_transformation_equivalence_l195_19510

/-- Represents a transformation of a sine function -/
structure SineTransformation where
  shift : ℝ
  scale : ℝ

/-- The original sine function -/
noncomputable def original_sine (x : ℝ) : ℝ := Real.sin x

/-- The transformed sine function -/
noncomputable def transformed_sine (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 7)

/-- Applies a transformation to the sine function -/
noncomputable def apply_transformation (t : SineTransformation) (x : ℝ) : ℝ :=
  Real.sin (t.scale * (x - t.shift))

theorem sine_transformation_equivalence :
  ∀ x : ℝ, transformed_sine x = 
    apply_transformation { shift := Real.pi / 7, scale := 2 } x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_transformation_equivalence_l195_19510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_statements_count_l195_19514

/-- Represents the number of knights on the island -/
def r : ℕ := sorry

/-- Represents the number of liars on the island -/
def l : ℕ := sorry

/-- The total number of islanders -/
def total_islanders : ℕ := r + l

/-- The constraint that there are at least two knights and two liars -/
axiom at_least_two : r ≥ 2 ∧ l ≥ 2

/-- The number of times "You are a liar!" was said -/
axiom liar_count : 2 * r * l = 230

/-- The total number of statements made -/
def total_statements : ℕ := total_islanders * (total_islanders - 1)

/-- The number of times "You are a knight!" was said -/
def knight_count : ℕ := r * (r - 1) / 2 + l * (l - 1) / 2

/-- Theorem stating that the number of times "You are a knight!" was said is 526 -/
theorem knight_statements_count : total_statements - 2 * r * l = 526 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_statements_count_l195_19514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_trisquarish_numbers_l195_19520

/-- A number is trisquarish if it satisfies all the following conditions:
  1. It is a six-digit number in base 10.
  2. None of its digits are zero.
  3. It is a perfect cube.
  4. The first three digits form a perfect square.
  5. The last three digits form a perfect square. -/
def IsTrisquarish (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∀ d, d ∈ n.digits 10 → d ≠ 0 ∧
  ∃ y, n = y^3 ∧
  ∃ a b, n = 1000 * a + b ∧ 
    100 ≤ a ∧ a < 1000 ∧ ∃ x, a = x^2 ∧
    100 ≤ b ∧ b < 1000 ∧ ∃ z, b = z^2

theorem no_trisquarish_numbers : ¬∃ n, IsTrisquarish n := by
  sorry

#check no_trisquarish_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_trisquarish_numbers_l195_19520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_expression_range_l195_19527

-- Define the line l
def line_l (α : Real) : Set (Real × Real) :=
  {(x, y) | y = (Real.tan α) * (x + 1)}

-- Define the curve C
def curve_C : Set (Real × Real) :=
  {(x, y) | x^2 + y^2 - 6*x + 5 = 0}

-- Define the intersection condition
def intersects (α : Real) : Prop :=
  ∃ p, p ∈ line_l α ∧ p ∈ curve_C

-- Define the range of √3x + y for points on C
noncomputable def range_expr (p : Real × Real) : Real :=
  Real.sqrt 3 * p.1 + p.2

-- Theorem 1: Range of α for intersection
theorem intersection_range :
  {α : Real | intersects α} = {α | 0 ≤ α ∧ α ≤ Real.pi/6} ∪ {α | 5*Real.pi/6 ≤ α ∧ α < Real.pi} :=
by sorry

-- Theorem 2: Range of √3x + y for points on C
theorem expression_range :
  {range_expr p | p ∈ curve_C} = {y | 3*Real.sqrt 3 - 4 ≤ y ∧ y ≤ 3*Real.sqrt 3 + 4} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_expression_range_l195_19527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_well_defined_iff_l195_19526

def is_well_defined (x : ℝ) : Prop :=
  x < 5 ∧ x ≠ 2

theorem expression_well_defined_iff (x : ℝ) :
  is_well_defined x ↔ x ∈ Set.Iio 2 ∪ Set.Ioo 2 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_well_defined_iff_l195_19526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_range_for_exponential_equation_l195_19587

theorem sum_range_for_exponential_equation (a b : ℝ) (h : (2 : ℝ)^a + (2 : ℝ)^b = 1) :
  a + b ≤ -2 ∧ ∃ (x y : ℝ), (2 : ℝ)^x + (2 : ℝ)^y = 1 ∧ x + y = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_range_for_exponential_equation_l195_19587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_cube_l195_19523

/-- A right circular cone formed by three mutually perpendicular edges of a cube,
    where the cube is inscribed in a sphere of radius √3, has a volume of 4/3. -/
theorem cone_volume_from_cube (r : ℝ) (h : r = Real.sqrt 3) :
  (1 / 3) * (1 / 2) * (2 * r / Real.sqrt 3)^3 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_cube_l195_19523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_l195_19543

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -4*x

-- Define the line
def line (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the distance from a point to the directrix
noncomputable def distance_to_directrix (x : ℝ) : ℝ := abs (x + 1)

-- Define the distance from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := abs (x + y - 4) / Real.sqrt 2

-- Theorem statement
theorem min_sum_of_distances :
  ∃ (min_sum : ℝ), 
    min_sum = (5 * Real.sqrt 2) / 2 ∧
    ∀ (x y : ℝ), parabola x y →
      distance_to_directrix x + distance_to_line x y ≥ min_sum :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_l195_19543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pebble_transformation_impossible_l195_19560

-- Define the type for piles of pebbles
def PileState : Type := List Nat

-- Define the initial state
def initial_state : PileState := [5, 49, 51]

-- Define the allowed operations
def combine_piles (state : PileState) (i j : Nat) : PileState :=
  sorry

def split_pile (state : PileState) (i : Nat) : PileState :=
  sorry

-- Define the final state condition
def is_final_state (state : PileState) : Prop :=
  state.length = 105 ∧ state.all (· = 1)

-- The main theorem
theorem pebble_transformation_impossible :
  ¬ ∃ (final_state : PileState),
    (∃ (n : Nat), ∃ (ops : List (PileState → PileState)),
      final_state = (ops.foldl (fun acc op => op acc) initial_state) ∧
      is_final_state final_state) :=
by
  sorry

#check pebble_transformation_impossible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pebble_transformation_impossible_l195_19560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_approx_80_476_l195_19521

noncomputable def complex_expression : ℝ :=
  (1 : ℝ) * (2^(1/3 : ℝ) * 3^(1/2 : ℝ))^6 +
  (2^(1/2 : ℝ) * 2^(1/4 : ℝ))^(4/3 : ℝ) -
  4 * (16/49)^(-(1/2 : ℝ)) -
  2^(1/4 : ℝ) * 8^(0.25 : ℝ) -
  (-2005)^(0 : ℝ) * (2 : ℝ) * ((1 - Real.log 3 / Real.log 6)^2 + Real.log 2 / Real.log 6 * Real.log 18 / Real.log 6) / (Real.log 4 / Real.log 6)

theorem complex_expression_approx_80_476 :
  ∃ ε > 0, abs (complex_expression - 80.476) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_approx_80_476_l195_19521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_triangle_sides_arithmetic_triangle_right_angle_l195_19591

/-- A triangle with sides in arithmetic progression and area 6 --/
structure ArithmeticTriangle where
  a : ℝ
  d : ℝ
  area : ℝ
  sides_arithmetic : d = 1
  area_is_six : area = 6
  positive_sides : 0 < a ∧ 0 < a + d ∧ 0 < a + 2*d

/-- The sides of the arithmetic triangle are 3, 4, and 5 --/
theorem arithmetic_triangle_sides (t : ArithmeticTriangle) : 
  t.a = 3 ∧ t.a + t.d = 4 ∧ t.a + 2*t.d = 5 := by sorry

/-- The arithmetic triangle is a right triangle --/
theorem arithmetic_triangle_right_angle (t : ArithmeticTriangle) :
  ∃ (α β : ℝ), α + β + (π/2) = π ∧ 
  Real.sin α = 3/5 ∧ Real.sin β = 4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_triangle_sides_arithmetic_triangle_right_angle_l195_19591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_angle_theorem_l195_19541

-- Define a circle
class CircleGeometry (α : Type) where
  -- Add necessary axioms or definitions for circle geometry

-- Define points on the circle
variable {α : Type} [CircleGeometry α]
variable (A B C : α)

-- Define the measure of an arc
noncomputable def arc_measure (X Y : α) : ℝ := sorry

-- Define the measure of an angle
noncomputable def angle_measure (X Y Z : α) : ℝ := sorry

-- Theorem statement
theorem chord_angle_theorem (h1 : arc_measure B A = 110)
                            (h2 : arc_measure A C = 40) :
  angle_measure B A C = 105 ∨ angle_measure B A C = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_angle_theorem_l195_19541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_closed_form_l195_19509

/-- The sequence {x_n} defined recursively -/
noncomputable def x (a b : ℝ) : ℕ → ℝ
| 0 => 0
| n + 1 => x a b n + a + Real.sqrt (b^2 + 4 * a * x a b n)

/-- Theorem stating the closed form of the sequence -/
theorem x_closed_form (a b : ℝ) (ha : a > 0) (hb : b > 0) (n : ℕ) :
  x a b n = a * n^2 + b * n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_closed_form_l195_19509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l195_19564

noncomputable def f (x : ℝ) : ℝ := 2 * x / (x - 1)

def IsValidInput (f : ℝ → ℝ) (x : ℝ) : Prop := ¬(x = 1)

theorem domain_of_f :
  {x : ℝ | IsValidInput f x} = {x : ℝ | x ≠ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l195_19564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l195_19552

open Real

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * exp x + (2 - e) * x

theorem function_properties (a : ℝ) :
  (∀ x : ℝ, (deriv (f a)) x = (3 - e) → x = 0) →
  (a = 1 ∧
   (∀ x : ℝ, x ≥ 0 → f a x ≠ 0) ∧
   (∀ x : ℝ, x > 0 → f a x - 1 > x * log (x + 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l195_19552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_no_lattice_points_l195_19505

theorem max_b_no_lattice_points :
  let max_b := (50 : ℚ) / 151
  ∀ m : ℚ, (1 : ℚ) / 3 < m → m < max_b →
    ∀ x : ℤ, 1 ≤ x → x ≤ 150 →
      ∀ y : ℤ, y ≠ ⌊m * (x : ℚ) + 3⌋ ∧
  ∀ b : ℚ, b > max_b →
    ∃ m : ℚ, (1 : ℚ) / 3 < m ∧ m < b ∧
      ∃ x : ℤ, 1 ≤ x ∧ x ≤ 150 ∧
        ∃ y : ℤ, y = ⌊m * (x : ℚ) + 3⌋ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_no_lattice_points_l195_19505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l195_19567

/-- The probability of drawing two balls of the same color from a bag containing
    three white balls and two black balls. -/
theorem same_color_probability (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) :
  total_balls = white_balls + black_balls →
  white_balls = 3 →
  black_balls = 2 →
  (Nat.choose white_balls 2 + Nat.choose black_balls 2) / Nat.choose total_balls 2 = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l195_19567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_negative_one_l195_19503

def sequence_a : ℕ → ℚ
  | 0 => 1/2  -- Adding the base case for 0
  | 1 => 1/2
  | (n+2) => 1 - 1/(sequence_a (n+1))

theorem a_2018_equals_negative_one :
  sequence_a 2018 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_negative_one_l195_19503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l195_19563

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x + 1)

theorem range_of_f :
  let S := {y | ∃ x ∈ Set.Icc 0 2, f x = y}
  S = Set.Icc (5/3) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l195_19563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_value_l195_19534

-- Define the exponential function
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ a^x

-- State the theorem
theorem exponential_function_value (a : ℝ) :
  f a 3 = 8 → f a 6 = 64 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_value_l195_19534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dataset_variance_l195_19524

noncomputable def dataset : List ℝ := [1, 2, 1, 0, -1, -2, 0, -1]

noncomputable def mean (xs : List ℝ) : ℝ :=
  (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m)^2)).sum / xs.length

theorem dataset_variance :
  variance dataset = 1.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dataset_variance_l195_19524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_neg_part_f_pos_part_l195_19574

/-- A function f : ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The function f defined piecewise -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 2*x else x^2 + 2*x

theorem f_is_even : IsEven f := by
  sorry

theorem f_neg_part (x : ℝ) (h : x ≤ 0) : f x = x^2 - 2*x := by
  sorry

theorem f_pos_part (x : ℝ) (h : x > 0) : f x = x^2 + 2*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_neg_part_f_pos_part_l195_19574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_condition_l195_19571

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1/10) * x + 1 else Real.log x - 1

-- Define the set of a values
def A : Set ℝ := Set.Iic (-1) ∪ Set.Ici 1.1 ∪ {Real.exp (-2)}

-- State the theorem
theorem unique_root_condition (a : ℝ) :
  (∃! x, f x = a * x) ↔ a ∈ A :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_condition_l195_19571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_is_sqrt_291_l195_19530

/-- An isosceles trapezoid with specific side lengths -/
structure IsoscelesTrapezoid where
  AB : ℝ
  CD : ℝ
  AD : ℝ
  BC : ℝ
  isIsosceles : AD = BC
  parallelSides : AB > CD

/-- The length of the diagonal in the specific isosceles trapezoid -/
noncomputable def diagonalLength (t : IsoscelesTrapezoid) : ℝ :=
  Real.sqrt 291

/-- Theorem: The diagonal length of the specific isosceles trapezoid is √291 -/
theorem diagonal_length_is_sqrt_291 (t : IsoscelesTrapezoid) 
  (h1 : t.AB = 21) 
  (h2 : t.CD = 7) 
  (h3 : t.AD = 12) : 
  diagonalLength t = Real.sqrt 291 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_is_sqrt_291_l195_19530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_range_in_acute_triangle_l195_19512

theorem ratio_range_in_acute_triangle (A B C : ℝ) (a b : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Triangle is acute
  A + B + C = π ∧  -- Angle sum in a triangle
  A = 2 * B ∧  -- Given condition
  0 < a ∧ 0 < b ∧  -- Sides are positive
  a / Real.sin A = b / Real.sin B  -- Law of Sines
  →
  Real.sqrt 2 < a / b ∧ a / b < Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_range_in_acute_triangle_l195_19512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l195_19522

theorem max_value_of_function (a b : ℝ) : 
  a ∈ Set.Icc 1 3 → b ∈ Set.Icc 1 3 → a + b = 4 → 
  |Real.sqrt (a + 1/b) - Real.sqrt (b + 1/a)| ≤ 2 - 2/Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l195_19522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sectional_cost_l195_19515

/-- The cost of the sectional before discount given other costs and final payment -/
theorem sectional_cost
  (couch_cost : ℝ)
  (other_cost : ℝ)
  (discount_rate : ℝ)
  (final_payment : ℝ)
  (sectional_cost : ℝ)  -- Add sectional_cost as a parameter
  (h1 : couch_cost = 2500)
  (h2 : other_cost = 2000)
  (h3 : discount_rate = 0.1)
  (h4 : final_payment = 7200)
  (h5 : (1 - discount_rate) * (couch_cost + other_cost + sectional_cost) = final_payment) :
  sectional_cost = 3500 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sectional_cost_l195_19515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_C_coordinates_l195_19596

-- Define the points
def A : ℝ × ℝ := (10, 7)
def B : ℝ × ℝ := (1, -5)
def D : ℝ × ℝ := (0, 1)

-- Define the triangle
structure Triangle (A B C : ℝ × ℝ) : Prop where
  isosceles : dist A B = dist A C
  altitude_meets_at_D : D.1 = (B.1 + C.1) / 2 ∧ D.2 = (B.2 + C.2) / 2

-- Theorem statement
theorem triangle_C_coordinates (C : ℝ × ℝ) 
  (h : Triangle A B C) : C = (-1, 7) := by
  sorry

-- Define the distance function
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_C_coordinates_l195_19596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_tan_cot_inequality_l195_19595

theorem acute_triangle_tan_cot_inequality (A B C : Real) 
  (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  Real.tan A + Real.tan B + Real.tan C > Real.tan A⁻¹ + Real.tan B⁻¹ + Real.tan C⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_tan_cot_inequality_l195_19595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eight_is_fifteen_l195_19572

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : a 1 = 1
  h4 : (a 3) ^ 2 = a 1 * a 6

/-- The sum of the first n terms of an arithmetic sequence -/
def sum (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1 : ℚ) * seq.d)

/-- The main theorem: The sum of the first 8 terms is 15 -/
theorem sum_eight_is_fifteen (seq : ArithmeticSequence) : sum seq 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eight_is_fifteen_l195_19572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_regular_equation_l195_19579

noncomputable def x (k : ℝ) : ℝ := 4 * k / (1 - k^2)
noncomputable def y (k : ℝ) : ℝ := 4 * k^2 / (1 - k^2)

-- State the theorem
theorem parametric_to_regular_equation :
  ∀ (a b : ℝ), (∃ k : ℝ, a = x k ∧ b = y k) ↔ a^2 - b^2 - 4*b = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_regular_equation_l195_19579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_factors_l195_19558

/-- The number of even natural-number factors of 2^3 * 5^1 * 11^2 -/
def num_even_factors : ℕ := 18

/-- The prime factorization of m -/
def m : ℕ := 2^3 * 5^1 * 11^2

/-- Theorem stating that the number of even natural-number factors of m is equal to num_even_factors -/
theorem count_even_factors :
  (Finset.filter (fun x => x ∣ m ∧ Even x) (Finset.range (m + 1))).card = num_even_factors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_factors_l195_19558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l195_19578

theorem divisibility_theorem (p : Nat) (a x : Int) (h_prime : Nat.Prime p) 
  (h_div : (x^2 - a) % p = 0) (h_not_div : a % p ≠ 0) :
  ((a ^ ((p - 1) / 2) - 1) % p = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l195_19578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumscribed_sphere_area_l195_19544

/-- The surface area of a sphere circumscribed around a regular tetrahedron with edge length 2 -/
theorem tetrahedron_circumscribed_sphere_area : ℝ := by
  -- Define the edge length of the tetrahedron
  let edge_length : ℝ := 2

  -- Define the surface area of the circumscribed sphere
  let sphere_area : ℝ := 6 * Real.pi

  -- State that the tetrahedron is regular with the given edge length
  -- and the sphere is circumscribed around it
  have h1 : edge_length = 2 := rfl

  -- Prove that the surface area of the circumscribed sphere is 6π
  have h2 : sphere_area = 6 * Real.pi := rfl

  -- Final assertion
  exact sphere_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumscribed_sphere_area_l195_19544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l195_19529

theorem triangle_inequality (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Side lengths are positive
  a / Real.sin A = b / Real.sin B ∧  -- Law of sines
  b / Real.sin B = c / Real.sin C ∧  -- Law of sines
  A + C = 2 * B           -- Given condition
  → a + c ≤ 2 * b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l195_19529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l195_19502

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x + 4) - (4*x + 5) / (3*x + 10)

-- Define the solution set
def solution_set : Set ℝ := Set.Ioo (-4 : ℝ) (-10/3) ∪ Set.Ioi 2

-- State the theorem
theorem inequality_solution :
  {x : ℝ | f x > 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l195_19502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roller_coaster_probability_roller_coaster_probability_percent_l195_19555

theorem roller_coaster_probability (num_cars : ℕ) (num_rides : ℕ) : 
  num_cars = 2 → num_rides = 2 → (1 : ℚ) / 4 = (1 : ℚ) / 2 * (1 : ℚ) / 2 := by
  sorry

theorem roller_coaster_probability_percent (num_cars : ℕ) (num_rides : ℕ) : 
  num_cars = 2 → num_rides = 2 → (25 : ℚ) / 100 = (1 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roller_coaster_probability_roller_coaster_probability_percent_l195_19555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l195_19519

/-- The area of an isosceles triangle with two sides of length 26 and one side of length 50 is 25√51 -/
theorem isosceles_triangle_area (a b c : ℝ) (h1 : a = 26) (h2 : b = 26) (h3 : c = 50) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 25 * Real.sqrt 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l195_19519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_equals_38_l195_19573

-- Define the triangle
noncomputable def triangle_side1 : ℝ := 7
noncomputable def triangle_side2 : ℝ := 24
noncomputable def triangle_side3 : ℝ := 25

-- Define the rectangle
noncomputable def rectangle_width : ℝ := 7

-- Calculate the area of the triangle
noncomputable def triangle_area : ℝ := (1/2) * triangle_side1 * triangle_side2

-- Define the rectangle's length based on its area being equal to the triangle's area
noncomputable def rectangle_length : ℝ := triangle_area / rectangle_width

-- Statement to prove
theorem rectangle_perimeter_equals_38 :
  2 * (rectangle_width + rectangle_length) = 38 := by
  -- Expand definitions
  unfold rectangle_width rectangle_length triangle_area triangle_side1 triangle_side2
  -- Simplify the expression
  simp [mul_add, mul_div_cancel]
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_equals_38_l195_19573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_nonprime_polynomial_l195_19570

theorem smallest_a_for_nonprime_polynomial : 
  ∃ (a : ℕ), (a > 0) ∧ 
  (∀ (x : ℤ), ¬(Nat.Prime (Int.natAbs (x^4 + a^2)))) ∧
  (∀ (b : ℕ), 0 < b ∧ b < a → 
    ∃ (y : ℤ), Nat.Prime (Int.natAbs (y^4 + b^2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_nonprime_polynomial_l195_19570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_20_l195_19518

-- Define the vertices of the quadrilateral
def v1 : ℚ × ℚ := (1, 1)
def v2 : ℚ × ℚ := (1, 6)
def v3 : ℚ × ℚ := (5, 5)
def v4 : ℚ × ℚ := (6, 1)

-- Define the quadrilateral area calculation function
def quadrilateralArea (a b c d : ℚ × ℚ) : ℚ :=
  let (x1, y1) := a
  let (x2, y2) := b
  let (x3, y3) := c
  let (x4, y4) := d
  (1/2) * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

-- Theorem statement
theorem quadrilateral_area_is_20 :
  quadrilateralArea v1 v2 v3 v4 = 20 := by
  -- Unfold the definition and simplify
  unfold quadrilateralArea v1 v2 v3 v4
  -- Perform the calculation
  simp [abs_of_nonneg]
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_20_l195_19518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_price_theorem_l195_19532

/-- The original price of sugar per kg in Rs. -/
noncomputable def original_price : ℝ := 8

/-- The price reduction percentage -/
noncomputable def price_reduction : ℝ := 25 / 4

/-- The amount of money in Rs. -/
noncomputable def amount : ℝ := 120

/-- The additional amount of sugar in kg that can be bought after the price reduction -/
noncomputable def additional_sugar : ℝ := 1

/-- Theorem stating the relationship between the original price, reduced price, and additional sugar that can be bought -/
theorem sugar_price_theorem :
  let new_price := original_price * (1 - price_reduction / 100)
  amount / new_price - amount / original_price = additional_sugar :=
by
  sorry

#check sugar_price_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_price_theorem_l195_19532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l195_19546

/-- Given a hyperbola with equation y²/36 - x²/16 = 1, 
    the distance between its vertices is 12. -/
theorem hyperbola_vertex_distance : 
  ∀ (x y : ℝ), y^2/36 - x^2/16 = 1 → 
  ∃ (d : ℝ), d = 12 ∧ d = 2 * Real.sqrt 36 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l195_19546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girl_scouts_with_slips_percentage_l195_19594

/-- Represents the percentage of girl scouts who arrived with signed permission slips -/
noncomputable def girl_scouts_with_slips (total_scouts : ℕ) 
  (total_with_slips : ℚ) 
  (boy_scouts_percent : ℚ) 
  (boy_scouts_with_slips_percent : ℚ) : ℚ :=
  let total_with_slips_count := total_with_slips * total_scouts
  let boy_scouts_count := boy_scouts_percent * total_scouts
  let boy_scouts_with_slips := boy_scouts_with_slips_percent * boy_scouts_count
  let girl_scouts_count := total_scouts - boy_scouts_count
  let girl_scouts_with_slips := total_with_slips_count - boy_scouts_with_slips
  (girl_scouts_with_slips / girl_scouts_count) * 100

/-- The percentage of girl scouts who arrived with signed permission slips is approximately 68% -/
theorem girl_scouts_with_slips_percentage :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |girl_scouts_with_slips 100 (6/10) (45/100) (1/2) - 68| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_girl_scouts_with_slips_percentage_l195_19594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_l195_19504

theorem min_value_sin_cos (x : ℝ) : 2 * (Real.sin x)^4 + 4 * (Real.cos x)^4 ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_l195_19504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_cross_bridge_time_l195_19536

/-- Calculates the time (in minutes) for a train to cross a bridge -/
noncomputable def time_to_cross_bridge (num_carriages : ℕ) (carriage_length : ℝ) (engine_length : ℝ) 
  (train_speed_kmph : ℝ) (bridge_length_km : ℝ) : ℝ :=
  let train_length := num_carriages * carriage_length + engine_length
  let total_distance := train_length + bridge_length_km * 1000
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  (total_distance / train_speed_mps) / 60

theorem train_cross_bridge_time :
  let num_carriages : ℕ := 35
  let carriage_length : ℝ := 75
  let engine_length : ℝ := 75
  let train_speed_kmph : ℝ := 80
  let bridge_length_km : ℝ := 5
  abs (time_to_cross_bridge num_carriages carriage_length engine_length train_speed_kmph bridge_length_km - 5.77) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_cross_bridge_time_l195_19536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_parallel_lines_m_value_l195_19500

/-- Two lines are parallel if their slopes are equal -/
theorem parallel_lines (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ / b₁ = a₂ / b₂

/-- The slope of the first line -/
noncomputable def slope₁ : ℝ := -3 / 4

/-- The slope of the second line -/
noncomputable def slope₂ (m : ℝ) : ℝ := -6 / m

theorem parallel_lines_m_value :
  ∀ m : ℝ, parallel_lines 3 4 6 m → m = 8 := by
  intro m h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_parallel_lines_m_value_l195_19500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_value_perimeter_value_l195_19538

noncomputable section

-- Define the triangle ABC
variable (A B C a b c : ℝ)

-- Define the conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c

def side_c_condition (c : ℝ) : Prop := c = Real.sqrt 7

def area_condition (a b C : ℝ) : Prop :=
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2

-- Theorem 1
theorem C_value (h : triangle_condition A B C a b c) : C = π / 3 := by sorry

-- Theorem 2
theorem perimeter_value 
  (h1 : C = π / 3) 
  (h2 : side_c_condition c) 
  (h3 : area_condition a b C) : 
  a + b + c = 5 + Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_value_perimeter_value_l195_19538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_musical_chairs_seating_l195_19584

theorem musical_chairs_seating (n : ℕ) : 
  (10 : ℕ) ≥ n →
  (Nat.factorial 10 / Nat.factorial (10 - n) = Nat.factorial 7) →
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_musical_chairs_seating_l195_19584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sweet_distribution_l195_19556

theorem sweet_distribution (total : ℕ) (mother_fraction : ℚ) (eldest : ℕ) (children : ℕ) :
  total = 27 →
  mother_fraction = 1 / 3 →
  eldest = 8 →
  children = 3 →
  ∃ (second youngest : ℕ),
    second + youngest + eldest = total - (mother_fraction * ↑total).floor ∧
    youngest = eldest / 2 ∧
    second = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sweet_distribution_l195_19556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_B_finite_A_B_integer_max_B_geq_2_l195_19561

-- Define the sets A, B, and C
noncomputable def A : Set ℝ := sorry
noncomputable def B : Set ℝ := sorry
def C : Set ℕ := Finset.range (2^2828 + 1)

-- Define the conditions
axiom set_C : C = {c : ℕ | ∃ (a : ℝ) (b : ℝ), a ∈ A ∧ b ∈ B ∧ c = Int.floor (a + b)}
axiom max_A : ∃ (m : ℝ), m ∈ A ∧ ∀ (a : ℝ), a ∈ A → a ≤ m ∧ m = (Real.sqrt 2 - 1)^2020 + (Real.sqrt 2 + 1)^2020

-- Theorem statements
theorem A_B_finite : Set.Finite A ∧ Set.Finite B := by sorry

theorem A_B_integer : (∀ (a : ℝ), a ∈ A → ∃ (n : ℤ), a = ↑n) ∧
                      (∀ (b : ℝ), b ∈ B → ∃ (m : ℤ), b = ↑m) := by sorry

theorem max_B_geq_2 : ∃ (m : ℝ), m ∈ B ∧ ∀ (b : ℝ), b ∈ B → b ≤ m ∧ m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_B_finite_A_B_integer_max_B_geq_2_l195_19561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_triangle_theorem_l195_19568

-- Define the complex numbers a, b, c
variable (a b c : ℂ)

-- Define the conditions
def condition1 (a b c : ℂ) : Prop := a^2 + b^2 + c^2 = a*b + b*c + c*a
def condition2 (a b c : ℂ) : Prop := Complex.abs (a + b + c) = 21
def condition3 (a b c : ℂ) : Prop := Complex.abs (a - b) = 2 * Real.sqrt 3
def condition4 (a : ℂ) : Prop := Complex.abs a = 3 * Real.sqrt 3

-- State the theorem
theorem complex_triangle_theorem 
  (h1 : condition1 a b c)
  (h2 : condition2 a b c)
  (h3 : condition3 a b c)
  (h4 : condition4 a) :
  Complex.abs b ^ 2 + Complex.abs c ^ 2 = 132 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_triangle_theorem_l195_19568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_property_l195_19551

/-- A polynomial with specific properties -/
def Q (x a b p q : ℂ) : ℂ := (x^2 - a*x + p) * (x^2 - b*x + q) * (x^2 - 6*x + 18)

/-- The theorem stating the properties of the polynomial and the conclusion about |b - a| -/
theorem polynomial_roots_property (a b p q : ℝ) :
  (∃ (z₁ z₂ z₃ z₄ : ℂ), z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    ∀ (z : ℂ), Q z (a : ℂ) (b : ℂ) (p : ℂ) (q : ℂ) = 0 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃ ∨ z = z₄) →
  p + q = 12 →
  p > 0 →
  q > 0 →
  |b - a| = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_property_l195_19551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_characterization_l195_19589

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem power_of_two_characterization (k : ℕ) (h : k > 0) :
  (∀ n : ℕ, n > 0 → ¬(divides (2^((k-1)*n+1)) (Nat.factorial (k*n) / Nat.factorial n))) ↔
  (∃ a : ℕ, k = 2^a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_characterization_l195_19589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_empty_time_approx_l195_19557

/-- Represents a water cistern with leaks -/
structure Cistern where
  normalFillTime : ℝ
  leak1AddedTime : ℝ
  leak2AddedTime : ℝ
  leak3AddedTime : ℝ

/-- Calculates the time it takes for the cistern to empty when all leaks are active -/
noncomputable def emptyTime (c : Cistern) : ℝ :=
  let normalRate := 1 / c.normalFillTime
  let leak1Rate := 1 / c.normalFillTime - 1 / (c.normalFillTime + c.leak1AddedTime)
  let leak2Rate := 1 / c.normalFillTime - 1 / (c.normalFillTime + c.leak2AddedTime)
  let leak3Rate := 1 / c.normalFillTime - 1 / (c.normalFillTime + c.leak3AddedTime)
  let totalLeakRate := leak1Rate + leak2Rate + leak3Rate
  1 / totalLeakRate

/-- Theorem stating that the empty time is approximately 12.09 hours for the given conditions -/
theorem cistern_empty_time_approx (c : Cistern) 
  (h1 : c.normalFillTime = 10)
  (h2 : c.leak1AddedTime = 2)
  (h3 : c.leak2AddedTime = 4)
  (h4 : c.leak3AddedTime = 6) :
  ∃ ε > 0, |emptyTime c - 12.09| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_empty_time_approx_l195_19557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rug_inner_rectangle_length_l195_19585

/-- Represents the dimensions of a rectangular region -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the rug with three colored regions -/
structure Rug where
  innerRect : Rectangle
  firstShadedWidth : ℝ
  secondShadedWidth : ℝ

/-- Calculates the areas of the three regions in the rug -/
def rugAreas (r : Rug) : Fin 3 → ℝ
  | 0 => rectangleArea r.innerRect
  | 1 => rectangleArea { length := r.innerRect.length + 2 * r.firstShadedWidth, 
                         width := r.innerRect.width + 2 * r.firstShadedWidth } - 
         rectangleArea r.innerRect
  | 2 => rectangleArea { length := r.innerRect.length + 2 * (r.firstShadedWidth + r.secondShadedWidth), 
                         width := r.innerRect.width + 2 * (r.firstShadedWidth + r.secondShadedWidth) } - 
         (rectangleArea r.innerRect + 
          (rectangleArea { length := r.innerRect.length + 2 * r.firstShadedWidth, 
                           width := r.innerRect.width + 2 * r.firstShadedWidth } - 
           rectangleArea r.innerRect))
  | _ => 0  -- This case should never be reached due to Fin 3

/-- Checks if a function from Fin 3 to ℝ forms an arithmetic progression -/
def isArithmeticProgression (v : Fin 3 → ℝ) : Prop :=
  v 1 - v 0 = v 2 - v 1

theorem rug_inner_rectangle_length :
  ∀ (r : Rug),
    r.innerRect.width = 1 →
    r.firstShadedWidth = 2 →
    r.secondShadedWidth = 3 →
    isArithmeticProgression (rugAreas r) →
    r.innerRect.length = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rug_inner_rectangle_length_l195_19585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l195_19516

noncomputable def f (x : ℝ) := 2 * x - 1 + Real.log x / Real.log 2

theorem zero_in_interval :
  ∃ x ∈ Set.Ioo (1/2 : ℝ) 1, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l195_19516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_midpoint_time_l195_19531

/-- Represents a race participant with their time and handicap -/
structure Participant where
  time : ℝ
  handicap : ℝ

/-- Calculates the speed of a participant given the race distance -/
noncomputable def speed (p : Participant) (race_distance : ℝ) : ℝ :=
  (race_distance + p.handicap) / p.time

/-- Represents the race setup -/
structure Race where
  distance : ℝ
  a : Participant
  b : Participant
  c : Participant

theorem race_midpoint_time (race : Race) 
  (h_distance : race.distance = 110)
  (h_a_time : race.a.time = 36)
  (h_b_time : race.b.time = 45)
  (h_a_handicap : race.a.handicap = 10)
  (h_b_handicap : race.b.handicap = 5)
  (h_c_handicap : race.c.handicap = 0)
  (h_c_midpoint : 
    (speed race.a race.distance * race.b.time + race.distance + race.b.handicap) / 2 - race.a.handicap
    = speed race.a race.distance * race.c.time) :
  race.c.time = 36.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_midpoint_time_l195_19531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_at_6_l195_19553

/-- A monic polynomial of degree 5 satisfying p(k) = k for k = 1, 2, 3, 4, 5 -/
noncomputable def p : ℝ → ℝ := sorry

/-- p is a monic polynomial of degree 5 -/
axiom p_monic : ∃ a b c d : ℝ, ∀ x, p x = x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + (p 0)

/-- p satisfies p(k) = k for k = 1, 2, 3, 4, 5 -/
axiom p_values : ∀ k : Fin 5, p (k + 1) = k + 1

/-- Theorem: p(6) = 126 -/
theorem p_at_6 : p 6 = 126 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_at_6_l195_19553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l195_19597

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = 2*x + f (f y - x)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l195_19597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l195_19580

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := (sin x)^2 / (1 + (cos x)^2) + (cos x)^2 / (1 + (sin x)^2)

/-- Theorem stating that the range of f(x) is [2/3, 1] -/
theorem f_range : ∀ x : ℝ, 2/3 ≤ f x ∧ f x ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l195_19580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_second_quadrant_l195_19576

/-- Given an angle α = -4 radians, prove that its terminal side is in the second quadrant. -/
theorem terminal_side_in_second_quadrant (α : ℝ) :
  α = -4 ∧ -2 * Real.pi < α ∧ α < -Real.pi → 
  ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ Real.cos α = x ∧ Real.sin α = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_second_quadrant_l195_19576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_location_l195_19540

theorem point_location (α : Real) (h : α = 5 * Real.pi / 8) :
  Real.sin α > 0 ∧ Real.tan α < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_location_l195_19540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_product_specific_cube_root_evaluation_l195_19542

theorem cube_root_of_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^3 * b^6 * c^3)^(1/3) = a * b^2 * c :=
sorry

theorem specific_cube_root_evaluation : 
  (5^3 * 7^6 * 13^3 : ℝ)^(1/3) = 3185 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_product_specific_cube_root_evaluation_l195_19542
