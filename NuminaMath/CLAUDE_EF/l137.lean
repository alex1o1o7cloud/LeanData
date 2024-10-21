import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l137_13747

/-- The sum of the infinite series Σ(k/4^k) for k from 1 to infinity -/
noncomputable def infiniteSeries : ℝ := ∑' k, k / (4 : ℝ) ^ k

/-- Theorem: The sum of the infinite series Σ(k/4^k) for k from 1 to infinity is equal to 4/9 -/
theorem infiniteSeriesSum : infiniteSeries = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l137_13747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_area_is_800_l137_13764

/-- Represents a rectangular field with specific fencing constraints. -/
structure RectangularField where
  uncoveredSide : ℝ
  fencingLength : ℝ

/-- Calculates the area of a rectangular field given the fencing constraints. -/
noncomputable def fieldArea (field : RectangularField) : ℝ :=
  let width := (field.fencingLength - field.uncoveredSide) / 2
  field.uncoveredSide * width

/-- Theorem stating that a rectangular field with given constraints has an area of 800 square feet. -/
theorem field_area_is_800 (field : RectangularField) 
    (h1 : field.uncoveredSide = 20)
    (h2 : field.fencingLength = 100) : 
  fieldArea field = 800 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval fieldArea { uncoveredSide := 20, fencingLength := 100 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_area_is_800_l137_13764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overtime_pay_ratio_l137_13769

/-- Calculates the ratio of overtime pay rate to regular pay rate given the following conditions:
  * Regular pay rate is $3 per hour
  * Regular hours worked is 40 hours
  * Total pay received is $168
  * Overtime hours worked is 8 hours
-/
theorem overtime_pay_ratio (regular_rate : ℚ) (regular_hours : ℚ) (total_pay : ℚ) (overtime_hours : ℚ) :
  regular_rate = 3 →
  regular_hours = 40 →
  total_pay = 168 →
  overtime_hours = 8 →
  (total_pay - regular_rate * regular_hours) / overtime_hours / regular_rate = 2 := by
  intros h1 h2 h3 h4
  -- We'll use 'sorry' to skip the proof for now
  sorry

-- Remove the #eval line as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overtime_pay_ratio_l137_13769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marc_speed_is_18_17_l137_13739

/-- Represents the hiking scenario with Chantal and Marc -/
structure HikingScenario where
  x : ℝ  -- represents the unit distance (1/4 of total trail length)
  chantal_speed1 : ℝ  -- Chantal's speed for first quarter
  chantal_speed2 : ℝ  -- Chantal's speed for next three quarters
  chantal_speed_descent : ℝ  -- Chantal's descent speed
  meeting_point : ℝ  -- Point where Chantal and Marc meet (in terms of x)

/-- Calculate Marc's average speed given a hiking scenario -/
noncomputable def marc_average_speed (scenario : HikingScenario) : ℝ :=
  let chantal_time := scenario.x / scenario.chantal_speed1 + 
                      (3 * scenario.x) / scenario.chantal_speed2 + 
                      scenario.x / scenario.chantal_speed_descent
  scenario.meeting_point / chantal_time

/-- Theorem stating that Marc's average speed is 18/17 under the given conditions -/
theorem marc_speed_is_18_17 (scenario : HikingScenario) 
    (h1 : scenario.chantal_speed1 = 3)
    (h2 : scenario.chantal_speed2 = 3/2)
    (h3 : scenario.chantal_speed_descent = 2)
    (h4 : scenario.meeting_point = 3 * scenario.x) : 
  marc_average_speed scenario = 18/17 := by
  sorry

#check marc_speed_is_18_17

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marc_speed_is_18_17_l137_13739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_probability_l137_13733

def remainder_probability (r : Fin 4) : ℚ :=
  match r with
  | 0 => 1/2
  | 1 => 1/8
  | 2 => 1/4
  | 3 => 1/8

theorem product_remainder_probability :
  ∀ (a b : ℤ),
  (∃ (r : Fin 4), (a * b) % 4 = r.val) ∧
  (∀ (r : Fin 4), (remainder_probability r : ℝ) =
    (Finset.filter (λ (p : Fin 4 × Fin 4) => (p.1.val * p.2.val) % 4 = r.val) (Finset.product (Finset.univ : Finset (Fin 4)) (Finset.univ : Finset (Fin 4)))).card /
    ((Finset.univ : Finset (Fin 4)).card * (Finset.univ : Finset (Fin 4)).card)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_probability_l137_13733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_primes_average_composite_l137_13749

theorem consecutive_odd_primes_average_composite (p₁ p₂ q : ℕ) :
  Nat.Prime p₁ →
  Nat.Prime p₂ →
  Odd p₁ →
  Odd p₂ →
  p₁ < p₂ →
  (∀ k, p₁ < k → k < p₂ → ¬Nat.Prime k) →
  q = (p₁ + p₂) / 2 →
  ¬Nat.Prime q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_primes_average_composite_l137_13749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_proof_l137_13730

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Round to nearest integer --/
noncomputable def round_to_nearest_int (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem compound_interest_rate_proof (principal time : ℕ) (interest : ℝ) : 
  principal = 14800 → time = 2 → interest = 4265.73 → 
  ∃ (rate : ℝ), 0 < rate ∧ rate < 1 ∧ 
  compound_interest (principal : ℝ) rate time = interest ∧
  round_to_nearest_int (rate * 100) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_proof_l137_13730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_l137_13766

/-- Represents a trapezium with given dimensions and area -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def trapeziumArea (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- Theorem: Given a trapezium with one side 20 cm, height 15 cm, and area 285 cm², the other side is 18 cm -/
theorem trapezium_other_side (t : Trapezium) (h1 : t.side1 = 20) (h2 : t.height = 15) (h3 : t.area = 285) :
  t.side2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_l137_13766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l137_13725

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Add this case to handle n = 0
  | 1 => 2
  | n + 2 => 2 * sequence_a (n + 1) - 1

theorem sequence_a_closed_form (n : ℕ) :
  n ≥ 1 → sequence_a n = 2^(n - 1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l137_13725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_necessary_not_sufficient_l137_13738

/-- Proposition p: sin x = 1/2 -/
def p (x : ℝ) : Prop := Real.sin x = 1/2

/-- Proposition q: x = π/6 + 2kπ, k ∈ ℤ -/
def q (x : ℝ) : Prop := ∃ k : ℤ, x = Real.pi/6 + 2*k*Real.pi

/-- p is necessary but not sufficient for q -/
theorem p_necessary_not_sufficient :
  (∀ x : ℝ, q x → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_necessary_not_sufficient_l137_13738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_after_2020_with_digit_sum_4_l137_13719

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a year is after 2020 and has digit sum 4 -/
def valid_year (year : ℕ) : Prop :=
  year > 2020 ∧ digit_sum year = 4

/-- 2200 is the first year after 2020 with digit sum 4 -/
theorem first_year_after_2020_with_digit_sum_4 :
  ∃ (y : ℕ), y = 2200 ∧ valid_year y ∧ ∀ (z : ℕ), z < y → ¬(valid_year z) := by
  sorry

#check first_year_after_2020_with_digit_sum_4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_after_2020_with_digit_sum_4_l137_13719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_60_degrees_range_of_b_plus_c_l137_13790

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  2 * t.b * Real.cos t.A = t.c * Real.cos t.A + t.a * Real.cos t.C

-- Theorem 1: Measure of angle A
theorem angle_A_is_60_degrees (t : Triangle) 
  (h : given_condition t) : t.A = Real.pi / 3 := by
  sorry

-- Theorem 2: Range of b + c when a = 6
theorem range_of_b_plus_c (t : Triangle) 
  (h1 : given_condition t) 
  (h2 : t.a = 6) : 
  6 < t.b + t.c ∧ t.b + t.c ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_60_degrees_range_of_b_plus_c_l137_13790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_analogous_reasoning_correctness_l137_13701

theorem analogous_reasoning_correctness : 
  ∃ (S : Finset ℕ), 
    Finset.card S = 2 ∧ 
    (∀ i ∈ S, i < 4) ∧
    (∀ i ∉ S, i < 4 → ¬(i ∈ S)) ∧
    (0 ∈ S ↔ ∀ (a b : ℂ), a - b = 0 → a = b) ∧
    (1 ∈ S ↔ ∀ (a b c d : ℝ), Complex.mk a b = Complex.mk c d → a = c ∧ b = d) ∧
    (2 ∉ S) ∧
    (3 ∈ S ↔ ∀ (a b : ℂ), a * b = 0 → a = 0 ∨ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_analogous_reasoning_correctness_l137_13701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dave_apps_l137_13778

/-- The number of apps Dave initially had on his phone -/
def initial_apps (deleted installed : ℕ) : ℕ := 23 - installed

theorem dave_apps (deleted installed final : ℕ) 
  (h1 : deleted = 18) 
  (h2 : final = 5) : 
  initial_apps deleted installed = 23 - installed :=
by
  -- Unfold the definition of initial_apps
  unfold initial_apps
  -- The goal is now trivially true by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dave_apps_l137_13778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l137_13788

/-- A power function that passes through a specific point -/
noncomputable def PowerFunction (k α : ℝ) : ℝ → ℝ := fun x ↦ k * x ^ α

/-- Theorem stating the existence of k and α satisfying the conditions -/
theorem power_function_sum :
  ∃ k α : ℝ, (∀ x, PowerFunction k α x = k * x ^ α) ∧ 
  PowerFunction k α (1/2) = Real.sqrt 2 ∧ 
  k + α = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l137_13788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_approx_l137_13787

/-- The distance Jonathan ran in kilometers -/
def jonathan_distance : ℝ := 7.5

/-- The distance Mercedes ran in kilometers -/
def mercedes_distance : ℝ := 2.5 * jonathan_distance

/-- The distance Davonte ran in kilometers -/
noncomputable def davonte_distance : ℝ := Real.sqrt (3.25 * mercedes_distance)

/-- The distance Felicia ran in kilometers -/
noncomputable def felicia_distance : ℝ := davonte_distance - 1.75

/-- The average distance of Jonathan, Davonte, and Felicia in kilometers -/
noncomputable def average_distance : ℝ := (jonathan_distance + davonte_distance + felicia_distance) / 3

/-- The distance Emilia ran in kilometers -/
noncomputable def emilia_distance : ℝ := average_distance ^ 2

/-- The total distance run by Mercedes, Davonte, Felicia, and Emilia in kilometers -/
noncomputable def total_distance : ℝ := mercedes_distance + davonte_distance + felicia_distance + emilia_distance

/-- Theorem stating that the total distance is approximately 83.321 kilometers -/
theorem total_distance_approx : 
  |total_distance - 83.321| < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_approx_l137_13787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_range_l137_13798

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x) * Real.cos x) / (1 - Real.sin x)

theorem f_domain_and_range :
  (∀ x : ℝ, f x ≠ 0 → ∀ k : ℤ, x ≠ 2 * Real.pi * ↑k + Real.pi / 2) ∧
  (∀ y : ℝ, y ∈ Set.Icc (-1/2) 4 → ∃ x : ℝ, f x = y) ∧
  (∀ x : ℝ, f x ∈ Set.Icc (-1/2) 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_range_l137_13798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_compositions_exist_l137_13784

theorem infinite_compositions_exist (n : ℕ) (hn : n > 1) (ℓ m : Fin n) :
  let f : Fin n → Fin n := λ i ↦ ⟨(2 * i) % n, by sorry⟩
  let g : Fin n → Fin n := λ i ↦ ⟨(2 * i + 1) % n, by sorry⟩
  ∃ (compositions : Set (Fin n → Fin n)), 
    (Set.Infinite compositions) ∧ 
    (∀ h ∈ compositions, h ℓ = m) ∧
    (∀ h ∈ compositions, ∃ (k : ℕ) (hs : List (Fin n → Fin n)),
      h = List.foldl (· ∘ ·) id (f::g::hs) ∧ hs.length = k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_compositions_exist_l137_13784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_four_element_set_l137_13796

theorem proper_subsets_of_four_element_set :
  ∀ (S : Finset Nat), Finset.card S = 4 →
  Finset.card (Finset.powerset S \ {S}) = 15 :=
by
  intro S hcard
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_four_element_set_l137_13796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l137_13728

theorem cone_base_radius 
  (slant_height : ℝ) 
  (lateral_surface_area : ℝ) 
  (h1 : slant_height = 6)
  (h2 : lateral_surface_area = 18 * Real.pi) :
  lateral_surface_area / (Real.pi * slant_height) = 3 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l137_13728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_test_result_l137_13757

/-- Chi-square test statistic -/
noncomputable def chi_square_statistic : ℝ := 100 / 21

/-- Critical value for 5% significance level with 1 degree of freedom -/
noncomputable def critical_value_5_percent : ℝ := 3.841

/-- Critical value for 2.5% significance level with 1 degree of freedom -/
noncomputable def critical_value_2_5_percent : ℝ := 5.024

/-- Theorem stating that the chi-square test statistic falls between the critical values for 5% and 2.5% significance levels -/
theorem chi_square_test_result : 
  critical_value_5_percent < chi_square_statistic ∧ 
  chi_square_statistic < critical_value_2_5_percent := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_test_result_l137_13757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l137_13775

noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x - 1/2

theorem max_value_of_f :
  ∀ x : ℝ, 0 ≤ x → x ≤ 2 → f x ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l137_13775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_B_height_l137_13737

/-- Tank C with height 10 meters and circumference 8 meters -/
def tank_C : ℝ × ℝ := (10, 8)

/-- Tank B with circumference 10 meters -/
def tank_B_circ : ℝ := 10

/-- The ratio of tank C's capacity to tank B's capacity -/
def capacity_ratio : ℝ := 0.8

/-- Calculate the volume of a cylinder given its height and circumference -/
noncomputable def cylinder_volume (height : ℝ) (circumference : ℝ) : ℝ :=
  (height * circumference^2) / (4 * Real.pi)

theorem tank_B_height :
  let h_C := tank_C.1
  let circ_C := tank_C.2
  let circ_B := tank_B_circ
  ∃ h_B : ℝ,
    cylinder_volume h_C circ_C = capacity_ratio * cylinder_volume h_B circ_B ∧
    h_B = 8 := by
  sorry

#check tank_B_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_B_height_l137_13737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rohit_final_distance_l137_13729

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Rohit's walk -/
def rohit_walk : Point → Point
| ⟨x, y⟩ => ⟨x + 35, y⟩

theorem rohit_final_distance :
  let start := Point.mk 0 0
  let end_point := rohit_walk start
  distance start end_point = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rohit_final_distance_l137_13729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_l137_13741

theorem expression_equals_one (a b c : ℝ) (ha : a = 7.4) (hb : b = 5/37) :
  (1/a + 1/b - 2*c/(a*b)) * (a + b + 2*c) / (1/a^2 + 1/b^2 + 2/(a*b) - 4*c^2/(a^2*b^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_l137_13741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_eq_neg_sin_l137_13785

open Real

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => sin
  | n + 1 => fun x => deriv (f n) x

-- Statement to prove
theorem f_2010_eq_neg_sin : ∀ x, f 2010 x = -sin x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_eq_neg_sin_l137_13785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l137_13704

theorem trigonometric_identity (x y : ℝ) : 
  0 < x → x < y → y < π/2 → Real.tan y = Real.tan x + 1 / Real.cos x → y - x/2 = π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l137_13704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l137_13786

def a : ℝ × ℝ := (1, 1)

noncomputable def b (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

theorem vector_properties :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π) →
  (b (π/4) = (Real.sqrt 2 / 2, Real.sqrt 2 / 2)) ∧
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π → 
    let φ := Real.arccos ((a.1 * (b θ).1 + a.2 * (b θ).2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt ((b θ).1^2 + (b θ).2^2)))
    0 ≤ φ ∧ φ ≤ 3*π/4) ∧
  (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧ 
    (a.1 + (b θ).1)^2 + (a.2 + (b θ).2)^2 = (a.1 - (b θ).1)^2 + (a.2 - (b θ).2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l137_13786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_n_l137_13777

def n : ℕ := 2^4 * 3^5 * 4^6 * 6^7

theorem number_of_factors_of_n : (Finset.filter (λ x => x ∣ n) (Finset.range (n + 1))).card = 312 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_n_l137_13777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l137_13734

/-- Represents the four quadrants of the coordinate plane. -/
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

/-- Returns the quadrant of an angle based on its measure in radians. -/
noncomputable def QuadrantOfAngle (θ : ℝ) : Quadrant :=
  if 0 ≤ θ % (2 * Real.pi) && θ % (2 * Real.pi) < Real.pi / 2 then Quadrant.First
  else if Real.pi / 2 ≤ θ % (2 * Real.pi) && θ % (2 * Real.pi) < Real.pi then Quadrant.Second
  else if Real.pi ≤ θ % (2 * Real.pi) && θ % (2 * Real.pi) < 3 * Real.pi / 2 then Quadrant.Third
  else Quadrant.Fourth

/-- If the sine and tangent of an angle are both negative, then the angle is in the fourth quadrant. -/
theorem angle_in_fourth_quadrant (θ : ℝ) : Real.sin θ < 0 → Real.tan θ < 0 → QuadrantOfAngle θ = Quadrant.Fourth := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l137_13734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l137_13722

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem tangent_line_at_one (x y : ℝ) :
  (∀ t, t > 0 → deriv f t = Real.log t + 1) →
  f 1 = 0 →
  deriv f 1 = 1 →
  (x - y - 1 = 0 ↔ y = f 1 + deriv f 1 * (x - 1)) :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l137_13722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_change_is_half_meter_l137_13740

/-- Represents the dimensions of a field -/
structure FieldDimensions where
  length : ℝ
  breadth : ℝ

/-- Represents the dimensions of a tank -/
structure TankDimensions where
  length : ℝ
  breadth : ℝ
  min_depth : ℝ
  max_depth : ℝ

/-- Calculates the change in height of a field after spreading earth from a tank -/
noncomputable def calculate_height_change (field : FieldDimensions) (tank : TankDimensions) : ℝ :=
  let field_area := field.length * field.breadth
  let tank_area := tank.length * tank.breadth
  let remaining_area := field_area - tank_area
  let avg_depth := (tank.min_depth + tank.max_depth) / 2
  let tank_volume := tank.length * tank.breadth * avg_depth
  tank_volume / remaining_area

/-- Theorem stating that the change in height of the field is 0.5 meters -/
theorem height_change_is_half_meter 
  (field : FieldDimensions)
  (tank : TankDimensions)
  (h1 : field.length = 90)
  (h2 : field.breadth = 50)
  (h3 : tank.length = 25)
  (h4 : tank.breadth = 20)
  (h5 : tank.min_depth = 2)
  (h6 : tank.max_depth = 6)
  : calculate_height_change field tank = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_change_is_half_meter_l137_13740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l137_13742

-- Define the ∇ operation as noncomputable
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_calculation : nabla (nabla 2 4) (nabla 5 6) = 19 / 23 := by
  -- Proof steps would go here
  sorry

-- Note: The positivity conditions are not directly used in the statement,
-- but they ensure that the operations are well-defined in the real number system.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l137_13742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_behavior_on_negative_interval_l137_13768

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of being even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of being increasing on an interval
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Define the property of being decreasing on an interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- State the theorem
theorem f_behavior_on_negative_interval :
  is_even f →
  is_increasing_on f 1 3 →
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≥ 5) →
  (∃ x, 1 ≤ x ∧ x ≤ 3 ∧ f x = 5) →
  is_decreasing_on f (-3) (-1) ∧
  (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_behavior_on_negative_interval_l137_13768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_k_range_l137_13765

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (k : ℝ) : ℝ := Real.sqrt (4 - k) / 2

/-- The theorem stating the range of k for a hyperbola with given eccentricity -/
theorem hyperbola_k_range :
  ∀ k : ℝ, (1 < eccentricity k ∧ eccentricity k < 2) ↔ -12 < k ∧ k < 0 := by
  sorry

#check hyperbola_k_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_k_range_l137_13765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_share_is_780_l137_13707

/-- Represents the time (in days) it takes for a person to complete the job alone -/
structure WorkRate where
  days : ℝ
  days_pos : days > 0

/-- Represents the total earnings for the job -/
def total_earnings : ℝ := 2340

/-- Calculate the rate of work per day -/
noncomputable def rate_per_day (w : WorkRate) : ℝ := 1 / w.days

/-- Calculate the combined rate of work per day for all workers -/
noncomputable def combined_rate (a b c : WorkRate) : ℝ :=
  rate_per_day a + rate_per_day b + rate_per_day c

/-- Calculate the fraction of work done by a specific worker -/
noncomputable def work_fraction (worker : WorkRate) (a b c : WorkRate) : ℝ :=
  rate_per_day worker / combined_rate a b c

/-- Calculate a worker's share of the earnings -/
noncomputable def worker_share (worker : WorkRate) (a b c : WorkRate) : ℝ :=
  total_earnings * work_fraction worker a b c

theorem b_share_is_780 (a b c : WorkRate)
  (ha : a.days = 6)
  (hb : b.days = 8)
  (hc : c.days = 12) :
  worker_share b a b c = 780 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_share_is_780_l137_13707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_positive_reals_l137_13703

-- Define the function f(x) = (1/3)^(x^2) - 9
noncomputable def f (x : ℝ) : ℝ := (1/3)^(x^2) - 9

-- State the theorem
theorem f_monotone_decreasing_on_positive_reals :
  ∀ a b : ℝ, 0 < a → a < b → f b < f a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_positive_reals_l137_13703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_in_geometric_triangle_l137_13791

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a, b, c form a geometric sequence and c = 2a, then cos B = 3/4 -/
theorem cosine_in_geometric_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  b^2 = a * c →  -- Condition for geometric sequence
  c = 2 * a →
  Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c) →  -- Cosine rule
  Real.cos B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_in_geometric_triangle_l137_13791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_16_octahedral_dice_l137_13776

def octahedral_die := Finset.range 8

theorem probability_sum_16_octahedral_dice :
  let outcomes := octahedral_die.product octahedral_die
  let favorable_outcomes := outcomes.filter (fun p => p.1 + p.2 + 4 = 16)
  (favorable_outcomes.card : ℚ) / outcomes.card = 3 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_16_octahedral_dice_l137_13776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_parabola_equation_our_parabola_equation_l137_13774

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parabola with specific properties -/
structure SpecialParabola where
  /-- The vertex is at the origin -/
  vertex_at_origin : Point
  /-- The axis of symmetry is along a coordinate axis -/
  axis_along_coordinate : Bool
  /-- The focus lies on the line x - y + 2 = 0 -/
  focus_on_line : Point → Prop

/-- The equation of the special parabola -/
def parabola_equation (p : Point) : Prop :=
  p.x^2 = 8 * p.y

/-- Theorem stating that a parabola with the given properties has the equation x² = 8y -/
theorem special_parabola_equation (sp : SpecialParabola) :
  ∀ p : Point, sp.focus_on_line p → parabola_equation p :=
by
  sorry

/-- The focus of the parabola satisfies the line equation -/
def focus_satisfies_line (p : Point) : Prop :=
  p.x - p.y + 2 = 0

/-- The special parabola instance -/
def our_parabola : SpecialParabola :=
  { vertex_at_origin := ⟨0, 0⟩
  , axis_along_coordinate := true
  , focus_on_line := focus_satisfies_line }

/-- Theorem stating that our specific parabola satisfies the equation -/
theorem our_parabola_equation :
  ∀ p : Point, our_parabola.focus_on_line p → parabola_equation p :=
by
  exact special_parabola_equation our_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_parabola_equation_our_parabola_equation_l137_13774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_circ_result_l137_13750

noncomputable def star (x y : ℝ) : ℝ := (x * y) / (x + y)

noncomputable def circ (x y : ℝ) : ℝ := x + y - x * y

theorem star_circ_result :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → 
  x = 2 → y = 3 → z = 4 → 
  circ (star x y) z = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_circ_result_l137_13750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l137_13718

def a (n : ℕ+) : ℕ :=
  if n = 1 then 2 else sorry

axiom a_add (p q : ℕ+) : a (p + q) = a p + a q

def S (n : ℕ+) : ℕ := sorry

def f (n : ℕ+) : ℚ := (S n + 60) / (n + 1)

theorem min_value_of_f :
  ∃ (n : ℕ+), f n = 29/2 ∧ ∀ (m : ℕ+), f m ≥ 29/2 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l137_13718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l137_13773

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 + x else ((-x)^2 + (-x))

theorem f_range (a : ℝ) :
  (∀ x, f (-x) = f x) →  -- f is even
  (∀ x ≥ 0, f x = x^2 + x) →  -- definition for x ≥ 0
  f a + f (-a) < 4 →  -- given condition
  -1 < a ∧ a < 1 := by  -- conclusion to prove
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l137_13773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l137_13761

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + (3/2) * x - 9

-- State the theorem
theorem f_has_zero_in_interval :
  ∃ x : ℝ, x > 5 ∧ x < 6 ∧ f x = 0 :=
by
  sorry

#check f_has_zero_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l137_13761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_plates_for_attenuation_l137_13714

/-- Represents the attenuation factor for 1 mm of glass thickness -/
noncomputable def a : ℝ := sorry

/-- Represents the number of 1 mm glass plates -/
def n : ℕ := sorry

/-- Represents the attenuation factor for a single gap between plates -/
noncomputable def x : ℝ := sorry

/-- The attenuation factor for a single gap is equal to the 9th root of the attenuation factor for 1 mm of glass -/
axiom gap_attenuation : x = a^(1/9 : ℝ)

/-- The attenuation through 10 plates with 9 gaps is equal to the attenuation through 11 mm of glass -/
axiom plate_equivalence : a^10 * x^9 = a^11

/-- The attenuation through n plates with (n-1) gaps is not greater than the attenuation through 20 mm of glass -/
def attenuation_condition (n : ℕ) : Prop :=
  a^20 ≥ a^n * x^(n-1)

/-- The theorem to be proved -/
theorem min_plates_for_attenuation :
  ∀ k : ℕ, (∀ m < k, ¬attenuation_condition m) → attenuation_condition k → k = 19 := by
  sorry

#check min_plates_for_attenuation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_plates_for_attenuation_l137_13714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_video_game_levels_l137_13756

theorem video_game_levels (beaten_levels : ℕ) (ratio : ℚ) : 
  beaten_levels = 24 → ratio = 3/1 → beaten_levels + (beaten_levels / ratio.num * ratio.den) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_video_game_levels_l137_13756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derived_numbers_distinct_count_non_distinct_derivations_l137_13715

def derived_numbers (a b c : ℕ) : List ℕ :=
  [a + b + c, a + b * c, b + a * c, c + a * b, (a + b) * c, (b + c) * a, (c + a) * b, a * b * c]

def is_prime (p : ℕ) : Prop := Nat.Prime p

def count_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem derived_numbers_distinct (n a b c : ℕ) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h2 : n / 2 < a ∧ a ≤ n) 
  (h3 : n / 2 < b ∧ b ≤ n) 
  (h4 : n / 2 < c ∧ c ≤ n) :
  (derived_numbers a b c).Nodup := by sorry

theorem count_non_distinct_derivations (n p : ℕ) 
  (h1 : is_prime p) 
  (h2 : n ≥ p^2) :
  (Finset.filter (fun x => ¬(derived_numbers p x.1 x.2).Nodup) 
    (Finset.product (Finset.range (n - p) \ {0}) (Finset.range (n - p) \ {0}))).card 
  = count_divisors (p - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derived_numbers_distinct_count_non_distinct_derivations_l137_13715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l137_13794

/-- A function that determines the quadrant of an angle in radians -/
noncomputable def quadrant (θ : ℝ) : ℕ :=
  let θ_norm := θ % (2 * Real.pi)
  if 0 ≤ θ_norm ∧ θ_norm < Real.pi / 2 then 1
  else if Real.pi / 2 ≤ θ_norm ∧ θ_norm < Real.pi then 2
  else if Real.pi ≤ θ_norm ∧ θ_norm < 3 * Real.pi / 2 then 3
  else 4

/-- Theorem stating that if α is in the second quadrant, 
    then α/2 is in either the first or third quadrant -/
theorem half_angle_quadrant (α : ℝ) (h : quadrant α = 2) :
  quadrant (α / 2) = 1 ∨ quadrant (α / 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l137_13794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_sizes_l137_13705

/-- Represents a stratified sampling problem for water heaters from two factories. -/
structure WaterHeaterSampling where
  total_batch : ℕ
  factory_a_count : ℕ
  factory_b_count : ℕ
  sample_size : ℕ

/-- Calculates the expected number of items in a sample from a given stratum. -/
def expected_sample_size (stratum_size : ℕ) (total : ℕ) (sample : ℕ) : ℚ :=
  (stratum_size : ℚ) * (sample : ℚ) / (total : ℚ)

/-- Rounds a rational number to the nearest natural number. -/
def roundToNearestNat (q : ℚ) : ℕ :=
  (q + 1/2).floor.toNat

/-- Theorem stating the correct stratified sample sizes for the water heater problem. -/
theorem stratified_sample_sizes (problem : WaterHeaterSampling) 
  (h1 : problem.total_batch = 98)
  (h2 : problem.factory_a_count = 56)
  (h3 : problem.factory_b_count = 42)
  (h4 : problem.sample_size = 14)
  (h5 : problem.total_batch = problem.factory_a_count + problem.factory_b_count) :
  roundToNearestNat (expected_sample_size problem.factory_a_count problem.total_batch problem.sample_size) = 8 ∧
  roundToNearestNat (expected_sample_size problem.factory_b_count problem.total_batch problem.sample_size) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_sizes_l137_13705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_grid_count_l137_13797

/-- Represents a 3x3 grid filled with numbers 1, 2, 3 -/
def Grid := Fin 3 → Fin 3 → Fin 3

/-- Checks if a row in the grid has no repeated numbers -/
def validRow (g : Grid) (row : Fin 3) : Prop :=
  ∀ i j : Fin 3, i ≠ j → g row i ≠ g row j

/-- Checks if a column in the grid has no repeated numbers -/
def validColumn (g : Grid) (col : Fin 3) : Prop :=
  ∀ i j : Fin 3, i ≠ j → g i col ≠ g j col

/-- Checks if the entire grid is valid (no repeated numbers in rows or columns) -/
def validGrid (g : Grid) : Prop :=
  (∀ row : Fin 3, validRow g row) ∧ (∀ col : Fin 3, validColumn g col)

/-- The set of all valid grid configurations -/
def validGrids : Set Grid :=
  { g : Grid | validGrid g }

/-- Instance to show that validGrids is finite -/
instance : Fintype validGrids := by
  sorry

/-- The theorem stating that the number of valid grid configurations is 12 -/
theorem valid_grid_count : Fintype.card validGrids = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_grid_count_l137_13797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_sin_theta_l137_13708

theorem perpendicular_lines_sin_theta (θ : Real) (h1 : θ ∈ Set.Ioo 0 (π / 2)) :
  (∀ x y : Real, x * Real.cos θ + 2 * y + 1 = 0 → x - y * Real.sin (2 * θ) - 3 = 0 → 
    (Real.cos θ / 2) * Real.sin (2 * θ) = -1) →
  Real.sin θ = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_sin_theta_l137_13708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_equals_consecutive_product_l137_13746

theorem power_product_equals_consecutive_product :
  {(a, b, k) : ℕ × ℕ × ℕ | 2^a * 3^b = k * (k + 1)} =
  {(1, 0, 1), (1, 1, 2), (3, 2, 8), (2, 1, 3)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_equals_consecutive_product_l137_13746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_upper_bound_l137_13716

/-- A sequence defined recursively -/
noncomputable def a (b : ℝ) : ℕ → ℝ
  | 0 => b
  | n + 1 => ((n + 2) * b * a b n) / (a b n + 2 * (n + 2) - 2)

/-- Theorem stating the upper bound of the sequence -/
theorem a_upper_bound (b : ℝ) (h : 0 < b) :
  ∀ n : ℕ, a b n ≤ (b ^ (n + 1)) / (2 ^ (n + 1)) + 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_upper_bound_l137_13716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l137_13799

noncomputable def distance_between_parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : ℝ :=
  abs (c1 - c2) / Real.sqrt (a1^2 + b1^2)

def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 3 = 0

def line2 (x y m : ℝ) : Prop := 6 * x + m * y + 1 = 0

def lines_parallel (m : ℝ) : Prop := (3 : ℝ) / 2 = 6 / m

theorem distance_between_given_lines :
  ∃ (m : ℝ), lines_parallel m ∧ 
  distance_between_parallel_lines 3 2 (-3) 6 m (-1) = 2 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l137_13799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_paths_ratio_l137_13795

/-- Represents a rectangle on a lattice -/
structure LatticeRectangle where
  width : ℕ
  height : ℕ

/-- Represents the number of shortest paths between two points on a lattice -/
def num_shortest_paths (start_x start_y end_x end_y : ℕ) : ℕ :=
  Nat.choose ((end_x - start_x) + (end_y - start_y)) (end_x - start_x)

theorem shortest_paths_ratio (rect : LatticeRectangle) (k : ℕ) 
  (h : rect.height = k * rect.width) :
  num_shortest_paths 0 1 rect.width rect.height = 
  k * num_shortest_paths 1 0 rect.width rect.height := by
  sorry

#check shortest_paths_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_paths_ratio_l137_13795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l137_13760

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 10)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 10} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l137_13760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_Cn_neq_Dn_l137_13793

/-- Cn is the sum of the first n terms of the geometric series 256 + 256/4 + 256/16 + ... -/
noncomputable def Cn (n : ℕ) : ℝ := 256 * (1 - (1/4)^n) / (1 - 1/4)

/-- Dn is the sum of the first n terms of the geometric series 1024 - 1024/4 + 1024/16 - ... -/
noncomputable def Dn (n : ℕ) : ℝ := 1024 * (1 - (1/(-4))^n) / (1 + 1/4)

/-- The smallest positive integer n for which Cn ≠ Dn is 1 -/
theorem smallest_n_for_Cn_neq_Dn :
  ∃ (n : ℕ), n > 0 ∧ Cn n ≠ Dn n ∧ ∀ (m : ℕ), 0 < m ∧ m < n → Cn m = Dn m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_Cn_neq_Dn_l137_13793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l137_13731

theorem trigonometric_simplification (x y : ℝ) :
  (Real.cos x) ^ 2 + (Real.cos (x - y)) ^ 2 - 2 * (Real.cos x) * (Real.cos y) * (Real.cos (x - y)) = 
  (Real.cos x) ^ 2 + (Real.sin y) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l137_13731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_line_l137_13752

theorem distance_origin_to_line : 
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := -5
  let line_eq := fun (x y : ℝ) ↦ a * x + b * y + c = 0
  let distance := |c| / Real.sqrt (a^2 + b^2)
  distance = Real.sqrt 5 := by
    -- Proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_line_l137_13752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l137_13772

/-- Given two planar vectors a and b satisfying certain conditions, 
    prove that the angle between them is π/3 --/
theorem angle_between_vectors (a b : ℝ × ℝ) :
  ‖a‖ = 3 →
  ‖b‖ = 1 →
  ‖a - 3 • b‖ = 3 →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖)) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l137_13772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sgn_eq_H_minus_H_l137_13732

noncomputable def H (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 else 0

noncomputable def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1 else if x = 0 then 0 else -1

theorem sgn_eq_H_minus_H (x : ℝ) : sgn x = H x - H (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sgn_eq_H_minus_H_l137_13732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_players_l137_13726

-- Define the sets of players for each sport
variable (K Kh S B V Ba : Finset ℕ)

-- Define the cardinalities of the sets
axiom card_K : K.card = 60
axiom card_Kh : Kh.card = 90
axiom card_S : S.card = 40
axiom card_B : B.card = 70
axiom card_V : V.card = 50
axiom card_Ba : Ba.card = 30

-- Define the cardinalities of intersections of two sets
axiom card_K_inter_Kh : (K ∩ Kh).card = 25
axiom card_K_inter_S : (K ∩ S).card = 15
axiom card_K_inter_B : (K ∩ B).card = 13
axiom card_K_inter_V : (K ∩ V).card = 20
axiom card_K_inter_Ba : (K ∩ Ba).card = 10
axiom card_Kh_inter_S : (Kh ∩ S).card = 35
axiom card_Kh_inter_B : (Kh ∩ B).card = 16
axiom card_Kh_inter_V : (Kh ∩ V).card = 30
axiom card_Kh_inter_Ba : (Kh ∩ Ba).card = 12
axiom card_S_inter_B : (S ∩ B).card = 20
axiom card_S_inter_V : (S ∩ V).card = 18
axiom card_S_inter_Ba : (S ∩ Ba).card = 7
axiom card_B_inter_V : (B ∩ V).card = 15
axiom card_B_inter_Ba : (B ∩ Ba).card = 8
axiom card_V_inter_Ba : (V ∩ Ba).card = 10

-- Define the cardinalities of intersections of three sets
axiom card_K_inter_Kh_inter_S : (K ∩ Kh ∩ S).card = 5
axiom card_K_inter_B_inter_V : (K ∩ B ∩ V).card = 4
axiom card_S_inter_B_inter_Ba : (S ∩ B ∩ Ba).card = 3
axiom card_V_inter_Ba_inter_Kh : (V ∩ Ba ∩ Kh).card = 2

-- Define the cardinality of the intersection of all sets
axiom card_all_inter : (K ∩ Kh ∩ S ∩ B ∩ V ∩ Ba).card = 1

-- Theorem to prove
theorem total_players : (K ∪ Kh ∪ S ∪ B ∪ V ∪ Ba).card = 139 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_players_l137_13726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l137_13758

/-- Calculates the time (in seconds) for a train to pass a bridge -/
noncomputable def time_to_pass_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Proves that a 485m long train traveling at 45 km/h takes 50 seconds to pass a 140m long bridge -/
theorem train_bridge_passing_time :
  time_to_pass_bridge 485 45 140 = 50 := by
  sorry

-- Use #eval only for nat, and use #check for ℝ
#check time_to_pass_bridge 485 45 140

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l137_13758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l137_13783

/-- Given vectors a and b in ℝ², prove that the value of lambda that makes (a - lambda*b) perpendicular to a is -2/3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (1, -1)) (h2 : b = (-1, 2)) :
  ∃ lambda : ℝ, (a.1 - lambda * b.1) * a.1 + (a.2 - lambda * b.2) * a.2 = 0 ∧ lambda = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l137_13783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ratio_l137_13713

/-- A point with integer coordinates on the circle x^2 + y^2 = 16 -/
structure CirclePoint where
  x : ℤ
  y : ℤ
  on_circle : x^2 + y^2 = 16

/-- The distance between two CirclePoints -/
noncomputable def distance (p q : CirclePoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Statement: The maximum ratio of two irrational distances on the circle is 2 -/
theorem max_distance_ratio (A B C D : CirclePoint) 
  (hAB : Irrational (distance A B))
  (hCD : Irrational (distance C D))
  (hDistinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  (∃ (E F G H : CirclePoint), 
    Irrational (distance E F) ∧ 
    Irrational (distance G H) ∧
    (distance E F) / (distance G H) ≤ 2) ∧
  (∀ (E F G H : CirclePoint),
    Irrational (distance E F) → 
    Irrational (distance G H) → 
    (distance E F) / (distance G H) ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ratio_l137_13713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_periodic_l137_13755

noncomputable def f (x : ℝ) : ℝ := (1 + Real.cos (2 * x)) * (Real.sin x) ^ 2

theorem f_is_even_and_periodic : 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, f (x + Real.pi/2) = f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_periodic_l137_13755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_3_l137_13767

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y² = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a hyperbola of the form x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def hyperbola_eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance from a point to the directrix of a parabola -/
noncomputable def distance_to_directrix (p : Point) (par : Parabola) : ℝ :=
  p.x + par.p / 2

theorem hyperbola_eccentricity_sqrt_3
  (c₁ : Parabola)
  (c₂ : Hyperbola)
  (A : Point)
  (h_on_parabola : A.y^2 = 2 * c₁.p * A.x)
  (h_on_asymptote : A.y / A.x = c₂.b / c₂.a)
  (h_distance : distance_to_directrix A c₁ = 3 * c₁.p / 2) :
  hyperbola_eccentricity c₂ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_3_l137_13767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_archimedean_spiral_end_point_l137_13710

noncomputable section

open Real

/-- Archimedean spiral function -/
def spiral (φ : ℝ) : ℝ := 0.5 * φ

/-- Set of φ values -/
def φ_values : List ℝ :=
  [0, π/4, π/2, 3*π/4, π, 5*π/4, 3*π/2, 7*π/4, 2*π]

theorem archimedean_spiral_end_point :
  ∀ φ ∈ φ_values, 0 ≤ φ ∧ φ ≤ 2*π →
  spiral (2*π) = π := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_archimedean_spiral_end_point_l137_13710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocal_distances_l137_13779

noncomputable section

/-- Line l with parametric equations x = 1 + (√3/2)t, y = (1/2)t -/
def line_l (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 3 / 2) * t, (1 / 2) * t)

/-- Curve C with polar equation ρ = 4cos θ -/
def curve_C (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ * Real.cos θ, 4 * Real.cos θ * Real.sin θ)

/-- Point P where line l intersects the x-axis -/
def point_P : ℝ × ℝ := (1, 0)

/-- Points A and B where line l intersects curve C -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, line_l t = p ∧ ∃ θ, curve_C θ = p}

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem intersection_sum_reciprocal_distances :
  ∃ A B, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    (1 / distance point_P A) + (1 / distance point_P B) = Real.sqrt 15 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocal_distances_l137_13779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_contains_one_or_four_l137_13762

def digitSquareSum (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d => d * d) |>.sum

def sequenceA : ℕ → ℕ → ℕ
  | a₁, 0 => a₁
  | a₁, n + 1 => digitSquareSum (sequenceA a₁ n)

theorem sequence_contains_one_or_four (a₁ : ℕ) (h : 100 ≤ a₁ ∧ a₁ ≤ 999) :
  ∃ n : ℕ, sequenceA a₁ n = 1 ∨ sequenceA a₁ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_contains_one_or_four_l137_13762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionAreaTheorem_l137_13720

/-- Representation of a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of a unit cube with specific points -/
structure UnitCube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D
  E : Point3D -- midpoint of CC₁
  F : Point3D -- midpoint of DD₁

/-- The cross-section area of a unit cube's circumscribed sphere cut by plane AEF -/
noncomputable def crossSectionArea (cube : UnitCube) : ℝ :=
  (7 / 10) * Real.pi

/-- Theorem: The cross-section area of a unit cube's circumscribed sphere cut by plane AEF is 7π/10 -/
theorem crossSectionAreaTheorem (cube : UnitCube) :
  crossSectionArea cube = (7 / 10) * Real.pi :=
by
  -- Unfold the definition of crossSectionArea
  unfold crossSectionArea
  -- The equality now follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionAreaTheorem_l137_13720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_2sin_max_value_l137_13763

theorem cos_plus_2sin_max_value (x : ℝ) : Real.cos x + 2 * Real.sin x ≤ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_2sin_max_value_l137_13763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_collection_books_l137_13724

/-- The number of books in a special collection at the end of a month, given the initial number of books,
    the percentage of loaned books returned, and the number of books loaned out. -/
def books_at_end_of_month (initial_books : ℕ) (return_rate : ℚ) (loaned_books : ℕ) : ℕ :=
  initial_books - (loaned_books - Int.toNat ((return_rate * loaned_books).floor))

/-- Theorem stating that the number of books in the special collection at the end of the month is 69. -/
theorem special_collection_books : books_at_end_of_month 75 (4/5) 30 = 69 := by
  -- Unfold the definition of books_at_end_of_month
  unfold books_at_end_of_month
  -- Evaluate the expression
  norm_num
  -- QED
  rfl

#eval books_at_end_of_month 75 (4/5) 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_collection_books_l137_13724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jorge_goals_three_seasons_l137_13709

theorem jorge_goals_three_seasons 
  (last_season : ℕ) 
  (season_before_last_percentage : ℚ)
  (this_season : ℕ) 
  (increase_percentage : ℚ) :
  last_season = 156 →
  season_before_last_percentage = 80/100 →
  this_season = 187 →
  increase_percentage = 25/100 →
  ∃ (season_before_last : ℕ),
    season_before_last = (last_season : ℚ) / season_before_last_percentage ∧
    this_season = last_season + (increase_percentage * last_season).floor ∧
    season_before_last + last_season + this_season = 546 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jorge_goals_three_seasons_l137_13709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_expression_value_l137_13711

theorem rational_expression_value (a b : ℚ) (h : a + 2 * b = 0) :
  |a / abs b - 1| + |abs a / b - 2| + |abs (a / b) - 3| = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_expression_value_l137_13711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_method_C_not_systematic_l137_13727

/-- Abstract type for sampling methods -/
structure SamplingMethod :=
  (method : String)

/-- Definition of systematic sampling -/
def systematic_sampling (method : SamplingMethod) : Prop :=
  ∃ (large_balanced_pool regular_extraction : SamplingMethod → Prop),
    large_balanced_pool method ∧ regular_extraction method

/-- Definition of the sampling method C -/
def method_C : SamplingMethod :=
  { method := "random selection at supermarket entrance" }

/-- Theorem: Method C is not systematic sampling -/
theorem method_C_not_systematic : ¬(systematic_sampling method_C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_method_C_not_systematic_l137_13727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l137_13712

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * (1 - x) else x * (1 + x)

-- State the theorem
theorem odd_function_extension :
  (∀ x, f (-x) = -f x) ∧ (∀ x ≥ 0, f x = x * (1 - x)) →
  ∀ x ≤ 0, f x = x * (1 + x) :=
by
  intro h
  intro x hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l137_13712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_one_l137_13717

/-- An odd function f defined on ℝ with f(x) = 3^x - x + 5b for x ≥ 0 -/
noncomputable def f (b : ℝ) : ℝ → ℝ :=
  fun x => if x ≥ 0 then Real.exp (Real.log 3 * x) - x + 5*b else -(Real.exp (Real.log 3 * (-x)) - (-x) + 5*b)

/-- Theorem: f(-1) = -1 for the given odd function f -/
theorem f_neg_one_eq_neg_one (b : ℝ) : f b (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_one_l137_13717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cone_lateral_area_ratio_l137_13736

/-- The lateral surface area ratio of a cylinder and a cone with equal slant heights and base radii -/
theorem cylinder_cone_lateral_area_ratio 
  (r : ℝ) -- radius of the base
  (h : ℝ) -- slant height
  (cylinder_area : ℝ := 2 * π * r * h) -- lateral surface area of cylinder
  (cone_area : ℝ := π * r * h) -- lateral surface area of cone
  (hr : r > 0) -- radius is positive
  (hh : h > 0) -- slant height is positive
  : cylinder_area / cone_area = 2 := by
  sorry

#check cylinder_cone_lateral_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cone_lateral_area_ratio_l137_13736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x_fx_lt_gx_l137_13748

noncomputable def f (x : ℝ) : ℝ := (2 / Real.exp 1) ^ x
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 / 3) ^ x

theorem exists_x_fx_lt_gx : ∃ x : ℝ, f x < g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x_fx_lt_gx_l137_13748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_study_time_l137_13723

/-- Represents the duration of study time in minutes -/
def n : ℝ := sorry

/-- Represents the first component of n in the form d - e√f -/
def d : ℕ := sorry

/-- Represents the second component of n in the form d - e√f -/
def e : ℕ := sorry

/-- Represents the third component of n in the form d - e√f -/
def f : ℕ := sorry

/-- The probability that one student arrives while the other is still in the library -/
def overlap_probability : ℝ := 0.3

/-- The time interval in minutes between 3 p.m. and 4 p.m. -/
def time_interval : ℝ := 60

theorem library_study_time :
  (n = d - e * Real.sqrt f) →
  (d > 0) →
  (e > 0) →
  (f > 0) →
  (∀ (p : ℕ), Prime p → ¬(p^2 ∣ f)) →
  (overlap_probability = 1 - (time_interval - n)^2 / time_interval^2) →
  (d + e + f = 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_study_time_l137_13723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_proof_l137_13745

/-- Represents a parabola in the form ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on the parabola --/
def Parabola.contains_point (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Calculates the vertex of the parabola --/
noncomputable def Parabola.vertex (p : Parabola) : ℝ × ℝ :=
  let h := -p.b / (2 * p.a)
  let k := p.a * h^2 + p.b * h + p.c
  (h, k)

/-- Checks if the parabola has a vertical axis of symmetry --/
def Parabola.has_vertical_axis_of_symmetry (p : Parabola) : Prop :=
  p.a ≠ 0

theorem parabola_equation_proof (p : Parabola) :
  p.a = -3 ∧ p.b = 12 ∧ p.c = -8 →
  p.vertex = (2, 4) ∧
  p.has_vertical_axis_of_symmetry ∧
  p.contains_point 1 1 := by
  sorry

#check parabola_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_proof_l137_13745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_theorem_l137_13759

-- Define the curve C in polar coordinates
noncomputable def curve_C (a : ℝ) (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 = 2 * a * Real.cos θ

-- Define the line l in parametric form
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-2 + Real.sqrt 2 / 2 * t, -4 + Real.sqrt 2 / 2 * t)

-- Define point P
def point_P : ℝ × ℝ := (-2, -4)

-- Define the intersection condition
def intersects (C : (ℝ → ℝ → Prop)) (l : ℝ → ℝ × ℝ) : Prop := 
  ∃ t₁ t₂, C (l t₁).1 (l t₁).2 ∧ C (l t₂).1 (l t₂).2 ∧ t₁ ≠ t₂

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- State the theorem
theorem curve_intersection_theorem (a : ℝ) (h₁ : a > 0) :
  intersects (curve_C a) line_l →
  (∃ t₁ t₂, t₁ ≠ t₂ ∧ 
    distance point_P (line_l t₁) * distance point_P (line_l t₂) = 
    distance (line_l t₁) (line_l t₂)^2) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_theorem_l137_13759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_groups_correct_l137_13721

structure Team :=
  (new_players : ℕ)
  (returning_players : ℕ)

def valid_group (t : Team) (g : ℕ) : Prop :=
  g * 2 ≤ t.new_players ∧ g * 3 ≤ t.returning_players

def max_groups (t : Team) : ℕ :=
  min (t.new_players / 2) (t.returning_players / 3)

theorem max_groups_correct (t : Team) (h1 : t.new_players = 4) (h2 : t.returning_players = 6) :
  max_groups t = 2 ∧ valid_group t (max_groups t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_groups_correct_l137_13721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_derivative_bound_l137_13754

open Real

-- Define the function f and its derivative
noncomputable def f (a b x : ℝ) : ℝ := -1/3 * x^3 + 2*a * x^2 - 3*a^2 * x + b
noncomputable def f_derivative (a x : ℝ) : ℝ := -x^2 + 4*a*x - 3*a^2

theorem function_derivative_bound (a b : ℝ) (ha : 0 < a) (ha' : a < 1) :
  (∀ x ∈ Set.Icc (a + 1) (a + 2), |f_derivative a x| ≤ a) ↔ 4/5 ≤ a ∧ a < 1 := by
  sorry

#check function_derivative_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_derivative_bound_l137_13754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_plus_y_max_x_plus_y_value_l137_13782

theorem max_x_plus_y (x y : ℕ) (h : (x - 4) * (x - 10) = 2^y) :
  ∀ (a b : ℕ), (a - 4) * (a - 10) = 2^b → x + y ≥ a + b :=
by sorry

theorem max_x_plus_y_value (x y : ℕ) (h : (x - 4) * (x - 10) = 2^y) :
  ∃ (a b : ℕ), (a - 4) * (a - 10) = 2^b ∧ a + b = 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_plus_y_max_x_plus_y_value_l137_13782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antiderivative_of_f_through_point_l137_13744

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x

-- Define the antiderivative F
def F (x : ℝ) : ℝ := x^2 + 2

-- Theorem statement
theorem antiderivative_of_f_through_point :
  (∀ x, (deriv F x) = f x) ∧ F 1 = 3 := by
  constructor
  · intro x
    -- Prove that the derivative of F is equal to f
    calc
      deriv F x = deriv (fun x => x^2 + 2) x := by rfl
      _ = 2 * x := by simp [deriv_add_const, deriv_pow]
      _ = f x := by rfl
  · -- Prove that F(1) = 3
    simp [F]
    norm_num

-- You can add more theorems or lemmas here if needed


end NUMINAMATH_CALUDE_ERRORFEEDBACK_antiderivative_of_f_through_point_l137_13744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_ratio_l137_13706

/-- The ratio of wire lengths forming an equilateral triangle and a square with equal areas -/
theorem wire_ratio (a b : ℝ) (h : a > 0) (k : b > 0) : 
  (Real.sqrt 3 / 4) * (a / 3)^2 = (b / 4)^2 → a / b = 2 * Real.sqrt 3 / 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_ratio_l137_13706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_homes_l137_13792

/-- Calculates the total distance between two people's homes given their speeds and the distance one has traveled when they meet in the middle. -/
noncomputable def totalDistance (maxwellSpeed bradSpeed maxwellDistance : ℝ) : ℝ :=
  let travelTime := maxwellDistance / maxwellSpeed
  let bradDistance := bradSpeed * travelTime
  maxwellDistance + bradDistance

/-- Theorem stating that given the problem conditions, the total distance between Maxwell and Brad's homes is 50 km. -/
theorem distance_between_homes :
  let maxwellSpeed : ℝ := 4
  let bradSpeed : ℝ := 6
  let maxwellDistance : ℝ := 20
  totalDistance maxwellSpeed bradSpeed maxwellDistance = 50 := by
  -- Unfold the definition of totalDistance
  unfold totalDistance
  -- Simplify the expression
  simp
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_homes_l137_13792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_problem_l137_13735

theorem gas_cost_problem (initial_friends final_friends : ℕ) 
  (cost_decrease : ℚ) (total_cost : ℚ) : 
  initial_friends = 5 →
  final_friends = 8 →
  cost_decrease = 15 →
  (total_cost / initial_friends) - (total_cost / final_friends) = cost_decrease →
  total_cost = 200 :=
by
  intros h1 h2 h3 h4
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_problem_l137_13735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geralds_speed_is_three_l137_13771

/-- Calculates the average speed of Gerald's car given the track length, Polly's laps, and time. -/
noncomputable def geralds_speed (track_length : ℝ) (polly_laps : ℕ) (time : ℝ) : ℝ :=
  let polly_distance := track_length * (polly_laps : ℝ)
  let polly_speed := polly_distance / time
  polly_speed / 2

/-- Theorem stating that Gerald's speed is 3 miles per hour under the given conditions. -/
theorem geralds_speed_is_three :
  geralds_speed 0.25 12 0.5 = 3 := by
  -- Unfold the definition of geralds_speed
  unfold geralds_speed
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geralds_speed_is_three_l137_13771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sequence_sum_l137_13743

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sequence_term (n : ℕ) : ℕ := factorial n + n

def sequence_sum : ℕ := (List.range 12).map (λ i => sequence_term (i + 1)) |>.sum

theorem units_digit_of_sequence_sum :
  sequence_sum % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sequence_sum_l137_13743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_n4_minus_n2_l137_13700

theorem largest_divisor_of_n4_minus_n2 :
  ∃ (k : ℕ), k > 0 ∧
  (∀ (n : ℤ), (k : ℤ) ∣ (n^4 - n^2)) ∧
  (∀ (m : ℕ), m > k → ∃ (n : ℤ), ¬((m : ℤ) ∣ (n^4 - n^2))) ∧
  k = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_n4_minus_n2_l137_13700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_like_terms_l137_13702

/-- Two terms are considered like terms if they have the same variables raised to the same powers. -/
def like_terms (term1 term2 : MvPolynomial (Fin 2) ℚ) : Prop :=
  term1.support = term2.support

/-- The first term in the problem -/
noncomputable def term1 : MvPolynomial (Fin 2) ℚ :=
  3 * (MvPolynomial.X 0)^2 * (MvPolynomial.X 1)

/-- The second term in the problem -/
noncomputable def term2 : MvPolynomial (Fin 2) ℚ :=
  -6 * (MvPolynomial.X 0) * (MvPolynomial.X 1)^2

/-- Theorem stating that term1 and term2 are not like terms -/
theorem not_like_terms : ¬(like_terms term1 term2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_like_terms_l137_13702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l137_13751

theorem quadratic_equation_properties (k : ℝ) :
  (∀ x : ℝ, (x^2 - k*x - k - 1 = 0) → (k^2 + 4*k + 4 ≥ 0)) ∧
  (∃ x : ℝ, x^2 - k*x - k - 1 = 0 ∧ x > 0) → k > -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l137_13751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l137_13781

theorem solve_exponential_equation :
  ∃ x : ℚ, (3 : ℝ) ^ (3 * (x : ℝ)) = 27 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l137_13781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_outside_interval_l137_13780

/-- The function f(x) = ln x - x/a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - x / a

/-- Theorem: If a > 0 and there exists x₀ such that f(x₁) < f(x₀) for all x₁ ∈ [1,2],
    then a ∈ (0,1) ∪ (2,+∞) -/
theorem function_max_outside_interval (a : ℝ) (h_a : a > 0) :
  (∃ x₀ : ℝ, ∀ x₁ : ℝ, x₁ ∈ Set.Icc 1 2 → f a x₁ < f a x₀) →
  a ∈ Set.union (Set.Ioo 0 1) (Set.Ioi 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_outside_interval_l137_13780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_R_l137_13789

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a square -/
structure Square where
  W : Point
  X : Point
  Y : Point
  Z : Point

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Check if a square is unit square -/
def isUnitSquare (s : Square) : Prop := sorry

/-- Check if a point is inside a square -/
def isInside (p : Point) (s : Square) : Prop := sorry

/-- Check if a point is on a side of a square -/
def isOnSide (p : Point) (s : Square) : Prop := sorry

/-- Check if a point is inside a triangle -/
def isInsideTriangle (p : Point) (t : Triangle) : Prop := sorry

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculate the area of a region -/
noncomputable def areaOfRegion (s : Square) (t : Triangle) : ℝ := sorry

/-- Main theorem -/
theorem area_of_region_R (s : Square) (t : Triangle) :
  isUnitSquare s →
  isEquilateral t →
  isInside t.A s →
  isOnSide t.C s →
  isOnSide t.B s →
  (∀ p : Point, isInside p s ∧ ¬(isInsideTriangle p t) → 
    1/4 < distance p s.Y ∧ distance p s.Y < 1/2) →
  areaOfRegion s t = (1 - Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_R_l137_13789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_polygon_sides_l137_13770

/-- The sum of interior angles of a polygon with n sides --/
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- The possible number of sides for the original polygon --/
def possible_sides : Set ℕ := {10, 11, 12}

/-- Theorem stating the possible number of sides for the original polygon --/
theorem original_polygon_sides (n : ℕ) :
  (∃ k : ℕ, k ∈ ({n - 1, n, n + 1} : Set ℕ) ∧ interior_angle_sum k = 1620) →
  n ∈ possible_sides :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_polygon_sides_l137_13770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_proof_l137_13753

theorem election_votes_proof (V : ℤ) (W L : ℤ) : 
  (W - L = V / 10) →
  ((W - 3000) - (L + 3000) = -(V / 10)) →
  V = 30000 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_proof_l137_13753
