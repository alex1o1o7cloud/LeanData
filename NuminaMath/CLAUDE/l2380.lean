import Mathlib

namespace NUMINAMATH_CALUDE_students_neither_outstanding_nor_pioneer_l2380_238075

theorem students_neither_outstanding_nor_pioneer (total : ℕ) (outstanding : ℕ) (pioneers : ℕ) (both : ℕ)
  (h_total : total = 87)
  (h_outstanding : outstanding = 58)
  (h_pioneers : pioneers = 63)
  (h_both : both = 49) :
  total - outstanding - pioneers + both = 15 :=
by sorry

end NUMINAMATH_CALUDE_students_neither_outstanding_nor_pioneer_l2380_238075


namespace NUMINAMATH_CALUDE_code_problem_l2380_238078

theorem code_problem (A B C : ℕ) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 →
  B > A →
  A < C →
  11 * B + 11 * A + 11 * C = 242 →
  ((A = 5 ∧ B = 8 ∧ C = 9) ∨ (A = 5 ∧ B = 9 ∧ C = 8)) :=
by sorry

end NUMINAMATH_CALUDE_code_problem_l2380_238078


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2380_238053

theorem complex_equation_solution (z : ℂ) : 
  z + (1 + 2*I) = 10 - 3*I → z = 9 - 5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2380_238053


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l2380_238017

/-- The area of the triangle formed by the tangent line to y = e^x at (2, e^2) and the coordinate axes -/
theorem tangent_triangle_area : 
  let f : ℝ → ℝ := fun x ↦ Real.exp x
  let tangent_point : ℝ × ℝ := (2, Real.exp 2)
  let slope : ℝ := Real.exp 2
  let x_intercept : ℝ := 1
  let y_intercept : ℝ := Real.exp 2
  let triangle_area : ℝ := (1/2) * Real.exp 2
  triangle_area = (1/2) * y_intercept * x_intercept :=
by sorry


end NUMINAMATH_CALUDE_tangent_triangle_area_l2380_238017


namespace NUMINAMATH_CALUDE_plane_graph_is_bipartite_plane_regions_two_colorable_l2380_238002

/-- A graph representing regions formed by lines dividing a plane -/
structure PlaneGraph where
  V : Type* -- Vertices (regions)
  E : V → V → Prop -- Edges (neighboring regions)

/-- Definition of a bipartite graph -/
def IsBipartite (G : PlaneGraph) : Prop :=
  ∃ (A B : Set G.V), (∀ v, v ∈ A ∨ v ∈ B) ∧ 
    (∀ u v, G.E u v → (u ∈ A ∧ v ∈ B) ∨ (u ∈ B ∧ v ∈ A))

/-- Theorem: The graph representing regions formed by lines dividing a plane is bipartite -/
theorem plane_graph_is_bipartite (G : PlaneGraph) : IsBipartite G := by
  sorry

/-- Corollary: Regions formed by lines dividing a plane can be colored with two colors -/
theorem plane_regions_two_colorable (G : PlaneGraph) : 
  ∃ (color : G.V → Bool), ∀ u v, G.E u v → color u ≠ color v := by
  sorry

end NUMINAMATH_CALUDE_plane_graph_is_bipartite_plane_regions_two_colorable_l2380_238002


namespace NUMINAMATH_CALUDE_distance_after_three_minutes_l2380_238082

/-- The distance between two vehicles after a given time, given their speeds and initial positions -/
def distance_between_vehicles (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed2 - speed1) * time

theorem distance_after_three_minutes :
  let truck_speed : ℝ := 65
  let car_speed : ℝ := 85
  let time_in_hours : ℝ := 3 / 60
  distance_between_vehicles truck_speed car_speed time_in_hours = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_three_minutes_l2380_238082


namespace NUMINAMATH_CALUDE_triangle_problem_l2380_238015

theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  (2 * a - c) * Real.sin A + (2 * c - a) * Real.sin C = 2 * b * Real.sin B →
  b = 1 →
  B = π / 3 ∧ 
  ∃ (p : ℝ), p = a + b + c ∧ Real.sqrt 3 + 1 < p ∧ p ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2380_238015


namespace NUMINAMATH_CALUDE_set_equality_l2380_238028

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {2, 3}
def M : Set ℕ := {x | x ∈ P ∧ x ∉ Q}

theorem set_equality : M = {1} := by sorry

end NUMINAMATH_CALUDE_set_equality_l2380_238028


namespace NUMINAMATH_CALUDE_arithmetic_mean_multiplied_by_three_l2380_238087

theorem arithmetic_mean_multiplied_by_three (a b c d e : ℝ) :
  let original_set := [a, b, c, d, e]
  let new_set := original_set.map (· * 3)
  let original_mean := (a + b + c + d + e) / 5
  let new_mean := (new_set.sum) / 5
  new_mean = 3 * original_mean := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_multiplied_by_three_l2380_238087


namespace NUMINAMATH_CALUDE_clock_strike_duration_clock_strike_six_duration_l2380_238051

-- Define the clock striking behavior
def ClockStrike (strikes : ℕ) (duration : ℝ) : Prop :=
  strikes > 0 ∧ duration > 0 ∧ (strikes - 1) * (duration / (strikes - 1)) = duration

-- Theorem statement
theorem clock_strike_duration (strikes₁ strikes₂ : ℕ) (duration₁ : ℝ) :
  ClockStrike strikes₁ duration₁ →
  strikes₂ > strikes₁ →
  ClockStrike strikes₂ ((strikes₂ - 1) * (duration₁ / (strikes₁ - 1))) :=
by
  sorry

-- The specific problem instance
theorem clock_strike_six_duration :
  ClockStrike 3 12 → ClockStrike 6 30 :=
by
  sorry

end NUMINAMATH_CALUDE_clock_strike_duration_clock_strike_six_duration_l2380_238051


namespace NUMINAMATH_CALUDE_mary_screws_on_hand_l2380_238039

def screws_needed (sections : ℕ) (screws_per_section : ℕ) : ℕ :=
  sections * screws_per_section

theorem mary_screws_on_hand 
  (sections : ℕ) 
  (screws_per_section : ℕ) 
  (buy_ratio : ℕ) 
  (h1 : sections = 4) 
  (h2 : screws_per_section = 6) 
  (h3 : buy_ratio = 2) :
  ∃ (initial_screws : ℕ), 
    initial_screws + buy_ratio * initial_screws = screws_needed sections screws_per_section ∧ 
    initial_screws = 8 :=
by sorry

end NUMINAMATH_CALUDE_mary_screws_on_hand_l2380_238039


namespace NUMINAMATH_CALUDE_scrunchies_to_barrettes_ratio_l2380_238010

/-- Represents the number of hair decorations Annie has --/
structure HairDecorations where
  barrettes : ℕ
  scrunchies : ℕ
  bobby_pins : ℕ

/-- Calculates the percentage of bobby pins in the total hair decorations --/
def bobby_pin_percentage (hd : HairDecorations) : ℚ :=
  (hd.bobby_pins : ℚ) / ((hd.barrettes + hd.scrunchies + hd.bobby_pins) : ℚ) * 100

/-- Theorem stating the ratio of scrunchies to barrettes --/
theorem scrunchies_to_barrettes_ratio (hd : HairDecorations) :
  hd.barrettes = 6 →
  hd.bobby_pins = hd.barrettes - 3 →
  bobby_pin_percentage hd = 14 →
  (hd.scrunchies : ℚ) / (hd.barrettes : ℚ) = 2 := by
  sorry

#check scrunchies_to_barrettes_ratio

end NUMINAMATH_CALUDE_scrunchies_to_barrettes_ratio_l2380_238010


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_l2380_238079

/-- Two planar vectors a and b are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_k (k : ℝ) :
  let a : ℝ × ℝ := (k, 3)
  let b : ℝ × ℝ := (1, 4)
  perpendicular a b → k = -12 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_l2380_238079


namespace NUMINAMATH_CALUDE_unique_n_congruence_l2380_238067

theorem unique_n_congruence : ∃! n : ℕ, 5 ≤ n ∧ n ≤ 10 ∧ n % 6 = 12345 % 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_congruence_l2380_238067


namespace NUMINAMATH_CALUDE_freshmen_psych_majors_percentage_l2380_238060

/-- The percentage of freshmen psychology majors in the School of Liberal Arts
    among all students at a certain college. -/
theorem freshmen_psych_majors_percentage
  (total_students : ℕ)
  (freshmen_percentage : ℚ)
  (liberal_arts_percentage : ℚ)
  (psychology_percentage : ℚ)
  (h1 : freshmen_percentage = 2/5)
  (h2 : liberal_arts_percentage = 1/2)
  (h3 : psychology_percentage = 1/2)
  : (freshmen_percentage * liberal_arts_percentage * psychology_percentage : ℚ) = 1/10 := by
  sorry

#check freshmen_psych_majors_percentage

end NUMINAMATH_CALUDE_freshmen_psych_majors_percentage_l2380_238060


namespace NUMINAMATH_CALUDE_division_property_l2380_238095

theorem division_property (n : ℕ) (hn : n > 0) :
  (5^(n-1) + 3^(n-1)) ∣ (5^n + 3^n) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_property_l2380_238095


namespace NUMINAMATH_CALUDE_chocolate_cost_720_l2380_238072

/-- Calculates the cost of buying a certain number of chocolate candies given the following conditions:
  - A box contains 30 chocolate candies
  - A box costs $10
  - If a customer buys more than 20 boxes, they get a 10% discount
-/
def chocolateCost (numCandies : ℕ) : ℚ :=
  let boxSize := 30
  let boxPrice := 10
  let discountThreshold := 20
  let discountRate := 0.1
  let numBoxes := (numCandies + boxSize - 1) / boxSize  -- Ceiling division
  let totalCost := numBoxes * boxPrice
  if numBoxes > discountThreshold then
    totalCost * (1 - discountRate)
  else
    totalCost

theorem chocolate_cost_720 : chocolateCost 720 = 216 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_720_l2380_238072


namespace NUMINAMATH_CALUDE_buffer_solution_composition_l2380_238094

/-- Represents the composition of a buffer solution -/
structure BufferSolution where
  chemicalA : Real
  water : Real
  chemicalB : Real
  totalVolume : Real

/-- Defines the specific buffer solution composition -/
def specificBuffer : BufferSolution where
  chemicalA := 0.05
  water := 0.025
  chemicalB := 0.02
  totalVolume := 0.075

/-- Theorem stating the required amounts of water and chemical B for 1.2 liters of buffer solution -/
theorem buffer_solution_composition 
  (desiredVolume : Real)
  (h1 : desiredVolume = 1.2) :
  let waterNeeded := desiredVolume * (specificBuffer.water / specificBuffer.totalVolume)
  let chemicalBNeeded := desiredVolume * (specificBuffer.chemicalB / specificBuffer.totalVolume)
  waterNeeded = 0.4 ∧ chemicalBNeeded = 0.032 := by
  sorry

end NUMINAMATH_CALUDE_buffer_solution_composition_l2380_238094


namespace NUMINAMATH_CALUDE_parabola_equation_l2380_238044

/-- A parabola that opens downward with focus at (0, -2) -/
structure DownwardParabola where
  focus : ℝ × ℝ
  opens_downward : focus.2 < 0
  focus_y : focus.1 = 0 ∧ focus.2 = -2

/-- The hyperbola y²/3 - x² = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 / 3 - p.1^2 = 1}

/-- The standard form of a downward-opening parabola -/
def ParabolaEquation (p : ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.1^2 = -2 * p * q.2}

theorem parabola_equation (C : DownwardParabola) 
    (h : C.focus ∈ Hyperbola) : 
    ParabolaEquation 4 = {q : ℝ × ℝ | q.1^2 = -8 * q.2} := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2380_238044


namespace NUMINAMATH_CALUDE_lowest_price_type_a_l2380_238038

/-- Calculates the final price of a pet food type given its MSRP, regular discount, additional discount, and sales tax rate -/
def finalPrice (msrp : ℝ) (regularDiscount : ℝ) (additionalDiscount : ℝ) (salesTax : ℝ) : ℝ :=
  let discountedPrice := msrp * (1 - regularDiscount)
  let furtherDiscountedPrice := discountedPrice * (1 - additionalDiscount)
  furtherDiscountedPrice * (1 + salesTax)

theorem lowest_price_type_a (msrp_a msrp_b msrp_c : ℝ) :
  msrp_a = 45 ∧ msrp_b = 55 ∧ msrp_c = 50 →
  finalPrice msrp_a 0.15 0.20 0.07 < finalPrice msrp_b 0.25 0.15 0.07 ∧
  finalPrice msrp_a 0.15 0.20 0.07 < finalPrice msrp_c 0.30 0.10 0.07 :=
by sorry

end NUMINAMATH_CALUDE_lowest_price_type_a_l2380_238038


namespace NUMINAMATH_CALUDE_parabola_c_value_l2380_238049

/-- A parabola with equation y = 2x^2 + bx + c passes through the points (1, 4) and (5, 4). -/
theorem parabola_c_value (b c : ℝ) : 
  (4 = 2 * 1^2 + b * 1 + c) → 
  (4 = 2 * 5^2 + b * 5 + c) → 
  c = 14 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2380_238049


namespace NUMINAMATH_CALUDE_worker_completion_time_l2380_238004

/-- Given workers x and y, where x can complete a job in 40 days,
    x works for 8 days, and y finishes the remaining work in 16 days,
    prove that y can complete the entire job alone in 20 days. -/
theorem worker_completion_time
  (x_completion_time : ℕ)
  (x_work_days : ℕ)
  (y_completion_time_for_remainder : ℕ)
  (h1 : x_completion_time = 40)
  (h2 : x_work_days = 8)
  (h3 : y_completion_time_for_remainder = 16) :
  (y_completion_time_for_remainder * x_completion_time) / 
  (x_completion_time - x_work_days) = 20 :=
by sorry

end NUMINAMATH_CALUDE_worker_completion_time_l2380_238004


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l2380_238052

/-- The area of the triangle formed by the tangent line to y = log₂ x at (1, 0) and the axes -/
theorem tangent_triangle_area : 
  let f (x : ℝ) := Real.log x / Real.log 2
  let tangent_line (x : ℝ) := (1 / Real.log 2) * (x - 1)
  let x_intercept : ℝ := 1
  let y_intercept : ℝ := -1 / Real.log 2
  let triangle_area : ℝ := (1/2) * x_intercept * (-y_intercept)
  triangle_area = 1 / (2 * Real.log 2) := by sorry

end NUMINAMATH_CALUDE_tangent_triangle_area_l2380_238052


namespace NUMINAMATH_CALUDE_sum_in_base8_l2380_238031

/-- Converts a base-8 number represented as a list of digits to a natural number. -/
def fromBase8 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits. -/
def toBase8 (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: toBase8 (n / 8)

theorem sum_in_base8 :
  let a := fromBase8 [4, 7, 6, 5]
  let b := fromBase8 [5, 6, 3, 2]
  toBase8 (a + b) = [6, 2, 2, 0, 1] := by
  sorry

#eval fromBase8 [4, 7, 6, 5]
#eval fromBase8 [5, 6, 3, 2]
#eval toBase8 (fromBase8 [4, 7, 6, 5] + fromBase8 [5, 6, 3, 2])

end NUMINAMATH_CALUDE_sum_in_base8_l2380_238031


namespace NUMINAMATH_CALUDE_combined_salaries_l2380_238003

theorem combined_salaries (average_salary : ℕ) (b_salary : ℕ) (total_people : ℕ) :
  average_salary = 8200 →
  b_salary = 5000 →
  total_people = 5 →
  (average_salary * total_people) - b_salary = 36000 :=
by sorry

end NUMINAMATH_CALUDE_combined_salaries_l2380_238003


namespace NUMINAMATH_CALUDE_tree_growth_theorem_l2380_238005

-- Define growth rates and initial heights
def growth_rate_A : ℚ := 25  -- 50 cm / 2 weeks
def growth_rate_B : ℚ := 70 / 3
def growth_rate_C : ℚ := 90 / 4
def initial_height_A : ℚ := 200
def initial_height_B : ℚ := 150
def initial_height_C : ℚ := 250
def weeks : ℕ := 16

-- Calculate final heights
def final_height_A : ℚ := initial_height_A + growth_rate_A * weeks
def final_height_B : ℚ := initial_height_B + growth_rate_B * weeks
def final_height_C : ℚ := initial_height_C + growth_rate_C * weeks

-- Define the combined final height
def combined_final_height : ℚ := final_height_A + final_height_B + final_height_C

-- Theorem to prove
theorem tree_growth_theorem :
  (combined_final_height : ℚ) = 1733.33 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_theorem_l2380_238005


namespace NUMINAMATH_CALUDE_vector_norm_sum_l2380_238069

theorem vector_norm_sum (a b c : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ‖a‖ = 2) (h2 : ‖b‖ = 2) (h3 : ‖c‖ = 3) : 
  ‖a - 3 • b‖^2 + ‖b - 3 • c‖^2 + ‖c - 3 • a‖^2 = 170 := by
  sorry

end NUMINAMATH_CALUDE_vector_norm_sum_l2380_238069


namespace NUMINAMATH_CALUDE_revenue_increase_percentage_l2380_238009

/-- Calculates the percentage increase in manufacturer's revenue after changing carton size and price -/
theorem revenue_increase_percentage
  (initial_size : ℝ)
  (initial_price : ℝ)
  (new_size : ℝ)
  (new_price : ℝ)
  (h1 : initial_size = 1)
  (h2 : initial_price = 80)
  (h3 : new_size = 0.9)
  (h4 : new_price = 99)
  : (((new_price / new_size) - (initial_price / initial_size)) / (initial_price / initial_size)) * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_revenue_increase_percentage_l2380_238009


namespace NUMINAMATH_CALUDE_well_depth_l2380_238035

/-- Proves that a circular well with diameter 2 meters and volume 31.41592653589793 cubic meters has a depth of 10 meters -/
theorem well_depth (diameter : ℝ) (volume : ℝ) (depth : ℝ) : 
  diameter = 2 → 
  volume = 31.41592653589793 → 
  volume = Real.pi * (diameter / 2)^2 * depth → 
  depth = 10 := by sorry

end NUMINAMATH_CALUDE_well_depth_l2380_238035


namespace NUMINAMATH_CALUDE_sum_of_abc_l2380_238090

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (eq1 : a^2 + b*c = 115)
  (eq2 : b^2 + a*c = 127)
  (eq3 : c^2 + a*b = 115) :
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_abc_l2380_238090


namespace NUMINAMATH_CALUDE_circle_condition_l2380_238073

/-- Represents a quadratic equation in two variables -/
structure QuadraticEquation :=
  (a b c d e f : ℝ)

/-- Checks if a QuadraticEquation represents a circle -/
def isCircle (eq : QuadraticEquation) : Prop :=
  eq.a = eq.b ∧ eq.d^2 + eq.e^2 - 4*eq.a*eq.f > 0

/-- The equation m^2x^2 + (m+2)y^2 + 2mx + m = 0 -/
def equation (m : ℝ) : QuadraticEquation :=
  ⟨m^2, m+2, 0, 2*m, 0, m⟩

/-- Theorem: The equation represents a circle if and only if m = -1 -/
theorem circle_condition :
  ∀ m : ℝ, isCircle (equation m) ↔ m = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l2380_238073


namespace NUMINAMATH_CALUDE_lcm_gcd_210_396_l2380_238021

theorem lcm_gcd_210_396 :
  (Nat.lcm 210 396 = 4620) ∧ (Nat.gcd 210 396 = 6) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_210_396_l2380_238021


namespace NUMINAMATH_CALUDE_double_root_equation_example_double_root_equation_condition_double_root_equation_m_value_l2380_238056

/-- Definition of a double root equation -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ (a * x^2 + b * x + c = 0) ∧ (a * y^2 + b * y + c = 0) ∧ (y = 2*x ∨ x = 2*y)

/-- Theorem 1: x^2 - 3x + 2 = 0 is a double root equation -/
theorem double_root_equation_example : is_double_root_equation 1 (-3) 2 := sorry

/-- Theorem 2: For (x-2)(x-m) = 0 to be a double root equation, m^2 + 2m + 2 = 26 or 5 -/
theorem double_root_equation_condition (m : ℝ) :
  is_double_root_equation 1 (-(2+m)) (2*m) →
  m^2 + 2*m + 2 = 26 ∨ m^2 + 2*m + 2 = 5 := sorry

/-- Theorem 3: For x^2 - (m-1)x + 32 = 0 to be a double root equation, m = 13 or -11 -/
theorem double_root_equation_m_value (m : ℝ) :
  is_double_root_equation 1 (-(m-1)) 32 →
  m = 13 ∨ m = -11 := sorry

end NUMINAMATH_CALUDE_double_root_equation_example_double_root_equation_condition_double_root_equation_m_value_l2380_238056


namespace NUMINAMATH_CALUDE_ratio_fraction_l2380_238025

theorem ratio_fraction (x y : ℝ) (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_l2380_238025


namespace NUMINAMATH_CALUDE_extreme_value_interval_equation_solution_range_l2380_238014

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem extreme_value_interval (a : ℝ) (h : a > 0) :
  (∃ x ∈ Set.Ioo a (a + 1/2), ∀ y ∈ Set.Ioo a (a + 1/2), f x ≥ f y) →
  1/2 < a ∧ a < 1 :=
sorry

theorem equation_solution_range (k : ℝ) :
  (∃ x ≥ 1, f x = k / (x + 1)) →
  k ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_interval_equation_solution_range_l2380_238014


namespace NUMINAMATH_CALUDE_expression_value_l2380_238081

theorem expression_value (z : ℝ) : (1 : ℝ)^(6*z-3) / (7⁻¹ + 4⁻¹) = 28/11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2380_238081


namespace NUMINAMATH_CALUDE_student_guinea_pig_difference_is_126_l2380_238007

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 24

/-- The number of guinea pigs in each classroom -/
def guinea_pigs_per_classroom : ℕ := 3

/-- The number of classrooms -/
def number_of_classrooms : ℕ := 6

/-- The difference between the total number of students and the total number of guinea pigs -/
def student_guinea_pig_difference : ℕ := 
  (students_per_classroom * number_of_classrooms) - (guinea_pigs_per_classroom * number_of_classrooms)

theorem student_guinea_pig_difference_is_126 : student_guinea_pig_difference = 126 := by
  sorry

end NUMINAMATH_CALUDE_student_guinea_pig_difference_is_126_l2380_238007


namespace NUMINAMATH_CALUDE_f_value_at_2012_l2380_238006

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h1 : ∀ x : ℝ, f (x + 3) ≤ f x + 3)
variable (h2 : ∀ x : ℝ, f (x + 2) ≥ f x + 2)
variable (h3 : f 998 = 1002)

-- State the theorem
theorem f_value_at_2012 : f 2012 = 2016 := by sorry

end NUMINAMATH_CALUDE_f_value_at_2012_l2380_238006


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l2380_238030

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l ^ 2 + w ^ 2 = d ^ 2) :
  l * w = (10 / 29) * d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l2380_238030


namespace NUMINAMATH_CALUDE_function_range_l2380_238099

theorem function_range (a : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - a*x + a + 3 < 0 ∧ x - a < 0)) → 
  a ∈ Set.Icc (-3) 6 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l2380_238099


namespace NUMINAMATH_CALUDE_basketball_points_difference_basketball_game_theorem_l2380_238043

/-- The difference between the combined points of Tobee and Jay and Sean's points is 2 -/
theorem basketball_points_difference : ℕ → ℕ → ℕ → Prop :=
  fun tobee_points jay_points_diff total_team_points =>
    let jay_points := tobee_points + jay_points_diff
    let combined_points := tobee_points + jay_points
    let sean_points := total_team_points - combined_points
    combined_points - sean_points = 2

/-- Given the conditions of the basketball game -/
theorem basketball_game_theorem :
  basketball_points_difference 4 6 26 := by
  sorry

end NUMINAMATH_CALUDE_basketball_points_difference_basketball_game_theorem_l2380_238043


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2380_238045

theorem cubic_roots_sum (a b c : ℝ) : 
  (10 * a^3 + 15 * a^2 + 2005 * a + 2010 = 0) →
  (10 * b^3 + 15 * b^2 + 2005 * b + 2010 = 0) →
  (10 * c^3 + 15 * c^2 + 2005 * c + 2010 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 907.125 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2380_238045


namespace NUMINAMATH_CALUDE_arctan_sum_three_four_l2380_238026

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_four_l2380_238026


namespace NUMINAMATH_CALUDE_remainder_problem_l2380_238085

theorem remainder_problem (N : ℤ) (h : N % 1423 = 215) :
  (N - (N / 109)^2) % 109 = 106 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2380_238085


namespace NUMINAMATH_CALUDE_lawn_width_l2380_238058

theorem lawn_width (area : ℝ) (length : ℝ) (width : ℝ) 
  (h1 : area = 20)
  (h2 : length = 4)
  (h3 : area = length * width) : 
  width = 5 := by
sorry

end NUMINAMATH_CALUDE_lawn_width_l2380_238058


namespace NUMINAMATH_CALUDE_roots_of_x_squared_equals_x_l2380_238074

theorem roots_of_x_squared_equals_x :
  {x : ℝ | x^2 = x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_roots_of_x_squared_equals_x_l2380_238074


namespace NUMINAMATH_CALUDE_angies_age_equation_l2380_238062

theorem angies_age_equation (angie_age : ℕ) (result : ℕ) : 
  angie_age = 8 → result = 2 * angie_age + 4 → result = 20 := by
  sorry

end NUMINAMATH_CALUDE_angies_age_equation_l2380_238062


namespace NUMINAMATH_CALUDE_male_kittens_count_l2380_238068

/-- Given an initial number of cats, number of female kittens, and total number of cats after kittens are born, 
    calculate the number of male kittens. -/
def male_kittens (initial_cats female_kittens total_cats : ℕ) : ℕ :=
  total_cats - initial_cats - female_kittens

/-- Theorem stating that given the problem conditions, the number of male kittens is 2. -/
theorem male_kittens_count : male_kittens 2 3 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_male_kittens_count_l2380_238068


namespace NUMINAMATH_CALUDE_katya_magic_pen_problem_l2380_238048

theorem katya_magic_pen_problem (p_katya : ℚ) (p_pen : ℚ) (total_problems : ℕ) (min_correct : ℚ) :
  p_katya = 4/5 →
  p_pen = 1/2 →
  total_problems = 20 →
  min_correct = 13 →
  ∃ x : ℕ, x ≥ 10 ∧
    x * p_katya + (total_problems - x) * p_pen ≥ min_correct ∧
    ∀ y : ℕ, y < 10 → y * p_katya + (total_problems - y) * p_pen < min_correct :=
by sorry

end NUMINAMATH_CALUDE_katya_magic_pen_problem_l2380_238048


namespace NUMINAMATH_CALUDE_investment_duration_l2380_238093

/-- Given an investment with simple interest, prove the duration is 2.5 years -/
theorem investment_duration (principal interest_rate interest : ℝ) 
  (h1 : principal = 7200)
  (h2 : interest_rate = 17.5)
  (h3 : interest = 3150) :
  interest = principal * interest_rate * 2.5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_investment_duration_l2380_238093


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2380_238029

theorem sqrt_product_equality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2380_238029


namespace NUMINAMATH_CALUDE_blue_then_red_probability_l2380_238055

/-- The probability of drawing a blue marble first and a red marble second -/
theorem blue_then_red_probability (red white blue : ℕ) 
  (h_red : red = 4)
  (h_white : white = 6)
  (h_blue : blue = 2) : 
  (blue : ℚ) / (red + white + blue) * red / (red + white + blue - 1) = 2 / 33 := by
sorry

end NUMINAMATH_CALUDE_blue_then_red_probability_l2380_238055


namespace NUMINAMATH_CALUDE_product_2022_sum_possibilities_l2380_238070

theorem product_2022_sum_possibilities (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧ 
  a * b * c * d * e = 2022 → 
  a + b + c + d + e = 342 ∨ 
  a + b + c + d + e = 338 ∨ 
  a + b + c + d + e = 336 ∨ 
  a + b + c + d + e = -332 :=
by sorry

end NUMINAMATH_CALUDE_product_2022_sum_possibilities_l2380_238070


namespace NUMINAMATH_CALUDE_money_ratio_problem_l2380_238076

/-- Given the ratios of money between Ram and Gopal (7:17) and between Gopal and Krishan (7:17),
    and that Ram has Rs. 588, prove that Krishan has Rs. 12,065. -/
theorem money_ratio_problem (ram gopal krishan : ℚ) : 
  ram / gopal = 7 / 17 →
  gopal / krishan = 7 / 17 →
  ram = 588 →
  krishan = 12065 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_problem_l2380_238076


namespace NUMINAMATH_CALUDE_equal_square_difference_subsequence_equal_square_difference_and_arithmetic_is_constant_l2380_238018

-- Define the property of being an "equal square difference sequence"
def is_equal_square_difference (a : ℕ → ℝ) : Prop :=
  ∃ p : ℝ, ∀ n : ℕ, n ≥ 2 → a n ^ 2 - a (n - 1) ^ 2 = p

-- Define the property of being an arithmetic sequence
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Theorem 1
theorem equal_square_difference_subsequence
  (a : ℕ → ℝ) (k : ℕ) (hk : k > 0) (ha : is_equal_square_difference a) :
  is_equal_square_difference (fun n ↦ a (k * n)) :=
sorry

-- Theorem 2
theorem equal_square_difference_and_arithmetic_is_constant
  (a : ℕ → ℝ) (ha1 : is_equal_square_difference a) (ha2 : is_arithmetic a) :
  ∃ c : ℝ, ∀ n : ℕ, a n = c :=
sorry

end NUMINAMATH_CALUDE_equal_square_difference_subsequence_equal_square_difference_and_arithmetic_is_constant_l2380_238018


namespace NUMINAMATH_CALUDE_proposition_relationship_l2380_238054

theorem proposition_relationship (a b : ℝ) : 
  (∀ a b : ℝ, (a > b ∧ a⁻¹ > b⁻¹) → a > 0) ∧ 
  (∃ a b : ℝ, a > 0 ∧ ¬(a > b ∧ a⁻¹ > b⁻¹)) := by
sorry

end NUMINAMATH_CALUDE_proposition_relationship_l2380_238054


namespace NUMINAMATH_CALUDE_exactly_one_incorrect_l2380_238084

-- Define the statements
def statement1 : Prop := ∀ (P : ℝ → Prop), (∀ x, P x) ↔ ¬(∃ x, ¬(P x))

def statement2 : Prop := ∀ (p q : Prop), ¬(p ∨ q) → (¬p ∧ ¬q)

def statement3 : Prop := ∀ (m n : ℝ), 
  (m * n > 0 → (∀ x y : ℝ, m * x^2 + n * y^2 = 1 ↔ (m > 0 ∧ n > 0 ∧ m ≠ n))) ∧
  (¬(∀ x y : ℝ, m * x^2 + n * y^2 = 1 ↔ (m > 0 ∧ n > 0 ∧ m ≠ n)) → m * n ≤ 0)

-- Theorem to prove
theorem exactly_one_incorrect : 
  (statement1 ∧ statement2 ∧ ¬statement3) ∨
  (statement1 ∧ ¬statement2 ∧ statement3) ∨
  (¬statement1 ∧ statement2 ∧ statement3) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_incorrect_l2380_238084


namespace NUMINAMATH_CALUDE_graduation_ceremony_attendance_l2380_238034

/-- Graduation ceremony attendance problem -/
theorem graduation_ceremony_attendance
  (graduates : ℕ)
  (chairs : ℕ)
  (parents_per_graduate : ℕ)
  (h_graduates : graduates = 50)
  (h_chairs : chairs = 180)
  (h_parents : parents_per_graduate = 2)
  (h_admins : administrators = teachers / 2) :
  teachers = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_graduation_ceremony_attendance_l2380_238034


namespace NUMINAMATH_CALUDE_total_toys_count_l2380_238011

-- Define the number of toys for each child
def jaxon_toys : ℝ := 15
def gabriel_toys : ℝ := 2.5 * jaxon_toys
def jerry_toys : ℝ := gabriel_toys + 8.5
def sarah_toys : ℝ := jerry_toys - 5.5
def emily_toys : ℝ := 1.5 * gabriel_toys

-- Define the total number of toys
def total_toys : ℝ := jerry_toys + gabriel_toys + jaxon_toys + sarah_toys + emily_toys

-- Theorem to prove
theorem total_toys_count : total_toys = 195.25 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_count_l2380_238011


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2380_238012

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ z₁ z₂ : ℂ, z₁^2 + 5*z₁ + m = 0 ∧ z₂^2 + 5*z₂ + m = 0 ∧ Complex.abs (z₁ - z₂) = 3) → 
  (m = 4 ∨ m = 17/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2380_238012


namespace NUMINAMATH_CALUDE_geometric_parallelism_l2380_238037

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the "contained in" relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_plane_parallel : Line → Plane → Prop)

-- State the theorem
theorem geometric_parallelism 
  (a : Line) (α β : Plane) (h : contained_in a α) :
  (plane_parallel α β → line_plane_parallel a β) ∧
  (¬ line_plane_parallel a β → ¬ plane_parallel α β) ∧
  ¬ (line_plane_parallel a β → plane_parallel α β) :=
sorry

end NUMINAMATH_CALUDE_geometric_parallelism_l2380_238037


namespace NUMINAMATH_CALUDE_oxen_equivalence_l2380_238098

/-- The amount of fodder a buffalo eats per day -/
def buffalo_fodder : ℝ := sorry

/-- The amount of fodder a cow eats per day -/
def cow_fodder : ℝ := sorry

/-- The amount of fodder an ox eats per day -/
def ox_fodder : ℝ := sorry

/-- The total amount of fodder available -/
def total_fodder : ℝ := sorry

theorem oxen_equivalence :
  (3 * buffalo_fodder = 4 * cow_fodder) →
  (15 * buffalo_fodder + 8 * ox_fodder + 24 * cow_fodder) * 36 = total_fodder →
  (30 * buffalo_fodder + 8 * ox_fodder + 64 * cow_fodder) * 18 = total_fodder →
  (∃ n : ℕ, n * ox_fodder = 3 * buffalo_fodder ∧ n * ox_fodder = 4 * cow_fodder ∧ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_oxen_equivalence_l2380_238098


namespace NUMINAMATH_CALUDE_sin_transformations_l2380_238013

open Real

theorem sin_transformations (x : ℝ) :
  (∀ x, sin (2 * (x - π/6)) = sin (2*x - π/3)) ∧
  (∀ x, sin (2 * (x - π/3)) = sin (2*x - π/3)) ∧
  (∀ x, sin (2 * (x + 5*π/6)) = sin (2*x - π/3)) :=
by sorry

end NUMINAMATH_CALUDE_sin_transformations_l2380_238013


namespace NUMINAMATH_CALUDE_right_angled_quadrilateral_distances_and_angles_l2380_238008

theorem right_angled_quadrilateral_distances_and_angles 
  (a a' α : ℝ) (h_pos_a : a > 0) (h_pos_a' : a' > 0) (h_pos_α : α > 0) (h_α_lt_pi_2 : α < π / 2) :
  let d := (2 * (a * Real.tan α - a') * Real.tan α) / (Real.tan α ^ 2 - 1)
  let d' := (2 * (a * Real.tan α - a')) / (Real.tan α ^ 2 - 1)
  let cf := (2 * (a' * Real.tan α ^ 2 - a * Real.tan α)) / (Real.tan α ^ 2 - 1)
  let ce := (2 * (a' * Real.tan α - a)) / (Real.tan α ^ 2 - 1)
  let angle_cba := Real.arctan ((a' * Real.tan α - a) / (a * Real.tan α - a'))
  (d = (2 * (a * Real.tan α - a') * Real.tan α) / (Real.tan α ^ 2 - 1)) ∧
  (d' = (2 * (a * Real.tan α - a')) / (Real.tan α ^ 2 - 1)) ∧
  (cf = (2 * (a' * Real.tan α ^ 2 - a * Real.tan α)) / (Real.tan α ^ 2 - 1)) ∧
  (ce = (2 * (a' * Real.tan α - a)) / (Real.tan α ^ 2 - 1)) ∧
  (angle_cba = Real.arctan ((a' * Real.tan α - a) / (a * Real.tan α - a'))) ∧
  (angle_cba = Real.arctan ((a' * Real.tan α - a) / (a * Real.tan α - a'))) :=
by
  sorry

end NUMINAMATH_CALUDE_right_angled_quadrilateral_distances_and_angles_l2380_238008


namespace NUMINAMATH_CALUDE_books_read_difference_l2380_238036

def total_books : ℕ := 20
def peter_percentage : ℚ := 40 / 100
def brother_percentage : ℚ := 10 / 100

theorem books_read_difference : 
  (peter_percentage * total_books : ℚ).floor - (brother_percentage * total_books : ℚ).floor = 6 := by
  sorry

end NUMINAMATH_CALUDE_books_read_difference_l2380_238036


namespace NUMINAMATH_CALUDE_andrews_numbers_l2380_238092

theorem andrews_numbers (x y : ℤ) : 
  3 * x + 4 * y = 161 → (x = 17 ∨ y = 17) → (x = 31 ∨ y = 31) := by
  sorry

end NUMINAMATH_CALUDE_andrews_numbers_l2380_238092


namespace NUMINAMATH_CALUDE_final_sugar_amount_l2380_238050

def sugar_calculation (initial : ℕ) (used : ℕ) (bought : ℕ) : ℕ :=
  initial - used + bought

theorem final_sugar_amount :
  sugar_calculation 65 18 50 = 97 := by
  sorry

end NUMINAMATH_CALUDE_final_sugar_amount_l2380_238050


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2011_l2380_238019

/-- The last four digits of 5^n -/
def lastFourDigits (n : ℕ) : ℕ := 5^n % 10000

/-- Given conditions -/
axiom base_case_5 : lastFourDigits 5 = 3125
axiom base_case_6 : lastFourDigits 6 = 5625
axiom base_case_7 : lastFourDigits 7 = 8125

/-- Theorem statement -/
theorem last_four_digits_of_5_pow_2011 : lastFourDigits 2011 = 8125 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2011_l2380_238019


namespace NUMINAMATH_CALUDE_parallel_lines_ratio_l2380_238080

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  L1 : ℝ → ℝ → Prop
  L2 : ℝ → ℝ → Prop
  a : ℝ
  c : ℝ
  c_pos : c > 0
  is_parallel : ∀ x y, L1 x y ↔ x - y + 1 = 0
  L2_eq : ∀ x y, L2 x y ↔ 3*x + a*y - c = 0
  distance : ℝ

/-- The theorem stating the value of (a-3)/c for the given parallel lines -/
theorem parallel_lines_ratio (lines : ParallelLines) 
  (h_dist : lines.distance = Real.sqrt 2) : 
  (lines.a - 3) / lines.c = -2 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_ratio_l2380_238080


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2380_238024

/-- Represents a cylinder with two spheres inside it -/
structure CylinderWithSpheres where
  cylinderRadius : ℝ
  sphereRadius : ℝ
  sphereCenterDistance : ℝ

/-- Represents the ellipse formed by the intersection of a plane with the cylinder -/
structure IntersectionEllipse where
  majorAxis : ℝ

/-- The length of the major axis of the ellipse formed by a plane tangent to both spheres 
    and intersecting the cylindrical surface is equal to the distance between sphere centers -/
theorem ellipse_major_axis_length 
  (c : CylinderWithSpheres) 
  (h1 : c.cylinderRadius = 6) 
  (h2 : c.sphereRadius = 6) 
  (h3 : c.sphereCenterDistance = 13) : 
  ∃ e : IntersectionEllipse, e.majorAxis = c.sphereCenterDistance :=
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2380_238024


namespace NUMINAMATH_CALUDE_difference_of_squares_70_30_l2380_238097

theorem difference_of_squares_70_30 : 70^2 - 30^2 = 4000 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_70_30_l2380_238097


namespace NUMINAMATH_CALUDE_solve_equation_l2380_238091

theorem solve_equation (x : ℚ) (h : (1/3 - 1/4) * 2 = 1/x) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2380_238091


namespace NUMINAMATH_CALUDE_jack_closet_capacity_l2380_238065

/-- Represents the storage capacity of a closet -/
structure ClosetCapacity where
  cansPerRow : ℕ
  rowsPerShelf : ℕ
  shelvesPerCloset : ℕ

/-- Calculates the total number of cans that can be stored in a closet -/
def totalCansPerCloset (c : ClosetCapacity) : ℕ :=
  c.cansPerRow * c.rowsPerShelf * c.shelvesPerCloset

/-- Theorem: Given Jack's closet configuration, he can store 480 cans in each closet -/
theorem jack_closet_capacity :
  let jackCloset : ClosetCapacity := {
    cansPerRow := 12,
    rowsPerShelf := 4,
    shelvesPerCloset := 10
  }
  totalCansPerCloset jackCloset = 480 := by
  sorry


end NUMINAMATH_CALUDE_jack_closet_capacity_l2380_238065


namespace NUMINAMATH_CALUDE_equation_solution_l2380_238033

theorem equation_solution (m n : ℝ) (h : m ≠ n) :
  let f := fun x : ℝ => x^2 + (x + m)^2 - (x + n)^2 - 2*m*n
  (∀ x, f x = 0 ↔ x = -m + n + Real.sqrt (2*(n^2 - m*n + m^2)) ∨
                   x = -m + n - Real.sqrt (2*(n^2 - m*n + m^2))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2380_238033


namespace NUMINAMATH_CALUDE_intersection_points_l2380_238063

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 10)^2 = 50

def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*(x - y) - 18 = 0

theorem intersection_points :
  (∃ (x y : ℝ), circle1 x y ∧ circle2 x y) ∧
  (circle1 3 3 ∧ circle2 3 3) ∧
  (circle1 (-3) 5 ∧ circle2 (-3) 5) ∧
  (∀ (x y : ℝ), circle1 x y ∧ circle2 x y → (x = 3 ∧ y = 3) ∨ (x = -3 ∧ y = 5)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_l2380_238063


namespace NUMINAMATH_CALUDE_solution_of_system_l2380_238041

/-- Given a system of equations, prove that the solutions are (2, 1) and (2/5, -1/5) -/
theorem solution_of_system :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ = 2 ∧ y₁ = 1) ∧
    (x₂ = 2/5 ∧ y₂ = -1/5) ∧
    (∀ x y : ℝ,
      (5 * x * (1 + 1 / (x^2 + y^2)) = 12 ∧
       5 * y * (1 - 1 / (x^2 + y^2)) = 4) ↔
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l2380_238041


namespace NUMINAMATH_CALUDE_power_inequality_condition_l2380_238020

theorem power_inequality_condition (n : ℤ) : n ∈ ({-2, -1, 0, 1, 2, 3} : Set ℤ) →
  ((-1/2 : ℚ)^n > (-1/5 : ℚ)^n ↔ n = -1 ∨ n = 2) := by sorry

end NUMINAMATH_CALUDE_power_inequality_condition_l2380_238020


namespace NUMINAMATH_CALUDE_value_of_s_l2380_238061

-- Define the variables as natural numbers
variable (a b c p q s : ℕ)

-- Define the conditions
axiom distinct_nonzero : a ≠ b ∧ a ≠ c ∧ a ≠ p ∧ a ≠ q ∧ a ≠ s ∧
                         b ≠ c ∧ b ≠ p ∧ b ≠ q ∧ b ≠ s ∧
                         c ≠ p ∧ c ≠ q ∧ c ≠ s ∧
                         p ≠ q ∧ p ≠ s ∧
                         q ≠ s ∧
                         a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ s ≠ 0

axiom eq1 : a + b = p
axiom eq2 : p + c = s
axiom eq3 : s + a = q
axiom eq4 : b + c + q = 18

-- Theorem to prove
theorem value_of_s : s = 9 :=
sorry

end NUMINAMATH_CALUDE_value_of_s_l2380_238061


namespace NUMINAMATH_CALUDE_exists_N_average_twelve_l2380_238046

theorem exists_N_average_twelve : ∃ N : ℝ, 11 < N ∧ N < 19 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_exists_N_average_twelve_l2380_238046


namespace NUMINAMATH_CALUDE_average_of_twenty_digits_l2380_238042

theorem average_of_twenty_digits :
  let total_count : ℕ := 20
  let group1_count : ℕ := 14
  let group2_count : ℕ := 6
  let group1_average : ℝ := 390
  let group2_average : ℝ := 756.67
  let total_average : ℝ := (group1_count * group1_average + group2_count * group2_average) / total_count
  total_average = 500.001 := by sorry

end NUMINAMATH_CALUDE_average_of_twenty_digits_l2380_238042


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_zero_is_exists_lt_zero_l2380_238047

theorem negation_of_forall_geq_zero_is_exists_lt_zero :
  (¬ ∀ x : ℝ, x^2 + x ≥ 0) ↔ (∃ x : ℝ, x^2 + x < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_zero_is_exists_lt_zero_l2380_238047


namespace NUMINAMATH_CALUDE_fraction_simplification_l2380_238064

theorem fraction_simplification :
  (1/2 - 1/3 + 1/5) / (1/3 - 1/2 + 1/7) = -77/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2380_238064


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l2380_238016

theorem largest_divisor_of_expression : 
  ∃ (x : ℕ), x = 18 ∧ 
  (∀ (y : ℕ), x ∣ (7^y + 12*y - 1)) ∧
  (∀ (z : ℕ), z > x → ∃ (w : ℕ), ¬(z ∣ (7^w + 12*w - 1))) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l2380_238016


namespace NUMINAMATH_CALUDE_circumradius_of_special_triangle_l2380_238022

/-- Given a triangle ABC with side lengths proportional to 7:5:3 and area 45√3,
    prove that the radius of its circumscribed circle is 14. -/
theorem circumradius_of_special_triangle (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a / b = 7 / 5 →
  b / c = 5 / 3 →
  (1 / 2) * a * b * Real.sin C = 45 * Real.sqrt 3 →
  R = (a / (2 * Real.sin A)) →
  R = 14 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_of_special_triangle_l2380_238022


namespace NUMINAMATH_CALUDE_evening_campers_l2380_238032

theorem evening_campers (afternoon_campers : ℕ) (difference : ℕ) : 
  afternoon_campers = 34 → difference = 24 → afternoon_campers - difference = 10 := by
  sorry

end NUMINAMATH_CALUDE_evening_campers_l2380_238032


namespace NUMINAMATH_CALUDE_range_of_a_l2380_238027

-- Define the sets A and C
def A : Set ℝ := {x | x^2 - 6*x + 5 < 0}
def C (a : ℝ) : Set ℝ := {x | 3*a - 2 < x ∧ x < 4*a - 3}

-- State the theorem
theorem range_of_a (a : ℝ) : C a ⊆ A ↔ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2380_238027


namespace NUMINAMATH_CALUDE_frustum_cone_height_l2380_238000

theorem frustum_cone_height (h : ℝ) (a_lower a_upper : ℝ) 
  (h_positive : h > 0)
  (a_lower_positive : a_lower > 0)
  (a_upper_positive : a_upper > 0)
  (h_value : h = 30)
  (a_lower_value : a_lower = 400 * Real.pi)
  (a_upper_value : a_upper = 100 * Real.pi) :
  let r_lower := (a_lower / Real.pi).sqrt
  let r_upper := (a_upper / Real.pi).sqrt
  let h_total := h * r_lower / (r_lower - r_upper)
  h_total / 3 = 15 := by sorry

end NUMINAMATH_CALUDE_frustum_cone_height_l2380_238000


namespace NUMINAMATH_CALUDE_guards_in_team_l2380_238089

theorem guards_in_team (s b n : ℕ) : 
  s > 0 ∧ b > 0 ∧ n > 0 →  -- positive integers
  s * b * n = 1001 →  -- total person-nights
  s < n →  -- guards in team less than nights slept
  n < b →  -- nights slept less than number of teams
  s = 7 :=  -- prove number of guards in a team is 7
by sorry

end NUMINAMATH_CALUDE_guards_in_team_l2380_238089


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l2380_238077

theorem minimum_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l2380_238077


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l2380_238088

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l2380_238088


namespace NUMINAMATH_CALUDE_y_divisibility_l2380_238086

def y : ℕ := 54 + 108 + 162 + 216 + 648 + 810 + 972

theorem y_divisibility :
  (∃ k : ℕ, y = 2 * k) ∧
  (∃ k : ℕ, y = 3 * k) ∧
  (∃ k : ℕ, y = 6 * k) ∧
  (∃ k : ℕ, y = 9 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l2380_238086


namespace NUMINAMATH_CALUDE_final_distance_after_two_hours_l2380_238057

/-- The distance between Jay and Paul after walking for a given time -/
def distance_after_time (initial_distance : ℝ) (jay_speed : ℝ) (paul_speed : ℝ) (time : ℝ) : ℝ :=
  initial_distance + jay_speed * time + paul_speed * time

/-- Theorem stating the final distance between Jay and Paul after 2 hours -/
theorem final_distance_after_two_hours :
  let initial_distance : ℝ := 3
  let jay_speed : ℝ := 1 / (20 / 60) -- miles per hour
  let paul_speed : ℝ := 3 / (40 / 60) -- miles per hour
  let time : ℝ := 2 -- hours
  distance_after_time initial_distance jay_speed paul_speed time = 18 := by
  sorry


end NUMINAMATH_CALUDE_final_distance_after_two_hours_l2380_238057


namespace NUMINAMATH_CALUDE_katya_age_l2380_238059

/-- Represents the ages of the children in the family -/
structure FamilyAges where
  anya : ℕ
  katya : ℕ
  vasya : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.anya + ages.katya = 19 ∧
  ages.anya + ages.vasya = 14 ∧
  ages.katya + ages.vasya = 7

/-- The theorem to prove Katya's age -/
theorem katya_age (ages : FamilyAges) (h : satisfiesConditions ages) : ages.katya = 6 := by
  sorry


end NUMINAMATH_CALUDE_katya_age_l2380_238059


namespace NUMINAMATH_CALUDE_harrys_family_age_ratio_l2380_238096

/-- Given Harry's age, the age difference between Harry and his father, and his mother's age when she gave birth to him, 
    prove that the ratio of the age difference between Harry's parents to Harry's age is 1:25. -/
theorem harrys_family_age_ratio (harry_age : ℕ) (father_age_diff : ℕ) (mother_age_at_birth : ℕ)
  (h1 : harry_age = 50)
  (h2 : father_age_diff = 24)
  (h3 : mother_age_at_birth = 22) :
  (father_age_diff + harry_age - (mother_age_at_birth + harry_age)) / harry_age = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_harrys_family_age_ratio_l2380_238096


namespace NUMINAMATH_CALUDE_calculate_second_discount_l2380_238071

/-- Given an article with a list price and two successive discounts, 
    calculate the second discount percentage. -/
theorem calculate_second_discount 
  (list_price : ℝ) 
  (first_discount : ℝ) 
  (final_price : ℝ) 
  (h1 : list_price = 70) 
  (h2 : first_discount = 10) 
  (h3 : final_price = 61.74) : 
  ∃ (second_discount : ℝ), 
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧ 
    second_discount = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_second_discount_l2380_238071


namespace NUMINAMATH_CALUDE_f_difference_l2380_238023

/-- The function f defined as f(x) = 5x^2 - 2x - 1 -/
def f (x : ℝ) : ℝ := 5 * x^2 - 2 * x - 1

/-- Theorem stating that f(x + h) - f(x) = h(10x + 5h - 2) for all real x and h -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (10 * x + 5 * h - 2) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l2380_238023


namespace NUMINAMATH_CALUDE_daffodil_bouquet_cost_l2380_238066

/-- The cost of a bouquet of daffodils given the number of flowers -/
def bouquet_cost (n : ℕ) : ℚ :=
  sorry

theorem daffodil_bouquet_cost :
  bouquet_cost 15 = 25 →
  (∀ (m n : ℕ), m * bouquet_cost n = n * bouquet_cost m) →
  bouquet_cost 45 = 75 := by
  sorry

end NUMINAMATH_CALUDE_daffodil_bouquet_cost_l2380_238066


namespace NUMINAMATH_CALUDE_geometric_progression_fifth_term_l2380_238040

theorem geometric_progression_fifth_term 
  (a₁ a₂ a₃ : ℝ)
  (h₁ : a₁ = 2^(1/4 : ℝ))
  (h₂ : a₂ = 2^(1/5 : ℝ))
  (h₃ : a₃ = 2^(1/6 : ℝ))
  (h₄ : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) :
  ∃ a₅ : ℝ, a₅ = 2^(11/60 : ℝ) ∧ 
    ∃ a₄ : ℝ, a₄ = a₃ * (a₂ / a₁) ∧ a₅ = a₄ * (a₂ / a₁) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_fifth_term_l2380_238040


namespace NUMINAMATH_CALUDE_rabbits_count_l2380_238001

/-- Represents the number of rabbits and peacocks in a zoo. -/
structure ZooAnimals where
  rabbits : ℕ
  peacocks : ℕ

/-- The total number of heads in the zoo is 60. -/
def total_heads (zoo : ZooAnimals) : Prop :=
  zoo.rabbits + zoo.peacocks = 60

/-- The total number of legs in the zoo is 192. -/
def total_legs (zoo : ZooAnimals) : Prop :=
  4 * zoo.rabbits + 2 * zoo.peacocks = 192

/-- Theorem stating that given the conditions, the number of rabbits is 36. -/
theorem rabbits_count (zoo : ZooAnimals) 
  (h1 : total_heads zoo) (h2 : total_legs zoo) : zoo.rabbits = 36 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_count_l2380_238001


namespace NUMINAMATH_CALUDE_ic_train_speed_ratio_l2380_238083

theorem ic_train_speed_ratio :
  ∀ (u v : ℝ), u > 0 → v > 0 →
  (u / v = ((u + v) / (u - v))) →
  (u / v = 1 + Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_ic_train_speed_ratio_l2380_238083
