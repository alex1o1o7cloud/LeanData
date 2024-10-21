import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l643_64337

open Set

-- Define the sets P and Q
def P (m : ℝ) : Set ℝ := {x | (x - m) * (x - m - 3) > 0}
def Q : Set ℝ := {x | x^2 + 3*x - 4 < 0}

-- State the theorem
theorem range_of_m :
  (∀ m : ℝ, Q ⊂ P m ∧ Q ≠ P m) ↔ m ∈ Iic (-7) ∪ Ici 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l643_64337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_power_function_implies_m_equals_two_l643_64352

/-- A power function with a coefficient depending on m -/
noncomputable def powerFunction (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^m

/-- Definition of an even function -/
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- Theorem stating that if the power function is even, then m = 2 -/
theorem even_power_function_implies_m_equals_two :
  ∃ m : ℝ, isEven (powerFunction m) → m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_power_function_implies_m_equals_two_l643_64352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tommy_flour_cost_l643_64322

/-- The minimum cost to buy flour for Tommy's bread -/
theorem tommy_flour_cost (loaves : ℕ) (flour_per_loaf : ℕ) 
  (bag10_cost : ℕ) (bag12_cost : ℕ) : 
  loaves = 12 → 
  flour_per_loaf = 4 → 
  bag10_cost = 10 → 
  bag12_cost = 13 → 
  min (Nat.ceil ((loaves * flour_per_loaf : ℚ) / 10) * bag10_cost)
      (Nat.ceil ((loaves * flour_per_loaf : ℚ) / 12) * bag12_cost) = 50 := by
  sorry

#check tommy_flour_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tommy_flour_cost_l643_64322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l643_64311

/-- Given a real polynomial p(x) of the form 1 + α₁x^(m₁) + α₂x^(m₂) + ... + αₙx^(mₙ)
    where 0 < m₁ < m₂ < ... < mₙ, and p(x) = (1 - x)^n * q(x) for some real polynomial q(x),
    prove that q(1) = (m₁ * m₂ * ... * mₙ) / n! -/
theorem polynomial_factorization (n : ℕ) (m : Fin n → ℝ) (α : Fin n → ℝ) 
    (h_m_increasing : ∀ i j, i < j → m i < m j) 
    (h_m_positive : ∀ i, 0 < m i) 
    (p q : ℝ → ℝ) 
    (h_p_def : p = λ x ↦ 1 + (Finset.sum Finset.univ (λ i ↦ α i * x^(m i))))
    (h_pq : p = λ x ↦ (1 - x)^n * q x) :
  q 1 = (Finset.prod Finset.univ m) / n! := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l643_64311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_triangle_l643_64303

open Real

-- Define the IsTriangle predicate
def IsTriangle (A B C : ℝ) : Prop := A + B + C = π

-- Define the OppositeAngle predicate
def OppositeAngle (angle : ℝ) (side1 side2 : ℝ) : Prop :=
  sin angle * side1 = sin angle * side2

theorem angle_measure_in_triangle (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  IsTriangle A B C →
  -- Side lengths correspond to opposite angles
  OppositeAngle A b c →
  OppositeAngle B a c →
  OppositeAngle C a b →
  -- Given conditions
  a = sqrt 2 →
  b = 2 →
  sin B - cos B = sqrt 2 →
  -- Conclusion
  A = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_triangle_l643_64303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_points_theorem_l643_64387

-- Define the circle Γ
variable (Γ : Set (ℝ × ℝ))

-- Define points on the plane
variable (A B C D E F G : ℝ × ℝ)

-- Define the property of being a circle
def is_circle (s : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property of being a chord of a circle
def is_chord (c : Set (ℝ × ℝ)) (s : Set (ℝ × ℝ)) : Prop := sorry

-- Define the midpoint of an arc
def is_midpoint_of_arc (p : ℝ × ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) (s : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property of a point being on a line segment
def on_segment (p : ℝ × ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop := sorry

-- Define the property of points being concyclic
def are_concyclic (a b c d : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem concyclic_points_theorem 
  (h1 : is_circle Γ)
  (h2 : is_chord {B, C} Γ)
  (h3 : is_midpoint_of_arc A B C Γ)
  (h4 : is_chord {A, D} Γ)
  (h5 : is_chord {A, E} Γ)
  (h6 : on_segment F B C)
  (h7 : on_segment G B C)
  (h8 : on_segment F A D)
  (h9 : on_segment G A E)
  : are_concyclic D E F G := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_points_theorem_l643_64387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_triple_angle_condition_l643_64386

theorem sin_triple_angle_condition (x : ℝ) : 
  Real.sin (3 * x) = 3 * Real.sin x ↔ Real.sin x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_triple_angle_condition_l643_64386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_integers_product_sum_l643_64338

theorem circle_integers_product_sum (a b c d : ℤ) (h : a + b + c + d = 0) :
  ∃ (n : ℤ), -(a * (b + d) + b * (a + c) + c * (b + d) + d * (a + c)) = 2 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_integers_product_sum_l643_64338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_time_ratio_l643_64320

theorem project_time_ratio :
  ∀ (kate_hours pat_hours mark_hours : ℕ),
    kate_hours + pat_hours + mark_hours = 144 →
    pat_hours = 2 * kate_hours →
    mark_hours = kate_hours + 80 →
    pat_hours * 3 = mark_hours :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_time_ratio_l643_64320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brokerage_percentage_calculation_l643_64306

/-- Represents the stock details and calculates the brokerage percentage -/
noncomputable def calculate_brokerage_percentage (stock_rate : ℝ) (income : ℝ) (investment : ℝ) (market_value : ℝ) : ℝ :=
  let face_value := income * 100 / stock_rate
  let market_price := face_value * market_value / 100
  let brokerage := investment - market_price
  (brokerage / market_price) * 100

/-- Theorem stating that given the specific stock details, the brokerage percentage is approximately 0.2578% -/
theorem brokerage_percentage_calculation :
  let stock_rate : ℝ := 10.5
  let income : ℝ := 756
  let investment : ℝ := 7000
  let market_value : ℝ := 96.97222222222223
  |calculate_brokerage_percentage stock_rate income investment market_value - 0.2578| < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brokerage_percentage_calculation_l643_64306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l643_64390

-- Define the square side length
noncomputable def square_side : ℝ := 9

-- Define the ratio of AE to EC
noncomputable def ratio : ℝ := 1/3

-- Theorem statement
theorem triangle_area : 
  let DE := square_side * (1 - ratio)
  let area := (1/2) * DE * square_side
  area = 243/8 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l643_64390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inequality_l643_64357

noncomputable def geometric_sequence (q : ℝ) (n : ℕ) : ℝ := q^(n-1)

noncomputable def b_sequence (q : ℝ) (n : ℕ) : ℝ := n * q^n

noncomputable def S_n (q : ℝ) (n : ℕ) : ℝ := (1 - q^n) / (1 - q)

noncomputable def T_n (q : ℝ) (n : ℕ) : ℝ := 
  (3/4) - (1/4) * q^(n-1) - (n/2) * q^n

theorem sum_inequality (n : ℕ) : 
  T_n (1/3) n < (S_n (1/3) n) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inequality_l643_64357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tara_road_trip_cost_l643_64369

/-- Calculates the total cost of a two-day road trip given specific conditions -/
noncomputable def road_trip_cost (day1_tank : ℝ) (day1_efficiency : ℝ) (day1_miles : ℝ)
                   (day2_tank : ℝ) (day2_efficiency : ℝ) (day2_miles : ℝ)
                   (price1 : ℝ) (price2 : ℝ) (price3 : ℝ) (price4 : ℝ) : ℝ :=
  let day1_gallons := day1_miles / day1_efficiency
  let day2_gallons := day2_miles / day2_efficiency
  let day1_cost := day1_tank * price1 + (day1_gallons - day1_tank + day1_tank) * price2
  let day2_cost := day2_tank * price3 + (day2_gallons - day2_tank + day2_tank) * price4
  day1_cost + day2_cost

/-- The total cost of Tara's road trip is $204.75 -/
theorem tara_road_trip_cost :
  road_trip_cost 12 30 315 15 25 400 3 3.5 4 4.5 = 204.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tara_road_trip_cost_l643_64369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_santiago_stay_duration_l643_64301

/-- Represents a month in a year -/
inductive Month
| January
| February
| March
| April
| May
| June
| July
| August
| September
| October
| November
| December

/-- Calculates the number of months between departure and return -/
def monthsBetween (departure : Month) (returnMonth : Month) : ℕ :=
  sorry

theorem santiago_stay_duration :
  let departure := Month.January
  let returnMonth := Month.December
  monthsBetween departure returnMonth = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_santiago_stay_duration_l643_64301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l643_64343

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l643_64343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_vertex_l643_64365

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs ((x₁ - x₃) * (y₂ - y₁) - (x₁ - x₂) * (y₃ - y₁))

theorem triangle_third_vertex (x : ℝ) : 
  x < 0 → 
  area_triangle (7,5) (0,0) (x,0) = 35 → 
  x = -14 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_vertex_l643_64365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l643_64382

/-- The area of a triangle with vertices at (1, 2), (7, 8), and (1, 6) is 12 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (7, 8)
  let C : ℝ × ℝ := (1, 6)
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  area = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l643_64382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_multiples_l643_64398

theorem sum_of_multiples (m n : ℕ) 
  (h1 : n * (m * (m + 1) / 2) = 120)
  (h2 : n^3 * (m^3 * (m^3 + 1) / 2) = 4032000) :
  n^2 * (m^2 * (m^2 + 1) / 2) = 20800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_multiples_l643_64398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_sum_count_l643_64304

theorem divisible_by_four_sum_count (n : ℕ) : 
  (Finset.filter (fun s : Finset (Fin (4*n)) => s.card = 3 ∧ (s.sum (fun i => i.val) % 4 = 0)) (Finset.powerset (Finset.univ))).card = 
  n^3 + 3*n*(n.choose 2) + n.choose 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_sum_count_l643_64304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_ratio_l643_64350

/-- The ratio of Ben's cards to Tim's cards after Ben's purchase -/
theorem cards_ratio (ben_initial tim ben_bought : ℕ) : 
  ben_initial = 37 → tim = 20 → ben_bought = 3 →
  (ben_initial + ben_bought : ℚ) / tim = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_ratio_l643_64350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_identity_implies_n_pow_m_l643_64392

theorem equation_identity_implies_n_pow_m (m n : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + m = (x - 1)*(x + n)) → n^m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_identity_implies_n_pow_m_l643_64392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_combined_capacity_l643_64383

-- Define the time interval
def TimeInterval : Set ℝ := { t : ℝ | 0 ≤ t ∧ t ≤ 12 }

-- Define the capacity functions for reservoirs A and B
noncomputable def f (t : ℝ) : ℝ := 2 + Real.sin t
def g (t : ℝ) : ℝ := 5 - |t - 6|

-- Define the combined capacity function
noncomputable def H (t : ℝ) : ℝ := f t + g t

-- State the theorem
theorem max_combined_capacity :
  ∃ (t_max : ℝ), t_max ∈ TimeInterval ∧
  (∀ (t : ℝ), t ∈ TimeInterval → H t ≤ H t_max) ∧
  t_max = 6 ∧ H t_max = 7 + Real.sin 6 := by
  sorry

#check max_combined_capacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_combined_capacity_l643_64383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_l643_64344

theorem class_average (total_students : ℕ) (high_score_students : ℕ) (high_score : ℕ)
  (zero_score_students : ℕ) (mid_score_students : ℕ) (mid_score : ℕ)
  (remaining_average : ℕ) :
  total_students = 40 →
  high_score_students = 8 →
  high_score = 95 →
  zero_score_students = 5 →
  mid_score_students = 10 →
  mid_score = 70 →
  remaining_average = 50 →
  (high_score_students * high_score + zero_score_students * 0 + mid_score_students * mid_score +
    (total_students - high_score_students - zero_score_students - mid_score_students) * remaining_average : ℚ) /
    total_students = 57.75 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_l643_64344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l643_64395

/-- The distance between locations A and B in nautical miles -/
noncomputable def distance : ℝ := 600

/-- The maximum sailing speed of the ship in nautical miles per hour -/
noncomputable def max_speed : ℝ := 60

/-- The proportionality constant for fuel cost -/
noncomputable def fuel_cost_constant : ℝ := 0.5

/-- The other costs per hour in yuan -/
noncomputable def other_costs : ℝ := 1250

/-- The total transportation cost as a function of speed -/
noncomputable def total_cost (speed : ℝ) : ℝ :=
  (distance / speed) * (other_costs + fuel_cost_constant * speed^2)

/-- Theorem stating that 50 nautical miles per hour minimizes the total transportation cost -/
theorem optimal_speed_minimizes_cost :
  ∀ x : ℝ, 0 < x → x ≤ max_speed → total_cost 50 ≤ total_cost x := by
  sorry

#check optimal_speed_minimizes_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l643_64395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hospital_bill_total_l643_64328

theorem hospital_bill_total 
  (medication_percent : Real) 
  (overnight_percent : Real) 
  (food_cost : Real) 
  (ambulance_cost : Real) 
  (h1 : medication_percent = 0.5)
  (h2 : overnight_percent = 0.25)
  (h3 : food_cost = 175)
  (h4 : ambulance_cost = 1700)
  : ∃ (total : Real), total = 5000 ∧ 
    medication_percent * total + 
    overnight_percent * (1 - medication_percent) * total + 
    food_cost + ambulance_cost = total := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hospital_bill_total_l643_64328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_properties_l643_64340

/-- A shape is centrally symmetric if it looks the same when rotated 180 degrees around its center. -/
def is_centrally_symmetric (shape : Type) : Prop :=
  sorry

/-- A shape is axially symmetric if there exists at least one axis about which the shape is symmetrical. -/
def is_axially_symmetric (shape : Type) : Prop :=
  sorry

/-- Represents a line segment -/
def line_segment : Type :=
  sorry

/-- Represents a circle -/
def circle' : Type :=
  sorry

/-- Represents an equilateral triangle -/
def equilateral_triangle : Type :=
  sorry

/-- The set of shapes we're considering -/
def shape_set : Set Type :=
  {line_segment, circle', equilateral_triangle}

theorem symmetry_properties :
  ∀ s ∈ shape_set,
    (is_centrally_symmetric s ∧ is_axially_symmetric s) ↔ (s = line_segment ∨ s = circle') :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_properties_l643_64340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_line_slope_range_for_intersection_l643_64391

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos θ, Real.sqrt 2 * Real.sin θ)

-- Define the line l in parametric form
noncomputable def line_l (α t : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, 2 + t * Real.sin α)

-- Theorem for the minimum distance
theorem min_distance_curve_line :
  let α : ℝ := 3 * Real.pi / 4
  ∃ d : ℝ, d = Real.sqrt 2 ∧
    ∀ θ t : ℝ, 
      let P := curve_C θ
      let Q := line_l α t
      d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) := by
  sorry

-- Theorem for the range of slope k
theorem slope_range_for_intersection :
  ∀ k : ℝ, 
    (∃ x₁ y₁ x₂ y₂ : ℝ, 
      x₁ ≠ x₂ ∧
      x₁^2 + y₁^2 = 2 ∧
      x₂^2 + y₂^2 = 2 ∧
      y₁ - 2 = k * (x₁ - 2) ∧
      y₂ - 2 = k * (x₂ - 2)) ↔
    2 - Real.sqrt 3 < k ∧ k < 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_line_slope_range_for_intersection_l643_64391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_cubed_l643_64316

noncomputable def f (x : ℝ) : ℝ := 25 / (7 + 4 * x)

theorem inverse_f_cubed : (Function.invFun f 3)⁻¹ ^ 3 = 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_cubed_l643_64316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_N_div_a5_a5_values_l643_64345

def is_valid_sequence (a : Fin 10 → ℕ) : Prop :=
  ∀ i j, i < j → a i < a j

def N (a : Fin 10 → ℕ) : ℕ :=
  Finset.lcm (Finset.range 10) (λ i => a i)

theorem min_N_div_a5 (a : Fin 10 → ℕ) (h : is_valid_sequence a) :
  (∃ (b : Fin 10 → ℕ), is_valid_sequence b ∧ N b / b 5 < N a / a 5) → N a / a 5 ≥ 6 :=
by sorry

theorem a5_values (a : Fin 10 → ℕ) (h : is_valid_sequence a) :
  (N a / a 1 = 2520) → (a 5 ∈ Set.Icc 1 1500) →
  a 5 = 420 ∨ a 5 = 840 ∨ a 5 = 1260 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_N_div_a5_a5_values_l643_64345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l643_64358

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 64 - x^2 / 16 = 1

-- Define the distance from a point to a focus
noncomputable def distance_to_focus (x y : ℝ) (focus_x focus_y : ℝ) : ℝ :=
  Real.sqrt ((x - focus_x)^2 + (y - focus_y)^2)

theorem hyperbola_focus_distance (x y : ℝ) (focus1_x focus1_y focus2_x focus2_y : ℝ) :
  hyperbola x y →
  distance_to_focus x y focus1_x focus1_y = 4 →
  distance_to_focus x y focus2_x focus2_y = 20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l643_64358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_tautologies_l643_64326

open Classical

-- Define IsTautology
def IsTautology (f : Prop) : Prop :=
  ∀ (v : Prop → Bool), v f = true

theorem all_tautologies :
  (∀ p, IsTautology (p ∨ ¬p)) ∧
  (∀ p, IsTautology ((¬¬p) ↔ p)) ∧
  (∀ p q, IsTautology (((p → q) → p) → p)) ∧
  (∀ p, IsTautology (¬(p ∧ ¬p))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_tautologies_l643_64326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_inscribed_rectangles_correct_l643_64332

/-- Number of inscribed rectangles with diagonal c in a rectangle with sides a ≤ b -/
noncomputable def num_inscribed_rectangles (a b c : ℝ) : ℕ :=
  if a ≤ b ∧ b < c ∧ c < Real.sqrt (a^2 + b^2) then 4
  else if a ≤ b ∧ b = c then 2
  else if a ≤ b ∧ c = Real.sqrt (a^2 + b^2) then 1
  else 0

theorem num_inscribed_rectangles_correct (a b c : ℝ) :
  num_inscribed_rectangles a b c =
    if a ≤ b ∧ b < c ∧ c < Real.sqrt (a^2 + b^2) then 4
    else if a ≤ b ∧ b = c then 2
    else if a ≤ b ∧ c = Real.sqrt (a^2 + b^2) then 1
    else 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_inscribed_rectangles_correct_l643_64332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_approximation_l643_64319

def a : ℕ → ℕ
  | 0 => 3  -- Adding the case for 0
  | 1 => 3
  | 2 => 8
  | n + 3 => 2 * a (n + 2) + 2 * a (n + 1)

theorem a_approximation (n : ℕ) (h : n ≥ 1) :
  |a n - ((1/2 + 1/Real.sqrt 3) * (Real.sqrt 3 + 1)^n)| < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_approximation_l643_64319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_characterization_l643_64381

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

theorem floor_characterization (x : ℝ) :
  ∃! (k : ℤ), (k : ℝ) ≤ x ∧ x < (k : ℝ) + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_characterization_l643_64381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_four_solutions_l643_64370

/-- The number of real solutions to the system of equations x^2 + y = 5 and x + y^2 = 3 -/
def num_solutions : ℕ := 4

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x^2 + y = 5 ∧ x + y^2 = 3

/-- Theorem stating that the system has exactly 4 real solutions -/
theorem system_has_four_solutions :
  ∃! (s : Set (ℝ × ℝ)), s = {(x, y) | system x y} ∧ Finite s ∧ Nat.card s = num_solutions :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_four_solutions_l643_64370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_natural_number_l643_64363

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Add a case for 0 to avoid missing case error
  | 1 => 1
  | n + 2 => (1/2) * a (n + 1) + 1 / (4 * a (n + 1))

theorem sqrt_natural_number (n : ℕ) (h : n > 1) : 
  ∃ k : ℕ, Real.sqrt (2 / (2 * (a n)^2 - 1)) = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_natural_number_l643_64363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_zero_l643_64372

noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ := λ i j ↦ Real.sin (↑i + ↑j + 1)

theorem det_A_eq_zero : Matrix.det A = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_zero_l643_64372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tens_digit_l643_64353

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_divisible_by_sum_of_digits (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem unique_tens_digit : 
  ∃! a : ℕ, 
    (let n := 10 * a + 8;
    is_divisible_by_sum_of_digits n ∧
    is_divisible_by_sum_of_digits (n + 1) ∧
    is_divisible_by_sum_of_digits (n + 2) ∧
    is_divisible_by_sum_of_digits (n + 3)) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tens_digit_l643_64353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l643_64308

/-- A function f with specific properties -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := 1/x^3 + a*x^3 - b*x - 5

/-- Theorem stating the property of f -/
theorem f_property (a b : ℝ) : 
  (f a b (-2) = 2) → (f a b 2 = -12) := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l643_64308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_subsequence_l643_64385

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticProgression (a d : ℝ) : ℕ → ℝ :=
  fun n ↦ a + n * d

/-- A geometric progression is a sequence where each term after the first is found by
    multiplying the previous term by a fixed, non-zero number. -/
def GeometricProgression (a r : ℝ) : ℕ → ℝ :=
  fun n ↦ a * r^n

/-- A subsequence of a sequence is a sequence that can be derived from the original
    sequence by deleting some or no elements without changing the order of the
    remaining elements. -/
def IsSubsequence {α : Type*} (f g : ℕ → α) : Prop :=
  ∃ h : ℕ → ℕ, StrictMono h ∧ g = f ∘ h

/-- Main theorem: An infinite arithmetic progression with positive first term and
    common difference has an infinite geometric subsequence if and only if the
    ratio of the first term to the common difference is rational. -/
theorem arithmetic_geometric_subsequence (a d : ℝ) (ha : a > 0) (hd : d > 0) :
  (∃ (b r : ℝ), IsSubsequence (GeometricProgression b r) (ArithmeticProgression a d)) ↔
  ∃ (m n : ℕ), n ≠ 0 ∧ a / d = m / n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_subsequence_l643_64385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l643_64377

noncomputable def g (x : ℝ) : ℝ := x^2 - 8*x + 20

noncomputable def f (x : ℝ) : ℝ := 1 / ⌊g x⌋

theorem domain_of_f : Set.range f = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l643_64377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l643_64354

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 - 3 / (x + 2)

-- Define the domain
def domain : Set ℝ := Set.Icc 3 5

-- Theorem statement
theorem f_properties :
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → f x < f y) ∧
  (∀ x ∈ domain, f x ≥ 2/5) ∧
  (∀ x ∈ domain, f x ≤ 4/7) ∧
  (∃ x ∈ domain, f x = 2/5) ∧
  (∃ x ∈ domain, f x = 4/7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l643_64354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_east_south_angle_in_twelve_ray_wheel_l643_64339

/-- A wheel pattern with equally-spaced rays -/
structure WheelPattern where
  num_rays : ℕ
  north_ray_index : ℕ

/-- Calculate the angle between two rays in a wheel pattern -/
noncomputable def angle_between_rays (w : WheelPattern) (ray1 : ℕ) (ray2 : ℕ) : ℝ :=
  (((ray2 - ray1) % w.num_rays : ℝ) * (360 / w.num_rays)) % 360

/-- The smaller angle between two rays -/
noncomputable def smaller_angle (w : WheelPattern) (ray1 : ℕ) (ray2 : ℕ) : ℝ :=
  min (angle_between_rays w ray1 ray2) (angle_between_rays w ray2 ray1)

theorem east_south_angle_in_twelve_ray_wheel :
  let w : WheelPattern := { num_rays := 12, north_ray_index := 0 }
  let east_ray_index : ℕ := (w.north_ray_index + 3) % w.num_rays
  let south_ray_index : ℕ := (w.north_ray_index + 6) % w.num_rays
  smaller_angle w east_ray_index south_ray_index = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_east_south_angle_in_twelve_ray_wheel_l643_64339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l643_64389

noncomputable section

/-- The function g as defined in the problem -/
def g (m n p q : ℝ) (x : ℝ) : ℝ := (m * x + n) / (p * x + q)

/-- The theorem statement -/
theorem unique_number_not_in_range 
  (m n p q : ℝ) 
  (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0)
  (h23 : g m n p q 23 = 23)
  (h53 : g m n p q 53 = 53)
  (hgg : ∀ x, x ≠ -q/p → g m n p q (g m n p q x) = x) :
  ∃! y, (∀ x, g m n p q x ≠ y) ∧ y = -38 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l643_64389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_book_cost_l643_64336

theorem coloring_book_cost (bracelet_cost keychain_cost total_spend coloring_book_cost : ℕ) 
  (h1 : bracelet_cost = 4)
  (h2 : keychain_cost = 5)
  (h3 : 2 * bracelet_cost + keychain_cost + (bracelet_cost + coloring_book_cost) = total_spend)
  (h4 : total_spend = 20) : coloring_book_cost = 3 := by
  sorry

#check coloring_book_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_book_cost_l643_64336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_function_equality_l643_64375

noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/2) * x^2
noncomputable def g (x b : ℝ) : ℝ := 1/x + x + b

theorem tangent_line_and_function_equality :
  -- Part 1: Prove that y = -1/2 is a tangent line of f(x) = ln x - (1/2)x²
  (∃ (x₀ : ℝ), x₀ > 0 ∧ f x₀ = -1/2 ∧ (deriv f) x₀ = 0) ∧
  -- Part 2: Prove the range of b
  (∀ (b : ℝ), 
    (∀ (x₁ : ℝ), x₁ ∈ Set.Icc 1 (Real.sqrt (Real.exp 1)) → 
      ∃ (x₂ : ℝ), x₂ ∈ Set.Icc 1 4 ∧ f x₁ = g x₂ b) ↔ 
    b ∈ Set.Icc (-19/4) (-3/2 - (Real.exp 1)/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_function_equality_l643_64375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_f_range_f_decreasing_interval_l643_64379

noncomputable section

def lg (x : ℝ) := Real.log x / Real.log 10

def f (x : ℝ) : ℝ := 10 ^ (3 * x * (3 - x))

axiom f_domain : ∀ x, 0 < x → x < 3 → f x ≠ 0

axiom f_property : ∀ x, 0 < x → x < 3 → lg (lg (f x)) = lg (3 * x) + lg (3 - x)

theorem f_formula : ∀ x, 0 < x → x < 3 → f x = 10 ^ (3 * x * (3 - x)) := by
  sorry

theorem f_range : Set.range f = Set.Ioo 1 (10 ^ (27 / 4)) := by
  sorry

theorem f_decreasing_interval : ∀ x, 3 / 2 ≤ x → x < 3 → 
  ∀ y, 3 / 2 ≤ y → y < x → f x < f y := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_f_range_f_decreasing_interval_l643_64379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charge_to_center_l643_64346

/-- The energy between two point charges of magnitude Q separated by distance d -/
noncomputable def energy (Q : ℝ) (d : ℝ) : ℝ := Q^2 / d

/-- The total energy stored in a configuration of four point charges at the vertices of a square -/
noncomputable def squareEnergy (Q : ℝ) (side : ℝ) : ℝ :=
  4 * energy Q side + 2 * energy Q (side * Real.sqrt 2)

/-- The total energy stored when one charge is moved to the center of the square -/
noncomputable def centerSquareEnergy (Q : ℝ) (side : ℝ) : ℝ :=
  4 * energy Q (side / Real.sqrt 2) + 3 * energy Q side

theorem charge_to_center (Q : ℝ) (side : ℝ) (h : squareEnergy Q side = 40) :
  centerSquareEnergy Q side = (320 + 200 * Real.sqrt 2) / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_charge_to_center_l643_64346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_for_line_intersecting_ln_curve_l643_64364

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := Real.log x / x

-- State the theorem
theorem max_k_for_line_intersecting_ln_curve : 
  ∃ (k_max : ℝ), k_max = 1 / Real.exp 1 ∧ 
  (∀ k : ℝ, (∃ x : ℝ, x > 0 ∧ k * x = Real.log x) → k ≤ k_max) := by
  -- Proof goes here
  sorry

-- Define the derivative of f
noncomputable def f_deriv (x : ℝ) : ℝ := (1 - Real.log x) / (x^2)

-- State a lemma about the behavior of f'(x)
lemma f_deriv_behavior :
  ∀ x : ℝ, x > 0 → 
    (x < Real.exp 1 → f_deriv x > 0) ∧
    (x > Real.exp 1 → f_deriv x < 0) := by
  -- Proof goes here
  sorry

-- State a lemma about the maximum value of f(x)
lemma f_max_value :
  ∀ x : ℝ, x > 0 → f x ≤ 1 / Real.exp 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_for_line_intersecting_ln_curve_l643_64364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_zero_l643_64323

theorem sum_of_solutions_is_zero :
  ∃ (s : Finset ℝ), (∀ x ∈ s, -12 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) ∧ x ≠ 1 ∧ x ≠ -1) ∧
                    (∀ x : ℝ, -12 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) ∧ x ≠ 1 ∧ x ≠ -1 → x ∈ s) ∧
                    (s.sum id = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_zero_l643_64323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l643_64380

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 / (4^x) - x

-- Define constants a, b, and c
def a : ℝ := 0
noncomputable def b : ℝ := Real.log 2 / Real.log 0.4
noncomputable def c : ℝ := Real.log 3 / Real.log 4

-- State the theorem
theorem f_inequality : f a < f c ∧ f c < f b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l643_64380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_equals_sum_of_roots_l643_64361

-- Define the points in the complex plane
noncomputable def O : ℂ := 0
noncomputable def A : ℂ := -3 + 10 * Complex.I
noncomputable def B : ℂ := 4 - 5 * Complex.I

-- Define the distance function between two complex numbers
noncomputable def distance (z w : ℂ) : ℝ := Complex.abs (z - w)

-- Define the perimeter of the triangle
noncomputable def trianglePerimeter : ℝ := distance O A + distance A B + distance B O

-- Theorem statement
theorem triangle_perimeter_equals_sum_of_roots : 
  trianglePerimeter = Real.sqrt 109 + Real.sqrt 274 + Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_equals_sum_of_roots_l643_64361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l643_64315

/-- The complex number z defined as (4+3i)/(1+i) -/
noncomputable def z : ℂ := (4 + 3*Complex.I) / (1 + Complex.I)

/-- Theorem stating that z is in the fourth quadrant -/
theorem z_in_fourth_quadrant : 
  Real.sign z.re = 1 ∧ Real.sign z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l643_64315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2015_is_zero_l643_64330

-- Define the complex number z
noncomputable def z : ℂ := Complex.exp (Complex.I * (Real.pi / 4))

-- Define the sequence of complex numbers
noncomputable def seq (n : ℕ) : ℂ := z ^ n

-- Define the real part of the sequence
noncomputable def a (n : ℕ) : ℝ := (seq n).re

-- Define the imaginary part of the sequence
noncomputable def b (n : ℕ) : ℝ := (seq n).im

-- State the theorem
theorem sum_2015_is_zero : a 2015 + b 2015 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2015_is_zero_l643_64330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_max_probability_l643_64397

/-- The number of trials in the binomial distribution -/
def n : ℕ := 15

/-- The probability of success in each trial -/
noncomputable def p : ℚ := 1/4

/-- The binomial probability mass function -/
noncomputable def binomialPMF (k : ℕ) : ℝ :=
  (n.choose k) * (p^k * (1-p)^(n-k))

/-- Theorem stating that the binomial PMF is maximized at k = 3 or k = 4 -/
theorem binomial_max_probability :
  ∀ k : ℕ, k ≤ n → binomialPMF k ≤ max (binomialPMF 3) (binomialPMF 4) := by
  sorry

#check binomial_max_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_max_probability_l643_64397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l643_64362

-- Define the constants as noncomputable
noncomputable def a : ℝ := (Real.log 2) / 2
noncomputable def b : ℝ := (Real.log 3) / 3
noncomputable def c : ℝ := (Real.log 5) / 5

-- State the theorem
theorem abc_inequality : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l643_64362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_travel_distance_l643_64399

/-- The total distance Jim traveled in kilometers -/
noncomputable def total_distance : ℝ := 168 / 23

/-- Jim's skateboard speed in kilometers per hour -/
noncomputable def skateboard_speed : ℝ := 15

/-- Jim's walking speed in kilometers per hour -/
noncomputable def walking_speed : ℝ := 4

/-- The fraction of the total distance Jim traveled by skateboard -/
noncomputable def skateboard_fraction : ℝ := 2 / 3

/-- The fraction of the total distance Jim traveled by walking -/
noncomputable def walking_fraction : ℝ := 1 / 3

/-- The total travel time in hours -/
noncomputable def total_time : ℝ := 14 / 15

theorem jim_travel_distance :
  (skateboard_fraction * total_distance / skateboard_speed +
   walking_fraction * total_distance / walking_speed) = total_time ∧
  (⌊total_distance * 2 + 0.5⌋ : ℝ) / 2 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_travel_distance_l643_64399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_cosine_sine_sum_l643_64388

theorem shift_cosine_sine_sum (x : ℝ) :
  Real.sqrt 2 * Real.cos (3 * x) = Real.sin (3 * (x + Real.pi/12)) + Real.cos (3 * (x + Real.pi/12)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_cosine_sine_sum_l643_64388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_calculation_l643_64327

/-- The speed of the current in a river --/
noncomputable def current_speed (rowing_speed : ℝ) (distance : ℝ) (time : ℝ) : ℝ :=
  let downstream_speed := distance / time
  let rowing_speed_ms := rowing_speed * 1000 / 3600
  (downstream_speed - rowing_speed_ms) * 3600 / 1000

/-- Theorem stating the speed of the current given the conditions --/
theorem current_speed_calculation :
  let rowing_speed := (20 : ℝ) -- km/h
  let distance := (60 : ℝ) -- meters
  let time := (9.390553103577801 : ℝ) -- seconds
  abs (current_speed rowing_speed distance time - 3.0125) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_calculation_l643_64327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_glucose_in_container_l643_64314

/-- Represents the concentration of glucose in a solution -/
structure GlucoseSolution where
  glucose_grams : ℚ
  solution_cc : ℚ

/-- Calculates the amount of glucose in a given volume of solution -/
def glucose_amount (solution : GlucoseSolution) (volume : ℚ) : ℚ :=
  (solution.glucose_grams * volume) / solution.solution_cc

theorem glucose_in_container 
  (solution : GlucoseSolution)
  (h1 : solution.glucose_grams = 10)
  (h2 : solution.solution_cc = 100)
  (container_volume : ℚ)
  (h3 : container_volume = 45) :
  glucose_amount solution container_volume = 4.5 := by
  sorry

#eval glucose_amount { glucose_grams := 10, solution_cc := 100 } 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_glucose_in_container_l643_64314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_line_perpendicular_to_parabola_axis_l643_64342

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = ax² -/
structure Parabola where
  a : ℝ
  h_a_pos : a > 0

/-- Represents a circle (x - b)² + (y - c)² = r² -/
structure Circle where
  center : Point
  radius : ℝ
  h_r_pos : radius > 0

/-- Theorem: The line connecting the midpoints of the arcs of a circle 
    outside a parabola is perpendicular to the parabola's axis -/
theorem midpoint_line_perpendicular_to_parabola_axis 
  (p : Parabola) (c : Circle) 
  (h_intersect : ∃ P Q R S : Point, 
    P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S ∧
    (P.y = p.a * P.x^2) ∧ (Q.y = p.a * Q.x^2) ∧ (R.y = p.a * R.x^2) ∧ (S.y = p.a * S.x^2) ∧
    ((P.x - c.center.x)^2 + (P.y - c.center.y)^2 = c.radius^2) ∧
    ((Q.x - c.center.x)^2 + (Q.y - c.center.y)^2 = c.radius^2) ∧
    ((R.x - c.center.x)^2 + (R.y - c.center.y)^2 = c.radius^2) ∧
    ((S.x - c.center.x)^2 + (S.y - c.center.y)^2 = c.radius^2))
  (M N : Point) 
  (h_M_midpoint : ∃ arc1 : Set Point, M ∈ arc1 ∧ 
    ∀ X ∈ arc1, (X.x - c.center.x)^2 + (X.y - c.center.y)^2 = c.radius^2 ∧ X.y > p.a * X.x^2)
  (h_N_midpoint : ∃ arc2 : Set Point, N ∈ arc2 ∧ 
    ∀ X ∈ arc2, (X.x - c.center.x)^2 + (X.y - c.center.y)^2 = c.radius^2 ∧ X.y > p.a * X.x^2)
  : M.y = N.y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_line_perpendicular_to_parabola_axis_l643_64342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_problem_l643_64305

theorem cube_root_problem :
  ∀ (a b c : ℝ),
    ((3 * a + 21) ^ (1/3 : ℝ) = 3) →
    (Real.sqrt (b - 1) = 2) →
    (Real.sqrt c = c) →
    (a = 2 ∧ b = 5 ∧ c = 0) ∧
    (Real.sqrt (3 * a + 10 * b + c) = 2 * Real.sqrt 14 ∨
     Real.sqrt (3 * a + 10 * b + c) = -2 * Real.sqrt 14) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_problem_l643_64305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l643_64366

noncomputable def f (x : ℝ) := Real.sin (2 * x) + Real.sin (2 * x - Real.pi / 3)

noncomputable def g (x m : ℝ) := f (x + m)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (m : ℝ), m > 0 ∧ m = 5 * Real.pi / 12 ∧ ∀ (x : ℝ), g x m = g (Real.pi / 4 - x) m) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-Real.sqrt 3 / 2) (3 / 2) ↔ ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 4) ∧ g x (5 * Real.pi / 12) = y) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l643_64366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_final_position_l643_64348

/-- Represents a rectangular park with given dimensions -/
structure RectangularPark where
  length : ℝ
  width : ℝ

/-- Represents a point on the perimeter of the park -/
structure PerimeterPoint where
  distanceFromStart : ℝ

/-- Calculates the perimeter of the park -/
def parkPerimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.width)

/-- Calculates the final position after running a given distance -/
noncomputable def finalPosition (park : RectangularPark) (start : PerimeterPoint) (distance : ℝ) : PerimeterPoint :=
  { distanceFromStart := (start.distanceFromStart + distance) % parkPerimeter park }

theorem athlete_final_position 
  (park : RectangularPark)
  (start : PerimeterPoint)
  (runDistance : ℝ) :
  park.length = 900 →
  park.width = 600 →
  start.distanceFromStart = 550 →
  runDistance = 15500 →
  (finalPosition park start runDistance).distanceFromStart = 1050 := by
  sorry

#check athlete_final_position

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_final_position_l643_64348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_line_l643_64300

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  (|A * x₀ + B * y₀ + C|) / Real.sqrt (A^2 + B^2)

/-- The focus of the parabola y² = 16x -/
def focus : ℝ × ℝ := (4, 0)

/-- The line l₁: x - y + 5 = 0 -/
def line_l₁ (x y : ℝ) : Prop := x - y + 5 = 0

theorem distance_focus_to_line :
  distance_point_to_line (focus.1) (focus.2) 1 (-1) 5 = 9 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_line_l643_64300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pm2_5_scientific_notation_l643_64334

/-- The diameter of PM2.5 particulate matter in meters -/
def pm2_5_diameter : ℝ := 0.0000025

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Theorem stating that pm2_5_diameter can be expressed in scientific notation as 2.5 × 10^(-6) -/
theorem pm2_5_scientific_notation :
  ∃ (sn : ScientificNotation), pm2_5_diameter = sn.coefficient * (10 : ℝ) ^ sn.exponent ∧
  sn.coefficient = 2.5 ∧ sn.exponent = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pm2_5_scientific_notation_l643_64334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l643_64368

theorem percentage_difference (x : ℝ) (x_pos : x > 0) : 
  let first := 0.65 * x
  let second := 0.58 * x
  ∃ ε > 0, abs ((first - second) / first * 100 - 10.77) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l643_64368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_plus_fraction_squared_eq_20_implies_y_values_l643_64378

theorem x_squared_plus_fraction_squared_eq_20_implies_y_values
  (x : ℝ) (hx : x^2 + 2 * (x / (x - 1))^2 = 20) :
  ∃ y : ℝ, ((x - 1)^3 * (x + 2)) / (2 * x - 1) = y ∧ (y = 14 ∨ y = -56/3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_plus_fraction_squared_eq_20_implies_y_values_l643_64378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_domain_interval_l643_64307

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3 * x + 1) / (x + 8)

-- State the theorem about the domain of g(x)
theorem domain_of_g :
  {x : ℝ | ∃ y, g x = y} = {x : ℝ | x ≠ -8} := by
  sorry

-- Additional theorem to express the domain in interval notation
theorem domain_interval :
  {x : ℝ | ∃ y, g x = y} = Set.Ioi (-8) ∪ Set.Iio (-8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_domain_interval_l643_64307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l643_64351

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℚ
  deriving Repr

/-- Represents a configuration of four squares aligned side-by-side -/
structure SquareConfiguration where
  squares : Fin 4 → Square
  deriving Repr

/-- Calculates the area of the trapezoid formed by the diagonal segment -/
noncomputable def trapezoidArea (config : SquareConfiguration) : ℚ :=
  sorry

/-- The main theorem stating the area of the trapezoid -/
theorem trapezoid_area_theorem (config : SquareConfiguration) 
  (h1 : config.squares 0 = ⟨1⟩)
  (h2 : config.squares 1 = ⟨3⟩)
  (h3 : config.squares 2 = ⟨5⟩)
  (h4 : config.squares 3 = ⟨7⟩) :
  trapezoidArea config = 455/32 := by
  sorry

#eval (455:ℚ)/32

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l643_64351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l643_64318

-- Define the lines
noncomputable def line1 (x : ℝ) : ℝ := 3 * x + 6
noncomputable def line2 (x : ℝ) : ℝ := -2 * x + 12

-- Define the intersection point of the two lines
noncomputable def intersection : ℝ × ℝ :=
  let x := 6 / 5
  (x, line1 x)

-- Define the y-intercepts
noncomputable def y_intercept1 : ℝ := line1 0
noncomputable def y_intercept2 : ℝ := line2 0

-- Define the base of the triangle
noncomputable def base : ℝ := y_intercept2 - y_intercept1

-- Define the height of the triangle
noncomputable def triangle_height : ℝ := intersection.1

-- Define the area of the triangle
noncomputable def triangle_area : ℝ := (1 / 2) * base * triangle_height

-- Theorem statement
theorem area_of_bounded_region :
  triangle_area = 18 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l643_64318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_3_neg3_l643_64376

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 && y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, if θ < 0 then θ + 2 * Real.pi else θ)

theorem rectangular_to_polar_3_neg3 :
  let (r, θ) := rectangular_to_polar 3 (-3)
  r = 3 * Real.sqrt 2 ∧ θ = 7 * Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_3_neg3_l643_64376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l643_64371

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_length + bridge_length
  total_distance / train_speed_mps

/-- Theorem stating that a train 100 meters long takes 12.5 seconds to cross a 150-meter bridge at 72 kmph -/
theorem train_crossing_bridge : train_crossing_time 100 150 72 = 12.5 := by
  -- Expand the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l643_64371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_properties_l643_64313

/-- Cylinder properties -/
structure Cylinder where
  base_diameter : ℝ
  height : ℝ

/-- Volume of a cylinder -/
noncomputable def cylinder_volume (c : Cylinder) : ℝ :=
  Real.pi * (c.base_diameter / 2) ^ 2 * c.height

/-- Curved surface area of a cylinder -/
noncomputable def cylinder_curved_surface_area (c : Cylinder) : ℝ :=
  2 * Real.pi * (c.base_diameter / 2) * c.height

/-- Theorem about the volume and curved surface area of a specific cylinder -/
theorem cylinder_properties :
  let c : Cylinder := { base_diameter := 14, height := 60 }
  (abs (cylinder_volume c - 9233.886) < 0.001) ∧
  (abs (cylinder_curved_surface_area c - 2638.936) < 0.001) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_properties_l643_64313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_c_sequence_l643_64341

noncomputable def sequence_a : ℕ → ℝ := sorry
noncomputable def sequence_b : ℕ → ℝ := sorry
noncomputable def sequence_c : ℕ → ℝ := sorry
noncomputable def S : ℕ → ℝ := sorry
noncomputable def T : ℕ → ℝ := sorry

axiom a_1 : sequence_a 1 = 2

axiom S_def : ∀ n : ℕ, 2 * S n = (n + 1)^2 * sequence_a n - n^2 * sequence_a (n + 1)

axiom b_1 : sequence_b 1 = sequence_a 1

axiom b_relation : ∀ n : ℕ, n * sequence_b (n + 1) = sequence_a n * sequence_b n

axiom c_def : ∀ n : ℕ, sequence_c n = sequence_a n + sequence_b n

theorem sum_of_c_sequence (n : ℕ) : T n = 2^(n + 1) + n^2 + n - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_c_sequence_l643_64341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_equals_sum_of_others_l643_64309

structure EquilateralTriangleInCircle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  circle_center : ℝ × ℝ
  radius : ℝ
  is_equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                   (B.1 - C.1)^2 + (B.1 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  on_circle : (M.1 - circle_center.1)^2 + (M.2 - circle_center.2)^2 = radius^2

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem longest_segment_equals_sum_of_others (t : EquilateralTriangleInCircle) :
  let MA := distance t.M t.A
  let MB := distance t.M t.B
  let MC := distance t.M t.C
  max MA (max MB MC) = min MA (min MB MC) + (MA + MB + MC - max MA (max MB MC) - min MA (min MB MC)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_equals_sum_of_others_l643_64309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimizes_sum_of_squared_distances_l643_64374

noncomputable section

-- Define the triangle vertices
def A : ℝ × ℝ := (5, 0)
def B : ℝ × ℝ := (0, 5)
def C : ℝ × ℝ := (-4, -3)

-- Define the point P
noncomputable def P : ℝ × ℝ := (Real.sqrt 5, 2 * Real.sqrt 5)

-- Define the distance squared function
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Define the sum of squared distances
def sumOfSquaredDistances (p : ℝ × ℝ) : ℝ :=
  distanceSquared p A + distanceSquared p B + distanceSquared p C

-- Define the circumcircle
def isOnCircumcircle (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 = 25

-- Theorem statement
theorem minimizes_sum_of_squared_distances :
  isOnCircumcircle P ∧
  ∀ q : ℝ × ℝ, isOnCircumcircle q → sumOfSquaredDistances P ≤ sumOfSquaredDistances q :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimizes_sum_of_squared_distances_l643_64374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l643_64310

/-- The parabola equation -/
def parabola_eq (p : ℝ × ℝ) : Prop := p.2 = 2 * p.1^2

/-- The focus of a parabola -/
def focus (f : ℝ × ℝ) (eq : ℝ × ℝ → Prop) : Prop :=
  ∀ p, eq p → (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.2 - f.2 + f.2/2)^2

/-- Theorem: The focus of the parabola y = 2x^2 has coordinates (0, 1/8) -/
theorem parabola_focus :
  focus (0, 1/8) parabola_eq := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l643_64310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_thirteen_plus_y_l643_64302

theorem mod_thirteen_plus_y (y : ℕ) (h : 7 * y ≡ 1 [MOD 31]) : (13 + y) % 31 = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_thirteen_plus_y_l643_64302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_diagonal_l643_64367

/-- IsoscelesTrapezoid represents an isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  long_base : ℝ
  short_base : ℝ
  side : ℝ

/-- diagonal_length calculates the length of the diagonal in an isosceles trapezoid -/
noncomputable def diagonal_length (t : IsoscelesTrapezoid) : ℝ :=
  Real.sqrt (((t.long_base + t.short_base) / 2) ^ 2 + (t.side ^ 2 - ((t.long_base - t.short_base) / 2) ^ 2))

/-- The theorem stating that the diagonal length of the specific isosceles trapezoid is √457 -/
theorem specific_trapezoid_diagonal :
  let t : IsoscelesTrapezoid := { long_base := 24, short_base := 12, side := 13 }
  diagonal_length t = Real.sqrt 457 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_diagonal_l643_64367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_after_transform_l643_64373

-- Define a type for a list of four real numbers
def FourNumbers := Fin 4 → ℝ

-- Define the mean of four numbers
noncomputable def mean (x : FourNumbers) : ℝ := (x 0 + x 1 + x 2 + x 3) / 4

-- Define the variance of four numbers
noncomputable def variance (x : FourNumbers) : ℝ :=
  ((x 0 - mean x)^2 + (x 1 - mean x)^2 + (x 2 - mean x)^2 + (x 3 - mean x)^2) / 4

-- Define the transformation function
def transform (x : FourNumbers) : FourNumbers :=
  fun i => 3 * x i + 5

-- Theorem statement
theorem variance_after_transform (x : FourNumbers) :
  variance x = 7 → variance (transform x) = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_after_transform_l643_64373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_b_minus_a_l643_64312

theorem sum_of_max_min_b_minus_a (a b : ℝ) (h1 : 0 ≤ a ∧ a ≤ 8) (h2 : 0 ≤ b ∧ b ≤ 8) (h3 : b^2 = 16 + a^2) :
  ∃ (max min : ℝ),
    let f := fun x => (16 : ℝ) / (Real.sqrt (16 + x^2) + x)
    max ∈ Set.Icc 0 8 ∧ min ∈ Set.Icc 0 8 ∧
    (∀ y ∈ Set.Icc 0 8, f min ≤ f y ∧ f y ≤ f max) ∧
    f max + f min = 12 - 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_b_minus_a_l643_64312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_8_value_l643_64384

def our_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 2 * a n + a (n + 1) = 0

theorem a_8_value (a : ℕ → ℝ) (h1 : our_sequence a) (h2 : a 3 = -2) : a 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_8_value_l643_64384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l643_64359

/-- Definition of the sequence a_n -/
noncomputable def a (n : ℕ+) : ℝ := n

/-- Definition of S_n (sum of first n terms of a_n) -/
noncomputable def S (n : ℕ+) : ℝ := (n * (n + 1)) / 2

/-- Definition of b_n -/
noncomputable def b (n : ℕ+) : ℝ := 1 / (a n)^3

theorem sequence_properties :
  (∀ n : ℕ+, 2 * S n = (a n)^2 + n) ∧
  (∀ n : ℕ+, a n > 0) ∧
  (∀ n : ℕ+, a n = n) ∧
  (∀ n : ℕ+, (Finset.range n).sum (λ i => b ⟨i + 1, Nat.succ_pos i⟩) < 5/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l643_64359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_composition_identity_l643_64324

-- Define a point in a 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a reflection operation over a point
def reflect (center : Point) (p : Point) : Point := sorry

-- Define the composition of reflections
def reflectComposition (o₁ o₂ o₃ : Point) (p : Point) : Point :=
  reflect o₃ (reflect o₂ (reflect o₁ p))

-- Define the identity transformation
def identityTransform (p : Point) : Point := p

-- Theorem statement
theorem reflection_composition_identity (o₁ o₂ o₃ : Point) :
  ∀ p : Point, reflectComposition o₃ o₂ o₁ (reflectComposition o₃ o₂ o₁ p) = identityTransform p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_composition_identity_l643_64324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_assignment_l643_64347

-- Define a type for points with integer coordinates
structure Point where
  x : Int
  y : Int

-- Define a function type for the assignment of natural numbers to points
def Assignment := Point → Nat

-- Define a predicate for three points being collinear
def areCollinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

-- Define a predicate for numbers having a common divisor greater than one
def hasCommonDivisorGreaterThanOne (a b c : Nat) : Prop :=
  ∃ (d : Nat), d > 1 ∧ d ∣ a ∧ d ∣ b ∧ d ∣ c

-- State the theorem
theorem no_valid_assignment :
  ¬∃ (f : Assignment),
    ∀ (p q r : Point),
      areCollinear p q r ↔ hasCommonDivisorGreaterThanOne (f p) (f q) (f r) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_assignment_l643_64347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_complex_root_of_equations_l643_64317

theorem common_complex_root_of_equations (m n : ℕ) (h : m ≠ n) :
  ∃ (x : ℂ), (x^(m+1) - x^n + 1 = 0 ∧ x^(n+1) - x^m + 1 = 0) ↔ 
  (x = (1/2 : ℂ) + (Complex.I * Real.sqrt 3/2) ∨ x = (1/2 : ℂ) - (Complex.I * Real.sqrt 3/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_complex_root_of_equations_l643_64317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_isosceles_triangle_properties_l643_64333

/-- Represents a right isosceles triangle with side length 8 -/
structure RightIsoscelesTriangle where
  /-- Side length of the triangle -/
  side : ℝ
  /-- The side length is positive -/
  side_pos : side > 0

/-- The area of a right isosceles triangle -/
noncomputable def area (t : RightIsoscelesTriangle) : ℝ := t.side * t.side / 2

/-- The diameter of the circumscribed circle of a right isosceles triangle -/
noncomputable def circumscribed_circle_diameter (t : RightIsoscelesTriangle) : ℝ := t.side * Real.sqrt 2

theorem right_isosceles_triangle_properties (t : RightIsoscelesTriangle) 
  (h : t.side = 8) : area t = 32 ∧ circumscribed_circle_diameter t = 8 * Real.sqrt 2 := by
  sorry

#check right_isosceles_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_isosceles_triangle_properties_l643_64333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sin_2x_squared_plus_x_l643_64356

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := sin (2 * x^2 + x)

-- State the theorem
theorem derivative_of_sin_2x_squared_plus_x :
  ∀ x : ℝ, deriv f x = (4 * x + 1) * cos (2 * x^2 + x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sin_2x_squared_plus_x_l643_64356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_spheres_l643_64393

def sphere1_center : ℝ × ℝ × ℝ := (0, -5, 3)
def sphere1_radius : ℝ := 23

def sphere2_center : ℝ × ℝ × ℝ := (13, 15, -20)
def sphere2_radius : ℝ := 92

theorem max_distance_between_spheres :
  ∀ (p1 p2 : ℝ × ℝ × ℝ),
    (‖p1 - sphere1_center‖ = sphere1_radius) →
    (‖p2 - sphere2_center‖ = sphere2_radius) →
    ‖p1 - p2‖ ≤ 148.1 :=
by
  sorry

#check max_distance_between_spheres

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_spheres_l643_64393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_of_2x_plus_5_when_x_is_2_l643_64349

theorem cube_of_2x_plus_5_when_x_is_2 :
  (λ x : ℝ ↦ (2 * x + 5) ^ 3) 2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_of_2x_plus_5_when_x_is_2_l643_64349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_mod_11_l643_64396

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def A : ℕ → ℝ
  | 0 => 0
  | n + 1 => A n + floor ((19 : ℝ) ^ (n + 1) / 20)

theorem A_mod_11 : ∃ (k : ℤ), A 2020 = 11 * k + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_mod_11_l643_64396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_in_interval_root_in_interval_l643_64335

/-- The function f(x) = ln x + 2x - 6 -/
noncomputable def f (x : ℝ) := Real.log x + 2 * x - 6

/-- Theorem: The function f(x) = ln x + 2x - 6 has a root in the interval (5/2, 3) -/
theorem f_has_root_in_interval :
  ∃ x, x ∈ Set.Ioo (5/2 : ℝ) 3 ∧ f x = 0 := by
  sorry

/-- Theorem: The root of f(x) = ln x + 2x - 6 is in the interval (k/2, (k+1)/2) where k = 5 -/
theorem root_in_interval :
  ∃ x, x ∈ Set.Ioo (5/2 : ℝ) 3 ∧ f x = 0 ∧ (∃ k : ℤ, k = 5 ∧ x ∈ Set.Ioo (k/2 : ℝ) ((k+1)/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_in_interval_root_in_interval_l643_64335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l643_64329

noncomputable def coefficient_of_x (p : ℝ → ℝ) : ℝ :=
  sorry -- Definition of coefficient_of_x goes here

theorem binomial_expansion_coefficient :
  let n : ℕ := 8
  let sum_of_coefficients : ℕ := 256
  let expansion := (λ x : ℝ => (1 - 3*x)^n)
  (sum_of_coefficients = 2^n) →
  (coefficient_of_x expansion = -56) :=
by
  sorry

#check binomial_expansion_coefficient

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l643_64329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_with_difference_nine_l643_64394

theorem least_n_with_difference_nine : ∃ n : ℕ, 
  (∀ S : Finset ℕ, S ⊆ Finset.range 100 → S.card ≥ n → 
    ∃ x y, x ∈ S ∧ y ∈ S ∧ y - x = 9) ∧ 
  (∀ m : ℕ, m < n → 
    ∃ T : Finset ℕ, T ⊆ Finset.range 100 ∧ T.card = m ∧ 
      ∀ x y, x ∈ T → y ∈ T → y - x ≠ 9) ∧
  n = 51 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_with_difference_nine_l643_64394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_point_of_f_l643_64321

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 2) / (x + 1)

-- State the theorem
theorem lowest_point_of_f :
  ∀ x : ℝ, x > -1 → f x ≥ 2 ∧ f 0 = 2 := by
  sorry

#check lowest_point_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_point_of_f_l643_64321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_color_circle_existence_l643_64331

-- Define a color type
inductive Color
  | First
  | Second
  | Third

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define a circle type
structure Circle where
  center : Point
  radius : ℝ

-- Define membership for Point in Circle
def Point.inCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

instance : Membership Point Circle where
  mem := Point.inCircle

-- State the theorem
theorem three_color_circle_existence 
  (coloring : Coloring) 
  (all_colors_used : ∀ c : Color, ∃ p : Point, coloring p = c) :
  ∃ circle : Circle, 
    (∃ p1 p2 p3 : Point, 
      p1 ∈ circle ∧ p2 ∈ circle ∧ p3 ∈ circle ∧
      coloring p1 = Color.First ∧ 
      coloring p2 = Color.Second ∧ 
      coloring p3 = Color.Third) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_color_circle_existence_l643_64331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_two_element_sets_with_distinct_sums_l643_64360

/-- The set of integers from 1 to 2018 -/
def Ω : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2018}

/-- The type of two-element subsets of Ω -/
def TwoElementSet : Type := {A : Finset ℕ // A.toSet ⊆ Ω ∧ A.card = 2}

/-- The sum of two sets -/
def setSum (A B : Finset ℕ) : Finset ℕ := (A.product B).image (fun p => p.1 + p.2)

/-- The property that all pairwise sums are distinct -/
def allSumsDistinct (S : Finset TwoElementSet) : Prop :=
  ∀ A B C D, A ∈ S → B ∈ S → C ∈ S → D ∈ S → 
    setSum A.val B.val = setSum C.val D.val → (A = C ∧ B = D) ∨ (A = D ∧ B = C)

/-- The main theorem -/
theorem max_two_element_sets_with_distinct_sums :
  (∃ (S : Finset TwoElementSet), S.card = 4033 ∧ allSumsDistinct S) ∧
  (∀ (S : Finset TwoElementSet), allSumsDistinct S → S.card ≤ 4033) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_two_element_sets_with_distinct_sums_l643_64360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orvin_max_balloons_l643_64355

/-- Represents the store's balloon sale policy -/
structure BalloonSale where
  regularPrice : ℚ
  saleDiscount : ℚ

/-- Calculates the maximum number of balloons that can be bought given a budget and sale policy -/
def maxBalloons (budget : ℚ) (sale : BalloonSale) : ℕ :=
  let pairCost := sale.regularPrice + (sale.regularPrice * (1 - sale.saleDiscount))
  let pairs := (budget / pairCost).floor
  let remainingMoney := budget - (pairs * pairCost)
  if remainingMoney ≥ sale.regularPrice then
    (2 * pairs + 1).toNat
  else
    (2 * pairs).toNat

/-- Theorem stating that given the conditions, Orvin can buy at most 53 balloons -/
theorem orvin_max_balloons :
  let regularPrice : ℚ := 3
  let budget : ℚ := 40 * regularPrice
  let sale := BalloonSale.mk regularPrice (1/2)
  maxBalloons budget sale = 53 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orvin_max_balloons_l643_64355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_numerator_simplify_denominator_l643_64325

theorem rationalize_numerator_simplify_denominator :
  let original := (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3
  let rationalized := original * (Real.sqrt 3 + Real.sqrt 2) / (Real.sqrt 3 + Real.sqrt 2)
  rationalized = 1 / (3 + Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_numerator_simplify_denominator_l643_64325
