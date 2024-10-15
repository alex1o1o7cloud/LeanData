import Mathlib

namespace NUMINAMATH_CALUDE_meat_for_community_event_l1579_157959

/-- The amount of meat (in pounds) needed to make a given number of hamburgers. -/
def meat_needed (hamburgers : ℕ) : ℚ :=
  (5 : ℚ) * hamburgers / 10

/-- Theorem stating that 15 pounds of meat are needed for 30 hamburgers. -/
theorem meat_for_community_event : meat_needed 30 = 15 := by
  sorry

end NUMINAMATH_CALUDE_meat_for_community_event_l1579_157959


namespace NUMINAMATH_CALUDE_rope_and_well_l1579_157994

theorem rope_and_well (x y : ℝ) (h : (1/4) * x = y + 3) : (1/5) * x = y + 2 := by
  sorry

end NUMINAMATH_CALUDE_rope_and_well_l1579_157994


namespace NUMINAMATH_CALUDE_complex_computation_l1579_157986

theorem complex_computation :
  let A : ℂ := 3 + 2*I
  let B : ℂ := -1 - 2*I
  let C : ℂ := 5*I
  let D : ℂ := 3 + I
  2 * (A - B + C + D) = 8 + 20*I :=
by sorry

end NUMINAMATH_CALUDE_complex_computation_l1579_157986


namespace NUMINAMATH_CALUDE_complex_modulus_example_l1579_157932

theorem complex_modulus_example : 
  let z : ℂ := 1 - (5/4)*I
  Complex.abs z = Real.sqrt 41 / 4 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l1579_157932


namespace NUMINAMATH_CALUDE_bracelet_selling_price_l1579_157934

theorem bracelet_selling_price 
  (total_bracelets : ℕ)
  (given_away : ℕ)
  (material_cost : ℚ)
  (profit : ℚ)
  (h1 : total_bracelets = 52)
  (h2 : given_away = 8)
  (h3 : material_cost = 3)
  (h4 : profit = 8) :
  let sold_bracelets := total_bracelets - given_away
  let total_sales := profit + material_cost
  let price_per_bracelet := total_sales / sold_bracelets
  price_per_bracelet = 1/4 := by
sorry

end NUMINAMATH_CALUDE_bracelet_selling_price_l1579_157934


namespace NUMINAMATH_CALUDE_add_7455_seconds_to_8_15_00_l1579_157981

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The starting time: 8:15:00 -/
def startTime : Time :=
  { hours := 8, minutes := 15, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 7455

/-- The expected final time: 10:19:15 -/
def expectedFinalTime : Time :=
  { hours := 10, minutes := 19, seconds := 15 }

theorem add_7455_seconds_to_8_15_00 :
  addSeconds startTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_add_7455_seconds_to_8_15_00_l1579_157981


namespace NUMINAMATH_CALUDE_complex_power_thousand_l1579_157925

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Main theorem: ((1 + i) / (1 - i)) ^ 1000 = 1 -/
theorem complex_power_thousand :
  ((1 + i) / (1 - i)) ^ 1000 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_complex_power_thousand_l1579_157925


namespace NUMINAMATH_CALUDE_perpendicular_line_proof_l1579_157947

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (0, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 6 = 0

-- Theorem statement
theorem perpendicular_line_proof :
  (∀ x y : ℝ, perpendicular_line x y → given_line x y → x * x + y * y = 0) ∧
  perpendicular_line point.1 point.2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_proof_l1579_157947


namespace NUMINAMATH_CALUDE_circle_area_through_points_l1579_157965

/-- The area of a circle with center P(2, -1) passing through Q(-4, 6) is 85π -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (2, -1)
  let Q : ℝ × ℝ := (-4, 6)
  let r := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  π * r^2 = 85 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l1579_157965


namespace NUMINAMATH_CALUDE_class_size_l1579_157916

theorem class_size (n : ℕ) (best_rank : ℕ) (worst_rank : ℕ) 
  (h1 : best_rank = 30) 
  (h2 : worst_rank = 25) 
  (h3 : n = (best_rank - 1) + (worst_rank - 1) + 1) : 
  n = 54 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1579_157916


namespace NUMINAMATH_CALUDE_inequality_properties_l1579_157915

theorem inequality_properties (x y : ℝ) (h : x > y) :
  (x - 3 > y - 3) ∧
  (x / 3 > y / 3) ∧
  (x + 3 > y + 3) ∧
  (1 - 3*x < 1 - 3*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l1579_157915


namespace NUMINAMATH_CALUDE_workers_gone_home_is_120_l1579_157930

/-- Represents the problem of workers leaving a factory for Chinese New Year --/
structure WorkerProblem where
  total_days : Nat
  weekend_days : Nat
  remaining_workers : Nat
  total_worker_days : Nat

/-- The specific instance of the worker problem --/
def factory_problem : WorkerProblem := {
  total_days := 15
  weekend_days := 4
  remaining_workers := 121
  total_worker_days := 2011
}

/-- Calculates the number of workers who have gone home --/
def workers_gone_home (p : WorkerProblem) : Nat :=
  sorry

/-- Theorem stating that 120 workers have gone home --/
theorem workers_gone_home_is_120 : 
  workers_gone_home factory_problem = 120 := by
  sorry

end NUMINAMATH_CALUDE_workers_gone_home_is_120_l1579_157930


namespace NUMINAMATH_CALUDE_scooter_gain_percentage_l1579_157900

/-- Calculates the overall gain percentage for three scooters -/
def overall_gain_percentage (purchase_price_A purchase_price_B purchase_price_C : ℚ)
                            (repair_cost_A repair_cost_B repair_cost_C : ℚ)
                            (selling_price_A selling_price_B selling_price_C : ℚ) : ℚ :=
  let total_cost := purchase_price_A + purchase_price_B + purchase_price_C +
                    repair_cost_A + repair_cost_B + repair_cost_C
  let total_revenue := selling_price_A + selling_price_B + selling_price_C
  let total_gain := total_revenue - total_cost
  (total_gain / total_cost) * 100

/-- Theorem stating that the overall gain percentage for the given scooter transactions is 10% -/
theorem scooter_gain_percentage :
  overall_gain_percentage 4700 3500 5400 600 800 1000 5800 4800 7000 = 10 := by
  sorry

end NUMINAMATH_CALUDE_scooter_gain_percentage_l1579_157900


namespace NUMINAMATH_CALUDE_max_value_ad_minus_bc_l1579_157931

theorem max_value_ad_minus_bc :
  ∀ a b c d : ℤ,
  a ∈ ({-1, 1, 2} : Set ℤ) →
  b ∈ ({-1, 1, 2} : Set ℤ) →
  c ∈ ({-1, 1, 2} : Set ℤ) →
  d ∈ ({-1, 1, 2} : Set ℤ) →
  (∀ x y z w : ℤ,
    x ∈ ({-1, 1, 2} : Set ℤ) →
    y ∈ ({-1, 1, 2} : Set ℤ) →
    z ∈ ({-1, 1, 2} : Set ℤ) →
    w ∈ ({-1, 1, 2} : Set ℤ) →
    x * w - y * z ≤ 6) ∧
  (∃ x y z w : ℤ,
    x ∈ ({-1, 1, 2} : Set ℤ) ∧
    y ∈ ({-1, 1, 2} : Set ℤ) ∧
    z ∈ ({-1, 1, 2} : Set ℤ) ∧
    w ∈ ({-1, 1, 2} : Set ℤ) ∧
    x * w - y * z = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_value_ad_minus_bc_l1579_157931


namespace NUMINAMATH_CALUDE_clayton_shells_proof_l1579_157978

/-- The number of shells collected by Jillian -/
def jillian_shells : ℕ := 29

/-- The number of shells collected by Savannah -/
def savannah_shells : ℕ := 17

/-- The number of friends who received shells -/
def num_friends : ℕ := 2

/-- The number of shells each friend received -/
def shells_per_friend : ℕ := 27

/-- The number of shells Clayton collected -/
def clayton_shells : ℕ := 8

theorem clayton_shells_proof :
  clayton_shells = 
    num_friends * shells_per_friend - (jillian_shells + savannah_shells) :=
by sorry

end NUMINAMATH_CALUDE_clayton_shells_proof_l1579_157978


namespace NUMINAMATH_CALUDE_annes_bottle_caps_l1579_157938

/-- Anne's initial number of bottle caps -/
def initial_caps : ℕ := sorry

/-- The number of bottle caps Anne finds -/
def found_caps : ℕ := 5

/-- Anne's final number of bottle caps -/
def final_caps : ℕ := 15

/-- Theorem stating that Anne's initial number of bottle caps plus the found caps equals her final number of caps -/
theorem annes_bottle_caps : initial_caps + found_caps = final_caps := by sorry

end NUMINAMATH_CALUDE_annes_bottle_caps_l1579_157938


namespace NUMINAMATH_CALUDE_library_visitors_sunday_visitors_proof_l1579_157997

/-- Calculates the average number of visitors on Sundays in a library -/
theorem library_visitors (non_sunday_visitors : ℕ) (total_average : ℕ) : ℕ :=
  let total_days : ℕ := 30
  let sundays : ℕ := 5
  let non_sundays : ℕ := total_days - sundays
  let sunday_visitors : ℕ := 
    (total_average * total_days - non_sunday_visitors * non_sundays) / sundays
  sunday_visitors

/-- Proves that the average number of Sunday visitors is 510 given the conditions -/
theorem sunday_visitors_proof (h1 : library_visitors 240 285 = 510) : 
  library_visitors 240 285 = 510 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_sunday_visitors_proof_l1579_157997


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1579_157905

theorem geometric_progression_fourth_term (a : ℝ) (r : ℝ) :
  (∃ (b c : ℝ), a = 3^(3/4) ∧ b = 3^(2/4) ∧ c = 3^(1/4) ∧ 
   b / a = c / b) → 
  c^2 / b = 1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1579_157905


namespace NUMINAMATH_CALUDE_inequality_holds_l1579_157910

theorem inequality_holds (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : 0 < c) (h4 : c < 1) :
  b * (a ^ c) < a * (b ^ c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l1579_157910


namespace NUMINAMATH_CALUDE_floor_equation_solutions_range_l1579_157927

theorem floor_equation_solutions_range (a : ℝ) (n : ℕ) 
  (h1 : a > 1) 
  (h2 : n ≥ 2) 
  (h3 : ∃! (S : Finset ℝ), S.card = n ∧ ∀ x ∈ S, ⌊a * x⌋ = x) :
  1 + 1 / n ≤ a ∧ a < 1 + 1 / (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_range_l1579_157927


namespace NUMINAMATH_CALUDE_proportional_division_l1579_157944

theorem proportional_division (total : ℚ) (a b c : ℚ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 104 →
  (a : ℚ) = 2 →
  (b : ℚ) = 1/2 →
  (c : ℚ) = 1/4 →
  ∃ (x : ℚ), a * x + b * x + c * x = total ∧ b * x = 208/11 := by
  sorry

end NUMINAMATH_CALUDE_proportional_division_l1579_157944


namespace NUMINAMATH_CALUDE_number_puzzle_l1579_157902

theorem number_puzzle (x y : ℝ) (h1 : x + y = 25) (h2 : x - y = 15) : x^2 - y^3 = 275 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1579_157902


namespace NUMINAMATH_CALUDE_reciprocal_problem_l1579_157963

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 5) : 200 * (1 / x) = 320 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l1579_157963


namespace NUMINAMATH_CALUDE_prime_power_sum_l1579_157953

theorem prime_power_sum (w x y z : ℕ) :
  2^w * 3^x * 5^y * 7^z = 882 →
  2*w + 3*x + 5*y + 7*z = 22 := by
sorry

end NUMINAMATH_CALUDE_prime_power_sum_l1579_157953


namespace NUMINAMATH_CALUDE_points_four_units_from_negative_one_l1579_157948

theorem points_four_units_from_negative_one : 
  ∀ x : ℝ, abs (x - (-1)) = 4 ↔ x = 3 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_points_four_units_from_negative_one_l1579_157948


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l1579_157971

theorem quadratic_root_sum (m n : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 - Complex.I * Real.sqrt 3)^2 + m * (1 - Complex.I * Real.sqrt 3) + n = 0 →
  m + n = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l1579_157971


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1579_157946

theorem inequality_and_equality_condition (a b c : ℝ) :
  (5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * b * c + 4 * a * c) ∧
  (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * b * c + 4 * a * c ↔ a = 0 ∧ b = 0 ∧ c = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1579_157946


namespace NUMINAMATH_CALUDE_smallest_angle_3_4_5_triangle_l1579_157951

theorem smallest_angle_3_4_5_triangle :
  ∀ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    a^2 + b^2 = c^2 →
    a / c = 3 / 5 ∧ b / c = 4 / 5 →
    min (Real.arctan (a / b)) (Real.arctan (b / a)) = Real.arctan (3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_3_4_5_triangle_l1579_157951


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1579_157941

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a - 8 > b - 8) → ¬(a > b)) ↔ ((a - 8 ≤ b - 8) → (a ≤ b)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1579_157941


namespace NUMINAMATH_CALUDE_billys_age_l1579_157913

theorem billys_age (B J A : ℕ) 
  (h1 : B = 3 * J)
  (h2 : J = A / 2)
  (h3 : B + J + A = 90) :
  B = 45 := by sorry

end NUMINAMATH_CALUDE_billys_age_l1579_157913


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l1579_157901

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 → x % 4 = 0 → x^3 < 500 → x ≤ 4 ∧ ∃ y : ℕ, y > 0 ∧ y % 4 = 0 ∧ y^3 < 500 ∧ y = 4 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l1579_157901


namespace NUMINAMATH_CALUDE_intersection_points_sum_l1579_157998

def f (x : ℝ) : ℝ := (x + 2) * (x - 4)

def g (x : ℝ) : ℝ := -f x

def h (x : ℝ) : ℝ := f (2 * x)

def intersection_points_fg : ℕ := 2

def intersection_points_fh : ℕ := 2

theorem intersection_points_sum : 
  10 * intersection_points_fg + intersection_points_fh = 22 := by sorry

end NUMINAMATH_CALUDE_intersection_points_sum_l1579_157998


namespace NUMINAMATH_CALUDE_max_value_theorem_l1579_157920

theorem max_value_theorem (x y z : ℝ) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0) 
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 8 + 6 * y * z ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1579_157920


namespace NUMINAMATH_CALUDE_cello_count_l1579_157922

/-- Given a music store with cellos and violas, prove the number of cellos. -/
theorem cello_count (violas : ℕ) (matching_pairs : ℕ) (probability : ℚ) (cellos : ℕ) : 
  violas = 600 →
  matching_pairs = 70 →
  probability = 70 / (cellos * 600) →
  probability = 0.00014583333333333335 →
  cellos = 800 := by
sorry

#eval (70 : ℚ) / (800 * 600)  -- To verify the probability

end NUMINAMATH_CALUDE_cello_count_l1579_157922


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l1579_157987

/-- Given that (-4, y₁) and (2, y₂) both lie on the line y = -2x + 3, prove that y₁ > y₂ -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : 
  y₁ = -2 * (-4) + 3 → y₂ = -2 * 2 + 3 → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l1579_157987


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1579_157917

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin point (0, 0, 0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Symmetric point with respect to the origin -/
def symmetricPoint (p : Point3D) : Point3D :=
  ⟨-p.x, -p.y, -p.z⟩

theorem symmetric_point_coordinates :
  let p : Point3D := ⟨1, -2, 1⟩
  let q : Point3D := symmetricPoint p
  q = ⟨-1, 2, -1⟩ := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1579_157917


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1579_157960

theorem circle_radius_from_area_circumference_ratio 
  (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) (h3 : P / Q = 10) : 
  ∃ r : ℝ, r > 0 ∧ P = π * r^2 ∧ Q = 2 * π * r ∧ r = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1579_157960


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l1579_157969

theorem smallest_gcd_bc (a b c : ℕ+) (hab : Nat.gcd a b = 120) (hac : Nat.gcd a c = 360) :
  ∃ (b' c' : ℕ+), Nat.gcd a b' = 120 ∧ Nat.gcd a c' = 360 ∧ Nat.gcd b' c' = 120 ∧
  ∀ (b'' c'' : ℕ+), Nat.gcd a b'' = 120 → Nat.gcd a c'' = 360 → Nat.gcd b'' c'' ≥ 120 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l1579_157969


namespace NUMINAMATH_CALUDE_monotonicity_intervals_when_a_2_range_of_a_with_extreme_point_l1579_157954

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3*x + 1

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 6*a*x + 3

-- Part I: Monotonicity intervals when a = 2
theorem monotonicity_intervals_when_a_2 :
  let a := 2
  ∀ x : ℝ, 
    (x ≤ 2 - Real.sqrt 3 ∨ x ≥ 2 + Real.sqrt 3 → f' a x > 0) ∧
    (2 - Real.sqrt 3 < x ∧ x < 2 + Real.sqrt 3 → f' a x < 0) :=
sorry

-- Part II: Range of a when f(x) has at least one extreme value point in (2,3)
theorem range_of_a_with_extreme_point :
  ∀ a : ℝ,
    (∃ x : ℝ, 2 < x ∧ x < 3 ∧ f' a x = 0) →
    (5/4 < a ∧ a < 5/3) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_intervals_when_a_2_range_of_a_with_extreme_point_l1579_157954


namespace NUMINAMATH_CALUDE_drought_periods_correct_max_water_storage_volume_l1579_157937

noncomputable def v (t : ℝ) : ℝ :=
  if 0 < t ∧ t ≤ 9 then
    (1 / 240) * (-(t^2) + 15*t - 51) * Real.exp t + 50
  else if 9 < t ∧ t ≤ 12 then
    4 * (t - 9) * (3*t - 41) + 50
  else
    0

def isDroughtPeriod (t : ℝ) : Prop := v t < 50

def monthToPeriod (m : ℕ) : Set ℝ := {t | m - 1 < t ∧ t ≤ m}

def droughtMonths : Set ℕ := {1, 2, 3, 4, 5, 10, 11, 12}

theorem drought_periods_correct (m : ℕ) (hm : m ∈ droughtMonths) :
  ∀ t ∈ monthToPeriod m, isDroughtPeriod t :=
sorry

theorem max_water_storage_volume :
  ∃ t ∈ Set.Icc (0 : ℝ) 12, v t = 150 ∧ ∀ s ∈ Set.Icc (0 : ℝ) 12, v s ≤ v t :=
sorry

axiom e_cubed_eq_20 : Real.exp 3 = 20

end NUMINAMATH_CALUDE_drought_periods_correct_max_water_storage_volume_l1579_157937


namespace NUMINAMATH_CALUDE_f_is_even_l1579_157909

def f (x : ℝ) : ℝ := 2 * x^2 - 1

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l1579_157909


namespace NUMINAMATH_CALUDE_mistaken_addition_correction_l1579_157999

theorem mistaken_addition_correction (x : ℤ) : x + 16 = 64 → x - 16 = 32 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_addition_correction_l1579_157999


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1579_157972

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (sample_size : ℕ) (start : ℕ) : ℕ → ℕ :=
  fun n => start + (n - 1) * (total / sample_size)

theorem systematic_sampling_theorem :
  let total := 200
  let sample_size := 40
  let group_size := total / sample_size
  let fifth_group_sample := 22
  systematic_sample total sample_size fifth_group_sample 8 = 37 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1579_157972


namespace NUMINAMATH_CALUDE_painting_choices_l1579_157988

/-- The number of traditional Chinese paintings -/
def traditional_paintings : ℕ := 5

/-- The number of oil paintings -/
def oil_paintings : ℕ := 2

/-- The number of watercolor paintings -/
def watercolor_paintings : ℕ := 7

/-- The number of ways to choose one painting from each category -/
def choose_one_each : ℕ := traditional_paintings * oil_paintings * watercolor_paintings

/-- The number of ways to choose two paintings of different types -/
def choose_two_different : ℕ := 
  traditional_paintings * oil_paintings + 
  traditional_paintings * watercolor_paintings + 
  oil_paintings * watercolor_paintings

theorem painting_choices :
  choose_one_each = 70 ∧ choose_two_different = 59 := by
  sorry

end NUMINAMATH_CALUDE_painting_choices_l1579_157988


namespace NUMINAMATH_CALUDE_shelter_dogs_l1579_157964

theorem shelter_dogs (D C R P : ℕ) : 
  D * 7 = C * 15 →  -- Initial ratio of dogs to cats
  R * 5 = P * 9 →   -- Initial ratio of rabbits to parrots
  D * 11 = (C + 8) * 15 →  -- New ratio of dogs to cats after adding 8 cats
  (R + 6) * 5 = P * 7 →    -- New ratio of rabbits to parrots after adding 6 rabbits
  D = 30 :=
by sorry

end NUMINAMATH_CALUDE_shelter_dogs_l1579_157964


namespace NUMINAMATH_CALUDE_tan_equality_periodic_l1579_157957

theorem tan_equality_periodic (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) → n = -30 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_periodic_l1579_157957


namespace NUMINAMATH_CALUDE_sandys_books_l1579_157906

theorem sandys_books (total_spent : ℕ) (books_second_shop : ℕ) (avg_price : ℕ) :
  total_spent = 1920 →
  books_second_shop = 55 →
  avg_price = 16 →
  ∃ (books_first_shop : ℕ), 
    books_first_shop = 65 ∧
    avg_price * (books_first_shop + books_second_shop) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_sandys_books_l1579_157906


namespace NUMINAMATH_CALUDE_range_of_m_l1579_157911

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : x + 2*y = x*y) 
  (h_ineq : ∀ m : ℝ, m^2 + 2*m < x + 2*y) : 
  m ∈ Set.Ioo (-2 : ℝ) 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1579_157911


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l1579_157976

/-- Represents a seating arrangement for 3 people on 5 chairs -/
structure SeatingArrangement where
  seats : Fin 5 → Option (Fin 3)
  all_seated : ∀ p : Fin 3, ∃ s : Fin 5, seats s = some p
  no_sharing : ∀ s : Fin 5, ∀ p q : Fin 3, seats s = some p → seats s = some q → p = q
  ab_adjacent : ∃ s : Fin 5, (seats s = some 0 ∧ seats (s + 1) = some 1) ∨ (seats s = some 1 ∧ seats (s + 1) = some 0)
  not_all_adjacent : ¬∃ s : Fin 5, (seats s).isSome ∧ (seats (s + 1)).isSome ∧ (seats (s + 2)).isSome

/-- The number of valid seating arrangements -/
def num_seating_arrangements : ℕ := sorry

/-- Theorem stating that there are exactly 12 valid seating arrangements -/
theorem seating_arrangements_count : num_seating_arrangements = 12 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l1579_157976


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l1579_157996

theorem tan_double_angle_special_case (θ : ℝ) :
  2 * Real.cos (θ - π / 3) = 3 * Real.cos θ →
  Real.tan (2 * θ) = -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l1579_157996


namespace NUMINAMATH_CALUDE_female_guests_from_jays_family_l1579_157949

def total_guests : ℕ := 240
def female_percentage : ℚ := 60 / 100
def jays_family_percentage : ℚ := 50 / 100

theorem female_guests_from_jays_family :
  (total_guests : ℚ) * female_percentage * jays_family_percentage = 72 := by
  sorry

end NUMINAMATH_CALUDE_female_guests_from_jays_family_l1579_157949


namespace NUMINAMATH_CALUDE_james_walking_distance_l1579_157958

def base7_to_base10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

theorem james_walking_distance :
  base7_to_base10 3 6 5 2 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_james_walking_distance_l1579_157958


namespace NUMINAMATH_CALUDE_range_of_a_l1579_157945

-- Define p as a predicate on m
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧ 
  ∃ (c : ℝ), c > 0 ∧ (∀ (x y : ℝ), x^2 / (m - 1) + y^2 / (2 - m) = 1 → y^2 ≤ c^2)

-- Define q as a predicate on m and a
def q (m a : ℝ) : Prop := m^2 - (2*a + 1)*m + a^2 + a < 0

-- State the theorem
theorem range_of_a : 
  (∀ m : ℝ, p m → ∃ a : ℝ, q m a) → 
  ∃ a : ℝ, 1/2 ≤ a ∧ a ≤ 1 ∧ 
    (∀ b : ℝ, (∀ m : ℝ, p m → q m b) → 1/2 ≤ b ∧ b ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1579_157945


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1579_157939

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 13*x₁ + 40 = 0) ∧
  (x₂^2 - 13*x₂ + 40 = 0) ∧
  x₁ = 5 ∧
  x₂ = 8 ∧
  x₁ > 0 ∧
  x₂ > 0 ∧
  x₂ > x₁ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1579_157939


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1579_157982

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_solution_set : ∀ x, f a b c x > 0 ↔ -1/2 < x ∧ x < 3) :
  c > 0 ∧ 4*a + 2*b + c > 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1579_157982


namespace NUMINAMATH_CALUDE_specific_polygon_triangulation_l1579_157973

/-- Represents a convex polygon with additional internal points -/
structure EnhancedPolygon where
  sides : ℕ
  internal_points : ℕ
  no_collinear_triples : Prop

/-- Represents the triangulation of an EnhancedPolygon -/
def triangulation (p : EnhancedPolygon) : ℕ := sorry

/-- The theorem stating the number of triangles in the specific polygon -/
theorem specific_polygon_triangulation :
  ∀ (p : EnhancedPolygon),
    p.sides = 1000 →
    p.internal_points = 500 →
    p.no_collinear_triples →
    triangulation p = 1998 := by sorry

end NUMINAMATH_CALUDE_specific_polygon_triangulation_l1579_157973


namespace NUMINAMATH_CALUDE_billy_points_billy_points_proof_l1579_157924

theorem billy_points : ℕ → Prop := fun b =>
  let friend_points : ℕ := 9
  let point_difference : ℕ := 2
  (b - friend_points = point_difference) → (b = 11)

-- The proof is omitted
theorem billy_points_proof : billy_points 11 := by sorry

end NUMINAMATH_CALUDE_billy_points_billy_points_proof_l1579_157924


namespace NUMINAMATH_CALUDE_triangle_max_value_l1579_157933

/-- In a triangle ABC, given the conditions, prove the maximum value of (1/2)b + a -/
theorem triangle_max_value (a b c : ℝ) (h1 : a^2 + b^2 = c^2 + a*b) (h2 : c = 1) :
  (∃ (x y : ℝ), x^2 + y^2 = 1^2 + x*y ∧ (1/2)*y + x ≤ (1/2)*b + a) ∧
  (∀ (x y : ℝ), x^2 + y^2 = 1^2 + x*y → (1/2)*y + x ≤ (1/2)*b + a) →
  (1/2)*b + a = Real.sqrt 21 / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_value_l1579_157933


namespace NUMINAMATH_CALUDE_pencil_count_l1579_157955

theorem pencil_count (reeta_pencils : ℕ) (anika_pencils : ℕ) : 
  reeta_pencils = 20 →
  anika_pencils = 2 * reeta_pencils + 4 →
  anika_pencils + reeta_pencils = 64 := by
sorry

end NUMINAMATH_CALUDE_pencil_count_l1579_157955


namespace NUMINAMATH_CALUDE_expression_evaluation_l1579_157956

theorem expression_evaluation (a b : ℝ) (h1 : a = 1) (h2 : b = -2) :
  (a * b - 3 * a^2) - 2 * b^2 - 5 * a * b - (a^2 - 2 * a * b) = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1579_157956


namespace NUMINAMATH_CALUDE_birthday_celebration_attendance_l1579_157912

theorem birthday_celebration_attendance 
  (total_guests : ℕ) 
  (women_percentage men_percentage : ℚ) 
  (men_left_fraction women_left_fraction : ℚ) 
  (children_left : ℕ) 
  (h1 : total_guests = 750)
  (h2 : women_percentage = 432 / 1000)
  (h3 : men_percentage = 314 / 1000)
  (h4 : men_left_fraction = 5 / 12)
  (h5 : women_left_fraction = 7 / 15)
  (h6 : children_left = 19) :
  ∃ (women_count men_count children_count : ℕ),
    women_count + men_count + children_count = total_guests ∧
    women_count = ⌊women_percentage * total_guests⌋ ∧
    men_count = ⌈men_percentage * total_guests⌉ ∧
    children_count = total_guests - women_count - men_count ∧
    total_guests - 
      (⌊men_left_fraction * men_count⌋ + 
       ⌊women_left_fraction * women_count⌋ + 
       children_left) = 482 := by
  sorry


end NUMINAMATH_CALUDE_birthday_celebration_attendance_l1579_157912


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l1579_157970

theorem quadratic_factorization_sum (p q r : ℤ) : 
  (∀ x, x^2 + 19*x + 88 = (x + p) * (x + q)) →
  (∀ x, x^2 - 23*x + 132 = (x - q) * (x - r)) →
  p + q + r = 31 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l1579_157970


namespace NUMINAMATH_CALUDE_basketball_problem_l1579_157923

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let descents := List.range (bounces + 1) |>.map (fun n => initialHeight * reboundFactor ^ n)
  let ascents := List.range bounces |>.map (fun n => initialHeight * reboundFactor ^ (n + 1))
  (descents.sum + ascents.sum)

/-- The basketball problem -/
theorem basketball_problem :
  totalDistance 150 (2/5) 5 = 347.952 := by
  sorry

end NUMINAMATH_CALUDE_basketball_problem_l1579_157923


namespace NUMINAMATH_CALUDE_inequality_solution_range_inequality_equal_solution_sets_l1579_157992

-- Define the inequality
def inequality (m x : ℝ) : Prop := m * x - 3 > 2 * x + m

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ := {x | x < (m + 3) / (m - 2)}

-- Define the alternate inequality
def alt_inequality (x : ℝ) : Prop := 2 * x - 1 > 3 - x

theorem inequality_solution_range (m : ℝ) :
  (∀ x, inequality m x ↔ x ∈ solution_set m) → m < 2 := by sorry

theorem inequality_equal_solution_sets (m : ℝ) :
  (∀ x, inequality m x ↔ alt_inequality x) → m = 17 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_inequality_equal_solution_sets_l1579_157992


namespace NUMINAMATH_CALUDE_fixed_point_on_circle_l1579_157918

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the endpoints of the transverse axis
def A₁ : ℝ × ℝ := (2, 0)
def A₂ : ℝ × ℝ := (-2, 0)

-- Define a point P on the hyperbola
def P : ℝ × ℝ → Prop := λ p => 
  hyperbola p.1 p.2 ∧ p ≠ A₁ ∧ p ≠ A₂

-- Define the line x = 1
def line_x_1 (x y : ℝ) : Prop := x = 1

-- Define the intersection points M₁ and M₂
def M₁ (p : ℝ × ℝ) : ℝ × ℝ := sorry
def M₂ (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define a circle with diameter M₁M₂
def circle_M₁M₂ (p c : ℝ × ℝ) : Prop := sorry

-- The theorem to prove
theorem fixed_point_on_circle : 
  ∃ c : ℝ × ℝ, ∀ p : ℝ × ℝ, P p → circle_M₁M₂ p c := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_circle_l1579_157918


namespace NUMINAMATH_CALUDE_parallel_vectors_k_l1579_157919

def a : Fin 2 → ℝ := ![2, -1]
def b : Fin 2 → ℝ := ![1, 1]
def c : Fin 2 → ℝ := ![-5, 1]

theorem parallel_vectors_k (k : ℝ) :
  (∀ i : Fin 2, (a i + k * b i) * c (1 - i) = (a (1 - i) + k * b (1 - i)) * c i) →
  k = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_l1579_157919


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l1579_157975

/-- A linear function y = (1-2m)x + m + 1 passes through the first, second, and third quadrants
    if and only if -1 < m < 1/2 -/
theorem linear_function_quadrants (m : ℝ) :
  (∀ x y : ℝ, y = (1 - 2*m)*x + m + 1 →
    (∃ x₁ y₁, x₁ > 0 ∧ y₁ > 0 ∧ y₁ = (1 - 2*m)*x₁ + m + 1) ∧
    (∃ x₂ y₂, x₂ < 0 ∧ y₂ > 0 ∧ y₂ = (1 - 2*m)*x₂ + m + 1) ∧
    (∃ x₃ y₃, x₃ < 0 ∧ y₃ < 0 ∧ y₃ = (1 - 2*m)*x₃ + m + 1)) ↔
  -1 < m ∧ m < 1/2 :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l1579_157975


namespace NUMINAMATH_CALUDE_receipts_change_l1579_157926

theorem receipts_change 
  (original_price : ℝ) 
  (original_sales : ℝ) 
  (price_reduction_rate : ℝ) 
  (sales_increase_rate : ℝ) 
  (h1 : price_reduction_rate = 0.3)
  (h2 : sales_increase_rate = 0.5) :
  let new_price := original_price * (1 - price_reduction_rate)
  let new_sales := original_sales * (1 + sales_increase_rate)
  let original_receipts := original_price * original_sales
  let new_receipts := new_price * new_sales
  (new_receipts - original_receipts) / original_receipts = 0.05 := by
sorry

end NUMINAMATH_CALUDE_receipts_change_l1579_157926


namespace NUMINAMATH_CALUDE_division_problem_l1579_157943

/-- Given a division with quotient 20, divisor 66, and remainder 55, the dividend is 1375. -/
theorem division_problem :
  ∀ (dividend quotient divisor remainder : ℕ),
    quotient = 20 →
    divisor = 66 →
    remainder = 55 →
    dividend = divisor * quotient + remainder →
    dividend = 1375 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1579_157943


namespace NUMINAMATH_CALUDE_total_flight_distance_l1579_157921

theorem total_flight_distance (beka_distance jackson_distance maria_distance : ℕ) 
  (h1 : beka_distance = 873)
  (h2 : jackson_distance = 563)
  (h3 : maria_distance = 786) :
  beka_distance + jackson_distance + maria_distance = 2222 := by
  sorry

end NUMINAMATH_CALUDE_total_flight_distance_l1579_157921


namespace NUMINAMATH_CALUDE_planet_can_be_fully_explored_l1579_157966

/-- Represents a spherical planet -/
structure Planet :=
  (equatorial_length : ℝ)

/-- Represents a rover's exploration path on the planet -/
structure ExplorationPath :=
  (length : ℝ)
  (covers_all_points : Bool)

/-- Checks if an exploration path fully explores the planet -/
def fully_explores (p : Planet) (path : ExplorationPath) : Prop :=
  path.length ≤ 600 ∧ path.covers_all_points = true

/-- Theorem stating that the planet can be fully explored -/
theorem planet_can_be_fully_explored (p : Planet) 
  (h : p.equatorial_length = 400) : 
  ∃ path : ExplorationPath, fully_explores p path :=
sorry

end NUMINAMATH_CALUDE_planet_can_be_fully_explored_l1579_157966


namespace NUMINAMATH_CALUDE_set_of_integers_between_10_and_16_l1579_157935

def S : Set ℤ := {n | 10 < n ∧ n < 16}

theorem set_of_integers_between_10_and_16 : S = {11, 12, 13, 14, 15} := by
  sorry

end NUMINAMATH_CALUDE_set_of_integers_between_10_and_16_l1579_157935


namespace NUMINAMATH_CALUDE_factor_expression_l1579_157907

theorem factor_expression (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) = (3 * y + 5) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1579_157907


namespace NUMINAMATH_CALUDE_heart_ratio_l1579_157991

/-- The heart operation defined as n ♥ m = n^2 * m^3 -/
def heart (n m : ℝ) : ℝ := n^2 * m^3

/-- Theorem stating that (3 ♥ 5) / (5 ♥ 3) = 5/3 -/
theorem heart_ratio : (heart 3 5) / (heart 5 3) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_heart_ratio_l1579_157991


namespace NUMINAMATH_CALUDE_children_outnumber_parents_l1579_157929

/-- Represents a family unit in the apartment block -/
structure Family where
  parents : Nat
  boys : Nat
  girls : Nat

/-- Represents the apartment block -/
structure ApartmentBlock where
  families : List Family

/-- Every couple has at least one child -/
axiom at_least_one_child (f : Family) : f.boys + f.girls ≥ 1

/-- Every child has exactly two parents -/
axiom two_parents (f : Family) : f.parents = 2

/-- Every little boy has a sister -/
axiom boys_have_sisters (f : Family) : f.boys > 0 → f.girls > 0

/-- Among the children, there are more boys than girls -/
axiom more_boys_than_girls (ab : ApartmentBlock) :
  (ab.families.map (λ f => f.boys)).sum > (ab.families.map (λ f => f.girls)).sum

/-- There are no grandparents living in the building -/
axiom no_grandparents (ab : ApartmentBlock) :
  (ab.families.map (λ f => f.parents)).sum = 2 * ab.families.length

theorem children_outnumber_parents (ab : ApartmentBlock) :
  (ab.families.map (λ f => f.boys + f.girls)).sum > (ab.families.map (λ f => f.parents)).sum :=
sorry

end NUMINAMATH_CALUDE_children_outnumber_parents_l1579_157929


namespace NUMINAMATH_CALUDE_mollys_bike_age_l1579_157977

/-- Molly's bike riding problem -/
theorem mollys_bike_age : 
  ∀ (miles_per_day : ℕ) (age_stopped : ℕ) (total_miles : ℕ) (days_per_year : ℕ),
  miles_per_day = 3 →
  age_stopped = 16 →
  total_miles = 3285 →
  days_per_year = 365 →
  age_stopped - (total_miles / miles_per_day / days_per_year) = 13 := by
sorry

end NUMINAMATH_CALUDE_mollys_bike_age_l1579_157977


namespace NUMINAMATH_CALUDE_product_of_roots_l1579_157984

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 22 → ∃ y : ℝ, (x + 3) * (x - 5) = 22 ∧ (y + 3) * (y - 5) = 22 ∧ x * y = -37 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1579_157984


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1579_157980

theorem sum_of_fractions : (10 + 20 + 30 + 40) / 10 + 10 / (10 + 20 + 30 + 40) = 10.1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1579_157980


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1579_157989

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then 1 - Real.log x
  else if x > 1 then -1 + Real.log x
  else 0  -- This case is added to make the function total, but it's not used in our problem

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_ab : f a = f b) :
  (∃ m : ℝ, m = 1 + 1 / Real.exp 2 ∧ ∀ x y : ℝ, 0 < x → 0 < y → f x = f y → 1 / x + 1 / y ≥ m) :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1579_157989


namespace NUMINAMATH_CALUDE_no_solution_for_quadratic_congruence_l1579_157962

theorem no_solution_for_quadratic_congruence :
  ∀ n : ℕ, ¬(100 ∣ (n^2 + 6*n + 2019)) := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_quadratic_congruence_l1579_157962


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l1579_157974

theorem consecutive_pages_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 285 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l1579_157974


namespace NUMINAMATH_CALUDE_distance_between_points_l1579_157928

theorem distance_between_points (m : ℝ) :
  let P : ℝ × ℝ × ℝ := (m, 0, 0)
  let P₁ : ℝ × ℝ × ℝ := (4, 1, 2)
  (m - 4)^2 + 1^2 + 2^2 = 30 → m = 9 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1579_157928


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1579_157904

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let base_radius := r / 2
  let cone_height := Real.sqrt (r^2 - base_radius^2)
  (1/3 : ℝ) * Real.pi * base_radius^2 * cone_height = 9 * Real.pi * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1579_157904


namespace NUMINAMATH_CALUDE_auto_store_sales_time_l1579_157995

theorem auto_store_sales_time (total_cars : ℕ) (salespeople : ℕ) (cars_per_person : ℕ) :
  total_cars = 500 →
  salespeople = 10 →
  cars_per_person = 10 →
  (total_cars / (salespeople * cars_per_person) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_auto_store_sales_time_l1579_157995


namespace NUMINAMATH_CALUDE_gcd_power_of_two_minus_one_l1579_157950

theorem gcd_power_of_two_minus_one :
  Nat.gcd (2^2022 - 1) (2^2036 - 1) = 2^14 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_of_two_minus_one_l1579_157950


namespace NUMINAMATH_CALUDE_sheila_weekly_earnings_l1579_157979

/-- Calculates the weekly earnings of a worker given their work schedule and hourly wage. -/
def weekly_earnings (hours_per_day_1 : ℕ) (days_1 : ℕ) (hours_per_day_2 : ℕ) (days_2 : ℕ) (hourly_wage : ℕ) : ℕ :=
  (hours_per_day_1 * days_1 + hours_per_day_2 * days_2) * hourly_wage

/-- Proves that Sheila's weekly earnings are $216 given her work schedule and hourly wage. -/
theorem sheila_weekly_earnings :
  weekly_earnings 8 3 6 2 6 = 216 := by
  sorry

end NUMINAMATH_CALUDE_sheila_weekly_earnings_l1579_157979


namespace NUMINAMATH_CALUDE_percentage_problem_l1579_157952

theorem percentage_problem (x : ℝ) : x * 2 = 0.8 → x * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1579_157952


namespace NUMINAMATH_CALUDE_fence_cost_calculation_l1579_157914

/-- The cost of building a fence around a rectangular plot -/
def fence_cost (length width price_length price_width : ℕ) : ℕ :=
  2 * (length * price_length + width * price_width)

/-- Theorem: The cost of the fence for the given dimensions and prices is 5408 -/
theorem fence_cost_calculation :
  fence_cost 17 21 59 81 = 5408 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_calculation_l1579_157914


namespace NUMINAMATH_CALUDE_target_state_reachable_l1579_157983

/-- Represents the state of the urn -/
structure UrnState where
  black : ℕ
  white : ℕ

/-- Represents the possible operations on the urn -/
inductive Operation
  | replaceBlacks
  | replaceBlackWhite
  | replaceWhiteBlack
  | replaceWhites

/-- Applies an operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.replaceBlacks => 
      if state.black ≥ 3 then UrnState.mk (state.black - 1) state.white
      else state
  | Operation.replaceBlackWhite => 
      if state.black ≥ 2 && state.white ≥ 1 then UrnState.mk (state.black - 2) (state.white + 1)
      else state
  | Operation.replaceWhiteBlack => 
      if state.black ≥ 1 && state.white ≥ 2 then UrnState.mk state.black (state.white - 1)
      else state
  | Operation.replaceWhites => 
      if state.white ≥ 3 then UrnState.mk (state.black + 1) (state.white - 3)
      else state

/-- Checks if the target state is reachable from the initial state -/
def isReachable (initial : UrnState) (target : UrnState) : Prop :=
  ∃ (sequence : List Operation), 
    List.foldl applyOperation initial sequence = target

/-- The main theorem stating that the target state is reachable -/
theorem target_state_reachable : 
  isReachable (UrnState.mk 80 120) (UrnState.mk 1 2) := by
  sorry

end NUMINAMATH_CALUDE_target_state_reachable_l1579_157983


namespace NUMINAMATH_CALUDE_inequality_proof_l1579_157967

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.sin (80 * π / 180))
  (hb : b = (1/2)⁻¹)
  (hc : c = Real.log 3 / Real.log (1/2)) :
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1579_157967


namespace NUMINAMATH_CALUDE_stamp_collection_problem_l1579_157942

theorem stamp_collection_problem (tom_initial : ℕ) (tom_final : ℕ) (harry_extra : ℕ) :
  tom_initial = 3000 →
  tom_final = 3061 →
  harry_extra = 10 →
  ∃ (mike : ℕ),
    mike = 17 ∧
    tom_final = tom_initial + mike + (2 * mike + harry_extra) :=
by sorry

end NUMINAMATH_CALUDE_stamp_collection_problem_l1579_157942


namespace NUMINAMATH_CALUDE_vector_equation_solution_l1579_157993

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a x : V) :
  3 • (a + x) = x → x = -(3/2 : ℝ) • a := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l1579_157993


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l1579_157985

theorem quadrilateral_inequality (a b c d e f : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (h1 : a + b > e) (h2 : c + d > e) (h3 : a + d > f) (h4 : b + c > f) :
  (a + b + c + d) * (e + f) > 2 * (e^2 + f^2) := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l1579_157985


namespace NUMINAMATH_CALUDE_vacation_cost_balance_l1579_157936

/-- Proves that the difference between what Tom and Dorothy owe Sammy is -50 --/
theorem vacation_cost_balance (tom_paid dorothy_paid sammy_paid t d : ℚ) : 
  tom_paid = 140 →
  dorothy_paid = 90 →
  sammy_paid = 220 →
  (tom_paid + t) = (dorothy_paid + d) →
  (tom_paid + t) = (sammy_paid - t - d) →
  t - d = -50 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_balance_l1579_157936


namespace NUMINAMATH_CALUDE_notebook_pen_equation_l1579_157903

theorem notebook_pen_equation (x : ℝ) : 
  (5 * (x - 2) + 3 * x = 14) ↔ 
  (∃ (notebook_price : ℝ), 
    notebook_price = x - 2 ∧ 
    5 * notebook_price + 3 * x = 14) :=
by sorry

end NUMINAMATH_CALUDE_notebook_pen_equation_l1579_157903


namespace NUMINAMATH_CALUDE_intersection_probability_l1579_157940

/-- Given probabilities for events a and b, prove their intersection probability -/
theorem intersection_probability (a b : Set α) (p : Set α → ℝ) 
  (ha : p a = 0.18)
  (hb : p b = 0.5)
  (hba : p (b ∩ a) / p a = 0.2) :
  p (a ∩ b) = 0.036 := by
  sorry

end NUMINAMATH_CALUDE_intersection_probability_l1579_157940


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1579_157990

theorem sum_of_numbers (a b : ℝ) 
  (h1 : a^2 - b^2 = 6) 
  (h2 : (a-2)^2 - (b-2)^2 = 18) : 
  a + b = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1579_157990


namespace NUMINAMATH_CALUDE_sales_equation_l1579_157961

theorem sales_equation (x : ℝ) 
  (h1 : x > 0) 
  (h2 : 5000 / (x + 1) > 0) -- sales quantity last year is positive
  (h3 : 5000 / x > 0) -- sales quantity this year is positive
  (h4 : 5000 / (x + 1) = 5000 / x) -- sales quantity remains the same
  : 5000 / (x + 1) = 5000 * (1 - 0.2) / x := by
  sorry

end NUMINAMATH_CALUDE_sales_equation_l1579_157961


namespace NUMINAMATH_CALUDE_bonus_remainder_l1579_157908

theorem bonus_remainder (X : ℕ) (h : X % 5 = 2) : (3 * X) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_bonus_remainder_l1579_157908


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l1579_157968

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (area : ℝ)
  (valid : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)

-- Define the theorem
theorem triangle_area_inequality (ABC : Triangle) :
  ∃ (A₁B₁C₁ : Triangle),
    A₁B₁C₁.a = Real.sqrt ABC.a ∧
    A₁B₁C₁.b = Real.sqrt ABC.b ∧
    A₁B₁C₁.c = Real.sqrt ABC.c ∧
    A₁B₁C₁.area ^ 2 ≥ (ABC.area * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l1579_157968
