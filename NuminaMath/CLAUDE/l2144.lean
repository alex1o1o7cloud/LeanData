import Mathlib

namespace NUMINAMATH_CALUDE_sam_spent_pennies_l2144_214495

/-- Given that Sam initially had 98 pennies and now has 5 pennies left,
    prove that he spent 93 pennies. -/
theorem sam_spent_pennies (initial : Nat) (remaining : Nat) (spent : Nat)
    (h1 : initial = 98)
    (h2 : remaining = 5)
    (h3 : spent = initial - remaining) :
  spent = 93 := by
  sorry

end NUMINAMATH_CALUDE_sam_spent_pennies_l2144_214495


namespace NUMINAMATH_CALUDE_platform_crossing_time_l2144_214432

def train_speed : Real := 36  -- km/h
def time_to_cross_pole : Real := 12  -- seconds
def time_to_cross_platform : Real := 44.99736021118311  -- seconds

theorem platform_crossing_time :
  time_to_cross_platform = 44.99736021118311 :=
by sorry

end NUMINAMATH_CALUDE_platform_crossing_time_l2144_214432


namespace NUMINAMATH_CALUDE_tangent_line_determines_b_l2144_214423

/-- A curve defined by y = x³ + ax + b -/
def curve (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- The derivative of the curve -/
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- A line defined by y = mx + c -/
def line (m c : ℝ) (x : ℝ) : ℝ := m*x + c

theorem tangent_line_determines_b (a b : ℝ) :
  curve a b 1 = 3 ∧
  curve_derivative a 1 = 2 →
  b = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_determines_b_l2144_214423


namespace NUMINAMATH_CALUDE_base_2_representation_of_125_l2144_214466

theorem base_2_representation_of_125 :
  ∃ (b : List Bool), 
    (b.length = 7) ∧
    (b.foldl (λ acc x => 2 * acc + if x then 1 else 0) 0 = 125) ∧
    (b = [true, true, true, true, true, false, true]) := by
  sorry

end NUMINAMATH_CALUDE_base_2_representation_of_125_l2144_214466


namespace NUMINAMATH_CALUDE_work_fraction_proof_l2144_214483

theorem work_fraction_proof (total_payment : ℚ) (b_payment : ℚ) 
  (h1 : total_payment = 529)
  (h2 : b_payment = 12) :
  (total_payment - b_payment) / total_payment = 517 / 529 := by
  sorry

end NUMINAMATH_CALUDE_work_fraction_proof_l2144_214483


namespace NUMINAMATH_CALUDE_fastest_student_survey_method_l2144_214401

/-- Represents a survey method -/
inductive SurveyMethod
| Comprehensive
| Sample

/-- Represents a scenario requiring a survey -/
structure Scenario where
  description : String
  requiredMethod : SurveyMethod

/-- Represents the selection of the fastest student in a school's short-distance race -/
def fastestStudentSelection : Scenario :=
  { description := "Selecting the fastest student in a school's short-distance race",
    requiredMethod := SurveyMethod.Comprehensive }

/-- Theorem: The appropriate survey method for selecting the fastest student
    in a school's short-distance race is a comprehensive survey -/
theorem fastest_student_survey_method :
  fastestStudentSelection.requiredMethod = SurveyMethod.Comprehensive :=
by sorry


end NUMINAMATH_CALUDE_fastest_student_survey_method_l2144_214401


namespace NUMINAMATH_CALUDE_dutch_americans_window_seats_fraction_l2144_214465

/-- The fraction of Dutch Americans who got window seats on William's bus -/
theorem dutch_americans_window_seats_fraction
  (total_people : ℕ)
  (dutch_fraction : ℚ)
  (dutch_american_fraction : ℚ)
  (dutch_americans_with_window_seats : ℕ)
  (h1 : total_people = 90)
  (h2 : dutch_fraction = 3 / 5)
  (h3 : dutch_american_fraction = 1 / 2)
  (h4 : dutch_americans_with_window_seats = 9) :
  (dutch_americans_with_window_seats : ℚ) / (dutch_fraction * dutch_american_fraction * total_people) = 1 / 3 := by
  sorry

#check dutch_americans_window_seats_fraction

end NUMINAMATH_CALUDE_dutch_americans_window_seats_fraction_l2144_214465


namespace NUMINAMATH_CALUDE_inequality_proof_l2144_214491

theorem inequality_proof (n : ℕ) (x y z : ℝ) 
  (h_n : n ≥ 3) 
  (h_x : x > 0) (h_y : y > 0) (h_z : z > 0)
  (h_sum : x + y + z = 1) :
  (1 / x^(n-1) - x) * (1 / y^(n-1) - y) * (1 / z^(n-1) - z) ≥ ((3^n - 1) / 3)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2144_214491


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2144_214456

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_s

/-- Proof that a train's length is approximately 119.97 meters -/
theorem train_length_proof (speed_kmh : ℝ) (time_s : ℝ) 
  (h1 : speed_kmh = 48) 
  (h2 : time_s = 9) : 
  ∃ ε > 0, |train_length speed_kmh time_s - 119.97| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l2144_214456


namespace NUMINAMATH_CALUDE_variance_scaled_and_shifted_l2144_214451

variable {n : ℕ}
variable (x : Fin n → ℝ)

def variance (y : Fin n → ℝ) : ℝ := sorry

theorem variance_scaled_and_shifted
  (h : variance x = 1) :
  variance (fun i => 2 * x i + 1) = 4 := by sorry

end NUMINAMATH_CALUDE_variance_scaled_and_shifted_l2144_214451


namespace NUMINAMATH_CALUDE_abc_fraction_value_l2144_214488

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 2)
  (h2 : b * c / (b + c) = 5)
  (h3 : c * a / (c + a) = 9) :
  a * b * c / (a * b + b * c + c * a) = 90 / 73 := by
  sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l2144_214488


namespace NUMINAMATH_CALUDE_mikes_current_age_l2144_214406

theorem mikes_current_age :
  ∀ (M B : ℕ),
  B = M / 2 →
  M - B = 24 - 16 →
  M = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_mikes_current_age_l2144_214406


namespace NUMINAMATH_CALUDE_correct_operations_l2144_214489

theorem correct_operations (x y : ℝ) (h : x ≠ y) :
  ((-3 * x * y) ^ 2 = 9 * x^2 * y^2) ∧
  ((x - y) / (2 * x * y - x^2 - y^2) = 1 / (y - x)) := by
  sorry

end NUMINAMATH_CALUDE_correct_operations_l2144_214489


namespace NUMINAMATH_CALUDE_triangle_theorem_l2144_214433

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle)
  (h1 : t.a * Real.cos t.B + t.b * Real.cos t.A = 2 * Real.cos t.C)
  (h2 : t.a + t.b = 4)
  (h3 : t.c = 2) :
  t.C = Real.pi / 3 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2144_214433


namespace NUMINAMATH_CALUDE_fifth_power_complex_equality_l2144_214469

theorem fifth_power_complex_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + b * Complex.I) ^ 5 = (a - b * Complex.I) ^ 5) : 
  b / a = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_complex_equality_l2144_214469


namespace NUMINAMATH_CALUDE_count_integers_satisfying_equation_l2144_214454

def count_satisfying_integers (lower upper : ℕ) : ℕ :=
  (upper - lower + 1) / 4 + 1

theorem count_integers_satisfying_equation : 
  count_satisfying_integers 1 2002 = 501 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_equation_l2144_214454


namespace NUMINAMATH_CALUDE_A_union_B_eq_A_l2144_214424

def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {x | 0 < x ∧ x < Real.exp 1}

theorem A_union_B_eq_A : A ∪ B = A := by sorry

end NUMINAMATH_CALUDE_A_union_B_eq_A_l2144_214424


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l2144_214422

theorem quadratic_two_roots (a b c : ℝ) (ha : a ≠ 0) (hac : a * c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l2144_214422


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2144_214448

theorem container_volume_ratio :
  ∀ (volume_container1 volume_container2 : ℚ),
  volume_container1 > 0 →
  volume_container2 > 0 →
  (2 : ℚ) / 3 * volume_container1 = (1 : ℚ) / 2 * volume_container2 →
  volume_container1 / volume_container2 = (3 : ℚ) / 4 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2144_214448


namespace NUMINAMATH_CALUDE_function_domain_range_nonempty_function_range_determined_single_element_domain_range_l2144_214481

-- Define a function type
def Function (α β : Type) := α → β

-- Statement 1: The domain and range of a function are both non-empty sets
theorem function_domain_range_nonempty {α β : Type} (f : Function α β) :
  Nonempty α ∧ Nonempty β :=
sorry

-- Statement 2: Once the domain and the rule of correspondence are determined,
-- the range of the function is also determined
theorem function_range_determined {α β : Type} (f g : Function α β) :
  (∀ x : α, f x = g x) → Set.range f = Set.range g :=
sorry

-- Statement 3: If there is only one element in the domain of a function,
-- then there is also only one element in its range
theorem single_element_domain_range {α β : Type} (f : Function α β) :
  (∃! x : α, True) → (∃! y : β, ∃ x : α, f x = y) :=
sorry

end NUMINAMATH_CALUDE_function_domain_range_nonempty_function_range_determined_single_element_domain_range_l2144_214481


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2144_214409

def complex_i : ℂ := Complex.I

theorem z_in_first_quadrant (z : ℂ) (h : (1 + complex_i) * z = 1 - 2 * complex_i^3) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2144_214409


namespace NUMINAMATH_CALUDE_windows_preference_count_l2144_214439

theorem windows_preference_count (total : ℕ) (mac_pref : ℕ) (no_pref : ℕ) : 
  total = 210 → 
  mac_pref = 60 → 
  no_pref = 90 → 
  ∃ (windows_pref : ℕ), 
    windows_pref = total - (mac_pref + (mac_pref / 3) + no_pref) ∧ 
    windows_pref = 40 := by
  sorry

end NUMINAMATH_CALUDE_windows_preference_count_l2144_214439


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2144_214482

theorem arithmetic_expression_equality : 5 * 7 - 6 * 8 + 9 * 2 + 7 * 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2144_214482


namespace NUMINAMATH_CALUDE_cube_circumscribed_sphere_radius_l2144_214408

/-- The radius of the circumscribed sphere of a cube with edge length 1 is √3/2 -/
theorem cube_circumscribed_sphere_radius :
  let cube_edge_length : ℝ := 1
  let circumscribed_sphere_radius : ℝ := (Real.sqrt 3) / 2
  cube_edge_length = 1 →
  circumscribed_sphere_radius = (Real.sqrt 3) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_cube_circumscribed_sphere_radius_l2144_214408


namespace NUMINAMATH_CALUDE_max_value_of_f_in_interval_l2144_214472

def f (x : ℝ) : ℝ := x^2 + 3*x + 2

theorem max_value_of_f_in_interval :
  ∃ (M : ℝ), M = 42 ∧ 
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x ≤ M) ∧
  (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ f x = M) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_in_interval_l2144_214472


namespace NUMINAMATH_CALUDE_crab_fishing_income_l2144_214435

/-- Calculate weekly income from crab fishing --/
theorem crab_fishing_income 
  (num_buckets : ℕ) 
  (crabs_per_bucket : ℕ) 
  (price_per_crab : ℕ) 
  (days_per_week : ℕ) 
  (h1 : num_buckets = 8)
  (h2 : crabs_per_bucket = 12)
  (h3 : price_per_crab = 5)
  (h4 : days_per_week = 7) :
  num_buckets * crabs_per_bucket * price_per_crab * days_per_week = 3360 := by
  sorry

end NUMINAMATH_CALUDE_crab_fishing_income_l2144_214435


namespace NUMINAMATH_CALUDE_revenue_is_288_l2144_214441

/-- Represents the rental business with canoes and kayaks -/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  canoe_kayak_ratio : ℚ
  canoe_kayak_difference : ℕ

/-- Calculates the total revenue for a day given the rental business conditions -/
def calculate_revenue (business : RentalBusiness) : ℕ :=
  let kayaks := business.canoe_kayak_difference * 2
  let canoes := kayaks + business.canoe_kayak_difference
  kayaks * business.kayak_price + canoes * business.canoe_price

/-- Theorem stating that the total revenue for the day is $288 -/
theorem revenue_is_288 (business : RentalBusiness) 
    (h1 : business.canoe_price = 14)
    (h2 : business.kayak_price = 15)
    (h3 : business.canoe_kayak_ratio = 3 / 2)
    (h4 : business.canoe_kayak_difference = 4) :
  calculate_revenue business = 288 := by
  sorry

#eval calculate_revenue { 
  canoe_price := 14, 
  kayak_price := 15, 
  canoe_kayak_ratio := 3 / 2, 
  canoe_kayak_difference := 4 
}

end NUMINAMATH_CALUDE_revenue_is_288_l2144_214441


namespace NUMINAMATH_CALUDE_complex_sum_product_nonzero_l2144_214445

theorem complex_sum_product_nonzero (z₁ z₂ z₃ z₄ : ℂ) 
  (h₁ : Complex.abs z₁ = 1) (h₂ : Complex.abs z₂ = 1) 
  (h₃ : Complex.abs z₃ = 1) (h₄ : Complex.abs z₄ = 1)
  (n₁ : z₁ ≠ 1) (n₂ : z₂ ≠ 1) (n₃ : z₃ ≠ 1) (n₄ : z₄ ≠ 1) :
  3 - z₁ - z₂ - z₃ - z₄ + z₁ * z₂ * z₃ * z₄ ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_product_nonzero_l2144_214445


namespace NUMINAMATH_CALUDE_min_value_theorem_l2144_214494

/-- The hyperbola equation -/
def hyperbola (m n x y : ℝ) : Prop := x^2 / m - y^2 / n = 1

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The condition that the hyperbola and ellipse have the same foci -/
def same_foci (m n : ℝ) : Prop := m + n = 1

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_hyperbola : ∃ x y, hyperbola m n x y)
  (h_ellipse : ∃ x y, ellipse x y)
  (h_foci : same_foci m n) :
  (∀ m' n', m' > 0 → n' > 0 → same_foci m' n' → 4/m + 1/n ≤ 4/m' + 1/n') ∧ 
  (∃ m₀ n₀, m₀ > 0 ∧ n₀ > 0 ∧ same_foci m₀ n₀ ∧ 4/m₀ + 1/n₀ = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2144_214494


namespace NUMINAMATH_CALUDE_fraction_comparison_l2144_214461

theorem fraction_comparison : 
  let original := -15 / 12
  let a := -30 / 24
  let b := -1 - 3 / 12
  let c := -1 - 9 / 36
  let d := -1 - 5 / 15
  let e := -1 - 25 / 100
  (a = original ∧ b = original ∧ c = original ∧ e = original) ∧ d ≠ original :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2144_214461


namespace NUMINAMATH_CALUDE_fraction_equality_l2144_214485

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) :
  1 / 7 + (2 * q - p) / (2 * q + p) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2144_214485


namespace NUMINAMATH_CALUDE_unique_function_property_l2144_214426

theorem unique_function_property (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 0 ↔ x = 0)
  (h2 : ∀ x y, f (x^2 + y * f x) + f (y^2 + x * f y) = (f (x + y))^2) :
  ∀ x, f x = x := by
  sorry

end NUMINAMATH_CALUDE_unique_function_property_l2144_214426


namespace NUMINAMATH_CALUDE_symmetric_line_theorem_l2144_214471

/-- The equation of a line symmetric to another line with respect to a vertical line. -/
def symmetric_line_equation (a b c : ℝ) (k : ℝ) : ℝ → ℝ → Prop :=
  fun x y => 2 * (k - a) * x + b * y + (c - 2 * k * (k - a)) = 0

/-- The original line equation. -/
def original_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

/-- The line of symmetry. -/
def symmetry_line (x : ℝ) : Prop := x = 3

theorem symmetric_line_theorem :
  symmetric_line_equation 1 (-2) 1 3 = fun x y => 2 * x + y - 8 = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_theorem_l2144_214471


namespace NUMINAMATH_CALUDE_car_distance_covered_l2144_214459

/-- Proves that a car traveling at 97.5 km/h for 4 hours covers a distance of 390 km -/
theorem car_distance_covered (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 97.5 → time = 4 → distance = speed * time → distance = 390 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_covered_l2144_214459


namespace NUMINAMATH_CALUDE_train_length_l2144_214442

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 58 → time_s = 9 → 
  ∃ length_m : ℝ, abs (length_m - 144.99) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2144_214442


namespace NUMINAMATH_CALUDE_trig_simplification_l2144_214468

theorem trig_simplification :
  (Real.sin (35 * π / 180))^2 / Real.sin (20 * π / 180) - 1 / (2 * Real.sin (20 * π / 180)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2144_214468


namespace NUMINAMATH_CALUDE_graduating_class_size_l2144_214497

theorem graduating_class_size (boys : ℕ) (girls : ℕ) : 
  boys = 127 → 
  girls = boys + 212 → 
  boys + girls = 466 := by sorry

end NUMINAMATH_CALUDE_graduating_class_size_l2144_214497


namespace NUMINAMATH_CALUDE_chatterbox_jokes_l2144_214403

def n : ℕ := 10  -- number of chatterboxes

-- Sum of natural numbers from 1 to m
def sum_to(m : ℕ) : ℕ := m * (m + 1) / 2

-- Total number of jokes told
def total_jokes : ℕ := sum_to 100 + sum_to 99

theorem chatterbox_jokes :
  total_jokes / n = 1000 :=
sorry

end NUMINAMATH_CALUDE_chatterbox_jokes_l2144_214403


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l2144_214402

theorem sum_of_three_consecutive_cubes_divisible_by_nine (n : ℕ) (h : n > 1) :
  ∃ k : ℤ, (n - 1)^3 + n^3 + (n + 1)^3 = 9 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l2144_214402


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l2144_214400

/-- Given a circle and a point of symmetry, find the equation of the symmetric circle -/
theorem symmetric_circle_equation (x y : ℝ) :
  let given_circle := (x - 2)^2 + (y - 1)^2 = 1
  let point_of_symmetry := (1, 2)
  let symmetric_circle := x^2 + (y - 3)^2 = 1
  symmetric_circle = true := by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l2144_214400


namespace NUMINAMATH_CALUDE_guest_cars_count_l2144_214499

/-- Calculates the number of guest cars given the total number of wheels and parent cars -/
def guest_cars (total_wheels : ℕ) (parent_cars : ℕ) : ℕ :=
  (total_wheels - 4 * parent_cars) / 4

/-- Theorem: Given 48 total wheels and 2 parent cars, the number of guest cars is 10 -/
theorem guest_cars_count : guest_cars 48 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_guest_cars_count_l2144_214499


namespace NUMINAMATH_CALUDE_unique_zero_of_f_l2144_214479

noncomputable def f (x : ℝ) := 2^x + x^3 - 2

theorem unique_zero_of_f :
  ∃! x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_zero_of_f_l2144_214479


namespace NUMINAMATH_CALUDE_N_is_composite_l2144_214420

/-- N is defined as 7 × 9 × 13 + 2020 × 2018 × 2014 -/
def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

/-- Theorem stating that N is composite -/
theorem N_is_composite : ¬ Nat.Prime N := by sorry

end NUMINAMATH_CALUDE_N_is_composite_l2144_214420


namespace NUMINAMATH_CALUDE_bridge_anchor_ratio_l2144_214404

/-- Proves that the ratio of concrete needed for each bridge anchor is 1:1 --/
theorem bridge_anchor_ratio
  (total_concrete : ℕ)
  (roadway_concrete : ℕ)
  (one_anchor_concrete : ℕ)
  (pillar_concrete : ℕ)
  (h1 : total_concrete = 4800)
  (h2 : roadway_concrete = 1600)
  (h3 : one_anchor_concrete = 700)
  (h4 : pillar_concrete = 1800)
  : (one_anchor_concrete : ℚ) / one_anchor_concrete = 1 := by
  sorry

#check bridge_anchor_ratio

end NUMINAMATH_CALUDE_bridge_anchor_ratio_l2144_214404


namespace NUMINAMATH_CALUDE_target_probabilities_l2144_214478

/-- Probability of hitting a target -/
structure TargetProbability where
  prob : ℚ
  prob_nonneg : 0 ≤ prob
  prob_le_one : prob ≤ 1

/-- Model for the target shooting scenario -/
structure TargetScenario where
  A : TargetProbability
  B : TargetProbability

/-- Given scenario with person A and B's probabilities -/
def given_scenario : TargetScenario :=
  { A := { prob := 3/4, prob_nonneg := by norm_num, prob_le_one := by norm_num },
    B := { prob := 4/5, prob_nonneg := by norm_num, prob_le_one := by norm_num } }

/-- Probability that A hits and B misses after one shot each -/
def prob_A_hits_B_misses (s : TargetScenario) : ℚ :=
  s.A.prob * (1 - s.B.prob)

/-- Probability of k successes in n independent trials -/
def binomial_prob (p : ℚ) (n k : ℕ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1-p)^(n-k)

/-- Probability that A and B have equal hits after two shots each -/
def prob_equal_hits (s : TargetScenario) : ℚ :=
  (binomial_prob s.A.prob 2 0) * (binomial_prob s.B.prob 2 0) +
  (binomial_prob s.A.prob 2 1) * (binomial_prob s.B.prob 2 1) +
  (binomial_prob s.A.prob 2 2) * (binomial_prob s.B.prob 2 2)

theorem target_probabilities (s : TargetScenario := given_scenario) :
  (prob_A_hits_B_misses s = 3/20) ∧
  (prob_equal_hits s = 193/400) := by
  sorry

end NUMINAMATH_CALUDE_target_probabilities_l2144_214478


namespace NUMINAMATH_CALUDE_geometric_sequence_quadratic_one_root_l2144_214490

/-- If real numbers a, b, c form a geometric sequence, then the function f(x) = ax^2 + 2bx + c has exactly one real root. -/
theorem geometric_sequence_quadratic_one_root
  (a b c : ℝ) (h_geometric : b^2 = a*c) :
  ∃! x, a*x^2 + 2*b*x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_quadratic_one_root_l2144_214490


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2144_214477

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 400)
  (h2 : profit_percentage = 25) : 
  ∃ (cost_price : ℝ), 
    cost_price = 320 ∧ 
    selling_price = cost_price * (1 + profit_percentage / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2144_214477


namespace NUMINAMATH_CALUDE_vector_properties_l2144_214418

/-- Given points A and B in a 2D Cartesian coordinate system, prove properties of vectors AB and OA·OB --/
theorem vector_properties (A B : ℝ × ℝ) (h1 : A = (-3, -4)) (h2 : B = (5, -12)) :
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let OA : ℝ × ℝ := A
  let OB : ℝ × ℝ := B
  (AB = (8, -8)) ∧
  (Real.sqrt ((AB.1)^2 + (AB.2)^2) = 8 * Real.sqrt 2) ∧
  (OA.1 * OB.1 + OA.2 * OB.2 = 33) := by
sorry

end NUMINAMATH_CALUDE_vector_properties_l2144_214418


namespace NUMINAMATH_CALUDE_complex_power_sum_l2144_214450

theorem complex_power_sum (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 2)
  (h2 : α₁^2 + α₂^2 + α₃^2 = 5)
  (h3 : α₁^3 + α₂^3 + α₃^3 = 10) :
  α₁^6 + α₂^6 + α₃^6 = 44 := by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2144_214450


namespace NUMINAMATH_CALUDE_zhu_shijie_wine_problem_l2144_214443

/-- The amount of wine in the jug after visiting n taverns and meeting n friends -/
def wine_amount (initial : ℝ) (n : ℕ) : ℝ :=
  (2^n) * initial - (2^n - 1)

theorem zhu_shijie_wine_problem :
  ∃ (initial : ℝ), initial > 0 ∧ wine_amount initial 3 = 0 ∧ initial = 0.875 := by
  sorry

end NUMINAMATH_CALUDE_zhu_shijie_wine_problem_l2144_214443


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2144_214437

theorem contrapositive_equivalence (a b : ℝ) :
  (¬ (-Real.sqrt b < a ∧ a < Real.sqrt b) → ¬ (a^2 < b)) ↔
  ((a ≥ Real.sqrt b ∨ a ≤ -Real.sqrt b) → a^2 ≥ b) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2144_214437


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2144_214473

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → a 1 + a 2 = -1 → a 3 = 4 → a 4 + a 5 = 17 := by
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2144_214473


namespace NUMINAMATH_CALUDE_vector_c_determination_l2144_214440

/-- Given vectors a and b, if vector c satisfies the conditions, then c = (2, 1) -/
theorem vector_c_determination (a b c : ℝ × ℝ) 
  (ha : a = (1, -1)) 
  (hb : b = (1, 2)) 
  (hperp : (c.1 + b.1, c.2 + b.2) • a = 0)  -- (c + b) ⊥ a
  (hpar : ∃ k : ℝ, (c.1 - a.1, c.2 - a.2) = (k * b.1, k * b.2))  -- (c - a) ∥ b
  : c = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_c_determination_l2144_214440


namespace NUMINAMATH_CALUDE_matt_bike_ride_distance_l2144_214419

theorem matt_bike_ride_distance 
  (distance_to_first_sign : ℕ)
  (distance_between_signs : ℕ)
  (distance_after_second_sign : ℕ)
  (h1 : distance_to_first_sign = 350)
  (h2 : distance_between_signs = 375)
  (h3 : distance_after_second_sign = 275) :
  distance_to_first_sign + distance_between_signs + distance_after_second_sign = 1000 :=
by sorry

end NUMINAMATH_CALUDE_matt_bike_ride_distance_l2144_214419


namespace NUMINAMATH_CALUDE_number_properties_l2144_214464

def number : ℕ := 52300600

-- Define a function to get the digit at a specific position
def digit_at_position (n : ℕ) (pos : ℕ) : ℕ :=
  (n / (10 ^ (pos - 1))) % 10

-- Define a function to get the value represented by a digit at a specific position
def value_at_position (n : ℕ) (pos : ℕ) : ℕ :=
  (digit_at_position n pos) * (10 ^ (pos - 1))

-- Define a function to convert a number to its word representation
def number_to_words (n : ℕ) : String :=
  sorry -- Implementation details omitted

theorem number_properties :
  (digit_at_position number 8 = 2) ∧
  (value_at_position number 8 = 20000000) ∧
  (digit_at_position number 9 = 5) ∧
  (value_at_position number 9 = 500000000) ∧
  (number_to_words number = "five hundred twenty-three million six hundred") := by
  sorry

end NUMINAMATH_CALUDE_number_properties_l2144_214464


namespace NUMINAMATH_CALUDE_trapezoid_rhombus_properties_triangle_parallelogram_properties_rectangle_circle_symmetry_l2144_214427

-- Define the geometric shapes
class ConvexPolygon
class Polygon extends ConvexPolygon
class Trapezoid extends ConvexPolygon
class Rhombus extends ConvexPolygon
class Triangle extends Polygon
class Parallelogram extends Polygon
class Rectangle extends Polygon
class Circle

-- Define properties
def hasExteriorAngleSum360 (shape : Type) : Prop := sorry
def lineIntersectsTwice (shape : Type) : Prop := sorry
def hasCentralSymmetry (shape : Type) : Prop := sorry

-- Theorem statements
theorem trapezoid_rhombus_properties :
  (hasExteriorAngleSum360 Trapezoid ∧ hasExteriorAngleSum360 Rhombus) ∧
  (lineIntersectsTwice Trapezoid ∧ lineIntersectsTwice Rhombus) := by sorry

theorem triangle_parallelogram_properties :
  (hasExteriorAngleSum360 Triangle ∧ hasExteriorAngleSum360 Parallelogram) ∧
  (lineIntersectsTwice Triangle ∧ lineIntersectsTwice Parallelogram) := by sorry

theorem rectangle_circle_symmetry :
  hasCentralSymmetry Rectangle ∧ hasCentralSymmetry Circle := by sorry

end NUMINAMATH_CALUDE_trapezoid_rhombus_properties_triangle_parallelogram_properties_rectangle_circle_symmetry_l2144_214427


namespace NUMINAMATH_CALUDE_cos_15_cos_45_minus_cos_75_sin_45_l2144_214447

theorem cos_15_cos_45_minus_cos_75_sin_45 :
  Real.cos (15 * π / 180) * Real.cos (45 * π / 180) - 
  Real.cos (75 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_15_cos_45_minus_cos_75_sin_45_l2144_214447


namespace NUMINAMATH_CALUDE_width_to_perimeter_ratio_l2144_214476

/-- Represents a rectangular classroom -/
structure Classroom where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a classroom -/
def perimeter (c : Classroom) : ℝ := 2 * (c.length + c.width)

/-- Theorem: The ratio of width to perimeter for a 15x10 classroom is 1:5 -/
theorem width_to_perimeter_ratio (c : Classroom) 
  (h1 : c.length = 15) 
  (h2 : c.width = 10) : 
  c.width / perimeter c = 1 / 5 := by
  sorry

#check width_to_perimeter_ratio

end NUMINAMATH_CALUDE_width_to_perimeter_ratio_l2144_214476


namespace NUMINAMATH_CALUDE_statements_classification_correct_l2144_214463

-- Define the type of statement
inductive StatementType
  | Universal
  | Existential

-- Define a structure to represent a statement
structure Statement where
  text : String
  type : StatementType
  isTrue : Bool

-- Define the four statements
def statement1 : Statement :=
  { text := "The diagonals of a square are perpendicular bisectors of each other"
  , type := StatementType.Universal
  , isTrue := true }

def statement2 : Statement :=
  { text := "All Chinese people speak Chinese"
  , type := StatementType.Universal
  , isTrue := false }

def statement3 : Statement :=
  { text := "Some numbers are greater than their squares"
  , type := StatementType.Existential
  , isTrue := true }

def statement4 : Statement :=
  { text := "Some real numbers have irrational square roots"
  , type := StatementType.Existential
  , isTrue := true }

-- Theorem to prove the correctness of the statements' classifications
theorem statements_classification_correct :
  statement1.type = StatementType.Universal ∧ statement1.isTrue = true ∧
  statement2.type = StatementType.Universal ∧ statement2.isTrue = false ∧
  statement3.type = StatementType.Existential ∧ statement3.isTrue = true ∧
  statement4.type = StatementType.Existential ∧ statement4.isTrue = true :=
by sorry

end NUMINAMATH_CALUDE_statements_classification_correct_l2144_214463


namespace NUMINAMATH_CALUDE_system_solution_arithmetic_progression_l2144_214415

/-- 
Given a system of equations:
  x + y + m*z = a
  x + m*y + z = b
  m*x + y + z = c
This theorem states that for m ≠ 1 and m ≠ -2, the system has a unique solution (x, y, z) 
in arithmetic progression if and only if a, b, c are in arithmetic progression.
-/
theorem system_solution_arithmetic_progression 
  (m a b c : ℝ) (hm1 : m ≠ 1) (hm2 : m ≠ -2) :
  (∃! x y z : ℝ, x + y + m*z = a ∧ x + m*y + z = b ∧ m*x + y + z = c ∧ 
   2*y = x + z) ↔ 2*b = a + c :=
sorry

end NUMINAMATH_CALUDE_system_solution_arithmetic_progression_l2144_214415


namespace NUMINAMATH_CALUDE_sixth_term_term_1994_l2144_214430

-- Define the sequence
def a (n : ℕ) : ℕ := n * (n + 1)

-- Theorem for the 6th term
theorem sixth_term : a 6 = 42 := by sorry

-- Theorem for the 1994th term
theorem term_1994 : a 1994 = 3978030 := by sorry

end NUMINAMATH_CALUDE_sixth_term_term_1994_l2144_214430


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l2144_214412

theorem solution_set_abs_inequality (x : ℝ) :
  |2 - x| ≥ 1 ↔ x ≤ 1 ∨ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l2144_214412


namespace NUMINAMATH_CALUDE_jenny_lasagna_profit_l2144_214474

/-- Calculates Jenny's profit from selling lasagna pans -/
def jennys_profit (cost_per_pan : ℝ) (num_pans : ℕ) (price_per_pan : ℝ) : ℝ :=
  (price_per_pan * num_pans) - (cost_per_pan * num_pans)

theorem jenny_lasagna_profit :
  let cost_per_pan : ℝ := 10
  let num_pans : ℕ := 20
  let price_per_pan : ℝ := 25
  jennys_profit cost_per_pan num_pans price_per_pan = 300 := by
  sorry

end NUMINAMATH_CALUDE_jenny_lasagna_profit_l2144_214474


namespace NUMINAMATH_CALUDE_pictures_in_first_album_l2144_214484

def total_pictures : ℕ := 25
def num_other_albums : ℕ := 5
def pics_per_other_album : ℕ := 3

theorem pictures_in_first_album :
  total_pictures - (num_other_albums * pics_per_other_album) = 10 := by
  sorry

end NUMINAMATH_CALUDE_pictures_in_first_album_l2144_214484


namespace NUMINAMATH_CALUDE_vendor_apples_thrown_away_l2144_214486

/-- Calculates the percentage of apples thrown away given the initial quantity and selling/discarding percentages --/
def apples_thrown_away (initial_quantity : ℕ) (sell_day1 sell_day2 discard_day1 : ℚ) : ℚ :=
  let remaining_after_sell1 := initial_quantity * (1 - sell_day1)
  let discarded_day1 := remaining_after_sell1 * discard_day1
  let remaining_after_discard1 := remaining_after_sell1 - discarded_day1
  let remaining_after_sell2 := remaining_after_discard1 * (1 - sell_day2)
  (discarded_day1 + remaining_after_sell2) / initial_quantity * 100

theorem vendor_apples_thrown_away :
  apples_thrown_away 100 (30/100) (50/100) (20/100) = 42 :=
by sorry

end NUMINAMATH_CALUDE_vendor_apples_thrown_away_l2144_214486


namespace NUMINAMATH_CALUDE_cars_meeting_time_l2144_214416

/-- The time when two cars meet on a highway -/
theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) :
  highway_length = 600 →
  speed1 = 65 →
  speed2 = 75 →
  (highway_length / (speed1 + speed2) : ℝ) = 30 / 7 :=
by sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l2144_214416


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l2144_214453

theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∃ (r s : ℤ), x^2 + b*x + 2008 = (x + r) * (x + s)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬∃ (r s : ℤ), x^2 + b'*x + 2008 = (x + r) * (x + s)) ∧
  b = 259 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l2144_214453


namespace NUMINAMATH_CALUDE_problem_solution_l2144_214414

theorem problem_solution :
  ∀ (a b c : ℕ+) (x y z : ℤ),
    x = -2272 →
    y = 1000 + 100 * c.val + 10 * b.val + a.val →
    z = 1 →
    a.val * x + b.val * y + c.val * z = 1 →
    a < b →
    b < c →
    y = 1987 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2144_214414


namespace NUMINAMATH_CALUDE_point_coordinates_l2144_214431

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

def distance_to_x_axis (y : ℝ) : ℝ := |y|

def distance_to_y_axis (x : ℝ) : ℝ := |x|

theorem point_coordinates :
  ∀ (x y : ℝ),
    fourth_quadrant x y →
    distance_to_x_axis y = 2 →
    distance_to_y_axis x = 4 →
    x = 4 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2144_214431


namespace NUMINAMATH_CALUDE_k_condition_necessary_not_sufficient_l2144_214493

/-- Defines the condition for k -/
def k_condition (k : ℝ) : Prop := 7 < k ∧ k < 9

/-- Defines the equation of the conic section -/
def is_conic_equation (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (9 - k) + y^2 / (k - 7) = 1

/-- Defines the conditions for the equation to represent an ellipse -/
def is_ellipse_equation (k : ℝ) : Prop :=
  9 - k > 0 ∧ k - 7 > 0 ∧ 9 - k ≠ k - 7

/-- Theorem stating that k_condition is necessary but not sufficient for is_ellipse_equation -/
theorem k_condition_necessary_not_sufficient :
  (∀ k, is_ellipse_equation k → k_condition k) ∧
  ¬(∀ k, k_condition k → is_ellipse_equation k) :=
sorry

end NUMINAMATH_CALUDE_k_condition_necessary_not_sufficient_l2144_214493


namespace NUMINAMATH_CALUDE_binomial_coefficient_1450_2_l2144_214480

theorem binomial_coefficient_1450_2 : Nat.choose 1450 2 = 1050205 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1450_2_l2144_214480


namespace NUMINAMATH_CALUDE_triangle_area_l2144_214407

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b^2 + c^2 = a^2 + b*c →
  b * c * Real.cos A = 4 →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l2144_214407


namespace NUMINAMATH_CALUDE_equation_has_real_root_l2144_214421

theorem equation_has_real_root (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l2144_214421


namespace NUMINAMATH_CALUDE_margaret_swimming_time_l2144_214444

/-- Billy's swimming times for different parts of the race in seconds -/
def billy_times : List ℕ := [120, 240, 60, 150]

/-- The time difference between Billy and Margaret in seconds -/
def time_difference : ℕ := 30

/-- Calculate the total time Billy spent swimming -/
def billy_total_time : ℕ := billy_times.sum

/-- Calculate Margaret's total swimming time in seconds -/
def margaret_time_seconds : ℕ := billy_total_time + time_difference

/-- Convert seconds to minutes -/
def seconds_to_minutes (seconds : ℕ) : ℕ := seconds / 60

theorem margaret_swimming_time :
  seconds_to_minutes margaret_time_seconds = 10 := by
  sorry

end NUMINAMATH_CALUDE_margaret_swimming_time_l2144_214444


namespace NUMINAMATH_CALUDE_intersection_M_N_l2144_214411

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2144_214411


namespace NUMINAMATH_CALUDE_product_is_square_l2144_214467

theorem product_is_square (g : ℕ) (h : g = 14) : ∃ n : ℕ, 3150 * g = n^2 := by
  sorry

end NUMINAMATH_CALUDE_product_is_square_l2144_214467


namespace NUMINAMATH_CALUDE_intersection_M_N_l2144_214436

def M : Set ℝ := {-1, 1, 2, 4}
def N : Set ℝ := {x | x > 2}

theorem intersection_M_N : M ∩ N = {4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2144_214436


namespace NUMINAMATH_CALUDE_range_of_a_l2144_214428

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 1 → x^2 + 2*x - a > 0) → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2144_214428


namespace NUMINAMATH_CALUDE_complement_intersection_subset_range_l2144_214452

-- Define the sets A and B
def A : Set ℝ := {x | 2 < x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

-- Theorem 1: Prove the complement intersection when a = 2
theorem complement_intersection :
  (Set.univ \ A) ∩ (Set.univ \ B 2) = {x | x ≤ 1 ∨ x > 5} := by sorry

-- Theorem 2: Prove the range of a for which B is a subset of A
theorem subset_range :
  {a : ℝ | B a ⊆ A} = {a | 3 ≤ a ∧ a ≤ 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_subset_range_l2144_214452


namespace NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l2144_214462

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, (f x - f y)^2 ≤ |x - y|^3) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l2144_214462


namespace NUMINAMATH_CALUDE_james_car_transaction_l2144_214458

/-- The amount James is out of pocket after selling his old car and buying a new one -/
def out_of_pocket (old_car_value : ℝ) (old_car_sale_percentage : ℝ) 
                  (new_car_sticker : ℝ) (new_car_buy_percentage : ℝ) : ℝ :=
  new_car_sticker * new_car_buy_percentage - old_car_value * old_car_sale_percentage

/-- Theorem stating that James is out of pocket by $11,000 -/
theorem james_car_transaction : 
  out_of_pocket 20000 0.8 30000 0.9 = 11000 := by
  sorry

end NUMINAMATH_CALUDE_james_car_transaction_l2144_214458


namespace NUMINAMATH_CALUDE_race_time_proof_l2144_214434

/-- 
Proves that in a 1000-meter race where runner A finishes 200 meters ahead of runner B, 
and the time difference between their finishes is 10 seconds, 
the time taken by runner A to complete the race is 50 seconds.
-/
theorem race_time_proof (race_length : ℝ) (distance_diff : ℝ) (time_diff : ℝ) 
  (h1 : race_length = 1000)
  (h2 : distance_diff = 200)
  (h3 : time_diff = 10) : 
  ∃ (time_A : ℝ), time_A = 50 ∧ 
    race_length / time_A = (race_length - distance_diff) / (time_A + time_diff) :=
by
  sorry

#check race_time_proof

end NUMINAMATH_CALUDE_race_time_proof_l2144_214434


namespace NUMINAMATH_CALUDE_largest_number_l2144_214457

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 997/1000) 
  (hb : b = 9799/10000) 
  (hc : c = 999/1000) 
  (hd : d = 9979/10000) 
  (he : e = 979/1000) : 
  c = max a (max b (max c (max d e))) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l2144_214457


namespace NUMINAMATH_CALUDE_pirate_coins_l2144_214429

theorem pirate_coins (x : ℚ) : 
  (3/7 * x + 0.51 * (4/7 * x) = (2.04/7) * x) →
  ((2.04/7) * x - (1.96/7) * x = 8) →
  x = 700 :=
by sorry

end NUMINAMATH_CALUDE_pirate_coins_l2144_214429


namespace NUMINAMATH_CALUDE_organize_four_men_five_women_l2144_214413

/-- The number of ways to organize men and women into groups -/
def organize_groups (num_men : ℕ) (num_women : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of ways to organize the groups -/
theorem organize_four_men_five_women :
  organize_groups 4 5 = 180 := by
  sorry

end NUMINAMATH_CALUDE_organize_four_men_five_women_l2144_214413


namespace NUMINAMATH_CALUDE_exponent_division_l2144_214487

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^8 / a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2144_214487


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_when_divided_by_8_l2144_214425

theorem largest_integer_less_than_100_remainder_5_when_divided_by_8 : 
  ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 → m % 8 = 5 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_when_divided_by_8_l2144_214425


namespace NUMINAMATH_CALUDE_cube_probability_l2144_214492

/-- A cube with side length 3 -/
def Cube := Fin 3 → Fin 3 → Fin 3

/-- The number of unit cubes in the larger cube -/
def totalCubes : ℕ := 27

/-- The number of unit cubes with exactly two painted faces -/
def twoPaintedFaces : ℕ := 4

/-- The number of unit cubes with no painted faces -/
def noPaintedFaces : ℕ := 8

/-- The probability of selecting one cube with two painted faces and one with no painted faces -/
def probability : ℚ := 32 / 351

theorem cube_probability : 
  probability = (twoPaintedFaces * noPaintedFaces : ℚ) / (totalCubes.choose 2) := by sorry

end NUMINAMATH_CALUDE_cube_probability_l2144_214492


namespace NUMINAMATH_CALUDE_completing_square_equiv_l2144_214455

theorem completing_square_equiv (x : ℝ) : 
  x^2 - 4*x + 1 = 0 ↔ (x - 2)^2 = 3 := by sorry

end NUMINAMATH_CALUDE_completing_square_equiv_l2144_214455


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2144_214405

open Real

theorem trigonometric_identities (α : ℝ) 
  (h : (sin α - 2 * cos α) / (sin α + 2 * cos α) = 3) : 
  ((sin α + 2 * cos α) / (5 * cos α - sin α) = -2/9) ∧
  ((sin α + cos α)^2 = 9/17) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2144_214405


namespace NUMINAMATH_CALUDE_supermarket_profit_l2144_214498

/-- Represents the daily sales quantity as a function of the selling price. -/
def sales_quantity (x : ℤ) : ℤ := -5 * x + 150

/-- Represents the daily profit as a function of the selling price. -/
def daily_profit (x : ℤ) : ℤ := (x - 8) * (sales_quantity x)

theorem supermarket_profit (x : ℤ) (h1 : 8 ≤ x) (h2 : x ≤ 15) :
  (daily_profit 14 = 480) ∧
  (∀ y : ℤ, 8 ≤ y → y ≤ 15 → daily_profit y ≤ daily_profit 15) ∧
  (daily_profit 15 = 525) :=
sorry


end NUMINAMATH_CALUDE_supermarket_profit_l2144_214498


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2144_214475

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    if the length of the minor axis is equal to the focal length,
    then the eccentricity of the ellipse is √2/2 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let c := Real.sqrt (a^2 - b^2)
  2 * b = 2 * c → Real.sqrt ((a^2 - b^2) / a^2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2144_214475


namespace NUMINAMATH_CALUDE_max_primitive_dinosaur_cells_l2144_214446

/-- Represents a dinosaur as a tree -/
structure Dinosaur where
  cells : ℕ
  is_connected : Bool
  max_degree : ℕ

/-- Defines a primitive dinosaur -/
def is_primitive (d : Dinosaur) : Prop :=
  ∀ (d1 d2 : Dinosaur), d.cells ≠ d1.cells + d2.cells ∨ d1.cells < 2007 ∨ d2.cells < 2007

/-- The main theorem stating the maximum number of cells in a primitive dinosaur -/
theorem max_primitive_dinosaur_cells :
  ∀ (d : Dinosaur),
    d.cells ≥ 2007 →
    d.is_connected = true →
    d.max_degree = 4 →
    is_primitive d →
    d.cells ≤ 8025 :=
sorry

end NUMINAMATH_CALUDE_max_primitive_dinosaur_cells_l2144_214446


namespace NUMINAMATH_CALUDE_sideline_time_l2144_214470

def game_duration : ℕ := 90
def first_play_time : ℕ := 20
def second_play_time : ℕ := 35

theorem sideline_time :
  game_duration - (first_play_time + second_play_time) = 35 := by
  sorry

end NUMINAMATH_CALUDE_sideline_time_l2144_214470


namespace NUMINAMATH_CALUDE_square_rectangle_area_difference_l2144_214438

theorem square_rectangle_area_difference :
  let square_side : ℝ := 2
  let rect_length : ℝ := 2
  let rect_width : ℝ := 2
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  square_area - rect_area = 0 := by
sorry

end NUMINAMATH_CALUDE_square_rectangle_area_difference_l2144_214438


namespace NUMINAMATH_CALUDE_hairstylist_normal_haircut_price_l2144_214496

theorem hairstylist_normal_haircut_price :
  let normal_price : ℝ := x
  let special_price : ℝ := 6
  let trendy_price : ℝ := 8
  let normal_per_day : ℕ := 5
  let special_per_day : ℕ := 3
  let trendy_per_day : ℕ := 2
  let days_per_week : ℕ := 7
  let weekly_earnings : ℝ := 413
  (normal_price * (normal_per_day * days_per_week : ℝ) +
   special_price * (special_per_day * days_per_week : ℝ) +
   trendy_price * (trendy_per_day * days_per_week : ℝ) = weekly_earnings) →
  normal_price = 5 := by
sorry

end NUMINAMATH_CALUDE_hairstylist_normal_haircut_price_l2144_214496


namespace NUMINAMATH_CALUDE_angle_MDN_is_acute_l2144_214417

/-- The parabola y^2 = 2x -/
def parabola (x y : ℝ) : Prop := y^2 = 2*x

/-- A line passing through point (2,0) -/
def line_through_P (k : ℝ) (x y : ℝ) : Prop := x = k*y + 2

/-- The vertical line x = -1/2 -/
def vertical_line (x : ℝ) : Prop := x = -1/2

/-- The dot product of two 2D vectors -/
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

theorem angle_MDN_is_acute (k t : ℝ) (xM yM xN yN : ℝ) :
  parabola xM yM →
  parabola xN yN →
  line_through_P k xM yM →
  line_through_P k xN yN →
  vertical_line (-1/2) →
  xM ≠ xN ∨ yM ≠ yN →
  dot_product (xM + 1/2) (yM - t) (xN + 1/2) (yN - t) > 0 :=
sorry

end NUMINAMATH_CALUDE_angle_MDN_is_acute_l2144_214417


namespace NUMINAMATH_CALUDE_digital_earth_capabilities_l2144_214449

structure DigitalEarth where
  simulate_environment : Bool
  monitor_crops : Bool
  predict_submersion : Bool
  simulate_past : Bool
  predict_future : Bool

theorem digital_earth_capabilities (de : DigitalEarth) :
  de.simulate_environment ∧
  de.monitor_crops ∧
  de.predict_submersion ∧
  de.simulate_past →
  ¬ de.predict_future :=
by sorry

end NUMINAMATH_CALUDE_digital_earth_capabilities_l2144_214449


namespace NUMINAMATH_CALUDE_inequality_proof_l2144_214460

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * Real.log x + a * x^2 + 1

theorem inequality_proof (a : ℝ) (h : a ≤ -2) :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → (f a x₁ - f a x₂) / (x₂ - x₁) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2144_214460


namespace NUMINAMATH_CALUDE_equal_sides_from_tangent_sum_l2144_214410

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively
  (sum_angles : A + B + C = π)  -- Sum of angles in a triangle is π
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)  -- Sides are positive

-- State the theorem
theorem equal_sides_from_tangent_sum (t : Triangle) :
  t.a * Real.tan t.A + t.b * Real.tan t.B = (t.a + t.b) * Real.tan ((t.A + t.B) / 2) →
  t.a = t.b :=
by sorry

end NUMINAMATH_CALUDE_equal_sides_from_tangent_sum_l2144_214410
