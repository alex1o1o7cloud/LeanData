import Mathlib

namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l611_61155

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (75/23, -64/23)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 8 * x - 5 * y = 40

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 14

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations : 
  line1 intersection_point.1 intersection_point.2 ∧ 
  line2 intersection_point.1 intersection_point.2 := by
  sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem intersection_point_unique (x y : ℚ) : 
  line1 x y ∧ line2 x y → (x, y) = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l611_61155


namespace NUMINAMATH_CALUDE_firework_explosion_velocity_firework_explosion_velocity_is_correct_l611_61177

/-- The magnitude of the second fragment's velocity after a firework explosion -/
theorem firework_explosion_velocity : ℝ :=
  let initial_velocity : ℝ := 20
  let gravity : ℝ := 10
  let explosion_time : ℝ := 1
  let mass_ratio : ℝ := 2
  let small_fragment_horizontal_velocity : ℝ := 16

  let velocity_at_explosion : ℝ := initial_velocity - gravity * explosion_time
  let small_fragment_mass : ℝ := 1
  let large_fragment_mass : ℝ := mass_ratio * small_fragment_mass

  let small_fragment_vertical_velocity : ℝ := velocity_at_explosion
  let large_fragment_horizontal_velocity : ℝ := 
    -(small_fragment_mass * small_fragment_horizontal_velocity) / large_fragment_mass
  let large_fragment_vertical_velocity : ℝ := velocity_at_explosion

  let large_fragment_velocity_magnitude : ℝ := 
    Real.sqrt (large_fragment_horizontal_velocity^2 + large_fragment_vertical_velocity^2)

  2 * Real.sqrt 41

theorem firework_explosion_velocity_is_correct : 
  firework_explosion_velocity = 2 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_firework_explosion_velocity_firework_explosion_velocity_is_correct_l611_61177


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l611_61169

theorem smallest_n_congruence : ∃! n : ℕ+, (∀ m : ℕ+, 5 * m ≡ 220 [MOD 26] → n ≤ m) ∧ 5 * n ≡ 220 [MOD 26] := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l611_61169


namespace NUMINAMATH_CALUDE_sum_digits_base8_888_l611_61164

/-- Converts a natural number to its base 8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.sum

theorem sum_digits_base8_888 : sumDigits (toBase8 888) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_base8_888_l611_61164


namespace NUMINAMATH_CALUDE_max_value_sine_function_l611_61178

theorem max_value_sine_function (x : Real) (h : -π/2 ≤ x ∧ x ≤ 0) :
  ∃ y_max : Real, y_max = 2 ∧ ∀ y : Real, y = 3 * Real.sin x + 2 → y ≤ y_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sine_function_l611_61178


namespace NUMINAMATH_CALUDE_intersecting_circles_sum_l611_61137

/-- Given two circles intersecting at points A and B, with their centers on a line,
    prove that m+2c equals 26 -/
theorem intersecting_circles_sum (m c : ℝ) : 
  let A : ℝ × ℝ := (-1, 3)
  let B : ℝ × ℝ := (-6, m)
  let center_line (x y : ℝ) := x - y + c = 0
  -- Assume the centers of both circles lie on the line x - y + c = 0
  -- Assume A and B are intersection points of the two circles
  m + 2*c = 26 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_sum_l611_61137


namespace NUMINAMATH_CALUDE_outfit_combinations_l611_61162

theorem outfit_combinations : 
  let total_items : ℕ := 3  -- shirts, pants, hats
  let colors_per_item : ℕ := 5
  let total_combinations := colors_per_item ^ total_items
  let same_color_combinations := 
    (total_items * colors_per_item * (colors_per_item - 1)) + colors_per_item
  total_combinations - same_color_combinations = 60 :=
by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l611_61162


namespace NUMINAMATH_CALUDE_planes_parallel_transitivity_l611_61129

-- Define non-coincident planes
variable (α β γ : Plane)
variable (h_distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- Define parallel relation
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_transitivity 
  (h1 : parallel α γ) 
  (h2 : parallel β γ) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_transitivity_l611_61129


namespace NUMINAMATH_CALUDE_person_a_work_time_l611_61130

theorem person_a_work_time (person_b_time : ℝ) (combined_work_rate : ℝ) (combined_work_time : ℝ) :
  person_b_time = 45 →
  combined_work_rate = 2/9 →
  combined_work_time = 4 →
  ∃ person_a_time : ℝ,
    person_a_time = 30 ∧
    combined_work_rate = combined_work_time * (1 / person_a_time + 1 / person_b_time) :=
by sorry

end NUMINAMATH_CALUDE_person_a_work_time_l611_61130


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l611_61174

theorem polynomial_coefficient_sum (a₄ a₃ a₂ a₁ a : ℝ) :
  (∀ x : ℝ, a₄ * (x + 1)^4 + a₃ * (x + 1)^3 + a₂ * (x + 1)^2 + a₁ * (x + 1) + a = x^4) →
  a₃ - a₂ + a₁ = -14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l611_61174


namespace NUMINAMATH_CALUDE_turquoise_survey_result_l611_61193

/-- Represents the survey about turquoise color perception -/
structure TurquoiseSurvey where
  total : ℕ
  blue : ℕ
  both : ℕ
  neither : ℕ

/-- The number of people who believe turquoise is "green-ish" -/
def green_count (s : TurquoiseSurvey) : ℕ :=
  s.total - (s.blue - s.both) - s.both - s.neither

/-- Theorem stating the result of the survey -/
theorem turquoise_survey_result (s : TurquoiseSurvey) 
  (h1 : s.total = 150)
  (h2 : s.blue = 90)
  (h3 : s.both = 40)
  (h4 : s.neither = 30) :
  green_count s = 70 := by
  sorry

#eval green_count ⟨150, 90, 40, 30⟩

end NUMINAMATH_CALUDE_turquoise_survey_result_l611_61193


namespace NUMINAMATH_CALUDE_box_length_proof_l611_61149

theorem box_length_proof (x : ℕ) (cube_side : ℕ) : 
  (x * 48 * 12 = 80 * cube_side^3) → 
  (x % cube_side = 0) → 
  (48 % cube_side = 0) → 
  (12 % cube_side = 0) → 
  x = 240 := by
sorry

end NUMINAMATH_CALUDE_box_length_proof_l611_61149


namespace NUMINAMATH_CALUDE_dot_product_equals_negative_31_l611_61146

def vector1 : Fin 2 → ℝ
  | 0 => -3
  | 1 => 2

def vector2 : Fin 2 → ℝ
  | 0 => 7
  | 1 => -5

theorem dot_product_equals_negative_31 :
  (Finset.univ.sum fun i => vector1 i * vector2 i) = -31 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_equals_negative_31_l611_61146


namespace NUMINAMATH_CALUDE_ellipse_slope_at_pi_third_l611_61141

/-- Given an ellipse with parametric equations x = 2cos(t) and y = 4sin(t),
    prove that the slope of the line OM, where M is the point on the ellipse
    corresponding to t = π/3 and O is the origin, is equal to 2√3. -/
theorem ellipse_slope_at_pi_third :
  let x : ℝ → ℝ := λ t ↦ 2 * Real.cos t
  let y : ℝ → ℝ := λ t ↦ 4 * Real.sin t
  let M : ℝ × ℝ := (x (π/3), y (π/3))
  let O : ℝ × ℝ := (0, 0)
  let slope := (M.2 - O.2) / (M.1 - O.1)
  slope = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_slope_at_pi_third_l611_61141


namespace NUMINAMATH_CALUDE_rent_to_expenses_ratio_l611_61188

/-- Given Kathryn's monthly finances, prove the ratio of rent to food and travel expenses -/
theorem rent_to_expenses_ratio 
  (rent : ℕ) 
  (salary : ℕ) 
  (remaining : ℕ) 
  (h1 : rent = 1200)
  (h2 : salary = 5000)
  (h3 : remaining = 2000) :
  (rent : ℚ) / ((salary - remaining) - rent) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rent_to_expenses_ratio_l611_61188


namespace NUMINAMATH_CALUDE_triangle_polynomial_roots_l611_61126

theorem triangle_polynomial_roots (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hac : a + c > b) (hbc : b + c > a) :
  ¬ (∃ x y : ℝ, x < 1/3 ∧ y < 1/3 ∧ a * x^2 - b * x + c = 0 ∧ a * y^2 - b * y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_polynomial_roots_l611_61126


namespace NUMINAMATH_CALUDE_smallest_argument_complex_l611_61111

/-- The complex number with the smallest argument satisfying |p - 25i| ≤ 15 -/
def smallest_arg_complex : ℂ := 12 + 16 * Complex.I

/-- The condition that complex numbers must satisfy -/
def satisfies_condition (p : ℂ) : Prop :=
  Complex.abs (p - 25 * Complex.I) ≤ 15

theorem smallest_argument_complex :
  satisfies_condition smallest_arg_complex ∧
  ∀ p : ℂ, satisfies_condition p → Complex.arg smallest_arg_complex ≤ Complex.arg p :=
by sorry

end NUMINAMATH_CALUDE_smallest_argument_complex_l611_61111


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l611_61104

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = 5 ∧ ∀ y ∈ Set.Icc 0 3, f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l611_61104


namespace NUMINAMATH_CALUDE_derivative_e_cubed_l611_61112

-- e is the base of the natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- Statement: The derivative of e^3 is e^3
theorem derivative_e_cubed : 
  deriv (fun x : ℝ => e^3) = fun x : ℝ => e^3 :=
sorry

end NUMINAMATH_CALUDE_derivative_e_cubed_l611_61112


namespace NUMINAMATH_CALUDE_average_tv_watching_three_weeks_l611_61159

def tv_watching (week1 week2 week3 : ℕ) : ℕ := week1 + week2 + week3

def average_tv_watching (total_hours num_weeks : ℕ) : ℚ :=
  (total_hours : ℚ) / (num_weeks : ℚ)

theorem average_tv_watching_three_weeks :
  let week1 : ℕ := 10
  let week2 : ℕ := 8
  let week3 : ℕ := 12
  let total_hours := tv_watching week1 week2 week3
  let num_weeks : ℕ := 3
  average_tv_watching total_hours num_weeks = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_tv_watching_three_weeks_l611_61159


namespace NUMINAMATH_CALUDE_max_negative_integers_l611_61127

theorem max_negative_integers
  (a b c d e f : ℤ)
  (h : a * b + c * d * e * f < 0) :
  ∃ (neg_count : ℕ),
    neg_count ≤ 4 ∧
    ∀ (n : ℕ),
      (∃ (neg_set : Finset ℤ),
        neg_set.card = n ∧
        neg_set ⊆ {a, b, c, d, e, f} ∧
        (∀ x ∈ neg_set, x < 0) ∧
        (∀ x ∈ {a, b, c, d, e, f} \ neg_set, x ≥ 0)) →
      n ≤ neg_count :=
sorry

end NUMINAMATH_CALUDE_max_negative_integers_l611_61127


namespace NUMINAMATH_CALUDE_simplify_expression_l611_61167

theorem simplify_expression
  (x m n : ℝ)
  (h_m : m ≠ 0)
  (h_n : n ≠ 0)
  (h_x_pos : x > 0)
  (h_x_neq : x ≠ 3^(m * n / (m - n))) :
  (x^(2/m) - 9*x^(2/n)) * (x^((1-m)/m) - 3*x^((1-n)/n)) /
  ((x^(1/m) + 3*x^(1/n))^2 - 12*x^((m+n)/(m*n))) =
  (x^(1/m) + 3*x^(1/n)) / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l611_61167


namespace NUMINAMATH_CALUDE_sum_of_integers_l611_61128

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + r = 7)
  (eq2 : q - r + s = 8)
  (eq3 : r - s + p = 4)
  (eq4 : s - p + q = 1) :
  p + q + r + s = 20 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l611_61128


namespace NUMINAMATH_CALUDE_digit_equation_solutions_l611_61107

theorem digit_equation_solutions (n : ℕ) (hn : n ≥ 2) :
  let a (x : ℕ) := x * (10^n - 1) / 9
  let b (y : ℕ) := y * (10^n - 1) / 9
  let c (z : ℕ) := z * (10^(2*n) - 1) / 9
  ∀ x y z : ℕ, (a x)^2 + b y = c z →
    ((x = 3 ∧ y = 2 ∧ z = 1) ∨
     (x = 6 ∧ y = 8 ∧ z = 4) ∨
     (x = 8 ∧ y = 3 ∧ z = 7)) :=
by sorry

end NUMINAMATH_CALUDE_digit_equation_solutions_l611_61107


namespace NUMINAMATH_CALUDE_inequality_range_l611_61192

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2*x + 1| - |2*x - 1| < a) → a > 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l611_61192


namespace NUMINAMATH_CALUDE_min_value_of_expression_l611_61105

theorem min_value_of_expression (x : ℝ) :
  ∃ (min : ℝ), min = -4356 ∧ ∀ y : ℝ, (14 - y) * (8 - y) * (14 + y) * (8 + y) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l611_61105


namespace NUMINAMATH_CALUDE_inverse_negation_correct_l611_61132

/-- The original statement -/
def original_statement (x : ℝ) : Prop := x ≥ 3 → x < 0

/-- The inverse negation of the original statement -/
def inverse_negation (x : ℝ) : Prop := x ≥ 0 → x < 3

/-- Theorem stating that the inverse_negation is correct -/
theorem inverse_negation_correct : 
  (∀ x, original_statement x) ↔ (∀ x, inverse_negation x) :=
sorry

end NUMINAMATH_CALUDE_inverse_negation_correct_l611_61132


namespace NUMINAMATH_CALUDE_power_multiplication_l611_61163

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l611_61163


namespace NUMINAMATH_CALUDE_distance_walked_proof_l611_61191

/-- Calculates the distance walked by a person given their step length, steps per minute, and duration of walk. -/
def distanceWalked (stepLength : ℝ) (stepsPerMinute : ℝ) (durationMinutes : ℝ) : ℝ :=
  stepLength * stepsPerMinute * durationMinutes

/-- Proves that a person walking 0.75 meters per step at 70 steps per minute for 13 minutes covers 682.5 meters. -/
theorem distance_walked_proof :
  distanceWalked 0.75 70 13 = 682.5 := by
  sorry

#eval distanceWalked 0.75 70 13

end NUMINAMATH_CALUDE_distance_walked_proof_l611_61191


namespace NUMINAMATH_CALUDE_zoo_ticket_price_l611_61175

theorem zoo_ticket_price (regular_price : ℝ) (discount_percentage : ℝ) (discounted_price : ℝ) : 
  regular_price = 15 →
  discount_percentage = 40 →
  discounted_price = regular_price * (1 - discount_percentage / 100) →
  discounted_price = 9 := by
sorry

end NUMINAMATH_CALUDE_zoo_ticket_price_l611_61175


namespace NUMINAMATH_CALUDE_train_length_calculation_train_length_proof_l611_61195

/-- Calculates the length of a train given its speed, the time it takes to pass a bridge, and the length of the bridge. -/
theorem train_length_calculation (train_speed : Real) (bridge_length : Real) (time_to_pass : Real) : Real :=
  let speed_ms := train_speed * 1000 / 3600
  let total_distance := speed_ms * time_to_pass
  total_distance - bridge_length

/-- Proves that a train traveling at 45 km/hour passing a 140-meter bridge in 42 seconds has a length of 385 meters. -/
theorem train_length_proof :
  train_length_calculation 45 140 42 = 385 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_train_length_proof_l611_61195


namespace NUMINAMATH_CALUDE_water_added_is_twelve_liters_l611_61118

/-- Represents the composition of the kola solution -/
structure KolaSolution where
  total_volume : ℝ
  water_percentage : ℝ
  concentrated_kola_percentage : ℝ
  sugar_percentage : ℝ

/-- Calculates the amount of water added to the solution -/
def water_added (initial : KolaSolution) 
  (sugar_added : ℝ) (concentrated_kola_added : ℝ) (final_sugar_percentage : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the amount of water added is 12 liters -/
theorem water_added_is_twelve_liters :
  let initial := KolaSolution.mk 340 0.75 0.05 0.20
  let sugar_added := 3.2
  let concentrated_kola_added := 6.8
  let final_sugar_percentage := 0.1966850828729282
  water_added initial sugar_added concentrated_kola_added final_sugar_percentage = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_added_is_twelve_liters_l611_61118


namespace NUMINAMATH_CALUDE_CCl4_formation_l611_61161

-- Define the initial amounts of reactants
def initial_C2H6 : ℝ := 2
def initial_Cl2 : ℝ := 14

-- Define the stoichiometric ratio for each step
def stoichiometric_ratio : ℝ := 1

-- Define the number of reaction steps
def num_steps : ℕ := 4

-- Theorem statement
theorem CCl4_formation (remaining_Cl2 : ℝ → ℝ) 
  (h1 : remaining_Cl2 0 = initial_Cl2)
  (h2 : ∀ n : ℕ, n < num_steps → 
    remaining_Cl2 (n + 1) = remaining_Cl2 n - stoichiometric_ratio * initial_C2H6)
  (h3 : ∀ n : ℕ, n ≤ num_steps → remaining_Cl2 n ≥ 0) :
  remaining_Cl2 num_steps = initial_Cl2 - num_steps * stoichiometric_ratio * initial_C2H6 ∧
  initial_C2H6 = initial_C2H6 :=
by sorry

end NUMINAMATH_CALUDE_CCl4_formation_l611_61161


namespace NUMINAMATH_CALUDE_hall_breadth_is_15_meters_l611_61190

-- Define the hall length in meters
def hall_length : ℝ := 36

-- Define stone dimensions in decimeters
def stone_length : ℝ := 6
def stone_width : ℝ := 5

-- Define the number of stones
def num_stones : ℕ := 1800

-- Define the conversion factor from square decimeters to square meters
def dm2_to_m2 : ℝ := 0.01

-- Statement to prove
theorem hall_breadth_is_15_meters :
  let stone_area_m2 := stone_length * stone_width * dm2_to_m2
  let total_area_m2 := stone_area_m2 * num_stones
  let hall_breadth := total_area_m2 / hall_length
  hall_breadth = 15 := by sorry

end NUMINAMATH_CALUDE_hall_breadth_is_15_meters_l611_61190


namespace NUMINAMATH_CALUDE_sqrt_divisors_characterization_l611_61117

/-- The number of natural divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- A natural number n has exactly √n natural divisors -/
def has_sqrt_divisors (n : ℕ) : Prop := num_divisors n = Nat.sqrt n

theorem sqrt_divisors_characterization (n : ℕ) : has_sqrt_divisors n ↔ n = 1 ∨ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_divisors_characterization_l611_61117


namespace NUMINAMATH_CALUDE_sixth_term_is_twelve_l611_61115

/-- An arithmetic sequence with its first term and sum of first three terms specified -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  a₁ : a 1 = 2
  S₃ : (a 1) + (a 2) + (a 3) = 12
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- The 6th term of the arithmetic sequence is 12 -/
theorem sixth_term_is_twelve (seq : ArithmeticSequence) : seq.a 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_twelve_l611_61115


namespace NUMINAMATH_CALUDE_batsman_average_l611_61166

/-- Given a batsman whose average increases by 3 after scoring 66 runs in the 17th inning,
    his new average after the 17th inning is 18. -/
theorem batsman_average (prev_average : ℝ) : 
  (16 * prev_average + 66) / 17 = prev_average + 3 → prev_average + 3 = 18 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l611_61166


namespace NUMINAMATH_CALUDE_rally_speaking_orders_l611_61125

theorem rally_speaking_orders (n : ℕ) : 
  2 * (n.factorial) * (n.factorial) = 72 → n = 3 :=
by sorry

end NUMINAMATH_CALUDE_rally_speaking_orders_l611_61125


namespace NUMINAMATH_CALUDE_phase_without_chromatids_is_interkinesis_l611_61119

-- Define the phases of meiosis
inductive MeiosisPhase
  | prophaseI
  | interkinesis
  | prophaseII
  | lateProphaseII

-- Define a property for the presence of chromatids
def hasChromatids (phase : MeiosisPhase) : Prop :=
  match phase with
  | MeiosisPhase.interkinesis => False
  | _ => True

-- Define a property for DNA replication
def hasDNAReplication (phase : MeiosisPhase) : Prop :=
  match phase with
  | MeiosisPhase.interkinesis => False
  | _ => True

-- Theorem statement
theorem phase_without_chromatids_is_interkinesis :
  ∀ phase : MeiosisPhase, ¬(hasChromatids phase) → phase = MeiosisPhase.interkinesis :=
by
  sorry

end NUMINAMATH_CALUDE_phase_without_chromatids_is_interkinesis_l611_61119


namespace NUMINAMATH_CALUDE_three_speakers_from_different_companies_l611_61187

/-- The number of companies in the meeting -/
def total_companies : ℕ := 5

/-- The number of representatives from Company A -/
def company_a_reps : ℕ := 2

/-- The number of representatives from each of the other companies -/
def other_company_reps : ℕ := 1

/-- The number of speakers at the meeting -/
def num_speakers : ℕ := 3

/-- The number of scenarios where 3 speakers come from 3 different companies -/
def num_scenarios : ℕ := 16

theorem three_speakers_from_different_companies :
  (Nat.choose company_a_reps 1 * Nat.choose (total_companies - 1) 2) +
  (Nat.choose (total_companies - 1) 3) = num_scenarios :=
sorry

end NUMINAMATH_CALUDE_three_speakers_from_different_companies_l611_61187


namespace NUMINAMATH_CALUDE_community_center_pairing_l611_61171

theorem community_center_pairing (s t : ℕ) : 
  s > 0 ∧ t > 0 ∧ 
  4 * (t / 4) = 3 * (s / 3) ∧ 
  t / 4 = s / 3 →
  (t / 4 + s / 3) / (t + s) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_community_center_pairing_l611_61171


namespace NUMINAMATH_CALUDE_arithmetic_expressions_l611_61116

-- Define the number of arithmetic expressions
def f (n : ℕ) : ℚ :=
  (7/10) * 12^n - (1/5) * (-3)^n

-- State the theorem
theorem arithmetic_expressions (n : ℕ) :
  f n = (7/10) * 12^n - (1/5) * (-3)^n ∧
  f 1 = 9 ∧
  f 2 = 99 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_expressions_l611_61116


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l611_61151

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ (1 - 2*i) / (2 + i) = -i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l611_61151


namespace NUMINAMATH_CALUDE_CO2_yield_calculation_l611_61198

-- Define the chemical equation
def chemical_equation : String := "HCl + NaHCO3 → NaCl + H2O + CO2"

-- Define the molar quantities of reactants
def moles_HCl : ℝ := 1
def moles_NaHCO3 : ℝ := 1

-- Define the molar mass of CO2
def molar_mass_CO2 : ℝ := 44.01

-- Define the theoretical yield function
def theoretical_yield (moles_reactant : ℝ) (molar_mass_product : ℝ) : ℝ :=
  moles_reactant * molar_mass_product

-- Theorem statement
theorem CO2_yield_calculation :
  theoretical_yield (min moles_HCl moles_NaHCO3) molar_mass_CO2 = 44.01 := by
  sorry


end NUMINAMATH_CALUDE_CO2_yield_calculation_l611_61198


namespace NUMINAMATH_CALUDE_seymours_venus_flytraps_l611_61135

/-- Represents the plant shop inventory and fertilizer requirements --/
structure PlantShop where
  petunia_flats : ℕ
  petunias_per_flat : ℕ
  rose_flats : ℕ
  roses_per_flat : ℕ
  fertilizer_per_petunia : ℕ
  fertilizer_per_rose : ℕ
  fertilizer_per_venus_flytrap : ℕ
  total_fertilizer : ℕ

/-- Calculates the number of Venus flytraps in the shop --/
def venus_flytraps (shop : PlantShop) : ℕ :=
  let petunia_fertilizer := shop.petunia_flats * shop.petunias_per_flat * shop.fertilizer_per_petunia
  let rose_fertilizer := shop.rose_flats * shop.roses_per_flat * shop.fertilizer_per_rose
  let remaining_fertilizer := shop.total_fertilizer - petunia_fertilizer - rose_fertilizer
  remaining_fertilizer / shop.fertilizer_per_venus_flytrap

/-- Theorem stating that Seymour's shop has 2 Venus flytraps --/
theorem seymours_venus_flytraps :
  let shop : PlantShop := {
    petunia_flats := 4,
    petunias_per_flat := 8,
    rose_flats := 3,
    roses_per_flat := 6,
    fertilizer_per_petunia := 8,
    fertilizer_per_rose := 3,
    fertilizer_per_venus_flytrap := 2,
    total_fertilizer := 314
  }
  venus_flytraps shop = 2 := by
  sorry

end NUMINAMATH_CALUDE_seymours_venus_flytraps_l611_61135


namespace NUMINAMATH_CALUDE_card_area_problem_l611_61153

theorem card_area_problem (length width : ℝ) : 
  length = 5 ∧ width = 7 →
  (∃ side, (side = length - 2 ∨ side = width - 2) ∧ side * (if side = length - 2 then width else length) = 21) →
  (if length - 2 < width - 2 then (width - 2) * length else (length - 2) * width) = 25 := by
  sorry

end NUMINAMATH_CALUDE_card_area_problem_l611_61153


namespace NUMINAMATH_CALUDE_C_is_liar_l611_61154

-- Define the Islander type
inductive Islander : Type
  | Knight : Islander
  | Liar : Islander

-- Define the statements made by A and B
def A_statement (B : Islander) : Prop :=
  B = Islander.Liar

def B_statement (A C : Islander) : Prop :=
  (A = Islander.Knight ∧ C = Islander.Knight) ∨ (A = Islander.Liar ∧ C = Islander.Liar)

-- Define the truthfulness of statements based on the islander type
def is_truthful (i : Islander) (s : Prop) : Prop :=
  (i = Islander.Knight ∧ s) ∨ (i = Islander.Liar ∧ ¬s)

-- Theorem statement
theorem C_is_liar (A B C : Islander) 
  (h1 : is_truthful A (A_statement B))
  (h2 : is_truthful B (B_statement A C)) :
  C = Islander.Liar :=
sorry

end NUMINAMATH_CALUDE_C_is_liar_l611_61154


namespace NUMINAMATH_CALUDE_area_ratio_of_triangles_l611_61152

/-- Given two triangles PQR and XYZ with known base and height measurements,
    prove that the area of PQR is 1/3 of the area of XYZ. -/
theorem area_ratio_of_triangles (base_PQR height_PQR base_XYZ height_XYZ : ℝ)
  (h1 : base_PQR = 3)
  (h2 : height_PQR = 2)
  (h3 : base_XYZ = 6)
  (h4 : height_XYZ = 3) :
  (1 / 2 * base_PQR * height_PQR) / (1 / 2 * base_XYZ * height_XYZ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_triangles_l611_61152


namespace NUMINAMATH_CALUDE_cube_difference_factorization_l611_61103

theorem cube_difference_factorization (a b : ℝ) :
  a^3 - 8*b^3 = (a - 2*b) * (a^2 + 2*a*b + 4*b^2) := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_factorization_l611_61103


namespace NUMINAMATH_CALUDE_ken_payment_l611_61113

/-- The price of steak per pound -/
def price_per_pound : ℕ := 7

/-- The number of pounds of steak Ken bought -/
def pounds_bought : ℕ := 2

/-- The amount of change Ken received -/
def change_received : ℕ := 6

/-- The amount Ken paid -/
def amount_paid : ℕ := 20

/-- Proof that Ken paid $20 given the conditions -/
theorem ken_payment :
  amount_paid = price_per_pound * pounds_bought + change_received :=
by sorry

end NUMINAMATH_CALUDE_ken_payment_l611_61113


namespace NUMINAMATH_CALUDE_seans_spend_is_21_l611_61110

/-- The total amount Sean spends on his Sunday purchases -/
def seans_total_spend : ℚ :=
  let almond_croissant : ℚ := 9/2
  let salami_cheese_croissant : ℚ := 9/2
  let plain_croissant : ℚ := 3
  let focaccia : ℚ := 4
  let latte : ℚ := 5/2
  almond_croissant + salami_cheese_croissant + plain_croissant + focaccia + 2 * latte

/-- Theorem stating that Sean's total spend is $21.00 -/
theorem seans_spend_is_21 : seans_total_spend = 21 := by
  sorry

end NUMINAMATH_CALUDE_seans_spend_is_21_l611_61110


namespace NUMINAMATH_CALUDE_combined_building_time_l611_61145

/-- The time it takes Felipe to build his house, in months -/
def felipe_time : ℕ := 30

/-- The time it takes Emilio to build his house, in months -/
def emilio_time : ℕ := 2 * felipe_time

/-- The combined time for both Felipe and Emilio to build their homes, in years -/
def combined_time_years : ℚ := (felipe_time + emilio_time) / 12

theorem combined_building_time :
  combined_time_years = 7.5 := by sorry

end NUMINAMATH_CALUDE_combined_building_time_l611_61145


namespace NUMINAMATH_CALUDE_polygon_with_equal_angle_sums_l611_61182

theorem polygon_with_equal_angle_sums (n : ℕ) : 
  (n - 2) * 180 = 360 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_equal_angle_sums_l611_61182


namespace NUMINAMATH_CALUDE_binomial_seven_four_l611_61181

theorem binomial_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_four_l611_61181


namespace NUMINAMATH_CALUDE_bond_paper_cost_l611_61140

/-- Represents the cost of bond paper for an office. -/
structure BondPaperCost where
  sheets_per_ream : ℕ
  sheets_needed : ℕ
  total_cost : ℚ

/-- Calculates the cost of one ream of bond paper. -/
def cost_per_ream (bpc : BondPaperCost) : ℚ :=
  bpc.total_cost / (bpc.sheets_needed / bpc.sheets_per_ream)

/-- Theorem stating that the cost of one ream of bond paper is $27. -/
theorem bond_paper_cost (bpc : BondPaperCost)
  (h1 : bpc.sheets_per_ream = 500)
  (h2 : bpc.sheets_needed = 5000)
  (h3 : bpc.total_cost = 270) :
  cost_per_ream bpc = 27 := by
  sorry

end NUMINAMATH_CALUDE_bond_paper_cost_l611_61140


namespace NUMINAMATH_CALUDE_bread_slices_remaining_l611_61100

theorem bread_slices_remaining (total_slices : ℕ) (breakfast_fraction : ℚ) (lunch_slices : ℕ) : 
  total_slices = 12 →
  breakfast_fraction = 1 / 3 →
  lunch_slices = 2 →
  total_slices - (breakfast_fraction * total_slices).num - lunch_slices = 6 := by
sorry

end NUMINAMATH_CALUDE_bread_slices_remaining_l611_61100


namespace NUMINAMATH_CALUDE_penalty_kick_probability_l611_61136

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem penalty_kick_probability :
  let n : ℕ := 5
  let k : ℕ := 3
  let p : ℝ := 0.05
  abs (binomial_probability n k p - 0.00113) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_penalty_kick_probability_l611_61136


namespace NUMINAMATH_CALUDE_small_tile_position_l611_61160

/-- Represents a tile on the grid -/
inductive Tile
| Small : Tile  -- 1x1 tile
| Large : Tile  -- 1x3 tile

/-- Represents a position on the 7x7 grid -/
structure Position :=
  (row : Fin 7)
  (col : Fin 7)

/-- Checks if a position is on the border of the grid -/
def is_border (p : Position) : Prop :=
  p.row = 0 ∨ p.row = 6 ∨ p.col = 0 ∨ p.col = 6

/-- Checks if a position is in the center of the grid -/
def is_center (p : Position) : Prop :=
  p.row = 3 ∧ p.col = 3

/-- Represents the state of the grid -/
structure GridState :=
  (tiles : List (Tile × Position))
  (small_tile_count : Nat)
  (large_tile_count : Nat)

/-- Checks if a GridState is valid according to the problem conditions -/
def is_valid_state (state : GridState) : Prop :=
  state.small_tile_count = 1 ∧
  state.large_tile_count = 16 ∧
  state.tiles.length = 17

/-- The main theorem to prove -/
theorem small_tile_position (state : GridState) :
  is_valid_state state →
  ∃ (p : Position), (Tile.Small, p) ∈ state.tiles ∧ (is_border p ∨ is_center p) :=
sorry

end NUMINAMATH_CALUDE_small_tile_position_l611_61160


namespace NUMINAMATH_CALUDE_complex_equation_solution_l611_61122

theorem complex_equation_solution (c d x : ℂ) : 
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 6 * Complex.I →
  x = 3 * Real.sqrt 21 :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l611_61122


namespace NUMINAMATH_CALUDE_group_difference_theorem_l611_61148

theorem group_difference_theorem :
  let A := 19 * 10 + 55 * 100
  let B := 173 + 224 * 5
  A - B = 4397 := by sorry

end NUMINAMATH_CALUDE_group_difference_theorem_l611_61148


namespace NUMINAMATH_CALUDE_one_common_root_l611_61172

def quadratic1 (x : ℝ) := x^2 + x - 6
def quadratic2 (x : ℝ) := x^2 - 7*x + 10

theorem one_common_root :
  ∃! r : ℝ, quadratic1 r = 0 ∧ quadratic2 r = 0 :=
sorry

end NUMINAMATH_CALUDE_one_common_root_l611_61172


namespace NUMINAMATH_CALUDE_orange_harvest_difference_l611_61194

theorem orange_harvest_difference (ripe_sacks unripe_sacks : ℕ) 
  (h1 : ripe_sacks = 44) 
  (h2 : unripe_sacks = 25) : 
  ripe_sacks - unripe_sacks = 19 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_difference_l611_61194


namespace NUMINAMATH_CALUDE_sugar_problem_l611_61197

theorem sugar_problem (total_sugar : ℕ) (num_bags : ℕ) (h1 : total_sugar = 24) (h2 : num_bags = 4) :
  let sugar_per_bag := total_sugar / num_bags
  let lost_sugar := sugar_per_bag / 2
  total_sugar - lost_sugar = 21 := by
sorry

end NUMINAMATH_CALUDE_sugar_problem_l611_61197


namespace NUMINAMATH_CALUDE_binomial_square_constant_l611_61176

theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 300*x + c = (x + a)^2) → c = 22500 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l611_61176


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l611_61147

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 7 = 2) →
  (a 3)^2 - 2*(a 3) - 3 = 0 →
  (a 7)^2 - 2*(a 7) - 3 = 0 →
  a 1 + a 9 = 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l611_61147


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l611_61143

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The symmetric point of a given point with respect to the origin. -/
def symmetricPoint (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- Theorem: The symmetric point of (3, -4) with respect to the origin is (-3, 4). -/
theorem symmetric_point_theorem :
  let p : Point := { x := 3, y := -4 }
  symmetricPoint p = { x := -3, y := 4 } := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_theorem_l611_61143


namespace NUMINAMATH_CALUDE_cost_price_calculation_l611_61199

theorem cost_price_calculation (cost_price : ℝ) : 
  cost_price * (1 + 0.2) * 0.9 - cost_price = 8 → cost_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l611_61199


namespace NUMINAMATH_CALUDE_polynomial_decomposition_l611_61168

theorem polynomial_decomposition (x : ℝ) :
  x^3 - 2*x^2 + 3*x + 5 = 11 + 7*(x - 2) + 4*(x - 2)^2 + (x - 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_decomposition_l611_61168


namespace NUMINAMATH_CALUDE_geometric_mean_problem_l611_61183

/-- Given a geometric sequence {a_n} where a_2 = 9 and a_5 = 243,
    the geometric mean of a_1 and a_7 is ±81. -/
theorem geometric_mean_problem (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 = 9 →
  a 5 = 243 →
  (∃ x : ℝ, x ^ 2 = a 1 * a 7 ∧ (x = 81 ∨ x = -81)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_problem_l611_61183


namespace NUMINAMATH_CALUDE_gcd_of_large_powers_l611_61120

theorem gcd_of_large_powers (n m : ℕ) : 
  Nat.gcd (2^1050 - 1) (2^1062 - 1) = 2^12 - 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_large_powers_l611_61120


namespace NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l611_61102

/-- Theorem: Ratio of cone height to base radius when cone volume is one-third of sphere volume -/
theorem cone_sphere_volume_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3 : ℝ) * ((4 / 3 : ℝ) * Real.pi * r^3) = (1 / 3 : ℝ) * Real.pi * r^2 * h → h / r = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l611_61102


namespace NUMINAMATH_CALUDE_no_real_solution_log_equation_l611_61108

theorem no_real_solution_log_equation :
  ¬∃ x : ℝ, (Real.log (x + 6) + Real.log (x - 2) = Real.log (x^2 - 3*x - 18)) ∧
             (x + 6 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 3*x - 18 > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_log_equation_l611_61108


namespace NUMINAMATH_CALUDE_circle_equation_proof_line_equation_proof_l611_61106

-- Define the points
def A : ℝ × ℝ := (-4, -3)
def B : ℝ × ℝ := (2, 9)
def P : ℝ × ℝ := (0, 2)

-- Define the circle C with AB as its diameter
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 3)^2 = 45}

-- Define the line l₀
def l₀ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 2 = 0}

theorem circle_equation_proof :
  C = {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 3)^2 = 45} :=
sorry

theorem line_equation_proof :
  l₀ = {p : ℝ × ℝ | p.1 - p.2 + 2 = 0} :=
sorry

end NUMINAMATH_CALUDE_circle_equation_proof_line_equation_proof_l611_61106


namespace NUMINAMATH_CALUDE_pizza_order_l611_61144

theorem pizza_order (people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) 
  (h1 : people = 10)
  (h2 : slices_per_person = 2)
  (h3 : slices_per_pizza = 4) :
  (people * slices_per_person) / slices_per_pizza = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_l611_61144


namespace NUMINAMATH_CALUDE_decimal_representation_of_three_fortieths_l611_61156

theorem decimal_representation_of_three_fortieths : (3 : ℚ) / 40 = 0.075 := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_of_three_fortieths_l611_61156


namespace NUMINAMATH_CALUDE_equation_solution_l611_61179

theorem equation_solution (C D : ℝ) : 
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ 5 → (C * x - 20) / (x^2 - 3*x - 10) = D / (x + 2) + 4 / (x - 5)) →
  C + D = 4.7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l611_61179


namespace NUMINAMATH_CALUDE_inequality_proof_l611_61158

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l611_61158


namespace NUMINAMATH_CALUDE_train_speed_calculation_l611_61114

/-- Proves that under given conditions, the train's speed is 45 km/hr -/
theorem train_speed_calculation (jogger_speed : ℝ) (initial_distance : ℝ) 
  (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  initial_distance = 200 →
  train_length = 210 →
  passing_time = 41 →
  ∃ (train_speed : ℝ), train_speed = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l611_61114


namespace NUMINAMATH_CALUDE_salary_problem_l611_61189

-- Define the salaries and total
variable (A B : ℝ)
def total : ℝ := 6000

-- Define the spending percentages
def A_spend_percent : ℝ := 0.95
def B_spend_percent : ℝ := 0.85

-- Define the theorem
theorem salary_problem :
  A + B = total ∧ 
  (1 - A_spend_percent) * A = (1 - B_spend_percent) * B →
  A = 4500 := by
sorry

end NUMINAMATH_CALUDE_salary_problem_l611_61189


namespace NUMINAMATH_CALUDE_jose_chickens_l611_61186

/-- Given that Jose has 46 fowls in total and 18 ducks, prove that he has 28 chickens. -/
theorem jose_chickens : 
  ∀ (total_fowls ducks chickens : ℕ), 
    total_fowls = 46 → 
    ducks = 18 → 
    total_fowls = ducks + chickens → 
    chickens = 28 := by
  sorry

end NUMINAMATH_CALUDE_jose_chickens_l611_61186


namespace NUMINAMATH_CALUDE_cos_75_proof_l611_61134

noncomputable def cos_75 : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

theorem cos_75_proof :
  let cos_60 : ℝ := 1/2
  let sin_60 : ℝ := Real.sqrt 3 / 2
  let cos_15 : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4
  let sin_15 : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4
  let angle_addition_formula (A B : ℝ) := 
    Real.cos (A + B) = Real.cos A * Real.cos B - Real.sin A * Real.sin B
  cos_75 = Real.cos (75 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_cos_75_proof_l611_61134


namespace NUMINAMATH_CALUDE_solution_sets_l611_61196

def A (p : ℝ) : Set ℝ := {x | 2 * x^2 + x + p = 0}
def B (q : ℝ) : Set ℝ := {x | 2 * x^2 + q * x + 2 = 0}

theorem solution_sets (p q : ℝ) : 
  (A p ∩ B q = {1/2}) → 
  (A p = {-1, 1/2} ∧ B q = {2, 1/2} ∧ A p ∪ B q = {-1, 2, 1/2}) := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_l611_61196


namespace NUMINAMATH_CALUDE_slope_range_theorem_l611_61150

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line t
def line_t (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the condition for O being outside the circle with diameter PQ
def O_outside_circle (P Q : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  x₁ * x₂ + y₁ * y₂ > 0

theorem slope_range_theorem (k : ℝ) :
  (∃ P Q : ℝ × ℝ, P ≠ Q ∧
    C₁ P.1 P.2 ∧ C₁ Q.1 Q.2 ∧
    line_t k P.1 P.2 ∧ line_t k Q.1 Q.2 ∧
    O_outside_circle P Q) →
  k ∈ Set.Ioo (-2 : ℝ) (-Real.sqrt 3 / 2) ∪ Set.Ioo (Real.sqrt 3 / 2) 2 :=
by sorry

end NUMINAMATH_CALUDE_slope_range_theorem_l611_61150


namespace NUMINAMATH_CALUDE_ice_cream_cost_is_3_l611_61133

/-- The cost of items in dollars -/
structure Costs where
  coffee : ℕ
  cake : ℕ
  total : ℕ

/-- The number of items ordered -/
structure Order where
  coffee : ℕ
  cake : ℕ
  icecream : ℕ

def ice_cream_cost (c : Costs) (mell_order : Order) (friend_order : Order) : ℕ :=
  (c.total - (mell_order.coffee * c.coffee + mell_order.cake * c.cake +
    2 * (friend_order.coffee * c.coffee + friend_order.cake * c.cake))) / (2 * friend_order.icecream)

theorem ice_cream_cost_is_3 (c : Costs) (mell_order : Order) (friend_order : Order) :
  c.coffee = 4 →
  c.cake = 7 →
  c.total = 51 →
  mell_order = { coffee := 2, cake := 1, icecream := 0 } →
  friend_order = { coffee := 2, cake := 1, icecream := 1 } →
  ice_cream_cost c mell_order friend_order = 3 := by
  sorry

#check ice_cream_cost_is_3

end NUMINAMATH_CALUDE_ice_cream_cost_is_3_l611_61133


namespace NUMINAMATH_CALUDE_fold_line_length_l611_61184

-- Define the triangle
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let a := dist B C
  let b := dist A C
  let c := dist A B
  a = 5 ∧ b = 12 ∧ c = 13

-- Define the right angle at C
def right_angle_at_C (A B C : ℝ × ℝ) : Prop :=
  (dist B C)^2 + (dist A C)^2 = (dist A B)^2

-- Define the perpendicular bisector of AB
def perp_bisector_AB (A B : ℝ × ℝ) (D : ℝ × ℝ) : Prop :=
  dist A D = dist B D ∧ 
  (A.1 - B.1) * (D.1 - A.1) + (A.2 - B.2) * (D.2 - A.2) = 0

-- Theorem statement
theorem fold_line_length 
  (A B C : ℝ × ℝ) 
  (h1 : triangle_ABC A B C) 
  (h2 : right_angle_at_C A B C) :
  ∃ D : ℝ × ℝ, perp_bisector_AB A B D ∧ dist C D = Real.sqrt 7.33475 :=
sorry

end NUMINAMATH_CALUDE_fold_line_length_l611_61184


namespace NUMINAMATH_CALUDE_track_circumference_track_circumference_is_720_l611_61139

/-- The circumference of a circular track given specific meeting conditions of two joggers --/
theorem track_circumference : ℝ → ℝ → ℝ → Prop :=
  fun first_meet second_meet circumference =>
    let half_circumference := circumference / 2
    first_meet = 150 ∧
    second_meet = circumference - 90 ∧
    first_meet / (half_circumference - first_meet) = (half_circumference + 90) / (circumference - 90) →
    circumference = 720

/-- The main theorem stating that the track circumference is 720 yards --/
theorem track_circumference_is_720 : ∃ (first_meet second_meet : ℝ),
  track_circumference first_meet second_meet 720 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_track_circumference_is_720_l611_61139


namespace NUMINAMATH_CALUDE_nathan_tokens_l611_61101

/-- Calculates the total number of tokens used by Nathan at the arcade -/
def total_tokens (air_hockey_games basketball_games skee_ball_games shooting_games racing_games : ℕ)
  (air_hockey_cost basketball_cost skee_ball_cost shooting_cost racing_cost : ℕ) : ℕ :=
  air_hockey_games * air_hockey_cost +
  basketball_games * basketball_cost +
  skee_ball_games * skee_ball_cost +
  shooting_games * shooting_cost +
  racing_games * racing_cost

theorem nathan_tokens :
  total_tokens 7 12 9 6 5 6 8 4 7 5 = 241 := by
  sorry

end NUMINAMATH_CALUDE_nathan_tokens_l611_61101


namespace NUMINAMATH_CALUDE_janet_earnings_l611_61180

/-- Calculates the hourly earnings for checking social media posts. -/
def hourly_earnings (pay_per_post : ℚ) (seconds_per_post : ℕ) : ℚ :=
  let posts_per_hour : ℕ := 3600 / seconds_per_post
  pay_per_post * posts_per_hour

/-- Proves that given a pay rate of 25 cents per post and a checking time of 10 seconds per post, the hourly earnings are $90. -/
theorem janet_earnings : hourly_earnings (25 / 100) 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_janet_earnings_l611_61180


namespace NUMINAMATH_CALUDE_percent_relation_l611_61170

/-- Given that x is p percent more than 1/y, prove that y = (100 + p) / (100x) -/
theorem percent_relation (x y p : ℝ) (h : x = (1 + p / 100) * (1 / y)) :
  y = (100 + p) / (100 * x) := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l611_61170


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l611_61185

theorem sphere_surface_area_ratio : 
  let r₁ : ℝ := 6
  let r₂ : ℝ := 3
  let surface_area (r : ℝ) := 4 * Real.pi * r^2
  (surface_area r₁) / (surface_area r₂) = 4 := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l611_61185


namespace NUMINAMATH_CALUDE_value_of_x_l611_61138

theorem value_of_x (x y z : ℚ) : 
  x = (1 / 3) * y → 
  y = (1 / 4) * z → 
  z = 96 → 
  x = 8 := by sorry

end NUMINAMATH_CALUDE_value_of_x_l611_61138


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l611_61123

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), (1021 + x) % 25 = 0 ∧ 
  ∀ (y : ℕ), (1021 + y) % 25 = 0 → x ≤ y :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l611_61123


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l611_61173

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 1 → 1 / a < 1) ∧ 
  (∃ a, 1 / a < 1 ∧ ¬(a > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l611_61173


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l611_61121

/-- A rhombus with given diagonal lengths has the specified perimeter -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side_length := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side_length = 52 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l611_61121


namespace NUMINAMATH_CALUDE_pyramid_top_value_l611_61165

/-- Represents a pyramid structure with four levels -/
structure Pyramid :=
  (bottom_left : ℕ)
  (bottom_right : ℕ)
  (second_left : ℕ)
  (second_right : ℕ)
  (third_left : ℕ)
  (third_right : ℕ)
  (top : ℕ)

/-- Checks if the pyramid satisfies the product rule -/
def is_valid_pyramid (p : Pyramid) : Prop :=
  p.bottom_left = p.second_left * p.third_left ∧
  p.bottom_right = p.second_right * p.third_right ∧
  p.second_left = p.top * p.third_left ∧
  p.second_right = p.top * p.third_right

theorem pyramid_top_value (p : Pyramid) :
  p.bottom_left = 300 ∧ 
  p.bottom_right = 1800 ∧ 
  p.second_left = 6 ∧ 
  p.second_right = 30 ∧
  is_valid_pyramid p →
  p.top = 60 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_top_value_l611_61165


namespace NUMINAMATH_CALUDE_minoxidil_concentration_l611_61157

/-- Proves that the initial concentration of Minoxidil is 2% --/
theorem minoxidil_concentration 
  (initial_volume : ℝ) 
  (added_volume : ℝ) 
  (added_concentration : ℝ) 
  (final_volume : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_volume = 70)
  (h2 : added_volume = 35)
  (h3 : added_concentration = 0.05)
  (h4 : final_volume = 105)
  (h5 : final_concentration = 0.03)
  (h6 : final_volume = initial_volume + added_volume) :
  ∃ (initial_concentration : ℝ), 
    initial_concentration = 0.02 ∧ 
    initial_volume * initial_concentration + added_volume * added_concentration = 
    final_volume * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_minoxidil_concentration_l611_61157


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l611_61109

/-- Given a polar coordinate equation ρ = 4sin θ + 2cos θ, 
    prove it is equivalent to the rectangular coordinate equation (x-1)^2 + (y-2)^2 = 5 -/
theorem polar_to_rectangular_equivalence :
  ∀ (x y ρ θ : ℝ), 
    ρ = 4 * Real.sin θ + 2 * Real.cos θ → 
    x = ρ * Real.cos θ → 
    y = ρ * Real.sin θ → 
    (x - 1)^2 + (y - 2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l611_61109


namespace NUMINAMATH_CALUDE_train_length_l611_61142

/-- Given a train that crosses a platform in 50 seconds and a signal pole in 42 seconds,
    with the platform length being 38.0952380952381 meters, prove that the length of the train is 200 meters. -/
theorem train_length (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
    (h1 : platform_crossing_time = 50)
    (h2 : pole_crossing_time = 42)
    (h3 : platform_length = 38.0952380952381) :
    ∃ train_length : ℝ, train_length = 200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l611_61142


namespace NUMINAMATH_CALUDE_inequality_proof_l611_61124

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : 1/x + 1/y + 1/z = 2) : 
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l611_61124


namespace NUMINAMATH_CALUDE_largest_prime_factor_l611_61131

def numbers : List Nat := [55, 63, 85, 94, 133]

def has_largest_prime_factor (n : Nat) (lst : List Nat) : Prop :=
  ∀ m ∈ lst, ∃ p q : Nat, 
    Nat.Prime p ∧ 
    n = p * q ∧ 
    ∀ r s : Nat, (Nat.Prime r ∧ m = r * s) → r ≤ p

theorem largest_prime_factor : 
  has_largest_prime_factor 94 numbers := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l611_61131
