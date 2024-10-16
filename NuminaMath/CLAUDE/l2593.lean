import Mathlib

namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2593_259376

/-- For all real x, the expression 
    sin(2x - π) * cos(x - 3π) + sin(2x - 9π/2) * cos(x + π/2) 
    is equal to sin(3x) -/
theorem trigonometric_simplification (x : ℝ) : 
  Real.sin (2 * x - Real.pi) * Real.cos (x - 3 * Real.pi) + 
  Real.sin (2 * x - 9 * Real.pi / 2) * Real.cos (x + Real.pi / 2) = 
  Real.sin (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2593_259376


namespace NUMINAMATH_CALUDE_expression_evaluation_l2593_259385

theorem expression_evaluation : (20 + 16 * 20) / (20 * 16) = 17 / 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2593_259385


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l2593_259383

theorem quadratic_roots_problem (m : ℝ) (x₁ x₂ : ℝ) :
  (x₁^2 + 2*(m+1)*x₁ + m^2 - 1 = 0) →
  (x₂^2 + 2*(m+1)*x₂ + m^2 - 1 = 0) →
  ((x₁ - x₂)^2 = 16 - x₁*x₂) →
  (m = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l2593_259383


namespace NUMINAMATH_CALUDE_ten_mile_taxi_cost_l2593_259313

/-- The cost of a taxi ride given the base fare, cost per mile, and distance traveled. -/
def taxi_cost (base_fare : ℝ) (cost_per_mile : ℝ) (distance : ℝ) : ℝ :=
  base_fare + cost_per_mile * distance

/-- Theorem: The cost of a 10-mile taxi ride with a $2.00 base fare and $0.30 per mile is $5.00. -/
theorem ten_mile_taxi_cost :
  taxi_cost 2 0.3 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ten_mile_taxi_cost_l2593_259313


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l2593_259399

theorem second_term_of_geometric_series :
  ∀ (a : ℝ) (r : ℝ) (S : ℝ),
    r = (1 : ℝ) / 4 →
    S = 16 →
    S = a / (1 - r) →
    a * r = 3 :=
by sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l2593_259399


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2593_259371

theorem quadratic_function_properties (a c : ℕ+) (m : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (a : ℝ) * x^2 + 2 * x + c
  (f 1 = 5) →
  (6 < f 2 ∧ f 2 < 11) →
  (∀ x ∈ Set.Icc (1/2 : ℝ) (3/2 : ℝ), f x - 2 * m * x ≤ 1) →
  (a = 1 ∧ c = 2 ∧ m ≥ 9/4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2593_259371


namespace NUMINAMATH_CALUDE_star_running_back_yardage_l2593_259393

/-- Represents the yardage gained by a player in a football game -/
structure Yardage where
  running : ℕ
  catching : ℕ

/-- Calculates the total yardage for a player -/
def totalYardage (y : Yardage) : ℕ :=
  y.running + y.catching

/-- Theorem: The total yardage of a player who gained 90 yards running and 60 yards catching is 150 yards -/
theorem star_running_back_yardage :
  let y : Yardage := { running := 90, catching := 60 }
  totalYardage y = 150 := by
  sorry

end NUMINAMATH_CALUDE_star_running_back_yardage_l2593_259393


namespace NUMINAMATH_CALUDE_peanut_ratio_l2593_259304

theorem peanut_ratio (initial : ℕ) (eaten_by_bonita : ℕ) (remaining : ℕ)
  (h1 : initial = 148)
  (h2 : eaten_by_bonita = 29)
  (h3 : remaining = 82) :
  (initial - remaining - eaten_by_bonita) / initial = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_peanut_ratio_l2593_259304


namespace NUMINAMATH_CALUDE_number_satisfying_condition_l2593_259360

theorem number_satisfying_condition : ∃! x : ℝ, x / 3 + 12 = 20 ∧ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_condition_l2593_259360


namespace NUMINAMATH_CALUDE_log_inequality_l2593_259316

theorem log_inequality (a b c : ℝ) 
  (ha : a = Real.log 3 / Real.log (1/2))
  (hb : b = Real.log 5 / Real.log (1/2))
  (hc : c = Real.log (1/2) / Real.log 3) :
  b < a ∧ a < c := by
sorry

end NUMINAMATH_CALUDE_log_inequality_l2593_259316


namespace NUMINAMATH_CALUDE_pizza_promotion_savings_l2593_259384

/-- Calculates the total savings from a pizza promotion -/
theorem pizza_promotion_savings 
  (regular_price : ℕ) 
  (promo_price : ℕ) 
  (num_pizzas : ℕ) 
  (h1 : regular_price = 18) 
  (h2 : promo_price = 5) 
  (h3 : num_pizzas = 3) : 
  (regular_price - promo_price) * num_pizzas = 39 := by
  sorry

#check pizza_promotion_savings

end NUMINAMATH_CALUDE_pizza_promotion_savings_l2593_259384


namespace NUMINAMATH_CALUDE_problem_solution_l2593_259387

theorem problem_solution : 25 * (216 / 3 + 36 / 6 + 16 / 25 + 2) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2593_259387


namespace NUMINAMATH_CALUDE_diamond_value_in_treasure_l2593_259337

/-- Represents the treasure of precious stones -/
structure Treasure where
  diamond_masses : List ℝ
  crystal_mass : ℝ
  total_value : ℝ
  martin_value : ℝ

/-- Calculates the value of diamonds given their masses -/
def diamond_value (masses : List ℝ) : ℝ :=
  100 * (masses.map (λ m => m^2)).sum

/-- Calculates the value of crystals given their mass -/
def crystal_value (mass : ℝ) : ℝ :=
  3 * mass

/-- The main theorem about the value of diamonds in the treasure -/
theorem diamond_value_in_treasure (t : Treasure) : 
  t.total_value = 5000000 ∧ 
  t.martin_value = 2000000 ∧ 
  t.total_value = diamond_value t.diamond_masses + crystal_value t.crystal_mass ∧
  t.martin_value = diamond_value (t.diamond_masses.map (λ m => m/2)) + crystal_value (t.crystal_mass/2) →
  diamond_value t.diamond_masses = 2000000 := by
  sorry

end NUMINAMATH_CALUDE_diamond_value_in_treasure_l2593_259337


namespace NUMINAMATH_CALUDE_larger_solution_quadratic_equation_l2593_259359

theorem larger_solution_quadratic_equation :
  ∃ (x y : ℝ), x ≠ y ∧ 
  x^2 - 13*x + 40 = 0 ∧ 
  y^2 - 13*y + 40 = 0 ∧ 
  (∀ z : ℝ, z^2 - 13*z + 40 = 0 → z = x ∨ z = y) ∧
  max x y = 8 := by
sorry

end NUMINAMATH_CALUDE_larger_solution_quadratic_equation_l2593_259359


namespace NUMINAMATH_CALUDE_tangent_slope_point_A_l2593_259395

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 2*x + 3

-- Theorem statement
theorem tangent_slope_point_A :
  ∃ (x y : ℝ), 
    f_derivative x = 7 ∧ 
    f x = y ∧ 
    x = 2 ∧ 
    y = 10 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_point_A_l2593_259395


namespace NUMINAMATH_CALUDE_calculate_expression_l2593_259390

theorem calculate_expression : (10^10 / (2 * 10^6)) * 3 = 15000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2593_259390


namespace NUMINAMATH_CALUDE_initial_puppies_count_l2593_259382

/-- The number of puppies Alyssa had initially --/
def initial_puppies : ℕ := 12

/-- The number of puppies Alyssa gave away --/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has left --/
def puppies_left : ℕ := 5

/-- Theorem stating that the initial number of puppies is equal to
    the sum of puppies given away and puppies left --/
theorem initial_puppies_count :
  initial_puppies = puppies_given_away + puppies_left := by
  sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l2593_259382


namespace NUMINAMATH_CALUDE_stones_division_l2593_259336

/-- Definition of similar sizes -/
def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

/-- A step in the combining process -/
inductive CombineStep
  | combine (x y : ℕ) (h : similar_sizes x y) : CombineStep

/-- A sequence of combining steps -/
def CombineSequence := List CombineStep

/-- The result of applying a sequence of combining steps -/
def apply_sequence (initial : List ℕ) (seq : CombineSequence) : List ℕ := sorry

/-- The theorem stating that any pile can be divided into single stones -/
theorem stones_division (n : ℕ) : 
  ∃ (seq : CombineSequence), 
    apply_sequence (List.replicate n 1) seq = [n] := sorry

end NUMINAMATH_CALUDE_stones_division_l2593_259336


namespace NUMINAMATH_CALUDE_standard_spherical_coordinates_example_l2593_259378

/-- 
Given a point in spherical coordinates (ρ, θ, φ), this function returns the 
standard representation coordinates (ρ', θ', φ') where:
- ρ' = ρ
- θ' is θ adjusted to be in the range [0, 2π)
- φ' is φ adjusted to be in the range [0, π]
-/
def standardSphericalCoordinates (ρ θ φ : Real) : Real × Real × Real :=
  sorry

theorem standard_spherical_coordinates_example :
  let (ρ, θ, φ) := (5, 3 * Real.pi / 8, 9 * Real.pi / 5)
  let (ρ', θ', φ') := standardSphericalCoordinates ρ θ φ
  ρ' = 5 ∧ θ' = 3 * Real.pi / 8 ∧ φ' = Real.pi / 5 :=
by sorry

end NUMINAMATH_CALUDE_standard_spherical_coordinates_example_l2593_259378


namespace NUMINAMATH_CALUDE_roots_transformation_l2593_259314

theorem roots_transformation (s₁ s₂ s₃ : ℂ) : 
  (s₁^3 - 4*s₁^2 + 5*s₁ - 1 = 0) ∧ 
  (s₂^3 - 4*s₂^2 + 5*s₂ - 1 = 0) ∧ 
  (s₃^3 - 4*s₃^2 + 5*s₃ - 1 = 0) →
  ((3*s₁)^3 - 12*(3*s₁)^2 + 135*(3*s₁) - 27 = 0) ∧
  ((3*s₂)^3 - 12*(3*s₂)^2 + 135*(3*s₂) - 27 = 0) ∧
  ((3*s₃)^3 - 12*(3*s₃)^2 + 135*(3*s₃) - 27 = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_transformation_l2593_259314


namespace NUMINAMATH_CALUDE_choose_10_4_l2593_259324

theorem choose_10_4 : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_10_4_l2593_259324


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l2593_259335

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.cos (10 * π / 180) =
  (2 * Real.cos (40 * π / 180)) / (Real.cos (10 * π / 180) ^ 2 * Real.cos (30 * π / 180) * Real.cos (60 * π / 180) * Real.cos (70 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l2593_259335


namespace NUMINAMATH_CALUDE_largest_six_digit_with_product_40320_l2593_259340

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).prod

theorem largest_six_digit_with_product_40320 :
  ∃ (n : ℕ), is_six_digit n ∧ digit_product n = 40320 ∧
  ∀ (m : ℕ), is_six_digit m → digit_product m = 40320 → m ≤ n :=
by
  use 988752
  sorry

#eval digit_product 988752  -- Should output 40320

end NUMINAMATH_CALUDE_largest_six_digit_with_product_40320_l2593_259340


namespace NUMINAMATH_CALUDE_sum_mod_seven_l2593_259341

theorem sum_mod_seven : (8171 + 8172 + 8173 + 8174 + 8175) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_seven_l2593_259341


namespace NUMINAMATH_CALUDE_max_rectangles_correct_max_rectangles_optimal_l2593_259346

def max_rectangles (n : ℕ) : ℕ :=
  match n with
  | 1 => 2
  | 2 => 5
  | 3 => 8
  | _ => 4 * n - 4

theorem max_rectangles_correct (n : ℕ) :
  max_rectangles n = 
    if n = 1 then 2
    else if n = 2 then 5
    else if n = 3 then 8
    else 4 * n - 4 :=
by sorry

theorem max_rectangles_optimal (n : ℕ) :
  max_rectangles n ≤ (2 * n * 2 * n) / (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_max_rectangles_correct_max_rectangles_optimal_l2593_259346


namespace NUMINAMATH_CALUDE_handshake_count_l2593_259398

theorem handshake_count (n : ℕ) (h : n = 8) : 
  (n * (n - 2)) / 2 = 24 := by
  sorry

#check handshake_count

end NUMINAMATH_CALUDE_handshake_count_l2593_259398


namespace NUMINAMATH_CALUDE_max_angle_sum_l2593_259355

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral :=
  (x : ℝ)
  (d : ℝ)
  (angle_sum : x + (x + 2*d) + (x + d) = 180)
  (angle_progression : x ≤ x + d ∧ x + d ≤ x + 2*d)
  (similarity : x + d = 60)

/-- The maximum sum of the largest angles in triangles ABC and ACD is 180° -/
theorem max_angle_sum (q : Quadrilateral) :
  ∃ (max_sum : ℝ), max_sum = 180 ∧
  ∀ (sum : ℝ), sum = (q.x + 2*q.d) + (q.x + 2*q.d) → sum ≤ max_sum :=
sorry

end NUMINAMATH_CALUDE_max_angle_sum_l2593_259355


namespace NUMINAMATH_CALUDE_range_of_a_l2593_259379

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | -3 ≤ x ∧ x ≤ a}
def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 3*x + 10}
def C (a : ℝ) : Set ℝ := {z | ∃ x ∈ A a, z = 5 - x}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (B a ∩ C a = C a) → (-2/3 ≤ a ∧ a ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2593_259379


namespace NUMINAMATH_CALUDE_files_sorted_in_one_and_half_hours_l2593_259389

/-- Represents the number of files sorted by a group of clerks under specific conditions. -/
def filesSortedInOneAndHalfHours (totalFiles : ℕ) (filesPerHourPerClerk : ℕ) (totalTime : ℚ) : ℕ :=
  let initialClerks := 22  -- Derived from the problem conditions
  let reassignedClerks := 3  -- Derived from the problem conditions
  initialClerks * filesPerHourPerClerk + (initialClerks - reassignedClerks) * (filesPerHourPerClerk / 2)

/-- Proves that under the given conditions, the number of files sorted in 1.5 hours is 945. -/
theorem files_sorted_in_one_and_half_hours :
  filesSortedInOneAndHalfHours 1775 30 (157/60) = 945 := by
  sorry

#eval filesSortedInOneAndHalfHours 1775 30 (157/60)

end NUMINAMATH_CALUDE_files_sorted_in_one_and_half_hours_l2593_259389


namespace NUMINAMATH_CALUDE_journey_speed_journey_speed_theorem_l2593_259328

/-- 
Given a journey of 24 km completed in 8 hours, where the first 4 hours are
traveled at speed v km/hr and the last 4 hours at 2 km/hr, prove that v = 4.
-/
theorem journey_speed : ℝ → Prop :=
  fun v : ℝ =>
    (4 * v + 4 * 2 = 24) →
    v = 4

-- The proof is omitted
axiom journey_speed_proof : journey_speed 4

#check journey_speed_proof

-- Proof
theorem journey_speed_theorem : ∃ v : ℝ, journey_speed v := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_journey_speed_theorem_l2593_259328


namespace NUMINAMATH_CALUDE_clock_malfunction_theorem_l2593_259305

/-- Represents a time in 24-hour format -/
structure Time where
  hours : Fin 24
  minutes : Fin 60

/-- Represents the possible changes to a digit due to malfunction -/
inductive DigitChange
  | Increase
  | Decrease

/-- Applies the digit change to a number, wrapping around if necessary -/
def applyDigitChange (n : Fin 10) (change : DigitChange) : Fin 10 :=
  match change with
  | DigitChange.Increase => (n + 1) % 10
  | DigitChange.Decrease => (n + 9) % 10

/-- Applies changes to both digits of a two-digit number -/
def applyTwoDigitChange (n : Fin 100) (tens_change : DigitChange) (units_change : DigitChange) : Fin 100 :=
  let tens := n / 10
  let units := n % 10
  (applyDigitChange tens tens_change) * 10 + (applyDigitChange units units_change)

theorem clock_malfunction_theorem (malfunctioned_time : Time) 
    (h : malfunctioned_time.hours = 9 ∧ malfunctioned_time.minutes = 9) :
    ∃ (original_time : Time) (hours_tens_change hours_units_change minutes_tens_change minutes_units_change : DigitChange),
      original_time.hours = 18 ∧
      original_time.minutes = 18 ∧
      applyTwoDigitChange original_time.hours hours_tens_change hours_units_change = malfunctioned_time.hours ∧
      applyTwoDigitChange original_time.minutes minutes_tens_change minutes_units_change = malfunctioned_time.minutes :=
by sorry

end NUMINAMATH_CALUDE_clock_malfunction_theorem_l2593_259305


namespace NUMINAMATH_CALUDE_layla_apples_l2593_259306

theorem layla_apples (maggie : ℕ) (kelsey : ℕ) (layla : ℕ) :
  maggie = 40 →
  kelsey = 28 →
  (maggie + kelsey + layla) / 3 = 30 →
  layla = 22 :=
by sorry

end NUMINAMATH_CALUDE_layla_apples_l2593_259306


namespace NUMINAMATH_CALUDE_platform_length_l2593_259317

/-- The length of a platform given train parameters -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmph = 60 →
  crossing_time = 15 →
  ∃ (platform_length : ℝ), abs (platform_length - 130.05) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_platform_length_l2593_259317


namespace NUMINAMATH_CALUDE_value_of_x_l2593_259362

theorem value_of_x : ∀ (w y z x : ℤ), 
  w = 50 → 
  z = w + 25 → 
  y = z + 15 → 
  x = y + 7 → 
  x = 97 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2593_259362


namespace NUMINAMATH_CALUDE_parabola_translation_l2593_259380

def parabola1 (x : ℝ) := -(x - 1)^2 + 3
def parabola2 (x : ℝ) := -x^2

def translation (f : ℝ → ℝ) (h k : ℝ) : ℝ → ℝ :=
  λ x => f (x + h) + k

theorem parabola_translation :
  ∃ h k : ℝ, (∀ x : ℝ, translation parabola1 h k x = parabola2 x) ∧ h = 1 ∧ k = -3 :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_l2593_259380


namespace NUMINAMATH_CALUDE_quadratic_solution_comparison_l2593_259300

/-- Given two quadratic equations ax² + bx + c = 0 and a'x² + b'x + c' = 0,
    where a and a' are non-zero, this theorem states that the largest solution
    of the first equation is less than the smallest solution of the second
    equation if and only if (b')² - 4a'c' > b² - 4ac. -/
theorem quadratic_solution_comparison
  (a a' b b' c c' : ℝ)
  (ha : a ≠ 0)
  (ha' : a' ≠ 0) :
  ((-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) < (-b' - Real.sqrt ((b')^2 - 4*a'*c')) / (2*a')) ↔
  ((b')^2 - 4*a'*c' > b^2 - 4*a*c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_comparison_l2593_259300


namespace NUMINAMATH_CALUDE_woody_saves_in_ten_weeks_l2593_259307

/-- The number of weeks required for Woody to save enough money for a games console. -/
def weeks_to_save (console_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) : ℕ :=
  ((console_cost - initial_savings) + weekly_allowance - 1) / weekly_allowance

/-- Theorem stating that it takes Woody 10 weeks to save for the games console. -/
theorem woody_saves_in_ten_weeks :
  weeks_to_save 282 42 24 = 10 := by
  sorry

end NUMINAMATH_CALUDE_woody_saves_in_ten_weeks_l2593_259307


namespace NUMINAMATH_CALUDE_negation_of_forall_x_squared_gt_x_l2593_259321

theorem negation_of_forall_x_squared_gt_x :
  (¬ ∀ x : ℕ, x^2 > x) ↔ (∃ x₀ : ℕ, x₀^2 ≤ x₀) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_x_squared_gt_x_l2593_259321


namespace NUMINAMATH_CALUDE_vegetable_ghee_mixture_weight_l2593_259352

/-- Calculates the weight of a mixture of two brands of vegetable ghee -/
theorem vegetable_ghee_mixture_weight
  (weight_a : ℝ)
  (weight_b : ℝ)
  (ratio_a : ℝ)
  (ratio_b : ℝ)
  (total_volume : ℝ)
  (h_weight_a : weight_a = 800)
  (h_weight_b : weight_b = 850)
  (h_ratio_a : ratio_a = 3)
  (h_ratio_b : ratio_b = 2)
  (h_total_volume : total_volume = 3) :
  let volume_a := (ratio_a / (ratio_a + ratio_b)) * total_volume
  let volume_b := (ratio_b / (ratio_a + ratio_b)) * total_volume
  let total_weight := (weight_a * volume_a + weight_b * volume_b) / 1000
  total_weight = 2.46 := by
  sorry


end NUMINAMATH_CALUDE_vegetable_ghee_mixture_weight_l2593_259352


namespace NUMINAMATH_CALUDE_larger_integer_value_l2593_259334

theorem larger_integer_value (a b : ℕ+) (h1 : a > b) (h2 : (a : ℝ) / (b : ℝ) = 7 / 3) (h3 : (a : ℕ) * b = 294) :
  (a : ℝ) = 7 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l2593_259334


namespace NUMINAMATH_CALUDE_opponent_total_score_l2593_259343

def hockey_team_goals : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

structure GameResult where
  team_score : Nat
  opponent_score : Nat

def is_lost_by_two (game : GameResult) : Bool :=
  game.opponent_score = game.team_score + 2

def is_half_or_double (game : GameResult) : Bool :=
  game.team_score = 2 * game.opponent_score ∨ 2 * game.team_score = game.opponent_score

theorem opponent_total_score (games : List GameResult) : 
  (games.length = 8) →
  (games.map (λ g => g.team_score) = hockey_team_goals) →
  (games.filter is_lost_by_two).length = 3 →
  (games.filter (λ g => ¬(is_lost_by_two g))).all is_half_or_double →
  (games.map (λ g => g.opponent_score)).sum = 56 := by
  sorry

end NUMINAMATH_CALUDE_opponent_total_score_l2593_259343


namespace NUMINAMATH_CALUDE_farm_chickens_l2593_259365

/-- Represents the number of roosters initially on the farm. -/
def initial_roosters : ℕ := sorry

/-- Represents the number of hens initially on the farm. -/
def initial_hens : ℕ := 6 * initial_roosters

/-- Represents the number of roosters added to the farm. -/
def added_roosters : ℕ := 60

/-- Represents the number of hens added to the farm. -/
def added_hens : ℕ := 60

/-- Represents the total number of roosters after additions. -/
def final_roosters : ℕ := initial_roosters + added_roosters

/-- Represents the total number of hens after additions. -/
def final_hens : ℕ := initial_hens + added_hens

/-- States that after additions, the number of hens is 4 times the number of roosters. -/
axiom final_ratio : final_hens = 4 * final_roosters

/-- Represents the total number of chickens initially on the farm. -/
def total_chickens : ℕ := initial_roosters + initial_hens

/-- Proves that the total number of chickens initially on the farm was 630. -/
theorem farm_chickens : total_chickens = 630 := by sorry

end NUMINAMATH_CALUDE_farm_chickens_l2593_259365


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l2593_259329

def number_of_people : ℕ := 10

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem seating_arrangements_with_restriction :
  let total_arrangements := factorial (number_of_people - 1) * 7
  total_arrangements = 2540160 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l2593_259329


namespace NUMINAMATH_CALUDE_liam_monthly_savings_l2593_259375

/-- Calculates the monthly savings given the trip cost, bills cost, saving period in years, and amount left after paying bills. -/
def monthly_savings (trip_cost bills_cost : ℚ) (saving_period : ℕ) (amount_left : ℚ) : ℚ :=
  (amount_left + bills_cost + trip_cost) / (saving_period * 12)

/-- Theorem stating that Liam's monthly savings are $791.67 given the problem conditions. -/
theorem liam_monthly_savings :
  let trip_cost : ℚ := 7000
  let bills_cost : ℚ := 3500
  let saving_period : ℕ := 2
  let amount_left : ℚ := 8500
  monthly_savings trip_cost bills_cost saving_period amount_left = 791.67 := by
  sorry

#eval monthly_savings 7000 3500 2 8500

end NUMINAMATH_CALUDE_liam_monthly_savings_l2593_259375


namespace NUMINAMATH_CALUDE_square_of_linear_expression_l2593_259345

theorem square_of_linear_expression (x : ℝ) :
  x = -2 → (3 * x + 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_linear_expression_l2593_259345


namespace NUMINAMATH_CALUDE_sasha_remaining_questions_l2593_259348

/-- Calculates the number of remaining questions given the completion rate, total questions, and work time. -/
def remaining_questions (completion_rate : ℕ) (total_questions : ℕ) (work_time : ℕ) : ℕ :=
  total_questions - completion_rate * work_time

/-- Proves that for Sasha's specific case, the number of remaining questions is 30. -/
theorem sasha_remaining_questions :
  remaining_questions 15 60 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sasha_remaining_questions_l2593_259348


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l2593_259323

/-- A hyperbola with eccentricity √3 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0
  h_ecc : b^2 = 2 * a^2

/-- A line with slope 1 -/
structure Line where
  k : ℝ

/-- Points P and Q on the hyperbola, and R on the y-axis -/
structure Points (h : Hyperbola) (l : Line) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  h_on_hyperbola : 
    (P.1^2 / h.a^2 - P.2^2 / h.b^2 = 1) ∧
    (Q.1^2 / h.a^2 - Q.2^2 / h.b^2 = 1)
  h_on_line : 
    (P.2 = P.1 + l.k) ∧
    (Q.2 = Q.1 + l.k)
  h_R : R = (0, l.k)
  h_dot_product : P.1 * Q.1 + P.2 * Q.2 = -3
  h_vector_ratio : (Q.1 - P.1, Q.2 - P.2) = (4 * (Q.1 - R.1), 4 * (Q.2 - R.2))

theorem hyperbola_line_intersection 
  (h : Hyperbola) (l : Line) (pts : Points h l) :
  (l.k = 1 ∨ l.k = -1) ∧ h.a = 1 := by sorry

#check hyperbola_line_intersection

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l2593_259323


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l2593_259318

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - 12*x^2 + 35*x + 48 < 0 ↔ x ∈ Set.Ioo (-1 : ℝ) 3 ∪ Set.Ioi 16 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l2593_259318


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l2593_259392

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x => Real.sin (x + Real.pi / 18) + Real.cos (x - Real.pi / 9)
  ∃ M : ℝ, (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l2593_259392


namespace NUMINAMATH_CALUDE_factorial_ratio_l2593_259358

theorem factorial_ratio : Nat.factorial 13 / Nat.factorial 12 = 13 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2593_259358


namespace NUMINAMATH_CALUDE_cheese_problem_l2593_259331

theorem cheese_problem (total_rats : ℕ) (cheese_first_night : ℕ) (rats_second_night : ℕ) :
  total_rats > rats_second_night →
  cheese_first_night = 10 →
  rats_second_night = 7 →
  (cheese_first_night : ℚ) / total_rats = 2 * ((1 : ℚ) / total_rats) →
  (∃ (original_cheese : ℕ), original_cheese = cheese_first_night + 1) :=
by
  sorry

#check cheese_problem

end NUMINAMATH_CALUDE_cheese_problem_l2593_259331


namespace NUMINAMATH_CALUDE_inequality_condition_not_sufficient_l2593_259381

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) → 0 ≤ a ∧ a < 4 :=
sorry

theorem not_sufficient : 
  ∃ a : ℝ, 0 ≤ a ∧ a < 4 ∧ ∃ x : ℝ, a * x^2 - a * x + 1 ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_not_sufficient_l2593_259381


namespace NUMINAMATH_CALUDE_exists_a_for_min_g_zero_l2593_259377

-- Define the function f
def f (x : ℝ) : ℝ := x^(3/2)

-- Define the function g
def g (a x : ℝ) : ℝ := x + a * (f x)^(1/3)

-- State the theorem
theorem exists_a_for_min_g_zero :
  (∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂) →  -- f is increasing
  ∃ a : ℝ, (∀ x ∈ Set.Icc 1 9, g a x ≥ 0) ∧ 
           (∃ x ∈ Set.Icc 1 9, g a x = 0) ∧
           a = -1 :=
by sorry

end NUMINAMATH_CALUDE_exists_a_for_min_g_zero_l2593_259377


namespace NUMINAMATH_CALUDE_M_intersect_N_l2593_259303

def M : Set ℕ := {0, 2, 4}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem M_intersect_N : M ∩ N = {0, 4} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l2593_259303


namespace NUMINAMATH_CALUDE_dvd_sales_l2593_259327

theorem dvd_sales (dvd cd : ℕ) : 
  dvd = (1.6 : ℝ) * (cd : ℝ) →
  dvd + cd = 273 →
  dvd = 168 := by
  sorry

end NUMINAMATH_CALUDE_dvd_sales_l2593_259327


namespace NUMINAMATH_CALUDE_probability_both_selected_l2593_259391

/-- The probability of both Ram and Ravi being selected in an exam -/
theorem probability_both_selected (prob_ram prob_ravi : ℚ) 
  (h_ram : prob_ram = 4 / 7)
  (h_ravi : prob_ravi = 1 / 5) :
  prob_ram * prob_ravi = 4 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l2593_259391


namespace NUMINAMATH_CALUDE_pet_food_discount_l2593_259349

theorem pet_food_discount (msrp : ℝ) (regular_discount_max : ℝ) (additional_discount : ℝ)
  (h1 : msrp = 30)
  (h2 : regular_discount_max = 0.3)
  (h3 : additional_discount = 0.2) :
  msrp * (1 - regular_discount_max) * (1 - additional_discount) = 16.8 :=
by sorry

end NUMINAMATH_CALUDE_pet_food_discount_l2593_259349


namespace NUMINAMATH_CALUDE_two_letter_selection_count_l2593_259322

def word : String := "УЧЕБНИК"

def is_vowel (c : Char) : Bool :=
  c = 'У' || c = 'Е' || c = 'И'

def is_consonant (c : Char) : Bool :=
  c = 'Ч' || c = 'Б' || c = 'Н' || c = 'К'

def count_vowels (s : String) : Nat :=
  s.toList.filter is_vowel |>.length

def count_consonants (s : String) : Nat :=
  s.toList.filter is_consonant |>.length

theorem two_letter_selection_count :
  count_vowels word * count_consonants word = 12 :=
by sorry

end NUMINAMATH_CALUDE_two_letter_selection_count_l2593_259322


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2593_259363

-- Define the surface area of the cube
def surface_area : ℝ := 1350

-- Theorem stating the relationship between surface area and volume
theorem cube_volume_from_surface_area :
  ∃ (side_length : ℝ), 
    surface_area = 6 * side_length^2 ∧
    side_length > 0 ∧
    side_length^3 = 3375 := by
  sorry


end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2593_259363


namespace NUMINAMATH_CALUDE_special_circle_equation_l2593_259308

/-- A circle with center (2, -3) and a diameter with endpoints on the x-axis and y-axis -/
structure SpecialCircle where
  center : ℝ × ℝ
  center_x : center.1 = 2
  center_y : center.2 = -3
  diameter_endpoint1 : ℝ × ℝ
  diameter_endpoint2 : ℝ × ℝ
  endpoint1_on_x_axis : diameter_endpoint1.2 = 0
  endpoint2_on_y_axis : diameter_endpoint2.1 = 0
  is_diameter : (diameter_endpoint1.1 - diameter_endpoint2.1)^2 + (diameter_endpoint1.2 - diameter_endpoint2.2)^2 = 4 * ((center.1 - diameter_endpoint1.1)^2 + (center.2 - diameter_endpoint1.2)^2)

/-- The equation of the special circle -/
def circle_equation (c : SpecialCircle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = 13

/-- Theorem stating that the equation of the special circle is (x-2)^2 + (y+3)^2 = 13 -/
theorem special_circle_equation (c : SpecialCircle) :
  ∀ x y : ℝ, circle_equation c x y ↔ (x - 2)^2 + (y + 3)^2 = 13 :=
sorry

end NUMINAMATH_CALUDE_special_circle_equation_l2593_259308


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2593_259320

theorem absolute_value_inequality (x : ℝ) :
  |x - 2| + |x + 3| > 7 ↔ x < -4 ∨ x > 3 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2593_259320


namespace NUMINAMATH_CALUDE_student_rank_from_right_l2593_259386

theorem student_rank_from_right 
  (total_students : ℕ) 
  (rank_from_left : ℕ) 
  (h1 : total_students = 21) 
  (h2 : rank_from_left = 5) : 
  total_students - rank_from_left + 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_from_right_l2593_259386


namespace NUMINAMATH_CALUDE_tan_45_degrees_l2593_259339

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l2593_259339


namespace NUMINAMATH_CALUDE_find_z_l2593_259357

/-- A structure representing the relationship between x, y, and z. -/
structure Relationship where
  x : ℝ
  y : ℝ
  z : ℝ
  k : ℝ
  prop : y = k * x^2 / z

/-- The theorem statement -/
theorem find_z (r : Relationship) (h1 : r.y = 8) (h2 : r.x = 2) (h3 : r.z = 4)
    (h4 : r.x = 4) (h5 : r.y = 72) : r.z = 16/9 := by
  sorry


end NUMINAMATH_CALUDE_find_z_l2593_259357


namespace NUMINAMATH_CALUDE_smallest_number_l2593_259309

theorem smallest_number (a b c d : ℝ) : 
  a = 3.25 → 
  b = 3.26 → 
  c = 3 + 1 / 5 → 
  d = 15 / 4 → 
  c < a ∧ c < b ∧ c < d := by sorry

end NUMINAMATH_CALUDE_smallest_number_l2593_259309


namespace NUMINAMATH_CALUDE_expression_value_l2593_259394

theorem expression_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2593_259394


namespace NUMINAMATH_CALUDE_distance_to_focus_l2593_259302

/-- Given a parabola y² = 8x and a point P on it, prove that the distance from P to the focus is 10 -/
theorem distance_to_focus (x₀ : ℝ) : 
  8^2 = 8 * x₀ →  -- P(x₀, 8) is on the parabola y² = 8x
  Real.sqrt ((x₀ - 2)^2 + 8^2) = 10 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l2593_259302


namespace NUMINAMATH_CALUDE_range_of_a_l2593_259311

-- Define propositions A and B
def propA (x : ℝ) : Prop := |x - 1| < 3
def propB (x a : ℝ) : Prop := (x + 2) * (x + a) < 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, propA x → propB x a) ∧ 
  (∃ x, propB x a ∧ ¬propA x)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ a < -4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2593_259311


namespace NUMINAMATH_CALUDE_store_price_difference_l2593_259319

/-- Calculates the final price after applying a discount percentage to a full price -/
def final_price (full_price : ℚ) (discount_percent : ℚ) : ℚ :=
  full_price * (1 - discount_percent / 100)

/-- Proves that Store A's smartphone is $2 cheaper than Store B's after discounts -/
theorem store_price_difference (store_a_full_price store_b_full_price : ℚ)
  (store_a_discount store_b_discount : ℚ)
  (h1 : store_a_full_price = 125)
  (h2 : store_b_full_price = 130)
  (h3 : store_a_discount = 8)
  (h4 : store_b_discount = 10) :
  final_price store_b_full_price store_b_discount -
  final_price store_a_full_price store_a_discount = 2 := by
sorry

end NUMINAMATH_CALUDE_store_price_difference_l2593_259319


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2593_259388

theorem quadratic_factorization (y : ℝ) : y^2 + 14*y + 40 = (y + 4) * (y + 10) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2593_259388


namespace NUMINAMATH_CALUDE_harkamal_fruit_purchase_l2593_259373

/-- The total amount paid by Harkamal for his fruit purchase --/
def total_amount : ℕ := by sorry

theorem harkamal_fruit_purchase :
  let grapes_quantity : ℕ := 8
  let grapes_price : ℕ := 80
  let mangoes_quantity : ℕ := 9
  let mangoes_price : ℕ := 55
  let apples_quantity : ℕ := 6
  let apples_price : ℕ := 120
  let oranges_quantity : ℕ := 4
  let oranges_price : ℕ := 75
  total_amount = grapes_quantity * grapes_price +
                 mangoes_quantity * mangoes_price +
                 apples_quantity * apples_price +
                 oranges_quantity * oranges_price :=
by sorry

end NUMINAMATH_CALUDE_harkamal_fruit_purchase_l2593_259373


namespace NUMINAMATH_CALUDE_count_fives_in_S_l2593_259354

/-- The sum of an arithmetic sequence with first term 1, common difference 9, and last term 10^2013 -/
def S : ℕ := (1 + 10^2013) * ((10^2013 + 8) / 18)

/-- Counts the occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

theorem count_fives_in_S : countDigit S 5 = 4022 := by sorry

end NUMINAMATH_CALUDE_count_fives_in_S_l2593_259354


namespace NUMINAMATH_CALUDE_first_group_size_l2593_259301

/-- The amount of work done by one person in one day -/
def work_per_person_per_day : ℝ := 1

/-- The number of days to complete the work -/
def days : ℕ := 7

/-- The number of persons in the second group -/
def persons_second_group : ℕ := 9

/-- The amount of work completed by the first group -/
def work_first_group : ℕ := 7

/-- The amount of work completed by the second group -/
def work_second_group : ℕ := 9

/-- The number of persons in the first group -/
def persons_first_group : ℕ := 9

theorem first_group_size :
  persons_first_group * days * work_per_person_per_day = work_first_group ∧
  persons_second_group * days * work_per_person_per_day = work_second_group →
  persons_first_group = 9 := by
sorry

end NUMINAMATH_CALUDE_first_group_size_l2593_259301


namespace NUMINAMATH_CALUDE_complex_inequality_l2593_259361

theorem complex_inequality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) :
  Complex.abs (a - b) ≥ (1/2 : ℝ) * (Complex.abs a + Complex.abs b) * Complex.abs ((a / Complex.abs a) - (b / Complex.abs b)) ∧
  (Complex.abs (a - b) = (1/2 : ℝ) * (Complex.abs a + Complex.abs b) * Complex.abs ((a / Complex.abs a) - (b / Complex.abs b)) ↔ Complex.abs a = Complex.abs b) :=
by sorry

end NUMINAMATH_CALUDE_complex_inequality_l2593_259361


namespace NUMINAMATH_CALUDE_markers_per_box_l2593_259356

theorem markers_per_box (total_students : ℕ) (boxes : ℕ) 
  (group1_students : ℕ) (group1_markers : ℕ)
  (group2_students : ℕ) (group2_markers : ℕ)
  (group3_markers : ℕ) :
  total_students = 30 →
  boxes = 22 →
  group1_students = 10 →
  group1_markers = 2 →
  group2_students = 15 →
  group2_markers = 4 →
  group3_markers = 6 →
  (boxes : ℚ) * ((group1_students * group1_markers + 
                  group2_students * group2_markers + 
                  (total_students - group1_students - group2_students) * group3_markers) / boxes : ℚ) = 
  (boxes : ℚ) * (5 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_markers_per_box_l2593_259356


namespace NUMINAMATH_CALUDE_square_perimeter_l2593_259333

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 325 →
  area = side ^ 2 →
  perimeter = 4 * side →
  perimeter = 20 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2593_259333


namespace NUMINAMATH_CALUDE_target_line_is_correct_l2593_259310

-- Define the line we're looking for
def target_line (x y : ℝ) : Prop := y = x + 1

-- Define the given line x + y = 0
def given_line (x y : ℝ) : Prop := x + y = 0

-- Define perpendicularity of two lines
def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄, 
    f x₁ y₁ ∧ f x₂ y₂ ∧ g x₃ y₃ ∧ g x₄ y₄ ∧ 
    x₁ ≠ x₂ ∧ x₃ ≠ x₄ → 
    (y₂ - y₁) / (x₂ - x₁) * (y₄ - y₃) / (x₄ - x₃) = -1

-- Theorem statement
theorem target_line_is_correct : 
  target_line (-1) 0 ∧ 
  perpendicular target_line given_line :=
sorry

end NUMINAMATH_CALUDE_target_line_is_correct_l2593_259310


namespace NUMINAMATH_CALUDE_ice_cream_cost_theorem_l2593_259374

/-- Ice cream shop prices and orders -/
structure IceCreamShop where
  chocolate_price : ℝ
  vanilla_price : ℝ
  strawberry_price : ℝ
  mint_price : ℝ
  waffle_cone_price : ℝ
  chocolate_chips_price : ℝ
  fudge_price : ℝ
  whipped_cream_price : ℝ

/-- Calculate the cost of Pierre's order -/
def pierre_order_cost (shop : IceCreamShop) : ℝ :=
  2 * shop.chocolate_price + shop.mint_price + shop.waffle_cone_price + shop.chocolate_chips_price

/-- Calculate the cost of Pierre's mother's order -/
def mother_order_cost (shop : IceCreamShop) : ℝ :=
  2 * shop.vanilla_price + shop.strawberry_price + shop.mint_price + 
  shop.waffle_cone_price + shop.fudge_price + shop.whipped_cream_price

/-- The total cost of both orders -/
def total_cost (shop : IceCreamShop) : ℝ :=
  pierre_order_cost shop + mother_order_cost shop

/-- Theorem stating that the total cost is $21.65 -/
theorem ice_cream_cost_theorem (shop : IceCreamShop) 
  (h1 : shop.chocolate_price = 2.50)
  (h2 : shop.vanilla_price = 2.00)
  (h3 : shop.strawberry_price = 2.25)
  (h4 : shop.mint_price = 2.20)
  (h5 : shop.waffle_cone_price = 1.50)
  (h6 : shop.chocolate_chips_price = 1.00)
  (h7 : shop.fudge_price = 1.25)
  (h8 : shop.whipped_cream_price = 0.75) :
  total_cost shop = 21.65 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_theorem_l2593_259374


namespace NUMINAMATH_CALUDE_shaded_region_area_l2593_259350

/-- The area of a shaded region formed by the intersection of two circles -/
theorem shaded_region_area (r : ℝ) (h : r = 5) : 
  (2 * (π * r^2 / 4) - r^2) = (50 * π - 100) / 4 := by
  sorry

#check shaded_region_area

end NUMINAMATH_CALUDE_shaded_region_area_l2593_259350


namespace NUMINAMATH_CALUDE_fraction_equality_l2593_259372

theorem fraction_equality (a b c : ℝ) (h1 : a/3 = b) (h2 : b/4 = c) : a*b/c^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2593_259372


namespace NUMINAMATH_CALUDE_ratio_inequality_l2593_259369

theorem ratio_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 2*b + 3*c)^2 / (a^2 + 2*b^2 + 3*c^2) ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_inequality_l2593_259369


namespace NUMINAMATH_CALUDE_minimum_excellence_rate_l2593_259347

theorem minimum_excellence_rate (math_rate : ℝ) (chinese_rate : ℝ) 
  (h1 : math_rate = 0.7) (h2 : chinese_rate = 0.25) : 
  math_rate * chinese_rate = 0.175 := by
  sorry

end NUMINAMATH_CALUDE_minimum_excellence_rate_l2593_259347


namespace NUMINAMATH_CALUDE_bowTie_equation_solution_l2593_259368

-- Define the bow tie operation
noncomputable def bowTie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem bowTie_equation_solution (y : ℝ) : bowTie 4 y = 10 → y = 30 := by
  sorry

end NUMINAMATH_CALUDE_bowTie_equation_solution_l2593_259368


namespace NUMINAMATH_CALUDE_prob_equal_prob_first_value_prob_second_value_l2593_259351

/-- Represents the number of classes -/
def total_classes : ℕ := 10

/-- Represents the specific class we're interested in (Class 5) -/
def target_class : ℕ := 5

/-- The probability of drawing the target class first -/
def prob_first : ℚ := 1 / total_classes

/-- The probability of drawing the target class second -/
def prob_second : ℚ := 1 / total_classes

/-- Theorem stating that the probabilities of drawing the target class first and second are equal -/
theorem prob_equal : prob_first = prob_second := by sorry

/-- Theorem stating that the probability of drawing the target class first is 1/10 -/
theorem prob_first_value : prob_first = 1 / 10 := by sorry

/-- Theorem stating that the probability of drawing the target class second is 1/10 -/
theorem prob_second_value : prob_second = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_prob_equal_prob_first_value_prob_second_value_l2593_259351


namespace NUMINAMATH_CALUDE_ladder_problem_l2593_259370

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 8.5)
  (h2 : height = 7.5) :
  ∃ base : ℝ, base = 4 ∧ base^2 + height^2 = ladder_length^2 :=
sorry

end NUMINAMATH_CALUDE_ladder_problem_l2593_259370


namespace NUMINAMATH_CALUDE_inequality_with_negative_square_l2593_259325

theorem inequality_with_negative_square (a b c : ℝ) 
  (h1 : a < b) (h2 : c < 0) : a * c^2 < b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_with_negative_square_l2593_259325


namespace NUMINAMATH_CALUDE_unique_reverse_multiple_of_nine_l2593_259397

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digits_reversed (n : ℕ) : ℕ :=
  let d₁ := n / 10000
  let d₂ := (n / 1000) % 10
  let d₃ := (n / 100) % 10
  let d₄ := (n / 10) % 10
  let d₅ := n % 10
  d₅ * 10000 + d₄ * 1000 + d₃ * 100 + d₂ * 10 + d₁

theorem unique_reverse_multiple_of_nine :
  ∃! n : ℕ, is_five_digit n ∧ 9 * n = digits_reversed n := by sorry

end NUMINAMATH_CALUDE_unique_reverse_multiple_of_nine_l2593_259397


namespace NUMINAMATH_CALUDE_distinct_arrangements_of_basic_l2593_259315

theorem distinct_arrangements_of_basic (n : ℕ) (h : n = 5) : 
  Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_of_basic_l2593_259315


namespace NUMINAMATH_CALUDE_quadratic_inequality_k_range_l2593_259338

theorem quadratic_inequality_k_range :
  (∀ x : ℝ, ∀ k : ℝ, k * x^2 - k * x + 1 > 0) ↔ k ∈ Set.Ici 0 ∩ Set.Iio 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_k_range_l2593_259338


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2593_259366

/-- A geometric sequence with first term 3 and the sum of first, third, and fifth terms equal to 21 has its third term equal to 6. -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n = 3 * q^(n-1)) →  -- Definition of geometric sequence
  a 1 = 3 →                   -- First term is 3
  a 1 + a 3 + a 5 = 21 →      -- Sum of first, third, and fifth terms is 21
  a 3 = 6 :=                  -- Conclusion: third term is 6
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2593_259366


namespace NUMINAMATH_CALUDE_man_downstream_speed_l2593_259353

/-- Given a man's upstream speed and the speed of a stream, calculate his downstream speed. -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  upstream_speed + 2 * stream_speed

/-- Theorem: Given a man's upstream speed of 8 kmph and a stream speed of 2 kmph, his downstream speed is 12 kmph. -/
theorem man_downstream_speed :
  downstream_speed 8 2 = 12 := by
  sorry

#eval downstream_speed 8 2

end NUMINAMATH_CALUDE_man_downstream_speed_l2593_259353


namespace NUMINAMATH_CALUDE_at_op_properties_l2593_259330

def at_op (a b : ℝ) : ℝ := a * b - 1

theorem at_op_properties (x y z : ℝ) : 
  (¬ (at_op x (y + z) = at_op x y + at_op x z)) ∧ 
  (¬ (x + at_op y z = at_op (x + y) (x + z))) ∧ 
  (¬ (at_op x (at_op y z) = at_op (at_op x y) (at_op x z))) :=
sorry

end NUMINAMATH_CALUDE_at_op_properties_l2593_259330


namespace NUMINAMATH_CALUDE_problem_statement_l2593_259342

/-- Given positive real numbers a and b that sum to 1, this theorem states:
    1. The minimum value of m such that ab ≤ m always holds is 1/4.
    2. The range of x such that (4/a) + (1/b) ≥ |2x-1| - |x+2| always holds is [-6, 12]. -/
theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ m : ℝ, m = 1/4 ∧ ∀ m' : ℝ, (a * b ≤ m' → m ≤ m')) ∧ 
  (∀ x : ℝ, (4/a + 1/b ≥ |2*x - 1| - |x + 2|) ↔ -6 ≤ x ∧ x ≤ 12) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2593_259342


namespace NUMINAMATH_CALUDE_double_negation_l2593_259332

theorem double_negation (x : ℝ) : -(-x) = x := by
  sorry

end NUMINAMATH_CALUDE_double_negation_l2593_259332


namespace NUMINAMATH_CALUDE_circle_radius_when_area_circumference_ratio_is_15_l2593_259367

theorem circle_radius_when_area_circumference_ratio_is_15 
  (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) : 
  ∃ r : ℝ, r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_when_area_circumference_ratio_is_15_l2593_259367


namespace NUMINAMATH_CALUDE_cone_base_circumference_l2593_259364

/-- The circumference of the base of a right circular cone with volume 24π cubic centimeters and height 6 cm is 4√3π cm. -/
theorem cone_base_circumference :
  ∀ (V h r : ℝ),
  V = 24 * Real.pi ∧
  h = 6 ∧
  V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 4 * Real.sqrt 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l2593_259364


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_twelve_l2593_259312

theorem sum_of_roots_eq_twelve : ∃ (x₁ x₂ : ℝ), 
  (x₁ - 6)^2 = 16 ∧ 
  (x₂ - 6)^2 = 16 ∧ 
  x₁ + x₂ = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_twelve_l2593_259312


namespace NUMINAMATH_CALUDE_bicentric_quadrilateral_segment_difference_l2593_259396

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (a b c d : ℝ)

-- Define the properties of the quadrilateral
def is_cyclic_bicentric (q : Quadrilateral) : Prop :=
  -- The quadrilateral is cyclic (inscribed in a circle)
  ∃ (r : ℝ), r > 0 ∧ 
  -- The quadrilateral has an incircle
  ∃ (s : ℝ), s > 0 ∧
  -- Additional conditions for cyclic and bicentric quadrilaterals
  -- (These are simplified representations and may need more detailed conditions)
  q.a + q.c = q.b + q.d

-- Define the theorem
theorem bicentric_quadrilateral_segment_difference 
  (q : Quadrilateral) 
  (h : is_cyclic_bicentric q) 
  (h_sides : q.a = 70 ∧ q.b = 90 ∧ q.c = 130 ∧ q.d = 110) : 
  ∃ (x y : ℝ), x + y = 130 ∧ |x - y| = 13 := by
  sorry

end NUMINAMATH_CALUDE_bicentric_quadrilateral_segment_difference_l2593_259396


namespace NUMINAMATH_CALUDE_length_of_GH_l2593_259344

-- Define the lengths of the segments
def AB : ℝ := 11
def CD : ℝ := 5
def FE : ℝ := 13

-- Define the length of GH as the sum of AB, CD, and FE
def GH : ℝ := AB + CD + FE

-- Theorem statement
theorem length_of_GH : GH = 29 := by sorry

end NUMINAMATH_CALUDE_length_of_GH_l2593_259344


namespace NUMINAMATH_CALUDE_equal_selection_probability_l2593_259326

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- The probability of an individual being selected given a sampling method -/
def selectionProbability (N n : ℕ) (method : SamplingMethod) : ℝ :=
  sorry

theorem equal_selection_probability (N n : ℕ) :
  ∀ (m₁ m₂ : SamplingMethod), selectionProbability N n m₁ = selectionProbability N n m₂ :=
sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l2593_259326
