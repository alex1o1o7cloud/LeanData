import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l3662_366237

theorem sqrt_eight_and_nine_sixteenths : 
  Real.sqrt (8 + 9/16) = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l3662_366237


namespace NUMINAMATH_CALUDE_joan_wednesday_spending_l3662_366240

/-- The number of half-dollars Joan spent on Wednesday -/
def wednesday_half_dollars : ℕ := 18 - 14

/-- The total amount Joan spent in half-dollars -/
def total_half_dollars : ℕ := 18

/-- The number of half-dollars Joan spent on Thursday -/
def thursday_half_dollars : ℕ := 14

theorem joan_wednesday_spending :
  wednesday_half_dollars = 4 :=
by sorry

end NUMINAMATH_CALUDE_joan_wednesday_spending_l3662_366240


namespace NUMINAMATH_CALUDE_michaels_estimate_greater_l3662_366266

theorem michaels_estimate_greater (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxy : x > y) : 
  3 * ((x + z) - (y - 2 * z)) > 3 * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_michaels_estimate_greater_l3662_366266


namespace NUMINAMATH_CALUDE_mod_pow_98_50_100_l3662_366263

theorem mod_pow_98_50_100 : 98^50 % 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_mod_pow_98_50_100_l3662_366263


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3662_366249

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) = q * a n) 
  (h_arithmetic : a 3 + a 4 = a 5) :
  q = (Real.sqrt 5 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3662_366249


namespace NUMINAMATH_CALUDE_ellipse_properties_l3662_366268

/-- Properties of an ellipse with given parameters -/
theorem ellipse_properties :
  let e : ℝ := 1/2  -- eccentricity
  let c : ℝ := 1    -- half the distance between foci
  let a : ℝ := 2    -- semi-major axis
  let b : ℝ := Real.sqrt 3  -- semi-minor axis
  let F₁ : ℝ × ℝ := (-1, 0)  -- left focus
  let A : ℝ × ℝ := (-2, 0)  -- left vertex
  ∀ x y : ℝ,
    (x^2 / 4 + y^2 / 3 = 1) →  -- point (x,y) is on the ellipse
    (0 ≤ (x + 1) * (x + 2) + y^2) ∧
    ((x + 1) * (x + 2) + y^2 ≤ 12) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3662_366268


namespace NUMINAMATH_CALUDE_inequality_proof_l3662_366225

theorem inequality_proof (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_squares : a^2 + b^2 + c^2 = 3) : 
  (a / (a + 5)) + (b / (b + 5)) + (c / (c + 5)) ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3662_366225


namespace NUMINAMATH_CALUDE_sector_arc_length_l3662_366210

/-- Given a circular sector with area 60π cm² and central angle 150°, its arc length is 10π cm. -/
theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) : 
  area = 60 * Real.pi ∧ angle = 150 → arc_length = 10 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3662_366210


namespace NUMINAMATH_CALUDE_number_difference_l3662_366209

theorem number_difference (L S : ℕ) (h1 : L = 1608) (h2 : L = 6 * S + 15) : L - S = 1343 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3662_366209


namespace NUMINAMATH_CALUDE_reciprocal_roots_sum_l3662_366264

theorem reciprocal_roots_sum (α β : ℝ) : 
  (∃ a b : ℝ, (7 * a^2 + 2 * a + 6 = 0) ∧ 
              (7 * b^2 + 2 * b + 6 = 0) ∧ 
              (α = 1 / a) ∧ 
              (β = 1 / b)) →
  α + β = -1/3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_roots_sum_l3662_366264


namespace NUMINAMATH_CALUDE_power_calculation_l3662_366226

theorem power_calculation : (-2 : ℝ)^2023 * (1/2 : ℝ)^2022 = -2 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l3662_366226


namespace NUMINAMATH_CALUDE_arithmetic_sequence_prime_divisibility_l3662_366272

theorem arithmetic_sequence_prime_divisibility 
  (n : ℕ) 
  (a : ℕ → ℕ) 
  (h_n : n ≥ 2021) 
  (h_arith : ∀ i j, i < j → j ≤ n → a j - a i = (j - i) * (a 2 - a 1))
  (h_inc : ∀ i j, i < j → j ≤ n → a i < a j)
  (h_first : a 1 > 2021)
  (h_prime : ∀ i, 1 ≤ i → i ≤ n → Nat.Prime (a i)) :
  ∀ p, p < 2021 → Nat.Prime p → (a 2 - a 1) % p = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_prime_divisibility_l3662_366272


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_l3662_366255

theorem dormitory_to_city_distance : ∃ (d : ℝ), 
  (1/5 : ℝ) * d + (2/3 : ℝ) * d + 14 = d ∧ d = 105 := by
  sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_l3662_366255


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l3662_366270

-- Define the hyperbola equation
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (2 - k) + y^2 / (k - 1) = 1

-- Define the condition for the real axis to be on the x-axis
def real_axis_on_x (k : ℝ) : Prop :=
  (2 - k > 0) ∧ (k - 1 < 0)

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, hyperbola_equation x y k ∧ real_axis_on_x k) ↔ k ∈ Set.Iio 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l3662_366270


namespace NUMINAMATH_CALUDE_remainder_17_power_1999_mod_29_l3662_366215

theorem remainder_17_power_1999_mod_29 : 17^1999 % 29 = 17 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_power_1999_mod_29_l3662_366215


namespace NUMINAMATH_CALUDE_complex_division_result_l3662_366253

theorem complex_division_result : ((-2 : ℂ) - I) / I = -1 + 2*I := by sorry

end NUMINAMATH_CALUDE_complex_division_result_l3662_366253


namespace NUMINAMATH_CALUDE_shirt_tie_combination_count_l3662_366204

/-- The number of possible shirt-and-tie combinations given:
  * total_shirts: The total number of shirts
  * total_ties: The total number of ties
  * incompatible_shirts: The number of shirts that are incompatible with some ties
  * incompatible_ties: The number of ties that are incompatible with some shirts
-/
def shirt_tie_combinations (total_shirts : ℕ) (total_ties : ℕ) 
  (incompatible_shirts : ℕ) (incompatible_ties : ℕ) : ℕ :=
  total_shirts * total_ties - incompatible_shirts * incompatible_ties

/-- Theorem stating that with 8 shirts, 7 ties, and 1 shirt incompatible with 2 ties,
    the total number of possible shirt-and-tie combinations is 54. -/
theorem shirt_tie_combination_count :
  shirt_tie_combinations 8 7 1 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_combination_count_l3662_366204


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l3662_366297

theorem smallest_b_for_factorization (b : ℕ) : b = 121 ↔ 
  (b > 0 ∧ 
   ∃ (r s : ℕ), r * s = 2020 ∧ r > s ∧
   ∀ (x : ℤ), x^2 + b*x + 2020 = (x + r) * (x + s) ∧
   ∀ (b' : ℕ), b' > 0 → 
     (∃ (r' s' : ℕ), r' * s' = 2020 ∧ r' > s' ∧
     ∀ (x : ℤ), x^2 + b'*x + 2020 = (x + r') * (x + s')) →
     b ≤ b') := by
sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l3662_366297


namespace NUMINAMATH_CALUDE_average_PQR_l3662_366283

theorem average_PQR (P Q R : ℚ) 
  (eq1 : 1001 * R - 3003 * P = 6006)
  (eq2 : 2002 * Q + 4004 * P = 8008) :
  (P + Q + R) / 3 = 2 * (P + 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_PQR_l3662_366283


namespace NUMINAMATH_CALUDE_max_additional_plates_is_24_l3662_366235

def initial_plates : ℕ := 3 * 2 * 4

def scenario1 : ℕ := (3 + 2) * 2 * 4
def scenario2 : ℕ := 3 * 2 * (4 + 2)
def scenario3 : ℕ := (3 + 1) * 2 * (4 + 1)
def scenario4 : ℕ := (3 + 1) * (2 + 1) * 4

def max_additional_plates : ℕ := max scenario1 (max scenario2 (max scenario3 scenario4)) - initial_plates

theorem max_additional_plates_is_24 : max_additional_plates = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_additional_plates_is_24_l3662_366235


namespace NUMINAMATH_CALUDE_inverse_prop_parallel_lines_interior_angles_l3662_366207

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- Definition of interior alternate angles -/
def interior_alternate_angles_equal (l1 l2 : Line) : Prop := sorry

/-- The inverse proposition of a statement "if P, then Q" is "if Q, then P" -/
def inverse_proposition (P Q : Prop) : Prop :=
  (Q → P) = (¬P → ¬Q)

/-- Theorem stating the inverse proposition of the given statement -/
theorem inverse_prop_parallel_lines_interior_angles :
  inverse_proposition
    (∀ l1 l2 : Line, parallel l1 l2 → interior_alternate_angles_equal l1 l2)
    (∀ l1 l2 : Line, interior_alternate_angles_equal l1 l2 → parallel l1 l2) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_prop_parallel_lines_interior_angles_l3662_366207


namespace NUMINAMATH_CALUDE_train_speed_l3662_366261

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 140)
  (h2 : bridge_length = 235)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3662_366261


namespace NUMINAMATH_CALUDE_log_9_81_equals_2_l3662_366245

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_9_81_equals_2 : log 9 81 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_9_81_equals_2_l3662_366245


namespace NUMINAMATH_CALUDE_employees_in_all_three_proof_l3662_366295

/-- The number of employees trained to work in all 3 restaurants -/
def employees_in_all_three : ℕ := 2

theorem employees_in_all_three_proof :
  let total_employees : ℕ := 39
  let min_restaurants : ℕ := 1
  let max_restaurants : ℕ := 3
  let family_buffet : ℕ := 15
  let dining_room : ℕ := 18
  let snack_bar : ℕ := 12
  let in_two_restaurants : ℕ := 4
  employees_in_all_three = 
    total_employees + employees_in_all_three - in_two_restaurants - 
    (family_buffet + dining_room + snack_bar) := by
  sorry

#check employees_in_all_three_proof

end NUMINAMATH_CALUDE_employees_in_all_three_proof_l3662_366295


namespace NUMINAMATH_CALUDE_average_of_remaining_two_l3662_366259

theorem average_of_remaining_two (total_avg : ℝ) (avg1 : ℝ) (avg2 : ℝ) :
  total_avg = 3.95 →
  avg1 = 4.2 →
  avg2 = 3.8000000000000007 →
  (6 * total_avg - 2 * avg1 - 2 * avg2) / 2 = 3.85 := by
sorry


end NUMINAMATH_CALUDE_average_of_remaining_two_l3662_366259


namespace NUMINAMATH_CALUDE_perpendicular_tangents_trajectory_l3662_366217

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a tangent line from P to the unit circle
def is_tangent (P : Point) (A : Point) : Prop :=
  unit_circle A.x A.y ∧ 
  (P.x - A.x) * A.x + (P.y - A.y) * A.y = 0

-- State the theorem
theorem perpendicular_tangents_trajectory :
  ∀ P : Point,
  (∃ A B : Point,
    is_tangent P A ∧
    is_tangent P B ∧
    (P.x - A.x) * (P.x - B.x) + (P.y - A.y) * (P.y - B.y) = 0) →
  P.x^2 + P.y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_trajectory_l3662_366217


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l3662_366277

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem fifth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : arithmetic_sequence a d)
  (h2 : a 1 = 2)
  (h3 : d = 1) : 
  a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l3662_366277


namespace NUMINAMATH_CALUDE_no_valid_function_l3662_366236

/-- The set M = {0, 1, 2, ..., 2022} -/
def M : Set Nat := Finset.range 2023

/-- The theorem stating that no function f satisfies both required conditions -/
theorem no_valid_function :
  ¬∃ (f : M → M → M),
    (∀ (a b : M), f a (f b a) = b) ∧
    (∀ (x : M), f x x ≠ x) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_function_l3662_366236


namespace NUMINAMATH_CALUDE_associate_professor_pencils_l3662_366211

theorem associate_professor_pencils :
  ∀ (A B P : ℕ),
    A + B = 8 →
    P * A + B = 10 →
    A + 2 * B = 14 →
    P = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_associate_professor_pencils_l3662_366211


namespace NUMINAMATH_CALUDE_third_number_proof_l3662_366274

theorem third_number_proof (sum : ℝ) (a b c : ℝ) (h : sum = a + b + c + 0.217) :
  sum - a - b - c = 0.217 :=
by sorry

end NUMINAMATH_CALUDE_third_number_proof_l3662_366274


namespace NUMINAMATH_CALUDE_min_value_theorem_l3662_366267

theorem min_value_theorem (x : ℝ) (h1 : x > 0) (h2 : Real.log x + 1 ≤ x) :
  (x^2 - Real.log x + x) / x ≥ 2 ∧
  ((x^2 - Real.log x + x) / x = 2 ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3662_366267


namespace NUMINAMATH_CALUDE_claire_balloons_count_l3662_366287

/-- The number of balloons Claire has at the end of the fair --/
def claire_balloons : ℕ :=
  let initial := 50
  let given_to_girl := 1
  let floated_away := 12
  let given_away_later := 9
  let grabbed_from_coworker := 11
  initial - given_to_girl - floated_away - given_away_later + grabbed_from_coworker

theorem claire_balloons_count : claire_balloons = 39 := by
  sorry

end NUMINAMATH_CALUDE_claire_balloons_count_l3662_366287


namespace NUMINAMATH_CALUDE_min_d_value_l3662_366201

theorem min_d_value (t a b d : ℕ) : 
  (3 * t = 2 * a + 2 * b + 2016) →  -- Triangle perimeter exceeds rectangle perimeter by 2016
  (t = a + d) →                     -- Triangle side exceeds one rectangle side by d
  (t = b + 2 * d) →                 -- Triangle side exceeds other rectangle side by 2d
  (a > 0 ∧ b > 0) →                 -- Rectangle has non-zero perimeter
  (∀ d' : ℕ, d' < d → 
    ¬(∃ t' a' b' : ℕ, 
      (3 * t' = 2 * a' + 2 * b' + 2016) ∧ 
      (t' = a' + d') ∧ 
      (t' = b' + 2 * d') ∧ 
      (a' > 0 ∧ b' > 0))) →
  d = 505 :=
by sorry

end NUMINAMATH_CALUDE_min_d_value_l3662_366201


namespace NUMINAMATH_CALUDE_complex_equation_real_part_condition_l3662_366202

theorem complex_equation_real_part_condition (z : ℂ) (a b : ℝ) : 
  z * (z + 2*I) * (z + 4*I) = 1001*I → 
  z = a + b*I → 
  a > 0 → 
  b > 0 → 
  a * (a^2 - b^2 - 6*b - 8) = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_real_part_condition_l3662_366202


namespace NUMINAMATH_CALUDE_no_lcm_arithmetic_progression_l3662_366218

theorem no_lcm_arithmetic_progression (n : ℕ) (h : n > 100) :
  ¬ ∃ (S : Finset ℕ) (d : ℕ) (first : ℕ),
    S.card = n ∧
    (∀ x ∈ S, ∀ y ∈ S, x ≠ y) ∧
    d > 0 ∧
    ∃ (f : Finset ℕ),
      f.card = n * (n - 1) / 2 ∧
      (∀ x ∈ S, ∀ y ∈ S, x < y → Nat.lcm x y ∈ f) ∧
      (∀ i < n * (n - 1) / 2, first + i * d ∈ f) :=
by sorry

end NUMINAMATH_CALUDE_no_lcm_arithmetic_progression_l3662_366218


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3662_366230

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x - 2

-- Define the solution set condition
def solution_set (a : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ 1/2 < x ∧ x < 2

-- Theorem statement
theorem quadratic_inequality (a : ℝ) (h : solution_set a) :
  a = -2 ∧
  ∀ x, a * x^2 - 5 * x + a^2 - 1 > 0 ↔ -1/2 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3662_366230


namespace NUMINAMATH_CALUDE_pressure_area_relation_l3662_366278

/-- Proves that given pressure P = F/S, force F = 50N, and P > 500Pa, the area S < 0.1m² -/
theorem pressure_area_relation (F S P : ℝ) : 
  F = 50 → P = F / S → P > 500 → S < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_pressure_area_relation_l3662_366278


namespace NUMINAMATH_CALUDE_notebook_cost_l3662_366220

theorem notebook_cost (notebook_cost pen_cost : ℝ) 
  (total_cost : notebook_cost + pen_cost = 3.50)
  (cost_difference : notebook_cost = pen_cost + 3) : 
  notebook_cost = 3.25 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l3662_366220


namespace NUMINAMATH_CALUDE_elizabeth_climb_time_l3662_366244

-- Define the climbing times
def tom_time : ℕ := 2 * 60  -- Tom's time in minutes
def elizabeth_time : ℕ := tom_time / 4  -- Elizabeth's time in minutes

-- State the theorem
theorem elizabeth_climb_time :
  (tom_time = 4 * elizabeth_time) →  -- Tom takes 4 times as long as Elizabeth
  (tom_time = 2 * 60) →  -- Tom takes 2 hours (120 minutes)
  elizabeth_time = 30 :=  -- Elizabeth takes 30 minutes
by
  sorry

end NUMINAMATH_CALUDE_elizabeth_climb_time_l3662_366244


namespace NUMINAMATH_CALUDE_min_value_is_nine_l3662_366276

/-- Two circles C₁ and C₂ with centers and radii -/
structure TwoCircles where
  a : ℝ
  b : ℝ
  h1 : a ≠ 0
  h2 : b ≠ 0

/-- The circles have only one common tangent -/
axiom one_common_tangent (c : TwoCircles) : 4 * c.a^2 + c.b^2 = 1

/-- The minimum value of 1/a² + 1/b² is 9 -/
theorem min_value_is_nine (c : TwoCircles) : 
  ∀ ε > 0, (1 / c.a^2 + 1 / c.b^2) > 9 - ε :=
sorry

end NUMINAMATH_CALUDE_min_value_is_nine_l3662_366276


namespace NUMINAMATH_CALUDE_train_journey_time_l3662_366288

/-- Represents the train journey from A to B -/
structure TrainJourney where
  d : ℝ  -- Total distance
  v : ℝ  -- Initial speed
  t : ℝ  -- Total scheduled time

/-- The conditions of the train journey -/
def journey_conditions (j : TrainJourney) : Prop :=
  j.d > 0 ∧ j.v > 0 ∧
  (j.d / (2 * j.v)) + 15 + (j.d / (8 * j.v)) = j.t

/-- The theorem stating that the total journey time is 40 minutes -/
theorem train_journey_time (j : TrainJourney) 
  (h : journey_conditions j) : j.t = 40 := by
  sorry

#check train_journey_time

end NUMINAMATH_CALUDE_train_journey_time_l3662_366288


namespace NUMINAMATH_CALUDE_cows_bought_calculation_l3662_366252

def cows_bought (initial : ℕ) (died : ℕ) (sold : ℕ) (increase : ℕ) (gift : ℕ) (final : ℕ) : ℕ :=
  final - (initial - died - sold + increase + gift)

theorem cows_bought_calculation :
  cows_bought 39 25 6 24 8 83 = 43 := by
  sorry

end NUMINAMATH_CALUDE_cows_bought_calculation_l3662_366252


namespace NUMINAMATH_CALUDE_boys_camp_total_l3662_366239

theorem boys_camp_total (total : ℕ) 
  (h1 : (total : ℚ) * (1/5) = (total : ℚ) * (20/100))
  (h2 : (total : ℚ) * (1/5) * (3/10) = (total : ℚ) * (1/5) * (30/100))
  (h3 : (total : ℚ) * (1/5) * (7/10) = 28) : 
  total = 200 := by
sorry

end NUMINAMATH_CALUDE_boys_camp_total_l3662_366239


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3662_366260

theorem trigonometric_product_equals_one : 
  (1 - 1 / Real.cos (30 * π / 180)) * 
  (1 + 1 / Real.sin (60 * π / 180)) * 
  (1 - 1 / Real.sin (30 * π / 180)) * 
  (1 + 1 / Real.cos (60 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3662_366260


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3662_366285

theorem rationalize_denominator :
  ∃ (A B C D E F : ℤ),
    (F > 0) ∧
    (1 / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) =
     (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F) ∧
    A = -13 ∧ B = -9 ∧ C = 3 ∧ D = 2 ∧ E = 165 ∧ F = 51 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3662_366285


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3662_366221

-- Define the function g
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then
    ⌈(Real.cos x) / (x + 3)⌉
  else if x < -3 then
    ⌊(Real.cos x) / (x + 3)⌋
  else
    0  -- This value doesn't matter as g is not defined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3662_366221


namespace NUMINAMATH_CALUDE_unique_solution_mod_125_l3662_366265

theorem unique_solution_mod_125 :
  ∃! x : ℕ, x < 125 ∧ (x^3 - 2*x + 6) % 125 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_mod_125_l3662_366265


namespace NUMINAMATH_CALUDE_geometric_sequence_a10_l3662_366248

/-- A geometric sequence with positive common ratio -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a10 (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 4 * a 8 = 2 * (a 5)^2 →
  a 2 = 1 →
  a 10 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a10_l3662_366248


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l3662_366299

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l3662_366299


namespace NUMINAMATH_CALUDE_A_and_D_independent_l3662_366203

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define events A and D
def A : Set Ω := {ω | ω.1 = 0}
def D : Set Ω := {ω | ω.1.val + ω.2.val + 2 = 7}

-- State the theorem
theorem A_and_D_independent : 
  P (A ∩ D) = P A * P D := by sorry

end NUMINAMATH_CALUDE_A_and_D_independent_l3662_366203


namespace NUMINAMATH_CALUDE_triangle_town_intersections_l3662_366212

/-- The number of intersections for n non-parallel lines in a plane where no three lines meet at a single point -/
def max_intersections (n : ℕ) : ℕ := n.choose 2

/-- Theorem: In a configuration of 10 non-parallel lines in a plane, 
    where no three lines intersect at a single point, 
    the maximum number of intersection points is 45 -/
theorem triangle_town_intersections :
  max_intersections 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_town_intersections_l3662_366212


namespace NUMINAMATH_CALUDE_problem_l3662_366289

def f (m : ℕ) (x : ℝ) : ℝ := |x - m| + |x|

theorem problem (m : ℕ) (h1 : m > 0) (h2 : ∃ x : ℝ, f m x < 2) :
  m = 1 ∧
  ∀ α β : ℝ, α > 1 → β > 1 → f m α + f m β = 6 → 4/α + 1/β ≥ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_problem_l3662_366289


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l3662_366234

/-- The number of games played in a chess tournament where each participant
    plays exactly one game with each other participant. -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that a chess tournament with 105 games has 15 participants. -/
theorem chess_tournament_participants :
  ∃ n : ℕ, n > 0 ∧ num_games n = 105 ∧ n = 15 := by
  sorry

#check chess_tournament_participants

end NUMINAMATH_CALUDE_chess_tournament_participants_l3662_366234


namespace NUMINAMATH_CALUDE_square_perimeter_area_l3662_366228

/-- Theorem: A square with a perimeter of 24 inches has an area of 36 square inches. -/
theorem square_perimeter_area : 
  ∀ (side : ℝ), 
  (4 * side = 24) → (side * side = 36) :=
by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_area_l3662_366228


namespace NUMINAMATH_CALUDE_arccos_one_half_l3662_366254

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by sorry

end NUMINAMATH_CALUDE_arccos_one_half_l3662_366254


namespace NUMINAMATH_CALUDE_two_a_plus_a_equals_three_a_l3662_366293

theorem two_a_plus_a_equals_three_a (a : ℝ) : 2 * a + a = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_two_a_plus_a_equals_three_a_l3662_366293


namespace NUMINAMATH_CALUDE_total_guests_served_l3662_366262

theorem total_guests_served (adults : ℕ) (children : ℕ) (seniors : ℕ) : 
  adults = 58 →
  children = adults - 35 →
  seniors = 2 * children →
  adults + children + seniors = 127 := by
  sorry

end NUMINAMATH_CALUDE_total_guests_served_l3662_366262


namespace NUMINAMATH_CALUDE_train_distance_l3662_366214

/-- Given a train that travels 1 mile every 2 minutes, prove it will travel 45 miles in 90 minutes -/
theorem train_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 1 / 2 → time = 90 → distance = speed * time → distance = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l3662_366214


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l3662_366269

/-- Two similar right triangles, one with legs 12 and 9, the other with legs y and 7 -/
def similar_triangles (y : ℝ) : Prop :=
  12 / y = 9 / 7

theorem similar_triangles_leg_length :
  ∃ y : ℝ, similar_triangles y ∧ y = 84 / 9 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l3662_366269


namespace NUMINAMATH_CALUDE_xoxoxox_probability_l3662_366227

def total_tiles : ℕ := 7
def x_tiles : ℕ := 4
def o_tiles : ℕ := 3

theorem xoxoxox_probability :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = (1 : ℚ) / 35 :=
sorry

end NUMINAMATH_CALUDE_xoxoxox_probability_l3662_366227


namespace NUMINAMATH_CALUDE_sum_of_50th_row_l3662_366290

/-- Represents the sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℕ :=
  2^n - 2 * (n * (n + 1) / 2)

/-- The triangular array property -/
axiom triangular_array_property (n : ℕ) :
  f n = 2 * f (n - 1) + n * (n + 1)

/-- Theorem: The sum of numbers in the 50th row is 2^50 - 2550 -/
theorem sum_of_50th_row :
  f 50 = 2^50 - 2550 := by sorry

end NUMINAMATH_CALUDE_sum_of_50th_row_l3662_366290


namespace NUMINAMATH_CALUDE_geometry_theorem_l3662_366224

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the theorem
theorem geometry_theorem 
  (m n : Line) (α β : Plane) 
  (distinct_lines : m ≠ n) 
  (distinct_planes : α ≠ β) :
  (perpendicular m n → perpendicularLP m α → ¬subset n α → parallel n α) ∧
  (perpendicularLP m β → perpendicularPP α β → (parallel m α ∨ subset m α)) ∧
  (perpendicular m n → perpendicularLP m α → perpendicularLP n β → perpendicularPP α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l3662_366224


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l3662_366229

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (a b : Line) (α β : Plane) :
  contained_in a β → parallel α β → line_parallel_to_plane a α :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l3662_366229


namespace NUMINAMATH_CALUDE_add_multiply_round_problem_l3662_366256

theorem add_multiply_round_problem : 
  let a := 73.5891
  let b := 24.376
  let sum := a + b
  let product := sum * 2
  (product * 100).round / 100 = 195.93 := by sorry

end NUMINAMATH_CALUDE_add_multiply_round_problem_l3662_366256


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l3662_366281

/-- The polar equation r = 1 / (sin θ + cos θ) represents a line in Cartesian coordinates -/
theorem polar_to_cartesian_line :
  ∀ (θ : ℝ) (r : ℝ), r = 1 / (Real.sin θ + Real.cos θ) →
  ∃ (x y : ℝ), x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ x + y = 1 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_line_l3662_366281


namespace NUMINAMATH_CALUDE_max_m_plus_2n_l3662_366246

-- Define the sets A and B
def A : Set ℕ := {x | ∃ k : ℕ+, x = 2 * k - 1}
def B : Set ℕ := {x | ∃ k : ℕ+, x = 8 * k - 8}

-- Define a function to calculate the sum of m different elements from A
def sumA (m : ℕ) : ℕ := m^2

-- Define a function to calculate the sum of n different elements from B
def sumB (n : ℕ) : ℕ := 4 * n^2 - 4 * n

-- State the theorem
theorem max_m_plus_2n (m n : ℕ) :
  sumA m + sumB n ≤ 967 → m + 2 * n ≤ 44 :=
sorry

end NUMINAMATH_CALUDE_max_m_plus_2n_l3662_366246


namespace NUMINAMATH_CALUDE_interview_segment_ratio_l3662_366279

/-- Represents the lengths of three interview segments in a radio show. -/
structure InterviewSegments where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Theorem stating the ratio of the third segment to the second segment is 1:2
    given the conditions of the radio show. -/
theorem interview_segment_ratio
  (segments : InterviewSegments)
  (total_time : segments.first + segments.second + segments.third = 90)
  (first_twice_others : segments.first = 2 * (segments.second + segments.third))
  (third_length : segments.third = 10) :
  segments.third / segments.second = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_interview_segment_ratio_l3662_366279


namespace NUMINAMATH_CALUDE_largest_result_operation_l3662_366247

theorem largest_result_operation : 
  let a := -1
  let b := -(1/2)
  let add_result := a + b
  let sub_result := a - b
  let mul_result := a * b
  let div_result := a / b
  (div_result > add_result) ∧ 
  (div_result > sub_result) ∧ 
  (div_result > mul_result) ∧
  (div_result = 2) := by
sorry

end NUMINAMATH_CALUDE_largest_result_operation_l3662_366247


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3662_366282

theorem quadratic_equal_roots :
  ∃ (x : ℝ), x^2 + 2*x + 1 = 0 ∧
  (∀ (y : ℝ), y^2 + 2*y + 1 = 0 → y = x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3662_366282


namespace NUMINAMATH_CALUDE_absolute_value_sum_range_l3662_366213

theorem absolute_value_sum_range (m : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| + |x - 3| ≥ m) ↔ m ∈ Set.Iic 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_range_l3662_366213


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3662_366291

theorem min_value_of_expression (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_product : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3662_366291


namespace NUMINAMATH_CALUDE_baked_goods_distribution_l3662_366206

/-- Calculates the number of items not placed in containers --/
def itemsNotPlaced (totalItems : Nat) (itemsPerContainer : Nat) : Nat :=
  totalItems % itemsPerContainer

theorem baked_goods_distribution (gingerbreadCookies sugarCookies fruitTarts : Nat) 
  (gingerbreadPerJar sugarPerBox tartsPerBox : Nat) :
  gingerbreadCookies = 47 → 
  sugarCookies = 78 → 
  fruitTarts = 36 → 
  gingerbreadPerJar = 6 → 
  sugarPerBox = 9 → 
  tartsPerBox = 4 → 
  (itemsNotPlaced gingerbreadCookies gingerbreadPerJar = 5 ∧ 
   itemsNotPlaced sugarCookies sugarPerBox = 6 ∧ 
   itemsNotPlaced fruitTarts tartsPerBox = 0) := by
  sorry

#eval itemsNotPlaced 47 6  -- Should output 5
#eval itemsNotPlaced 78 9  -- Should output 6
#eval itemsNotPlaced 36 4  -- Should output 0

end NUMINAMATH_CALUDE_baked_goods_distribution_l3662_366206


namespace NUMINAMATH_CALUDE_soccer_team_combinations_l3662_366298

def soccer_team_size : ℕ := 16
def quadruplets_count : ℕ := 4
def starting_lineup_size : ℕ := 7
def max_quadruplets_in_lineup : ℕ := 2

theorem soccer_team_combinations :
  (Nat.choose (soccer_team_size - quadruplets_count) starting_lineup_size) +
  (quadruplets_count * Nat.choose (soccer_team_size - quadruplets_count) (starting_lineup_size - 1)) +
  (Nat.choose quadruplets_count 2 * Nat.choose (soccer_team_size - quadruplets_count) (starting_lineup_size - 2)) = 9240 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_combinations_l3662_366298


namespace NUMINAMATH_CALUDE_rectangle_area_l3662_366208

/-- Given a rectangle with perimeter 28 cm and length 9 cm, its area is 45 cm² -/
theorem rectangle_area (perimeter length : ℝ) (h1 : perimeter = 28) (h2 : length = 9) :
  let width := (perimeter - 2 * length) / 2
  length * width = 45 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3662_366208


namespace NUMINAMATH_CALUDE_no_solution_implies_m_greater_2023_l3662_366242

theorem no_solution_implies_m_greater_2023 (m : ℝ) :
  (∀ x : ℝ, ¬(x ≥ m ∧ x ≤ 2023)) → m > 2023 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_greater_2023_l3662_366242


namespace NUMINAMATH_CALUDE_product_increase_2016_l3662_366284

theorem product_increase_2016 : ∃ (a b c : ℕ), 
  ((a - 3) * (b - 3) * (c - 3)) - (a * b * c) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_product_increase_2016_l3662_366284


namespace NUMINAMATH_CALUDE_vector_problem_l3662_366222

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (-1, 7)

theorem vector_problem (a b : ℝ × ℝ) (ha : a = (3, 4)) (hb : b = (-1, 7)) :
  (a.1 * b.1 + a.2 * b.2 = 25) ∧ 
  (Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3662_366222


namespace NUMINAMATH_CALUDE_factorization_problems_l3662_366231

variable (a b : ℝ)

theorem factorization_problems :
  (-25 + a^4 = (a^2 + 5) * (a + 5) * (a - 5)) ∧
  (a^3 * b - 10 * a^2 * b + 25 * a * b = a * b * (a - 5)^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l3662_366231


namespace NUMINAMATH_CALUDE_cone_circumscribed_sphere_surface_area_l3662_366292

/-- Given a cone with base area π and lateral area twice the base area, 
    the surface area of its circumscribed sphere is 16π/3 -/
theorem cone_circumscribed_sphere_surface_area 
  (base_area : ℝ) 
  (lateral_area : ℝ) 
  (h1 : base_area = π) 
  (h2 : lateral_area = 2 * base_area) : 
  ∃ (r : ℝ), 
    r > 0 ∧ 
    4 * π * r^2 = 16 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_circumscribed_sphere_surface_area_l3662_366292


namespace NUMINAMATH_CALUDE_price_increase_quantity_decrease_l3662_366232

theorem price_increase_quantity_decrease (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let original_cost := P * Q
  let new_price := P * 1.15
  let new_quantity := Q * 0.6
  let new_cost := new_price * new_quantity
  new_cost = original_cost * 0.69 :=
by sorry

end NUMINAMATH_CALUDE_price_increase_quantity_decrease_l3662_366232


namespace NUMINAMATH_CALUDE_union_P_Q_l3662_366241

def P : Set ℝ := { x | -1 < x ∧ x < 1 }
def Q : Set ℝ := { x | x^2 - 2*x < 0 }

theorem union_P_Q : P ∪ Q = { x | -1 < x ∧ x < 2 } := by sorry

end NUMINAMATH_CALUDE_union_P_Q_l3662_366241


namespace NUMINAMATH_CALUDE_triangle_property_l3662_366271

theorem triangle_property (A B C : Real) (a b c R : Real) :
  0 < B → B < π / 2 →
  2 * R - b = 2 * b * Real.sin B →
  a = Real.sqrt 3 →
  c = 3 →
  B = π / 6 ∧ Real.sin C = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l3662_366271


namespace NUMINAMATH_CALUDE_product_equality_l3662_366238

theorem product_equality : 100 * 19.98 * 2.998 * 1000 = 5994004 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3662_366238


namespace NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_2730_l3662_366286

theorem sum_of_extreme_prime_factors_of_2730 : 
  ∃ (smallest largest : ℕ), 
    smallest.Prime ∧ 
    largest.Prime ∧ 
    smallest ∣ 2730 ∧ 
    largest ∣ 2730 ∧ 
    (∀ p : ℕ, p.Prime → p ∣ 2730 → p ≥ smallest ∧ p ≤ largest) ∧ 
    smallest + largest = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_2730_l3662_366286


namespace NUMINAMATH_CALUDE_sandy_saturday_hours_l3662_366200

/-- Sandy's hourly rate in dollars -/
def hourly_rate : ℚ := 15

/-- Hours Sandy worked on Friday -/
def friday_hours : ℚ := 10

/-- Hours Sandy worked on Sunday -/
def sunday_hours : ℚ := 14

/-- Total earnings for Friday, Saturday, and Sunday in dollars -/
def total_earnings : ℚ := 450

/-- Calculates the number of hours Sandy worked on Saturday -/
def saturday_hours : ℚ :=
  (total_earnings - hourly_rate * (friday_hours + sunday_hours)) / hourly_rate

theorem sandy_saturday_hours :
  saturday_hours = 6 := by sorry

end NUMINAMATH_CALUDE_sandy_saturday_hours_l3662_366200


namespace NUMINAMATH_CALUDE_glass_volume_l3662_366243

/-- The volume of a glass given pessimist and optimist perspectives --/
theorem glass_volume (V : ℝ) 
  (h_pessimist : 0.4 * V = V - 0.6 * V) 
  (h_optimist : 0.6 * V = V - 0.4 * V) 
  (h_difference : 0.6 * V - 0.4 * V = 46) : 
  V = 230 := by
  sorry

end NUMINAMATH_CALUDE_glass_volume_l3662_366243


namespace NUMINAMATH_CALUDE_number_divided_by_005_equals_1500_l3662_366223

theorem number_divided_by_005_equals_1500 (x : ℝ) : x / 0.05 = 1500 → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_005_equals_1500_l3662_366223


namespace NUMINAMATH_CALUDE_stratified_sampling_equality_l3662_366205

/-- Represents the number of people in each age group -/
structure Population where
  elderly : ℕ
  middleAged : ℕ

/-- Represents the number of people selected from each age group -/
structure Selected where
  elderly : ℕ
  middleAged : ℕ

/-- Checks if the selection maintains equal probability across strata -/
def isEqualProbability (pop : Population) (sel : Selected) : Prop :=
  (sel.elderly : ℚ) / pop.elderly = (sel.middleAged : ℚ) / pop.middleAged

theorem stratified_sampling_equality 
  (pop : Population) (sel : Selected) 
  (h1 : pop.elderly = 140) 
  (h2 : pop.middleAged = 210) 
  (h3 : sel.elderly = 4) 
  (h4 : isEqualProbability pop sel) : 
  sel.middleAged = 6 := by
  sorry

#check stratified_sampling_equality

end NUMINAMATH_CALUDE_stratified_sampling_equality_l3662_366205


namespace NUMINAMATH_CALUDE_pool_drain_time_l3662_366233

/-- Represents the pool draining problem -/
structure PoolDraining where
  capacity : ℝ
  fillTime : ℝ
  drainTime : ℝ
  elapsedTime : ℝ
  remainingWater : ℝ

/-- Theorem stating the solution to the pool draining problem -/
theorem pool_drain_time (p : PoolDraining) 
  (h_capacity : p.capacity = 120)
  (h_fillTime : p.fillTime = 6)
  (h_elapsedTime : p.elapsedTime = 3)
  (h_remainingWater : p.remainingWater = 90) :
  p.drainTime = 4 := by
  sorry


end NUMINAMATH_CALUDE_pool_drain_time_l3662_366233


namespace NUMINAMATH_CALUDE_meeting_probability_4x3_grid_l3662_366219

/-- Represents a grid network --/
structure GridNetwork where
  rows : ℕ
  cols : ℕ

/-- Represents a person moving on the grid --/
structure Person where
  start_row : ℕ
  start_col : ℕ
  end_row : ℕ
  end_col : ℕ

/-- The probability of two persons meeting on a grid network --/
def meeting_probability (grid : GridNetwork) (p1 p2 : Person) : ℚ :=
  sorry

/-- Theorem stating the probability of meeting in a 4x3 grid --/
theorem meeting_probability_4x3_grid :
  let grid : GridNetwork := ⟨4, 3⟩
  let person1 : Person := ⟨0, 0, 3, 4⟩  -- A to B
  let person2 : Person := ⟨3, 4, 0, 0⟩  -- B to A
  meeting_probability grid person1 person2 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_meeting_probability_4x3_grid_l3662_366219


namespace NUMINAMATH_CALUDE_road_building_equation_l3662_366216

theorem road_building_equation (x : ℝ) 
  (h_positive : x > 0) 
  (h_team_a_length : 9 > 0) 
  (h_team_b_length : 12 > 0) 
  (h_team_b_faster : x + 1 > x) : 
  9 / x - 12 / (x + 1) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_road_building_equation_l3662_366216


namespace NUMINAMATH_CALUDE_hourly_wage_calculation_l3662_366258

/-- Calculates the hourly wage given the total earnings, hours worked, widgets produced, and widget bonus rate. -/
def calculate_hourly_wage (total_earnings : ℚ) (hours_worked : ℚ) (widgets_produced : ℚ) (widget_bonus_rate : ℚ) : ℚ :=
  (total_earnings - widgets_produced * widget_bonus_rate) / hours_worked

theorem hourly_wage_calculation :
  let total_earnings : ℚ := 620
  let hours_worked : ℚ := 40
  let widgets_produced : ℚ := 750
  let widget_bonus_rate : ℚ := 0.16
  calculate_hourly_wage total_earnings hours_worked widgets_produced widget_bonus_rate = 12.5 := by
sorry

#eval calculate_hourly_wage 620 40 750 0.16

end NUMINAMATH_CALUDE_hourly_wage_calculation_l3662_366258


namespace NUMINAMATH_CALUDE_compute_expression_l3662_366257

theorem compute_expression : 11 * (1 / 17) * 34 - 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3662_366257


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l3662_366251

/-- Given a quadratic equation ax² + bx + c = 0 with a > 0 and no real roots,
    the solution set of ax² + bx + c < 0 is empty. -/
theorem quadratic_inequality_empty_solution_set
  (a b c : ℝ) 
  (h_a_pos : a > 0)
  (h_no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) :
  {x : ℝ | a * x^2 + b * x + c < 0} = ∅ :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l3662_366251


namespace NUMINAMATH_CALUDE_three_card_selection_count_l3662_366280

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 13

/-- 
  Theorem: The number of ways to select three different cards in sequence 
  from a standard deck is 132600.
-/
theorem three_card_selection_count : 
  standard_deck_size * (standard_deck_size - 1) * (standard_deck_size - 2) = 132600 := by
  sorry


end NUMINAMATH_CALUDE_three_card_selection_count_l3662_366280


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l3662_366250

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l3662_366250


namespace NUMINAMATH_CALUDE_book_distribution_l3662_366275

/-- The number of books -/
def num_books : ℕ := 15

/-- The number of exercise books -/
def num_exercise_books : ℕ := 26

/-- The number of students in the first scenario -/
def students_scenario1 : ℕ := (num_exercise_books / 2)

/-- The number of students in the second scenario -/
def students_scenario2 : ℕ := (num_books / 3)

theorem book_distribution :
  (students_scenario1 + 2 = num_books) ∧
  (2 * students_scenario1 = num_exercise_books) ∧
  (3 * students_scenario2 = num_books) ∧
  (5 * students_scenario2 + 1 = num_exercise_books) :=
by sorry

end NUMINAMATH_CALUDE_book_distribution_l3662_366275


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l3662_366296

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 90)
  (h2 : average_speed = 66) : 
  (2 * average_speed - speed_first_hour) = 42 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l3662_366296


namespace NUMINAMATH_CALUDE_keith_receives_144_messages_l3662_366273

/-- Represents the number of messages sent between people in a day -/
structure MessageCount where
  juan_to_laurence : ℕ
  juan_to_keith : ℕ
  laurence_to_missy : ℕ

/-- The conditions of the messaging problem -/
def messaging_problem (m : MessageCount) : Prop :=
  m.juan_to_keith = 8 * m.juan_to_laurence ∧
  m.laurence_to_missy = m.juan_to_laurence ∧
  m.laurence_to_missy = 18

/-- The theorem stating that Keith receives 144 messages from Juan -/
theorem keith_receives_144_messages (m : MessageCount) 
  (h : messaging_problem m) : m.juan_to_keith = 144 := by
  sorry

end NUMINAMATH_CALUDE_keith_receives_144_messages_l3662_366273


namespace NUMINAMATH_CALUDE_equation_solutions_l3662_366294

theorem equation_solutions :
  (∀ x : ℝ, 4 * (2 * x - 1)^2 = 36 ↔ x = 2 ∨ x = -1) ∧
  (∀ x : ℝ, (1/4) * (2 * x + 3)^3 - 54 = 0 ↔ x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3662_366294
