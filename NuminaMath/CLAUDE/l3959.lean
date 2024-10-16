import Mathlib

namespace NUMINAMATH_CALUDE_power_of_five_mod_eighteen_l3959_395993

theorem power_of_five_mod_eighteen (x : ℕ) : ∃ x, x > 0 ∧ (5^x : ℤ) % 18 = 13 ∧ ∀ y, 0 < y ∧ y < x → (5^y : ℤ) % 18 ≠ 13 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_mod_eighteen_l3959_395993


namespace NUMINAMATH_CALUDE_range_of_m_l3959_395944

theorem range_of_m : ∀ m : ℝ,
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0 ↔ m > 2) →
  (¬∃ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 = 0 ↔ 1 < m ∧ m < 3) →
  ((m > 2 ∨ (1 < m ∧ m < 3)) ∧ ¬(m > 2 ∧ 1 < m ∧ m < 3)) →
  m ≥ 3 ∨ (1 < m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3959_395944


namespace NUMINAMATH_CALUDE_sphere_containment_l3959_395957

/-- A point in 3-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A sphere in 3-dimensional space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Predicate to check if a point is inside or on a sphere -/
def Point3D.inSphere (p : Point3D) (s : Sphere) : Prop :=
  (p.x - s.center.x)^2 + (p.y - s.center.y)^2 + (p.z - s.center.z)^2 ≤ s.radius^2

/-- The main theorem -/
theorem sphere_containment (n : ℕ) (points : Fin n → Point3D) 
    (h : n ≥ 5)
    (h_four : ∀ (a b c d : Fin n), ∃ (s : Sphere), 
      s.radius = 1 ∧ 
      (points a).inSphere s ∧ 
      (points b).inSphere s ∧ 
      (points c).inSphere s ∧ 
      (points d).inSphere s) :
    ∃ (s : Sphere), s.radius = 1 ∧ ∀ (i : Fin n), (points i).inSphere s := by
  sorry

end NUMINAMATH_CALUDE_sphere_containment_l3959_395957


namespace NUMINAMATH_CALUDE_savings_equality_l3959_395903

def total_salary : ℝ := 4000
def a_salary : ℝ := 3000
def a_spend_rate : ℝ := 0.95
def b_spend_rate : ℝ := 0.85

def b_salary : ℝ := total_salary - a_salary

def a_savings : ℝ := a_salary * (1 - a_spend_rate)
def b_savings : ℝ := b_salary * (1 - b_spend_rate)

theorem savings_equality : a_savings = b_savings := by
  sorry

end NUMINAMATH_CALUDE_savings_equality_l3959_395903


namespace NUMINAMATH_CALUDE_unique_root_monotonic_continuous_l3959_395930

theorem unique_root_monotonic_continuous {f : ℝ → ℝ} {a b : ℝ} (h_mono : Monotone f) (h_cont : Continuous f) (h_sign : f a * f b < 0) (h_le : a ≤ b) :
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_root_monotonic_continuous_l3959_395930


namespace NUMINAMATH_CALUDE_expected_boy_girl_pairs_l3959_395947

/-- The expected number of adjacent boy-girl pairs when 10 boys and 14 girls
    are seated randomly around a circular table with 24 seats. -/
theorem expected_boy_girl_pairs :
  let num_boys : ℕ := 10
  let num_girls : ℕ := 14
  let total_seats : ℕ := 24
  let prob_boy_girl : ℚ := (num_boys : ℚ) * num_girls / (total_seats * (total_seats - 1))
  let prob_girl_boy : ℚ := (num_girls : ℚ) * num_boys / (total_seats * (total_seats - 1))
  let prob_adjacent_pair : ℚ := prob_boy_girl + prob_girl_boy
  let expected_pairs : ℚ := (total_seats : ℚ) * prob_adjacent_pair
  expected_pairs = 280 / 23 :=
by sorry

end NUMINAMATH_CALUDE_expected_boy_girl_pairs_l3959_395947


namespace NUMINAMATH_CALUDE_time_after_316h59m59s_l3959_395987

/-- Represents time on a 12-hour digital clock -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

def addTime (t : Time) (hours minutes seconds : Nat) : Time :=
  let totalSeconds := t.hours * 3600 + t.minutes * 60 + t.seconds + hours * 3600 + minutes * 60 + seconds
  let newHours := (totalSeconds / 3600) % 12
  let newMinutes := (totalSeconds % 3600) / 60
  let newSeconds := totalSeconds % 60
  { hours := if newHours = 0 then 12 else newHours, minutes := newMinutes, seconds := newSeconds }

def sumDigits (t : Time) : Nat :=
  t.hours + t.minutes + t.seconds

theorem time_after_316h59m59s (startTime : Time) :
  startTime.hours = 3 ∧ startTime.minutes = 0 ∧ startTime.seconds = 0 →
  sumDigits (addTime startTime 316 59 59) = 125 := by
  sorry

end NUMINAMATH_CALUDE_time_after_316h59m59s_l3959_395987


namespace NUMINAMATH_CALUDE_delivery_growth_rate_l3959_395989

/-- Represents the monthly growth rate of deliveries -/
def monthly_growth_rate : ℝ := 0.1

/-- The initial number of deliveries in October -/
def initial_deliveries : ℕ := 100000

/-- The final number of deliveries in December -/
def final_deliveries : ℕ := 121000

/-- The number of months between October and December -/
def months : ℕ := 2

theorem delivery_growth_rate :
  (initial_deliveries : ℝ) * (1 + monthly_growth_rate) ^ months = final_deliveries := by
  sorry

end NUMINAMATH_CALUDE_delivery_growth_rate_l3959_395989


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l3959_395986

theorem subtraction_of_decimals : 3.75 - 0.48 = 3.27 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l3959_395986


namespace NUMINAMATH_CALUDE_mail_difference_l3959_395915

/-- Proves that the difference between mail sent on Thursday and Wednesday is 15 --/
theorem mail_difference (monday tuesday wednesday thursday : ℕ) : 
  monday = 65 →
  tuesday = monday + 10 →
  wednesday = tuesday - 5 →
  thursday > wednesday →
  monday + tuesday + wednesday + thursday = 295 →
  thursday - wednesday = 15 :=
by sorry

end NUMINAMATH_CALUDE_mail_difference_l3959_395915


namespace NUMINAMATH_CALUDE_cosine_value_for_given_point_l3959_395996

theorem cosine_value_for_given_point (α : Real) :
  (∃ r : Real, r > 0 ∧ r^2 = 1 + 3) →
  (1, -Real.sqrt 3) ∈ {(x, y) | x = r * Real.cos α ∧ y = r * Real.sin α} →
  Real.cos α = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cosine_value_for_given_point_l3959_395996


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l3959_395958

theorem discount_percentage_calculation (MP : ℝ) (h1 : MP > 0) : 
  let CP := 0.36 * MP
  let gain_percent := 122.22222222222223
  let SP := CP * (1 + gain_percent / 100)
  (MP - SP) / MP * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_discount_percentage_calculation_l3959_395958


namespace NUMINAMATH_CALUDE_cubic_minimum_condition_l3959_395959

/-- A cubic function with parameters p, q, and r -/
def cubic_function (p q r x : ℝ) : ℝ := x^3 + 3*p*x^2 + 3*q*x + r

/-- The derivative of the cubic function with respect to x -/
def cubic_derivative (p q x : ℝ) : ℝ := 3*x^2 + 6*p*x + 3*q

theorem cubic_minimum_condition (p q r : ℝ) :
  (∀ x : ℝ, cubic_function p q r x ≥ cubic_function p q r (-p)) ∧
  cubic_function p q r (-p) = -27 →
  r = -27 - 2*p^3 + 3*p*q :=
by sorry

end NUMINAMATH_CALUDE_cubic_minimum_condition_l3959_395959


namespace NUMINAMATH_CALUDE_P_roots_l3959_395965

def P : ℕ → (ℝ → ℝ)
  | 0 => λ _ => 1
  | n + 1 => λ x => x^(5*(n+1)) - P n x

theorem P_roots (n : ℕ) :
  (n % 2 = 1 → ∃! x : ℝ, P n x = 0 ∧ x = 1) ∧
  (n % 2 = 0 → ¬∃ x : ℝ, P n x = 0) := by
  sorry

end NUMINAMATH_CALUDE_P_roots_l3959_395965


namespace NUMINAMATH_CALUDE_fraction_of_total_l3959_395980

theorem fraction_of_total (total : ℚ) (r_amount : ℚ) : 
  total = 9000 → r_amount = 3600 → r_amount / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_total_l3959_395980


namespace NUMINAMATH_CALUDE_unique_triplet_sum_l3959_395918

theorem unique_triplet_sum (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c)
  (heq : (25 : ℚ) / 84 = 1 / a + 1 / (a * b) + 1 / (a * b * c)) :
  a + b + c = 17 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_sum_l3959_395918


namespace NUMINAMATH_CALUDE_radius_of_combined_lead_spheres_l3959_395960

/-- The radius of a sphere formed by combining the volume of multiple smaller spheres -/
def radiusOfCombinedSphere (n : ℕ) (r : ℝ) : ℝ :=
  ((n : ℝ) * r^3)^(1/3)

/-- Theorem: The radius of a sphere formed by combining 12 spheres of radius 2 cm is ∛96 cm -/
theorem radius_of_combined_lead_spheres :
  radiusOfCombinedSphere 12 2 = (96 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_radius_of_combined_lead_spheres_l3959_395960


namespace NUMINAMATH_CALUDE_intersection_and_chord_properties_l3959_395970

/-- Given two points M and N in a 2D Cartesian coordinate system -/
def M : ℝ × ℝ := (1, -3)
def N : ℝ × ℝ := (5, 1)

/-- Point C satisfies the given condition -/
def C (t : ℝ) : ℝ × ℝ :=
  (t * M.1 + (1 - t) * N.1, t * M.2 + (1 - t) * N.2)

/-- The parabola y^2 = 4x -/
def on_parabola (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

/-- Perpendicularity of two vectors -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Main theorem -/
theorem intersection_and_chord_properties :
  (∃ A B : ℝ × ℝ, 
    (∃ t : ℝ, C t = A) ∧ 
    (∃ t : ℝ, C t = B) ∧ 
    on_parabola A ∧ 
    on_parabola B ∧ 
    perpendicular A B) ∧
  (∃ P : ℝ × ℝ, P.1 = 4 ∧ P.2 = 0 ∧
    ∀ Q R : ℝ × ℝ, 
      on_parabola Q ∧ 
      on_parabola R ∧ 
      (Q.2 - P.2) * (R.1 - P.1) = (Q.1 - P.1) * (R.2 - P.2) →
      (Q.1 * R.1 + Q.2 * R.2 = 0)) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_chord_properties_l3959_395970


namespace NUMINAMATH_CALUDE_survey_respondents_count_l3959_395904

theorem survey_respondents_count :
  let brand_x_count : ℕ := 360
  let brand_x_to_y_ratio : ℚ := 9 / 1
  let total_respondents : ℕ := brand_x_count + (brand_x_count / brand_x_to_y_ratio.num * brand_x_to_y_ratio.den).toNat
  total_respondents = 400 :=
by sorry

end NUMINAMATH_CALUDE_survey_respondents_count_l3959_395904


namespace NUMINAMATH_CALUDE_sacks_filled_l3959_395901

theorem sacks_filled (wood_per_sack : ℕ) (total_wood : ℕ) (h1 : wood_per_sack = 20) (h2 : total_wood = 80) :
  total_wood / wood_per_sack = 4 := by
  sorry

end NUMINAMATH_CALUDE_sacks_filled_l3959_395901


namespace NUMINAMATH_CALUDE_bullying_instances_l3959_395971

def days_per_bullying : ℕ := 3
def typical_fingers_and_toes : ℕ := 20
def additional_suspension_days : ℕ := 14

def total_suspension_days : ℕ := 3 * typical_fingers_and_toes + additional_suspension_days

theorem bullying_instances : 
  (total_suspension_days / days_per_bullying : ℕ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_bullying_instances_l3959_395971


namespace NUMINAMATH_CALUDE_unequal_probabilities_after_adding_balls_l3959_395900

/-- Represents the contents of the bag -/
structure BagContents where
  white : ℕ
  red : ℕ

/-- Calculates the probability of drawing a specific color ball -/
def probability (bag : BagContents) (color : ℕ) : ℚ :=
  color / (bag.white + bag.red : ℚ)

/-- The initial contents of the bag -/
def initialBag : BagContents := { white := 1, red := 2 }

/-- The bag after adding 1 white ball and 2 red balls -/
def updatedBag : BagContents := { white := initialBag.white + 1, red := initialBag.red + 2 }

theorem unequal_probabilities_after_adding_balls :
  probability updatedBag updatedBag.white ≠ probability updatedBag updatedBag.red := by
  sorry

end NUMINAMATH_CALUDE_unequal_probabilities_after_adding_balls_l3959_395900


namespace NUMINAMATH_CALUDE_triangle_case1_triangle_case2_l3959_395973

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem 1: In a triangle ABC with c = 2, C = π/3, and area = √3, a = 2 and b = 2 -/
theorem triangle_case1 (t : Triangle) 
  (h1 : t.c = 2)
  (h2 : t.C = π / 3)
  (h3 : (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 3) :
  t.a = 2 ∧ t.b = 2 := by
  sorry

/-- Theorem 2: In a triangle ABC with c = 2, C = π/3, and sin C + sin(B-A) = sin 2A,
    either (a = 4√3/3 and b = 2√3/3) or (a = 2 and b = 2) -/
theorem triangle_case2 (t : Triangle)
  (h1 : t.c = 2)
  (h2 : t.C = π / 3)
  (h3 : Real.sin t.C + Real.sin (t.B - t.A) = Real.sin (2 * t.A)) :
  (t.a = (4 * Real.sqrt 3) / 3 ∧ t.b = (2 * Real.sqrt 3) / 3) ∨ 
  (t.a = 2 ∧ t.b = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_case1_triangle_case2_l3959_395973


namespace NUMINAMATH_CALUDE_binomial_sum_identity_l3959_395933

theorem binomial_sum_identity (p q n : ℕ+) :
  (∑' k, (Nat.choose (p + k) p) * (Nat.choose (q + n - k) q)) = Nat.choose (p + q + n + 1) (p + q + 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_sum_identity_l3959_395933


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3959_395910

theorem simplify_and_evaluate (a b : ℝ) (ha : a = 3) (hb : b = 2) :
  (a + b) / (a * b) / ((a / b) - (b / a)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3959_395910


namespace NUMINAMATH_CALUDE_hundred_thousandth_digit_position_l3959_395999

/-- Represents the position of a digit in a number, starting from the units digit (position 1) -/
def DigitPosition : ℕ → ℕ
  | 1 => 1  -- units
  | 2 => 2  -- tens
  | 3 => 3  -- hundreds
  | 4 => 4  -- thousands
  | 5 => 5  -- ten thousands
  | 6 => 6  -- hundred thousands
  | _ => 7  -- million and beyond

/-- The position of the hundred thousandth digit when counting from the units digit -/
theorem hundred_thousandth_digit_position : DigitPosition 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hundred_thousandth_digit_position_l3959_395999


namespace NUMINAMATH_CALUDE_chihuahua_grooming_time_l3959_395988

/-- The time Karen takes to groom different types of dogs -/
structure GroomingTimes where
  rottweiler : ℕ
  border_collie : ℕ
  chihuahua : ℕ

/-- The number of each type of dog Karen grooms -/
structure DogCounts where
  rottweilers : ℕ
  border_collies : ℕ
  chihuahuas : ℕ

/-- Calculates the total grooming time for all dogs -/
def totalGroomingTime (times : GroomingTimes) (counts : DogCounts) : ℕ :=
  times.rottweiler * counts.rottweilers +
  times.border_collie * counts.border_collies +
  times.chihuahua * counts.chihuahuas

theorem chihuahua_grooming_time :
  ∀ (times : GroomingTimes) (counts : DogCounts),
  times.rottweiler = 20 →
  times.border_collie = 10 →
  counts.rottweilers = 6 →
  counts.border_collies = 9 →
  counts.chihuahuas = 1 →
  totalGroomingTime times counts = 255 →
  times.chihuahua = 45 := by
  sorry

end NUMINAMATH_CALUDE_chihuahua_grooming_time_l3959_395988


namespace NUMINAMATH_CALUDE_tv_sales_increase_l3959_395954

theorem tv_sales_increase (original_price : ℝ) (original_quantity : ℝ) 
  (h_positive_price : original_price > 0) (h_positive_quantity : original_quantity > 0) :
  let new_price := 0.9 * original_price
  let new_total_value := 1.665 * (original_price * original_quantity)
  ∃ (new_quantity : ℝ), 
    new_price * new_quantity = new_total_value ∧ 
    (new_quantity - original_quantity) / original_quantity = 0.85 :=
by sorry

end NUMINAMATH_CALUDE_tv_sales_increase_l3959_395954


namespace NUMINAMATH_CALUDE_marked_price_calculation_l3959_395941

theorem marked_price_calculation (purchase_price : ℝ) (discount_percentage : ℝ) : 
  purchase_price = 50 ∧ discount_percentage = 60 → 
  (purchase_price / ((100 - discount_percentage) / 100)) / 2 = 62.50 := by
sorry

end NUMINAMATH_CALUDE_marked_price_calculation_l3959_395941


namespace NUMINAMATH_CALUDE_current_speed_l3959_395983

theorem current_speed (boat_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) 
  (h1 : boat_speed = 16)
  (h2 : upstream_time = 20 / 60)
  (h3 : downstream_time = 15 / 60) :
  ∃ c : ℝ, c = 16 / 7 ∧ 
    (boat_speed - c) * upstream_time = (boat_speed + c) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_current_speed_l3959_395983


namespace NUMINAMATH_CALUDE_min_sum_of_distances_min_sum_of_distances_achievable_l3959_395935

theorem min_sum_of_distances (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ Real.sqrt 20 := by
  sorry

theorem min_sum_of_distances_achievable : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = Real.sqrt 20 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_distances_min_sum_of_distances_achievable_l3959_395935


namespace NUMINAMATH_CALUDE_sequence_existence_condition_l3959_395974

def is_valid_sequence (x : ℕ → Fin 2) (n m : ℕ) : Prop :=
  (∀ i, x i = 0 → x (i + m) = 1) ∧ (∀ i, x i = 1 → x (i + n) = 0)

theorem sequence_existence_condition (n m : ℕ) :
  (∃ x : ℕ → Fin 2, is_valid_sequence x n m) ↔
  (∃ (d p q : ℕ), n = 2^d * p ∧ m = 2^d * q ∧ Odd p ∧ Odd q) :=
sorry

end NUMINAMATH_CALUDE_sequence_existence_condition_l3959_395974


namespace NUMINAMATH_CALUDE_mikes_house_payments_l3959_395906

theorem mikes_house_payments (lower_rate higher_rate total_payments num_lower_payments num_higher_payments : ℚ) :
  higher_rate = 310 →
  total_payments = 3615 →
  num_lower_payments = 5 →
  num_higher_payments = 7 →
  num_lower_payments + num_higher_payments = 12 →
  num_lower_payments * lower_rate + num_higher_payments * higher_rate = total_payments →
  lower_rate = 289 := by
sorry

end NUMINAMATH_CALUDE_mikes_house_payments_l3959_395906


namespace NUMINAMATH_CALUDE_contradictory_implies_mutually_exclusive_but_not_conversely_l3959_395909

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutually_exclusive (A B : Set Ω) : Prop := A ∩ B = ∅

/-- Two events are contradictory if one event is the complement of the other -/
def contradictory (A B : Set Ω) : Prop := A = Bᶜ

/-- Theorem: Contradictory events are mutually exclusive, but mutually exclusive events are not necessarily contradictory -/
theorem contradictory_implies_mutually_exclusive_but_not_conversely :
  (∀ A B : Set Ω, contradictory A B → mutually_exclusive A B) ∧
  ¬(∀ A B : Set Ω, mutually_exclusive A B → contradictory A B) := by
  sorry

end NUMINAMATH_CALUDE_contradictory_implies_mutually_exclusive_but_not_conversely_l3959_395909


namespace NUMINAMATH_CALUDE_rotation_90_ccw_parabola_l3959_395972

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the original function
def original_function (x : ℝ) : ℝ := x^2

-- Define the rotation operation
def rotate_90_ccw (p : Point) : Point :=
  { x := -p.y, y := p.x }

-- Define the rotated function
def rotated_function (y : ℝ) : ℝ := -y^2

-- Theorem statement
theorem rotation_90_ccw_parabola :
  ∀ (p : Point), p.y = original_function p.x →
  (rotate_90_ccw p).y = rotated_function (rotate_90_ccw p).x :=
sorry

end NUMINAMATH_CALUDE_rotation_90_ccw_parabola_l3959_395972


namespace NUMINAMATH_CALUDE_business_hours_per_week_l3959_395940

-- Define the operating hours for weekdays and weekends
def weekdayHours : ℕ := 6
def weekendHours : ℕ := 4

-- Define the number of weekdays and weekend days in a week
def weekdays : ℕ := 5
def weekendDays : ℕ := 2

-- Define the total hours open in a week
def totalHoursOpen : ℕ := weekdayHours * weekdays + weekendHours * weekendDays

-- Theorem statement
theorem business_hours_per_week :
  totalHoursOpen = 38 := by
sorry

end NUMINAMATH_CALUDE_business_hours_per_week_l3959_395940


namespace NUMINAMATH_CALUDE_bookstore_repricing_l3959_395963

theorem bookstore_repricing (n : Nat) (p₁ p₂ : Nat) (h₁ : n = 1452) (h₂ : p₁ = 42) (h₃ : p₂ = 45) :
  (n * p₁) % p₂ = 9 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_repricing_l3959_395963


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l3959_395948

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 100 ∧ n % 5 = 2 ∧ ∀ m : ℕ, m < 100 ∧ m % 5 = 2 → m ≤ n → n = 97 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l3959_395948


namespace NUMINAMATH_CALUDE_french_speaking_percentage_l3959_395946

theorem french_speaking_percentage (total : ℕ) (french_and_english : ℕ) (french_only : ℕ) 
  (h1 : total = 200)
  (h2 : french_and_english = 10)
  (h3 : french_only = 40) :
  (total - (french_and_english + french_only)) / total * 100 = 75 :=
by sorry

end NUMINAMATH_CALUDE_french_speaking_percentage_l3959_395946


namespace NUMINAMATH_CALUDE_go_stones_count_l3959_395907

theorem go_stones_count (n : ℕ) (h1 : n^2 + 3 + 44 = (n + 2)^2) : n^2 + 3 = 103 := by
  sorry

#check go_stones_count

end NUMINAMATH_CALUDE_go_stones_count_l3959_395907


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3959_395969

/-- Given that P(1, -3) is the midpoint of line segment CD and C is located at (7, 5),
    prove that the sum of the coordinates of point D is -16. -/
theorem midpoint_coordinate_sum (C D : ℝ × ℝ) : 
  C = (7, 5) →
  (1, -3) = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3959_395969


namespace NUMINAMATH_CALUDE_car_speed_problem_l3959_395975

/-- Given two cars traveling on a 500-mile highway from opposite ends, 
    one at speed v and the other at 60 mph, meeting after 5 hours, 
    prove that the speed v of the first car is 40 mph. -/
theorem car_speed_problem (v : ℝ) 
  (h1 : v > 0) -- Assuming speed is positive
  (h2 : 5 * v + 5 * 60 = 500) : v = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3959_395975


namespace NUMINAMATH_CALUDE_olympic_audience_conversion_l3959_395981

def opening_ceremony_audience : ℕ := 316000000
def closing_ceremony_audience : ℕ := 236000000

def million_to_full_number (x : ℕ) : ℕ := x * 1000000
def million_to_billion (x : ℕ) : ℚ := x / 1000

/-- Rounds a rational number to one decimal place -/
def round_to_one_decimal (x : ℚ) : ℚ :=
  (x * 10).floor / 10

theorem olympic_audience_conversion :
  (million_to_full_number 316 = opening_ceremony_audience) ∧
  (round_to_one_decimal (million_to_billion closing_ceremony_audience) = 2.4) :=
sorry

end NUMINAMATH_CALUDE_olympic_audience_conversion_l3959_395981


namespace NUMINAMATH_CALUDE_nelly_painting_bid_l3959_395977

/-- The amount Nelly paid for the painting -/
def nelly_bid (joe_bid sarah_bid : ℕ) : ℕ :=
  max
    (3 * joe_bid + 2000)
    (max (4 * sarah_bid + 1500) (2 * (joe_bid + sarah_bid) + 1000))

/-- Theorem: Given the conditions, Nelly paid $482,000 for the painting -/
theorem nelly_painting_bid :
  nelly_bid 160000 50000 = 482000 := by
  sorry

end NUMINAMATH_CALUDE_nelly_painting_bid_l3959_395977


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3959_395950

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a → a 3 + a 7 = 37 → a 2 + a 4 + a 6 + a 8 = 74 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3959_395950


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l3959_395964

/-- Given a line passing through points (2, -1) and (5, 2), 
    prove that the sum of its slope and y-intercept is -2 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (2 : ℝ) * m + b = -1 ∧ 
  (5 : ℝ) * m + b = 2 →
  m + b = -2 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l3959_395964


namespace NUMINAMATH_CALUDE_vehicle_speeds_l3959_395936

/-- Represents the initial speeds and distance of two vehicles --/
structure VehicleData where
  bus_speed : ℝ
  car_speed : ℝ
  final_distance : ℝ

/-- Calculates the total distance traveled by both vehicles --/
def total_distance (data : VehicleData) : ℝ :=
  2 * data.bus_speed + 2 * data.car_speed + 2 * data.bus_speed + 2 * (data.car_speed - 10)

/-- Theorem stating the initial speeds of the vehicles --/
theorem vehicle_speeds : ∃ (data : VehicleData),
  data.car_speed = data.bus_speed + 8 ∧
  data.final_distance = 384 ∧
  total_distance data = data.final_distance ∧
  data.bus_speed = 46.5 ∧
  data.car_speed = 54.5 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_speeds_l3959_395936


namespace NUMINAMATH_CALUDE_number_of_bags_l3959_395927

theorem number_of_bags (students : ℕ) (nuts_per_student : ℕ) (nuts_per_bag : ℕ) : 
  students = 13 → nuts_per_student = 75 → nuts_per_bag = 15 →
  (students * nuts_per_student) / nuts_per_bag = 65 := by
  sorry

end NUMINAMATH_CALUDE_number_of_bags_l3959_395927


namespace NUMINAMATH_CALUDE_fraction_value_unchanged_keep_fraction_unchanged_l3959_395931

theorem fraction_value_unchanged (a b c : ℤ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a : ℚ) / b = (a + c) / (b + ((a + c) * b / a - b)) :=
by sorry

theorem keep_fraction_unchanged :
  let original_numerator := 3
  let original_denominator := 4
  let numerator_increase := 9
  let new_numerator := original_numerator + numerator_increase
  let denominator_increase := new_numerator * original_denominator / original_numerator - original_denominator
  denominator_increase = 12 :=
by sorry

end NUMINAMATH_CALUDE_fraction_value_unchanged_keep_fraction_unchanged_l3959_395931


namespace NUMINAMATH_CALUDE_circle_properties_l3959_395937

theorem circle_properties :
  let center : ℝ × ℝ := (1, -1)
  let radius : ℝ := Real.sqrt 2
  let origin : ℝ × ℝ := (0, 0)
  let tangent_point : ℝ × ℝ := (2, 0)
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let on_circle (p : ℝ × ℝ) := distance p center = radius
  let tangent_line (x y : ℝ) := x + y - 2 = 0
  
  (on_circle origin) ∧ 
  (on_circle tangent_point) ∧
  (tangent_line tangent_point.1 tangent_point.2) ∧
  (∀ (p : ℝ × ℝ), tangent_line p.1 p.2 → distance p center ≥ radius) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3959_395937


namespace NUMINAMATH_CALUDE_percentage_below_eight_l3959_395995

/-- Proves that the percentage of students below 8 years of age is 20% -/
theorem percentage_below_eight (total : ℕ) (eight_years : ℕ) (above_eight : ℕ) 
  (h1 : total = 25)
  (h2 : eight_years = 12)
  (h3 : above_eight = 2 * eight_years / 3)
  : (total - eight_years - above_eight) * 100 / total = 20 := by
  sorry

#check percentage_below_eight

end NUMINAMATH_CALUDE_percentage_below_eight_l3959_395995


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l3959_395926

/-- Represents a rectangle with diagonals divided into 12 equal segments -/
structure DividedRectangle where
  blank_area : ℝ
  total_area : ℝ

/-- The theorem stating the relationship between blank and shaded areas -/
theorem shaded_area_theorem (rect : DividedRectangle) 
  (h1 : rect.blank_area = 10) 
  (h2 : rect.total_area = rect.blank_area + 14) : 
  rect.total_area - rect.blank_area = 14 := by
  sorry

#check shaded_area_theorem

end NUMINAMATH_CALUDE_shaded_area_theorem_l3959_395926


namespace NUMINAMATH_CALUDE_car_travel_distance_l3959_395924

/-- Proves that a car traveling at 70 kmh for a certain time covers a distance of 105 km,
    given that if it had traveled 35 kmh faster, the trip would have lasted 30 minutes less. -/
theorem car_travel_distance :
  ∀ (time : ℝ),
  time > 0 →
  let distance := 70 * time
  let faster_time := time - 0.5
  let faster_speed := 70 + 35
  distance = faster_speed * faster_time →
  distance = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l3959_395924


namespace NUMINAMATH_CALUDE_present_age_of_b_l3959_395943

theorem present_age_of_b (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- In 10 years, A will be twice as old as B was 10 years ago
  (a = b + 9) →              -- A is currently 9 years older than B
  b = 39                     -- B's present age is 39 years
:= by sorry

end NUMINAMATH_CALUDE_present_age_of_b_l3959_395943


namespace NUMINAMATH_CALUDE_max_three_roots_l3959_395938

-- Define the function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem max_three_roots 
  (a b c : ℝ) 
  (x₁ x₂ : ℝ) 
  (h1 : ∃ x₁' x₂', x₁' ≠ x₂' ∧ (∀ x, x ≠ x₁' ∧ x ≠ x₂' → (deriv (f a b c)) x ≠ 0)) 
  (h2 : f a b c x₁ = x₁) :
  ∃ S : Finset ℝ, (∀ x, 3*(f a b c x)^2 + 2*a*(f a b c x) + b = 0 ↔ x ∈ S) ∧ Finset.card S ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_three_roots_l3959_395938


namespace NUMINAMATH_CALUDE_product_xyz_equals_one_l3959_395932

theorem product_xyz_equals_one 
  (x y z : ℝ) 
  (eq1 : x + 1/y = 2) 
  (eq2 : y + 1/z = 3) 
  (eq3 : z + 1/x = 4) : 
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_one_l3959_395932


namespace NUMINAMATH_CALUDE_water_displaced_volume_squared_l3959_395917

/-- The square of the volume of water displaced by a cube in a cylindrical barrel -/
theorem water_displaced_volume_squared
  (barrel_radius : ℝ)
  (barrel_height : ℝ)
  (cube_side_length : ℝ)
  (h_radius : barrel_radius = 5)
  (h_height : barrel_height = 10)
  (h_side : cube_side_length = 6) :
  let diagonal := cube_side_length * Real.sqrt 3
  let triangle_side := barrel_radius * Real.sqrt 3
  let tetrahedron_leg := (5 * Real.sqrt 6) / 2
  let volume := (375 * Real.sqrt 6) / 8
  volume ^ 2 = 843750 / 64 := by
  sorry

#eval (843750 / 64 : Float)  -- Should output approximately 13141.855

end NUMINAMATH_CALUDE_water_displaced_volume_squared_l3959_395917


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l3959_395923

theorem largest_n_for_factorization : ∃ (n : ℤ),
  (∀ m : ℤ, (∃ (a b : ℤ), 3 * X^2 + m * X + 54 = (3 * X + a) * (X + b)) → m ≤ n) ∧
  (∃ (a b : ℤ), 3 * X^2 + n * X + 54 = (3 * X + a) * (X + b)) ∧
  n = 163 :=
by sorry


end NUMINAMATH_CALUDE_largest_n_for_factorization_l3959_395923


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l3959_395945

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : a^4 + b^4 < c^4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l3959_395945


namespace NUMINAMATH_CALUDE_no_lattice_polygon1994_l3959_395942

/-- A polygon with 1994 sides where side lengths are √(i^2 + 4) -/
def Polygon1994 : Type :=
  { vertices : Fin 1995 → ℤ × ℤ // 
    ∀ i : Fin 1994, 
      let (x₁, y₁) := vertices i
      let (x₂, y₂) := vertices (i + 1)
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = i^2 + 4 ∧
    vertices 0 = vertices 1994 }

/-- Theorem stating that such a polygon cannot exist with all vertices on lattice points -/
theorem no_lattice_polygon1994 : ¬ ∃ (p : Polygon1994), True := by
  sorry

end NUMINAMATH_CALUDE_no_lattice_polygon1994_l3959_395942


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3959_395984

theorem quadratic_is_square_of_binomial :
  ∃ (r s : ℝ), (225/16 : ℝ) * x^2 + 15 * x + 4 = (r * x + s)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3959_395984


namespace NUMINAMATH_CALUDE_min_students_with_all_characteristics_l3959_395920

theorem min_students_with_all_characteristics 
  (total : ℕ) 
  (brown_eyes : ℕ) 
  (lunch_boxes : ℕ) 
  (glasses : ℕ) 
  (h1 : total = 35) 
  (h2 : brown_eyes = 15) 
  (h3 : lunch_boxes = 25) 
  (h4 : glasses = 10) : 
  ∃ (min_all : ℕ), min_all = 5 ∧ 
  min_all ≤ brown_eyes ∧ 
  min_all ≤ lunch_boxes ∧ 
  min_all ≤ glasses ∧
  min_all ≤ total ∧
  ∀ (x : ℕ), x < min_all → 
    ¬(x ≤ brown_eyes ∧ x ≤ lunch_boxes ∧ x ≤ glasses ∧ x ≤ total) :=
by
  sorry

end NUMINAMATH_CALUDE_min_students_with_all_characteristics_l3959_395920


namespace NUMINAMATH_CALUDE_absolute_value_minus_self_nonnegative_l3959_395914

theorem absolute_value_minus_self_nonnegative (m : ℚ) : 0 ≤ |m| - m := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_minus_self_nonnegative_l3959_395914


namespace NUMINAMATH_CALUDE_probability_of_dime_l3959_395955

/-- Represents the types of coins in the jar -/
inductive Coin
| Dime
| Nickel
| Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
| Coin.Dime => 10
| Coin.Nickel => 5
| Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def total_value : Coin → ℕ
| Coin.Dime => 500
| Coin.Nickel => 300
| Coin.Penny => 200

/-- The number of coins of each type in the jar -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Dime + coin_count Coin.Nickel + coin_count Coin.Penny

/-- The probability of randomly selecting a dime from the jar -/
theorem probability_of_dime : 
  (coin_count Coin.Dime : ℚ) / total_coins = 5 / 31 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_dime_l3959_395955


namespace NUMINAMATH_CALUDE_root_sum_transformation_l3959_395979

theorem root_sum_transformation (α β γ : ℂ) : 
  (x^3 - x + 1 = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_transformation_l3959_395979


namespace NUMINAMATH_CALUDE_fraction_equality_l3959_395916

theorem fraction_equality (x : ℝ) : (3 + x) / (5 + x) = (1 + x) / (2 + x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3959_395916


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3959_395956

theorem solution_set_quadratic_inequality :
  let S := {x : ℝ | 2 * x^2 - x - 3 ≥ 0}
  S = {x : ℝ | x ≤ -1 ∨ x ≥ 3/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3959_395956


namespace NUMINAMATH_CALUDE_gas_price_difference_l3959_395997

/-- Represents the price difference per gallon between two states --/
def price_difference (nc_price va_price : ℚ) : ℚ := va_price - nc_price

/-- Proves the price difference per gallon between Virginia and North Carolina --/
theorem gas_price_difference 
  (nc_gallons va_gallons : ℚ) 
  (nc_price : ℚ) 
  (total_spent : ℚ) :
  nc_gallons = 10 →
  va_gallons = 10 →
  nc_price = 2 →
  total_spent = 50 →
  price_difference nc_price ((total_spent - nc_gallons * nc_price) / va_gallons) = 1 := by
  sorry


end NUMINAMATH_CALUDE_gas_price_difference_l3959_395997


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3959_395921

theorem min_value_of_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) (hsum : x + y = 6) :
  ((x - 1)^2 / (y - 2) + (y - 1)^2 / (x - 2)) ≥ 8 ∧
  ∃ x y : ℝ, x > 2 ∧ y > 2 ∧ x + y = 6 ∧ ((x - 1)^2 / (y - 2) + (y - 1)^2 / (x - 2)) = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3959_395921


namespace NUMINAMATH_CALUDE_smallest_cube_box_volume_l3959_395962

/-- Represents the dimensions of a pyramid -/
structure PyramidDimensions where
  height : ℝ
  baseSide : ℝ

/-- Calculates the volume of a cube given its side length -/
def cubeVolume (side : ℝ) : ℝ := side ^ 3

/-- Theorem: The smallest cube-shaped box that can contain a pyramid with given dimensions has a volume of 3375 cubic inches -/
theorem smallest_cube_box_volume
  (pyramid : PyramidDimensions)
  (h_height : pyramid.height = 15)
  (h_base : pyramid.baseSide = 14) :
  cubeVolume (max pyramid.height pyramid.baseSide) = 3375 := by
  sorry

#eval cubeVolume 15  -- Should output 3375

end NUMINAMATH_CALUDE_smallest_cube_box_volume_l3959_395962


namespace NUMINAMATH_CALUDE_min_value_of_D_l3959_395908

noncomputable def D (x a : ℝ) : ℝ :=
  Real.sqrt ((x - a^2) + (Real.log x - a^2 / 4)^2) + a^2 / 4 + 1

theorem min_value_of_D :
  ∃ (m : ℝ), ∀ (x a : ℝ), D x a ≥ m ∧ ∃ (x₀ a₀ : ℝ), D x₀ a₀ = m ∧ m = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_D_l3959_395908


namespace NUMINAMATH_CALUDE_average_weight_problem_l3959_395966

/-- Given the average weights of three people and two of them, along with the weight of one person,
    prove that the average weight of the other two is as stated. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 → 
  (a + b) / 2 = 40 → 
  b = 33 → 
  (b + c) / 2 = 44 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l3959_395966


namespace NUMINAMATH_CALUDE_range_of_x_l3959_395961

open Set

def S : Set ℝ := {x | x ∈ Icc 2 5 ∨ x < 1 ∨ x > 4}

theorem range_of_x (h : ¬ ∀ x, x ∈ S) : 
  {x : ℝ | x ∈ Ico 1 2} = {x : ℝ | ¬ (x ∈ S)} := by sorry

end NUMINAMATH_CALUDE_range_of_x_l3959_395961


namespace NUMINAMATH_CALUDE_line_equation_proof_l3959_395952

/-- Given two lines in the xy-plane -/
def line1 (x y : ℝ) : Prop := x - 2*y - 3 = 0
def line2 (x y : ℝ) : Prop := 2*x - 3*y - 2 = 0

/-- The intersection point of the two lines -/
def intersection_point : ℝ × ℝ := sorry

/-- The equation of the line passing through (2, 1) and the intersection point -/
def target_line (x y : ℝ) : Prop := 5*x - 7*y - 3 = 0

theorem line_equation_proof :
  (∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ (x, y) = intersection_point) →
  target_line (2 : ℝ) 1 ∧
  target_line (intersection_point.1) (intersection_point.2) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3959_395952


namespace NUMINAMATH_CALUDE_inequality_preservation_l3959_395968

theorem inequality_preservation (a b c : ℝ) (h : a > b) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3959_395968


namespace NUMINAMATH_CALUDE_rectangular_plot_longer_side_l3959_395976

theorem rectangular_plot_longer_side 
  (width : ℝ) 
  (num_poles : ℕ) 
  (pole_distance : ℝ) :
  width = 40 ∧ 
  num_poles = 36 ∧ 
  pole_distance = 5 →
  ∃ length : ℝ, 
    length > width ∧
    2 * (length + width) = (num_poles - 1 : ℝ) * pole_distance ∧
    length = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_longer_side_l3959_395976


namespace NUMINAMATH_CALUDE_perpendicular_plane_implies_perpendicular_lines_not_always_perpendicular_plane_l3959_395990

-- Define the plane and lines
variable (α : Plane)
variable (a b c : Line)

-- Define the perpendicular relationship
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Define the condition that b and c are in plane α
def lines_in_plane (l1 l2 : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem perpendicular_plane_implies_perpendicular_lines 
  (h : lines_in_plane b c α) : 
  perpendicular a α → perpendicular_lines a b ∧ perpendicular_lines a c :=
sorry

-- State that the converse is not always true
theorem not_always_perpendicular_plane 
  (h : lines_in_plane b c α) : 
  ¬(∀ (a : Line), perpendicular_lines a b ∧ perpendicular_lines a c → perpendicular a α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_plane_implies_perpendicular_lines_not_always_perpendicular_plane_l3959_395990


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l3959_395994

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, (Real.sqrt (3 * Real.sqrt (t - 3)) = (9 - t) ^ (1/4)) → t = 3.6 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l3959_395994


namespace NUMINAMATH_CALUDE_product_over_sum_equals_6608_l3959_395939

theorem product_over_sum_equals_6608 : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) = 6608 := by
  sorry

end NUMINAMATH_CALUDE_product_over_sum_equals_6608_l3959_395939


namespace NUMINAMATH_CALUDE_wrong_mark_correction_l3959_395928

theorem wrong_mark_correction (n : ℕ) (initial_avg correct_avg : ℚ) (correct_mark : ℚ) (x : ℚ) 
  (h1 : n = 30)
  (h2 : initial_avg = 100)
  (h3 : correct_avg = 98)
  (h4 : correct_mark = 10) :
  (n : ℚ) * initial_avg - x + correct_mark = n * correct_avg → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_wrong_mark_correction_l3959_395928


namespace NUMINAMATH_CALUDE_highest_elevation_l3959_395934

/-- The elevation function of a particle projected vertically -/
def s (t : ℝ) : ℝ := 100 * t - 5 * t^2

/-- The initial velocity of the particle in meters per second -/
def initial_velocity : ℝ := 100

theorem highest_elevation :
  ∃ (t_max : ℝ), ∀ (t : ℝ), s t ≤ s t_max ∧ s t_max = 500 := by
  sorry

end NUMINAMATH_CALUDE_highest_elevation_l3959_395934


namespace NUMINAMATH_CALUDE_cube_surface_area_l3959_395925

/-- The surface area of a cube with edge length 6a is 216a² -/
theorem cube_surface_area (a : ℝ) : 
  let edge_length : ℝ := 6 * a
  let surface_area : ℝ := 6 * (edge_length ^ 2)
  surface_area = 216 * (a ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3959_395925


namespace NUMINAMATH_CALUDE_only_η_hypergeometric_l3959_395951

/-- Represents the total number of balls -/
def total_balls : ℕ := 10

/-- Represents the number of black balls -/
def black_balls : ℕ := 6

/-- Represents the number of white balls -/
def white_balls : ℕ := 4

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 4

/-- Represents the score for a black ball -/
def black_score : ℕ := 2

/-- Represents the score for a white ball -/
def white_score : ℕ := 1

/-- Represents the maximum number drawn -/
def X : ℕ → ℕ := sorry

/-- Represents the minimum number drawn -/
def Y : ℕ → ℕ := sorry

/-- Represents the total score of the drawn balls -/
def ξ : ℕ → ℕ := sorry

/-- Represents the number of black balls drawn -/
def η : ℕ → ℕ := sorry

/-- Defines a hypergeometric distribution -/
def is_hypergeometric (f : ℕ → ℕ) : Prop := sorry

theorem only_η_hypergeometric :
  is_hypergeometric η ∧
  ¬is_hypergeometric X ∧
  ¬is_hypergeometric Y ∧
  ¬is_hypergeometric ξ :=
sorry

end NUMINAMATH_CALUDE_only_η_hypergeometric_l3959_395951


namespace NUMINAMATH_CALUDE_power_function_through_point_l3959_395998

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 3 = Real.sqrt 3 → f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3959_395998


namespace NUMINAMATH_CALUDE_distance_to_focus_l3959_395992

-- Define the parabola C: y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point P
def P : ℝ × ℝ := (-2, 0)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define points A and B as intersection points
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ

-- A and B are on the parabola
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2

-- A, B, and P are collinear
axiom collinear : ∃ (t : ℝ), B.1 - P.1 = t * (A.1 - P.1) ∧ B.2 - P.2 = t * (A.2 - P.2)

-- |PA| = 1/2 |AB|
axiom distance_relation : Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) = 1/2 * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Theorem to prove
theorem distance_to_focus :
  Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2) = 5/3 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l3959_395992


namespace NUMINAMATH_CALUDE_eldorado_license_plates_l3959_395902

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letter positions in a license plate -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate -/
def digit_positions : ℕ := 4

/-- The total number of possible valid license plates in Eldorado -/
def total_license_plates : ℕ := num_letters ^ letter_positions * num_digits ^ digit_positions

theorem eldorado_license_plates :
  total_license_plates = 175760000 := by
  sorry

end NUMINAMATH_CALUDE_eldorado_license_plates_l3959_395902


namespace NUMINAMATH_CALUDE_total_cost_theorem_l3959_395985

/-- The cost of a single shirt -/
def shirt_cost : ℝ := sorry

/-- The cost of a single trouser -/
def trouser_cost : ℝ := sorry

/-- The cost of a single tie -/
def tie_cost : ℝ := sorry

/-- The total cost of 7 shirts, 2 trousers, and 2 ties is $50 -/
axiom condition1 : 7 * shirt_cost + 2 * trouser_cost + 2 * tie_cost = 50

/-- The total cost of 3 trousers, 5 shirts, and 2 ties is $70 -/
axiom condition2 : 5 * shirt_cost + 3 * trouser_cost + 2 * tie_cost = 70

/-- The theorem to be proved -/
theorem total_cost_theorem : 
  3 * shirt_cost + 4 * trouser_cost + 2 * tie_cost = 90 := by sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l3959_395985


namespace NUMINAMATH_CALUDE_hike_attendance_l3959_395905

/-- The number of people who went on the hike --/
def total_hikers (num_cars num_taxis num_vans : ℕ) 
                 (car_capacity taxi_capacity van_capacity : ℕ) : ℕ :=
  num_cars * car_capacity + num_taxis * taxi_capacity + num_vans * van_capacity

/-- Theorem stating that 58 people went on the hike --/
theorem hike_attendance : 
  total_hikers 3 6 2 4 6 5 = 58 := by
  sorry

end NUMINAMATH_CALUDE_hike_attendance_l3959_395905


namespace NUMINAMATH_CALUDE_expression_equals_three_l3959_395978

theorem expression_equals_three (m : ℝ) (h : m = -1) : 
  (2 * m + 3) * (2 * m - 3) - (m - 1) * (m + 5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_three_l3959_395978


namespace NUMINAMATH_CALUDE_cube_of_complex_number_l3959_395967

/-- Given that z = sin(π/3) + i*cos(π/3), prove that z^3 = i -/
theorem cube_of_complex_number (z : ℂ) (h : z = Complex.exp (Complex.I * (π / 3))) :
  z^3 = Complex.I := by sorry

end NUMINAMATH_CALUDE_cube_of_complex_number_l3959_395967


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3959_395929

theorem quadratic_equation_solutions : 
  ∀ x : ℝ, x^2 = 6*x ↔ x = 0 ∨ x = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3959_395929


namespace NUMINAMATH_CALUDE_simplify_expression_l3959_395911

theorem simplify_expression (m : ℝ) : (3*m + 2) - 3*(m^2 - m + 1) + (3 - 6*m) = -3*m^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3959_395911


namespace NUMINAMATH_CALUDE_pie_chart_statement_is_false_l3959_395912

-- Define the characteristics of different chart types
def BarChart : Type := Unit
def LineChart : Type := Unit
def PieChart : Type := Unit

-- Define what each chart type can represent
def represents_amount (chart : Type) : Prop := sorry
def represents_changes (chart : Type) : Prop := sorry
def represents_part_whole (chart : Type) : Prop := sorry

-- State the known characteristics of each chart type
axiom bar_chart_amount : represents_amount BarChart
axiom line_chart_amount_and_changes : represents_amount LineChart ∧ represents_changes LineChart
axiom pie_chart_part_whole : represents_part_whole PieChart

-- The statement we want to prove false
def pie_chart_statement : Prop :=
  represents_amount PieChart ∧ represents_changes PieChart

-- The theorem to prove
theorem pie_chart_statement_is_false : ¬pie_chart_statement := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_statement_is_false_l3959_395912


namespace NUMINAMATH_CALUDE_forty_percent_of_number_equals_144_l3959_395919

theorem forty_percent_of_number_equals_144 (x : ℝ) : 0.4 * x = 144 → x = 360 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_number_equals_144_l3959_395919


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l3959_395982

theorem complex_sum_of_powers (x y : ℂ) (hxy : x ≠ 0 ∧ y ≠ 0) (h : x^2 + x*y + y^2 = 0) :
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l3959_395982


namespace NUMINAMATH_CALUDE_machine_present_value_l3959_395953

/-- The present value of a machine given its depreciation rate, selling price after two years, and profit made. -/
theorem machine_present_value
  (depreciation_rate : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (h1 : depreciation_rate = 0.2)
  (h2 : selling_price = 118000.00000000001)
  (h3 : profit = 22000) :
  ∃ (present_value : ℝ),
    present_value = 150000.00000000002 ∧
    present_value * (1 - depreciation_rate)^2 = selling_price - profit :=
by sorry

end NUMINAMATH_CALUDE_machine_present_value_l3959_395953


namespace NUMINAMATH_CALUDE_race_track_outer_radius_l3959_395922

/-- Given a circular race track with an inner circumference of 880 m and a width of 25 m,
    the radius of the outer circle is 165 m. -/
theorem race_track_outer_radius :
  ∀ (inner_radius outer_radius : ℝ),
    inner_radius * 2 * Real.pi = 880 →
    outer_radius = inner_radius + 25 →
    outer_radius = 165 := by
  sorry

end NUMINAMATH_CALUDE_race_track_outer_radius_l3959_395922


namespace NUMINAMATH_CALUDE_custom_op_result_l3959_395949

/-- Custom operation * for non-zero integers -/
def custom_op (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

/-- Theorem stating the result of the custom operation given specific conditions -/
theorem custom_op_result (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 11) (h4 : a * b = 24) :
  custom_op a b = 11 / 24 := by
  sorry

#check custom_op_result

end NUMINAMATH_CALUDE_custom_op_result_l3959_395949


namespace NUMINAMATH_CALUDE_min_a_correct_l3959_395913

/-- The number of cards in the deck -/
def n : ℕ := 52

/-- The probability that Alex and Dylan are on the same team given Alex's card number a -/
def p (a : ℕ) : ℚ :=
  let lower := (n - (a + 6) + 1).choose 2
  let higher := (a - 1).choose 2
  (lower + higher : ℚ) / (n - 2).choose 2

/-- The minimum value of a such that p(a) ≥ 1/2 -/
def min_a : ℕ := 14

theorem min_a_correct :
  (∀ a < min_a, p a < 1/2) ∧ p min_a ≥ 1/2 :=
sorry

#eval min_a

end NUMINAMATH_CALUDE_min_a_correct_l3959_395913


namespace NUMINAMATH_CALUDE_unique_decimal_property_l3959_395991

theorem unique_decimal_property : ∃! (x : ℝ), x > 0 ∧ 10000 * x = 4 * (1 / x) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_decimal_property_l3959_395991
