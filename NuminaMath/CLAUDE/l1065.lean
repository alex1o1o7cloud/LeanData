import Mathlib

namespace NUMINAMATH_CALUDE_rational_equation_proof_l1065_106583

theorem rational_equation_proof (m n : ℚ) 
  (h1 : 3 * m + 2 * n = 0) 
  (h2 : m * n ≠ 0) : 
  m / n - n / m = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_proof_l1065_106583


namespace NUMINAMATH_CALUDE_only_prop3_correct_l1065_106563

-- Define a sequence as a function from ℕ to ℝ
def Sequence := ℕ → ℝ

-- Define the limit of a sequence
def LimitOf (a : Sequence) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - L| < ε

-- State the four propositions
def Prop1 (a : Sequence) (A : ℝ) : Prop :=
  LimitOf (fun n => (a n)^2) (A^2) → LimitOf a A

def Prop2 (a : Sequence) (A : ℝ) : Prop :=
  (∀ n, a n > 0) → LimitOf a A → A > 0

def Prop3 (a : Sequence) (A : ℝ) : Prop :=
  LimitOf a A → LimitOf (fun n => (a n)^2) (A^2)

def Prop4 (a b : Sequence) : Prop :=
  LimitOf (fun n => a n - b n) 0 → 
  (∃ L, LimitOf a L ∧ LimitOf b L)

-- Theorem stating that only Prop3 is correct
theorem only_prop3_correct :
  (¬ ∀ a A, Prop1 a A) ∧
  (¬ ∀ a A, Prop2 a A) ∧
  (∀ a A, Prop3 a A) ∧
  (¬ ∀ a b, Prop4 a b) :=
sorry

end NUMINAMATH_CALUDE_only_prop3_correct_l1065_106563


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l1065_106598

/-- Given two parabolas that intersect the coordinate axes in four points forming a kite -/
structure ParabolaKite where
  a' : ℝ
  b' : ℝ
  intersection_points : Fin 4 → ℝ × ℝ
  is_kite : Bool
  kite_area : ℝ

/-- The theorem stating the sum of a' and b' -/
theorem parabola_kite_sum (pk : ParabolaKite)
  (h1 : pk.is_kite = true)
  (h2 : pk.kite_area = 18)
  (h3 : ∀ (i : Fin 4), (pk.intersection_points i).1 = 0 ∨ (pk.intersection_points i).2 = 0)
  (h4 : ∀ (x y : ℝ), y = pk.a' * x^2 + 3 ∨ y = 6 - pk.b' * x^2) :
  pk.a' + pk.b' = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l1065_106598


namespace NUMINAMATH_CALUDE_share_division_l1065_106506

/-- The problem of dividing $500 among five people with specific ratios -/
theorem share_division (total : ℚ) (a b c d e : ℚ) : 
  total = 500 →
  a = (5/7) * (b + c + d + e) →
  b = (11/16) * (a + c + d) →
  c = (3/8) * (a + b + e) →
  d = (7/12) * (a + b + c) →
  a + b + c + d + e = total →
  a = (5/12) * total :=
by sorry

end NUMINAMATH_CALUDE_share_division_l1065_106506


namespace NUMINAMATH_CALUDE_cycle_selling_price_l1065_106585

/-- Calculates the selling price of a cycle given its cost price and gain percent -/
def calculate_selling_price (cost_price : ℚ) (gain_percent : ℚ) : ℚ :=
  cost_price * (1 + gain_percent / 100)

/-- Theorem: The selling price of a cycle bought for 840 with 45.23809523809524% gain is 1220 -/
theorem cycle_selling_price : 
  calculate_selling_price 840 45.23809523809524 = 1220 := by
  sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l1065_106585


namespace NUMINAMATH_CALUDE_min_sum_tangents_l1065_106593

/-- In an acute triangle ABC, given that a = 2b * sin(C), 
    the minimum value of tan(A) + tan(B) + tan(C) is 3√3 -/
theorem min_sum_tangents (A B C : Real) (a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  a = 2 * b * Real.sin C ∧  -- Given condition
  a = c * Real.sin A ∧  -- Law of sines
  b = c * Real.sin B ∧  -- Law of sines
  c = c * Real.sin C  -- Law of sines
  →
  (Real.tan A + Real.tan B + Real.tan C ≥ 3 * (3 : Real).sqrt) ∧
  ∃ (A' B' C' : Real), 
    0 < A' ∧ 0 < B' ∧ 0 < C' ∧
    A' + B' + C' = π ∧
    Real.tan A' + Real.tan B' + Real.tan C' = 3 * (3 : Real).sqrt :=
by sorry

end NUMINAMATH_CALUDE_min_sum_tangents_l1065_106593


namespace NUMINAMATH_CALUDE_cost_of_shoes_l1065_106581

def monthly_allowance : ℕ := 5
def months_saved : ℕ := 3
def lawn_mowing_fee : ℕ := 15
def lawns_mowed : ℕ := 4
def driveway_shoveling_fee : ℕ := 7
def driveways_shoveled : ℕ := 5
def change_left : ℕ := 15

def total_saved : ℕ := 
  monthly_allowance * months_saved + 
  lawn_mowing_fee * lawns_mowed + 
  driveway_shoveling_fee * driveways_shoveled

theorem cost_of_shoes : 
  total_saved - change_left = 95 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_shoes_l1065_106581


namespace NUMINAMATH_CALUDE_total_investment_l1065_106589

/-- Given two investments with different interest rates, proves the total amount invested. -/
theorem total_investment (amount_at_8_percent : ℝ) (amount_at_9_percent : ℝ) 
  (h1 : amount_at_8_percent = 6000)
  (h2 : 0.08 * amount_at_8_percent + 0.09 * amount_at_9_percent = 840) :
  amount_at_8_percent + amount_at_9_percent = 10000 := by
  sorry

end NUMINAMATH_CALUDE_total_investment_l1065_106589


namespace NUMINAMATH_CALUDE_internal_diagonal_intersects_576_cubes_l1065_106570

def rectangular_solid_dimensions : ℕ × ℕ × ℕ := (120, 210, 336)

-- Function to calculate the number of cubes intersected by the diagonal
def intersected_cubes (dims : ℕ × ℕ × ℕ) : ℕ :=
  let (x, y, z) := dims
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

theorem internal_diagonal_intersects_576_cubes :
  intersected_cubes rectangular_solid_dimensions = 576 := by
  sorry

end NUMINAMATH_CALUDE_internal_diagonal_intersects_576_cubes_l1065_106570


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1065_106505

theorem trigonometric_simplification :
  let tan_sum := Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
                 Real.tan (40 * π / 180) + Real.tan (60 * π / 180)
  tan_sum / Real.sin (80 * π / 180) = 
    2 * (Real.cos (40 * π / 180) / (Real.sqrt 3 * Real.cos (10 * π / 180) * Real.cos (20 * π / 180)) + 
         2 / Real.cos (40 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1065_106505


namespace NUMINAMATH_CALUDE_oprah_car_collection_reduction_l1065_106599

def reduce_car_collection (initial_cars : ℕ) (target_cars : ℕ) (cars_given_per_year : ℕ) : ℕ :=
  (initial_cars - target_cars) / cars_given_per_year

theorem oprah_car_collection_reduction :
  reduce_car_collection 3500 500 50 = 60 := by
  sorry

end NUMINAMATH_CALUDE_oprah_car_collection_reduction_l1065_106599


namespace NUMINAMATH_CALUDE_remainder_thirteen_six_twelve_seven_eleven_eight_mod_five_l1065_106503

theorem remainder_thirteen_six_twelve_seven_eleven_eight_mod_five :
  (13^6 + 12^7 + 11^8) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_thirteen_six_twelve_seven_eleven_eight_mod_five_l1065_106503


namespace NUMINAMATH_CALUDE_number_count_l1065_106582

theorem number_count (n : ℕ) 
  (h1 : (n : ℝ) * 30 = (4 : ℝ) * 25 + (3 : ℝ) * 35 - 25)
  (h2 : n > 4) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_count_l1065_106582


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l1065_106557

def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

theorem derivative_f_at_one :
  deriv f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l1065_106557


namespace NUMINAMATH_CALUDE_line_equation_l1065_106571

/-- Given a line with inclination angle π/3 and y-intercept 2, its equation is √3x - y + 2 = 0 -/
theorem line_equation (x y : ℝ) :
  let angle : ℝ := π / 3
  let y_intercept : ℝ := 2
  (Real.sqrt 3 * x - y + y_intercept = 0) ↔ 
    (y = Real.tan angle * x + y_intercept) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l1065_106571


namespace NUMINAMATH_CALUDE_parallelogram_area_l1065_106550

-- Define the conversion factor
def inch_to_mm : ℝ := 25.4

-- Define the parallelogram's dimensions
def base_inches : ℝ := 18
def height_mm : ℝ := 25.4

-- Theorem statement
theorem parallelogram_area :
  (base_inches * (height_mm / inch_to_mm)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1065_106550


namespace NUMINAMATH_CALUDE_equation_solution_l1065_106513

theorem equation_solution : ∃ (x y : ℝ), 
  (1 / 6 + 6 / x = 14 / x + 1 / 14 + y) ∧ (x = 84) ∧ (y = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1065_106513


namespace NUMINAMATH_CALUDE_divide_by_reciprocal_twelve_divided_by_one_twelfth_l1065_106509

theorem divide_by_reciprocal (x y : ℚ) (h : y ≠ 0) : x / y = x * (1 / y) := by sorry

theorem twelve_divided_by_one_twelfth : 12 / (1 / 12) = 144 := by sorry

end NUMINAMATH_CALUDE_divide_by_reciprocal_twelve_divided_by_one_twelfth_l1065_106509


namespace NUMINAMATH_CALUDE_initial_crayons_count_l1065_106536

/-- 
Given:
- initial_crayons is the number of crayons initially in the drawer
- added_crayons is the number of crayons Benny added (3)
- total_crayons is the total number of crayons after adding (12)

Prove that the initial number of crayons is 9.
-/
theorem initial_crayons_count (initial_crayons added_crayons total_crayons : ℕ) 
  (h1 : added_crayons = 3)
  (h2 : total_crayons = 12)
  (h3 : initial_crayons + added_crayons = total_crayons) : 
  initial_crayons = 9 := by
sorry

end NUMINAMATH_CALUDE_initial_crayons_count_l1065_106536


namespace NUMINAMATH_CALUDE_point_transformation_l1065_106597

/-- Rotation of 90° counterclockwise around a point -/
def rotate90 (x y cx cy : ℝ) : ℝ × ℝ :=
  (cx - (y - cy), cy + (x - cx))

/-- Reflection about y = x line -/
def reflectYeqX (x y : ℝ) : ℝ × ℝ := (y, x)

/-- The main theorem -/
theorem point_transformation (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate90 a b 2 3
  let final := reflectYeqX rotated.1 rotated.2
  final = (5, 1) → b - a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1065_106597


namespace NUMINAMATH_CALUDE_perpendicular_condition_parallel_condition_l1065_106555

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 2]

-- Define the dot product of two 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define perpendicularity of two 2D vectors
def perpendicular (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

-- Define parallelism of two 2D vectors
def parallel (v w : Fin 2 → ℝ) : Prop := ∃ (c : ℝ), ∀ (i : Fin 2), v i = c * w i

-- Define the vector operations
def add_vectors (v w : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => v i + w i
def scale_vector (k : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => k * v i

-- Theorem statements
theorem perpendicular_condition (k : ℝ) : 
  perpendicular (add_vectors (scale_vector k a) b) (add_vectors a (scale_vector (-3) b)) ↔ k = -3 := by sorry

theorem parallel_condition (k : ℝ) : 
  parallel (add_vectors (scale_vector k a) b) (add_vectors a (scale_vector (-3) b)) ↔ k = -1/3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_parallel_condition_l1065_106555


namespace NUMINAMATH_CALUDE_max_value_z_l1065_106586

theorem max_value_z (x y : ℝ) (h1 : 6 ≤ x + y) (h2 : x + y ≤ 8) (h3 : -2 ≤ x - y) (h4 : x - y ≤ 0) :
  ∃ (z : ℝ), z = 2 * x + 5 * y ∧ z ≤ 8 ∧ ∀ (w : ℝ), w = 2 * x + 5 * y → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_value_z_l1065_106586


namespace NUMINAMATH_CALUDE_thirtieth_term_is_61_l1065_106594

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem thirtieth_term_is_61 :
  arithmetic_sequence 3 2 30 = 61 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_is_61_l1065_106594


namespace NUMINAMATH_CALUDE_complex_magnitude_l1065_106518

theorem complex_magnitude (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1065_106518


namespace NUMINAMATH_CALUDE_percentage_passed_both_subjects_l1065_106522

theorem percentage_passed_both_subjects (total_students : ℕ) 
  (failed_hindi : ℕ) (failed_english : ℕ) (failed_both : ℕ) :
  failed_hindi = (35 * total_students) / 100 →
  failed_english = (45 * total_students) / 100 →
  failed_both = (20 * total_students) / 100 →
  total_students > 0 →
  ((total_students - (failed_hindi + failed_english - failed_both)) * 100) / total_students = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_passed_both_subjects_l1065_106522


namespace NUMINAMATH_CALUDE_ron_siblings_product_l1065_106559

/-- Represents a family structure --/
structure Family :=
  (sisters : ℕ)
  (brothers : ℕ)

/-- The problem setup --/
def problem_setup (harry_family : Family) (harriet : Family) (ron : Family) : Prop :=
  harry_family.sisters = 4 ∧
  harry_family.brothers = 6 ∧
  harriet.sisters = harry_family.sisters - 1 ∧
  harriet.brothers = harry_family.brothers ∧
  ron.sisters = harriet.sisters ∧
  ron.brothers = harriet.brothers + 2

/-- The main theorem --/
theorem ron_siblings_product (harry_family : Family) (harriet : Family) (ron : Family) 
  (h : problem_setup harry_family harriet ron) : 
  ron.sisters * ron.brothers = 32 := by
  sorry


end NUMINAMATH_CALUDE_ron_siblings_product_l1065_106559


namespace NUMINAMATH_CALUDE_f_simplification_f_value_in_second_quadrant_l1065_106588

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 5 * Real.pi / 2) * Real.cos (3 * Real.pi / 2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (Real.pi - α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by sorry

theorem f_value_in_second_quadrant (α : Real) 
  (h1 : Real.cos (α + 3 * Real.pi / 2) = 1/5) 
  (h2 : 0 < α ∧ α < Real.pi) : 
  f α = 2 * Real.sqrt 6 / 5 := by sorry

end NUMINAMATH_CALUDE_f_simplification_f_value_in_second_quadrant_l1065_106588


namespace NUMINAMATH_CALUDE_domain_of_f_l1065_106584

def f (x : ℝ) : ℝ := (x + 1) ^ 0

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -1} :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l1065_106584


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_example_l1065_106504

/-- Calculates the interval for systematic sampling -/
def systematicSamplingInterval (population : ℕ) (sampleSize : ℕ) : ℕ :=
  population / sampleSize

/-- Theorem: The systematic sampling interval for 1000 students with a sample size of 50 is 20 -/
theorem systematic_sampling_interval_example :
  systematicSamplingInterval 1000 50 = 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_example_l1065_106504


namespace NUMINAMATH_CALUDE_singh_family_seating_arrangements_l1065_106502

/-- Represents a family with parents and children -/
structure Family :=
  (parents : ℕ)
  (children : ℕ)

/-- Represents a van with front and back seats -/
structure Van :=
  (front_seats : ℕ)
  (back_seats : ℕ)

/-- Calculates the number of seating arrangements for a family in a van -/
def seating_arrangements (f : Family) (v : Van) : ℕ :=
  sorry

/-- The Singh family -/
def singh_family : Family :=
  { parents := 2, children := 3 }

/-- The Singh family van -/
def singh_van : Van :=
  { front_seats := 2, back_seats := 3 }

theorem singh_family_seating_arrangements :
  seating_arrangements singh_family singh_van = 48 :=
sorry

end NUMINAMATH_CALUDE_singh_family_seating_arrangements_l1065_106502


namespace NUMINAMATH_CALUDE_red_ball_probability_l1065_106558

theorem red_ball_probability (w r : ℕ) : 
  r > w ∧ r < 2 * w ∧ 2 * w + 3 * r = 60 → 
  (r : ℚ) / (w + r : ℚ) = 14 / 23 := by
sorry

end NUMINAMATH_CALUDE_red_ball_probability_l1065_106558


namespace NUMINAMATH_CALUDE_andrews_apples_l1065_106561

theorem andrews_apples (n : ℕ) : 
  (6 * n = 5 * (n + 2)) → (6 * n = 60) := by
  sorry

end NUMINAMATH_CALUDE_andrews_apples_l1065_106561


namespace NUMINAMATH_CALUDE_ethans_candles_weight_l1065_106549

/-- The combined weight of Ethan's candles -/
def combined_weight (total_candles : ℕ) (beeswax_per_candle : ℕ) (coconut_oil_per_candle : ℕ) : ℕ :=
  total_candles * (beeswax_per_candle + coconut_oil_per_candle)

/-- Theorem: The combined weight of Ethan's candles is 63 ounces -/
theorem ethans_candles_weight :
  combined_weight (10 - 3) 8 1 = 63 := by
  sorry

end NUMINAMATH_CALUDE_ethans_candles_weight_l1065_106549


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l1065_106516

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l1065_106516


namespace NUMINAMATH_CALUDE_sum_of_50th_terms_l1065_106535

theorem sum_of_50th_terms (a₁ a₅₀ : ℝ) (d : ℝ) (g₁ g₅₀ : ℝ) (r : ℝ) : 
  a₁ = 3 → d = 6 → g₁ = 2 → r = 3 →
  a₅₀ = a₁ + 49 * d →
  g₅₀ = g₁ * r^49 →
  a₅₀ + g₅₀ = 297 + 2 * 3^49 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_50th_terms_l1065_106535


namespace NUMINAMATH_CALUDE_agnes_flight_cost_l1065_106528

/-- Represents the cost structure for different transportation modes -/
structure TransportCost where
  busCostPerKm : ℝ
  airplaneCostPerKm : ℝ
  airplaneBookingFee : ℝ

/-- Represents the distances between cities -/
structure CityDistances where
  xToY : ℝ
  xToZ : ℝ

/-- Calculates the cost of an airplane trip -/
def airplaneTripCost (cost : TransportCost) (distance : ℝ) : ℝ :=
  cost.airplaneBookingFee + cost.airplaneCostPerKm * distance

theorem agnes_flight_cost (cost : TransportCost) (distances : CityDistances) :
  cost.busCostPerKm = 0.20 →
  cost.airplaneCostPerKm = 0.12 →
  cost.airplaneBookingFee = 120 →
  distances.xToY = 4500 →
  distances.xToZ = 4000 →
  airplaneTripCost cost distances.xToY = 660 := by
  sorry


end NUMINAMATH_CALUDE_agnes_flight_cost_l1065_106528


namespace NUMINAMATH_CALUDE_quadratic_polynomial_proof_l1065_106560

theorem quadratic_polynomial_proof : ∃ (q : ℝ → ℝ),
  (∀ x, q x = (19 * x^2 - 2 * x + 13) / 15) ∧
  q (-2) = 9 ∧
  q 1 = 2 ∧
  q 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_proof_l1065_106560


namespace NUMINAMATH_CALUDE_older_sibling_age_l1065_106520

def mother_charge : ℚ := 495 / 100
def child_charge_per_year : ℚ := 35 / 100
def total_bill : ℚ := 985 / 100

def is_valid_age_combination (twin_age older_age : ℕ) : Prop :=
  twin_age ≤ older_age ∧
  mother_charge + child_charge_per_year * (2 * twin_age + older_age) = total_bill

theorem older_sibling_age :
  ∃ (twin_age older_age : ℕ), is_valid_age_combination twin_age older_age ∧
  (older_age = 4 ∨ older_age = 6) :=
by sorry

end NUMINAMATH_CALUDE_older_sibling_age_l1065_106520


namespace NUMINAMATH_CALUDE_set_A_representation_l1065_106547

def A : Set (ℤ × ℤ) := {p | p.1^2 = p.2 + 1 ∧ |p.1| < 2}

theorem set_A_representation : A = {(-1, 0), (0, -1), (1, 0)} := by
  sorry

end NUMINAMATH_CALUDE_set_A_representation_l1065_106547


namespace NUMINAMATH_CALUDE_age_difference_proof_l1065_106569

theorem age_difference_proof :
  ∀ (a b : ℕ),
    a + b = 2 →
    (10 * a + b) + (10 * b + a) = 22 →
    (10 * a + b + 7) = 3 * (10 * b + a + 7) →
    (10 * a + b) - (10 * b + a) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1065_106569


namespace NUMINAMATH_CALUDE_triangle_similarity_l1065_106548

-- Define the points in the plane
variable (A B C A' B' C' S M N : ℝ × ℝ)

-- Define the properties of the triangles and points
def is_equilateral (X Y Z : ℝ × ℝ) : Prop := sorry
def is_center (S X Y Z : ℝ × ℝ) : Prop := sorry
def is_midpoint (M X Y : ℝ × ℝ) : Prop := sorry
def are_similar (T1 T2 T3 U1 U2 U3 : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem triangle_similarity 
  (h1 : is_equilateral A B C)
  (h2 : is_equilateral A' B' C')
  (h3 : is_center S A B C)
  (h4 : A' ≠ S)
  (h5 : B' ≠ S)
  (h6 : is_midpoint M A' B)
  (h7 : is_midpoint N A B') :
  are_similar S B' M S A' N := by sorry

end NUMINAMATH_CALUDE_triangle_similarity_l1065_106548


namespace NUMINAMATH_CALUDE_find_divisor_l1065_106525

theorem find_divisor (x : ℕ) (h_x : x = 75) :
  ∃ D : ℕ,
    (∃ Q R : ℕ, x = D * Q + R ∧ R < D ∧ Q = (x % 34) + 8) ∧
    (∀ D' : ℕ, D' < D → ¬(∃ Q' R' : ℕ, x = D' * Q' + R' ∧ R' < D' ∧ Q' = (x % 34) + 8)) ∧
    D = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1065_106525


namespace NUMINAMATH_CALUDE_eighth_number_in_set_l1065_106534

theorem eighth_number_in_set (known_numbers : List ℕ) (average : ℚ) : 
  known_numbers = [1, 2, 4, 5, 6, 9, 9, 12] ∧ 
  average = 7 ∧
  (List.sum known_numbers + 12) / 9 = average →
  ∃ x : ℕ, x = 3 ∧ x ∈ (known_numbers ++ [12]) :=
by sorry

end NUMINAMATH_CALUDE_eighth_number_in_set_l1065_106534


namespace NUMINAMATH_CALUDE_fraction_simplification_l1065_106565

theorem fraction_simplification (x : ℝ) (h : x = 3) :
  (x^8 + 20*x^4 + 100) / (x^4 + 10) = 91 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1065_106565


namespace NUMINAMATH_CALUDE_simplify_negative_fraction_power_l1065_106527

theorem simplify_negative_fraction_power : 
  ((-1 : ℝ) / 343) ^ (-(2 : ℝ) / 3) = 49 := by
  sorry

end NUMINAMATH_CALUDE_simplify_negative_fraction_power_l1065_106527


namespace NUMINAMATH_CALUDE_more_wins_probability_correct_l1065_106553

/-- The probability of winning, losing, or tying a single match -/
def match_probability : ℚ := 1/3

/-- The number of matches played -/
def num_matches : ℕ := 6

/-- The probability of finishing with more wins than losses -/
def more_wins_probability : ℚ := 98/243

theorem more_wins_probability_correct :
  let outcomes := 3^num_matches
  let equal_wins_losses := (num_matches.choose (num_matches/2))
                         + (num_matches.choose ((num_matches-2)/2)) * (num_matches.choose 2)
                         + (num_matches.choose ((num_matches-4)/2)) * (num_matches.choose 4)
                         + 1
  (1 - equal_wins_losses / outcomes) / 2 = more_wins_probability :=
sorry

end NUMINAMATH_CALUDE_more_wins_probability_correct_l1065_106553


namespace NUMINAMATH_CALUDE_water_tower_height_l1065_106519

/-- Given a bamboo pole and a water tower under the same lighting conditions,
    this theorem proves the height of the water tower based on the similar triangles concept. -/
theorem water_tower_height
  (bamboo_height : ℝ)
  (bamboo_shadow : ℝ)
  (tower_shadow : ℝ)
  (h_bamboo_height : bamboo_height = 2)
  (h_bamboo_shadow : bamboo_shadow = 1.5)
  (h_tower_shadow : tower_shadow = 24) :
  bamboo_height / bamboo_shadow * tower_shadow = 32 :=
by sorry

end NUMINAMATH_CALUDE_water_tower_height_l1065_106519


namespace NUMINAMATH_CALUDE_incorrect_statement_l1065_106573

theorem incorrect_statement (p q : Prop) 
  (hp : p ↔ (2 + 2 = 5)) 
  (hq : q ↔ (3 > 2)) : 
  ¬((¬p ∧ ¬q) ∧ ¬p) := by
sorry

end NUMINAMATH_CALUDE_incorrect_statement_l1065_106573


namespace NUMINAMATH_CALUDE_triangle_properties_l1065_106537

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle -/
def area (t : Triangle) : ℝ := sorry

theorem triangle_properties (t : Triangle) 
  (h : area t = t.a^2 / 2) : 
  (Real.tan t.A = 2 * t.a^2 / (t.b^2 + t.c^2 - t.a^2)) ∧ 
  (∃ (x : ℝ), x = Real.sqrt 5 ∧ ∀ (y : ℝ), t.c / t.b + t.b / t.c ≤ x) ∧
  (∃ (m : ℝ), ∀ (x : ℝ), m ≤ t.b * t.c / t.a^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1065_106537


namespace NUMINAMATH_CALUDE_courses_last_year_is_six_l1065_106515

/-- Represents the number of courses taken last year -/
def courses_last_year : ℕ := 6

/-- Represents the average grade for the entire two-year period -/
def two_year_average : ℚ := 81

/-- Represents the number of courses taken the year before last -/
def courses_year_before : ℕ := 5

/-- Represents the average grade for the year before last -/
def average_year_before : ℚ := 60

/-- Represents the average grade for last year -/
def average_last_year : ℚ := 100

theorem courses_last_year_is_six :
  (courses_year_before * average_year_before + courses_last_year * average_last_year) / 
  (courses_year_before + courses_last_year : ℚ) = two_year_average :=
sorry

end NUMINAMATH_CALUDE_courses_last_year_is_six_l1065_106515


namespace NUMINAMATH_CALUDE_seven_pow_2015_ends_with_43_l1065_106587

/-- The last two digits of a natural number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- 7^2015 ends with 43 -/
theorem seven_pow_2015_ends_with_43 : lastTwoDigits (7^2015) = 43 := by
  sorry

#check seven_pow_2015_ends_with_43

end NUMINAMATH_CALUDE_seven_pow_2015_ends_with_43_l1065_106587


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1065_106552

theorem quadratic_equation_solution (k : ℝ) : 
  (8 * ((-15 - Real.sqrt 145) / 8)^2 + 15 * ((-15 - Real.sqrt 145) / 8) + k = 0) → 
  (k = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1065_106552


namespace NUMINAMATH_CALUDE_problem_solution_l1065_106540

theorem problem_solution (a b m n : ℚ) 
  (ha_neg : a < 0) 
  (ha_abs : |a| = 7/4) 
  (hb_recip : b⁻¹ = -3/2) 
  (hmn_opp : m = -n) : 
  4 * a / b + 3 * (m + n) = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1065_106540


namespace NUMINAMATH_CALUDE_problem_solution_l1065_106524

-- Define the set of integers
def Z : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n}

-- Define the set of x satisfying the conditions
def S : Set ℝ := {x : ℝ | (|x - 1| < 2 ∨ x ∉ Z) ∧ x ∈ Z}

-- Define the target set
def T : Set ℝ := {0, 1, 2}

-- Theorem statement
theorem problem_solution : S = T := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1065_106524


namespace NUMINAMATH_CALUDE_distribute_and_combine_l1065_106517

theorem distribute_and_combine (a b : ℝ) : 2 * (a - b) + 3 * b = 2 * a + b := by
  sorry

end NUMINAMATH_CALUDE_distribute_and_combine_l1065_106517


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1065_106556

theorem largest_angle_in_special_triangle (A B C : Real) (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = π) (h3 : Real.sin A / Real.sin B = 3 / 5)
  (h4 : Real.sin B / Real.sin C = 5 / 7) :
  max A (max B C) = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1065_106556


namespace NUMINAMATH_CALUDE_smallest_equal_burgers_and_buns_l1065_106578

theorem smallest_equal_burgers_and_buns :
  ∃ n : ℕ+, (∀ k : ℕ+, (∃ m : ℕ+, 5 * k = 7 * m) → n ≤ k) ∧ (∃ m : ℕ+, 5 * n = 7 * m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_equal_burgers_and_buns_l1065_106578


namespace NUMINAMATH_CALUDE_fraction_equality_l1065_106591

theorem fraction_equality (a b : ℝ) (x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + b) / (a - b) = (x + 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1065_106591


namespace NUMINAMATH_CALUDE_distance_difference_l1065_106577

/-- The line l -/
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

/-- The ellipse C₁ -/
def ellipse_C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Point F -/
def point_F : ℝ × ℝ := (1, 0)

/-- Point F₁ -/
def point_F₁ : ℝ × ℝ := (-1, 0)

/-- Theorem stating the difference of distances -/
theorem distance_difference (A B : ℝ × ℝ) 
  (h_line_A : line_l A.1 A.2)
  (h_line_B : line_l B.1 B.2)
  (h_ellipse_A : ellipse_C₁ A.1 A.2)
  (h_ellipse_B : ellipse_C₁ B.1 B.2)
  (h_above : A.2 > B.2) :
  |point_F₁.1 - A.1|^2 + |point_F₁.2 - A.2|^2 - 
  (|point_F₁.1 - B.1|^2 + |point_F₁.2 - B.2|^2) = (6 * Real.sqrt 2 / 7)^2 :=
sorry

end NUMINAMATH_CALUDE_distance_difference_l1065_106577


namespace NUMINAMATH_CALUDE_jumping_contest_l1065_106508

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper frog mouse : ℕ) 
  (h1 : grasshopper = 14)
  (h2 : mouse = frog - 16)
  (h3 : mouse = grasshopper + 21) :
  frog - grasshopper = 37 := by
  sorry

end NUMINAMATH_CALUDE_jumping_contest_l1065_106508


namespace NUMINAMATH_CALUDE_wooden_stick_problem_xiao_hong_age_problem_l1065_106575

-- Problem 1: Wooden stick
theorem wooden_stick_problem (x : ℝ) :
  60 - 2 * x = 10 → x = 25 := by sorry

-- Problem 2: Xiao Hong's age
theorem xiao_hong_age_problem (y : ℝ) :
  2 * y + 10 = 30 → y = 10 := by sorry

end NUMINAMATH_CALUDE_wooden_stick_problem_xiao_hong_age_problem_l1065_106575


namespace NUMINAMATH_CALUDE_right_triangle_m_values_l1065_106590

-- Define points A, B, and P
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (4, 0)
def P (m : ℝ) : ℝ × ℝ := (m, 0.5 * m + 2)

-- Define the condition for a right-angled triangle
def isRightAngled (a b c : ℝ × ℝ) : Prop :=
  (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0 ∨
  (a.1 - b.1) * (c.1 - b.1) + (a.2 - b.2) * (c.2 - b.2) = 0 ∨
  (a.1 - c.1) * (b.1 - c.1) + (a.2 - c.2) * (b.2 - c.2) = 0

-- State the theorem
theorem right_triangle_m_values :
  ∀ m : ℝ, isRightAngled A B (P m) →
    m = -2 ∨ m = 4 ∨ m = (4 * Real.sqrt 5) / 5 ∨ m = -(4 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_m_values_l1065_106590


namespace NUMINAMATH_CALUDE_fertilizer_production_equation_l1065_106554

/-- Given a fertilizer factory with:
  * Original production plan of x tons per day
  * New production of x + 3 tons per day
  * Time to produce 180 tons (new rate) = Time to produce 120 tons (original rate)
  Prove that the equation 120/x = 180/(x + 3) correctly represents the relationship
  between the original production rate x and the time taken to produce different
  quantities of fertilizer. -/
theorem fertilizer_production_equation (x : ℝ) (h : x > 0) :
  (120 : ℝ) / x = 180 / (x + 3) ↔
  (120 : ℝ) / x = (180 : ℝ) / (x + 3) :=
by sorry

end NUMINAMATH_CALUDE_fertilizer_production_equation_l1065_106554


namespace NUMINAMATH_CALUDE_sushi_eating_orders_l1065_106510

/-- Represents a 2 × 3 grid of sushi pieces -/
structure SushiGrid :=
  (pieces : Fin 6 → Bool)

/-- Checks if a piece is adjacent to at most two other pieces -/
def isEatable (grid : SushiGrid) (pos : Fin 6) : Bool :=
  sorry

/-- Generates all valid eating orders for a given SushiGrid -/
def validEatingOrders (grid : SushiGrid) : List (List (Fin 6)) :=
  sorry

/-- The number of valid eating orders for a 2 × 3 sushi grid -/
def numValidOrders : Nat :=
  sorry

theorem sushi_eating_orders :
  numValidOrders = 360 :=
sorry

end NUMINAMATH_CALUDE_sushi_eating_orders_l1065_106510


namespace NUMINAMATH_CALUDE_not_all_products_of_two_primes_l1065_106579

theorem not_all_products_of_two_primes (q : ℕ) (hq : Nat.Prime q) (hodd : Odd q) :
  ∃ k : ℕ, k ∈ Finset.range (q - 1) ∧ ¬∃ p₁ p₂ : ℕ, Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ k^2 + k + q = p₁ * p₂ := by
  sorry

end NUMINAMATH_CALUDE_not_all_products_of_two_primes_l1065_106579


namespace NUMINAMATH_CALUDE_point_b_location_l1065_106531

/-- Represents a point on the number line -/
structure Point where
  value : ℝ

/-- The distance between two points on the number line -/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_b_location (a b : Point) :
  a.value = -2 ∧ distance a b = 3 → b.value = -5 ∨ b.value = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_b_location_l1065_106531


namespace NUMINAMATH_CALUDE_mean_reading_days_l1065_106562

def reading_data : List (Nat × Nat) := [
  (2, 1), (4, 2), (5, 3), (10, 4), (7, 5), (3, 6), (2, 7)
]

def total_days : Nat := (reading_data.map (λ (students, days) => students * days)).sum

def total_students : Nat := (reading_data.map (λ (students, _) => students)).sum

theorem mean_reading_days : 
  (total_days : ℚ) / (total_students : ℚ) = 4 := by sorry

end NUMINAMATH_CALUDE_mean_reading_days_l1065_106562


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1065_106576

def expression (x : ℝ) : ℝ :=
  4 * (x^2 - 2*x^3 + 2*x) + 2 * (x + 3*x^3 - 2*x^2 + 4*x^5 - x^3) - 6 * (2 + 2*x - 5*x^3 - 3*x^2 + x^4)

theorem coefficient_of_x_cubed : 
  (deriv (deriv (deriv expression))) 0 / 6 = 26 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1065_106576


namespace NUMINAMATH_CALUDE_parabola_equation_l1065_106512

/-- The equation of a parabola with focus at (2, 1) and the y-axis as its directrix -/
theorem parabola_equation (x y : ℝ) :
  let focus : ℝ × ℝ := (2, 1)
  let directrix : Set (ℝ × ℝ) := {p | p.1 = 0}
  let parabola_equation : ℝ × ℝ → Prop := λ p => (p.2 - 1)^2 = 4 * (p.1 - 1)
  (∀ p, p ∈ directrix → dist p focus = dist p (x, y)) ↔ parabola_equation (x, y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1065_106512


namespace NUMINAMATH_CALUDE_hawkeye_battery_budget_l1065_106501

/-- Hawkeye's battery charging problem -/
theorem hawkeye_battery_budget
  (cost_per_charge : ℝ)
  (num_charges : ℕ)
  (money_left : ℝ)
  (h1 : cost_per_charge = 3.5)
  (h2 : num_charges = 4)
  (h3 : money_left = 6) :
  cost_per_charge * num_charges + money_left = 20 := by
  sorry

end NUMINAMATH_CALUDE_hawkeye_battery_budget_l1065_106501


namespace NUMINAMATH_CALUDE_multiple_of_112_implies_multiple_of_8_l1065_106551

theorem multiple_of_112_implies_multiple_of_8 (n : ℤ) : 
  (∃ k : ℤ, 14 * n = 112 * k) → (∃ m : ℤ, n = 8 * m) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_112_implies_multiple_of_8_l1065_106551


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1065_106592

theorem lcm_hcf_problem (a b : ℕ+) (h1 : b = 15) (h2 : Nat.lcm a b = 60) (h3 : Nat.gcd a b = 3) : a = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1065_106592


namespace NUMINAMATH_CALUDE_count_integer_pairs_l1065_106545

theorem count_integer_pairs : ∃ (count : ℕ),
  count = (Finset.filter (fun p : ℕ × ℕ => 
    let m := p.1
    let n := p.2
    1 ≤ m ∧ m ≤ 2012 ∧ 
    (5 : ℝ)^n < (2 : ℝ)^m ∧ 
    (2 : ℝ)^m < (2 : ℝ)^(m+2) ∧ 
    (2 : ℝ)^(m+2) < (5 : ℝ)^(n+1))
  (Finset.product (Finset.range 2013) (Finset.range (2014 + 1)))).card ∧
  (2 : ℝ)^2013 < (5 : ℝ)^867 ∧ (5 : ℝ)^867 < (2 : ℝ)^2014 ∧
  count = 279 := by
  sorry

end NUMINAMATH_CALUDE_count_integer_pairs_l1065_106545


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l1065_106572

theorem largest_n_divisible_by_seven (n : ℕ) : 
  n < 80000 → 
  (9 * (n - 3)^6 - 3 * n^3 + 21 * n - 33) % 7 = 0 → 
  n ≤ 79993 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l1065_106572


namespace NUMINAMATH_CALUDE_set_equality_l1065_106544

theorem set_equality : {x : ℤ | -3 < x ∧ x < 1} = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l1065_106544


namespace NUMINAMATH_CALUDE_final_value_is_15_l1065_106566

def loop_operation (x : ℕ) (s : ℕ) : ℕ := s * x + 1

def iterate_n_times (n : ℕ) (x : ℕ) (initial_s : ℕ) : ℕ :=
  match n with
  | 0 => initial_s
  | m + 1 => loop_operation x (iterate_n_times m x initial_s)

theorem final_value_is_15 :
  let x : ℕ := 2
  let initial_s : ℕ := 0
  let n : ℕ := 4
  iterate_n_times n x initial_s = 15 := by
  sorry

#eval iterate_n_times 4 2 0

end NUMINAMATH_CALUDE_final_value_is_15_l1065_106566


namespace NUMINAMATH_CALUDE_bass_strings_l1065_106539

theorem bass_strings (num_basses : ℕ) (num_guitars : ℕ) (num_8string_guitars : ℕ) 
  (guitar_strings : ℕ) (total_strings : ℕ) :
  num_basses = 3 →
  num_guitars = 2 * num_basses →
  guitar_strings = 6 →
  num_8string_guitars = num_guitars - 3 →
  total_strings = 72 →
  ∃ bass_strings : ℕ, 
    bass_strings * num_basses + guitar_strings * num_guitars + 8 * num_8string_guitars = total_strings ∧
    bass_strings = 4 :=
by sorry

end NUMINAMATH_CALUDE_bass_strings_l1065_106539


namespace NUMINAMATH_CALUDE_sugar_purchase_proof_l1065_106500

/-- The number of pounds of sugar bought by the housewife -/
def sugar_pounds : ℕ := 24

/-- The price per pound of sugar in cents -/
def price_per_pound : ℕ := 9

/-- The total cost of the sugar purchase in cents -/
def total_cost : ℕ := 216

/-- Proves that the number of pounds of sugar bought is correct given the conditions -/
theorem sugar_purchase_proof :
  (sugar_pounds * price_per_pound = total_cost) ∧
  (sugar_pounds + 3) * (price_per_pound - 1) = total_cost :=
by sorry

#check sugar_purchase_proof

end NUMINAMATH_CALUDE_sugar_purchase_proof_l1065_106500


namespace NUMINAMATH_CALUDE_logarithm_simplification_l1065_106529

theorem logarithm_simplification 
  (p q r s t z : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hz : z > 0) : 
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log (p * t / (s * z)) = Real.log (z / t) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l1065_106529


namespace NUMINAMATH_CALUDE_cheerful_team_tasks_l1065_106542

theorem cheerful_team_tasks (correct_points : ℕ) (incorrect_points : ℕ) (total_points : ℤ) (max_tasks : ℕ) :
  correct_points = 9 →
  incorrect_points = 5 →
  total_points = 57 →
  max_tasks = 15 →
  ∃ (x y : ℕ),
    x + y ≤ max_tasks ∧
    (x : ℤ) * correct_points - (y : ℤ) * incorrect_points = total_points ∧
    x = 8 :=
by sorry

end NUMINAMATH_CALUDE_cheerful_team_tasks_l1065_106542


namespace NUMINAMATH_CALUDE_spratilish_word_count_mod_1000_l1065_106567

/-- Represents a Spratilish letter -/
inductive SpratilishLetter
| M
| P
| Z
| O

/-- Checks if a SpratilishLetter is a consonant -/
def isConsonant (l : SpratilishLetter) : Bool :=
  match l with
  | SpratilishLetter.M => true
  | SpratilishLetter.P => true
  | _ => false

/-- Checks if a SpratilishLetter is a vowel -/
def isVowel (l : SpratilishLetter) : Bool :=
  match l with
  | SpratilishLetter.Z => true
  | SpratilishLetter.O => true
  | _ => false

/-- Represents a Spratilish word as a list of SpratilishLetters -/
def SpratilishWord := List SpratilishLetter

/-- Checks if a SpratilishWord is valid (at least three consonants between any two vowels) -/
def isValidSpratilishWord (w : SpratilishWord) : Bool :=
  sorry

/-- Counts the number of valid 9-letter Spratilish words -/
def countValidSpratilishWords : Nat :=
  sorry

/-- The main theorem: The number of valid 9-letter Spratilish words is congruent to 704 modulo 1000 -/
theorem spratilish_word_count_mod_1000 :
  countValidSpratilishWords % 1000 = 704 := by sorry

end NUMINAMATH_CALUDE_spratilish_word_count_mod_1000_l1065_106567


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1065_106568

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 1 + a 5 = 8 →                                       -- given condition
  a 3 = 4 :=                                            -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1065_106568


namespace NUMINAMATH_CALUDE_percentage_less_l1065_106530

theorem percentage_less (w e y z : ℝ) 
  (hw : w = 0.60 * e) 
  (hz : z = 0.54 * y) 
  (hzw : z = 1.5000000000000002 * w) : 
  e = 0.60 * y := by
sorry

end NUMINAMATH_CALUDE_percentage_less_l1065_106530


namespace NUMINAMATH_CALUDE_two_colorable_l1065_106521

-- Define a graph with 2000 vertices
def Graph := Fin 2000 → Set (Fin 2000)

-- Define a property that each vertex has at least one edge
def HasEdges (g : Graph) : Prop :=
  ∀ v : Fin 2000, ∃ u : Fin 2000, u ∈ g v

-- Define a coloring function
def Coloring := Fin 2000 → Bool

-- Define a valid coloring
def ValidColoring (g : Graph) (c : Coloring) : Prop :=
  ∀ v u : Fin 2000, u ∈ g v → c v ≠ c u

-- Theorem statement
theorem two_colorable (g : Graph) (h : HasEdges g) :
  ∃ c : Coloring, ValidColoring g c :=
sorry

end NUMINAMATH_CALUDE_two_colorable_l1065_106521


namespace NUMINAMATH_CALUDE_jenny_house_improvements_l1065_106533

/-- Represents the problem of calculating the maximum value of improvements Jenny can make to her house. -/
theorem jenny_house_improvements
  (tax_rate : ℝ)
  (initial_house_value : ℝ)
  (rail_project_increase : ℝ)
  (max_affordable_tax : ℝ)
  (h1 : tax_rate = 0.02)
  (h2 : initial_house_value = 400000)
  (h3 : rail_project_increase = 0.25)
  (h4 : max_affordable_tax = 15000) :
  let new_house_value := initial_house_value * (1 + rail_project_increase)
  let max_house_value := max_affordable_tax / tax_rate
  max_house_value - new_house_value = 250000 :=
by sorry

end NUMINAMATH_CALUDE_jenny_house_improvements_l1065_106533


namespace NUMINAMATH_CALUDE_people_disliking_both_tv_and_games_l1065_106526

def total_surveyed : ℕ := 1500
def tv_dislike_percentage : ℚ := 25 / 100
def both_dislike_percentage : ℚ := 15 / 100

theorem people_disliking_both_tv_and_games :
  ⌊(tv_dislike_percentage * total_surveyed : ℚ) * both_dislike_percentage⌋ = 56 := by
  sorry

end NUMINAMATH_CALUDE_people_disliking_both_tv_and_games_l1065_106526


namespace NUMINAMATH_CALUDE_no_rational_solution_l1065_106507

theorem no_rational_solution : ¬∃ (a b : ℚ), a ≠ 0 ∧ b ≠ 0 ∧ a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6) := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l1065_106507


namespace NUMINAMATH_CALUDE_r_2011_equals_2_l1065_106596

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

def r (n : ℕ) : ℕ := fib n % 3

theorem r_2011_equals_2 : r 2011 = 2 := by
  sorry

end NUMINAMATH_CALUDE_r_2011_equals_2_l1065_106596


namespace NUMINAMATH_CALUDE_cherry_price_level_6_l1065_106543

noncomputable def cherryPrice (a b x : ℝ) : ℝ := Real.exp (a * x + b)

theorem cherry_price_level_6 (a b : ℝ) :
  (cherryPrice a b 1 / cherryPrice a b 5 = 3) →
  (cherryPrice a b 3 = 60) →
  ∃ ε > 0, |cherryPrice a b 6 - 170| < ε := by
sorry

end NUMINAMATH_CALUDE_cherry_price_level_6_l1065_106543


namespace NUMINAMATH_CALUDE_tan_45_degrees_l1065_106546

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l1065_106546


namespace NUMINAMATH_CALUDE_existence_of_special_point_set_l1065_106538

/-- A closed region bounded by a regular polygon -/
structure RegularPolygonRegion where
  vertices : Set (ℝ × ℝ)
  is_regular : Bool
  is_closed : Bool

/-- A set of points in the plane -/
def PointSet := Set (ℝ × ℝ)

/-- Predicate to check if a set of points can be covered by a region -/
def IsCovered (S : PointSet) (C : RegularPolygonRegion) : Prop := sorry

/-- Predicate to check if any n points from a set can be covered by a region -/
def AnyNPointsCovered (S : PointSet) (C : RegularPolygonRegion) (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem existence_of_special_point_set (C : RegularPolygonRegion) (n : ℕ) :
  ∃ (S : PointSet), AnyNPointsCovered S C n ∧ ¬IsCovered S C := by sorry

end NUMINAMATH_CALUDE_existence_of_special_point_set_l1065_106538


namespace NUMINAMATH_CALUDE_expand_product_l1065_106595

theorem expand_product (x : ℝ) : 3 * (x - 2) * (x^2 + 6) = 3*x^3 - 6*x^2 + 18*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1065_106595


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l1065_106514

/-- Given vectors a, b, and c in ℝ², prove that if (a - c) is parallel to b, then k = 5. -/
theorem vector_parallel_condition (k : ℝ) : 
  let a : Fin 2 → ℝ := ![3, 1]
  let b : Fin 2 → ℝ := ![1, 3]
  let c : Fin 2 → ℝ := ![k, 7]
  (∃ (t : ℝ), (a - c) = t • b) → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l1065_106514


namespace NUMINAMATH_CALUDE_undefined_value_of_fraction_l1065_106564

theorem undefined_value_of_fraction (a : ℝ) : a^3 - 8 = 0 ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_undefined_value_of_fraction_l1065_106564


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1065_106574

theorem inequality_equivalence (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1065_106574


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l1065_106523

theorem base_10_to_base_7 : 
  ∃ (a b c : Nat), 
    234 = a * 7^2 + b * 7^1 + c * 7^0 ∧ 
    a < 7 ∧ b < 7 ∧ c < 7 ∧
    a = 4 ∧ b = 5 ∧ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l1065_106523


namespace NUMINAMATH_CALUDE_triangle_line_equations_l1065_106511

/-- Triangle with vertices A(4, 0), B(6, 7), and C(0, 3) -/
structure Triangle where
  A : ℝ × ℝ := (4, 0)
  B : ℝ × ℝ := (6, 7)
  C : ℝ × ℝ := (0, 3)

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The line passing through the midpoints of sides BC and AB -/
def midpointLine (t : Triangle) : LineEquation :=
  { a := 3, b := 4, c := -29 }

/-- The perpendicular bisector of side BC -/
def perpendicularBisector (t : Triangle) : LineEquation :=
  { a := 3, b := 2, c := -19 }

theorem triangle_line_equations (t : Triangle) :
  (midpointLine t = { a := 3, b := 4, c := -29 }) ∧
  (perpendicularBisector t = { a := 3, b := 2, c := -19 }) := by
  sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l1065_106511


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l1065_106541

theorem complex_modulus_equality (t : ℝ) :
  t > 0 → (Complex.abs (8 + 2 * t * Complex.I) = 14 ↔ t = Real.sqrt 33) := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l1065_106541


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1065_106532

/-- A cubic function with specific properties -/
def f (a c d : ℝ) (x : ℝ) : ℝ := a * x^3 + c * x + d

theorem cubic_function_properties (a c d : ℝ) (h_a : a ≠ 0) :
  (∀ x, f a c d x = -f a c d (-x)) →  -- f is odd
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a c d 1 ≤ f a c d x) →  -- f(1) is an extreme value
  (f a c d 1 = -2) →  -- f(1) = -2
  (∀ x, f a c d x = x^3 - 3*x) ∧  -- f(x) = x^3 - 3x
  (∀ x, f a c d x ≤ 2)  -- maximum value is 2
  := by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1065_106532


namespace NUMINAMATH_CALUDE_min_value_cubic_expression_l1065_106580

theorem min_value_cubic_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^3 + y^3 - 5*x*y ≥ -125/27 := by sorry

end NUMINAMATH_CALUDE_min_value_cubic_expression_l1065_106580
