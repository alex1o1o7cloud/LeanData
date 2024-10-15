import Mathlib

namespace NUMINAMATH_GPT_evaluate_expression_l2375_237535

def star (A B : ℚ) : ℚ := (A + B) / 3

theorem evaluate_expression : star (star 7 15) 10 = 52 / 9 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2375_237535


namespace NUMINAMATH_GPT_melissa_games_played_l2375_237509

-- Define the conditions mentioned:
def points_per_game := 12
def total_points := 36

-- State the proof problem:
theorem melissa_games_played : total_points / points_per_game = 3 :=
by sorry

end NUMINAMATH_GPT_melissa_games_played_l2375_237509


namespace NUMINAMATH_GPT_factor_poly_l2375_237502

theorem factor_poly (P Q : ℝ) (h1 : ∃ b c : ℝ, 
  (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + P*x^2 + Q)
  : P + Q = 50 :=
sorry

end NUMINAMATH_GPT_factor_poly_l2375_237502


namespace NUMINAMATH_GPT_solve_positive_integer_l2375_237556

theorem solve_positive_integer (n : ℕ) (h : ∀ m : ℕ, m > 0 → n^m ≥ m^n) : n = 3 :=
sorry

end NUMINAMATH_GPT_solve_positive_integer_l2375_237556


namespace NUMINAMATH_GPT_parabola_point_focus_distance_l2375_237525

/-- 
  Given a point P on the parabola y^2 = 4x, and the distance from P to the line x = -2
  is 5 units, prove that the distance from P to the focus of the parabola is 4 units.
-/
theorem parabola_point_focus_distance {P : ℝ × ℝ} 
  (hP : P.2^2 = 4 * P.1) 
  (h_dist : (P.1 + 2)^2 + P.2^2 = 25) : 
  dist P (1, 0) = 4 :=
sorry

end NUMINAMATH_GPT_parabola_point_focus_distance_l2375_237525


namespace NUMINAMATH_GPT_groom_age_proof_l2375_237544

theorem groom_age_proof (G B : ℕ) (h1 : B = G + 19) (h2 : G + B = 185) : G = 83 :=
by
  sorry

end NUMINAMATH_GPT_groom_age_proof_l2375_237544


namespace NUMINAMATH_GPT_rita_swimming_months_l2375_237530

theorem rita_swimming_months
    (total_required_hours : ℕ := 1500)
    (backstroke_hours : ℕ := 50)
    (breaststroke_hours : ℕ := 9)
    (butterfly_hours : ℕ := 121)
    (monthly_hours : ℕ := 220) :
    (total_required_hours - (backstroke_hours + breaststroke_hours + butterfly_hours)) / monthly_hours = 6 := 
by
    -- Proof is omitted
    sorry

end NUMINAMATH_GPT_rita_swimming_months_l2375_237530


namespace NUMINAMATH_GPT_percentage_relation_l2375_237545

theorem percentage_relation (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 12) : y = 60 := by
  sorry

end NUMINAMATH_GPT_percentage_relation_l2375_237545


namespace NUMINAMATH_GPT_value_S3_S2_S5_S3_l2375_237588

variable {a : ℕ → ℝ} {S : ℕ → ℝ}
variable {d : ℝ}
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (d_ne_zero : d ≠ 0)
variable (h_geom_seq : (a 1 + 2 * d) ^ 2 = (a 1) * (a 1 + 3 * d))
variable (S_def : ∀ n, S n = n * a 1 + d * (n * (n - 1)) / 2)

theorem value_S3_S2_S5_S3 : (S 3 - S 2) / (S 5 - S 3) = 2 := by
  sorry

end NUMINAMATH_GPT_value_S3_S2_S5_S3_l2375_237588


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_for_lt_l2375_237500

variable {a b : ℝ}

theorem necessary_but_not_sufficient_for_lt (h : a < b + 1) : a < b := 
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_for_lt_l2375_237500


namespace NUMINAMATH_GPT_geometric_seq_sum_l2375_237568

theorem geometric_seq_sum :
  ∀ (a : ℕ → ℤ) (q : ℤ), 
    (∀ n, a (n + 1) = a n * q) ∧ 
    (a 4 + a 7 = 2) ∧ 
    (a 5 * a 6 = -8) → 
    a 1 + a 10 = -7 := 
by sorry

end NUMINAMATH_GPT_geometric_seq_sum_l2375_237568


namespace NUMINAMATH_GPT_property_depreciation_rate_l2375_237529

noncomputable def initial_value : ℝ := 25599.08977777778
noncomputable def final_value : ℝ := 21093
noncomputable def annual_depreciation_rate : ℝ := 0.063

theorem property_depreciation_rate :
  final_value = initial_value * (1 - annual_depreciation_rate)^3 :=
sorry

end NUMINAMATH_GPT_property_depreciation_rate_l2375_237529


namespace NUMINAMATH_GPT_min_value_ineq_l2375_237579

theorem min_value_ineq (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 3 * y = 1) :
  (1 / x) + (1 / (3 * y)) ≥ 4 :=
  sorry

end NUMINAMATH_GPT_min_value_ineq_l2375_237579


namespace NUMINAMATH_GPT_probability_of_passing_test_l2375_237578

theorem probability_of_passing_test (p : ℝ) (h : p + p * (1 - p) + p * (1 - p)^2 = 0.784) : p = 0.4 :=
sorry

end NUMINAMATH_GPT_probability_of_passing_test_l2375_237578


namespace NUMINAMATH_GPT_maximum_sum_each_side_equals_22_l2375_237528

theorem maximum_sum_each_side_equals_22 (A B C D : ℕ) :
  (∀ i, 1 ≤ i ∧ i ≤ 10)
  → (∀ S, S = A ∨ S = B ∨ S = C ∨ S = D ∧ A + B + C + D = 33)
  → (A + B + C + D + 55) / 4 = 22 :=
by
  sorry

end NUMINAMATH_GPT_maximum_sum_each_side_equals_22_l2375_237528


namespace NUMINAMATH_GPT_cuboid_count_l2375_237561

def length_small (m : ℕ) : ℕ := 6
def width_small (m : ℕ) : ℕ := 4
def height_small (m : ℕ) : ℕ := 3

def length_large (m : ℕ): ℕ := 18
def width_large (m : ℕ) : ℕ := 15
def height_large (m : ℕ) : ℕ := 2

def volume (l : ℕ) (w : ℕ) (h : ℕ) : ℕ := l * w * h

def n_small_cuboids (v_large v_small : ℕ) : ℕ := v_large / v_small

theorem cuboid_count : 
  n_small_cuboids (volume (length_large 1) (width_large 1) (height_large 1)) (volume (length_small 1) (width_small 1) (height_small 1)) = 7 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_count_l2375_237561


namespace NUMINAMATH_GPT_complement_union_l2375_237506

def is_pos_int_less_than_9 (x : ℕ) : Prop := x > 0 ∧ x < 9

def U : Set ℕ := {x | is_pos_int_less_than_9 x}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_union :
  (U \ (M ∪ N)) = {2, 4, 8} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l2375_237506


namespace NUMINAMATH_GPT_initial_amount_100000_l2375_237567

noncomputable def compound_interest_amount (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def future_value (P CI : ℝ) : ℝ :=
  P + CI

theorem initial_amount_100000
  (CI : ℝ) (P : ℝ) (r : ℝ) (n t : ℕ) 
  (h1 : CI = 8243.216)
  (h2 : r = 0.04)
  (h3 : n = 2)
  (h4 : t = 2)
  (h5 : future_value P CI = compound_interest_amount P r n t) :
  P = 100000 :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_100000_l2375_237567


namespace NUMINAMATH_GPT_bacteria_growth_l2375_237501

theorem bacteria_growth (d : ℕ) (t : ℕ) (initial final : ℕ) 
  (h_doubling : d = 4) 
  (h_initial : initial = 500) 
  (h_final : final = 32000) 
  (h_ratio : final / initial = 2^6) :
  t = d * 6 → t = 24 :=
by
  sorry

end NUMINAMATH_GPT_bacteria_growth_l2375_237501


namespace NUMINAMATH_GPT_sequence_general_formula_l2375_237562

theorem sequence_general_formula (a : ℕ → ℚ) 
  (h1 : a 1 = 1 / 2) 
  (h_rec : ∀ n : ℕ, a (n + 2) = 3 * a (n + 1) / (a (n + 1) + 3)) 
  (n : ℕ) : 
  a (n + 1) = 3 / (n + 6) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l2375_237562


namespace NUMINAMATH_GPT_complement_union_equals_l2375_237507

def universal_set : Set ℤ := {-2, -1, 0, 1, 2, 3, 4, 5}
def A : Set ℤ := {-1, 0, 1, 2, 3}
def B : Set ℤ := {-2, 0, 2}

def C_I (I : Set ℤ) (s : Set ℤ) : Set ℤ := I \ s

theorem complement_union_equals :
  C_I universal_set (A ∪ B) = {4, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_equals_l2375_237507


namespace NUMINAMATH_GPT_trigonometric_identity_l2375_237547

theorem trigonometric_identity (m : ℝ) (h : m < 0) :
  2 * (3 / -5) + 4 / -5 = -2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2375_237547


namespace NUMINAMATH_GPT_flood_monitoring_technology_l2375_237581

def geographicInformationTechnologies : Type := String

def RemoteSensing : geographicInformationTechnologies := "Remote Sensing"
def GlobalPositioningSystem : geographicInformationTechnologies := "Global Positioning System"
def GeographicInformationSystem : geographicInformationTechnologies := "Geographic Information System"
def DigitalEarth : geographicInformationTechnologies := "Digital Earth"

def effectiveFloodMonitoring (tech1 tech2 : geographicInformationTechnologies) : Prop :=
  (tech1 = RemoteSensing ∧ tech2 = GeographicInformationSystem) ∨ 
  (tech1 = GeographicInformationSystem ∧ tech2 = RemoteSensing)

theorem flood_monitoring_technology :
  effectiveFloodMonitoring RemoteSensing GeographicInformationSystem :=
by
  sorry

end NUMINAMATH_GPT_flood_monitoring_technology_l2375_237581


namespace NUMINAMATH_GPT_confectioner_customers_l2375_237532

theorem confectioner_customers (x : ℕ) (h : 0 < x) :
  (49 * (392 / x - 6) = 392) → x = 28 :=
by
sorry

end NUMINAMATH_GPT_confectioner_customers_l2375_237532


namespace NUMINAMATH_GPT_union_of_A_and_B_l2375_237569

def A : Set ℝ := { x | 1 < x ∧ x < 3 }
def B : Set ℝ := { x | 2 < x ∧ x < 4 }

theorem union_of_A_and_B : A ∪ B = { x | 1 < x ∧ x < 4 } := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l2375_237569


namespace NUMINAMATH_GPT_min_value_y_minus_one_over_x_l2375_237570

variable {x y : ℝ}

-- Condition 1: x is the median of the dataset
def is_median (x : ℝ) : Prop := 3 ≤ x ∧ x ≤ 5

-- Condition 2: The average of the dataset is 1
def average_is_one (x y : ℝ) : Prop := 1 + 2 + x^2 - y = 4

-- The statement to be proved
theorem min_value_y_minus_one_over_x :
  ∀ (x y : ℝ), is_median x → average_is_one x y → y = x^2 - 1 → (y - 1/x) ≥ 23/3 :=
by 
  -- This is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_min_value_y_minus_one_over_x_l2375_237570


namespace NUMINAMATH_GPT_length_of_qr_l2375_237598

theorem length_of_qr (Q : ℝ) (PQ QR : ℝ) 
  (h1 : Real.sin Q = 0.6)
  (h2 : PQ = 15) :
  QR = 18.75 :=
by
  sorry

end NUMINAMATH_GPT_length_of_qr_l2375_237598


namespace NUMINAMATH_GPT_solve_inequality_l2375_237519

theorem solve_inequality (a : ℝ) : 
  {x : ℝ | x^2 - (a + 2) * x + 2 * a > 0} = 
  (if a > 2 then {x | x < 2 ∨ x > a}
   else if a = 2 then {x | x ≠ 2}
   else {x | x < a ∨ x > 2}) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l2375_237519


namespace NUMINAMATH_GPT_problem_arith_sequences_l2375_237593

theorem problem_arith_sequences (a b : ℕ → ℕ) 
  (ha : ∀ n, a (n + 1) = a n + d)
  (hb : ∀ n, b (n + 1) = b n + e)
  (h1 : a 1 = 25)
  (h2 : b 1 = 75)
  (h3 : a 2 + b 2 = 100) : 
  a 37 + b 37 = 100 := 
sorry

end NUMINAMATH_GPT_problem_arith_sequences_l2375_237593


namespace NUMINAMATH_GPT_daragh_sisters_count_l2375_237573

theorem daragh_sisters_count (initial_bears : ℕ) (favorite_bears : ℕ) (eden_initial_bears : ℕ) (eden_total_bears : ℕ) 
    (remaining_bears := initial_bears - favorite_bears)
    (eden_received_bears := eden_total_bears - eden_initial_bears)
    (bears_per_sister := eden_received_bears) :
    initial_bears = 20 → favorite_bears = 8 → eden_initial_bears = 10 → eden_total_bears = 14 → 
    remaining_bears / bears_per_sister = 3 := 
by
  sorry

end NUMINAMATH_GPT_daragh_sisters_count_l2375_237573


namespace NUMINAMATH_GPT_find_y_l2375_237565

theorem find_y : 
  let mean1 := (7 + 9 + 14 + 23) / 4
  let mean2 := (18 + y) / 2
  mean1 = mean2 → y = 8.5 :=
by
  let y := 8.5
  sorry

end NUMINAMATH_GPT_find_y_l2375_237565


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_six_l2375_237520

theorem smallest_four_digit_divisible_by_six : ∃ n, n ≥ 1000 ∧ n < 10000 ∧ n % 6 = 0 ∧ ∀ m, m ≥ 1000 ∧ m < n → ¬ (m % 6 = 0) :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_six_l2375_237520


namespace NUMINAMATH_GPT_part_one_part_two_range_l2375_237587

/-
Definitions based on conditions from the problem:
- Given vectors ax = (\cos x, \sin x), bx = (3, - sqrt(3))
- Domain for x is [0, π]
--
- Prove if a + b is parallel to b, then x = 5π / 6
- Definition of function f(x), and g(x) based on problem requirements.
- Prove the range of g(x) is [-3, sqrt(3)]
-/

/-
Part (1):
Given ax + bx = (cos x + 3, sin x - sqrt(3)) is parallel to bx =  (3, - sqrt(3));
Prove that x = 5π / 6 under x ∈ [0, π].
-/
noncomputable def vector_ax (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_bx : ℝ × ℝ := (3, - Real.sqrt 3)

theorem part_one (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) 
  (h_parallel : (vector_ax x).1 + vector_bx.1 = (vector_ax x).2 + vector_bx.2) :
  x = 5 * Real.pi / 6 :=
  sorry

/-
Part (2):
Let f(x) = 3 cos x - sqrt(3) sin x.
The function g(x) = -2 sqrt(3) sin(1/2 x - 2π/3) is defined by shifting f(x) right by π/3 and doubling the horizontal coordinate.
Prove the range of g(x) is [-3, sqrt(3)].
-/
noncomputable def f (x : ℝ) := 3 * Real.cos x - Real.sqrt 3 * Real.sin x
noncomputable def g (x : ℝ) := -2 * Real.sqrt 3 * Real.sin (0.5 * x - 2 * Real.pi / 3)

theorem part_two_range (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) : 
  -3 ≤ g x ∧ g x ≤ Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_part_one_part_two_range_l2375_237587


namespace NUMINAMATH_GPT_geometric_sequence_a3a5_l2375_237584

theorem geometric_sequence_a3a5 (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r) (h2 : a 4 = 5) : a 3 * a 5 = 25 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3a5_l2375_237584


namespace NUMINAMATH_GPT_min_trig_expression_l2375_237540

theorem min_trig_expression (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = Real.pi) : 
  ∃ (x : ℝ), (x = 16 - 8 * Real.sqrt 2) ∧ (∀ A B C, 0 < A → 0 < B → 0 < C → A + B + C = Real.pi → 
    (1 / (Real.sin A)^2 + 1 / (Real.sin B)^2 + 4 / (1 + Real.sin C)) ≥ x) := 
sorry

end NUMINAMATH_GPT_min_trig_expression_l2375_237540


namespace NUMINAMATH_GPT_percentage_conversion_l2375_237539

-- Define the condition
def decimal_fraction : ℝ := 0.05

-- Define the target percentage
def percentage : ℝ := 5

-- State the theorem
theorem percentage_conversion (df : ℝ) (p : ℝ) (h1 : df = 0.05) (h2 : p = 5) : df * 100 = p :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_percentage_conversion_l2375_237539


namespace NUMINAMATH_GPT_oliver_final_amount_l2375_237515

variable (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ)

def final_amount_after_transactions (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ) : ℕ :=
  initial_amount + saved_amount - (spent_on_frisbee + spent_on_puzzle) + gift

theorem oliver_final_amount :
  final_amount_after_transactions 9 5 4 3 8 = 15 :=
by
  -- We can fill in the exact calculations here to provide the proof.
  sorry

end NUMINAMATH_GPT_oliver_final_amount_l2375_237515


namespace NUMINAMATH_GPT_min_value_of_expression_l2375_237572

-- Define the conditions in the problem
def conditions (m n : ℝ) : Prop :=
  (2 * m + n = 2) ∧ (m > 0) ∧ (n > 0)

-- Define the problem statement
theorem min_value_of_expression (m n : ℝ) (h : conditions m n) : 
  (∀ m n, conditions m n → (1 / m + 2 / n) ≥ 4) :=
by 
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l2375_237572


namespace NUMINAMATH_GPT_train_length_55_meters_l2375_237599

noncomputable def V_f := 47 * 1000 / 3600 -- Speed of the faster train in m/s
noncomputable def V_s := 36 * 1000 / 3600 -- Speed of the slower train in m/s
noncomputable def t := 36 -- Time in seconds

theorem train_length_55_meters (L : ℝ) (Vf : ℝ := V_f) (Vs : ℝ := V_s) (time : ℝ := t) :
  (2 * L = (Vf - Vs) * time) → L = 55 :=
by
  sorry

end NUMINAMATH_GPT_train_length_55_meters_l2375_237599


namespace NUMINAMATH_GPT_x12_is_1_l2375_237591

noncomputable def compute_x12 (x : ℝ) (h : x + 1 / x = Real.sqrt 5) : ℝ :=
  x ^ 12

theorem x12_is_1 (x : ℝ) (h : x + 1 / x = Real.sqrt 5) : compute_x12 x h = 1 :=
  sorry

end NUMINAMATH_GPT_x12_is_1_l2375_237591


namespace NUMINAMATH_GPT_pavan_travel_time_l2375_237524

theorem pavan_travel_time (D : ℝ) (V1 V2 : ℝ) (distance : D = 300) (speed1 : V1 = 30) (speed2 : V2 = 25) : 
  ∃ t : ℝ, t = 11 := 
  by
    sorry

end NUMINAMATH_GPT_pavan_travel_time_l2375_237524


namespace NUMINAMATH_GPT_inequality_3var_l2375_237521

theorem inequality_3var (x y z : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y) (h₃ : 0 ≤ z) (h₄ : x * y + y * z + z * x = 1) : 
    1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 5 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_3var_l2375_237521


namespace NUMINAMATH_GPT_find_value_l2375_237543

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic : ∀ x : ℝ, f (x + Real.pi) = f x
axiom value_at_neg_pi_third : f (-Real.pi / 3) = 1 / 2

theorem find_value : f (2017 * Real.pi / 3) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_value_l2375_237543


namespace NUMINAMATH_GPT_min_ab_value_l2375_237575

theorem min_ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 / a + 20 / b = 4) : ab = 25 :=
sorry

end NUMINAMATH_GPT_min_ab_value_l2375_237575


namespace NUMINAMATH_GPT_bruno_coconuts_per_trip_is_8_l2375_237536

-- Definitions related to the problem conditions
def total_coconuts : ℕ := 144
def barbie_coconuts_per_trip : ℕ := 4
def trips : ℕ := 12
def bruno_coconuts_per_trip : ℕ := total_coconuts - (barbie_coconuts_per_trip * trips)

-- The main theorem stating the question and the answer
theorem bruno_coconuts_per_trip_is_8 : bruno_coconuts_per_trip / trips = 8 :=
by
  sorry

end NUMINAMATH_GPT_bruno_coconuts_per_trip_is_8_l2375_237536


namespace NUMINAMATH_GPT_total_nails_used_l2375_237523

-- Given definitions from the conditions
def square_side_length : ℕ := 36
def nails_per_side : ℕ := 40
def sides_of_square : ℕ := 4
def corners_of_square : ℕ := 4

-- Statement of the problem proof
theorem total_nails_used : nails_per_side * sides_of_square - corners_of_square = 156 := by
  sorry

end NUMINAMATH_GPT_total_nails_used_l2375_237523


namespace NUMINAMATH_GPT_find_x_l2375_237511

noncomputable def x : ℝ :=
  sorry

theorem find_x (h : ∃ x : ℝ, x > 0 ∧ ⌊x⌋ * x = 48) : x = 8 :=
  sorry

end NUMINAMATH_GPT_find_x_l2375_237511


namespace NUMINAMATH_GPT_dissection_impossible_l2375_237595

theorem dissection_impossible :
  ∀ (n m : ℕ), n = 1000 → m = 2016 → ¬(∃ (k l : ℕ), k * (n * m) = 1 * 2015 + l * 3) :=
by
  intros n m hn hm
  sorry

end NUMINAMATH_GPT_dissection_impossible_l2375_237595


namespace NUMINAMATH_GPT_Dvaneft_percentage_bounds_l2375_237541

noncomputable def percentageDvaneftShares (x y z : ℤ) (n m : ℕ) : ℚ :=
  n * 100 / (2 * (n + m))

theorem Dvaneft_percentage_bounds
  (x y z : ℤ) (n m : ℕ)
  (h1 : 4 * x * n = y * m)
  (h2 : x * n + y * m = z * (n + m))
  (h3 : 16 ≤ y - x)
  (h4 : y - x ≤ 20)
  (h5 : 42 ≤ z)
  (h6 : z ≤ 60) :
  12.5 ≤ percentageDvaneftShares x y z n m ∧ percentageDvaneftShares x y z n m ≤ 15 := by
  sorry

end NUMINAMATH_GPT_Dvaneft_percentage_bounds_l2375_237541


namespace NUMINAMATH_GPT_age_of_son_l2375_237560

theorem age_of_son (S M : ℕ) (h1 : M = S + 28) (h2 : M + 2 = 2 * (S + 2)) : S = 26 := by
  sorry

end NUMINAMATH_GPT_age_of_son_l2375_237560


namespace NUMINAMATH_GPT_range_of_x_l2375_237558

open Real

def p (x : ℝ) : Prop := log (x^2 - 2 * x - 2) ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4
def not_p (x : ℝ) : Prop := -1 < x ∧ x < 3
def not_q (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 4

theorem range_of_x (x : ℝ) :
  (¬ p x ∧ ¬ q x ∧ (p x ∨ q x)) →
  x ≤ -1 ∨ (0 < x ∧ x < 3) ∨ x ≥ 4 :=
sorry

end NUMINAMATH_GPT_range_of_x_l2375_237558


namespace NUMINAMATH_GPT_utilities_cost_l2375_237585

theorem utilities_cost
    (rent1 : ℝ) (utility1 : ℝ) (rent2 : ℝ) (utility2 : ℝ)
    (distance1 : ℝ) (distance2 : ℝ) 
    (cost_per_mile : ℝ) 
    (drive_days : ℝ) (cost_difference : ℝ)
    (h1 : rent1 = 800)
    (h2 : rent2 = 900)
    (h3 : utility2 = 200)
    (h4 : distance1 = 31)
    (h5 : distance2 = 21)
    (h6 : cost_per_mile = 0.58)
    (h7 : drive_days = 20)
    (h8 : cost_difference = 76)
    : utility1 = 259.60 := 
by
  sorry

end NUMINAMATH_GPT_utilities_cost_l2375_237585


namespace NUMINAMATH_GPT_completing_square_solution_l2375_237596

theorem completing_square_solution (x : ℝ) : x^2 - 4 * x - 22 = 0 ↔ (x - 2)^2 = 26 := sorry

end NUMINAMATH_GPT_completing_square_solution_l2375_237596


namespace NUMINAMATH_GPT_correct_histogram_height_representation_l2375_237551

   def isCorrectHeightRepresentation (heightRep : String) : Prop :=
     heightRep = "ratio of the frequency of individuals in that group within the sample to the class interval"

   theorem correct_histogram_height_representation :
     isCorrectHeightRepresentation "ratio of the frequency of individuals in that group within the sample to the class interval" :=
   by 
     sorry
   
end NUMINAMATH_GPT_correct_histogram_height_representation_l2375_237551


namespace NUMINAMATH_GPT_smallest_divisor_after_323_l2375_237566

-- Let n be an even 4-digit number such that 323 is a divisor of n.
def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisor (d n : ℕ) : Prop :=
  n % d = 0

theorem smallest_divisor_after_323 (n : ℕ) (h1 : is_even n) (h2 : is_4_digit n) (h3 : is_divisor 323 n) : ∃ k, k > 323 ∧ is_divisor k n ∧ k = 340 :=
by
  sorry

end NUMINAMATH_GPT_smallest_divisor_after_323_l2375_237566


namespace NUMINAMATH_GPT_acute_angle_is_three_pi_over_eight_l2375_237557

noncomputable def acute_angle_concentric_circles : Real :=
  let r₁ := 4
  let r₂ := 3
  let r₃ := 2
  let total_area := (r₁ * r₁ * Real.pi) + (r₂ * r₂ * Real.pi) + (r₃ * r₃ * Real.pi)
  let unshaded_area := 5 * (total_area / 8)
  let shaded_area := (3 / 5) * unshaded_area
  let theta := shaded_area / total_area * 2 * Real.pi
  theta

theorem acute_angle_is_three_pi_over_eight :
  acute_angle_concentric_circles = (3 * Real.pi / 8) :=
by
  sorry

end NUMINAMATH_GPT_acute_angle_is_three_pi_over_eight_l2375_237557


namespace NUMINAMATH_GPT_edward_original_amount_l2375_237597

-- Given conditions
def spent : ℝ := 16
def remaining : ℝ := 6

-- Question: How much did Edward have before he spent his money?
-- Correct answer: 22
theorem edward_original_amount : (spent + remaining) = 22 :=
by sorry

end NUMINAMATH_GPT_edward_original_amount_l2375_237597


namespace NUMINAMATH_GPT_terminal_side_angles_l2375_237505

theorem terminal_side_angles (k : ℤ) (β : ℝ) :
  β = (Real.pi / 3) + 2 * k * Real.pi → -2 * Real.pi ≤ β ∧ β < 4 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_terminal_side_angles_l2375_237505


namespace NUMINAMATH_GPT_train_speed_is_144_l2375_237594

-- Definitions for the conditions
def length_of_train_passing_pole (S : ℝ) := S * 8
def length_of_train_passing_stationary_train (S : ℝ) := S * 18 - 400

-- The main theorem to prove the speed of the train
theorem train_speed_is_144 (S : ℝ) :
  (length_of_train_passing_pole S = length_of_train_passing_stationary_train S) →
  (S * 3.6 = 144) :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_144_l2375_237594


namespace NUMINAMATH_GPT_determine_a_l2375_237553

theorem determine_a (a : ℕ) : 
  (2 * 10^10 + a ) % 11 = 0 ∧ 0 ≤ a ∧ a < 11 → a = 9 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l2375_237553


namespace NUMINAMATH_GPT_base4_sum_conversion_to_base10_l2375_237534

theorem base4_sum_conversion_to_base10 :
  let n1 := 2213
  let n2 := 2703
  let n3 := 1531
  let base := 4
  let sum_base4 := n1 + n2 + n3 
  let sum_base10 :=
    (1 * base^4) + (0 * base^3) + (2 * base^2) + (5 * base^1) + (1 * base^0)
  sum_base10 = 309 :=
by
  sorry

end NUMINAMATH_GPT_base4_sum_conversion_to_base10_l2375_237534


namespace NUMINAMATH_GPT_complex_number_solution_l2375_237527

theorem complex_number_solution (z : ℂ) (i : ℂ) (h_i : i^2 = -1) 
  (h : -i * z = (3 + 2 * i) * (1 - i)) : z = 1 + 5 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_number_solution_l2375_237527


namespace NUMINAMATH_GPT_value_at_17pi_over_6_l2375_237592

variable (f : Real → Real)

-- Defining the conditions
def period (f : Real → Real) (T : Real) := ∀ x, f (x + T) = f x
def specific_value (f : Real → Real) (x : Real) (v : Real) := f x = v

-- The main theorem statement
theorem value_at_17pi_over_6 : 
  period f (π / 2) →
  specific_value f (π / 3) 1 →
  specific_value f (17 * π / 6) 1 :=
by
  intros h_period h_value
  sorry

end NUMINAMATH_GPT_value_at_17pi_over_6_l2375_237592


namespace NUMINAMATH_GPT_exists_five_digit_number_with_property_l2375_237574

theorem exists_five_digit_number_with_property :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (n^2 % 100000) = n := 
sorry

end NUMINAMATH_GPT_exists_five_digit_number_with_property_l2375_237574


namespace NUMINAMATH_GPT_ratio_john_to_jenna_l2375_237559

theorem ratio_john_to_jenna (J : ℕ) 
  (h1 : 100 - J - 40 = 35) : 
  J = 25 ∧ (J / 100 = 1 / 4) := 
by
  sorry

end NUMINAMATH_GPT_ratio_john_to_jenna_l2375_237559


namespace NUMINAMATH_GPT_find_m_l2375_237576

-- Define the arithmetic sequence and its properties
variable {α : Type*} [OrderedRing α]
variable (a : Nat → α) (S : Nat → α) (m : ℕ)

-- The conditions from the problem
variable (is_arithmetic_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
variable (sum_of_terms : ∀ n, S n = (n * (a 0 + a (n - 1))) / 2)
variable (m_gt_one : m > 1)
variable (condition1 : a (m - 1) + a (m + 1) - a m ^ 2 - 1 = 0)
variable (condition2 : S (2 * m - 1) = 39)

-- Prove that m = 20
theorem find_m : m = 20 :=
sorry

end NUMINAMATH_GPT_find_m_l2375_237576


namespace NUMINAMATH_GPT_visible_sides_probability_l2375_237516

theorem visible_sides_probability
  (r : ℝ)
  (side_length : ℝ := 4)
  (probability : ℝ := 3 / 4) :
  r = 8 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_visible_sides_probability_l2375_237516


namespace NUMINAMATH_GPT_remainder_of_large_power_l2375_237510

def powerMod (base exp mod_ : ℕ) : ℕ := (base ^ exp) % mod_

theorem remainder_of_large_power :
  powerMod 2 (2^(2^2)) 500 = 536 :=
sorry

end NUMINAMATH_GPT_remainder_of_large_power_l2375_237510


namespace NUMINAMATH_GPT_expected_value_of_flipped_coins_l2375_237512

theorem expected_value_of_flipped_coins :
  let p := 1
  let n := 5
  let d := 10
  let q := 25
  let f := 50
  let prob := (1:ℝ) / 2
  let V := prob * p + prob * n + prob * d + prob * q + prob * f
  V = 45.5 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_flipped_coins_l2375_237512


namespace NUMINAMATH_GPT_power_of_a_point_l2375_237522

noncomputable def PA : ℝ := 4
noncomputable def PB : ℝ := 14 + 2 * Real.sqrt 13
noncomputable def PT : ℝ := PB - 8
noncomputable def AB : ℝ := PB - PA

theorem power_of_a_point (PA PB PT : ℝ) (h1 : PA = 4) (h2 : PB = 14 + 2 * Real.sqrt 13) (h3 : PT = PB - 8) : 
  PA * PB = PT * PT :=
by
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_power_of_a_point_l2375_237522


namespace NUMINAMATH_GPT_base_four_product_l2375_237552

def base_four_to_decimal (n : ℕ) : ℕ :=
  -- definition to convert base 4 to decimal, skipping details for now
  sorry

def decimal_to_base_four (n : ℕ) : ℕ :=
  -- definition to convert decimal to base 4, skipping details for now
  sorry

theorem base_four_product : 
  base_four_to_decimal 212 * base_four_to_decimal 13 = base_four_to_decimal 10322 :=
sorry

end NUMINAMATH_GPT_base_four_product_l2375_237552


namespace NUMINAMATH_GPT_inequalities_not_equivalent_l2375_237542

theorem inequalities_not_equivalent (x : ℝ) (h1 : x ≠ 1) :
  (x + 3 - (1 / (x - 1)) > -x + 2 - (1 / (x - 1))) ↔ (x + 3 > -x + 2) → False :=
by
  sorry

end NUMINAMATH_GPT_inequalities_not_equivalent_l2375_237542


namespace NUMINAMATH_GPT_find_speed_ratio_l2375_237563

noncomputable def circular_track_speed_ratio (v_V v_P C : ℝ) (H1 : v_V > 0) (H2 : v_P > 0) : Prop :=
  let t_1 := C / (v_V + v_P)
  let t_2 := (C * (2 * v_V + v_P)) / (v_V * (v_V + v_P))
  ∃ (r : ℝ), r = (1 + Real.sqrt 5) / 2 ∧ v_V / v_P = r

theorem find_speed_ratio
  (v_V v_P C : ℝ) (H1 : v_V > 0) (H2 : v_P > 0)
  (meeting1 : v_V * (C / (v_V + v_P)) + v_P * (C / (v_V + v_P)) = C)
  (lap_vasya : v_V * ((C * (2 * v_V + v_P)) / (v_V * (v_V + v_P))) = C + v_V * (C / (v_V + v_P)))
  (lap_petya : v_P * ((C * (2 * v_P + v_V)) / (v_P * (v_V + v_P))) = C + v_P * (C / (v_V + v_P))) :
  ∃ (r : ℝ), r = (1 + Real.sqrt 5) / 2 ∧ v_V / v_P = r :=
  sorry

end NUMINAMATH_GPT_find_speed_ratio_l2375_237563


namespace NUMINAMATH_GPT_isosceles_triangle_y_value_l2375_237546

theorem isosceles_triangle_y_value :
  ∃ y : ℝ, (y = 1 + Real.sqrt 51 ∨ y = 1 - Real.sqrt 51) ∧ 
  (Real.sqrt ((y - 1)^2 + (4 - (-3))^2) = 10) :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_y_value_l2375_237546


namespace NUMINAMATH_GPT_probability_all_quit_same_tribe_l2375_237508

-- Define the number of participants and the number of tribes
def numParticipants : ℕ := 18
def numTribes : ℕ := 2
def tribeSize : ℕ := 9 -- Each tribe has 9 members

-- Define the problem statement
theorem probability_all_quit_same_tribe : 
  (numParticipants.choose 3) = 816 ∧
  ((tribeSize.choose 3) * numTribes) = 168 ∧
  ((tribeSize.choose 3) * numTribes) / (numParticipants.choose 3) = 7 / 34 :=
by
  sorry

end NUMINAMATH_GPT_probability_all_quit_same_tribe_l2375_237508


namespace NUMINAMATH_GPT_total_liters_needed_to_fill_two_tanks_l2375_237548

theorem total_liters_needed_to_fill_two_tanks (capacity : ℕ) (liters_first_tank : ℕ) (liters_second_tank : ℕ) (percent_filled : ℕ) :
  liters_first_tank = 300 → 
  liters_second_tank = 450 → 
  percent_filled = 45 → 
  capacity = (liters_second_tank * 100) / percent_filled → 
  1000 - 300 = 700 → 
  1000 - 450 = 550 → 
  700 + 550 = 1250 :=
by sorry

end NUMINAMATH_GPT_total_liters_needed_to_fill_two_tanks_l2375_237548


namespace NUMINAMATH_GPT_swim_distance_l2375_237513

theorem swim_distance (v d : ℝ) (c : ℝ := 2.5) :
  (8 = d / (v + c)) ∧ (8 = 24 / (v - c)) → d = 84 :=
by
  sorry

end NUMINAMATH_GPT_swim_distance_l2375_237513


namespace NUMINAMATH_GPT_incorrect_multiplicative_inverse_product_l2375_237582

theorem incorrect_multiplicative_inverse_product:
  ∃ (a b : ℝ), a + b = 0 ∧ a * b ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_incorrect_multiplicative_inverse_product_l2375_237582


namespace NUMINAMATH_GPT_find_M_l2375_237538

theorem find_M : 
  let S := (981 + 983 + 985 + 987 + 989 + 991 + 993 + 995 + 997 + 999)
  let Target := 5100 - M
  S = Target → M = 4800 :=
by
  sorry

end NUMINAMATH_GPT_find_M_l2375_237538


namespace NUMINAMATH_GPT_zero_point_exists_between_2_and_3_l2375_237526

noncomputable def f (x : ℝ) := 2^(x-1) + x - 5

theorem zero_point_exists_between_2_and_3 :
  ∃ x₀ ∈ Set.Ioo (2 : ℝ) 3, f x₀ = 0 :=
sorry

end NUMINAMATH_GPT_zero_point_exists_between_2_and_3_l2375_237526


namespace NUMINAMATH_GPT_arccos_one_over_sqrt_two_l2375_237555

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_arccos_one_over_sqrt_two_l2375_237555


namespace NUMINAMATH_GPT_discount_percentage_l2375_237533

theorem discount_percentage (cp mp pm : ℤ) (x : ℤ) 
    (Hcp : cp = 160) 
    (Hmp : mp = 240) 
    (Hpm : pm = 20) 
    (Hcondition : mp * (100 - x) = cp * (100 + pm)) : 
  x = 20 := 
  sorry

end NUMINAMATH_GPT_discount_percentage_l2375_237533


namespace NUMINAMATH_GPT_depth_notation_l2375_237549

theorem depth_notation (x y : ℤ) (hx : x = 9050) (hy : y = -10907) : -y = x :=
by
  sorry

end NUMINAMATH_GPT_depth_notation_l2375_237549


namespace NUMINAMATH_GPT_square_pyramid_properties_l2375_237537

-- Definitions for the square pyramid with a square base
def square_pyramid_faces : Nat := 4 + 1
def square_pyramid_edges : Nat := 4 + 4
def square_pyramid_vertices : Nat := 4 + 1

-- Definition for the number of diagonals in a square
def diagonals_in_square_base (n : Nat) : Nat := n * (n - 3) / 2

-- Theorem statement
theorem square_pyramid_properties :
  (square_pyramid_faces + square_pyramid_edges + square_pyramid_vertices = 18) ∧ (diagonals_in_square_base 4 = 2) :=
by
  sorry

end NUMINAMATH_GPT_square_pyramid_properties_l2375_237537


namespace NUMINAMATH_GPT_find_m_l2375_237503

-- Definitions for the given vectors
def a : ℝ × ℝ := (3, 4)
def b (m : ℝ) : ℝ × ℝ := (-1, 2 * m)
def c (m : ℝ) : ℝ × ℝ := (m, -4)

-- Definition of vector addition
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- Definition of dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition that c is perpendicular to a + b
def perpendicular_condition (m : ℝ) : Prop :=
  dot_product (c m) (vector_add a (b m)) = 0

-- Proof statement
theorem find_m : ∃ m : ℝ, perpendicular_condition m ∧ m = -8 / 3 :=
sorry

end NUMINAMATH_GPT_find_m_l2375_237503


namespace NUMINAMATH_GPT_parabola_intersection_at_1_2003_l2375_237586

theorem parabola_intersection_at_1_2003 (p q : ℝ) (h : p + q = 2002) :
  (1, (1 : ℝ)^2 + p * 1 + q) = (1, 2003) :=
by
  sorry

end NUMINAMATH_GPT_parabola_intersection_at_1_2003_l2375_237586


namespace NUMINAMATH_GPT_exist_alpha_beta_l2375_237554

variables {a b : ℝ} {f : ℝ → ℝ}

-- Assume that f has the Intermediate Value Property (for simplicity, define it as a predicate)
def intermediate_value_property (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ k ∈ Set.Icc (min (f a) (f b)) (max (f a) (f b)),
    ∃ c ∈ Set.Ioo a b, f c = k

-- Assume the conditions from the problem
variables (h_ivp : intermediate_value_property f a b) (h_sign_change : f a * f b < 0)

-- The theorem we need to prove
theorem exist_alpha_beta (hivp : intermediate_value_property f a b) (hsign : f a * f b < 0) :
  ∃ α β, a < α ∧ α < β ∧ β < b ∧ f α + f β = f α * f β :=
sorry

end NUMINAMATH_GPT_exist_alpha_beta_l2375_237554


namespace NUMINAMATH_GPT_regular_pentagonal_pyramid_angle_l2375_237564

noncomputable def angle_between_slant_height_and_non_intersecting_edge (base_edge_slant_height : ℝ) : ℝ :=
  -- Assuming the base edge and slant height are given as input and equal
  if base_edge_slant_height > 0 then 36 else 0

theorem regular_pentagonal_pyramid_angle
  (base_edge_slant_height : ℝ)
  (h : base_edge_slant_height > 0) :
  angle_between_slant_height_and_non_intersecting_edge base_edge_slant_height = 36 :=
by
  -- omitted proof steps
  sorry

end NUMINAMATH_GPT_regular_pentagonal_pyramid_angle_l2375_237564


namespace NUMINAMATH_GPT_income_of_first_member_l2375_237504

-- Define the number of family members
def num_members : ℕ := 4

-- Define the average income per member
def avg_income : ℕ := 10000

-- Define the known incomes of the other three members
def income2 : ℕ := 15000
def income3 : ℕ := 6000
def income4 : ℕ := 11000

-- Define the total income of the family
def total_income : ℕ := avg_income * num_members

-- Define the total income of the other three members
def total_other_incomes : ℕ := income2 + income3 + income4

-- Define the income of the first member
def income1 : ℕ := total_income - total_other_incomes

-- The theorem to prove
theorem income_of_first_member : income1 = 8000 := by
  sorry

end NUMINAMATH_GPT_income_of_first_member_l2375_237504


namespace NUMINAMATH_GPT_check_error_difference_l2375_237518

-- Let us define x and y as two-digit natural numbers
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem check_error_difference
    (x y : ℕ)
    (hx : isTwoDigit x)
    (hy : isTwoDigit y)
    (hxy : x > y)
    (h_difference : (100 * y + x) - (100 * x + y) = 2187)
    : x - y = 22 :=
by
  sorry

end NUMINAMATH_GPT_check_error_difference_l2375_237518


namespace NUMINAMATH_GPT_quarters_difference_nickels_eq_l2375_237550

variable (q : ℕ)

def charles_quarters := 7 * q + 2
def richard_quarters := 3 * q + 7
def quarters_difference := charles_quarters q - richard_quarters q
def money_difference_in_nickels := 5 * quarters_difference q

theorem quarters_difference_nickels_eq :
  money_difference_in_nickels q = 20 * (q - 5/4) :=
by
  sorry

end NUMINAMATH_GPT_quarters_difference_nickels_eq_l2375_237550


namespace NUMINAMATH_GPT_promotional_pricing_plan_l2375_237514

theorem promotional_pricing_plan (n : ℕ) : 
  (8 * 100 = 800) ∧ 
  (∀ n > 100, 6 * n < 640) :=
by
  sorry

end NUMINAMATH_GPT_promotional_pricing_plan_l2375_237514


namespace NUMINAMATH_GPT_range_of_n_l2375_237571

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (n : ℝ) : Set ℝ := {x | n-1 < x ∧ x < n+1}

-- Define the condition A ∩ B ≠ ∅
def A_inter_B_nonempty (n : ℝ) : Prop := ∃ x, x ∈ A ∧ x ∈ B n

-- Prove the range of n for which A ∩ B ≠ ∅ is (-2, 2)
theorem range_of_n : ∀ n, A_inter_B_nonempty n ↔ (-2 < n ∧ n < 2) := by
  sorry

end NUMINAMATH_GPT_range_of_n_l2375_237571


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l2375_237517

variable {a b : ℝ} (n : ℕ)

theorem relationship_between_a_and_b (h₁ : a^n = a + 1) (h₂ : b^(2 * n) = b + 3 * a)
  (h₃ : 2 ≤ n) (h₄ : 1 < a) (h₅ : 1 < b) : a > b ∧ b > 1 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l2375_237517


namespace NUMINAMATH_GPT_inequality_xy_gt_xz_l2375_237589

theorem inequality_xy_gt_xz (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 1) : 
  x * y > x * z := 
by
  sorry  -- Proof is not required as per the instructions

end NUMINAMATH_GPT_inequality_xy_gt_xz_l2375_237589


namespace NUMINAMATH_GPT_find_k_value_l2375_237580

theorem find_k_value (k : ℝ) :
  (∃ (x y : ℝ), x + k * y = 0 ∧ 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0) ↔ k = -1/2 := 
by
  sorry

end NUMINAMATH_GPT_find_k_value_l2375_237580


namespace NUMINAMATH_GPT_days_to_shovel_l2375_237583

-- Defining conditions as formal statements
def original_task_time := 10
def original_task_people := 10
def original_task_weight := 10000
def new_task_weight := 40000
def new_task_people := 5

-- Definition of rate in terms of weight, people and time
def rate_per_person (total_weight : ℕ) (total_people : ℕ) (total_time : ℕ) : ℕ :=
  total_weight / total_people / total_time

-- Theorem statement to prove
theorem days_to_shovel (t : ℕ) :
  (rate_per_person original_task_weight original_task_people original_task_time) * new_task_people * t = new_task_weight := sorry

end NUMINAMATH_GPT_days_to_shovel_l2375_237583


namespace NUMINAMATH_GPT_problem1_problem2_l2375_237577

variable {a b : ℝ}

theorem problem1
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a^3 + b^3 = 2)
  : (a + b) * (a^5 + b^5) ≥ 4 := sorry

theorem problem2
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a^3 + b^3 = 2)
  : a + b ≤ 2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l2375_237577


namespace NUMINAMATH_GPT_nonneg_reals_sum_to_one_implies_ineq_l2375_237590

theorem nonneg_reals_sum_to_one_implies_ineq
  (x y z : ℝ)
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 :=
sorry

end NUMINAMATH_GPT_nonneg_reals_sum_to_one_implies_ineq_l2375_237590


namespace NUMINAMATH_GPT_sine_of_angle_from_point_l2375_237531

theorem sine_of_angle_from_point (x y : ℤ) (r : ℝ) (h : r = Real.sqrt ((x : ℝ)^2 + (y : ℝ)^2)) (hx : x = -12) (hy : y = 5) :
  Real.sin (Real.arctan (y / x)) = y / r := 
by
  sorry

end NUMINAMATH_GPT_sine_of_angle_from_point_l2375_237531
